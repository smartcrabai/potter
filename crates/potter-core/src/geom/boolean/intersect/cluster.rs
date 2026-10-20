//! Coplanar clusters of triangles (Blender `CoplanarCluster`, `CoplanarClusterInfo`, and
//! `find_clusters`).
//!
//! A cluster is a set of coplanar triangles that intersect one another, chained by overlapping
//! boxes. Planes are compared in canonical form, so coplanar triangles that face opposite ways
//! share a plane group. Cluster ids follow the slot order of Blender's `plane_cls` map, and the
//! coplanar candidates follow the slot order of its `itt_map` (see `blender_map`).

use num_traits::{One, Signed, Zero};

use super::super::blender_map::{BlenderMap, pair_hash};
use super::super::exact::{Rat, Rat3, rat_from_f64, rat_to_f64};
use super::super::imesh::{Arena, Tri};
use super::tri_tri::IttKind;
use super::{Aabb, IttMap, tri_plane};

/// A set of coplanar triangles (Blender `CoplanarCluster`). The first triangle fixes the cluster's
/// plane, and the union box covers all of them.
#[derive(Clone, Debug)]
pub(super) struct CoplanarCluster {
    first: usize,
    tris: Vec<usize>,
    bb: Aabb,
}

impl CoplanarCluster {
    fn new(t: usize, bb: &Aabb) -> Self {
        Self {
            first: t,
            tris: vec![t],
            bb: *bb,
        }
    }

    fn add_tri(&mut self, t: usize, bb: &Aabb) {
        self.tris.push(t);
        self.bb.union(bb);
    }

    pub(super) fn first(&self) -> usize {
        self.first
    }

    pub(super) fn tris(&self) -> &[usize] {
        &self.tris
    }

    fn bounding_box(&self) -> &Aabb {
        &self.bb
    }
}

/// Indexed set of clusters, with the cluster of each triangle (Blender `CoplanarClusterInfo`).
pub(super) struct CoplanarClusterInfo {
    clusters: Vec<CoplanarCluster>,
    tri_cluster: Vec<Option<usize>>,
}

impl CoplanarClusterInfo {
    fn new(numtri: usize) -> Self {
        Self {
            clusters: Vec::new(),
            tri_cluster: vec![None; numtri],
        }
    }

    /// The cluster containing triangle `t`, or `None` when it is in no cluster.
    pub(super) fn tri_cluster(&self, t: usize) -> Option<usize> {
        self.tri_cluster.get(t).copied().flatten()
    }

    fn add_cluster(&mut self, cl: CoplanarCluster) {
        let c = self.clusters.len();
        for &t in &cl.tris {
            if let Some(slot) = self.tri_cluster.get_mut(t) {
                *slot = Some(c);
            }
        }
        self.clusters.push(cl);
    }

    pub(super) fn tot_cluster(&self) -> usize {
        self.clusters.len()
    }

    pub(super) fn cluster(&self, c: usize) -> &CoplanarCluster {
        &self.clusters[c]
    }
}

/// Blender `Plane::make_canonical`: scales `(normal, d)` so that the first nonzero normal component
/// is 1, or so that the normal is a unit axis vector. Equal planes then compare equal.
fn canonical_plane(normal: &Rat3, d: &Rat) -> (Rat3, Rat) {
    let n = normal;
    if !n[0].is_zero() {
        let den = &n[0];
        ([Rat::one(), &n[1] / den, &n[2] / den], d / den)
    } else if !n[1].is_zero() {
        let den = &n[1];
        ([Rat::zero(), Rat::one(), &n[2] / den], d / den)
    } else if !n[2].is_zero() {
        let den = &n[2];
        ([Rat::zero(), Rat::zero(), Rat::one()], d / den)
    } else {
        ([Rat::zero(), Rat::zero(), Rat::zero()], Rat::zero())
    }
}

/// Blender `mpq_class::get_d`: the double nearest to `value` in magnitude toward zero, so an
/// exact rational that is not representable is truncated, not rounded. Zero is `+0.0`.
fn rat_get_d(value: &Rat) -> f64 {
    let nearest = rat_to_f64(value);
    if !nearest.is_finite() || nearest == 0.0 {
        return 0.0;
    }
    if rat_from_f64(nearest).abs() > value.abs() {
        // Rounded away from zero: step one ulp toward zero (same sign, smaller magnitude).
        f64::from_bits(nearest.to_bits() - 1)
    } else {
        nearest
    }
}

/// Blender `Plane::hash()` (`mesh_intersect.cc`): mixes the bit patterns of the canonical normal
/// and offset, which are the doubles of `Plane::make_canonical`.
fn plane_hash(key: &(Rat3, Rat)) -> u64 {
    let (norm, d) = key;
    let mut x = rat_get_d(&norm[0]).to_bits();
    let mut y = rat_get_d(&norm[1]).to_bits();
    let mut z = rat_get_d(&norm[2]).to_bits();
    let mut dd = rat_get_d(d).to_bits();
    x = (x >> 56) ^ (x >> 46) ^ x;
    y = (y >> 55) ^ (y >> 45) ^ y;
    z = (z >> 54) ^ (z >> 44) ^ z;
    dd = (dd >> 53) ^ (dd >> 43) ^ dd;
    x ^ y ^ z ^ dd
}

/// Blender `find_clusters`: groups the triangles that take part in a coplanar intersection into
/// clusters, keeping those with more than one triangle.
///
/// `overlap_count` is the size of Blender's overlap list (`tri_ov.overlap().size()`), which sizes
/// `itt_map` before its inserts. Cluster ids follow the slot order of Blender's maps, so the
/// output order matches Blender exactly.
pub(super) fn find_clusters(
    arena: &Arena,
    tris: &[Tri],
    bbs: &[Aabb],
    itts: &IttMap,
    overlap_count: usize,
) -> CoplanarClusterInfo {
    let mut ans = CoplanarClusterInfo::new(tris.len());

    // Blender `itt_map`: keys are inserted in overlap order, and the coplanar pairs are read back
    // in slot order.
    let mut itt_slots: BlenderMap<(usize, usize), ()> = BlenderMap::new();
    itt_slots.reserve(overlap_count);
    for &((t1, t2), _) in &itts.entries {
        itt_slots.add_new((t1, t2), pair_hash(t1, t2), ());
    }

    // Candidate triangles of the coplanar pairs, in first-seen order (Blender `VectorSet`).
    let mut maybe: Vec<usize> = Vec::new();
    let mut seen = vec![false; tris.len()];
    for ((t1, t2), ()) in itt_slots.into_items() {
        let coplanar = itts
            .get((t1, t2))
            .is_some_and(|entry| matches!(entry.itt.kind, IttKind::Coplanar));
        if !coplanar {
            continue;
        }
        for t in [t1, t2] {
            if let Some(s) = seen.get_mut(t)
                && !*s
            {
                *s = true;
                maybe.push(t);
            }
        }
    }
    if maybe.is_empty() {
        return ans;
    }

    // Blender `plane_cls`: one plane group per canonical plane, in a map sized to the candidates.
    // Each group holds the clusters lying in one plane, in creation order.
    let mut plane_cls: BlenderMap<(Rat3, Rat), Vec<CoplanarCluster>> = BlenderMap::new();
    plane_cls.reserve(maybe.len());
    for &t in &maybe {
        let (normal, d) = tri_plane(arena, &tris[t]);
        let key = canonical_plane(&normal, &d);
        let hash = plane_hash(&key);

        let t_bb = &bbs[t];
        if let Some(curcls) = plane_cls.get_mut(&key, hash) {
            let int_idx: Vec<usize> = curcls
                .iter()
                .enumerate()
                .filter(|(_, cl)| cl.bounding_box().overlaps(t_bb))
                .map(|(i, _)| i)
                .collect();
            match int_idx.as_slice() {
                // `t` meets no existing cluster in its plane: it starts one of its own.
                [] => curcls.push(CoplanarCluster::new(t, t_bb)),
                [i] => curcls[*i].add_tri(t, t_bb),
                // `t` bridges several clusters: merge them, keeping the merged cluster first.
                _ => {
                    let mut merged = CoplanarCluster::new(t, t_bb);
                    for &i in &int_idx {
                        for &u in curcls[i].tris() {
                            merged.add_tri(u, &bbs[u]);
                        }
                    }
                    let old = std::mem::take(curcls);
                    let mut rebuilt = vec![merged];
                    rebuilt.extend(
                        old.into_iter()
                            .enumerate()
                            .filter(|(i, _)| !int_idx.contains(i))
                            .map(|(_, cl)| cl),
                    );
                    *curcls = rebuilt;
                }
            }
        } else {
            plane_cls.add_new(key, hash, vec![CoplanarCluster::new(t, t_bb)]);
        }
    }

    // Blender numbers clusters by `plane_cls.items()`.
    for (_, group) in plane_cls.into_items() {
        for cl in group {
            if cl.tris.len() > 1 {
                ans.add_cluster(cl);
            }
        }
    }
    ans
}
