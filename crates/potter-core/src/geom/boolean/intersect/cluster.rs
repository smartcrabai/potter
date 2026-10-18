//! Coplanar clusters of triangles (Blender `CoplanarCluster`, `CoplanarClusterInfo`, and
//! `find_clusters`).
//!
//! A cluster is a set of coplanar triangles that intersect one another, chained by overlapping
//! boxes. Planes are compared in canonical form, so coplanar triangles that face opposite ways
//! share a plane group. Plane groups and their clusters follow first-seen order, where Blender
//! iterates a hash map.

use std::collections::HashMap;

use num_traits::{One, Zero};

use super::super::exact::{Rat, Rat3};
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

/// Blender `find_clusters`: groups the triangles that take part in a coplanar intersection into
/// clusters, keeping those with more than one triangle.
pub(super) fn find_clusters(
    arena: &Arena,
    tris: &[Tri],
    bbs: &[Aabb],
    itts: &IttMap,
) -> CoplanarClusterInfo {
    let mut ans = CoplanarClusterInfo::new(tris.len());

    // Candidate triangles of the coplanar pairs, in first-seen order (Blender `VectorSet`).
    let mut maybe: Vec<usize> = Vec::new();
    let mut seen = vec![false; tris.len()];
    for ((t1, t2), entry) in &itts.entries {
        if !matches!(entry.itt.kind, IttKind::Coplanar) {
            continue;
        }
        for t in [*t1, *t2] {
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

    // Each plane group holds the clusters lying in one canonical plane, in creation order.
    let mut group_of: HashMap<(Rat3, Rat), usize> = HashMap::new();
    let mut groups: Vec<Vec<CoplanarCluster>> = Vec::new();
    for &t in &maybe {
        let (normal, d) = tri_plane(arena, &tris[t]);
        let key = canonical_plane(&normal, &d);
        let g = if let Some(&g) = group_of.get(&key) {
            g
        } else {
            let g = groups.len();
            group_of.insert(key, g);
            groups.push(Vec::new());
            g
        };

        let t_bb = &bbs[t];
        let curcls = &mut groups[g];
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
    }

    for group in groups {
        for cl in group {
            if cl.tris.len() > 1 {
                ans.add_cluster(cl);
            }
        }
    }
    ans
}
