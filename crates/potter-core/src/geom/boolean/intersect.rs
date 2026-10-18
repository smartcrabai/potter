//! Triangle-mesh intersection for the Mesh Arrangements port: Blender `trimesh_nary_intersect`
//! (`blenlib/intern/mesh_intersect.cc`).
//!
//! Each input triangle is intersected with the triangles whose boxes overlap it. Coplanar
//! triangles that meet form clusters, and each cluster is subdivided with one constrained Delaunay
//! triangulation of its plane. Other intersecting triangles are subdivided with a triangulation of
//! their own plane. Output triangles keep the `orig` of the input triangle they came from.
//!
//! Deviations from Blender:
//! - Single-threaded. Blender's parallel loops compute independent values; the serial extraction
//!   order, and therefore the arena vertex order, is kept.
//! - Overlap candidates come from a sort-and-sweep over `f64` boxes padded by Blender's rule
//!   (`calc_face_bounding_boxes`), not from a float BVH. Both find every pair that intersects;
//!   they can only differ on pairs whose boxes touch within float rounding, and those produce no
//!   intersection value.
//! - Triangle planes are recomputed exactly from arena vertices. Blender shares `Plane` objects
//!   holding the same values.
//! - Cluster ids and plane groups follow first-seen order. Blender iterates a hash map there.
//! - Blender's `len == 1 && indexB == t` check is omitted: the overlap search never reports a
//!   triangle against itself.
//! - The CDT must encode origins as Blender's `delaunay_2d_calc` does: input edge `i` as `i`,
//!   edge `p` of input face `f` as `(f + 1) * face_edge_offset + p` with
//!   `face_edge_offset = max(longest face, edge count)`, and face origin `f` as `f`.

mod cluster;
mod tri_tri;

use std::collections::HashMap;

use glam::DVec3;
use num_traits::Signed;

use super::cdt::{CdtInput, CdtOutput, delaunay_2d_calc_inside};
use super::exact::{Rat, Rat2, Rat3, cross3, dot3, sgn, sub3, supremum_dot_cross};
use super::imesh::{Arena, NO_INDEX, Tri};
use cluster::{CoplanarClusterInfo, find_clusters};
use tri_tri::{Itt, IttKind, intersect_tri_tri};

/// Blender `index_dot_cross`: error-analysis index of the dot product of two cross products.
const INDEX_DOT_CROSS: f64 = 11.0;

fn to_u32(i: usize) -> u32 {
    u32::try_from(i).unwrap_or(u32::MAX)
}

fn to_usize(i: u32) -> usize {
    usize::try_from(i).unwrap_or(usize::MAX)
}

/// Blender `sorted_int_pair`: the pair with the smaller element first.
fn sorted_pair<T: Ord>(a: T, b: T) -> (T, T) {
    if a <= b { (a, b) } else { (b, a) }
}

/// Axis-aligned box in double precision (Blender `BoundingBox`).
#[derive(Clone, Copy, Debug)]
pub(super) struct Aabb {
    min: DVec3,
    max: DVec3,
}

impl Aabb {
    fn empty() -> Self {
        Self {
            min: DVec3::splat(f64::INFINITY),
            max: DVec3::splat(f64::NEG_INFINITY),
        }
    }

    fn include(&mut self, p: DVec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    pub(super) fn union(&mut self, other: &Self) {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }

    fn expand(&mut self, pad: f64) {
        self.min -= DVec3::splat(pad);
        self.max += DVec3::splat(pad);
    }

    /// Inclusive overlap test (Blender `isect_aabb_aabb_v3`).
    pub(super) fn overlaps(&self, other: &Self) -> bool {
        self.min.cmple(other.max).all() && other.min.cmple(self.max).all()
    }
}

/// Blender `calc_face_bounding_boxes`: boxes of the triangles' double positions, padded by a
/// multiple of the largest coordinate so that float rounding cannot separate touching triangles.
fn calc_face_bounding_boxes(arena: &Arena, tris: &[Tri]) -> Vec<Aabb> {
    let mut max_abs = 0.0_f64;
    let mut boxes: Vec<Aabb> = tris
        .iter()
        .map(|tri| {
            let mut bb = Aabb::empty();
            for &v in &tri.v {
                let co = arena.vert(v).co_f;
                bb.include(co);
                max_abs = max_abs.max(co.abs().max_element());
            }
            bb
        })
        .collect();
    let eps = f64::from(f32::EPSILON);
    let pad = 10.0
        * if max_abs > 0.0 {
            2.0 * eps * max_abs
        } else {
            eps
        };
    for bb in &mut boxes {
        bb.expand(pad);
    }
    boxes
}

/// Exact plane of a triangle (Blender `Face::populate_plane(true)`): normal `cross(v0 - v2, v1 - v2)`
/// and `d = -dot(normal, v0)`, so that `dot(normal, p) + d = 0` on the plane.
pub(super) fn tri_plane(arena: &Arena, tri: &Tri) -> (Rat3, Rat) {
    let [i0, i1, i2] = tri.v;
    let (p0, p1, p2) = (&arena.vert(i0).co, &arena.vert(i1).co, &arena.vert(i2).co);
    let normal = cross3(&sub3(p0, p2), &sub3(p1, p2));
    let d = -dot3(&normal, p0);
    (normal, d)
}

/// Blender `face_is_degenerate`: repeated vertices or zero area. The double filter decides when
/// the area is clearly nonzero; otherwise the exact cross product decides.
fn is_degenerate(arena: &Arena, tri: &Tri) -> bool {
    let [v0, v1, v2] = tri.v;
    if v0 == v1 || v0 == v2 || v1 == v2 {
        return true;
    }
    let (p0, p1, p2) = (arena.vert(v0), arena.vert(v1), arena.vert(v2));
    let dab = (p2.co_f - p0.co_f).cross(p2.co_f - p1.co_f);
    let err_bound = supremum_dot_cross(dab, dab) * INDEX_DOT_CROSS * f64::EPSILON;
    if dab.length_squared() > err_bound {
        return false;
    }
    let ab = cross3(&sub3(&p2.co, &p0.co), &sub3(&p2.co, &p1.co));
    ab.iter().all(num_traits::Zero::is_zero)
}

/// Blender `TriOverlaps`: every ordered overlapping pair `(a, b)`, sorted by `(a, b)`, so the
/// partners of each triangle form one run.
struct TriOverlaps {
    pairs: Vec<(usize, usize)>,
    /// Range of `pairs` whose first element is the triangle (Blender `first_overlap_`).
    runs: Vec<std::ops::Range<usize>>,
}

impl TriOverlaps {
    /// With two shapes and no self-intersection, only pairs from different shapes are kept.
    /// Otherwise pairs whose `orig` differ are kept, and with one shape every pair is kept.
    fn new(
        tris: &[Tri],
        bbs: &[Aabb],
        nshapes: usize,
        shape_fn: &dyn Fn(i32) -> usize,
        use_self: bool,
    ) -> Self {
        let two_trees_no_self = nshapes == 2 && !use_self;
        let shapes: Vec<usize> = tris.iter().map(|t| shape_fn(t.orig)).collect();
        let participates = |t: usize| !two_trees_no_self || shapes[t] < 2;
        let accepts = |a: usize, b: usize| {
            if two_trees_no_self {
                shapes[a] != shapes[b]
            } else if nshapes == 1 {
                true
            } else {
                tris[a].orig != tris[b].orig
            }
        };

        let mut order: Vec<usize> = (0..tris.len()).filter(|&t| participates(t)).collect();
        order.sort_by(|&a, &b| bbs[a].min.x.total_cmp(&bbs[b].min.x));
        let mut pairs = Vec::new();
        for (i, &a) in order.iter().enumerate() {
            for &b in &order[i + 1..] {
                // Sorted by min x: no later triangle can reach `a`.
                if bbs[b].min.x > bbs[a].max.x {
                    break;
                }
                if bbs[a].overlaps(&bbs[b]) && accepts(a, b) {
                    pairs.push((a, b));
                    pairs.push((b, a));
                }
            }
        }
        pairs.sort_unstable();

        let mut runs = vec![0..0; tris.len()];
        let mut i = 0;
        while i < pairs.len() {
            let a = pairs[i].0;
            let start = i;
            while i < pairs.len() && pairs[i].0 == a {
                i += 1;
            }
            runs[a] = start..i;
        }
        Self { pairs, runs }
    }

    /// Overlapping pairs whose first element is `t`.
    fn partners(&self, t: usize) -> &[(usize, usize)] {
        &self.pairs[self.runs[t].clone()]
    }
}

/// Intersection value of a triangle pair (Blender `ITT_value`). `t_source` is the larger index of
/// the pair, which Blender records as the source of a coplanar value.
pub(super) struct IttEntry {
    itt: Itt,
    t_source: usize,
}

/// Map from canonical triangle pairs to intersection values, in first-seen order
/// (Blender `Map<pair<int, int>, ITT_value>`).
#[derive(Default)]
pub(super) struct IttMap {
    entries: Vec<((usize, usize), IttEntry)>,
    index: HashMap<(usize, usize), usize>,
}

impl IttMap {
    fn insert(&mut self, key: (usize, usize), entry: IttEntry) {
        self.index.insert(key, self.entries.len());
        self.entries.push((key, entry));
    }

    fn get(&self, key: (usize, usize)) -> Option<&IttEntry> {
        self.index.get(&key).map(|&i| &self.entries[i].1)
    }
}

/// Blender `calc_overlap_itts`: the intersection value of each overlapping pair, computed once for
/// `a < b`.
fn calc_overlap_itts(arena: &Arena, tris: &[Tri], ov: &TriOverlaps) -> IttMap {
    let mut itts = IttMap::default();
    for &(a, b) in &ov.pairs {
        if a < b {
            let itt = intersect_tri_tri(arena, &tris[a], &tris[b]);
            itts.insert((a, b), IttEntry { itt, t_source: b });
        }
    }
    itts
}

/// Blender `dominant_axis`: index of the largest absolute component, ties going to the later axis.
fn dominant_axis(n: &Rat3) -> usize {
    let (x, y, z) = (n[0].abs(), n[1].abs(), n[2].abs());
    if x > y {
        if x > z { 0 } else { 2 }
    } else if y > z {
        1
    } else {
        2
    }
}

/// Blender `project_3d_to_2d`: drops coordinate `axis`.
fn project(p: &Rat3, axis: usize) -> Rat2 {
    match axis {
        0 => [p[1].clone(), p[2].clone()],
        1 => [p[0].clone(), p[2].clone()],
        _ => [p[0].clone(), p[1].clone()],
    }
}

/// Blender `CDT_state::face_edge_offset`: edge ids below it are input edges, and the edges of input
/// face `f` are numbered from `(f + 1) * offset`.
fn cdt_face_edge_offset(input: &CdtInput) -> u32 {
    let longest_face = input.faces.iter().map(Vec::len).max().unwrap_or(0);
    to_u32(longest_face.max(input.edges.len()))
}

/// Blender `prepare_need_*` state: the 2D input of one constrained triangulation, projected along
/// the dominant axis of the plane of `plane_tri`.
struct CdtBuilder {
    plane_n: Rat3,
    plane_d: Rat,
    proj_axis: usize,
    verts: Vec<Rat2>,
    edges: Vec<(u32, u32)>,
    faces: Vec<Vec<u32>>,
    input_face: Vec<usize>,
    is_reversed: Vec<bool>,
}

impl CdtBuilder {
    fn new(arena: &Arena, plane_tri: &Tri) -> Self {
        let (plane_n, plane_d) = tri_plane(arena, plane_tri);
        let proj_axis = dominant_axis(&plane_n);
        Self {
            plane_n,
            plane_d,
            proj_axis,
            verts: Vec::new(),
            edges: Vec::new(),
            faces: Vec::new(),
            input_face: Vec::new(),
            is_reversed: Vec::new(),
        }
    }

    /// Blender `prepare_need_vert`: duplicates are left for the CDT to merge.
    fn need_vert(&mut self, p: &Rat3) -> u32 {
        self.verts.push(project(p, self.proj_axis));
        to_u32(self.verts.len() - 1)
    }

    /// Blender `prepare_need_edge`.
    fn need_edge(&mut self, p1: &Rat3, p2: &Rat3) {
        let a = self.need_vert(p1);
        let b = self.need_vert(p2);
        self.edges.push((a, b));
    }

    /// Blender `prepare_need_tri`: adds input triangle `t` as a face, reversed when its normal
    /// points against the projection direction so that the projected face is counter-clockwise.
    fn need_tri(&mut self, arena: &Arena, tri: &Tri, t: usize) {
        let [i0, i1, i2] = tri.v;
        let v0 = self.need_vert(&arena.vert(i0).co);
        let v1 = self.need_vert(&arena.vert(i1).co);
        let v2 = self.need_vert(&arena.vert(i2).co);
        let (normal, _) = tri_plane(arena, tri);
        let rev = if sgn(&normal[self.proj_axis]) >= 0 {
            self.proj_axis == 1
        } else {
            self.proj_axis != 1
        };
        self.faces.push(if rev {
            vec![v0, v2, v1]
        } else {
            vec![v0, v1, v2]
        });
        self.input_face.push(t);
        self.is_reversed.push(rev);
    }

    /// Runs the constrained Delaunay triangulation (Blender `do_cdt`).
    fn run(self) -> CdtData {
        let input = CdtInput {
            verts: self.verts,
            edges: self.edges,
            faces: self.faces,
        };
        let face_edge_offset = cdt_face_edge_offset(&input);
        let out = delaunay_2d_calc_inside(&input);
        let mut verts_to_edge = HashMap::with_capacity(out.edges.len());
        for (e, &(a, b)) in out.edges.iter().enumerate() {
            verts_to_edge.entry(sorted_pair(a, b)).or_insert(e);
        }
        CdtData {
            plane_n: self.plane_n,
            plane_d: self.plane_d,
            proj_axis: self.proj_axis,
            input_face: self.input_face,
            is_reversed: self.is_reversed,
            out,
            face_edge_offset,
            verts_to_edge,
        }
    }
}

/// Blender `CDT_data`: a solved triangulation of one plane and what is needed to map it back to 3D.
struct CdtData {
    plane_n: Rat3,
    plane_d: Rat,
    proj_axis: usize,
    /// Parallels the CDT faces: index into `tris` of the input triangle of each face.
    input_face: Vec<usize>,
    /// Parallels `input_face`: whether the face was reversed for projection.
    is_reversed: Vec<bool>,
    out: CdtOutput,
    face_edge_offset: u32,
    /// Output edge for each sorted output vertex pair (Blender `verts_to_edge`).
    verts_to_edge: HashMap<(u32, u32), usize>,
}

impl CdtData {
    /// Blender `prepare_cdt_input` followed by `do_cdt` for non-cluster triangle `t` and its
    /// nonempty intersection values.
    fn for_tri(arena: &Arena, tris: &[Tri], t: usize, itts: &[&IttEntry]) -> Self {
        let mut builder = CdtBuilder::new(arena, &tris[t]);
        builder.need_tri(arena, &tris[t], t);
        for entry in itts {
            match &entry.itt.kind {
                IttKind::Point => {
                    builder.need_vert(&entry.itt.p1);
                }
                IttKind::Segment => builder.need_edge(&entry.itt.p1, &entry.itt.p2),
                IttKind::Coplanar => {
                    builder.need_tri(arena, &tris[entry.t_source], entry.t_source);
                }
                IttKind::None => {}
            }
        }
        builder.run()
    }

    /// Blender `unproject_cdt_vert`: recovers the dropped coordinate from the plane equation.
    fn unproject(&self, p2: &Rat2) -> Rat3 {
        let n = &self.plane_n;
        let d = &self.plane_d;
        match self.proj_axis {
            0 => {
                let num = -(&n[1] * &p2[0] + &n[2] * &p2[1] + d);
                [&num / &n[0], p2[0].clone(), p2[1].clone()]
            }
            1 => {
                let num = -(&n[0] * &p2[0] + &n[2] * &p2[1] + d);
                [p2[0].clone(), &num / &n[1], p2[1].clone()]
            }
            _ => {
                let num = -(&n[0] * &p2[0] + &n[1] * &p2[1] + d);
                [p2[0].clone(), p2[1].clone(), &num / &n[2]]
            }
        }
    }

    /// Blender `get_cdt_edge_orig`: the original input edge of the output edge between two output
    /// vertices, preferring a face edge. The flag is set when the edge came from an intersection.
    fn cdt_edge_orig(&self, i0: u32, i1: u32, tris: &[Tri]) -> (i32, bool) {
        let Some(&e) = self.verts_to_edge.get(&sorted_pair(i0, i1)) else {
            return (NO_INDEX, false);
        };
        let foff = self.face_edge_offset;
        let mut face_eorig = NO_INDEX;
        let mut have_non_face_eorig = false;
        for &orig_index in &self.out.edge_orig[e] {
            if orig_index >= foff {
                if face_eorig == NO_INDEX {
                    let in_face = to_usize(orig_index / foff - 1);
                    let pos = to_usize(orig_index % foff);
                    let (Some(&t_in), Some(&reversed)) =
                        (self.input_face.get(in_face), self.is_reversed.get(in_face))
                    else {
                        continue;
                    };
                    let idx = if reversed {
                        2_usize.saturating_sub(pos)
                    } else {
                        pos
                    };
                    if let Some(&eorig) = tris[t_in].edge_orig.get(idx)
                        && eorig != NO_INDEX
                    {
                        face_eorig = eorig;
                    }
                }
            } else {
                have_non_face_eorig = true;
                if face_eorig != NO_INDEX {
                    break;
                }
            }
        }
        if face_eorig != NO_INDEX {
            return (face_eorig, false);
        }
        (NO_INDEX, have_non_face_eorig)
    }

    /// Blender `cdt_tri_as_imesh_face`: output triangle `out_t` as a triangle of input `in_t`.
    fn cdt_tri(&self, arena: &mut Arena, tris: &[Tri], out_t: usize, in_t: usize) -> Tri {
        let [i0, i1, i2] = self.out.faces[out_t];
        let cos = [i0, i1, i2].map(|i| self.unproject(&self.out.verts[to_usize(i)]));
        // No origin is given: an existing vertex at the same position keeps its own.
        let [v0, v1, v2] = cos.map(|co| arena.add_or_find_vert(co, NO_INDEX));
        let orig = tris[self.input_face[in_t]].orig;
        let (v, edges) = if self.is_reversed[in_t] {
            ([v0, v2, v1], [(i0, i2), (i2, i1), (i1, i0)])
        } else {
            ([v0, v1, v2], [(i0, i1), (i1, i2), (i2, i0)])
        };
        let info = edges.map(|(a, b)| self.cdt_edge_orig(a, b, tris));
        Tri {
            v,
            orig,
            edge_orig: info.map(|(eo, _)| eo),
            is_intersect: info.map(|(_, is)| is),
        }
    }

    /// Blender `extract_subdivided_tri`: the output triangles lying inside input triangle `t`.
    fn extract_tri(&self, arena: &mut Arena, tris: &[Tri], t: usize) -> Vec<Tri> {
        let Some(t_in_cdt) = self.input_face.iter().rposition(|&f| f == t) else {
            return Vec::new();
        };
        let mut faces = Vec::new();
        for (out_t, origs) in self.out.face_orig.iter().enumerate() {
            if origs.contains(&to_u32(t_in_cdt)) {
                faces.push(self.cdt_tri(arena, tris, out_t, t_in_cdt));
            }
        }
        faces
    }
}

/// Blender `calc_subdivided_non_cluster_tris`: subdivides each triangle that is not in a cluster,
/// and keeps it whole when nothing cuts it. Triangles in clusters are left for
/// [`calc_cluster_tris`].
fn calc_subdivided_non_cluster_tris(
    arena: &mut Arena,
    subdivided: &mut [Vec<Tri>],
    tris: &[Tri],
    itts: &IttMap,
    clinfo: &CoplanarClusterInfo,
    ov: &TriOverlaps,
) {
    for (t, (tri, out)) in tris.iter().zip(subdivided.iter_mut()).enumerate() {
        if clinfo.tri_cluster(t).is_some() {
            continue;
        }
        let picked: Vec<&IttEntry> = ov
            .partners(t)
            .iter()
            .filter_map(|&(_, other)| itts.get(sorted_pair(t, other)))
            .filter(|entry| !matches!(entry.itt.kind, IttKind::None))
            .collect();
        if !picked.is_empty() {
            let cd = CdtData::for_tri(arena, tris, t, &picked);
            *out = cd.extract_tri(arena, tris, t);
        }
        if out.is_empty() {
            *out = vec![*tri];
        }
    }
}

/// Blender `calc_cluster_subdivided`: triangulates cluster `c` together with the points and
/// segments that other triangles put on its plane.
fn calc_cluster_subdivided(
    arena: &Arena,
    tris: &[Tri],
    clinfo: &CoplanarClusterInfo,
    c: usize,
    ov: &TriOverlaps,
    itts: &IttMap,
) -> CdtData {
    let cl = clinfo.cluster(c);
    let mut picked: Vec<&IttEntry> = Vec::new();
    for &t in cl.tris() {
        for &(_, other) in ov.partners(t) {
            if clinfo.tri_cluster(other) == Some(c) {
                continue;
            }
            if let Some(entry) = itts.get(sorted_pair(t, other))
                && matches!(entry.itt.kind, IttKind::Point | IttKind::Segment)
            {
                picked.push(entry);
            }
        }
    }
    let mut builder = CdtBuilder::new(arena, &tris[cl.first()]);
    for &t in cl.tris() {
        builder.need_tri(arena, &tris[t], t);
    }
    for entry in picked {
        match &entry.itt.kind {
            IttKind::Point => {
                builder.need_vert(&entry.itt.p1);
            }
            IttKind::Segment => builder.need_edge(&entry.itt.p1, &entry.itt.p2),
            IttKind::None | IttKind::Coplanar => {}
        }
    }
    builder.run()
}

/// Blender `calc_cluster_tris`: each cluster triangle receives the output triangles of its cluster
/// that lie inside it. A triangle covered by several inputs yields a copy for each.
fn calc_cluster_tris(
    arena: &mut Arena,
    tris: &[Tri],
    cluster_cd: &[CdtData],
    subdivided: &mut [Vec<Tri>],
) {
    for cd in cluster_cd {
        let mut face_vec: Vec<Vec<Tri>> = vec![Vec::new(); cd.input_face.len()];
        for (out_t, origs) in cd.out.face_orig.iter().enumerate() {
            for &in_t in origs {
                let face = cd.cdt_tri(arena, tris, out_t, to_usize(in_t));
                if let Some(slot) = face_vec.get_mut(to_usize(in_t)) {
                    slot.push(face);
                }
            }
        }
        for (&tm_t, faces) in cd.input_face.iter().zip(face_vec) {
            if let Some(out) = subdivided.get_mut(tm_t) {
                *out = faces;
            }
        }
    }
}

/// Blender `trimesh_nary_intersect`: intersects the triangles of `tris` with each other and returns
/// them subdivided along every intersection.
///
/// `shape_fn` maps a triangle's `orig` to its operand index. With two operands and `use_self`
/// false, only triangles of different operands are intersected.
pub(crate) fn trimesh_nary_intersect(
    arena: &mut Arena,
    tris: &[Tri],
    nshapes: usize,
    shape_fn: &dyn Fn(i32) -> usize,
    use_self: bool,
) -> Vec<Tri> {
    // Blender removes degenerate triangles before anything else (`remove_degenerate_tris`).
    let clean: Vec<Tri> = {
        let ar: &Arena = arena;
        tris.iter()
            .filter(|t| !is_degenerate(ar, t))
            .copied()
            .collect()
    };
    let bbs = calc_face_bounding_boxes(arena, &clean);
    let ov = TriOverlaps::new(&clean, &bbs, nshapes, shape_fn, use_self);
    let itts = calc_overlap_itts(arena, &clean, &ov);
    let clinfo = find_clusters(arena, &clean, &bbs, &itts);
    let mut subdivided: Vec<Vec<Tri>> = vec![Vec::new(); clean.len()];
    calc_subdivided_non_cluster_tris(arena, &mut subdivided, &clean, &itts, &clinfo, &ov);
    let cluster_cd: Vec<CdtData> = {
        let ar: &Arena = arena;
        (0..clinfo.tot_cluster())
            .map(|c| calc_cluster_subdivided(ar, &clean, &clinfo, c, &ov, &itts))
            .collect()
    };
    calc_cluster_tris(arena, &clean, &cluster_cd, &mut subdivided);
    subdivided.into_iter().flatten().collect()
}

#[cfg(test)]
mod tests {
    use glam::DVec3;
    use num_traits::Zero;

    use super::super::exact::{Rat, Rat3, cross3, sub3};
    use super::super::imesh::{Arena, NO_INDEX, Tri};
    use super::cluster::find_clusters;
    use super::{TriOverlaps, calc_face_bounding_boxes, calc_overlap_itts, trimesh_nary_intersect};

    fn shape(orig: i32) -> usize {
        usize::from(orig != 0)
    }

    fn make_tri(arena: &mut Arena, pts: [[f64; 3]; 3], orig: i32) -> Tri {
        let v = pts.map(|p| arena.add_or_find_dvec(DVec3::from_array(p), NO_INDEX));
        Tri::new(v, orig, [NO_INDEX; 3])
    }

    fn int_point(p: [i64; 3]) -> Rat3 {
        p.map(|x| Rat::from_integer(x.into()))
    }

    /// Twice the vector area `cross(v1 - v0, v2 - v0)` summed exactly over the triangles of `orig`.
    /// Subdividing a triangle preserves this sum when the pieces keep its orientation.
    fn area2(arena: &Arena, tris: &[Tri], orig: i32) -> Rat3 {
        let mut sum: Rat3 = [Rat::zero(), Rat::zero(), Rat::zero()];
        for tri in tris.iter().filter(|t| t.orig == orig) {
            let [p0, p1, p2] = tri.v.map(|v| arena.vert(v).co.clone());
            let c = cross3(&sub3(&p1, &p0), &sub3(&p2, &p0));
            for (acc, x) in sum.iter_mut().zip(&c) {
                *acc += x;
            }
        }
        sum
    }

    fn count(tris: &[Tri], orig: i32) -> usize {
        tris.iter().filter(|t| t.orig == orig).count()
    }

    fn has_point(arena: &Arena, tris: &[Tri], orig: i32, want: &Rat3) -> bool {
        tris.iter()
            .filter(|t| t.orig == orig)
            .any(|t| t.v.iter().any(|&v| arena.vert(v).co == *want))
    }

    #[test]
    fn crossing_triangles_are_split_and_keep_area() {
        // `b` lies in the plane y = 1 and crosses the plane z = 0 of `a` along a segment from
        // (1/2, 1, 0) to (3, 1, 0) inside `a`.
        let mut arena = Arena::new();
        let a = make_tri(
            &mut arena,
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
            0,
        );
        let b = make_tri(
            &mut arena,
            [[-1.0, 1.0, -1.0], [5.0, 1.0, -1.0], [2.0, 1.0, 1.0]],
            1,
        );
        let out = trimesh_nary_intersect(&mut arena, &[a, b], 2, &shape, false);

        assert!(out.iter().all(|t| t.orig == 0 || t.orig == 1));
        assert!(count(&out, 0) > 1);
        assert!(count(&out, 1) > 1);
        // Twice the vector areas: a = (0, 0, 16), b = (0, -12, 0).
        assert_eq!(area2(&arena, &out, 0), int_point([0, 0, 16]));
        assert_eq!(area2(&arena, &out, 1), int_point([0, -12, 0]));
        let half_start = [
            Rat::new(1.into(), 2.into()),
            Rat::from_integer(1.into()),
            Rat::zero(),
        ];
        assert!(has_point(&arena, &out, 0, &half_start));
        assert!(has_point(&arena, &out, 0, &int_point([3, 1, 0])));
    }

    #[test]
    fn coplanar_overlapping_squares_form_one_cluster_without_duplicate_area() {
        // Squares [0, 2]^2 (shape 0) and [1, 3]^2 (shape 1), each as two triangles.
        let mut arena = Arena::new();
        let tris = [
            make_tri(
                &mut arena,
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0]],
                0,
            ),
            make_tri(
                &mut arena,
                [[0.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0]],
                0,
            ),
            make_tri(
                &mut arena,
                [[1.0, 1.0, 0.0], [3.0, 1.0, 0.0], [3.0, 3.0, 0.0]],
                1,
            ),
            make_tri(
                &mut arena,
                [[1.0, 1.0, 0.0], [3.0, 3.0, 0.0], [1.0, 3.0, 0.0]],
                1,
            ),
        ];
        let bbs = calc_face_bounding_boxes(&arena, &tris);
        let ov = TriOverlaps::new(&tris, &bbs, 2, &shape, false);
        let itts = calc_overlap_itts(&arena, &tris, &ov);
        assert_eq!(find_clusters(&arena, &tris, &bbs, &itts).tot_cluster(), 1);

        let out = trimesh_nary_intersect(&mut arena, &tris, 2, &shape, false);
        // Each square's pieces cover exactly that square once: 2 * area = 8 each.
        assert_eq!(area2(&arena, &out, 0), int_point([0, 0, 8]));
        assert_eq!(area2(&arena, &out, 1), int_point([0, 0, 8]));
    }

    #[test]
    fn disjoint_triangles_are_returned_unchanged() {
        let mut arena = Arena::new();
        let input = [
            make_tri(
                &mut arena,
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                0,
            ),
            make_tri(
                &mut arena,
                [[5.0, 5.0, 5.0], [6.0, 5.0, 5.0], [5.0, 6.0, 5.0]],
                1,
            ),
        ];
        let verts_before = arena.num_verts();
        let out = trimesh_nary_intersect(&mut arena, &input, 2, &shape, false);
        assert_eq!(out, input.to_vec());
        assert_eq!(arena.num_verts(), verts_before);
    }

    #[test]
    fn pierced_triangle_is_split_along_the_intersection_segment() {
        // `b` lies in the plane x = 1 and pierces `a` through its interior. The intersection is the
        // segment from (1, 1, 0) to (1, 2, 0), which has both ends inside `a`.
        let mut arena = Arena::new();
        let a = make_tri(
            &mut arena,
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
            0,
        );
        let b = make_tri(
            &mut arena,
            [[1.0, 1.0, -1.0], [1.0, 1.0, 1.0], [1.0, 3.0, -1.0]],
            1,
        );
        let out = trimesh_nary_intersect(&mut arena, &[a, b], 2, &shape, false);

        assert!(count(&out, 0) > 1);
        assert!(count(&out, 1) > 1);
        assert_eq!(area2(&arena, &out, 0), int_point([0, 0, 16]));
        assert_eq!(area2(&arena, &out, 1), int_point([-4, 0, 0]));
        assert!(has_point(&arena, &out, 0, &int_point([1, 1, 0])));
        assert!(has_point(&arena, &out, 0, &int_point([1, 2, 0])));
    }

    #[test]
    fn overlap_pairs_are_cross_shape_only_for_two_shapes() {
        // X and Y are shape 0, Z is shape 1. All three boxes overlap, but X and Y must not pair.
        let mut arena = Arena::new();
        let tris = [
            make_tri(
                &mut arena,
                [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
                0,
            ),
            make_tri(
                &mut arena,
                [[1.0, 1.0, -1.0], [1.0, 1.0, 1.0], [1.0, 3.0, -1.0]],
                0,
            ),
            make_tri(
                &mut arena,
                [[-1.0, 1.0, -1.0], [5.0, 1.0, -1.0], [2.0, 1.0, 1.0]],
                1,
            ),
        ];
        let bbs = calc_face_bounding_boxes(&arena, &tris);
        let ov = TriOverlaps::new(&tris, &bbs, 2, &shape, false);
        assert_eq!(ov.pairs, vec![(0, 2), (1, 2), (2, 0), (2, 1)]);
    }
}
