//! COLLAPSE decimation: a port of Blender 5.2.2 `BM_mesh_decimate_collapse`
//! (`bmesh/tools/bmesh_decimate_collapse.cc`) on a BMesh-shaped topology. Disk and radial cycle order,
//! edge and face creation order, heap tie order, float32 costs and the triangulate-then-join steps
//! follow Blender so that the surviving edges, and therefore the output, match vertex for vertex.

use std::collections::{BTreeMap, HashMap};

use glam::DVec3;
use serde_json::{Value, json};

use super::{
    CornerUv, PoolSlots,
    heap::{Heap, NodeId},
    interp::{float2_equal, interp_float2},
    kdtree::KdTree,
    normals::{face_center_median, face_normal, project_axis_dominant, vertex_normals},
    polyfill::triangulate_ngon,
    quad::{is_quad_convex, quad_split_24},
};
use crate::{
    color::{linear_to_srgb_f32, srgb_to_linear},
    error::{ErrorCode, PotError, Result},
    geom::{
        Mesh,
        provenance::{Origin, default_value},
    },
    model::Modifier,
};

const NONE: usize = usize::MAX;
/// `BOUNDARY_PRESERVE_WEIGHT`: boundary planes weigh more than face planes.
const BOUNDARY_PRESERVE_WEIGHT: f64 = 100.0;
/// `OPTIMIZE_EPS`: determinant threshold for the quadric's optimal position.
const OPTIMIZE_EPS: f64 = 1.0e-8;
/// `TOPOLOGY_FALLBACK_EPS`: quadric costs below this fall back to the topology cost.
const TOPOLOGY_FALLBACK_EPS: f32 = 1.0e-12;
/// `COST_INVALID`: heap value of edges that must not be collapsed until the mesh changes.
const COST_INVALID: f32 = f32::MAX;
/// `symmetry_eps` in `MOD_decimate.cc`.
const SYMMETRY_EPS: f32 = 0.00002;

/// `Quadric` of `BLI_quadric.h`, in struct order a2, ab, ac, ad, b2, bc, bd, c2, cd, d2.
#[derive(Clone, Copy, Default)]
struct Quadric([f64; 10]);

impl Quadric {
    fn from_plane(plane: [f64; 4]) -> Self {
        let [a, b, c, d] = plane;
        Self([
            a * a,
            a * b,
            a * c,
            a * d,
            b * b,
            b * c,
            b * d,
            c * c,
            c * d,
            d * d,
        ])
    }

    fn add(&mut self, other: &Self) {
        for (value, other) in self.0.iter_mut().zip(other.0) {
            *value += other;
        }
    }

    fn scaled(mut self, factor: f64) -> Self {
        for value in &mut self.0 {
            *value *= factor;
        }
        self
    }

    /// `BLI_quadric_evaluate`.
    fn evaluate(&self, v: [f64; 3]) -> f64 {
        let q = &self.0;
        let v00 = v[0] * v[0];
        let v01 = v[0] * v[1];
        let v02 = v[0] * v[2];
        let v11 = v[1] * v[1];
        let v12 = v[1] * v[2];
        let v22 = v[2] * v[2];
        (q[0] * v00)
            + (q[1] * 2.0 * v01)
            + (q[2] * 2.0 * v02)
            + (q[3] * 2.0 * v[0])
            + (q[4] * v11)
            + (q[5] * 2.0 * v12)
            + (q[6] * 2.0 * v[1])
            + (q[7] * v22)
            + (q[8] * 2.0 * v[2])
            + q[9]
    }

    /// `BLI_quadric_optimize`: the position minimising the quadric, or `None` when it is singular.
    fn optimize(&self, epsilon: f64) -> Option<[f64; 3]> {
        let [a2, ab, ac, ad, b2, bc, bd, c2, cd, _] = self.0;
        let det = a2 * (b2 * c2 - bc * bc) - ab * (ab * c2 - ac * bc) + ac * (ab * bc - ac * b2);
        if det.abs() <= epsilon {
            return None;
        }
        let invdet = 1.0 / det;
        let mut m = [[0.0_f64; 3]; 3];
        m[0][0] = (b2 * c2 - bc * bc) * invdet;
        m[1][0] = (bc * ac - ab * c2) * invdet;
        m[2][0] = (ab * bc - b2 * ac) * invdet;
        m[0][1] = (ac * bc - ab * c2) * invdet;
        m[1][1] = (a2 * c2 - ac * ac) * invdet;
        m[2][1] = (ab * ac - a2 * bc) * invdet;
        m[0][2] = (ab * bc - ac * b2) * invdet;
        m[1][2] = (ac * ab - a2 * bc) * invdet;
        m[2][2] = (a2 * b2 - ab * ab) * invdet;
        let t = [ad, bd, cd];
        let r = [
            m[0][0] * t[0] + m[1][0] * t[1] + m[2][0] * t[2],
            m[0][1] * t[0] + m[1][1] * t[1] + m[2][1] * t[2],
            m[0][2] * t[0] + m[1][2] * t[1] + m[2][2] * t[2],
        ];
        Some([-r[0], -r[1], -r[2]])
    }
}

#[derive(Clone, Copy)]
struct DiskLink {
    next: usize,
    prev: usize,
}

const NO_DISK: DiskLink = DiskLink {
    next: NONE,
    prev: NONE,
};

struct Vert {
    co: [f32; 3],
    no: [f32; 3],
    /// First edge of the disk cycle, or `NONE`.
    edge: usize,
    alive: bool,
    tag: bool,
    /// Provenance: input vertex storage indices with weights summing to one.
    origin: Vec<(usize, f64)>,
    /// POINT attribute values, one per [`BMesh::point_types`] entry (Blender's vertex customdata).
    point: Vec<Value>,
}

#[derive(Clone, Copy)]
struct EdgeFlags {
    seam: bool,
    /// Blender stores sharpness inversely as `BM_ELEM_SMOOTH`; edge splice ORs this flag.
    smooth: bool,
}

impl Default for EdgeFlags {
    fn default() -> Self {
        Self {
            seam: false,
            smooth: true,
        }
    }
}

struct Edge {
    v: [usize; 2],
    disk: [DiskLink; 2],
    /// First loop of the radial cycle, or `NONE`.
    l: usize,
    alive: bool,
    /// EDGE attribute values, one per [`BMesh::edge_types`] entry.
    attributes: Vec<Value>,
    flags: EdgeFlags,
    /// `BLI_mempool` slot of this edge (see [`PoolSlots`]).
    slot: usize,
}

struct Loop {
    v: usize,
    e: usize,
    f: usize,
    next: usize,
    prev: usize,
    radial_next: usize,
    radial_prev: usize,
    /// Triangulation marker: the input face a diagonal loop came from, or -1.
    mark: isize,
}

struct Face {
    first: usize,
    len: usize,
    no: [f32; 3],
    mat: u32,
    /// Input polygon storage index.
    origin: usize,
    alive: bool,
    tag: bool,
    /// `BLI_mempool` slot of this face (see [`PoolSlots`]).
    slot: usize,
}

/// The BMesh-shaped topology the collapse works on. Elements live in arenas indexed by creation order;
/// a killed element is marked dead and never reused, so arena order is Blender's element order for
/// every operation the collapse performs.
struct BMesh {
    verts: Vec<Vert>,
    edges: Vec<Edge>,
    loops: Vec<Loop>,
    faces: Vec<Face>,
    /// UV values per loop: `uv[loop * uv_stride + layer]`.
    uv: Vec<[f32; 2]>,
    uv_stride: usize,
    totface: usize,
    /// Blender's `BMFace` pool slots; live faces are output in slot order.
    face_slots: PoolSlots,
    edge_slots: PoolSlots,
    /// Data type of each POINT attribute, in the order of every vertex's `point` values.
    point_types: Vec<String>,
    /// Data type of each EDGE attribute, in the order of every edge's `attributes` values.
    edge_types: Vec<String>,
    track_edge_flags: bool,
}

impl BMesh {
    fn disk_index(&self, e: usize, v: usize) -> usize {
        usize::from(self.edges[e].v[1] == v)
    }

    fn disk_next(&self, e: usize, v: usize) -> usize {
        self.edges[e].disk[self.disk_index(e, v)].next
    }

    fn vert_in_edge(&self, e: usize, v: usize) -> bool {
        self.edges[e].v[0] == v || self.edges[e].v[1] == v
    }

    fn edge_other_vert(&self, e: usize, v: usize) -> usize {
        if self.edges[e].v[0] == v {
            self.edges[e].v[1]
        } else {
            self.edges[e].v[0]
        }
    }

    fn edge_is_manifold(&self, e: usize) -> bool {
        let l = self.edges[e].l;
        l != NONE
            && self.loops[l].radial_next != l
            && self.loops[self.loops[l].radial_next].radial_next == l
    }

    fn edge_is_boundary(&self, e: usize) -> bool {
        let l = self.edges[e].l;
        l != NONE && self.loops[l].radial_next == l
    }

    fn loop_is_manifold(&self, l: usize) -> bool {
        l != self.loops[l].radial_next && l == self.loops[self.loops[l].radial_next].radial_next
    }

    fn edge_length(&self, e: usize) -> f32 {
        len_vv(
            self.verts[self.edges[e].v[0]].co,
            self.verts[self.edges[e].v[1]].co,
        )
    }

    /// `BM_edge_loop_pair`.
    fn edge_loop_pair(&self, e: usize) -> Option<(usize, usize)> {
        let la = self.edges[e].l;
        if la == NONE {
            return None;
        }
        let lb = self.loops[la].radial_next;
        (la != lb && self.loops[lb].radial_next == la).then_some((la, lb))
    }

    /// `bmesh_disk_edge_append`: appends `e` at the tail of `v`'s disk cycle.
    fn disk_edge_append(&mut self, e: usize, v: usize) {
        let idx = self.disk_index(e, v);
        let head = self.verts[v].edge;
        if head == NONE {
            self.verts[v].edge = e;
            self.edges[e].disk[idx] = DiskLink { next: e, prev: e };
            return;
        }
        let head_idx = self.disk_index(head, v);
        let tail = self.edges[head].disk[head_idx].prev;
        self.edges[e].disk[idx] = DiskLink {
            next: head,
            prev: tail,
        };
        self.edges[head].disk[head_idx].prev = e;
        if tail != NONE {
            let tail_idx = self.disk_index(tail, v);
            self.edges[tail].disk[tail_idx].next = e;
        }
    }

    /// `bmesh_disk_edge_remove`.
    fn disk_edge_remove(&mut self, e: usize, v: usize) {
        let idx = self.disk_index(e, v);
        let DiskLink { next, prev } = self.edges[e].disk[idx];
        if prev != NONE {
            let i = self.disk_index(prev, v);
            self.edges[prev].disk[i].next = next;
        }
        if next != NONE {
            let i = self.disk_index(next, v);
            self.edges[next].disk[i].prev = prev;
        }
        if self.verts[v].edge == e {
            self.verts[v].edge = if e == next { NONE } else { next };
        }
        self.edges[e].disk[idx] = NO_DISK;
    }

    /// `bmesh_radial_loop_append`: the appended loop becomes the head of the radial cycle.
    fn radial_append(&mut self, e: usize, l: usize) {
        let head = self.edges[e].l;
        if head == NONE {
            self.edges[e].l = l;
            self.loops[l].radial_next = l;
            self.loops[l].radial_prev = l;
        } else {
            let after = self.loops[head].radial_next;
            self.loops[l].radial_prev = head;
            self.loops[l].radial_next = after;
            self.loops[after].radial_prev = l;
            self.loops[head].radial_next = l;
            self.edges[e].l = l;
        }
        self.loops[l].e = e;
    }

    /// `bmesh_radial_loop_remove`.
    fn radial_remove(&mut self, e: usize, l: usize) {
        if self.loops[l].radial_next != l {
            if l == self.edges[e].l {
                self.edges[e].l = self.loops[l].radial_next;
            }
            let (next, prev) = (self.loops[l].radial_next, self.loops[l].radial_prev);
            self.loops[next].radial_prev = prev;
            self.loops[prev].radial_next = next;
        } else if l == self.edges[e].l {
            self.edges[e].l = NONE;
        }
        self.loops[l].radial_next = NONE;
        self.loops[l].radial_prev = NONE;
        self.loops[l].e = NONE;
    }

    /// `bmesh_disk_vert_replace`: moves edge `e` from `src` to `dst` in the disk cycles.
    fn disk_vert_replace(&mut self, e: usize, dst: usize, src: usize) {
        self.disk_edge_remove(e, src);
        let idx = self.disk_index(e, src);
        self.edges[e].v[idx] = dst;
        self.edges[e].disk[idx] = NO_DISK;
        self.disk_edge_append(e, dst);
    }

    /// `bmesh_edge_vert_swap`.
    fn edge_vert_swap(&mut self, e: usize, dst: usize, src: usize) {
        let first = self.edges[e].l;
        if first != NONE {
            let mut l = first;
            loop {
                if self.loops[l].v == src {
                    self.loops[l].v = dst;
                } else {
                    let next = self.loops[l].next;
                    if self.loops[next].v == src {
                        self.loops[next].v = dst;
                    }
                }
                l = self.loops[l].radial_next;
                if l == first {
                    break;
                }
            }
        }
        self.disk_vert_replace(e, dst, src);
    }

    /// `BM_vert_splice`: moves every edge of `src` onto `dst` and kills `src`.
    fn vert_splice(&mut self, dst: usize, src: usize) {
        while self.verts[src].edge != NONE {
            let e = self.verts[src].edge;
            self.edge_vert_swap(e, dst, src);
        }
        self.verts[src].alive = false;
    }

    /// `BM_edge_splice`: moves the faces of `src` onto `dst`, then kills `src`.
    fn edge_splice(&mut self, dst: usize, src: usize) {
        while self.edges[src].l != NONE {
            let l = self.edges[src].l;
            self.radial_remove(src, l);
            self.radial_append(dst, l);
        }
        self.edges[dst].flags.seam |= self.edges[src].flags.seam;
        self.edges[dst].flags.smooth |= self.edges[src].flags.smooth;
        self.edge_kill(src);
    }

    /// `BM_edge_kill`: kills the faces around `e`, then `e`.
    fn edge_kill(&mut self, e: usize) {
        while self.edges[e].l != NONE {
            let f = self.loops[self.edges[e].l].f;
            self.face_kill(f);
        }
        let [v1, v2] = self.edges[e].v;
        self.disk_edge_remove(e, v1);
        self.disk_edge_remove(e, v2);
        self.edges[e].alive = false;
        self.edge_slots.release(self.edges[e].slot);
    }

    /// `BM_face_kill`: unlinks the face and releases its pool slot.
    fn face_kill(&mut self, f: usize) {
        self.face_unlink(f);
        self.face_slots.release(self.faces[f].slot);
    }

    /// The loop and radial part of `BM_face_kill`, leaving the pool slot to the caller.
    fn face_unlink(&mut self, f: usize) {
        let first = self.faces[f].first;
        let mut l = first;
        loop {
            let next = self.loops[l].next;
            let e = self.loops[l].e;
            self.radial_remove(e, l);
            l = next;
            if l == first {
                break;
            }
        }
        self.faces[f].alive = false;
        self.totface -= 1;
    }

    /// `BM_edge_exists`.
    fn edge_exists(&self, v1: usize, v2: usize) -> Option<usize> {
        let first = self.verts[v1].edge;
        if first == NONE {
            return None;
        }
        let mut e = first;
        loop {
            if self.vert_in_edge(e, v2) {
                return Some(e);
            }
            e = self.disk_next(e, v1);
            if e == first {
                return None;
            }
        }
    }

    /// `BM_edge_create`, with `BM_CREATE_NO_DOUBLE` when `no_double`.
    fn edge_create(&mut self, v1: usize, v2: usize, no_double: bool) -> usize {
        if no_double && let Some(e) = self.edge_exists(v1, v2) {
            return e;
        }
        let e = self.edges.len();
        let slot = self.edge_slots.alloc();
        self.edges.push(Edge {
            v: [v1, v2],
            disk: [NO_DISK, NO_DISK],
            l: NONE,
            alive: true,
            flags: EdgeFlags::default(),
            attributes: self
                .edge_types
                .iter()
                .map(|data_type| collapse_customdata_default(data_type))
                .collect(),
            slot,
        });
        self.disk_edge_append(e, v1);
        self.disk_edge_append(e, v2);
        e
    }

    /// `BM_face_create`: one loop per corner, each radially appended to its edge. The new face takes the
    /// normal, material and provenance of `example`; loop UVs are copied by the caller.
    fn face_create(&mut self, verts: &[usize], edges: &[usize], example: usize) -> usize {
        let f = self.faces.len();
        let first = self.loops.len();
        let n = verts.len();
        for k in 0..n {
            self.loops.push(Loop {
                v: verts[k],
                e: edges[k],
                f,
                next: first + (k + 1) % n,
                prev: first + (k + n - 1) % n,
                radial_next: NONE,
                radial_prev: NONE,
                mark: -1,
            });
            self.radial_append(edges[k], first + k);
        }
        self.uv.resize(self.loops.len() * self.uv_stride, [0.0; 2]);
        let slot = self.face_slots.alloc();
        let source = &self.faces[example];
        let face = Face {
            first,
            len: n,
            no: source.no,
            mat: source.mat,
            origin: source.origin,
            alive: true,
            tag: false,
            slot,
        };
        self.faces.push(face);
        self.totface += 1;
        f
    }

    /// `BM_edges_from_verts_ensure` followed by `BM_face_create_verts`.
    fn face_create_verts(&mut self, verts: &[usize], example: usize) -> usize {
        let n = verts.len();
        let mut edges = vec![NONE; n];
        let mut prev = n - 1;
        for i in 0..n {
            edges[prev] = self.edge_create(verts[prev], verts[i], true);
            prev = i;
        }
        self.face_create(verts, &edges, example)
    }

    /// `BM_edge_other_loop`.
    fn edge_other_loop(&self, e: usize, l: usize) -> usize {
        let start = if self.loops[l].e == e {
            l
        } else {
            self.loops[l].prev
        };
        let other = self.loops[start].radial_next;
        if self.loops[other].v == self.loops[l].v {
            other
        } else {
            self.loops[other].next
        }
    }

    /// `BM_vert_step_fan_loop`.
    fn step_fan_loop(&self, l: usize, e_step: &mut usize) -> Option<usize> {
        let e_prev = *e_step;
        let e_next = if self.loops[l].e == e_prev {
            self.loops[self.loops[l].prev].e
        } else if self.loops[self.loops[l].prev].e == e_prev {
            self.loops[l].e
        } else {
            return None;
        };
        if self.edge_is_manifold(e_next) {
            *e_step = e_next;
            Some(self.edge_other_loop(e_next, l))
        } else {
            None
        }
    }

    fn radial_facevert_count(&self, l: usize, v: usize) -> usize {
        let mut count = 0;
        let mut it = l;
        loop {
            count += usize::from(self.loops[it].v == v);
            it = self.loops[it].radial_next;
            if it == l {
                return count;
            }
        }
    }

    fn disk_facevert_count(&self, v: usize) -> usize {
        let first = self.verts[v].edge;
        if first == NONE {
            return 0;
        }
        let mut count = 0;
        let mut e = first;
        loop {
            if self.edges[e].l != NONE {
                count += self.radial_facevert_count(self.edges[e].l, v);
            }
            e = self.disk_next(e, v);
            if e == first {
                return count;
            }
        }
    }

    fn disk_faceloop_find_first(&self, e: usize, v: usize) -> usize {
        let mut it = e;
        loop {
            let l = self.edges[it].l;
            if l != NONE {
                return if self.loops[l].v == v {
                    l
                } else {
                    self.loops[l].next
                };
            }
            it = self.disk_next(it, v);
            if it == e {
                return NONE;
            }
        }
    }

    fn radial_faceloop_find_first(&self, l: usize, v: usize) -> usize {
        let mut it = l;
        loop {
            if self.loops[it].v == v {
                return it;
            }
            it = self.loops[it].radial_next;
            if it == l {
                return NONE;
            }
        }
    }

    fn radial_faceloop_find_next(&self, l: usize, v: usize) -> usize {
        let mut it = self.loops[l].radial_next;
        loop {
            if self.loops[it].v == v {
                return it;
            }
            it = self.loops[it].radial_next;
            if it == l {
                return l;
            }
        }
    }

    fn disk_faceedge_find_next(&self, e: usize, v: usize) -> usize {
        let mut found = self.disk_next(e, v);
        loop {
            if self.edges[found].l != NONE && self.radial_facevert_count(self.edges[found].l, v) > 0
            {
                return found;
            }
            found = self.disk_next(found, v);
            if found == e {
                return e;
            }
        }
    }

    /// Loops around `v` in `BM_LOOPS_OF_VERT` order, written to `out`.
    fn loops_of_vert(&self, v: usize, out: &mut Vec<usize>) {
        out.clear();
        let mut count = self.disk_facevert_count(v);
        if count == 0 {
            return;
        }
        let mut l_first = self.disk_faceloop_find_first(self.verts[v].edge, v);
        let mut e_next = self.loops[l_first].e;
        let mut l_next = l_first;
        while count > 0 {
            let l_curr = l_next;
            count -= 1;
            l_next = self.radial_faceloop_find_next(l_next, v);
            if l_next == l_first {
                e_next = self.disk_faceedge_find_next(e_next, v);
                l_first = self.radial_faceloop_find_first(self.edges[e_next].l, v);
                l_next = l_first;
            }
            out.push(l_curr);
        }
    }

    /// Loops of face `f` in face order, written to `out`.
    fn face_loops(&self, f: usize, out: &mut Vec<usize>) {
        out.clear();
        let first = self.faces[f].first;
        let mut l = first;
        loop {
            out.push(l);
            l = self.loops[l].next;
            if l == first {
                return;
            }
        }
    }

    fn face_points(&self, f: usize) -> Vec<[f32; 3]> {
        let mut loops = Vec::new();
        self.face_loops(f, &mut loops);
        loops
            .iter()
            .map(|&l| self.verts[self.loops[l].v].co)
            .collect()
    }

    fn loop_uv(&self, l: usize, layer: usize) -> [f32; 2] {
        self.uv[l * self.uv_stride + layer]
    }

    /// `bm_decim_triangulate_begin`: triangulates every n-gon (quads by the beauty split, others by
    /// polyfill), keeping the source face's normal, material and provenance on the new triangles. Returns
    /// whether any diagonal was created.
    fn triangulate_all(&mut self) -> bool {
        let originals: Vec<usize> = (0..self.faces.len())
            .filter(|&f| self.faces[f].alive && self.faces[f].len > 3)
            .collect();
        let mut doubles = Vec::new();
        let mut has_cut = false;
        for f in originals {
            has_cut |= self.triangulate_face(f, &mut doubles);
        }
        for face in doubles.into_iter().rev() {
            self.face_kill(face);
        }
        has_cut
    }

    /// `bm_face_triangulate` / `BM_face_triangulate` for one face.
    fn triangulate_face(&mut self, f: usize, doubles: &mut Vec<usize>) -> bool {
        let mut loops = Vec::new();
        self.face_loops(f, &mut loops);
        let n = loops.len();
        let (tris, order): (Vec<[usize; 3]>, Vec<usize>) = if n == 4 {
            let (l1, l2, l3, l0) = (loops[1], loops[2], loops[3], loops[0]);
            let split_24 = quad_split_24(
                self.verts[self.loops[l1].v].co,
                self.verts[self.loops[l2].v].co,
                self.verts[self.loops[l3].v].co,
                self.verts[self.loops[l0].v].co,
            );
            let (a, b) = if split_24 { (l0, l2) } else { (l1, l3) };
            (
                vec![[0, 1, 2], [0, 2, 3]],
                vec![a, self.loops[a].next, b, self.loops[b].next],
            )
        } else {
            let points = self.face_points(f);
            let projected = project_axis_dominant(self.faces[f].no, &points);
            (triangulate_ngon(&projected), loops.clone())
        };
        let origin = self.faces[f].origin;
        let Some(last) = tris.len().checked_sub(1) else {
            return false;
        };
        let mut new_faces = Vec::new();
        let mut new_edges = Vec::new();
        let mut last_face = NONE;
        for (i, tri) in tris.iter().enumerate() {
            let ltri = [order[tri[0]], order[tri[1]], order[tri[2]]];
            let verts = [
                self.loops[ltri[0]].v,
                self.loops[ltri[1]].v,
                self.loops[ltri[2]].v,
            ];
            let face = self.face_create_verts(&verts, f);
            last_face = face;
            let first = self.faces[face].first;
            let first_next = self.loops[first].next;
            let first_prev = self.loops[first].prev;
            if self.loops[first].radial_next != first {
                let mut it = self.loops[first].radial_next;
                loop {
                    if self.faces[self.loops[it].f].len == 3
                        && self.loops[first_prev].v == self.loops[self.loops[it].prev].v
                    {
                        doubles.push(face);
                        break;
                    }
                    it = self.loops[it].radial_next;
                    if it == first {
                        break;
                    }
                }
            }
            for (new_loop, source) in [first, first_next, first_prev].into_iter().zip(ltri) {
                for layer in 0..self.uv_stride {
                    self.uv[new_loop * self.uv_stride + layer] =
                        self.uv[source * self.uv_stride + layer];
                }
            }
            let mut it = first;
            loop {
                if self.loops[it].radial_next == it {
                    new_edges.push(self.loops[it].e);
                }
                it = self.loops[it].next;
                if it == first {
                    break;
                }
            }
            if i != last {
                new_faces.push(face);
            }
        }
        for &e in &new_edges {
            let start = self.edges[e].l;
            let mut it = start;
            loop {
                self.loops[it].mark = origin as isize;
                it = self.loops[it].radial_next;
                if it == start {
                    break;
                }
            }
        }
        for face in new_faces {
            let points = self.face_points(face);
            self.faces[face].no = face_normal(&points);
        }
        // Blender swaps the last triangle into `f`'s pool slot and frees the slot it was created in.
        let released = self.faces[last_face].slot;
        self.face_slots.release(released);
        self.faces[last_face].slot = self.faces[f].slot;
        self.face_unlink(f);
        !new_edges.is_empty()
    }

    /// `bm_decim_triangulate_end`: the edges whose two triangles came from the same input face and
    /// merge into a convex quad (or a flat one) are joined back; returns them in edge order.
    fn join_edges(&self) -> Vec<usize> {
        let mut join = Vec::new();
        let can_merge = |bm: &Self, l: usize, mark: isize| {
            bm.loop_is_manifold(l)
                && bm.loops[l].v != bm.loops[bm.loops[l].radial_next].v
                && mark == bm.loops[l].mark
                && mark == bm.loops[bm.loops[l].radial_next].mark
        };
        // BM_ITER_MESH order: edge slots equal arena indices here, no edge is created after a kill.
        for e in 0..self.edges.len() {
            if !self.edges[e].alive {
                continue;
            }
            let Some((la, lb)) = self.edge_loop_pair(e) else {
                continue;
            };
            let la_index = self.loops[la].mark;
            if la_index == -1 || la_index != self.loops[lb].mark {
                continue;
            }
            if self.loops[la].v == self.loops[lb].v {
                continue;
            }
            let (na, pa, nb, pb) = (
                self.loops[la].next,
                self.loops[la].prev,
                self.loops[lb].next,
                self.loops[lb].prev,
            );
            if self.faces[self.loops[la].f].len == 3
                && self.faces[self.loops[lb].f].len == 3
                && !can_merge(self, na, la_index)
                && !can_merge(self, pa, la_index)
                && !can_merge(self, nb, la_index)
                && !can_merge(self, pb, la_index)
            {
                let [v1, v2] = self.edges[e].v;
                let quad = [
                    v1,
                    if self.vert_in_edge(e, self.loops[na].v) {
                        self.loops[pa].v
                    } else {
                        self.loops[na].v
                    },
                    v2,
                    if self.vert_in_edge(e, self.loops[nb].v) {
                        self.loops[pb].v
                    } else {
                        self.loops[nb].v
                    },
                ];
                if !is_quad_convex(
                    self.verts[quad[0]].co,
                    self.verts[quad[1]].co,
                    self.verts[quad[2]].co,
                    self.verts[quad[3]].co,
                ) {
                    continue;
                }
            }
            join.push(e);
        }
        join
    }

    /// `BM_faces_join` of the two faces across `e` (`do_del` false), then `BM_edge_kill` of `e` once no
    /// face uses it. The joined ring starts at the first boundary loop of `faces[0]` (the radial head of
    /// `e`), which is `BM_faces_join`'s `v1`; each new corner copies UVs from the boundary loop on its edge.
    /// A join that doubles an existing face (`BM_face_find_double`) is killed after the joined faces, so
    /// the existing face stays (`bm_decim_triangulate_end`, #159549).
    /// shortcut: a boundary vertex with several open boundary edges takes the first in discovery order, Blender takes disk order; only non-manifold boundaries differ.
    fn join_across(&mut self, e: usize) {
        let Some((la, lb)) = self.edge_loop_pair(e) else {
            return;
        };
        let faces = [self.loops[la].f, self.loops[lb].f];
        if faces[0] == faces[1] {
            return;
        }
        let mut boundary = Vec::new();
        let mut loops = Vec::new();
        for &f in &faces {
            self.face_loops(f, &mut loops);
            for &l in &loops {
                match self.radial_count_in(self.loops[l].e, faces) {
                    1 => boundary.push(l),
                    2 => {}
                    _ => return,
                }
            }
        }
        let Some(&first) = boundary.first() else {
            return;
        };
        // Walks the boundary edge by edge from `first`, as `bm_edges_sort_winding` does.
        let v1 = self.loops[first].v;
        let mut open: Vec<usize> = boundary.iter().map(|&l| self.loops[l].e).collect();
        let mut verts = vec![v1];
        let mut edges = Vec::with_capacity(boundary.len());
        let mut edge = self.loops[first].e;
        let mut v = v1;
        loop {
            let Some(pos) = open.iter().position(|&x| x == edge) else {
                return;
            };
            open.swap_remove(pos);
            edges.push(edge);
            v = self.edge_other_vert(edge, v);
            if edges.len() == boundary.len() {
                if v != v1 {
                    return;
                }
                break;
            }
            verts.push(v);
            let Some(&next) = open.iter().find(|&&x| self.vert_in_edge(x, v)) else {
                return;
            };
            edge = next;
        }
        let mut sources = Vec::with_capacity(verts.len());
        for (&edge_k, &vert_k) in edges.iter().zip(&verts) {
            let Some(&source) = boundary.iter().find(|&&l| self.loops[l].e == edge_k) else {
                return;
            };
            sources.push(if self.loops[source].v == vert_k {
                source
            } else {
                self.loops[source].next
            });
        }
        let joined = self.face_create(&verts, &edges, faces[0]);
        // `BM_faces_join` checks for a double before it kills the joined faces.
        let double = face_find_double(self, joined).is_some();
        let first_new = self.faces[joined].first;
        for (k, &source) in sources.iter().enumerate() {
            for layer in 0..self.uv_stride {
                self.uv[(first_new + k) * self.uv_stride + layer] =
                    self.uv[source * self.uv_stride + layer];
            }
        }
        self.face_kill(faces[0]);
        self.face_kill(faces[1]);
        if double {
            self.face_kill(joined);
        }
        if self.edges[e].l == NONE {
            self.edge_kill(e);
        }
    }

    /// Loops of `e`'s radial cycle that belong to one of `faces` (`bm_loop_systag_count_radial`).
    fn radial_count_in(&self, e: usize, faces: [usize; 2]) -> usize {
        let start = self.edges[e].l;
        let mut count = 0;
        let mut l = start;
        loop {
            count += usize::from(faces.contains(&self.loops[l].f));
            l = self.loops[l].radial_next;
            if l == start {
                break;
            }
        }
        count
    }
}

/// The fields of a loop that [`face_find_double`] reads.
#[derive(Clone, Copy)]
pub(super) struct Ring {
    pub(super) radial_next: usize,
    pub(super) next: usize,
    pub(super) prev: usize,
    pub(super) v: usize,
    pub(super) e: usize,
    pub(super) f: usize,
}

/// The ring queries of [`face_find_double`]. The `BMesh` of COLLAPSE and the `BMesh` of DISSOLVE both
/// implement it, so `BM_face_find_double` exists once.
pub(super) trait FaceRings {
    /// `BM_FACE_FIRST_LOOP` and the corner count of face `f`.
    fn face_first_len(&self, f: usize) -> (usize, usize);
    fn loop_ring(&self, l: usize) -> Ring;
}

impl FaceRings for BMesh {
    fn face_first_len(&self, f: usize) -> (usize, usize) {
        (self.faces[f].first, self.faces[f].len)
    }

    fn loop_ring(&self, l: usize) -> Ring {
        let lp = &self.loops[l];
        Ring {
            radial_next: lp.radial_next,
            next: lp.next,
            prev: lp.prev,
            v: lp.v,
            e: lp.e,
            f: lp.f,
        }
    }
}

/// `BM_face_find_double`: an existing face with the same corners, in either winding, found on the
/// radial cycle of the first loop of `f`.
pub(super) fn face_find_double<B: FaceRings>(bm: &B, f: usize) -> Option<usize> {
    let (l_first, len) = bm.face_first_len(f);
    let first = bm.loop_ring(l_first);
    let mut l_iter = first.radial_next;
    while l_iter != l_first {
        let iter = bm.loop_ring(l_iter);
        if bm.face_first_len(iter.f).1 == len {
            let same_winding = iter.v == first.v;
            let mut a = l_first;
            let mut b = l_iter;
            let mut same = true;
            loop {
                let (ra, rb) = (bm.loop_ring(a), bm.loop_ring(b));
                if ra.e != rb.e {
                    same = false;
                    break;
                }
                a = if same_winding { ra.next } else { ra.prev };
                b = rb.next;
                if b == l_iter {
                    break;
                }
            }
            if same {
                return Some(iter.f);
            }
        }
        l_iter = iter.radial_next;
    }
    None
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// `len_squared_v3v3(a, b)`.
fn len_sq_vv(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d = sub(b, a);
    dot(d, d)
}

/// `len_v3v3(a, b)`.
fn len_vv(a: [f32; 3], b: [f32; 3]) -> f32 {
    len_sq_vv(a, b).sqrt()
}

/// `normalize_v3`.
fn normalize(a: [f32; 3]) -> [f32; 3] {
    let d = dot(a, a);
    if d > 1.0e-35 {
        let mul = 1.0 / d.sqrt();
        [a[0] * mul, a[1] * mul, a[2] * mul]
    } else {
        [0.0; 3]
    }
}

/// `interp_v3_v3v3`.
fn interp(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    let s = 1.0 - t;
    [
        s * a[0] + t * b[0],
        s * a[1] + t * b[1],
        s * a[2] + t * b[2],
    ]
}

/// `line_point_factor_v3` with zero epsilon and fallback.
fn line_point_factor(p: [f32; 3], l1: [f32; 3], l2: [f32; 3]) -> f32 {
    let u = sub(l2, l1);
    let h = sub(p, l1);
    let len_sq = dot(u, u);
    if len_sq > 0.0 {
        dot(u, h) / len_sq
    } else {
        0.0
    }
}

/// `compare_v3v3`.
fn compare_v3(a: [f32; 3], b: [f32; 3], limit: f32) -> bool {
    (0..3).all(|k| (a[k] - b[k]).abs() <= limit)
}

/// `mid_v3_v3v3`.
#[expect(
    clippy::manual_midpoint,
    reason = "Blender computes 0.5 * (a + b), which f32::midpoint does not reproduce bit for bit"
)]
fn mid(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        0.5 * (a[0] + b[0]),
        0.5 * (a[1] + b[1]),
        0.5 * (a[2] + b[2]),
    ]
}

/// `normalize_v3_length_db` with unit length 1: returns the length before normalization.
fn normalize_db(n: &mut [f64; 3]) -> f64 {
    let mut d = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
    if d > 1.0e-35 {
        d = d.sqrt();
        let mul = 1.0 / d;
        n[0] *= mul;
        n[1] *= mul;
        n[2] *= mul;
    } else {
        *n = [0.0; 3];
        d = 0.0;
    }
    d
}

/// Provenance of a vertex interpolated between `keep` (weight `1 - fac`) and `clear` (weight `fac`),
/// as `BM_data_interp_from_verts` blends vertex data.
fn blend_origin(keep: &[(usize, f64)], clear: &[(usize, f64)], fac: f32) -> Vec<(usize, f64)> {
    // `bm_data_interp_from_elem` copies one source outside (0, 1) rather than extrapolating.
    if fac <= 0.0 {
        return keep.to_vec();
    }
    if fac >= 1.0 {
        return clear.to_vec();
    }
    let keep_weight = f64::from(1.0 - fac);
    let clear_weight = f64::from(fac);
    let mut blended = BTreeMap::new();
    for &(index, weight) in keep {
        *blended.entry(index).or_insert(0.0) += weight * keep_weight;
    }
    for &(index, weight) in clear {
        *blended.entry(index).or_insert(0.0) += weight * clear_weight;
    }
    blended.into_iter().collect()
}

/// A POINT attribute of `blender_attributes` that the collapse blends per vertex, with one value per
/// input vertex in storage order.
struct PointAttribute {
    name: String,
    data_type: String,
    values: Vec<Value>,
}

/// The POINT entries of `blender_attributes`; vertices past the stored values take the type default.
fn collect_point_attributes(mesh: &Mesh) -> Vec<PointAttribute> {
    let entries = mesh
        .attributes
        .get("blender_attributes")
        .and_then(Value::as_array)
        .map_or(&[][..], Vec::as_slice);
    entries
        .iter()
        .filter(|entry| entry["domain"].as_str() == Some("POINT"))
        .filter_map(|entry| {
            let name = entry["name"].as_str()?.to_owned();
            let data_type = entry["data_type"].as_str()?.to_owned();
            let stored = entry["values"].as_array()?;
            let values: Vec<Value> = (0..mesh.vertices.len())
                .map(|index| {
                    stored
                        .get(index)
                        .cloned()
                        .unwrap_or_else(|| collapse_customdata_default(&data_type))
                })
                .collect();
            Some(PointAttribute {
                name,
                data_type,
                values,
            })
        })
        .collect()
}

/// An EDGE attribute of `blender_attributes` with one value per input edge in storage order.
pub(super) struct EdgeAttribute {
    pub(super) name: String,
    pub(super) data_type: String,
    pub(super) values: Vec<Value>,
}

/// The EDGE entries of `blender_attributes`; missing edge values take the type default.
pub(super) fn collect_edge_attributes(mesh: &Mesh) -> Vec<EdgeAttribute> {
    let entries = mesh
        .attributes
        .get("blender_attributes")
        .and_then(Value::as_array)
        .map_or(&[][..], Vec::as_slice);
    entries
        .iter()
        .filter(|entry| entry["domain"].as_str() == Some("EDGE"))
        .filter_map(|entry| {
            let name = entry["name"].as_str()?.to_owned();
            let data_type = entry["data_type"].as_str()?.to_owned();
            let stored = entry["values"].as_array()?;
            let values: Vec<Value> = (0..mesh.edges.len())
                .map(|index| {
                    stored
                        .get(index)
                        .cloned()
                        .unwrap_or_else(|| collapse_customdata_default(&data_type))
                })
                .collect();
            Some(EdgeAttribute {
                name,
                data_type,
                values,
            })
        })
        .collect()
}

/// Blender edge flags carried in `BMesh`'s edge header and combined with bitwise `OR` during edge splices.
/// Compat `blender_edge_flags` and native `sharp`/`seam` edge attributes both feed the header.
fn collect_edge_flags(mesh: &Mesh) -> Vec<EdgeFlags> {
    let blender_flags = mesh
        .attributes
        .get("blender_edge_flags")
        .and_then(Value::as_array);
    let native_values = |name: &str| {
        mesh.attributes
            .get(name)
            .and_then(Value::as_object)
            .and_then(|attribute| attribute.get("values"))
            .and_then(Value::as_object)
    };
    let sharp_values = native_values("sharp");
    let seam_values = native_values("seam");
    if blender_flags.is_none() && sharp_values.is_none() && seam_values.is_none() {
        return Vec::new();
    }
    mesh.edges
        .iter()
        .enumerate()
        .map(|(index, edge)| {
            let flags = blender_flags.and_then(|flags| flags.get(index));
            let key = format!("e{}", edge.id);
            let native = |values: Option<&serde_json::Map<String, Value>>| {
                values
                    .and_then(|values| values.get(&key))
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
            };
            let sharp = flags
                .and_then(|flags| flags["sharp"].as_bool())
                .unwrap_or(false)
                || native(sharp_values);
            let seam = flags
                .and_then(|flags| flags["seam"].as_bool())
                .unwrap_or(false)
                || native(seam_values);
            EdgeFlags {
                seam,
                smooth: !sharp,
            }
        })
        .collect()
}

/// Blender's generic color customdata initializes to white (`layerDefault_propcol`, `layerDefault_mloopcol`).
pub(super) fn collapse_customdata_default(data_type: &str) -> Value {
    if matches!(data_type, "FLOAT_COLOR" | "BYTE_COLOR") {
        json!([1.0, 1.0, 1.0, 1.0])
    } else {
        default_value(data_type)
    }
}

fn customdata_has_interp(data_type: &str) -> bool {
    matches!(
        data_type,
        "BOOLEAN"
            | "BOOL"
            | "INT"
            | "FLOAT"
            | "FLOAT2"
            | "FLOAT_VECTOR"
            | "FLOAT_COLOR"
            | "BYTE_COLOR"
    )
}

/// `typeInfo->interp` of Blender's interpolated vertex and edge customdata, blending `sources` with
/// `weights`: `layerInterp_propInt` rounds, `layerInterp_propbool` ORs over positive weights, float
/// values blend linearly, and `layerInterp_mloopcol` blends stored sRGB bytes and rounds each collapse.
/// `None` for types without an interpolation callback, which keep the destination's value.
fn interp_customdata_value(
    data_type: &str,
    sources: [&Value; 2],
    weights: [f32; 2],
) -> Option<Value> {
    match data_type {
        "BOOLEAN" | "BOOL" => {
            Some(json!(sources.iter().zip(weights).any(
                |(value, weight)| weight > 0.0 && value.as_bool() == Some(true)
            )))
        }
        "INT" => {
            let (blended, _) = blend_customdata_components(sources, weights)?;
            Some(json!(blended[0].round() as i64))
        }
        "FLOAT" | "FLOAT2" | "FLOAT_VECTOR" | "FLOAT_COLOR" => {
            let (blended, length) = blend_customdata_components(sources, weights)?;
            Some(if sources[0].is_array() {
                Value::Array(
                    blended[..length]
                        .iter()
                        .map(|component| json!(f64::from(*component)))
                        .collect(),
                )
            } else {
                json!(f64::from(blended[0]))
            })
        }
        "BYTE_COLOR" => {
            let first = byte_color_bytes(sources[0])?;
            let second = byte_color_bytes(sources[1])?;
            let blended = std::array::from_fn(|index| {
                round_to_byte(first[index] * weights[0] + second[index] * weights[1])
            });
            Some(byte_color_linear(blended))
        }
        _ => None,
    }
}

/// Components of two customdata values blended as `a * w0 + b * w1` in float32. Blender's
/// interpolated scalar, float2, float3 and float4 values fit in this stack buffer.
fn blend_customdata_components(
    sources: [&Value; 2],
    weights: [f32; 2],
) -> Option<([f32; 4], usize)> {
    let mut blended = [0.0_f32; 4];
    match (sources[0], sources[1]) {
        (Value::Array(first), Value::Array(second)) if first.len() == second.len() => {
            if first.len() > blended.len() {
                return None;
            }
            for (index, (a, b)) in first.iter().zip(second).enumerate() {
                blended[index] = a.as_f64()? as f32 * weights[0] + b.as_f64()? as f32 * weights[1];
            }
            Some((blended, first.len()))
        }
        (first, second) => {
            blended[0] = first.as_f64()? as f32 * weights[0] + second.as_f64()? as f32 * weights[1];
            Some((blended, 1))
        }
    }
}

/// `round_fl_to_uchar_clamp` as an integral byte value.
fn round_to_byte(value: f32) -> f32 {
    (value.clamp(0.0, 255.0) + 0.5).floor()
}

/// Stored `BYTE_COLOR` bytes of a serialized linear RGBA value: RGB is sRGB-encoded, alpha is a linear byte.
fn byte_color_bytes(value: &Value) -> Option<[f32; 4]> {
    let linear = value
        .as_array()
        .filter(|components| components.len() == 4)?;
    let mut bytes = [0.0_f32; 4];
    for (index, component) in linear.iter().enumerate() {
        let channel = component.as_f64()?.clamp(0.0, 1.0) as f32;
        let stored = if index < 3 {
            linear_to_srgb_f32(channel)
        } else {
            channel
        };
        bytes[index] = round_to_byte(stored * 255.0);
    }
    Some(bytes)
}

/// Serialized linear RGBA of stored `BYTE_COLOR` bytes; the inverse of [`byte_color_bytes`].
fn byte_color_linear(bytes: [f32; 4]) -> Value {
    Value::Array(
        bytes
            .iter()
            .enumerate()
            .map(|(index, byte)| {
                let encoded = f64::from(*byte) / 255.0;
                json!(if index < 3 {
                    srgb_to_linear(encoded)
                } else {
                    encoded
                })
            })
            .collect(),
    )
}

/// Builds the `BMesh` of `mesh` as `BKE_mesh_to_bmesh_ex` does for decimation: vertices, then edges in
/// mesh order, then faces with radially appended loops. Face normals are recomputed and vertex normals
/// are the mesh normals; corner UVs come from `uv_values[layer][corner]`.
fn bmesh_from_mesh(
    mesh: &Mesh,
    uv_values: &[Vec<[f64; 2]>],
    point: &[PointAttribute],
    edge_attributes: &[EdgeAttribute],
    edge_flags: &[EdgeFlags],
) -> BMesh {
    let index_of = super::super::id_indices(mesh);
    let points: Vec<[f32; 3]> = mesh
        .vertices
        .iter()
        .map(|v| [v.co.x as f32, v.co.y as f32, v.co.z as f32])
        .collect();
    let faces_indices: Vec<Vec<usize>> = mesh
        .faces
        .iter()
        .map(|face| face.vertices.iter().map(|id| index_of[id]).collect())
        .collect();
    let normals = vertex_normals(&points, &faces_indices);
    let uv_stride = uv_values.len();
    let mut bm = BMesh {
        verts: points
            .iter()
            .zip(&normals)
            .enumerate()
            .map(|(index, (&co, &no))| Vert {
                co,
                no,
                edge: NONE,
                alive: true,
                tag: false,
                origin: vec![(index, 1.0)],
                point: point
                    .iter()
                    .map(|attribute| attribute.values[index].clone())
                    .collect(),
            })
            .collect(),
        edges: Vec::with_capacity(mesh.edges.len()),
        loops: Vec::new(),
        faces: Vec::with_capacity(mesh.faces.len()),
        uv: Vec::new(),
        uv_stride,
        totface: 0,
        face_slots: PoolSlots::default(),
        edge_slots: PoolSlots::default(),
        point_types: point
            .iter()
            .map(|attribute| attribute.data_type.clone())
            .collect(),
        edge_types: edge_attributes
            .iter()
            .map(|attribute| attribute.data_type.clone())
            .collect(),
        track_edge_flags: !edge_flags.is_empty(),
    };
    let mut edge_lookup = HashMap::with_capacity(mesh.edges.len());
    for (edge_index, edge) in mesh.edges.iter().enumerate() {
        let v = [index_of[&edge.vertices[0]], index_of[&edge.vertices[1]]];
        let e = bm.edge_create(v[0], v[1], false);
        for (value, attribute) in bm.edges[e].attributes.iter_mut().zip(edge_attributes) {
            *value = attribute.values[edge_index].clone();
        }
        bm.edges[e].flags = edge_flags.get(edge_index).copied().unwrap_or_default();
        edge_lookup.insert((v[0].min(v[1]), v[0].max(v[1])), e);
    }
    let mut corner = 0;
    for (face_index, indices) in faces_indices.iter().enumerate() {
        let n = indices.len();
        let edges: Vec<usize> = (0..n)
            .map(|k| {
                let (a, b) = (indices[k], indices[(k + 1) % n]);
                edge_lookup[&(a.min(b), a.max(b))]
            })
            .collect();
        let first = bm.loops.len();
        let f = bm.faces.len();
        for k in 0..n {
            bm.loops.push(Loop {
                v: indices[k],
                e: edges[k],
                f,
                next: first + (k + 1) % n,
                prev: first + (k + n - 1) % n,
                radial_next: NONE,
                radial_prev: NONE,
                mark: -1,
            });
            bm.radial_append(edges[k], first + k);
        }
        bm.uv.resize(bm.loops.len() * uv_stride, [0.0; 2]);
        for (layer, values) in uv_values.iter().enumerate() {
            for k in 0..n {
                let uv = values[corner + k];
                bm.uv[(first + k) * uv_stride + layer] = [uv[0] as f32, uv[1] as f32];
            }
        }
        corner += n;
        let verts: Vec<[f32; 3]> = indices.iter().map(|&i| points[i]).collect();
        let slot = bm.face_slots.alloc();
        bm.faces.push(Face {
            first,
            len: n,
            no: face_normal(&verts),
            mat: mesh.faces[face_index].material_index,
            origin: face_index,
            alive: true,
            tag: false,
            slot,
        });
        bm.totface += 1;
    }
    bm
}

struct Decimator {
    bm: BMesh,
    quadrics: Vec<Quadric>,
    vweights: Option<Vec<f32>>,
    vweight_factor: f32,
    heap: Heap,
    /// Heap node of each edge, `None` when the edge is not queued.
    table: Vec<Option<NodeId>>,
    /// Mirror edge of each edge, when the symmetry option is on.
    sym: Option<Vec<Option<usize>>>,
    sym_axis: usize,
    scratch: Vec<usize>,
}

impl Decimator {
    fn clear_edge_cost(&mut self, e: usize) {
        if let Some(node) = self.table[e].take() {
            self.heap.remove(node);
        }
    }

    fn invalid_edge_cost(&mut self, e: usize) {
        self.table[e] = Some(self.heap.insert(COST_INVALID, e));
    }

    #[expect(
        clippy::manual_midpoint,
        reason = "Blender computes 0.5 * (a + b), which f64::midpoint does not reproduce bit for bit"
    )]
    fn target_co_db(&self, e: usize) -> [f64; 3] {
        let [v1, v2] = self.bm.edges[e].v;
        let mut q = self.quadrics[v1];
        q.add(&self.quadrics[v2]);
        q.optimize(OPTIMIZE_EPS).unwrap_or_else(|| {
            let (a, b) = (self.bm.verts[v1].co, self.bm.verts[v2].co);
            [
                0.5 * (f64::from(a[0]) + f64::from(b[0])),
                0.5 * (f64::from(a[1]) + f64::from(b[1])),
                0.5 * (f64::from(a[2]) + f64::from(b[2])),
            ]
        })
    }

    fn target_co(&self, e: usize) -> [f32; 3] {
        let co = self.target_co_db(e);
        [co[0] as f32, co[1] as f32, co[2] as f32]
    }

    /// `bm_edge_collapse_is_degenerate_flip`.
    fn is_flip(&mut self, e: usize, optimize: [f32; 3]) -> bool {
        let mut loops = std::mem::take(&mut self.scratch);
        let mut flipped = false;
        'verts: for side in 0..2 {
            let v = self.bm.edges[e].v[side];
            self.bm.loops_of_vert(v, &mut loops);
            for &l in &loops {
                let lp = &self.bm.loops[l];
                if lp.e == e || self.bm.loops[lp.prev].e == e {
                    continue;
                }
                let co_prev = self.bm.verts[self.bm.loops[lp.prev].v].co;
                let co_next = self.bm.verts[self.bm.loops[lp.next].v].co;
                let co_v = self.bm.verts[v].co;
                let vec_other = sub(co_prev, co_next);
                let cross_exist = cross(vec_other, sub(co_prev, co_v));
                let cross_optim = cross(vec_other, sub(co_prev, optimize));
                if dot(cross_exist, cross_optim)
                    <= (dot(cross_exist, cross_exist) + dot(cross_optim, cross_optim)) * 0.01
                {
                    flipped = true;
                    break 'verts;
                }
            }
        }
        self.scratch = loops;
        flipped
    }

    fn manifold_or_boundary(&self, e: usize) -> bool {
        let l = self.bm.edges[e].l;
        l != NONE && self.bm.loops[self.bm.loops[l].radial_next].radial_next == l
    }

    fn tag_set(&mut self, e: usize, value: bool) {
        let [v1, v2] = self.bm.edges[e].v;
        self.bm.verts[v1].tag = value;
        self.bm.verts[v2].tag = value;
        let l = self.bm.edges[e].l;
        if l != NONE {
            let f = self.bm.loops[l].f;
            self.bm.faces[f].tag = value;
            let next = self.bm.loops[l].radial_next;
            if l != next {
                let f = self.bm.loops[next].f;
                self.bm.faces[f].tag = value;
            }
        }
    }

    fn tag_test(&self, e: usize) -> bool {
        let [v1, v2] = self.bm.edges[e].v;
        if self.bm.verts[v1].tag || self.bm.verts[v2].tag {
            return true;
        }
        let l = self.bm.edges[e].l;
        if l == NONE {
            return false;
        }
        if self.bm.faces[self.bm.loops[l].f].tag {
            return true;
        }
        let next = self.bm.loops[l].radial_next;
        l != next && self.bm.faces[self.bm.loops[next].f].tag
    }

    /// `bm_edge_collapse_is_degenerate_topology`: tags both disks to find a face or vertex that the
    /// collapse would duplicate.
    fn is_topologically_degenerate(&mut self, e_first: usize) -> bool {
        let [v1, v2] = self.bm.edges[e_first].v;
        for pivot in [v1, v2] {
            let mut e_iter = e_first;
            loop {
                if !self.manifold_or_boundary(e_iter) {
                    return true;
                }
                self.tag_set(e_iter, false);
                e_iter = self.bm.disk_next(e_iter, pivot);
                if e_iter == e_first {
                    break;
                }
            }
        }
        let mut e_iter = e_first;
        loop {
            self.tag_set(e_iter, true);
            e_iter = self.bm.disk_next(e_iter, v1);
            if e_iter == e_first {
                break;
            }
        }
        let radial = self.bm.edges[e_first].l;
        let other = self.bm.loops[radial].radial_next;
        for start in [radial, other]
            .into_iter()
            .take(if radial == other { 1 } else { 2 })
        {
            self.bm.faces[self.bm.loops[start].f].tag = false;
            let mut l = start;
            for _ in 0..3 {
                self.bm.verts[self.bm.loops[l].v].tag = false;
                l = self.bm.loops[l].next;
            }
        }
        let mut e_iter = e_first;
        loop {
            if self.tag_test(e_iter) {
                return true;
            }
            e_iter = self.bm.disk_next(e_iter, v2);
            if e_iter == e_first {
                return false;
            }
        }
    }

    /// `bm_decim_build_edge_cost_single`.
    fn build_edge_cost(&mut self, e: usize) {
        let [v1, v2] = self.bm.edges[e].v;
        if let Some(w) = &self.vweights
            && (w[v1] == 0.0 || w[v2] == 0.0)
        {
            self.clear_edge_cost(e);
            return;
        }
        let l = self.bm.edges[e].l;
        let collapsible = if self.bm.edge_is_boundary(e) {
            self.bm.faces[self.bm.loops[l].f].len == 3
        } else if self.bm.edge_is_manifold(e) {
            let next = self.bm.loops[l].radial_next;
            self.bm.faces[self.bm.loops[l].f].len == 3
                && self.bm.faces[self.bm.loops[next].f].len == 3
        } else {
            false
        };
        if !collapsible {
            self.clear_edge_cost(e);
            return;
        }
        let optimize = self.target_co_db(e);
        let cost_db = self.quadrics[v1].evaluate(optimize) + self.quadrics[v2].evaluate(optimize);
        let mut cost = (cost_db as f32).abs();
        if cost < TOPOLOGY_FALLBACK_EPS {
            let (no1, no2) = (self.bm.verts[v1].no, self.bm.verts[v2].no);
            let (co1, co2) = (self.bm.verts[v1].co, self.bm.verts[v2].co);
            let dot_no = dot(no1, no2).abs();
            if let Some(w) = &self.vweights {
                let e_weight = w[v1] + w[v2];
                cost = dot_no / (-len_vv(co1, co2)).min(-f32::EPSILON) - cost;
                if e_weight != 0.0 {
                    cost *= 1.0 + e_weight * self.vweight_factor;
                }
            } else {
                cost = dot_no / (-len_sq_vv(co1, co2)).min(-f32::EPSILON) - cost;
            }
        } else if let Some(w) = &self.vweights {
            let e_weight = 2.0 - (w[v1] + w[v2]);
            if e_weight != 0.0 {
                cost += self.bm.edge_length(e) * (e_weight * self.vweight_factor);
            }
        }
        self.heap.insert_or_update(&mut self.table[e], cost, e);
    }

    /// `bm_decim_build_quadrics`: a plane per face at each of its corners, plus boundary planes weighted
    /// by [`BOUNDARY_PRESERVE_WEIGHT`].
    fn build_quadrics(&mut self) {
        let mut loops = Vec::new();
        for f in 0..self.bm.faces.len() {
            if !self.bm.faces[f].alive {
                continue;
            }
            let center = face_center_median(&self.bm.face_points(f));
            let no = self.bm.faces[f].no;
            let (nx, ny, nz) = (f64::from(no[0]), f64::from(no[1]), f64::from(no[2]));
            let d = -(nx * f64::from(center[0])
                + ny * f64::from(center[1])
                + nz * f64::from(center[2]));
            let q = Quadric::from_plane([nx, ny, nz, d]);
            self.bm.face_loops(f, &mut loops);
            for &l in &loops {
                let v = self.bm.loops[l].v;
                self.quadrics[v].add(&q);
            }
        }
        for e in 0..self.bm.edges.len() {
            if !self.bm.edges[e].alive || !self.bm.edge_is_boundary(e) {
                continue;
            }
            let [v1, v2] = self.bm.edges[e].v;
            let (co1, co2) = (self.bm.verts[v1].co, self.bm.verts[v2].co);
            let f = self.bm.loops[self.bm.edges[e].l].f;
            let edge_plane = cross(sub(co2, co1), self.bm.faces[f].no);
            let mut plane = [
                f64::from(edge_plane[0]),
                f64::from(edge_plane[1]),
                f64::from(edge_plane[2]),
            ];
            if normalize_db(&mut plane) > f64::from(f32::EPSILON) {
                let center = mid(co1, co2);
                let d = -(plane[0] * f64::from(center[0])
                    + plane[1] * f64::from(center[1])
                    + plane[2] * f64::from(center[2]));
                let q = Quadric::from_plane([plane[0], plane[1], plane[2], d])
                    .scaled(BOUNDARY_PRESERVE_WEIGHT);
                self.quadrics[v1].add(&q);
                self.quadrics[v2].add(&q);
            }
        }
    }

    /// `bm_edge_symmetry_map`: the mirror of each edge across `axis`, found by the kd-tree first hit.
    fn symmetry_map(&self, axis: usize) -> Vec<Option<usize>> {
        let edges: Vec<usize> = (0..self.bm.edges.len())
            .filter(|&e| self.bm.edges[e].alive)
            .collect();
        let mids: Vec<[f32; 3]> = edges
            .iter()
            .map(|&e| {
                let [a, b] = self.bm.edges[e].v;
                mid(self.bm.verts[a].co, self.bm.verts[b].co)
            })
            .collect();
        let tree = KdTree::new(&mids);
        let limit_sq = SYMMETRY_EPS * SYMMETRY_EPS;
        let mut map: Vec<Option<usize>> = vec![None; self.bm.edges.len()];
        for &i in &edges {
            let [a, b] = self.bm.edges[i].v;
            let mut co = mid(self.bm.verts[a].co, self.bm.verts[b].co);
            co[axis] *= -1.0;
            let mut v1_co = self.bm.verts[a].co;
            let mut v2_co = self.bm.verts[b].co;
            v1_co[axis] *= -1.0;
            v2_co[axis] *= -1.0;
            let e_dir = sub(v2_co, v1_co);
            let mut found = None;
            tree.range_search(co, SYMMETRY_EPS, |index, _, _| {
                let other = edges[index];
                let [o1, o2] = self.bm.edges[other].v;
                let other_dir = sub(self.bm.verts[o2].co, self.bm.verts[o1].co);
                let mismatch = if dot(other_dir, e_dir) > 0.0 {
                    len_sq_vv(v1_co, self.bm.verts[o1].co) > limit_sq
                        || len_sq_vv(v2_co, self.bm.verts[o2].co) > limit_sq
                } else {
                    len_sq_vv(v1_co, self.bm.verts[o2].co) > limit_sq
                        || len_sq_vv(v2_co, self.bm.verts[o1].co) > limit_sq
                };
                if mismatch {
                    true
                } else {
                    found = Some(other);
                    false
                }
            });
            if let Some(other) = found {
                map[i] = Some(other);
                map[other] = Some(i);
            }
        }
        map
    }

    /// `BM_data_interp_from_verts(bm, v_other, v_clear, v_other, fac)` for POINT customdata.
    fn interpolate_point(&mut self, v_other: usize, v_clear: usize, fac: f32) {
        if !self
            .bm
            .point_types
            .iter()
            .any(|ty| customdata_has_interp(ty))
        {
            return;
        }
        if fac <= 0.0 {
            return;
        }
        if fac >= 1.0 {
            self.bm.verts[v_other].point = self.bm.verts[v_clear].point.clone();
            return;
        }
        let weights = [1.0 - fac, fac];
        for (index, data_type) in self.bm.point_types.iter().enumerate() {
            let sources = [
                &self.bm.verts[v_other].point[index],
                &self.bm.verts[v_clear].point[index],
            ];
            if let Some(value) = interp_customdata_value(data_type, sources, weights) {
                self.bm.verts[v_other].point[index] = value;
            }
        }
    }

    /// `BM_data_interp_from_edges` for the edge that survives each adjacent-edge splice.
    fn interpolate_edge(&mut self, destination: usize, source: usize, fac: f32) {
        if !self
            .bm
            .edge_types
            .iter()
            .any(|ty| customdata_has_interp(ty))
        {
            return;
        }
        if fac <= 0.0 {
            return;
        }
        if fac >= 1.0 {
            self.bm.edges[destination].attributes = self.bm.edges[source].attributes.clone();
            return;
        }
        let weights = [1.0 - fac, fac];
        for (index, data_type) in self.bm.edge_types.iter().enumerate() {
            let sources = [
                &self.bm.edges[destination].attributes[index],
                &self.bm.edges[source].attributes[index],
            ];
            if let Some(value) = interp_customdata_value(data_type, sources, weights) {
                self.bm.edges[destination].attributes[index] = value;
            }
        }
    }

    /// `bm_edge_collapse_loop_customdata`: interpolates the UV corners around the collapsed edge, layer
    /// by layer, where a corner still holds its source's value.
    fn interpolate_loops(&mut self, l: usize, v_clear: usize, fac: f32) {
        let stride = self.bm.uv_stride;
        let is_manifold = self.bm.edge_is_manifold(self.bm.loops[l].e);
        let (l_clear, l_other) = if self.bm.loops[l].v == v_clear {
            (l, self.bm.loops[l].next)
        } else {
            (self.bm.loops[l].next, l)
        };
        let f_exit = if is_manifold {
            self.bm.loops[self.bm.loops[l].radial_next].f
        } else {
            NONE
        };
        for side in 0..2 {
            let mut e_prev = self.bm.loops[l].e;
            let (first, src0, src1, w) = if side == 0 {
                (l_clear, l_clear, l_other, [fac, 1.0 - fac])
            } else {
                (l_other, l_other, l_clear, [1.0 - fac, fac])
            };
            let mut l_iter = first;
            while let Some(next) = self.bm.step_fan_loop(l_iter, &mut e_prev) {
                l_iter = next;
                if l_iter == first || (f_exit != NONE && f_exit == self.bm.loops[l_iter].f) {
                    break;
                }
                for layer in 0..stride {
                    let value0 = self.bm.loop_uv(src0, layer);
                    if float2_equal(value0, self.bm.loop_uv(l_iter, layer)) {
                        let value1 = self.bm.loop_uv(src1, layer);
                        self.bm.uv[l_iter * stride + layer] = interp_float2(&[value0, value1], &w);
                    }
                }
            }
        }
    }

    /// `loop` edges of triangle `l` beside `v_clear`: the one through `v_clear` first.
    fn loop_other_edges(&self, l: usize, v_clear: usize) -> (usize, usize) {
        let prev = self.bm.loops[self.bm.loops[l].prev].e;
        let next = self.bm.loops[self.bm.loops[l].next].e;
        if self.bm.vert_in_edge(prev, v_clear) {
            (prev, next)
        } else {
            (next, prev)
        }
    }

    /// `bm_edge_collapse`: collapses `e_clear` onto `v_clear`'s other end. On success returns the edges
    /// that were removed beside the collapsed edge (`None` for a boundary edge).
    fn edge_collapse(
        &mut self,
        e_clear: usize,
        v_clear: usize,
        fac: f32,
    ) -> Option<[Option<usize>; 2]> {
        let v_other = self.bm.edge_other_vert(e_clear, v_clear);
        let manifold = self.bm.edge_is_manifold(e_clear);
        let (pairs, killed) = if manifold {
            let (la, lb) = self.bm.edge_loop_pair(e_clear)?;
            let (a0, a1) = self.loop_other_edges(la, v_clear);
            let (b0, b1) = self.loop_other_edges(lb, v_clear);
            if a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1 {
                return None;
            }
            (vec![(a0, a1), (b0, b1)], [Some(a0), Some(b0)])
        } else if self.bm.edge_is_boundary(e_clear) {
            let (a0, a1) = self.loop_other_edges(self.bm.edges[e_clear].l, v_clear);
            (vec![(a0, a1)], [Some(a0), None])
        } else {
            return None;
        };
        for &(source, destination) in &pairs {
            self.interpolate_edge(destination, source, fac);
        }
        if self.bm.uv_stride > 0 {
            let first = self.bm.edges[e_clear].l;
            self.interpolate_loops(first, v_clear, fac);
            if manifold {
                let second = self.bm.loops[first].radial_next;
                self.interpolate_loops(second, v_clear, fac);
            }
        }
        self.bm.edge_kill(e_clear);
        self.bm.vert_splice(v_other, v_clear);
        for &(keep, far) in &pairs {
            self.bm.edge_splice(far, keep);
        }
        if let Some(sym) = &mut self.sym {
            for (&(_, far), gone) in pairs.iter().zip(killed) {
                if let Some(gone) = gone
                    && let Some(mirror) = sym[gone]
                {
                    sym[mirror] = Some(far);
                }
            }
        }
        Some(killed)
    }

    /// `bm_decim_edge_collapse`. `co` is the target position; with `calc` it is recomputed and checked
    /// for topology and flips first.
    fn decim_edge_collapse(&mut self, e: usize, co: &mut [f32; 3], calc: bool) -> bool {
        let [v_other, v_clear] = self.bm.edges[e].v;
        let v_clear_no = self.bm.verts[v_clear].no;
        if calc {
            if self.is_topologically_degenerate(e) {
                self.invalid_edge_cost(e);
                return false;
            }
            *co = self.target_co(e);
            if self.is_flip(e, *co) {
                self.invalid_edge_cost(e);
                return false;
            }
        }
        let (c1, c2) = (self.bm.verts[v_other].co, self.bm.verts[v_clear].co);
        let fac = if compare_v3(c1, c2, f32::EPSILON) {
            0.5
        } else {
            line_point_factor(*co, c1, c2)
        };
        let Some(killed) = self.edge_collapse(e, v_clear, fac) else {
            self.invalid_edge_cost(e);
            return false;
        };
        if let Some(w) = &mut self.vweights {
            let (keep, clear) = (w[v_other], w[v_clear]);
            w[v_other] = (fac * keep + (1.0 - fac) * clear).clamp(0.0, 1.0);
        }
        self.bm.verts[v_other].co = *co;
        for gone in killed.iter().flatten() {
            if let Some(node) = self.table[*gone].take() {
                self.heap.remove(node);
            }
        }
        let clear_quadric = self.quadrics[v_clear];
        self.quadrics[v_other].add(&clear_quadric);
        let origin = blend_origin(
            &self.bm.verts[v_other].origin,
            &self.bm.verts[v_clear].origin,
            fac,
        );
        self.bm.verts[v_other].origin = origin;
        self.interpolate_point(v_other, v_clear, fac);
        self.bm.verts[v_other].no = normalize(interp(self.bm.verts[v_other].no, v_clear_no, fac));
        let disk_first = self.bm.verts[v_other].edge;
        let mut e_iter = disk_first;
        loop {
            self.build_edge_cost(e_iter);
            e_iter = self.bm.disk_next(e_iter, v_other);
            if e_iter == disk_first {
                break;
            }
        }
        let mut loops = std::mem::take(&mut self.scratch);
        self.bm.loops_of_vert(v_other, &mut loops);
        for &l in &loops {
            if self.bm.faces[self.bm.loops[l].f].len == 3 {
                let prev_edge = self.bm.loops[self.bm.loops[l].prev].e;
                let e_outer = if self.bm.vert_in_edge(prev_edge, self.bm.loops[l].v) {
                    self.bm.loops[self.bm.loops[l].next].e
                } else {
                    prev_edge
                };
                self.build_edge_cost(e_outer);
            }
        }
        self.scratch = loops;
        true
    }

    /// The main loop of `BM_mesh_decimate_collapse` without symmetry.
    fn run(&mut self, target: usize) {
        while self.bm.totface > target
            && !self.heap.is_empty()
            && self.heap.top_value() != COST_INVALID
        {
            let e = self.heap.pop_min().unwrap_or(NONE);
            self.table[e] = None;
            let mut co = [0.0_f32; 3];
            self.decim_edge_collapse(e, &mut co, true);
        }
    }

    /// The main loop of `BM_mesh_decimate_collapse` with symmetry: an edge and its mirror collapse
    /// together, and edges sharing a vertex with their mirror are dropped.
    fn run_symmetric(&mut self, target: usize) {
        let axis = self.sym_axis;
        while self.bm.totface > target
            && !self.heap.is_empty()
            && self.heap.top_value() != COST_INVALID
        {
            let e = self.heap.pop_min().unwrap_or(NONE);
            let mirror = self.sym.as_ref().and_then(|sym| sym[e]);
            self.table[e] = None;
            let mut e_mirr = None;
            let mut invalidate = 0_u8;
            let mut to_invalidate = false;
            match mirror {
                Some(m) if m == e => {}
                Some(m) if self.table[m].is_some() => {
                    e_mirr = Some(m);
                    to_invalidate = self.share_vert(e, m);
                }
                Some(_) => {
                    invalidate |= 1;
                    to_invalidate = true;
                }
                None => {}
            }
            if !to_invalidate {
                let ok_a = !self.is_topologically_degenerate(e);
                let ok_b = e_mirr.is_none_or(|m| !self.is_topologically_degenerate(m));
                let mut co = self.target_co(e);
                if mirror == Some(e) {
                    co[axis] = 0.0;
                }
                let degenerate = !ok_a || !ok_b;
                if degenerate || self.is_flip(e, co) {
                    invalidate |= 1 | if e_mirr.is_some() { 2 } else { 0 };
                    to_invalidate = true;
                } else if self.decim_edge_collapse(e, &mut co, false) {
                    if let Some(m) = e_mirr
                        && let Some(node) = self.table[m].take()
                    {
                        self.heap.remove(node);
                        co[axis] *= -1.0;
                        self.decim_edge_collapse(m, &mut co, false);
                    }
                    continue;
                } else if e_mirr.is_some_and(|m| self.table[m].is_some()) {
                    invalidate |= 2;
                    to_invalidate = true;
                } else {
                    continue;
                }
            }
            debug_assert!(to_invalidate);
            if invalidate & 1 != 0 {
                self.invalid_edge_cost(e);
            }
            if invalidate & 2 != 0
                && let Some(m) = e_mirr
                && let Some(node) = self.table[m].take()
            {
                self.heap.remove(node);
                self.invalid_edge_cost(m);
            }
        }
    }

    fn share_vert(&self, e1: usize, e2: usize) -> bool {
        let [a, b] = self.bm.edges[e1].v;
        let [c, d] = self.bm.edges[e2].v;
        a == c || a == d || b == c || b == d
    }

    /// Output polygons, provenance and corner UVs of the decimated mesh. With `join`, the triangles are
    /// joined back first (`bm_decim_triangulate_end`); every live face is then emitted from its own ring
    /// start, as `BM_mesh_bm_to_me` does.
    fn output(&mut self, join: bool, layer_count: usize) -> Output {
        if join {
            for e in self.bm.join_edges() {
                self.bm.join_across(e);
            }
        }
        let mut out_vertex = vec![NONE; self.bm.verts.len()];
        let mut points = Vec::new();
        let mut origins = Vec::new();
        let mut point = Vec::new();
        for (index, vert) in self.bm.verts.iter().enumerate() {
            if vert.alive {
                out_vertex[index] = points.len();
                points.push(DVec3::new(
                    f64::from(vert.co[0]),
                    f64::from(vert.co[1]),
                    f64::from(vert.co[2]),
                ));
                origins.push(vert.origin.clone());
                if !vert.point.is_empty() {
                    point.push(vert.point.clone());
                }
            }
        }
        let mut polygons: Vec<Polygon> = Vec::new();
        let mut faces_origin = Vec::new();
        let mut corner_uv: Vec<CornerUv> = Vec::new();
        let mut ring = Vec::new();
        let mut order: Vec<usize> = (0..self.bm.faces.len())
            .filter(|&f| self.bm.faces[f].alive)
            .collect();
        order.sort_by_key(|&f| self.bm.faces[f].slot);
        for f in order {
            self.bm.face_loops(f, &mut ring);
            let verts = ring
                .iter()
                .map(|&l| out_vertex[self.bm.loops[l].v])
                .collect();
            polygons.push((verts, self.bm.faces[f].mat));
            faces_origin.push(Some(self.bm.faces[f].origin));
            for &l in &ring {
                corner_uv.push(
                    (0..layer_count)
                        .map(|layer| {
                            let uv = self.bm.loop_uv(l, layer);
                            [f64::from(uv[0]), f64::from(uv[1])]
                        })
                        .collect(),
                );
            }
        }
        let loose = self
            .bm
            .edges
            .iter()
            .filter(|edge| edge.alive && edge.l == NONE)
            .map(|edge| [out_vertex[edge.v[0]], out_vertex[edge.v[1]]])
            .collect();
        // Blender's edge pool order: every per-edge row below is in this order, and `decimate_compact`
        // puts the rebuilt mesh's edges in it.
        let mut pool: Vec<usize> = (0..self.bm.edges.len())
            .filter(|&e| self.bm.edges[e].alive)
            .collect();
        pool.sort_by_key(|&e| self.bm.edges[e].slot);
        let edge_order = pool
            .iter()
            .map(|&e| {
                let [first, second] = self.bm.edges[e].v.map(|vertex| out_vertex[vertex]);
                [first.min(second), first.max(second)]
            })
            .collect();
        let edge_attributes: Vec<Vec<Value>> = if self.bm.edge_types.is_empty() {
            Vec::new()
        } else {
            pool.iter()
                .map(|&e| self.bm.edges[e].attributes.clone())
                .collect()
        };
        let edge_flags: Vec<EdgeFlags> = if self.bm.track_edge_flags {
            pool.iter().map(|&e| self.bm.edges[e].flags).collect()
        } else {
            Vec::new()
        };
        Output {
            edge_order,
            points,
            polygons,
            loose,
            origins,
            point,
            faces: faces_origin,
            corner_uv,
            edge_attributes,
            edge_flags,
        }
    }
}

/// The decimated mesh before it is compacted into a [`Mesh`].
struct Output {
    points: Vec<DVec3>,
    polygons: Vec<Polygon>,
    loose: Vec<[usize; 2]>,
    origins: Vec<Vec<(usize, f64)>>,
    /// POINT attribute values per output vertex when present, in [`collect_point_attributes`] order.
    point: Vec<Vec<Value>>,
    faces: Vec<Option<usize>>,
    corner_uv: Vec<CornerUv>,
    /// EDGE attribute values per alive edge, in pool order (one row per edge).
    edge_attributes: Vec<Vec<Value>>,
    /// Blender's seam/sharp header flags per alive edge, in pool order, when tracked.
    edge_flags: Vec<EdgeFlags>,
    /// Alive edges as sorted output-vertex pairs, in Blender's edge-pool order.
    edge_order: Vec<[usize; 2]>,
}

/// `Polygon` of `modifiers.rs`: vertex indices and material index.
type Polygon = super::super::Polygon;

/// Writes the collapse's POINT values into the `blender_attributes` entries of `mesh`: `values[v][a]` is
/// attribute `a` (in `attributes` order) of output vertex `v`.
fn collapse_write_point(
    mesh: &mut Mesh,
    attributes: &[PointAttribute],
    values: &[Vec<Value>],
) -> Result<()> {
    if attributes.is_empty() {
        return Ok(());
    }
    let dropped = || {
        PotError::new(
            ErrorCode::InternalError,
            "decimate point attributes do not match the rebuilt mesh",
        )
    };
    if values.len() != mesh.vertices.len() {
        return Err(dropped());
    }
    let entries = mesh
        .attributes
        .get_mut("blender_attributes")
        .and_then(Value::as_array_mut)
        .ok_or_else(dropped)?;
    for (index, attribute) in attributes.iter().enumerate() {
        let entry = entries
            .iter_mut()
            .find(|entry| {
                entry["domain"].as_str() == Some("POINT")
                    && entry["name"].as_str() == Some(attribute.name.as_str())
            })
            .ok_or_else(dropped)?;
        entry["values"] = Value::Array(values.iter().map(|row| row[index].clone()).collect());
    }
    Ok(())
}

/// Writes the EDGE values into `blender_attributes`. `rows` holds one row of `attributes` values per edge,
/// in the rebuilt mesh's edge order (the pool order `decimate_compact` gives the edges), so row `k` is edge `k`.
pub(super) fn collapse_write_edge(
    mesh: &mut Mesh,
    attributes: &[EdgeAttribute],
    rows: Vec<Vec<Value>>,
) -> Result<()> {
    if attributes.is_empty() {
        return Ok(());
    }
    let dropped = || {
        PotError::new(
            ErrorCode::InternalError,
            "decimate edge attributes do not match the rebuilt mesh",
        )
    };
    if rows.len() != mesh.edges.len() || rows.iter().any(|row| row.len() != attributes.len()) {
        return Err(dropped());
    }
    let mut by_attribute: Vec<Vec<Value>> = (0..attributes.len())
        .map(|_| Vec::with_capacity(rows.len()))
        .collect();
    for row in rows {
        for (attribute, value) in by_attribute.iter_mut().zip(row) {
            attribute.push(value);
        }
    }
    let entries = mesh
        .attributes
        .get_mut("blender_attributes")
        .and_then(Value::as_array_mut)
        .ok_or_else(dropped)?;
    for (index, attribute) in attributes.iter().enumerate() {
        let entry = entries
            .iter_mut()
            .find(|entry| {
                entry["domain"].as_str() == Some("EDGE")
                    && entry["name"].as_str() == Some(attribute.name.as_str())
            })
            .ok_or_else(dropped)?;
        entry["values"] = Value::Array(std::mem::take(&mut by_attribute[index]));
    }
    Ok(())
}

/// A native edge-flag attribute and the `EdgeFlags` value it stores.
type FlagStore = (&'static str, fn(&EdgeFlags) -> bool);

/// Writes `BMesh` seam/sharp flags onto the output edges; `BM_edge_splice` ORs these header flags. `flags`
/// holds one value per edge in the rebuilt mesh's edge order, so entry `k` is edge `k`.
fn collapse_write_edge_flags(mesh: &mut Mesh, flags: &[EdgeFlags]) -> Result<()> {
    if flags.is_empty() {
        return Ok(());
    }
    let invalid = || {
        PotError::new(
            ErrorCode::InternalError,
            "decimate edge flags do not match the rebuilt mesh",
        )
    };
    if flags.len() != mesh.edges.len() {
        return Err(invalid());
    }
    let natives: [FlagStore; 2] = [
        ("sharp", |flags| !flags.smooth),
        ("seam", |flags| flags.seam),
    ];
    for (name, flag) in natives {
        if let Some(attribute) = mesh.attributes.get_mut(name) {
            let values = attribute
                .get_mut("values")
                .and_then(Value::as_object_mut)
                .ok_or_else(invalid)?;
            values.clear();
            for (edge, edge_flags) in mesh.edges.iter().zip(flags) {
                values.insert(format!("e{}", edge.id), json!(flag(edge_flags)));
            }
        }
    }
    if let Some(blender_edge_flags) = mesh.attributes.get_mut("blender_edge_flags") {
        let values = blender_edge_flags.as_array_mut().ok_or_else(invalid)?;
        if values.len() != flags.len() {
            return Err(invalid());
        }
        for (value, edge_flags) in values.iter_mut().zip(flags) {
            let fields = value.as_object_mut().ok_or_else(invalid)?;
            fields.insert("seam".to_owned(), json!(edge_flags.seam));
            fields.insert("sharp".to_owned(), json!(!edge_flags.smooth));
        }
    }
    Ok(())
}

/// COLLAPSE decimation of `mesh` to `ratio` of its triangles, as Blender's Decimate modifier does.
pub(in crate::geom::modifiers) fn decimate_collapse(
    mesh: &Mesh,
    ratio: f64,
    modifier: &Modifier,
) -> Result<Mesh> {
    if mesh.faces.len() <= 3 {
        return Ok(mesh.clone());
    }
    let groups = super::decimate_group_weights(mesh, modifier)?;
    let weight_factor = super::number_param(modifier, "vertex_group_factor")?;
    if !(0.0..=1.0).contains(&weight_factor) {
        return Err(super::invalid_parameter(
            modifier,
            "vertex_group_factor",
            "a number from 0 through 1",
        ));
    }
    let do_triangulate = super::bool_param(modifier, "use_collapse_triangulate")?;
    let symmetry_axis = if super::bool_param(modifier, "use_symmetry")? {
        Some(super::decimate_symmetry_axis(modifier)?)
    } else {
        None
    };
    let (layer_names, layer_values) = super::collapse_uv_layers(mesh);
    let point_attributes = collect_point_attributes(mesh);
    let edge_attributes = collect_edge_attributes(mesh);
    let edge_flags = collect_edge_flags(mesh);
    let mut bm = bmesh_from_mesh(
        mesh,
        &layer_values,
        &point_attributes,
        &edge_attributes,
        &edge_flags,
    );
    let has_cut = bm.triangulate_all();
    let target = (bm.totface as f32 * ratio as f32) as usize;
    let vertex_count = bm.verts.len();
    let edge_count = bm.edges.len();
    let vweights = if weight_factor > 0.0 {
        groups.map(|weights| weights.into_iter().map(|w| w as f32).collect())
    } else {
        None
    };
    let mut dec = Decimator {
        bm,
        quadrics: vec![Quadric::default(); vertex_count],
        vweights,
        vweight_factor: weight_factor as f32,
        heap: Heap::with_capacity(edge_count),
        table: vec![None; edge_count],
        sym: None,
        sym_axis: 0,
        scratch: Vec::new(),
    };
    dec.build_quadrics();
    for e in 0..edge_count {
        if dec.bm.edges[e].alive {
            dec.build_edge_cost(e);
        }
    }
    if let Some(axis) = symmetry_axis {
        dec.sym = Some(dec.symmetry_map(axis));
        dec.sym_axis = axis;
        dec.run_symmetric(target);
    } else {
        dec.run(target);
    }
    let output = dec.output(!do_triangulate && has_cut, layer_names.len());
    let keep_vertices = vec![true; output.points.len()];
    let origin = Origin {
        vertices: output.origins,
        source_vertices: None,
        faces: output.faces,
        corners: None,
    };
    let mut out = super::super::decimate_compact(
        output.points,
        output.polygons,
        output.loose,
        origin,
        mesh,
        &keep_vertices,
        &output.edge_order,
    )?;
    super::collapse_write_uv(&mut out, &layer_names, &output.corner_uv)?;
    collapse_write_point(&mut out, &point_attributes, &output.point)?;
    collapse_write_edge(&mut out, &edge_attributes, output.edge_attributes)?;
    collapse_write_edge_flags(&mut out, &output.edge_flags)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{BMesh, Decimator, Heap, PoolSlots, interp_customdata_value};
    use serde_json::json;

    #[test]
    fn customdata_interpolation_matches_blender_types() {
        let first_collapse =
            interp_customdata_value("INT", [&json!(0), &json!(1)], [0.5, 0.5]).unwrap_or_default();
        assert_eq!(
            interp_customdata_value("INT", [&first_collapse, &json!(0)], [0.5, 0.5]),
            Some(json!(1))
        );
        assert_eq!(
            interp_customdata_value("BOOLEAN", [&json!(false), &json!(true)], [0.25, 0.75]),
            Some(json!(true))
        );
        assert_eq!(
            interp_customdata_value("FLOAT", [&json!(0.25), &json!(0.75)], [0.25, 0.75]),
            Some(json!(0.625))
        );
        assert_eq!(
            interp_customdata_value(
                "FLOAT_COLOR",
                [&json!([0.0, 0.0, 0.0, 1.0]), &json!([1.0, 0.5, 0.25, 0.0])],
                [0.25, 0.75],
            ),
            Some(json!([0.75, 0.375, 0.1875, 0.25]))
        );
        let midpoint = interp_customdata_value(
            "BYTE_COLOR",
            [&json!([0.0, 0.0, 0.0, 1.0]), &json!([1.0, 1.0, 1.0, 1.0])],
            [0.5, 0.5],
        )
        .unwrap_or_default();
        assert_eq!(
            super::byte_color_bytes(&midpoint),
            Some([128.0, 128.0, 128.0, 255.0])
        );
        let linear = midpoint[0].as_f64().unwrap_or(f64::NAN);
        assert!((linear - 0.215_860_530_734_062_2).abs() < 1.0e-6);
    }
    #[test]
    fn edge_customdata_interpolates_before_splice() {
        let mut bm = BMesh {
            verts: (0..3)
                .map(|_| super::Vert {
                    co: [0.0; 3],
                    no: [0.0; 3],
                    edge: super::NONE,
                    alive: true,
                    tag: false,
                    origin: Vec::new(),
                    point: Vec::new(),
                })
                .collect(),
            edges: Vec::new(),
            loops: Vec::new(),
            faces: Vec::new(),
            uv: Vec::new(),
            uv_stride: 0,
            totface: 0,
            point_types: Vec::new(),
            face_slots: PoolSlots::default(),
            edge_slots: PoolSlots::default(),
            edge_types: vec!["INT".to_owned()],
            track_edge_flags: false,
        };
        let destination = bm.edge_create(0, 1, false);
        let source = bm.edge_create(1, 2, false);
        bm.edges[destination].attributes[0] = json!(0);
        bm.edges[source].attributes[0] = json!(1);
        let mut decimator = Decimator {
            bm,
            quadrics: Vec::new(),
            vweights: None,
            vweight_factor: 0.0,
            heap: Heap::with_capacity(0),
            table: Vec::new(),
            sym: None,
            sym_axis: 0,
            scratch: Vec::new(),
        };

        decimator.interpolate_edge(destination, source, 0.5);
        assert_eq!(decimator.bm.edges[destination].attributes[0], json!(1));
    }
}
