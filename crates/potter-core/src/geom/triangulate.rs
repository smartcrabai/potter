// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   source/blender/bmesh/intern/bmesh_polygon.cc, source/blender/bmesh/tools/bmesh_triangulate.cc,
//   source/blender/bmesh/tools/bmesh_beautify.cc, source/blender/blenlib/intern/polyfill_2d.cc,
//   source/blender/blenlib/intern/polyfill_2d_beautify.cc,
//   source/blender/blenlib/intern/BLI_heap.cc,
//   SPDX-FileCopyrightText: 2001-2023 Blender Authors, GPL-2.0-or-later.

//! Blender's `BM_mesh_triangulate` kernels, evaluated in single precision.
//!
//! Ports `BM_face_triangulate` (`bmesh/intern/bmesh_polygon.cc`), `BM_verts_calc_rotate_beauty`
//! (`bmesh/tools/bmesh_beautify.cc`), `BLI_polyfill_calc_arena` (`blenlib/intern/polyfill_2d.cc`),
//! `BLI_polyfill_beautify` with its quad helpers (`blenlib/intern/polyfill_2d_beautify.cc`) and
//! `BLI_heap` (`blenlib/intern/BLI_heap.cc`). Diagonal choice, tie-breaking and output winding all
//! follow from exact `float` comparisons, so every expression keeps Blender's operand order.
//!
//! The ear search is the brute-force form of Blender's KD-tree search. The tree only prunes
//! subtrees outside the candidate triangle's bounds, so both report the same blocking vertices.

const NONE: usize = usize::MAX;

/// Blender's `quad_method` (`MOD_TRIANGULATE_QUAD_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QuadMethod {
    Beauty,
    Fixed,
    FixedAlternate,
    ShortestDiagonal,
    LongestDiagonal,
}

/// Blender's `ngon_method` (`MOD_TRIANGULATE_NGON_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NgonMethod {
    Beauty,
    Clip,
}

/// One output polygon: its source face and the face-local corners it is built from, in winding
/// order. A triangle has three corners; a face that `min_vertices` leaves alone is one polygon with
/// all its corners.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Polygon {
    pub(crate) face: usize,
    pub(crate) corners: Vec<usize>,
}

type Vec2 = [f32; 2];
type Vec3 = [f32; 3];

/// Triangulates every face of a mesh with at least `min_vertices` corners, returning the output
/// polygons in Blender's order. Faces with fewer corners pass through as one polygon, as
/// `BM_mesh_triangulate` skips them.
///
/// `faces[f]` holds the positions of face `f`'s corners in loop order.
///
/// Order follows `BM_mesh_triangulate`'s in-place layout: a triangulated face keeps its slot and
/// takes its last triangle, the other triangles take new slots, and the slot freed by the last
/// triangle is the next one reused (LIFO). The output is the slots in index order.
pub(crate) fn triangulate_mesh(
    faces: &[Vec<Vec3>],
    quad_method: QuadMethod,
    ngon_method: NgonMethod,
    min_vertices: usize,
) -> Vec<Polygon> {
    let mut slots: Vec<Option<Polygon>> = faces
        .iter()
        .enumerate()
        .map(|(face, corners)| {
            (corners.len() < min_vertices).then(|| Polygon {
                face,
                corners: (0..corners.len()).collect(),
            })
        })
        .collect();
    let mut free: Vec<usize> = Vec::new();
    for (face, corners) in faces.iter().enumerate() {
        if corners.len() < min_vertices {
            continue;
        }
        let mut allocated = Vec::new();
        for triangle in triangulate_face(corners, quad_method, ngon_method) {
            let slot = free.pop().unwrap_or_else(|| {
                slots.push(None);
                slots.len() - 1
            });
            slots[slot] = Some(Polygon {
                face,
                corners: triangle.to_vec(),
            });
            allocated.push(slot);
        }
        let Some(&last) = allocated.last() else {
            continue;
        };
        slots[face] = slots[last].take();
        free.push(last);
    }
    slots.into_iter().flatten().collect()
}

/// `BM_face_triangulate` for one face: the triangles as indices into `corners`.
fn triangulate_face(
    corners: &[Vec3],
    quad_method: QuadMethod,
    ngon_method: NgonMethod,
) -> Vec<[usize; 3]> {
    match corners {
        [_, _, _] => vec![[0, 1, 2]],
        [a, b, c, d] => triangulate_quad([*a, *b, *c, *d], quad_method).to_vec(),
        [_, _, _, _, ..] => triangulate_ngon(corners, face_normal(corners), ngon_method),
        [..] => Vec::new(),
    }
}

fn dot3(a: Vec3, b: Vec3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub3(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross3(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn len_squared3(a: Vec3, b: Vec3) -> f32 {
    let d = sub3(b, a);
    dot3(d, d)
}

/// `cross_tri_v3`: the (unnormalised) normal of the corner `v1, v2, v3`.
fn cross_tri3(v1: Vec3, v2: Vec3, v3: Vec3) -> Vec3 {
    cross3(sub3(v1, v2), sub3(v2, v3))
}

/// `cross_tri_v2`: twice the signed area of `v1, v2, v3`.
fn cross_tri2(v1: Vec2, v2: Vec2, v3: Vec2) -> f32 {
    (v1[0] - v2[0]) * (v2[1] - v3[1]) + (v1[1] - v2[1]) * (v3[0] - v2[0])
}

fn len2(a: Vec2, b: Vec2) -> f32 {
    let x = a[0] - b[0];
    let y = a[1] - b[1];
    (x * x + y * y).sqrt()
}

/// `normalize_v3`: returns the unit vector and the length before normalising (0 when degenerate).
fn normalize3(v: Vec3) -> (Vec3, f32) {
    let mut d = dot3(v, v);
    if d > 1.0e-35 {
        d = d.sqrt();
        let scale = 1.0 / d;
        ([v[0] * scale, v[1] * scale, v[2] * scale], d)
    } else {
        ([0.0; 3], 0.0)
    }
}

/// `ortho_basis_v3v3_v3`: two unit vectors spanning the plane of `n`.
fn ortho_basis(n: Vec3) -> (Vec3, Vec3) {
    let f = n[0] * n[0] + n[1] * n[1];
    if f > f32::EPSILON {
        let d = 1.0 / f.sqrt();
        let n1 = [n[1] * d, -n[0] * d, 0.0];
        let n2 = [-n[2] * n1[1], n[2] * n1[0], n[0] * n1[1] - n[1] * n1[0]];
        (n1, n2)
    } else {
        let n1 = [if n[2] < 0.0 { -1.0 } else { 1.0 }, 0.0, 0.0];
        (n1, [0.0, 1.0, 0.0])
    }
}

/// `axis_dominant_v3_to_m3` followed by `mul_v2_m3v3`: projects `point` onto the plane of `normal`.
fn project(normal: Vec3, point: Vec3) -> Vec2 {
    let (n1, n2) = ortho_basis(normal);
    [dot3(n1, point), dot3(n2, point)]
}

/// Newell's method over the corners, as `bm_face_calc_poly_normal` (`bmesh_polygon.cc`). A face
/// with no net area (a collinear or self-cancelling n-gon) keeps the zero vector: the BMesh
/// triangulation projects onto it as is. Only the Mesh tessellation substitutes +Z, see
/// [`loop_triangles`].
fn face_normal(corners: &[Vec3]) -> Vec3 {
    let mut normal = [0.0_f32; 3];
    let Some(&last) = corners.last() else {
        return normal;
    };
    let mut prev = last;
    for &curr in corners {
        normal[0] += (prev[1] - curr[1]) * (prev[2] + curr[2]);
        normal[1] += (prev[2] - curr[2]) * (prev[0] + curr[0]);
        normal[2] += (prev[0] - curr[0]) * (prev[1] + curr[1]);
        prev = curr;
    }
    let (unit, length) = normalize3(normal);
    if length == 0.0 { [0.0; 3] } else { unit }
}

/// `is_quad_flip_v3`: bit 0 when the 0-2 diagonal is concave, bit 1 for the 1-3 diagonal.
fn is_quad_flip(v1: Vec3, v2: Vec3, v3: Vec3, v4: Vec3) -> (bool, bool) {
    let d12 = sub3(v1, v2);
    let d23 = sub3(v2, v3);
    let d34 = sub3(v3, v4);
    let d41 = sub3(v4, v1);
    let flip_02 = dot3(cross3(d12, d23), cross3(d34, d41)) < 0.0;
    let flip_13 = dot3(cross3(d23, d34), cross3(d41, d12)) < 0.0;
    (flip_02, flip_13)
}

/// `BLI_polyfill_edge_calc_rotate_beauty__area`: the 3D rotation cost of the quad `v1..v4`,
/// negative when the 1-3 diagonal (rather than 2-4) is the better choice.
fn rotate_beauty3(v1: Vec3, v2: Vec3, v3: Vec3, v4: Vec3) -> f32 {
    let no_a = cross_tri3(v2, v3, v4);
    let no_b = cross_tri3(v2, v4, v1);
    let no = [no_a[0] + no_b[0], no_a[1] + no_b[1], no_a[2] + no_b[2]];
    let (unit, no_scale) = normalize3(no);
    if no_scale == 0.0 {
        return f32::MAX;
    }
    let eps = 1e-5_f32;
    let p1 = project(unit, v1);
    let p2 = project(unit, v2);
    let p3 = project(unit, v3);
    let p4 = project(unit, v4);
    let sign_a = signum_eps(cross_tri2(p2, p3, p4) / no_scale, eps);
    let sign_b = signum_eps(cross_tri2(p2, p4, p1) / no_scale, eps);
    if sign_a + sign_b == 0 {
        return f32::MAX;
    }
    quad_rotate_cost(p1, p2, p3, p4).0
}

/// `signum_i_ex`.
fn signum_eps(a: f32, eps: f32) -> i32 {
    if a > eps {
        1
    } else if a < -eps {
        -1
    } else {
        0
    }
}

/// `BLI_polyfill_beautify_quad_rotate_calc_ex`: the 2D rotation cost of the quad `v1..v4` and its
/// area measure. Costs below zero favour rotating the diagonal to 1-3.
fn quad_rotate_cost(v1: Vec2, v2: Vec2, v3: Vec2, v4: Vec2) -> (f32, f32) {
    let a234 = cross_tri2(v2, v3, v4);
    let a241 = cross_tri2(v2, v4, v1);
    let a123 = cross_tri2(v1, v2, v3);
    let a134 = cross_tri2(v1, v3, v4);
    let area = (a234.abs() + a241.abs() + a123.abs() + a134.abs()) / 8.0;
    let eps_zero_area = 1e-12_f32;

    let cost = 'cost: {
        if (a123 >= 0.0) != (a134 >= 0.0) {
            break 'cost f32::MAX;
        }
        if a123.abs() <= eps_zero_area || a134.abs() <= eps_zero_area {
            break 'cost f32::MAX;
        }
        if (a234 >= 0.0) != (a241 >= 0.0) {
            break 'cost -f32::MAX;
        }
        if a234.abs() <= eps_zero_area || a241.abs() <= eps_zero_area {
            break 'cost -f32::MAX;
        }
        let len_12 = len2(v1, v2);
        let len_23 = len2(v2, v3);
        let len_34 = len2(v3, v4);
        let len_41 = len2(v4, v1);
        let len_13 = len2(v1, v3);
        let len_24 = len2(v2, v4);

        let fac_24 =
            (a234.abs() / (len_23 + len_34 + len_24)) + (a241.abs() / (len_41 + len_12 + len_24));
        let fac_13 =
            (a123.abs() / (len_12 + len_23 + len_13)) + (a134.abs() / (len_34 + len_41 + len_13));
        fac_24 - fac_13
    };
    (cost, area)
}

/// `BM_face_triangulate` for a quad. Blender walks the loops from a start corner (`loops`) and
/// splits the quad along 0-2 or 1-3; the result indexes the original corners.
fn triangulate_quad(corners: [Vec3; 4], method: QuadMethod) -> [[usize; 3]; 2] {
    let [co0, co1, co2, co3] = corners;
    let slots: [usize; 4] = match method {
        QuadMethod::Fixed => [0, 1, 2, 3],
        QuadMethod::FixedAlternate => [1, 2, 3, 0],
        QuadMethod::ShortestDiagonal | QuadMethod::LongestDiagonal | QuadMethod::Beauty => {
            let split_02 = match method {
                QuadMethod::ShortestDiagonal => {
                    let d1 = len_squared3(co0, co2);
                    let d2 = len_squared3(co1, co3);
                    (d2 - d1) > 0.0
                }
                QuadMethod::LongestDiagonal => {
                    let d1 = len_squared3(co0, co2);
                    let d2 = len_squared3(co1, co3);
                    (d2 - d1) < 0.0
                }
                _ => {
                    let (flip_02, flip_13) = is_quad_flip(co1, co2, co3, co0);
                    if flip_02 {
                        true
                    } else if flip_13 {
                        false
                    } else {
                        rotate_beauty3(co1, co2, co3, co0) > 0.0
                    }
                }
            };
            if split_02 { [0, 1, 2, 3] } else { [1, 2, 3, 0] }
        }
    };
    [
        [slots[0], slots[1], slots[2]],
        [slots[0], slots[2], slots[3]],
    ]
}

fn triangulate_ngon(corners: &[Vec3], normal: Vec3, method: NgonMethod) -> Vec<[usize; 3]> {
    let negated = [-normal[0], -normal[1], -normal[2]];
    let coords: Vec<Vec2> = corners
        .iter()
        .map(|&corner| project(negated, corner))
        .collect();
    let mut triangles = polyfill(&coords);
    if method == NgonMethod::Beauty {
        beautify(&coords, &mut triangles);
    }
    triangles
}

/// Blender's `Mesh::loop_triangles` tessellation: f32 quad flip predicate and f32 `Clip` polyfill.
/// `corners` are face-local positions rounded to Blender's float vertex storage; each result triple
/// indexes the input corner array in Blender's loop-triangle order.
pub(crate) fn loop_triangles(corners: &[[f32; 3]]) -> Vec<[usize; 3]> {
    match corners {
        [_, _, _] => vec![[0, 1, 2]],
        [v1, v2, v3, v4] => {
            let d12 = sub3(*v2, *v1);
            let d13 = sub3(*v3, *v1);
            let d14 = sub3(*v4, *v1);
            let cross_a = cross3(d12, d13);
            let cross_b = cross3(d14, d13);
            if dot3(cross_a, cross_b) > 0.0 {
                vec![[0, 1, 3], [1, 2, 3]]
            } else {
                vec![[0, 1, 2], [0, 2, 3]]
            }
        }
        _ => {
            // `normal_calc_ngon`: a zero Newell sum projects onto +Z (`normal[2] = 1`).
            let mut normal = face_normal(corners);
            if normal == [0.0; 3] {
                normal = [0.0, 0.0, 1.0];
            }
            triangulate_ngon(corners, normal, NgonMethod::Clip)
        }
    }
}

const CONCAVE: i8 = -1;
const TANGENTIAL: i8 = 0;
const CONVEX: i8 = 1;

fn signum_enum(a: f32) -> i8 {
    if a > 0.0 {
        CONVEX
    } else if a == 0.0 {
        TANGENTIAL
    } else {
        CONCAVE
    }
}

/// `span_tri_v2_sign`: the turn direction of `v1 -> v2 -> v3`.
fn span_sign(v1: Vec2, v2: Vec2, v3: Vec2) -> i8 {
    let d2 = [v2[0] - v3[0], v2[1] - v3[1]];
    let d3 = [v1[0] - v3[0], v1[1] - v3[1]];
    signum_enum(d2[0] * d3[1] - d3[0] * d2[1])
}

/// `tri_isect_precomputed_*`: the inclusive point-in-triangle test Blender's ear search uses.
struct TriangleTest {
    edge: [Vec2; 3],
    c: [f32; 3],
}

impl TriangleTest {
    fn new(vs: [Vec2; 3]) -> Self {
        let edge = [
            [vs[1][0] - vs[0][0], vs[1][1] - vs[0][1]],
            [vs[2][0] - vs[1][0], vs[2][1] - vs[1][1]],
            [vs[0][0] - vs[2][0], vs[0][1] - vs[2][1]],
        ];
        let c = [0, 1, 2].map(|i| edge[i][0] * vs[i][1] - vs[i][0] * edge[i][1]);
        Self { edge, c }
    }

    fn contains(&self, co: Vec2) -> bool {
        let side = |i: usize| self.edge[i][1] * co[0] - self.edge[i][0] * co[1] + self.c[i] >= 0.0;
        side(1) && side(2) && side(0)
    }
}

#[derive(Clone, Copy)]
struct PolyNode {
    next: usize,
    prev: usize,
    sign: i8,
}

/// Ear clipping over a clockwise polygon, as `BLI_polyfill_calc_arena` with `coords_sign == 1`.
/// Node ids are coordinate indices.
struct PolyFill<'a> {
    coords: &'a [Vec2],
    nodes: Vec<PolyNode>,
    head: usize,
    coords_num: usize,
    concave_num: usize,
    tris: Vec<[usize; 3]>,
}

fn polyfill(coords: &[Vec2]) -> Vec<[usize; 3]> {
    let n = coords.len();
    let nodes = (0..n)
        .map(|i| PolyNode {
            next: (i + 1) % n,
            prev: (i + n - 1) % n,
            sign: CONVEX,
        })
        .collect();
    let mut pf = PolyFill {
        coords,
        nodes,
        head: 0,
        coords_num: n,
        concave_num: 0,
        tris: Vec::with_capacity(n.saturating_sub(2)),
    };
    for i in 0..n {
        pf.sign_calc(i);
        if pf.nodes[i].sign != CONVEX {
            pf.concave_num += 1;
        }
    }
    pf.triangulate();
    pf.tris
}

impl PolyFill<'_> {
    fn sign_calc(&mut self, i: usize) {
        let node = self.nodes[i];
        self.nodes[i].sign = span_sign(
            self.coords[node.prev],
            self.coords[i],
            self.coords[node.next],
        );
    }

    fn triangulate(&mut self) {
        let mut ear_init = self.head;
        let mut reverse = false;
        while self.coords_num > 3 {
            let ear = self.ear_tip_find(ear_init, reverse);
            if self.nodes[ear].sign != CONVEX {
                self.concave_num -= 1;
            }
            let prev = self.nodes[ear].prev;
            let next = self.nodes[ear].next;
            self.ear_tip_cut(ear);

            let sign_prev = self.nodes[prev].sign;
            let sign_next = self.nodes[next].sign;
            if sign_prev != CONVEX {
                self.sign_calc(prev);
                if self.nodes[prev].sign == CONVEX {
                    self.concave_num -= 1;
                }
            }
            if sign_next != CONVEX {
                self.sign_calc(next);
                if self.nodes[next].sign == CONVEX {
                    self.concave_num -= 1;
                }
            }

            ear_init = if reverse {
                self.nodes[prev].prev
            } else {
                self.nodes[next].next
            };
            if self.nodes[ear_init].sign != CONVEX {
                ear_init = if reverse {
                    self.nodes[ear_init].prev
                } else {
                    self.nodes[ear_init].next
                };
                reverse = !reverse;
            }
        }
        let a = self.head;
        let b = self.nodes[a].next;
        let c = self.nodes[b].next;
        self.tris.push([a, b, c]);
    }

    fn ear_tip_find(&self, ear_init: usize, reverse: bool) -> usize {
        for sign_accept in [CONVEX, TANGENTIAL] {
            let mut ear = ear_init;
            for _ in 0..self.coords_num {
                if self.ear_tip_check(ear, sign_accept) {
                    return ear;
                }
                ear = if reverse {
                    self.nodes[ear].prev
                } else {
                    self.nodes[ear].next
                };
            }
        }

        // Degenerate polygon: no ear passes, so take the first vertex that is not concave.
        let mut ear = ear_init;
        for _ in 0..self.coords_num {
            if self.nodes[ear].sign != CONCAVE {
                return ear;
            }
            ear = self.nodes[ear].next;
        }
        ear
    }

    fn ear_tip_check(&self, tip: usize, sign_accept: i8) -> bool {
        if self.concave_num == 0 {
            return true;
        }
        if self.nodes[tip].sign != sign_accept {
            return false;
        }
        let next = self.nodes[tip].next;
        let prev = self.nodes[tip].prev;
        let test = TriangleTest::new([self.coords[tip], self.coords[next], self.coords[prev]]);
        let mut node_id = self.head;
        loop {
            let node = self.nodes[node_id];
            if node.sign != CONVEX
                && node_id != tip
                && node_id != next
                && node_id != prev
                && test.contains(self.coords[node_id])
            {
                return false;
            }
            node_id = node.next;
            if node_id == self.head {
                return true;
            }
        }
    }

    fn ear_tip_cut(&mut self, tip: usize) {
        let prev = self.nodes[tip].prev;
        let next = self.nodes[tip].next;
        self.tris.push([prev, tip, next]);
        self.nodes[prev].next = next;
        self.nodes[next].prev = prev;
        if self.head == tip {
            self.head = next;
        }
        self.coords_num -= 1;
    }
}

/// `BLI_heap` node: the value orders the heap, `index` is the node's slot in the tree.
#[derive(Clone, Copy)]
struct HeapNode {
    value: f32,
    index: usize,
    ptr: usize,
}

/// Binary min-heap with Blender's sift rules, so equal costs pop in Blender's order.
#[derive(Default)]
struct Heap {
    nodes: Vec<HeapNode>,
    tree: Vec<usize>,
}

impl Heap {
    fn is_empty(&self) -> bool {
        self.tree.is_empty()
    }

    fn value(&self, slot: usize) -> f32 {
        self.nodes[self.tree[slot]].value
    }

    fn swap(&mut self, i: usize, j: usize) {
        self.tree.swap(i, j);
        self.nodes[self.tree[i]].index = i;
        self.nodes[self.tree[j]].index = j;
    }

    fn sift_up(&mut self, mut i: usize) {
        while i > 0 {
            let parent = (i - 1) >> 1;
            if self.value(parent) < self.value(i) {
                break;
            }
            self.swap(parent, i);
            i = parent;
        }
    }

    fn sift_down(&mut self, mut i: usize) {
        let size = self.tree.len();
        loop {
            let left = 2 * i + 1;
            let right = 2 * i + 2;
            let mut smallest = i;
            if left < size && self.value(left) < self.value(smallest) {
                smallest = left;
            }
            if right < size && self.value(right) < self.value(smallest) {
                smallest = right;
            }
            if smallest == i {
                break;
            }
            self.swap(i, smallest);
            i = smallest;
        }
    }

    fn insert(&mut self, value: f32, ptr: usize) -> usize {
        let node = self.nodes.len();
        self.nodes.push(HeapNode {
            value,
            index: self.tree.len(),
            ptr,
        });
        self.tree.push(node);
        self.sift_up(self.tree.len() - 1);
        node
    }

    /// `BLI_heap_insert_or_update`.
    fn insert_or_update(&mut self, slot: &mut Option<usize>, value: f32, ptr: usize) {
        match *slot {
            None => *slot = Some(self.insert(value, ptr)),
            Some(node) => {
                self.nodes[node].ptr = ptr;
                let current = self.nodes[node].value;
                if value < current {
                    self.nodes[node].value = value;
                    self.sift_up(self.nodes[node].index);
                } else if value > current {
                    self.nodes[node].value = value;
                    self.sift_down(self.nodes[node].index);
                }
            }
        }
    }

    /// `BLI_heap_remove`.
    fn remove(&mut self, node: usize) {
        let mut i = self.nodes[node].index;
        while i > 0 {
            let parent = (i - 1) >> 1;
            self.swap(parent, i);
            i = parent;
        }
        self.pop_min();
    }

    fn pop_min(&mut self) -> usize {
        let ptr = self.nodes[self.tree[0]].ptr;
        let last = self.tree.len() - 1;
        self.swap(0, last);
        self.tree.pop();
        if !self.tree.is_empty() {
            self.sift_down(0);
        }
        ptr
    }
}

#[derive(Clone, Copy)]
struct HalfEdge {
    v: usize,
    e_next: usize,
    e_radial: usize,
    base: usize,
}

fn is_boundary_edge(a: usize, b: usize, coord_last: usize) -> bool {
    a + 1 == b || (a == 0 && b == coord_last)
}

/// `BLI_polyfill_beautify`: flips internal diagonals while that improves the triangulation.
fn beautify(coords: &[Vec2], tris: &mut [[usize; 3]]) {
    let coords_num = coords.len();
    let coord_last = coords_num - 1;
    let tris_len = coords_num - 2;
    let edges_len = tris_len - 1;
    let half_len = 3 * tris_len;

    let mut half = vec![
        HalfEdge {
            v: NONE,
            e_next: NONE,
            e_radial: NONE,
            base: NONE,
        };
        half_len
    ];
    let mut order: Vec<([usize; 2], usize)> = Vec::with_capacity(2 * edges_len);
    for (i, tri) in tris.iter().enumerate() {
        for (j_curr, j_prev) in [(0, 2), (1, 0), (2, 1)] {
            let e_prev = 3 * i + j_prev;
            let e_curr = 3 * i + j_curr;
            half[e_prev].v = tri[j_prev];
            half[e_prev].e_next = e_curr;
            let (a, b) = if tri[j_prev] > tri[j_curr] {
                (tri[j_curr], tri[j_prev])
            } else {
                (tri[j_prev], tri[j_curr])
            };
            if !is_boundary_edge(a, b, coord_last) {
                order.push(([a, b], e_prev));
            }
        }
    }
    order.sort_unstable();

    let mut eheap_table: Vec<Option<usize>> = vec![None; edges_len];
    for (base, pair) in order.as_chunks::<2>().0.iter().enumerate() {
        let [(_, a), (_, b)] = pair;
        half[*a].e_radial = *b;
        half[*b].e_radial = *a;
        half[*a].base = base;
        half[*b].base = base;
    }

    let mut heap = Heap::default();
    for i in 0..half_len {
        if half[i].e_radial < i {
            let (cost, _) = rotate_cost(coords, &half, i);
            eheap_table[half[i].base] = if cost < 0.0 {
                Some(heap.insert(cost, i))
            } else {
                None
            };
        }
    }

    while !heap.is_empty() {
        let e = heap.pop_min();
        eheap_table[half[e].base] = None;
        rotate_edge(&mut half, e);
        let e_next = half[e].e_next;
        let e_next_next = half[e_next].e_next;
        let radial = half[e].e_radial;
        let radial_next = half[radial].e_next;
        let radial_next_next = half[radial_next].e_next;
        for idx in [e_next, e_next_next, radial_next, radial_next_next] {
            if half[idx].base != NONE {
                let base = half[idx].base;
                let (cost, area) = rotate_cost(coords, &half, idx);
                if cost < -1e-6 * area.max(1.0) {
                    heap.insert_or_update(&mut eheap_table[base], cost, idx);
                } else if let Some(node) = eheap_table[base].take() {
                    heap.remove(node);
                }
            }
        }
    }

    let mut out = 0;
    for i in 0..half_len {
        if half[i].v != NONE {
            let mut tri = [0; 3];
            let mut e = i;
            for slot in &mut tri {
                *slot = half[e].v;
                half[e].v = NONE;
                e = half[e].e_next;
            }
            tris[out] = tri;
            out += 1;
        }
    }
}

/// `polyedge_rotate_beauty_calc`: the rotation cost of the internal edge `e`.
fn rotate_cost(coords: &[Vec2], half: &[HalfEdge], e: usize) -> (f32, f32) {
    let radial = half[e].e_radial;
    let e_other = half[half[e].e_next].e_next;
    let radial_other = half[half[radial].e_next].e_next;
    quad_rotate_cost(
        coords[half[e_other].v],
        coords[half[e].v],
        coords[half[radial_other].v],
        coords[half[radial].v],
    )
}

/// `polyedge_rotate`: turns the internal edge `e` to the other diagonal of its quad.
fn rotate_edge(half: &mut [HalfEdge], e: usize) {
    let ed0 = e;
    let ed1 = half[ed0].e_next;
    let ed2 = half[ed1].e_next;
    let ed3 = half[ed0].e_radial;
    let ed4 = half[ed3].e_next;
    let ed5 = half[ed4].e_next;
    half[ed0].e_next = ed2;
    half[ed1].e_next = ed3;
    half[ed2].e_next = ed4;
    half[ed3].e_next = ed5;
    half[ed4].e_next = ed0;
    half[ed5].e_next = ed1;
    half[ed0].v = half[ed5].v;
    half[ed3].v = half[ed2].v;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rotates each triangle to start at its lowest index and sorts, so comparisons ignore order.
    fn canonical(mut tris: Vec<[usize; 3]>) -> Vec<[usize; 3]> {
        for tri in &mut tris {
            let start = (0..3).min_by_key(|&i| tri[i]).unwrap_or(0);
            *tri = [tri[start], tri[(start + 1) % 3], tri[(start + 2) % 3]];
        }
        tris.sort_unstable();
        tris
    }

    fn corners(points: &[[f64; 3]]) -> Vec<Vec3> {
        points.iter().map(|point| point.map(|c| c as f32)).collect()
    }

    #[test]
    fn quad_tie_keeps_the_13_diagonal_under_shortest_and_splits_fixed() {
        // Equal diagonals: Blender's SHORTEST_DIAGONAL only splits 0-2 when it is strictly shorter.
        let rect = corners(&[
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]);
        assert_eq!(
            canonical(triangulate_face(
                &rect,
                QuadMethod::ShortestDiagonal,
                NgonMethod::Beauty
            )),
            canonical(vec![[1, 2, 3], [1, 3, 0]]),
        );
        assert_eq!(
            canonical(triangulate_face(
                &rect,
                QuadMethod::Fixed,
                NgonMethod::Beauty
            )),
            canonical(vec![[0, 1, 2], [0, 2, 3]]),
        );
    }

    #[test]
    fn ngon_beauty_and_clip_match_blender_output() {
        // Non-planar 10-gon triangulated by Blender 5.2.2: BEAUTY flips diagonals that CLIP keeps.
        let points = corners(&[
            [
                0.991_348_158_153_005_3,
                0.081_591_676_090_520_52,
                0.102_818_421_109_174_98,
            ],
            [
                0.056_375_647_637_263_24,
                0.620_621_377_515_415_5,
                0.782_081_128_863_462_1,
            ],
            [
                0.040_104_945_702_991_37,
                0.621_109_866_288_396_3,
                0.782_696_702_004_914_9,
            ],
            [
                -0.028_789_473_950_126_338,
                0.621_352_309_191_189_1,
                0.783_002_218_421_156_1,
            ],
            [
                -0.167_134_067_382_136_54,
                0.612_866_513_157_691_5,
                0.772_308_772_817_088_1,
            ],
            [
                -0.927_270_904_593_232,
                -0.232_725_217_988_163_54,
                -0.293_270_595_878_978_3,
            ],
            [
                -0.597_473_492_250_254_4,
                -0.498_461_985_670_385_9,
                -0.628_140_967_378_996_3,
            ],
            [
                0.754_824_270_391_026_1,
                -0.407_730_805_014_141,
                -0.513_805_324_486_984_8,
            ],
            [
                0.874_135_928_004_623_8,
                -0.301_904_372_168_343_4,
                -0.380_447_275_502_313_9,
            ],
            [
                0.926_847_588_382_773_1,
                -0.233_375_884_111_535_5,
                -0.294_090_538_141_227_15,
            ],
        ]);
        assert_eq!(
            canonical(triangulate_face(
                &points,
                QuadMethod::ShortestDiagonal,
                NgonMethod::Beauty
            )),
            canonical(vec![
                [5, 7, 9],
                [9, 0, 3],
                [0, 1, 3],
                [1, 2, 3],
                [3, 4, 5],
                [5, 6, 7],
                [7, 8, 9],
                [3, 5, 9],
            ]),
        );
        assert_eq!(
            canonical(triangulate_face(
                &points,
                QuadMethod::ShortestDiagonal,
                NgonMethod::Clip
            )),
            canonical(vec![
                [3, 7, 9],
                [9, 0, 1],
                [1, 2, 3],
                [3, 4, 5],
                [5, 6, 7],
                [7, 8, 9],
                [9, 1, 3],
                [3, 5, 7],
            ]),
        );
    }

    #[test]
    fn mesh_layout_reuses_slots_like_blender() {
        // A single quad: Blender emits the 1-3 split as [1, 3, 0] then [1, 2, 3].
        let rect = corners(&[
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]);
        let layout = triangulate_mesh(&[rect], QuadMethod::ShortestDiagonal, NgonMethod::Beauty, 4);
        assert_eq!(
            layout,
            vec![
                Polygon {
                    face: 0,
                    corners: vec![1, 3, 0],
                },
                Polygon {
                    face: 0,
                    corners: vec![1, 2, 3],
                },
            ]
        );
    }

    #[test]
    fn min_vertices_leaves_smaller_faces_whole() {
        // With `min_vertices` 5 a quad is not triangulated and keeps its four corners in its slot.
        let rect = corners(&[
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]);
        let layout = triangulate_mesh(&[rect], QuadMethod::ShortestDiagonal, NgonMethod::Beauty, 5);
        assert_eq!(
            layout,
            vec![Polygon {
                face: 0,
                corners: vec![0, 1, 2, 3],
            }]
        );
    }
}
