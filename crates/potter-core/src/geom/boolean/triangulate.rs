//! Polygon triangulation for the Mesh Arrangements port (PR17a-1).
//!
//! Ports Blender's `triangulate_poly` chain from `blenlib/intern/mesh_intersect.cc`
//! (`is_quad_flip_first_third`, `polyfill_triangulate_poly`, `exact_triangulate_poly`,
//! `any_degenerate_tris_fast`, `triangulate_poly`) and the float ear clipper behind
//! `BLI_polyfill_calc` (`blenlib/intern/polyfill_2d.cc`).
//!
//! Deviations from Blender:
//! - `polyfill_2d.cc` finds blocking vertices through a KD-tree with an index cache. Both only
//!   accelerate the search: the ear accepted at each step is the same. This port tests the
//!   non-convex ring vertices directly, using the KD-tree's precomputed-edge predicate.
//! - `face_is_degenerate` has no caller in Blender 5.2 and is not ported.
//! - Only the clockwise-projection branch of `BLI_polyfill_calc` is ported, because
//!   `polyfill_triangulate_poly` always passes `coords_sign == 1`.

use std::collections::HashMap;

use glam::DVec3;

use crate::geom::edge_key;

use super::cdt::{CdtInput, delaunay_2d_calc_inside};
use super::exact::Rat2;
use super::imesh::{Arena, NO_INDEX, Tri};

/// Blender `CONCAVE`, `TANGENTIAL`, `CONVEX` ear classes of `polyfill_2d.cc`.
const CONCAVE: i8 = -1;
const TANGENTIAL: i8 = 0;
const CONVEX: i8 = 1;

/// Triangulate one input face (Blender `triangulate_poly` for faces of any length).
///
/// `verts` are arena indices in face order. `edge_orig[k]` is the original index of the edge from
/// `verts[k]` to `verts[k + 1]`, or [`NO_INDEX`]. Triangles keep the face winding. Diagonals
/// added by the triangulation get [`NO_INDEX`]. Faces with fewer than three vertices give nothing.
pub(crate) fn triangulate_face(
    arena: &mut Arena,
    verts: &[u32],
    orig: i32,
    edge_orig: &[i32],
) -> Vec<Tri> {
    debug_assert_eq!(verts.len(), edge_orig.len());
    match verts.len() {
        0..=2 => Vec::new(),
        3 => vec![Tri::new(
            [verts[0], verts[1], verts[2]],
            orig,
            [edge_orig[0], edge_orig[1], edge_orig[2]],
        )],
        _ => triangulate_poly(arena, verts, orig, edge_orig),
    }
}

/// Blender `triangulate_poly`: fast float triangulation, with the exact CDT fallback when the
/// result contains a degenerate triangle.
fn triangulate_poly(arena: &Arena, verts: &[u32], orig: i32, edge_orig: &[i32]) -> Vec<Tri> {
    let tris = polyfill_triangulate_poly(arena, verts, orig, edge_orig);
    if any_degenerate_tris_fast(arena, &tris) {
        return exact_triangulate_poly(arena, verts, orig, edge_orig);
    }
    tris
}

/// Blender `is_quad_flip_first_third`: whether the quad's first-third diagonal (`v0`-`v2`) is
/// invalid, i.e. the two triangles it creates have opposite orientation.
fn is_quad_flip_first_third(v1: DVec3, v2: DVec3, v3: DVec3, v4: DVec3) -> bool {
    let d12 = v2 - v1;
    let d13 = v3 - v1;
    let d14 = v4 - v1;
    let cross_a = d12.cross(d13);
    let cross_b = d14.cross(d13);
    cross_a.dot(cross_b) > 0.0
}

/// Blender `polyfill_triangulate_poly`: quads choose a diagonal directly; larger polygons are
/// projected along the negated normal and ear-clipped in float.
fn polyfill_triangulate_poly(
    arena: &Arena,
    verts: &[u32],
    orig: i32,
    edge_orig: &[i32],
) -> Vec<Tri> {
    if let (&[v0, v1, v2, v3], &[e01, e12, e23, e30]) = (verts, edge_orig) {
        return quad_triangulation(arena, [v0, v1, v2, v3], orig, [e01, e12, e23, e30]);
    }
    let flen = verts.len();
    let normal = poly_normal(arena, verts);
    let (basis0, basis1) = projection_axes(normal);
    let projected: Vec<[f32; 2]> = verts
        .iter()
        .map(|&v| project(arena.vert(v).co_f, basis0, basis1))
        .collect();
    polyfill_calc(&projected)
        .into_iter()
        .map(|tri| {
            let mut eo = [NO_INDEX; 3];
            for (k, slot) in eo.iter_mut().enumerate() {
                let from = tri[k];
                let to = tri[(k + 1) % 3];
                // A triangle edge between successive face corners is an input edge.
                if (from + 1) % flen == to {
                    *slot = edge_orig[from];
                }
            }
            Tri::new(tri.map(|i| verts[i]), orig, eo)
        })
        .collect()
}

/// Blender quad branch of `polyfill_triangulate_poly`.
fn quad_triangulation(arena: &Arena, verts: [u32; 4], orig: i32, eo: [i32; 4]) -> Vec<Tri> {
    let [v0, v1, v2, v3] = verts;
    let [eo01, eo12, eo23, eo30] = eo;
    let c = verts.map(|v| arena.vert(v).co_f);
    if is_quad_flip_first_third(c[0], c[1], c[2], c[3]) {
        vec![
            Tri::new([v0, v1, v3], orig, [eo01, NO_INDEX, eo30]),
            Tri::new([v1, v2, v3], orig, [eo12, eo23, NO_INDEX]),
        ]
    } else {
        vec![
            Tri::new([v0, v1, v2], orig, [eo01, eo12, NO_INDEX]),
            Tri::new([v0, v2, v3], orig, [NO_INDEX, eo23, eo30]),
        ]
    }
}

/// Blender `exact_triangulate_poly`: constrained Delaunay triangulation of the polygon projected
/// along its dominant normal axis, with exact coordinates. Falls back to the float path when the
/// CDT needs a Steiner point.
fn exact_triangulate_poly(arena: &Arena, verts: &[u32], orig: i32, edge_orig: &[i32]) -> Vec<Tri> {
    let flen = verts.len();
    let normal = poly_normal(arena, verts);
    let axis = dominant_axis(normal);
    // Dropping the dominant axis reverses orientation for y, and again for a negative normal.
    let rev = (axis == 1) ^ (normal[axis] < 0.0);
    let kept = match axis {
        0 => [1, 2],
        1 => [0, 2],
        _ => [0, 1],
    };
    let in_verts: Vec<Rat2> = verts
        .iter()
        .map(|&v| {
            let co = &arena.vert(v).co;
            [co[kept[0]].clone(), co[kept[1]].clone()]
        })
        .collect();
    let cdt = delaunay_2d_calc_inside(&CdtInput {
        verts: in_verts,
        edges: Vec::new(),
        faces: vec![(0..flen).map(index_u32).collect()],
    });
    // Blender `CDT_result::face_edge_offset`: face edge ids are `(face + 1) * offset + edge`.
    let foff = cdt.face_edge_offset;
    let edge_lookup: HashMap<(u32, u32), usize> = cdt
        .edges
        .iter()
        .enumerate()
        .map(|(e, &(a, b))| (edge_key(a, b), e))
        .collect();

    let mut tris = Vec::with_capacity(cdt.faces.len());
    for face in &cdt.faces {
        let mut local = [0usize; 3];
        for (slot, &cdt_vert) in local.iter_mut().zip(face) {
            let Some(&corner) = cdt
                .vert_orig
                .get(cdt_vert as usize)
                .and_then(|origins| origins.first())
            else {
                return polyfill_triangulate_poly(arena, verts, orig, edge_orig);
            };
            *slot = corner as usize;
        }
        let pairs = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])];
        let mut eo = [NO_INDEX; 3];
        for (slot, (from, to)) in eo.iter_mut().zip(pairs) {
            let Some(origins) = edge_lookup
                .get(&edge_key(from, to))
                .and_then(|&e| cdt.edge_orig.get(e))
            else {
                continue;
            };
            // The single input face here is face 0, so its edge `e` has id `foff + e`.
            for &origin in origins {
                if origin >= foff {
                    *slot = edge_orig[(origin % foff) as usize];
                    break;
                }
            }
        }
        let [v0, v1, v2] = local.map(|c| verts[c]);
        let [e0, e1, e2] = eo;
        tris.push(if rev {
            Tri::new([v0, v2, v1], orig, [e2, e1, e0])
        } else {
            Tri::new([v0, v1, v2], orig, [e0, e1, e2])
        });
    }
    tris
}

/// Blender `any_degenerate_tris_fast`: true when some triangle has coincident corners or a
/// near-zero sine of its corner angle. Errs toward true, as Blender does.
fn any_degenerate_tris_fast(arena: &Arena, tris: &[Tri]) -> bool {
    tris.iter().any(|tri| {
        let [a, b, c] = tri.v;
        if a == b || a == c || b == c {
            return true;
        }
        let v0 = arena.vert(a).co_f;
        let v1 = arena.vert(b).co_f;
        let v2 = arena.vert(c).co_f;
        let da = v2 - v0;
        let db = v2 - v1;
        let da_length_squared = da.length_squared();
        let db_length_squared = db.length_squared();
        if da_length_squared <= 0.0 || db_length_squared <= 0.0 {
            return true;
        }
        // sin^2(t) = |da x db|^2 / (|da|^2 |db|^2)
        let dab_length_squared = da.cross(db).length_squared();
        dab_length_squared / (da_length_squared * db_length_squared) < 1e-8
    })
}

/// Blender `Face::populate_plane(false)` normal of a polygon with at least four vertices:
/// Newell's method (`math::cross_poly`) over the double-precision corner positions.
fn poly_normal(arena: &Arena, verts: &[u32]) -> DVec3 {
    let Some(&last) = verts.last() else {
        return DVec3::ZERO;
    };
    let mut prev = arena.vert(last).co_f;
    let mut normal = DVec3::ZERO;
    for &v in verts {
        let cur = arena.vert(v).co_f;
        normal.x += (prev.y - cur.y) * (prev.z + cur.z);
        normal.y += (prev.z - cur.z) * (prev.x + cur.x);
        normal.z += (prev.x - cur.x) * (prev.y + cur.y);
        prev = cur;
    }
    normal
}

/// Blender `math::dominant_axis`: index of the largest-magnitude component.
fn dominant_axis(v: DVec3) -> usize {
    let b = v.abs();
    if b.x > b.y {
        if b.x > b.z { 0 } else { 2 }
    } else if b.y > b.z {
        1
    } else {
        2
    }
}

fn index_u32(index: usize) -> u32 {
    u32::try_from(index).unwrap_or(u32::MAX)
}

/// Blender `axis_dominant_v3_to_m3_negate` on the float-rounded, normalized polygon normal.
/// Returns the in-plane basis `(b0, b1)` with `b0 x b1 = -n`, so that projected coordinates
/// `(b0.p, b1.p)` are clockwise for a polygon whose normal is `n`.
fn projection_axes(normal: DVec3) -> ([f32; 3], [f32; 3]) {
    let mut n = [normal.x as f32, normal.y as f32, normal.z as f32];
    // Blender `normalize_v3`: vectors with a squared length at or below 1e-35 become zero.
    let length_squared = dot3(n, n);
    if length_squared > 1.0e-35 {
        let inv = 1.0 / length_squared.sqrt();
        n = n.map(|c| c * inv);
    } else {
        n = [0.0; 3];
    }
    ortho_basis(n.map(|c| -c))
}

/// Blender `ortho_basis_v3v3_v3`.
fn ortho_basis(n: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let f = n[0] * n[0] + n[1] * n[1];
    if f > f32::EPSILON {
        let d = 1.0 / f.sqrt();
        let b0 = [n[1] * d, -n[0] * d, 0.0];
        let b1 = [-n[2] * b0[1], n[2] * b0[0], n[0] * b0[1] - n[1] * b0[0]];
        (b0, b1)
    } else {
        let b0 = [if n[2] < 0.0 { -1.0 } else { 1.0 }, 0.0, 0.0];
        (b0, [0.0, 1.0, 0.0])
    }
}

/// Blender `mul_v2_m3v3` against the projection basis, in float.
fn project(co: DVec3, basis0: [f32; 3], basis1: [f32; 3]) -> [f32; 2] {
    let p = [co.x as f32, co.y as f32, co.z as f32];
    [dot3(basis0, p), dot3(basis1, p)]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub2(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

/// Blender `signum_enum`: sign of the doubled signed area, with NaN treated as concave.
fn signum_enum(area2: f32) -> i8 {
    if area2 > 0.0 {
        CONVEX
    } else if area2 >= 0.0 {
        TANGENTIAL
    } else {
        CONCAVE
    }
}

/// Blender `span_tri_v2_sign`: orientation class of the corner `v2` formed by `v1`, `v2`, `v3`.
fn span_tri_sign(v1: [f32; 2], v2: [f32; 2], v3: [f32; 2]) -> i8 {
    // Blender `area_tri_signed_v2_alt_2x(v3, v2, v1)`.
    let d2 = sub2(v2, v3);
    let d3 = sub2(v1, v3);
    signum_enum(d2[0] * d3[1] - d3[0] * d2[1])
}

/// Blender `pf_coord_sign_calc` for ring corner `cur`.
fn corner_sign(coords: &[[f32; 2]], prev: usize, cur: usize, next: usize) -> i8 {
    span_tri_sign(coords[prev], coords[cur], coords[next])
}

/// Blender `TriIsectPrecomputed`: inside-or-on test of a triangle with precomputed edge vectors.
struct EarTest {
    edge: [[f32; 2]; 3],
    c: [f32; 3],
}

impl EarTest {
    fn new(v: [[f32; 2]; 3]) -> Self {
        let edge = [sub2(v[1], v[0]), sub2(v[2], v[1]), sub2(v[0], v[2])];
        let c = [0, 1, 2].map(|i| edge[i][0] * v[i][1] - v[i][0] * edge[i][1]);
        Self { edge, c }
    }

    fn contains(&self, co: [f32; 2]) -> bool {
        // Edge 1 fails most often; check it first as Blender does.
        let side = |k: usize| self.edge[k][1] * co[0] - self.edge[k][0] * co[1] + self.c[k] >= 0.0;
        side(1) && side(2) && side(0)
    }
}

#[derive(Clone, Copy)]
struct Corner {
    prev: usize,
    next: usize,
    sign: i8,
}

/// Circular ear-clipping state of Blender's `PolyFill`. Corner `i` is coordinate `i`.
struct EarClipper<'a> {
    coords: &'a [[f32; 2]],
    corners: Vec<Corner>,
    head: usize,
    count: usize,
    /// Corners whose sign is not `CONVEX`.
    concave: usize,
    tris: Vec<[usize; 3]>,
}

/// Blender `BLI_polyfill_calc` for clockwise projected coordinates (`coords_sign == 1`). Returns
/// `n - 2` triangles as indices into `coords`, each keeping the ring winding.
fn polyfill_calc(coords: &[[f32; 2]]) -> Vec<[usize; 3]> {
    let n = coords.len();
    let mut corners: Vec<Corner> = (0..n)
        .map(|i| Corner {
            prev: (i + n - 1) % n,
            next: (i + 1) % n,
            sign: CONVEX,
        })
        .collect();
    let signs: Vec<i8> = corners
        .iter()
        .enumerate()
        .map(|(i, c)| corner_sign(coords, c.prev, i, c.next))
        .collect();
    for (corner, sign) in corners.iter_mut().zip(signs) {
        corner.sign = sign;
    }
    let concave = corners.iter().filter(|c| c.sign != CONVEX).count();
    let mut clipper = EarClipper {
        coords,
        corners,
        head: 0,
        count: n,
        concave,
        tris: Vec::with_capacity(n.saturating_sub(2)),
    };
    clipper.clip_all();
    clipper.tris
}

impl EarClipper<'_> {
    /// Blender `pf_coord_sign_calc` stored on the corner.
    fn recompute_sign(&mut self, i: usize) {
        let c = self.corners[i];
        self.corners[i].sign = corner_sign(self.coords, c.prev, i, c.next);
    }

    /// Blender `pf_ear_tip_check` with the KD-tree's predicate: `tip` is an ear of class
    /// `sign_accept` when no other non-convex corner lies inside or on its triangle.
    fn is_ear(&self, tip: usize, sign_accept: i8) -> bool {
        if self.concave == 0 {
            return true;
        }
        let c = self.corners[tip];
        if c.sign != sign_accept {
            return false;
        }
        let (next, prev) = (c.next, c.prev);
        let triangle = EarTest::new([self.coords[tip], self.coords[next], self.coords[prev]]);
        let mut checked = 0;
        let mut curr = self.corners[next].next;
        while curr != prev {
            if self.corners[curr].sign != CONVEX {
                if triangle.contains(self.coords[curr]) {
                    return false;
                }
                checked += 1;
                if checked == self.concave {
                    break;
                }
            }
            curr = self.corners[curr].next;
        }
        true
    }

    /// Blender `pf_ear_tip_find`: two passes for proper ears, then the desperate fallback.
    fn find_ear(&self, init: usize, reverse: bool) -> usize {
        for sign_accept in [CONVEX, TANGENTIAL] {
            let mut ear = init;
            for _ in 0..self.count {
                if self.is_ear(ear, sign_accept) {
                    return ear;
                }
                ear = self.step(ear, reverse);
            }
        }
        let mut ear = init;
        for _ in 0..self.count {
            if self.corners[ear].sign != CONCAVE {
                return ear;
            }
            ear = self.corners[ear].next;
        }
        ear
    }

    fn step(&self, i: usize, reverse: bool) -> usize {
        if reverse {
            self.corners[i].prev
        } else {
            self.corners[i].next
        }
    }

    /// Blender `pf_ear_tip_cut`: emit `(prev, ear, next)` and unlink `ear`.
    fn cut(&mut self, ear: usize) {
        let Corner { prev, next, .. } = self.corners[ear];
        self.tris.push([prev, ear, next]);
        self.corners[next].prev = prev;
        self.corners[prev].next = next;
        if self.head == ear {
            self.head = next;
        }
        self.count -= 1;
    }

    /// Blender `pf_triangulate`.
    fn clip_all(&mut self) {
        let mut ear_init = self.head;
        let mut reverse = false;
        while self.count > 3 {
            let ear = self.find_ear(ear_init, reverse);
            if self.corners[ear].sign != CONVEX {
                self.concave -= 1;
            }
            let Corner { prev, next, .. } = self.corners[ear];
            self.cut(ear);

            for neighbor in [prev, next] {
                if self.corners[neighbor].sign != CONVEX {
                    self.recompute_sign(neighbor);
                    if self.corners[neighbor].sign == CONVEX {
                        self.concave -= 1;
                    }
                }
            }

            // Clip-even sweep: start the next search past the ear just clipped, and step once more
            // when that corner is not convex. The direction alternates (Blender USE_CLIP_SWEEP).
            ear_init = if reverse {
                self.corners[prev].prev
            } else {
                self.corners[next].next
            };
            if self.corners[ear_init].sign != CONVEX {
                ear_init = self.step(ear_init, reverse);
                reverse = !reverse;
            }
        }
        if self.count == 3 {
            let a = self.head;
            let b = self.corners[a].next;
            let c = self.corners[b].next;
            self.tris.push([a, b, c]);
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::DVec3;

    use super::super::exact::{Rat, cross3, rat_from_f64, sgn, sub3};
    use super::*;

    fn arena_with(points: &[DVec3]) -> (Arena, Vec<u32>) {
        let mut arena = Arena::new();
        let ids = points
            .iter()
            .enumerate()
            .map(|(i, &p)| arena.add_or_find_dvec(p, index_i32(i)))
            .collect();
        (arena, ids)
    }

    fn index_i32(i: usize) -> i32 {
        i32::try_from(i).unwrap_or(-1)
    }

    /// Exact doubled area of a triangle in the xy plane.
    fn twice_area(arena: &Arena, tri: &Tri) -> Rat {
        let [a, b, c] = tri.v.map(|v| arena.vert(v).co.clone());
        cross3(&sub3(&b, &a), &sub3(&c, &a))[2].clone()
    }

    fn total_twice_area(arena: &Arena, tris: &[Tri]) -> Rat {
        tris.iter()
            .map(|tri| twice_area(arena, tri))
            .fold(rat_from_f64(0.0), |sum, area| sum + area)
    }

    #[test]
    fn convex_quad_uses_diagonal_zero_two_when_not_flipped() {
        let (mut arena, ids) = arena_with(&[
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(1.0, 1.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
        ]);
        // d12 x d13 = +z and d14 x d13 = -z, so the dot product is negative: no flip.
        let tris = triangulate_face(&mut arena, &ids, 7, &[10, 11, 12, 13]);
        assert_eq!(tris.len(), 2);
        assert_eq!(
            tris[0],
            Tri::new([ids[0], ids[1], ids[2]], 7, [10, 11, NO_INDEX])
        );
        assert_eq!(
            tris[1],
            Tri::new([ids[0], ids[2], ids[3]], 7, [NO_INDEX, 12, 13])
        );
        assert_eq!(total_twice_area(&arena, &tris), rat_from_f64(2.0));
    }

    #[test]
    fn dart_quad_flips_to_diagonal_one_three() {
        // Reflex corner at (1, 1): diagonal 0-2 lies outside, so the rule flips to 1-3.
        let (mut arena, ids) = arena_with(&[
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 1.0, 0.0),
            DVec3::new(2.0, 0.0, 0.0),
            DVec3::new(1.0, 3.0, 0.0),
        ]);
        // d12 x d13 = -2z and d14 x d13 = -6z: positive dot product, flip.
        let tris = triangulate_face(&mut arena, &ids, 2, &[20, 21, 22, 23]);
        assert_eq!(
            tris,
            vec![
                Tri::new([ids[0], ids[1], ids[3]], 2, [20, NO_INDEX, 23]),
                Tri::new([ids[1], ids[2], ids[3]], 2, [21, 22, NO_INDEX]),
            ]
        );
        assert_eq!(total_twice_area(&arena, &tris), rat_from_f64(4.0));
    }

    #[test]
    fn concave_pentagon_gives_three_triangles_with_exact_area() {
        let (mut arena, ids) = arena_with(&[
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(4.0, 0.0, 0.0),
            DVec3::new(4.0, 4.0, 0.0),
            DVec3::new(2.0, 1.0, 0.0),
            DVec3::new(0.0, 4.0, 0.0),
        ]);
        let tris = triangulate_face(&mut arena, &ids, 0, &[0, 1, 2, 3, 4]);
        assert_eq!(tris.len(), 3);
        // Shoelace area of the pentagon is 10, so doubled it is 20.
        assert_eq!(total_twice_area(&arena, &tris), rat_from_f64(20.0));
        for tri in &tris {
            assert_eq!(sgn(&twice_area(&arena, tri)), 1);
        }
    }

    #[test]
    fn collinear_quad_falls_back_to_exact_without_zero_area_triangles() {
        // Corners 0, 1, 2 are collinear, so the fast triangulation yields a zero-area triangle.
        let (mut arena, ids) = arena_with(&[
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(2.0, 0.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
        ]);
        let tris = triangulate_face(&mut arena, &ids, 5, &[0, 1, 2, 3]);
        assert_eq!(tris.len(), 2);
        for tri in &tris {
            assert_eq!(sgn(&twice_area(&arena, tri)), 1);
            assert_eq!(tri.orig, 5);
        }
        assert_eq!(total_twice_area(&arena, &tris), rat_from_f64(2.0));
    }

    #[test]
    fn triangle_passes_through_with_its_edges() {
        let (mut arena, ids) = arena_with(&[
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
        ]);
        let tris = triangulate_face(&mut arena, &ids, 9, &[4, NO_INDEX, 6]);
        assert_eq!(
            tris,
            vec![Tri::new([ids[0], ids[1], ids[2]], 9, [4, NO_INDEX, 6])]
        );
    }
}
