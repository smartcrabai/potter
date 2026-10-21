//! Float-ray fallback for non-PWN exact-boolean intersection meshes (Blender `raycast.rs`).
//!
//! Each test point casts Blender's six rays. A hit is decided by Blender's float test
//! (`isect_ray_tri_epsilon_v3`), and each hit counts the orientation of the ray origin against the
//! hit triangle (Blender `orient3d`, decided exactly by `robust` on the double vertex positions).
//! Candidate triangles come from one [`KdopTree`] per classification (`geom::bvh`, Blender's k-DOP
//! tree with the watertight leaf walk of `BLI_bvhtree_ray_cast`), plus a per-ray list of triangles
//! whose float test is too ill-conditioned to trust box culling.

use glam::DVec3;
use robust::{Coord3D, orient3d};

use super::super::imesh::{Arena, Tri};
use super::{BoolOp, apply_bool_op, patches::find_patches, topology::Topology};
use crate::geom::bvh::{KdopTree, Ray, RayHit};

/// Blender `FLT_EPSILON`: slack on the barycentric bounds of `isect_ray_tri_epsilon_v3`.
const RAY_EPSILON: f32 = f32::EPSILON;

/// Blender's `r1`, `ra`, `rb` in `test_tri_inside_shapes`, as the `f32` values of the source.
const R1: f32 = 0.998_702_5;
const RA: f32 = 0.049_935_125;
const RB: f32 = 0.009_987_025_5;
const RAYS: [[f32; 3]; 6] = [
    [R1, RA, RB],
    [-R1, -RA, -RB],
    [RB, R1, RA],
    [-RB, -R1, -RA],
    [RA, RB, R1],
    [-RA, -RB, -R1],
];

/// Box padding of the BVH, as a fraction of the largest extent `L` of the triangles.
///
/// The float test's hit point is off the exact one by about `k * f32::EPSILON * |origin - v0| * c`,
/// where `c` is the triangle's condition (see [`is_ill_conditioned`]) and `k` counts rounding
/// steps. With `c <= MAX_CONDITION` and `|origin - v0| <= 4 L`, that is at most `k * 4.9e-4 * L`,
/// so this padding covers every `k <= 20`.
const BOX_PAD_FRACTION: f64 = 1.0e-2;

/// Largest condition of a triangle for which the BVH is trusted to cull it for a ray.
const MAX_CONDITION: f64 = 1024.0;

/// Blender's non-PWN classification: float ray/triangle hits with robust orientation parity.
pub(super) fn raycast_boolean(
    arena: &mut Arena,
    tris: &[Tri],
    nshapes: usize,
    shape_fn: &dyn Fn(i32) -> usize,
    op: BoolOp,
    hole_tolerant: bool,
) -> Vec<Tri> {
    let float_tris: Vec<[[f32; 3]; 3]> = tris
        .iter()
        .map(|tri| {
            tri.v.map(|vertex| {
                let co = arena.vert(vertex).co_f;
                [co.x as f32, co.y as f32, co.z as f32]
            })
        })
        .collect();
    let index = RayIndex::new(float_tris);

    if hole_tolerant {
        raycast_tris_boolean(arena, tris, &index, nshapes, shape_fn, op)
    } else {
        raycast_patches_boolean(arena, tris, &index, nshapes, shape_fn, op)
    }
}

/// The float triangles of one classification, with the BVH and conditioning data for the six rays.
struct RayIndex {
    float_tris: Vec<[[f32; 3]; 3]>,
    bvh: KdopTree,
    /// Bit `r` is set when triangle `t` is ill-conditioned for `RAYS[r]`.
    ill_conditioned: Vec<u8>,
    /// Triangles that are ill-conditioned for `RAYS[r]`. They are tested for that ray directly,
    /// because box culling cannot bound their float test.
    always_tested: Vec<Vec<usize>>,
}

impl RayIndex {
    /// Builds the BVH over the padded triangle boxes and classifies every triangle for every ray.
    ///
    /// Cost: O(T log T) for the BVH, O(6 T) for the conditioning test.
    fn new(float_tris: Vec<[[f32; 3]; 3]>) -> Self {
        // Padding is BOX_PAD_FRACTION of the extent, plus one f32 ulp per unit of the largest
        // coordinate: the leaf boxes are stored as f32, and that guard keeps their corners outside
        // the triangle however far the mesh sits from the origin. A wider box only adds candidates,
        // and every candidate still goes through the exact float test.
        let (extent, magnitude) = extent_and_magnitude(&float_tris);
        let pad = BOX_PAD_FRACTION * extent + magnitude * f64::from(f32::EPSILON);
        let leaf_bounds: Vec<[f32; 6]> = float_tris
            .iter()
            .map(|tri| padded_bounds(tri, pad))
            .collect();
        let bvh = KdopTree::from_leaf_bounds(&leaf_bounds);

        let mut ill_conditioned = vec![0_u8; float_tris.len()];
        let mut always_tested = vec![Vec::new(); RAYS.len()];
        for (ray, &direction) in RAYS.iter().enumerate() {
            let direction = to_dvec3(direction);
            for (t, tri) in float_tris.iter().enumerate() {
                if is_ill_conditioned(tri, direction) {
                    ill_conditioned[t] |= 1 << ray;
                    always_tested[ray].push(t);
                }
            }
        }
        Self {
            float_tris,
            bvh,
            ill_conditioned,
            always_tested,
        }
    }
}

fn to_dvec3(v: [f32; 3]) -> DVec3 {
    DVec3::new(f64::from(v[0]), f64::from(v[1]), f64::from(v[2]))
}

/// The largest extent `L` of the triangles on any axis and the largest coordinate magnitude, both
/// in f64; zero for no triangles.
fn extent_and_magnitude(float_tris: &[[[f32; 3]; 3]]) -> (f64, f64) {
    if float_tris.is_empty() {
        return (0.0, 0.0);
    }
    let (lo, hi) = float_tris.iter().flatten().fold(
        (DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY)),
        |(lo, hi), &vertex| {
            let v = to_dvec3(vertex);
            (lo.min(v), hi.max(v))
        },
    );
    let magnitude = lo.abs().max(hi.abs()).max_element();
    ((hi - lo).max_element().max(0.0), magnitude)
}

/// The leaf box of a triangle, grown by `pad` on every side, in the `[min, max]` per-axis layout
/// of [`KdopTree::from_leaf_bounds`]. The box is computed in f64 and rounded to f32 once per corner.
fn padded_bounds(tri: &[[f32; 3]; 3], pad: f64) -> [f32; 6] {
    let mut bounds = [0.0_f32; 6];
    for axis in 0..3 {
        let min = tri
            .iter()
            .map(|v| f64::from(v[axis]))
            .fold(f64::INFINITY, f64::min);
        let max = tri
            .iter()
            .map(|v| f64::from(v[axis]))
            .fold(f64::NEG_INFINITY, f64::max);
        bounds[2 * axis] = (min - pad) as f32;
        bounds[2 * axis + 1] = (max + pad) as f32;
    }
    bounds
}

/// Blender `BLI_bvhtree_ray_cast`'s leaf walk over the classification's tree: appends to `leaves`
/// every triangle whose padded box the ray `origin + s * direction`, `s >= 0`, crosses. The walk is
/// unbounded (`RayHit::open`), as Blender's `BLI_bvhtree_ray_cast_all` is, and the callback only
/// records leaves, so no leaf is pruned by a hit found earlier.
fn bvh_leaves(index: &RayIndex, origin: [f32; 3], direction: [f32; 3], leaves: &mut Vec<usize>) {
    let mut hit = RayHit::open();
    index.bvh.ray_cast(
        origin,
        direction,
        &mut hit,
        &mut |leaf: usize, _ray: &Ray, _hit: &mut RayHit| leaves.push(leaf),
    );
}

/// Whether the float test of Blender's `isect_ray_tri_epsilon_v3` for `tri` and `direction` is too
/// ill-conditioned for box culling: `|a| * MAX_CONDITION <= |d| |e1| |e2|`, with
/// `a = dot(e1, d x e2)` the pivot of the test. Degenerate triangles (`a = 0`) are included.
///
/// Computed in f64 from the f32 vertices, so it is exact up to the rounding of the differences.
fn is_ill_conditioned(tri: &[[f32; 3]; 3], direction: DVec3) -> bool {
    let [v0, v1, v2] = tri.map(to_dvec3);
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let pivot = direction.dot(e2.cross(e1)).abs();
    pivot * MAX_CONDITION <= direction.length() * e1.length() * e2.length()
}

fn raycast_tris_boolean(
    arena: &Arena,
    tris: &[Tri],
    index: &RayIndex,
    nshapes: usize,
    shape_fn: &dyn Fn(i32) -> usize,
    op: BoolOp,
) -> Vec<Tri> {
    let mut output = Vec::with_capacity(tris.len());
    let mut inside = vec![0.0; nshapes];
    let mut winding = vec![0; nshapes];
    let mut scratch = RayScratch::new(nshapes);

    for (test_index, tri) in tris.iter().enumerate() {
        let shape = shape_fn(tri.orig);
        test_tri_inside_shapes(
            arena,
            tris,
            index,
            shape_fn,
            test_index,
            &mut scratch,
            &mut inside,
        );
        if let Some(do_flip) = raycast_test_remove(op, shape, &inside, &mut winding) {
            output.push(if do_flip { flipped(*tri) } else { *tri });
        }
    }
    output
}

fn raycast_patches_boolean(
    arena: &Arena,
    tris: &[Tri],
    index: &RayIndex,
    nshapes: usize,
    shape_fn: &dyn Fn(i32) -> usize,
    op: BoolOp,
) -> Vec<Tri> {
    let topo = Topology::new(tris);
    let pinfo = find_patches(tris, &topo);
    let mut output = Vec::with_capacity(tris.len());
    let mut inside = vec![0.0; nshapes];
    let mut winding = vec![0; nshapes];
    let mut scratch = RayScratch::new(nshapes);

    for patch in &pinfo.patches {
        let test_index = patch.tris[patch.tris.len() / 2];
        let test_tri = tris[test_index];
        let shape = shape_fn(test_tri.orig);
        test_tri_inside_shapes(
            arena,
            tris,
            index,
            shape_fn,
            test_index,
            &mut scratch,
            &mut inside,
        );
        if let Some(do_flip) = raycast_test_remove(op, shape, &inside, &mut winding) {
            output.extend(patch.tris.iter().map(|&index| {
                if do_flip {
                    flipped(tris[index])
                } else {
                    tris[index]
                }
            }));
        }
    }
    output
}

/// `Some(do_flip)` retains the triangle/patch; `None` removes it.
fn raycast_test_remove(
    op: BoolOp,
    shape: usize,
    inside: &[f32],
    winding: &mut [i32],
) -> Option<bool> {
    winding.fill(0);
    let need_high_confidence = op == BoolOp::Intersect || (op == BoolOp::Difference && shape != 0);
    let threshold = if need_high_confidence { 0.5 } else { 0.1 };
    for other_shape in 0..winding.len() {
        if other_shape != shape {
            winding[other_shape] = i32::from(inside[other_shape] >= threshold);
        }
    }
    winding[shape] = 0;
    let in_output_volume_0 = apply_bool_op(op, winding);
    winding[shape] = 1;
    let in_output_volume_1 = apply_bool_op(op, winding);
    if in_output_volume_0 == in_output_volume_1 {
        None
    } else {
        Some(op == BoolOp::Difference && shape != 0)
    }
}

/// Reused buffers of one classification: the BVH leaf list of the current ray, and the per-shape
/// orientation parity of the current ray.
struct RayScratch {
    hits: Vec<usize>,
    hit_parity: Vec<i32>,
}

impl RayScratch {
    fn new(nshapes: usize) -> Self {
        Self {
            hits: Vec::new(),
            hit_parity: vec![0; nshapes],
        }
    }
}

/// One test ray: the `f32` origin and direction of Blender's `inside_shape_callback`, and the
/// origin as an exact `f64` value.
struct TestRay {
    origin: [f32; 3],
    origin_exact: DVec3,
    direction: [f32; 3],
}

/// Blender `test_tri_inside_shapes`: sets `inside[j]` to the fraction of the six rays from the
/// offset interior point of triangle `test_index` that count as inside shape `j`. The test
/// triangle's own shape is 1.
///
/// Cost per call: six BVH queries, each O(log T + k) for k returned candidates, then one float
/// test per candidate and per triangle that is ill-conditioned for that ray. The second term is
/// g, the ill-conditioned count of the ray: near zero for well-shaped meshes, but up to T for a
/// mesh of degenerate faces.
fn test_tri_inside_shapes(
    arena: &Arena,
    tris: &[Tri],
    index: &RayIndex,
    shape_fn: &dyn Fn(i32) -> usize,
    test_index: usize,
    scratch: &mut RayScratch,
    inside: &mut [f32],
) {
    let test_tri = tris[test_index];
    let shape = shape_fn(test_tri.orig);
    let co = test_tri.v.map(|vertex| arena.vert(vertex).co_f);
    let test_point = co[0] / 3.0 + co[1] / 3.0 + co[2] / 3.0;
    let normal = (co[0] - co[2]).cross(co[1] - co[2]);
    let length_squared = normal.x * normal.x + normal.y * normal.y + normal.z * normal.z;
    let normal = if length_squared > 1.0e-70 {
        normal / length_squared.sqrt()
    } else {
        DVec3::ZERO
    };
    let offset_test_point = test_point + normal * 1.0e-5;
    let origin = [
        offset_test_point.x as f32,
        offset_test_point.y as f32,
        offset_test_point.z as f32,
    ];
    let origin_exact = to_dvec3(origin);
    inside.fill(0.0);

    for (ray_index, &direction) in RAYS.iter().enumerate() {
        let ray = TestRay {
            origin,
            origin_exact,
            direction,
        };
        scratch.hit_parity.fill(0);
        scratch.hits.clear();
        bvh_leaves(index, origin, direction, &mut scratch.hits);
        let ray_bit = 1_u8 << ray_index;
        let bvh_candidates = scratch
            .hits
            .iter()
            .filter(|&&t| (index.ill_conditioned[t] & ray_bit) == 0);
        for &t in bvh_candidates.chain(index.always_tested[ray_index].iter()) {
            if let Some(parity) = ray_parity(arena, tris, index, t, &ray) {
                scratch.hit_parity[shape_fn(tris[t].orig)] += parity;
            }
        }
        for (other_shape, (&parity, confidence)) in
            scratch.hit_parity.iter().zip(inside.iter_mut()).enumerate()
        {
            if other_shape != shape && parity > 0 {
                *confidence += 1.0;
            }
        }
    }

    for (other_shape, confidence) in inside.iter_mut().enumerate() {
        if other_shape == shape {
            *confidence = 1.0;
        } else {
            *confidence /= RAYS.len() as f32;
        }
    }
}

/// Blender `inside_shape_callback`: the orientation parity of triangle `t` if the test ray hits it
/// in the float test. Cost: O(1).
fn ray_parity(
    arena: &Arena,
    tris: &[Tri],
    index: &RayIndex,
    t: usize,
    ray: &TestRay,
) -> Option<i32> {
    if !ray_intersects_triangle(ray.origin, ray.direction, &index.float_tris[t]) {
        return None;
    }
    let [v0, v1, v2] = tris[t].v.map(|vertex| arena.vert(vertex).co_f);
    Some(orient3d_sign(v0, v1, v2, ray.origin_exact))
}

/// Blender's `isect_ray_tri_epsilon_v3`, kept in f32 and in source operation order.
fn ray_intersects_triangle(
    origin: [f32; 3],
    direction: [f32; 3],
    [v0, v1, v2]: &[[f32; 3]; 3],
) -> bool {
    let e1 = subf(v1, v0);
    let e2 = subf(v2, v0);
    let p = crossf(&direction, &e2);
    let a = dotf(&e1, &p);
    if a == 0.0 {
        return false;
    }
    let f = 1.0 / a;
    let s = subf(&origin, v0);
    let u = f * dotf(&s, &p);
    if u < -RAY_EPSILON || u > 1.0 + RAY_EPSILON {
        return false;
    }
    let q = crossf(&s, &e1);
    let v = f * dotf(&direction, &q);
    if v < -RAY_EPSILON || u + v > 1.0 + RAY_EPSILON {
        return false;
    }
    f * dotf(&e2, &q) >= 0.0
}

fn subf(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn crossf(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dotf(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Blender `orient3d`: `+1` when `d` lies below the plane of `a, b, c` (counterclockwise seen from
/// above), `-1` above, `0` on it. The sign is exact for the given double coordinates.
fn orient3d_sign(a: DVec3, b: DVec3, c: DVec3, d: DVec3) -> i32 {
    let det = orient3d(coord(a), coord(b), coord(c), coord(d));
    if det > 0.0 {
        1
    } else if det < 0.0 {
        -1
    } else {
        0
    }
}

fn coord(p: DVec3) -> Coord3D<f64> {
    Coord3D {
        x: p.x,
        y: p.y,
        z: p.z,
    }
}

fn flipped(tri: Tri) -> Tri {
    Tri {
        v: [tri.v[0], tri.v[2], tri.v[1]],
        orig: tri.orig,
        edge_orig: [tri.edge_orig[2], tri.edge_orig[1], tri.edge_orig[0]],
        is_intersect: [
            tri.is_intersect[2],
            tri.is_intersect[1],
            tri.is_intersect[0],
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::{
        RAYS, RayIndex, bvh_leaves, is_ill_conditioned, orient3d_sign, ray_intersects_triangle,
    };
    use glam::DVec3;

    #[test]
    fn float_ray_triangle_test_is_forward_and_epsilon_inclusive() {
        let triangle = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert!(ray_intersects_triangle(
            [0.25, 0.25, 1.0],
            [0.0, 0.0, -1.0],
            &triangle
        ));
        assert!(!ray_intersects_triangle(
            [0.25, 0.25, 1.0],
            [0.0, 0.0, 1.0],
            &triangle
        ));
        assert!(ray_intersects_triangle(
            [-f32::EPSILON / 2.0, 0.25, 1.0],
            [0.0, 0.0, -1.0],
            &triangle,
        ));
        assert!(!ray_intersects_triangle(
            [1.01, 1.01, 1.0],
            [0.0, 0.0, -1.0],
            &triangle
        ));
    }

    #[test]
    fn orient_sign_tracks_triangle_normal_direction() {
        let a = DVec3::new(0.0, 0.0, 0.0);
        let b = DVec3::new(1.0, 0.0, 0.0);
        let c = DVec3::new(0.0, 1.0, 0.0);
        assert_eq!(orient3d_sign(a, b, c, DVec3::new(0.0, 0.0, -1.0)), 1);
        assert_eq!(orient3d_sign(a, b, c, DVec3::new(0.0, 0.0, 1.0)), -1);
        assert_eq!(orient3d_sign(a, b, c, DVec3::new(0.25, 0.25, 0.0)), 0);
    }

    #[test]
    fn well_shaped_triangles_are_not_ill_conditioned_and_degenerate_ones_are() {
        let direction = DVec3::new(0.998_702_5, 0.049_935_1, 0.009_987_0);
        let good = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let flat = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        assert!(!is_ill_conditioned(&good, direction));
        assert!(is_ill_conditioned(&flat, direction));
    }

    /// The tree walk visits every well-conditioned triangle that the float test hits, the property
    /// the candidate culling relies on. Ill-conditioned triangles are tested directly instead.
    #[test]
    fn tree_walk_visits_every_triangle_the_float_test_hits() {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1_u64 << 24) as f32
        };
        let tris: Vec<[[f32; 3]; 3]> = (0..400)
            .map(|_| {
                std::array::from_fn(|_| {
                    [next() * 4.0 - 2.0, next() * 4.0 - 2.0, next() * 4.0 - 2.0]
                })
            })
            .collect();
        let index = RayIndex::new(tris);
        let mut leaves = Vec::new();
        let mut hits_checked = 0_usize;
        for _ in 0..200 {
            let origin = [next() * 4.0 - 2.0, next() * 4.0 - 2.0, next() * 4.0 - 2.0];
            for (ray, &direction) in RAYS.iter().enumerate() {
                leaves.clear();
                bvh_leaves(&index, origin, direction, &mut leaves);
                for (t, tri) in index.float_tris.iter().enumerate() {
                    let conditioned = (index.ill_conditioned[t] & (1_u8 << ray)) == 0;
                    if conditioned && ray_intersects_triangle(origin, direction, tri) {
                        hits_checked += 1;
                        assert!(
                            leaves.contains(&t),
                            "triangle {t} is hit by ray {ray} but was not visited"
                        );
                    }
                }
            }
        }
        assert!(
            hits_checked > 1000,
            "the fixture must produce many float hits"
        );
    }
}
