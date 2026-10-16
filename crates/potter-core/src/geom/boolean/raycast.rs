//! Float-ray fallback for non-PWN exact-boolean intersection meshes (Blender `raycast.rs`).

use glam::DVec3;

use super::super::{
    exact::{Rat3, cross3, dot3, rat3_from_dvec3, sgn, sub3},
    imesh::{Arena, Tri},
};
use super::{BoolOp, apply_bool_op, patches::find_patches, topology::Topology};

const RAY_EPSILON: f32 = f32::EPSILON;
const R1: f32 = 0.998_702_5;
const RA: f32 = 0.049_935_13;
const RB: f32 = 0.009_987_025_5;
const RAYS: [[f32; 3]; 6] = [
    [R1, RA, RB],
    [-R1, -RA, -RB],
    [RB, R1, RA],
    [-RB, -R1, -RA],
    [RA, RB, R1],
    [-RA, -RB, -R1],
];

/// Blender's non-PWN classification: float ray/triangle hits with exact orientation parity.
pub(super) fn raycast_boolean(
    arena: &mut Arena,
    tris: &[Tri],
    nshapes: usize,
    shape_fn: &dyn Fn(i32) -> usize,
    op: BoolOp,
    hole_tolerant: bool,
) -> Vec<Tri> {
    // shortcut: linear scan per test ray; add a BVH only if open-mesh workloads require it.
    let float_tris: Vec<[[f32; 3]; 3]> = tris
        .iter()
        .map(|tri| {
            tri.v.map(|vertex| {
                let co = arena.vert(vertex).co_f;
                [co.x as f32, co.y as f32, co.z as f32]
            })
        })
        .collect();

    if hole_tolerant {
        raycast_tris_boolean(arena, tris, &float_tris, nshapes, shape_fn, op)
    } else {
        raycast_patches_boolean(arena, tris, &float_tris, nshapes, shape_fn, op)
    }
}

fn raycast_tris_boolean(
    arena: &Arena,
    tris: &[Tri],
    float_tris: &[[[f32; 3]; 3]],
    nshapes: usize,
    shape_fn: &dyn Fn(i32) -> usize,
    op: BoolOp,
) -> Vec<Tri> {
    let mut output = Vec::with_capacity(tris.len());
    let mut inside = vec![0.0; nshapes];
    let mut hit_parity = vec![0; nshapes];
    let mut winding = vec![0; nshapes];

    for (test_index, tri) in tris.iter().enumerate() {
        let shape = shape_fn(tri.orig);
        test_tri_inside_shapes(
            arena,
            tris,
            float_tris,
            shape_fn,
            test_index,
            &mut inside,
            &mut hit_parity,
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
    float_tris: &[[[f32; 3]; 3]],
    nshapes: usize,
    shape_fn: &dyn Fn(i32) -> usize,
    op: BoolOp,
) -> Vec<Tri> {
    let topo = Topology::new(tris);
    let pinfo = find_patches(tris, &topo);
    let mut output = Vec::with_capacity(tris.len());
    let mut inside = vec![0.0; nshapes];
    let mut hit_parity = vec![0; nshapes];
    let mut winding = vec![0; nshapes];

    for patch in &pinfo.patches {
        let test_index = patch.tris[patch.tris.len() / 2];
        let test_tri = tris[test_index];
        let shape = shape_fn(test_tri.orig);
        test_tri_inside_shapes(
            arena,
            tris,
            float_tris,
            shape_fn,
            test_index,
            &mut inside,
            &mut hit_parity,
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

fn test_tri_inside_shapes(
    arena: &Arena,
    tris: &[Tri],
    float_tris: &[[[f32; 3]; 3]],
    shape_fn: &dyn Fn(i32) -> usize,
    test_index: usize,
    inside: &mut [f32],
    hit_parity: &mut [i32],
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
    let exact_origin = rat3_from_dvec3(DVec3::new(
        f64::from(origin[0]),
        f64::from(origin[1]),
        f64::from(origin[2]),
    ));
    inside.fill(0.0);

    for direction in RAYS {
        hit_parity.fill(0);
        for (index, tri) in tris.iter().enumerate() {
            let other_shape = shape_fn(tri.orig);
            if ray_intersects_triangle(origin, direction, &float_tris[index]) {
                hit_parity[other_shape] += orient3d_sign(
                    &arena.vert(tri.v[0]).co,
                    &arena.vert(tri.v[1]).co,
                    &arena.vert(tri.v[2]).co,
                    &exact_origin,
                );
            }
        }
        for (other_shape, (&parity, confidence)) in
            hit_parity.iter().zip(inside.iter_mut()).enumerate()
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

/// Blender `orient3d`: positive when the ray origin is below the oriented triangle plane.
fn orient3d_sign(a: &Rat3, b: &Rat3, c: &Rat3, d: &Rat3) -> i32 {
    i32::from(sgn(&dot3(&sub3(a, d), &cross3(&sub3(b, d), &sub3(c, d)))))
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
    use super::{orient3d_sign, ray_intersects_triangle};
    use crate::geom::boolean::exact::{Rat3, rat3_from_dvec3};
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
    fn exact_orient_parity_tracks_triangle_normal_direction() {
        let a: Rat3 = rat3_from_dvec3(DVec3::new(0.0, 0.0, 0.0));
        let b: Rat3 = rat3_from_dvec3(DVec3::new(1.0, 0.0, 0.0));
        let c: Rat3 = rat3_from_dvec3(DVec3::new(0.0, 1.0, 0.0));
        let below: Rat3 = rat3_from_dvec3(DVec3::new(0.0, 0.0, -1.0));
        let above: Rat3 = rat3_from_dvec3(DVec3::new(0.0, 0.0, 1.0));
        assert_eq!(orient3d_sign(&a, &b, &c, &below), 1);
        assert_eq!(orient3d_sign(&a, &b, &c, &above), -1);
    }
}
