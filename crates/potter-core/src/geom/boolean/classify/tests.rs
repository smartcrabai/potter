//! Exact checks of the PWN classification against hand-derived volumes of cube booleans.
//!
//! Operand A is the cube `[-1, 1]^3` (faces 0..6). Operand B (faces 6..12) is either the offset
//! box `[0.5, 2.5] x [-0.5, 1.5] x [0, 2]` or the nested cube `[-0.5, 0.5]^3`. Volumes come from
//! the exact signed volume of the output triangles, so they are checked without tolerance.

use std::collections::HashMap;

use glam::{DMat4, DVec3};
use num_traits::Zero;

use super::{BoolOp, apply_bool_op, boolean_tris};
use crate::geom::boolean::{
    exact::{Rat, Rat3, cross3, dot3, rat_from_f64, sub3},
    imesh::{Arena, Tri},
    input::{Operand, meshes_to_tris},
};
use crate::geom::{BoxParams, Mesh};

type TestResult = Result<(), String>;

/// Number of faces of each box operand; operand B's faces start at this index.
const BOX_FACES: i32 = 6;

fn box_mesh(size: DVec3) -> Result<Mesh, String> {
    Mesh::box_mesh(BoxParams { size }).map_err(|error| format!("{error:?}"))
}

/// Intersects operand A (`a_size` cube at the origin) with operand B (`b_size` box moved by
/// `b_offset`) and classifies the result. When `open_b`, B's top faces at `top_z` are removed.
fn run(
    op: BoolOp,
    a_size: f64,
    b_size: DVec3,
    b_offset: DVec3,
    open_b: Option<f64>,
) -> Result<(Arena, Vec<Tri>), String> {
    let a = box_mesh(DVec3::splat(a_size))?;
    let b = box_mesh(b_size)?;
    let operands = [
        Operand {
            mesh: &a,
            transform: DMat4::IDENTITY,
        },
        Operand {
            mesh: &b,
            transform: DMat4::from_translation(b_offset),
        },
    ];
    let mut arena = Arena::new();
    let mut inputs = meshes_to_tris(&operands, &mut arena).map_err(|e| format!("{e:?}"))?;
    if let Some(top_z) = open_b {
        let top = rat_from_f64(top_z);
        inputs
            .retain(|t| !(t.orig >= BOX_FACES && t.v.iter().all(|&v| arena.vert(v).co[2] == top)));
    }
    let out = boolean_tris(
        &mut arena,
        &inputs,
        2,
        &|orig| usize::from(orig >= BOX_FACES),
        op,
        false,
        false,
    )
    .map_err(|e| format!("{e:?}"))?;
    Ok((arena, out))
}

/// Exact signed volume: sum of `dot(v0, cross(v1, v2)) / 6` over the triangles.
fn signed_volume(arena: &Arena, tris: &[Tri]) -> Rat {
    let mut sum = Rat::zero();
    for tri in tris {
        let [a, b, c] = tri.v.map(|v| &arena.vert(v).co);
        sum += dot3(a, &cross3(b, c));
    }
    sum / rat_from_f64(6.0)
}

/// Twice the vector area of a triangle, exact.
fn double_area(arena: &Arena, tri: &Tri) -> Rat3 {
    let [a, b, c] = tri.v.map(|v| &arena.vert(v).co);
    cross3(&sub3(b, a), &sub3(c, a))
}

/// Every directed edge occurs once and its reverse occurs once: a closed, consistently oriented
/// surface.
fn assert_closed_and_oriented(tris: &[Tri]) -> TestResult {
    let mut directed: HashMap<(u32, u32), usize> = HashMap::new();
    for tri in tris {
        for k in 0..3 {
            *directed.entry((tri.v[k], tri.v[(k + 1) % 3])).or_default() += 1;
        }
    }
    for (&(a, b), &count) in &directed {
        if count != 1 || directed.get(&(b, a)) != Some(&1) {
            return Err(format!(
                "edge {a}->{b} is not matched by exactly one reverse edge"
            ));
        }
    }
    Ok(())
}

/// Sum of vector areas of all triangles of a closed surface is zero.
fn assert_vector_area_zero(arena: &Arena, tris: &[Tri]) -> TestResult {
    let mut sum: Rat3 = [Rat::zero(), Rat::zero(), Rat::zero()];
    for tri in tris {
        let area = double_area(arena, tri);
        for axis in 0..3 {
            sum[axis] += &area[axis];
        }
    }
    if sum.iter().all(Zero::is_zero) {
        Ok(())
    } else {
        Err("vector area of a closed surface is not zero".to_owned())
    }
}

fn expect_volume(actual: &Rat, expected: f64) -> TestResult {
    let want = rat_from_f64(expected);
    if *actual == want {
        Ok(())
    } else {
        Err(format!("exact volume {actual} != {expected}"))
    }
}

fn offset_b() -> (DVec3, DVec3) {
    (DVec3::splat(2.0), DVec3::new(1.5, 0.5, 1.0))
}

#[test]
fn offset_cube_difference_is_exact_7_25() -> TestResult {
    let (b_size, b_offset) = offset_b();
    let (arena, out) = run(BoolOp::Difference, 2.0, b_size, b_offset, None)?;
    expect_volume(&signed_volume(&arena, &out), 7.25)?;
    assert_closed_and_oriented(&out)?;
    assert_vector_area_zero(&arena, &out)
}

#[test]
fn offset_cube_union_is_exact_15_25() -> TestResult {
    let (b_size, b_offset) = offset_b();
    let (arena, out) = run(BoolOp::Union, 2.0, b_size, b_offset, None)?;
    expect_volume(&signed_volume(&arena, &out), 15.25)?;
    assert_closed_and_oriented(&out)?;
    assert_vector_area_zero(&arena, &out)
}

#[test]
fn offset_cube_intersect_is_exact_0_75() -> TestResult {
    let (b_size, b_offset) = offset_b();
    let (arena, out) = run(BoolOp::Intersect, 2.0, b_size, b_offset, None)?;
    expect_volume(&signed_volume(&arena, &out), 0.75)?;
    assert_closed_and_oriented(&out)?;
    assert_vector_area_zero(&arena, &out)
}

#[test]
fn untouched_face_of_a_keeps_its_vector_area() -> TestResult {
    // A's bottom face (z = -1) lies outside B, so its triangles pass through unchanged.
    let (b_size, b_offset) = offset_b();
    let (arena, out) = run(BoolOp::Difference, 2.0, b_size, b_offset, None)?;
    let a = box_mesh(DVec3::splat(2.0))?;
    let operands = [Operand {
        mesh: &a,
        transform: DMat4::IDENTITY,
    }];
    let mut input_arena = Arena::new();
    let input = meshes_to_tris(&operands, &mut input_arena).map_err(|e| format!("{e:?}"))?;
    let bottom = rat_from_f64(-1.0);
    let bottom_origs: Vec<i32> = input
        .iter()
        .filter(|t| t.v.iter().all(|&v| input_arena.vert(v).co[2] == bottom))
        .map(|t| t.orig)
        .collect();
    if bottom_origs.is_empty() {
        return Err("no bottom face found in operand A".to_owned());
    }
    let sum_of = |arena: &Arena, tris: &[Tri]| -> (usize, Rat3) {
        let mut count = 0;
        let mut sum: Rat3 = [Rat::zero(), Rat::zero(), Rat::zero()];
        for tri in tris.iter().filter(|t| bottom_origs.contains(&t.orig)) {
            count += 1;
            let area = double_area(arena, tri);
            for axis in 0..3 {
                sum[axis] += &area[axis];
            }
        }
        (count, sum)
    };
    if sum_of(&input_arena, &input) != sum_of(&arena, &out) {
        return Err("untouched face of A changed count or vector area".to_owned());
    }
    Ok(())
}

#[test]
fn nested_cube_difference_is_exact_7() -> TestResult {
    let (arena, out) = run(
        BoolOp::Difference,
        2.0,
        DVec3::splat(1.0),
        DVec3::ZERO,
        None,
    )?;
    expect_volume(&signed_volume(&arena, &out), 7.0)?;
    assert_closed_and_oriented(&out)?;
    assert_vector_area_zero(&arena, &out)
}

#[test]
fn nested_cube_union_is_exact_8() -> TestResult {
    let (arena, out) = run(BoolOp::Union, 2.0, DVec3::splat(1.0), DVec3::ZERO, None)?;
    expect_volume(&signed_volume(&arena, &out), 8.0)?;
    assert_closed_and_oriented(&out)
}

#[test]
fn nested_cube_intersect_is_exact_1() -> TestResult {
    let (arena, out) = run(BoolOp::Intersect, 2.0, DVec3::splat(1.0), DVec3::ZERO, None)?;
    expect_volume(&signed_volume(&arena, &out), 1.0)?;
    assert_closed_and_oriented(&out)
}

#[test]
fn open_operand_uses_patch_raycast_fallback() -> TestResult {
    let (b_size, b_offset) = offset_b();
    // Removing B's top at z=2 leaves an open face outside A.
    let (arena, out) = run(BoolOp::Difference, 2.0, b_size, b_offset, Some(2.0))?;
    expect_volume(&signed_volume(&arena, &out), 7.25)
}

#[test]
fn apply_bool_op_table() -> TestResult {
    let cases: [(BoolOp, &[i32], bool); 9] = [
        (BoolOp::Intersect, &[1, 2], true),
        (BoolOp::Intersect, &[1, 0], false),
        (BoolOp::Union, &[0, -1], true),
        (BoolOp::Union, &[0, 0], false),
        (BoolOp::Difference, &[1, 0], true),
        (BoolOp::Difference, &[1, 1], false),
        (BoolOp::Difference, &[0, 1], false),
        (BoolOp::Difference, &[1], true),
        (BoolOp::Difference, &[1, -1], true),
    ];
    for (op, winding, want) in cases {
        if apply_bool_op(op, winding) != want {
            return Err(format!("{op:?} {winding:?} != {want}"));
        }
    }
    Ok(())
}
