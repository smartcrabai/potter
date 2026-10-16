//! Cross-slice tests of the exact intersection stage against Blender 5.2.2 output.
//!
//! Expected coordinates come from Blender's EXACT DIFFERENCE result for an offset cube
//! (research-pr17-boolean.md §3.1, probe `probe_boolean.py`): the vertices of that result that are
//! not operand corners are triangle-triangle intersection points.

use std::collections::{HashMap, HashSet};

use glam::{DMat4, DVec3};

use num_traits::Zero;

use super::{
    exact::{Rat, Rat3, add3, cross3, rat_from_f64, sub3},
    imesh::{Arena, Tri},
    input::{Operand, meshes_to_tris},
    intersect::trimesh_nary_intersect,
};
use crate::geom::{BoxParams, Mesh};

type TestResult = Result<(), String>;

/// Triangles of an offset-cube boolean before and after the exact intersection stage.
struct OffsetCubes {
    arena: Arena,
    inputs: Vec<Tri>,
    outputs: Vec<Tri>,
}

/// Operand A: cube `[-1, 1]^3`. Operand B: `[0.5, 2.5] x [-0.5, 1.5] x [0, 2]`, then A's
/// faces are indices 0..6 and B's are 6..12.
fn intersect_offset_cubes() -> Result<OffsetCubes, String> {
    let cube = Mesh::box_mesh(BoxParams {
        size: DVec3::splat(2.0),
    })
    .map_err(|error| format!("{error:?}"))?;
    let operands = [
        Operand {
            mesh: &cube,
            transform: DMat4::IDENTITY,
        },
        Operand {
            mesh: &cube,
            transform: DMat4::from_translation(DVec3::new(1.5, 0.5, 1.0)),
        },
    ];
    let mut arena = Arena::new();
    let inputs = meshes_to_tris(&operands, &mut arena).map_err(|error| format!("{error:?}"))?;
    let outputs = trimesh_nary_intersect(
        &mut arena,
        &inputs,
        2,
        &|orig| usize::from(orig >= 6),
        false,
    );
    Ok(OffsetCubes {
        arena,
        inputs,
        outputs,
    })
}

fn zero3() -> Rat3 {
    [Rat::zero(), Rat::zero(), Rat::zero()]
}

fn point(x: f64, y: f64, z: f64) -> Rat3 {
    [rat_from_f64(x), rat_from_f64(y), rat_from_f64(z)]
}

/// Twice the vector area of a triangle, exact.
fn double_area(arena: &Arena, tri: &Tri) -> Rat3 {
    let [a, b, c] = tri.v.map(|v| &arena.vert(v).co);
    cross3(&sub3(b, a), &sub3(c, a))
}

#[test]
fn offset_cube_cut_points_match_blender_exact_difference() -> TestResult {
    let cubes = intersect_offset_cubes()?;
    let present: HashSet<Rat3> = cubes
        .outputs
        .iter()
        .flat_map(|t| t.v.iter().map(|&v| cubes.arena.vert(v).co.clone()))
        .collect();
    // Blender 5.2.2 EXACT DIFFERENCE vertices that are not operand corners (§3.1 ring list).
    let blender_cut_points = [
        point(0.5, 1.0, 1.0),
        point(0.5, -0.5, 1.0),
        point(1.0, -0.5, 1.0),
        point(0.5, 1.0, 0.0),
        point(1.0, 1.0, 0.0),
        point(1.0, -0.5, 0.0),
        point(0.5, -0.5, 0.0),
    ];
    for expected in &blender_cut_points {
        if !present.contains(expected) {
            return Err(format!("missing Blender cut point {expected:?}"));
        }
    }
    Ok(())
}

#[test]
fn subdivision_conserves_exact_vector_area_per_input_face() -> TestResult {
    let cubes = intersect_offset_cubes()?;
    let mut want: HashMap<i32, Rat3> = HashMap::new();
    for tri in &cubes.inputs {
        let area = double_area(&cubes.arena, tri);
        let entry = want.entry(tri.orig).or_insert_with(zero3);
        *entry = add3(entry, &area);
    }
    let mut got: HashMap<i32, Rat3> = HashMap::new();
    for tri in &cubes.outputs {
        let area = double_area(&cubes.arena, tri);
        let entry = got.entry(tri.orig).or_insert_with(zero3);
        *entry = add3(entry, &area);
    }
    // Splitting a triangle only partitions its area, so exact vector areas must add up.
    for (orig, expected) in &want {
        if got.get(orig) != Some(expected) {
            return Err(format!("vector area of input face {orig} changed"));
        }
    }
    Ok(())
}
