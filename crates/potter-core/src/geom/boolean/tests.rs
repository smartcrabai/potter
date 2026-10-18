//! Cross-slice tests of Boolean kernels and Blender 5.2.2 behavior.
//!
//! Expected coordinates come from Blender's EXACT DIFFERENCE result for an offset cube
//! (research-pr17-boolean.md §3.1, probe `probe_boolean.py`): the vertices of that result that are
//! not operand corners are triangle-triangle intersection points.

use std::collections::{HashMap, HashSet};

use glam::{DMat4, DVec3};

use num_traits::Zero;

use super::{
    BooleanOperand, FloatObjectQuickMesh, MaterialMode, Operation,
    exact::{Rat, Rat3, add3, cross3, rat_from_f64, sub3},
    float_object_quick_mesh,
    imesh::{Arena, Tri},
    input::{Operand, meshes_to_tris},
    intersect::trimesh_nary_intersect,
    material_remaps,
};
use crate::geom::{BoxParams, Mesh};
use crate::model::Id;

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

#[test]
fn material_remaps_index_matches_subject_slots_and_transfer_pools_new_materials() {
    let subject = [Some("a0"), Some("a1")];
    let operands = [vec![Some("a1"), Some("b0")], vec![Some("b0"), Some("c0")]];
    assert_eq!(
        run_remaps(MaterialMode::Index, &subject, &operands),
        (vec![vec![], vec![1, 1], vec![0, 1]], owned_slots(&subject))
    );
    assert_eq!(
        run_remaps(MaterialMode::Transfer, &subject, &operands),
        (
            vec![vec![], vec![1, 2], vec![2, 3]],
            owned_slots(&[Some("a0"), Some("a1"), Some("b0"), Some("c0")])
        )
    );
}

#[test]
fn material_remaps_index_prefers_same_slot_over_earlier_match() {
    // Slot 1 holds `a` in both, so it keeps index 1 although subject slot 0 also holds `a`.
    let subject = [Some("a"), Some("a")];
    let operands = [vec![Some("x"), Some("a")]];
    assert_eq!(
        run_remaps(MaterialMode::Index, &subject, &operands),
        (vec![vec![], vec![0, 1]], owned_slots(&subject))
    );
}

#[test]
fn material_remaps_index_takes_first_matching_subject_slot() {
    // Blender's material map keeps the first subject slot per material, so `a` maps to 0.
    let subject = [Some("a"), Some("b"), Some("a")];
    let operands = [vec![Some("c"), Some("a")]];
    assert_eq!(
        run_remaps(MaterialMode::Index, &subject, &operands),
        (vec![vec![], vec![0, 0]], owned_slots(&subject))
    );
}

#[test]
fn material_remaps_index_unmatched_slot_keeps_local_index_in_range_else_zero() {
    let subject = [Some("a"), Some("b")];
    let operands = [
        vec![Some("x"), Some("y")],
        vec![Some("x"), Some("y"), Some("z")],
    ];
    assert_eq!(
        run_remaps(MaterialMode::Index, &subject, &operands),
        (
            vec![vec![], vec![0, 1], vec![0, 1, 0]],
            owned_slots(&subject)
        )
    );
}

#[test]
fn material_remaps_empty_operand_maps_by_mode() {
    let subject = [Some("a")];
    let operands = [vec![]];
    assert_eq!(
        run_remaps(MaterialMode::Index, &subject, &operands),
        (vec![vec![], vec![0]], owned_slots(&subject))
    );
    assert_eq!(
        run_remaps(MaterialMode::Transfer, &subject, &operands),
        (vec![vec![], vec![]], owned_slots(&subject))
    );
}

#[test]
fn material_remaps_transfer_empty_subject_starts_pool_with_none() {
    let subject: [Option<&str>; 0] = [];
    let operands = [vec![Some("a"), None]];
    assert_eq!(
        run_remaps(MaterialMode::Transfer, &subject, &operands),
        (vec![vec![], vec![1, 0]], owned_slots(&[None, Some("a")]))
    );
    // With no subject slots, INDEX has no pool and each operand slot falls back to 0.
    assert_eq!(
        run_remaps(MaterialMode::Index, &subject, &operands),
        (vec![vec![], vec![0, 0]], owned_slots(&[]))
    );
}

#[test]
fn material_remaps_transfer_matches_none_to_pooled_none() {
    let subject = [None, Some("a")];
    let operands = [vec![None, Some("b"), None]];
    assert_eq!(
        run_remaps(MaterialMode::Transfer, &subject, &operands),
        (
            vec![vec![], vec![0, 2, 0]],
            owned_slots(&[None, Some("a"), Some("b")])
        )
    );
}

#[test]
fn material_remaps_transfer_pools_each_subject_material_once() {
    // Blender's pool is a VectorSet, so the duplicate `a` takes one pool entry.
    let subject = [Some("a"), Some("a"), Some("b")];
    let operands = [vec![Some("b"), Some("c")]];
    assert_eq!(
        run_remaps(MaterialMode::Transfer, &subject, &operands),
        (
            vec![vec![], vec![1, 2]],
            owned_slots(&[Some("a"), Some("b"), Some("c")])
        )
    );
}

/// Owned material slots for the remap tests; `None` is an empty slot.
fn owned_slots(names: &[Option<&str>]) -> Vec<Option<Id>> {
    names
        .iter()
        .map(|&name| name.map(Id::from_static))
        .collect()
}

/// Borrows owned slots in the shape `material_remaps` takes.
fn borrowed(slots: &[Option<Id>]) -> Vec<Option<&Id>> {
    slots.iter().map(Option::as_ref).collect()
}

/// Runs `material_remaps` on named slots and returns `(remaps, pool)`.
fn run_remaps(
    mode: MaterialMode,
    subject: &[Option<&str>],
    operands: &[Vec<Option<&str>>],
) -> (Vec<Vec<u32>>, Vec<Option<Id>>) {
    let subject_slots = owned_slots(subject);
    let operand_slots: Vec<Vec<Option<Id>>> =
        operands.iter().map(|names| owned_slots(names)).collect();
    let subject_refs = borrowed(&subject_slots);
    let operand_refs: Vec<Vec<Option<&Id>>> =
        operand_slots.iter().map(|slots| borrowed(slots)).collect();
    material_remaps(mode, &subject_refs, &operand_refs)
}

fn quick_mesh_triangle() -> Result<Mesh, String> {
    Mesh::from_positions_and_faces(
        vec![
            DVec3::new(1.0, 2.0, 3.0),
            DVec3::new(-1.0, 0.5, 2.0),
            DVec3::new(0.0, -2.0, 1.0),
        ],
        vec![vec![0, 1, 2]],
    )
    .map_err(|error| format!("{error:?}"))
}

fn quick_mesh_wire() -> Result<Mesh, String> {
    let mut mesh = Mesh::from_positions_and_faces(vec![DVec3::ZERO, DVec3::X, DVec3::Y], vec![])
        .map_err(|error| format!("{error:?}"))?;
    mesh.insert_edge([0, 1]).map_err(|error| format!("{error:?}"))?;
    Ok(mesh)
}

fn quick_mesh_operand(mesh: Mesh, to_subject: DMat4) -> BooleanOperand {
    BooleanOperand {
        mesh,
        to_subject,
        float_flip: None,
        material_remap: vec![],
    }
}

#[test]
fn float_object_quick_mesh_returns_none_when_both_meshes_have_faces() -> TestResult {
    let subject = quick_mesh_triangle()?;
    let operand = quick_mesh_operand(quick_mesh_triangle()?, DMat4::IDENTITY);

    assert!(float_object_quick_mesh(&subject, &operand, Operation::Union).is_none());
    Ok(())
}

#[test]
fn float_object_quick_mesh_matches_blender_faceless_object_results() -> TestResult {
    // Blender 5.2.2 MOD_boolean.cc:get_quick_mesh returns an empty mesh for INTERSECT and
    // the subject for DIFFERENCE; UNION keeps the subject if it has faces, else copies operand.
    let subject = quick_mesh_triangle()?;
    let empty_operand = quick_mesh_operand(quick_mesh_wire()?, DMat4::IDENTITY);
    assert!(matches!(
        float_object_quick_mesh(&subject, &empty_operand, Operation::Intersect),
        Some(FloatObjectQuickMesh::Empty)
    ));
    assert!(matches!(
        float_object_quick_mesh(&subject, &empty_operand, Operation::Union),
        Some(FloatObjectQuickMesh::Subject)
    ));
    assert!(matches!(
        float_object_quick_mesh(&subject, &empty_operand, Operation::Difference),
        Some(FloatObjectQuickMesh::Subject)
    ));

    let empty_subject = quick_mesh_wire()?;
    let operand = quick_mesh_operand(quick_mesh_triangle()?, DMat4::IDENTITY);
    assert!(matches!(
        float_object_quick_mesh(&empty_subject, &operand, Operation::Intersect),
        Some(FloatObjectQuickMesh::Empty)
    ));
    assert!(matches!(
        float_object_quick_mesh(&empty_subject, &operand, Operation::Union),
        Some(FloatObjectQuickMesh::Operand(_))
    ));
    assert!(matches!(
        float_object_quick_mesh(&empty_subject, &operand, Operation::Difference),
        Some(FloatObjectQuickMesh::Subject)
    ));
    assert!(matches!(
        float_object_quick_mesh(&empty_subject, &empty_operand, Operation::Intersect),
        Some(FloatObjectQuickMesh::Empty)
    ));
    assert!(matches!(
        float_object_quick_mesh(&empty_subject, &empty_operand, Operation::Union),
        Some(FloatObjectQuickMesh::Operand(_))
    ));
    assert!(matches!(
        float_object_quick_mesh(&empty_subject, &empty_operand, Operation::Difference),
        Some(FloatObjectQuickMesh::Subject)
    ));
    Ok(())
}

#[test]
fn float_object_quick_mesh_union_preserves_and_transforms_operand() -> TestResult {
    let subject = quick_mesh_wire()?;
    let mut operand_mesh = quick_mesh_triangle()?;
    operand_mesh.faces[0].material_index = 2;
    operand_mesh.attributes.insert(
        "source_marker".to_owned(),
        serde_json::json!({"domain":"point","type":"float","values":{"v0":1.25}}),
    );
    let transform = DMat4::from_scale_rotation_translation(
        DVec3::new(2.0, 3.0, 0.5),
        glam::DQuat::IDENTITY,
        DVec3::new(10.0, -4.0, 2.0),
    );
    let operand = quick_mesh_operand(operand_mesh.clone(), transform);

    let Some(FloatObjectQuickMesh::Operand(result)) =
        float_object_quick_mesh(&subject, &operand, Operation::Union)
    else {
        return Err("a faceless subject unions to a transformed operand copy".to_owned());
    };

    // Blender's quick path transforms vertex coordinates but retains the copied mesh data.
    let transformed_positions = [
        DVec3::new(12.0, 2.0, 3.5),
        DVec3::new(8.0, -2.5, 3.0),
        DVec3::new(10.0, -10.0, 2.5),
    ];
    let mut expected = operand_mesh;
    for (vertex, position) in expected.vertices.iter_mut().zip(transformed_positions) {
        vertex.co = position;
    }
    assert_eq!(result, expected);
    Ok(())
}
