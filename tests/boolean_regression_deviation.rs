//! Evidence for the one-quantum face window in the Boolean comparator. For the three cases whose only
//! mismatches sit on a 1e-5 quantum boundary, this reads Potter's unquantized world-space rings and
//! UVs and reports the largest absolute deviation from the Blender expectation. The expectations are
//! rounded to 1e-6 by the generator, so the deviation floor is about 5e-7 plus f32 noise. The test
//! fails if any corner deviates by more than the comparator's 1e-5 tolerance, or if a face has no
//! match.

#[path = "common/blender_file.rs"]
mod blender_file;
use blender_file::blender_executable;

#[expect(
    dead_code,
    reason = "this probe uses a subset of the shared comparator"
)]
#[path = "fixtures/boolean_parity/comparator.rs"]
mod comparator;
use comparator::{TestResult, find_node, import_blend_scene, run_generator};

use std::{collections::HashMap, fs, path::Path};

use glam::{DMat4, DVec3};
use potter_core::{
    eval::{EvaluationContext, Snapshot},
    model::{Id, SceneDoc},
};
use serde_json::Value;
use tempfile::tempdir;

const BLENDER_REQUIRED: &str = "Blender is required for the Boolean deviation probe: install Blender 5.2.2, or set POTTER_BLENDER to its executable";

/// The cases whose mismatches sit on a quantum boundary.
const CASES: [&str; 3] = [
    "rotated_touch_30",
    "rotated_touch_41",
    "unset_object_monkey",
];

/// The comparator's stated coordinate and UV tolerance.
const TOLERANCE: f64 = 1.0e-5;

struct Face {
    ring: Vec<DVec3>,
    uv: Option<Vec<[f64; 2]>>,
    material: u64,
}

fn point(value: &Value) -> TestResult<DVec3> {
    let values = value.as_array().ok_or("expected point is not an array")?;
    let coordinate = |index: usize| -> TestResult<f64> {
        values
            .get(index)
            .and_then(Value::as_f64)
            .ok_or_else(|| "expected coordinate is missing".into())
    };
    Ok(DVec3::new(coordinate(0)?, coordinate(1)?, coordinate(2)?))
}

fn pair(value: &Value) -> TestResult<[f64; 2]> {
    let values = value.as_array().ok_or("expected UV is not an array")?;
    let component = |index: usize| -> TestResult<f64> {
        values
            .get(index)
            .and_then(Value::as_f64)
            .ok_or_else(|| "expected UV component is missing".into())
    };
    Ok([component(0)?, component(1)?])
}

/// Blender's expectation, faces in the JSON's order.
fn expected_faces(json: &Value) -> TestResult<Vec<Face>> {
    let faces = json["faces"]
        .as_array()
        .ok_or("expected faces are missing")?;
    faces
        .iter()
        .map(|face| -> TestResult<Face> {
            let ring = face["ring"]
                .as_array()
                .ok_or("expected ring is missing")?
                .iter()
                .map(point)
                .collect::<TestResult<Vec<_>>>()?;
            let uv = match face.get("uv") {
                Some(values) => Some(
                    values
                        .as_array()
                        .ok_or("expected UV list is invalid")?
                        .iter()
                        .map(pair)
                        .collect::<TestResult<Vec<_>>>()?,
                ),
                None => None,
            };
            let material = face["material_index"]
                .as_u64()
                .ok_or("expected material index is invalid")?;
            Ok(Face { ring, uv, material })
        })
        .collect()
}

/// Potter's evaluated Boolean in world space, with unquantized coordinates. UVs come from the
/// `uv_layers` store's `UVMap` and are flipped to Blender's V-up convention.
fn potter_faces(document: &SceneDoc, node_id: &Id) -> TestResult<Vec<Face>> {
    let mut boolean_only = document.clone();
    for (id, node) in &mut boolean_only.nodes {
        for modifier in &mut node.modifiers {
            modifier.enabled = id == node_id && modifier.modifier_type == "boolean";
        }
    }
    let snapshot = Snapshot::evaluate(&boolean_only, &EvaluationContext::default())
        .map_err(|error| format!("{}: {error}", node_id.as_str()))?;
    let mesh = snapshot
        .meshes
        .get(node_id)
        .ok_or("Potter did not evaluate the node")?;
    let node = snapshot
        .nodes
        .get(node_id)
        .ok_or("Potter has no evaluated node")?;
    let world = DMat4::from_cols_array(&node.world_matrix);
    let positions: HashMap<u32, DVec3> = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, world.transform_point3(vertex.co)))
        .collect();
    let face_index: HashMap<u64, usize> = mesh
        .faces
        .iter()
        .enumerate()
        .map(|(index, face)| (u64::from(face.id), index))
        .collect();
    let mut uv_rows: Vec<Option<Vec<[f64; 2]>>> = vec![None; mesh.faces.len()];
    if let Some(layers) = mesh
        .attributes
        .get("uv_layers")
        .and_then(|store| store.get("layers"))
        .and_then(Value::as_array)
    {
        if let Some(layer) = layers
            .iter()
            .find(|layer| layer.get("name").and_then(Value::as_str) == Some("UVMap"))
        {
            for entry in layer["faces"].as_array().into_iter().flatten() {
                let Some(&index) = entry
                    .get("face_id")
                    .and_then(Value::as_u64)
                    .and_then(|id| face_index.get(&id))
                else {
                    continue;
                };
                if let Some(row) = entry.get("uv").and_then(Value::as_array) {
                    uv_rows[index] = Some(
                        row.iter()
                            .map(pair)
                            .collect::<TestResult<Vec<_>>>()?
                            .into_iter()
                            .map(|[u, v]| [u, 1.0 - v])
                            .collect(),
                    );
                }
            }
        }
    }
    mesh.faces
        .iter()
        .enumerate()
        .map(|(index, face)| -> TestResult<Face> {
            let ring = face
                .vertices
                .iter()
                .map(|id| {
                    positions
                        .get(id)
                        .copied()
                        .ok_or("Potter face references a missing vertex")
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Face {
                ring,
                uv: uv_rows[index].clone(),
                material: u64::from(face.material_index),
            })
        })
        .collect()
}

/// Largest coordinate and UV deviation of `actual` against `expected`, for the rotation of `actual`
/// that minimizes the coordinate deviation. `None` when the sizes or materials differ.
fn deviation(actual: &Face, expected: &Face) -> Option<(f64, f64)> {
    let count = expected.ring.len();
    if actual.ring.len() != count || actual.material != expected.material {
        return None;
    }
    let mut best: Option<(f64, f64)> = None;
    for shift in 0..count {
        let mut coordinate = 0.0_f64;
        let mut uv = 0.0_f64;
        for index in 0..count {
            let source = (index + shift) % count;
            coordinate = coordinate.max(
                (actual.ring[source] - expected.ring[index])
                    .abs()
                    .max_element(),
            );
            if let (Some(left), Some(right)) = (&actual.uv, &expected.uv) {
                uv = uv
                    .max((left[source][0] - right[index][0]).abs())
                    .max((left[source][1] - right[index][1]).abs());
            }
        }
        if best.is_none_or(|(best_coordinate, _)| coordinate < best_coordinate) {
            best = Some((coordinate, uv));
        }
    }
    best
}

#[test]
fn rotated_and_monkey_deviations_are_within_tolerance() -> TestResult {
    let blender = blender_executable().ok_or(BLENDER_REQUIRED)?;
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/boolean_parity/regressions");
    let directory = tempdir()?;
    let root = fs::canonicalize(directory.path())?;
    run_generator(
        &blender,
        &fixture.join("generate_regression_expectations.py"),
        &root,
    )?;
    let document = import_blend_scene(&blender, &root.join("scene.blend"), &root.join("project"))?;

    let mut failures = Vec::new();
    for case in CASES {
        let json: Value = serde_json::from_slice(&fs::read(
            fixture.join("expected").join(format!("{case}.json")),
        )?)?;
        let expected = expected_faces(&json)?;
        let node_id = find_node(&document, case)?;
        let actual = potter_faces(&document, &node_id)?;
        let mut used = vec![false; actual.len()];
        let mut max_coordinate = 0.0_f64;
        let mut max_uv = 0.0_f64;
        let mut worst_face = None;
        let mut unmatched = Vec::new();
        for (index, expected_face) in expected.iter().enumerate() {
            let mut best: Option<(usize, f64, f64)> = None;
            for (candidate, actual_face) in actual.iter().enumerate() {
                if used[candidate] {
                    continue;
                }
                if let Some((coordinate, uv)) = deviation(actual_face, expected_face)
                    && best.is_none_or(|(_, best_coordinate, _)| coordinate < best_coordinate)
                {
                    best = Some((candidate, coordinate, uv));
                }
            }
            match best {
                Some((candidate, coordinate, uv)) => {
                    used[candidate] = true;
                    if coordinate > max_coordinate {
                        max_coordinate = coordinate;
                        worst_face = Some(index);
                    }
                    max_uv = max_uv.max(uv);
                }
                None => unmatched.push(index),
            }
        }
        eprintln!(
            "PROBE {case}: potter_faces={} blender_faces={} unmatched={:?} max_coord_dev={max_coordinate:.3e} max_uv_dev={max_uv:.3e} worst_face={worst_face:?}",
            actual.len(),
            expected.len(),
            unmatched
        );
        if actual.len() != expected.len() || !unmatched.is_empty() {
            failures.push(format!("{case}: face counts or matches differ"));
        }
        if max_coordinate > TOLERANCE || max_uv > TOLERANCE {
            failures.push(format!(
                "{case}: deviation coord {max_coordinate:.3e} uv {max_uv:.3e} exceeds {TOLERANCE:.0e}"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}
