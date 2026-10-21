//! Potter's FLOAT union of a unit box and a near-coplanar operand box, against Blender 5.2.2.
//!
//! The oracle is `tests/fixtures/boolean_parity/float_sliver/float_sliver_union.json`, written headless by
//! `generate_union_sliver.py`. It covers the proptest input that `boolean_float_union_of_overlapping_boxes_is_watertight`
//! shrank to (gap 2.846e-6), a sweep of signed gaps from -1e-4 to 1e-4 at the operand's -x face (negative: the
//! operand face inside the subject), and controls, each at the default `double_threshold` (1e-6) and at 0. Blender's
//! answer there is not the analytic union, so this test asserts Blender's answer, not the analytic volume.

#![expect(
    clippy::unwrap_used,
    reason = "integration tests use concise assertions"
)]

use std::{collections::HashMap, fs, path::Path};

use glam::{DMat4, DVec3};
use potter_core::geom::{BoxParams, Mesh, boolean::BooleanOperand, modifiers::evaluate_modifiers};
use potter_core::model::{Id, Modifier, ModifierRuntime};
use serde_json::{Value, json};

const ORACLE: &str = "tests/fixtures/boolean_parity/float_sliver/float_sliver_union.json";

/// Absolute tolerance on the signed volume. Both sides use the same f32-rounded inputs, so what remains is
/// rounding of the output vertices (Blender keeps its mesh in f32). It is below the 5.8e-6 sliver volume of
/// the failing input, so a result that drops the sliver fails here.
const VOLUME_TOLERANCE: f64 = 1.0e-6;

fn modifier(modifier_type: &str, params: Value) -> Modifier {
    Modifier {
        id: Id::new("modifier_test").unwrap(),
        modifier_type: modifier_type.to_owned(),
        name: modifier_type.to_owned(),
        enabled: true,
        show_viewport: true,
        show_render: true,
        params: serde_json::from_value(params).unwrap(),
        binding_data: None,
        runtime: ModifierRuntime::default(),
    }
}

/// `value` rounded to f32, as Blender stores it.
fn f32_exact(value: f64) -> f64 {
    f64::from(value as f32)
}

fn signed_volume(mesh: &Mesh) -> f64 {
    let vertices: HashMap<_, _> = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.co))
        .collect();
    mesh.faces
        .iter()
        .map(|face| {
            let points: Vec<_> = face.vertices.iter().map(|id| vertices[id]).collect();
            (1..points.len() - 1)
                .map(|index| points[0].dot(points[index].cross(points[index + 1])) / 6.0)
                .sum::<f64>()
        })
        .sum::<f64>()
}

/// The number of edges that do not border exactly two faces.
fn edges_not_two_faces(mesh: &Mesh) -> usize {
    let mut counts: HashMap<[u32; 2], usize> = HashMap::new();
    for face in &mesh.faces {
        for index in 0..face.vertices.len() {
            let first = face.vertices[index];
            let second = face.vertices[(index + 1) % face.vertices.len()];
            *counts
                .entry([first.min(second), first.max(second)])
                .or_default() += 1;
        }
    }
    counts.values().filter(|count| **count != 2).count()
}

/// Potter's FLOAT union of the origin-centred subject `[-1, 1]^3` with a box of `size` at `offset`.
fn potter_union(size: f64, offset: DVec3, threshold: f64) -> Result<Mesh, String> {
    let subject = Mesh::box_mesh(BoxParams::default()).unwrap();
    let mut operand = Mesh::box_mesh(BoxParams {
        size: DVec3::splat(size),
    })
    .unwrap();
    for vertex in &mut operand.vertices {
        vertex.co += offset;
    }
    let mut boolean = modifier(
        "boolean",
        json!({"solver": "FLOAT", "operation": "UNION", "double_threshold": threshold}),
    );
    boolean.runtime.boolean_operands = std::iter::once(BooleanOperand {
        mesh: operand,
        to_subject: DMat4::IDENTITY,
        float_flip: None,
        material_remap: Vec::new(),
    })
    .collect();
    evaluate_modifiers(&subject, &[boolean], "TestMesh").map_err(|error| format!("{error:?}"))
}

#[test]
fn boolean_float_sliver_sweep_matches_blender() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(ORACLE);
    let oracle: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    let cases = oracle["cases"].as_array().unwrap();
    let results = oracle["results"].as_array().unwrap();
    assert_eq!(results.len(), 2 * cases.len(), "two thresholds per case");

    let mut failures = Vec::new();
    for record in results {
        let name = record["case"].as_str().unwrap();
        let threshold = record["threshold"].as_f64().unwrap();
        let case = cases
            .iter()
            .find(|case| case["name"].as_str() == Some(name))
            .unwrap();
        let size = f32_exact(case["size"].as_f64().unwrap());
        let offset = DVec3::new(
            f32_exact(case["x"].as_f64().unwrap()),
            f32_exact(case["y"].as_f64().unwrap()),
            f32_exact(case["z"].as_f64().unwrap()),
        );
        let label = format!("{name} at double_threshold {threshold:e}");
        let mesh = match potter_union(size, offset, threshold) {
            Ok(mesh) => mesh,
            Err(error) => {
                failures.push(format!("{label}: Potter returned an error: {error}"));
                continue;
            }
        };

        let blender_faces = record["faces"].as_u64().unwrap();
        let blender_open = record["edges_not_two_faces"].as_u64().unwrap();
        let blender_volume = record["volume"].as_f64().unwrap();
        let faces = mesh.faces.len() as u64;
        let open = edges_not_two_faces(&mesh) as u64;
        let volume = signed_volume(&mesh);
        if faces != blender_faces {
            failures.push(format!("{label}: {faces} faces, Blender {blender_faces}"));
        }
        if open != blender_open {
            failures.push(format!(
                "{label}: {open} edges not bordering two faces, Blender {blender_open}"
            ));
        }
        if (volume - blender_volume).abs() > VOLUME_TOLERANCE {
            failures.push(format!(
                "{label}: volume {volume:.9}, Blender {blender_volume:.9}"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "Potter differs from Blender 5.2.2 FLOAT:\n{}",
        failures.join("\n")
    );
}
