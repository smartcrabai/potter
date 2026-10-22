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

fn modifier(modifier_type: &str, params: Value) -> Modifier {
    Modifier {
        id: Id::new("modifier_test").unwrap(),
        modifier_type: modifier_type.to_owned(),
        name: modifier_type.to_owned(),
        enabled: true,
        show_viewport: true,
        show_render: true,
        params: serde_json::from_value(params).unwrap(),
        baked_ocean_cache: None,
        binding_data: None,
        runtime: ModifierRuntime::default(),
    }
}

/// `value` rounded to the nearest f32, as Blender stores it.
fn f32_exact(value: f64) -> f64 {
    f64::from(value as f32)
}

/// The position with each coordinate rounded to f32, as Blender's mesh holds it.
fn round_to_f32(point: DVec3) -> DVec3 {
    DVec3::new(f32_exact(point.x), f32_exact(point.y), f32_exact(point.z))
}

/// The spacing of f32 values at the magnitude of `value`: one unit in the last place.
fn f32_ulp(value: f64) -> f64 {
    let magnitude = (value as f32).abs();
    f64::from(f32::from_bits(magnitude.to_bits() + 1) - magnitude)
}

/// The signed volume of a mesh whose positions are rounded to f32, and a bound on how far that volume moves when
/// every coordinate moves by one f32 ulp.
///
/// The volume is the fan sum of `p0 . (p1 x p2) / 6` over each face's triangles. The derivative of a triangle's
/// term with respect to corner `k` is the cross product of the other two corners, divided by 6, so the first-order
/// change from moving coordinate `c` of corner `k` by one ulp is `|grad_c| * ulp(c)`. The bound sums that over all
/// corners and coordinates. It is first order: second-order terms are of order ulp squared, far below the bound.
struct F32Volume {
    volume: f64,
    ulp_bound: f64,
}

fn f32_volume(mesh: &Mesh) -> F32Volume {
    let vertices: HashMap<_, DVec3> = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, round_to_f32(vertex.co)))
        .collect();
    let mut volume = 0.0;
    let mut ulp_bound = 0.0;
    for face in &mesh.faces {
        let points: Vec<DVec3> = face.vertices.iter().map(|id| vertices[id]).collect();
        for index in 1..points.len() - 1 {
            let triangle = [points[0], points[index], points[index + 1]];
            volume += triangle[0].dot(triangle[1].cross(triangle[2])) / 6.0;
            for corner in 0..3 {
                let gradient = triangle[(corner + 1) % 3].cross(triangle[(corner + 2) % 3]) / 6.0;
                for axis in 0..3 {
                    ulp_bound += gradient[axis].abs() * f32_ulp(triangle[corner][axis]);
                }
            }
        }
    }
    F32Volume { volume, ulp_bound }
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
        let F32Volume { volume, ulp_bound } = f32_volume(&mesh);
        if faces != blender_faces {
            failures.push(format!("{label}: {faces} faces, Blender {blender_faces}"));
        }
        if open != blender_open {
            failures.push(format!(
                "{label}: {open} edges not bordering two faces, Blender {blender_open}"
            ));
        }
        if (volume - blender_volume).abs() > ulp_bound {
            failures.push(format!(
                "{label}: volume {volume:.9}, Blender {blender_volume:.9}, difference {:.3e} exceeds the one-ulp bound {ulp_bound:.3e}",
                (volume - blender_volume).abs()
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "Potter differs from Blender 5.2.2 FLOAT:\n{}",
        failures.join("\n")
    );
}
