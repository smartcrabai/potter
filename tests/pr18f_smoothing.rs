#![expect(
    clippy::unwrap_used,
    reason = "integration tests use concise assertions"
)]

//! Laplacian Smooth and Corrective Smooth regressions against Blender 5.2.2.
//!
//! `fixtures/pr18f_smoothing/blender_oracle.json` holds the Blender outputs for a 4x6 open tube
//! (flat rest, bent deformed input, vertex group G) and a closed torus, produced by running the
//! Blender modifiers headless on the same inputs. Corrective BIND cases bind the flat tube, then
//! move the mesh to the bent positions before evaluation.

use glam::DVec3;
use potter_core::{
    error::ErrorCode,
    geom::{Mesh, modifiers::evaluate_modifiers},
    model::{Id, Modifier, ModifierRuntime},
};
use serde_json::{Map, Value, json};

const ORACLE: &str = include_str!("fixtures/pr18f_smoothing/blender_oracle.json");
/// Blender stores float32; the implementation computes in f64 against the same algorithm.
const TOLERANCE: f64 = 2.0e-5;
const SUBJECT_NAME: &str = "SmoothingSubject";

fn oracle() -> Value {
    serde_json::from_str(ORACLE).unwrap()
}

fn points(value: &Value) -> Vec<DVec3> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|point| {
            let coords: Vec<f64> = point
                .as_array()
                .unwrap()
                .iter()
                .map(|coord| coord.as_f64().unwrap())
                .collect();
            DVec3::new(coords[0], coords[1], coords[2])
        })
        .collect()
}

fn modifier(modifier_type: &str, params: Value) -> Modifier {
    Modifier {
        id: Id::new("modifier_test").unwrap(),
        modifier_type: modifier_type.to_owned(),
        name: modifier_type.to_owned(),
        enabled: true,
        params: serde_json::from_value(params).unwrap(),
        binding_data: None,
        runtime: ModifierRuntime::default(),
    }
}

fn bound(modifier: Modifier, flat_coords: &Value) -> Modifier {
    Modifier {
        binding_data: Some(json!({
            "format": "blender_corrective_bind_v1",
            "coords": flat_coords,
        })),
        ..modifier
    }
}

/// Mesh from the oracle's positions and face list; `closed` selects the torus topology.
fn tube(oracle: &Value, positions: &Value, closed: bool) -> Mesh {
    let faces_key = if closed { "faces_closed" } else { "faces_open" };
    let faces: Vec<Vec<usize>> = oracle[faces_key]
        .as_array()
        .unwrap()
        .iter()
        .map(|face| {
            face.as_array()
                .unwrap()
                .iter()
                .map(|index| index.as_u64().unwrap() as usize)
                .collect()
        })
        .collect();
    Mesh::from_positions_and_faces(points(positions), faces).unwrap()
}

/// Attaches the oracle's vertex group `G`; vertices absent from it have weight zero.
fn with_group(mut mesh: Mesh, oracle: &Value) -> Mesh {
    let weights: Map<String, Value> = oracle["groups"]
        .as_array()
        .unwrap()
        .iter()
        .map(|pair| (format!("v{}", pair[0]), pair[1].clone()))
        .collect();
    mesh.attributes
        .insert("vertex_groups".to_owned(), json!({ "G": weights }));
    mesh
}

fn input_for(oracle: &Value, case_params: &Value, bent: &Value) -> Mesh {
    let mesh = tube(oracle, bent, false);
    if case_params.get("vertex_group").is_some() {
        with_group(mesh, oracle)
    } else {
        mesh
    }
}

fn expected(oracle: &Value, case: &str) -> Vec<DVec3> {
    points(&oracle["cases"][case]["out"])
}

fn assert_points(label: &str, actual: &Mesh, expected: &[DVec3]) {
    assert_eq!(
        actual.vertices.len(),
        expected.len(),
        "{label}: vertex count"
    );
    for (index, (vertex, want)) in actual.vertices.iter().zip(expected).enumerate() {
        let error = (vertex.co - *want).abs().max_element();
        assert!(
            error < TOLERANCE,
            "{label}: vertex {index} is {:?}, Blender gives {want:?}",
            vertex.co
        );
    }
}

fn evaluate(mesh: &Mesh, modifiers: &[Modifier]) -> Mesh {
    evaluate_modifiers(mesh, modifiers, SUBJECT_NAME).unwrap()
}

#[test]
fn laplacian_smooth_matches_blender_open_tube_cases() {
    let oracle = oracle();
    for case in [
        "lap_a_plain_iter3",
        "lap_b_norm_vol_iter2",
        "lap_c_negative_axes",
        "lap_d_neg_border_noz",
        "lap_h_zero_lambda_border",
        "lap_i_bent_norm",
        "lap_e_group",
        "lap_f_group_inv_plain",
        "lap_g_missing_group",
    ] {
        let params = oracle["cases"][case]["params"].clone();
        let mesh = input_for(&oracle, &params, &oracle["bent"]);
        let result = evaluate(&mesh, &[modifier("laplacian_smooth", params)]);
        assert_points(case, &result, &expected(&oracle, case));
    }
}

#[test]
fn laplacian_smooth_matches_blender_closed_torus() {
    let oracle = oracle();
    let params = oracle["cases"]["lap_t_closed_torus"]["params"].clone();
    let mesh = tube(&oracle, &oracle["bent"], true);
    let result = evaluate(&mesh, &[modifier("laplacian_smooth", params)]);
    assert_points(
        "lap_t_closed_torus",
        &result,
        &expected(&oracle, "lap_t_closed_torus"),
    );
}

#[test]
fn laplacian_smooth_zero_weight_vertices_stay_put() {
    let oracle = oracle();
    let params = oracle["cases"]["lap_e_group"]["params"].clone();
    let mesh = input_for(&oracle, &params, &oracle["bent"]);
    let result = evaluate(&mesh, &[modifier("laplacian_smooth", params)]);
    let bent = points(&oracle["bent"]);
    // Vertices absent from group G have weight 0: their row is the identity.
    for index in (3..24).step_by(4) {
        let error = (result.vertices[index].co - bent[index])
            .abs()
            .max_element();
        assert!(error < 1.0e-9, "vertex {index} moved by {error}");
    }
}

#[test]
fn corrective_smooth_bind_matches_blender() {
    let oracle = oracle();
    for case in [
        "cor_a_simple_bind",
        "cor_b_length_bind",
        "cor_c_pin_bind",
        "cor_d_group_bind",
        "cor_e_group_inv_length_pin",
        "cor_f_only_smooth",
        "cor_g_only_smooth_length_group",
    ] {
        let mut params = oracle["cases"][case]["params"].clone();
        params["rest_source"] = json!("BIND");
        let mesh = input_for(&oracle, &params, &oracle["bent"]);
        let bind = bound(modifier("corrective_smooth", params), &oracle["flat"]);
        let result = evaluate(&mesh, &[bind]);
        assert_points(case, &result, &expected(&oracle, case));
    }
}

#[test]
fn corrective_smooth_pins_boundary_rings() {
    let oracle = oracle();
    let mut params = oracle["cases"]["cor_c_pin_bind"]["params"].clone();
    params["rest_source"] = json!("BIND");
    let mesh = tube(&oracle, &oracle["bent"], false);
    let bind = bound(modifier("corrective_smooth", params), &oracle["flat"]);
    let result = evaluate(&mesh, &[bind]);
    let bent = points(&oracle["bent"]);
    // Rings 0 and 3 are the open ends (one face per edge): pinned, so they keep the input.
    for index in (0..6).chain(18..24) {
        let error = (result.vertices[index].co - bent[index])
            .abs()
            .max_element();
        assert!(error < 1.0e-6, "pinned vertex {index} moved by {error}");
    }
}

#[test]
fn corrective_smooth_zero_weight_vertices_stay_put() {
    let oracle = oracle();
    let mut params = oracle["cases"]["cor_d_group_bind"]["params"].clone();
    params["rest_source"] = json!("BIND");
    let mesh = input_for(&oracle, &params, &oracle["bent"]);
    let bind = bound(modifier("corrective_smooth", params), &oracle["flat"]);
    let result = evaluate(&mesh, &[bind]);
    let bent = points(&oracle["bent"]);
    for index in (3..24).step_by(4) {
        let error = (result.vertices[index].co - bent[index])
            .abs()
            .max_element();
        assert!(error < 1.0e-9, "vertex {index} moved by {error}");
    }
}

#[test]
fn corrective_smooth_orco_uses_the_mesh_rest() {
    let oracle = oracle();
    for case in ["cor_h_orco_scale1", "cor_i_orco_scale05"] {
        let params = oracle["cases"][case]["params"].clone();
        let mesh = tube(&oracle, &oracle["bent"], false);
        let result = evaluate(&mesh, &[modifier("corrective_smooth", params)]);
        assert_points(case, &result, &expected(&oracle, case));
    }
    // Scale 1 with the input as its own rest reproduces the input exactly.
    let bent = points(&oracle["bent"]);
    let mesh = tube(&oracle, &oracle["bent"], false);
    let params = oracle["cases"]["cor_h_orco_scale1"]["params"].clone();
    let result = evaluate(&mesh, &[modifier("corrective_smooth", params)]);
    for (vertex, want) in result.vertices.iter().zip(&bent) {
        assert!((vertex.co - *want).abs().max_element() < 1.0e-9);
    }
}

#[test]
fn corrective_smooth_orco_rest_is_the_stack_input_not_the_previous_modifier_output() {
    let oracle = oracle();
    let mesh = tube(&oracle, &oracle["bent"], false);
    let stack = [
        modifier(
            "laplacian_smooth",
            oracle["cases"]["lap_b_norm_vol_iter2"]["params"].clone(),
        ),
        modifier(
            "corrective_smooth",
            json!({"factor": 0.5, "iterations": 3, "scale": 0.5, "rest_source": "ORCO"}),
        ),
    ];
    let result = evaluate(&mesh, &stack);
    assert_points(
        "chain_lap_b_then_cor_orco",
        &result,
        &expected(&oracle, "chain_lap_b_then_cor_orco"),
    );
}

#[test]
fn corrective_bind_then_laplacian_chain_matches_blender() {
    let oracle = oracle();
    let mesh = tube(&oracle, &oracle["bent"], false);
    let stack = [
        bound(
            modifier(
                "corrective_smooth",
                json!({"factor": 0.5, "iterations": 4, "scale": 1.0, "rest_source": "BIND"}),
            ),
            &oracle["flat"],
        ),
        modifier(
            "laplacian_smooth",
            oracle["cases"]["lap_a_plain_iter3"]["params"].clone(),
        ),
    ];
    let result = evaluate(&mesh, &stack);
    assert_points(
        "chain_cor_bind_then_lap_a",
        &result,
        &expected(&oracle, "chain_cor_bind_then_lap_a"),
    );
}

#[test]
fn corrective_smooth_bind_errors_are_typed() {
    let oracle = oracle();
    let mesh = tube(&oracle, &oracle["bent"], false);

    let unbound = modifier(
        "corrective_smooth",
        json!({"factor": 0.5, "iterations": 2, "rest_source": "BIND"}),
    );
    let error = evaluate_modifiers(&mesh, &[unbound], SUBJECT_NAME).err().unwrap();
    assert!(matches!(error.code, ErrorCode::DependencyMissing));

    let short_coords: Vec<Value> = oracle["flat"].as_array().unwrap()[..23].to_vec();
    let mismatched = bound(
        modifier(
            "corrective_smooth",
            json!({"factor": 0.5, "iterations": 2, "rest_source": "BIND"}),
        ),
        &Value::Array(short_coords),
    );
    let error = evaluate_modifiers(&mesh, &[mismatched], SUBJECT_NAME).err().unwrap();
    assert!(matches!(error.code, ErrorCode::InvalidArgument));
}

#[test]
fn corrective_only_smooth_does_not_need_bind_data() {
    let oracle = oracle();
    let mesh = tube(&oracle, &oracle["bent"], false);
    let params = json!({
        "factor": 0.5,
        "iterations": 3,
        "scale": 1.0,
        "rest_source": "BIND",
        "use_only_smooth": true,
    });
    let result = evaluate(&mesh, &[modifier("corrective_smooth", params)]);
    assert_points(
        "cor_f_only_smooth",
        &result,
        &expected(&oracle, "cor_f_only_smooth"),
    );
}
