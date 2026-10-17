//! Bevel on UV layers. Square-profile even-segment corners are built like Blender's, so they are
//! never rejected with an empty list; EDGES bevels that the vertex-mesh construction does not cover
//! are rejected with `modifier.bevel.uv_profile` and a non-empty `rejected_corners`.

use std::error::Error;

use glam::DVec3;
use potter_core::{
    error::{ErrorCode, PotError},
    geom::{Mesh, modifiers::evaluate_modifiers},
    model::{Id, Modifier, ModifierRuntime},
};
use serde_json::{Value, json};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

/// The unit cube of the Blender oracle cases: vertex positions and quad faces.
const CUBE_VERTICES: [[f64; 3]; 8] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
];
const CUBE_FACES: [[usize; 4]; 6] = [
    [0, 3, 2, 1],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [3, 7, 6, 2],
    [0, 4, 7, 3],
    [1, 2, 6, 5],
];

fn modifier(params: &Value) -> TestResult<Modifier> {
    Ok(Modifier {
        id: Id::new("bevel_uv")?,
        modifier_type: "bevel".to_owned(),
        name: "bevel".to_owned(),
        enabled: true,
        show_viewport: true,
        show_render: true,
        params: params
            .as_object()
            .ok_or("bevel params must be an object")?
            .clone(),
        binding_data: None,
        runtime: ModifierRuntime::default(),
    })
}

fn evaluate_bevel(mesh: &Mesh, params: &Value) -> TestResult<Result<Mesh, PotError>> {
    Ok(evaluate_modifiers(mesh, &[modifier(params)?], "BevelCube"))
}

/// A unit cube with one UV layer, `UVMap`, whose corner values depend only on the corner's vertex.
fn uv_cube() -> TestResult<Mesh> {
    let positions = CUBE_VERTICES
        .iter()
        .map(|&[x, y, z]| DVec3::new(x, y, z))
        .collect();
    let faces = CUBE_FACES.iter().map(|face| face.to_vec()).collect();
    let mut mesh = Mesh::from_positions_and_faces(positions, faces)?;
    let values: Vec<Value> = CUBE_FACES
        .iter()
        .flatten()
        .map(|&vertex| {
            let [x, y, z] = CUBE_VERTICES[vertex];
            json!([0.2 + 0.5 * x + 0.1 * y, 0.3 + 0.6 * z])
        })
        .collect();
    let faces: Vec<Value> = mesh
        .faces
        .iter()
        .zip(values.chunks(4))
        .map(|(face, uv)| json!({"face_id": face.id, "uv": uv}))
        .collect();
    mesh.attributes.insert(
        "uv_layers".to_owned(),
        json!({"layers": [{"name": "UVMap", "faces": faces}], "active": "UVMap", "render": "UVMap"}),
    );
    Ok(mesh)
}

/// Asserts the `modifier.bevel.uv_profile` rejection with a non-empty `rejected_corners` list, and
/// returns its details.
fn uv_profile_rejection(error: &PotError) -> TestResult<&Value> {
    assert_eq!(
        error.code,
        ErrorCode::UnsupportedFeature,
        "{}",
        error.message
    );
    assert_eq!(error.details["feature_id"], "modifier.bevel.uv_profile");
    let rejected = error.details["rejected_corners"]
        .as_array()
        .ok_or("rejected_corners must be an array")?;
    assert!(
        !rejected.is_empty(),
        "rejected_corners must name the uncovered corners: {}",
        error.message
    );
    Ok(&error.details)
}

#[test]
fn edges_bevel_with_uv_layers_rejects_partial_edge_selection() -> TestResult {
    let mut mesh = uv_cube()?;
    // Only the bottom face's vertices are weighted, so its four edges are selected and the vertical
    // edges are not: each bottom corner has three edges, of which two are beveled.
    mesh.attributes.insert(
        "bottom".to_owned(),
        json!([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    );
    let params = json!({
        "width": 0.1,
        "segments": 3,
        "limit_method": "VGROUP",
        "vertex_group": "bottom",
    });
    let Err(error) = evaluate_bevel(&mesh, &params)? else {
        return Err("a partial edge selection on UV layers must be rejected".into());
    };
    let details = uv_profile_rejection(&error)?;
    assert_eq!(details["parameter"], "segments");
    assert_eq!(details["rejected_corners"][0]["valence"], 3);
    assert_eq!(details["rejected_corners"][0]["selected_edges"], 2);
    assert!(
        error.message.starts_with(
            "bevel with UV layers supports only three-edge cube-like corners with every edge selected; corner at vertex 0 has valence 3 / selected 2 edges"
        ),
        "unexpected message: {}",
        error.message
    );
    Ok(())
}

#[test]
fn edges_bevel_with_uv_layers_rejects_square_inward_profile_by_name() -> TestResult {
    let mesh = uv_cube()?;
    for segments in [2, 3] {
        let params =
            json!({"width": 0.1, "segments": segments, "profile": 0.0, "limit_method": "NONE"});
        let Err(error) = evaluate_bevel(&mesh, &params)? else {
            return Err(
                format!("profile 0 on UV layers must be rejected (segments {segments})").into(),
            );
        };
        let details = uv_profile_rejection(&error)?;
        assert_eq!(details["parameter"], "profile", "segments {segments}");
        assert!(
            error.message.contains("profile 0"),
            "unexpected message: {}",
            error.message
        );
    }
    Ok(())
}

#[test]
fn square_profile_even_segments_bevel_uv_cube_without_empty_rejection() -> TestResult {
    let mesh = uv_cube()?;
    // Vertex and face counts of Blender 5.2.2 for the same bevel (profile 1.0, width 0.1).
    for (segments, vertices, faces) in [(2, 56, 54), (3, 96, 98), (4, 152, 150)] {
        let params = json!({
            "width": 0.1,
            "segments": segments,
            "profile": 1.0,
            "limit_method": "NONE",
        });
        let output = evaluate_bevel(&mesh, &params)??;
        assert_eq!(
            output.vertices.len(),
            vertices,
            "vertices at segments {segments}"
        );
        assert_eq!(output.faces.len(), faces, "faces at segments {segments}");
    }
    Ok(())
}
