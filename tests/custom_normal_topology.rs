#![expect(clippy::unwrap_used, reason = "topology tests use fixed valid meshes")]

use potter_core::geom::{BoxParams, Mesh, corner_normal_vectors, edit};
use serde_json::json;

#[test]
fn custom_normals_survive_extrude_then_delete_with_one_row_per_face() {
    let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
    edit::apply(
        &mut mesh,
        "set_custom_normals",
        &json!({
            "elements": {"domain": "face", "ids": ["f0"]},
            "normals": {
                "f0": [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
            }
        }),
    )
    .unwrap();
    edit::apply(
        &mut mesh,
        "extrude",
        &json!({"elements": {"domain": "face", "ids": ["f0"]}, "distance": 0.5}),
    )
    .unwrap();
    edit::apply(
        &mut mesh,
        "delete",
        &json!({"elements": {"domain": "face", "ids": ["f1"]}}),
    )
    .unwrap();

    let rows = mesh.attributes["custom_normal"]["values"]
        .as_object()
        .unwrap();
    assert_eq!(rows.len(), mesh.faces.len());
    for face in &mesh.faces {
        assert!(
            rows.contains_key(&format!("f{}", face.id)),
            "face f{} has no custom normal row",
            face.id
        );
    }
    let corners: usize = mesh.faces.iter().map(|face| face.vertices.len()).sum();
    assert_eq!(corner_normal_vectors(&mesh).unwrap().len(), corners);
}
