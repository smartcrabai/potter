#![expect(clippy::unwrap_used, reason = "geometry tests use fixed valid meshes")]

use std::{error::Error, fs, path::Path, process::Command};

use potter::geom::{BoxParams, Mesh, PlaneParams, edit};
use proptest::prelude::*;
use serde_json::{Value, json};
use tempfile::{TempDir, tempdir};

fn create_cli_scene() -> Result<(TempDir, std::path::PathBuf), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let output = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("init")
        .arg(&scene)
        .arg("--json")
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    Ok((directory, scene))
}

fn apply_cli(scene: &Path, operations: &Value) -> Result<Value, Box<dyn Error>> {
    let batch = scene.with_extension("operations.json");
    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version":1,
            "base_revision":0,
            "operations":operations
        }))?,
    )?;
    let output = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("apply")
        .arg(scene)
        .arg("--file")
        .arg(batch)
        .arg("--json")
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    Ok(serde_json::from_slice(&output.stdout)?)
}

proptest! {
    #[test]
    fn position_box_selector_matches_brute_force(
        x_low in -1.2_f64..1.0,
        x_high in -1.0_f64..1.2,
        y_low in -1.2_f64..1.0,
        y_high in -1.0_f64..1.2,
        z_low in -1.2_f64..1.0,
        z_high in -1.0_f64..1.2,
    ) {
        let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
        let before = mesh.clone();
        edit::apply(
            &mut mesh,
            "transform_elements",
            &json!({
                "elements": {
                    "domain": "vertex",
                    "selector": {
                        "type": "position_box",
                        "min": [x_low.min(x_high), y_low.min(y_high), z_low.min(z_high)],
                        "max": [x_low.max(x_high), y_low.max(y_high), z_low.max(z_high)]
                    }
                },
                "translation": [0.125, 0.0, 0.0]
            }),
        ).unwrap();

        for original in &before.vertices {
            let expected = (0..3).all(|axis| {
                original.co[axis] >= [x_low.min(x_high), y_low.min(y_high), z_low.min(z_high)][axis]
                    && original.co[axis] <= [x_low.max(x_high), y_low.max(y_high), z_low.max(z_high)][axis]
            });
            let actual = mesh.vertex(original.id).unwrap().co != original.co;
            prop_assert_eq!(actual, expected, "vertex v{} selector mismatch", original.id);
        }
    }
}

#[test]
fn typed_attributes_create_update_delete_and_interpolate_on_subdivision() {
    let mut mesh = Mesh::plane(PlaneParams::default()).unwrap();
    let original_vertices = mesh.vertices.clone();
    let original_edges = mesh.edges.clone();
    edit::apply(
        &mut mesh,
        "attribute_create",
        &json!({"name":"weight", "domain":"point", "type":"float"}),
    )
    .unwrap();
    for vertex in &original_vertices {
        edit::apply(
            &mut mesh,
            "attribute_update",
            &json!({
                "name":"weight",
                "elements":{"domain":"vertex","ids":[format!("v{}", vertex.id)]},
                "value":f64::from(vertex.id)
            }),
        )
        .unwrap();
    }
    assert_eq!(mesh.attributes["weight"]["values"]["v0"], json!(0.0));
    edit::apply(
        &mut mesh,
        "subdivide",
        &json!({"elements":{"domain":"face","ids":["f0"]}}),
    )
    .unwrap();

    let values = mesh.attributes["weight"]["values"].as_object().unwrap();
    for vertex in mesh.vertices.iter().filter(|vertex| vertex.id >= 4) {
        let edge = original_edges.iter().find(|edge| {
            let first = original_vertices
                .iter()
                .find(|candidate| candidate.id == edge.vertices[0])
                .unwrap()
                .co;
            let second = original_vertices
                .iter()
                .find(|candidate| candidate.id == edge.vertices[1])
                .unwrap()
                .co;
            vertex.co.distance((first + second) * 0.5) <= 1.0e-10
        });
        if let Some(edge) = edge {
            let expected = f64::midpoint(f64::from(edge.vertices[0]), f64::from(edge.vertices[1]));
            assert!(
                (values[&format!("v{}", vertex.id)].as_f64().unwrap() - expected).abs() <= 1.0e-10,
                "new vertex v{} should receive interpolated value {expected}",
                vertex.id
            );
        }
    }

    edit::apply(
        &mut mesh,
        "attribute_delete",
        &json!({"name":"weight", "elements":{"domain":"vertex","ids":["v0"]}}),
    )
    .unwrap();
    assert!(mesh.attributes["weight"]["values"].get("v0").is_none());
    edit::apply(&mut mesh, "attribute_delete", &json!({"name":"weight"})).unwrap();
    assert!(mesh.attributes.get("weight").is_none());
}

#[test]
fn proportional_transform_falls_off_with_euclidean_distance() {
    let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
    let original = mesh.clone();
    let center = original.vertices[0].co;
    edit::apply(
        &mut mesh,
        "transform_elements",
        &json!({
            "elements":{"domain":"vertex","ids":["v0"]},
            "translation":[1.0,0.0,0.0],
            "proportional":{"radius":3.0,"falloff":"linear","connected_only":false}
        }),
    )
    .unwrap();
    let center_move = mesh.vertex(0).unwrap().co.x - center.x;
    assert!((center_move - 1.0).abs() <= 1.0e-12);
    let remote = original
        .vertices
        .iter()
        .max_by(|left, right| {
            left.co
                .distance(center)
                .total_cmp(&right.co.distance(center))
        })
        .unwrap();
    assert_eq!(mesh.vertex(remote.id).unwrap().co, remote.co);
    let adjacent = original
        .vertices
        .iter()
        .filter(|vertex| vertex.id != 0)
        .min_by(|left, right| {
            left.co
                .distance(center)
                .total_cmp(&right.co.distance(center))
        })
        .unwrap();
    let adjacent_move = mesh.vertex(adjacent.id).unwrap().co.x - adjacent.co.x;
    assert!(adjacent_move > 0.0 && adjacent_move < 1.0);
}

#[test]
fn poke_splits_selected_face_into_center_fan() {
    let mut mesh = Mesh::plane(PlaneParams::default()).unwrap();
    let initial_vertices = mesh.vertices.len();
    let result = edit::apply(
        &mut mesh,
        "poke",
        &json!({"elements":{"domain":"face","ids":["f0"]}}),
    )
    .unwrap();

    mesh.validate().unwrap();
    assert_eq!(mesh.vertices.len(), initial_vertices + 1);
    assert_eq!(mesh.faces.len(), 4);
    assert_eq!(result["created"]["faces"].as_array().unwrap().len(), 4);
    assert_eq!(result["deleted"]["faces"], json!(["f0"]));
}

#[test]
fn individual_extrusion_and_face_corner_shading_attributes_are_persisted() {
    let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
    let face_ids = mesh
        .faces
        .iter()
        .map(|face| format!("f{}", face.id))
        .collect::<Vec<_>>();
    edit::apply(
        &mut mesh,
        "extrude",
        &json!({
            "elements":{"domain":"face","ids":face_ids},
            "distance":0.25,
            "individual":true
        }),
    )
    .unwrap();
    assert_eq!(mesh.vertices.len(), 32);
    assert_eq!(mesh.faces.len(), 30);
    mesh.validate().unwrap();

    edit::apply(
        &mut mesh,
        "shade_smooth",
        &json!({"elements":{"domain":"face","ids":["f6"]}}),
    )
    .unwrap();
    assert_eq!(mesh.attributes["shade_smooth"]["values"]["f6"], json!(true));
    edit::apply(
        &mut mesh,
        "set_custom_normals",
        &json!({
            "elements":{"domain":"face","ids":["f6"]},
            "normals":{"f6":[[0,0,3],[0,0,3],[0,0,3],[0,0,3]]}
        }),
    )
    .unwrap();
    assert_eq!(mesh.attributes["custom_normal"]["domain"], json!("corner"));
    assert_eq!(
        mesh.attributes["custom_normal"]["values"]["f6"],
        json!([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0]
        ])
    );
}

#[test]
fn sliding_uses_connected_topology_and_angle_marks_sharp_edges() {
    let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
    let edge_endpoints = mesh.edges[0].vertices;
    let before = mesh.clone();
    edit::apply(
        &mut mesh,
        "edge_slide",
        &json!({"elements":{"domain":"edge","ids":["e0"]},"factor":0.5}),
    )
    .unwrap();
    assert!(
        edge_endpoints
            .iter()
            .any(|id| mesh.vertex(*id).unwrap().co != before.vertex(*id).unwrap().co)
    );
    edit::apply(&mut mesh, "mark_sharp_by_angle", &json!({"angle":0.5})).unwrap();
    let sharp = mesh.attributes["sharp"]["values"].as_object().unwrap();
    assert!(sharp.values().any(|value| value == &json!(true)));
    mesh.validate().unwrap();
}

#[test]
fn spin_extrudes_selected_face_around_axis_in_steps() {
    let mut mesh = Mesh::plane(PlaneParams::default()).unwrap();
    let result = edit::apply(
        &mut mesh,
        "spin",
        &json!({
            "elements":{"domain":"face","ids":["f0"]},
            "axis":[0,0,1],
            "angle":std::f64::consts::FRAC_PI_2,
            "center":[0,0,0],
            "steps":2
        }),
    )
    .unwrap();

    mesh.validate().unwrap();
    assert_eq!(mesh.vertices.len(), 12);
    assert_eq!(mesh.faces.len(), 9);
    assert_eq!(result["created"]["faces"].as_array().unwrap().len(), 9);
    assert_eq!(result["deleted"]["faces"], json!(["f0"]));
}

#[test]
fn cli_resolves_position_selectors_for_attribute_edits_and_transforms() -> Result<(), Box<dyn Error>>
{
    let (_directory, scene) = create_cli_scene()?;
    apply_cli(
        &scene,
        &json!([
            {"op":"node.create","id":"body","kind":"box","params":{"size":2}},
            {"op":"mesh.attribute_create","target":{"id":"body"},"name":"weight","domain":"point","type":"float"},
            {
                "op":"mesh.attribute_update",
                "target":{"id":"body"},
                "elements":{"domain":"vertex","selector":{"type":"position_box","min":[-1.01,-1.1,-1.1],"max":[-0.99,1.1,1.1]}},
                "name":"weight",
                "value":0.75
            },
            {
                "op":"mesh.transform_elements",
                "target":{"id":"body"},
                "elements":{"domain":"vertex","selector":{"type":"position_box","min":[-1.01,-1.1,-1.1],"max":[-0.99,1.1,1.1]}},
                "translation":[0.5,0,0]
            }
        ]),
    )?;
    let document: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let mesh = &document["data_blocks"]["body_mesh"]["mesh"];
    let values = mesh["attributes"]["weight"]["values"].as_object().unwrap();
    for vertex in mesh["vertices"].as_array().unwrap() {
        let id = vertex["id"].as_u64().unwrap();
        let x = vertex["co"][0].as_f64().unwrap();
        if values[&format!("v{id}")] == json!(0.75) {
            assert!((x + 0.5).abs() <= 1.0e-12);
        } else {
            assert!((x - 1.0).abs() <= 1.0 || (x + 1.0).abs() <= 1.0);
        }
    }
    Ok(())
}

#[test]
fn cli_joins_world_transformed_meshes_and_separates_selected_faces() -> Result<(), Box<dyn Error>> {
    let (_directory, scene) = create_cli_scene()?;
    apply_cli(
        &scene,
        &json!([
            {"op":"node.create","id":"body","kind":"box","params":{"size":2}},
            {"op":"node.create","id":"other","kind":"box","params":{"size":2},"transform":{"translation":[3,0,0]}},
            {"op":"node.join","targets":["body","other"]},
            {"op":"mesh.separate","target":{"id":"body"},"mode":"selection","elements":{"domain":"face","ids":["f0"]}}
        ]),
    )?;
    let document: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    assert!(document["nodes"].get("other").is_none());
    let nodes = document["nodes"].as_object().unwrap();
    assert_eq!(nodes.len(), 2);
    let mut face_counts = Vec::new();
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    for node in nodes.values() {
        let data_id = node["data"].as_str().unwrap();
        let mesh = &document["data_blocks"][data_id]["mesh"];
        face_counts.push(mesh["faces"].as_array().unwrap().len());
        for vertex in mesh["vertices"].as_array().unwrap() {
            let x = vertex["co"][0].as_f64().unwrap();
            min_x = min_x.min(x);
            max_x = max_x.max(x);
        }
    }
    face_counts.sort_unstable();
    assert_eq!(face_counts, [1, 11]);
    assert!((min_x + 1.0).abs() <= 1.0e-12);
    assert!((max_x - 4.0).abs() <= 1.0e-12);
    Ok(())
}

#[test]
fn cli_angle_and_conformal_unwrap_preserve_planar_grid_similarity() -> Result<(), Box<dyn Error>> {
    for method in ["angle_based", "conformal"] {
        let (_directory, scene) = create_cli_scene()?;
        apply_cli(
            &scene,
            &json!([
                {
                    "op":"node.create",
                    "id":"grid",
                    "kind":"grid",
                    "params":{"size":2.0,"x_subdivisions":2,"y_subdivisions":2}
                },
                {"op":"uv.unwrap","target":{"id":"grid"},"method":method}
            ]),
        )?;
        let document: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
        let mesh = &document["data_blocks"]["grid_mesh"]["mesh"];
        let vertices = mesh["vertices"].as_array().unwrap();
        let mut uv_by_vertex: Vec<Option<[f64; 2]>> = vec![None; vertices.len()];
        let uv_entries = mesh["attributes"]["uv_map"].as_array().unwrap();
        for face in mesh["faces"].as_array().unwrap() {
            let face_id = face["id"].as_u64().unwrap();
            let entry = uv_entries
                .iter()
                .find(|entry| entry["face_id"].as_u64() == Some(face_id))
                .unwrap();
            let face_vertices = face["v"].as_array().unwrap();
            for (corner, vertex_id) in face_vertices.iter().enumerate() {
                let index = usize::try_from(vertex_id.as_u64().unwrap())?;
                let uv = [
                    entry["uv"][corner][0].as_f64().unwrap(),
                    entry["uv"][corner][1].as_f64().unwrap(),
                ];
                if let Some(previous) = uv_by_vertex[index] {
                    assert!(
                        (previous[0] - uv[0]).hypot(previous[1] - uv[1]) < 1.0e-8,
                        "{method} produced a seam without a marked edge"
                    );
                } else {
                    uv_by_vertex[index] = Some(uv);
                }
            }
        }

        let positions = vertices
            .iter()
            .map(|vertex| {
                let co = vertex["co"].as_array().unwrap();
                [co[0].as_f64().unwrap(), co[1].as_f64().unwrap()]
            })
            .collect::<Vec<_>>();
        let points = uv_by_vertex
            .iter()
            .enumerate()
            .filter_map(|(index, uv)| uv.map(|uv| (positions[index], uv)))
            .collect::<Vec<_>>();
        let mut anchors = (0, 1);
        let mut longest = 0.0_f64;
        for left in 0..points.len() {
            for right in left + 1..points.len() {
                let distance = (points[right].0[0] - points[left].0[0])
                    .hypot(points[right].0[1] - points[left].0[1]);
                if distance > longest {
                    longest = distance;
                    anchors = (left, right);
                }
            }
        }
        let (source_origin, uv_origin) = points[anchors.0];
        let (source_end, uv_end) = points[anchors.1];
        let source_delta = [
            source_end[0] - source_origin[0],
            source_end[1] - source_origin[1],
        ];
        let uv_delta = [uv_end[0] - uv_origin[0], uv_end[1] - uv_origin[1]];
        let denominator = source_delta[0] * source_delta[0] + source_delta[1] * source_delta[1];
        let cosine = (source_delta[0] * uv_delta[0] + source_delta[1] * uv_delta[1]) / denominator;
        let sine = (source_delta[0] * uv_delta[1] - source_delta[1] * uv_delta[0]) / denominator;
        let maximum_error = points
            .iter()
            .map(|(position, uv)| {
                let dx = position[0] - source_origin[0];
                let dy = position[1] - source_origin[1];
                let predicted = [
                    uv_origin[0] + cosine * dx - sine * dy,
                    uv_origin[1] + sine * dx + cosine * dy,
                ];
                (predicted[0] - uv[0]).hypot(predicted[1] - uv[1])
            })
            .fold(0.0, f64::max);
        assert!(
            maximum_error < 1.0e-8,
            "{method} failed planar similarity: {maximum_error}"
        );
    }
    Ok(())
}
