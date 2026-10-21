#![expect(clippy::unwrap_used, reason = "geometry tests use fixed valid meshes")]

use std::{error::Error, fs, path::Path, process::Command};

use potter_core::geom::{BoxParams, Mesh, PlaneParams, edit};
use proptest::prelude::*;
use serde_json::{Value, json};
use tempfile::{TempDir, tempdir};

#[path = "common/exchange_normals.rs"]
#[expect(
    dead_code,
    reason = "this test binary uses only the quantized normal tolerance from the shared helper"
)]
mod exchange_normals;

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
    // The stored custom normals are code pairs; the vectors are the public evaluator's decoded corners.
    let decoded = potter_core::geom::corner_normal_vectors(&mesh).unwrap();
    let mut corner_base = 0;
    let mut f6_corners = None;
    for face in &mesh.faces {
        let count = face.vertices.len();
        if face.id == 6 {
            f6_corners = Some(corner_base..corner_base + count);
        }
        corner_base += count;
    }
    let Some(f6_corners) = f6_corners else {
        panic!("face f6 is missing");
    };
    assert_eq!(decoded[f6_corners.clone()].len(), 4);
    let tolerance = exchange_normals::normal_tolerance(true);
    for corner in &decoded[f6_corners] {
        assert!(
            corner.x.abs() < tolerance
                && corner.y.abs() < tolerance
                && (corner.z - 1.0).abs() < tolerance,
            "{corner:?}"
        );
    }
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

/// Blender's `Mesh.set_sharp_from_angle` (`rna_mesh_api.cc`) from a cube with face 4 flat, for every angle in
/// the oracle: 180 degrees makes every face smooth with no sharp edges, 0 makes every face flat, and other
/// angles mark the edges whose face normals differ by more than the angle. `sharp_face` is removed when every
/// face is smooth.
#[test]
fn whole_mesh_sharp_angle_matches_blender_set_sharp_from_angle() {
    let oracle = blender_oracle();
    for (degrees, expected) in oracle["sharp_from_angle"].as_object().unwrap() {
        let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
        edit::apply(
            &mut mesh,
            "shade_flat",
            &json!({"elements":{"domain":"face","ids":["f4"]}}),
        )
        .unwrap();
        let angle = degrees.parse::<f64>().unwrap().to_radians();
        edit::apply(&mut mesh, "mark_sharp_by_angle", &json!({ "angle": angle })).unwrap();
        let actual = sharp_state(&mesh);
        for field in [
            "faces",
            "edges",
            "smooth",
            "sharp_edges",
            "sharp_face_present",
        ] {
            assert_eq!(actual[field], expected[field], "{degrees} degrees: {field}");
        }
        mesh.validate().unwrap();
    }
}

/// Blender's `mesh.set_sharpness_by_angle` (edit mode, every edge selected) from the default cube, for every
/// angle in the oracle. Adjacent faces of a cube meet at 90 degrees, so 30 degrees marks every edge and 120
/// marks none.
#[test]
fn selected_edges_sharp_angle_matches_blender_set_sharpness_by_angle() {
    let oracle = blender_oracle();
    for (degrees, expected) in oracle["sharpness_by_angle"].as_object().unwrap() {
        let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
        let edges = mesh
            .edges
            .iter()
            .map(|edge| format!("e{}", edge.id))
            .collect::<Vec<_>>();
        let angle = degrees.parse::<f64>().unwrap().to_radians();
        edit::apply(
            &mut mesh,
            "mark_sharp_by_angle",
            &json!({"elements":{"domain":"edge","ids":edges},"angle":angle}),
        )
        .unwrap();
        let actual = sharp_state(&mesh);
        for field in ["faces", "edges", "smooth", "sharp_edges"] {
            assert_eq!(actual[field], expected[field], "{degrees} degrees: {field}");
        }
        mesh.validate().unwrap();
    }
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
        let uv_entries = mesh["attributes"]["uv_layers"]["layers"][0]["faces"]
            .as_array()
            .unwrap();
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

/// The flat pattern of a mesh's `shade_smooth` values: `1` marks a flat face, in storage order.
fn flat_pattern(mesh: &Mesh) -> String {
    mesh.faces
        .iter()
        .map(|face| {
            let key = format!("f{}", face.id);
            if mesh.attributes["shade_smooth"]["values"][key.as_str()] == json!(false) {
                '1'
            } else {
                '0'
            }
        })
        .collect()
}

/// The state the Blender oracle records: topology counts, smooth and sharp counts, and whether `sharp_face`
/// exists. A flat face makes it exist; otherwise it is the recorded presence flag.
fn sharp_state(mesh: &Mesh) -> Value {
    let shading = mesh.attributes.get("shade_smooth");
    let values = shading.and_then(|attribute| attribute["values"].as_object());
    let smooth = values.map_or(0, |values| {
        values
            .values()
            .filter(|value| *value == &json!(true))
            .count()
    });
    let any_flat = values.is_some_and(|values| values.values().any(|value| value == &json!(false)));
    let recorded =
        shading.and_then(|attribute| attribute.get("sharp_face_present")) == Some(&json!(true));
    let sharp_edges = mesh.attributes.get("sharp").map_or(0, |attribute| {
        attribute["values"].as_object().map_or(0, |values| {
            values
                .values()
                .filter(|value| *value == &json!(true))
                .count()
        })
    });
    json!({
        "faces": mesh.faces.len(),
        "edges": mesh.edges.len(),
        "smooth": smooth,
        "sharp_edges": sharp_edges,
        "sharp_face_present": any_flat || recorded,
    })
}

/// Blender 5.2.2 results from `tests/fixtures/remesh-reproject/generate.py`, which runs `object.voxel_remesh`,
/// `Mesh.set_sharp_from_angle` and `mesh.set_sharpness_by_angle` headless.
fn blender_oracle() -> Value {
    serde_json::from_str(include_str!("fixtures/remesh-reproject/expected.json")).unwrap()
}

/// Runs the voxel remesh on `mesh` and checks the result against the Blender oracle case `case`.
fn assert_remesh_matches_blender(case: &str, mesh: &mut Mesh) {
    let oracle = blender_oracle();
    let expected = &oracle["remesh"][case];
    let size = oracle["remesh_voxel_size"].as_f64().unwrap();
    let result = edit::voxel_remesh(mesh, &[], &mut edit::VertexWeights::new(), size)
        .unwrap()
        .unwrap();
    assert!(
        result.dropped.is_empty(),
        "{case}: dropped {:?}",
        result.dropped
    );
    assert_eq!(
        json!(mesh.vertices.len()),
        expected["vertices"],
        "{case}: vertices"
    );
    assert_eq!(json!(mesh.faces.len()), expected["faces"], "{case}: faces");
    let pattern = flat_pattern(mesh);
    assert_eq!(
        json!(pattern),
        expected["flat_pattern"],
        "{case}: flat faces"
    );
    assert_eq!(
        json!(!pattern.contains('1')),
        expected["all_smooth"],
        "{case}: all smooth"
    );
    assert_eq!(
        sharp_state(mesh)["sharp_face_present"],
        expected["sharp_face_present"],
        "{case}: sharp_face presence"
    );
    // Storage order (`mesh_apply_spatial_organization`): each face's material, and the leading vertices.
    let materials: Vec<u32> = mesh.faces.iter().map(|face| face.material_index).collect();
    assert_eq!(
        json!(materials),
        expected["material_indices"],
        "{case}: face material order"
    );
    for (index, expected_vertex) in expected["first_vertices"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let co = mesh.vertices[index].co;
        for (axis, value) in expected_vertex.as_array().unwrap().iter().enumerate() {
            let value = value.as_f64().unwrap();
            assert!(
                (co[axis] - value).abs() <= 1.0e-5,
                "{case}: vertex {index} axis {axis} is {}, Blender has {value}",
                co[axis]
            );
        }
    }
    mesh.validate().unwrap();
}

/// Blender's default cube is flat, so the voxel remesh of the default cube is the common case.
#[test]
fn voxel_remesh_of_a_flat_cube_matches_blender() {
    let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
    assert_remesh_matches_blender("flat_cube", &mut mesh);
}

/// Face 4 flat and the rest smooth: each new face takes the `sharp_face` of its nearest source triangle.
#[test]
fn voxel_remesh_reprojects_sharp_face_from_nearest_source_triangles() {
    let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
    edit::apply(
        &mut mesh,
        "shade_smooth",
        &json!({"elements":{"domain":"face","ids":["f0","f1","f2","f3","f5"]}}),
    )
    .unwrap();
    assert_remesh_matches_blender("mixed_cube", &mut mesh);
}

/// A smooth cube has no `sharp_face`, and its remesh output is smooth without one.
#[test]
fn voxel_remesh_of_an_all_smooth_cube_stays_smooth() {
    let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
    let faces = mesh
        .faces
        .iter()
        .map(|face| format!("f{}", face.id))
        .collect::<Vec<_>>();
    edit::apply(
        &mut mesh,
        "shade_smooth",
        &json!({"elements":{"domain":"face","ids":faces}}),
    )
    .unwrap();
    assert_remesh_matches_blender("all_smooth_cube", &mut mesh);
}

/// Two materials, faces 3..5 in the second, all smooth. Blender's `use_smooth` assignment keeps `sharp_face`
/// present (all false), so the input records that presence as `record_sharp_face` does. The voxel remesh
/// stores its faces grouped by material (`mesh_apply_spatial_organization`), which the oracle pins.
#[test]
fn voxel_remesh_of_a_two_material_cube_matches_blender_storage_order() {
    let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
    let faces = mesh
        .faces
        .iter()
        .map(|face| format!("f{}", face.id))
        .collect::<Vec<_>>();
    edit::apply(
        &mut mesh,
        "shade_smooth",
        &json!({"elements":{"domain":"face","ids":faces}}),
    )
    .unwrap();
    mesh.attributes
        .get_mut("shade_smooth")
        .unwrap()
        .as_object_mut()
        .unwrap()
        .insert("sharp_face_present".to_owned(), json!(true));
    for face in mesh.faces.iter_mut().skip(3) {
        face.material_index = 1;
    }
    assert_remesh_matches_blender("two_material_cube", &mut mesh);
}
