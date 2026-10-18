#![expect(
    clippy::unwrap_used,
    reason = "Screw parity fixtures contain fixed valid data"
)]

use std::collections::{BTreeMap, HashMap, HashSet};

use glam::DVec3;
use potter_core::{
    eval::{EvaluationContext, EvaluationPurpose, Snapshot},
    geom::Mesh,
    model::{DataBlock, Id, Material, Node, SceneDoc},
};
use serde_json::{Map, Value, json};

const SCREW_FIXTURES: &str = include_str!("fixtures/construction_p/screw/screw.json");
const POSITION_TOLERANCE: f32 = 1.0e-5;

fn float32_close(actual: f64, expected: f64) -> bool {
    (actual as f32 - expected as f32).abs() <= POSITION_TOLERANCE
}

fn assert_vec_close(actual: &[f64], expected: &[Value], context: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{context} vector dimensions differ"
    );
    for (axis, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            float32_close(*actual, expected.as_f64().unwrap()),
            "{context}[{axis}] differs: actual={actual} expected={expected}"
        );
    }
}

fn install_fixture_colors(mesh: &mut Mesh, input: &Value) {
    let colors = input["colors"].as_array().unwrap();
    if colors.is_empty() {
        return;
    }
    let mut encodings = Map::new();
    for layer in colors {
        let name = layer["name"].as_str().unwrap();
        let mut values = Map::new();
        match layer["domain"].as_str().unwrap() {
            "POINT" => {
                for (vertex, value) in mesh
                    .vertices
                    .iter()
                    .zip(layer["values"].as_array().unwrap())
                {
                    values.insert(format!("v{}", vertex.id), value.clone());
                }
            }
            "CORNER" => {
                let input = layer["values"].as_array().unwrap();
                let mut start = 0;
                for face in &mesh.faces {
                    let end = start + face.vertices.len();
                    values.insert(
                        format!("f{}", face.id),
                        Value::Array(input[start..end].to_vec()),
                    );
                    start = end;
                }
            }
            domain => panic!("unexpected fixture color domain {domain}"),
        }
        let domain = if layer["domain"] == "POINT" {
            "point"
        } else {
            "corner"
        };
        mesh.attributes.insert(
            name.to_owned(),
            json!({"domain":domain, "type":"color", "values":values}),
        );
        encodings.insert(name.to_owned(), layer["data_type"].clone());
    }
    mesh.attributes
        .insert("color_layers".to_owned(), input["color_layers"].clone());
    mesh.attributes.insert(
        "blender_metadata".to_owned(),
        json!({"color_data_types": encodings}),
    );
}

fn install_fixture_normals(mesh: &mut Mesh, input: &Value) {
    let Some(normals) = input["custom_normals"]
        .as_array()
        .filter(|normals| !normals.is_empty())
    else {
        return;
    };
    let values: Map<String, Value> = mesh
        .faces
        .iter()
        .zip(normals)
        .map(|(face, row)| (format!("f{}", face.id), row.clone()))
        .collect();
    let ids: Vec<String> = mesh
        .faces
        .iter()
        .map(|face| format!("f{}", face.id))
        .collect();
    potter_core::geom::edit::apply(
        mesh,
        "set_custom_normals",
        &json!({"elements":{"domain":"face","ids":ids},"normals":values}),
    )
    .unwrap();
}

fn actual_color_values(mesh: &Mesh, name: &str) -> Vec<Value> {
    let attribute = &mesh.attributes[name];
    let values = attribute["values"].as_object().unwrap();
    match attribute["domain"].as_str().unwrap() {
        "point" => mesh
            .vertices
            .iter()
            .map(|vertex| values[&format!("v{}", vertex.id)].clone())
            .collect(),
        "corner" => mesh
            .faces
            .iter()
            .flat_map(|face| values[&format!("f{}", face.id)].as_array().unwrap().clone())
            .collect(),
        domain => panic!("unexpected canonical color domain {domain}"),
    }
}

fn blender_uv_to_potter(value: &Value) -> Value {
    let uv = value.as_array().unwrap();
    json!([uv[0].as_f64().unwrap(), 1.0 - uv[1].as_f64().unwrap()])
}

fn install_fixture_uv_layers(mesh: &mut Mesh, input: &Value) {
    let source_layers = input["uv_layers"].as_array().unwrap();
    if source_layers.is_empty() {
        return;
    }
    let layers: Vec<Value> = source_layers
        .iter()
        .map(|layer| {
            let values = layer["values"].as_array().unwrap();
            let mut start = 0;
            let faces: Vec<Value> = mesh
                .faces
                .iter()
                .map(|face| {
                    let end = start + face.vertices.len();
                    let uv: Vec<Value> = values[start..end]
                        .iter()
                        .map(blender_uv_to_potter)
                        .collect();
                    start = end;
                    json!({"face_id": face.id, "uv": uv})
                })
                .collect();
            json!({"name": layer["name"], "faces": faces})
        })
        .collect();
    mesh.attributes.insert(
        "uv_layers".to_owned(),
        json!({
            "layers": layers,
            "active": input.get("uv_active").cloned().unwrap_or(Value::Null),
            "render": input.get("uv_render").cloned().unwrap_or(Value::Null),
        }),
    );
}

fn source_mesh(case: &Value) -> Mesh {
    let input = &case["input"];
    let mut mesh = Mesh::new();
    let vertex_ids: Vec<u32> = input["vertices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|vertex| {
            mesh.insert_vertex(DVec3::new(
                vertex[0].as_f64().unwrap(),
                vertex[1].as_f64().unwrap(),
                vertex[2].as_f64().unwrap(),
            ))
            .unwrap()
        })
        .collect();
    for edge in input["edges"].as_array().unwrap() {
        mesh.insert_edge([
            vertex_ids[edge[0].as_u64().unwrap() as usize],
            vertex_ids[edge[1].as_u64().unwrap() as usize],
        ])
        .unwrap();
    }
    let mut face_smooth = Map::new();
    for face in input["faces"].as_array().unwrap() {
        let vertices = face["vertices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|index| vertex_ids[index.as_u64().unwrap() as usize])
            .collect();
        let face_id = mesh
            .insert_face(vertices, face["material"].as_u64().unwrap() as u32)
            .unwrap();
        face_smooth.insert(
            format!("f{face_id}"),
            json!(face["smooth"].as_bool().unwrap()),
        );
    }
    let sharp_edges: HashSet<[usize; 2]> = input["sharp_edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|edge| {
            let mut pair = [
                edge[0].as_u64().unwrap() as usize,
                edge[1].as_u64().unwrap() as usize,
            ];
            pair.sort_unstable();
            pair
        })
        .collect();
    let vertex_index: HashMap<u32, usize> = vertex_ids
        .iter()
        .enumerate()
        .map(|(index, &id)| (id, index))
        .collect();
    let sharp: Map<String, Value> = mesh
        .edges
        .iter()
        .map(|edge| {
            let mut pair = [
                vertex_index[&edge.vertices[0]],
                vertex_index[&edge.vertices[1]],
            ];
            pair.sort_unstable();
            (format!("e{}", edge.id), json!(sharp_edges.contains(&pair)))
        })
        .collect();
    mesh.attributes.insert(
        "sharp".to_owned(),
        json!({"domain":"edge","type":"bool","values":sharp}),
    );
    if !face_smooth.is_empty() {
        mesh.attributes.insert(
            "shade_smooth".to_owned(),
            json!({"domain":"face","type":"bool","values":face_smooth}),
        );
    }
    install_fixture_uv_layers(&mut mesh, input);
    install_fixture_normals(&mut mesh, input);
    install_fixture_colors(&mut mesh, input);
    mesh
}

fn scene_for_case(case: &Value) -> (SceneDoc, Id) {
    let input = &case["input"];
    let owner_id = Id::new("screw_owner").unwrap();
    let mesh_id = Id::new("screw_mesh").unwrap();
    let mut doc = SceneDoc::new("550e8400-e29b-41d4-a716-446655440000".to_owned());
    let mesh = source_mesh(case);
    let mut group_ids = BTreeMap::new();
    let mut vertex_groups = Vec::new();
    for (index, name) in input["groups"].as_object().unwrap().keys().enumerate() {
        let group_id = format!("screw_group_{index}");
        vertex_groups.push(json!({"id": group_id, "name": name}));
        group_ids.insert(name.clone(), group_id);
    }
    let mut vertex_weights = Map::new();
    for (group_name, weights) in input["groups"].as_object().unwrap() {
        for (index, weight) in weights.as_object().unwrap() {
            let vertex_id = mesh.vertices[index.parse::<usize>().unwrap()].id;
            let row = vertex_weights
                .entry(vertex_id.to_string())
                .or_insert_with(|| Value::Object(Map::new()));
            row.as_object_mut()
                .unwrap()
                .insert(group_ids[group_name].clone(), weight.clone());
        }
    }
    let block: DataBlock = serde_json::from_value(json!({
        "type": "mesh",
        "mesh": mesh,
        "vertex_groups": vertex_groups,
        "vertex_weights": vertex_weights,
    }))
    .unwrap();
    doc.data_blocks.insert(mesh_id.clone(), block);

    let mut material_ids = Vec::new();
    for slot in 0..input["material_slots"].as_u64().unwrap() {
        let material_id = Id::new(format!("screw_material_{slot}")).unwrap();
        doc.materials.insert(
            material_id.clone(),
            serde_json::from_value::<Material>(json!({"name": format!("Slot {slot}")})).unwrap(),
        );
        material_ids.push(material_id);
    }
    let owner: Node = serde_json::from_value(json!({
        "name": "Screw owner",
        "kind": "mesh",
        "data": mesh_id,
        "transform": case["owner_transform"],
        "materials": material_ids,
        "modifiers": [{
            "id": "screw_modifier",
            "type": "screw",
            "name": "Screw parity",
            "enabled": true,
            "params": case["modifier"]["params"],
            "binding_data": null,
        }],
    }))
    .unwrap();
    doc.nodes.insert(owner_id.clone(), owner);
    if let Some(transform) = case["axis_transform"].as_object() {
        let axis: Node = serde_json::from_value(json!({
            "name": "Screw axis",
            "kind": "empty",
            "transform": transform,
        }))
        .unwrap();
        doc.nodes.insert(Id::new("screw_axis").unwrap(), axis);
    }
    (doc, owner_id)
}

fn actual_face_normal(points: &[DVec3]) -> DVec3 {
    let mut normal = DVec3::ZERO;
    for index in 0..points.len() {
        let current = points[index];
        let next = points[(index + 1) % points.len()];
        normal.x += (current.y - next.y) * (current.z + next.z);
        normal.y += (current.z - next.z) * (current.x + next.x);
        normal.z += (current.x - next.x) * (current.y + next.y);
    }
    normal.normalize()
}

fn compare_numeric_values(actual: &Value, expected: &Value, context: &str) {
    match (actual, expected) {
        (Value::Array(actual), Value::Array(expected)) => {
            assert_eq!(actual.len(), expected.len(), "{context} lengths differ");
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                compare_numeric_values(actual, expected, &format!("{context}[{index}]"));
            }
        }
        (Value::Number(actual), Value::Number(expected)) => assert!(
            float32_close(actual.as_f64().unwrap(), expected.as_f64().unwrap()),
            "{context} differs: actual={actual} expected={expected}"
        ),
        _ => assert_eq!(actual, expected, "{context} differs"),
    }
}

fn compare_case(case: &Value) {
    let purpose = if case["render_mode"].as_bool().unwrap() {
        EvaluationPurpose::Render
    } else {
        EvaluationPurpose::Preview
    };
    let snapshot = Snapshot::evaluate(
        &doc,
        &EvaluationContext {
            purpose,
            ..EvaluationContext::default()
        },
    )
    .unwrap_or_else(|error| panic!("{}: evaluation failed: {error}", case["name"]));
    let mesh = &snapshot.meshes[&owner_id];
    let expected = if case["render_mode"].as_bool().unwrap() {
        &case["render"]
    } else {
        &case["viewport"]
    };
    let label = case["name"].as_str().unwrap();
    assert_eq!(
        mesh.vertices.len(),
        expected["counts"]["vertices"].as_u64().unwrap() as usize,
        "{label} vertices"
    );
    assert_eq!(
        mesh.edges.len(),
        expected["counts"]["edges"].as_u64().unwrap() as usize,
        "{label} edges"
    );
    assert_eq!(
        mesh.faces.len(),
        expected["counts"]["faces"].as_u64().unwrap() as usize,
        "{label} faces"
    );

    for (index, (vertex, wanted)) in mesh
        .vertices
        .iter()
        .zip(expected["vertices"].as_array().unwrap())
        .enumerate()
    {
        assert_vec_close(
            &vertex.co.to_array(),
            wanted.as_array().unwrap(),
            &format!("{label} vertex {index}"),
        );
    }
    let vertex_index: HashMap<u32, usize> = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let actual_edges: HashSet<[usize; 2]> = mesh
        .edges
        .iter()
        .map(|edge| {
            let mut pair = [
                vertex_index[&edge.vertices[0]],
                vertex_index[&edge.vertices[1]],
            ];
            pair.sort_unstable();
            pair
        })
        .collect();
    let expected_edges: HashSet<[usize; 2]> = expected["edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|edge| {
            let mut pair = [
                edge["vertices"][0].as_u64().unwrap() as usize,
                edge["vertices"][1].as_u64().unwrap() as usize,
            ];
            pair.sort_unstable();
            pair
        })
        .collect();
    assert_eq!(actual_edges, expected_edges, "{label} edge associations");
    let actual_edge_sharp: HashMap<[usize; 2], bool> = mesh
        .edges
        .iter()
        .map(|edge| {
            let mut pair = [
                vertex_index[&edge.vertices[0]],
                vertex_index[&edge.vertices[1]],
            ];
            pair.sort_unstable();
            let sharp = mesh
                .attributes
                .get("sharp")
                .and_then(|attribute| attribute["values"].get(format!("e{}", edge.id)))
                .and_then(Value::as_bool)
                .unwrap_or(false);
            (pair, sharp)
        })
        .collect();
    let expected_edge_sharp: HashMap<[usize; 2], bool> = expected["edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|edge| {
            let mut pair = [
                edge["vertices"][0].as_u64().unwrap() as usize,
                edge["vertices"][1].as_u64().unwrap() as usize,
            ];
            pair.sort_unstable();
            (pair, edge["sharp"].as_bool().unwrap())
        })
        .collect();
    assert_eq!(
        actual_edge_sharp, expected_edge_sharp,
        "{label} edge sharp flags"
    );

    let mut loop_start = 0;
    for (face_index, (face, wanted)) in mesh
        .faces
        .iter()
        .zip(expected["faces"].as_array().unwrap())
        .enumerate()
    {
        let indices: Vec<usize> = face.vertices.iter().map(|id| vertex_index[id]).collect();
        let wanted_indices: Vec<usize> = wanted["vertices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|index| index.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(indices, wanted_indices, "{label} face {face_index} corners");
        assert_eq!(
            face.material_index,
            wanted["material"].as_u64().unwrap() as u32,
            "{label} face {face_index} material"
        );
        let expected_smooth = wanted["smooth"].as_bool().unwrap();
        let actual_smooth = mesh
            .attributes
            .get("shade_smooth")
            .and_then(|attribute| attribute.get("values"))
            .and_then(|values| values.get(format!("f{}", face.id).as_str()))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        assert_eq!(
            actual_smooth, expected_smooth,
            "{label} face {face_index} smooth shading"
        );
        let face_points: Vec<DVec3> = face
            .vertices
            .iter()
            .map(|id| mesh.vertex(*id).unwrap().co)
            .collect();
        let normal = actual_face_normal(&face_points);
        assert_vec_close(
            &normal.to_array(),
            wanted["normal"].as_array().unwrap(),
            &format!("{label} face {face_index} normal"),
        );
        loop_start += face.vertices.len();
    }
    assert_eq!(
        loop_start,
        expected["counts"]["corners"].as_u64().unwrap() as usize,
        "{label} corner count"
    );

    let uv_store = mesh.attributes.get("uv_layers");
    let source_layers = case["input"]["uv_layers"].as_array().unwrap();
    let expected_uv_layers = expected["uv_layers"].as_object().unwrap();
    assert_eq!(
        expected_uv_layers.len(),
        source_layers.len(),
        "{label} UV layer count"
    );
    if source_layers.is_empty() {
        assert!(uv_store.is_none(), "{label} unexpected UV store");
        assert!(
            expected["uv_active"].is_null(),
            "{label} empty UV active layer"
        );
        assert!(
            expected["uv_render"].is_null(),
            "{label} empty UV render layer"
        );
    } else {
        let uv_store = uv_store.unwrap_or_else(|| panic!("{label} canonical UV store missing"));
        assert_eq!(
            uv_store["active"], expected["uv_active"],
            "{label} active UV layer"
        );
        assert_eq!(
            uv_store["render"], expected["uv_render"],
            "{label} render UV layer"
        );
        let actual_layers = uv_store["layers"].as_array().unwrap();
        assert_eq!(
            actual_layers.len(),
            source_layers.len(),
            "{label} UV layer count"
        );
        for (actual_layer, input_layer) in actual_layers.iter().zip(source_layers) {
            let name = input_layer["name"].as_str().unwrap();
            assert_eq!(
                actual_layer["name"], input_layer["name"],
                "{label} UV layer order"
            );
            let expected_values = expected_uv_layers
                .get(name)
                .unwrap_or_else(|| panic!("{label} Blender output lacks UV layer {name}"))
                .as_array()
                .unwrap();
            let actual_faces = actual_layer["faces"].as_array().unwrap();
            assert_eq!(
                actual_faces.len(),
                mesh.faces.len(),
                "{label} {name} face count"
            );
            let actual_values: Vec<Value> = mesh
                .faces
                .iter()
                .flat_map(|face| {
                    let entry = actual_faces
                        .iter()
                        .find(|entry| entry["face_id"].as_u64() == Some(u64::from(face.id)))
                        .unwrap_or_else(|| panic!("{label} {name} missing face {}", face.id));
                    entry["uv"].as_array().unwrap().clone()
                })
                .collect();
            let expected_values: Vec<Value> =
                expected_values.iter().map(blender_uv_to_potter).collect();
            compare_numeric_values(
                &Value::Array(actual_values),
                &Value::Array(expected_values),
                &format!("{label} UV {name}"),
            );
        }
    }

    let expected_colors = expected["colors"].as_object().unwrap();
    if expected_colors.is_empty() {
        assert!(
            mesh.attributes.get("color_layers").is_none(),
            "{label} unexpected color registry"
        );
    } else {
        assert_eq!(
            mesh.attributes.get("color_layers"),
            Some(&expected["color_layers"]),
            "{label} color layer order and active/render names"
        );
    }
    let empty_types = Map::new();
    let color_types = mesh
        .attributes
        .get("blender_metadata")
        .and_then(|metadata| metadata.get("color_data_types"))
        .and_then(Value::as_object)
        .unwrap_or(&empty_types);
    for (name, wanted) in expected_colors {
        let encoding = color_types.get(name).unwrap();
        assert_eq!(
            encoding, &wanted["data_type"],
            "{label} {name} source encoding"
        );
        let attr = &mesh.attributes[name];
        assert_eq!(
            attr["domain"],
            if wanted["domain"] == "POINT" {
                "point"
            } else {
                "corner"
            },
            "{label} {name} native color domain"
        );
        compare_numeric_values(
            &Value::Array(actual_color_values(mesh, name)),
            &wanted["values"],
            &format!("{label} color {name}"),
        );
    }
    assert!(
        mesh.attributes.get("blender_attributes").is_none(),
        "{label} native colors must not be duplicated"
    );
    if !case["input"]["custom_normals"]
        .as_array()
        .unwrap()
        .is_empty()
    {
        let state = &mesh.attributes["custom_normal"];
        assert_eq!(
            state["type"], "normal_code",
            "{label} canonical normal state"
        );
        let values = state["values"].as_object().unwrap();
        for face in &mesh.faces {
            let row = values[&format!("f{}", face.id)].as_array().unwrap();
            assert_eq!(
                row.len(),
                face.vertices.len(),
                "{label} normal-code corner count"
            );
            for code in row {
                assert!(
                    code.is_null()
                        || code.as_str() == Some("explicit_zero")
                        || code.as_array().is_some_and(|pair| {
                            pair.len() == 2 && pair.iter().all(Value::is_i64)
                        }),
                    "{label} raw normal code must be Auto, an i16 pair, or explicit_zero: {code}"
                );
            }
        }
        let mut face_edges = HashSet::new();
        for face in case["input"]["faces"].as_array().unwrap() {
            let vertices: Vec<usize> = face["vertices"]
                .as_array()
                .unwrap()
                .iter()
                .map(|index| index.as_u64().unwrap() as usize)
                .collect();
            for corner in 0..vertices.len() {
                let mut edge = [vertices[corner], vertices[(corner + 1) % vertices.len()]];
                edge.sort_unstable();
                face_edges.insert(edge);
            }
        }
        let loose_edges = case["input"]["edges"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|edge| {
                let mut pair = [
                    edge[0].as_u64().unwrap() as usize,
                    edge[1].as_u64().unwrap() as usize,
                ];
                pair.sort_unstable();
                !face_edges.contains(&pair)
            })
            .count();
        let faces_per_edge = mesh.faces.len() / case["input"]["edges"].as_array().unwrap().len();
        let generated_faces = loose_edges * faces_per_edge;
        for face in mesh.faces.iter().skip(mesh.faces.len() - generated_faces) {
            let row = values[&format!("f{}", face.id)].as_array().unwrap();
            assert!(
                row.iter().all(Value::is_null),
                "{label} loose-edge corners are Auto codes"
            );
        }
        if let Some(layer) = expected_colors.get("CornerTint")
            && layer["data_type"] == "FLOAT_COLOR"
            && layer["domain"] == "CORNER"
        {
            let colors = layer["values"].as_array().unwrap();
            let generated_corners = generated_faces * 4;
            assert!(
                colors.len() >= generated_corners,
                "{label} generated corner color count"
            );
            assert!(
                colors[colors.len() - generated_corners..]
                    .iter()
                    .all(|color| color == &json!([1.0, 1.0, 1.0, 1.0])),
                "{label} loose-edge FLOAT_COLOR defaults must be opaque white"
            );
        }
    }

    let actual_groups = mesh
        .attributes
        .get("vertex_groups")
        .and_then(Value::as_object);
    for (vertex, wanted_weights) in expected["weights"].as_array().unwrap().iter().enumerate() {
        for group in case["input"]["groups"].as_object().unwrap().keys() {
            let expected_weight = wanted_weights
                .get(group.as_str())
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            let actual_weight = actual_groups
                .and_then(|groups| groups.get(group.as_str()))
                .and_then(Value::as_object)
                .and_then(|weights| weights.get(format!("v{}", mesh.vertices[vertex].id).as_str()))
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            assert!(
                float32_close(actual_weight, expected_weight),
                "{label} vertex {vertex} group {group}: {actual_weight} vs {expected_weight}"
            );
        }
    }
}

#[test]
fn screw_matches_blender_522_construction_corpus() {
    let fixture: Value = serde_json::from_str(SCREW_FIXTURES).unwrap();
    assert_eq!(fixture["blender_version"], "5.2.2");
    let expected_defaults = json!({
        "object": null,
        "use_object_screw_offset": false,
        "iterations": 1,
        "steps": 16,
        "render_steps": 16,
        "angle": 6.2831854820251465,
        "axis": "Z",
        "screw_offset": 0.0,
        "merge_threshold": 0.009999999776482582,
        "use_merge_vertices": false,
        "use_normal_calculate": false,
        "use_normal_flip": false,
        "use_smooth_shade": true,
        "use_stretch_u": false,
        "use_stretch_v": false,
    });
    assert_eq!(
        fixture["cases"][0]["modifier"]["defaults"],
        expected_defaults
    );
    for case in fixture["cases"].as_array().unwrap() {
        assert_eq!(case["modifier"]["defaults"], expected_defaults);
    }
    for case in fixture["cases"].as_array().unwrap() {
        compare_case(case);
    }
}

#[test]
fn screw_preserves_first_ring_source_float32_with_large_axis_pivot() {
    let fixture: Value = serde_json::from_str(SCREW_FIXTURES).unwrap();
    let case = fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == "large_pivot_preserves_first_ring_f32")
        .unwrap();
    assert_eq!(
        case["input"]["vertices"][0][0].as_f64().unwrap() as f32,
        0.0001_f32
    );
    let translation = case["axis_transform"]["translation"].as_array().unwrap();
    assert_eq!(translation[0].as_f64().unwrap(), 10_000.0);
    assert_eq!(translation[1].as_f64().unwrap(), 0.0);
    assert_eq!(translation[2].as_f64().unwrap(), 0.0);

    let (doc, owner_id) = scene_for_case(case);
    let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default()).unwrap();
    assert_eq!(
        snapshot.meshes[&owner_id].vertices[0].co.x as f32,
        0.0001_f32
    );
}

#[test]
fn screw_normal_calculation_matches_blender_for_asymmetric_radial_neighbors() {
    let fixture: Value = serde_json::from_str(SCREW_FIXTURES).unwrap();
    let case = fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == "normal_calculate_asymmetric_radial_neighbors")
        .unwrap();
    compare_numeric_values(
        &case["input"]["vertices"],
        &json!([[2.0, 0.0, 0.0], [1.9, 0.0, 1.0], [0.0, 0.0, 2.0]]),
        "asymmetric screw profile vertices",
    );
    assert_eq!(case["input"]["edges"], json!([[0, 1], [0, 2]]));
    assert_eq!(case["modifier"]["params"]["axis"], "Z");
    assert_eq!(case["modifier"]["params"]["use_normal_calculate"], true);
    compare_case(case);
}
