#![expect(
    clippy::unwrap_used,
    reason = "fixed Blender parity fixtures have valid serialized inputs"
)]

use std::{collections::BTreeMap, fs, path::Path};

use glam::DVec3;
use potter_core::{
    eval::{EvaluationContext, Snapshot},
    geom::Mesh,
    model::{DataBlock, Id, Material, Node, SceneDoc},
};
use serde_json::{Map, Value, json};

const FIXTURE: &str = "tests/fixtures/construction_p/array/array_parity.json";
const POSITION_TOLERANCE: f64 = 1.0e-5;
const UV_TOLERANCE: f64 = 2.0e-6;

fn fixture() -> Value {
    serde_json::from_slice(&fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE)).unwrap())
        .unwrap()
}

fn id(value: &str) -> Id {
    Id::new(value).unwrap()
}

fn vector3(value: &Value) -> DVec3 {
    let values = value.as_array().unwrap();
    DVec3::new(
        values[0].as_f64().unwrap(),
        values[1].as_f64().unwrap(),
        values[2].as_f64().unwrap(),
    )
}

fn install_fixture_colors(mesh: &mut Mesh, spec: &Value) {
    let Some(layers) = spec["colors"]
        .as_array()
        .filter(|layers| !layers.is_empty())
    else {
        return;
    };
    let mut encodings = Map::new();
    for layer in layers {
        let name = layer["name"].as_str().unwrap();
        let domain = layer["domain"].as_str().unwrap();
        let input = layer["values"].as_array().unwrap();
        let mut values = Map::new();
        match domain {
            "POINT" => {
                for (vertex, value) in mesh.vertices.iter().zip(input) {
                    values.insert(format!("v{}", vertex.id), value.clone());
                }
            }
            "CORNER" => {
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
            _ => unreachable!("fixture color domain"),
        }
        let domain = if domain == "POINT" { "point" } else { "corner" };
        mesh.attributes.insert(
            name.to_owned(),
            json!({"domain": domain, "type": "color", "values": values}),
        );
        encodings.insert(name.to_owned(), layer["data_type"].clone());
    }
    mesh.attributes
        .insert("color_layers".to_owned(), spec["color_layers"].clone());
    mesh.attributes.insert(
        "blender_metadata".to_owned(),
        json!({"color_data_types": encodings}),
    );
}

fn install_fixture_normals(mesh: &mut Mesh, spec: &Value) {
    if let Some(rows) = spec["custom_normal_corner_order"].as_array() {
        let corner_order: Map<String, Value> = mesh
            .faces
            .iter()
            .zip(rows)
            .map(|(face, row)| (format!("f{}", face.id), row.clone()))
            .collect();
        mesh.attributes.insert(
            "custom_normal".to_owned(),
            json!({"domain":"corner","type":"normal_code","corner_order":corner_order}),
        );
    }
    let Some(normals) = spec["custom_normals"].as_array() else {
        return;
    };
    let values: Map<String, Value> = mesh
        .faces
        .iter()
        .zip(normals)
        .map(|(face, normals)| (format!("f{}", face.id), normals.clone()))
        .collect();
    let ids: Vec<String> = mesh
        .faces
        .iter()
        .map(|face| format!("f{}", face.id))
        .collect();
    potter_core::geom::edit::apply(
        mesh,
        "set_custom_normals",
        &json!({
            "elements": {"domain":"face", "ids":ids},
            "normals": values,
        }),
    )
    .unwrap();
}

fn flattened_corner_order(mesh: &Mesh) -> Option<Vec<usize>> {
    let rows = mesh
        .attributes
        .get("custom_normal")?
        .get("corner_order")?
        .as_object()?;
    mesh.faces.iter().try_fold(Vec::new(), |mut order, face| {
        let row = rows.get(&format!("f{}", face.id))?.as_array()?;
        if row.len() != face.vertices.len() {
            return None;
        }
        for ordinal in row {
            order.push(usize::try_from(ordinal.as_u64()?).ok()?);
        }
        Some(order)
    })
}

fn blender_uv_to_potter(value: &Value) -> Value {
    let uv = value.as_array().unwrap();
    json!([uv[0].as_f64().unwrap(), 1.0 - uv[1].as_f64().unwrap()])
}

fn install_fixture_uv_layers(mesh: &mut Mesh, spec: &Value) {
    let Some(source_layers) = spec["uv_layers"].as_array() else {
        return;
    };
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
    if layers.is_empty() {
        return;
    }
    mesh.attributes.insert(
        "uv_layers".to_owned(),
        json!({
            "layers": layers,
            "active": spec.get("uv_active").cloned().unwrap_or(Value::Null),
            "render": spec.get("uv_render").cloned().unwrap_or(Value::Null),
        }),
    );
}

fn mesh_from_fixture(spec: &Value) -> Mesh {
    let positions = spec["vertices"]
        .as_array()
        .unwrap()
        .iter()
        .map(vector3)
        .collect();
    let faces: Vec<Vec<usize>> = spec["faces"]
        .as_array()
        .unwrap()
        .iter()
        .map(|face| {
            face.as_array()
                .unwrap()
                .iter()
                .map(|vertex| usize::try_from(vertex.as_u64().unwrap()).unwrap())
                .collect()
        })
        .collect();
    let mut mesh = Mesh::from_positions_and_faces(positions, faces).unwrap();
    if let Some(materials) = spec["materials"].as_array() {
        for (face, material) in mesh.faces.iter_mut().zip(materials) {
            face.material_index = u32::try_from(material.as_u64().unwrap()).unwrap();
        }
    }
    install_fixture_uv_layers(&mut mesh, spec);

    if let Some(smooth) = spec["face_smooth"].as_array() {
        let values: Map<String, Value> = smooth
            .iter()
            .enumerate()
            .map(|(index, flag)| (format!("f{index}"), flag.clone()))
            .collect();
        mesh.attributes.insert(
            "shade_smooth".to_owned(),
            json!({"domain":"face","type":"bool","values":values}),
        );
    }
    install_fixture_normals(&mut mesh, spec);
    install_fixture_colors(&mut mesh, spec);
    mesh
}

fn vertex_group_data(
    spec: &Value,
    object_id: &str,
    group_order: Option<&Value>,
) -> (Vec<Value>, Value) {
    let groups = spec["groups"].as_object().unwrap();
    let names: Vec<String> = match group_order.and_then(Value::as_array) {
        Some(order) => order
            .iter()
            .map(|name| name.as_str().unwrap().to_owned())
            .collect(),
        None => groups.keys().cloned().collect(),
    };
    let mut group_ids = BTreeMap::new();
    let metadata: Vec<Value> = names
        .iter()
        .map(|name| {
            let group_id = format!("{object_id}_group_{}", name.to_lowercase());
            group_ids.insert(name.clone(), group_id.clone());
            json!({"id": group_id, "name": name})
        })
        .collect();
    let mut weights = Map::new();
    for (name, values) in groups {
        let group_id = group_ids.get(name).unwrap();
        for (index, weight) in values.as_array().unwrap().iter().enumerate() {
            if weight.is_null() {
                continue;
            }
            let vertex = weights
                .entry(index.to_string())
                .or_insert_with(|| Value::Object(Map::new()))
                .as_object_mut()
                .unwrap();
            vertex.insert(group_id.clone(), weight.clone());
        }
    }
    (metadata, Value::Object(weights))
}

fn insert_mesh_node(
    doc: &mut SceneDoc,
    node_id: &str,
    spec: &Value,
    transform: Option<&Value>,
    material_slots: usize,
    group_order: Option<&Value>,
    modifier: Option<&Value>,
) {
    let mesh = mesh_from_fixture(spec);
    let data_id = id(&format!("{node_id}_data"));
    let (vertex_groups, vertex_weights) = vertex_group_data(spec, node_id, group_order);
    let block: DataBlock = serde_json::from_value(json!({
        "type": "mesh",
        "mesh": mesh,
        "vertex_groups": vertex_groups,
        "vertex_weights": vertex_weights,
    }))
    .unwrap();
    doc.data_blocks.insert(data_id.clone(), block);

    let material_ids: Vec<Id> = (0..material_slots)
        .map(|index| {
            let material_id = id(&format!("{node_id}_material_{index}"));
            doc.materials.insert(
                material_id.clone(),
                serde_json::from_value::<Material>(
                    json!({"name": format!("{node_id} material {index}")}),
                )
                .unwrap(),
            );
            material_id
        })
        .collect();
    let mut node_value = json!({
        "name": node_id,
        "kind": "mesh",
        "data": data_id,
        "materials": material_ids,
    });
    if let Some(transform) = transform {
        node_value["transform"] = transform.clone();
    }
    if let Some(modifier) = modifier {
        let mut params = modifier.clone();
        for key in ["offset_object", "start_cap", "end_cap", "curve"] {
            if let Some(target) = params.get(key).and_then(Value::as_str) {
                params[key] = json!(target.to_ascii_lowercase());
            }
        }
        node_value["modifiers"] = json!([{
            "id": "array_parity_modifier",
            "type": "array",
            "name": "Array parity",
            "enabled": true,
            "params": params,
            "binding_data": null,
        }]);
    }
    let node: Node = serde_json::from_value(node_value).unwrap();
    doc.nodes.insert(id(node_id), node);
}

fn insert_empty_node(doc: &mut SceneDoc, node_id: &str, transform: &Value) {
    doc.nodes.insert(
        id(node_id),
        serde_json::from_value(json!({
            "name": node_id,
            "kind": "empty",
            "transform": transform,
        }))
        .unwrap(),
    );
}

fn insert_curve_node(doc: &mut SceneDoc, curve: &Value) {
    let data_id = id("fit_path_data");
    let points: Vec<Value> = curve["points"]
        .as_array()
        .unwrap()
        .iter()
        .map(|co| json!({"co": co}))
        .collect();
    let block: DataBlock = serde_json::from_value(json!({
        "type": "curve",
        "curve": {
            "resolution_u": 1,
            "use_path": curve["use_path"],
            "splines": [{"type": "poly", "points": points}],
        },
    }))
    .unwrap();
    doc.data_blocks.insert(data_id.clone(), block);
    doc.nodes.insert(
        id("fitpath"),
        serde_json::from_value(json!({
            "name": "FitPath",
            "kind": "curve",
            "data": data_id,
            "transform": curve["transform"],
        }))
        .unwrap(),
    );
}

fn scene_for_case(case: &Value) -> SceneDoc {
    let mut doc = SceneDoc::new("faea6a7b-0050-4afc-8598-8e09369bb1f0".to_owned());
    let owner_slots = case["owner_material_slots"].as_u64().unwrap_or(3) as usize;
    insert_mesh_node(
        &mut doc,
        "body",
        &case["input"],
        case.get("owner_transform"),
        owner_slots,
        case.get("owner_groups_order"),
        Some(&case["params"]),
    );
    if let Some(objects) = case["objects"].as_object() {
        for (name, object) in objects {
            let node_id = name.to_ascii_lowercase();
            insert_mesh_node(
                &mut doc,
                &node_id,
                &object["mesh"],
                object.get("transform"),
                object["material_slots"].as_u64().unwrap_or(3) as usize,
                object.get("groups_order"),
                None,
            );
        }
    }
    if let Some(empties) = case["empties"].as_object() {
        for (name, transform) in empties {
            insert_empty_node(&mut doc, &name.to_ascii_lowercase(), transform);
        }
    }
    if let Some(curve) = case.get("curve").filter(|value| !value.is_null()) {
        insert_curve_node(&mut doc, curve);
    }
    doc
}

fn actual_faces(mesh: &Mesh) -> Vec<Value> {
    let index: BTreeMap<u32, usize> = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    mesh.faces
        .iter()
        .map(|face| {
            let smooth = mesh
                .attributes
                .get("shade_smooth")
                .and_then(|attribute| attribute.get("values"))
                .and_then(Value::as_object)
                .and_then(|values| values.get(&format!("f{}", face.id)))
                .and_then(Value::as_bool)
                .unwrap_or(false);
            json!({
                "vertices": face.vertices.iter().map(|vertex| index[vertex]).collect::<Vec<_>>(),
                "material": face.material_index,
                "smooth": smooth,
            })
        })
        .collect()
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

fn assert_float_arrays(actual: &[Value], expected: &[Value], tolerance: f64, context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context} value count");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let actual = actual.as_f64().unwrap();
        let expected = expected.as_f64().unwrap();
        assert!(
            (actual - expected).abs() <= tolerance,
            "{context}[{index}]: Potter {actual:?}, Blender {expected:?}"
        );
    }
}

fn assert_node_transform(doc: &SceneDoc, node_name: &str, transform: &Value) {
    let rotation = transform["rotation"].as_array().unwrap();
    assert_eq!(
        rotation.len(),
        4,
        "{node_name} transform quaternion must be XYZW"
    );
    let node = serde_json::to_value(&doc.nodes[&id(node_name)]).unwrap();
    for field in ["translation", "rotation", "scale"] {
        assert_float_arrays(
            node["transform"][field].as_array().unwrap(),
            transform[field].as_array().unwrap(),
            POSITION_TOLERANCE,
            &format!("{node_name} transform {field}"),
        );
    }
}

fn assert_case(case: &Value) {
    let doc = scene_for_case(case);
    let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default()).unwrap();
    let mesh = &snapshot.meshes[&id("body")];
    let expected = &case["expected"];
    let name = case["name"].as_str().unwrap();

    assert_eq!(
        mesh.vertices.len(),
        expected["vertices"].as_array().unwrap().len(),
        "{name} vertex count"
    );
    assert_eq!(
        mesh.faces.len(),
        expected["faces"].as_array().unwrap().len(),
        "{name} face count"
    );
    assert_eq!(
        mesh.edges.len(),
        expected["edges"].as_array().unwrap().len(),
        "{name} edge count"
    );
    let actual_positions: Vec<Value> = mesh
        .vertices
        .iter()
        .map(|vertex| json!(vertex.co.to_array()))
        .collect();
    let expected_positions: Vec<Value> = expected["vertices"].as_array().unwrap().clone();
    for (vertex, (actual, expected)) in actual_positions.iter().zip(&expected_positions).enumerate()
    {
        let actual = actual.as_array().unwrap();
        let expected = expected.as_array().unwrap();
        assert_float_arrays(
            actual,
            expected,
            POSITION_TOLERANCE,
            &format!("{name} vertex {vertex}"),
        );
    }
    assert_eq!(
        actual_faces(mesh),
        expected["faces"].as_array().unwrap().clone(),
        "{name} face indices and material indices"
    );

    assert_node_transform(&doc, "body", &case["owner_transform"]);
    for (name, object) in case["objects"].as_object().unwrap() {
        assert_node_transform(&doc, &name.to_ascii_lowercase(), &object["transform"]);
    }
    for (name, transform) in case["empties"].as_object().unwrap() {
        assert_node_transform(&doc, &name.to_ascii_lowercase(), transform);
    }
    if let Some(curve) = case.get("curve").filter(|curve| !curve.is_null()) {
        assert_node_transform(&doc, "fitpath", &curve["transform"]);
    }

    let index: BTreeMap<u32, usize> = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let mut actual_edges: Vec<[usize; 2]> = mesh
        .edges
        .iter()
        .map(|edge| {
            let first = index[&edge.vertices[0]];
            let second = index[&edge.vertices[1]];
            [first.min(second), first.max(second)]
        })
        .collect();
    let mut expected_edges: Vec<[usize; 2]> = expected["edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|edge| {
            let values = edge.as_array().unwrap();
            let first = usize::try_from(values[0].as_u64().unwrap()).unwrap();
            let second = usize::try_from(values[1].as_u64().unwrap()).unwrap();
            [first.min(second), first.max(second)]
        })
        .collect();
    actual_edges.sort_unstable();
    expected_edges.sort_unstable();
    assert_eq!(actual_edges, expected_edges, "{name} edge endpoints");

    let expected_layers = expected["uv_layers"].as_array().unwrap();
    if expected_layers.is_empty() {
        assert!(
            mesh.attributes.get("uv_layers").is_none(),
            "{name} unexpected UV store"
        );
        assert!(
            expected["uv_active"].is_null(),
            "{name} empty UV active layer"
        );
        assert!(
            expected["uv_render"].is_null(),
            "{name} empty UV render layer"
        );
    } else {
        let uv_store = &mesh.attributes["uv_layers"];
        assert_eq!(
            uv_store["active"], expected["uv_active"],
            "{name} active UV layer"
        );
        assert_eq!(
            uv_store["render"], expected["uv_render"],
            "{name} render UV layer"
        );
        let actual_layers = uv_store["layers"].as_array().unwrap();
        assert_eq!(
            actual_layers.len(),
            expected_layers.len(),
            "{name} UV layer count"
        );
        for (actual_layer, expected_layer) in actual_layers.iter().zip(expected_layers) {
            let layer_name = expected_layer["name"].as_str().unwrap();
            assert_eq!(
                actual_layer["name"], expected_layer["name"],
                "{name} UV layer order"
            );
            let actual_faces = actual_layer["faces"].as_array().unwrap();
            let expected_faces = expected_layer["faces"].as_array().unwrap();
            assert_eq!(
                expected_faces.len(),
                mesh.faces.len(),
                "{name} {layer_name} face count"
            );
            for (face_index, (face, expected_face)) in
                mesh.faces.iter().zip(expected_faces).enumerate()
            {
                let actual_face = actual_faces
                    .iter()
                    .find(|entry| entry["face_id"].as_u64() == Some(u64::from(face.id)))
                    .unwrap_or_else(|| panic!("{name} {layer_name} missing face {}", face.id));
                let actual_uv = actual_face["uv"].as_array().unwrap();
                let expected_uv: Vec<Value> = expected_face
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(blender_uv_to_potter)
                    .collect();
                assert_eq!(
                    actual_uv.len(),
                    expected_uv.len(),
                    "{name} {layer_name} face {face_index} corner count"
                );
                for (corner, (actual, expected)) in actual_uv.iter().zip(&expected_uv).enumerate() {
                    assert_float_arrays(
                        actual.as_array().unwrap(),
                        expected.as_array().unwrap(),
                        UV_TOLERANCE,
                        &format!("{name} {layer_name} face {face_index} corner {corner}"),
                    );
                }
            }
        }
    }
    let expected_colors = expected["colors"].as_array().unwrap();
    if expected_colors.is_empty() {
        assert!(
            mesh.attributes.get("color_layers").is_none(),
            "{name} unexpected color registry"
        );
    } else {
        assert_eq!(
            mesh.attributes.get("color_layers"),
            Some(&expected["color_layers"]),
            "{name} color layer order and active/render names"
        );
    }
    let empty_types = Map::new();
    let color_types = mesh
        .attributes
        .get("blender_metadata")
        .and_then(|metadata| metadata.get("color_data_types"))
        .and_then(Value::as_object)
        .unwrap_or(&empty_types);
    for layer in expected_colors {
        let layer_name = layer["name"].as_str().unwrap();
        assert_eq!(
            color_types[layer_name], layer["data_type"],
            "{name} {layer_name} source encoding"
        );
        let actual = actual_color_values(mesh, layer_name);
        let expected_values = layer["values"].as_array().unwrap();
        assert_eq!(
            actual.len(),
            expected_values.len(),
            "{name} {layer_name} value count"
        );
        for (index, (actual, expected)) in actual.iter().zip(expected_values).enumerate() {
            assert_float_arrays(
                actual.as_array().unwrap(),
                expected.as_array().unwrap(),
                UV_TOLERANCE,
                &format!("{name} {layer_name} value {index}"),
            );
        }
    }
    assert!(
        mesh.attributes.get("blender_attributes").is_none(),
        "{name} native colors must not be duplicated"
    );

    if let Some(expected_groups) = expected["groups"].as_object() {
        let empty_groups = Map::new();
        let actual_groups = mesh
            .attributes
            .get("vertex_groups")
            .and_then(Value::as_object)
            .unwrap_or(&empty_groups);
        for (group_name, weights) in expected_groups {
            let actual_weights = actual_groups[group_name].as_object().unwrap();
            for (vertex, weight) in weights.as_array().unwrap().iter().enumerate() {
                let key = format!("v{}", mesh.vertices[vertex].id);
                match weight.as_f64() {
                    Some(expected) => {
                        let actual = actual_weights.get(&key).unwrap().as_f64().unwrap();
                        assert!(
                            (actual - expected).abs() <= UV_TOLERANCE,
                            "{name} group {group_name} vertex {vertex}: Potter {actual}, Blender {expected}"
                        );
                    }
                    None => assert!(
                        !actual_weights.contains_key(&key),
                        "{name} group {group_name} unexpectedly weights vertex {vertex}"
                    ),
                }
            }
        }
    }
    if case["input"]["custom_normals"].is_array() {
        let custom_normal = &mesh.attributes["custom_normal"];
        assert_eq!(
            custom_normal["type"], "normal_code",
            "{name} canonical normal storage"
        );
        let values = custom_normal["values"].as_object().unwrap();
        let source = mesh_from_fixture(&case["input"]);
        let source_values = source.attributes["custom_normal"]["values"]
            .as_object()
            .unwrap();
        for (source_face, output_face) in source.faces.iter().zip(&mesh.faces) {
            let actual = values[&format!("f{}", output_face.id)].as_array().unwrap();
            let original = source_values[&format!("f{}", source_face.id)]
                .as_array()
                .unwrap();
            assert_eq!(
                actual, original,
                "{name} untransformed first-copy raw normal codes"
            );
        }
        for face in &mesh.faces {
            for code in values[&format!("f{}", face.id)].as_array().unwrap() {
                assert!(
                    code.is_null()
                        || code.as_str() == Some("explicit_zero")
                        || code.as_array().is_some_and(|pair| {
                            pair.len() == 2
                                && pair.iter().all(|value| {
                                    value
                                        .as_i64()
                                        .is_some_and(|value| i16::try_from(value).is_ok())
                                })
                                && pair.iter().any(|value| value.as_i64() != Some(0))
                        }),
                    "{name} raw normal code must be canonical Auto, an i16 pair, or explicit_zero: {code}"
                );
            }
        }
        if case["input"]["custom_normal_corner_order"].is_array() {
            let source_order = flattened_corner_order(&source).unwrap();
            let mut sorted_source_order = source_order.clone();
            sorted_source_order.sort_unstable();
            assert_eq!(
                sorted_source_order,
                (0..source_order.len()).collect::<Vec<_>>(),
                "{name} input corner order must be a permutation"
            );
            assert_ne!(
                source_order,
                (0..source_order.len()).collect::<Vec<_>>(),
                "{name} input corner order must be nonidentity"
            );
            let output_order = flattened_corner_order(mesh).unwrap_or_else(|| {
                panic!("{name} output must persist its nonidentity corner order")
            });
            let mut sorted_output_order = output_order.clone();
            sorted_output_order.sort_unstable();
            assert_eq!(
                sorted_output_order,
                (0..output_order.len()).collect::<Vec<_>>(),
                "{name} output corner order must be a permutation"
            );
            assert_ne!(
                output_order,
                (0..output_order.len()).collect::<Vec<_>>(),
                "{name} output must preserve a nonidentity loop basis"
            );
        }
    }

    let expected_warning = match name {
        "maximum_rna_count_falls_back" => Some("modifier.array.count_too_high"),
        "fit_length_tiny_offset_fallback" => Some("modifier.array.offset_too_small"),
        _ => None,
    };
    if let Some(feature_id) = expected_warning {
        assert!(
            snapshot.warnings.iter().any(|warning| {
                warning.details.get("feature_id").and_then(Value::as_str) == Some(feature_id)
            }),
            "{name} fallback must be exposed as a modifier warning"
        );
    }
}

#[test]
fn empty_source_array_huge_count_preserves_caps_and_empty_output() {
    let fixture = fixture();
    let cases = fixture["cases"].as_array().unwrap();
    for name in [
        "empty_source_huge_count_no_caps",
        "empty_source_huge_count_with_caps",
        "empty_source_huge_count_with_caps_nonbinary_offset",
    ] {
        let case = cases
            .iter()
            .find(|case| case["name"].as_str() == Some(name))
            .unwrap_or_else(|| panic!("missing fixture {name}"));
        assert_eq!(case["params"]["count"].as_u64(), Some(i32::MAX as u64));
        assert_case(case);
    }
}

#[test]
fn empty_source_array_object_offsets_preserve_caps() {
    let fixture = fixture();
    let cases = fixture["cases"].as_array().unwrap();
    for (name, count) in [
        ("empty_source_object_offset_reflection_odd_count", 3_u64),
        ("empty_source_object_offset_reflection_even_count", 4),
        ("empty_source_object_offset_rotated_count", 5),
    ] {
        let case = cases
            .iter()
            .find(|case| case["name"].as_str() == Some(name))
            .unwrap_or_else(|| panic!("missing fixture {name}"));
        assert_eq!(case["params"]["count"].as_u64(), Some(count));
        assert_case(case);
    }
}

#[test]
fn array_merge_split_faces_keep_blender_smooth_default() {
    let fixture = fixture();
    let case = fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == "merge_split_faces_keep_smooth_default")
        .unwrap_or_else(|| panic!("missing Array split-face smoothing oracle case"));
    let faces = case["expected"]["faces"].as_array().unwrap();
    assert_eq!(
        faces.len(),
        3,
        "the second copy splits at merged corners 0 and 3"
    );
    assert!(
        faces
            .iter()
            .all(|face| face["smooth"].as_bool() == Some(true)),
        "Blender defaults every generated Array split face to smooth"
    );
    assert_case(case);
}

#[test]
fn array_matches_blender_522_parity_fixtures() {
    let fixture = fixture();
    assert_eq!(fixture["blender_version"], "5.2.2 LTS");
    for case in fixture["cases"].as_array().unwrap() {
        assert_case(case);
    }
}
