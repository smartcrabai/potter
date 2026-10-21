#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "Blender oracle fixtures contain fixed valid meshes and expected outputs"
)]

use std::collections::{HashMap, HashSet};

use glam::DVec3;
use potter_core::{
    geom::{Mesh, modifiers::evaluate_modifiers_with_groups_at_frame},
    graph::NodeGroup,
    model::{Id, Modifier, ModifierRuntime, Registry},
};
use serde_json::{Map, Value, json};

const BUILD_FIXTURE: &str = include_str!("fixtures/construction_p/build_wireframe/build.json");
const WIREFRAME_FIXTURE: &str =
    include_str!("fixtures/construction_p/build_wireframe/wireframe.json");
const SUBJECT_NAME: &str = "ConstructionSubject";

fn number(value: &Value) -> f64 {
    value.as_f64().expect("fixture number")
}

fn index_rows(value: &Value) -> Vec<Vec<usize>> {
    value
        .as_array()
        .expect("fixture index rows")
        .iter()
        .map(|row| {
            row.as_array()
                .expect("fixture index row")
                .iter()
                .map(|item| item.as_u64().expect("fixture index") as usize)
                .collect()
        })
        .collect()
}

fn mesh_for(variant: &Value) -> Mesh {
    let spec = &variant["mesh"];
    let vertices = spec["vertices"]
        .as_array()
        .expect("fixture vertices")
        .iter()
        .map(|point| {
            let coordinates = point.as_array().expect("fixture point");
            DVec3::new(
                number(&coordinates[0]),
                number(&coordinates[1]),
                number(&coordinates[2]),
            )
        })
        .collect();
    let faces = index_rows(&spec["faces"]);
    let mut mesh = Mesh::from_positions_and_faces(vertices, faces).expect("valid fixture mesh");
    for (face, material) in mesh
        .faces
        .iter_mut()
        .zip(spec["materials"].as_array().expect("fixture materials"))
    {
        face.material_index = material.as_u64().expect("fixture material") as u32;
    }
    for [first, second] in index_rows(&spec["edges"])
        .into_iter()
        .map(|edge| [edge[0], edge[1]])
    {
        mesh.insert_edge([first as u32, second as u32])
            .expect("valid loose fixture edge");
    }
    mesh.attributes = spec["attributes"]
        .as_object()
        .expect("fixture mesh attributes")
        .iter()
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect();
    assert!(
        mesh.attributes.contains_key("uv_layers"),
        "fixture mesh uses the canonical UV store",
    );
    assert!(
        !mesh.attributes.contains_key("blender_uv_layers"),
        "fixture mesh must not use the legacy UV layer array",
    );
    let sharp_pairs: HashSet<[usize; 2]> = spec
        .get("sharp_edges")
        .map(index_rows)
        .unwrap_or_default()
        .into_iter()
        .map(|edge| [edge[0].min(edge[1]), edge[0].max(edge[1])])
        .collect();
    if !sharp_pairs.is_empty() {
        let indices = indices_by_id(&mesh);
        let values: Map<String, Value> = mesh
            .edges
            .iter()
            .map(|edge| {
                let first = indices[&edge.vertices[0]];
                let second = indices[&edge.vertices[1]];
                let pair = [first.min(second), first.max(second)];
                (
                    format!("e{}", edge.id),
                    Value::Bool(sharp_pairs.contains(&pair)),
                )
            })
            .collect();
        mesh.attributes.insert(
            "sharp".to_owned(),
            json!({"domain": "edge", "type": "bool", "values": values}),
        );
    }
    let groups = variant["groups"].as_object().expect("fixture groups");
    let mut vertex_groups = Map::new();
    for (name, weights) in groups {
        let mut values = Map::new();
        for (index, weight) in weights.as_object().expect("fixture group weights") {
            values.insert(format!("v{index}"), weight.clone());
        }
        vertex_groups.insert(name.clone(), Value::Object(values));
    }
    if !groups.is_empty() {
        mesh.attributes
            .insert("vertex_groups".to_owned(), Value::Object(vertex_groups));
    }
    mesh
}

fn modifier_for(variant: &Value) -> Modifier {
    let data = &variant["modifier"];
    Modifier {
        id: Id::new("construction_parity").expect("static modifier ID"),
        modifier_type: data["type"]
            .as_str()
            .expect("fixture modifier type")
            .to_owned(),
        name: "Construction parity".to_owned(),
        enabled: true,
        show_viewport: true,
        show_render: true,
        params: data["params"]
            .as_object()
            .expect("fixture modifier params")
            .iter()
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect(),
        binding_data: None,
        runtime: ModifierRuntime::default(),
    }
}

fn indices_by_id(mesh: &Mesh) -> HashMap<u32, usize> {
    mesh.vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect()
}

fn parse_float_rows(value: &Value) -> Vec<Vec<f64>> {
    value
        .as_array()
        .expect("fixture float rows")
        .iter()
        .map(|row| {
            row.as_array()
                .expect("fixture float row")
                .iter()
                .map(number)
                .collect()
        })
        .collect()
}
fn compare_floats(actual: &[Vec<f64>], expected: &[Vec<f64>], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: row count");
    for (row, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(actual.len(), expected.len(), "{label}: row {row} width");
        for (column, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let actual = actual as f32;
            let expected = expected as f32;
            let tolerance = f32::EPSILON * 8.0 * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= tolerance,
                "{label}: [{row}][{column}] {actual} != {expected} (tol {tolerance})"
            );
        }
    }
}

fn canonical_corner_uvs(mesh: &Mesh, name: &str) -> Vec<Vec<f64>> {
    let store = mesh
        .attributes
        .get("uv_layers")
        .and_then(Value::as_object)
        .expect("canonical UV store");
    let layer = store["layers"]
        .as_array()
        .expect("canonical UV layers")
        .iter()
        .find(|layer| layer["name"].as_str() == Some(name))
        .expect("canonical UV layer");
    let faces = layer["faces"].as_array().expect("canonical UV face rows");
    mesh.faces
        .iter()
        .flat_map(|face| {
            let row = faces
                .iter()
                .find(|row| row["face_id"].as_u64() == Some(u64::from(face.id)))
                .unwrap_or_else(|| panic!("canonical UV row missing for face {}", face.id));
            parse_float_rows(&row["uv"])
        })
        .collect()
}

fn canonical_corner_colors(mesh: &Mesh, name: &str) -> Vec<Vec<f64>> {
    let attribute = &mesh.attributes[name];
    let values = attribute["values"]
        .as_object()
        .expect("canonical color values");
    mesh.faces
        .iter()
        .flat_map(|face| parse_float_rows(&values[&format!("f{}", face.id)]))
        .collect()
}

fn blender_attribute_values<'a>(mesh: &'a Mesh, name: &str) -> Option<&'a Value> {
    mesh.attributes
        .get("blender_attributes")
        .and_then(Value::as_array)?
        .iter()
        .find(|attribute| attribute.get("name").and_then(Value::as_str) == Some(name))?
        .get("values")
}

fn compare_positions(mesh: &Mesh, expected: &Value, label: &str) {
    let positions = parse_float_rows(expected);
    assert_eq!(
        mesh.vertices.len(),
        positions.len(),
        "{label}: vertex count"
    );
    for (index, (vertex, expected)) in mesh.vertices.iter().zip(positions).enumerate() {
        for axis in 0..3 {
            let actual = vertex.co[axis] as f32;
            let expected = expected[axis] as f32;
            let tolerance = f32::EPSILON * 8.0 * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= tolerance,
                "{label}: vertex {index} axis {axis}: {actual} != {expected} (tol {tolerance})"
            );
        }
    }
}

fn compare_output(mesh: &Mesh, expected: &Value, group_names: &[String], label: &str) {
    let counts = &expected["counts"];
    assert_eq!(
        mesh.vertices.len(),
        counts["vertices"].as_u64().unwrap() as usize,
        "{label}: vertices"
    );
    assert_eq!(
        mesh.edges.len(),
        counts["edges"].as_u64().unwrap() as usize,
        "{label}: edges"
    );
    assert_eq!(
        mesh.faces.len(),
        counts["faces"].as_u64().unwrap() as usize,
        "{label}: faces"
    );
    let corner_count: usize = mesh.faces.iter().map(|face| face.vertices.len()).sum();
    assert_eq!(
        corner_count,
        counts["corners"].as_u64().unwrap() as usize,
        "{label}: corners"
    );
    compare_positions(mesh, &expected["vertices"], label);

    let indices = indices_by_id(mesh);
    let actual_faces: Vec<Vec<usize>> = mesh
        .faces
        .iter()
        .map(|face| face.vertices.iter().map(|id| indices[id]).collect())
        .collect();
    assert_eq!(
        actual_faces,
        index_rows(&expected["faces"]),
        "{label}: ordered face indices"
    );
    let expected_materials: Vec<u32> = expected["materials"]
        .as_array()
        .expect("fixture face materials")
        .iter()
        .map(|material| material.as_u64().expect("fixture material") as u32)
        .collect();
    let actual_materials: Vec<u32> = mesh.faces.iter().map(|face| face.material_index).collect();
    assert_eq!(
        actual_materials, expected_materials,
        "{label}: face material indices"
    );

    let mut actual_edges: Vec<[usize; 2]> = mesh
        .edges
        .iter()
        .map(|edge| {
            let mut pair = [indices[&edge.vertices[0]], indices[&edge.vertices[1]]];
            pair.sort_unstable();
            pair
        })
        .collect();
    actual_edges.sort_unstable();
    let mut expected_edges: Vec<[usize; 2]> = index_rows(&expected["edges"])
        .into_iter()
        .map(|edge| [edge[0], edge[1]])
        .collect();
    expected_edges.sort_unstable();
    assert_eq!(
        actual_edges, expected_edges,
        "{label}: edge endpoint indices"
    );
    let expected_sharp = expected["sharp"].as_array().expect("fixture sharp flags");
    let mut expected_sharp_by_edge = HashMap::new();
    for (edge, value) in index_rows(&expected["edges"])
        .into_iter()
        .zip(expected_sharp)
    {
        expected_sharp_by_edge.insert(
            [edge[0].min(edge[1]), edge[0].max(edge[1])],
            value.as_bool().expect("fixture sharp flag"),
        );
    }
    let sharp_values = mesh
        .attributes
        .get("sharp")
        .and_then(|attribute| attribute.get("values"))
        .and_then(Value::as_object);
    let mut actual_sharp_by_edge = HashMap::new();
    for edge in &mesh.edges {
        let first = indices[&edge.vertices[0]];
        let second = indices[&edge.vertices[1]];
        let pair = [first.min(second), first.max(second)];
        let sharp = sharp_values
            .and_then(|values| values.get(&format!("e{}", edge.id)))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        actual_sharp_by_edge.insert(pair, sharp);
    }
    assert_eq!(
        actual_sharp_by_edge, expected_sharp_by_edge,
        "{label}: sharp edge flags"
    );

    let mut expected_uv = parse_float_rows(&expected["uv"]);
    for value in &mut expected_uv {
        value[1] = 1.0 - value[1];
    }
    let actual_uv = canonical_corner_uvs(mesh, "UVMap");
    assert_eq!(actual_uv, expected_uv, "{label}: UV corner values");
    let uv_store = &mesh.attributes["uv_layers"];
    let actual_uv_names: Vec<&str> = uv_store["layers"]
        .as_array()
        .expect("canonical UV layers")
        .iter()
        .map(|layer| layer["name"].as_str().expect("canonical UV layer name"))
        .collect();
    let expected_uv_names: Vec<&str> = expected["uv_metadata"]["layers"]
        .as_array()
        .expect("Blender UV layer names")
        .iter()
        .map(|name| name.as_str().expect("Blender UV layer name"))
        .collect();
    assert_eq!(
        actual_uv_names, expected_uv_names,
        "{label}: UV layer order"
    );
    assert_eq!(
        uv_store["active"], expected["uv_metadata"]["active"],
        "{label}: active UV layer"
    );
    assert_eq!(
        uv_store["render"], expected["uv_metadata"]["render"],
        "{label}: render UV layer"
    );
    let expected_colors = parse_float_rows(&expected["colors"]);
    let actual_colors = canonical_corner_colors(mesh, "Col");
    assert_eq!(
        mesh.attributes.get("color_layers"),
        Some(&expected["color_layers"]),
        "{label}: color layer order and active/render names"
    );
    assert_eq!(
        mesh.attributes["blender_metadata"]["color_data_types"]["Col"], expected["color_data_type"],
        "{label}: color source encoding"
    );
    compare_floats(
        &actual_colors,
        &expected_colors,
        &format!("{label}: corner colors"),
    );
    assert!(
        !mesh
            .attributes
            .get("blender_attributes")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .any(|attribute| attribute.get("name").and_then(Value::as_str) == Some("Col")),
        "{label}: named colors are native and must not be duplicated"
    );

    let expected_crease = expected["crease"]
        .as_array()
        .expect("fixture crease values");
    let actual_crease = blender_attribute_values(mesh, "crease_edge")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_else(|| vec![json!(0.0); mesh.edges.len()]);
    assert_eq!(
        actual_crease.len(),
        mesh.edges.len(),
        "{label}: crease edge data length"
    );
    let mut expected_crease_by_edge = HashMap::new();
    for (edge, value) in index_rows(&expected["edges"])
        .into_iter()
        .zip(expected_crease)
    {
        expected_crease_by_edge.insert([edge[0].min(edge[1]), edge[0].max(edge[1])], number(value));
    }
    let mut actual_crease_by_edge = HashMap::new();
    for (edge, value) in mesh.edges.iter().zip(actual_crease) {
        let pair = [
            indices[&edge.vertices[0]].min(indices[&edge.vertices[1]]),
            indices[&edge.vertices[0]].max(indices[&edge.vertices[1]]),
        ];
        actual_crease_by_edge.insert(pair, number(&value));
    }
    for (edge, expected) in expected_crease_by_edge {
        assert_eq!(
            actual_crease_by_edge.get(&edge),
            Some(&expected),
            "{label}: crease on {edge:?}"
        );
    }

    let output_groups = expected["groups"]
        .as_object()
        .expect("fixture output groups");
    let mesh_groups = mesh
        .attributes
        .get("vertex_groups")
        .and_then(Value::as_object);
    for name in group_names {
        let expected_weights: Vec<f64> = output_groups[name]
            .as_array()
            .expect("fixture group output")
            .iter()
            .map(number)
            .collect();
        assert_eq!(
            expected_weights.len(),
            mesh.vertices.len(),
            "{label}: {name} output group length"
        );
        let values = mesh_groups
            .and_then(|groups| groups.get(name))
            .and_then(Value::as_object);
        let actual_weights: Vec<f64> = mesh
            .vertices
            .iter()
            .map(|vertex| {
                values
                    .and_then(|weights| weights.get(&format!("v{}", vertex.id)))
                    .map_or(0.0, number)
            })
            .collect();
        assert_eq!(actual_weights, expected_weights, "{label}: {name} weights");
    }
}

fn run_fixture(document: &str, modifier_type: &str) {
    let fixture: Value = serde_json::from_str(document).expect("valid committed Blender fixture");
    assert_eq!(fixture["blender_version"], "5.2.2 LTS");
    for variant in fixture["variants"].as_array().expect("fixture variants") {
        assert_eq!(variant["modifier"]["type"], modifier_type);
        let mesh = mesh_for(variant);
        let modifier = modifier_for(variant);
        let groups: Vec<String> = variant["groups"]
            .as_object()
            .expect("fixture groups")
            .keys()
            .cloned()
            .collect();
        for case in variant["cases"].as_array().expect("fixture cases") {
            let frame = number(&case["frame"]);
            let evaluated = evaluate_modifiers_with_groups_at_frame(
                &mesh,
                std::slice::from_ref(&modifier),
                &Registry::<NodeGroup>::new(),
                frame,
                SUBJECT_NAME,
                Some(
                    variant["material_slots"]
                        .as_u64()
                        .expect("fixture slot count") as usize,
                ),
            )
            .expect("modifier evaluation succeeds");
            let label = format!("{} frame {frame}", variant["name"].as_str().unwrap());
            compare_output(&evaluated.mesh, &case["output"], &groups, &label);
        }
    }
}

#[test]
fn build_matches_blender_522_outputs_at_frames_and_random_seeds() {
    run_fixture(BUILD_FIXTURE, "build");
}
#[test]
fn build_face_prefix_keeps_valid_custom_normal_corner_order() {
    let mut mesh = Mesh::from_positions_and_faces(
        vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(1.0, 1.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(0.0, 0.0, 2.0),
            DVec3::new(1.0, 0.0, 2.0),
            DVec3::new(1.0, 1.0, 2.0),
            DVec3::new(0.0, 1.0, 2.0),
        ],
        vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
    )
    .expect("valid two-quad mesh");
    let first_face = mesh.faces[0].id;
    let second_face = mesh.faces[1].id;
    let first_codes = json!([[12000, 3000], [13000, -2000], [9000, 4000], [15000, 1000]]);
    let mut values = Map::new();
    values.insert(format!("f{first_face}"), first_codes.clone());
    values.insert(
        format!("f{second_face}"),
        json!([
            [-11000, 2000],
            [-9000, -3000],
            [-14000, 1000],
            [-12000, -4000]
        ]),
    );
    let mut corner_order = Map::new();
    corner_order.insert(format!("f{first_face}"), json!([1, 0, 2, 3]));
    corner_order.insert(format!("f{second_face}"), json!([5, 4, 6, 7]));
    mesh.attributes.insert(
        "custom_normal".to_owned(),
        json!({
            "domain": "corner",
            "type": "normal_code",
            "values": values,
            "corner_order": corner_order,
        }),
    );

    let modifier = Modifier {
        id: Id::new("build_normal_prefix").expect("static modifier ID"),
        modifier_type: "build".to_owned(),
        name: "Build".to_owned(),
        enabled: true,
        show_viewport: true,
        show_render: true,
        params: serde_json::from_value(json!({
            "frame_start": 1.0,
            "frame_duration": 2.0,
            "use_reverse": false,
            "use_random_order": false,
            "seed": 0,
        }))
        .expect("valid Build parameters"),
        binding_data: None,
        runtime: ModifierRuntime::default(),
    };
    let output = evaluate_modifiers_with_groups_at_frame(
        &mesh,
        &[modifier],
        &Registry::<NodeGroup>::new(),
        2.0,
        SUBJECT_NAME,
        None,
    )
    .expect("Build preserves the corner-normal state on a face prefix")
    .mesh;
    assert_eq!(output.faces.len(), 1, "Build emits one source face");
    assert_eq!(output.faces[0].vertices.len(), 4, "Build emits one quad");

    let state = &output.attributes["custom_normal"];
    assert_eq!(state["type"], "normal_code");
    assert_eq!(
        state["values"][&format!("f{}", output.faces[0].id)],
        first_codes,
        "identical raw codes and retained loop basis preserve decoded corner normals",
    );
    let order_rows = state["corner_order"]
        .as_object()
        .expect("valid compacted corner-order rows");
    assert_eq!(
        order_rows.len(),
        1,
        "the omitted face has no loop-order row"
    );
    let order: Vec<usize> = order_rows[&format!("f{}", output.faces[0].id)]
        .as_array()
        .expect("four-corner loop order")
        .iter()
        .map(|ordinal| ordinal.as_u64().expect("corner ordinal") as usize)
        .collect();
    assert_eq!(order, [1, 0, 2, 3], "retained nonidentity loop order");
    let mut sorted = order;
    sorted.sort_unstable();
    assert_eq!(sorted, [0, 1, 2, 3], "valid four-corner permutation");
}

#[test]
fn wireframe_matches_blender_522_geometry_attributes_groups_and_materials() {
    run_fixture(WIREFRAME_FIXTURE, "wireframe");
}

#[test]
fn wireframe_material_offset_rejects_non_integer_values() {
    let mesh =
        Mesh::from_positions_and_faces(vec![DVec3::ZERO, DVec3::X, DVec3::Y], vec![vec![0, 1, 2]])
            .expect("valid triangle");
    let modifier = Modifier {
        id: Id::new("wireframe_non_integer_offset").expect("static modifier ID"),
        modifier_type: "wireframe".to_owned(),
        name: "Wireframe".to_owned(),
        enabled: true,
        show_viewport: true,
        show_render: true,
        params: serde_json::from_value(json!({"material_offset": 1.5})).expect("modifier params"),
        binding_data: None,
        runtime: ModifierRuntime::default(),
    };
    assert!(
        evaluate_modifiers_with_groups_at_frame(
            &mesh,
            &[modifier],
            &Registry::<NodeGroup>::new(),
            1.0,
            SUBJECT_NAME,
            Some(3),
        )
        .is_err()
    );
}
