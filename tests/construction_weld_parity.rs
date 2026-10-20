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

const WELD_FIXTURE: &str = include_str!("fixtures/construction_p/weld/weld.json");

fn color_row(value: &Value) -> Vec<f64> {
    value.as_array().unwrap().iter().map(number).collect()
}

fn actual_color_values(mesh: &Mesh, name: &str) -> Vec<Vec<f64>> {
    let attribute = &mesh.attributes[name];
    let values = attribute["values"]
        .as_object()
        .expect("canonical color values");
    match attribute["domain"]
        .as_str()
        .expect("canonical color domain")
    {
        "point" => mesh
            .vertices
            .iter()
            .map(|vertex| color_row(&values[&format!("v{}", vertex.id)]))
            .collect(),
        "corner" => mesh
            .faces
            .iter()
            .flat_map(|face| {
                values[&format!("f{}", face.id)]
                    .as_array()
                    .expect("canonical corner color row")
                    .iter()
                    .map(color_row)
            })
            .collect(),
        domain => panic!("unexpected canonical color domain {domain}"),
    }
}

fn install_fixture_normals(mesh: &mut Mesh, spec: &Value) {
    let Some(normals) = spec["custom_normals"]
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
    .expect("canonical custom-normal input");
}
fn blender_uv_to_potter(value: &Value) -> Vec<f64> {
    let uv = value.as_array().unwrap();
    vec![number(&uv[0]), 1.0 - number(&uv[1])]
}

fn number(value: &Value) -> f64 {
    value.as_f64().expect("fixture number")
}

fn rows(value: &Value) -> Vec<Vec<usize>> {
    value
        .as_array()
        .expect("fixture rows")
        .iter()
        .map(|row| {
            row.as_array()
                .expect("fixture row")
                .iter()
                .map(|value| value.as_u64().expect("fixture index") as usize)
                .collect()
        })
        .collect()
}

fn float_rows(value: &Value) -> Vec<Vec<f64>> {
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

fn mesh_for(variant: &Value) -> Mesh {
    let spec = &variant["mesh"];
    let vertices: Vec<DVec3> = spec["vertices"]
        .as_array()
        .expect("fixture vertices")
        .iter()
        .map(|point| {
            let point = point.as_array().expect("fixture point");
            DVec3::new(number(&point[0]), number(&point[1]), number(&point[2]))
        })
        .collect();
    let mut mesh =
        Mesh::from_positions_and_faces(vertices, rows(&spec["faces"])).expect("valid fixture mesh");
    for (face, material) in mesh
        .faces
        .iter_mut()
        .zip(spec["materials"].as_array().expect("fixture materials"))
    {
        face.material_index = material.as_u64().expect("fixture material") as u32;
    }
    for edge in rows(&spec["edges"]) {
        let first = mesh.vertices[edge[0]].id;
        let second = mesh.vertices[edge[1]].id;
        mesh.insert_edge([first, second]).expect("valid loose edge");
    }
    let input_sharp_edges: HashSet<[usize; 2]> = rows(&spec["sharp_edges"])
        .into_iter()
        .map(|edge| {
            let mut pair = [edge[0], edge[1]];
            pair.sort_unstable();
            pair
        })
        .collect();
    let vertex_indices = indices_by_id(&mesh);
    let edge_flags = mesh
        .edges
        .iter()
        .map(|edge| {
            let mut pair = [
                vertex_indices[&edge.vertices[0]],
                vertex_indices[&edge.vertices[1]],
            ];
            pair.sort_unstable();
            json!({"sharp": input_sharp_edges.contains(&pair), "seam": false})
        })
        .collect();
    if !input_sharp_edges.is_empty() {
        mesh.attributes
            .insert("blender_edge_flags".to_owned(), Value::Array(edge_flags));
    }
    for (name, value) in spec["attributes"].as_object().expect("fixture attributes") {
        mesh.attributes.insert(name.clone(), value.clone());
    }
    install_fixture_normals(&mut mesh, spec);
    let groups = variant["groups"].as_object().expect("fixture groups");
    if !groups.is_empty() {
        let mut vertex_groups = Map::new();
        for (name, weights) in groups {
            let values = weights
                .as_object()
                .expect("fixture group weights")
                .iter()
                .map(|(index, weight)| (format!("v{index}"), weight.clone()))
                .collect::<Map<String, Value>>();
            vertex_groups.insert(name.clone(), Value::Object(values));
        }
        mesh.attributes
            .insert("vertex_groups".to_owned(), Value::Object(vertex_groups));
    }
    mesh
}

fn modifier_for(case: &Value) -> Modifier {
    let params = case["params"].as_object().expect("fixture parameters");
    let mut values = Map::new();
    values.insert("mode".to_owned(), params["mode"].clone());
    values.insert("merge_threshold".to_owned(), params["threshold"].clone());
    values.insert("vertex_group".to_owned(), params["vertex_group"].clone());
    values.insert("invert_vertex_group".to_owned(), params["invert"].clone());
    if let Some(loose_edges) = params.get("loose_edges") {
        values.insert("loose_edges".to_owned(), loose_edges.clone());
    }
    Modifier {
        id: Id::new("construction_weld_parity").expect("static modifier ID"),
        modifier_type: "weld".to_owned(),
        name: "Weld parity".to_owned(),
        enabled: true,
        show_viewport: true,
        show_render: true,
        params: values,
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

fn compare_custom_normal_codes(mesh: &Mesh, label: &str) {
    let state = mesh
        .attributes
        .get("custom_normal")
        .unwrap_or_else(|| panic!("{label}: output custom-normal code state missing"));
    assert_eq!(state["type"], "normal_code", "{label}: normal storage type");
    let values = state["values"].as_object().expect("normal-code values");
    for face in &mesh.faces {
        let row = values[&format!("f{}", face.id)]
            .as_array()
            .expect("normal-code row");
        assert_eq!(
            row.len(),
            face.vertices.len(),
            "{label}: normal-code corner count"
        );
        for code in row {
            assert!(
                code.is_null()
                    || code.as_str() == Some("explicit_zero")
                    || code
                        .as_array()
                        .is_some_and(|pair| { pair.len() == 2 && pair.iter().all(Value::is_i64) }),
                "{label}: non-canonical raw normal code {code}"
            );
        }
    }
}

fn compare_output(mesh: &Mesh, expected: &Value, variant: &str, case: &Value) {
    let label = format!("{variant}: {:?}", case["params"]);
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
    let corners: usize = mesh.faces.iter().map(|face| face.vertices.len()).sum();
    assert_eq!(
        corners,
        counts["corners"].as_u64().unwrap() as usize,
        "{label}: corners"
    );

    let actual_positions: Vec<Vec<f64>> = mesh
        .vertices
        .iter()
        .map(|vertex| vec![vertex.co.x, vertex.co.y, vertex.co.z])
        .collect();
    compare_floats(
        &actual_positions,
        &float_rows(&expected["vertices"]),
        &format!("{label}: positions"),
    );

    let indices = indices_by_id(mesh);
    let actual_faces: Vec<Vec<usize>> = mesh
        .faces
        .iter()
        .map(|face| face.vertices.iter().map(|vertex| indices[vertex]).collect())
        .collect();
    assert_eq!(
        actual_faces,
        rows(&expected["faces"]),
        "{label}: ordered face corners"
    );
    let actual_materials: Vec<u32> = mesh.faces.iter().map(|face| face.material_index).collect();
    let expected_materials: Vec<u32> = expected["materials"]
        .as_array()
        .expect("fixture materials")
        .iter()
        .map(|value| value.as_u64().expect("fixture material") as u32)
        .collect();
    assert_eq!(
        actual_materials, expected_materials,
        "{label}: face materials"
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
    let mut expected_edges: Vec<[usize; 2]> = rows(&expected["edges"])
        .into_iter()
        .map(|edge| {
            let mut pair = [edge[0], edge[1]];
            pair.sort_unstable();
            pair
        })
        .collect();
    expected_edges.sort_unstable();
    assert_eq!(actual_edges, expected_edges, "{label}: edges");

    let mut actual_sharp_edges: Vec<[usize; 2]> = mesh
        .edges
        .iter()
        .zip(
            mesh.attributes
                .get("blender_edge_flags")
                .and_then(Value::as_array)
                .into_iter()
                .flatten(),
        )
        .filter(|(_, flags)| flags.get("sharp").and_then(Value::as_bool) == Some(true))
        .map(|(edge, _)| {
            let mut pair = [indices[&edge.vertices[0]], indices[&edge.vertices[1]]];
            pair.sort_unstable();
            pair
        })
        .collect();
    actual_sharp_edges.sort_unstable();
    let mut expected_sharp_edges: Vec<[usize; 2]> = rows(&expected["sharp_edges"])
        .into_iter()
        .map(|edge| {
            let mut pair = [edge[0], edge[1]];
            pair.sort_unstable();
            pair
        })
        .collect();
    expected_sharp_edges.sort_unstable();
    assert_eq!(
        actual_sharp_edges, expected_sharp_edges,
        "{label}: sharp edges"
    );

    let actual_smooth: Vec<bool> = mesh.attributes["blender_polygon_smooth"]
        .as_array()
        .expect("output polygon smooth flags")
        .iter()
        .map(|flag| flag.as_bool().expect("smooth flag"))
        .collect();
    let expected_smooth: Vec<bool> = expected["face_smooth"]
        .as_array()
        .expect("fixture smooth flags")
        .iter()
        .map(|flag| flag.as_bool().expect("smooth flag"))
        .collect();
    assert_eq!(actual_smooth, expected_smooth, "{label}: face shading");

    let actual_store = mesh.attributes.get("uv_layers");
    let expected_layers = expected["uv_layers"].as_array().expect("fixture UV layers");
    if expected_layers.is_empty() {
        assert!(actual_store.is_none(), "{label}: unexpected UV store");
        assert!(
            expected["uv_active"].is_null(),
            "{label}: empty UV active layer"
        );
        assert!(
            expected["uv_render"].is_null(),
            "{label}: empty UV render layer"
        );
    } else {
        let actual_store =
            actual_store.unwrap_or_else(|| panic!("{label}: canonical UV store missing"));
        assert_eq!(
            actual_store["active"], expected["uv_active"],
            "{label}: active UV layer"
        );
        assert_eq!(
            actual_store["render"], expected["uv_render"],
            "{label}: render UV layer"
        );
        let actual_layers = actual_store["layers"]
            .as_array()
            .expect("canonical UV layers");
        assert_eq!(
            actual_layers.len(),
            expected_layers.len(),
            "{label}: UV layer count"
        );
        for (actual_layer, expected_layer) in actual_layers.iter().zip(expected_layers) {
            assert_eq!(
                actual_layer["name"], expected_layer["name"],
                "{label}: UV layer order"
            );
            let actual_faces = actual_layer["faces"]
                .as_array()
                .expect("canonical UV faces");
            assert_eq!(
                actual_faces.len(),
                mesh.faces.len(),
                "{label}: UV face count"
            );
            let actual_values: Vec<Vec<f64>> = mesh
                .faces
                .iter()
                .flat_map(|face| {
                    let entry = actual_faces
                        .iter()
                        .find(|entry| entry["face_id"].as_u64() == Some(u64::from(face.id)))
                        .unwrap_or_else(|| panic!("{label}: UV face {} missing", face.id));
                    float_rows(&entry["uv"])
                })
                .collect();
            let expected_values: Vec<Vec<f64>> = expected_layer["values"]
                .as_array()
                .expect("fixture UV values")
                .iter()
                .map(blender_uv_to_potter)
                .collect();
            compare_floats(
                &actual_values,
                &expected_values,
                &format!("{label}: UV {}", expected_layer["name"].as_str().unwrap()),
            );
        }
    }
    let expected_colors = expected["colors"]
        .as_object()
        .expect("fixture color layers");
    if expected_colors.is_empty() {
        assert!(
            mesh.attributes.get("color_layers").is_none(),
            "{label}: unexpected color registry"
        );
    } else {
        assert_eq!(
            mesh.attributes.get("color_layers"),
            Some(&expected["color_layers"]),
            "{label}: color layer order and active/render names"
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
        let attribute = &mesh.attributes[name];
        assert_eq!(
            attribute["domain"],
            if wanted["domain"] == "POINT" {
                "point"
            } else {
                "corner"
            },
            "{label}: {name} color domain"
        );
        assert_eq!(
            color_types[name], wanted["data_type"],
            "{label}: {name} source encoding"
        );
        compare_floats(
            &actual_color_values(mesh, name),
            &float_rows(&wanted["values"]),
            &format!("{label}: color {name}"),
        );
    }
    assert!(
        mesh.attributes.get("blender_attributes").is_none(),
        "{label}: named colors are native"
    );

    let expected_corner_normals = float_rows(&expected["corner_normals"]);
    if !expected_corner_normals.is_empty() {
        compare_custom_normal_codes(mesh, &label);
    }

    let expected_groups = expected["groups"]
        .as_object()
        .expect("fixture output groups");
    let actual_groups = mesh
        .attributes
        .get("vertex_groups")
        .and_then(Value::as_object);
    for (name, values) in expected_groups {
        let values = values.as_array().expect("fixture output group weights");
        let actual = mesh
            .vertices
            .iter()
            .map(|vertex| {
                actual_groups
                    .and_then(|groups| groups.get(name))
                    .and_then(Value::as_object)
                    .and_then(|weights| weights.get(&format!("v{}", vertex.id)))
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0)
            })
            .collect::<Vec<_>>();
        let expected = values.iter().map(number).collect::<Vec<_>>();
        compare_floats(&[actual], &[expected], &format!("{label}: group {name}"));
    }
}

#[test]
fn weld_matches_blender_522_selection_topology_and_attributes() {
    let fixture: Value = serde_json::from_str(WELD_FIXTURE).expect("valid Blender fixture");
    assert_eq!(fixture["blender_version"], "5.2.2 LTS");
    for variant in fixture["variants"].as_array().expect("fixture variants") {
        let mesh = mesh_for(variant);
        for case in variant["cases"].as_array().expect("fixture cases") {
            let modifier = modifier_for(case);
            let evaluated = evaluate_modifiers_with_groups_at_frame(
                &mesh,
                &[modifier],
                &Registry::<NodeGroup>::new(),
                1.0,
                "Cube",
                Some(variant["material_slots"].as_u64().expect("material slots") as usize),
            )
            .expect("Weld evaluation succeeds");
            compare_output(
                &evaluated.mesh,
                &case["output"],
                variant["name"].as_str().expect("variant name"),
                case,
            );
        }
    }
}
