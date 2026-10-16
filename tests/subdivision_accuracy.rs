use std::{collections::{BTreeMap, BTreeSet}, error::Error};

use glam::DVec3;
use potter_core::{
    geom::{Mesh, modifiers::evaluate_modifiers},
    model::{Id, Modifier, ModifierRuntime},
};
use serde_json::{Map, Value, json};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const ORACLE: &str = include_str!("fixtures/pr16_subdivision/oracle.json");
// Blender f32 output is rounded to seven fixture decimals; generation measured a maximum
// serialization error of 5.000000003e-8. These bounds cover f32/OpenSubdiv arithmetic only.
const POSITION_TOLERANCE: f64 = 3.0e-6;
const UV_TOLERANCE: f64 = 2.0e-6;
const NORMAL_TOLERANCE: f64 = 2.0e-6;
const CREASE_TOLERANCE: f64 = 1.0e-6;

fn number(value: &Value) -> TestResult<f64> {
    value.as_f64().ok_or_else(|| "expected a number".into())
}

fn indices(value: &Value) -> TestResult<Vec<usize>> {
    value
        .as_array()
        .ok_or("expected an index array")?
        .iter()
        .map(|item| -> TestResult<usize> {
            let index = item.as_u64().ok_or("index must be an unsigned integer")?;
            Ok(usize::try_from(index)?)
        })
        .collect()
}

fn point(value: &Value) -> TestResult<DVec3> {
    let values = value.as_array().ok_or("point must be an array")?;
    if values.len() != 3 {
        return Err("point must have three coordinates".into());
    }
    Ok(DVec3::new(
        number(&values[0])?,
        number(&values[1])?,
        number(&values[2])?,
    ))
}

fn uv(value: &Value) -> TestResult<[f64; 2]> {
    let values = value.as_array().ok_or("UV must be an array")?;
    if values.len() != 2 {
        return Err("UV must have two coordinates".into());
    }
    Ok([number(&values[0])?, number(&values[1])?])
}

fn canonical_cycle(vertices: &[usize]) -> (Vec<usize>, usize) {
    let start = (0..vertices.len())
        .min_by_key(|offset| {
            (0..vertices.len())
                .map(|index| vertices[(*offset + index) % vertices.len()])
                .collect::<Vec<_>>()
        })
        .expect("face cycle is non-empty");
    let cycle = (0..vertices.len())
        .map(|index| vertices[(start + index) % vertices.len()])
        .collect();
    (cycle, start)
}

fn rotated<T: Clone>(values: &[T], start: usize) -> Vec<T> {
    (0..values.len())
        .map(|index| values[(start + index) % values.len()].clone())
        .collect()
}

fn edge_key(first: usize, second: usize) -> (usize, usize) {
    if first <= second { (first, second) } else { (second, first) }
}

fn input_mesh(spec: &Value) -> TestResult<Mesh> {
    let vertices = spec["vertices"]
        .as_array()
        .ok_or("mesh vertices missing")?
        .iter()
        .map(point)
        .collect::<TestResult<Vec<_>>>()?;
    let faces = spec["faces"]
        .as_array()
        .ok_or("mesh faces missing")?
        .iter()
        .map(indices)
        .collect::<TestResult<Vec<_>>>()?;
    let mut mesh = Mesh::from_positions_and_faces(vertices, faces)?;

    let seams = spec["seams"]
        .as_array()
        .ok_or("mesh seams missing")?
        .iter()
        .map(indices)
        .map(|edge| {
            let edge = edge?;
            if edge.len() != 2 {
                return Err("seam must reference two vertices".into());
            }
            Ok(edge_key(edge[0], edge[1]))
        })
        .collect::<TestResult<BTreeSet<_>>>()?;
    let crease_pairs = spec["creases"]
        .as_array()
        .ok_or("mesh creases missing")?
        .iter()
        .map(|item| -> TestResult<((usize, usize), f64)> {
            let values = item.as_array().ok_or("crease must be an array")?;
            if values.len() != 3 {
                return Err("crease must have two endpoints and a value".into());
            }
            let first = usize::try_from(values[0].as_u64().ok_or("crease endpoint missing")?)?;
            let second = usize::try_from(values[1].as_u64().ok_or("crease endpoint missing")?)?;
            Ok((edge_key(first, second), number(&values[2])?))
        })
        .collect::<TestResult<BTreeMap<_, _>>>()?;
    let mut flags = Vec::with_capacity(mesh.edges.len());
    let mut crease_values = Map::new();
    for edge in &mesh.edges {
        let endpoints = edge_key(edge.vertices[0] as usize, edge.vertices[1] as usize);
        flags.push(json!({"sharp":false, "seam":seams.contains(&endpoints)}));
        if let Some(value) = crease_pairs.get(&endpoints) {
            crease_values.insert(format!("e{}", edge.id), json!(value));
        }
    }
    mesh.attributes
        .insert("blender_edge_flags".to_owned(), Value::Array(flags));
    mesh.attributes
        .insert("blender_polygon_smooth".to_owned(), spec["smooth"].clone());
    mesh.attributes
        .insert("blender_uv_layers".to_owned(), spec["uv_layers"].clone());
    mesh.attributes.insert(
        "crease_edge".to_owned(),
        json!({"domain":"edge", "type":"float", "values":crease_values}),
    );

    let source_normals = spec["custom_normals"]
        .as_array()
        .ok_or("mesh custom normals missing")?;
    let mut corner = 0;
    let mut normal_values = Map::new();
    for face in &mesh.faces {
        let end = corner + face.vertices.len();
        if end > source_normals.len() {
            return Err("source custom normals do not cover every corner".into());
        }
        normal_values.insert(
            format!("f{}", face.id),
            Value::Array(source_normals[corner..end].to_vec()),
        );
        corner = end;
    }
    if corner != source_normals.len() {
        return Err("source custom normals contain extra corners".into());
    }
    mesh.attributes.insert(
        "custom_normal".to_owned(),
        json!({"domain":"corner", "type":"float3", "values":normal_values}),
    );
    Ok(mesh)
}

fn compare_case(case: &Value, spec: &Value) -> TestResult {
    let name = case["name"].as_str().ok_or("case name missing")?;
    let params: Map<String, Value> = serde_json::from_value(case["params"].clone())?;
    let mesh = input_mesh(spec)?;
    let modifier = Modifier {
        id: Id::new(format!("subdivision_{name}"))?,
        modifier_type: "subdivision".to_owned(),
        name: "Subdivision oracle".to_owned(),
        enabled: true,
        params,
        binding_data: None,
        runtime: ModifierRuntime::default(),
    };
    let actual = evaluate_modifiers(&mesh, &[modifier])?;
    let expected = &case["expected"];
    let expected_positions = expected["vertices"]
        .as_array()
        .ok_or("oracle positions missing")?
        .iter()
        .map(point)
        .collect::<TestResult<Vec<_>>>()?;
    assert_eq!(
        actual.vertices.len(),
        expected_positions.len(),
        "{name}: vertex count"
    );
    assert_eq!(
        actual.edges.len(),
        expected["edges"].as_array().ok_or("oracle edges missing")?.len(),
        "{name}: edge count"
    );
    assert_eq!(
        actual.faces.len(),
        expected["faces"].as_array().ok_or("oracle faces missing")?.len(),
        "{name}: face count"
    );

    let mut seen_positions = vec![false; expected_positions.len()];
    let mut actual_to_expected = Vec::with_capacity(actual.vertices.len());
    let mut max_position_error: f64 = 0.0;
    for vertex in &actual.vertices {
        let (expected_index, error) = expected_positions
            .iter()
            .enumerate()
            .map(|(index, position)| (index, vertex.co.distance(*position)))
            .min_by(|first, second| first.1.total_cmp(&second.1))
            .ok_or("oracle has no vertices")?;
        max_position_error = max_position_error.max(error);
        assert!(
            !seen_positions[expected_index],
            "{name}: two Potter vertices map to Blender vertex {expected_index}"
        );
        seen_positions[expected_index] = true;
        actual_to_expected.push(expected_index);
    }
    assert!(seen_positions.iter().all(|seen| *seen), "{name}: missing Blender vertex");
    assert!(
        max_position_error <= POSITION_TOLERANCE,
        "{name}: max position error {max_position_error:.9e} exceeds {POSITION_TOLERANCE:.9e}"
    );

    let actual_vertex_indices = actual
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<BTreeMap<_, _>>();
    let expected_faces = expected["faces"].as_array().ok_or("oracle faces missing")?;
    let mut expected_cycles = Vec::with_capacity(expected_faces.len());
    let mut expected_smooth = BTreeMap::new();
    let mut expected_uv = BTreeMap::new();
    let mut expected_normals = BTreeMap::new();
    for face in expected_faces {
        let vertices = indices(&face["v"])?;
        let (cycle, rotation) = canonical_cycle(&vertices);
        expected_cycles.push(cycle.clone());
        expected_smooth.insert(
            cycle.clone(),
            face["smooth"].as_bool().ok_or("oracle smooth flag missing")?,
        );
        let mut face_uv = BTreeMap::new();
        for (layer, corners) in face["uvs"].as_object().ok_or("oracle UV map missing")? {
            let values = corners
                .as_array()
                .ok_or("oracle UV corners missing")?
                .iter()
                .map(uv)
                .collect::<TestResult<Vec<_>>>()?;
            face_uv.insert(layer.clone(), rotated(&values, rotation));
        }
        expected_uv.insert(cycle.clone(), face_uv);
        if expected["has_custom_normals"].as_bool() == Some(true) {
            let values = face["normals"]
                .as_array()
                .ok_or("oracle custom normals missing")?
                .iter()
                .map(point)
                .collect::<TestResult<Vec<_>>>()?;
            expected_normals.insert(cycle, rotated(&values, rotation));
        }
    }

    let mut actual_cycles = Vec::with_capacity(actual.faces.len());
    let mut actual_uv = BTreeMap::new();
    let uv_layers = actual
        .attributes
        .get("blender_uv_layers")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{name}: output UV layers missing"))?;
    let mut flattened_uv = BTreeMap::new();
    for layer in uv_layers {
        let layer_name = layer["name"].as_str().ok_or("output UV layer name missing")?;
        let values = layer["values"].as_array().ok_or("output UV values missing")?;
        flattened_uv.insert(
            layer_name.to_owned(),
            values.iter().map(uv).collect::<TestResult<Vec<_>>>()?,
        );
    }
    let mut loop_offset = 0;
    for face in &actual.faces {
        let vertices = face
            .vertices
            .iter()
            .map(|id| {
                let index = actual_vertex_indices
                    .get(id)
                    .ok_or("output face references a missing vertex")?;
                Ok(actual_to_expected[*index])
            })
            .collect::<TestResult<Vec<_>>>()?;
        let (cycle, rotation) = canonical_cycle(&vertices);
        actual_cycles.push(cycle.clone());
        let mut face_uv = BTreeMap::new();
        for (layer_name, values) in &flattened_uv {
            let end = loop_offset + face.vertices.len();
            if end > values.len() {
                return Err(format!("{name}: UV layer {layer_name} is shorter than face loops").into());
            }
            face_uv.insert(layer_name.clone(), rotated(&values[loop_offset..end], rotation));
        }
        actual_uv.insert(cycle, face_uv);
        loop_offset += face.vertices.len();
    }
    for values in flattened_uv.values() {
        assert_eq!(loop_offset, values.len(), "{name}: UV layer loop count");
    }
    expected_cycles.sort();
    actual_cycles.sort();
    assert_eq!(
        actual_cycles, expected_cycles,
        "{name}: canonical oriented face topology"
    );
    assert_eq!(
        actual_uv.keys().collect::<Vec<_>>(),
        expected_uv.keys().collect::<Vec<_>>(),
        "{name}: UV face coverage"
    );
    let input_uv_names = spec["uv_layers"]
        .as_array()
        .ok_or("input UV layers missing")?
        .iter()
        .map(|layer| {
            layer["name"]
                .as_str()
                .map(str::to_owned)
                .ok_or("input UV layer name missing")
        })
        .collect::<TestResult<BTreeSet<_>>>()?;
    assert_eq!(
        flattened_uv.keys().cloned().collect::<BTreeSet<_>>(),
        input_uv_names,
        "{name}: UV layer names"
    );

    let mut max_uv_error: f64 = 0.0;
    for (cycle, expected_layers) in &expected_uv {
        let actual_layers = &actual_uv[cycle];
        assert_eq!(actual_layers.keys().collect::<Vec<_>>(), expected_layers.keys().collect::<Vec<_>>(),
                   "{name}: UV layers on face {cycle:?}");
        for (layer, expected_values) in expected_layers {
            let actual_values = &actual_layers[layer];
            assert_eq!(actual_values.len(), expected_values.len(), "{name}: UV corner count");
            for (actual_uv, expected_uv) in actual_values.iter().zip(expected_values) {
                max_uv_error = max_uv_error
                    .max((actual_uv[0] - expected_uv[0]).abs())
                    .max((actual_uv[1] - expected_uv[1]).abs());
            }
        }
    }
    assert!(
        max_uv_error <= UV_TOLERANCE,
        "{name}: max corner UV error {max_uv_error:.9e} exceeds {UV_TOLERANCE:.9e}"
    );

    let actual_smooth = actual
        .attributes
        .get("blender_polygon_smooth")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{name}: output polygon smooth flags missing"))?;
    assert_eq!(actual_smooth.len(), actual.faces.len(), "{name}: smooth flag count");
    let mut actual_smooth_by_face = BTreeMap::new();
    for (face, smooth) in actual.faces.iter().zip(actual_smooth) {
        let vertices = face
            .vertices
            .iter()
            .map(|id| {
                let index = actual_vertex_indices
                    .get(id)
                    .ok_or("output face references a missing vertex")?;
                Ok(actual_to_expected[*index])
            })
            .collect::<TestResult<Vec<_>>>()?;
        let smooth = smooth.as_bool().ok_or("output smooth flag is not boolean")?;
        actual_smooth_by_face.insert(canonical_cycle(&vertices).0, smooth);
    }
    assert_eq!(
        actual_smooth_by_face, expected_smooth,
        "{name}: polygon interpolation flags"
    );

    let expected_edges = expected["edges"].as_array().ok_or("oracle edges missing")?;
    let mut expected_edge_flags = BTreeMap::new();
    for edge in expected_edges {
        let pair = indices(&edge["v"])?;
        if pair.len() != 2 {
            return Err("oracle edge must reference two vertices".into());
        }
        expected_edge_flags.insert(
            edge_key(pair[0], pair[1]),
            (
                edge["seam"].as_bool().ok_or("oracle seam flag missing")?,
                number(&edge["crease"])?,
            ),
        );
    }
    let actual_flags = actual
        .attributes
        .get("blender_edge_flags")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{name}: output edge flags missing"))?;
    assert_eq!(actual_flags.len(), actual.edges.len(), "{name}: edge flag count");
    let crease_values = actual
        .attributes
        .get("crease_edge")
        .and_then(|attribute| attribute.get("values"))
        .and_then(Value::as_object);
    let mut actual_edge_flags = BTreeMap::new();
    let mut max_crease_error: f64 = 0.0;
    for (edge_index, edge) in actual.edges.iter().enumerate() {
        let endpoints = edge.vertices.map(|id| {
            let index = actual_vertex_indices[&id];
            actual_to_expected[index]
        });
        let flags = &actual_flags[edge_index];
        let seam = flags["seam"].as_bool().ok_or("output seam flag is not boolean")?;
        let crease = crease_values
            .and_then(|values| values.get(&format!("e{}", edge.id)))
            .map(number)
            .transpose()?
            .unwrap_or(0.0);
        actual_edge_flags.insert(edge_key(endpoints[0], endpoints[1]), (seam, crease));
    }
    assert_eq!(
        actual_edge_flags.keys().collect::<Vec<_>>(),
        expected_edge_flags.keys().collect::<Vec<_>>(),
        "{name}: canonical edge topology"
    );
    for (edge, expected_flags) in &expected_edge_flags {
        let actual_flags = actual_edge_flags[edge];
        assert_eq!(actual_flags.0, expected_flags.0, "{name}: seam flag on {edge:?}");
        max_crease_error = max_crease_error.max((actual_flags.1 - expected_flags.1).abs());
    }
    assert!(
        max_crease_error <= CREASE_TOLERANCE,
        "{name}: max propagated crease error {max_crease_error:.9e} exceeds {CREASE_TOLERANCE:.9e}"
    );

    let expected_has_custom = expected["has_custom_normals"]
        .as_bool()
        .ok_or("oracle custom-normal flag missing")?;
    let actual_custom = actual
        .attributes
        .get("custom_normal")
        .and_then(|attribute| attribute.get("values"))
        .and_then(Value::as_object);
    assert_eq!(
        actual_custom.is_some(),
        expected_has_custom,
        "{name}: custom-normal interpolation flag"
    );
    if expected_has_custom {
        let custom_values = actual_custom.ok_or("custom normals missing")?;
        let mut actual_normals = BTreeMap::new();
        for face in &actual.faces {
            let vertices = face
                .vertices
                .iter()
                .map(|id| {
                    let index = actual_vertex_indices
                        .get(id)
                        .ok_or("output face references a missing vertex")?;
                    Ok(actual_to_expected[*index])
                })
                .collect::<TestResult<Vec<_>>>()?;
            let (cycle, rotation) = canonical_cycle(&vertices);
            let values = custom_values
                .get(&format!("f{}", face.id))
                .and_then(Value::as_array)
                .ok_or("custom normals missing for output face")?
                .iter()
                .map(point)
                .collect::<TestResult<Vec<_>>>()?;
            actual_normals.insert(cycle, rotated(&values, rotation));
        }
        assert_eq!(
            actual_normals.keys().collect::<Vec<_>>(),
            expected_normals.keys().collect::<Vec<_>>(),
            "{name}: custom-normal face coverage"
        );
        let mut max_normal_error: f64 = 0.0;
        for (cycle, expected_values) in &expected_normals {
            let actual_values = actual_normals
                .get(cycle)
                .ok_or("custom-normal face missing")?;
            assert_eq!(
                actual_values.len(),
                expected_values.len(),
                "{name}: normal corner count"
            );
            for (actual_normal, expected_normal) in actual_values.iter().zip(expected_values) {
                max_normal_error = max_normal_error
                    .max((actual_normal.x - expected_normal.x).abs())
                    .max((actual_normal.y - expected_normal.y).abs())
                    .max((actual_normal.z - expected_normal.z).abs());
            }
        }
        assert!(
            max_normal_error <= NORMAL_TOLERANCE,
            "{name}: max custom-normal error {max_normal_error:.9e} exceeds {NORMAL_TOLERANCE:.9e}"
        );
    }
    Ok(())
}

#[test]
fn subdivision_surface_matches_blender_522_topology_attributes_and_settings() -> TestResult {
    let oracle: Value = serde_json::from_str(ORACLE)?;
    assert_eq!(oracle["blender_version"], "5.2.2 LTS");
    assert_eq!(oracle["round_digits"], 7);
    let max_serialization_error = oracle["max_serialization_rounding_error"]
        .as_str()
        .ok_or("fixture rounding metadata missing")?
        .parse::<f64>()?;
    assert!(
        max_serialization_error <= 5.1e-8,
        "fixture max serialization rounding error {max_serialization_error:.9e} exceeds the seven-decimal bound"
    );
    let meshes = oracle["meshes"].as_object().ok_or("oracle mesh inputs missing")?;
    let cases = oracle["cases"].as_array().ok_or("oracle cases missing")?;
    let levels = cases
        .iter()
        .map(|case| -> TestResult<u64> {
            Ok(case["params"]["levels"].as_u64().ok_or("level missing")?)
        })
        .collect::<TestResult<BTreeSet<_>>>()?;
    assert_eq!(levels, BTreeSet::from([1, 2, 3, 4, 5]));
    let qualities = cases
        .iter()
        .map(|case| -> TestResult<u64> {
            Ok(case["params"]["quality"].as_u64().ok_or("quality missing")?)
        })
        .collect::<TestResult<BTreeSet<_>>>()?;
    assert_eq!(qualities, BTreeSet::from([1, 2, 3, 4, 5, 6]));
    let uv_cases = cases
        .iter()
        .filter(|case| case["name"].as_str().is_some_and(|name| name.starts_with("uv_")))
        .collect::<Vec<_>>();
    let uv_modes = uv_cases
        .iter()
        .map(|case| {
            case["name"]
                .as_str()
                .and_then(|name| name.strip_prefix("uv_"))
                .unwrap_or_default()
                .to_ascii_uppercase()
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        uv_modes,
        BTreeSet::from([
            "NONE".to_owned(),
            "PRESERVE_CORNERS".to_owned(),
            "PRESERVE_CORNERS_AND_JUNCTIONS".to_owned(),
            "PRESERVE_CORNERS_JUNCTIONS_AND_CONCAVE".to_owned(),
            "PRESERVE_BOUNDARIES".to_owned(),
            "SMOOTH_ALL".to_owned(),
        ])
    );
    for case in &uv_cases {
        let mode = case["name"]
            .as_str()
            .and_then(|name| name.strip_prefix("uv_"))
            .ok_or("UV mode case name is malformed")?
            .to_ascii_uppercase();
        assert_eq!(case["params"]["uv_smooth"].as_str(), Some(mode.as_str()));
    }
    assert!(uv_cases.iter().all(|case| case["mesh"].as_str() == Some("cube")));
    let boundaries = cases
        .iter()
        .map(|case| -> TestResult<String> {
            Ok(case["params"]["boundary_smooth"]
                .as_str()
                .ok_or("boundary rule missing")?
                .to_owned())
        })
        .collect::<TestResult<BTreeSet<_>>>()?;
    assert_eq!(
        boundaries,
        BTreeSet::from(["ALL".to_owned(), "PRESERVE_CORNERS".to_owned()])
    );
    let boundary_cases = cases
        .iter()
        .filter(|case| case["name"].as_str().is_some_and(|name| name.starts_with("boundary_")))
        .collect::<Vec<_>>();
    assert_eq!(boundary_cases.len(), 2);
    assert!(
        boundary_cases
            .iter()
            .all(|case| case["mesh"].as_str() == Some("curved_grid"))
    );
    let topology_cases = cases
        .iter()
        .filter_map(|case| case["name"].as_str().filter(|name| name.starts_with("topology_")))
        .map(|name| name.strip_prefix("topology_").expect("prefix"))
        .collect::<BTreeSet<_>>();
    assert_eq!(topology_cases, BTreeSet::from(["cube", "ngon", "nonmanifold"]));
    let custom_modes = cases
        .iter()
        .filter(|case| case["name"].as_str().is_some_and(|name| name.starts_with("custom_normals_")))
        .map(|case| -> TestResult<bool> {
            Ok(case["params"]["use_custom_normals"]
                .as_bool()
                .ok_or("custom-normal mode missing")?)
        })
        .collect::<TestResult<BTreeSet<_>>>()?;
    assert_eq!(custom_modes, BTreeSet::from([false, true]));
    let crease_cases = cases
        .iter()
        .filter(|case| case["name"].as_str().is_some_and(|name| name.starts_with("crease_")))
        .collect::<Vec<_>>();
    let crease_values = crease_cases
        .iter()
        .map(|case| -> TestResult<i64> {
            assert_eq!(case["params"]["levels"].as_u64(), Some(2));
            assert_eq!(case["params"]["use_creases"].as_bool(), Some(true));
            let mesh_name = case["mesh"].as_str().ok_or("crease mesh missing")?;
            let crease = meshes[mesh_name]["creases"]
                .as_array()
                .ok_or("crease inputs missing")?
                .first()
                .ok_or("crease input missing")?;
            Ok((number(&crease[2])? * 100.0).round() as i64)
        })
        .collect::<TestResult<BTreeSet<_>>>()?;
    assert_eq!(crease_values, BTreeSet::from([0, 25, 50, 100]));
    let curved_grid = meshes.get("curved_grid").ok_or("curved grid missing")?;
    let grid_positions = curved_grid["vertices"]
        .as_array()
        .ok_or("curved grid vertices missing")?
        .iter()
        .map(point)
        .collect::<TestResult<Vec<_>>>()?;
    let z_min = grid_positions.iter().map(|point| point.z).fold(f64::INFINITY, f64::min);
    let z_max = grid_positions.iter().map(|point| point.z).fold(f64::NEG_INFINITY, f64::max);
    assert!(z_max - z_min > 0.1, "curved grid must exercise non-planar geometry");
    for name in ["curved_grid", "cube", "ngon", "nonmanifold"] {
        assert!(
            !meshes[name]["seams"].as_array().ok_or("seam input missing")?.is_empty(),
            "{name} must carry seam edges"
        );
    }

    for case in cases {
        let mesh_name = case["mesh"].as_str().ok_or("case mesh missing")?;
        let spec = meshes.get(mesh_name).ok_or("case mesh not present in oracle")?;
        compare_case(case, spec)?;
    }
    Ok(())
}
