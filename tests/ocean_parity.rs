use std::{collections::HashMap, error::Error};

use glam::DVec3;
use potter_core::{
    eval::{EvaluationContext, EvaluationPurpose, Snapshot},
    geom::Mesh,
    model::{DataBlock, Id, Modifier, Node, SceneDoc},
};
use serde_json::{Map, Value, json};

const CORPUS: &str = include_str!("fixtures/ocean_parity/corpus.json");

fn vec3(value: &Value) -> Result<DVec3, Box<dyn Error>> {
    let values = value.as_array().ok_or("mesh position is not an array")?;
    Ok(DVec3::new(
        values[0].as_f64().ok_or("mesh x is not numeric")?,
        values[1].as_f64().ok_or("mesh y is not numeric")?,
        values[2].as_f64().ok_or("mesh z is not numeric")?,
    ))
}

fn source_mesh(source: &Value) -> Result<Mesh, Box<dyn Error>> {
    let positions = source["vertices"]
        .as_array()
        .ok_or("fixture source vertices are missing")?
        .iter()
        .map(vec3)
        .collect::<Result<Vec<_>, _>>()?;
    let polygons = source["polygons"]
        .as_array()
        .ok_or("fixture source polygons are missing")?
        .iter()
        .map(|polygon| {
            polygon
                .as_array()
                .ok_or("fixture polygon is not an array")?
                .iter()
                .map(|index| {
                    index
                        .as_u64()
                        .map(|index| index as usize)
                        .ok_or_else(|| "fixture polygon index is not an integer".into())
                })
                .collect::<Result<Vec<_>, Box<dyn Error>>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut mesh = Mesh::from_positions_and_faces(positions, polygons)?;
    if let Some(values) = source["uv"].as_array() {
        let mut faces = Vec::with_capacity(mesh.faces.len());
        let mut offset = 0;
        for face in &mesh.faces {
            let count = face.vertices.len();
            let row = values[offset..offset + count]
                .iter()
                .map(|value| {
                    let pair = value.as_array().ok_or("fixture UV is not an array")?;
                    let u = pair[0].as_f64().ok_or("fixture UV u is not numeric")?;
                    let v = pair[1].as_f64().ok_or("fixture UV v is not numeric")?;
                    Ok(json!([u, 1.0 - v]))
                })
                .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
            faces.push(json!({"face_id":face.id,"uv":row}));
            offset += count;
        }
        mesh.attributes.insert(
            "uv_layers".to_owned(),
            json!({"layers":[{"name":"UVMap","faces":faces}],"active":"UVMap","render":"UVMap"}),
        );
    }
    if let Some(layers) = source["color_layers"].as_object() {
        let mut layer_names = Vec::with_capacity(layers.len());
        let mut color_data_types = Map::new();
        for (name, layer) in layers {
            let colors = layer["values"]
                .as_array()
                .ok_or("fixture corner color values are missing")?;
            let mut values = Map::new();
            let mut offset = 0;
            for face in &mesh.faces {
                let count = face.vertices.len();
                values.insert(
                    format!("f{}", face.id),
                    Value::Array(colors[offset..offset + count].to_vec()),
                );
                offset += count;
            }
            mesh.attributes.insert(
                name.clone(),
                json!({"domain":"corner","type":"color","values":values}),
            );
            layer_names.push(name.clone());
            color_data_types.insert(
                name.clone(),
                json!(
                    layer["data_type"]
                        .as_str()
                        .ok_or("fixture color encoding is missing")?
                ),
            );
        }
        if !layer_names.is_empty() {
            let active = layer_names[0].clone();
            mesh.attributes.insert(
                "color_layers".to_owned(),
                json!({"layers":layer_names,"active":active.clone(),"render":active}),
            );
            mesh.attributes.insert(
                "blender_metadata".to_owned(),
                json!({"color_data_types":color_data_types}),
            );
        }
    }
    Ok(mesh)
}

fn output_polygons(mesh: &Mesh) -> Vec<Vec<usize>> {
    let indices: HashMap<u32, usize> = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    mesh.faces
        .iter()
        .map(|face| face.vertices.iter().map(|id| indices[id]).collect())
        .collect()
}
fn output_uvs(mesh: &Mesh) -> Result<Vec<[f64; 2]>, Box<dyn Error>> {
    let layer = mesh
        .attributes
        .get("uv_layers")
        .and_then(|store| store.get("layers"))
        .and_then(Value::as_array)
        .and_then(|layers| layers.first())
        .ok_or("Ocean UV layer is missing")?;
    let rows = layer["faces"]
        .as_array()
        .ok_or("Ocean UV face rows are missing")?;
    let mut by_face = HashMap::new();
    for row in rows {
        let face_id = row["face_id"]
            .as_u64()
            .ok_or("Ocean UV face ID is not an integer")?;
        let values = row["uv"]
            .as_array()
            .ok_or("Ocean UV face coordinates are missing")?;
        by_face.insert(face_id, values);
    }
    let mut output = Vec::new();
    for face in &mesh.faces {
        let values = by_face
            .get(&u64::from(face.id))
            .ok_or("Ocean UV face row is missing")?;
        if values.len() != face.vertices.len() {
            return Err("Ocean UV corner count does not match its face".into());
        }
        for value in *values {
            let pair = value
                .as_array()
                .ok_or("Ocean UV coordinate is not an array")?;
            let u = pair[0].as_f64().ok_or("Ocean UV u is not numeric")?;
            let v = pair[1].as_f64().ok_or("Ocean UV v is not numeric")?;
            output.push([u, 1.0 - v]);
        }
    }
    Ok(output)
}

fn output_corner_colors(mesh: &Mesh, name: &str) -> Result<Vec<[f64; 4]>, Box<dyn Error>> {
    let serialized = serde_json::to_value(mesh)?;
    let attributes = serialized["attributes"]
        .as_object()
        .ok_or("serialized mesh attributes are missing")?;
    let is_corner_color = |layer: &Value| layer["domain"] == "corner" && layer["type"] == "color";
    let escaped_name = format!("user:{name}");
    let layer = attributes
        .get(name)
        .filter(|layer| is_corner_color(layer))
        .or_else(|| {
            attributes
                .get(&escaped_name)
                .filter(|layer| is_corner_color(layer))
        })
        .ok_or_else(|| format!("ocean output color layer {name} is missing"))?;
    let values = layer["values"]
        .as_object()
        .ok_or("corner color values are missing")?;
    let mut output = Vec::new();
    for face in &mesh.faces {
        let face_values = values
            .get(&format!("f{}", face.id))
            .and_then(Value::as_array)
            .ok_or_else(|| format!("corner colors for face {} are missing", face.id))?;
        for color in face_values {
            let components = color.as_array().ok_or("corner color is not an array")?;
            output.push([
                components[0].as_f64().ok_or("color red is not numeric")?,
                components[1].as_f64().ok_or("color green is not numeric")?,
                components[2].as_f64().ok_or("color blue is not numeric")?,
                components[3].as_f64().ok_or("color alpha is not numeric")?,
            ]);
        }
    }
    Ok(output)
}

fn evaluate_case(
    source: Mesh,
    modifier: Modifier,
    purpose: EvaluationPurpose,
    name: &str,
) -> Result<Mesh, Box<dyn Error>> {
    let node_id = modifier.id.clone();
    let data_id = Id::new(format!("mesh_{name}"))?;
    let mut doc = SceneDoc::new(format!("ocean_parity_{name}"));
    doc.nodes.insert(
        node_id.clone(),
        Node {
            name: format!("Ocean {name}"),
            kind: "mesh".to_owned(),
            data: Some(data_id.clone()),
            modifiers: vec![modifier],
            ..Node::default()
        },
    );
    doc.collections
        .get_mut(&Id::new("collection_root")?)
        .ok_or("default scene root collection is missing")?
        .objects
        .push(node_id.clone());
    doc.data_blocks.insert(
        data_id,
        DataBlock {
            data_type: "mesh".to_owned(),
            mesh: Some(source),
            ..DataBlock::default()
        },
    );
    let mut snapshot = Snapshot::evaluate(
        &doc,
        &EvaluationContext {
            purpose,
            ..EvaluationContext::default()
        },
    )?;
    snapshot
        .meshes
        .remove(&node_id)
        .ok_or_else(|| format!("{name}: evaluated ocean mesh is missing").into())
}

#[test]
fn ocean_live_blender_corpus_matches_geometry_uvs_and_corner_colors() -> Result<(), Box<dyn Error>>
{
    let corpus: Value = serde_json::from_str(CORPUS)?;
    let cases = corpus["cases"]
        .as_array()
        .ok_or("Ocean corpus cases are missing")?;
    for case in cases
        .iter()
        .filter(|case| !case["name"].as_str().unwrap_or_default().starts_with('B'))
    {
        let name = case["name"]
            .as_str()
            .ok_or("Ocean corpus case has no name")?;
        let purpose = match case["purpose"].as_str() {
            Some("render") => EvaluationPurpose::Render,
            Some("viewport") => EvaluationPurpose::Preview,
            Some(purpose) => {
                return Err(format!("{name}: unknown evaluation purpose {purpose}").into());
            }
            None => return Err(format!("{name}: evaluation purpose is missing").into()),
        };
        let modifier = Modifier {
            id: Id::new(format!("ocean_{name}"))?,
            modifier_type: "ocean".to_owned(),
            name: "Ocean".to_owned(),
            enabled: true,
            show_viewport: true,
            show_render: true,
            params: serde_json::from_value::<Map<String, Value>>(case["params"].clone())?,
            binding_data: None,
            baked_ocean_cache: None,
            runtime: Default::default(),
        };
        let source = source_mesh(&case["source_mesh"])?;
        let output = evaluate_case(source, modifier, purpose, name)?;
        assert!(
            !output.attributes.contains_key("ocean_normal"),
            "{name}: Blender does not emit an ocean_normal mesh attribute",
        );
        let expected = &case["output"];
        let expected_vertices = expected["vertices"]
            .as_array()
            .ok_or("expected Ocean vertices are missing")?;
        assert_eq!(
            output.vertices.len(),
            expected_vertices.len(),
            "{name}: vertex count"
        );
        for (index, (actual, expected)) in output.vertices.iter().zip(expected_vertices).enumerate()
        {
            let expected = vec3(expected)?;
            let error = (actual.co - expected).abs().max_element();
            assert!(error <= 2.0e-5, "{name}: vertex {index} differs by {error}");
        }
        let expected_polygons = expected["polygons"]
            .as_array()
            .ok_or("expected Ocean polygons are missing")?;
        assert_eq!(
            output_polygons(&output),
            serde_json::from_value::<Vec<Vec<usize>>>(Value::Array(expected_polygons.clone()))?,
            "{name}: polygon topology"
        );

        if let Some(expected_uvs) = expected["uv"].as_array() {
            let actual_uvs = output_uvs(&output)?;
            assert_eq!(
                actual_uvs.len(),
                expected_uvs.len(),
                "{name}: UV corner count"
            );
            for (corner, (actual, expected)) in actual_uvs.iter().zip(expected_uvs).enumerate() {
                let expected = expected.as_array().ok_or("expected UV is not an array")?;
                for axis in 0..2 {
                    let error = (actual[axis]
                        - expected[axis]
                            .as_f64()
                            .ok_or("expected UV is not numeric")?)
                    .abs();
                    assert!(
                        error <= 1.0e-6,
                        "{name}: UV corner {corner} axis {axis} differs by {error}"
                    );
                }
            }
        }
        let expected_layers = expected["color_layers"]
            .as_object()
            .ok_or("expected Blender color layers are missing")?;
        let mut actual_names: Vec<&str> = match output.attributes.get("color_layers") {
            Some(registry) => registry["layers"]
                .as_array()
                .ok_or_else(|| format!("{name}: canonical color layer registry is malformed"))?
                .iter()
                .map(|layer| layer.as_str().ok_or("color layer name is not a string"))
                .collect::<Result<_, _>>()?,
            None if expected_layers.is_empty() => Vec::new(),
            None => return Err(format!("{name}: canonical color layer registry is missing").into()),
        };
        actual_names.sort_unstable();
        let mut expected_names: Vec<_> = expected_layers.keys().map(String::as_str).collect();
        expected_names.sort_unstable();
        assert_eq!(
            actual_names, expected_names,
            "{name}: color layer registry names"
        );

        let color_data_types = output
            .attributes
            .get("blender_metadata")
            .and_then(|metadata| metadata.get("color_data_types"))
            .and_then(Value::as_object);
        let mut actual_type_names: Vec<&str> = color_data_types
            .map(|types| types.keys().map(String::as_str).collect())
            .unwrap_or_default();
        actual_type_names.sort_unstable();
        assert_eq!(
            actual_type_names, expected_names,
            "{name}: BYTE_COLOR metadata names"
        );

        for (layer_name, expected_layer) in expected_layers {
            assert_eq!(
                color_data_types
                    .and_then(|types| types.get(layer_name))
                    .and_then(Value::as_str),
                Some("BYTE_COLOR"),
                "{name}: {layer_name} source encoding",
            );
            let actual = output_corner_colors(&output, layer_name)?;
            let expected = expected_layer["values"]
                .as_array()
                .ok_or("expected color values are missing")?;
            assert_eq!(
                actual.len(),
                expected.len(),
                "{name}: {layer_name} corner count"
            );
            for (corner, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                let expected = expected
                    .as_array()
                    .ok_or("expected color is not an array")?;
                for channel in 0..4 {
                    let error = (actual[channel]
                        - expected[channel]
                            .as_f64()
                            .ok_or("expected color is not numeric")?)
                    .abs();
                    assert!(
                        error <= 2.0e-6,
                        "{name}: {layer_name} corner {corner} channel {channel} differs by {error}"
                    );
                }
            }
        }
    }
    Ok(())
}
