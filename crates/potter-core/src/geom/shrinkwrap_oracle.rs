// SPDX-License-Identifier: GPL-3.0-or-later

//! Shrinkwrap differential oracle against Blender 5.2.2. `tests/fixtures/shrinkwrap-differential/corpus.json`
//! holds each case's inputs and Blender's evaluated output; `generate.py` wrote both in one Blender run.
//! Each case is built as a Potter scene straight from those inputs and evaluated. Topology must match
//! exactly; positions, corner normals, UVs, colors and vertex weights must match within the corpus
//! tolerances. The test starts no Blender process.

use std::{
    collections::{BTreeMap, HashMap},
    error::Error,
};

use glam::{DQuat, DVec3, EulerRot};
use serde_json::{Map, Value, json};

use super::{
    Mesh,
    color_layers::{self, ColorDomain, ColorLayer, ColorStore},
    corner_normal_vectors,
    normals::SHARP_FACE_PRESENT,
    uv_layers::{self, UvLayer, UvStore},
};
use crate::{
    eval::{EvaluationContext, Snapshot, mesh_vertex_weights},
    model::{DataBlock, Id, Modifier, ModifierRuntime, Node, SceneDoc, Transform, VertexGroup},
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const CORPUS: &str = include_str!("../../../../tests/fixtures/shrinkwrap-differential/corpus.json");
const BLENDER_VERSION: &str = "5.2.2 LTS";
const SCENE_ID: &str = "3d8b1c0e-9f7a-4b2d-8e61-5c4a2b1f0e9d";
/// The subject object that carries the Shrinkwrap stack; every other corpus input is a target.
const SUBJECT: &str = "source";

#[test]
fn shrinkwrap_matches_blender_522_differential_corpus() -> TestResult {
    let corpus: Value = serde_json::from_str(CORPUS)?;
    assert_eq!(corpus["blender_version"], BLENDER_VERSION);
    let absolute = corpus["comparison"]["absolute_tolerance"]
        .as_f64()
        .ok_or("missing absolute tolerance")?;
    let relative = corpus["comparison"]["relative_tolerance"]
        .as_f64()
        .ok_or("missing relative tolerance")?;
    let defaults = corpus["defaults"].as_object().ok_or("missing defaults")?;
    let subject = Id::new(SUBJECT)?;

    // The subject without a stack must reproduce Blender's evaluated source mesh.
    let (doc, snapshot) = evaluate(&corpus["inputs"], Vec::new())?;
    let source = output_summary(&doc, &snapshot, &subject)?;
    assert_potter_match(
        "source mesh",
        &corpus["source_blender_input"],
        &source,
        absolute,
        relative,
    );

    for case in corpus["cases"]
        .as_array()
        .ok_or("missing Shrinkwrap cases")?
    {
        let id = case["id"].as_str().ok_or("case id is not a string")?;
        let parameters = case["params"]
            .as_object()
            .ok_or("case params are missing")?;
        assert_eq!(
            parameters.len(),
            defaults.len(),
            "{id}: fixture parameters must be fully specified"
        );
        let (doc, snapshot) = evaluate(&corpus["inputs"], vec![shrinkwrap_modifier(parameters)?])?;
        let actual = output_summary(&doc, &snapshot, &subject)?;
        assert_potter_match(id, &case["blender"], &actual, absolute, relative);
    }
    Ok(())
}

fn shrinkwrap_modifier(params: &Map<String, Value>) -> TestResult<Modifier> {
    Ok(Modifier {
        id: Id::new("shrinkwrap")?,
        modifier_type: "shrinkwrap".to_owned(),
        name: "Shrinkwrap".to_owned(),
        enabled: true,
        show_viewport: true,
        show_render: true,
        params: params.clone(),
        binding_data: None,
        runtime: ModifierRuntime::default(),
    })
}

/// A scene with every corpus input as a mesh object; the subject carries `stack`.
fn evaluate(inputs: &Value, stack: Vec<Modifier>) -> TestResult<(SceneDoc, Snapshot)> {
    let mut doc = SceneDoc::new(SCENE_ID.to_owned());
    let root = Id::new("collection_root")?;
    let inputs = inputs
        .as_object()
        .ok_or("corpus inputs must be an object")?;
    let mut stack = Some(stack);
    for (name, raw) in inputs {
        let is_subject = name.as_str() == SUBJECT;
        let node_id = Id::new(name)?;
        let data_id = Id::new(format!("data_{name}"))?;
        let mesh = corpus_mesh(raw, is_subject)?;
        let vertex_groups = if is_subject {
            corpus_vertex_groups(raw)?
        } else {
            Vec::new()
        };
        let modifiers = if is_subject {
            stack.take().unwrap_or_default()
        } else {
            Vec::new()
        };
        doc.data_blocks.insert(
            data_id.clone(),
            DataBlock {
                data_type: "mesh".to_owned(),
                mesh: Some(mesh),
                vertex_groups,
                ..DataBlock::default()
            },
        );
        doc.nodes.insert(
            node_id.clone(),
            Node {
                name: name.clone(),
                kind: "mesh".to_owned(),
                transform: corpus_transform(&raw["transform"])?,
                modifiers,
                data: Some(data_id),
                ..Node::default()
            },
        );
        doc.collections
            .get_mut(&root)
            .ok_or("root collection")?
            .objects
            .push(node_id);
    }
    doc.validate()?;
    let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
    Ok((doc, snapshot))
}

/// Blender's XYZ Euler rotation is `Rz * Ry * Rx`, glam's ZYX intrinsic order.
fn corpus_transform(raw: &Value) -> TestResult<Transform> {
    let degrees = vector(raw, "rotation_euler_deg")?;
    let radians = degrees.map(f64::to_radians);
    let rotation = DQuat::from_euler(EulerRot::ZYX, radians[2], radians[1], radians[0]).to_array();
    Ok(Transform {
        translation: vector(raw, "translation")?,
        rotation,
        scale: vector(raw, "scale")?,
        ..Transform::default()
    })
}

fn vector(raw: &Value, key: &str) -> TestResult<[f64; 3]> {
    let values = raw
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("transform {key} is missing"))?;
    let [x, y, z] = values.as_slice() else {
        return Err(format!("transform {key} needs three components").into());
    };
    Ok([
        x.as_f64().ok_or(key)?,
        y.as_f64().ok_or(key)?,
        z.as_f64().ok_or(key)?,
    ])
}

/// Blender stores positions, colors and weights as float32; the corpus decimals round the same way.
fn single(value: f64) -> f64 {
    f64::from(value as f32)
}

fn point(value: &Value) -> TestResult<DVec3> {
    let [x, y, z] = vector_values(value)?;
    Ok(DVec3::new(single(x), single(y), single(z)))
}

fn vector_values(value: &Value) -> TestResult<[f64; 3]> {
    let [x, y, z] = value
        .as_array()
        .ok_or("vector must be an array")?
        .as_slice()
    else {
        return Err("vector needs three components".into());
    };
    Ok([
        x.as_f64().ok_or("vector component")?,
        y.as_f64().ok_or("vector component")?,
        z.as_f64().ok_or("vector component")?,
    ])
}

fn indices(value: &Value) -> TestResult<Vec<usize>> {
    value
        .as_array()
        .ok_or("indices must be an array")?
        .iter()
        .map(|index| {
            index
                .as_u64()
                .and_then(|index| usize::try_from(index).ok())
                .ok_or_else(|| "index must be a non-negative integer".into())
        })
        .collect()
}

/// One corpus mesh: geometry, sharp edges, shading and, for the subject, its UV, color and group data.
fn corpus_mesh(raw: &Value, subject: bool) -> TestResult<Mesh> {
    let positions = raw["vertices"]
        .as_array()
        .ok_or("vertices")?
        .iter()
        .map(point)
        .collect::<TestResult<Vec<_>>>()?;
    let faces = raw["faces"]
        .as_array()
        .ok_or("faces")?
        .iter()
        .map(indices)
        .collect::<TestResult<Vec<_>>>()?;
    let mut mesh = Mesh::from_positions_and_faces(positions, faces)?;
    set_sharp_edges(&mut mesh, raw)?;
    set_shading(&mut mesh, raw)?;
    if subject {
        set_uv_layers(&mut mesh, raw)?;
        set_color_layers(&mut mesh, raw)?;
        set_vertex_group(&mut mesh, raw)?;
    }
    Ok(mesh)
}

fn set_sharp_edges(mesh: &mut Mesh, raw: &Value) -> TestResult {
    let Some(pairs) = raw.get("sharp_edges").and_then(Value::as_array) else {
        return Ok(());
    };
    let mut sharp = Vec::new();
    for pair in pairs {
        let [a, b] = indices(pair)?[..] else {
            return Err("sharp edge needs two vertices".into());
        };
        sharp.push((a.min(b), a.max(b)));
    }
    let values: Map<String, Value> = mesh
        .edges
        .iter()
        .map(|edge| {
            let [a, b] = edge.vertices.map(|vertex| vertex as usize);
            let is_sharp = sharp.contains(&(a.min(b), a.max(b)));
            (format!("e{}", edge.id), json!(is_sharp))
        })
        .collect();
    mesh.attributes.insert(
        "sharp".to_owned(),
        json!({"domain":"edge","type":"bool","values":values}),
    );
    Ok(())
}

/// `use_smooth` on every face keeps Blender's `sharp_face` attribute, which `shade_smooth()` removes.
fn set_shading(mesh: &mut Mesh, raw: &Value) -> TestResult {
    let attributes = raw.get("attributes");
    let operator = attributes
        .and_then(|attributes| attributes.get("shade_smooth_operator"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let list = attributes
        .and_then(|attributes| attributes.get("shade_smooth"))
        .and_then(Value::as_array);
    let per_face: Vec<bool> = match (operator, list) {
        (true, _) => vec![true; mesh.faces.len()],
        (false, Some(list)) => list
            .iter()
            .map(|flag| flag.as_bool().ok_or("shade_smooth must be bool"))
            .collect::<Result<_, _>>()?,
        (false, None) => return Ok(()),
    };
    let values: Map<String, Value> = mesh
        .faces
        .iter()
        .zip(&per_face)
        .map(|(face, flag)| (format!("f{}", face.id), json!(flag)))
        .collect();
    let mut shading = json!({"domain":"face","type":"bool","values":values});
    if list.is_some() {
        shading[SHARP_FACE_PRESENT] = json!(true);
    }
    mesh.attributes.insert("shade_smooth".to_owned(), shading);
    Ok(())
}

fn set_uv_layers(mesh: &mut Mesh, raw: &Value) -> TestResult {
    let Some(entries) = raw
        .get("attributes")
        .and_then(|attributes| attributes.get("uv_map"))
        .and_then(Value::as_array)
    else {
        return Ok(());
    };
    let mut layers: Vec<UvLayer> = Vec::new();
    for entry in entries {
        let name = entry
            .get("layer")
            .and_then(Value::as_str)
            .unwrap_or("UVMap");
        let face = usize::try_from(entry["face_id"].as_u64().ok_or("uv face_id")?)?;
        let corners = entry["uv"]
            .as_array()
            .ok_or("uv corners")?
            .iter()
            .map(|pair| {
                let [u, v] = vector_values_2(pair)?;
                // Potter stores Blender's UV v flipped, as its importer does.
                Ok([u, 1.0 - v])
            })
            .collect::<TestResult<Vec<_>>>()?;
        let index = match layers.iter().position(|candidate| candidate.name == name) {
            Some(index) => index,
            None => {
                layers.push(UvLayer::new(name, vec![None; mesh.faces.len()]));
                layers.len() - 1
            }
        };
        let slot = layers[index]
            .faces
            .get_mut(face)
            .ok_or("uv face is out of range")?;
        *slot = Some(corners);
    }
    uv_layers::write(
        mesh,
        &UvStore {
            layers,
            active: None,
            render: None,
        },
    );
    Ok(())
}

fn vector_values_2(value: &Value) -> TestResult<[f64; 2]> {
    let [u, v] = value.as_array().ok_or("uv must be an array")?.as_slice() else {
        return Err("uv needs two components".into());
    };
    Ok([u.as_f64().ok_or("uv")?, v.as_f64().ok_or("uv")?])
}

fn set_color_layers(mesh: &mut Mesh, raw: &Value) -> TestResult {
    let Some(attributes) = raw
        .get("attributes")
        .and_then(|attributes| attributes.get("blender_attributes"))
        .and_then(Value::as_array)
    else {
        return Ok(());
    };
    let mut layers = Vec::new();
    let mut data_types = Map::new();
    for attribute in attributes {
        let name = attribute["name"].as_str().ok_or("color name")?;
        let domain = attribute["domain"].as_str().ok_or("color domain")?;
        if domain != "CORNER" {
            return Err(format!("corpus color domain {domain} is not supported").into());
        }
        let values = attribute["values"]
            .as_array()
            .ok_or("color values")?
            .iter()
            .map(|color| {
                let [r, g, b, a] = color.as_array().ok_or("color")?.as_slice() else {
                    return Err("color needs four components".into());
                };
                Ok([
                    single(r.as_f64().ok_or("color")?),
                    single(g.as_f64().ok_or("color")?),
                    single(b.as_f64().ok_or("color")?),
                    single(a.as_f64().ok_or("color")?),
                ])
            })
            .collect::<TestResult<Vec<_>>>()?;
        data_types.insert(name.to_owned(), attribute["data_type"].clone());
        layers.push(ColorLayer {
            name: name.to_owned(),
            domain: ColorDomain::Corner,
            values,
        });
    }
    color_layers::write(
        mesh,
        &ColorStore {
            layers,
            active: None,
            render: None,
        },
    );
    mesh.attributes.insert(
        "blender_metadata".to_owned(),
        json!({"color_data_types": data_types}),
    );
    Ok(())
}

fn set_vertex_group(mesh: &mut Mesh, raw: &Value) -> TestResult {
    let Some(group) = raw.get("vertex_group") else {
        return Ok(());
    };
    let name = group["name"].as_str().ok_or("vertex group name")?;
    let weights = group["weights"].as_array().ok_or("vertex group weights")?;
    let values: Map<String, Value> = mesh
        .vertices
        .iter()
        .zip(weights)
        .filter_map(|(vertex, amount)| {
            amount
                .as_f64()
                .filter(|amount| *amount != 0.0)
                .map(|amount| (format!("v{}", vertex.id), json!(amount)))
        })
        .collect();
    let mut groups = Map::new();
    groups.insert(name.to_owned(), Value::Object(values));
    mesh.attributes
        .insert("vertex_groups".to_owned(), Value::Object(groups));
    Ok(())
}

fn corpus_vertex_groups(raw: &Value) -> TestResult<Vec<VertexGroup>> {
    let Some(group) = raw.get("vertex_group") else {
        return Ok(Vec::new());
    };
    let name = group["name"].as_str().ok_or("vertex group name")?;
    Ok(vec![VertexGroup {
        id: Id::new(format!("group_{}", name.to_ascii_lowercase()))?,
        name: name.to_owned(),
    }])
}

/// The evaluated subject's summary in the corpus's fields. Its vertex, edge and face IDs must match the
/// subject's input IDs, since Shrinkwrap only moves vertices.
fn output_summary(doc: &SceneDoc, snapshot: &Snapshot, subject: &Id) -> TestResult<Value> {
    let data_id = doc
        .nodes
        .get(subject)
        .and_then(|node| node.data.as_ref())
        .ok_or("subject has no data")?;
    let data = doc
        .data_blocks
        .get(data_id)
        .ok_or("subject data is missing")?;
    let input = data.mesh.as_ref().ok_or("subject data has no mesh")?;
    let output = snapshot
        .meshes
        .get(subject)
        .ok_or("evaluated subject mesh is missing")?;
    let output_weights = mesh_vertex_weights(output, &data.vertex_groups);
    let input_ids = mesh_summary(input, &data.vertex_groups, &data.vertex_weights)?;
    let actual = mesh_summary(output, &data.vertex_groups, &output_weights)?;
    for field in ["vertex_ids", "edge_ids", "face_ids"] {
        assert_eq!(input_ids[field], actual[field], "{field} IDs changed");
    }
    Ok(actual)
}

fn mesh_summary(
    mesh: &Mesh,
    groups: &[VertexGroup],
    weights: &BTreeMap<u32, BTreeMap<Id, f64>>,
) -> Result<Value, Box<dyn Error>> {
    let vertex_index: HashMap<u32, usize> = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let vertex_ids: Vec<_> = mesh.vertices.iter().map(|vertex| vertex.id).collect();
    let edge_ids: Vec<_> = mesh
        .edges
        .iter()
        .map(|edge| json!({"id":edge.id,"vertices":edge.vertices}))
        .collect();
    let face_ids: Vec<_> = mesh
        .faces
        .iter()
        .map(|face| json!({"id":face.id,"vertices":&face.vertices}))
        .collect();
    let positions: Vec<_> = mesh
        .vertices
        .iter()
        .map(|vertex| json!(vertex.co.to_array()))
        .collect();
    let mut edges: Vec<Vec<usize>> = mesh
        .edges
        .iter()
        .map(|edge| {
            let mut indices = [
                vertex_index[&edge.vertices[0]],
                vertex_index[&edge.vertices[1]],
            ];
            indices.sort_unstable();
            indices.to_vec()
        })
        .collect();
    edges.sort_unstable();
    let faces: Vec<Vec<usize>> = mesh
        .faces
        .iter()
        .map(|face| face.vertices.iter().map(|id| vertex_index[id]).collect())
        .collect();
    let corner_normals: Vec<_> = corner_normal_vectors(mesh)?
        .into_iter()
        .map(|normal| json!(normal.to_array()))
        .collect();
    let uv_layers: Vec<Value> = uv_layers::read(mesh)
        .layers
        .into_iter()
        .map(|layer| {
            let values = mesh
                .faces
                .iter()
                .enumerate()
                .flat_map(|(index, face)| {
                    let face_values = layer.faces.get(index).and_then(Option::as_ref);
                    (0..face.vertices.len()).map(move |corner| {
                        let [u, v] = face_values
                            .and_then(|values| values.get(corner))
                            .copied()
                            .unwrap_or([0.0, 0.0]);
                        json!([u, 1.0 - v])
                    })
                })
                .collect::<Vec<_>>();
            json!({"name":layer.name,"values":values})
        })
        .collect();
    let color_data_types = mesh
        .attributes
        .get("blender_metadata")
        .and_then(|metadata| metadata.get("color_data_types"))
        .and_then(Value::as_object);
    let color_attributes: Vec<Value> = color_layers::read(mesh)
        .layers
        .into_iter()
        .map(|layer| {
            let data_type = color_data_types
                .and_then(|types| types.get(&layer.name))
                .and_then(Value::as_str)
                .ok_or("Blender color data type metadata is missing")?;
            let domain = match layer.domain {
                ColorDomain::Point => "POINT",
                ColorDomain::Corner => "CORNER",
            };
            Ok(json!({
                "name":layer.name,
                "domain":domain,
                "data_type":data_type,
                "values":layer.values
            }))
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let vertex_groups: Vec<Value> = groups
        .iter()
        .map(|group| {
            let values: Vec<f64> = mesh
                .vertices
                .iter()
                .map(|vertex| {
                    weights
                        .get(&vertex.id)
                        .and_then(|groups| groups.get(&group.id))
                        .copied()
                        .unwrap_or(0.0)
                })
                .collect();
            json!({"name":group.name,"values":values})
        })
        .collect();
    Ok(json!({
        "positions":positions,
        "edges":edges,
        "faces":faces,
        "vertex_ids":vertex_ids,
        "edge_ids":edge_ids,
        "face_ids":face_ids,
        "corner_normals":corner_normals,
        "uv_layers":uv_layers,
        "color_attributes":color_attributes,
        "vertex_groups":vertex_groups
    }))
}

fn assert_potter_match(
    context: &str,
    expected: &Value,
    actual: &Value,
    absolute: f64,
    relative: f64,
) {
    assert_eq!(
        expected["edges"], actual["edges"],
        "{context}: Potter edges differ from Blender"
    );
    assert_eq!(
        expected["faces"], actual["faces"],
        "{context}: Potter faces differ from Blender"
    );
    for field in [
        "positions",
        "corner_normals",
        "uv_layers",
        "color_attributes",
        "vertex_groups",
    ] {
        assert_json_close(
            &format!("{context}.{field}"),
            &expected[field],
            &actual[field],
            absolute,
            relative,
        );
    }
}

fn assert_json_close(
    context: &str,
    expected: &Value,
    actual: &Value,
    absolute: f64,
    relative: f64,
) {
    match (expected, actual) {
        (Value::Number(expected), Value::Number(actual)) => {
            let Some(expected) = expected.as_f64() else {
                panic!("{context}: expected finite JSON number");
            };
            let Some(actual) = actual.as_f64() else {
                panic!("{context}: actual finite JSON number");
            };
            let tolerance = absolute + relative * expected.abs();
            assert!(
                (expected - actual).abs() <= tolerance,
                "{context}: {actual} differs from Blender {expected} by more than {tolerance}"
            );
        }
        (Value::Array(expected), Value::Array(actual)) => {
            assert_eq!(
                expected.len(),
                actual.len(),
                "{context}: array length differs"
            );
            for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
                assert_json_close(
                    &format!("{context}[{index}]"),
                    expected,
                    actual,
                    absolute,
                    relative,
                );
            }
        }
        (Value::Object(expected), Value::Object(actual)) => {
            assert_eq!(
                expected.keys().collect::<Vec<_>>(),
                actual.keys().collect::<Vec<_>>(),
                "{context}: object fields differ"
            );
            for (key, expected) in expected {
                assert_json_close(
                    &format!("{context}.{key}"),
                    expected,
                    &actual[key],
                    absolute,
                    relative,
                );
            }
        }
        _ => assert_eq!(expected, actual, "{context}: values differ"),
    }
}
