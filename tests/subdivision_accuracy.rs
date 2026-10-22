use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    error::Error,
};

use glam::DVec3;
use potter_core::{
    geom::{Mesh, modifiers::evaluate_modifiers},
    model::{Id, Modifier, ModifierRuntime},
};
use serde_json::{Map, Value, json};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

type UvLayerRecord = (String, BTreeMap<u64, (Vec<[f64; 2]>, Vec<bool>)>);
type EdgeRecords = BTreeMap<(usize, usize), Vec<(bool, bool, f64)>>;

// Blender's custom-normal codes vary by one beta step across platforms; see
// tests/common/blender_code_tolerance.rs.
#[path = "common/blender_code_tolerance.rs"]
mod blender_code_tolerance;
use blender_code_tolerance::code_within_beta_step;
#[path = "common/normal_codes.rs"]
mod normal_codes;
use normal_codes::is_auto_code;

const ORACLE: &str = include_str!("fixtures/pr16_subdivision/oracle.json");
/// One stored custom-normal code: Auto (`[0, 0]`, or the legacy `null`), otherwise an authored `[i64, i64]` pair.
/// Auto matches only Auto; an authored pair must match alpha exactly and beta within one step.
fn codes_within_beta_step(actual: &Value, expected: &Value) -> bool {
    if actual == expected {
        return true;
    }
    if is_auto_code(actual) || is_auto_code(expected) {
        return is_auto_code(actual) && is_auto_code(expected);
    }
    let pair = |value: &Value| -> Option<[i64; 2]> {
        let [first, second] = value.as_array()?.as_slice() else {
            return None;
        };
        Some([first.as_i64()?, second.as_i64()?])
    };
    match (pair(actual), pair(expected)) {
        (Some(a), Some(e)) => code_within_beta_step(a, e),
        _ => false,
    }
}

// Fixture values are rounded to seven decimals, a serialization error of at most 5e-8. These
// bounds cover f32/OpenSubdiv arithmetic only.
const POSITION_TOLERANCE: f64 = 3.0e-6;
const UV_TOLERANCE: f64 = 2.0e-6;
const CREASE_TOLERANCE: f64 = 1.0e-6;
// Non-manifold splits put several Blender vertices at one position; vertices closer than this
// are one position, and each position must have the same multiplicity on both sides.
const CLUSTER_TOLERANCE: f64 = 1.0e-4;
// Generic values are float32 interpolations of values near 300 (one ulp is 3e-5 there), so they
// are bounded relative to magnitude at eight float32 ulps; this check is new, not an existing bound.
const GENERIC_ATTRIBUTE_TOLERANCE: f64 = 1.0e-5;

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

fn rotated<T: Clone>(values: &[T], start: usize) -> Vec<T> {
    (0..values.len())
        .map(|index| values[(start + index) % values.len()].clone())
        .collect()
}

/// Rotates a cycle to its lexicographically smallest form. Returns the rotated cycle and the
/// rotation applied, so per-corner data can be rotated the same way.
fn canonical_cycle(keys: &[usize]) -> (Vec<usize>, usize) {
    let len = keys.len();
    let start = (0..len)
        .min_by(|first, second| {
            (0..len)
                .map(|offset| keys[(first + offset) % len])
                .cmp((0..len).map(|offset| keys[(second + offset) % len]))
        })
        .unwrap_or(0);
    (rotated(keys, start), start)
}

fn edge_key(first: usize, second: usize) -> (usize, usize) {
    if first <= second {
        (first, second)
    } else {
        (second, first)
    }
}

/// One output or oracle face, with vertices as indices into its shape's point list.
#[derive(Clone)]
struct FaceRecord {
    vertices: Vec<usize>,
    material_index: u32,
    smooth: bool,
    uv: BTreeMap<String, Vec<[f64; 2]>>,
    uv_pins: BTreeMap<String, Vec<bool>>,
    normal_codes: Option<Vec<Value>>,
}

struct EdgeRecord {
    vertices: [usize; 2],
    seam: bool,
    sharp: bool,
    crease: f64,
}
/// A mesh compared by position: only point coordinates and index relations are read.
struct Shape {
    points: Vec<DVec3>,
    faces: Vec<FaceRecord>,
    edges: Vec<EdgeRecord>,
    has_custom_normals: bool,
    vertex_groups: BTreeMap<String, Vec<f64>>,
}

fn oracle_shape(expected: &Value) -> TestResult<Shape> {
    let points = expected["vertices"]
        .as_array()
        .ok_or("oracle positions missing")?
        .iter()
        .map(point)
        .collect::<TestResult<Vec<_>>>()?;
    let faces = expected["faces"]
        .as_array()
        .ok_or("oracle faces missing")?
        .iter()
        .map(|face| -> TestResult<FaceRecord> {
            let mut uv_layers = BTreeMap::new();
            for (layer, corners) in face["uvs"].as_object().ok_or("oracle UV map missing")? {
                let values = corners
                    .as_array()
                    .ok_or("oracle UV corners missing")?
                    .iter()
                    .map(uv)
                    .collect::<TestResult<Vec<_>>>()?;
                uv_layers.insert(layer.clone(), values);
            }
            let pins = face["uv_pins"]
                .as_object()
                .ok_or("oracle UV pins missing")?
                .iter()
                .map(|(layer, corners)| -> TestResult<(String, Vec<bool>)> {
                    let pins = corners
                        .as_array()
                        .ok_or("oracle UV pin row missing")?
                        .iter()
                        .map(|value| value.as_bool().ok_or("oracle UV pin must be boolean"))
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok((layer.clone(), pins))
                })
                .collect::<TestResult<BTreeMap<_, _>>>()?;
            let normal_codes = match face.get("normal_codes") {
                Some(values) => Some(
                    values
                        .as_array()
                        .ok_or("oracle custom normal codes missing")?
                        .clone(),
                ),
                None => None,
            };
            Ok(FaceRecord {
                vertices: indices(&face["v"])?,
                material_index: u32::try_from(
                    face["material"]
                        .as_u64()
                        .ok_or("oracle material index missing")?,
                )?,
                smooth: face["smooth"]
                    .as_bool()
                    .ok_or("oracle smooth flag missing")?,
                uv: uv_layers,
                uv_pins: pins,
                normal_codes,
            })
        })
        .collect::<TestResult<Vec<_>>>()?;
    let edges = expected["edges"]
        .as_array()
        .ok_or("oracle edges missing")?
        .iter()
        .map(|edge| -> TestResult<EdgeRecord> {
            let pair = indices(&edge["v"])?;
            if pair.len() != 2 {
                return Err("oracle edge must reference two vertices".into());
            }
            Ok(EdgeRecord {
                vertices: [pair[0], pair[1]],
                seam: edge["seam"].as_bool().ok_or("oracle seam flag missing")?,
                sharp: edge["sharp"].as_bool().ok_or("oracle sharp flag missing")?,
                crease: number(&edge["crease"])?,
            })
        })
        .collect::<TestResult<Vec<_>>>()?;
    let vertex_groups = expected["vertex_groups"]
        .as_object()
        .ok_or("oracle vertex groups missing")?
        .iter()
        .map(|(name, values)| -> TestResult<(String, Vec<f64>)> {
            let values = values
                .as_array()
                .ok_or("oracle vertex-group weights missing")?
                .iter()
                .map(number)
                .collect::<TestResult<Vec<_>>>()?;
            if values.len() != points.len() {
                return Err(format!("oracle group {name} has a wrong vertex count").into());
            }
            Ok((name.clone(), values))
        })
        .collect::<TestResult<BTreeMap<_, _>>>()?;
    Ok(Shape {
        points,
        faces,
        edges,
        vertex_groups,
        has_custom_normals: expected["has_custom_normals"]
            .as_bool()
            .ok_or("oracle custom-normal flag missing")?,
    })
}

fn potter_shape(mesh: &Mesh) -> TestResult<Shape> {
    let vertex_index: BTreeMap<u32, usize> = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let vertex_groups = mesh
        .attributes
        .get("vertex_groups")
        .and_then(Value::as_object)
        .map(|groups| {
            groups
                .iter()
                .map(|(name, values)| -> TestResult<(String, Vec<f64>)> {
                    let values = values
                        .as_object()
                        .ok_or("output vertex group must be a map")?;
                    let weights = mesh
                        .vertices
                        .iter()
                        .map(|vertex| {
                            values
                                .get(&format!("v{}", vertex.id))
                                .map(number)
                                .transpose()
                                .map(|weight| weight.unwrap_or(0.0))
                        })
                        .collect::<TestResult<Vec<_>>>()?;
                    Ok((name.clone(), weights))
                })
                .collect::<TestResult<BTreeMap<_, _>>>()
        })
        .transpose()?
        .unwrap_or_default();
    let points = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    let uv_layers = mesh
        .attributes
        .get("uv_layers")
        .and_then(|store| store.get("layers"))
        .and_then(Value::as_array)
        .ok_or("output UV layers missing")?
        .iter()
        .map(|layer| -> TestResult<UvLayerRecord> {
            let name = layer["name"]
                .as_str()
                .ok_or("output UV layer name missing")?;
            let mut faces = BTreeMap::new();
            for face in layer["faces"].as_array().ok_or("output UV faces missing")? {
                let face_id = face["face_id"]
                    .as_u64()
                    .ok_or("output UV face ID missing")?;
                let values = face["uv"]
                    .as_array()
                    .ok_or("output UV corners missing")?
                    .iter()
                    .map(|value| -> TestResult<[f64; 2]> {
                        let [u, v] = uv(value)?;
                        Ok([u, 1.0 - v])
                    })
                    .collect::<TestResult<Vec<_>>>()?;
                let pins = face
                    .get("pinned")
                    .map(|pins| -> TestResult<Vec<bool>> {
                        let mut flags = Vec::new();
                        for value in pins.as_array().ok_or("output UV pin row missing")? {
                            flags.push(value.as_bool().ok_or("output UV pin must be boolean")?);
                        }
                        Ok(flags)
                    })
                    .transpose()?
                    .unwrap_or_else(|| vec![false; values.len()]);
                faces.insert(face_id, (values, pins));
            }
            Ok((name.to_owned(), faces))
        })
        .collect::<TestResult<Vec<_>>>()?;
    let smooth = mesh
        .attributes
        .get("shade_smooth")
        .and_then(|attribute| attribute.get("values"))
        .and_then(Value::as_object)
        .ok_or("output shade_smooth attribute missing")?;
    let custom = mesh
        .attributes
        .get("custom_normal")
        .and_then(|attribute| attribute.get("values"))
        .and_then(Value::as_object);
    let mut faces = Vec::with_capacity(mesh.faces.len());
    for face in &mesh.faces {
        let vertices = face
            .vertices
            .iter()
            .map(|id| {
                vertex_index
                    .get(id)
                    .copied()
                    .ok_or("output face references a missing vertex")
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut uv_map = BTreeMap::new();
        let mut uv_pins = BTreeMap::new();
        for (name, layer_faces) in &uv_layers {
            let (values, pins) = layer_faces
                .get(&u64::from(face.id))
                .ok_or_else(|| format!("UV layer {name} is missing output face {}", face.id))?;
            if values.len() != vertices.len() || pins.len() != vertices.len() {
                return Err(format!(
                    "UV layer {name} has a wrong corner or pin count for face {}",
                    face.id
                )
                .into());
            }
            uv_map.insert(name.clone(), values.clone());
            uv_pins.insert(name.clone(), pins.clone());
        }
        let normal_codes = match custom {
            Some(values) => Some(
                values
                    .get(&format!("f{}", face.id))
                    .and_then(Value::as_array)
                    .ok_or("custom normal codes missing for output face")?
                    .clone(),
            ),
            None => None,
        };
        faces.push(FaceRecord {
            vertices,
            material_index: face.material_index,
            smooth: smooth
                .get(&format!("f{}", face.id))
                .and_then(Value::as_bool)
                .ok_or("output shade_smooth value is missing")?,
            uv: uv_map,
            uv_pins,
            normal_codes,
        });
    }
    let sharp_values = mesh
        .attributes
        .get("sharp")
        .and_then(|attribute| attribute.get("values"))
        .and_then(Value::as_object);
    let seam_values = mesh
        .attributes
        .get("seam")
        .and_then(|attribute| attribute.get("values"))
        .and_then(Value::as_object);
    let crease_values = mesh
        .attributes
        .get("crease_edge")
        .and_then(|attribute| attribute.get("values"))
        .and_then(Value::as_object);
    let mut edges = Vec::with_capacity(mesh.edges.len());
    for edge in &mesh.edges {
        let mut endpoints = [0; 2];
        for (slot, id) in edge.vertices.iter().enumerate() {
            endpoints[slot] = *vertex_index
                .get(id)
                .ok_or("output edge references a missing vertex")?;
        }
        let key = format!("e{}", edge.id);
        let crease = crease_values
            .and_then(|values| values.get(&key))
            .map(number)
            .transpose()?
            .unwrap_or(0.0);
        edges.push(EdgeRecord {
            vertices: endpoints,
            seam: seam_values
                .and_then(|values| values.get(&key))
                .and_then(Value::as_bool)
                .unwrap_or(false),
            sharp: sharp_values
                .and_then(|values| values.get(&key))
                .and_then(Value::as_bool)
                .unwrap_or(false),
            crease,
        });
    }
    Ok(Shape {
        points,
        faces,
        edges,
        vertex_groups,
        has_custom_normals: custom.is_some(),
    })
}

fn input_mesh(spec: &Value, blender_input_edges: &Value) -> TestResult<Mesh> {
    let vertices = spec["vertices"]
        .as_array()
        .ok_or("mesh vertices missing")?
        .iter()
        .map(point)
        .collect::<TestResult<Vec<_>>>()?;
    let logical_faces: Vec<Vec<usize>> = spec["faces"]
        .as_array()
        .ok_or("mesh faces missing")?
        .iter()
        .map(indices)
        .collect::<TestResult<_>>()?;
    // A rotated-storage input stores face f's corners rotated by (f + 1) mod n and attaches
    // Blender's loop order as corner_order. Blender's result for the unrotated mesh is the reference.
    let rotated_storage = spec["rotated_storage"]
        .as_bool()
        .ok_or("mesh rotated_storage flag missing")?;
    let shifts: Vec<usize> = logical_faces
        .iter()
        .enumerate()
        .map(|(index, face)| {
            if rotated_storage {
                (index + 1) % face.len()
            } else {
                0
            }
        })
        .collect();
    let faces: Vec<Vec<usize>> = logical_faces
        .iter()
        .zip(&shifts)
        .map(|(face, &shift)| rotated(face, shift))
        .collect();
    let mut mesh = Mesh::from_positions_and_faces(vertices, faces)?;
    let vertex_ids: Vec<u32> = mesh.vertices.iter().map(|vertex| vertex.id).collect();
    let vertex_index: HashMap<u32, usize> = vertex_ids
        .iter()
        .enumerate()
        .map(|(index, &id)| (id, index))
        .collect();
    for loose in spec["loose_edges"]
        .as_array()
        .ok_or("mesh loose edges missing")?
    {
        let pair = indices(loose)?;
        if pair.len() != 2 {
            return Err("loose edge must reference two vertices".into());
        }
        mesh.insert_edge([vertex_ids[pair[0]], vertex_ids[pair[1]]])?;
    }
    let mut edges_by_pair = HashMap::new();
    for edge in std::mem::take(&mut mesh.edges) {
        let first = *vertex_index
            .get(&edge.vertices[0])
            .ok_or("edge vertex missing")?;
        let second = *vertex_index
            .get(&edge.vertices[1])
            .ok_or("edge vertex missing")?;
        edges_by_pair.insert(edge_key(first, second), edge);
    }
    mesh.edges = blender_input_edges
        .as_array()
        .ok_or("oracle input edges missing")?
        .iter()
        .map(|pair| -> TestResult<_> {
            let pair = indices(pair)?;
            let mut edge = edges_by_pair
                .remove(&edge_key(pair[0], pair[1]))
                .ok_or("Blender input edge missing from Potter input")?;
            edge.vertices = [vertex_ids[pair[0]], vertex_ids[pair[1]]];
            Ok(edge)
        })
        .collect::<TestResult<_>>()?;
    if !edges_by_pair.is_empty() {
        return Err("Potter input edge is absent from Blender's input edge order".into());
    }
    let material_values = spec["materials"]
        .as_array()
        .ok_or("mesh material indices missing")?;
    if material_values.len() != mesh.faces.len() {
        return Err("mesh material index count differs from face count".into());
    }
    for (face, material) in mesh.faces.iter_mut().zip(material_values) {
        face.material_index = u32::try_from(
            material
                .as_u64()
                .ok_or("mesh material index must be unsigned")?,
        )?;
    }
    if let Some(group_specs) = spec.get("vertex_groups").and_then(Value::as_object) {
        let mut groups = Map::new();
        for (name, weights) in group_specs {
            let weights = weights
                .as_array()
                .ok_or("mesh vertex-group weights missing")?;
            if weights.len() != mesh.vertices.len() {
                return Err(format!("mesh group {name} has a wrong vertex count").into());
            }
            let mut members = Map::new();
            for (vertex, weight) in mesh.vertices.iter().zip(weights) {
                let weight = number(weight)?;
                if weight > 0.0 {
                    members.insert(format!("v{}", vertex.id), json!(weight));
                }
            }
            groups.insert(name.clone(), Value::Object(members));
        }
        if !groups.is_empty() {
            mesh.attributes
                .insert("vertex_groups".to_owned(), Value::Object(groups));
        }
    }

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
    let sharp_edges = spec["sharp_edges"]
        .as_array()
        .ok_or("mesh sharp edges missing")?
        .iter()
        .map(indices)
        .map(|edge| {
            let edge = edge?;
            if edge.len() != 2 {
                return Err("sharp edge must reference two vertices".into());
            }
            Ok(edge_key(edge[0], edge[1]))
        })
        .collect::<TestResult<BTreeSet<_>>>()?;
    let mut sharp_values = Map::new();
    let mut seam_values = Map::new();
    let mut crease_values = Map::new();
    for edge in &mesh.edges {
        let endpoints = edge_key(edge.vertices[0] as usize, edge.vertices[1] as usize);
        let sharp = sharp_edges.contains(&endpoints);
        let seam = seams.contains(&endpoints);
        sharp_values.insert(format!("e{}", edge.id), json!(sharp));
        seam_values.insert(format!("e{}", edge.id), json!(seam));
        if let Some(value) = crease_pairs.get(&endpoints) {
            crease_values.insert(format!("e{}", edge.id), json!(value));
        }
    }
    let mut uv_layers = Vec::new();
    let mut active_uv = None;
    for layer in spec["uv_layers"]
        .as_array()
        .ok_or("input UV layers missing")?
    {
        let name = layer["name"]
            .as_str()
            .ok_or("input UV layer name missing")?;
        let values = layer["values"]
            .as_array()
            .ok_or("input UV values missing")?
            .iter()
            .map(|value| -> TestResult<[f64; 2]> {
                let [u, v] = uv(value)?;
                Ok([u, 1.0 - v])
            })
            .collect::<TestResult<Vec<_>>>()?;
        let pins = layer
            .get("pinned")
            .map(|values| {
                values
                    .as_array()
                    .ok_or("input UV pins missing")
                    .and_then(|values| {
                        values
                            .iter()
                            .map(|value| value.as_bool().ok_or("input UV pin must be boolean"))
                            .collect::<Result<Vec<_>, _>>()
                    })
            })
            .transpose()?;
        if pins.as_ref().is_some_and(|pins| pins.len() != values.len()) {
            return Err(format!("UV layer {name} has a wrong pin count").into());
        }
        let mut offset = 0;
        let mut layer_faces = Vec::with_capacity(mesh.faces.len());
        for (face, &shift) in mesh.faces.iter().zip(&shifts) {
            let end = offset + face.vertices.len();
            if end > values.len() {
                return Err(format!("UV layer {name} is shorter than input loops").into());
            }
            let mut entry = json!({"face_id": face.id, "uv": rotated(&values[offset..end], shift)});
            if let Some(pins) = &pins {
                entry["pinned"] = json!(rotated(&pins[offset..end], shift));
            }
            layer_faces.push(entry);
            offset = end;
        }
        if offset != values.len() {
            return Err(format!("UV layer {name} has extra input loops").into());
        }
        active_uv.get_or_insert(name);
        uv_layers.push(json!({"name": name, "faces": layer_faces}));
    }
    mesh.attributes.insert(
        "uv_layers".to_owned(),
        json!({"layers": uv_layers, "active": active_uv, "render": active_uv}),
    );
    let smooth_values: Map<String, Value> = mesh
        .faces
        .iter()
        .zip(
            spec["smooth"]
                .as_array()
                .ok_or("mesh smooth flags missing")?,
        )
        .map(|(face, smooth)| (format!("f{}", face.id), smooth.clone()))
        .collect();
    mesh.attributes.insert(
        "shade_smooth".to_owned(),
        json!({"domain": "face", "type": "bool", "values": smooth_values}),
    );
    mesh.attributes.insert(
        "sharp".to_owned(),
        json!({"domain": "edge", "type": "bool", "values": sharp_values}),
    );
    mesh.attributes.insert(
        "seam".to_owned(),
        json!({"domain": "edge", "type": "bool", "values": seam_values}),
    );
    mesh.attributes.insert(
        "crease_edge".to_owned(),
        json!({"domain": "edge", "type": "float", "values": crease_values}),
    );

    let source_codes = spec["custom_normal_codes"]
        .as_array()
        .ok_or("mesh custom normal codes missing")?;
    let mut corner = 0;
    let mut normal_values = Map::new();
    let mut corner_order = Map::new();
    for (face, &shift) in mesh.faces.iter().zip(&shifts) {
        let len = face.vertices.len();
        let end = corner + len;
        if end > source_codes.len() {
            return Err("source custom normal codes do not cover every corner".into());
        }
        normal_values.insert(
            format!("f{}", face.id),
            Value::Array(rotated(&source_codes[corner..end], shift)),
        );
        corner_order.insert(
            format!("f{}", face.id),
            Value::Array(
                (0..len)
                    .map(|k| json!(corner + (k + shift) % len))
                    .collect(),
            ),
        );
        corner = end;
    }
    if corner != source_codes.len() {
        return Err("source custom normal codes contain extra corners".into());
    }
    let mut layer = json!({"domain": "corner", "type": "normal_code", "values": normal_values});
    if rotated_storage {
        layer["corner_order"] = Value::Object(corner_order);
    }
    mesh.attributes.insert("custom_normal".to_owned(), layer);

    let vertex_creases = spec["vertex_creases"]
        .as_array()
        .ok_or("mesh vertex creases missing")?
        .iter()
        .map(|item| -> TestResult<(usize, f64)> {
            let values = item.as_array().ok_or("vertex crease must be an array")?;
            if values.len() != 2 {
                return Err("vertex crease must have a vertex and a value".into());
            }
            let vertex = usize::try_from(values[0].as_u64().ok_or("vertex crease index missing")?)?;
            Ok((vertex, number(&values[1])?))
        })
        .collect::<TestResult<BTreeMap<_, _>>>()?;
    if !vertex_creases.is_empty() {
        let mut values = Map::new();
        for (index, vertex) in mesh.vertices.iter().enumerate() {
            if let Some(&value) = vertex_creases.get(&index) {
                values.insert(format!("v{}", vertex.id), json!(value));
            }
        }
        mesh.attributes.insert(
            "crease_vert".to_owned(),
            json!({"domain": "point", "type": "float", "values": values}),
        );
    }
    // Generic attributes live in the native keyed store under their own names, keyed by element ID. Values
    // mirror `order_values` in generate.py: POINT, EDGE, and FACE index their element, and CORNER indexes
    // Blender's loop. Storage corner `c` of face `f` is logical corner `(c + shift) mod n`, and the logical
    // loops of face `f` start after the earlier faces' corners.
    let loop_offsets: Vec<usize> = logical_faces
        .iter()
        .scan(0, |total, face| {
            let start = *total;
            *total += face.len();
            Some(start)
        })
        .collect();
    for definition in spec["order_attributes"]
        .as_array()
        .ok_or("order attribute definitions missing")?
    {
        let name = definition["name"]
            .as_str()
            .ok_or("order attribute name missing")?;
        let domain = definition["domain"]
            .as_str()
            .ok_or("order attribute domain missing")?;
        let data_type = definition["type"]
            .as_str()
            .ok_or("order attribute type missing")?;
        let mut values = Map::new();
        match domain {
            "POINT" => {
                for (index, vertex) in mesh.vertices.iter().enumerate() {
                    values.insert(
                        format!("v{}", vertex.id),
                        order_value(domain, data_type, index, 0)?,
                    );
                }
            }
            "EDGE" => {
                for (index, edge) in mesh.edges.iter().enumerate() {
                    values.insert(
                        format!("e{}", edge.id),
                        order_value(domain, data_type, index, 0)?,
                    );
                }
            }
            "FACE" => {
                for (index, face) in mesh.faces.iter().enumerate() {
                    values.insert(
                        format!("f{}", face.id),
                        order_value(domain, data_type, index, 0)?,
                    );
                }
            }
            "CORNER" => {
                for (index, face) in mesh.faces.iter().enumerate() {
                    let size = face.vertices.len();
                    let row = (0..size)
                        .map(|corner| {
                            let logical = (corner + shifts[index]) % size;
                            order_value(domain, data_type, index, loop_offsets[index] + logical)
                        })
                        .collect::<TestResult<Vec<_>>>()?;
                    values.insert(format!("f{}", face.id), Value::Array(row));
                }
            }
            _ => return Err(format!("unknown order attribute domain {domain}").into()),
        }
        let native_domain = domain.to_ascii_lowercase();
        let native_type = match data_type {
            "FLOAT" => "float",
            "INT" => "int",
            "QUATERNION" => "quaternion",
            _ => return Err(format!("unknown order attribute type {data_type}").into()),
        };
        mesh.attributes.insert(
            name.to_owned(),
            json!({"domain": native_domain, "type": native_type, "values": values}),
        );
    }
    // BYTE_COLOR layers enter Potter as linear colours; their sRGB byte source is recorded in
    // `blender_metadata.color_data_types`, which `encode_byte_colors` re-blends from. The bytes come from
    // `byte_color_values` in generate.py, by Blender loop index for CORNER.
    let mut color_names = Vec::new();
    let mut color_types = Map::new();
    for definition in spec
        .get("byte_colors")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let name = definition["name"]
            .as_str()
            .ok_or("byte colour name missing")?;
        let domain = definition["domain"]
            .as_str()
            .ok_or("byte colour domain missing")?;
        let mut values = Map::new();
        match domain {
            "POINT" => {
                for (index, vertex) in mesh.vertices.iter().enumerate() {
                    values.insert(
                        format!("v{}", vertex.id),
                        linear_color(byte_color_bytes(domain, index)),
                    );
                }
            }
            "CORNER" => {
                for (index, face) in mesh.faces.iter().enumerate() {
                    let size = face.vertices.len();
                    let row: Vec<Value> = (0..size)
                        .map(|corner| {
                            let logical = loop_offsets[index] + (corner + shifts[index]) % size;
                            linear_color(byte_color_bytes(domain, logical))
                        })
                        .collect();
                    values.insert(format!("f{}", face.id), Value::Array(row));
                }
            }
            _ => return Err(format!("unknown byte colour domain {domain}").into()),
        }
        mesh.attributes.insert(
            name.to_owned(),
            json!({"domain": domain.to_ascii_lowercase(), "type": "color", "values": values}),
        );
        color_names.push(Value::String(name.to_owned()));
        color_types.insert(name.to_owned(), json!("BYTE_COLOR"));
    }
    if !color_names.is_empty() {
        mesh.attributes.insert(
            "color_layers".to_owned(),
            json!({"layers": color_names, "active": null, "render": null}),
        );
        mesh.attributes.insert(
            "blender_metadata".to_owned(),
            json!({"color_data_types": color_types}),
        );
    }
    Ok(mesh)
}

/// Blender's value of one order-attribute element, as `order_values` in generate.py computes it. `element`
/// is the POINT, EDGE, or FACE index; `loop_index` is the Blender loop of a CORNER element.
fn order_value(
    domain: &str,
    data_type: &str,
    element: usize,
    loop_index: usize,
) -> TestResult<Value> {
    Ok(match (domain, data_type) {
        ("POINT", "FLOAT") => json!(element as f64 + 0.25),
        ("POINT", "INT") => json!((element * 3) % 7),
        ("POINT", "QUATERNION") => json!(order_quaternion(element)),
        ("EDGE", "FLOAT") => json!(100.25 + element as f64),
        ("FACE", "FLOAT") => json!(200.25 + element as f64),
        ("CORNER", "FLOAT") => json!(300.25 + loop_index as f64),
        ("CORNER", "INT") => json!((loop_index * 5) % 13),
        ("CORNER", "QUATERNION") => json!(order_corner_quaternion(loop_index)),
        _ => return Err(format!("unsupported order attribute {data_type} on {domain}").into()),
    })
}

/// The unit quaternion `[x, y, z, w]` of loop `loop_index` for the CORNER quaternion layer: `corner_quaternion_for`
/// in generate.py, converted to Potter's native order. Neighbouring loops have a negative dot product.
fn order_corner_quaternion(loop_index: usize) -> [f64; 4] {
    let axis = [1.0, 0.5, 0.25];
    let length = axis
        .iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt();
    let angle = 0.2 + 0.05 * loop_index as f64;
    let sign = if loop_index % 2 == 1 { -1.0 } else { 1.0 };
    let scale = sign * (angle / 2.0).sin() / length;
    [
        axis[0] * scale,
        axis[1] * scale,
        axis[2] * scale,
        sign * (angle / 2.0).cos(),
    ]
}

/// The unit quaternion `[x, y, z, w]` of vertex `index` in the ordered-attribute fixtures: Potter's native order
/// for generic quaternions. It is `quaternion_for` in generate.py, which Blender stores as `w, x, y, z`. For some
/// vertices the angle exceeds pi, so w is negative.
fn order_quaternion(index: usize) -> [f64; 4] {
    let angle = 0.3 + 0.37 * index as f64;
    let axis = [
        1.0,
        0.5 * (index % 3) as f64,
        0.25 * (index % 5) as f64 + 0.1,
    ];
    let length = axis
        .iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt();
    let scale = (angle / 2.0).sin() / length;
    [
        axis[0] * scale,
        axis[1] * scale,
        axis[2] * scale,
        (angle / 2.0).cos(),
    ]
}

/// Groups vertices whose positions coincide. Returns each vertex's cluster, each cluster's
/// representative position, and each cluster's multiplicity.
fn cluster_positions(points: &[DVec3]) -> (Vec<usize>, Vec<DVec3>, Vec<usize>) {
    let cell = |point: DVec3| {
        (
            (point.x / CLUSTER_TOLERANCE).floor() as i64,
            (point.y / CLUSTER_TOLERANCE).floor() as i64,
            (point.z / CLUSTER_TOLERANCE).floor() as i64,
        )
    };
    let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    let mut representatives: Vec<DVec3> = Vec::new();
    let mut multiplicities: Vec<usize> = Vec::new();
    let mut assignment = Vec::with_capacity(points.len());
    for point in points {
        let (x, y, z) = cell(*point);
        let mut found = None;
        'search: for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(clusters) = grid.get(&(x + dx, y + dy, z + dz)) {
                        for &cluster in clusters {
                            if representatives[cluster].distance(*point) <= CLUSTER_TOLERANCE {
                                found = Some(cluster);
                                break 'search;
                            }
                        }
                    }
                }
            }
        }
        let cluster = if let Some(cluster) = found {
            multiplicities[cluster] += 1;
            cluster
        } else {
            representatives.push(*point);
            multiplicities.push(1);
            let cluster = representatives.len() - 1;
            grid.entry((x, y, z)).or_default().push(cluster);
            cluster
        };
        assignment.push(cluster);
    }
    (assignment, representatives, multiplicities)
}

#[derive(Default)]
struct Errors {
    position: f64,
    uv: f64,
    crease: f64,
}

/// Checks Blender's consumer-visible vertex/edge/face/loop storage order, not only topology after
/// position clustering. Geometry values retain their existing numeric tolerances.
fn compare_storage_order(actual: &Shape, expected: &Shape) -> Result<(), String> {
    if actual.points.len() != expected.points.len()
        || actual.edges.len() != expected.edges.len()
        || actual.faces.len() != expected.faces.len()
    {
        return Err(format!(
            "ordered counts differ: vertices {}/{}, edges {}/{}, faces {}/{}",
            actual.points.len(),
            expected.points.len(),
            actual.edges.len(),
            expected.edges.len(),
            actual.faces.len(),
            expected.faces.len()
        ));
    }
    for (index, (actual, expected)) in actual.points.iter().zip(&expected.points).enumerate() {
        let error = actual.distance(*expected);
        if error > POSITION_TOLERANCE {
            return Err(format!(
                "vertex {index} is out of Blender order or position: {actual:?} vs {expected:?}, error {error:.9e}"
            ));
        }
    }
    for (index, (actual, expected)) in actual.edges.iter().zip(&expected.edges).enumerate() {
        if actual.vertices != expected.vertices {
            return Err(format!(
                "edge {index} endpoints differ in Blender order: {:?} != {:?}",
                actual.vertices, expected.vertices
            ));
        }
        if actual.seam != expected.seam || actual.sharp != expected.sharp {
            return Err(format!("edge {index} seam/sharp flags differ"));
        }
        if (actual.crease - expected.crease).abs() > CREASE_TOLERANCE {
            return Err(format!("edge {index} crease differs"));
        }
    }
    for (index, (actual, expected)) in actual.faces.iter().zip(&expected.faces).enumerate() {
        if actual.vertices != expected.vertices {
            return Err(format!(
                "face {index} loop order differs: {:?} != {:?}",
                actual.vertices, expected.vertices
            ));
        }
        if actual.material_index != expected.material_index || actual.smooth != expected.smooth {
            return Err(format!("face {index} material/shade flag differs"));
        }
        if actual.uv.keys().collect::<Vec<_>>() != expected.uv.keys().collect::<Vec<_>>()
            || actual.uv_pins != expected.uv_pins
        {
            return Err(format!("face {index} UV layers or pin rows differ"));
        }
        for (layer, actual_uvs) in &actual.uv {
            let expected_uvs = &expected.uv[layer];
            if actual_uvs.len() != expected_uvs.len()
                || actual_uvs.iter().zip(expected_uvs).any(|(a, b)| {
                    (a[0] - b[0]).abs() > UV_TOLERANCE || (a[1] - b[1]).abs() > UV_TOLERANCE
                })
            {
                return Err(format!("face {index} UV loop order differs in {layer}"));
            }
        }
        match (&actual.normal_codes, &expected.normal_codes) {
            (Some(a), Some(b)) if a.len() == b.len() => {
                if !a.iter().zip(b).all(|(x, y)| codes_within_beta_step(x, y)) {
                    return Err(format!("face {index} custom-normal loop order differs"));
                }
            }
            (None, None) => {}
            _ => {
                return Err(format!(
                    "face {index} custom-normal layer/loop count differs"
                ));
            }
        }
    }
    if actual.vertex_groups.keys().collect::<Vec<_>>()
        != expected.vertex_groups.keys().collect::<Vec<_>>()
    {
        return Err("ordered vertex-group names differ".to_owned());
    }
    for (name, actual_values) in &actual.vertex_groups {
        let expected_values = &expected.vertex_groups[name];
        if actual_values.len() != expected_values.len()
            || actual_values
                .iter()
                .zip(expected_values)
                .any(|(a, b)| (a - b).abs() > 1.0e-6)
        {
            return Err(format!("vertex group {name} differs in vertex order"));
        }
    }
    Ok(())
}

/// Compares Blender's generic POINT/EDGE/FACE/CORNER values, in storage order, with the native keyed store of
/// `mesh`, componentwise for quaternion arrays. Provenance leaves an element without a source out of the store,
/// and readers take the type's default (`provenance.rs` remap docs). Blender's `subdiv_copy_edge_data`
/// (`subdiv_mesh.cc` L1199-1222) gives inner edges that default too, so an absent entry is accepted only where
/// Blender's value is 0.0. Any other absent element fails. Present values must match within
/// `GENERIC_ATTRIBUTE_TOLERANCE`, relative to magnitude, and quaternion signs must match exactly.
fn compare_ordered_attributes(mesh: &Mesh, expected: &Value) -> TestResult<()> {
    let expected = expected
        .as_array()
        .ok_or("oracle ordered attributes missing")?;
    for definition in expected {
        let name = definition["name"]
            .as_str()
            .ok_or("oracle ordered attribute name missing")?;
        let domain = definition["domain"]
            .as_str()
            .ok_or("oracle ordered attribute domain missing")?;
        let attribute = mesh
            .attributes
            .get(name)
            .ok_or_else(|| format!("generic attribute {name} is missing"))?;
        let native_domain = domain.to_ascii_lowercase();
        if attribute["domain"] != native_domain.as_str() {
            return Err(
                format!("generic attribute {name} is not on the {native_domain} domain").into(),
            );
        }
        let values = attribute["values"]
            .as_object()
            .ok_or_else(|| format!("generic attribute {name} has no values"))?;
        let actual: Vec<Option<&Value>> = match domain {
            "POINT" => mesh
                .vertices
                .iter()
                .map(|vertex| values.get(&format!("v{}", vertex.id)))
                .collect(),
            "EDGE" => mesh
                .edges
                .iter()
                .map(|edge| values.get(&format!("e{}", edge.id)))
                .collect(),
            "FACE" => mesh
                .faces
                .iter()
                .map(|face| values.get(&format!("f{}", face.id)))
                .collect(),
            "CORNER" => {
                let mut corners = Vec::new();
                for face in &mesh.faces {
                    let row = values
                        .get(&format!("f{}", face.id))
                        .and_then(Value::as_array)
                        .ok_or_else(|| format!("generic attribute {name} has no corner row"))?;
                    if row.len() != face.vertices.len() {
                        return Err(format!(
                            "generic attribute {name} corner row has the wrong length"
                        )
                        .into());
                    }
                    corners.extend(row.iter().map(Some));
                }
                corners
            }
            _ => return Err(format!("unknown order attribute domain {domain}").into()),
        };
        let expected_values = definition["values"]
            .as_array()
            .ok_or("oracle ordered attribute values missing")?;
        if actual.len() != expected_values.len() {
            return Err(format!(
                "generic attribute {name} has {} elements, Blender has {}",
                actual.len(),
                expected_values.len()
            )
            .into());
        }
        for (index, (actual, expected)) in actual.into_iter().zip(expected_values).enumerate() {
            match actual {
                Some(actual) if generic_close(actual, expected) => {}
                None if expected.as_f64() == Some(0.0) => {}
                _ => {
                    return Err(format!(
                        "generic attribute {name} {domain} element {index} is {actual:?}, Blender has {expected}"
                    )
                    .into());
                }
            }
        }
    }
    Ok(())
}

/// Whether a stored value matches Blender's: numbers within tolerance relative to magnitude, arrays
/// componentwise, and nothing else.
fn generic_close(actual: &Value, expected: &Value) -> bool {
    match (actual.as_array(), expected.as_array()) {
        (Some(actual), Some(expected)) => {
            actual.len() == expected.len()
                && actual
                    .iter()
                    .zip(expected)
                    .all(|(actual, expected)| generic_close(actual, expected))
        }
        (None, None) => match (actual.as_f64(), expected.as_f64()) {
            (Some(actual), Some(expected)) => {
                (actual - expected).abs() <= GENERIC_ATTRIBUTE_TOLERANCE * expected.abs().max(1.0)
            }
            _ => false,
        },
        _ => false,
    }
}

/// Linear value of an sRGB byte, as Blender's colour attribute `color` reports it.
fn srgb_to_linear(encoded: f64) -> f64 {
    if encoded <= 0.04045 {
        encoded / 12.92
    } else {
        ((encoded + 0.055) / 1.055).powf(2.4)
    }
}

/// sRGB-encoded value of a linear one, the inverse of [`srgb_to_linear`].
fn linear_to_srgb(linear: f64) -> f64 {
    if linear <= 0.003_130_8 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

/// The sRGB bytes `[r, g, b, a]` of a `BYTE_COLOR` element (`byte_color_values` in generate.py). `element` is the
/// POINT vertex index or the Blender loop index of a CORNER.
fn byte_color_bytes(domain: &str, element: usize) -> [u8; 4] {
    let (base, alpha_base) = if domain == "POINT" {
        (17, 101)
    } else {
        (53, 7)
    };
    [
        ((element * 37 + base) % 256) as u8,
        ((element * 37 + 91 + base) % 256) as u8,
        ((element * 37 + 182 + base) % 256) as u8,
        ((element * 29 + alpha_base) % 256) as u8,
    ]
}

/// Potter's stored linear colour of a `BYTE_COLOR` element: each colour channel decoded from its sRGB byte, alpha
/// linear.
fn linear_color(bytes: [u8; 4]) -> Value {
    let [r, g, b, a] = bytes.map(|byte| f64::from(byte) / 255.0);
    json!([srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), a])
}

/// Compares Blender's `BYTE_COLOR` values, in storage order, with Potter's output encoded back to sRGB bytes. The
/// bytes must match exactly, so the per-slot truncation of `encode_byte_colors` is what is checked.
fn compare_byte_colors(mesh: &Mesh, expected: &Value) -> TestResult<()> {
    for definition in expected.as_array().ok_or("oracle byte colours missing")? {
        let name = definition["name"]
            .as_str()
            .ok_or("oracle byte colour name missing")?;
        let domain = definition["domain"]
            .as_str()
            .ok_or("oracle byte colour domain missing")?;
        let attribute = mesh
            .attributes
            .get(name)
            .ok_or_else(|| format!("byte colour {name} is missing"))?;
        let values = attribute["values"]
            .as_object()
            .ok_or_else(|| format!("byte colour {name} has no values"))?;
        let mut actual: Vec<&Value> = Vec::new();
        match domain {
            "POINT" => {
                for vertex in &mesh.vertices {
                    let value = values.get(&format!("v{}", vertex.id)).ok_or_else(|| {
                        format!("byte colour {name} has no value for v{}", vertex.id)
                    })?;
                    actual.push(value);
                }
            }
            "CORNER" => {
                for face in &mesh.faces {
                    let row = values
                        .get(&format!("f{}", face.id))
                        .and_then(Value::as_array)
                        .ok_or_else(|| format!("byte colour {name} has no row for f{}", face.id))?;
                    actual.extend(row.iter());
                }
            }
            _ => return Err(format!("unknown byte colour domain {domain}").into()),
        }
        let expected_rows = definition["values"]
            .as_array()
            .ok_or("oracle byte colour values missing")?;
        if actual.len() != expected_rows.len() {
            return Err(format!(
                "byte colour {name} has {} elements, Blender has {}",
                actual.len(),
                expected_rows.len()
            )
            .into());
        }
        for (index, (actual, expected)) in actual.into_iter().zip(expected_rows).enumerate() {
            let linear = actual
                .as_array()
                .ok_or("byte colour value is not a colour")?
                .iter()
                .map(number)
                .collect::<TestResult<Vec<f64>>>()?;
            let bytes: Vec<i64> = linear
                .iter()
                .enumerate()
                .map(|(channel, value)| {
                    let encoded = if channel < 3 {
                        linear_to_srgb(*value)
                    } else {
                        *value
                    };
                    (encoded * 255.0).round() as i64
                })
                .collect();
            let wanted: Vec<i64> = expected
                .as_array()
                .ok_or("oracle byte colour is not a colour")?
                .iter()
                .map(|byte| byte.as_i64().ok_or("oracle byte is not an integer"))
                .collect::<Result<Vec<i64>, &str>>()?;
            if bytes != wanted {
                return Err(format!(
                    "byte colour {name} {domain} element {index} is {bytes:?}, Blender has {wanted:?}"
                )
                .into());
            }
        }
    }
    Ok(())
}

/// Compares two shapes by position. Faces match by their canonical position-cluster cycle, edges
/// by their endpoint clusters, so the comparison does not depend on output vertex order.
fn compare_shapes(actual: &Shape, expected: &Shape) -> Result<Errors, String> {
    if actual.points.len() != expected.points.len() {
        return Err(format!(
            "vertex count {} != {}",
            actual.points.len(),
            expected.points.len()
        ));
    }
    if actual.faces.len() != expected.faces.len() {
        return Err(format!(
            "face count {} != {}",
            actual.faces.len(),
            expected.faces.len()
        ));
    }
    if actual.edges.len() != expected.edges.len() {
        return Err(format!(
            "edge count {} != {}",
            actual.edges.len(),
            expected.edges.len()
        ));
    }
    if actual.has_custom_normals != expected.has_custom_normals {
        return Err("custom-normal layer presence differs".to_owned());
    }
    let (actual_clusters, actual_reps, actual_counts) = cluster_positions(&actual.points);
    let (expected_clusters, expected_reps, expected_counts) = cluster_positions(&expected.points);
    if actual_reps.len() != expected_reps.len() {
        return Err(format!(
            "distinct positions {} != {}",
            actual_reps.len(),
            expected_reps.len()
        ));
    }
    let mut used = vec![false; expected_reps.len()];
    let mut to_expected = vec![0; actual_reps.len()];
    for (cluster, representative) in actual_reps.iter().enumerate() {
        let (target, error) = expected_reps
            .iter()
            .enumerate()
            .filter(|(index, _)| !used[*index])
            .map(|(index, other)| (index, other.distance(*representative)))
            .min_by(|first, second| first.1.total_cmp(&second.1))
            .ok_or("no Blender position left to match")?;
        if error > POSITION_TOLERANCE {
            return Err(format!(
                "output position {representative:?} has no Blender position within {POSITION_TOLERANCE:.1e} (nearest {error:.9e})"
            ));
        }
        if actual_counts[cluster] != expected_counts[target] {
            return Err(format!(
                "position {representative:?} occurs {} times, Blender {} times",
                actual_counts[cluster], expected_counts[target]
            ));
        }
        used[target] = true;
        to_expected[cluster] = target;
    }
    let mut errors = Errors::default();
    for (index, point) in actual.points.iter().enumerate() {
        let target = expected_reps[to_expected[actual_clusters[index]]];
        errors.position = errors.position.max(point.distance(target));
    }
    if errors.position > POSITION_TOLERANCE {
        return Err(format!(
            "max position error {:.9e} exceeds {POSITION_TOLERANCE:.9e}",
            errors.position
        ));
    }
    if actual.vertex_groups.keys().collect::<Vec<_>>()
        != expected.vertex_groups.keys().collect::<Vec<_>>()
    {
        return Err("vertex-group sets differ".to_owned());
    }
    for (name, actual_weights) in &actual.vertex_groups {
        let expected_weights = expected
            .vertex_groups
            .get(name)
            .ok_or_else(|| format!("Blender vertex group {name} missing"))?;
        if actual_weights.len() != actual.points.len()
            || expected_weights.len() != expected.points.len()
        {
            return Err(format!(
                "vertex group {name} has a wrong output vertex count"
            ));
        }
        let mut actual_by_target = vec![Vec::new(); expected_reps.len()];
        for (vertex, weight) in actual_weights.iter().enumerate() {
            let target = to_expected[actual_clusters[vertex]];
            actual_by_target[target].push(*weight);
        }
        let mut expected_by_target = vec![Vec::new(); expected_reps.len()];
        for (vertex, weight) in expected_weights.iter().enumerate() {
            expected_by_target[expected_clusters[vertex]].push(*weight);
        }
        for (target, (actual_values, expected_values)) in actual_by_target
            .iter_mut()
            .zip(&mut expected_by_target)
            .enumerate()
        {
            actual_values.sort_by(f64::total_cmp);
            expected_values.sort_by(f64::total_cmp);
            if actual_values.len() != expected_values.len() {
                return Err(format!(
                    "vertex group {name} multiplicity differs at position {target}"
                ));
            }
            for (actual_weight, expected_weight) in actual_values.iter().zip(expected_values.iter())
            {
                if (actual_weight - expected_weight).abs() > 1.0e-6 {
                    return Err(format!(
                        "vertex group {name} weight {actual_weight:.9} != Blender {expected_weight:.9}"
                    ));
                }
            }
        }
    }

    let mut actual_faces: BTreeMap<Vec<usize>, Vec<(&FaceRecord, usize)>> = BTreeMap::new();
    for face in &actual.faces {
        let keys: Vec<usize> = face
            .vertices
            .iter()
            .map(|vertex| to_expected[actual_clusters[*vertex]])
            .collect();
        let (cycle, rotation) = canonical_cycle(&keys);
        actual_faces
            .entry(cycle)
            .or_default()
            .push((face, rotation));
    }
    let mut expected_faces: BTreeMap<Vec<usize>, Vec<(&FaceRecord, usize)>> = BTreeMap::new();
    for face in &expected.faces {
        let keys: Vec<usize> = face
            .vertices
            .iter()
            .map(|vertex| expected_clusters[*vertex])
            .collect();
        let (cycle, rotation) = canonical_cycle(&keys);
        expected_faces
            .entry(cycle)
            .or_default()
            .push((face, rotation));
    }
    if actual_faces.keys().collect::<Vec<_>>() != expected_faces.keys().collect::<Vec<_>>() {
        return Err("canonical oriented face topology differs".to_owned());
    }
    for (cycle, expected_list) in &expected_faces {
        let actual_list = &actual_faces[cycle];
        if actual_list.len() != expected_list.len() {
            return Err(format!(
                "face {cycle:?} occurs {} times, Blender {}",
                actual_list.len(),
                expected_list.len()
            ));
        }
        for ((actual_face, actual_rotation), (expected_face, expected_rotation)) in
            actual_list.iter().zip(expected_list)
        {
            // Both faces must start at the same corner, so their per-corner data lines up.
            if to_expected[actual_clusters[actual_face.vertices[0]]]
                != expected_clusters[expected_face.vertices[0]]
            {
                return Err(format!(
                    "face {cycle:?} starts at a different corner than Blender's face"
                ));
            }
            if actual_face.smooth != expected_face.smooth {
                return Err(format!("face {cycle:?} polygon interpolation flag differs"));
            }
            if actual_face.material_index != expected_face.material_index {
                return Err(format!(
                    "face {cycle:?} material index {} != Blender {}",
                    actual_face.material_index, expected_face.material_index
                ));
            }
            if actual_face.uv.keys().collect::<Vec<_>>()
                != expected_face.uv.keys().collect::<Vec<_>>()
            {
                return Err(format!("face {cycle:?} UV layer set differs"));
            }
            for (layer, expected_values) in &expected_face.uv {
                let actual_values = rotated(&actual_face.uv[layer], *actual_rotation);
                let expected_values = rotated(expected_values, *expected_rotation);
                for (actual_uv, expected_uv) in actual_values.iter().zip(&expected_values) {
                    errors.uv = errors
                        .uv
                        .max((actual_uv[0] - expected_uv[0]).abs())
                        .max((actual_uv[1] - expected_uv[1]).abs());
                }
            }
            if actual_face.uv_pins.keys().collect::<Vec<_>>()
                != expected_face.uv_pins.keys().collect::<Vec<_>>()
            {
                return Err(format!("face {cycle:?} UV pin layer set differs"));
            }
            for (layer, expected_pins) in &expected_face.uv_pins {
                let actual_pins = rotated(&actual_face.uv_pins[layer], *actual_rotation);
                let expected_pins = rotated(expected_pins, *expected_rotation);
                if actual_pins != expected_pins {
                    return Err(format!(
                        "face {cycle:?} UV pin flags differ in layer {layer}: {actual_pins:?} != {expected_pins:?}"
                    ));
                }
            }
            match (&actual_face.normal_codes, &expected_face.normal_codes) {
                (Some(actual_codes), Some(expected_codes)) => {
                    let actual_codes = rotated(actual_codes, *actual_rotation);
                    let expected_codes = rotated(expected_codes, *expected_rotation);
                    // Platform variance: see tests/common/blender_code_tolerance.rs.
                    let within_variance = actual_codes.len() == expected_codes.len()
                        && actual_codes
                            .iter()
                            .zip(&expected_codes)
                            .all(|(actual, expected)| codes_within_beta_step(actual, expected));
                    if !within_variance {
                        return Err(format!(
                            "face {cycle:?} custom-normal codes outside Blender's platform variance (alpha exact, beta ±1): {actual_codes:?} != {expected_codes:?}"
                        ));
                    }
                }
                (None, None) => {}
                _ => {
                    return Err(format!(
                        "face {cycle:?} custom-normal layer presence differs"
                    ));
                }
            }
        }
    }
    if errors.uv > UV_TOLERANCE {
        return Err(format!(
            "max corner UV error {:.9e} exceeds {UV_TOLERANCE:.9e}",
            errors.uv
        ));
    }

    let mut actual_edges: EdgeRecords = BTreeMap::new();
    for edge in &actual.edges {
        let first = to_expected[actual_clusters[edge.vertices[0]]];
        let second = to_expected[actual_clusters[edge.vertices[1]]];
        actual_edges
            .entry(edge_key(first, second))
            .or_default()
            .push((edge.seam, edge.sharp, edge.crease));
    }
    let mut expected_edges: EdgeRecords = BTreeMap::new();
    for edge in &expected.edges {
        expected_edges
            .entry(edge_key(
                expected_clusters[edge.vertices[0]],
                expected_clusters[edge.vertices[1]],
            ))
            .or_default()
            .push((edge.seam, edge.sharp, edge.crease));
    }
    if actual_edges.keys().collect::<Vec<_>>() != expected_edges.keys().collect::<Vec<_>>() {
        return Err("canonical edge topology differs".to_owned());
    }
    for (key, expected_list) in &expected_edges {
        let mut actual_list = actual_edges[key].clone();
        let mut expected_list = expected_list.clone();
        actual_list.sort_by(|first, second| {
            first
                .partial_cmp(second)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        expected_list.sort_by(|first, second| {
            first
                .partial_cmp(second)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if actual_list.len() != expected_list.len() {
            return Err(format!(
                "edge {key:?} occurs {} times, Blender {}",
                actual_list.len(),
                expected_list.len()
            ));
        }
        for (
            (actual_seam, actual_sharp, actual_crease),
            (expected_seam, expected_sharp, expected_crease),
        ) in actual_list.iter().zip(&expected_list)
        {
            if actual_seam != expected_seam {
                return Err(format!("seam flag differs on edge {key:?}"));
            }
            if actual_sharp != expected_sharp {
                return Err(format!("sharp flag differs on edge {key:?}"));
            }
            errors.crease = errors.crease.max((actual_crease - expected_crease).abs());
        }
    }
    if errors.crease > CREASE_TOLERANCE {
        return Err(format!(
            "max propagated crease error {:.9e} exceeds {CREASE_TOLERANCE:.9e}",
            errors.crease
        ));
    }
    Ok(errors)
}

fn compare_case(case: &Value, spec: &Value) -> TestResult<Errors> {
    let name = case["name"].as_str().ok_or("case name missing")?;
    let params: Map<String, Value> = serde_json::from_value(case["params"].clone())?;
    let mesh = input_mesh(spec, &case["expected"]["input_edges"])?;
    let input_vertex_index: HashMap<u32, usize> = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let input_edges: Vec<[usize; 2]> = mesh
        .edges
        .iter()
        .map(|edge| {
            Ok([
                *input_vertex_index
                    .get(&edge.vertices[0])
                    .ok_or("input edge's first vertex missing")?,
                *input_vertex_index
                    .get(&edge.vertices[1])
                    .ok_or("input edge's second vertex missing")?,
            ])
        })
        .collect::<TestResult<_>>()?;
    let oracle_input_edges: Vec<[usize; 2]> = case["expected"]["input_edges"]
        .as_array()
        .ok_or("oracle input edge order missing")?
        .iter()
        .map(indices)
        .map(|pair| -> TestResult<[usize; 2]> {
            let pair = pair?;
            Ok([pair[0], pair[1]])
        })
        .collect::<TestResult<_>>()?;
    if input_edges != oracle_input_edges {
        return Err(format!("{name}: input mesh edge storage order differs from Blender").into());
    }
    let modifier = Modifier {
        baked_ocean_cache: None,
        id: Id::new("subdivision_oracle")?,
        modifier_type: "subdivision".to_owned(),
        name: "Subdivision oracle".to_owned(),
        enabled: true,
        show_viewport: true,
        show_render: true,
        params,
        binding_data: None,
        runtime: ModifierRuntime::default(),
    };
    let actual_mesh = evaluate_modifiers(&mesh, &[modifier], name)?;
    compare_ordered_attributes(&actual_mesh, &case["expected"]["ordered_attributes"])?;
    compare_byte_colors(&actual_mesh, &case["expected"]["byte_colors"])?;
    let actual = potter_shape(&actual_mesh)?;
    let expected = oracle_shape(&case["expected"])?;
    compare_storage_order(&actual, &expected)
        .map_err(|message| format!("{name}: storage order: {message}"))?;
    compare_shapes(&actual, &expected).map_err(|message| format!("{name}: {message}").into())
}

#[test]
fn subdivision_surface_matches_blender_522_across_the_modifier_matrix() -> TestResult {
    let oracle: Value = serde_json::from_str(ORACLE)?;
    assert_eq!(oracle["blender_version"], "5.2.2 LTS");
    assert_eq!(oracle["round_digits"], 7);
    let meshes = oracle["meshes"]
        .as_object()
        .ok_or("oracle mesh inputs missing")?;
    let cases = oracle["cases"].as_array().ok_or("oracle cases missing")?;
    assert_matrix_coverage(cases, meshes)?;

    let mut failures = Vec::new();
    let mut worst = Errors::default();
    for case in cases {
        let case_name = case["name"].as_str().ok_or("case name missing")?;
        let mesh_name = case["mesh"].as_str().ok_or("case mesh missing")?;
        let spec = meshes
            .get(mesh_name)
            .ok_or("case mesh not present in oracle")?;
        let outcome =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| compare_case(case, spec)));
        match outcome {
            Ok(Ok(errors)) => {
                worst.position = worst.position.max(errors.position);
                worst.uv = worst.uv.max(errors.uv);
                worst.crease = worst.crease.max(errors.crease);
            }
            Ok(Err(error)) => failures.push(format!("{case_name}: {error}")),
            Err(panic) => {
                let message = panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|text| (*text).to_owned()))
                    .unwrap_or_default();
                failures.push(format!("{case_name}: {message}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} oracle cases failed:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
    eprintln!(
        "subdivision corpus: {} cases; worst position {:.3e}, UV {:.3e}, crease {:.3e}; custom-normal codes within alpha-exact, beta ±1",
        cases.len(),
        worst.position,
        worst.uv,
        worst.crease
    );
    Ok(())
}

/// Measures which settings, mesh classes, and crease inputs the corpus exercises. A missing
/// combination fails the test, so the corpus cannot silently narrow.
fn assert_matrix_coverage(cases: &[Value], meshes: &Map<String, Value>) -> TestResult {
    assert_eq!(
        cases.len(),
        186,
        "the complete Blender oracle corpus is required"
    );
    let mut levels = BTreeSet::new();
    let mut qualities = BTreeSet::new();
    let mut quad_level_quality_pairs = BTreeSet::new();
    let mut uv_modes = BTreeSet::new();
    let mut uv_mesh_pairs = BTreeSet::new();
    let mut boundary_modes = BTreeSet::new();
    let mut uv_boundary_pairs = BTreeSet::new();
    let mut limit_settings = BTreeSet::new();
    let mut subdivision_types = BTreeSet::new();
    let mut crease_settings = BTreeSet::new();
    let mut crease_dependent_uv_modes = BTreeSet::new();
    let mut custom_normal_settings = BTreeSet::new();
    let mut used_meshes = BTreeSet::new();
    let mut material_indices = BTreeSet::new();
    let mut crease_values = BTreeSet::new();
    let mut default_uv_levels = BTreeSet::new();
    let mut cube_above_quality = false;
    for case in cases {
        let params = &case["params"];
        let level = params["levels"].as_u64().ok_or("levels missing")?;
        let quality = params["quality"].as_u64().ok_or("quality missing")?;
        let uv = params["uv_smooth"]
            .as_str()
            .ok_or("uv_smooth missing")?
            .to_owned();
        let boundary = params["boundary_smooth"]
            .as_str()
            .ok_or("boundary rule missing")?
            .to_owned();
        let mesh = case["mesh"].as_str().ok_or("case mesh missing")?;
        let limit = params["use_limit_surface"]
            .as_bool()
            .ok_or("limit flag missing")?;
        if matches!(
            case["name"].as_str(),
            Some("quad_default_uv_l4" | "quad_default_uv_l5")
        ) {
            if mesh != "quad"
                || quality != 2
                || boundary != "ALL"
                || uv != "PRESERVE_BOUNDARIES"
                || !limit
            {
                return Err("default UV L4/L5 cases must use the Blender default settings".into());
            }
            default_uv_levels.insert(level);
        }
        if mesh == "quad" && limit && (1..=5).contains(&level) && (1..=6).contains(&quality) {
            quad_level_quality_pairs.insert((level, quality));
        }
        levels.insert(level);
        qualities.insert(quality);
        uv_boundary_pairs.insert((uv.clone(), boundary.clone()));
        if (uv == "PRESERVE_CORNERS_AND_JUNCTIONS"
            || uv == "PRESERVE_CORNERS_JUNCTIONS_AND_CONCAVE")
            && params["use_creases"].as_bool() == Some(true)
        {
            crease_dependent_uv_modes.insert(uv.clone());
        }
        if matches!(
            mesh,
            "cube" | "curved_grid" | "ngon" | "nonmanifold" | "mixed"
        ) {
            uv_mesh_pairs.insert((mesh.to_owned(), uv.clone()));
        }
        uv_modes.insert(uv);
        boundary_modes.insert(boundary);
        limit_settings.insert(limit);
        subdivision_types.insert(
            params["subdivision_type"]
                .as_str()
                .ok_or("type missing")?
                .to_owned(),
        );
        crease_settings.insert(
            params["use_creases"]
                .as_bool()
                .ok_or("crease flag missing")?,
        );
        custom_normal_settings.insert(
            params["use_custom_normals"]
                .as_bool()
                .ok_or("custom-normal flag missing")?,
        );
        cube_above_quality |= mesh == "cube" && limit && level > quality;
        used_meshes.insert(mesh.to_owned());
        let mesh_spec = meshes.get(mesh).ok_or("case mesh not present in oracle")?;
        for crease in mesh_spec["creases"]
            .as_array()
            .ok_or("crease inputs missing")?
        {
            crease_values.insert((number(&crease[2])? * 100.0).round() as i64);
        }
        for face in case["expected"]["faces"]
            .as_array()
            .ok_or("oracle expected faces missing")?
        {
            material_indices.insert(
                face["material"]
                    .as_u64()
                    .ok_or("oracle material index missing")?,
            );
        }
    }
    assert_eq!(levels, BTreeSet::from([0, 1, 2, 3, 4, 5]), "levels");
    assert_eq!(
        qualities,
        BTreeSet::from([1, 2, 3, 4, 5, 6, 10]),
        "qualities"
    );
    assert_eq!(
        default_uv_levels,
        BTreeSet::from([4, 5]),
        "default UV interpolation must cover levels 4 and 5"
    );
    let expected_quad_pairs: BTreeSet<(u64, u64)> = (1_u64..=5)
        .flat_map(|level| (1_u64..=6).map(move |quality| (level, quality)))
        .collect();
    assert_eq!(
        quad_level_quality_pairs, expected_quad_pairs,
        "every quality 1–6 must cover every quad output level 1–5"
    );
    let all_uv_modes = [
        "NONE",
        "PRESERVE_CORNERS",
        "PRESERVE_CORNERS_AND_JUNCTIONS",
        "PRESERVE_CORNERS_JUNCTIONS_AND_CONCAVE",
        "PRESERVE_BOUNDARIES",
        "SMOOTH_ALL",
    ];
    let mut expected_uv_mesh_pairs = BTreeSet::new();
    for mesh_name in ["cube", "curved_grid", "ngon", "nonmanifold", "mixed"] {
        for &mode in &all_uv_modes {
            expected_uv_mesh_pairs.insert((mesh_name.to_owned(), mode.to_owned()));
        }
    }
    assert_eq!(
        uv_mesh_pairs, expected_uv_mesh_pairs,
        "all six UV modes must cover extraordinary, grid, n-gon, non-manifold, and mixed topology"
    );
    assert_eq!(
        uv_modes,
        all_uv_modes.iter().map(|mode| (*mode).to_owned()).collect(),
        "uv_smooth modes"
    );
    assert_eq!(
        boundary_modes,
        BTreeSet::from(["ALL".to_owned(), "PRESERVE_CORNERS".to_owned()]),
        "boundary_smooth modes"
    );
    assert_eq!(
        uv_boundary_pairs.len(),
        all_uv_modes.len() * 2,
        "uv_smooth x boundary_smooth pairs"
    );
    assert_eq!(
        limit_settings,
        BTreeSet::from([false, true]),
        "use_limit_surface"
    );
    assert_eq!(
        subdivision_types,
        BTreeSet::from(["CATMULL_CLARK".to_owned(), "SIMPLE".to_owned()]),
        "subdivision types"
    );
    assert_eq!(
        crease_settings,
        BTreeSet::from([false, true]),
        "use_creases"
    );
    assert_eq!(
        custom_normal_settings,
        BTreeSet::from([false, true]),
        "use_custom_normals"
    );
    assert_eq!(
        crease_values,
        BTreeSet::from([10, 20, 25, 30, 50, 60, 75, 80, 90, 100]),
        "crease input values"
    );
    assert_eq!(
        material_indices,
        BTreeSet::from([0, 1, 2]),
        "multiple material indices must be exercised"
    );
    let pinned_case = cases
        .iter()
        .find(|case| case["name"] == "level_1_quality_1")
        .ok_or("pinned quad case missing")?;
    let input_ramp = meshes["quad"]["vertex_groups"]["Ramp"]
        .as_array()
        .ok_or("quad Ramp input weights missing")?;
    assert_eq!(input_ramp.len(), 4, "Ramp must vary over every quad corner");
    let distinct_input_weights: BTreeSet<i64> = input_ramp
        .iter()
        .map(|weight| Ok((number(weight)? * 1000.0).round() as i64))
        .collect::<TestResult<_>>()?;
    assert_eq!(
        distinct_input_weights.len(),
        4,
        "input vertex-group ramp must vary"
    );
    let output_ramp = pinned_case["expected"]["vertex_groups"]["Ramp"]
        .as_array()
        .ok_or("Blender Subsurf Ramp output missing")?;
    assert_eq!(
        output_ramp.len(),
        9,
        "level-one quad must carry nine group weights"
    );
    let distinct_weights: BTreeSet<i64> = output_ramp
        .iter()
        .map(|weight| Ok((number(weight)? * 1000.0).round() as i64))
        .collect::<TestResult<_>>()?;
    assert!(
        distinct_weights.len() >= 5,
        "Blender Subsurf must interpolate the non-uniform vertex-group ramp"
    );
    let expected = &pinned_case["expected"];
    let positions = expected["vertices"]
        .as_array()
        .ok_or("pinned quad output positions missing")?;
    let mut pin_count = 0;
    let mut pinned_pairs = BTreeSet::new();
    for face in expected["faces"]
        .as_array()
        .ok_or("pinned quad output faces missing")?
    {
        let vertices = indices(&face["v"])?;
        let uvs = face["uvs"]["UVMap"]
            .as_array()
            .ok_or("pinned quad UV output missing")?;
        let pins = face["uv_pins"]["UVMap"]
            .as_array()
            .ok_or("pinned quad pin output missing")?;
        for ((vertex, uv), pin) in vertices.iter().zip(uvs).zip(pins) {
            if pin.as_bool() == Some(true) {
                pin_count += 1;
                pinned_pairs.insert(serde_json::to_string(&(
                    positions[*vertex].clone(),
                    uv.clone(),
                ))?);
            }
        }
    }
    assert_eq!(
        pin_count, 9,
        "Blender Subsurf must propagate the pinned corner to nine loops"
    );
    assert_eq!(
        pinned_pairs.len(),
        4,
        "pinned output must preserve four unique position/UV pairs"
    );
    assert_eq!(
        crease_dependent_uv_modes,
        BTreeSet::from([
            "PRESERVE_CORNERS_AND_JUNCTIONS".to_owned(),
            "PRESERVE_CORNERS_JUNCTIONS_AND_CONCAVE".to_owned(),
        ]),
        "dependent UV corner-sharpness modes with creases"
    );
    assert!(
        cube_above_quality,
        "a limit-surface cube must run with levels above quality"
    );
    for mesh in [
        "quad",
        "cube",
        "cube_flat",
        "ngon",
        "nonmanifold",
        "mixed",
        "mixed_consistent",
        "dependent_uv_plus1",
        "dependent_uv_plus2",
        "loose",
        "curved_grid",
        "l_sharp_plane",
        "plane_crease",
        "plane_border_crease",
        "cube_crease_0_2",
        "quad_all_crease",
        "loose_corner",
        "uv_group",
        "cube_rotated_storage",
        "ngon_rotated_storage",
        "mixed_rotated_storage",
        "vertex_crease_grid",
        "vertex_crease_cube",
    ] {
        assert!(used_meshes.contains(mesh), "{mesh} must be in the corpus");
    }
    for mesh in [
        "cube_rotated_storage",
        "ngon_rotated_storage",
        "mixed_rotated_storage",
    ] {
        assert_eq!(
            meshes[mesh]["rotated_storage"],
            json!(true),
            "{mesh} must store rotated corners"
        );
    }
    let sharp_inputs = meshes["l_sharp_plane"]["sharp_edges"]
        .as_array()
        .ok_or("L-crease sharp edges missing")?;
    assert_eq!(
        sharp_inputs.len(),
        2,
        "the L-shaped crease needs two sharp edges"
    );
    for mesh in ["curved_grid", "cube", "ngon", "nonmanifold", "mixed"] {
        let seams = meshes[mesh]["seams"]
            .as_array()
            .ok_or("seam input missing")?;
        assert!(!seams.is_empty(), "{mesh} must carry seam edges");
    }
    let loose_edges = meshes["loose"]["loose_edges"]
        .as_array()
        .ok_or("loose edges missing")?;
    assert!(
        !loose_edges.is_empty(),
        "loose mesh must carry a loose edge"
    );
    let flat = meshes["cube_flat"]["smooth"]
        .as_array()
        .ok_or("smooth flags missing")?;
    assert!(
        flat.iter().any(|flag| flag.as_bool() == Some(false)),
        "flat faces must be in the corpus"
    );
    Ok(())
}
