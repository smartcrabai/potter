//! Canonical comparator for Blender-backed Boolean parity tests.
//!
//! `tests/blend_boolean_parity.rs`, `tests/blend_boolean_regressions.rs`, and
//! `tests/blend_boolean_float_regressions.rs` include this file via `#[path]`. Blender face
//! rings are in world space, rotated to a canonical vertex; face order is ignored. Potter faces
//! are matched by cyclic vertex connectivity, then each payload is compared on the matched face/corner.
//! Coordinates, UVs, and custom normals use absolute 1e-5 tolerance for Blender's f32 storage; vertex-group weights use 1e-6. Topology (counts, face sizes, edge
//! endpoints, and face connectivity), material, `fid`, and payload presence are exact. No coordinate
//! or payload is rounded into a lookup key. Signed volume uses relative tolerance 1e-6, or absolute
//! tolerance 1e-9 for expected zero.
//!
//! Per face: ring, material, `fid`, UVs/pins/named UV layers, FACE/CORNER attributes, per-corner point attributes,
//! and custom-normal vectors/codes. Edge points retain per-endpoint point payloads; global point and
//! edge attributes remain associated with canonical positions/endpoints and are matched as multisets.
//! Missing association fields in historical expectations are unrecorded. Color-layer order/domain/type
//! and active/render names are exact; RGBA values are linear.
//! Raw OBJECT assignments (aligned to slot count, `None` for DATA-linked slots), effective slot materials, raw DATA slots, and link modes are exact.
//! Vertex groups are matched by position and weight. Counts, topology and payload presence are exact;
//! numeric values use only the documented tolerances.

#![expect(
    clippy::ref_option,
    reason = "the comparator borrows optional canonical payloads as stored in the canonical form"
)]

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    error::Error,
    fs,
    path::Path,
    process::Command,
};

use glam::{DMat4, DVec3};
use potter_core::{
    eval::{EvaluationContext, Snapshot},
    geom::Mesh,
    model::{Id, MaterialLink, SceneDoc},
};
use serde_json::Value;

pub type TestResult<T = ()> = Result<T, Box<dyn Error>>;

/// Absolute tolerance for Blender f32 coordinates, UVs, and custom normals.
const FLOAT_TOLERANCE: f64 = 1.0e-5;
/// Absolute tolerance for vertex-group weights.
const WEIGHT_TOLERANCE: f64 = 1.0e-6;
const VOLUME_RELATIVE_TOLERANCE: f64 = 1.0e-6;
const VOLUME_ABSOLUTE_TOLERANCE: f64 = 1.0e-9;

/// One face in canonical form. The per-corner lists stay aligned with the ring.
#[derive(Clone, Debug)]
pub struct CanonicalFace {
    ring: Vec<[f64; 3]>,
    material: u64,
    fid: Option<i64>,
    uv: Option<Vec<[f64; 2]>>,
    uv_pins: Option<Vec<bool>>,
    uv_layers: Option<BTreeMap<String, UvLayerFace>>,
    point_attributes: Option<Vec<AttributeValues>>,
    custom_normals: Option<Vec<[f64; 3]>>,
    face_attributes: Option<BTreeMap<String, Value>>,
    corner_attributes: Option<BTreeMap<String, Vec<Value>>>,
    custom_normal_codes: Option<Vec<NormalCode>>,
}

/// A vertex-group entry: the world-space position and weight (`None` outside the group).
type GroupEntry = ([f64; 3], Option<f64>);

type AttributeValues = BTreeMap<String, Value>;

#[derive(Clone, Debug, Default)]
struct UvLayerFace {
    uv: Vec<[f64; 2]>,
    pins: Vec<bool>,
}

#[derive(Clone, Debug)]
struct PointAttribute {
    co: [f64; 3],
    values: AttributeValues,
}

#[derive(Clone, Debug)]
struct EdgeAttribute {
    edge: [[f64; 3]; 2],
    values: AttributeValues,
}

#[derive(Clone, Debug)]
struct EdgePointEndpoint {
    co: [f64; 3],
    values: Option<AttributeValues>,
}

#[derive(Clone, Debug)]
struct EdgePoint {
    edge: [[f64; 3]; 2],
    endpoints: [EdgePointEndpoint; 2],
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ColorLayer {
    name: String,
    domain: String,
    data_type: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ColorLayers {
    layers: Vec<ColorLayer>,
    active: Option<String>,
    render: Option<String>,
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct UvLayerMetadata {
    layers: Vec<String>,
    active: Option<String>,
    render: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NormalCode {
    Auto,
    Authored([i16; 2]),
    ExplicitZero,
}

#[derive(Debug)]
struct FacePayload {
    material: u64,
    fid: Option<i64>,
    uv: Option<Vec<[f64; 2]>>,
    uv_pins: Option<Vec<bool>>,
    uv_layers: Option<BTreeMap<String, UvLayerFace>>,
    point_attributes: Option<Vec<AttributeValues>>,
    custom_normals: Option<Vec<[f64; 3]>>,
    face_attributes: Option<AttributeValues>,
    corner_attributes: Option<BTreeMap<String, Vec<Value>>>,
    custom_normal_codes: Option<Vec<NormalCode>>,
}

/// One Boolean result in canonical form. `faces` is `None` when the expectation records counts,
/// face sizes, and volume only.
#[derive(Debug)]
pub struct Canonical {
    vertices: usize,
    edges: usize,
    face_count: usize,
    edge_topology: Option<Vec<[[f64; 3]; 2]>>,
    edge_points: Option<Vec<EdgePoint>>,
    face_sizes: BTreeMap<usize, usize>,
    faces: Option<Vec<CanonicalFace>>,
    vertex_groups: BTreeMap<String, Vec<GroupEntry>>,
    point_attributes: Option<Vec<PointAttribute>>,
    edge_attributes: Option<Vec<EdgeAttribute>>,
    color_layers: Option<ColorLayers>,
    uv_layers: Option<UvLayerMetadata>,
    object_materials: Option<Vec<Option<String>>>,
    effective_materials: Option<Vec<Option<String>>>,
    data_materials: Option<Vec<Option<String>>>,
    material_links: Option<Vec<MaterialLink>>,
    volume: f64,
}

fn point_matches(actual: [f64; 3], expected: [f64; 3]) -> bool {
    actual
        .into_iter()
        .zip(expected)
        .all(|(actual, expected)| (actual - expected).abs() <= FLOAT_TOLERANCE)
}

fn value_matches(actual: f64, expected: f64, tolerance: f64) -> bool {
    (actual - expected).abs() <= tolerance
}

/// Canonical face: the ring keeps its source order; matching considers every cyclic start.
fn canonical_face(ring: &[DVec3], payload: FacePayload) -> CanonicalFace {
    CanonicalFace {
        ring: ring.iter().map(glam::DVec3::to_array).collect(),
        material: payload.material,
        fid: payload.fid,
        uv: payload.uv,
        uv_pins: payload.uv_pins,
        uv_layers: payload.uv_layers,
        point_attributes: payload.point_attributes,
        custom_normals: payload.custom_normals,
        face_attributes: payload.face_attributes,
        corner_attributes: payload.corner_attributes,
        custom_normal_codes: payload.custom_normal_codes,
    }
}

fn same_edge(left: [[f64; 3]; 2], right: [[f64; 3]; 2]) -> bool {
    (point_matches(left[0], right[0]) && point_matches(left[1], right[1]))
        || (point_matches(left[0], right[1]) && point_matches(left[1], right[0]))
}

fn expected_edge_topology(faces: &[CanonicalFace]) -> Vec<[[f64; 3]; 2]> {
    let mut edges: Vec<[[f64; 3]; 2]> = Vec::new();
    for face in faces {
        for index in 0..face.ring.len() {
            let edge = [face.ring[index], face.ring[(index + 1) % face.ring.len()]];
            if !edges.iter().any(|existing| same_edge(*existing, edge)) {
                edges.push(edge);
            }
        }
    }
    edges
}

fn signed_volume(rings: &[Vec<DVec3>]) -> f64 {
    rings
        .iter()
        .map(|ring| {
            ring.split_first().map_or(0.0, |(first, rest)| {
                rest.windows(2)
                    .map(|pair| first.dot(pair[0].cross(pair[1])) / 6.0)
                    .sum::<f64>()
            })
        })
        .sum()
}

fn number(value: &Value) -> TestResult<f64> {
    value.as_f64().ok_or_else(|| "expected a number".into())
}

fn point_from_json(value: &Value) -> TestResult<DVec3> {
    let values = value
        .as_array()
        .filter(|values| values.len() == 3)
        .ok_or("a point must be an array of three numbers")?;
    Ok(DVec3::new(
        number(&values[0])?,
        number(&values[1])?,
        number(&values[2])?,
    ))
}

fn points_from_json(value: &Value) -> TestResult<Vec<DVec3>> {
    value
        .as_array()
        .ok_or("expected an array of points")?
        .iter()
        .map(point_from_json)
        .collect()
}

fn normal_code_from_json(value: &Value) -> TestResult<NormalCode> {
    if value.is_null() {
        return Ok(NormalCode::Auto);
    }
    if value.as_str() == Some("explicit_zero") {
        return Ok(NormalCode::ExplicitZero);
    }
    let pair = value
        .as_array()
        .filter(|pair| pair.len() == 2)
        .ok_or("a custom-normal code must be an i16 pair, null, or explicit_zero")?;
    let component = |value: &Value| {
        value
            .as_i64()
            .and_then(|value| i16::try_from(value).ok())
            .ok_or("a custom-normal code component must fit i16")
    };
    let pair = [component(&pair[0])?, component(&pair[1])?];
    Ok(if pair == [0, 0] {
        NormalCode::Auto
    } else {
        NormalCode::Authored(pair)
    })
}

fn normal_codes_from_json(value: &Value) -> TestResult<Vec<NormalCode>> {
    value
        .as_array()
        .ok_or("custom-normal codes must be an array")?
        .iter()
        .map(normal_code_from_json)
        .collect()
}

fn uv_pairs(value: &Value) -> TestResult<Vec<[f64; 2]>> {
    let mut pairs = Vec::new();
    for pair in value.as_array().ok_or("UV values must be an array")? {
        let values = pair
            .as_array()
            .filter(|values| values.len() == 2)
            .ok_or("a UV corner must be a pair")?;
        pairs.push([number(&values[0])?, number(&values[1])?]);
    }

    Ok(pairs)
}
fn bools_from_json(value: &Value) -> TestResult<Vec<bool>> {
    value
        .as_array()
        .ok_or("UV pins must be an array")?
        .iter()
        .map(|value| {
            value
                .as_bool()
                .ok_or_else(|| "UV pin must be a boolean".into())
        })
        .collect()
}

fn expected_uv_layers(value: &Value, ring_len: usize) -> TestResult<BTreeMap<String, UvLayerFace>> {
    let mut layers = BTreeMap::new();
    for (name, layer) in value
        .as_object()
        .ok_or("Blender UV layers must be an object")?
    {
        let uv = uv_pairs(layer.get("uv").ok_or("Blender UV layer has no UV values")?)?;
        let pins = bools_from_json(layer.get("pins").ok_or("Blender UV layer has no pins")?)?;
        if uv.len() != ring_len || pins.len() != ring_len {
            return Err(format!(
                "Blender UV layer `{name}` has {} UVs and {} pins for {ring_len} face corners",
                uv.len(),
                pins.len()
            )
            .into());
        }
        layers.insert(name.clone(), UvLayerFace { uv, pins });
    }
    Ok(layers)
}

fn count(value: &Value, name: &str) -> TestResult<usize> {
    let count = value
        .as_u64()
        .ok_or_else(|| format!("Blender {name} count is invalid"))?;
    Ok(usize::try_from(count)?)
}

fn attribute_default(attribute: &Value) -> Value {
    match attribute.get("type").and_then(Value::as_str) {
        Some("bool") => Value::Bool(false),
        Some("int") => Value::from(0),
        Some("float") => Value::from(0.0),
        Some("float2") => serde_json::json!([0.0, 0.0]),
        Some("float3" | "vector") => serde_json::json!([0.0, 0.0, 0.0]),
        Some("color") => serde_json::json!([0.0, 0.0, 0.0, 1.0]),
        _ => Value::Null,
    }
}

fn user_attribute_name(storage_name: &str) -> &str {
    storage_name.strip_prefix("user:").unwrap_or(storage_name)
}

fn color_attribute<'a>(mesh: &'a Mesh, name: &str) -> Option<&'a Value> {
    mesh.attributes
        .iter()
        .find_map(|(storage_name, attribute)| {
            (user_attribute_name(storage_name) == name
                && attribute.get("type").and_then(Value::as_str) == Some("color"))
            .then_some(attribute)
        })
}

/// Values for one Blender mesh domain, in vertex/edge/face/corner order.
fn attribute_rows(mesh: &Mesh, domain: &str) -> TestResult<Option<Vec<AttributeValues>>> {
    let native_domain = match domain {
        "POINT" => "point",
        "EDGE" => "edge",
        "FACE" => "face",
        "CORNER" => "corner",
        _ => return Err(format!("unknown Blender mesh domain `{domain}`").into()),
    };
    let corner_count: usize = mesh.faces.iter().map(|face| face.vertices.len()).sum();
    let count = match domain {
        "POINT" => mesh.vertices.len(),
        "EDGE" => mesh.edges.len(),
        "FACE" => mesh.faces.len(),
        "CORNER" => corner_count,
        _ => unreachable!("domain was checked above"),
    };
    let mut rows = vec![BTreeMap::new(); count];
    let mut names = HashSet::new();

    for (storage_name, attribute) in &mesh.attributes {
        // Blender-internal state (`.select_*`, `.hide_*`) is not a user attribute; the generator skips it too.
        if storage_name == "custom_normal"
            || user_attribute_name(storage_name).starts_with('.')
            || attribute.get("domain").and_then(Value::as_str) != Some(native_domain)
        {
            continue;
        }
        let name = user_attribute_name(storage_name);
        let values = attribute
            .get("values")
            .and_then(Value::as_object)
            .ok_or_else(|| format!("Potter `{name}` {native_domain} values are not an object"))?;
        let default = attribute_default(attribute);
        match domain {
            "POINT" => {
                for (index, vertex) in mesh.vertices.iter().enumerate() {
                    let key = format!("v{}", vertex.id);
                    rows[index].insert(
                        name.to_owned(),
                        values.get(&key).cloned().unwrap_or_else(|| default.clone()),
                    );
                }
            }
            "EDGE" => {
                for (index, edge) in mesh.edges.iter().enumerate() {
                    let key = format!("e{}", edge.id);
                    rows[index].insert(
                        name.to_owned(),
                        values.get(&key).cloned().unwrap_or_else(|| default.clone()),
                    );
                }
            }
            "FACE" => {
                for (index, face) in mesh.faces.iter().enumerate() {
                    let key = format!("f{}", face.id);
                    rows[index].insert(
                        name.to_owned(),
                        values.get(&key).cloned().unwrap_or_else(|| default.clone()),
                    );
                }
            }
            "CORNER" => {
                let mut offset = 0;
                for face in &mesh.faces {
                    let key = format!("f{}", face.id);
                    match values.get(&key) {
                        Some(row) => {
                            let row = row
                                .as_array()
                                .filter(|row| row.len() == face.vertices.len())
                                .ok_or_else(|| {
                                    format!("Potter `{name}` corner row does not match face {key}")
                                })?;
                            for value in row {
                                rows[offset].insert(name.to_owned(), value.clone());
                                offset += 1;
                            }
                        }
                        None => {
                            for _ in &face.vertices {
                                rows[offset].insert(name.to_owned(), default.clone());
                                offset += 1;
                            }
                        }
                    }
                }
            }
            _ => unreachable!("domain was checked above"),
        }
        names.insert(name.to_owned());
    }

    if let Some(entries) = mesh.attributes.get("blender_attributes") {
        for entry in entries
            .as_array()
            .ok_or("Potter blender_attributes is not an array")?
        {
            if entry.get("domain").and_then(Value::as_str) != Some(domain) {
                continue;
            }
            let name = entry
                .get("name")
                .and_then(Value::as_str)
                .ok_or("Potter Blender attribute name is missing")?;
            if name.starts_with('.') {
                continue;
            }
            if !names.insert(name.to_owned()) {
                continue;
            }
            let values = entry
                .get("values")
                .and_then(Value::as_array)
                .ok_or("Potter Blender attribute values are missing")?;
            if values.len() != count {
                return Err(format!(
                    "Potter `{name}` {domain} attribute has {} values for {count} elements",
                    values.len()
                )
                .into());
            }
            for (row, value) in rows.iter_mut().zip(values) {
                row.insert(name.to_owned(), value.clone());
            }
        }
    }
    Ok((!names.is_empty()).then_some(rows))
}

/// Per-face UV corners and their pin flags, indexed by output face.
type UvMapFaces = (Vec<Option<Vec<[f64; 2]>>>, Vec<Option<Vec<bool>>>);

/// Per-face UV corners of layer `UVMap` in Blender's V-up convention. The native `uv_layers` store
/// uses persistent face IDs and Potter's V-down convention; faces without a row remain `None`.
fn uv_map_faces(mesh: &Mesh) -> TestResult<UvMapFaces> {
    let mut faces: Vec<Option<Vec<[f64; 2]>>> = vec![None; mesh.faces.len()];
    let mut pins: Vec<Option<Vec<bool>>> = vec![None; mesh.faces.len()];
    let Some(store) = mesh.attributes.get("uv_layers") else {
        return Ok((faces, pins));
    };
    let layers = store
        .get("layers")
        .and_then(Value::as_array)
        .ok_or("Potter uv_layers list is missing")?;
    let Some(layer) = layers
        .iter()
        .find(|layer| layer.get("name").and_then(Value::as_str) == Some("UVMap"))
    else {
        return Ok((faces, pins));
    };
    pins = mesh
        .faces
        .iter()
        .map(|face| Some(vec![false; face.vertices.len()]))
        .collect();
    let entries = layer
        .get("faces")
        .and_then(Value::as_array)
        .ok_or("Potter UVMap face rows are missing")?;
    let face_index: HashMap<u64, usize> = mesh
        .faces
        .iter()
        .enumerate()
        .map(|(index, face)| (u64::from(face.id), index))
        .collect();
    for entry in entries {
        let id = entry
            .get("face_id")
            .and_then(Value::as_u64)
            .ok_or("Potter UVMap row has an invalid face ID")?;
        let index = *face_index
            .get(&id)
            .ok_or("Potter UVMap row references a missing face")?;
        let mut uv = uv_pairs(entry.get("uv").ok_or("Potter UVMap row has no values")?)?;
        if uv.len() != mesh.faces[index].vertices.len() {
            return Err(format!(
                "Potter UVMap face {} has {} corners, expected {}",
                mesh.faces[index].id,
                uv.len(),
                mesh.faces[index].vertices.len()
            )
            .into());
        }
        if faces[index].is_some() {
            return Err(format!(
                "Potter UVMap has duplicate rows for face {}",
                mesh.faces[index].id
            )
            .into());
        }
        let pin_row = if let Some(value) = entry.get("pinned").filter(|value| !value.is_null()) {
            let row = bools_from_json(value)?;
            if row.len() != mesh.faces[index].vertices.len() {
                return Err(format!(
                    "Potter UVMap face {} has {} pins, expected {}",
                    mesh.faces[index].id,
                    row.len(),
                    mesh.faces[index].vertices.len()
                )
                .into());
            }
            row
        } else {
            vec![false; mesh.faces[index].vertices.len()]
        };
        pins[index] = Some(pin_row);
        for pair in &mut uv {
            pair[1] = 1.0 - pair[1];
        }
        faces[index] = Some(uv);
    }
    Ok((faces, pins))
}

/// All named UV layers by face in Blender's V-up convention; layer rows are keyed by persistent face ID.
fn uv_layers_faces(mesh: &Mesh) -> TestResult<Vec<BTreeMap<String, UvLayerFace>>> {
    let mut faces = vec![BTreeMap::new(); mesh.faces.len()];
    let Some(store) = mesh.attributes.get("uv_layers") else {
        return Ok(faces);
    };
    let layers = store
        .get("layers")
        .and_then(Value::as_array)
        .ok_or("Potter uv_layers list is missing")?;
    let face_index: HashMap<u64, usize> = mesh
        .faces
        .iter()
        .enumerate()
        .map(|(index, face)| (u64::from(face.id), index))
        .collect();
    for layer in layers {
        let name = layer
            .get("name")
            .and_then(Value::as_str)
            .ok_or("Potter UV layer has an invalid name")?
            .to_owned();
        let entries = layer
            .get("faces")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("Potter UV layer `{name}` face rows are missing"))?;
        // Keep the layer name even when a face has no row, so name presence remains comparable.
        for face in &mut faces {
            if face.insert(name.clone(), UvLayerFace::default()).is_some() {
                return Err(format!("Potter has duplicate UV layer name `{name}`").into());
            }
        }
        let mut seen_rows = HashSet::new();
        for entry in entries {
            let id = entry
                .get("face_id")
                .and_then(Value::as_u64)
                .ok_or_else(|| format!("Potter UV layer `{name}` row has an invalid face ID"))?;
            let index = *face_index
                .get(&id)
                .ok_or_else(|| format!("Potter UV layer `{name}` row references a missing face"))?;
            if !seen_rows.insert(index) {
                return Err(format!(
                    "Potter UV layer `{name}` has duplicate rows for face {}",
                    mesh.faces[index].id
                )
                .into());
            }
            let mut uv = uv_pairs(
                entry
                    .get("uv")
                    .ok_or_else(|| format!("Potter UV layer `{name}` row has no values"))?,
            )?;
            let corner_count = mesh.faces[index].vertices.len();
            if uv.len() != corner_count {
                return Err(format!(
                    "Potter UV layer `{name}` face {} has {} corners, expected {corner_count}",
                    mesh.faces[index].id,
                    uv.len()
                )
                .into());
            }
            let pins = if let Some(value) = entry.get("pinned").filter(|value| !value.is_null()) {
                let pins = bools_from_json(value)?;
                if pins.len() != corner_count {
                    return Err(format!(
                        "Potter UV layer `{name}` face {} has {} pins, expected {corner_count}",
                        mesh.faces[index].id,
                        pins.len()
                    )
                    .into());
                }
                pins
            } else {
                vec![false; corner_count]
            };
            for pair in &mut uv {
                pair[1] = 1.0 - pair[1];
            }
            let payload = faces[index]
                .get_mut(&name)
                .ok_or_else(|| format!("Potter UV layer `{name}` is missing from its face"))?;
            payload.uv = uv;
            payload.pins = pins;
        }
    }
    Ok(faces)
}

/// Canonical form of an evaluated mesh. `world` maps the mesh's local space to world space.
fn potter_canonical(
    mesh: &Mesh,
    world: DMat4,
    object_materials: Vec<Option<String>>,
    effective_materials: Vec<Option<String>>,
    data_materials: Vec<Option<String>>,
    material_links: Vec<MaterialLink>,
) -> TestResult<Canonical> {
    let positions: HashMap<u32, DVec3> = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, world.transform_point3(vertex.co)))
        .collect();
    let edge_topology = mesh
        .edges
        .iter()
        .map(|edge| -> TestResult<_> {
            let first = positions
                .get(&edge.vertices[0])
                .copied()
                .ok_or("Potter edge references a missing vertex")?;
            let second = positions
                .get(&edge.vertices[1])
                .copied()
                .ok_or("Potter edge references a missing vertex")?;
            Ok([first.to_array(), second.to_array()])
        })
        .collect::<TestResult<Vec<_>>>()?;
    let point_attribute_rows = attribute_rows(mesh, "POINT")?;
    let point_attributes = point_attribute_rows.as_ref().map(|rows| {
        mesh.vertices
            .iter()
            .zip(rows)
            .map(|(vertex, values)| PointAttribute {
                co: world.transform_point3(vertex.co).to_array(),
                values: values.clone(),
            })
            .collect()
    });
    let point_values_by_vertex: HashMap<u32, AttributeValues> = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| {
            (
                vertex.id,
                point_attribute_rows
                    .as_ref()
                    .map_or_else(BTreeMap::new, |rows| rows[index].clone()),
            )
        })
        .collect();
    let edge_points = mesh
        .edges
        .iter()
        .map(|edge| -> TestResult<_> {
            let endpoint = |id| -> TestResult<_> {
                let co = positions
                    .get(&id)
                    .copied()
                    .ok_or("Potter edge references a missing vertex")?;
                let values = point_values_by_vertex
                    .get(&id)
                    .ok_or("Potter edge references a missing vertex")?;
                Ok(EdgePointEndpoint {
                    co: co.to_array(),
                    values: (!values.is_empty()).then(|| values.clone()),
                })
            };
            let endpoints = [endpoint(edge.vertices[0])?, endpoint(edge.vertices[1])?];
            Ok(EdgePoint {
                edge: [endpoints[0].co, endpoints[1].co],
                endpoints,
            })
        })
        .collect::<TestResult<Vec<_>>>()?;
    let edge_attributes = attribute_rows(mesh, "EDGE")?
        .map(|rows| {
            mesh.edges
                .iter()
                .zip(rows)
                .map(|(edge, values)| -> TestResult<_> {
                    let first = positions
                        .get(&edge.vertices[0])
                        .copied()
                        .ok_or("Potter edge references a missing vertex")?;
                    let second = positions
                        .get(&edge.vertices[1])
                        .copied()
                        .ok_or("Potter edge references a missing vertex")?;
                    Ok(EdgeAttribute {
                        edge: [first.to_array(), second.to_array()],
                        values,
                    })
                })
                .collect::<TestResult<Vec<_>>>()
        })
        .transpose()?;
    let face_attribute_rows = attribute_rows(mesh, "FACE")?;
    let corner_attribute_rows = attribute_rows(mesh, "CORNER")?;
    let (uv_faces, uv_pin_faces) = uv_map_faces(mesh)?;
    let uv_layer_faces = uv_layers_faces(mesh)?;
    let uv_layer_metadata = potter_uv_layer_metadata(mesh)?;
    let normal_rows = match mesh.attributes.get("custom_normal") {
        Some(attribute) if attribute.get("type").and_then(Value::as_str) != Some("normal_code") => {
            Some(
                attribute
                    .get("values")
                    .and_then(Value::as_object)
                    .ok_or("Potter custom_normal values are not an object")?,
            )
        }
        _ => None,
    };
    // Raw code rows are in current face/corner order; corner_order selects only the fan basis.
    let code_rows = match mesh.attributes.get("custom_normal") {
        Some(attribute) if attribute.get("type").and_then(Value::as_str) == Some("normal_code") => {
            Some(
                attribute
                    .get("values")
                    .and_then(Value::as_object)
                    .ok_or("Potter custom_normal values are not an object")?,
            )
        }
        _ => None,
    };

    let derived_normal_rows = if code_rows.is_some() {
        Some(potter_core::geom::corner_normal_vectors(mesh)?)
    } else {
        None
    };
    for rows in [normal_rows, code_rows].into_iter().flatten() {
        for key in rows.keys() {
            if !mesh
                .faces
                .iter()
                .any(|face| key == &format!("f{}", face.id))
            {
                return Err(format!("Potter custom normal has unmatched face key `{key}`").into());
            }
        }
    }
    let mut faces = Vec::with_capacity(mesh.faces.len());
    let mut rings = Vec::with_capacity(mesh.faces.len());
    let mut face_sizes = BTreeMap::new();
    let mut corner_offset = 0;
    for (index, (face, uv_layers)) in mesh.faces.iter().zip(uv_layer_faces).enumerate() {
        let ring = face
            .vertices
            .iter()
            .map(|id| {
                positions
                    .get(id)
                    .copied()
                    .ok_or("Potter face references a missing vertex")
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut face_attribute_values =
            face_attribute_rows.as_ref().map(|rows| rows[index].clone());
        let fid = face_attribute_values
            .as_mut()
            .and_then(|attributes| attributes.remove("fid"))
            .map(|value| value.as_i64().ok_or("Potter fid is not an integer"))
            .transpose()?;
        let face_attributes = face_attribute_values.filter(|attributes| !attributes.is_empty());
        let uv = uv_faces[index].as_deref();
        let key = format!("f{}", face.id);
        let normals = normal_rows
            .and_then(|rows| rows.get(&key))
            .map(points_from_json)
            .transpose()?;
        let custom_normal_codes = code_rows
            .and_then(|rows| rows.get(&key))
            .map(normal_codes_from_json)
            .transpose()?;
        let corner_count = face.vertices.len();
        let corner_attributes = corner_attribute_rows.as_ref().map(|rows| {
            let mut attributes = BTreeMap::<String, Vec<Value>>::new();
            for row in &rows[corner_offset..corner_offset + corner_count] {
                for (name, value) in row {
                    attributes
                        .entry(name.clone())
                        .or_default()
                        .push(value.clone());
                }
            }
            attributes
        });
        let uv_pins = uv_pin_faces[index].as_deref();
        let point_attribute_values = if point_attribute_rows.is_some() {
            Some(
                face.vertices
                    .iter()
                    .map(|id| {
                        point_values_by_vertex
                            .get(id)
                            .cloned()
                            .ok_or("Potter face references a missing vertex")
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )
        } else {
            None
        };
        let custom_normals = normals
            .map(|values| values.into_iter().map(|normal| normal.to_array()).collect())
            .or_else(|| {
                derived_normal_rows.as_ref().map(|values| {
                    values[corner_offset..corner_offset + corner_count]
                        .iter()
                        .map(glam::DVec3::to_array)
                        .collect()
                })
            });
        corner_offset += corner_count;
        faces.push(canonical_face(
            &ring,
            FacePayload {
                material: u64::from(face.material_index),
                fid,
                uv: uv.map(<[[f64; 2]]>::to_vec),
                uv_pins: uv_pins.map(<[bool]>::to_vec),
                uv_layers: Some(uv_layers),
                point_attributes: point_attribute_values,
                custom_normals,
                face_attributes,
                corner_attributes,
                custom_normal_codes,
            },
        ));
        *face_sizes.entry(face.vertices.len()).or_insert(0) += 1;
        rings.push(ring);
    }

    let mut vertex_groups = BTreeMap::new();
    if let Some(value) = mesh.attributes.get("vertex_groups") {
        let groups = value
            .as_object()
            .ok_or("Potter vertex_groups is not an object")?;
        for (name, value) in groups {
            let weights = value
                .as_object()
                .ok_or("Potter vertex group weights are not an object")?;
            for key in weights.keys() {
                let id = key
                    .strip_prefix('v')
                    .and_then(|id| id.parse::<u32>().ok())
                    .ok_or("Potter vertex group has an invalid vertex key")?;
                if !mesh.vertices.iter().any(|vertex| vertex.id == id) {
                    return Err(format!(
                        "Potter vertex group `{name}` references missing vertex `{key}`"
                    )
                    .into());
                }
            }
            let entries = mesh
                .vertices
                .iter()
                .map(|vertex| -> TestResult<_> {
                    let key = format!("v{}", vertex.id);
                    let weight = weights
                        .get(&key)
                        .map(|value| {
                            value
                                .as_f64()
                                .ok_or("Potter vertex-group weight is not numeric")
                        })
                        .transpose()?;
                    Ok((world.transform_point3(vertex.co).to_array(), weight))
                })
                .collect::<TestResult<Vec<_>>>()?;
            vertex_groups.insert(name.clone(), entries);
        }
    }

    Ok(Canonical {
        vertices: mesh.vertices.len(),
        edges: mesh.edges.len(),
        face_count: mesh.faces.len(),
        edge_topology: Some(edge_topology),
        edge_points: Some(edge_points),
        face_sizes,
        faces: Some(faces),
        vertex_groups,
        object_materials: Some(object_materials),
        effective_materials: Some(effective_materials),
        data_materials: Some(data_materials),
        material_links: Some(material_links),
        point_attributes,
        edge_attributes,
        color_layers: potter_color_layers(mesh)?,
        uv_layers: Some(uv_layer_metadata),
        volume: signed_volume(&rings),
    })
}

fn expected_face_sizes(value: &Value) -> TestResult<BTreeMap<usize, usize>> {
    let mut sizes = BTreeMap::new();
    for (size, count_value) in value.as_object().ok_or("Blender face sizes are missing")? {
        sizes.insert(size.parse::<usize>()?, count(count_value, "face size")?);
    }
    Ok(sizes)
}

fn expected_faces(value: &Value) -> TestResult<Vec<CanonicalFace>> {
    let mut faces = Vec::new();
    for face in value.as_array().ok_or("Blender faces are missing")? {
        let ring = points_from_json(&face["ring"])?;
        let material = face["material_index"]
            .as_u64()
            .ok_or("Blender material index is invalid")?;
        let fid = match face.get("fid") {
            Some(fid) => Some(fid.as_i64().ok_or("Blender fid is invalid")?),
            None => None,
        };
        let uv = face.get("uv").map(uv_pairs).transpose()?;
        let uv_pins = face.get("uv_pins").map(bools_from_json).transpose()?;
        let uv_layers = face
            .get("uv_layers")
            .map(|value| expected_uv_layers(value, ring.len()))
            .transpose()?;
        if let Some(pins) = &uv_pins
            && pins.len() != ring.len()
        {
            return Err(format!(
                "Blender UV pins have {} values for {} face corners",
                pins.len(),
                ring.len()
            )
            .into());
        }
        let point_attributes = face
            .get("point_attributes")
            .map(|value| expected_face_point_attributes(value, ring.len()))
            .transpose()?
            .flatten();
        let normals = face
            .get("custom_normals")
            .map(points_from_json)
            .transpose()?;
        let face_attributes = face
            .get("face_attributes")
            .map(expected_attribute_values)
            .transpose()?;
        let corner_attributes = face
            .get("corner_attributes")
            .map(expected_corner_attributes)
            .transpose()?;
        if let Some(attributes) = &corner_attributes {
            for (name, values) in attributes {
                if values.len() != ring.len() {
                    return Err(format!(
                        "Blender corner attribute `{name}` has {} values for {} face corners",
                        values.len(),
                        ring.len()
                    )
                    .into());
                }
            }
        }
        let custom_normal_codes = face
            .get("custom_normal_codes")
            .map(normal_codes_from_json)
            .transpose()?;
        if let Some(codes) = &custom_normal_codes
            && codes.len() != ring.len()
        {
            return Err(format!(
                "Blender custom-normal codes have {} values for {} face corners",
                codes.len(),
                ring.len()
            )
            .into());
        }
        faces.push(canonical_face(
            &ring,
            FacePayload {
                material,
                fid,
                uv,
                uv_pins,
                uv_layers,
                point_attributes,
                custom_normals: normals
                    .map(|values| values.into_iter().map(|normal| normal.to_array()).collect()),
                face_attributes,
                corner_attributes,
                custom_normal_codes,
            },
        ));
    }
    Ok(faces)
}

fn expected_attribute_values(value: &Value) -> TestResult<AttributeValues> {
    Ok(value
        .as_object()
        .ok_or("Blender attribute values must be an object")?
        .iter()
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect())
}

fn expected_face_point_attributes(
    value: &Value,
    ring_len: usize,
) -> TestResult<Option<Vec<AttributeValues>>> {
    let rows = value
        .as_array()
        .ok_or("Blender face point attributes must be an array")?;
    if rows.len() != ring_len {
        return Err(format!(
            "Blender face point attributes have {} values for {ring_len} face corners",
            rows.len()
        )
        .into());
    }
    let rows = rows
        .iter()
        .map(expected_attribute_values)
        .collect::<TestResult<Vec<_>>>()?;
    Ok(rows.iter().any(|values| !values.is_empty()).then_some(rows))
}

fn expected_corner_attributes(value: &Value) -> TestResult<BTreeMap<String, Vec<Value>>> {
    value
        .as_object()
        .ok_or("Blender corner attributes must be an object")?
        .iter()
        .map(|(name, values)| {
            Ok((
                name.clone(),
                values
                    .as_array()
                    .ok_or("Blender corner attribute values must be an array")?
                    .clone(),
            ))
        })
        .collect()
}

fn expected_point_attributes(value: &Value) -> TestResult<Vec<PointAttribute>> {
    value
        .as_array()
        .ok_or("Blender point attributes must be an array")?
        .iter()
        .map(|entry| {
            Ok(PointAttribute {
                co: point_from_json(&entry["co"])?.to_array(),
                values: expected_attribute_values(&entry["values"])?,
            })
        })
        .collect()
}

fn expected_edge_endpoints(value: &Value) -> TestResult<[[f64; 3]; 2]> {
    let endpoints = value
        .as_array()
        .filter(|endpoints| endpoints.len() == 2)
        .ok_or("a Blender edge must contain two endpoints")?;
    Ok([
        point_from_json(&endpoints[0])?.to_array(),
        point_from_json(&endpoints[1])?.to_array(),
    ])
}

fn expected_edges(value: &Value) -> TestResult<Vec<[[f64; 3]; 2]>> {
    value
        .as_array()
        .ok_or("Blender edges are missing")?
        .iter()
        .map(expected_edge_endpoints)
        .collect()
}

fn expected_edge_point_endpoint(value: &Value) -> TestResult<EdgePointEndpoint> {
    let values = expected_attribute_values(
        value
            .get("values")
            .ok_or("Blender edge-point endpoint values are missing")?,
    )?;
    Ok(EdgePointEndpoint {
        co: point_from_json(
            value
                .get("co")
                .ok_or("Blender edge-point endpoint coordinate is missing")?,
        )?
        .to_array(),
        values: (!values.is_empty()).then_some(values),
    })
}

fn expected_edge_points(value: &Value) -> TestResult<Vec<EdgePoint>> {
    value
        .as_array()
        .ok_or("Blender edge points must be an array")?
        .iter()
        .map(|entry| {
            let endpoints = entry
                .get("endpoints")
                .and_then(Value::as_array)
                .filter(|endpoints| endpoints.len() == 2)
                .ok_or("Blender edge points must have two endpoints")?;
            Ok(EdgePoint {
                edge: expected_edge_endpoints(
                    entry
                        .get("edge")
                        .ok_or("Blender edge-point coordinates are missing")?,
                )?,
                endpoints: [
                    expected_edge_point_endpoint(&endpoints[0])?,
                    expected_edge_point_endpoint(&endpoints[1])?,
                ],
            })
        })
        .collect()
}

fn expected_edge_attributes(value: &Value) -> TestResult<Vec<EdgeAttribute>> {
    value
        .as_array()
        .ok_or("Blender edge attributes must be an array")?
        .iter()
        .map(|entry| {
            Ok(EdgeAttribute {
                edge: expected_edge_endpoints(&entry["edge"])?,
                values: expected_attribute_values(&entry["values"])?,
            })
        })
        .collect()
}

fn optional_layer_name(value: &Value, key: &str, owner: &str) -> TestResult<Option<String>> {
    let field = value
        .get(key)
        .ok_or_else(|| format!("{owner} `{key}` is missing"))?;
    if field.is_null() {
        Ok(None)
    } else {
        field
            .as_str()
            .map(|text| Some(text.to_owned()))
            .ok_or_else(|| format!("{owner} `{key}` is invalid").into())
    }
}

fn expected_color_layers(value: &Value) -> TestResult<Option<ColorLayers>> {
    let Some(value) = value.get("color_layers") else {
        return Ok(None);
    };
    let layers = value
        .get("layers")
        .and_then(Value::as_array)
        .ok_or("Blender color-layer list is missing")?
        .iter()
        .map(|layer| {
            Ok(ColorLayer {
                name: layer["name"]
                    .as_str()
                    .ok_or("Blender color-layer name is missing")?
                    .to_owned(),
                domain: layer["domain"]
                    .as_str()
                    .ok_or("Blender color-layer domain is missing")?
                    .to_owned(),
                data_type: layer["data_type"]
                    .as_str()
                    .ok_or("Blender color-layer type is missing")?
                    .to_owned(),
            })
        })
        .collect::<TestResult<Vec<_>>>()?;
    Ok(Some(ColorLayers {
        layers,
        active: optional_layer_name(value, "active", "Blender color-layer")?,
        render: optional_layer_name(value, "render", "Blender color-layer")?,
    }))
}
fn expected_uv_layer_metadata(value: &Value) -> TestResult<Option<UvLayerMetadata>> {
    let Some(value) = value.get("uv_layers") else {
        return Ok(None);
    };
    let layers = value
        .get("layers")
        .and_then(Value::as_array)
        .ok_or("Blender UV-layer list is missing")?
        .iter()
        .map(|layer| {
            let name = layer.as_str().ok_or("Blender UV-layer name is invalid")?;
            Ok(name.to_owned())
        })
        .collect::<TestResult<Vec<_>>>()?;
    Ok(Some(UvLayerMetadata {
        layers,
        active: optional_layer_name(value, "active", "Blender UV-layer")?,
        render: optional_layer_name(value, "render", "Blender UV-layer")?,
    }))
}

fn potter_uv_layer_metadata(mesh: &Mesh) -> TestResult<UvLayerMetadata> {
    let Some(store) = mesh.attributes.get("uv_layers") else {
        return Ok(UvLayerMetadata::default());
    };
    let layers = store
        .get("layers")
        .and_then(Value::as_array)
        .ok_or("Potter UV-layer list is missing")?
        .iter()
        .map(|layer| {
            let name = layer
                .get("name")
                .and_then(Value::as_str)
                .ok_or("Potter UV-layer name is missing")?;
            Ok(name.to_owned())
        })
        .collect::<TestResult<Vec<_>>>()?;
    Ok(UvLayerMetadata {
        layers,
        active: optional_layer_name(store, "active", "Potter UV-layer")?,
        render: optional_layer_name(store, "render", "Potter UV-layer")?,
    })
}

fn potter_color_layers(mesh: &Mesh) -> TestResult<Option<ColorLayers>> {
    let Some(store) = mesh.attributes.get("color_layers") else {
        return Ok(None);
    };
    let names = store
        .get("layers")
        .and_then(Value::as_array)
        .ok_or("Potter color-layer list is missing")?;
    let data_types = mesh
        .attributes
        .get("blender_metadata")
        .and_then(|metadata| metadata.get("color_data_types"))
        .and_then(Value::as_object);
    let layers = names
        .iter()
        .map(|name| -> TestResult<_> {
            let name = name.as_str().ok_or("Potter color-layer name is invalid")?;
            let attribute =
                color_attribute(mesh, name).ok_or("Potter color-layer values are missing")?;
            let domain = match attribute.get("domain").and_then(Value::as_str) {
                Some("point") => "POINT",
                Some("corner") => "CORNER",
                _ => {
                    return Err(format!("Potter color layer `{name}` has an invalid domain").into());
                }
            };
            let data_type = data_types
                .and_then(|types| types.get(name))
                .and_then(Value::as_str)
                .ok_or_else(|| format!("Potter color-layer type `{name}` is missing"))?;
            Ok(ColorLayer {
                name: name.to_owned(),
                domain: domain.to_owned(),
                data_type: data_type.to_owned(),
            })
        })
        .collect::<TestResult<Vec<_>>>()?;
    Ok(Some(ColorLayers {
        layers,
        active: optional_layer_name(store, "active", "Potter color-layer")?,
        render: optional_layer_name(store, "render", "Potter color-layer")?,
    }))
}

fn expected_vertex_groups(value: &Value) -> TestResult<BTreeMap<String, Vec<GroupEntry>>> {
    let mut groups = BTreeMap::new();
    for (name, entries) in value
        .as_object()
        .ok_or("Blender vertex groups must be an object")?
    {
        let mut list = Vec::new();
        for entry in entries
            .as_array()
            .ok_or("Blender vertex group entries are missing")?
        {
            let co = point_from_json(&entry["co"])?.to_array();
            let weight = match entry.get("weight") {
                Some(weight) if !weight.is_null() => Some(number(weight)?),
                _ => None,
            };
            list.push((co, weight));
        }
        groups.insert(name.clone(), list);
    }
    Ok(groups)
}

/// Nullable material names in slot order; `null` is an empty slot.
fn expected_material_names(materials: &Value, owner: &str) -> TestResult<Vec<Option<String>>> {
    materials
        .as_array()
        .ok_or_else(|| format!("Blender {owner} must be an array"))?
        .iter()
        .map(|material| -> TestResult<Option<String>> {
            if material.is_null() {
                Ok(None)
            } else {
                Ok(Some(
                    material
                        .as_str()
                        .ok_or_else(|| format!("Blender {owner} entry must be a name or null"))?
                        .to_owned(),
                ))
            }
        })
        .collect()
}

/// Link modes in slot order: `data` or `object`.
fn expected_material_links(links: &Value) -> TestResult<Vec<MaterialLink>> {
    links
        .as_array()
        .ok_or("Blender material links must be an array")?
        .iter()
        .map(|link| -> TestResult<MaterialLink> {
            match link.as_str() {
                Some("data") => Ok(MaterialLink::Data),
                Some("object") => Ok(MaterialLink::Object),
                _ => Err("Blender material link must be \"data\" or \"object\"".into()),
            }
        })
        .collect()
}

/// The canonical form of a Blender expectation (`expected/<case>.json`).
pub fn expected_canonical(expected: &Value) -> TestResult<Canonical> {
    let faces = match expected.get("faces") {
        Some(faces) => Some(expected_faces(faces)?),
        None => None,
    };
    let vertex_groups = match expected.get("vertex_groups") {
        Some(groups) => expected_vertex_groups(groups)?,
        None => BTreeMap::new(),
    };
    let point_attributes = expected
        .get("point_attributes")
        .map(expected_point_attributes)
        .transpose()?;
    let edge_points = expected
        .get("edge_points")
        .map(expected_edge_points)
        .transpose()?;
    let edge_attributes = expected
        .get("edge_attributes")
        .map(expected_edge_attributes)
        .transpose()?;
    let color_layers = expected_color_layers(expected)?;
    let uv_layers = expected_uv_layer_metadata(expected)?;
    let edge_topology = match expected.get("edges") {
        Some(edges) => Some(expected_edges(edges)?),
        None => faces.as_deref().map(expected_edge_topology),
    };
    let object_materials = expected
        .get("object_materials")
        .map(|materials| expected_material_names(materials, "object materials"))
        .transpose()?;
    let effective_materials = expected
        .get("effective_materials")
        .map(|materials| expected_material_names(materials, "effective materials"))
        .transpose()?;
    let data_materials = expected
        .get("data_materials")
        .map(|materials| expected_material_names(materials, "data materials"))
        .transpose()?;
    let material_links = expected
        .get("material_links")
        .map(expected_material_links)
        .transpose()?;
    Ok(Canonical {
        vertices: count(&expected["counts"]["vertices"], "vertex")?,
        face_count: count(&expected["counts"]["faces"], "face")?,
        edges: count(&expected["counts"]["edges"], "edge")?,
        edge_topology,
        edge_points,
        face_sizes: expected_face_sizes(&expected["face_sizes"])?,
        faces,
        vertex_groups,
        point_attributes,
        edge_attributes,
        color_layers,
        object_materials,
        effective_materials,
        data_materials,
        material_links,
        uv_layers,
        volume: expected["volume"]
            .as_f64()
            .ok_or("Blender volume is missing")?,
    })
}

fn volume_matches(actual: f64, expected: f64) -> bool {
    if expected.abs() <= VOLUME_ABSOLUTE_TOLERANCE {
        actual.abs() <= VOLUME_ABSOLUTE_TOLERANCE
    } else {
        ((actual - expected) / expected).abs() <= VOLUME_RELATIVE_TOLERANCE
    }
}

fn uv_layers_match_rotation(
    actual: &BTreeMap<String, UvLayerFace>,
    expected: &BTreeMap<String, UvLayerFace>,
    ring_len: usize,
    rotation: usize,
) -> bool {
    actual.keys().eq(expected.keys())
        && expected.iter().all(|(name, expected_layer)| {
            let Some(actual_layer) = actual.get(name) else {
                return false;
            };
            actual_layer.uv.len() == ring_len
                && actual_layer.pins.len() == ring_len
                && expected_layer.uv.len() == ring_len
                && expected_layer.pins.len() == ring_len
                && expected_layer
                    .uv
                    .iter()
                    .enumerate()
                    .all(|(corner, expected_uv)| {
                        actual_layer.uv[(rotation + corner) % ring_len]
                            .into_iter()
                            .zip(*expected_uv)
                            .all(|(actual, expected)| {
                                value_matches(actual, expected, FLOAT_TOLERANCE)
                            })
                    })
                && expected_layer
                    .pins
                    .iter()
                    .enumerate()
                    .all(|(corner, expected_pin)| {
                        actual_layer.pins[(rotation + corner) % ring_len] == *expected_pin
                    })
        })
}

fn face_signature_matches_rotation(
    actual: &CanonicalFace,
    expected: &CanonicalFace,
    rotation: usize,
) -> bool {
    let point_attributes_match = match &expected.point_attributes {
        None => true,
        Some(expected_rows) => actual.point_attributes.as_ref().is_some_and(|actual_rows| {
            expected_rows.len() == actual.ring.len()
                && actual_rows.len() == actual.ring.len()
                && expected_rows
                    .iter()
                    .enumerate()
                    .all(|(corner, expected_values)| {
                        attribute_values_match(
                            &actual_rows[(rotation + corner) % actual.ring.len()],
                            expected_values,
                        )
                    })
        }),
    };
    let uv_layers_match = match &expected.uv_layers {
        None => true,
        Some(expected_layers) => actual.uv_layers.as_ref().is_some_and(|actual_layers| {
            uv_layers_match_rotation(actual_layers, expected_layers, actual.ring.len(), rotation)
        }),
    };
    let uv_pins_match = match &expected.uv_pins {
        None => true,
        Some(expected_pins) => actual.uv_pins.as_ref().is_some_and(|actual_pins| {
            expected_pins.len() == actual.ring.len()
                && actual_pins.len() == actual.ring.len()
                && expected_pins
                    .iter()
                    .enumerate()
                    .all(|(corner, expected_pin)| {
                        actual_pins[(rotation + corner) % actual.ring.len()] == *expected_pin
                    })
        }),
    };
    point_attributes_match && uv_pins_match && uv_layers_match
}

fn face_rotations(actual: &CanonicalFace, expected: &CanonicalFace) -> Vec<usize> {
    if actual.ring.len() != expected.ring.len() {
        return Vec::new();
    }
    let rotations: Vec<usize> = (0..actual.ring.len())
        .filter(|&start| {
            expected.ring.iter().enumerate().all(|(corner, point)| {
                point_matches(actual.ring[(start + corner) % actual.ring.len()], *point)
            })
        })
        .collect();
    let signatures = rotations
        .iter()
        .copied()
        .filter(|&rotation| face_signature_matches_rotation(actual, expected, rotation))
        .collect::<Vec<_>>();
    if signatures.is_empty() {
        rotations
    } else {
        signatures
    }
}

fn scalar_option_matches(actual: Option<f64>, expected: Option<f64>, tolerance: f64) -> bool {
    match (actual, expected) {
        (Some(actual), Some(expected)) => value_matches(actual, expected, tolerance),
        (None, None) => true,
        _ => false,
    }
}

fn json_value_matches(actual: &Value, expected: &Value) -> bool {
    match (actual.as_f64(), expected.as_f64()) {
        (Some(actual), Some(expected)) => value_matches(actual, expected, FLOAT_TOLERANCE),
        _ => match (actual, expected) {
            (Value::Array(actual), Value::Array(expected)) => {
                actual.len() == expected.len()
                    && actual
                        .iter()
                        .zip(expected)
                        .all(|(actual, expected)| json_value_matches(actual, expected))
            }
            (Value::Object(actual), Value::Object(expected)) => {
                actual.len() == expected.len()
                    && actual.iter().all(|(key, value)| {
                        expected
                            .get(key)
                            .is_some_and(|expected| json_value_matches(value, expected))
                    })
            }
            _ => actual == expected,
        },
    }
}

fn attribute_values_match(actual: &AttributeValues, expected: &AttributeValues) -> bool {
    actual.len() == expected.len()
        && actual.iter().all(|(name, value)| {
            expected
                .get(name)
                .is_some_and(|expected| json_value_matches(value, expected))
        })
}

fn compare_face_attribute_map(
    actual: &Option<AttributeValues>,
    expected: &Option<AttributeValues>,
    context: &str,
    name: &str,
) -> Vec<String> {
    match (actual, expected) {
        (Some(actual), Some(expected)) if attribute_values_match(actual, expected) => Vec::new(),
        (Some(actual), Some(expected)) => {
            vec![format!(
                "{context}: {name} {actual:?} differs from Blender {expected:?}"
            )]
        }
        (Some(actual), None) => vec![format!("{context}: unexpected {name} {actual:?}")],
        (None, Some(expected)) => {
            vec![format!("{context}: missing Blender {name} {expected:?}")]
        }
        (None, None) => Vec::new(),
    }
}

fn compare_corner_attributes(
    actual: &Option<BTreeMap<String, Vec<Value>>>,
    expected: &Option<BTreeMap<String, Vec<Value>>>,
    ring_len: usize,
    rotation: usize,
    context: &str,
) -> Vec<String> {
    let mut differences = Vec::new();
    match (actual, expected) {
        (Some(actual), Some(expected)) => {
            if actual.keys().ne(expected.keys()) {
                differences.push(format!(
                    "{context}: corner attribute names {:?} differ from Blender {:?}",
                    actual.keys().collect::<Vec<_>>(),
                    expected.keys().collect::<Vec<_>>()
                ));
            }
            for (name, expected_values) in expected {
                let Some(actual_values) = actual.get(name) else {
                    continue;
                };
                if actual_values.len() != expected_values.len() {
                    differences.push(format!(
                        "{context}: corner attribute `{name}` has {} Potter values, Blender has {}",
                        actual_values.len(),
                        expected_values.len()
                    ));
                }
                for (corner, expected_value) in expected_values.iter().enumerate() {
                    if ring_len == 0 {
                        break;
                    }
                    let actual_corner = (rotation + corner) % ring_len;
                    let Some(actual_value) = actual_values.get(actual_corner) else {
                        continue;
                    };
                    if !json_value_matches(actual_value, expected_value) {
                        differences.push(format!(
                            "{context}, corner {corner}: attribute `{name}` value {actual_value:?} differs from Blender {expected_value:?}"
                        ));
                    }
                }
            }
        }
        (Some(actual), None) => {
            differences.push(format!(
                "{context}: unexpected corner attributes {actual:?}"
            ));
        }
        (None, Some(expected)) => {
            differences.push(format!(
                "{context}: missing Blender corner attributes {expected:?}"
            ));
        }
        (None, None) => {}
    }
    differences
}

fn compare_face_point_attributes(
    actual: &Option<Vec<AttributeValues>>,
    expected: &Option<Vec<AttributeValues>>,
    ring_len: usize,
    rotation: usize,
    context: &str,
) -> Vec<String> {
    let mut differences = Vec::new();
    match (actual, expected) {
        (_, None) => {}
        (Some(actual), Some(expected)) => {
            if actual.len() != expected.len() {
                differences.push(format!(
                    "{context}: point-attribute corner count {} differs from Blender {}",
                    actual.len(),
                    expected.len()
                ));
            }
            for (corner, expected_values) in expected.iter().enumerate() {
                if ring_len == 0 {
                    break;
                }
                let actual_corner = (rotation + corner) % ring_len;
                let Some(actual_values) = actual.get(actual_corner) else {
                    continue;
                };
                if !attribute_values_match(actual_values, expected_values) {
                    differences.push(format!(
                        "{context}, corner {corner}: point attributes {actual_values:?} differ from Blender {expected_values:?}"
                    ));
                }
            }
        }
        (None, Some(expected)) => differences.push(format!(
            "{context}: missing Blender point-attribute corners {expected:?}"
        )),
    }
    differences
}

fn compare_uv_pins(
    actual: &Option<Vec<bool>>,
    expected: &Option<Vec<bool>>,
    ring_len: usize,
    rotation: usize,
    context: &str,
) -> Vec<String> {
    let mut differences = Vec::new();
    match (actual, expected) {
        (_, None) => {}
        (Some(actual), Some(expected)) => {
            if actual.len() != expected.len() {
                differences.push(format!(
                    "{context}: UV pin count {} differs from Blender {}",
                    actual.len(),
                    expected.len()
                ));
            }
            for (corner, expected_pin) in expected.iter().enumerate() {
                if ring_len == 0 {
                    break;
                }
                let actual_corner = (rotation + corner) % ring_len;
                let Some(actual_pin) = actual.get(actual_corner) else {
                    continue;
                };
                if actual_pin != expected_pin {
                    differences.push(format!(
                        "{context}, corner {corner}: UV pin {actual_pin} differs from Blender {expected_pin}"
                    ));
                }
            }
        }
        (None, Some(expected)) => {
            differences.push(format!("{context}: missing Blender UV pins {expected:?}"));
        }
    }
    differences
}

fn compare_uv_layers(
    actual: &Option<BTreeMap<String, UvLayerFace>>,
    expected: &Option<BTreeMap<String, UvLayerFace>>,
    ring_len: usize,
    rotation: usize,
    context: &str,
) -> Vec<String> {
    let mut differences = Vec::new();
    match (actual, expected) {
        (_, None) => {}
        (None, Some(expected)) => differences.push(format!(
            "{context}: missing Blender named UV layers {:?}",
            expected.keys().collect::<Vec<_>>()
        )),
        (Some(actual), Some(expected)) => {
            let missing = expected
                .keys()
                .filter(|name| !actual.contains_key(*name))
                .cloned()
                .collect::<Vec<_>>();
            let unexpected = actual
                .keys()
                .filter(|name| !expected.contains_key(*name))
                .cloned()
                .collect::<Vec<_>>();
            if !missing.is_empty() {
                differences.push(format!("{context}: missing Blender UV layers {missing:?}"));
            }
            if !unexpected.is_empty() {
                differences.push(format!("{context}: unexpected UV layers {unexpected:?}"));
            }
            for (name, expected_layer) in expected {
                let Some(actual_layer) = actual.get(name) else {
                    continue;
                };
                if actual_layer.uv.len() != expected_layer.uv.len() {
                    differences.push(format!(
                        "{context}: UV layer `{name}` corner count {} differs from Blender {}",
                        actual_layer.uv.len(),
                        expected_layer.uv.len()
                    ));
                }
                if actual_layer.pins.len() != expected_layer.pins.len() {
                    differences.push(format!(
                        "{context}: UV layer `{name}` pin count {} differs from Blender {}",
                        actual_layer.pins.len(),
                        expected_layer.pins.len()
                    ));
                }
                for (corner, expected_uv) in expected_layer.uv.iter().enumerate() {
                    if ring_len == 0 {
                        break;
                    }
                    if corner >= ring_len {
                        differences.push(format!(
                            "{context}: Blender UV layer `{name}` corner {corner} has no associated face vertex"
                        ));
                        continue;
                    }
                    let actual_corner = (rotation + corner) % ring_len;
                    let Some(actual_uv) = actual_layer.uv.get(actual_corner) else {
                        continue;
                    };
                    if !(*actual_uv)
                        .into_iter()
                        .zip(*expected_uv)
                        .all(|(actual, expected)| value_matches(actual, expected, FLOAT_TOLERANCE))
                    {
                        differences.push(format!(
                            "{context}, UV layer `{name}` corner {corner}: UV {actual_uv:?} differs from Blender {expected_uv:?}"
                        ));
                    }
                }
                for (corner, expected_pin) in expected_layer.pins.iter().enumerate() {
                    if ring_len == 0 {
                        break;
                    }
                    if corner >= ring_len {
                        differences.push(format!(
                            "{context}: Blender UV layer `{name}` pin {corner} has no associated face vertex"
                        ));
                        continue;
                    }
                    let actual_corner = (rotation + corner) % ring_len;
                    let Some(actual_pin) = actual_layer.pins.get(actual_corner) else {
                        continue;
                    };
                    if actual_pin != expected_pin {
                        differences.push(format!(
                            "{context}, UV layer `{name}` corner {corner}: pin {actual_pin} differs from Blender {expected_pin}"
                        ));
                    }
                }
            }
        }
    }
    differences
}

fn compare_custom_normal_codes(
    actual: &Option<Vec<NormalCode>>,
    expected: &Option<Vec<NormalCode>>,
    ring_len: usize,
    rotation: usize,
    context: &str,
) -> Vec<String> {
    let mut differences = Vec::new();
    match (actual, expected) {
        (Some(actual), Some(expected)) => {
            if actual.len() != expected.len() {
                differences.push(format!(
                    "{context}: custom-normal code count {} differs from Blender {}",
                    actual.len(),
                    expected.len()
                ));
            }
            for (corner, expected_code) in expected.iter().enumerate() {
                if ring_len == 0 {
                    break;
                }
                let actual_corner = (rotation + corner) % ring_len;
                let Some(actual_code) = actual.get(actual_corner) else {
                    continue;
                };
                if actual_code != expected_code {
                    differences.push(format!(
                        "{context}, corner {corner}: custom-normal code {actual_code:?} differs from Blender {expected_code:?}"
                    ));
                }
            }
        }
        (Some(actual), None) => {
            differences.push(format!(
                "{context}: unexpected custom-normal codes {actual:?}"
            ));
        }
        (None, Some(expected)) => {
            differences.push(format!(
                "{context}: missing Blender custom-normal codes {expected:?}"
            ));
        }
        (None, None) => {}
    }
    differences
}

fn compare_face_payload(
    actual: &CanonicalFace,
    expected: &CanonicalFace,
    actual_index: usize,
    expected_index: usize,
    rotation: usize,
) -> Vec<String> {
    let context = format!(
        "Blender face #{expected_index} {:?} (Potter face #{actual_index})",
        expected.ring
    );
    let mut differences = Vec::new();
    if actual.material != expected.material {
        differences.push(format!(
            "{context}: material index {} differs from Blender {}",
            actual.material, expected.material
        ));
    }
    if actual.fid != expected.fid {
        differences.push(format!(
            "{context}: fid {:?} differs from Blender {:?}",
            actual.fid, expected.fid
        ));
    }
    differences.extend(compare_face_attribute_map(
        &actual.face_attributes,
        &expected.face_attributes,
        &context,
        "face attributes",
    ));
    differences.extend(compare_corner_attributes(
        &actual.corner_attributes,
        &expected.corner_attributes,
        actual.ring.len(),
        rotation,
        &context,
    ));
    differences.extend(compare_face_point_attributes(
        &actual.point_attributes,
        &expected.point_attributes,
        actual.ring.len(),
        rotation,
        &context,
    ));
    differences.extend(compare_uv_pins(
        &actual.uv_pins,
        &expected.uv_pins,
        actual.ring.len(),
        rotation,
        &context,
    ));
    differences.extend(compare_uv_layers(
        &actual.uv_layers,
        &expected.uv_layers,
        actual.ring.len(),
        rotation,
        &context,
    ));
    differences.extend(compare_custom_normal_codes(
        &actual.custom_normal_codes,
        &expected.custom_normal_codes,
        actual.ring.len(),
        rotation,
        &context,
    ));
    match (&actual.uv, &expected.uv) {
        (Some(actual_uv), Some(expected_uv)) => {
            if actual_uv.len() != expected_uv.len() {
                differences.push(format!(
                    "{context}: UV corner count {} differs from Blender {}",
                    actual_uv.len(),
                    expected_uv.len()
                ));
            }
            for (corner, expected_value) in expected_uv.iter().enumerate() {
                if actual.ring.is_empty() || actual_uv.is_empty() {
                    break;
                }
                if corner >= expected.ring.len() {
                    differences.push(format!(
                        "{context}: Blender UV corner {corner} has no associated face vertex"
                    ));
                    continue;
                }
                let actual_corner = (rotation + corner) % actual.ring.len();
                let actual_value = actual_uv[actual_corner % actual_uv.len()];
                if !actual_value
                    .into_iter()
                    .zip(*expected_value)
                    .all(|(actual, expected)| value_matches(actual, expected, FLOAT_TOLERANCE))
                {
                    differences.push(format!(
                        "{context}, corner {corner} at {:?}, UV {:?} differs from Blender {:?}",
                        expected.ring[corner], actual_value, expected_value
                    ));
                }
            }
        }
        (Some(actual_uv), None) => {
            differences.push(format!("{context}: unexpected UV corners {actual_uv:?}"));
        }
        (None, Some(expected_uv)) => {
            differences.push(format!(
                "{context}: missing Blender UV corners {expected_uv:?}"
            ));
        }
        (None, None) => {}
    }
    if expected.custom_normals.is_some()
        || actual.custom_normal_codes.is_none()
        || expected.custom_normal_codes.is_none()
    {
        match (&actual.custom_normals, &expected.custom_normals) {
            (Some(actual_normals), Some(expected_normals)) => {
                if actual_normals.len() != expected_normals.len() {
                    differences.push(format!(
                        "{context}: custom-normal corner count {} differs from Blender {}",
                        actual_normals.len(),
                        expected_normals.len()
                    ));
                }
                for (corner, expected_normal) in expected_normals.iter().enumerate() {
                    if actual.ring.is_empty() || actual_normals.is_empty() {
                        break;
                    }
                    if corner >= expected.ring.len() {
                        differences.push(format!(
                            "{context}: Blender custom-normal corner {corner} has no associated face vertex"
                        ));
                        continue;
                    }
                    let actual_corner = (rotation + corner) % actual.ring.len();
                    let actual_normal = actual_normals[actual_corner % actual_normals.len()];
                    if !actual_normal
                        .into_iter()
                        .zip(*expected_normal)
                        .all(|(actual, expected)| value_matches(actual, expected, FLOAT_TOLERANCE))
                    {
                        let uv = expected.uv.as_ref().and_then(|values| values.get(corner));
                        differences.push(format!(
                            "{context}, corner {corner} at {:?}, UV {:?}: custom normal {:?} differs from Blender {:?}",
                            expected.ring[corner], uv, actual_normal, expected_normal
                        ));
                    }
                }
            }
            (Some(actual_normals), None) => {
                differences.push(format!(
                    "{context}: unexpected custom normals {actual_normals:?}"
                ));
            }
            (None, Some(expected_normals)) => {
                differences.push(format!(
                    "{context}: missing Blender custom normals {expected_normals:?}"
                ));
            }
            (None, None) => {}
        }
    }
    differences
}

fn compare_faces(actual: &[CanonicalFace], expected: &[CanonicalFace]) -> Vec<String> {
    let mut differences = Vec::new();
    if actual.len() != expected.len() {
        differences.push(format!(
            "face count {} differs from Blender {}",
            actual.len(),
            expected.len()
        ));
    }
    let mut used = vec![false; actual.len()];
    for (expected_index, expected_face) in expected.iter().enumerate() {
        let candidates: Vec<(usize, Vec<usize>)> = actual
            .iter()
            .enumerate()
            .filter(|(index, _)| !used[*index])
            .map(|(index, face)| (index, face_rotations(face, expected_face)))
            .filter(|(_, rotations)| !rotations.is_empty())
            .collect();
        if candidates.is_empty() {
            let rings: Vec<_> = actual
                .iter()
                .enumerate()
                .filter(|(_, face)| face.ring.len() == expected_face.ring.len())
                .map(|(index, face)| (index, &face.ring))
                .collect();
            differences.push(format!(
                "Blender face #{expected_index} topology/ring {:?} has no matching Potter face; same-sized Potter rings: {:?}",
                expected_face.ring, rings
            ));
            continue;
        }
        let selected = if candidates.len() == 1 {
            0
        } else {
            candidates
                .iter()
                .position(|(actual_index, rotations)| {
                    rotations.len() == 1
                        && compare_face_payload(
                            &actual[*actual_index],
                            expected_face,
                            *actual_index,
                            expected_index,
                            rotations[0],
                        )
                        .is_empty()
                })
                .unwrap_or(0)
        };
        let (actual_index, rotations) = &candidates[selected];
        if rotations.len() != 1 {
            differences.push(format!(
                "Blender face #{expected_index} topology/ring {:?} has ambiguous cyclic connectivity in Potter face #{actual_index}",
                expected_face.ring
            ));
            used[*actual_index] = true;
            continue;
        }
        used[*actual_index] = true;
        differences.extend(compare_face_payload(
            &actual[*actual_index],
            expected_face,
            *actual_index,
            expected_index,
            rotations[0],
        ));
    }
    for (index, face) in actual.iter().enumerate() {
        if !used[index] {
            differences.push(format!(
                "unexpected Potter face #{index} ring {:?}",
                face.ring
            ));
        }
    }
    differences
}

fn compare_edges(actual: &[[[f64; 3]; 2]], expected: &[[[f64; 3]; 2]]) -> Vec<String> {
    let mut differences = Vec::new();
    if actual.len() != expected.len() {
        differences.push(format!(
            "edge topology has {} Potter edges, Blender has {}",
            actual.len(),
            expected.len()
        ));
    }
    let mut used = vec![false; actual.len()];
    for (expected_index, expected_edge) in expected.iter().enumerate() {
        let matched = actual
            .iter()
            .enumerate()
            .find(|(index, edge)| !used[*index] && same_edge(**edge, *expected_edge));
        if let Some((actual_index, _)) = matched {
            used[actual_index] = true;
        } else {
            differences.push(format!(
                "Blender edge #{expected_index} {expected_edge:?} has no matching Potter edge; Potter edges: {actual:?}"
            ));
        }
    }
    for (index, edge) in actual.iter().enumerate() {
        if !used[index] {
            differences.push(format!("unexpected Potter edge #{index} {edge:?}"));
        }
    }
    differences
}

fn edge_point_endpoint_matches(actual: &EdgePointEndpoint, expected: &EdgePointEndpoint) -> bool {
    point_matches(actual.co, expected.co)
        && match (&actual.values, &expected.values) {
            (Some(actual), Some(expected)) => attribute_values_match(actual, expected),
            (None, None) => true,
            _ => false,
        }
}

fn edge_point_matches(actual: &EdgePoint, expected: &EdgePoint) -> bool {
    same_edge(actual.edge, expected.edge)
        && ((edge_point_endpoint_matches(&actual.endpoints[0], &expected.endpoints[0])
            && edge_point_endpoint_matches(&actual.endpoints[1], &expected.endpoints[1]))
            || (edge_point_endpoint_matches(&actual.endpoints[0], &expected.endpoints[1])
                && edge_point_endpoint_matches(&actual.endpoints[1], &expected.endpoints[0])))
}

fn compare_edge_points(
    actual: &Option<Vec<EdgePoint>>,
    expected: &Option<Vec<EdgePoint>>,
) -> Vec<String> {
    let mut differences = Vec::new();
    match (actual, expected) {
        (_, None) => {}
        (None, Some(expected)) => differences.push(format!(
            "Potter edge-point associations are missing; Blender has {} rows",
            expected.len()
        )),
        (Some(actual), Some(expected)) => {
            if actual.len() != expected.len() {
                differences.push(format!(
                    "edge-point associations have {} Potter rows, Blender has {}",
                    actual.len(),
                    expected.len()
                ));
            }
            let mut used = vec![false; actual.len()];
            for (expected_index, expected_entry) in expected.iter().enumerate() {
                let matched = actual.iter().enumerate().find(|(index, actual_entry)| {
                    !used[*index] && edge_point_matches(actual_entry, expected_entry)
                });
                if let Some((actual_index, _)) = matched {
                    used[actual_index] = true;
                } else {
                    differences.push(format!(
                        "Blender edge-point row #{expected_index} {:?} has no matching Potter row",
                        expected_entry.endpoints
                    ));
                }
            }
            for (index, entry) in actual.iter().enumerate() {
                if !used[index] {
                    differences.push(format!(
                        "unexpected Potter edge-point row #{index} {:?}",
                        entry.endpoints
                    ));
                }
            }
        }
    }
    differences
}

fn compare_point_attributes(
    actual: &Option<Vec<PointAttribute>>,
    expected: &Option<Vec<PointAttribute>>,
) -> Vec<String> {
    let mut differences = Vec::new();
    match (actual, expected) {
        (Some(actual), Some(expected)) => {
            if actual.len() != expected.len() {
                differences.push(format!(
                    "point attributes have {} Potter rows, Blender has {}",
                    actual.len(),
                    expected.len()
                ));
            }
            let mut used = vec![false; actual.len()];
            for expected_entry in expected {
                let matched = actual.iter().enumerate().find(|(index, actual_entry)| {
                    !used[*index]
                        && point_matches(actual_entry.co, expected_entry.co)
                        && attribute_values_match(&actual_entry.values, &expected_entry.values)
                });
                if let Some((index, _)) = matched {
                    used[index] = true;
                } else {
                    differences.push(format!(
                        "missing Blender point attribute at {:?}: {:?}",
                        expected_entry.co, expected_entry.values
                    ));
                }
            }
            for (index, entry) in actual.iter().enumerate() {
                if !used[index] {
                    differences.push(format!(
                        "unexpected Potter point attribute at {:?}: {:?}",
                        entry.co, entry.values
                    ));
                }
            }
        }
        (Some(actual), None) => {
            differences.push(format!("unexpected Potter point attributes {actual:?}"));
        }
        (None, Some(expected)) => {
            differences.push(format!("missing Blender point attributes {expected:?}"));
        }
        (None, None) => {}
    }
    differences
}

fn compare_edge_attributes(
    actual: &Option<Vec<EdgeAttribute>>,
    expected: &Option<Vec<EdgeAttribute>>,
) -> Vec<String> {
    let mut differences = Vec::new();
    match (actual, expected) {
        (Some(actual), Some(expected)) => {
            if actual.len() != expected.len() {
                differences.push(format!(
                    "edge attributes have {} Potter rows, Blender has {}",
                    actual.len(),
                    expected.len()
                ));
            }
            let mut used = vec![false; actual.len()];
            for expected_entry in expected {
                let matched = actual.iter().enumerate().find(|(index, actual_entry)| {
                    !used[*index]
                        && same_edge(actual_entry.edge, expected_entry.edge)
                        && attribute_values_match(&actual_entry.values, &expected_entry.values)
                });
                if let Some((index, _)) = matched {
                    used[index] = true;
                } else {
                    differences.push(format!(
                        "missing Blender edge attributes on {:?}: {:?}",
                        expected_entry.edge, expected_entry.values
                    ));
                }
            }
            for (index, entry) in actual.iter().enumerate() {
                if !used[index] {
                    differences.push(format!(
                        "unexpected Potter edge attributes on {:?}: {:?}",
                        entry.edge, entry.values
                    ));
                }
            }
        }
        (Some(actual), None) => {
            differences.push(format!("unexpected Potter edge attributes {actual:?}"));
        }
        (None, Some(expected)) => {
            differences.push(format!("missing Blender edge attributes {expected:?}"));
        }
        (None, None) => {}
    }
    differences
}

fn compare_vertex_groups(
    actual: &BTreeMap<String, Vec<GroupEntry>>,
    expected: &BTreeMap<String, Vec<GroupEntry>>,
) -> Vec<String> {
    let mut differences = Vec::new();
    if actual.keys().ne(expected.keys()) {
        differences.push(format!(
            "vertex-group names {:?} differ from Blender {:?}",
            actual.keys().collect::<Vec<_>>(),
            expected.keys().collect::<Vec<_>>()
        ));
    }
    for (name, expected_entries) in expected {
        let actual_entries = actual.get(name).map(Vec::as_slice).unwrap_or_default();
        if actual_entries.len() != expected_entries.len() {
            differences.push(format!(
                "vertex group `{name}` has {} Potter entries, Blender has {}",
                actual_entries.len(),
                expected_entries.len()
            ));
        }
        let mut used = vec![false; actual_entries.len()];
        for (expected_point, expected_weight) in expected_entries {
            let matched = actual_entries
                .iter()
                .enumerate()
                .find(|(index, (point, weight))| {
                    !used[*index]
                        && point_matches(*point, *expected_point)
                        && scalar_option_matches(*weight, *expected_weight, WEIGHT_TOLERANCE)
                });
            if let Some((index, _)) = matched {
                used[index] = true;
                continue;
            }
            let positional = actual_entries
                .iter()
                .enumerate()
                .find(|(index, (point, _))| {
                    !used[*index] && point_matches(*point, *expected_point)
                });
            if let Some((index, (_, actual_weight))) = positional {
                used[index] = true;
                differences.push(format!(
                    "vertex group `{name}` at {expected_point:?}: weight {actual_weight:?} differs from Blender {expected_weight:?}"
                ));
            } else {
                differences.push(format!(
                    "vertex group `{name}` missing Blender point {expected_point:?} with weight {expected_weight:?}"
                ));
            }
        }
        for (index, (point, weight)) in actual_entries.iter().enumerate() {
            if !used[index] {
                differences.push(format!(
                    "vertex group `{name}` has unexpected Potter point {point:?} with weight {weight:?}"
                ));
            }
        }
    }
    for name in actual.keys().filter(|name| !expected.contains_key(*name)) {
        differences.push(format!("unexpected Potter vertex group `{name}`"));
    }
    differences
}

/// Differences between Potter's canonical result and a Blender expectation. Empty when they match.
pub fn mismatches(actual: &Canonical, expected: &Canonical) -> Vec<String> {
    let mut differences = Vec::new();
    if actual.vertices != expected.vertices {
        differences.push(format!(
            "vertex count {} differs from Blender {}",
            actual.vertices, expected.vertices
        ));
    }
    if actual.edges != expected.edges {
        differences.push(format!(
            "edge count {} differs from Blender {}",
            actual.edges, expected.edges
        ));
    }
    if actual.face_count != expected.face_count {
        differences.push(format!(
            "face count {} differs from Blender {}",
            actual.face_count, expected.face_count
        ));
    }
    if actual.face_sizes != expected.face_sizes {
        differences.push(format!(
            "face sizes {:?} differ from Blender {:?}",
            actual.face_sizes, expected.face_sizes
        ));
    }
    match (&actual.faces, &expected.faces) {
        (Some(actual_faces), Some(expected_faces)) => {
            differences.extend(compare_faces(actual_faces, expected_faces));
        }
        (Some(_) | None, None) => {}
        (None, Some(_)) => differences.push("Potter face topology is missing".to_owned()),
    }
    match (&actual.edge_topology, &expected.edge_topology) {
        (Some(actual_edges), Some(expected_edges)) => {
            differences.extend(compare_edges(actual_edges, expected_edges));
        }
        (Some(_) | None, None) => {}
        (None, Some(_)) => differences.push("Potter edge topology is missing".to_owned()),
    }
    differences.extend(compare_edge_points(
        &actual.edge_points,
        &expected.edge_points,
    ));
    differences.extend(compare_point_attributes(
        &actual.point_attributes,
        &expected.point_attributes,
    ));
    differences.extend(compare_edge_attributes(
        &actual.edge_attributes,
        &expected.edge_attributes,
    ));
    match (&actual.uv_layers, &expected.uv_layers) {
        (_, None) => {}
        (Some(actual), Some(expected)) if actual == expected => {}
        (Some(actual), Some(expected)) => differences.push(format!(
            "UV layers {actual:?} differ from Blender {expected:?}"
        )),
        (None, Some(expected)) => {
            differences.push(format!("missing Blender UV layers {expected:?}"));
        }
    }
    match (&actual.color_layers, &expected.color_layers) {
        (Some(actual), Some(expected)) if actual == expected => {}
        (Some(actual), Some(expected)) => differences.push(format!(
            "color layers {actual:?} differ from Blender {expected:?}"
        )),
        (Some(actual), None) => {
            differences.push(format!("unexpected Potter color layers {actual:?}"));
        }
        (None, Some(expected)) => {
            differences.push(format!("missing Blender color layers {expected:?}"));
        }
        (None, None) => {}
    }
    match (&actual.object_materials, &expected.object_materials) {
        (_, None) => {}
        (Some(actual), Some(expected)) if actual == expected => {}
        (Some(actual), Some(expected)) => differences.push(format!(
            "object assignments {actual:?} differ from Blender {expected:?}"
        )),
        (None, Some(expected)) => differences.push(format!(
            "Potter object assignments are missing; Blender has {expected:?}"
        )),
    }
    match (&actual.effective_materials, &expected.effective_materials) {
        (_, None) => {}
        (Some(actual), Some(expected)) if actual == expected => {}
        (Some(actual), Some(expected)) => differences.push(format!(
            "effective material slots {actual:?} differ from Blender {expected:?}"
        )),
        (None, Some(expected)) => differences.push(format!(
            "Potter effective material slots are missing; Blender has {expected:?}"
        )),
    }
    match (&actual.data_materials, &expected.data_materials) {
        (_, None) => {}
        (Some(actual), Some(expected)) if actual == expected => {}
        (Some(actual), Some(expected)) => differences.push(format!(
            "data material slots {actual:?} differ from Blender {expected:?}"
        )),
        (None, Some(expected)) => differences.push(format!(
            "Potter data material slots are missing; Blender has {expected:?}"
        )),
    }
    match (&actual.material_links, &expected.material_links) {
        (_, None) => {}
        (Some(actual), Some(expected)) if actual == expected => {}
        (Some(actual), Some(expected)) => differences.push(format!(
            "material links {actual:?} differ from Blender {expected:?}"
        )),
        (None, Some(expected)) => differences.push(format!(
            "Potter material links are missing; Blender has {expected:?}"
        )),
    }
    differences.extend(compare_vertex_groups(
        &actual.vertex_groups,
        &expected.vertex_groups,
    ));
    if !volume_matches(actual.volume, expected.volume) {
        differences.push(format!(
            "volume {:.9} differs from Blender {:.9}",
            actual.volume, expected.volume
        ));
    }
    differences
}

/// Name of a material ID, or an error when an evaluated slot references a missing material.
fn material_name(document: &SceneDoc, node_id: &Id, material_id: &Id) -> TestResult<String> {
    document
        .materials
        .get(material_id)
        .map(|material| material.name.clone())
        .ok_or_else(|| {
            format!(
                "Potter material slot for `{}` references missing material `{}`",
                node_id.as_str(),
                material_id.as_str()
            )
            .into()
        })
}

/// Name of an optional material slot; `None` is an empty slot.
fn optional_material_name(
    document: &SceneDoc,
    node_id: &Id,
    material: Option<&Id>,
) -> TestResult<Option<String>> {
    material
        .map(|material_id| material_name(document, node_id, material_id))
        .transpose()
}

/// Evaluates `node_id` with its Boolean as the only enabled modifier, in world space.
pub fn evaluate_boolean(document: &SceneDoc, node_id: &Id) -> TestResult<Canonical> {
    let mut boolean_only = document.clone();
    for (id, node) in &mut boolean_only.nodes {
        for modifier in &mut node.modifiers {
            modifier.enabled = id == node_id && modifier.modifier_type == "boolean";
        }
    }
    let snapshot = Snapshot::evaluate(&boolean_only, &EvaluationContext::default())
        .map_err(|error| format!("{}: {error}", node_id.as_str()))?;
    let mesh = snapshot
        .meshes
        .get(node_id)
        .ok_or_else(|| format!("Potter did not evaluate `{}`", node_id.as_str()))?;
    let node = snapshot
        .nodes
        .get(node_id)
        .ok_or_else(|| format!("Potter has no evaluated node `{}`", node_id.as_str()))?;
    let model_node = boolean_only
        .nodes
        .get(node_id)
        .ok_or_else(|| format!("Potter has no model node `{}`", node_id.as_str()))?;
    let data = model_node
        .data
        .as_ref()
        .and_then(|data_id| boolean_only.data_blocks.get(data_id));
    let slot_count = node.materials.len();
    let object_materials = (0..slot_count)
        .map(|slot| {
            let material = match model_node.material_link(slot) {
                MaterialLink::Object => model_node.materials.get(slot).and_then(Option::as_ref),
                MaterialLink::Data => None,
            };
            optional_material_name(&boolean_only, node_id, material)
        })
        .collect::<TestResult<Vec<_>>>()?;
    let effective_materials = node
        .materials
        .iter()
        .map(|material| optional_material_name(&boolean_only, node_id, material.as_ref()))
        .collect::<TestResult<Vec<_>>>()?;
    let data_materials = data
        .map_or(&[][..], |data| data.materials.as_slice())
        .iter()
        .map(|material| optional_material_name(&boolean_only, node_id, material.as_ref()))
        .collect::<TestResult<Vec<_>>>()?;
    let material_links = (0..slot_count)
        .map(|slot| model_node.material_link(slot))
        .collect();
    potter_canonical(
        mesh,
        DMat4::from_cols_array(&node.world_matrix),
        object_materials,
        effective_materials,
        data_materials,
        material_links,
    )
}

/// Runs `pot` with `args` and returns its JSON output. A failing command fails the test.
pub fn run_pot(args: &[&str]) -> TestResult<Value> {
    let output = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(args)
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "pot {} failed: stdout={} stderr={}",
            args.first().copied().unwrap_or("command"),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(serde_json::from_slice(&output.stdout)?)
}

/// Runs a fixture generator headless: `blender --background --factory-startup --python script -- output`.
pub fn run_generator(blender: &Path, script: &Path, output: &Path) -> TestResult {
    let result = Command::new(blender)
        .args(["--background", "--factory-startup", "--python"])
        .arg(script)
        .arg("--")
        .arg(output)
        .output()?;
    if result.status.success() {
        return Ok(());
    }
    Err(format!(
        "Blender generator {} failed: stdout={} stderr={}",
        script.display(),
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    )
    .into())
}

/// Reads the scene document of a project.
pub fn read_scene(project: &Path) -> TestResult<SceneDoc> {
    Ok(serde_json::from_slice(&fs::read(
        project.join("scene.json"),
    )?)?)
}

/// Imports the Blender scene at `source` into a new project at `project` and returns its document.
/// The import must report no losses.
pub fn import_blend_scene(blender: &Path, source: &Path, project: &Path) -> TestResult<SceneDoc> {
    let project_path = project.to_str().ok_or("project path is not UTF-8")?;
    let source_path = source.to_str().ok_or("Blend path is not UTF-8")?;
    let blender_path = blender.to_str().ok_or("Blender path is not UTF-8")?;
    run_pot(&["init", project_path])?;
    let imported = run_pot(&[
        "import",
        project_path,
        "--file",
        source_path,
        "--format",
        "blend",
        "--mode",
        "replace",
        "--base-revision",
        "0",
        "--blender",
        blender_path,
    ])?;
    if imported["result"]["losses"] != serde_json::json!([]) {
        return Err(format!(
            "Blender import reported losses: {}",
            imported["result"]["losses"]
        )
        .into());
    }
    read_scene(project)
}

/// The ID of the node named `name`.
pub fn find_node(document: &SceneDoc, name: &str) -> TestResult<Id> {
    document
        .nodes
        .iter()
        .find(|(_, node)| node.name == name)
        .map(|(id, _)| id.clone())
        .ok_or_else(|| format!("Blender host `{name}` is missing from the imported scene").into())
}

#[cfg(test)]
mod uv_layer_tests {
    use super::{UvLayerFace, uv_layers_match_rotation};
    use std::collections::BTreeMap;

    fn layer(uv: [[f64; 2]; 3], pins: [bool; 3]) -> UvLayerFace {
        UvLayerFace {
            uv: uv.to_vec(),
            pins: pins.to_vec(),
        }
    }

    #[test]
    fn named_layer_values_and_pins_follow_face_ring_rotation() {
        let actual = BTreeMap::from([
            (
                "Base".to_owned(),
                layer([[0.0, 0.0], [0.5, 0.25], [0.75, 0.5]], [true, false, false]),
            ),
            (
                "Detail".to_owned(),
                layer([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], [false, true, false]),
            ),
        ]);
        let expected = BTreeMap::from([
            (
                "Base".to_owned(),
                layer([[0.5, 0.25], [0.75, 0.5], [0.0, 0.0]], [false, false, true]),
            ),
            (
                "Detail".to_owned(),
                layer([[0.3, 0.4], [0.5, 0.6], [0.1, 0.2]], [true, false, false]),
            ),
        ]);
        assert!(uv_layers_match_rotation(&actual, &expected, 3, 1));

        let mut wrong_pins = expected.clone();
        assert!(wrong_pins.get_mut("Base").is_some_and(|layer| {
            layer.pins[0] = true;
            true
        }));
        assert!(!uv_layers_match_rotation(&actual, &wrong_pins, 3, 1));

        let mut wrong_uv = expected.clone();
        assert!(wrong_uv.get_mut("Detail").is_some_and(|layer| {
            layer.uv[0][0] = 0.9;
            true
        }));
        assert!(!uv_layers_match_rotation(&actual, &wrong_uv, 3, 1));

        let mut wrong_names = expected;
        wrong_names.remove("Detail");
        assert!(!uv_layers_match_rotation(&actual, &wrong_names, 3, 1));
    }
}

#[cfg(test)]
mod edge_topology_tests {
    use super::{
        AttributeValues, EdgeAttribute, EdgePoint, EdgePointEndpoint, compare_edge_attributes,
        compare_edge_points, compare_edges,
    };
    use std::collections::BTreeMap;

    fn values(name: &str) -> AttributeValues {
        BTreeMap::from([(
            "source".to_owned(),
            serde_json::Value::String(name.to_owned()),
        )])
    }

    fn edge_point(edge: [[f64; 3]; 2], first: &str, second: &str) -> EdgePoint {
        EdgePoint {
            edge,
            endpoints: [
                EdgePointEndpoint {
                    co: edge[0],
                    values: Some(values(first)),
                },
                EdgePointEndpoint {
                    co: edge[1],
                    values: Some(values(second)),
                },
            ],
        }
    }

    #[test]
    fn coincident_edges_match_as_a_multiset() {
        let edge = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        assert_eq!(
            compare_edges(&[edge, edge], &[edge, edge]),
            Vec::<String>::new()
        );

        let missing_duplicate = compare_edges(&[edge], &[edge, edge]);
        assert!(
            missing_duplicate
                .iter()
                .any(|difference| difference.contains("no matching Potter edge"))
        );

        let extra_duplicate = compare_edges(&[edge, edge], &[edge]);
        assert!(
            extra_duplicate
                .iter()
                .any(|difference| difference.contains("unexpected Potter edge"))
        );
    }

    #[test]
    fn coincident_edge_payloads_keep_their_associations() {
        let edge = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let actual_attributes = vec![
            EdgeAttribute {
                edge,
                values: values("first"),
            },
            EdgeAttribute {
                edge,
                values: values("second"),
            },
        ];
        let expected_attributes = vec![actual_attributes[1].clone(), actual_attributes[0].clone()];
        assert_eq!(
            compare_edge_attributes(&Some(actual_attributes), &Some(expected_attributes)),
            Vec::<String>::new()
        );

        let actual_points = vec![
            edge_point(edge, "left", "right"),
            edge_point(edge, "up", "down"),
        ];
        let expected_points = vec![
            edge_point(edge, "up", "down"),
            edge_point(edge, "left", "right"),
        ];
        assert_eq!(
            compare_edge_points(&Some(actual_points), &Some(expected_points)),
            Vec::<String>::new()
        );

        let misassociated_points = vec![
            edge_point(edge, "right", "left"),
            edge_point(edge, "up", "down"),
        ];
        let expected_associations = vec![
            edge_point(edge, "left", "right"),
            edge_point(edge, "up", "down"),
        ];
        assert_ne!(
            compare_edge_points(&Some(misassociated_points), &Some(expected_associations)),
            Vec::<String>::new()
        );
    }
}
