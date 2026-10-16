//! Canonical comparator for the Blender-backed EXACT Boolean tests.
//!
//! `tests/blend_boolean_parity.rs` and `tests/blend_boolean_regressions.rs` include this file with
//! `#[path]`. A Blender expectation is canonical: each face ring is in world space, rotated to start
//! at its minimum vertex, and the face list is sorted. Potter's result is reduced to the same form.
//! Coordinates and per-corner values are quantized to 1e-6, so a difference at that resolution is a
//! mismatch. The signed volume is compared with relative tolerance 1e-6, or with absolute tolerance
//! 1e-9 when the expected volume is zero.
//!
//! Per face: the ring, the material index, `fid`, UV corners, and custom-normal corners. Per vertex:
//! the vertex-group weights. Per mesh: vertex and edge counts, the face-size histogram, and volume.
//! A face field is compared exactly when the expectation records it, and Potter must then also
//! carry it. A field the expectation omits must be absent from Potter's result.

use std::{
    collections::{BTreeMap, HashMap},
    error::Error,
    fs,
    path::Path,
    process::Command,
};

use glam::{DMat4, DVec3};
use potter_core::{
    eval::{EvaluationContext, Snapshot},
    geom::Mesh,
    model::{Id, SceneDoc},
};
use serde_json::Value;

pub type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const COORDINATE_SCALE: f64 = 1.0e6;
const VOLUME_RELATIVE_TOLERANCE: f64 = 1.0e-6;
const VOLUME_ABSOLUTE_TOLERANCE: f64 = 1.0e-9;

/// One face in canonical form. The per-corner lists rotate with the ring.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct CanonicalFace {
    ring: Vec<[i64; 3]>,
    material: u64,
    fid: Option<i64>,
    uv: Option<Vec<[i64; 2]>>,
    custom_normals: Option<Vec<[i64; 3]>>,
}

/// A vertex-group entry: the quantized position, and the quantized weight (`None` outside the group).
type GroupEntry = ([i64; 3], Option<i64>);

/// One Boolean result in canonical form. `faces` is `None` when the expectation records counts,
/// face sizes, and volume only.
#[derive(Debug, PartialEq)]
pub struct Canonical {
    vertices: usize,
    edges: usize,
    face_sizes: BTreeMap<usize, usize>,
    faces: Option<Vec<CanonicalFace>>,
    vertex_groups: BTreeMap<String, Vec<GroupEntry>>,
    volume: f64,
}

fn quantize(value: f64) -> i64 {
    (value * COORDINATE_SCALE).round() as i64
}

fn quantize_point(point: DVec3) -> [i64; 3] {
    point.to_array().map(quantize)
}

fn rotated<T: Clone>(values: &[T], start: usize) -> Vec<T> {
    let mut rotated = values.to_vec();
    rotated.rotate_left(start);
    rotated
}

/// Canonical face: the ring starts at its minimum quantized vertex, and every per-corner list
/// rotates with it.
fn canonical_face(
    ring: &[DVec3],
    material: u64,
    fid: Option<i64>,
    uv: Option<&[[f64; 2]]>,
    normals: Option<&[DVec3]>,
) -> CanonicalFace {
    let quantized: Vec<[i64; 3]> = ring.iter().copied().map(quantize_point).collect();
    let start = quantized
        .iter()
        .enumerate()
        .min_by_key(|&(_, point)| *point)
        .map_or(0, |(index, _)| index);
    CanonicalFace {
        ring: rotated(&quantized, start),
        material,
        fid,
        uv: uv.map(|values| {
            let corners: Vec<[i64; 2]> = values.iter().map(|&pair| pair.map(quantize)).collect();
            rotated(&corners, start)
        }),
        custom_normals: normals.map(|values| {
            let corners: Vec<[i64; 3]> = values.iter().copied().map(quantize_point).collect();
            rotated(&corners, start)
        }),
    }
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

fn count(value: &Value, name: &str) -> TestResult<usize> {
    let count = value
        .as_u64()
        .ok_or_else(|| format!("Blender {name} count is invalid"))?;
    Ok(usize::try_from(count)?)
}

/// The per-face values of a FACE `blender_attributes` entry, in `mesh.faces` order.
fn face_values<'a>(mesh: &'a Mesh, name: &str) -> Option<&'a [Value]> {
    let entries = mesh.attributes.get("blender_attributes")?.as_array()?;
    entries
        .iter()
        .find(|entry| {
            entry.get("name").and_then(Value::as_str) == Some(name)
                && entry.get("domain").and_then(Value::as_str) == Some("FACE")
        })?
        .get("values")?
        .as_array()
        .map(Vec::as_slice)
}

/// Per-face UV corners of layer `UVMap`, in `mesh.faces` order. Read as Potter's UV reader does: the
/// `blender_uv_layers` `UVMap` values split by corner count, then each `uv_map` row of that layer
/// (`face_id`, `layer` defaulting to `UVMap`) replaces its face. A face without a value is `None`.
fn uv_map_faces(mesh: &Mesh) -> TestResult<Vec<Option<Vec<[f64; 2]>>>> {
    let mut faces: Vec<Option<Vec<[f64; 2]>>> = vec![None; mesh.faces.len()];
    let base = mesh
        .attributes
        .get("blender_uv_layers")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .find(|layer| layer.get("name").and_then(Value::as_str) == Some("UVMap"));
    if let Some(layer) = base {
        let values = uv_pairs(layer.get("values").unwrap_or(&Value::Null))?;
        let corners: usize = mesh.faces.iter().map(|face| face.vertices.len()).sum();
        if values.len() == corners {
            let mut next = values.into_iter();
            for (index, face) in mesh.faces.iter().enumerate() {
                faces[index] = Some(next.by_ref().take(face.vertices.len()).collect());
            }
        }
    }
    let face_index: HashMap<u64, usize> = mesh
        .faces
        .iter()
        .enumerate()
        .map(|(index, face)| (u64::from(face.id), index))
        .collect();
    for entry in mesh
        .attributes
        .get("uv_map")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let layer = entry.get("layer").and_then(Value::as_str).unwrap_or("UVMap");
        let Some(&index) = entry
            .get("face_id")
            .and_then(Value::as_u64)
            .and_then(|id| face_index.get(&id))
        else {
            continue;
        };
        if layer != "UVMap" {
            continue;
        }
        if let Some(row) = entry.get("uv").filter(|value| value.is_array()) {
            let uv = uv_pairs(row)?;
            if uv.len() == mesh.faces[index].vertices.len() {
                faces[index] = Some(uv);
            }
        }
    }
    Ok(faces)
}

/// Canonical form of an evaluated mesh. `world` maps the mesh's local space to world space.
fn potter_canonical(mesh: &Mesh, world: DMat4) -> TestResult<Canonical> {
    let positions: HashMap<u32, DVec3> = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, world.transform_point3(vertex.co)))
        .collect();
    let fids = face_values(mesh, "fid");
    let uv_faces = uv_map_faces(mesh)?;
    let normal_rows = mesh
        .attributes
        .get("custom_normal")
        .and_then(|attribute| attribute.get("values"))
        .and_then(Value::as_object);
    let mut faces = Vec::with_capacity(mesh.faces.len());
    let mut rings = Vec::with_capacity(mesh.faces.len());
    let mut face_sizes = BTreeMap::new();
    for (index, face) in mesh.faces.iter().enumerate() {
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
        let fid = fids
            .and_then(|values| values.get(index))
            .and_then(Value::as_i64);
        let uv = uv_faces[index].as_deref();
        let normals = normal_rows
            .and_then(|rows| rows.get(&format!("f{}", face.id)))
            .map(points_from_json)
            .transpose()?;
        faces.push(canonical_face(
            &ring,
            u64::from(face.material_index),
            fid,
            uv,
            normals.as_deref(),
        ));
        *face_sizes.entry(face.vertices.len()).or_insert(0) += 1;
        rings.push(ring);
    }
    faces.sort();

    let mut vertex_groups = BTreeMap::new();
    if let Some(groups) = mesh
        .attributes
        .get("vertex_groups")
        .and_then(Value::as_object)
    {
        for (name, weights) in groups {
            let mut entries: Vec<GroupEntry> = mesh
                .vertices
                .iter()
                .map(|vertex| {
                    let weight = weights
                        .get(format!("v{}", vertex.id).as_str())
                        .and_then(Value::as_f64)
                        .map(quantize);
                    (quantize_point(world.transform_point3(vertex.co)), weight)
                })
                .collect();
            entries.sort();
            vertex_groups.insert(name.clone(), entries);
        }
    }

    Ok(Canonical {
        vertices: mesh.vertices.len(),
        edges: mesh.edges.len(),
        face_sizes,
        faces: Some(faces),
        vertex_groups,
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
        let normals = face
            .get("custom_normals")
            .map(points_from_json)
            .transpose()?;
        faces.push(canonical_face(
            &ring,
            material,
            fid,
            uv.as_deref(),
            normals.as_deref(),
        ));
    }
    faces.sort();
    Ok(faces)
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
            let co = quantize_point(point_from_json(&entry["co"])?);
            let weight = match entry.get("weight") {
                Some(weight) if !weight.is_null() => Some(quantize(number(weight)?)),
                _ => None,
            };
            list.push((co, weight));
        }
        list.sort();
        groups.insert(name.clone(), list);
    }
    Ok(groups)
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
    Ok(Canonical {
        vertices: count(&expected["counts"]["vertices"], "vertex")?,
        edges: count(&expected["counts"]["edges"], "edge")?,
        face_sizes: expected_face_sizes(&expected["face_sizes"])?,
        faces,
        vertex_groups,
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
    if actual.face_sizes != expected.face_sizes {
        differences.push(format!(
            "face sizes {:?} differ from Blender {:?}",
            actual.face_sizes, expected.face_sizes
        ));
    }
    if let (Some(actual_faces), Some(expected_faces)) = (&actual.faces, &expected.faces) {
        if actual_faces.len() != expected_faces.len() {
            differences.push(format!(
                "face count {} differs from Blender {}",
                actual_faces.len(),
                expected_faces.len()
            ));
        } else if let Some(index) = actual_faces
            .iter()
            .zip(expected_faces)
            .position(|(left, right)| left != right)
        {
            differences.push(format!(
                "face {index} differs: actual {:?}, Blender {:?}",
                actual_faces[index], expected_faces[index]
            ));
        }
    }
    if actual.vertex_groups != expected.vertex_groups {
        differences.push(format!(
            "vertex groups differ: actual {:?}, Blender {:?}",
            actual.vertex_groups, expected.vertex_groups
        ));
    }
    if !volume_matches(actual.volume, expected.volume) {
        differences.push(format!(
            "volume {:.9} differs from Blender {:.9}",
            actual.volume, expected.volume
        ));
    }
    differences
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
    potter_canonical(mesh, DMat4::from_cols_array(&node.world_matrix))
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
