//! Each exchange format reads back the UV layers it writes: names, order, corner values, and the
//! active and render layers where the format carries them. Sources are hand-built files whose
//! values are written out in the test, so the expected numbers do not come from Potter.

use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use serde_json::{Value, json};
use tempfile::tempdir;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

/// Float32 storage (glTF, PLY, GLB) and the 6-digit OBJ text both stay well inside this bound.
const TOLERANCE: f64 = 1.0e-6;

/// Triangle corners of `UVMap` (TEXCOORD_0), in glTF convention (v down).
const SET0: [[f32; 2]; 3] = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]];
/// Triangle corners of `UVMap.001` (TEXCOORD_1), in glTF convention (v down).
const SET1: [[f32; 2]; 3] = [[0.7, 0.8], [0.9, 0.1], [0.2, 0.3]];
const POSITIONS: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

/// Node that holds the triangle. glTF import names the node by its `name`.
const NODE: &str = "tri";

fn widen(values: &[[f32; 2]]) -> Vec<[f64; 2]> {
    values
        .iter()
        .map(|&[u, v]| [f64::from(u), f64::from(v)])
        .collect()
}

fn pot_json(configure: impl FnOnce(&mut Command)) -> TestResult<Value> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_pot"));
    configure(&mut command);
    command.arg("--json");
    let output = command.output()?;
    if !output.status.success() {
        return Err(format!(
            "pot failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(serde_json::from_slice(&output.stdout)?)
}

/// Creates an empty project at `root/name`.
fn init(root: &Path, name: &str) -> TestResult<PathBuf> {
    let project = root.join(name);
    pot_json(|command| {
        command.arg("init").arg(&project);
    })?;
    Ok(project)
}

/// Imports `file` into `project`, which must be empty. Losses are allowed so the tests can look at
/// UV losses alone; the import report is returned.
fn import(project: &Path, file: &Path, format: &str) -> TestResult<Value> {
    pot_json(|command| {
        command
            .arg("import")
            .arg(project)
            .arg("--file")
            .arg(file)
            .args([
                "--format",
                format,
                "--base-revision",
                "0",
                "--mode",
                "replace",
                "--allow-lossy",
            ]);
    })
}

/// Creates a project at `root/name` and imports `file` into it.
fn import_new(root: &Path, name: &str, file: &Path, format: &str) -> TestResult<PathBuf> {
    let project = init(root, name)?;
    import(&project, file, format)?;
    Ok(project)
}

/// Exports `project` to `out`. Without `allow_lossy`, any loss makes the export fail.
fn export(project: &Path, format: &str, out: &Path, allow_lossy: bool) -> TestResult<Value> {
    pot_json(|command| {
        command
            .arg("export")
            .arg(project)
            .args(["--format", format, "--out"])
            .arg(out)
            .arg("--overwrite");
        if allow_lossy {
            command.arg("--allow-lossy");
        }
    })
}

/// Makes `layer` the active UV layer of the node `node`, as a `uv.layer_set_active` operation.
fn set_active_layer(
    root: &Path,
    project: &Path,
    base_revision: u64,
    node: &str,
    layer: &str,
) -> TestResult {
    let file = root.join("operations.json");
    fs::write(
        &file,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": base_revision,
            "operations": [
                {"op": "uv.layer_set_active", "target": {"id": node}, "name": layer}
            ],
        }))?,
    )?;
    pot_json(|command| {
        command.arg("apply").arg(project).arg("--file").arg(&file);
    })?;
    Ok(())
}

/// Fails when an export or import report carries a UV layer loss.
fn assert_no_uv_losses(report: &Value) -> TestResult {
    let losses = report["result"]["losses"]
        .as_array()
        .ok_or("report has no losses array")?;
    let uv = losses
        .iter()
        .filter(|loss| loss["feature_id"] == "mesh.uv_layer")
        .collect::<Vec<_>>();
    assert!(uv.is_empty(), "unexpected UV losses: {uv:?}");
    Ok(())
}

/// One stored corner `[u, v]` from a `uv` array.
fn pair(value: &Value) -> TestResult<[f64; 2]> {
    let values = value.as_array().ok_or("UV value is not an array")?;
    match (
        values.first().and_then(Value::as_f64),
        values.get(1).and_then(Value::as_f64),
    ) {
        (Some(u), Some(v)) => Ok([u, v]),
        _ => Err("UV value is not a pair of numbers".into()),
    }
}

/// The imported data-block mesh that stores UV layers.
fn mesh_of(document: &Value) -> TestResult<&Value> {
    let mesh = document["data_blocks"]
        .as_object()
        .ok_or("scene data blocks are not an object")?
        .values()
        .filter_map(|block| block.get("mesh"))
        .find(|mesh| mesh["attributes"].get("uv_layers").is_some())
        .ok_or("imported mesh has no uv_layers store")?;
    Ok(mesh)
}

/// The UV store of a project's mesh, read back from `scene.json`.
struct Stored {
    /// Layer names in store order, each with its corner values flattened in face order, in the
    /// store's storage convention (v down).
    layers: Vec<(String, Vec<[f64; 2]>)>,
    active: Option<String>,
    render: Option<String>,
}

fn stored(project: &Path) -> TestResult<Stored> {
    let document: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let mesh = mesh_of(&document)?;
    let store = &mesh["attributes"]["uv_layers"];
    let mut layers = Vec::new();
    for layer in store["layers"].as_array().ok_or("store has no layers")? {
        let name = layer["name"]
            .as_str()
            .ok_or("layer has no name")?
            .to_owned();
        let mut corners = Vec::new();
        for face in mesh["faces"].as_array().ok_or("mesh has no faces")? {
            let id = face["id"].as_u64().ok_or("face has no id")?;
            let entry = layer["faces"]
                .as_array()
                .ok_or("layer has no face entries")?
                .iter()
                .find(|entry| entry["face_id"].as_u64() == Some(id))
                .ok_or("layer has no entry for a face")?;
            for corner in entry["uv"].as_array().ok_or("face entry has no uv")? {
                corners.push(pair(corner)?);
            }
        }
        layers.push((name, corners));
    }
    Ok(Stored {
        layers,
        active: store["active"].as_str().map(str::to_owned),
        render: store["render"].as_str().map(str::to_owned),
    })
}

fn layer_names(stored: &Stored) -> Vec<&str> {
    stored
        .layers
        .iter()
        .map(|(name, _)| name.as_str())
        .collect()
}

/// Asserts that layer `name` holds `expected`, corner by corner.
fn assert_layer(stored: &Stored, name: &str, expected: &[[f64; 2]]) -> TestResult {
    let (_, actual) = stored
        .layers
        .iter()
        .find(|(layer, _)| layer.as_str() == name)
        .ok_or_else(|| format!("layer `{name}` is missing"))?;
    assert_eq!(actual.len(), expected.len(), "{name}: corner count");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual[0] - expected[0]).abs() <= TOLERANCE
                && (actual[1] - expected[1]).abs() <= TOLERANCE,
            "{name} corner {index}: stored {actual:?}, expected {expected:?}"
        );
    }
    Ok(())
}

/// Little-endian bytes of `rows` of float32 values.
fn push_floats<const N: usize>(bytes: &mut Vec<u8>, rows: &[[f32; N]]) {
    for row in rows {
        for value in row {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
}

/// The triangle buffer: indices (3 x u16) at 0, positions at 8, TEXCOORD_0 at 44, TEXCOORD_1 at 68.
fn triangle_buffer() -> Vec<u8> {
    let mut bytes = Vec::new();
    for index in [0_u16, 1, 2] {
        bytes.extend_from_slice(&index.to_le_bytes());
    }
    bytes.extend_from_slice(&[0, 0]);
    push_floats(&mut bytes, POSITIONS.as_slice());
    push_floats(&mut bytes, SET0.as_slice());
    push_floats(&mut bytes, SET1.as_slice());
    bytes
}

/// glTF document of one triangle with two TEXCOORD sets. `buffer_uri` names an external buffer;
/// without it the document expects a GLB binary chunk.
fn triangle_document(buffer_uri: Option<&str>) -> Value {
    let mut buffer = json!({"byteLength": 92});
    if let Some(uri) = buffer_uri {
        buffer["uri"] = json!(uri);
    }
    json!({
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": NODE, "mesh": 0}],
        "meshes": [{
            "name": NODE,
            "primitives": [{
                "attributes": {"POSITION": 1, "TEXCOORD_0": 2, "TEXCOORD_1": 3},
                "indices": 0
            }]
        }],
        "accessors": [
            {"bufferView": 0, "componentType": 5123, "count": 3, "type": "SCALAR"},
            {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3",
             "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0]},
            {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC2"},
            {"bufferView": 3, "componentType": 5126, "count": 3, "type": "VEC2"}
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 6},
            {"buffer": 0, "byteOffset": 8, "byteLength": 36},
            {"buffer": 0, "byteOffset": 44, "byteLength": 24},
            {"buffer": 0, "byteOffset": 68, "byteLength": 24}
        ],
        "buffers": [buffer]
    })
}

/// Packs a glTF document and its binary chunk into a GLB container.
fn glb_bytes(document: &Value, binary: &[u8]) -> TestResult<Vec<u8>> {
    let mut json_chunk = serde_json::to_vec(document)?;
    while json_chunk.len() % 4 != 0 {
        json_chunk.push(b' ');
    }
    let mut bin_chunk = binary.to_vec();
    while bin_chunk.len() % 4 != 0 {
        bin_chunk.push(0);
    }
    let total = 12 + 8 + json_chunk.len() + 8 + bin_chunk.len();
    let mut bytes = Vec::with_capacity(total);
    bytes.extend_from_slice(b"glTF");
    bytes.extend_from_slice(&2_u32.to_le_bytes());
    bytes.extend_from_slice(&u32::try_from(total)?.to_le_bytes());
    bytes.extend_from_slice(&u32::try_from(json_chunk.len())?.to_le_bytes());
    bytes.extend_from_slice(b"JSON");
    bytes.extend_from_slice(&json_chunk);
    bytes.extend_from_slice(&u32::try_from(bin_chunk.len())?.to_le_bytes());
    bytes.extend_from_slice(b"BIN\0");
    bytes.extend_from_slice(&bin_chunk);
    Ok(bytes)
}

/// A project with the triangle's two layers, `UVMap` and `UVMap.001`, imported from glTF, with
/// `UVMap.001` active and `UVMap` as the render layer. Returns the project.
fn two_layer_project(root: &Path) -> TestResult<PathBuf> {
    let source = root.join("tri.gltf");
    fs::write(root.join("tri.bin"), triangle_buffer())?;
    fs::write(
        &source,
        serde_json::to_vec(&triangle_document(Some("tri.bin")))?,
    )?;
    let project = init(root, "source")?;
    let imported = import(&project, &source, "gltf")?;
    let revision = imported["result"]["candidate_revision"]
        .as_u64()
        .ok_or("import report has no candidate revision")?;
    set_active_layer(root, &project, revision, NODE, "UVMap.001")?;
    Ok(project)
}

/// The two-layer project's stored layers, checked before any format sees them.
fn assert_two_layer_store(project: &Path) -> TestResult {
    let read = stored(project)?;
    assert_eq!(layer_names(&read), ["UVMap", "UVMap.001"]);
    assert_eq!(read.active.as_deref(), Some("UVMap.001"));
    assert_eq!(read.render.as_deref(), Some("UVMap"));
    assert_layer(&read, "UVMap", &widen(&SET0))?;
    assert_layer(&read, "UVMap.001", &widen(&SET1))
}

#[test]
fn gltf_reads_each_texcoord_set_as_a_named_layer_in_order() -> TestResult {
    let root = tempdir()?;
    let source = root.path().join("tri.gltf");
    fs::write(root.path().join("tri.bin"), triangle_buffer())?;
    fs::write(
        &source,
        serde_json::to_vec(&triangle_document(Some("tri.bin")))?,
    )?;
    let project = import_new(root.path(), "imported", &source, "gltf")?;
    let read = stored(&project)?;
    // glTF has no active or render flag: both fall on the first set.
    assert_eq!(layer_names(&read), ["UVMap", "UVMap.001"]);
    assert_eq!(read.active.as_deref(), Some("UVMap"));
    assert_eq!(read.render.as_deref(), Some("UVMap"));
    assert_layer(&read, "UVMap", &widen(&SET0))?;
    assert_layer(&read, "UVMap.001", &widen(&SET1))
}

#[test]
fn gltf_round_trip_keeps_both_layers() -> TestResult {
    let root = tempdir()?;
    let project = two_layer_project(root.path())?;
    assert_two_layer_store(&project)?;
    let out = root.path().join("out.gltf");
    assert_no_uv_losses(&export(&project, "gltf", &out, false)?)?;
    let reread = import_new(root.path(), "reread", &out, "gltf")?;
    let read = stored(&reread)?;
    assert_eq!(layer_names(&read), ["UVMap", "UVMap.001"]);
    assert_layer(&read, "UVMap", &widen(&SET0))?;
    assert_layer(&read, "UVMap.001", &widen(&SET1))
}

#[test]
fn glb_round_trip_keeps_both_layers() -> TestResult {
    let root = tempdir()?;
    let project = two_layer_project(root.path())?;
    let out = root.path().join("out.glb");
    assert_no_uv_losses(&export(&project, "glb", &out, false)?)?;
    let reread = import_new(root.path(), "reread", &out, "glb")?;
    let read = stored(&reread)?;
    assert_eq!(layer_names(&read), ["UVMap", "UVMap.001"]);
    assert_layer(&read, "UVMap", &widen(&SET0))?;
    assert_layer(&read, "UVMap.001", &widen(&SET1))
}

#[test]
fn vrm_import_reads_each_texcoord_set_as_a_named_layer() -> TestResult {
    let root = tempdir()?;
    let mut document = triangle_document(None);
    document["extensionsUsed"] = json!(["VRMC_vrm"]);
    document["extensions"] = json!({
        "VRMC_vrm": {
            "specVersion": "1.0",
            "meta": {
                "name": NODE,
                "version": "1",
                "authors": ["potter-test"],
                "licenseUrl": "https://vrm.dev/licenses/1.0/"
            },
            "humanoid": {"humanBones": {}}
        }
    });
    let binary = triangle_buffer();
    let file = root.path().join("tri.vrm");
    fs::write(&file, glb_bytes(&document, &binary)?)?;
    let project = import_new(root.path(), "imported", &file, "vrm")?;
    let read = stored(&project)?;
    assert_eq!(layer_names(&read), ["UVMap", "UVMap.001"]);
    assert_layer(&read, "UVMap", &widen(&SET0))?;
    assert_layer(&read, "UVMap.001", &widen(&SET1))
}

#[test]
fn obj_writes_and_reads_the_active_layer_in_blender_v() -> TestResult {
    let root = tempdir()?;
    let project = two_layer_project(root.path())?;
    let out = root.path().join("out.obj");
    let report = export(&project, "obj", &out, true)?;
    // OBJ carries one layer. The inactive `UVMap` is reported as lost.
    let losses = report["result"]["losses"]
        .as_array()
        .ok_or("export report has no losses array")?;
    assert!(
        losses.iter().any(|loss| {
            loss["feature_id"] == "mesh.uv_layer"
                && loss["reason"]
                    .as_str()
                    .is_some_and(|reason| reason.contains("`UVMap`"))
        }),
        "inactive layer loss missing: {losses:?}"
    );
    let reread = import_new(root.path(), "reread", &out, "obj")?;
    let read = stored(&reread)?;
    assert_eq!(layer_names(&read), ["UVMap"]);
    assert_layer(&read, "UVMap", &widen(&SET1))
}

#[test]
fn ply_writes_and_reads_the_active_layer_in_blender_v() -> TestResult {
    let root = tempdir()?;
    let project = two_layer_project(root.path())?;
    let out = root.path().join("out.ply");
    let report = export(&project, "ply", &out, true)?;
    let losses = report["result"]["losses"]
        .as_array()
        .ok_or("export report has no losses array")?;
    assert!(
        losses.iter().any(|loss| {
            loss["feature_id"] == "mesh.uv_layer"
                && loss["reason"]
                    .as_str()
                    .is_some_and(|reason| reason.contains("`UVMap`"))
        }),
        "inactive layer loss missing: {losses:?}"
    );
    let reread = import_new(root.path(), "reread", &out, "ply")?;
    let read = stored(&reread)?;
    assert_eq!(layer_names(&read), ["UVMap"]);
    assert_layer(&read, "UVMap", &widen(&SET1))
}

#[test]
fn fbx_round_trip_keeps_every_layer_with_active_and_render_selection() -> TestResult {
    let root = tempdir()?;
    let project = two_layer_project(root.path())?;
    let out = root.path().join("out.fbx");
    assert_no_uv_losses(&export(&project, "fbx", &out, true)?)?;
    let reread = import_new(root.path(), "reread", &out, "fbx")?;
    let read = stored(&reread)?;
    assert_eq!(layer_names(&read), ["UVMap", "UVMap.001"]);
    assert_eq!(read.active.as_deref(), Some("UVMap.001"));
    assert_eq!(read.render.as_deref(), Some("UVMap"));
    assert_layer(&read, "UVMap", &widen(&SET0))?;
    assert_layer(&read, "UVMap.001", &widen(&SET1))
}

#[test]
fn obj_reads_a_single_value_vt_as_u_with_v_zero() -> TestResult {
    let root = tempdir()?;
    let file = root.path().join("single.obj");
    // `vt 0.5` has no v: OBJ defaults it to 0. Blender's V-up 0 is storage V 1.
    fs::write(
        &file,
        "o tri\nv 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0.5\nvt 0.25 0.75\nf 1/1 2/2 3/1\n",
    )?;
    let project = import_new(root.path(), "imported", &file, "obj")?;
    let read = stored(&project)?;
    assert_layer(&read, "UVMap", &[[0.5, 1.0], [0.25, 0.25], [0.5, 1.0]])
}

/// Writes a binary PLY triangle whose vertices carry UV properties named `u_name` and `v_name`.
fn write_uv_ply(path: &Path, u_name: &str, v_name: &str) -> TestResult {
    let header = format!(
        "ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nproperty float {u_name}\nproperty float {v_name}\nelement face 1\nproperty list uint uint vertex_indices\nend_header\n"
    );
    let mut bytes = header.into_bytes();
    for (position, uv) in POSITIONS.iter().zip(SET0) {
        push_floats(&mut bytes, &[*position]);
        push_floats(&mut bytes, &[uv]);
    }
    bytes.extend_from_slice(&3_u32.to_le_bytes());
    for index in [0_u32, 1, 2] {
        bytes.extend_from_slice(&index.to_le_bytes());
    }
    fs::write(path, bytes)?;
    Ok(())
}

#[test]
fn ply_reads_uv_properties_under_each_accepted_name_in_blender_v() -> TestResult {
    let root = tempdir()?;
    // The file's v points up. Storage v is 1 minus that value.
    let expected = widen(&SET0)
        .iter()
        .map(|[u, v]| [*u, 1.0 - *v])
        .collect::<Vec<_>>();
    for (label, u_name, v_name) in [
        ("s-t", "s", "t"),
        ("u-v", "u", "v"),
        ("texture", "texture_u", "texture_v"),
    ] {
        let file = root.path().join(format!("{label}.ply"));
        write_uv_ply(&file, u_name, v_name)?;
        let project = import_new(root.path(), label, &file, "ply")?;
        let read = stored(&project)?;
        assert_eq!(layer_names(&read), ["UVMap"], "{label}");
        assert_layer(&read, "UVMap", &expected)?;
    }
    Ok(())
}
