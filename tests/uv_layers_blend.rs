#![expect(clippy::expect_used, reason = "integration test fixtures")]

use std::{
    error::Error,
    fs, io,
    path::{Path, PathBuf},
    process::{Command, Output},
};

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/process.rs"]
mod process;

use blender_checked::blender_executable;

use serde_json::{Value, json};
use tempfile::tempdir;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

/// Creation order of the layers. Active (index 2) and render (index 1) differ from the defaults.
const LAYER_NAMES: [&str; 3] = ["Base", "Light", "Detail"];
const ACTIVE_INDEX: u64 = 2;
const RENDER_INDEX: u64 = 1;
const TOLERANCE: f64 = 1.0e-6;

/// Saves a cube with three UV layers, each with distinct per-loop values, to the path after `--`.
const BUILD_SCRIPT: &str = r#"
import sys

import bpy

out = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cube_add()
mesh = bpy.context.object.data
for layer_index, name in enumerate(["Base", "Light", "Detail"]):
    if layer_index == 0:
        layer = mesh.uv_layers[0]
        layer.name = name
    else:
        layer = mesh.uv_layers.new(name=name)
    for loop_index, item in enumerate(layer.data):
        item.uv = (layer_index * 0.25 + loop_index * 0.01, 0.1 * (layer_index + 1) + loop_index * 0.01)
        if layer_index == 1:
            item.pin_uv = loop_index % 2 == 0
mesh.uv_layers.active_index = 2
mesh.uv_layers.active_render_index = 1
bpy.ops.wm.save_as_mainfile(filepath=out)
"#;

/// Writes the UV layers of the only mesh that has any, as JSON, to the path after `--`.
const DUMP_SCRIPT: &str = r#"
import json
import sys

import bpy

out = sys.argv[sys.argv.index("--") + 1]
meshes = [mesh for mesh in bpy.data.meshes if len(mesh.uv_layers)]
if len(meshes) != 1:
    raise SystemExit("expected one mesh with UV layers, found %d" % len(meshes))
mesh = meshes[0]
result = {
    "layers": [
        {"name": layer.name, "values": [[float(item.uv[0]), float(item.uv[1])] for item in layer.data], "pins": [bool(item.pin_uv) for item in layer.data]}
        for layer in mesh.uv_layers
    ],
    "active": mesh.uv_layers.active_index,
    "render": mesh.uv_layers.active_render_index,
}
with open(out, "w", encoding="utf-8") as handle:
    json.dump(result, handle)
"#;

fn checked_output(command: &mut Command, label: &str) -> TestResult<Output> {
    let output = command.output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "{label} failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        ))
        .into());
    }
    Ok(output)
}

/// Runs `pot` with the arguments added by `configure` and parses its `--json` output.
fn run_pot(configure: impl FnOnce(&mut Command)) -> TestResult<Value> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_pot"));
    configure(&mut command);
    command.arg("--json");
    let output = checked_output(&mut command, "pot")?;
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn write_script(root: &Path, name: &str, source: &str) -> TestResult<PathBuf> {
    let path = root.join(name);
    fs::write(&path, source)?;
    Ok(path)
}

/// Runs Blender in the background. `file` is opened first when given; `argument` follows `--`.
fn run_blender(blender: &Path, script: &Path, file: Option<&Path>, argument: &Path) -> TestResult {
    let mut command = Command::new(blender);
    command.args([
        "--background",
        "--factory-startup",
        "--disable-autoexec",
        "--python-exit-code",
        "3",
    ]);
    if let Some(file) = file {
        command.arg(file);
    }
    command.arg("--python").arg(script).arg("--").arg(argument);
    checked_output(&mut command, "Blender")?;
    Ok(())
}

/// Layer names in order.
fn layer_names(document: &Value) -> Vec<String> {
    document["layers"]
        .as_array()
        .expect("layers are an array")
        .iter()
        .map(|layer| {
            layer["name"]
                .as_str()
                .expect("layer name is a string")
                .to_owned()
        })
        .collect()
}

/// Blender-convention corner values of one dumped layer, in loop order.
fn blender_values(layer: &Value) -> TestResult<Vec<[f64; 2]>> {
    layer["values"]
        .as_array()
        .ok_or("dumped layer values are missing")?
        .iter()
        .map(pair)
        .collect()
}

fn pair(value: &Value) -> TestResult<[f64; 2]> {
    let values = value.as_array().ok_or("UV value is not an array")?;
    let (Some(u), Some(v)) = (
        values.first().and_then(Value::as_f64),
        values.get(1).and_then(Value::as_f64),
    ) else {
        return Err("UV value is not a pair of numbers".into());
    };
    Ok([u, v])
}

/// The mesh JSON of the imported data-block that stores UV layers.
fn mesh_with_uv_layers(scene: &Value) -> TestResult<&Value> {
    let mesh = scene["data_blocks"]
        .as_object()
        .ok_or("scene data blocks are not an object")?
        .values()
        .filter_map(|block| block.get("mesh"))
        .find(|mesh| mesh["attributes"].get("uv_layers").is_some())
        .ok_or("imported mesh has no uv_layers store")?;
    Ok(mesh)
}

/// Stored corner values of one layer, flattened in the mesh's face order (storage convention).
fn stored_corners(mesh: &Value, layer: &Value) -> TestResult<Vec<[f64; 2]>> {
    let entries = layer["faces"]
        .as_array()
        .ok_or("stored layer has no face entries")?;
    let mut corners = Vec::new();
    for face in mesh["faces"].as_array().ok_or("mesh has no faces")? {
        let id = face["id"].as_u64().ok_or("face ID is missing")?;
        let entry = entries
            .iter()
            .find(|entry| entry["face_id"].as_u64() == Some(id))
            .ok_or("stored layer has no entry for a face")?;
        for corner in entry["uv"].as_array().ok_or("stored face has no UVs")? {
            corners.push(pair(corner)?);
        }
    }
    Ok(corners)
}

/// Stored pin flags of one layer, flattened in the mesh's face order. A face without a `pinned`
/// list has no pinned corner.
fn stored_pins(mesh: &Value, layer: &Value) -> TestResult<Vec<bool>> {
    let entries = layer["faces"]
        .as_array()
        .ok_or("stored layer has no face entries")?;
    let mut pins = Vec::new();
    for face in mesh["faces"].as_array().ok_or("mesh has no faces")? {
        let id = face["id"].as_u64().ok_or("face ID is missing")?;
        let entry = entries
            .iter()
            .find(|entry| entry["face_id"].as_u64() == Some(id))
            .ok_or("stored layer has no entry for a face")?;
        match entry.get("pinned").and_then(Value::as_array) {
            Some(flags) => {
                for flag in flags {
                    pins.push(flag.as_bool().ok_or("stored pin is not a bool")?);
                }
            }
            None => {
                let corners = entry["uv"].as_array().map_or(0, Vec::len);
                pins.extend(std::iter::repeat_n(false, corners));
            }
        }
    }
    Ok(pins)
}

/// Blender per-loop pin flags of one dumped layer, in loop order.
fn blender_pins(layer: &Value) -> TestResult<Vec<bool>> {
    let mut pins = Vec::new();
    for pin in layer["pins"]
        .as_array()
        .ok_or("dumped layer pins are missing")?
    {
        pins.push(pin.as_bool().ok_or("dumped pin is not a bool")?);
    }
    Ok(pins)
}

fn assert_close(actual: f64, expected: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= TOLERANCE,
        "{label}: {actual} differs from {expected}"
    );
}

#[test]
fn three_uv_layers_round_trip_through_pot_and_blender() -> TestResult {
    let Some(blender) = blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let build = write_script(root.path(), "build.py", BUILD_SCRIPT)?;
    let dump = write_script(root.path(), "dump.py", DUMP_SCRIPT)?;

    let source_blend = root.path().join("source.blend");
    run_blender(&blender, &build, None, &source_blend)?;
    let source_json = root.path().join("source.json");
    run_blender(&blender, &dump, Some(source_blend.as_path()), &source_json)?;
    let source: Value = serde_json::from_slice(&fs::read(&source_json)?)?;
    assert_eq!(layer_names(&source), LAYER_NAMES);
    assert_eq!(source["active"], json!(ACTIVE_INDEX));
    assert_eq!(source["render"], json!(RENDER_INDEX));
    // The fixture pins every even loop of `Light` (layer 1) on the 24-loop cube.
    assert_eq!(
        blender_pins(&source["layers"][1])?,
        (0..24).map(|index| index % 2 == 0).collect::<Vec<bool>>()
    );

    let project = root.path().join("project");
    run_pot(|command| {
        command.arg("init").arg(&project);
    })?;
    let imported = run_pot(|command| {
        command
            .arg("import")
            .arg(&project)
            .arg("--file")
            .arg(&source_blend)
            .args([
                "--format",
                "blend",
                "--mode",
                "replace",
                "--base-revision",
                "0",
                "--blender",
            ])
            .arg(&blender);
    })?;
    assert_eq!(imported["result"]["losses"], json!([]));

    // The store holds storage values (v down): each stored v is 1 minus the Blender v.
    let scene: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let mesh = mesh_with_uv_layers(&scene)?;
    let store = &mesh["attributes"]["uv_layers"];
    assert_eq!(layer_names(store), LAYER_NAMES);
    assert_eq!(store["active"], json!("Detail"));
    assert_eq!(store["render"], json!("Light"));
    let stored_layers = store["layers"]
        .as_array()
        .ok_or("stored layers are missing")?;
    let source_layers = source["layers"]
        .as_array()
        .ok_or("dumped layers are missing")?;
    assert_eq!(stored_layers.len(), source_layers.len());
    for (stored_layer, source_layer) in stored_layers.iter().zip(source_layers) {
        let name = source_layer["name"].as_str().unwrap_or_default();
        let stored = stored_corners(mesh, stored_layer)?;
        let source_corners = blender_values(source_layer)?;
        assert_eq!(stored.len(), source_corners.len(), "{name}: corner count");
        for (loop_index, (stored, source)) in stored.iter().zip(&source_corners).enumerate() {
            assert_close(stored[0], source[0], &format!("{name} loop {loop_index} u"));
            assert_close(
                stored[1],
                1.0 - source[1],
                &format!("{name} loop {loop_index} storage v"),
            );
        }
        assert_eq!(
            stored_pins(mesh, stored_layer)?,
            blender_pins(source_layer)?,
            "{name}: stored pins"
        );
    }

    let exported_blend = root.path().join("round-trip.blend");
    let exported = run_pot(|command| {
        command
            .arg("export")
            .arg(&project)
            .args(["--format", "blend", "--out"])
            .arg(&exported_blend)
            .arg("--blender")
            .arg(&blender);
    })?;
    assert_eq!(exported["result"]["losses"], json!([]));

    let reopened_json = root.path().join("reopened.json");
    run_blender(
        &blender,
        &dump,
        Some(exported_blend.as_path()),
        &reopened_json,
    )?;
    let reopened: Value = serde_json::from_slice(&fs::read(&reopened_json)?)?;
    assert_eq!(layer_names(&reopened), layer_names(&source));
    assert_eq!(reopened["active"], source["active"]);
    assert_eq!(reopened["render"], source["render"]);
    let reopened_layers = reopened["layers"]
        .as_array()
        .ok_or("reopened layers are missing")?;
    for (reopened_layer, source_layer) in reopened_layers.iter().zip(source_layers) {
        let name = source_layer["name"].as_str().unwrap_or_default();
        let actual = blender_values(reopened_layer)?;
        let expected = blender_values(source_layer)?;
        assert_eq!(actual.len(), expected.len(), "{name}: loop count");
        for (loop_index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert_close(
                actual[0],
                expected[0],
                &format!("{name} loop {loop_index} u"),
            );
            assert_close(
                actual[1],
                expected[1],
                &format!("{name} loop {loop_index} v"),
            );
        }
        assert_eq!(
            blender_pins(reopened_layer)?,
            blender_pins(source_layer)?,
            "{name}: pins after export"
        );
    }
    Ok(())
}
