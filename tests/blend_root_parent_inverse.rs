//! A Blender root with a non-identity `matrix_parent_inverse` evaluates as Blender does: Blender
//! ignores the inverse without a parent. Before and after an edit the world matches Blender, and the
//! inverse's bits round-trip through export.

use std::{
    error::Error,
    fs,
    process::{Command, Output},
};

#[path = "common/blender_file.rs"]
mod blender_file;
use blender_file::blender_executable;

use serde_json::{Value, json};
use tempfile::tempdir;

type TestResult<T> = Result<T, Box<dyn Error>>;

const BLENDER_REQUIRED: &str = "Blender is required for the root parent-inverse test: install Blender 5.2, or set POTTER_BLENDER to its executable";
const WORLD_TOLERANCE: f64 = 1.0e-5;

/// The root carries a shear-bearing inverse set directly: Blender clears `matrix_parent_inverse` when
/// `parent` is assigned None, so a root with a non-identity inverse comes from a file that has one.
const ROOT_FIXTURE: &str = r#"
import bpy
import json
import math
import os
import sys
from mathutils import Euler, Matrix

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cube_add(size=1.0)
obj = bpy.context.object
obj.name = "Root Inverse"
obj.location = (0.5, -1.0, 0.25)
obj.rotation_euler = (0.2, 0.1, math.radians(40))
inverse = Matrix.Translation((2.0, 3.0, 4.0)) @ Euler((0.0, 0.0, math.radians(25))).to_matrix().to_4x4()
inverse = inverse @ Matrix(((1.0, 0.25, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)))
obj.matrix_parent_inverse = inverse
bpy.context.view_layer.update()


def rows(matrix):
    return [list(row) for row in matrix]


expected = {"before": {"world": rows(obj.matrix_world), "parent_inverse": rows(obj.matrix_parent_inverse)}}
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, "source.blend"))
obj.location = (-3.0, 1.0, 0.0)
bpy.context.view_layer.update()
expected["after"] = {"world": rows(obj.matrix_world), "parent_inverse": rows(obj.matrix_parent_inverse)}
with open(os.path.join(root, "expected.json"), "w", encoding="utf-8") as output:
    json.dump(expected, output)
"#;

const ROUND_TRIP_DUMPER: &str = r#"
import bpy
import json
import sys

output_path = sys.argv[sys.argv.index("--") + 1]
bpy.context.view_layer.update()
obj = bpy.data.objects["Root Inverse"]
with open(output_path, "w", encoding="utf-8") as output:
    json.dump({
        "parent_inverse": [list(row) for row in obj.matrix_parent_inverse],
        "world": [list(row) for row in obj.matrix_world],
    }, output)
"#;

fn checked_output(mut command: Command, label: &str) -> TestResult<Output> {
    let output = command.output()?;
    if !output.status.success() {
        return Err(format!(
            "{label} failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        )
        .into());
    }
    Ok(output)
}

fn run_pot(args: &[&str]) -> TestResult<Value> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_pot"));
    command.args(args).arg("--json");
    let output = checked_output(command, args.first().copied().unwrap_or("pot"))?;
    Ok(serde_json::from_slice(&output.stdout)?)
}

/// Blender's row lists flattened column-major, the order Potter's matrices use.
fn column_major(rows: &Value) -> TestResult<Vec<f64>> {
    let rows = rows.as_array().ok_or("matrix rows are not an array")?;
    if rows.len() != 4 {
        return Err("matrix does not have four rows".into());
    }
    let mut values = Vec::with_capacity(16);
    for column in 0..4 {
        for row in rows {
            values.push(
                row.as_array()
                    .and_then(|elements| elements.get(column))
                    .and_then(Value::as_f64)
                    .ok_or("matrix element is malformed")?,
            );
        }
    }
    Ok(values)
}

fn assert_world_close(actual: &[f64], expected: &[f64], context: &str) {
    assert_eq!(actual.len(), 16, "{context}: world matrix has 16 elements");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= WORLD_TOLERANCE,
            "{context}: element {index} is {actual}, Blender has {expected}"
        );
    }
}

fn node_id(document: &Value, name: &str) -> TestResult<String> {
    document["nodes"]
        .as_object()
        .and_then(|nodes| {
            nodes
                .iter()
                .find(|(_, node)| node["name"].as_str() == Some(name))
                .map(|(id, _)| id.clone())
        })
        .ok_or_else(|| format!("imported node `{name}` is missing").into())
}

fn world_matrix(project: &str, id: &str) -> TestResult<Vec<f64>> {
    let inspected = run_pot(&["inspect", project, "--id", id, "--frame", "1"])?;
    let item = inspected["result"]["items"]
        .as_array()
        .and_then(|items| items.iter().find(|item| item["id"].as_str() == Some(id)))
        .ok_or("inspect returned no item for the root")?;
    item["transform"]["world"]["matrix"]
        .as_array()
        .ok_or("world matrix is missing")?
        .iter()
        .map(|value| value.as_f64().ok_or_else(|| "world element is not numeric".into()))
        .collect()
}

#[test]
fn blender_root_parent_inverse_evaluates_as_blender_and_round_trips() -> TestResult<()> {
    let blender = blender_executable().ok_or(BLENDER_REQUIRED)?;
    let directory = tempdir()?;
    let root = directory.path();
    let fixture = root.join("root_parent_inverse.py");
    fs::write(&fixture, ROOT_FIXTURE)?;
    let canonical_root = fs::canonicalize(root)?;
    let mut build = Command::new(&blender);
    build
        .args(["--background", "--factory-startup", "--python"])
        .arg(&fixture)
        .arg("--")
        .arg(&canonical_root);
    checked_output(build, "building root parent-inverse fixture")?;
    let expected: Value = serde_json::from_slice(&fs::read(root.join("expected.json"))?)?;
    let source = root.join("source.blend");
    let source_text = source.to_str().ok_or("source path is not UTF-8")?;
    let blender_text = blender.to_str().ok_or("Blender path is not UTF-8")?;
    let project = root.join("project");
    let project_text = project.to_str().ok_or("project path is not UTF-8")?;

    run_pot(&["init", project_text])?;
    let imported = run_pot(&[
        "import",
        project_text,
        "--file",
        source_text,
        "--format",
        "blend",
        "--mode",
        "replace",
        "--base-revision",
        "0",
        "--blender",
        blender_text,
    ])?;
    assert_eq!(imported["result"]["losses"], json!([]));
    let document: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let id = node_id(&document, "Root Inverse")?;
    let node = &document["nodes"][&id];
    assert!(
        node["parent_inverse"].is_null(),
        "a Blender root's inverse must not be applied by evaluation"
    );
    assert_eq!(
        node["blender_parent_inverse"].as_array().map(|values| {
            values
                .iter()
                .filter_map(Value::as_f64)
                .collect::<Vec<f64>>()
        }),
        Some(column_major(&expected["before"]["parent_inverse"])?),
        "the Blender inverse must be kept bit for bit as round-trip data"
    );

    let before = column_major(&expected["before"]["world"])?;
    assert_world_close(&world_matrix(project_text, &id)?, &before, "import, before edit");

    let batch = root.join("edit.json");
    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": document["revision"],
            "operations": [{
                "op": "node.update",
                "target": {"id": id},
                "set": {"transform": {"translation": [-3.0, 1.0, 0.0]}}
            }]
        }))?,
    )?;
    let batch_text = batch.to_str().ok_or("batch path is not UTF-8")?;
    run_pot(&["apply", project_text, "--file", batch_text])?;
    let after = column_major(&expected["after"]["world"])?;
    assert_world_close(&world_matrix(project_text, &id)?, &after, "after the same edit");

    let exported_path = root.join("round-trip.blend");
    let exported_text = exported_path
        .to_str()
        .ok_or("export path is not UTF-8")?;
    let exported = run_pot(&[
        "export",
        project_text,
        "--format",
        "blend",
        "--out",
        exported_text,
        "--blender",
        blender_text,
    ])?;
    assert_eq!(exported["result"]["losses"], json!([]));

    let dumper = root.join("dump_root.py");
    fs::write(&dumper, ROUND_TRIP_DUMPER)?;
    let dump_path = root.join("dump.json");
    let mut reopen = Command::new(&blender);
    reopen
        .args(["--background", "--disable-autoexec"])
        .arg(&exported_path)
        .args(["--python"])
        .arg(&dumper)
        .args(["--"])
        .arg(&dump_path);
    checked_output(reopen, "reopening the exported root in Blender")?;
    let dumped: Value = serde_json::from_slice(&fs::read(&dump_path)?)?;
    assert_eq!(
        column_major(&dumped["parent_inverse"])?,
        column_major(&expected["before"]["parent_inverse"])?,
        "the exported parent inverse must keep Blender's bits"
    );
    assert_world_close(
        &column_major(&dumped["world"])?,
        &after,
        "Blender world of the exported edited root",
    );
    Ok(())
}
