#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/native_oracle.rs"]
pub mod native_oracle;
#[path = "common/process.rs"]
mod process;

use std::fs;

use native_oracle::{
    TestResult, apply_operations, export_raw, import_blend, pot_json, read_scene, run_blender,
    write_script,
};
use serde_json::{Value, json};
use tempfile::tempdir;

const BUILD_SCRIPT: &str = r#"
import bpy
import os
import sys

root = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
for name, value in (("SourceEnabled", True), ("SourceDisabled", False)):
    material = bpy.data.materials.new(name)
    material.use_fake_user = True
    material.use_nodes = True
    material.use_raytrace_refraction = value
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, "source.blend"))
"#;

const DUMP_SCRIPT: &str = r#"
import bpy
import json
import sys

output = sys.argv[sys.argv.index("--") + 1]
with open(output, "w", encoding="utf-8") as stream:
    json.dump({material.name_full: getattr(material, "use_raytrace_refraction", None)
               for material in bpy.data.materials}, stream, sort_keys=True)
"#;

fn material_id(scene: &Value, name: &str) -> TestResult<String> {
    scene["materials"]
        .as_object()
        .ok_or("scene materials are not an object")?
        .iter()
        .find(|(_, material)| material["name"].as_str() == Some(name))
        .map(|(id, _)| id.clone())
        .ok_or_else(|| format!("no material named {name}").into())
}

fn has_raytrace_loss(report: &Value, id: &str) -> bool {
    let losses = report["result"]["losses"]
        .as_array()
        .or_else(|| report["error"]["details"]["losses"].as_array());
    losses.is_some_and(|losses| {
        losses.iter().any(|loss| {
            loss["feature_id"].as_str() == Some("blender.material.use_raytrace_refraction")
                && loss["data_id"].as_str() == Some(id)
        })
    })
}

#[test]
fn material_raytrace_refraction_import_export_and_unknown_loss() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let build = write_script(root.path(), "build_materials.py", BUILD_SCRIPT)?;
    run_blender(&blender, &build, None, root.path())?;
    let source = root.path().join("source.blend");
    let (project, _) = import_blend(&blender, root.path(), "project", &source)?;

    let scene = read_scene(&project)?;
    for (name, value) in [("SourceEnabled", true), ("SourceDisabled", false)] {
        let id = material_id(&scene, name)?;
        assert_eq!(
            scene["materials"][id.as_str()]["use_raytrace_refraction"],
            json!(value)
        );
        let inspected = pot_json(|command| {
            command.arg("inspect").arg(&project).arg("--id").arg(&id);
        })?;
        assert_eq!(
            inspected["result"]["items"][0]["use_raytrace_refraction"],
            json!(value)
        );
    }

    apply_operations(
        &project,
        root.path(),
        json!([
            {"op":"material.create","id":"created_default","name":"CreatedDefault"},
            {"op":"material.create","id":"created_enabled","name":"CreatedEnabled","use_raytrace_refraction":true}
        ]),
    )?;
    let scene = read_scene(&project)?;
    assert_eq!(
        scene["materials"]["created_default"]["use_raytrace_refraction"],
        json!(false)
    );
    assert_eq!(
        scene["materials"]["created_enabled"]["use_raytrace_refraction"],
        json!(true)
    );

    let exported = root.path().join("roundtrip.blend");
    let export = export_raw(&blender, &project, "blend", &exported, true)?;
    assert!(
        export.status.success(),
        "lossy export failed: {}",
        String::from_utf8_lossy(&export.stdout)
    );
    let export: Value = serde_json::from_slice(&export.stdout)?;
    assert!(
        !export["result"]["losses"].as_array().is_some_and(|losses| {
            losses
                .iter()
                .any(|loss| loss["feature_id"] == "blender.material.use_raytrace_refraction")
        }),
        "known Blender settings must not be reported as losses: {export}"
    );
    let dump = write_script(root.path(), "dump_materials.py", DUMP_SCRIPT)?;
    let values_path = root.path().join("material_values.json");
    run_blender(&blender, &dump, Some(&exported), &values_path)?;
    let values: Value = serde_json::from_slice(&fs::read(values_path)?)?;
    for (name, value) in [
        ("SourceEnabled", true),
        ("SourceDisabled", false),
        ("CreatedDefault", false),
        ("CreatedEnabled", true),
    ] {
        assert_eq!(values[name], json!(value), "Blender readback for {name}");
    }

    apply_operations(
        &project,
        root.path(),
        json!([{
            "op":"material.update",
            "target":{"id":"created_default"},
            "set":{"use_raytrace_refraction":null}
        }]),
    )?;
    let inspected_unknown = pot_json(|command| {
        command
            .arg("inspect")
            .arg(&project)
            .arg("--id")
            .arg("created_default");
    })?;
    assert_eq!(
        inspected_unknown["result"]["items"][0]["use_raytrace_refraction"],
        Value::Null
    );
    let strict_output = root.path().join("strict_unknown.blend");
    let rejected = export_raw(&blender, &project, "blend", &strict_output, false)?;
    assert!(
        !rejected.status.success(),
        "strict export must reject an unknown setting"
    );
    let rejected: Value = serde_json::from_slice(&rejected.stdout)?;
    assert_eq!(rejected["error"]["code"], "UNREPRESENTABLE_FEATURE");
    assert!(
        has_raytrace_loss(&rejected, "created_default"),
        "strict export did not identify the unknown material setting: {rejected}"
    );

    let lossy_output = root.path().join("lossy_unknown.blend");
    let lossy = export_raw(&blender, &project, "blend", &lossy_output, true)?;
    assert!(
        lossy.status.success(),
        "lossy export failed: {}",
        String::from_utf8_lossy(&lossy.stdout)
    );
    let lossy: Value = serde_json::from_slice(&lossy.stdout)?;
    assert!(
        has_raytrace_loss(&lossy, "created_default"),
        "loss was not reported: {lossy}"
    );
    Ok(())
}
