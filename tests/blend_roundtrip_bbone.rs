#![recursion_limit = "512"]
#![expect(
    clippy::unwrap_used,
    reason = "Blender fixtures use fixed identifiers, paths, and generated test values"
)]

use std::{
    fs,
    path::Path,
    process::{Command, Output},
    sync::Mutex,
};

use potter::model::SceneDoc;
use serde_json::{Value, json};
use tempfile::tempdir;
#[path = "common/blender_checked.rs"]
mod blender_checked;
mod common;
#[path = "common/process.rs"]
mod process;

use blender_checked::blender_executable as blender;
use common::TestResult;
use process::run_guarded;

static BLENDER_BBONE_TEST_LOCK: Mutex<()> = Mutex::new(());

const BLENDER_FIXTURE: &str = r#"
import bpy
import os
import sys

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
case = sys.argv[sys.argv.index("--") + 2]
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

armature = bpy.data.armatures.new("BboneRigData")
rig = bpy.data.objects.new("Rig", armature)
scene.collection.objects.link(rig)
rig.select_set(True)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="EDIT")
root_bone = armature.edit_bones.new("Root")
root_bone.head = (0.0, -1.0, 0.0)
root_bone.tail = (0.0, 0.0, 0.0)
bendy = armature.edit_bones.new("Bendy")
bendy.head = (0.0, 0.0, 0.0)
bendy.tail = (0.0, 2.0, 0.0)
bendy.parent = root_bone
start_handle = armature.edit_bones.new("StartHandle")
start_handle.head = (-0.6, 0.2, 0.0)
start_handle.tail = (-0.6, 0.7, 0.0)
end_handle = armature.edit_bones.new("EndHandle")
end_handle.head = (0.6, 1.3, 0.0)
end_handle.tail = (0.6, 1.8, 0.0)
bpy.ops.object.mode_set(mode="OBJECT")
rig.select_set(False)

bone = armature.bones["Bendy"]
bone.bbone_segments = 5
bone.bbone_mapping_mode = "CURVED"
bone.bbone_x = 0.17
bone.bbone_z = 0.23
bone.bbone_handle_type_start = "TANGENT"
bone.bbone_custom_handle_start = armature.bones["StartHandle"]
bone.bbone_handle_use_scale_start = (True, False, True)
bone.bbone_handle_use_ease_start = True
bone.bbone_handle_type_end = "ABSOLUTE"
bone.bbone_custom_handle_end = armature.bones["EndHandle"]
bone.bbone_handle_use_scale_end = (False, True, False)
bone.bbone_handle_use_ease_end = False
bone.bbone_rollin = 0.13
bone.bbone_rollout = -0.21
bone.bbone_curveinx = 0.11
bone.bbone_curveinz = 0.24
bone.bbone_curveoutx = -0.14
bone.bbone_curveoutz = -0.27
bone.bbone_easein = 0.31
bone.bbone_easeout = 0.42
bone.bbone_scalein = (1.1, 0.8, 1.2)
bone.bbone_scaleout = (0.9, 1.3, 0.7)

if case == "target_shape":
    owner = bpy.data.objects.new("CaseOwner", None)
    scene.collection.objects.link(owner)
    constraint = owner.constraints.new("COPY_LOCATION")
    constraint.name = "BboneShapeTarget"
    constraint.target = rig
    constraint.subtarget = "Bendy"
    constraint.head_tail = 0.35
    constraint.use_bbone_shape = True
elif case == "armature_modifier":
    mesh = bpy.data.meshes.new("CaseOwnerData")
    mesh.from_pydata(
        [(-0.5, 0.0, 0.0), (0.5, 0.0, 0.0), (-0.5, 1.0, 0.0), (0.5, 1.0, 0.0)],
        [],
        [(0, 1, 3, 2)],
    )
    mesh.update()
    owner = bpy.data.objects.new("CaseOwner", mesh)
    scene.collection.objects.link(owner)
    group = owner.vertex_groups.new(name="Bendy")
    group.add(list(range(len(mesh.vertices))), 1.0, "REPLACE")
    modifier = owner.modifiers.new("WeightedBbone", "ARMATURE")
    modifier.object = rig
    modifier.use_vertex_groups = True
    modifier.use_bone_envelopes = False
elif case == "armature_constraint":
    owner = bpy.data.objects.new("CaseOwner", None)
    scene.collection.objects.link(owner)
    constraint = owner.constraints.new("ARMATURE")
    constraint.name = "WeightedBboneTarget"
    target = constraint.targets.new()
    target.target = rig
    target.subtarget = "Bendy"
    target.weight = 0.75
else:
    raise ValueError("unknown fixture case: " + case)

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, "source.blend"), check_existing=False)
"#;

const BLENDER_REOPEN: &str = r#"
import bpy
import json
import sys

output_path = sys.argv[sys.argv.index("--") + 1]
case = sys.argv[sys.argv.index("--") + 2]
rig = bpy.data.objects["Rig"]
bone = rig.data.bones["Bendy"]
keys = [
    "bbone_segments",
    "bbone_mapping_mode",
    "bbone_x",
    "bbone_z",
    "bbone_handle_type_start",
    "bbone_custom_handle_start",
    "bbone_handle_use_scale_start",
    "bbone_handle_use_ease_start",
    "bbone_handle_type_end",
    "bbone_custom_handle_end",
    "bbone_handle_use_scale_end",
    "bbone_handle_use_ease_end",
    "bbone_rollin",
    "bbone_rollout",
    "bbone_curveinx",
    "bbone_curveinz",
    "bbone_curveoutx",
    "bbone_curveoutz",
    "bbone_easein",
    "bbone_easeout",
    "bbone_scalein",
    "bbone_scaleout",
]
settings = {}
for key in keys:
    value = getattr(bone, key)
    if key in ("bbone_custom_handle_start", "bbone_custom_handle_end"):
        value = value.name if value else None
    elif key in ("bbone_handle_use_scale_start", "bbone_handle_use_scale_end", "bbone_scalein", "bbone_scaleout"):
        value = list(value)
    settings[key] = value
result = {"settings": settings}
if case == "target_shape":
    constraint = bpy.data.objects["CaseOwner"].constraints["BboneShapeTarget"]
    result["constraint"] = {
        "use_bbone_shape": constraint.use_bbone_shape,
        "head_tail": constraint.head_tail,
        "target": constraint.target.name if constraint.target else None,
        "subtarget": constraint.subtarget,
    }
with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(result, handle)
"#;

fn expected_settings() -> Value {
    json!({
        "bbone_segments": 5,
        "bbone_mapping_mode": "CURVED",
        "bbone_x": 0.17,
        "bbone_z": 0.23,
        "bbone_handle_type_start": "TANGENT",
        "bbone_custom_handle_start": "StartHandle",
        "bbone_handle_use_scale_start": [true, false, true],
        "bbone_handle_use_ease_start": true,
        "bbone_handle_type_end": "ABSOLUTE",
        "bbone_custom_handle_end": "EndHandle",
        "bbone_handle_use_scale_end": [false, true, false],
        "bbone_handle_use_ease_end": false,
        "bbone_rollin": 0.13,
        "bbone_rollout": -0.21,
        "bbone_curveinx": 0.11,
        "bbone_curveinz": 0.24,
        "bbone_curveoutx": -0.14,
        "bbone_curveoutz": -0.27,
        "bbone_easein": 0.31,
        "bbone_easeout": 0.42,
        "bbone_scalein": [1.1, 0.8, 1.2],
        "bbone_scaleout": [0.9, 1.3, 0.7],
    })
}
fn assert_json_approximately_equal(actual: &Value, expected: &Value, context: &str) {
    match (actual, expected) {
        (Value::Number(actual), Value::Number(expected)) => {
            let actual = actual.as_f64().unwrap();
            let expected = expected.as_f64().unwrap();
            assert!(
                (actual - expected).abs() <= 1.0e-6,
                "{context}: {actual} != {expected}"
            );
        }
        (Value::Array(actual), Value::Array(expected)) => {
            assert_eq!(actual.len(), expected.len(), "{context}: array length");
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                assert_json_approximately_equal(actual, expected, &format!("{context}[{index}]"));
            }
        }
        (Value::Object(actual), Value::Object(expected)) => {
            assert_eq!(actual.len(), expected.len(), "{context}: object keys");
            for (key, expected) in expected {
                assert_json_approximately_equal(
                    actual.get(key).unwrap(),
                    expected,
                    &format!("{context}.{key}"),
                );
            }
        }
        _ => assert_eq!(actual, expected, "{context}"),
    }
}

fn run_blender(blender: &Path, root: &Path, case: &str) -> TestResult {
    let script_path = root.join("make_fixture.py");
    fs::write(&script_path, BLENDER_FIXTURE)?;
    let root = fs::canonicalize(root)?;
    let mut command = Command::new(blender);
    command
        .args(["--background", "--factory-startup", "--python"])
        .arg(script_path)
        .arg("--")
        .arg(root)
        .arg(case);
    assert_success(&run_guarded(command)?, "Blender fixture")
}

fn run_reopen(blender: &Path, root: &Path, exported: &Path, case: &str) -> TestResult<Value> {
    let script_path = root.join("reopen.py");
    let output_path = root.join("reopened.json");
    fs::write(&script_path, BLENDER_REOPEN)?;
    let mut command = Command::new(blender);
    command
        .arg("--background")
        .arg(exported)
        .arg("--python")
        .arg(script_path)
        .arg("--")
        .arg(&output_path)
        .arg(case);
    assert_success(&run_guarded(command)?, "Blender reopen")?;
    Ok(serde_json::from_slice(&fs::read(output_path)?)?)
}

fn assert_success(output: &Output, operation: &str) -> TestResult {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() || stdout.contains("Traceback") || stderr.contains("Traceback") {
        return Err(format!("{operation} failed: stdout={stdout} stderr={stderr}").into());
    }
    Ok(())
}

fn pot_output(arguments: &[&str]) -> TestResult<Output> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_pot"));
    command.args(arguments).arg("--json");
    run_guarded(command)
}

fn pot_json(arguments: &[&str]) -> TestResult<Value> {
    let output = pot_output(arguments)?;
    assert_success(&output, "pot command")?;
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn assert_imported_bbone_settings(doc: &SceneDoc) {
    let rig = doc.nodes.values().find(|node| node.name == "Rig").unwrap();
    let armature = doc.data_blocks[rig.data.as_ref().unwrap()]
        .armature
        .as_ref()
        .unwrap();
    let bendy = armature
        .bones
        .values()
        .find(|bone| bone.name == "Bendy")
        .unwrap();
    let expected = expected_settings();
    for (key, value) in expected.as_object().unwrap() {
        assert_json_approximately_equal(
            bendy.bbone_settings.get(key).unwrap(),
            value,
            &format!("model setting {key}"),
        );
    }
    for name in ["Root", "StartHandle", "EndHandle"] {
        let ordinary = armature
            .bones
            .values()
            .find(|bone| bone.name == name)
            .unwrap();
        let segments = ordinary
            .bbone_settings
            .get("bbone_segments")
            .and_then(Value::as_u64)
            .unwrap_or(1);
        assert_eq!(segments, 1, "ordinary bone {name} must remain unsegmented");
    }
}

fn run_case(case: &str, expected_feature_id: &str) -> TestResult {
    let _guard = BLENDER_BBONE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(blender) = blender() else {
        eprintln!("skipping B-Bone integration test: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    run_blender(&blender, root, case)?;

    let project = root.join("project");
    pot_json(&["init", project.to_str().unwrap()])?;
    let source = root.join("source.blend");
    let imported = pot_json(&[
        "import",
        project.to_str().unwrap(),
        "--file",
        source.to_str().unwrap(),
        "--format",
        "blend",
        "--mode",
        "replace",
        "--base-revision",
        "0",
        "--blender",
        blender.to_str().unwrap(),
    ])?;
    assert!(
        imported["result"]["losses"].as_array().unwrap().is_empty(),
        "B-Bone fixture import reported losses: {imported}"
    );
    let owner_id = imported["result"]["id_mappings"]["Object:CaseOwner"]
        .as_str()
        .unwrap();
    let doc: SceneDoc = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    assert_imported_bbone_settings(&doc);

    let evaluation = pot_output(&["inspect", project.to_str().unwrap(), "--id", owner_id])?;
    assert!(
        !evaluation.status.success(),
        "{case}: evaluation unexpectedly accepted unsupported B-Bone behavior"
    );
    let response: Value = serde_json::from_slice(&evaluation.stdout)?;
    assert_eq!(
        response["error"]["code"], "UNSUPPORTED_FEATURE",
        "{case}: {response}"
    );
    assert_eq!(
        response["error"]["details"]["feature_id"], expected_feature_id,
        "{case}: {response}"
    );

    // Export/reopen only inspects source RNA and deliberately does not evaluate the gated path.
    let exported = root.join("roundtrip.blend");
    pot_json(&[
        "export",
        project.to_str().unwrap(),
        "--format",
        "blend",
        "--out",
        exported.to_str().unwrap(),
        "--blender",
        blender.to_str().unwrap(),
    ])?;
    let reopened = run_reopen(&blender, root, &exported, case)?;
    assert_json_approximately_equal(
        &reopened["settings"],
        &expected_settings(),
        &format!("{case}: reopened B-Bone RNA"),
    );
    if case == "target_shape" {
        assert_json_approximately_equal(
            &reopened["constraint"],
            &json!({
                "use_bbone_shape": true,
                "head_tail": 0.35,
                "target": "Rig",
                "subtarget": "Bendy",
            }),
            "shape-target constraint settings",
        );
    }
    Ok(())
}

#[test]
fn bbone_shape_target_gate_and_blend_round_trip() -> TestResult {
    run_case("target_shape", "constraint.target_bbone_shape")
}

#[test]
fn armature_modifier_bbone_gate_and_blend_round_trip() -> TestResult {
    run_case("armature_modifier", "rig.bbone.segments")
}

#[test]
fn armature_constraint_bbone_gate_and_blend_round_trip() -> TestResult {
    run_case("armature_constraint", "rig.bbone.segments")
}
