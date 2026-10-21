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

use potter_core::model::SceneDoc;
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
import json
import os
import sys

arguments = sys.argv[sys.argv.index("--") + 1:]
root = os.path.realpath(arguments[0])
case = arguments[1]
units = arguments[2:] or [case]
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
geometry = {}

for unit in units:
    # One rig and one owner per unit. A single-case unit keeps the plain names; a mode unit gets a suffix.
    suffix = "" if unit == case else "_" + unit
    mode = unit
    armature = bpy.data.armatures.new("BboneRigData" + suffix)
    rig = bpy.data.objects.new("Rig" + suffix, armature)
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
        owner = bpy.data.objects.new("CaseOwner" + suffix, None)
        scene.collection.objects.link(owner)
        constraint = owner.constraints.new("COPY_LOCATION")
        constraint.name = "BboneShapeTarget"
        constraint.target = rig
        constraint.subtarget = "Bendy"
        constraint.head_tail = 0.35
        constraint.use_bbone_shape = True
    elif case == "armature_modifier":
        if mode not in (
            "curved_linear", "curved_preserve_volume", "straight_linear", "curved_envelopes",
            "straight_preserve_volume",
            "curved_envelope_multiply", "curved_zyx", "curved_axis_angle",
        ):
            raise ValueError("unknown B-Bone deformation mode: " + mode)
        bone.bbone_mapping_mode = (
            "STRAIGHT" if mode in ("straight_linear", "straight_preserve_volume") else "CURVED"
        )
        if mode == "curved_envelopes":
            bone.head_radius = 0.5
            bone.tail_radius = 0.5
            bone.envelope_distance = 0.75
            bone.envelope_weight = 1.0
        elif mode == "curved_envelope_multiply":
            bone.head_radius = 0.24
            bone.tail_radius = 0.36
            bone.envelope_distance = 1.05
            bone.envelope_weight = 0.73
            bone.use_envelope_multiply = True
        rig.location = (0.31, -0.22, 0.18)
        rig.rotation_euler = (0.11, -0.16, 0.09)
        bendy_pose = rig.pose.bones["Bendy"]
        bendy_pose.rotation_mode = "XYZ"
        bendy_pose.rotation_euler = (0.19, -0.23, 0.14)
        bendy_pose.location = (0.07, 0.11, -0.05)
        bendy_pose.scale = (1.08, 0.93, 1.12)
        bendy_pose.bbone_rollin = 0.27
        if mode == "curved_zyx":
            bendy_pose.rotation_mode = "ZYX"
        elif mode == "curved_axis_angle":
            from mathutils import Euler
            bendy_pose.rotation_mode = "AXIS_ANGLE"
            axis, angle = Euler((0.19, -0.23, 0.14), "XYZ").to_quaternion().to_axis_angle()
            bendy_pose.rotation_axis_angle = (angle, axis.x, axis.y, axis.z)
        start_handle_pose = rig.pose.bones["StartHandle"]
        start_handle_pose.rotation_mode = "XYZ"
        start_handle_pose.rotation_euler = (0.12, -0.18, 4.0)
        end_handle_pose = rig.pose.bones["EndHandle"]
        end_handle_pose.rotation_mode = "XYZ"
        end_handle_pose.rotation_euler = (-0.14, 0.1, -0.12)
        vertices = [
            (-0.31, 0.08, 0.18), (0.43, 0.12, -0.11),
            (-0.18, 0.37, -0.27), (0.33, 0.43, 0.24),
            (-0.45, 0.78, 0.13), (0.16, 0.82, -0.31),
            (-0.09, 1.18, 0.33), (0.39, 1.22, 0.04),
            (-0.29, 1.58, -0.22), (0.25, 1.62, 0.29),
            (-0.13, 1.9, 0.12), (0.36, 1.88, -0.17),
        ]
        faces = [(0, 1, 3, 2), (2, 3, 5, 4), (4, 5, 7, 6),
                 (6, 7, 9, 8), (8, 9, 11, 10)]
        mesh = bpy.data.meshes.new("CaseOwnerData" + suffix)
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        owner = bpy.data.objects.new("CaseOwner" + suffix, mesh)
        scene.collection.objects.link(owner)
        owner.location = (-0.27, 0.13, 0.24)
        owner.rotation_euler = (0.09, -0.12, 0.17)
        group = owner.vertex_groups.new(name="Bendy")
        if mode == "curved_envelopes":
            group.add([0], 0.0, "REPLACE")
        else:
            weights = (0.45, 1.0, 0.72, 0.38, 1.0, 0.61,
                       0.83, 0.52, 0.3, 1.0, 0.67, 0.9)
            for index, weight in enumerate(weights):
                group.add([index], weight, "REPLACE")
        modifier = owner.modifiers.new("WeightedBbone", "ARMATURE")
        modifier.object = rig
        modifier.use_vertex_groups = True
        modifier.use_bone_envelopes = mode in ("curved_envelopes", "curved_envelope_multiply")
        modifier.use_deform_preserve_volume = mode in ("curved_preserve_volume", "straight_preserve_volume")
        bpy.context.view_layer.update()
        evaluated = owner.evaluated_get(bpy.context.evaluated_depsgraph_get())
        evaluated_mesh = evaluated.to_mesh()
        positions = [[float(vertex.co.x), float(vertex.co.y), float(vertex.co.z)]
                     for vertex in evaluated_mesh.vertices]
        evaluated.to_mesh_clear()
        pose_basis = {}
        for name in ("Root", "Bendy", "StartHandle", "EndHandle"):
            pose_bone = rig.pose.bones[name]
            pose_basis[name] = {"rotation_mode": pose_bone.rotation_mode,
                                "rotation_euler": [float(value) for value in pose_bone.rotation_euler],
                                "matrix_basis": [list(row) for row in pose_bone.matrix_basis]}
        geometry[mode] = {"input_positions": vertices, "positions": positions, "pose": pose_basis}
    elif case == "armature_constraint":
        owner = bpy.data.objects.new("CaseOwner" + suffix, None)
        scene.collection.objects.link(owner)
        constraint = owner.constraints.new("ARMATURE")
        constraint.name = "WeightedBboneTarget"
        target = constraint.targets.new()
        target.target = rig
        target.subtarget = "Bendy"
        target.weight = 0.75
    else:
        raise ValueError("unknown fixture case: " + case)

if case == "armature_modifier":
    with open(os.path.join(root, "blender_geometry.json"), "w", encoding="utf-8") as handle:
        json.dump(geometry, handle)

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, "source.blend"), check_existing=False)
"#;

const BLENDER_REOPEN: &str = r#"
import bpy
import json
import sys

arguments = sys.argv[sys.argv.index("--") + 1:]
output_path = arguments[0]
case = arguments[1]
units = arguments[2:] or [case]
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
result = {}
for unit in units:
    suffix = "" if unit == case else "_" + unit
    rig = bpy.data.objects["Rig" + suffix]
    bone = rig.data.bones["Bendy"]
    settings = {}
    for key in keys:
        value = getattr(bone, key)
        if key in ("bbone_custom_handle_start", "bbone_custom_handle_end"):
            value = value.name if value else None
        elif key in ("bbone_handle_use_scale_start", "bbone_handle_use_scale_end", "bbone_scalein", "bbone_scaleout"):
            value = list(value)
        settings[key] = value
    entry = {"settings": settings}
    if case == "target_shape":
        constraint = bpy.data.objects["CaseOwner" + suffix].constraints["BboneShapeTarget"]
        entry["constraint"] = {
            "use_bbone_shape": constraint.use_bbone_shape,
            "head_tail": constraint.head_tail,
            "target": constraint.target.name if constraint.target else None,
            "subtarget": constraint.subtarget,
        }
    entry["pose"] = {}
    for name in ("Root", "Bendy", "StartHandle", "EndHandle"):
        pose_bone = rig.pose.bones[name]
        entry["pose"][name] = {"rotation_mode": pose_bone.rotation_mode,
                               "rotation_euler": [float(value) for value in pose_bone.rotation_euler],
                               "matrix_basis": [list(row) for row in pose_bone.matrix_basis]}
    result[unit] = entry
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

/// Pose-bone rotation modes and local transforms must survive export and reopen in Blender.
fn assert_reopened_pose(reopened: &Value, original: &Value, mode: &str) {
    let expected_bendy_mode = match mode {
        "curved_zyx" => "ZYX",
        "curved_axis_angle" => "AXIS_ANGLE",
        _ => "XYZ",
    };
    assert_eq!(
        original["Bendy"]["rotation_mode"], expected_bendy_mode,
        "{mode}: fixture pose mode"
    );
    for bone in ["Root", "Bendy", "StartHandle", "EndHandle"] {
        assert_eq!(
            reopened[bone]["rotation_mode"], original[bone]["rotation_mode"],
            "{mode}: {bone} rotation mode after reopen"
        );
        let actual = reopened[bone]["matrix_basis"].as_array().unwrap();
        let expected = original[bone]["matrix_basis"].as_array().unwrap();
        assert_eq!(actual.len(), 4, "{mode}: {bone} matrix rows");
        for (row, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let actual = actual.as_array().unwrap();
            let expected = expected.as_array().unwrap();
            for (column, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                let (actual, expected) = (actual.as_f64().unwrap(), expected.as_f64().unwrap());
                assert!(
                    (actual - expected).abs() <= 1.0e-5,
                    "{mode}: {bone} matrix_basis[{row}][{column}]: {actual} != {expected}"
                );
            }
        }
        if original[bone]["rotation_mode"] != "QUATERNION"
            && original[bone]["rotation_mode"] != "AXIS_ANGLE"
        {
            // A no-op round trip writes the source Euler values back unchanged: no 2*pi wrapping.
            let actual = reopened[bone]["rotation_euler"].as_array().unwrap();
            let expected = original[bone]["rotation_euler"].as_array().unwrap();
            for (axis, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                let (actual, expected) = (actual.as_f64().unwrap(), expected.as_f64().unwrap());
                assert!(
                    (actual - expected).abs() <= 1.0e-5,
                    "{mode}: {bone} rotation_euler[{axis}]: {actual} != {expected}"
                );
            }
        }
    }
    let start_handle_z = reopened["StartHandle"]["rotation_euler"][2]
        .as_f64()
        .unwrap();
    assert!(
        (start_handle_z - 4.0).abs() <= 1.0e-5,
        "{mode}: StartHandle rotation_euler z must survive a no-op round trip as 4.0 rad, got {start_handle_z}"
    );
}

fn assert_positions_close(actual: &Value, expected: &Value, context: &str) {
    let actual = actual.as_array().unwrap();
    let expected = expected.as_array().unwrap();
    assert_eq!(actual.len(), expected.len(), "{context}: vertex count");
    for (vertex, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let actual = actual.as_array().unwrap();
        let expected = expected.as_array().unwrap();
        assert_eq!(actual.len(), 3, "{context}: actual vertex {vertex}");
        assert_eq!(expected.len(), 3, "{context}: Blender vertex {vertex}");
        for axis in 0..3 {
            let actual = actual[axis].as_f64().unwrap();
            let expected = expected[axis].as_f64().unwrap();
            assert!(
                (actual - expected).abs() <= 1.0e-4,
                "{context}: vertex {vertex} axis {axis}: Potter {actual} != Blender {expected}"
            );
        }
    }
}

fn assert_positions_move(input: &Value, output: &Value, context: &str) {
    let input = input.as_array().unwrap();
    let output = output.as_array().unwrap();
    assert_eq!(input.len(), output.len(), "{context}: oracle vertex count");
    assert!(
        input.iter().zip(output).any(|(input, output)| {
            (0..3).any(|axis| {
                (input[axis].as_f64().unwrap() - output[axis].as_f64().unwrap()).abs() > 1.0e-4
            })
        }),
        "{context}: Blender fixture did not deform any vertex"
    );
}

/// Runs the fixture once. `units` are the rigs to build (one per deformation mode for
/// `armature_modifier`); an empty list builds the single unit named by `case`.
fn run_blender(blender: &Path, root: &Path, case: &str, units: &[&str]) -> TestResult {
    let script_path = root.join("make_fixture.py");
    fs::write(&script_path, BLENDER_FIXTURE)?;
    let root = fs::canonicalize(root)?;
    let mut command = Command::new(blender);
    command
        .args(["--background", "--factory-startup", "--python"])
        .arg(script_path)
        .arg("--")
        .arg(root)
        .arg(case)
        .args(units);
    assert_success(&run_guarded(command)?, "Blender fixture")
}

/// Reopens the exported scene once and returns the reopened document keyed by unit name.
fn run_reopen(
    blender: &Path,
    root: &Path,
    exported: &Path,
    case: &str,
    units: &[&str],
) -> TestResult<Value> {
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
        .arg(case)
        .args(units);
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

fn assert_imported_bbone_settings(doc: &SceneDoc, rig_name: &str, mapping_mode: &str) {
    let rig = doc
        .nodes
        .values()
        .find(|node| node.name == rig_name)
        .unwrap();
    let armature = doc.data_blocks[rig.data.as_ref().unwrap()]
        .armature
        .as_ref()
        .unwrap();
    let bendy = armature
        .bones
        .values()
        .find(|bone| bone.name == "Bendy")
        .unwrap();
    let mut expected = expected_settings();
    expected["bbone_mapping_mode"] = json!(mapping_mode);
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
    run_blender(&blender, root, case, &[])?;

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
    assert_imported_bbone_settings(&doc, "Rig", "CURVED");

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
    let reopened_units = run_reopen(&blender, root, &exported, case, &[])?;
    let reopened = &reopened_units[case];
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

/// Eight B-Bone deformation modes in one scene. Each mode is one unit: a `Rig_<mode>` armature and a
/// `CaseOwner_<mode>` mesh, built by the fixture's `armature_modifier` branch and keyed by mode in the
/// oracle and reopen documents. One fixture launch, one Potter import, one export and one reopen
/// cover all eight; the per-mode assertions are unchanged. Before, each mode ran its own fixture,
/// import, export and reopen (32 Blender launches and 8 Potter imports and exports). The eight
/// Potter inspects stay, since each owner's evaluated geometry is its own assertion.
#[test]
fn armature_modifier_bbone_gate_and_blend_round_trip() -> TestResult {
    const MODES: [&str; 8] = [
        "curved_linear",
        "curved_preserve_volume",
        "straight_linear",
        "straight_preserve_volume",
        "curved_envelopes",
        "curved_envelope_multiply",
        "curved_zyx",
        "curved_axis_angle",
    ];
    let _guard = BLENDER_BBONE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(blender) = blender() else {
        eprintln!("skipping B-Bone integration test: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    run_blender(&blender, root, "armature_modifier", &MODES)?;
    let oracles: Value = serde_json::from_slice(&fs::read(root.join("blender_geometry.json"))?)?;

    let project = root.join("project");
    pot_json(&["init", project.to_str().unwrap()])?;
    let imported = pot_json(&[
        "import",
        project.to_str().unwrap(),
        "--file",
        root.join("source.blend").to_str().unwrap(),
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
    let mappings = &imported["result"]["id_mappings"];
    let doc: SceneDoc = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    for mode in MODES {
        let mapping_mode = if mode == "straight_linear" || mode == "straight_preserve_volume" {
            "STRAIGHT"
        } else {
            "CURVED"
        };
        assert_imported_bbone_settings(&doc, &format!("Rig_{mode}"), mapping_mode);
    }

    for mode in MODES {
        let oracle = &oracles[mode];
        let owner_id = mappings[&format!("Object:CaseOwner_{mode}")]
            .as_str()
            .unwrap();
        let inspected = pot_json(&["inspect", project.to_str().unwrap(), "--id", owner_id])?;
        let owner = inspected["result"]["items"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["id"] == owner_id)
            .unwrap();
        let expected_positions = &oracle["positions"];
        assert_positions_move(&oracle["input_positions"], expected_positions, mode);
        if mode == "curved_envelopes" {
            let input = oracle["input_positions"].as_array().unwrap();
            let output = expected_positions.as_array().unwrap();
            for axis in 0..3 {
                assert!(
                    (input[0][axis].as_f64().unwrap() - output[0][axis].as_f64().unwrap()).abs()
                        <= 1.0e-4,
                    "curved_envelopes: matching zero-weight group fell back to envelopes"
                );
            }
            assert!(
                input
                    .iter()
                    .skip(1)
                    .zip(output.iter().skip(1))
                    .any(|(input, output)| {
                        (0..3).any(|axis| {
                            (input[axis].as_f64().unwrap() - output[axis].as_f64().unwrap()).abs()
                                > 1.0e-4
                        })
                    }),
                "curved_envelopes: vertices without a matching group were not envelope-deformed"
            );
        }
        assert_positions_close(
            &owner["evaluated_geometry"]["positions"],
            expected_positions,
            mode,
        );
    }

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
    let reopened = run_reopen(&blender, root, &exported, "armature_modifier", &MODES)?;
    for mode in MODES {
        assert_reopened_pose(&reopened[mode]["pose"], &oracles[mode]["pose"], mode);
        if mode == "curved_linear" {
            assert_json_approximately_equal(
                &reopened[mode]["settings"],
                &expected_settings(),
                "armature modifier: reopened B-Bone RNA",
            );
        }
    }
    Ok(())
}

#[test]
fn armature_constraint_bbone_gate_and_blend_round_trip() -> TestResult {
    run_case("armature_constraint", "rig.bbone.segments")
}
