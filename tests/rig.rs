use std::{error::Error, fs, io, path::Path, process::Command};

use glam::{DMat4, DVec3};
use potter::{
    eval::{EvaluationContext, Snapshot},
    model::{Id, SceneDoc},
};
use proptest::{prelude::*, test_runner::TestCaseError};
use serde_json::{Value, json};
use tempfile::{TempDir, tempdir};

type IntegrationResult<T = ()> = Result<T, Box<dyn Error>>;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn apply_batch(project: &Path, base_revision: u64, operations: &Value) -> IntegrationResult {
    let batch = project.join("rig-operations.json");
    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": base_revision,
            "operations": operations,
        }))?,
    )?;
    let output = pot()
        .arg("apply")
        .arg(project)
        .arg("--file")
        .arg(batch)
        .arg("--json")
        .output()?;
    assert!(
        output.status.success(),
        "pot apply failed ({}): stdout={} stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    Ok(())
}

fn new_project(operations: &Value) -> IntegrationResult<(TempDir, std::path::PathBuf)> {
    let directory = tempdir()?;
    let project = directory.path().join("project");
    let output = pot().arg("init").arg(&project).arg("--json").output()?;
    assert!(
        output.status.success(),
        "pot init failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    apply_batch(&project, 0, operations)?;
    Ok((directory, project))
}

fn snapshot(project: &Path, frame: Option<f64>) -> IntegrationResult<Snapshot> {
    let document: SceneDoc = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    Ok(Snapshot::evaluate(
        &document,
        &EvaluationContext {
            frame,
            ..EvaluationContext::default()
        },
    )?)
}
fn assert_inspect_evaluation_failure(project: &Path) -> IntegrationResult<Value> {
    let output = pot().arg("inspect").arg(project).arg("--json").output()?;
    assert!(!output.status.success());
    let envelope: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(envelope["error"]["code"], "EVALUATION_FAILED");
    assert!(
        envelope["error"]["details"]
            .as_object()
            .is_some_and(|details| !details.is_empty())
    );
    Ok(envelope)
}

fn mesh_vertices(snapshot: &Snapshot, mesh_id: &str) -> IntegrationResult<Vec<(u32, DVec3)>> {
    let (_, mesh) = snapshot
        .meshes
        .iter()
        .find(|(id, _)| id.as_str() == mesh_id)
        .ok_or_else(|| io::Error::other(format!("evaluated mesh `{mesh_id}` is missing")))?;
    Ok(mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.co))
        .collect())
}

fn mesh_vertex(snapshot: &Snapshot, mesh_id: &str, vertex_id: u32) -> IntegrationResult<DVec3> {
    mesh_vertices(snapshot, mesh_id)?
        .into_iter()
        .find_map(|(id, position)| (id == vertex_id).then_some(position))
        .ok_or_else(|| {
            io::Error::other(format!("vertex {vertex_id} is missing from `{mesh_id}`")).into()
        })
}

fn node_matrix(snapshot: &Snapshot, node_id: &str) -> IntegrationResult<DMat4> {
    let (_, node) = snapshot
        .nodes
        .iter()
        .find(|(id, _)| id.as_str() == node_id)
        .ok_or_else(|| io::Error::other(format!("evaluated node `{node_id}` is missing")))?;
    Ok(DMat4::from_cols_array(&node.world_matrix))
}

fn bone_matrix(snapshot: &Snapshot, armature_id: &str, bone_id: &str) -> IntegrationResult<DMat4> {
    let (_, bones) = snapshot
        .bone_matrices
        .iter()
        .find(|(id, _)| id.as_str() == armature_id)
        .ok_or_else(|| {
            io::Error::other(format!("armature `{armature_id}` has no evaluated bones"))
        })?;
    let (_, matrix) = bones
        .iter()
        .find(|(id, _)| id.as_str() == bone_id)
        .ok_or_else(|| io::Error::other(format!("evaluated bone `{bone_id}` is missing")))?;
    Ok(DMat4::from_cols_array(matrix))
}

fn bone_endpoint(
    snapshot: &Snapshot,
    armature_id: &str,
    bone_id: &str,
    length: f64,
) -> IntegrationResult<DVec3> {
    Ok(bone_matrix(snapshot, armature_id, bone_id)?.transform_point3(DVec3::new(0.0, length, 0.0)))
}

fn assert_vec3_close(actual: DVec3, expected: DVec3, tolerance: f64) {
    let error = (actual - expected).abs().max_element();
    assert!(
        error <= tolerance,
        "expected {expected:?}, got {actual:?} (max component error {error})"
    );
}

fn armature_chain_operations(target: [f64; 3]) -> Value {
    json!([
        {"op":"node.create","id":"arm","kind":"armature"},
        {"op":"bone.create","target":{"id":"arm"},"id":"root","name":"Root","head":[0.0,0.0,0.0],"tail":[0.0,1.0,0.0]},
        {"op":"bone.create","target":{"id":"arm"},"id":"tip","name":"Tip","head":[0.0,1.0,0.0],"tail":[0.0,2.0,0.0],"parent":"root"},
        {"op":"node.create","id":"goal","kind":"empty","transform":{"translation":target}},
        {"op":"constraint.create","target":{"id":"arm"},"id":"reach","type":"ik","constraint_target":"goal","owner_bone":"tip","params":{"chain_count":2}}
    ])
}

const BLENDER_IK_SINGULAR_FIXTURE: &str = r#"
import bpy
import json
import os
import sys
from mathutils import Vector

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"IK singularity fixture requires Blender 5.2.2, found {bpy.app.version_string}")

fixture_root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
with open(os.path.join(fixture_root, "ik_cases.json"), encoding="utf-8") as source:
    cases = json.load(source)
bpy.ops.wm.read_factory_settings(use_empty=True)
poses = {}
for index, case in enumerate(cases):
    armature = bpy.data.armatures.new(f"Armature{index}")
    rig = bpy.data.objects.new(f"Arm{index}", armature)
    bpy.context.collection.objects.link(rig)
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    root_bone = armature.edit_bones.new("Root")
    root_bone.head = (0.0, 0.0, 0.0)
    root_bone.tail = (0.0, 1.0, 0.0)
    tip_bone = armature.edit_bones.new("Tip")
    tip_bone.head = (0.0, 1.0, 0.0)
    tip_bone.tail = (0.0, 2.0, 0.0)
    tip_bone.parent = root_bone
    tip_bone.use_connect = True
    bpy.ops.object.mode_set(mode="OBJECT")

    goal = bpy.data.objects.new(f"Goal{index}", None)
    bpy.context.collection.objects.link(goal)
    goal.location = case["target"]
    ik = rig.pose.bones["Tip"].constraints.new("IK")
    ik.target = goal
    ik.chain_count = 2
    poses[case["id"]] = {
        "settings": {
            "iterations": ik.iterations,
            "chain_count": ik.chain_count,
            "use_location": ik.use_location,
            "use_rotation": ik.use_rotation,
            "use_stretch": ik.use_stretch,
            "weight": ik.weight,
            "orient_weight": ik.orient_weight,
            "influence": ik.influence,
        },
        "rig": rig,
}

bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
for case in cases:
    pose = poses[case["id"]]
    rig = pose.pop("rig")
    evaluated = rig.evaluated_get(depsgraph)
    root_pose = evaluated.pose.bones["Root"].matrix
    tip_pose = evaluated.pose.bones["Tip"].matrix
    end = evaluated.matrix_world @ tip_pose @ Vector((0.0, 1.0, 0.0))
    pose["endpoint"] = [float(value) for value in end]
    pose["root_matrix"] = [float(root_pose[row][column]) for column in range(4) for row in range(4)]
    pose["tip_matrix"] = [float(tip_pose[row][column]) for column in range(4) for row in range(4)]
with open(os.path.join(fixture_root, "ik_poses.json"), "w", encoding="utf-8") as output:
    json.dump({"version": list(bpy.app.version), "poses": poses}, output)
"#;

fn blender_executable() -> Option<std::path::PathBuf> {
    fn usable(path: std::path::PathBuf) -> Option<std::path::PathBuf> {
        path.is_file().then_some(path)
    }
    if let Some(path) = std::env::var_os("POTTER_BLENDER") {
        return usable(path.into());
    }
    std::env::var_os("PATH")
        .and_then(|path| {
            std::env::split_paths(&path)
                .map(|directory| directory.join("blender"))
                .find_map(usable)
        })
        .or_else(|| usable("/Applications/Blender.app/Contents/MacOS/Blender".into()))
}

fn blender_ik_poses(blender: &Path, root: &Path, cases: &Value) -> IntegrationResult<Value> {
    fs::write(root.join("ik_cases.json"), serde_json::to_vec(cases)?)?;
    let script = root.join("ik_singularity.py");
    fs::write(&script, BLENDER_IK_SINGULAR_FIXTURE)?;
    let output = Command::new(blender)
        .args(["--background", "--factory-startup", "--python"])
        .arg(script)
        .arg("--")
        .arg(root)
        .output()?;
    assert!(
        output.status.success(),
        "Blender IK fixture failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(serde_json::from_slice(&fs::read(
        root.join("ik_poses.json"),
    )?)?)
}

fn blender_vec3(value: &Value) -> IntegrationResult<DVec3> {
    let components = value
        .as_array()
        .filter(|components| components.len() == 3)
        .ok_or_else(|| io::Error::other("Blender IK endpoint must have three components"))?;
    Ok(DVec3::new(
        components[0]
            .as_f64()
            .ok_or_else(|| io::Error::other("Blender IK endpoint x is invalid"))?,
        components[1]
            .as_f64()
            .ok_or_else(|| io::Error::other("Blender IK endpoint y is invalid"))?,
        components[2]
            .as_f64()
            .ok_or_else(|| io::Error::other("Blender IK endpoint z is invalid"))?,
    ))
}

// Blender 5.2.2 outer failed offsets at 0.05-degree steps for radii 0.2..1.9
// in 0.05 increments. The adjacent +0.05-degree samples reach the target.
const BLENDER_IK_PROBE_RADII: [f64; 35] = [
    0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
    1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50, 1.55, 1.60, 1.65, 1.70, 1.75,
    1.80, 1.85, 1.90,
];
const BLENDER_IK_POSITIVE_Y_FAILURE_DEGREES: [f64; 35] = [
    1.30, 1.05, 0.90, 0.75, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.40, 0.35, 0.35, 0.30, 0.30, 0.30,
    0.30, 0.25, 0.25, 0.25, 0.25, 0.25, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20,
    0.20, 0.20, 0.20,
];
const BLENDER_IK_NEGATIVE_Y_FAILURE_DEGREES: [f64; 35] = [
    1.30, 1.05, 0.85, 0.75, 0.65, 0.55, 0.50, 0.45, 0.45, 0.40, 0.35, 0.35, 0.30, 0.30, 0.30, 0.25,
    0.25, 0.25, 0.25, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,
    0.15, 0.15, 0.15,
];

fn blender_ik_failure_half_angle(radius_index: usize, positive_y: bool) -> f64 {
    let thresholds = if positive_y {
        &BLENDER_IK_POSITIVE_Y_FAILURE_DEGREES
    } else {
        &BLENDER_IK_NEGATIVE_Y_FAILURE_DEGREES
    };
    (thresholds[radius_index] + 0.025).to_radians()
}

fn blender_ik_failed_near_axis(angle: f64, radius_index: usize) -> bool {
    let distance_from_axis = angle.cos().abs().asin();
    let positive_y = angle.sin() >= 0.0;
    distance_from_axis <= blender_ik_failure_half_angle(radius_index, positive_y)
}

fn near_axis_ik_cases() -> Vec<(String, [f64; 3], bool)> {
    let mut cases = vec![(
        "proptest_seed".to_owned(),
        [0.000_114_892_242_555_205_53, 0.199_999_966_999_428_8, 0.0],
        false,
    )];
    for (index, radius) in BLENDER_IK_PROBE_RADII.into_iter().enumerate() {
        let axis_samples = [
            (1.0, BLENDER_IK_POSITIVE_Y_FAILURE_DEGREES[index]),
            (-1.0, BLENDER_IK_NEGATIVE_Y_FAILURE_DEGREES[index]),
        ];
        for (axis_direction, last_failed_offset) in axis_samples {
            let offsets = [
                (0.0, false),
                (last_failed_offset, false),
                (last_failed_offset + 0.05, true),
            ];
            for (offset_degrees, reaches) in offsets {
                let offset = offset_degrees.to_radians();
                let sides = [-1.0, 1.0];
                let side_count = if offset_degrees <= f64::EPSILON {
                    1
                } else {
                    sides.len()
                };
                for side in sides.into_iter().take(side_count) {
                    let id = format!(
                        "r{radius:.2}_axis{axis_direction:+.0}_offset{offset_degrees:.2}_side{side:+.0}"
                    );
                    cases.push((
                        id,
                        [
                            side * radius * offset.sin(),
                            axis_direction * radius * offset.cos(),
                            0.0,
                        ],
                        reaches,
                    ));
                }
            }
        }
    }
    cases
}

fn assert_default_blender_ik_settings(settings: &Value) {
    assert_eq!(settings["iterations"].as_u64(), Some(500));
    assert_eq!(settings["chain_count"].as_u64(), Some(2));
    assert_eq!(settings["use_location"].as_bool(), Some(true));
    assert_eq!(settings["use_rotation"].as_bool(), Some(false));
    assert_eq!(settings["use_stretch"].as_bool(), Some(true));
    for field in ["weight", "orient_weight", "influence"] {
        assert!(
            settings[field]
                .as_f64()
                .is_some_and(|value| (value - 1.0).abs() <= f64::EPSILON),
            "unexpected Blender IK default for {field}: {}",
            settings[field]
        );
    }
}

fn assert_blender_ik_sample(
    case_id: &str,
    target: [f64; 3],
    blender_reaches: bool,
    blender_pose: &Value,
) -> IntegrationResult {
    let target = DVec3::from_array(target);
    let blender_endpoint = blender_vec3(&blender_pose["endpoint"])?;
    assert_eq!(
        (blender_endpoint - target).length() <= 1.0e-3,
        blender_reaches,
        "{case_id}: Blender reach classification changed: target={target:?}, endpoint={blender_endpoint:?}"
    );
    assert_default_blender_ik_settings(&blender_pose["settings"]);
    let (_directory, project) = new_project(&armature_chain_operations(target.to_array()))?;
    let evaluated = snapshot(&project, None)?;
    let endpoint = bone_endpoint(&evaluated, "arm", "tip", 1.0)?;
    // Without a pole target, intermediate bend planes are underdetermined; parity is the endpoint pose.
    let error = (endpoint - blender_endpoint).length();
    assert!(
        error <= 1.0e-3,
        "{case_id}: Potter endpoint {endpoint:?} differs from Blender {blender_endpoint:?} by {error}; Blender root pose={}, tip pose={}",
        blender_pose["root_matrix"],
        blender_pose["tip_matrix"]
    );
    Ok(())
}

#[test]
fn blender_legacy_ik_near_axis_endpoints_match_potter() -> IntegrationResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Blender IK singularity parity: Blender is unavailable");
        return Ok(());
    };
    let cases = near_axis_ik_cases();
    let blender_cases = json!(
        cases
            .iter()
            .map(|(id, target, _)| json!({"id": id, "target": target}))
            .collect::<Vec<_>>()
    );
    let directory = tempdir()?;
    let blender_poses = blender_ik_poses(&blender, directory.path(), &blender_cases)?;
    assert_eq!(blender_poses["version"], json!([5, 2, 2]));
    for (id, target, reaches) in cases {
        assert_blender_ik_sample(&id, target, reaches, &blender_poses["poses"][id.as_str()])?;
    }
    Ok(())
}

#[test]
fn posed_two_bone_armature_deforms_a_weighted_mesh_vertex() -> IntegrationResult {
    let (_directory, project) = new_project(&json!([
        {"op":"node.create","id":"mesh","kind":"plane","params":{"size":2.0}},
        {"op":"node.create","id":"arm","kind":"armature"},
        {"op":"bone.create","target":{"id":"arm"},"id":"root","name":"Root","head":[0.0,0.0,0.0],"tail":[0.0,1.0,0.0]},
        {"op":"bone.create","target":{"id":"arm"},"id":"tip","name":"Tip","head":[0.0,1.0,0.0],"tail":[0.0,2.0,0.0],"parent":"root"},
        {"op":"vertex_group.create","target":{"id":"mesh"},"id":"root","name":"Root"},
        {"op":"vertex_group.assign","target":{"id":"mesh"},"group_id":"root","weights":[{"vertex_id":0,"weight":1.0}]},
        {"op":"modifier.create","target":{"id":"mesh"},"id":"skin","type":"armature","params":{"object":"arm","use_vertex_groups":true}}
    ]))?;
    let rest = snapshot(&project, None)?;
    let rest_position = mesh_vertex(&rest, "mesh", 0)?;
    assert_vec3_close(rest_position, DVec3::new(-1.0, -1.0, 0.0), 1.0e-12);

    let half_sqrt = std::f64::consts::FRAC_1_SQRT_2;
    apply_batch(
        &project,
        1,
        &json!([{
            "op":"pose.set",
            "target":{"id":"arm"},
            "bone_id":"root",
            "set":{"rotation":[0.0,half_sqrt,0.0,half_sqrt]}
        }]),
    )?;
    let posed = snapshot(&project, None)?;
    assert_vec3_close(
        mesh_vertex(&posed, "mesh", 0)?,
        DVec3::new(0.0, -1.0, 1.0),
        1.0e-6,
    );
    apply_batch(
        &project,
        2,
        &json!([{"op":"modifier.apply","target":{"id":"mesh"},"id":"skin"}]),
    )?;
    let baked = snapshot(&project, None)?;
    assert_vec3_close(
        mesh_vertex(&baked, "mesh", 0)?,
        DVec3::new(0.0, -1.0, 1.0),
        1.0e-6,
    );
    Ok(())
}

#[test]
fn armature_skinning_is_stable_when_mesh_and_rig_share_a_world_offset() -> IntegrationResult {
    let half_sqrt = std::f64::consts::FRAC_1_SQRT_2;
    let (_directory, project) = new_project(&json!([
        {"op":"node.create","id":"mesh_origin","kind":"plane","params":{"size":2.0}},
        {"op":"node.create","id":"arm_origin","kind":"armature"},
        {"op":"node.create","id":"mesh_offset","kind":"plane","params":{"size":2.0},"transform":{"translation":[0.0,0.0,5.0]}},
        {"op":"node.create","id":"arm_offset","kind":"armature","transform":{"translation":[0.0,0.0,5.0]}},
        {"op":"bone.create","target":{"id":"arm_origin"},"id":"root_origin","name":"Root","head":[0.0,0.0,0.0],"tail":[0.0,1.0,0.0]},
        {"op":"bone.create","target":{"id":"arm_origin"},"id":"tip_origin","name":"Tip","head":[0.0,1.0,0.0],"tail":[0.0,2.0,0.0],"parent":"root_origin"},
        {"op":"bone.create","target":{"id":"arm_offset"},"id":"root_offset","name":"Root","head":[0.0,0.0,0.0],"tail":[0.0,1.0,0.0]},
        {"op":"bone.create","target":{"id":"arm_offset"},"id":"tip_offset","name":"Tip","head":[0.0,1.0,0.0],"tail":[0.0,2.0,0.0],"parent":"root_offset"},
        {"op":"pose.set","target":{"id":"arm_origin"},"bone_id":"root_origin","set":{"rotation":[0.0,half_sqrt,0.0,half_sqrt]}},
        {"op":"pose.set","target":{"id":"arm_offset"},"bone_id":"root_offset","set":{"rotation":[0.0,half_sqrt,0.0,half_sqrt]}},
        {"op":"vertex_group.create","target":{"id":"mesh_origin"},"id":"root_group_origin","name":"Root"},
        {"op":"vertex_group.assign","target":{"id":"mesh_origin"},"group_id":"root_group_origin","weights":[{"vertex_id":0,"weight":1.0}]},
        {"op":"vertex_group.create","target":{"id":"mesh_offset"},"id":"root_group_offset","name":"Root"},
        {"op":"vertex_group.assign","target":{"id":"mesh_offset"},"group_id":"root_group_offset","weights":[{"vertex_id":0,"weight":1.0}]},
        {"op":"modifier.create","target":{"id":"mesh_origin"},"id":"skin_origin","type":"armature","params":{"object":"arm_origin","use_vertex_groups":true}},
        {"op":"modifier.create","target":{"id":"mesh_offset"},"id":"skin_offset","type":"armature","params":{"object":"arm_offset","use_vertex_groups":true}}
    ]))?;

    let evaluated = snapshot(&project, None)?;
    let expected = DVec3::new(0.0, -1.0, 1.0);
    assert_vec3_close(mesh_vertex(&evaluated, "mesh_origin", 0)?, expected, 1.0e-6);
    assert_vec3_close(mesh_vertex(&evaluated, "mesh_offset", 0)?, expected, 1.0e-6);
    Ok(())
}

#[test]
fn shape_key_value_blends_basis_and_target_positions() -> IntegrationResult {
    let (_directory, project) = new_project(&json!([
        {"op":"node.create","id":"mesh","kind":"plane","params":{"size":2.0}},
        {"op":"shape_key.create","target":{"id":"mesh"},"id":"raise","name":"Raise","positions":{"0":[3.0,1.0,0.0]}},
        {"op":"shape_key.update","target":{"id":"mesh"},"id":"raise","set":{"value":0.5}}
    ]))?;

    let evaluated = snapshot(&project, None)?;
    assert_vec3_close(
        mesh_vertex(&evaluated, "mesh", 0)?,
        DVec3::new(1.0, 0.0, 0.0),
        1.0e-12,
    );
    Ok(())
}
#[test]
fn absolute_shape_keys_interpolate_at_animated_evaluation_time() -> IntegrationResult {
    let (_directory, project) = new_project(&json!([
        {"op":"node.create","id":"mesh","kind":"plane","params":{"size":2.0}},
        {"op":"shape_key.create","target":{"id":"mesh"},"id":"early","name":"Early","frame":1.0,"positions":{"0":[1.0,-1.0,0.0]}},
        {"op":"shape_key.create","target":{"id":"mesh"},"id":"late","name":"Late","frame":3.0,"positions":{"0":[3.0,-1.0,0.0]}},
        {"op":"action.create","id":"shape_action"},
        {"op":"node.update","target":{"id":"mesh"},"set":{"action":"shape_action"}},
        {"op":"keyframe.insert","target":{"id":"mesh"},"path":"shape_keys.evaluation_time","index":0,"frame":1.0,"value":1.0},
        {"op":"keyframe.insert","target":{"id":"mesh"},"path":"shape_keys.evaluation_time","index":0,"frame":3.0,"value":3.0}
    ]))?;

    let evaluated = snapshot(&project, Some(2.0))?;
    assert_vec3_close(
        mesh_vertex(&evaluated, "mesh", 0)?,
        DVec3::new(2.0, -1.0, 0.0),
        1.0e-12,
    );
    Ok(())
}
#[test]
fn bone_collections_assign_bones_and_preserve_visibility_and_custom_shapes() -> IntegrationResult {
    let (_directory, project) = new_project(&json!([
        {"op":"node.create","id":"arm","kind":"armature"},
        {"op":"bone.create","target":{"id":"arm"},"id":"root","name":"Root","head":[0.0,0.0,0.0],"tail":[0.0,1.0,0.0]},
        {"op":"node.create","id":"shape","kind":"empty"},
        {"op":"bone.update","target":{"id":"arm"},"bone_id":"root","set":{"custom_shape":"shape","envelope_distance":0.75,"envelope_weight":0.8,"head_radius":0.2,"tail_radius":0.4}},
        {"op":"bone_collection.create","target":{"id":"arm"},"id":"deform","name":"Deform","visible":false},
        {"op":"bone_collection.assign","target":{"id":"arm"},"collection_id":"deform","bone_ids":["root"]}
    ]))?;
    let document: SceneDoc = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let arm_id = Id::new("arm")?;
    let data_id = document
        .nodes
        .get(&arm_id)
        .and_then(|node| node.data.as_ref())
        .ok_or_else(|| io::Error::other("armature has no data"))?;
    let armature = document
        .data_blocks
        .get(data_id)
        .and_then(|data| data.armature.as_ref())
        .ok_or_else(|| io::Error::other("armature data is missing"))?;
    let root_id = Id::new("root")?;
    let bone = armature
        .bones
        .get(&root_id)
        .ok_or_else(|| io::Error::other("root bone is missing"))?;
    assert_eq!(bone.custom_shape.as_ref().map(Id::as_str), Some("shape"));
    assert!((bone.envelope_distance - 0.75).abs() < 1.0e-12);
    assert!((bone.envelope_weight - 0.8).abs() < 1.0e-12);
    assert!((bone.head_radius - 0.2).abs() < 1.0e-12);
    assert!((bone.tail_radius - 0.4).abs() < 1.0e-12);
    let collection_id = Id::new("deform")?;
    let collection = armature
        .bone_collections
        .get(&collection_id)
        .ok_or_else(|| io::Error::other("deform collection is missing"))?;
    assert!(!collection.visible);
    assert_eq!(
        collection.bones.iter().map(Id::as_str).collect::<Vec<_>>(),
        ["root"]
    );
    Ok(())
}
#[test]
fn rig_generate_basic_human_creates_scaled_deform_and_control_rigs() -> IntegrationResult {
    let (_directory, project) = new_project(&json!([
        {"op":"node.create","id":"arm","kind":"armature"},
        {"op":"rig.generate_basic_human","target":{"id":"arm"},"scale":2.0}
    ]))?;
    let document: SceneDoc = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let arm_id = Id::new("arm")?;
    let data_id = document
        .nodes
        .get(&arm_id)
        .and_then(|node| node.data.as_ref())
        .ok_or_else(|| io::Error::other("armature has no data"))?;
    let armature = document
        .data_blocks
        .get(data_id)
        .and_then(|data| data.armature.as_ref())
        .ok_or_else(|| io::Error::other("armature data is missing"))?;
    let root_id = Id::new("root")?;
    let root = armature
        .bones
        .get(&root_id)
        .ok_or_else(|| io::Error::other("generated root bone is missing"))?;
    let head_id = Id::new("head")?;
    let head = armature
        .bones
        .get(&head_id)
        .ok_or_else(|| io::Error::other("generated head bone is missing"))?;
    let hand_id = Id::new("hand_ik_l")?;
    let hand_control = armature
        .bones
        .get(&hand_id)
        .ok_or_else(|| io::Error::other("generated hand IK control is missing"))?;
    assert_vec3_close(
        DVec3::from_array(root.tail),
        DVec3::new(0.0, 0.0, 0.3),
        1.0e-12,
    );
    assert_eq!(head.name, "head");
    assert!(head.deform);
    assert!(!hand_control.deform);
    let deform_collection_id = Id::new("rigify_deform")?;
    let controls_collection_id = Id::new("rigify_controls")?;
    assert_eq!(
        armature.bone_collections[&deform_collection_id].bones.len(),
        23
    );
    assert_eq!(
        armature.bone_collections[&controls_collection_id]
            .bones
            .len(),
        8
    );
    let node = document
        .nodes
        .get(&arm_id)
        .ok_or_else(|| io::Error::other("armature node is missing"))?;
    assert_eq!(node.constraints.len(), 4);
    assert!(node.constraints.iter().all(|constraint| {
        constraint.constraint_type == potter::model::ConstraintType::Ik && constraint.enabled
    }));
    let hand_target = Id::new("rig_arm_hand_ik_l")?;
    assert!(document.nodes.contains_key(&hand_target));
    Ok(())
}

#[test]
fn armature_modifier_uses_bone_envelopes_without_vertex_groups() -> IntegrationResult {
    let (_directory, project) = new_project(&json!([
        {"op":"node.create","id":"mesh","kind":"plane","params":{"size":2.0}},
        {"op":"node.create","id":"arm","kind":"armature"},
        {"op":"bone.create","target":{"id":"arm"},"id":"root","name":"Root","head":[-1.0,-1.0,0.0],"tail":[-1.0,1.0,0.0],"envelope_distance":0.5,"head_radius":0.5,"tail_radius":0.5},
        {"op":"pose.set","target":{"id":"arm"},"bone_id":"root","set":{"translation":[1.0,0.0,0.0]}},
        {"op":"modifier.create","target":{"id":"mesh"},"id":"skin","type":"armature","params":{"object":"arm","use_vertex_groups":false,"use_bone_envelopes":true}}
    ]))?;
    let evaluated = snapshot(&project, None)?;
    assert_vec3_close(
        mesh_vertex(&evaluated, "mesh", 0)?,
        DVec3::new(0.0, -1.0, 0.0),
        1.0e-12,
    );
    assert_vec3_close(
        mesh_vertex(&evaluated, "mesh", 1)?,
        DVec3::new(1.0, -1.0, 0.0),
        1.0e-12,
    );
    Ok(())
}

#[test]
fn copy_location_and_track_to_constraints_evaluate_their_targets() -> IntegrationResult {
    let quarter_turn = std::f64::consts::FRAC_1_SQRT_2;
    let (_directory, project) = new_project(&json!([
        {"op":"node.create","id":"copy_target","kind":"empty","transform":{"translation":[2.0,3.0,4.0]}},
        {"op":"node.create","id":"copy_owner","kind":"empty"},
        {"op":"constraint.create","target":{"id":"copy_owner"},"id":"copy","type":"copy_location","constraint_target":"copy_target"},
        {"op":"node.create","id":"track_target","kind":"empty","transform":{"translation":[0.0,4.0,0.0]}},
        {"op":"node.create","id":"track_owner","kind":"empty","transform":{"rotation":[quarter_turn,0.0,0.0,quarter_turn]}},
        {"op":"constraint.create","target":{"id":"track_owner"},"id":"track","type":"track_to","constraint_target":"track_target","params":{}}
    ]))?;

    let evaluated = snapshot(&project, None)?;
    let copied = node_matrix(&evaluated, "copy_owner")?.w_axis.truncate();
    assert_vec3_close(copied, DVec3::new(2.0, 3.0, 4.0), 1.0e-12);

    let tracked_axis = node_matrix(&evaluated, "track_owner")?
        .transform_vector3(DVec3::Z)
        .normalize();
    assert_vec3_close(tracked_axis, -DVec3::Y, 1.0e-6);
    Ok(())
}

#[test]
fn node_parent_can_target_a_bone() -> IntegrationResult {
    let (_directory, project) = new_project(&json!([
        {"op":"node.create","id":"arm","kind":"armature"},
        {"op":"bone.create","target":{"id":"arm"},"id":"root","name":"Root","head":[0.0,0.0,0.0],"tail":[0.0,2.0,0.0]},
        {"op":"node.create","id":"child","kind":"empty","transform":{"translation":[0.0,1.0,0.0]}},
        {"op":"node.parent","target":{"id":"child"},"parent":"arm","parent_type":"bone","parent_bone":"root","keep_world":false}
    ]))?;
    let evaluated = snapshot(&project, None)?;
    let child_world = node_matrix(&evaluated, "child")?.w_axis.truncate();
    assert_vec3_close(child_world, DVec3::new(0.0, 1.0, 0.0), 1.0e-12);
    Ok(())
}

#[test]
fn two_bone_ik_reaches_a_reachable_target() -> IntegrationResult {
    let (_directory, project) = new_project(&armature_chain_operations([1.0, 1.0, 0.0]))?;
    let evaluated = snapshot(&project, None)?;
    let endpoint = bone_endpoint(&evaluated, "arm", "tip", 1.0)?;
    assert_vec3_close(
        endpoint,
        DVec3::new(1.000_038_385_391_235_4, 1.000_020_980_834_961, 0.0),
        1.0e-6,
    );
    Ok(())
}

#[test]
fn scripted_driver_expression_drives_a_transform_component() -> IntegrationResult {
    let (_directory, project) = new_project(&json!([
        {"op":"node.create","id":"source","kind":"empty","transform":{"translation":[3.0,0.0,0.0]}},
        {"op":"node.create","id":"driven","kind":"empty"},
        {"op":"driver.create","target":{"id":"driven"},"id":"double_x","path":"transform.translation","index":0,"type":"scripted_expression","variables":[{"name":"var","type":"transforms","target":"source","path":"transform.translation","index":0}],"expression":"var*2"}
    ]))?;

    let evaluated = snapshot(&project, None)?;
    assert!((node_matrix(&evaluated, "driven")?.w_axis.x - 6.0).abs() < 1.0e-12);
    Ok(())
}
#[test]
fn driver_location_rotation_difference_and_context_property_variables_evaluate() -> IntegrationResult
{
    let (_directory, project) = new_project(&json!([
        {"op":"node.create","id":"source","kind":"empty"},
        {"op":"node.create","id":"other","kind":"empty","transform":{"translation":[3.0,4.0,0.0],"rotation":[0.0,0.0,std::f64::consts::FRAC_1_SQRT_2,std::f64::consts::FRAC_1_SQRT_2]}},
        {"op":"node.create","id":"driven","kind":"empty"},
        {"op":"driver.create","target":{"id":"driven"},"id":"location_distance","path":"transform.translation","index":0,"type":"scripted_expression","variables":[{"name":"distance","type":"loc_diff","target":"source","target_2":"other","transform_space":"world"}],"expression":"distance"},
        {"op":"driver.create","target":{"id":"driven"},"id":"rotation_distance","path":"transform.translation","index":1,"type":"scripted_expression","variables":[{"name":"angle","type":"rotation_diff","target":"source","target_2":"other","transform_space":"local"}],"expression":"angle"},
        {"op":"driver.create","target":{"id":"driven"},"id":"custom_property","path":"transform.translation","index":2,"type":"scripted_expression","variables":[{"name":"weight","type":"context_prop","target":"source","path":"properties.weight","transform_space":"local"}],"expression":"weight"}
    ]))?;
    let scene_file = project.join("scene.json");
    let mut document: Value = serde_json::from_slice(&fs::read(&scene_file)?)?;
    document["nodes"]["source"]["properties"]["weight"] = json!(7.0);
    fs::write(&scene_file, serde_json::to_vec(&document)?)?;

    let evaluated = snapshot(&project, None)?;
    assert_vec3_close(
        node_matrix(&evaluated, "driven")?.w_axis.truncate(),
        DVec3::new(5.0, std::f64::consts::FRAC_PI_2, 7.0),
        1.0e-12,
    );
    Ok(())
}

#[test]
fn constraint_and_driver_dependency_cycles_fail_evaluation_with_details() -> IntegrationResult {
    let (_directory, project) = new_project(&json!([
        {"op":"node.create","id":"first","kind":"empty"},
        {"op":"node.create","id":"second","kind":"empty"}
    ]))?;
    let scene_file = project.join("scene.json");
    let mut document: Value = serde_json::from_slice(&fs::read(&scene_file)?)?;
    document["nodes"]["first"]["constraints"] = json!([{
        "id":"to_second",
        "type":"copy_location",
        "name":"To Second",
        "target":"second"
    }]);
    document["nodes"]["second"]["constraints"] = json!([{
        "id":"to_first",
        "type":"copy_location",
        "name":"To First",
        "target":"first"
    }]);
    fs::write(&scene_file, serde_json::to_vec(&document)?)?;
    let constraint_error = assert_inspect_evaluation_failure(&project)?;
    assert!(
        constraint_error["error"]["details"]["cycle"]
            .as_array()
            .is_some_and(|cycle| cycle.len() >= 2)
    );

    document["nodes"]["first"]["constraints"] = json!([]);
    document["nodes"]["second"]["constraints"] = json!([]);
    document["nodes"]["first"]["drivers"] = json!([{
        "id":"from_second",
        "path":"transform.translation",
        "index":0,
        "type":"scripted_expression",
        "variables":[{"name":"var","type":"transforms","target":"second","path":"transform.translation","index":0}],
        "expression":"var"
    }]);
    document["nodes"]["second"]["drivers"] = json!([{
        "id":"from_first",
        "path":"transform.translation",
        "index":0,
        "type":"scripted_expression",
        "variables":[{"name":"var","type":"transforms","target":"first","path":"transform.translation","index":0}],
        "expression":"var"
    }]);
    fs::write(&scene_file, serde_json::to_vec(&document)?)?;
    let driver_error = assert_inspect_evaluation_failure(&project)?;
    assert!(driver_error["error"]["details"]["node"].is_string());
    assert!(driver_error["error"]["details"]["driver_id"].is_string());
    Ok(())
}

#[test]
fn pose_bone_keyframes_evaluate_at_a_fractional_frame() -> IntegrationResult {
    let (_directory, project) = new_project(&json!([
        {"op":"node.create","id":"arm","kind":"armature"},
        {"op":"bone.create","target":{"id":"arm"},"id":"root","name":"Root","head":[0.0,0.0,0.0],"tail":[0.0,1.0,0.0]},
        {"op":"action.create","id":"pose_action","name":"Pose"},
        {"op":"node.update","target":{"id":"arm"},"set":{"action":"pose_action"}},
        {"op":"keyframe.insert","target":{"id":"arm"},"path":"pose.root.translation","index":0,"frame":1.0,"value":0.0,"interpolation":"linear"},
        {"op":"keyframe.insert","target":{"id":"arm"},"path":"pose.root.translation","index":0,"frame":2.0,"value":2.0,"interpolation":"linear"}
    ]))?;

    let evaluated = snapshot(&project, Some(1.5))?;
    assert!((bone_matrix(&evaluated, "arm", "root")?.w_axis.x - 1.0).abs() < 1.0e-12);
    Ok(())
}

#[test]
fn rig_rest_pose_shape_key_animation_and_inspect_round_trip() -> IntegrationResult {
    let (_directory, project) = new_project(&json!([
        {"op":"node.create","id":"mesh","kind":"plane","params":{"size":2.0}},
        {"op":"shape_key.create","target":{"id":"mesh"},"id":"raise","name":"Raise","positions":{"0":[2.0,-1.0,0.0]}},
        {"op":"action.create","id":"shape_action","name":"Shape"},
        {"op":"node.update","target":{"id":"mesh"},"set":{"action":"shape_action"}},
        {"op":"keyframe.insert","target":{"id":"mesh"},"path":"shape_key.raise.value","index":0,"frame":1.0,"value":0.0,"interpolation":"linear"},
        {"op":"keyframe.insert","target":{"id":"mesh"},"path":"shape_key.raise.value","index":0,"frame":2.0,"value":1.0,"interpolation":"linear"},
        {"op":"node.create","id":"arm","kind":"armature"},
        {"op":"bone.create","target":{"id":"arm"},"id":"root","name":"Root","head":[0.0,0.0,0.0],"tail":[0.0,1.0,0.0]},
        {"op":"bone.update","target":{"id":"arm"},"bone_id":"root","set":{"tail":[0.0,2.0,0.0]}},
        {"op":"pose.set","target":{"id":"arm"},"bone_id":"root","set":{"translation":[3.0,0.0,0.0]}},
        {"op":"pose.reset","target":{"id":"arm"},"bone_id":"root"}
    ]))?;

    let evaluated = snapshot(&project, Some(2.0))?;
    assert_vec3_close(
        mesh_vertex(&evaluated, "mesh", 0)?,
        DVec3::new(2.0, -1.0, 0.0),
        1.0e-12,
    );
    assert_vec3_close(
        bone_endpoint(&evaluated, "arm", "root", 2.0)?,
        DVec3::new(0.0, 2.0, 0.0),
        1.0e-12,
    );

    let output = pot()
        .arg("inspect")
        .arg(&project)
        .arg("--id")
        .arg("mesh")
        .arg("--frame")
        .arg("2")
        .arg("--json")
        .output()?;
    assert!(
        output.status.success(),
        "pot inspect failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let envelope: Value = serde_json::from_slice(&output.stdout)?;
    assert!(envelope["result"]["items"][0]["action"].is_string());
    assert!(envelope["result"]["items"][0]["evaluated_geometry"]["vertex_count"].is_number());
    assert_eq!(
        envelope["result"]["items"][0]["evaluated_shape_keys"]["raise"],
        json!(1.0)
    );
    Ok(())
}

#[test]
fn extended_transform_constraints_evaluate_expected_axes_and_bounds() -> IntegrationResult {
    let quarter_turn = std::f64::consts::FRAC_1_SQRT_2;
    let (_directory, project) = new_project(&json!([
        {"op":"node.create","id":"look_target","kind":"empty","transform":{"translation":[3.0,0.0,0.0]}},
        {"op":"node.create","id":"locked","kind":"empty"},
        {"op":"constraint.create","target":{"id":"locked"},"id":"locked_track","type":"locked_track","constraint_target":"look_target","params":{"track_axis":"TRACK_Z","lock_axis":"LOCK_Y"}},
        {"op":"node.create","id":"stretch_target","kind":"empty","transform":{"translation":[0.0,3.0,0.0]}},
        {"op":"node.create","id":"stretched","kind":"empty"},
        {"op":"constraint.create","target":{"id":"stretched"},"id":"stretch","type":"stretch_to","constraint_target":"stretch_target","params":{"rest_length":1.0}},
        {"op":"node.create","id":"value_source","kind":"empty","transform":{"translation":[5.0,0.0,0.0]}},
        {"op":"node.create","id":"transformed","kind":"empty"},
        {"op":"constraint.create","target":{"id":"transformed"},"id":"map","type":"transformation","constraint_target":"value_source","params":{"map_from":"LOCATION","map_to":"SCALE","map_to_y_from":"X","from_min_x":0.0,"from_max_x":10.0,"to_min_y_scale":1.0,"to_max_y_scale":3.0}},
        {"op":"node.create","id":"volume","kind":"empty","transform":{"scale":[1.0,2.0,1.0]}},
        {"op":"constraint.create","target":{"id":"volume"},"id":"volume_keep","type":"maintain_volume","params":{"free_axis":"SAMEVOL_Y","volume":1.0}},
        {"op":"node.create","id":"floor","kind":"empty"},
        {"op":"node.create","id":"below_floor","kind":"empty","transform":{"translation":[0.0,0.0,-1.0]}},
        {"op":"constraint.create","target":{"id":"below_floor"},"id":"floor_lock","type":"floor","constraint_target":"floor","params":{"floor_location":"FLOOR_Z"}},
        {"op":"node.create","id":"pivot","kind":"empty","transform":{"rotation":[0.0,0.0,quarter_turn,quarter_turn]}},
        {"op":"node.create","id":"orbit","kind":"empty","transform":{"translation":[1.0,0.0,0.0]}},
        {"op":"constraint.create","target":{"id":"orbit"},"id":"orbit_pivot","type":"pivot","constraint_target":"pivot"},
        {"op":"node.create","id":"surface","kind":"plane","params":{"size":2.0}},
        {"op":"node.create","id":"wrapped","kind":"empty","transform":{"translation":[1.0,-1.0,2.0]}},
        {"op":"constraint.create","target":{"id":"wrapped"},"id":"wrap","type":"shrinkwrap","constraint_target":"surface"}
    ]))?;
    let evaluated = snapshot(&project, None)?;

    let locked = node_matrix(&evaluated, "locked")?;
    assert_vec3_close(locked.transform_vector3(DVec3::Z), DVec3::X, 1.0e-6);
    assert_vec3_close(locked.transform_vector3(DVec3::Y), DVec3::Y, 1.0e-6);
    assert!(
        (node_matrix(&evaluated, "stretched")?
            .transform_vector3(DVec3::Y)
            .length()
            - 3.0)
            .abs()
            < 1.0e-6
    );
    assert!(
        (node_matrix(&evaluated, "transformed")?
            .transform_vector3(DVec3::Y)
            .length()
            - 2.0)
            .abs()
            < 1.0e-6
    );
    let volume_scale = node_matrix(&evaluated, "volume")?
        .to_scale_rotation_translation()
        .0;
    assert!((volume_scale.x * volume_scale.y * volume_scale.z - 1.0).abs() < 1.0e-6);
    assert!((node_matrix(&evaluated, "below_floor")?.w_axis.z).abs() < 1.0e-12);
    assert_vec3_close(
        node_matrix(&evaluated, "orbit")?.w_axis.truncate(),
        DVec3::new(1.0, 0.0, 0.0),
        1.0e-6,
    );
    assert!((node_matrix(&evaluated, "wrapped")?.w_axis.z).abs() < 1.0e-12);
    Ok(())
}

#[test]
fn ik_chain_count_and_pole_target_are_evaluated() -> IntegrationResult {
    let (_directory, project) = new_project(&json!([
        {"op":"node.create","id":"arm","kind":"armature"},
        {"op":"bone.create","target":{"id":"arm"},"id":"root","name":"Root","head":[0.0,0.0,0.0],"tail":[0.0,1.0,0.0]},
        {"op":"bone.create","target":{"id":"arm"},"id":"tip","name":"Tip","head":[0.0,1.0,0.0],"tail":[0.0,2.0,0.0],"parent":"root"},
        {"op":"node.create","id":"goal","kind":"empty","transform":{"translation":[1.0,1.0,0.0]}},
        {"op":"node.create","id":"pole","kind":"empty","transform":{"translation":[0.0,0.0,2.0]}},
        {"op":"constraint.create","target":{"id":"arm"},"id":"reach","type":"ik","constraint_target":"goal","owner_bone":"tip","params":{"chain_count":2,"pole_target":"pole"}}
    ]))?;
    let evaluated = snapshot(&project, None)?;
    assert_vec3_close(
        bone_endpoint(&evaluated, "arm", "tip", 1.0)?,
        DVec3::new(1.414_213_538_169_860_8, 1.414_213_538_169_860_8, 0.0),
        1.0e-6,
    );
    Ok(())
}

#[test]
fn pose_bone_ik_owner_import_is_inspectable_and_editable() -> IntegrationResult {
    let (_directory, project) = new_project(&json!([
        {"op":"node.create","id":"arm","kind":"armature"},
        {"op":"bone.create","target":{"id":"arm"},"id":"root","name":"Root",
         "head":[0.0,0.0,0.0],"tail":[0.0,1.0,0.0]},
        {"op":"bone.create","target":{"id":"arm"},"id":"tip","name":"Tip",
         "head":[0.0,1.0,0.0],"tail":[0.0,2.0,0.0],"parent":"root"},
        {"op":"node.create","id":"goal","kind":"empty",
         "transform":{"translation":[1.0,1.0,0.0]}},
        {"op":"constraint.create","target":{"id":"arm"},"id":"pose_reach",
         "type":"ik","name":"Pose Reach","constraint_target":"goal",
         "owner_bone":"tip","influence":1.0,"params":{"chain_count":2}}
    ]))?;
    let inspected = pot()
        .arg("inspect")
        .arg(&project)
        .arg("--id")
        .arg("arm")
        .arg("--json")
        .output()?;
    assert!(
        inspected.status.success(),
        "pot inspect failed: {}",
        String::from_utf8_lossy(&inspected.stderr)
    );
    let envelope: Value = serde_json::from_slice(&inspected.stdout)?;
    let arm = envelope["result"]["items"]
        .as_array()
        .and_then(|items| items.iter().find(|item| item["id"] == "arm"))
        .ok_or("inspect omitted the armature node")?;
    assert_eq!(arm["constraints"][0]["owner_bone"], "tip");
    let matrix_values = arm["evaluated_bones"]["tip"]
        .as_array()
        .ok_or("inspect omitted evaluated tip bone")?;
    let mut matrix = [0.0; 16];
    for (target, value) in matrix.iter_mut().zip(matrix_values) {
        *target = value
            .as_f64()
            .ok_or("evaluated bone matrix is not numeric")?;
    }
    assert_vec3_close(
        DMat4::from_cols_array(&matrix).transform_point3(DVec3::Y),
        DVec3::new(1.000_038_385_391_235_4, 1.000_020_980_834_961, 0.0),
        1.0e-6,
    );

    apply_batch(
        &project,
        1,
        &json!([{
            "op":"constraint.update",
            "target":{"id":"arm"},
            "id":"pose_reach",
            "set":{"influence":0.5,"params":{"chain_count":1}}
        }]),
    )?;
    let edited = snapshot(&project, None)?;
    let edited_endpoint = bone_endpoint(&edited, "arm", "tip", 1.0)?;
    let edited_distance = edited_endpoint.distance(DVec3::new(1.0, 1.0, 0.0));
    assert!(
        edited_distance > 1.0e-3,
        "edited pose IK influence/chain_count did not affect evaluated bones: \
         endpoint={edited_endpoint:?} distance={edited_distance}"
    );
    Ok(())
}

#[test]
fn pose_bone_copy_location_updates_descendant_bone_matrices() -> IntegrationResult {
    let (_directory, project) = new_project(&json!([
        {"op":"node.create","id":"arm","kind":"armature"},
        {"op":"bone.create","target":{"id":"arm"},"id":"root","name":"Root",
         "head":[0.0,0.0,0.0],"tail":[0.0,1.0,0.0]},
        {"op":"bone.create","target":{"id":"arm"},"id":"tip","name":"Tip",
         "head":[0.0,1.0,0.0],"tail":[0.0,2.0,0.0],"parent":"root"},
        {"op":"bone.create","target":{"id":"arm"},"id":"end","name":"End",
         "head":[0.0,2.0,0.0],"tail":[0.0,3.0,0.0],"parent":"tip"},
        {"op":"node.create","id":"goal","kind":"empty",
         "transform":{"translation":[1.0,1.0,0.0]}},
        {"op":"constraint.create","target":{"id":"arm"},"id":"pose_copy",
         "type":"copy_location","name":"Move Tip","constraint_target":"goal",
         "owner_bone":"tip"}
    ]))?;
    let evaluated = snapshot(&project, None)?;
    assert_vec3_close(
        bone_matrix(&evaluated, "arm", "tip")?.w_axis.truncate(),
        DVec3::new(1.0, 1.0, 0.0),
        1.0e-6,
    );
    assert_vec3_close(
        bone_matrix(&evaluated, "arm", "end")?.w_axis.truncate(),
        DVec3::new(1.0, 2.0, 0.0),
        1.0e-6,
    );
    Ok(())
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one fixture verifies all evaluator-supported pose constraints"
)]
fn pose_bone_constraints_apply_supported_constraint_transforms() -> IntegrationResult {
    #[expect(
        clippy::needless_pass_by_value,
        reason = "test helper takes json! temporaries"
    )]
    fn add_pose_constraint(
        operations: &mut Vec<Value>,
        bone_id: &str,
        constraint_type: &str,
        target: Option<&str>,
        subtarget: Option<&str>,
        params: Value,
    ) {
        operations.push(json!({
            "op":"bone.create","target":{"id":"arm"},"id":bone_id,"name":bone_id,
            "head":[0.0,0.0,0.0],"tail":[0.0,1.0,0.0]
        }));
        operations.push(json!({
            "op":"constraint.create",
            "target":{"id":"arm"},
            "id":format!("constraint_{bone_id}"),
            "type":constraint_type,
            "owner_bone":bone_id,
            "constraint_target":target,
            "subtarget":subtarget,
            "params":params
        }));
    }

    let quarter_turn = std::f64::consts::FRAC_1_SQRT_2;
    let mut operations = vec![
        json!({"op":"node.create","id":"arm","kind":"armature"}),
        json!({"op":"node.create","id":"target_location","kind":"empty",
               "transform":{"translation":[2.0,3.0,4.0]}}),
        json!({"op":"node.create","id":"target_rotation","kind":"empty",
               "transform":{"rotation":[0.0,0.0,quarter_turn,quarter_turn]}}),
        json!({"op":"node.create","id":"target_scale","kind":"empty",
               "transform":{"scale":[2.0,3.0,4.0]}}),
        json!({"op":"node.create","id":"target_x","kind":"empty",
               "transform":{"translation":[3.0,0.0,0.0]}}),
        json!({"op":"node.create","id":"target_z","kind":"empty",
               "transform":{"translation":[0.0,0.0,3.0]}}),
        json!({"op":"node.create","id":"target_y","kind":"empty",
               "transform":{"translation":[0.0,3.0,0.0]}}),
        json!({"op":"node.create","id":"origin","kind":"empty"}),
        json!({"op":"node.create","id":"pivot","kind":"empty",
               "transform":{"translation":[1.0,0.0,0.0],
                            "rotation":[0.0,0.0,quarter_turn,quarter_turn]}}),
        json!({"op":"node.create","id":"copy_transform_target","kind":"empty",
               "transform":{"translation":[4.0,5.0,6.0],"scale":[2.0,3.0,4.0]}}),
        json!({"op":"node.create","id":"surface","kind":"plane","params":{"size":2.0}}),
        json!({"op":"curve.create","id":"path","splines":[{"type":"poly","points":[
            {"co":[0.0,0.0,0.0]},{"co":[2.0,0.0,0.0]},{"co":[2.0,2.0,0.0]}
        ]}]}),
        json!({"op":"node.create","id":"target_arm","kind":"armature"}),
        json!({"op":"bone.create","target":{"id":"target_arm"},"id":"target_bone",
               "name":"Target","head":[2.0,0.0,0.0],"tail":[2.0,1.0,0.0]}),
    ];
    add_pose_constraint(
        &mut operations,
        "copy_location",
        "copy_location",
        Some("target_location"),
        None,
        json!({}),
    );
    add_pose_constraint(
        &mut operations,
        "copy_rotation",
        "copy_rotation",
        Some("target_rotation"),
        None,
        json!({}),
    );
    add_pose_constraint(
        &mut operations,
        "copy_scale",
        "copy_scale",
        Some("target_scale"),
        None,
        json!({}),
    );
    add_pose_constraint(
        &mut operations,
        "track_to",
        "track_to",
        Some("target_x"),
        None,
        json!({"track_axis":"TRACK_Z","up_axis":"UP_Y"}),
    );
    add_pose_constraint(
        &mut operations,
        "damped_track",
        "damped_track",
        Some("target_z"),
        None,
        json!({}),
    );
    add_pose_constraint(
        &mut operations,
        "locked_track",
        "locked_track",
        Some("target_x"),
        None,
        json!({"track_axis":"TRACK_Z","lock_axis":"LOCK_Y"}),
    );
    add_pose_constraint(
        &mut operations,
        "stretch_to",
        "stretch_to",
        Some("target_y"),
        None,
        json!({"rest_length":1.0}),
    );
    add_pose_constraint(
        &mut operations,
        "transformation",
        "transformation",
        Some("target_location"),
        None,
        json!({"map_from":"LOCATION","map_to":"SCALE","map_to_y_from":"X",
               "from_min_x":0.0,"from_max_x":4.0,
               "to_min_y_scale":1.0,"to_max_y_scale":3.0}),
    );
    add_pose_constraint(
        &mut operations,
        "maintain_volume",
        "maintain_volume",
        None,
        None,
        json!({"free_axis":"SAMEVOL_Y","volume":1.0}),
    );
    add_pose_constraint(
        &mut operations,
        "floor",
        "floor",
        Some("origin"),
        None,
        json!({"floor_location":"FLOOR_Z"}),
    );
    add_pose_constraint(
        &mut operations,
        "pivot",
        "pivot",
        Some("pivot"),
        None,
        json!({}),
    );
    add_pose_constraint(
        &mut operations,
        "shrinkwrap",
        "shrinkwrap",
        Some("surface"),
        None,
        json!({}),
    );
    add_pose_constraint(
        &mut operations,
        "limit_location",
        "limit_location",
        None,
        None,
        json!({"min_x":-1.0,"max_x":1.0,"use_min_x":true,"use_max_x":true}),
    );
    add_pose_constraint(
        &mut operations,
        "limit_rotation",
        "limit_rotation",
        None,
        None,
        json!({"min_z":-0.25,"max_z":0.25,"use_limit_z":true}),
    );
    add_pose_constraint(
        &mut operations,
        "limit_scale",
        "limit_scale",
        None,
        None,
        json!({"max_x":1.0,"use_max_x":true}),
    );
    add_pose_constraint(
        &mut operations,
        "child_of",
        "child_of",
        Some("target_location"),
        None,
        json!({}),
    );
    add_pose_constraint(
        &mut operations,
        "armature_constraint",
        "armature",
        Some("target_arm"),
        Some("target_bone"),
        json!({}),
    );
    add_pose_constraint(
        &mut operations,
        "clamp_to",
        "clamp_to",
        Some("path"),
        None,
        json!({}),
    );
    add_pose_constraint(
        &mut operations,
        "copy_transforms",
        "copy_transforms",
        Some("copy_transform_target"),
        None,
        json!({}),
    );
    add_pose_constraint(
        &mut operations,
        "follow_path",
        "follow_path",
        Some("path"),
        None,
        json!({"offset_factor":0.5,"use_fixed_location":true}),
    );
    add_pose_constraint(
        &mut operations,
        "limit_distance",
        "limit_distance",
        Some("origin"),
        None,
        json!({"distance":2.0}),
    );
    for (bone_id, set) in [
        ("maintain_volume", json!({"scale":[1.0,2.0,1.0]})),
        ("floor", json!({"translation":[0.0,0.0,-1.0]})),
        ("pivot", json!({"translation":[2.0,0.0,0.0]})),
        ("shrinkwrap", json!({"translation":[0.0,0.0,2.0]})),
        ("limit_location", json!({"translation":[3.0,0.0,0.0]})),
        (
            "limit_rotation",
            json!({"rotation":[0.0,0.0,0.247_403_959_254_522_94,
                                               0.968_912_421_710_644_7]}),
        ),
        ("limit_scale", json!({"scale":[2.0,1.0,1.0]})),
        ("child_of", json!({"translation":[1.0,0.0,0.0]})),
        ("clamp_to", json!({"translation":[1.0,2.0,0.0]})),
        ("limit_distance", json!({"translation":[4.0,0.0,0.0]})),
    ] {
        operations.push(json!({
            "op":"pose.set","target":{"id":"arm"},"bone_id":bone_id,"set":set
        }));
    }
    let (_directory, project) = new_project(&Value::Array(operations))?;
    let evaluated = snapshot(&project, None)?;
    let copied_location = bone_matrix(&evaluated, "arm", "copy_location")?;
    assert_vec3_close(
        copied_location.w_axis.truncate(),
        DVec3::new(2.0, 3.0, 4.0),
        1.0e-6,
    );
    assert_vec3_close(
        bone_matrix(&evaluated, "arm", "copy_rotation")?.transform_vector3(DVec3::X),
        DVec3::Y,
        1.0e-6,
    );
    assert_vec3_close(
        bone_matrix(&evaluated, "arm", "copy_scale")?
            .to_scale_rotation_translation()
            .0,
        DVec3::new(2.0, 3.0, 4.0),
        1.0e-6,
    );
    assert_vec3_close(
        bone_matrix(&evaluated, "arm", "track_to")?.transform_vector3(DVec3::Z),
        DVec3::X,
        1.0e-6,
    );
    assert_vec3_close(
        bone_matrix(&evaluated, "arm", "damped_track")?.transform_vector3(DVec3::Y),
        DVec3::Z,
        1.0e-6,
    );
    assert_vec3_close(
        bone_matrix(&evaluated, "arm", "locked_track")?.transform_vector3(DVec3::Z),
        DVec3::X,
        1.0e-6,
    );
    let stretch_scale = bone_matrix(&evaluated, "arm", "stretch_to")?
        .to_scale_rotation_translation()
        .0;
    assert!((stretch_scale.y - 3.0).abs() < 1.0e-6);
    let transformation_scale = bone_matrix(&evaluated, "arm", "transformation")?
        .to_scale_rotation_translation()
        .0;
    assert!((transformation_scale.y - 2.0).abs() < 1.0e-6);
    let volume_scale = bone_matrix(&evaluated, "arm", "maintain_volume")?
        .to_scale_rotation_translation()
        .0;
    assert!((volume_scale.x * volume_scale.y * volume_scale.z - 1.0).abs() < 1.0e-6);
    assert!(bone_matrix(&evaluated, "arm", "floor")?.w_axis.z.abs() < 1.0e-6);
    assert_vec3_close(
        bone_matrix(&evaluated, "arm", "pivot")?.w_axis.truncate(),
        DVec3::new(2.0, 0.0, 0.0),
        1.0e-6,
    );
    assert!(bone_matrix(&evaluated, "arm", "shrinkwrap")?.w_axis.z.abs() < 1.0e-6);
    assert!((bone_matrix(&evaluated, "arm", "limit_location")?.w_axis.x - 1.0).abs() < 1.0e-6);
    let limited_rotation = bone_matrix(&evaluated, "arm", "limit_rotation")?
        .to_scale_rotation_translation()
        .1
        .to_euler(glam::EulerRot::XYZ);
    assert!((limited_rotation.2 - 0.25).abs() < 1.0e-6);
    assert!(
        (bone_matrix(&evaluated, "arm", "limit_scale")?
            .x_axis
            .length()
            - 1.0)
            .abs()
            < 1e-6
    );
    assert_vec3_close(
        bone_matrix(&evaluated, "arm", "child_of")?
            .w_axis
            .truncate(),
        DVec3::new(3.0, 3.0, 4.0),
        1.0e-6,
    );
    assert_vec3_close(
        bone_matrix(&evaluated, "arm", "armature_constraint")?
            .w_axis
            .truncate(),
        DVec3::new(2.0, 0.0, 0.0),
        1.0e-6,
    );
    assert_vec3_close(
        bone_matrix(&evaluated, "arm", "clamp_to")?
            .w_axis
            .truncate(),
        DVec3::new(2.0, 0.0, 0.0),
        1.0e-6,
    );
    let copied_transforms = bone_matrix(&evaluated, "arm", "copy_transforms")?;
    assert_vec3_close(
        copied_transforms.w_axis.truncate(),
        DVec3::new(4.0, 5.0, 6.0),
        1.0e-6,
    );
    assert_vec3_close(
        copied_transforms.to_scale_rotation_translation().0,
        DVec3::new(2.0, 3.0, 4.0),
        1.0e-6,
    );
    assert_vec3_close(
        bone_matrix(&evaluated, "arm", "follow_path")?
            .w_axis
            .truncate(),
        DVec3::new(2.0, 0.0, 0.0),
        1.0e-6,
    );
    assert_vec3_close(
        bone_matrix(&evaluated, "arm", "limit_distance")?
            .w_axis
            .truncate(),
        DVec3::new(2.0, 0.0, 0.0),
        1.0e-6,
    );
    Ok(())
}

#[test]
fn pose_bone_constraint_targets_are_converted_to_armature_space() -> IntegrationResult {
    let quarter_turn = std::f64::consts::FRAC_1_SQRT_2;
    let (_directory, project) = new_project(&json!([
        {"op":"node.create","id":"arm","kind":"armature",
         "transform":{"rotation":[0.0,0.0,quarter_turn,quarter_turn]}},
        {"op":"bone.create","target":{"id":"arm"},"id":"tip","name":"Tip",
         "head":[0.0,0.0,0.0],"tail":[0.0,1.0,0.0]},
        {"op":"node.create","id":"goal","kind":"empty",
         "transform":{"translation":[2.0,2.0,0.0]}},
        {"op":"constraint.create","target":{"id":"arm"},"id":"copy_x",
         "type":"copy_location","constraint_target":"goal","owner_bone":"tip",
         "params":{"use_x":true,"use_y":false,"use_z":false}}
    ]))?;
    let evaluated = snapshot(&project, None)?;
    assert_vec3_close(
        bone_matrix(&evaluated, "arm", "tip")?.w_axis.truncate(),
        DVec3::new(2.0, 0.0, 0.0),
        1.0e-6,
    );
    Ok(())
}

#[test]
fn constraint_update_reassigns_pose_bone_owner_and_rejects_missing_bones() -> IntegrationResult {
    let (_directory, project) = new_project(&json!([
        {"op":"node.create","id":"arm","kind":"armature"},
        {"op":"bone.create","target":{"id":"arm"},"id":"root","name":"Root",
         "head":[0.0,0.0,0.0],"tail":[0.0,1.0,0.0]},
        {"op":"bone.create","target":{"id":"arm"},"id":"tip","name":"Tip",
         "head":[0.0,1.0,0.0],"tail":[0.0,2.0,0.0],"parent":"root"},
        {"op":"node.create","id":"goal","kind":"empty",
         "transform":{"translation":[1.0,0.0,0.0]}},
        {"op":"constraint.create","target":{"id":"arm"},"id":"move_tip",
         "type":"copy_location","constraint_target":"goal","owner_bone":"tip"}
    ]))?;
    let document: SceneDoc = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let updated = potter::ops::apply_batch(
        &document,
        &json!({
            "schema_version":1,
            "base_revision":document.revision,
            "operations":[{
                "op":"constraint.update","target":{"id":"arm"},"id":"move_tip",
                "set":{"owner_bone":"root"}
            }]
        }),
    )?;
    let arm_id = Id::new("arm")?;
    let constraint_id = Id::new("move_tip")?;
    let constraint = updated
        .doc
        .nodes
        .get(&arm_id)
        .and_then(|node| {
            node.constraints
                .iter()
                .find(|item| item.id == constraint_id)
        })
        .ok_or_else(|| io::Error::other("updated pose-bone constraint is missing"))?;
    assert_eq!(constraint.owner_bone, Some(Id::new("root")?));

    let Err(error) = potter::ops::apply_batch(
        &updated.doc,
        &json!({
            "schema_version":1,
            "base_revision":updated.doc.revision,
            "operations":[{
                "op":"constraint.update","target":{"id":"arm"},"id":"move_tip",
                "set":{"owner_bone":"missing"}
            }]
        }),
    ) else {
        return Err(io::Error::other("constraint accepted a missing owner bone").into());
    };
    assert_eq!(error.code, potter::error::ErrorCode::TargetNotFound);
    Ok(())
}

#[test]
fn spline_ik_distributes_chain_over_a_poly_curve() -> IntegrationResult {
    let (_directory, project) = new_project(&json!([
        {"op":"node.create","id":"arm","kind":"armature"},
        {"op":"bone.create","target":{"id":"arm"},"id":"root","name":"Root","head":[0.0,0.0,0.0],"tail":[0.0,1.0,0.0]},
        {"op":"bone.create","target":{"id":"arm"},"id":"tip","name":"Tip","head":[0.0,1.0,0.0],"tail":[0.0,2.0,0.0],"parent":"root"},
        {"op":"curve.create","id":"path","splines":[{"type":"poly","points":[{"co":[0.0,0.0,0.0]},{"co":[1.0,1.0,0.0]},{"co":[2.0,1.0,0.0]}]}]},
        {"op":"constraint.create","target":{"id":"arm"},"id":"follow","type":"spline_ik","constraint_target":"path","owner_bone":"tip","params":{"chain_count":2}}
    ]))?;
    let evaluated = snapshot(&project, None)?;
    assert_vec3_close(
        bone_endpoint(&evaluated, "arm", "tip", 1.0)?,
        DVec3::new(2.0, 1.0, 0.0),
        1.0e-6,
    );
    Ok(())
}

#[test]
fn copy_rotation_scale_damped_track_limits_and_child_of_are_evaluated() -> IntegrationResult {
    let quarter_turn = std::f64::consts::FRAC_1_SQRT_2;
    let (_directory, project) = new_project(&json!([
        {"op":"node.create","id":"rotation_target","kind":"empty","transform":{"rotation":[0.0,0.0,quarter_turn,quarter_turn]}},
        {"op":"node.create","id":"rotation_copy","kind":"empty"},
        {"op":"constraint.create","target":{"id":"rotation_copy"},"id":"copy_rotation","type":"copy_rotation","constraint_target":"rotation_target"},
        {"op":"node.create","id":"scale_target","kind":"empty","transform":{"scale":[2.0,3.0,4.0]}},
        {"op":"node.create","id":"scale_copy","kind":"empty"},
        {"op":"constraint.create","target":{"id":"scale_copy"},"id":"copy_scale","type":"copy_scale","constraint_target":"scale_target"},
        {"op":"node.create","id":"track_target","kind":"empty","transform":{"translation":[0.0,0.0,3.0]}},
        {"op":"node.create","id":"damped","kind":"empty"},
        {"op":"constraint.create","target":{"id":"damped"},"id":"damped_track","type":"damped_track","constraint_target":"track_target"},
        {"op":"node.create","id":"limited","kind":"empty","transform":{"translation":[3.0,0.0,0.0],"rotation":[0.0,0.0,0.479_425_538_604_203,0.877_582_561_890_372_8],"scale":[2.0,2.0,2.0]}},
        {"op":"constraint.create","target":{"id":"limited"},"id":"location_limit","type":"limit_location","params":{"min_x":-1.0,"max_x":1.0,"use_min_x":true,"use_max_x":true}},
        {"op":"constraint.create","target":{"id":"limited"},"id":"scale_limit","type":"limit_scale","params":{"min_x":0.5,"max_x":1.0,"use_min_x":true,"use_max_x":true}},
        {"op":"constraint.create","target":{"id":"limited"},"id":"rotation_limit","type":"limit_rotation","params":{"min_z":-0.25,"max_z":0.25,"use_limit_z":true}},
        {"op":"node.create","id":"child_target","kind":"empty","transform":{"translation":[3.0,0.0,0.0]}},
        {"op":"node.create","id":"child_of_owner","kind":"empty","transform":{"translation":[1.0,0.0,0.0]}},
        {"op":"constraint.create","target":{"id":"child_of_owner"},"id":"child_of","type":"child_of","constraint_target":"child_target"}
    ]))?;
    let evaluated = snapshot(&project, None)?;
    assert_vec3_close(
        node_matrix(&evaluated, "rotation_copy")?.transform_vector3(DVec3::X),
        DVec3::Y,
        1.0e-6,
    );
    let copied_scale = node_matrix(&evaluated, "scale_copy")?
        .to_scale_rotation_translation()
        .0;
    assert_vec3_close(copied_scale, DVec3::new(2.0, 3.0, 4.0), 1.0e-6);
    assert_vec3_close(
        node_matrix(&evaluated, "damped")?.transform_vector3(DVec3::Y),
        DVec3::Z,
        1.0e-6,
    );
    let limited = node_matrix(&evaluated, "limited")?;
    assert!((limited.w_axis.x - 1.0).abs() < 1.0e-12);
    assert!((limited.to_scale_rotation_translation().0.x - 1.0).abs() < 1.0e-12);
    let limited_rotation = limited
        .to_scale_rotation_translation()
        .1
        .to_euler(glam::EulerRot::XYZ);
    assert!((limited_rotation.2 - 0.25).abs() < 1.0e-6);
    assert_vec3_close(
        node_matrix(&evaluated, "child_of_owner")?.w_axis.truncate(),
        DVec3::new(4.0, 0.0, 0.0),
        1.0e-12,
    );
    Ok(())
}

#[test]
fn copy_transforms_limit_distance_follow_path_clamp_to_and_armature_constraints_evaluate()
-> IntegrationResult {
    let (_directory, project) = new_project(&json!([
        {"op":"node.create","id":"copy_target","kind":"empty","transform":{"translation":[4.0,5.0,6.0],"scale":[2.0,3.0,4.0]}},
        {"op":"node.create","id":"copy_owner","kind":"empty"},
        {"op":"constraint.create","target":{"id":"copy_owner"},"id":"copy","type":"copy_transforms","constraint_target":"copy_target"},
        {"op":"node.create","id":"limit_target","kind":"empty"},
        {"op":"node.create","id":"limit_owner","kind":"empty","transform":{"translation":[4.0,0.0,0.0]}},
        {"op":"constraint.create","target":{"id":"limit_owner"},"id":"distance","type":"limit_distance","constraint_target":"limit_target","params":{"distance":2.0}},
        {"op":"curve.create","id":"route","splines":[{"type":"poly","points":[{"co":[0.0,0.0,0.0]},{"co":[2.0,0.0,0.0]},{"co":[2.0,2.0,0.0]}]}]},
        {"op":"node.create","id":"follow_owner","kind":"empty"},
        {"op":"constraint.create","target":{"id":"follow_owner"},"id":"follow","type":"follow_path","constraint_target":"route","params":{"offset_factor":0.5,"use_fixed_location":true}},
        {"op":"node.create","id":"clamp_owner","kind":"empty","transform":{"translation":[1.0,2.0,0.0]}},
        {"op":"constraint.create","target":{"id":"clamp_owner"},"id":"clamp","type":"clamp_to","constraint_target":"route"},
        {"op":"node.create","id":"arm","kind":"armature"},
        {"op":"bone.create","target":{"id":"arm"},"id":"root","name":"Root","head":[0.0,0.0,0.0],"tail":[0.0,1.0,0.0]},
        {"op":"pose.set","target":{"id":"arm"},"bone_id":"root","set":{"translation":[0.0,2.0,0.0]}},
        {"op":"node.create","id":"armature_owner","kind":"empty"},
        {"op":"constraint.create","target":{"id":"armature_owner"},"id":"armature","type":"armature","constraint_target":"arm","subtarget":"root"}
    ]))?;
    let evaluated = snapshot(&project, None)?;
    let copy = node_matrix(&evaluated, "copy_owner")?;
    assert_vec3_close(copy.w_axis.truncate(), DVec3::new(4.0, 5.0, 6.0), 1.0e-12);
    assert_vec3_close(
        copy.to_scale_rotation_translation().0,
        DVec3::new(2.0, 3.0, 4.0),
        1.0e-12,
    );
    assert_vec3_close(
        node_matrix(&evaluated, "limit_owner")?.w_axis.truncate(),
        DVec3::new(2.0, 0.0, 0.0),
        1.0e-12,
    );
    assert_vec3_close(
        node_matrix(&evaluated, "follow_owner")?.w_axis.truncate(),
        DVec3::new(2.0, 0.0, 0.0),
        1.0e-12,
    );
    assert_vec3_close(
        node_matrix(&evaluated, "clamp_owner")?.w_axis.truncate(),
        DVec3::new(2.0, 0.0, 0.0),
        1.0e-12,
    );
    assert_vec3_close(
        node_matrix(&evaluated, "armature_owner")?.w_axis.truncate(),
        DVec3::new(0.0, 2.0, 0.0),
        1.0e-12,
    );
    Ok(())
}

#[test]
fn spline_ik_evaluates_nurbs_curve_targets() -> IntegrationResult {
    let (_directory, project) = new_project(&json!([
        {"op":"node.create","id":"arm","kind":"armature"},
        {"op":"bone.create","target":{"id":"arm"},"id":"root","name":"Root","head":[0.0,0.0,0.0],"tail":[0.0,1.0,0.0]},
        {"op":"curve.create","id":"path","splines":[{"type":"nurbs","order":2,"resolution":2,"use_endpoint":true,"points":[{"co":[0.0,0.0,0.0]},{"co":[1.0,0.0,0.0]},{"co":[2.0,0.0,0.0]}]}]},
        {"op":"constraint.create","target":{"id":"arm"},"id":"follow","type":"spline_ik","constraint_target":"path","owner_bone":"root","params":{"chain_count":1}}
    ]))?;
    let evaluated = snapshot(&project, None)?;
    assert_vec3_close(
        bone_endpoint(&evaluated, "arm", "root", 1.0)?,
        DVec3::new(2.0, 0.0, 0.0),
        1.0e-6,
    );
    Ok(())
}

#[test]
fn spline_ik_uses_first_curve_spline() -> IntegrationResult {
    let (_directory, project) = new_project(&json!([
        {"op":"node.create","id":"arm","kind":"armature"},
        {"op":"bone.create","target":{"id":"arm"},"id":"root","name":"Root","head":[0.0,0.0,0.0],"tail":[0.0,1.0,0.0]},
        {"op":"curve.create","id":"path","splines":[{"type":"poly","points":[{"co":[0.0,0.0,0.0]},{"co":[2.0,0.0,0.0]}]},{"type":"poly","points":[{"co":[0.0,2.0,0.0]},{"co":[2.0,2.0,0.0]}]}]},
        {"op":"constraint.create","target":{"id":"arm"},"id":"follow","type":"spline_ik","constraint_target":"path","owner_bone":"root"}
    ]))?;
    let evaluated = snapshot(&project, None)?;
    assert_vec3_close(
        bone_endpoint(&evaluated, "arm", "root", 1.0)?,
        DVec3::new(2.0, 0.0, 0.0),
        1.0e-6,
    );
    Ok(())
}

fn as_case<T>(result: IntegrationResult<T>) -> Result<T, TestCaseError> {
    result.map_err(|error| TestCaseError::fail(error.to_string()))
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 8, .. ProptestConfig::default() })]

    #[test]
    fn identity_pose_lbs_preserves_all_fully_weighted_vertices(side in 0.25_f64..5.0) {
        let (_directory, project) = as_case(new_project(&json!([
            {"op":"node.create","id":"mesh","kind":"plane","params":{"size":side}}
        ])))?;
        let rest = as_case(snapshot(&project, None))?;
        let rest_vertices = as_case(mesh_vertices(&rest, "mesh"))?;
        let weights = (0_u32..4)
            .map(|vertex_id| json!({"vertex_id":vertex_id,"weight":1.0}))
            .collect::<Vec<_>>();
        as_case(apply_batch(
            &project,
            1,
            &json!([
                {"op":"node.create","id":"arm","kind":"armature"},
                {"op":"bone.create","target":{"id":"arm"},"id":"root","name":"Root","head":[0.0,0.0,0.0],"tail":[0.0,1.0,0.0]},
                {"op":"vertex_group.create","target":{"id":"mesh"},"id":"root","name":"Root"},
                {"op":"vertex_group.assign","target":{"id":"mesh"},"group_id":"root","weights":weights},
                {"op":"modifier.create","target":{"id":"mesh"},"id":"skin","type":"armature","params":{"object":"arm","use_vertex_groups":true}}
            ]),
        ))?;
        let evaluated = as_case(snapshot(&project, None))?;
        for (vertex_id, rest_position) in rest_vertices {
            let position = as_case(mesh_vertex(&evaluated, "mesh", vertex_id))?;
            prop_assert!(
                (position - rest_position).length() <= 1.0e-10,
                "identity-pose skinning moved vertex {vertex_id}: {rest_position:?} -> {position:?}"
            );
        }
    }

    #[test]
    fn two_bone_ik_endpoint_stays_within_chain_length(
        angle in 0.0_f64..std::f64::consts::TAU,
        radius in 0.2_f64..1.9
    ) {
        let target = [radius * angle.cos(), radius * angle.sin(), 0.0];
        let (_directory, project) = as_case(new_project(&armature_chain_operations(target)))?;
        let evaluated = as_case(snapshot(&project, None))?;
        let endpoint = as_case(bone_endpoint(&evaluated, "arm", "tip", 1.0))?;
        prop_assert!(endpoint.length() <= 2.0 + 1.0e-6);
    }

    #[test]
    fn two_bone_ik_reaches_targets_outside_blenders_singular_axis_band(
        angle in 0.0_f64..std::f64::consts::TAU,
        radius_index in 0_usize..BLENDER_IK_PROBE_RADII.len()
    ) {
        let radius = BLENDER_IK_PROBE_RADII[radius_index];
        let target = [radius * angle.cos(), radius * angle.sin(), 0.0];
        let (_directory, project) = as_case(new_project(&armature_chain_operations(target)))?;
        let evaluated = as_case(snapshot(&project, None))?;
        let endpoint = as_case(bone_endpoint(&evaluated, "arm", "tip", 1.0))?;
        let target_position = DVec3::from_array(target);
        // Blender's grid fails inside this measured band and reaches just beyond it.
        if !blender_ik_failed_near_axis(angle, radius_index) {
            prop_assert!(
                (endpoint - target_position).length() <= 1.0e-3,
                "reachable IK target {target_position:?} ended at {endpoint:?}"
            );
        }
        prop_assert!(endpoint.length() <= 2.0 + 1.0e-6);
    }
}
