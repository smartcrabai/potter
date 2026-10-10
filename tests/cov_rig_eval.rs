#![expect(
    clippy::expect_used,
    reason = "coverage regression fixtures assert expected evaluation failures directly"
)]
#![expect(
    clippy::too_many_lines,
    reason = "each parity fixture keeps its inputs and output checks together"
)]

#[path = "common/blender_file.rs"]
mod blender_file;

use std::{error::Error, fs, io, path::Path, process::Command};

use glam::{DMat4, DQuat, EulerRot};
use potter_core::{
    eval::{EvaluationContext, Snapshot},
    model::{Id, ReconstructedPoint, SceneDoc, TrackingMarker},
    ops,
};
use serde_json::{Value, json};
use tempfile::tempdir;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

#[derive(Clone, Copy)]
struct SpaceCase {
    id: &'static str,
    owner_kind: &'static str,
    owner_space: &'static str,
    target_space: &'static str,
    influence: f64,
    enabled: bool,
}

const SPACE_CASES: [SpaceCase; 9] = [
    SpaceCase {
        id: "object_local",
        owner_kind: "object",
        owner_space: "LOCAL",
        target_space: "LOCAL",
        influence: 1.0,
        enabled: true,
    },
    SpaceCase {
        id: "object_custom",
        owner_kind: "object",
        owner_space: "CUSTOM",
        target_space: "CUSTOM",
        influence: 1.0,
        enabled: true,
    },
    SpaceCase {
        id: "pose_local",
        owner_kind: "pose",
        owner_space: "LOCAL",
        target_space: "LOCAL",
        influence: 1.0,
        enabled: true,
    },
    SpaceCase {
        id: "pose_parent",
        owner_kind: "pose",
        owner_space: "LOCAL_WITH_PARENT",
        target_space: "LOCAL_WITH_PARENT",
        influence: 1.0,
        enabled: true,
    },
    SpaceCase {
        id: "pose_pose",
        owner_kind: "pose",
        owner_space: "POSE",
        target_space: "POSE",
        influence: 1.0,
        enabled: true,
    },
    SpaceCase {
        id: "pose_custom",
        owner_kind: "pose",
        owner_space: "CUSTOM",
        target_space: "CUSTOM",
        influence: 1.0,
        enabled: true,
    },
    SpaceCase {
        id: "object_blend",
        owner_kind: "object",
        owner_space: "WORLD",
        target_space: "WORLD",
        influence: 0.35,
        enabled: true,
    },
    SpaceCase {
        id: "object_zero",
        owner_kind: "object",
        owner_space: "WORLD",
        target_space: "WORLD",
        influence: 0.0,
        enabled: true,
    },
    SpaceCase {
        id: "object_muted",
        owner_kind: "object",
        owner_space: "WORLD",
        target_space: "WORLD",
        influence: 1.0,
        enabled: false,
    },
];

fn transform(translation: [f64; 3], rotation: [f64; 3], scale: [f64; 3]) -> Value {
    let quaternion = DQuat::from_euler(EulerRot::XYZ, rotation[0], rotation[1], rotation[2]);
    json!({
        "translation": translation,
        "rotation": [quaternion.x, quaternion.y, quaternion.z, quaternion.w],
        "scale": scale,
    })
}

fn space_fixture() -> Value {
    json!({
        "owner_parent": transform([1.1, -0.4, 0.7], [0.19, -0.27, 0.43], [1.0, 1.0, 1.0]),
        "target_parent": transform([-0.6, 1.3, -0.2], [-0.31, 0.22, -0.18], [1.0, 1.0, 1.0]),
        "space": transform([0.4, 0.8, -0.9], [0.23, 0.39, -0.16], [1.0, 1.0, 1.0]),
        "owner_local": transform([0.7, -0.2, 1.0], [-0.13, 0.34, 0.21], [1.0, 1.0, 1.0]),
        "target_local": transform([-0.3, 0.9, 0.6], [0.32, -0.11, 0.27], [1.0, 1.0, 1.0]),
        "pose_owner": transform([-0.7, 0.5, 1.2], [0.14, -0.28, 0.37], [1.0, 1.0, 1.0]),
        "pose_target": transform([0.2, -1.0, 0.4], [-0.25, 0.17, -0.36], [1.0, 1.0, 1.0]),
        "cases": SPACE_CASES.iter().map(|case| json!({
            "id": case.id,
            "owner_kind": case.owner_kind,
            "owner_space": case.owner_space,
            "target_space": case.target_space,
            "influence": case.influence,
            "enabled": case.enabled,
        })).collect::<Vec<_>>(),
    })
}

fn constraint_space_doc(fixture: &Value) -> TestResult<SceneDoc> {
    let mut operations = vec![
        json!({"op":"node.create","id":"owner_parent","kind":"empty","transform":fixture["owner_parent"]}),
        json!({"op":"node.create","id":"target_parent","kind":"empty","transform":fixture["target_parent"]}),
        json!({"op":"node.create","id":"space_ref","kind":"empty","transform":fixture["space"]}),
    ];
    for case in SPACE_CASES {
        let owner_id = if case.owner_kind == "pose" {
            format!("owner_arm_{}", case.id)
        } else {
            format!("owner_{}", case.id)
        };
        let target_id = if case.owner_kind == "pose" {
            format!("target_arm_{}", case.id)
        } else {
            format!("target_{}", case.id)
        };
        if case.owner_kind == "pose" {
            let owner_bone = format!("owner_bone_{}", case.id);
            let target_bone = format!("target_bone_{}", case.id);
            operations.push(json!({"op":"node.create","id":owner_id,"kind":"armature","transform":fixture["pose_owner"]}));
            operations.push(json!({"op":"bone.create","target":{"id":owner_id},"id":owner_bone,"name":"OwnerBone","head":[0.2,-0.3,0.5],"tail":[1.2,0.8,0.7]}));
            operations.push(json!({"op":"node.create","id":target_id,"kind":"armature","transform":fixture["pose_target"]}));
            operations.push(json!({"op":"bone.create","target":{"id":target_id},"id":target_bone,"name":"TargetBone","head":[0.2,0.1,-0.1],"tail":[0.2,1.4,0.3]}));
            operations.push(json!({
                "op":"constraint.create","target":{"id":owner_id},"owner_bone":owner_bone,
                "id":format!("constraint_{}", case.id),"type":"copy_location",
                "constraint_target":target_id,"subtarget":target_bone,
                "influence":case.influence,"enabled":case.enabled,
                "params":{"owner_space":case.owner_space,"target_space":case.target_space,"space_object":"space_ref"}
            }));
        } else {
            operations.push(json!({"op":"node.create","id":owner_id,"kind":"empty","parent":"owner_parent","transform":fixture["owner_local"]}));
            operations.push(json!({"op":"node.create","id":target_id,"kind":"empty","parent":"target_parent","transform":fixture["target_local"]}));
            operations.push(json!({
                "op":"constraint.create","target":{"id":owner_id},
                "id":format!("constraint_{}", case.id),"type":"copy_location",
                "constraint_target":target_id,"influence":case.influence,"enabled":case.enabled,
                "params":{"owner_space":case.owner_space,"target_space":case.target_space,"space_object":"space_ref"}
            }));
        }
    }
    Ok(ops::apply_batch(
        &SceneDoc::new("00000000-0000-4000-8000-000000000002".to_owned()),
        &json!({"schema_version":1,"base_revision":0,"operations":operations}),
    )?
    .doc)
}

fn node_matrix(snapshot: &Snapshot, node: &str) -> TestResult<DMat4> {
    let id = Id::new(node.to_owned())?;
    snapshot
        .nodes
        .get(&id)
        .map(|evaluated| DMat4::from_cols_array(&evaluated.world_matrix))
        .ok_or_else(|| io::Error::other(format!("evaluated node `{node}` is missing")).into())
}

fn bone_matrix(snapshot: &Snapshot, armature: &str, bone: &str) -> TestResult<DMat4> {
    let armature = Id::new(armature.to_owned())?;
    let bone = Id::new(bone.to_owned())?;
    snapshot
        .bone_matrices
        .get(&armature)
        .and_then(|matrices| matrices.get(&bone))
        .map(DMat4::from_cols_array)
        .ok_or_else(|| io::Error::other(format!("evaluated pose bone `{bone}` is missing")).into())
}

fn matrix_from_json(value: &Value) -> TestResult<DMat4> {
    let values = value
        .as_array()
        .filter(|values| values.len() == 16)
        .ok_or_else(|| io::Error::other("Blender matrix must contain 16 values"))?;
    let mut matrix = [0.0; 16];
    for (destination, value) in matrix.iter_mut().zip(values) {
        *destination = value
            .as_f64()
            .ok_or_else(|| io::Error::other("Blender matrix value is not numeric"))?;
    }
    Ok(DMat4::from_cols_array(&matrix))
}

fn assert_matrix_parity(actual: DMat4, expected: DMat4, context: &str) {
    let actual_values = actual.to_cols_array();
    let expected_values = expected.to_cols_array();
    for (component, (actual_component, expected_component)) in
        actual_values.iter().zip(expected_values).enumerate()
    {
        let error = (actual_component - expected_component).abs();
        assert!(
            error < 1.0e-5,
            "{context} component {component} differs by {error:.8e}: Potter={actual_component:?}, Blender={expected_component:?}; full Potter={actual_values:?}, Blender={expected_values:?}"
        );
    }
}

fn run_blender(blender: &Path, script: &str, args: &[&str], directory: &Path) -> TestResult {
    let script_path = directory.join("rig_eval_fixture.py");
    fs::write(&script_path, script)?;
    let output = Command::new(blender)
        .args([
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
        ])
        .arg(script_path)
        .args(["--"])
        .args(args)
        .output()?;
    assert!(
        output.status.success(),
        "Blender fixture failed: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

#[test]
fn object_and_pose_constraint_spaces_blend_and_mute_like_blender() -> TestResult {
    let Some(blender) = blender_file::blender_executable() else {
        eprintln!("skipping rig-space Blender parity: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = fs::canonicalize(directory.path())?;
    let fixture = space_fixture();
    let fixture_path = root.join("space_cases.json");
    let output_path = root.join("space_expected.json");
    fs::write(&fixture_path, serde_json::to_vec(&fixture)?)?;
    let script = r#"
import bpy, json, os, sys
from mathutils import Quaternion
if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"space parity requires Blender 5.2.2, found {bpy.app.version_string}")
fixture_path, output_path = sys.argv[sys.argv.index("--") + 1:sys.argv.index("--") + 3]
with open(fixture_path, encoding="utf-8") as source:
    fixture = json.load(source)
scene = bpy.context.scene

def set_transform(obj, value):
    obj.location = value["translation"]
    obj.rotation_mode = "QUATERNION"
    rotation = value["rotation"]
    obj.rotation_quaternion = Quaternion((rotation[3], rotation[0], rotation[1], rotation[2]))
    obj.scale = value["scale"]

def make_empty(name, value, parent=None):
    obj = bpy.data.objects.new(name, None)
    scene.collection.objects.link(obj)
    if parent is not None:
        obj.parent = parent
    set_transform(obj, value)
    return obj

def make_armature(name, bone_name, value, head, tail):
    bpy.ops.object.select_all(action="DESELECT")
    data = bpy.data.armatures.new(name + "Data")
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bone = data.edit_bones.new(bone_name)
    bone.head = head
    bone.tail = tail
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.select_set(False)
    set_transform(obj, value)
    return obj

owner_parent = make_empty("OwnerParent", fixture["owner_parent"])
target_parent = make_empty("TargetParent", fixture["target_parent"])
space_ref = make_empty("SpaceReference", fixture["space"])
owners = {}
for case in fixture["cases"]:
    name = case["id"]
    if case["owner_kind"] == "pose":
        owner = make_armature("Owner_" + name, "OwnerBone", fixture["pose_owner"], (0.2,-0.3,0.5), (1.2,0.8,0.7))
        target = make_armature("Target_" + name, "TargetBone", fixture["pose_target"], (0.2,0.1,-0.1), (0.2,1.4,0.3))
        constraint = owner.pose.bones["OwnerBone"].constraints.new("COPY_LOCATION")
        constraint.subtarget = "TargetBone"
    else:
        owner = make_empty("Owner_" + name, fixture["owner_local"], owner_parent)
        target = make_empty("Target_" + name, fixture["target_local"], target_parent)
        constraint = owner.constraints.new("COPY_LOCATION")
    constraint.target = target
    constraint.owner_space = case["owner_space"]
    constraint.target_space = case["target_space"]
    constraint.influence = case["influence"]
    constraint.mute = not case["enabled"]
    if "CUSTOM" in (case["owner_space"], case["target_space"]):
        constraint.space_object = space_ref
    owners[name] = owner
scene.frame_set(1)
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
expected = {}
for case in fixture["cases"]:
    owner = owners[case["id"]].evaluated_get(depsgraph)
    if case["owner_kind"] == "pose":
        matrix = owner.matrix_world @ owner.pose.bones["OwnerBone"].matrix
    else:
        matrix = owner.matrix_world
    expected[case["id"]] = [float(matrix[row][column]) for column in range(4) for row in range(4)]
with open(output_path, "w", encoding="utf-8") as output:
    json.dump(expected, output)
"#;
    run_blender(
        &blender,
        script,
        &[
            fixture_path.to_str().ok_or("fixture path is not UTF-8")?,
            output_path.to_str().ok_or("output path is not UTF-8")?,
        ],
        &root,
    )?;
    assert!(
        output_path.is_file(),
        "Blender did not create the space matrix output at {output_path:?}"
    );
    let blender_expected: Value = serde_json::from_slice(&fs::read(output_path)?)?;
    let doc = constraint_space_doc(&fixture)?;
    let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
    for case in SPACE_CASES {
        let actual = if case.owner_kind == "pose" {
            bone_matrix(
                &snapshot,
                &format!("owner_arm_{}", case.id),
                &format!("owner_bone_{}", case.id),
            )?
        } else {
            node_matrix(&snapshot, &format!("owner_{}", case.id))?
        };
        let expected = matrix_from_json(&blender_expected[case.id])?;
        assert_matrix_parity(actual, expected, case.id);
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct FollowCase {
    id: &'static str,
    clip: &'static str,
    active: bool,
    use_3d: bool,
    undistorted: bool,
    frame_method: &'static str,
    camera: &'static str,
}

const FOLLOW_CASES: [FollowCase; 9] = [
    FollowCase {
        id: "active_undistorted",
        clip: "active",
        active: true,
        use_3d: false,
        undistorted: true,
        frame_method: "STRETCH",
        camera: "perspective",
    },
    FollowCase {
        id: "active_raw",
        clip: "active",
        active: true,
        use_3d: false,
        undistorted: false,
        frame_method: "STRETCH",
        camera: "perspective",
    },
    FollowCase {
        id: "active_fit",
        clip: "active",
        active: true,
        use_3d: false,
        undistorted: false,
        frame_method: "FIT",
        camera: "perspective",
    },
    FollowCase {
        id: "active_crop",
        clip: "active",
        active: true,
        use_3d: false,
        undistorted: false,
        frame_method: "CROP",
        camera: "perspective",
    },
    FollowCase {
        id: "active_3d",
        clip: "active",
        active: true,
        use_3d: true,
        undistorted: false,
        frame_method: "STRETCH",
        camera: "perspective",
    },
    FollowCase {
        id: "explicit_raw",
        clip: "explicit",
        active: false,
        use_3d: false,
        undistorted: false,
        frame_method: "FIT",
        camera: "perspective",
    },
    FollowCase {
        id: "explicit_undistorted",
        clip: "explicit",
        active: false,
        use_3d: false,
        undistorted: true,
        frame_method: "FIT",
        camera: "perspective",
    },
    FollowCase {
        id: "active_ortho",
        clip: "active",
        active: true,
        use_3d: false,
        undistorted: false,
        frame_method: "CROP",
        camera: "orthographic",
    },
    FollowCase {
        id: "active_depth_hit",
        clip: "active",
        active: true,
        use_3d: false,
        undistorted: false,
        frame_method: "STRETCH",
        camera: "perspective",
    },
];

fn follow_doc(expected: &Value) -> TestResult<SceneDoc> {
    let active_width = 160_u32;
    let active_height = 90_u32;
    let explicit_width = 80_u32;
    let explicit_height = 120_u32;
    let mut operations = vec![
        json!({"op":"tracking.clip_create","id":"clip_active","name":"Active","width":active_width,"height":active_height}),
        json!({"op":"tracking.clip_create","id":"clip_explicit","name":"Explicit","width":explicit_width,"height":explicit_height}),
        json!({"op":"scene.update","target":{"id":"scene_main"},"set":{"active_clip":"clip_active"}}),
        json!({"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_x":160,"resolution_y":100}}),
        json!({"op":"camera.create","id":"camera","name":"Perspective","projection":"perspective","lens_mm":50.0,"sensor_width_mm":36.0}),
        json!({"op":"camera.create","id":"ortho_camera","name":"Orthographic","projection":"orthographic","ortho_scale":3.5}),
        json!({"op":"tracking.track_add","id":"clip_active","track":"active_track","name":"Active Track","frame":1,"co":[80.0,45.0]}),
        json!({"op":"node.create","id":"depth","kind":"plane","params":{"size":2.0},"transform":{"translation":[0.0,0.0,-2.0],"scale":[10.0,10.0,1.0]}}),
        json!({"op":"tracking.track_add","id":"clip_explicit","track":"explicit_track","name":"Explicit Track","frame":1,"co":[40.0,60.0]}),
    ];
    for case in FOLLOW_CASES {
        let clip = if case.clip == "active" {
            "clip_active"
        } else {
            "clip_explicit"
        };
        let track = if case.clip == "active" {
            "active_track"
        } else {
            "explicit_track"
        };
        let camera = if case.camera == "orthographic" {
            "ortho_camera"
        } else {
            "camera"
        };
        operations.push(json!({"op":"node.create","id":format!("owner_{}",case.id),"kind":"empty","transform":{"translation":[0.0,0.0,-4.0]}}));
        operations.push(json!({
            "op":"constraint.create","target":{"id":format!("owner_{}",case.id)},"id":"follow_track","type":"follow_track",
            "constraint_target":camera,
            "params":{"clip":if case.active { Value::Null } else { json!(clip) },"use_active_clip":case.active,"track":track,"use_3d_position":case.use_3d,"use_undistorted_position":case.undistorted,"frame_method":case.frame_method,"camera":camera,"depth_object":if case.id == "active_depth_hit" { json!("depth") } else { Value::Null }}
        }));
    }
    let mut doc = ops::apply_batch(
        &SceneDoc::new("00000000-0000-4000-8000-000000000002".to_owned()),
        &json!({"schema_version":1,"base_revision":0,"operations":operations}),
    )?
    .doc;
    let active_id = Id::new("clip_active".to_owned())?;
    let active = doc
        .movie_clips
        .get_mut(&active_id)
        .ok_or("active clip missing")?;
    let active_markers = expected["active_markers"]
        .as_array()
        .ok_or("active markers missing")?;
    for marker in active_markers.iter().skip(1) {
        active.tracking.tracks[0].markers.push(TrackingMarker {
            frame: marker["frame"]
                .as_f64()
                .ok_or("active marker frame missing")?,
            co: [
                marker["co"][0].as_f64().ok_or("active marker x missing")?
                    * f64::from(active_width),
                (1.0 - marker["co"][1].as_f64().ok_or("active marker y missing")?)
                    * f64::from(active_height),
            ],
            ..TrackingMarker::default()
        });
    }
    active.tracking.tracks[0].markers[0].co = [
        active_markers[0]["co"][0]
            .as_f64()
            .ok_or("active marker x missing")?
            * f64::from(active_width),
        (1.0 - active_markers[0]["co"][1]
            .as_f64()
            .ok_or("active marker y missing")?)
            * f64::from(active_height),
    ];
    active.tracking.camera.k1 = 0.08;
    active
        .tracking
        .reconstruction
        .points
        .push(ReconstructedPoint {
            track: "active_track".to_owned(),
            co: [
                expected["active_bundle"][0]
                    .as_f64()
                    .ok_or("active bundle x missing")?,
                expected["active_bundle"][1]
                    .as_f64()
                    .ok_or("active bundle y missing")?,
                expected["active_bundle"][2]
                    .as_f64()
                    .ok_or("active bundle z missing")?,
            ],
        });
    let explicit_id = Id::new("clip_explicit".to_owned())?;
    let explicit = doc
        .movie_clips
        .get_mut(&explicit_id)
        .ok_or("explicit clip missing")?;
    let explicit_markers = expected["explicit_markers"]
        .as_array()
        .ok_or("explicit markers missing")?;
    for marker in explicit_markers.iter().skip(1) {
        explicit.tracking.tracks[0].markers.push(TrackingMarker {
            frame: marker["frame"]
                .as_f64()
                .ok_or("explicit marker frame missing")?,
            co: [
                marker["co"][0]
                    .as_f64()
                    .ok_or("explicit marker x missing")?
                    * f64::from(explicit_width),
                (1.0 - marker["co"][1]
                    .as_f64()
                    .ok_or("explicit marker y missing")?)
                    * f64::from(explicit_height),
            ],
            ..TrackingMarker::default()
        });
    }
    explicit.tracking.tracks[0].markers[0].co = [
        explicit_markers[0]["co"][0]
            .as_f64()
            .ok_or("explicit marker x missing")?
            * f64::from(explicit_width),
        (1.0 - explicit_markers[0]["co"][1]
            .as_f64()
            .ok_or("explicit marker y missing")?)
            * f64::from(explicit_height),
    ];
    explicit.tracking.camera.k1 = 0.05;
    "PIXELS".clone_into(&mut explicit.tracking.camera.units);
    explicit.tracking.camera.focal_mm = 70.0;
    Ok(doc)
}

#[test]
fn follow_track_modes_and_clip_selection_match_blender_depsgraph() -> TestResult {
    let Some(blender) = blender_file::blender_executable() else {
        eprintln!("skipping Follow Track Blender parity: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = fs::canonicalize(directory.path())?;
    let output_path = root.join("follow_expected.json");
    let script = r#"
import bpy, json, os, sys
if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"Follow Track parity requires Blender 5.2.2, found {bpy.app.version_string}")
root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
scene = bpy.context.scene

def make_clip(prefix, width, height):
    for frame in (1,2,3):
        image = bpy.data.images.new(prefix + str(frame), width, height)
        image.filepath_raw = os.path.join(root, prefix + "_%04d.png" % frame)
        image.file_format = "PNG"
        image.save()
    clip = bpy.data.movieclips.load(os.path.join(root, prefix + "_0001.png"), check_existing=False)
    clip.tracking.camera.units = "MILLIMETERS"
    clip.tracking.camera.sensor_width = 36.0
    clip.tracking.camera.focal_length = 50.0
    clip.tracking.camera.k1 = 0.0
    return clip

active = make_clip("active", 160, 90)
explicit = make_clip("explicit", 80, 120)
scene.active_clip = active
scene.render.resolution_x = 160
scene.render.resolution_y = 100
scene.render.resolution_percentage = 100
focal_pixels = 50.0 / 36.0 * 160
points = [(-2,-1,10),(1,-1,11),(2,2,12),(-1,3,13),(3,-2,14),(-3,2,15),(0.3,0.7,10.8),(1.7,0.2,13.5)]
tracks = []
for index, point in enumerate(points):
    track = active.tracking.tracks.new(name="Track%d" % index)
    tracks.append(track)
    for frame, camera_x in ((1,0.0),(2,0.35),(3,1.0)):
        x,y,z = point
        track.markers.insert_frame(frame, co=(0.5 + focal_pixels * (x-camera_x) / z / 160.0, 0.5 - focal_pixels*y/z/90.0))
tracking_object = active.tracking.objects.active
tracking_object.keyframe_a = 1
tracking_object.keyframe_b = 3
area = next(area for area in bpy.context.screen.areas if area.type == "VIEW_3D")
area.type = "CLIP_EDITOR"
area.spaces.active.clip = active
with bpy.context.temp_override(area=area, space_data=area.spaces.active):
    bpy.ops.clip.solve_camera()
scene.render.resolution_x = 160
scene.render.resolution_y = 100
active.tracking.camera.k1 = 0.08
active_track = tracks[0]
explicit_track = explicit.tracking.tracks.new(name="ExplicitTrack")
for frame, co in ((1,(0.73,0.28)),(2,(0.69,0.32)),(3,(0.65,0.36))):
    explicit_track.markers.insert_frame(frame, co=co)
explicit.tracking.camera.k1 = 0.05
explicit.tracking.camera.units = "PIXELS"
explicit.tracking.camera.focal_length_pixels = 70.0
camera_data = bpy.data.cameras.new("PerspectiveData")
camera_data.lens = 50.0
camera_data.sensor_width = 36.0
camera = bpy.data.objects.new("Perspective", camera_data)
scene.collection.objects.link(camera)
ortho_data = bpy.data.cameras.new("OrthographicData")
ortho_data.type = "ORTHO"
ortho_data.ortho_scale = 3.5
ortho_camera = bpy.data.objects.new("Orthographic", ortho_data)
scene.collection.objects.link(ortho_camera)
depth_mesh = bpy.data.meshes.new("DepthMesh")
depth_mesh.from_pydata([(-10,-10,0),(10,-10,0),(10,10,0),(-10,10,0)], [], [(0,1,2,3)])
depth_object = bpy.data.objects.new("DepthPlane", depth_mesh)
depth_object.location = (0.0, 0.0, -2.0)
scene.collection.objects.link(depth_object)
cases = [
    ("active_undistorted", active, active_track, True, False, True, "STRETCH", camera),
    ("active_raw", active, active_track, True, False, False, "STRETCH", camera),
    ("active_fit", active, active_track, True, False, False, "FIT", camera),
    ("active_crop", active, active_track, True, False, False, "CROP", camera),
    ("active_3d", active, active_track, True, True, False, "STRETCH", camera),
    ("explicit_raw", explicit, explicit_track, False, False, False, "FIT", camera),
    ("explicit_undistorted", explicit, explicit_track, False, False, True, "FIT", camera),
    ("active_ortho", active, active_track, True, False, False, "CROP", ortho_camera),
    ("active_depth_hit", active, active_track, True, False, False, "STRETCH", camera),
]
owners = {}
for name, clip, track, use_active, use_3d, undistorted, frame_method, target_camera in cases:
    owner = bpy.data.objects.new("Owner_" + name, None)
    owner.location = (0.0, 0.0, -4.0)
    scene.collection.objects.link(owner)
    constraint = owner.constraints.new("FOLLOW_TRACK")
    constraint.clip = None if use_active else clip
    constraint.use_active_clip = use_active
    constraint.track = track.name
    if name == "active_depth_hit":
        constraint.depth_object = depth_object
    constraint.camera = target_camera
    constraint.use_3d_position = use_3d
    constraint.use_undistorted_position = undistorted
    constraint.frame_method = frame_method
    owners[name] = owner
scene.frame_set(2)
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
def matrix_values(matrix):
    return [float(matrix[row][column]) for column in range(4) for row in range(4)]
expected = {
    "active_markers": [{"frame": float(marker.frame), "co": [float(marker.co.x), float(marker.co.y)]} for marker in active_track.markers],
    "explicit_markers": [{"frame": float(marker.frame), "co": [float(marker.co.x), float(marker.co.y)]} for marker in explicit_track.markers],
    "active_bundle": [float(value) for value in active_track.bundle],
    "matrices": {},
}
for name, owner in owners.items():
    expected["matrices"][name] = matrix_values(owner.evaluated_get(depsgraph).matrix_world)
with open(os.path.join(root, "follow_expected.json"), "w", encoding="utf-8") as output:
    json.dump(expected, output)
"#;
    run_blender(
        &blender,
        script,
        &[root.to_str().ok_or("temporary path is not UTF-8")?],
        &root,
    )?;
    let expected: Value = serde_json::from_slice(&fs::read(output_path)?)?;
    let doc = follow_doc(&expected)?;
    let snapshot = Snapshot::evaluate(
        &doc,
        &EvaluationContext {
            frame: Some(2.0),
            ..EvaluationContext::default()
        },
    )?;
    assert_matrix_parity(
        node_matrix(&snapshot, "owner_active_raw")?,
        matrix_from_json(&expected["matrices"]["active_raw"])?,
        "active_raw",
    );
    for case in FOLLOW_CASES {
        let actual = node_matrix(&snapshot, &format!("owner_{}", case.id))?;
        let expected = matrix_from_json(&expected["matrices"][case.id])?;
        assert_matrix_parity(actual, expected, case.id);
    }
    let undistorted = node_matrix(&snapshot, "owner_active_undistorted")?;
    let raw = node_matrix(&snapshot, "owner_active_raw")?;
    let lens_adjustment = undistorted
        .to_cols_array()
        .iter()
        .zip(raw.to_cols_array())
        .map(|(undistorted, raw)| (undistorted - raw).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        lens_adjustment > 1.0e-6,
        "nonzero lens distortion must move the undistorted tracking marker"
    );
    Ok(())
}

#[test]
fn follow_track_reports_missing_reconstruction_marker_and_clip_data() -> TestResult {
    let base_ops = json!([
        {"op":"tracking.clip_create","id":"clip","name":"Clip","width":100,"height":50},
        {"op":"tracking.track_add","id":"clip","track":"track","name":"Track","frame":1,"co":[50.0,25.0]},
        {"op":"camera.create","id":"camera","name":"Camera","lens_mm":50.0,"sensor_width_mm":36.0},
        {"op":"node.create","id":"owner","kind":"empty","transform":{"translation":[0.0,0.0,-4.0]}},
        {"op":"constraint.create","target":{"id":"owner"},"id":"follow","type":"follow_track","constraint_target":"camera","params":{"clip":"clip","use_active_clip":false,"track":"track","use_3d_position":false,"use_undistorted_position":false,"frame_method":"STRETCH","camera":"camera"}}
    ]);
    let mut doc = ops::apply_batch(
        &SceneDoc::new("00000000-0000-4000-8000-000000000002".to_owned()),
        &json!({"schema_version":1,"base_revision":0,"operations":base_ops.clone()}),
    )?
    .doc;
    let id = Id::new("clip".to_owned())?;
    doc.movie_clips
        .get_mut(&id)
        .ok_or("clip missing")?
        .tracking
        .tracks[0]
        .markers[0]
        .disabled = true;
    let error = Snapshot::evaluate(&doc, &EvaluationContext::default())
        .expect_err("disabled markers have no sample");
    assert_eq!(error.code, potter_core::error::ErrorCode::EvaluationFailed);
    assert_eq!(
        error.message,
        "follow-track constraint has no enabled markers"
    );
    assert!(
        error
            .details
            .as_object()
            .is_some_and(serde_json::Map::is_empty)
    );

    let mut doc = ops::apply_batch(
        &SceneDoc::new("00000000-0000-4000-8000-000000000002".to_owned()),
        &json!({"schema_version":1,"base_revision":0,"operations":base_ops.clone()}),
    )?
    .doc;
    let constraint = &mut doc
        .nodes
        .get_mut(&Id::new("owner".to_owned())?)
        .ok_or("owner missing")?
        .constraints[0];
    constraint
        .params
        .insert("track".to_owned(), json!("absent"));
    let error = Snapshot::evaluate(&doc, &EvaluationContext::default())
        .expect_err("unknown tracks are rejected");
    assert_eq!(error.code, potter_core::error::ErrorCode::EvaluationFailed);
    assert_eq!(
        error.message,
        "Follow Track constraint track does not exist"
    );
    assert_eq!(error.details["track"], "absent");
    let mut doc = ops::apply_batch(
        &SceneDoc::new("00000000-0000-4000-8000-000000000002".to_owned()),
        &json!({"schema_version":1,"base_revision":0,"operations":base_ops.clone()}),
    )?
    .doc;
    doc.nodes
        .get_mut(&Id::new("owner".to_owned())?)
        .ok_or("owner missing")?
        .constraints[0]
        .params
        .insert("use_3d_position".to_owned(), json!(true));
    let error = Snapshot::evaluate(&doc, &EvaluationContext::default())
        .expect_err("3D sampling requires a reconstructed point");
    assert_eq!(error.code, potter_core::error::ErrorCode::EvaluationFailed);
    assert_eq!(
        error.message,
        "Follow Track 3D position requires a reconstructed point for its track"
    );
    assert_eq!(error.details["constraint_id"], "follow");
    assert_eq!(error.details["track"], "track");

    let mut doc = ops::apply_batch(
        &SceneDoc::new("00000000-0000-4000-8000-000000000002".to_owned()),
        &json!({"schema_version":1,"base_revision":0,"operations":base_ops.clone()}),
    )?
    .doc;
    let clip = doc.movie_clips.get_mut(&id).ok_or("clip missing")?;
    clip.width = 0;
    clip.height = 0;
    let error = Snapshot::evaluate(&doc, &EvaluationContext::default())
        .expect_err("2D sampling requires image dimensions");
    assert_eq!(error.code, potter_core::error::ErrorCode::EvaluationFailed);
    assert_eq!(
        error.message,
        "Follow Track requires movie-clip image dimensions for 2D marker sampling"
    );
    assert_eq!(error.details["constraint_id"], "follow");
    assert_eq!(error.details["clip"], "Clip");

    let mut doc = ops::apply_batch(
        &SceneDoc::new("00000000-0000-4000-8000-000000000002".to_owned()),
        &json!({"schema_version":1,"base_revision":0,"operations":base_ops.clone()}),
    )?
    .doc;
    doc.nodes
        .get_mut(&Id::new("owner".to_owned())?)
        .ok_or("owner missing")?
        .constraints[0]
        .params
        .insert("frame_method".to_owned(), json!("INVALID"));
    let error = Snapshot::evaluate(&doc, &EvaluationContext::default())
        .expect_err("unknown frame methods are rejected");
    assert_eq!(error.code, potter_core::error::ErrorCode::EvaluationFailed);
    assert_eq!(error.message, "Follow Track frame_method is invalid");
    let mut doc = ops::apply_batch(
        &SceneDoc::new("00000000-0000-4000-8000-000000000002".to_owned()),
        &json!({"schema_version":1,"base_revision":0,"operations":base_ops}),
    )?
    .doc;
    doc.nodes
        .get_mut(&Id::new("owner".to_owned())?)
        .ok_or("owner missing")?
        .constraints[0]
        .params
        .insert("use_active_clip".to_owned(), json!(true));
    doc.scenes
        .get_mut(&Id::new("scene_main".to_owned())?)
        .ok_or("scene missing")?
        .active_clip = None;
    let error = Snapshot::evaluate(&doc, &EvaluationContext::default())
        .expect_err("active clip selection requires a configured clip");
    assert_eq!(error.code, potter_core::error::ErrorCode::EvaluationFailed);
    assert_eq!(
        error.message,
        "constraint use_active_clip requires an active scene movie clip"
    );
    assert_eq!(error.details["constraint_id"], "follow");
    Ok(())
}
