use std::{error::Error, fs, io, path::Path, process::Command};

use glam::{DMat4, DQuat, DVec3};
use potter_core::{
    eval::{EvaluationContext, Snapshot},
    model::{Id, ReconstructedPoint, SceneDoc, TrackingMarker},
    ops,
    tracking::{CameraModel, CameraObservation},
};
use serde_json::{Value, json};
use tempfile::tempdir;
#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/process.rs"]
mod process;
use blender_checked::blender_executable;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn apply(project: &Path, operations: &Value) -> TestResult {
    let batch = project.join("operations.json");
    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": 0,
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
        "pot apply failed: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    Ok(())
}

fn project(operations: &Value) -> TestResult<(tempfile::TempDir, std::path::PathBuf)> {
    let directory = tempdir()?;
    let project = directory.path().join("scene");
    let output = pot().arg("init").arg(&project).arg("--json").output()?;
    assert!(output.status.success());
    apply(&project, operations)?;
    Ok((directory, project))
}

fn evaluated_matrix(project: &Path, node: &str) -> TestResult<DMat4> {
    let document: SceneDoc = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let snapshot = Snapshot::evaluate(&document, &EvaluationContext::default())?;
    let matrix = snapshot
        .nodes
        .get(&potter_core::model::Id::new(node.to_owned())?)
        .map(|evaluated| DMat4::from_cols_array(&evaluated.world_matrix))
        .ok_or_else(|| format!("evaluated world matrix for {node} is missing"))?;
    Ok(matrix)
}
fn blender_matrix(value: &Value) -> TestResult<DMat4> {
    let components = value
        .as_array()
        .filter(|components| components.len() == 16)
        .ok_or_else(|| io::Error::other("Blender matrix must contain 16 components"))?;
    let mut matrix = [0.0; 16];
    for (index, component) in components.iter().enumerate() {
        matrix[index] = component
            .as_f64()
            .ok_or_else(|| io::Error::other("Blender matrix component is invalid"))?;
    }
    Ok(DMat4::from_cols_array(&matrix))
}

fn max_matrix_error(left: DMat4, right: DMat4) -> f64 {
    left.to_cols_array()
        .iter()
        .zip(right.to_cols_array())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max)
}
fn object_reprojection_errors(
    doc: &SceneDoc,
    clip_id: &Id,
    object_id: &str,
    width: u32,
    height: u32,
) -> TestResult<Vec<(f64, f64)>> {
    let clip = doc
        .movie_clips
        .get(clip_id)
        .ok_or_else(|| io::Error::other("tracking clip is missing"))?;
    let object = clip
        .tracking
        .objects
        .iter()
        .find(|object| object.id == object_id)
        .ok_or_else(|| io::Error::other("tracking object is missing"))?;
    object
        .reconstruction
        .iter()
        .map(|pose| {
            let frame = pose.frame;
            let solved_camera = clip
                .tracking
                .reconstruction
                .cameras
                .iter()
                .find(|camera| (f64::from(camera.frame) - frame).abs() < 1.0e-9)
                .ok_or_else(|| io::Error::other("object pose has no solved camera frame"))?;
            let camera = CameraModel {
                matrix: solved_camera.matrix,
            };
            let camera_world = DMat4::from_cols_array(&potter_core::tracking::camera_world_matrix(
                &camera,
                &clip.tracking.camera,
                width,
                height,
            )?);
            let object_world = camera_world * DMat4::from_cols_array(&pose.matrix).inverse();
            let mut squared_error = 0.0;
            let mut count = 0_u32;
            for track_id in &object.tracks {
                let track = clip
                    .tracking
                    .tracks
                    .iter()
                    .find(|track| track.id == *track_id)
                    .ok_or_else(|| io::Error::other("object track is missing"))?;
                let Some(marker) = track
                    .markers
                    .iter()
                    .find(|marker| (marker.frame - frame).abs() < 1.0e-9 && !marker.disabled)
                else {
                    continue;
                };
                let point = clip
                    .tracking
                    .reconstruction
                    .points
                    .iter()
                    .find(|point| point.track == *track_id)
                    .ok_or_else(|| io::Error::other("object track bundle is missing"))?;
                let world = object_world
                    .transform_point3(DVec3::from_array(point.co))
                    .to_array();
                let projected = camera.project(world)?;
                let dx = projected[0] - marker.co[0];
                let dy = projected[1] - marker.co[1];
                squared_error += dx * dx + dy * dy;
                count += 1;
            }
            if count == 0 {
                return Err(io::Error::other("object pose has no enabled track markers").into());
            }
            Ok((frame, (squared_error / f64::from(count)).sqrt()))
        })
        .collect()
}

#[test]
fn action_constraint_maps_a_target_channel_to_the_assigned_action_frame() -> TestResult {
    let (_directory, project) = project(&json!([
        {"op":"node.create","id":"owner","kind":"empty"},
        {"op":"node.create","id":"driver","kind":"empty","transform":{"translation":[0.5,0.0,0.0]}},
        {"op":"action.create","id":"drive","name":"Drive"},
        {"op":"action.update","target":{"id":"drive"},"set":{"fcurves":[{"path":"transform.translation","index":0,"keyframes":[{"frame":1.0,"value":0.0},{"frame":11.0,"value":10.0}]}]}},
        {"op":"constraint.create","target":{"id":"owner"},"id":"action_constraint","type":"action","constraint_target":"driver","params":{"target":"driver","action":"drive","transform_channel":"LOCATION_X","target_space":"WORLD","min":0.0,"max":1.0,"frame_start":1,"frame_end":11,"mix_mode":"AFTER_FULL","use_eval_time":false,"eval_time":0.0,"use_bone_object_action":false}}
    ]))?;
    let matrix = evaluated_matrix(&project, "owner")?;
    assert!(
        (matrix.w_axis.x - 5.0).abs() < 1.0e-5,
        "action result was {matrix:?}"
    );
    Ok(())
}

#[test]
fn geometry_attribute_constraint_reads_a_named_target_mesh_attribute() -> TestResult {
    let (_directory, project) = project(&json!([
        {"op":"node.create","id":"owner","kind":"empty"},
        {"op":"node.create","id":"target","kind":"box","params":{}},
        {"op":"mesh.attribute_create","target":{"id":"target"},"name":"location","domain":"point","type":"float3"},
        {"op":"mesh.attribute_update","target":{"id":"target"},"name":"location","elements":{"domain":"vertex","ids":["v0"]},"value":[3.0,4.0,5.0]},
        {"op":"constraint.create","target":{"id":"owner"},"id":"geometry","type":"geometry_attribute","constraint_target":"target","params":{"target":"target","attribute_name":"location","data_type":"VECTOR","domain":"POINT","sample_index":0,"mix_mode":"REPLACE","apply_target_transform":false,"mix_loc":true,"mix_rot":false,"mix_scl":false}}
    ]))?;
    let matrix = evaluated_matrix(&project, "owner")?;
    let location = matrix.w_axis.truncate();
    assert!(
        location.abs_diff_eq(DVec3::new(3.0, 4.0, 5.0), 1.0e-5),
        "attribute result was {location:?}"
    );
    Ok(())
}
#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the tracking solve regression constructs correspondences, solves, and checks persisted object poses end to end"
)]
fn tracking_solve_object_persists_rigid_poses_from_camera_reconstruction() -> TestResult {
    let focal_pixels = 50.0 / 36.0 * 100.0;
    let camera = CameraModel {
        matrix: [
            [focal_pixels, 0.0, 50.0, 0.0],
            [0.0, focal_pixels, 50.0, 0.0],
            [0.0, 0.0, 1.0, 8.0],
        ],
    };
    let points = [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -0.5],
        [-1.0, 1.0, 0.2],
        [1.0, 1.0, 0.8],
        [-0.4, 0.3, 1.4],
        [0.7, -0.2, 1.8],
    ];
    let camera_observations = points
        .iter()
        .map(|point| {
            Ok(CameraObservation {
                world: *point,
                image: camera.project(*point)?,
            })
        })
        .collect::<potter_core::error::Result<Vec<_>>>()?;
    let expected = DMat4::from_scale_rotation_translation(
        DVec3::splat(1.0),
        DQuat::from_euler(glam::EulerRot::XYZ, 0.06, -0.1, 0.18),
        DVec3::new(0.35, -0.25, 0.4),
    );
    let expected_first = DMat4::from_scale_rotation_translation(
        DVec3::splat(1.0),
        DQuat::from_euler(glam::EulerRot::XYZ, 0.02, -0.04, 0.07),
        DVec3::new(0.12, -0.08, 0.18),
    );
    let track_ids = (0..points.len())
        .map(|index| format!("track_{index}"))
        .collect::<Vec<_>>();
    let mut operations = vec![
        json!({"op":"camera.create","id":"camera","name":"Solved Camera"}),
        json!({"op":"node.create","id":"moving_owner","kind":"empty"}),
        json!({"op":"node.create","id":"inverse_owner","kind":"empty","transform":{"translation":[1.0,2.0,3.0]}}),
        json!({
            "op":"tracking.clip_create",
            "id":"clip",
            "name":"Synthetic",
            "width":100,
            "height":100
        }),
    ];
    for (index, (track_id, point)) in track_ids.iter().zip(points).enumerate() {
        operations.push(json!({
            "op":"tracking.track_add",
            "id":"clip",
            "track":track_id,
            "name":format!("Track {index}"),
            "frame":1.0,
            "co":camera.project(expected_first.transform_point3(DVec3::from_array(point)).to_array())?
        }));
    }
    for frame in [1, 2] {
        operations.push(json!({
            "op":"tracking.solve_camera",
            "id":"clip",
            "frame":frame,
            "observations":camera_observations.clone()
        }));
    }
    let mut outcome = ops::apply_batch(
        &SceneDoc::new("00000000-0000-4000-8000-000000000001".to_owned()),
        &json!({
            "schema_version":1,
            "base_revision":0,
            "operations":operations
        }),
    )?;
    let clip_id = Id::new("clip".to_owned())?;
    let clip = outcome
        .doc
        .movie_clips
        .get_mut(&clip_id)
        .ok_or_else(|| io::Error::other("tracking clip is missing after creation"))?;
    clip.tracking.reconstruction.points = track_ids
        .iter()
        .zip(points)
        .map(|(track, co)| ReconstructedPoint {
            track: track.clone(),
            co,
        })
        .collect();
    for ((track, point), index) in clip.tracking.tracks.iter_mut().zip(points).zip(0..) {
        let image = camera.project(
            expected
                .transform_point3(DVec3::from_array(point))
                .to_array(),
        )?;
        track.markers.push(TrackingMarker {
            frame: 2.0,
            co: image,
            ..TrackingMarker::default()
        });
        assert_eq!(track.id, track_ids[index]);
    }
    let solved = ops::apply_batch(
        &outcome.doc,
        &json!({
            "schema_version":1,
            "base_revision":outcome.doc.revision,
            "operations":[{
                "op":"tracking.solve_object",
                "id":"clip",
                "object":"moving_object",
                "name":"Moving Object",
                "tracks":track_ids
            }]
        }),
    )?;
    let constrained = ops::apply_batch(
        &solved.doc,
        &json!({
            "schema_version":1,
            "base_revision":solved.doc.revision,
            "operations":[
                {"op":"scene.update","target":{"id":"scene_main"},"set":{"active_clip":"clip"}},
                {"op":"constraint.create","target":{"id":"camera"},"id":"camera_solve","type":"camera_solver","params":{"use_active_clip":true,"clip":null}},
                {"op":"constraint.create","target":{"id":"moving_owner"},"id":"object_solve","type":"object_solver","constraint_target":"camera","params":{"use_active_clip":true,"clip":null,"object":"moving_object","camera":"camera","set_inverse_pending":false}},
                {"op":"constraint.create","target":{"id":"inverse_owner"},"id":"inverse_solve","type":"object_solver","constraint_target":"camera","params":{"use_active_clip":true,"clip":null,"object":"moving_object","camera":"camera","set_inverse_pending":true}}
            ]
        }),
    )?;
    let object = &solved.doc.movie_clips[&clip_id].tracking.objects[0];
    assert_eq!(object.id, "moving_object");
    assert_eq!(object.reconstruction.len(), 2);
    let clip = &solved.doc.movie_clips[&clip_id];
    let camera_world_at = |frame: i32| -> TestResult<DMat4> {
        let camera = clip
            .tracking
            .reconstruction
            .cameras
            .iter()
            .find(|camera| camera.frame == frame)
            .ok_or_else(|| io::Error::other("solved camera frame is missing"))?;
        Ok(DMat4::from_cols_array(
            &potter_core::tracking::camera_world_matrix(
                &CameraModel {
                    matrix: camera.matrix,
                },
                &clip.tracking.camera,
                clip.width,
                clip.height,
            )?,
        ))
    };
    let camera_world_one = camera_world_at(1)?;
    let camera_world_two = camera_world_at(2)?;
    let object_camera_one = DMat4::from_cols_array(&object.reconstruction[0].matrix);
    let object_camera_two = DMat4::from_cols_array(&object.reconstruction[1].matrix);
    let parent_one = camera_world_one * object_camera_one.inverse();
    let parent_two = camera_world_two * object_camera_two.inverse();
    let snapshot_two = Snapshot::evaluate(
        &constrained.doc,
        &EvaluationContext {
            frame: Some(2.0),
            ..EvaluationContext::default()
        },
    )?;
    let matrix_for = |node: &str| -> TestResult<DMat4> {
        let id = Id::new(node.to_owned())?;
        let matrix = snapshot_two
            .nodes
            .get(&id)
            .map(|item| DMat4::from_cols_array(&item.world_matrix))
            .ok_or_else(|| io::Error::other(format!("evaluated node {node} is missing")))?;
        Ok(matrix)
    };
    let assert_matrix_close = |actual: DMat4, expected: DMat4, tolerance: f64| {
        for (actual, expected) in actual.to_cols_array().iter().zip(expected.to_cols_array()) {
            assert!(
                (actual - expected).abs() < tolerance,
                "matrix component mismatch: actual={actual}, expected={expected}, tolerance={tolerance}"
            );
        }
    };
    assert_matrix_close(matrix_for("camera")?, camera_world_two, 2.0e-4);
    assert_matrix_close(matrix_for("moving_owner")?, parent_two, 3.0e-4);
    let original_inverse_owner = DMat4::from_translation(DVec3::new(1.0, 2.0, 3.0));
    assert_matrix_close(
        matrix_for("inverse_owner")?,
        parent_two * parent_one.inverse() * original_inverse_owner,
        4.0e-4,
    );
    let cleared = ops::apply_batch(
        &constrained.doc,
        &json!({
            "schema_version":1,
            "base_revision":constrained.doc.revision,
            "operations":[{"op":"constraint.update","target":{"id":"inverse_owner"},"id":"inverse_solve","set":{"clear_inverse":true}}]
        }),
    )?;
    let cleared_snapshot = Snapshot::evaluate(
        &cleared.doc,
        &EvaluationContext {
            frame: Some(2.0),
            ..EvaluationContext::default()
        },
    )?;
    let cleared_matrix = cleared_snapshot
        .nodes
        .get(&Id::new("inverse_owner".to_owned())?)
        .map(|item| DMat4::from_cols_array(&item.world_matrix))
        .ok_or_else(|| io::Error::other("cleared inverse owner is missing"))?;
    assert_matrix_close(cleared_matrix, parent_two * original_inverse_owner, 4.0e-4);
    let mut rescaled_document = solved.doc.clone();
    let tracks = {
        let rescaled_clip = rescaled_document
            .movie_clips
            .get_mut(&clip_id)
            .ok_or_else(|| io::Error::other("tracking clip is missing before re-solve"))?;
        let tracked_object = rescaled_clip
            .tracking
            .objects
            .iter_mut()
            .find(|object| object.id == "moving_object")
            .ok_or_else(|| io::Error::other("tracking object is missing before re-solve"))?;
        tracked_object.scale = 1.25;
        tracked_object.tracks.clone()
    };
    let re_solved = ops::apply_batch(
        &rescaled_document,
        &json!({
            "schema_version":1,
            "base_revision":rescaled_document.revision,
            "operations":[{
                "op":"tracking.solve_object",
                "id":"clip",
                "object":"moving_object",
                "name":"Moving Object",
                "tracks":tracks
            }]
        }),
    )?;
    assert!((re_solved.doc.movie_clips[&clip_id].tracking.objects[0].scale - 1.25).abs() < 1.0e-10);
    let pose = object
        .reconstruction
        .iter()
        .find(|pose| (pose.frame - 2.0).abs() < 1.0e-9)
        .ok_or_else(|| io::Error::other("frame-2 tracking pose is missing"))?;
    let solved_camera = clip
        .tracking
        .reconstruction
        .cameras
        .iter()
        .find(|camera| camera.frame == 2)
        .ok_or_else(|| io::Error::other("frame-2 solved camera is missing"))?;
    let camera_world = DMat4::from_cols_array(&potter_core::tracking::camera_world_matrix(
        &CameraModel {
            matrix: solved_camera.matrix,
        },
        &clip.tracking.camera,
        clip.width,
        clip.height,
    )?);
    let expected_camera_to_object = (camera_world.inverse() * expected).inverse();
    let actual = DMat4::from_cols_array(&pose.matrix).to_cols_array();
    let expected = expected_camera_to_object.to_cols_array();
    for (index, (actual_value, expected_value)) in actual.iter().zip(expected).enumerate() {
        // DLT and Blender's reconstruction choose nearby but not identical scale gauges.
        assert!(
            (actual_value - expected_value).abs() < 3.0e-2,
            "tracking object matrix element {index} differs: Potter={actual_value}, expected={expected_value}; actual={actual:?}, expected={expected:?}"
        );
    }
    Ok(())
}
#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the parity fixture keeps its channel, range, and mix-mode cross-checks together"
)]
fn blender_action_constraint_channels_mix_modes_and_spaces_match_depsgraph() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Action constraint Blender parity test: Blender is unavailable");
        return Ok(());
    };
    let action_modes = [
        "REPLACE",
        "BEFORE_FULL",
        "BEFORE",
        "BEFORE_SPLIT",
        "AFTER_FULL",
        "AFTER",
        "AFTER_SPLIT",
    ];
    let channels = [
        ("LOCATION_X", -0.5, 1.5),
        ("LOCATION_Y", -1.0, 1.0),
        ("LOCATION_Z", -0.5, 1.5),
        ("ROTATION_X", -45.0, 45.0),
        ("ROTATION_Y", -60.0, 60.0),
        ("ROTATION_Z", -90.0, 90.0),
        ("SCALE_X", 0.5, 2.5),
        ("SCALE_Y", 0.2, 2.0),
        ("SCALE_Z", 0.5, 2.5),
    ];
    let parent_rotation = DQuat::from_euler(glam::EulerRot::XYZ, 0.2, 0.4, -0.2);
    let driver_rotation =
        DQuat::from_rotation_z(0.71) * DQuat::from_rotation_y(-0.52) * DQuat::from_rotation_x(0.38);
    let armature_rotation = DQuat::from_euler(glam::EulerRot::XYZ, 0.14, 0.32, -0.11);
    let owner_rotation = DQuat::IDENTITY;
    let mut cases = Vec::new();
    let mut operations = vec![
        json!({"op":"node.create","id":"driver_parent","kind":"empty","transform":{"translation":[1.2,-0.4,0.7],"rotation":[parent_rotation.x,parent_rotation.y,parent_rotation.z,parent_rotation.w],"scale":[0.8,1.2,1.1]}}),
        json!({"op":"node.create","id":"driver","kind":"empty","parent":"driver_parent","transform":{"translation":[0.35,-0.4,0.65],"rotation":[driver_rotation.x,driver_rotation.y,driver_rotation.z,driver_rotation.w],"scale":[1.4,0.7,1.8]}}),
        json!({"op":"node.create","id":"target_arm","kind":"armature","transform":{"translation":[0.4,0.3,-0.2],"rotation":[armature_rotation.x,armature_rotation.y,armature_rotation.z,armature_rotation.w],"scale":[1.1,0.9,1.2]}}),
        json!({"op":"bone.create","target":{"id":"target_arm"},"id":"target_bone","name":"TargetBone","head":[2.0,0.4,0.1],"tail":[2.2,1.3,0.4]}),
        json!({"op":"action.create","id":"drive","name":"Drive"}),
    ];
    let curves = [
        ("transform.translation", 0, -1.0, 4.0),
        ("transform.translation", 1, 2.0, -3.0),
        ("transform.translation", 2, 0.5, 7.0),
        ("transform.rotation_euler", 0, -0.35, 1.1),
        ("transform.rotation_euler", 1, 0.2, -0.9),
        ("transform.rotation_euler", 2, -0.5, 1.7),
        ("transform.scale", 0, 0.5, 2.6),
        ("transform.scale", 1, 1.8, 0.4),
        ("transform.scale", 2, 0.7, 1.9),
    ]
    .into_iter()
    .map(|(path, index, start, end)| {
        json!({
            "path":path,
            "index":index,
            "keyframes":[
                {"frame":1.0,"value":start,"interpolation":"linear"},
                {"frame":11.0,"value":end,"interpolation":"linear"}
            ]
        })
    })
    .collect::<Vec<_>>();
    operations.push(json!({"op":"action.update","target":{"id":"drive"},"set":{"fcurves":curves}}));

    let mut add_case = |case_id: String,
                        target: &str,
                        subtarget: Option<&str>,
                        channel: &str,
                        target_space: &str,
                        minimum: f64,
                        maximum: f64,
                        mix_mode: &str,
                        frame_start: i32,
                        frame_end: i32| {
        let mut case = json!({
            "id":case_id,
            "target":target,
            "channel":channel,
            "target_space":target_space,
            "min":minimum,
            "max":maximum,
            "mix_mode":mix_mode,
            "frame_start":frame_start,
            "frame_end":frame_end
        });
        if let Some(subtarget) = subtarget {
            case["subtarget"] = json!(subtarget);
        }
        operations.push(json!({
            "op":"node.create",
            "id":case_id,
            "kind":"empty",
            "transform":{
                "translation":[1.2,-0.7,2.0],
                "rotation":[owner_rotation.x,owner_rotation.y,owner_rotation.z,owner_rotation.w],
                "scale":[1.0,1.0,1.0]
            }
        }));
        let mut constraint = json!({
            "op":"constraint.create",
            "target":{"id":case_id},
            "id":"action_constraint",
            "type":"action",
            "constraint_target":target,
            "params":{
                "target":target,
                "action":"drive",
                "transform_channel":channel,
                "target_space":target_space,
                "min":minimum,
                "max":maximum,
                "frame_start":frame_start,
                "frame_end":frame_end,
                "mix_mode":mix_mode,
                "use_eval_time":false,
                "eval_time":0.0,
                "use_bone_object_action":false
            }
        });
        if let Some(subtarget) = subtarget {
            constraint["subtarget"] = json!(subtarget);
        }
        operations.push(constraint);
        cases.push(case);
    };

    for (index, (channel, minimum, maximum)) in channels.iter().enumerate() {
        add_case(
            format!("channel_{index}"),
            "driver",
            None,
            channel,
            "LOCAL",
            *minimum,
            *maximum,
            action_modes[index % action_modes.len()],
            2,
            10,
        );
    }
    for (index, mix_mode) in action_modes.iter().enumerate() {
        add_case(
            format!("mix_{index}"),
            "driver",
            None,
            "LOCATION_X",
            "WORLD",
            -3.0,
            4.0,
            mix_mode,
            1,
            11,
        );
    }
    for (case_id, target, subtarget, target_space, minimum, maximum) in [
        ("space_world", "driver", None, "WORLD", -3.0, 4.0),
        ("space_local", "driver", None, "LOCAL", -1.0, 1.0),
        (
            "space_pose",
            "target_arm",
            Some("target_bone"),
            "POSE",
            0.0,
            4.0,
        ),
        (
            "space_local_with_parent",
            "target_arm",
            Some("target_bone"),
            "LOCAL_WITH_PARENT",
            0.0,
            4.0,
        ),
    ] {
        add_case(
            case_id.to_owned(),
            target,
            subtarget,
            "LOCATION_X",
            target_space,
            minimum,
            maximum,
            "AFTER_SPLIT",
            2,
            10,
        );
    }

    let (_directory, project) = project(&Value::Array(operations))?;
    let fixture = json!({
        "cases":cases,
        "rotations":{
            "parent":[parent_rotation.x,parent_rotation.y,parent_rotation.z,parent_rotation.w],
            "driver":[driver_rotation.x,driver_rotation.y,driver_rotation.z,driver_rotation.w],
            "armature":[armature_rotation.x,armature_rotation.y,armature_rotation.z,armature_rotation.w],
            "owner":[owner_rotation.x,owner_rotation.y,owner_rotation.z,owner_rotation.w]
        }
    });
    let fixture_json = serde_json::to_string(&fixture)?;
    let fixture_literal = serde_json::to_string(&fixture_json)?;
    let script = format!(
        r#"
import bpy, json
fixture = json.loads({fixture_literal})
cases = fixture["cases"]
rotations = fixture["rotations"]
def set_rotation(obj, values):
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = (values[3],values[0],values[1],values[2])
scene = bpy.context.scene
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
parent = bpy.data.objects.new("DriverParent", None)
driver = bpy.data.objects.new("Driver", None)
scene.collection.objects.link(parent)
scene.collection.objects.link(driver)
parent.location = (1.2,-0.4,0.7)
set_rotation(parent, rotations["parent"])
parent.scale = (0.8,1.2,1.1)
driver.parent = parent
driver.location = (0.35,-0.4,0.65)
driver.rotation_mode = "XYZ"
driver.rotation_euler = (0.38,-0.52,0.71)
driver.scale = (1.4,0.7,1.8)
armature_data = bpy.data.armatures.new("TargetArmature")
armature = bpy.data.objects.new("TargetArmature", armature_data)
scene.collection.objects.link(armature)
armature.location = (0.4,0.3,-0.2)
set_rotation(armature, rotations["armature"])
armature.scale = (1.1,0.9,1.2)
bpy.context.view_layer.objects.active = armature
armature.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
bone = armature_data.edit_bones.new("TargetBone")
bone.head = (2.0,0.4,0.1)
bone.tail = (2.2,1.3,0.4)
bpy.ops.object.mode_set(mode="OBJECT")
owners = {{}}
for case in cases:
    owner = bpy.data.objects.new(case["id"], None)
    scene.collection.objects.link(owner)
    owner.location = (1.2,-0.7,2.0)
    owner.rotation_mode = "XYZ"
    owner.rotation_euler = (0.0,0.0,0.0)
    owner.scale = (1.0,1.0,1.0)
    owners[case["id"]] = owner
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
def world_matrix(obj):
    matrix = obj.evaluated_get(depsgraph).matrix_world
    return [float(matrix[row][column]) for column in range(4) for row in range(4)]
baseline = {{case["id"]: world_matrix(owners[case["id"]]) for case in cases}}
action = bpy.data.actions.new("Drive")
slot = action.slots.new(id_type="OBJECT", name="Drive")
dummy = bpy.data.objects.new("ActionSlotOwner", None)
scene.collection.objects.link(dummy)
dummy.animation_data_create()
dummy.animation_data.action = action
dummy.animation_data.action_slot = slot
curve_spec = [
    ("location",0,-1.0,4.0),("location",1,2.0,-3.0),("location",2,0.5,7.0),
    ("rotation_euler",0,-0.35,1.1),("rotation_euler",1,0.2,-0.9),("rotation_euler",2,-0.5,1.7),
    ("scale",0,0.5,2.6),("scale",1,1.8,0.4),("scale",2,0.7,1.9)
]
for path, index, start, end in curve_spec:
    curve = action.fcurve_ensure_for_datablock(dummy, path, index=index)
    for frame, value in ((1.0,start),(11.0,end)):
        point = curve.keyframe_points.insert(frame, value)
        point.interpolation = "LINEAR"
    curve.update()
dummy.animation_data.action = None
for case in cases:
    constraint = owners[case["id"]].constraints.new("ACTION")
    constraint.target = driver if case["target"] == "driver" else armature
    if "subtarget" in case:
        constraint.subtarget = "TargetBone"
    constraint.action = action
    constraint.action_slot = slot
    constraint.transform_channel = case["channel"]
    constraint.target_space = case["target_space"]
    constraint.min = case["min"]
    constraint.max = case["max"]
    constraint.frame_start = int(case["frame_start"])
    constraint.frame_end = int(case["frame_end"])
    constraint.mix_mode = case["mix_mode"]
scene.frame_set(1)
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
result = {{
    case["id"]: {{"matrix":world_matrix(owners[case["id"]]),"baseline":baseline[case["id"]]}}
    for case in cases
}}
print("POTTER_ACTION_PARITY=" + json.dumps(result))
"#
    );
    let output = Command::new(blender)
        .args(["--background", "--factory-startup", "--python-expr"])
        .arg(script)
        .output()?;
    assert!(
        output.status.success(),
        "Blender Action constraint fixture failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let blender_stdout = String::from_utf8_lossy(&output.stdout);
    let line = blender_stdout
        .lines()
        .find_map(|line| line.strip_prefix("POTTER_ACTION_PARITY="))
        .ok_or_else(|| io::Error::other(format!(
            "Blender did not return Action constraint matrices; stdout={blender_stdout}; stderr={}",
            String::from_utf8_lossy(&output.stderr)
        )))?;
    let blender_matrices: Value = serde_json::from_str(line)?;
    let mut maximum_error = 0.0_f64;
    for case in &cases {
        let id = case["id"]
            .as_str()
            .ok_or_else(|| io::Error::other("Action case ID is missing"))?;
        let expected = blender_matrices[id]["matrix"]
            .as_array()
            .ok_or_else(|| io::Error::other("Blender Action matrix is missing"))?;
        let baseline = blender_matrices[id]["baseline"]
            .as_array()
            .ok_or_else(|| io::Error::other("Blender Action baseline is missing"))?;
        let actual = evaluated_matrix(&project, id)?.to_cols_array();
        let mut maximum_effect = 0.0_f64;
        let mut linear_effect = 0.0_f64;
        for (index, (actual_value, expected_value)) in actual.iter().zip(expected).enumerate() {
            let expected_value = expected_value
                .as_f64()
                .ok_or_else(|| io::Error::other("Blender Action matrix component is invalid"))?;
            let baseline_value = baseline[index]
                .as_f64()
                .ok_or_else(|| io::Error::other("Blender Action baseline component is invalid"))?;
            maximum_effect = maximum_effect.max((expected_value - baseline_value).abs());
            if [0, 1, 2, 4, 5, 6, 8, 9, 10].contains(&index) {
                linear_effect = linear_effect.max((expected_value - baseline_value).abs());
            }
            let error = (actual_value - expected_value).abs();
            maximum_error = maximum_error.max(error);
            assert!(
                error < 2.0e-4,
                "{id} Blender Action matrix component {index} differs: Potter={actual_value}, Blender={expected_value}, actual={actual:?}, Blender matrix={expected:?}, case={case}"
            );
        }
        assert!(
            maximum_effect > 1.0e-2,
            "{id} Action fixture had no material effect over baseline"
        );
        assert!(
            linear_effect > 1.0e-2,
            "{id} Action result only changed translation; it must exercise rotation/scale mixing"
        );
    }
    let world = blender_matrix(&blender_matrices["space_world"]["matrix"])?;
    let local = blender_matrix(&blender_matrices["space_local"]["matrix"])?;
    let space_difference = world
        .to_cols_array()
        .iter()
        .zip(local.to_cols_array())
        .map(|(world, local)| (world - local).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        space_difference > 1.0e-2,
        "WORLD and LOCAL target spaces were not materially distinct"
    );
    eprintln!(
        "Blender Action constraint matrix max error across 9 channels, 7 mix modes, and 4 target spaces: {maximum_error:.6e}"
    );
    Ok(())
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the parity fixture keeps all sampled data types and mesh domains auditable together"
)]
fn blender_geometry_attribute_types_and_domains_match_depsgraph() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Geometry Attribute Blender parity test: Blender is unavailable");
        return Ok(());
    };
    let domains = [
        ("POINT", "point"),
        ("EDGE", "edge"),
        ("FACE", "face"),
        ("FACE_CORNER", "corner"),
    ];
    let data_types = ["VECTOR", "QUATERNION", "FLOAT4X4"];
    let mix_modes = [
        "REPLACE",
        "BEFORE_FULL",
        "BEFORE_SPLIT",
        "AFTER_FULL",
        "AFTER_SPLIT",
    ];
    let owner_rotation = DQuat::from_euler(glam::EulerRot::XYZ, 0.16, 0.37, -0.25);
    let target_rotation = DQuat::from_euler(glam::EulerRot::XYZ, 0.28, -0.19, 0.42);
    let mut cases = Vec::new();
    let mut operations = vec![json!({
        "op":"node.create",
        "id":"geometry_target",
        "kind":"box",
        "params":{},
        "transform":{
            "translation":[0.7,-1.1,2.3],
            "rotation":[target_rotation.x,target_rotation.y,target_rotation.z,target_rotation.w],
            "scale":[1.3,0.75,1.6]
        }
    })];
    for (type_index, data_type) in data_types.iter().enumerate() {
        for (domain_index, (domain, _stored_domain)) in domains.iter().enumerate() {
            let attr_name = format!("attr_{}_{}", data_type.to_ascii_lowercase(), domain_index);
            let owner_id = format!("owner_{}_{}", data_type.to_ascii_lowercase(), domain_index);
            let angle = 0.31 + f64::from(u32::try_from(domain_index)?) * 0.17;
            let rotation = DQuat::from_euler(glam::EulerRot::XYZ, angle, -angle * 0.7, angle * 1.3);
            let matrix = DMat4::from_scale_rotation_translation(
                DVec3::new(1.35 + angle * 0.1, 0.72 + angle * 0.05, 1.18 + angle * 0.08),
                rotation,
                DVec3::new(2.4 + angle, -1.8 + angle * 0.4, 0.6 - angle * 0.2),
            );
            let matrix_columns = matrix.transpose().to_cols_array();
            let matrix_rows = matrix_columns
                .as_chunks::<4>()
                .0
                .iter()
                .map(|row| row.to_vec())
                .collect::<Vec<_>>();
            let vector = [
                2.3 + f64::from(u32::try_from(domain_index)?) * 0.43,
                -1.7 + f64::from(u32::try_from(domain_index)?) * 0.29,
                0.8 - f64::from(u32::try_from(domain_index)?) * 0.17,
            ];
            let quaternion = [rotation.x, rotation.y, rotation.z, rotation.w];
            let mix_mode = mix_modes[(type_index * domains.len() + domain_index) % mix_modes.len()];
            let case = json!({
                "id":owner_id,
                "attribute":attr_name,
                "data_type":data_type,
                "domain":domain,
                "mix_mode":mix_mode,
                "vector":vector,
                "quaternion":quaternion,
                "matrix":matrix_rows,
                "owner_rotation":[owner_rotation.x,owner_rotation.y,owner_rotation.z,owner_rotation.w],
                "target_rotation":[target_rotation.x,target_rotation.y,target_rotation.z,target_rotation.w]
            });
            operations.push(json!({
                "op":"node.create",
                "id":owner_id,
                "kind":"empty",
                "transform":{
                    "translation":[-0.8,1.2,2.5],
                    "rotation":[owner_rotation.x,owner_rotation.y,owner_rotation.z,owner_rotation.w],
                    "scale":[0.8,1.2,1.1]
                }
            }));
            operations.push(json!({
                "op":"constraint.create",
                "target":{"id":owner_id},
                "id":"geometry_constraint",
                "type":"geometry_attribute",
                "constraint_target":"geometry_target",
                "params":{
                    "target":"geometry_target",
                    "attribute_name":attr_name,
                    "data_type":data_type,
                    "domain":domain,
                    "sample_index":0,
                    "mix_mode":mix_mode,
                    "apply_target_transform":true,
                    "mix_loc":true,
                    "mix_rot":true,
                    "mix_scl":true
                }
            }));
            cases.push(case);
        }
    }
    let (_directory, project) = project(&Value::Array(operations))?;
    let scene_path = project.join("scene.json");
    let mut scene: Value = serde_json::from_slice(&fs::read(&scene_path)?)?;
    let data_id = scene["nodes"]["geometry_target"]["data"]
        .as_str()
        .ok_or_else(|| io::Error::other("Potter geometry target data ID is missing"))?
        .to_owned();
    let mesh = &mut scene["data_blocks"][data_id]["mesh"];
    let faces = mesh["faces"]
        .as_array()
        .ok_or_else(|| io::Error::other("Potter geometry target faces are missing"))?;
    let corner_count = faces[0]["v"]
        .as_array()
        .ok_or_else(|| io::Error::other("Potter target face vertices are missing"))?
        .len();
    let attributes = mesh["attributes"]
        .as_object_mut()
        .ok_or_else(|| io::Error::other("Potter geometry attributes are missing"))?;
    for case in &cases {
        let attr_name = case["attribute"]
            .as_str()
            .ok_or_else(|| io::Error::other("Geometry Attribute name is missing"))?;
        let domain = case["domain"]
            .as_str()
            .ok_or_else(|| io::Error::other("Geometry Attribute domain is missing"))?;
        let data_type = case["data_type"]
            .as_str()
            .ok_or_else(|| io::Error::other("Geometry Attribute data type is missing"))?;
        let key = match domain {
            "POINT" => "v0".to_owned(),
            "EDGE" => "e0".to_owned(),
            "FACE" | "FACE_CORNER" => "f0".to_owned(),
            _ => return Err(io::Error::other("unsupported fixture domain").into()),
        };
        let sampled = match data_type {
            "VECTOR" => case["vector"].clone(),
            "QUATERNION" => case["quaternion"].clone(),
            "FLOAT4X4" => case["matrix"].clone(),
            _ => return Err(io::Error::other("unsupported fixture type").into()),
        };
        let stored = if domain == "FACE_CORNER" {
            Value::Array(vec![sampled; corner_count])
        } else {
            sampled
        };
        let values = json!({key:stored});
        let stored_domain = domains
            .iter()
            .find(|(blender_domain, _)| *blender_domain == domain)
            .map(|(_, stored_domain)| *stored_domain)
            .ok_or_else(|| io::Error::other("Geometry Attribute domain mapping is missing"))?;
        let stored_type = match data_type {
            "VECTOR" => "float3",
            "QUATERNION" => "quaternion",
            "FLOAT4X4" => "float4x4",
            _ => return Err(io::Error::other("unsupported fixture type").into()),
        };
        attributes.insert(
            attr_name.to_owned(),
            json!({"domain":stored_domain,"type":stored_type,"values":values}),
        );
    }
    fs::write(&scene_path, serde_json::to_vec(&scene)?)?;

    let cases_json = serde_json::to_string(&cases)?;
    let cases_literal = serde_json::to_string(&cases_json)?;
    let script = format!(
        r#"
import bpy, json
from mathutils import Matrix, Quaternion, Vector
cases = json.loads({cases_literal})
def set_rotation(obj, values):
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = (values[3],values[0],values[1],values[2])
scene = bpy.context.scene
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
mesh = bpy.data.meshes.new("GeometryTargetMesh")
mesh.from_pydata([(-1,-1,0),(1,-1,0.3),(0,1,-0.2),(0.3,0.2,1)],[(0,1),(1,2),(2,0),(0,3)],[(0,1,2),(0,2,3)])
mesh.update()
target = bpy.data.objects.new("GeometryTarget", mesh)
scene.collection.objects.link(target)
target.location = (0.7,-1.1,2.3)
set_rotation(target, cases[0]["target_rotation"])
target.scale = (1.3,0.75,1.6)
owners = {{}}
for case in cases:
    owner = bpy.data.objects.new(case["id"], None)
    scene.collection.objects.link(owner)
    owner.location = (-0.8,1.2,2.5)
    set_rotation(owner, case["owner_rotation"])
    owner.scale = (0.8,1.2,1.1)
    owners[case["id"]] = owner
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
def world_matrix(obj):
    matrix = obj.evaluated_get(depsgraph).matrix_world
    return [float(matrix[row][column]) for column in range(4) for row in range(4)]
baseline = {{case["id"]: world_matrix(owners[case["id"]]) for case in cases}}
domains = {{"POINT":"POINT","EDGE":"EDGE","FACE":"FACE","FACE_CORNER":"CORNER"}}
types = {{"VECTOR":"FLOAT_VECTOR","QUATERNION":"QUATERNION","FLOAT4X4":"FLOAT4X4"}}
for case in cases:
    attribute = mesh.attributes.new(case["attribute"], types[case["data_type"]], domains[case["domain"]])
    sample = attribute.data[0]
    if case["data_type"] == "VECTOR":
        sample.vector = Vector(case["vector"])
    elif case["data_type"] == "QUATERNION":
        x,y,z,w = case["quaternion"]
        sample.value = Quaternion((w,x,y,z))
    else:
        sample.value = Matrix(case["matrix"])
    constraint = owners[case["id"]].constraints.new("GEOMETRY_ATTRIBUTE")
    constraint.target = target
    constraint.attribute_name = case["attribute"]
    constraint.data_type = case["data_type"]
    constraint.domain = case["domain"]
    constraint.sample_index = 0
    constraint.mix_mode = case["mix_mode"]
    constraint.apply_target_transform = True
    constraint.mix_loc = True
    constraint.mix_rot = True
    constraint.mix_scl = True
scene.frame_set(1)
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
print("POTTER_GEOMETRY_PARITY=" + json.dumps({{
    case["id"]: {{"matrix":world_matrix(owners[case["id"]]),"baseline":baseline[case["id"]]}}
    for case in cases
}}))
"#
    );
    let output = Command::new(blender)
        .args(["--background", "--factory-startup", "--python-expr"])
        .arg(script)
        .output()?;
    assert!(
        output.status.success(),
        "Blender Geometry Attribute fixture failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let blender_stdout = String::from_utf8_lossy(&output.stdout);
    let line = blender_stdout
        .lines()
        .find_map(|line| line.strip_prefix("POTTER_GEOMETRY_PARITY="))
        .ok_or_else(|| io::Error::other(format!(
            "Blender did not return Geometry Attribute matrices; stdout={blender_stdout}; stderr={}",
            String::from_utf8_lossy(&output.stderr)
        )))?;
    let blender_matrices: Value = serde_json::from_str(line)?;
    let mut maximum_error = 0.0_f64;
    for case in &cases {
        let id = case["id"]
            .as_str()
            .ok_or_else(|| io::Error::other("Geometry Attribute case ID is missing"))?;
        let expected = blender_matrices[id]["matrix"]
            .as_array()
            .ok_or_else(|| io::Error::other("Blender Geometry Attribute matrix is missing"))?;
        let baseline = blender_matrices[id]["baseline"]
            .as_array()
            .ok_or_else(|| io::Error::other("Blender Geometry Attribute baseline is missing"))?;
        let actual = evaluated_matrix(&project, id)?.to_cols_array();
        let mut maximum_effect = 0.0_f64;
        let mut linear_effect = 0.0_f64;
        for (index, (actual_value, expected_value)) in actual.iter().zip(expected).enumerate() {
            let expected_value = expected_value
                .as_f64()
                .ok_or_else(|| io::Error::other("Blender Geometry Attribute matrix is invalid"))?;
            let baseline_value = baseline[index].as_f64().ok_or_else(|| {
                io::Error::other("Blender Geometry Attribute baseline is invalid")
            })?;
            maximum_effect = maximum_effect.max((expected_value - baseline_value).abs());
            if [0, 1, 2, 4, 5, 6, 8, 9, 10].contains(&index) {
                linear_effect = linear_effect.max((expected_value - baseline_value).abs());
            }
            let error = (actual_value - expected_value).abs();
            maximum_error = maximum_error.max(error);
            assert!(
                error < 2.0e-4,
                "{id} Blender Geometry Attribute component {index} differs: Potter={actual_value}, Blender={expected_value}, actual={actual:?}, Blender matrix={expected:?}, baseline={baseline:?}, case={case}"
            );
        }
        assert!(
            maximum_effect > 1.0e-2,
            "{id} Geometry Attribute fixture had no material effect over baseline"
        );
        if case["data_type"] == "FLOAT4X4" {
            assert!(
                linear_effect > 1.0e-2,
                "{id} FLOAT4X4 result did not change its rotation/scale basis"
            );
        }
    }
    eprintln!(
        "Blender Geometry Attribute matrix max error across 3 data types, 4 domains, and 5 mix modes: {maximum_error:.6e}"
    );
    Ok(())
}
#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the end-to-end solve fixture exercises Blender and Potter inputs, outputs, and tolerances in one scenario"
)]
fn blender_tracking_camera_and_object_solver_parity() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping tracking Blender parity test: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = fs::canonicalize(directory.path())?;
    let script_path = root.join("tracking_solver_fixture.py");
    let script = r#"
import bpy
import json
import os
import sys

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
for frame in range(1, 4):
    image = bpy.data.images.new("Source%d" % frame, 160, 90)
    image.filepath_raw = os.path.join(root, "frame_%04d.png" % frame)
    image.file_format = "PNG"
    image.save()
clip = bpy.data.movieclips.load(os.path.join(root, "frame_0001.png"), check_existing=False)
scene = bpy.context.scene
scene.active_clip = clip
scene.render.resolution_x = 160
scene.render.resolution_y = 100
clip.tracking.camera.units = "MILLIMETERS"
clip.tracking.camera.focal_length = 50.0
clip.tracking.camera.sensor_width = 36.0
background_points = [
    (-2,-1,10),(1,-1,11),(2,2,12),(-1,3,13),
    (3,-2,14),(-3,2,15),(0.3,0.7,10.8),(1.7,0.2,13.5)
]
object_points = [
    (-1,-1,-1),(1,-1,-0.5),(-1,1,0.2),(1,1,0.8),
    (-0.4,0.3,1.4),(0.7,-0.2,1.8),(-0.6,0.8,2.3),(0.2,-0.7,2.6)
]
focal_pixels = 50.0 / 36.0 * 160
background_tracks = []
for index, point in enumerate(background_points):
    track = clip.tracking.tracks.new(name="Static%d" % index)
    background_tracks.append(track)
    for frame, camera_x in ((1,0.0),(2,0.35),(3,1.0)):
        x,y,z = point
        track.markers.insert_frame(
            frame,
            co=(0.5 + focal_pixels * (x-camera_x) / z / 160.0,
                0.5 - focal_pixels * y / z / 90.0))
camera_tracking_object = clip.tracking.objects.active
camera_tracking_object.keyframe_a = 1
camera_tracking_object.keyframe_b = 3
area = next(area for area in bpy.context.screen.areas if area.type == "VIEW_3D")
area.type = "CLIP_EDITOR"
area.spaces.active.clip = clip
with bpy.context.temp_override(area=area, space_data=area.spaces.active):
    bpy.ops.clip.solve_camera()
moving_tracking_object = clip.tracking.objects.new(name="Moving")
clip.tracking.objects.active = moving_tracking_object
moving_tracking_object.keyframe_a = 1
moving_tracking_object.keyframe_b = 3
moving_tracks = []
for index, point in enumerate(object_points):
    track = moving_tracking_object.tracks.new(name="Object%d" % index)
    moving_tracks.append(track)
    for frame, object_x in ((1,0.1),(2,0.3),(3,0.5)):
        camera_x = (0.0,0.35,1.0)[frame-1]
        x,y,z = point
        track.markers.insert_frame(
            frame,
            co=(0.5 + focal_pixels * (x+object_x-camera_x) / (z+8.0) / 160.0,
                0.5 - focal_pixels * y / (z+8.0) / 90.0))
with bpy.context.temp_override(area=area, space_data=area.spaces.active):
    bpy.ops.clip.solve_camera()
scene.render.resolution_x = 160
scene.render.resolution_y = 100
camera_data = bpy.data.cameras.new("CameraData")
camera_data.lens = 50.0
camera_data.sensor_width = 36.0
camera = bpy.data.objects.new("SolvedCamera", camera_data)
scene.collection.objects.link(camera)
solver_camera_data = bpy.data.cameras.new("ObjectSolverCameraData")
solver_camera = bpy.data.objects.new("ObjectSolverCamera", solver_camera_data)
scene.collection.objects.link(solver_camera)
follow_camera_data = bpy.data.cameras.new("FollowTrackCameraData")
follow_camera_data.lens = 50.0
follow_camera_data.sensor_width = 36.0
follow_camera = bpy.data.objects.new("FollowTrackCamera", follow_camera_data)
scene.collection.objects.link(follow_camera)
explicit_camera = bpy.data.objects.new("ExplicitCamera", bpy.data.cameras.new("ExplicitCameraData"))
explicit_camera.data.lens = 50.0
explicit_camera.data.sensor_width = 36.0
scene.collection.objects.link(explicit_camera)
owner = bpy.data.objects.new("ObjectSolverOwner", None)
scene.collection.objects.link(owner)
explicit_owner = bpy.data.objects.new("ExplicitObjectSolverOwner", None)
scene.collection.objects.link(explicit_owner)
inverse_owner = bpy.data.objects.new("InverseSolverOwner", None)
inverse_owner.location = (1.0, 2.0, 3.0)
scene.collection.objects.link(inverse_owner)
follow_settings = [
    ("follow_stretch_2d", "STRETCH", False),
    ("follow_fit_2d", "FIT", False),
    ("follow_crop_2d", "CROP", False),
    ("follow_stretch_3d", "STRETCH", True),
    ("follow_fit_3d", "FIT", True),
    ("follow_crop_3d", "CROP", True),
]
followers = {}
for name, frame_method, use_3d in follow_settings:
    follow_owner = bpy.data.objects.new(name, None)
    follow_owner.location = (0.0, 0.0, -4.0)
    scene.collection.objects.link(follow_owner)
    followers[name] = follow_owner
bpy.context.view_layer.update()
def matrix_values(matrix):
    return [float(matrix[row][column]) for column in range(4) for row in range(4)]
def world_matrix(obj, depsgraph=None):
    depsgraph = depsgraph or bpy.context.evaluated_depsgraph_get()
    return matrix_values(obj.evaluated_get(depsgraph).matrix_world)
baseline_inverse_matrix = inverse_owner.matrix_world.copy()
baseline = {
    "camera_active": world_matrix(camera),
    "solver_camera": world_matrix(solver_camera),
    "camera_follow": world_matrix(follow_camera),
    "camera_explicit": world_matrix(explicit_camera),
    "object_active": world_matrix(owner),
    "object_explicit": world_matrix(explicit_owner),
    "inverse": world_matrix(inverse_owner),
    "followers": {name: world_matrix(follow_owner) for name, follow_owner in followers.items()}
}
camera_solver = camera.constraints.new("CAMERA_SOLVER")
camera_solver.clip = None
camera_solver.use_active_clip = True
solver_camera_constraint = solver_camera.constraints.new("CAMERA_SOLVER")
solver_camera_constraint.clip = None
solver_camera_constraint.use_active_clip = True
explicit_camera_solver = explicit_camera.constraints.new("CAMERA_SOLVER")
explicit_camera_solver.clip = clip
explicit_camera_solver.use_active_clip = False
for solver_owner, use_active in ((owner, True), (explicit_owner, False)):
    object_solver = solver_owner.constraints.new("OBJECT_SOLVER")
    object_solver.clip = None if use_active else clip
    object_solver.object = moving_tracking_object.name
    object_solver.camera = solver_camera
    object_solver.use_active_clip = use_active
scene.frame_set(1)
view_layer = bpy.context.view_layer
view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
depsgraph.update()
for name, frame_method, use_3d in follow_settings:
    follow_constraint = followers[name].constraints.new("FOLLOW_TRACK")
    follow_constraint.clip = None
    follow_constraint.track = background_tracks[0].name
    follow_constraint.camera = follow_camera
    follow_constraint.use_active_clip = True
    follow_constraint.use_3d_position = use_3d
    follow_constraint.use_undistorted_position = False
    follow_constraint.frame_method = frame_method
    follow_constraint.depth_object = None
def marker_values(track):
    return [{"frame": int(marker.frame),
             "co": [float(marker.co.x), float(marker.co.y)]}
            for marker in track.markers]
expected = {}
# Avoid Blender's scheduling-sensitive deferred inverse capture; derive it from this run's owner path.
owner_initial_matrix = None
for frame in (1,2,3):
    scene.frame_set(frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    object_active_matrix = owner.evaluated_get(depsgraph).matrix_world.copy()
    if frame == 1:
        owner_initial_matrix = object_active_matrix.copy()
    inverse_matrix = (
        object_active_matrix @ owner_initial_matrix.inverted() @ baseline_inverse_matrix
    )
    expected[str(frame)] = {
        "camera_active": world_matrix(camera, depsgraph),
        "solver_camera": world_matrix(solver_camera, depsgraph),
        "camera_follow": world_matrix(follow_camera, depsgraph),
        "camera_explicit": world_matrix(explicit_camera, depsgraph),
        "object_active": matrix_values(object_active_matrix),
        "object_explicit": world_matrix(explicit_owner, depsgraph),
        # Match the captured-inverse contract using Blender's same-run solver path.
        "inverse": matrix_values(inverse_matrix),
        "followers": {
            name: world_matrix(follow_owner, depsgraph)
            for name, follow_owner in followers.items()
        }
    }
with open(os.path.join(root, "tracking_expected.json"), "w", encoding="utf-8") as output:
    json.dump({
        "width": int(clip.size[0]),
        "height": int(clip.size[1]),
        "camera_tracks": [{"name": track.name,"markers": marker_values(track)}
                          for track in background_tracks],
        "object_tracks": [{"name": track.name,"markers": marker_values(track)}
                          for track in moving_tracks],
        "camera_reconstruction_points": [
            {"has_bundle": bool(track.has_bundle),
             "co": [float(value) for value in track.bundle]}
            for track in background_tracks],
        "background_points": background_points,
        "object_points": object_points,
        "camera_reconstruction_error": float(camera_tracking_object.reconstruction.average_error),
        "object_reconstruction_error": float(moving_tracking_object.reconstruction.average_error),
        "baselines": baseline,
        "frames": expected
    }, output)
"#;
    fs::write(&script_path, script)?;
    let output = Command::new(&blender)
        .args(["--background", "--factory-startup", "--python"])
        .arg(&script_path)
        .args(["--", root.to_str().ok_or("temporary path is not UTF-8")?])
        .output()?;
    assert!(
        output.status.success(),
        "Blender tracking fixture failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let expected: Value = serde_json::from_slice(&fs::read(root.join("tracking_expected.json"))?)?;
    let width = u32::try_from(
        expected["width"]
            .as_u64()
            .ok_or_else(|| io::Error::other("Blender clip width is missing"))?,
    )?;
    let height = u32::try_from(
        expected["height"]
            .as_u64()
            .ok_or_else(|| io::Error::other("Blender clip height is missing"))?,
    )?;
    let follow_cases = [
        ("follow_stretch_2d", "STRETCH", false),
        ("follow_fit_2d", "FIT", false),
        ("follow_crop_2d", "CROP", false),
        ("follow_stretch_3d", "STRETCH", true),
        ("follow_fit_3d", "FIT", true),
        ("follow_crop_3d", "CROP", true),
    ];
    let mut operations = vec![
        json!({"op":"camera.create","id":"camera","name":"Solved Camera","lens_mm":50.0,"sensor_width_mm":36.0}),
        json!({"op":"camera.create","id":"solver_camera","name":"Object Solver Camera","lens_mm":50.0,"sensor_width_mm":36.0}),
        json!({"op":"camera.create","id":"follow_camera","name":"Follow Track Camera","lens_mm":50.0,"sensor_width_mm":36.0}),
        json!({"op":"camera.create","id":"explicit_camera","name":"Explicit Camera","lens_mm":50.0,"sensor_width_mm":36.0}),
        json!({"op":"node.create","id":"object_owner","kind":"empty"}),
        json!({"op":"node.create","id":"explicit_object_owner","kind":"empty"}),
        json!({"op":"node.create","id":"inverse_owner","kind":"empty","transform":{"translation":[1.0,2.0,3.0]}}),
        json!({"op":"tracking.clip_create","id":"clip","name":"Synthetic","width":width,"height":height}),
    ];
    for (owner, _, _) in follow_cases {
        operations.push(json!({
            "op":"node.create",
            "id":owner,
            "kind":"empty",
            "transform":{"translation":[0.0,0.0,-4.0]}
        }));
    }
    let mut camera_track_ids = Vec::new();
    for (index, track) in expected["camera_tracks"]
        .as_array()
        .ok_or_else(|| io::Error::other("Blender camera tracks are missing"))?
        .iter()
        .enumerate()
    {
        let id = format!("static_{index}");
        camera_track_ids.push(id.clone());
        let first = track["markers"][0]["co"]
            .as_array()
            .ok_or_else(|| io::Error::other("camera marker is missing"))?;
        operations.push(json!({
            "op":"tracking.track_add",
            "id":"clip",
            "track":id,
            "name":id,
            "frame":1,
            "co":[
                first[0].as_f64().ok_or("invalid camera marker")? * f64::from(width),
                (1.0 - first[1].as_f64().ok_or("invalid camera marker")?) * f64::from(height)
            ]
        }));
    }
    let background_points = expected["background_points"]
        .as_array()
        .ok_or_else(|| io::Error::other("synthetic static landmarks are missing"))?;
    for frame in 1_i64..=3 {
        let observations = expected["camera_tracks"]
            .as_array()
            .ok_or_else(|| io::Error::other("Blender camera tracks are missing"))?
            .iter()
            .enumerate()
            .map(|(index, track)| {
                let marker = track["markers"]
                    .as_array()
                    .and_then(|markers| markers.iter().find(|marker| marker["frame"] == frame))
                    .ok_or_else(|| io::Error::other("camera frame marker is missing"))?;
                let world = background_points
                    .get(index)
                    .ok_or_else(|| io::Error::other("static landmark correspondence is missing"))?;
                Ok(json!({
                    "world":world,
                    "image":[
                        marker["co"][0].as_f64().ok_or("invalid camera marker")? * f64::from(width),
                        (1.0 - marker["co"][1].as_f64().ok_or("invalid camera marker")?) * f64::from(height)
                    ]
                }))
            })
            .collect::<TestResult<Vec<_>>>()?;
        operations.push(json!({
            "op":"tracking.solve_camera",
            "id":"clip",
            "frame":frame,
            "observations":observations
        }));
    }
    let first_batch = ops::apply_batch(
        &SceneDoc::new("00000000-0000-4000-8000-000000000002".to_owned()),
        &json!({"schema_version":1,"base_revision":0,"operations":operations}),
    )?;
    let mut first_batch = first_batch;
    let clip_id = Id::new("clip".to_owned())?;
    let clip = first_batch
        .doc
        .movie_clips
        .get_mut(&clip_id)
        .ok_or_else(|| io::Error::other("Potter tracking clip is missing"))?;
    let object_tracks = expected["object_tracks"]
        .as_array()
        .ok_or_else(|| io::Error::other("Blender object tracks are missing"))?;
    let mut object_track_ids = Vec::new();
    for (index, track) in object_tracks.iter().enumerate() {
        let id = format!("object_{index}");
        object_track_ids.push(id.clone());
        let markers = track["markers"]
            .as_array()
            .ok_or_else(|| io::Error::other("Blender object markers are missing"))?;
        let first = markers
            .iter()
            .find(|marker| marker["frame"] == 1)
            .ok_or_else(|| io::Error::other("frame-1 object marker is missing"))?;
        clip.tracking
            .tracks
            .push(potter_core::model::TrackingTrack {
                id: id.clone(),
                name: id,
                markers: vec![TrackingMarker {
                    frame: 1.0,
                    co: [
                        first["co"][0].as_f64().ok_or("invalid object marker")? * f64::from(width),
                        (1.0 - first["co"][1].as_f64().ok_or("invalid object marker")?)
                            * f64::from(height),
                    ],
                    ..TrackingMarker::default()
                }],
            });
    }
    for (id, camera_track) in camera_track_ids.iter().zip(
        expected["camera_tracks"]
            .as_array()
            .ok_or_else(|| io::Error::other("Blender camera tracks are missing"))?,
    ) {
        let track = clip
            .tracking
            .tracks
            .iter_mut()
            .find(|track| &track.id == id)
            .ok_or_else(|| io::Error::other("Potter camera track is missing"))?;
        for marker in camera_track["markers"]
            .as_array()
            .ok_or_else(|| io::Error::other("Blender camera markers are missing"))?
            .iter()
            .filter(|marker| marker["frame"].as_i64().is_some_and(|frame| frame > 1))
        {
            track.markers.push(TrackingMarker {
                frame: f64::from(i32::try_from(
                    marker["frame"]
                        .as_i64()
                        .ok_or_else(|| io::Error::other("camera frame is missing"))?,
                )?),
                co: [
                    marker["co"][0].as_f64().ok_or("invalid camera marker")? * f64::from(width),
                    (1.0 - marker["co"][1].as_f64().ok_or("invalid camera marker")?)
                        * f64::from(height),
                ],
                ..TrackingMarker::default()
            });
        }
    }
    for track in &mut clip.tracking.tracks {
        if let Some(object_track) = object_track_ids
            .iter()
            .position(|id| id == &track.id)
            .and_then(|index| object_tracks.get(index))
        {
            for marker in object_track["markers"]
                .as_array()
                .ok_or_else(|| io::Error::other("Blender object markers are missing"))?
                .iter()
                .filter(|marker| marker["frame"].as_i64().is_some_and(|frame| frame > 1))
            {
                track.markers.push(TrackingMarker {
                    frame: f64::from(i32::try_from(
                        marker["frame"]
                            .as_i64()
                            .ok_or_else(|| io::Error::other("object frame is missing"))?,
                    )?),
                    co: [
                        marker["co"][0].as_f64().ok_or("invalid object marker")? * f64::from(width),
                        (1.0 - marker["co"][1].as_f64().ok_or("invalid object marker")?)
                            * f64::from(height),
                    ],
                    ..TrackingMarker::default()
                });
            }
        }
    }
    let camera_bundle_points = expected["camera_reconstruction_points"]
        .as_array()
        .ok_or_else(|| io::Error::other("Blender camera track bundles are missing"))?;
    let mut reconstructed_points = camera_bundle_points
        .iter()
        .zip(&camera_track_ids)
        .map(|(world, id)| {
            if world["has_bundle"].as_bool() != Some(true) {
                return Err(
                    io::Error::other("Blender camera track has no reconstruction bundle").into(),
                );
            }
            let coordinates = world["co"]
                .as_array()
                .ok_or_else(|| io::Error::other("static landmark coordinates are missing"))?;
            Ok(ReconstructedPoint {
                track: id.clone(),
                co: [
                    coordinates[0].as_f64().ok_or("invalid static landmark")?,
                    coordinates[1].as_f64().ok_or("invalid static landmark")?,
                    coordinates[2].as_f64().ok_or("invalid static landmark")?,
                ],
            })
        })
        .collect::<TestResult<Vec<_>>>()?;
    let object_points = expected["object_points"]
        .as_array()
        .ok_or_else(|| io::Error::other("synthetic moving-object landmarks are missing"))?;
    reconstructed_points.extend(
        object_points
            .iter()
            .zip(&object_track_ids)
            .map(|(world, id)| {
                let coordinates = world
                    .as_array()
                    .ok_or_else(|| io::Error::other("object landmark coordinates are missing"))?;
                Ok(ReconstructedPoint {
                    track: id.clone(),
                    co: [
                        coordinates[0].as_f64().ok_or("invalid object landmark")?,
                        coordinates[1].as_f64().ok_or("invalid object landmark")?,
                        coordinates[2].as_f64().ok_or("invalid object landmark")?,
                    ],
                })
            })
            .collect::<TestResult<Vec<_>>>()?,
    );
    clip.tracking.reconstruction.points = reconstructed_points;
    let solved_objects = ops::apply_batch(
        &first_batch.doc,
        &json!({
            "schema_version":1,
            "base_revision":first_batch.doc.revision,
            "operations":[{
                "op":"tracking.solve_object",
                "id":"clip",
                "object":"moving",
                "name":"Moving",
                "tracks":object_track_ids
            }]
        }),
    )?;
    let object_reprojection_errors =
        object_reprojection_errors(&solved_objects.doc, &clip_id, "moving", width, height)?;
    let maximum_object_reprojection_error = object_reprojection_errors
        .iter()
        .map(|(_, error)| *error)
        .fold(0.0_f64, f64::max);
    let blender_object_reconstruction_error = expected["object_reconstruction_error"]
        .as_f64()
        .ok_or_else(|| io::Error::other("Blender object reprojection error is missing"))?;
    eprintln!(
        "Tracking object maximum reprojection error: Potter={maximum_object_reprojection_error:.6e}px, Blender average={blender_object_reconstruction_error:.6e}"
    );
    let mut failures = Vec::new();
    if maximum_object_reprojection_error > 2.0e-2 {
        failures.push(format!(
            "Potter object reprojection RMS {maximum_object_reprojection_error:.6e}px exceeds 0.02px"
        ));
    }
    if blender_object_reconstruction_error > 2.0e-2 {
        failures.push(format!(
            "Blender reported object reconstruction error {blender_object_reconstruction_error:.6e} exceeds 0.02 in Blender's native error units"
        ));
    }
    let mut constraint_operations = vec![
        json!({"op":"scene.update","target":{"id":"scene_main"},"set":{"active_clip":"clip"}}),
        json!({"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_x":160,"resolution_y":100}}),
        json!({"op":"constraint.create","target":{"id":"camera"},"id":"camera_solver","type":"camera_solver","params":{"use_active_clip":true,"clip":null}}),
        json!({"op":"constraint.create","target":{"id":"solver_camera"},"id":"active_camera_solver","type":"camera_solver","params":{"use_active_clip":true,"clip":null}}),
        json!({"op":"constraint.create","target":{"id":"explicit_camera"},"id":"explicit_camera_solver","type":"camera_solver","params":{"use_active_clip":false,"clip":"clip"}}),
        json!({"op":"constraint.create","target":{"id":"object_owner"},"id":"object_solver","type":"object_solver","constraint_target":"solver_camera","params":{"use_active_clip":true,"clip":null,"object":"moving","camera":"solver_camera","set_inverse_pending":false}}),
        json!({"op":"constraint.create","target":{"id":"explicit_object_owner"},"id":"explicit_object_solver","type":"object_solver","constraint_target":"solver_camera","params":{"use_active_clip":false,"clip":"clip","object":"moving","camera":"solver_camera","set_inverse_pending":false}}),
        json!({"op":"constraint.create","target":{"id":"inverse_owner"},"id":"inverse_solver","type":"object_solver","constraint_target":"solver_camera","params":{"use_active_clip":true,"clip":null,"object":"moving","camera":"solver_camera","set_inverse_pending":true}}),
    ];
    for (owner, frame_method, use_3d_position) in follow_cases {
        constraint_operations.push(json!({
            "op":"constraint.create",
            "target":{"id":owner},
            "id":"follow_track",
            "type":"follow_track",
            "constraint_target":"follow_camera",
            "params":{
                "use_active_clip":true,
                "clip":null,
                "track":"static_0",
                "object":"",
                "use_3d_position":use_3d_position,
                "use_undistorted_position":false,
                "frame_method":frame_method,
                "camera":"follow_camera",
                "depth_object":null
            }
        }));
    }
    let constrained = ops::apply_batch(
        &solved_objects.doc,
        &json!({
            "schema_version":1,
            "base_revision":solved_objects.doc.revision,
            "operations":constraint_operations
        }),
    )?;
    let comparisons = [
        ("camera", "camera_active", 1.0e-2),
        ("solver_camera", "solver_camera", 1.0e-2),
        ("explicit_camera", "camera_explicit", 1.0e-2),
        ("object_owner", "object_active", 2.0e-2),
        ("explicit_object_owner", "object_explicit", 2.0e-2),
        ("inverse_owner", "inverse", 2.0e-2),
        ("follow_stretch_2d", "followers", 1.0e-2),
        ("follow_fit_2d", "followers", 1.0e-2),
        ("follow_crop_2d", "followers", 1.0e-2),
        ("follow_stretch_3d", "followers", 1.0e-2),
        ("follow_fit_3d", "followers", 1.0e-2),
        ("follow_crop_3d", "followers", 1.0e-2),
    ];
    let mut potter_frames = Vec::new();
    let mut potter_solver_cameras = Vec::new();
    let mut potter_follow_cameras = Vec::new();
    for frame in [1.0, 2.0, 3.0] {
        let snapshot = Snapshot::evaluate(
            &constrained.doc,
            &EvaluationContext {
                frame: Some(frame),
                ..EvaluationContext::default()
            },
        )?;
        let matrices = comparisons
            .iter()
            .map(|(node_id, _, _)| {
                let id = Id::new((*node_id).to_owned())?;
                snapshot
                    .nodes
                    .get(&id)
                    .map(|node| DMat4::from_cols_array(&node.world_matrix))
                    .ok_or_else(|| io::Error::other("evaluated tracking node is missing").into())
            })
            .collect::<TestResult<Vec<_>>>()?;
        let solver_camera_id = Id::new("solver_camera".to_owned())?;
        let solver_camera = snapshot
            .nodes
            .get(&solver_camera_id)
            .map(|node| DMat4::from_cols_array(&node.world_matrix))
            .ok_or_else(|| io::Error::other("evaluated solver camera is missing"))?;
        let follow_camera_id = Id::new("follow_camera".to_owned())?;
        let follow_camera = snapshot
            .nodes
            .get(&follow_camera_id)
            .map(|node| DMat4::from_cols_array(&node.world_matrix))
            .ok_or_else(|| io::Error::other("evaluated Follow Track camera is missing"))?;
        potter_frames.push(matrices);
        potter_solver_cameras.push(solver_camera);
        potter_follow_cameras.push(follow_camera);
    }

    // Monocular reconstruction has a scale gauge. Compare camera paths in their
    // own frame-1 basis and normalize translation by the frame-1-to-frame-3 travel.
    // Object/follow matrices are compared in camera-local reconstruction units.
    let blender_reconstruction_scale = (blender_matrix(&expected["frames"]["3"]["camera_active"])?
        .w_axis
        .truncate()
        - blender_matrix(&expected["frames"]["1"]["camera_active"])?
            .w_axis
            .truncate())
    .length();
    let potter_reconstruction_scale = (potter_solver_cameras[2].w_axis.truncate()
        - potter_solver_cameras[0].w_axis.truncate())
    .length();
    assert!(
        blender_reconstruction_scale > 1.0e-6 && potter_reconstruction_scale > 1.0e-6,
        "camera reconstruction did not establish a scale gauge"
    );
    let mut maximum_errors = vec![0.0_f64; comparisons.len()];
    let object_owner_index = comparisons
        .iter()
        .position(|(node_id, _, _)| *node_id == "object_owner")
        .ok_or_else(|| io::Error::other("object solver parity case is missing"))?;
    let inverse_owner_index = comparisons
        .iter()
        .position(|(node_id, _, _)| *node_id == "inverse_owner")
        .ok_or_else(|| io::Error::other("inverse-pending parity case is missing"))?;
    let baseline_inverse = blender_matrix(&expected["baselines"]["inverse"])?;
    let potter_inverse_initial = potter_frames[0][inverse_owner_index];
    let potter_initial_error = max_matrix_error(potter_inverse_initial, baseline_inverse);
    if potter_initial_error > 1.0e-2 {
        failures.push(format!(
            "set_inverse_pending did not preserve the initial owner transform at its capture frame: Potter={potter_initial_error:.6e}"
        ));
    }
    let mut inverse_path_error = 0.0_f64;
    for frame_index in 1..3 {
        let owner_delta = potter_frames[frame_index][object_owner_index]
            * potter_frames[0][object_owner_index].inverse();
        let inverse_delta =
            potter_frames[frame_index][inverse_owner_index] * potter_inverse_initial.inverse();
        inverse_path_error = inverse_path_error.max(max_matrix_error(owner_delta, inverse_delta));
        let frame = (frame_index + 1).to_string();
        let inverse_displacement = max_matrix_error(
            potter_frames[frame_index][inverse_owner_index],
            potter_inverse_initial,
        );
        if inverse_displacement <= 1.0e-2 {
            failures.push(format!(
                "set_inverse_pending did not follow the moving reconstruction at frame {frame}: Potter={inverse_displacement:.6e}"
            ));
        }
    }
    eprintln!("Potter Object Solver inverse-pending path error: {inverse_path_error:.6e}");
    if inverse_path_error > 1.0e-4 {
        failures.push(format!(
            "set_inverse_pending path differs from the uncompensated Object Solver path by {inverse_path_error:.6e}"
        ));
    }
    maximum_errors[inverse_owner_index] = inverse_path_error;
    // The two moving-object solves use independent scale/pose gauges. Compare their raw-marker
    // reprojection errors above instead of treating their reconstruction-local matrices as equal.
    for (case_index, (node_id, expected_key, tolerance)) in comparisons.iter().enumerate() {
        let camera_comparison = matches!(
            *expected_key,
            "camera_active" | "camera_explicit" | "solver_camera"
        );
        let object_comparison = matches!(
            *expected_key,
            "object_active" | "object_explicit" | "inverse"
        );
        if object_comparison {
            if *expected_key != "inverse" {
                maximum_errors[case_index] = maximum_object_reprojection_error;
            }
            continue;
        }
        let blender_first = if *expected_key == "followers" {
            blender_matrix(&expected["frames"]["1"]["followers"][*node_id])?
        } else {
            blender_matrix(&expected["frames"]["1"][*expected_key])?
        };
        let blender_last = if *expected_key == "followers" {
            blender_matrix(&expected["frames"]["3"]["followers"][*node_id])?
        } else {
            blender_matrix(&expected["frames"]["3"][*expected_key])?
        };
        let potter_first = potter_frames[0][case_index];
        let potter_last = potter_frames[2][case_index];
        let mut blender_scale = 1.0;
        let mut potter_scale = 1.0;
        if camera_comparison {
            blender_scale =
                (blender_last.w_axis.truncate() - blender_first.w_axis.truncate()).length();
            potter_scale =
                (potter_last.w_axis.truncate() - potter_first.w_axis.truncate()).length();
            assert!(
                blender_scale > 1.0e-6 && potter_scale > 1.0e-6,
                "{node_id} camera solve did not recover nonzero frame travel"
            );
        }
        for (frame_index, frame) in [1.0, 2.0, 3.0].into_iter().enumerate() {
            let frame_key = frame.to_string();
            let blender_world = if *expected_key == "followers" {
                blender_matrix(&expected["frames"][&frame_key]["followers"][*node_id])?
            } else {
                blender_matrix(&expected["frames"][&frame_key][*expected_key])?
            };
            let potter_world = potter_frames[frame_index][case_index];
            let (actual, expected_matrix) = if camera_comparison {
                let blender_relative = blender_first.inverse() * blender_world;
                let potter_relative = potter_first.inverse() * potter_world;
                let mut blender_relative = blender_relative.to_cols_array();
                let mut potter_relative = potter_relative.to_cols_array();
                for value in &mut blender_relative[12..15] {
                    *value /= blender_scale;
                }
                for value in &mut potter_relative[12..15] {
                    *value /= potter_scale;
                }
                (potter_relative, blender_relative)
            } else {
                let blender_camera =
                    blender_matrix(&expected["frames"][&frame_key]["camera_follow"])?;
                let potter_camera = potter_follow_cameras[frame_index];
                (
                    (potter_camera.inverse() * potter_world).to_cols_array(),
                    (blender_camera.inverse() * blender_world).to_cols_array(),
                )
            };
            for (component, (actual_component, expected_component)) in
                actual.iter().zip(expected_matrix).enumerate()
            {
                let error = (actual_component - expected_component).abs();
                maximum_errors[case_index] = maximum_errors[case_index].max(error);
                if error >= *tolerance {
                    failures.push(format!(
                        "{node_id} frame {frame} reconstruction-local matrix component {component} differs by {error}; Potter={actual:?}; Blender={expected_matrix:?}; tolerance={tolerance} scene units (camera scale gauge normalized for camera poses)"
                    ));
                }
            }
        }
    }
    eprintln!(
        "Blender object reconstruction reported average reprojection error: {blender_object_reconstruction_error:.6e} native units"
    );
    for (case_index, (node_id, expected_key, _)) in comparisons.iter().enumerate() {
        match *expected_key {
            "object_active" | "object_explicit" => eprintln!(
                "Potter tracking {node_id} maximum reprojection RMS: {:.6e}px",
                maximum_errors[case_index]
            ),
            "inverse" => eprintln!(
                "Tracking {node_id} max inverse path error: {:.6e}",
                maximum_errors[case_index]
            ),
            _ => eprintln!(
                "Blender tracking {node_id} max per-element error: {:.6e}",
                maximum_errors[case_index]
            ),
        }
        let baseline = if *expected_key == "followers" {
            blender_matrix(&expected["baselines"]["followers"][*node_id])?
        } else {
            blender_matrix(&expected["baselines"][*expected_key])?
        };
        let frame_three = if *expected_key == "followers" {
            blender_matrix(&expected["frames"]["3"]["followers"][*node_id])?
        } else {
            blender_matrix(&expected["frames"]["3"][*expected_key])?
        };
        let baseline_displacement = baseline
            .to_cols_array()
            .iter()
            .zip(frame_three.to_cols_array())
            .map(|(baseline, evaluated)| (baseline - evaluated).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            baseline_displacement > 1.0e-2,
            "{node_id} Blender result did not differ materially from its unconstrained baseline"
        );
    }
    let camera = blender_matrix(&expected["frames"]["2"]["camera_follow"])?;
    for (method, fit_owner, baseline_owner) in [
        ("FIT", "follow_fit_2d", "follow_stretch_2d"),
        ("CROP", "follow_crop_2d", "follow_stretch_2d"),
    ] {
        let fit = blender_matrix(&expected["frames"]["2"]["followers"][fit_owner])?;
        let baseline = blender_matrix(&expected["frames"]["2"]["followers"][baseline_owner])?;
        let difference = (camera.inverse() * fit).to_cols_array();
        let baseline = (camera.inverse() * baseline).to_cols_array();
        let difference = difference
            .iter()
            .zip(baseline)
            .map(|(first, second)| (first - second).abs())
            .fold(0.0_f64, f64::max);
        if difference <= 1.0e-2 {
            failures.push(format!(
                "Follow Track {method} did not differ from STRETCH for the mismatched clip/render aspect: {difference}"
            ));
        }
    }
    let two_d = blender_matrix(&expected["frames"]["2"]["followers"]["follow_stretch_2d"])?;
    let three_d = blender_matrix(&expected["frames"]["2"]["followers"]["follow_stretch_3d"])?;
    let position_mode_difference = (camera.inverse() * two_d)
        .to_cols_array()
        .iter()
        .zip((camera.inverse() * three_d).to_cols_array())
        .map(|(two_d, three_d)| (two_d - three_d).abs())
        .fold(0.0_f64, f64::max);
    if position_mode_difference <= 1.0e-2 {
        failures.push(format!(
            "Follow Track use_3d_position did not produce a distinct transform: {position_mode_difference}"
        ));
    }
    let camera_markers = expected["camera_tracks"][0]["markers"]
        .as_array()
        .ok_or_else(|| io::Error::other("camera raw markers are missing"))?;
    let marker_delta = (camera_markers[0]["co"][0]
        .as_f64()
        .ok_or_else(|| io::Error::other("frame-1 camera marker is invalid"))?
        - camera_markers[2]["co"][0]
            .as_f64()
            .ok_or_else(|| io::Error::other("frame-3 camera marker is invalid"))?)
    .abs()
        * f64::from(width);
    assert!(
        marker_delta > 1.0,
        "synthetic camera tracks did not move materially in image pixels"
    );
    let object_markers = expected["object_tracks"][0]["markers"]
        .as_array()
        .ok_or_else(|| io::Error::other("moving-object raw markers are missing"))?;
    let object_marker_delta = (object_markers[0]["co"][0]
        .as_f64()
        .ok_or_else(|| io::Error::other("frame-1 object marker is invalid"))?
        - object_markers[2]["co"][0]
            .as_f64()
            .ok_or_else(|| io::Error::other("frame-3 object marker is invalid"))?)
    .abs()
        * f64::from(width);
    assert!(
        object_marker_delta > 1.0,
        "synthetic moving-object tracks did not move materially in image pixels"
    );
    assert!(
        failures.is_empty(),
        "tracking parity failures:\n{}",
        failures.join("\n")
    );
    Ok(())
}
#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the Transform Cache parity fixture keeps cache construction and frame-by-frame matrix checks together"
)]
fn blender_pose_bone_transform_cache_matrix_parity() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping pose Transform Cache parity test: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = fs::canonicalize(directory.path())?;
    let abc_path = root.join("animated.abc");
    let output_path = root.join("blender_pose_matrices.json");
    let script_path = root.join("make_pose_cache.py");
    let script = r#"
import bpy
import json
import os
import sys

cache_path, output_path = sys.argv[sys.argv.index("--") + 1:sys.argv.index("--") + 3]
scene = bpy.context.scene
mesh = bpy.data.meshes.new("AnimatedMesh")
mesh.from_pydata([(0,0,0),(1,0,0),(0,1,0)], [], [(0,1,2)])
source = bpy.data.objects.new("Animated", mesh)
scene.collection.objects.link(source)
for frame, location in ((1,0.0),(2,1.0),(3,2.0)):
    source.location.x = location
    source.rotation_euler.z = 0.1 * (frame - 1)
    source.keyframe_insert(data_path="location", frame=frame)
    source.keyframe_insert(data_path="rotation_euler", frame=frame)
source.select_set(True)
bpy.context.view_layer.objects.active = source
bpy.ops.wm.alembic_export(
    filepath=cache_path,
    start=1,
    end=3,
    selected=True,
    flatten=False,
    uvs=False,
    normals=False,
    as_background_job=False)
assert os.path.isfile(cache_path), "Blender did not write the Alembic fixture"
bpy.ops.object.select_all(action="DESELECT")
armature_data = bpy.data.armatures.new("Rig")
armature = bpy.data.objects.new("Rig", armature_data)
scene.collection.objects.link(armature)
armature.select_set(True)
bpy.context.view_layer.objects.active = armature
bpy.ops.object.mode_set(mode="EDIT")
bone = armature_data.edit_bones.new("Root")
bone.head = (0.0, 0.0, 0.0)
bone.tail = (0.0, 1.0, 0.0)
bpy.ops.object.mode_set(mode="OBJECT")
bpy.ops.cachefile.open(filepath=cache_path)
bpy.ops.wm.alembic_import(filepath=cache_path, as_background_job=False)
cache_file = bpy.data.cache_files.get(os.path.basename(cache_path))
object_path = next(path.path for path in cache_file.object_paths if path.path.startswith("/Animated/"))
cache_file.scale = 1.0
constraint = armature.pose.bones["Root"].constraints.new("TRANSFORM_CACHE")
constraint.cache_file = cache_file
constraint.object_path = object_path
assert constraint.is_valid, "Blender pose-bone Transform Cache constraint is invalid"
expected = {}
for frame in (1,2,3):
    scene.frame_set(frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    evaluated = armature.evaluated_get(depsgraph)
    matrix = evaluated.matrix_world @ evaluated.pose.bones["Root"].matrix
    expected[str(frame)] = [float(matrix[row][column])
                            for column in range(4) for row in range(4)]
assert max(abs(left - right) for left, right in zip(expected["1"], expected["3"])) > 1.0, \
    "Transform Cache expected matrices did not change across frames"
with open(output_path, "w", encoding="utf-8") as output:
    json.dump(expected, output)
"#;
    fs::write(&script_path, script)?;
    let blender_output = Command::new(&blender)
        .args(["--background", "--factory-startup", "--python"])
        .arg(&script_path)
        .args([
            "--",
            abc_path.to_str().ok_or("Alembic path is not UTF-8")?,
            output_path.to_str().ok_or("matrix path is not UTF-8")?,
        ])
        .output()?;
    assert!(
        blender_output.status.success(),
        "Blender pose cache fixture failed: {}",
        String::from_utf8_lossy(&blender_output.stderr)
    );
    assert!(
        abc_path.is_file(),
        "Blender did not create the pose-cache Alembic fixture: {}{}",
        String::from_utf8_lossy(&blender_output.stdout),
        String::from_utf8_lossy(&blender_output.stderr)
    );
    assert!(
        output_path.is_file(),
        "Blender did not create the pose-cache matrix fixture: {}{}",
        String::from_utf8_lossy(&blender_output.stdout),
        String::from_utf8_lossy(&blender_output.stderr)
    );
    let cache_path = fs::canonicalize(&abc_path)?;
    let (_scene_directory, project) = project(&json!([
        {"op":"node.create","id":"arm","kind":"armature"},
        {"op":"bone.create","target":{"id":"arm"},"id":"root","name":"Root","head":[0.0,0.0,0.0],"tail":[0.0,1.0,0.0]},
        {"op":"resource.pack","id":"cache","uri":cache_path,"kind":"alembic"},
        {"op":"constraint.create","target":{"id":"arm"},"id":"pose_cache","type":"transform_cache","owner_bone":"root","params":{"resource":"cache","object_path":"/Animated/AnimatedMesh","frame_offset":0.0,"scale":1.0,"override_frame":null}}
    ]))?;
    let document: SceneDoc = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let blender_matrices: Value = serde_json::from_slice(&fs::read(&output_path)?)?;
    for frame in [1.0, 2.0, 3.0] {
        let snapshot = Snapshot::evaluate_with_cache(
            &document,
            &EvaluationContext {
                frame: Some(frame),
                ..EvaluationContext::default()
            },
            Some(&project),
        )?;
        let armature_id = Id::new("arm".to_owned())?;
        let bone_id = Id::new("root".to_owned())?;
        let actual = snapshot
            .bone_matrices
            .get(&armature_id)
            .and_then(|bones| bones.get(&bone_id))
            .copied()
            .ok_or_else(|| io::Error::other("evaluated pose-bone matrix is missing"))?;
        let expected = blender_matrices[frame.to_string()]
            .as_array()
            .ok_or_else(|| io::Error::other("Blender pose-bone matrix is missing"))?;
        let mut maximum_error = 0.0_f64;
        for (component, (actual_component, expected_component)) in
            actual.iter().zip(expected).enumerate()
        {
            let expected_component = expected_component
                .as_f64()
                .ok_or_else(|| io::Error::other("Blender matrix component is invalid"))?;
            let error = (actual_component - expected_component).abs();
            maximum_error = maximum_error.max(error);
            assert!(
                error < 1.0e-5,
                "pose-bone Transform Cache matrix differs at frame {frame}, component {component}: Potter={actual:?}, Blender={expected:?}"
            );
        }
        eprintln!(
            "Pose-bone Transform Cache frame {frame} maximum component error: {maximum_error:.6e}"
        );
    }
    Ok(())
}
#[test]
fn transform_cache_usd_reports_its_unsupported_feature_id() -> TestResult {
    let source_directory = tempdir()?;
    let archive = source_directory.path().join("cache.usd");
    fs::write(&archive, b"USD cache fixture")?;
    let (_directory, project) = project(&json!([
        {"op":"node.create","id":"owner","kind":"empty"},
        {"op":"resource.pack","id":"cache","uri":archive,"kind":"usd"},
        {"op":"constraint.create","target":{"id":"owner"},"id":"cache_constraint","type":"transform_cache","params":{"resource":"cache","object_path":"/Root","frame_offset":0.0,"scale":1.0,"override_frame":null}}
    ]))?;
    let document: SceneDoc = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let error = Snapshot::evaluate(&document, &EvaluationContext::default())
        .err()
        .ok_or_else(|| io::Error::other("USD Transform Cache evaluation unexpectedly succeeded"))?;
    assert_eq!(
        error.code,
        potter_core::error::ErrorCode::UnsupportedFeature
    );
    assert_eq!(
        error.details.get("feature_id").and_then(Value::as_str),
        Some("constraint.transform_cache.usd")
    );
    Ok(())
}
