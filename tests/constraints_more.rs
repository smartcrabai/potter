use std::{error::Error, fs, io, path::Path, process::Command};

use glam::{DMat4, DQuat, DVec3};
use potter_core::{
    eval::{EvaluationContext, EvaluationPurpose, EvaluationSession, Snapshot},
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

struct ActiveGuardTarget {
    target_node: String,
    modifier_id: String,
    constraint: Option<(String, String)>,
}

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
fn inspected_bone_matrix(project: &Path, rig: &str, bone: &str, frame: &str) -> TestResult<DMat4> {
    let output = pot()
        .arg("inspect")
        .arg(project)
        .arg("--id")
        .arg(rig)
        .arg("--frame")
        .arg(frame)
        .arg("--json")
        .output()?;
    assert!(
        output.status.success(),
        "pot inspect failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let inspected: Value = serde_json::from_slice(&output.stdout)?;
    let item = inspected["result"]["items"]
        .as_array()
        .and_then(|items| items.iter().find(|item| item["id"] == rig))
        .ok_or_else(|| io::Error::other("inspect omitted the armature node"))?;
    blender_matrix(&item["evaluated_bones"][bone])
}

fn assert_mesh_positions_match(mesh: &potter_core::geom::Mesh, expected: &Value) -> TestResult {
    let expected: Vec<[f64; 3]> = serde_json::from_value(expected.clone())?;
    assert_eq!(mesh.vertices.len(), expected.len());
    let mut actual = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co.to_array())
        .collect::<Vec<_>>();
    for expected in expected {
        let Some(index) = actual.iter().position(|point| {
            point
                .iter()
                .zip(expected)
                .all(|(actual, expected)| (actual - expected).abs() < 1.0e-5)
        }) else {
            return Err(io::Error::other(format!(
                "cache mesh vertex {expected:?} was not present in {actual:?}"
            ))
            .into());
        };
        actual.swap_remove(index);
    }
    assert!(
        actual.is_empty(),
        "unexpected cache mesh vertices: {actual:?}"
    );
    Ok(())
}

fn assert_json_positions_match(actual: &Value, expected: &Value) -> TestResult {
    let mut actual: Vec<[f64; 3]> = serde_json::from_value(actual.clone())?;
    let expected: Vec<[f64; 3]> = serde_json::from_value(expected.clone())?;
    assert_eq!(
        actual.len(),
        expected.len(),
        "evaluated vertex counts differ"
    );
    for expected in expected {
        let Some(index) = actual.iter().position(|point| {
            point
                .iter()
                .zip(expected)
                .all(|(actual, expected)| (actual - expected).abs() < 1.0e-5)
        }) else {
            return Err(io::Error::other(format!(
                "Blender oracle vertex {expected:?} was not present in {actual:?}"
            ))
            .into());
        };
        actual.swap_remove(index);
    }
    assert!(
        actual.is_empty(),
        "unexpected evaluated vertices: {actual:?}"
    );
    Ok(())
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
fn action_constraint_legacy_pose_mode_keeps_euler_curves() -> TestResult {
    let directory = tempdir()?;
    let project = directory.path().join("legacy-pose");
    let initialized = pot().arg("init").arg(&project).arg("--json").output()?;
    assert!(
        initialized.status.success(),
        "pot init failed: {}",
        String::from_utf8_lossy(&initialized.stderr)
    );
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/legacy_pose_rotation_mode/scene.json");
    fs::write(project.join("scene.json"), fs::read(fixture)?)?;

    let before = inspected_bone_matrix(&project, "legacy_rig", "legacy_bone", "3")?;
    apply(
        &project,
        &json!([{
            "op":"constraint.create",
            "target":{"id":"legacy_rig"},
            "id":"legacy_euler_constraint",
            "type":"action",
            "owner_bone":"legacy_bone",
            "params":{
                "action":"legacy_euler_action",
                "transform_channel":"LOCATION_X",
                "target_space":"WORLD",
                "min":0.0,
                "max":1.0,
                "frame_start":1,
                "frame_end":3,
                "mix_mode":"REPLACE",
                "use_eval_time":true,
                "eval_time":1.0,
                "use_bone_object_action":false
            }
        }]),
    )?;
    let after = inspected_bone_matrix(&project, "legacy_rig", "legacy_bone", "3")?;
    assert!(
        max_matrix_error(after, before) < 1.0e-5,
        "Action constraint dropped the legacy pose bone's Euler curves: before={before:?}, after={after:?}"
    );
    Ok(())
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

fn instance_constraint_driver_operations() -> Value {
    json!([
        {"op":"node.create","id":"geometry_target","kind":"box","params":{}},
        {"op":"node.create","id":"owner","kind":"empty"},
        {"op":"node.create","id":"child","kind":"empty","parent":"owner"},
        {"op":"node.create","id":"reference","kind":"empty"},
        {"op":"node.create","id":"probe","kind":"box","params":{}},
        {"op":"graph.create","id":"instances","kind":"geometry"},
        {"op":"graph.node_add","graph":"instances","id":"points","type":"GeometryNodeMeshCube"},
        {"op":"graph.node_add","graph":"instances","id":"instance","type":"GeometryNodeInstanceOnPoints"},
        {"op":"graph.node_add","graph":"instances","id":"output","type":"NodeGroupOutput"},
        {"op":"graph.link","graph":"instances","from_node":"points","from_socket":"Mesh","to_node":"instance","to_socket":"Points"},
        {"op":"graph.link","graph":"instances","from_node":"points","from_socket":"Mesh","to_node":"instance","to_socket":"Instance"},
        {"op":"graph.link","graph":"instances","from_node":"instance","from_socket":"Instances","to_node":"output","to_socket":"Geometry"},
        {"op":"modifier.create","target":{"id":"geometry_target"},"id":"instances_modifier","type":"nodes","params":{"node_group":"instances"}},
        {"op":"constraint.create","target":{"id":"owner"},"id":"geometry","type":"geometry_attribute","constraint_target":"geometry_target","params":{"target":"geometry_target","attribute_name":"instance_transform","data_type":"FLOAT4X4","domain":"INSTANCE","sample_index":0,"mix_mode":"REPLACE","apply_target_transform":false,"mix_loc":true,"mix_rot":true,"mix_scl":true}}
    ])
}

#[test]
fn snapshot_refuses_instance_geometry_with_evaluated_transform_drivers() -> TestResult {
    let base_operations = instance_constraint_driver_operations();
    for (driver_id, variable) in [
        (
            "world",
            json!({"name":"x","type":"transforms","target":"child","path":"transform.translation","index":0,"transform_space":"world"}),
        ),
        (
            "difference",
            json!({"name":"x","type":"loc_diff","target":"child","target_2":"reference","transform_space":"world"}),
        ),
    ] {
        let mut operations = base_operations
            .as_array()
            .ok_or_else(|| io::Error::other("base driver operations must be an array"))?
            .clone();
        operations.push(json!({
            "op":"driver.create",
            "target":{"id":"probe"},
            "id":driver_id,
            "path":"transform.translation",
            "index":0,
            "type":"scripted_expression",
            "variables":[variable],
            "expression":"x",
        }));
        let (_directory, project) = project(&Value::Array(operations))?;
        let document: SceneDoc = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
        let Err(error) = Snapshot::evaluate(&document, &EvaluationContext::default()) else {
            return Err(io::Error::other("evaluated-transform driver was accepted").into());
        };
        assert_eq!(
            error.code,
            potter_core::error::ErrorCode::UnsupportedFeature
        );
        assert_eq!(
            error.details.get("feature_id").and_then(Value::as_str),
            Some("driver.geometry_attribute.instance_dependency")
        );
        assert_eq!(
            error.details.get("driver_owner").and_then(Value::as_str),
            Some("probe")
        );
    }
    Ok(())
}

#[test]
fn instance_geometry_allows_local_transform_and_plain_property_drivers() -> TestResult {
    let mut operations = instance_constraint_driver_operations()
        .as_array()
        .ok_or_else(|| io::Error::other("base driver operations must be an array"))?
        .clone();
    operations.extend([
        json!({"op":"node.create","id":"property_probe","kind":"empty"}),
        json!({"op":"driver.create","target":{"id":"probe"},"id":"local","path":"transform.translation","index":0,"type":"scripted_expression","variables":[{"name":"x","type":"transforms","target":"child","path":"transform.translation","index":0,"transform_space":"local"}],"expression":"x"}),
        json!({"op":"driver.create","target":{"id":"property_probe"},"id":"property","path":"transform.translation","index":0,"type":"scripted_expression","variables":[{"name":"weight","type":"single_prop","target":"reference","path":"properties.weight"}],"expression":"weight"}),
    ]);
    let (_directory, project) = project(&Value::Array(operations))?;
    let scene_file = project.join("scene.json");
    let mut document: SceneDoc = serde_json::from_slice(&fs::read(&scene_file)?)?;
    let reference = Id::new("reference".to_owned())?;
    document
        .nodes
        .get_mut(&reference)
        .ok_or_else(|| io::Error::other("driver property source is missing"))?
        .properties
        .insert("weight".to_owned(), json!(4.5));
    fs::write(&scene_file, serde_json::to_vec(&document)?)?;

    let snapshot = Snapshot::evaluate(&document, &EvaluationContext::default())?;
    for (node_id, expected_x) in [("probe", 0.0), ("property_probe", 4.5)] {
        let id = Id::new(node_id.to_owned())?;
        let node = snapshot
            .nodes
            .get(&id)
            .ok_or_else(|| io::Error::other("evaluated driver target is missing"))?;
        let actual_x = DMat4::from_cols_array(&node.world_matrix).w_axis.x;
        assert!((actual_x - expected_x).abs() < 1.0e-9);
    }

    let inspected = pot()
        .args(["inspect"])
        .arg(&project)
        .args(["--id", "property_probe", "--json"])
        .output()?;
    assert!(
        inspected.status.success(),
        "inspect rejected safe INSTANCE-scene drivers: {}",
        String::from_utf8_lossy(&inspected.stdout)
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
                "scale":[1.0,1.0,1.0],
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
    set_rotation(owner, rotations["owner"])
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
    reason = "the committed Action-space oracle keeps the full Blender capture coverage auditable"
)]
fn committed_action_target_space_oracle_matches_snapshot() -> TestResult {
    let fixture: Value = serde_json::from_str(include_str!(
        "fixtures/action_constraint_target_space/oracle.json"
    ))?;
    assert!(
        fixture["blender_version"]
            .as_str()
            .is_some_and(|version| version.starts_with("5.2.2"))
    );
    let cases = fixture["cases"]
        .as_array()
        .ok_or_else(|| io::Error::other("Action oracle cases are missing"))?;
    assert_eq!(cases.len(), 55);
    let default_space_case = cases
        .iter()
        .find(|case| case["id"] == "parented_object_default_owner_space")
        .ok_or_else(|| {
            io::Error::other("Action oracle is missing the parented default owner-space case")
        })?;
    assert_eq!(default_space_case["owner_kind"], "object");
    assert_eq!(
        default_space_case["owner_space_omitted"].as_bool(),
        Some(true)
    );
    assert_eq!(default_space_case["mix_mode"], "AFTER");
    assert_eq!(
        default_space_case["blender_mix_mode_omitted"].as_bool(),
        Some(true)
    );
    assert_eq!(
        default_space_case["owner_parent"],
        "owner_default_space_parent"
    );
    let operations = fixture["operations"]
        .as_array()
        .ok_or_else(|| io::Error::other("Action oracle operations are missing"))?;
    let action_operation = operations
        .iter()
        .find(|operation| operation["id"] == "action_parented_object_default_owner_space")
        .ok_or_else(|| {
            io::Error::other("Action oracle default owner-space constraint is missing")
        })?;
    assert!(
        action_operation["params"].get("owner_space").is_none(),
        "default owner-space regression must omit params.owner_space"
    );
    assert_eq!(action_operation["params"]["mix_mode"], "AFTER");

    let owner_operation = operations
        .iter()
        .find(|operation| operation["id"] == "owner_default_space")
        .ok_or_else(|| io::Error::other("Action oracle parented owner is missing"))?;
    assert_eq!(
        owner_operation["parent"],
        default_space_case["owner_parent"]
    );

    let (_directory, project) = project(&fixture["operations"])?;
    let document: SceneDoc = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let snapshot = Snapshot::evaluate(&document, &EvaluationContext::default())?;
    for case in cases {
        let id = case["id"]
            .as_str()
            .ok_or_else(|| io::Error::other("Action oracle case ID is missing"))?;
        let owner = case["owner"]
            .as_str()
            .ok_or_else(|| io::Error::other("Action oracle owner is missing"))?;
        let actual = if case["owner_kind"] == "pose" {
            let owner_id = Id::new(owner.to_owned())?;
            let bone_id = Id::new(
                case["owner_bone"]
                    .as_str()
                    .ok_or_else(|| io::Error::other("Action oracle owner bone is missing"))?
                    .to_owned(),
            )?;
            snapshot
                .bone_matrices
                .get(&owner_id)
                .and_then(|bones| bones.get(&bone_id))
                .map(DMat4::from_cols_array)
                .ok_or_else(|| {
                    io::Error::other(format!("Action oracle bone {owner}/{bone_id} is missing"))
                })?
        } else {
            let owner_id = Id::new(owner.to_owned())?;
            snapshot
                .nodes
                .get(&owner_id)
                .map(|node| DMat4::from_cols_array(&node.world_matrix))
                .ok_or_else(|| io::Error::other(format!("Action oracle node {owner} is missing")))?
        };
        let expected = blender_matrix(&fixture["expected"][id]["matrix"])?;
        let baseline = blender_matrix(&fixture["expected"][id]["baseline"])?;
        let error = max_matrix_error(actual, expected);
        assert!(
            error < 2.0e-4,
            "{id} Blender Action matrix differs by {error:.6e}: Potter={actual:?}, Blender={expected:?}, case={case}"
        );
        assert!(
            max_matrix_error(expected, baseline) > 1.0e-2,
            "{id} Action constraint had no material effect over its Blender baseline"
        );
    }

    for space in [
        "WORLD",
        "POSE",
        "LOCAL_WITH_PARENT",
        "LOCAL",
        "LOCAL_OWNER_ORIENT",
        "CUSTOM",
    ] {
        for channel in [
            "LOCATION_X",
            "LOCATION_Y",
            "LOCATION_Z",
            "ROTATION_X",
            "ROTATION_Y",
            "ROTATION_Z",
            "SCALE_X",
            "SCALE_Y",
            "SCALE_Z",
        ] {
            assert!(
                cases
                    .iter()
                    .any(|case| { case["target_space"] == space && case["channel"] == channel }),
                "Action oracle is missing {space}/{channel}"
            );
        }
    }
    for mode in [
        "REPLACE",
        "BEFORE_FULL",
        "BEFORE",
        "BEFORE_SPLIT",
        "AFTER_FULL",
        "AFTER",
        "AFTER_SPLIT",
    ] {
        assert!(
            cases.iter().any(|case| case["mix_mode"] == mode),
            "Action oracle is missing mix mode {mode}"
        );
    }
    for owner_kind in ["object", "pose"] {
        for target_kind in ["object", "bone"] {
            assert!(
                cases.iter().any(|case| {
                    case["owner_kind"] == owner_kind && case["target_kind"] == target_kind
                }),
                "Action oracle is missing {owner_kind} owner / {target_kind} target"
            );
        }
    }
    for owner_kind in ["object", "pose"] {
        for rotation_mode in ["quaternion", "XYZ", "AXIS_ANGLE"] {
            assert!(
                cases.iter().any(|case| {
                    case["owner_kind"] == owner_kind && case["rotation_mode"] == rotation_mode
                }),
                "Action oracle is missing {owner_kind} owner rotation mode {rotation_mode}"
            );
        }
    }
    assert!(
        cases
            .iter()
            .any(|case| case["use_eval_time"].as_bool() == Some(true))
    );
    assert!(
        cases
            .iter()
            .any(|case| case["use_bone_object_action"].as_bool() == Some(true))
    );
    assert!(
        cases.iter().any(|case| {
            case["target_space"] == "CUSTOM" && case["space_object"] == "space_ref"
        })
    );
    assert!(cases.iter().any(|case| {
        case["target_space"] == "CUSTOM"
            && case["space_object"] == "space_arm"
            && case["space_subtarget"] == "SpaceBone"
    }));
    Ok(())
}

#[test]
fn action_constraint_uses_eval_time_without_a_target() -> TestResult {
    let mut operations = json!([
        {"op":"node.create","id":"owner","kind":"empty"},
        {"op":"action.create","id":"drive","name":"Drive"},
        {"op":"action.update","target":{"id":"drive"},"set":{"fcurves":[
            {"path":"transform.translation","index":0,"keyframes":[
                {"frame":1.0,"value":0.0,"interpolation":"linear"},
                {"frame":11.0,"value":10.0,"interpolation":"linear"}
            ]}
        ]}},
        {"op":"constraint.create","target":{"id":"owner"},"id":"action_constraint","type":"action","params":{
            "action":"drive",
            "transform_channel":"LOCATION_X",
            "target_space":"WORLD",
            "min":0.0,
            "max":1.0,
            "frame_start":1,
            "frame_end":11,
            "mix_mode":"REPLACE",
            "owner_space":"WORLD",
            "use_eval_time":true,
            "eval_time":0.25,
            "use_bone_object_action":false
        }}
    ]);
    let (_directory, project) = project(&operations)?;
    let matrix = evaluated_matrix(&project, "owner")?;
    let expected = DMat4::from_translation(DVec3::new(2.5, 0.0, 0.0));
    assert!(
        max_matrix_error(matrix, expected) < 1.0e-5,
        "targetless Action eval-time matrix differs: Potter={matrix:?}, expected={expected:?}"
    );

    operations[3]["params"]["use_eval_time"] = json!(false);
    let rejected = ops::apply_batch(
        &SceneDoc::new("00000000-0000-4000-8000-000000000003".to_owned()),
        &json!({
            "schema_version":1,
            "base_revision":0,
            "operations":operations
        }),
    );
    assert!(
        rejected.is_err(),
        "Action constraint without a target was accepted when use_eval_time is false"
    );
    Ok(())
}

#[test]
fn omitted_action_owner_space_uses_world_on_a_parented_object() -> TestResult {
    let identity = [0.0, 0.0, 0.0, 1.0];
    let operations = json!([
        {"op":"node.create","id":"parent","kind":"empty","transform":{
            "translation":[2.0,-1.0,0.5],"rotation":[0.0,0.0,std::f64::consts::FRAC_1_SQRT_2,std::f64::consts::FRAC_1_SQRT_2],"scale":[1.5,1.0,0.8]}},
        {"op":"node.create","id":"default_owner","kind":"empty","parent":"parent","transform":{"rotation":identity}},
        {"op":"node.create","id":"world_owner","kind":"empty","parent":"parent","transform":{"rotation":identity}},
        {"op":"node.create","id":"local_owner","kind":"empty","parent":"parent","transform":{"rotation":identity}},
        {"op":"action.create","id":"drive","name":"Drive"},
        {"op":"action.update","target":{"id":"drive"},"set":{"fcurves":[
            {"path":"transform.translation","index":0,"keyframes":[
                {"frame":1.0,"value":0.0,"interpolation":"linear"},
                {"frame":11.0,"value":10.0,"interpolation":"linear"}
            ]}
        ]}},
        {"op":"constraint.create","target":{"id":"default_owner"},"id":"action","type":"action","params":{
            "action":"drive","transform_channel":"LOCATION_X","target_space":"WORLD",
            "min":0.0,"max":1.0,"frame_start":1,"frame_end":11,"mix_mode":"REPLACE",
            "use_eval_time":true,"eval_time":0.25,"use_bone_object_action":false}},
        {"op":"constraint.create","target":{"id":"world_owner"},"id":"action","type":"action","params":{
            "action":"drive","transform_channel":"LOCATION_X","target_space":"WORLD",
            "min":0.0,"max":1.0,"frame_start":1,"frame_end":11,"mix_mode":"REPLACE",
            "owner_space":"WORLD","use_eval_time":true,"eval_time":0.25,"use_bone_object_action":false}},
        {"op":"constraint.create","target":{"id":"local_owner"},"id":"action","type":"action","params":{
            "action":"drive","transform_channel":"LOCATION_X","target_space":"WORLD",
            "min":0.0,"max":1.0,"frame_start":1,"frame_end":11,"mix_mode":"REPLACE",
            "owner_space":"LOCAL","use_eval_time":true,"eval_time":0.25,"use_bone_object_action":false}}
    ]);
    let (_directory, project) = project(&operations)?;
    let default = evaluated_matrix(&project, "default_owner")?;
    let world = evaluated_matrix(&project, "world_owner")?;
    let local = evaluated_matrix(&project, "local_owner")?;
    assert!(
        max_matrix_error(default, world) < 1.0e-12,
        "omitted owner_space differs from WORLD: default={default:?}, world={world:?}"
    );
    assert!(
        max_matrix_error(default, local) > 1.0e-2,
        "parented fixture did not distinguish WORLD and LOCAL: default={default:?}, local={local:?}"
    );
    Ok(())
}

#[test]
fn action_object_axis_angle_angle_only_keeps_the_zero_workob_axis() -> TestResult {
    let operations = json!([
        {"op":"node.create","id":"owner","kind":"empty","transform":{"rotation_mode":"AXIS_ANGLE"}},
        {"op":"action.create","id":"drive","name":"Drive"},
        {"op":"action.update","target":{"id":"drive"},"set":{"fcurves":[
            {"path":"transform.rotation_axis_angle","index":0,"keyframes":[
                {"frame":1.0,"value":0.0,"interpolation":"linear"},
                {"frame":11.0,"value":std::f64::consts::FRAC_PI_2,"interpolation":"linear"}
            ]}
        ]}},
        {"op":"constraint.create","target":{"id":"owner"},"id":"action_constraint","type":"action","params":{
            "action":"drive","transform_channel":"LOCATION_X","target_space":"WORLD",
            "min":0.0,"max":1.0,"frame_start":1,"frame_end":11,"mix_mode":"REPLACE",
            "owner_space":"WORLD","use_eval_time":true,"eval_time":1.0,"use_bone_object_action":false}}
    ]);
    let (_directory, project) = project(&operations)?;
    let matrix = evaluated_matrix(&project, "owner")?;
    assert!(
        max_matrix_error(matrix, DMat4::IDENTITY) < 1.0e-12,
        "Action object workob angle-only curve must retain its zero axis, got {matrix:?}"
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

fn import_blend_project(blender: &Path, project: &Path, source: &Path) -> TestResult<Value> {
    let init = pot().arg("init").arg(project).arg("--json").output()?;
    assert!(
        init.status.success(),
        "pot init failed: {}",
        String::from_utf8_lossy(&init.stdout)
    );
    let imported = pot()
        .args(["import"])
        .arg(project)
        .args(["--file"])
        .arg(source)
        .args([
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "0",
            "--blender",
        ])
        .arg(blender)
        .arg("--json")
        .output()?;
    assert!(
        imported.status.success(),
        "pot import failed: {}",
        String::from_utf8_lossy(&imported.stdout)
    );
    Ok(serde_json::from_slice(&imported.stdout)?)
}

fn imported_node_id(project: &Path, name: &str) -> TestResult<String> {
    let scene: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    scene["nodes"]
        .as_object()
        .and_then(|nodes| {
            nodes
                .iter()
                .find(|(_, node)| node["name"].as_str() == Some(name))
                .map(|(id, _)| id.clone())
        })
        .ok_or_else(|| io::Error::other(format!("imported node {name} is missing")).into())
}

fn blender_join_link_oracle(blender: &Path, blend: &Path) -> TestResult<Value> {
    let script = r#"
import bpy
import json
import sys

blend = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.open_mainfile(filepath=blend)
group = bpy.data.node_groups["JoinedInstanceGeometry"]
join = next(node for node in group.nodes if node.bl_idname == "GeometryNodeJoinGeometry")
multi = sorted(
    [
        {
            "source_scale": [float(component) for component in link.from_node.inputs["Scale"].default_value],
            "sort_id": int(link.multi_input_sort_id),
        }
        for link in group.links
        if link.to_node == join
    ],
    key=lambda item: item["sort_id"],
    reverse=True,
)
ordinary = [
    [float(component) for component in link.to_node.inputs["Translation"].default_value]
    for link in group.links
    if link.to_node.bl_idname == "GeometryNodeTransform"
]
print("POTTER_JOIN_LINK_ORACLE=" + json.dumps({
    "sorted_multi_input": multi,
    "ordinary_transform_order": ordinary,
}, sort_keys=True))
"#;
    let output = Command::new(blender)
        .args([
            "--background",
            "--factory-startup",
            "--python-expr",
            script,
            "--",
        ])
        .arg(blend)
        .output()?;
    assert!(
        output.status.success(),
        "Blender Join link oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let marker = stdout
        .lines()
        .find_map(|line| line.strip_prefix("POTTER_JOIN_LINK_ORACLE="))
        .ok_or_else(|| io::Error::other("Blender Join link oracle output is missing"))?;
    Ok(serde_json::from_str(marker)?)
}

fn assert_hair_curves_inspect(project: &Path, oracle: &Value, cases: &[Value]) -> TestResult<f64> {
    let inspected = pot()
        .args(["inspect"])
        .arg(project)
        .arg("--json")
        .output()?;
    assert!(
        inspected.status.success(),
        "pot inspect failed: {}",
        String::from_utf8_lossy(&inspected.stdout)
    );
    let inspected: Value = serde_json::from_slice(&inspected.stdout)?;
    let items = inspected["result"]["items"]
        .as_array()
        .ok_or_else(|| io::Error::other("pot inspect returned no node items"))?;
    let target = items
        .iter()
        .find(|item| item["name"] == "GeometryAttributeTarget")
        .ok_or_else(|| io::Error::other("Curves target is missing from pot inspect"))?;
    let target_error = max_matrix_error(
        blender_matrix(&target["transform"]["world"]["matrix"])?,
        blender_matrix(&oracle["target_matrix"])?,
    );
    assert!(
        target_error < 2.0e-4,
        "Curves target matrix differs by {target_error:e}"
    );
    let mut maximum_error = 0.0_f64;
    for case in cases {
        let id = case["id"]
            .as_str()
            .ok_or_else(|| io::Error::other("Geometry Attribute Curves owner ID is missing"))?;
        let item = items
            .iter()
            .find(|item| item["name"].as_str() == Some(id))
            .ok_or_else(|| io::Error::other(format!("pot inspect omitted owner {id}")))?;
        let expected = blender_matrix(&oracle["matrices"][id])?;
        let actual = blender_matrix(&item["transform"]["world"]["matrix"])?;
        let error = max_matrix_error(actual, expected);
        maximum_error = maximum_error.max(error);
        assert!(
            error < 2.0e-4,
            "{id} Blender hair Curves Geometry Attribute matrix differs by {error:e}: Potter={actual:?}, Blender={expected:?}, case={case}"
        );
    }
    Ok(maximum_error)
}

#[test]
fn blender_geometry_attribute_hair_curve_domains_match_depsgraph() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Geometry Attribute hair Curves parity test: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/geometry_attribute_curve_domain.py");
    let output = Command::new(&blender)
        .args(["--background", "--factory-startup", "--python"])
        .arg(fixture)
        .arg("--")
        .arg(root)
        .output()?;
    assert!(
        output.status.success(),
        "Blender Geometry Attribute Curves oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let oracle: Value = serde_json::from_slice(&fs::read(root.join("expected.json"))?)?;
    let cases = oracle["cases"]
        .as_array()
        .ok_or_else(|| io::Error::other("Geometry Attribute Curves oracle cases are missing"))?;
    assert_eq!(cases.len(), 33);

    let project = root.join("project");
    let imported = import_blend_project(
        &blender,
        &project,
        &root.join("geometry_attribute_curves.blend"),
    )?;
    assert_eq!(imported["result"]["losses"], json!([]));
    let maximum_error = assert_hair_curves_inspect(&project, &oracle, cases)?;

    let roundtrip_blend = root.join("roundtrip.blend");
    let exported = pot()
        .args(["export"])
        .arg(&project)
        .args(["--format", "blend", "--out"])
        .arg(&roundtrip_blend)
        .args(["--blender"])
        .arg(&blender)
        .arg("--json")
        .output()?;
    assert!(
        exported.status.success(),
        "pot export failed: {}",
        String::from_utf8_lossy(&exported.stdout)
    );
    let exported: Value = serde_json::from_slice(&exported.stdout)?;
    assert_eq!(exported["result"]["losses"], json!([]));

    let roundtrip_project = root.join("roundtrip_project");
    let roundtrip_import = import_blend_project(&blender, &roundtrip_project, &roundtrip_blend)?;
    assert_eq!(roundtrip_import["result"]["losses"], json!([]));
    let roundtrip_error = assert_hair_curves_inspect(&roundtrip_project, &oracle, cases)?;
    let roundtrip_scene: Value =
        serde_json::from_slice(&fs::read(roundtrip_project.join("scene.json"))?)?;
    let nodes = roundtrip_scene["nodes"]
        .as_object()
        .ok_or_else(|| io::Error::other("round-trip Curves scene has no nodes"))?;
    let target_node = nodes
        .values()
        .find(|node| node["name"] == "GeometryAttributeTarget")
        .ok_or_else(|| io::Error::other("round-trip Curves target is missing"))?;
    let data_id = target_node["data"]
        .as_str()
        .ok_or_else(|| io::Error::other("round-trip Curves target has no data ID"))?;
    let hair = &roundtrip_scene["data_blocks"][data_id]["hair_curves"];
    assert_eq!(
        hair["attributes"]["radius"]["values"],
        oracle["point_radii"]
    );
    assert!(hair["attributes"].get("radius.001").is_none());
    assert_eq!(
        hair["surface_uv_map"],
        oracle["surface_binding"]["surface_uv_map"]
    );
    assert_eq!(
        hair["use_mirror_x"],
        oracle["surface_binding"]["use_mirror_x"]
    );
    assert!(!hair["use_mirror_y"].as_bool().unwrap_or(false));
    assert_eq!(
        hair["use_mirror_z"],
        oracle["surface_binding"]["use_mirror_z"]
    );
    assert_eq!(
        hair["selection_domain"],
        oracle["surface_binding"]["selection_domain"]
    );
    let surface_id = hair["surface"]
        .as_str()
        .ok_or_else(|| io::Error::other("round-trip Curves surface ID is missing"))?;
    assert_eq!(
        nodes[surface_id]["name"],
        oracle["surface_binding"]["object"]
    );
    assert_hair_curves_surface_unlink(&blender, &project, root)?;
    eprintln!(
        "Blender 5.2.2 hair Curves Geometry Attribute CLI matrices: import max error {maximum_error:.6e}, .blend export/import max error {roundtrip_error:.6e} across 33 POINT/CURVE, type, index, mix-mode, and adapted-domain cases"
    );
    Ok(())
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the INSTANCE parity matrix keeps topology, conversion, and guard cases together"
)]
fn blender_geometry_attribute_instance_domain_matches_depsgraph() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Geometry Attribute INSTANCE parity test: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/geometry_attribute_instances.py");
    let output = Command::new(&blender)
        .args(["--background", "--factory-startup", "--python"])
        .arg(fixture)
        .arg("--")
        .arg(root)
        .output()?;
    assert!(
        output.status.success(),
        "Blender Geometry Attribute INSTANCE oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let oracle: Value = serde_json::from_slice(&fs::read(root.join("expected.json"))?)?;
    let cases = oracle["cases"]
        .as_array()
        .ok_or_else(|| io::Error::other("Geometry Attribute INSTANCE oracle cases are missing"))?;
    assert!(
        cases.len() >= 90,
        "Geometry Attribute INSTANCE oracle omitted its extended cases: {}",
        cases.len()
    );
    for index in [0, 2, 999] {
        assert!(
            cases
                .iter()
                .any(|case| case["sample_index"] == json!(index)),
            "Geometry Attribute INSTANCE fixture omitted sample index {index}"
        );
    }
    for mix_mode in [
        "REPLACE",
        "BEFORE_FULL",
        "BEFORE_SPLIT",
        "AFTER_FULL",
        "AFTER_SPLIT",
    ] {
        assert!(
            cases.iter().any(|case| case["mix_mode"] == mix_mode),
            "Geometry Attribute INSTANCE fixture omitted mix mode {mix_mode}"
        );
    }
    assert!(cases.iter().all(|case| case["domain"] == "INSTANCE"));
    for (data_type, attribute) in [
        ("FLOAT4X4", "instance_transform"),
        ("QUATERNION", "instance_transform"),
        ("QUATERNION", "geometry_attribute_instance_quaternion"),
        ("QUATERNION", "geometry_attribute_instance_vector"),
        ("VECTOR", "geometry_attribute_instance_vector"),
        ("VECTOR", "geometry_attribute_instance_float"),
        ("VECTOR", "geometry_attribute_instance_quaternion"),
        ("FLOAT4X4", "geometry_attribute_instance_quaternion"),
        ("QUATERNION", "geometry_attribute_instance_float"),
        ("QUATERNION", "geometry_attribute_instance_float2"),
        ("VECTOR", "geometry_attribute_instance_int"),
        ("VECTOR", "geometry_attribute_instance_boolean"),
        ("VECTOR", "geometry_attribute_instance_float2"),
        ("VECTOR", "geometry_attribute_instance_float_color"),
        ("VECTOR", "geometry_attribute_instance_byte_color"),
    ] {
        for mix_mode in [
            "REPLACE",
            "BEFORE_FULL",
            "BEFORE_SPLIT",
            "AFTER_FULL",
            "AFTER_SPLIT",
        ] {
            assert!(
                cases.iter().any(|case| {
                    case["data_type"] == data_type
                        && case["attribute"] == attribute
                        && case["mix_mode"] == mix_mode
                }),
                "Geometry Attribute INSTANCE fixture omitted {data_type}/{attribute}/{mix_mode}"
            );
        }
        for index in [0, 2, 999] {
            assert!(
                cases.iter().any(|case| {
                    case["data_type"] == data_type
                        && case["attribute"] == attribute
                        && case["sample_index"] == json!(index)
                }),
                "Geometry Attribute INSTANCE fixture omitted {data_type}/{attribute} index {index}"
            );
        }
    }
    let partial = cases
        .iter()
        .filter(|case| {
            case["id"]
                .as_str()
                .is_some_and(|id| id.starts_with("owner_partial_"))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        partial.len(),
        3,
        "partial mix flags must cover each component"
    );
    for flag in ["mix_loc", "mix_rot", "mix_scl"] {
        assert!(
            partial.iter().any(|case| {
                case[flag] == false
                    && ["mix_loc", "mix_rot", "mix_scl"]
                        .into_iter()
                        .filter(|other| *other != flag)
                        .all(|other| case[other] == true)
            }),
            "partial mix fixture omitted independent {flag}=false"
        );
    }
    for finding in ["F1", "F2", "F3", "F4", "F5", "F6"] {
        assert!(
            cases.iter().any(|case| case["review_finding"] == finding),
            "INSTANCE parity no longer exercises review finding {finding}"
        );
    }
    for index in [0, 2, 999] {
        assert!(
            cases.iter().any(|case| {
                case["review_finding"] == "F1"
                    && case["target"].as_str() == Some("NestedInstancesTarget")
                    && case["sample_index"] == json!(index)
            }),
            "nested-source INSTANCE coverage omitted index {index}"
        );
    }
    for id in [
        "owner_join_3",
        "owner_realized_then_instance",
        "owner_realized",
        "owner_stacked_2",
        "owner_set_position_2",
        "owner_distribute_points_2",
        "owner_distribute_control_2",
        "owner_vertex_group_2",
        "owner_second_geometry_input",
        "owner_nested_source_2_replace",
    ] {
        assert!(
            cases.iter().any(|case| case["id"].as_str() == Some(id)),
            "Geometry Attribute INSTANCE fixture omitted {id}"
        );
    }

    let project = root.join("project");
    let imported = import_blend_project(
        &blender,
        &project,
        &root.join("geometry_attribute_instances.blend"),
    )?;
    assert_eq!(imported["result"]["losses"], json!([]));
    let inspected = pot()
        .args(["inspect"])
        .arg(&project)
        .arg("--json")
        .output()?;
    assert!(
        inspected.status.success(),
        "pot inspect refused supported INSTANCE cases (including the old unconditional refusal): {}",
        String::from_utf8_lossy(&inspected.stdout)
    );
    let inspected: Value = serde_json::from_slice(&inspected.stdout)?;
    let items = inspected["result"]["items"]
        .as_array()
        .ok_or_else(|| io::Error::other("pot inspect returned no node items"))?;
    let mut maximum_error = 0.0_f64;
    for case in cases {
        let id = case["id"]
            .as_str()
            .ok_or_else(|| io::Error::other("Geometry Attribute INSTANCE owner ID is missing"))?;
        let item = items
            .iter()
            .find(|item| item["name"].as_str() == Some(id))
            .ok_or_else(|| io::Error::other(format!("pot inspect omitted owner {id}")))?;
        let expected = blender_matrix(&oracle["matrices"][id])?;
        let actual = blender_matrix(&item["transform"]["world"]["matrix"])?;
        let error = max_matrix_error(actual, expected);
        maximum_error = maximum_error.max(error);
        assert!(
            error < 2.0e-4,
            "{id} Blender Geometry Attribute INSTANCE matrix differs by {error:e}: Potter={actual:?}, Blender={expected:?}, case={case}"
        );
        let baseline = blender_matrix(&case["baseline"])?;
        let effect = max_matrix_error(expected, baseline);
        assert!(
            effect > 1.0e-2,
            "{id} Blender INSTANCE fixture did not materially change its owner transform"
        );
        if case["data_type"] == "FLOAT4X4" {
            let expected_columns = expected.to_cols_array();
            let baseline_columns = baseline.to_cols_array();
            let linear_effect = [0, 1, 2, 4, 5, 6, 8, 9, 10]
                .into_iter()
                .map(|index| (expected_columns[index] - baseline_columns[index]).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                linear_effect > 1.0e-2,
                "{id} FLOAT4X4 result did not change its rotation/scale basis"
            );
        }
    }
    for index in [0, 2, 999] {
        let distributed = format!("owner_distribute_points_{index}");
        let control = format!("owner_distribute_control_{index}");
        let distributed_matrix = blender_matrix(&oracle["matrices"][distributed.as_str()])?;
        let control_matrix = blender_matrix(&oracle["matrices"][control.as_str()])?;
        let error = max_matrix_error(distributed_matrix, control_matrix);
        assert!(
            error < 2.0e-4,
            "Distribute Points on Faces changed top-level instance {index} matrix by {error:e}"
        );
    }

    // This specifically proves that only interface geometry input 0 is populated.
    let second_input_case = cases
        .iter()
        .find(|case| case["review_finding"] == "F6")
        .ok_or_else(|| io::Error::other("F6 geometry-input case is missing"))?;
    let target_name = second_input_case["target"]
        .as_str()
        .ok_or_else(|| io::Error::other("F6 target name is missing"))?;
    let expected_geometry = &oracle["target_geometry"][target_name];
    let target_item = items
        .iter()
        .find(|item| item["name"].as_str() == Some(target_name))
        .ok_or_else(|| io::Error::other("pot inspect omitted the F6 geometry target"))?;
    for key in ["vertex_count", "edge_count", "face_count"] {
        assert_eq!(
            target_item["evaluated_geometry"][key], expected_geometry[key],
            "F6 second Geometry interface input was not empty for {key}"
        );
    }
    let roundtrip_case = cases
        .iter()
        .find(|case| case["id"] == "owner_join_3")
        .ok_or_else(|| io::Error::other("swapped Join INSTANCE case is missing"))?;
    let roundtrip_id = roundtrip_case["id"]
        .as_str()
        .ok_or_else(|| io::Error::other("round-trip owner ID is missing"))?;
    let roundtrip_blend = root.join("geometry_attribute_instances_roundtrip.blend");
    let exported = pot()
        .args(["export"])
        .arg(&project)
        .args(["--format", "blend", "--out"])
        .arg(&roundtrip_blend)
        .args(["--blender"])
        .arg(&blender)
        .arg("--json")
        .output()?;
    assert!(
        exported.status.success(),
        "pot export of INSTANCE Geometry Nodes failed: {}",
        String::from_utf8_lossy(&exported.stdout)
    );
    let exported: Value = serde_json::from_slice(&exported.stdout)?;
    assert_eq!(exported["result"]["losses"], json!([]));
    let roundtrip_project = root.join("geometry_attribute_instances_roundtrip_project");
    let roundtrip_import = import_blend_project(&blender, &roundtrip_project, &roundtrip_blend)?;
    assert_eq!(roundtrip_import["result"]["losses"], json!([]));
    let roundtrip_inspected = pot()
        .args(["inspect"])
        .arg(&roundtrip_project)
        .args(["--json"])
        .output()?;
    assert!(
        roundtrip_inspected.status.success(),
        "pot inspect after Geometry Nodes .blend round-trip failed: {}",
        String::from_utf8_lossy(&roundtrip_inspected.stdout)
    );
    let roundtrip_inspected: Value = serde_json::from_slice(&roundtrip_inspected.stdout)?;
    let roundtrip_items = roundtrip_inspected["result"]["items"]
        .as_array()
        .ok_or_else(|| io::Error::other("round-trip inspect returned no node items"))?;
    for case in cases {
        let id = case["id"]
            .as_str()
            .ok_or_else(|| io::Error::other("Geometry Attribute INSTANCE owner ID is missing"))?;
        assert!(
            roundtrip_items
                .iter()
                .any(|item| item["name"].as_str() == Some(id)),
            "round-trip inspect omitted owner {id}"
        );
    }
    let roundtrip_item = roundtrip_items
        .iter()
        .find(|item| item["name"].as_str() == Some(roundtrip_id))
        .ok_or_else(|| io::Error::other("round-trip inspect omitted its Join owner"))?;
    let roundtrip_expected = blender_matrix(&oracle["matrices"][roundtrip_id])?;
    let roundtrip_actual = blender_matrix(&roundtrip_item["transform"]["world"]["matrix"])?;
    let roundtrip_error = max_matrix_error(roundtrip_actual, roundtrip_expected);
    assert!(
        roundtrip_error < 2.0e-4,
        "Geometry Nodes Rotation input changed across .blend export/reimport by {roundtrip_error:e}: Potter={roundtrip_actual:?}, Blender={roundtrip_expected:?}"
    );

    let stable_blend = root.join("geometry_attribute_instances_roundtrip_stable.blend");
    let stable_export = pot()
        .args(["export"])
        .arg(&roundtrip_project)
        .args(["--format", "blend", "--out"])
        .arg(&stable_blend)
        .args(["--blender"])
        .arg(&blender)
        .arg("--json")
        .output()?;
    assert!(
        stable_export.status.success(),
        "pot re-export of Geometry Nodes project failed: {}",
        String::from_utf8_lossy(&stable_export.stdout)
    );
    let stable_export: Value = serde_json::from_slice(&stable_export.stdout)?;
    assert_eq!(stable_export["result"]["losses"], json!([]));
    let stable_join_links = blender_join_link_oracle(&blender, &stable_blend)?;
    assert_eq!(
        stable_join_links, oracle["join_link_oracle"],
        "swapped multi-input IDs or interleaved ordinary links changed across round-trips"
    );
    let stable_project = root.join("geometry_attribute_instances_stable_project");
    let stable_import = import_blend_project(&blender, &stable_project, &stable_blend)?;
    assert_eq!(stable_import["result"]["losses"], json!([]));
    let stable_inspected = pot()
        .args(["inspect"])
        .arg(&stable_project)
        .args(["--json"])
        .output()?;
    assert!(
        stable_inspected.status.success(),
        "pot inspect after repeated Geometry Nodes round-trip failed: {}",
        String::from_utf8_lossy(&stable_inspected.stdout)
    );
    let stable_inspected: Value = serde_json::from_slice(&stable_inspected.stdout)?;
    let stable_items = stable_inspected["result"]["items"]
        .as_array()
        .ok_or_else(|| io::Error::other("stable inspect returned no node items"))?;
    for case in cases {
        let id = case["id"]
            .as_str()
            .ok_or_else(|| io::Error::other("Geometry Attribute INSTANCE owner ID is missing"))?;
        assert!(
            stable_items
                .iter()
                .any(|item| item["name"].as_str() == Some(id)),
            "stable inspect omitted owner {id}"
        );
    }
    let stable_item = stable_items
        .iter()
        .find(|item| item["name"].as_str() == Some(roundtrip_id))
        .ok_or_else(|| io::Error::other("stable round-trip inspect omitted Join owner"))?;
    let stable_matrix = blender_matrix(&stable_item["transform"]["world"]["matrix"])?;
    let stable_error = max_matrix_error(stable_matrix, roundtrip_expected);
    assert!(
        stable_error < 2.0e-4,
        "swapped Join source order changed across repeated export/import by {stable_error:e}"
    );

    let guard_cases = oracle["guards"]
        .as_array()
        .ok_or_else(|| io::Error::other("Geometry Attribute INSTANCE guards are missing"))?;
    assert_eq!(
        guard_cases.len(),
        8,
        "each guarded topology needs its own fixture"
    );
    for feature_id in [
        "constraint.geometry_attribute.instances.picked_instance",
        "constraint.geometry_attribute.instances.realized_points",
        "constraint.geometry_attribute.instances.non_point_attribute",
        "constraint.geometry_attribute.instances.attribute_conversion",
        "constraint.geometry_attribute.instances.transform_matrix_mode",
        "graph.instance_component.realize_attribute_columns",
        "graph.instance_component.store_named_attribute_uncaptured",
    ] {
        assert!(
            guard_cases
                .iter()
                .any(|guard| guard["feature_id"] == feature_id),
            "INSTANCE fixture omitted guarded feature {feature_id}"
        );
    }
    let guard_project = root.join("guard_project");
    let guard_import = import_blend_project(
        &blender,
        &guard_project,
        &root.join("geometry_attribute_instances_guards.blend"),
    )?;
    assert_eq!(guard_import["result"]["losses"], json!([]));
    let guard_scene: Value = serde_json::from_slice(&fs::read(guard_project.join("scene.json"))?)?;
    let guard_operations_path = root.join("instance_guard_operations.json");
    // Inspect evaluates the whole document, so enable one guarded target stack at a time.
    let mut active_guard_target: Option<ActiveGuardTarget> = None;
    for guard in guard_cases {
        let guard_name = guard["id"]
            .as_str()
            .ok_or_else(|| io::Error::other("guard fixture ID is missing"))?;
        let feature_id = guard["feature_id"]
            .as_str()
            .ok_or_else(|| io::Error::other("guard fixture feature ID is missing"))?;
        let inspect_name = guard["inspect_id"]
            .as_str()
            .ok_or_else(|| io::Error::other("guard inspect name is missing"))?;
        let node_id = imported_node_id(&guard_project, inspect_name)?;
        let constraint = guard_scene["nodes"]
            .get(node_id.as_str())
            .and_then(|node| node["constraints"].as_array())
            .and_then(|constraints| constraints.first());
        let constraint_id = constraint
            .and_then(|constraint| constraint["id"].as_str())
            .map(str::to_owned);
        let constraint_target = constraint
            .and_then(|constraint| constraint["target"].as_str())
            .map(str::to_owned);
        let expects_constraint = !matches!(
            guard_name,
            "store_named_attribute_uncaptured" | "transform_matrix_plain_mesh"
        );
        assert_eq!(
            constraint_id.is_some(),
            expects_constraint,
            "{guard_name} constraint fixture mismatch"
        );
        assert_eq!(
            constraint_target.is_some(),
            expects_constraint,
            "{guard_name} constraint target mismatch"
        );
        let target_node_id = constraint_target.clone().unwrap_or_else(|| node_id.clone());
        let target_modifier = guard_scene["nodes"]
            .get(target_node_id.as_str())
            .and_then(|node| node["modifiers"].as_array())
            .and_then(|modifiers| {
                modifiers
                    .iter()
                    .find(|modifier| modifier["type"] == "nodes")
            })
            .ok_or_else(|| io::Error::other("guard Geometry Nodes modifier is missing"))?;
        let modifier_id = target_modifier["id"]
            .as_str()
            .ok_or_else(|| io::Error::other("guard Geometry Nodes modifier ID is missing"))?
            .to_owned();
        assert_eq!(
            target_modifier["show_viewport"], false,
            "{guard_name} target modifier was not imported disabled"
        );

        let mut operations = Vec::with_capacity(4);
        if let Some(previous) = active_guard_target.take() {
            operations.push(json!({
                "op":"modifier.update",
                "target":{"id":previous.target_node},
                "id":previous.modifier_id,
                "set":{"show_viewport":false}
            }));
            if let Some((previous_owner, previous_id)) = previous.constraint {
                operations.push(json!({
                    "op":"constraint.update",
                    "target":{"id":previous_owner},
                    "id":previous_id,
                    "set":{"enabled":false,"target":previous.target_node}
                }));
            }
        }
        operations.push(json!({
            "op":"modifier.update",
            "target":{"id":target_node_id},
            "id":modifier_id,
            "set":{"show_viewport":true}
        }));
        let active_constraint = if let Some(constraint_id) = constraint_id {
            let constraint =
                constraint.ok_or_else(|| io::Error::other("guard constraint is missing"))?;
            assert_eq!(
                constraint["enabled"], false,
                "{guard_name} was not imported muted"
            );
            operations.push(json!({
                "op":"constraint.update",
                "target":{"id":node_id},
                "id":constraint_id,
                "set":{"enabled":true,"target":target_node_id}
            }));
            Some((node_id.clone(), constraint_id))
        } else {
            None
        };
        active_guard_target = Some(ActiveGuardTarget {
            target_node: target_node_id,
            modifier_id,
            constraint: active_constraint,
        });
        let current_scene: Value =
            serde_json::from_slice(&fs::read(guard_project.join("scene.json"))?)?;
        let revision = current_scene["revision"]
            .as_u64()
            .ok_or_else(|| io::Error::other("guard project revision is missing"))?;
        fs::write(
            &guard_operations_path,
            serde_json::to_vec(&json!({
                "schema_version":1,
                "base_revision":revision,
                "operations":operations
            }))?,
        )?;
        let applied = pot()
            .arg("apply")
            .arg(&guard_project)
            .arg("--file")
            .arg(&guard_operations_path)
            .arg("--json")
            .output()?;
        assert!(
            applied.status.success(),
            "guard topology setup failed: {}",
            String::from_utf8_lossy(&applied.stdout)
        );
        let output = pot()
            .arg("inspect")
            .arg(&guard_project)
            .args(["--id", node_id.as_str(), "--json"])
            .output()?;
        if guard_name == "transform_matrix_plain_mesh" && output.status.success() {
            let response: Value = serde_json::from_slice(&output.stdout)?;
            let item = response["result"]["items"]
                .as_array()
                .and_then(|items| {
                    items
                        .iter()
                        .find(|item| item["name"].as_str() == Some(inspect_name))
                })
                .ok_or_else(|| io::Error::other("plain-mesh Matrix inspect omitted its target"))?;
            assert_json_positions_match(
                &item["evaluated_geometry"]["positions"],
                &guard["oracle_positions"],
            )?;
            continue;
        }
        assert!(
            !output.status.success(),
            "{guard_name} guarded INSTANCE case unexpectedly evaluated"
        );
        let response: Value = serde_json::from_slice(&output.stdout)?;
        assert_eq!(
            response["error"]["code"], "UNSUPPORTED_FEATURE",
            "{guard_name}: {response}"
        );
        assert_eq!(
            response["error"]["details"]["feature_id"], feature_id,
            "{guard_name}: {response}"
        );
        if matches!(
            guard_name,
            "store_named_attribute_uncaptured" | "realize_attribute_columns"
        ) {
            let values: Vec<[f64; 3]> =
                serde_json::from_value(guard["oracle_attribute_values"].clone())?;
            assert!(
                !values.is_empty(),
                "{guard_name} lacks Blender attribute oracle rows"
            );
            assert!(
                values.iter().all(|value| {
                    (value[0] - 0.5).abs() < 1.0e-5
                        && (value[1] + 1.25).abs() < 1.0e-5
                        && (value[2] - 2.0).abs() < 1.0e-5
                }),
                "{guard_name} Blender realized INSTANCE attribute differs: {values:?}"
            );
        }
    }

    let fresh_oracle = &oracle["fresh_session"];
    let fresh_baseline = blender_matrix(&fresh_oracle["owner_baseline"])?;
    let fresh_expected = blender_matrix(&fresh_oracle["owner_matrix"])?;
    assert!(
        max_matrix_error(fresh_baseline, fresh_expected) > 1.0e-2,
        "Blender fresh-session owner constraint did not change the join input transform"
    );
    let fresh_project = root.join("fresh_session_project");
    let fresh_import = import_blend_project(
        &blender,
        &fresh_project,
        &root.join("geometry_attribute_instances_fresh_session.blend"),
    )?;
    assert_eq!(fresh_import["result"]["losses"], json!([]));
    let fresh_scene: Value = serde_json::from_slice(&fs::read(fresh_project.join("scene.json"))?)?;
    let revision = fresh_scene["revision"]
        .as_u64()
        .ok_or_else(|| io::Error::other("fresh-session project revision is missing"))?;
    let join_batch = root.join("fresh_session_join.json");
    fs::write(
        &join_batch,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": revision,
            "operations": [{
                "op": "node.join",
                "targets": [
                    imported_node_id(&fresh_project, "FreshSessionJoinDestination")?,
                    imported_node_id(&fresh_project, "FreshSessionConstraintOwner")?,
                ]
            }]
        }))?,
    )?;
    let joined = pot()
        .arg("apply")
        .arg(&fresh_project)
        .arg("--file")
        .arg(&join_batch)
        .arg("--json")
        .output()?;
    assert!(
        !joined.status.success(),
        "node.join silently used a default transform for an INSTANCE constraint target"
    );
    let joined: Value = serde_json::from_slice(&joined.stdout)?;
    assert_eq!(joined["error"]["code"], "UNSUPPORTED_FEATURE", "{joined}");
    assert_eq!(
        joined["error"]["details"]["feature_id"],
        "constraint.geometry_attribute.instances.no_evaluated_geometry",
        "{joined}"
    );
    let particle_oracle = &oracle["particle_halo"];
    assert_eq!(particle_oracle["render_type"], "HALO");
    let particle_project = root.join("particle_halo_project");
    let particle_import = import_blend_project(
        &blender,
        &particle_project,
        &root.join("geometry_attribute_instances_particle_halo.blend"),
    )?;
    assert!(
        particle_import["result"]["losses"].is_array(),
        "separate HALO particle import must report its own loss list"
    );
    let particle_target_name = particle_oracle["target"]
        .as_str()
        .ok_or_else(|| io::Error::other("HALO particle target name is missing"))?;
    let particle_scene: SceneDoc =
        serde_json::from_slice(&fs::read(particle_project.join("scene.json"))?)?;
    let particle_target = particle_scene
        .nodes
        .values()
        .find(|node| node.name.as_str() == particle_target_name)
        .ok_or_else(|| io::Error::other("imported HALO particle target is missing"))?;
    assert!(
        particle_target.modifiers.iter().any(|modifier| {
            modifier.modifier_type == "particle_system"
                && modifier.enabled
                && modifier.show_viewport
                && modifier.show_render
        }),
        "HALO particle target did not retain an active legacy particle system"
    );
    let particle_settings = particle_target
        .properties
        .get("physics_particle_systems")
        .and_then(Value::as_object)
        .and_then(|systems| {
            systems
                .values()
                .find(|settings| settings["render_type"] == "HALO")
        })
        .ok_or_else(|| io::Error::other("imported particle settings are not HALO"))?;
    assert_eq!(particle_settings["render_type"], "HALO");
    let particle_id = particle_oracle["id"]
        .as_str()
        .ok_or_else(|| io::Error::other("HALO particle owner ID is missing"))?;
    let particle_inspected = pot()
        .args(["inspect"])
        .arg(&particle_project)
        .args(["--json"])
        .output()?;
    assert!(
        particle_inspected.status.success(),
        "HALO particles must not refuse Geometry Attribute INSTANCE sampling: {}",
        String::from_utf8_lossy(&particle_inspected.stdout)
    );
    let particle_inspected: Value = serde_json::from_slice(&particle_inspected.stdout)?;
    let particle_item = particle_inspected["result"]["items"]
        .as_array()
        .and_then(|items| {
            items
                .iter()
                .find(|item| item["name"].as_str() == Some(particle_id))
        })
        .ok_or_else(|| io::Error::other("HALO particle inspect omitted its owner"))?;
    let particle_expected = blender_matrix(&particle_oracle["matrix"])?;
    let particle_actual = blender_matrix(&particle_item["transform"]["world"]["matrix"])?;
    let particle_error = max_matrix_error(particle_actual, particle_expected);
    assert!(
        particle_error < 2.0e-4,
        "HALO particle Geometry Attribute INSTANCE matrix differs by {particle_error:e}: Potter={particle_actual:?}, Blender={particle_expected:?}"
    );
    assert!(
        max_matrix_error(
            particle_expected,
            blender_matrix(&particle_oracle["baseline"])?
        ) > 1.0e-2,
        "HALO particle oracle did not materially exercise its INSTANCE constraint"
    );
    let owner_name = cases[0]
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| io::Error::other("INSTANCE driver owner is missing"))?;
    let owner_id = imported_node_id(&project, owner_name)?;
    let baseline_owner_inspect = pot()
        .args(["inspect"])
        .arg(&project)
        .arg("--id")
        .arg(&owner_id)
        .arg("--json")
        .output()?;
    assert!(
        baseline_owner_inspect.status.success(),
        "baseline selected-owner inspect failed: {}",
        String::from_utf8_lossy(&baseline_owner_inspect.stdout)
    );
    let imported_scene: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let revision = imported_scene["revision"]
        .as_u64()
        .ok_or_else(|| io::Error::other("imported project revision is missing"))?;
    let driver_batch = root.join("instance_driver_operations.json");
    fs::write(
        &driver_batch,
        serde_json::to_vec(&json!({
            "schema_version":1,
            "base_revision":revision,
            "operations":[
                {"op":"node.create","id":"probe","kind":"empty"},
                {"op":"driver.create","target":{"id":"probe"},"id":"world_instance_target","path":"transform.translation","index":0,"type":"scripted_expression","variables":[{"name":"x","type":"transforms","target":owner_id.clone(),"path":"transform.translation","index":0,"transform_space":"world"}],"expression":"x"}
            ]
        }))?,
    )?;
    let applied_driver = pot()
        .arg("apply")
        .arg(&project)
        .arg("--file")
        .arg(&driver_batch)
        .arg("--json")
        .output()?;
    assert!(
        applied_driver.status.success(),
        "world-space driver setup failed: {}",
        String::from_utf8_lossy(&applied_driver.stdout)
    );
    for id in [owner_id.as_str(), "probe"] {
        let inspected = pot()
            .args(["inspect"])
            .arg(&project)
            .args(["--id", id, "--json"])
            .output()?;
        assert!(
            !inspected.status.success(),
            "selected inspect of {id} accepted a world-space driver with an INSTANCE constraint"
        );
        let response: Value = serde_json::from_slice(&inspected.stdout)?;
        assert_eq!(
            response["error"]["code"], "UNSUPPORTED_FEATURE",
            "{id}: {response}"
        );
        assert_eq!(
            response["error"]["details"]["feature_id"],
            "driver.geometry_attribute.instance_dependency",
            "{id}: {response}"
        );
    }

    let driven_scene: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let driven_revision = driven_scene["revision"]
        .as_u64()
        .ok_or_else(|| io::Error::other("driven project revision is missing"))?;
    let local_driver_batch = root.join("instance_local_driver_operations.json");
    fs::write(
        &local_driver_batch,
        serde_json::to_vec(&json!({
            "schema_version":1,
            "base_revision":driven_revision,
            "operations":[{
                "op":"driver.update",
                "target":{"id":"probe"},
                "id":"world_instance_target",
                "set":{"variables":[{"name":"x","type":"transforms","target":owner_id.clone(),"path":"transform.translation","index":0,"transform_space":"local"}]}
            }]
        }))?,
    )?;
    let updated_driver = pot()
        .args(["apply"])
        .arg(&project)
        .args(["--file"])
        .arg(&local_driver_batch)
        .args(["--json"])
        .output()?;
    assert!(
        updated_driver.status.success(),
        "local-space driver update failed: {}",
        String::from_utf8_lossy(&updated_driver.stdout)
    );
    for id in [owner_id.as_str(), "probe"] {
        let inspected = pot()
            .args(["inspect"])
            .arg(&project)
            .args(["--id", id, "--json"])
            .output()?;
        assert!(
            inspected.status.success(),
            "selected inspect of {id} failed with a local-space driver and an INSTANCE constraint: {}",
            String::from_utf8_lossy(&inspected.stdout)
        );
    }
    eprintln!(
        "Blender 5.2.2 Geometry Attribute INSTANCE CLI matrices: import max error {maximum_error:.6e} across {} supported cases, {} guarded topologies, HALO particle matrix error {particle_error:.6e}",
        cases.len(),
        guard_cases.len()
    );
    Ok(())
}

fn assert_hair_curves_surface_unlink(
    blender: &Path,
    project: &Path,
    output_directory: &Path,
) -> TestResult {
    let scene_path = project.join("scene.json");
    let scene: Value = serde_json::from_slice(&fs::read(&scene_path)?)?;
    let nodes = scene["nodes"]
        .as_object()
        .ok_or_else(|| io::Error::other("Geometry Attribute Curves scene has no nodes"))?;
    let surface_id = nodes
        .iter()
        .find(|(_, node)| node["name"] == "GeometryAttributeSurface")
        .map(|(id, _)| id.clone())
        .ok_or_else(|| io::Error::other("Geometry Attribute surface mesh is missing"))?;
    let hair_node = nodes
        .values()
        .find(|node| node["name"] == "GeometryAttributeTarget")
        .ok_or_else(|| io::Error::other("Geometry Attribute Curves target is missing"))?;
    let data_id = hair_node["data"]
        .as_str()
        .ok_or_else(|| io::Error::other("Geometry Attribute Curves data ID is missing"))?
        .to_owned();
    assert_eq!(
        scene["data_blocks"][&data_id]["hair_curves"]["surface"].as_str(),
        Some(surface_id.as_str())
    );

    let operations_path = output_directory.join("delete_hair_surface.json");
    let operations = json!({
        "schema_version": 1,
        "base_revision": scene["revision"],
        "operations": [{"op":"node.delete","target":{"id":surface_id}}]
    });
    fs::write(&operations_path, serde_json::to_vec(&operations)?)?;
    let applied = pot()
        .args(["apply"])
        .arg(project)
        .args(["--file"])
        .arg(&operations_path)
        .args(["--json"])
        .output()?;
    assert!(
        applied.status.success(),
        "deleting the hair Curves surface failed: {}",
        String::from_utf8_lossy(&applied.stdout)
    );
    let applied: Value = serde_json::from_slice(&applied.stdout)?;
    let updated_data_blocks = applied["result"]["changes"]["data_blocks"]["updated"]
        .as_array()
        .ok_or_else(|| io::Error::other("node.delete returned no updated data blocks"))?;
    assert!(
        updated_data_blocks
            .iter()
            .any(|updated_id| updated_id.as_str() == Some(data_id.as_str())),
        "node.delete did not report hair Curves data block {data_id} as updated: {updated_data_blocks:?}"
    );

    let doc: SceneDoc = serde_json::from_slice(&fs::read(&scene_path)?)?;
    doc.validate()?;
    let hair_data_id = Id::new(data_id.clone())?;
    assert!(
        doc.data_blocks
            .get(&hair_data_id)
            .and_then(|block| block.hair_curves.as_ref())
            .is_some_and(|hair| hair.surface.is_none()),
        "node.delete retained the deleted mesh in the hair Curves surface binding"
    );

    let inspected = pot()
        .args(["inspect"])
        .arg(project)
        .args(["--json"])
        .output()?;
    assert!(
        inspected.status.success(),
        "pot inspect failed after deleting the hair Curves surface: {}",
        String::from_utf8_lossy(&inspected.stdout)
    );

    let exported = pot()
        .args(["export"])
        .arg(project)
        .args(["--format", "blend", "--out"])
        .arg(output_directory.join("hair_surface_unlinked.blend"))
        .args(["--blender"])
        .arg(blender)
        .args(["--json"])
        .output()?;
    assert!(
        exported.status.success(),
        "pot .blend export failed after deleting the hair Curves surface: {}",
        String::from_utf8_lossy(&exported.stdout)
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
            Some(&project.join(".potter/cache")),
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
fn blender_transform_cache_sequence_gap_reuses_the_selected_archive() -> TestResult {
    let Some(blender) = blender_executable() else {
        return Err(
            io::Error::other("Blender is required for the Transform Cache sequence probe").into(),
        );
    };
    let directory = tempdir()?;
    let root = fs::canonicalize(directory.path())?;
    let output_path = root.join("sequence_matrices.json");
    let script = r#"
import bpy, json, os, sys
root = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.fps = 1
def new_source():
    mesh = bpy.data.meshes.new("Mesh")
    mesh.from_pydata([(-1,-1,0),(1,-1,0),(1,1,0),(-1,1,0)], [], [(0,1,2,3)])
    obj = bpy.data.objects.new("Animated", mesh); scene.collection.objects.link(obj)
    return obj, mesh
for frame, x in ((1,10.0),(2,20.0)):
    scene.frame_set(frame); obj, mesh = new_source()
    obj.location.x = x; obj.select_set(True); bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.alembic_export(filepath=os.path.join(root,f"seq.{frame:04d}.abc"),
        start=frame,end=frame,selected=True,flatten=False,uvs=False,normals=False,as_background_job=False)
    bpy.data.objects.remove(obj,do_unlink=True); bpy.data.meshes.remove(mesh)
scene.frame_set(3); obj, mesh = new_source()
basis = obj.shape_key_add(name="Basis")
lift = obj.shape_key_add(name="Lift")
lift.data[0].co.z = 2.0
lift.value = 0.0; lift.keyframe_insert(data_path="value",frame=3)
lift.value = 1.0; lift.keyframe_insert(data_path="value",frame=4)
obj.location.x = 30.0; obj.keyframe_insert(data_path="location",frame=3)
obj.location.x = 40.0; obj.keyframe_insert(data_path="location",frame=4)
obj.select_set(True); bpy.context.view_layer.objects.active = obj
bpy.ops.wm.alembic_export(filepath=os.path.join(root,"seq.0004.abc"),
    start=3,end=4,selected=True,flatten=False,uvs=False,normals=False,as_background_job=False)
bpy.data.objects.remove(obj,do_unlink=True); bpy.data.meshes.remove(mesh)
scene.render.fps = 24
first_path = os.path.join(root,"seq.0001.abc")
bpy.ops.wm.alembic_import(filepath=first_path,set_frame_range=False,always_add_cache_reader=True,as_background_job=False)
cache = bpy.data.cache_files[-1]
object_path = cache.object_paths[0].path
cache.filepath = first_path; cache.is_sequence = True
for obj in list(bpy.data.objects):
    if obj.type == "MESH": bpy.data.objects.remove(obj,do_unlink=True)
def new_owner(name):
    obj = bpy.data.objects.new(name,None); scene.collection.objects.link(obj)
    obj.location=(3.0,5.0,7.0); obj.rotation_euler=(0.2,-0.3,0.4); obj.scale=(1.5,2.0,0.75)
    return obj
def matrix(obj):
    value=obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world
    return [float(value[r][c]) for c in range(4) for r in range(4)]
def positions(obj):
    evaluated=obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh=evaluated.to_mesh()
    try: return [[float(v.co[i]) for i in range(3)] for v in mesh.vertices]
    finally: evaluated.to_mesh_clear()
owner = new_owner("SequenceOwner")
constraint = owner.constraints.new("TRANSFORM_CACHE")
constraint.cache_file=cache; constraint.object_path=object_path
receiver_mesh=bpy.data.meshes.new("ReceiverMesh")
receiver_mesh.from_pydata([(-1,-1,0),(1,-1,0),(1,1,0),(-1,1,0)], [], [(0,1,2,3)])
receiver=bpy.data.objects.new("Receiver",receiver_mesh); scene.collection.objects.link(receiver)
modifier=receiver.modifiers.new("Sequence Cache","MESH_SEQUENCE_CACHE")
modifier.cache_file=cache; modifier.object_path=object_path; modifier.read_data={"VERT","POLY"}
sequence={}
for frame in (4,3):
    scene.frame_set(frame); bpy.context.view_layer.update()
    sequence[str(frame)]={"matrix":matrix(owner),"vertices":positions(receiver)}
assert sequence["4"]["matrix"] != sequence["3"]["matrix"]
assert sequence["4"]["vertices"] != sequence["3"]["vertices"]
cache.filepath = os.path.join(root,"seq.0002.abc")
from_two = {}
for frame in (2,3):
    scene.frame_set(frame); bpy.context.view_layer.update()
    from_two[str(frame)]={"matrix":matrix(owner),"vertices":positions(receiver)}
cache.filepath = os.path.join(root,"seq.0004.abc")
missing_cache_path=os.path.join(root,"seq.0003.abc")
bpy.ops.cachefile.open(filepath=missing_cache_path)
missing_cache=bpy.data.cache_files[-1]; missing_cache.is_sequence=True
fresh_owner=new_owner("FreshOwner")
fresh_constraint=fresh_owner.constraints.new("TRANSFORM_CACHE")
fresh_constraint.cache_file=missing_cache; fresh_constraint.object_path=object_path
fresh_mesh=bpy.data.meshes.new("FreshMesh")
fresh_mesh.from_pydata([(-1,-1,0),(1,-1,0),(1,1,0),(-1,1,0)], [], [(0,1,2,3)])
fresh_receiver=bpy.data.objects.new("FreshReceiver",fresh_mesh); scene.collection.objects.link(fresh_receiver)
fresh_modifier=fresh_receiver.modifiers.new("Missing Cache","MESH_SEQUENCE_CACHE")
fresh_modifier.cache_file=missing_cache; fresh_modifier.object_path=object_path; fresh_modifier.read_data={"VERT","POLY"}
scene.frame_set(3); bpy.context.view_layer.update()
fresh_missing={"matrix":matrix(fresh_owner),"vertices":positions(fresh_receiver)}
with open(os.path.join(root,"sequence_matrices.json"),"w",encoding="utf-8") as f:
    json.dump({"sequence":sequence,"from_two":from_two,"fresh_missing":fresh_missing},f)
"#;
    let script_path = root.join("make_transform_sequence.py");
    fs::write(&script_path, script)?;
    let blender_output = Command::new(&blender)
        .args(["--background", "--factory-startup", "--python"])
        .arg(&script_path)
        .args(["--", root.to_str().ok_or("probe path is not UTF-8")?])
        .output()?;
    assert!(
        blender_output.status.success(),
        "Blender sequence probe failed: {}",
        String::from_utf8_lossy(&blender_output.stderr)
    );
    let pattern = root.join("seq.####.abc");
    let (_directory, project_path) = project(&json!([
        {"op":"camera.create","id":"camera_main","transform":{"translation":[0.0,0.0,4.0]},"projection":"orthographic","ortho_scale":3.0},
        {"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"camera_main"}},
        {"op":"node.create","id":"owner","kind":"empty","transform":{"translation":[3.0,5.0,7.0],"rotation_deg":[11.459_156,-17.188_734,22.918_312],"scale":[1.5,2.0,0.75]}},
        {"op":"node.create","id":"receiver","kind":"plane","params":{"size":2.0}},
        {"op":"resource.pack","id":"cache","uri":pattern,"kind":"alembic","sequence":{"start":1,"end":2}},
        {"op":"constraint.create","target":{"id":"owner"},"id":"cache","type":"transform_cache","params":{"resource":"cache","object_path":"/Animated/Mesh","frame_offset":0.0,"scale":1.0,"override_frame":null}},
        {"op":"modifier.create","target":{"id":"receiver"},"id":"sequence","type":"mesh_sequence_cache","params":{"resource":"cache","object_path":"/Animated/Mesh","is_sequence":true,"read_data":["VERT","POLY"]}}
    ]))?;
    let frame4 = fs::read(root.join("seq.0004.abc"))?;
    let digest = potter_core::hash::sha256(&frame4);
    let mut document: Value = serde_json::from_slice(&fs::read(project_path.join("scene.json"))?)?;
    document["resources"]["cache"]["sequence_end"] = json!(4);
    document["resources"]["cache"]["sequence_frames"]["4"] = json!({
        "uri":root.join("seq.0004.abc"),"kind":"alembic","hash":digest,"packed":true,
        "bytes":serde_json::to_value(frame4)?
    });
    document["scenes"]["scene_main"]["render"]["resolution_x"] = json!(64);
    document["scenes"]["scene_main"]["render"]["resolution_y"] = json!(64);
    document["scenes"]["scene_main"]["render"]["resolution_percentage"] = json!(100);
    document["scenes"]["scene_main"]["render"]["samples"] = json!(1);
    fs::write(
        project_path.join("scene.json"),
        serde_json::to_vec_pretty(&document)?,
    )?;
    let document: SceneDoc = serde_json::from_slice(&fs::read(project_path.join("scene.json"))?)?;
    let oracle: Value = serde_json::from_slice(&fs::read(output_path)?)?;
    let assert_frame = |snapshot: &Snapshot, frame: i32| -> TestResult {
        let oracle_frame = &oracle["sequence"][frame.to_string()];
        let actual = snapshot.nodes[&Id::new("owner")?].world_matrix;
        assert!(
            max_matrix_error(
                DMat4::from_cols_array(&actual),
                blender_matrix(&oracle_frame["matrix"])?
            ) < 1.0e-5,
            "Transform Cache differs at frame {frame}"
        );
        let mesh = snapshot
            .meshes
            .get(&Id::new("receiver")?)
            .ok_or_else(|| io::Error::other("Mesh Sequence Cache receiver mesh is missing"))?;
        assert_mesh_positions_match(mesh, &oracle_frame["vertices"])?;
        Ok(())
    };
    let mut session = EvaluationSession::default();
    let fresh_frame4 = Snapshot::evaluate_with_cache_session(
        &document,
        &EvaluationContext {
            frame: Some(4.0),
            purpose: EvaluationPurpose::Render,
            ..EvaluationContext::default()
        },
        Some(&project_path),
        Some(&project_path.join(".potter/cache")),
        &mut session,
    )?;
    let warm_frame4 = Snapshot::evaluate_with_cache_session(
        &document,
        &EvaluationContext {
            frame: Some(4.0),
            purpose: EvaluationPurpose::Render,
            ..EvaluationContext::default()
        },
        Some(&project_path),
        Some(&project_path.join(".potter/cache")),
        &mut session,
    )?;
    assert_eq!(
        fresh_frame4.evaluation_hash, warm_frame4.evaluation_hash,
        "same-frame reevaluation with the retained reader must keep its key stable"
    );
    assert_frame(&fresh_frame4, 4)?;
    assert_frame(&warm_frame4, 4)?;
    let warm_frame3 = Snapshot::evaluate_with_cache_session(
        &document,
        &EvaluationContext {
            frame: Some(3.0),
            purpose: EvaluationPurpose::Render,
            ..EvaluationContext::default()
        },
        Some(&project_path),
        Some(&project_path.join(".potter/cache")),
        &mut session,
    )?;
    assert_frame(&warm_frame3, 3)?;

    let mut mesh_only_document = document.clone();
    mesh_only_document
        .nodes
        .get_mut(&Id::new("owner")?)
        .ok_or_else(|| io::Error::other("Transform Cache owner is missing"))?
        .constraints
        .clear();
    let mut mesh_only_session = EvaluationSession::default();
    let fresh_mesh_only = Snapshot::evaluate_with_cache_session(
        &mesh_only_document,
        &EvaluationContext {
            frame: Some(4.0),
            purpose: EvaluationPurpose::Render,
            ..EvaluationContext::default()
        },
        Some(&project_path),
        Some(&project_path.join(".potter/cache")),
        &mut mesh_only_session,
    )?;
    let warm_mesh_only = Snapshot::evaluate_with_cache_session(
        &mesh_only_document,
        &EvaluationContext {
            frame: Some(4.0),
            purpose: EvaluationPurpose::Render,
            ..EvaluationContext::default()
        },
        Some(&project_path),
        Some(&project_path.join(".potter/cache")),
        &mut mesh_only_session,
    )?;
    assert_eq!(
        fresh_mesh_only.evaluation_hash, warm_mesh_only.evaluation_hash,
        "Mesh Sequence Cache source selection must be included after modifier evaluation"
    );
    assert_mesh_positions_match(
        warm_mesh_only
            .meshes
            .get(&Id::new("receiver")?)
            .ok_or_else(|| io::Error::other("Mesh Sequence Cache receiver mesh is missing"))?,
        &oracle["sequence"]["4"]["vertices"],
    )?;
    let retained_missing = Snapshot::evaluate_with_cache_session(
        &document,
        &EvaluationContext {
            frame: Some(3.0),
            purpose: EvaluationPurpose::Render,
            ..EvaluationContext::default()
        },
        Some(&project_path),
        Some(&project_path.join(".potter/cache")),
        &mut session,
    )?;
    let mut fresh_session = EvaluationSession::default();
    let fresh_missing = Snapshot::evaluate_with_cache_session(
        &document,
        &EvaluationContext {
            frame: Some(3.0),
            purpose: EvaluationPurpose::Render,
            ..EvaluationContext::default()
        },
        Some(&project_path),
        Some(&project_path.join(".potter/cache")),
        &mut fresh_session,
    )?;
    assert_ne!(
        retained_missing.evaluation_hash, fresh_missing.evaluation_hash,
        "the selected CacheFile archive participates in the evaluation key"
    );
    let actual = fresh_missing.nodes[&Id::new("owner")?].world_matrix;
    assert!(
        max_matrix_error(
            DMat4::from_cols_array(&actual),
            blender_matrix(&oracle["fresh_missing"]["matrix"])?
        ) < 1.0e-5,
        "without a selected CacheFile archive, Transform Cache must preserve the owner's base matrix"
    );
    let mesh = fresh_missing
        .meshes
        .get(&Id::new("receiver")?)
        .ok_or_else(|| io::Error::other("fresh missing-frame receiver mesh is missing"))?;
    assert_mesh_positions_match(mesh, &oracle["fresh_missing"]["vertices"])?;

    let mut from_two_session = EvaluationSession::default();
    let _ = Snapshot::evaluate_with_cache_session(
        &document,
        &EvaluationContext {
            frame: Some(2.0),
            purpose: EvaluationPurpose::Render,
            ..EvaluationContext::default()
        },
        Some(&project_path),
        Some(&project_path.join(".potter/cache")),
        &mut from_two_session,
    )?;
    let from_two_missing = Snapshot::evaluate_with_cache_session(
        &document,
        &EvaluationContext {
            frame: Some(3.0),
            purpose: EvaluationPurpose::Render,
            ..EvaluationContext::default()
        },
        Some(&project_path),
        Some(&project_path.join(".potter/cache")),
        &mut from_two_session,
    )?;
    let from_two_oracle = &oracle["from_two"]["3"];
    assert!(
        max_matrix_error(
            DMat4::from_cols_array(&from_two_missing.nodes[&Id::new("owner")?].world_matrix),
            blender_matrix(&from_two_oracle["matrix"])?
        ) < 1.0e-5,
        "a gap after frame 2 must sample the retained frame-2 archive"
    );
    assert_ne!(
        retained_missing.evaluation_hash, from_two_missing.evaluation_hash,
        "different retained archives at frame 3 must produce distinct evaluation keys"
    );

    let render_output = project_path.join("render-cache-sequence");
    let render = pot()
        .arg("render")
        .arg(&project_path)
        .args([
            "--frames",
            "2:3",
            "--camera",
            "camera_main",
            "--engine",
            "realtime",
            "--format",
            "png",
            "--out",
        ])
        .arg(&render_output)
        .arg("--json")
        .output()?;
    assert!(
        render.status.success(),
        "sequence CLI render failed: {}",
        String::from_utf8_lossy(&render.stdout)
    );
    let manifest: Value =
        serde_json::from_slice(&fs::read(render_output.join("render.manifest.json"))?)?;
    assert_eq!(
        manifest["frames"][1]["evaluation_hash"],
        from_two_missing.evaluation_hash
    );
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

#[test]
fn blender_euler_curves_and_transform_drivers_share_rotation_mode() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Blender Euler driver parity: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let generator = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/action_constraint_target_space/generate_euler_driver_oracle.py");
    let mut generate = Command::new(&blender);
    generate
        .args([
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "3",
            "--python",
        ])
        .arg(generator)
        .arg("--")
        .arg(root);
    let generated = process::run_guarded(generate)?;
    assert!(
        generated.status.success(),
        "Blender Euler driver oracle failed: stdout={} stderr={}",
        String::from_utf8_lossy(&generated.stdout),
        String::from_utf8_lossy(&generated.stderr),
    );
    let expected: Value = serde_json::from_slice(&fs::read(root.join("driver_expected.json"))?)?;

    assert!(root.join("driver_source.blend").is_file());
    let (_project_directory, project) = project(&json!([
        {"op":"node.create","id":"source","name":"EulerDriverSource","kind":"empty","transform":{"rotation_mode":"XZY"}},
        {"op":"node.create","id":"probe","name":"EulerDriverProbe","kind":"empty","transform":{"rotation_mode":"XZY"}},
        {"op":"action.create","id":"turn","name":"Turn"},
        {"op":"action.update","target":{"id":"turn"},"set":{"fcurves":[
            {"path":"transform.rotation_euler","index":0,"keyframes":[
                {"frame":1.0,"value":0.5,"interpolation":"constant"},
                {"frame":10.0,"value":0.5,"interpolation":"constant"}
            ]},
            {"path":"transform.rotation_euler","index":1,"keyframes":[
                {"frame":1.0,"value":0.5,"interpolation":"constant"},
                {"frame":10.0,"value":0.5,"interpolation":"constant"}
            ]},
            {"path":"transform.rotation_euler","index":2,"keyframes":[
                {"frame":1.0,"value":0.3,"interpolation":"constant"},
                {"frame":10.0,"value":0.3,"interpolation":"constant"}
            ]}
        ]}},
        {"op":"node.update","target":{"id":"source"},"set":{"action":"turn"}},
        {"op":"node.update","target":{"id":"probe"},"set":{"action":"turn"}},
        {"op":"driver.create","target":{"id":"probe"},"id":"euler_driver","path":"transform.rotation_euler","index":2,"type":"scripted_expression","variables":[{"name":"x","type":"transforms","target":"source","path":"transform.rotation_euler","index":0,"transform_space":"world"}],"expression":"x"}
    ]))?;

    let inspected = pot()
        .arg("inspect")
        .arg(&project)
        .arg("--frame")
        .arg(
            expected["frame"]
                .as_u64()
                .ok_or("Blender frame is missing")?
                .to_string(),
        )
        .arg("--json")
        .output()?;
    assert!(
        inspected.status.success(),
        "pot inspect failed: stdout={} stderr={}",
        String::from_utf8_lossy(&inspected.stdout),
        String::from_utf8_lossy(&inspected.stderr)
    );
    let inspected: Value = serde_json::from_slice(&inspected.stdout)?;
    let probe = inspected["result"]["items"]
        .as_array()
        .and_then(|items| items.iter().find(|item| item["name"] == "EulerDriverProbe"))
        .ok_or("imported EulerDriverProbe is missing")?;
    let actual = blender_matrix(&probe["transform"]["world"]["matrix"])?;
    let expected_matrix = blender_matrix(&expected["world_matrix"])?;
    let matrix_error = max_matrix_error(actual, expected_matrix);
    assert!(
        matrix_error <= 1.0e-5,
        "Potter Euler curve/driver world matrix differs from Blender: max error {matrix_error}, Potter={actual:?}, Blender={expected_matrix:?}"
    );
    let rotation: [f64; 4] =
        serde_json::from_value(probe["transform"]["world"]["rotation"].clone())?;
    let (_, driver_result, _) =
        glam::DQuat::from_xyzw(rotation[0], rotation[1], rotation[2], rotation[3])
            .to_euler(glam::EulerRot::XZYEx);
    let expected_driver_result = expected["driver_result"]
        .as_f64()
        .ok_or("Blender driver result is missing")?;
    assert!(
        (driver_result - expected_driver_result).abs() <= 1.0e-5,
        "Potter Euler driver result differs from Blender: Potter={driver_result}, Blender={expected_driver_result}"
    );
    Ok(())
}
