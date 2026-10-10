#![expect(
    clippy::unwrap_used,
    reason = "focused CLI regression tests use deterministic fixtures and fail immediately on setup errors"
)]

use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use glam::{DMat4, DQuat, DVec3};
use potter_core::{
    model::{
        CameraIntrinsics, Id, MovieClip, MovieTracking, ReconstructedPoint, Reconstruction,
        SceneDoc, SolvedCamera, TrackingMarker, TrackingTrack,
    },
    tracking::CameraModel,
};
use serde_json::{Value, json};
use tempfile::{TempDir, tempdir};

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn response(output: &Output) -> Value {
    serde_json::from_slice(&output.stdout).unwrap()
}

fn init(scene: &Path) {
    let output = pot().arg("init").arg(scene).arg("--json").output().unwrap();
    assert!(
        output.status.success(),
        "pot init failed: {}",
        String::from_utf8_lossy(&output.stdout)
    );
}

fn apply(scene: &Path, base_revision: u64, operations: &Value) -> Output {
    let directory = tempdir().unwrap();
    let batch = directory.path().join("operations.json");
    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": base_revision,
            "operations": operations,
        }))
        .unwrap(),
    )
    .unwrap();
    pot()
        .arg("apply")
        .arg(scene)
        .arg("--file")
        .arg(batch)
        .arg("--json")
        .output()
        .unwrap()
}

fn setup_scene(operations: &Value) -> (TempDir, PathBuf) {
    let directory = tempdir().unwrap();
    let scene = directory.path().join("scene");
    init(&scene);
    if !operations.as_array().unwrap().is_empty() {
        let output = apply(&scene, 0, operations);
        assert!(
            output.status.success(),
            "scene setup failed: {}",
            String::from_utf8_lossy(&output.stdout)
        );
    }
    (directory, scene)
}

struct InvalidCase {
    name: &'static str,
    setup: Value,
    operation: Value,
    code: &'static str,
    pointer: Option<&'static str>,
    feature_id: Option<&'static str>,
}

fn invalid_case(
    name: &'static str,
    setup: Value,
    operation: Value,
    code: &'static str,
    pointer: Option<&'static str>,
    feature_id: Option<&'static str>,
) -> InvalidCase {
    InvalidCase {
        name,
        setup,
        operation,
        code,
        pointer,
        feature_id,
    }
}

fn assert_invalid(
    output: &Output,
    name: &str,
    code: &str,
    pointer: Option<&str>,
    feature_id: Option<&str>,
) {
    assert!(!output.status.success(), "{name} unexpectedly succeeded");
    let envelope = response(output);
    assert_eq!(envelope["error"]["code"], code, "{name}: {envelope}");
    if let Some(pointer) = pointer {
        assert_eq!(
            envelope["error"]["details"]["pointer"], pointer,
            "{name}: {envelope}"
        );
        assert_eq!(
            envelope["error"]["details"]["operation_index"], 0,
            "{name}: {envelope}"
        );
    }
    if let Some(feature_id) = feature_id {
        assert_eq!(
            envelope["error"]["details"]["feature_id"], feature_id,
            "{name}: {envelope}"
        );
    }
}

fn run_invalid_cases(cases: &[InvalidCase]) {
    for case in cases {
        let (_directory, scene) = setup_scene(&case.setup);
        let base_revision = u64::from(!case.setup.as_array().unwrap().is_empty());
        let output = apply(&scene, base_revision, &json!([case.operation]));
        assert_invalid(&output, case.name, case.code, case.pointer, case.feature_id);
    }
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the constraint error table exercises distinct public validation contracts"
)]
fn constraint_parameter_rejections_report_codes_and_pointers() {
    let cases = [
        invalid_case(
            "negative limit distance",
            json!([{"op":"node.create","id":"owner","kind":"empty"},{"op":"node.create","id":"target","kind":"empty"}]),
            json!({"op":"constraint.create","target":{"id":"owner"},"id":"limit","type":"limit_distance","constraint_target":"target","params":{"distance":-0.01}}),
            "INVALID_OPERATION",
            Some("/operations/0/params/distance"),
            None,
        ),
        invalid_case(
            "zero IK iterations",
            json!([{"op":"node.create","id":"owner","kind":"armature"},{"op":"node.create","id":"target","kind":"empty"}]),
            json!({"op":"constraint.create","target":{"id":"owner"},"id":"ik","type":"ik","constraint_target":"target","params":{"iterations":0}}),
            "INVALID_OPERATION",
            Some("/operations/0/params/iterations"),
            None,
        ),
        invalid_case(
            "invalid tracking axis",
            json!([{"op":"node.create","id":"owner","kind":"empty"},{"op":"node.create","id":"target","kind":"empty"}]),
            json!({"op":"constraint.create","target":{"id":"owner"},"id":"track","type":"track_to","constraint_target":"target","params":{"track_axis":"TRACK_W"}}),
            "INVALID_ARGUMENT",
            Some("/operations/0/params/track_axis"),
            None,
        ),
        invalid_case(
            "unsupported transformation mapping",
            json!([{"op":"node.create","id":"owner","kind":"empty"},{"op":"node.create","id":"target","kind":"empty"}]),
            json!({"op":"constraint.create","target":{"id":"owner"},"id":"map","type":"transformation","constraint_target":"target","params":{"map_from":"POSITION"}}),
            "INVALID_ARGUMENT",
            Some("/operations/0/params/map_from"),
            None,
        ),
        invalid_case(
            "unsupported geometry attribute type",
            json!([
                {"op":"node.create","id":"owner","kind":"empty"},
                {"op":"node.create","id":"source","kind":"box","params":{"size":1.0}}
            ]),
            json!({"op":"constraint.create","target":{"id":"owner"},"id":"attribute","type":"geometry_attribute","constraint_target":"source","params":{"target":"source","attribute_name":"position","data_type":"FLOAT","domain":"POINT","sample_index":0,"mix_mode":"REPLACE","apply_target_transform":false,"mix_loc":true,"mix_rot":false,"mix_scl":false}}),
            "INVALID_ARGUMENT",
            Some("/operations/0/params/data_type"),
            None,
        ),
        invalid_case(
            "missing tracking clip",
            json!([{"op":"node.create","id":"owner","kind":"empty"},{"op":"node.create","id":"camera","kind":"camera"}]),
            json!({"op":"constraint.create","target":{"id":"owner"},"id":"follow","type":"follow_track","constraint_target":"camera","params":{"clip":"absent","track":"track","camera":"camera","frame_method":"STRETCH"}}),
            "TARGET_NOT_FOUND",
            Some("/operations/0/params/clip"),
            None,
        ),
        invalid_case(
            "missing object-solver object",
            json!([
                {"op":"node.create","id":"owner","kind":"empty"},
                {"op":"node.create","id":"camera","kind":"camera"},
                {"op":"tracking.clip_create","id":"clip","name":"Clip"}
            ]),
            json!({"op":"constraint.create","target":{"id":"owner"},"id":"solver","type":"object_solver","constraint_target":"camera","params":{"clip":"clip","object":"absent","camera":"camera","set_inverse_pending":false}}),
            "TARGET_NOT_FOUND",
            Some("/operations/0/params/object"),
            None,
        ),
        invalid_case(
            "missing action constraint action",
            json!([
                {"op":"node.create","id":"owner","kind":"empty"},
                {"op":"node.create","id":"driver","kind":"empty"}
            ]),
            json!({"op":"constraint.create","target":{"id":"owner"},"id":"action","type":"action","constraint_target":"driver","params":{"target":"driver","action":"absent","transform_channel":"LOCATION_X","target_space":"WORLD","mix_mode":"REPLACE","min":0.0,"max":1.0,"frame_start":1,"frame_end":10,"eval_time":0.0}}),
            "TARGET_NOT_FOUND",
            Some("/operations/0/params/action"),
            None,
        ),
        invalid_case(
            "missing IK pole node",
            json!([{"op":"node.create","id":"owner","kind":"armature"},{"op":"node.create","id":"target","kind":"empty"}]),
            json!({"op":"constraint.create","target":{"id":"owner"},"id":"ik","type":"ik","constraint_target":"target","params":{"iterations":1,"pole_target":"absent"}}),
            "TARGET_NOT_FOUND",
            Some("/operations/0/params/pole_target"),
            None,
        ),
        invalid_case(
            "follow-path factor above its maximum",
            json!([
                {"op":"node.create","id":"owner","kind":"empty"},
                {"op":"curve.create","id":"path","splines":[{"type":"poly","points":[{"co":[0.0,0.0,0.0]},{"co":[1.0,0.0,0.0]}]}]}
            ]),
            json!({"op":"constraint.create","target":{"id":"owner"},"id":"follow","type":"follow_path","constraint_target":"path","params":{"offset_factor":1.01}}),
            "INVALID_OPERATION",
            Some("/operations/0/params/offset_factor"),
            None,
        ),
        invalid_case(
            "constraint update rejects a negative distance",
            json!([
                {"op":"node.create","id":"owner","kind":"empty"},
                {"op":"node.create","id":"target","kind":"empty"},
                {"op":"constraint.create","target":{"id":"owner"},"id":"limit","type":"limit_distance","constraint_target":"target","params":{"distance":1.0}}
            ]),
            json!({"op":"constraint.update","target":{"id":"owner"},"id":"limit","set":{"params":{"distance":-0.01}}}),
            "INVALID_OPERATION",
            Some("/operations/0/set/params/distance"),
            None,
        ),
    ];
    run_invalid_cases(&cases);
}

#[test]
fn constraint_valid_boundaries_are_committed() {
    let (_directory, scene) = setup_scene(&json!([
        {"op":"node.create","id":"owner","kind":"empty"},
        {"op":"curve.create","id":"path","splines":[{"type":"poly","points":[{"co":[0.0,0.0,0.0]},{"co":[1.0,0.0,0.0]}]}]}
    ]));
    let output = apply(
        &scene,
        1,
        &json!([
            {"op":"constraint.create","target":{"id":"owner"},"id":"distance","type":"limit_distance","constraint_target":"path","params":{"distance":0.0}},
            {"op":"constraint.create","target":{"id":"owner"},"id":"follow","type":"follow_path","constraint_target":"path","params":{"offset_factor":1.0}},
            {"op":"constraint.create","target":{"id":"owner"},"id":"map","type":"transformation","constraint_target":"path","params":{"map_from":"LOCATION","map_to":"LOCATION","from_min_x":0.0,"from_max_x":4.440_892_098_500_626e-16,"to_min_x":0.0,"to_max_x":1.0,"from_min_y":0.0,"from_max_y":0.0,"to_min_y":0.0,"to_max_y":1.0}},
            {"op":"constraint.create","target":{"id":"owner"},"id":"track","type":"track_to","constraint_target":"path","params":{"track_axis":"TRACK_NEGATIVE_X","up_axis":"UP_Z"}},
            {"op":"constraint.create","target":{"id":"owner"},"id":"stretch","type":"stretch_to","constraint_target":"path","params":{"rest_length":0.0}}
        ]),
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let document: SceneDoc =
        serde_json::from_slice(&fs::read(scene.join("scene.json")).unwrap()).unwrap();
    let constraints = &document.nodes[&Id::new("owner").unwrap()].constraints;
    assert_eq!(constraints.len(), 5);
    assert_eq!(constraints[0].params["distance"], 0.0);
    assert_eq!(constraints[1].params["offset_factor"], 1.0);
    assert_eq!(constraints[2].params["from_max_y"], 0.0);
    assert_eq!(
        constraints[2].params["from_max_x"],
        4.440_892_098_500_626e-16
    );
    assert_eq!(constraints[3].params["track_axis"], "TRACK_NEGATIVE_X");
    assert_eq!(constraints[4].params["rest_length"], 0.0);
}
#[test]
fn tracking_and_action_constraints_accept_valid_references_and_boundaries() {
    let (_directory, scene) = setup_scene(&json!([
        {"op":"node.create","id":"owner","kind":"empty"},
        {"op":"node.create","id":"source","kind":"box","params":{"size":1.0}},
        {"op":"node.create","id":"camera","kind":"camera"},
        {"op":"node.create","id":"driver","kind":"empty"},
        {"op":"tracking.clip_create","id":"clip","name":"Clip"},
        {"op":"tracking.track_add","id":"clip","track":"track","name":"Track","frame":1,"co":[0.5,0.5]},
        {"op":"action.create","id":"drive","name":"Drive"}
    ]));
    let output = apply(
        &scene,
        1,
        &json!([
            {"op":"constraint.create","target":{"id":"owner"},"id":"attribute","type":"geometry_attribute","constraint_target":"source","params":{"target":"source","attribute_name":"position","data_type":"VECTOR","domain":"POINT","sample_index":0,"mix_mode":"REPLACE","apply_target_transform":false,"mix_loc":true,"mix_rot":false,"mix_scl":false}},
            {"op":"constraint.create","target":{"id":"owner"},"id":"follow","type":"follow_track","constraint_target":"camera","params":{"clip":"clip","track":"Track","object":"","camera":"camera","frame_method":"CROP","use_3d_position":false,"use_undistorted_position":false}},
            {"op":"constraint.create","target":{"id":"camera"},"id":"camera_solver","type":"camera_solver","params":{"clip":"clip","use_active_clip":false}},
            {"op":"constraint.create","target":{"id":"owner"},"id":"action","type":"action","constraint_target":"driver","params":{"target":"driver","action":"drive","transform_channel":"SCALE_Z","target_space":"LOCAL_OWNER_ORIENT","mix_mode":"AFTER_SPLIT","min":-1000.0,"max":1000.0,"frame_start":-1_048_574,"frame_end":1_048_574,"eval_time":1.0,"use_eval_time":false,"use_bone_object_action":false}}
        ]),
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let document: SceneDoc =
        serde_json::from_slice(&fs::read(scene.join("scene.json")).unwrap()).unwrap();
    let owner_constraints = &document.nodes[&Id::new("owner").unwrap()].constraints;
    assert_eq!(owner_constraints.len(), 3);
    assert_eq!(owner_constraints[0].params["data_type"], "VECTOR");
    assert_eq!(owner_constraints[1].params["frame_method"], "CROP");
    assert_eq!(owner_constraints[2].params["frame_start"], -1_048_574);
    assert_eq!(owner_constraints[2].params["frame_end"], 1_048_574);
    assert_eq!(
        document.nodes[&Id::new("camera").unwrap()].constraints[0].constraint_type,
        potter_core::model::ConstraintType::CameraSolver
    );
}

#[test]
fn attribute_modifier_rejections_include_json_pointers_and_features() {
    let cases = [
        invalid_case(
            "malformed vertex-weight curve",
            json!([{"op":"node.create","id":"body","kind":"box","params":{"size":1.0}}]),
            json!({"op":"modifier.create","target":{"id":"body"},"id":"weights","type":"vertex_weight_edit","params":{"vertex_group":"weights","map_curve":[[-0.1,0.0],[1.0,1.0]]}}),
            "INVALID_OPERATION",
            Some("/operations/0/params/map_curve"),
            Some("modifier.vertex_weight_edit"),
        ),
        invalid_case(
            "missing required vertex group",
            json!([{"op":"node.create","id":"body","kind":"box","params":{"size":1.0}}]),
            json!({"op":"modifier.create","target":{"id":"body"},"id":"weights","type":"vertex_weight_edit","params":{}}),
            "INVALID_OPERATION",
            Some("/operations/0/params/vertex_group"),
            Some("modifier.vertex_weight_edit"),
        ),
        invalid_case(
            "invalid weighted-normal mode",
            json!([{"op":"node.create","id":"body","kind":"box","params":{"size":1.0}}]),
            json!({"op":"modifier.create","target":{"id":"body"},"id":"normal","type":"weighted_normal","params":{"mode":"UNIFORM"}}),
            "INVALID_ARGUMENT",
            Some("/operations/0/params/mode"),
            None,
        ),
        invalid_case(
            "modifier update reports its parameter pointer",
            json!([
                {"op":"node.create","id":"body","kind":"box","params":{"size":1.0}},
                {"op":"modifier.create","target":{"id":"body"},"id":"weights","type":"vertex_weight_edit","params":{"vertex_group":"weights"}}
            ]),
            json!({"op":"modifier.update","target":{"id":"body"},"id":"weights","set":{"params":{"vertex_group":"weights","map_curve":[[-0.1,0.0],[1.0,1.0]]}}}),
            "INVALID_OPERATION",
            Some("/operations/0/set/params/map_curve"),
            Some("modifier.vertex_weight_edit"),
        ),
    ];
    run_invalid_cases(&cases);
}

#[test]
fn attribute_modifier_boundary_parameters_are_accepted() {
    let (_directory, scene) =
        setup_scene(&json!([{"op":"node.create","id":"body","kind":"box","params":{"size":1.0}}]));
    let output = apply(
        &scene,
        1,
        &json!([{"op":"modifier.create","target":{"id":"body"},"id":"weights","type":"vertex_weight_edit","params":{"vertex_group":"weights","falloff_type":"STEP","map_curve":[[0.0,0.0],[1.0,1.0]],"default_weight":0.0,"add_threshold":0.0,"remove_threshold":1.0,"use_add":false,"use_remove":true,"normalize":false}}]),
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let document: SceneDoc =
        serde_json::from_slice(&fs::read(scene.join("scene.json")).unwrap()).unwrap();
    let modifier = &document.nodes[&Id::new("body").unwrap()].modifiers[0];
    assert_eq!(modifier.params["falloff_type"], "STEP");
    assert_eq!(
        modifier.params["map_curve"],
        json!([[0.0, 0.0], [1.0, 1.0]])
    );
}

#[test]
fn physics_reference_errors_report_exact_codes_and_pointers() {
    let cases = [
        invalid_case(
            "object particles need an instance ID",
            json!([{"op":"node.create","id":"emitter","kind":"box","params":{"size":1.0}}]),
            json!({"op":"physics.particle_emitter.create","target":{"id":"emitter"},"settings":{"render_type":"OBJECT"}}),
            "INVALID_OPERATION",
            Some("/settings/instance_object"),
            None,
        ),
        invalid_case(
            "particle emitter cannot instance itself",
            json!([{"op":"node.create","id":"emitter","kind":"box","params":{"size":1.0}}]),
            json!({"op":"physics.particle_emitter.create","target":{"id":"emitter"},"settings":{"render_type":"OBJECT","instance_object":"emitter"}}),
            "INVALID_OPERATION",
            Some("/settings/instance_object"),
            None,
        ),
        invalid_case(
            "particle instance object is missing",
            json!([{"op":"node.create","id":"emitter","kind":"box","params":{"size":1.0}}]),
            json!({"op":"physics.particle_emitter.create","target":{"id":"emitter"},"settings":{"render_type":"OBJECT","instance_object":"absent"}}),
            "TARGET_NOT_FOUND",
            Some("/settings/instance_object"),
            None,
        ),
        invalid_case(
            "particle collection is missing",
            json!([{"op":"node.create","id":"emitter","kind":"box","params":{"size":1.0}}]),
            json!({"op":"physics.particle_emitter.create","target":{"id":"emitter"},"settings":{"render_type":"COLLECTION","instance_collection":"absent"}}),
            "TARGET_NOT_FOUND",
            Some("/settings/instance_collection"),
            None,
        ),
        invalid_case(
            "dynamic-paint target is not a brush",
            json!([
                {"op":"node.create","id":"canvas","kind":"grid","params":{"size":1.0,"x_subdivisions":1,"y_subdivisions":1}},
                {"op":"node.create","id":"brush","kind":"box","params":{"size":1.0}},
                {"op":"physics.dynamic_paint.create","target":{"id":"canvas"},"settings":{"role":"canvas"}}
            ]),
            json!({"op":"physics.dynamic_paint.update","target":{"id":"canvas"},"set":{"brushes":["brush"]}}),
            "INVALID_OPERATION",
            Some("/settings/brushes/0"),
            None,
        ),
        invalid_case(
            "dynamic-paint brush object is missing",
            json!([
                {"op":"node.create","id":"canvas","kind":"grid","params":{"size":1.0,"x_subdivisions":1,"y_subdivisions":1}},
                {"op":"physics.dynamic_paint.create","target":{"id":"canvas"},"settings":{"role":"canvas"}}
            ]),
            json!({"op":"physics.dynamic_paint.update","target":{"id":"canvas"},"set":{"brushes":["absent"]}}),
            "TARGET_NOT_FOUND",
            Some("/settings/brushes/0"),
            None,
        ),
    ];
    run_invalid_cases(&cases);
}

#[test]
fn physics_references_accept_existing_instance_and_brush_objects() {
    let (_directory, scene) = setup_scene(&json!([
        {"op":"node.create","id":"emitter","kind":"box","params":{"size":1.0}},
        {"op":"node.create","id":"instance","kind":"box","params":{"size":0.5}},
        {"op":"node.create","id":"brush","kind":"box","params":{"size":0.25}},
        {"op":"node.create","id":"canvas","kind":"grid","params":{"size":1.0,"x_subdivisions":2,"y_subdivisions":2}}
    ]));
    let output = apply(
        &scene,
        1,
        &json!([
            {"op":"physics.particle_emitter.create","target":{"id":"emitter"},"settings":{"render_type":"OBJECT","instance_object":"instance"}},
            {"op":"physics.dynamic_paint.create","target":{"id":"brush"},"settings":{"role":"brush"}},
            {"op":"physics.dynamic_paint.create","target":{"id":"canvas"},"settings":{"role":"canvas","brushes":["brush"]}}
        ]),
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let document: SceneDoc =
        serde_json::from_slice(&fs::read(scene.join("scene.json")).unwrap()).unwrap();
    assert_eq!(
        document.nodes[&Id::new("emitter").unwrap()].properties["physics_particle_emitter"]["instance_object"],
        "instance"
    );
    assert_eq!(
        document.nodes[&Id::new("canvas").unwrap()].properties["physics_dynamic_paint"]["brushes"],
        json!(["brush"])
    );
}

fn tracking_setup() -> Value {
    json!([
        {"op":"tracking.clip_create","id":"clip","name":"Clip","width":32,"height":32},
        {"op":"tracking.track_add","id":"clip","track":"track","name":"Track","frame":1,"co":[16.0,16.0]}
    ])
}

#[test]
fn klt_configuration_rejections_point_to_the_bad_field() {
    let cases = [
        invalid_case(
            "KLT config must be an object",
            tracking_setup(),
            json!({"op":"tracking.track","id":"clip","track":"track","initial_frame":1,"frames":[],"config":4}),
            "INVALID_OPERATION",
            Some("/operations/0/config"),
            None,
        ),
        invalid_case(
            "unknown KLT config key",
            tracking_setup(),
            json!({"op":"tracking.track","id":"clip","track":"track","initial_frame":1,"frames":[],"config":{"unknown":1}}),
            "INVALID_OPERATION",
            Some("/operations/0/config/unknown"),
            None,
        ),
        invalid_case(
            "KLT integer field has the wrong type",
            tracking_setup(),
            json!({"op":"tracking.track","id":"clip","track":"track","initial_frame":1,"frames":[],"config":{"patch_radius":2.5}}),
            "INVALID_OPERATION",
            Some("/operations/0/config/patch_radius"),
            None,
        ),
    ];
    run_invalid_cases(&cases);
}

#[expect(
    clippy::cast_precision_loss,
    reason = "fixture pixel coordinates are bounded to a deterministic 32-by-32 image"
)]
fn textured_frame(width: usize, height: usize) -> Value {
    let pixels = (0..height)
        .flat_map(|y| {
            (0..width).map(move |x| ((x * 37 + y * 61 + x * y * 13) % 251) as f64 / 250.0)
        })
        .collect::<Vec<_>>();
    json!({"width":width,"height":height,"pixels":pixels})
}

#[test]
fn klt_configuration_accepts_supported_lower_bound_values() {
    let (_directory, scene) = setup_scene(&tracking_setup());
    let frame = textured_frame(32, 32);
    let output = apply(
        &scene,
        1,
        &json!([{"op":"tracking.track","id":"clip","track":"track","initial_frame":1,"frames":[frame.clone(),frame],"config":{"patch_radius":2,"max_iterations":1,"max_pyramid_levels":1,"convergence_threshold":1.0e-12,"min_eigenvalue":1.0e-12}}]),
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let document: SceneDoc =
        serde_json::from_slice(&fs::read(scene.join("scene.json")).unwrap()).unwrap();
    let markers = &document.movie_clips[&Id::new("clip").unwrap()]
        .tracking
        .tracks[0]
        .markers;
    assert_eq!(markers.len(), 2);
    assert!((markers[0].frame - 1.0).abs() <= f64::EPSILON);
    assert!((markers[1].frame - 2.0).abs() <= f64::EPSILON);
    assert!(
        markers
            .iter()
            .all(|marker| marker.co.iter().all(|coordinate| coordinate.is_finite()))
    );
}

fn write_solve_object_fixture(scene: &Path, include_camera: bool) -> [String; 4] {
    let scene_file = scene.join("scene.json");
    let mut document: SceneDoc = serde_json::from_slice(&fs::read(&scene_file).unwrap()).unwrap();
    let focal_pixels = 50.0 / 36.0 * 100.0;
    let camera_model = CameraModel {
        matrix: [
            [focal_pixels, 0.0, 50.0, 0.0],
            [0.0, focal_pixels, 50.0, 0.0],
            [0.0, 0.0, 1.0, 8.0],
        ],
    };
    let object_pose = DMat4::from_scale_rotation_translation(
        DVec3::splat(1.0),
        DQuat::from_euler(glam::EulerRot::XYZ, 0.06, -0.1, 0.18),
        DVec3::new(0.35, -0.25, 0.4),
    );
    let points = [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -0.5],
        [-1.0, 1.0, 0.2],
        [1.0, 1.0, 0.8],
    ];
    let track_ids = std::array::from_fn(|index| format!("track_{index}"));
    let tracks = track_ids
        .iter()
        .zip(points)
        .map(|(track_id, point)| TrackingTrack {
            id: track_id.clone(),
            name: track_id.clone(),
            markers: vec![TrackingMarker {
                frame: 1.0,
                co: camera_model
                    .project(
                        object_pose
                            .transform_point3(DVec3::from_array(point))
                            .to_array(),
                    )
                    .unwrap(),
                ..TrackingMarker::default()
            }],
        })
        .collect();
    let cameras = include_camera
        .then_some(SolvedCamera {
            frame: 1,
            matrix: camera_model.matrix,
            average_error: 0.0,
            matrix_is_camera_to_world: false,
        })
        .into_iter()
        .collect();
    let reconstructed_points = track_ids
        .iter()
        .zip(points)
        .map(|(track, co)| ReconstructedPoint {
            track: track.clone(),
            co,
        })
        .collect();
    document.movie_clips.insert(
        Id::new("clip").unwrap(),
        MovieClip {
            name: "Synthetic clip".to_owned(),
            width: 100,
            height: 100,
            tracking: MovieTracking {
                tracks,
                camera: CameraIntrinsics::default(),
                reconstruction: Reconstruction {
                    cameras,
                    points: reconstructed_points,
                    is_valid: true,
                    average_error: 0.0,
                },
                ..MovieTracking::default()
            },
            ..MovieClip::default()
        },
    );
    fs::write(&scene_file, serde_json::to_vec(&document).unwrap()).unwrap();
    track_ids
}

#[test]
fn solve_object_rejections_cover_cardinality_uniqueness_and_missing_reconstruction() {
    let cases = [
        invalid_case(
            "fewer than four object tracks",
            json!([{"op":"tracking.clip_create","id":"clip","name":"Clip"}]),
            json!({"op":"tracking.solve_object","id":"clip","object":"moving","tracks":["a","b","c"]}),
            "INVALID_OPERATION",
            Some("/operations/0/tracks"),
            None,
        ),
        invalid_case(
            "duplicate object track IDs",
            json!([{"op":"tracking.clip_create","id":"clip","name":"Clip"}]),
            json!({"op":"tracking.solve_object","id":"clip","object":"moving","tracks":["a","b","c","a"]}),
            "INVALID_OPERATION",
            Some("/operations/0/tracks"),
            None,
        ),
        invalid_case(
            "object solve track is not reconstructed",
            json!([{"op":"tracking.clip_create","id":"clip","name":"Clip"}]),
            json!({"op":"tracking.solve_object","id":"clip","object":"moving","tracks":["a","b","c","d"]}),
            "TARGET_NOT_FOUND",
            Some("/operations/0/tracks"),
            None,
        ),
    ];
    run_invalid_cases(&cases);

    let (_directory, scene) = setup_scene(&json!([]));
    let tracks = write_solve_object_fixture(&scene, false);
    let output = apply(
        &scene,
        0,
        &json!([{"op":"tracking.solve_object","id":"clip","object":"moving","tracks":tracks}]),
    );
    assert_invalid(
        &output,
        "object solve without camera poses",
        "INVALID_OPERATION",
        Some("/operations/0/id"),
        None,
    );
}

#[test]
fn solve_object_accepts_the_four_track_boundary_and_persists_a_valid_solve() {
    let (_directory, scene) = setup_scene(&json!([]));
    let tracks = write_solve_object_fixture(&scene, true);
    let nodes = apply(
        &scene,
        0,
        &json!([
            {"op":"node.create","id":"owner","kind":"empty"},
            {"op":"node.create","id":"camera","kind":"camera"}
        ]),
    );
    assert!(
        nodes.status.success(),
        "{}",
        String::from_utf8_lossy(&nodes.stdout)
    );
    let output = apply(
        &scene,
        1,
        &json!([{"op":"tracking.solve_object","id":"clip","object":"moving","tracks":tracks}]),
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let constraint = apply(
        &scene,
        2,
        &json!([{"op":"constraint.create","target":{"id":"owner"},"id":"object_solver","type":"object_solver","constraint_target":"camera","params":{"clip":"clip","object":"moving","camera":"camera","set_inverse_pending":false}}]),
    );
    assert!(
        constraint.status.success(),
        "{}",
        String::from_utf8_lossy(&constraint.stdout)
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let document: SceneDoc =
        serde_json::from_slice(&fs::read(scene.join("scene.json")).unwrap()).unwrap();
    let solved = &document.movie_clips[&Id::new("clip").unwrap()]
        .tracking
        .objects[0];
    assert_eq!(solved.tracks.len(), 4);
    assert!(solved.reconstruction_is_valid);
    assert_eq!(solved.reconstruction.len(), 1);
    assert_eq!(
        document.nodes[&Id::new("owner").unwrap()].constraints[0].params["object"],
        "moving"
    );
    assert!(
        solved.reconstruction[0]
            .matrix
            .iter()
            .all(|value| value.is_finite())
    );
}

#[test]
fn modifier_apply_reports_shape_key_and_unsupported_failures() {
    let cases = [
        invalid_case(
            "shape keys guard modifier application",
            json!([
                {"op":"node.create","id":"subject","kind":"plane","params":{"size":2.0}},
                {"op":"shape_key.create","target":{"id":"subject"},"id":"smile","name":"Smile","positions":{"0":[1.0,-1.0,0.25]}},
                {"op":"modifier.create","target":{"id":"subject"},"id":"wave","type":"wave","params":{"height":0.5}}
            ]),
            json!({"op":"modifier.apply","target":{"id":"subject"},"modifier_id":"wave"}),
            "UNSUPPORTED_FEATURE",
            None,
            Some("modifier.apply.shape_key_guard"),
        ),
        invalid_case(
            "skin modifier branch hull is unsupported",
            json!([
                {"op":"node.create","id":"skeleton","kind":"box","params":{"size":1.0}},
                {"op":"modifier.create","target":{"id":"skeleton"},"id":"skin","type":"skin","params":{}}
            ]),
            json!({"op":"modifier.apply","target":{"id":"skeleton"},"modifier_id":"skin"}),
            "UNSUPPORTED_FEATURE",
            None,
            Some("modifier.skin.branch_hull"),
        ),
    ];
    run_invalid_cases(&cases);
}

#[test]
fn modifier_apply_commits_topology_changes_and_removes_the_applied_modifier() {
    let (_directory, scene) = setup_scene(&json!([
        {"op":"node.create","id":"body","kind":"plane","params":{"size":2.0}},
        {"op":"modifier.create","target":{"id":"body"},"id":"triangulate","type":"triangulate","params":{}}
    ]));
    let output = apply(
        &scene,
        1,
        &json!([{"op":"modifier.apply","target":{"id":"body"},"modifier_id":"triangulate"}]),
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let document: SceneDoc =
        serde_json::from_slice(&fs::read(scene.join("scene.json")).unwrap()).unwrap();
    let node = &document.nodes[&Id::new("body").unwrap()];
    assert!(node.modifiers.is_empty(), "{:?}", node.modifiers);
    let data_id = node.data.as_ref().unwrap();
    let mesh = document.data_blocks[data_id].mesh.as_ref().unwrap();
    assert_eq!(mesh.faces.len(), 2);
    assert!(mesh.faces.iter().all(|face| face.vertices.len() == 3));
}
