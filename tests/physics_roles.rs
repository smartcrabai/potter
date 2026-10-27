#![expect(
    clippy::unwrap_used,
    reason = "focused physics role regressions use deterministic CLI fixtures"
)]

use std::{fs, path::Path, process::Output};

use assert_cmd::Command;
use serde_json::{Value, json};
use tempfile::tempdir;

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
    let batch = json!({
        "schema_version": 1,
        "base_revision": base_revision,
        "operations": operations,
    });
    let schema = potter_core::schema::schema("operations", None).unwrap();
    let validator = jsonschema::validator_for(&schema).unwrap();
    assert!(
        validator.is_valid(&batch),
        "invalid test operation batch: {batch}"
    );
    let directory = tempdir().unwrap();
    let path = directory.path().join("operations.json");
    fs::write(&path, serde_json::to_vec(&batch).unwrap()).unwrap();
    pot()
        .arg("apply")
        .arg(scene)
        .arg("--file")
        .arg(path)
        .arg("--json")
        .output()
        .unwrap()
}

fn apply_ok(scene: &Path, base_revision: u64, operations: &Value) {
    let output = apply(scene, base_revision, operations);
    assert!(
        output.status.success(),
        "pot apply failed: {}",
        String::from_utf8_lossy(&output.stdout)
    );
}

fn bake(scene: &Path, output_dir: &Path, frames: &str, last_frame_index: usize) -> Value {
    let output = pot()
        .arg("bake")
        .arg(scene)
        .arg("--kind")
        .arg("simulation")
        .arg("--frames")
        .arg(frames)
        .arg("--out")
        .arg(output_dir)
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "pot bake failed: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    let path = output_dir.join(format!("frame_{last_frame_index:08}.json"));
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

fn scene_document(scene: &Path) -> Value {
    serde_json::from_slice(&fs::read(scene.join("scene.json")).unwrap()).unwrap()
}

fn raw_mesh(document: &Value, node_id: &str) -> Value {
    let data_id = document["nodes"][node_id]["data"].as_str().unwrap();
    document["data_blocks"][data_id]["mesh"].clone()
}

fn assert_mesh_matches(actual: &Value, expected: &Value, context: &str) {
    assert_eq!(
        actual["vertices"], expected["vertices"],
        "{context} vertices"
    );
    assert_eq!(actual["faces"], expected["faces"], "{context} faces");
}

fn mesh_counts(mesh: &Value) -> (usize, usize) {
    (
        mesh["vertices"].as_array().unwrap().len(),
        mesh["faces"].as_array().unwrap().len(),
    )
}

#[test]
fn dynamic_paint_role_and_ui_type_stay_consistent_on_every_write_path() {
    let directory = tempdir().unwrap();
    let scene = directory.path().join("dynamic-paint-roles");
    init(&scene);
    apply_ok(
        &scene,
        0,
        &json!([
            {"op":"node.create","id":"brush","kind":"box","params":{"size":0.1}},
            {"op":"node.create","id":"canvas","kind":"grid","params":{"size":1.0,"x_subdivisions":2,"y_subdivisions":2}},
            {"op":"node.create","id":"direct","kind":"box","params":{"size":0.5}},
            {"op":"physics.dynamic_paint.create","target":{"id":"brush"},"settings":{"role":"brush"}},
            {"op":"physics.dynamic_paint.create","target":{"id":"canvas"},"settings":{"role":"canvas","brushes":["brush"],"surface_format":"weight"}},
            {"op":"modifier.create","target":{"id":"direct"},"id":"direct_dp","type":"dynamic_paint","params":{"ui_type":"BRUSH"}}
        ]),
    );
    let document = scene_document(&scene);
    for (node_id, expected_role, expected_ui_type) in [
        ("brush", "brush", "BRUSH"),
        ("canvas", "canvas", "CANVAS"),
        ("direct", "brush", "BRUSH"),
    ] {
        let node = &document["nodes"][node_id];
        assert_eq!(
            node["properties"]["physics_dynamic_paint"]["role"],
            expected_role
        );
        let modifier = node["modifiers"].as_array().unwrap().first().unwrap();
        assert_eq!(modifier["params"]["ui_type"], expected_ui_type);
    }

    apply_ok(
        &scene,
        1,
        &json!([{
            "op":"physics.dynamic_paint.update",
            "target":{"id":"canvas"},
            "set":{"strength":0.5}
        }]),
    );
    let document = scene_document(&scene);
    assert_eq!(
        document["nodes"]["canvas"]["properties"]["physics_dynamic_paint"]["brushes"],
        json!(["brush"])
    );
    assert_eq!(
        document["nodes"]["canvas"]["properties"]["physics_dynamic_paint"]["surface_format"],
        "weight"
    );

    apply_ok(
        &scene,
        2,
        &json!([{
            "op":"physics.dynamic_paint.update",
            "target":{"id":"canvas"},
            "set":{"role":"brush"}
        }]),
    );
    let document = scene_document(&scene);
    assert_eq!(
        document["nodes"]["canvas"]["properties"]["physics_dynamic_paint"]["role"],
        "brush"
    );
    assert!(
        document["nodes"]["canvas"]["properties"]["physics_dynamic_paint"]
            .get("brushes")
            .is_none()
    );
    assert!(
        document["nodes"]["canvas"]["properties"]["physics_dynamic_paint"]
            .get("surface_format")
            .is_none()
    );
    assert_eq!(
        document["nodes"]["canvas"]["modifiers"][0]["params"]["ui_type"],
        "BRUSH"
    );

    apply_ok(
        &scene,
        3,
        &json!([{
            "op":"modifier.update",
            "target":{"id":"canvas"},
            "modifier_id":"physics_dynamic_paint",
            "set":{"params":{"ui_type":"CANVAS"}}
        }]),
    );
    let document = scene_document(&scene);
    assert_eq!(
        document["nodes"]["canvas"]["properties"]["physics_dynamic_paint"]["role"],
        "canvas"
    );
    assert_eq!(
        document["nodes"]["canvas"]["modifiers"][0]["params"]["ui_type"],
        "CANVAS"
    );
    assert!(
        document["nodes"]["canvas"]["properties"]["physics_dynamic_paint"]
            .get("brushes")
            .is_none()
    );
    assert!(
        document["nodes"]["canvas"]["properties"]["physics_dynamic_paint"]
            .get("surface_format")
            .is_none()
    );
}

#[test]
fn legacy_dynamic_paint_ui_mismatch_migrates_canvas_radius_without_rewriting_ui_type() {
    let directory = tempdir().unwrap();
    let scene = directory.path().join("legacy-dynamic-paint");
    init(&scene);
    apply_ok(
        &scene,
        0,
        &json!([
            {"op":"node.create","id":"canvas","kind":"grid","params":{"size":1.0,"x_subdivisions":2,"y_subdivisions":2}},
            {"op":"node.create","id":"brush","kind":"box","params":{"size":0.1}},
            {"op":"physics.dynamic_paint.create","target":{"id":"canvas"},"settings":{"role":"canvas"}},
            {"op":"physics.dynamic_paint.create","target":{"id":"brush"},"settings":{"role":"brush","paint_source":"POINT"}}
        ]),
    );
    let mut document = scene_document(&scene);
    document["nodes"]["canvas"]["modifiers"][0]["params"]["ui_type"] = json!("BRUSH");
    document["nodes"]["brush"]["modifiers"][0]["params"]["ui_type"] = json!("CANVAS");
    document["schema_version"] = json!(9);
    let brush_settings = document["nodes"]["brush"]["properties"]["physics_dynamic_paint"]
        .as_object_mut()
        .unwrap();
    let _ = brush_settings.remove("paint_source");
    fs::write(
        scene.join("scene.json"),
        serde_json::to_vec(&document).unwrap(),
    )
    .unwrap();

    let frame = bake(
        &scene,
        &directory.path().join("legacy-dynamic-paint-bake"),
        "1:1",
        0,
    );
    assert!(frame["paint_colors"]["canvas"]["0"].is_array());
    assert!(frame["paint_colors"].get("brush").is_none());
    assert!(
        frame["paint_colors"]["canvas"]["0"]
            .as_array()
            .unwrap()
            .iter()
            .any(|color| color[3].as_f64().unwrap() > 0.0)
    );
    apply_ok(
        &scene,
        1,
        &json!([{"op":"node.create","id":"migration_commit","kind":"empty","params":{}}]),
    );
    let after = scene_document(&scene);
    assert_eq!(after["schema_version"], 12);
    assert_eq!(
        after["nodes"]["brush"]["properties"]["physics_dynamic_paint"]["paint_source"],
        "POINT"
    );
    assert!(
        after["nodes"]["brush"]["properties"]["physics_dynamic_paint"]
            .get("radius")
            .is_none()
    );
    assert_eq!(
        after["nodes"]["canvas"]["properties"]["physics_dynamic_paint"]["role"],
        "canvas"
    );
    assert_eq!(
        after["nodes"]["canvas"]["modifiers"][0]["params"]["ui_type"],
        "BRUSH"
    );
    assert_eq!(
        after["nodes"]["brush"]["properties"]["physics_dynamic_paint"]["role"],
        "brush"
    );
    assert_eq!(
        after["nodes"]["brush"]["modifiers"][0]["params"]["ui_type"],
        "CANVAS"
    );
}
#[test]
fn volume_dynamic_paint_rejects_native_radius_in_operations_and_loaded_settings() {
    let directory = tempdir().unwrap();
    let operation_scene = directory.path().join("dynamic-paint-radius-operation");
    init(&operation_scene);
    apply_ok(
        &operation_scene,
        0,
        &json!([{"op":"node.create","id":"brush","kind":"box","params":{"size":1.0}}]),
    );
    let rejected = apply(
        &operation_scene,
        1,
        &json!([{
            "op":"physics.dynamic_paint.create",
            "target":{"id":"brush"},
            "settings":{"role":"brush","paint_source":"VOLUME","radius":2.0}
        }]),
    );
    assert!(!rejected.status.success());
    assert_eq!(response(&rejected)["error"]["code"], "UNSUPPORTED_FEATURE");
    assert_eq!(
        response(&rejected)["error"]["details"]["feature_id"],
        "physics.dynamic_paint_brush.radius"
    );

    let stored_scene = directory.path().join("dynamic-paint-radius-stored");
    init(&stored_scene);
    apply_ok(
        &stored_scene,
        0,
        &json!([
            {"op":"node.create","id":"canvas","kind":"grid","params":{"size":1.0,"x_subdivisions":2,"y_subdivisions":2}},
            {"op":"node.create","id":"brush","kind":"box","params":{"size":1.0}},
            {"op":"physics.dynamic_paint.create","target":{"id":"brush"},"settings":{"role":"brush","paint_source":"VOLUME"}},
            {"op":"physics.dynamic_paint.create","target":{"id":"canvas"},"settings":{"role":"canvas","brushes":["brush"],"radius":2.0}}
        ]),
    );
    let mut document = scene_document(&stored_scene);
    document["nodes"]["brush"]["properties"]["physics_dynamic_paint"]["radius"] = json!(2.0);
    fs::write(
        stored_scene.join("scene.json"),
        serde_json::to_vec(&document).unwrap(),
    )
    .unwrap();
    let output = pot()
        .arg("bake")
        .arg(&stored_scene)
        .arg("--kind")
        .arg("simulation")
        .arg("--frames")
        .arg("1:1")
        .arg("--out")
        .arg(directory.path().join("dynamic-paint-radius-bake"))
        .arg("--json")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert_eq!(response(&output)["error"]["code"], "UNSUPPORTED_FEATURE");
    assert_eq!(
        response(&output)["error"]["details"]["feature_id"],
        "physics.dynamic_paint_brush.radius"
    );
}

#[test]
fn volume_dynamic_paint_without_canvas_surface_fails_bake_with_unsupported_feature() {
    let directory = tempdir().unwrap();
    let scene = directory
        .path()
        .join("dynamic-paint-volume-without-surface");
    init(&scene);
    apply_ok(
        &scene,
        0,
        &json!([
            {"op":"node.create","id":"canvas","kind":"grid","params":{"size":1.0,"x_subdivisions":2,"y_subdivisions":2}},
            {"op":"node.create","id":"brush","kind":"box","params":{"size":1.0}},
            {"op":"physics.dynamic_paint.create","target":{"id":"brush"},"settings":{"role":"brush","paint_source":"VOLUME"}},
            {"op":"physics.dynamic_paint.create","target":{"id":"canvas"},"settings":{"role":"canvas","brushes":["brush"]}}
        ]),
    );
    let output = pot()
        .arg("bake")
        .arg(&scene)
        .arg("--kind")
        .arg("simulation")
        .arg("--frames")
        .arg("1:1")
        .arg("--out")
        .arg(
            directory
                .path()
                .join("dynamic-paint-volume-without-surface-bake"),
        )
        .arg("--json")
        .output()
        .unwrap();
    assert!(!output.status.success());
    let result = response(&output);
    assert_eq!(result["error"]["code"], "UNSUPPORTED_FEATURE");
    assert_eq!(
        result["error"]["details"]["feature_id"],
        "physics.dynamic_paint_canvas.volume_without_surface"
    );
    let mut document = scene_document(&scene);
    document["nodes"]["brush"]["properties"]["physics_dynamic_paint"]["radius"] = json!(2.0);
    fs::write(
        scene.join("scene.json"),
        serde_json::to_vec(&document).unwrap(),
    )
    .unwrap();
    let output = pot()
        .arg("bake")
        .arg(&scene)
        .arg("--kind")
        .arg("simulation")
        .arg("--frames")
        .arg("1:1")
        .arg("--out")
        .arg(
            directory
                .path()
                .join("dynamic-paint-volume-with-native-radius-bake"),
        )
        .arg("--json")
        .output()
        .unwrap();
    assert!(!output.status.success());
    let result = response(&output);
    assert_eq!(result["error"]["code"], "UNSUPPORTED_FEATURE");
    assert_eq!(
        result["error"]["details"]["feature_id"],
        "physics.dynamic_paint_brush.radius"
    );
}

#[test]
fn parented_legacy_brush_loads_then_refuses_bake_until_source_is_fixed() {
    let directory = tempdir().unwrap();
    let scene = directory.path().join("parented-legacy-dynamic-paint");
    init(&scene);
    apply_ok(
        &scene,
        0,
        &json!([
            {"op":"node.create","id":"parent","kind":"empty","params":{},"transform":{"translation":[3.0,0.0,0.0]}},
            {"op":"node.create","id":"brush","kind":"box","params":{"size":0.1},"parent":"parent"},
            {"op":"node.create","id":"canvas","kind":"grid","params":{"size":1.0,"x_subdivisions":2,"y_subdivisions":2}},
            {"op":"physics.dynamic_paint.create","target":{"id":"brush"},"settings":{"role":"brush","paint_source":"POINT"}}
        ]),
    );
    let mut document = scene_document(&scene);
    document["schema_version"] = json!(9);
    let brush_settings = document["nodes"]["brush"]["properties"]["physics_dynamic_paint"]
        .as_object_mut()
        .unwrap();
    let _ = brush_settings.remove("paint_source");
    let _ = brush_settings.insert("radius".to_owned(), json!(1.0));
    fs::write(
        scene.join("scene.json"),
        serde_json::to_vec(&document).unwrap(),
    )
    .unwrap();

    let inspection = pot()
        .arg("inspect")
        .arg(&scene)
        .arg("--id")
        .arg("brush")
        .arg("--frame")
        .arg("1")
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        inspection.status.success(),
        "inspect should open the legacy project: {}",
        String::from_utf8_lossy(&inspection.stdout)
    );

    apply_ok(
        &scene,
        1,
        &json!([{
            "op":"physics.dynamic_paint.create",
            "target":{"id":"canvas"},
            "settings":{"role":"canvas","brushes":["brush"]}
        }]),
    );
    let rejected = pot()
        .arg("bake")
        .arg(&scene)
        .arg("--kind")
        .arg("simulation")
        .arg("--frames")
        .arg("1:1")
        .arg("--out")
        .arg(directory.path().join("parented-legacy-dynamic-paint-bake"))
        .arg("--json")
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    assert_eq!(response(&rejected)["error"]["code"], "UNSUPPORTED_FEATURE");
    assert_eq!(
        response(&rejected)["error"]["details"]["feature_id"],
        "physics.dynamic_paint_brush.legacy_transform"
    );

    apply_ok(
        &scene,
        2,
        &json!([{
            "op":"physics.dynamic_paint.update",
            "target":{"id":"brush"},
            "set":{"paint_source":"POINT"}
        }]),
    );
    let frame = bake(
        &scene,
        &directory.path().join("fixed-parented-dynamic-paint-bake"),
        "1:1",
        0,
    );
    assert!(frame["paint_colors"]["canvas"]["0"].is_array());
}

#[test]
fn none_and_domain_without_flows_preserve_their_input_meshes() {
    let directory = tempdir().unwrap();
    let scene = directory.path().join("fluid-no-flow");
    init(&scene);
    apply_ok(
        &scene,
        0,
        &json!([
            {"op":"node.create","id":"domain","kind":"box","params":{"size":4.0}},
            {"op":"node.create","id":"none","kind":"box","params":{"size":1.0}},
            {"op":"physics.fluid.create","target":{"id":"domain"},"settings":{"domain_type":"LIQUID","resolution":4}},
            {"op":"modifier.create","target":{"id":"none"},"id":"none_fluid","type":"fluid"}
        ]),
    );
    let before = scene_document(&scene);
    let frame = bake(
        &scene,
        &directory.path().join("fluid-no-flow-bake"),
        "1:1",
        0,
    );
    // Blender 5.2.2 leaves this 4-unit domain at its input 8 vertices and 6 faces without a FLOW.
    for node_id in ["domain", "none"] {
        assert_mesh_matches(
            &frame["deformed_meshes"][node_id],
            &raw_mesh(&before, node_id),
            node_id,
        );
    }
    assert!(frame["fluid_particles"].get("domain").is_none());
    assert!(frame["fluid_particles"].get("none").is_none());
    assert_eq!(
        before["nodes"]["none"]["modifiers"][0]["params"]["fluid_type"],
        "NONE"
    );
}

/// Effector height for the flow scene: its top (z = 0.95) lies below the flow's bottom face (z = 1.0),
/// so no particle starts inside it, and the falling particles reach it within the baked frames, so
/// Potter's collision decides the domain output.
const EFFECTOR_BELOW_FLOW_Z: f64 = 0.8;
/// Effector height that overlaps the flow's bottom face (z 0.8 to 1.1). Blender's oracle for this
/// layout is the known gap: the liquid spawned inside the effector is slowed by Blender and not pushed out.
const EFFECTOR_OVERLAPPING_FLOW_Z: f64 = 0.95;

fn setup_flow_scene(scene: &Path, with_effector: bool) {
    setup_flow_scene_at(scene, with_effector.then_some(EFFECTOR_BELOW_FLOW_Z));
}

fn setup_flow_scene_at(scene: &Path, effector_z: Option<f64>) {
    init(scene);
    let mut operations = vec![
        json!({"op":"node.create","id":"domain","kind":"box","params":{"size":4.0}}),
        json!({"op":"node.create","id":"flow","kind":"box","params":{"size":1.0},"transform":{"translation":[0.0,0.0,1.5]}}),
        json!({"op":"node.create","id":"none","kind":"box","params":{"size":0.5}}),
        json!({"op":"physics.fluid.create","target":{"id":"domain"},"settings":{"domain_type":"LIQUID","resolution":4}}),
        json!({"op":"physics.fluid.create","target":{"id":"flow"},"settings":{}}),
        json!({"op":"modifier.update","target":{"id":"flow"},"modifier_id":"physics_fluid","set":{"params":{"fluid_type":"FLOW"}}}),
        json!({"op":"physics.fluid.update","target":{"id":"flow"},"set":{"inflow_rate":8.0}}),
        json!({"op":"modifier.create","target":{"id":"none"},"id":"none_fluid","type":"fluid"}),
    ];
    if let Some(effector_z) = effector_z {
        operations.push(json!({"op":"node.create","id":"effector","kind":"box","params":{"size":0.3},"transform":{"translation":[0.0,0.0,effector_z]}}));
        operations
            .push(json!({"op":"physics.fluid.create","target":{"id":"effector"},"settings":{}}));
        operations.push(json!({"op":"modifier.update","target":{"id":"effector"},"modifier_id":"physics_fluid","set":{"params":{"fluid_type":"EFFECTOR"}}}));
    }
    apply_ok(scene, 0, &Value::Array(operations));
}

// Blender FlowSettings are imported directly, outside the native physics operation schema.
fn set_imported_flow_settings(scene: &Path, flow_type: &str, flow_behavior: &str, density: f64) {
    let mut document = scene_document(scene);
    let settings = document["nodes"]["flow"]["properties"]["physics_fluid"]
        .as_object_mut()
        .unwrap();
    settings.remove("inflow_rate");
    settings.insert("flow_type".to_owned(), json!(flow_type));
    settings.insert("flow_behavior".to_owned(), json!(flow_behavior));
    settings.insert("density".to_owned(), json!(density));
    fs::write(
        scene.join("scene.json"),
        serde_json::to_vec(&document).unwrap(),
    )
    .unwrap();
}

#[test]
fn flow_emits_into_domains_and_effectors_only_affect_domain_output() {
    let directory = tempdir().unwrap();
    let without_effector = directory.path().join("fluid-with-flow");
    let with_effector = directory.path().join("fluid-with-flow-and-effector");
    setup_flow_scene(&without_effector, false);
    setup_flow_scene(&with_effector, true);
    let plain_frame = bake(
        &without_effector,
        &directory.path().join("plain-fluid-bake"),
        "1:6",
        5,
    );
    let blocked_frame = bake(
        &with_effector,
        &directory.path().join("effector-fluid-bake"),
        "1:6",
        5,
    );

    // The effector sits below the flow here, so Potter's own collision acts on the falling liquid.
    // Blender 5.2.2's overlapping layout (effector at z=0.95) is a known gap, see
    // `flow_spawning_inside_an_effector_reports_a_known_gap`. Potter compares role behavior and
    // relative native output, not Mantaflow's tessellation.
    assert_ne!(
        plain_frame["fluid_particles"]["domain"].as_array().unwrap(),
        &Vec::<Value>::new()
    );
    assert_ne!(
        mesh_counts(&plain_frame["deformed_meshes"]["domain"]),
        (8, 6)
    );
    assert_mesh_matches(
        &plain_frame["deformed_meshes"]["flow"],
        &raw_mesh(&scene_document(&without_effector), "flow"),
        "FLOW own mesh",
    );
    assert_mesh_matches(
        &plain_frame["deformed_meshes"]["none"],
        &raw_mesh(&scene_document(&without_effector), "none"),
        "NONE own mesh",
    );
    assert!(plain_frame["fluid_particles"].get("flow").is_none());
    assert!(plain_frame["fluid_particles"].get("none").is_none());

    assert_mesh_matches(
        &blocked_frame["deformed_meshes"]["effector"],
        &raw_mesh(&scene_document(&with_effector), "effector"),
        "EFFECTOR own mesh",
    );
    let plain_particles = &plain_frame["fluid_particles"]["domain"];
    let blocked_particles = &blocked_frame["fluid_particles"]["domain"];
    assert_ne!(plain_particles, blocked_particles);
}

/// Blender 5.2.2 keeps liquid spawned inside an effector in the effector's column and slows it there
/// (measured with this layout: the column's lowest liquid descends about 0.012 per frame inside the
/// effector against about 0.075 without it, and leaves through the bottom face around frame 8). It
/// does not push the liquid out. Potter has no grid obstacle rule, so the liquid is not slowed; the
/// loss is reported on the domain instead of hidden.
#[test]
fn flow_spawning_inside_an_effector_reports_a_known_gap() {
    let directory = tempdir().unwrap();
    let overlapping = directory.path().join("fluid-overlapping-effector");
    setup_flow_scene_at(&overlapping, Some(EFFECTOR_OVERLAPPING_FLOW_Z));
    let output = pot()
        .arg("inspect")
        .arg(&overlapping)
        .arg("--id")
        .arg("domain")
        .arg("--frame")
        .arg("3")
        .arg("--json")
        .output()
        .unwrap();
    assert!(output.status.success(), "pot inspect failed");
    let envelope = response(&output);
    let warnings = envelope["warnings"].as_array().unwrap();
    assert!(
        warnings.iter().any(|warning| {
            warning["details"]["feature_id"] == "physics.fluid_domain.emitter_inside_effector"
                && warning["details"]["node_id"] == "domain"
        }),
        "{envelope}"
    );

    let separate = directory.path().join("fluid-effector-below-flow");
    setup_flow_scene(&separate, true);
    let output = pot()
        .arg("inspect")
        .arg(&separate)
        .arg("--id")
        .arg("domain")
        .arg("--frame")
        .arg("3")
        .arg("--json")
        .output()
        .unwrap();
    assert!(output.status.success(), "pot inspect failed");
    let envelope = response(&output);
    assert!(
        envelope["warnings"]
            .as_array()
            .unwrap()
            .iter()
            .all(|warning| warning["details"]["feature_id"]
                != "physics.fluid_domain.emitter_inside_effector"),
        "{envelope}"
    );
}

#[test]
fn imported_liquid_flow_emits_only_at_a_native_rate_and_other_modes_stay_unsupported() {
    let directory = tempdir().unwrap();
    let liquid = directory.path().join("imported-liquid-flow");
    setup_flow_scene(&liquid, false);
    set_imported_flow_settings(&liquid, "LIQUID", "INFLOW", 8.0);
    let document = scene_document(&liquid);
    let flow_settings = &document["nodes"]["flow"]["properties"]["physics_fluid"];
    assert!(flow_settings.get("inflow_rate").is_none());
    assert_eq!(flow_settings["flow_type"], "LIQUID");
    assert_eq!(flow_settings["flow_behavior"], "INFLOW");
    assert_eq!(flow_settings["density"], 8.0);

    // An imported LIQUID flow emits only at an explicit Potter inflow_rate; without one the bake is refused (see the mode loop).
    let native = directory.path().join("native-flow-rate");
    setup_flow_scene(&native, false);
    // Emission starts on the second frame: the first frame has no particles yet.
    let frame = bake(
        &native,
        &directory.path().join("native-flow-rate-bake"),
        "1:2",
        1,
    );
    assert_ne!(
        frame["fluid_particles"]["domain"].as_array().unwrap(),
        &Vec::<Value>::new()
    );

    let native_override = directory.path().join("imported-flow-native-rate");
    setup_flow_scene(&native_override, false);
    set_imported_flow_settings(&native_override, "LIQUID", "INFLOW", 1.0);
    let mut native_override_document = scene_document(&native_override);
    native_override_document["nodes"]["flow"]["properties"]["physics_fluid"]["inflow_rate"] =
        json!(8.0);
    fs::write(
        native_override.join("scene.json"),
        serde_json::to_vec(&native_override_document).unwrap(),
    )
    .unwrap();
    let native_override_frame = bake(
        &native_override,
        &directory.path().join("imported-flow-native-rate-bake"),
        "1:2",
        1,
    );
    assert_eq!(
        native_override_frame["fluid_particles"]["domain"],
        frame["fluid_particles"]["domain"]
    );

    for (flow_type, flow_behavior, feature_id) in [
        ("SMOKE", "INFLOW", "physics.fluid.smoke_fire"),
        ("LIQUID", "GEOMETRY", "physics.fluid.flow_behavior"),
        ("LIQUID", "INFLOW", "physics.fluid.emission"),
    ] {
        let scene = directory
            .path()
            .join(format!("unsupported-{flow_type}-{flow_behavior}"));
        setup_flow_scene(&scene, false);
        set_imported_flow_settings(&scene, flow_type, flow_behavior, 1.0);
        let output = pot()
            .arg("bake")
            .arg(&scene)
            .arg("--kind")
            .arg("simulation")
            .arg("--frames")
            .arg("1:1")
            .arg("--out")
            .arg(
                directory
                    .path()
                    .join(format!("{flow_type}-{flow_behavior}-bake")),
            )
            .arg("--json")
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert_eq!(
            response(&output)["error"]["details"]["feature_id"],
            feature_id
        );
    }
}

fn assert_operation_error(output: &Output, pointer: &str) {
    assert!(!output.status.success());
    let envelope = response(output);
    assert_eq!(envelope["error"]["code"], "INVALID_OPERATION");
    assert_eq!(envelope["error"]["details"]["pointer"], pointer);
}

#[test]
fn fluid_settings_follow_modifier_role_and_role_changes_drop_inapplicable_values() {
    let directory = tempdir().unwrap();
    let scene = directory.path().join("fluid-role-validation");
    setup_flow_scene(&scene, true);

    assert_operation_error(
        &apply(
            &scene,
            1,
            &json!([{"op":"physics.fluid.update","target":{"id":"flow"},"set":{"resolution":8}}]),
        ),
        "/operations/0/settings/resolution",
    );
    assert_operation_error(
        &apply(
            &scene,
            1,
            &json!([{"op":"physics.fluid.update","target":{"id":"effector"},"set":{"resolution":8}}]),
        ),
        "/operations/0/settings/resolution",
    );

    apply_ok(
        &scene,
        1,
        &json!([{
            "op":"modifier.update",
            "target":{"id":"flow"},
            "modifier_id":"physics_fluid",
            "set":{"params":{"fluid_type":"DOMAIN"}}
        }]),
    );
    let document = scene_document(&scene);
    assert!(
        document["nodes"]["flow"]["properties"]["physics_fluid"]
            .get("inflow_rate")
            .is_none()
    );
    assert_eq!(
        document["nodes"]["flow"]["modifiers"][0]["params"]["fluid_type"],
        "DOMAIN"
    );

    assert_operation_error(
        &apply(
            &scene,
            2,
            &json!([{"op":"physics.fluid.update","target":{"id":"flow"},"set":{"inflow_rate":8.0}}]),
        ),
        "/operations/0/settings/inflow_rate",
    );
}

#[test]
fn volume_modifier_object_types_reject_create_and_set_operations() {
    let directory = tempdir().unwrap();

    let mesh = directory.path().join("mesh-to-volume-create");
    init(&mesh);
    apply_ok(
        &mesh,
        0,
        &json!([{"op":"node.create","id":"mesh","kind":"box","params":{"size":1.0}}]),
    );
    assert_operation_error(
        &apply(
            &mesh,
            1,
            &json!([{"op":"modifier.create","target":{"id":"mesh"},"id":"bad","type":"mesh_to_volume"}]),
        ),
        "/operations/0/target",
    );
    assert_operation_error(
        &apply(
            &mesh,
            1,
            &json!([{"op":"modifier.create","target":{"id":"mesh"},"id":"bad_displace","type":"volume_displace"}]),
        ),
        "/operations/0/target",
    );
    let missing = apply(
        &mesh,
        1,
        &json!([{"op":"modifier.create","target":{"id":"missing"},"id":"missing_modifier","type":"mesh_to_volume"}]),
    );
    assert!(!missing.status.success());
    assert_eq!(response(&missing)["error"]["code"], "TARGET_NOT_FOUND");

    let set_mesh_type = directory.path().join("mesh-to-volume-set");
    init(&set_mesh_type);
    apply_ok(
        &set_mesh_type,
        0,
        &json!([
            {"op":"node.create","id":"mesh","kind":"box","params":{"size":1.0}},
            {"op":"modifier.create","target":{"id":"mesh"},"id":"base","type":"solidify"}
        ]),
    );
    assert_operation_error(
        &apply(
            &set_mesh_type,
            1,
            &json!([{"op":"modifier.update","target":{"id":"mesh"},"modifier_id":"base","set":{"type":"mesh_to_volume"}}]),
        ),
        "/operations/0/target",
    );

    let volume = directory.path().join("volume-to-mesh-create");
    init(&volume);
    apply_ok(
        &volume,
        0,
        &json!([{"op":"volume.create","id":"volume","grids":[{"dims":[1,1,1],"voxel_size":1.0,"origin":[0.0,0.0,0.0],"values":[0.0]}]}]),
    );
    assert_operation_error(
        &apply(
            &volume,
            1,
            &json!([{"op":"modifier.create","target":{"id":"volume"},"id":"bad","type":"volume_to_mesh"}]),
        ),
        "/operations/0/target",
    );

    let set_type = directory.path().join("volume-to-mesh-set");
    init(&set_type);
    apply_ok(
        &set_type,
        0,
        &json!([
            {"op":"volume.create","id":"volume","grids":[{"dims":[1,1,1],"voxel_size":1.0,"origin":[0.0,0.0,0.0],"values":[0.0]}]},
            {"op":"modifier.create","target":{"id":"volume"},"id":"displace","type":"volume_displace"}
        ]),
    );
    assert_operation_error(
        &apply(
            &set_type,
            1,
            &json!([{"op":"modifier.update","target":{"id":"volume"},"modifier_id":"displace","set":{"type":"volume_to_mesh"}}]),
        ),
        "/operations/0/target",
    );
}

#[test]
fn volume_displace_object_mapping_without_object_falls_back_to_local() {
    let directory = tempdir().unwrap();
    let scene = directory.path().join("volume-displace-object-fallback");
    init(&scene);
    apply_ok(
        &scene,
        0,
        &json!([
            {"op":"node.create","id":"mesh","kind":"box","params":{"size":1.0}},
            {"op":"volume.create","id":"volume","grids":[{"dims":[1,1,1],"voxel_size":1.0,"origin":[0.0,0.0,0.0],"values":[0.0]}]},
            {"op":"modifier.create","target":{"id":"volume"},"id":"to_volume","type":"mesh_to_volume","params":{"object":"mesh","voxel_amount":8}},
            {"op":"modifier.create","target":{"id":"volume"},"id":"displace","type":"volume_displace","params":{"texture_map_mode":"OBJECT","texture_map_object":null}}
        ]),
    );
    let output = pot()
        .arg("inspect")
        .arg(&scene)
        .arg("--id")
        .arg("volume")
        .arg("--frame")
        .arg("1")
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "Volume Displace OBJECT mapping without an object should use LOCAL coordinates: {}",
        String::from_utf8_lossy(&output.stdout)
    );
}

#[test]
fn legacy_mesh_volume_modifiers_fail_evaluation_with_object_type_errors() {
    let directory = tempdir().unwrap();
    for modifier_type in ["mesh_to_volume", "volume_displace"] {
        let scene = directory.path().join(modifier_type);
        init(&scene);
        apply_ok(
            &scene,
            0,
            &json!([
                {"op":"node.create","id":"mesh","kind":"box","params":{"size":1.0}},
                {"op":"modifier.create","target":{"id":"mesh"},"id":"legacy","type":"solidify"}
            ]),
        );
        let mut document = scene_document(&scene);
        document["nodes"]["mesh"]["modifiers"][0]["type"] = json!(modifier_type);
        document["nodes"]["mesh"]["modifiers"][0]["params"] = json!({});
        fs::write(
            scene.join("scene.json"),
            serde_json::to_vec(&document).unwrap(),
        )
        .unwrap();

        let output = pot()
            .arg("inspect")
            .arg(&scene)
            .arg("--id")
            .arg("mesh")
            .arg("--frame")
            .arg("1")
            .arg("--json")
            .output()
            .unwrap();
        assert!(
            !output.status.success(),
            "{modifier_type} unexpectedly evaluated"
        );
        let envelope = response(&output);
        assert_eq!(envelope["error"]["code"], "INVALID_OPERATION");
        assert_eq!(envelope["error"]["details"]["object_type"], "mesh");
        assert!(
            envelope["error"]["message"]
                .as_str()
                .unwrap()
                .contains("requires a volume Object")
        );
    }
}
