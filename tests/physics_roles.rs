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
    assert!(validator.is_valid(&batch), "invalid test operation batch: {batch}");
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
    assert_eq!(actual["vertices"], expected["vertices"], "{context} vertices");
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
            {"op":"physics.dynamic_paint.create","target":{"id":"canvas"},"settings":{"role":"canvas"}},
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
        assert_eq!(node["properties"]["physics_dynamic_paint"]["role"], expected_role);
        let modifier = node["modifiers"].as_array().unwrap().first().unwrap();
        assert_eq!(modifier["params"]["ui_type"], expected_ui_type);
    }

    apply_ok(
        &scene,
        1,
        &json!([{
            "op":"physics.dynamic_paint.update",
            "target":{"id":"canvas"},
            "set":{"role":"brush"}
        }]),
    );
    let document = scene_document(&scene);
    assert_eq!(document["nodes"]["canvas"]["properties"]["physics_dynamic_paint"]["role"], "brush");
    assert_eq!(document["nodes"]["canvas"]["modifiers"][0]["params"]["ui_type"], "BRUSH");

    apply_ok(
        &scene,
        2,
        &json!([{
            "op":"modifier.update",
            "target":{"id":"canvas"},
            "modifier_id":"physics_dynamic_paint",
            "set":{"params":{"ui_type":"CANVAS"}}
        }]),
    );
    let document = scene_document(&scene);
    assert_eq!(document["nodes"]["canvas"]["properties"]["physics_dynamic_paint"]["role"], "canvas");
    assert_eq!(document["nodes"]["canvas"]["modifiers"][0]["params"]["ui_type"], "CANVAS");
}

#[test]
fn legacy_dynamic_paint_ui_mismatch_evaluates_from_stored_role_without_rewriting() {
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
            {"op":"physics.dynamic_paint.create","target":{"id":"brush"},"settings":{"role":"brush"}}
        ]),
    );
    let mut document = scene_document(&scene);
    document["nodes"]["canvas"]["modifiers"][0]["params"]["ui_type"] = json!("BRUSH");
    document["nodes"]["brush"]["modifiers"][0]["params"]["ui_type"] = json!("CANVAS");
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
    assert!(frame["paint_colors"]["canvas"].is_array());
    assert!(frame["paint_colors"].get("brush").is_none());
    assert!(
        frame["paint_colors"]["canvas"]
            .as_array()
            .unwrap()
            .iter()
            .any(|color| color[3].as_f64().unwrap() > 0.0)
    );
    let after = scene_document(&scene);
    assert_eq!(after["nodes"]["canvas"]["properties"]["physics_dynamic_paint"]["role"], "canvas");
    assert_eq!(after["nodes"]["canvas"]["modifiers"][0]["params"]["ui_type"], "BRUSH");
    assert_eq!(after["nodes"]["brush"]["properties"]["physics_dynamic_paint"]["role"], "brush");
    assert_eq!(after["nodes"]["brush"]["modifiers"][0]["params"]["ui_type"], "CANVAS");
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
            {"op":"physics.fluid.create","target":{"id":"domain"},"settings":{"type":"liquid","resolution":4}},
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

fn setup_flow_scene(scene: &Path, with_effector: bool) {
    init(scene);
    let mut operations = vec![
        json!({"op":"node.create","id":"domain","kind":"box","params":{"size":4.0}}),
        json!({"op":"node.create","id":"flow","kind":"box","params":{"size":1.0},"transform":{"translation":[0.0,0.0,1.5]}}),
        json!({"op":"node.create","id":"none","kind":"box","params":{"size":0.5}}),
        json!({"op":"physics.fluid.create","target":{"id":"domain"},"settings":{"type":"liquid","resolution":4}}),
        json!({"op":"physics.fluid.create","target":{"id":"flow"},"settings":{"type":"liquid"}}),
        json!({"op":"modifier.update","target":{"id":"flow"},"modifier_id":"physics_fluid","set":{"params":{"fluid_type":"FLOW"}}}),
        json!({"op":"physics.fluid.update","target":{"id":"flow"},"set":{"inflow_rate":8.0}}),
        json!({"op":"modifier.create","target":{"id":"none"},"id":"none_fluid","type":"fluid"}),
    ];
    if with_effector {
        operations.push(json!({"op":"node.create","id":"effector","kind":"box","params":{"size":0.3},"transform":{"translation":[0.0,0.0,0.95]}}));
        operations.push(json!({"op":"physics.fluid.create","target":{"id":"effector"},"settings":{"type":"liquid"}}));
        operations.push(json!({"op":"modifier.update","target":{"id":"effector"},"modifier_id":"physics_fluid","set":{"params":{"fluid_type":"EFFECTOR"}}}));
    }
    apply_ok(scene, 0, &Value::Array(operations));
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

    // Blender 5.2.2 gives the size-1 FLOW at z=1.5 a 432V/860F domain surface at frame 1;
    // the size-0.3 effector at z=0.95 changes that oracle to 434V/864F. Potter compares role
    // behavior and relative native output, not Mantaflow's tessellation.
    assert!(!plain_frame["fluid_particles"]["domain"].as_array().unwrap().is_empty());
    assert_ne!(mesh_counts(&plain_frame["deformed_meshes"]["domain"]), (8, 6));
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
        document["nodes"]["flow"]["properties"]["physics_fluid"]["type"],
        "liquid"
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
        &json!([{"op":"node.create","id":"volume","kind":"volume"}]),
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
            {"op":"node.create","id":"volume","kind":"volume"},
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
        assert!(!output.status.success(), "{modifier_type} unexpectedly evaluated");
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
