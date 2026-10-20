//! Realtime Beauty's ray-visibility warnings follow what EEVEE honours (`pot render --engine realtime`).
//!
//! Blender's panel shows each object type's flags for EEVEE (`scripts/startup/bl_ui/properties_object.py`,
//! `OBJECT_PT_visibility`): a mesh has Camera, Shadow and Raycast, and a light has Camera, Diffuse,
//! Glossy, Transmission and Volume Scatter. A light whose camera visibility is on shows its emitter in
//! EEVEE (`draw/engines/eevee/shaders/eevee_light_shape_display.bsl.hh`, which returns unless
//! `visible_camera`), and realtime Beauty draws no emitters. EEVEE applies a light's transmission and
//! volume-scatter visibility to its power (`draw/engines/eevee/eevee_light.cc`, `transmission_visibility`
//! and `volume_visibility`), and realtime Beauty does not. So a light's camera, transmission and
//! volume-scatter flags warn, while the mesh diffuse, glossy, transmission and volume flags, which EEVEE
//! ignores, do not.

#[path = "common/native_oracle.rs"]
#[expect(
    dead_code,
    reason = "each test binary uses a subset of the shared oracle helpers"
)]
mod native_oracle;
#[path = "common/process.rs"]
mod process;

use native_oracle::{TestResult, apply_operations, checked, pot, pot_json};
use serde_json::{Value, json};
use tempfile::tempdir;

/// The warnings a realtime Beauty render reports for `node_id` with `feature_id`.
fn warnings_for(render: &Value, node_id: &str) -> Vec<String> {
    render["result"]["warnings"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|warning| warning["node_id"] == node_id)
        .filter_map(|warning| warning["feature_id"].as_str().map(str::to_owned))
        .collect()
}

#[test]
fn realtime_beauty_warns_for_light_flags_it_does_not_honour_only() -> TestResult {
    let root = tempdir()?;
    let scene = root.path().join("scene");
    pot_json(|command| {
        command.arg("init").arg(&scene);
    })?;
    let applied = apply_operations(
        &scene,
        root.path(),
        json!([
            {"op": "node.create", "id": "plane", "kind": "plane", "params": {"size": 2.0}},
            {"op": "light.create", "id": "lamp", "light_type": "point", "energy": 100.0,
             "transform": {"translation": [0.0, 0.0, 3.0], "rotation": [0.0, 0.0, 0.0, 1.0]}},
            {"op": "camera.create", "id": "camera_top", "projection": "orthographic",
             "ortho_scale": 2.2, "transform": {"translation": [0.0, 0.0, 5.0],
             "rotation": [0.0, 0.0, 0.0, 1.0]}},
            {"op": "scene.update", "target": {"id": "scene_main"}, "set": {"camera": "camera_top"}},
            {"op": "render.update", "target": {"id": "scene_main"},
             "set": {"resolution_x": 32, "resolution_y": 32, "resolution_percentage": 100}},
            // Light flags EEVEE honours and realtime does not (camera: EEVEE draws the emitter;
            // transmission and volume scatter: EEVEE applies them to the light): they must warn.
            {"op": "node.update", "target": {"id": "lamp"},
             "set": {"ray_visibility": {"camera": true, "transmission": false, "volume_scatter": false}}},
            // Mesh flags EEVEE does not honour (diffuse, glossy): no warning. Shadow is honoured.
            {"op": "node.update", "target": {"id": "plane"},
             "set": {"ray_visibility": {"diffuse": false, "glossy": false, "shadow": false}}},
        ]),
    )?;
    assert_eq!(applied["ok"], json!(true), "scene apply failed: {applied}");

    let mut command = pot();
    command
        .arg("render")
        .arg(&scene)
        .args([
            "--camera",
            "camera_top",
            "--engine",
            "realtime",
            "--format",
            "png",
            "--out",
        ])
        .arg(root.path().join("realtime"))
        .arg("--json");
    let render: Value = serde_json::from_slice(&checked(&mut command, "realtime render")?.stdout)?;

    let lamp = warnings_for(&render, "lamp");
    for flag in ["camera", "transmission", "volume_scatter"] {
        let feature = format!("object.ray_visibility.{flag}");
        assert!(
            lamp.contains(&feature),
            "a light's {flag} flag must warn in realtime Beauty, got {lamp:?}"
        );
    }
    let plane = warnings_for(&render, "plane");
    assert!(
        plane.is_empty(),
        "EEVEE has no mesh diffuse or glossy flag and honours shadow, so the plane must not warn, got {plane:?}"
    );
    Ok(())
}
