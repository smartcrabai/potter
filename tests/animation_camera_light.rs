use std::{error::Error, fs, path::Path, process::Command};

use serde_json::{Value, json};
use tempfile::tempdir;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn output_json(output: &std::process::Output) -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn init(scene: &Path) -> Result<(), Box<dyn Error>> {
    let output = pot().arg("init").arg(scene).arg("--json").output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    Ok(())
}

fn apply(scene: &Path, batch: &Value) -> Result<Value, Box<dyn Error>> {
    let batch_file = scene.with_file_name("animation-batch.json");
    fs::write(&batch_file, serde_json::to_vec(batch)?)?;
    let output = pot()
        .arg("apply")
        .arg(scene)
        .arg("--file")
        .arg(batch_file)
        .arg("--json")
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    output_json(&output)
}

fn inspect(scene: &Path, frame: &str) -> Result<Value, Box<dyn Error>> {
    let output = pot()
        .arg("inspect")
        .arg(scene)
        .arg("--frame")
        .arg(frame)
        .arg("--json")
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    output_json(&output)
}

#[test]
fn fractional_frame_animation_camera_assignment_and_light_data_are_inspectable()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init(&scene)?;

    let result = apply(
        &scene,
        &json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [
                {"op":"node.create", "id":"animated", "kind":"empty"},
                {"op":"action.create", "id":"move_action", "name":"Move"},
                {"op":"node.update", "target":{"id":"animated"}, "set":{"action":"move_action"}},
                {"op":"keyframe.insert", "target":{"id":"animated"}, "path":"transform.translation", "index":0, "frame":1.0, "value":0.0, "interpolation":"linear"},
                {"op":"keyframe.insert", "target":{"id":"animated"}, "path":"transform.translation", "index":0, "frame":24.0, "value":10.0, "interpolation":"linear"},
                {"op":"camera.create", "id":"main_camera", "name":"Main Camera", "lens_mm":35.0},
                {"op":"scene.update", "target":{"id":"scene_main"}, "set":{"camera":"main_camera"}},
                {"op":"camera.update", "target":{"id":"main_camera"}, "set":{"projection":"orthographic"}},
                {"op":"light.create", "id":"key_light", "name":"Key Light", "light_type":"point"},
                {"op":"light.update", "target":{"id":"key_light"}, "set":{"energy":240.0}},
                {"op":"world.create", "id":"world_main", "color":[0.1, 0.2, 0.3], "strength":0.5},
                {"op":"world.update", "target":{"id":"world_main"}, "set":{"strength":1.5}},
                {"op":"scene.update", "target":{"id":"scene_main"}, "set":{"world":"world_main"}}
            ]
        }),
    )?;
    assert_eq!(result["result"]["committed"], true);

    let inspected = inspect(&scene, "12.5")?;
    let items = inspected["result"]["items"]
        .as_array()
        .ok_or("items missing")?;
    let animated = items
        .iter()
        .find(|item| item["id"] == "animated")
        .ok_or("animated node missing")?;
    assert_eq!(animated["transform"]["evaluated"]["translation"][0], 5.0);
    let camera = items
        .iter()
        .find(|item| item["id"] == "main_camera")
        .ok_or("camera missing")?;
    assert_eq!(camera["camera"]["lens_mm"], 35.0);
    assert_eq!(camera["camera"]["projection"], "orthographic");
    let active_scene = pot()
        .arg("inspect")
        .arg(&scene)
        .arg("--id")
        .arg("scene_main")
        .arg("--json")
        .output()?;
    assert!(active_scene.status.success());
    assert_eq!(
        output_json(&active_scene)?["result"]["items"][0]["camera"],
        "main_camera"
    );
    assert_eq!(
        output_json(&active_scene)?["result"]["items"][0]["world"],
        "world_main"
    );
    let world = items
        .iter()
        .find(|item| item["id"] == "world_main")
        .ok_or("world missing")?;
    assert_eq!(world["color"], json!([0.1, 0.2, 0.3]));
    assert_eq!(world["strength"], 1.5);
    let light = items
        .iter()
        .find(|item| item["id"] == "key_light")
        .ok_or("light missing")?;
    assert_eq!(light["light"]["energy"], 240.0);
    Ok(())
}

#[test]
fn a_single_quaternion_channel_uses_unkeyed_static_components() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init(&scene)?;
    apply(
        &scene,
        &json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [
                {"op":"node.create", "id":"animated", "kind":"empty",
                    "transform":{"rotation":[0.0, 0.6, 0.0, 0.8]}},
                {"op":"action.create", "id":"turn_action", "name":"Turn"},
                {"op":"node.update", "target":{"id":"animated"}, "set":{"action":"turn_action"}},
                {"op":"keyframe.insert", "target":{"id":"animated"},
                    "path":"transform.rotation_quaternion", "index":3,
                    "frame":1.0, "value":0.3, "interpolation":"linear"}
            ]
        }),
    )?;

    let inspected = inspect(&scene, "1.0")?;
    let items = inspected["result"]["items"]
        .as_array()
        .ok_or("items missing")?;
    let rotation = items
        .iter()
        .find(|item| item["id"] == "animated")
        .ok_or("animated node missing")?["transform"]["evaluated"]["rotation"]
        .as_array()
        .ok_or("evaluated rotation missing")?;
    let norm = 0.6_f64.hypot(0.3).hypot(0.8);
    let expected = [0.0, 0.6 / norm, 0.3 / norm, 0.8 / norm];
    assert_eq!(rotation.len(), expected.len());
    for (actual, expected) in rotation.iter().zip(expected) {
        let actual = actual.as_f64().ok_or("rotation component missing")?;
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "expected {expected}, got {actual}"
        );
    }
    Ok(())
}
