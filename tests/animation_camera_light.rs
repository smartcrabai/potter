use std::{error::Error, fs, path::Path, process::Command};

use serde_json::{Value, json};
use tempfile::tempdir;
#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/process.rs"]
mod process;

use blender_checked::blender_executable;

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

fn scene_revision(scene: &Path) -> Result<u64, Box<dyn Error>> {
    let document: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    document["revision"]
        .as_u64()
        .ok_or_else(|| "scene revision is missing".into())
}

fn inspected_light(inspected: &Value) -> Result<&Value, Box<dyn Error>> {
    inspected["result"]["items"]
        .as_array()
        .ok_or("inspect items are missing")?
        .iter()
        .find(|item| item.get("light").is_some_and(Value::is_object))
        .ok_or_else(|| "light item is missing".into())
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
                {"op":"light.create", "id":"key_light", "name":"Key Light", "light_type":"point", "use_shadow":false},
                {"op":"light.update", "target":{"id":"key_light"}, "set":{"energy":240.0, "use_shadow":true}},
                {"op":"light.create", "id":"fill_light", "name":"Fill Light", "use_shadow":true},
                {"op":"light.create", "id":"created_shadowless", "name":"Created Shadowless", "use_shadow":false},
                {"op":"light.create", "id":"updated_shadowless", "name":"Updated Shadowless"},
                {"op":"light.update", "target":{"id":"updated_shadowless"}, "set":{"use_shadow":false}},
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
    assert_eq!(light["light"]["use_shadow"], true);
    let fill_light = items
        .iter()
        .find(|item| item["id"] == "fill_light")
        .ok_or("fill light missing")?;
    assert_eq!(fill_light["light"]["use_shadow"], true);
    let created_shadowless = items
        .iter()
        .find(|item| item["id"] == "created_shadowless")
        .ok_or("created shadowless light missing")?;
    assert_eq!(created_shadowless["light"]["use_shadow"], false);
    let updated_shadowless = items
        .iter()
        .find(|item| item["id"] == "updated_shadowless")
        .ok_or("updated shadowless light missing")?;
    assert_eq!(updated_shadowless["light"]["use_shadow"], false);
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
    let norm = 0.6_f64.hypot(0.3);
    let expected = [0.0, 0.6 / norm, 0.0, 0.3 / norm];
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

fn assert_rotation(rotation: &Value, expected: [f64; 4]) -> Result<(), Box<dyn Error>> {
    let rotation = rotation.as_array().ok_or("evaluated rotation missing")?;
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

#[test]
fn default_node_euler_action_is_visible_through_inspect() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init(&scene)?;
    apply(
        &scene,
        &json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [
                {"op":"node.create", "id":"animated", "kind":"empty"},
                {"op":"action.create", "id":"turn_action", "name":"Turn"},
                {"op":"node.update", "target":{"id":"animated"}, "set":{"action":"turn_action"}},
                {"op":"keyframe.insert", "target":{"id":"animated"}, "path":"transform.rotation_euler", "index":2, "frame":1.0, "value":0.0},
                {"op":"keyframe.insert", "target":{"id":"animated"}, "path":"transform.rotation_euler", "index":2, "frame":3.0, "value":std::f64::consts::FRAC_PI_2}
            ]
        }),
    )?;

    let inspected = inspect(&scene, "3.0")?;
    let animated = inspected["result"]["items"]
        .as_array()
        .ok_or("items missing")?
        .iter()
        .find(|item| item["id"] == "animated")
        .ok_or("animated node missing")?;
    assert_rotation(
        &animated["transform"]["evaluated"]["rotation"],
        [
            0.0,
            0.0,
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
        ],
    )
}

#[test]
fn axis_angle_animation_with_only_angle_keyed_uses_the_default_y_axis() -> Result<(), Box<dyn Error>>
{
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init(&scene)?;
    apply(
        &scene,
        &json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [
                {"op":"node.create", "id":"animated", "kind":"empty"},
                {"op":"action.create", "id":"turn_action", "name":"Turn"},
                {"op":"node.update", "target":{"id":"animated"}, "set":{"action":"turn_action"}},
                {"op":"keyframe.insert", "target":{"id":"animated"}, "path":"transform.rotation_axis_angle", "index":0, "frame":1.0, "value":0.0},
                {"op":"keyframe.insert", "target":{"id":"animated"}, "path":"transform.rotation_axis_angle", "index":0, "frame":3.0, "value":std::f64::consts::FRAC_PI_2}
            ]
        }),
    )?;

    let inspected = inspect(&scene, "3.0")?;
    let animated = inspected["result"]["items"]
        .as_array()
        .ok_or("items missing")?
        .iter()
        .find(|item| item["id"] == "animated")
        .ok_or("animated node missing")?;
    assert_rotation(
        &animated["transform"]["evaluated"]["rotation"],
        [
            0.0,
            std::f64::consts::FRAC_1_SQRT_2,
            0.0,
            std::f64::consts::FRAC_1_SQRT_2,
        ],
    )
}

#[test]
fn imported_light_shadow_survives_cli_apply_and_strict_blend_roundtrip()
-> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping light shadow Blender roundtrip: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let project = directory.path().join("project");
    init(&project)?;
    let source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/native_oracle/shader_bsdf_wave12/eevee_refraction_layers.blend");
    let imported_output = pot()
        .args(["import"])
        .arg(&project)
        .args(["--file"])
        .arg(&source)
        .args([
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "0",
            "--blender",
        ])
        .arg(&blender)
        .arg("--json")
        .output()?;
    assert!(
        imported_output.status.success(),
        "Blender source import failed:\n{}\n{}",
        String::from_utf8_lossy(&imported_output.stdout),
        String::from_utf8_lossy(&imported_output.stderr)
    );
    let imported = output_json(&imported_output)?;
    assert_eq!(imported["result"]["losses"], json!([]), "{imported}");
    let light_name = imported["result"]["id_mappings"]
        .as_object()
        .and_then(|mappings| mappings.keys().find_map(|key| key.strip_prefix("Light:")))
        .ok_or("imported Blender light mapping is missing")?
        .to_owned();
    let light_id = inspected_light(&inspect(&project, "1")?)?["id"]
        .as_str()
        .ok_or("imported light node ID is missing")?
        .to_owned();
    assert_eq!(
        inspected_light(&inspect(&project, "1")?)?["light"]["use_shadow"],
        json!(false)
    );

    for setting in [true, false] {
        let applied = apply(
            &project,
            &json!({
                "schema_version":1,
                "base_revision":scene_revision(&project)?,
                "operations":[{
                    "op":"light.update",
                    "target":{"id":light_id.clone()},
                    "set":{"use_shadow":setting}
                }]
            }),
        )?;
        assert_eq!(applied["result"]["committed"], true, "{applied}");
        let inspected = inspect(&project, "1")?;
        assert_eq!(
            inspected_light(&inspected)?["light"]["use_shadow"],
            json!(setting)
        );

        let exported_blend = directory
            .path()
            .join(format!("light-shadow-{setting}.blend"));
        let export_output = pot()
            .args(["export"])
            .arg(&project)
            .args(["--format", "blend", "--out"])
            .arg(&exported_blend)
            .args(["--blender"])
            .arg(&blender)
            .arg("--json")
            .output()?;
        assert!(
            export_output.status.success(),
            "strict Blender export failed:\n{}\n{}",
            String::from_utf8_lossy(&export_output.stdout),
            String::from_utf8_lossy(&export_output.stderr)
        );
        let exported = output_json(&export_output)?;
        assert_eq!(exported["result"]["losses"], json!([]), "{exported}");

        let python_bool = if setting { "True" } else { "False" };
        let expression = format!(
            "import bpy\n\
             if bpy.app.version != (5, 2, 2): raise RuntimeError('expected Blender 5.2.2')\n\
             light = bpy.data.lights.get({})\n\
             assert light is not None\n\
             assert light.use_shadow is {}\n",
            serde_json::to_string(&light_name)?,
            python_bool
        );
        let reopened = Command::new(&blender)
            .args(["--background", "--disable-autoexec"])
            .arg(&exported_blend)
            .args(["--python-exit-code", "1", "--python-expr"])
            .arg(expression)
            .output()?;
        assert!(
            reopened.status.success(),
            "Blender RNA readback failed:\n{}\n{}",
            String::from_utf8_lossy(&reopened.stdout),
            String::from_utf8_lossy(&reopened.stderr)
        );
    }
    Ok(())
}
