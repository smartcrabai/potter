use std::{error::Error, fs, process::Command};

use serde_json::json;
use tempfile::tempdir;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

/// A scene with one panorama camera, set as the scene camera.
fn panorama_scene(scene: &std::path::Path) -> Result<(), Box<dyn Error>> {
    let init = pot().args(["init"]).arg(scene).arg("--json").output()?;
    assert!(
        init.status.success(),
        "{}",
        String::from_utf8_lossy(&init.stdout)
    );
    let batch = scene
        .parent()
        .ok_or("scene directory has no parent")?
        .join("operations.json");
    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [
                {"op": "camera.create", "id": "camera_main",
                 "transform": {"translation": [0.0, -4.0, 4.0], "rotation": [0.0, 0.0, 0.0, 1.0]},
                 "projection": "panorama", "panorama_type": "equirectangular"},
                {"op": "scene.update", "target": {"id": "scene_main"}, "set": {"camera": "camera_main"}}
            ]
        }))?,
    )?;
    let applied = pot()
        .args(["apply"])
        .arg(scene)
        .args(["--file"])
        .arg(&batch)
        .arg("--json")
        .output()?;
    assert!(
        applied.status.success(),
        "{}",
        String::from_utf8_lossy(&applied.stdout)
    );
    Ok(())
}

#[test]
fn gltf_export_of_a_panorama_camera_reports_its_projection_as_lost() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    panorama_scene(&scene)?;
    let out = directory.path().join("scene.glb");

    // glTF writes a panorama camera as a perspective camera, so the strict export refuses and names the loss.
    let strict = pot()
        .args(["export"])
        .arg(&scene)
        .args(["--format", "glb", "--out"])
        .arg(&out)
        .arg("--json")
        .output()?;
    assert!(
        !strict.status.success(),
        "a panorama camera must not export to glb silently"
    );
    let report = String::from_utf8_lossy(&strict.stdout).into_owned();
    assert!(
        report.contains("camera.panorama"),
        "the strict export must name the loss: {report}"
    );

    // With --allow-lossy the export proceeds and records the loss.
    let lossy = pot()
        .args(["export"])
        .arg(&scene)
        .args(["--format", "glb", "--out"])
        .arg(&out)
        .args(["--allow-lossy", "--json"])
        .output()?;
    assert!(
        lossy.status.success(),
        "{}",
        String::from_utf8_lossy(&lossy.stdout)
    );
    let response: serde_json::Value = serde_json::from_slice(&lossy.stdout)?;
    let losses = response["result"]["losses"]
        .as_array()
        .ok_or("export losses are missing")?;
    assert!(
        losses
            .iter()
            .any(|loss| loss["feature_id"] == "camera.panorama"),
        "the lossy export must record the panorama loss: {losses:?}"
    );
    Ok(())
}
