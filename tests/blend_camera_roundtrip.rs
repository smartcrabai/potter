use std::{
    error::Error,
    fs,
    path::Path,
    process::{Command, Output},
};

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/process.rs"]
mod process;

use blender_checked::blender_executable;
use serde_json::{Value, json};
use tempfile::tempdir;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn run(command: &mut Command, operation: &str) -> Result<Output, Box<dyn Error>> {
    let output = command.output()?;
    if !output.status.success() {
        return Err(format!(
            "{operation} failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(output)
}

/// A scene whose only camera is `camera_main`, built with `camera_settings` (projection and its options).
fn build_scene(scene: &Path, camera_settings: &Value) -> Result<(), Box<dyn Error>> {
    run(
        pot().args(["init"]).arg(scene).arg("--json"),
        "initializing the scene",
    )?;
    let mut camera = json!({
        "op": "camera.create",
        "id": "camera_main",
        "transform": {"translation": [0.0, -4.0, 4.0], "rotation": [1.0, 0.0, 0.0, 0.0]},
    });
    if let (Some(object), Some(settings)) = (camera.as_object_mut(), camera_settings.as_object()) {
        object.extend(
            settings
                .iter()
                .map(|(key, value)| (key.clone(), value.clone())),
        );
    }
    let operations = [
        camera,
        json!({"op": "scene.update", "target": {"id": "scene_main"}, "set": {"camera": "camera_main"}}),
    ];
    let directory = scene.parent().ok_or("scene directory has no parent")?;
    let batch = directory.join("operations.json");
    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": operations,
        }))?,
    )?;
    run(
        pot()
            .args(["apply"])
            .arg(scene)
            .args(["--file"])
            .arg(&batch)
            .arg("--json"),
        "building the scene",
    )?;
    Ok(())
}

fn export_command(scene: &Path, blend: &Path, blender: &Path, extra: &[&str]) -> Command {
    let mut command = pot();
    command
        .args(["export"])
        .arg(scene)
        .args(["--format", "blend", "--out"])
        .arg(blend)
        .args(["--blender"])
        .arg(blender)
        .args(extra)
        .arg("--json");
    command
}

fn import_command(scene: &Path, blend: &Path, blender: &Path, extra: &[&str]) -> Command {
    let mut command = pot();
    command
        .args(["import"])
        .arg(scene)
        .args(["--file"])
        .arg(blend)
        .args([
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "1",
            "--blender",
        ])
        .arg(blender)
        .args(extra)
        .arg("--json");
    command
}

/// Runs `expression` in Blender on `blend`; a Python exception (a failed assertion) fails the check.
fn blender_python(blender: &Path, blend: &Path, expression: &str) -> Result<(), Box<dyn Error>> {
    let output = Command::new(blender)
        .args(["--background", "--disable-autoexec"])
        .arg(blend)
        .args(["--python-exit-code", "1", "--python-expr"])
        .arg(expression)
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "Blender check failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(())
}

fn camera_data(scene: &Path) -> Result<Value, Box<dyn Error>> {
    let document: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let blocks = document["data_blocks"]
        .as_object()
        .ok_or("data blocks are missing")?;
    blocks
        .values()
        .find(|block| block["type"] == "camera")
        .and_then(|block| block.get("camera").cloned())
        .ok_or_else(|| "the scene has no camera data block".into())
}

#[test]
fn panorama_camera_round_trips_as_a_blender_pano_camera() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping the panorama camera round trip; no executable Blender was found");
        return Ok(());
    };
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    build_scene(
        &scene,
        &json!({"projection": "panorama", "panorama_type": "equirectangular"}),
    )?;
    let blend = directory.path().join("panorama.blend");
    run(
        &mut export_command(&scene, &blend, &blender, &[]),
        "exporting the panorama camera",
    )?;
    blender_python(
        &blender,
        &blend,
        "import bpy; cams = [c for c in bpy.data.cameras if c.type == 'PANO' and c.panorama_type == 'EQUIRECTANGULAR']; assert len(cams) == 1, [(c.name, c.type) for c in bpy.data.cameras]",
    )?;

    let output = run(
        &mut import_command(&scene, &blend, &blender, &[]),
        "importing the panorama camera",
    )?;
    let imported: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(imported["result"]["losses"], json!([]));
    let camera = camera_data(&scene)?;
    assert_eq!(camera["projection"], "panorama");
    assert_eq!(camera["panorama_type"], "equirectangular");
    Ok(())
}

#[test]
fn panorama_ranges_round_trip_without_loss() -> Result<(), Box<dyn Error>> {
    let Some(camera) = panorama_round_trip(
        &json!({
            "projection": "panorama",
            "panorama_type": "equirectangular",
            "longitude_min": -1.2,
            "longitude_max": 2.0,
            "latitude_min": -0.7,
            "latitude_max": 1.1,
        }),
        "import bpy; cam = [c for c in bpy.data.cameras if c.type == 'PANO'][0]; assert cam.panorama_type == 'EQUIRECTANGULAR'; assert abs(cam.longitude_min + 1.2) < 1e-6 and abs(cam.longitude_max - 2.0) < 1e-6; assert abs(cam.latitude_min + 0.7) < 1e-6 and abs(cam.latitude_max - 1.1) < 1e-6",
    )?
    else {
        return Ok(());
    };
    assert_eq!(camera["panorama_type"], "equirectangular");
    assert_close_value(&camera["longitude_min"], -1.2);
    assert_close_value(&camera["longitude_max"], 2.0);
    assert_close_value(&camera["latitude_min"], -0.7);
    assert_close_value(&camera["latitude_max"], 1.1);
    Ok(())
}

#[test]
fn panorama_ranges_blender_cannot_hold_import_as_a_reported_loss() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping the panorama range import; no executable Blender was found");
        return Ok(());
    };
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    build_scene(
        &scene,
        &json!({"projection": "panorama", "panorama_type": "equirectangular"}),
    )?;
    let blend = directory.path().join("ranged.blend");
    run(
        &mut export_command(&scene, &blend, &blender, &[]),
        "exporting the panorama camera",
    )?;
    // Blender-side edit: a longitude span beyond a full turn, which Potter's panorama cannot hold.
    blender_python(
        &blender,
        &blend,
        "import bpy; cam = [c for c in bpy.data.cameras if c.type == 'PANO'][0]; cam.longitude_min = 0.0; cam.longitude_max = 7.0; bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)",
    )?;

    // Strict import refuses the loss and names it.
    let strict = import_command(&scene, &blend, &blender, &[]).output()?;
    assert!(
        !strict.status.success(),
        "an unrepresentable panorama range must not import silently"
    );
    let report = format!(
        "{}{}",
        String::from_utf8_lossy(&strict.stdout),
        String::from_utf8_lossy(&strict.stderr)
    );
    assert!(
        report.contains("blender.camera.panorama_range"),
        "the strict import must name the range loss: {report}"
    );

    // With --allow-lossy the import proceeds, keeps the panorama at the full sphere, and records the loss.
    let output = run(
        &mut import_command(&scene, &blend, &blender, &["--allow-lossy"]),
        "importing the ranged panorama with --allow-lossy",
    )?;
    let imported: Value = serde_json::from_slice(&output.stdout)?;
    let losses = imported["result"]["losses"]
        .as_array()
        .ok_or("import losses are missing")?;
    assert!(
        losses
            .iter()
            .any(|loss| loss["feature_id"] == "blender.camera.panorama_range"),
        "the lossy import must record the range loss: {losses:?}"
    );
    let camera = camera_data(&scene)?;
    assert_eq!(camera["projection"], "panorama");
    assert!(
        camera.get("longitude_max").is_none(),
        "the unrepresentable range imports at the default limits: {camera}"
    );
    Ok(())
}

#[test]
fn fisheye_equidistant_round_trips_as_a_blender_fisheye_panorama() -> Result<(), Box<dyn Error>> {
    let Some(camera) = panorama_round_trip(
        &json!({"projection": "panorama", "panorama_type": "fisheye_equidistant", "fisheye_fov": 2.5}),
        "import bpy; cam = [c for c in bpy.data.cameras if c.type == 'PANO'][0]; assert cam.panorama_type == 'FISHEYE_EQUIDISTANT'; assert abs(cam.fisheye_fov - 2.5) < 1e-6",
    )?
    else {
        return Ok(());
    };
    assert_eq!(camera["projection"], "panorama");
    assert_eq!(camera["panorama_type"], "fisheye_equidistant");
    assert_close_value(&camera["fisheye_fov"], 2.5);
    Ok(())
}

#[test]
fn fisheye_equisolid_round_trips_its_lens_and_its_perspective_lens_apart()
-> Result<(), Box<dyn Error>> {
    // `lens` is the perspective focal length and `fisheye_lens` the equisolid one: Blender keeps them separate, so both
    // round-trip, and the equisolid lens may be any value in Blender's fisheye range.
    let Some(camera) = panorama_round_trip(
        &json!({"projection": "panorama", "panorama_type": "fisheye_equisolid", "lens_mm": 35.0, "fisheye_lens": 12.0, "fisheye_fov": 3.0}),
        "import bpy; cam = [c for c in bpy.data.cameras if c.type == 'PANO'][0]; assert cam.panorama_type == 'FISHEYE_EQUISOLID'; assert abs(cam.lens - 35.0) < 1e-6; assert abs(cam.fisheye_lens - 12.0) < 1e-6; assert abs(cam.fisheye_fov - 3.0) < 1e-6",
    )?
    else {
        return Ok(());
    };
    assert_eq!(camera["panorama_type"], "fisheye_equisolid");
    assert_close_value(&camera["lens_mm"], 35.0);
    assert_close_value(&camera["fisheye_lens"], 12.0);
    assert_close_value(&camera["fisheye_fov"], 3.0);
    Ok(())
}

#[test]
fn settings_a_camera_projection_ignores_round_trip_without_loss() -> Result<(), Box<dyn Error>> {
    // A perspective camera that holds panorama settings (Blender stores them on every camera) keeps them through Blender.
    let Some(camera) = panorama_round_trip(
        &json!({"projection": "perspective", "lens_mm": 28.0, "fisheye_lens": 12.0, "fisheye_fov": 3.0,
                "longitude_min": -1.2, "longitude_max": 2.0, "latitude_min": -0.7, "latitude_max": 1.1}),
        "import bpy; cam = [c for c in bpy.data.cameras if c.type == 'PERSP'][0]; assert abs(cam.lens - 28.0) < 1e-6; assert abs(cam.fisheye_lens - 12.0) < 1e-6; assert abs(cam.fisheye_fov - 3.0) < 1e-6; assert abs(cam.longitude_min + 1.2) < 1e-6; assert abs(cam.latitude_max - 1.1) < 1e-6",
    )?
    else {
        return Ok(());
    };
    assert_eq!(camera["projection"], "perspective");
    assert_close_value(&camera["lens_mm"], 28.0);
    assert_close_value(&camera["fisheye_lens"], 12.0);
    assert_close_value(&camera["fisheye_fov"], 3.0);
    assert_close_value(&camera["longitude_min"], -1.2);
    Ok(())
}

/// Builds a panorama camera from `settings`, exports it strictly so that Potter loses nothing, checks the Blender camera with
/// the Python `check` expression, and imports it back without losses. Returns the imported camera data, or `None` when
/// no Blender is installed.
fn panorama_round_trip(settings: &Value, check: &str) -> Result<Option<Value>, Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping the panorama round trip; no executable Blender was found");
        return Ok(None);
    };
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    build_scene(&scene, settings)?;
    let blend = directory.path().join("panorama.blend");
    run(
        &mut export_command(&scene, &blend, &blender, &[]),
        "exporting the panorama camera",
    )?;
    blender_python(&blender, &blend, check)?;
    let output = run(
        &mut import_command(&scene, &blend, &blender, &[]),
        "importing the panorama camera",
    )?;
    let imported: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(imported["result"]["losses"], json!([]));
    Ok(Some(camera_data(&scene)?))
}

/// Blender stores its camera values as float32, so a round-tripped value matches to well within 1e-6.
fn assert_close_value(value: &Value, expected: f64) {
    let actual = value
        .as_f64()
        .unwrap_or_else(|| panic!("expected a number, found {value}"));
    assert!(
        (actual - expected).abs() <= 1.0e-6,
        "{actual} differs from {expected}"
    );
}

#[test]
fn camera_settings_blender_cannot_hold_are_refused_before_export() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    for (index, settings) in [
        json!({"projection": "panorama", "panorama_type": "fisheye_equisolid", "fisheye_lens": 150.0}),
        json!({"projection": "panorama", "panorama_type": "fisheye_equidistant", "fisheye_fov": 0.1}),
        json!({"projection": "perspective", "lens_mm": 0.5}),
        json!({"projection": "fisheye"}),
    ]
    .iter()
    .enumerate()
    {
        let scene = directory.path().join(format!("scene{index}"));
        assert!(
            build_scene(&scene, settings).is_err(),
            "the camera settings must be refused: {settings}"
        );
    }
    Ok(())
}
