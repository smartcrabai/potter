//! Realtime Beauty emission on flat and smooth materials, checked against Blender 5.2.2's EEVEE.
//!
//! `fixtures/native_oracle/eevee_flat_emission.py` renders two Principled planes under a black
//! world with no lights: `FlatEmitter` (flat faces) and `SmoothEmitter` (smooth shading). Both
//! emit colour (0.6, 0.3, 0.1) at strength 1.5, so the radiance is colour times strength. The
//! planes are imported from the saved `.blend`, rendered by `pot render --engine realtime` to a
//! linear EXR, and each plane's centre pixel is compared with EEVEE's pixel at the same UV.
//!
//! Blender-gated: with `POTTER_REQUIRE_BLENDER` set, a missing Blender fails the test.

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/native_oracle.rs"]
#[expect(
    dead_code,
    reason = "each test binary uses a subset of the shared oracle helpers"
)]
mod native_oracle;
#[path = "common/process.rs"]
mod process;

use std::{error::Error, fs, path::Path};

use native_oracle::{
    TestResult, apply_operations, floats, import_blend, pot, read_scene, run_blender_arguments,
    vec4, write_script,
};
use potter_core::image::{ImageInterpolation, load_image_data, sample};
use potter_core::model::Image;
use serde_json::{Value, json};
use tempfile::tempdir;

/// Linear tolerance against EEVEE's pixel. EEVEE stores its render in half float, so the largest
/// channel (0.9) rounds by at most half of the half-float step 2^-11 (about 2.4e-4); the oracle
/// script measures its own interior deviation at 9.8e-5. The bound is two half-float steps at the
/// peak, so a dropped or mis-scaled emission (0.6 or 0.9 off) fails.
const TOLERANCE: f64 = 1.0e-3;

/// Blender 5.2.2 EEVEE oracle: the two emitter planes, the saved `.blend`, and the measured pixels.
const EEVEE_SCRIPT: &str = include_str!("fixtures/native_oracle/eevee_flat_emission.py");

/// The planes, by the names `eevee_flat_emission.py` writes into its JSON.
const PLANES: [&str; 2] = ["flat", "smooth"];

/// RGBA of the image `id` in `project`, sampled at `uv` with Potter's own EXR loader.
fn pixel_at(project: &Path, id: &str, uv: [f64; 2]) -> Result<[f64; 4], Box<dyn Error>> {
    let scene = read_scene(project)?;
    let image: Image = serde_json::from_value(scene["images"][id].clone())?;
    let data = load_image_data(&image, project, ImageInterpolation::Closest)?;
    Ok(sample(&data, uv, false))
}

#[test]
fn realtime_flat_and_smooth_emission_match_eevee() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "eevee_flat_emission.py", EEVEE_SCRIPT)?;
    let blend = root.path().join("flat_emission.blend");
    let eevee_exr = root.path().join("eevee_flat_emission.exr");
    let oracle_json = root.path().join("eevee_flat_emission.json");
    run_blender_arguments(
        &blender,
        &script,
        None,
        &[blend.as_path(), eevee_exr.as_path(), oracle_json.as_path()],
    )?;
    let oracle: Value = serde_json::from_slice(&fs::read(&oracle_json)?)?;
    let expected = floats(&oracle["expected_rgb"])?;
    for plane in PLANES {
        let deviation = oracle[plane]["interior_max_deviation"]
            .as_f64()
            .ok_or("interior deviation is missing")?;
        assert!(
            deviation <= TOLERANCE,
            "EEVEE {plane} interior deviates {deviation} from colour times strength"
        );
        let rgba = vec4(&oracle[plane]["rgba"])?;
        for channel in 0..3 {
            assert!(
                (rgba[channel] - expected[channel]).abs() <= TOLERANCE,
                "EEVEE {plane} channel {channel} is {}, expected {}",
                rgba[channel],
                expected[channel]
            );
        }
    }

    let (project, _) = import_blend(&blender, root.path(), "project", &blend)?;
    let scene = read_scene(&project)?;
    let active = scene["active_scene"]
        .as_str()
        .ok_or("active scene is missing")?
        .to_owned();
    let applied = apply_operations(
        &project,
        root.path(),
        json!([
            {"op": "camera.create", "id": "camera_main", "transform": {"translation": [0.0, 0.0, 6.0]}, "projection": "orthographic", "ortho_scale": 4.0},
            {"op": "scene.update", "target": {"id": active}, "set": {"camera": "camera_main"}},
            {"op": "render.update", "target": {"id": active}, "set": {"resolution_x": 64, "resolution_y": 64, "samples": 1, "seed": 0}}
        ]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");

    let frames = root.path().join("frames");
    let rendered = pot()
        .arg("render")
        .arg(&project)
        .args(["--engine", "realtime", "--format", "exr", "--out"])
        .arg(&frames)
        .arg("--json")
        .output()?;
    assert!(
        rendered.status.success(),
        "realtime render failed: {}",
        String::from_utf8_lossy(&rendered.stdout)
    );
    let exr = frames.join("frame_0001.exr");
    let loaded = apply_operations(
        &project,
        root.path(),
        json!([{"op": "image.load", "id": "render", "path": exr.to_string_lossy()}]),
    )?;
    assert_eq!(loaded["ok"], true, "{loaded}");

    for plane in PLANES {
        let uv = floats(&oracle[plane]["uv"])?;
        let actual = pixel_at(&project, "render", [uv[0], uv[1]])?;
        let wanted = vec4(&oracle[plane]["rgba"])?;
        for channel in 0..3 {
            assert!(
                (actual[channel] - wanted[channel]).abs() <= TOLERANCE,
                "{plane} plane channel {channel}: Potter realtime {} differs from EEVEE {}",
                actual[channel],
                wanted[channel]
            );
        }
    }
    Ok(())
}
