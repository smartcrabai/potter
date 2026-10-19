//! Realtime Beauty emission on flat, smooth and graph-per-pixel materials, checked against Blender
//! 5.2.2's EEVEE.
//!
//! `fixtures/native_oracle/eevee_flat_emission.py` renders three Principled planes under a black
//! world with no lights: `FlatEmitter` (flat faces), `SmoothEmitter` (smooth shading), and
//! `GraphEmitter` (flat faces, emission colour from a Vertex Color node, so it is evaluated per
//! pixel). Each emits colour times strength. The planes are imported from the saved `.blend`,
//! rendered by `pot render --engine realtime` to a linear EXR, and each plane's centre pixel is
//! compared with EEVEE's pixel at the same UV.
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

/// Linear tolerance against EEVEE's pixel. EEVEE stores its render in half float: one step at
/// [0.5, 1) is 2^-11 (about 4.9e-4), and the oracle measures its own deviation from the analytic
/// radiance at 9.8e-5 (flat and smooth) and 3.9e-4 (graph). The bound is two steps, so a dropped
/// or mis-scaled emission (0.3 or more off) fails.
const TOLERANCE: f64 = 1.0e-3;

/// Blender 5.2.2 EEVEE oracle: the three emitter planes, the saved `.blend`, and the measured pixels.
const EEVEE_SCRIPT: &str = include_str!("fixtures/native_oracle/eevee_flat_emission.py");

/// The planes, by the names `eevee_flat_emission.py` writes into its JSON.
const PLANES: [&str; 3] = ["flat", "smooth", "graph"];

/// RGBA of the image `id` in `project`, sampled at `uv` with Potter's own EXR loader.
fn pixel_at(project: &Path, id: &str, uv: [f64; 2]) -> Result<[f64; 4], Box<dyn Error>> {
    let scene = read_scene(project)?;
    let image: Image = serde_json::from_value(scene["images"][id].clone())?;
    let data = load_image_data(&image, project, ImageInterpolation::Closest)?;
    Ok(sample(&data, uv, false))
}

#[test]
fn realtime_flat_smooth_and_graph_emission_match_eevee() -> TestResult {
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
    for plane in PLANES {
        let deviation = oracle[plane]["interior_max_deviation"]
            .as_f64()
            .ok_or("interior deviation is missing")?;
        assert!(
            deviation <= TOLERANCE,
            "EEVEE {plane} interior deviates {deviation} from its radiance"
        );
        let expected = floats(&oracle[plane]["expected_rgb"])?;
        let rgba = vec4(&oracle[plane]["rgba"])?;
        for (channel, (actual, wanted)) in rgba.into_iter().zip(&expected).take(3).enumerate() {
            assert!(
                (actual - wanted).abs() <= TOLERANCE,
                "EEVEE {plane} channel {channel} is {actual}, expected {wanted}"
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
            {"op": "camera.create", "id": "camera_main", "transform": {"translation": [0.0, 0.0, 6.0]}, "projection": "orthographic", "ortho_scale": 6.0},
            {"op": "scene.update", "target": {"id": active}, "set": {"camera": "camera_main"}},
            {"op": "render.update", "target": {"id": active}, "set": {"resolution_x": 96, "resolution_y": 96, "samples": 1, "seed": 0}}
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
        for (channel, (actual, wanted)) in actual.into_iter().zip(wanted).take(3).enumerate() {
            assert!(
                (actual - wanted).abs() <= TOLERANCE,
                "{plane} plane channel {channel}: Potter realtime {actual} differs from EEVEE {wanted}"
            );
        }
    }
    Ok(())
}
