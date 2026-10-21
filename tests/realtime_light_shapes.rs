//! Camera-visible light shapes in the realtime Beauty image, checked against Blender 5.2.2's EEVEE.
//!
//! `fixtures/native_oracle/eevee_light_shapes.py` renders a square area light and a radius-1 point light, both with
//! `visible_camera` on, on a black world. EEVEE draws each emitter at its radiance (`eevee_light_shape_display`), which
//! is the path tracer's emitter radiance for the same light. The test imports the saved `.blend` and renders it with
//! `pot render --engine realtime`, then compares the mean red value of a 7 x 7 window at each light's centre, and the
//! background, with EEVEE's.
//!
//! The tolerance is 0.005: EEVEE's own window means lie within 0.0005 of the emitter formulas (`P / (π A)` for the
//! area light and `P / (4 π² r²)` for the sphere), so the bound is ten times that gap, far below either light's value.
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
    TestResult, apply_operations, import_blend, pot, read_scene, run_blender_arguments,
    write_script,
};
use potter_core::image::{ImageInterpolation, load_image_data, sample};
use potter_core::model::Image;
use serde_json::{Value, json};
use tempfile::tempdir;

/// The scene and EEVEE render (`eevee_light_shapes.py`).
const SCRIPT: &str = include_str!("fixtures/native_oracle/eevee_light_shapes.py");

/// Render settings shared with `eevee_light_shapes.py`.
const RESOLUTION: u32 = 64;

/// Linear tolerance on a window mean of red against EEVEE. See the module comment.
const TOLERANCE: f64 = 0.005;

/// The mean red value of the `(2 * half + 1)` square window centred on `(column, row)` in the image `id`, read with
/// Potter's own EXR loader at the texel centres.
fn window_mean(
    project: &Path,
    id: &str,
    column: usize,
    row: usize,
    half: usize,
) -> Result<f64, Box<dyn Error>> {
    let scene = read_scene(project)?;
    let image: Image = serde_json::from_value(scene["images"][id].clone())?;
    let data = load_image_data(&image, project, ImageInterpolation::Closest)?;
    let texels = f64::from(RESOLUTION);
    let mut total = 0.0;
    let mut count = 0.0;
    for r in row - half..=row + half {
        for c in column - half..=column + half {
            let uv = [(c as f64 + 0.5) / texels, (r as f64 + 0.5) / texels];
            total += sample(&data, uv, false)[0];
            count += 1.0;
        }
    }
    Ok(total / count)
}

#[test]
fn realtime_camera_visible_lights_match_eevee_emitters() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "eevee_light_shapes.py", SCRIPT)?;
    let blend = root.path().join("light_shapes.blend");
    let oracle_json = root.path().join("eevee_light_shapes.json");
    let eevee_exr = root.path().join("eevee_light_shapes.exr");
    run_blender_arguments(
        &blender,
        &script,
        None,
        &[blend.as_path(), oracle_json.as_path(), eevee_exr.as_path()],
    )?;
    let oracle: Value = serde_json::from_slice(&fs::read(&oracle_json)?)?;
    let half = oracle["window"].as_u64().ok_or("window is missing")? as usize;
    let row = RESOLUTION as usize / 2;
    let area_column = oracle["area_centre_column"]
        .as_u64()
        .ok_or("area column is missing")? as usize;
    let point_column = oracle["point_centre_column"]
        .as_u64()
        .ok_or("point column is missing")? as usize;

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
            {"op": "render.update", "target": {"id": active}, "set": {"resolution_x": RESOLUTION, "resolution_y": RESOLUTION, "samples": 4, "seed": 0}}
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
    let loaded = apply_operations(
        &project,
        root.path(),
        json!([{"op": "image.load", "id": "render", "path": frames.join("frame_0001.exr").to_string_lossy()}]),
    )?;
    assert_eq!(loaded["ok"], true, "{loaded}");

    let checks = [
        (
            "area light",
            window_mean(&project, "render", area_column, row, half)?,
            oracle["area_mean_red"]
                .as_f64()
                .ok_or("area mean is missing")?,
        ),
        (
            "point light",
            window_mean(&project, "render", point_column, row, half)?,
            oracle["point_mean_red"]
                .as_f64()
                .ok_or("point mean is missing")?,
        ),
        (
            "background",
            window_mean(
                &project,
                "render",
                RESOLUTION as usize / 4,
                RESOLUTION as usize / 4,
                half,
            )?,
            oracle["background_mean_red"]
                .as_f64()
                .ok_or("background mean is missing")?,
        ),
    ];
    for (name, actual, expected) in checks {
        assert!(
            (actual - expected).abs() <= TOLERANCE,
            "{name}: Potter {actual} differs from EEVEE {expected}"
        );
    }
    Ok(())
}
