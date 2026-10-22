//! Realtime Beauty's volume in-scatter follows a light's `volume_factor`, as EEVEE's `power[LIGHT_VOLUME]` does.
//!
//! `fixtures/native_oracle/eevee_volume_factor.py` renders a pure scatterer, a 2 m cube, lit by a 200 W point light above it,
//! three times: with the light's Volume Factor at 1, at 0, and at 0.5. The black world leaves the in-scatter alone, so the
//! three renders test one linear law: the in-scatter is proportional to the factor, and a factor of 0 switches it off.
//!
//! - EEVEE: the factor-0.5 render is half the factor-1 render within 1.5e-3 (EEVEE's own sampling noise: the largest pixel
//!   deviation measured over the 96 by 96 image is 7.0e-4, and 1.5e-3 is twice that, rounded up); the factor-0 render is black.
//! - Potter: the same three relations hold for Potter's renders, to 1e-6, since the in-scatter is multiplied by the factor
//!   exactly.
//!
//! The absolute in-scatter is not compared with EEVEE: Potter's realtime volume is a single-scatter model with Cycles'
//! light radiance, which differs from EEVEE's volume light (`point_radiance_get`) for area and sun lights.
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
use potter_core::image::{ImageData, ImageInterpolation, load_image_data, sample};
use potter_core::model::Image;
use serde_json::{Value, json};
use tempfile::tempdir;

/// Blender 5.2.2 EEVEE oracle: the scatterer with its light's factor, the saved `.blend`, and every pixel.
const EEVEE_SCRIPT: &str = include_str!("fixtures/native_oracle/eevee_volume_factor.py");

const SIZE: u32 = 96;
const ORTHO_SCALE: f64 = 3.0;

/// The EEVEE half-factor bound (see the module docs).
const EEVEE_LINEAR_TOLERANCE: f64 = 1.5e-3;
/// Potter's relations are exact up to the render's float precision.
const POTTER_LINEAR_TOLERANCE: f64 = 1.0e-6;

#[test]
fn realtime_volume_in_scatter_follows_the_light_volume_factor() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "eevee_volume_factor.py", EEVEE_SCRIPT)?;
    let mut eevee = Vec::new();
    let mut potter = Vec::new();
    for case in ["volume_base", "volume_zero", "volume_half"] {
        let blend = root.path().join(format!("{case}.blend"));
        let eevee_exr = root.path().join(format!("{case}.exr"));
        let oracle_json = root.path().join(format!("{case}.json"));
        run_blender_arguments(
            &blender,
            &script,
            None,
            &[
                Path::new(case),
                blend.as_path(),
                eevee_exr.as_path(),
                oracle_json.as_path(),
            ],
        )?;
        let oracle: Value = serde_json::from_slice(&fs::read(&oracle_json)?)?;
        eevee.push(oracle_values(&oracle)?);
        let data = render_potter(&blender, root.path(), case, &blend)?;
        potter.push(plane_values(&data));
    }
    let [eevee_base, eevee_zero, eevee_half] =
        <[Vec<f64>; 3]>::try_from(eevee).map_err(|_| "three EEVEE renders are expected")?;
    let [potter_base, potter_zero, potter_half] =
        <[Vec<f64>; 3]>::try_from(potter).map_err(|_| "three Potter renders are expected")?;

    let eevee_half_error = worst_difference(&eevee_half, &scaled(&eevee_base, 0.5));
    assert!(
        eevee_half_error <= EEVEE_LINEAR_TOLERANCE,
        "EEVEE's half-factor render is not half its base: error {eevee_half_error}"
    );
    let eevee_zero_max = eevee_zero.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
    assert!(
        eevee_zero_max <= EEVEE_LINEAR_TOLERANCE,
        "EEVEE's zero-factor render is not black: {eevee_zero_max}"
    );

    let potter_zero_max = potter_zero.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
    assert!(
        potter_zero_max <= POTTER_LINEAR_TOLERANCE,
        "Potter's zero-factor render keeps in-scatter: {potter_zero_max}"
    );
    let potter_half_error = worst_difference(&potter_half, &scaled(&potter_base, 0.5));
    assert!(
        potter_half_error <= POTTER_LINEAR_TOLERANCE,
        "Potter's half-factor render is not half its base: error {potter_half_error}"
    );
    let potter_base_max = potter_base.iter().fold(0.0_f64, |a, b| a.max(*b));
    assert!(
        potter_base_max > 1.0e-3,
        "Potter's volume shows no in-scatter, so the factor is not tested ({potter_base_max})"
    );
    Ok(())
}

/// Every pixel's red channel, in the oracle's row order. A volume is grey, so one channel carries it.
fn oracle_values(oracle: &Value) -> Result<Vec<f64>, Box<dyn Error>> {
    oracle["samples"]
        .as_array()
        .ok_or("the EEVEE samples are missing")?
        .iter()
        .map(|sample_value| {
            sample_value["rgb"]
                .as_array()
                .and_then(|items| items.first())
                .and_then(Value::as_f64)
                .ok_or_else(|| "an RGB red channel is missing".into())
        })
        .collect()
}

/// Potter's red channel at the oracle's pixel centres, in the same order (row by row, bottom row first).
fn plane_values(data: &ImageData) -> Vec<f64> {
    let count = SIZE as usize;
    (0..count)
        .flat_map(|row| (0..count).map(move |column| (column, row)))
        .map(|(column, row)| {
            let uv = [
                (column as f64 + 0.5) / count as f64,
                (row as f64 + 0.5) / count as f64,
            ];
            sample(data, uv, false)[0]
        })
        .collect()
}

fn scaled(values: &[f64], factor: f64) -> Vec<f64> {
    values.iter().map(|value| value * factor).collect()
}

fn worst_difference(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0_f64, f64::max)
}

/// Renders a case's `.blend` in Potter's realtime engine with the oracle's camera and size, and loads the frame.
fn render_potter(
    blender: &Path,
    root: &Path,
    case: &str,
    blend: &Path,
) -> Result<ImageData, Box<dyn Error>> {
    let project_name = format!("project_{case}");
    let (project, _) = import_blend(blender, root, &project_name, blend)?;
    let scene_data = read_scene(&project)?;
    let active = scene_data["active_scene"]
        .as_str()
        .ok_or("active scene is missing")?
        .to_owned();
    let applied = apply_operations(
        &project,
        root,
        json!([
            {"op": "camera.create", "id": "camera_main", "transform": {"translation": [0.0, 0.0, 10.0]}, "projection": "orthographic", "ortho_scale": ORTHO_SCALE},
            {"op": "scene.update", "target": {"id": active}, "set": {"camera": "camera_main"}},
            {"op": "render.update", "target": {"id": active}, "set": {"resolution_x": SIZE, "resolution_y": SIZE, "samples": 1, "seed": 0}}
        ]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");
    let frames = root.join(format!("frames_{case}"));
    let rendered = pot()
        .arg("render")
        .arg(&project)
        .args(["--engine", "realtime", "--format", "exr", "--out"])
        .arg(&frames)
        .arg("--json")
        .output()?;
    // Diagnostic: the render's stderr carries the DIAG lines of `scene.rs` and `path.rs`.
    eprint!("{}", String::from_utf8_lossy(&rendered.stderr));
    assert!(
        rendered.status.success(),
        "{case}: realtime render failed: {}",
        String::from_utf8_lossy(&rendered.stdout)
    );
    let loaded = apply_operations(
        &project,
        root,
        json!([{"op": "image.load", "id": "render", "path": frames.join("frame_0001.exr").to_string_lossy()}]),
    )?;
    assert_eq!(loaded["ok"], true, "{loaded}");
    let image: Image = serde_json::from_value(read_scene(&project)?["images"]["render"].clone())?;
    Ok(load_image_data(
        &image,
        &project,
        ImageInterpolation::Closest,
    )?)
}
