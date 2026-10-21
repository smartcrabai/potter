//! Realtime Beauty's light closure factors and the sun's disc against Blender 5.2.2's EEVEE with its default settings.
//!
//! `fixtures/native_oracle/eevee_light_factors.py` renders nine scenes on a black world. The factors are linear, and the
//! Python mirror (`fixtures/native_oracle/eevee_light_factors_mirror.py`) reproduces EEVEE's pixels to float rounding:
//! `diffuse_half` is half `diffuse_base` (error 5e-7), `specular_half` is half `specular_base` (7e-7), and `glass_half` is the
//! reflection plus half the transmitted light (3e-7). The test therefore checks Potter's factors as ratios of its own renders,
//! and checks the disc against EEVEE:
//! - the diffuse base against EEVEE within 2.5e-3 (the point light's cut-off oracle's bound);
//! - `diffuse_half` and `specular_half` within 1e-3 of half their base, and `glass_half` within 1e-3 of the model in which
//!   only the transmitted light is halved;
//! - the sun's disc effect: the mean of `sun_disc` over the mean of `sun_delta`, against EEVEE's 0.8169. The mirror's
//!   16-sample GGX prediction is 0.8107, so the bound is 1.2e-2 (twice the 6e-3 model residual, rounded up). A centre-only
//!   lobe would give 1.0, which fails by 0.18.
//!
//! Potter's absolute specular is not compared with EEVEE here: no mirror of Potter's Principled energy tables exists yet, so
//! only the factor and the disc's effect are bounded.
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

/// Blender 5.2.2 EEVEE oracle: the factor and disc scenes, the saved `.blend`, and every pixel.
const EEVEE_SCRIPT: &str = include_str!("fixtures/native_oracle/eevee_light_factors.py");

const SIZE: u32 = 128;
const ORTHO_SCALE: f64 = 12.0;

/// The diffuse base's absolute bound (see the module docs).
const DIFFUSE_TOLERANCE: f64 = 2.5e-3;
/// The factor bound: the factor's ratio of Potter's own renders, and the glass's transmitted-light model.
const FACTOR_TOLERANCE: f64 = 1.0e-3;
/// The sun's disc effect bound: the disc-to-delta mean ratio against EEVEE's.
const DISC_RATIO_TOLERANCE: f64 = 1.2e-2;
/// EEVEE's disc-to-delta mean ratio on the same plane (`sun_disc` over `sun_delta`).
const EEVEE_DISC_RATIO: f64 = 0.816_921;

const CASES: [&str; 9] = [
    "diffuse_base",
    "diffuse_half",
    "specular_base",
    "specular_half",
    "glass_base",
    "glass_zero",
    "glass_half",
    "sun_delta",
    "sun_disc",
];

#[test]
fn realtime_light_factors_and_sun_disc_match_eevee() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "eevee_light_factors.py", EEVEE_SCRIPT)?;
    let mut potter = Vec::new();
    let mut eevee_diffuse = Vec::new();
    for case in CASES {
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
        if case == "diffuse_base" {
            eevee_diffuse = oracle_values(&oracle)?;
        }
        let data = render_potter(&blender, root.path(), case, &blend)?;
        potter.push((case, plane_values(&data)));
    }
    let values = |name: &str| -> Result<&Vec<[f64; 3]>, Box<dyn Error>> {
        potter
            .iter()
            .find(|(case, _)| *case == name)
            .map(|(_, values)| values)
            .ok_or_else(|| format!("no render of {name}").into())
    };

    // The diffuse base is compared with EEVEE's pixels.
    let worst = worst_deviation(values("diffuse_base")?, &eevee_diffuse)?;
    assert!(
        worst <= DIFFUSE_TOLERANCE,
        "diffuse base deviates from EEVEE by {worst} (tolerance {DIFFUSE_TOLERANCE})"
    );

    // Each factor scales its closure class: Potter's half against Potter's base.
    let half_error = worst_difference(
        values("diffuse_half")?,
        &scaled(values("diffuse_base")?, 0.5),
    );
    assert!(
        half_error <= FACTOR_TOLERANCE,
        "diffuse_factor 0.5 is not half of Potter's diffuse base: error {half_error}"
    );
    let specular_error = worst_difference(
        values("specular_half")?,
        &scaled(values("specular_base")?, 0.5),
    );
    assert!(
        specular_error <= FACTOR_TOLERANCE,
        "specular_factor 0.5 is not half of Potter's specular base: error {specular_error}"
    );
    // The transmitted light alone is halved: the reflection (`glass_zero`) does not change.
    let glass_model: Vec<[f64; 3]> = values("glass_base")?
        .iter()
        .zip(values("glass_zero")?)
        .map(|(base, zero)| std::array::from_fn(|c| zero[c] + 0.5 * (base[c] - zero[c])))
        .collect();
    let glass_error = worst_difference(values("glass_half")?, &glass_model);
    assert!(
        glass_error <= FACTOR_TOLERANCE,
        "transmission_factor 0.5 is not the reflection plus half the transmitted light: error {glass_error}"
    );

    // The sun's disc: the disc-to-delta ratio of Potter's renders against EEVEE's.
    let potter_ratio = mean_sum(values("sun_disc")?) / mean_sum(values("sun_delta")?);
    assert!(
        (potter_ratio - EEVEE_DISC_RATIO).abs() <= DISC_RATIO_TOLERANCE,
        "the sun's disc ratio is {potter_ratio} in Potter and {EEVEE_DISC_RATIO} in EEVEE (tolerance {DISC_RATIO_TOLERANCE})"
    );
    Ok(())
}

/// The plane or sphere pixels of a case, as RGB, in the oracle's order.
fn plane_values(data: &ImageData) -> Vec<[f64; 3]> {
    oracle_positions()
        .into_iter()
        .map(|uv| {
            let rgba = sample(data, uv, false);
            [rgba[0], rgba[1], rgba[2]]
        })
        .collect()
}

/// Potter's pixel centres, as the oracle lays them out: the UV of each pixel of the square image.
fn oracle_positions() -> Vec<[f64; 2]> {
    let count = SIZE as usize;
    (0..count)
        .flat_map(|row| {
            (0..count).map(move |column| {
                [
                    (column as f64 + 0.5) / count as f64,
                    (row as f64 + 0.5) / count as f64,
                ]
            })
        })
        .collect()
}

fn oracle_values(oracle: &Value) -> Result<Vec<[f64; 3]>, Box<dyn Error>> {
    oracle["samples"]
        .as_array()
        .ok_or("the EEVEE samples are missing")?
        .iter()
        .map(|sample_value| triple(&sample_value["rgb"]))
        .collect()
}

fn scaled(values: &[[f64; 3]], factor: f64) -> Vec<[f64; 3]> {
    values
        .iter()
        .map(|rgb| rgb.map(|channel| channel * factor))
        .collect()
}

fn worst_difference(actual: &[[f64; 3]], expected: &[[f64; 3]]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .flat_map(|(a, e)| a.iter().zip(e).map(|(a, e)| (a - e).abs()))
        .fold(0.0_f64, f64::max)
}

fn worst_deviation(actual: &[[f64; 3]], expected: &[[f64; 3]]) -> Result<f64, Box<dyn Error>> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{} Potter pixels against {} EEVEE pixels",
            actual.len(),
            expected.len()
        )
        .into());
    }
    Ok(worst_difference(actual, expected))
}

/// The mean over the plane of the sum of a pixel's channels, as a light's brightness.
fn mean_sum(values: &[[f64; 3]]) -> f64 {
    values
        .iter()
        .map(|rgb| rgb.iter().sum::<f64>())
        .sum::<f64>()
        / values.len() as f64
}

/// Renders a case's `.blend` in Potter's realtime engine, with the oracle's camera and size, and loads the frame.
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
            {"op": "camera.create", "id": "camera_main", "transform": {"translation": [0.0, 0.0, 20.0]}, "projection": "orthographic", "ortho_scale": ORTHO_SCALE},
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

fn triple(value: &Value) -> Result<[f64; 3], Box<dyn Error>> {
    let items = value.as_array().ok_or("an RGB is not an array")?;
    let channel = |index: usize| {
        items
            .get(index)
            .and_then(Value::as_f64)
            .ok_or("an RGB channel is missing")
    };
    Ok([channel(0)?, channel(1)?, channel(2)?])
}
