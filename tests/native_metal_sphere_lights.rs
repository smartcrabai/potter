//! A rough metal sphere under one light, or under every light and a world, against Blender's Cycles.
//!
//! Blender renders the scene for several seeds at a fixed sample count and writes the mean RGB of
//! the central 16×16 patch of a 32×32 frame, where the sphere faces the camera. Potter imports the
//! same `.blend`, renders it with the same seeds, sample count and four bounces, and reduces each
//! frame to the same patch mean.
//!
//! The tolerance is `T · √(s_B² / n + s_P² / n)`, with `s` the seed-to-seed sample standard
//! deviations of each side, `n = SEEDS`, and `T` the two-sided 99.9% t quantile at 7 degrees of
//! freedom (5.408), the same bound as `native_subsurface_sphere`.
//!
//! Measured headless with the committed generator (Blender 5.2.2, 1024 samples per pixel, 32 seeds,
//! per-run relative seed standard deviation of the patch mean, red channel): point 0.03%, rectangle
//! 0.32%, sun 0.05%, world under 0.01%, all lights 0.21% (the green and blue channels differ by at
//! most 0.03%). At eight seeds the Blender standard error is a third of that. The rectangle is the noisiest light; the tolerance
//! follows the measured spread, so a sum of lights cannot hide a single light's error.
//!
//! Cost: each case renders `SEEDS` frames of 1024 samples per pixel on 32 × 32 pixels in Potter.
//! Its time in the debug test profile is not measured here; if it exceeds the 40 s budget, lower
//! `SAMPLES` (the tolerance tracks the seed spread, so a lower count stays valid but looser).
//!
//! The single-light cases localise a per-shape normalisation: `point` (Cycles `P / (4π)` radiant
//! intensity), `rect` (`P / (π · area)` radiance), and `sun` (irradiance equal to the strength). The
//! `world` case isolates the world's constant colour, and `all` combines them, so the sum is checked
//! at the same time. The sphere's roughness is 0.3, which keeps the lobe broad enough that the patch
//! sees each light.

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

use std::{fs, path::Path};

use native_oracle::{
    TestResult, apply_operations, import_blend, pot, read_scene, run_blender_arguments,
    write_script,
};
use potter_core::image::{ImageInterpolation, load_image_data, sample};
use potter_core::model::Image;
use serde_json::{Value, json};
use tempfile::tempdir;

const SCRIPT: &str = include_str!("fixtures/native_oracle/metal_sphere_lights.py");
/// Texel centres of the central 16×16 block of the 32 px frame, the same as `metal_sphere_lights.py`.
const PATCH: [f64; 16] = [
    8.5 / 32.0,
    9.5 / 32.0,
    10.5 / 32.0,
    11.5 / 32.0,
    12.5 / 32.0,
    13.5 / 32.0,
    14.5 / 32.0,
    15.5 / 32.0,
    16.5 / 32.0,
    17.5 / 32.0,
    18.5 / 32.0,
    19.5 / 32.0,
    20.5 / 32.0,
    21.5 / 32.0,
    22.5 / 32.0,
    23.5 / 32.0,
];
/// Samples per pixel and seeds, the same for Blender and Potter. Eight seeds give a seed-spread
/// estimate of the combined error, which the difference is judged against.
const SAMPLES: u32 = 1024;
const SEEDS: u32 = 8;
/// Two-sided 99.9% Student t quantile at 7 degrees of freedom (`SEEDS - 1`).
const T_99_9: f64 = 5.408;

/// Mean RGB of the central patch of the frame, one per render.
fn patch_mean(project: &Path, image_id: &str) -> TestResult<[f64; 3]> {
    let scene = read_scene(project)?;
    let image: Image = serde_json::from_value(scene["images"][image_id].clone())?;
    let data = load_image_data(&image, project, ImageInterpolation::Closest)?;
    let mut total = [0.0_f64; 3];
    for v in PATCH {
        for u in PATCH {
            let pixel = sample(&data, [u, v], false);
            for (sum, value) in total.iter_mut().zip(pixel.iter().take(3)) {
                *sum += value;
            }
        }
    }
    let count = (PATCH.len() * PATCH.len()) as f64;
    Ok(total.map(|sum| sum / count))
}

/// The mean of per-seed patch means, and the standard error of that mean from the seed spread.
fn mean_and_error(runs: &[[f64; 3]]) -> ([f64; 3], [f64; 3]) {
    let count = runs.len() as f64;
    let mut mean = [0.0_f64; 3];
    for run in runs {
        for (sum, value) in mean.iter_mut().zip(run) {
            *sum += value / count;
        }
    }
    let mut error = [0.0_f64; 3];
    for (channel, slot) in error.iter_mut().enumerate() {
        let variance = runs
            .iter()
            .map(|run| (run[channel] - mean[channel]).powi(2))
            .sum::<f64>()
            / (count - 1.0);
        *slot = (variance / count).sqrt();
    }
    (mean, error)
}

fn check_scenario(scenario: &str) -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "metal_sphere_lights.py", SCRIPT)?;
    let blend = root.path().join("scene.blend");
    let json_path = root.path().join("blender.json");
    let samples_arg = SAMPLES.to_string();
    let seeds_arg = SEEDS.to_string();
    let arguments: [&Path; 5] = [
        &blend,
        &json_path,
        Path::new(scenario),
        Path::new(&samples_arg),
        Path::new(&seeds_arg),
    ];
    run_blender_arguments(&blender, &script, None, &arguments)?;

    let expected: Value = serde_json::from_slice(&fs::read(&json_path)?)?;
    let mut wanted_runs = Vec::new();
    for run in expected["patch_means"]
        .as_array()
        .ok_or("the Blender patch means are missing")?
    {
        let channels = run.as_array().ok_or("a Blender patch mean is not a list")?;
        let mut rgb = [0.0_f64; 3];
        for (slot, value) in rgb.iter_mut().zip(channels) {
            *slot = value
                .as_f64()
                .ok_or("a Blender patch mean is not a number")?;
        }
        wanted_runs.push(rgb);
    }
    assert_eq!(
        wanted_runs.len(),
        SEEDS as usize,
        "one Blender run per seed"
    );
    let (wanted, wanted_error) = mean_and_error(&wanted_runs);
    assert!(
        wanted.iter().all(|value| *value > 0.005),
        "{scenario}: the Blender reference must be lit, got {wanted:?}"
    );

    let (project, _) = import_blend(&blender, root.path(), "project", &blend)?;
    let scene = read_scene(&project)?;
    let active = scene["active_scene"]
        .as_str()
        .ok_or("active scene is missing")?
        .to_owned();
    let mut potter_runs = Vec::new();
    for seed in 0..SEEDS {
        let applied = apply_operations(
            &project,
            root.path(),
            json!([
                {"op": "render.update", "target": {"id": active}, "set": {"samples": SAMPLES, "seed": seed, "max_bounces": 4}}
            ]),
        )?;
        assert_eq!(applied["ok"], true, "{applied}");

        let frames = root.path().join(format!("frames_{seed}"));
        let rendered = pot()
            .arg("render")
            .arg(&project)
            .args(["--format", "exr", "--out"])
            .arg(&frames)
            .arg("--json")
            .output()?;
        assert!(
            rendered.status.success(),
            "render failed: {}",
            String::from_utf8_lossy(&rendered.stdout)
        );
        let image_id = format!("render_{seed}");
        let loaded = apply_operations(
            &project,
            root.path(),
            json!([{"op": "image.load", "id": image_id, "path": frames.join("frame_0001.exr").to_string_lossy()}]),
        )?;
        assert_eq!(loaded["ok"], true, "{loaded}");
        potter_runs.push(patch_mean(&project, &image_id)?);
    }
    let (actual, actual_error) = mean_and_error(&potter_runs);
    eprintln!(
        "METAL_SPHERE {scenario}: potter {actual:?} ± {actual_error:?} | cycles {wanted:?} ± {wanted_error:?}"
    );

    for channel in 0..3 {
        let tolerance =
            T_99_9 * (wanted_error[channel].powi(2) + actual_error[channel].powi(2)).sqrt();
        assert!(
            (actual[channel] - wanted[channel]).abs() <= tolerance,
            "{scenario} channel {channel}: Potter {} differs from Cycles {} by more than {tolerance} \
             (the 99.9% t bound times the combined standard error; Potter ± {}, Cycles ± {})",
            actual[channel],
            wanted[channel],
            actual_error[channel],
            wanted_error[channel],
        );
    }
    Ok(())
}

#[test]
fn point_light_on_a_rough_metal_sphere_matches_cycles() -> TestResult {
    check_scenario("point")
}

#[test]
fn rectangle_light_on_a_rough_metal_sphere_matches_cycles() -> TestResult {
    check_scenario("rect")
}

#[test]
fn sun_on_a_rough_metal_sphere_matches_cycles() -> TestResult {
    check_scenario("sun")
}

#[test]
fn world_on_a_rough_metal_sphere_matches_cycles() -> TestResult {
    check_scenario("world")
}

#[test]
fn point_rectangle_and_sun_with_world_on_a_rough_metal_sphere_match_cycles() -> TestResult {
    check_scenario("all")
}
