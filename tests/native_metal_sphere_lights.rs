//! A rough metal sphere under one light, or under every light and a world, against Blender's Cycles.
//!
//! The Blender reference is committed (`fixtures/native_oracle/metal_sphere_lights_reference.json`):
//! for each scenario, the mean RGB of the central 16 × 16 patch of a 32 × 32 frame at 1024 samples
//! per pixel, four bounces, for 32 seeds. This test never renders in Blender. It imports the
//! scene the generator writes (`metal_sphere_lights.py`, frame 16, a camera zoomed by two so the
//! 16 × 16 frame covers the same directions and pixel grid as the reference's central patch), and
//! renders Potter for `SEEDS` seeds at `SAMPLES` samples per pixel. The frame mean is the patch mean.
//!
//! The tolerance is `T · √(s_B² / n_B + s_P² / n_P)`, with `s` the seed-to-seed sample standard
//! deviations of each side and `T` the two-sided 99.9% t quantile at 7 degrees of freedom (5.408),
//! the same bound as `native_subsurface_sphere`. Potter's own spread is measured in each run, so
//! the tolerance follows Potter's noise at `SAMPLES`. Fewer samples per seed with more seeds keeps
//! the mean's error at a fixed total sample count, and the t quantile falls from 12.9 at four seeds
//! to 5.4 at eight.
//!
//! Cost: `SEEDS` renders of 16 × 16 pixels at `SAMPLES` samples per pixel per scenario, and one
//! Blender import (which dumps the scene; no render).

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

use std::path::Path;

use native_oracle::{
    TestResult, apply_operations, import_blend, pot, read_scene, run_blender_arguments,
    write_script,
};
use potter_core::image::{ImageInterpolation, load_image_data, sample};
use potter_core::model::Image;
use serde_json::{Value, json};
use tempfile::tempdir;

const SCRIPT: &str = include_str!("fixtures/native_oracle/metal_sphere_lights.py");
const REFERENCE: &str = include_str!("fixtures/native_oracle/metal_sphere_lights_reference.json");
/// Frame size of the Potter scene. Its 16 × 16 frame is the reference's central patch.
const FRAME: u32 = 16;
/// Samples per pixel and seeds for Potter. The reference has 1024 samples and 32 seeds.
const SAMPLES: u32 = 128;
const SEEDS: u32 = 8;
/// Two-sided 99.9% Student t quantile at 7 degrees of freedom (`SEEDS - 1`).
const T_99_9: f64 = 5.408;

/// Texel centres of the whole 16 × 16 frame.
fn patch() -> [f64; 16] {
    std::array::from_fn(|index| (index as f64 + 0.5) / f64::from(FRAME))
}

/// Mean RGB of the central patch of the frame, one per render.
fn patch_mean(project: &Path, image_id: &str) -> TestResult<[f64; 3]> {
    let scene = read_scene(project)?;
    let image: Image = serde_json::from_value(scene["images"][image_id].clone())?;
    let data = load_image_data(&image, project, ImageInterpolation::Closest)?;
    let mut total = [0.0_f64; 3];
    let texels = patch();
    for v in texels {
        for u in texels {
            let pixel = sample(&data, [u, v], false);
            for (sum, value) in total.iter_mut().zip(pixel.iter().take(3)) {
                *sum += value;
            }
        }
    }
    let count = (texels.len() * texels.len()) as f64;
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

/// The committed Blender patch means of one scenario, one RGB triple per seed.
fn reference_runs(scenario: &str) -> TestResult<Vec<[f64; 3]>> {
    let reference: Value = serde_json::from_str(REFERENCE)?;
    let runs = reference["scenarios"][scenario]
        .as_array()
        .ok_or("the reference has no runs for this scenario")?;
    let mut rgb_runs = Vec::with_capacity(runs.len());
    for run in runs {
        let channels = run.as_array().ok_or("a reference run is not a list")?;
        let mut rgb = [0.0_f64; 3];
        for (slot, value) in rgb.iter_mut().zip(channels) {
            *slot = value.as_f64().ok_or("a reference value is not a number")?;
        }
        rgb_runs.push(rgb);
    }
    assert!(rgb_runs.len() >= 2, "{scenario}: the reference needs seeds");
    Ok(rgb_runs)
}

fn check_scenario(scenario: &str) -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "metal_sphere_lights.py", SCRIPT)?;
    let blend = root.path().join("scene.blend");
    let json_path = root.path().join("blender.json");
    // Blender only builds the scene (1 sample, no seeds to render); the reference is committed.
    let arguments: [&Path; 6] = [
        &blend,
        &json_path,
        Path::new(scenario),
        Path::new("1"),
        Path::new("0"),
        Path::new("16"),
    ];
    run_blender_arguments(&blender, &script, None, &arguments)?;

    let wanted_runs = reference_runs(scenario)?;
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
