//! Glossy highlights of a disk area light against Blender's Cycles render.
//!
//! Blender renders a glossy plane under a disk light for several seeds, and Potter renders the same
//! scene for the same seeds. Each render is reduced to the mean RGB of the central patch of the
//! frame, where the plane mirrors the light. The tolerance is the combined seed-to-seed standard
//! error of the two renderers' means, so each side's own noise sets the bound. The sharp (roughness
//! 0.05) lobe under a small light is the case that BSDF-sampled hits on the light make converge.

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

const SCRIPT: &str = include_str!("fixtures/native_oracle/glossy_area_light.py");
/// Texel centres of the central 8×8 block of the 16 px frame, the same as `glossy_area_light.py`.
/// The block is symmetric under a row flip, so the mean does not depend on the row origin.
const PATCH: [f64; 8] = [
    4.5 / 16.0,
    5.5 / 16.0,
    6.5 / 16.0,
    7.5 / 16.0,
    8.5 / 16.0,
    9.5 / 16.0,
    10.5 / 16.0,
    11.5 / 16.0,
];
/// Samples per pixel of the two small-light cases.
const SAMPLES: u32 = 256;
/// Samples per pixel of the large sharp-gloss case. Its four-seed mean's standard error rises from
/// 0.55% at 256 samples to about 0.63% at 192, so its tolerance goes from about 2.3% to 2.7%.
const SHARP_LARGE_SAMPLES: u32 = 192;
/// Samples per pixel of the large broad-gloss case: its seed spread is the same at 64 as at 256.
const BROAD_LARGE_SAMPLES: u32 = 64;
const SEEDS: u32 = 4;
/// Standard errors of the combined means that the difference may span.
const SIGMAS: f64 = 3.0;

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

fn check_scenario(roughness: f64, light_size: f64, samples: u32) -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "glossy_area_light.py", SCRIPT)?;
    let blend = root.path().join("scene.blend");
    let json_path = root.path().join("blender.json");
    let roughness_arg = roughness.to_string();
    let size_arg = light_size.to_string();
    let samples_arg = samples.to_string();
    let seeds_arg = SEEDS.to_string();
    let arguments: [&Path; 6] = [
        &blend,
        &json_path,
        Path::new(&roughness_arg),
        Path::new(&size_arg),
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
    let label = format!("roughness {roughness}, light size {light_size}");
    assert!(
        wanted.iter().all(|value| *value > 0.1),
        "{label}: the Blender reference must be lit, got {wanted:?}"
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
                {"op": "render.update", "target": {"id": active}, "set": {"samples": samples, "seed": seed, "max_bounces": 1}}
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
    // Seed spread (sample standard deviation of the per-seed patch means, red channel) for the
    // before/after comparison; `--nocapture` shows it.
    let spread = |runs: &[[f64; 3]]| {
        let mean = runs.iter().map(|run| run[0]).sum::<f64>() / runs.len() as f64;
        (runs.iter().map(|run| (run[0] - mean).powi(2)).sum::<f64>() / (runs.len() as f64 - 1.0))
            .sqrt()
    };
    eprintln!(
        "POTTER_SPREAD {label}: potter mean {:.5} sd {:.5} | cycles mean {:.5} sd {:.5}",
        actual[0],
        spread(&potter_runs),
        wanted[0],
        spread(&wanted_runs),
    );

    for channel in 0..3 {
        let tolerance =
            SIGMAS * (wanted_error[channel].powi(2) + actual_error[channel].powi(2)).sqrt();
        assert!(
            (actual[channel] - wanted[channel]).abs() <= tolerance,
            "{label} channel {channel}: Potter {} differs from Cycles {} by more than {tolerance} \
             ({SIGMAS} combined standard errors; Potter {}, Cycles {})",
            actual[channel],
            wanted[channel],
            actual_error[channel],
            wanted_error[channel],
        );
    }
    Ok(())
}

/// The sample counts are per case (see the constants). The broad large-light case's seed spread is
/// 0.11% of the mean at both 64 and 256 samples (eight Blender seeds, measured headless); the sharp
/// large-light case goes to 192 with its tolerance up from about 2.3% to 2.7%.
#[test]
fn small_light_on_a_sharp_gloss_matches_cycles() -> TestResult {
    check_scenario(0.05, 0.2, SAMPLES)
}

#[test]
fn large_light_on_a_sharp_gloss_matches_cycles() -> TestResult {
    check_scenario(0.05, 1.0, SHARP_LARGE_SAMPLES)
}

#[test]
fn small_light_on_a_broad_gloss_matches_cycles() -> TestResult {
    check_scenario(0.24, 0.2, SAMPLES)
}

#[test]
fn large_light_on_a_broad_gloss_matches_cycles() -> TestResult {
    check_scenario(0.24, 1.0, BROAD_LARGE_SAMPLES)
}
