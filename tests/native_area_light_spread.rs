//! A spread area light against Blender's Cycles render, off the light's axis.
//!
//! Blender renders a diffuse plane under a square area light whose spread confines its emission to a
//! cone, for several seeds, and Potter renders the same scene from the saved `.blend` for the same
//! seeds. Each render is reduced to the mean RGB of three column strips of the frame: one on the
//! light's axis, one about 0.4 m off it, and one about 0.9 m off it, where a narrow cone reaches only
//! the light's edge. The tolerance is the combined seed-to-seed standard error of the two renderers'
//! means. The spread also survives a `.blend` export and re-import as Blender reads it back.

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
    TestResult, apply_operations, export_raw, import_blend, pot, read_scene, run_blender,
    run_blender_arguments, write_script,
};
use potter_core::image::{ImageInterpolation, load_image_data, sample};
use potter_core::model::Image;
use serde_json::{Value, json};
use tempfile::tempdir;

const SCRIPT: &str = include_str!("fixtures/native_oracle/area_light_spread.py");
const DUMP_SCRIPT: &str = include_str!("fixtures/native_oracle/light_spread_dump.py");
/// Frame width in pixels, the same as `area_light_spread.py`.
const WIDTH: u32 = 16;
/// Inclusive pixel columns of the three strips, the same as `STRIPS` in `area_light_spread.py`.
const STRIPS: [(u32, u32); 3] = [(7, 8), (10, 11), (13, 14)];
/// Samples per pixel and seeds, the same for Blender and Potter.
const SAMPLES: u32 = 256;
/// Samples per pixel of the wide-spread case (see `a_wide_spread_matches_cycles`).
const WIDE_SPREAD_SAMPLES: u32 = 128;
const SEEDS: u32 = 4;
/// Standard errors of the combined means that the difference may span.
const SIGMAS: f64 = 3.0;

/// Mean RGB of each strip of the frame, from the rendered image.
fn strip_means(project: &Path, image_id: &str) -> TestResult<Vec<[f64; 3]>> {
    let scene = read_scene(project)?;
    let image: Image = serde_json::from_value(scene["images"][image_id].clone())?;
    let data = load_image_data(&image, project, ImageInterpolation::Closest)?;
    let mut means = Vec::new();
    for (first, last) in STRIPS {
        let mut total = [0.0_f64; 3];
        let mut count = 0.0_f64;
        for column in first..=last {
            for row in 0..WIDTH {
                let u = (f64::from(column) + 0.5) / f64::from(WIDTH);
                let v = (f64::from(row) + 0.5) / f64::from(WIDTH);
                let pixel = sample(&data, [u, v], false);
                for (sum, value) in total.iter_mut().zip(pixel.iter().take(3)) {
                    *sum += value;
                }
                count += 1.0;
            }
        }
        means.push(total.map(|sum| sum / count));
    }
    Ok(means)
}

/// The mean of per-seed means, and the standard error of that mean from the seed spread.
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

fn parse_rgb(value: &Value) -> TestResult<[f64; 3]> {
    let channels = value
        .as_array()
        .ok_or("a Blender strip mean is not a list")?;
    let mut rgb = [0.0_f64; 3];
    for (slot, channel) in rgb.iter_mut().zip(channels) {
        *slot = channel
            .as_f64()
            .ok_or("a Blender strip mean is not a number")?;
    }
    Ok(rgb)
}

fn check_spread(spread: f64, samples: u32) -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "area_light_spread.py", SCRIPT)?;
    let blend = root.path().join("scene.blend");
    let json_path = root.path().join("blender.json");
    let spread_arg = spread.to_string();
    let samples_arg = samples.to_string();
    let seeds_arg = SEEDS.to_string();
    let arguments: [&Path; 5] = [
        &blend,
        &json_path,
        Path::new(&spread_arg),
        Path::new(&samples_arg),
        Path::new(&seeds_arg),
    ];
    run_blender_arguments(&blender, &script, None, &arguments)?;

    let expected: Value = serde_json::from_slice(&fs::read(&json_path)?)?;
    let mut wanted_runs = Vec::new();
    for seed in expected["strip_means"]
        .as_array()
        .ok_or("the Blender strip means are missing")?
    {
        let mut strips = Vec::new();
        for strip in seed.as_array().ok_or("a Blender seed is not a list")? {
            strips.push(parse_rgb(strip)?);
        }
        wanted_runs.push(strips);
    }
    assert_eq!(
        wanted_runs.len(),
        SEEDS as usize,
        "one Blender run per seed"
    );

    let (project, _) = import_blend(&blender, root.path(), "project", &blend)?;
    // The spread survives a `.blend` round trip: Potter exports the imported scene, and Blender reads
    // the area light back with the same spread.
    let exported = root.path().join("roundtrip.blend");
    let report = export_raw(&blender, &project, "blend", &exported, true)?;
    assert!(
        report.status.success(),
        "blend export failed: {}",
        String::from_utf8_lossy(&report.stdout)
    );
    let dump_script = write_script(root.path(), "light_spread_dump.py", DUMP_SCRIPT)?;
    let dumped = root.path().join("roundtrip.json");
    run_blender(&blender, &dump_script, Some(&exported), &dumped)?;
    let lights: Value = serde_json::from_slice(&fs::read(&dumped)?)?;
    let area = lights["lights"]
        .as_array()
        .ok_or("the round-trip lights are missing")?
        .iter()
        .find(|light| light["type"] == "AREA")
        .ok_or("the round-trip area light is missing")?;
    let round_tripped = area["spread"]
        .as_f64()
        .ok_or("the round-trip spread is missing")?;
    assert!(
        (round_tripped - spread).abs() <= 1.0e-6,
        "spread {spread} came back from the .blend as {round_tripped}"
    );

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
        potter_runs.push(strip_means(&project, &image_id)?);
    }

    for strip in 0..STRIPS.len() {
        let wanted_strip: Vec<[f64; 3]> = wanted_runs.iter().map(|run| run[strip]).collect();
        let actual_strip: Vec<[f64; 3]> = potter_runs.iter().map(|run| run[strip]).collect();
        let (wanted, wanted_error) = mean_and_error(&wanted_strip);
        let (actual, actual_error) = mean_and_error(&actual_strip);
        let label = format!("spread {spread}, strip {strip}");
        if strip == 0 {
            assert!(
                wanted.iter().all(|value| *value > 0.1),
                "{label}: the Blender reference must be lit, got {wanted:?}"
            );
        }
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
    }
    Ok(())
}

/// The wide-spread case takes 128 samples. Its edge strip's four-seed standard error is 0.10% at 256
/// samples (measured on Blender's eight seeds) and about 0.14% to 0.2% at 128: square-root scaling
/// predicts 0.14%, and 64 samples measured 0.30%, above that. The edge tolerance therefore rises from
/// about 0.4% to between 0.6% and 0.9%. The on-axis strip stays under 0.3%.
#[test]
fn a_zero_spread_lights_only_under_the_emitter_as_cycles() -> TestResult {
    check_spread(0.0, SAMPLES)
}

#[test]
fn a_narrow_spread_matches_cycles() -> TestResult {
    check_spread(0.3, SAMPLES)
}

#[test]
fn a_wide_spread_matches_cycles() -> TestResult {
    check_spread(1.0, WIDE_SPREAD_SAMPLES)
}
