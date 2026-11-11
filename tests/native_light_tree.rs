//! Cycles light-tree probability parity for surface and volume next-event estimation.
//!
//! Blender captures use eight independent seeds, BOX filtering with width 1, denoising off, and a fixed sample count.
//! The first case also renders Potter with tree selection disabled and requires that flat-selection clamping miss the
//! Cycles tree oracle by more than the combined Welch bound.

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/native_oracle.rs"]
pub mod native_oracle;
#[path = "common/process.rs"]
mod process;

use std::{fs, path::Path};

use blender_checked::blender_executable;
use native_oracle::{
    TestResult, apply_operations, import_blend, pot, read_scene, run_blender_arguments,
    write_script,
};
use potter_core::{
    image::{ImageInterpolation, load_image_data, sample},
    model::Image,
};
use serde_json::{Value, json};
use tempfile::tempdir;

const GENERATOR: &str = include_str!("fixtures/native_oracle/light_tree/generate.py");
const SIZE: usize = 16;
const SAMPLES: u32 = 256;
const SEEDS: u32 = 8;
const SIGMAS: f64 = 4.0;
const SCENARIOS: &[&str] = &["multi_light_clamp", "sun_local_world", "volume_segment"];

fn mean_and_sigma(runs: &[[f64; 3]]) -> ([f64; 3], [f64; 3]) {
    let count = runs.len() as f64;
    let mut mean = [0.0; 3];
    for run in runs {
        for (sum, value) in mean.iter_mut().zip(run) {
            *sum += *value / count;
        }
    }
    let mut sigma = [0.0; 3];
    for channel in 0..3 {
        let variance = runs
            .iter()
            .map(|run| (run[channel] - mean[channel]).powi(2))
            .sum::<f64>()
            / (count - 1.0);
        sigma[channel] = variance.sqrt();
    }
    (mean, sigma)
}

fn patch_mean(project: &Path, image_id: &str) -> TestResult<[f64; 3]> {
    let scene = read_scene(project)?;
    let image: Image = serde_json::from_value(scene["images"][image_id].clone())?;
    let data = load_image_data(&image, project, ImageInterpolation::Closest)?;
    let mut sums = [0.0; 3];
    let mut count = 0.0;
    for y in 6..10 {
        for x in 6..10 {
            let pixel = sample(
                &data,
                [
                    (x as f64 + 0.5) / SIZE as f64,
                    (y as f64 + 0.5) / SIZE as f64,
                ],
                false,
            );
            for (sum, channel) in sums.iter_mut().zip(pixel.iter().take(3)) {
                *sum += *channel;
            }
            count += 1.0;
        }
    }
    Ok(sums.map(|sum| sum / count))
}

fn render_potter_runs(
    project: &Path,
    root: &Path,
    scene_id: &str,
    use_light_tree: bool,
) -> TestResult<Vec<[f64; 3]>> {
    let mut runs = Vec::with_capacity(SEEDS as usize);
    for seed in 0..SEEDS {
        let applied = apply_operations(
            project,
            root,
            json!([{
                "op": "render.update",
                "target": {"id": scene_id},
                "set": {
                    "samples": SAMPLES,
                    "seed": seed,
                    "use_light_tree": use_light_tree
                }
            }]),
        )?;
        assert_eq!(applied["ok"], true, "{applied}");
        let frames = root.join(format!("frames_{use_light_tree}_{seed}"));
        let rendered = pot()
            .arg("render")
            .arg(project)
            .args(["--engine", "path", "--format", "exr", "--out"])
            .arg(&frames)
            .arg("--json")
            .output()?;
        assert!(
            rendered.status.success(),
            "render failed: {}{}",
            String::from_utf8_lossy(&rendered.stdout),
            String::from_utf8_lossy(&rendered.stderr)
        );
        let image_id = format!("light_tree_{use_light_tree}_{seed}");
        let loaded = apply_operations(
            project,
            root,
            json!([{
                "op": "image.load",
                "id": image_id,
                "path": frames.join("frame_0001.exr").to_string_lossy()
            }]),
        )?;
        assert_eq!(loaded["ok"], true, "{loaded}");
        runs.push(patch_mean(project, &image_id)?);
    }
    Ok(runs)
}

fn blender_means(blender: &Path, root: &Path, scenario: &str) -> TestResult<Vec<[f64; 3]>> {
    let script = write_script(root, "light_tree_generate.py", GENERATOR)?;
    let blend = root.join(format!("{scenario}.blend"));
    let capture_path = root.join(format!("{scenario}.json"));
    let samples = SAMPLES.to_string();
    let seeds = SEEDS.to_string();
    let args = [
        blend.as_path(),
        capture_path.as_path(),
        Path::new(scenario),
        Path::new(&samples),
        Path::new(&seeds),
    ];
    run_blender_arguments(blender, &script, None, &args)?;
    let capture: Value = serde_json::from_slice(&fs::read(capture_path)?)?;
    let values = capture["patch_means"]
        .as_array()
        .ok_or("Blender light-tree patch means are missing")?;
    let means = values
        .iter()
        .map(|value| {
            let values = value
                .as_array()
                .ok_or("Blender light-tree patch mean is not an array")?;
            let mut rgb = [0.0; 3];
            for (channel, value) in rgb.iter_mut().zip(values) {
                *channel = value
                    .as_f64()
                    .ok_or("Blender light-tree patch mean is not numeric")?;
            }
            Ok(rgb)
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    assert_eq!(means.len(), SEEDS as usize);
    Ok(means)
}

fn assert_welch(actual: &[[f64; 3]], expected: &[[f64; 3]], label: &str) {
    let (actual_mean, actual_sigma) = mean_and_sigma(actual);
    let (expected_mean, expected_sigma) = mean_and_sigma(expected);
    eprintln!(
        "{label}: tree-on Potter mean {actual_mean:?} sigma {actual_sigma:?}; Blender mean {expected_mean:?} sigma {expected_sigma:?}"
    );
    for channel in 0..3 {
        let standard_error = (actual_sigma[channel].powi(2) / actual.len() as f64
            + expected_sigma[channel].powi(2) / expected.len() as f64)
            .sqrt();
        let tolerance = SIGMAS * standard_error;
        let error = (actual_mean[channel] - expected_mean[channel]).abs();
        assert!(
            error <= tolerance,
            "{label} channel {channel}: Potter {actual_mean:?} vs Cycles {expected_mean:?}; error {error} exceeds {SIGMAS}x Welch bound {tolerance}",
        );
    }
}

#[test]
fn blender_light_tree_surface_volume_oracles_match_and_flat_clamp_fails() -> TestResult<()> {
    let Some(blender) = blender_executable() else {
        return Ok(());
    };
    for scenario in SCENARIOS {
        let root = tempdir()?;
        let expected = blender_means(&blender, root.path(), scenario)?;
        let blend = root.path().join(format!("{scenario}.blend"));
        let (project, _) = import_blend(&blender, root.path(), "project", &blend)?;
        let scene = read_scene(&project)?;
        let scene_id = scene["active_scene"]
            .as_str()
            .ok_or("active scene is missing")?
            .to_owned();
        let render = &scene["scenes"][&scene_id]["render"];
        assert_eq!(render["pixel_filter_type"], "BOX");
        assert_eq!(render["filter_width"], 1.0);
        assert!(render["use_light_tree"].as_bool().unwrap_or(true));
        assert_eq!(
            render["sample_clamp_indirect"].as_f64().unwrap_or(10.0),
            10.0
        );

        let actual = render_potter_runs(&project, root.path(), &scene_id, true)?;
        if *scenario == "multi_light_clamp" {
            let flat = render_potter_runs(&project, root.path(), &scene_id, false)?;
            let (flat_mean, flat_sigma) = mean_and_sigma(&flat);
            let (cycles_mean, cycles_sigma) = mean_and_sigma(&expected);
            let red_error = (flat_mean[0] - cycles_mean[0]).abs();
            let red_bound = SIGMAS
                * (flat_sigma[0].powi(2) / flat.len() as f64
                    + cycles_sigma[0].powi(2) / expected.len() as f64)
                    .sqrt();
            eprintln!(
                "{scenario}: flat Potter mean {flat_mean:?} sigma {flat_sigma:?}; Blender mean {cycles_mean:?} sigma {cycles_sigma:?}; red error {red_error}, Welch bound {red_bound}"
            );
            assert!(
                red_error > red_bound,
                "flat light selection unexpectedly passed the clamp oracle: Potter {flat_mean:?}, Cycles {cycles_mean:?}, error {red_error}, Welch bound {red_bound}"
            );
        }
        if *scenario == "sun_local_world" {
            let flat = render_potter_runs(&project, root.path(), &scene_id, false)?;
            eprintln!("{scenario}: flat Potter mean {:?}", mean_and_sigma(&flat).0);
        }
        assert_welch(&actual, &expected, scenario);
    }
    Ok(())
}
