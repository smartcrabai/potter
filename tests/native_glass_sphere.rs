//! A rough glass sphere under a point and a disk light, against Blender 5.2.2's Cycles render
//! (`fixtures/native_oracle/glass/glass_sphere.py`).
//!
//! Blender and Potter each render the same scene with four seeds. Each side's frame mean is the
//! average over the whole frame, so the two estimators share their expectation. The tolerance is four
//! times the standard error of the difference of the two four-seed means, the standard errors taken
//! from each side's seed-to-seed spread, so it is the Monte Carlo noise of the comparison and nothing
//! looser.
//!
//! The sphere's interior shadow rays are blocked by the glass, as Cycles blocks them
//! (`kernel/integrator/surface_shader.h`), so light reaches the camera through the front reflection,
//! the refracted path to the disk, and nothing from inside the sphere's own direct lighting.

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

const SCRIPT: &str = include_str!("fixtures/native_oracle/glass/glass_sphere.py");
/// Seeds of both renders. Four gives a usable seed-to-seed spread per side.
const SEEDS: [u32; 4] = [0, 1, 2, 3];
const RESOLUTION: u32 = 16;
const SAMPLES: u32 = 256;

/// Mean and standard error of the mean of `values` (sample standard deviation over `√n`).
fn mean_and_stderr(values: &[f64]) -> (f64, f64) {
    let count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / count;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (count - 1.0);
    (mean, (variance / count).sqrt())
}

/// The frame mean RGB of a loaded render.
fn frame_mean(data: &potter_core::image::ImageData) -> [f64; 3] {
    let mut total = [0.0_f64; 3];
    for y in 0..RESOLUTION {
        for x in 0..RESOLUTION {
            let u = (f64::from(x) + 0.5) / f64::from(RESOLUTION);
            let v = (f64::from(y) + 0.5) / f64::from(RESOLUTION);
            let pixel = sample(data, [u, v], false);
            for (sum, value) in total.iter_mut().zip(pixel.iter()) {
                *sum += value;
            }
        }
    }
    let count = f64::from(RESOLUTION * RESOLUTION);
    total.map(|sum| sum / count)
}

/// Renders the sphere of `kind` in Blender and in Potter and compares the four-seed frame means.
fn check_kind(kind: &str) -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "glass_sphere.py", SCRIPT)?;
    let blend = root.path().join("scene.blend");
    let json_path = root.path().join("blender.json");
    let arguments: [&Path; 3] = [&blend, &json_path, Path::new(kind)];
    run_blender_arguments(&blender, &script, None, &arguments)?;

    let expected: Value = serde_json::from_slice(&fs::read(&json_path)?)?;
    let blender_seeds = expected["seed_means"]
        .as_array()
        .ok_or("the Blender seed means are missing")?;
    let mut blender_means = vec![Vec::new(); 3];
    for seed in blender_seeds {
        let mean = seed.as_array().ok_or("a Blender seed mean is missing")?;
        for (channel, values) in blender_means.iter_mut().enumerate() {
            values.push(
                mean[channel]
                    .as_f64()
                    .ok_or("a Blender channel is not a number")?,
            );
        }
    }

    let (project, _) = import_blend(&blender, root.path(), "project", &blend)?;
    let scene = read_scene(&project)?;
    let active = scene["active_scene"]
        .as_str()
        .ok_or("active scene is missing")?
        .to_owned();
    let mut potter_means = vec![Vec::new(); 3];
    for seed in SEEDS {
        let applied = apply_operations(
            &project,
            root.path(),
            json!([
                {"op": "render.update", "target": {"id": active}, "set": {"resolution_x": RESOLUTION, "resolution_y": RESOLUTION, "samples": SAMPLES, "seed": seed, "max_bounces": 3}}
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
        let id = format!("render_{seed}");
        let loaded = apply_operations(
            &project,
            root.path(),
            json!([{"op": "image.load", "id": id, "path": frames.join("frame_0001.exr").to_string_lossy()}]),
        )?;
        assert_eq!(loaded["ok"], true, "{loaded}");

        let scene = read_scene(&project)?;
        let image: Image = serde_json::from_value(scene["images"][&id].clone())?;
        let data = load_image_data(&image, &project, ImageInterpolation::Closest)?;
        let mean = frame_mean(&data);
        for (channel, values) in potter_means.iter_mut().enumerate() {
            values.push(mean[channel]);
        }
    }

    for channel in 0..3 {
        let (wanted, wanted_stderr) = mean_and_stderr(&blender_means[channel]);
        let (actual, actual_stderr) = mean_and_stderr(&potter_means[channel]);
        assert!(wanted > 0.0, "{kind}: the Blender frame mean is empty");
        let tolerance = 4.0 * wanted_stderr.hypot(actual_stderr);
        assert!(
            (actual - wanted).abs() <= tolerance,
            "{kind} channel {channel}: Potter {actual} differs from Cycles {wanted} by more than {tolerance} (4 standard errors)"
        );
    }
    Ok(())
}

#[test]
fn rough_glass_sphere_at_roughness_0_2_matches_cycles() -> TestResult {
    check_kind("r02")
}

#[test]
fn rough_glass_sphere_at_roughness_0_5_matches_cycles() -> TestResult {
    check_kind("r05")
}
