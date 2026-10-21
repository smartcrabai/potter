//! A sphere with each evaluated subsurface method (Burley, random walk, random walk skin, random walk legacy) under a
//! point and a disk light, against Blender 5.2.2's Cycles render (`fixtures/native_oracle/subsurface/subsurface_sphere.py`).
//!
//! Blender renders the sphere at 256 samples per pixel, eight seeds. Potter imports the same scene and renders eight
//! seeds on a 16 × 16 frame at `SAMPLES` samples per pixel, in-process with the scene's seed set in memory. Each side's
//! frame mean is taken over the whole frame, so both estimate the same expectation.
//!
//! The tolerance is `T · √(s_B² / n + s_P² / n)`, with `s` the seed-to-seed sample standard deviations of each side and
//! `T` the two-sided 99.9% t quantile at 7 degrees of freedom (5.408).
//!
//! What the test detects. The oracle's subsurface scale is 6.0, so every profile spreads light across the sphere.
//! Blender's frame means (red) are: Burley 0.15974, no subsurface 0.19963, random walk 0.04077, random walk skin 0.19571,
//! random walk legacy 0.17539. Burley against no subsurface differs by 0.0399 in red, and random walk against Burley by
//! 0.119. So an absent or wrongly scaled closure of any of the four methods is detected at any noise this test can reach.
//!
//! Cost: each case runs in about 12–17 s of test time on the lane (Blender's renders included), at `SAMPLES` = 32
//! samples per pixel on eight seeds, so each case stays well inside the 40 s budget.

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

use native_oracle::{TestResult, import_blend, run_blender_arguments, write_script};
use potter_core::eval::{EvaluationContext, EvaluationPurpose};
use potter_core::image::decode_pixels;
use potter_core::model::{Id, ImageColorspace};
use potter_core::render::render_sequence;
use potter_core::store::Project;
use serde_json::Value;
use tempfile::tempdir;

const SCRIPT: &str = include_str!("fixtures/native_oracle/subsurface/subsurface_sphere.py");
/// Seeds of both renders. Eight seeds keep the sample spread, and so the tolerance, stable.
const SEEDS: u32 = 8;
const RESOLUTION: u32 = 16;
/// Potter's samples per pixel. Blender renders at 256 (see `subsurface_sphere.py`).
const SAMPLES: u32 = 32;
/// The two-sided 99.9% Student t quantile at 7 degrees of freedom (see the module docs).
const T_QUANTILE: f64 = 5.408;

/// Mean and standard error of the mean of `values` (sample standard deviation over `√n`).
fn mean_and_stderr(values: &[f64]) -> (f64, f64) {
    let count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / count;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (count - 1.0);
    (mean, (variance / count).sqrt())
}

/// The per-seed frame means of the Blender case, as RGB channel vectors.
fn blender_channels(case: &Value) -> TestResult<[Vec<f64>; 3]> {
    let mut channels = [Vec::new(), Vec::new(), Vec::new()];
    for seed in case["seed_means"]
        .as_array()
        .ok_or("the Blender seed means are missing")?
    {
        let mean = seed.as_array().ok_or("a Blender seed mean is missing")?;
        for (channel, values) in channels.iter_mut().enumerate() {
            values.push(
                mean[channel]
                    .as_f64()
                    .ok_or("a Blender channel is not a number")?,
            );
        }
    }
    Ok(channels)
}

#[test]
fn burley_subsurface_sphere_matches_cycles() -> TestResult {
    matches_cycles("burley")
}

/// The Burley sphere with a zero bounce budget: Cycles still enters the surface and shades the exit, so the result
/// matches the full-budget render, and Potter must not cut the entry at the budget.
#[test]
fn burley_bounce0_subsurface_sphere_matches_cycles() -> TestResult {
    matches_cycles("burley_bounce0")
}

/// The Burley sphere over a floor with a budget of one. The entry takes no bounce, so the exit is the first scatter and
/// the floor is the second vertex: the light that leaves the exit bounces off the floor to the camera, and a bounce
/// counted at the entry would drop it (`subsurface_bounce` returns before `path_state_next`).
#[test]
fn burley_floor_bounce1_subsurface_sphere_matches_cycles() -> TestResult {
    matches_cycles("burley_floor_bounce1")
}

#[test]
fn random_walk_subsurface_sphere_matches_cycles() -> TestResult {
    matches_cycles("random_walk")
}

#[test]
fn random_walk_skin_subsurface_sphere_matches_cycles() -> TestResult {
    matches_cycles("random_walk_skin")
}

#[test]
fn random_walk_legacy_subsurface_sphere_matches_cycles() -> TestResult {
    matches_cycles("random_walk_legacy")
}

/// Renders the sphere of `case` (a case of `subsurface_sphere.py`) in Blender and in Potter at the case's bounce
/// budget, and compares the channel means of the seeds with the tolerance of the module docs.
fn matches_cycles(case: &str) -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "subsurface_sphere.py", SCRIPT)?;
    let blend = root.path().join("scene.blend");
    let json_path = root.path().join("blender.json");
    let arguments: [&Path; 3] = [&blend, &json_path, Path::new(case)];
    run_blender_arguments(&blender, &script, None, &arguments)?;
    let expected: Value = serde_json::from_slice(&fs::read(&json_path)?)?;
    let wanted = blender_channels(&expected["cases"][case])?;
    let budget = u32::try_from(
        expected["cases"][case]["max_bounces"]
            .as_u64()
            .ok_or("the Blender case has no max_bounces")?,
    )?;

    let (project_path, _) = import_blend(&blender, root.path(), "project", &blend)?;
    let project = Project::open(&project_path)?;
    let mut doc = project.doc().clone();
    let scene_id = doc.active_scene.clone();
    let (camera, frame, film_transparent) = {
        let scene = doc
            .scenes
            .get_mut(&scene_id)
            .ok_or("the active scene is missing")?;
        scene.render.resolution_x = RESOLUTION;
        scene.render.resolution_y = RESOLUTION;
        scene.render.resolution_percentage = 100;
        scene.render.samples = SAMPLES;
        scene.render.max_bounces = budget;
        (
            scene.camera.clone(),
            scene.frame_current,
            scene.render.film_transparent,
        )
    };
    let context = EvaluationContext {
        purpose: EvaluationPurpose::Render,
        ..EvaluationContext::default()
    };

    let mut potter = [Vec::new(), Vec::new(), Vec::new()];
    for seed in 0..SEEDS {
        doc.scenes
            .get_mut(&scene_id)
            .ok_or("the active scene is missing")?
            .render
            .seed = seed;
        let out = root.path().join(format!("frames_{seed}"));
        render_sequence(
            &doc,
            &context,
            project.path(),
            project.cache_directory(),
            camera.as_ref().map(Id::as_str),
            &[frame],
            "exr",
            &out,
            true,
            "path",
            RESOLUTION,
            RESOLUTION,
            SAMPLES,
            seed,
            film_transparent,
        )?;
        let bytes = std::fs::read(out.join("frame_0001.exr"))?;
        let (_, _, pixels) = decode_pixels(&bytes, ImageColorspace::Linear)?;
        let count = pixels.len() as f64;
        for (channel, values) in potter.iter_mut().enumerate() {
            let total: f64 = pixels.iter().map(|pixel| pixel[channel]).sum();
            values.push(total / count);
        }
    }

    for channel in 0..3 {
        let (wanted_mean, wanted_stderr) = mean_and_stderr(&wanted[channel]);
        let (actual_mean, actual_stderr) = mean_and_stderr(&potter[channel]);
        assert!(wanted_mean > 0.0, "the Blender frame mean is empty");
        let tolerance = T_QUANTILE * wanted_stderr.hypot(actual_stderr);
        assert!(
            (actual_mean - wanted_mean).abs() <= tolerance,
            "channel {channel}: Potter {actual_mean} differs from Cycles {wanted_mean} by more than {tolerance}"
        );
    }
    Ok(())
}
