//! The binary alpha clip that Potter's `.blend` export writes, rendered by Potter against Blender 5.2.2 Cycles.
//!
//! Blender's glTF importer builds a MASK as `Alpha = SUBTRACT(1, LESS_THAN(x, cutoff))` on a DITHERED material
//! (`io_scene_gltf2/blender/imp/pbrMetallicRoughness.py` 668-700). The oracle plane is shaded by that graph, with the
//! U coordinate as `x`, so the cutoff falls on the plane's middle column and every pixel is fully opaque or fully
//! transparent, except the pixels next to the cutoff edge.
//!
//! Blender (`fixtures/native_oracle/blend_clip_mask_render.py`) saves the scene, then renders eight seeds at 16
//! samples per pixel. Each seed gives the fractions of a 32 × 32 window of the 64 × 64 frame that are opaque (alpha
//! at least 0.99), transparent (at most 0.01), or in between. Potter imports the same `.blend`, renders the same
//! eight seeds in process at 16 samples, and reads the same window from the alpha channel.
//!
//! Both renderers sample each pixel with a box filter one pixel wide: Blender's scene is set to `BOX` with width 1
//! (`pixel_filter_type`, `filter_width`), matching Potter's sampling. So the cutoff, which lies on a pixel boundary,
//! classifies whole columns on both sides, and the only difference left is Monte Carlo noise.
//!
//! The tolerance is `T · √(s_B² / n + s_P² / n)`, with `s` the seed-to-seed sample standard deviation of each side,
//! `n = 8`, and `T` the two-sided 99.9% t quantile at 7 degrees of freedom (5.408), as the glass tests use. When both
//! sides are deterministic at this edge the tolerance is zero, and the check is an exact comparison of the fractions.
//!
//! What the test must detect: a lost clip. Without the clip stage the alpha is the base alpha, a constant, so the
//! window would be all one value, and the opaque fraction would be 0 or 1, far outside the tolerance.

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

const SCRIPT: &str = include_str!("fixtures/native_oracle/blend_clip_mask_render.py");
/// Seeds of both renders. Eight seeds give 7 degrees of freedom for the Student quantile.
const SEEDS: u32 = 8;
const RESOLUTION: u32 = 64;
/// Blender's samples per pixel, set in the script; Potter renders the same count.
const SAMPLES: u32 = 16;
/// The centred window that the script and this test measure: pixels 16 to 47 on both axes.
const WINDOW: std::ops::Range<usize> = 16..48;
/// The two-sided 99.9% Student t quantile at 7 degrees of freedom.
const T_QUANTILE: f64 = 5.408;
/// The cutoff of the oracle's MASK graph (`blend_clip_mask_render.py`, `CUTOFF`).
const CLIP_CUTOFF: f64 = 0.5;
/// The opaque, transparent and in-between fractions, in that order.
const METRICS: [&str; 3] = ["opaque", "transparent", "fractional"];

/// Mean and standard error of the mean of `values` (sample standard deviation over `√n`).
fn mean_and_stderr(values: &[f64]) -> (f64, f64) {
    let count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / count;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (count - 1.0);
    (mean, (variance / count).sqrt())
}

/// The window fractions of one frame's alpha channel: opaque, transparent, and in between.
fn window_fractions(pixels: &[[f64; 4]], width: usize) -> [f64; 3] {
    let alphas: Vec<f64> = WINDOW
        .flat_map(|y| WINDOW.map(move |x| (y, x)))
        .map(|(y, x)| pixels[y * width + x][3])
        .collect();
    let count = alphas.len() as f64;
    let opaque = alphas.iter().filter(|&&a| a >= 0.99).count() as f64 / count;
    let transparent = alphas.iter().filter(|&&a| a <= 0.01).count() as f64 / count;
    [opaque, transparent, 1.0 - opaque - transparent]
}

/// The per-seed fractions of the Blender run, one vector per metric.
fn blender_metrics(expected: &Value) -> TestResult<[Vec<f64>; 3]> {
    let mut metrics = [Vec::new(), Vec::new(), Vec::new()];
    for seed in expected["seeds"]
        .as_array()
        .ok_or("the Blender seeds are missing")?
    {
        for (index, name) in METRICS.iter().enumerate() {
            metrics[index].push(
                seed[name]
                    .as_f64()
                    .ok_or("a Blender window fraction is not a number")?,
            );
        }
    }
    Ok(metrics)
}

#[test]
fn clip_mask_cutout_matches_cycles_window_statistics() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "blend_clip_mask_render.py", SCRIPT)?;
    let blend = root.path().join("scene.blend");
    let json_path = root.path().join("blender.json");
    let arguments: [&Path; 2] = [&blend, &json_path];
    run_blender_arguments(&blender, &script, None, &arguments)?;
    let expected: Value = serde_json::from_slice(&fs::read(&json_path)?)?;
    let wanted = blender_metrics(&expected)?;

    let (project_path, _) = import_blend(&blender, root.path(), "project", &blend)?;
    let project = Project::open(&project_path)?;
    let mut doc = project.doc().clone();
    // The oracle's `.blend` has no `potter.alpha_mode` marker, so this is the detection of Blender's MASK graph alone:
    // the plane's DITHERED material reads back as clip with the graph's cutoff as its threshold. Without that detection
    // the material would be opaque, and the render would not show the cutout.
    let clips: Vec<_> = doc
        .materials
        .values()
        .filter(|material| material.alpha_mode == "clip")
        .collect();
    assert_eq!(clips.len(), 1, "the plane's material must import as clip");
    assert!(
        (clips[0].alpha_threshold - CLIP_CUTOFF).abs() <= 1.0e-6,
        "imported threshold {} differs from the graph cutoff {CLIP_CUTOFF}",
        clips[0].alpha_threshold
    );
    let scene_id = doc.active_scene.clone();
    let (camera, frame) = {
        let scene = doc
            .scenes
            .get_mut(&scene_id)
            .ok_or("the active scene is missing")?;
        scene.render.resolution_x = RESOLUTION;
        scene.render.resolution_y = RESOLUTION;
        scene.render.resolution_percentage = 100;
        scene.render.samples = SAMPLES;
        scene.render.film_transparent = true;
        (scene.camera.clone(), scene.frame_current)
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
            true,
        )?;
        let bytes = fs::read(out.join("frame_0001.exr"))?;
        let (width, _, pixels) = decode_pixels(&bytes, ImageColorspace::Linear)?;
        let fractions = window_fractions(&pixels, width as usize);
        for (index, values) in potter.iter_mut().enumerate() {
            values.push(fractions[index]);
        }
    }

    for (index, name) in METRICS.iter().enumerate() {
        let (wanted_mean, wanted_stderr) = mean_and_stderr(&wanted[index]);
        let (actual_mean, actual_stderr) = mean_and_stderr(&potter[index]);
        let tolerance = T_QUANTILE * wanted_stderr.hypot(actual_stderr);
        assert!(
            (actual_mean - wanted_mean).abs() <= tolerance,
            "{name}: Potter {actual_mean} differs from Cycles {wanted_mean} by more than {tolerance}"
        );
    }
    Ok(())
}
