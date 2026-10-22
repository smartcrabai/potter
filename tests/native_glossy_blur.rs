//! Filter Glossy's glossy blur, against Blender 5.2.2's Cycles render (`fixtures/native_oracle/bump/glossy_blur.py`).
//!
//! A diffuse floor is lit only through a bounce off a roughness-0 metal wall, which reflects the light of a large
//! emitter that light sampling does not reach. The floor bounce has a cosine pdf below one, so Cycles blurs the wall's
//! specular closure (`surface_shader_prepare_closures`, `bsdf_microfacet_blur`): a sharp mirror becomes rough and spreads
//! the emitter's light differently. Blender's Filter Glossy 1.0 and 0.0 renders differ by about 20 times the tolerance
//! in the frame mean, so the case separates the blur. The test checks that separation, that the imported `.blend`
//! carries Filter Glossy 1.0, and that Potter matches Blender at 1.0 and at 0.0.
//!
//! The tolerance is `T · √(s_B² / n + s_P² / n)`, as in `native_subsurface_sphere.rs`.

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

const SCRIPT: &str = include_str!("fixtures/native_oracle/bump/glossy_blur.py");
/// Seeds of both renders.
const SEEDS: u32 = 8;
const RESOLUTION: u32 = 16;
/// Samples per pixel on both sides (`glossy_blur.py` renders at 64).
const SAMPLES: u32 = 64;
/// The two-sided 99.9% Student t quantile at 7 degrees of freedom (see the module docs).
const T_QUANTILE: f64 = 5.408;

/// Mean and standard error of the mean of `values` (sample standard deviation over `√n`).
fn mean_and_stderr(values: &[f64]) -> (f64, f64) {
    let count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / count;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (count - 1.0);
    (mean, (variance / count).sqrt())
}

/// The per-seed frame means of a Blender render under `key`, as RGB channel vectors.
fn blender_channels(expected: &Value, key: &str) -> TestResult<[Vec<f64>; 3]> {
    let mut channels = [Vec::new(), Vec::new(), Vec::new()];
    for seed in expected[key]
        .as_array()
        .ok_or("the Blender seed means are missing")?
    {
        let mean = seed.as_array().ok_or("a Blender seed mean is missing")?;
        for (channel, values) in channels.iter_mut().enumerate() {
            values.push(
                mean[channel]
                    .as_f64()
                    .ok_or("a Blender channel mean is not a number")?,
            );
        }
    }
    Ok(channels)
}

/// Potter's per-seed frame means of the imported scene rendered at `filter_glossy`, as RGB channel vectors.
fn potter_channels(
    project: &Project,
    root: &Path,
    filter_glossy: f64,
) -> TestResult<[Vec<f64>; 3]> {
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
        scene.render.filter_glossy = filter_glossy;
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
        let out = root.join(format!("frames_{filter_glossy}_{seed}"));
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
        let bytes = fs::read(out.join("frame_0001.exr"))?;
        let (_, _, pixels) = decode_pixels(&bytes, ImageColorspace::Linear)?;
        let count = pixels.len() as f64;
        for (channel, values) in potter.iter_mut().enumerate() {
            let total: f64 = pixels.iter().map(|pixel| pixel[channel]).sum();
            values.push(total / count);
        }
    }
    Ok(potter)
}

/// Asserts that Potter's per-seed means `actual` match Blender's `wanted` in every channel, within the t-quantile
/// tolerance of both seed spreads.
fn assert_matches(wanted: &[Vec<f64>; 3], actual: &[Vec<f64>; 3], setting: &str) {
    for (channel, (wanted_values, actual_values)) in wanted.iter().zip(actual).enumerate() {
        let (wanted_mean, wanted_stderr) = mean_and_stderr(wanted_values);
        let (actual_mean, actual_stderr) = mean_and_stderr(actual_values);
        let tolerance = T_QUANTILE * wanted_stderr.hypot(actual_stderr);
        assert!(
            (actual_mean - wanted_mean).abs() <= tolerance,
            "Filter Glossy {setting}, channel {channel}: Potter {actual_mean} differs from Cycles {wanted_mean} by more than {tolerance}"
        );
    }
}

#[test]
fn glossy_blur_matches_cycles() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "glossy_blur.py", SCRIPT)?;
    let blend = root.path().join("scene.blend");
    let json_path = root.path().join("blender.json");
    let arguments: [&Path; 2] = [&blend, &json_path];
    run_blender_arguments(&blender, &script, None, &arguments)?;
    let expected: Value = serde_json::from_slice(&fs::read(&json_path)?)?;
    let widened = blender_channels(&expected, "seed_means")?;
    let plain = blender_channels(&expected, "seed_means_filter_glossy_0")?;

    // The case must separate the blur in Blender: a Potter that ignores the blur fails below.
    for channel in [0, 1] {
        let (widened_mean, widened_stderr) = mean_and_stderr(&widened[channel]);
        let (plain_mean, plain_stderr) = mean_and_stderr(&plain[channel]);
        let tolerance = T_QUANTILE * widened_stderr.hypot(plain_stderr);
        assert!(
            (widened_mean - plain_mean).abs() > tolerance,
            "channel {channel}: Blender's Filter Glossy 1.0 ({widened_mean}) and 0.0 ({plain_mean}) are within \
             {tolerance}, so the case does not separate the blur"
        );
    }

    let (project_path, _) = import_blend(&blender, root.path(), "project", &blend)?;
    let project = Project::open(&project_path)?;
    let imported = project
        .doc()
        .scenes
        .get(&project.doc().active_scene)
        .ok_or("the active scene is missing")?
        .render
        .filter_glossy;
    assert!(
        (imported - 1.0).abs() < 1.0e-12,
        "the .blend's Filter Glossy imports as {imported}, not Blender's 1.0"
    );
    assert_matches(
        &widened,
        &potter_channels(&project, root.path(), 1.0)?,
        "1.0",
    );
    assert_matches(&plain, &potter_channels(&project, root.path(), 0.0)?, "0.0");
    Ok(())
}
