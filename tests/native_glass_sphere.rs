//! A rough glass sphere under a point and a disk light, against Blender 5.2.2's Cycles render
//! (`fixtures/native_oracle/glass/glass_sphere.py`, roughness 0.2).
//!
//! Blender renders the sphere at 256 samples per pixel, eight seeds. Potter imports the same scene once, then
//! renders eight seeds at 32 samples per pixel on a 16 × 16 frame. The seeds are rendered in-process, with the
//! scene's seed set in memory, so no process starts per seed. Each side's frame mean is taken over the whole
//! frame, so both estimate the same expectation.
//!
//! The tolerance is `T · √(s_B² / n + s_P² / n)`, with `s` the seed-to-seed sample standard deviations of each
//! side and `T` the two-sided 99.9% t quantile at 7 degrees of freedom (5.408). Welch's degrees of freedom for
//! two 8-seed samples are at least 7, and the quantile falls as they grow, so `T` is conservative.
//!
//! Cost (measured on the lane, see the commit message): the import and Blender's render are fixed costs; the
//! eight Potter renders are the rest. The per-seed cost is set by the 16 × 16 × 32 render of about 120 sphere
//! pixels, not by process overhead.
//!
//! What the test must detect, as a margin per case (effect over tolerance, in frame-mean units). The figures are
//! Blender 5.2.2 renders of the same scene, with Potter's noise at 32 samples taken as Blender's at 32 samples, so
//! they are estimates until the lane run measures Potter's noise:
//!   - the old pass-through shadow rule, where the interior glass is lit through the sphere (Blender with the
//!     sphere casting no shadow): 0.0438 over a tolerance of 0.0085, a margin of 5.1×;
//!   - a roughness error in Potter's lobe (Potter at 0.5 where the reference is 0.2, the roughness swap of the
//!     two oracle roughnesses): 0.0238 over 0.0074, a margin of 3.2×. The plane oracle checks both roughnesses
//!     exactly.
//!
//! The sqrt(base) transmission tint is not checked here: at this noise its difference per channel (0.0013 to
//! 0.0018 at 64 samples) is below the tolerance. The plane oracle in `render/principled_bsdf.rs` (`glass_tests`)
//! checks the tint at 4 standard errors of Cycles' 16-seed plane means, where a base colour used directly in
//! place of its square root misses by 6 to 16 radiance units.
//!
//! The sphere's interior shadow rays are blocked by the glass, as Cycles blocks them
//! (`kernel/integrator/surface_shader.h`), so light reaches the camera through the front reflection, the
//! refracted path to the disk, and nothing from inside the sphere's own direct lighting.

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

const SCRIPT: &str = include_str!("fixtures/native_oracle/glass/glass_sphere.py");
/// Seeds of both renders. Eight seeds keep the sample spread, and so the tolerance, stable.
const SEEDS: u32 = 8;
const RESOLUTION: u32 = 16;
/// Potter's samples per pixel. Blender renders at 256 (see `glass_sphere.py`).
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

/// The per-seed frame means of the Blender run, as RGB channel vectors.
fn blender_channels(expected: &Value) -> TestResult<[Vec<f64>; 3]> {
    let mut channels = [Vec::new(), Vec::new(), Vec::new()];
    for seed in expected["seed_means"]
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
fn rough_glass_sphere_matches_cycles_at_roughness_0_2() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "glass_sphere.py", SCRIPT)?;
    let blend = root.path().join("scene.blend");
    let json_path = root.path().join("blender.json");
    let arguments: [&Path; 2] = [&blend, &json_path];
    run_blender_arguments(&blender, &script, None, &arguments)?;
    let expected: Value = serde_json::from_slice(&fs::read(&json_path)?)?;
    let wanted = blender_channels(&expected)?;

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
        scene.render.max_bounces = 3;
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
        let bytes = fs::read(out.join("frame_0001.exr"))?;
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
