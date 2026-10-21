//! Texture Coordinate Generated on a cube whose texture space is manual (`use_auto_texspace` off), against Blender 5.2.2's
//! Cycles render (`fixtures/native_oracle/texture_space/manual_texspace_cube.py`), and a Blender round trip of the
//! stored texture space.
//!
//! The cube's stored texture space (location (0.5, -0.25, 0.1), size (2, 1, 1.5)) differs from its bounds, so the frame
//! mean depends on the stored values: the green channel is 0.125 per covered pixel from the stored location, where the
//! automatic bounds would give 0. The round trip imports the `.blend`, exports it again, and reads the exported mesh's
//! `use_auto_texspace`, `texspace_location` and `texspace_size` back through Blender.
//!
//! The tolerance is `T · √(s_B² / n + s_P² / n)`, with `s` the seed-to-seed sample standard deviations of each side and
//! `T` the two-sided 99.9% t quantile at 7 degrees of freedom (5.408), as in `native_subsurface_sphere.rs`.

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
    TestResult, assert_floats, dump_blend, export, import_blend, losses, run_blender_arguments,
    vec3, write_script,
};
use potter_core::eval::{EvaluationContext, EvaluationPurpose};
use potter_core::image::decode_pixels;
use potter_core::model::{Id, ImageColorspace};
use potter_core::render::render_sequence;
use potter_core::store::Project;
use serde_json::{Value, json};
use tempfile::tempdir;

const SCRIPT: &str = include_str!("fixtures/native_oracle/texture_space/manual_texspace_cube.py");
/// The stored texture space of the generator's cube.
const STORED_LOCATION: [f64; 3] = [0.5, -0.25, 0.1];
const STORED_SIZE: [f64; 3] = [2.0, 1.0, 1.5];
/// Seeds of both renders.
const SEEDS: u32 = 8;
const RESOLUTION: u32 = 16;
/// Samples per pixel on both sides (`manual_texspace_cube.py` renders at 64).
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

/// The per-seed frame means of the Blender render, as RGB channel vectors.
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
                    .ok_or("a Blender channel mean is not a number")?,
            );
        }
    }
    Ok(channels)
}

#[test]
fn manual_texture_space_matches_cycles() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "manual_texspace_cube.py", SCRIPT)?;
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

    for (channel, (wanted_values, actual_values)) in wanted.iter().zip(&potter).enumerate() {
        let (wanted_mean, wanted_stderr) = mean_and_stderr(wanted_values);
        let (actual_mean, actual_stderr) = mean_and_stderr(actual_values);
        let tolerance = T_QUANTILE * wanted_stderr.hypot(actual_stderr);
        assert!(
            (actual_mean - wanted_mean).abs() <= tolerance,
            "channel {channel}: Potter {actual_mean} differs from Cycles {wanted_mean} by more than {tolerance}"
        );
    }
    Ok(())
}

/// The stored texture space survives a Blender round trip: Potter exports the imported project to `.blend`, and
/// Blender reads the exported mesh back with its texture space off and the stored location and size.
#[test]
fn manual_texture_space_round_trips_through_blend() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "manual_texspace_cube.py", SCRIPT)?;
    let blend = root.path().join("scene.blend");
    let json_path = root.path().join("blender.json");
    let arguments: [&Path; 2] = [&blend, &json_path];
    run_blender_arguments(&blender, &script, None, &arguments)?;

    let (project_path, _) = import_blend(&blender, root.path(), "project", &blend)?;
    let exported_blend = root.path().join("roundtrip.blend");
    let exported = export(&blender, &project_path, "blend", &exported_blend)?;
    assert!(
        losses(&exported).is_empty(),
        "blend export must not lose the manual texture space: {exported}"
    );
    let dump = dump_blend(&blender, root.path(), &exported_blend, "roundtrip")?;
    let manual = dump["objects"]
        .as_array()
        .ok_or("the exported dump has no objects")?
        .iter()
        .map(|object| &object["texspace"])
        .find(|texspace| texspace["auto"] == json!(false))
        .ok_or("the exported mesh has no manual texture space")?;
    assert_floats(
        &vec3(&manual["location"])?,
        &STORED_LOCATION,
        1.0e-6,
        "texspace_location",
    );
    assert_floats(
        &vec3(&manual["size"])?,
        &STORED_SIZE,
        1.0e-6,
        "texspace_size",
    );
    Ok(())
}
