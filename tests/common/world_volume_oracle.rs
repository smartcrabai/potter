use std::{fs, path::Path};

use crate::{
    blender_checked::blender_executable,
    native_oracle::{
        TestResult, apply_operations, checked, import_blend, loss_features, pot, read_scene,
        run_blender_arguments, write_script,
    },
};
use serde_json::{Value, json};
use tempfile::tempdir;

const SIZE: u32 = 16;
const SEEDS: usize = 8;
const WORLD_VOLUME_SCRIPT: &str = include_str!("../fixtures/native_oracle/world_volume.py");
const WELCH_T_95: [f64; 8] = [
    2.3646, 2.3060, 2.2622, 2.2281, 2.2010, 2.1788, 2.1604, 2.1448,
];

fn potter_sample(
    project: &Path,
    root: &Path,
    scene_id: &str,
    case: &str,
    samples: u32,
    seed: u32,
) -> TestResult<[f64; 3]> {
    let applied = apply_operations(
        project,
        root,
        json!([{
            "op": "render.update",
            "target": {"id": scene_id},
            "set": {
                "resolution_x": SIZE,
                "resolution_y": SIZE,
                "samples": samples,
                "seed": seed,
                "max_bounces": 4,
                "diffuse_bounces": 4,
                "glossy_bounces": 4,
                "transmission_bounces": 4,
                "volume_bounces": 4,
                "transparent_max_bounces": 8,
                "min_light_bounces": 0,
                "min_transparent_bounces": 0,
                "sample_clamp_direct": 0.0,
                "sample_clamp_indirect": 0.0,
                "pixel_filter_type": "BOX",
                "filter_width": 1.0
            }
        }]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");
    let output_directory = root.join(format!("potter_{case}_{seed}"));
    let mut command = pot();
    command
        .arg("render")
        .arg(project)
        .args(["--engine", "path", "--format", "exr", "--out"])
        .arg(&output_directory)
        .arg("--json");
    checked(&mut command, "Potter World Volume render")?;
    image_mean(&output_directory.join("frame_0001.exr"))
}

struct RgbImage {
    width: usize,
    height: usize,
    pixels: Vec<[f32; 3]>,
}

fn image_mean(path: &Path) -> TestResult<[f64; 3]> {
    let image = exr::prelude::read_first_rgba_layer_from_file(
        path,
        |resolution, _| RgbImage {
            width: resolution.width(),
            height: resolution.height(),
            pixels: vec![[0.0_f32; 3]; resolution.width() * resolution.height()],
        },
        |pixels: &mut RgbImage, position, (red, green, blue, _alpha): (f32, f32, f32, f32)| {
            pixels.pixels[position.y() * pixels.width + position.x()] = [red, green, blue];
        },
    )?;
    let pixels = image.layer_data.channel_data.pixels;
    if pixels.width != SIZE as usize || pixels.height != SIZE as usize {
        return Err(format!("{} is not a {SIZE}x{SIZE} image", path.display()).into());
    }
    let count = (SIZE * SIZE) as f64;
    let mut mean = [0.0; 3];
    for pixel in pixels.pixels {
        for channel in 0..3 {
            mean[channel] += f64::from(pixel[channel]) / count;
        }
    }
    Ok(mean)
}

fn mean_variance(samples: &[[f64; 3]], channel: usize) -> (f64, f64) {
    let count = samples.len() as f64;
    let mean = samples.iter().map(|sample| sample[channel]).sum::<f64>() / count;
    let variance = samples
        .iter()
        .map(|sample| (sample[channel] - mean).powi(2))
        .sum::<f64>()
        / (count - 1.0);
    (mean, variance)
}

fn assert_welch_compatible(case: &str, blender: &[[f64; 3]], potter: &[[f64; 3]]) {
    for channel in 0..3 {
        let (blender_mean, blender_variance) = mean_variance(blender, channel);
        let (potter_mean, potter_variance) = mean_variance(potter, channel);
        let blender_term = blender_variance / blender.len() as f64;
        let potter_term = potter_variance / potter.len() as f64;
        let combined_error = (blender_term + potter_term).sqrt();
        let welch_degrees = (blender_term + potter_term).powi(2)
            / (blender_term.powi(2) / (blender.len() as f64 - 1.0)
                + potter_term.powi(2) / (potter.len() as f64 - 1.0));
        let statistic = (blender_mean - potter_mean).abs() / combined_error;
        let conservative_df = (welch_degrees.floor() as usize).clamp(7, 14);
        let critical = WELCH_T_95[conservative_df - 7];
        eprintln!(
            "{case} channel {channel}: Blender mean={blender_mean:.6}, Potter mean={potter_mean:.6}, Welch t={statistic:.3}, df={welch_degrees:.2}, critical={critical:.3}"
        );
        assert!(
            statistic <= critical,
            "{case} channel {channel}: Welch t={statistic:.3}, df={welch_degrees:.2}, critical={critical:.3}; Blender mean={blender_mean:.6}, Potter mean={potter_mean:.6}; Blender seeds={blender:?}; Potter seeds={potter:?}"
        );
    }
}

fn assert_case_matches(
    blender: &Path,
    root: &Path,
    script: &Path,
    case: &str,
    samples: u32,
) -> TestResult {
    let blend = root.join(format!("{case}.blend"));
    let expected = root.join(format!("{case}.json"));
    let samples_argument = samples.to_string();
    run_blender_arguments(
        blender,
        script,
        None,
        &[
            blend.as_path(),
            expected.as_path(),
            Path::new(case),
            Path::new(&samples_argument),
        ],
    )?;
    let oracle: Value = serde_json::from_slice(&fs::read(&expected)?)?;
    assert_eq!(oracle["samples"].as_u64(), Some(u64::from(samples)));
    let blender_samples = oracle["per_seed_files"]
        .as_array()
        .ok_or("Blender World Volume oracle has no per-seed images")?
        .iter()
        .map(|file| {
            let file = file
                .as_str()
                .ok_or("Blender World Volume oracle image path is not a string")?;
            image_mean(&root.join(file))
        })
        .collect::<TestResult<Vec<_>>>()?;
    assert_eq!(blender_samples.len(), SEEDS);
    let (project, report) = import_blend(blender, root, case, &blend)?;
    assert!(
        loss_features(&report)
            .iter()
            .all(|feature| { feature != "render.path.world_volume" }),
        "World.Volume import reported a path loss: {report}"
    );
    let scene = read_scene(&project)?;
    let scene_id = scene["active_scene"]
        .as_str()
        .ok_or("active scene is missing")?;
    let mut potter_samples = Vec::with_capacity(SEEDS);
    for seed in 0..SEEDS {
        potter_samples.push(potter_sample(
            &project,
            root,
            scene_id,
            case,
            samples,
            seed as u32,
        )?);
    }
    assert_welch_compatible(case, &blender_samples, &potter_samples);
    Ok(())
}

pub fn run_cases(cases: &[&str], samples: u32) -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping World Volume parity; no executable Blender was found");
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "world_volume.py", WORLD_VOLUME_SCRIPT)?;
    for &case in cases {
        assert_case_matches(&blender, root.path(), &script, case, samples)?;
    }
    Ok(())
}
