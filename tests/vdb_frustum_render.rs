#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/native_oracle.rs"]
#[expect(
    dead_code,
    reason = "this integration binary uses a subset of the shared native oracle helpers"
)]
mod native_oracle;
#[path = "common/process.rs"]
mod process;

use std::path::Path;

use native_oracle::{TestResult, apply_operations, import_blend, pot, read_scene};
use serde_json::{Value, json};
use tempfile::tempdir;

const SIZE: usize = 16;
const SEED_COUNT: usize = 4;
const MATCH_SIGMAS: f64 = 3.0;
const CYCLES_NUMERIC_FLOOR: f64 = 2.0e-3;
const MATERIAL_NAME: &str = "FrustumPrincipled";
const ORACLE_JSON: &str = include_str!("fixtures/vdb_frustum_render/oracle.json");

#[derive(Debug)]
struct FloatImage {
    width: usize,
    height: usize,
    rgb: Vec<[f64; 3]>,
}

fn fixture_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/vdb_frustum_render")
}

fn read_exr(path: &Path) -> TestResult<FloatImage> {
    let image = exr::prelude::read_first_rgba_layer_from_file(
        path,
        |resolution, _| FloatImage {
            width: resolution.width(),
            height: resolution.height(),
            rgb: vec![[0.0; 3]; resolution.width() * resolution.height()],
        },
        |pixels: &mut FloatImage, position, (red, green, blue, _alpha): (f32, f32, f32, f32)| {
            let index = position.y() * pixels.width + position.x();
            pixels.rgb[index] = [f64::from(red), f64::from(green), f64::from(blue)];
        },
    )?;
    let pixels = image.layer_data.channel_data.pixels;
    if pixels.width != SIZE || pixels.height != SIZE || pixels.rgb.len() != SIZE * SIZE {
        return Err(format!(
            "{} is {}x{}, expected {SIZE}x{SIZE}",
            path.display(),
            pixels.width,
            pixels.height
        )
        .into());
    }
    Ok(pixels)
}

fn mean_image(images: &[FloatImage]) -> Vec<[f64; 3]> {
    let mut mean = vec![[0.0; 3]; images[0].rgb.len()];
    for image in images {
        for (pixel, sample) in mean.iter_mut().zip(&image.rgb) {
            for channel in 0..3 {
                pixel[channel] += sample[channel] / images.len() as f64;
            }
        }
    }
    mean
}

/// The sample standard deviation across seed renders, pooled over pixels and RGB channels.
fn render_noise_rms(images: &[FloatImage]) -> f64 {
    let mean = mean_image(images);
    let squared_error = images
        .iter()
        .flat_map(|image| image.rgb.iter().zip(&mean))
        .flat_map(|(sample, average)| {
            (0..3).map(move |channel| (sample[channel] - average[channel]).powi(2))
        })
        .sum::<f64>();
    (squared_error / ((images.len() - 1) * mean.len() * 3) as f64).sqrt()
}

fn rmse(first: &[[f64; 3]], second: &[[f64; 3]]) -> f64 {
    assert_eq!(first.len(), second.len());
    let squared_error = first
        .iter()
        .zip(second)
        .flat_map(|(first, second)| {
            (0..3).map(move |channel| (first[channel] - second[channel]).powi(2))
        })
        .sum::<f64>();
    (squared_error / (first.len() * 3) as f64).sqrt()
}

fn subtract(first: &[[f64; 3]], second: &[[f64; 3]]) -> Vec<[f64; 3]> {
    first
        .iter()
        .zip(second)
        .map(|(first, second)| std::array::from_fn(|channel| first[channel] - second[channel]))
        .collect()
}

fn expected_mean(oracle: &Value, mode: &str) -> TestResult<Vec<[f64; 3]>> {
    let pixels: Vec<[f64; 3]> =
        serde_json::from_value(oracle["cycles"]["cases"][mode]["mean_rgb"].clone())?;
    if pixels.len() != SIZE * SIZE {
        return Err(format!("{mode} Cycles oracle has {} pixels", pixels.len()).into());
    }
    Ok(pixels)
}

fn native_cycles_frames(oracle: &Value, mode: &str) -> TestResult<Vec<FloatImage>> {
    let captures = oracle["cycles"]["cases"][mode]["captures"]
        .as_array()
        .ok_or_else(|| format!("{mode} Cycles captures are missing"))?;
    let mut frames = Vec::with_capacity(captures.len());
    for capture in captures {
        let filename = capture["file"]
            .as_str()
            .ok_or_else(|| format!("{mode} Cycles capture filename is missing"))?;
        frames.push(read_exr(&fixture_root().join(filename))?);
    }
    if frames.len() != SEED_COUNT {
        return Err(format!("{mode} Cycles oracle has {} seeds", frames.len()).into());
    }
    Ok(frames)
}

fn imported_scene(
    blender: &Path,
    root: &Path,
    name: &str,
) -> TestResult<(std::path::PathBuf, String, String)> {
    let blend = fixture_root().join("scene.blend");
    let (project, _) = import_blend(blender, root, name, &blend)?;
    let scene = read_scene(&project)?;
    let active_scene = scene["active_scene"]
        .as_str()
        .ok_or("imported active scene is missing")?
        .to_owned();
    let materials = scene["materials"]
        .as_object()
        .ok_or("imported materials are missing")?;
    let material_id = materials
        .iter()
        .find(|(_, material)| material["name"].as_str() == Some(MATERIAL_NAME))
        .map(|(id, _)| id.clone())
        .ok_or("imported Principled Volume material is missing")?;
    Ok((project, active_scene, material_id))
}

fn apply_render_settings(
    project: &Path,
    root: &Path,
    active_scene: &str,
    material_id: &str,
    mode: &str,
    seed: u32,
) -> TestResult {
    let applied = apply_operations(
        project,
        root,
        json!([
            {"op":"material.update","target":{"id":material_id},"set":{"volume_interpolation":mode}},
            {"op":"render.update","target":{"id":active_scene},"set":{
                "resolution_x":SIZE,
                "resolution_y":SIZE,
                "samples":64,
                "seed":seed,
                "max_bounces":0,
                "volume_bounces":0,
                "pixel_filter_type":"BOX",
                "filter_width":1.0
            }}
        ]),
    )?;
    if applied["ok"] != true {
        return Err(format!("could not set {mode} volume render settings: {applied}").into());
    }
    Ok(())
}

fn render_cli(
    project: &Path,
    root: &Path,
    engine: &str,
    label: &str,
) -> TestResult<std::path::PathBuf> {
    let output_dir = root.join(format!("frames_{label}"));
    let mut command = pot();
    command
        .arg("render")
        .arg(project)
        .args(["--engine", engine, "--format", "exr", "--out"])
        .arg(&output_dir)
        .arg("--json");
    let output = command.output()?;
    if !output.status.success() {
        return Err(std::io::Error::other(format!(
            "Potter {engine} render {label} failed: {}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ))
        .into());
    }
    Ok(output_dir.join("frame_0001.exr"))
}

fn potter_cycles_frames(blender: &Path, root: &Path, mode: &str) -> TestResult<Vec<FloatImage>> {
    let (project, active_scene, material_id) =
        imported_scene(blender, root, &format!("cycles_{mode}"))?;
    let mut frames = Vec::with_capacity(SEED_COUNT);
    for seed in 0..SEED_COUNT as u32 {
        apply_render_settings(&project, root, &active_scene, &material_id, mode, seed)?;
        let frame = render_cli(&project, root, "path", &format!("cycles_{mode}_{seed}"))?;
        frames.push(read_exr(&frame)?);
    }
    Ok(frames)
}

#[test]
fn frustum_vdb_path_sampling_matches_cycles() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let oracle: Value = serde_json::from_str(ORACLE_JSON)?;
    assert_eq!(oracle["blender_version"], "5.2.2 LTS");
    assert_eq!(oracle["source_vdb"]["map"], "NonlinearFrustumMap");
    assert_eq!(oracle["cycles"]["engine"], "CYCLES");
    assert_eq!(oracle["cycles"]["device"], "CPU");
    assert_eq!(oracle["cycles"]["view_transform"], "Raw");
    assert_eq!(oracle["resolution"], json!([SIZE, SIZE]));

    let root = tempdir()?;
    let mut native = Vec::with_capacity(2);
    let mut potter = Vec::with_capacity(2);
    let mut native_noise = Vec::with_capacity(2);
    let mut potter_noise = Vec::with_capacity(2);
    for mode in ["linear", "cubic"] {
        let native_frames = native_cycles_frames(&oracle, mode)?;
        let native_mean = mean_image(&native_frames);
        let recorded_mean = expected_mean(&oracle, mode)?;
        let recorded_error = rmse(&native_mean, &recorded_mean);
        assert!(
            recorded_error <= 1.0e-6,
            "{mode} JSON pixel oracle differs from its EXR captures by {recorded_error}"
        );

        let potter_frames = potter_cycles_frames(&blender, root.path(), mode)?;
        let potter_mean = mean_image(&potter_frames);
        let combined_se = (render_noise_rms(&native_frames).powi(2)
            + render_noise_rms(&potter_frames).powi(2))
        .sqrt()
            / (SEED_COUNT as f64).sqrt();
        let tolerance = MATCH_SIGMAS * combined_se + CYCLES_NUMERIC_FLOOR;
        let error = rmse(&potter_mean, &native_mean);
        assert!(
            error <= tolerance,
            "Potter {mode} Cycles image differs from Blender's affine-map oracle: RMSE {error:.6}, noise-aware tolerance {tolerance:.6}"
        );
        native_noise.push(render_noise_rms(&native_frames));
        potter_noise.push(render_noise_rms(&potter_frames));
        native.push(native_mean);
        potter.push(potter_mean);
    }

    let native_linear = &native[0];
    let native_cubic = &native[1];
    let potter_linear = &potter[0];
    let potter_cubic = &potter[1];
    let native_mode_signal = rmse(native_cubic, native_linear);
    let native_mode_noise =
        (native_noise[0].powi(2) + native_noise[1].powi(2)).sqrt() / (SEED_COUNT as f64).sqrt();
    assert!(
        native_mode_signal > 6.0 * native_mode_noise,
        "native Cycles fixture does not resolve linear versus cubic VDB sampling: signal {native_mode_signal:.6}, noise {native_mode_noise:.6}"
    );
    let native_mode_delta = subtract(native_cubic, native_linear);
    let potter_mode_delta = subtract(potter_cubic, potter_linear);
    let mode_delta_error = rmse(&potter_mode_delta, &native_mode_delta);
    let mode_delta_tolerance = MATCH_SIGMAS
        * (native_noise[0].powi(2)
            + native_noise[1].powi(2)
            + potter_noise[0].powi(2)
            + potter_noise[1].powi(2))
        .sqrt()
        / (SEED_COUNT as f64).sqrt()
        + CYCLES_NUMERIC_FLOOR;
    assert!(
        mode_delta_error <= mode_delta_tolerance,
        "linear/cubic frustum interpolation delta differs from Cycles: RMSE {mode_delta_error:.6}, noise-aware tolerance {mode_delta_tolerance:.6}"
    );

    Ok(())
}
