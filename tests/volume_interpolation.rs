//! Voxel `volume_interpolation`. The oracle is `tests/fixtures/volume_interpolation`, rendered by Blender 5.2.2 with
//! `generate_volume_interpolation_oracle.py`: a 16³ hashed binary grid (`node_value` below), imported as a Volume
//! object with CUBIC or LINEAR interpolation, a Volume Scatter at density 0.2, single scatter under a point light.
//! Cycles' CUBIC is the B-spline of `kernel_image_interp_tricubic_nanovdb`, which Potter's `cubic_weights` follows.
//!
//! Thresholds come from noise, not from a fixed tolerance. Each image is compared over the whole 24×24 frame. The
//! noise floor of a comparison is `sqrt(sigma_potter^2 + sigma_blender^2)`, where `sigma` is the per-pixel RMS of an
//! image's mean about its expectation, relative to the image mean. `sigma_blender` is the oracle's
//! (`noise.sigma_blender`, four seeds). `sigma_potter` is measured here from two seeds per mode, `sigma_single / sqrt(2)`.

use std::{error::Error, fs, path::Path, process::Output};

use assert_cmd::Command;
use serde_json::{Value, json};
use tempfile::tempdir;

const SIZE: u32 = 24;
const N: usize = 16;
const ENERGY: f64 = 100.0;
const POSITION: [f64; 3] = [1.6, 0.6, 0.9];
const DENSITY_SCALE: f64 = 0.2;
const SAMPLES: u32 = 32768;
/// Potter seeds per mode. The mean of the two has noise `sigma_single / sqrt(2)`.
const SEEDS: [u32; 2] = [0, 1];
/// A match is within this many noise floors (three RMS deviations of noise alone).
const MATCH_BANDS: f64 = 3.0;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn assert_cli_success(output: &Output) -> Result<Value, Box<dyn Error>> {
    if !output.status.success() {
        return Err(std::io::Error::other(format!(
            "CLI exited with {:?}: {}{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ))
        .into());
    }
    Ok(serde_json::from_slice(&output.stdout)?)
}

/// The grid value at node `(i, j, k)`: 1 where the hash is at least 512, else 0, as the oracle writes it.
fn node_value(i: i64, j: i64, k: i64) -> f64 {
    let hash = ((i * 73_856_093) ^ (j * 19_349_663) ^ (k * 83_492_791)) & 1023;
    if hash >= 512 { 1.0 } else { 0.0 }
}

fn make_scene(
    root: &Path,
    interpolation: &str,
    seed: u32,
) -> Result<std::path::PathBuf, Box<dyn Error>> {
    fs::create_dir_all(root)?;
    let mut values = Vec::with_capacity(N * N * N);
    for k in 0..N {
        for j in 0..N {
            for i in 0..N {
                values.push(node_value(i as i64, j as i64, k as i64));
            }
        }
    }
    let operations = vec![
        json!({"op":"world.create","id":"world_main","color":[0.0,0.0,0.0],"strength":0.0}),
        json!({"op":"camera.create","id":"camera_main","transform":{"translation":[0.0,0.0,3.0]},"projection":"orthographic","ortho_scale":2.0}),
        json!({"op":"light.create","id":"light_main","light_type":"point","energy":ENERGY,"radius":0.0,"transform":{"translation":POSITION}}),
        json!({"op":"graph.create","id":"voxel_graph","kind":"shader"}),
        json!({"op":"graph.node_add","graph":"voxel_graph","id":"output","type":"ShaderNodeOutputMaterial"}),
        json!({"op":"graph.node_add","graph":"voxel_graph","id":"scatter","type":"ShaderNodeVolumeScatter","inputs":{"Color":[1.0,1.0,1.0,1.0],"Density":DENSITY_SCALE,"Anisotropy":0.0}}),
        json!({"op":"graph.link","graph":"voxel_graph","from_node":"scatter","from_socket":"Volume","to_node":"output","to_socket":"Volume"}),
        json!({"op":"material.create","id":"voxels","graph":"voxel_graph","volume_interpolation":interpolation}),
        json!({"op":"volume.create","id":"voxel_volume","materials":["voxels"],"grids":[{"dims":[N,N,N],"voxel_size":1.0 / N as f64,"origin":[-0.5,-0.5,-0.5],"values":values}]}),
        json!({"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"camera_main","world":"world_main"}}),
        json!({"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_x":SIZE,"resolution_y":SIZE,"samples":SAMPLES,"seed":seed,"max_bounces":0}}),
    ];
    let scene = root.join("scene");
    let output = pot().arg("init").arg(&scene).arg("--json").output()?;
    assert_cli_success(&output)?;
    let batch = root.join("operations.json");
    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": operations,
        }))?,
    )?;
    let output = pot()
        .arg("apply")
        .arg(&scene)
        .arg("--file")
        .arg(&batch)
        .arg("--json")
        .output()?;
    assert_cli_success(&output)?;
    Ok(scene)
}

struct Pixels {
    width: usize,
    red: Vec<f32>,
}

/// The red channel of the whole rendered image, top-down: `position.y()` counts from the top row, as the oracle's
/// `row_order` says.
fn render_image(scene: &Path, out: &Path) -> Result<Vec<f64>, Box<dyn Error>> {
    let output = pot()
        .arg("render")
        .arg(scene)
        .args(["--engine", "path", "--format", "exr", "--out"])
        .arg(out)
        .arg("--json")
        .output()?;
    assert_cli_success(&output)?;
    let image = exr::prelude::read_first_rgba_layer_from_file(
        out.join("frame_0001.exr"),
        |resolution, _| Pixels {
            width: resolution.width(),
            red: vec![0.0_f32; resolution.width() * resolution.height()],
        },
        |pixels: &mut Pixels, position, (red, _green, _blue, _alpha): (f32, f32, f32, f32)| {
            pixels.red[position.y() * pixels.width + position.x()] = red;
        },
    )?;
    let pixels = image.layer_data.channel_data.pixels;
    Ok(pixels.red.iter().map(|value| f64::from(*value)).collect())
}

fn oracle() -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_slice(&fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/volume_interpolation/blender_volume_interpolation.json"),
    )?)?)
}

/// The oracle's per-pixel mean over its seeds, in top-down row order, for a case.
fn blender_pixels(oracle: &Value, case: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    oracle["cases"][case]["pixels"]
        .as_array()
        .ok_or("oracle pixels are missing")?
        .iter()
        .map(|value| value.as_f64().ok_or("oracle pixel is not a number"))
        .collect::<Result<Vec<f64>, &str>>()
        .map_err(Into::into)
}

fn blender_noise(oracle: &Value, case: &str) -> Result<f64, Box<dyn Error>> {
    oracle["noise"]["sigma_blender"][case]
        .as_f64()
        .ok_or_else(|| format!("oracle noise for {case} is missing").into())
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// The root-mean-square difference of two images, relative to the mean of the reference.
fn relative_rms(measured: &[f64], reference: &[f64]) -> f64 {
    let squares: f64 = measured
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    (squares / reference.len() as f64).sqrt() / mean(reference)
}

/// The per-render noise of two independent renders of one scene, relative to the mean of the first: the RMS of one
/// render about its expectation, since the difference of two has variance `2 * sigma^2`.
fn single_render_noise(first: &[f64], second: &[f64]) -> f64 {
    let squares: f64 = first
        .iter()
        .zip(second)
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    (squares / first.len() as f64 / 2.0).sqrt() / mean(first)
}

fn average(first: &[f64], second: &[f64]) -> Vec<f64> {
    first
        .iter()
        .zip(second)
        .map(|(a, b)| (a + b) / 2.0)
        .collect()
}

/// Renders the scene at each of `SEEDS` for one interpolation mode. Returns the seed mean and its noise,
/// `sigma_single / sqrt(2)`.
fn render_mode(directory: &Path, interpolation: &str) -> Result<(Vec<f64>, f64), Box<dyn Error>> {
    let mut runs = Vec::new();
    for seed in SEEDS {
        let root = directory.join(format!("{interpolation}_{seed}"));
        let scene = make_scene(&root, interpolation, seed)?;
        runs.push(render_image(&scene, &root.join("render"))?);
    }
    let sigma = single_render_noise(&runs[0], &runs[1]) / (SEEDS.len() as f64).sqrt();
    Ok((average(&runs[0], &runs[1]), sigma))
}

#[test]
fn cubic_and_linear_voxel_interpolation_match_blender_per_pixel() -> Result<(), Box<dyn Error>> {
    let oracle = oracle()?;
    let blender_cubic = blender_pixels(&oracle, "cubic_noise")?;
    let blender_linear = blender_pixels(&oracle, "linear_noise")?;
    let sigma_blender_cubic = blender_noise(&oracle, "cubic_noise")?;
    let sigma_blender_linear = blender_noise(&oracle, "linear_noise")?;
    let signal = oracle["noise"]["linear_vs_cubic_signal"]
        .as_f64()
        .ok_or("oracle signal is missing")?;

    let directory = tempdir()?;
    let (cubic, sigma_potter_cubic) = render_mode(directory.path(), "cubic")?;
    let (linear, sigma_potter_linear) = render_mode(directory.path(), "linear")?;

    let cubic_floor = (sigma_potter_cubic.powi(2) + sigma_blender_cubic.powi(2)).sqrt();
    let linear_floor = (sigma_potter_linear.powi(2) + sigma_blender_linear.powi(2)).sqrt();
    let cubic_error = relative_rms(&cubic, &blender_cubic);
    let linear_error = relative_rms(&linear, &blender_linear);
    assert!(
        cubic_error < MATCH_BANDS * cubic_floor,
        "CUBIC voxels: per-pixel error {:.2} % against Blender's CUBIC render, over {:.2} % (3 noise floors)",
        cubic_error * 100.0,
        MATCH_BANDS * cubic_floor * 100.0
    );
    assert!(
        linear_error < MATCH_BANDS * linear_floor,
        "LINEAR voxels: per-pixel error {:.2} % against Blender's LINEAR render, over {:.2} % (3 noise floors)",
        linear_error * 100.0,
        MATCH_BANDS * linear_floor * 100.0
    );

    // The modes must be distinguishable. A LINEAR render against Blender's CUBIC has the noise floor of that pair
    // and, if the modes differ, the Blender LINEAR-to-CUBIC signal on top. The threshold is halfway between the two.
    let cross_floor = (sigma_potter_linear.powi(2) + sigma_blender_cubic.powi(2)).sqrt();
    let expected_separation = (cross_floor.powi(2) + signal.powi(2)).sqrt();
    let threshold = cross_floor.midpoint(expected_separation);
    let linear_against_cubic = relative_rms(&linear, &blender_cubic);
    assert!(
        linear_against_cubic > threshold,
        "LINEAR voxels differ from Blender's CUBIC by {:.2} %, under the {:.2} % that separates the modes: the \
         interpolation mode is not taking effect",
        linear_against_cubic * 100.0,
        threshold * 100.0
    );
    Ok(())
}
