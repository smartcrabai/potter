//! Voxel `volume_interpolation`. The oracle is `tests/fixtures/volume_interpolation`, rendered by Blender 5.2.2 with
//! `generate_volume_interpolation_oracle.py`: a 16³ hashed binary grid (`node_value` below) at density 2.0, single scatter
//! (`max_bounces = volume_bounces = 0`), a 4 × 4 frame spanning the unit cube, a one-pixel box filter, eight seeds at 8192 spp.
//!
//! The compared quantities are frame means (all 16 pixels). The discriminating quantity is the cubic-minus-linear delta,
//! `(cubic - linear) / cubic`. Blender's delta is 5.2 % ± 1.5 % over the eight seeds. The earlier 8.8 % came from four seeds
//! and is within noise of that. The analytic single-scatter quadrature for this grid gives 2.0 % (Potter's boundary cut) to
//! 2.6 % (Cycles' background fade), so a Potter delta far from 5 % would be a real difference.
//!
//! Each tolerance is three noise floors. A Potter render's relative spread is `POTTER_SD_*`, measured over four seeds at
//! `SAMPLES`. Blender's spreads are the standard errors of its eight-seed means, read from the oracle.

use std::{error::Error, fs, path::Path, process::Output};

use assert_cmd::Command;
use serde_json::{Value, json};
use tempfile::tempdir;

const SIZE: u32 = 4;
const N: usize = 16;
const ENERGY: f64 = 100.0;
const POSITION: [f64; 3] = [1.6, 0.6, 0.9];
const DENSITY_SCALE: f64 = 2.0;
/// Samples per pixel. A voxel render takes about 16 s at this count in a debug build.
const SAMPLES: u32 = 512;
/// Potter's single-render frame-mean spread, relative to the mode's mean, over four seeds at `SAMPLES` on this scene.
const POTTER_SD_CUBIC: f64 = 0.0132;
const POTTER_SD_LINEAR: f64 = 0.0135;
const MATCH_FLOORS: f64 = 3.0;

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

fn make_scene(root: &Path, interpolation: &str, seed: u32) -> Result<std::path::PathBuf, Box<dyn Error>> {
    std::fs::create_dir_all(root)?;
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
        json!({"op":"camera.create","id":"camera_main","transform":{"translation":[0.0,0.0,3.0]},"projection":"orthographic","ortho_scale":1.0}),
        json!({"op":"light.create","id":"light_main","light_type":"point","energy":ENERGY,"radius":0.0,"transform":{"translation":POSITION}}),
        json!({"op":"graph.create","id":"voxel_graph","kind":"shader"}),
        json!({"op":"graph.node_add","graph":"voxel_graph","id":"output","type":"ShaderNodeOutputMaterial"}),
        json!({"op":"graph.node_add","graph":"voxel_graph","id":"scatter","type":"ShaderNodeVolumeScatter","inputs":{"Color":[1.0,1.0,1.0,1.0],"Density":DENSITY_SCALE,"Anisotropy":0.0}}),
        json!({"op":"graph.link","graph":"voxel_graph","from_node":"scatter","from_socket":"Volume","to_node":"output","to_socket":"Volume"}),
        json!({"op":"material.create","id":"voxels","graph":"voxel_graph","volume_interpolation":interpolation}),
        json!({"op":"volume.create","id":"voxel_volume","materials":["voxels"],"grids":[{"dims":[N,N,N],"voxel_size":1.0 / 16.0,"origin":[-0.5,-0.5,-0.5],"values":values}]}),
        json!({"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"camera_main","world":"world_main"}}),
        json!({"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_x":SIZE,"resolution_y":SIZE,"samples":SAMPLES,"seed":seed,"max_bounces":0}}),
    ];
    let scene = root.join("scene");
    assert_cli_success(&pot().arg("init").arg(&scene).arg("--json").output()?)?;
    let batch = root.join("operations.json");
    fs::write(
        &batch,
        serde_json::to_vec(&json!({"schema_version": 1, "base_revision": 0, "operations": operations}))?,
    )?;
    assert_cli_success(&pot().arg("apply").arg(&scene).arg("--file").arg(&batch).arg("--json").output()?)?;
    Ok(scene)
}

struct Pixels {
    width: usize,
    red: Vec<f32>,
}

/// The mean red value over the whole 4 × 4 frame of one render of `interpolation` at `seed`.
fn frame_mean(interpolation: &str, seed: u32) -> Result<f64, Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = make_scene(&directory.path().join("scene_root"), interpolation, seed)?;
    let out = directory.path().join("render");
    assert_cli_success(
        &pot()
            .arg("render")
            .arg(&scene)
            .args(["--engine", "path", "--format", "exr", "--out"])
            .arg(&out)
            .arg("--json")
            .output()?,
    )?;
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
    let red = image.layer_data.channel_data.pixels.red;
    Ok(red.iter().map(|value| f64::from(*value)).sum::<f64>() / red.len() as f64)
}

/// Blender's frame means and their standard errors over the oracle's seeds, for CUBIC and LINEAR.
struct BlenderFrames {
    cubic: f64,
    cubic_se: f64,
    linear: f64,
    linear_se: f64,
}

fn blender_frames() -> Result<BlenderFrames, Box<dyn Error>> {
    let oracle: Value = serde_json::from_slice(&fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/volume_interpolation/blender_volume_interpolation.json"),
    )?)?;
    let frames_of = |case: &str| -> Result<(f64, f64), Box<dyn Error>> {
        let seeds = oracle["cases"][case]["seed_pixels"]
            .as_array()
            .ok_or("oracle seed pixels are missing")?;
        let mut frames = Vec::with_capacity(seeds.len());
        for seed in seeds {
            let pixels = seed.as_array().ok_or("oracle seed is not an array")?;
            let mut sum = 0.0;
            for pixel in pixels {
                sum += pixel.as_f64().ok_or("oracle pixel is not a number")?;
            }
            frames.push(sum / pixels.len() as f64);
        }
        let mean = frames.iter().sum::<f64>() / frames.len() as f64;
        let variance = frames.iter().map(|frame| (frame - mean).powi(2)).sum::<f64>() / (frames.len() - 1) as f64;
        Ok((mean, (variance / frames.len() as f64).sqrt()))
    };
    let (cubic, cubic_se) = frames_of("cubic_noise")?;
    let (linear, linear_se) = frames_of("linear_noise")?;
    Ok(BlenderFrames { cubic, cubic_se, linear, linear_se })
}

#[test]
fn cubic_frame_mean_matches_blender() -> Result<(), Box<dyn Error>> {
    let blender = blender_frames()?;
    let measured = frame_mean("cubic", 0)?;
    let floor = (POTTER_SD_CUBIC.powi(2) + (blender.cubic_se / blender.cubic).powi(2)).sqrt();
    let error = (measured - blender.cubic).abs() / blender.cubic;
    assert!(
        error < MATCH_FLOORS * floor,
        "CUBIC frame mean {measured:.6} vs Blender {:.6}: {:.2} % apart, over {:.2} % (three floors)",
        blender.cubic,
        error * 100.0,
        MATCH_FLOORS * floor * 100.0
    );
    Ok(())
}

#[test]
fn linear_frame_mean_matches_blender() -> Result<(), Box<dyn Error>> {
    let blender = blender_frames()?;
    let measured = frame_mean("linear", 0)?;
    let floor = (POTTER_SD_LINEAR.powi(2) + (blender.linear_se / blender.linear).powi(2)).sqrt();
    let error = (measured - blender.linear).abs() / blender.linear;
    assert!(
        error < MATCH_FLOORS * floor,
        "LINEAR frame mean {measured:.6} vs Blender {:.6}: {:.2} % apart, over {:.2} % (three floors)",
        blender.linear,
        error * 100.0,
        MATCH_FLOORS * floor * 100.0
    );
    Ok(())
}

/// The cubic-minus-linear delta `(cubic - linear) / cubic` of Potter against Blender. Each side's delta has the error
/// propagated from its two means, `d(delta) = sqrt((l / c² · σ_c)² + (σ_l / c)²)`. Potter's σ are its single-render spreads
/// (relative, times the mean); Blender's are the standard errors of its seed means. The tolerance is three floors of the
/// two deltas' combined spread.
#[test]
fn cubic_minus_linear_delta_matches_blender() -> Result<(), Box<dyn Error>> {
    let blender = blender_frames()?;
    let cubic = frame_mean("cubic", 0)?;
    let linear = frame_mean("linear", 0)?;
    let potter_delta = (cubic - linear) / cubic;
    let blender_delta = (blender.cubic - blender.linear) / blender.cubic;
    let delta_spread = |c: f64, l: f64, sigma_c: f64, sigma_l: f64| {
        ((l / (c * c) * sigma_c).powi(2) + (sigma_l / c).powi(2)).sqrt()
    };
    let potter_spread = delta_spread(cubic, linear, POTTER_SD_CUBIC * cubic, POTTER_SD_LINEAR * linear);
    let blender_spread = delta_spread(blender.cubic, blender.linear, blender.cubic_se, blender.linear_se);
    let tolerance = MATCH_FLOORS * (potter_spread.powi(2) + blender_spread.powi(2)).sqrt();
    let difference = (potter_delta - blender_delta).abs();
    assert!(
        difference < tolerance,
        "cubic-minus-linear delta: Potter {:.2} % vs Blender {:.2} %, {:.2} % apart, over {:.2} % (three floors)",
        potter_delta * 100.0,
        blender_delta * 100.0,
        difference * 100.0,
        tolerance * 100.0
    );
    Ok(())
}
