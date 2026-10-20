//! Voxel `volume_interpolation`. The oracle is `tests/fixtures/volume_interpolation`, rendered by Blender 5.2.2 with
//! `generate_volume_interpolation_oracle.py`: a 16³ hashed binary grid (`node_value` below) at density 2.0, single scatter
//! (`max_bounces = volume_bounces = 0`), a 4 × 4 frame spanning the unit cube, a one-pixel box filter, four seeds at 4096 spp.
//!
//! The comparison is the frame mean (all 16 pixels). At this density the cubic and linear frame means differ by about 8.8 %
//! in Blender, far above Blender's noise on the mean (about 1.4 %). The per-pixel comparison is too noisy for the budget.
//!
//! Thresholds are noise floors. A single Potter render has a frame-mean spread of `POTTER_SD_CUBIC` or `POTTER_SD_LINEAR`
//! (measured over four seeds at `SAMPLES`, see the constants). Blender's frame-mean noise is its per-pixel sigma over the
//! four seeds divided by √16. A match is within three floors, and the discrimination check requires LINEAR against Blender's
//! CUBIC to exceed the midpoint between the cross floor and the expected separation.

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
/// Potter's single-render frame-mean spread, relative, measured over four seeds at `SAMPLES` on this scene.
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
        serde_json::to_vec(
            &json!({"schema_version": 1, "base_revision": 0, "operations": operations}),
        )?,
    )?;
    assert_cli_success(
        &pot()
            .arg("apply")
            .arg(&scene)
            .arg("--file")
            .arg(&batch)
            .arg("--json")
            .output()?,
    )?;
    Ok(scene)
}

struct Pixels {
    width: usize,
    red: Vec<f32>,
}

/// The mean red value over the whole 4 × 4 frame of one render at `seed`.
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

/// Blender's frame means and frame-mean noise from the oracle: `(cubic, linear, sigma_cubic, sigma_linear)`. The frame
/// mean's noise is the four-seed per-pixel sigma divided by √16, since the sixteen pixels' noise is independent.
fn blender_frame() -> Result<(f64, f64, f64, f64), Box<dyn Error>> {
    let oracle: Value = serde_json::from_slice(&fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/volume_interpolation/blender_volume_interpolation.json"),
    )?)?;
    let mean = |case: &str| {
        oracle["cases"][case]["mean"]
            .as_f64()
            .ok_or_else(|| format!("oracle mean for {case} is missing"))
    };
    let sigma = |case: &str| {
        oracle["noise"]["sigma_blender"][case]
            .as_f64()
            .map(|pixel| pixel / 4.0)
            .ok_or_else(|| format!("oracle sigma for {case} is missing"))
    };
    Ok((
        mean("cubic_noise")?,
        mean("linear_noise")?,
        sigma("cubic_noise")?,
        sigma("linear_noise")?,
    ))
}

#[test]
fn cubic_frame_mean_matches_blender() -> Result<(), Box<dyn Error>> {
    let (cubic, _linear, sigma_cubic, _sigma_linear) = blender_frame()?;
    let measured = frame_mean("cubic", 0)?;
    let floor = (POTTER_SD_CUBIC.powi(2) + sigma_cubic.powi(2)).sqrt();
    let error = (measured - cubic).abs() / cubic;
    assert!(
        error < MATCH_FLOORS * floor,
        "CUBIC frame mean {measured:.6} vs Blender {cubic:.6}: {:.2} % apart, over {:.2} % (three floors)",
        error * 100.0,
        MATCH_FLOORS * floor * 100.0
    );
    Ok(())
}

#[test]
fn linear_frame_mean_matches_blender() -> Result<(), Box<dyn Error>> {
    let (_cubic, linear, _sigma_cubic, sigma_linear) = blender_frame()?;
    let measured = frame_mean("linear", 0)?;
    let floor = (POTTER_SD_LINEAR.powi(2) + sigma_linear.powi(2)).sqrt();
    let error = (measured - linear).abs() / linear;
    assert!(
        error < MATCH_FLOORS * floor,
        "LINEAR frame mean {measured:.6} vs Blender {linear:.6}: {:.2} % apart, over {:.2} % (three floors)",
        error * 100.0,
        MATCH_FLOORS * floor * 100.0
    );
    Ok(())
}

/// The modes must be distinguishable in the frame mean: Potter's LINEAR render must sit further from Blender's CUBIC mean than
/// the midpoint between the cross noise floor and the expected separation. The expected separation is the Blender
/// cubic-to-linear difference (`delta`), with its noise removed, combined with the cross floor.
#[test]
fn linear_frame_mean_is_distinguishable_from_blender_cubic() -> Result<(), Box<dyn Error>> {
    let (cubic, linear, sigma_cubic, sigma_linear) = blender_frame()?;
    let delta = (cubic - linear).abs() / cubic;
    let signal = (delta.powi(2) - sigma_cubic.powi(2) - sigma_linear.powi(2))
        .max(0.0)
        .sqrt();
    let cross_floor = (POTTER_SD_LINEAR.powi(2) + sigma_cubic.powi(2)).sqrt();
    let expected_separation = (cross_floor.powi(2) + signal.powi(2)).sqrt();
    let threshold = (cross_floor + expected_separation) / 2.0;
    let measured = frame_mean("linear", 0)?;
    let separation = (measured - cubic).abs() / cubic;
    assert!(
        separation > threshold,
        "LINEAR frame mean {measured:.6} is {:.2} % from Blender's CUBIC {cubic:.6}; the modes separate beyond {:.2} %",
        separation * 100.0,
        threshold * 100.0
    );
    Ok(())
}
