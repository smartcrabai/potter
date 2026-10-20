//! Voxel `volume_interpolation`. The oracle is `tests/fixtures/volume_interpolation`, rendered by Blender 5.2.2 with
//! `generate_volume_interpolation_oracle.py`: a 16³ hashed grid (the same values as `hashed` below), imported as a
//! Volume object with CUBIC or LINEAR interpolation and a Volume Scatter at density 0.05, single scatter under a
//! point light. Cycles' CUBIC is the B-spline of `kernel_image_interp_tricubic_nanovdb`, which Potter's
//! `cubic_weights` follows. The block mean barely separates the two modes (the B-spline preserves the field's mean),
//! so the comparison is per pixel over the whole image.

use std::{error::Error, fs, path::Path, process::Output};

use assert_cmd::Command;
use serde_json::{Value, json};
use tempfile::tempdir;

const SIZE: u32 = 24;
const N: usize = 16;
const ENERGY: f64 = 100.0;
const POSITION: [f64; 3] = [1.6, 0.6, 0.9];
const DENSITY_SCALE: f64 = 0.05;
const SAMPLES: u32 = 8192;

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

/// The grid value at node `(i, j, k)`, as `generate_volume_interpolation_oracle.py` writes it.
fn hashed(i: i64, j: i64, k: i64) -> f64 {
    (((i * 73_856_093) ^ (j * 19_349_663) ^ (k * 83_492_791)) & 1023) as f64 / 1023.0
}

fn make_scene(root: &Path, interpolation: &str) -> Result<std::path::PathBuf, Box<dyn Error>> {
    let mut values = Vec::with_capacity(N * N * N);
    for k in 0..N {
        for j in 0..N {
            for i in 0..N {
                values.push(hashed(i as i64, j as i64, k as i64));
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
        json!({"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_x":SIZE,"resolution_y":SIZE,"samples":SAMPLES,"seed":0,"max_bounces":0}}),
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

/// The red channel of the whole rendered image, in the EXR's row order.
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

/// The oracle's per-pixel image for a case, in the row order of `generate_volume_interpolation_oracle.py`.
fn blender_image(case: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    let oracle: Value = serde_json::from_slice(&fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/volume_interpolation/blender_volume_interpolation.json"),
    )?)?;
    oracle["cases"][case]["pixels"]
        .as_array()
        .ok_or("oracle pixels are missing")?
        .iter()
        .map(|value| value.as_f64().ok_or("oracle pixel is not a number"))
        .collect::<Result<Vec<f64>, &str>>()
        .map_err(Into::into)
}

/// The root-mean-square difference of two images, relative to the mean of the reference.
fn relative_rms(measured: &[f64], reference: &[f64]) -> f64 {
    let squares: f64 = measured
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    let rms = (squares / reference.len() as f64).sqrt();
    let mean = reference.iter().sum::<f64>() / reference.len() as f64;
    rms / mean
}

/// `relative_rms` under the better of the two row orders. The EXR row origin of the two renderers is a file-format
/// convention, not part of the sampler under test.
fn oriented_relative_rms(measured: &[f64], reference: &[f64]) -> f64 {
    let width = SIZE as usize;
    let flipped: Vec<f64> = (0..reference.len())
        .map(|index| measured[(width - 1 - index / width) * width + index % width])
        .collect();
    relative_rms(measured, reference).min(relative_rms(&flipped, reference))
}

#[test]
fn cubic_and_linear_voxel_interpolation_match_blender_per_pixel() -> Result<(), Box<dyn Error>> {
    let cubic_blender = blender_image("cubic_noise")?;
    let linear_blender = blender_image("linear_noise")?;
    let directory = tempdir()?;
    let cubic = render_image(
        &make_scene(&directory.path().join("cubic"), "cubic")?,
        &directory.path().join("cubic_render"),
    )?;
    let linear = render_image(
        &make_scene(&directory.path().join("linear"), "linear")?,
        &directory.path().join("linear_render"),
    )?;
    let cubic_error = oriented_relative_rms(&cubic, &cubic_blender);
    let linear_error = oriented_relative_rms(&linear, &linear_blender);
    // The two modes must be distinguishable: a linear render must not match Blender's cubic one.
    let linear_against_cubic = oriented_relative_rms(&linear, &cubic_blender);
    assert!(
        cubic_error < 0.03,
        "CUBIC voxels: per-pixel error {:.2} % against Blender's CUBIC render",
        cubic_error * 100.0
    );
    assert!(
        linear_error < 0.03,
        "LINEAR voxels: per-pixel error {:.2} % against Blender's LINEAR render",
        linear_error * 100.0
    );
    assert!(
        linear_against_cubic > 0.045,
        "LINEAR voxels match CUBIC too closely ({:.2} %): the interpolation mode is not taking effect",
        linear_against_cubic * 100.0
    );
    Ok(())
}
