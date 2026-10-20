//! Single-scatter direct lighting from a point light. The oracle is `tests/fixtures/volume_nee`, rendered
//! by Blender 5.2.2 with `generate_volume_nee_oracle.py`: a unit cube of Volume Scatter (Color 1, Density
//! 0.05, so optical depth 0.05 and multiple scattering is about 0.2 %), homogeneous or Linear-gradient
//! density, under a point light of 100 W, with `volume_bounces = max_bounces = 1`. The `diffuse` case is a
//! white Lambert plane under the same light and calibrates the light unit (I = P / 4π, irradiance E = I / d²,
//! Lambert radiance E / π). A volume's in-scatter has no 1/π, since its phase function is per steradian.
//!
//! Voxel volumes carry the density, because a mesh material's medium is not a stochastic medium and gets no
//! light sampling (see the CHANGELOG).

use std::{error::Error, fs, path::Path, process::Output};

use assert_cmd::Command;
use serde_json::{Value, json};
use tempfile::tempdir;

const SIZE: u32 = 24;
const ENERGY: f64 = 100.0;
const POSITION: [f64; 3] = [1.6, 0.6, 0.9];
const DENSITY: f64 = 0.05;
const SAMPLES: u32 = 1024;
/// The relative tolerance of a volume case against Blender's oracle, which the Blender stderr of the
/// gradient cases (6.5 %) bounds from below.
const VOLUME_TOLERANCE: f64 = 0.06;
const DIFFUSE_TOLERANCE: f64 = 0.02;

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

fn make_scene(root: &Path, operations: &[Value]) -> Result<std::path::PathBuf, Box<dyn Error>> {
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

/// The common scene: a black world, an orthographic camera at z = 3 (ortho scale 2), a point light of
/// radius 0 at `POSITION`, and single-scatter render settings.
fn base_operations() -> Vec<Value> {
    vec![
        json!({"op":"world.create","id":"world_main","color":[0.0,0.0,0.0],"strength":0.0}),
        json!({"op":"camera.create","id":"camera_main","transform":{"translation":[0.0,0.0,3.0]},"projection":"orthographic","ortho_scale":2.0}),
        json!({"op":"light.create","id":"light_main","light_type":"point","energy":ENERGY,"radius":0.0,"transform":{"translation":POSITION}}),
        json!({"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"camera_main","world":"world_main"}}),
        json!({"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_x":SIZE,"resolution_y":SIZE,"samples":SAMPLES,"seed":0,"max_bounces":0}}),
    ]
}

fn diffuse_scene(root: &Path) -> Result<std::path::PathBuf, Box<dyn Error>> {
    let mut operations = base_operations();
    operations.extend([
        json!({"op":"graph.create","id":"white_graph","kind":"shader"}),
        json!({"op":"graph.node_add","graph":"white_graph","id":"output","type":"ShaderNodeOutputMaterial"}),
        json!({"op":"graph.node_add","graph":"white_graph","id":"diffuse","type":"ShaderNodeBsdfDiffuse","inputs":{"Color":[1.0,1.0,1.0,1.0]}}),
        json!({"op":"graph.link","graph":"white_graph","from_node":"diffuse","from_socket":"BSDF","to_node":"output","to_socket":"Surface"}),
        json!({"op":"material.create","id":"white","graph":"white_graph"}),
        json!({"op":"node.create","id":"plane_main","kind":"plane","params":{"size":4.0},"material":"white"}),
    ]);
    make_scene(root, &operations)
}

/// A 16³ voxel grid over the unit cube. `gradient` sets the value to the Generated coordinate
/// (x + 0.5) at each voxel centre, so with Density 0.05 the density is Blender's Linear gradient.
fn scatter_scene(
    root: &Path,
    anisotropy: f64,
    gradient: bool,
) -> Result<std::path::PathBuf, Box<dyn Error>> {
    const N: usize = 16;
    let mut values = Vec::with_capacity(N * N * N);
    for _z in 0..N {
        for _y in 0..N {
            for x in 0..N {
                let centre = -0.5 + (x as f64 + 0.5) / N as f64;
                values.push(if gradient { centre + 0.5 } else { 1.0 });
            }
        }
    }
    let mut operations = base_operations();
    operations.extend([
        json!({"op":"graph.create","id":"medium_graph","kind":"shader"}),
        json!({"op":"graph.node_add","graph":"medium_graph","id":"output","type":"ShaderNodeOutputMaterial"}),
        json!({"op":"graph.node_add","graph":"medium_graph","id":"scatter","type":"ShaderNodeVolumeScatter","inputs":{"Color":[1.0,1.0,1.0,1.0],"Density":DENSITY,"Anisotropy":anisotropy}}),
        json!({"op":"graph.link","graph":"medium_graph","from_node":"scatter","from_socket":"Volume","to_node":"output","to_socket":"Volume"}),
        json!({"op":"material.create","id":"medium","graph":"medium_graph"}),
        json!({"op":"volume.create","id":"medium_volume","materials":["medium"],"grids":[{"dims":[N,N,N],"voxel_size":1.0 / N as f64,"origin":[-0.5,-0.5,-0.5],"values":values}]}),
    ]);
    make_scene(root, &operations)
}

/// Renders the scene and returns the red channel's mean over the 4 × 4 pixels at the image centre, the
/// block that the oracle measures.
fn render_centre_mean(scene: &Path, out: &Path) -> Result<f64, Box<dyn Error>> {
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
    let width = pixels.width;
    let mut sum = 0.0;
    let mut count = 0.0;
    for y in 10..14 {
        for x in 10..14 {
            sum += f64::from(pixels.red[y * width + x]);
            count += 1.0;
        }
    }
    Ok(sum / count)
}

struct Pixels {
    width: usize,
    red: Vec<f32>,
}

/// The Blender oracle's mean for a case.
fn blender_mean(case: &str) -> Result<f64, Box<dyn Error>> {
    let oracle: Value = serde_json::from_slice(&fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/volume_nee/blender_volume_nee.json"),
    )?)?;
    oracle["cases"][case]["mean"]
        .as_f64()
        .ok_or_else(|| format!("oracle case {case} is missing").into())
}

fn assert_within(case: &str, measured: f64, tolerance: f64) -> Result<(), Box<dyn Error>> {
    let expected = blender_mean(case)?;
    let relative = (measured - expected).abs() / expected;
    assert!(
        relative < tolerance,
        "{case}: Potter {measured:.6} vs Blender {expected:.6} ({:.1} % apart, tolerance {:.0} %)",
        relative * 100.0,
        tolerance * 100.0
    );
    Ok(())
}

#[test]
fn point_light_irradiance_unit_matches_blender_on_a_diffuse_plane() -> Result<(), Box<dyn Error>> {
    // Calibrates the light unit both paths share: if this fails, the volume cases below cannot be read.
    let directory = tempdir()?;
    let scene = diffuse_scene(directory.path())?;
    let measured = render_centre_mean(&scene, &directory.path().join("render"))?;
    assert_within("diffuse", measured, DIFFUSE_TOLERANCE)
}

#[test]
fn homogeneous_single_scatter_under_a_point_light_matches_blender() -> Result<(), Box<dyn Error>> {
    for (case, anisotropy) in [("homogeneous_g0", 0.0), ("homogeneous_g05", 0.5)] {
        let directory = tempdir()?;
        let scene = scatter_scene(directory.path(), anisotropy, false)?;
        let measured = render_centre_mean(&scene, &directory.path().join("render"))?;
        assert_within(case, measured, VOLUME_TOLERANCE)?;
    }
    Ok(())
}

#[test]
fn gradient_single_scatter_under_a_point_light_matches_blender() -> Result<(), Box<dyn Error>> {
    for (case, anisotropy) in [("gradient_g0", 0.0), ("gradient_g05", 0.5)] {
        let directory = tempdir()?;
        let scene = scatter_scene(directory.path(), anisotropy, true)?;
        let measured = render_centre_mean(&scene, &directory.path().join("render"))?;
        assert_within(case, measured, VOLUME_TOLERANCE)?;
    }
    Ok(())
}
