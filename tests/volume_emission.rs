//! Emission of volume media (`intern/cycles/kernel/integrator/shade_volume.h`). An absorbing emissive cube is the closed form
//! `E · (1 - e^-σ) / σ` (Cycles' homogeneous `volume_emission_integrate`), with σ the Density and E the Emission Colour
//! times its strength, unscaled by the Density. A cube that scatters adds what its scattered paths collect along their
//! terminal rays, and an emissive voxel grid is stochastic, so its collisions end the emission. The oracles are
//! `tests/fixtures/volume_emission`, rendered by Blender 5.2.2 with `generate_volume_emission_oracle.py` and
//! `generate_volume_emission_voxel_oracle.py`.

use std::{error::Error, fs, path::Path, process::Output};

use assert_cmd::Command;
use serde_json::{Value, json};
use tempfile::tempdir;

const SIZE: u32 = 24;
/// The pixels on each side of the centre that are averaged. The subject fills the centre of the frame, so every averaged
/// pixel has the same expectation, and the block mean reduces the noise of a stochastic render.
const BLOCK: usize = 4;
const GRAPH: &str = "emission_graph";
const SAMPLES: u32 = 1024;
/// The seeds for a stochastic case: their spread is Potter's own noise.
const SEEDS: u32 = 4;
/// The Principled Volume's Density, which is σ of an absorbing medium of unit colour.
const DENSITY: f64 = 2.0;

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

fn apply_operations(scene: &Path, operations: &[Value]) -> Result<Value, Box<dyn Error>> {
    let directory = scene
        .parent()
        .ok_or_else(|| std::io::Error::other("scene directory has no parent"))?;
    let batch_path = directory.join("operations.json");
    fs::write(
        &batch_path,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": operations,
        }))?,
    )?;
    let output = pot()
        .arg("apply")
        .arg(scene)
        .arg("--file")
        .arg(batch_path)
        .arg("--json")
        .output()?;
    assert_cli_success(&output)
}

fn make_scene(scene: &Path, operations: &[Value]) -> Result<Value, Box<dyn Error>> {
    let output = pot().arg("init").arg(scene).arg("--json").output()?;
    assert_cli_success(&output)?;
    apply_operations(scene, operations)
}

/// A shader-graph node operation in `GRAPH`.
fn node(id: &str, node_type: &str, inputs: Value) -> Value {
    json!({"op":"graph.node_add","graph":GRAPH,"id":id,"type":node_type,"inputs":inputs})
}

/// A shader-graph link operation in `GRAPH`.
fn link(from: &str, from_socket: &str, to: &str, to_socket: &str) -> Value {
    json!({"op":"graph.link","graph":GRAPH,"from_node":from,"from_socket":from_socket,"to_node":to,"to_socket":to_socket})
}

/// The graph's operations: its Output Material and the volume node, linked to the output's Volume socket.
fn graph_operations(volume_inputs: Value) -> Vec<Value> {
    vec![
        json!({"op":"graph.create","id":GRAPH,"kind":"shader"}),
        json!({"op":"graph.node_add","graph":GRAPH,"id":"output","type":"ShaderNodeOutputMaterial"}),
        node("volume", "ShaderNodeVolumePrincipled", volume_inputs),
        link("volume", "Volume", "output", "Volume"),
    ]
}

/// The Principled Volume's inputs: a Colour of `albedo` (its scatter fraction), the Density, and emission of unit colour
/// and strength.
fn volume_inputs(albedo: f64) -> Value {
    json!({
        "Color": [albedo, albedo, albedo, 1.0],
        "Density": DENSITY,
        "Anisotropy": 0.0,
        "Emission Color": [1.0, 1.0, 1.0, 1.0],
        "Emission Strength": 1.0,
    })
}

/// The camera, the black world, the render settings, and the scene's bounce budget. Volume scatter takes Cycles'
/// `volume_bounces` budget, here at Blender's default of zero, and no light reaches the camera but emission.
fn frame_operations(seed: u32) -> Vec<Value> {
    vec![
        json!({"op":"world.create","id":"world_main","color":[0.0,0.0,0.0],"strength":1.0}),
        json!({"op":"camera.create","id":"camera_main","transform":{"translation":[0.0,0.0,3.0]},"projection":"orthographic","ortho_scale":2.0}),
        json!({"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"camera_main","world":"world_main"}}),
        json!({"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_x":SIZE,"resolution_y":SIZE,"samples":SAMPLES,"seed":seed,"max_bounces":0,"volume_bounces":0}}),
    ]
}

/// A unit cube of the emissive medium.
fn cube_operations(albedo: f64, seed: u32) -> Vec<Value> {
    let mut operations = graph_operations(volume_inputs(albedo));
    operations.push(json!({"op":"material.create","id":"emitter","graph":GRAPH}));
    operations.push(json!({"op":"node.create","id":"cube","kind":"box","params":{"size":1.0},"material":"emitter"}));
    operations.extend(frame_operations(seed));
    operations
}

/// A 16³ voxel grid of ones, voxel size 0.125, its index (i, j, k) at -0.9375 + 0.125 (i, j, k) as Blender's VDB is
/// placed (`generate_volume_emission_voxel_oracle.py`).
fn voxel_operations(albedo: f64, seed: u32) -> Vec<Value> {
    let mut operations = graph_operations(volume_inputs(albedo));
    operations.push(json!({"op":"material.create","id":"emitter","graph":GRAPH}));
    operations.push(json!({"op":"volume.create","id":"emissive","materials":["emitter"],"grids":[{"dims":[16,16,16],"voxel_size":0.125,"origin":[-0.9375,-0.9375,-0.9375],"values":vec![1.0_f64; 16 * 16 * 16]}]}));
    operations.extend(frame_operations(seed));
    operations
}

struct Pixels {
    width: usize,
    rgb: Vec<[f32; 3]>,
}

/// Renders the frame in EXR and returns the mean of the central `BLOCK` × `BLOCK` pixels, in linear RGB.
fn render_center_block(operations: &[Value]) -> Result<[f64; 3], Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    make_scene(&scene, operations)?;
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
            rgb: vec![[0.0_f32; 3]; resolution.width() * resolution.height()],
        },
        |pixels: &mut Pixels, position, (red, green, blue, _alpha): (f32, f32, f32, f32)| {
            pixels.rgb[position.y() * pixels.width + position.x()] = [red, green, blue];
        },
    )?;
    let pixels = image.layer_data.channel_data.pixels;
    let first = (SIZE as usize - BLOCK) / 2;
    let mut sum = [0.0_f64; 3];
    for y in first..first + BLOCK {
        for x in first..first + BLOCK {
            let value = pixels.rgb[y * pixels.width + x];
            for (channel, total) in sum.iter_mut().enumerate() {
                *total += f64::from(value[channel]);
            }
        }
    }
    let count = (BLOCK * BLOCK) as f64;
    Ok(sum.map(|total| total / count))
}

/// The Blender oracle's mean and standard error of one case of `file`, per channel.
fn oracle_case(file: &str, name: &str) -> Result<([f64; 3], [f64; 3]), Box<dyn Error>> {
    let oracle: Value = serde_json::from_slice(&fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/volume_emission")
            .join(file),
    )?)?;
    let case = &oracle["cases"][name];
    Ok((
        serde_json::from_value(case["mean"].clone())?,
        serde_json::from_value(case["stderr"].clone())?,
    ))
}

/// Checks a stochastic case: Potter's mean over `SEEDS` renders matches Blender's mean within three noise floors, the
/// floor being Blender's seed standard error combined with Potter's standard error of the mean across seeds.
fn assert_matches_blender(
    file: &str,
    name: &str,
    operations: impl Fn(u32) -> Vec<Value>,
) -> Result<(), Box<dyn Error>> {
    let (blender, blender_error) = oracle_case(file, name)?;
    let mut renders = Vec::new();
    for seed in 0..SEEDS {
        renders.push(render_center_block(&operations(seed))?);
    }
    let count = f64::from(SEEDS);
    for channel in 0..3 {
        let values: Vec<f64> = renders.iter().map(|render| render[channel]).collect();
        let mean = values.iter().sum::<f64>() / count;
        let variance = values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (count - 1.0);
        let potter_error = (variance / count).sqrt();
        let floor = (blender_error[channel].powi(2) + potter_error.powi(2)).sqrt();
        assert!(
            (mean - blender[channel]).abs() <= 3.0 * floor,
            "{name} channel {channel}: Potter {mean:.5} vs Blender {:.5} outside three floors ({:.5})",
            blender[channel],
            3.0 * floor
        );
    }
    Ok(())
}

/// The emission integral of an absorbing unit cube along a ray through its centre: `E · (1 - e^-σ) / σ`, with E = 1.
fn absorbing_closed_form() -> f64 {
    (1.0 - (-DENSITY).exp()) / DENSITY
}

/// An absorbing cube has no scatter, so its segment is deterministic (`volume_integrate_heterogeneous`, "Absorption only"),
/// and each pixel is the closed form to f32 rounding. The Blender oracle agrees with the closed form to its 6 decimals.
#[test]
fn absorbing_emissive_cube_matches_the_closed_form() -> Result<(), Box<dyn Error>> {
    let closed = absorbing_closed_form();
    let (blender, _) = oracle_case("blender_volume_emission.json", "emissive_cube_color0")?;
    let potter = render_center_block(&cube_operations(0.0, 0))?;
    for channel in 0..3 {
        assert!(
            (blender[channel] - closed).abs() <= 1.0e-5,
            "Blender {} differs from the closed form {closed}",
            blender[channel]
        );
        assert!(
            (potter[channel] - closed).abs() <= 1.0e-4,
            "Potter {} differs from the closed form {closed}",
            potter[channel]
        );
    }
    Ok(())
}

/// A scattering cube: its scattered paths continue on a terminal ray, which collects the cube's emission
/// (`volume_bounces` 0 takes one scatter vertex), so the means exceed the closed form.
#[test]
fn scattering_emissive_cube_matches_blender() -> Result<(), Box<dyn Error>> {
    assert_matches_blender(
        "blender_volume_emission.json",
        "emissive_cube_color05",
        |seed| cube_operations(0.5, seed),
    )?;
    assert_matches_blender(
        "blender_volume_emission.json",
        "emissive_cube_color1",
        |seed| cube_operations(1.0, seed),
    )
}

/// An absorbing emissive voxel grid: no scatter, so its segment is deterministic. Blender's oracle, whose grid has a
/// boundary fade, is the reference.
#[test]
fn absorbing_emissive_voxel_matches_blender() -> Result<(), Box<dyn Error>> {
    assert_matches_blender(
        "blender_volume_emission_voxel.json",
        "emissive_voxel_color0",
        |seed| voxel_operations(0.0, seed),
    )
}

/// A scattering emissive voxel grid is stochastic: its collisions end the emission, and it takes no extra transmittance
/// weight. Blender's oracle is the reference.
#[test]
fn scattering_emissive_voxel_matches_blender() -> Result<(), Box<dyn Error>> {
    assert_matches_blender(
        "blender_volume_emission_voxel.json",
        "emissive_voxel_color05",
        |seed| voxel_operations(0.5, seed),
    )
}
