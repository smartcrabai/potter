//! Cycles' volume closures (`intern/cycles/kernel/svm/closure.h`): Volume Absorption has extinction
//! `(1 - Color) · Density`, Principled Volume has scatter `Color · Density` and absorption
//! `(1 - Color) · (1 - sqrt(Absorption Color)) · Density`, and Volume Scatter has extinction and scatter
//! `Color · Density` with albedo 1. The oracle is `tests/fixtures/volume_extinction`, rendered by Blender
//! 5.2.2 with `generate_volume_extinction_oracle.py`.

use std::{error::Error, fs, io::Cursor, path::Path, process::Output};

use assert_cmd::Command;
use serde_json::{Value, json};
use tempfile::tempdir;

const SIZE: u32 = 24;

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

fn srgb_to_linear(encoded: f64) -> f64 {
    if encoded <= 0.04045 {
        encoded / 12.92
    } else {
        ((encoded + 0.055) / 1.055).powf(2.4)
    }
}

/// Renders the centre pixel of the scene in linear RGB.
fn render_center_pixel(scene: &Path, output_directory: &Path) -> Result<[f64; 3], Box<dyn Error>> {
    let output = pot()
        .arg("render")
        .arg(scene)
        .args(["--engine", "path", "--format", "png", "--out"])
        .arg(output_directory)
        .arg("--json")
        .output()?;
    assert_cli_success(&output)?;
    let image = fs::read(output_directory.join("frame_0001.png"))?;
    let mut reader = png::Decoder::new(Cursor::new(image)).read_info()?;
    let mut pixels = vec![
        0;
        reader
            .output_buffer_size()
            .ok_or("PNG buffer size is unknown")?
    ];
    let info = reader.next_frame(&mut pixels)?;
    if info.color_type != png::ColorType::Rgba || info.width != SIZE {
        return Err("renderer PNG is not the expected RGBA image".into());
    }
    let offset = ((SIZE / 2) * SIZE + SIZE / 2) as usize * 4;
    Ok([0_usize, 1, 2].map(|channel| srgb_to_linear(f64::from(pixels[offset + channel]) / 255.0)))
}

/// Renders one unit slab with a volume-only material in front of a white world of strength 1.
/// `volume_node` is the `graph.node_add` operation of the volume shader; its id and graph are set here.
fn render_slab(mut volume_node: Value) -> Result<[f64; 3], Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    volume_node["id"] = json!("volume");
    volume_node["graph"] = json!("volume_graph");
    let operations = vec![
        json!({"op":"graph.create","id":"volume_graph","kind":"shader"}),
        json!({"op":"graph.node_add","graph":"volume_graph","id":"output","type":"ShaderNodeOutputMaterial"}),
        volume_node,
        json!({"op":"graph.link","graph":"volume_graph","from_node":"volume","from_socket":"Volume","to_node":"output","to_socket":"Volume"}),
        json!({"op":"material.create","id":"slab","graph":"volume_graph"}),
        json!({"op":"node.create","id":"slab_cube","kind":"box","params":{"size":1.0},"material":"slab"}),
        json!({"op":"world.create","id":"world_main","color":[1.0,1.0,1.0],"strength":1.0}),
        json!({"op":"camera.create","id":"camera_main","transform":{"translation":[0.0,0.0,3.0]},"projection":"orthographic","ortho_scale":2.0}),
        json!({"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"camera_main","world":"world_main"}}),
        json!({"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_x":SIZE,"resolution_y":SIZE,"samples":1,"seed":0,"max_bounces":2}}),
    ];
    make_scene(&scene, &operations)?;
    render_center_pixel(&scene, &directory.path().join("render"))
}

/// The Blender oracle's mean and standard error of one case, per channel.
fn oracle_case(name: &str) -> Result<([f64; 3], [f64; 3]), Box<dyn Error>> {
    let oracle: Value = serde_json::from_slice(&fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/volume_extinction/blender_volume_extinction.json"),
    )?)?;
    let case = &oracle["cases"][name];
    let triple = |field: &str| -> Result<[f64; 3], Box<dyn Error>> {
        let values = case[field].as_array().ok_or("oracle field is missing")?;
        let number = |index: usize| {
            values
                .get(index)
                .and_then(Value::as_f64)
                .ok_or("oracle value is not a number")
        };
        Ok([number(0)?, number(1)?, number(2)?])
    };
    Ok((triple("mean")?, triple("stderr")?))
}

/// Checks Potter's centre pixel against the analytic value (within 0.01) and against Blender's oracle
/// (within 0.01 plus four Blender standard errors).
fn assert_matches(
    name: &str,
    measured: [f64; 3],
    analytic: [f64; 3],
) -> Result<(), Box<dyn Error>> {
    let (blender, stderr) = oracle_case(name)?;
    for channel in 0..3 {
        assert!(
            (measured[channel] - analytic[channel]).abs() < 0.01,
            "{name} channel {channel}: Potter {} vs analytic {}",
            measured[channel],
            analytic[channel]
        );
        let tolerance = 0.01 + 4.0 * stderr[channel];
        assert!(
            (measured[channel] - blender[channel]).abs() < tolerance,
            "{name} channel {channel}: Potter {} vs Blender {} ± {}",
            measured[channel],
            blender[channel],
            stderr[channel]
        );
    }
    Ok(())
}

#[test]
fn volume_absorption_transmits_exp_of_one_minus_colour_per_channel() -> Result<(), Box<dyn Error>> {
    let measured = render_slab(json!({
        "op": "graph.node_add",
        "type": "ShaderNodeVolumeAbsorption",
        "inputs": {"Color": [0.2, 0.5, 0.8, 1.0], "Density": 1.0},
    }))?;
    let analytic = [0.2, 0.5, 0.8].map(|color: f64| (-(1.0 - color)).exp());
    assert_matches("volume_absorption", measured, analytic)
}

#[test]
fn principled_absorption_colour_transmits_exp_of_one_minus_sqrt_per_channel()
-> Result<(), Box<dyn Error>> {
    let measured = render_slab(json!({
        "op": "graph.node_add",
        "type": "ShaderNodeVolumePrincipled",
        "inputs": {
            "Color": [0.0, 0.0, 0.0, 1.0],
            "Absorption Color": [0.36, 0.25, 0.04, 1.0],
            "Density": 1.0,
        },
    }))?;
    let analytic = [0.36, 0.25, 0.04].map(|absorption: f64| (-(1.0 - absorption.sqrt())).exp());
    assert_matches("principled_absorption", measured, analytic)
}

#[test]
fn volume_scatter_conserves_a_white_environment() -> Result<(), Box<dyn Error>> {
    // Albedo 1 in a uniform white world: every path ends in the environment, so the radiance is 1.
    // The former grey model (extinction Density, albedo Color) gave 0.68 for Color 0.5.
    let measured = render_slab(json!({
        "op": "graph.node_add",
        "type": "ShaderNodeVolumeScatter",
        "inputs": {"Color": [0.2, 0.5, 0.8, 1.0], "Density": 1.0},
    }))?;
    assert_matches("volume_scatter_white", measured, [1.0; 3])
}
