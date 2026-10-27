//! Cycles' volume closures (`intern/cycles/kernel/svm/closure.h`): Volume Absorption has extinction
//! `(1 - Color) · Density`, Principled Volume has scatter `Color · Density` and absorption
//! `(1 - Color) · (1 - sqrt(Absorption Color)) · Density`, and Volume Scatter has extinction and scatter
//! `Color · Density` with albedo 1. Add Shader sums each closure's extinction and Mix Shader weights it.
//! The oracle is `tests/fixtures/volume_extinction`, rendered by Blender 5.2.2 with
//! `generate_volume_extinction_oracle.py`.

use std::{error::Error, fs, io::Cursor, path::Path, process::Output};

use assert_cmd::Command;
use serde_json::{Value, json};
use tempfile::tempdir;

const SIZE: u32 = 24;
/// The pixels on each side of the centre that are averaged. The slab fills the centre of the frame, so every
/// averaged pixel has the same expectation, and the block mean reduces the sampling noise of a stochastic render.
const BLOCK: u32 = 4;
const GRAPH: &str = "volume_graph";
const ABSORPTION: &str = "ShaderNodeVolumeAbsorption";

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

/// A shader-graph node operation in `GRAPH`.
fn node(id: &str, node_type: &str, inputs: Value) -> Value {
    let mut operation = json!({"op":"graph.node_add","graph":GRAPH,"id":id,"type":node_type});
    operation["inputs"] = inputs;
    operation
}

/// A shader-graph link operation in `GRAPH`.
fn link(from: &str, from_socket: &str, to: &str, to_socket: &str) -> Value {
    json!({"op":"graph.link","graph":GRAPH,"from_node":from,"from_socket":from_socket,"to_node":to,"to_socket":to_socket})
}

/// The graph's operations: its Output Material plus `nodes` and `links`. The caller links the final
/// closure to the output's Volume socket.
fn graph_operations(nodes: Vec<Value>, links: Vec<Value>) -> Vec<Value> {
    let mut operations = vec![
        json!({"op":"graph.create","id":GRAPH,"kind":"shader"}),
        json!({"op":"graph.node_add","graph":GRAPH,"id":"output","type":"ShaderNodeOutputMaterial"}),
    ];
    operations.extend(nodes);
    operations.extend(links);
    operations
}

/// Renders the frame and returns the mean of the central `BLOCK` × `BLOCK` pixels in linear RGB.
fn render_center_block(scene: &Path, output_directory: &Path) -> Result<[f64; 3], Box<dyn Error>> {
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
    let mut sum = [0.0; 3];
    let first = SIZE / 2 - BLOCK / 2;
    for y in first..first + BLOCK {
        for x in first..first + BLOCK {
            let offset = ((y * SIZE + x) as usize) * 4;
            for (channel, total) in sum.iter_mut().enumerate() {
                *total += srgb_to_linear(f64::from(pixels[offset + channel]) / 255.0);
            }
        }
    }
    let count = f64::from(BLOCK * BLOCK);
    Ok(sum.map(|total| total / count))
}

/// Renders one unit slab with a volume-only material in front of a white world of strength 1, at `samples` per
/// pixel and `max_bounces`. The graph is `nodes` and `links`; the caller links the final closure to the output's
/// Volume socket.
fn render_slab_at(
    nodes: Vec<Value>,
    links: Vec<Value>,
    samples: u32,
    max_bounces: u32,
) -> Result<[f64; 3], Box<dyn Error>> {
    render_slab_seeded(nodes, links, samples, max_bounces, 0)
}

/// `render_slab_at` with an explicit seed, for estimating Potter's own noise across seeds.
fn render_slab_seeded(
    nodes: Vec<Value>,
    links: Vec<Value>,
    samples: u32,
    max_bounces: u32,
    seed: u32,
) -> Result<[f64; 3], Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let mut operations = graph_operations(nodes, links);
    operations.extend([
        json!({"op":"material.create","id":"slab","graph":GRAPH}),
        json!({"op":"node.create","id":"slab_cube","kind":"box","params":{"size":1.0},"material":"slab"}),
        json!({"op":"world.create","id":"world_main","color":[1.0,1.0,1.0],"strength":1.0}),
        json!({"op":"camera.create","id":"camera_main","transform":{"translation":[0.0,0.0,3.0]},"projection":"orthographic","ortho_scale":2.0}),
        json!({"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"camera_main","world":"world_main"}}),
        json!({"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_x":SIZE,"resolution_y":SIZE,"samples":samples,"seed":seed,"max_bounces":max_bounces,"volume_bounces":max_bounces}}),
    ]);
    make_scene(&scene, &operations)?;
    render_center_block(&scene, &directory.path().join("render"))
}

/// Renders the slab for an absorption-only graph. Cycles attenuates such a slab deterministically
/// (`volume_integrate_heterogeneous`, "Absorption only"), so one sample per pixel is exact.
fn render_slab(nodes: Vec<Value>, links: Vec<Value>) -> Result<[f64; 3], Box<dyn Error>> {
    render_slab_at(nodes, links, 1, 2)
}

/// Renders the slab with one volume node of `node_type` and `inputs` feeding the output.
fn render_single_volume(node_type: &str, inputs: Value) -> Result<[f64; 3], Box<dyn Error>> {
    render_slab(
        vec![node("volume", node_type, inputs)],
        vec![link("volume", "Volume", "output", "Volume")],
    )
}

/// Potter's samples per pixel and seeds for the zero-budget single-scatter case: four seeds give Potter's own spread.
const SINGLE_SCATTER_SAMPLES: u32 = 1024;
const SINGLE_SCATTER_SEEDS: u32 = 4;

/// Blender 5.2.2's single-scatter radiance of a white-world scatterer at `max_bounces = volume_bounces = 0`
/// (`fixtures/volume_single_scatter`). Such a path takes one scatter vertex, and its environment light arrives both as
/// direct light and as the phase-sampled ray traced on past the vertex (Cycles' MIS pair). A Potter renderer that
/// drops that ray biases the mean low. Potter's centre block is compared against Blender's mean within three noise
/// floors: Blender's seed standard error combined with Potter's standard error across seeds.
#[test]
fn volume_scatter_single_scatter_at_zero_budget_matches_blender() -> Result<(), Box<dyn Error>> {
    let oracle: Value = serde_json::from_slice(&fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/volume_single_scatter/blender_volume_single_scatter.json"),
    )?)?;
    let case = &oracle["cases"]["volume_scatter_white_max0"];
    let blender: [f64; 3] = serde_json::from_value(case["mean"].clone())?;
    let blender_error: [f64; 3] = serde_json::from_value(case["stderr"].clone())?;
    let nodes = vec![node(
        "volume",
        "ShaderNodeVolumeScatter",
        json!({"Color":[1.0,1.0,1.0,1.0],"Density":1.0}),
    )];
    let links = vec![link("volume", "Volume", "output", "Volume")];
    let mut renders = Vec::new();
    for seed in 0..SINGLE_SCATTER_SEEDS {
        renders.push(render_slab_seeded(
            nodes.clone(),
            links.clone(),
            SINGLE_SCATTER_SAMPLES,
            0,
            seed,
        )?);
    }
    let seeds = f64::from(SINGLE_SCATTER_SEEDS);
    for channel in 0..3 {
        let values: Vec<f64> = renders.iter().map(|render| render[channel]).collect();
        let mean = values.iter().sum::<f64>() / seeds;
        let variance = values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (seeds - 1.0);
        let potter_error = (variance / seeds).sqrt();
        let floor = (blender_error[channel].powi(2) + potter_error.powi(2)).sqrt();
        assert!(
            (mean - blender[channel]).abs() <= 3.0 * floor,
            "channel {channel}: Potter {mean:.5} vs Blender {:.5} outside three floors ({:.5})",
            blender[channel],
            3.0 * floor
        );
    }
    Ok(())
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

/// Checks Potter's centre block against the analytic value (within 0.01) and against Blender's oracle
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
    let measured = render_single_volume(
        ABSORPTION,
        json!({"Color": [0.2, 0.5, 0.8, 1.0], "Density": 1.0}),
    )?;
    let analytic = [0.2, 0.5, 0.8].map(|color: f64| (-(1.0 - color)).exp());
    assert_matches("volume_absorption", measured, analytic)
}

#[test]
fn volume_absorption_with_colour_unset_uses_blenders_default_0_8() -> Result<(), Box<dyn Error>> {
    // Blender's Volume Absorption Color defaults to 0.8, so the absorption is 0.2 in every channel.
    let measured = render_single_volume(ABSORPTION, json!({"Density": 1.0}))?;
    assert_matches("volume_absorption_default", measured, [(-0.2_f64).exp(); 3])
}

#[test]
fn principled_absorption_colour_transmits_exp_of_one_minus_sqrt_per_channel()
-> Result<(), Box<dyn Error>> {
    let measured = render_single_volume(
        "ShaderNodeVolumePrincipled",
        json!({
            "Color": [0.0, 0.0, 0.0, 1.0],
            "Absorption Color": [0.36, 0.25, 0.04, 1.0],
            "Density": 1.0,
        }),
    )?;
    let analytic = [0.36, 0.25, 0.04].map(|absorption: f64| (-(1.0 - absorption.sqrt())).exp());
    assert_matches("principled_absorption", measured, analytic)
}

#[test]
fn volume_scatter_conserves_a_white_environment() -> Result<(), Box<dyn Error>> {
    // Albedo 1 in a uniform white world: every path ends in the environment, so the radiance is 1. Blender's oracle
    // runs with 64 bounces (`generate_volume_extinction_oracle.py`): a path that scatters more than twice is cut off
    // at the 2-bounce budget of the absorption slabs, so this case uses the oracle's budget. A scattering path is
    // stochastic, so the block is rendered at 512 samples per pixel.
    let measured = render_slab_at(
        vec![node(
            "volume",
            "ShaderNodeVolumeScatter",
            json!({"Color": [0.2, 0.5, 0.8, 1.0], "Density": 1.0}),
        )],
        vec![link("volume", "Volume", "output", "Volume")],
        512,
        64,
    )?;
    assert_matches("volume_scatter_white", measured, [1.0; 3])
}

#[test]
fn add_shader_sums_each_closures_own_absorption() -> Result<(), Box<dyn Error>> {
    // σa = (0.8, 0.5, 0.2) + (0.25, 0.25, 0.25) per channel: each closure keeps its own density.
    let measured = render_slab(
        vec![
            node(
                "first",
                ABSORPTION,
                json!({"Color": [0.2, 0.5, 0.8, 1.0], "Density": 1.0}),
            ),
            node(
                "second",
                ABSORPTION,
                json!({"Color": [0.5, 0.5, 0.5, 1.0], "Density": 0.5}),
            ),
            node("add", "ShaderNodeAddShader", json!({})),
        ],
        vec![
            link("first", "Volume", "add", "Shader"),
            link("second", "Volume", "add", "Shader_001"),
            link("add", "Shader", "output", "Volume"),
        ],
    )?;
    let analytic = [1.05, 0.75, 0.45].map(|extinction: f64| (-extinction).exp());
    assert_matches("add_absorption", measured, analytic)
}

#[test]
fn mix_shader_weights_each_closures_own_absorption() -> Result<(), Box<dyn Error>> {
    // Fac 0.25 on the second closure: σa = 0.75 · (0.8, 0.5, 0.2) + 0.25 · (0.25, 0.25, 0.25).
    let measured = render_slab(
        vec![
            node(
                "first",
                ABSORPTION,
                json!({"Color": [0.2, 0.5, 0.8, 1.0], "Density": 1.0}),
            ),
            node(
                "second",
                ABSORPTION,
                json!({"Color": [0.5, 0.5, 0.5, 1.0], "Density": 0.5}),
            ),
            node("mix", "ShaderNodeMixShader", json!({"Fac": 0.25})),
        ],
        vec![
            link("first", "Volume", "mix", "Shader"),
            link("second", "Volume", "mix", "Shader_001"),
            link("mix", "Shader", "output", "Volume"),
        ],
    )?;
    let analytic = [0.6625, 0.4375, 0.2125].map(|extinction: f64| (-extinction).exp());
    assert_matches("mix_absorption", measured, analytic)
}

/// Renders a voxel volume (a 2 × 2 × 2 grid of unit density) whose material has a Volume Absorption with
/// `absorption_inputs`, and returns the render's `shader.volume.spectral_extinction` warnings.
fn voxel_spectral_warnings(absorption_inputs: Value) -> Result<Vec<Value>, Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let mut operations = graph_operations(
        vec![node("volume", ABSORPTION, absorption_inputs)],
        vec![link("volume", "Volume", "output", "Volume")],
    );
    operations.extend([
        json!({"op":"material.create","id":"fog","graph":GRAPH}),
        json!({"op":"volume.create","id":"fog_volume","materials":["fog"],"grids":[{"dims":[2,2,2],"voxel_size":1.0,"origin":[-0.5,-0.5,-0.5],"values":vec![1.0; 8]}]}),
        json!({"op":"world.create","id":"world_main","color":[1.0,1.0,1.0],"strength":1.0}),
        json!({"op":"camera.create","id":"camera_main","transform":{"translation":[0.0,0.0,3.0]},"projection":"orthographic","ortho_scale":2.0}),
        json!({"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"camera_main","world":"world_main"}}),
        json!({"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_x":8,"resolution_y":8,"samples":1,"seed":0,"max_bounces":2,"volume_bounces":2}}),
    ]);
    make_scene(&scene, &operations)?;
    let output = pot()
        .arg("render")
        .arg(&scene)
        .args(["--engine", "path", "--format", "png", "--out"])
        .arg(directory.path().join("render"))
        .arg("--json")
        .output()?;
    let result = assert_cli_success(&output)?;
    let warnings = result["result"]["warnings"]
        .as_array()
        .ok_or("render warnings are missing")?;
    Ok(warnings
        .iter()
        .filter(|warning| warning["feature_id"] == "shader.volume.spectral_extinction")
        .cloned()
        .collect())
}

#[test]
fn voxel_volume_with_coloured_absorption_reports_no_spectral_extinction_warning()
-> Result<(), Box<dyn Error>> {
    let warnings = voxel_spectral_warnings(json!({"Color": [0.2, 0.5, 0.8, 1.0], "Density": 1.0}))?;
    assert!(
        warnings.is_empty(),
        "supported coloured voxel absorption must not report shader.volume.spectral_extinction: {warnings:?}"
    );
    Ok(())
}

#[test]
fn voxel_volume_with_default_absorption_colour_reports_no_spectral_extinction()
-> Result<(), Box<dyn Error>> {
    let warnings = voxel_spectral_warnings(json!({"Density": 1.0}))?;
    assert!(
        warnings.is_empty(),
        "a grey voxel absorption must not report shader.volume.spectral_extinction: {warnings:?}"
    );
    Ok(())
}

/// Renders two coloured-scattering voxel volumes with realtime and returns their spectral-scattering warnings.
fn realtime_coloured_scattering_warnings() -> Result<Vec<Value>, Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let mut operations = graph_operations(
        vec![node(
            "volume",
            "ShaderNodeVolumeScatter",
            json!({"Color": [0.05, 0.4, 0.9, 1.0], "Density": 1.0}),
        )],
        vec![link("volume", "Volume", "output", "Volume")],
    );
    operations.extend([
        json!({"op":"material.create","id":"fog","graph":GRAPH}),
        json!({"op":"volume.create","id":"fog_volume","materials":["fog"],"grids":[{"dims":[2,2,2],"voxel_size":1.0,"origin":[-0.5,-0.5,-0.5],"values":vec![1.0; 8]}]}),
        json!({"op":"volume.create","id":"fog_volume_duplicate","materials":["fog"],"grids":[{"dims":[2,2,2],"voxel_size":1.0,"origin":[-0.5,-0.5,-0.5],"values":vec![1.0; 8]}]}),
        json!({"op":"world.create","id":"world_main","color":[1.0,1.0,1.0],"strength":1.0}),
        json!({"op":"camera.create","id":"camera_main","transform":{"translation":[0.0,0.0,3.0]},"projection":"orthographic","ortho_scale":2.0}),
        json!({"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"camera_main","world":"world_main"}}),
        json!({"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_x":8,"resolution_y":8,"samples":1,"seed":0,"max_bounces":2,"volume_bounces":2}}),
    ]);
    make_scene(&scene, &operations)?;
    let output = pot()
        .arg("render")
        .arg(&scene)
        .args(["--engine", "realtime", "--format", "png", "--out"])
        .arg(directory.path().join("render"))
        .arg("--json")
        .output()?;
    let result = assert_cli_success(&output)?;
    let warnings = result["result"]["warnings"]
        .as_array()
        .ok_or("render warnings are missing")?;
    Ok(warnings
        .iter()
        .filter(|warning| {
            warning["feature_id"] == "shader.volume.spectral_extinction.realtime_scattering"
        })
        .cloned()
        .collect())
}

#[test]
fn realtime_coloured_volume_scattering_reports_one_deduplicated_warning()
-> Result<(), Box<dyn Error>> {
    let warnings = realtime_coloured_scattering_warnings()?;
    assert_eq!(
        warnings.len(),
        1,
        "expected one warning for two coloured media: {warnings:?}"
    );
    assert_eq!(warnings[0]["code"], "UNSUPPORTED_FEATURE");
    Ok(())
}
