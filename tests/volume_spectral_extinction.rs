//! Colored voxel absorption, heterogeneous spectral scatter, mixed media, and realtime EEVEE checks.

use std::{error::Error, fs, path::Path, process::Output};

use assert_cmd::Command;
use serde_json::{Value, json};
use tempfile::tempdir;

const SIZE: usize = 4;
const GRID_SIZE: usize = 17;
const ORACLE_SEEDS: [u32; 4] = [0, 1, 2, 3];
const ABSORPTION_SAMPLES: u32 = 32;
// Preserve the existing deterministic absorption reference band; new scattering cases use measured standard errors.
const ABSORPTION_TOLERANCE: f64 = 0.09;

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

fn make_absorption_scene(root: &Path) -> Result<std::path::PathBuf, Box<dyn Error>> {
    fs::create_dir_all(root)?;
    let scene = root.join("scene");
    assert_cli_success(&pot().arg("init").arg(&scene).arg("--json").output()?)?;

    let values = vec![1.0; GRID_SIZE * GRID_SIZE * GRID_SIZE];
    let operations = vec![
        json!({"op":"graph.create","id":"volume_graph","kind":"shader"}),
        json!({"op":"graph.node_add","graph":"volume_graph","id":"output","type":"ShaderNodeOutputMaterial"}),
        json!({"op":"graph.node_add","graph":"volume_graph","id":"volume","type":"ShaderNodeVolumeAbsorption","inputs":{"Color":[0.1,0.4,0.9,1.0],"Density":1.0}}),
        json!({"op":"graph.link","graph":"volume_graph","from_node":"volume","from_socket":"Volume","to_node":"output","to_socket":"Volume"}),
        json!({"op":"material.create","id":"voxel_material","graph":"volume_graph","volume_sampling":"multiple_importance"}),
        json!({"op":"volume.create","id":"voxel_volume","materials":["voxel_material"],"grids":[{"dims":[GRID_SIZE,GRID_SIZE,GRID_SIZE],"voxel_size":1.0 / 16.0,"origin":[-0.5,-0.5,-0.5],"values":values}]}),
        json!({"op":"world.create","id":"world_main","color":[1.0,1.0,1.0],"strength":1.0}),
        json!({"op":"camera.create","id":"camera_main","transform":{"translation":[0.0,0.0,3.0]},"projection":"orthographic","ortho_scale":1.0}),
        json!({"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"camera_main","world":"world_main"}}),
        json!({"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_x":SIZE,"resolution_y":SIZE,"samples":ABSORPTION_SAMPLES,"seed":0,"max_bounces":0,"volume_bounces":0,"pixel_filter_type":"BOX","filter_width":1.0}}),
    ];

    let batch = root.join("operations.json");
    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": operations,
        }))?,
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

struct RgbImage {
    width: usize,
    height: usize,
    pixels: Vec<[f64; 3]>,
}

fn image_mean(path: &Path) -> Result<[f64; 3], Box<dyn Error>> {
    let image = exr::prelude::read_first_rgba_layer_from_file(
        path,
        |resolution, _| RgbImage {
            width: resolution.width(),
            height: resolution.height(),
            pixels: vec![[0.0; 3]; resolution.width() * resolution.height()],
        },
        |pixels: &mut RgbImage, position, (red, green, blue, _alpha): (f32, f32, f32, f32)| {
            pixels.pixels[position.y() * pixels.width + position.x()] =
                [f64::from(red), f64::from(green), f64::from(blue)];
        },
    )?;
    let pixels = image.layer_data.channel_data.pixels;
    if pixels.width != SIZE || pixels.height != SIZE {
        return Err(format!(
            "{} is {}x{}, expected {SIZE}x{SIZE}",
            path.display(),
            pixels.width,
            pixels.height
        )
        .into());
    }
    let mut mean = [0.0; 3];
    for pixel in &pixels.pixels {
        for channel in 0..3 {
            mean[channel] += pixel[channel] / (SIZE * SIZE) as f64;
        }
    }
    Ok(mean)
}

fn render_mean(
    scene: &Path,
    output_directory: &Path,
    engine: &str,
) -> Result<[f64; 3], Box<dyn Error>> {
    assert_cli_success(
        &pot()
            .arg("render")
            .arg(scene)
            .args(["--engine", engine, "--format", "exr", "--out"])
            .arg(output_directory)
            .arg("--json")
            .output()?,
    )?;
    image_mean(&output_directory.join("frame_0001.exr"))
}

fn assert_channels_match(
    description: &str,
    measured: [f64; 3],
    reference: [f64; 3],
    relative_tolerance: f64,
) {
    for channel in 0..3 {
        let relative_error =
            (measured[channel] - reference[channel]).abs() / reference[channel].abs();
        assert!(
            relative_error < relative_tolerance,
            "{description} channel {channel}: Potter {:.6} vs Cycles {:.6} ({:.1}% apart, tolerance {:.1}%)",
            measured[channel],
            reference[channel],
            relative_error * 100.0,
            relative_tolerance * 100.0,
        );
    }
}

#[test]
fn colored_voxel_absorption_matches_cycles_per_channel() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let absorption_scene = make_absorption_scene(&directory.path().join("absorption"))?;
    let render_root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/volume_spectral_extinction");

    let absorption = render_mean(
        &absorption_scene,
        &directory.path().join("absorption_render"),
        "path",
    )?;
    let absorption_cycles = image_mean(&render_root.join("volume_absorption_cycles.exr"))?;
    assert!(
        absorption_cycles[0] < absorption_cycles[1] && absorption_cycles[1] < absorption_cycles[2],
        "colored absorption oracle must retain distinct RGB transmittances: {absorption_cycles:?}"
    );
    assert_channels_match(
        "colored voxel absorption transmittance",
        absorption,
        absorption_cycles,
        ABSORPTION_TOLERANCE,
    );
    let realtime_absorption = render_mean(
        &absorption_scene,
        &directory.path().join("absorption_realtime_render"),
        "realtime",
    )?;
    assert_channels_match(
        "realtime colored voxel absorption transmittance",
        realtime_absorption,
        absorption_cycles,
        ABSORPTION_TOLERANCE,
    );
    Ok(())
}

/// Four fixed seeds provide the independent Potter samples for the standard-error comparison.
const HETEROGENEOUS_PATH_SAMPLES: u32 = 1024;

#[derive(Clone, Copy)]
enum HeterogeneousScene {
    Principled,
    MixedVoxelMesh,
}

impl HeterogeneousScene {
    fn label(self) -> &'static str {
        match self {
            Self::Principled => "principled_heterogeneous",
            Self::MixedVoxelMesh => "mixed_voxel_mesh",
        }
    }
}

fn spectral_oracle_case(name: &str) -> Result<([f64; 3], [f64; 3]), Box<dyn Error>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/volume_spectral_extinction/volume_spectral_extinction.json");
    let oracle: Value = serde_json::from_slice(&fs::read(path)?)?;
    let case = oracle
        .get("cases")
        .and_then(|cases| cases.get(name))
        .ok_or("spectral volume oracle case is missing")?;
    let seed_means: Vec<[f64; 3]> = serde_json::from_value(
        case.get("seed_means")
            .ok_or("spectral volume oracle seed means are missing")?
            .clone(),
    )?;
    if seed_means.len() != ORACLE_SEEDS.len() {
        return Err(format!(
            "spectral volume oracle has {} seed means, expected {}",
            seed_means.len(),
            ORACLE_SEEDS.len()
        )
        .into());
    }
    let rgb = |field: &str| -> Result<[f64; 3], Box<dyn Error>> {
        Ok(serde_json::from_value(
            case.get(field)
                .ok_or("spectral volume oracle RGB field is missing")?
                .clone(),
        )?)
    };
    Ok((rgb("mean")?, rgb("stderr")?))
}

fn hashed_density_grid() -> Vec<f64> {
    const N: usize = 17;
    const Z: usize = 49;
    let mut values = Vec::with_capacity(N * N * Z);
    for z in 0..Z {
        for y in 0..N {
            for x in 0..N {
                let mut hash = (x as u32)
                    .wrapping_mul(0x9E37_79B1)
                    .wrapping_add((y as u32).wrapping_mul(0x85EB_CA77))
                    .wrapping_add((z as u32).wrapping_mul(0xC2B2_AE3D))
                    .wrapping_add(0x27D4_EB2F);
                hash ^= hash >> 16;
                hash = hash.wrapping_mul(0x7FEB_352D);
                hash ^= hash >> 15;
                hash = hash.wrapping_mul(0x846C_A68B);
                hash ^= hash >> 16;
                values
                    .push(0.25 + 0.75 * f64::from((hash >> 8) & 0x00FF_FFFF) / 0x00FF_FFFF as f64);
            }
        }
    }
    values
}
fn make_heterogeneous_scene(
    root: &Path,
    kind: HeterogeneousScene,
    samples: u32,
    seed: u32,
) -> Result<std::path::PathBuf, Box<dyn Error>> {
    const N: usize = 17;
    const Z: usize = 49;
    fs::create_dir_all(root)?;
    let scene = root.join("scene");
    assert_cli_success(&pot().arg("init").arg(&scene).arg("--json").output()?)?;

    let mut operations = vec![
        json!({"op":"graph.create","id":"principled_graph","kind":"shader"}),
        json!({"op":"graph.node_add","graph":"principled_graph","id":"output","type":"ShaderNodeOutputMaterial"}),
        json!({"op":"graph.node_add","graph":"principled_graph","id":"principled","type":"ShaderNodeVolumePrincipled","inputs":{"Color":[0.05,0.4,0.9,1.0],"Absorption Color":[0.0,0.8,0.8,1.0],"Density":1.2,"Anisotropy":0.0}}),
        json!({"op":"graph.link","graph":"principled_graph","from_node":"principled","from_socket":"Volume","to_node":"output","to_socket":"Volume"}),
        json!({"op":"material.create","id":"principled_material","graph":"principled_graph","volume_sampling":"multiple_importance"}),
        json!({"op":"volume.create","id":"hashed_volume","materials":["principled_material"],"grids":[{"dims":[N,N,Z],"voxel_size":1.0 / 16.0,"origin":[-0.5,-0.5,-1.5],"values":hashed_density_grid()}]}),
        json!({"op":"world.create","id":"world_main","color":[0.0,0.0,0.0],"strength":0.0}),
        json!({"op":"camera.create","id":"camera_main","transform":{"translation":[0.0,0.0,3.0]},"projection":"orthographic","ortho_scale":1.0}),
        json!({"op":"light.create","id":"light_main","light_type":"point","energy":1000.0,"radius":0.0,"transform":{"translation":[4.0,2.0,-1.9]}}),
        json!({"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"camera_main","world":"world_main"}}),
        json!({"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_x":SIZE,"resolution_y":SIZE,"samples":samples,"seed":seed,"max_bounces":4,"volume_bounces":3,"pixel_filter_type":"BOX","filter_width":1.0}}),
    ];
    if matches!(kind, HeterogeneousScene::MixedVoxelMesh) {
        operations.extend([
            json!({"op":"graph.create","id":"mesh_graph","kind":"shader"}),
            json!({"op":"graph.node_add","graph":"mesh_graph","id":"mesh_output","type":"ShaderNodeOutputMaterial"}),
            json!({"op":"graph.node_add","graph":"mesh_graph","id":"mesh_absorption","type":"ShaderNodeVolumeAbsorption","inputs":{"Color":[0.5,0.5,0.5,1.0],"Density":0.3}}),
            json!({"op":"graph.link","graph":"mesh_graph","from_node":"mesh_absorption","from_socket":"Volume","to_node":"mesh_output","to_socket":"Volume"}),
            json!({"op":"material.create","id":"mesh_material","graph":"mesh_graph","volume_sampling":"multiple_importance"}),
            json!({"op":"node.create","id":"overlapping_mesh","kind":"box","params":{"size":1.0},"material":"mesh_material"}),
        ]);
    }

    let batch = root.join("operations.json");
    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": operations,
        }))?,
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

fn heterogeneous_seed_means(
    root: &Path,
    kind: HeterogeneousScene,
    engine: &str,
    samples: u32,
) -> Result<Vec<[f64; 3]>, Box<dyn Error>> {
    let mut means = Vec::with_capacity(ORACLE_SEEDS.len());
    for seed in ORACLE_SEEDS {
        let seed_root = root.join(format!("{}_seed_{seed}", kind.label()));
        let scene = make_heterogeneous_scene(&seed_root, kind, samples, seed)?;
        means.push(render_mean(&scene, &seed_root.join("render"), engine)?);
    }
    Ok(means)
}

fn assert_matches_seed_oracle(
    description: &str,
    measured: &[[f64; 3]],
    blender_mean: [f64; 3],
    blender_stderr: [f64; 3],
) {
    let count = measured.len() as f64;
    let mut potter_mean = [0.0; 3];
    for render in measured {
        for channel in 0..3 {
            potter_mean[channel] += render[channel] / count;
        }
    }
    for channel in 0..3 {
        let sample_variance = measured
            .iter()
            .map(|render| (render[channel] - potter_mean[channel]).powi(2))
            .sum::<f64>()
            / (count - 1.0);
        let potter_stderr = (sample_variance / count).sqrt();
        let combined_stderr = (blender_stderr[channel].powi(2) + potter_stderr.powi(2)).sqrt();
        assert!(
            (potter_mean[channel] - blender_mean[channel]).abs() <= 3.0 * combined_stderr,
            "{description} channel {channel}: Potter {:.6} ± {:.6} vs Blender {:.6} ± {:.6}, outside three combined standard errors",
            potter_mean[channel],
            potter_stderr,
            blender_mean[channel],
            blender_stderr[channel],
        );
    }
}

#[test]
fn principled_heterogeneous_scattering_matches_blender_per_channel() -> Result<(), Box<dyn Error>> {
    let root = tempdir()?;
    let (blender_mean, blender_stderr) = spectral_oracle_case("principled_heterogeneous")?;
    let measured = heterogeneous_seed_means(
        root.path(),
        HeterogeneousScene::Principled,
        "path",
        HETEROGENEOUS_PATH_SAMPLES,
    )?;
    assert_matches_seed_oracle(
        "heterogeneous Principled volume",
        &measured,
        blender_mean,
        blender_stderr,
    );
    Ok(())
}

#[test]
fn mixed_voxel_mesh_scattering_matches_blender_per_channel() -> Result<(), Box<dyn Error>> {
    let root = tempdir()?;
    let (blender_mean, blender_stderr) = spectral_oracle_case("mixed_voxel_mesh")?;
    let measured = heterogeneous_seed_means(
        root.path(),
        HeterogeneousScene::MixedVoxelMesh,
        "path",
        HETEROGENEOUS_PATH_SAMPLES,
    )?;
    assert_matches_seed_oracle(
        "overlapping voxel and mesh volumes",
        &measured,
        blender_mean,
        blender_stderr,
    );
    Ok(())
}
