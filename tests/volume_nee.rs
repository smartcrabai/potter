//! Single-scatter direct lighting from a point light. The oracle is `tests/fixtures/volume_nee`, rendered by
//! Blender 5.2.2 with `generate_volume_nee_oracle.py`: a unit cube (a mesh with a volume-only material) of
//! Volume Scatter (Color 1, Density 0.05, so optical depth 0.05), under a point light of 100 W, with
//! `volume_bounces = max_bounces = 0`, which is single scatter. Potter renders the same budget (`max_bounces: 0`).
//! At a budget of 1 Cycles adds a second scatter vertex's direct light, so that budget is not single scatter.
//! The `homogeneous_*` cases are the mesh cube, and the `gradient_*` cases a voxel Linear gradient. The `diffuse`
//! case is a white Lambert plane under the same light and calibrates the light unit (I = P / 4π, irradiance
//! E = I / d², Lambert radiance E / π). A volume's in-scatter has no 1/π, since its phase function is per steradian.
//! The oracle frame is 4 x 4 and spans the unit cube (ortho scale 1), averaged over all 16 pixels with a one-pixel box
//! filter, which is how Potter integrates each pixel.
//! The voxel cubes use a 17-node grid (the cube's full extent, as Blender's VDB). The mesh cube is Potter's
//! closed-mesh medium.

use std::{error::Error, fs, path::Path, process::Output};

use assert_cmd::Command;
use serde_json::{Value, json};
use tempfile::tempdir;

const SIZE: u32 = 4;
const ENERGY: f64 = 100.0;
const POSITION: [f64; 3] = [1.6, 0.6, 0.9];
const DENSITY: f64 = 0.05;
const SAMPLES: u32 = 2048;
/// Voxel cubes take about 100 s per render at `SAMPLES` in a debug build (a mesh render takes about 1 s), so they use
/// fewer samples. Their tolerances are derived from their own measured spread at this count.
const VOXEL_SAMPLES: u32 = 512;
// Tolerances are three times the measured single-render spread of each configuration, rounded up. Potter's sd was
// measured over four seeds (each seed renders the scene once), on this scene: mesh MIS 0.34-0.36 % at 2048 spp, mesh
// distance 4.30 %, mesh equiangular 0.07 % (2048 spp); voxel MIS 2.78-2.93 %, gradient 4.52-4.65 %, voxel distance
// 1.54 %, voxel equiangular 0.64 % (512 spp); diffuse 0.11 % (2048 spp). Blender's own spread at 65536 spp is about
// 0.1 %, so it does not widen these. The tests use seed 0, so each result is fixed; the tolerance is the noise band
// a different seed could reach.
const DIFFUSE_TOLERANCE: f64 = 0.005;
const MESH_TOLERANCE: f64 = 0.015;
const MESH_DISTANCE_TOLERANCE: f64 = 0.13;
const MESH_EQUIANGULAR_TOLERANCE: f64 = 0.005;
const VOXEL_TOLERANCE: f64 = 0.09;
const VOXEL_DISTANCE_TOLERANCE: f64 = 0.05;
const VOXEL_EQUIANGULAR_TOLERANCE: f64 = 0.02;
const GRADIENT_TOLERANCE: f64 = 0.14;

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

/// The common scene: a black world, an orthographic camera at z = 3 (ortho scale 1, so the frame is the unit cube), a
/// `POSITION`, and single-scatter render settings.
fn base_operations(samples: u32) -> Vec<Value> {
    vec![
        json!({"op":"world.create","id":"world_main","color":[0.0,0.0,0.0],"strength":0.0}),
        json!({"op":"camera.create","id":"camera_main","transform":{"translation":[0.0,0.0,3.0]},"projection":"orthographic","ortho_scale":1.0}),
        json!({"op":"light.create","id":"light_main","light_type":"point","energy":ENERGY,"radius":0.0,"transform":{"translation":POSITION}}),
        json!({"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"camera_main","world":"world_main"}}),
        json!({"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_x":SIZE,"resolution_y":SIZE,"samples":samples,"seed":0,"max_bounces":0}}),
    ]
}

fn diffuse_scene(root: &Path) -> Result<std::path::PathBuf, Box<dyn Error>> {
    let mut operations = base_operations(SAMPLES);
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

/// How the scatter cube's density is represented.
#[derive(Clone, Copy)]
enum Medium {
    /// A unit mesh cube whose material is the closed-mesh medium, with the constant Density 0.05.
    MeshCube,
    /// A 17³ voxel grid over the unit cube (nodes at -0.5 + i / 16) with the constant density 0.05.
    VoxelHomogeneous,
    /// A 17³ voxel grid whose value at node i is i / 16, the Generated coordinate of the node, so the density is
    /// Blender's Linear gradient, 0.05 · (x + 0.5).
    VoxelGradient,
}

fn scatter_scene(
    root: &Path,
    anisotropy: f64,
    medium: Medium,
    sampling: &str,
) -> Result<std::path::PathBuf, Box<dyn Error>> {
    let samples = match medium {
        Medium::MeshCube => SAMPLES,
        Medium::VoxelHomogeneous | Medium::VoxelGradient => VOXEL_SAMPLES,
    };
    let mut operations = base_operations(samples);
    operations.extend([
        json!({"op":"graph.create","id":"medium_graph","kind":"shader"}),
        json!({"op":"graph.node_add","graph":"medium_graph","id":"output","type":"ShaderNodeOutputMaterial"}),
        json!({"op":"graph.node_add","graph":"medium_graph","id":"scatter","type":"ShaderNodeVolumeScatter","inputs":{"Color":[1.0,1.0,1.0,1.0],"Density":DENSITY,"Anisotropy":anisotropy}}),
        json!({"op":"graph.link","graph":"medium_graph","from_node":"scatter","from_socket":"Volume","to_node":"output","to_socket":"Volume"}),
        json!({"op":"material.create","id":"medium","graph":"medium_graph","volume_sampling":sampling}),
    ]);
    match medium {
        Medium::MeshCube => operations.push(
            json!({"op":"node.create","id":"cube_main","kind":"box","params":{"size":1.0},"material":"medium"}),
        ),
        Medium::VoxelHomogeneous | Medium::VoxelGradient => {
            // Blender's VDB for the same cube has 17 nodes at 1/16 spacing from -0.5, which covers the cube's full
            // extent. Potter's trilinear grid is zero outside its nodes, so 16 nodes would leave the far half-voxel empty.
            const N: usize = 17;
            let gradient = matches!(medium, Medium::VoxelGradient);
            let mut values = Vec::with_capacity(N * N * N);
            for _z in 0..N {
                for _y in 0..N {
                    for x in 0..N {
                        // Node x sits at world -0.5 + x / 16, so the Generated coordinate (x + 0.5) is x / 16 here.
                        values.push(if gradient { x as f64 / 16.0 } else { 1.0 });
                    }
                }
            }
            operations.push(json!({"op":"volume.create","id":"medium_volume","materials":["medium"],"grids":[{"dims":[N,N,N],"voxel_size":1.0 / 16.0,"origin":[-0.5,-0.5,-0.5],"values":values}]}));
        }
    }
    make_scene(root, &operations)
}

/// Renders the scene and returns the red channel's mean over all of its 4 × 4 pixels, the block the
/// oracle measures.
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
    for y in 0..SIZE as usize {
        for x in 0..SIZE as usize {
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
        "{case}: Potter {measured:.6} vs Blender {expected:.6} ({:.1} % apart, tolerance {:.1} %)",
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

/// Renders one case at its medium's sample count and compares its frame mean with Blender's mean for `case`.
fn assert_case_matches(
    case: &str,
    anisotropy: f64,
    medium: Medium,
    sampling: &str,
    tolerance: f64,
) -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = scatter_scene(directory.path(), anisotropy, medium, sampling)?;
    let measured = render_centre_mean(&scene, &directory.path().join("render"))?;
    assert_within(case, measured, tolerance)
}

#[test]
fn mesh_material_single_scatter_matches_blender_at_anisotropy_0() -> Result<(), Box<dyn Error>> {
    assert_case_matches(
        "homogeneous_g0",
        0.0,
        Medium::MeshCube,
        "multiple_importance",
        MESH_TOLERANCE,
    )
}

#[test]
fn mesh_material_single_scatter_matches_blender_at_anisotropy_0_5() -> Result<(), Box<dyn Error>> {
    assert_case_matches(
        "homogeneous_g05",
        0.5,
        Medium::MeshCube,
        "multiple_importance",
        MESH_TOLERANCE,
    )
}

#[test]
fn voxel_homogeneous_single_scatter_matches_blender_at_anisotropy_0() -> Result<(), Box<dyn Error>>
{
    assert_case_matches(
        "homogeneous_g0",
        0.0,
        Medium::VoxelHomogeneous,
        "multiple_importance",
        VOXEL_TOLERANCE,
    )
}

#[test]
fn voxel_homogeneous_single_scatter_matches_blender_at_anisotropy_0_5() -> Result<(), Box<dyn Error>>
{
    assert_case_matches(
        "homogeneous_g05",
        0.5,
        Medium::VoxelHomogeneous,
        "multiple_importance",
        VOXEL_TOLERANCE,
    )
}

#[test]
fn gradient_single_scatter_matches_blender_at_anisotropy_0() -> Result<(), Box<dyn Error>> {
    assert_case_matches(
        "gradient_g0",
        0.0,
        Medium::VoxelGradient,
        "multiple_importance",
        GRADIENT_TOLERANCE,
    )
}

#[test]
fn gradient_single_scatter_matches_blender_at_anisotropy_0_5() -> Result<(), Box<dyn Error>> {
    assert_case_matches(
        "gradient_g05",
        0.5,
        Medium::VoxelGradient,
        "multiple_importance",
        GRADIENT_TOLERANCE,
    )
}

// Distance and equiangular sampling are estimators of the same integral as multiple importance sampling, so each
// must reproduce Blender's homogeneous cube at anisotropy 0.
#[test]
fn distance_sampling_matches_blender_on_mesh_cube() -> Result<(), Box<dyn Error>> {
    assert_case_matches(
        "homogeneous_g0",
        0.0,
        Medium::MeshCube,
        "distance",
        MESH_DISTANCE_TOLERANCE,
    )
}

#[test]
fn distance_sampling_matches_blender_on_voxel_cube() -> Result<(), Box<dyn Error>> {
    assert_case_matches(
        "homogeneous_g0",
        0.0,
        Medium::VoxelHomogeneous,
        "distance",
        VOXEL_DISTANCE_TOLERANCE,
    )
}

#[test]
fn equiangular_sampling_matches_blender_on_mesh_cube() -> Result<(), Box<dyn Error>> {
    assert_case_matches(
        "homogeneous_g0",
        0.0,
        Medium::MeshCube,
        "equiangular",
        MESH_EQUIANGULAR_TOLERANCE,
    )
}

#[test]
fn equiangular_sampling_matches_blender_on_voxel_cube() -> Result<(), Box<dyn Error>> {
    assert_case_matches(
        "homogeneous_g0",
        0.0,
        Medium::VoxelHomogeneous,
        "equiangular",
        VOXEL_EQUIANGULAR_TOLERANCE,
    )
}
