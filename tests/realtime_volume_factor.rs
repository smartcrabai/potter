//! Realtime Beauty's volume in-scatter follows a light's `volume_factor`, as EEVEE's `power[LIGHT_VOLUME]` does.
//!
//! `fixtures/native_oracle/eevee_volume_factor.py` renders a pure scatterer, a 2 m cube, lit by a 200 W point light above it,
//! three times: with the light's Volume Factor at 1, at 0, and at 0.5. The black world leaves the in-scatter alone, so the
//! three renders test one linear law: the in-scatter is proportional to the factor, and a factor of 0 switches it off.
//!
//! - EEVEE: the factor-0.5 render is half the factor-1 render within 1.5e-3 (EEVEE's own sampling noise: the largest pixel
//!   deviation measured over the 96 by 96 image is 7.0e-4, and 1.5e-3 is twice that, rounded up); the factor-0 render is black.
//! - Potter: the same three relations hold for Potter's renders, to 1e-6, since the in-scatter is multiplied by the factor
//!   exactly.
//!
//! The absolute in-scatter is not compared with EEVEE: Potter's realtime volume is a single-scatter model with Cycles'
//! light radiance, which differs from EEVEE's volume light (`point_radiance_get`) for area and sun lights.
//!
//! EEVEE's three renders are the committed macOS arm64 renders (`fixtures/native_oracle/eevee_macos/`; see its
//! `MANIFEST.json`), not live renders, so the test has no platform term.
//!
//! Blender-gated: with `POTTER_REQUIRE_BLENDER` set, a missing Blender fails the test.

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/native_oracle.rs"]
#[expect(
    dead_code,
    reason = "each test binary uses a subset of the shared oracle helpers"
)]
mod native_oracle;
#[path = "common/process.rs"]
mod process;

use std::{error::Error, path::Path};

use native_oracle::{
    TestResult, apply_operations, import_blend, pot, read_scene, run_blender_arguments,
    write_script,
};
use potter_core::image::{ImageData, ImageInterpolation, load_image_data, sample};
use potter_core::model::Image;
use serde_json::json;
use tempfile::tempdir;

/// The oracle's script: it saves each case's `.blend` and, given `-` for the render paths, stops before rendering.
const EEVEE_SCRIPT: &str = include_str!("fixtures/native_oracle/eevee_volume_factor.py");

/// The three cases with their committed macOS arm64 EEVEE renders, as little-endian f32 RGB rows bottom-up.
const EEVEE_RENDERS: [(&str, &[u8]); 3] = [
    (
        "volume_base",
        include_bytes!("fixtures/native_oracle/eevee_macos/volume_volume_base.f32le"),
    ),
    (
        "volume_zero",
        include_bytes!("fixtures/native_oracle/eevee_macos/volume_volume_zero.f32le"),
    ),
    (
        "volume_half",
        include_bytes!("fixtures/native_oracle/eevee_macos/volume_volume_half.f32le"),
    ),
];

const SIZE: u32 = 96;
const ORTHO_SCALE: f64 = 3.0;

/// The EEVEE half-factor bound (see the module docs).
const EEVEE_LINEAR_TOLERANCE: f64 = 1.5e-3;
/// Potter's relations are exact up to the render's float precision.
const POTTER_LINEAR_TOLERANCE: f64 = 1.0e-6;

#[test]
fn realtime_volume_in_scatter_follows_the_light_volume_factor() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "eevee_volume_factor.py", EEVEE_SCRIPT)?;
    let mut eevee = Vec::new();
    let mut potter = Vec::new();
    for (case, eevee_bytes) in EEVEE_RENDERS {
        let blend = root.path().join(format!("{case}.blend"));
        run_blender_arguments(
            &blender,
            &script,
            None,
            &[
                Path::new(case),
                blend.as_path(),
                Path::new("-"),
                Path::new("-"),
            ],
        )?;
        eevee.push(eevee_red(eevee_bytes)?);
        let data = render_potter(&blender, root.path(), case, &blend)?;
        potter.push(plane_values(&data));
    }
    let [eevee_base, eevee_zero, eevee_half] =
        <[Vec<f64>; 3]>::try_from(eevee).map_err(|_| "three EEVEE renders are expected")?;
    let [potter_base, potter_zero, potter_half] =
        <[Vec<f64>; 3]>::try_from(potter).map_err(|_| "three Potter renders are expected")?;

    let eevee_half_error = worst_difference(&eevee_half, &scaled(&eevee_base, 0.5));
    assert!(
        eevee_half_error <= EEVEE_LINEAR_TOLERANCE,
        "EEVEE's half-factor render is not half its base: error {eevee_half_error}"
    );
    let eevee_zero_max = eevee_zero.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
    assert!(
        eevee_zero_max <= EEVEE_LINEAR_TOLERANCE,
        "EEVEE's zero-factor render is not black: {eevee_zero_max}"
    );

    let potter_zero_max = potter_zero.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
    assert!(
        potter_zero_max <= POTTER_LINEAR_TOLERANCE,
        "Potter's zero-factor render keeps in-scatter: {potter_zero_max}"
    );
    let potter_half_error = worst_difference(&potter_half, &scaled(&potter_base, 0.5));
    assert!(
        potter_half_error <= POTTER_LINEAR_TOLERANCE,
        "Potter's half-factor render is not half its base: error {potter_half_error}"
    );
    let potter_base_max = potter_base.iter().fold(0.0_f64, |a, b| a.max(*b));
    assert!(
        potter_base_max > 1.0e-3,
        "Potter's volume shows no in-scatter, so the factor is not tested ({potter_base_max})"
    );
    Ok(())
}

/// A linked Volume output with no Surface stays transparent even when its sampled medium is zero.
#[test]
fn realtime_zero_density_volume_only_mesh_matches_no_object() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "eevee_volume_factor.py", EEVEE_SCRIPT)?;
    let blend = root.path().join("volume_only_zero_density.blend");
    run_blender_arguments(
        &blender,
        &script,
        None,
        &[
            Path::new(EEVEE_RENDERS[0].0),
            blend.as_path(),
            Path::new("-"),
            Path::new("-"),
        ],
    )?;
    let (project, _) = import_blend(&blender, root.path(), "zero_density", &blend)?;
    let scene = read_scene(&project)?;
    let material = scene["materials"]
        .as_object()
        .ok_or("imported materials are missing")?
        .values()
        .find(|material| material["name"] == "Scatter")
        .ok_or("the Scatter material is missing")?;
    let graph_id = material["node_tree"]
        .as_str()
        .ok_or("the Scatter graph is missing")?;
    let graph = scene["node_groups"]
        .get(graph_id)
        .ok_or("the Scatter graph data is missing")?;
    let nodes = graph["nodes"]
        .as_object()
        .ok_or("the Scatter graph nodes are missing")?;
    let (volume_node_id, volume_node) = nodes
        .iter()
        .find(|(_, node)| node["type"] == "ShaderNodeVolumeScatter")
        .ok_or("the Volume Scatter node is missing")?;
    let mut volume_inputs = volume_node["inputs"]
        .as_object()
        .cloned()
        .ok_or("the Volume Scatter inputs are missing")?;
    volume_inputs.insert("Density".to_owned(), json!(0.0));
    let output_node_id = nodes
        .iter()
        .find(|(_, node)| {
            matches!(
                node["type"].as_str(),
                Some("OutputMaterial" | "ShaderNodeOutputMaterial")
            )
        })
        .map(|(id, _)| id.as_str())
        .ok_or("the Material Output node is missing")?;
    let surface_link = graph["links"]
        .as_array()
        .and_then(|links| {
            links.iter().find(|link| {
                link["to_node"].as_str() == Some(output_node_id)
                    && link["to_socket"] == "Surface"
            })
        })
        .ok_or("the transparent Surface link is missing")?;
    let surface_link = [
        surface_link["from_node"]
            .as_str()
            .ok_or("the Surface source node is missing")?,
        surface_link["from_socket"]
            .as_str()
            .ok_or("the Surface source socket is missing")?,
        surface_link["to_node"]
            .as_str()
            .ok_or("the Surface destination node is missing")?,
        surface_link["to_socket"]
            .as_str()
            .ok_or("the Surface destination socket is missing")?,
    ];
    let active_scene = scene["active_scene"]
        .as_str()
        .ok_or("active scene is missing")?;
    let configured = apply_operations(
        &project,
        root.path(),
        json!([
            {"op": "graph.node_update", "graph": graph_id, "node_id": volume_node_id,
             "set": {"inputs": volume_inputs}},
            {"op": "graph.unlink", "graph": graph_id,
             "from_node": surface_link[0], "from_socket": surface_link[1],
             "to_node": surface_link[2], "to_socket": surface_link[3]},
            {"op": "camera.create", "id": "camera_main",
             "transform": {"translation": [0.0, 0.0, 10.0]},
             "projection": "orthographic", "ortho_scale": ORTHO_SCALE},
            {"op": "scene.update", "target": {"id": active_scene}, "set": {"camera": "camera_main"}},
            {"op": "render.update", "target": {"id": active_scene},
             "set": {"resolution_x": SIZE, "resolution_y": SIZE, "samples": 1, "seed": 0}}
        ]),
    )?;
    assert_eq!(configured["ok"], true, "{configured}");
    let volume_only = render_project(root.path(), &project, "volume_only")?;
    let cube_id = native_oracle::node_id(&scene, "Cube")?;
    let hidden = apply_operations(
        &project,
        root.path(),
        json!([{"op": "node.update", "target": {"id": cube_id}, "set": {"visible": false}}]),
    )?;
    assert_eq!(hidden["ok"], true, "{hidden}");
    let no_object = render_project(root.path(), &project, "no_object")?;
    let difference = worst_difference(&plane_values(&volume_only), &plane_values(&no_object));
    assert!(
        difference <= POTTER_LINEAR_TOLERANCE,
        "a zero-density Volume-only mesh changes the no-object image: {difference}"
    );
    Ok(())
}

/// The committed EEVEE render's red channel, one value per pixel in Blender's row order. A volume is grey, so one
/// channel carries it.
fn eevee_red(bytes: &[u8]) -> Result<Vec<f64>, Box<dyn Error>> {
    let pixel_count = (SIZE * SIZE) as usize;
    if bytes.len() != pixel_count * 12 {
        return Err(format!(
            "the EEVEE fixture holds {} bytes, expected {}",
            bytes.len(),
            pixel_count * 12
        )
        .into());
    }
    Ok((0..pixel_count)
        .map(|pixel| {
            let at = pixel * 12;
            f64::from(f32::from_le_bytes([
                bytes[at],
                bytes[at + 1],
                bytes[at + 2],
                bytes[at + 3],
            ]))
        })
        .collect())
}

/// Potter's red channel at the oracle's pixel centres, in the same order (row by row, bottom row first).
fn plane_values(data: &ImageData) -> Vec<f64> {
    let count = SIZE as usize;
    (0..count)
        .flat_map(|row| (0..count).map(move |column| (column, row)))
        .map(|(column, row)| {
            let uv = [
                (column as f64 + 0.5) / count as f64,
                (row as f64 + 0.5) / count as f64,
            ];
            sample(data, uv, false)[0]
        })
        .collect()
}

fn scaled(values: &[f64], factor: f64) -> Vec<f64> {
    values.iter().map(|value| value * factor).collect()
}

fn worst_difference(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0_f64, f64::max)
}

/// Renders a case's `.blend` in Potter's realtime engine with the oracle's camera and size, and loads the frame.
fn render_potter(
    blender: &Path,
    root: &Path,
    case: &str,
    blend: &Path,
) -> Result<ImageData, Box<dyn Error>> {
    let project_name = format!("project_{case}");
    let (project, _) = import_blend(blender, root, &project_name, blend)?;
    let scene_data = read_scene(&project)?;
    let active = scene_data["active_scene"]
        .as_str()
        .ok_or("active scene is missing")?
        .to_owned();
    let applied = apply_operations(
        &project,
        root,
        json!([
            {"op": "camera.create", "id": "camera_main", "transform": {"translation": [0.0, 0.0, 10.0]}, "projection": "orthographic", "ortho_scale": ORTHO_SCALE},
            {"op": "scene.update", "target": {"id": active}, "set": {"camera": "camera_main"}},
            {"op": "render.update", "target": {"id": active}, "set": {"resolution_x": SIZE, "resolution_y": SIZE, "samples": 1, "seed": 0}}
        ]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");
    render_project(root, &project, case)
}

fn render_project(
    root: &Path,
    project: &Path,
    case: &str,
) -> Result<ImageData, Box<dyn Error>> {
    let frames = root.join(format!("frames_{case}"));
    let rendered = pot()
        .arg("render")
        .arg(project)
        .args(["--engine", "realtime", "--format", "exr", "--out"])
        .arg(&frames)
        .arg("--json")
        .output()?;
    assert!(
        rendered.status.success(),
        "{case}: realtime render failed: {}",
        String::from_utf8_lossy(&rendered.stdout)
    );
    let image_id = format!("render_{case}");
    let loaded = apply_operations(
        project,
        root,
        json!([{"op": "image.load", "id": image_id, "path": frames.join("frame_0001.exr").to_string_lossy()}]),
    )?;
    assert_eq!(loaded["ok"], true, "{loaded}");
    let image: Image =
        serde_json::from_value(read_scene(project)?["images"][image_id.as_str()].clone())?;
    Ok(load_image_data(
        &image,
        project,
        ImageInterpolation::Closest,
    )?)
}
