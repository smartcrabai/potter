//! A DITHERED material's graph transparency against Blender's Cycles and EEVEE renders.
//!
//! The surface is `Mix Shader(0.5, Transparent, Principled)` in a DITHERED material, in front of an emitter plane.
//! Cycles ignores the render method and honours the graph, so the emitter shows through half the rays. EEVEE dithers
//! the same graph. Both renders reduce to the mean RGB of a central patch, and Potter's path tracer and realtime
//! Beauty must match them. Potter imports the `.blend` as DITHERED `opaque`, so the graph alone sets transparency:
//! the path tracer forced alpha 1 for `opaque` before, and the realtime raster ignored the graph's alpha.

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

use std::{
    fs,
    path::{Path, PathBuf},
};

use blender_checked::blender_executable;
use native_oracle::{
    TestResult, apply_operations, checked, import_blend, pot, read_scene, run_blender_arguments,
    write_script,
};
use potter_core::image::{ImageInterpolation, load_image_data, sample};
use potter_core::model::Image;
use serde_json::{Value, json};
use tempfile::tempdir;

const SCRIPT: &str = include_str!("fixtures/native_oracle/graph_transparent.py");
/// Patch grid of the Blender oracle (`graph_transparent.py`): fractions of the frame.
const PATCH: [f64; 9] = [
    0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75,
];
/// The patch mean the script's constants give: half the emitter (0.1, 0.6, 0.2) and half the diffuse albedo
/// (0.8, 0.3, 0.1) lit by the uniform world (0.5).
const EXPECTED: [f64; 3] = [0.25, 0.375, 0.125];
/// Linear render tolerance. Both estimators average 81 patch pixels, so this covers the sampling noise of 64
/// samples per pixel with margin, and it is far below the 0.25 a missing graph transparency causes.
const PIXEL_TOLERANCE: f64 = 0.01;

/// Runs the oracle script with `engine` (`CYCLES` or `BLENDER_EEVEE`); returns the scene it wrote and the patch mean.
fn blender_render(blender: &Path, root: &Path, engine: &str) -> TestResult<(PathBuf, Vec<f64>)> {
    let dir = root.join(engine);
    fs::create_dir_all(&dir)?;
    let script = write_script(&dir, "graph_transparent.py", SCRIPT)?;
    let blend = dir.join("scene.blend");
    let image = dir.join("blender.exr");
    let json_path = dir.join("blender.json");
    run_blender_arguments(
        blender,
        &script,
        None,
        &[
            blend.as_path(),
            image.as_path(),
            json_path.as_path(),
            Path::new(engine),
        ],
    )?;
    let reference: Value = serde_json::from_slice(&fs::read(&json_path)?)?;
    let mean = reference["patch_mean"]
        .as_array()
        .ok_or("the Blender patch mean is missing")?
        .iter()
        .map(|value| {
            value
                .as_f64()
                .ok_or_else(|| "the Blender patch mean is not a number".into())
        })
        .collect::<TestResult<Vec<f64>>>()?;
    Ok((blend, mean))
}

/// Imports the Blender scene, sets its render size and samples, and returns the project and its active scene.
fn import_scene(blender: &Path, root: &Path, blend: &Path) -> TestResult<(PathBuf, String)> {
    let (project, _) = import_blend(blender, root, "project", blend)?;
    let scene = read_scene(&project)?;
    let active = scene["active_scene"]
        .as_str()
        .ok_or("active scene is missing")?
        .to_owned();
    let applied = apply_operations(
        &project,
        root,
        json!([
            {"op": "render.update", "target": {"id": active}, "set": {"resolution_x": 64, "resolution_y": 64, "samples": 64, "seed": 0, "max_bounces": 4}}
        ]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");
    Ok((project, active))
}

/// The patch mean of Potter's render of `project` with the `pot render` arguments `render_args`, read back as linear EXR.
fn potter_patch_mean(
    project: &Path,
    root: &Path,
    name: &str,
    render_args: &[&str],
) -> TestResult<Vec<f64>> {
    let frames = root.join(name);
    checked(
        pot()
            .arg("render")
            .arg(project)
            .args(render_args)
            .args(["--format", "exr", "--out"])
            .arg(&frames)
            .arg("--json"),
        name,
    )?;
    let loaded = apply_operations(
        project,
        root,
        json!([{"op": "image.load", "id": name, "path": frames.join("frame_0001.exr").to_string_lossy()}]),
    )?;
    assert_eq!(loaded["ok"], true, "{loaded}");
    let scene = read_scene(project)?;
    let image: Image = serde_json::from_value(scene["images"][name].clone())?;
    let data = load_image_data(&image, project, ImageInterpolation::Closest)?;
    let mut total = [0.0_f64; 3];
    for v in PATCH {
        for u in PATCH {
            let pixel = sample(&data, [u, v], false);
            for (sum, value) in total.iter_mut().zip(pixel.iter().take(3)) {
                *sum += value;
            }
        }
    }
    let count = (PATCH.len() * PATCH.len()) as f64;
    Ok(total.iter().map(|sum| sum / count).collect())
}

fn assert_patch_matches(actual: &[f64], wanted: &[f64], renderer: &str) {
    for (channel, (actual, wanted)) in actual.iter().zip(wanted).enumerate() {
        assert!(
            (actual - wanted).abs() <= PIXEL_TOLERANCE,
            "channel {channel}: Potter {actual} differs from {renderer} {wanted}"
        );
    }
}

#[test]
fn dithered_graph_transparency_matches_cycles_in_the_path_tracer() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping the graph transparency oracle; no executable Blender was found");
        return Ok(());
    };
    let root = tempdir()?;
    let (blend, wanted) = blender_render(&blender, root.path(), "CYCLES")?;
    assert_patch_matches(&wanted, &EXPECTED, "the analytic Cycles mean");

    let (project, _) = import_scene(&blender, root.path(), &blend)?;
    let actual = potter_patch_mean(&project, root.path(), "path", &[])?;
    assert_patch_matches(&actual, &wanted, "Cycles");
    Ok(())
}

#[test]
fn dithered_graph_transparency_matches_eevee_in_realtime() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!(
            "skipping the graph transparency realtime oracle; no executable Blender was found"
        );
        return Ok(());
    };
    let root = tempdir()?;
    let (blend, wanted) = blender_render(&blender, root.path(), "BLENDER_EEVEE")?;
    assert_patch_matches(&wanted, &EXPECTED, "the analytic EEVEE mean");

    let (project, _) = import_scene(&blender, root.path(), &blend)?;
    let actual = potter_patch_mean(&project, root.path(), "realtime", &["--engine", "realtime"])?;
    assert_patch_matches(&actual, &wanted, "EEVEE");
    Ok(())
}
