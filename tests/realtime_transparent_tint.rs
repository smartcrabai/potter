//! Realtime Beauty tints what a Transparent BSDF passes, as EEVEE 5.2.2 does, for BLENDED and DITHERED panes.
//!
//! `realtime_transparent_tint.py` renders a pane whose Transparent BSDF has the Color (0.2, 0.8, 0.4) in front of an
//! emitter of radiance 1. A BLENDED pane keeps the Color per channel, so the pixel is the Color. A DITHERED pane stores
//! one alpha, the Color's channel mean, so the pixel is that mean on every channel. EEVEE's means and tolerances are
//! committed in `realtime_transparent_tint.json`. This test writes each scene with Blender, renders only Potter, and
//! compares the mean of a 9 by 9 grid over the frame with EEVEE's.

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
use potter_core::image::{ImageData, ImageInterpolation, load_image_data, sample};
use potter_core::model::Image;
use serde_json::{Value, json};
use tempfile::tempdir;

const SCRIPT: &str = include_str!("fixtures/native_oracle/realtime_transparent_tint.py");
/// EEVEE's patch means and tolerances (`realtime_transparent_tint.json`, with the notes on each method).
const EEVEE: &str = include_str!("fixtures/native_oracle/realtime_transparent_tint.json");
/// Patch grid of the Blender oracle (`realtime_transparent_tint.py`): fractions of the frame.
const PATCH: [f64; 9] = [
    0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75,
];
/// The render methods of the oracle's pane: BLENDED keeps the Color per channel, DITHERED stores its mean.
const METHODS: [&str; 2] = ["BLENDED", "DITHERED"];

/// EEVEE's patch mean per channel and tolerance for `method`.
fn eevee(method: &str) -> TestResult<([f64; 3], f64)> {
    let values: Value = serde_json::from_str(EEVEE)?;
    let entry = &values["methods"][method];
    let means = entry["patch_mean"]
        .as_array()
        .ok_or_else(|| format!("no EEVEE patch mean for {method}"))?;
    let mut mean = [0.0; 3];
    for (value, slot) in means.iter().zip(mean.iter_mut()) {
        *slot = value
            .as_f64()
            .ok_or_else(|| format!("EEVEE patch mean for {method} is not a number"))?;
    }
    let tolerance = entry["tolerance"]
        .as_f64()
        .ok_or_else(|| format!("no EEVEE tolerance for {method}"))?;
    Ok((mean, tolerance))
}

/// Writes the Blender scene of `method` as a .blend, without rendering it, and returns its path.
fn blender_scene(blender: &Path, root: &Path, method: &str) -> TestResult<PathBuf> {
    let dir = root.join(method);
    fs::create_dir_all(&dir)?;
    let script = write_script(&dir, "realtime_transparent_tint.py", SCRIPT)?;
    let blend = dir.join("scene.blend");
    // The render flag `0` stops the generator after saving the .blend, so the image and JSON paths are not written.
    run_blender_arguments(
        blender,
        &script,
        None,
        &[
            blend.as_path(),
            dir.join("unused.exr").as_path(),
            dir.join("unused.json").as_path(),
            Path::new(method),
            Path::new("0"),
        ],
    )?;
    Ok(blend)
}

/// Imports the Blender scene into its own project, since `pot init` refuses a non-empty directory.
fn import_scene(blender: &Path, root: &Path, blend: &Path) -> TestResult<PathBuf> {
    let name = blend
        .parent()
        .and_then(Path::file_name)
        .and_then(|name| name.to_str())
        .ok_or("the blend has no directory name")?;
    let (project, _) = import_blend(blender, root, &format!("project_{name}"), blend)?;
    let scene = read_scene(&project)?;
    let active = scene["active_scene"]
        .as_str()
        .ok_or("active scene is missing")?
        .to_owned();
    let applied = apply_operations(
        &project,
        root,
        json!([
            {"op": "render.update", "target": {"id": active}, "set": {"resolution_x": 64, "resolution_y": 64, "samples": 16, "seed": 0, "max_bounces": 4}}
        ]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");
    Ok(project)
}

/// Renders `project` in the realtime engine to a linear EXR and loads the frame.
fn potter_realtime_frame(project: &Path, root: &Path, name: &str) -> TestResult<ImageData> {
    let frames = root.join(name);
    let mut render = pot();
    render
        .arg("render")
        .arg(project)
        .args(["--engine", "realtime", "--format", "exr", "--out"])
        .arg(&frames)
        .arg("--json");
    checked(&mut render, name)?;
    let loaded = apply_operations(
        project,
        root,
        json!([{"op": "image.load", "id": name, "path": frames.join("frame_0001.exr").to_string_lossy()}]),
    )?;
    assert_eq!(loaded["ok"], true, "{loaded}");
    let scene = read_scene(project)?;
    let image: Image = serde_json::from_value(scene["images"][name].clone())?;
    Ok(load_image_data(
        &image,
        project,
        ImageInterpolation::Closest,
    )?)
}

/// The mean per RGB channel of the 9 by 9 grid of a rendered frame, as the Blender oracle takes it.
fn patch_mean(data: &ImageData) -> [f64; 3] {
    let mut total = [0.0; 3];
    for v in PATCH {
        for u in PATCH {
            let pixel = sample(data, [u, v], false);
            for (sum, value) in total.iter_mut().zip(pixel) {
                *sum += value;
            }
        }
    }
    let count = (PATCH.len() * PATCH.len()) as f64;
    total.map(|sum| sum / count)
}

#[test]
fn transparent_tint_composites_as_eevee_does() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping the realtime transparent tint oracle; no executable Blender was found");
        return Ok(());
    };
    let root = tempdir()?;
    for method in METHODS {
        let blend = blender_scene(&blender, root.path(), method)?;
        let project = import_scene(&blender, root.path(), &blend)?;
        let frame = potter_realtime_frame(&project, root.path(), &method.to_lowercase())?;
        let actual = patch_mean(&frame);
        let (expected, tolerance) = eevee(method)?;
        for (channel, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "{method} channel {channel}: Potter {actual} differs from EEVEE {expected} by more than {tolerance}"
            );
        }
    }
    Ok(())
}
