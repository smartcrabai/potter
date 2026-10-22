//! Object transmission visibility in Potter's path tracer, checked against Blender's Cycles render of
//! the same scene.
//!
//! `fixtures/native_oracle/glass_ray_visibility.py` renders a tinted glass sphere in front of an
//! emitting backdrop twice with Cycles (48 x 48, 96 samples, seed 0): once with the sphere's
//! `visible_transmission` off and once at the default. The test runs it headless into a temporary
//! directory and imports the flagged `.blend`. Potter renders it with the same settings, and both
//! images are reduced to the mean RGB of square patches in world metres on the 3 m frame (+Y up).

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

use std::path::Path;

use native_oracle::{
    TestResult, apply_operations, import_blend, pot, read_scene, run_blender_arguments,
    write_script,
};
use potter_core::image::{ImageData, ImageInterpolation, load_image_data, sample};
use potter_core::model::Image;
use serde_json::{Value, json};
use tempfile::tempdir;

/// The scene builder and Cycles renders (`glass_ray_visibility.py`).
const SCRIPT: &str = include_str!("fixtures/native_oracle/glass_ray_visibility.py");

/// Render settings shared with `glass_ray_visibility.py`.
const RESOLUTION: u32 = 48;
const SAMPLES: u32 = 96;
const MAX_BOUNCES: u32 = 8;
/// The orthographic frame spans this many metres on each axis, centred on the origin.
const FRAME_SIZE: f64 = 3.0;
/// Sample points per patch axis; each patch is the mean over `GRID * GRID` points.
const GRID: usize = 9;

/// Mean-RGB tolerance (linear), the bound the other Cycles patch oracles use.
const PIXEL_TOLERANCE: f64 = 0.01;

/// (name, centre in metres, half width in metres).
/// - `sphere_centre`: the sphere's centre, where a transmitted path refracts through the tinted
///   glass twice. With `visible_transmission` off the transmitted light is not tinted, so the
///   green channel is much higher than in the default render.
/// - `backdrop_control`: backdrop outside the sphere. Neither flag changes it.
const PATCHES: [(&str, [f64; 2], f64); 2] = [
    ("sphere_centre", [0.0, 0.0], 0.15),
    ("backdrop_control", [1.8, 0.0], 0.15),
];

/// The mean RGB of the patch centred on `centre` (metres) with half width `half`.
fn patch_mean(data: &ImageData, centre: [f64; 2], half: f64) -> [f64; 3] {
    let mut total = [0.0_f64; 3];
    for row in 0..GRID {
        for column in 0..GRID {
            let offset = |index: usize| 2.0 * half * (index as f64 + 0.5) / GRID as f64 - half;
            let x = centre[0] + offset(column);
            let y = centre[1] + offset(row);
            // Potter's `sample` takes V down: row 0 is the top, +Y. Measured against the Blender
            // reference, U runs with +X and V runs against +Y.
            let uv = [
                (x + FRAME_SIZE / 2.0) / FRAME_SIZE,
                (FRAME_SIZE / 2.0 - y) / FRAME_SIZE,
            ];
            let pixel = sample(data, uv, false);
            for (sum, value) in total.iter_mut().zip(pixel.iter().take(3)) {
                *sum += value;
            }
        }
    }
    let count = (GRID * GRID) as f64;
    total.map(|sum| sum / count)
}

/// The image `id` of the project's scene, decoded with the same sampler the renderer uses.
fn image_data(project: &Path, scene: &Value, id: &str) -> TestResult<ImageData> {
    let image: Image = serde_json::from_value(scene["images"][id].clone())?;
    Ok(load_image_data(
        &image,
        project,
        ImageInterpolation::Closest,
    )?)
}

fn patch_centre(name: &str) -> TestResult<([f64; 2], f64)> {
    PATCHES
        .iter()
        .copied()
        .find(|(patch, _, _)| *patch == name)
        .map(|(_, centre, half)| (centre, half))
        .ok_or_else(|| format!("no patch named {name}").into())
}

#[test]
fn glass_transmission_visibility_render_matches_cycles_patch_means() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "glass_ray_visibility.py", SCRIPT)?;
    let blend = root.path().join("scene.blend");
    let flagged_exr = root.path().join("blender_flagged.exr");
    let default_exr = root.path().join("blender_default.exr");
    run_blender_arguments(
        &blender,
        &script,
        None,
        &[
            blend.as_path(),
            flagged_exr.as_path(),
            default_exr.as_path(),
        ],
    )?;

    let (project, _) = import_blend(&blender, root.path(), "project", &blend)?;
    let scene = read_scene(&project)?;
    let active = scene["active_scene"]
        .as_str()
        .ok_or("active scene is missing")?
        .to_owned();

    let applied = apply_operations(
        &project,
        root.path(),
        json!([
            {"op": "render.update", "target": {"id": active}, "set": {"resolution_x": RESOLUTION, "resolution_y": RESOLUTION, "samples": SAMPLES, "seed": 0, "max_bounces": MAX_BOUNCES}}
        ]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");

    let frames = root.path().join("frames");
    let rendered = pot()
        .arg("render")
        .arg(&project)
        .args(["--format", "exr", "--out"])
        .arg(&frames)
        .arg("--json")
        .output()?;
    assert!(
        rendered.status.success(),
        "render failed: {}",
        String::from_utf8_lossy(&rendered.stdout)
    );
    let loaded = apply_operations(
        &project,
        root.path(),
        json!([{"op": "image.load", "id": "render", "path": frames.join("frame_0001.exr").to_string_lossy()}]),
    )?;
    assert_eq!(loaded["ok"], true, "{loaded}");
    for (id, path) in [("reference", &flagged_exr), ("default", &default_exr)] {
        let referenced = apply_operations(
            &project,
            root.path(),
            json!([{"op": "image.load", "id": id, "path": path.to_string_lossy()}]),
        )?;
        assert_eq!(referenced["ok"], true, "{referenced}");
    }

    let scene = read_scene(&project)?;
    let potter = image_data(&project, &scene, "render")?;
    let reference = image_data(&project, &scene, "reference")?;
    let default = image_data(&project, &scene, "default")?;

    // The references must show the flag's effect, so a regenerated reference that lost the scene's
    // point cannot pass by matching a broken render. With the flag off the sphere's centre keeps the
    // untinted backdrop, which is greener than the tinted default render. The backdrop control is
    // unchanged by the flag.
    let (centre, half) = patch_centre("sphere_centre")?;
    let flagged_green = patch_mean(&reference, centre, half)[1];
    let default_green = patch_mean(&default, centre, half)[1];
    assert!(
        flagged_green - default_green > 0.08,
        "visible_transmission off must leave the sphere's centre greener: flagged {flagged_green}, default {default_green}"
    );
    let (control, half) = patch_centre("backdrop_control")?;
    let flagged_control = patch_mean(&reference, control, half);
    let default_control = patch_mean(&default, control, half);
    for (channel, (flagged, unflagged)) in flagged_control.iter().zip(default_control).enumerate() {
        assert!(
            (flagged - unflagged).abs() <= PIXEL_TOLERANCE,
            "the backdrop control must not change with the flag, channel {channel}: {flagged} vs {unflagged}"
        );
    }

    for (name, centre, half) in PATCHES {
        let wanted = patch_mean(&reference, centre, half);
        let actual = patch_mean(&potter, centre, half);
        for (channel, (actual, wanted)) in actual.iter().zip(wanted).enumerate() {
            assert!(
                (actual - wanted).abs() <= PIXEL_TOLERANCE,
                "{name} channel {channel}: Potter {actual} differs from Cycles {wanted}"
            );
        }
    }
    Ok(())
}
