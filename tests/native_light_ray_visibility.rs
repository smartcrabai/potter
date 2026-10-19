//! Light ray visibility in Potter's path tracer, checked against Blender's Cycles render of the same
//! scene.
//!
//! `fixtures/native_oracle/light_ray_visibility.py` builds a diffuse floor, a mirror sphere, and two
//! disk area lights, and renders the flagged scene and the default-flag scene with Cycles (96 x 96,
//! 128 samples, seed 0, one bounce, a 1 px box filter). The test runs it headless into a temporary
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

/// The scene builder and Cycles renders (`light_ray_visibility.py`).
const SCRIPT: &str = include_str!("fixtures/native_oracle/light_ray_visibility.py");

/// Render settings shared with `light_ray_visibility.py`.
const RESOLUTION: u32 = 96;
const SAMPLES: u32 = 128;
const MAX_BOUNCES: u32 = 1;
/// The orthographic frame spans this many metres on each axis, centred on the origin.
const FRAME_SIZE: f64 = 3.0;
/// Sample points per patch axis; each patch is the mean over `GRID * GRID` points.
const GRID: usize = 9;

/// Mean-RGB tolerance (linear), the bound `native_object_ray_visibility` uses for Cycles patches.
const PIXEL_TOLERANCE: f64 = 0.01;

/// (name, centre in metres, half width in metres). Each probe shows one light flag's effect:
/// - `highlight`: the sphere's centre, where it mirrors `GlossyOff` straight back up. `GlossyOff` is
///   `visible_glossy=False`, so the reference is dark here, while the default render is bright.
/// - `floor_diffuse`: the floor under `DiffuseOff`. `visible_diffuse=False` drops the disk's direct
///   diffuse light, so the reference is much darker than the default render.
/// - `floor_far`: a floor point that both lights reach directly, away from the sphere. `DiffuseOff`'s
///   share of it is dropped as well, so it is darker than the default render too.
const PATCHES: [(&str, [f64; 2], f64); 3] = [
    ("highlight", [0.0, 0.0], 0.03),
    ("floor_diffuse", [1.0, -0.9], 0.15),
    ("floor_far", [-1.0, 1.0], 0.1),
];

/// The mean RGB of the patch centred on `centre` (metres) with half width `half`.
fn patch_mean(data: &ImageData, centre: [f64; 2], half: f64) -> [f64; 3] {
    let mut total = [0.0_f64; 3];
    for row in 0..GRID {
        for column in 0..GRID {
            let offset = |index: usize| 2.0 * half * (index as f64 + 0.5) / GRID as f64 - half;
            let x = centre[0] + offset(column);
            let y = centre[1] + offset(row);
            let uv = [
                (x + FRAME_SIZE / 2.0) / FRAME_SIZE,
                (y + FRAME_SIZE / 2.0) / FRAME_SIZE,
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
fn light_ray_visibility_render_matches_cycles_patch_means() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "light_ray_visibility.py", SCRIPT)?;
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

    // The references must show each light flag's effect, so a regenerated reference that lost the
    // scene's point cannot pass by matching a broken render.
    let (highlight, half) = patch_centre("highlight")?;
    let glossy_off = patch_mean(&reference, highlight, half)[0];
    let glossy_on = patch_mean(&default, highlight, half)[0];
    assert!(
        glossy_off < 0.05 && glossy_on > 1.0,
        "GlossyOff must be dark in the flagged reference ({glossy_off}) and bright by default ({glossy_on})"
    );
    let (diffuse, half) = patch_centre("floor_diffuse")?;
    let diffuse_off = patch_mean(&reference, diffuse, half)[0];
    let diffuse_on = patch_mean(&default, diffuse, half)[0];
    assert!(
        diffuse_on - diffuse_off > 0.1,
        "DiffuseOff must darken the floor under it: flagged {diffuse_off}, default {diffuse_on}"
    );

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
