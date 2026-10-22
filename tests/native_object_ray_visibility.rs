//! Object ray visibility in Potter's path tracer, checked against Blender's Cycles render of the
//! same scene.
//!
//! `fixtures/native_oracle/object_ray_visibility.py` builds the scene and renders it with Cycles
//! (64 x 64, 32 samples, seed 0, two bounces, a 1 px box filter). The test runs it headless into a
//! temporary directory, which writes the `.blend` and the linear EXR. Potter imports the `.blend`,
//! renders it with the same settings, and both images are reduced to the mean RGB of six square
//! patches. Each patch is in world metres on the 3 m square frame (+Y up). The patches sit where a
//! flag changes the image.

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

/// The scene builder and Cycles render (`object_ray_visibility.py`).
const SCRIPT: &str = include_str!("fixtures/native_oracle/object_ray_visibility.py");

/// Render settings shared with `object_ray_visibility.py`.
const RESOLUTION: u32 = 64;
const SAMPLES: u32 = 32;
const MAX_BOUNCES: u32 = 2;
/// The orthographic frame spans this many metres on each axis, centred on the origin.
const FRAME_SIZE: f64 = 3.0;
/// Sample points per patch axis; each patch is the mean over `GRID * GRID` points.
const GRID: usize = 9;

/// Mean-RGB tolerance (linear), the bound `native_point_spot_light` uses for Cycles patch means.
const PIXEL_TOLERANCE: f64 = 0.01;

/// (name, centre in metres, half width in metres). Each probe shows one flag's effect:
/// - `floor_lit`: lit floor away from every object and shadow; scales the light.
/// - `cube_camera`: the `HiddenCube` footprint. `visible_camera=False` shows floor here, not the cube.
/// - `cube_shadow`: the `HiddenCube`'s shadow on the floor. `visible_shadow` stays on, so it is dark.
/// - `occluder_top`: the `Occluder`'s top, which the camera sees.
/// - `occluder_shadow`: where the `Occluder`'s shadow would fall. `visible_shadow=False` keeps it lit.
/// - `mirror_glossy`: the `Reflector`'s disc, where it reflects the `GlossyOffMirror` only while that
///   mirror is `visible_glossy`.
const PATCHES: [(&str, [f64; 2], f64); 6] = [
    ("floor_lit", [-1.0, 1.0], 0.1),
    ("cube_camera", [-0.5, 0.45], 0.1),
    ("cube_shadow", [0.0, 0.8], 0.1),
    ("occluder_top", [0.7, 0.6], 0.1),
    ("occluder_shadow", [1.1, 0.9], 0.08),
    ("mirror_glossy", [-0.8, -0.5], 0.06),
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

#[test]
fn object_ray_visibility_render_matches_cycles_patch_means() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "object_ray_visibility.py", SCRIPT)?;
    let blend = root.path().join("scene.blend");
    let reference_exr = root.path().join("blender_reference.exr");
    run_blender_arguments(
        &blender,
        &script,
        None,
        &[blend.as_path(), reference_exr.as_path()],
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
    let referenced = apply_operations(
        &project,
        root.path(),
        json!([{"op": "image.load", "id": "reference", "path": reference_exr.to_string_lossy()}]),
    )?;
    assert_eq!(referenced["ok"], true, "{referenced}");

    let scene = read_scene(&project)?;
    let potter = image_data(&project, &scene, "render")?;
    let reference = image_data(&project, &scene, "reference")?;

    // The reference itself must show each flag's effect, so a regenerated reference that lost the
    // scene's point cannot pass by matching a broken render.
    let lit = patch_mean(&reference, [-1.0, 1.0], 0.1);
    assert!(
        lit.iter().all(|value| *value > 0.1),
        "the Cycles reference floor must be lit, got {lit:?}"
    );
    for (name, expect_dark) in [
        ("cube_camera", true),
        ("cube_shadow", true),
        ("occluder_shadow", false),
        ("mirror_glossy", true),
    ] {
        let (_, centre, half) = PATCHES
            .iter()
            .copied()
            .find(|(patch, _, _)| *patch == name)
            .ok_or_else(|| format!("no patch named {name}"))?;
        let mean = patch_mean(&reference, centre, half)[0];
        assert_eq!(
            mean < 0.05,
            expect_dark,
            "{name}: the Cycles reference mean {mean} must be {}",
            if expect_dark { "dark" } else { "lit" }
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
