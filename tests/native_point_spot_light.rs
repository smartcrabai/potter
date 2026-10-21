//! Point and spot light intensity against Blender's Cycles render.
//!
//! Blender builds a diffuse plane under one light, saves it as a `.blend`, and renders it. Potter
//! imports that `.blend`, renders it, and both images are reduced to the mean RGB of the same
//! central patch of the plane. Cycles gives a point light radiant intensity `P / (4π)` and a spot
//! light the same intensity inside a `smoothstep` cone (`intern/cycles/scene/light.cpp`,
//! `kernel/light/spot.h`); the three cases cover the soft-falloff disk, the sphere, and the spot.

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

use std::{fs, path::Path};

use native_oracle::{
    TestResult, apply_operations, import_blend, pot, read_scene, run_blender_arguments,
    write_script,
};
use potter_core::image::{ImageInterpolation, load_image_data, sample};
use potter_core::model::Image;
use serde_json::{Value, json};
use tempfile::tempdir;

const SCRIPT: &str = include_str!("fixtures/native_oracle/point_spot_light.py");
/// Patch grid of the Blender oracle (`point_spot_light.py`): fractions of the frame.
const PATCH: [f64; 9] = [
    0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75,
];
/// Render-to-stored-value tolerance (linear units), the same bound as `native_render_color`. The
/// patch means are about 0.5, so this is a 2% band; a 4π intensity error is a 12.6× difference.
const PIXEL_TOLERANCE: f64 = 0.01;
/// Samples per pixel of the point lights. Their patch-mean noise at this count is about 6 x 10^-4
/// per seed, a margin of about 12 standard errors against the tolerance (`point_spot_light.py`).
const POINT_SAMPLES: u32 = 8;
/// Samples per pixel of the spot light. Its noise is about 2.5 x 10^-3 per seed at 8 samples, a
/// margin of only about 3, so it keeps 64, where the margin is about 13.
const SPOT_SAMPLES: u32 = 64;

/// Renders `kind` at `samples` per pixel in Blender and in Potter, and compares the patch means.
fn check_scenario(kind: &str, samples: u32) -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "point_spot_light.py", SCRIPT)?;
    let blend = root.path().join("scene.blend");
    let exr = root.path().join("blender.exr");
    let json_path = root.path().join("blender.json");
    let samples_arg = samples.to_string();
    let arguments: [&Path; 5] = [
        &blend,
        &exr,
        &json_path,
        Path::new(kind),
        Path::new(&samples_arg),
    ];
    run_blender_arguments(&blender, &script, None, &arguments)?;

    let expected: Value = serde_json::from_slice(&fs::read(&json_path)?)?;
    let wanted = expected["patch_mean"]
        .as_array()
        .ok_or("the Blender patch mean is missing")?;
    assert!(
        wanted
            .iter()
            .all(|value| value.as_f64().is_some_and(|value| value > 0.1)),
        "{kind}: the Blender reference must be lit, got {wanted:?}"
    );

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
            {"op": "render.update", "target": {"id": active}, "set": {"resolution_x": 64, "resolution_y": 64, "samples": samples, "seed": 0, "max_bounces": 0}}
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

    let scene = read_scene(&project)?;
    let image: Image = serde_json::from_value(scene["images"]["render"].clone())?;
    let data = load_image_data(&image, &project, ImageInterpolation::Closest)?;
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
    for (channel, (actual, wanted)) in total.iter().map(|sum| sum / count).zip(wanted).enumerate() {
        let wanted = wanted
            .as_f64()
            .ok_or("the Blender patch mean is not a number")?;
        assert!(
            (actual - wanted).abs() <= PIXEL_TOLERANCE,
            "{kind} channel {channel}: Potter {actual} differs from Cycles {wanted}"
        );
    }
    Ok(())
}

#[test]
fn point_light_with_soft_falloff_matches_cycles() -> TestResult {
    check_scenario("point_soft", POINT_SAMPLES)
}

#[test]
fn point_light_as_a_sphere_matches_cycles() -> TestResult {
    check_scenario("point_sphere", POINT_SAMPLES)
}

#[test]
fn spot_light_matches_cycles() -> TestResult {
    check_scenario("spot", SPOT_SAMPLES)
}
