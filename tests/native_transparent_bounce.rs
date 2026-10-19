//! The transparent bounce budget against Blender's Cycles render.
//!
//! Blender stacks six alpha-0.5 planes in front of a coloured world and renders with
//! `max_bounces` 4. Cycles counts transparent passes in `transparent_bounce` against
//! `transparent_max_bounce`, not against the regular `bounce`
//! (`intern/cycles/kernel/integrator/path_state.h`, `path_state_next`), so the world shows through
//! the stack with probability `0.5^6 = 1/64` even though six passes exceed the regular budget.
//! Potter imports that `.blend`, renders it, and both images are reduced to the mean RGB of the
//! same central patch. A pass-through that consumed the regular budget renders the patch black,
//! a 0.0156 error — far outside the tolerance. The test also round-trips a non-default
//! `transparent_max_bounces` through the `.blend` exporter and verifies it in Blender.

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
    error::Error,
    fs,
    path::Path,
    process::Command,
};

use blender_checked::blender_executable;
use native_oracle::{
    TestResult, apply_operations, export, import_blend, pot, read_scene, run_blender_arguments,
    write_script,
};
use potter_core::image::{ImageInterpolation, load_image_data, sample};
use potter_core::model::Image;
use serde_json::{Value, json};
use tempfile::tempdir;

const SCRIPT: &str = include_str!("fixtures/native_oracle/transparent_bounce.py");
/// Patch grid of the Blender oracle (`transparent_bounce.py`): fractions of the frame.
const PATCH: [f64; 9] = [
    0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75,
];
/// Render-to-stored-value tolerance (linear units), the same bound as `native_point_spot_light`.
/// The patch mean is about 0.0156 (1/64 of the world) and the two independent estimators (81 patch
/// pixels x 64 samples each) differ with a standard deviation of about 0.0024, so this is a 4-sigma
/// band; a budget-consuming pass-through renders the patch black.
const PIXEL_TOLERANCE: f64 = 0.01;

/// Runs `expression` in Blender on `blend`; a Python exception (a failed assertion) fails the check.
fn blender_python(blender: &Path, blend: &Path, expression: &str) -> Result<(), Box<dyn Error>> {
    let output = Command::new(blender)
        .args(["--background", "--disable-autoexec"])
        .arg(blend)
        .args(["--python-exit-code", "1", "--python-expr"])
        .arg(expression)
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "Blender check failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(())
}

#[test]
fn stacked_transparent_planes_keep_the_world_behind_the_regular_bounce_budget() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping the transparent bounce oracle; no executable Blender was found");
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "transparent_bounce.py", SCRIPT)?;
    let blend = root.path().join("scene.blend");
    let exr = root.path().join("blender.exr");
    let json_path = root.path().join("blender.json");
    run_blender_arguments(&blender, &script, None, &[&blend, &exr, &json_path])?;

    let expected: Value = serde_json::from_slice(&fs::read(&json_path)?)?;
    let wanted = expected["patch_mean"]
        .as_array()
        .ok_or("the Blender patch mean is missing")?;
    assert!(
        wanted[0].as_f64().is_some_and(|value| value > 0.005),
        "the Blender reference must show the world through the stack, got {wanted:?}"
    );

    let (project, _) = import_blend(&blender, root.path(), "project", &blend)?;
    // The scene stores Blender's default transparent_max_bounces of 8, which the scene JSON omits.
    let scene = read_scene(&project)?;
    assert!(
        scene["render"].get("transparent_max_bounces").is_none(),
        "the default transparent_max_bounces must be omitted: {}",
        scene["render"]
    );

    let active = scene["active_scene"]
        .as_str()
        .ok_or("active scene is missing")?
        .to_owned();
    let applied = apply_operations(
        &project,
        root.path(),
        json!([
            {"op": "render.update", "target": {"id": active}, "set": {"resolution_x": 64, "resolution_y": 64, "samples": 64, "seed": 0, "max_bounces": 4}}
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
            "channel {channel}: Potter {actual} differs from Cycles {wanted}"
        );
    }

    // A non-default transparent_max_bounces is stored, and round-trips through the .blend exporter.
    let applied = apply_operations(
        &project,
        root.path(),
        json!([
            {"op": "render.update", "target": {"id": active}, "set": {"transparent_max_bounces": 3}}
        ]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");
    let scene = read_scene(&project)?;
    assert_eq!(
        scene["render"]["transparent_max_bounces"].as_u64(),
        Some(3),
        "a non-default transparent_max_bounces must be stored"
    );
    let roundtrip = root.path().join("roundtrip.blend");
    export(&blender, &project, "blend", &roundtrip)?;
    blender_python(
        &blender,
        &roundtrip,
        "import bpy; assert bpy.context.scene.cycles.transparent_max_bounces == 3",
    )
}
