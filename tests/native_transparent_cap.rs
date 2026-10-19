//! The transparent bounce cap and the shadow-ray budget against Blender's Cycles render.
//!
//! Blender puts N alpha-0.5 planes between a point light and a lit diffuse floor, with
//! `transparent_max_bounces` C, and saves the scene as a `.blend`. Potter imports it, renders it,
//! and both images are reduced to the mean RGB of the same central patch. The four cases pin the
//! two Cycles rules that `transparent_max_bounces` governs (`transparent_bounce` counts the passes
//! a path has made):
//!   - the pass that reaches C ends the path at the next surface, so the floor behind two planes
//!     with C = 2 contributes nothing (`path_state.h`, `PATH_RAY_TERMINATE_ON_NEXT_SURFACE`);
//!   - a shadow ray may cross at most C minus the path's passes transparent surfaces and is
//!     blocked past that (`kernel/bvh/intersect_filter.h`, `integrate_shadow_max_transparent_hits`).
//!
//! `n2c4` and `n1c2` are lit, so a budget that is ignored or a shadow that is not transmitted
//! through the remaining planes shows as a wrong mean. `n2c3` and `n2c2` are zero in Cycles, so
//! a shadow that ignores its budget or a floor that is shaded past the cap shows as a nonzero mean.

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

const SCRIPT: &str = include_str!("fixtures/native_oracle/transparent_cap.py");
/// Patch grid of the Blender oracle (`transparent_cap.py`): fractions of the frame.
const PATCH: [f64; 9] = [
    0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75,
];
/// Render-to-stored-value tolerance (linear units). The lit cases are about 0.06 and 0.23 and the
/// patch has 81 pixels of 256 samples; each mean's sampling error is about 0.002, so this bound is
/// several sigma for the difference of two estimators, and a dropped shadow budget (the floor
/// lit through all planes) is far outside it.
const PIXEL_TOLERANCE: f64 = 0.01;

fn check_scenario(kind: &str) -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "transparent_cap.py", SCRIPT)?;
    let blend = root.path().join("scene.blend");
    let exr = root.path().join("blender.exr");
    let json_path = root.path().join("blender.json");
    let arguments: [&Path; 4] = [&blend, &exr, &json_path, Path::new(kind)];
    run_blender_arguments(&blender, &script, None, &arguments)?;

    let expected: Value = serde_json::from_slice(&fs::read(&json_path)?)?;
    let wanted = expected["patch_mean"]
        .as_array()
        .ok_or("the Blender patch mean is missing")?;

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
            {"op": "render.update", "target": {"id": active}, "set": {"resolution_x": 32, "resolution_y": 32, "samples": 256, "seed": 0, "max_bounces": 4}}
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
fn floor_behind_two_planes_with_budget_four_is_shadowed_through_both() -> TestResult {
    check_scenario("n2c4")
}

#[test]
fn shadow_ray_past_its_remaining_budget_is_blocked() -> TestResult {
    check_scenario("n2c3")
}

#[test]
fn floor_at_the_transparent_cap_is_terminated_without_direct_light() -> TestResult {
    check_scenario("n2c2")
}

#[test]
fn one_plane_with_budget_two_leaves_one_shadow_crossing() -> TestResult {
    check_scenario("n1c2")
}
