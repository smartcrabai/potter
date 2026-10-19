//! The transparent bounce cap, the shadow-ray budget, and volume film alpha against Blender's Cycles
//! render.
//!
//! Plane cases: Blender puts N alpha-0.5 planes between a point light and a lit diffuse floor, with
//! `transparent_max_bounces` C, and saves the scene as a `.blend`. Potter imports it, renders it,
//! and both images are reduced to the mean RGB of the same central patch. The four cases pin the
//! two Cycles rules that `transparent_max_bounces` governs (`transparent_bounce` counts the passes
//! a path has made):
//!   - the pass that reaches C ends the path at the next surface, so the floor behind two planes
//!     with C = 2 contributes nothing (`path_state.h`, `PATH_RAY_TERMINATE_ON_NEXT_SURFACE`);
//!   - a shadow ray may cross at most C minus the path's passes transparent surfaces and is
//!     blocked past that (`kernel/bvh/intersect_filter.h`, `integrate_shadow_max_transparent_hits`).
//!
//! Volume case: a homogeneous absorbing slab over a transparent card with `film_transparent`. Every
//! camera ray crosses the slab, so the film alpha is `1 - exp(-sigma * thickness)`. The film alpha
//! passes through the camera-segment opacity that the path accumulates, so a regression that resets
//! it on a later pass-through shows here.
//!
//! Every case compares the RGB and the alpha patch means of Potter with Blender's.

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

/// Renders `kind` in Blender and in Potter, compares their patch means, and returns Blender's JSON.
fn check_scenario(kind: &str) -> TestResult<Value> {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(json!(null));
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
    let wanted_alpha = expected["patch_alpha_mean"]
        .as_f64()
        .ok_or("the Blender patch alpha mean is missing")?;

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
    let mut total = [0.0_f64; 4];
    for v in PATCH {
        for u in PATCH {
            let pixel = sample(&data, [u, v], false);
            for (sum, value) in total.iter_mut().zip(pixel.iter()) {
                *sum += value;
            }
        }
    }
    let count = (PATCH.len() * PATCH.len()) as f64;
    let actual = total.map(|sum| sum / count);
    for (channel, (actual, wanted)) in actual.iter().take(3).zip(wanted).enumerate() {
        let wanted = wanted
            .as_f64()
            .ok_or("the Blender patch mean is not a number")?;
        assert!(
            (actual - wanted).abs() <= PIXEL_TOLERANCE,
            "{kind} channel {channel}: Potter {actual} differs from Cycles {wanted}"
        );
    }
    assert!(
        (actual[3] - wanted_alpha).abs() <= PIXEL_TOLERANCE,
        "{kind} alpha: Potter {} differs from Cycles {wanted_alpha}",
        actual[3]
    );
    Ok(expected)
}

#[test]
fn floor_behind_two_planes_with_budget_four_is_shadowed_through_both() -> TestResult {
    check_scenario("n2c4").map(|_| ())
}

#[test]
fn shadow_ray_past_its_remaining_budget_is_blocked() -> TestResult {
    check_scenario("n2c3").map(|_| ())
}

#[test]
fn floor_at_the_transparent_cap_is_terminated_without_direct_light() -> TestResult {
    check_scenario("n2c2").map(|_| ())
}

#[test]
fn one_plane_with_budget_two_leaves_one_shadow_crossing() -> TestResult {
    check_scenario("n1c2").map(|_| ())
}

#[test]
fn absorbing_slab_film_alpha_is_one_minus_transmittance() -> TestResult {
    let Some(_) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let expected = check_scenario("volume_alpha")?;
    let analytic = expected["beer_lambert_alpha"]
        .as_f64()
        .ok_or("the analytic alpha is missing")?;
    // Blender's own alpha must equal 1 - exp(-sigma * thickness), or the oracle is not the model.
    let blender_alpha = expected["patch_alpha_mean"]
        .as_f64()
        .ok_or("the Blender patch alpha mean is missing")?;
    assert!(
        (blender_alpha - analytic).abs() <= PIXEL_TOLERANCE,
        "Blender alpha {blender_alpha} differs from 1 - exp(-sigma d) = {analytic}"
    );
    Ok(())
}
