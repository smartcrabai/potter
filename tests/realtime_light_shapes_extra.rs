//! Sun discs, spot emitters, and a light seen through absorbing volume, in the realtime Beauty image, checked against
//! Blender 5.2.2's EEVEE.
//!
//! `fixtures/native_oracle/eevee_light_shapes_extra.py` renders three scenes on a black world:
//!
//! - a sun of angle 8° seen by a perspective camera that looks straight at it, whose disc EEVEE draws at
//!   `energy · (1 + 1 / r²) / π` with `r = tan(angle / 2)`;
//! - a spot light facing an orthographic camera, next to the area and point lights, whose disc EEVEE draws at the
//!   sphere emitter radiance `P / (4 π² r²)` on axis;
//! - the area light behind a slab of absorbing volume, where EEVEE attenuates the light shape by the transmittance in
//!   front of it.
//!
//! The test imports each saved `.blend`, renders it with `pot render --engine realtime`, and compares window means with
//! EEVEE's. The sun and spot tolerances are 0.01: EEVEE's own values lie within 0.5% of the formulas. The absorber is
//! looser, see its constant.
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

use std::{error::Error, fs, path::Path};

use native_oracle::{
    TestResult, apply_operations, import_blend, pot, read_scene, run_blender_arguments,
    write_script,
};
use potter_core::image::{ImageInterpolation, load_image_data, sample};
use potter_core::model::Image;
use serde_json::{Value, json};
use tempfile::tempdir;

/// The scenes and EEVEE renders (`eevee_light_shapes_extra.py`).
const SCRIPT: &str = include_str!("fixtures/native_oracle/eevee_light_shapes_extra.py");

/// Render settings shared with `eevee_light_shapes_extra.py`.
const RESOLUTION: u32 = 64;

/// Linear tolerance on a window mean against EEVEE, for the sun and the spot. EEVEE's disc values lie within 0.5% of
/// their formulas, so the bound is about twice EEVEE's own spread from the formula at these magnitudes.
const EMITTER_TOLERANCE: f64 = 0.01;

/// Linear tolerance on the area light seen through the absorber. PROVISIONAL: EEVEE's transmittance through the
/// 2 m slab is 0.411, where exact Beer-Lambert for density 0.5 gives 0.368, so EEVEE's froxel volume is about 12% more
/// transmissive than the exact integral. Potter's realtime volume integrates exactly, so this bound must cover the
/// froxel gap, and the lane must confirm that the gap does not hide an error: the absorber must still darken the light.
const ABSORBER_TOLERANCE: f64 = 0.03;

/// The mean red value of the `(2 * half + 1)` square window centred on `(column, row)` (top-down) in the image `id`.
fn window_mean(
    project: &Path,
    id: &str,
    column: usize,
    row: usize,
    half: usize,
) -> Result<f64, Box<dyn Error>> {
    let scene = read_scene(project)?;
    let image: Image = serde_json::from_value(scene["images"][id].clone())?;
    let data = load_image_data(&image, project, ImageInterpolation::Closest)?;
    let texels = f64::from(RESOLUTION);
    let mut total = 0.0;
    let mut count = 0.0;
    for r in row - half..=row + half {
        for c in column - half..=column + half {
            let uv = [(c as f64 + 0.5) / texels, (r as f64 + 0.5) / texels];
            total += sample(&data, uv, false)[0];
            count += 1.0;
        }
    }
    Ok(total / count)
}

/// Imports `blend` into `root/name`, renders it in the realtime engine with `samples` samples, and loads the frame as
/// the image `id`. Returns the project's directory.
fn render_realtime(
    blender: &Path,
    root: &Path,
    name: &str,
    blend: &Path,
    id: &str,
) -> Result<std::path::PathBuf, Box<dyn Error>> {
    let (project, _) = import_blend(blender, root, name, blend)?;
    let scene = read_scene(&project)?;
    let active = scene["active_scene"]
        .as_str()
        .ok_or("active scene is missing")?
        .to_owned();
    let applied = apply_operations(
        &project,
        root,
        json!([
            {"op": "render.update", "target": {"id": active}, "set": {"resolution_x": RESOLUTION, "resolution_y": RESOLUTION, "samples": 4, "seed": 0}}
        ]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");
    let frames = root.join(format!("frames_{name}"));
    let rendered = pot()
        .arg("render")
        .arg(&project)
        .args(["--engine", "realtime", "--format", "exr", "--out"])
        .arg(&frames)
        .arg("--json")
        .output()?;
    assert!(
        rendered.status.success(),
        "realtime render failed: {}",
        String::from_utf8_lossy(&rendered.stdout)
    );
    let loaded = apply_operations(
        &project,
        root,
        json!([{"op": "image.load", "id": id, "path": frames.join("frame_0001.exr").to_string_lossy()}]),
    )?;
    assert_eq!(loaded["ok"], true, "{loaded}");
    Ok(project)
}

#[test]
fn realtime_sun_disc_and_spot_emitter_match_eevee() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "eevee_light_shapes_extra.py", SCRIPT)?;
    let sun_blend = root.path().join("sun.blend");
    let spot_blend = root.path().join("spot.blend");
    let absorber_blend = root.path().join("absorber.blend");
    let oracle_json = root.path().join("eevee_light_shapes_extra.json");
    let sun_exr = root.path().join("eevee_sun.exr");
    let spot_exr = root.path().join("eevee_spot.exr");
    let absorber_exr = root.path().join("eevee_absorber.exr");
    run_blender_arguments(
        &blender,
        &script,
        None,
        &[
            sun_blend.as_path(),
            spot_blend.as_path(),
            absorber_blend.as_path(),
            oracle_json.as_path(),
            sun_exr.as_path(),
            spot_exr.as_path(),
            absorber_exr.as_path(),
        ],
    )?;
    let oracle: Value = serde_json::from_slice(&fs::read(&oracle_json)?)?;
    let number = |key: &str| -> Result<f64, Box<dyn Error>> {
        Ok(oracle[key]
            .as_f64()
            .ok_or_else(|| format!("{key} is missing"))?)
    };
    let index = |key: &str| -> Result<usize, Box<dyn Error>> {
        Ok(oracle[key]
            .as_u64()
            .ok_or_else(|| format!("{key} is missing"))? as usize)
    };
    let half = oracle["window"].as_u64().ok_or("window is missing")? as usize;
    let centre = RESOLUTION as usize / 2;

    // The sun: the centre of the disc and a corner outside it.
    let project = render_realtime(&blender, root.path(), "sun", &sun_blend, "sun")?;
    let sun_centre = window_mean(&project, "sun", centre, centre, half)?;
    let sun_corner = window_mean(
        &project,
        "sun",
        RESOLUTION as usize / 8,
        RESOLUTION as usize / 8,
        half,
    )?;
    let expected = number("sun_centre_mean_red")?;
    assert!(
        (sun_centre - expected).abs() <= EMITTER_TOLERANCE,
        "sun disc: Potter {sun_centre} differs from EEVEE {expected}"
    );
    assert!(
        sun_corner.abs() <= EMITTER_TOLERANCE / 2.0,
        "the sun must show no disc outside its angle, got {sun_corner}"
    );

    // The spot emitter on axis, the area and point lights beside it.
    let project = render_realtime(&blender, root.path(), "spot", &spot_blend, "spot")?;
    let spot = window_mean(
        &project,
        "spot",
        index("spot_column")?,
        index("spot_row_top_down")?,
        half,
    )?;
    let expected = number("spot_mean_red")?;
    assert!(
        (spot - expected).abs() <= EMITTER_TOLERANCE,
        "spot emitter: Potter {spot} differs from EEVEE {expected}"
    );
    let area_clear = window_mean(&project, "spot", index("area_column")?, centre, half)?;
    let expected = number("area_clear_mean_red")?;
    assert!(
        (area_clear - expected).abs() <= EMITTER_TOLERANCE,
        "area light (no absorber): Potter {area_clear} differs from EEVEE {expected}"
    );

    // The area light seen through the absorber: attenuated, and within the froxel bound of EEVEE.
    let project = render_realtime(
        &blender,
        root.path(),
        "absorber",
        &absorber_blend,
        "absorber",
    )?;
    let through = window_mean(&project, "absorber", index("area_column")?, centre, half)?;
    let expected = number("area_through_absorber_mean_red")?;
    assert!(
        through < 0.9 * number("area_clear_mean_red")?,
        "the absorber must darken the area light, got {through}"
    );
    assert!(
        (through - expected).abs() <= ABSORBER_TOLERANCE,
        "area light through the absorber: Potter {through} differs from EEVEE {expected}"
    );
    Ok(())
}
