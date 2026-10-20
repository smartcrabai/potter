//! Realtime shadows from transparent surfaces against Blender's EEVEE 5.2.2 renders.
//!
//! A transparent occluder (a BLENDED Principled with Alpha 0.5) casts a shadow half as dense as an opaque one, as
//! EEVEE's transparent shadow pass gives (`eevee_surf_shadow.bsl.hh`). With `use_transparent_shadow` off the occluder
//! casts an opaque shadow. Each render is reduced to the mean RGB of a 9 by 9 grid over the frame. The test compares
//! the shadowed-to-lit ratio of each occluder, which does not depend on the light's absolute scale, against EEVEE's.

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

const SCRIPT: &str = include_str!("fixtures/native_oracle/realtime_transparent_shadow.py");
/// Patch grid of the Blender oracle (`realtime_transparent_shadow.py`): fractions of the frame.
const PATCH: [f64; 9] = [
    0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75,
];
/// The measured tolerances (`measure_realtime_transparent_shadow_spread.py` writes the file and derives each term): per
/// light and ratio, four times EEVEE's noise over independent realisations, plus EEVEE's bias against the exact ratio of
/// the occluder geometry, plus Potter's edge-placement bound (the largest change of the exact ratio when the shadow
/// edges move by the bound on Potter's own placement error: 0.003 rad PCF for a sun, half a 64-texel depth-map texel for
/// a point light).
const SPREAD: &str = include_str!("fixtures/native_oracle/realtime_transparent_shadow_spread.json");
fn tolerance(light: &str, ratio: &str) -> TestResult<f64> {
    let spread: Value = serde_json::from_str(SPREAD)?;
    spread["lights"][light][ratio]["tolerance"]
        .as_f64()
        .ok_or_else(|| format!("no tolerance for {light} {ratio}").into())
}

/// The occluder states of the oracle: a transparent shadow, an opaque one, and no occluder.
#[derive(Clone, Copy)]
enum Occluder {
    Alpha,
    AlphaOff,
    Absent,
}

impl Occluder {
    fn name(self) -> &'static str {
        match self {
            Self::Alpha => "ALPHA",
            Self::AlphaOff => "ALPHA_OFF",
            Self::Absent => "NONE",
        }
    }
}

/// The patch mean of Blender's EEVEE render of `light` with `occluder`, and the scene it wrote.
fn blender_render(
    blender: &Path,
    root: &Path,
    light: &str,
    occluder: Occluder,
) -> TestResult<(PathBuf, Vec<f64>)> {
    let dir = root.join(format!("{light}_{}", occluder.name()));
    fs::create_dir_all(&dir)?;
    let script = write_script(&dir, "realtime_transparent_shadow.py", SCRIPT)?;
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
            Path::new(light),
            Path::new(occluder.name()),
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

/// Imports the Blender scene and returns the project, its active scene, and the imported scene JSON.
fn import_scene(blender: &Path, root: &Path, blend: &Path) -> TestResult<(PathBuf, String, Value)> {
    let (project, _) = import_blend(blender, root, "project", blend)?;
    let scene = read_scene(&project)?;
    let active = scene["active_scene"]
        .as_str()
        .ok_or("active scene is missing")?
        .to_owned();
    Ok((project, active, scene))
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

/// The mean over RGB of the 9 by 9 grid of a rendered frame, as the Blender oracle takes it.
fn patch_mean(data: &ImageData) -> f64 {
    let mut total = 0.0;
    for v in PATCH {
        for u in PATCH {
            let pixel = sample(data, [u, v], false);
            total += (pixel[0] + pixel[1] + pixel[2]) / 3.0;
        }
    }
    total / (PATCH.len() * PATCH.len()) as f64
}

/// The patch mean of Blender's EEVEE render, averaged over its channels.
fn reference_mean(mean: &[f64]) -> f64 {
    mean.iter().sum::<f64>() / mean.len() as f64
}

/// Runs one light through the three occluder states in both renderers and compares their shadow ratios.
fn assert_shadow_ratios_match(light: &str) -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!(
            "skipping the realtime transparent shadow oracle; no executable Blender was found"
        );
        return Ok(());
    };
    let root = tempdir()?;

    let (blender_none, none_mean) = blender_render(&blender, root.path(), light, Occluder::Absent)?;
    let (blender_alpha, alpha_mean) =
        blender_render(&blender, root.path(), light, Occluder::Alpha)?;
    let (blender_off, off_mean) = blender_render(&blender, root.path(), light, Occluder::AlphaOff)?;
    let eevee_none = reference_mean(&none_mean);
    let eevee_alpha = reference_mean(&alpha_mean) / eevee_none;
    let eevee_off = reference_mean(&off_mean) / eevee_none;
    assert!(
        eevee_alpha > eevee_off,
        "EEVEE's transparent shadow must be lighter than its opaque one: {eevee_alpha} vs {eevee_off}"
    );

    let mut potter = Vec::new();
    for (name, blend) in [
        (Occluder::Absent, blender_none),
        (Occluder::Alpha, blender_alpha),
        (Occluder::AlphaOff, blender_off),
    ] {
        let (project, _, _) = import_scene(&blender, root.path(), &blend)?;
        let frame = potter_realtime_frame(&project, root.path(), &name.name().to_lowercase())?;
        potter.push(patch_mean(&frame));
    }
    let potter_none = potter[0];
    let potter_alpha = potter[1] / potter_none;
    let potter_off = potter[2] / potter_none;
    let alpha_tolerance = tolerance(light, "alpha_ratio")?;
    let off_tolerance = tolerance(light, "alpha_off_ratio")?;
    assert!(
        (potter_alpha - eevee_alpha).abs() <= alpha_tolerance,
        "{light}: Potter's transparent shadow ratio {potter_alpha:.4} differs from EEVEE's {eevee_alpha:.4} by more \
         than {alpha_tolerance:.4} (lit: Potter {potter_none:.4}, EEVEE {eevee_none:.4})"
    );
    assert!(
        (potter_off - eevee_off).abs() <= off_tolerance,
        "{light}: Potter's opaque shadow ratio {potter_off:.4} differs from EEVEE's {eevee_off:.4} by more than \
         {off_tolerance:.4}"
    );
    Ok(())
}

#[test]
fn sun_transparent_shadow_matches_eevee_in_realtime() -> TestResult {
    assert_shadow_ratios_match("SUN")
}

#[test]
fn point_transparent_shadow_matches_eevee_in_realtime() -> TestResult {
    assert_shadow_ratios_match("POINT")
}

#[test]
fn use_transparent_shadow_round_trips_through_blend() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!(
            "skipping the use_transparent_shadow round trip; no executable Blender was found"
        );
        return Ok(());
    };
    let root = tempdir()?;
    let (alpha_blend, _) = blender_render(&blender, root.path(), "SUN", Occluder::Alpha)?;
    let (off_blend, _) = blender_render(&blender, root.path(), "SUN", Occluder::AlphaOff)?;
    for (blend, expected) in [(alpha_blend, true), (off_blend, false)] {
        let (_, _, scene) = import_scene(&blender, root.path(), &blend)?;
        let occluder = scene["materials"]
            .as_object()
            .and_then(|materials| {
                materials
                    .values()
                    .find(|material| material["name"] == "Occluder")
            })
            .ok_or("the occluder material is missing")?;
        assert_eq!(
            occluder["use_transparent_shadow"], expected,
            "use_transparent_shadow must import from the .blend: {occluder}"
        );
    }
    Ok(())
}
