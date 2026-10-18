//! Compares Potter's display encoding with Blender 5.2.2's own OCIO display processor for every look
//! Blender offers on the `Standard`, `Filmic`, `Raw`, `AgX` and `False Color` views, on display `sRGB`,
//! at two exposure and gamma settings.
//!
//! The Blender side builds the chain Blender's display processor builds (`libocio_display_processor.cc`):
//! the scene-linear exposure matrix `2^exposure`; then, for a look, a `LookTransform` from `Linear Rec.709`
//! into the look's process space (a data view keeps the source space, as Blender does); then the
//! `DisplayViewTransform` with `looksBypass` set when a look applies; then an `ExponentTransform` `1/gamma`
//! when gamma is not 1. It uses Blender's default CPU processor (`OPTIMIZATION_DEFAULT`), the one Blender
//! renders with. It does not render a camera image.
//!
//! The samples are a grid over five levels per channel plus named samples, so each tetrahedral branch of
//! the look and view LUTs is exercised. Floats are compared before byte quantisation. The float bound per
//! view comes from the fused-versus-unfused spread of the float32 emulation (`/tmp/looks-probe`, see
//! `color/ocio.rs`), with the Blender residual of that emulation as the floor. The one-byte check is the
//! binding criterion.

use std::{collections::BTreeMap, error::Error, fs, io, process::Command};

use potter_core::color::{ColorManagement, ViewTransform};
use serde_json::json;
use tempfile::tempdir;

#[path = "common/blender_file.rs"]
mod blender_file;
use blender_file::blender_executable;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

/// Named scene-linear samples: greys across the exposure range, a saturated colour, and an
/// out-of-gamut colour with a large blue channel.
const SAMPLES: [[f32; 3]; 7] = [
    [0.0, 0.0, 0.0],
    [0.05, 0.05, 0.05],
    [0.18, 0.18, 0.18],
    [1.0, 1.0, 1.0],
    [8.0, 8.0, 8.0],
    [0.8, 0.1, 0.05],
    [0.02, 0.5, 4.0],
];

/// Grid levels per channel: every ordering of the three channels appears, with values between lattice
/// nodes, so each tetrahedral branch of the look and view LUTs is exercised.
const GRID_LEVELS: [f32; 5] = [0.0, 0.07, 0.3, 1.1, 4.5];

/// Exposure (stops) and gamma pairs. `(0, 1)` is the default; `(0.75, 2)` exercises both.
const SETTINGS: [(f64, f64); 2] = [(0.0, 1.0), (0.75, 2.0)];

/// The `Filmic Log` looks, as Blender names them and as Potter's tokens spell them. Blender lists them
/// for `Standard`, `Filmic` and `Raw`.
const FILMIC_LOOKS: [(&str, &str); 7] = [
    ("Very High Contrast", "very_high_contrast"),
    ("High Contrast", "high_contrast"),
    ("Medium High Contrast", "medium_high_contrast"),
    ("Medium Contrast", "medium_contrast"),
    ("Medium Low Contrast", "medium_low_contrast"),
    ("Low Contrast", "low_contrast"),
    ("Very Low Contrast", "very_low_contrast"),
];

/// The `AgX Log` look suffixes. `AgX` takes them as `AgX - <suffix>`, and `False Color` as
/// `False Color - <suffix>`.
const AGX_SUFFIXES: [(&str, &str); 9] = [
    ("Punchy", "punchy"),
    ("Greyscale", "greyscale"),
    ("Very High Contrast", "very_high_contrast"),
    ("High Contrast", "high_contrast"),
    ("Medium High Contrast", "medium_high_contrast"),
    ("Base Contrast", "base_contrast"),
    ("Medium Low Contrast", "medium_low_contrast"),
    ("Low Contrast", "low_contrast"),
    ("Very Low Contrast", "very_low_contrast"),
];

/// Float bound for display values in `[0, 1]`: twice the largest fused-versus-unfused spread of the
/// emulation for each view and look family (the same rule as the base views in
/// `color_view_transform_blender.rs`), rounded up. Measured spreads: Standard and Raw 0, Filmic
/// 4.8e-7 (looks) and 1.8e-7 (none), AgX with no look 7.7e-7, AgX looks 4.8e-6, False Color 1.7e-3.
/// The AgX-look residual is 9.3e-6 (`Punchy`, gamma 2), inside 9.6e-6 with little margin. Its source is
/// the GradingTone shadows (`ComputeHSFwd`, `GradingToneOpCPU.cpp:499`): the look-stage deviation is at
/// most 2.4e-7 (ULP level, confined to the 0.1-0.4 shadow zone), and the 25-stop log-to-linear step of
/// the display view multiplies it by about 17 before the display encode. The fused and unfused variants
/// do not reproduce Blender bit-for-bit; the compiler's contraction choice is unverified (CMake flags
/// UNREAD). False Color's palette step has a slope of about 2000 per unit, so its residual (7.7e-4) sits
/// in the 3.5e-3 bound.
fn float_tolerance(view: ViewTransform, has_look: bool) -> f64 {
    match view {
        ViewTransform::Standard | ViewTransform::Filmic | ViewTransform::Raw => 1.0e-6,
        ViewTransform::AgX if has_look => 9.6e-6,
        ViewTransform::AgX => 1.6e-6,
        ViewTransform::FalseColor => 3.5e-3,
    }
}

const BLENDER_SCRIPT: &str = r#"
import bpy
import json
import os
import sys

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"look parity requires Blender 5.2.2, found {bpy.app.version_string}")

import numpy
import PyOpenColorIO as OCIO

root = sys.argv[sys.argv.index("--") + 1]
with open(os.path.join(root, "samples.json"), encoding="utf-8") as source:
    spec = json.load(source)
config_path = os.path.join(
    bpy.utils.system_resource("DATAFILES"), "colormanagement", "config.ocio"
)
config = OCIO.Config.CreateFromFile(config_path)
display_name = "sRGB"
scene_linear = "Linear Rec.709"
result = {}
for case in spec["cases"]:
    group = OCIO.GroupTransform()
    scale = 2.0 ** case["exposure"]
    exposure = OCIO.MatrixTransform()
    exposure.setMatrix([scale, 0, 0, 0, 0, scale, 0, 0, 0, 0, scale, 0, 0, 0, 0, 1])
    group.appendTransform(exposure)
    source = scene_linear
    looks_bypass = False
    look = case["look"]
    if look is not None:
        view_space = config.getDisplayViewColorSpaceName(display_name, case["view"])
        is_data = config.getColorSpace(view_space).isData()
        process = config.getLook(look).getProcessSpace()
        look_transform = OCIO.LookTransform()
        look_transform.setSrc(scene_linear)
        look_transform.setDst(scene_linear if is_data else process)
        look_transform.setLooks(look)
        group.appendTransform(look_transform)
        if not is_data:
            source = process
        looks_bypass = True
    display = OCIO.DisplayViewTransform()
    display.setSrc(source)
    display.setLooksBypass(looks_bypass)
    display.setDisplay(display_name)
    display.setView(case["view"])
    group.appendTransform(display)
    if case["gamma"] != 1.0:
        gamma = OCIO.ExponentTransform()
        gamma.setValue([1.0 / case["gamma"]] * 3 + [1.0])
        group.appendTransform(gamma)
    cpu = config.getProcessor(group).getDefaultCPUProcessor()
    encoded = []
    for sample in spec["samples"]:
        pixel = numpy.array(sample, dtype=numpy.float32)
        cpu.applyRGB(pixel)
        encoded.append([float(value) for value in pixel])
    result[case["name"]] = encoded
with open(os.path.join(root, "blender_looks.json"), "w", encoding="utf-8") as destination:
    json.dump(result, destination)
"#;

/// One Blender display case: the Potter view and look, the Blender names, and the settings.
struct Case {
    name: String,
    view: ViewTransform,
    look_token: String,
    blender_view: &'static str,
    blender_look: Option<String>,
    exposure: f64,
    gamma: f64,
}

fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    let mut add_view = |view: ViewTransform,
                        key: &'static str,
                        blender_view: &'static str,
                        looks: Vec<(String, String)>| {
        for (index, (exposure, gamma)) in SETTINGS.iter().copied().enumerate() {
            cases.push(Case {
                name: format!("{key}_none_{index}"),
                view,
                look_token: "none".to_owned(),
                blender_view,
                blender_look: None,
                exposure,
                gamma,
            });
            for (blender_name, token) in &looks {
                cases.push(Case {
                    name: format!("{key}_{token}_{index}"),
                    view,
                    look_token: token.clone(),
                    blender_view,
                    blender_look: Some(blender_name.clone()),
                    exposure,
                    gamma,
                });
            }
        }
    };
    let filmic: Vec<(String, String)> = FILMIC_LOOKS
        .iter()
        .map(|(name, token)| ((*name).to_owned(), (*token).to_owned()))
        .collect();
    add_view(
        ViewTransform::Standard,
        "standard",
        "Standard",
        filmic.clone(),
    );
    add_view(ViewTransform::Filmic, "filmic", "Filmic", filmic.clone());
    add_view(ViewTransform::Raw, "raw", "Raw", filmic);
    let agx: Vec<(String, String)> = AGX_SUFFIXES
        .iter()
        .map(|(name, token)| (format!("AgX - {name}"), format!("agx_{token}")))
        .collect();
    add_view(ViewTransform::AgX, "agx", "AgX", agx);
    let false_color: Vec<(String, String)> = AGX_SUFFIXES
        .iter()
        .map(|(name, token)| {
            (
                format!("False Color - {name}"),
                format!("false_color_{token}"),
            )
        })
        .collect();
    add_view(
        ViewTransform::FalseColor,
        "false_color",
        "False Color",
        false_color,
    );
    cases
}

fn grid() -> Vec<[f32; 3]> {
    let mut samples = Vec::new();
    for &red in &GRID_LEVELS {
        for &green in &GRID_LEVELS {
            for &blue in &GRID_LEVELS {
                samples.push([red, green, blue]);
            }
        }
    }
    samples
}

fn byte(value: f64) -> i64 {
    (value * 255.0).round() as i64
}

#[test]
fn display_looks_match_blender_ocio_display_processor() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping color look Blender parity: Blender is unavailable");
        return Ok(());
    };
    let samples: Vec<[f32; 3]> = SAMPLES.iter().copied().chain(grid()).collect();
    let cases = cases();

    let directory = tempdir()?;
    let root = fs::canonicalize(directory.path())?;
    let case_specs: Vec<_> = cases
        .iter()
        .map(|case| {
            json!({
                "name": case.name,
                "view": case.blender_view,
                "look": case.blender_look,
                "exposure": case.exposure,
                "gamma": case.gamma,
            })
        })
        .collect();
    fs::write(
        root.join("samples.json"),
        serde_json::to_vec(&json!({"samples": samples, "cases": case_specs}))?,
    )?;
    let script_path = root.join("color_looks.py");
    fs::write(&script_path, BLENDER_SCRIPT)?;
    let output = Command::new(&blender)
        .args([
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
        ])
        .arg(&script_path)
        .arg("--")
        .arg(&root)
        .output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success() && !stdout.contains("Traceback") && !stderr.contains("Traceback"),
        "Blender look fixture failed: stdout={stdout} stderr={stderr}"
    );
    let blender_looks: BTreeMap<String, Vec<[f64; 3]>> =
        serde_json::from_slice(&fs::read(root.join("blender_looks.json"))?)?;

    let mut worst = (0.0_f64, String::new());
    let mut checked = 0_usize;
    for case in &cases {
        let has_look = case.look_token != "none";
        let tolerance = float_tolerance(case.view, has_look);
        let blender_rows = case_rows(&blender_looks, &case.name, samples.len())?;
        let management = ColorManagement {
            view_transform: case.view,
            look: case.look_token.clone(),
            exposure: case.exposure,
            gamma: case.gamma,
            ..ColorManagement::default()
        };
        for (sample_index, sample) in samples.iter().enumerate() {
            let potter_rgb = management.display_encode(*sample)?;
            for channel in 0..3 {
                let blender_value = blender_rows[sample_index][channel].clamp(0.0, 1.0);
                let potter_value = f64::from(potter_rgb[channel]);
                let difference = (potter_value - blender_value).abs();
                let message = format!(
                    "{}, sample {sample:?}, channel {channel}: Potter {potter_value} vs Blender {blender_value}",
                    case.name
                );
                assert!(
                    difference <= tolerance,
                    "{message}: difference {difference:e} exceeds {tolerance:e}"
                );
                assert!(
                    (byte(potter_value) - byte(blender_value)).abs() <= 1,
                    "{message}: byte {} vs {}",
                    byte(potter_value),
                    byte(blender_value),
                );
                checked += 1;
                if difference > worst.0 {
                    worst = (difference, message);
                }
            }
        }
    }
    eprintln!(
        "checked {checked} channel values over {} cases; largest float difference {:.3e} at {}",
        cases.len(),
        worst.0,
        worst.1
    );
    Ok(())
}

/// Blender's output rows for one case, checked against the sample count.
fn case_rows<'a>(
    views: &'a BTreeMap<String, Vec<[f64; 3]>>,
    name: &str,
    expected: usize,
) -> TestResult<&'a Vec<[f64; 3]>> {
    let rows = views
        .get(name)
        .ok_or_else(|| io::Error::other(format!("Blender output has no {name} case")))?;
    if rows.len() != expected {
        return Err(io::Error::other(format!(
            "Blender {name} returned {} samples, expected {expected}",
            rows.len()
        ))
        .into());
    }
    Ok(rows)
}
