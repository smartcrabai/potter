//! Compares Potter's display encoding with Blender 5.2.2's own OCIO display processor for the same
//! views: `Standard`, `AgX`, `Filmic`, `Raw`, and `FalseColor` on the sRGB display with look `None`.
//!
//! The Blender side builds the chain Blender's display processor builds
//! (`libocio_display_processor.cc`): a scene-linear exposure matrix `2^exposure`, the
//! `DisplayViewTransform`, then an `ExponentTransform` `1/gamma` appended only when gamma is not 1.
//! It uses Blender's default CPU processor (`OPTIMIZATION_DEFAULT`), which is the one Blender renders
//! with. The view curve is not part of the OCIO processor (Blender applies it before exposure in
//! `colormanagement.cc`), so this test runs with an identity curve. It does not run a camera render.
//!
//! The samples include a grid that reaches all six tetrahedral branches of the `AgX`, `Filmic`, and
//! `FalseColor` LUTs, plus off-lattice values. Floats are compared before byte quantisation. The float
//! bound per view is set from the fused-versus-unfused spread of Potter's float32 emulation (see
//! `color/ocio.rs`). The one-byte check is the binding criterion.

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

/// Grid levels per channel. Every ordering of the three channels appears, with values between
/// lattice nodes, so each tetrahedral branch of the view LUTs is exercised.
const GRID_LEVELS: [f32; 5] = [0.0, 0.07, 0.3, 1.1, 4.5];

/// Exposure (stops) and gamma pairs. `(0, 1)` is the default; `(0.75, 2)` exercises both.
const SETTINGS: [(f64, f64); 2] = [(0.0, 1.0), (0.75, 2.0)];

/// Potter view, the Potter key, and the Blender config view name.
const VIEWS: [(ViewTransform, &str, &str); 5] = [
    (ViewTransform::Standard, "standard", "Standard"),
    (ViewTransform::AgX, "ag_x", "AgX"),
    (ViewTransform::Filmic, "filmic", "Filmic"),
    (ViewTransform::Raw, "raw", "Raw"),
    (ViewTransform::FalseColor, "false_color", "False Color"),
];

/// Float bound for display values in `[0, 1]`, per view. Each bound is about 2.5 times the largest
/// fused-versus-unfused difference measured over these samples and settings, or at least `1e-6`.
/// The 3e-3 `FalseColor` bound covers the palette step, where the lerp slope is about 2000 per unit.
/// Linux x86_64 may round the scalar paths differently from macOS arm64; CI decides whether the bound
/// holds there, and the one-byte check is kept as well.
fn float_tolerance(view: ViewTransform) -> f64 {
    match view {
        ViewTransform::AgX => 2.5e-6,
        ViewTransform::Standard | ViewTransform::Filmic | ViewTransform::Raw => 1.0e-6,
        ViewTransform::FalseColor => 3.0e-3,
    }
}

const BLENDER_SCRIPT: &str = r#"
import bpy
import json
import os
import sys

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"display parity requires Blender 5.2.2, found {bpy.app.version_string}")

import numpy
import PyOpenColorIO as OCIO

root = sys.argv[sys.argv.index("--") + 1]
with open(os.path.join(root, "samples.json"), encoding="utf-8") as source:
    spec = json.load(source)
config_path = os.path.join(
    bpy.utils.system_resource("DATAFILES"), "colormanagement", "config.ocio"
)
config = OCIO.Config.CreateFromFile(config_path)
result = {}
for case in spec["cases"]:
    display = OCIO.DisplayViewTransform()
    display.setSrc("Linear Rec.709")
    display.setDisplay("sRGB")
    display.setView(case["view"])
    scale = 2.0 ** case["exposure"]
    exposure = OCIO.MatrixTransform()
    exposure.setMatrix([scale, 0, 0, 0, 0, scale, 0, 0, 0, 0, scale, 0, 0, 0, 0, 1])
    group = OCIO.GroupTransform()
    group.appendTransform(exposure)
    group.appendTransform(display)
    if case["gamma"] != 1.0:
        gamma = OCIO.ExponentTransform()
        gamma.setValue([1.0 / case["gamma"]] * 3 + [1.0])
        group.appendTransform(gamma)
    processor = config.getProcessor(group)
    cpu = processor.getDefaultCPUProcessor()
    encoded = []
    for sample in spec["samples"]:
        pixel = numpy.array(sample, dtype=numpy.float32)
        cpu.applyRGB(pixel)
        encoded.append([float(value) for value in pixel])
    result[case["name"]] = encoded
with open(os.path.join(root, "blender_display.json"), "w", encoding="utf-8") as destination:
    json.dump(result, destination)
"#;

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
fn display_views_match_blender_ocio_display_processor() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping color view transform Blender parity: Blender is unavailable");
        return Ok(());
    };
    let samples: Vec<[f32; 3]> = SAMPLES.iter().copied().chain(grid()).collect();

    let mut cases = Vec::new();
    for (_, key, view) in VIEWS {
        for (index, (exposure, gamma)) in SETTINGS.iter().copied().enumerate() {
            cases.push(json!({
                "name": format!("{key}_{index}"),
                "view": view,
                "exposure": exposure,
                "gamma": gamma,
            }));
        }
    }

    let directory = tempdir()?;
    let root = fs::canonicalize(directory.path())?;
    fs::write(
        root.join("samples.json"),
        serde_json::to_vec(&json!({"samples": samples, "cases": cases}))?,
    )?;
    let script_path = root.join("color_view_transform.py");
    fs::write(&script_path, BLENDER_SCRIPT)?;
    let output = Command::new(&blender)
        .args(["--background", "--factory-startup", "--python"])
        .arg(&script_path)
        .arg("--")
        .arg(&root)
        .output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success() && !stdout.contains("Traceback") && !stderr.contains("Traceback"),
        "Blender display view fixture failed: stdout={stdout} stderr={stderr}"
    );
    let blender_views: BTreeMap<String, Vec<[f64; 3]>> =
        serde_json::from_slice(&fs::read(root.join("blender_display.json"))?)?;

    let mut worst = (0.0_f64, String::new());
    for (view, key, _) in VIEWS {
        let tolerance = float_tolerance(view);
        for (index, (exposure, gamma)) in SETTINGS.iter().copied().enumerate() {
            let name = format!("{key}_{index}");
            let blender_rows = case_rows(&blender_views, &name, samples.len())?;
            let management = ColorManagement {
                view_transform: view,
                exposure,
                gamma,
                ..ColorManagement::default()
            };
            for (sample_index, sample) in samples.iter().enumerate() {
                let potter_rgb = management.display_encode(*sample)?;
                for channel in 0..3 {
                    let blender_value = blender_rows[sample_index][channel].clamp(0.0, 1.0);
                    let potter_value = f64::from(potter_rgb[channel]);
                    let difference = (potter_value - blender_value).abs();
                    let message = format!(
                        "{name}, sample {sample:?}, channel {channel}: Potter {potter_value} vs Blender {blender_value}"
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
                    if difference > worst.0 {
                        worst = (difference, message);
                    }
                }
            }
        }
    }
    eprintln!("largest float difference {:.3e} at {}", worst.0, worst.1);
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
