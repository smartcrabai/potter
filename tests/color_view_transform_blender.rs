//! Compares Potter's display encoding with Blender 5.2.2's own OCIO display processor for the same
//! views: `Standard`, `AgX`, `Filmic`, `Raw`, and `FalseColor` on the sRGB display with look `None`.
//!
//! The Blender side builds the chain Blender's display processor builds
//! (`libocio_display_processor.cc`): a scene-linear exposure matrix `2^exposure`, the
//! `DisplayViewTransform`, then an `ExponentTransform` `1/gamma` appended only when gamma is not 1.
//! The view curve is not part of the OCIO processor (Blender applies it before exposure in
//! `colormanagement.cc`), so this test runs with an identity curve. It does not run a camera render.
//!
//! The samples include a grid that reaches all six tetrahedral branches of the `AgX`, `Filmic`, and
//! `FalseColor` LUTs, plus off-lattice values. Floats are compared before byte quantisation. The float
//! tolerance covers only f32 OCIO arithmetic against Potter's f64; the byte check is kept as well.

use std::{collections::BTreeMap, error::Error, fs, io, process::Command};

use potter_core::color::{ColorManagement, ViewTransform};
use serde_json::json;
use tempfile::tempdir;

#[path = "common/blender_file.rs"]
mod blender_file;
use blender_file::blender_executable;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

/// Float comparison bound for display values in [0, 1]. The largest residual measured outside the
/// optimiser band is 2.13e-5 (`AgX`, gamma 2, sample `[0, 0.07, 0.07]`, red). Emulating the chain in
/// f32 does not explain it, so the bound is set above that residual and is not a claim that f32
/// arithmetic accounts for it. The one-byte check below is the binding criterion.
const FLOAT_TOLERANCE: f64 = 3.0e-5;

/// Blender's default processor and its lossless chain differ by more than this for a channel only
/// where OCIO's default optimiser resamples the False Color palette (measured 1.1e-2 at one sample).
const OPTIMIZER_SENSITIVITY: f64 = 1.0e-4;

/// Potter against Blender's lossless chain at those channels. Measured: Blender's lossless f32 output
/// differs from Potter by 6.2e-5 there, because the palette slope is about 2000 per unit of position.
const LOSSLESS_TOLERANCE: f64 = 1.0e-4;

/// Named scene-linear samples: greys across the exposure range, a saturated colour, and an
/// out-of-gamut colour with a large blue channel.
const SAMPLES: [[f64; 3]; 7] = [
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
const GRID_LEVELS: [f64; 5] = [0.0, 0.07, 0.3, 1.1, 4.5];

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
    lossless = processor.getOptimizedCPUProcessor(OCIO.OPTIMIZATION_LOSSLESS)
    encoded = []
    exact = []
    for sample in spec["samples"]:
        pixel = numpy.array(sample, dtype=numpy.float32)
        cpu.applyRGB(pixel)
        encoded.append([float(value) for value in pixel])
        pixel = numpy.array(sample, dtype=numpy.float32)
        lossless.applyRGB(pixel)
        exact.append([float(value) for value in pixel])
    result[case["name"]] = encoded
    result[case["name"] + "__lossless"] = exact
with open(os.path.join(root, "blender_display.json"), "w", encoding="utf-8") as destination:
    json.dump(result, destination)
"#;

fn grid() -> Vec<[f64; 3]> {
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
    let samples: Vec<[f64; 3]> = SAMPLES.iter().copied().chain(grid()).collect();

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
    let mut optimizer_sensitive = 0_usize;
    for (view, key, _) in VIEWS {
        for (index, (exposure, gamma)) in SETTINGS.iter().copied().enumerate() {
            let name = format!("{key}_{index}");
            let default_rows = case_rows(&blender_views, &name, samples.len())?;
            let exact_rows =
                case_rows(&blender_views, &format!("{name}__lossless"), samples.len())?;
            let management = ColorManagement {
                view_transform: view,
                exposure,
                gamma,
                ..ColorManagement::default()
            };
            for (sample_index, sample) in samples.iter().enumerate() {
                let potter_rgb = management.display_encode(*sample)?;
                for channel in 0..3 {
                    let default_value = default_rows[sample_index][channel].clamp(0.0, 1.0);
                    let exact_value = exact_rows[sample_index][channel].clamp(0.0, 1.0);
                    let potter_value = potter_rgb[channel];
                    let reference = if (default_value - exact_value).abs() > OPTIMIZER_SENSITIVITY {
                        // Blender's default processor differs from its own lossless chain here
                        // (OCIO resamples the palette step): compare with the lossless chain, which
                        // Potter ports, and only for the False Color palette.
                        assert_eq!(
                            key, "false_color",
                            "{name}, sample {sample:?}: optimiser-sensitive outside False Color"
                        );
                        assert!(
                            (potter_value - exact_value).abs() <= LOSSLESS_TOLERANCE,
                            "{name}, sample {sample:?}, channel {channel}: Potter {potter_value} vs Blender lossless {exact_value} differ by more than {LOSSLESS_TOLERANCE:e}",
                        );
                        optimizer_sensitive += 1;
                        exact_value
                    } else {
                        let difference = (potter_value - default_value).abs();
                        if difference > worst.0 {
                            worst = (
                                difference,
                                format!(
                                    "{name}, sample {sample:?}, channel {channel}: Potter {potter_value} vs Blender {default_value}"
                                ),
                            );
                        }
                        default_value
                    };
                    assert!(
                        (byte(potter_value) - byte(reference)).abs() <= 1,
                        "{name}, sample {sample:?}, channel {channel}: byte {} vs {}",
                        byte(potter_value),
                        byte(reference),
                    );
                }
            }
        }
    }
    eprintln!(
        "largest float difference outside the optimiser-sensitive band {:.3e} at {}; {optimizer_sensitive} channel values compared with Blender's lossless chain",
        worst.0, worst.1
    );
    assert!(
        worst.0 <= FLOAT_TOLERANCE,
        "largest float difference {:.3e} exceeds {FLOAT_TOLERANCE:e}: {}",
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
