//! Compares Potter's display encoding with Blender 5.2.2's own OCIO display processor for the
//! same views: Standard, AgX, Filmic, Raw and False Color on the sRGB display with look "None",
//! exposure 0 and gamma 1, which is the processor Blender's render view uses. It does not run a
//! camera render. The LUT bytes and the view chains are documented in
//! `crates/potter-core/src/color/ocio.rs`.

use std::{collections::BTreeMap, error::Error, fs, io, process::Command};

use potter_core::color::{ColorManagement, ViewTransform};
use tempfile::tempdir;

#[path = "common/blender_file.rs"]
mod blender_file;
use blender_file::blender_executable;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

/// Scene-linear Rec.709 samples: greys across the exposure range, a saturated colour and an
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

/// Potter view and the key the Blender script writes its results under.
const VIEWS: [(ViewTransform, &str); 5] = [
    (ViewTransform::Standard, "standard"),
    (ViewTransform::AgX, "ag_x"),
    (ViewTransform::Filmic, "filmic"),
    (ViewTransform::Raw, "raw"),
    (ViewTransform::FalseColor, "false_color"),
];

const BLENDER_SCRIPT: &str = r#"
import bpy
import json
import os
import sys

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"display parity requires Blender 5.2.2, found {bpy.app.version_string}")

import PyOpenColorIO as OCIO

root = sys.argv[sys.argv.index("--") + 1]
with open(os.path.join(root, "samples.json"), encoding="utf-8") as source:
    samples = json.load(source)
config_path = os.path.join(
    bpy.utils.system_resource("DATAFILES"), "colormanagement", "config.ocio"
)
config = OCIO.Config.CreateFromFile(config_path)
views = {
    "standard": "Standard",
    "ag_x": "AgX",
    "filmic": "Filmic",
    "raw": "Raw",
    "false_color": "False Color",
}
result = {}
for key, view in views.items():
    transform = OCIO.DisplayViewTransform()
    transform.setSrc("Linear Rec.709")
    transform.setDisplay("sRGB")
    transform.setView(view)
    processor = config.getProcessor(transform)
    cpu = processor.getDefaultCPUProcessor()
    encoded = []
    for sample in samples:
        pixel = list(sample)
        cpu.applyRGB(pixel)
        encoded.append(pixel)
    result[key] = encoded
with open(os.path.join(root, "blender_display.json"), "w", encoding="utf-8") as destination:
    json.dump(result, destination)
"#;

fn byte(value: f64) -> i64 {
    (value * 255.0).round() as i64
}

#[test]
fn display_views_match_blender_ocio_display_processor() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping color view transform Blender parity: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = fs::canonicalize(directory.path())?;
    fs::write(root.join("samples.json"), serde_json::to_vec(&SAMPLES)?)?;
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

    for (view, key) in VIEWS {
        let blender_rows = blender_views
            .get(key)
            .ok_or_else(|| io::Error::other(format!("Blender output has no {key} view")))?;
        if blender_rows.len() != SAMPLES.len() {
            return Err(io::Error::other(format!(
                "Blender {key} view returned {} samples, expected {}",
                blender_rows.len(),
                SAMPLES.len()
            ))
            .into());
        }
        let management = ColorManagement {
            view_transform: view,
            ..ColorManagement::default()
        };
        for (sample, blender_rgb) in SAMPLES.iter().zip(blender_rows) {
            let potter_rgb = management.display_encode(*sample)?;
            for channel in 0..3 {
                let potter_byte = byte(potter_rgb[channel]);
                let blender_byte = byte(blender_rgb[channel].clamp(0.0, 1.0));
                assert!(
                    (potter_byte - blender_byte).abs() <= 1,
                    "{key} view, sample {sample:?}, channel {channel}: Potter {} (byte {potter_byte}) \
                     vs Blender {} (byte {blender_byte})",
                    potter_rgb[channel],
                    blender_rgb[channel],
                );
            }
        }
    }
    Ok(())
}
