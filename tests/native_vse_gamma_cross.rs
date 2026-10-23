//! The imported Blender VSE `GammaCross` strip renders through Potter's CLI against a saved Blender EXR.

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

use native_oracle::{TestResult, checked, import_blend, pot, read_scene};
use potter_core::{image::decode_pixels, model::ImageColorspace};
use serde_json::Value;
use tempfile::tempdir;

const BLENDER_BLEND: &[u8] =
    include_bytes!("fixtures/native_oracle/eevee_macos/gamma_cross_native.blend");
const BLENDER_EXR: &[u8] =
    include_bytes!("fixtures/native_oracle/eevee_macos/gamma_cross_native.exr");
const RESOLUTION: u32 = 64;
const CHANNEL_TOLERANCE: f64 = 1.0e-6;

fn active_scene_id(scene: &Value) -> TestResult<String> {
    scene["active_scene"]
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| "imported scene has no active scene ID".into())
}

fn decode_exr(bytes: &[u8], label: &str) -> TestResult<Vec<[f64; 4]>> {
    assert_eq!(
        bytes.get(..4),
        Some(&[0x76, 0x2f, 0x31, 0x01][..]),
        "{label} is not an OpenEXR file"
    );
    let (width, height, pixels) = decode_pixels(bytes, ImageColorspace::Linear)?;
    assert_eq!(
        (width, height),
        (RESOLUTION, RESOLUTION),
        "{label} dimensions"
    );
    Ok(pixels)
}

fn render_exr(project: &Path, scene_id: &str, root: &Path) -> TestResult<Vec<u8>> {
    let output_dir = root.join("gamma_cross_frames");
    let mut command = pot();
    command
        .arg("render")
        .arg(project)
        .args(["--frame", "3", "--scene-id"])
        .arg(scene_id)
        .args(["--engine", "realtime", "--format", "exr", "--out"])
        .arg(&output_dir)
        .arg("--json");
    checked(&mut command, "Potter GammaCross render")?;
    Ok(fs::read(output_dir.join("frame_0003.exr"))?)
}

fn assert_matches_native(actual: &[[f64; 4]], expected: &[[f64; 4]]) {
    assert_eq!(actual.len(), expected.len(), "GammaCross pixel count");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        for channel in 0..4 {
            assert!(
                (actual[channel] - expected[channel]).abs() <= CHANNEL_TOLERANCE,
                "GammaCross EXR pixel {index}, channel {channel}: Potter={} Blender={} tolerance={CHANNEL_TOLERANCE}",
                actual[channel],
                expected[channel]
            );
        }
    }
}

#[test]
fn imported_gamma_cross_matches_blenders_vse_exr_consumer() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        if std::env::var_os("POTTER_REQUIRE_BLENDER").is_some() {
            return Err("Blender is required for the VSE GammaCross oracle".into());
        }
        eprintln!("skipping VSE GammaCross oracle; Blender is unavailable");
        return Ok(());
    };

    let directory = tempdir()?;
    let root = directory.path();
    let source = root.join("gamma_cross_native.blend");
    fs::write(&source, BLENDER_BLEND)?;
    let (project, import) = import_blend(&blender, root, "gamma_cross_project", &source)?;
    assert_eq!(import["result"]["committed"], true);
    let scene = read_scene(&project)?;
    let scene_id = active_scene_id(&scene)?;
    let actual = decode_exr(
        &render_exr(&project, &scene_id, root)?,
        "Potter VSE GammaCross output",
    )?;
    let expected = decode_exr(BLENDER_EXR, "Blender VSE GammaCross capture")?;
    assert_matches_native(&actual, &expected);
    Ok(())
}
