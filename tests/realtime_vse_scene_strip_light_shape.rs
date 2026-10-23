//! EEVEE camera-light radiance through real VSE Scene strips, compared to Blender's saved EXR consumers.

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
    TestResult, import_blend, pot, read_scene, run_blender_arguments, write_script,
};
use potter_core::image::decode_pixels;
use potter_core::model::ImageColorspace;
use serde_json::Value;
use tempfile::tempdir;

const SCRIPT: &str = include_str!("fixtures/native_oracle/eevee_vse_light_shape_scene_strip.py");
const BLENDER_REPLACE_EXR: &[u8] =
    include_bytes!("fixtures/native_oracle/eevee_macos/vse_light_shape_scene_strip_replace.exr");
const BLENDER_ALPHA_OVER_EXR: &[u8] =
    include_bytes!("fixtures/native_oracle/eevee_macos/vse_light_shape_scene_strip_alpha_over.exr");
const CROSSFADE_SCRIPT: &str =
    include_str!("fixtures/native_oracle/eevee_vse_light_shape_crossfade.py");
const BLENDER_CROSSFADE_EXR: &[u8] =
    include_bytes!("fixtures/native_oracle/eevee_macos/vse_light_shape_crossfade.exr");
const RESOLUTION: u32 = 64;
const RGB_TOLERANCE: f64 = 0.01;
const ALPHA_TOLERANCE: f64 = 1.0e-6;

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

fn scene_id_named(document: &Value, name: &str) -> TestResult<String> {
    document["scenes"]
        .as_object()
        .ok_or("imported scene registry is missing")?
        .iter()
        .find(|(_, scene)| scene["name"].as_str() == Some(name))
        .map(|(id, _)| id.clone())
        .ok_or_else(|| format!("imported Blender scene `{name}` is missing").into())
}

fn render_scene_strip(
    project: &Path,
    scene_id: &str,
    frame: u32,
    root: &Path,
    name: &str,
) -> TestResult<Vec<u8>> {
    let output_dir = root.join(format!("frames_{name}"));
    let mut command = pot();
    command
        .arg("render")
        .arg(project)
        .arg("--frame")
        .arg(frame.to_string())
        .args([
            "--scene-id",
            scene_id,
            "--engine",
            "realtime",
            "--format",
            "exr",
            "--out",
        ])
        .arg(&output_dir)
        .arg("--json");
    let output = command.output()?;
    assert!(
        output.status.success(),
        "Potter VSE render `{name}` failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(fs::read(output_dir.join(format!("frame_{frame:04}.exr")))?)
}

fn assert_replace_inputs(label: &str, pixels: &[[f64; 4]]) {
    let alpha_zero_shape = pixels
        .iter()
        .filter(|pixel| pixel[3].abs() <= ALPHA_TOLERANCE)
        .max_by(|a, b| {
            let energy_a = a[0].max(a[1]).max(a[2]);
            let energy_b = b[0].max(b[1]).max(b[2]);
            energy_a.total_cmp(&energy_b)
        })
        .expect("Replace output contains an alpha-zero pixel");
    assert!(
        alpha_zero_shape[0]
            .max(alpha_zero_shape[1])
            .max(alpha_zero_shape[2])
            > 0.01,
        "{label} lost camera-shape RGB at alpha zero: {alpha_zero_shape:?}"
    );

    assert!(
        pixels.iter().any(|pixel| {
            (pixel[3] - 0.5).abs() <= ALPHA_TOLERANCE
                && (0.1..0.15).contains(&pixel[0])
                && (0.04..0.06).contains(&pixel[1])
                && (0.02..0.03).contains(&pixel[2])
        }),
        "{label} has no half-alpha base-only control pixel"
    );
}

fn assert_matches_native(label: &str, actual: &[[f64; 4]], expected: &[[f64; 4]]) {
    assert_eq!(actual.len(), expected.len(), "{label} pixel count");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        for channel in 0..4 {
            let tolerance = if channel == 3 {
                ALPHA_TOLERANCE
            } else {
                RGB_TOLERANCE
            };
            assert!(
                (actual[channel] - expected[channel]).abs() <= tolerance,
                "{label} EXR pixel {index}, channel {channel}: Potter={} Blender={} tolerance={tolerance}",
                actual[channel],
                expected[channel]
            );
        }
    }
}

#[test]
fn scene_strip_light_shape_matches_blender_vse_replace_alpha_over_and_crossfade_exrs() -> TestResult
{
    let Some(blender) = blender_checked::blender_executable() else {
        if std::env::var_os("POTTER_REQUIRE_BLENDER").is_some() {
            return Err("Blender is required for the VSE Scene-strip light-shape oracle".into());
        }
        eprintln!("skipping VSE Scene-strip light-shape oracle; Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let script = write_script(root, "eevee_vse_light_shape_scene_strip.py", SCRIPT)?;
    let blend = root.join("vse_light_shape_scene_strip.blend");
    run_blender_arguments(
        &blender,
        &script,
        None,
        &[
            &blend,
            &root.join("unused_replace.exr"),
            &root.join("unused_alpha_over.exr"),
            Path::new("0"),
        ],
    )?;

    let (project, imported) = import_blend(&blender, root, "project", &blend)?;
    assert_eq!(imported["result"]["committed"], true);
    let document = read_scene(&project)?;
    let replace_id = scene_id_named(&document, "LightShapeVSEReplace")?;
    let alpha_over_id = scene_id_named(&document, "LightShapeVSEAlphaOver")?;

    let blender_replace = decode_exr(BLENDER_REPLACE_EXR, "Blender Replace oracle")?;
    let blender_alpha_over = decode_exr(BLENDER_ALPHA_OVER_EXR, "Blender AlphaOver oracle")?;
    assert_replace_inputs("Blender Replace oracle", &blender_replace);
    let potter_replace_bytes = render_scene_strip(&project, &replace_id, 1, root, "replace")?;
    let potter_replace = decode_exr(&potter_replace_bytes, "Potter Replace output")?;
    assert_replace_inputs("Potter Replace output", &potter_replace);
    assert_matches_native("Scene-strip Replace", &potter_replace, &blender_replace);

    let potter_alpha_over_bytes =
        render_scene_strip(&project, &alpha_over_id, 1, root, "alpha_over")?;
    let potter_alpha_over = decode_exr(&potter_alpha_over_bytes, "Potter AlphaOver output")?;
    assert_matches_native(
        "Scene-strip AlphaOver",
        &potter_alpha_over,
        &blender_alpha_over,
    );
    let crossfade_script =
        write_script(root, "eevee_vse_light_shape_crossfade.py", CROSSFADE_SCRIPT)?;
    let crossfade_blend = root.join("vse_light_shape_crossfade.blend");
    run_blender_arguments(
        &blender,
        &crossfade_script,
        None,
        &[
            &crossfade_blend,
            &root.join("unused_crossfade.exr"),
            Path::new("0"),
        ],
    )?;
    let (crossfade_project, crossfade_import) =
        import_blend(&blender, root, "crossfade_project", &crossfade_blend)?;
    assert_eq!(crossfade_import["result"]["committed"], true);
    let crossfade_document = read_scene(&crossfade_project)?;
    let crossfade_id = scene_id_named(&crossfade_document, "LightShapeVSECrossfade")?;

    let blender_crossfade = decode_exr(BLENDER_CROSSFADE_EXR, "Blender Cross oracle")?;
    let center = blender_crossfade
        [(RESOLUTION as usize / 2) * RESOLUTION as usize + RESOLUTION as usize / 2];
    let background = blender_crossfade[4 * RESOLUTION as usize + 4];
    assert!(
        (center[3] - 0.5).abs() <= ALPHA_TOLERANCE
            && (background[3] - 0.5).abs() <= ALPHA_TOLERANCE
            && center[..3]
                .iter()
                .zip(&background[..3])
                .any(|(shape, base)| *shape - *base > 0.01),
        "Blender Cross transition lost midpoint alpha-zero radiance: center={center:?} background={background:?}"
    );
    let potter_crossfade_bytes =
        render_scene_strip(&crossfade_project, &crossfade_id, 3, root, "crossfade")?;
    let potter_crossfade = decode_exr(&potter_crossfade_bytes, "Potter Cross output")?;
    assert_matches_native("Scene-strip Cross", &potter_crossfade, &blender_crossfade);
    Ok(())
}
