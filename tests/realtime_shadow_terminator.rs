//! Realtime EEVEE shadow-terminator normal and geometry offsets against Blender 5.2.2.
//!
//! `fixtures/native_oracle/eevee_shadow_terminator.py` reuses the known-good plane and smooth-sphere Sun-shadow scene from
//! `realtime_direct_light_cases.rs`, with zero, normal-offset, and geometry-cutoff variants.
//! Every frame is compared absolutely with Blender EEVEE using the existing 0.1 maximum and 1e-3 mean pixel bounds.

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/process.rs"]
mod process;

use std::{error::Error, fs, path::Path, process::Command};

use potter_core::{
    image::{ImageData, ImageInterpolation, load_image_data, sample},
    model::Image,
};
use serde_json::{Value, json};
use tempfile::tempdir;

const SIZE: u32 = 128;
const PIXEL_MAX_TOLERANCE: f64 = 0.1;
const PIXEL_MEAN_TOLERANCE: f64 = 1.0e-3;
const MIN_TERMINATOR_CHANGE: f64 = 1.0e-2;
const SCRIPT: &str = include_str!("fixtures/native_oracle/eevee_shadow_terminator.py");

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

#[test]
fn realtime_shadow_terminator_offsets_match_eevee() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        eprintln!("skipping EEVEE shadow-terminator oracle: Blender is unavailable");
        return Ok(());
    };
    let root = tempdir()?;
    let script = root.path().join("eevee_shadow_terminator.py");
    fs::write(&script, SCRIPT)?;

    let cases = ["zero", "normal", "geometry"];
    let mut actual = Vec::with_capacity(cases.len());
    let mut expected = Vec::with_capacity(cases.len());
    for case in cases {
        let blend = root.path().join(format!("{case}.blend"));
        let exr = root.path().join(format!("{case}.exr"));
        let oracle = root.path().join(format!("{case}.json"));
        run_blender_case(&blender, &script, case, &blend, &exr, &oracle)?;
        expected.push(indexed_pixels(read_eevee_pixels(&oracle)?));
        actual.push(indexed_pixels(render_potter(
            &blender,
            root.path(),
            case,
            &blend,
        )?));
    }

    for ((case, potter), eevee) in cases.into_iter().zip(&actual).zip(&expected) {
        let deviation = pixel_deviation(potter, eevee);
        let column = deviation.pixel % SIZE as usize;
        let row = deviation.pixel / SIZE as usize;
        assert!(
            deviation.maximum <= PIXEL_MAX_TOLERANCE,
            "{case}: Potter realtime {:?} differs from EEVEE {:?} by {} at pixel ({column}, {row}) (tolerance {PIXEL_MAX_TOLERANCE})",
            deviation.actual,
            deviation.expected,
            deviation.maximum,
        );
        assert!(
            deviation.mean <= PIXEL_MEAN_TOLERANCE,
            "{case}: Potter realtime mean pixel deviation from EEVEE is {} (tolerance {PIXEL_MEAN_TOLERANCE})",
            deviation.mean,
        );
    }

    let eevee_normal_change = maximum_difference(&expected[0], &expected[1]);
    let potter_normal_change = maximum_difference(&actual[0], &actual[1]);
    assert!(
        eevee_normal_change > MIN_TERMINATOR_CHANGE,
        "Blender EEVEE's nonzero normal offset changed pixels by only {eevee_normal_change}"
    );
    assert!(
        potter_normal_change > MIN_TERMINATOR_CHANGE,
        "Potter's nonzero normal offset changed pixels by only {potter_normal_change}"
    );
    let eevee_geometry_change = maximum_difference(&expected[1], &expected[2]);
    let potter_geometry_change = maximum_difference(&actual[1], &actual[2]);
    assert!(
        eevee_geometry_change > MIN_TERMINATOR_CHANGE,
        "Blender EEVEE's geometry cutoff changed pixels by only {eevee_geometry_change}"
    );
    assert!(
        potter_geometry_change > MIN_TERMINATOR_CHANGE,
        "Potter's geometry cutoff changed pixels by only {potter_geometry_change}"
    );
    Ok(())
}

fn run_blender_case(
    blender: &Path,
    script: &Path,
    case: &str,
    blend: &Path,
    exr: &Path,
    oracle: &Path,
) -> TestResult {
    let mut command = Command::new(blender);
    command.args([
        "--background",
        "--factory-startup",
        "--disable-autoexec",
        "--python-exit-code",
        "3",
    ]);
    command
        .arg("--python")
        .arg(script)
        .arg("--")
        .arg(case)
        .arg(blend)
        .arg(exr)
        .arg(oracle);
    checked(command, "Blender EEVEE shadow-terminator render")?;
    Ok(())
}

fn render_potter(
    blender: &Path,
    root: &Path,
    case: &str,
    blend: &Path,
) -> TestResult<Vec<[f64; 3]>> {
    let project = root.join(format!("project_{case}"));
    pot_json(|command| {
        command.arg("init").arg(&project);
    })?;
    pot_json(|command| {
        command
            .arg("import")
            .arg(&project)
            .arg("--file")
            .arg(blend)
            .args([
                "--format",
                "blend",
                "--mode",
                "replace",
                "--base-revision",
                "0",
                "--blender",
            ])
            .arg(blender);
    })?;

    let frames = root.join(format!("frames_{case}"));
    pot_json(|command| {
        command
            .arg("render")
            .arg(&project)
            .args(["--engine", "realtime", "--format", "exr", "--out"])
            .arg(&frames);
    })?;
    let frame = frames.join("frame_0001.exr");
    let inspected = pot_json(|command| {
        command.arg("inspect").arg(&project);
    })?;
    let revision = inspected["scene"]["revision"]
        .as_u64()
        .ok_or("project revision is missing")?;
    let operations = root.join(format!("load_{case}.json"));
    fs::write(
        &operations,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": revision,
            "operations": [{
                "op": "image.load",
                "id": "render",
                "path": frame.to_string_lossy(),
            }],
        }))?,
    )?;
    pot_json(|command| {
        command
            .arg("apply")
            .arg(&project)
            .arg("--file")
            .arg(&operations);
    })?;

    let scene: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let image: Image = serde_json::from_value(scene["images"]["render"].clone())?;
    Ok(image_pixels(&load_image_data(
        &image,
        &project,
        ImageInterpolation::Closest,
    )?))
}

fn image_pixels(data: &ImageData) -> Vec<[f64; 3]> {
    (0..SIZE)
        .flat_map(|row| {
            (0..SIZE).map(move |column| {
                let rgba = sample(
                    data,
                    [
                        (f64::from(column) + 0.5) / f64::from(SIZE),
                        (f64::from(row) + 0.5) / f64::from(SIZE),
                    ],
                    false,
                );
                [rgba[0], rgba[1], rgba[2]]
            })
        })
        .collect()
}

fn indexed_pixels(pixels: Vec<[f64; 3]>) -> Vec<(usize, [f64; 3])> {
    pixels.into_iter().enumerate().collect()
}

fn read_eevee_pixels(path: &Path) -> TestResult<Vec<[f64; 3]>> {
    let oracle: Value = serde_json::from_slice(&fs::read(path)?)?;
    assert!(
        oracle["version"]
            .as_str()
            .is_some_and(|version| version.starts_with("5.2.2")),
        "unexpected Blender oracle version: {}",
        oracle["version"]
    );
    assert_eq!(oracle["width"].as_u64(), Some(u64::from(SIZE)));
    assert_eq!(oracle["height"].as_u64(), Some(u64::from(SIZE)));
    let pixels: Vec<[f64; 3]> = serde_json::from_value(oracle["pixels"].clone())?;
    assert_eq!(pixels.len(), (SIZE * SIZE) as usize);
    Ok(pixels)
}

struct PixelDeviation {
    maximum: f64,
    mean: f64,
    pixel: usize,
    actual: [f64; 3],
    expected: [f64; 3],
}

fn pixel_deviation(actual: &[(usize, [f64; 3])], expected: &[(usize, [f64; 3])]) -> PixelDeviation {
    let mut deviation = PixelDeviation {
        maximum: 0.0,
        mean: 0.0,
        pixel: 0,
        actual: [0.0; 3],
        expected: [0.0; 3],
    };
    let mut total = 0.0;
    let mut count = 0;
    for ((index, actual), (expected_index, expected)) in actual.iter().zip(expected) {
        assert_eq!(index, expected_index, "sphere pixel indices differ");
        let actual_pixel = *actual;
        let expected_pixel = *expected;
        for (actual, expected) in actual_pixel.iter().zip(&expected_pixel) {
            let difference = (*actual - *expected).abs();
            if difference > deviation.maximum {
                deviation.maximum = difference;
                deviation.pixel = *index;
                deviation.actual = actual_pixel;
                deviation.expected = expected_pixel;
            }
            total += difference;
            count += 1;
        }
    }
    deviation.mean = total / count as f64;
    deviation
}

fn maximum_difference(first: &[(usize, [f64; 3])], second: &[(usize, [f64; 3])]) -> f64 {
    first
        .iter()
        .zip(second)
        .flat_map(|((_, first), (_, second))| first.iter().zip(second))
        .map(|(first, second)| (first - second).abs())
        .fold(0.0, f64::max)
}

fn pot_json(configure: impl FnOnce(&mut Command)) -> TestResult<Value> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_pot"));
    configure(&mut command);
    command.arg("--json");
    let output = checked(command, "pot")?;
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn checked(command: Command, label: &str) -> TestResult<std::process::Output> {
    let output = process::run_guarded(command)?;
    if !output.status.success() {
        return Err(format!(
            "{label} failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        )
        .into());
    }
    Ok(output)
}
