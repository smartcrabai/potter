//! The Cycles pixel filter against Blender's Cycles render (`fixtures/native_oracle/pixel_filter.py`).
//!
//! A thin-line checker on a white floor is rendered by Blender with each filter, 32 seeds of 256
//! samples per pixel. The same `.blend` (with `pixel_filter_type` and `filter_width` set, so the
//! import path is exercised too) is rendered by Potter at 256 samples per pixel.
//!
//! The per-pixel values differ between filters, while the frame mean barely does, so the check is
//! per pixel. Potter's image is fitted as `own + alpha * (alternative - own)`, where `own` is
//! Blender's reference for the filter under test and `alternative` is Blender's reference for a
//! filter Potter might wrongly use. `alpha` is 0 when Potter uses the right filter and 1 when it
//! uses the alternative. The test requires `|alpha| < 0.5`.
//!
//! The standard error of `alpha` is computed from the measured noise of both images:
//! `sqrt(sum (sigma_p^2 + sigma_b^2) d^2) / sum d^2`, where `d` is the difference of the two
//! references, `sigma_b` is Blender's per-pixel standard error, and `sigma_p` is Potter's
//! per-pixel noise, scaled from Blender's measured per-seed deviation to 256 samples. At these
//! settings the standard error is 0.02 to 0.06 (1024 samples gave 0.01 to 0.03; it scales with the
//! inverse square root of the sample count), so the threshold 0.5 is about 9 or more standard errors from
//! either hypothesis. The threshold is not loosened to pass.
//!
//! Alternatives: the Potter box filter (what Potter did before this filter existed) is an
//! alternative for both kinds; the Blackman-Harris filter is one for the Gaussian.

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/pot_json.rs"]
mod pot_json;
#[path = "common/process.rs"]
mod process;

use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use potter_core::image::{ImageInterpolation, load_image_data, sample};
use potter_core::model::Image;
use serde_json::{Value, json};
use tempfile::tempdir;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const SCRIPT: &str = include_str!("fixtures/native_oracle/pixel_filter.py");
/// Frame edge in pixels of the oracle scene (`pixel_filter.py`).
const RESOLUTION: u32 = 32;
/// Blender seeds per reference (`pixel_filter.py` `seeds` argument).
const BLENDER_SEEDS: u32 = 32;
/// Blender samples per pixel per seed (`pixel_filter.py` `SAMPLES`).
const BLENDER_SAMPLES: f64 = 256.0;
/// Potter samples per pixel. Potter's noise is scaled from Blender's to this count. 1024 would
/// take about 100 s per Potter render in a debug build, past the per-test budget.
const POTTER_SAMPLES: u32 = 256;
/// The threshold on `alpha`: the midpoint between the correct filter (0) and an alternative (1).
const ALPHA_LIMIT: f64 = 0.5;
/// A sanity bound: the standard error must leave the threshold far from either hypothesis.
const MAX_ALPHA_STANDARD_ERROR: f64 = 0.1;

/// Blender's reference for one filter: the mean and standard error of each pixel.
struct Reference {
    mean: Vec<f64>,
    stderr: Vec<f64>,
}

fn text(path: &Path) -> TestResult<&str> {
    Ok(path.to_str().ok_or("path is not UTF-8")?)
}

fn read_scene(project: &Path) -> TestResult<Value> {
    Ok(serde_json::from_slice(&fs::read(
        project.join("scene.json"),
    )?)?)
}

/// Applies `operations` to `project` atomically, against the project's current revision.
fn apply(project: &Path, root: &Path, operations: &Value) -> TestResult<Value> {
    let revision = read_scene(project)?["revision"]
        .as_u64()
        .ok_or("scene revision is missing")?;
    let batch = root.join("batch.json");
    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": revision,
            "operations": operations,
        }))?,
    )?;
    pot_json::pot_json(&["apply", text(project)?, "--file", text(&batch)?])
}

/// The numbers under `key` in a Blender reference JSON.
fn reference_numbers(value: &Value, kind: &str, key: &str) -> TestResult<Vec<f64>> {
    let items = value[key]
        .as_array()
        .ok_or_else(|| format!("{kind} reference has no `{key}`"))?;
    let numbers = items
        .iter()
        .map(|item| {
            item.as_f64()
                .ok_or_else(|| format!("{kind} `{key}` holds a non-number"))
        })
        .collect::<Result<Vec<f64>, String>>()?;
    Ok(numbers)
}

/// Renders `kind` in Blender, returning its `.blend` and reference.
fn blender_reference(
    blender: &Path,
    root: &Path,
    script: &Path,
    kind: &str,
) -> TestResult<(PathBuf, Reference)> {
    let blend = root.join(format!("{kind}.blend"));
    let json_path = root.join(format!("{kind}.json"));
    let output = Command::new(blender)
        .args([
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            "--python-exit-code",
            "3",
            "--python",
        ])
        .arg(script)
        .arg("--")
        .arg(&blend)
        .arg(&json_path)
        .arg(kind)
        .arg(BLENDER_SEEDS.to_string())
        .output()?;
    assert!(
        output.status.success(),
        "Blender {kind} reference failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&fs::read(&json_path)?)?;
    let reference = Reference {
        mean: reference_numbers(&value, kind, "mean")?,
        stderr: reference_numbers(&value, kind, "stderr")?,
    };
    let pixels = (RESOLUTION * RESOLUTION) as usize;
    if reference.mean.len() != pixels || reference.stderr.len() != pixels {
        return Err(format!("{kind} reference is not {RESOLUTION} x {RESOLUTION}").into());
    }
    Ok((blend, reference))
}

/// The fit of Potter's image to `own + alpha * (alternative - own)`, and its standard error.
fn fit_alpha(potter: &[f64], own: &Reference, alternative: &Reference) -> (f64, f64) {
    let scale = (BLENDER_SAMPLES / f64::from(POTTER_SAMPLES)).sqrt();
    let blender_seed_scale = f64::from(BLENDER_SEEDS).sqrt();
    let mut numerator = 0.0;
    let mut difference_energy = 0.0;
    let mut variance = 0.0;
    for (((&measured, &own_mean), &own_error), &alternative_mean) in potter
        .iter()
        .zip(&own.mean)
        .zip(&own.stderr)
        .zip(&alternative.mean)
    {
        let d = alternative_mean - own_mean;
        numerator += (measured - own_mean) * d;
        difference_energy += d * d;
        // Blender's per-seed deviation is its standard error times sqrt(seeds); Potter's noise at
        // its own sample count follows from that.
        let potter_sd = own_error * blender_seed_scale * scale;
        variance += (potter_sd * potter_sd + own_error * own_error) * d * d;
    }
    (
        numerator / difference_energy,
        variance.sqrt() / difference_energy,
    )
}

/// Imports `blend` into a fresh project, renders it in Potter at `POTTER_SAMPLES`, and returns the
/// pixels, row by row.
fn potter_pixels(blender: &Path, root: &Path, blend: &Path) -> TestResult<Vec<f64>> {
    let project = root.join("project");
    pot_json::pot_json(&["init", text(&project)?])?;
    let envelope = pot_json::pot_json(&[
        "import",
        text(&project)?,
        "--file",
        text(blend)?,
        "--format",
        "blend",
        "--mode",
        "replace",
        "--base-revision",
        "0",
        "--blender",
        text(blender)?,
    ])?;
    assert_eq!(envelope["result"]["losses"], json!([]), "{envelope}");

    let scene = read_scene(&project)?;
    let active = scene["active_scene"]
        .as_str()
        .ok_or("active scene is missing")?
        .to_owned();
    let applied = apply(
        &project,
        root,
        &json!([
            {"op": "render.update", "target": {"id": active}, "set": {"resolution_x": RESOLUTION, "resolution_y": RESOLUTION, "samples": POTTER_SAMPLES, "seed": 0, "max_bounces": 4}}
        ]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");

    let frames = root.join("frames");
    pot_json::pot_json(&[
        "render",
        text(&project)?,
        "--format",
        "exr",
        "--out",
        text(&frames)?,
    ])?;
    let loaded = apply(
        &project,
        root,
        &json!([{"op": "image.load", "id": "render", "path": frames.join("frame_0001.exr").to_string_lossy()}]),
    )?;
    assert_eq!(loaded["ok"], true, "{loaded}");

    let scene = read_scene(&project)?;
    let image: Image = serde_json::from_value(scene["images"]["render"].clone())?;
    let data = load_image_data(&image, &project, ImageInterpolation::Closest)?;
    let mut pixels = Vec::with_capacity((RESOLUTION * RESOLUTION) as usize);
    for y in 0..RESOLUTION {
        for x in 0..RESOLUTION {
            let u = (f64::from(x) + 0.5) / f64::from(RESOLUTION);
            let v = (f64::from(y) + 0.5) / f64::from(RESOLUTION);
            pixels.push(sample(&data, [u, v], false)[0]);
        }
    }
    Ok(pixels)
}

/// Checks Potter's `kind` render against Blender's `kind` reference and each alternative.
fn check_kind(kind: &str, alternatives: &[&str]) -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = root.path().join("pixel_filter.py");
    fs::write(&script, SCRIPT)?;
    let (blend, own) = blender_reference(&blender, root.path(), &script, kind)?;
    let mut references = Vec::new();
    for alternative in alternatives {
        let (_, reference) = blender_reference(&blender, root.path(), &script, alternative)?;
        references.push((*alternative, reference));
    }

    let potter = potter_pixels(&blender, root.path(), &blend)?;
    for (alternative, reference) in &references {
        let (alpha, standard_error) = fit_alpha(&potter, &own, reference);
        assert!(
            standard_error < MAX_ALPHA_STANDARD_ERROR,
            "{kind} against {alternative}: standard error {standard_error:.4} is too large to tell the filters apart"
        );
        assert!(
            alpha.abs() < ALPHA_LIMIT,
            "{kind}: Potter's image moves {alpha:.3} (standard error {standard_error:.3}) of the way to Blender's {alternative} filter; expected 0"
        );
    }
    Ok(())
}

#[test]
fn blackman_harris_pixel_filter_matches_cycles_and_not_box() -> TestResult {
    check_kind("blackman_harris", &["box"])
}

#[test]
fn gaussian_pixel_filter_matches_cycles_and_not_box_or_blackman_harris() -> TestResult {
    check_kind("gaussian", &["box", "blackman_harris"])
}

/// The filter type and width Potter writes to a `.blend` are the ones Blender reads back.
#[test]
fn pixel_filter_settings_round_trip_through_blend() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = root.path().join("pixel_filter.py");
    fs::write(&script, SCRIPT)?;
    let (blend, _) = blender_reference(&blender, root.path(), &script, "blackman_harris")?;

    let project = root.path().join("project");
    pot_json::pot_json(&["init", text(&project)?])?;
    let envelope = pot_json::pot_json(&[
        "import",
        text(&project)?,
        "--file",
        text(&blend)?,
        "--format",
        "blend",
        "--mode",
        "replace",
        "--base-revision",
        "0",
        "--blender",
        text(&blender)?,
    ])?;
    assert_eq!(envelope["result"]["losses"], json!([]), "{envelope}");
    let scene = read_scene(&project)?;
    let active = scene["active_scene"]
        .as_str()
        .ok_or("active scene is missing")?
        .to_owned();
    let applied = apply(
        &project,
        root.path(),
        &json!([
            {"op": "render.update", "target": {"id": active}, "set": {"pixel_filter_type": "GAUSSIAN", "filter_width": 2.5}}
        ]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");

    let out = root.path().join("round_trip.blend");
    pot_json::pot_json(&[
        "export",
        text(&project)?,
        "--format",
        "blend",
        "--out",
        text(&out)?,
        "--blender",
        text(&blender)?,
    ])?;
    let expression = "import bpy; cycles = bpy.context.scene.cycles; \
        assert cycles.pixel_filter_type == 'GAUSSIAN', cycles.pixel_filter_type; \
        assert abs(cycles.filter_width - 2.5) < 1e-6, cycles.filter_width";
    let checked = Command::new(&blender)
        .args(["--background", "--disable-autoexec"])
        .arg(&out)
        .args(["--python-exit-code", "1", "--python-expr"])
        .arg(expression)
        .output()?;
    assert!(
        checked.status.success(),
        "Blender did not read the exported filter back: {}",
        String::from_utf8_lossy(&checked.stderr)
    );
    Ok(())
}
