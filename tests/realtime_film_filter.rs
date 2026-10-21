//! Realtime Beauty's film filter (`render.filter_size`, Blender's Filter Size), checked against Blender
//! 5.2.2's EEVEE on a sub-pixel edge.
//!
//! `fixtures/native_oracle/eevee_film_filter.py` renders a white emitter whose edge lies at pixel column 32.3 of a
//! 64 px frame, once per filter size, with EEVEE at 256 samples. Its column profiles are committed in
//! `eevee_film_filter.json`; run with a render directory of `0`, the script only saves the `.blend`, which the test
//! imports (Filter Size 1.5). The test renders the same frame with `pot render --engine realtime` at each filter size,
//! with 16 samples, and compares each column's mean over the rows against EEVEE's column profile for that size.
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

use std::{error::Error, path::Path};

use native_oracle::{
    TestResult, apply_operations, floats, import_blend, pot, read_scene, run_blender_arguments,
    write_script,
};
use potter_core::image::{ImageInterpolation, load_image_data, sample};
use potter_core::model::Image;
use serde_json::{Value, json};
use tempfile::tempdir;

/// The scene, which Blender saves as a `.blend` (`eevee_film_filter.py`), and the EEVEE column profiles it rendered
/// at 256 samples, committed beside it.
const SCRIPT: &str = include_str!("fixtures/native_oracle/eevee_film_filter.py");
const ORACLE: &str = include_str!("fixtures/native_oracle/eevee_film_filter.json");

/// Render settings shared with `eevee_film_filter.py`, and the sample count of Potter's realtime render.
const RESOLUTION: u32 = 64;
const SAMPLES: u32 = 16;

/// The filter sizes compared, as the keys of the oracle's `profiles`. `0.01` is the point sample below the
/// jitter threshold, `0.5` a jitter inside a half-pixel disk, and `1.5` Blender's default.
const FILTER_SIZES: [&str; 3] = ["0.01", "0.5", "1.5"];

/// The columns compared: the edge at pixel 32.3 and the neighbours that a filter of 1.5 px reaches.
const COLUMNS: std::ops::RangeInclusive<usize> = 28..=36;

/// Linear tolerance on a column's row-averaged red value against EEVEE. PROVISIONAL. Potter's Halton pattern is
/// deterministic, and EEVEE's 256-sample profiles are committed. Potter at 16 samples (this test) deviates by up to
/// 0.0157 (1.5 px), a quadrature error. Potter at EEVEE's 256 samples, same pattern and same film, deviates by up to
/// 0.0095 at 0.5 px (column 32, the edge), 0.0022 at 1.5 px, and 0 at 0.01 px. That 0.0095 is not EXR quantisation: the
/// committed profile is half-float, but that error is near 5e-4. EEVEE's film also weights each sample by Karis's
/// `1 / (4 + Y)` luma (`eevee_film.bsl.hh`, `sample_accum_combined`). Applying it in Potter closes the 0.5 px gap to
/// 0.0036 but opens 1.5 px to 0.048, so the source model does not yet explain EEVEE's weighting. Until it does, the
/// bound stays at 0.03.
const TOLERANCE: f64 = 0.03;

/// The mean red value of each column of the image `id`, averaged over the rows, read with Potter's own EXR loader.
fn column_profile(project: &Path, id: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    let scene = read_scene(project)?;
    let image: Image = serde_json::from_value(scene["images"][id].clone())?;
    let data = load_image_data(&image, project, ImageInterpolation::Closest)?;
    let size = RESOLUTION as usize;
    let texels = f64::from(RESOLUTION);
    Ok((0..size)
        .map(|column| {
            let u = (column as f64 + 0.5) / texels;
            let total: f64 = (0..size)
                .map(|row| sample(&data, [u, (row as f64 + 0.5) / texels], false)[0])
                .sum();
            total / size as f64
        })
        .collect())
}

#[test]
fn realtime_film_filter_matches_eevee_column_profiles() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "eevee_film_filter.py", SCRIPT)?;
    let blend = root.path().join("film_filter.blend");
    // The render directory `0` stops the generator after saving the .blend: the profiles are committed.
    run_blender_arguments(
        &blender,
        &script,
        None,
        &[
            blend.as_path(),
            root.path().join("unused.json").as_path(),
            Path::new("0"),
        ],
    )?;
    let oracle: Value = serde_json::from_str(ORACLE)?;

    let (project, _) = import_blend(&blender, root.path(), "project", &blend)?;
    let scene = read_scene(&project)?;
    let active = scene["active_scene"]
        .as_str()
        .ok_or("active scene is missing")?
        .to_owned();

    let mut report: Vec<String> = Vec::new();
    for (index, size) in FILTER_SIZES.iter().enumerate() {
        let filter_size: f64 = size.parse()?;
        let applied = apply_operations(
            &project,
            root.path(),
            json!([
                {"op": "render.update", "target": {"id": active}, "set": {"resolution_x": RESOLUTION, "resolution_y": RESOLUTION, "samples": SAMPLES, "seed": 0, "filter_size": filter_size}}
            ]),
        )?;
        assert_eq!(applied["ok"], true, "{applied}");

        let frames = root.path().join(format!("frames_{index}"));
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
        let id = format!("film_{index}");
        let loaded = apply_operations(
            &project,
            root.path(),
            json!([{"op": "image.load", "id": id, "path": frames.join("frame_0001.exr").to_string_lossy()}]),
        )?;
        assert_eq!(loaded["ok"], true, "{loaded}");

        let actual = column_profile(&project, &id)?;
        let expected = floats(&oracle["profiles"][*size])?;
        let mut failures = Vec::new();
        for column in COLUMNS {
            let deviation = (actual[column] - expected[column]).abs();
            if deviation > TOLERANCE {
                failures.push(format!(
                    "column {column}: Potter {:.5} vs EEVEE {:.5} (deviation {deviation:.5})",
                    actual[column], expected[column]
                ));
            }
        }
        let table: Vec<String> = COLUMNS
            .map(|column| {
                format!(
                    "  column {column}: Potter {:.5} EEVEE {:.5}",
                    actual[column], expected[column]
                )
            })
            .collect();
        if !failures.is_empty() {
            report.push(format!(
                "filter {size}, tolerance {TOLERANCE}:\n{}\nwindow:\n{}",
                failures.join("\n"),
                table.join("\n")
            ));
        }
    }
    assert!(report.is_empty(), "{}", report.join("\n\n"));
    Ok(())
}
