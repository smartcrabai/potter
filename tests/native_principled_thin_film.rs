#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/native_oracle.rs"]
#[expect(
    dead_code,
    reason = "the shared oracle helpers include support for unrelated tests"
)]
mod native_oracle;
#[path = "common/process.rs"]
mod process;

use std::path::Path;

use blender_checked::blender_executable;
use native_oracle::{
    TestResult, apply_operations, checked, import_blend, pot, read_scene, run_blender_arguments,
    write_script,
};
use potter_core::image::{ImageData, ImageInterpolation, load_image_data, sample};
use potter_core::model::Image;
use serde_json::{Value, json};
use tempfile::tempdir;

const SCRIPT: &str =
    include_str!("fixtures/native_oracle/principled/single_bounce_thin_film_probe.py");
const CAPTURE: &str =
    include_str!("fixtures/native_oracle/principled/single_bounce_thin_film_means.json");
const VARIANTS: [&str; 10] = [
    "dielectric_no_film",
    "dielectric_cutoff_0_1nm",
    "dielectric_120nm_ior133",
    "dielectric_450nm_ior170",
    "dielectric_900nm_ior133",
    "metallic_no_film",
    "metallic_120nm_ior133",
    "metallic_450nm_ior170",
    "metallic_900nm_ior133",
    "transmission_1_thin_wall_off",
];

fn render_patch_mean(project: &Path, root: &Path, name: &str, seed: u64) -> TestResult<[f64; 3]> {
    let id = format!("{name}_seed{seed}");
    let frames = root.join(format!("{id}_frames"));
    let mut render = pot();
    render
        .arg("render")
        .arg(project)
        .args(["--format", "exr", "--out"])
        .arg(&frames)
        .arg("--json");
    let output = checked(&mut render, name)?;
    let response: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(response["result"]["warnings"], json!([]), "{response}");

    let exr = frames.join("frame_0001.exr");
    let loaded = apply_operations(
        project,
        root,
        json!([{"op":"image.load","id":id,"path":exr.to_string_lossy()}]),
    )?;
    assert_eq!(loaded["ok"], true, "{loaded}");
    let scene = read_scene(project)?;
    let image: Image = serde_json::from_value(scene["images"][&id].clone())?;
    let data = load_image_data(&image, project, ImageInterpolation::Closest)?;
    Ok(patch_mean(&data))
}

fn patch_mean(data: &ImageData) -> [f64; 3] {
    let mut total = [0.0; 3];
    for y in 4..12 {
        for x in 4..12 {
            let pixel = sample(
                data,
                [((x as f64) + 0.5) / 16.0, ((y as f64) + 0.5) / 16.0],
                false,
            );
            for (sum, value) in total.iter_mut().zip(pixel.iter().take(3)) {
                *sum += value;
            }
        }
    }
    total.map(|value| value / 64.0)
}

fn capture_vector(value: &Value) -> TestResult<[f64; 3]> {
    let values = value.as_array().ok_or("capture vector is missing")?;
    let mut result = [0.0; 3];
    for (destination, source) in result.iter_mut().zip(values) {
        *destination = source.as_f64().ok_or("capture value is not a number")?;
    }
    Ok(result)
}

#[test]
fn principled_thin_film_reaches_the_path_traced_surface() -> TestResult {
    let Some(blender) = blender_executable() else {
        return Ok(());
    };
    let expected: Value = serde_json::from_str(CAPTURE)?;
    assert_eq!(
        expected["variants"]["dielectric_no_film"]["mean"],
        expected["variants"]["dielectric_cutoff_0_1nm"]["mean"],
        "the 0.1 nm cutoff control must match the no-film reference"
    );
    assert_eq!(
        expected["variants"]["dielectric_no_film"]["stderr"],
        expected["variants"]["dielectric_cutoff_0_1nm"]["stderr"],
        "the 0.1 nm cutoff control must match the no-film reference"
    );

    let root = tempdir()?;
    let script = write_script(root.path(), "single_bounce_thin_film_probe.py", SCRIPT)?;
    for variant in VARIANTS {
        let output_dir = root.path().join("blender_output");
        let blend = root.path().join(format!("{variant}.blend"));
        run_blender_arguments(
            &blender,
            &script,
            None,
            &[
                output_dir.as_path(),
                Path::new("--save-only"),
                Path::new(variant),
                blend.as_path(),
            ],
        )?;
        let (project, imported) = import_blend(&blender, root.path(), variant, &blend)?;
        assert_eq!(imported["ok"], true, "{imported}");
        let scene = read_scene(&project)?;
        let active = scene["active_scene"]
            .as_str()
            .ok_or("active scene is missing")?;
        let updated = apply_operations(
            &project,
            root.path(),
            json!([{
                "op":"render.update",
                "target":{"id":active},
                "set":{"resolution_x":16,"resolution_y":16,"resolution_percentage":100,"samples":4096,"seed":0,"max_bounces":4}
            }]),
        )?;
        assert_eq!(updated["ok"], true, "{updated}");

        let reference = &expected["variants"][variant];
        let mean = capture_vector(&reference["mean"])?;
        let blender_stderr = capture_vector(&reference["stderr"])?;
        let mut samples = Vec::with_capacity(4);
        for seed in 0..4 {
            let updated = apply_operations(
                &project,
                root.path(),
                json!([{
                    "op":"render.update",
                    "target":{"id":active},
                    "set":{"seed":seed}
                }]),
            )?;
            assert_eq!(updated["ok"], true, "{updated}");
            samples.push(render_patch_mean(&project, root.path(), variant, seed)?);
        }
        let actual: [f64; 3] = std::array::from_fn(|channel| {
            samples.iter().map(|sample| sample[channel]).sum::<f64>() / samples.len() as f64
        });
        let potter_stderr: [f64; 3] = std::array::from_fn(|channel| {
            let variance = samples
                .iter()
                .map(|sample| (sample[channel] - actual[channel]).powi(2))
                .sum::<f64>()
                / (samples.len() - 1) as f64;
            variance.sqrt() / (samples.len() as f64).sqrt()
        });
        eprintln!("thin-film Potter CLI {variant}: {actual:?}, stderr {potter_stderr:?}");
        for channel in 0..3 {
            // Compare the independent four-seed Potter and 16-seed Cycles means at four combined
            // standard errors, measured from each renderer's own Monte Carlo runs.
            let tolerance = 4.0 * blender_stderr[channel].hypot(potter_stderr[channel]);
            assert!(
                (actual[channel] - mean[channel]).abs() <= tolerance,
                "{variant} channel {channel}: Potter {} vs Blender {} (tolerance {tolerance})",
                actual[channel],
                mean[channel]
            );
        }
    }
    Ok(())
}
