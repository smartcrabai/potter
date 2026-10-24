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
    include_str!("fixtures/native_oracle/principled/single_bounce_anisotropic_probe.py");
const CAPTURE: &str =
    include_str!("fixtures/native_oracle/principled/single_bounce_anisotropic_means.json");
const VARIANTS: [&str; 12] = [
    "uv_sphere_tangent_emission",
    "a09_r00_uv_sphere",
    "a05_r00_radial",
    "a05_r00_uv_map",
    "a05_r25_radial",
    "a05_r25_uv_map",
    "a09_r00_radial",
    "a09_r00_uv_map",
    "a09_r25_radial",
    "a09_r25_uv_map",
    "isotropic",
    "a09_r00_unlinked",
];
const POTTER_SEEDS: [u64; 4] = [0, 1, 2, 3];

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
    Ok(patch_mean(&data, name))
}

fn patch_mean(data: &ImageData, name: &str) -> [f64; 3] {
    let mut total = [0.0; 3];
    let x_start = if name == "uv_sphere_tangent_emission" {
        8
    } else {
        4
    };
    let x_count = if name == "uv_sphere_tangent_emission" {
        4
    } else {
        8
    };
    for y in 4..12 {
        for x in x_start..x_start + x_count {
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
    total.map(|value| value / (x_count * 8) as f64)
}

fn capture_vector(value: &Value) -> TestResult<[f64; 3]> {
    let values = value.as_array().ok_or("capture vector is missing")?;
    let mut result = [0.0; 3];
    for (destination, source) in result.iter_mut().zip(values) {
        *destination = source.as_f64().ok_or("capture value is not a number")?;
    }
    Ok(result)
}

fn standard_error(samples: &[[f64; 3]], channel: usize) -> f64 {
    let mean = samples.iter().map(|sample| sample[channel]).sum::<f64>() / samples.len() as f64;
    let variance = samples
        .iter()
        .map(|sample| (sample[channel] - mean).powi(2))
        .sum::<f64>()
        / (samples.len() - 1) as f64;
    variance.sqrt() / (samples.len() as f64).sqrt()
}

#[test]
fn principled_anisotropy_reaches_the_path_traced_surface() -> TestResult {
    let Some(blender) = blender_executable() else {
        return Ok(());
    };
    let expected: Value = serde_json::from_str(CAPTURE)?;
    let isotropic = &expected["variants"]["isotropic"];
    let radial = &expected["variants"]["a09_r00_radial"];
    let unlinked = &expected["variants"]["a09_r00_unlinked"];
    let isotropic_mean = capture_vector(&isotropic["mean"])?;
    let isotropic_stderr = capture_vector(&isotropic["stderr"])?;
    let radial_mean = capture_vector(&radial["mean"])?;
    let radial_stderr = capture_vector(&radial["stderr"])?;
    let unlinked_mean = capture_vector(&unlinked["mean"])?;
    let unlinked_stderr = capture_vector(&unlinked["stderr"])?;
    for channel in 0..3 {
        let radial_tolerance = 4.0 * radial_stderr[channel].hypot(unlinked_stderr[channel]);
        assert!(
            (unlinked_mean[channel] - radial_mean[channel]).abs() <= radial_tolerance,
            "unlinked Tangent channel {channel}: Blender {} vs Radial Tangent axis Z {} (tolerance {radial_tolerance})",
            unlinked_mean[channel],
            radial_mean[channel]
        );
        let isotropic_tolerance = 4.0 * isotropic_stderr[channel].hypot(unlinked_stderr[channel]);
        assert!(
            (unlinked_mean[channel] - isotropic_mean[channel]).abs() > isotropic_tolerance,
            "unlinked Tangent channel {channel}: Blender {} should differ from isotropic {} beyond Monte Carlo tolerance {isotropic_tolerance}",
            unlinked_mean[channel],
            isotropic_mean[channel]
        );
    }

    let root = tempdir()?;
    let script = write_script(root.path(), "single_bounce_anisotropic_probe.py", SCRIPT)?;
    let mut potter_samples = std::collections::HashMap::new();
    let mut potter_stderr = std::collections::HashMap::new();
    for variant in VARIANTS {
        let blend = root.path().join(format!("{variant}.blend"));
        run_blender_arguments(
            &blender,
            &script,
            None,
            &[
                root.path(),
                Path::new("--save-only"),
                Path::new(variant),
                blend.as_path(),
            ],
        )?;
        let (project, imported) = import_blend(&blender, root.path(), variant, &blend)?;
        assert_eq!(imported["ok"], true, "{imported}");
        assert_eq!(imported["result"]["warnings"], json!([]), "{imported}");
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
        let mut samples = Vec::with_capacity(POTTER_SEEDS.len());
        for seed in POTTER_SEEDS {
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
        let actual_stderr: [f64; 3] =
            std::array::from_fn(|channel| standard_error(&samples, channel));
        eprintln!("anisotropy Potter CLI {variant}: {actual:?}, stderr {actual_stderr:?}");
        for channel in 0..3 {
            // Four independent Potter seeds and sixteen Cycles seeds estimate independent MC means;
            // the four-sigma bound uses both measured standard errors, not an image-space constant.
            let tolerance = 4.0 * blender_stderr[channel].hypot(actual_stderr[channel]);
            assert!(
                (actual[channel] - mean[channel]).abs() <= tolerance,
                "{variant} channel {channel}: Potter {} vs Blender {} (tolerance {tolerance})",
                actual[channel],
                mean[channel]
            );
        }
        potter_stderr.insert(variant, actual_stderr);
        potter_samples.insert(variant, actual);
    }

    let actual_iso = potter_samples["isotropic"];
    let actual_radial = potter_samples["a09_r00_radial"];
    let actual_unlinked = potter_samples["a09_r00_unlinked"];
    let stderr_iso = potter_stderr["isotropic"];
    let stderr_radial = potter_stderr["a09_r00_radial"];
    let stderr_unlinked = potter_stderr["a09_r00_unlinked"];
    for channel in 0..3 {
        let radial_tolerance = 4.0 * stderr_radial[channel].hypot(stderr_unlinked[channel]);
        assert!(
            (actual_unlinked[channel] - actual_radial[channel]).abs() <= radial_tolerance,
            "Potter unlinked Tangent channel {channel}: {} vs Radial Tangent axis Z {} (tolerance {radial_tolerance})",
            actual_unlinked[channel],
            actual_radial[channel]
        );
        let isotropic_tolerance = 4.0 * stderr_iso[channel].hypot(stderr_unlinked[channel]);
        assert!(
            (actual_iso[channel] - actual_unlinked[channel]).abs() > isotropic_tolerance,
            "Potter unlinked Tangent channel {channel}: {} should differ from isotropic {} beyond Monte Carlo tolerance {isotropic_tolerance}",
            actual_unlinked[channel],
            actual_iso[channel]
        );
    }
    Ok(())
}
