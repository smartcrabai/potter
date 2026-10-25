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

use std::path::{Path, PathBuf};

use blender_checked::blender_executable;
use native_oracle::{TestResult, apply_operations, checked, import_blend, pot, read_scene};
use potter_core::image::{ImageData, ImageInterpolation, load_image_data, sample};
use potter_core::model::Image;
use serde_json::{Value, json};
use tempfile::tempdir;

const CAPTURE: &str =
    include_str!("fixtures/native_oracle/principled/single_bounce_thin_wall_means.json");
const VARIANTS: [&str; 5] = [
    "transmission_pane_thin_wall",
    "translucent_leaf_thin_wall",
    "transmission_pane_thin_wall_off",
    "translucent_leaf_thin_wall_light_no_diffuse",
    "translucent_leaf_thin_wall_light_no_transmission",
];
const POTTER_SEEDS: [u64; 4] = [0, 1, 2, 3];
const SAMPLES: u64 = 2048;

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

fn standard_error(samples: &[[f64; 3]], channel: usize) -> f64 {
    let mean = samples.iter().map(|sample| sample[channel]).sum::<f64>() / samples.len() as f64;
    let variance = samples
        .iter()
        .map(|sample| (sample[channel] - mean).powi(2))
        .sum::<f64>()
        / (samples.len() - 1) as f64;
    variance.sqrt() / (samples.len() as f64).sqrt()
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/native_oracle/principled")
        .join(format!("{name}.blend"))
}

#[test]
fn principled_thin_wall_reaches_transmission_and_translucent_surfaces() -> TestResult {
    let Some(blender) = blender_executable() else {
        return Ok(());
    };
    let expected: Value = serde_json::from_str(CAPTURE)?;
    let pane = &expected["variants"]["transmission_pane_thin_wall"];
    let pane_off = &expected["variants"]["transmission_pane_thin_wall_off"];
    let pane_mean = capture_vector(&pane["mean"])?;
    let pane_off_mean = capture_vector(&pane_off["mean"])?;
    let pane_stderr = capture_vector(&pane["stderr"])?;
    let pane_off_stderr = capture_vector(&pane_off["stderr"])?;
    assert!(
        (0..3).any(|channel| {
            (pane_mean[channel] - pane_off_mean[channel]).abs()
                > 8.0 * pane_stderr[channel].hypot(pane_off_stderr[channel])
        }),
        "Blender's thin wall pane and off control must differ beyond capture noise"
    );
    let leaf = &expected["variants"]["translucent_leaf_thin_wall"];
    let leaf_no_diffuse = &expected["variants"]["translucent_leaf_thin_wall_light_no_diffuse"];
    let leaf_no_transmission =
        &expected["variants"]["translucent_leaf_thin_wall_light_no_transmission"];
    let leaf_mean = capture_vector(&leaf["mean"])?;
    let leaf_stderr = capture_vector(&leaf["stderr"])?;
    let leaf_no_diffuse_mean = capture_vector(&leaf_no_diffuse["mean"])?;
    let leaf_no_diffuse_stderr = capture_vector(&leaf_no_diffuse["stderr"])?;
    let leaf_no_transmission_mean = capture_vector(&leaf_no_transmission["mean"])?;
    let leaf_no_transmission_stderr = capture_vector(&leaf_no_transmission["stderr"])?;
    assert!(
        (0..3).any(|channel| {
            (leaf_mean[channel] - leaf_no_diffuse_mean[channel]).abs()
                > 8.0 * leaf_stderr[channel].hypot(leaf_no_diffuse_stderr[channel])
        }),
        "Cycles must drop the backlit thin-wall leaf term when the light is not diffuse-visible"
    );
    assert!(
        (0..3).all(|channel| {
            (leaf_mean[channel] - leaf_no_transmission_mean[channel]).abs()
                <= 8.0 * leaf_stderr[channel].hypot(leaf_no_transmission_stderr[channel])
        }),
        "Cycles must retain the diffuse-class thin-wall leaf term when transmission visibility is off"
    );
    let root = tempdir()?;

    for variant in VARIANTS {
        let (project, imported) = import_blend(&blender, root.path(), variant, &fixture(variant))?;
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
                "set":{"resolution_x":16,"resolution_y":16,"resolution_percentage":100,"samples":SAMPLES,"seed":0,"max_bounces":8}
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
        eprintln!("thin-wall Potter CLI {variant}: {actual:?}");
        for channel in 0..3 {
            let potter_stderr = standard_error(&samples, channel);
            let tolerance = 4.0 * blender_stderr[channel].hypot(potter_stderr);
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
