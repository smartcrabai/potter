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

const SCRIPT: &str = include_str!("fixtures/native_oracle/principled/single_bounce_probe.py");
const CAPTURE: &str =
    include_str!("fixtures/native_oracle/principled/single_bounce_diffuse_roughness_means.json");
const VARIANTS: [(&str, &str); 2] = [
    ("diffuse_roughness_r050", "diffuse_roughness_r050"),
    ("diffuse_bsdf_r050", "diffuse_bsdf_r050"),
];

fn render_patch_mean(project: &Path, root: &Path, name: &str) -> TestResult<[f64; 3]> {
    let frames = root.join(format!("{name}_frames"));
    let mut render = pot();
    render
        .arg("render")
        .arg(project)
        .args(["--format", "exr", "--out"])
        .arg(&frames)
        .arg("--json");
    checked(&mut render, name)?;

    let exr = frames.join("frame_0001.exr");
    let loaded = apply_operations(
        project,
        root,
        json!([{"op":"image.load","id":name,"path":exr.to_string_lossy()}]),
    )?;
    assert_eq!(loaded["ok"], true, "{loaded}");
    let scene = read_scene(project)?;
    let image: Image = serde_json::from_value(scene["images"][name].clone())?;
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
fn principled_and_diffuse_roughness_reach_the_rendered_surface() -> TestResult {
    let Some(blender) = blender_executable() else {
        return Ok(());
    };
    let expected: Value = serde_json::from_str(CAPTURE)?;
    let root = tempdir()?;
    let script = write_script(root.path(), "single_bounce_probe.py", SCRIPT)?;
    for (variant, capture) in VARIANTS {
        let output_dir = root.path().join(format!("blender_output_{variant}"));
        let blend = root.path().join(format!("{variant}.blend"));
        run_blender_arguments(
            &blender,
            &script,
            None,
            &[
                output_dir.as_path(),
                Path::new("--save-only"),
                Path::new(variant),
                Path::new("2.4"),
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

        let actual = render_patch_mean(&project, root.path(), variant)?;
        let reference = &expected["variants"][capture];
        let mean = capture_vector(&reference["mean"])?;
        let stderr = capture_vector(&reference["stderr"])?;
        for channel in 0..3 {
            // The 16-seed capture's standard error scales by sqrt(16) for one Potter render; use
            // a four-sigma bound for that single-render Monte Carlo noise.
            let tolerance = 16.0 * stderr[channel];
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
