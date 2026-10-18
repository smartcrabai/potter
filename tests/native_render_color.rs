//! PR9 render of a stored color layer, checked against the explicit oracle value.
//!
//! The oracle's `Emitter` is a 2x2 quad at z = 3 whose material is emission from the Vertex Color
//! node reading the layer `Ink` (constant, linear `(0.0625, 0.25, 0.5, 1.0)`, strength 1). Shading
//! reads colors by layer name at hit time, so the centre pixel of an orthographic render equals
//! `Ink` to within the render's sampling. The pixel is read back through Potter's own image
//! loader after `image.load` of the rendered EXR. The bound is 0.01: far below the gap between a
//! linear value and its sRGB encoding (0.25 becomes 0.5377), so a linear/sRGB mix-up fails.

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

use std::{error::Error, fs, path::Path};

use native_oracle::{
    TestResult, apply_operations, build_oracle, import_blend, pot, read_scene, vec4,
};
use potter_core::image::{ImageInterpolation, load_image_data, sample};
use potter_core::model::Image;
use serde_json::{Value, json};
use tempfile::tempdir;

/// Render-to-stored-value tolerance (linear units). Path-tracing noise on a constant emitter is
/// far smaller; the sRGB-encoded alternative differs by more than 0.25.
const PIXEL_TOLERANCE: f64 = 0.01;

fn pixel_at_center(project: &Path, image_id: &str) -> Result<[f64; 4], Box<dyn Error>> {
    let scene: Value = read_scene(project)?;
    let image: Image = serde_json::from_value(scene["images"][image_id].clone())?;
    let data = load_image_data(&image, project, ImageInterpolation::Closest)?;
    Ok(sample(&data, [0.5, 0.5], 0))
}

#[test]
fn emitter_centre_pixel_equals_its_stored_vertex_color_layer() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let (blend, expected) = build_oracle(&blender, root.path())?;
    let (project, _) = import_blend(&blender, root.path(), "project", &blend)?;
    let scene = read_scene(&project)?;
    let active = scene["active_scene"]
        .as_str()
        .ok_or("active scene is missing")?
        .to_owned();

    let applied = apply_operations(
        &project,
        root.path(),
        json!([
            {"op": "camera.create", "id": "camera_main", "transform": {"translation": [0.0, 0.0, 6.0]}, "projection": "orthographic", "ortho_scale": 4.0},
            {"op": "scene.update", "target": {"id": active}, "set": {"camera": "camera_main"}},
            {"op": "render.update", "target": {"id": active}, "set": {"resolution_x": 16, "resolution_y": 16, "samples": 8, "seed": 0, "max_bounces": 0}}
        ]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");

    let frames = root.path().join("frames");
    let rendered = pot()
        .arg("render")
        .arg(&project)
        .args(["--format", "exr", "--out"])
        .arg(&frames)
        .arg("--json")
        .output()?;
    assert!(
        rendered.status.success(),
        "render failed: {}",
        String::from_utf8_lossy(&rendered.stdout)
    );
    let stdout = String::from_utf8_lossy(&rendered.stdout);
    assert!(
        !stdout.contains("color_layer_missing"),
        "the emitter's layer exists, so no missing-layer warning may appear: {stdout}"
    );
    let exr = frames.join("frame_0001.exr");
    assert!(fs::metadata(&exr)?.len() > 0, "EXR frame is empty");

    let loaded = apply_operations(
        &project,
        root.path(),
        json!([{"op": "image.load", "id": "render", "path": exr.to_string_lossy()}]),
    )?;
    assert_eq!(loaded["ok"], true, "{loaded}");

    let actual = pixel_at_center(&project, "render")?;
    let wanted = vec4(&expected["emitter"]["value"])?;
    for (channel, (actual, wanted)) in actual.iter().zip(wanted).enumerate() {
        assert!(
            (actual - wanted).abs() <= PIXEL_TOLERANCE,
            "channel {channel}: rendered {actual} differs from stored Ink {wanted}"
        );
    }
    Ok(())
}
