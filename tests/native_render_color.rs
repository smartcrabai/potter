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

use std::{error::Error, fs, io::BufReader, path::Path};

use native_oracle::{
    TestResult, apply_operations, build_oracle, import_blend, pot, read_scene,
    run_blender_arguments, vec4, write_script,
};
use potter_core::image::{ImageInterpolation, load_image_data, sample};
use potter_core::model::Image;
use serde_json::{Value, json};
use tempfile::tempdir;

/// Render-to-stored-value tolerance (linear units). Path-tracing noise on a constant emitter is
/// far smaller; the sRGB-encoded alternative differs by more than 0.25.
const PIXEL_TOLERANCE: f64 = 0.01;

/// Blender 5.2.2 script for the AgX end-to-end check (see `agx_emitter_preview.py`).
const AGX_SCRIPT: &str = include_str!("fixtures/native_oracle/agx_emitter_preview.py");

/// Centre RGB of the 8-bit PNG at `path`.
fn png_centre_rgb(path: &Path) -> Result<[u8; 3], Box<dyn Error>> {
    let mut reader = png::Decoder::new(BufReader::new(fs::File::open(path)?)).read_info()?;
    let mut pixels = vec![
        0;
        reader
            .output_buffer_size()
            .ok_or("PNG output buffer size is unknown")?
    ];
    let info = reader.next_frame(&mut pixels)?;
    let width = info.width as usize;
    let channels = info.color_type.samples();
    let index = ((info.height as usize / 2) * width + width / 2) * channels;
    Ok([pixels[index], pixels[index + 1], pixels[index + 2]])
}

/// Potter's beauty preview of an imported AgX scene matches Blender's AgX render of the same scene
/// to one 8-bit step. The emitter is a constant 0.18 grey, so the centre pixel is the view
/// transform of that grey: the import of `view_transform`, the AgX port, and the sRGB encoding all
/// have to agree.
#[test]
fn agx_beauty_preview_centre_pixel_matches_blender_render_within_one_byte() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let (oracle, _) = build_oracle(&blender, root.path())?;
    let agx_blend = root.path().join("agx_emitter.blend");
    let blender_png = root.path().join("blender_agx.png");
    let script = write_script(root.path(), "agx_emitter_preview.py", AGX_SCRIPT)?;
    run_blender_arguments(
        &blender,
        &script,
        Some(oracle.as_path()),
        &[agx_blend.as_path(), blender_png.as_path()],
    )?;

    let (project, _) = import_blend(&blender, root.path(), "project", &agx_blend)?;
    let scene = read_scene(&project)?;
    let active = scene["active_scene"]
        .as_str()
        .ok_or("active scene is missing")?
        .to_owned();
    assert_eq!(
        scene["scenes"][active.as_str()]["color_management"]["view_transform"],
        "ag_x",
        "the imported scene must carry Blender's AgX view transform"
    );

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

    let previews = root.path().join("previews");
    fs::create_dir_all(&previews)?;
    let preview = pot()
        .arg("preview")
        .arg(&project)
        .args([
            "--camera",
            "camera_main",
            "--mode",
            "beauty",
            "--size",
            "16",
            "--out",
        ])
        .arg(&previews)
        .arg("--json")
        .output()?;
    assert!(
        preview.status.success(),
        "pot preview failed: {}",
        String::from_utf8_lossy(&preview.stdout)
    );
    let potter_png = fs::read_dir(&previews)?
        .next()
        .ok_or("pot preview wrote no image")??
        .path();

    let potter = png_centre_rgb(&potter_png)?;
    let blender_rgb = png_centre_rgb(&blender_png)?;
    assert!(
        blender_rgb.iter().all(|&channel| channel > 0),
        "Blender AgX render centre is black: {blender_rgb:?}"
    );
    for channel in 0..3 {
        assert!(
            potter[channel].abs_diff(blender_rgb[channel]) <= 1,
            "channel {channel}: Potter AgX preview {potter:?} differs from Blender AgX render {blender_rgb:?} by more than one 8-bit step"
        );
    }
    Ok(())
}

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
