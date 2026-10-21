//! A textured simple clip material, exported to `.blend`, reopened in Blender 5.2.2 and rendered with Cycles, gives
//! the same binary cutout as Potter's own render of the same scene.
//!
//! The review's example: the base alpha is 0.5, the cutoff 0.37, and the texel alpha 0.6. The clip compares the
//! texel alpha times the base alpha, 0.3, with the cutoff, so the pixel is hidden. A second case, base alpha 0.9,
//! gives 0.54 and is visible. That control shows the plane covers the sample point, so a hidden result is not an
//! empty frame.
//!
//! A `.blend` that dropped the base alpha factor would compare 0.6 with 0.37 and show the hidden case, so the first
//! case catches that export bug in Blender.
//!
//! Each case: Potter imports the oracle scene, authors the clip material, renders the centre pixel, exports `.blend`,
//! Blender renders the exported file with Cycles (one-pixel box filter, as the Potter render uses), and the test
//! reads both centre alphas. It then reimports the `.blend` and checks the material reads back as clip with the same
//! threshold.

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

use std::path::Path;

use native_oracle::{
    FLOAT_TOLERANCE, TestResult, apply_operations, assert_close, build_oracle, export,
    import_blend, losses, node_id, number, pot, read_scene, run_blender_arguments, write_script,
};
use potter_core::image::{ImageInterpolation, load_image_data, sample};
use potter_core::model::Image;
use serde_json::{Value, json};
use tempfile::tempdir;

const TEXEL_ALPHA: f64 = 0.6;
const CUTOFF: f64 = 0.37;
const RENDER_SAMPLES: u32 = 16;
/// Both renderers are binary here, so the centre alpha is 0 or 1 up to float storage.
const ALPHA_TOLERANCE: f64 = 1.0e-3;

const BLENDER_RENDER_SCRIPT: &str = r#"
import bpy
import sys

blend_path, exr_path = sys.argv[sys.argv.index("--") + 1:][:2]
bpy.ops.wm.open_mainfile(filepath=blend_path)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = 16
scene.cycles.device = "CPU"
scene.cycles.pixel_filter_type = "BOX"
scene.cycles.filter_width = 1.0
scene.render.film_transparent = True
scene.render.resolution_x = scene.render.resolution_y = 4
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.render.image_settings.color_mode = "RGBA"
scene.render.filepath = exr_path
bpy.ops.render.render(write_still=True)
"#;

/// The centre alpha of a Potter project's image, sampled as the texture alpha tests do.
fn center_alpha(project: &Path, image_id: &str) -> Result<f64, Box<dyn std::error::Error>> {
    let scene = read_scene(project)?;
    let image: Image = serde_json::from_value(scene["images"][image_id].clone())?;
    let data = load_image_data(&image, project, ImageInterpolation::Closest)?;
    Ok(sample(&data, [0.5, 0.5], false)[3])
}

/// Authors the textured simple clip on the oracle plane, with a camera and render settings that see it.
fn author_clip_scene(
    project: &Path,
    root: &Path,
    plane: &str,
    active_scene: &str,
    base_alpha: f64,
) -> TestResult {
    let authored = apply_operations(
        project,
        root,
        json!([
            {"op": "image.create", "id": "clip_texture", "name": "Clip Texture",
             "width": 1, "height": 1, "colorspace": "linear",
             "fill_color": [1.0, 1.0, 1.0, TEXEL_ALPHA]},
            {"op": "material.create", "id": "clip_material", "name": "Textured Clip",
             "base_color": [1.0, 1.0, 1.0, base_alpha], "alpha_mode": "clip",
             "alpha_threshold": CUTOFF,
             "base_color_texture": {"image": "clip_texture", "uv_layer": "Base", "interpolation": "closest"}},
            {"op": "node.update", "target": {"id": plane}, "set": {"materials": ["clip_material"]}},
            {"op": "camera.create", "id": "clip_camera",
             "transform": {"translation": [8.0, 0.0, 6.0]},
             "projection": "orthographic", "ortho_scale": 2.0},
            {"op": "scene.update", "target": {"id": active_scene}, "set": {"camera": "clip_camera"}},
            {"op": "render.update", "target": {"id": active_scene},
             "set": {"resolution_x": 4, "resolution_y": 4, "resolution_percentage": 100,
                     "samples": RENDER_SAMPLES, "seed": 0, "max_bounces": 1,
                     "film_transparent": true, "engine": "path"}}
        ]),
    )?;
    assert_eq!(authored["ok"], true, "clip scene setup failed: {authored}");
    Ok(())
}

/// Potter's own centre alpha: renders the project to EXR and loads it back.
fn potter_center_alpha(project: &Path, root: &Path, name: &str) -> TestResult<f64> {
    let exr_dir = root.join(format!("{name}_potter"));
    let mut render = pot();
    render
        .arg("render")
        .arg(project)
        .args(["--format", "exr", "--out"])
        .arg(&exr_dir)
        .arg("--json");
    let output = native_oracle::checked(&mut render, "Potter clip render")?;
    let report: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(report["ok"], true, "Potter render failed: {report}");
    let exr = exr_dir.join("frame_0001.exr");
    let loaded = apply_operations(
        project,
        root,
        json!([{"op": "image.load", "id": format!("{name}_potter_render"), "path": exr.to_string_lossy()}]),
    )?;
    assert_eq!(loaded["ok"], true, "could not load Potter's EXR: {loaded}");
    center_alpha(project, &format!("{name}_potter_render"))
}

/// Cycles' centre alpha of the exported `.blend`, loaded through Potter's image loader.
fn blender_center_alpha(
    blender: &Path,
    project: &Path,
    root: &Path,
    name: &str,
    blend: &Path,
) -> TestResult<f64> {
    let script = write_script(root, "render_exported_clip.py", BLENDER_RENDER_SCRIPT)?;
    let exr = root.join(format!("{name}_blender.exr"));
    let arguments: [&Path; 2] = [blend, &exr];
    run_blender_arguments(blender, &script, None, &arguments)?;
    let loaded = apply_operations(
        project,
        root,
        json!([{"op": "image.load", "id": format!("{name}_blender_render"), "path": exr.to_string_lossy()}]),
    )?;
    assert_eq!(loaded["ok"], true, "could not load Blender's EXR: {loaded}");
    center_alpha(project, &format!("{name}_blender_render"))
}

fn run_case(
    blender: &Path,
    oracle: &Path,
    root: &Path,
    name: &str,
    base_alpha: f64,
    expected: f64,
) -> TestResult {
    let (project, report) = import_blend(blender, root, name, oracle)?;
    assert!(
        losses(&report).is_empty(),
        "oracle import lost data: {report}"
    );
    let scene = read_scene(&project)?;
    let plane = node_id(&scene, "TexturePlane")?;
    let active_scene = scene["active_scene"]
        .as_str()
        .ok_or("active scene is missing")?
        .to_owned();
    author_clip_scene(&project, root, &plane, &active_scene, base_alpha)?;

    let potter = potter_center_alpha(&project, root, name)?;
    assert!(
        (potter - expected).abs() <= ALPHA_TOLERANCE,
        "{name}: Potter centre alpha {potter} differs from the expected {expected}"
    );

    let exported = root.join(format!("{name}.blend"));
    let export_report = export(blender, &project, "blend", &exported)?;
    assert!(
        losses(&export_report).is_empty(),
        "clip export lost data: {export_report}"
    );
    let cycles = blender_center_alpha(blender, &project, root, name, &exported)?;
    assert!(
        (cycles - expected).abs() <= ALPHA_TOLERANCE,
        "{name}: Cycles centre alpha {cycles} differs from the expected {expected}"
    );

    let (reimported, reimport_report) =
        import_blend(blender, root, &format!("{name}_reimported"), &exported)?;
    assert!(
        losses(&reimport_report).is_empty(),
        "clip reimport lost data: {reimport_report}"
    );
    let materials = read_scene(&reimported)?["materials"].clone();
    let clips: Vec<&Value> = materials
        .as_object()
        .ok_or("materials are missing")?
        .values()
        .filter(|material| material["alpha_mode"] == "clip")
        .collect();
    assert_eq!(
        clips.len(),
        1,
        "{name}: the exported clip did not reopen as clip"
    );
    assert_close(
        number(&clips[0]["alpha_threshold"])?,
        CUTOFF,
        FLOAT_TOLERANCE,
        "reopened clip threshold",
    );
    Ok(())
}

#[test]
fn textured_simple_clip_round_trip_agrees_with_cycles_cutout() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let (oracle, _) = build_oracle(&blender, root.path())?;
    // The review's example: 0.5 * 0.6 = 0.3 < 0.37, hidden. The control: 0.9 * 0.6 = 0.54 >= 0.37, visible.
    run_case(&blender, &oracle, root.path(), "hidden", 0.5, 0.0)?;
    run_case(&blender, &oracle, root.path(), "visible", 0.9, 1.0)?;
    Ok(())
}
