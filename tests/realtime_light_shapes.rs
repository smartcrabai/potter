//! Camera-visible light shapes in the realtime Beauty image, checked against committed Blender 5.2.2 EEVEE captures.
//!
//! `fixtures/native_oracle/eevee_light_shapes.py` saves a square area light and a radius-1 point light, both with
//! `visible_camera` on, on a black world. The committed macOS EEVEE capture draws each emitter at its radiance
//! (`eevee_light_shape_display`), which is the path tracer's emitter radiance for the same light. The test imports the
//! saved `.blend` and renders it with `pot render --engine realtime`, then compares the mean red value of a 7 x 7
//! window at each light's centre and the background with the captured EEVEE pixels.
//!
//! The tolerance is 0.005: EEVEE's own window means lie within 0.0005 of the emitter formulas (`P / (π A)` for the
//! area light and `P / (4 π² r²)` for the sphere), so the bound is ten times that gap, far below either light's value.
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
    TestResult, apply_operations, import_blend, pot, read_scene, run_blender_arguments,
    write_script,
};
use potter_core::image::{ImageInterpolation, load_image_data, sample};
use potter_core::model::Image;
use serde_json::{Value, json};
use tempfile::tempdir;

/// The scene generator (`eevee_light_shapes.py`).
const SCRIPT: &str = include_str!("fixtures/native_oracle/eevee_light_shapes.py");

/// Render settings shared with `eevee_light_shapes.py`.
const RESOLUTION: u32 = 64;
const WINDOW: usize = 3;
/// Linear tolerance on a window mean against EEVEE.
const TOLERANCE: f64 = 0.005;

/// Blender's bottom-up RGB f32le capture for the area and point light scene.
const EEVEE_AREA_POINT: &[u8] =
    include_bytes!("fixtures/native_oracle/eevee_macos/area_point.f32le");

/// The mean red value of the `(2 * half + 1)` square window centred on `(column, row)` in the image `id`, read with
/// Potter's own EXR loader at the texel centres.
fn window_mean(
    project: &Path,
    id: &str,
    column: usize,
    row: usize,
    half: usize,
) -> Result<f64, Box<dyn Error>> {
    let scene = read_scene(project)?;
    let image: Image = serde_json::from_value(scene["images"][id].clone())?;
    let data = load_image_data(&image, project, ImageInterpolation::Closest)?;
    let texels = f64::from(RESOLUTION);
    let mut total = 0.0;
    let mut count = 0.0;
    for r in row - half..=row + half {
        for c in column - half..=column + half {
            let uv = [(c as f64 + 0.5) / texels, (r as f64 + 0.5) / texels];
            total += sample(&data, uv, false)[0];
            count += 1.0;
        }
    }
    Ok(total / count)
}

/// The mean red value of an RGB f32le capture window; fixture rows are bottom-up, unlike Potter's top-down rows.
fn oracle_window_mean(capture: &[u8], column: usize, row: usize, half: usize) -> f64 {
    let resolution = RESOLUTION as usize;
    assert_eq!(capture.len(), resolution * resolution * 3 * 4);
    let mut total = 0.0;
    for top_down_row in row - half..=row + half {
        let bottom_up_row = resolution - 1 - top_down_row;
        for sample_column in column - half..=column + half {
            let offset = (bottom_up_row * resolution + sample_column) * 3 * 4;
            let red = f32::from_le_bytes([
                capture[offset],
                capture[offset + 1],
                capture[offset + 2],
                capture[offset + 3],
            ]);
            total += f64::from(red);
        }
    }
    total / ((2 * half + 1) * (2 * half + 1)) as f64
}
#[test]
fn realtime_camera_visible_lights_match_eevee_emitters() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "eevee_light_shapes.py", SCRIPT)?;
    let blend = root.path().join("light_shapes.blend");
    run_blender_arguments(
        &blender,
        &script,
        None,
        &[blend.as_path(), Path::new("-"), Path::new("-")],
    )?;

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
            {"op": "render.update", "target": {"id": active}, "set": {"resolution_x": RESOLUTION, "resolution_y": RESOLUTION, "samples": 4, "seed": 0}}
        ]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");

    let frames = root.path().join("frames");
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
    let loaded = apply_operations(
        &project,
        root.path(),
        json!([{"op": "image.load", "id": "render", "path": frames.join("frame_0001.exr").to_string_lossy()}]),
    )?;
    assert_eq!(loaded["ok"], true, "{loaded}");

    let row = RESOLUTION as usize / 2;
    let checks = [
        (
            "area light",
            window_mean(&project, "render", 20, row, WINDOW)?,
            oracle_window_mean(EEVEE_AREA_POINT, 20, row, WINDOW),
        ),
        (
            "point light",
            window_mean(&project, "render", 44, row, WINDOW)?,
            oracle_window_mean(EEVEE_AREA_POINT, 44, row, WINDOW),
        ),
        (
            "background",
            window_mean(&project, "render", 16, 16, WINDOW)?,
            oracle_window_mean(EEVEE_AREA_POINT, 16, 16, WINDOW),
        ),
    ];
    for (name, actual, expected) in checks {
        assert!(
            (actual - expected).abs() <= TOLERANCE,
            "{name}: Potter {actual} differs from EEVEE {expected}"
        );
    }
    Ok(())
}
fn render_overlap_case(root: &Path, name: &str, lights: &[Value]) -> TestResult<[f64; 3]> {
    let project = root.join(name);
    let initialized = pot().arg("init").arg(&project).arg("--json").output()?;
    assert!(
        initialized.status.success(),
        "scene initialization failed: {}",
        String::from_utf8_lossy(&initialized.stdout)
    );
    let mut operations = vec![
        json!({"op":"world.create","id":"world_main","color":[0.0,0.0,0.0],"strength":0.0}),
        json!({"op":"camera.create","id":"camera_main","transform":{"translation":[0.0,0.0,10.0]},"projection":"orthographic","ortho_scale":8.0}),
    ];
    operations.extend(lights.iter().cloned());
    operations.extend([
        json!({"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"camera_main","world":"world_main"}}),
        json!({"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_x":RESOLUTION,"resolution_y":RESOLUTION,"samples":1,"seed":0,"max_bounces":0}}),
    ]);
    let applied = apply_operations(&project, root, json!(operations))?;
    assert_eq!(applied["ok"], true, "{applied}");

    let frames = root.join(format!("frames_{name}"));
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
    let loaded = apply_operations(
        &project,
        root,
        json!([{"op":"image.load","id":"render","path":frames.join("frame_0001.exr").to_string_lossy()}]),
    )?;
    assert_eq!(loaded["ok"], true, "{loaded}");
    let scene = read_scene(&project)?;
    let image: Image = serde_json::from_value(scene["images"]["render"].clone())?;
    let data = load_image_data(&image, &project, ImageInterpolation::Closest)?;
    let center = (f64::from(RESOLUTION / 2) + 0.5) / f64::from(RESOLUTION);
    let color = sample(&data, [center, center], false);
    Ok([color[0], color[1], color[2]])
}

fn point_light_operations(id: &str, color: [f64; 3], energy: f64, depth: f64) -> [Value; 2] {
    [
        json!({
            "op":"light.create",
            "id":id,
            "light_type":"point",
            "color":color,
            "energy":energy,
            "radius":0.5,
            "transform":{"translation":[0.0,0.0,depth]}
        }),
        json!({"op":"node.update","target":{"id":id},"set":{"ray_visibility":{"camera":true}}}),
    ]
}

/// EEVEE adds every camera-visible shape independently; an overlapping shape at another depth must not occlude it.
#[test]
fn overlapping_camera_visible_lights_add_in_linear_space() -> TestResult {
    let root = tempdir()?;
    let red = point_light_operations("red", [1.0, 0.0, 0.0], 1.0, 5.0);
    let blue = point_light_operations("blue", [0.0, 0.0, 1.0], 2.0, 0.0);
    let red_only = render_overlap_case(root.path(), "red_only", &red)?;
    let blue_only = render_overlap_case(root.path(), "blue_only", &blue)?;
    let both_lights = [
        red[0].clone(),
        red[1].clone(),
        blue[0].clone(),
        blue[1].clone(),
    ];
    let both = render_overlap_case(root.path(), "both", &both_lights)?;
    for channel in 0..3 {
        let expected = red_only[channel] + blue_only[channel];
        assert!(
            (both[channel] - expected).abs() <= 1.0e-5,
            "channel {channel}: overlapping lights {both:?}, isolated sum {expected} (red {red_only:?}, blue {blue_only:?})"
        );
    }
    assert!(red_only[0] > 0.0 && blue_only[2] > 0.0);
    Ok(())
}
