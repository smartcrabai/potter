//! Realtime Beauty's direct light through double-sided glass viewed from its back, exercised through the CLI render path.

#[path = "common/native_oracle.rs"]
#[expect(
    dead_code,
    reason = "this test binary uses a subset of the shared CLI helpers"
)]
mod native_oracle;

use native_oracle::{TestResult, apply_operations, pot, pot_json, read_scene};
use potter_core::{
    image::{ImageInterpolation, load_image_data, sample},
    model::Image,
};
use serde_json::{Value, json};
use tempfile::tempdir;

fn assert_backface_glass_lit(light: Value, case: &str) -> TestResult {
    let root = tempdir()?;
    let project = root.path().join("project");
    pot_json(|command| {
        command.arg("init").arg(&project);
    })?;

    let applied = apply_operations(
        &project,
        root.path(),
        json!([
            {"op": "material.create", "id": "glass", "base_color": [1.0, 1.0, 1.0, 1.0],
             "transmission": 1.0, "ior": 1.45, "roughness": 0.0, "double_sided": true},
            {"op": "node.create", "id": "glass_plane", "kind": "plane", "params": {"size": 2.0},
             "transform": {"rotation": [1.0, 0.0, 0.0, 0.0]}, "material": "glass"},
            light,
            {"op": "camera.create", "id": "camera_back", "projection": "orthographic", "ortho_scale": 2.2,
             "transform": {"translation": [0.0, 0.0, 4.0]}},
            {"op": "scene.update", "target": {"id": "scene_main"}, "set": {"camera": "camera_back"}},
            {"op": "render.update", "target": {"id": "scene_main"},
             "set": {"resolution_x": 32, "resolution_y": 32, "samples": 1, "seed": 0}}
        ]),
    )?;
    assert_eq!(applied["ok"], true, "glass scene apply failed: {applied}");

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
        "realtime render for {case} failed: {}",
        String::from_utf8_lossy(&rendered.stdout)
    );

    let loaded = apply_operations(
        &project,
        root.path(),
        json!([{"op": "image.load", "id": "render", "path": frames.join("frame_0001.exr").to_string_lossy()}]),
    )?;
    assert_eq!(loaded["ok"], true, "render load failed: {loaded}");
    let image: Image = serde_json::from_value(read_scene(&project)?["images"]["render"].clone())?;
    let data = load_image_data(&image, &project, ImageInterpolation::Closest)?;
    let pixel = sample(&data, [0.5, 0.5], false);
    assert!(
        pixel[..3]
            .iter()
            .any(|channel| channel.is_finite() && *channel > 1.0e-4),
        "back-side glass did not receive {case}: {pixel:?}"
    );
    Ok(())
}

#[test]
fn realtime_glass_transmits_toward_a_viewer_on_its_back_side() -> TestResult {
    assert_backface_glass_lit(
        json!({
            "op": "light.create", "id": "transmitted_point", "light_type": "point", "energy": 1000.0,
            "radius": 0.1, "transform": {"translation": [0.0, 0.0, -2.0]}
        }),
        "point-light transmission",
    )
}

#[test]
fn realtime_glass_reflects_direct_light_on_its_back_side() -> TestResult {
    assert_backface_glass_lit(
        json!({
            "op": "light.create", "id": "reflected_point", "light_type": "point", "energy": 1000.0,
            "radius": 0.1, "transform": {"translation": [0.0, 0.0, 2.0]}
        }),
        "direct reflection",
    )
}

#[test]
fn realtime_glass_transmits_area_light_toward_a_viewer_on_its_back_side() -> TestResult {
    assert_backface_glass_lit(
        json!({
            "op": "light.create", "id": "transmitted_area", "light_type": "area", "energy": 1000.0,
            "transform": {
                "translation": [0.0, 0.0, -2.0],
                "rotation": [0.0, 1.0, 0.0, 0.0]
            }
        }),
        "area-light transmission",
    )
}
