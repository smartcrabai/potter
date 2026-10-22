//! Realtime Beauty transmits direct light through a double-sided glass face viewed from its back. The camera and
//! transmission normal are antiparallel, exercising the LTC frame's zero-tangent case through the CLI render path.

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
use serde_json::json;
use tempfile::tempdir;

#[test]
fn realtime_glass_transmits_toward_a_viewer_on_its_back_side() -> TestResult {
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
            {"op": "light.create", "id": "transmitted_light", "light_type": "point", "energy": 1000.0,
             "radius": 0.1, "transform": {"translation": [0.0, 0.0, -2.0]}},
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
        "realtime render failed: {}",
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
        pixel[..3].iter().any(|channel| channel.is_finite() && *channel > 1.0e-4),
        "back-side glass did not transmit the light below it: {pixel:?}"
    );
    Ok(())
}
