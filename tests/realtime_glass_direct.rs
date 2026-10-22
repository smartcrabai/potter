//! Realtime Beauty's direct light through glass against Blender 5.2.2's EEVEE with its default settings, on a black world.
//!
//! `fixtures/native_oracle/eevee_glass_direct.py` renders two glass spheres (IOR 1.45, Transmission 1, Roughness 0 and
//! 0.3) under one light, in four scenes: a point light and a 2 m square area light above the spheres (the glass reflects
//! them and transmits none), and the same two lights below (they refract through the spheres to the viewer). The light
//! is the only source, so every pixel is the glass's direct light: EEVEE's LTC reflection and transmission, the
//! transmission's two events with the object's bounds as thickness, and the thickness-amended shadow. The spheres are
//! imported from the `.blend` that the same script saves (its save-only mode, with `-` for the render paths), and
//! `pot render --engine realtime` renders them.
//!
//! EEVEE's pixels are the committed macOS arm64 render (`fixtures/native_oracle/eevee_macos/`, made by Blender 5.2.2 on
//! macOS; see its `MANIFEST.json`), not a live render, so the test has no platform term. Every pixel whose centre lies
//! within 0.75 of a sphere's radius is compared with the committed pixel at the same UV.
//!
//! The tolerance is derived from the Python mirror (`fixtures/native_oracle/eevee_glass_direct_mirror.py`), which
//! reproduces the same macOS render over the same pixels with a worst deviation of 8.0e-3, at the rim of the rough sphere
//! in the point light below (the transmitted lobe is narrow there and crosses the light's edge) and 4.0e-3 for the area
//! light below. The clear sphere's worst is 1.3e-3. The bound is twice the largest, rounded up to 1.6e-2. The mirror with
//! no thickness (no second event and no amended shadow) misses the below-surface scenes by 0.40 to 0.53, far outside it.
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
    TestResult, apply_operations, import_blend, node_id, pot, read_scene, run_blender_arguments,
    write_script,
};
use potter_core::image::{ImageData, ImageInterpolation, load_image_data, sample};
use potter_core::model::Image;
use serde_json::json;
use tempfile::tempdir;

/// Largest allowed deviation from EEVEE's pixel (see the module docs).
const TOLERANCE: f64 = 1.6e-2;

/// The oracle's script: it saves the scene's `.blend` and, given `-` for the render paths, stops before rendering.
const EEVEE_SCRIPT: &str = include_str!("fixtures/native_oracle/eevee_glass_direct.py");

/// The committed macOS arm64 EEVEE renders, as little-endian f32 RGB (rows bottom-up, as Blender stores them).
const EEVEE_POINT_ABOVE: &[u8] =
    include_bytes!("fixtures/native_oracle/eevee_macos/glass_direct_point_above.f32le");
const EEVEE_AREA_ABOVE: &[u8] =
    include_bytes!("fixtures/native_oracle/eevee_macos/glass_direct_area_above.f32le");
const EEVEE_POINT_BELOW: &[u8] =
    include_bytes!("fixtures/native_oracle/eevee_macos/glass_direct_point_below.f32le");
const EEVEE_AREA_BELOW: &[u8] =
    include_bytes!("fixtures/native_oracle/eevee_macos/glass_direct_area_below.f32le");

/// The four scenes, by the names `eevee_glass_direct.py` takes, with their committed EEVEE renders.
const SCENES: [(&str, &[u8]); 4] = [
    ("point_above", EEVEE_POINT_ABOVE),
    ("area_above", EEVEE_AREA_ABOVE),
    ("point_below", EEVEE_POINT_BELOW),
    ("area_below", EEVEE_AREA_BELOW),
];

/// The spheres' names and centre X, as `eevee_glass_direct.py` places them.
const SPHERES: [(&str, f64); 2] = [("clear", -2.4), ("rough", 2.4)];

/// The render size and orthographic scale the oracle uses, so Potter's pixel grid is EEVEE's.
const SIZE: u32 = 96;
const ORTHO_SCALE: f64 = 7.6;

/// The interior radius of a sphere, in units of its radius (`INTERIOR` in the oracle).
const INTERIOR: f64 = 0.75;

#[test]
fn realtime_glass_direct_light_matches_eevee() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "eevee_glass_direct.py", EEVEE_SCRIPT)?;
    for (scene, eevee) in SCENES {
        let data = render_realtime_scene(&blender, root.path(), &script, scene, None)?;
        let eevee_pixels = eevee_rgb(eevee)?;
        for (name, centre_x) in SPHERES {
            let worst = worst_deviation(&data, &eevee_pixels, sphere_centre_px(centre_x))?;
            assert!(
                worst.max <= TOLERANCE,
                "{scene} {name} sphere: Potter realtime {:?} differs from EEVEE {:?} by {} at world {:?} (tolerance {TOLERANCE})",
                worst.actual,
                worst.wanted,
                worst.max,
                worst.world,
            );
        }
    }
    Ok(())
}

#[test]
fn realtime_glass_direct_reflection_honors_light_visible_glossy() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "eevee_glass_direct.py", EEVEE_SCRIPT)?;
    let data = render_realtime_scene(&blender, root.path(), &script, "point_above", Some(false))?;
    let black = vec![[0.0; 3]; (SIZE * SIZE) as usize];
    for (name, centre_x) in SPHERES {
        let worst = worst_deviation(&data, &black, sphere_centre_px(centre_x))?;
        assert!(
            worst.max <= 1.0e-6,
            "{name} sphere still reflects a light with visible_glossy=false: RGB {:?} at world {:?}",
            worst.actual,
            worst.world,
        );
    }
    let transmission =
        render_realtime_scene(&blender, root.path(), &script, "point_below", Some(false))?;
    let eevee_pixels = eevee_rgb(EEVEE_POINT_BELOW)?;
    for (name, centre_x) in SPHERES {
        let worst = worst_deviation(&transmission, &eevee_pixels, sphere_centre_px(centre_x))?;
        assert!(
            worst.max <= TOLERANCE,
            "{name} glass transmission changed with light visible_glossy=false: {:?} differs from EEVEE {:?} by {}",
            worst.actual,
            worst.wanted,
            worst.max,
        );
    }
    Ok(())
}

fn render_realtime_scene(
    blender: &Path,
    root: &Path,
    script: &Path,
    scene: &str,
    visible_glossy: Option<bool>,
) -> TestResult<ImageData> {
    let blend = root.join(format!("{scene}.blend"));
    run_blender_arguments(
        blender,
        script,
        None,
        &[
            Path::new(scene),
            blend.as_path(),
            Path::new("-"),
            Path::new("-"),
        ],
    )?;

    let project_name = format!("project_{scene}");
    let (project, _) = import_blend(blender, root, &project_name, &blend)?;
    let scene_data = read_scene(&project)?;
    let active = scene_data["active_scene"]
        .as_str()
        .ok_or("active scene is missing")?
        .to_owned();
    let mut operations = vec![
        json!({"op": "camera.create", "id": "camera_main", "transform": {"translation": [0.0, 0.0, 10.0]}, "projection": "orthographic", "ortho_scale": ORTHO_SCALE}),
        json!({"op": "scene.update", "target": {"id": active}, "set": {"camera": "camera_main"}}),
        json!({"op": "render.update", "target": {"id": active}, "set": {"resolution_x": SIZE, "resolution_y": SIZE, "samples": 1, "seed": 0}}),
    ];
    if let Some(visible_glossy) = visible_glossy {
        let light = node_id(&scene_data, &format!("Light_{scene}"))?;
        operations.push(json!({
            "op": "node.update",
            "target": {"id": light},
            "set": {"ray_visibility": {
                "camera": false,
                "diffuse": true,
                "glossy": visible_glossy,
                "transmission": true,
                "volume_scatter": true,
                "shadow": true
            }}
        }));
    }
    let applied = apply_operations(&project, root, json!(operations))?;
    assert_eq!(applied["ok"], true, "{applied}");

    let frames = root.join(format!("frames_{scene}"));
    let rendered = pot()
        .arg("render")
        .arg(&project)
        .args(["--engine", "realtime", "--format", "exr", "--out"])
        .arg(&frames)
        .arg("--json")
        .output()?;
    assert!(
        rendered.status.success(),
        "{scene}: realtime render failed: {}",
        String::from_utf8_lossy(&rendered.stdout)
    );
    let loaded = apply_operations(
        &project,
        root,
        json!([{"op": "image.load", "id": "render", "path": frames.join("frame_0001.exr").to_string_lossy()}]),
    )?;
    assert_eq!(loaded["ok"], true, "{loaded}");

    let image: Image = serde_json::from_value(read_scene(&project)?["images"]["render"].clone())?;
    Ok(load_image_data(
        &image,
        &project,
        ImageInterpolation::Closest,
    )?)
}

/// The committed EEVEE render's RGB values, one `[f64; 3]` per pixel in Blender's row order.
fn eevee_rgb(bytes: &[u8]) -> Result<Vec<[f64; 3]>, Box<dyn Error>> {
    let pixel_count = (SIZE * SIZE) as usize;
    if bytes.len() != pixel_count * 12 {
        return Err(format!(
            "the EEVEE fixture holds {} bytes, expected {}",
            bytes.len(),
            pixel_count * 12
        )
        .into());
    }
    let channel = |index: usize| {
        let at = index * 4;
        f64::from(f32::from_le_bytes([
            bytes[at],
            bytes[at + 1],
            bytes[at + 2],
            bytes[at + 3],
        ]))
    };
    Ok((0..pixel_count)
        .map(|pixel| {
            [
                channel(pixel * 3),
                channel(pixel * 3 + 1),
                channel(pixel * 3 + 2),
            ]
        })
        .collect())
}

/// A sphere's centre in pixels along X, as the oracle computes it: `(x + ORTHO_SCALE / 2) * SIZE / ORTHO_SCALE`.
fn sphere_centre_px(centre_x: f64) -> f64 {
    (centre_x + ORTHO_SCALE / 2.0) * f64::from(SIZE) / ORTHO_SCALE
}

/// The largest per-channel deviation of Potter's realtime pixels from EEVEE's over one sphere's interior pixels.
struct Deviation {
    max: f64,
    world: [f64; 2],
    actual: [f64; 3],
    wanted: [f64; 3],
}

/// Compares every interior pixel of one sphere: a pixel's centre within `INTERIOR` of the sphere's radius, measured in
/// pixels, as the oracle selects them.
fn worst_deviation(
    data: &potter_core::image::ImageData,
    eevee: &[[f64; 3]],
    centre_px: f64,
) -> Result<Deviation, Box<dyn Error>> {
    let size = f64::from(SIZE);
    let scale = size / ORTHO_SCALE;
    let mut worst = Deviation {
        max: 0.0,
        world: [0.0; 2],
        actual: [0.0; 3],
        wanted: [0.0; 3],
    };
    let mut count = 0_usize;
    for row in 0..SIZE {
        for column in 0..SIZE {
            let dx = (f64::from(column) + 0.5 - centre_px) / scale;
            let dy = (f64::from(row) + 0.5 - size / 2.0) / scale;
            if (dx * dx + dy * dy).sqrt() > INTERIOR {
                continue;
            }
            count += 1;
            let uv = [
                (f64::from(column) + 0.5) / size,
                (f64::from(row) + 0.5) / size,
            ];
            let wanted = eevee[(row * SIZE + column) as usize];
            let rgba = sample(data, uv, false);
            let actual = [rgba[0], rgba[1], rgba[2]];
            let deviation = actual
                .iter()
                .zip(wanted)
                .map(|(a, w)| (a - w).abs())
                .fold(0.0_f64, f64::max);
            if deviation > worst.max {
                let pixel = ORTHO_SCALE / size;
                worst = Deviation {
                    max: deviation,
                    world: [
                        -ORTHO_SCALE / 2.0 + (f64::from(column) + 0.5) * pixel,
                        -ORTHO_SCALE / 2.0 + (f64::from(row) + 0.5) * pixel,
                    ],
                    actual,
                    wanted,
                };
            }
        }
    }
    if count < 100 {
        return Err(format!("only {count} interior pixels; the oracle needs more").into());
    }
    Ok(worst)
}
