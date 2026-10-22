//! Realtime Beauty transmission over a coloured, non-uniform world, against Blender 5.2.2's EEVEE with default settings,
//! on macOS arm64.
//!
//! `fixtures/native_oracle/eevee_glass_world.py` renders three glass spheres (IOR 1.45, Transmission 1) under an
//! Environment Texture whose colour depends only on elevation: red-heavy at the top, blue-heavy at the bottom. The
//! spheres are a clear one, a tinted one (Base Color (0.9, 0.5, 0.2)) and a rough one (Roughness 0.5). Off-centre pixels
//! see the world at tilted directions, so the reflected and refracted probe directions, the sphere thickness and the
//! probe's blur all decide them. The spheres are imported from the `.blend` that the same script saves (its save-only
//! mode, with `-` for the render paths; the environment EXR it writes is the one the `.blend` references), and
//! `pot render --engine realtime` renders them.
//!
//! EEVEE's pixels are the committed macOS arm64 render (`fixtures/native_oracle/eevee_macos/`, made by Blender 5.2.2 on
//! macOS; see its `MANIFEST.json`), not a live render, so the test has no platform term. Every pixel whose centre lies
//! within 0.75 of a sphere's radius is compared with the committed pixel at the same UV.
//!
//! The tolerance per sphere is the model bound alone: the largest deviation of the Python mirror over the world
//! (`fixtures/native_oracle/eevee_glass_world_mirror.py`, which reads the world through the render's 64-texel cube-map
//! probe, as Potter does) from the committed render, rounded up. Those are clear 0.01049 (bound 1.1e-2), tinted 0.00341
//! (3.5e-3) and rough 0.01038 (1.1e-2). The worst clear and rough pixels sit on the probe's 5-bit blue truncation and its
//! texel edges, where EEVEE and the model both sample a texel. The same mirror with a single refraction event (a thin
//! object) gives 0.062 (clear), 0.187 (tinted) and 0.059 (rough), far outside the bounds.
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
use serde_json::json;
use tempfile::tempdir;

/// The oracle's script: it writes the environment EXR and saves the scene's `.blend`, then stops before rendering when
/// the render paths are `-`.
const EEVEE_SCRIPT: &str = include_str!("fixtures/native_oracle/eevee_glass_world.py");

/// The committed macOS arm64 EEVEE render, as little-endian f32 RGB (rows bottom-up, as Blender stores them).
const EEVEE_WORLD: &[u8] = include_bytes!("fixtures/native_oracle/eevee_macos/glass_world.f32le");

/// The spheres, as `eevee_glass_world.py` places them: name, centre X, and the largest allowed deviation from EEVEE's
/// pixel (see the module docs: the model bound, rounded up).
const SPHERES: [(&str, f64, f64); 3] = [
    ("clear", -2.4, 1.1e-2),
    ("tinted", 0.0, 3.5e-3),
    ("rough", 2.4, 1.1e-2),
];

/// The render size and orthographic scale the oracle uses, so Potter's pixel grid is EEVEE's.
const SIZE: u32 = 96;
const ORTHO_SCALE: f64 = 7.6;

/// The interior radius of a sphere, in units of its radius (`INTERIOR` in the oracle).
const INTERIOR: f64 = 0.75;

#[test]
fn realtime_glass_world_transmission_matches_eevee() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "eevee_glass_world.py", EEVEE_SCRIPT)?;
    let blend = root.path().join("glass_world.blend");
    let environment = root.path().join("environment.exr");
    run_blender_arguments(
        &blender,
        &script,
        None,
        &[
            blend.as_path(),
            Path::new("-"),
            Path::new("-"),
            environment.as_path(),
        ],
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
            {"op": "camera.create", "id": "camera_main", "transform": {"translation": [0.0, 0.0, 10.0]}, "projection": "orthographic", "ortho_scale": ORTHO_SCALE},
            {"op": "scene.update", "target": {"id": active}, "set": {"camera": "camera_main"}},
            {"op": "render.update", "target": {"id": active}, "set": {"resolution_x": SIZE, "resolution_y": SIZE, "samples": 1, "seed": 0}}
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

    let image: Image = serde_json::from_value(read_scene(&project)?["images"]["render"].clone())?;
    let data = load_image_data(&image, &project, ImageInterpolation::Closest)?;
    let eevee = eevee_rgb(EEVEE_WORLD)?;
    // Every sphere is measured before any assertion, so one failure does not hide the others' numbers.
    let mut results = Vec::new();
    for (name, centre_x, tolerance) in SPHERES {
        results.push((
            name,
            tolerance,
            worst_deviation(&data, &eevee, sphere_centre_px(centre_x))?,
        ));
    }
    for (name, tolerance, worst) in results {
        assert!(
            worst.max <= tolerance,
            "{name} sphere: Potter realtime {:?} differs from EEVEE {:?} by {} at uv {:?} (tolerance {tolerance})",
            worst.actual,
            worst.wanted,
            worst.max,
            worst.uv,
        );
    }
    Ok(())
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
    uv: [f64; 2],
    actual: [f64; 3],
    wanted: [f64; 3],
}

fn worst_deviation(
    data: &potter_core::image::ImageData,
    eevee: &[[f64; 3]],
    centre_px: f64,
) -> Result<Deviation, Box<dyn Error>> {
    let size = f64::from(SIZE);
    let scale = size / ORTHO_SCALE;
    let mut worst = Deviation {
        max: 0.0,
        uv: [0.0; 2],
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
                worst = Deviation {
                    max: deviation,
                    uv,
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
