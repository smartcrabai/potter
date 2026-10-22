//! Realtime Beauty's direct light cut-offs, spot cone and sun disc against Blender 5.2.2's EEVEE with its default settings.
//!
//! `fixtures/native_oracle/eevee_direct_light_cases.py` renders a Lambert plane (albedo 0.8) under one light in seven
//! cases, on a black world. Each case is imported from the `.blend` that the same script saves (its save-only mode, with
//! `-` for the render paths), rendered by `pot render --engine realtime`, and every pixel is compared with EEVEE's pixel
//! at the same world position:
//! - the influence cut-off: a point light with the default light threshold, with a raised threshold, with a custom
//!   distance, and an area light;
//! - the spot cone on the plane, unscaled and with its object scaled (1, 1, 2), which EEVEE stretches;
//! - a sun of 10 degrees casting a sphere's shadow, whose penumbra is the sun's disc.
//!
//! EEVEE's pixels are the committed macOS arm64 renders (`fixtures/native_oracle/eevee_macos/`, made by Blender 5.2.2 on
//! macOS; see its `MANIFEST.json`), not live renders, so the test has no platform term.
//!
//! The bounds come from the Python mirror (`fixtures/native_oracle/eevee_direct_light_mirror.py`), which reproduces
//! the same macOS renders over the same pixels: the largest deviation is 5.0e-4 for the point light, 1.2e-3 for the
//! threshold and the area light, 2.2e-3 for the spot, and the means are at most 6.0e-5. The bounds are about twice those,
//! rounded. The sun's largest deviation is at the penumbra's edge, so the sun's bound is 0.1 for the maximum and 1e-3 for
//! the mean. Potter's 16-ray disc misses EEVEE's penumbra by 7.3e-2 at most (mean 3.3e-4), where EEVEE's own symmetric
//! pixels disagree by 4.7e-2: the bound is about twice that. The scaled spot's Cycles cone misses EEVEE by 0.80; the old
//! fixed 0.003 rad penumbra misses by 0.30.
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

/// The oracle's script: it saves each case's `.blend` and, given `-` for the render paths, stops before rendering.
const EEVEE_SCRIPT: &str = include_str!("fixtures/native_oracle/eevee_direct_light_cases.py");

/// The render size and orthographic scale the oracle uses, so Potter's pixel grid is EEVEE's.
const SIZE: u32 = 128;
const ORTHO_SCALE: f64 = 12.0;

/// The sun case's sphere occluder, as `eevee_direct_light_cases.py` places it (its centre's X and Y, and its radius).
const OCCLUDER_CENTRE: [f64; 2] = [0.0, 0.0];
const OCCLUDER_RADIUS: f64 = 0.5;

/// The cases, by the names `eevee_direct_light_cases.py` takes, with their largest and mean allowed deviation, and
/// whether the case's sphere occludes the plane (only the sun case has one).
const CASES: [Case; 7] = [
    Case::new("influence_point", 2.5e-3, 2.0e-4, false),
    Case::new("influence_threshold", 2.5e-3, 2.0e-4, false),
    Case::new("influence_custom", 2.5e-3, 2.0e-4, false),
    Case::new("influence_area", 2.5e-3, 2.0e-4, false),
    Case::new("spot_plain", 5.0e-3, 2.0e-4, false),
    Case::new("spot_scaled", 5.0e-3, 2.0e-4, false),
    Case::new("sun_angle", 1.0e-1, 1.0e-3, true),
];

struct Case {
    name: &'static str,
    max: f64,
    mean: f64,
    occluder: bool,
}

impl Case {
    const fn new(name: &'static str, max: f64, mean: f64, occluder: bool) -> Self {
        Self {
            name,
            max,
            mean,
            occluder,
        }
    }

    /// The committed EEVEE render for this case, as little-endian f32 RGB rows bottom-up.
    fn eevee(&self) -> Result<&'static [u8], Box<dyn Error>> {
        let bytes: &'static [u8] = match self.name {
            "influence_point" => {
                include_bytes!("fixtures/native_oracle/eevee_macos/direct_influence_point.f32le")
            }
            "influence_threshold" => include_bytes!(
                "fixtures/native_oracle/eevee_macos/direct_influence_threshold.f32le"
            ),
            "influence_custom" => {
                include_bytes!("fixtures/native_oracle/eevee_macos/direct_influence_custom.f32le")
            }
            "influence_area" => {
                include_bytes!("fixtures/native_oracle/eevee_macos/direct_influence_area.f32le")
            }
            "spot_plain" => {
                include_bytes!("fixtures/native_oracle/eevee_macos/direct_spot_plain.f32le")
            }
            "spot_scaled" => {
                include_bytes!("fixtures/native_oracle/eevee_macos/direct_spot_scaled.f32le")
            }
            "sun_angle" => {
                include_bytes!("fixtures/native_oracle/eevee_macos/direct_sun_angle.f32le")
            }
            other => return Err(format!("no committed EEVEE render for {other}").into()),
        };
        Ok(bytes)
    }
}

#[test]
fn realtime_direct_light_cases_match_eevee() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "eevee_direct_light_cases.py", EEVEE_SCRIPT)?;
    for case in CASES {
        let blend = root.path().join(format!("{}.blend", case.name));
        run_blender_arguments(
            &blender,
            &script,
            None,
            &[
                Path::new(case.name),
                blend.as_path(),
                Path::new("-"),
                Path::new("-"),
            ],
        )?;

        let project_name = format!("project_{}", case.name);
        let (project, _) = import_blend(&blender, root.path(), &project_name, &blend)?;
        let scene_data = read_scene(&project)?;
        let active = scene_data["active_scene"]
            .as_str()
            .ok_or("active scene is missing")?
            .to_owned();
        let applied = apply_operations(
            &project,
            root.path(),
            json!([
                {"op": "camera.create", "id": "camera_main", "transform": {"translation": [0.0, 0.0, 20.0]}, "projection": "orthographic", "ortho_scale": ORTHO_SCALE},
                {"op": "scene.update", "target": {"id": active}, "set": {"camera": "camera_main"}},
                {"op": "render.update", "target": {"id": active}, "set": {"resolution_x": SIZE, "resolution_y": SIZE, "samples": 1, "seed": 0}}
            ]),
        )?;
        assert_eq!(applied["ok"], true, "{applied}");

        let frames = root.path().join(format!("frames_{}", case.name));
        let rendered = pot()
            .arg("render")
            .arg(&project)
            .args(["--engine", "realtime", "--format", "exr", "--out"])
            .arg(&frames)
            .arg("--json")
            .output()?;
        assert!(
            rendered.status.success(),
            "{}: realtime render failed: {}",
            case.name,
            String::from_utf8_lossy(&rendered.stdout)
        );
        let loaded = apply_operations(
            &project,
            root.path(),
            json!([{"op": "image.load", "id": "render", "path": frames.join("frame_0001.exr").to_string_lossy()}]),
        )?;
        assert_eq!(loaded["ok"], true, "{loaded}");

        let image: Image =
            serde_json::from_value(read_scene(&project)?["images"]["render"].clone())?;
        let data = load_image_data(&image, &project, ImageInterpolation::Closest)?;
        let deviation = deviation(&data, &eevee_rgb(case.eevee()?)?, case.occluder)?;
        assert!(
            deviation.max <= case.max,
            "{}: the largest deviation {} at world {:?} (Potter {:?}, EEVEE {:?}) exceeds {}",
            case.name,
            deviation.max,
            deviation.world,
            deviation.actual,
            deviation.wanted,
            case.max
        );
        assert!(
            deviation.mean <= case.mean,
            "{}: the mean deviation {} exceeds {}",
            case.name,
            deviation.mean,
            case.mean
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

/// The deviations of Potter's plane pixels from EEVEE's. A sun case leaves out the sphere's disc, which the camera sees
/// instead of the plane.
struct Deviation {
    max: f64,
    mean: f64,
    world: [f64; 2],
    actual: f64,
    wanted: f64,
}

fn deviation(
    data: &potter_core::image::ImageData,
    eevee: &[[f64; 3]],
    occluder: bool,
) -> Result<Deviation, Box<dyn Error>> {
    let size = f64::from(SIZE);
    let pixel = ORTHO_SCALE / size;
    let mut worst = Deviation {
        max: 0.0,
        mean: 0.0,
        world: [0.0; 2],
        actual: 0.0,
        wanted: 0.0,
    };
    let mut total = 0.0;
    let mut counted = 0_usize;
    for row in 0..SIZE {
        for column in 0..SIZE {
            let world = [
                -ORTHO_SCALE / 2.0 + (f64::from(column) + 0.5) * pixel,
                -ORTHO_SCALE / 2.0 + (f64::from(row) + 0.5) * pixel,
            ];
            if occluder {
                let offset = [world[0] - OCCLUDER_CENTRE[0], world[1] - OCCLUDER_CENTRE[1]];
                if offset[0].hypot(offset[1]) < OCCLUDER_RADIUS {
                    continue;
                }
            }
            let uv = [
                (world[0] + ORTHO_SCALE / 2.0) / ORTHO_SCALE,
                (world[1] + ORTHO_SCALE / 2.0) / ORTHO_SCALE,
            ];
            let wanted = eevee[(row * SIZE + column) as usize];
            let rgba = sample(data, uv, false);
            let actual = [rgba[0], rgba[1], rgba[2]];
            let deviation = actual
                .iter()
                .zip(wanted)
                .map(|(a, w)| (a - w).abs())
                .fold(0.0_f64, f64::max);
            total += deviation;
            counted += 1;
            if deviation > worst.max {
                worst.max = deviation;
                worst.world = world;
                worst.actual = actual[0];
                worst.wanted = wanted[0];
            }
        }
    }
    if counted == 0 {
        return Err("no plane pixels were compared".into());
    }
    worst.mean = total / counted as f64;
    Ok(worst)
}
