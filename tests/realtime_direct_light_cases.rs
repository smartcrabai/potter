//! Realtime Beauty's direct light cut-offs, spot cone and sun disc against Blender 5.2.2's EEVEE with its default settings.
//!
//! `fixtures/native_oracle/eevee_direct_light_cases.py` renders a Lambert plane (albedo 0.8) under one light in seven
//! cases, on a black world. Each case is imported from its saved `.blend`, rendered by `pot render --engine realtime`, and
//! every pixel is compared with EEVEE's pixel at the same world position:
//! - the influence cut-off: a point light with the default light threshold, with a raised threshold, with a custom
//!   distance, and an area light;
//! - the spot cone on the plane, unscaled and with its object scaled (1, 1, 2), which EEVEE stretches;
//! - a sun of 10 degrees casting a sphere's shadow, whose penumbra is the sun's disc.
//!
//! The bounds come from the Python mirror (`fixtures/native_oracle/eevee_direct_light_mirror.py`), which reproduces
//! EEVEE over the same pixels: the largest deviation is 5.0e-4 for the point light, 1.2e-3 for the threshold and the area
//! light, 2.2e-3 for the spot, and the means are at most 6.0e-5. The bounds are about twice those, rounded. The sun's
//! largest deviation is at the penumbra's edge, so the
//! sun's bound is 0.1 for the maximum and 1e-3 for the mean. Potter's 16-ray disc misses EEVEE's penumbra by 7.3e-2 at most
//! (mean 3.3e-4), where EEVEE's own symmetric pixels disagree by 4.7e-2: the bound is about twice that. The scaled spot's
//! Cycles cone misses EEVEE by 0.80; the old fixed 0.003 rad penumbra misses by 0.30.
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

use std::{error::Error, fs, path::Path};

use native_oracle::{
    TestResult, apply_operations, import_blend, pot, read_scene, run_blender_arguments,
    write_script,
};
use potter_core::image::{ImageInterpolation, load_image_data, sample};
use potter_core::model::Image;
use serde_json::{Value, json};
use tempfile::tempdir;

/// Blender 5.2.2 EEVEE oracle: one light per case, the saved `.blend`, and every pixel of the plane.
const EEVEE_SCRIPT: &str = include_str!("fixtures/native_oracle/eevee_direct_light_cases.py");

/// The render size and orthographic scale the oracle uses, so Potter's pixel grid is EEVEE's.
const SIZE: u32 = 128;
const ORTHO_SCALE: f64 = 12.0;

/// The cases, by the names `eevee_direct_light_cases.py` takes, with their largest and mean allowed deviation.
const CASES: [Case; 7] = [
    Case::new("influence_point", 2.5e-3, 2.0e-4),
    Case::new("influence_threshold", 2.5e-3, 2.0e-4),
    Case::new("influence_custom", 2.5e-3, 2.0e-4),
    Case::new("influence_area", 2.5e-3, 2.0e-4),
    Case::new("spot_plain", 5.0e-3, 2.0e-4),
    Case::new("spot_scaled", 5.0e-3, 2.0e-4),
    Case::new("sun_angle", 1.0e-1, 1.0e-3),
];

struct Case {
    name: &'static str,
    max: f64,
    mean: f64,
}

impl Case {
    const fn new(name: &'static str, max: f64, mean: f64) -> Self {
        Self { name, max, mean }
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
        let eevee_exr = root.path().join(format!("{}.exr", case.name));
        let oracle_json = root.path().join(format!("{}.json", case.name));
        run_blender_arguments(
            &blender,
            &script,
            None,
            &[
                Path::new(case.name),
                blend.as_path(),
                eevee_exr.as_path(),
                oracle_json.as_path(),
            ],
        )?;
        let oracle: Value = serde_json::from_slice(&fs::read(&oracle_json)?)?;

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
        let deviation = deviation(&data, &oracle)?;
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
    oracle: &Value,
) -> Result<Deviation, Box<dyn Error>> {
    let occluder = &oracle["occluder"];
    let occluder_centre = if occluder.is_null() {
        None
    } else {
        Some((
            pair(&occluder["centre"])?,
            occluder["radius"]
                .as_f64()
                .ok_or("the occluder radius is missing")?,
        ))
    };
    let samples = oracle["samples"]
        .as_array()
        .ok_or("the EEVEE samples are missing")?;
    let mut worst = Deviation {
        max: 0.0,
        mean: 0.0,
        world: [0.0; 2],
        actual: 0.0,
        wanted: 0.0,
    };
    let mut total = 0.0;
    let mut counted = 0_usize;
    for sample_value in samples {
        let world = pair(&sample_value["world"])?;
        if let Some((centre, radius)) = occluder_centre {
            let offset = [world[0] - centre[0], world[1] - centre[1]];
            if offset[0].hypot(offset[1]) < radius {
                continue;
            }
        }
        let uv = [
            (world[0] + ORTHO_SCALE / 2.0) / ORTHO_SCALE,
            (world[1] + ORTHO_SCALE / 2.0) / ORTHO_SCALE,
        ];
        let wanted = triple(&sample_value["rgb"])?;
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
    if counted == 0 {
        return Err("no plane pixels were compared".into());
    }
    worst.mean = total / counted as f64;
    Ok(worst)
}

fn pair(value: &Value) -> Result<[f64; 2], Box<dyn Error>> {
    let items = value
        .as_array()
        .ok_or("a coordinate pair is not an array")?;
    Ok([
        items
            .first()
            .and_then(Value::as_f64)
            .ok_or("the first coordinate is missing")?,
        items
            .get(1)
            .and_then(Value::as_f64)
            .ok_or("the second coordinate is missing")?,
    ])
}

fn triple(value: &Value) -> Result<[f64; 3], Box<dyn Error>> {
    let items = value.as_array().ok_or("an RGB is not an array")?;
    let channel = |index: usize| {
        items
            .get(index)
            .and_then(Value::as_f64)
            .ok_or("an RGB channel is missing")
    };
    Ok([channel(0)?, channel(1)?, channel(2)?])
}
