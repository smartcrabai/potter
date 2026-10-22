//! Realtime Beauty transmission against Blender 5.2.2's EEVEE with its default settings.
//!
//! `fixtures/native_oracle/eevee_glass_transmission.py` renders three glass spheres (IOR 1.45, Transmission 1) under a
//! white world with no lights: a clear sphere (Base Color white, Roughness 0), a tinted one (Base Color (0.9, 0.5,
//! 0.2), so the transmission tint is the square root of the base), and a rough one (Roughness 0.5). EEVEE's default
//! closures are a reflection and a transmission sampled from the light probe, and the transmission takes the object's
//! bounds as its thickness, so it is two refraction events. The spheres are imported from the saved `.blend`, rendered
//! by `pot render --engine realtime`, and every pixel whose centre lies within 0.75 of a sphere's radius is compared
//! with EEVEE's pixel at the same UV.
//!
//! The tolerance is derived from the model. `eevee_transmission.rs` (R + (T · √base)², with the LUT split and the
//! thickness squared) differs from EEVEE by at most 0.00169 over those pixels (the rough sphere at its rim, r = 0.746;
//! the clear sphere is at most 0.00134 and the tinted one 0.00113). EEVEE stores its render in half float, one step at
//! [0.5, 1) being 4.9e-4. The bound is that worst deviation plus one half-float step, rounded up to 2.5e-3. A dropped
//! transmission, a single refraction event (T instead of T², 0.033 at the centre), or a tint that is the base colour
//! rather than its square root (0.1 or more on the tinted sphere) fails by an order of magnitude.
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

use std::{error::Error, fs};

use native_oracle::{
    TestResult, apply_operations, import_blend, pot, read_scene, run_blender_arguments,
    write_script,
};
use potter_core::image::{ImageInterpolation, load_image_data, sample};
use potter_core::model::Image;
use serde_json::{Value, json};
use tempfile::tempdir;

/// Largest allowed deviation from EEVEE's pixel (see the module docs).
const TOLERANCE: f64 = 2.5e-3;

/// Blender 5.2.2 EEVEE oracle: the three spheres, the saved `.blend`, and the interior pixels with EEVEE's RGB.
const EEVEE_SCRIPT: &str = include_str!("fixtures/native_oracle/eevee_glass_transmission.py");

/// The spheres, by the names `eevee_glass_transmission.py` writes into its JSON.
const SPHERES: [&str; 3] = ["clear", "tinted", "rough"];

/// The render size and orthographic scale the oracle uses, so Potter's pixel grid is EEVEE's.
const SIZE: u32 = 96;
const ORTHO_SCALE: f64 = 7.6;

#[test]
fn realtime_glass_transmission_matches_eevee() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "eevee_glass_transmission.py", EEVEE_SCRIPT)?;
    let blend = root.path().join("glass_transmission.blend");
    let eevee_exr = root.path().join("eevee_glass_transmission.exr");
    let oracle_json = root.path().join("eevee_glass_transmission.json");
    run_blender_arguments(
        &blender,
        &script,
        None,
        &[blend.as_path(), eevee_exr.as_path(), oracle_json.as_path()],
    )?;
    let oracle: Value = serde_json::from_slice(&fs::read(&oracle_json)?)?;

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
    for name in SPHERES {
        eprintln!("GLASS_SPHERE {name}");
        let worst = worst_deviation(&data, &oracle["spheres"][name])?;
        assert!(
            worst.deviation <= TOLERANCE,
            "{name} sphere: Potter realtime {:?} differs from EEVEE {:?} by {} at uv {:?} (tolerance {TOLERANCE})",
            worst.actual,
            worst.wanted,
            worst.deviation,
            worst.uv,
        );
    }
    Ok(())
}

/// The largest per-channel deviation of Potter's realtime pixels from EEVEE's over one sphere's interior samples.
struct Deviation {
    deviation: f64,
    uv: [f64; 2],
    actual: [f64; 3],
    wanted: [f64; 3],
}

fn worst_deviation(
    data: &potter_core::image::ImageData,
    sphere: &Value,
) -> Result<Deviation, Box<dyn Error>> {
    let samples = sphere["samples"]
        .as_array()
        .ok_or("the EEVEE samples are missing")?;
    if samples.len() < 100 {
        return Err(format!(
            "only {} interior samples; the oracle needs more",
            samples.len()
        )
        .into());
    }
    let mut worst = Deviation {
        deviation: 0.0,
        uv: [0.0; 2],
        actual: [0.0; 3],
        wanted: [0.0; 3],
    };
    for sample_value in samples {
        let uv = pair(&sample_value["uv"])?;
        let wanted = triple(&sample_value["rgb"])?;
        let rgba = sample(data, uv, false);
        let actual = [rgba[0], rgba[1], rgba[2]];
        let deviation = actual
            .iter()
            .zip(wanted)
            .map(|(a, w)| (a - w).abs())
            .fold(0.0_f64, f64::max);
        // Diagnostic: every checked pixel's EEVEE value (this platform's Blender) and Potter's, for measuring the
        // platform difference against the model (`GLASS_PX uv_u uv_v eevee_r eevee_g eevee_b potter_r potter_g potter_b`).
        eprintln!(
            "GLASS_PX {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
            uv[0],
            uv[1],
            wanted[0],
            wanted[1],
            wanted[2],
            actual[0],
            actual[1],
            actual[2],
            deviation
        );
        if deviation > worst.deviation {
            worst = Deviation {
                deviation,
                uv,
                actual,
                wanted,
            };
        }
    }
    Ok(worst)
}

fn pair(value: &Value) -> Result<[f64; 2], Box<dyn Error>> {
    let items = value.as_array().ok_or("a UV is not an array")?;
    Ok([
        items
            .first()
            .and_then(Value::as_f64)
            .ok_or("UV u is missing")?,
        items
            .get(1)
            .and_then(Value::as_f64)
            .ok_or("UV v is missing")?,
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
