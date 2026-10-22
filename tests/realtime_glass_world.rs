//! Realtime Beauty transmission over a coloured, non-uniform world, against Blender 5.2.2's EEVEE with default settings.
//!
//! `fixtures/native_oracle/eevee_glass_world.py` renders three glass spheres (IOR 1.45, Transmission 1) under an
//! Environment Texture whose colour depends only on elevation: red-heavy at the top, blue-heavy at the bottom. The
//! spheres are a clear one, a tinted one (Base Color (0.9, 0.5, 0.2)) and a rough one (Roughness 0.5). Off-centre pixels
//! see the world at tilted directions, so the reflected and refracted probe directions, the sphere thickness and the
//! probe's blur all decide them. The spheres are imported from the saved `.blend`, rendered by
//! `pot render --engine realtime`, and every pixel whose centre lies within 0.75 of a sphere's radius is compared with
//! EEVEE's pixel at the same UV.
//!
//! The tolerance is derived from the model (the Python mirror of `eevee_transmission.rs` over the same pixels): the
//! median deviation of the clear and rough spheres is 0.0017 and 0.0026, and the worst is 0.0105 (clear) and 0.0099
//! (rough). The worst pixels sit on a 5-bit truncation step of the probe's blue channel (a world value of 0.7498 is
//! captured as 0.734), where EEVEE truncates its texels and the model truncates each sample, so the bound is that worst
//! deviation rounded up to 0.0125. The same mirror with the probe straight through gives a worst deviation of 0.050, and
//! with the single-event (thin) probe 0.035, so the bound separates a wrong probe direction from the truncation noise.
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
const TOLERANCE: f64 = 1.25e-2;

/// Blender 5.2.2 EEVEE oracle: the three spheres, the saved `.blend`, the environment EXR and the interior pixels.
const EEVEE_SCRIPT: &str = include_str!("fixtures/native_oracle/eevee_glass_world.py");

/// The spheres, by the names `eevee_glass_transmission.py` writes into its JSON.
const SPHERES: [&str; 3] = ["clear", "tinted", "rough"];

/// The render size and orthographic scale the oracle uses, so Potter's pixel grid is EEVEE's.
const SIZE: u32 = 96;
const ORTHO_SCALE: f64 = 7.6;

#[test]
fn realtime_glass_world_transmission_matches_eevee() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "eevee_glass_world.py", EEVEE_SCRIPT)?;
    let blend = root.path().join("glass_world.blend");
    let eevee_exr = root.path().join("eevee_glass_world.exr");
    let oracle_json = root.path().join("eevee_glass_world.json");
    let environment = root.path().join("environment.exr");
    run_blender_arguments(
        &blender,
        &script,
        None,
        &[
            blend.as_path(),
            eevee_exr.as_path(),
            oracle_json.as_path(),
            environment.as_path(),
        ],
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
    // Diagnostic: the render's stderr carries the DIAG lines of `scene.rs` and `path.rs`.
    eprint!("{}", String::from_utf8_lossy(&rendered.stderr));
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
