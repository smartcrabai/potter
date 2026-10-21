//! Realtime Beauty's direct light through glass against Blender 5.2.2's EEVEE with its default settings, on a black world.
//!
//! `fixtures/native_oracle/eevee_glass_direct.py` renders two glass spheres (IOR 1.45, Transmission 1, Roughness 0 and
//! 0.3) under one light, in four scenes: a point light and a 2 m square area light above the spheres (the glass reflects
//! them and transmits none), and the same two lights below (they refract through the spheres to the viewer). The light
//! is the only source, so every pixel is the glass's direct light: EEVEE's LTC reflection and transmission, the
//! transmission's two events with the object's bounds as thickness, and the thickness-amended shadow. The spheres are
//! imported from the saved `.blend`, rendered by `pot render --engine realtime`, and every pixel whose centre lies within
//! 0.75 of a sphere's radius is compared with EEVEE's pixel at the same UV.
//!
//! The tolerance is derived from the Python mirror (`fixtures/native_oracle/eevee_glass_direct_mirror.py`), which
//! reproduces EEVEE over the same pixels with a worst deviation of 8.0e-3, at the rim of the rough sphere in the point
//! light below (the transmitted lobe is narrow there and crosses the light's edge) and 4.0e-3 for the area light below.
//! The clear sphere's worst is 1.3e-3. The bound is twice the largest, rounded up to 1.6e-2. The mirror with no thickness
//! (no second event and no amended shadow) misses the below-surface scenes by 0.40 to 0.53, far outside it.
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

/// Largest allowed deviation from EEVEE's pixel (see the module docs).
const TOLERANCE: f64 = 1.6e-2;

/// Blender 5.2.2 EEVEE oracle: the spheres, the light of one scene, and the saved `.blend` and interior pixels.
const EEVEE_SCRIPT: &str = include_str!("fixtures/native_oracle/eevee_glass_direct.py");

/// The four scenes the oracle renders, by the names `eevee_glass_direct.py` takes.
const SCENES: [&str; 4] = ["point_above", "area_above", "point_below", "area_below"];

/// The spheres, by the names `eevee_glass_direct.py` writes into its JSON.
const SPHERES: [&str; 2] = ["clear", "rough"];

/// The render size and orthographic scale the oracle uses, so Potter's pixel grid is EEVEE's.
const SIZE: u32 = 96;
const ORTHO_SCALE: f64 = 7.6;

#[test]
fn realtime_glass_direct_light_matches_eevee() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "eevee_glass_direct.py", EEVEE_SCRIPT)?;
    for scene in SCENES {
        let blend = root.path().join(format!("{scene}.blend"));
        let eevee_exr = root.path().join(format!("{scene}.exr"));
        let oracle_json = root.path().join(format!("{scene}.json"));
        run_blender_arguments(
            &blender,
            &script,
            None,
            &[
                Path::new(scene),
                blend.as_path(),
                eevee_exr.as_path(),
                oracle_json.as_path(),
            ],
        )?;
        let oracle: Value = serde_json::from_slice(&fs::read(&oracle_json)?)?;

        let project_name = format!("project_{scene}");
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
                {"op": "camera.create", "id": "camera_main", "transform": {"translation": [0.0, 0.0, 10.0]}, "projection": "orthographic", "ortho_scale": ORTHO_SCALE},
                {"op": "scene.update", "target": {"id": active}, "set": {"camera": "camera_main"}},
                {"op": "render.update", "target": {"id": active}, "set": {"resolution_x": SIZE, "resolution_y": SIZE, "samples": 1, "seed": 0}}
            ]),
        )?;
        assert_eq!(applied["ok"], true, "{applied}");

        let frames = root.path().join(format!("frames_{scene}"));
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
            root.path(),
            json!([{"op": "image.load", "id": "render", "path": frames.join("frame_0001.exr").to_string_lossy()}]),
        )?;
        assert_eq!(loaded["ok"], true, "{loaded}");

        let image: Image =
            serde_json::from_value(read_scene(&project)?["images"]["render"].clone())?;
        let data = load_image_data(&image, &project, ImageInterpolation::Closest)?;
        for name in SPHERES {
            let worst = worst_deviation(&data, &oracle["spheres"][name])?;
            assert!(
                worst.deviation <= TOLERANCE,
                "{scene} {name} sphere: Potter realtime {:?} differs from EEVEE {:?} by {} at world {:?} (tolerance {TOLERANCE})",
                worst.actual,
                worst.wanted,
                worst.deviation,
                worst.world,
            );
        }
    }
    Ok(())
}

/// The largest per-channel deviation of Potter's realtime pixels from EEVEE's over one sphere's interior samples.
struct Deviation {
    deviation: f64,
    world: [f64; 2],
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
        world: [0.0; 2],
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
                world: pair(&sample_value["world"])?,
                actual,
                wanted,
            };
        }
    }
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
