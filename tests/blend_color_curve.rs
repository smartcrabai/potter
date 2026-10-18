//! View-curve export through Potter's `.blend` bridge, readback in Blender 5.2.x, and production re-import.

use std::{error::Error, fs};

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/blender_script.rs"]
mod blender_script;
#[path = "common/process.rs"]
mod process;

use blender_checked::blender_executable;
use blender_script::run_blender_script;
use potter_core::{
    color::curve::{CurveMapping, CurvePoint, HandleType},
    eval::EvaluationContext,
    exchange::{
        ExportOptions,
        blend::{export_blend, import_blend},
    },
    model::SceneDoc,
    store::Project,
};
use serde_json::Value;
use tempfile::tempdir;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const SAMPLES: [f32; 7] = [1.5, 2.0, 2.125, 2.375, 2.625, 3.0, 3.5];

const BLENDER_READBACK: &str = r#"
import bpy
if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"curve bridge parity requires Blender 5.2.2, found {bpy.app.version_string}")
import json
import sys
from pathlib import Path

root = Path(sys.argv[sys.argv.index("--") + 1])
bpy.ops.wm.open_mainfile(filepath=str(root / "view_curve.blend"))
view = bpy.context.scene.view_settings
mapping = view.curve_mapping
assert view.use_curve_mapping is True
assert mapping.extend == "HORIZONTAL", mapping.extend
assert mapping.use_clip is True
clip = [float(mapping.clip_min_x), float(mapping.clip_min_y),
        float(mapping.clip_max_x), float(mapping.clip_max_y)]
assert clip == [2.0, 0.0, 3.0, 1.0], clip
assert mapping.tone == "STANDARD", mapping.tone
black = [float(value) for value in mapping.black_level]
white = [float(value) for value in mapping.white_level]
assert black == [0.0, 0.0, 0.0], black
assert white == [1.0, 1.0, 1.0], white
curves = [
    [{"location": [float(point.location[0]), float(point.location[1])],
      "handle": point.handle_type} for point in curve.points]
    for curve in mapping.curves
]
expected_combined = [
    {"location": [2.0, 0.125], "handle": "VECTOR"},
    {"location": [2.5, 0.625], "handle": "AUTO_CLAMPED"},
    {"location": [3.0, 0.875], "handle": "VECTOR"},
]
assert curves[3] == expected_combined, curves[3]
samples = [1.5, 2.0, 2.125, 2.375, 2.625, 3.0, 3.5]
with (root / "view_curve_readback.json").open("w", encoding="utf-8") as stream:
    json.dump({
        "enabled": bool(view.use_curve_mapping),
        "extend": mapping.extend,
        "use_clip": bool(mapping.use_clip),
        "clip": clip,
        "tone": mapping.tone,
        "black": black,
        "white": white,
        "curves": curves,
        "sample_x": samples,
        "sample_values": [float(mapping.evaluate(mapping.curves[3], x)) for x in samples],
    }, stream)
"#;

#[test]
fn view_curve_round_trips_through_blend_and_matches_blender() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping view-curve roundtrip: Blender is unavailable");
        return Ok(());
    };

    let directory = tempdir()?;
    let root = directory.path();
    let project = Project::init(root.join("project"))?;
    let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
    let mut expected_curve = CurveMapping {
        extend_extrapolate: false,
        clip: [2.0, 0.0, 3.0, 1.0],
        ..CurveMapping::default()
    };
    expected_curve.curves[3].points = vec![
        CurvePoint {
            location: [2.0, 0.125],
            handle: HandleType::Vector,
        },
        CurvePoint {
            location: [2.5, 0.625],
            handle: HandleType::AutoClamped,
        },
        CurvePoint {
            location: [3.0, 0.875],
            handle: HandleType::Vector,
        },
    ];
    let scene = doc
        .scenes
        .get_mut(&doc.active_scene)
        .ok_or("new scene document has an active scene")?;
    scene.color_management.curve = Some(expected_curve.clone());

    let blend = root.join("view_curve.blend");
    let context = EvaluationContext::default();
    export_blend(
        &doc,
        &project,
        &blend,
        &ExportOptions {
            allow_lossy: false,
            pack: false,
            blender: Some(blender.as_path()),
            context: &context,
        },
    )?;
    run_blender_script(
        &blender,
        "read_view_curve.py",
        BLENDER_READBACK,
        root,
        "Blender view-curve readback failed",
    )?;
    let blender_data: Value =
        serde_json::from_slice(&fs::read(root.join("view_curve_readback.json"))?)?;

    let imported = import_blend(&blend, Some(blender.as_path()))?;
    let imported_curve = imported.doc.scenes[&imported.doc.active_scene]
        .color_management
        .curve
        .as_ref()
        .ok_or("imported view curve is missing")?;
    assert_eq!(
        imported_curve.extend_extrapolate,
        expected_curve.extend_extrapolate
    );
    assert_eq!(imported_curve.use_clip, expected_curve.use_clip);
    assert_eq!(imported_curve.tone, expected_curve.tone);
    for (actual, expected) in imported_curve.clip.iter().zip(expected_curve.clip) {
        assert!(
            (*actual - expected).abs() <= 1.0e-6,
            "clip {actual} != {expected}"
        );
    }
    for (actual, expected) in imported_curve.black.iter().zip(expected_curve.black) {
        assert!(
            (*actual - expected).abs() <= 1.0e-6,
            "black {actual} != {expected}"
        );
    }
    for (actual, expected) in imported_curve.white.iter().zip(expected_curve.white) {
        assert!(
            (*actual - expected).abs() <= 1.0e-6,
            "white {actual} != {expected}"
        );
    }
    for (actual_curve, expected_map) in imported_curve.curves.iter().zip(&expected_curve.curves) {
        assert_eq!(actual_curve.points.len(), expected_map.points.len());
        for (actual, expected) in actual_curve.points.iter().zip(&expected_map.points) {
            assert_eq!(actual.handle, expected.handle);
            for (actual, expected) in actual.location.iter().zip(expected.location) {
                assert!(
                    (*actual - expected).abs() <= 1.0e-6,
                    "point {actual} != {expected}"
                );
            }
        }
    }

    assert!(
        blender_data["enabled"]
            .as_bool()
            .ok_or("Blender curve is disabled")?
    );
    assert_eq!(blender_data["extend"], "HORIZONTAL");
    assert!(
        blender_data["use_clip"]
            .as_bool()
            .ok_or("Blender curve clip is disabled")?
    );
    assert_eq!(
        blender_data["clip"],
        serde_json::json!([2.0, 0.0, 3.0, 1.0])
    );
    assert_eq!(blender_data["tone"], "STANDARD");
    assert_eq!(blender_data["curves"][3][1]["handle"], "AUTO_CLAMPED");
    let sample_x = blender_data["sample_x"]
        .as_array()
        .ok_or("Blender sample inputs are missing")?;
    let sample_values = blender_data["sample_values"]
        .as_array()
        .ok_or("Blender curve samples are missing")?;
    assert_eq!(sample_x.len(), SAMPLES.len());
    assert_eq!(sample_values.len(), SAMPLES.len());
    let prepared = imported_curve.prepare();
    for (index, x) in SAMPLES.into_iter().enumerate() {
        let blender_x = sample_x[index]
            .as_f64()
            .ok_or("Blender sample input is not a number")?;
        assert!((blender_x - f64::from(x)).abs() <= f64::EPSILON);
        let blender_value = sample_values[index]
            .as_f64()
            .ok_or("Blender curve sample is not a number")?;
        let potter_value = f64::from(prepared.evaluate(3, x));
        assert!(
            (potter_value - blender_value).abs() <= 1.0e-5,
            "combined curve at x={x}: Potter={potter_value}, Blender={blender_value}"
        );
    }
    Ok(())
}
