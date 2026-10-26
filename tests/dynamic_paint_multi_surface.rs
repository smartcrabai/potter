#![expect(
    clippy::unwrap_used,
    reason = "The committed Blender oracle uses fixed valid test data"
)]

use std::{
    collections::BTreeMap,
    error::Error,
    fs, io,
    path::PathBuf,
    process::{Command, Output},
};

use serde_json::Value;
use tempfile::tempdir;

#[path = "common/blender_file.rs"]
mod blender_file;
use blender_file::blender_executable;

const ORACLE: &str = include_str!("fixtures/dynamic_paint_multi_surface/oracle.json");
const COLOR_TOLERANCE: f64 = 1.0 / 255.0 + 1.0e-6;
const WEIGHT_TOLERANCE: f64 = 1.0e-5;

type TestResult<T> = Result<T, Box<dyn Error>>;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn checked_output(command: &mut Command, label: &str) -> TestResult<Output> {
    let output = command.output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "{label} failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        ))
        .into());
    }
    Ok(output)
}

fn run_pot_json(command: &mut Command, label: &str) -> TestResult<Value> {
    let output = checked_output(command, label)?;
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn compare_arrays(
    expected: &Value,
    actual: &Value,
    tolerance: f64,
    context: &str,
) -> TestResult<()> {
    let expected = expected
        .as_array()
        .ok_or_else(|| io::Error::other(format!("{context}: oracle array missing")))?;
    let actual = actual
        .as_array()
        .ok_or_else(|| io::Error::other(format!("{context}: Potter array missing")))?;
    assert_eq!(expected.len(), actual.len(), "{context}: value count");
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        match (expected.as_array(), actual.as_array()) {
            (Some(expected), Some(actual)) => {
                assert_eq!(expected.len(), actual.len(), "{context}[{index}]: channels");
                for (channel, (expected, actual)) in expected.iter().zip(actual).enumerate() {
                    compare_number(
                        expected,
                        actual,
                        tolerance,
                        &format!("{context}[{index}][{channel}]"),
                    )?;
                }
            }
            (None, None) => {
                compare_number(expected, actual, tolerance, &format!("{context}[{index}]"))?;
            }
            _ => return Err(format!("{context}[{index}]: value shape differs").into()),
        }
    }
    Ok(())
}

fn compare_number(
    expected: &Value,
    actual: &Value,
    tolerance: f64,
    context: &str,
) -> TestResult<()> {
    let expected = expected
        .as_f64()
        .ok_or_else(|| io::Error::other(format!("{context}: oracle number missing")))?;
    let actual = actual
        .as_f64()
        .ok_or_else(|| io::Error::other(format!("{context}: Potter number missing")))?;
    assert!(
        (expected - actual).abs() <= tolerance,
        "{context}: expected {expected}, got {actual} (tolerance {tolerance})"
    );
    Ok(())
}

fn linear_to_srgb(value: f64) -> f64 {
    if value < 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

fn assert_corner_layer(
    mesh: &Value,
    name: &str,
    expected: &Value,
    context: &str,
) -> TestResult<()> {
    let vertices = mesh["vertices"]
        .as_array()
        .ok_or_else(|| io::Error::other(format!("{context}: mesh vertices missing")))?;
    let vertex_indices: BTreeMap<u64, usize> = vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex["id"].as_u64().unwrap(), index))
        .collect();
    let attribute = &mesh["attributes"][name];
    assert_eq!(attribute["domain"], "corner", "{context}: domain");
    let values = attribute["values"]
        .as_object()
        .ok_or_else(|| io::Error::other(format!("{context}: corner values missing")))?;
    for face in mesh["faces"].as_array().ok_or("mesh faces are missing")? {
        let face_id = face["id"].as_u64().unwrap();
        let corners = values[&format!("f{face_id}")]
            .as_array()
            .ok_or_else(|| io::Error::other(format!("{context}: face {face_id} values missing")))?;
        for (corner, vertex_id) in face["v"].as_array().unwrap().iter().enumerate() {
            let vertex_id = vertex_id.as_u64().unwrap();
            let vertex_index = vertex_indices[&vertex_id];
            compare_arrays(
                &Value::Array(vec![expected[vertex_index].clone()]),
                &Value::Array(vec![corners[corner].clone()]),
                COLOR_TOLERANCE,
                &format!("{context}, face {face_id}, corner {corner}"),
            )?;
        }
    }
    Ok(())
}

fn assert_weight_group(
    mesh: &Value,
    name: &str,
    expected: &Value,
    context: &str,
) -> TestResult<()> {
    let vertices = mesh["vertices"]
        .as_array()
        .ok_or_else(|| io::Error::other(format!("{context}: mesh vertices missing")))?;
    let values = mesh["attributes"]["vertex_groups"][name]
        .as_object()
        .ok_or_else(|| io::Error::other(format!("{context}: vertex group `{name}` missing")))?;
    let expected = expected
        .as_array()
        .ok_or_else(|| io::Error::other(format!("{context}: oracle weights missing")))?;
    assert_eq!(vertices.len(), expected.len(), "{context}: vertex count");
    for (index, vertex) in vertices.iter().enumerate() {
        let vertex_id = vertex["id"].as_u64().unwrap();
        let actual = values
            .get(&format!("v{vertex_id}"))
            .cloned()
            .unwrap_or(Value::from(0.0));
        compare_number(
            &expected[index],
            &actual,
            WEIGHT_TOLERANCE,
            &format!("{context}, vertex {index}"),
        )?;
    }
    Ok(())
}

#[test]
fn mixed_dynamic_paint_surfaces_match_blender_per_frame() -> TestResult<()> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping mixed Dynamic Paint oracle; Blender is unavailable");
        return Ok(());
    };
    let oracle: Value = serde_json::from_str(ORACLE)?;
    assert_eq!(oracle["blender_version"], "5.2.2 LTS");
    let directory = tempdir()?;
    let root = directory.path();
    let generated = root.join("generated");
    let generator = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/dynamic_paint_multi_surface/generate.py");
    checked_output(
        Command::new(&blender)
            .args(["-b", "--factory-startup", "--threads", "1", "--python"])
            .arg(generator)
            .arg("--")
            .arg(&generated),
        "Blender mixed Dynamic Paint oracle generator",
    )?;
    let generated_oracle: Value =
        serde_json::from_slice(&fs::read(generated.join("oracle.json"))?)?;
    assert_eq!(
        generated_oracle, oracle,
        "Blender capture differs from committed oracle"
    );

    let project = root.join("project");
    checked_output(pot().args(["init"]).arg(&project).arg("--json"), "pot init")?;
    let source_blend = generated.join("DynamicPaintMultiSurface.blend");
    run_pot_json(
        pot()
            .args(["import"])
            .arg(&project)
            .args(["--file"])
            .arg(&source_blend)
            .args([
                "--format",
                "blend",
                "--allow-lossy",
                "--mode",
                "replace",
                "--base-revision",
                "0",
                "--blender",
            ])
            .arg(&blender)
            .arg("--json"),
        "pot import mixed Dynamic Paint fixture",
    )?;
    let scene: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let nodes = scene["nodes"].as_object().ok_or("project nodes missing")?;
    let (canvas_id, canvas) = nodes
        .iter()
        .find(|(_, node)| node["name"] == "MultiSurfaceCanvas")
        .ok_or("imported multi-surface canvas missing")?;
    let surfaces = canvas["properties"]["physics_dynamic_paint"]["canvas_surfaces"]
        .as_array()
        .ok_or("imported canvas surfaces missing")?;
    assert_eq!(surfaces.len(), 3);
    for (actual, expected) in surfaces.iter().zip(oracle["surfaces"].as_array().unwrap()) {
        for key in [
            "surface_type",
            "surface_format",
            "is_active",
            "frame_start",
            "frame_end",
            "output_name_a",
            "output_name_b",
            "use_output_a",
            "use_output_b",
        ] {
            assert_eq!(actual[key], expected[key], "surface setting {key}");
        }
        assert!(actual["brush_collection"].as_str().is_some());
    }

    let bake_dir = root.join("bake");
    let bake = run_pot_json(
        pot()
            .args(["bake"])
            .arg(&project)
            .args(["--kind", "simulation", "--frames", "1:4", "--out"])
            .arg(&bake_dir)
            .arg("--json"),
        "pot bake mixed Dynamic Paint surfaces",
    )?;
    assert_eq!(bake["result"]["frame_count"], 4);
    let manifest: Value = serde_json::from_slice(&fs::read(bake_dir.join("manifest.json"))?)?;
    assert_eq!(manifest["schema_version"], 2);
    assert!(
        manifest["frames"]
            .as_array()
            .ok_or("bake manifest frame summaries missing")?
            .iter()
            .all(|frame| frame["painted_objects"].as_u64() == Some(1)),
        "multiple surfaces on one canvas count as one painted object"
    );

    for frame_number in 1..=4 {
        let frame: Value = serde_json::from_slice(&fs::read(
            bake_dir.join(format!("frame_{:08}.json", frame_number - 1)),
        )?)?;
        let frame_key = frame_number.to_string();
        let expected_frame = &oracle["frames"][&frame_key];
        let canvas_id = canvas_id.as_str();
        let paint_colors = &frame["paint_colors"][canvas_id]["0"];
        compare_arrays(
            &expected_frame["colors"]["MultiPaint"],
            paint_colors,
            COLOR_TOLERANCE,
            &format!("frame {frame_number} surface 0 PAINT colors"),
        )?;
        let expected_wetness = expected_frame["colors"]["MultiWetness"]
            .as_array()
            .ok_or("Blender wetness output missing")?
            .iter()
            .map(|color| linear_to_srgb(color[0].as_f64().unwrap()))
            .map(Value::from)
            .collect::<Vec<_>>();
        compare_arrays(
            &Value::Array(expected_wetness),
            &frame["paint_wetness"][canvas_id]["0"],
            COLOR_TOLERANCE,
            &format!("frame {frame_number} surface 0 wetness"),
        )?;
        compare_arrays(
            &expected_frame["weights"],
            &frame["paint_weights"][canvas_id]["1"],
            WEIGHT_TOLERANCE,
            &format!("frame {frame_number} surface 1 WEIGHT values"),
        )?;
        assert!(frame["paint_colors"][canvas_id].get("2").is_none());
        assert!(frame["paint_wetness"][canvas_id].get("2").is_none());
        assert!(frame["paint_weights"][canvas_id].get("2").is_none());

        let mesh = &frame["deformed_meshes"][canvas_id];
        assert_corner_layer(
            mesh,
            "MultiPaint",
            &expected_frame["colors"]["MultiPaint"],
            &format!("frame {frame_number} PAINT output A"),
        )?;
        assert_corner_layer(
            mesh,
            "MultiWetness",
            &expected_frame["colors"]["MultiWetness"],
            &format!("frame {frame_number} PAINT output B"),
        )?;
        assert_corner_layer(
            mesh,
            "InactivePaint",
            &expected_frame["colors"]["InactivePaint"],
            &format!("frame {frame_number} inactive output"),
        )?;
        assert_weight_group(
            mesh,
            "MultiWeights",
            &expected_frame["weights"],
            &format!("frame {frame_number} WEIGHT output"),
        )?;
    }
    let center_paint = oracle["frames"]["1"]["colors"]["MultiPaint"][4][3]
        .as_f64()
        .ok_or("paint oracle center alpha missing")?;
    let right_paint = oracle["frames"]["1"]["colors"]["MultiPaint"][5][3]
        .as_f64()
        .ok_or("paint oracle right alpha missing")?;
    let center_weight = oracle["frames"]["2"]["weights"][4]
        .as_f64()
        .ok_or("weight oracle center value missing")?;
    let right_weight = oracle["frames"]["2"]["weights"][5]
        .as_f64()
        .ok_or("weight oracle right value missing")?;
    assert!(center_paint > 0.0);
    assert_eq!(right_paint, 0.0);
    assert_eq!(center_weight, 0.0);
    assert!(right_weight > 0.0);
    Ok(())
}
