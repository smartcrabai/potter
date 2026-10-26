//! Blender 5.2.2 oracle for Dynamic Paint WEIGHT surface history and named-group stack output.

use std::{
    collections::BTreeMap,
    error::Error,
    fs, io,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use serde_json::{Value, json};
use tempfile::tempdir;

#[path = "common/blender_file.rs"]
mod blender_file;
use blender_file::blender_executable;

const ORACLE: &str = include_str!("fixtures/dynamic_paint_weight/oracle.json");
const WEIGHT_TOLERANCE: f64 = 1.0e-5;
const DISSOLVE_SPEED: u64 = 4;

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

fn inspect(project: &Path, frame: u64) -> TestResult<Value> {
    run_pot_json(
        pot()
            .args(["inspect"])
            .arg(project)
            .args(["--frame"])
            .arg(frame.to_string())
            .arg("--json"),
        "pot inspect Dynamic Paint frame",
    )
}

fn number(value: &Value, context: &str) -> TestResult<f64> {
    value.as_f64().ok_or_else(|| {
        io::Error::other(format!("{context}: expected a number, got {value}")).into()
    })
}

fn assert_close(expected: f64, actual: f64, tolerance: f64, context: &str) {
    assert!(
        (expected - actual).abs() <= tolerance,
        "{context}: expected {expected}, got {actual} (tolerance {tolerance})"
    );
}

fn assert_numeric_arrays(
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
    assert_eq!(
        expected.len(),
        actual.len(),
        "{context}: array lengths differ"
    );
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        match (expected.as_array(), actual.as_array()) {
            (Some(expected), Some(actual)) => {
                assert_eq!(
                    expected.len(),
                    actual.len(),
                    "{context}[{index}] width differs"
                );
                for (axis, (expected, actual)) in expected.iter().zip(actual).enumerate() {
                    assert_close(
                        number(expected, context)?,
                        number(actual, context)?,
                        tolerance,
                        &format!("{context}[{index}][{axis}]"),
                    );
                }
            }
            (None, None) => assert_close(
                number(expected, context)?,
                number(actual, context)?,
                tolerance,
                &format!("{context}[{index}]"),
            ),
            _ => return Err(format!("{context}[{index}]: array shape differs").into()),
        }
    }
    Ok(())
}

fn assert_source_group(project: &Path, oracle_canvas: &Value) -> TestResult<String> {
    let document: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let nodes = document["nodes"]
        .as_object()
        .ok_or("imported project nodes are missing")?;
    let (canvas_id, canvas) = nodes
        .iter()
        .find(|(_, node)| node["name"] == oracle_canvas["canvas"])
        .ok_or("pot import did not retain the Dynamic Paint canvas")?;
    let data_id = canvas["data"]
        .as_str()
        .ok_or("imported Dynamic Paint canvas has no mesh data")?;
    let data = document["data_blocks"]
        .get(data_id)
        .ok_or("imported Dynamic Paint canvas mesh data is missing")?;
    let groups = data["vertex_groups"]
        .as_array()
        .ok_or("imported source vertex groups are missing")?;
    let group = groups
        .iter()
        .find(|group| group["name"] == oracle_canvas["group"])
        .ok_or("pot import did not retain the named source vertex group")?;
    let group_id = group["id"]
        .as_str()
        .ok_or("imported source vertex group has no ID")?;
    let source_weights = oracle_canvas["source_group_weights"]
        .as_object()
        .ok_or("oracle source group weights are missing")?;
    let vertex_weights = data["vertex_weights"]
        .as_object()
        .ok_or("imported source vertex weights are missing")?;
    for (vertex_key, expected_weight) in source_weights {
        let vertex_id = vertex_key
            .as_str()
            .strip_prefix('v')
            .ok_or("oracle source vertex key is malformed")?;
        let actual_weight = vertex_weights
            .get(vertex_id)
            .and_then(|weights| weights.get(group_id))
            .ok_or_else(|| format!("pot import lost source assignment {vertex_key}"))?;
        assert_eq!(
            number(expected_weight, "oracle source vertex weight")?,
            number(actual_weight, "imported source vertex weight")?,
            "pot import changed source assignment {vertex_key}"
        );
    }
    Ok(canvas_id.clone())
}
fn compare_bake_frame(
    frame: &Value,
    oracle_frame: &Value,
    canvas_id: &str,
    group_name: &str,
    source_group_weights: &Value,
    frame_number: u64,
) -> TestResult<()> {
    assert_numeric_arrays(
        &oracle_frame["weights"],
        &frame["paint_weights"][canvas_id]["0"],
        WEIGHT_TOLERANCE,
        &format!("frame {frame_number} paint_weights"),
    )?;

    let mesh = &frame["deformed_meshes"][canvas_id];
    let vertices = mesh["vertices"].as_array().ok_or_else(|| {
        io::Error::other(format!("frame {frame_number} deformed vertices missing"))
    })?;
    let vertex_groups = mesh["attributes"]["vertex_groups"]
        .as_object()
        .ok_or_else(|| io::Error::other(format!("frame {frame_number} vertex groups missing")))?;
    let actual_group = vertex_groups
        .get(group_name)
        .and_then(Value::as_object)
        .ok_or_else(|| {
            io::Error::other(format!("frame {frame_number} group `{group_name}` missing"))
        })?;
    let expected_weights = oracle_frame["edited_weights"]
        .as_array()
        .ok_or("oracle Vertex Weight Edit group weights missing")?;
    assert_eq!(
        vertices.len(),
        expected_weights.len(),
        "frame {frame_number} edited vertex count differs"
    );
    for (vertex, expected) in vertices.iter().zip(expected_weights) {
        let vertex_id = vertex["id"]
            .as_u64()
            .ok_or_else(|| io::Error::other(format!("frame {frame_number} vertex ID missing")))?;
        let key = format!("v{vertex_id}");
        let actual = actual_group
            .get(&key)
            .map(|weight| number(weight, "deformed vertex-group weight"))
            .transpose()?
            .unwrap_or(0.0);
        assert_close(
            number(expected, "oracle frame vertex-group weight")?,
            actual,
            WEIGHT_TOLERANCE,
            &format!("frame {frame_number} group weight {key}"),
        );
    }
    for (vertex_key, expected) in source_group_weights
        .as_object()
        .ok_or("oracle source group weights are missing")?
    {
        let vertex_id = vertex_key
            .as_str()
            .strip_prefix('v')
            .ok_or("oracle source vertex key is malformed")?;
        let key = format!("v{vertex_id}");
        let actual = actual_group
            .get(&key)
            .ok_or_else(|| format!("frame {frame_number} lost prior zero assignment {key}"))?;
        assert_eq!(
            number(expected, "oracle source group weight")?,
            number(actual, "baked prior group weight")?,
            "frame {frame_number} prior group assignment {key} changed"
        );
    }
    Ok(())
}

fn compare_inspect_frame(
    inspected: &Value,
    oracle_frame: &Value,
    canvas_id: &str,
    group_name: &str,
    frame_number: u64,
) -> TestResult<()> {
    assert_close(
        frame_number as f64,
        number(&inspected["result"]["evaluation"]["frame"], "inspect frame")?,
        1.0e-9,
        "inspect returned the wrong frame",
    );
    let item = inspected["result"]["items"]
        .as_array()
        .and_then(|items| items.iter().find(|item| item["id"] == canvas_id))
        .ok_or_else(|| format!("pot inspect omitted canvas at frame {frame_number}"))?;
    assert_numeric_arrays(
        &oracle_frame["edited_weights"],
        &item["evaluated_geometry"]["vertex_groups"][group_name],
        WEIGHT_TOLERANCE,
        &format!("frame {frame_number} inspected vertex-weight edit result"),
    )
}
#[test]
fn dynamic_paint_weight_bake_matches_blender_history_and_vertex_weight_edit_stack() -> TestResult<()>
{
    let blender = blender_executable().ok_or("Blender 5.2.2 is required for this oracle test")?;
    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dynamic_paint_weight");
    let directory = tempdir()?;
    let generated = directory.path().join("generated");
    let generated_capture = generated.join("oracle.json");
    let source_blend = generated.join("source.blend");
    let generator = fixture.join("generate_oracle.py");

    checked_output(
        Command::new(&blender)
            .args(["-b", "--factory-startup", "--threads", "1", "--python"])
            .arg(generator)
            .arg("--")
            .arg(&generated),
        "Blender Dynamic Paint oracle generator",
    )?;
    let generated_oracle: Value = serde_json::from_slice(&fs::read(generated_capture)?)?;
    let committed_oracle: Value = serde_json::from_str(ORACLE)?;
    assert_eq!(
        generated_oracle, committed_oracle,
        "Blender's generated Dynamic Paint capture differs from committed oracle"
    );
    assert_eq!(committed_oracle["blender_version"], "5.2.2 LTS");

    let canvases = committed_oracle["canvases"]
        .as_object()
        .ok_or("committed Blender canvas oracles are missing")?;
    let linear = &canvases["CanvasHistoryLinear"];
    let history_raw = &canvases["CanvasHistoryRaw"];
    let log = &canvases["CanvasHistoryLog"];
    let erase = &canvases["CanvasErase"];
    let no_erase = &canvases["CanvasNoErase"];
    let smooth = &canvases["CanvasSmooth"];
    let unit_scale = &canvases["CanvasUnitScale"];
    assert_eq!(linear["surface"]["dissolve_speed"], DISSOLVE_SPEED);
    assert_eq!(linear["surface"]["use_dissolve_log"], false);
    assert_eq!(log["surface"]["dissolve_speed"], DISSOLVE_SPEED);
    assert_eq!(log["surface"]["use_dissolve_log"], true);
    assert_ne!(
        linear["frames"]["3"]["weights"], log["frames"]["3"]["weights"],
        "linear and logarithmic dissolve must produce distinct history"
    );
    assert_eq!(linear["downstream_modifier"]["type"], "VERTEX_WEIGHT_EDIT");
    assert!(history_raw["downstream_modifier"].is_null());
    assert_eq!(
        linear["frames"]["3"]["weights"],
        history_raw["frames"]["3"]["weights"]
    );
    let frame1_weights = history_raw["frames"]["1"]["weights"]
        .as_array()
        .ok_or("history control frame 1 weights are missing")?;
    let trail_index = frame1_weights
        .iter()
        .position(|weight| weight.as_f64().is_some_and(|weight| weight > 0.0))
        .ok_or("history control frame 1 has no painted trail")?;
    let frame3_weights = history_raw["frames"]["3"]["weights"]
        .as_array()
        .ok_or("history control frame 3 weights are missing")?;
    assert!(
        number(
            frame3_weights
                .get(trail_index)
                .ok_or("history control trail vertex is missing in frame 3")?,
            "history control frame 3 trail weight",
        )? > 0.0,
        "linear dissolve speed 4 must retain the frame-1 trail in frame 3"
    );
    assert_close(
        0.8,
        number(
            &no_erase["surface"]["brush_radius_scale"],
            "scaled brush radius",
        )?,
        1.0e-6,
        "scaled brush radius",
    );
    assert_close(
        0.55,
        number(
            &no_erase["surface"]["brush_influence_scale"],
            "scaled brush influence",
        )?,
        1.0e-6,
        "scaled brush influence",
    );
    assert_close(
        1.0,
        number(
            &unit_scale["surface"]["brush_radius_scale"],
            "unit brush radius",
        )?,
        1.0e-6,
        "unit brush radius",
    );
    assert_close(
        1.0,
        number(
            &unit_scale["surface"]["brush_influence_scale"],
            "unit brush influence",
        )?,
        1.0e-6,
        "unit brush influence",
    );
    assert_ne!(
        linear["frames"]["1"]["weights"], linear["frames"]["8"]["weights"],
        "linear dissolve history must change the paint weights"
    );
    assert_ne!(
        log["frames"]["1"]["weights"], log["frames"]["8"]["weights"],
        "logarithmic dissolve history must change the paint weights"
    );
    assert_ne!(
        erase["frames"]["2"]["weights"], no_erase["frames"]["2"]["weights"],
        "the oracle's eraser must change WEIGHT output"
    );
    assert_ne!(
        no_erase["frames"]["1"]["weights"], smooth["frames"]["1"]["weights"],
        "the oracle's CONSTANT falloff must differ from SMOOTH"
    );
    assert_ne!(
        no_erase["frames"]["1"]["weights"], unit_scale["frames"]["1"]["weights"],
        "the oracle's non-unit brush radius/influence scales must change WEIGHT output"
    );
    assert!(
        no_erase["frames"]["1"]["weights"]
            .as_array()
            .ok_or("oracle point brush weights are missing")?
            .iter()
            .any(|weight| weight.as_f64().is_some_and(|weight| weight > 0.0)),
        "the parented, delta-offset POINT brush must paint at its local location"
    );
    let volume = &canvases["CanvasVolume"];
    let volume_brush = &committed_oracle["brushes"]["volume"];
    assert_eq!(volume_brush["source"], "VOLUME");
    assert_eq!(volume_brush["proximity_falloff"], "RAMP");
    assert_close(
        0.001,
        number(&volume_brush["distance"], "volume paint distance")?,
        1.0e-6,
        "volume paint distance is ignored",
    );
    assert_eq!(volume_brush["locations"]["1"], json!([-2.0, 0.0, 0.0]));
    assert_eq!(volume_brush["locations"]["2"], json!([-1.0, 0.0, 0.0]));
    let volume_frame1 = volume["frames"]["1"]["weights"]
        .as_array()
        .ok_or("volume frame 1 weights are missing")?;
    assert!(
        volume_frame1
            .iter()
            .any(|weight| weight.as_f64().is_some_and(|weight| weight > 0.0))
            && volume_frame1.iter().any(|weight| weight
                .as_f64()
                .is_some_and(|weight| weight.abs() <= WEIGHT_TOLERANCE)),
        "the VOLUME brush must cover only part of the WEIGHT canvas"
    );
    assert_ne!(
        volume["frames"]["1"]["weights"], volume["frames"]["2"]["weights"],
        "moving the VOLUME brush mesh must change its coverage"
    );
    let volume_max = volume_frame1
        .iter()
        .filter_map(Value::as_f64)
        .fold(0.0_f64, f64::max);
    let expected_volume_strength = number(&volume_brush["paint_alpha"], "volume brush alpha")?
        * number(
            &volume["surface"]["brush_influence_scale"],
            "volume surface influence scale",
        )?;
    assert_close(
        expected_volume_strength,
        volume_max,
        WEIGHT_TOLERANCE,
        "VOLUME uses full inside influence before brush alpha and surface scaling",
    );

    let point_brush = &committed_oracle["brushes"]["point"];
    let erase_brush = &committed_oracle["brushes"]["erase"];
    assert_eq!(point_brush["source"], "POINT");
    assert_eq!(point_brush["proximity_falloff"], "CONSTANT");
    assert_eq!(point_brush["parent"], "BrushParent");
    assert_ne!(point_brush["parent_location"], json!([0.0, 0.0, 0.0]));
    assert_ne!(point_brush["delta_location"], json!([0.0, 0.0, 0.0]));
    assert_eq!(erase_brush["use_paint_erase"], true);
    assert_eq!(erase_brush["proximity_falloff"], "CONSTANT");
    let project = directory.path().join("dynamic-paint-weight-project");
    checked_output(pot().args(["init"]).arg(&project).arg("--json"), "pot init")?;
    let imported = run_pot_json(
        pot()
            .args(["import"])
            .arg(&project)
            .args(["--file"])
            .arg(&source_blend)
            .args([
                "--format",
                "blend",
                "--mode",
                "replace",
                "--base-revision",
                "0",
                "--blender",
            ])
            .arg(&blender)
            .arg("--json"),
        "strict pot import Blender Dynamic Paint fixture",
    )?;
    let volume_brush_losses = [
        "blender.physics.dynamic_paint_brush.paint_source",
        "blender.physics.dynamic_paint_brush.paint_alpha",
        "blender.physics.dynamic_paint_brush.paint_distance",
        "blender.physics.dynamic_paint_brush.proximity_falloff",
    ];
    for loss in imported["result"]["losses"]
        .as_array()
        .into_iter()
        .flatten()
    {
        let feature = loss["feature_id"].as_str().unwrap_or_default();
        assert!(
            !volume_brush_losses.contains(&feature),
            "supported VOLUME brush loss: {loss}"
        );
    }

    let roundtrip_blend = directory
        .path()
        .join("dynamic-paint-weight-roundtrip.blend");
    checked_output(
        pot()
            .args(["export"])
            .arg(&project)
            .args(["--format", "blend", "--out"])
            .arg(&roundtrip_blend)
            .args(["--blender"])
            .arg(&blender)
            .arg("--json"),
        "pot export Dynamic Paint WEIGHT fixture",
    )?;
    let roundtrip_project = directory.path().join("dynamic-paint-weight-roundtrip");
    checked_output(
        pot().args(["init"]).arg(&roundtrip_project).arg("--json"),
        "pot init round-trip project",
    )?;
    checked_output(
        pot()
            .args(["import"])
            .arg(&roundtrip_project)
            .args(["--file"])
            .arg(&roundtrip_blend)
            .args([
                "--format",
                "blend",
                "--mode",
                "replace",
                "--base-revision",
                "0",
                "--blender",
            ])
            .arg(&blender)
            .arg("--json"),
        "strict pot reimport exported Dynamic Paint WEIGHT fixture",
    )?;

    // Strict reimport must preserve each named output group and its surface configuration.
    let document: Value = serde_json::from_slice(&fs::read(roundtrip_project.join("scene.json"))?)?;
    let nodes = document["nodes"]
        .as_object()
        .ok_or("round-tripped project nodes are missing")?;
    let mut canvas_ids = BTreeMap::new();
    for (name, canvas_oracle) in canvases {
        let canvas_id = assert_source_group(&roundtrip_project, canvas_oracle)?;
        let canvas = nodes
            .get(&canvas_id)
            .ok_or("round-tripped canvas node is missing")?;
        let surface = &canvas["properties"]["physics_dynamic_paint"]["canvas_surfaces"][0];
        let expected_surface = &canvas_oracle["surface"];
        assert_eq!(surface["surface_type"], expected_surface["type"]);
        assert_eq!(surface["surface_format"], expected_surface["format"]);
        assert_eq!(surface["output_name_a"], expected_surface["output_name"]);
        assert_eq!(surface["use_dissolve"], expected_surface["use_dissolve"]);
        assert_eq!(
            surface["dissolve_speed"],
            expected_surface["dissolve_speed"]
        );
        assert_eq!(
            surface["use_dissolve_log"],
            expected_surface["use_dissolve_log"]
        );
        assert_eq!(
            surface["brush_radius_scale"],
            expected_surface["brush_radius_scale"]
        );
        assert_eq!(
            surface["brush_influence_scale"],
            expected_surface["brush_influence_scale"]
        );
        canvas_ids.insert(name.clone(), canvas_id);
    }
    for brush_oracle in committed_oracle["brushes"]
        .as_object()
        .ok_or("oracle Dynamic Paint brushes are missing")?
        .values()
    {
        let brush = nodes
            .values()
            .find(|node| node["name"] == brush_oracle["name"])
            .ok_or("round-tripped Dynamic Paint brush is missing")?;
        let settings = &brush["properties"]["physics_dynamic_paint"];
        assert_eq!(settings["paint_source"], brush_oracle["source"]);
        assert_eq!(settings["paint_distance"], brush_oracle["distance"]);
        assert_eq!(
            settings["proximity_falloff"],
            brush_oracle["proximity_falloff"]
        );
        assert_eq!(settings["use_paint_erase"], brush_oracle["use_paint_erase"]);
    }

    let point_node = nodes
        .values()
        .find(|node| node["name"] == point_brush["name"])
        .ok_or("round-tripped POINT brush is missing")?;
    let (parent_id, parent_node) = nodes
        .iter()
        .find(|(_, node)| node["name"] == point_brush["parent"])
        .ok_or("round-tripped POINT brush parent is missing")?;
    assert_eq!(point_node["parent"], parent_id.as_str());
    assert_numeric_arrays(
        &point_brush["parent_location"],
        &parent_node["transform"]["translation"],
        WEIGHT_TOLERANCE,
        "round-tripped brush parent location",
    )?;
    assert_numeric_arrays(
        &point_brush["delta_location"],
        &point_node["properties"]["blender_object_metadata"]["delta_location"],
        WEIGHT_TOLERANCE,
        "round-tripped POINT brush delta location",
    )?;

    let frame_start = committed_oracle["frame_start"]
        .as_u64()
        .ok_or("oracle frame start is missing")?;
    let frame_end = committed_oracle["frame_end"]
        .as_u64()
        .ok_or("oracle frame end is missing")?;
    let bake_dir = directory.path().join("bake");
    let bake = run_pot_json(
        pot()
            .args(["bake"])
            .arg(&roundtrip_project)
            .args(["--kind", "simulation", "--frames"])
            .arg(format!("{frame_start}:{frame_end}"))
            .args(["--out"])
            .arg(&bake_dir)
            .arg("--json"),
        "pot bake Dynamic Paint WEIGHT frames",
    )?;
    assert_eq!(bake["result"]["frame_count"], frame_end - frame_start + 1);

    for frame_number in frame_start..=frame_end {
        let frame_key = frame_number.to_string();
        let frame_path = bake_dir.join(format!("frame_{:08}.json", frame_number - frame_start));
        let baked_frame: Value = serde_json::from_slice(&fs::read(frame_path)?)?;
        let inspected = inspect(&roundtrip_project, frame_number)?;
        for (name, canvas_oracle) in canvases {
            let canvas_id = &canvas_ids[name];
            let oracle_frame = canvas_oracle["frames"]
                .get(&frame_key)
                .ok_or_else(|| format!("oracle canvas {name} frame {frame_number} is missing"))?;
            let group_name = canvas_oracle["group"]
                .as_str()
                .ok_or("oracle vertex group name is missing")?;
            compare_bake_frame(
                &baked_frame,
                oracle_frame,
                canvas_id,
                group_name,
                &canvas_oracle["source_group_weights"],
                frame_number,
            )?;
            compare_inspect_frame(
                &inspected,
                oracle_frame,
                canvas_id,
                group_name,
                frame_number,
            )?;
        }
    }
    Ok(())
}

#[test]
fn dynamic_paint_weight_ops_accept_supported_brush_settings() -> TestResult<()> {
    let oracle: Value = serde_json::from_str(ORACLE)?;
    let canvas_oracle = &oracle["canvases"]["CanvasErase"];
    let directory = tempdir()?;
    let project = directory.path().join("dynamic-paint-weight-ops");
    checked_output(
        pot().args(["init"]).arg(&project).arg("--json"),
        "pot init ops-only Dynamic Paint project",
    )?;

    let batch_path = directory.path().join("operations.json");
    let batch = json!({
        "schema_version": 1,
        "base_revision": 0,
        "operations": [
            {
                "op":"node.create",
                "id":"canvas",
                "kind":"grid",
                "params":{"size":4.0,"x_subdivisions":8,"y_subdivisions":4}
            },
            {
                "op":"mesh.transform_elements",
                "target":{"id":"canvas","elements":{
                    "domain":"vertex",
                    "ids":(0..45).map(|index|format!("v{index}")).collect::<Vec<_>>()
                }},
                "scale":[1.0,0.5,1.0],
                "pivot":[0.0,0.0,0.0]
            },
            {"op":"vertex_group.create","target":{"id":"canvas"},"id":"weight_group","name":"EraseWeight"},
            {"op":"vertex_group.assign","target":{"id":"canvas"},"group_id":"weight_group","weights":[{"vertex_id":44,"weight":0.0}]},
            {
                "op":"node.create",
                "id":"point_brush",
                "kind":"box",
                "params":{"size":0.1},
                "transform":{"translation":[-2.0,0.0,0.25]}
            },
            {
                "op":"node.create",
                "id":"erase_brush",
                "kind":"box",
                "params":{"size":0.1},
                "transform":{"translation":[-5.0,0.0,0.25]}
            },
            {"op":"action.create","id":"point_motion","name":"Point Brush Motion"},
            {"op":"action.update","target":{"id":"point_motion"},"set":{"fcurves":[{"path":"transform.translation","index":0,"keyframes":[{"frame":1.0,"value":-2.0},{"frame":2.0,"value":-1.0}]}]}},
            {"op":"node.update","target":{"id":"point_brush"},"set":{"action":"point_motion"}},
            {"op":"action.create","id":"erase_motion","name":"Erase Brush Motion"},
            {"op":"action.update","target":{"id":"erase_motion"},"set":{"fcurves":[{"path":"transform.translation","index":0,"keyframes":[{"frame":1.0,"value":-5.0},{"frame":2.0,"value":-2.0}]}]}},
            {"op":"node.update","target":{"id":"erase_brush"},"set":{"action":"erase_motion"}},
            {
                "op":"physics.dynamic_paint.create",
                "target":{"id":"point_brush"},
                "settings":{"role":"brush","paint_source":"POINT","paint_distance":1.0,"proximity_falloff":"CONSTANT"}
            },
            {"op":"physics.dynamic_paint.create","target":{"id":"erase_brush"},"settings":{"role":"brush"}},
            {
                "op":"physics.dynamic_paint.update",
                "target":{"id":"erase_brush"},
                "set":{"paint_source":"POINT","paint_distance":0.4,"proximity_falloff":"CONSTANT","use_paint_erase":true}
            },
            {
                "op":"physics.dynamic_paint.create",
                "target":{"id":"canvas"},
                "settings":{
                    "role":"canvas",
                    "brushes":["point_brush","erase_brush"],
                    "surface_format":"weight",
                    "canvas_surfaces":[{
                        "surface_type":"WEIGHT",
                        "surface_format":"VERTEX",
                        "output_name_a":"EraseWeight",
                        "frame_start":1,
                        "frame_end":8,
                        "use_dissolve":true,
                        "dissolve_speed":4,
                        "use_dissolve_log":false,
                        "brush_radius_scale":0.8,
                        "brush_influence_scale":0.55
                    }]
                }
            }
        ]
    });
    fs::write(&batch_path, serde_json::to_vec(&batch)?)?;
    let applied = run_pot_json(
        pot()
            .args(["apply"])
            .arg(&project)
            .args(["--file"])
            .arg(&batch_path)
            .arg("--json"),
        "pot apply ops-only Dynamic Paint settings",
    )?;
    assert_eq!(applied["result"]["committed"], true);

    let bake_directory = directory.path().join("bake");
    let bake = run_pot_json(
        pot()
            .args(["bake"])
            .arg(&project)
            .args(["--kind", "simulation", "--frames", "1:2", "--out"])
            .arg(&bake_directory)
            .arg("--json"),
        "pot bake ops-only Dynamic Paint WEIGHT frames",
    )?;
    assert_eq!(bake["result"]["frame_count"], 2);
    for frame_number in 1..=2 {
        let frame_key = frame_number.to_string();
        let baked: Value = serde_json::from_slice(&fs::read(
            bake_directory.join(format!("frame_{:08}.json", frame_number - 1)),
        )?)?;
        assert_numeric_arrays(
            &canvas_oracle["frames"][frame_key.as_str()]["weights"],
            &baked["paint_weights"]["canvas"]["0"],
            WEIGHT_TOLERANCE,
            &format!("ops-only WEIGHT frame {frame_number}"),
        )?;
    }
    assert_ne!(
        canvas_oracle["frames"]["2"]["weights"],
        oracle["canvases"]["CanvasNoErase"]["frames"]["2"]["weights"],
        "the oracle eraser must alter frame 2"
    );
    let unsupported_project = directory.path().join("dynamic-paint-weight-ramp");
    checked_output(
        pot().args(["init"]).arg(&unsupported_project).arg("--json"),
        "pot init unsupported brush project",
    )?;
    let unsupported_batch_path = directory.path().join("unsupported-operations.json");
    fs::write(
        &unsupported_batch_path,
        serde_json::to_vec(&json!({
            "schema_version":1,
            "base_revision":0,
            "operations":[
                {"op":"node.create","id":"brush","kind":"box","params":{"size":0.1}},
                {"op":"physics.dynamic_paint.create","target":{"id":"brush"},"settings":{"role":"brush","paint_source":"POINT","proximity_falloff":"RAMP"}}
            ]
        }))?,
    )?;
    let unsupported = pot()
        .args(["apply"])
        .arg(&unsupported_project)
        .args(["--file"])
        .arg(&unsupported_batch_path)
        .arg("--json")
        .output()?;
    assert!(
        !unsupported.status.success(),
        "unsupported RAMP brush falloff must stay rejected"
    );
    Ok(())
}
