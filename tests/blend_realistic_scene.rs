use std::{
    error::Error,
    fs, io,
    path::Path,
    process::{Command, Output},
};

#[path = "common/blender_file.rs"]
mod blender_file;
use blender_file::blender_executable;

use serde_json::{Value, json};
use tempfile::tempdir;

const SCENE_BUILDER: &str = include_str!("fixtures/blend_realistic_scene.py");
const SCENE_DUMPER: &str = include_str!("fixtures/blend_realistic_scene_dump.py");
const FRAME_TOLERANCE: f64 = 1.0e-5;
const GATED_OBJECTS: [&str; 2] = [
    "Gear_Array_Bevel_Subdivision_Normal",
    "Housing_Boolean_Decimate_Solidify",
];

fn checked_output(mut command: Command, label: &str) -> Result<Output, Box<dyn Error>> {
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

fn run_pot(args: &[&Path], strings: &[&str], label: &str) -> Result<Value, Box<dyn Error>> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_pot"));
    for arg in args {
        command.arg(arg);
    }
    command.args(strings).arg("--json");
    let output = checked_output(command, label)?;
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn import_blend(
    project: &Path,
    source: &Path,
    blender: &Path,
    extra: &[&str],
    label: &str,
) -> Result<Value, Box<dyn Error>> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_pot"));
    command
        .arg("import")
        .arg(project)
        .arg("--file")
        .arg(source)
        .args([
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "0",
            "--blender",
        ])
        .arg(blender)
        .args(extra)
        .arg("--json");
    let output = checked_output(command, label)?;
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn inspect_node(project: &Path, id: &str, frame: u32) -> Result<Value, Box<dyn Error>> {
    let output = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("inspect")
        .arg(project)
        .args(["--id"])
        .arg(id)
        .args(["--frame"])
        .arg(frame.to_string())
        .arg("--json")
        .output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "inspecting `{id}` at frame {frame} failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        ))
        .into());
    }
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn assert_close(expected: f64, actual: f64, context: &str) {
    assert!(
        (expected - actual).abs() <= FRAME_TOLERANCE,
        "{context}: expected {expected}, got {actual}"
    );
}

fn assert_json_close(
    expected: &Value,
    actual: &Value,
    context: &str,
) -> Result<(), Box<dyn Error>> {
    match (expected, actual) {
        (Value::Number(expected), Value::Number(actual)) => assert_close(
            expected
                .as_f64()
                .ok_or("expected numeric value is not f64")?,
            actual.as_f64().ok_or("actual numeric value is not f64")?,
            context,
        ),
        (Value::Array(expected), Value::Array(actual)) => {
            assert_eq!(
                expected.len(),
                actual.len(),
                "{context}: array length differs"
            );
            for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
                assert_json_close(expected, actual, &format!("{context}[{index}]"))?;
            }
        }
        (Value::Object(expected), Value::Object(actual)) => {
            assert_eq!(
                expected.keys().collect::<Vec<_>>(),
                actual.keys().collect::<Vec<_>>(),
                "{context}: object fields differ"
            );
            for (key, expected) in expected {
                assert_json_close(expected, &actual[key], &format!("{context}.{key}"))?;
            }
        }
        _ => assert_eq!(expected, actual, "{context}"),
    }
    Ok(())
}

fn matrix_from_blender(expected: &Value) -> Result<Vec<f64>, Box<dyn Error>> {
    let rows = expected
        .as_array()
        .ok_or("Blender matrix is not an array")?;
    if rows.len() != 4 {
        return Err("Blender matrix does not have four rows".into());
    }
    let mut flattened = Vec::with_capacity(16);
    for column in 0..4 {
        for row in rows {
            flattened.push(
                row.as_array()
                    .and_then(|values| values.get(column))
                    .and_then(Value::as_f64)
                    .ok_or("Blender matrix element is malformed")?,
            );
        }
    }
    Ok(flattened)
}

fn world_point(point: &Value, matrix: &[f64]) -> Result<[f64; 3], Box<dyn Error>> {
    let point = point.as_array().ok_or("vertex is not an array")?;
    let x = point
        .first()
        .and_then(Value::as_f64)
        .ok_or("vertex x missing")?;
    let y = point
        .get(1)
        .and_then(Value::as_f64)
        .ok_or("vertex y missing")?;
    let z = point
        .get(2)
        .and_then(Value::as_f64)
        .ok_or("vertex z missing")?;
    Ok([
        matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12],
        matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13],
        matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14],
    ])
}

fn point_cloud(points: &Value) -> Result<Vec<[f64; 3]>, Box<dyn Error>> {
    points
        .as_array()
        .ok_or("point cloud is not an array")?
        .iter()
        .map(|point| {
            let values = point.as_array().ok_or("point is not an array")?;
            Ok([
                values
                    .first()
                    .and_then(Value::as_f64)
                    .ok_or("point x missing")?,
                values
                    .get(1)
                    .and_then(Value::as_f64)
                    .ok_or("point y missing")?,
                values
                    .get(2)
                    .and_then(Value::as_f64)
                    .ok_or("point z missing")?,
            ])
        })
        .collect()
}

fn directed_cloud_worst(from: &[[f64; 3]], to: &[[f64; 3]]) -> (f64, [f64; 3], [f64; 3]) {
    let mut worst = (0.0, from[0], to[0]);
    for point in from {
        if let Some((distance, nearest)) = to
            .iter()
            .map(|candidate| {
                (
                    (point[0] - candidate[0])
                        .hypot(point[1] - candidate[1])
                        .hypot(point[2] - candidate[2]),
                    *candidate,
                )
            })
            .min_by(|left, right| left.0.total_cmp(&right.0))
            && distance > worst.0
        {
            worst = (distance, *point, nearest);
        }
    }
    worst
}
fn directed_cloud_error(from: &[[f64; 3]], to: &[[f64; 3]]) -> f64 {
    directed_cloud_worst(from, to).0
}

fn assert_cloud_matches(
    expected: &Value,
    actual: &Value,
    context: &str,
) -> Result<(), Box<dyn Error>> {
    let expected = point_cloud(expected)?;
    let actual = point_cloud(actual)?;
    if expected.is_empty() || actual.is_empty() {
        assert_eq!(
            expected, actual,
            "{context}: empty/nonempty geometry differs"
        );
        return Ok(());
    }
    let expected_to_actual = directed_cloud_worst(&expected, &actual);
    let actual_to_expected = directed_cloud_worst(&actual, &expected);
    let worst = if expected_to_actual.0 >= actual_to_expected.0 {
        expected_to_actual
    } else {
        actual_to_expected
    };
    assert!(
        worst.0 <= FRAME_TOLERANCE,
        "{context}: symmetric nearest-vertex error {} exceeds {FRAME_TOLERANCE} (expected {}, actual {}, unmatched {:?}, nearest {:?})",
        worst.0,
        expected.len(),
        actual.len(),
        worst.1,
        worst.2
    );
    Ok(())
}

fn object_ids_by_name(document: &Value) -> Result<Vec<(String, String)>, Box<dyn Error>> {
    let nodes = document["nodes"]
        .as_object()
        .ok_or("scene nodes are missing")?;
    let mut result = nodes
        .iter()
        .filter_map(|(id, node)| {
            node["name"]
                .as_str()
                .map(|name| (name.to_owned(), id.clone()))
        })
        .collect::<Vec<_>>();
    result.sort();
    Ok(result)
}

fn assert_modifier_reference(
    document: &Value,
    owner_name: &str,
    modifier_name: &str,
    parameter: &str,
    target_name: &str,
) -> Result<(), Box<dyn Error>> {
    let nodes = document["nodes"]
        .as_object()
        .ok_or("scene nodes are missing")?;
    let owner_id = nodes
        .iter()
        .find(|(_, node)| node["name"].as_str() == Some(owner_name))
        .map(|(id, _)| id)
        .ok_or_else(|| io::Error::other(format!("owner `{owner_name}` is missing")))?;
    let owner = &nodes[owner_id];
    let modifiers = owner["modifiers"]
        .as_array()
        .ok_or_else(|| io::Error::other(format!("{owner_name}: modifiers are missing")))?;
    let modifier = modifiers
        .iter()
        .find(|modifier| modifier["name"].as_str() == Some(modifier_name))
        .ok_or_else(|| {
            io::Error::other(format!("{owner_name}: modifier `{modifier_name}` missing"))
        })?;
    let target_id = modifier["params"][parameter].as_str().ok_or_else(|| {
        io::Error::other(format!("{owner_name}: `{parameter}` reference missing"))
    })?;
    let target = nodes
        .get(target_id)
        .ok_or_else(|| io::Error::other(format!("{owner_name}: `{parameter}` ID is unresolved")))?;
    assert_eq!(
        target["name"], target_name,
        "{owner_name}: {parameter} target"
    );
    Ok(())
}

fn assert_surface_binding_data(document: &Value) -> Result<(), Box<dyn Error>> {
    let nodes = document["nodes"]
        .as_object()
        .ok_or("scene nodes are missing")?;
    let plane = nodes
        .values()
        .find(|node| node["name"] == "Plane_SurfaceDeform_Bound")
        .ok_or("bound Surface Deform plane is missing")?;
    let modifier = plane["modifiers"]
        .as_array()
        .and_then(|modifiers| modifiers.first())
        .ok_or("bound Surface Deform modifier is missing")?;
    assert_eq!(
        modifier["binding_data"]["format"], "blender_native_bind_v1",
        "imported Surface Deform must retain native binding data on Modifier"
    );
    assert!(
        modifier["params"].get("is_bound").is_none(),
        "is_bound is Blender RNA state, not a Potter modifier param"
    );
    Ok(())
}

fn assert_shape_key_action_slot(document: &Value) -> Result<(), Box<dyn Error>> {
    let nodes = document["nodes"]
        .as_object()
        .ok_or("scene nodes are missing")?;
    let (target_id, target) = nodes
        .iter()
        .find(|(_, node)| node["name"] == "SurfaceDeform_Target")
        .ok_or("shape-key target is missing")?;
    let data_id = target["data"].as_str().ok_or("shape-key mesh is missing")?;
    let shape_keys = document["data_blocks"][data_id]["shape_keys"]
        .as_object()
        .ok_or("shape-key data is missing")?;
    let action_id = shape_keys["action"]
        .as_str()
        .ok_or("shape-key action is missing")?;
    assert_eq!(shape_keys["action_slot"], "Key");
    let action_slots = document["actions"][action_id]["slots"]
        .as_array()
        .ok_or("shape-key action slots are missing")?;
    assert_eq!(
        action_slots
            .iter()
            .filter(|slot| slot["node"].as_str() == Some(target_id.as_str()))
            .count(),
        1,
        "Key Action slot must be associated with its shape-key owner"
    );
    Ok(())
}

fn assert_import_inspection(
    project: &Path,
    source: &Value,
    document: &Value,
) -> Result<(), Box<dyn Error>> {
    let ids = object_ids_by_name(document)?;
    let expected_names = source["objects"]
        .as_object()
        .ok_or("source object metadata is missing")?
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    let actual_names = ids.iter().map(|(name, _)| name.clone()).collect::<Vec<_>>();
    assert_eq!(actual_names, expected_names, "imported object set differs");
    for (name, id) in ids {
        if GATED_OBJECTS.contains(&name.as_str()) {
            continue;
        }
        for frame in [1_u32, 5, 10] {
            let inspected = inspect_node(project, &id, frame)?;
            let item = inspected["result"]["items"]
                .as_array()
                .and_then(|items| items.first())
                .ok_or_else(|| io::Error::other(format!("{name}: inspect item missing")))?;
            let frame_data = &source["frames"][frame.to_string()][&name];
            let expected_matrix = matrix_from_blender(&frame_data["matrix_world"])?;
            let actual_matrix = item["transform"]["world"]["matrix"]
                .as_array()
                .ok_or_else(|| io::Error::other(format!("{name}: world matrix missing")))?
                .iter()
                .map(|value| {
                    value
                        .as_f64()
                        .ok_or_else(|| io::Error::other("matrix value is not numeric"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            assert_eq!(actual_matrix.len(), 16, "{name}: Potter matrix malformed");
            for (index, expected) in expected_matrix.iter().enumerate() {
                assert_close(
                    *expected,
                    actual_matrix[index],
                    &format!("{name} frame {frame} matrix[{index}]"),
                );
            }

            if frame_data["vertices_world"].is_array() {
                let local_positions = item["evaluated_geometry"]["positions"]
                    .as_array()
                    .ok_or_else(|| {
                        io::Error::other(format!("{name}: evaluated positions missing"))
                    })?;
                let actual_world = local_positions
                    .iter()
                    .map(|position| world_point(position, &actual_matrix))
                    .collect::<Result<Vec<_>, _>>()?;
                let expected_world = point_cloud(&frame_data["vertices_world"])?;
                let error = directed_cloud_error(&expected_world, &actual_world)
                    .max(directed_cloud_error(&actual_world, &expected_world));
                assert!(
                    error <= FRAME_TOLERANCE,
                    "{name} frame {frame}: symmetric nearest-vertex error {error} exceeds \
                     {FRAME_TOLERANCE}"
                );
            }
        }
    }
    Ok(())
}

fn assert_gated_feature_error(
    project: &Path,
    document: &Value,
    object: &str,
    feature: &str,
) -> Result<(), Box<dyn Error>> {
    let (_, id) = object_ids_by_name(document)?
        .into_iter()
        .find(|(name, _)| name == object)
        .ok_or_else(|| io::Error::other(format!("gated object `{object}` missing")))?;
    let output = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("inspect")
        .arg(project)
        .args(["--id"])
        .arg(&id)
        .args(["--frame", "1", "--json"])
        .output()?;
    assert!(!output.status.success(), "{object} unexpectedly evaluated");
    let envelope: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(envelope["error"]["code"], "UNSUPPORTED_FEATURE", "{object}");
    assert_eq!(
        envelope["error"]["details"]["feature_id"], feature,
        "{object}"
    );
    Ok(())
}

#[test]
fn imported_default_blender_scene_renders_with_sequencer_and_no_audio_codec()
-> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping default Blender scene render regression: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let builder_path = root.join("build_default_scene.py");
    fs::write(
        &builder_path,
        r#"import bpy
from pathlib import Path

scene = bpy.context.scene
scene.render.use_sequencer = True
scene.render.ffmpeg.audio_codec = "NONE"
scene.render.resolution_x = 32
scene.render.resolution_y = 32
scene.render.resolution_percentage = 100
bpy.ops.wm.save_as_mainfile(filepath=str(Path(__file__).resolve().parent / "default.blend"))
"#,
    )?;
    let mut build = Command::new(&blender);
    build
        .args(["--background", "--factory-startup", "--python"])
        .arg(&builder_path);
    checked_output(build, "building default Blender scene")?;

    let source_path = root.join("default.blend");
    let project = root.join("project");
    run_pot(
        &[Path::new("init"), project.as_path()],
        &[],
        "initializing default-scene project",
    )?;
    let imported = import_blend(
        &project,
        &source_path,
        &blender,
        &[],
        "importing default Blender scene",
    )?;
    assert_eq!(imported["result"]["committed"], true);

    let document: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let scene_id = document["active_scene"]
        .as_str()
        .ok_or("active scene id is missing")?;
    let scene = &document["scenes"][scene_id];
    assert_eq!(scene["render"]["use_sequencer"], true);
    assert_eq!(scene["render"]["audio_codec"], "none");
    assert_eq!(scene["sequencer"]["strips"], json!([]));

    let output_path = root.join("render");
    let mut render = Command::new(env!("CARGO_BIN_EXE_pot"));
    render
        .arg("render")
        .arg(&project)
        .args(["--format", "png", "--out"])
        .arg(&output_path)
        .arg("--json");
    let rendered = checked_output(render, "rendering imported default Blender scene")?;
    let envelope: Value = serde_json::from_slice(&rendered.stdout)?;
    assert_eq!(envelope["result"]["frame_count"], 1);
    assert!(output_path.join("frame_0001.png").is_file());
    Ok(())
}

#[test]
fn realistic_blender_scene_import_export_reopen_preserves_settings_and_evaluation()
-> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping realistic Blender scene acceptance test: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let builder_path = root.join("build_scene.py");
    let dumper_path = root.join("blend_realistic_scene_dump.py");
    fs::write(&builder_path, SCENE_BUILDER)?;
    fs::write(&dumper_path, SCENE_DUMPER)?;
    let root_text = root.to_str().ok_or("temporary fixture path is not UTF-8")?;

    let mut build = Command::new(&blender);
    build
        .args(["--background", "--factory-startup", "--python"])
        .arg(&builder_path)
        .args(["--", root_text]);
    checked_output(build, "building realistic Blender fixture")?;
    let source_path = root.join("stacked.blend");
    let source: Value = serde_json::from_slice(&fs::read(root.join("original_dump.json"))?)?;

    let project = root.join("project");
    run_pot(
        &[Path::new("init"), project.as_path()],
        &[],
        "initializing project",
    )?;
    let dry_run = import_blend(
        &project,
        &source_path,
        &blender,
        &["--dry-run"],
        "dry-run import",
    )?;
    assert_eq!(dry_run["result"]["committed"], false);
    assert_eq!(dry_run["result"]["losses"], json!([]));

    let imported = import_blend(&project, &source_path, &blender, &[], "committing import")?;
    assert_eq!(imported["result"]["committed"], true);
    assert_eq!(imported["result"]["losses"], json!([]));
    let document: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let scene_id = document["active_scene"]
        .as_str()
        .ok_or("active scene id is missing")?;
    let imported_scene = &document["scenes"][scene_id];
    assert_eq!(imported_scene["render"]["use_sequencer"], true);
    assert_eq!(imported_scene["render"]["audio_codec"], "none");
    assert_eq!(imported_scene["sequencer"]["strips"], json!([]));
    let render = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("render")
        .arg(&project)
        .args(["--format", "png", "--out"])
        .arg(root.join("unsupported-render"))
        .arg("--json")
        .output()?;
    assert!(
        !render.status.success(),
        "render unexpectedly evaluated unsupported stack"
    );
    let render_error: Value = serde_json::from_slice(&render.stdout)?;
    assert_eq!(render_error["error"]["code"], "UNSUPPORTED_FEATURE");
    let details = &render_error["error"]["details"];
    let feature = details["feature_id"]
        .as_str()
        .ok_or("render error lacks feature_id")?;
    let expected_object = match feature {
        "modifier.bevel.high_valence_vertex_mesh" => GATED_OBJECTS[0],
        "modifier.boolean.downstream_topology" => GATED_OBJECTS[1],
        _ => return Err(format!("unexpected render feature {feature}").into()),
    };
    let node_id = details["node_id"]
        .as_str()
        .ok_or("render error lacks node_id")?;
    assert_eq!(document["nodes"][node_id]["name"], expected_object);
    assert_modifier_reference(
        &document,
        "Rigged_Cylinder_Skin",
        "01 | three-link armature skin",
        "object",
        "Rig_ThreeBones_IK",
    )?;
    assert_modifier_reference(
        &document,
        GATED_OBJECTS[0],
        "02 | three-stage train with end cap",
        "end_cap",
        "Gear_Array_EndCap_Source",
    )?;
    assert_modifier_reference(
        &document,
        GATED_OBJECTS[1],
        "01 | exact through-bore",
        "object",
        "Housing_Bore_Cutter",
    )?;
    assert_modifier_reference(
        &document,
        "Plane_SurfaceDeform_Bound",
        "01 | bound surface deformation",
        "target",
        "SurfaceDeform_Target",
    )?;
    assert_surface_binding_data(&document)?;
    assert_shape_key_action_slot(&document)?;
    assert_import_inspection(&project, &source, &document)?;
    assert_gated_feature_error(
        &project,
        &document,
        GATED_OBJECTS[0],
        "modifier.bevel.high_valence_vertex_mesh",
    )?;
    assert_gated_feature_error(
        &project,
        &document,
        GATED_OBJECTS[1],
        "modifier.boolean.downstream_topology",
    )?;

    let lossy_project = root.join("allow-lossy-project");
    run_pot(
        &[Path::new("init"), lossy_project.as_path()],
        &[],
        "initializing allow-lossy project",
    )?;
    let lossy_import = import_blend(
        &lossy_project,
        &source_path,
        &blender,
        &["--allow-lossy"],
        "allow-lossy import",
    )?;
    assert_eq!(lossy_import["result"]["losses"], json!([]));

    let output_path = root.join("round-trip.blend");
    let mut export = Command::new(env!("CARGO_BIN_EXE_pot"));
    export
        .arg("export")
        .arg(&project)
        .args(["--format", "blend", "--out"])
        .arg(&output_path)
        .args(["--blender"])
        .arg(&blender)
        .arg("--json");
    let exported = checked_output(export, "exporting realistic Blender fixture")?;
    let export_result: Value = serde_json::from_slice(&exported.stdout)?;
    assert_eq!(export_result["result"]["losses"], json!([]));

    let mut reopen = Command::new(&blender);
    reopen
        .args(["--background", "--disable-autoexec"])
        .arg(&output_path)
        .args(["--python"])
        .arg(&dumper_path)
        .args(["--", "--reopen", root_text]);
    checked_output(reopen, "reopening and evaluating exported Blender fixture")?;
    let roundtrip: Value = serde_json::from_slice(&fs::read(root.join("roundtrip_dump.json"))?)?;

    let source_meshes = &source["raw_meshes"];
    let roundtrip_meshes = &roundtrip["raw_meshes"];
    for name in [
        "Housing_Boolean_Decimate_Solidify",
        "Housing_Bore_Cutter",
        "EdgeOrderRegression",
    ] {
        assert_eq!(
            source_meshes[name]["matrix_world_bits"], roundtrip_meshes[name]["matrix_world_bits"],
            "{name} matrix_world differs bit-for-bit"
        );
        assert_eq!(
            source_meshes[name]["mesh"], roundtrip_meshes[name]["mesh"],
            "{name} raw mesh arrays differ"
        );
    }
    let housing = "Housing_Boolean_Decimate_Solidify";
    for prefix in ["1", "2", "3"] {
        assert_eq!(
            source_meshes[housing]["prefixes"][prefix],
            roundtrip_meshes[housing]["prefixes"][prefix],
            "{housing} modifier prefix {prefix} raw mesh arrays differ"
        );
    }
    let explicit_edges = source_meshes["EdgeOrderRegression"]["mesh"]["edges"]
        .as_array()
        .ok_or("explicit edge-order mesh edges are missing")?
        .iter()
        .map(|edge| edge["vertices"].clone())
        .collect::<Vec<_>>();
    assert_eq!(
        explicit_edges,
        [json!([2, 3]), json!([0, 3]), json!([1, 2]), json!([0, 1])]
    );
    assert_ne!(
        explicit_edges,
        [json!([0, 1]), json!([1, 2]), json!([2, 3]), json!([3, 0])],
        "fixture edge order must differ from Potter's face-boundary order"
    );

    assert_json_close(&source["scene"], &roundtrip["scene"], "scene settings")?;
    assert_json_close(
        &source["objects"],
        &roundtrip["objects"],
        "object RNA settings",
    )?;
    for frame in ["1", "5", "10"] {
        let expected_objects = source["frames"][frame]
            .as_object()
            .ok_or("source frame objects missing")?;
        let actual_objects = roundtrip["frames"][frame]
            .as_object()
            .ok_or("round-trip frame objects missing")?;
        assert_eq!(
            expected_objects.keys().collect::<Vec<_>>(),
            actual_objects.keys().collect::<Vec<_>>(),
            "frame {frame} objects differ"
        );
        for (name, expected) in expected_objects {
            let actual = &actual_objects[name];
            if name == housing || name == "Housing_Bore_Cutter" {
                assert_eq!(
                    expected["matrix_world_bits"], actual["matrix_world_bits"],
                    "{name} frame {frame} matrix_world differs bit-for-bit"
                );
            }
            assert_json_close(
                &expected["matrix_world"],
                &actual["matrix_world"],
                &format!("{name} frame {frame} matrix"),
            )?;
            if let Some(expected_vertices) = expected["vertices_world"].as_array() {
                let actual_vertices = actual["vertices_world"].as_array().ok_or_else(|| {
                    io::Error::other(format!("{name} frame {frame}: geometry missing"))
                })?;
                assert_cloud_matches(
                    &Value::Array(expected_vertices.clone()),
                    &Value::Array(actual_vertices.clone()),
                    &format!("{name} frame {frame} evaluated geometry"),
                )?;
            }
        }
    }
    Ok(())
}
