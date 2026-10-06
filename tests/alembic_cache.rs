use std::{error::Error, fs, path::Path, process::Command};

use serde_json::{Value, json};
use tempfile::tempdir;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn run_ok(command: &mut Command) -> Result<std::process::Output, Box<dyn Error>> {
    let output = command.output()?;
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(output)
}

fn init(scene: &Path) -> Result<(), Box<dyn Error>> {
    run_ok(pot().args(["init"]).arg(scene).arg("--json"))?;
    Ok(())
}

fn apply(scene: &Path, operations: &Value) -> Result<(), Box<dyn Error>> {
    let document_path = scene.join("scene.json");
    let document: Value = serde_json::from_slice(&fs::read(&document_path)?)?;
    let batch = scene.with_extension("alembic-cache-ops.json");
    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": document["revision"],
            "operations": operations,
        }))?,
    )?;
    run_ok(
        pot()
            .args(["apply"])
            .arg(scene)
            .args(["--file"])
            .arg(batch)
            .arg("--json"),
    )?;
    Ok(())
}

fn export_animated_cache(scene: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
    init(scene)?;
    apply(
        scene,
        &json!([
            {"op":"node.create", "id":"body", "kind":"plane", "params":{}},
            {"op":"modifier.create", "target":{"id":"body"}, "id":"wave", "type":"wave", "params":{"speed":0.25,"height":0.5,"width":1.5,"narrowness":1.5}},
            {"op":"action.create", "id":"samples", "name":"Samples"},
            {"op":"node.update", "target":{"id":"body"}, "set":{"action":"samples"}},
            {"op":"keyframe.insert", "target":{"id":"body"}, "path":"transform.translation", "index":0, "frame":1.0, "value":0.0},
            {"op":"keyframe.insert", "target":{"id":"body"}, "path":"transform.translation", "index":0, "frame":3.0, "value":0.0}
        ]),
    )?;
    run_ok(
        pot()
            .args(["export"])
            .arg(scene)
            .args(["--format", "alembic", "--allow-lossy", "--out"])
            .arg(output)
            .args(["--overwrite", "--json"]),
    )?;
    Ok(())
}

fn add_external_cache_modifier(
    scene: &Path,
    resource_path: &Path,
    expected_hash: &str,
) -> Result<(), Box<dyn Error>> {
    let document_path = scene.join("scene.json");
    let mut document: Value = serde_json::from_slice(&fs::read(&document_path)?)?;
    document["resources"]["cache"] = json!({
        "uri": resource_path,
        "hash": expected_hash,
        "kind": "alembic",
    });
    fs::write(&document_path, serde_json::to_vec_pretty(&document)?)?;
    apply(
        scene,
        &json!([{
            "op":"modifier.create",
            "target":{"id":"body"},
            "id":"cache",
            "type":"mesh_sequence_cache",
            "params":{
                "resource":"cache",
                "object_path":"/root/body/geometry",
                "read_data":["VERT","POLY","UV","COLOR"],
                "use_vertex_interpolation":true
            }
        }]),
    )?;
    Ok(())
}

fn inspect(scene: &Path, frame: f64) -> Result<std::process::Output, Box<dyn Error>> {
    Ok(pot()
        .args(["inspect"])
        .arg(scene)
        .args(["--id", "body", "--frame"])
        .arg(frame.to_string())
        .arg("--json")
        .output()?)
}

fn bounds(scene: &Path, frame: f64) -> Result<Value, Box<dyn Error>> {
    let output = run_ok(
        pot()
            .args(["inspect"])
            .arg(scene)
            .args(["--id", "body", "--frame"])
            .arg(frame.to_string())
            .arg("--json"),
    )?;
    let envelope: Value = serde_json::from_slice(&output.stdout)?;
    let bounds = envelope
        .get("result")
        .and_then(|result| result.get("items"))
        .and_then(Value::as_array)
        .and_then(|items| items.first())
        .and_then(|item| item.get("bounds"))
        .ok_or("inspect response is missing evaluated bounds")?;
    Ok(bounds.clone())
}

fn mesh_positions(scene: &Path, frame: f64) -> Result<Vec<[f64; 3]>, Box<dyn Error>> {
    let project = potter::store::Project::open(scene)?;
    let snapshot = potter::eval::Snapshot::evaluate_with_cache(
        project.doc(),
        &potter::eval::EvaluationContext {
            frame: Some(frame),
            ..potter::eval::EvaluationContext::default()
        },
        Some(project.path()),
    )?;
    let mesh = snapshot
        .meshes
        .values()
        .next()
        .ok_or("evaluated mesh is missing")?;
    Ok(mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co.to_array())
        .collect())
}

fn assert_positions_close(actual: &[[f64; 3]], expected: &[[f64; 3]], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().flatten().zip(expected.iter().flatten()) {
        assert!((actual - expected).abs() <= tolerance);
    }
}

fn assert_bounds_close(actual: &Value, expected: &Value) -> Result<(), Box<dyn Error>> {
    for side in ["min", "max"] {
        let actual_values = actual
            .get(side)
            .and_then(Value::as_array)
            .ok_or("inspect bounds are missing a coordinate array")?;
        let expected_values = expected
            .get(side)
            .and_then(Value::as_array)
            .ok_or("expected bounds are missing a coordinate array")?;
        for (actual, expected) in actual_values.iter().zip(expected_values) {
            let actual = actual
                .as_f64()
                .ok_or("actual bounds coordinate is not numeric")?;
            let expected = expected
                .as_f64()
                .ok_or("expected bounds coordinate is not numeric")?;
            assert!((actual - expected).abs() <= 1.0e-6);
        }
    }
    Ok(())
}

fn midpoint_position(first: [f64; 3], last: [f64; 3]) -> [f64; 3] {
    let [first_x, first_y, first_z] = first;
    let [last_x, last_y, last_z] = last;
    [
        first_x + (last_x - first_x) * 0.5,
        first_y + (last_y - first_y) * 0.5,
        first_z + (last_z - first_z) * 0.5,
    ]
}

#[test]
fn mesh_sequence_cache_samples_and_interpolates_exported_alembic_meshes()
-> Result<(), Box<dyn Error>> {
    let temporary = tempdir()?;
    let source = temporary.path().join("source.pot");
    let target = temporary.path().join("target.pot");
    let archive = temporary.path().join("animated.abc");
    export_animated_cache(&source, &archive)?;
    init(&target)?;
    apply(
        &target,
        &json!([
            {"op":"node.create", "id":"body", "kind":"plane", "params":{}},
            {"op":"resource.pack", "id":"cache", "uri":archive, "kind":"alembic"},
            {"op":"modifier.create", "target":{"id":"body"}, "id":"cache", "type":"mesh_sequence_cache", "params":{"resource":"cache", "object_path":"/root/body/geometry", "read_data":["VERT","POLY","UV","COLOR"], "use_vertex_interpolation":true}}
        ]),
    )?;
    let archive_hash = potter::hash::sha256(&fs::read(&archive)?);
    let hash_hex = archive_hash.strip_prefix("sha256:").unwrap_or_default();
    let asset_uri = format!("assets/sha256/{hash_hex}/animated.abc");
    let asset_path = target.join(&asset_uri);
    fs::create_dir_all(asset_path.parent().ok_or("asset path has no parent")?)?;
    fs::copy(&archive, &asset_path)?;
    let scene_path = target.join("scene.json");
    let mut document: Value = serde_json::from_slice(&fs::read(&scene_path)?)?;
    document["resources"]["cache"]["uri"] = json!(asset_uri);
    document["resources"]["cache"]["packed"] = json!(false);
    if let Some(resource) = document["resources"]["cache"].as_object_mut() {
        resource.remove("bytes");
    }
    fs::write(scene_path, serde_json::to_vec_pretty(&document)?)?;

    for frame in [1.0, 3.0] {
        assert_positions_close(
            &mesh_positions(&target, frame)?,
            &mesh_positions(&source, frame)?,
            1.0e-6,
        );
        assert_bounds_close(&bounds(&target, frame)?, &bounds(&source, frame)?)?;
    }
    let first = mesh_positions(&target, 1.0)?;
    let last = mesh_positions(&target, 3.0)?;
    let expected_middle = first
        .iter()
        .zip(&last)
        .map(|(first, last)| midpoint_position(*first, *last))
        .collect::<Vec<_>>();
    assert_positions_close(&mesh_positions(&target, 2.0)?, &expected_middle, 1.0e-12);
    Ok(())
}

#[test]
fn mesh_sequence_cache_reports_missing_and_changed_external_files() -> Result<(), Box<dyn Error>> {
    let temporary = tempdir()?;
    let source = temporary.path().join("source.pot");
    let archive = temporary.path().join("animated.abc");
    export_animated_cache(&source, &archive)?;
    let original = fs::read(&archive)?;
    let expected_hash = potter::hash::sha256(&original);

    let missing_scene = temporary.path().join("missing.pot");
    init(&missing_scene)?;
    apply(
        &missing_scene,
        &json!([{"op":"node.create", "id":"body", "kind":"plane", "params":{}}]),
    )?;
    let absent = temporary.path().join("missing.abc");
    add_external_cache_modifier(&missing_scene, &absent, &expected_hash)?;
    let output = inspect(&missing_scene, 1.0)?;
    assert_eq!(output.status.code(), Some(3));
    let error: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(error["error"]["code"], "DEPENDENCY_MISSING");

    let changed_file = temporary.path().join("changed.abc");
    let mut changed = original;
    let last = changed.len() - 1;
    changed[last] ^= 1;
    fs::write(&changed_file, changed)?;
    let changed_scene = temporary.path().join("changed.pot");
    init(&changed_scene)?;
    apply(
        &changed_scene,
        &json!([{"op":"node.create", "id":"body", "kind":"plane", "params":{}}]),
    )?;
    add_external_cache_modifier(&changed_scene, &changed_file, &expected_hash)?;
    let output = inspect(&changed_scene, 1.0)?;
    assert_eq!(output.status.code(), Some(4));
    let error: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(error["error"]["code"], "ASSET_CHANGED");
    Ok(())
}

#[test]
fn mesh_sequence_cache_vertex_only_read_rejects_count_mismatches() -> Result<(), Box<dyn Error>> {
    let temporary = tempdir()?;
    let source = temporary.path().join("source.pot");
    let archive = temporary.path().join("animated.abc");
    export_animated_cache(&source, &archive)?;
    let target = temporary.path().join("target.pot");
    init(&target)?;
    apply(
        &target,
        &json!([
            {"op":"node.create", "id":"body", "kind":"box", "params":{"size":2.0}},
            {"op":"resource.pack", "id":"cache", "uri":archive, "kind":"alembic"},
            {"op":"modifier.create", "target":{"id":"body"}, "id":"cache", "type":"mesh_sequence_cache", "params":{"resource":"cache", "object_path":"/root/body/geometry", "read_data":["VERT"]}}
        ]),
    )?;
    let output = inspect(&target, 1.0)?;
    assert_eq!(output.status.code(), Some(4));
    let error: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(error["error"]["code"], "EVALUATION_FAILED");
    assert_eq!(error["error"]["details"]["input_vertex_count"], 8);
    assert_eq!(error["error"]["details"]["cache_vertex_count"], 4);
    Ok(())
}

#[test]
fn mesh_sequence_cache_schema_describes_modifier_parameters() -> Result<(), Box<dyn Error>> {
    let schema = potter::schema::schema("operations", Some("modifier.create"))?;
    let cache_params = schema
        .get("allOf")
        .and_then(Value::as_array)
        .and_then(|conditions| {
            conditions.iter().find(|condition| {
                condition
                    .get("if")
                    .and_then(|condition| condition.get("properties"))
                    .and_then(|properties| properties.get("type"))
                    .and_then(|type_schema| type_schema.get("const"))
                    .and_then(Value::as_str)
                    == Some("mesh_sequence_cache")
            })
        })
        .and_then(|condition| condition.get("then"))
        .and_then(|then| then.get("properties"))
        .and_then(|properties| properties.get("params"))
        .ok_or("modifier schema is missing conditional cache parameters")?;
    assert_eq!(
        cache_params.get("required"),
        Some(&json!(["resource", "object_path"]))
    );
    assert_eq!(
        cache_params
            .get("properties")
            .and_then(|properties| properties.get("read_data"))
            .and_then(|read_data| read_data.get("items"))
            .and_then(|items| items.get("enum")),
        Some(&json!(["VERT", "POLY", "UV", "COLOR", "ATTRIBUTES"]))
    );
    for field in [
        "resource",
        "object_path",
        "read_data",
        "frame_offset",
        "scale",
        "override_frame",
        "velocity_scale",
        "use_vertex_interpolation",
    ] {
        let field_present = cache_params
            .get("properties")
            .and_then(|properties| properties.get(field))
            .is_some();
        assert!(field_present);
    }
    Ok(())
}
