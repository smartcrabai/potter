use std::{
    error::Error,
    fs::{self, OpenOptions},
    io::{self, Write},
    path::Path,
    process::{Command, Output, Stdio},
};

use fs4::fs_std::FileExt;
use serde_json::Value;
use tempfile::tempdir;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn fixture(name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn json(output: &Output) -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn apply_file(scene: &Path, file: &Path, extra: &[&str]) -> Result<Output, Box<dyn Error>> {
    let output = pot()
        .arg("apply")
        .arg(scene)
        .arg("--file")
        .arg(file)
        .args(extra)
        .arg("--json")
        .output()?;
    Ok(output)
}

fn init_scene(scene: &Path) -> Result<Value, Box<dyn Error>> {
    let output = pot().arg("init").arg(scene).arg("--json").output()?;
    assert!(output.status.success());
    json(&output)
}

fn inspect(scene: &Path, args: &[&str]) -> Result<Value, Box<dyn Error>> {
    let output = pot()
        .arg("inspect")
        .arg(scene)
        .args(args)
        .arg("--json")
        .output()?;
    assert!(
        output.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    json(&output)
}

fn primitive_params<'a>(schema: &'a Value, kind: &str) -> Option<&'a Value> {
    schema["allOf"].as_array()?.iter().find_map(|condition| {
        (condition["if"]["properties"]["kind"]["const"].as_str() == Some(kind))
            .then_some(&condition["then"]["properties"]["params"])
    })
}

fn assert_error(output: &Output, code: &str, exit_code: i32) -> Result<Value, Box<dyn Error>> {
    assert_eq!(output.status.code(), Some(exit_code));
    let envelope = json(output)?;
    assert_eq!(envelope["schema_version"], 1);
    assert_eq!(envelope["ok"], false);
    assert_eq!(envelope["error"]["code"], code);
    assert!(envelope["warnings"].is_array());
    Ok(envelope)
}

#[test]
fn node_create_schema_describes_primitive_params_by_kind() -> Result<(), Box<dyn Error>> {
    let output = pot()
        .arg("schema")
        .arg("--op")
        .arg("node.create")
        .arg("--json")
        .output()?;
    assert!(output.status.success());
    let schema = json(&output)?["result"]["schema"].clone();

    let box_params = primitive_params(&schema, "box")
        .ok_or_else(|| io::Error::other("box params schema is missing"))?;
    assert_eq!(box_params["properties"]["size"]["default"], 2.0);
    assert_eq!(box_params["properties"]["size"]["type"], "number");
    assert_eq!(box_params["additionalProperties"], false);

    let sphere_params = primitive_params(&schema, "uv_sphere")
        .ok_or_else(|| io::Error::other("uv_sphere params schema is missing"))?;
    assert_eq!(sphere_params["properties"]["segments"]["default"], 32);
    assert_eq!(sphere_params["properties"]["segments"]["minimum"], 3);
    assert_eq!(sphere_params["properties"]["ring_count"]["default"], 16);
    assert_eq!(sphere_params["properties"]["radius"]["default"], 1.0);

    let cylinder_params = primitive_params(&schema, "cylinder")
        .ok_or_else(|| io::Error::other("cylinder params schema is missing"))?;
    assert_eq!(cylinder_params["properties"]["vertices"]["default"], 32);
    assert_eq!(cylinder_params["properties"]["radius"]["default"], 1.0);
    assert_eq!(cylinder_params["properties"]["depth"]["default"], 2.0);
    assert_eq!(
        cylinder_params["properties"]["end_fill_type"]["enum"],
        serde_json::json!(["NOTHING", "NGON", "TRIFAN"])
    );

    let plane_params = primitive_params(&schema, "plane")
        .ok_or_else(|| io::Error::other("plane params schema is missing"))?;
    assert_eq!(plane_params["properties"]["size"]["default"], 2.0);
    assert_eq!(plane_params["properties"]["size"]["exclusiveMinimum"], 0);

    let cone_params = primitive_params(&schema, "cone")
        .ok_or_else(|| io::Error::other("cone params schema is missing"))?;
    assert_eq!(cone_params["properties"]["vertices"]["minimum"], 3);
    assert_eq!(cone_params["properties"]["radius1"]["default"], 1.0);
    assert_eq!(cone_params["properties"]["radius2"]["default"], 0.0);
    assert!(cone_params["properties"]["end_fill_type"].is_object());

    let torus_params = primitive_params(&schema, "torus")
        .ok_or_else(|| io::Error::other("torus params schema is missing"))?;
    assert_eq!(torus_params["properties"]["major_radius"]["default"], 1.0);
    assert_eq!(torus_params["properties"]["minor_radius"]["default"], 0.25);
    assert_eq!(torus_params["properties"]["major_segments"]["default"], 48);
    assert_eq!(torus_params["properties"]["minor_segments"]["default"], 12);
    assert_eq!(torus_params["properties"]["mode"]["default"], "MAJOR_MINOR");
    assert_eq!(
        torus_params["properties"]["abso_major_rad"]["default"],
        1.25
    );
    assert_eq!(
        torus_params["properties"]["abso_minor_rad"]["default"],
        0.75
    );

    let icosphere_params = primitive_params(&schema, "icosphere")
        .ok_or_else(|| io::Error::other("icosphere params schema is missing"))?;
    assert_eq!(icosphere_params["properties"]["subdivisions"]["minimum"], 1);
    assert_eq!(
        icosphere_params["properties"]["subdivisions"]["maximum"],
        10
    );

    let circle_params = primitive_params(&schema, "circle")
        .ok_or_else(|| io::Error::other("circle params schema is missing"))?;
    assert_eq!(circle_params["properties"]["vertices"]["minimum"], 3);
    assert_eq!(
        circle_params["properties"]["fill_type"]["default"],
        "NOTHING"
    );

    let grid_params = primitive_params(&schema, "grid")
        .ok_or_else(|| io::Error::other("grid params schema is missing"))?;
    assert_eq!(grid_params["properties"]["size"]["default"], 2.0);
    assert_eq!(grid_params["properties"]["x_subdivisions"]["default"], 10);
    assert_eq!(grid_params["properties"]["x_subdivisions"]["minimum"], 1);

    let schema_kinds = schema["properties"]["kind"]["enum"]
        .as_array()
        .ok_or_else(|| io::Error::other("node.create kind enum is missing"))?;
    let geometric_kinds = [
        "box",
        "uv_sphere",
        "sphere",
        "cylinder",
        "cone",
        "torus",
        "icosphere",
        "circle",
        "grid",
        "plane",
    ];
    for kind in geometric_kinds {
        let params = primitive_params(&schema, kind)
            .ok_or_else(|| io::Error::other(format!("{kind} params schema is missing")))?;
        assert_eq!(params["additionalProperties"], false, "{kind}");
        assert!(schema_kinds.contains(&serde_json::json!(kind)));
    }
    for kind in [
        "camera",
        "light",
        "curve",
        "surface",
        "text",
        "metaball",
        "lattice",
        "pointcloud",
        "volume",
        "armature",
        "grease_pencil",
        "collection_instance",
        "group",
    ] {
        let params = primitive_params(&schema, kind)
            .ok_or_else(|| io::Error::other(format!("{kind} params schema is missing")))?;
        assert_eq!(params["properties"], serde_json::json!({}), "{kind}");
        assert_eq!(params["additionalProperties"], false, "{kind}");
        assert!(schema_kinds.contains(&serde_json::json!(kind)));
    }
    Ok(())
}

#[test]
fn node_create_accepts_empty_params_for_non_geometry_kinds() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init_scene(&scene)?;
    let operations = directory.path().join("operations.json");
    fs::write(
        &operations,
        r#"{"schema_version":1,"base_revision":0,"operations":[
            {"op":"node.create","id":"rig","kind":"armature","params":{}},
            {"op":"node.create","id":"camera","kind":"camera","params":{}},
            {"op":"node.create","id":"container","kind":"group","params":{}}
        ]}"#,
    )?;
    let output = apply_file(&scene, &operations, &[])?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert_eq!(json(&output)?["scene"]["revision"], 1);
    assert_eq!(
        inspect(&scene, &[])?["result"]["summary"]["counts"]["nodes"],
        3
    );
    Ok(())
}

#[test]
fn cli_envelopes_unknown_flags_and_help_version_are_stable() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let failure = pot()
        .arg("init")
        .arg(&scene)
        .arg("--not-a-flag")
        .arg("--json")
        .output()?;
    let envelope = assert_error(&failure, "INVALID_ARGUMENT", 2)?;
    assert_eq!(envelope["command"], Value::Null);

    let missing_revision = pot()
        .arg("import")
        .arg(&scene)
        .args(["--file", "input.obj", "--format", "obj", "--json"])
        .output()?;
    let import_error = assert_error(&missing_revision, "INVALID_ARGUMENT", 2)?;
    assert_eq!(import_error["command"], Value::Null);

    let help = pot().arg("--help").arg("--json").output()?;
    assert!(help.status.success());

    assert!(String::from_utf8_lossy(&help.stdout).contains("Usage:"));
    assert!(serde_json::from_slice::<Value>(&help.stdout).is_err());

    let version = pot().arg("--version").arg("--json").output()?;
    assert!(version.status.success());
    assert!(String::from_utf8_lossy(&version.stdout).contains("pot"));
    Ok(())
}

#[test]
fn init_rejects_nonempty_directories() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("occupied");
    fs::create_dir(&scene)?;
    fs::write(scene.join("keep.txt"), "user file")?;
    let output = pot().arg("init").arg(&scene).arg("--json").output()?;
    assert_error(&output, "OUTPUT_EXISTS", 5)?;
    assert_eq!(fs::read_to_string(scene.join("keep.txt"))?, "user file");
    Ok(())
}

#[test]
fn first_and_second_example_batches_produce_expected_inspect_bounds() -> Result<(), Box<dyn Error>>
{
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let initialized = init_scene(&scene)?;
    assert_eq!(initialized["result"]["created"], true);
    assert_eq!(initialized["scene"]["revision"], 0);

    let first = apply_file(&scene, &fixture("first.json"), &[])?;
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stdout)
    );
    let first_value = json(&first)?;
    assert_eq!(first_value["scene"]["revision"], 1);
    assert_eq!(first_value["result"]["committed"], true);

    let inspect_body = inspect(&scene, &["--tag", "body"])?;
    let body = &inspect_body["result"]["items"][0];
    assert_eq!(body["bounds"]["min"], serde_json::json!([-0.5, -0.3, 0.0]));
    assert_eq!(body["bounds"]["max"], serde_json::json!([0.5, 0.3, 0.8]));
    assert_eq!(body["dimensions"], serde_json::json!([1.0, 0.6, 0.8]));

    let second = apply_file(&scene, &fixture("second.json"), &[])?;
    assert!(
        second.status.success(),
        "{}",
        String::from_utf8_lossy(&second.stdout)
    );
    let second_value = json(&second)?;
    assert_eq!(second_value["scene"]["revision"], 2);
    let updated = inspect(&scene, &["--tag", "body"])?;
    let body = &updated["result"]["items"][0];
    let half_x = 0.6 * 15.0_f64.to_radians().cos() + 0.3 * 15.0_f64.to_radians().sin();
    let half_y = 0.6 * 15.0_f64.to_radians().sin() + 0.3 * 15.0_f64.to_radians().cos();
    let min_x = body["bounds"]["min"][0].as_f64().ok_or("missing bound")?;
    assert!(
        (min_x + half_x).abs() < 1.0e-12,
        "min_x={min_x} expected={}",
        -half_x
    );
    assert!((body["bounds"]["min"][1].as_f64().ok_or("missing bound")? + half_y).abs() < 1.0e-12);
    assert!(
        body["bounds"]["min"][2]
            .as_f64()
            .ok_or("missing bound")?
            .abs()
            < 1.0e-12
    );
    assert!((body["bounds"]["max"][0].as_f64().ok_or("missing bound")? - half_x).abs() < 1.0e-12);
    assert!((body["bounds"]["max"][1].as_f64().ok_or("missing bound")? - half_y).abs() < 1.0e-12);
    assert!((body["bounds"]["max"][2].as_f64().ok_or("missing bound")? - 0.8).abs() < 1.0e-12);
    Ok(())
}

#[test]
fn apply_detects_revision_conflict_and_atomic_failure() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init_scene(&scene)?;
    let first = apply_file(&scene, &fixture("first.json"), &[])?;
    assert!(first.status.success());
    let conflict = apply_file(&scene, &fixture("first.json"), &[])?;
    assert_error(&conflict, "REVISION_CONFLICT", 5)?;

    let invalid = tempfile::NamedTempFile::new_in(directory.path())?;
    fs::write(
        invalid.path(),
        r#"{"schema_version":1,"base_revision":1,"operations":[{"op":"material.create","id":"new_material"},{"op":"node.delete","target":{"id":"missing"}}]}"#,
    )?;
    let failed = apply_file(&scene, invalid.path(), &[])?;
    assert_error(&failed, "TARGET_NOT_FOUND", 3)?;
    let current = inspect(&scene, &[])?;
    assert_eq!(current["scene"]["revision"], 1);
    assert_eq!(current["result"]["summary"]["counts"]["materials"], 1);
    Ok(())
}

#[test]
fn dry_run_and_no_op_do_not_write_or_advance_revision() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init_scene(&scene)?;
    let dry = apply_file(&scene, &fixture("first.json"), &["--dry-run"])?;
    assert!(dry.status.success());
    let dry_value = json(&dry)?;
    assert_eq!(dry_value["result"]["committed"], false);
    assert_eq!(dry_value["result"]["changed"], true);
    assert_eq!(dry_value["result"]["candidate_revision"], 1);
    assert_eq!(dry_value["scene"]["revision"], 0);
    assert_eq!(
        inspect(&scene, &[])?["result"]["summary"]["counts"]["nodes"],
        0
    );

    let noop = tempfile::NamedTempFile::new_in(directory.path())?;
    fs::write(
        noop.path(),
        r#"{"schema_version":1,"base_revision":0,"operations":[]}"#,
    )?;
    let unchanged = apply_file(&scene, noop.path(), &[])?;
    assert!(unchanged.status.success());
    let unchanged_value = json(&unchanged)?;
    assert_eq!(unchanged_value["result"]["changed"], false);
    assert_eq!(unchanged_value["result"]["candidate_revision"], 0);
    assert_eq!(unchanged_value["scene"]["revision"], 0);
    Ok(())
}

#[test]
fn apply_reads_json_from_stdin() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init_scene(&scene)?;
    let mut child = pot()
        .arg("apply")
        .arg(&scene)
        .arg("--file")
        .arg("-")
        .arg("--json")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| io::Error::other("child stdin unavailable"))?;
    stdin.write_all(
        br#"{"schema_version":1,"base_revision":0,"operations":[{"op":"node.create","id":"group","kind":"group"}]}"#,
    )?;
    drop(stdin);
    let output = child.wait_with_output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let response = json(&output)?;
    assert_eq!(response["result"]["committed"], true);
    assert_eq!(response["scene"]["revision"], 1);
    Ok(())
}

#[test]
fn undo_redo_create_new_revisions_and_restore_scene_state() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init_scene(&scene)?;
    assert!(
        apply_file(&scene, &fixture("first.json"), &[])?
            .status
            .success()
    );

    let undo = pot()
        .arg("undo")
        .arg(&scene)
        .arg("--base-revision")
        .arg("1")
        .arg("--json")
        .output()?;
    assert!(
        undo.status.success(),
        "{}",
        String::from_utf8_lossy(&undo.stdout)
    );
    assert_eq!(json(&undo)?["scene"]["revision"], 2);
    assert_eq!(
        inspect(&scene, &[])?["result"]["summary"]["counts"]["nodes"],
        0
    );

    let redo = pot()
        .arg("redo")
        .arg(&scene)
        .arg("--base-revision")
        .arg("2")
        .arg("--json")
        .output()?;
    assert!(
        redo.status.success(),
        "{}",
        String::from_utf8_lossy(&redo.stdout)
    );
    assert_eq!(json(&redo)?["scene"]["revision"], 3);
    assert_eq!(
        inspect(&scene, &["--tag", "body"])?["result"]["items"]
            .as_array()
            .ok_or("missing items")?
            .len(),
        1
    );
    Ok(())
}

#[test]
fn apply_rejects_unknown_operation_fields_and_resolves_targets_strictly()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init_scene(&scene)?;
    let invalid_field = tempfile::NamedTempFile::new_in(directory.path())?;
    fs::write(
        invalid_field.path(),
        r#"{"schema_version":1,"base_revision":0,"operations":[{"op":"node.create","id":"group","kind":"group","mystery":true}]}"#,
    )?;
    let invalid = apply_file(&scene, invalid_field.path(), &[])?;
    let envelope = assert_error(&invalid, "INVALID_OPERATION", 2)?;
    assert_eq!(
        envelope["error"]["details"]["pointer"],
        "/operations/0/mystery"
    );

    let missing = tempfile::NamedTempFile::new_in(directory.path())?;
    fs::write(
        missing.path(),
        r#"{"schema_version":1,"base_revision":0,"operations":[{"op":"node.update","target":{"id":"missing"},"set":{"name":"x"}}]}"#,
    )?;
    let missing_output = apply_file(&scene, missing.path(), &[])?;
    assert_error(&missing_output, "TARGET_NOT_FOUND", 3)?;

    let ambiguous = tempfile::NamedTempFile::new_in(directory.path())?;
    fs::write(
        ambiguous.path(),
        r#"{"schema_version":1,"base_revision":0,"operations":[{"op":"node.create","id":"one","kind":"group","tags":["pair"]},{"op":"node.create","id":"two","kind":"group","tags":["pair"]},{"op":"node.update","target":{"tag":"pair"},"set":{"name":"same"}}]}"#,
    )?;
    let ambiguous_output = apply_file(&scene, ambiguous.path(), &[])?;
    assert_error(&ambiguous_output, "AMBIGUOUS_TARGET", 5)?;
    assert_eq!(
        inspect(&scene, &[])?["result"]["summary"]["counts"]["nodes"],
        0
    );
    Ok(())
}

#[test]
fn readers_succeed_during_writer_lock_and_writers_fail_fast() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init_scene(&scene)?;

    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .open(scene.join(".potter/lock"))?;
    assert!(lock.try_lock_exclusive()?);

    let reader = pot().arg("validate").arg(&scene).arg("--json").output()?;
    assert!(
        reader.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&reader.stdout),
        String::from_utf8_lossy(&reader.stderr)
    );
    assert_eq!(json(&reader)?["scene"]["revision"], 0);

    let operations = tempfile::NamedTempFile::new_in(directory.path())?;
    fs::write(
        operations.path(),
        r#"{"schema_version":1,"base_revision":0,"operations":[{"op":"node.create","id":"pending","kind":"group"}]}"#,
    )?;
    let busy = apply_file(&scene, operations.path(), &[])?;
    assert_error(&busy, "SCENE_BUSY", 5)?;
    lock.unlock()?;
    Ok(())
}

#[test]
fn concurrent_preview_and_validate_succeed_on_one_scene() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init_scene(&scene)?;
    let operations = tempfile::NamedTempFile::new_in(directory.path())?;
    fs::write(
        operations.path(),
        r#"{"schema_version":1,"base_revision":0,"operations":[{"op":"node.create","id":"box","kind":"box","params":{"size":1}}]}"#,
    )?;
    let applied = apply_file(&scene, operations.path(), &[])?;
    assert!(applied.status.success());

    let mut preview_command = pot();
    let preview = preview_command
        .arg("preview")
        .arg(&scene)
        .arg("--size")
        .arg("1024")
        .arg("--json")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let mut validate_command = pot();
    let validate = validate_command
        .arg("validate")
        .arg(&scene)
        .arg("--json")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let preview = preview.wait_with_output()?;
    let validate = validate.wait_with_output()?;
    assert!(
        preview.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&preview.stdout),
        String::from_utf8_lossy(&preview.stderr)
    );
    assert!(
        validate.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&validate.stdout),
        String::from_utf8_lossy(&validate.stderr)
    );
    assert_eq!(json(&preview)?["scene"]["revision"], 1);
    assert_eq!(json(&validate)?["scene"]["revision"], 1);
    Ok(())
}
