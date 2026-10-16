//! A real schema-version-1 project, written by the release before native UV layers, upgrades on load
//! and keeps its revisions, history, asset hashes, UV values, and pins across inspect, apply, undo,
//! redo, and export. The fixture is `tests/fixtures/schema_v1_project`, created by `pot` at the
//! pre-PR12 revision with a textured box, a pinned face, and two committed applies.

use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use serde_json::{Value, json};
use tempfile::TempDir;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/schema_v1_project")
}

fn copy_tree(from: &Path, to: &Path) -> TestResult {
    if from.is_dir() {
        fs::create_dir_all(to)?;
        for entry in fs::read_dir(from)? {
            let entry = entry?;
            copy_tree(&entry.path(), &to.join(entry.file_name()))?;
        }
    } else {
        fs::copy(from, to)?;
    }
    Ok(())
}

/// Copies the fixture into `directory/scene` and returns the project path. The runtime directories that
/// `init` creates are not tracked by git, so they are recreated here.
fn copy_fixture(directory: &Path) -> TestResult<PathBuf> {
    let scene = directory.join("scene");
    copy_tree(&fixture(), &scene)?;
    fs::create_dir_all(scene.join(".potter/tmp"))?;
    Ok(scene)
}

fn envelope(command: &mut Command) -> TestResult<Value> {
    let output = command.output()?;
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn apply(scene: &Path, base_revision: u64, operations: &Value) -> TestResult<Value> {
    let ops = scene
        .parent()
        .ok_or("scene has no parent")?
        .join("ops.json");
    fs::write(
        &ops,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": base_revision,
            "operations": operations,
        }))?,
    )?;
    envelope(
        pot()
            .arg("apply")
            .arg(scene)
            .arg("--file")
            .arg(&ops)
            .arg("--json"),
    )
}

fn history(scene: &Path, direction: &str, base_revision: u64) -> TestResult<Value> {
    envelope(pot().arg(direction).arg(scene).args([
        "--base-revision",
        &base_revision.to_string(),
        "--json",
    ]))
}

fn document(scene: &Path) -> TestResult<Value> {
    Ok(serde_json::from_slice(&fs::read(
        scene.join("scene.json"),
    )?)?)
}

/// The attributes of the project's only mesh, as stored on disk.
fn mesh_attributes(scene: &Path) -> TestResult<Value> {
    let document = document(scene)?;
    let block = document["data_blocks"]
        .as_object()
        .and_then(|blocks| blocks.values().find(|block| block["mesh"].is_object()))
        .ok_or("the project has no mesh")?;
    Ok(block["mesh"]["attributes"].clone())
}

/// The first corner and pin flags of face 0 as stored on disk. Version 1 keeps them in `uv_map`
/// (glTF convention); version 2 keeps them in the first layer of `uv_layers`.
fn face_zero(scene: &Path) -> TestResult<(Value, Value)> {
    let attributes = mesh_attributes(scene)?;
    let entry = if document(scene)?["schema_version"] == 1 {
        attributes["uv_map"][0].clone()
    } else {
        attributes["uv_layers"]["layers"][0]["faces"][0].clone()
    };
    Ok((entry["uv"][0].clone(), entry["pinned"].clone()))
}

fn assert_corner(scene: &Path, expected: [f64; 2]) -> TestResult {
    let (corner, _) = face_zero(scene)?;
    let actual = [
        corner[0].as_f64().ok_or("u is not a number")?,
        corner[1].as_f64().ok_or("v is not a number")?,
    ];
    assert!(
        (actual[0] - expected[0]).abs() < 1.0e-9 && (actual[1] - expected[1]).abs() < 1.0e-9,
        "corner {actual:?} differs from {expected:?}"
    );
    Ok(())
}

#[test]
fn schema_v1_project_keeps_revisions_uv_state_and_hashes_through_edit_history_and_export()
-> TestResult {
    let directory = TempDir::new()?;
    let scene = copy_fixture(directory.path())?;
    let stored = fs::read(scene.join("scene.json"))?;

    let inspected = envelope(pot().arg("inspect").arg(&scene).arg("--json"))?;
    assert_eq!(inspected["ok"], true, "{inspected}");
    assert_eq!(inspected["scene"]["revision"], 2);
    assert_eq!(
        fs::read(scene.join("scene.json"))?,
        stored,
        "opening a version 1 project must not rewrite it"
    );

    let original = [-0.9, -0.8];
    assert_corner(&scene, original)?;
    assert_eq!(
        face_zero(&scene)?.1,
        json!([true, true, true, true]),
        "the pinned face keeps its pins"
    );

    let stale = apply(
        &scene,
        1,
        &json!([{"op": "uv.transform", "target": {"id": "cube"}, "translation": [0.05, 0.0]}]),
    )?;
    assert_eq!(stale["error"]["code"], "REVISION_CONFLICT", "{stale}");

    let edited = apply(
        &scene,
        2,
        &json!([{"op": "uv.transform", "target": {"id": "cube"}, "translation": [0.05, 0.0]}]),
    )?;
    assert_eq!(edited["ok"], true, "{edited}");
    assert_eq!(edited["scene"]["revision"], 3);
    assert_eq!(document(&scene)?["schema_version"], 3);
    let store = &mesh_attributes(&scene)?["uv_layers"];
    assert_eq!(store["layers"][0]["name"], "UVMap");
    assert_eq!(store["active"], "UVMap");
    assert_eq!(store["render"], "UVMap");
    assert_corner(&scene, [-0.85, -0.8])?;

    let undone = history(&scene, "undo", 3)?;
    assert_eq!(undone["ok"], true, "{undone}");
    assert_eq!(undone["scene"]["revision"], 4);
    assert_corner(&scene, original)?;
    assert_eq!(
        face_zero(&scene)?.1,
        json!([true, true, true, true]),
        "undo restores the pins"
    );

    let redone = history(&scene, "redo", 4)?;
    assert_eq!(redone["ok"], true, "{redone}");
    assert_eq!(redone["scene"]["revision"], 5);
    assert_corner(&scene, [-0.85, -0.8])?;

    let exported = envelope(
        pot()
            .arg("export")
            .arg(&scene)
            .args(["--format", "glb", "--out"])
            .arg(directory.path().join("cube.glb"))
            .arg("--json"),
    )?;
    assert_eq!(exported["ok"], true, "{exported}");
    Ok(())
}

#[test]
fn schema_v1_history_is_verified_before_it_is_upgraded() -> TestResult {
    let directory = TempDir::new()?;
    let scene = copy_fixture(directory.path())?;
    let undo_target = document(&scene)?["history"]["undo"]
        .as_array()
        .and_then(|undo| undo.last())
        .and_then(Value::as_str)
        .ok_or("the fixture has no undo history")?
        .to_owned();
    let digest = undo_target
        .strip_prefix("sha256:")
        .ok_or("undo entry is not a content hash")?;
    let blob = scene.join("history/sha256").join(digest).join("blob.json");
    let mut bytes = fs::read(&blob)?;
    bytes.push(b'\n');
    fs::write(&blob, bytes)?;

    let refused = history(&scene, "undo", 2)?;
    assert_eq!(refused["ok"], false, "{refused}");
    assert_eq!(refused["error"]["code"], "INTERNAL_ERROR", "{refused}");
    Ok(())
}
