//! Real schema-version-1 projects, written by the release before native UV layers. They upgrade on
//! load and keep their revisions, history, UV values, and pins across inspect, apply, undo, redo,
//! and export. `tests/fixtures/schema_v1_project` is a textured box with a pinned face and two
//! applies. `schema_v1_project_deleted_face`, `schema_v1_project_extruded_blender`, and
//! `schema_v1_project_layer_limit` hold legacy data the upgrade must skip or drop: a `uv_map` row for
//! a deleted face, Blender layers left stale by an extrude, and nine candidate layers. Each case
//! checks its warning codes and names, that reads leave every file unchanged, and that the original
//! `scene.json` bytes stay recoverable under `compat/sha256/<hash>/scene.json` once a schema-3 write
//! happens.

use std::{
    collections::BTreeMap,
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
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
/// (glTF convention); version 3 keeps them in the first layer of `uv_layers`.
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
    assert_eq!(document(&scene)?["schema_version"], 14);
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

    assert_eq!(
        face_zero(&scene)?.1,
        json!([true, true, true, true]),
        "the export precondition: face zero is pinned"
    );
    // The cube's face zero is pinned, and glb does not write per-corner pin flags, so the export
    // reports that loss and needs `--allow-lossy`.
    let exported = envelope(
        pot()
            .arg("export")
            .arg(&scene)
            .args(["--format", "glb", "--out"])
            .arg(directory.path().join("cube.glb"))
            .arg("--allow-lossy")
            .arg("--json"),
    )?;
    assert_eq!(exported["ok"], true, "{exported}");
    let losses = exported["result"]["losses"]
        .as_array()
        .ok_or("export result has no losses array")?;
    assert!(
        losses.iter().any(|loss| {
            loss["feature_id"] == "mesh.uv_layer"
                && loss["data_id"] == "cube"
                && loss["reason"]
                    .as_str()
                    .is_some_and(|reason| reason.contains("UVMap"))
        }),
        "glb export did not report the pinned UVMap loss: {exported}"
    );
    Ok(())
}

#[test]
fn schema_v1_active_blender_layer_survives_migration_and_omitted_selector_uv_transform()
-> TestResult {
    let directory = TempDir::new()?;
    let scene = copy_fixture(directory.path())?;
    let scene_file = scene.join("scene.json");
    let mut legacy: Value = serde_json::from_slice(&fs::read(&scene_file)?)?;
    legacy["data_blocks"]["cube_mesh"]["mesh"]["attributes"]["blender_uv_layers"] = json!([
        {
            "name": "Light",
            "active": true,
            "values": vec![json!([0.25, 0.5]); 24]
        }
    ]);
    fs::write(&scene_file, serde_json::to_vec_pretty(&legacy)?)?;
    let original = fs::read(&scene_file)?;
    let expected = read_only_digests(&scene)?;
    let head = document(&scene)?["history"]["head"]
        .as_str()
        .ok_or("the fixture has no history head")?
        .to_owned();
    let head_blob = scene.join(blob_path("history", "blob.json", &head));
    let head_bytes = fs::read(&head_blob)?;

    let inspected =
        envelope(
            pot()
                .arg("inspect")
                .arg(&scene)
                .args(["--id", "cube_mesh", "--json"]),
        )?;
    assert_eq!(inspected["ok"], true, "{inspected}");
    let uv_layers = inspected["result"]["items"][0]["uv_layers"]
        .as_array()
        .ok_or("mesh inspect omitted UV layers")?;
    assert!(
        uv_layers
            .iter()
            .any(|layer| layer["name"] == "Light" && layer["active"] == true),
        "{inspected}"
    );
    assert!(
        uv_layers.iter().any(|layer| {
            layer["name"] == "UVMap" && layer["render"] == true && layer["active"] == false
        }),
        "{inspected}"
    );
    assert_eq!(
        fs::read(&scene_file)?,
        original,
        "inspect rewrote the v1 project"
    );
    assert_eq!(
        read_only_digests(&scene)?,
        expected,
        "inspect rewrote project files"
    );
    assert_eq!(
        fs::read(&head_blob)?,
        head_bytes,
        "inspect rewrote the history head bytes"
    );

    let applied = apply(
        &scene,
        2,
        &json!([{
            "op": "uv.transform",
            "target": {"id": "cube"},
            "translation": [0.25, 0.0]
        }]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");
    assert_eq!(applied["scene"]["revision"], 3);
    assert_eq!(document(&scene)?["schema_version"], 14);
    let attributes = mesh_attributes(&scene)?;
    assert_eq!(attributes["uv_layers"]["active"], "Light");
    assert_eq!(attributes["uv_layers"]["render"], "UVMap");
    assert_corner_close(stored_corner(&scene, "Light", 0)?, [0.5, 0.5]);
    assert_corner_close(stored_corner(&scene, "UVMap", 0)?, [-0.9, -0.8]);
    assert_original_kept(&scene, &original)?;
    assert_history_kept(&scene, &expected)?;
    assert_eq!(
        fs::read(&head_blob)?,
        head_bytes,
        "apply rewrote the history head bytes"
    );
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

/// The fixture of one legacy case: `tests/fixtures/schema_v1_project_<case>`.
fn case_fixture(case: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(format!("schema_v1_project_{case}"))
}

/// Copies one legacy case into `directory/scene` and returns the project path.
fn copy_case(directory: &Path, case: &str) -> TestResult<PathBuf> {
    let scene = directory.join("scene");
    copy_tree(&case_fixture(case), &scene)?;
    fs::create_dir_all(scene.join(".potter/tmp"))?;
    Ok(scene)
}

/// `sha256:<hex>` of `bytes`, the form in which the project names content.
fn digest(bytes: &[u8]) -> String {
    format!("sha256:{}", hex::encode(Sha256::digest(bytes)))
}

/// Relative path of the content-addressed `category` blob that holds `digest`, stored as `name`.
fn blob_path(category: &str, name: &str, digest: &str) -> String {
    let hex = digest.strip_prefix("sha256:").unwrap_or(digest);
    format!("{category}/sha256/{hex}/{name}")
}

/// Digests of `scene.json` and of every history blob, keyed by path relative to the project.
fn read_only_digests(scene: &Path) -> TestResult<BTreeMap<String, String>> {
    let mut digests = BTreeMap::new();
    digests.insert(
        "scene.json".to_owned(),
        digest(&fs::read(scene.join("scene.json"))?),
    );
    collect_digests(&scene.join("history"), scene, &mut digests)?;
    Ok(digests)
}

fn collect_digests(
    directory: &Path,
    root: &Path,
    digests: &mut BTreeMap<String, String>,
) -> TestResult {
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_digests(&path, root, digests)?;
        } else {
            let relative = path.strip_prefix(root)?.to_string_lossy().into_owned();
            digests.insert(relative, digest(&fs::read(&path)?));
        }
    }
    Ok(())
}

/// Every history blob in `expected` still holds the same bytes in `scene`.
fn assert_history_kept(scene: &Path, expected: &BTreeMap<String, String>) -> TestResult {
    let now = read_only_digests(scene)?;
    for (path, expected_digest) in expected
        .iter()
        .filter(|(path, _)| path.starts_with("history/"))
    {
        assert_eq!(now.get(path), Some(expected_digest), "{path} was rewritten");
    }
    Ok(())
}

/// The original schema-1 `scene.json` bytes are recoverable, byte for byte, under their hash.
fn assert_original_kept(scene: &Path, original: &[u8]) -> TestResult {
    let recovered = fs::read(scene.join(blob_path("compat", "scene.json", &digest(original))))?;
    assert!(
        recovered.as_slice() == original,
        "the original schema-1 bytes are not recoverable"
    );
    Ok(())
}

/// The original `scene.json` as recovered from its content-addressed blob.
fn recovered_scene(scene: &Path, original: &[u8]) -> TestResult<Value> {
    let bytes = fs::read(scene.join(blob_path("compat", "scene.json", &digest(original))))?;
    Ok(serde_json::from_slice(&bytes)?)
}

/// The warnings of `envelope` whose stable `code` is `code`.
fn coded(envelope: &Value, code: &str) -> Vec<Value> {
    envelope["warnings"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|warning| warning["code"] == code)
        .cloned()
        .collect()
}

/// The history head that the fixture of `case` recorded, which is its newest committed revision.
fn fixture_head(case: &str) -> TestResult<String> {
    Ok(document(&case_fixture(case))?["history"]["head"]
        .as_str()
        .ok_or("the fixture has no history head")?
        .to_owned())
}

/// The `[u, v]` of one corner value.
fn corner_value(value: &Value) -> TestResult<[f64; 2]> {
    Ok([
        value[0].as_f64().ok_or("u is not a number")?,
        value[1].as_f64().ok_or("v is not a number")?,
    ])
}

/// Corner 0 of face `face_id` in the version-1 `uv_map` of the fixture of `case`.
fn fixture_corner(case: &str, face_id: u64) -> TestResult<[f64; 2]> {
    let attributes = mesh_attributes(&case_fixture(case))?;
    let entry = attributes["uv_map"]
        .as_array()
        .ok_or("the fixture has no uv_map")?
        .iter()
        .find(|entry| entry["face_id"] == face_id)
        .ok_or("the fixture has no row for the face")?;
    corner_value(&entry["uv"][0])
}

/// The schema-version-3 layer `name` of the project's mesh, as stored in `scene.json`.
fn stored_layer(scene: &Path, name: &str) -> TestResult<Value> {
    let attributes = mesh_attributes(scene)?;
    attributes["uv_layers"]["layers"]
        .as_array()
        .and_then(|layers| layers.iter().find(|layer| layer["name"] == name))
        .cloned()
        .ok_or_else(|| format!("layer `{name}` is not stored").into())
}

/// The names of the schema-version-3 layers of the project's mesh, in stored order.
fn stored_layer_names(scene: &Path) -> TestResult<Vec<String>> {
    let attributes = mesh_attributes(scene)?;
    Ok(attributes["uv_layers"]["layers"]
        .as_array()
        .ok_or("the mesh stores no uv_layers")?
        .iter()
        .filter_map(|layer| layer["name"].as_str().map(str::to_owned))
        .collect())
}

/// The face IDs that the schema-version-3 layer `name` stores values for, in stored order.
fn stored_face_ids(scene: &Path, name: &str) -> TestResult<Vec<u64>> {
    let layer = stored_layer(scene, name)?;
    Ok(layer["faces"]
        .as_array()
        .ok_or("the layer has no faces")?
        .iter()
        .filter_map(|face| face["face_id"].as_u64())
        .collect())
}

/// Corner 0 of face `face_id` in the schema-version-3 layer `name`.
fn stored_corner(scene: &Path, name: &str, face_id: u64) -> TestResult<[f64; 2]> {
    let layer = stored_layer(scene, name)?;
    let face = layer["faces"]
        .as_array()
        .ok_or("the layer has no faces")?
        .iter()
        .find(|face| face["face_id"] == face_id)
        .ok_or("the face has no stored value")?;
    corner_value(&face["uv"][0])
}

fn assert_corner_close(actual: [f64; 2], expected: [f64; 2]) {
    assert!(
        (actual[0] - expected[0]).abs() < 1.0e-9 && (actual[1] - expected[1]).abs() < 1.0e-9,
        "corner {actual:?} differs from {expected:?}"
    );
}

#[test]
fn schema_v1_deleted_face_row_is_reported_and_its_bytes_stay_recoverable() -> TestResult {
    let directory = TempDir::new()?;
    let scene = copy_case(directory.path(), "deleted_face")?;
    let fixture = case_fixture("deleted_face");
    let original = fs::read(fixture.join("scene.json"))?;
    let expected = read_only_digests(&fixture)?;
    let original_hash = digest(&original);
    let head = fixture_head("deleted_face")?;
    let before = fixture_corner("deleted_face", 1)?;

    let inspected = envelope(pot().arg("inspect").arg(&scene).arg("--json"))?;
    assert_eq!(inspected["ok"], true, "{inspected}");
    assert_eq!(inspected["scene"]["revision"], 2);
    let skipped = coded(&inspected, "uv.legacy_row_skipped");
    assert_eq!(skipped.len(), 1, "{inspected}");
    assert_eq!(skipped[0]["mesh"], "body_mesh");
    assert_eq!(skipped[0]["layer"], "UVMap");
    assert_eq!(skipped[0]["face_id"], 0);
    assert_eq!(skipped[0]["reason"], "face_missing");
    assert_eq!(
        skipped[0]["original"],
        json!(blob_path("compat", "scene.json", &original_hash))
    );
    assert_eq!(skipped[0]["original_hash"], json!(original_hash));
    assert_eq!(
        read_only_digests(&scene)?,
        expected,
        "a read rewrote a file"
    );

    let listed = envelope(pot().arg("history").arg(&scene).arg("--json"))?;
    assert_eq!(listed["ok"], true, "{listed}");
    // The scene's own warning, plus the one from the head record, whose snapshot still has the row.
    let listed_skips = coded(&listed, "uv.legacy_row_skipped");
    assert_eq!(listed_skips.len(), 2, "{listed}");
    assert!(
        listed_skips
            .iter()
            .any(|warning| warning["original"] == json!(blob_path("history", "blob.json", &head))),
        "{listed}"
    );
    assert_eq!(
        read_only_digests(&scene)?,
        expected,
        "history rewrote a file"
    );

    let applied = apply(
        &scene,
        2,
        &json!([{"op": "uv.transform", "target": {"id": "body"}, "translation": [0.05, 0.0]}]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");
    assert_eq!(applied["scene"]["revision"], 3);
    assert_eq!(
        coded(&applied, "uv.legacy_row_skipped").len(),
        1,
        "{applied}"
    );
    assert_eq!(document(&scene)?["schema_version"], 14);
    assert_eq!(stored_face_ids(&scene, "UVMap")?, vec![1_u64, 2, 3, 4, 5]);
    assert_corner_close(
        stored_corner(&scene, "UVMap", 1)?,
        [before[0] + 0.05, before[1]],
    );
    assert_original_kept(&scene, &original)?;
    let recovered = recovered_scene(&scene, &original)?;
    let stale = recovered["data_blocks"]["body_mesh"]["mesh"]["attributes"]["uv_map"]
        .as_array()
        .ok_or("the recovered scene lost its uv_map")?
        .iter()
        .any(|entry| entry["face_id"] == 0);
    assert!(stale, "the recovered scene must keep the stale face 0 row");

    let undone = history(&scene, "undo", 3)?;
    assert_eq!(undone["ok"], true, "{undone}");
    assert_eq!(undone["scene"]["revision"], 4);
    let restored = coded(&undone, "uv.legacy_row_skipped");
    assert_eq!(restored.len(), 1, "{undone}");
    assert_eq!(
        restored[0]["original"],
        json!(blob_path("history", "blob.json", &head))
    );
    assert_eq!(restored[0]["original_hash"], json!(head));
    assert_eq!(stored_face_ids(&scene, "UVMap")?, vec![1_u64, 2, 3, 4, 5]);
    assert_corner_close(stored_corner(&scene, "UVMap", 1)?, before);
    assert_history_kept(&scene, &expected)?;

    let redone = history(&scene, "redo", 4)?;
    assert_eq!(redone["ok"], true, "{redone}");
    assert_eq!(redone["scene"]["revision"], 5);
    assert!(
        coded(&redone, "uv.legacy_row_skipped").is_empty(),
        "{redone}"
    );
    assert_corner_close(
        stored_corner(&scene, "UVMap", 1)?,
        [before[0] + 0.05, before[1]],
    );
    assert_original_kept(&scene, &original)?;
    assert_history_kept(&scene, &expected)?;
    Ok(())
}

#[test]
fn schema_v1_extruded_blender_layers_are_skipped_with_warnings_and_stay_recoverable() -> TestResult
{
    let directory = TempDir::new()?;
    let scene = copy_case(directory.path(), "extruded_blender")?;
    let fixture = case_fixture("extruded_blender");
    let original = fs::read(fixture.join("scene.json"))?;
    let expected = read_only_digests(&fixture)?;
    let original_hash = digest(&original);
    let head = fixture_head("extruded_blender")?;

    let inspected = envelope(pot().arg("inspect").arg(&scene).arg("--json"))?;
    assert_eq!(inspected["ok"], true, "{inspected}");
    assert_eq!(inspected["scene"]["revision"], 2);
    let skipped = coded(&inspected, "uv.legacy_layer_skipped");
    assert_eq!(skipped.len(), 2, "{inspected}");
    assert_eq!(skipped[0]["layer"], "UVMap");
    assert_eq!(skipped[1]["layer"], "Light");
    for warning in &skipped {
        assert_eq!(warning["mesh"], "mesh_cube");
        assert_eq!(warning["reason"], "corner_count");
        assert!(
            warning["message"]
                .as_str()
                .is_some_and(|message| message.contains("24 values for 40 corners")),
            "{warning}"
        );
        assert_eq!(warning["original_hash"], json!(original_hash));
    }
    assert_eq!(
        read_only_digests(&scene)?,
        expected,
        "a read rewrote a file"
    );

    let listed = envelope(pot().arg("history").arg(&scene).arg("--json"))?;
    assert_eq!(listed["ok"], true, "{listed}");
    // The scene's two warnings, plus the two from the head record, whose snapshot has both layers.
    let listed_skips = coded(&listed, "uv.legacy_layer_skipped");
    assert_eq!(listed_skips.len(), 4, "{listed}");
    assert_eq!(
        listed_skips
            .iter()
            .filter(|warning| warning["original"] == json!(blob_path("history", "blob.json", &head)))
            .count(),
        2,
        "{listed}"
    );
    assert_eq!(
        read_only_digests(&scene)?,
        expected,
        "history rewrote a file"
    );

    let applied = apply(
        &scene,
        2,
        &json!([{
            "op": "node.update",
            "target": {"id": "node_cube"},
            "set": {"transform": {"translation": [1, 2, 3]}}
        }]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");
    assert_eq!(applied["scene"]["revision"], 3);
    assert_eq!(
        coded(&applied, "uv.legacy_layer_skipped").len(),
        2,
        "{applied}"
    );
    assert_eq!(document(&scene)?["schema_version"], 14);
    let attributes = mesh_attributes(&scene)?;
    assert!(
        attributes.get("uv_layers").is_none(),
        "no skipped layer is kept live"
    );
    assert!(attributes.get("blender_uv_layers").is_none());
    let duplicates = attributes["blender_attributes"]
        .as_array()
        .ok_or("the mesh has no blender_attributes")?
        .iter()
        .filter(|entry| entry["data_type"] == "FLOAT2" && entry["domain"] == "CORNER")
        .count();
    assert_eq!(duplicates, 0, "stale FLOAT2 duplicates stayed live");
    assert_original_kept(&scene, &original)?;
    let recovered = recovered_scene(&scene, &original)?;
    let layers = recovered["data_blocks"]["mesh_cube"]["mesh"]["attributes"]["blender_uv_layers"]
        .as_array()
        .ok_or("the recovered scene lost its Blender layers")?;
    assert_eq!(
        layers.len(),
        2,
        "both stale Blender layers must be recoverable"
    );

    let undone = history(&scene, "undo", 3)?;
    assert_eq!(undone["ok"], true, "{undone}");
    assert_eq!(undone["scene"]["revision"], 4);
    let restored = coded(&undone, "uv.legacy_layer_skipped");
    assert_eq!(restored.len(), 2, "{undone}");
    assert_eq!(
        restored[0]["original"],
        json!(blob_path("history", "blob.json", &head))
    );
    assert_eq!(restored[0]["original_hash"], json!(head));
    assert_history_kept(&scene, &expected)?;

    let redone = history(&scene, "redo", 4)?;
    assert_eq!(redone["ok"], true, "{redone}");
    assert_eq!(redone["scene"]["revision"], 5);
    assert!(
        coded(&redone, "uv.legacy_layer_skipped").is_empty(),
        "{redone}"
    );
    assert_original_kept(&scene, &original)?;
    assert_history_kept(&scene, &expected)?;
    Ok(())
}

#[test]
fn schema_v1_layer_limit_keeps_eight_layers_and_reports_the_dropped_one() -> TestResult {
    let directory = TempDir::new()?;
    let scene = copy_case(directory.path(), "layer_limit")?;
    let fixture = case_fixture("layer_limit");
    let original = fs::read(fixture.join("scene.json"))?;
    let expected = read_only_digests(&fixture)?;
    let original_hash = digest(&original);
    let head = fixture_head("layer_limit")?;
    let before = fixture_corner("layer_limit", 1)?;

    let inspected = envelope(pot().arg("inspect").arg(&scene).arg("--json"))?;
    assert_eq!(inspected["ok"], true, "{inspected}");
    assert_eq!(inspected["scene"]["revision"], 2);
    let dropped = coded(&inspected, "uv.legacy_layer_dropped");
    assert_eq!(dropped.len(), 1, "{inspected}");
    assert_eq!(dropped[0]["mesh"], "mesh_cube");
    assert_eq!(dropped[0]["layer"], "Set7");
    assert_eq!(dropped[0]["reason"], "layer_limit");
    assert_eq!(dropped[0]["referenced"], false);
    assert_eq!(dropped[0]["original_hash"], json!(original_hash));
    assert_eq!(
        read_only_digests(&scene)?,
        expected,
        "a read rewrote a file"
    );

    let applied = apply(
        &scene,
        2,
        &json!([{"op": "uv.transform", "target": {"id": "node_cube"}, "translation": [0.05, 0.0]}]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");
    assert_eq!(applied["scene"]["revision"], 3);
    assert_eq!(
        coded(&applied, "uv.legacy_layer_dropped").len(),
        1,
        "{applied}"
    );
    assert_eq!(
        stored_layer_names(&scene)?,
        [
            "Base", "Set1", "Set2", "Set3", "Set4", "Set5", "Set6", "UVMap"
        ]
    );
    let attributes = mesh_attributes(&scene)?;
    assert_eq!(attributes["uv_layers"]["render"], "UVMap");
    assert_eq!(attributes["uv_layers"]["active"], "UVMap");
    assert_corner_close(
        stored_corner(&scene, "UVMap", 1)?,
        [before[0] + 0.05, before[1]],
    );
    assert_original_kept(&scene, &original)?;
    let recovered = recovered_scene(&scene, &original)?;
    let layers = recovered["data_blocks"]["mesh_cube"]["mesh"]["attributes"]["blender_uv_layers"]
        .as_array()
        .ok_or("the recovered scene lost its Blender layers")?;
    assert!(
        layers.iter().any(|layer| layer["name"] == "Set7"),
        "the dropped layer must be recoverable"
    );

    let undone = history(&scene, "undo", 3)?;
    assert_eq!(undone["ok"], true, "{undone}");
    assert_eq!(undone["scene"]["revision"], 4);
    let restored = coded(&undone, "uv.legacy_layer_dropped");
    assert_eq!(restored.len(), 1, "{undone}");
    assert_eq!(restored[0]["layer"], "Set7");
    assert_eq!(
        restored[0]["original"],
        json!(blob_path("history", "blob.json", &head))
    );
    assert_corner_close(stored_corner(&scene, "UVMap", 1)?, before);
    assert_history_kept(&scene, &expected)?;

    let redone = history(&scene, "redo", 4)?;
    assert_eq!(redone["ok"], true, "{redone}");
    assert_eq!(redone["scene"]["revision"], 5);
    assert!(
        coded(&redone, "uv.legacy_layer_dropped").is_empty(),
        "{redone}"
    );
    assert_corner_close(
        stored_corner(&scene, "UVMap", 1)?,
        [before[0] + 0.05, before[1]],
    );
    assert_original_kept(&scene, &original)?;
    assert_history_kept(&scene, &expected)?;
    Ok(())
}
