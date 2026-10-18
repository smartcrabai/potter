//! Project and history hash compatibility for modifier visibility flags.
//!
//! `show_viewport` and `show_render` are omitted from JSON while true, so projects written before
//! the flags existed keep their canonical bytes, scene hashes and history blob addresses.
//!
//! `tests/fixtures/pr18l_old_project/` was written by the pre-change `pot` 0.2.0 binary (commit
//! 3754653), which has no visibility flags. Regenerate it only with that binary:
//!
//! ```text
//! POT=/Users/admin/apps/potter/target/debug/pot
//! $POT --json init project
//! $POT --json apply project --file batch1.json   # node.create body box size 2;
//!                                                # modifier.create array_one type array count 2
//! $POT --json apply project --file batch2.json   # modifier.update array_one set.params.count 3
//! ```
//!
//! Only `scene.json` and `history/sha256/*/blob.json` are kept; `.potter/lock` is dropped and the
//! empty `.potter/tmp` directory is recreated by each test because git does not track it.

use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use potter_core::{
    cli::{ApplyArgs, Context, InspectArgs, UndoArgs},
    commands::{apply, inspect, redo, undo},
    hash,
    model::{DataBlock, Id, LightData, Modifier, Node, SceneDoc},
    ops,
    store::{HistoryRecord, Project},
};
use serde_json::{Value, json};
use tempfile::TempDir;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const SCENE_ID: &str = "00000000-0000-4000-8000-000000000000";
/// Revision 2 `info().hash` reported by the pre-change binary for the fixture.
const OLD_HASH: &str = "sha256:b2acec7357ecf23d15804adcbae41481cde794f4f1de3193792eabc244b96bbe";
/// Head history record id (blob address) reported by the pre-change binary.
const OLD_HEAD_ID: &str = "sha256:2d921bef8d505c3e6b81b9d2f8bf621bbe59f4fa099148665aa9c6b8622d1ffa";
/// `scene_hash` stored in the head history record by the pre-change binary.
const OLD_HEAD_SCENE_HASH: &str =
    "sha256:74acb93db8d0c988a02e2e5fc6b1e2eb8c20059be1abb10c81fd37111cd2bf38";
const OLD_REVISION: u64 = 2;
const OLD_BLOB_COUNT: usize = 3;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pr18l_old_project")
}

fn copy_dir(source: &Path, destination: &Path) -> TestResult {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let target = destination.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir(&entry.path(), &target)?;
        } else {
            fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}

/// Copies the fixture into a temporary project the test may commit to.
fn writable_fixture() -> TestResult<(TempDir, PathBuf)> {
    let directory = tempfile::tempdir()?;
    let scene = directory.path().join("scene");
    copy_dir(&fixture_dir(), &scene)?;
    // Atomic writes stage in `.potter/tmp`, which git cannot carry in the fixture.
    fs::create_dir_all(scene.join(".potter/tmp"))?;
    Ok((directory, scene))
}

fn history_blob_paths(scene: &Path) -> TestResult<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(scene.join("history/sha256"))? {
        paths.push(entry?.path().join("blob.json"));
    }
    paths.sort();
    Ok(paths)
}

/// Checks each history blob against its content address and checks that re-serializing the
/// record with the current code reproduces the stored bytes exactly.
fn verify_history_blobs(scene: &Path) -> TestResult<usize> {
    let paths = history_blob_paths(scene)?;
    for path in &paths {
        let raw = fs::read(path)?;
        let hex = path
            .parent()
            .and_then(Path::file_name)
            .and_then(|name| name.to_str())
            .ok_or("history blob directory must be named by its digest")?;
        let digest = format!("sha256:{hex}");
        assert_eq!(hash::sha256(&raw), digest, "{}", path.display());
        let record: HistoryRecord = serde_json::from_slice(&raw)?;
        let reserialized = hash::canonicalize(&serde_json::to_value(&record)?)?;
        assert_eq!(reserialized, raw, "{}", path.display());
    }
    Ok(paths.len())
}

/// Fails if any stored scene or history blob spells out a visibility flag.
fn assert_no_visibility_keys(scene: &Path) -> TestResult {
    let mut files = vec![scene.join("scene.json")];
    files.extend(history_blob_paths(scene)?);
    for file in files {
        let text = fs::read_to_string(&file)?;
        assert!(
            !text.contains("show_viewport") && !text.contains("show_render"),
            "default-true visibility flags must be omitted: {}",
            file.display()
        );
    }
    Ok(())
}

fn find_modifier<'a>(value: &'a Value, id: &str) -> Option<&'a Value> {
    match value {
        Value::Object(map) => {
            let listed = map
                .get("modifiers")
                .and_then(Value::as_array)
                .and_then(|modifiers| {
                    modifiers
                        .iter()
                        .find(|item| item.get("id").and_then(Value::as_str) == Some(id))
                });
            listed.or_else(|| map.values().find_map(|child| find_modifier(child, id)))
        }
        Value::Array(items) => items.iter().find_map(|child| find_modifier(child, id)),
        _ => None,
    }
}

fn array_count(scene: &Path) -> TestResult<u64> {
    let value = serde_json::to_value(Project::open(scene)?.doc())?;
    find_modifier(&value, "array_one")
        .and_then(|modifier| modifier["params"]["count"].as_u64())
        .ok_or_else(|| "array_one params.count must be present".into())
}

fn without_revision_and_history(mut value: Value) -> Value {
    if let Some(object) = value.as_object_mut() {
        object.remove("revision");
        object.remove("history");
    }
    value
}

#[test]
fn old_project_loads_with_its_recorded_hashes() -> TestResult {
    let scene = fixture_dir();
    let project = Project::open(&scene)?;
    let info = project.info()?;
    assert_eq!(info.revision, OLD_REVISION);
    assert_eq!(info.hash, OLD_HASH);

    let (records, _) = project.history_records()?;
    assert_eq!(records.len(), OLD_BLOB_COUNT);
    assert_eq!(records[0].0, OLD_HEAD_ID);
    assert_eq!(records[0].1.scene_hash, OLD_HEAD_SCENE_HASH);
    assert_eq!(verify_history_blobs(&scene)?, OLD_BLOB_COUNT);
    Ok(())
}

#[test]
fn old_project_stores_no_visibility_keys() -> TestResult {
    assert_no_visibility_keys(&fixture_dir())
}

#[test]
fn inspect_reports_both_flags_for_old_project() -> TestResult {
    let (_, value) = inspect::run(InspectArgs {
        scene: fixture_dir(),
        id: Some("body".to_owned()),
        tag: None,
        features: false,
        context: Context {
            scene_id: None,
            view_layer: None,
            frame: None,
        },
    })?;
    let modifier = find_modifier(&value, "array_one").ok_or("array_one missing from inspect")?;
    assert_eq!(modifier.get("show_viewport"), Some(&json!(true)));
    assert_eq!(modifier.get("show_render"), Some(&json!(true)));
    Ok(())
}

#[test]
fn apply_undo_redo_keep_old_project_hashes_verifiable() -> TestResult {
    let (directory, scene) = writable_fixture()?;
    let batch = directory.path().join("batch.json");
    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": OLD_REVISION,
            "operations": [
                {"op": "modifier.update", "target": {"id": "body"}, "id": "array_one",
                 "set": {"params": {"count": 4}}}
            ]
        }))?,
    )?;

    let (applied, _) = apply::run(ApplyArgs {
        scene: scene.clone(),
        file: batch,
        preview: None,
        size: None,
        dry_run: false,
    })?;
    let applied = applied.ok_or("apply must commit")?;
    assert_eq!(applied.revision, 3);
    assert_eq!(applied.hash, Project::open(&scene)?.info()?.hash);
    assert_eq!(verify_history_blobs(&scene)?, OLD_BLOB_COUNT + 1);
    assert_no_visibility_keys(&scene)?;
    assert_eq!(array_count(&scene)?, 4);
    let applied_doc = serde_json::to_value(Project::open(&scene)?.doc())?;

    let (undone, _) = undo::run(UndoArgs {
        scene: scene.clone(),
        base_revision: 3,
        steps: 1,
    })?;
    let undone = undone.ok_or("undo must commit")?;
    assert_eq!(undone.revision, 4);
    assert_eq!(undone.hash, Project::open(&scene)?.info()?.hash);
    assert_eq!(array_count(&scene)?, 3);
    assert_eq!(verify_history_blobs(&scene)?, OLD_BLOB_COUNT + 2);

    let (redone, _) = redo::run(UndoArgs {
        scene: scene.clone(),
        base_revision: 4,
        steps: 1,
    })?;
    let redone = redone.ok_or("redo must commit")?;
    assert_eq!(redone.revision, 5);
    assert_eq!(redone.hash, Project::open(&scene)?.info()?.hash);
    assert_eq!(array_count(&scene)?, 4);
    assert_eq!(verify_history_blobs(&scene)?, OLD_BLOB_COUNT + 3);
    assert_no_visibility_keys(&scene)?;

    let redone_doc = serde_json::to_value(Project::open(&scene)?.doc())?;
    // Undo and redo each record a new history entry, so only the scene content must match.
    assert_eq!(
        without_revision_and_history(redone_doc),
        without_revision_and_history(applied_doc)
    );
    Ok(())
}

#[test]
fn render_only_modifier_serializes_its_flag_and_round_trips() -> TestResult {
    let outcome = ops::apply_batch(
        &SceneDoc::new(SCENE_ID.to_owned()),
        &json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [
                {"op": "node.create", "id": "body", "kind": "box", "params": {"size": 2}},
                {"op": "modifier.create", "target": {"id": "body"}, "id": "array_one",
                 "type": "array", "show_render": false, "params": {"count": 2}}
            ]
        }),
    )?;
    let value = serde_json::to_value(&outcome.doc)?;
    let modifier = find_modifier(&value, "array_one").ok_or("array_one missing")?;
    assert_eq!(modifier.get("show_render"), Some(&json!(false)));
    assert!(
        modifier.get("show_viewport").is_none(),
        "a true show_viewport must be omitted"
    );

    let round_trip: SceneDoc = serde_json::from_value(value.clone())?;
    assert_eq!(serde_json::to_value(&round_trip)?, value);
    let typed = round_trip
        .nodes
        .values()
        .flat_map(|node| &node.modifiers)
        .find(|modifier| modifier.modifier_type == "array")
        .ok_or("typed array modifier missing")?;
    assert!(typed.show_viewport);
    assert!(!typed.show_render);
    Ok(())
}

#[test]
fn default_visibility_flags_are_omitted_and_read_as_true() -> TestResult {
    let modifier: Modifier = serde_json::from_value(json!({
        "id": "array_one",
        "type": "array",
        "name": "array",
        "enabled": true,
        "params": {}
    }))?;
    assert!(modifier.show_viewport);
    assert!(modifier.show_render);
    let value = serde_json::to_value(&modifier)?;
    assert!(value.get("show_viewport").is_none());
    assert!(value.get("show_render").is_none());
    Ok(())
}

#[test]
fn legacy_light_without_soft_falloff_keeps_its_canonical_hash() -> TestResult {
    let mut document = SceneDoc::new(SCENE_ID.to_owned());
    let data_id = Id::new("light_data")?;
    let node_id = Id::new("light_node")?;
    document.data_blocks.insert(
        data_id.clone(),
        DataBlock {
            data_type: "light".to_owned(),
            mesh: None,
            light: Some(LightData::default()),
            ..DataBlock::default()
        },
    );
    document.nodes.insert(
        node_id.clone(),
        Node {
            name: "Light".to_owned(),
            kind: "light".to_owned(),
            data: Some(data_id.clone()),
            ..Node::default()
        },
    );
    let root = document.scenes[&document.active_scene]
        .root_collection
        .clone();
    document
        .collections
        .get_mut(&root)
        .ok_or("root collection is missing")?
        .objects
        .push(node_id);

    // A schema-2 document has an actual light but predates this field.
    let mut legacy = serde_json::to_value(&document)?;
    legacy["data_blocks"][data_id.as_str()]["light"]
        .as_object_mut()
        .ok_or("light data is missing")?
        .remove("soft_falloff");
    let old_bytes = hash::canonicalize(&legacy)?;
    let old_hash = hash::sha256(&old_bytes);

    let loaded: SceneDoc = serde_json::from_value(legacy)?;
    assert!(
        loaded.data_blocks[&data_id]
            .light
            .as_ref()
            .is_some_and(|light| light.soft_falloff)
    );
    let current_bytes = hash::canonicalize(&serde_json::to_value(&loaded)?)?;
    assert_eq!(current_bytes, old_bytes);
    assert_eq!(hash::sha256(&current_bytes), old_hash);

    let mut disabled = loaded;
    disabled
        .data_blocks
        .get_mut(&data_id)
        .and_then(|data| data.light.as_mut())
        .ok_or("light data is missing")?
        .soft_falloff = false;
    let serialized = serde_json::to_value(disabled)?;
    assert_eq!(
        serialized["data_blocks"][data_id.as_str()]["light"]["soft_falloff"],
        false
    );
    Ok(())
}
