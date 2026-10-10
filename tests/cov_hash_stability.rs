#![expect(clippy::unwrap_used, reason = "tests")]

use std::{error::Error, fs, path::Path, process::Command};

use potter_core::{
    hash,
    model::{Id, Node, SceneDoc, Transform},
    store::{HistoryDirection, Project},
};
use proptest::prelude::*;

fn document_with_camera(angles: [f64; 3]) -> SceneDoc {
    let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
    let camera = Node {
        kind: "camera".to_owned(),
        transform: Transform::from_rotation_deg([4.0, -4.0, 3.0], angles, [1.0; 3]),
        ..Node::default()
    };
    doc.nodes.insert(Id::new("camera").unwrap(), camera);
    doc
}

fn canonical_scene_hash(doc: &SceneDoc) -> Result<String, Box<dyn Error>> {
    let value = serde_json::to_value(doc)?;
    Ok(hash::sha256(&hash::canonicalize(&value)?))
}

use serde_json::{Value, json};
use tempfile::tempdir;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn run_json(command: &mut Command) -> Result<Value, Box<dyn Error>> {
    let output = command.output()?;
    assert!(
        output.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn scene_hash(response: &Value) -> &str {
    response["scene"]["hash"].as_str().unwrap()
}

fn inspect(scene: &Path) -> Result<Value, Box<dyn Error>> {
    run_json(pot().arg("inspect").arg(scene).arg("--json"))
}

#[test]
fn apply_undo_and_redo_hashes_match_reopened_scene() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    run_json(pot().arg("init").arg(&scene).arg("--json"))?;

    let batch = json!({
        "schema_version": 1,
        "base_revision": 0,
        "operations": [
            {"op":"node.create","id":"body","kind":"box","params":{"size":1.0},"transform":{"translation":[0.0,0.0,0.4],"scale":[1.0,0.6,0.8]}},
            {"op":"camera.create","id":"cam","transform":{"translation":[4.0,-4.0,3.0],"rotation_deg":[63.0,0.0,45.0]}},
            {"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"cam"}},
            {"op":"light.create","id":"sun","light_type":"sun","energy":3.0},
            {"op":"action.create","id":"body_anim"},
            {"op":"node.update","target":{"id":"body"},"set":{"action":"body_anim"}},
            {"op":"keyframe.insert","target":{"id":"body"},"path":"transform.translation","index":2,"frame":1,"value":0.4,"interpolation":"bezier"},
            {"op":"keyframe.insert","target":{"id":"body"},"path":"transform.translation","index":2,"frame":24,"value":2.0,"interpolation":"bezier"}
        ]
    });
    let batch_path = directory.path().join("batch.json");
    fs::write(&batch_path, serde_json::to_vec(&batch)?)?;

    let applied = run_json(
        pot()
            .arg("apply")
            .arg(&scene)
            .arg("--file")
            .arg(&batch_path)
            .arg("--json"),
    )?;
    assert_eq!(scene_hash(&applied), scene_hash(&inspect(&scene)?));

    let undone = run_json(
        pot()
            .arg("undo")
            .arg(&scene)
            .args(["--base-revision", "1", "--json"]),
    )?;
    assert_eq!(scene_hash(&undone), scene_hash(&inspect(&scene)?));

    let redone = run_json(
        pot()
            .arg("redo")
            .arg(&scene)
            .args(["--base-revision", "2", "--json"]),
    )?;
    assert_eq!(scene_hash(&redone), scene_hash(&inspect(&scene)?));
    Ok(())
}

proptest! {
    #[test]
    fn scene_hash_is_stable_after_serialization_round_trip(
        angles in prop::array::uniform3(-180.0_f64..180.0),
    ) {
        let doc = document_with_camera(angles);
        let bytes = serde_json::to_vec_pretty(&doc).unwrap();
        let reloaded: SceneDoc = serde_json::from_slice(&bytes).unwrap();
        prop_assert_eq!(
            canonical_scene_hash(&doc).unwrap(),
            canonical_scene_hash(&reloaded).unwrap(),
        );
    }
}

#[test]
fn import_undo_and_redo_store_hashes_match_reopened_projects() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let path = directory.path().join("scene");
    let mut project = Project::init(&path)?;

    let mut imported = project.doc().clone();
    let camera = Node {
        kind: "camera".to_owned(),
        transform: Transform::from_rotation_deg([4.0, -4.0, 3.0], [63.0, 0.0, 45.0], [1.0; 3]),
        ..Node::default()
    };
    imported.nodes.insert(Id::new("camera")?, camera);

    let imported_info = project.commit_import(imported, json!([]), json!({}))?;
    assert_eq!(imported_info.hash, Project::open(&path)?.info()?.hash);

    let (undo_snapshot, undo_target) = project.history_target(HistoryDirection::Undo, 1)?;
    let undone_info = project.commit_history(
        HistoryDirection::Undo,
        undo_snapshot,
        undo_target,
        1,
        json!({}),
    )?;
    assert_eq!(undone_info.hash, Project::open(&path)?.info()?.hash);

    let (redo_snapshot, redo_target) = project.history_target(HistoryDirection::Redo, 1)?;
    let redone_info = project.commit_history(
        HistoryDirection::Redo,
        redo_snapshot,
        redo_target,
        1,
        json!({}),
    )?;
    assert_eq!(redone_info.hash, Project::open(&path)?.info()?.hash);
    Ok(())
}
