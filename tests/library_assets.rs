#![expect(
    clippy::unwrap_used,
    reason = "integration tests use fixtures with fixed valid data"
)]

use std::{fs, path::Path, process::Command};

use serde_json::{Value, json};
use tempfile::tempdir;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn run_apply(scene: &Path, operations: &Value) -> std::process::Output {
    run_apply_at(scene, 0, operations)
}

fn run_apply_at(scene: &Path, base_revision: u64, operations: &Value) -> std::process::Output {
    let directory = tempdir().unwrap();
    let batch = directory.path().join("operations.json");
    fs::write(
        &batch,
        serde_json::to_vec(
            &json!({"schema_version":1,"base_revision":base_revision,"operations":operations}),
        )
        .unwrap(),
    )
    .unwrap();
    pot()
        .arg("apply")
        .arg(scene)
        .arg("--file")
        .arg(batch)
        .arg("--json")
        .output()
        .unwrap()
}

fn init(path: &Path) {
    let output = pot().arg("init").arg(path).arg("--json").output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
}

#[test]
fn text_properties_and_asset_metadata_round_trip() {
    let directory = tempdir().unwrap();
    let scene = directory.path().join("scene");
    init(&scene);

    let output = run_apply(
        &scene,
        &json!([
            {"op":"node.create","id":"body","kind":"empty"},
            {"op":"text.create","id":"script","body":"print('hello')","language":"python"},
            {"op":"property.set","target":{"id":"body"},"name":"opacity","type":"float","subtype":"factor","value":0.75,"min":0.0,"max":1.0},
            {"op":"asset.catalog_create","id":"models","name":"Models"},
            {"op":"asset.mark","target":{"id":"body"},"catalog_id":"models","description":"A linked test asset","author":"Potter","license":"CC0","tags":["demo"]}
        ]),
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let document: Value =
        serde_json::from_slice(&fs::read(scene.join("scene.json")).unwrap()).unwrap();
    assert_eq!(document["data_blocks"]["script"]["type"], "script_text");
    assert_eq!(
        document["data_blocks"]["script"]["text"]["body"],
        "print('hello')"
    );
    assert_eq!(
        document["nodes"]["body"]["properties"]["opacity"]["value"],
        0.75
    );
    assert_eq!(
        document["nodes"]["body"]["properties"]["opacity"]["type"],
        "float"
    );
    assert_eq!(
        document["compatibility"]["assets"]["nodes:body"]["catalog_id"],
        "models"
    );

    let bad = run_apply_at(
        &scene,
        1,
        &json!([{"op":"property.set","target":{"id":"body"},"name":"opacity","type":"float","value":1.5,"min":0.0,"max":1.0}]),
    );
    assert_eq!(bad.status.code(), Some(2));
    let envelope: Value = serde_json::from_slice(&bad.stdout).unwrap();
    assert_eq!(envelope["error"]["code"], "INVALID_OPERATION");
}

#[test]
fn resources_text_properties_assets_and_extensions_round_trip() {
    let directory = tempdir().unwrap();
    let scene = directory.path().join("scene");
    let source_file = directory.path().join("resource.bin");
    let unpacked_file = directory.path().join("unpacked.bin");
    let payload = [0_u8, 1, 127, 255];
    fs::write(&source_file, payload).unwrap();
    init(&scene);

    let packed = run_apply(
        &scene,
        &json!([
            {"op":"node.create","id":"body","kind":"empty"},
            {"op":"text.create","id":"script","body":"old","language":"python"},
            {"op":"property.set","target":{"id":"body"},"name":"weight","type":"int","subtype":"factor","value":5,"min":0,"max":10},
            {"op":"asset.mark","target":{"id":"body"},"description":"before","tags":["demo"]},
            {"op":"resource.pack","id":"payload","uri":source_file},
            {"op":"extension.register","id":"native","version":"1.0.0","permissions":["read"],"features":["mesh.edit"]}
        ]),
    );
    assert!(
        packed.status.success(),
        "{}",
        String::from_utf8_lossy(&packed.stdout)
    );
    let document: Value =
        serde_json::from_slice(&fs::read(scene.join("scene.json")).unwrap()).unwrap();
    assert_eq!(document["resources"]["payload"]["packed"], true);
    assert_eq!(
        document["resources"]["payload"]["hash"]
            .as_str()
            .unwrap()
            .len(),
        71
    );
    assert_eq!(
        document["compatibility"]["extensions"]["native"]["scripts_executed"],
        false
    );
    fs::remove_file(&source_file).unwrap();
    let checked = pot()
        .arg("assets")
        .arg(&scene)
        .arg("--check")
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        checked.status.success(),
        "{}",
        String::from_utf8_lossy(&checked.stdout)
    );
    let report: Value = serde_json::from_slice(&checked.stdout).unwrap();
    assert_eq!(report["result"]["summary"]["missing"], 0);
    assert_eq!(report["result"]["summary"]["changed"], 0);

    let updated = run_apply_at(
        &scene,
        1,
        &json!([
            {"op":"text.update","target":{"id":"script"},"set":{"body":"new","language":"javascript"}},
            {"op":"asset.update","target":{"id":"body"},"set":{"description":"after"}},
            {"op":"resource.unpack","id":"payload","path":unpacked_file}
        ]),
    );
    assert!(
        updated.status.success(),
        "{}",
        String::from_utf8_lossy(&updated.stdout)
    );
    assert_eq!(fs::read(&unpacked_file).unwrap(), payload);
    let document: Value =
        serde_json::from_slice(&fs::read(scene.join("scene.json")).unwrap()).unwrap();
    assert_eq!(document["data_blocks"]["script"]["text"]["body"], "new");
    assert_eq!(
        document["data_blocks"]["script"]["descriptor"]["params"]["language"],
        "javascript"
    );
    assert_eq!(
        document["compatibility"]["assets"]["nodes:body"]["description"],
        "after"
    );
    assert_eq!(document["resources"]["payload"]["packed"], false);

    let deleted = run_apply_at(
        &scene,
        2,
        &json!([
            {"op":"text.delete","target":{"id":"script"}},
            {"op":"property.delete","target":{"id":"body"},"name":"weight"},
            {"op":"asset.clear","target":{"id":"body"}}
        ]),
    );
    assert!(
        deleted.status.success(),
        "{}",
        String::from_utf8_lossy(&deleted.stdout)
    );
    let document: Value =
        serde_json::from_slice(&fs::read(scene.join("scene.json")).unwrap()).unwrap();
    assert!(document["data_blocks"].get("script").is_none());
    assert!(
        document["nodes"]["body"]["properties"]
            .get("weight")
            .is_none()
    );
    assert!(
        document["compatibility"]["assets"]
            .get("nodes:body")
            .is_none()
    );
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one integration scenario covers the complete linked-library lifecycle"
)]
fn linked_library_detects_changed_source_until_reload() {
    let directory = tempdir().unwrap();
    let source = directory.path().join("source");
    let destination = directory.path().join("destination");
    init(&source);
    init(&destination);

    let create = directory.path().join("create.json");
    fs::write(
        &create,
        serde_json::to_vec(&json!({
            "schema_version":1,
            "base_revision":0,
            "operations":[
                {"op":"material.create","id":"source_node"},
                {"op":"node.create","id":"source_node","kind":"empty","material":"source_node"}
            ]
        }))
        .unwrap(),
    )
    .unwrap();
    let output = pot()
        .arg("apply")
        .arg(&source)
        .arg("--file")
        .arg(&create)
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let local_node = directory.path().join("local-node.json");
    fs::write(
        &local_node,
        serde_json::to_vec(&json!({
            "schema_version":1,
            "base_revision":0,
            "operations":[{"op":"node.create","id":"source_node","kind":"empty"}]
        }))
        .unwrap(),
    )
    .unwrap();
    let output = pot()
        .arg("apply")
        .arg(&destination)
        .arg("--file")
        .arg(&local_node)
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );

    let linked = directory.path().join("link.json");
    fs::write(&linked, serde_json::to_vec(&json!({
        "schema_version":1,
        "base_revision":1,
        "operations":[{"op":"library.link","id":"source_lib","uri":source,"items":[{"registry":"nodes","id":"source_node"}]}]
    })).unwrap()).unwrap();
    let output = pot()
        .arg("apply")
        .arg(&destination)
        .arg("--file")
        .arg(&linked)
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let destination_doc: Value =
        serde_json::from_slice(&fs::read(destination.join("scene.json")).unwrap()).unwrap();
    assert_eq!(
        destination_doc["nodes"]["source_lib__source_node"]["properties"]["editable"],
        false
    );
    assert_eq!(
        destination_doc["nodes"]["source_lib__source_node"]["properties"]["library"],
        "source_lib"
    );
    let read_only = run_apply_at(
        &destination,
        2,
        &json!([{"op":"node.update","target":{"id":"source_lib__source_node"},"set":{"name":"forbidden"}}]),
    );
    assert_eq!(read_only.status.code(), Some(2));
    let error: Value = serde_json::from_slice(&read_only.stdout).unwrap();
    assert_eq!(error["error"]["code"], "INVALID_OPERATION");

    let library_edits = directory.path().join("library-edits.json");
    fs::write(&library_edits, serde_json::to_vec(&json!({
        "schema_version":1,
        "base_revision":2,
        "operations":[
            {"op":"library.override","target":{"id":"source_lib__source_node"},"id":"overridden_node","operations":[{"op":"set","path":"name","value":"Override"}]},
            {"op":"library.append","library_id":"source_lib","items":[{"registry":"nodes","id":"source_node"}]}
        ]
    })).unwrap()).unwrap();
    let edited = pot()
        .arg("apply")
        .arg(&destination)
        .arg("--file")
        .arg(&library_edits)
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        edited.status.success(),
        "{}",
        String::from_utf8_lossy(&edited.stdout)
    );
    let destination_doc: Value =
        serde_json::from_slice(&fs::read(destination.join("scene.json")).unwrap()).unwrap();
    assert_eq!(
        destination_doc["nodes"]["overridden_node"]["name"],
        "Override"
    );
    assert_eq!(
        destination_doc["nodes"]["overridden_node"]["properties"]["library_override"]["operations"]
            [0]["path"],
        "name"
    );
    assert_eq!(
        destination_doc["nodes"]["source_node_copy1"]["materials"][0],
        "source_node"
    );
    let update = directory.path().join("update.json");
    fs::write(&update, serde_json::to_vec(&json!({
        "schema_version":1,
        "base_revision":1,
        "operations":[{"op":"node.update","target":{"id":"source_node"},"set":{"name":"changed"}}]
    })).unwrap()).unwrap();
    let changed = pot()
        .arg("apply")
        .arg(&source)
        .arg("--file")
        .arg(&update)
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        changed.status.success(),
        "{}",
        String::from_utf8_lossy(&changed.stdout)
    );

    let assets = pot()
        .arg("assets")
        .arg(&destination)
        .arg("--check")
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        assets.status.success(),
        "{}",
        String::from_utf8_lossy(&assets.stdout)
    );
    let report: Value = serde_json::from_slice(&assets.stdout).unwrap();
    assert_eq!(report["result"]["summary"]["changed"], 1);
    let blocked = pot()
        .arg("inspect")
        .arg(&destination)
        .arg("--json")
        .output()
        .unwrap();
    assert_eq!(blocked.status.code(), Some(4));
    let error: Value = serde_json::from_slice(&blocked.stdout).unwrap();
    assert_eq!(error["error"]["code"], "ASSET_CHANGED");

    let reload = directory.path().join("reload.json");
    fs::write(
        &reload,
        serde_json::to_vec(&json!({
            "schema_version":1,
            "base_revision":3,
            "operations":[{"op":"library.reload","id":"source_lib"}]
        }))
        .unwrap(),
    )
    .unwrap();
    let reloaded = pot()
        .arg("apply")
        .arg(&destination)
        .arg("--file")
        .arg(&reload)
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        reloaded.status.success(),
        "{}",
        String::from_utf8_lossy(&reloaded.stdout)
    );
    let evaluated = pot()
        .arg("inspect")
        .arg(&destination)
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        evaluated.status.success(),
        "{}",
        String::from_utf8_lossy(&evaluated.stdout)
    );
    let assets = pot()
        .arg("assets")
        .arg(&destination)
        .arg("--check")
        .arg("--json")
        .output()
        .unwrap();
    let report: Value = serde_json::from_slice(&assets.stdout).unwrap();
    assert_eq!(report["result"]["summary"]["changed"], 0);
    let relocated_source = directory.path().join("relocated-source");
    fs::create_dir(&relocated_source).unwrap();
    fs::copy(
        source.join("scene.json"),
        relocated_source.join("scene.json"),
    )
    .unwrap();
    let relocate = directory.path().join("relocate.json");
    let relocated_scene = relocated_source.join("scene.json");
    fs::write(
        &relocate,
        serde_json::to_vec(&json!({
            "schema_version":1,
            "base_revision":4,
            "operations":[{"op":"library.relocate","id":"source_lib","uri":relocated_scene}]
        }))
        .unwrap(),
    )
    .unwrap();
    let relocated = pot()
        .arg("apply")
        .arg(&destination)
        .arg("--file")
        .arg(&relocate)
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        relocated.status.success(),
        "{}",
        String::from_utf8_lossy(&relocated.stdout)
    );
    let destination_doc: Value =
        serde_json::from_slice(&fs::read(destination.join("scene.json")).unwrap()).unwrap();
    assert_eq!(
        destination_doc["libraries"]["source_lib"]["uri"],
        relocated_scene.to_str().unwrap()
    );
}
