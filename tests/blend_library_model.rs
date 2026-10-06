#![expect(
    clippy::unwrap_used,
    reason = "integration tests use fixed valid scene and library fixtures"
)]

use std::{fs, path::Path, process::Command};

use potter::hash;
use serde_json::{Value, json};
use tempfile::tempdir;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn apply(scene: &Path, revision: u64, operations: &Value) -> std::process::Output {
    let directory = tempdir().unwrap();
    let batch = directory.path().join("operations.json");
    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": revision,
            "operations": operations
        }))
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

fn init(scene: &Path) {
    let output = pot().arg("init").arg(scene).arg("--json").output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
}

fn inspect(scene: &Path, id: &str) -> std::process::Output {
    pot()
        .arg("inspect")
        .arg(scene)
        .arg("--id")
        .arg(id)
        .arg("--json")
        .output()
        .unwrap()
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one scenario covers link, inspect, override, export, asset, and reload behavior"
)]
fn blend_library_links_override_reload_and_asset_status_are_native() {
    let directory = tempdir().unwrap();
    let scene = directory.path().join("scene");
    let blend = directory.path().join("props.blend");
    fs::write(&blend, b"first blend payload").unwrap();
    let copied_path = scene.join(format!(
        "assets/sha256/{}/props.blend",
        hash::sha256(b"first blend payload")
            .strip_prefix("sha256:")
            .unwrap()
    ));
    init(&scene);
    fs::create_dir_all(copied_path.parent().unwrap()).unwrap();

    let created = apply(
        &scene,
        0,
        &json!([
            {"op":"material.create","id":"linked_material"},
            {"op":"node.create","id":"linked_node","kind":"box","params":{}},
            {"op":"resource.pack","id":"library_resource","uri":blend},
            {"op":"resource.unpack","id":"library_resource","path":copied_path}
        ]),
    );
    assert!(
        created.status.success(),
        "{}",
        String::from_utf8_lossy(&created.stdout)
    );

    let registered = apply(
        &scene,
        1,
        &json!([{
            "op":"library.register",
            "id":"asset_library",
            "name":"Props",
            "kind":"blend",
            "uri":"//lib/props.blend",
            "resolved_path":copied_path,
            "resource":"library_resource",
            "items":[
                {"registry":"nodes","id":"linked_node","name":"Prop"},
                {"registry":"data_blocks","id":"linked_node_mesh","name":"PropMesh"},
                {"registry":"materials","id":"linked_material","name":"Paint"}
            ]
        }]),
    );
    assert!(
        registered.status.success(),
        "{}",
        String::from_utf8_lossy(&registered.stdout)
    );

    let library_info = inspect(&scene, "asset_library");
    assert!(
        library_info.status.success(),
        "{}",
        String::from_utf8_lossy(&library_info.stdout)
    );
    let library_info: Value = serde_json::from_slice(&library_info.stdout).unwrap();
    assert_eq!(library_info["result"]["items"][0]["kind"], "blend");
    assert_eq!(
        library_info["result"]["items"][0]["uri"],
        "//lib/props.blend"
    );
    assert_eq!(
        library_info["result"]["items"][0]["linked_ids"]["nodes"][0],
        "linked_node"
    );
    assert_eq!(
        library_info["result"]["items"][0]["resource"],
        "library_resource"
    );

    for (id, library_name) in [
        ("linked_node", "Prop"),
        ("linked_node_mesh", "PropMesh"),
        ("linked_material", "Paint"),
    ] {
        let output = inspect(&scene, id);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stdout)
        );
        let envelope: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(
            envelope["result"]["items"][0]["library"]["id"],
            "asset_library"
        );
        assert_eq!(envelope["result"]["items"][0]["library_name"], library_name);
        assert_eq!(envelope["result"]["items"][0]["editable"], false);
    }
    let linked = inspect(&scene, "linked_node");
    let linked: Value = serde_json::from_slice(&linked.stdout).unwrap();
    assert_eq!(linked["result"]["items"][0]["library"]["name"], "Props");
    assert_eq!(linked["result"]["items"][0]["library_name"], "Prop");

    let gltf_path = directory.path().join("linked.gltf");
    let gltf_export = pot()
        .arg("export")
        .arg(&scene)
        .args(["--format", "gltf", "--out"])
        .arg(&gltf_path)
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        gltf_export.status.success(),
        "{}",
        String::from_utf8_lossy(&gltf_export.stdout)
    );
    let gltf: Value = serde_json::from_slice(&fs::read(gltf_path).unwrap()).unwrap();
    assert_eq!(gltf["meshes"].as_array().unwrap().len(), 1);

    let obj_path = directory.path().join("linked.obj");
    let obj_export = pot()
        .arg("export")
        .arg(&scene)
        .args(["--format", "obj", "--out"])
        .arg(&obj_path)
        .args(["--allow-lossy", "--json"])
        .output()
        .unwrap();
    assert!(
        obj_export.status.success(),
        "{}",
        String::from_utf8_lossy(&obj_export.stdout)
    );
    let obj = fs::read_to_string(obj_path).unwrap();
    assert!(obj.lines().any(|line| line.starts_with("f ")));

    let rejected = apply(
        &scene,
        2,
        &json!([{
            "op":"node.update",
            "target":{"id":"linked_node"},
            "set":{"name":"forbidden"}
        }]),
    );
    assert_eq!(rejected.status.code(), Some(2));
    let rejected: Value = serde_json::from_slice(&rejected.stdout).unwrap();
    assert_eq!(rejected["error"]["code"], "INVALID_OPERATION");
    assert_eq!(
        rejected["error"]["details"]["reason"],
        "linked data is read-only"
    );

    let overridden = apply(
        &scene,
        2,
        &json!([{
            "op":"library.override",
            "target":{"id":"linked_node"},
            "id":"local_override",
            "operations":[{"op":"set","path":"properties.label","value":"local"}]
        }]),
    );
    assert!(
        overridden.status.success(),
        "{}",
        String::from_utf8_lossy(&overridden.stdout)
    );
    let override_info = inspect(&scene, "local_override");
    assert!(
        override_info.status.success(),
        "{}",
        String::from_utf8_lossy(&override_info.stdout)
    );
    let override_info: Value = serde_json::from_slice(&override_info.stdout).unwrap();
    assert_eq!(
        override_info["result"]["items"][0]["properties"]["label"],
        "local"
    );
    assert_eq!(
        override_info["result"]["items"][0]["library_override"]["reference_id"],
        "linked_node"
    );
    let persisted: Value =
        serde_json::from_slice(&fs::read(scene.join("scene.json")).unwrap()).unwrap();
    assert_eq!(
        persisted["libraries"]["asset_library"]["overrides"][0]["reference_id"],
        "linked_node"
    );
    assert_eq!(
        persisted["libraries"]["asset_library"]["overrides"][0]["properties"][0]["path"],
        "properties.label"
    );
    assert_eq!(
        persisted["libraries"]["asset_library"]["overrides"][0]["properties"][0]["operation"],
        "replace"
    );

    let rejected_override = apply(
        &scene,
        3,
        &json!([{
            "op":"library.override",
            "target":{"id":"linked_node"},
            "id":"invalid_override",
            "operations":[{"op":"set","path":"selectable","value":false}]
        }]),
    );
    assert_eq!(rejected_override.status.code(), Some(2));
    let rejected_override: Value = serde_json::from_slice(&rejected_override.stdout).unwrap();
    assert_eq!(rejected_override["error"]["code"], "INVALID_OPERATION");

    fs::write(&copied_path, b"changed blend payload").unwrap();
    let changed = pot()
        .arg("assets")
        .arg(&scene)
        .arg("--check")
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        changed.status.success(),
        "{}",
        String::from_utf8_lossy(&changed.stdout)
    );
    let changed: Value = serde_json::from_slice(&changed.stdout).unwrap();
    let changed_library = changed["result"]["assets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|asset| asset["library_id"] == "asset_library")
        .unwrap();
    assert_eq!(changed_library["changed"], true);
    let blocked = inspect(&scene, "linked_node");
    assert_eq!(blocked.status.code(), Some(4));
    let blocked: Value = serde_json::from_slice(&blocked.stdout).unwrap();
    assert_eq!(blocked["error"]["code"], "ASSET_CHANGED");

    let reloaded = apply(
        &scene,
        3,
        &json!([{"op":"library.reload","id":"asset_library"}]),
    );
    assert!(
        reloaded.status.success(),
        "{}",
        String::from_utf8_lossy(&reloaded.stdout)
    );
    let evaluated = inspect(&scene, "linked_node");
    assert!(
        evaluated.status.success(),
        "{}",
        String::from_utf8_lossy(&evaluated.stdout)
    );

    fs::remove_file(&copied_path).unwrap();
    let missing = pot()
        .arg("assets")
        .arg(&scene)
        .arg("--check")
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        missing.status.success(),
        "{}",
        String::from_utf8_lossy(&missing.stdout)
    );
    let missing: Value = serde_json::from_slice(&missing.stdout).unwrap();
    let missing_library = missing["result"]["assets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|asset| asset["library_id"] == "asset_library")
        .unwrap();
    assert_eq!(missing_library["missing"], true);
}
