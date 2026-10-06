#![expect(clippy::expect_used, reason = "integration test fixtures")]

use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use serde_json::{Value, json};
use tempfile::tempdir;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn blender_executable() -> Option<PathBuf> {
    fn usable(path: PathBuf) -> Option<PathBuf> {
        path.is_file()
            .then(|| Command::new(&path).arg("--version").output().ok())
            .flatten()
            .filter(|output| output.status.success())
            .map(|_| path)
    }

    if let Some(path) = env::var_os("POTTER_BLENDER") {
        return usable(PathBuf::from(path));
    }
    if let Some(path) = env::split_paths(&env::var_os("PATH")?)
        .map(|dir| dir.join("blender"))
        .find_map(usable)
    {
        return Some(path);
    }
    usable(PathBuf::from(
        "/Applications/Blender.app/Contents/MacOS/Blender",
    ))
}

fn run_ok(command: &mut Command, operation: &str) -> Result<Output, Box<dyn Error>> {
    let output = command.output()?;
    assert!(
        output.status.success(),
        "{operation} failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(output)
}

fn id_sets(document: &Value) -> BTreeMap<String, BTreeSet<String>> {
    let mut result = BTreeMap::new();
    for registry in [
        "scenes",
        "collections",
        "nodes",
        "data_blocks",
        "materials",
        "images",
        "worlds",
        "node_groups",
        "actions",
    ] {
        let ids = document[registry]
            .as_object()
            .expect("scene registry is an object")
            .keys()
            .cloned()
            .collect();
        result.insert(registry.to_owned(), ids);
    }

    let mut view_layer_ids = BTreeSet::new();
    for scene in document["scenes"]
        .as_object()
        .expect("scenes are an object")
        .values()
    {
        view_layer_ids.extend(
            scene["view_layers"]
                .as_object()
                .expect("view layers are an object")
                .keys()
                .cloned(),
        );
    }
    result.insert("view_layers".to_owned(), view_layer_ids);

    let mut action_slot_ids = BTreeSet::new();
    for action in document["actions"]
        .as_object()
        .expect("actions are an object")
        .values()
    {
        action_slot_ids.extend(
            action["slots"]
                .as_array()
                .expect("action slots are an array")
                .iter()
                .map(|slot| slot["id"].as_str().expect("slot ID is a string").to_owned()),
        );
    }
    result.insert("action_slots".to_owned(), action_slot_ids);

    let mut graph_node_ids = BTreeSet::new();
    for group in document["node_groups"]
        .as_object()
        .expect("node groups are an object")
        .values()
    {
        graph_node_ids.extend(
            group["nodes"]
                .as_object()
                .expect("graph nodes are an object")
                .keys()
                .cloned(),
        );
    }
    result.insert("graph_nodes".to_owned(), graph_node_ids);
    result
}

fn add_representative_ids(scene: &Path) -> Result<(), Box<dyn Error>> {
    let document_path = scene.join("scene.json");
    let mut document: Value = serde_json::from_slice(&fs::read(&document_path)?)?;
    document["scenes"]["scene_main"]["camera"] = json!("camera_node");
    document["scenes"]["scene_main"]["world"] = json!("world_main");
    document["scenes"]["scene_main"]["view_layers"] = json!({
        "view_main": {"name": "View Layer", "excluded_collections": []},
        "view_review": {"name": "Review Layer", "excluded_collections": []}
    });
    document["actions"]["action_main"] = json!({
        "name": "Persistent Action",
        "fcurves": [],
        "slots": [{"id": "action_slot_main", "node": "body"}]
    });
    document["nodes"]["body"]["action"] = json!("action_main");
    document["node_groups"]["graph_main"] = json!({
        "name": "Persistent Geometry",
        "kind": "geometry",
        "interface": {
            "inputs": [],
            "outputs": [{
                "id": "geometry_output",
                "name": "Geometry",
                "socket_type": "geometry",
                "default": null
            }]
        },
        "nodes": {
            "graph_node_main": {
                "type": "NodeGroupOutput",
                "name": "Group Output",
                "location": [200.0, 0.0],
                "properties": {},
                "inputs": {}
            }
        },
        "links": []
    });
    document["worlds"]["world_main"]["node_tree"] = json!("graph_world");
    document["node_groups"]["graph_world"] = json!({
        "name": "Persistent World Nodes",
        "kind": "shader",
        "interface": {"inputs": [], "outputs": []},
        "nodes": {
            "world_background": {
                "type": "ShaderNodeBackground",
                "name": "Background",
                "location": [0.0, 0.0],
                "properties": {},
                "inputs": {"Color": [0.1, 0.2, 0.3, 1.0], "Strength": 0.5}
            },
            "world_output": {
                "type": "ShaderNodeOutputWorld",
                "name": "World Output",
                "location": [300.0, 0.0],
                "properties": {},
                "inputs": {}
            }
        },
        "links": [{
            "from_node": "world_background",
            "from_socket": "Background",
            "to_node": "world_output",
            "to_socket": "Surface"
        }]
    });
    document["nodes"]["body"]["modifiers"] = json!([{
        "id": "modifier_graph",
        "type": "nodes",
        "name": "Persistent Graph Modifier",
        "enabled": true,
        "params": {"node_group": "graph_main"}
    }]);
    fs::write(document_path, serde_json::to_vec_pretty(&document)?)?;
    Ok(())
}

#[test]
fn blender_open_save_replace_preserves_all_represented_ids() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Blender ID round-trip; no executable Blender was found");
        return Ok(());
    };

    let directory = tempdir()?;
    let scene = directory.path().join("source");
    let mut init = pot();
    init.args(["init"]).arg(&scene).arg("--json");
    run_ok(&mut init, "initializing the scene")?;

    let first_operations = scene.with_extension("first.json");
    fs::write(&first_operations, include_str!("fixtures/first.json"))?;
    let mut apply_first = pot();
    apply_first
        .args(["apply"])
        .arg(&scene)
        .args(["--file"])
        .arg(&first_operations)
        .arg("--json");
    run_ok(&mut apply_first, "creating the mesh and material")?;

    let second_operations = scene.with_extension("second.json");
    fs::write(
        &second_operations,
        serde_json::to_vec_pretty(&json!({
            "schema_version": 1,
            "base_revision": 1,
            "operations": [
                {"op": "collection.create", "id": "collection_secondary", "name": "Secondary"},
                {"op": "camera.create", "id": "camera_node", "name": "Camera"},
                {"op": "light.create", "id": "light_node", "name": "Spot Light", "light_type": "spot", "spot_size": 0.75, "spot_blend": 0.2},
                {"op": "world.create", "id": "world_main", "color": [0.1, 0.2, 0.3], "strength": 0.5},
                {"op": "image.create", "id": "image_main", "name": "Round Trip Image", "width": 2, "height": 2}
            ]
        }))?,
    )?;
    let mut apply_second = pot();
    apply_second
        .args(["apply"])
        .arg(&scene)
        .args(["--file"])
        .arg(&second_operations)
        .arg("--json");
    run_ok(&mut apply_second, "creating the remaining represented IDs")?;
    add_representative_ids(&scene)?;

    let source_document: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let expected_ids = id_sets(&source_document);
    assert!(!expected_ids["data_blocks"].is_empty());
    for (registry, id) in [
        ("scenes", "scene_main"),
        ("collections", "collection_secondary"),
        ("nodes", "camera_node"),
        ("nodes", "light_node"),
        ("materials", "clay"),
        ("images", "image_main"),
        ("worlds", "world_main"),
        ("node_groups", "graph_main"),
        ("node_groups", "graph_world"),
        ("actions", "action_main"),
        ("view_layers", "view_review"),
        ("action_slots", "action_slot_main"),
        ("graph_nodes", "graph_node_main"),
        ("graph_nodes", "world_background"),
        ("graph_nodes", "world_output"),
    ] {
        assert!(
            expected_ids[registry].contains(id),
            "missing {registry} ID {id}"
        );
    }

    let blend_file = directory.path().join("round-trip.blend");
    let mut export = pot();
    export
        .args(["export"])
        .arg(&scene)
        .args(["--format", "blend", "--out"])
        .arg(&blend_file)
        .args(["--blender"])
        .arg(&blender)
        .arg("--json");
    run_ok(&mut export, "exporting the Blender file")?;

    let mut open_and_save = Command::new(&blender);
    open_and_save
        .args(["--background", "--disable-autoexec"])
        .arg(&blend_file)
        .args(["--python-exit-code", "1", "--python-expr"])
        .arg("import bpy; bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)");
    run_ok(&mut open_and_save, "opening and saving the Blender file")?;

    let mut import = pot();
    import
        .args(["import"])
        .arg(&scene)
        .args(["--file"])
        .arg(&blend_file)
        .args([
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "2",
            "--blender",
        ])
        .arg(&blender)
        .arg("--json");
    run_ok(&mut import, "replacing the scene from Blender")?;

    let imported_document: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    assert_eq!(id_sets(&imported_document), expected_ids);
    let source_light_id = source_document["nodes"]["light_node"]["data"]
        .as_str()
        .ok_or("source light data ID is missing")?;
    let imported_light_id = imported_document["nodes"]["light_node"]["data"]
        .as_str()
        .ok_or("imported light data ID is missing")?;
    let source_light = &source_document["data_blocks"][source_light_id]["light"];
    let imported_light = &imported_document["data_blocks"][imported_light_id]["light"];
    assert_eq!(source_light["light_type"], "spot");
    assert_eq!(source_light["spot_size"], json!(0.75));
    assert_eq!(source_light["spot_blend"], json!(0.2));
    for field in ["spot_size", "spot_blend"] {
        let expected = source_light[field]
            .as_f64()
            .ok_or("source spot-light setting is missing")?;
        let actual = imported_light[field]
            .as_f64()
            .ok_or("imported spot-light setting is missing")?;
        assert!(
            (actual - expected).abs() < 1.0e-6,
            "spot light {field} changed from {expected} to {actual}"
        );
    }
    Ok(())
}
