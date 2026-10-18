use std::{
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

fn run_ok(command: &mut Command, action: &str) -> Result<Output, Box<dyn Error>> {
    let output = command.output()?;
    if !output.status.success() {
        return Err(format!(
            "{action} failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        )
        .into());
    }
    Ok(output)
}

fn run_json(command: &mut Command, action: &str) -> Result<Value, Box<dyn Error>> {
    command.arg("--json");
    Ok(serde_json::from_slice(&run_ok(command, action)?.stdout)?)
}

fn document(project: &Path) -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_slice(&fs::read(
        project.join("scene.json"),
    )?)?)
}

fn apply(project: &Path, batch_file: &Path, operations: Value) -> Result<(), Box<dyn Error>> {
    let revision = document(project)?["revision"]
        .as_u64()
        .ok_or_else(|| std::io::Error::other("scene revision is missing"))?;
    fs::write(
        batch_file,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": revision,
            "operations": operations,
        }))?,
    )?;
    run_json(
        pot()
            .arg("apply")
            .arg(project)
            .arg("--file")
            .arg(batch_file),
        "applying scene operations",
    )?;
    Ok(())
}

fn bake(
    project: &Path,
    frames: Option<&str>,
    mode: Option<&str>,
    delete: bool,
) -> Result<Value, Box<dyn Error>> {
    let mut command = pot();
    command.arg("bake").arg(project).args([
        "--kind",
        "node-bake",
        "--target",
        "body",
        "--modifier",
        "geometry_modifier",
        "--node",
        "bake",
    ]);
    if let Some(frames) = frames {
        command.args(["--frames", frames]);
    }
    if let Some(mode) = mode {
        command.args(["--mode", mode]);
    }
    if delete {
        command.arg("--delete");
    }
    run_json(&mut command, "baking geometry node")
}

fn inspect(project: &Path, frame: f64) -> Result<Value, Box<dyn Error>> {
    let output = run_json(
        pot()
            .arg("inspect")
            .arg(project)
            .args(["--id", "body", "--frame"])
            .arg(frame.to_string()),
        "inspecting evaluated node",
    )?;
    let items = output["result"]["items"]
        .as_array()
        .ok_or_else(|| std::io::Error::other("inspect returned no items array"))?;
    items
        .iter()
        .find(|item| item["id"] == "body")
        .cloned()
        .ok_or_else(|| std::io::Error::other("inspect omitted body").into())
}

fn mesh_positions(item: &Value) -> Result<Vec<[f64; 3]>, Box<dyn Error>> {
    let positions = item["evaluated_geometry"]["positions"]
        .as_array()
        .ok_or_else(|| std::io::Error::other("evaluated mesh positions are missing"))?;
    positions
        .iter()
        .map(|position| {
            let coordinates = position
                .as_array()
                .filter(|coordinates| coordinates.len() == 3)
                .ok_or_else(|| std::io::Error::other("mesh position is not a three-vector"))?;
            Ok([
                coordinates[0]
                    .as_f64()
                    .ok_or_else(|| std::io::Error::other("mesh x position is not numeric"))?,
                coordinates[1]
                    .as_f64()
                    .ok_or_else(|| std::io::Error::other("mesh y position is not numeric"))?,
                coordinates[2]
                    .as_f64()
                    .ok_or_else(|| std::io::Error::other("mesh z position is not numeric"))?,
            ])
        })
        .collect()
}

fn dimensions(item: &Value) -> Result<[f64; 3], Box<dyn Error>> {
    let positions = mesh_positions(item)?;
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for position in positions {
        for axis in 0..3 {
            min[axis] = min[axis].min(position[axis]);
            max[axis] = max[axis].max(position[axis]);
        }
    }
    Ok([max[0] - min[0], max[1] - min[1], max[2] - min[2]])
}

fn minimum_z(item: &Value) -> Result<f64, Box<dyn Error>> {
    mesh_positions(item)?
        .into_iter()
        .map(|position| position[2])
        .reduce(f64::min)
        .ok_or_else(|| std::io::Error::other("evaluated mesh has no vertices").into())
}

fn minimum_x(item: &Value) -> Result<f64, Box<dyn Error>> {
    mesh_positions(item)?
        .into_iter()
        .map(|position| position[0])
        .reduce(f64::min)
        .ok_or_else(|| std::io::Error::other("evaluated mesh has no vertices").into())
}

fn assert_dimensions(item: &Value, expected: [f64; 3]) -> Result<(), Box<dyn Error>> {
    let actual = dimensions(item)?;
    for axis in 0..3 {
        assert!(
            (actual[axis] - expected[axis]).abs() < 1.0e-9,
            "mesh dimensions differ on axis {axis}: expected {expected:?}, got {actual:?}"
        );
    }
    Ok(())
}

fn modifier_bakes(item: &Value) -> Result<&Value, Box<dyn Error>> {
    let modifiers = item["modifiers"]
        .as_array()
        .ok_or_else(|| std::io::Error::other("inspected node has no modifiers"))?;
    let modifier = modifiers
        .iter()
        .find(|modifier| modifier["id"] == "geometry_modifier")
        .ok_or_else(|| std::io::Error::other("geometry modifier is missing"))?;
    Ok(&modifier["params"]["bakes"])
}

fn assert_bake_record(item: &Value) -> Result<(), Box<dyn Error>> {
    let bakes = modifier_bakes(item)?;
    let present = match bakes {
        Value::Array(records) => !records.is_empty(),
        Value::Object(records) => !records.is_empty(),
        _ => false,
    };
    assert!(
        present,
        "node bake record was not persisted on the modifier"
    );
    Ok(())
}

fn bake_graph_operations(target: &str, bake_directory: &str, animation: bool) -> Vec<Value> {
    let mut operations = vec![
        json!({"op":"material.create","id":"surface","base_color":[0.2,0.3,0.4,1.0]}),
        json!({"op":"node.create","id":"body","kind":"box","params":{"size":1.0},"material":"surface"}),
        json!({"op":"graph.create","id":"geo","kind":"geometry"}),
    ];
    if animation {
        operations.extend([
            json!({"op":"graph.node_add","graph":"geo","id":"cube","type":"GeometryNodeMeshCube","properties":{"size":[1.0,1.0,1.0]}}),
            json!({"op":"graph.node_add","graph":"geo","id":"simulation_input","type":"GeometryNodeSimulationInput"}),
            json!({"op":"graph.node_add","graph":"geo","id":"translate","type":"GeometryNodeTransform","inputs":{"Translation":[0.0,0.0,1.0]}}),
            json!({"op":"graph.node_add","graph":"geo","id":"simulation_output","type":"GeometryNodeSimulationOutput"}),
        ]);
    } else {
        operations.push(json!({
            "op":"graph.node_add",
            "graph":"geo",
            "id":"cube",
            "type":"GeometryNodeMeshCube",
            "properties":{"size":[2.0,2.0,2.0]}
        }));
    }
    operations.extend([
        json!({
            "op":"graph.node_add",
            "graph":"geo",
            "id":"bake",
            "type":"GeometryNodeBake",
            "properties":{
                "bake_items":[{"id":"Item_0","name":"Geometry","socket_type":"geometry"}]
            }
        }),
        json!({"op":"graph.node_add","graph":"geo","id":"output","type":"NodeGroupOutput"}),
    ]);
    if animation {
        operations.extend([
            json!({"op":"graph.link","graph":"geo","from_node":"cube","from_socket":"Mesh","to_node":"simulation_input","to_socket":"Geometry"}),
            json!({"op":"graph.link","graph":"geo","from_node":"simulation_input","from_socket":"Geometry","to_node":"translate","to_socket":"Geometry"}),
            json!({"op":"graph.link","graph":"geo","from_node":"translate","from_socket":"Geometry","to_node":"simulation_output","to_socket":"Geometry"}),
            json!({"op":"graph.link","graph":"geo","from_node":"simulation_output","from_socket":"Geometry","to_node":"bake","to_socket":"Item_0"}),
        ]);
    } else {
        operations.push(json!({"op":"graph.link","graph":"geo","from_node":"cube","from_socket":"Mesh","to_node":"bake","to_socket":"Item_0"}));
    }
    operations.extend([
        json!({"op":"graph.link","graph":"geo","from_node":"bake","from_socket":"Item_0","to_node":"output","to_socket":"Geometry"}),
        json!({
            "op":"modifier.create",
            "target":{"id":"body"},
            "id":"geometry_modifier",
            "type":"nodes",
            "params":{"node_group":"geo","bake_target":target,"bake_directory":bake_directory}
        }),
    ]);
    operations
}

fn scalar_bake_graph_operations() -> Vec<Value> {
    vec![
        json!({"op":"material.create","id":"surface","base_color":[0.2,0.3,0.4,1.0]}),
        json!({"op":"node.create","id":"body","kind":"box","params":{"size":1.0},"material":"surface"}),
        json!({"op":"graph.create","id":"geo","kind":"geometry"}),
        json!({"op":"graph.interface_update","graph":"geo","set":{"inputs":[{"id":"shift","name":"Shift","socket_type":"float","default":0.5}],"outputs":[]}}),
        json!({"op":"graph.node_add","graph":"geo","id":"input","type":"NodeGroupInput"}),
        json!({"op":"graph.node_add","graph":"geo","id":"cube","type":"GeometryNodeMeshCube","properties":{"size":[2.0,2.0,2.0]}}),
        json!({"op":"graph.node_add","graph":"geo","id":"bake","type":"GeometryNodeBake","properties":{"bake_items":[{"id":"Item_0","name":"Geometry","socket_type":"geometry"},{"id":"Item_1","name":"Shift","socket_type":"float"}]}}),
        json!({"op":"graph.node_add","graph":"geo","id":"combine","type":"ShaderNodeCombineXYZ"}),
        json!({"op":"graph.node_add","graph":"geo","id":"transform","type":"GeometryNodeTransform"}),
        json!({"op":"graph.node_add","graph":"geo","id":"output","type":"NodeGroupOutput"}),
        json!({"op":"graph.link","graph":"geo","from_node":"cube","from_socket":"Mesh","to_node":"bake","to_socket":"Item_0"}),
        json!({"op":"graph.link","graph":"geo","from_node":"input","from_socket":"Shift","to_node":"bake","to_socket":"Item_1"}),
        json!({"op":"graph.link","graph":"geo","from_node":"bake","from_socket":"Item_0","to_node":"transform","to_socket":"Geometry"}),
        json!({"op":"graph.link","graph":"geo","from_node":"bake","from_socket":"Item_1","to_node":"combine","to_socket":"X"}),
        json!({"op":"graph.link","graph":"geo","from_node":"combine","from_socket":"Vector","to_node":"transform","to_socket":"Translation"}),
        json!({"op":"graph.link","graph":"geo","from_node":"transform","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
        json!({"op":"modifier.create","target":{"id":"body"},"id":"geometry_modifier","type":"nodes","params":{"node_group":"geo","bake_target":"PACKED","bake_directory":"","inputs":{"shift":0.5}}}),
    ]
}
fn create_project(
    project: &Path,
    batch_file: &Path,
    target: &str,
    bake_directory: &str,
    animation: bool,
) -> Result<(), Box<dyn Error>> {
    run_json(pot().arg("init").arg(project), "initializing project")?;
    apply(
        project,
        batch_file,
        json!(bake_graph_operations(target, bake_directory, animation)),
    )
}

fn set_cube_size(project: &Path, batch_file: &Path, size: f64) -> Result<(), Box<dyn Error>> {
    apply(
        project,
        batch_file,
        json!([{
            "op":"graph.node_update",
            "graph":"geo",
            "node_id":"cube",
            "set":{"properties":{"size":[size,size,size]}}
        }]),
    )
}

fn set_translation(project: &Path, batch_file: &Path, z: f64) -> Result<(), Box<dyn Error>> {
    apply(
        project,
        batch_file,
        json!([{
            "op":"graph.node_update",
            "graph":"geo",
            "node_id":"translate",
            "set":{"inputs":{"Translation":[0.0,0.0,z]}}
        }]),
    )
}
fn set_modifier_shift(project: &Path, batch_file: &Path, shift: f64) -> Result<(), Box<dyn Error>> {
    let mut params = document(project)?["nodes"]["body"]["modifiers"][0]["params"].clone();
    params["inputs"]["shift"] = json!(shift);
    apply(
        project,
        batch_file,
        json!([{
            "op":"modifier.update",
            "target":{"id":"body"},
            "modifier_id":"geometry_modifier",
            "set":{"params":params}
        }]),
    )
}

fn set_disk_directory(
    project: &Path,
    batch_file: &Path,
    bake_directory: &str,
) -> Result<(), Box<dyn Error>> {
    let mut params = document(project)?["nodes"]["body"]["modifiers"][0]["params"].clone();
    params["bake_directory"] = json!(bake_directory);
    apply(
        project,
        batch_file,
        json!([{
            "op":"modifier.update",
            "target":{"id":"body"},
            "modifier_id":"geometry_modifier",
            "set":{"params":params}
        }]),
    )
}

fn files_below(directory: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    if !directory.exists() {
        return Ok(Vec::new());
    }
    let mut files = Vec::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            files.extend(files_below(&path)?);
        } else if entry.file_type()?.is_file() {
            files.push(path);
        }
    }
    Ok(files)
}

fn assert_project_disk_payload(project: &Path, directory: &Path) -> Result<(), Box<dyn Error>> {
    let project_root = fs::canonicalize(project)?;
    let files = files_below(directory)?;
    assert!(
        !files.is_empty(),
        "DISK bake wrote no payload under {directory:?}"
    );
    assert!(
        files.iter().any(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .and_then(|name| name.strip_prefix("sha256-"))
                .and_then(|name| name.strip_suffix(".json"))
                .is_some_and(|digest| {
                    digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
                })
        }),
        "DISK bake payload was not addressed by a SHA-256 digest"
    );
    for path in files {
        assert!(
            fs::canonicalize(path)?.starts_with(&project_root),
            "DISK bake payload escaped its project root"
        );
    }
    Ok(())
}

#[test]
fn packed_node_bake_survives_live_input_changes_reopen_and_delete() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let project = directory.path().join("packed-project");
    let batch_file = directory.path().join("operations.json");
    create_project(&project, &batch_file, "PACKED", "", false)?;
    let unbaked = inspect(&project, 1.0)?;
    assert_dimensions(&unbaked, [2.0, 2.0, 2.0])?;

    let baked = bake(&project, None, None, false)?;
    assert_eq!(baked["result"]["scene_modified"], true);
    assert_eq!(baked["result"]["frame_count"], 1);
    let initial = inspect(&project, 1.0)?;
    assert_dimensions(&initial, [2.0, 2.0, 2.0])?;
    assert_eq!(initial["evaluated_geometry"]["vertex_count"], 8);
    assert_eq!(initial["evaluated_geometry"]["face_count"], 6);
    assert_bake_record(&initial)?;
    assert!(!project.join(".potter/bakes").exists());

    set_cube_size(&project, &batch_file, 4.0)?;
    let reopened = inspect(&project, 1.0)?;
    assert_dimensions(&reopened, [2.0, 2.0, 2.0])?;
    assert_bake_record(&reopened)?;
    assert_eq!(reopened["materials"], json!(["surface"]));
    assert_eq!(
        reopened["evaluated_material_refs"][0]["materials"],
        json!(["surface"])
    );

    let scene = document(&project)?;
    let resource_id =
        scene["nodes"]["body"]["modifiers"][0]["params"]["bakes"]["bake"]["resource_id"]
            .as_str()
            .ok_or_else(|| std::io::Error::other("packed bake resource ID is missing"))?;
    let bake_resource = &scene["resources"][resource_id];
    assert_eq!(bake_resource["storage"], "inline");
    assert_eq!(bake_resource["kind"], "node_bake");
    assert_eq!(
        bake_resource["frames"][0]["values"]["Item_0"]["material_refs"],
        json!(["surface"])
    );
    assert_eq!(
        scene["materials"]["surface"]["base_color"],
        json!([0.2, 0.3, 0.4, 1.0])
    );
    let body = &scene["nodes"]["body"];
    assert_eq!(body["materials"], json!(["surface"]));
    let data_id = body["data"]
        .as_str()
        .ok_or_else(|| std::io::Error::other("body has no mesh data block"))?;
    let data_block = &scene["data_blocks"][data_id];
    assert_eq!(data_block["type"], "mesh");
    assert_eq!(data_block["mesh"]["faces"][0]["material_index"], 0);

    let deleted = bake(&project, None, None, true)?;
    assert_eq!(deleted["result"]["scene_modified"], true);
    let live = inspect(&project, 1.0)?;
    assert_dimensions(&live, [4.0, 4.0, 4.0])?;
    assert_eq!(live["materials"], json!(["surface"]));
    Ok(())
}

#[test]
fn float_bake_value_keeps_transform_after_modifier_input_changes() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let project = directory.path().join("float-project");
    let batch_file = directory.path().join("operations.json");
    run_json(pot().arg("init").arg(&project), "initializing project")?;
    apply(&project, &batch_file, json!(scalar_bake_graph_operations()))?;

    let unbaked = inspect(&project, 1.0)?;
    assert!((minimum_x(&unbaked)? + 0.5).abs() < 1.0e-9);
    let baked = bake(&project, None, None, false)?;
    assert_eq!(baked["result"]["scene_modified"], true);
    let scene = document(&project)?;
    let resource_id =
        scene["nodes"]["body"]["modifiers"][0]["params"]["bakes"]["bake"]["resource_id"]
            .as_str()
            .ok_or_else(|| std::io::Error::other("packed bake resource ID is missing"))?;
    assert_eq!(
        scene["resources"][resource_id]["frames"][0]["values"]["Item_1"],
        json!({"type":"scalar","value":0.5})
    );

    set_modifier_shift(&project, &batch_file, 9.0)?;
    let still_baked = inspect(&project, 1.0)?;
    assert!((minimum_x(&still_baked)? + 0.5).abs() < 1.0e-9);
    assert_eq!(
        still_baked["evaluated_material_refs"][0]["materials"],
        json!(["surface"])
    );

    bake(&project, None, None, true)?;
    let live = inspect(&project, 1.0)?;
    assert!((minimum_x(&live)? - 8.0).abs() < 1.0e-9);
    Ok(())
}
#[test]
fn disk_node_bake_is_project_rooted_and_missing_paths_or_payloads_use_live_geometry()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let project = directory.path().join("disk-project");
    let batch_file = directory.path().join("operations.json");
    let cache_directory = project.join("cache-a");
    let cache_b_directory = project.join("cache-b");
    create_project(&project, &batch_file, "DISK", "//cache-a", false)?;
    let unbaked = inspect(&project, 1.0)?;
    assert_dimensions(&unbaked, [2.0, 2.0, 2.0])?;
    assert!(files_below(&cache_directory)?.is_empty());

    let baked = bake(&project, None, None, false)?;
    assert_eq!(baked["result"]["scene_modified"], true);
    let initial = inspect(&project, 1.0)?;
    assert_dimensions(&initial, [2.0, 2.0, 2.0])?;
    assert_bake_record(&initial)?;
    assert_project_disk_payload(&project, &cache_directory)?;

    set_cube_size(&project, &batch_file, 4.0)?;
    let reopened = inspect(&project, 1.0)?;
    assert_dimensions(&reopened, [2.0, 2.0, 2.0])?;

    let deleted = bake(&project, None, None, true)?;
    assert_eq!(deleted["result"]["scene_modified"], true);
    assert!(files_below(&cache_directory)?.is_empty());
    let live_after_delete = inspect(&project, 1.0)?;
    assert_dimensions(&live_after_delete, [4.0, 4.0, 4.0])?;

    let rebaked = bake(&project, None, None, false)?;
    assert_eq!(rebaked["result"]["scene_modified"], true);
    assert_project_disk_payload(&project, &cache_directory)?;
    set_cube_size(&project, &batch_file, 6.0)?;
    let still_baked = inspect(&project, 1.0)?;
    assert_dimensions(&still_baked, [4.0, 4.0, 4.0])?;

    set_disk_directory(&project, &batch_file, "//cache-b")?;
    assert!(!files_below(&cache_directory)?.is_empty());
    let changed_path = inspect(&project, 1.0)?;
    assert_dimensions(&changed_path, [6.0, 6.0, 6.0])?;

    let rebaked_at_changed_path = bake(&project, None, None, false)?;
    assert_eq!(rebaked_at_changed_path["result"]["scene_modified"], true);
    assert_project_disk_payload(&project, &cache_b_directory)?;
    set_cube_size(&project, &batch_file, 8.0)?;
    let still_baked_at_changed_path = inspect(&project, 1.0)?;
    assert_dimensions(&still_baked_at_changed_path, [6.0, 6.0, 6.0])?;
    for payload in files_below(&cache_b_directory)? {
        fs::remove_file(payload)?;
    }
    assert!(files_below(&cache_b_directory)?.is_empty());
    let missing_payload = inspect(&project, 1.0)?;
    assert_dimensions(&missing_payload, [8.0, 8.0, 8.0])?;
    Ok(())
}

#[test]
fn animated_node_bake_clamps_and_interpolates_frames_after_reopen() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let project = directory.path().join("animation-project");
    let batch_file = directory.path().join("operations.json");
    create_project(&project, &batch_file, "PACKED", "", true)?;

    let baked = bake(&project, Some("1:5"), Some("animation"), false)?;
    assert_eq!(baked["result"]["scene_modified"], true);
    assert_eq!(baked["result"]["frame_count"], 5);

    set_translation(&project, &batch_file, 9.0)?;
    let first = inspect(&project, 0.0)?;
    let between = inspect(&project, 2.5)?;
    let last = inspect(&project, 6.0)?;
    assert!((minimum_z(&first)? - 0.5).abs() < 1.0e-9);
    assert!((minimum_z(&between)? - 2.0).abs() < 1.0e-9);
    assert!((minimum_z(&last)? - 4.5).abs() < 1.0e-9);
    assert_bake_record(&between)?;
    Ok(())
}
