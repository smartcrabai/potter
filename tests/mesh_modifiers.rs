use std::{error::Error, fs, process::Command};

use serde_json::{Value, json};
use tempfile::tempdir;

#[test]
fn apply_array_modifier_changes_inspected_geometry_counts_and_bounds() -> Result<(), Box<dyn Error>>
{
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let batch = directory.path().join("operations.json");
    let init = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("init")
        .arg(&scene)
        .arg("--json")
        .output()?;
    assert!(init.status.success());

    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [
                {"op":"node.create", "id":"body", "kind":"box", "params":{"size":2}},
                {
                    "op":"modifier.create",
                    "target":{"id":"body"},
                    "id":"array_one",
                    "type":"array",
                    "params":{"count":2,"use_constant_offset":true,"relative_offset_displace":[0.0,0.0,0.0],"constant_offset_displace":[2.0,0.0,0.0]}
                }
            ]
        }))?,
    )?;
    let apply = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("apply")
        .arg(&scene)
        .arg("--file")
        .arg(&batch)
        .arg("--json")
        .output()?;
    assert!(
        apply.status.success(),
        "{}",
        String::from_utf8_lossy(&apply.stdout)
    );

    let inspect = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("inspect")
        .arg(&scene)
        .arg("--id")
        .arg("body")
        .arg("--json")
        .output()?;
    assert!(
        inspect.status.success(),
        "{}",
        String::from_utf8_lossy(&inspect.stdout)
    );
    let envelope: Value = serde_json::from_slice(&inspect.stdout)?;
    let item = &envelope["result"]["items"][0];
    assert_eq!(item["bounds"]["min"], json!([-1.0, -1.0, -1.0]));
    assert_eq!(item["bounds"]["max"], json!([3.0, 1.0, 1.0]));
    assert_eq!(item["evaluated_geometry"]["vertex_count"], 16);
    assert_eq!(item["evaluated_geometry"]["edge_count"], 24);
    assert_eq!(item["evaluated_geometry"]["face_count"], 12);
    assert_eq!(item["evaluated_geometry"]["triangle_count"], 24);
    Ok(())
}
#[test]
fn mesh_transform_selection_changes_bounds_and_drops_primitive_descriptor()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let batch = directory.path().join("operations.json");
    let init = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("init")
        .arg(&scene)
        .arg("--json")
        .output()?;
    assert!(init.status.success());

    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [
                {"op":"node.create", "id":"body", "kind":"box", "params":{"size":2}},
                {
                    "op":"mesh.transform_elements",
                    "target":{
                        "id":"body",
                        "elements":{"domain":"vertex","ids":["v0","v1","v2","v3","v4","v5","v6","v7"]}
                    },
                    "translation":[2.0,0.0,0.0]
                }
            ]
        }))?,
    )?;
    let apply = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("apply")
        .arg(&scene)
        .arg("--file")
        .arg(&batch)
        .arg("--json")
        .output()?;
    assert!(
        apply.status.success(),
        "{}",
        String::from_utf8_lossy(&apply.stdout)
    );
    let applied: Value = serde_json::from_slice(&apply.stdout)?;
    assert_eq!(
        applied["result"]["id_mappings"]["mesh_descriptor_dropped"]["body_mesh"],
        true
    );

    let inspect = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("inspect")
        .arg(&scene)
        .arg("--id")
        .arg("body")
        .arg("--json")
        .output()?;
    assert!(inspect.status.success());
    let envelope: Value = serde_json::from_slice(&inspect.stdout)?;
    let item = &envelope["result"]["items"][0];
    assert_eq!(item["bounds"]["min"], json!([1.0, -1.0, -1.0]));
    assert_eq!(item["bounds"]["max"], json!([3.0, 1.0, 1.0]));
    assert_eq!(item["params"], Value::Null);
    assert_eq!(item["primitive"], Value::Null);
    Ok(())
}
#[test]
fn shared_mesh_edits_require_scope_and_single_user_scope_isolated() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let batch = directory.path().join("operations.json");
    let init = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("init")
        .arg(&scene)
        .arg("--json")
        .output()?;
    assert!(init.status.success());
    let prefix = json!([
        {"op":"node.create", "id":"body", "kind":"box", "params":{"size":2}},
        {"op":"node.duplicate", "target":{"id":"body"}, "id":"body_copy", "mode":"linked"}
    ]);
    let vertices = json!({"domain":"vertex","ids":["v0","v1","v2","v3","v4","v5","v6","v7"]});
    let invalid_batch = json!({
        "schema_version":1,
        "base_revision":0,
        "operations":[
            prefix[0].clone(),
            prefix[1].clone(),
            {"op":"mesh.transform_elements","target":{"id":"body_copy","elements":vertices.clone()},"translation":[1.0,0.0,0.0]}
        ]
    });
    fs::write(&batch, serde_json::to_vec(&invalid_batch)?)?;
    let rejected = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("apply")
        .arg(&scene)
        .arg("--file")
        .arg(&batch)
        .arg("--json")
        .output()?;
    assert!(!rejected.status.success());
    let error: Value = serde_json::from_slice(&rejected.stdout)?;
    assert_eq!(error["error"]["code"], "SHARED_DATA_REQUIRES_SCOPE");

    let valid_batch = json!({
        "schema_version":1,
        "base_revision":0,
        "operations":[
            prefix[0].clone(),
            prefix[1].clone(),
            {"op":"mesh.transform_elements","target":{"id":"body_copy","elements":vertices},"translation":[1.0,0.0,0.0],"scope":"single_user"}
        ]
    });
    fs::write(&batch, serde_json::to_vec(&valid_batch)?)?;
    let applied = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("apply")
        .arg(&scene)
        .arg("--file")
        .arg(&batch)
        .arg("--json")
        .output()?;
    assert!(
        applied.status.success(),
        "{}",
        String::from_utf8_lossy(&applied.stdout)
    );
    let inspect = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("inspect")
        .arg(&scene)
        .arg("--id")
        .arg("body_copy")
        .arg("--json")
        .output()?;
    assert!(inspect.status.success());
    let envelope: Value = serde_json::from_slice(&inspect.stdout)?;
    let item = &envelope["result"]["items"][0];
    assert_eq!(item["bounds"]["min"], json!([0.0, -1.0, -1.0]));
    assert_eq!(item["bounds"]["max"], json!([2.0, 1.0, 1.0]));
    assert_eq!(item["sharing"]["count"], 1);
    Ok(())
}
#[test]
fn uv_unwrap_and_transform_store_per_corner_coordinates() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let batch = directory.path().join("operations.json");
    let init = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("init")
        .arg(&scene)
        .arg("--json")
        .output()?;
    assert!(init.status.success());
    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version":1,
            "base_revision":0,
            "operations":[
                {"op":"node.create","id":"body","kind":"box","params":{"size":2}},
                {"op":"uv.unwrap","target":{"id":"body"},"method":"smart"},
                {"op":"uv.transform","target":{"id":"body"},"translation":[1.0,2.0],"scale":[2.0,3.0]}
            ]
        }))?,
    )?;
    let applied = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("apply")
        .arg(&scene)
        .arg("--file")
        .arg(&batch)
        .arg("--json")
        .output()?;
    assert!(
        applied.status.success(),
        "{}",
        String::from_utf8_lossy(&applied.stdout)
    );
    let persisted: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let entries = persisted["data_blocks"]["body_mesh"]["mesh"]["attributes"]["uv_map"]
        .as_array()
        .ok_or("missing UV map")?;
    assert_eq!(entries.len(), 6);
    for entry in entries {
        let corners = entry["uv"].as_array().ok_or("missing UV corners")?;
        assert_eq!(corners.len(), 4);
        for corner in corners {
            let pair = corner.as_array().ok_or("invalid UV corner")?;
            assert_eq!(pair.len(), 2);
            let u = pair[0].as_f64().ok_or("missing U")?;
            let v = pair[1].as_f64().ok_or("missing V")?;
            assert!((1.0..=3.0).contains(&u));
            assert!((2.0..=5.0).contains(&v));
        }
    }
    Ok(())
}
#[test]
fn modifier_lifecycle_updates_reorders_deletes_and_applies() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let batch = directory.path().join("operations.json");
    let init = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("init")
        .arg(&scene)
        .arg("--json")
        .output()?;
    assert!(init.status.success());
    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version":1,
            "base_revision":0,
            "operations":[
                {"op":"node.create","id":"body","kind":"box","params":{"size":2}},
                {
                    "op":"modifier.create","target":{"id":"body"},"id":"array_one",
                    "type":"array","params":{"count":2,"relative_offset_displace":[0.0,0.0,0.0],"constant_offset_displace":[2.0,0.0,0.0]}
                },
                {"op":"modifier.create","target":{"id":"body"},"id":"triangulate_one","type":"triangulate"},
                {"op":"modifier.update","target":{"id":"body"},"id":"array_one","set":{"params":{"count":3}}},
                {"op":"modifier.reorder","target":{"id":"body"},"id":"triangulate_one","to_index":0},
                {"op":"modifier.delete","target":{"id":"body"},"id":"array_one"},
                {"op":"modifier.apply","target":{"id":"body"},"id":"triangulate_one"}
            ]
        }))?,
    )?;
    let applied = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("apply")
        .arg(&scene)
        .arg("--file")
        .arg(&batch)
        .arg("--json")
        .output()?;
    assert!(
        applied.status.success(),
        "{}",
        String::from_utf8_lossy(&applied.stdout)
    );
    let applied: Value = serde_json::from_slice(&applied.stdout)?;
    assert_eq!(
        applied["result"]["id_mappings"]["mesh_descriptor_dropped"]["body_mesh"],
        true
    );
    let inspect = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("inspect")
        .arg(&scene)
        .arg("--id")
        .arg("body")
        .arg("--json")
        .output()?;
    assert!(inspect.status.success());
    let envelope: Value = serde_json::from_slice(&inspect.stdout)?;
    let item = &envelope["result"]["items"][0];
    assert_eq!(item["modifiers"], json!([]));
    assert_eq!(item["evaluated_geometry"]["vertex_count"], 8);
    assert_eq!(item["evaluated_geometry"]["edge_count"], 18);
    assert_eq!(item["evaluated_geometry"]["face_count"], 12);
    assert_eq!(item["evaluated_geometry"]["triangle_count"], 12);
    Ok(())
}
#[test]
fn armature_modifier_apply_uses_the_selected_evaluation_frame() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let batch = directory.path().join("operations.json");
    let init = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("init")
        .arg(&scene)
        .arg("--json")
        .output()?;
    assert!(init.status.success());

    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version":1,
            "base_revision":0,
            "evaluation":{"frame":2.0},
            "operations":[
                {"op":"node.create","id":"body","kind":"plane","params":{"size":2.0}},
                {"op":"node.create","id":"arm","kind":"armature"},
                {"op":"bone.create","target":{"id":"arm"},"id":"root","name":"Root","head":[0.0,0.0,0.0],"tail":[0.0,1.0,0.0]},
                {"op":"vertex_group.create","target":{"id":"body"},"id":"root","name":"Root"},
                {"op":"vertex_group.assign","target":{"id":"body"},"group_id":"root","weights":[{"vertex_id":0,"weight":1.0}]},
                {"op":"action.create","id":"pose_action","name":"Pose"},
                {"op":"node.update","target":{"id":"arm"},"set":{"action":"pose_action"}},
                {"op":"keyframe.insert","target":{"id":"arm"},"path":"pose.root.translation","index":0,"frame":1.0,"value":0.0,"interpolation":"linear"},
                {"op":"keyframe.insert","target":{"id":"arm"},"path":"pose.root.translation","index":0,"frame":2.0,"value":2.0,"interpolation":"linear"},
                {"op":"modifier.create","target":{"id":"body"},"id":"skin","type":"armature","params":{"object":"arm","use_vertex_groups":true}},
                {"op":"modifier.apply","target":{"id":"body"},"id":"skin"}
            ]
        }))?,
    )?;
    let applied = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("apply")
        .arg(&scene)
        .arg("--file")
        .arg(&batch)
        .arg("--json")
        .output()?;
    assert!(
        applied.status.success(),
        "{}",
        String::from_utf8_lossy(&applied.stdout)
    );

    let document: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let vertex = &document["data_blocks"]["body_mesh"]["mesh"]["vertices"][0]["co"];
    assert_eq!(vertex, &json!([1.0, -1.0, 0.0]));
    assert_eq!(document["nodes"]["body"]["modifiers"], json!([]));
    Ok(())
}
#[test]
fn lattice_modifier_apply_uses_its_lattice_geometry() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let batch = directory.path().join("operations.json");
    let init = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("init")
        .arg(&scene)
        .arg("--json")
        .output()?;
    assert!(init.status.success());

    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version":1,
            "base_revision":0,
            "operations":[
                {"op":"node.create","id":"body","kind":"plane","params":{"size":2.0}},
                {"op":"lattice.create","id":"cage"},
                {"op":"lattice.update","target":{"id":"cage"},"set":{"points":[
                    [0.0,-1.0,-1.0],[2.0,-1.0,-1.0],[0.0,1.0,-1.0],[2.0,1.0,-1.0],
                    [0.0,-1.0,1.0],[2.0,-1.0,1.0],[0.0,1.0,1.0],[2.0,1.0,1.0]
                ]}},
                {"op":"modifier.create","target":{"id":"body"},"id":"lattice_mod","type":"lattice","params":{"object":"cage"}},
                {"op":"modifier.apply","target":{"id":"body"},"id":"lattice_mod"}
            ]
        }))?,
    )?;
    let applied = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("apply")
        .arg(&scene)
        .arg("--file")
        .arg(&batch)
        .arg("--json")
        .output()?;
    assert!(
        applied.status.success(),
        "{}",
        String::from_utf8_lossy(&applied.stdout)
    );

    let document: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let vertex = &document["data_blocks"]["body_mesh"]["mesh"]["vertices"][0]["co"];
    assert_eq!(vertex, &json!([0.0, -1.0, 0.0]));
    assert_eq!(document["nodes"]["body"]["modifiers"], json!([]));
    Ok(())
}
#[test]
fn uv_pin_and_pack_preserve_pins_and_pack_face_islands() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let batch = directory.path().join("operations.json");
    let init = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("init")
        .arg(&scene)
        .arg("--json")
        .output()?;
    assert!(init.status.success());
    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version":1,
            "base_revision":0,
            "operations":[
                {"op":"node.create","id":"body","kind":"box","params":{"size":2}},
                {"op":"uv.unwrap","target":{"id":"body"},"method":"smart"},
                {
                    "op":"uv.pin","target":{"id":"body"},
                    "elements":{"domain":"face","ids":["f0"]},"pinned":true
                },
                {"op":"uv.pack","target":{"id":"body"},"margin":0.04}
            ]
        }))?,
    )?;
    let applied = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("apply")
        .arg(&scene)
        .arg("--file")
        .arg(&batch)
        .arg("--json")
        .output()?;
    assert!(
        applied.status.success(),
        "{}",
        String::from_utf8_lossy(&applied.stdout)
    );
    let persisted: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let entries = persisted["data_blocks"]["body_mesh"]["mesh"]["attributes"]["uv_map"]
        .as_array()
        .ok_or("missing UV map")?;
    assert_eq!(entries.len(), 6);
    let mut bounds = Vec::<[f64; 4]>::new();
    for entry in entries {
        let corners = entry["uv"].as_array().ok_or("missing UV corners")?;
        assert_eq!(corners.len(), 4);
        let face_id = entry["face_id"].as_u64().ok_or("missing face ID")?;
        if let Some(pins) = entry.get("pinned").and_then(Value::as_array) {
            assert_eq!(pins.len(), 4);
            assert_eq!(
                pins.iter()
                    .filter(|pin| pin.as_bool() == Some(true))
                    .count(),
                if face_id == 0 { 4 } else { 0 }
            );
        } else {
            assert_ne!(face_id, 0, "explicitly pinned face should retain flags");
        }
        let mut bounds_entry = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        for corner in corners {
            let pair = corner.as_array().ok_or("invalid UV corner")?;
            let u = pair[0].as_f64().ok_or("missing U")?;
            let v = pair[1].as_f64().ok_or("missing V")?;
            assert!((0.0..=1.0).contains(&u));
            assert!((0.0..=1.0).contains(&v));
            bounds_entry[0] = bounds_entry[0].min(u);
            bounds_entry[1] = bounds_entry[1].min(v);
            bounds_entry[2] = bounds_entry[2].max(u);
            bounds_entry[3] = bounds_entry[3].max(v);
        }
        bounds.push(bounds_entry);
    }
    for first in 0..bounds.len() {
        for second in first + 1..bounds.len() {
            let [first_min_u, first_min_v, first_max_u, first_max_v] = bounds[first];
            let [second_min_u, second_min_v, second_max_u, second_max_v] = bounds[second];
            assert!(
                first_max_u <= second_min_u
                    || second_max_u <= first_min_u
                    || first_max_v <= second_min_v
                    || second_max_v <= first_min_v,
                "UV island bounding boxes overlap: {bounds:?}"
            );
        }
    }
    Ok(())
}

#[test]
fn voxel_remesh_produces_watertight_surface_near_source_bounds() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let batch = directory.path().join("operations.json");
    let init = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("init")
        .arg(&scene)
        .arg("--json")
        .output()?;
    assert!(init.status.success());
    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version":1,
            "base_revision":0,
            "operations":[
                {"op":"node.create","id":"body","kind":"box","params":{"size":2}},
                {"op":"mesh.remesh","target":{"id":"body"},"voxel_size":0.5}
            ]
        }))?,
    )?;
    let applied = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("apply")
        .arg(&scene)
        .arg("--file")
        .arg(&batch)
        .arg("--json")
        .output()?;
    assert!(
        applied.status.success(),
        "{}",
        String::from_utf8_lossy(&applied.stdout)
    );
    let persisted: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let mesh = &persisted["data_blocks"]["body_mesh"]["mesh"];
    let vertices = mesh["vertices"]
        .as_array()
        .ok_or("missing remesh vertices")?;
    let faces = mesh["faces"].as_array().ok_or("missing remesh faces")?;
    assert!(!vertices.is_empty());
    assert!(!faces.is_empty());
    let mut bounds = [[f64::INFINITY; 3], [f64::NEG_INFINITY; 3]];
    for vertex in vertices {
        let co = vertex["co"].as_array().ok_or("invalid remesh vertex")?;
        for axis in 0..3 {
            let value = co[axis].as_f64().ok_or("invalid coordinate")?;
            bounds[0][axis] = bounds[0][axis].min(value);
            bounds[1][axis] = bounds[1][axis].max(value);
        }
    }
    for (minimum, maximum) in bounds[0].iter().zip(&bounds[1]) {
        assert!((*minimum + 1.0).abs() <= 0.5);
        assert!((*maximum - 1.0).abs() <= 0.5);
    }
    let mut edge_incidence = std::collections::BTreeMap::<(u64, u64), usize>::new();
    for face in faces {
        let corners = face["v"].as_array().ok_or("invalid remesh face")?;
        for index in 0..corners.len() {
            let first = corners[index].as_u64().ok_or("invalid vertex reference")?;
            let second = corners[(index + 1) % corners.len()]
                .as_u64()
                .ok_or("invalid vertex reference")?;
            let edge = if first < second {
                (first, second)
            } else {
                (second, first)
            };
            *edge_incidence.entry(edge).or_default() += 1;
        }
    }
    assert!(edge_incidence.values().all(|incidence| *incidence == 2));
    Ok(())
}
