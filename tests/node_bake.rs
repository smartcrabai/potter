#![expect(
    clippy::unwrap_used,
    reason = "integration tests assert the expected error"
)]

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

fn apply(project: &Path, batch_file: &Path, operations: &Value) -> Result<(), Box<dyn Error>> {
    let revision = document(project)?["revision"]
        .as_u64()
        .ok_or_else(|| std::io::Error::other("scene revision is missing"))?;
    let document = json!({
        "schema_version": 1,
        "base_revision": revision,
        "operations": operations,
    });
    fs::write(batch_file, serde_json::to_vec(&document)?)?;
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

#[derive(Clone, Copy)]
enum BakeAction {
    Bake,
    Pack,
    Unpack,
    Delete,
}

fn bake(
    project: &Path,
    frames: Option<&str>,
    mode: Option<&str>,
    action: BakeAction,
) -> Result<Value, Box<dyn Error>> {
    bake_with_unpack_method(project, frames, mode, action, None)
}

fn bake_with_unpack_method(
    project: &Path,
    frames: Option<&str>,
    mode: Option<&str>,
    action: BakeAction,
    unpack_method: Option<&str>,
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
    match action {
        BakeAction::Bake => {}
        BakeAction::Pack => {
            command.arg("--pack");
        }
        BakeAction::Unpack => {
            command.arg("--unpack");
        }
        BakeAction::Delete => {
            command.arg("--delete");
        }
    }
    if let Some(method) = unpack_method {
        command.args(["--unpack-method", method]);
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
    position_values(&item["evaluated_geometry"]["positions"])
}

fn position_values(values: &Value) -> Result<Vec<[f64; 3]>, Box<dyn Error>> {
    let mut positions = values
        .as_array()
        .ok_or_else(|| std::io::Error::other("mesh positions are missing"))?
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
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    positions.sort_by(|a, b| {
        a[0].total_cmp(&b[0])
            .then(a[1].total_cmp(&b[1]))
            .then(a[2].total_cmp(&b[2]))
    });
    Ok(positions)
}

fn assert_positions_equal(actual: &[[f64; 3]], expected: &[[f64; 3]]) {
    assert_eq!(actual.len(), expected.len(), "vertex counts differ");
    for (actual, expected) in actual.iter().zip(expected) {
        for axis in 0..3 {
            assert!(
                (actual[axis] - expected[axis]).abs() < 1.0e-9,
                "position differs on axis {axis}: expected {expected:?}, got {actual:?}"
            );
        }
    }
}
fn assert_png_varies(path: &Path) -> Result<(), Box<dyn Error>> {
    let pixels = image::open(path)?.to_rgba8();
    let mut pixels = pixels.pixels();
    let first = pixels
        .next()
        .copied()
        .ok_or_else(|| std::io::Error::other("PNG contains no pixels"))?;
    assert!(
        pixels.any(|pixel| *pixel != first),
        "PNG contains no visible pixel variation"
    );
    Ok(())
}

fn blender_still_bake_positions() -> Result<Vec<[f64; 3]>, Box<dyn Error>> {
    let oracle: Value = serde_json::from_str(include_str!("fixtures/node_bake/expected.json"))?;
    let scenarios = oracle["scenarios"]
        .as_array()
        .ok_or_else(|| std::io::Error::other("Blender oracle has no scenarios"))?;
    let scenario = scenarios
        .iter()
        .find(|scenario| scenario["scenario"] == "S3_still")
        .ok_or_else(|| std::io::Error::other("Blender still-bake scenario is missing"))?;
    let sample = scenario["samples"]
        .as_array()
        .and_then(|samples| {
            samples.iter().find(|sample| {
                sample["phase"] == "after_still_bake_at_3"
                    && sample["frame"] == 3
                    && sample["subframe"] == 0.0
            })
        })
        .ok_or_else(|| std::io::Error::other("Blender still-bake sample is missing"))?;
    position_values(&sample["positions"])
}

/// A baked Field item stores its values as a hidden attribute on the baked Geometry item
/// (`.bake_<node>_<item>`), so the domain and per-element data come from that attribute.
fn baked_field_attribute(
    doc: &Value,
    resource_id: &str,
    item: &str,
) -> Result<(String, Vec<Value>), Box<dyn Error>> {
    let frame = &doc["resources"][resource_id]["frames"][0]["values"];
    let field = &frame[item];
    if field["type"] != "field" {
        return Err(std::io::Error::other("baked item is not a field").into());
    }
    let attribute_name = field["attribute_name"]
        .as_str()
        .ok_or_else(|| std::io::Error::other("baked field attribute name is missing"))?;
    let attribute = &frame["Item_0"]["mesh"]["attributes"][attribute_name];
    let domain = attribute["domain"]
        .as_str()
        .ok_or_else(|| std::io::Error::other("baked field domain is missing"))?
        .to_owned();
    let values = attribute["data"]
        .as_array()
        .ok_or_else(|| std::io::Error::other("baked field data is missing"))?
        .clone();
    Ok((domain, values))
}

fn blender_point_field_positions(phase: &str) -> Result<Vec<[f64; 3]>, Box<dyn Error>> {
    let oracle: Value = serde_json::from_str(include_str!("fixtures/node_bake/expected.json"))?;
    let scenario = oracle["scenarios"]
        .as_array()
        .and_then(|scenarios| {
            scenarios
                .iter()
                .find(|scenario| scenario["scenario"] == "S12_point_field_item")
        })
        .ok_or_else(|| std::io::Error::other("Blender point-field scenario is missing"))?;
    let sample = scenario["samples"]
        .as_array()
        .and_then(|samples| {
            samples.iter().find(|sample| {
                sample["phase"] == phase && sample["frame"] == 1 && sample["subframe"] == 0.0
            })
        })
        .ok_or_else(|| std::io::Error::other("Blender point-field sample is missing"))?;
    position_values(&sample["positions"])
}

fn blender_attribute_field_positions(phase: &str) -> Result<Vec<[f64; 3]>, Box<dyn Error>> {
    let oracle: Value = serde_json::from_str(include_str!("fixtures/node_bake/expected.json"))?;
    let scenario = oracle["scenarios"]
        .as_array()
        .and_then(|scenarios| {
            scenarios
                .iter()
                .find(|scenario| scenario["scenario"] == "S14_point_attribute_item")
        })
        .ok_or_else(|| std::io::Error::other("Blender attribute-field scenario is missing"))?;
    let sample = scenario["samples"]
        .as_array()
        .and_then(|samples| {
            samples.iter().find(|sample| {
                sample["phase"] == phase && sample["frame"] == 1 && sample["subframe"] == 0.0
            })
        })
        .ok_or_else(|| std::io::Error::other("Blender attribute-field sample is missing"))?;
    position_values(&sample["positions"])
}

fn blender_attribute_field_values() -> Result<Vec<f64>, Box<dyn Error>> {
    let oracle: Value = serde_json::from_str(include_str!("fixtures/node_bake/expected.json"))?;
    let scenario = oracle["scenarios"]
        .as_array()
        .and_then(|scenarios| {
            scenarios
                .iter()
                .find(|scenario| scenario["scenario"] == "S14_point_attribute_item")
        })
        .ok_or_else(|| std::io::Error::other("Blender attribute-field scenario is missing"))?;
    let values = scenario["observations"]["attribute_after_bake"]["values"]
        .as_array()
        .ok_or_else(|| std::io::Error::other("Blender baked attribute values are missing"))?;
    let mut values = values
        .iter()
        .map(|value| {
            value
                .as_f64()
                .ok_or_else(|| Box::<dyn Error>::from("Blender attribute value is not numeric"))
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    values.sort_by(f64::total_cmp);
    Ok(values)
}

fn blender_missing_frame_live_status() -> Result<Vec<(f64, bool)>, Box<dyn Error>> {
    let oracle: Value = serde_json::from_str(include_str!("fixtures/node_bake/expected.json"))?;
    let scenario = oracle["scenarios"]
        .as_array()
        .and_then(|scenarios| {
            scenarios
                .iter()
                .find(|scenario| scenario["scenario"] == "S11_disk_missing_frame")
        })
        .ok_or_else(|| std::io::Error::other("Blender missing-frame scenario is missing"))?;
    scenario["samples"]
        .as_array()
        .ok_or_else(|| std::io::Error::other("Blender missing-frame samples are missing"))?
        .iter()
        .filter(|sample| sample["subframe"].as_f64() == Some(0.0))
        .map(|sample| {
            let frame = sample["frame"]
                .as_f64()
                .ok_or_else(|| std::io::Error::other("Blender sample frame is missing"))?;
            let subframe = sample["subframe"]
                .as_f64()
                .ok_or_else(|| std::io::Error::other("Blender sample subframe is missing"))?;
            let min_x = sample["min_x"]
                .as_f64()
                .ok_or_else(|| std::io::Error::other("Blender sample min_x is missing"))?;
            Ok((frame + subframe, min_x > 50.0))
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

fn transformed_cube_bake_operations(target: &str, bake_directory: &str) -> Vec<Value> {
    vec![
        json!({"op":"material.create","id":"surface","base_color":[0.2,0.3,0.4,1.0]}),
        json!({"op":"node.create","id":"body","kind":"box","params":{"size":1.0},"material":"surface"}),
        json!({"op":"graph.create","id":"geo","kind":"geometry"}),
        json!({"op":"graph.node_add","graph":"geo","id":"cube","type":"GeometryNodeMeshCube","properties":{"size":[2.0,2.0,2.0]}}),
        json!({"op":"graph.node_add","graph":"geo","id":"transform","type":"GeometryNodeTransform","inputs":{"Translation":[1.5,0.0,0.0]}}),
        json!({"op":"graph.node_add","graph":"geo","id":"bake","type":"GeometryNodeBake","properties":{"bake_items":[{"id":"Item_0","name":"Geometry","socket_type":"geometry"}]}}),
        json!({"op":"graph.node_add","graph":"geo","id":"output","type":"NodeGroupOutput"}),
        json!({"op":"graph.link","graph":"geo","from_node":"cube","from_socket":"Mesh","to_node":"transform","to_socket":"Geometry"}),
        json!({"op":"graph.link","graph":"geo","from_node":"transform","from_socket":"Geometry","to_node":"bake","to_socket":"Item_0"}),
        json!({"op":"graph.link","graph":"geo","from_node":"bake","from_socket":"Item_0","to_node":"output","to_socket":"Geometry"}),
        json!({"op":"modifier.create","target":{"id":"body"},"id":"geometry_modifier","type":"nodes","params":{"node_group":"geo","bake_target":target,"bake_directory":bake_directory}}),
    ]
}

fn point_field_bake_operations(is_attribute: bool) -> Vec<Value> {
    vec![
        json!({"op":"material.create","id":"surface","base_color":[0.2,0.3,0.4,1.0]}),
        json!({"op":"node.create","id":"body","kind":"box","params":{"size":1.0},"material":"surface"}),
        json!({"op":"graph.create","id":"geo","kind":"geometry"}),
        json!({"op":"graph.interface_update","graph":"geo","set":{"inputs":[{"id":"shift","name":"Shift","socket_type":"float","default":0.5}],"outputs":[]}}),
        json!({"op":"graph.node_add","graph":"geo","id":"input","type":"NodeGroupInput"}),
        json!({"op":"graph.node_add","graph":"geo","id":"cube","type":"GeometryNodeMeshCube","properties":{"size":[2.0,2.0,2.0]}}),
        json!({"op":"graph.node_add","graph":"geo","id":"position","type":"GeometryNodeInputPosition"}),
        json!({"op":"graph.node_add","graph":"geo","id":"separate","type":"ShaderNodeSeparateXYZ"}),
        json!({"op":"graph.node_add","graph":"geo","id":"add","type":"ShaderNodeMath","properties":{"operation":"ADD"}}),
        json!({"op":"graph.node_add","graph":"geo","id":"bake","type":"GeometryNodeBake","properties":{"bake_items":[{"id":"Item_0","name":"Geometry","socket_type":"geometry"},{"id":"Item_1","name":"X","socket_type":"float","attribute_domain":"POINT","is_attribute":is_attribute}]}}),
        json!({"op":"graph.node_add","graph":"geo","id":"combine","type":"ShaderNodeCombineXYZ"}),
        json!({"op":"graph.node_add","graph":"geo","id":"set_position","type":"GeometryNodeSetPosition"}),
        json!({"op":"graph.node_add","graph":"geo","id":"output","type":"NodeGroupOutput"}),
        json!({"op":"graph.link","graph":"geo","from_node":"position","from_socket":"Position","to_node":"separate","to_socket":"Vector"}),
        json!({"op":"graph.link","graph":"geo","from_node":"separate","from_socket":"X","to_node":"add","to_socket":"Value"}),
        json!({"op":"graph.link","graph":"geo","from_node":"input","from_socket":"Shift","to_node":"add","to_socket":"Value_001"}),
        json!({"op":"graph.link","graph":"geo","from_node":"cube","from_socket":"Mesh","to_node":"bake","to_socket":"Item_0"}),
        json!({"op":"graph.link","graph":"geo","from_node":"add","from_socket":"Value","to_node":"bake","to_socket":"Item_1"}),
        json!({"op":"graph.link","graph":"geo","from_node":"bake","from_socket":"Item_0","to_node":"set_position","to_socket":"Geometry"}),
        json!({"op":"graph.link","graph":"geo","from_node":"bake","from_socket":"Item_1","to_node":"combine","to_socket":"X"}),
        json!({"op":"graph.link","graph":"geo","from_node":"combine","from_socket":"Vector","to_node":"set_position","to_socket":"Offset"}),
        json!({"op":"graph.link","graph":"geo","from_node":"set_position","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
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
        &json!(bake_graph_operations(target, bake_directory, animation)),
    )
}

fn set_cube_size(project: &Path, batch_file: &Path, size: f64) -> Result<(), Box<dyn Error>> {
    apply(
        project,
        batch_file,
        &json!([{
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
        &json!([{
            "op":"graph.node_update",
            "graph":"geo",
            "node_id":"translate",
            "set":{"inputs":{"Translation":[0.0,0.0,z]}}
        }]),
    )
}

fn set_transform_translation(
    project: &Path,
    batch_file: &Path,
    x: f64,
) -> Result<(), Box<dyn Error>> {
    apply(
        project,
        batch_file,
        &json!([{
            "op":"graph.node_update",
            "graph":"geo",
            "node_id":"transform",
            "set":{"inputs":{"Translation":[x,0.0,0.0]}}
        }]),
    )
}
fn set_modifier_shift(project: &Path, batch_file: &Path, shift: f64) -> Result<(), Box<dyn Error>> {
    let mut params = document(project)?["nodes"]["body"]["modifiers"][0]["params"].clone();
    params["inputs"]["shift"] = json!(shift);
    apply(
        project,
        batch_file,
        &json!([{
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
        &json!([{
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

/// Asserts that `directory` holds no files, listing whatever it does hold on failure.
fn assert_no_files(directory: &Path) -> Result<(), Box<dyn Error>> {
    let files = files_below(directory)?;
    assert_eq!(
        files,
        Vec::<PathBuf>::new(),
        "files left in {}",
        directory.display()
    );
    Ok(())
}

/// Asserts that `directory` holds at least one file.
fn assert_some_files(directory: &Path) -> Result<(), Box<dyn Error>> {
    assert_ne!(
        files_below(directory)?,
        Vec::<PathBuf>::new(),
        "no files in {}",
        directory.display()
    );
    Ok(())
}

fn assert_project_disk_payload(project: &Path, directory: &Path) -> Result<(), Box<dyn Error>> {
    let project_root = fs::canonicalize(project)?;
    let files = files_below(directory)?;
    assert!(
        !files.is_empty(),
        "DISK bake wrote no payload under {}",
        directory.display()
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

fn node_bake_payload_path(
    directory: &Path,
    manifest: &Value,
    frame: f64,
) -> Result<PathBuf, Box<dyn Error>> {
    let frames = manifest
        .get("frames")
        .and_then(Value::as_array)
        .ok_or_else(|| std::io::Error::other("disk bake manifest is missing"))?;
    let digest = frames
        .iter()
        .find(|entry| entry["frame"].as_f64() == Some(frame))
        .and_then(|entry| entry["digest"].as_str())
        .ok_or_else(|| std::io::Error::other(format!("frame {frame} digest is missing")))?;
    let hex = digest
        .strip_prefix("sha256:")
        .ok_or_else(|| std::io::Error::other("disk bake digest prefix is invalid"))?;
    Ok(directory.join(format!("sha256-{hex}.json")))
}

#[test]
fn packed_node_bake_survives_live_input_changes_reopen_and_delete() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let project = directory.path().join("packed-project");
    let batch_file = directory.path().join("operations.json");
    create_project(&project, &batch_file, "PACKED", "", false)?;
    let unbaked = inspect(&project, 1.0)?;
    assert_dimensions(&unbaked, [2.0, 2.0, 2.0])?;

    let baked = bake(&project, None, None, BakeAction::Bake)?;
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

    let deleted = bake(&project, None, None, BakeAction::Delete)?;
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
    apply(
        &project,
        &batch_file,
        &json!(scalar_bake_graph_operations()),
    )?;

    let unbaked = inspect(&project, 1.0)?;
    assert!((minimum_x(&unbaked)? + 0.5).abs() < 1.0e-9);
    let baked = bake(&project, None, None, BakeAction::Bake)?;
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

    bake(&project, None, None, BakeAction::Delete)?;
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
    assert_no_files(&cache_directory)?;

    let baked = bake(&project, None, None, BakeAction::Bake)?;
    assert_eq!(baked["result"]["scene_modified"], true);
    let initial = inspect(&project, 1.0)?;
    assert_dimensions(&initial, [2.0, 2.0, 2.0])?;
    assert_bake_record(&initial)?;
    assert_project_disk_payload(&project, &cache_directory)?;

    set_cube_size(&project, &batch_file, 4.0)?;
    let reopened = inspect(&project, 1.0)?;
    assert_dimensions(&reopened, [2.0, 2.0, 2.0])?;

    let deleted = bake(&project, None, None, BakeAction::Delete)?;
    assert_eq!(deleted["result"]["scene_modified"], true);
    assert_no_files(&cache_directory)?;
    let live_after_delete = inspect(&project, 1.0)?;
    assert_dimensions(&live_after_delete, [4.0, 4.0, 4.0])?;

    let rebaked = bake(&project, None, None, BakeAction::Bake)?;
    assert_eq!(rebaked["result"]["scene_modified"], true);
    assert_project_disk_payload(&project, &cache_directory)?;
    set_cube_size(&project, &batch_file, 6.0)?;
    let still_baked = inspect(&project, 1.0)?;
    assert_dimensions(&still_baked, [4.0, 4.0, 4.0])?;

    set_disk_directory(&project, &batch_file, "//cache-b")?;
    assert_some_files(&cache_directory)?;
    let changed_path = inspect(&project, 1.0)?;
    assert_dimensions(&changed_path, [6.0, 6.0, 6.0])?;

    let rebaked_at_changed_path = bake(&project, None, None, BakeAction::Bake)?;
    assert_eq!(rebaked_at_changed_path["result"]["scene_modified"], true);
    assert_project_disk_payload(&project, &cache_b_directory)?;
    set_cube_size(&project, &batch_file, 8.0)?;
    let still_baked_at_changed_path = inspect(&project, 1.0)?;
    assert_dimensions(&still_baked_at_changed_path, [6.0, 6.0, 6.0])?;
    for payload in files_below(&cache_b_directory)? {
        fs::remove_file(payload)?;
    }
    assert_no_files(&cache_b_directory)?;
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

    let baked = bake(&project, Some("1:5"), Some("animation"), BakeAction::Bake)?;
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

#[test]
fn packed_and_disk_node_bakes_match_blender_geometry_after_input_change_and_delete()
-> Result<(), Box<dyn Error>> {
    let expected = blender_still_bake_positions()?;
    let directory = tempdir()?;
    for target in ["PACKED", "DISK"] {
        let project = directory.path().join(format!("{target}-project"));
        let batch_file = directory.path().join(format!("{target}-operations.json"));
        let bake_directory = if target == "DISK" { "//cache-a" } else { "" };
        run_json(
            pot().arg("init").arg(&project),
            "initializing Blender-oracle project",
        )?;
        apply(
            &project,
            &batch_file,
            &json!(transformed_cube_bake_operations(target, bake_directory)),
        )?;

        let unbaked = inspect(&project, 1.0)?;
        assert_positions_equal(&mesh_positions(&unbaked)?, &expected);
        let captured = bake(&project, None, None, BakeAction::Bake)?;
        assert_eq!(captured["result"]["scene_modified"], true);
        if target == "DISK" {
            assert_project_disk_payload(&project, &project.join("cache-a"))?;
        }

        set_transform_translation(&project, &batch_file, 100.0)?;
        let reopened = inspect(&project, 1.0)?;
        assert_eq!(reopened["evaluated_geometry"]["vertex_count"], 8);
        assert_eq!(reopened["evaluated_geometry"]["face_count"], 6);
        assert_positions_equal(&mesh_positions(&reopened)?, &expected);

        if target == "DISK" {
            let packed = bake(&project, None, None, BakeAction::Pack)?;
            assert_eq!(packed["result"]["storage"], "packed");
            for payload in files_below(&project.join("cache-a"))? {
                fs::remove_file(payload)?;
            }
            let packed_geometry = inspect(&project, 1.0)?;
            assert_positions_equal(&mesh_positions(&packed_geometry)?, &expected);

            let unpacked = bake(&project, None, None, BakeAction::Unpack)?;
            assert_eq!(unpacked["result"]["storage"], "disk");
            assert_project_disk_payload(&project, &project.join(".potter/bakes"))?;
            let unpacked_geometry = inspect(&project, 1.0)?;
            assert_positions_equal(&mesh_positions(&unpacked_geometry)?, &expected);
        }
        bake(&project, None, None, BakeAction::Delete)?;
        let live = inspect(&project, 1.0)?;
        let expected_after_delete = expected
            .iter()
            .map(|position| [position[0] + 98.5, position[1], position[2]])
            .collect::<Vec<_>>();
        assert_positions_equal(&mesh_positions(&live)?, &expected_after_delete);
        if target == "DISK" {
            assert_no_files(&project.join("cache-a"))?;
            assert_no_files(&project.join(".potter/bakes"))?;
        }
    }
    Ok(())
}
#[test]
fn failed_multiframe_disk_rebake_rolls_back_new_payloads_and_scene() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let project = directory.path().join("atomic-project");
    let batch_file = directory.path().join("operations.json");
    create_project(&project, &batch_file, "DISK", "//cache-a", true)?;
    bake(&project, Some("1:3"), Some("animation"), BakeAction::Bake)?;

    let before = document(&project)?;
    let frames = before["nodes"]["body"]["modifiers"][0]["params"]["bakes"]["bake"]["frames"]
        .as_array()
        .ok_or_else(|| std::io::Error::other("disk bake manifest is missing"))?;
    let digest_for = |frame: f64| -> Result<String, Box<dyn Error>> {
        frames
            .iter()
            .find(|entry| entry["frame"] == frame)
            .and_then(|entry| entry["digest"].as_str())
            .map(str::to_owned)
            .ok_or_else(|| std::io::Error::other(format!("frame {frame} digest is missing")).into())
    };
    let payload_path = |digest: &str| -> Result<PathBuf, Box<dyn Error>> {
        let hex = digest
            .strip_prefix("sha256:")
            .ok_or_else(|| std::io::Error::other("disk bake digest prefix is invalid"))?;
        Ok(project
            .join("cache-a/body/geometry_modifier/bake")
            .join(format!("sha256-{hex}.json")))
    };
    let first_path = payload_path(&digest_for(1.0)?)?;
    let second_path = payload_path(&digest_for(2.0)?)?;
    let third_path = payload_path(&digest_for(3.0)?)?;
    fs::remove_file(&first_path)?;
    fs::write(&second_path, b"corrupt payload")?;

    let error = bake(&project, Some("1:3"), Some("animation"), BakeAction::Bake).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("existing node-bake file has invalid content")
    );
    let after = document(&project)?;
    assert_eq!(after["revision"], before["revision"]);
    assert_eq!(
        after["nodes"]["body"]["modifiers"][0]["params"]["bakes"]["bake"],
        before["nodes"]["body"]["modifiers"][0]["params"]["bakes"]["bake"]
    );
    assert!(!first_path.exists());
    assert_eq!(fs::read(&second_path)?, b"corrupt payload");
    assert!(third_path.exists());
    assert_eq!(files_below(&project.join("cache-a"))?.len(), 2);
    Ok(())
}

#[test]
fn disk_missing_frame_fallback_matches_blender_after_reopen() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let project = directory.path().join("missing-frame-project");
    let batch_file = directory.path().join("operations.json");
    create_project(&project, &batch_file, "DISK", "//cache-a", true)?;
    bake(&project, Some("1:5"), Some("animation"), BakeAction::Bake)?;
    set_translation(&project, &batch_file, 9.0)?;

    let doc = document(&project)?;
    let entries = doc["nodes"]["body"]["modifiers"][0]["params"]["bakes"]["bake"]["frames"]
        .as_array()
        .ok_or_else(|| std::io::Error::other("disk bake manifest is missing"))?;
    let digest = entries
        .iter()
        .find(|entry| entry["frame"] == 3.0)
        .and_then(|entry| entry["digest"].as_str())
        .ok_or_else(|| std::io::Error::other("frame-3 digest is missing"))?;
    let hex = digest
        .strip_prefix("sha256:")
        .ok_or_else(|| std::io::Error::other("frame-3 digest prefix is invalid"))?;
    fs::remove_file(
        project
            .join("cache-a/body/geometry_modifier/bake")
            .join(format!("sha256-{hex}.json")),
    )?;

    for (frame, expected_live) in blender_missing_frame_live_status()? {
        let sample = inspect(&project, frame)?;
        let actual_live = minimum_z(&sample)? > 5.0;
        assert_eq!(
            actual_live, expected_live,
            "frame {frame} fallback differs from the independent Blender cache oracle"
        );
    }
    Ok(())
}

#[test]
fn point_field_bake_matches_blender_and_survives_modifier_input_changes()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let project = directory.path().join("field-project");
    let batch_file = directory.path().join("operations.json");
    run_json(
        pot().arg("init").arg(&project),
        "initializing field-bake project",
    )?;
    apply(
        &project,
        &batch_file,
        &json!(point_field_bake_operations(false)),
    )?;

    let expected = blender_point_field_positions("after_bake")?;
    let unbaked = inspect(&project, 1.0)?;
    assert_positions_equal(&mesh_positions(&unbaked)?, &expected);
    bake(&project, None, None, BakeAction::Bake)?;

    let doc = document(&project)?;
    let resource_id =
        doc["nodes"]["body"]["modifiers"][0]["params"]["bakes"]["bake"]["resource_id"]
            .as_str()
            .ok_or_else(|| std::io::Error::other("point-field cache resource ID is missing"))?;
    let (domain, values) = baked_field_attribute(&doc, resource_id, "Item_1")?;
    assert_eq!(domain, "POINT");
    assert_eq!(values.len(), 8);

    set_modifier_shift(&project, &batch_file, 100.0)?;
    let reopened = inspect(&project, 1.0)?;
    assert_positions_equal(&mesh_positions(&reopened)?, &expected);

    bake(&project, None, None, BakeAction::Delete)?;
    let deleted = inspect(&project, 1.0)?;
    assert_positions_equal(
        &mesh_positions(&deleted)?,
        &blender_point_field_positions("after_delete")?,
    );
    Ok(())
}

#[test]
fn baked_attribute_field_matches_blender_and_survives_input_changes() -> Result<(), Box<dyn Error>>
{
    let directory = tempdir()?;
    let project = directory.path().join("attribute-field-project");
    let batch_file = directory.path().join("operations.json");
    run_json(
        pot().arg("init").arg(&project),
        "initializing attribute-field project",
    )?;
    apply(
        &project,
        &batch_file,
        &json!(point_field_bake_operations(true)),
    )?;

    let expected_positions = blender_attribute_field_positions("after_bake")?;
    let unbaked = inspect(&project, 1.0)?;
    assert_positions_equal(&mesh_positions(&unbaked)?, &expected_positions);
    bake(&project, None, None, BakeAction::Bake)?;

    let doc = document(&project)?;
    let resource_id =
        doc["nodes"]["body"]["modifiers"][0]["params"]["bakes"]["bake"]["resource_id"]
            .as_str()
            .ok_or_else(|| std::io::Error::other("attribute-field cache resource ID is missing"))?;
    let (domain, values) = baked_field_attribute(&doc, resource_id, "Item_1")?;
    assert_eq!(domain, "POINT");
    let mut actual_values = values
        .iter()
        .map(|value| {
            value
                .as_f64()
                .ok_or_else(|| Box::<dyn Error>::from("attribute-field cache value is not numeric"))
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    actual_values.sort_by(f64::total_cmp);
    let expected_values = blender_attribute_field_values()?;
    assert_eq!(actual_values.len(), expected_values.len());
    for (actual, expected) in actual_values.iter().zip(expected_values) {
        assert!((actual - expected).abs() < 1.0e-9);
    }

    set_modifier_shift(&project, &batch_file, 100.0)?;
    let cached = inspect(&project, 1.0)?;
    assert_positions_equal(&mesh_positions(&cached)?, &expected_positions);
    bake(&project, None, None, BakeAction::Delete)?;
    let live = inspect(&project, 1.0)?;
    assert_positions_equal(
        &mesh_positions(&live)?,
        &blender_attribute_field_positions("after_delete")?,
    );
    Ok(())
}

#[test]
fn imported_bake_node_settings_drive_disk_target_custom_path_and_frame_range()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let project = directory.path().join("node-settings-project");
    let batch_file = directory.path().join("operations.json");
    run_json(
        pot().arg("init").arg(&project),
        "initializing node-settings project",
    )?;
    let mut operations = transformed_cube_bake_operations("PACKED", "//modifier-cache");
    let modifier = operations
        .iter_mut()
        .find(|operation| operation["op"] == "modifier.create")
        .ok_or_else(|| std::io::Error::other("modifier operation is missing"))?;
    modifier["params"]["bakes"] = json!({"bake": {
        "bake_mode": "ANIMATION",
        "bake_target": "DISK",
        "use_custom_simulation_frame_range": true,
        "frame_start": 2,
        "frame_end": 4,
        "use_custom_path": true,
        "directory": "//node-cache"
    }});
    apply(&project, &batch_file, &json!(operations))?;

    let baked = bake(&project, None, None, BakeAction::Bake)?;
    assert_eq!(baked["result"]["storage"], "disk");
    let baked_doc = document(&project)?;
    let config = &baked_doc["nodes"]["body"]["modifiers"][0]["params"]["bakes"]["bake"];
    assert_eq!(config["storage"], "disk");
    assert_eq!(config["directory"], "//node-cache");
    assert_eq!(config["cache_directory"], "//node-cache");
    assert_eq!(config["use_custom_simulation_frame_range"], true);
    assert_eq!(config["frames"].as_array().map(Vec::len), Some(3));
    assert_eq!(config["frames"][0]["frame"], 2.0);
    assert_eq!(config["frames"][1]["frame"], 3.0);
    assert_eq!(config["frames"][2]["frame"], 4.0);
    assert_project_disk_payload(&project, &project.join("node-cache"))?;
    assert_no_files(&project.join("modifier-cache"))?;

    set_transform_translation(&project, &batch_file, 100.0)?;
    assert_positions_equal(
        &mesh_positions(&inspect(&project, 3.0)?)?,
        &blender_still_bake_positions()?,
    );
    bake(&project, None, None, BakeAction::Delete)?;
    assert_no_files(&project.join("node-cache"))?;
    Ok(())
}

#[test]
fn disk_node_bake_pack_unpack_and_delete_preserve_baked_geometry() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let project = directory.path().join("pack-unpack-project");
    let batch_file = directory.path().join("operations.json");
    let cache_directory = project.join("cache-a");
    let local_cache_directory = project.join(".potter/bakes");
    create_project(&project, &batch_file, "DISK", "//cache-a", false)?;

    let baked = bake(&project, None, None, BakeAction::Bake)?;
    assert_eq!(baked["result"]["scene_modified"], true);
    assert_project_disk_payload(&project, &cache_directory)?;
    set_cube_size(&project, &batch_file, 4.0)?;
    let disk_files = files_below(&cache_directory)?;
    assert_some_files(&cache_directory)?;
    assert_dimensions(&inspect(&project, 1.0)?, [2.0, 2.0, 2.0])?;

    let packed = bake(&project, None, None, BakeAction::Pack)?;
    assert_eq!(packed["result"]["scene_modified"], true);
    let packed_scene = document(&project)?;
    let packed_params = &packed_scene["nodes"]["body"]["modifiers"][0]["params"];
    assert_eq!(packed_params["bake_target"], "DISK");
    assert_eq!(packed_params["bakes"]["bake"]["bake_target"], "PACKED");
    let packed_resource_id = packed_params["bakes"]["bake"]["resource_id"]
        .as_str()
        .ok_or_else(|| std::io::Error::other("packed bake resource ID is missing"))?
        .to_owned();
    assert_eq!(
        packed_scene["resources"][packed_resource_id.as_str()]["storage"],
        "inline"
    );
    assert_eq!(files_below(&cache_directory)?.len(), disk_files.len());
    assert!(disk_files.iter().all(|path| path.is_file()));

    for payload in files_below(&cache_directory)? {
        fs::remove_file(payload)?;
    }
    assert_no_files(&cache_directory)?;
    let packed_result = inspect(&project, 1.0)?;
    assert_dimensions(&packed_result, [2.0, 2.0, 2.0])?;
    assert_bake_record(&packed_result)?;

    let unpacked = bake(&project, None, None, BakeAction::Unpack)?;
    assert_eq!(unpacked["result"]["scene_modified"], true);
    let unpacked_scene = document(&project)?;
    let unpacked_params = &unpacked_scene["nodes"]["body"]["modifiers"][0]["params"];
    assert_eq!(unpacked_params["bake_target"], "DISK");
    assert_eq!(unpacked_params["bakes"]["bake"]["bake_target"], "DISK");
    assert!(
        unpacked_scene["resources"]
            .get(packed_resource_id.as_str())
            .is_none()
    );
    assert_project_disk_payload(&project, &local_cache_directory)?;
    let unpacked_result = inspect(&project, 1.0)?;
    assert_dimensions(&unpacked_result, [2.0, 2.0, 2.0])?;
    assert_bake_record(&unpacked_result)?;

    let repacked = bake(&project, None, None, BakeAction::Pack)?;
    assert_eq!(repacked["result"]["storage"], "packed");
    let local_payloads = files_below(&local_cache_directory)?;
    assert_eq!(local_payloads.len(), 1);
    fs::write(&local_payloads[0], b"corrupt local bake payload")?;
    let write_local = bake_with_unpack_method(
        &project,
        None,
        None,
        BakeAction::Unpack,
        Some("write-local"),
    )?;
    assert_eq!(write_local["result"]["storage"], "disk");
    assert_project_disk_payload(&project, &local_cache_directory)?;
    assert_dimensions(&inspect(&project, 1.0)?, [2.0, 2.0, 2.0])?;

    let deleted = bake(&project, None, None, BakeAction::Delete)?;
    assert_eq!(deleted["result"]["scene_modified"], true);
    assert_no_files(&local_cache_directory)?;
    assert_no_files(&cache_directory)?;
    assert_dimensions(&inspect(&project, 1.0)?, [4.0, 4.0, 4.0])?;
    Ok(())
}

#[test]
fn deleting_packed_disk_node_bake_cleans_payloads_and_uses_live_geometry()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let project = directory.path().join("packed-disk-delete-project");
    let batch_file = directory.path().join("operations.json");
    let cache_directory = project.join("cache-a");
    create_project(&project, &batch_file, "DISK", "//cache-a", false)?;

    let baked = bake(&project, None, None, BakeAction::Bake)?;
    assert_eq!(baked["result"]["scene_modified"], true);
    assert_project_disk_payload(&project, &cache_directory)?;
    let disk_files = files_below(&cache_directory)?;

    let packed = bake(&project, None, None, BakeAction::Pack)?;
    assert_eq!(packed["result"]["scene_modified"], true);
    assert_eq!(packed["result"]["storage"], "packed");
    let packed_scene = document(&project)?;
    let packed_params = &packed_scene["nodes"]["body"]["modifiers"][0]["params"];
    assert_eq!(packed_params["bakes"]["bake"]["bake_target"], "PACKED");
    let resource_id = packed_params["bakes"]["bake"]["resource_id"]
        .as_str()
        .ok_or_else(|| std::io::Error::other("packed bake resource ID is missing"))?
        .to_owned();
    assert_eq!(packed_scene["resources"][&resource_id]["storage"], "inline");
    assert_eq!(files_below(&cache_directory)?.len(), disk_files.len());
    assert!(disk_files.iter().all(|path| path.is_file()));

    set_cube_size(&project, &batch_file, 4.0)?;
    let packed_result = inspect(&project, 1.0)?;
    assert_dimensions(&packed_result, [2.0, 2.0, 2.0])?;
    assert_bake_record(&packed_result)?;

    let deleted = bake(&project, None, None, BakeAction::Delete)?;
    assert_eq!(deleted["result"]["scene_modified"], true);
    let deleted_scene = document(&project)?;
    assert!(deleted_scene["resources"].get(&resource_id).is_none());
    let settings = &deleted_scene["nodes"]["body"]["modifiers"][0]["params"]["bakes"]["bake"];
    assert!(settings.get("storage").is_none());
    assert!(settings.get("resource_id").is_none());
    assert_eq!(settings["bake_target"], "PACKED");
    assert_no_files(&cache_directory)?;
    assert!(disk_files.iter().all(|path| !path.exists()));
    assert_dimensions(&inspect(&project, 1.0)?, [4.0, 4.0, 4.0])?;
    Ok(())
}

#[test]
fn failed_node_bake_unpack_preserves_scene_and_conflicting_payload() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let project = directory.path().join("failed-unpack-project");
    let batch_file = directory.path().join("operations.json");
    let cache_directory = project.join("cache-a");
    create_project(&project, &batch_file, "DISK", "//cache-a", false)?;
    bake(&project, None, None, BakeAction::Bake)?;
    bake(&project, None, None, BakeAction::Pack)?;

    let before = document(&project)?;
    let before_params = &before["nodes"]["body"]["modifiers"][0]["params"];
    let before_config = before_params["bakes"]["bake"].clone();
    assert_eq!(before_config["storage"], "packed");
    let payload_path = files_below(&cache_directory)?
        .into_iter()
        .next()
        .ok_or_else(|| std::io::Error::other("disk payload is missing"))?;
    assert!(payload_path.is_file());
    fs::write(&payload_path, b"corrupt existing payload")?;

    let error = bake_with_unpack_method(
        &project,
        None,
        None,
        BakeAction::Unpack,
        Some("use-original"),
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("existing node-bake file has invalid content")
    );
    let after = document(&project)?;
    assert_eq!(after["revision"], before["revision"]);
    assert_eq!(
        after["nodes"]["body"]["modifiers"][0]["params"]["bakes"]["bake"],
        before_config
    );
    assert_eq!(
        after["nodes"]["body"]["modifiers"][0]["params"]["bake_target"],
        before_params["bake_target"]
    );
    assert_eq!(fs::read(&payload_path)?, b"corrupt existing payload");
    let unpacked = bake_with_unpack_method(
        &project,
        None,
        None,
        BakeAction::Unpack,
        Some("write-original"),
    )?;
    assert_eq!(unpacked["result"]["storage"], "disk");
    assert_dimensions(&inspect(&project, 1.0)?, [2.0, 2.0, 2.0])?;
    assert!(
        files_below(&cache_directory)?
            .iter()
            .all(|path| path.is_file())
    );
    Ok(())
}

#[cfg(unix)]
#[test]
fn failed_multiframe_write_original_unpack_restores_conflicting_payload()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let project = directory.path().join("multiframe-unpack-project");
    let batch_file = directory.path().join("operations.json");
    let payload_directory = project.join("cache-a/body/geometry_modifier/bake");
    create_project(&project, &batch_file, "DISK", "//cache-a", true)?;
    bake(&project, Some("1:3"), Some("animation"), BakeAction::Bake)?;

    let disk_scene = document(&project)?;
    let disk_manifest =
        disk_scene["nodes"]["body"]["modifiers"][0]["params"]["bakes"]["bake"].clone();
    let disk_frames = disk_manifest["frames"]
        .as_array()
        .ok_or_else(|| std::io::Error::other("disk bake manifest is missing"))?;
    assert_eq!(disk_frames.len(), 3);
    let first_payload = node_bake_payload_path(&payload_directory, &disk_manifest, 1.0)?;
    let later_payload = node_bake_payload_path(&payload_directory, &disk_manifest, 2.0)?;
    assert_ne!(first_payload, later_payload);
    assert!(first_payload.is_file());
    assert!(later_payload.is_file());
    bake(&project, None, None, BakeAction::Pack)?;

    let before = document(&project)?;
    let before_config = before["nodes"]["body"]["modifiers"][0]["params"]["bakes"]["bake"].clone();
    assert_eq!(before_config["storage"], "packed");

    let conflicting_payload = b"pre-existing conflicting payload";
    fs::write(&first_payload, conflicting_payload)?;
    let symlink_target = directory.path().join("unpack-symlink-target.json");
    let symlink_target_bytes = b"preserve symlink target";
    fs::write(&symlink_target, symlink_target_bytes)?;
    fs::remove_file(&later_payload)?;
    std::os::unix::fs::symlink(&symlink_target, &later_payload)?;

    bake_with_unpack_method(
        &project,
        None,
        None,
        BakeAction::Unpack,
        Some("write-original"),
    )
    .unwrap_err();

    let after = document(&project)?;
    assert_eq!(after["revision"], before["revision"]);
    assert_eq!(
        after["nodes"]["body"]["modifiers"][0]["params"]["bakes"]["bake"],
        before_config
    );
    assert_eq!(after, before);
    assert_eq!(fs::read(&first_payload)?, conflicting_payload);
    assert!(
        fs::symlink_metadata(&later_payload)?
            .file_type()
            .is_symlink()
    );
    assert_eq!(fs::read_link(&later_payload)?, symlink_target);
    assert_eq!(fs::read(&symlink_target)?, symlink_target_bytes);
    Ok(())
}

#[cfg(unix)]
#[test]
fn failed_multiframe_disk_delete_preserves_scene_and_earlier_payload() -> Result<(), Box<dyn Error>>
{
    let directory = tempdir()?;
    let project = directory.path().join("multiframe-delete-project");
    let batch_file = directory.path().join("operations.json");
    let payload_directory = project.join("cache-a/body/geometry_modifier/bake");
    create_project(&project, &batch_file, "DISK", "//cache-a", true)?;
    bake(&project, Some("1:3"), Some("animation"), BakeAction::Bake)?;

    let before = document(&project)?;
    let before_config = before["nodes"]["body"]["modifiers"][0]["params"]["bakes"]["bake"].clone();
    assert_eq!(before_config["storage"], "disk");
    let frames = before_config["frames"]
        .as_array()
        .ok_or_else(|| std::io::Error::other("disk bake manifest is missing"))?;
    assert_eq!(frames.len(), 3);
    let first_payload = node_bake_payload_path(&payload_directory, &before_config, 1.0)?;
    let later_payload = node_bake_payload_path(&payload_directory, &before_config, 2.0)?;
    assert_ne!(first_payload, later_payload);
    let first_payload_bytes = fs::read(&first_payload)?;
    let symlink_target = directory.path().join("delete-symlink-target.json");
    let symlink_target_bytes = b"preserve delete symlink target";
    fs::write(&symlink_target, symlink_target_bytes)?;
    fs::remove_file(&later_payload)?;
    std::os::unix::fs::symlink(&symlink_target, &later_payload)?;

    bake(&project, None, None, BakeAction::Delete).unwrap_err();

    let after = document(&project)?;
    assert_eq!(after["revision"], before["revision"]);
    assert_eq!(
        after["nodes"]["body"]["modifiers"][0]["params"]["bakes"]["bake"],
        before_config
    );
    assert_eq!(after, before);
    assert_eq!(fs::read(&first_payload)?, first_payload_bytes);
    assert!(
        fs::symlink_metadata(&later_payload)?
            .file_type()
            .is_symlink()
    );
    assert_eq!(fs::read_link(&later_payload)?, symlink_target);
    assert_eq!(fs::read(&symlink_target)?, symlink_target_bytes);
    Ok(())
}

#[test]
fn rebaking_a_shared_packed_cache_preserves_other_resource_users() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let project = directory.path().join("shared-bake-project");
    let batch_file = directory.path().join("operations.json");
    create_project(&project, &batch_file, "PACKED", "", false)?;
    bake(&project, None, None, BakeAction::Bake)?;
    let before = document(&project)?;
    let modifiers = before["nodes"]["body"]["modifiers"]
        .as_array()
        .ok_or_else(|| std::io::Error::other("body modifier list is missing"))?;
    let original_modifier = modifiers
        .iter()
        .find(|modifier| modifier["id"] == "geometry_modifier")
        .ok_or_else(|| std::io::Error::other("geometry modifier is missing"))?;
    let shared_config = original_modifier["params"]["bakes"]["bake"].clone();
    let shared_id = shared_config["resource_id"]
        .as_str()
        .ok_or_else(|| std::io::Error::other("packed bake resource ID is missing"))?
        .to_owned();
    let shared_resource = before["resources"][shared_id.as_str()].clone();

    apply(
        &project,
        &batch_file,
        &json!([{
            "op":"modifier.create",
            "target":{"id":"body"},
            "id":"geometry_modifier_copy",
            "type":"nodes",
            "params":{"node_group":"geo","bake_target":"PACKED","bakes":{"bake":shared_config}}
        }]),
    )?;
    bake(&project, None, None, BakeAction::Bake)?;

    let after = document(&project)?;
    let modifiers = after["nodes"]["body"]["modifiers"]
        .as_array()
        .ok_or_else(|| std::io::Error::other("body modifier list is missing"))?;
    let source_config = modifiers
        .iter()
        .find(|modifier| modifier["id"] == "geometry_modifier")
        .map(|modifier| &modifier["params"]["bakes"]["bake"])
        .ok_or_else(|| std::io::Error::other("source modifier disappeared"))?;
    let shared_config = modifiers
        .iter()
        .find(|modifier| modifier["id"] == "geometry_modifier_copy")
        .map(|modifier| &modifier["params"]["bakes"]["bake"])
        .ok_or_else(|| std::io::Error::other("shared modifier disappeared"))?;
    assert_ne!(source_config["resource_id"], shared_id);
    assert_eq!(shared_config["resource_id"], shared_id);
    assert_eq!(after["resources"][shared_id.as_str()], shared_resource);
    Ok(())
}

#[test]
fn packed_bake_feeds_preview_render_and_gltf_after_input_change() -> Result<(), Box<dyn Error>> {
    let expected = blender_still_bake_positions()?;
    let directory = tempdir()?;
    let project = directory.path().join("packed-consumers-project");
    let batch_file = directory.path().join("operations.json");
    run_json(
        pot().arg("init").arg(&project),
        "initializing packed Bake consumer project",
    )?;
    apply(
        &project,
        &batch_file,
        &json!(transformed_cube_bake_operations("PACKED", "")),
    )?;

    let baked = bake(&project, None, None, BakeAction::Bake)?;
    assert_eq!(baked["result"]["storage"], "packed");
    set_transform_translation(&project, &batch_file, 100.0)?;
    let cached = inspect(&project, 1.0)?;
    assert_eq!(cached["evaluated_geometry"]["vertex_count"], 8);
    assert_eq!(cached["evaluated_geometry"]["face_count"], 6);
    assert_positions_equal(&mesh_positions(&cached)?, &expected);

    apply(
        &project,
        &batch_file,
        &json!([
            {"op":"camera.create","id":"camera_main","transform":{"translation":[1.5,0.0,5.0]},"projection":"orthographic","ortho_scale":4.0},
            {"op":"light.create","id":"key_light","light_type":"point","energy":12.0,"radius":0.8,"transform":{"translation":[-1.0,-1.0,3.0]}},
            {"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"camera_main"}},
            {"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_x":64,"resolution_y":64,"samples":1,"seed":0,"max_bounces":1}}
        ]),
    )?;

    let preview_directory = directory.path().join("preview");
    run_json(
        pot()
            .arg("preview")
            .arg(&project)
            .args(["--views", "iso", "--size", "64", "--out"])
            .arg(&preview_directory),
        "previewing packed Bake output",
    )?;
    let preview_path = preview_directory.join("iso.png");
    assert!(preview_path.is_file(), "preview PNG was not created");
    let preview_bytes = fs::read(&preview_path)?;
    assert!(!preview_bytes.is_empty(), "preview PNG is empty");
    assert!(
        preview_bytes.starts_with(b"\x89PNG\r\n\x1a\n"),
        "preview output does not have a PNG signature"
    );
    assert_png_varies(&preview_path)?;

    let render_directory = directory.path().join("render");
    run_json(
        pot()
            .arg("render")
            .arg(&project)
            .args(["--format", "png", "--frames", "1:1", "--out"])
            .arg(&render_directory),
        "rendering packed Bake output",
    )?;
    let render_path = render_directory.join("frame_0001.png");
    assert!(render_path.is_file(), "render PNG was not created");
    let render_bytes = fs::read(&render_path)?;
    assert!(!render_bytes.is_empty(), "render PNG is empty");
    assert!(
        render_bytes.starts_with(b"\x89PNG\r\n\x1a\n"),
        "render output does not have a PNG signature"
    );
    assert_png_varies(&render_path)?;

    let glb_path = directory.path().join("packed-bake.glb");
    run_json(
        pot()
            .arg("export")
            .arg(&project)
            .args(["--format", "glb", "--out"])
            .arg(&glb_path),
        "exporting packed Bake output as GLB",
    )?;
    assert!(glb_path.is_file(), "GLB export was not created");
    let glb_bytes = fs::read(&glb_path)?;
    assert!(!glb_bytes.is_empty(), "GLB export is empty");
    assert!(
        glb_bytes.starts_with(b"glTF"),
        "export does not have a GLB signature"
    );

    let imported_project = directory.path().join("imported-project");
    run_json(
        pot().arg("init").arg(&imported_project),
        "initializing GLB import project",
    )?;
    let imported_revision = document(&imported_project)?["revision"]
        .as_u64()
        .ok_or_else(|| std::io::Error::other("initialized project revision is missing"))?;
    run_json(
        pot()
            .arg("import")
            .arg(&imported_project)
            .arg("--file")
            .arg(&glb_path)
            .args(["--format", "glb", "--base-revision"])
            .arg(imported_revision.to_string()),
        "importing packed Bake GLB",
    )?;
    let imported = run_json(
        pot().arg("inspect").arg(&imported_project),
        "inspecting imported packed Bake GLB",
    )?;
    let imported_items = imported["result"]["items"]
        .as_array()
        .ok_or_else(|| std::io::Error::other("imported inspect returned no items array"))?;
    let imported_geometry = imported_items
        .iter()
        .find(|item| {
            if item["evaluated_geometry"]["vertex_count"].as_u64() != Some(8)
                || item["evaluated_geometry"]["face_count"].as_u64() != Some(6)
            {
                return false;
            }
            let Ok(positions) = mesh_positions(item) else {
                return false;
            };
            positions.len() == expected.len()
                && positions.iter().zip(&expected).all(|(actual, expected)| {
                    (0..3).all(|axis| (actual[axis] - expected[axis]).abs() < 1.0e-9)
                })
        })
        .ok_or_else(|| std::io::Error::other("imported GLB omitted cached baked geometry"))?;
    assert_eq!(imported_geometry["evaluated_geometry"]["vertex_count"], 8);
    assert_eq!(imported_geometry["evaluated_geometry"]["face_count"], 6);
    assert_positions_equal(&mesh_positions(imported_geometry)?, &expected);
    Ok(())
}

#[test]
fn modifier_documents_without_bake_settings_have_no_bakes_key_and_load_unchanged()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let project = directory.path().join("old-document");
    let batch_file = directory.path().join("operations.json");
    create_project(&project, &batch_file, "PACKED", "", false)?;
    let scene = document(&project)?;
    let params = &scene["nodes"]["body"]["modifiers"][0]["params"];
    assert!(params.get("bakes").is_none(), "{params}");
    let unbaked = inspect(&project, 1.0)?;
    assert_dimensions(&unbaked, [2.0, 2.0, 2.0])?;
    Ok(())
}

/// The Geometry Nodes Bake node of a scene: its node group ID and node ID.
fn find_bake_node(scene: &Value) -> Result<(String, String), Box<dyn Error>> {
    for (group_id, group) in scene["node_groups"].as_object().into_iter().flatten() {
        for (node_id, node) in group["nodes"].as_object().into_iter().flatten() {
            if node["type"] == "GeometryNodeBake" {
                return Ok((group_id.clone(), node_id.clone()));
            }
        }
    }
    Err(std::io::Error::other("scene has no Bake node").into())
}

/// Number of links that touch the given node of a node group.
fn links_touching(scene: &Value, group_id: &str, node_id: &str) -> usize {
    scene["node_groups"][group_id]["links"]
        .as_array()
        .map_or(0, |links| {
            links
                .iter()
                .filter(|link| link["from_node"] == node_id || link["to_node"] == node_id)
                .count()
        })
}

#[test]
fn bake_node_survives_blend_export_and_import() -> Result<(), Box<dyn Error>> {
    // Needs Blender for the .blend export and import; skip like the other Blender tests when absent.
    if std::env::var_os("POTTER_BLENDER").is_none() {
        assert!(
            std::env::var_os("POTTER_REQUIRE_BLENDER").is_none(),
            "POTTER_REQUIRE_BLENDER is set but POTTER_BLENDER is not"
        );
        eprintln!("skipping Bake node .blend round trip: POTTER_BLENDER is unset");
        return Ok(());
    }
    let directory = tempdir()?;
    let project = directory.path().join("bake-node-project");
    let batch_file = directory.path().join("operations.json");
    create_project(&project, &batch_file, "PACKED", "", false)?;
    let before = document(&project)?;
    let (group_id, node_id) = find_bake_node(&before)?;
    let links_before = links_touching(&before, &group_id, &node_id);
    assert!(links_before > 0, "the Bake node has no links to round-trip");
    let items_before =
        &before["node_groups"][&group_id]["nodes"][&node_id]["properties"]["bake_items"];
    assert_eq!(items_before[0]["name"], "Geometry");

    let blend = directory.path().join("bake-node.blend");
    run_json(
        pot()
            .arg("export")
            .arg(&project)
            .args(["--format", "blend", "--out"])
            .arg(&blend),
        "exporting the Bake node to .blend",
    )?;
    let imported = directory.path().join("imported-bake-node");
    run_json(
        pot().arg("init").arg(&imported),
        "initializing the import project",
    )?;
    let revision = document(&imported)?["revision"]
        .as_u64()
        .ok_or_else(|| std::io::Error::other("initialized project revision is missing"))?;
    run_json(
        pot()
            .arg("import")
            .arg(&imported)
            .arg("--file")
            .arg(&blend)
            .args(["--format", "blend", "--base-revision"])
            .arg(revision.to_string()),
        "importing the Bake node .blend",
    )?;

    let after = document(&imported)?;
    let (imported_group, imported_node) = find_bake_node(&after)?;
    let items =
        &after["node_groups"][&imported_group]["nodes"][&imported_node]["properties"]["bake_items"];
    assert_eq!(items.as_array().map(Vec::len), Some(1), "{items}");
    assert_eq!(items[0]["name"], "Geometry", "{items}");
    assert_eq!(
        links_touching(&after, &imported_group, &imported_node),
        links_before,
        "the Bake node's links changed across the .blend round trip"
    );
    Ok(())
}

#[test]
fn packed_bake_without_data_block_references_is_reported_on_import() -> Result<(), Box<dyn Error>> {
    let Some(blender) = std::env::var_os("POTTER_BLENDER") else {
        assert!(
            std::env::var_os("POTTER_REQUIRE_BLENDER").is_none(),
            "POTTER_REQUIRE_BLENDER is set but POTTER_BLENDER is not"
        );
        eprintln!("skipping packed Bake loss check: POTTER_BLENDER is unset");
        return Ok(());
    };
    let directory = tempdir()?;
    let blend = directory.path().join("packed-bake.blend");
    let script = r#"
import bpy, sys
blend = sys.argv[sys.argv.index("--") + 1]
obj = bpy.context.object
tree = bpy.data.node_groups.new("Geometry", "GeometryNodeTree")
tree.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
cube = tree.nodes.new("GeometryNodeMeshCube")
bake_node = tree.nodes.new("GeometryNodeBake")
bake_node.bake_items.new("GEOMETRY", "Geometry")
output = tree.nodes.new("NodeGroupOutput")
tree.links.new(cube.outputs["Mesh"], bake_node.inputs["Geometry"])
tree.links.new(bake_node.outputs["Geometry"], output.inputs["Geometry"])
modifier = obj.modifiers.new("GeometryNodes", "NODES")
modifier.node_group = tree
modifier.bake_target = "PACKED"
bake = modifier.bakes[0]
assert len(bake.data_blocks) == 0
result = bpy.ops.object.geometry_node_bake_single(
    session_uid=obj.session_uid, modifier_name=modifier.name, bake_id=bake.bake_id)
assert "FINISHED" in result
assert len(bake.data_blocks) == 0
bpy.ops.wm.save_as_mainfile(filepath=blend)
"#;
    run_ok(
        Command::new(blender)
            .args([
                "-b",
                "--factory-startup",
                "--python-exit-code",
                "1",
                "--python-expr",
                script,
                "--",
            ])
            .arg(&blend),
        "creating a packed geometry Bake with no data-block references",
    )?;
    let imported = directory.path().join("packed-bake-import");
    run_json(
        pot().arg("init").arg(&imported),
        "initializing the import project",
    )?;
    let revision = document(&imported)?["revision"]
        .as_u64()
        .ok_or_else(|| std::io::Error::other("initialized project revision is missing"))?;
    let output = pot()
        .arg("import")
        .arg(&imported)
        .arg("--file")
        .arg(&blend)
        .args(["--format", "blend", "--base-revision"])
        .arg(revision.to_string())
        .arg("--json")
        .output()?;
    assert!(!output.status.success());
    let response: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(response["error"]["code"], "UNREPRESENTABLE_FEATURE");
    assert!(
        response["result"]["losses"]
            .as_array()
            .is_some_and(|losses| {
                losses
                    .iter()
                    .any(|loss| loss["feature_id"] == "blender.modifier.nodes_bake_packed")
            })
    );
    Ok(())
}
