use std::{error::Error, fs, path::Path, process::Command};

use serde_json::{Value, json};
use tempfile::tempdir;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn output_json(output: &std::process::Output) -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn init(scene: &Path) -> Result<(), Box<dyn Error>> {
    let output = pot().arg("init").arg(scene).arg("--json").output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    Ok(())
}

fn apply(scene: &Path, batch: &Value) -> Result<Value, Box<dyn Error>> {
    let batch_file = scene.with_file_name("inspect-world-batch.json");
    fs::write(&batch_file, serde_json::to_vec(batch)?)?;
    let output = pot()
        .arg("apply")
        .arg(scene)
        .arg("--file")
        .arg(batch_file)
        .arg("--json")
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    output_json(&output)
}

fn inspect(scene: &Path, args: &[&str]) -> Result<Value, Box<dyn Error>> {
    let output = pot()
        .arg("inspect")
        .arg(scene)
        .args(args)
        .arg("--json")
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    output_json(&output)
}

fn new_scene() -> Result<(tempfile::TempDir, std::path::PathBuf), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().canonicalize()?.join("scene");
    init(&scene)?;
    Ok((directory, scene))
}

fn item_by_id<'a>(items: &'a [Value], id: &str) -> Result<&'a Value, Box<dyn Error>> {
    items
        .iter()
        .find(|item| item["id"] == id)
        .ok_or_else(|| format!("item {id} missing").into())
}

fn assert_vec3(actual: &Value, expected: [f64; 3]) -> Result<(), Box<dyn Error>> {
    let actual = actual.as_array().ok_or("vector missing")?;
    assert_eq!(actual.len(), expected.len());
    for (value, expected) in actual.iter().zip(expected) {
        let value = value.as_f64().ok_or("vector component missing")?;
        assert!(
            (value - expected).abs() <= 1.0e-9,
            "expected {expected}, got {value}"
        );
    }
    Ok(())
}

#[test]
fn parented_child_reports_local_and_world_transforms() -> Result<(), Box<dyn Error>> {
    let (_directory, scene) = new_scene()?;
    apply(
        &scene,
        &json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [
                {"op":"node.create", "id":"parent", "kind":"empty", "transform":{"translation":[5.0,2.0,0.0]}},
                {"op":"node.create", "id":"child", "kind":"empty", "parent":"parent", "transform":{"translation":[1.0,0.0,0.0]}}
            ]
        }),
    )?;

    let inspected = inspect(&scene, &[])?;
    let items = inspected["result"]["items"]
        .as_array()
        .ok_or("items missing")?;
    let child = item_by_id(items, "child")?;
    assert_vec3(
        &child["transform"]["evaluated"]["translation"],
        [1.0, 0.0, 0.0],
    )?;
    assert_vec3(&child["transform"]["world"]["translation"], [6.0, 2.0, 0.0])?;
    assert_eq!(child["transform"]["world"]["decomposable"], true);
    assert_eq!(child["transform"]["evaluated"]["matrix"][12], 1.0);
    assert_eq!(child["transform"]["world"]["matrix"][12], 6.0);
    Ok(())
}

#[test]
fn copy_location_keeps_local_value_and_reports_constrained_world_placement()
-> Result<(), Box<dyn Error>> {
    let (_directory, scene) = new_scene()?;
    apply(
        &scene,
        &json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [
                {"op":"node.create", "id":"target", "kind":"empty", "transform":{"translation":[2.0,3.0,4.0]}},
                {"op":"node.create", "id":"owner", "kind":"empty", "transform":{"translation":[5.0,6.0,7.0]}},
                {"op":"constraint.create", "target":{"id":"owner"}, "id":"follow", "type":"copy_location", "constraint_target":"target"}
            ]
        }),
    )?;

    let inspected = inspect(&scene, &["--id", "owner"])?;
    let owner = &inspected["result"]["items"][0];
    assert_vec3(
        &owner["transform"]["evaluated"]["translation"],
        [5.0, 6.0, 7.0],
    )?;
    assert_vec3(&owner["transform"]["world"]["translation"], [2.0, 3.0, 4.0])?;
    Ok(())
}

#[test]
fn animated_local_and_world_transforms_use_requested_frame() -> Result<(), Box<dyn Error>> {
    let (_directory, scene) = new_scene()?;
    apply(
        &scene,
        &json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [
                {"op":"node.create", "id":"parent", "kind":"empty", "transform":{"translation":[10.0,0.0,0.0]}},
                {"op":"node.create", "id":"animated", "kind":"empty", "parent":"parent"},
                {"op":"action.create", "id":"move_action", "name":"Move"},
                {"op":"node.update", "target":{"id":"animated"}, "set":{"action":"move_action"}},
                {"op":"keyframe.insert", "target":{"id":"animated"}, "path":"transform.translation", "index":0, "frame":1.0, "value":0.0, "interpolation":"linear"},
                {"op":"keyframe.insert", "target":{"id":"animated"}, "path":"transform.translation", "index":0, "frame":24.0, "value":10.0, "interpolation":"linear"}
            ]
        }),
    )?;

    let inspected = inspect(&scene, &["--frame", "12.5"])?;
    let items = inspected["result"]["items"]
        .as_array()
        .ok_or("items missing")?;
    let animated = item_by_id(items, "animated")?;
    assert_eq!(inspected["result"]["evaluation"]["frame"], 12.5);
    assert_vec3(
        &animated["transform"]["evaluated"]["translation"],
        [5.0, 0.0, 0.0],
    )?;
    assert_vec3(
        &animated["transform"]["world"]["translation"],
        [15.0, 0.0, 0.0],
    )?;
    Ok(())
}

#[test]
fn sheared_world_matrix_is_present_but_not_decomposable() -> Result<(), Box<dyn Error>> {
    let (_directory, scene) = new_scene()?;
    apply(
        &scene,
        &json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [
                {"op":"node.create", "id":"shear_parent", "kind":"empty", "transform":{"scale":[2.0,1.0,1.0]}},
                {"op":"node.create", "id":"shear_child", "kind":"empty", "parent":"shear_parent", "transform":{"translation":[1.0,0.0,0.0],"rotation":[0.0,0.0,0.382_683_432_365_089_8,0.923_879_532_511_286_7]}},
                {"op":"node.delete", "target":{"id":"shear_parent"}, "reparent":"to_root"}
            ]
        }),
    )?;

    let inspected = inspect(&scene, &["--id", "shear_child"])?;
    let child = &inspected["result"]["items"][0];
    assert_eq!(child["parent"], Value::Null);
    assert_eq!(child["transform"]["world"]["decomposable"], false);
    assert_eq!(
        child["transform"]["world"]["matrix"]
            .as_array()
            .map(Vec::len),
        Some(16)
    );
    assert_eq!(child["transform"]["world"]["translation"], Value::Null);
    assert_eq!(child["transform"]["world"]["rotation"], Value::Null);
    assert_eq!(child["transform"]["world"]["scale"], Value::Null);
    Ok(())
}

#[test]
fn world_translation_is_inside_world_bounds_for_a_box() -> Result<(), Box<dyn Error>> {
    let (_directory, scene) = new_scene()?;
    apply(
        &scene,
        &json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [
                {"op":"node.create", "id":"box", "kind":"box", "params":{"size":2.0}, "transform":{"translation":[10.0,-2.0,3.0],"scale":[1.0,2.0,3.0]}}
            ]
        }),
    )?;

    let inspected = inspect(&scene, &["--id", "box"])?;
    let box_item = &inspected["result"]["items"][0];
    let world = &box_item["transform"]["world"]["translation"];
    for axis in 0..3 {
        let coordinate = world[axis].as_f64().ok_or("world coordinate missing")?;
        let minimum = box_item["bounds"]["min"][axis]
            .as_f64()
            .ok_or("bounds minimum missing")?;
        let maximum = box_item["bounds"]["max"][axis]
            .as_f64()
            .ok_or("bounds maximum missing")?;
        assert!(minimum <= coordinate && coordinate <= maximum);
    }
    assert_vec3(world, [10.0, -2.0, 3.0])?;
    Ok(())
}
#[test]
fn negative_world_scale_uses_glam_decomposition_convention() -> Result<(), Box<dyn Error>> {
    let (_directory, scene) = new_scene()?;
    apply(
        &scene,
        &json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [
                {"op":"node.create", "id":"reflected", "kind":"empty", "transform":{"scale":[-2.0,3.0,4.0]}}
            ]
        }),
    )?;

    let inspected = inspect(&scene, &["--id", "reflected"])?;
    let reflected = &inspected["result"]["items"][0]["transform"]["world"];
    assert_eq!(reflected["decomposable"], true);
    let scale = reflected["scale"].as_array().ok_or("world scale missing")?;
    assert_eq!(scale.len(), 3);
    assert_eq!(scale[0], -2.0);
    assert_eq!(scale[1], 3.0);
    assert_eq!(scale[2], 4.0);
    Ok(())
}
