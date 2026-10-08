#![expect(
    clippy::unwrap_used,
    reason = "integration fixtures use fixed paths and valid JSON"
)]

use std::{
    error::Error,
    fs,
    path::Path,
    process::{Command, Output},
};

use serde_json::{Value, json};
use tempfile::tempdir;

type TestResult = Result<(), Box<dyn Error>>;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn init(scene: &Path) -> TestResult {
    let output = pot().arg("init").arg(scene).arg("--json").output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    Ok(())
}

fn apply(scene: &Path, revision: u64, operations: &Value) -> Result<Output, Box<dyn Error>> {
    let file = scene.with_extension("ops.json");
    fs::write(
        &file,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": revision,
            "operations": operations
        }))?,
    )?;
    Ok(pot()
        .arg("apply")
        .arg(scene)
        .arg("--file")
        .arg(file)
        .arg("--json")
        .output()?)
}

fn apply_ok(scene: &Path, revision: u64, operations: &Value) -> Result<Value, Box<dyn Error>> {
    let output = apply(scene, revision, operations)?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn inspect(scene: &Path, id: &str, frame: Option<&str>) -> Result<Value, Box<dyn Error>> {
    let mut command = pot();
    command.arg("inspect").arg(scene).arg("--id").arg(id);
    if let Some(frame) = frame {
        command.arg("--frame").arg(frame);
    }
    let output = command.arg("--json").output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn assert_invalid_operation(output: &Output, pointer_suffix: &str) -> Result<(), Box<dyn Error>> {
    assert!(!output.status.success());
    let response: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(response["error"]["code"], "INVALID_OPERATION");
    let pointer = response["error"]["details"]["pointer"]
        .as_str()
        .ok_or("operation error omitted its JSON pointer")?;
    assert!(
        pointer.ends_with(pointer_suffix),
        "expected pointer ending in {pointer_suffix}, got {pointer}"
    );
    Ok(())
}

fn item<'a>(response: &'a Value, id: &str) -> &'a Value {
    response["result"]["items"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["id"] == id)
        .unwrap()
}

#[test]
fn node_create_collections_and_keep_world_parenting_are_independent() -> TestResult {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init(&scene)?;
    apply_ok(
        &scene,
        0,
        &json!([
            {"op":"collection.create","id":"props"},
            {"op":"node.create","id":"root","kind":"group","transform":{"translation":[4.0,0.0,0.0]}},
            {"op":"node.create","id":"sphere","kind":"sphere","params":{"radius":0.5},"transform":{"translation":[1.0,0.0,0.0]},"collection":"props"},
            {"op":"node.create","id":"sphere_alias","kind":"uv_sphere","params":{"radius":0.25},"collection":"props"},
            {"op":"node.parent","target":{"id":"sphere"},"parent":"root","keep_world":true}
        ]),
    )?;

    let document: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    assert_eq!(document["nodes"]["root"]["kind"], "empty");
    assert_eq!(document["nodes"]["sphere"]["kind"], "mesh");
    assert_eq!(document["nodes"]["sphere"]["parent"], "root");
    let sphere_data = document["nodes"]["sphere"]["data"].as_str().unwrap();
    assert_eq!(
        document["data_blocks"][sphere_data]["descriptor"]["primitive"],
        "uv_sphere"
    );
    assert!(
        document["collections"]["props"]["objects"]
            .as_array()
            .unwrap()
            .iter()
            .any(|id| id == "sphere")
    );
    assert!(
        document["collections"]["collection_root"]["children"]
            .as_array()
            .unwrap()
            .iter()
            .any(|id| id == "props")
    );

    let sphere_response = inspect(&scene, "sphere", None)?;
    let sphere = item(&sphere_response, "sphere");
    assert_eq!(
        sphere["transform"]["world"]["translation"],
        json!([1.0, 0.0, 0.0])
    );
    let cycle = apply(
        &scene,
        1,
        &json!([{"op":"node.parent","target":{"id":"root"},"parent":"sphere"}]),
    )?;
    assert_invalid_operation(&cycle, "/parent")?;
    Ok(())
}

#[test]
fn node_join_merges_shape_key_basis_and_positions_in_destination_space() -> TestResult {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init(&scene)?;
    apply_ok(
        &scene,
        0,
        &json!([
            {"op":"node.create","id":"destination","kind":"box","params":{"size":2.0}},
            {"op":"shape_key.create","target":{"id":"destination"},"id":"lift","name":"Lift","positions":{"0":[-1.0,-1.0,0.5]}},
            {"op":"node.create","id":"source","kind":"box","params":{"size":2.0},"transform":{"translation":[3.0,0.0,0.0]}},
            {"op":"shape_key.create","target":{"id":"source"},"id":"lift","name":"Lift","positions":{"0":[-1.0,-1.0,0.75]}},
            {"op":"node.join","targets":["destination","source"]}
        ]),
    )?;

    let document: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    assert!(document["nodes"].get("source").is_none());
    let data_id = document["nodes"]["destination"]["data"].as_str().unwrap();
    let shape_keys = &document["data_blocks"][data_id]["shape_keys"];
    assert_eq!(shape_keys["basis"].as_object().unwrap().len(), 16);
    let positions = shape_keys["keys"]["lift"]["positions"]
        .as_object()
        .unwrap()
        .values()
        .map(|position| serde_json::from_value::<[f64; 3]>(position.clone()).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(positions.len(), 2);
    assert!(positions.contains(&[-1.0, -1.0, 0.5]));
    assert!(positions.contains(&[2.0, -1.0, 0.75]));
    Ok(())
}

#[test]
fn sculpt_reports_sample_and_dyntopo_errors_and_refines_under_brush() -> TestResult {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init(&scene)?;
    apply_ok(
        &scene,
        0,
        &json!([{"op":"node.create","id":"body","kind":"box","params":{"size":2.0}}]),
    )?;
    let invalid_sample = apply(
        &scene,
        1,
        &json!([{
            "op":"sculpt.stroke","target":{"id":"body"},"brush":"draw",
            "samples":[{"position":[0.0,0.0,1.0],"pressure":1.0,"radius":1.0,"strength":0.5,"time":0.0,"unexpected":true}]
        }]),
    )?;
    assert_invalid_operation(&invalid_sample, "/samples/0")?;
    let invalid_edge = apply(
        &scene,
        1,
        &json!([{
            "op":"sculpt.dyntopo","target":{"id":"body"},"edge_length":0.0,
            "samples":[{"position":[0.0,0.0,0.0],"pressure":1.0,"radius":10.0,"strength":1.0,"time":0.0}]
        }]),
    )?;
    assert_invalid_operation(&invalid_edge, "/edge_length")?;

    let before: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let data_id = before["nodes"]["body"]["data"].as_str().unwrap();
    let before_vertices = before["data_blocks"][data_id]["mesh"]["vertices"]
        .as_array()
        .unwrap()
        .len();
    apply_ok(
        &scene,
        1,
        &json!([{
            "op":"sculpt.dyntopo","target":{"id":"body"},"edge_length":0.5,
            "falloff":"constant","samples":[{"position":[0.0,0.0,0.0],"pressure":1.0,"radius":10.0,"strength":1.0,"time":0.0}]
        }]),
    )?;
    let after: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let after_data = after["nodes"]["body"]["data"].as_str().unwrap();
    let after_vertices = after["data_blocks"][after_data]["mesh"]["vertices"]
        .as_array()
        .unwrap()
        .len();
    assert!(after_vertices > before_vertices);
    Ok(())
}

#[test]
fn library_override_replaces_inserts_and_deletes_modifier_properties() -> TestResult {
    let directory = tempdir()?;
    let source = directory.path().join("library_source");
    let scene = directory.path().join("scene");
    init(&source)?;
    apply_ok(
        &source,
        0,
        &json!([
            {"op":"node.create","id":"asset","kind":"box","params":{"size":1.0}},
            {"op":"modifier.create","target":{"id":"asset"},"id":"wave_base","type":"wave"}
        ]),
    )?;
    init(&scene)?;
    apply_ok(
        &scene,
        0,
        &json!([{
            "op":"library.link","id":"source_lib","uri":source,
            "items":[{"registry":"nodes","id":"asset"}]
        }]),
    )?;
    let linked_id = "source_lib__asset";
    apply_ok(
        &scene,
        1,
        &json!([{
            "op":"library.override","target":{"id":linked_id},"id":"local_asset",
            "operations":[
                {"op":"set","path":"transform.translation","value":[2.0,0.0,0.0]},
                {"op":"insert_after","path":"modifiers.wave_base","value":{"id":"wave_added","type":"wave","name":"Added Wave","enabled":true,"params":{"height":0.25}}},
                {"op":"delete","path":"modifiers.wave_base","value":null}
            ]
        }]),
    )?;

    let document: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let local = &document["nodes"]["local_asset"];
    assert_eq!(local["transform"]["translation"], json!([2.0, 0.0, 0.0]));
    assert_eq!(local["modifiers"].as_array().unwrap().len(), 1);
    assert_eq!(local["modifiers"][0]["id"], "wave_added");
    assert_eq!(
        local["properties"]["library_override"]["reference_id"],
        linked_id
    );
    let missing_modifier = apply(
        &scene,
        2,
        &json!([{
            "op":"library.override","target":{"id":linked_id},"id":"bad_override",
            "operations":[{"op":"delete","path":"modifiers.absent","value":null}]
        }]),
    )?;
    assert_invalid_operation(&missing_modifier, "/path")?;
    Ok(())
}

#[test]
fn simulated_particle_render_types_add_visible_particle_geometry() -> TestResult {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init(&scene)?;
    apply_ok(
        &scene,
        0,
        &json!([
            {"op":"collection.create","id":"particle_assets"},
            {"op":"node.create","id":"point_emitter","kind":"plane","params":{}},
            {"op":"physics.particle_emitter.create","target":{"id":"point_emitter"},"settings":{"emit_from":"VERT","count":1,"frame_start":1.0,"frame_end":1.0,"lifetime":8.0,"physics_type":"NO","normal_factor":0.0,"render_type":"HALO","seed":3}},
            {"op":"node.create","id":"object_asset","kind":"box","params":{"size":0.25},"collection":"particle_assets"},
            {"op":"node.create","id":"object_emitter","kind":"plane","params":{},"transform":{"translation":[3.0,0.0,0.0]}},
            {"op":"physics.particle_emitter.create","target":{"id":"object_emitter"},"settings":{"emit_from":"VERT","count":1,"frame_start":1.0,"frame_end":1.0,"lifetime":8.0,"physics_type":"NO","normal_factor":0.0,"render_type":"OBJECT","instance_object":"object_asset","seed":5}},
            {"op":"node.create","id":"collection_emitter","kind":"plane","params":{},"transform":{"translation":[6.0,0.0,0.0]}},
            {"op":"physics.particle_emitter.create","target":{"id":"collection_emitter"},"settings":{"emit_from":"VERT","count":1,"frame_start":1.0,"frame_end":1.0,"lifetime":8.0,"physics_type":"NO","normal_factor":0.0,"render_type":"COLLECTION","instance_collection":"particle_assets","seed":7}},
            {"op":"node.create","id":"hair_emitter","kind":"plane","params":{},"transform":{"translation":[9.0,0.0,0.0]}},
            {"op":"physics.particle_emitter.create","target":{"id":"hair_emitter"},"settings":{"emit_from":"VERT","count":1,"frame_start":1.0,"frame_end":1.0,"lifetime":8.0,"physics_type":"NO","normal_factor":0.0,"render_type":"PATH","hair_length":0.5,"seed":9}}
        ]),
    )?;

    let point_response = inspect(&scene, "point_emitter", Some("1"))?;
    let object_response = inspect(&scene, "object_emitter", Some("1"))?;
    let collection_response = inspect(&scene, "collection_emitter", Some("1"))?;
    let hair_response = inspect(&scene, "hair_emitter", Some("1"))?;
    let points = item(&point_response, "point_emitter");
    let object = item(&object_response, "object_emitter");
    let collection = item(&collection_response, "collection_emitter");
    let hair = item(&hair_response, "hair_emitter");
    assert!(points["evaluated_geometry"]["face_count"].as_u64().unwrap() > 1);
    assert!(hair["evaluated_geometry"]["face_count"].as_u64().unwrap() > 1);
    assert_eq!(object["evaluated_geometry"]["face_count"], 7);
    assert_eq!(collection["evaluated_geometry"]["face_count"], 7);
    Ok(())
}

#[test]
fn active_sphere_bounces_from_static_box_without_tunneling() -> TestResult {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let bake_dir = directory.path().join("simulation");
    init(&scene)?;
    apply_ok(
        &scene,
        0,
        &json!([
            {"op":"node.create","id":"wall","kind":"box","params":{"size":1.0},"transform":{"scale":[1.0,4.0,4.0]}},
            {"op":"node.create","id":"ball","kind":"uv_sphere","params":{"radius":0.25,"segments":12,"ring_count":6},"transform":{"translation":[-2.0,0.0,0.0]}},
            {"op":"physics.world.update","target":{"id":"scene_main"},"set":{"enabled":true,"gravity":[0.0,0.0,0.0],"substeps":8,"solver_iterations":16,"frame_start":1,"frame_end":16}},
            {"op":"physics.rigid_body.create","target":{"id":"wall"},"type":"passive","shape":"box","restitution":1.0,"friction":0.0},
            {"op":"physics.rigid_body.create","target":{"id":"ball"},"type":"active","shape":"sphere","mass":1.0,"restitution":1.0,"friction":0.0,"linear_damping":0.0,"angular_damping":0.0,"initial_velocity":[6.0,0.0,0.0]}
        ]),
    )?;

    let output = pot()
        .arg("bake")
        .arg(&scene)
        .arg("--kind")
        .arg("simulation")
        .arg("--frames")
        .arg("1:16")
        .arg("--out")
        .arg(&bake_dir)
        .arg("--json")
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let manifest: Value = serde_json::from_slice(&fs::read(bake_dir.join("manifest.json"))?)?;
    let last = manifest["frames"].as_array().unwrap().last().unwrap();
    let frame: Value =
        serde_json::from_slice(&fs::read(bake_dir.join(last["file"].as_str().unwrap()))?)?;
    let transform = &frame["transforms"]["ball"];
    assert!(transform["world_matrix"][12].as_f64().unwrap() < 0.0);
    assert!(transform["linear_velocity"][0].as_f64().unwrap() < 0.0);
    Ok(())
}
