use std::{
    error::Error,
    fs,
    path::Path,
    process::{Command, Output},
};

use glam::{DMat4, DQuat, DVec3, EulerRot};
use serde_json::{Map, Value, json};
use tempfile::tempdir;
#[path = "common/blender_file.rs"]
mod blender_file;

use blender_file::blender_executable;

fn evaluated_volume<'a>(
    document: &potter_core::model::SceneDoc,
    snapshot: &'a potter_core::eval::Snapshot,
    node_name: &str,
) -> Result<&'a potter_core::geom::volume::VolumeData, Box<dyn Error>> {
    let node = document
        .nodes
        .get(&potter_core::model::Id::new(node_name.to_owned())?)
        .ok_or_else(|| format!("volume object `{node_name}` is missing"))?;
    let data_id = node.data.as_ref().ok_or("volume Data-Block is missing")?;
    snapshot
        .volume_data
        .get(data_id)
        .ok_or_else(|| format!("evaluated volume `{node_name}` is missing").into())
}

const MODIFIER_POSITION_TOLERANCE: f64 = 1.0e-4;
const FACE_DISTRIBUTION_TOLERANCE: f64 = 0.03;
const VELOCITY_STATISTIC_TOLERANCE: f64 = 0.25;
const SUBDIVISION_POSITION_TOLERANCE: f64 = 0.005;
const CLOTH_STIMULUS_TOLERANCE: f64 = 0.02;
const CLOTH_ORDER_TOLERANCE: f64 = 0.05;
const CLOTH_RMS_TOLERANCE: f64 = 0.30;
// The 2 m pinned-edge fixture has a non-planar free edge; 0.30 m limits RMS, centroid, and
// Z-envelope drift to 15% while exact rest/pin checks and order sensitivity cover other failures.
// A 0.05 m order response is 2.5% of the fixture span and must appear in Blender and Potter.
const CLOTH_CENTROID_TOLERANCE: f64 = 0.30;
const CLOTH_BOUNDS_TOLERANCE: f64 = 0.30;
const DENSITY_SAMPLE_TOLERANCE: f64 = 0.0001;
const MINIMUM_SURFACE_DISPLACEMENT: f64 = 1.0;
fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn run(command: &mut Command, action: &str) -> Result<Output, Box<dyn Error>> {
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

fn initialize(scene: &Path) -> Result<(), Box<dyn Error>> {
    run(
        pot().arg("init").arg(scene).arg("--json"),
        "scene initialization",
    )?;
    Ok(())
}

fn apply(
    scene: &Path,
    batch_path: &Path,
    operations: &Value,
    revision: u64,
) -> Result<(), Box<dyn Error>> {
    fs::write(
        batch_path,
        serde_json::to_vec(&json!({
            "schema_version":1,
            "base_revision":revision,
            "operations":operations,
        }))?,
    )?;
    run(
        pot()
            .arg("apply")
            .arg(scene)
            .arg("--file")
            .arg(batch_path)
            .arg("--json"),
        "scene operation batch",
    )?;
    Ok(())
}

#[test]
fn physics_operations_create_stack_entries_linked_to_their_settings() -> Result<(), Box<dyn Error>>
{
    let directory = tempdir()?;
    let scene = directory.path().join("physics-stack");
    let batch = directory.path().join("operations.json");
    initialize(&scene)?;
    apply(
        &scene,
        &batch,
        &json!([
            {"op":"node.create","id":"body","kind":"box","params":{}},
            {"op":"physics.cloth.create","target":{"id":"body"},"settings":{"tension_stiffness":12.0}},
            {"op":"physics.soft_body.create","target":{"id":"body"},"settings":{"goal_strength":0.7}},
            {"op":"physics.dynamic_paint.create","target":{"id":"body"},"settings":{"role":"canvas"}},
            {"op":"physics.fluid.create","target":{"id":"body"},"settings":{"type":"liquid","resolution":4}},
            {"op":"physics.particle_emitter.create","target":{"id":"body"},"settings":{"count":1,"lifetime":12.0}},
            {"op":"physics.collision.create","target":{"id":"body"},"settings":{"thickness":0.01}}
        ]),
        0,
    )?;

    let document: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let node = &document["nodes"]["body"];
    let modifiers = node["modifiers"]
        .as_array()
        .ok_or("modifier stack missing")?;
    for (modifier_type, id, property) in [
        ("cloth", "physics_cloth", "physics_cloth"),
        ("soft_body", "physics_soft_body", "physics_soft_body"),
        (
            "dynamic_paint",
            "physics_dynamic_paint",
            "physics_dynamic_paint",
        ),
        ("fluid", "physics_fluid", "physics_fluid"),
        (
            "particle_system",
            "physics_particle_emitter",
            "physics_particle_emitter",
        ),
        ("collision", "physics_collision", "physics_collision"),
    ] {
        let modifier = modifiers
            .iter()
            .find(|modifier| modifier["id"] == id)
            .ok_or_else(|| format!("{modifier_type} modifier was not created"))?;
        assert_eq!(modifier["type"], modifier_type);
        assert_eq!(modifier["params"]["settings_id"], id);
        assert!(node["properties"].get(property).is_some());
    }

    apply(
        &scene,
        &batch,
        &json!([{
            "op":"physics.cloth.update",
            "target":{"id":"body"},
            "set":{"compression_stiffness":10.0}
        }]),
        1,
    )?;
    let updated: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    assert_eq!(
        updated["nodes"]["body"]["properties"]["physics_cloth"]["compression_stiffness"],
        10.0
    );
    Ok(())
}

#[test]
fn modifier_flag_updates_keep_linked_physics_settings() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("physics-flags");
    let batch = directory.path().join("operations.json");
    initialize(&scene)?;
    apply(
        &scene,
        &batch,
        &json!([
            {"op":"node.create","id":"body","kind":"box","params":{}},
            {"op":"physics.cloth.create","target":{"id":"body"},"settings":{"mass":2.0,"quality":7,"tension_stiffness":12.0}},
            {"op":"physics.particle_emitter.create","target":{"id":"body"},"settings":{"count":1,"lifetime":12.0}},
            {"op":"modifier.create","target":{"id":"body"},"id":"second_emitter","type":"particle_system","params":{"settings_id":"second_emitter","show_expanded":true}}
        ]),
        0,
    )?;
    // The modifier path stores no emitter settings for a registry entry (import provides them), so
    // seed the entry the way an import does, then check the flag-only update leaves it intact.
    let seeded_path = scene.join("scene.json");
    let mut seeded: Value = serde_json::from_slice(&fs::read(&seeded_path)?)?;
    seeded["nodes"]["body"]["properties"]["physics_particle_systems"]["second_emitter"] =
        json!({"count":3,"lifetime":5.0});
    fs::write(&seeded_path, serde_json::to_vec(&seeded)?)?;
    let before: Value = serde_json::from_slice(&fs::read(&seeded_path)?)?;
    let stored_before = before["nodes"]["body"]["properties"].clone();
    assert_eq!(stored_before["physics_cloth"]["mass"], 2.0);
    assert_eq!(
        stored_before["physics_particle_systems"]["second_emitter"]["count"],
        3.0
    );

    apply(
        &scene,
        &batch,
        &json!([
            {"op":"modifier.update","target":{"id":"body"},"modifier_id":"physics_cloth","set":{"params":{"show_expanded":false}}},
            {"op":"modifier.update","target":{"id":"body"},"modifier_id":"second_emitter","set":{"params":{"show_expanded":false}}}
        ]),
        1,
    )?;
    let after: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let stored_after = &after["nodes"]["body"]["properties"];
    assert_eq!(
        stored_after["physics_cloth"],
        stored_before["physics_cloth"]
    );
    assert_eq!(
        stored_after["physics_particle_emitter"],
        stored_before["physics_particle_emitter"]
    );
    assert_eq!(
        stored_after["physics_particle_systems"],
        stored_before["physics_particle_systems"]
    );
    let modifiers = after["nodes"]["body"]["modifiers"]
        .as_array()
        .ok_or("modifier stack missing")?;
    for id in ["physics_cloth", "second_emitter"] {
        let modifier = modifiers
            .iter()
            .find(|modifier| modifier["id"] == id)
            .ok_or_else(|| format!("{id} modifier missing"))?;
        assert_eq!(modifier["params"]["show_expanded"], false);
    }
    Ok(())
}

fn injected_particles(
    states: &Value,
) -> Result<Vec<potter_core::sim::ParticleState>, Box<dyn Error>> {
    let states = states
        .as_array()
        .ok_or("Blender particle states are missing")?;
    states
        .iter()
        .map(|state| {
            let position: [f64; 3] = serde_json::from_value(state["position"].clone())?;
            let birth_position: [f64; 3] = serde_json::from_value(state["birth_position"].clone())?;
            let velocity: [f64; 3] = serde_json::from_value(state["velocity"].clone())?;
            let birth_frame = state["birth_frame"]
                .as_f64()
                .ok_or("Blender particle birth frame is missing")?;
            let death_frame = state["death_frame"]
                .as_f64()
                .ok_or("Blender particle death frame is missing")?;
            let normal: [f64; 3] = serde_json::from_value(state["normal"].clone())?;
            let rotation: [f64; 4] = serde_json::from_value(state["rotation"].clone())?;
            let birth_rotation: [f64; 4] = serde_json::from_value(state["birth_rotation"].clone())?;
            let size = state["size"]
                .as_f64()
                .ok_or("Blender particle size is missing")?;
            let path = state["path"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|key| {
                    Ok(potter_core::sim::ParticlePathKey {
                        time: key["time"]
                            .as_f64()
                            .ok_or("Blender path-key time is missing")?,
                        position: serde_json::from_value(key["position"].clone())?,
                        velocity: serde_json::from_value(key["velocity"].clone())?,
                        rotation: serde_json::from_value(key["rotation"].clone())?,
                    })
                })
                .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
            let system_index = state["system_index"].as_u64().unwrap_or(0) as u32;
            let is_child = state["is_child"].as_bool().unwrap_or(false);
            let parent_index = state["parent_index"]
                .as_u64()
                .map(u32::try_from)
                .transpose()?;
            let path_root_rotation = state
                .get("path_root_rotation")
                .filter(|rotation| !rotation.is_null())
                .map(|rotation| serde_json::from_value::<[f64; 4]>(rotation.clone()))
                .transpose()?;
            let life_state = match state["life_state"]
                .as_str()
                .ok_or("Blender particle life state is missing")?
            {
                "ALIVE" => potter_core::sim::ParticleLifeState::Alive,
                "DEAD" => potter_core::sim::ParticleLifeState::Dead,
                "UNBORN" => potter_core::sim::ParticleLifeState::Unborn,
                state => return Err(format!("unknown Blender particle state `{state}`").into()),
            };
            let path_interpolation = match state["path_interpolation"].as_str() {
                Some("linear") => potter_core::sim::ParticlePathInterpolation::Linear,
                Some("hair_cache") => potter_core::sim::ParticlePathInterpolation::HairCache,
                Some("hair_cardinal") => potter_core::sim::ParticlePathInterpolation::HairCardinal,
                Some("hair_bspline") => potter_core::sim::ParticlePathInterpolation::HairBspline,
                Some("hermite") | None => potter_core::sim::ParticlePathInterpolation::Hermite,
                Some(mode) => {
                    return Err(format!("unknown particle path interpolation `{mode}`").into());
                }
            };
            Ok(potter_core::sim::ParticleState {
                position,
                birth_position,
                normal,
                rotation,
                birth_rotation,
                velocity,
                birth_frame,
                death_frame,
                life_state,
                system_index,
                is_child,
                parent_index,
                path_root_rotation,
                path,
                path_interpolation,
                size,
                ..potter_core::sim::ParticleState::default()
            })
        })
        .collect()
}

fn add_corner_fixture_data(
    mesh: &mut potter_core::geom::Mesh,
    raw_codes: &Value,
    include_fixture_color: bool,
    include_hair_color: bool,
) {
    let raw_codes = raw_codes.as_array().expect("Blender INT16_2D corner codes");
    let color_layer = if include_fixture_color {
        Some(("FixtureCornerColor", "BYTE_COLOR"))
    } else if include_hair_color {
        Some(("HairCornerColor", "FLOAT_COLOR"))
    } else {
        None
    };
    let mut source_uv_faces = Vec::with_capacity(mesh.faces.len());
    let mut particle_uv_faces = Vec::with_capacity(mesh.faces.len());
    let mut normal_codes = serde_json::Map::new();
    let mut color_codes = serde_json::Map::new();
    let mut loop_index = 0;
    for (face_index, face) in mesh.faces.iter_mut().enumerate() {
        face.material_index = (face_index % 3) as u32;
        let mut source_uvs = Vec::with_capacity(face.vertices.len());
        let mut particle_uvs = Vec::with_capacity(face.vertices.len());
        let mut color_row = Vec::with_capacity(face.vertices.len());
        let mut code_row = Vec::with_capacity(face.vertices.len());
        for corner in 0..face.vertices.len() {
            let source_uv = [
                0.1 + 0.17 * face_index as f64 + 0.11 * corner as f64,
                0.2 + 0.13 * face_index as f64 + 0.07 * corner as f64,
            ];
            source_uvs.push(json!([source_uv[0], 1.0 - source_uv[1]]));
            particle_uvs.push(json!([source_uv[0] + 0.031, 1.0 - (source_uv[1] + 0.047)]));
            if include_fixture_color {
                color_row.push(match corner % 4 {
                    0 => json!([1.0, 0.0, 0.0, 1.0]),
                    1 => json!([0.0, 1.0, 0.0, 1.0]),
                    2 => json!([0.0, 0.0, 1.0, 1.0]),
                    _ => json!([1.0, 1.0, 1.0, 1.0]),
                });
            } else if include_hair_color {
                color_row.push(json!([
                    0.12 + 0.11 * face_index as f64 + 0.07 * corner as f64,
                    0.21 + 0.09 * face_index as f64 + 0.05 * corner as f64,
                    0.31 + 0.07 * face_index as f64 + 0.03 * corner as f64,
                    1.0
                ]));
            }
            code_row.push(raw_codes[loop_index].clone());
            loop_index += 1;
        }
        source_uv_faces.push(json!({"face_id":face.id,"uv":source_uvs}));
        particle_uv_faces.push(json!({"face_id":face.id,"uv":particle_uvs}));
        normal_codes.insert(format!("f{}", face.id), Value::Array(code_row));
        if color_layer.is_some() {
            color_codes.insert(format!("f{}", face.id), Value::Array(color_row));
        }
    }
    assert_eq!(loop_index, raw_codes.len(), "Blender source code count");
    mesh.attributes.insert(
        "uv_layers".to_owned(),
        json!({
            "layers":[
                {"name":"SourceUV","faces":source_uv_faces},
                {"name":"ParticleUV","faces":particle_uv_faces}
            ],
            "active":"SourceUV",
            "render":"SourceUV"
        }),
    );
    mesh.attributes.insert(
        "custom_normal".to_owned(),
        json!({"domain":"corner", "type":"normal_code", "values":normal_codes}),
    );
    if let Some((name, data_type)) = color_layer {
        let mut attribute = serde_json::Map::new();
        attribute.insert("domain".to_owned(), json!("corner"));
        attribute.insert("type".to_owned(), json!("color"));
        attribute.insert("values".to_owned(), Value::Object(color_codes));
        mesh.attributes
            .insert(name.to_owned(), Value::Object(attribute));
        mesh.attributes.insert(
            "color_layers".to_owned(),
            json!({"layers":[name],"active":name,"render":name}),
        );
        let mut data_types = serde_json::Map::new();
        data_types.insert(name.to_owned(), json!(data_type));
        mesh.attributes.insert(
            "blender_metadata".to_owned(),
            json!({"color_data_types":data_types}),
        );
    }
}

fn assert_numeric_rows(actual: &Value, expected: &Value, context: &str) {
    let actual = actual.as_array().expect("actual numeric rows");
    let expected = expected.as_array().expect("expected numeric rows");
    assert_eq!(actual.len(), expected.len(), "{context}: row count");
    for (row, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let actual = actual.as_array().expect("actual numeric row");
        let expected = expected.as_array().expect("expected numeric row");
        assert_eq!(actual.len(), expected.len(), "{context}: row {row} width");
        for (axis, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let delta = actual.as_f64().expect("actual number")
                - expected.as_f64().expect("expected number");
            assert!(
                delta.abs() <= MODIFIER_POSITION_TOLERANCE,
                "{context}: row {row} component {axis} differs by {delta}",
            );
        }
    }
}

fn assert_ordered_mesh_matches_blender(
    mesh: &potter_core::geom::Mesh,
    expected: &Value,
    context: &str,
) {
    assert_numeric_rows(
        &serde_json::to_value(
            mesh.vertices
                .iter()
                .map(|vertex| vertex.co.to_array())
                .collect::<Vec<_>>(),
        )
        .unwrap(),
        &expected["positions"],
        &format!("{context} positions"),
    );
    let vertex_indices = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<std::collections::BTreeMap<_, _>>();
    let faces = mesh
        .faces
        .iter()
        .map(|face| {
            face.vertices
                .iter()
                .map(|id| vertex_indices[id])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let expected_faces: Vec<Vec<usize>> =
        serde_json::from_value(expected["faces"].clone()).expect("Blender face loops");
    assert_eq!(faces, expected_faces, "{context} ordered face loops");
    let materials = mesh
        .faces
        .iter()
        .map(|face| face.material_index)
        .collect::<Vec<_>>();
    let expected_materials: Vec<u32> =
        serde_json::from_value(expected["materials"].clone()).expect("Blender material indices");
    assert_eq!(materials, expected_materials, "{context} material indices");

    for (name, expected_uvs) in expected["uv_layers"].as_object().unwrap() {
        let layer = mesh
            .attributes
            .get("uv_layers")
            .and_then(|store| store.get("layers"))
            .and_then(Value::as_array)
            .and_then(|layers| layers.iter().find(|layer| layer["name"] == *name))
            .unwrap_or_else(|| panic!("{context}: missing UV layer {name}"));
        let faces = layer["faces"]
            .as_array()
            .expect("canonical UV face entries");
        let mut actual_uvs = Vec::new();
        for face in &mesh.faces {
            let row = faces
                .iter()
                .find(|entry| entry["face_id"].as_u64() == Some(u64::from(face.id)))
                .and_then(|entry| entry["uv"].as_array())
                .unwrap_or_else(|| {
                    panic!("{context}: missing UVs for layer {name} face {}", face.id)
                });
            assert_eq!(
                row.len(),
                face.vertices.len(),
                "{context} UV {name} face {}",
                face.id
            );
            for uv in row {
                let uv = uv.as_array().expect("canonical UV pair");
                let u = uv[0].as_f64().expect("canonical UV u");
                let v = uv[1].as_f64().expect("canonical UV v");
                actual_uvs.push(json!([u, 1.0 - v]));
            }
        }
        assert_numeric_rows(
            &Value::Array(actual_uvs),
            expected_uvs,
            &format!("{context} UV {name}"),
        );
    }

    let normal_state = mesh.attributes.get("custom_normal");
    if let Some(expected_codes) = expected["custom_normal_codes"].as_array() {
        let normal_state = normal_state.expect("canonical custom-normal state");
        assert_eq!(normal_state["domain"], "corner", "{context} normal domain");
        assert_eq!(
            normal_state["type"], "normal_code",
            "{context} normal code type"
        );
        let values = normal_state["values"].as_object().unwrap();
        let corner_order = normal_state.get("corner_order").and_then(Value::as_object);
        let corner_count = mesh
            .faces
            .iter()
            .map(|face| face.vertices.len())
            .sum::<usize>();
        let mut actual_codes = vec![Value::Null; corner_count];
        let mut offset = 0;
        for face in &mesh.faces {
            let key = format!("f{}", face.id);
            let row = values.get(&key).and_then(Value::as_array).unwrap();
            let order = corner_order
                .and_then(|orders| orders.get(&key))
                .and_then(Value::as_array);
            assert_eq!(
                row.len(),
                face.vertices.len(),
                "{context} {key} normal code count"
            );
            for (corner, code) in row.iter().enumerate() {
                let ordinal = order
                    .and_then(|order| order.get(corner))
                    .and_then(Value::as_u64)
                    .map_or(offset + corner, |value| value as usize);
                actual_codes[ordinal] = code.clone();
            }
            offset += face.vertices.len();
        }
        assert_eq!(
            Value::Array(actual_codes),
            Value::Array(expected_codes.clone()),
            "{context} Blender INT16_2D custom-normal codes",
        );
    }

    let expected_layer_names: Vec<String> =
        serde_json::from_value(expected["color_layer_names"].clone()).unwrap();
    let color_registry = mesh
        .attributes
        .get("color_layers")
        .and_then(Value::as_object);
    let actual_layer_names: Vec<String> = color_registry
        .and_then(|registry| registry.get("layers"))
        .and_then(Value::as_array)
        .map(|layers| {
            layers
                .iter()
                .map(|layer| layer.as_str().unwrap().to_owned())
                .collect()
        })
        .unwrap_or_default();
    assert_eq!(
        actual_layer_names, expected_layer_names,
        "{context} color layer order"
    );
    for (name, expected_color) in expected["color_attributes"].as_object().unwrap() {
        // These fixture names are not reserved, so the canonical storage key is the user name.
        let attribute = mesh
            .attributes
            .get(name)
            .unwrap_or_else(|| panic!("{context}: missing color layer {name}"));
        assert_eq!(attribute["domain"], "corner", "{context} {name} domain");
        assert_eq!(attribute["type"], "color", "{context} {name} type");
        assert_eq!(
            mesh.attributes["blender_metadata"]["color_data_types"][name],
            expected_color["data_type"],
            "{context} {name} source encoding",
        );
        let values = attribute["values"].as_object().unwrap();
        let actual_values = mesh
            .faces
            .iter()
            .flat_map(|face| {
                values
                    .get(&format!("f{}", face.id))
                    .and_then(Value::as_array)
                    .unwrap()
                    .iter()
                    .cloned()
            })
            .collect::<Vec<_>>();
        assert_numeric_rows(
            &Value::Array(actual_values),
            &expected_color["values"],
            &format!("{context} color {name}"),
        );
    }
    let expected_groups = expected["vertex_groups"].as_object().unwrap();
    let actual_groups = mesh
        .attributes
        .get("vertex_groups")
        .and_then(Value::as_object);
    assert_eq!(
        actual_groups.map_or(0, serde_json::Map::len),
        expected_groups.len(),
        "{context} vertex-group count",
    );
    for (name, expected_weights) in expected_groups {
        let expected_weights = expected_weights.as_array().unwrap();
        let stored = actual_groups
            .and_then(|groups| groups.get(name))
            .and_then(Value::as_object)
            .unwrap_or_else(|| panic!("{context}: missing vertex group {name}"));
        assert_eq!(
            mesh.vertices.len(),
            expected_weights.len(),
            "{context} {name} size"
        );
        for (index, vertex) in mesh.vertices.iter().enumerate() {
            let actual = stored
                .get(&format!("v{}", vertex.id))
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            let wanted = expected_weights[index].as_f64().unwrap();
            assert!(
                (actual - wanted).abs() <= MODIFIER_POSITION_TOLERANCE,
                "{context}: {name} weight at vertex {index} differs: {actual} != {wanted}",
            );
        }
    }
}

fn assert_snapshot_mesh_matches_blender(
    document: &potter_core::model::SceneDoc,
    object_id: &str,
    frame: f64,
    expected: &Value,
    context: &str,
) -> Result<(), Box<dyn Error>> {
    let id = potter_core::model::Id::new(object_id.to_owned())?;
    let snapshot = potter_core::eval::Snapshot::evaluate(
        document,
        &potter_core::eval::EvaluationContext {
            frame: Some(frame),
            ..potter_core::eval::EvaluationContext::default()
        },
    )?;
    let mesh = snapshot
        .meshes
        .get(&id)
        .ok_or("evaluated particle mesh is missing")?;
    assert_ordered_mesh_matches_blender(mesh, expected, context);
    Ok(())
}
#[test]
fn particle_instance_transforms_and_explode_face_splitting_match_blender()
-> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping modifier simulation Blender parity; Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let blender_script = root.join("modifier_parity.py");
    let blender_expected = root.join("blender_expected.json");
    fs::write(
        &blender_script,
        r"
import bpy, json, os, sys
from mathutils import Vector
if tuple(bpy.app.version[:3]) != (5,2,2):
    raise RuntimeError('particle parity requires Blender 5.2.2')
root = sys.argv[sys.argv.index('--') + 1]

def matrix_array(matrix):
    return [matrix[row][column] for column in range(4) for row in range(4)]
scene = bpy.context.scene
scene.use_gravity = False
scene.frame_set(1)

def particle_births(system):
    return [{
        'position':list(particle.location),
        'rotation':[particle.rotation.x, particle.rotation.y, particle.rotation.z, particle.rotation.w],
    } for particle in system.particles]

def particle_states(system, births=None):
    states = [{
        'position':list(particle.location),
        'velocity':list(particle.velocity),
        'birth_frame':particle.birth_time,
        'death_frame':particle.birth_time + particle.lifetime,
        'normal':list(particle.rotation @ Vector((1,0,0))),
        'rotation':[particle.rotation.x, particle.rotation.y, particle.rotation.z, particle.rotation.w],
        'birth_position':list(particle.prev_location),
        'birth_rotation':[particle.prev_rotation.x, particle.prev_rotation.y, particle.prev_rotation.z, particle.prev_rotation.w],
        'size':particle.size,
        'life_state':particle.alive_state,
    } for particle in system.particles]
    if births is not None:
        for state, birth in zip(states, births):
            state['birth_position'] = birth['position']
            state['birth_rotation'] = birth['rotation']
    return states
def add_corner_data(mesh):
    for material_index in range(3):
        material = bpy.data.materials.new('FixtureMaterial%d' % material_index)
        mesh.materials.append(material)
    for polygon in mesh.polygons:
        polygon.material_index = polygon.index % 3
    source_uv = mesh.uv_layers.new(name='SourceUV')
    particle_uv = mesh.uv_layers.new(name='ParticleUV')
    for polygon in mesh.polygons:
        for local_corner, loop_index in enumerate(range(polygon.loop_start, polygon.loop_start + polygon.loop_total)):
            uv = (0.1 + 0.17 * polygon.index + 0.11 * local_corner,
                  0.2 + 0.13 * polygon.index + 0.07 * local_corner)
            source_uv.data[loop_index].uv = uv
            particle_uv.data[loop_index].uv = (uv[0] + 0.031, uv[1] + 0.047)
    corner_color = mesh.color_attributes.new(name='FixtureCornerColor', type='BYTE_COLOR', domain='CORNER')
    fixture_colors = ((1,0,0,1), (0,1,0,1), (0,0,1,1), (1,1,1,1))
    for polygon in mesh.polygons:
        for local_corner, loop_index in enumerate(range(polygon.loop_start, polygon.loop_start + polygon.loop_total)):
            corner_color.data[loop_index].color = fixture_colors[local_corner % len(fixture_colors)]
    mesh.normals_split_custom_set([
        Vector((0.13 * ((loop.index % 3) - 1), 0.09 * ((loop.index % 4) - 1.5), 1.0)).normalized()
        for loop in mesh.loops
    ])
    mesh.update()
def corner_normal_codes(mesh):
    attribute = mesh.attributes.get('custom_normal')
    if attribute is None:
        return None
    if attribute.domain != 'CORNER' or attribute.data_type != 'INT16_2D':
        raise RuntimeError('custom normals must be Blender INT16_2D CORNER codes')
    codes = []
    for item in attribute.data:
        pair = [int(value) for value in item.value]
        codes.append(None if pair == [0,0] else pair)
    return codes


def mesh_snapshot(mesh, obj):
    vertex_groups = {}
    for group in obj.vertex_groups:
        values = []
        for vertex in mesh.vertices:
            weight = next((item.weight for item in vertex.groups if item.group == group.index), 0.0)
            values.append(weight)
        vertex_groups[group.name] = values
    return {
        'positions':[list(vertex.co) for vertex in mesh.vertices],
        'faces':[list(polygon.vertices) for polygon in mesh.polygons],
        'materials':[int(polygon.material_index) for polygon in mesh.polygons],
        'uv_layers':{layer.name:[list(item.uv) for item in layer.data] for layer in mesh.uv_layers},
        'custom_normal_codes':corner_normal_codes(mesh),
        'color_layer_names':[layer.name for layer in mesh.color_attributes if layer.domain == 'CORNER'],
        'color_attributes':{
            layer.name:{'domain':layer.domain, 'data_type':layer.data_type,
                        'values':[list(item.color) for item in layer.data]}
            for layer in mesh.color_attributes if layer.domain == 'CORNER'
        },
        'vertex_groups':vertex_groups,
    }


emitter_mesh = bpy.data.meshes.new('asymmetric_emitter_mesh')
emitter_mesh.from_pydata(
    [(-1,-0.5,0), (1.3,-0.6,0.1), (0.7,1.2,0.2), (-0.5,0.7,0.5)],
    [],
    [(0,1,2,3)],
)
emitter = bpy.data.objects.new('Emitter', emitter_mesh)
scene.collection.objects.link(emitter)
emitter.location = (3,-1,0.5)
emitter.rotation_euler = (0,0,0)
emitter.scale = (1,1,1)
bpy.context.view_layer.objects.active = emitter
emitter.select_set(True)
bpy.ops.object.particle_system_add()
emitter_system = emitter.particle_systems[0]
emitter_system.seed = 17
emitter_system.child_seed = 31
particle_settings = emitter_system.settings
particle_settings.count = 1
particle_settings.frame_start = 1
particle_settings.frame_end = 1
particle_settings.lifetime = 12
particle_settings.emit_from = 'VERT'
particle_settings.physics_type = 'NEWTON'
particle_settings.normal_factor = 1.3
particle_settings.factor_random = 0.0
particle_settings.particle_size = 0.7
particle_settings.size_random = 0.25
particle_settings.child_type = 'SIMPLE'
particle_settings.child_percent = 1
particle_settings.child_radius = 0.0
particle_settings.child_size = 1.0
scene.frame_set(1)
bpy.context.view_layer.update()
birth_graph = bpy.context.evaluated_depsgraph_get()
emitter_births = particle_births(emitter.evaluated_get(birth_graph).particle_systems[0])
scene.frame_set(2)
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
evaluated_emitter = emitter.evaluated_get(depsgraph)
instance_particle_states = particle_states(evaluated_emitter.particle_systems[0], emitter_births)
child_particle_count = len(evaluated_emitter.particle_systems[0].child_particles)
emitter_world_matrix = matrix_array(evaluated_emitter.matrix_world)
emitter.select_set(False)

instance_mesh = bpy.data.meshes.new('asymmetric_instance_mesh')
instance_mesh.from_pydata(
    [(-0.7,-0.3,0.1), (1.1,-0.2,0), (0.3,0.9,0.2), (-0.4,0.4,0.6)],
    [],
    [(0,1,2,3)],
)
add_corner_data(instance_mesh)
instance_source_normal_codes = corner_normal_codes(instance_mesh)
instance = bpy.data.objects.new('Instance', instance_mesh)
scene.collection.objects.link(instance)
instance.location = (0,0,0)
instance.rotation_euler = (0,0,0)
instance.scale = (1,1,1)
instance.track_axis = 'POS_Z'
modifier = instance.modifiers.new('Particle Instance', 'PARTICLE_INSTANCE')
modifier.object = emitter
modifier.particle_system_index = 1
modifier.axis = 'Z'
modifier.use_normal = True
modifier.use_children = False
modifier.use_size = True
modifier.use_path = False
modifier.show_alive = True
modifier.show_dead = False
modifier.show_unborn = False
modifier.space = 'WORLD'
modifier.rotation = 0.0
modifier.index_layer_name = 'InstanceIndex'
modifier.value_layer_name = 'InstanceValue'
depsgraph = bpy.context.evaluated_depsgraph_get()
evaluated = instance.evaluated_get(depsgraph)
evaluated_mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
instance_points = [list(vertex.co) for vertex in evaluated_mesh.vertices]
instance_mesh_snapshot = mesh_snapshot(evaluated_mesh, evaluated)
instance_world_matrix = matrix_array(evaluated.matrix_world)
evaluated.to_mesh_clear()

instance_local = bpy.data.objects.new('InstanceLocal', instance_mesh)
scene.collection.objects.link(instance_local)
instance_local.location = (0,0,0)
instance_local.rotation_euler = (0,0,0)
instance_local.scale = (1,1,1)
instance_local.track_axis = 'POS_Z'
local_modifier = instance_local.modifiers.new('Local Particle Instance', 'PARTICLE_INSTANCE')
local_modifier.object = emitter
local_modifier.particle_system_index = 1
local_modifier.axis = 'Z'
local_modifier.use_normal = True
local_modifier.use_children = False
local_modifier.use_size = True
local_modifier.show_alive = True
local_modifier.show_dead = False
local_modifier.show_unborn = False
local_modifier.space = 'LOCAL'
local_modifier.index_layer_name = 'InstanceIndex'
local_modifier.value_layer_name = 'InstanceValue'
depsgraph = bpy.context.evaluated_depsgraph_get()
evaluated_local = instance_local.evaluated_get(depsgraph)
local_mesh = evaluated_local.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
instance_local_points = [list(vertex.co) for vertex in local_mesh.vertices]
instance_local_mesh_snapshot = mesh_snapshot(local_mesh, evaluated_local)
instance_local_world_matrix = matrix_array(evaluated_local.matrix_world)
evaluated_local.to_mesh_clear()
def make_visibility_instance(name, show_alive, show_dead, show_unborn):
    obj = bpy.data.objects.new(name, instance_mesh)
    scene.collection.objects.link(obj)
    obj.location = (0,0,0)
    obj.rotation_euler = (0,0,0)
    obj.scale = (1,1,1)
    obj.track_axis = 'POS_Z'
    modifier = obj.modifiers.new(name, 'PARTICLE_INSTANCE')
    modifier.object = emitter
    modifier.particle_system_index = 1
    modifier.axis = 'Z'
    modifier.use_normal = True
    modifier.use_children = False
    modifier.use_size = True
    modifier.show_alive = show_alive
    modifier.show_dead = show_dead
    modifier.show_unborn = show_unborn
    modifier.space = 'WORLD'
    modifier.index_layer_name = 'InstanceIndex'
    modifier.value_layer_name = 'InstanceValue'
    return obj

def evaluated_points(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    points = [list(vertex.co) for vertex in mesh.vertices]
    evaluated.to_mesh_clear()
    return points
def evaluated_mesh_snapshot(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    snapshot = mesh_snapshot(mesh, evaluated)
    evaluated.to_mesh_clear()
    return snapshot


unborn_instance = make_visibility_instance('InstanceUnborn', False, False, True)
scene.frame_set(0)
bpy.context.view_layer.update()
unborn_particles = particle_states(
    emitter.evaluated_get(bpy.context.evaluated_depsgraph_get()).particle_systems[0],
    emitter_births,
)
unborn_instance_points = evaluated_points(unborn_instance)
unborn_instance_mesh_snapshot = evaluated_mesh_snapshot(unborn_instance)

dead_instance = make_visibility_instance('InstanceDead', False, True, False)
scene.frame_set(14)
bpy.context.view_layer.update()
dead_particles = particle_states(
    emitter.evaluated_get(bpy.context.evaluated_depsgraph_get()).particle_systems[0],
    emitter_births,
)
dead_instance_points = evaluated_points(dead_instance)
dead_instance_mesh_snapshot = evaluated_mesh_snapshot(dead_instance)

emitter.select_set(True)
bpy.context.view_layer.objects.active = emitter
particle_cache = emitter.particle_systems[0].point_cache
particle_cache.frame_start = 1
particle_cache.frame_end = 14
bpy.ops.ptcache.bake_all(bake=True)
scene.frame_set(14)
bpy.context.view_layer.update()

explode_mesh_data = bpy.data.meshes.new('explode_asymmetric_mesh')
explode_mesh_data.from_pydata(
    [(-0.7,-0.3,0.1), (1.1,-0.2,0), (0.3,0.9,0.2), (-0.4,0.4,0.6)],
    [],
    [(0,1,2,3)],
)
add_corner_data(explode_mesh_data)
exploder = bpy.data.objects.new('Exploder', explode_mesh_data)
scene.collection.objects.link(exploder)
protect_group = exploder.vertex_groups.new(name='Protect')
for vertex_index, weight in enumerate([0.0, 0.35, 0.0, 0.85]):
    if weight > 0.0:
        protect_group.add([vertex_index], weight, 'REPLACE')
exploder.location = (0.4,-0.2,0.6)
exploder.rotation_euler = (0.23,0.38,-0.19)
exploder.scale = (1.2,0.8,1.1)
bpy.context.view_layer.objects.active = exploder
exploder.select_set(True)
bpy.ops.object.particle_system_add()
explode_system = exploder.particle_systems[0]
explode_system.seed = 23
explode_system.child_seed = 37
exploder_settings = explode_system.settings
exploder_settings.count = 2
exploder_settings.frame_start = 1
exploder_settings.frame_end = 1
exploder_settings.lifetime = 12
exploder_settings.emit_from = 'VERT'
exploder_settings.physics_type = 'NEWTON'
exploder_settings.normal_factor = 1.1
explode = exploder.modifiers.new('Explode', 'EXPLODE')
explode.use_edge_cut = True
explode.use_size = True
explode.show_alive = True
explode.show_dead = False
explode.show_unborn = False
explode.vertex_group = 'Protect'
explode.protect = 0.6
explode.invert_vertex_group = True
explode.particle_uv = 'ParticleUV'
scene.frame_set(1)
bpy.context.view_layer.update()
birth_graph = bpy.context.evaluated_depsgraph_get()
explode_births = particle_births(exploder.evaluated_get(birth_graph).particle_systems[0])
scene.frame_set(2)
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
evaluated_exploder = exploder.evaluated_get(depsgraph)
explode_particle_states = particle_states(
    evaluated_exploder.particle_systems[0], explode_births
)
exploder_world_matrix = matrix_array(evaluated_exploder.matrix_world)
exploded_mesh = evaluated_exploder.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
explode_world_points = [
    list(evaluated_exploder.matrix_world @ vertex.co) for vertex in exploded_mesh.vertices
]
face_count = len(exploded_mesh.polygons)
explode_mesh_snapshot = mesh_snapshot(exploded_mesh, evaluated_exploder)
evaluated_exploder.to_mesh_clear()

bpy.context.view_layer.objects.active = None
cube_mesh_data = bpy.data.meshes.new('ordered_cube_mesh')
cube_mesh_data.from_pydata(
    [(-1,-1,-1), (1,-1,-1), (1,1,-1), (-1,1,-1),
     (-1,-1,1), (1,-1,1), (1,1,1), (-1,1,1)],
    [],
    [(0,3,2,1), (4,5,6,7), (0,1,5,4), (1,2,6,5), (2,3,7,6), (3,0,4,7)],
)
add_corner_data(cube_mesh_data)
cube_source_normal_codes = corner_normal_codes(cube_mesh_data)
cube_exploder = bpy.data.objects.new('CubeExploder', cube_mesh_data)
scene.collection.objects.link(cube_exploder)
cube_exploder.rotation_euler = (-0.31,0.24,0.42)
cube_exploder.scale = (0.9,1.25,0.75)
bpy.context.view_layer.objects.active = cube_exploder
cube_exploder.select_set(True)
bpy.ops.object.particle_system_add()
cube_system = cube_exploder.particle_systems[0]
cube_system.seed = 29
cube_system.child_seed = 41
cube_settings = cube_system.settings
cube_settings.count = 2
cube_settings.frame_start = 1
cube_settings.frame_end = 1
cube_settings.lifetime = 12
cube_settings.emit_from = 'VERT'
cube_settings.physics_type = 'NEWTON'
cube_settings.normal_factor = 1.1
cube_explode = cube_exploder.modifiers.new('Explode with edge cuts', 'EXPLODE')
cube_explode.use_edge_cut = True
cube_explode.use_size = True
cube_explode.show_alive = True
cube_explode.show_dead = False
cube_explode.show_unborn = False
cube_explode.particle_uv = 'ParticleUV'
scene.frame_set(1)
bpy.context.view_layer.update()
cube_birth_graph = bpy.context.evaluated_depsgraph_get()
cube_births = particle_births(cube_exploder.evaluated_get(cube_birth_graph).particle_systems[0])
cube_inverse = cube_exploder.matrix_world.inverted()
cube_birth_local = []
for birth in cube_births:
    sampled_position = cube_inverse @ Vector(birth['position'])
    source_vertex = min(
        cube_mesh_data.vertices,
        key=lambda vertex: (vertex.co - sampled_position).length_squared,
    )
    cube_birth_local.append(source_vertex.co.copy())
    birth['position'] = list(cube_exploder.matrix_world @ source_vertex.co)
cube_face_owners = []
cube_vertex_owners = [len(cube_births)] * len(cube_mesh_data.vertices)
for polygon in cube_mesh_data.polygons:
    center = sum(
        (cube_mesh_data.vertices[index].co for index in polygon.vertices),
        Vector((0,0,0)),
    ) / len(polygon.vertices)
    owner = min(
        range(len(cube_birth_local)),
        key=lambda index: (cube_birth_local[index] - center).length_squared,
    )
    cube_face_owners.append(owner)
    for index in polygon.vertices:
        cube_vertex_owners[index] = owner
cube_cut_masks = []
for polygon in cube_mesh_data.polygons:
    mask = 0
    vertices = list(polygon.vertices)
    for index in range(len(vertices)):
        if cube_vertex_owners[vertices[index]] != cube_vertex_owners[vertices[(index + 1) % len(vertices)]]:
            mask |= 1 << index
    cube_cut_masks.append(mask)
scene.frame_set(2)
bpy.context.view_layer.update()
cube_depsgraph = bpy.context.evaluated_depsgraph_get()
evaluated_cube_exploder = cube_exploder.evaluated_get(cube_depsgraph)
cube_particles = particle_states(
    evaluated_cube_exploder.particle_systems[0], cube_births
)
cube_matrix = matrix_array(evaluated_cube_exploder.matrix_world)
cube_mesh = evaluated_cube_exploder.to_mesh(preserve_all_data_layers=True, depsgraph=cube_depsgraph)
cube_world_points = [
    list(evaluated_cube_exploder.matrix_world @ vertex.co) for vertex in cube_mesh.vertices
]
cube_face_count = len(cube_mesh.polygons)
cube_mesh_snapshot = mesh_snapshot(cube_mesh, evaluated_cube_exploder)
evaluated_cube_exploder.to_mesh_clear()
with open(os.path.join(root, 'blender_expected.json'), 'w', encoding='utf-8') as output:
    json.dump({
        'instance_points':instance_points,
'instance_mesh':instance_mesh_snapshot,
'instance_local_mesh':instance_local_mesh_snapshot,
'unborn_instance_mesh':unborn_instance_mesh_snapshot,
'dead_instance_mesh':dead_instance_mesh_snapshot,
'explode_mesh':explode_mesh_snapshot,
'cube_explode_mesh':cube_mesh_snapshot,
'instance_source_normal_codes':instance_source_normal_codes,
'cube_source_normal_codes':cube_source_normal_codes,
        'instance_child_seed':emitter_system.child_seed,
        'instance_local_points':instance_local_points,
        'instance_particle_states':instance_particle_states,
        'child_particle_count':child_particle_count,
        'unborn_instance_points':unborn_instance_points,
        'unborn_particle_states':unborn_particles,
        'dead_instance_points':dead_instance_points,
        'dead_particle_states':dead_particles,
        'emitter_world_matrix':emitter_world_matrix,
        'instance_seed':emitter_system.seed,
        'explode_seed':explode_system.seed,
        'cube_explode_seed':cube_system.seed,
        'instance_world_matrix':instance_world_matrix,
        'instance_local_world_matrix':instance_local_world_matrix,
        'exploder_world_matrix':exploder_world_matrix,
        'explode_world_points':explode_world_points,
        'explode_particle_states':explode_particle_states,
        'explode_face_count':face_count,
        'cube_explode_world_points':cube_world_points,
        'cube_explode_particle_states':cube_particles,
        'cube_exploder_world_matrix':cube_matrix,
        'cube_face_owners':cube_face_owners,
        'cube_cut_masks':cube_cut_masks,
        'cube_birth_local':[list(position) for position in cube_birth_local],
        'cube_explode_face_count':cube_face_count,
    }, output)
",
    )?;
    let blender_output = Command::new(&blender)
        .args(["--background", "--factory-startup", "--python"])
        .arg(&blender_script)
        .arg("--")
        .arg(root)
        .output()?;
    if !blender_output.status.success() {
        return Err(format!(
            "Blender modifier fixture failed: stdout={} stderr={}",
            String::from_utf8_lossy(&blender_output.stdout),
            String::from_utf8_lossy(&blender_output.stderr),
        )
        .into());
    }
    if !blender_expected.is_file() {
        return Err(format!(
            "Blender did not write its evaluated modifier output: stdout={} stderr={}",
            String::from_utf8_lossy(&blender_output.stdout),
            String::from_utf8_lossy(&blender_output.stderr),
        )
        .into());
    }
    let blender: Value = serde_json::from_slice(&fs::read(&blender_expected)?)?;
    let instance_particles = injected_particles(&blender["instance_particle_states"])?;
    let child_particle_count = blender["child_particle_count"]
        .as_u64()
        .ok_or("Blender child particle count is missing")?;
    assert!(
        child_particle_count > 0,
        "Blender's evaluated Simple-child system must contain children",
    );
    let explode_particles = injected_particles(&blender["explode_particle_states"])?;
    assert_eq!(instance_particles.len(), 1);
    assert_eq!(explode_particles.len(), 2);
    let mut instance_source = potter_core::geom::Mesh::from_positions_and_faces(
        vec![
            DVec3::new(-0.7, -0.3, 0.1),
            DVec3::new(1.1, -0.2, 0.0),
            DVec3::new(0.3, 0.9, 0.2),
            DVec3::new(-0.4, 0.4, 0.6),
        ],
        vec![vec![0, 1, 2, 3]],
    )?;
    add_corner_fixture_data(
        &mut instance_source,
        &blender["instance_source_normal_codes"],
        true,
        false,
    );
    let emitter_columns: [f64; 16] =
        serde_json::from_value(blender["emitter_world_matrix"].clone())?;
    let emitter_world = DMat4::from_cols_array(&emitter_columns);
    let instance_seed = blender["instance_seed"]
        .as_u64()
        .ok_or("Blender particle system seed is missing")? as u32;
    let direct_instance = potter_core::geom::modifiers::particle_instance::particle_instance(
        &instance_source,
        &instance_particles,
        instance_seed,
        emitter_world,
        2,
        &Map::from_iter([
            ("use_normal".to_owned(), json!(true)),
            ("use_size".to_owned(), json!(true)),
            ("use_path".to_owned(), json!(false)),
            ("axis".to_owned(), json!("Z")),
            ("space".to_owned(), json!("WORLD")),
            ("index_layer_name".to_owned(), json!("InstanceIndex")),
            ("value_layer_name".to_owned(), json!("InstanceValue")),
        ]),
    )?;
    assert_ordered_mesh_matches_blender(
        &direct_instance,
        &blender["instance_mesh"],
        "Particle Instance",
    );
    let direct_instance_points = direct_instance
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    let blender_instance_points = blender_points(&blender["instance_points"])?;
    assert_eq!(
        direct_instance_points.len(),
        blender_instance_points.len(),
        "Blender instance points={}, particle states={instance_particles:?}",
        blender_instance_points.len(),
    );
    let instance_error =
        max_corresponding_distance(&direct_instance_points, &blender_instance_points)?;
    assert!(
        instance_error <= MODIFIER_POSITION_TOLERANCE,
        "particle instance per-vertex error {instance_error} exceeds {MODIFIER_POSITION_TOLERANCE}; Potter={direct_instance_points:?}; Blender={blender_instance_points:?}; particles={instance_particles:?}",
    );
    let direct_local_instance = potter_core::geom::modifiers::particle_instance::particle_instance(
        &instance_source,
        &instance_particles,
        instance_seed,
        emitter_world,
        2,
        &Map::from_iter([
            ("use_normal".to_owned(), json!(true)),
            ("use_size".to_owned(), json!(true)),
            ("use_path".to_owned(), json!(false)),
            ("axis".to_owned(), json!("Z")),
            ("space".to_owned(), json!("LOCAL")),
            ("index_layer_name".to_owned(), json!("InstanceIndex")),
            ("value_layer_name".to_owned(), json!("InstanceValue")),
        ]),
    )?;
    assert_ordered_mesh_matches_blender(
        &direct_local_instance,
        &blender["instance_local_mesh"],
        "LOCAL Particle Instance",
    );
    let direct_local_points = direct_local_instance
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    let blender_local_points = blender_points(&blender["instance_local_points"])?;
    assert_eq!(direct_local_points.len(), blender_local_points.len());
    let local_error = max_corresponding_distance(&direct_local_points, &blender_local_points)?;
    assert!(
        local_error <= MODIFIER_POSITION_TOLERANCE,
        "LOCAL particle instance per-vertex error {local_error} exceeds {MODIFIER_POSITION_TOLERANCE}; Potter={direct_local_points:?}; Blender={blender_local_points:?}; particle={instance_particles:?}",
    );
    for (states_key, points_key, mesh_key, show_dead, show_unborn) in [
        (
            "unborn_particle_states",
            "unborn_instance_points",
            "unborn_instance_mesh",
            false,
            true,
        ),
        (
            "dead_particle_states",
            "dead_instance_points",
            "dead_instance_mesh",
            true,
            false,
        ),
    ] {
        let lifecycle_particles = injected_particles(&blender[states_key])?;
        let lifecycle_instance =
            potter_core::geom::modifiers::particle_instance::particle_instance(
                &instance_source,
                &lifecycle_particles,
                instance_seed,
                emitter_world,
                2,
                &Map::from_iter([
                    ("use_normal".to_owned(), json!(true)),
                    ("use_size".to_owned(), json!(true)),
                    ("show_alive".to_owned(), json!(false)),
                    ("show_dead".to_owned(), json!(show_dead)),
                    ("show_unborn".to_owned(), json!(show_unborn)),
                    ("axis".to_owned(), json!("Z")),
                    ("space".to_owned(), json!("WORLD")),
                    ("index_layer_name".to_owned(), json!("InstanceIndex")),
                    ("value_layer_name".to_owned(), json!("InstanceValue")),
                ]),
            )?;
        assert_ordered_mesh_matches_blender(&lifecycle_instance, &blender[mesh_key], states_key);
        let lifecycle_points = lifecycle_instance
            .vertices
            .iter()
            .map(|vertex| vertex.co)
            .collect::<Vec<_>>();
        let blender_lifecycle_points = blender_points(&blender[points_key])?;
        assert_eq!(
            lifecycle_points.len(),
            blender_lifecycle_points.len(),
            "{states_key}: visible copy count differs",
        );
        let lifecycle_error =
            max_corresponding_distance(&lifecycle_points, &blender_lifecycle_points)?;
        assert!(
            lifecycle_error <= MODIFIER_POSITION_TOLERANCE,
            "{states_key} per-vertex instance error {lifecycle_error} exceeds {MODIFIER_POSITION_TOLERANCE}; Potter={lifecycle_points:?}, Blender={blender_lifecycle_points:?}, states={lifecycle_particles:?}",
        );
    }
    let baseline_instance = instance_source
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    let nontrivial_instance_displacement = blender_instance_points
        .iter()
        .map(|point| {
            baseline_instance
                .iter()
                .map(|baseline| point.distance(*baseline))
                .fold(f64::INFINITY, f64::min)
        })
        .fold(0.0, f64::max);
    assert!(
        nontrivial_instance_displacement > 1.0e-6,
        "particle instance fixture must move instances off the source, observed {nontrivial_instance_displacement}",
    );
    let mut explode_source = instance_source.clone();
    let protect_weights = instance_source
        .vertices
        .iter()
        .zip([0.0, 0.35, 0.0, 0.85])
        .map(|(vertex, weight)| (format!("v{}", vertex.id), json!(weight)))
        .collect::<Map<_, _>>();
    explode_source.attributes.insert(
        "vertex_groups".to_owned(),
        json!({"Protect":protect_weights}),
    );
    let explode_params = Map::from_iter([
        ("use_edge_cut".to_owned(), json!(true)),
        ("use_size".to_owned(), json!(true)),
        ("show_alive".to_owned(), json!(true)),
        ("show_dead".to_owned(), json!(false)),
        ("show_unborn".to_owned(), json!(false)),
        ("vertex_group".to_owned(), json!("Protect")),
        ("protect".to_owned(), json!(0.6)),
        ("invert_vertex_group".to_owned(), json!(true)),
        ("particle_uv".to_owned(), json!("ParticleUV")),
    ]);
    let exploder_columns: [f64; 16] =
        serde_json::from_value(blender["exploder_world_matrix"].clone())?;
    let exploder_world = DMat4::from_cols_array(&exploder_columns);
    let explode_seed = blender["explode_seed"]
        .as_u64()
        .ok_or("Blender explode particle-system seed is missing")? as u32;
    let direct_explode = potter_core::geom::modifiers::explode::explode(
        &explode_source,
        &explode_particles,
        explode_seed,
        exploder_world,
        &explode_params,
        2.0,
    )?;
    assert_ordered_mesh_matches_blender(
        &direct_explode,
        &blender["explode_mesh"],
        "Explode with UV, groups, materials, and normals",
    );
    assert_eq!(
        direct_explode.faces.len(),
        blender["explode_face_count"],
        "Potter vertices={}, Blender vertices={}, particles={explode_particles:?}",
        direct_explode.vertices.len(),
        blender["explode_world_points"]
            .as_array()
            .map_or(0, Vec::len),
    );
    let direct_explode_world = direct_explode
        .vertices
        .iter()
        .map(|vertex| exploder_world.transform_point3(vertex.co))
        .collect::<Vec<_>>();
    let blender_explode_world = blender_points(&blender["explode_world_points"])?;
    assert_eq!(
        direct_explode_world.len(),
        blender_explode_world.len(),
        "Potter explode vertices={}, Blender vertices={}, particles={explode_particles:?}",
        direct_explode_world.len(),
        blender_explode_world.len(),
    );
    let explode_error = max_geometric_match_error(&direct_explode_world, &blender_explode_world)?;
    assert!(
        explode_error <= MODIFIER_POSITION_TOLERANCE,
        "explode per-vertex geometric error {explode_error} exceeds {MODIFIER_POSITION_TOLERANCE}",
    );
    let mut cube_particles = injected_particles(&blender["cube_explode_particle_states"])?;
    let cube_columns: [f64; 16] =
        serde_json::from_value(blender["cube_exploder_world_matrix"].clone())?;
    let cube_world = DMat4::from_cols_array(&cube_columns);
    let mut cube_source = potter_core::geom::Mesh::from_positions_and_faces(
        vec![
            DVec3::new(-1.0, -1.0, -1.0),
            DVec3::new(1.0, -1.0, -1.0),
            DVec3::new(1.0, 1.0, -1.0),
            DVec3::new(-1.0, 1.0, -1.0),
            DVec3::new(-1.0, -1.0, 1.0),
            DVec3::new(1.0, -1.0, 1.0),
            DVec3::new(1.0, 1.0, 1.0),
            DVec3::new(-1.0, 1.0, 1.0),
        ],
        vec![
            vec![0, 3, 2, 1],
            vec![4, 5, 6, 7],
            vec![0, 1, 5, 4],
            vec![1, 2, 6, 5],
            vec![2, 3, 7, 6],
            vec![3, 0, 4, 7],
        ],
    )?;
    add_corner_fixture_data(
        &mut cube_source,
        &blender["cube_source_normal_codes"],
        true,
        false,
    );
    let cube_local_births = blender_points(&blender["cube_birth_local"])?;
    assert_eq!(cube_local_births.len(), cube_particles.len());
    for (particle, birth_position) in cube_particles.iter_mut().zip(&cube_local_births) {
        particle.birth_position = cube_world.transform_point3(*birth_position).to_array();
    }
    let cube_positions = cube_source
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.co))
        .collect::<std::collections::BTreeMap<_, _>>();
    let cube_face_owners = cube_source
        .faces
        .iter()
        .map(|face| {
            let points = face
                .vertices
                .iter()
                .map(|id| {
                    cube_positions.get(id).copied().ok_or_else(|| {
                        std::io::Error::other("cube face references a missing vertex").into()
                    })
                })
                .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
            let center = points.iter().copied().sum::<DVec3>() / points.len() as f64;
            let mut nearest = 0;
            let mut nearest_distance = f32::INFINITY;
            for (index, position) in cube_local_births.iter().enumerate() {
                let distance = center.as_vec3().distance_squared(position.as_vec3());
                if index == 0 || distance < nearest_distance {
                    nearest = index;
                    nearest_distance = distance;
                }
            }
            Ok(nearest)
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    assert_eq!(cube_face_owners, [0, 0, 0, 0, 0, 1]);
    let blender_cube_face_owners: Vec<usize> =
        serde_json::from_value(blender["cube_face_owners"].clone())?;
    assert_eq!(
        cube_face_owners, blender_cube_face_owners,
        "local births={cube_local_births:?}, world={cube_world:?}",
    );
    let cube_vertex_indices = cube_source
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut cube_vertex_owners = vec![cube_particles.len(); cube_source.vertices.len()];
    for (face, owner) in cube_source.faces.iter().zip(&cube_face_owners) {
        for vertex in &face.vertices {
            cube_vertex_owners[cube_vertex_indices[vertex]] = *owner;
        }
    }
    let cube_masks = cube_source
        .faces
        .iter()
        .map(|face| {
            let mut mask = 0_usize;
            for index in 0..face.vertices.len() {
                let first = face.vertices[index];
                let second = face.vertices[(index + 1) % face.vertices.len()];
                if cube_vertex_owners[cube_vertex_indices[&first]]
                    != cube_vertex_owners[cube_vertex_indices[&second]]
                {
                    mask |= 1_usize << index;
                }
            }
            mask
        })
        .collect::<Vec<_>>();
    let blender_cube_masks: Vec<usize> = serde_json::from_value(blender["cube_cut_masks"].clone())?;
    assert_eq!(cube_masks, [10, 5, 5, 0, 5, 0]);
    assert_eq!(
        cube_masks, blender_cube_masks,
        "face split masks must match Blender's particle-owner classification",
    );
    let cube_seed = blender["cube_explode_seed"]
        .as_u64()
        .ok_or("Blender cube explode particle-system seed is missing")? as u32;
    let cube_explode_params = Map::from_iter([
        ("use_edge_cut".to_owned(), json!(true)),
        ("use_size".to_owned(), json!(true)),
        ("show_alive".to_owned(), json!(true)),
        ("show_dead".to_owned(), json!(false)),
        ("show_unborn".to_owned(), json!(false)),
        ("particle_uv".to_owned(), json!("ParticleUV")),
    ]);
    let cube_explode = potter_core::geom::modifiers::explode::explode(
        &cube_source,
        &cube_particles,
        cube_seed,
        cube_world,
        &cube_explode_params,
        2.0,
    )?;
    assert_ordered_mesh_matches_blender(
        &cube_explode,
        &blender["cube_explode_mesh"],
        "edge-cut cube Explode",
    );
    assert_eq!(cube_explode.faces.len(), 10);
    assert_eq!(
        cube_explode.faces.len(),
        blender["cube_explode_face_count"],
        "edge-cut cube face count, particles={cube_particles:?}, local births={cube_local_births:?}, owners={cube_face_owners:?}, masks={cube_masks:?}",
    );
    let cube_explode_world = cube_explode
        .vertices
        .iter()
        .map(|vertex| cube_world.transform_point3(vertex.co))
        .collect::<Vec<_>>();
    let blender_cube_points = blender_points(&blender["cube_explode_world_points"])?;
    assert_eq!(cube_explode.vertices.len(), 16);
    assert_eq!(
        cube_explode_world.len(),
        blender_cube_points.len(),
        "edge-cut cube vertices Potter={}, Blender={}, owners={cube_face_owners:?}, masks={cube_masks:?}",
        cube_explode_world.len(),
        blender_cube_points.len(),
    );
    let cube_error = max_geometric_match_error(&cube_explode_world, &blender_cube_points)?;
    assert!(
        cube_error <= MODIFIER_POSITION_TOLERANCE,
        "edge-cut cube vertex error {cube_error} exceeds {MODIFIER_POSITION_TOLERANCE}",
    );

    let scene_path = root.join("potter-scene");
    let batch_path = root.join("operations.json");
    initialize(&scene_path)?;
    apply(
        &scene_path,
        &batch_path,
        &json!([
            {"op":"node.create","id":"a_emitter","kind":"plane","params":{},"transform":{"translation":[3.0,-1.0,0.5],"scale":[1.3,0.8,1.1]}},
            {"op":"physics.particle_emitter.create","target":{"id":"a_emitter"},"settings":{"emit_from":"VERT","count":3,"frame_start":1.0,"frame_end":1.0,"lifetime":12.0,"physics_type":"NEWTON","normal_factor":0.8,"particle_size":0.7,"seed":3}},
            {"op":"node.create","id":"b_instance","kind":"plane","params":{}},
            {"op":"modifier.create","target":{"id":"b_instance"},"id":"particle_copy","type":"particle_instance","params":{"object":"a_emitter","particle_system_index":1,"use_normal":true,"use_children":false,"use_size":true,"show_alive":true,"show_dead":false,"show_unborn":false,"position":0,"random_position":0.0,"axis":"X","space":"WORLD","use_path":false}},
            {"op":"node.create","id":"c_exploder","kind":"box","params":{}},
            {"op":"physics.particle_emitter.create","target":{"id":"c_exploder"},"settings":{"emit_from":"VERT","count":2,"frame_start":1.0,"frame_end":1.0,"lifetime":12.0,"physics_type":"NEWTON","normal_factor":1.1,"seed":3}},
            {"op":"modifier.create","target":{"id":"c_exploder"},"id":"explode_faces","type":"explode","params":{"use_edge_cut":true,"show_alive":true,"show_dead":false,"show_unborn":false,"use_size":false,"protect":0.0}}
        ]),
        0,
    )?;
    let instance_output = run(
        pot()
            .arg("inspect")
            .arg(&scene_path)
            .arg("--id")
            .arg("b_instance")
            .arg("--frame")
            .arg("1")
            .arg("--json"),
        "particle instance inspection",
    )?;
    let explode_output = run(
        pot()
            .arg("inspect")
            .arg(&scene_path)
            .arg("--id")
            .arg("c_exploder")
            .arg("--frame")
            .arg("1")
            .arg("--json"),
        "explode inspection",
    )?;
    let instance_json: Value = serde_json::from_slice(&instance_output.stdout)?;
    let explode_json: Value = serde_json::from_slice(&explode_output.stdout)?;
    let item = &instance_json["result"]["items"][0];
    let minimum: [f64; 3] = serde_json::from_value(item["bounds"]["min"].clone())?;
    let maximum: [f64; 3] = serde_json::from_value(item["bounds"]["max"].clone())?;
    let extent = DVec3::from_array(maximum).distance(DVec3::from_array(minimum));
    assert!(
        extent > 1.0,
        "CLI particle instance should contain normal-sized, spatially separated copies",
    );
    let exploded_faces = explode_json["result"]["items"][0]["evaluated_geometry"]["face_count"]
        .as_u64()
        .ok_or("CLI explode face count is missing")?;
    assert!(
        exploded_faces > 6,
        "CLI explode should split the six source cube faces: {exploded_faces}",
    );
    let scene_document: Value = serde_json::from_slice(&fs::read(scene_path.join("scene.json"))?)?;
    let revision = scene_document["revision"]
        .as_u64()
        .ok_or("scene revision is missing")?;
    apply(
        &scene_path,
        &batch_path,
        &json!([{"op":"modifier.apply","target":{"id":"c_exploder"},"modifier_id":"explode_faces"}]),
        revision,
    )?;
    let applied_document: Value =
        serde_json::from_slice(&fs::read(scene_path.join("scene.json"))?)?;
    let exploder = &applied_document["nodes"]["c_exploder"];
    let stack_after_apply = exploder["modifiers"]
        .as_array()
        .ok_or("applied Explode modifier stack is missing")?;
    assert!(
        stack_after_apply
            .iter()
            .any(|modifier| modifier["type"] == "particle_system")
    );
    assert!(
        stack_after_apply
            .iter()
            .all(|modifier| modifier["id"] != "explode_faces")
    );
    let data_id = exploder["data"]
        .as_str()
        .ok_or("Explode mesh data ID is missing")?;
    let persisted_mesh = &applied_document["data_blocks"][data_id]["mesh"];
    let persisted_vertices = persisted_mesh["vertices"]
        .as_array()
        .ok_or("applied Explode vertices are missing")?;
    let persisted_faces = persisted_mesh["faces"]
        .as_array()
        .ok_or("applied Explode faces are missing")?;
    assert_eq!(persisted_faces.len() as u64, exploded_faces);
    let applied_inspection = inspect(&scene_path, "c_exploder", "1")?;
    let applied_item = &applied_inspection["result"]["items"][0];
    let applied_stack = applied_item["modifiers"]
        .as_array()
        .ok_or("inspected Explode modifier stack is missing")?;
    assert!(
        applied_stack
            .iter()
            .any(|modifier| modifier["type"] == "particle_system")
    );
    assert!(
        applied_stack
            .iter()
            .all(|modifier| modifier["id"] != "explode_faces")
    );
    assert_eq!(
        applied_item["evaluated_geometry"]["face_count"],
        exploded_faces
    );
    let persisted_positions = Value::Array(
        persisted_vertices
            .iter()
            .map(|vertex| vertex["co"].clone())
            .collect(),
    );
    assert_numeric_rows(
        &persisted_positions,
        &applied_item["evaluated_geometry"]["positions"],
        "persisted Explode mesh positions",
    );
    let vertex_ids = persisted_vertices
        .iter()
        .map(|vertex| vertex["id"].as_u64().ok_or("Explode vertex ID is missing"))
        .collect::<Result<Vec<_>, _>>()?;
    let mut expected_faces = Vec::with_capacity(persisted_faces.len());
    for face in persisted_faces {
        let vertices = face["v"]
            .as_array()
            .ok_or("applied Explode face vertices are missing")?;
        let mut polygon = Vec::with_capacity(vertices.len());
        for vertex in vertices {
            let id = vertex.as_u64().ok_or("Explode face vertex ID is missing")?;
            let index = vertex_ids
                .iter()
                .position(|candidate| *candidate == id)
                .ok_or("Explode face references a missing vertex")?;
            polygon.push(index);
        }
        expected_faces.push(polygon);
    }
    let expected_faces = json!(expected_faces);
    let exported = root.join("explode_cli_roundtrip.blend");
    let blender_path = blender_executable().ok_or("Blender is unavailable")?;
    let mut export = pot();
    export
        .arg("export")
        .arg(&scene_path)
        .args(["--format", "blend", "--out"])
        .arg(&exported)
        .arg("--blender")
        .arg(&blender_path)
        .arg("--json");
    run(&mut export, "exporting applied Explode mesh")?;
    let reopen_script = root.join("reopen_explode_cli.py");
    let reopened_path = root.join("reopened_explode_cli.json");
    fs::write(
        &reopen_script,
        r#"
import bpy, json, sys
output = sys.argv[sys.argv.index('--') + 1]
obj = bpy.data.objects['c_exploder']
result = {
    'modifier_types':[modifier.type for modifier in obj.modifiers],
    'positions':[list(vertex.co) for vertex in obj.data.vertices],
    'faces':[list(polygon.vertices) for polygon in obj.data.polygons],
}
with open(output, 'w', encoding='utf-8') as handle:
    json.dump(result, handle)
"#,
    )?;
    let mut reopen = Command::new(&blender_path);
    reopen
        .args(["--background", "--disable-autoexec"])
        .arg(&exported)
        .args(["--python-exit-code", "1", "--python"])
        .arg(&reopen_script)
        .arg("--")
        .arg(&reopened_path);
    run(&mut reopen, "reopening applied Explode blend")?;
    let reopened: Value = serde_json::from_slice(&fs::read(reopened_path)?)?;
    let reopened_modifier_types = reopened["modifier_types"]
        .as_array()
        .ok_or("reopened Explode modifier types are missing")?;
    assert!(
        reopened_modifier_types
            .iter()
            .any(|kind| kind.as_str() == Some("PARTICLE_SYSTEM"))
    );
    assert!(
        reopened_modifier_types
            .iter()
            .all(|kind| kind.as_str() != Some("EXPLODE"))
    );
    assert_numeric_rows(
        &reopened["positions"],
        &persisted_positions,
        "reopened applied Explode mesh",
    );
    assert_eq!(reopened["faces"], expected_faces);
    Ok(())
}

#[test]
fn particle_instance_cli_frames_apply_export_reimport_preserves_track_axis()
-> Result<(), Box<dyn Error>> {
    let Some(blender_path) = blender_executable() else {
        eprintln!("skipping particle CLI round-trip; Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let fixture_script = root.join("particle_cli_fixture.py");
    let expected_path = root.join("particle_cli_expected.json");
    fs::write(
        &fixture_script,
        r#"
import bpy, json, os, sys
from mathutils import Vector
if tuple(bpy.app.version[:3]) != (5,2,2):
    raise RuntimeError('particle CLI parity requires Blender 5.2.2')
root = sys.argv[sys.argv.index('--') + 1]
scene = bpy.context.scene
scene.use_gravity = False
scene.frame_start = 1
scene.frame_end = 3
scene.frame_set(1)

def add_corner_data(mesh):
    for index in range(3):
        mesh.materials.append(bpy.data.materials.new('CLI Material %d' % index))
    for polygon in mesh.polygons:
        polygon.material_index = polygon.index % 3
    source_uv = mesh.uv_layers.new(name='SourceUV')
    particle_uv = mesh.uv_layers.new(name='ParticleUV')
    for polygon in mesh.polygons:
        for corner, loop_index in enumerate(range(polygon.loop_start, polygon.loop_start + polygon.loop_total)):
            uv = (0.1 + 0.17 * polygon.index + 0.11 * corner,
                  0.2 + 0.13 * polygon.index + 0.07 * corner)
            source_uv.data[loop_index].uv = uv
            particle_uv.data[loop_index].uv = (uv[0] + 0.031, uv[1] + 0.047)
    mesh.normals_split_custom_set([
        Vector((0.13 * ((loop.index % 3) - 1), 0.09 * ((loop.index % 4) - 1.5), 1.0)).normalized()
        for loop in mesh.loops
    ])
    mesh.update()
def corner_normal_codes(mesh):
    attribute = mesh.attributes.get('custom_normal')
    if attribute is None:
        return None
    if attribute.domain != 'CORNER' or attribute.data_type != 'INT16_2D':
        raise RuntimeError('custom normals must be Blender INT16_2D CORNER codes')
    return [None if list(item.value) == [0,0] else [int(value) for value in item.value]
            for item in attribute.data]


def mesh_snapshot(mesh):
    return {
        'positions':[list(vertex.co) for vertex in mesh.vertices],
        'faces':[list(polygon.vertices) for polygon in mesh.polygons],
        'materials':[int(polygon.material_index) for polygon in mesh.polygons],
        'uv_layers':{layer.name:[list(item.uv) for item in layer.data] for layer in mesh.uv_layers},
        'custom_normal_codes':corner_normal_codes(mesh),
        'color_layer_names':[layer.name for layer in mesh.color_attributes if layer.domain == 'CORNER'],
        'color_attributes':{
            layer.name:{'domain':layer.domain, 'data_type':layer.data_type,
                        'values':[list(item.color) for item in layer.data]}
            for layer in mesh.color_attributes if layer.domain == 'CORNER'
        },
        'vertex_groups':{},
    }

emitter_mesh = bpy.data.meshes.new('ParticleEmitterMesh')
emitter_mesh.from_pydata([(-1,-1,0),(1,-1,0),(1,1,0),(-1,1,0)], [], [(0,1,2,3)])
emitter = bpy.data.objects.new('ParticleEmitter', emitter_mesh)
scene.collection.objects.link(emitter)
emitter.location = (1.25,-0.75,0.5)
for selected in list(bpy.context.selected_objects):
    selected.select_set(False)
emitter.select_set(True)
bpy.context.view_layer.objects.active = emitter
bpy.ops.object.particle_system_add()
system = emitter.particle_systems[0]
system.seed = 7
system.child_seed = 67
settings = system.settings
settings.count = 4
settings.emit_from = 'VERT'
settings.frame_start = 1.0
settings.frame_end = 2.0
settings.lifetime = 8.0
settings.physics_type = 'NEWTON'
settings.normal_factor = 0.8
settings.factor_random = 0.0
settings.particle_size = 0.65
settings.size_random = 0.0
settings.child_type = 'INTERPOLATED'
settings.child_percent = 25
settings.rendered_child_count = 3
settings.child_radius = 0.25
settings.child_size = 0.8

instance_mesh = bpy.data.meshes.new('ParticleInstancerMesh')
instance_mesh.from_pydata([(-0.6,-0.2,0),(0.7,-0.15,0.1),(0.25,0.65,0.35),(-0.35,0.4,0.55)], [], [(0,1,2,3)])
add_corner_data(instance_mesh)
instancer = bpy.data.objects.new('ParticleInstancer', instance_mesh)
scene.collection.objects.link(instancer)
instancer.track_axis = 'POS_Z'
modifier = instancer.modifiers.new('Particle Instance', 'PARTICLE_INSTANCE')
modifier.object = emitter
modifier.particle_system_index = 1
modifier.axis = 'Z'
modifier.space = 'WORLD'
modifier.use_normal = True
modifier.use_children = True
modifier.particle_amount = 0.5
modifier.particle_offset = 0.25
modifier.use_path = False
modifier.use_preserve_shape = False
modifier.use_size = True
modifier.position = 0.35
modifier.random_position = 0.0
modifier.rotation = 0.2
modifier.random_rotation = 0.0
modifier.show_alive = True
modifier.show_dead = False
modifier.show_unborn = False
modifier.index_layer_name = 'InstanceIndex'
modifier.value_layer_name = 'InstanceValue'

def evaluated_mesh(obj, frame):
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    try:
        return mesh_snapshot(mesh)
    finally:
        evaluated.to_mesh_clear()

frames = {str(frame):evaluated_mesh(instancer, frame) for frame in (1,2)}
scene.frame_set(1)
bpy.context.view_layer.update()
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, 'particle_cli_source.blend'))
with open(os.path.join(root, 'particle_cli_expected.json'), 'w', encoding='utf-8') as output:
    json.dump({
        'frames':frames,
        'track_axis':instancer.track_axis,
        'particle_seed':system.seed,
        'child_seed':system.child_seed,
        'child_percent':settings.child_percent,
        'rendered_child_count':settings.rendered_child_count,
    }, output)
"#,
    )?;
    let mut create_fixture = Command::new(&blender_path);
    create_fixture
        .args([
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
        ])
        .arg(&fixture_script)
        .arg("--")
        .arg(root);
    let blender_output = run(&mut create_fixture, "Blender particle CLI fixture")?;
    let blender: Value = serde_json::from_slice(&fs::read(&expected_path)?)?;
    assert_eq!(blender["track_axis"], "POS_Z");
    assert_eq!(blender["child_seed"], 67);
    assert_eq!(blender["child_percent"], 25);
    assert_eq!(blender["rendered_child_count"], 3);
    let blender_frame_one = &blender["frames"]["1"];
    let blender_frame_two = &blender["frames"]["2"];
    assert_ne!(
        blender_frame_one["positions"],
        blender_frame_two["positions"]
    );
    assert!(
        blender_frame_two["positions"]
            .as_array()
            .is_some_and(|positions| positions.len() > 4),
        "Blender fixture must evaluate parent and child instances",
    );

    let project = root.join("particle-project");
    initialize(&project)?;
    let mut import = pot();
    import
        .arg("import")
        .arg(&project)
        .arg("--file")
        .arg(root.join("particle_cli_source.blend"))
        .args([
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "0",
            "--blender",
        ])
        .arg(&blender_path)
        .arg("--json");
    let imported: Value =
        serde_json::from_slice(&run(&mut import, "importing particle fixture")?.stdout)?;
    assert_eq!(imported["result"]["losses"], json!([]), "{imported}");

    let document: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let node_id_by_name = |name: &str| -> Result<String, Box<dyn Error>> {
        document["nodes"]
            .as_object()
            .and_then(|nodes| {
                nodes
                    .iter()
                    .find(|(_, node)| node["name"] == name)
                    .map(|(id, _)| id.clone())
            })
            .ok_or_else(|| format!("imported Blender object `{name}` is missing").into())
    };
    let instancer_id = node_id_by_name("ParticleInstancer")?;
    let emitter_id = node_id_by_name("ParticleEmitter")?;
    let instancer_node = &document["nodes"][instancer_id.as_str()];
    assert_eq!(instancer_node["track_axis"], "POS_Z",);
    let emitter_node = &document["nodes"][emitter_id.as_str()];
    let particle_settings = emitter_node["properties"]["physics_particle_systems"]
        .as_object()
        .and_then(|systems| {
            systems
                .values()
                .find(|settings| settings["seed"] == blender["particle_seed"])
        })
        .ok_or("imported particle-system settings are not serialized")?;
    assert_eq!(particle_settings["child_seed"], blender["child_seed"]);
    assert_eq!(particle_settings["child_percent"], blender["child_percent"]);
    assert_eq!(
        particle_settings["rendered_child_count"],
        blender["rendered_child_count"]
    );
    let modifier = instancer_node["modifiers"]
        .as_array()
        .and_then(|modifiers| {
            modifiers
                .iter()
                .find(|modifier| modifier["type"] == "particle_instance")
        })
        .ok_or("imported particle-instance modifier is missing")?;
    let modifier_id = modifier["id"]
        .as_str()
        .ok_or("particle modifier ID is missing")?;
    let typed_document: potter_core::model::SceneDoc = serde_json::from_value(document.clone())?;
    assert_snapshot_mesh_matches_blender(
        &typed_document,
        &instancer_id,
        1.0,
        blender_frame_one,
        "native child-instance frame 1 corner data",
    )?;
    assert_snapshot_mesh_matches_blender(
        &typed_document,
        &instancer_id,
        2.0,
        blender_frame_two,
        "native child-instance frame 2 corner data",
    )?;
    let assert_cli_mesh = |actual: &Value, expected: &Value, context: &str| {
        let geometry = &actual["result"]["items"][0]["evaluated_geometry"];
        assert_numeric_rows(
            &geometry["positions"],
            &expected["positions"],
            &format!("{context} positions"),
        );
        assert_eq!(
            geometry["faces"], expected["faces"],
            "{context} ordered faces"
        );
    };
    let frame_one = inspect(&project, &instancer_id, "1")?;
    let frame_two = inspect(&project, &instancer_id, "2")?;
    assert_cli_mesh(&frame_one, blender_frame_one, "Potter CLI frame 1");
    assert_cli_mesh(&frame_two, blender_frame_two, "Potter CLI frame 2");
    assert_ne!(
        frame_one["result"]["items"][0]["evaluated_geometry"]["positions"],
        frame_two["result"]["items"][0]["evaluated_geometry"]["positions"],
        "native particle evaluation must change across frames",
    );

    let revision = document["revision"]
        .as_u64()
        .ok_or("imported scene revision is missing")?;
    let batch = root.join("apply_particle_instance.json");
    apply(
        &project,
        &batch,
        &json!([{"op":"modifier.apply","target":{"id":instancer_id},"modifier_id":modifier_id}]),
        revision,
    )?;
    let applied_document: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let typed_applied_document: potter_core::model::SceneDoc =
        serde_json::from_value(applied_document.clone())?;
    assert_snapshot_mesh_matches_blender(
        &typed_applied_document,
        &instancer_id,
        1.0,
        blender_frame_one,
        "applied particle corner data",
    )?;
    assert_eq!(
        applied_document["nodes"][instancer_id.as_str()]["track_axis"],
        "POS_Z",
    );
    assert!(
        applied_document["nodes"][instancer_id.as_str()]["modifiers"]
            .as_array()
            .is_some_and(|modifiers| modifiers
                .iter()
                .all(|modifier| modifier["id"] != modifier_id)),
        "applying the particle modifier must persist its evaluated mesh and remove the stack entry",
    );
    let applied = inspect(&project, &instancer_id, "1")?;
    assert_cli_mesh(&applied, blender_frame_one, "applied particle mesh");

    let exported = root.join("particle_cli_roundtrip.blend");
    let mut export = pot();
    export
        .arg("export")
        .arg(&project)
        .args(["--format", "blend", "--out"])
        .arg(&exported)
        .arg("--blender")
        .arg(&blender_path)
        .arg("--json");
    run(&mut export, "exporting applied particle mesh")?;
    let reopen_script = root.join("reopen_particle_cli.py");
    let reopened_path = root.join("reopened_particle_cli.json");
    fs::write(
        &reopen_script,
        r#"
import bpy, json, sys
output = sys.argv[sys.argv.index('--') + 1]
obj = bpy.data.objects['ParticleInstancer']
scene = bpy.context.scene
scene.frame_set(1)
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
evaluated = obj.evaluated_get(depsgraph)
mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
try:
    result = {
        'track_axis':obj.track_axis,
        'modifier_count':len(obj.modifiers),
        'positions':[list(vertex.co) for vertex in mesh.vertices],
        'faces':[list(polygon.vertices) for polygon in mesh.polygons],
    }
finally:
    evaluated.to_mesh_clear()
with open(output, 'w', encoding='utf-8') as handle:
    json.dump(result, handle)
"#,
    )?;
    let mut reopen = Command::new(&blender_path);
    reopen
        .args(["--background", "--disable-autoexec"])
        .arg(&exported)
        .args(["--python-exit-code", "1", "--python"])
        .arg(&reopen_script)
        .arg("--")
        .arg(&reopened_path);
    run(&mut reopen, "reopening exported particle blend")?;
    let reopened: Value = serde_json::from_slice(&fs::read(&reopened_path)?)?;
    assert_eq!(reopened["track_axis"], "POS_Z");
    assert_eq!(reopened["modifier_count"], 0);
    assert_numeric_rows(
        &reopened["positions"],
        &blender_frame_one["positions"],
        "reopened Blender mesh",
    );
    assert_eq!(reopened["faces"], blender_frame_one["faces"]);

    let reimport_project = root.join("particle-reimport-project");
    initialize(&reimport_project)?;
    let mut reimport = pot();
    reimport
        .arg("import")
        .arg(&reimport_project)
        .arg("--file")
        .arg(&exported)
        .args([
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "0",
            "--blender",
        ])
        .arg(&blender_path)
        .arg("--json");
    let reimported: Value =
        serde_json::from_slice(&run(&mut reimport, "reimporting exported particle blend")?.stdout)?;
    assert_eq!(reimported["result"]["losses"], json!([]), "{reimported}");
    let reimported_document: Value =
        serde_json::from_slice(&fs::read(reimport_project.join("scene.json"))?)?;
    let typed_reimported_document: potter_core::model::SceneDoc =
        serde_json::from_value(reimported_document.clone())?;
    let reimported_instancer_id = reimported_document["nodes"]
        .as_object()
        .and_then(|nodes| {
            nodes
                .iter()
                .find(|(_, node)| node["name"] == "ParticleInstancer")
                .map(|(id, _)| id.clone())
        })
        .ok_or("reimported ParticleInstancer is missing")?;
    assert_eq!(
        reimported_document["nodes"][reimported_instancer_id.as_str()]["track_axis"],
        "POS_Z",
    );
    assert!(
        reimported_document["nodes"][reimported_instancer_id.as_str()]["modifiers"]
            .as_array()
            .is_some_and(Vec::is_empty),
        "reimported applied particle object must stay modifier-free",
    );
    let after_reimport = inspect(&reimport_project, &reimported_instancer_id, "1")?;
    assert_snapshot_mesh_matches_blender(
        &typed_reimported_document,
        &reimported_instancer_id,
        1.0,
        blender_frame_one,
        "reimported applied corner data",
    )?;
    assert_cli_mesh(
        &after_reimport,
        blender_frame_one,
        "Potter reimported applied mesh",
    );
    let _ = blender_output;
    Ok(())
}

#[test]
fn particle_instance_baked_path_velocity_matches_blender() -> Result<(), Box<dyn Error>> {
    let Some(blender_path) = blender_executable() else {
        eprintln!("skipping baked particle path parity; Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let fixture_script = root.join("particle_baked_path.py");
    let expected_path = root.join("particle_baked_path.json");
    fs::write(
        &fixture_script,
        r#"
import bpy, json, math, os, sys
if tuple(bpy.app.version[:3]) != (5,2,2):
    raise RuntimeError('baked particle path parity requires Blender 5.2.2')
root = sys.argv[sys.argv.index('--') + 1]
scene = bpy.context.scene
scene.use_gravity = False
scene.frame_start = 1
scene.frame_end = 10
scene.frame_set(1)

emitter_mesh = bpy.data.meshes.new('BakedPathEmitterMesh')
emitter_mesh.from_pydata([(0,0,0),(1,0,0),(0,1,0)], [], [(0,1,2)])
emitter = bpy.data.objects.new('BakedPathEmitter', emitter_mesh)
scene.collection.objects.link(emitter)
for selected in list(bpy.context.selected_objects):
    selected.select_set(False)
emitter.select_set(True)
bpy.context.view_layer.objects.active = emitter
bpy.ops.object.particle_system_add()
system = emitter.particle_systems[0]
system.seed = 7
settings = system.settings
settings.count = 1
settings.frame_start = 1.25
settings.frame_end = 1.25
settings.lifetime = 2.0
settings.emit_from = 'FACE'
settings.physics_type = 'NEWTON'
settings.normal_factor = 1.25
settings.factor_random = 0.0
settings.particle_size = 1.0
settings.size_random = 0.0
settings.render_type = 'NONE'
settings.child_type = 'NONE'
cache = system.point_cache
cache.frame_start = 1
cache.frame_end = 10
cache.frame_step = 1
with bpy.context.temp_override(object=emitter, active_object=emitter, point_cache=cache):
    result = bpy.ops.ptcache.bake(bake=True)
if 'FINISHED' not in result or not cache.is_baked:
    raise RuntimeError('failed to bake the particle point cache')

instancer_mesh = bpy.data.meshes.new('BakedPathInstancerMesh')
instancer_mesh.from_pydata([(0,0,0)], [], [])
instancer = bpy.data.objects.new('BakedPathInstancer', instancer_mesh)
scene.collection.objects.link(instancer)
instancer.track_axis = 'POS_Z'
modifier = instancer.modifiers.new('Baked Path Instance', 'PARTICLE_INSTANCE')
modifier.object = emitter
modifier.particle_system_index = 1
modifier.axis = 'Z'
modifier.space = 'WORLD'
modifier.use_normal = True
modifier.use_children = False
modifier.use_size = False
modifier.use_path = True
modifier.use_preserve_shape = True
modifier.position = 0.25
modifier.random_position = 0.0
modifier.rotation = 0.0
modifier.random_rotation = 0.0
modifier.particle_amount = 1.0
modifier.particle_offset = 0.0
modifier.show_alive = True
modifier.show_dead = False
modifier.show_unborn = False

def evaluate_mesh(frame):
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = instancer.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    try:
        return {
            'positions':[list(vertex.co) for vertex in mesh.vertices],
            'faces':[list(polygon.vertices) for polygon in mesh.polygons],
        }
    finally:
        evaluated.to_mesh_clear()

scene.frame_set(1)
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
particle = emitter.evaluated_get(depsgraph).particle_systems[0].particles[0]
birth_time = float(particle.birth_time)
die_time = float(particle.die_time)
duration = die_time - birth_time
time_tweak = float(getattr(settings, 'time_tweak', 1.0))
particle_keys = list(getattr(particle, 'particle_keys', ()))
path_keys = []
path_source = 'particle_keys' if particle_keys else 'point_cache'
if particle_keys:
    time_start = float(particle_keys[0].time)
    time_span = float(particle_keys[-1].time) - time_start
    for key in particle_keys:
        path_keys.append({
            'time':((float(key.time) - time_start) / time_span if time_span else 0.0),
            'position':[float(value) for value in key.location],
            'velocity':[float(value) * time_span * 0.04 * time_tweak for value in key.velocity],
        })
else:
    sample_times = [
        float(frame)
        for frame in range(cache.frame_start, cache.frame_end + 1, cache.frame_step)
        if birth_time <= frame <= die_time
    ]
    for endpoint in (birth_time, die_time):
        if cache.frame_start <= endpoint <= cache.frame_end and not any(
                math.isclose(endpoint, sample_time, rel_tol=0.0, abs_tol=1e-6)
                for sample_time in sample_times):
            sample_times.append(endpoint)
    sample_times.sort()
    for sample_time in sample_times:
        frame = math.floor(sample_time)
        scene.frame_set(frame, subframe=float(sample_time - frame))
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        particle = emitter.evaluated_get(depsgraph).particle_systems[0].particles[0]
        path_keys.append({
            'time':(sample_time - birth_time) / duration,
            'position':[float(value) for value in particle.location],
            'velocity':[float(value) * duration * 0.04 * time_tweak for value in particle.velocity],
        })
scene.frame_set(2)
bpy.context.view_layer.update()
with open(os.path.join(root, 'particle_baked_path.json'), 'w', encoding='utf-8') as output:
    json.dump({
        'path_source':path_source,
        'birth_time':birth_time,
        'die_time':die_time,
        'time_tweak':time_tweak,
        'cache':{'frame_start':cache.frame_start,'frame_end':cache.frame_end,'frame_step':cache.frame_step,'is_baked':cache.is_baked},
        'path_keys':path_keys,
        'mesh':evaluate_mesh(2),
    }, output)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, 'particle_baked_path.blend'))
"#,
    )?;
    let mut create_fixture = Command::new(&blender_path);
    create_fixture
        .args([
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
        ])
        .arg(&fixture_script)
        .arg("--")
        .arg(root);
    run(&mut create_fixture, "Blender baked particle path fixture")?;
    let blender: Value = serde_json::from_slice(&fs::read(&expected_path)?)?;
    assert_eq!(blender["cache"]["is_baked"], true);
    assert_eq!(blender["path_source"], "point_cache");
    let expected_keys = blender["path_keys"]
        .as_array()
        .ok_or("Blender baked particle path keys are missing")?;
    assert_eq!(
        expected_keys.len(),
        4,
        "fixture includes birth/death endpoints and in-lifetime cache samples"
    );

    let project = root.join("particle-baked-path-project");
    initialize(&project)?;
    let mut import = pot();
    import
        .arg("import")
        .arg(&project)
        .arg("--file")
        .arg(root.join("particle_baked_path.blend"))
        .args([
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "0",
            "--blender",
        ])
        .arg(&blender_path)
        .arg("--json");
    let imported: Value =
        serde_json::from_slice(&run(&mut import, "importing baked particle path fixture")?.stdout)?;
    assert_eq!(imported["result"]["losses"], json!([]), "{imported}");
    let document: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let node_id_by_name = |name: &str| -> Result<String, Box<dyn Error>> {
        document["nodes"]
            .as_object()
            .and_then(|nodes| {
                nodes
                    .iter()
                    .find(|(_, node)| node["name"] == name)
                    .map(|(id, _)| id.clone())
            })
            .ok_or_else(|| format!("imported Blender object `{name}` is missing").into())
    };
    let emitter_id = node_id_by_name("BakedPathEmitter")?;
    let instancer_id = node_id_by_name("BakedPathInstancer")?;
    let emitter = &document["nodes"][emitter_id.as_str()];
    let system_settings = emitter["properties"]["physics_particle_systems"]
        .as_object()
        .and_then(|systems| systems.values().find(|settings| settings["seed"] == 7))
        .ok_or("imported baked particle system is missing")?;
    let path_arrays = system_settings["_potter_path_keys"]
        .as_array()
        .ok_or("baked path keys were not captured by the bridge")?;
    assert_eq!(path_arrays.len(), 1);
    let actual_keys = path_arrays[0]
        .as_array()
        .ok_or("imported baked path is not an ordered key list")?;
    assert_eq!(actual_keys.len(), expected_keys.len());
    for (index, (actual, expected)) in actual_keys.iter().zip(expected_keys).enumerate() {
        assert_numeric_rows(
            &json!([[actual["time"]]]),
            &json!([[expected["time"]]]),
            &format!("baked path key {index} time"),
        );
        assert_numeric_rows(
            &json!([actual["position"]]),
            &json!([expected["position"]]),
            &format!("baked path key {index} position"),
        );
        assert_numeric_rows(
            &json!([actual["velocity"]]),
            &json!([expected["velocity"]]),
            &format!("baked path key {index} normalized velocity"),
        );
    }
    let inspection = inspect(&project, &instancer_id, "2")?;
    let geometry = &inspection["result"]["items"][0]["evaluated_geometry"];
    assert_numeric_rows(
        &geometry["positions"],
        &blender["mesh"]["positions"],
        "baked path CLI particle position",
    );
    assert_eq!(geometry["faces"], blender["mesh"]["faces"]);
    Ok(())
}
#[test]
fn particle_instance_hair_lifecycle_matches_blender() -> Result<(), Box<dyn Error>> {
    let Some(blender_path) = blender_executable() else {
        eprintln!("skipping Hair lifecycle Blender parity; Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let fixture_script = root.join("particle_hair_lifecycle.py");
    let expected_path = root.join("particle_hair_lifecycle.json");
    fs::write(
        &fixture_script,
        r#"
import bpy, json, os, sys
if tuple(bpy.app.version[:3]) != (5,2,2):
    raise RuntimeError('Hair lifecycle parity requires Blender 5.2.2')
root = sys.argv[sys.argv.index('--') + 1]
scene = bpy.context.scene
scene.use_gravity = False
scene.frame_start = 1
scene.frame_end = 10
scene.frame_set(1)

emitter_mesh = bpy.data.meshes.new('HairLifecycleEmitterMesh')
emitter_mesh.from_pydata([(-1,-1,0),(1,-1,0),(0,1,0)], [], [(0,1,2)])
emitter = bpy.data.objects.new('HairLifecycleEmitter', emitter_mesh)
scene.collection.objects.link(emitter)
for selected in list(bpy.context.selected_objects):
    selected.select_set(False)
emitter.select_set(True)
bpy.context.view_layer.objects.active = emitter
bpy.ops.object.particle_system_add()
system = emitter.particle_systems[0]
system.seed = 29
settings = system.settings
settings.type = 'HAIR'
settings.count = 4
settings.frame_start = 1.0
settings.frame_end = 10.0
settings.hair_length = 1.0
settings.hair_step = 2
settings.particle_size = 0.6
settings.size_random = 0.0
settings.child_type = 'NONE'
scene.frame_set(1)
bpy.context.view_layer.update()

instance_mesh = bpy.data.meshes.new('HairLifecycleInstancerMesh')
instance_mesh.from_pydata([(0,0,0),(0.5,0,0),(0,0.5,0)], [], [(0,1,2)])
instancer = bpy.data.objects.new('HairLifecycleInstancer', instance_mesh)
scene.collection.objects.link(instancer)
instancer.track_axis = 'POS_Z'
modifier = instancer.modifiers.new('Hair Lifecycle Instance', 'PARTICLE_INSTANCE')
modifier.object = emitter
modifier.particle_system_index = 1
modifier.axis = 'Z'
modifier.space = 'WORLD'
modifier.use_normal = True
modifier.use_children = False
modifier.use_size = False
modifier.use_path = False
modifier.show_alive = True
modifier.show_dead = False
modifier.show_unborn = False

depsgraph = bpy.context.evaluated_depsgraph_get()
evaluated_emitter = emitter.evaluated_get(depsgraph)
particles = [
    {'birth_time':float(particle.birth_time),
     'die_time':float(particle.die_time),
     'alive_state':particle.alive_state,
     'position':list(particle.location),
     'velocity':list(particle.velocity)}
    for particle in evaluated_emitter.particle_systems[0].particles
]
evaluated_instancer = instancer.evaluated_get(depsgraph)
mesh = evaluated_instancer.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
try:
    expected_mesh = {
        'positions':[list(vertex.co) for vertex in mesh.vertices],
        'faces':[list(polygon.vertices) for polygon in mesh.polygons],
    }
finally:
    evaluated_instancer.to_mesh_clear()
with open(os.path.join(root, 'particle_hair_lifecycle.json'), 'w', encoding='utf-8') as output:
    json.dump({'particles':particles, 'mesh':expected_mesh}, output)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, 'particle_hair_lifecycle.blend'))
"#,
    )?;
    let mut create_fixture = Command::new(&blender_path);
    create_fixture
        .args([
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
        ])
        .arg(&fixture_script)
        .arg("--")
        .arg(root);
    run(&mut create_fixture, "Blender Hair lifecycle fixture")?;
    let blender: Value = serde_json::from_slice(&fs::read(&expected_path)?)?;
    let expected_particles = blender["particles"]
        .as_array()
        .ok_or("Blender Hair particle states are missing")?;
    assert_eq!(expected_particles.len(), 4);
    assert!(
        expected_particles.iter().all(|particle| {
            particle["birth_time"] == 0.0 && particle["alive_state"] == "ALIVE"
        })
    );

    let project = root.join("particle-hair-lifecycle-project");
    initialize(&project)?;
    let mut import = pot();
    import
        .arg("import")
        .arg(&project)
        .arg("--file")
        .arg(root.join("particle_hair_lifecycle.blend"))
        .args([
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "0",
            "--blender",
        ])
        .arg(&blender_path)
        .arg("--json");
    let imported: Value =
        serde_json::from_slice(&run(&mut import, "importing Hair lifecycle fixture")?.stdout)?;
    assert_eq!(imported["result"]["losses"], json!([]), "{imported}");
    let document: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let node_id_by_name = |name: &str| -> Result<potter_core::model::Id, Box<dyn Error>> {
        let id = document["nodes"]
            .as_object()
            .and_then(|nodes| {
                nodes
                    .iter()
                    .find(|(_, node)| node["name"] == name)
                    .map(|(id, _)| id.clone())
            })
            .ok_or_else(|| format!("imported Blender object `{name}` is missing"))?;
        Ok(potter_core::model::Id::new(id)?)
    };
    let emitter_id = node_id_by_name("HairLifecycleEmitter")?;
    let instancer_id = node_id_by_name("HairLifecycleInstancer")?;
    let typed_document: potter_core::model::SceneDoc = serde_json::from_value(document)?;
    let evaluated = potter_core::eval::Snapshot::evaluate(
        &typed_document,
        &potter_core::eval::EvaluationContext {
            frame: Some(1.0),
            ..potter_core::eval::EvaluationContext::default()
        },
    )?;
    let native_particles = evaluated
        .simulated_particles
        .get(&emitter_id)
        .ok_or("native Hair ParticleStates are missing")?;
    assert_eq!(native_particles.len(), expected_particles.len());
    for (index, (actual, expected)) in native_particles.iter().zip(expected_particles).enumerate() {
        let birth_time = expected["birth_time"]
            .as_f64()
            .ok_or("Blender Hair birth time is missing")?;
        assert!((actual.birth_frame - birth_time).abs() <= MODIFIER_POSITION_TOLERANCE);
        assert_eq!(
            actual.life_state,
            potter_core::sim::ParticleLifeState::Alive,
            "Hair {index}"
        );
        assert_numeric_rows(
            &json!([actual.position]),
            &json!([expected["position"]]),
            &format!("Hair {index} current position"),
        );
        assert_numeric_rows(
            &json!([actual.velocity]),
            &json!([expected["velocity"]]),
            &format!("Hair {index} current velocity"),
        );
    }
    let inspection = inspect(&project, &instancer_id.to_string(), "1")?;
    let geometry = &inspection["result"]["items"][0]["evaluated_geometry"];
    assert_numeric_rows(
        &geometry["positions"],
        &blender["mesh"]["positions"],
        "native Hair lifecycle instance positions",
    );
    assert_eq!(geometry["faces"], blender["mesh"]["faces"]);
    Ok(())
}
#[test]
fn particle_instance_hair_path_preserve_shape_randomness_and_corners_match_blender()
-> Result<(), Box<dyn Error>> {
    let Some(blender_path) = blender_executable() else {
        eprintln!("skipping hair path Blender parity; Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let script = root.join("particle_hair_path.py");
    let expected_path = root.join("particle_hair_path.json");
    fs::write(
        &script,
        r#"
import bpy, json, os, sys
from mathutils import Matrix, Vector
if tuple(bpy.app.version[:3]) != (5,2,2):
    raise RuntimeError('hair path parity requires Blender 5.2.2')
root = sys.argv[sys.argv.index('--') + 1]
scene = bpy.context.scene
scene.use_gravity = False
scene.frame_set(1)

def add_corner_data(mesh):
    for material_index in range(3):
        material = bpy.data.materials.new('HairSourceMaterial%d' % material_index)
        mesh.materials.append(material)
    for polygon in mesh.polygons:
        polygon.material_index = polygon.index % 3
    source_uv = mesh.uv_layers.new(name='SourceUV')
    particle_uv = mesh.uv_layers.new(name='ParticleUV')
    for polygon in mesh.polygons:
        for local_corner, loop_index in enumerate(range(polygon.loop_start, polygon.loop_start + polygon.loop_total)):
            uv = (0.1 + 0.17 * polygon.index + 0.11 * local_corner,
                  0.2 + 0.13 * polygon.index + 0.07 * local_corner)
            source_uv.data[loop_index].uv = uv
            particle_uv.data[loop_index].uv = (uv[0] + 0.031, uv[1] + 0.047)
    corner_color = mesh.color_attributes.new(name='HairCornerColor', type='FLOAT_COLOR', domain='CORNER')
    for polygon in mesh.polygons:
        for local_corner, loop_index in enumerate(range(polygon.loop_start, polygon.loop_start + polygon.loop_total)):
            corner_color.data[loop_index].color = (
                0.12 + 0.11 * polygon.index + 0.07 * local_corner,
                0.21 + 0.09 * polygon.index + 0.05 * local_corner,
                0.31 + 0.07 * polygon.index + 0.03 * local_corner,
                1.0,
            )
    mesh.normals_split_custom_set([
        Vector((0.13 * ((loop.index % 3) - 1), 0.09 * ((loop.index % 4) - 1.5), 1.0)).normalized()
        for loop in mesh.loops
    ])
    mesh.update()

def corner_normal_codes(mesh):
    attribute = mesh.attributes.get('custom_normal')
    if attribute is None:
        return None
    if attribute.domain != 'CORNER' or attribute.data_type != 'INT16_2D':
        raise RuntimeError('custom normals must be Blender INT16_2D CORNER codes')
    return [None if list(item.value) == [0,0] else [int(value) for value in item.value]
            for item in attribute.data]

def mesh_snapshot(mesh):
    return {
        'positions':[list(vertex.co) for vertex in mesh.vertices],
        'faces':[list(polygon.vertices) for polygon in mesh.polygons],
        'materials':[int(polygon.material_index) for polygon in mesh.polygons],
        'uv_layers':{layer.name:[list(item.uv) for item in layer.data] for layer in mesh.uv_layers},
        'custom_normal_codes':corner_normal_codes(mesh),
        'color_layer_names':[layer.name for layer in mesh.color_attributes if layer.domain == 'CORNER'],
        'color_attributes':{
            layer.name:{'domain':layer.domain, 'data_type':layer.data_type,
                        'values':[list(item.color) for item in layer.data]}
            for layer in mesh.color_attributes if layer.domain == 'CORNER'
        },
        'vertex_groups':{},
    }

emitter_mesh = bpy.data.meshes.new('HairEmitterMesh')
emitter_mesh.from_pydata([(0,0,0),(1,0,0),(1.2,1,0.3),(0.2,1,0.3)], [], [(0,1,2,3)])
emitter = bpy.data.objects.new('HairEmitter', emitter_mesh)
scene.collection.objects.link(emitter)
bpy.context.view_layer.update()
face_points = [Vector(emitter_mesh.vertices[index].co) for index in emitter_mesh.polygons[0].vertices]
orig_space = [Vector((0,0)), Vector((1,0)), Vector((1,1))]
du = orig_space[1] - orig_space[0]
dv = orig_space[2] - orig_space[0]
determinant = du.x * dv.y - dv.x * du.y
edge_u = ((face_points[1] - face_points[0]) * dv.y - (face_points[2] - face_points[0]) * du.y) / determinant
edge_v = ((face_points[2] - face_points[0]) * du.x - (face_points[1] - face_points[0]) * dv.x) / determinant
edge_u.normalize()
edge_v = edge_v - edge_u * edge_v.dot(edge_u)
edge_v.normalize()
face_normal = edge_u.cross(edge_v).normalized()
face_x = edge_u.cross(face_normal).normalized()
origspace_basis = Matrix((face_x, edge_u, face_normal)).transposed()
root_basis_quaternion = (emitter.matrix_world.to_3x3() @ origspace_basis).to_quaternion()
root_basis_frame = [root_basis_quaternion.x, root_basis_quaternion.y,
                    root_basis_quaternion.z, root_basis_quaternion.w]
for selected in list(bpy.context.selected_objects):
    selected.select_set(False)
emitter.select_set(True)
bpy.context.view_layer.objects.active = emitter
bpy.ops.object.particle_system_add()
system = emitter.particle_systems[0]
system.seed = 19
settings = system.settings
settings.type = 'HAIR'
settings.count = 1
settings.hair_length = 2.0
settings.hair_step = 3
settings.particle_size = 0.75
settings.size_random = 0.0
settings.child_type = 'NONE'
render_mesh = bpy.data.meshes.new('HairRenderObjectMesh')
render_mesh.from_pydata([(0,0,0),(0.1,0,0),(0,0.1,0)], [], [(0,1,2)])
render_object = bpy.data.objects.new('HairRenderObject', render_mesh)
scene.collection.objects.link(render_object)
settings.render_type = 'OBJECT'
settings.instance_object = render_object
settings.display_step = 3
scene.frame_set(1)
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
evaluated_emitter = emitter.evaluated_get(depsgraph)
particle = evaluated_emitter.particle_systems[0].particles[0]
for index, key in enumerate(particle.hair_keys):
    t = float(index) / max(1, len(particle.hair_keys) - 1)
    position = Vector(key.co)
    position.x += 0.55 * t * t
    position.y += 0.25 * t * t * (1.0 - t)
    key.co = position
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
evaluated_emitter = emitter.evaluated_get(depsgraph)
particle = evaluated_emitter.particle_systems[0].particles[0]
root_basis_mesh = bpy.data.meshes.new('HairRootBasisMesh')
root_basis_mesh.from_pydata(
    [(0,0,0), (1,0,0), (0,1,0), (0,0,1)],
    [],
    [(0,1,2), (0,1,3), (0,2,3), (1,2,3)],
)
root_probe = bpy.data.objects.new('HairRootBasisProbe', root_basis_mesh)
scene.collection.objects.link(root_probe)
root_probe.track_axis = 'POS_Z'
root_modifier = root_probe.modifiers.new('Hair Root Frame Probe', 'PARTICLE_INSTANCE')
root_modifier.object = emitter
root_modifier.particle_system_index = 1
root_modifier.axis = 'Z'
root_modifier.space = 'WORLD'
root_modifier.use_normal = True
root_modifier.use_children = False
root_modifier.use_path = True
root_modifier.use_preserve_shape = True
root_modifier.use_size = False
root_modifier.position = 0.0
root_modifier.random_position = 0.0
root_modifier.rotation = 0.0
root_modifier.random_rotation = 0.0
root_modifier.show_alive = True
root_modifier.show_dead = True
root_modifier.show_unborn = True
def path_keys(evaluated_emitter, particle, particle_rotation, use_bspline):
    hair_keys = list(particle.hair_keys)
    times = [float(key.time) / 100.0 for key in hair_keys]
    positions = [list(key.co) for key in hair_keys]
    control_path = []
    for index, (time, position) in enumerate(zip(times, positions)):
        if index == 0:
            interval = times[1] - times[0]
            before, after = 0, 1
            divisor = 2.0 * interval
        elif index + 1 == len(times):
            interval = times[-1] - times[-2]
            before, after = index - 1, index
            divisor = 2.0 * interval
        else:
            before, after = index - 1, index + 1
            divisor = times[after] - times[before]
        velocity = [0.0, 0.0, 0.0] if divisor <= 0.0 else [
            (positions[after][axis] - positions[before][axis]) / divisor
            for axis in range(3)
        ]
        # HairKey has no rotation payload; retain Particle.rotation as raw key state.
        control_path.append({'time':time, 'position':position, 'velocity':velocity,
                             'rotation':particle_rotation})

    particle_system = evaluated_emitter.particle_systems[0]
    segments = 1 << int(settings.display_step)
    try:
        cache_positions = [
            [float(value) for value in particle_system.co_hair(evaluated_emitter, 0, step)]
            for step in range(segments + 1)
        ]
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        cache_positions = []
    if cache_positions and any(value != 0.0 for position in cache_positions for value in position):
        path = []
        for step, position in enumerate(cache_positions):
            previous = cache_positions[step - 1] if step else position
            following = cache_positions[step + 1] if step == 0 else position
            velocity = [(following[axis] - previous[axis]) * segments for axis in range(3)]
            path.append({'time':float(step) / segments, 'position':position,
                         'velocity':velocity, 'rotation':particle_rotation})
        return path, 'hair_cache', control_path
    return control_path, 'hair_bspline' if use_bspline else 'hair_cardinal', control_path

hair_states = []
hair_root_basis_probe_error = None
hair_root_probe_frame = None
hair_particle_rotation = None
hair_world_matrix = None
for use_bspline in (False, True):
    settings.use_hair_bspline = use_bspline
    scene.frame_set(1)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_emitter = emitter.evaluated_get(depsgraph)
    particle = evaluated_emitter.particle_systems[0].particles[0]
    particle_rotation = [particle.rotation.x, particle.rotation.y,
                         particle.rotation.z, particle.rotation.w]
    root_evaluated = root_probe.evaluated_get(depsgraph)
    root_output = root_evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    try:
        root_basis_positions = [Vector(vertex.co) for vertex in root_output.vertices]
    finally:
        root_evaluated.to_mesh_clear()
    root_axes = [root_basis_positions[index] - root_basis_positions[0] for index in (1,2,3)]
    root_probe_quaternion = Matrix(root_axes).transposed().to_quaternion()
    root_probe_frame = [root_probe_quaternion.x, root_probe_quaternion.y,
                        root_probe_quaternion.z, root_probe_quaternion.w]
    root_basis_probe_error = root_basis_quaternion.rotation_difference(root_probe_quaternion).angle
    path, path_interpolation, control_path = path_keys(
        evaluated_emitter, particle, particle_rotation, use_bspline)
    hair_state = {
        'position':list(particle.location),
        'birth_position':list(particle.prev_location),
        'normal':list(particle.rotation @ Vector((1,0,0))),
        'rotation':particle_rotation,
        'birth_rotation':[particle.prev_rotation.x, particle.prev_rotation.y,
                          particle.prev_rotation.z, particle.prev_rotation.w],
        'velocity':list(particle.velocity),
        'birth_frame':float(particle.birth_time),
        'death_frame':float(particle.die_time),
        'life_state':particle.alive_state,
        'size':float(particle.size),
        'system_index':0,
        'is_child':False,
        'parent_index':None,
        'path_root_rotation':root_probe_frame,
        'path_interpolation':path_interpolation,
        'control_path':control_path,
        'hair_key_data':[{'time':float(key.time), 'weight':float(key.weight), 'co_local':[float(value) for value in key.co_local]} for key in particle.hair_keys],
        'path':path,
        'root_basis_probe_error':root_basis_probe_error,
    }
    hair_states.append(hair_state)
    if not use_bspline:
        hair_root_basis_probe_error = root_basis_probe_error
        hair_root_probe_frame = root_probe_frame
        hair_particle_rotation = particle_rotation
        hair_world_matrix = [evaluated_emitter.matrix_world[row][column]
                             for column in range(4) for row in range(4)]

source_mesh = bpy.data.meshes.new('HairInstanceSourceMesh')
source_mesh.from_pydata([(-0.7,-0.3,0.1),(1.1,-0.2,0),(0.3,0.9,0.2),(-0.4,0.4,0.6)], [], [(0,1,2,3)])
add_corner_data(source_mesh)
source_normal_codes = corner_normal_codes(source_mesh)
cases = []
cli_instancer = None
for state_index, use_bspline in enumerate((False, True)):
    settings.use_hair_bspline = use_bspline
    scene.frame_set(1)
    bpy.context.view_layer.update()
    for preserve in (False, True):
        name = ('hair_bspline_' if use_bspline else 'hair_cardinal_') + (
            'preserve' if preserve else 'project')
        obj = bpy.data.objects.new(name, source_mesh)
        scene.collection.objects.link(obj)
        obj.track_axis = 'POS_Z'
        modifier = obj.modifiers.new(name, 'PARTICLE_INSTANCE')
        modifier.object = emitter
        modifier.particle_system_index = 1
        modifier.axis = 'Z'
        modifier.space = 'WORLD'
        modifier.use_normal = True
        modifier.use_children = False
        modifier.use_path = True
        modifier.use_preserve_shape = preserve
        modifier.use_size = True
        modifier.position = 0.45
        modifier.random_position = 0.18
        modifier.rotation = 0.27
        modifier.random_rotation = 0.31
        modifier.show_alive = True
        modifier.show_dead = True
        modifier.show_unborn = True
        modifier.index_layer_name = 'InstanceIndex'
        modifier.value_layer_name = 'InstanceValue'
        bpy.context.view_layer.update()
        if name == 'hair_bspline_preserve':
            cli_instancer = obj
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
        try:
            cases.append({
                'name':name,
                'particle_state_index':state_index,
                'settings':{
                    'use_preserve_shape':preserve,
                    'use_hair_bspline':use_bspline,
                    'position':modifier.position,
                    'random_position':modifier.random_position,
                    'rotation':modifier.rotation,
                    'random_rotation':modifier.random_rotation,
                },
                'mesh':mesh_snapshot(mesh),
            })
        finally:
            evaluated.to_mesh_clear()

hair_collection = bpy.data.collections.new('HairPathRenderCollection')
scene.collection.children.link(hair_collection)
collection_render_mesh = bpy.data.meshes.new('HairCollectionRenderMesh')
collection_render_mesh.from_pydata([(0,0,0),(0.1,0,0),(0,0.1,0)], [], [(0,1,2)])
collection_render_object = bpy.data.objects.new('HairCollectionRenderObject', collection_render_mesh)
hair_collection.objects.link(collection_render_object)
collection_emitter_mesh = bpy.data.meshes.new('HairCollectionEmitterMesh')
collection_emitter_mesh.from_pydata([(0,0,0),(1,0,0),(0,1,0)], [], [(0,1,2)])
collection_emitter = bpy.data.objects.new('HairCollectionEmitter', collection_emitter_mesh)
scene.collection.objects.link(collection_emitter)
for selected in list(bpy.context.selected_objects):
    selected.select_set(False)
collection_emitter.select_set(True)
bpy.context.view_layer.objects.active = collection_emitter
bpy.ops.object.particle_system_add()
collection_system = collection_emitter.particle_systems[0]
collection_system.seed = 23
collection_settings = collection_system.settings
collection_settings.type = 'HAIR'
collection_settings.count = 1
collection_settings.hair_length = 1.0
collection_settings.hair_step = 3
collection_settings.child_type = 'NONE'
collection_settings.render_type = 'COLLECTION'
collection_settings.instance_collection = hair_collection
collection_settings.display_step = 3
scene.frame_set(1)
bpy.context.view_layer.update()
collection_depsgraph = bpy.context.evaluated_depsgraph_get()
evaluated_collection_emitter = collection_emitter.evaluated_get(collection_depsgraph)
collection_system_eval = evaluated_collection_emitter.particle_systems[0]
collection_cache_positions = [
    [float(value) for value in collection_system_eval.co_hair(
        evaluated_collection_emitter, 0, step)]
    for step in range((1 << collection_settings.display_step) + 1)
]
if not any(value != 0.0 for position in collection_cache_positions for value in position):
    raise RuntimeError('Blender did not evaluate the Hair COLLECTION path cache')
if cli_instancer is None:
    raise RuntimeError('hair_bspline_preserve CLI instancer was not created')
scene.frame_set(1)
bpy.context.view_layer.update()
for selected in list(bpy.context.selected_objects):
    selected.select_set(False)
cli_instancer.select_set(True)
bpy.context.view_layer.objects.active = cli_instancer
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, 'particle_hair_path_source.blend'))


with open(os.path.join(root, 'particle_hair_path.json'), 'w', encoding='utf-8') as output:
    json.dump({
        'hair_seed':system.seed,
        'collection_hair_seed':collection_system.seed,
        'collection_cache_positions':collection_cache_positions,
        'hair_world_matrix':hair_world_matrix,
        'hair_root_basis_frame':root_basis_frame,
        'hair_root_probe_frame':hair_root_probe_frame,
        'hair_root_basis_probe_error':hair_root_basis_probe_error,
        'hair_particle_rotation':hair_particle_rotation,
        'source_normal_codes':source_normal_codes,
        'hair_states':hair_states,
        'cases':cases,
    }, output)
"#,
    )?;
    let mut command = Command::new(&blender_path);
    command
        .args([
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
        ])
        .arg(&script)
        .arg("--")
        .arg(root);
    run(&mut command, "Blender hair path fixture")?;
    let blender: Value = serde_json::from_slice(&fs::read(expected_path)?)?;
    let particles = injected_particles(&blender["hair_states"])?;
    assert_eq!(particles.len(), 2);
    assert!(
        particles.iter().all(|particle| particle.path.len() >= 2),
        "Blender Hair keys must provide full paths",
    );
    assert!(
        particles.iter().all(|particle| {
            particle.path_interpolation == potter_core::sim::ParticlePathInterpolation::HairCache
                && particle.path.len() == 9
        }),
        "Hair oracle must use nine evaluated pathcache samples in hair-cache mode",
    );
    let control_path: Vec<potter_core::sim::ParticlePathKey> =
        serde_json::from_value(blender["hair_states"][0]["control_path"].clone())?;
    let first_interval = control_path[1].time - control_path[0].time;
    let first_slope = (DVec3::from_array(control_path[1].position)
        - DVec3::from_array(control_path[0].position))
        / (2.0 * first_interval);
    assert!(
        DVec3::from_array(control_path[0].velocity).distance(first_slope) < 1.0e-12,
        "Hair control oracle must use the duplicated-endpoint half-slope at its root",
    );
    let last = control_path.len() - 1;
    let last_interval = control_path[last].time - control_path[last - 1].time;
    let last_slope = (DVec3::from_array(control_path[last].position)
        - DVec3::from_array(control_path[last - 1].position))
        / (2.0 * last_interval);
    assert!(
        DVec3::from_array(control_path[last].velocity).distance(last_slope) < 1.0e-12,
        "Hair control oracle must use the duplicated-endpoint half-slope at its tip",
    );
    let root_basis_probe_error = blender["hair_root_basis_probe_error"]
        .as_f64()
        .ok_or("Blender Hair root basis probe error is missing")?;
    assert!(
        root_basis_probe_error <= MODIFIER_POSITION_TOLERANCE,
        "OrigSpace-derived root={:?}, evaluated basis probe={:?}, Particle.rotation={:?}, angular error={root_basis_probe_error}",
        blender["hair_root_basis_frame"],
        blender["hair_root_probe_frame"],
        blender["hair_particle_rotation"],
    );
    for (index, particle) in particles.iter().enumerate() {
        let root_rotation: [f64; 4] =
            serde_json::from_value(blender["hair_states"][index]["path_root_rotation"].clone())?;
        assert_eq!(particle.path_root_rotation, Some(root_rotation));
        let root_basis_probe_error = blender["hair_states"][index]["root_basis_probe_error"]
            .as_f64()
            .ok_or("Blender Hair root basis probe error is missing")?;
        assert!(
            root_basis_probe_error <= MODIFIER_POSITION_TOLERANCE,
            "Hair root frame error {root_basis_probe_error} exceeds {MODIFIER_POSITION_TOLERANCE}",
        );
    }
    let root_rotation: [f64; 4] = serde_json::from_value(blender["hair_root_probe_frame"].clone())?;
    assert_eq!(particles[0].path_root_rotation, Some(root_rotation));
    assert_numeric_rows(
        &json!([particles[0].path[0].rotation]),
        &json!([blender["hair_particle_rotation"]]),
        "Hair ParticleKey rotation remains raw while path root is separate",
    );
    assert_numeric_rows(
        &json!([particles[1].path[0].rotation]),
        &json!([blender["hair_states"][1]["rotation"]]),
        "Hair path keeps the raw particle quaternion",
    );
    let source = potter_core::geom::Mesh::from_positions_and_faces(
        vec![
            DVec3::new(-0.7, -0.3, 0.1),
            DVec3::new(1.1, -0.2, 0.0),
            DVec3::new(0.3, 0.9, 0.2),
            DVec3::new(-0.4, 0.4, 0.6),
        ],
        vec![vec![0, 1, 2, 3]],
    )?;
    let mut source = source;
    add_corner_fixture_data(&mut source, &blender["source_normal_codes"], false, true);
    let world_columns: [f64; 16] = serde_json::from_value(blender["hair_world_matrix"].clone())?;
    let source_world = DMat4::from_cols_array(&world_columns);
    let system_seed = blender["hair_seed"]
        .as_u64()
        .ok_or("hair system seed is missing")? as u32;
    let emitter_mesh = potter_core::geom::Mesh::from_positions_and_faces(
        vec![
            DVec3::ZERO,
            DVec3::X,
            DVec3::new(1.2, 1.0, 0.3),
            DVec3::new(0.2, 1.0, 0.3),
        ],
        vec![vec![0, 1, 2, 3]],
    )?;
    let first_path = serde_json::to_value(&particles[0].path)?;
    let path_interpolation = serde_json::to_value(particles[0].path_interpolation)?;
    let simulation_settings = json!({
        "type":"HAIR",
        "count":1,
        "frame_start":1.0,
        "frame_end":1.0,
        "lifetime":100.0,
        "physics_type":"NO",
        "_potter_path_interpolation":path_interpolation,
        "_potter_path_keys":[first_path],
    });
    let simulated = potter_core::sim::particles::simulate_lifecycle(
        &emitter_mesh,
        &simulation_settings,
        source_world,
        1.0,
        24.0,
        DVec3::ZERO,
        system_seed,
        &[],
        &[],
    )?;
    assert_eq!(simulated.len(), 1);
    let native_root = DQuat::from_array(
        simulated[0]
            .path_root_rotation
            .ok_or("native hair root frame is missing")?,
    )
    .normalize();
    let oracle_root = DQuat::from_array(root_rotation).normalize();
    let root_error = 2.0 * native_root.dot(oracle_root).abs().clamp(0.0, 1.0).acos();
    assert!(
        root_error <= MODIFIER_POSITION_TOLERANCE,
        "native emitter root frame error {root_error} exceeds {MODIFIER_POSITION_TOLERANCE}",
    );
    let emitter_id = potter_core::model::Id::new("hair_emitter")?;
    let data_id = potter_core::model::Id::new("hair_emitter_mesh")?;
    let mut state_doc = potter_core::model::SceneDoc::new("hair-path-state".to_owned());
    state_doc
        .collections
        .get_mut(&potter_core::model::Id::new("collection_root")?)
        .expect("default root collection")
        .objects
        .push(emitter_id.clone());
    state_doc.data_blocks.insert(
        data_id.clone(),
        potter_core::model::DataBlock {
            mesh: Some(emitter_mesh.clone()),
            ..potter_core::model::DataBlock::default()
        },
    );
    // The recorded Blender world matrix is the emitter's transform (TRS; Blender object matrices are affine).
    let (scale, rotation, translation) = source_world.to_scale_rotation_translation();
    state_doc.nodes.insert(
        emitter_id.clone(),
        potter_core::model::Node {
            name: "Hair emitter".to_owned(),
            kind: "mesh".to_owned(),
            data: Some(data_id),
            transform: potter_core::model::Transform {
                translation: translation.to_array(),
                rotation: rotation.to_array(),
                scale: scale.to_array(),
                ..potter_core::model::Transform::default()
            },
            properties: Map::from_iter([(
                "physics_particle_emitter".to_owned(),
                simulation_settings.clone(),
            )]),
            ..potter_core::model::Node::default()
        },
    );
    let instancer_id = potter_core::model::Id::new("hair_instancer")?;
    state_doc.nodes.insert(
        instancer_id,
        potter_core::model::Node {
            name: "Hair instancer".to_owned(),
            modifiers: vec![potter_core::model::Modifier {
                id: potter_core::model::Id::new("hair_instance")?,
                modifier_type: "particle_instance".to_owned(),
                name: "Hair path instance".to_owned(),
                enabled: true,
                show_viewport: true,
                show_render: true,
                params: Map::from_iter([
                    ("object".to_owned(), json!("hair_emitter")),
                    ("particle_system_index".to_owned(), json!(1)),
                    ("use_path".to_owned(), json!(true)),
                ]),
                binding_data: None,
                runtime: Default::default(),
            }],
            ..potter_core::model::Node::default()
        },
    );
    let evaluated = potter_core::eval::Snapshot::evaluate(
        &state_doc,
        &potter_core::eval::EvaluationContext {
            frame: Some(1.0),
            ..potter_core::eval::EvaluationContext::default()
        },
    )?;
    let hair_state = evaluated
        .simulated_particles
        .get(&emitter_id)
        .and_then(|states| states.first())
        .ok_or("generated Hair ParticleState is missing")?;
    assert_eq!(
        hair_state.path_interpolation,
        potter_core::sim::ParticlePathInterpolation::HairCache,
    );
    assert!(
        !hair_state.path.is_empty(),
        "Hair path mode must populate the state path"
    );
    let blender_rotation: [f64; 4] =
        serde_json::from_value(blender["hair_particle_rotation"].clone())?;
    let generated_rotation = DQuat::from_array(hair_state.rotation).normalize();
    let blender_rotation = DQuat::from_array(blender_rotation).normalize();
    let rotation_error = 2.0
        * generated_rotation
            .dot(blender_rotation)
            .abs()
            .clamp(0.0, 1.0)
            .acos();
    assert!(
        rotation_error <= MODIFIER_POSITION_TOLERANCE,
        "generated Hair current rotation error {rotation_error} exceeds {MODIFIER_POSITION_TOLERANCE}",
    );
    for case in blender["cases"]
        .as_array()
        .ok_or("Hair path cases are missing")?
    {
        let settings = &case["settings"];
        let state_index = case["particle_state_index"]
            .as_u64()
            .ok_or("Hair path case particle state index is missing")?
            as usize;
        let particle = particles
            .get(state_index)
            .ok_or("Hair path case particle state index is out of range")?;
        let result = potter_core::geom::modifiers::particle_instance::particle_instance(
            &source,
            std::slice::from_ref(particle),
            system_seed,
            source_world,
            2,
            &Map::from_iter([
                ("use_normal".to_owned(), json!(true)),
                ("use_children".to_owned(), json!(false)),
                ("use_path".to_owned(), json!(true)),
                (
                    "use_preserve_shape".to_owned(),
                    settings["use_preserve_shape"].clone(),
                ),
                ("use_size".to_owned(), json!(true)),
                ("show_alive".to_owned(), json!(true)),
                ("show_dead".to_owned(), json!(true)),
                ("show_unborn".to_owned(), json!(true)),
                ("position".to_owned(), settings["position"].clone()),
                (
                    "random_position".to_owned(),
                    settings["random_position"].clone(),
                ),
                ("rotation".to_owned(), settings["rotation"].clone()),
                (
                    "random_rotation".to_owned(),
                    settings["random_rotation"].clone(),
                ),
                ("axis".to_owned(), json!("Z")),
                ("space".to_owned(), json!("WORLD")),
                ("index_layer_name".to_owned(), json!("InstanceIndex")),
                ("value_layer_name".to_owned(), json!("InstanceValue")),
            ]),
        )?;
        assert_ordered_mesh_matches_blender(
            &result,
            &case["mesh"],
            case["name"].as_str().unwrap_or("hair path"),
        );
    }
    let cli_case = blender["cases"]
        .as_array()
        .and_then(|cases| {
            cases
                .iter()
                .find(|case| case["name"] == "hair_bspline_preserve")
        })
        .ok_or("Blender B-spline preserve case is missing")?;
    let project = root.join("particle-hair-path-project");
    initialize(&project)?;
    let mut import = pot();
    import
        .arg("import")
        .arg(&project)
        .arg("--file")
        .arg(root.join("particle_hair_path_source.blend"))
        .args([
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "0",
            "--blender",
        ])
        .arg(&blender_path)
        .arg("--json");
    let imported: Value =
        serde_json::from_slice(&run(&mut import, "importing Hair path fixture")?.stdout)?;
    assert_eq!(imported["result"]["losses"], json!([]), "{imported}");

    let document: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let node_id_by_name = |name: &str| -> Result<String, Box<dyn Error>> {
        document["nodes"]
            .as_object()
            .and_then(|nodes| {
                nodes
                    .iter()
                    .find(|(_, node)| node["name"] == name)
                    .map(|(id, _)| id.clone())
            })
            .ok_or_else(|| format!("imported Blender object `{name}` is missing").into())
    };
    let instancer_id = node_id_by_name("hair_bspline_preserve")?;
    let emitter_id = node_id_by_name("HairEmitter")?;
    let instancer_node = &document["nodes"][instancer_id.as_str()];
    let modifier = instancer_node["modifiers"]
        .as_array()
        .and_then(|modifiers| {
            modifiers.iter().find(|modifier| {
                modifier["type"] == "particle_instance"
                    && modifier["name"] == "hair_bspline_preserve"
            })
        })
        .ok_or("imported Hair path particle-instance modifier is missing")?;
    assert_eq!(modifier["params"]["use_path"], true);
    assert_eq!(modifier["params"]["use_preserve_shape"], true);
    let modifier_id = modifier["id"]
        .as_str()
        .ok_or("Hair path modifier ID is missing")?;
    let emitter_node = &document["nodes"][emitter_id.as_str()];
    let particle_settings = emitter_node["properties"]["physics_particle_systems"]
        .as_object()
        .and_then(|systems| {
            systems
                .values()
                .find(|settings| settings["seed"] == blender["hair_seed"])
        })
        .ok_or("imported Hair particle-system settings are missing")?;
    assert_eq!(
        particle_settings["_potter_path_interpolation"],
        "hair_cache"
    );
    let imported_paths = particle_settings["_potter_path_keys"]
        .as_array()
        .ok_or("Blender hair path cache was not serialized by the bridge")?;
    assert_eq!(imported_paths.len(), 1);
    let imported_path = imported_paths[0]
        .as_array()
        .ok_or("imported Hair path cache is not an ordered key list")?;
    let blender_path_keys = blender["hair_states"][1]["path"]
        .as_array()
        .ok_or("Blender Hair path oracle is missing")?;
    assert_eq!(
        imported_path.len(),
        9,
        "bridge must capture the full path cache"
    );
    assert_eq!(imported_path.len(), blender_path_keys.len());
    for (key_index, (actual, expected)) in imported_path.iter().zip(blender_path_keys).enumerate() {
        assert_eq!(
            actual["time"], expected["time"],
            "bridge Hair path key {key_index} time",
        );
        assert_numeric_rows(
            &json!([actual["position"]]),
            &json!([expected["position"]]),
            &format!("bridge Hair path key {key_index} position"),
        );
    }
    let collection_emitter_id = node_id_by_name("HairCollectionEmitter")?;
    let collection_emitter_node = &document["nodes"][collection_emitter_id.as_str()];
    let collection_settings = collection_emitter_node["properties"]["physics_particle_systems"]
        .as_object()
        .and_then(|systems| {
            systems
                .values()
                .find(|settings| settings["seed"] == blender["collection_hair_seed"])
        })
        .ok_or("imported Hair COLLECTION settings are missing")?;
    assert_eq!(
        collection_settings["_potter_path_interpolation"],
        "hair_cache"
    );
    let collection_paths = collection_settings["_potter_path_keys"]
        .as_array()
        .and_then(|paths| paths.first())
        .and_then(Value::as_array)
        .ok_or("Hair COLLECTION cache path was not imported")?;
    let collection_positions = blender["collection_cache_positions"]
        .as_array()
        .ok_or("Blender Hair COLLECTION cache positions are missing")?;
    assert_eq!(collection_paths.len(), collection_positions.len());
    assert_eq!(collection_paths.len(), 9);
    for (index, (actual, expected_position)) in collection_paths
        .iter()
        .zip(collection_positions)
        .enumerate()
    {
        assert_numeric_rows(
            &json!([[actual["time"]]]),
            &json!([[index as f64 / 8.0]]),
            &format!("Hair COLLECTION cache key {index} time"),
        );
        assert_numeric_rows(
            &json!([actual["position"]]),
            &json!([expected_position]),
            &format!("Hair COLLECTION cache key {index} position"),
        );
    }

    let typed_document: potter_core::model::SceneDoc = serde_json::from_value(document.clone())?;
    assert_snapshot_mesh_matches_blender(
        &typed_document,
        &instancer_id,
        1.0,
        &cli_case["mesh"],
        "imported Hair path full corner snapshot",
    )?;
    let assert_cli_mesh = |actual: &Value, expected: &Value, context: &str| {
        let geometry = &actual["result"]["items"][0]["evaluated_geometry"];
        assert_numeric_rows(
            &geometry["positions"],
            &expected["positions"],
            &format!("{context} ordered positions"),
        );
        assert_eq!(
            geometry["faces"], expected["faces"],
            "{context} ordered faces"
        );
    };
    let inspected = inspect(&project, &instancer_id, "1")?;
    assert_cli_mesh(&inspected, &cli_case["mesh"], "Hair path CLI frame 1");
    let live_exported = root.join("particle_hair_path_live.blend");
    let mut live_export = pot();
    live_export
        .arg("export")
        .arg(&project)
        .args(["--format", "blend", "--out"])
        .arg(&live_exported)
        .arg("--blender")
        .arg(&blender_path)
        .arg("--json");
    run(&mut live_export, "exporting live Hair path modifier")?;
    let live_project = root.join("particle-hair-path-live-reimport");
    initialize(&live_project)?;
    let mut live_import = pot();
    live_import
        .arg("import")
        .arg(&live_project)
        .arg("--file")
        .arg(&live_exported)
        .args([
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "0",
            "--blender",
        ])
        .arg(&blender_path)
        .arg("--json");
    let live_imported: Value = serde_json::from_slice(
        &run(&mut live_import, "reimporting live Hair path modifier")?.stdout,
    )?;
    assert_eq!(
        live_imported["result"]["losses"],
        json!([]),
        "{live_imported}"
    );
    let live_document: Value = serde_json::from_slice(&fs::read(live_project.join("scene.json"))?)?;
    let live_instancer_id = live_document["nodes"]
        .as_object()
        .and_then(|nodes| {
            nodes
                .iter()
                .find(|(_, node)| node["name"] == "hair_bspline_preserve")
                .map(|(id, _)| id.clone())
        })
        .ok_or("live Hair path instancer is missing after export")?;
    let live_instancer = &live_document["nodes"][live_instancer_id.as_str()];
    let live_modifier = live_instancer["modifiers"]
        .as_array()
        .and_then(|modifiers| {
            modifiers
                .iter()
                .find(|modifier| modifier["type"] == "particle_instance")
        })
        .ok_or("live Hair path modifier did not survive Blender export")?;
    assert_eq!(live_modifier["params"]["use_path"], true);
    let typed_live_document: potter_core::model::SceneDoc =
        serde_json::from_value(live_document.clone())?;
    assert_snapshot_mesh_matches_blender(
        &typed_live_document,
        &live_instancer_id,
        1.0,
        &cli_case["mesh"],
        "live Hair path export full corner snapshot",
    )?;
    let live_inspection = inspect(&live_project, &live_instancer_id, "1")?;
    assert_cli_mesh(
        &live_inspection,
        &cli_case["mesh"],
        "live Hair path export mesh",
    );

    let revision = document["revision"]
        .as_u64()
        .ok_or("imported Hair scene revision is missing")?;
    let batch = root.join("apply_hair_path_particle_instance.json");
    apply(
        &project,
        &batch,
        &json!([{
            "op":"modifier.apply",
            "target":{"id":instancer_id},
            "modifier_id":modifier_id
        }]),
        revision,
    )?;
    let applied_document: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let applied_instancer = &applied_document["nodes"][instancer_id.as_str()];
    assert!(
        applied_instancer["modifiers"]
            .as_array()
            .is_some_and(|modifiers| modifiers
                .iter()
                .all(|modifier| modifier["id"] != modifier_id)),
        "applying the Hair path modifier must remove its stack entry",
    );
    let applied_emitter = &applied_document["nodes"][emitter_id.as_str()];
    let applied_particle_settings = applied_emitter["properties"]["physics_particle_systems"]
        .as_object()
        .and_then(|systems| {
            systems
                .values()
                .find(|settings| settings["seed"] == blender["hair_seed"])
        })
        .ok_or("applying the instancer must preserve its source Hair particle system")?;
    assert_eq!(
        applied_particle_settings["_potter_path_keys"], particle_settings["_potter_path_keys"],
        "applying the instancer must retain the source path cache",
    );
    assert_eq!(
        applied_particle_settings["_potter_hair_key_data"],
        particle_settings["_potter_hair_key_data"],
        "applying the instancer must retain source Hair key controls",
    );
    let typed_applied_document: potter_core::model::SceneDoc =
        serde_json::from_value(applied_document.clone())?;
    assert_snapshot_mesh_matches_blender(
        &typed_applied_document,
        &instancer_id,
        1.0,
        &cli_case["mesh"],
        "applied Hair path full corner snapshot",
    )?;
    let applied = inspect(&project, &instancer_id, "1")?;
    assert_cli_mesh(&applied, &cli_case["mesh"], "applied Hair path mesh");

    let exported = root.join("particle_hair_path_applied.blend");
    let mut export = pot();
    export
        .arg("export")
        .arg(&project)
        .args(["--format", "blend", "--out"])
        .arg(&exported)
        .arg("--blender")
        .arg(&blender_path)
        .arg("--json");
    run(&mut export, "exporting applied Hair path mesh")?;
    let reopen_script = root.join("reopen_particle_hair_path.py");
    let reopened_path = root.join("reopened_particle_hair_path.json");
    fs::write(
        &reopen_script,
        r#"
import bpy, json, sys
output = sys.argv[sys.argv.index('--') + 1]
obj = bpy.data.objects['hair_bspline_preserve']
mesh = obj.data
emitter = bpy.data.objects['HairEmitter']
scene = bpy.context.scene
scene.frame_set(1)
bpy.context.view_layer.update()
def corner_normal_codes(mesh):
    attribute = mesh.attributes.get('custom_normal')
    if attribute is None:
        return None
    if attribute.domain != 'CORNER' or attribute.data_type != 'INT16_2D':
        raise RuntimeError('custom normals must be Blender INT16_2D CORNER codes')
    return [None if list(item.value) == [0,0] else [int(value) for value in item.value]
            for item in attribute.data]
result = {
    'track_axis':obj.track_axis,
    'modifier_count':len(obj.modifiers),
    'emitter_particle_system_count':len(emitter.particle_systems),
    'emitter_hair_key_data':[{'time':float(key.time), 'weight':float(key.weight), 'co_local':[float(value) for value in key.co_local]} for key in emitter.particle_systems[0].particles[0].hair_keys],
    'positions':[list(vertex.co) for vertex in mesh.vertices],
    'faces':[list(polygon.vertices) for polygon in mesh.polygons],
    'materials':[int(polygon.material_index) for polygon in mesh.polygons],
    'uv_layers':{layer.name:[list(item.uv) for item in layer.data] for layer in mesh.uv_layers},
    'custom_normal_codes':corner_normal_codes(mesh),
    'color_layer_names':[layer.name for layer in mesh.color_attributes if layer.domain == 'CORNER'],
    'color_attributes':{
        layer.name:{'domain':layer.domain, 'data_type':layer.data_type,
                    'values':[list(item.color) for item in layer.data]}
        for layer in mesh.color_attributes if layer.domain == 'CORNER'
    },
}
with open(output, 'w', encoding='utf-8') as handle:
    json.dump(result, handle)
"#,
    )?;
    let mut reopen = Command::new(&blender_path);
    reopen
        .args(["--background", "--disable-autoexec"])
        .arg(&exported)
        .args(["--python-exit-code", "1", "--python"])
        .arg(&reopen_script)
        .arg("--")
        .arg(&reopened_path);
    run(&mut reopen, "reopening exported Hair path blend")?;
    let reopened: Value = serde_json::from_slice(&fs::read(&reopened_path)?)?;
    assert_eq!(reopened["track_axis"], "POS_Z");
    assert_eq!(reopened["modifier_count"], 0);
    assert!(
        reopened["emitter_particle_system_count"]
            .as_u64()
            .is_some_and(|count| count > 0),
        "exported scene must retain the source Hair particle system",
    );
    let expected_hair_keys = blender["hair_states"][1]["hair_key_data"]
        .as_array()
        .ok_or("Blender Hair key source data is missing")?;
    let exported_hair_keys = reopened["emitter_hair_key_data"]
        .as_array()
        .ok_or("exported Hair key data is missing")?;
    assert_eq!(exported_hair_keys.len(), expected_hair_keys.len());
    for (index, (actual, expected)) in exported_hair_keys
        .iter()
        .zip(expected_hair_keys)
        .enumerate()
    {
        assert_numeric_rows(
            &json!([[actual["time"]]]),
            &json!([[expected["time"]]]),
            &format!("reopened Hair key {index} time"),
        );
        assert_numeric_rows(
            &json!([[actual["weight"]]]),
            &json!([[expected["weight"]]]),
            &format!("reopened Hair key {index} weight"),
        );
        assert_numeric_rows(
            &json!([actual["co_local"]]),
            &json!([expected["co_local"]]),
            &format!("reopened Hair key {index} coordinates"),
        );
    }
    assert_numeric_rows(
        &reopened["positions"],
        &cli_case["mesh"]["positions"],
        "reopened Hair path static mesh vertices",
    );
    assert_eq!(reopened["faces"], cli_case["mesh"]["faces"]);
    assert_eq!(reopened["materials"], cli_case["mesh"]["materials"]);
    for (name, expected_uvs) in cli_case["mesh"]["uv_layers"]
        .as_object()
        .ok_or("Hair path UV layer oracle is missing")?
    {
        let actual_uvs = reopened["uv_layers"]
            .get(name)
            .ok_or_else(|| format!("reopened Hair mesh is missing UV layer {name}"))?;
        assert_numeric_rows(
            actual_uvs,
            expected_uvs,
            &format!("reopened Hair path UV {name}"),
        );
    }
    assert_eq!(
        reopened["custom_normal_codes"], cli_case["mesh"]["custom_normal_codes"],
        "reopened Hair path corner custom normals",
    );
    assert_eq!(
        reopened["color_layer_names"], cli_case["mesh"]["color_layer_names"],
        "reopened Hair path corner color layer names",
    );
    for (name, expected_color) in cli_case["mesh"]["color_attributes"]
        .as_object()
        .ok_or("Hair path color attribute oracle is missing")?
    {
        let actual_color = &reopened["color_attributes"][name];
        assert_eq!(actual_color["domain"], expected_color["domain"]);
        assert_eq!(actual_color["data_type"], expected_color["data_type"]);
        assert_numeric_rows(
            &actual_color["values"],
            &expected_color["values"],
            &format!("reopened Hair path corner color {name}"),
        );
    }
    let reimport_project = root.join("particle-hair-path-reimport");
    initialize(&reimport_project)?;
    let mut reimport = pot();
    reimport
        .arg("import")
        .arg(&reimport_project)
        .arg("--file")
        .arg(&exported)
        .args([
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "0",
            "--blender",
        ])
        .arg(&blender_path)
        .arg("--json");
    let reimported: Value =
        serde_json::from_slice(&run(&mut reimport, "reimporting applied Hair path blend")?.stdout)?;
    assert_eq!(reimported["result"]["losses"], json!([]), "{reimported}");
    let reimported_document: Value =
        serde_json::from_slice(&fs::read(reimport_project.join("scene.json"))?)?;
    let reimported_instancer_id = reimported_document["nodes"]
        .as_object()
        .and_then(|nodes| {
            nodes
                .iter()
                .find(|(_, node)| node["name"] == "hair_bspline_preserve")
                .map(|(id, _)| id.clone())
        })
        .ok_or("reimported Hair path instancer is missing")?;
    let reimported_instancer = &reimported_document["nodes"][reimported_instancer_id.as_str()];
    assert_eq!(reimported_instancer["track_axis"], "POS_Z",);
    assert!(
        reimported_instancer["modifiers"]
            .as_array()
            .is_some_and(Vec::is_empty),
        "reimported applied Hair path mesh must remain modifier-free",
    );
    let typed_reimported_document: potter_core::model::SceneDoc =
        serde_json::from_value(reimported_document.clone())?;
    assert_snapshot_mesh_matches_blender(
        &typed_reimported_document,
        &reimported_instancer_id,
        1.0,
        &cli_case["mesh"],
        "reimported Hair path full corner snapshot",
    )?;
    let reimported_inspection = inspect(&reimport_project, &reimported_instancer_id, "1")?;
    assert_cli_mesh(
        &reimported_inspection,
        &cli_case["mesh"],
        "reimported Hair path mesh",
    );
    Ok(())
}

#[test]
fn particle_emission_counts_face_distribution_and_velocity_statistics_match_blender()
-> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping particle emission Blender parity; Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let script = root.join("particle_emission.py");
    let expected_path = root.join("particle_emission.json");
    fs::write(
        &script,
        r"
import bpy, json, os, sys
root = sys.argv[sys.argv.index('--') + 1]
scene = bpy.context.scene
scene.use_gravity = False
mesh = bpy.data.meshes.new('unequal_face_emitter')
mesh.from_pydata(
    [(-5,-2,0), (-1,-2,0), (-1,2,0), (-5,2,0),
     (1,-0.5,0), (2,-0.5,0), (2,0.5,0), (1,0.5,0)],
    [],
    [(0,1,2,3), (4,5,6,7)],
)
emitter = bpy.data.objects.new('Emitter', mesh)
scene.collection.objects.link(emitter)
emitter.location = (1.7,-0.3,0.2)
emitter.rotation_euler = (0.0,-1.2,0.0)
emitter.scale = (1.2,1.2,1.2)
bpy.context.view_layer.objects.active = emitter
emitter.select_set(True)
bpy.ops.object.particle_system_add()
system = emitter.particle_systems[0]
system.seed = 37
settings = system.settings
settings.count = 1000
settings.frame_start = 1
settings.frame_end = 10
settings.lifetime = 20
settings.emit_from = 'FACE'
settings.physics_type = 'NEWTON'
settings.normal_factor = 3.5
settings.factor_random = 0.0
frames = {}
for frame in [1, 4, 7, 10, 12]:
    scene.frame_set(frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    evaluated = emitter.evaluated_get(depsgraph)
    particles = evaluated.particle_systems[0].particles
    frames[str(frame)] = {
        'positions':[list(particle.location) for particle in particles],
        'velocities':[list(particle.velocity) for particle in particles],
        'birth_frames':[particle.birth_time for particle in particles],
        'life_states':[particle.alive_state for particle in particles],
    }
with open(os.path.join(root, 'particle_emission.json'), 'w', encoding='utf-8') as output:
    json.dump(frames, output)
",
    )?;
    run(
        Command::new(&blender)
            .args(["--background", "--factory-startup", "--python"])
            .arg(&script)
            .arg("--")
            .arg(root),
        "particle emission fixture",
    )?;
    let blender_frames: Value = serde_json::from_slice(&fs::read(expected_path)?)?;
    let mesh = potter_core::geom::Mesh::from_positions_and_faces(
        vec![
            DVec3::new(-5.0, -2.0, 0.0),
            DVec3::new(-1.0, -2.0, 0.0),
            DVec3::new(-1.0, 2.0, 0.0),
            DVec3::new(-5.0, 2.0, 0.0),
            DVec3::new(1.0, -0.5, 0.0),
            DVec3::new(2.0, -0.5, 0.0),
            DVec3::new(2.0, 0.5, 0.0),
            DVec3::new(1.0, 0.5, 0.0),
        ],
        vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
    )?;
    let emitter_world = DMat4::from_scale_rotation_translation(
        DVec3::splat(1.2),
        DQuat::from_euler(EulerRot::XYZ, 0.0, -1.2, 0.0),
        DVec3::new(1.7, -0.3, 0.2),
    );
    let inverse_emitter = emitter_world.inverse();
    let emitter_normal = emitter_world.transform_vector3(DVec3::Z).normalize();
    let settings = json!({
        "count":1000,
        "frame_start":1.0,
        "frame_end":10.0,
        "lifetime":20.0,
        "emit_from":"FACE",
        "physics_type":"NEWTON",
        "normal_factor":3.5,
        "factor_random":0.0,
        "seed":37
    });
    for frame_number in [1_u32, 4, 7, 10, 12] {
        let frame = f64::from(frame_number);
        let blender_frame = &blender_frames[frame_number.to_string()];
        let blender_positions = blender_points(&blender_frame["positions"])?;
        let blender_velocities = blender_points(&blender_frame["velocities"])?;
        let potter_particles = potter_core::sim::particles::simulate(
            &mesh,
            &settings,
            emitter_world,
            frame,
            24.0,
            DVec3::ZERO,
            37,
            &[],
            &[],
        )?;
        let blender_emitted_count = blender_frame["life_states"]
            .as_array()
            .ok_or("Blender particle life states are missing")?
            .iter()
            .filter(|life_state| life_state.as_str().is_some_and(|state| state != "UNBORN"))
            .count();
        assert!(
            potter_particles.len().abs_diff(blender_emitted_count) <= 2,
            "emitted particle count at frame {frame}: Potter={}, Blender={blender_emitted_count}",
            potter_particles.len(),
        );
        if frame_number == 10 {
            let fraction = |count: usize, total: usize| -> Result<f64, Box<dyn Error>> {
                if total == 0 {
                    return Err("particle face distribution is empty".into());
                }
                Ok(f64::from(u32::try_from(count)?) / f64::from(u32::try_from(total)?))
            };
            let blender_fraction = fraction(
                blender_positions
                    .iter()
                    .filter(|position| inverse_emitter.transform_point3(**position).x < 0.0)
                    .count(),
                blender_positions.len(),
            )?;
            let potter_fraction = fraction(
                potter_particles
                    .iter()
                    .filter(|particle| {
                        inverse_emitter
                            .transform_point3(DVec3::from_array(particle.position))
                            .x
                            < 0.0
                    })
                    .count(),
                potter_particles.len(),
            )?;
            let expected_large_face_fraction = 16.0 / 17.0;
            assert!(
                (blender_fraction - expected_large_face_fraction).abs()
                    <= FACE_DISTRIBUTION_TOLERANCE,
                "Blender large-face emission fraction is {blender_fraction}",
            );
            assert!(
                (potter_fraction - expected_large_face_fraction).abs()
                    <= FACE_DISTRIBUTION_TOLERANCE,
                "Potter large-face emission fraction is {potter_fraction}",
            );
            assert!(
                (potter_fraction - blender_fraction).abs() <= FACE_DISTRIBUTION_TOLERANCE,
                "large-face emission differs: Potter={potter_fraction}, Blender={blender_fraction}",
            );
            assert!(
                (blender_fraction - 0.5).abs() > 10.0 * FACE_DISTRIBUTION_TOLERANCE,
                "unequal face areas must distinguish area-weighted from uniform-face sampling",
            );
            let mean_normal_velocity = |values: &[DVec3]| -> Result<f64, Box<dyn Error>> {
                if values.is_empty() {
                    return Err("particle velocities are empty".into());
                }
                let count = u32::try_from(values.len())?;
                Ok(values
                    .iter()
                    .map(|velocity| velocity.dot(emitter_normal))
                    .sum::<f64>()
                    / f64::from(count))
            };
            let mean_speed = |values: &[DVec3]| -> Result<f64, Box<dyn Error>> {
                if values.is_empty() {
                    return Err("particle velocities are empty".into());
                }
                let count = u32::try_from(values.len())?;
                Ok(values.iter().map(|velocity| velocity.length()).sum::<f64>() / f64::from(count))
            };
            let potter_velocities = potter_particles
                .iter()
                .map(|particle| DVec3::from_array(particle.velocity))
                .collect::<Vec<_>>();
            let blender_normal_velocity = mean_normal_velocity(&blender_velocities)?;
            let potter_normal_velocity = mean_normal_velocity(&potter_velocities)?;
            let blender_speed = mean_speed(&blender_velocities)?;
            let potter_speed = mean_speed(&potter_velocities)?;
            assert!(
                blender_normal_velocity > 10.0 * VELOCITY_STATISTIC_TOLERANCE,
                "Blender emitter must produce nonzero normal velocity, mean={blender_normal_velocity}",
            );
            assert!(
                (blender_normal_velocity - potter_normal_velocity).abs()
                    <= VELOCITY_STATISTIC_TOLERANCE,
                "mean world-space normal particle velocity differs: Potter={potter_normal_velocity}, Blender={blender_normal_velocity}",
            );
            assert!(
                (blender_speed - potter_speed).abs() <= VELOCITY_STATISTIC_TOLERANCE,
                "mean world-space particle speed differs: Potter={potter_speed}, Blender={blender_speed}",
            );
        }
    }
    Ok(())
}

fn mean_z(mesh: &potter_core::geom::Mesh) -> Result<f64, Box<dyn Error>> {
    let vertex_count = u32::try_from(mesh.vertices.len())?;
    if vertex_count == 0 {
        return Err("cloth mesh has no vertices".into());
    }
    Ok(mesh.vertices.iter().map(|vertex| vertex.co.z).sum::<f64>() / f64::from(vertex_count))
}
fn blender_points(value: &Value) -> Result<Vec<glam::DVec3>, Box<dyn Error>> {
    let coordinates: Vec<[f64; 3]> = serde_json::from_value(value.clone())?;
    Ok(coordinates
        .into_iter()
        .map(glam::DVec3::from_array)
        .collect())
}

// Subdivision backends may number corresponding vertices differently. The fixture deforms
// vertically, so projected rest-space coordinates give a stable one-to-one correspondence.
fn max_pointwise_distance(
    first: &[glam::DVec3],
    second: &[glam::DVec3],
) -> Result<f64, Box<dyn Error>> {
    if first.len() != second.len() {
        return Err(format!(
            "point arrays have different lengths: {} and {}",
            first.len(),
            second.len(),
        )
        .into());
    }
    let mut unmatched = second.to_vec();
    let mut maximum = 0.0_f64;
    for point in first {
        let nearest = unmatched
            .iter()
            .enumerate()
            .min_by(|(_, first), (_, second)| {
                let first_delta = first.truncate() - point.truncate();
                let second_delta = second.truncate() - point.truncate();
                first_delta
                    .length_squared()
                    .total_cmp(&second_delta.length_squared())
            })
            .map(|(index, _)| index)
            .ok_or("point correspondence exhausted unexpectedly")?;
        maximum = maximum.max(point.distance(unmatched.swap_remove(nearest)));
    }
    Ok(maximum)
}
fn pointwise_error_metrics(
    first: &[glam::DVec3],
    second: &[glam::DVec3],
) -> Result<(f64, f64), Box<dyn Error>> {
    if first.len() != second.len() || first.is_empty() {
        return Err("point arrays must have matching nonzero lengths".into());
    }
    let mut unmatched = second.to_vec();
    let mut squared_error_sum = 0.0;
    let mut maximum_error = 0.0_f64;
    for point in first {
        let nearest = unmatched
            .iter()
            .enumerate()
            .min_by(|(_, first), (_, second)| {
                let first_delta = first.truncate() - point.truncate();
                let second_delta = second.truncate() - point.truncate();
                first_delta
                    .length_squared()
                    .total_cmp(&second_delta.length_squared())
            })
            .map(|(index, _)| index)
            .ok_or("point correspondence exhausted unexpectedly")?;
        let error = point.distance(unmatched.swap_remove(nearest));
        squared_error_sum += error * error;
        maximum_error = maximum_error.max(error);
    }
    let count = f64::from(u32::try_from(first.len())?);
    Ok(((squared_error_sum / count).sqrt(), maximum_error))
}

fn centroid(points: &[glam::DVec3]) -> Result<glam::DVec3, Box<dyn Error>> {
    let count = f64::from(u32::try_from(points.len())?);
    if count == 0.0 {
        return Err("point array is empty".into());
    }
    Ok(points.iter().copied().sum::<glam::DVec3>() / count)
}
fn max_corresponding_distance(
    first: &[glam::DVec3],
    second: &[glam::DVec3],
) -> Result<f64, Box<dyn Error>> {
    if first.len() != second.len() {
        return Err(format!(
            "point arrays have different lengths: {} and {}",
            first.len(),
            second.len(),
        )
        .into());
    }
    Ok(first
        .iter()
        .zip(second)
        .map(|(first, second)| first.distance(*second))
        .fold(0.0, f64::max))
}

fn max_geometric_match_error(
    first: &[glam::DVec3],
    second: &[glam::DVec3],
) -> Result<f64, Box<dyn Error>> {
    if first.len() != second.len() {
        return Err(format!(
            "point arrays have different lengths: {} and {}",
            first.len(),
            second.len(),
        )
        .into());
    }
    let mut unmatched = second.to_vec();
    let mut maximum = 0.0_f64;
    for point in first {
        let nearest = unmatched
            .iter()
            .enumerate()
            .min_by(|(_, first), (_, second)| {
                first
                    .distance_squared(*point)
                    .total_cmp(&second.distance_squared(*point))
            })
            .map(|(index, _)| index)
            .ok_or("point correspondence exhausted unexpectedly")?;
        maximum = maximum.max(point.distance(unmatched.swap_remove(nearest)));
    }
    Ok(maximum)
}
// Different volume polygonizers do not share vertex indices; nearest-point distances compare
// every sampled surface vertex in both directions without reducing the surface to bounds.
fn max_surface_correspondence_error(
    first: &[glam::DVec3],
    second: &[glam::DVec3],
) -> Result<f64, Box<dyn Error>> {
    if first.is_empty() || second.is_empty() {
        return Err("evaluated volume surface has no vertices".into());
    }
    let directed_distance = |source: &[glam::DVec3], target: &[glam::DVec3]| {
        source
            .iter()
            .map(|point| {
                target
                    .iter()
                    .map(|other| point.distance(*other))
                    .fold(f64::INFINITY, f64::min)
            })
            .fold(0.0, f64::max)
    };
    Ok(directed_distance(first, second).max(directed_distance(second, first)))
}
#[test]
fn subdivision_surface_matches_blender_for_cube_plane_and_creased_open_grid()
-> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping subdivision Blender parity; Blender is unavailable");
        return Ok(());
    };
    let plane = potter_core::geom::Mesh::from_positions_and_faces(
        vec![
            DVec3::new(-1.5, -0.75, 0.0),
            DVec3::new(1.5, -0.75, 0.0),
            DVec3::new(1.5, 0.75, 0.0),
            DVec3::new(-1.5, 0.75, 0.0),
        ],
        vec![vec![0, 1, 2, 3]],
    )?;
    let cube = potter_core::geom::Mesh::from_positions_and_faces(
        vec![
            DVec3::new(-1.0, -1.5, -0.7),
            DVec3::new(1.0, -1.5, -0.7),
            DVec3::new(1.0, 1.5, -0.7),
            DVec3::new(-1.0, 1.5, -0.7),
            DVec3::new(-1.0, -1.5, 0.7),
            DVec3::new(1.0, -1.5, 0.7),
            DVec3::new(1.0, 1.5, 0.7),
            DVec3::new(-1.0, 1.5, 0.7),
        ],
        vec![
            vec![0, 3, 2, 1],
            vec![4, 5, 6, 7],
            vec![0, 1, 5, 4],
            vec![1, 2, 6, 5],
            vec![2, 3, 7, 6],
            vec![3, 0, 4, 7],
        ],
    )?;
    let mut open_grid = potter_core::geom::Mesh::grid(potter_core::geom::GridParams {
        size_x: 2.0,
        size_y: 1.5,
        x_subdivisions: 3,
        y_subdivisions: 3,
    })?;
    for vertex in &mut open_grid.vertices {
        vertex.co.z = 0.15 * vertex.co.x * vertex.co.x + 0.1 * vertex.co.x * vertex.co.y;
    }
    let centerline_crease = open_grid
        .edges
        .iter()
        .find(|edge| {
            let first = open_grid
                .vertices
                .iter()
                .find(|vertex| vertex.id == edge.vertices[0])
                .map(|vertex| vertex.co);
            let second = open_grid
                .vertices
                .iter()
                .find(|vertex| vertex.id == edge.vertices[1])
                .map(|vertex| vertex.co);
            matches!((first, second), (Some(first), Some(second))
                if first.y.abs() <= f64::EPSILON
                    && second.y.abs() <= f64::EPSILON
                    && (first.x - second.x).abs() > f64::EPSILON)
        })
        .ok_or("creased grid has no centerline edge")?;
    let mut crease_values = Map::new();
    crease_values.insert(format!("e{}", centerline_crease.id), json!(0.85));
    open_grid.attributes.insert(
        "crease_edge".to_owned(),
        json!({"domain":"edge","type":"float","values":crease_values}),
    );

    let cases = vec![
        (
            "plane_level_one",
            plane.clone(),
            false,
            json!({
                "levels":1,
                "render_levels":2,
                "use_limit_surface":true,
                "boundary_smooth":"ALL",
                "quality":4,
                "uv_smooth":"PRESERVE_BOUNDARIES",
                "use_creases":false
            }),
        ),
        (
            "plane",
            plane,
            true,
            json!({
                "levels":1,
                "render_levels":2,
                "use_limit_surface":true,
                "boundary_smooth":"ALL",
                "quality":4,
                "uv_smooth":"PRESERVE_BOUNDARIES",
                "use_creases":false
            }),
        ),
        (
            "cube",
            cube,
            false,
            json!({
                "levels":2,
                "render_levels":1,
                "use_limit_surface":false,
                "boundary_smooth":"PRESERVE_CORNERS",
                "quality":6,
                "uv_smooth":"NONE",
                "use_creases":false
            }),
        ),
        (
            "open_grid",
            open_grid,
            false,
            json!({
                "levels":1,
                "render_levels":2,
                "use_limit_surface":true,
                "boundary_smooth":"PRESERVE_CORNERS",
                "quality":5,
                "uv_smooth":"PRESERVE_BOUNDARIES",
                "use_creases":true
            }),
        ),
    ];
    let mut inputs = Map::new();
    for (name, mesh, use_render_levels, params) in &cases {
        let vertex_indices = mesh
            .vertices
            .iter()
            .enumerate()
            .map(|(index, vertex)| (vertex.id, index))
            .collect::<std::collections::BTreeMap<_, _>>();
        let vertices = mesh
            .vertices
            .iter()
            .map(|vertex| vertex.co.to_array())
            .collect::<Vec<_>>();
        let faces = mesh
            .faces
            .iter()
            .map(|face| {
                face.vertices
                    .iter()
                    .map(|id| {
                        vertex_indices.get(id).copied().ok_or_else(|| {
                            std::io::Error::other(
                                "subdivision fixture face references a missing vertex",
                            )
                            .into()
                        })
                    })
                    .collect::<Result<Vec<_>, Box<dyn Error>>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let crease_attribute = mesh
            .attributes
            .get("crease_edge")
            .and_then(|attribute| attribute.get("values"))
            .and_then(Value::as_object);
        let mut creases = Vec::new();
        for edge in &mesh.edges {
            let Some(value) = crease_attribute
                .and_then(|values| values.get(&format!("e{}", edge.id)))
                .and_then(Value::as_f64)
            else {
                continue;
            };
            creases.push(json!([
                vertex_indices[&edge.vertices[0]],
                vertex_indices[&edge.vertices[1]],
                value
            ]));
        }
        inputs.insert(
            (*name).to_owned(),
            json!({
                "vertices":vertices,
                "faces":faces,
                "creases":creases,
                "params":params,
                "use_render_levels":use_render_levels
            }),
        );
    }

    let directory = tempdir()?;
    let root = directory.path();
    let script = root.join("subdivision_parity.py");
    let input_path = root.join("subdivision_inputs.json");
    let expected_path = root.join("subdivision_expected.json");
    fs::write(&input_path, serde_json::to_vec(&Value::Object(inputs))?)?;
    fs::write(
        &script,
        r"
import bpy, json, os, sys
root = sys.argv[sys.argv.index('--') + 1]
with open(os.path.join(root, 'subdivision_inputs.json'), encoding='utf-8') as source:
    inputs = json.load(source)
results = {}
for name, spec in inputs.items():
    mesh = bpy.data.meshes.new(name + 'Mesh')
    mesh.from_pydata(spec['vertices'], [], spec['faces'])
    mesh.update()
    if spec['creases']:
        crease = mesh.attributes.new(name='crease_edge', type='FLOAT', domain='EDGE')
        values = {tuple(sorted(item[:2])):item[2] for item in spec['creases']}
        for edge in mesh.edges:
            crease.data[edge.index].value = values.get(tuple(sorted(edge.vertices)), 0.0)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    params = spec['params']
    modifier = obj.modifiers.new('Subdivision', 'SUBSURF')
    modifier.subdivision_type = 'CATMULL_CLARK'
    modifier.levels = params['render_levels'] if spec['use_render_levels'] else params['levels']
    modifier.render_levels = params['render_levels']
    modifier.use_limit_surface = params['use_limit_surface']
    modifier.boundary_smooth = params['boundary_smooth']
    modifier.quality = params['quality']
    modifier.uv_smooth = params['uv_smooth']
    modifier.use_creases = params['use_creases']
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    evaluated = obj.evaluated_get(depsgraph)
    result_mesh = evaluated.to_mesh()
    results[name] = {
        'points':[list(vertex.co) for vertex in result_mesh.vertices],
        'face_count':len(result_mesh.polygons),
    }
    evaluated.to_mesh_clear()
with open(os.path.join(root, 'subdivision_expected.json'), 'w', encoding='utf-8') as output:
    json.dump(results, output)
",
    )?;
    run(
        Command::new(&blender)
            .args(["--background", "--factory-startup", "--python"])
            .arg(&script)
            .arg("--")
            .arg(root),
        "subdivision fixture",
    )?;
    let expected: Value = serde_json::from_slice(&fs::read(expected_path)?)?;
    for (name, mesh, use_render_levels, params) in cases {
        let params: Map<String, Value> = serde_json::from_value(params)?;
        let node_id = potter_core::model::Id::new(name.to_owned())?;
        let data_id = potter_core::model::Id::new(format!("subdivision_{name}"))?;
        let modifier = potter_core::model::Modifier {
            id: potter_core::model::Id::new("subdivision".to_owned())?,
            modifier_type: "subdivision".to_owned(),
            name: "Subdivision".to_owned(),
            enabled: true,
            show_viewport: true,
            show_render: true,
            params,
            binding_data: None,
            runtime: potter_core::model::ModifierRuntime::default(),
        };
        let mut doc =
            potter_core::model::SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        doc.data_blocks.insert(
            data_id.clone(),
            potter_core::model::DataBlock {
                mesh: Some(mesh.clone()),
                ..potter_core::model::DataBlock::default()
            },
        );
        doc.nodes.insert(
            node_id.clone(),
            potter_core::model::Node {
                name: name.to_owned(),
                kind: "mesh".to_owned(),
                data: Some(data_id),
                modifiers: vec![modifier],
                ..potter_core::model::Node::default()
            },
        );
        doc.collections
            .get_mut(&potter_core::model::Id::new("collection_root".to_owned())?)
            .ok_or("root collection is missing")?
            .objects
            .push(node_id.clone());
        let snapshot = potter_core::eval::Snapshot::evaluate(
            &doc,
            &potter_core::eval::EvaluationContext {
                purpose: if use_render_levels {
                    potter_core::eval::EvaluationPurpose::Render
                } else {
                    potter_core::eval::EvaluationPurpose::Preview
                },
                ..potter_core::eval::EvaluationContext::default()
            },
        )?;
        let evaluated = snapshot
            .meshes
            .get(&node_id)
            .ok_or("evaluated subdivision mesh is missing")?;
        let blender_positions = blender_points(&expected[name]["points"])?;
        assert_eq!(
            evaluated.vertices.len(),
            blender_positions.len(),
            "{name} vertex count"
        );
        assert_eq!(
            evaluated.faces.len(),
            expected[name]["face_count"],
            "{name} face count",
        );
        let potter_positions = evaluated
            .vertices
            .iter()
            .map(|vertex| vertex.co)
            .collect::<Vec<_>>();
        let error = max_geometric_match_error(&potter_positions, &blender_positions)?;
        assert!(
            error <= SUBDIVISION_POSITION_TOLERANCE,
            "{name} per-vertex error {error} exceeds {SUBDIVISION_POSITION_TOLERANCE}; Potter={potter_positions:?}; Blender={blender_positions:?}",
        );
        let original_positions = mesh
            .vertices
            .iter()
            .map(|vertex| vertex.co)
            .collect::<Vec<_>>();
        let stimulus = blender_positions
            .iter()
            .map(|point| {
                original_positions
                    .iter()
                    .map(|original| point.distance(*original))
                    .fold(f64::INFINITY, f64::min)
            })
            .fold(0.0, f64::max);
        assert!(
            stimulus > 10.0 * SUBDIVISION_POSITION_TOLERANCE,
            "{name} subdivision stimulus is too close to the unmodified mesh: {stimulus}",
        );
    }
    Ok(())
}
#[test]

fn subdivision_and_cloth_stack_order_match_blender_at_frame_one() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping cloth stack Blender parity; Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let script_path = root.join("cloth_stack.py");
    let expected_path = root.join("cloth_stack.json");
    fs::write(
        &script_path,
        r"
import bpy, json, os, sys
root = sys.argv[sys.argv.index('--') + 1]
scene = bpy.context.scene
scene.frame_start = 1

def add_cloth(name, subdivision_first):
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0,0,0))
    obj = bpy.context.object
    obj.name = name
    obj.data.vertices[1].co.z = 0.4
    obj.data.vertices[3].co.z = -0.2
    group = obj.vertex_groups.new(name='Pin')
    group.add([0,2], 1.0, 'REPLACE')
    if subdivision_first:
        subd = obj.modifiers.new('Subdivide before cloth', 'SUBSURF')
        subd.levels = 1
    cloth = obj.modifiers.new('Cloth', 'CLOTH')
    cloth.settings.quality = 1
    cloth.settings.mass = 1.0
    cloth.settings.air_damping = 0.01
    cloth.settings.tension_stiffness = 12.0
    cloth.settings.compression_stiffness = 12.0
    cloth.settings.shear_stiffness = 5.0
    cloth.settings.bending_stiffness = 0.5
    cloth.settings.vertex_group_mass = group.name
    if not subdivision_first:
        subd = obj.modifiers.new('Subdivide after cloth', 'SUBSURF')
        subd.levels = 1
    return obj

before = add_cloth('SubdivisionFirst', True)
after = add_cloth('ClothFirst', False)
def scene_stats():
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    result = {}
    for key, obj in [('subdivision_first', before), ('cloth_first', after)]:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        points = [evaluated.matrix_world @ vertex.co for vertex in mesh.vertices]
        result[key] = {
            'vertex_count':len(mesh.vertices),
            'face_count':len(mesh.polygons),
            'points':[list(point) for point in points],
            'centroid':[sum(point[axis] for point in points)/len(points) for axis in range(3)],
            'bounds_min_z':min(point[2] for point in points),
            'bounds_max_z':max(point[2] for point in points),
        }
        evaluated.to_mesh_clear()
    return result

scene.frame_set(1)
frame_one = scene_stats()
for frame in range(2, 10):
    scene.frame_set(frame)
    scene_stats()
frame_nine = scene_stats()
with open(os.path.join(root, 'cloth_stack.json'), 'w', encoding='utf-8') as output:
    json.dump({'frame_one':frame_one,'frame_nine':frame_nine}, output)
",
    )?;
    let blender_output = Command::new(&blender)
        .args(["--background", "--factory-startup", "--python"])
        .arg(&script_path)
        .arg("--")
        .arg(root)
        .output()?;
    if !blender_output.status.success() || !expected_path.is_file() {
        return Err(format!(
            "Blender cloth stack fixture failed: stdout={} stderr={}",
            String::from_utf8_lossy(&blender_output.stdout),
            String::from_utf8_lossy(&blender_output.stderr),
        )
        .into());
    }
    let blender_values: Value = serde_json::from_slice(&fs::read(expected_path)?)?;
    let scene = root.join("cloth-stack-project");
    let batch = root.join("cloth-operations.json");
    initialize(&scene)?;
    apply(
        &scene,
        &batch,
        &json!([
            {"op":"node.create","id":"a_subdivision_first","kind":"plane","params":{}},
            {"op":"mesh.transform_elements","target":{"id":"a_subdivision_first"},"elements":{"domain":"vertex","ids":["v1"]},"translation":[0.0,0.0,0.4]},
            {"op":"mesh.transform_elements","target":{"id":"a_subdivision_first"},"elements":{"domain":"vertex","ids":["v3"]},"translation":[0.0,0.0,-0.2]},
            {"op":"vertex_group.create","target":{"id":"a_subdivision_first"},"id":"pin","name":"Pin"},
            {"op":"vertex_group.assign","target":{"id":"a_subdivision_first"},"group_id":"pin","weights":[{"vertex_id":0,"weight":1.0},{"vertex_id":2,"weight":1.0}]},
            {"op":"modifier.create","target":{"id":"a_subdivision_first"},"id":"subd","type":"subdivision","params":{"levels":1}},
            {"op":"physics.cloth.create","target":{"id":"a_subdivision_first"},"settings":{"vertex_group_mass":"Pin","quality":1,"tension_stiffness":12.0,"compression_stiffness":12.0,"mass":1.0,"air_damping":0.01,"shear_stiffness":5.0,"bending_stiffness":0.5}},
            {"op":"node.create","id":"b_cloth_first","kind":"plane","params":{}},
            {"op":"mesh.transform_elements","target":{"id":"b_cloth_first"},"elements":{"domain":"vertex","ids":["v1"]},"translation":[0.0,0.0,0.4]},
            {"op":"mesh.transform_elements","target":{"id":"b_cloth_first"},"elements":{"domain":"vertex","ids":["v3"]},"translation":[0.0,0.0,-0.2]},
            {"op":"vertex_group.create","target":{"id":"b_cloth_first"},"id":"pin","name":"Pin"},
            {"op":"vertex_group.assign","target":{"id":"b_cloth_first"},"group_id":"pin","weights":[{"vertex_id":0,"weight":1.0},{"vertex_id":2,"weight":1.0}]},
            {"op":"physics.cloth.create","target":{"id":"b_cloth_first"},"settings":{"vertex_group_mass":"Pin","quality":1,"tension_stiffness":12.0,"compression_stiffness":12.0,"mass":1.0,"air_damping":0.01,"shear_stiffness":5.0,"bending_stiffness":0.5}},
            {"op":"modifier.create","target":{"id":"b_cloth_first"},"id":"subd","type":"subdivision","params":{"levels":1}}
        ]),
        0,
    )?;
    let document: potter_core::model::SceneDoc =
        serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let snapshot = potter_core::eval::Snapshot::evaluate(
        &document,
        &potter_core::eval::EvaluationContext {
            frame: Some(1.0),
            ..potter_core::eval::EvaluationContext::default()
        },
    )?;
    for (potter_id, blender_key) in [
        ("a_subdivision_first", "subdivision_first"),
        ("b_cloth_first", "cloth_first"),
    ] {
        let mesh = snapshot
            .meshes
            .get(&potter_core::model::Id::new(potter_id.to_owned())?)
            .ok_or("evaluated cloth mesh is missing")?;
        let expected = &blender_values["frame_one"][blender_key];
        assert_eq!(
            mesh.vertices.len(),
            expected["vertex_count"],
            "{potter_id} frame-one vertices"
        );
        assert_eq!(
            mesh.faces.len(),
            expected["face_count"],
            "{potter_id} frame-one faces"
        );
        let actual_points = mesh
            .vertices
            .iter()
            .map(|vertex| vertex.co)
            .collect::<Vec<_>>();
        let blender_positions = blender_points(&expected["points"])?;
        let rest_error = max_pointwise_distance(&actual_points, &blender_positions)?;
        assert!(
            rest_error <= 1.0e-5,
            "{potter_id} rest-frame vertex error {rest_error}; Potter={actual_points:?}; Blender={blender_positions:?}",
        );
        let centroid_z = mean_z(mesh)?;
        let blender_centroid = expected["centroid"][2]
            .as_f64()
            .ok_or("Blender cloth centroid is missing")?;
        assert!(
            (centroid_z - blender_centroid).abs() <= 0.02,
            "{potter_id} frame-one centroid: Potter={centroid_z}, Blender={blender_centroid}",
        );
    }
    let snapshot_nine = potter_core::eval::Snapshot::evaluate(
        &document,
        &potter_core::eval::EvaluationContext {
            frame: Some(9.0),
            ..potter_core::eval::EvaluationContext::default()
        },
    )?;
    // Blender's implicit cloth solver is not numerically equivalent to Potter's XPBD
    // constraints. Apply the fixture-specific aggregate tolerances above while retaining
    // exact rest-state/pin checks and independent stack-order sensitivity.
    for (potter_id, blender_key) in [
        ("a_subdivision_first", "subdivision_first"),
        ("b_cloth_first", "cloth_first"),
    ] {
        let mesh = snapshot_nine
            .meshes
            .get(&potter_core::model::Id::new(potter_id.to_owned())?)
            .ok_or("frame-nine cloth mesh is missing")?;
        let expected = &blender_values["frame_nine"][blender_key];
        assert_eq!(mesh.vertices.len(), expected["vertex_count"]);
        assert_eq!(mesh.faces.len(), expected["face_count"]);
        let actual_points = mesh
            .vertices
            .iter()
            .map(|vertex| vertex.co)
            .collect::<Vec<_>>();
        let blender_positions = blender_points(&expected["points"])?;
        let frame_one_positions =
            blender_points(&blender_values["frame_one"][blender_key]["points"])?;
        let baseline_displacement =
            max_pointwise_distance(&blender_positions, &frame_one_positions)?;
        assert!(
            baseline_displacement > 10.0 * CLOTH_STIMULUS_TOLERANCE,
            "{potter_id} fixture must produce nontrivial cloth motion, observed {baseline_displacement}",
        );
        let (rms_error, maximum_error) =
            pointwise_error_metrics(&actual_points, &blender_positions)?;
        assert!(
            rms_error <= CLOTH_RMS_TOLERANCE,
            "{potter_id} frame-nine per-vertex RMS={rms_error}, max={maximum_error}; RMS tolerance={CLOTH_RMS_TOLERANCE}",
        );
        let blender_centroid: [f64; 3] = serde_json::from_value(expected["centroid"].clone())?;
        let centroid_error =
            centroid(&actual_points)?.distance(glam::DVec3::from_array(blender_centroid));
        assert!(
            centroid_error <= CLOTH_CENTROID_TOLERANCE,
            "{potter_id} frame-nine centroid error {centroid_error} exceeds {CLOTH_CENTROID_TOLERANCE}",
        );
        let actual_min_z = actual_points
            .iter()
            .map(|point| point.z)
            .fold(f64::INFINITY, f64::min);
        let actual_max_z = actual_points
            .iter()
            .map(|point| point.z)
            .fold(f64::NEG_INFINITY, f64::max);
        let minimum_z = expected["bounds_min_z"]
            .as_f64()
            .ok_or("Blender cloth minimum Z bound is missing")?;
        let maximum_z = expected["bounds_max_z"]
            .as_f64()
            .ok_or("Blender cloth maximum Z bound is missing")?;
        let bounds_error = (actual_min_z - minimum_z)
            .abs()
            .max((actual_max_z - maximum_z).abs());
        assert!(
            bounds_error <= CLOTH_BOUNDS_TOLERANCE,
            "{potter_id} frame-nine Z-bounds error {bounds_error} exceeds {CLOTH_BOUNDS_TOLERANCE}",
        );
        if potter_id == "a_subdivision_first" {
            let frame_one_mesh =
                &snapshot.meshes[&potter_core::model::Id::new(potter_id.to_owned())?];
            let pinned_vertex_error = actual_points[0].distance(frame_one_mesh.vertices[0].co);
            assert!(
                pinned_vertex_error <= 1.0e-8,
                "subdivision-first pinned vertex moved {pinned_vertex_error}",
            );
        }
    }
    let blender_first =
        blender_points(&blender_values["frame_nine"]["subdivision_first"]["points"])?;
    let blender_second = blender_points(&blender_values["frame_nine"]["cloth_first"]["points"])?;
    let blender_order_difference = max_pointwise_distance(&blender_first, &blender_second)?;
    assert!(
        blender_order_difference > CLOTH_ORDER_TOLERANCE,
        "Blender fixture must make stack order observable, observed {blender_order_difference}",
    );
    let first =
        &snapshot_nine.meshes[&potter_core::model::Id::new("a_subdivision_first".to_owned())?];
    let second = &snapshot_nine.meshes[&potter_core::model::Id::new("b_cloth_first".to_owned())?];
    let first_points = first
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    let second_points = second
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    let potter_order_difference = max_pointwise_distance(&first_points, &second_points)?;
    assert!(
        potter_order_difference > CLOTH_ORDER_TOLERANCE,
        "Potter stack order must change evaluated positions, observed {potter_order_difference}",
    );
    let catalog = potter_core::catalog::feature_catalog();
    let cloth_parity = catalog["features"]
        .as_array()
        .and_then(|features| {
            features
                .iter()
                .find(|feature| feature["feature_id"] == "physics.cloth.blender_solver_parity")
        })
        .ok_or("Blender cloth-solver parity is missing from the capability catalog")?;
    assert_eq!(cloth_parity["status"], "not_supported");
    Ok(())
}
fn inspect(scene: &Path, object: &str, frame: &str) -> Result<Value, Box<dyn Error>> {
    let output = run(
        pot()
            .arg("inspect")
            .arg(scene)
            .arg("--id")
            .arg(object)
            .arg("--frame")
            .arg(frame)
            .arg("--json"),
        "modifier inspection",
    )?;
    Ok(serde_json::from_slice(&output.stdout)?)
}

#[test]
fn indexed_particles_lifecycle_children_and_bake_are_evaluated() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("particle-lifecycle");
    let batch = directory.path().join("operations.json");
    let bake_directory = directory.path().join("bake");
    initialize(&scene)?;
    apply(
        &scene,
        &batch,
        &json!([
            {"op":"node.create","id":"a_emitter","kind":"plane","params":{}},
            {"op":"physics.particle_emitter.create","target":{"id":"a_emitter"},"settings":{"emit_from":"VERT","count":2,"frame_start":1.0,"frame_end":2.0,"lifetime":1.0,"physics_type":"NO","normal_factor":0.0,"particle_size":0.75,"child_radius":0.6,"child_type":"SIMPLE","child_seed":31,"child_percent":2,"rendered_child_count":2,"seed":11}},
            {"op":"modifier.create","target":{"id":"a_emitter"},"id":"particles_second","type":"particle_system","params":{"settings_id":"particles_second"}},
            {"op":"node.create","id":"b_alive","kind":"plane","params":{}},
            {"op":"modifier.create","target":{"id":"b_alive"},"id":"alive_instances","type":"particle_instance","params":{"object":"a_emitter","particle_system_index":1,"use_normal":true,"use_children":true,"use_size":false,"show_alive":true,"show_dead":false,"show_unborn":false,"position":0,"random_position":0.0,"axis":"X","space":"WORLD","use_path":false}},
            {"op":"node.create","id":"c_secondary","kind":"plane","params":{}},
            {"op":"modifier.create","target":{"id":"c_secondary"},"id":"secondary_instances","type":"particle_instance","params":{"object":"a_emitter","particle_system_index":2,"use_normal":true,"use_children":false,"use_size":false,"show_alive":true,"show_dead":false,"show_unborn":false,"position":0,"random_position":0.0,"axis":"X","space":"WORLD","use_path":false}},
            {"op":"node.create","id":"d_no_unborn","kind":"plane","params":{}},
            {"op":"modifier.create","target":{"id":"d_no_unborn"},"id":"without_unborn","type":"particle_instance","params":{"object":"a_emitter","particle_system_index":1,"use_normal":true,"use_children":false,"use_size":false,"show_alive":true,"show_dead":false,"show_unborn":false,"position":0,"random_position":0.0,"axis":"X","space":"WORLD","use_path":false}},
            {"op":"node.create","id":"g_with_unborn","kind":"plane","params":{}},
            {"op":"modifier.create","target":{"id":"g_with_unborn"},"id":"with_unborn","type":"particle_instance","params":{"object":"a_emitter","particle_system_index":1,"use_normal":true,"use_children":false,"use_size":false,"show_alive":true,"show_dead":false,"show_unborn":true,"position":0,"random_position":0.0,"axis":"X","space":"WORLD","use_path":false}},
            {"op":"node.create","id":"e_dead","kind":"plane","params":{}},
            {"op":"modifier.create","target":{"id":"e_dead"},"id":"dead_instances","type":"particle_instance","params":{"object":"a_emitter","particle_system_index":1,"use_normal":true,"use_children":true,"use_size":true,"show_alive":false,"show_dead":true,"show_unborn":false,"position":0,"random_position":0.0,"axis":"X","space":"WORLD","use_path":false}},
            {"op":"node.create","id":"f_exploder","kind":"box","params":{}},
            {"op":"physics.particle_emitter.create","target":{"id":"f_exploder"},"settings":{"emit_from":"VERT","count":1,"frame_start":1.0,"frame_end":1.0,"lifetime":10.0,"physics_type":"NO","normal_factor":0.0,"seed":13}},
            {"op":"modifier.create","target":{"id":"f_exploder"},"id":"explode_faces","type":"explode","params":{"use_edge_cut":true,"show_alive":true,"show_dead":false,"show_unborn":false,"use_size":false,"protect":0.0}}
        ]),
        0,
    )?;
    let stored: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let child_settings = &stored["nodes"]["a_emitter"]["properties"]["physics_particle_emitter"];
    assert_eq!(child_settings["child_seed"], 31);
    assert_eq!(child_settings["child_percent"], 2);
    assert_eq!(child_settings["rendered_child_count"], 2);

    for (object, frame, expected_faces) in [
        ("b_alive", "1", 3),
        ("c_secondary", "1", 1),
        ("d_no_unborn", "1", 3),
        ("g_with_unborn", "1", 6),
        ("e_dead", "3", 3),
        ("f_exploder", "1", 6),
    ] {
        let response = inspect(&scene, object, frame)?;
        assert_eq!(
            response["result"]["items"][0]["evaluated_geometry"]["face_count"], expected_faces,
            "{object} at frame {frame}: {response}",
        );
    }

    let bake_output = run(
        pot()
            .arg("bake")
            .arg(&scene)
            .arg("--kind")
            .arg("simulation")
            .arg("--frames")
            .arg("1:3")
            .arg("--out")
            .arg(&bake_directory)
            .arg("--json"),
        "particle lifecycle bake",
    )?;
    let response: Value = serde_json::from_slice(&bake_output.stdout)?;
    assert_eq!(response["result"]["frame_count"], 3);
    let frame_one: Value =
        serde_json::from_slice(&fs::read(bake_directory.join("frame_00000000.json"))?)?;
    let frame_three: Value =
        serde_json::from_slice(&fs::read(bake_directory.join("frame_00000002.json"))?)?;
    let frame_one_states = frame_one["particles"]["a_emitter"]
        .as_array()
        .ok_or("frame-one particle states are missing")?;
    assert_eq!(frame_one_states.len(), 6);
    assert!(
        frame_one_states
            .iter()
            .any(|state| state["is_child"] == true),
        "baked particle states must retain native child instances",
    );
    assert_eq!(
        frame_one_states
            .iter()
            .filter(|state| state["life_state"] == "unborn")
            .count(),
        3,
    );
    let frame_three_states = frame_three["particles"]["a_emitter"]
        .as_array()
        .ok_or("frame-three particle states are missing")?;
    assert!(
        frame_three_states
            .iter()
            .any(|state| state["life_state"] == "dead"),
        "expired particle state must survive in the bake cache",
    );
    Ok(())
}

#[test]
fn volume_displace_resamples_inline_density_with_object_texture_mapping()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("volume-displacement");
    let batch = directory.path().join("operations.json");
    initialize(&scene)?;
    apply(
        &scene,
        &batch,
        &json!([
            {"op":"image.create","id":"displace_texture","name":"Displacement","width":2,"height":1,"colorspace":"linear","fill_color":[0.0,0.5,0.5,1.0]},
            {"op":"image.set_pixels","id":"displace_texture","x":1,"width":1,"height":1,"pixels":[[1.0,0.5,0.5,1.0]]},
            {"op":"node.create","id":"texture_frame","kind":"plane","params":{},"transform":{"translation":[-0.25,0.0,0.0]}},
            {"op":"volume.create","id":"density","grids":[{"dims":[3,1,1],"voxel_size":0.5,"origin":[0.0,0.0,0.0],"values":[0.0,1.0,0.0]}]},
            {"op":"modifier.create","target":{"id":"density"},"id":"displace","type":"volume_displace","params":{"texture":"displace_texture","texture_map_mode":"OBJECT","texture_map_object":"texture_frame","strength":0.5,"texture_mid_level":[0.5,0.5,0.5],"texture_sample_radius":0.0}}
        ]),
        0,
    )?;
    let document: potter_core::model::SceneDoc =
        serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let node = document
        .nodes
        .get(&potter_core::model::Id::new("density".to_owned())?)
        .ok_or("volume object is missing")?;
    let data_id = node.data.as_ref().ok_or("volume Data-Block is missing")?;
    let snapshot = potter_core::eval::Snapshot::evaluate_with_cache(
        &document,
        &potter_core::eval::EvaluationContext::default(),
        Some(&scene),
        Some(&scene.join(".potter/cache")),
    )?;
    let values = snapshot
        .volume_data
        .get(data_id)
        .and_then(|volume| volume.grids.first())
        .and_then(|grid| grid.values.as_deref())
        .ok_or("displaced volume samples are missing")?;
    assert_eq!(values, [0.0_f32, 0.75, 0.0]);
    Ok(())
}

#[test]
fn volume_displace_resamples_decoded_vdb_density() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping VDB Volume Displace test; Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let script = root.join("create_density.py");
    let vdb_path = root.join("density.vdb");
    let blender_expected_path = root.join("blender_expected.json");
    fs::write(
        &script,
        r"
import bpy, openvdb, json, os, sys, math
from mathutils import Vector
root = os.path.realpath(sys.argv[sys.argv.index('--') + 1])
grid = openvdb.FloatGrid()
grid.name = 'density'
accessor = grid.getAccessor()
for z in range(3):
    for y in range(3):
        for x in range(4):
            density = 0.15 + 0.16*x + 0.09*y + 0.07*z + 0.045*x*y - 0.025*x*z
            accessor.setValueOn((x,y,z), density)
vdb_path = os.path.join(root, 'density.vdb')
openvdb.write(vdb_path, grids=[grid])
bpy.ops.object.volume_import(filepath=vdb_path)
volume = bpy.context.object
image = bpy.data.images.new('Displacement', width=4, height=3, alpha=True, float_buffer=True)
pixels = []
for y in range(3):
    for x in range(4):
        pixels.extend([x/3, y/2, ((2*x+y) % 4)/3, 1.0])
image.pixels[:] = pixels
texture = bpy.data.textures.new('Displacement', type='IMAGE')
texture.image = image
mapping_mesh = bpy.data.meshes.new('mapping_object_mesh')
mapping_object = bpy.data.objects.new('mapping_object', mapping_mesh)
bpy.context.scene.collection.objects.link(mapping_object)
mapping_object.location = (-0.25, 0.2, -0.1)
mapping_object.rotation_euler = (0.2, 0.35, -0.15)
mapping_object.scale = (1.3, 0.8, 1.1)
displace = volume.modifiers.new('Volume Displace', 'VOLUME_DISPLACE')
displace.texture = texture
displace.texture_map_mode = 'OBJECT'
displace.texture_map_object = mapping_object
displace.strength = 3.0
displace.texture_mid_level = (0.5, 0.5, 0.5)
displace.texture_sample_radius = 1.0
baseline_volume = bpy.data.objects.new('BaselineVolume', volume.data)
bpy.context.scene.collection.objects.link(baseline_volume)

def make_surface(name, source_volume):
    surface = bpy.data.objects.new(name, bpy.data.meshes.new(name + 'Mesh'))
    bpy.context.scene.collection.objects.link(surface)
    to_mesh = surface.modifiers.new('Volume to Mesh', 'VOLUME_TO_MESH')
    to_mesh.object = source_volume
    to_mesh.threshold = 0.25
    to_mesh.resolution_mode = 'GRID'
    return surface

surface = make_surface('DisplacedSurface', volume)
baseline_surface = make_surface('BaselineSurface', baseline_volume)
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
depsgraph.update()
evaluated = surface.evaluated_get(depsgraph)
evaluated_mesh = evaluated.to_mesh()
points = [evaluated.matrix_world @ vertex.co for vertex in evaluated_mesh.vertices]
baseline_evaluated = baseline_surface.evaluated_get(depsgraph)
baseline_mesh = baseline_evaluated.to_mesh()
baseline_points = [
    baseline_evaluated.matrix_world @ vertex.co for vertex in baseline_mesh.vertices
]
mapping_world = [
    mapping_object.matrix_world[row][column]
    for column in range(4)
    for row in range(4)
]
def sample_density(coordinate):
    lower = [math.floor(value) for value in coordinate]
    fraction = [coordinate[axis] - lower[axis] for axis in range(3)]
    value = 0.0
    for z_offset in range(2):
        for y_offset in range(2):
            for x_offset in range(2):
                weight = (
                    (fraction[0] if x_offset else 1.0-fraction[0]) *
                    (fraction[1] if y_offset else 1.0-fraction[1]) *
                    (fraction[2] if z_offset else 1.0-fraction[2])
                )
                value += accessor.getValue((
                    lower[0] + x_offset,
                    lower[1] + y_offset,
                    lower[2] + z_offset,
                )) * weight
    return value

mapping_world_to_local = mapping_object.matrix_world.inverted()
volume_to_world = volume.matrix_world
blender_samples = []
for z in range(-2,5):
    for y in range(-2,5):
        for x in range(-2,6):
            index = Vector((x,y,z))
            texture_position = mapping_world_to_local @ (volume_to_world @ index)
            color = texture.evaluate(texture_position)
            displacement = Vector((
                (color[0]-0.5)*3.0,
                (color[1]-0.5)*3.0,
                (color[2]-0.5)*3.0,
            ))
            source = index-displacement
            sample = sample_density(source)
            blender_samples.append(sample)
evaluated_volume = volume.evaluated_get(depsgraph)
evaluated_grid_names = [grid.name for grid in evaluated_volume.data.grids]
expected = {
    'points':[list(point) for point in points],
    'baseline_points':[list(point) for point in baseline_points],
    'mapping_world':mapping_world,
    'vdb_samples':blender_samples,
    'evaluated_grid_names':evaluated_grid_names,
}
with open(os.path.join(root, 'blender_expected.json'), 'w', encoding='utf-8') as output:
    json.dump(expected, output)
evaluated.to_mesh_clear()
baseline_evaluated.to_mesh_clear()
        ",
    )?;
    run(
        Command::new(&blender)
            .args(["--background", "--factory-startup", "--python"])
            .arg(&script)
            .arg("--")
            .arg(root),
        "VDB density fixture generation",
    )?;
    let blender_expected: Value = serde_json::from_slice(&fs::read(&blender_expected_path)?)?;
    let expected_samples: Vec<f64> =
        serde_json::from_value(blender_expected["vdb_samples"].clone())?;
    let evaluated_grid_names: Vec<String> =
        serde_json::from_value(blender_expected["evaluated_grid_names"].clone())?;
    assert_eq!(evaluated_grid_names, ["density"]);
    let compare_density_samples = |name: &str, actual: &[f32]| {
        assert_eq!(
            actual.len(),
            expected_samples.len(),
            "{name} and Blender voxel sample counts differ",
        );
        let mut maximum_error = 0.0_f64;
        let mut maximum_index = 0;
        let mut maximum_actual = 0.0_f32;
        let mut maximum_expected = 0.0_f64;
        for (index, (actual, expected)) in actual.iter().zip(&expected_samples).enumerate() {
            let error = (f64::from(*actual) - expected).abs();
            if error > maximum_error {
                maximum_error = error;
                maximum_index = index;
                maximum_actual = *actual;
                maximum_expected = *expected;
            }
        }
        assert!(
            maximum_error <= DENSITY_SAMPLE_TOLERANCE,
            "{name} per-voxel density error {maximum_error} at sample {maximum_index} (Potter={maximum_actual}, Blender={maximum_expected}) exceeds {DENSITY_SAMPLE_TOLERANCE}",
        );
    };

    // The Potter side enters through the scene path: the VDB is packed as a resource, the volume
    // references it as File source, and the displacement modifier resolves the image texture and
    // object mapping from the scene. The mapping object uses Blender's XYZ Euler transform.
    let scene = root.join("volume-displacement-vdb");
    let batch = root.join("operations.json");
    initialize(&scene)?;
    // Blender's image rows run bottom-up (the oracle's `image.pixels` row 0 is the bottom row), while Potter's
    // storage, which `image.set_pixels` writes, runs top-down. The same visual image is therefore the Blender
    // rows in reverse order, so the Potter side passes its rows top-down.
    let texture_pixels = (0_u32..3)
        .rev()
        .flat_map(|y| {
            (0_u32..4).map(move |x| {
                [
                    f64::from(x) / 3.0,
                    f64::from(y) / 2.0,
                    f64::from((2 * x + y) % 4) / 3.0,
                    1.0,
                ]
            })
        })
        .collect::<Vec<_>>();
    let mut inline_values = Vec::with_capacity(36);
    for z in 0_u8..3 {
        for y in 0_u8..3 {
            for x in 0_u8..4 {
                let x = f32::from(x);
                let y = f32::from(y);
                let z = f32::from(z);
                inline_values.push(
                    0.15_f32 + 0.16_f32 * x + 0.09_f32 * y + 0.07_f32 * z + 0.045_f32 * x * y
                        - 0.025_f32 * x * z,
                );
            }
        }
    }
    let vdb_file_source = json!({
        "kind":"file",
        "metadata":{
            "format":"vdb",
            "content_ref":"asset://density_vdb",
            "grid_names":["density"],
        },
    });
    let displacement_params = json!({
        "texture":"displace_texture",
        "texture_map_mode":"OBJECT",
        "texture_map_object":"mapping_object",
        "strength":3.0,
        "texture_mid_level":[0.5,0.5,0.5],
        "texture_sample_radius":1.0,
    });
    apply(
        &scene,
        &batch,
        &json!([
            {"op":"resource.pack","id":"density_vdb","uri":vdb_path.to_string_lossy(),"kind":"vdb"},
            {"op":"image.create","id":"displace_texture","name":"Displacement","width":4,"height":3,"colorspace":"linear","fill_color":[0.0,0.0,0.0,1.0]},
            {"op":"image.set_pixels","id":"displace_texture","x":0,"y":0,"width":4,"height":3,"pixels":texture_pixels},
            {"op":"node.create","id":"mapping_object","kind":"plane","params":{},"transform":{"translation":[-0.25,0.2,-0.1],"rotation_deg":[0.2_f64.to_degrees(),0.35_f64.to_degrees(),(-0.15_f64).to_degrees()],"scale":[1.3,0.8,1.1]}},
            {"op":"volume.create","id":"density","source":vdb_file_source},
            {"op":"volume.create","id":"baseline_volume","source":vdb_file_source},
            {"op":"volume.create","id":"inline_density","grids":[{"dims":[4,3,3],"voxel_size":1.0,"origin":[0.0,0.0,0.0],"values":inline_values}]},
            {"op":"modifier.create","target":{"id":"density"},"id":"displace","type":"volume_displace","params":displacement_params},
            {"op":"modifier.create","target":{"id":"inline_density"},"id":"displace","type":"volume_displace","params":displacement_params}
        ]),
        0,
    )?;
    let document: potter_core::model::SceneDoc =
        serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let snapshot = potter_core::eval::Snapshot::evaluate_with_cache(
        &document,
        &potter_core::eval::EvaluationContext::default(),
        Some(&scene),
        Some(&scene.join(".potter/cache")),
    )?;
    let displaced = evaluated_volume(&document, &snapshot, "density")?;
    let baseline = evaluated_volume(&document, &snapshot, "baseline_volume")?;
    let inline_displaced = evaluated_volume(&document, &snapshot, "inline_density")?;
    assert_eq!(
        displaced.source,
        potter_core::geom::volume::VolumeSource::Generated(
            potter_core::geom::volume::VolumeGeneratedSource {
                algorithm: "volume_displace_vdb".to_owned(),
            },
        )
    );
    let vdb_grid = displaced
        .grids
        .first()
        .ok_or("displaced VDB grid is missing")?;
    assert_eq!(vdb_grid.dims, [8, 7, 7]);
    compare_density_samples(
        "decoded VDB",
        vdb_grid
            .values
            .as_deref()
            .ok_or("displaced VDB samples are missing")?,
    );
    let inline_grid = inline_displaced
        .grids
        .first()
        .ok_or("displaced inline grid is missing")?;
    assert_eq!(inline_grid.dims, [8, 7, 7]);
    compare_density_samples(
        "inline grid",
        inline_grid
            .values
            .as_deref()
            .ok_or("displaced inline samples are missing")?,
    );
    let potter_mesh = potter_core::geom::volume::volume_to_mesh(displaced, 0.25)
        .map_err(|error| std::io::Error::other(error.to_string()))?;
    let baseline_mesh = potter_core::geom::volume::volume_to_mesh(baseline, 0.25)
        .map_err(|error| std::io::Error::other(error.to_string()))?;
    let potter_points = potter_mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    let potter_baseline_points = baseline_mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    let blender_surface_points = blender_points(&blender_expected["points"])?;
    let blender_baseline = blender_points(&blender_expected["baseline_points"])?;
    let displaced_surface_distance =
        max_surface_correspondence_error(&blender_surface_points, &blender_baseline)?;
    assert!(
        displaced_surface_distance > MINIMUM_SURFACE_DISPLACEMENT,
        "spatially varying texture must materially displace the Blender surface by more than one voxel, observed {displaced_surface_distance}",
    );
    let potter_displacement =
        max_surface_correspondence_error(&potter_points, &potter_baseline_points)?;
    assert!(
        potter_displacement > MINIMUM_SURFACE_DISPLACEMENT,
        "Potter VDB surface must differ from its unmodified baseline by more than one voxel, observed {potter_displacement}",
    );
    Ok(())
}

#[test]
fn affine_inline_volume_displacement_preserves_sampling_and_mesh_space()
-> Result<(), Box<dyn Error>> {
    let index_to_local = DMat4::from_cols_array(&[
        0.0, 0.5, 0.0, 0.0, -2.0, 0.3, 0.0, 0.0, 0.0, 0.0, 1.5, 0.0, 4.0, -2.0, 3.0, 1.0,
    ]);
    let volume = potter_core::geom::volume::VolumeData {
        grids: vec![potter_core::geom::volume::VolumeGrid {
            dims: [2, 2, 2],
            voxel_size: index_to_local
                .transform_vector3(DVec3::X)
                .length()
                .max(index_to_local.transform_vector3(DVec3::Y).length())
                .max(index_to_local.transform_vector3(DVec3::Z).length()),
            origin: index_to_local.transform_point3(DVec3::ZERO),
            index_to_local: Some(index_to_local.to_cols_array()),
            values: Some(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
            ..potter_core::geom::volume::VolumeGrid::default()
        }],
        ..potter_core::geom::volume::VolumeData::default()
    };
    let affine_roundtrip: potter_core::geom::volume::VolumeData =
        serde_json::from_value(serde_json::to_value(&volume)?)?;
    assert_eq!(affine_roundtrip, volume);
    let legacy_json = serde_json::to_value(potter_core::geom::volume::VolumeGrid::default())?;
    assert!(legacy_json.get("index_to_local").is_none());
    let legacy_grid: potter_core::geom::volume::VolumeGrid = serde_json::from_value(legacy_json)?;
    assert_eq!(legacy_grid.index_to_local, None);
    let mut invalid_matrices = vec![DMat4::from_scale(DVec3::new(1.0, 0.0, 1.0)).to_cols_array()];
    let mut nonfinite_matrix = DMat4::IDENTITY.to_cols_array();
    nonfinite_matrix[0] = f64::NAN;
    invalid_matrices.push(nonfinite_matrix);
    for matrix in invalid_matrices {
        let mut invalid_volume = volume.clone();
        invalid_volume.grids[0].index_to_local = Some(matrix);
        assert!(potter_core::geom::volume::validate_data(&invalid_volume).is_err());
    }
    let sample_index = DVec3::new(0.25, 0.5, 0.75);
    // The density is recovered through the matrix inverse, so it carries f64 rounding (about 4e-16 here).
    let sampled = potter_core::geom::volume::sample_density_checked(
        index_to_local.transform_point3(sample_index),
        &volume,
    )?;
    assert!(
        (sampled - sample_index.x).abs() <= 1.0e-12,
        "affine density round trip {sampled} differs from {}",
        sample_index.x
    );

    let parameters = Map::from_iter([
        ("texture".into(), json!("affine_texture")),
        ("texture_map_mode".into(), json!("LOCAL")),
        ("strength".into(), json!(1.0)),
        ("texture_mid_level".into(), json!([0.5, 0.5, 0.5])),
        ("texture_sample_radius".into(), json!(1.0)),
    ]);
    let grids = serde_json::to_value(&volume.grids)?;
    let directory = tempdir()?;
    let scene = directory.path().join("affine-volume-displacement");
    let batch = directory.path().join("operations.json");
    initialize(&scene)?;
    apply(
        &scene,
        &batch,
        &json!([
            {"op":"image.create","id":"affine_texture","name":"Affine displacement","width":1,"height":1,"colorspace":"linear","fill_color":[1.0,0.5,0.5,1.0]},
            {"op":"volume.create","id":"affine_density","grids":grids},
            {"op":"modifier.create","target":{"id":"affine_density"},"id":"displace","type":"volume_displace","params":parameters},
            {"op":"volume.create","id":"affine_repeated","grids":grids},
            {"op":"modifier.create","target":{"id":"affine_repeated"},"id":"first","type":"volume_displace","params":parameters},
            {"op":"modifier.create","target":{"id":"affine_repeated"},"id":"second","type":"volume_displace","params":parameters}
        ]),
        0,
    )?;
    let document: potter_core::model::SceneDoc =
        serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let snapshot = potter_core::eval::Snapshot::evaluate_with_cache(
        &document,
        &potter_core::eval::EvaluationContext::default(),
        Some(&scene),
        Some(&scene.join(".potter/cache")),
    )?;
    let displaced = evaluated_volume(&document, &snapshot, "affine_density")?;
    let grid = displaced
        .grids
        .first()
        .ok_or("affine displaced grid is missing")?;
    assert_eq!(grid.dims, [4, 4, 4]);
    let expected_index_to_local = index_to_local * DMat4::from_translation(-DVec3::ONE);
    assert_eq!(
        grid.index_to_local,
        Some(expected_index_to_local.to_cols_array())
    );
    assert_eq!(
        grid.origin,
        expected_index_to_local.transform_point3(DVec3::ZERO),
    );
    let sample_position = expected_index_to_local.transform_point3(DVec3::new(2.0, 1.0, 1.0));
    // Blender's Volume Displace subtracts `(texture - mid) * strength / max_voxel_side_length` in index space
    // (MOD_volume_displace.cc DisplaceOp). The texture displacement is (0.5, 0, 0) in object space, and OpenVDB's
    // voxelSize() of this affine map is [0.5, 2.0223748, 1.5] (queried in Blender's bundled OpenVDB), so the
    // density at index 1 - 0.5 / 2.0223748 is the linear ramp value 1 - 0.5 / max voxel.
    let max_voxel_side_length = index_to_local
        .transform_vector3(DVec3::X)
        .length()
        .max(index_to_local.transform_vector3(DVec3::Y).length())
        .max(index_to_local.transform_vector3(DVec3::Z).length());
    let expected_sample = 1.0 - 0.5 / max_voxel_side_length;
    let sampled = potter_core::geom::volume::sample_density_checked(sample_position, displaced)?;
    assert!(
        // The displaced density is stored as f32 (Blender's grid is float too), so the bound is one f32 epsilon.
        (sampled - expected_sample).abs() <= f64::from(f32::EPSILON),
        "affine displaced density {sampled} differs from Blender's index-space value {expected_sample}",
    );
    let repeated = evaluated_volume(&document, &snapshot, "affine_repeated")?;
    let repeated_grid = repeated
        .grids
        .first()
        .ok_or("repeated affine displaced grid is missing")?;
    assert_eq!(repeated_grid.dims, [6, 6, 6]);
    let expected_repeated_transform =
        expected_index_to_local * DMat4::from_translation(-DVec3::ONE);
    assert_eq!(
        repeated_grid.index_to_local,
        Some(expected_repeated_transform.to_cols_array()),
    );

    let affine_mesh = potter_core::geom::volume::volume_to_mesh(displaced, 0.25)?;
    let index_space_volume = potter_core::geom::volume::VolumeData {
        grids: vec![potter_core::geom::volume::VolumeGrid {
            dims: grid.dims,
            voxel_size: 1.0,
            origin: DVec3::ZERO,
            values: grid.values.clone(),
            ..potter_core::geom::volume::VolumeGrid::default()
        }],
        ..potter_core::geom::volume::VolumeData::default()
    };
    let index_mesh = potter_core::geom::volume::volume_to_mesh(&index_space_volume, 0.25)?;
    let affine_points = affine_mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    let expected_points = index_mesh
        .vertices
        .iter()
        .map(|vertex| expected_index_to_local.transform_point3(vertex.co))
        .collect::<Vec<_>>();
    assert!(
        max_surface_correspondence_error(&affine_points, &expected_points)?
            <= MODIFIER_POSITION_TOLERANCE,
        "volume mesh vertices must use the full affine index-to-local transform",
    );
    Ok(())
}
