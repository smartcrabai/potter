use std::{collections::BTreeMap, error::Error, fs, path::Path, process::Command};

use glam::{DMat4, DQuat, DVec3};
use potter_core::{
    eval::{EvaluationContext, Snapshot},
    geom::{
        Mesh,
        modifiers::{
            attributes::{SceneOperand, evaluate_mesh_cache, evaluate_with_context},
            evaluate_modifiers,
        },
    },
    image::{ImageData, ImageInterpolation},
    model::{CameraData, Modifier, SceneDoc},
};
use serde_json::{Value, json};
use tempfile::tempdir;
#[path = "common/blender_file.rs"]
mod blender_file;

use blender_file::blender_executable;

#[expect(
    clippy::needless_pass_by_value,
    reason = "modifier test fixtures accept convenient owned json! values"
)]
fn modifier(modifier_type: &str, params: Value) -> Result<Modifier, Box<dyn Error>> {
    Ok(Modifier {
        id: potter_core::model::Id::new("test_modifier")?,
        modifier_type: modifier_type.to_owned(),
        name: modifier_type.to_owned(),
        enabled: true,
        params: params
            .as_object()
            .ok_or("modifier params must be an object")?
            .clone(),
        binding_data: None,
        runtime: potter_core::model::ModifierRuntime::default(),
    })
}

fn weighted_line() -> Result<Mesh, Box<dyn Error>> {
    let mut mesh = Mesh::from_positions_and_faces(vec![DVec3::ZERO, DVec3::X], vec![])?;
    mesh.attributes.insert(
        "vertex_groups".to_owned(),
        json!({"A":{"v0":0.25,"v1":0.75},"B":{"v0":0.5,"v1":0.25}}),
    );
    Ok(mesh)
}

fn weights(mesh: &Mesh, group: &str) -> Result<[f64; 2], Box<dyn Error>> {
    let groups = mesh.attributes["vertex_groups"]
        .as_object()
        .ok_or("vertex group table missing")?;
    let values = groups[group].as_object().ok_or("weight table missing")?;
    Ok([
        values.get("v0").and_then(Value::as_f64).unwrap_or(0.0),
        values.get("v1").and_then(Value::as_f64).unwrap_or(0.0),
    ])
}

fn scene_operand(mesh: Mesh, local_to_subject: DMat4, camera: Option<CameraData>) -> SceneOperand {
    SceneOperand {
        mesh,
        local_to_subject,
        camera,
        armature: false,
    }
}

#[test]
fn vertex_weight_mix_applies_blender_operation_to_selected_vertices() -> Result<(), Box<dyn Error>>
{
    let mesh = weighted_line()?;
    let result = evaluate_modifiers(
        &mesh,
        &[modifier(
            "vertex_weight_mix",
            json!({
                "vertex_group_a":"A",
                "vertex_group_b":"B",
                "mix_mode":"ADD",
                "mix_set":"ALL",
                "default_weight_a":0.0,
                "default_weight_b":0.0,
                "normalize":false
            }),
        )?],
    )?;
    assert_eq!(weights(&result, "A")?, [0.75, 1.0]);
    Ok(())
}

#[test]
fn vertex_weight_edit_supports_curve_mapping_and_threshold_removal() -> Result<(), Box<dyn Error>> {
    let mesh = weighted_line()?;
    let groups = mesh.attributes["vertex_groups"]
        .as_object()
        .ok_or("vertex group table missing")?;
    assert!(groups.contains_key("A"));
    let result = evaluate_modifiers(
        &mesh,
        &[modifier(
            "vertex_weight_edit",
            json!({
                "vertex_group":"A",
                "falloff_type":"CURVE",
                "map_curve":[[0.0,0.0],[1.0,1.0]],
                "use_add":false,
                "use_remove":true,
                "remove_threshold":0.5,
                "default_weight":0.0,
                "normalize":false
            }),
        )?],
    )?;
    assert_eq!(weights(&result, "A")?, [0.0, 0.75]);
    Ok(())
}

#[test]
fn weighted_normal_writes_unit_corner_normals() -> Result<(), Box<dyn Error>> {
    let mesh =
        Mesh::from_positions_and_faces(vec![DVec3::ZERO, DVec3::X, DVec3::Y], vec![vec![0, 1, 2]])?;
    let result = evaluate_modifiers(
        &mesh,
        &[modifier(
            "weighted_normal",
            json!({"mode":"FACE_AREA","weight":50}),
        )?],
    )?;
    let normals = result.attributes["custom_normal"]["values"]["f0"]
        .as_array()
        .ok_or("custom corner normals missing")?;
    assert_eq!(normals.len(), 3);
    for normal in normals {
        let values = normal.as_array().ok_or("normal must be a vector")?;
        let length = values
            .iter()
            .map(|value| value.as_f64().unwrap_or_default().powi(2))
            .sum::<f64>()
            .sqrt();
        assert!((length - 1.0).abs() < 1.0e-12);
    }
    Ok(())
}

#[test]
fn uv_warp_preserves_unknown_attributes_and_updates_named_uv_layer() -> Result<(), Box<dyn Error>> {
    let mut mesh =
        Mesh::from_positions_and_faces(vec![DVec3::ZERO, DVec3::X, DVec3::Y], vec![vec![0, 1, 2]])?;
    mesh.attributes.insert(
        "uv_map".to_owned(),
        json!([{"face_id":0,"layer":"UVMap","uv":[[0.0,0.0],[1.0,0.0],[0.0,1.0]]}]),
    );
    mesh.attributes.insert("marker".to_owned(), json!(true));
    let result = evaluate_modifiers(
        &mesh,
        &[modifier(
            "uv_warp",
            json!({
                "center":[0.0,0.0],
                "axis_u":"X",
                "axis_v":"Y",
                "offset":[0.25,-0.5],
                "scale":[1.0,1.0],
                "rotation":0.0,
                "uv_layer":"UVMap"
            }),
        )?],
    )?;
    assert_eq!(result.attributes["marker"], json!(true));
    assert_eq!(
        result.attributes["uv_map"][0]["uv"],
        json!([[0.25, -0.5], [1.25, -0.5], [0.25, 0.5]])
    );
    Ok(())
}

#[test]
fn attribute_modifiers_do_not_mutate_the_source_mesh() -> Result<(), Box<dyn Error>> {
    let source = weighted_line()?;
    let original = source.clone();
    let result = evaluate_modifiers(
        &source,
        &[modifier(
            "vertex_weight_mix",
            json!({"vertex_group_a":"A","vertex_group_b":"B","mix_mode":"SUB","normalize":false}),
        )?],
    )?;
    assert_ne!(result, source);
    assert_eq!(source, original);
    Ok(())
}
#[test]
fn proximity_modifier_uses_target_object_distance_and_assigns_vertex_weights()
-> Result<(), Box<dyn Error>> {
    let mesh = weighted_line()?;
    let mut operands = BTreeMap::new();
    operands.insert(
        "target".to_owned(),
        vec![scene_operand(
            Mesh::default(),
            DMat4::from_translation(DVec3::X),
            None,
        )],
    );
    let proximity = modifier(
        "vertex_weight_proximity",
        json!({
            "target":"target_node",
            "vertex_group":"A",
            "proximity_mode":"OBJECT",
            "proximity_geometry":["VERTEX"],
            "min_dist":0.0,
            "max_dist":2.0,
            "normalize":false
        }),
    )?;
    let result = evaluate_with_context(&mesh, &proximity, &operands, None, DMat4::IDENTITY)?;
    assert_eq!(weights(&result, "A")?, [0.5, 0.5]);
    Ok(())
}

#[test]
fn normal_edit_writes_target_radial_split_normals() -> Result<(), Box<dyn Error>> {
    let mesh =
        Mesh::from_positions_and_faces(vec![DVec3::ZERO, DVec3::X, DVec3::Y], vec![vec![0, 1, 2]])?;
    let mut operands = BTreeMap::new();
    operands.insert(
        "target".to_owned(),
        vec![scene_operand(
            Mesh::default(),
            DMat4::from_translation(DVec3::new(2.0, 0.0, 0.0)),
            None,
        )],
    );
    let edit = modifier(
        "normal_edit",
        json!({
            "mode":"RADIAL",
            "target":"target_node",
            "mix_mode":"COPY",
            "mix_factor":1.0,
            "mix_limit":std::f64::consts::PI
        }),
    )?;
    let result = evaluate_with_context(&mesh, &edit, &operands, None, DMat4::IDENTITY)?;
    let normal = result.attributes["custom_normal"]["values"]["f0"][0]
        .as_array()
        .ok_or("custom normal missing")?;
    assert!((normal[0].as_f64().ok_or("normal x missing")? + 1.0).abs() < 1.0e-12);
    assert!(normal[1].as_f64().ok_or("normal y missing")?.abs() < 1.0e-12);
    Ok(())
}

#[test]
fn uv_project_uses_camera_projection_and_writes_corner_uvs() -> Result<(), Box<dyn Error>> {
    let mesh = Mesh::from_positions_and_faces(
        vec![DVec3::ZERO, DVec3::X, DVec3::new(1.0, 1.0, 0.0), DVec3::Y],
        vec![vec![0, 1, 2, 3]],
    )?;
    let camera = CameraData::default();
    let mut operands = BTreeMap::new();
    operands.insert(
        "projectors".to_owned(),
        vec![scene_operand(
            Mesh::default(),
            DMat4::from_translation(DVec3::new(0.5, 0.5, 3.0)),
            Some(camera),
        )],
    );
    let projection = modifier(
        "uv_project",
        json!({
            "projectors":["camera"],
            "aspect_x":1.0,
            "aspect_y":1.0,
            "scale_x":1.0,
            "scale_y":1.0,
            "uv_layer":"UVMap"
        }),
    )?;
    let result = evaluate_with_context(&mesh, &projection, &operands, None, DMat4::IDENTITY)?;
    let uv = result.attributes["uv_map"][0]["uv"][0]
        .as_array()
        .ok_or("projected UV missing")?;
    let expected = 0.5 - 0.5 / (2.0 * 3.0 * (36.0 / 100.0));
    assert!((uv[0].as_f64().ok_or("U missing")? - expected).abs() < 1.0e-5);
    assert!((uv[1].as_f64().ok_or("V missing")? - expected).abs() < 1.0e-5);
    Ok(())
}

#[test]
fn data_transfer_topology_replaces_vertex_group_weights() -> Result<(), Box<dyn Error>> {
    let destination = weighted_line()?;
    let mut source = weighted_line()?;
    source.attributes["vertex_groups"]["A"]["v0"] = json!(0.9);
    source.attributes["vertex_groups"]["A"]["v1"] = json!(0.1);
    let mut operands = BTreeMap::new();
    operands.insert(
        "object".to_owned(),
        vec![scene_operand(source, DMat4::IDENTITY, None)],
    );
    let transfer = modifier(
        "data_transfer",
        json!({
            "object":"source",
            "use_object_transform":true,
            "use_vert_data":true,
            "data_types_verts":["VGROUP_WEIGHTS"],
            "vert_mapping":"TOPOLOGY",
            "mix_mode":"REPLACE",
            "mix_factor":1.0
        }),
    )?;
    let result = evaluate_with_context(&destination, &transfer, &operands, None, DMat4::IDENTITY)?;
    let transferred = weights(&result, "A")?;
    assert!((transferred[0] - 0.9).abs() < 1.0e-12);
    assert!((transferred[1] - 0.1).abs() < 1.0e-12);
    Ok(())
}

#[test]
fn mesh_cache_mdd_interpolates_resource_frames() -> Result<(), Box<dyn Error>> {
    let mut mesh = Mesh::from_positions_and_faces(vec![DVec3::ZERO], vec![])?;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&2_i32.to_be_bytes());
    bytes.extend_from_slice(&1_i32.to_be_bytes());
    bytes.extend_from_slice(&0.0_f32.to_be_bytes());
    bytes.extend_from_slice(&1.0_f32.to_be_bytes());
    for x in [0.0_f32, 2.0] {
        bytes.extend_from_slice(&x.to_be_bytes());
        bytes.extend_from_slice(&0.0_f32.to_be_bytes());
        bytes.extend_from_slice(&0.0_f32.to_be_bytes());
    }
    let cache = modifier(
        "mesh_cache",
        json!({
            "resource":"cache",
            "cache_format":"MDD",
            "play_mode":"CUSTOM",
            "time_mode":"TIME",
            "eval_time":0.5,
            "interpolation":"LINEAR",
            "deform_mode":"OVERWRITE",
            "factor":1.0
        }),
    )?;
    evaluate_mesh_cache(&mut mesh, &cache, 1.0, 24, 1.0, &bytes)?;
    assert!((mesh.vertices[0].co.x - 1.0).abs() < 1.0e-12);
    Ok(())
}

#[test]
fn vertex_weight_edit_blends_with_image_mask_channel() -> Result<(), Box<dyn Error>> {
    let source = weighted_line()?;
    let image = ImageData {
        width: 1,
        height: 1,
        pixels: vec![[0.0, 0.0, 0.0, 1.0]],
        tiles: BTreeMap::new(),
        interpolation: ImageInterpolation::Closest,
    };
    let edit = modifier(
        "vertex_weight_edit",
        json!({
            "vertex_group":"A",
            "falloff_type":"SHARP",
            "normalize":false,
            "mask_texture":"mask_image",
            "mask_tex_use_channel":"RED",
            "mask_tex_mapping":"LOCAL"
        }),
    )?;
    let result = evaluate_with_context(
        &source,
        &edit,
        &BTreeMap::new(),
        Some(&image),
        DMat4::IDENTITY,
    )?;
    assert_eq!(weights(&result, "A")?, [0.25, 0.75]);
    Ok(())
}

#[test]
#[expect(
    clippy::cast_possible_truncation,
    reason = "PC2 test data stores inspected f64 positions as f32 samples"
)]
fn cli_applies_resource_backed_mesh_cache_and_inspects_interpolated_positions()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let root = directory.path();
    let scene = root.join("scene");
    let operations_path = root.join("operations.json");
    let initialized = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["init"])
        .arg(&scene)
        .arg("--json")
        .output()?;
    assert!(
        initialized.status.success(),
        "{}",
        String::from_utf8_lossy(&initialized.stdout)
    );

    fs::write(
        &operations_path,
        serde_json::to_vec(&json!({
            "schema_version":1,
            "base_revision":0,
            "operations":[{"op":"node.create","id":"body","kind":"plane","params":{}}]
        }))?,
    )?;
    let applied = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["apply"])
        .arg(&scene)
        .args(["--file"])
        .arg(&operations_path)
        .arg("--json")
        .output()?;
    assert!(
        applied.status.success(),
        "{}",
        String::from_utf8_lossy(&applied.stdout)
    );

    let original = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["inspect"])
        .arg(&scene)
        .args(["--id", "body", "--json"])
        .output()?;
    assert!(
        original.status.success(),
        "{}",
        String::from_utf8_lossy(&original.stdout)
    );
    let original: Value = serde_json::from_slice(&original.stdout)?;
    let positions = original["result"]["items"][0]["evaluated_geometry"]["positions"]
        .as_array()
        .ok_or("plane positions missing from pot inspect")?;

    let cache_path = root.join("plane.pc2");
    let mut cache = b"POINTCACHE2\0".to_vec();
    cache.extend_from_slice(&1_i32.to_le_bytes());
    cache.extend_from_slice(&i32::try_from(positions.len())?.to_le_bytes());
    cache.extend_from_slice(&1.0_f32.to_le_bytes());
    cache.extend_from_slice(&1.0_f32.to_le_bytes());
    cache.extend_from_slice(&2_i32.to_le_bytes());
    for frame_offset in [0.0_f64, 2.0] {
        for position in positions {
            let components = position.as_array().ok_or("position is not a vector")?;
            for component in components {
                let position = component
                    .as_f64()
                    .ok_or("position component is not numeric")?;
                cache.extend_from_slice(&((position + frame_offset) as f32).to_le_bytes());
            }
        }
    }
    fs::write(&cache_path, cache)?;

    fs::write(
        &operations_path,
        serde_json::to_vec(&json!({
            "schema_version":1,
            "base_revision":1,
            "operations":[
                {"op":"resource.pack","id":"plane_cache","uri":cache_path.to_string_lossy(),"kind":"mesh_cache"},
                {"op":"modifier.create","target":{"id":"body"},"id":"cache_modifier","type":"mesh_cache","params":{
                    "resource":"plane_cache","cache_format":"PC2","play_mode":"SCENE",
                    "time_mode":"FRAME","frame_start":0.0,"frame_scale":1.0,
                    "interpolation":"LINEAR","deform_mode":"OVERWRITE","factor":1.0
                }}
            ]
        }))?,
    )?;
    let applied = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["apply"])
        .arg(&scene)
        .args(["--file"])
        .arg(&operations_path)
        .arg("--json")
        .output()?;
    assert!(
        applied.status.success(),
        "{}",
        String::from_utf8_lossy(&applied.stdout)
    );

    let inspected = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["inspect"])
        .arg(&scene)
        .args(["--id", "body", "--frame", "1.5", "--json"])
        .output()?;
    assert!(
        inspected.status.success(),
        "{}",
        String::from_utf8_lossy(&inspected.stdout)
    );
    let inspected: Value = serde_json::from_slice(&inspected.stdout)?;
    let actual = inspected["result"]["items"][0]["evaluated_geometry"]["positions"]
        .as_array()
        .ok_or("evaluated positions missing from pot inspect")?;
    assert_eq!(actual.len(), positions.len());
    for (expected, actual) in positions.iter().zip(actual) {
        for (expected, actual) in expected
            .as_array()
            .ok_or("expected position is not a vector")?
            .iter()
            .zip(actual.as_array().ok_or("actual position is not a vector")?)
        {
            let expected = expected
                .as_f64()
                .ok_or("expected component is not numeric")?
                + 1.0;
            let actual = actual.as_f64().ok_or("actual component is not numeric")?;
            assert!((actual - expected).abs() < 1.0e-5);
        }
    }
    Ok(())
}
#[test]
fn cli_applies_image_texture_mask_and_procedural_masks_are_typed_unsupported()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let operations_path = directory.path().join("mask-operations.json");
    let initialized = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["init"])
        .arg(&scene)
        .arg("--json")
        .output()?;
    assert!(initialized.status.success());
    fs::write(
        &operations_path,
        serde_json::to_vec(&json!({
            "schema_version":1,
            "base_revision":0,
            "operations":[
                {"op":"image.create","id":"mask","width":1,"height":1,"colorspace":"linear","fill_color":[0.0,0.0,0.0,1.0]},
                {"op":"node.create","id":"body","kind":"plane","params":{}},
                {"op":"vertex_group.create","target":{"id":"body"},"id":"group_a","name":"A"},
                {"op":"vertex_group.assign","target":{"id":"body"},"group_id":"group_a","weights":[
                    {"vertex_id":0,"weight":0.25},{"vertex_id":1,"weight":0.5},
                    {"vertex_id":2,"weight":0.75},{"vertex_id":3,"weight":1.0}
                ]},
                {"op":"modifier.create","target":{"id":"body"},"id":"weight_edit","type":"vertex_weight_edit","params":{
                    "vertex_group":"A","falloff_type":"SHARP","normalize":false,
                    "mask_texture":"mask","mask_tex_use_channel":"RED","mask_tex_mapping":"LOCAL"
                }}
            ]
        }))?,
    )?;
    let applied = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["apply"])
        .arg(&scene)
        .args(["--file"])
        .arg(&operations_path)
        .arg("--json")
        .output()?;
    assert!(
        applied.status.success(),
        "{}",
        String::from_utf8_lossy(&applied.stdout)
    );
    let scene_path = scene.join("scene.json");
    let serialized = fs::read(&scene_path)?;
    let doc: SceneDoc = serde_json::from_slice(&serialized)?;
    let snapshot =
        Snapshot::evaluate_with_cache(&doc, &EvaluationContext::default(), Some(&scene))?;
    let body = snapshot
        .meshes
        .get(&potter_core::model::Id::new("body")?)
        .ok_or("evaluated body mesh missing")?;
    let weights = vertex_weights(body, "A")?;
    assert_eq!(weights, [0.25, 0.5, 0.75, 1.0]);

    let mut procedural_scene: Value = serde_json::from_slice(&serialized)?;
    procedural_scene["nodes"]["body"]["modifiers"][0]["params"]["mask_texture"] =
        json!("procedural");
    let procedural_scene: SceneDoc = serde_json::from_value(procedural_scene)?;
    let error = Snapshot::evaluate_with_cache(
        &procedural_scene,
        &EvaluationContext::default(),
        Some(&scene),
    )
    .err()
    .ok_or("procedural texture mask unexpectedly evaluated")?;
    assert_eq!(
        error.details["feature_id"],
        json!("modifier.vertex_weight_edit.texture_type")
    );
    Ok(())
}

fn vertex_weights(mesh: &Mesh, group: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    let groups = mesh.attributes["vertex_groups"]
        .as_object()
        .ok_or("vertex groups missing")?;
    let values = groups[group]
        .as_object()
        .ok_or("vertex group values missing")?;
    mesh.vertices
        .iter()
        .map(|vertex| {
            Ok(values
                .get(&format!("v{}", vertex.id))
                .and_then(Value::as_f64)
                .unwrap_or(0.0))
        })
        .collect()
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "cache fixtures serialize four finite mesh coordinates as f32"
)]
fn pc2_cache(samples: &[[DVec3; 4]; 2]) -> Vec<u8> {
    let mut bytes = b"POINTCACHE2\0".to_vec();
    bytes.extend_from_slice(&1_i32.to_le_bytes());
    bytes.extend_from_slice(&(samples[0].len() as i32).to_le_bytes());
    bytes.extend_from_slice(&0.0_f32.to_le_bytes());
    bytes.extend_from_slice(&1.0_f32.to_le_bytes());
    bytes.extend_from_slice(&(samples.len() as i32).to_le_bytes());
    for sample in samples {
        for point in sample {
            for component in [point.x, point.y, point.z] {
                bytes.extend_from_slice(&(component as f32).to_le_bytes());
            }
        }
    }
    bytes
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "cache fixtures serialize four finite mesh coordinates as f32"
)]
fn mdd_cache(samples: &[[DVec3; 4]; 2]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(samples.len() as i32).to_be_bytes());
    bytes.extend_from_slice(&(samples[0].len() as i32).to_be_bytes());
    bytes.extend_from_slice(&0.0_f32.to_be_bytes());
    bytes.extend_from_slice(&1.0_f32.to_be_bytes());
    for sample in samples {
        for point in sample {
            for component in [point.x, point.y, point.z] {
                bytes.extend_from_slice(&(component as f32).to_be_bytes());
            }
        }
    }
    bytes
}

fn flatten_numbers(value: &Value, output: &mut Vec<f64>) -> Result<(), Box<dyn Error>> {
    if let Some(number) = value.as_f64() {
        output.push(number);
        return Ok(());
    }
    if let Some(value) = value.as_bool() {
        output.push(f64::from(u8::from(value)));
        return Ok(());
    }
    if let Some(values) = value.as_object() {
        for value in values.values() {
            flatten_numbers(value, output)?;
        }
        return Ok(());
    }
    let values = value
        .as_array()
        .ok_or_else(|| format!("parity output is not numeric data: {value}"))?;
    for value in values {
        flatten_numbers(value, output)?;
    }
    Ok(())
}

fn max_numeric_error(first: &Value, second: &Value) -> Result<f64, Box<dyn Error>> {
    let mut first_values = Vec::new();
    let mut second_values = Vec::new();
    flatten_numbers(first, &mut first_values)?;
    flatten_numbers(second, &mut second_values)?;
    assert_eq!(
        first_values.len(),
        second_values.len(),
        "stimulus value count"
    );
    Ok(first_values
        .iter()
        .zip(second_values)
        .map(|(first, second)| (first - second).abs())
        .fold(0.0_f64, f64::max))
}

fn assert_material_change(
    label: &str,
    baseline: &Value,
    expected: &Value,
    tolerance: f64,
) -> Result<(), Box<dyn Error>> {
    let change = max_numeric_error(baseline, expected)?;
    assert!(
        change > 10.0 * tolerance,
        "{label} Blender result did not materially differ from its unmodified baseline: {change}"
    );
    Ok(())
}

fn assert_wrong_candidate_fails(
    label: &str,
    expected: &Value,
    wrong_candidate: &Value,
    tolerance: f64,
) -> Result<(), Box<dyn Error>> {
    let error = max_numeric_error(expected, wrong_candidate)?;
    assert!(
        error > 10.0 * tolerance,
        "{label} fixture does not distinguish its plausible wrong implementation: {error}"
    );
    Ok(())
}

fn rigid_translation_candidate(
    baseline: &Value,
    expected: &Value,
) -> Result<Value, Box<dyn Error>> {
    let baseline = baseline
        .as_array()
        .ok_or("cache baseline is not an array")?;
    let expected = expected.as_array().ok_or("cache result is not an array")?;
    assert_eq!(
        baseline.len(),
        expected.len(),
        "cache vertex correspondence"
    );
    let point = |value: &Value| -> Result<DVec3, Box<dyn Error>> {
        let coordinates = value.as_array().ok_or("cache position is not an array")?;
        Ok(DVec3::new(
            coordinates
                .first()
                .and_then(Value::as_f64)
                .ok_or("missing x coordinate")?,
            coordinates
                .get(1)
                .and_then(Value::as_f64)
                .ok_or("missing y coordinate")?,
            coordinates
                .get(2)
                .and_then(Value::as_f64)
                .ok_or("missing z coordinate")?,
        ))
    };
    let translation = point(expected.first().ok_or("cache result has no vertices")?)?
        - point(baseline.first().ok_or("cache baseline has no vertices")?)?;
    let translated = baseline
        .iter()
        .map(|position| Ok((point(position)? + translation).to_array()))
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    Ok(json!(translated))
}

fn assert_normal_stimulus(
    label: &str,
    baseline: &Value,
    expected: &Value,
    tolerance: f64,
) -> Result<(), Box<dyn Error>> {
    let mut baseline_values = Vec::new();
    let mut expected_values = Vec::new();
    flatten_numbers(baseline, &mut baseline_values)?;
    flatten_numbers(expected, &mut expected_values)?;
    assert_eq!(
        baseline_values.len(),
        expected_values.len(),
        "{label} baseline normal count"
    );
    let baseline_chunks = baseline_values.as_chunks::<3>().0;
    let expected_chunks = expected_values.as_chunks::<3>().0;
    let maximum_angle = baseline_chunks
        .iter()
        .zip(expected_chunks.iter())
        .map(|(baseline, expected)| {
            DVec3::from_array(*baseline)
                .normalize_or_zero()
                .angle_between(DVec3::from_array(*expected).normalize_or_zero())
        })
        .fold(0.0_f64, f64::max);
    assert!(
        maximum_angle > 10.0 * tolerance,
        "{label} Blender result did not materially differ from its unmodified baseline: {maximum_angle}"
    );
    Ok(())
}

fn assert_parity(
    label: &str,
    actual: &Value,
    expected: &Value,
    tolerance: f64,
) -> Result<(), Box<dyn Error>> {
    let mut actual_values = Vec::new();
    let mut expected_values = Vec::new();
    flatten_numbers(actual, &mut actual_values)?;
    flatten_numbers(expected, &mut expected_values)?;
    assert_eq!(
        actual_values.len(),
        expected_values.len(),
        "{label} value count: actual={actual}, expected={expected}"
    );
    let max_error = actual_values
        .iter()
        .zip(&expected_values)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max);
    eprintln!("Blender parity {label}: max absolute error {max_error:.8e}");
    assert!(
        max_error <= tolerance,
        "{label} max absolute error {max_error} exceeds {tolerance}; actual={actual}, expected={expected}"
    );
    Ok(())
}

const BLENDER_MODIFIER_PARITY_SCRIPT: &str = r#"
import bpy
import json
import math
import os
import sys

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
depsgraph = bpy.context.evaluated_depsgraph_get()

def make_object(name, vertices, faces):
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj

def add_group(obj, name, weights):
    group = obj.vertex_groups.new(name=name)
    for index, weight in enumerate(weights):
        if weight:
            group.add([index], float(weight), "REPLACE")
    return group

def set_uv(obj):
    layer = obj.data.uv_layers.new(name="UVMap")
    coordinates = [(0, 0), (1, 0), (1, 1), (0, 1)]
    for loop in obj.data.loops:
        layer.data[loop.index].uv = coordinates[loop.vertex_index % len(coordinates)]
def evaluated_mesh(obj):
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    return evaluated, evaluated.to_mesh()

def read_weights(obj, group_name):
    evaluated, mesh = evaluated_mesh(obj)
    group = evaluated.vertex_groups.get(group_name)
    values = []
    for vertex in mesh.vertices:
        value = 0.0
        if group:
            for assignment in vertex.groups:
                if assignment.group == group.index:
                    value = assignment.weight
                    break
        values.append(value)
    evaluated.to_mesh_clear()
    return values

def read_positions(obj):
    evaluated, mesh = evaluated_mesh(obj)
    values = [list(vertex.co) for vertex in mesh.vertices]
    evaluated.to_mesh_clear()
    return values

def read_normals(obj):
    evaluated, mesh = evaluated_mesh(obj)
    values = [list(loop.normal) for loop in mesh.loops]
    evaluated.to_mesh_clear()
    return values

def read_uv(obj):
    evaluated, mesh = evaluated_mesh(obj)
    values = [list(item.uv) for item in mesh.uv_layers["UVMap"].data]
    evaluated.to_mesh_clear()
    return values

line_vertices = [(0, 0, 0), (1, 0, 0)]
line = make_object("Edit", line_vertices, [])
add_group(line, "A", [0.25, 0.75])
result = {"baseline_vertex_weight_edit": read_weights(line, "A")}
edit = line.modifiers.new("Edit", "VERTEX_WEIGHT_EDIT")
edit.vertex_group = "A"
edit.falloff_type = "SHARP"
edit.normalize = False
result["vertex_weight_edit"] = read_weights(line, "A")

falloff_vertices = [(0, 0, 0), (1, 0, 0), (2, 0.5, 0), (3, 1, 0)]
falloff_weights = [0.12, 0.31, 0.64, 0.87]
result["baseline_vertex_weight_edit_falloffs"] = {}
result["vertex_weight_edit_falloffs"] = {}
for falloff in ("LINEAR", "SMOOTH", "ROOT", "ICON_SPHERECURVE", "STEP", "SHARP"):
    falloff_obj = make_object("EditFalloff" + falloff, falloff_vertices, [])
    add_group(falloff_obj, "A", falloff_weights)
    result["baseline_vertex_weight_edit_falloffs"][falloff] = read_weights(falloff_obj, "A")
    falloff_modifier = falloff_obj.modifiers.new("EditFalloff", "VERTEX_WEIGHT_EDIT")
    falloff_modifier.vertex_group = "A"
    falloff_modifier.falloff_type = falloff
    falloff_modifier.normalize = False
    result["vertex_weight_edit_falloffs"][falloff] = read_weights(falloff_obj, "A")

mix = make_object("Mix", line_vertices, [])
add_group(mix, "A", [0.25, 0.75])
add_group(mix, "B", [0.5, 0.25])
result["baseline_vertex_weight_mix"] = read_weights(mix, "A")
weight_mix = mix.modifiers.new("Mix", "VERTEX_WEIGHT_MIX")
weight_mix.vertex_group_a = "A"
weight_mix.vertex_group_b = "B"
weight_mix.mix_mode = "ADD"
weight_mix.mix_set = "ALL"
weight_mix.normalize = False
result["vertex_weight_mix"] = read_weights(mix, "A")

mix_mode_vertices = [(0, 0, 0), (1, 0, 0), (2, 0.5, 0), (3, 1, 0)]
mix_mode_a = [0.17, 0.39, 0.68, 0.82]
mix_mode_b = [0.71, 0.24, 0.45, 0.16]
result["baseline_vertex_weight_mix_modes"] = {}
result["vertex_weight_mix_modes"] = {}
for mix_mode in ("SET", "ADD", "SUB", "MUL", "DIV", "DIF", "AVG", "MIN", "MAX"):
    mix_obj = make_object("MixMode" + mix_mode, mix_mode_vertices, [])
    add_group(mix_obj, "A", mix_mode_a)
    add_group(mix_obj, "B", mix_mode_b)
    result["baseline_vertex_weight_mix_modes"][mix_mode] = read_weights(mix_obj, "A")
    mode_modifier = mix_obj.modifiers.new("MixMode", "VERTEX_WEIGHT_MIX")
    mode_modifier.vertex_group_a = "A"
    mode_modifier.vertex_group_b = "B"
    mode_modifier.mix_mode = mix_mode
    mode_modifier.mix_set = "ALL"
    mode_modifier.normalize = False
    result["vertex_weight_mix_modes"][mix_mode] = read_weights(mix_obj, "A")

proximity = make_object("Proximity", line_vertices, [])
proximity.location = (2, -1, 0)
proximity.scale = (2, 1, 1)
add_group(proximity, "A", [0.25, 0.75])
target = bpy.data.objects.new("Target", None)
bpy.context.collection.objects.link(target)
target.location = (3.5, -1, 0)
result["baseline_vertex_weight_proximity"] = read_weights(proximity, "A")
proximity_modifier = proximity.modifiers.new("Proximity", "VERTEX_WEIGHT_PROXIMITY")
proximity_modifier.target = target
proximity_modifier.vertex_group = "A"
proximity_modifier.proximity_mode = "OBJECT"
proximity_modifier.min_dist = 0
proximity_modifier.max_dist = 3.0
proximity_modifier.normalize = False
result["vertex_weight_proximity"] = read_weights(proximity, "A")
proximity_geometry = make_object("ProximityGeometry", line_vertices, [])
add_group(proximity_geometry, "A", [0.25, 0.75])
target_geometry = make_object("ProximityVertex", [(0, 0, 0)], [])
result["baseline_vertex_weight_proximity_geometry"] = read_weights(proximity_geometry, "A")
geometry_modifier = proximity_geometry.modifiers.new("ProximityGeometry", "VERTEX_WEIGHT_PROXIMITY")
geometry_modifier.target = target_geometry
geometry_modifier.vertex_group = "A"
geometry_modifier.proximity_mode = "GEOMETRY"
geometry_modifier.proximity_geometry = {"VERTEX"}
geometry_modifier.min_dist = 0
geometry_modifier.max_dist = 1
geometry_modifier.normalize = False
result["vertex_weight_proximity_geometry"] = read_weights(proximity_geometry, "A")

proximity_subject_vertices = [
    (0.5, 0.5, 0.25),
    (1.5, 1.5, 0.0),
    (2.5, 0.0, 0.0),
    (0.5, 0.0, 1.2),
]
proximity_surface = make_object(
    "ProximitySurface",
    [(0, 0, 0), (2, 0, 0), (0, 2, 0)],
    [(0, 1, 2)],
)
result["baseline_vertex_weight_proximity_geometry_falloffs"] = {}
result["vertex_weight_proximity_geometry_falloffs"] = {}
for geometry in ("EDGE", "FACE"):
    for falloff in ("LINEAR", "SMOOTH", "ROOT", "ICON_SPHERECURVE", "STEP", "SHARP"):
        geometry_obj = make_object(
            "Proximity" + geometry + falloff,
            proximity_subject_vertices,
            [],
        )
        add_group(geometry_obj, "A", [0.1, 0.3, 0.6, 0.9])
        key = geometry + "_" + falloff
        result["baseline_vertex_weight_proximity_geometry_falloffs"][key] = read_weights(
            geometry_obj, "A"
        )
        geometry_modifier = geometry_obj.modifiers.new("Proximity", "VERTEX_WEIGHT_PROXIMITY")
        geometry_modifier.target = proximity_surface
        geometry_modifier.vertex_group = "A"
        geometry_modifier.proximity_mode = "GEOMETRY"
        geometry_modifier.proximity_geometry = {geometry}
        geometry_modifier.min_dist = 0.0
        geometry_modifier.max_dist = 1.5
        geometry_modifier.falloff_type = falloff
        geometry_modifier.normalize = False
        result["vertex_weight_proximity_geometry_falloffs"][key] = read_weights(
            geometry_obj, "A"
        )

weighted_vertices = [(0, 0, 0), (4, 0, 0), (0, 1, 0), (0, 0, 3), (-2, 0, 0)]
weighted_faces = [(0, 1, 2), (0, 2, 3), (0, 3, 4)]
weighted_obj = make_object("Weighted", weighted_vertices, weighted_faces)
for index, polygon in enumerate(weighted_obj.data.polygons):
    polygon.use_smooth = True
    polygon.material_index = 1 if index == 2 else 0
result["baseline_weighted_normal"] = read_normals(weighted_obj)
weighted = weighted_obj.modifiers.new("Weighted", "WEIGHTED_NORMAL")
weighted.mode = "FACE_AREA"
weighted.weight = 50
weighted.keep_sharp = False
result["weighted_normal"] = read_normals(weighted_obj)

weighted_sharp_obj = make_object("WeightedSharp", weighted_vertices, weighted_faces)
for polygon in weighted_sharp_obj.data.polygons:
    polygon.use_smooth = True
for edge in weighted_sharp_obj.data.edges:
    if set(edge.vertices) == {0, 2}:
        edge.use_edge_sharp = True
result["baseline_weighted_normal_sharp"] = read_normals(weighted_sharp_obj)
weighted_sharp = weighted_sharp_obj.modifiers.new("WeightedSharp", "WEIGHTED_NORMAL")
weighted_sharp.mode = "FACE_AREA"
weighted_sharp.weight = 50
weighted_sharp.keep_sharp = True
result["weighted_normal_sharp"] = read_normals(weighted_sharp_obj)

weighted_face_obj = make_object("WeightedFaceInfluence", weighted_vertices, weighted_faces)
for index, polygon in enumerate(weighted_face_obj.data.polygons):
    polygon.use_smooth = True
    polygon.material_index = 1 if index == 2 else 0
strength = weighted_face_obj.data.attributes.new(name="__mod_weightednormals_faceweight", type="INT", domain="FACE")
for index, value in enumerate((0, 1, 2)):
    strength.data[index].value = value
result["baseline_weighted_normal_face_influence"] = read_normals(weighted_face_obj)
weighted_face = weighted_face_obj.modifiers.new("WeightedFaceInfluence", "WEIGHTED_NORMAL")
weighted_face.mode = "FACE_AREA"
weighted_face.weight = 50
weighted_face.use_face_influence = True
result["weighted_normal_face_influence"] = read_normals(weighted_face_obj)
quad_vertices = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
normal_obj = make_object("Normal", quad_vertices, [(0, 1, 2, 3)])
normal_target = bpy.data.objects.new("NormalTarget", None)
bpy.context.collection.objects.link(normal_target)
normal_target.location = (2, 0, 0)
result["baseline_normal_edit"] = read_normals(normal_obj)
normal_modifier = normal_obj.modifiers.new("Normal", "NORMAL_EDIT")
normal_modifier.target = normal_target
normal_modifier.mode = "RADIAL"
normal_modifier.mix_mode = "COPY"
normal_modifier.mix_factor = 1.0
normal_modifier.mix_limit = 3.141592653589793
normal_modifier.use_direction_parallel = False
result["normal_edit"] = read_normals(normal_obj)

directional_obj = make_object("DirectionalNormal", quad_vertices, [(0, 1, 2, 3)])
directional_target = bpy.data.objects.new("DirectionalTarget", None)
bpy.context.collection.objects.link(directional_target)
directional_target.location = (2, -1, 0)
result["baseline_normal_edit_directional"] = read_normals(directional_obj)
directional_modifier = directional_obj.modifiers.new("DirectionalNormal", "NORMAL_EDIT")
directional_modifier.target = directional_target
directional_modifier.mode = "DIRECTIONAL"
directional_modifier.mix_mode = "ADD"
directional_modifier.mix_factor = 0.65
directional_modifier.mix_limit = 3.141592653589793
directional_modifier.offset = (0.5, 0.1, 0)
directional_modifier.use_direction_parallel = True
result["normal_edit_directional"] = read_normals(directional_obj)

project_vertices = [
    (0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0),
    (3, 0, 0), (3, 2, 0), (3, 2, 1), (3, 0, 1),
]
project_faces = [(0, 1, 2, 3), (4, 5, 6, 7)]

def add_camera(name, location, rotation):
    camera_data = bpy.data.cameras.new(name)
    camera = bpy.data.objects.new(name, camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = location
    camera.rotation_euler = rotation
    return camera

camera_z = add_camera("ProjectorZ", (1, 0.5, 4), (0, 0, 0))
camera_x = add_camera("ProjectorX", (5, 0.5, 0.5), (0, math.pi / 2, 0))
project_obj = make_object("Project", project_vertices, project_faces)
set_uv(project_obj)
result["baseline_uv_project"] = read_uv(project_obj)
project = project_obj.modifiers.new("Project", "UV_PROJECT")
project.uv_layer = "UVMap"
project.projector_count = 2
project.projectors[0].object = camera_z
project.projectors[1].object = camera_x
result["uv_project"] = read_uv(project_obj)
single_projected = []
for name, camera in (("z", camera_z), ("x", camera_x)):
    single_obj = make_object("SingleProjector" + name, project_vertices, project_faces)
    set_uv(single_obj)
    single = single_obj.modifiers.new("Single", "UV_PROJECT")
    single.uv_layer = "UVMap"
    single.projector_count = 1
    single.projectors[0].object = camera
    single_projected.append(read_uv(single_obj))
result["uv_project_average_wrong"] = [
    [(first[0] + second[0]) * 0.5, (first[1] + second[1]) * 0.5]
    for first, second in zip(single_projected[0], single_projected[1])
]
ortho_camera = add_camera("ProjectorOrtho", (0.25, 0.4, 3), (0, 0, 0))
ortho_camera.data.type = "ORTHO"
ortho_camera.data.ortho_scale = 3.5
ortho_obj = make_object("ProjectOrtho", project_vertices, project_faces)
set_uv(ortho_obj)
result["baseline_uv_project_orthographic"] = read_uv(ortho_obj)
ortho_project = ortho_obj.modifiers.new("ProjectOrtho", "UV_PROJECT")
ortho_project.uv_layer = "UVMap"
ortho_project.projector_count = 1
ortho_project.projectors[0].object = ortho_camera
ortho_project.aspect_x = 1.4
ortho_project.aspect_y = 1.2
ortho_project.scale_x = 0.8
ortho_project.scale_y = 1.3
result["uv_project_orthographic"] = read_uv(ortho_obj)
ortho_clamped_obj = make_object("ProjectOrthoClamped", project_vertices, project_faces)
set_uv(ortho_clamped_obj)
result["baseline_uv_project_orthographic_clamped"] = read_uv(ortho_clamped_obj)
ortho_clamped = ortho_clamped_obj.modifiers.new("ProjectOrthoClamped", "UV_PROJECT")
ortho_clamped.uv_layer = "UVMap"
ortho_clamped.projector_count = 1
ortho_clamped.projectors[0].object = ortho_camera
ortho_clamped.aspect_x = 1.4
ortho_clamped.aspect_y = 0.7
ortho_clamped.scale_x = 0.8
ortho_clamped.scale_y = 1.3
result["uv_project_orthographic_clamped"] = read_uv(ortho_clamped_obj)
warp_transform_obj = make_object("WarpTransform", quad_vertices, [(0, 1, 2, 3)])
set_uv(warp_transform_obj)
add_group(warp_transform_obj, "A", [0.1, 0.4, 0.7, 1.0])
result["baseline_uv_warp_transforms"] = read_uv(warp_transform_obj)
warp_from = bpy.data.objects.new("WarpFrom", None)
bpy.context.collection.objects.link(warp_from)
warp_from.location = (0.25, 0.15, 0)
warp_from.rotation_euler[2] = 0.3
warp_from.scale = (1.1, 0.9, 1)
warp_to = bpy.data.objects.new("WarpTo", None)
bpy.context.collection.objects.link(warp_to)
warp_to.location = (-0.2, 0.1, 0)
warp_to.rotation_euler[2] = -0.25
warp_to.scale = (0.8, 1.3, 1)
warp_transform = warp_transform_obj.modifiers.new("WarpTransform", "UV_WARP")
warp_transform.uv_layer = "UVMap"
warp_transform.center = (0.5, 0.5)
warp_transform.object_from = warp_from
warp_transform.object_to = warp_to
warp_transform.vertex_group = "A"
result["uv_warp_transforms"] = read_uv(warp_transform_obj)

warp_obj = make_object("Warp", quad_vertices, [(0, 1, 2, 3)])
set_uv(warp_obj)
result["baseline_uv_warp"] = read_uv(warp_obj)
warp = warp_obj.modifiers.new("Warp", "UV_WARP")
warp.uv_layer = "UVMap"
warp.center = (0.5, 0.5)
warp.axis_u = "X"
warp.axis_v = "Y"
warp.offset = (0.25, -0.5)
warp.scale = (1.25, 0.75)
warp.rotation = 0.35
result["uv_warp"] = read_uv(warp_obj)

transfer_source = make_object("TransferSource", quad_vertices, [(0, 1, 2, 3)])
add_group(transfer_source, "A", [0.9, 0.8, 0.7, 0.6])
transfer_dest = make_object("TransferDest", quad_vertices, [(0, 1, 2, 3)])
add_group(transfer_dest, "A", [0.1, 0.2, 0.3, 0.4])
result["baseline_data_transfer"] = read_weights(transfer_dest, "A")
transfer = transfer_dest.modifiers.new("Transfer", "DATA_TRANSFER")
transfer.object = transfer_source
transfer.use_object_transform = True
transfer.use_vert_data = True
transfer.data_types_verts = {"VGROUP_WEIGHTS"}
transfer.vert_mapping = "TOPOLOGY"
transfer.mix_mode = "REPLACE"
transfer.mix_factor = 1.0
result["data_transfer"] = read_weights(transfer_dest, "A")
poly_source = make_object("PolySource", quad_vertices, [(0, 1, 2, 3)])
add_group(poly_source, "A", [0.0, 1.0, 1.0, 0.0])
for mapping, point, key in (
    ("NEAREST", (0.1, 0.1, 0.0), "data_transfer_nearest"),
    ("POLYINTERP_NEAREST", (0.5, 0.5, 0.0), "data_transfer_polyinterp"),
):
    point_dest = make_object("Point" + key, [point], [])
    add_group(point_dest, "A", [0.2])
    result["baseline_" + key] = read_weights(point_dest, "A")
    point_transfer = point_dest.modifiers.new("Transfer", "DATA_TRANSFER")
    point_transfer.object = poly_source
    point_transfer.use_vert_data = True
    point_transfer.data_types_verts = {"VGROUP_WEIGHTS"}
    point_transfer.vert_mapping = mapping
    point_transfer.mix_mode = "REPLACE"
    point_transfer.mix_factor = 1.0
    result[key] = read_weights(point_dest, "A")
domain_source = make_object("DomainSource", quad_vertices, [(0, 1, 2, 3)])
set_uv(domain_source)
for loop in domain_source.data.loops:
    layer = domain_source.data.uv_layers["UVMap"].data[loop.index]
    layer.uv = (layer.uv[0] + 0.25, layer.uv[1] + 0.25)
for edge in domain_source.data.edges:
    edge.use_seam = set(edge.vertices) == {1, 2}
    edge.use_edge_sharp = set(edge.vertices) == {2, 3}
crease = domain_source.data.attributes.get("crease_edge")
if crease is None:
    crease = domain_source.data.attributes.new(name="crease_edge", type="FLOAT", domain="EDGE")
for edge in domain_source.data.edges:
    crease.data[edge.index].value = 0.75 if set(edge.vertices) == {0, 1} else 0.0
color = domain_source.data.color_attributes.new(name="Color", type="FLOAT_COLOR", domain="CORNER")
for loop in domain_source.data.loops:
    color.data[loop.index].color = (0.2 + loop.index * 0.1, 0.3, 0.4, 1.0)
domain_source.data.normals_split_custom_set([(0.0, 0.0, 1.0)] * len(domain_source.data.loops))
domain_source.data.polygons[0].use_smooth = True
freestyle = domain_source.data.attributes.new(name="freestyle_face", type="BOOLEAN", domain="FACE")
freestyle.data[0].value = True
domain_dest = make_object(
    "DomainDest",
    [(0, 0, 0), (1.2, 0.1, 0), (1.6, 1.3, 0), (0, 1, 0)],
    [(0, 1, 2), (0, 2, 3)],
)
set_uv(domain_dest)
domain_dest.data.color_attributes.new(name="Color", type="FLOAT_COLOR", domain="CORNER")
domain_dest.data.normals_split_custom_set([(1.0, 0.0, 0.0)] * len(domain_dest.data.loops))
domain_dest.data.attributes.new(name="crease_edge", type="FLOAT", domain="EDGE")
domain_dest.data.attributes.new(name="sharp_edge", type="BOOLEAN", domain="EDGE")
domain_dest.data.attributes.new(name="uv_seam", type="BOOLEAN", domain="EDGE")
domain_dest.data.attributes.new(name="freestyle_face", type="BOOLEAN", domain="FACE")

def domain_values(mesh):
    def edge_values(getter):
        return {
            "%d-%d" % tuple(sorted(edge.vertices)): getter(edge)
            for edge in mesh.edges
        }

    return {
        "sharp": edge_values(lambda edge: edge.use_edge_sharp),
        "seam": edge_values(lambda edge: edge.use_seam),
        "crease": edge_values(
            lambda edge: mesh.attributes["crease_edge"].data[edge.index].value
        ),
        "uv": [list(item.uv) for item in mesh.uv_layers["UVMap"].data],
        "color": [list(item.color) for item in mesh.color_attributes["Color"].data],
        "normal": [list(loop.normal) for loop in mesh.loops],
        "smooth": [poly.use_smooth for poly in mesh.polygons],
        "freestyle": [
            mesh.attributes["freestyle_face"].data[index].value
            for index in range(len(mesh.polygons))
        ],
    }

baseline_eval, baseline_mesh = evaluated_mesh(domain_dest)
result["baseline_data_transfer_domains"] = domain_values(baseline_mesh)
baseline_eval.to_mesh_clear()
domain_transfer = domain_dest.modifiers.new("DomainTransfer", "DATA_TRANSFER")
domain_transfer.object = domain_source
domain_transfer.use_object_transform = True
domain_transfer.use_edge_data = True
domain_transfer.use_loop_data = True
domain_transfer.use_poly_data = True
domain_transfer.data_types_edges = {"SHARP_EDGE", "SEAM", "CREASE"}
domain_transfer.data_types_loops = {"CUSTOM_NORMAL", "UV", "COLOR_CORNER"}
domain_transfer.data_types_polys = {"SMOOTH", "FREESTYLE_FACE"}
domain_transfer.edge_mapping = "NEAREST"
domain_transfer.loop_mapping = "POLYINTERP_LNORPROJ"
domain_transfer.poly_mapping = "POLYINTERP_PNORPROJ"
domain_transfer.layers_uv_select_src = "ALL"
domain_transfer.layers_uv_select_dst = "NAME"
domain_transfer.mix_mode = "REPLACE"
domain_transfer.mix_factor = 1.0
domain_eval, domain_mesh = evaluated_mesh(domain_dest)
result["data_transfer_domains"] = domain_values(domain_mesh)
domain_eval.to_mesh_clear()

for mapping in ("LOCAL", "GLOBAL", "OBJECT", "UV"):
    mask_obj = make_object("Mask" + mapping, quad_vertices, [(0, 1, 2, 3)])
    mask_obj.location = (0.25, -0.25, 0)
    mask_obj.rotation_euler = (0, 0, 0.2)
    mask_obj.scale = (0.8, 0.7, 1.0)
    set_uv(mask_obj)
    if mapping == "UV":
        uv_coordinates = [(0.13, 0.17), (0.73, 0.24), (0.84, 0.81), (0.2, 0.69)]
        for loop in mask_obj.data.loops:
            mask_obj.data.uv_layers["UVMap"].data[loop.index].uv = uv_coordinates[loop.vertex_index]
    add_group(mask_obj, "A", [0.25, 0.5, 0.75, 1.0])
    image = bpy.data.images.new(
        "PatternMask" + mapping,
        width=2,
        height=2,
        alpha=True,
        float_buffer=True,
    )
    image.colorspace_settings.name = "Non-Color"
    image.pixels = [
        channel
        for red in (0.0, 0.25, 0.6, 1.0)
        for channel in (red, red, red, 1.0)
    ]
    image.update()
    texture = bpy.data.textures.new("PatternTexture" + mapping, type="IMAGE")
    texture.image = image
    texture.use_interpolation = False
    mask_object = bpy.data.objects.new("MaskObject" + mapping, None)
    bpy.context.collection.objects.link(mask_object)
    mask_object.location = (0.75, 0.25, 0)
    mask_object.rotation_euler = (0, 0, math.pi / 4)
    mask_object.scale = (1.5, 0.75, 1.0)
    result["baseline_texture_mask_" + mapping.lower()] = read_weights(mask_obj, "A")
    modifier = mask_obj.modifiers.new("Mask", "VERTEX_WEIGHT_EDIT")
    modifier.vertex_group = "A"
    modifier.mask_texture = texture
    modifier.falloff_type = "SHARP"
    modifier.mask_tex_use_channel = "RED"
    modifier.mask_tex_mapping = mapping
    modifier.mask_tex_uv_layer = "UVMap"
    if mapping == "OBJECT":
        modifier.mask_tex_map_object = mask_object
    result["texture_mask_" + mapping.lower()] = read_weights(mask_obj, "A")

for cache_format, filename, time_mode in (
    ("PC2", "quad.pc2", "FRAME"),
    ("MDD", "quad.mdd", "TIME"),
):
    cache_obj = make_object("Cache" + cache_format, quad_vertices, [(0, 1, 2, 3)])
    result["baseline_mesh_cache_" + cache_format.lower()] = read_positions(cache_obj)
    modifier = cache_obj.modifiers.new("Cache", "MESH_CACHE")
    modifier.cache_format = cache_format
    modifier.filepath = os.path.join(root, filename)
    modifier.play_mode = "CUSTOM"
    modifier.time_mode = time_mode
    modifier.interpolation = "LINEAR"
    modifier.deform_mode = "OVERWRITE"
    modifier.factor = 1.0
    if cache_format == "PC2":
        modifier.frame_start = 0.0
        modifier.frame_scale = 1.0
        modifier.forward_axis = "POS_X"
        modifier.up_axis = "POS_Z"
        result["mesh_cache_pc2"] = {}
        for evaluation_frame in (0.25, 0.75):
            modifier.eval_frame = evaluation_frame
            result["mesh_cache_pc2"][str(evaluation_frame)] = read_positions(cache_obj)
    else:
        result["mesh_cache_mdd"] = {}
        for evaluation_time in (0.25, 0.75):
            modifier.eval_time = evaluation_time
            result["mesh_cache_mdd"][str(evaluation_time)] = read_positions(cache_obj)

for cache_format, time_mode, evaluation, filename, result_key in (
    ("PC2", "TIME", 0.4, "quad.pc2", "mesh_cache_integrate_time"),
    ("MDD", "FACTOR", 0.35, "quad.mdd", "mesh_cache_integrate_factor"),
):
    cache_obj = make_object("CacheIntegrate" + time_mode, quad_vertices, [(0, 1, 2, 3)])
    add_group(cache_obj, "A", [0.1, 0.4, 0.7, 1.0])
    result["baseline_" + result_key] = read_positions(cache_obj)
    modifier = cache_obj.modifiers.new("CacheIntegrate", "MESH_CACHE")
    modifier.cache_format = cache_format
    modifier.filepath = os.path.join(root, filename)
    modifier.play_mode = "CUSTOM"
    modifier.time_mode = time_mode
    modifier.interpolation = "LINEAR"
    modifier.deform_mode = "INTEGRATE"
    modifier.vertex_group = "A"
    modifier.factor = 0.8
    modifier.forward_axis = "POS_X"
    modifier.up_axis = "POS_Z"
    if time_mode == "TIME":
        modifier.eval_time = evaluation
    else:
        modifier.eval_factor = evaluation
    result[result_key] = read_positions(cache_obj)

with open(os.path.join(root, "expected.json"), "w", encoding="utf-8") as stream:
    json.dump(result, stream)
"#;
const BLENDER_LOOP_NORMAL_QUANTIZATION_SCRIPT: &str = r#"
import bpy
import json
import os
import sys

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
with open(os.path.join(root, "normal_quantization_input.json"), encoding="utf-8") as stream:
    data = json.load(stream)
mesh = bpy.data.meshes.new("NormalQuantization")
mesh.from_pydata(data["vertices"], [], data["faces"])
mesh.update()
for index, polygon in enumerate(mesh.polygons):
    polygon.use_smooth = True
    polygon.material_index = data["materials"][index]
sharp_edges = {tuple(sorted(edge)) for edge in data["sharp_edges"]}
for edge in mesh.edges:
    edge.use_edge_sharp = tuple(sorted(edge.vertices)) in sharp_edges
mesh.update()
mesh.normals_split_custom_set(data["normals"])
obj = bpy.data.objects.new("NormalQuantization", mesh)
bpy.context.collection.objects.link(obj)
evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
evaluated_mesh = evaluated.to_mesh()
normals = [list(loop.normal) for loop in evaluated_mesh.loops]
evaluated.to_mesh_clear()
with open(os.path.join(root, "normal_quantization_output.json"), "w", encoding="utf-8") as stream:
    json.dump(normals, stream)
"#;

fn blender_quantize_loop_normals(
    blender: &Path,
    root: &Path,
    mesh: &Mesh,
    normals: &Value,
) -> Result<Value, Box<dyn Error>> {
    let vertex_indices = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<BTreeMap<_, _>>();
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
                .map(|vertex_id| {
                    vertex_indices
                        .get(vertex_id)
                        .copied()
                        .ok_or("normal quantization face vertex missing")
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let sharp_edges = mesh
        .edges
        .iter()
        .filter(|edge| {
            mesh.attributes
                .get("sharp_edges")
                .and_then(Value::as_array)
                .is_some_and(|ids| ids.iter().any(|id| id.as_u64() == Some(u64::from(edge.id))))
        })
        .map(|edge| {
            edge.vertices
                .iter()
                .map(|vertex_id| {
                    vertex_indices
                        .get(vertex_id)
                        .copied()
                        .ok_or("normal quantization sharp-edge vertex missing")
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let materials = mesh
        .faces
        .iter()
        .map(|face| face.material_index)
        .collect::<Vec<_>>();
    fs::write(
        root.join("normal_quantization_input.json"),
        serde_json::to_vec(
            &json!({"vertices":vertices,"faces":faces,"materials":materials,"sharp_edges":sharp_edges,"normals":normals}),
        )?,
    )?;
    let script_path = root.join("normal_quantization.py");
    fs::write(&script_path, BLENDER_LOOP_NORMAL_QUANTIZATION_SCRIPT)?;
    let output = Command::new(blender)
        .args(["--background", "--factory-startup", "--python"])
        .arg(script_path)
        .args(["--"])
        .arg(root)
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "Blender loop-normal quantization failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(serde_json::from_slice(&fs::read(
        root.join("normal_quantization_output.json"),
    )?)?)
}
#[test]
fn blender_parity_covers_all_owned_modifier_types_and_image_mask_mappings()
-> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping modifier Blender parity test: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = fs::canonicalize(directory.path())?;
    let quad = quad_mesh_for_parity()?;
    let cache_samples = [
        [
            DVec3::new(0.2, 0.1, 0.4),
            DVec3::new(1.2, -0.2, 0.7),
            DVec3::new(0.6, 1.1, -0.5),
            DVec3::new(-0.4, 0.8, 1.4),
        ],
        [
            DVec3::new(0.8, -0.3, 0.2),
            DVec3::new(1.6, 0.6, -0.1),
            DVec3::new(1.5, 0.2, 0.9),
            DVec3::new(-0.2, 1.7, 0.3),
        ],
    ];
    let pc2_path = root.join("quad.pc2");
    let mdd_path = root.join("quad.mdd");
    let pc2 = pc2_cache(&cache_samples);
    let mdd = mdd_cache(&cache_samples);
    fs::write(&pc2_path, &pc2)?;
    fs::write(&mdd_path, &mdd)?;
    let script_path = root.join("modifier_parity.py");
    fs::write(&script_path, BLENDER_MODIFIER_PARITY_SCRIPT)?;
    let output = Command::new(&blender)
        .args(["--background", "--factory-startup", "--python"])
        .arg(&script_path)
        .args(["--"])
        .arg(&root)
        .output()?;
    assert!(
        output.status.success(),
        "Blender modifier fixture failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let expected_path = root.join("expected.json");
    if !expected_path.is_file() {
        return Err(format!(
            "Blender modifier fixture omitted expected.json:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    let expected: Value = serde_json::from_slice(&fs::read(expected_path)?)?;

    let line = weighted_line()?;
    let edited = evaluate_modifiers(
        &line,
        &[modifier(
            "vertex_weight_edit",
            json!({"vertex_group":"A","falloff_type":"SHARP","normalize":false}),
        )?],
    )?;
    assert_material_change(
        "vertex_weight_edit",
        &expected["baseline_vertex_weight_edit"],
        &expected["vertex_weight_edit"],
        1.0e-5,
    )?;
    assert_parity(
        "vertex_weight_edit",
        &json!(vertex_weights(&edited, "A")?),
        &expected["vertex_weight_edit"],
        1.0e-5,
    )?;

    let mixed = evaluate_modifiers(
        &line,
        &[modifier(
            "vertex_weight_mix",
            json!({"vertex_group_a":"A","vertex_group_b":"B","mix_mode":"ADD","mix_set":"ALL","normalize":false}),
        )?],
    )?;
    assert_material_change(
        "vertex_weight_mix",
        &expected["baseline_vertex_weight_mix"],
        &expected["vertex_weight_mix"],
        1.0e-5,
    )?;
    assert_parity(
        "vertex_weight_mix",
        &json!(vertex_weights(&mixed, "A")?),
        &expected["vertex_weight_mix"],
        1.0e-5,
    )?;

    let falloff_mesh = weight_falloff_mesh()?;
    for falloff in [
        "LINEAR",
        "SMOOTH",
        "ROOT",
        "ICON_SPHERECURVE",
        "STEP",
        "SHARP",
    ] {
        let falloff_result = evaluate_modifiers(
            &falloff_mesh,
            &[modifier(
                "vertex_weight_edit",
                json!({"vertex_group":"A","falloff_type":falloff,"normalize":false}),
            )?],
        )?;
        let actual = json!(vertex_weights(&falloff_result, "A")?);
        if falloff != "LINEAR" {
            assert_material_change(
                &format!("vertex_weight_edit {falloff} falloff"),
                &expected["baseline_vertex_weight_edit_falloffs"][falloff],
                &expected["vertex_weight_edit_falloffs"][falloff],
                1.0e-5,
            )?;
        }
        assert_parity(
            &format!("vertex_weight_edit {falloff} falloff"),
            &actual,
            &expected["vertex_weight_edit_falloffs"][falloff],
            1.0e-5,
        )?;
    }

    let mix_modes_mesh = weight_mix_modes_mesh()?;
    for mix_mode in [
        "SET", "ADD", "SUB", "MUL", "DIV", "DIF", "AVG", "MIN", "MAX",
    ] {
        let mix_result = evaluate_modifiers(
            &mix_modes_mesh,
            &[modifier(
                "vertex_weight_mix",
                json!({
                    "vertex_group_a":"A","vertex_group_b":"B",
                    "mix_mode":mix_mode,"mix_set":"ALL","normalize":false
                }),
            )?],
        )?;
        let actual = json!(vertex_weights(&mix_result, "A")?);
        assert_material_change(
            &format!("vertex_weight_mix {mix_mode}"),
            &expected["baseline_vertex_weight_mix_modes"][mix_mode],
            &expected["vertex_weight_mix_modes"][mix_mode],
            1.0e-5,
        )?;
        assert_parity(
            &format!("vertex_weight_mix {mix_mode}"),
            &actual,
            &expected["vertex_weight_mix_modes"][mix_mode],
            1.0e-5,
        )?;
    }

    let proximity_subject_world = DMat4::from_scale_rotation_translation(
        DVec3::new(2.0, 1.0, 1.0),
        DQuat::IDENTITY,
        DVec3::new(2.0, -1.0, 0.0),
    );
    let proximity_target_world = DMat4::from_translation(DVec3::new(3.5, -1.0, 0.0));
    let mut proximity_operands = BTreeMap::new();
    proximity_operands.insert(
        "target".to_owned(),
        vec![scene_operand(
            Mesh::default(),
            proximity_subject_world.inverse() * proximity_target_world,
            None,
        )],
    );
    let proximity = modifier(
        "vertex_weight_proximity",
        json!({"target":"target","vertex_group":"A","proximity_mode":"OBJECT","proximity_geometry":["VERTEX"],"min_dist":0.0,"max_dist":3.0,"normalize":false}),
    )?;
    let proximity_result = evaluate_with_context(
        &line,
        &proximity,
        &proximity_operands,
        None,
        proximity_subject_world,
    )?;
    assert_material_change(
        "vertex_weight_proximity OBJECT",
        &expected["baseline_vertex_weight_proximity"],
        &expected["vertex_weight_proximity"],
        1.0e-5,
    )?;
    assert_wrong_candidate_fails(
        "vertex_weight_proximity OBJECT world-space origins",
        &expected["vertex_weight_proximity"],
        &json!([0.25, 0.25]),
        1.0e-5,
    )?;
    assert_parity(
        "vertex_weight_proximity",
        &json!(vertex_weights(&proximity_result, "A")?),
        &expected["vertex_weight_proximity"],
        1.0e-5,
    )?;

    let mut geometry_operands = BTreeMap::new();
    geometry_operands.insert(
        "target".to_owned(),
        vec![scene_operand(
            Mesh::from_positions_and_faces(vec![DVec3::ZERO], vec![])?,
            DMat4::IDENTITY,
            None,
        )],
    );
    let geometry_modifier = modifier(
        "vertex_weight_proximity",
        json!({"target":"target","vertex_group":"A","proximity_mode":"GEOMETRY","proximity_geometry":["VERTEX"],"min_dist":0.0,"max_dist":1.0,"normalize":false}),
    )?;
    let geometry_result = evaluate_with_context(
        &line,
        &geometry_modifier,
        &geometry_operands,
        None,
        DMat4::IDENTITY,
    )?;
    assert_material_change(
        "vertex_weight_proximity GEOMETRY/VERTEX",
        &expected["baseline_vertex_weight_proximity_geometry"],
        &expected["vertex_weight_proximity_geometry"],
        1.0e-5,
    )?;
    assert_parity(
        "vertex_weight_proximity GEOMETRY/VERTEX",
        &json!(vertex_weights(&geometry_result, "A")?),
        &expected["vertex_weight_proximity_geometry"],
        1.0e-5,
    )?;

    let proximity_surface_mesh = Mesh::from_positions_and_faces(
        vec![
            DVec3::ZERO,
            DVec3::new(2.0, 0.0, 0.0),
            DVec3::new(0.0, 2.0, 0.0),
        ],
        vec![vec![0, 1, 2]],
    )?;
    let mut proximity_geometry_subject = Mesh::from_positions_and_faces(
        vec![
            DVec3::new(0.5, 0.5, 0.25),
            DVec3::new(1.5, 1.5, 0.0),
            DVec3::new(2.5, 0.0, 0.0),
            DVec3::new(0.5, 0.0, 1.2),
        ],
        vec![],
    )?;
    proximity_geometry_subject.attributes.insert(
        "vertex_groups".to_owned(),
        json!({"A":{"v0":0.1,"v1":0.3,"v2":0.6,"v3":0.9}}),
    );
    let mut proximity_surface_operands = BTreeMap::new();
    proximity_surface_operands.insert(
        "target".to_owned(),
        vec![scene_operand(proximity_surface_mesh, DMat4::IDENTITY, None)],
    );
    for geometry in ["EDGE", "FACE"] {
        for falloff in [
            "LINEAR",
            "SMOOTH",
            "ROOT",
            "ICON_SPHERECURVE",
            "STEP",
            "SHARP",
        ] {
            let result_mesh = evaluate_with_context(
                &proximity_geometry_subject,
                &modifier(
                    "vertex_weight_proximity",
                    json!({
                        "target":"target","vertex_group":"A",
                        "proximity_mode":"GEOMETRY","proximity_geometry":[geometry],
                        "min_dist":0.0,"max_dist":1.5,"falloff_type":falloff,
                        "normalize":false
                    }),
                )?,
                &proximity_surface_operands,
                None,
                DMat4::IDENTITY,
            )?;
            let key = format!("{geometry}_{falloff}");
            let label = format!("vertex_weight_proximity GEOMETRY/{geometry} {falloff}");
            assert_material_change(
                &label,
                &expected["baseline_vertex_weight_proximity_geometry_falloffs"][&key],
                &expected["vertex_weight_proximity_geometry_falloffs"][&key],
                1.0e-5,
            )?;
            assert_parity(
                &label,
                &json!(vertex_weights(&result_mesh, "A")?),
                &expected["vertex_weight_proximity_geometry_falloffs"][&key],
                1.0e-5,
            )?;
        }
    }

    let weighted_mesh = weighted_normal_mesh(false)?;
    let weighted = evaluate_modifiers(
        &weighted_mesh,
        &[modifier(
            "weighted_normal",
            json!({"mode":"FACE_AREA","weight":50,"keep_sharp":false}),
        )?],
    )?;
    let weighted_corners = blender_quantize_loop_normals(
        &blender,
        &root,
        &weighted_mesh,
        &corner_normals(&weighted)?,
    )?;
    assert_normal_stimulus(
        "weighted_normal FACE_AREA",
        &expected["baseline_weighted_normal"],
        &expected["weighted_normal"],
        1.0e-5,
    )?;
    assert_normal_parity(
        "weighted_normal FACE_AREA",
        &weighted_corners,
        &expected["weighted_normal"],
        1.0e-5,
    )?;

    let sharp_mesh = weighted_normal_mesh(true)?;
    let weighted_sharp = evaluate_modifiers(
        &sharp_mesh,
        &[modifier(
            "weighted_normal",
            json!({"mode":"FACE_AREA","weight":50,"keep_sharp":true}),
        )?],
    )?;
    assert_normal_stimulus(
        "weighted_normal keep_sharp",
        &expected["baseline_weighted_normal_sharp"],
        &expected["weighted_normal_sharp"],
        1.0e-5,
    )?;
    assert_normal_parity(
        "weighted_normal keep_sharp",
        &blender_quantize_loop_normals(
            &blender,
            &root,
            &sharp_mesh,
            &corner_normals(&weighted_sharp)?,
        )?,
        &expected["weighted_normal_sharp"],
        1.0e-5,
    )?;

    let mut weighted_face_mesh = weighted_mesh.clone();
    weighted_face_mesh.attributes.insert(
        "__mod_weightednormals_faceweight".to_owned(),
        json!({"domain":"face","type":"int","values":{"f0":0,"f1":1,"f2":2}}),
    );
    let weighted_face = evaluate_modifiers(
        &weighted_face_mesh,
        &[modifier(
            "weighted_normal",
            json!({"mode":"FACE_AREA","weight":50,"keep_sharp":false,"use_face_influence":true}),
        )?],
    )?;
    assert_normal_stimulus(
        "weighted_normal use_face_influence",
        &expected["baseline_weighted_normal_face_influence"],
        &expected["weighted_normal_face_influence"],
        1.0e-5,
    )?;
    assert_normal_parity(
        "weighted_normal use_face_influence",
        &blender_quantize_loop_normals(
            &blender,
            &root,
            &weighted_face_mesh,
            &corner_normals(&weighted_face)?,
        )?,
        &expected["weighted_normal_face_influence"],
        1.0e-5,
    )?;

    let mut normal_operands = BTreeMap::new();
    normal_operands.insert(
        "target".to_owned(),
        vec![scene_operand(
            Mesh::default(),
            DMat4::from_translation(DVec3::new(2.0, 0.0, 0.0)),
            None,
        )],
    );
    let normal_edit = modifier(
        "normal_edit",
        json!({"mode":"RADIAL","target":"target","mix_mode":"COPY","mix_factor":1.0,"mix_limit":std::f64::consts::PI,"use_direction_parallel":false}),
    )?;
    let normals =
        evaluate_with_context(&quad, &normal_edit, &normal_operands, None, DMat4::IDENTITY)?;
    assert_normal_stimulus(
        "normal_edit RADIAL/COPY",
        &expected["baseline_normal_edit"],
        &expected["normal_edit"],
        1.0e-4,
    )?;
    assert_normal_parity(
        "normal_edit RADIAL/COPY",
        &normals.attributes["custom_normal"]["values"]["f0"],
        &expected["normal_edit"],
        1.0e-4,
    )?;

    let directional_target = DMat4::from_translation(DVec3::new(2.0, -1.0, 0.0));
    let mut directional_operands = BTreeMap::new();
    directional_operands.insert(
        "target".to_owned(),
        vec![scene_operand(Mesh::default(), directional_target, None)],
    );
    let directional = modifier(
        "normal_edit",
        json!({"mode":"DIRECTIONAL","target":"target","offset":[0.5,0.1,0.0],"mix_mode":"ADD","mix_factor":0.65,"mix_limit":std::f64::consts::PI,"use_direction_parallel":true}),
    )?;
    let directional_result = evaluate_with_context(
        &quad,
        &directional,
        &directional_operands,
        None,
        DMat4::IDENTITY,
    )?;
    assert_normal_stimulus(
        "normal_edit DIRECTIONAL/ADD",
        &expected["baseline_normal_edit_directional"],
        &expected["normal_edit_directional"],
        1.0e-4,
    )?;
    assert_normal_parity(
        "normal_edit DIRECTIONAL/ADD",
        &directional_result.attributes["custom_normal"]["values"]["f0"],
        &expected["normal_edit_directional"],
        1.0e-4,
    )?;

    let project_mesh = uv_project_mesh_for_parity()?;
    let camera = CameraData {
        lens_mm: 50.0,
        sensor_width_mm: 36.0,
        ..CameraData::default()
    };
    let projectors = BTreeMap::from([(
        "projectors".to_owned(),
        vec![
            scene_operand(
                Mesh::default(),
                DMat4::from_translation(DVec3::new(1.0, 0.5, 4.0)),
                Some(camera.clone()),
            ),
            scene_operand(
                Mesh::default(),
                DMat4::from_translation(DVec3::new(5.0, 0.5, 0.5))
                    * DMat4::from_quat(DQuat::from_rotation_y(std::f64::consts::FRAC_PI_2)),
                Some(camera),
            ),
        ],
    )]);
    let projected = evaluate_with_context(
        &project_mesh,
        &modifier(
            "uv_project",
            json!({"projectors":["camera_z","camera_x"],"uv_layer":"UVMap","aspect_x":1.0,"aspect_y":1.0,"scale_x":1.0,"scale_y":1.0}),
        )?,
        &projectors,
        None,
        DMat4::IDENTITY,
    )?;
    let projected_uvs = json!([
        projected.attributes["uv_map"][0]["uv"],
        projected.attributes["uv_map"][1]["uv"]
    ]);
    assert_material_change(
        "uv_project multi-projector",
        &expected["baseline_uv_project"],
        &expected["uv_project"],
        1.0e-5,
    )?;
    assert_wrong_candidate_fails(
        "uv_project per-face best-projector selection",
        &expected["uv_project"],
        &expected["uv_project_average_wrong"],
        1.0e-5,
    )?;
    assert_parity(
        "uv_project per-face best-projector",
        &projected_uvs,
        &expected["uv_project"],
        1.0e-5,
    )?;

    let orthographic = evaluate_with_context(
        &project_mesh,
        &modifier(
            "uv_project",
            json!({
                "projectors":["orthographic"],
                "uv_layer":"UVMap",
                "aspect_x":1.4,"aspect_y":1.2,
                "scale_x":0.8,"scale_y":1.3
            }),
        )?,
        &BTreeMap::from([(
            "projectors".to_owned(),
            vec![scene_operand(
                Mesh::default(),
                DMat4::from_translation(DVec3::new(0.25, 0.4, 3.0)),
                Some(CameraData {
                    projection: potter_core::model::CameraProjection::Orthographic,
                    ortho_scale: 3.5,
                    ..CameraData::default()
                }),
            )],
        )]),
        None,
        DMat4::IDENTITY,
    )?;
    let orthographic_uvs = json!([
        orthographic.attributes["uv_map"][0]["uv"],
        orthographic.attributes["uv_map"][1]["uv"]
    ]);
    assert_material_change(
        "uv_project orthographic aspect and scale",
        &expected["baseline_uv_project_orthographic"],
        &expected["uv_project_orthographic"],
        1.0e-5,
    )?;
    assert_parity(
        "uv_project orthographic aspect and scale",
        &orthographic_uvs,
        &expected["uv_project_orthographic"],
        1.0e-5,
    )?;

    let orthographic_clamped = evaluate_with_context(
        &project_mesh,
        &modifier(
            "uv_project",
            json!({
                "projectors":["orthographic"],
                "uv_layer":"UVMap",
                "aspect_x":1.4,"aspect_y":0.7,
                "scale_x":0.8,"scale_y":1.3
            }),
        )?,
        &BTreeMap::from([(
            "projectors".to_owned(),
            vec![scene_operand(
                Mesh::default(),
                DMat4::from_translation(DVec3::new(0.25, 0.4, 3.0)),
                Some(CameraData {
                    projection: potter_core::model::CameraProjection::Orthographic,
                    ortho_scale: 3.5,
                    ..CameraData::default()
                }),
            )],
        )]),
        None,
        DMat4::IDENTITY,
    )?;
    let orthographic_clamped_uvs = json!([
        orthographic_clamped.attributes["uv_map"][0]["uv"],
        orthographic_clamped.attributes["uv_map"][1]["uv"]
    ]);
    assert_material_change(
        "uv_project orthographic aspect clamp",
        &expected["baseline_uv_project_orthographic_clamped"],
        &expected["uv_project_orthographic_clamped"],
        1.0e-5,
    )?;
    assert_parity(
        "uv_project orthographic aspect clamp",
        &orthographic_clamped_uvs,
        &expected["uv_project_orthographic_clamped"],
        1.0e-5,
    )?;

    let warped = evaluate_modifiers(
        &quad,
        &[modifier(
            "uv_warp",
            json!({"center":[0.5,0.5],"axis_u":"X","axis_v":"Y","offset":[0.25,-0.5],"scale":[1.25,0.75],"rotation":0.35,"uv_layer":"UVMap"}),
        )?],
    )?;
    assert_material_change(
        "uv_warp scale/rotation",
        &expected["baseline_uv_warp"],
        &expected["uv_warp"],
        1.0e-5,
    )?;
    assert_parity(
        "uv_warp scale/rotation",
        &warped.attributes["uv_map"][0]["uv"],
        &expected["uv_warp"],
        1.0e-5,
    )?;

    let mut warp_transform_mesh = quad.clone();
    warp_transform_mesh.attributes.insert(
        "vertex_groups".to_owned(),
        json!({"A":{"v0":0.1,"v1":0.4,"v2":0.7,"v3":1.0}}),
    );
    let warp_from = DMat4::from_scale_rotation_translation(
        DVec3::new(1.1, 0.9, 1.0),
        DQuat::from_rotation_z(0.3),
        DVec3::new(0.25, 0.15, 0.0),
    );
    let warp_to = DMat4::from_scale_rotation_translation(
        DVec3::new(0.8, 1.3, 1.0),
        DQuat::from_rotation_z(-0.25),
        DVec3::new(-0.2, 0.1, 0.0),
    );
    let warp_operands = BTreeMap::from([
        (
            "object_from".to_owned(),
            vec![scene_operand(Mesh::default(), warp_from, None)],
        ),
        (
            "object_to".to_owned(),
            vec![scene_operand(Mesh::default(), warp_to, None)],
        ),
    ]);
    let warp_transform_result = evaluate_with_context(
        &warp_transform_mesh,
        &modifier(
            "uv_warp",
            json!({
                "object_from":"from","object_to":"to","vertex_group":"A",
                "center":[0.5,0.5],"axis_u":"X","axis_v":"Y","uv_layer":"UVMap"
            }),
        )?,
        &warp_operands,
        None,
        DMat4::IDENTITY,
    )?;
    assert_material_change(
        "uv_warp object transforms and vertex group",
        &expected["baseline_uv_warp_transforms"],
        &expected["uv_warp_transforms"],
        1.0e-5,
    )?;
    assert_parity(
        "uv_warp object transforms and vertex group",
        &warp_transform_result.attributes["uv_map"][0]["uv"],
        &expected["uv_warp_transforms"],
        1.0e-5,
    )?;

    let mut source = quad.clone();
    source.attributes.insert(
        "vertex_groups".to_owned(),
        json!({"A":{"v0":0.9,"v1":0.8,"v2":0.7,"v3":0.6}}),
    );
    let mut destination = quad.clone();
    destination.attributes.insert(
        "vertex_groups".to_owned(),
        json!({"A":{"v0":0.1,"v1":0.2,"v2":0.3,"v3":0.4}}),
    );
    let mut source_operands = BTreeMap::new();
    source_operands.insert(
        "object".to_owned(),
        vec![scene_operand(source, DMat4::IDENTITY, None)],
    );
    let transferred = evaluate_with_context(
        &destination,
        &modifier(
            "data_transfer",
            json!({"object":"source","use_object_transform":true,"use_vert_data":true,"data_types_verts":["VGROUP_WEIGHTS"],"vert_mapping":"TOPOLOGY","mix_mode":"REPLACE","mix_factor":1.0}),
        )?,
        &source_operands,
        None,
        DMat4::IDENTITY,
    )?;
    assert_material_change(
        "data_transfer VGROUP_WEIGHTS/TOPOLOGY",
        &expected["baseline_data_transfer"],
        &expected["data_transfer"],
        1.0e-5,
    )?;
    assert_parity(
        "data_transfer VGROUP_WEIGHTS/TOPOLOGY",
        &json!(vertex_weights(&transferred, "A")?),
        &expected["data_transfer"],
        1.0e-5,
    )?;

    let mut interpolation_source = quad.clone();
    interpolation_source.attributes.insert(
        "vertex_groups".to_owned(),
        json!({"A":{"v0":0.0,"v1":1.0,"v2":1.0,"v3":0.0}}),
    );
    for (mapping, point, key) in [
        (
            "NEAREST",
            DVec3::new(0.1, 0.1, 0.0),
            "data_transfer_nearest",
        ),
        (
            "POLYINTERP_NEAREST",
            DVec3::splat(0.5),
            "data_transfer_polyinterp",
        ),
    ] {
        let mut point_destination = Mesh::from_positions_and_faces(vec![point], vec![])?;
        point_destination
            .attributes
            .insert("vertex_groups".to_owned(), json!({"A":{"v0":0.2}}));
        let mut point_operands = BTreeMap::new();
        point_operands.insert(
            "object".to_owned(),
            vec![scene_operand(
                interpolation_source.clone(),
                DMat4::IDENTITY,
                None,
            )],
        );
        let transferred = evaluate_with_context(
            &point_destination,
            &modifier(
                "data_transfer",
                json!({
                    "object":"source","use_object_transform":true,
                    "use_vert_data":true,"data_types_verts":["VGROUP_WEIGHTS"],
                    "vert_mapping":mapping,"mix_mode":"REPLACE","mix_factor":1.0
                }),
            )?,
            &point_operands,
            None,
            DMat4::IDENTITY,
        )?;
        assert_material_change(
            &format!("data_transfer VGROUP_WEIGHTS/{mapping}"),
            &expected[&format!("baseline_{key}")],
            &expected[key],
            1.0e-5,
        )?;
        assert_parity(
            &format!("data_transfer VGROUP_WEIGHTS/{mapping}"),
            &json!(vertex_weights(&transferred, "A")?),
            &expected[key],
            1.0e-5,
        )?;
    }
    let mut domain_source = quad.clone();
    domain_source.attributes.insert(
        "uv_map".to_owned(),
        json!([{"face_id":0,"layer":"UVMap","uv":[[0.25,0.25],[1.25,0.25],[1.25,1.25],[0.25,1.25]]}]),
    );
    domain_source
        .attributes
        .insert("sharp_edges".to_owned(), json!([2]));
    domain_source.attributes.insert(
        "uv_seam".to_owned(),
        json!({"domain":"edge","type":"bool","values":{"e0":false,"e1":true,"e2":false,"e3":false}}),
    );
    domain_source.attributes.insert(
        "crease_edge".to_owned(),
        json!({"domain":"edge","type":"float","values":{"e0":0.75,"e1":0.0,"e2":0.0,"e3":0.0}}),
    );
    domain_source.attributes.insert(
        "custom_normal".to_owned(),
        json!({"domain":"corner","type":"float3","values":{"f0":[[0.0,0.0,1.0],[0.0,0.0,1.0],[0.0,0.0,1.0],[0.0,0.0,1.0]]}}),
    );
    domain_source.attributes.insert(
        "color".to_owned(),
        json!({"domain":"corner","type":"color","values":{"f0":[[0.2,0.3,0.4,1.0],[0.3,0.3,0.4,1.0],[0.4,0.3,0.4,1.0],[0.5,0.3,0.4,1.0]]}}),
    );
    domain_source.attributes.insert(
        "smooth".to_owned(),
        json!({"domain":"face","type":"bool","values":{"f0":true}}),
    );
    domain_source.attributes.insert(
        "freestyle_face".to_owned(),
        json!({"domain":"face","type":"bool","values":{"f0":true}}),
    );
    let mut domain_operands = BTreeMap::new();
    domain_operands.insert(
        "object".to_owned(),
        vec![scene_operand(domain_source, DMat4::IDENTITY, None)],
    );
    let domain_transfer = modifier(
        "data_transfer",
        json!({
            "object":"domain_source","use_object_transform":true,
            "use_edge_data":true,"data_types_edges":["SHARP_EDGE","SEAM","CREASE"],
            "edge_mapping":"NEAREST",
            "use_loop_data":true,"data_types_loops":["CUSTOM_NORMAL","UV","COLOR_CORNER"],
            "loop_mapping":"POLYINTERP_LNORPROJ","layers_uv_select_src":"ALL","layers_uv_select_dst":"NAME",
            "use_poly_data":true,"data_types_polys":["SMOOTH","FREESTYLE_FACE"],
            "poly_mapping":"POLYINTERP_PNORPROJ","mix_mode":"REPLACE","mix_factor":1.0
        }),
    )?;
    let domain_destination = transfer_domain_destination_mesh()?;
    let domains = evaluate_with_context(
        &domain_destination,
        &domain_transfer,
        &domain_operands,
        None,
        DMat4::IDENTITY,
    )?;
    let domain_values = transfer_domain_values(&domains)?;
    assert_material_change(
        "data_transfer nonmatching edge/loop/polygon topology",
        &expected["baseline_data_transfer_domains"],
        &expected["data_transfer_domains"],
        1.0e-4,
    )?;
    assert_normal_parity(
        "data_transfer CUSTOM_NORMAL",
        &domain_values["normal"],
        &expected["data_transfer_domains"]["normal"],
        1.0e-4,
    )?;
    assert_parity(
        "data_transfer nonmatching EDGE/LOOP/POLY topology",
        &domain_values,
        &expected["data_transfer_domains"],
        1.0e-4,
    )?;
    let image = ImageData {
        width: 2,
        height: 2,
        pixels: vec![
            [0.0, 0.0, 0.0, 1.0],
            [0.25, 0.25, 0.25, 1.0],
            [0.6, 0.6, 0.6, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        tiles: BTreeMap::new(),
        interpolation: ImageInterpolation::Closest,
    };
    let subject_world = DMat4::from_scale_rotation_translation(
        DVec3::new(0.8, 0.7, 1.0),
        DQuat::from_rotation_z(0.2),
        DVec3::new(0.25, -0.25, 0.0),
    );
    let map_object_world = DMat4::from_scale_rotation_translation(
        DVec3::new(1.5, 0.75, 1.0),
        DQuat::from_rotation_z(std::f64::consts::FRAC_PI_4),
        DVec3::new(0.75, 0.25, 0.0),
    );
    for mapping in ["LOCAL", "GLOBAL", "OBJECT", "UV"] {
        let mut mask_mesh = quad.clone();
        if mapping == "UV" {
            mask_mesh.attributes.insert(
                "uv_map".to_owned(),
                json!([{
                    "face_id":0,
                    "layer":"UVMap",
                    "uv":[[0.13,0.17],[0.73,0.24],[0.84,0.81],[0.2,0.69]]
                }]),
            );
        }
        let mut operands = BTreeMap::new();
        if mapping == "OBJECT" {
            operands.insert(
                "mask_tex_map_object".to_owned(),
                vec![scene_operand(
                    Mesh::default(),
                    subject_world.inverse() * map_object_world,
                    None,
                )],
            );
        }
        let mask_modifier = modifier(
            "vertex_weight_edit",
            json!({
                "vertex_group":"A","falloff_type":"SHARP","normalize":false,
                "mask_texture":"image_mask","mask_tex_use_channel":"RED",
                "mask_tex_mapping":mapping,"mask_tex_uv_layer":"UVMap",
                "mask_tex_map_object":"mask_object"
            }),
        )?;
        let masked = evaluate_with_context(
            &mask_mesh,
            &mask_modifier,
            &operands,
            Some(&image),
            subject_world,
        )?;
        let key = format!("texture_mask_{}", mapping.to_lowercase());
        let baseline_key = format!("baseline_{key}");
        assert_material_change(
            &format!("image mask mapping {mapping}"),
            &expected[&baseline_key],
            &expected[&key],
            1.0e-5,
        )?;
        assert_parity(
            &format!("image mask mapping {mapping}"),
            &json!(vertex_weights(&masked, "A")?),
            &expected[&key],
            1.0e-5,
        )?;
    }

    for evaluation_frame in [0.25, 0.75] {
        let mut pc2_mesh = quad.clone();
        let pc2_modifier = modifier(
            "mesh_cache",
            json!({
                "resource":"pc2",
                "cache_format":"PC2",
                "play_mode":"CUSTOM",
                "time_mode":"FRAME",
                "frame_start":0.0,
                "frame_scale":1.0,
                "eval_frame":evaluation_frame,
                "forward_axis":"POS_X",
                "up_axis":"POS_Z",
                "interpolation":"LINEAR",
                "deform_mode":"OVERWRITE",
                "factor":1.0
            }),
        )?;
        evaluate_mesh_cache(
            &mut pc2_mesh,
            &pc2_modifier,
            evaluation_frame,
            24,
            1.0,
            &pc2,
        )?;
        let key = format!("{evaluation_frame:.2}");
        let expected_positions = expected["mesh_cache_pc2"]
            .get(&key)
            .ok_or("PC2 sample missing")?;
        assert_material_change(
            &format!("mesh_cache PC2 frame {evaluation_frame}"),
            &expected["baseline_mesh_cache_pc2"],
            expected_positions,
            1.0e-5,
        )?;
        assert_wrong_candidate_fails(
            &format!("mesh_cache PC2 non-rigid deformation at {evaluation_frame}"),
            expected_positions,
            &rigid_translation_candidate(&expected["baseline_mesh_cache_pc2"], expected_positions)?,
            1.0e-5,
        )?;
        assert_parity(
            &format!("mesh_cache PC2 interpolated frame {evaluation_frame}"),
            &json!(
                pc2_mesh
                    .vertices
                    .iter()
                    .map(|vertex| vertex.co.to_array())
                    .collect::<Vec<_>>()
            ),
            expected_positions,
            1.0e-5,
        )?;
    }

    for evaluation_time in [0.25, 0.75] {
        let mut mdd_mesh = quad.clone();
        let mdd_modifier = modifier(
            "mesh_cache",
            json!({
                "resource":"mdd",
                "cache_format":"MDD",
                "play_mode":"CUSTOM",
                "time_mode":"TIME",
                "eval_time":evaluation_time,
                "interpolation":"LINEAR",
                "deform_mode":"OVERWRITE",
                "factor":1.0
            }),
        )?;
        evaluate_mesh_cache(&mut mdd_mesh, &mdd_modifier, 1.0, 24, 1.0, &mdd)?;
        let key = format!("{evaluation_time:.2}");
        let expected_positions = expected["mesh_cache_mdd"]
            .get(&key)
            .ok_or("MDD sample missing")?;
        assert_material_change(
            &format!("mesh_cache MDD time {evaluation_time}"),
            &expected["baseline_mesh_cache_mdd"],
            expected_positions,
            1.0e-5,
        )?;
        assert_wrong_candidate_fails(
            &format!("mesh_cache MDD non-rigid deformation at {evaluation_time}"),
            expected_positions,
            &rigid_translation_candidate(&expected["baseline_mesh_cache_mdd"], expected_positions)?,
            1.0e-5,
        )?;
        assert_parity(
            &format!("mesh_cache MDD interpolated time {evaluation_time}"),
            &json!(
                mdd_mesh
                    .vertices
                    .iter()
                    .map(|vertex| vertex.co.to_array())
                    .collect::<Vec<_>>()
            ),
            expected_positions,
            1.0e-5,
        )?;
    }

    for (cache_format, time_mode, evaluation, bytes, key) in [
        ("PC2", "TIME", 0.4, &pc2, "mesh_cache_integrate_time"),
        ("MDD", "FACTOR", 0.35, &mdd, "mesh_cache_integrate_factor"),
    ] {
        let mut integrated_mesh = quad.clone();
        integrated_mesh.attributes.insert(
            "vertex_groups".to_owned(),
            json!({"A":{"v0":0.1,"v1":0.4,"v2":0.7,"v3":1.0}}),
        );
        let cache_modifier = modifier(
            "mesh_cache",
            json!({
                "resource":cache_format.to_lowercase(),
                "cache_format":cache_format,
                "play_mode":"CUSTOM",
                "time_mode":time_mode,
                "eval_time":evaluation,
                "eval_factor":evaluation,
                "interpolation":"LINEAR",
                "deform_mode":"INTEGRATE",
                "vertex_group":"A",
                "factor":0.8,
                "forward_axis":"POS_X",
                "up_axis":"POS_Z"
            }),
        )?;
        evaluate_mesh_cache(&mut integrated_mesh, &cache_modifier, 1.0, 24, 1.0, bytes)?;
        let actual = json!(
            integrated_mesh
                .vertices
                .iter()
                .map(|vertex| vertex.co.to_array())
                .collect::<Vec<_>>()
        );
        let label = format!("mesh_cache INTEGRATE {cache_format} {time_mode}");
        assert_material_change(
            &label,
            &expected[&format!("baseline_{key}")],
            &expected[key],
            1.0e-5,
        )?;
        assert_parity(&label, &actual, &expected[key], 1.0e-5)?;
    }
    Ok(())
}

fn quad_mesh_for_parity() -> Result<Mesh, Box<dyn Error>> {
    let mut mesh = Mesh::from_positions_and_faces(
        vec![DVec3::ZERO, DVec3::X, DVec3::new(1.0, 1.0, 0.0), DVec3::Y],
        vec![vec![0, 1, 2, 3]],
    )?;
    mesh.attributes.insert(
        "uv_map".to_owned(),
        json!([{"face_id":0,"layer":"UVMap","uv":[[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]]}]),
    );
    mesh.attributes.insert(
        "vertex_groups".to_owned(),
        json!({"A":{"v0":0.25,"v1":0.5,"v2":0.75,"v3":1.0}}),
    );
    Ok(mesh)
}

fn weight_falloff_mesh() -> Result<Mesh, Box<dyn Error>> {
    let mut mesh = Mesh::from_positions_and_faces(
        vec![
            DVec3::ZERO,
            DVec3::X,
            DVec3::new(2.0, 0.5, 0.0),
            DVec3::new(3.0, 1.0, 0.0),
        ],
        vec![],
    )?;
    mesh.attributes.insert(
        "vertex_groups".to_owned(),
        json!({"A":{"v0":0.12,"v1":0.31,"v2":0.64,"v3":0.87}}),
    );
    Ok(mesh)
}

fn weight_mix_modes_mesh() -> Result<Mesh, Box<dyn Error>> {
    let mut mesh = Mesh::from_positions_and_faces(
        vec![
            DVec3::ZERO,
            DVec3::X,
            DVec3::new(2.0, 0.5, 0.0),
            DVec3::new(3.0, 1.0, 0.0),
        ],
        vec![],
    )?;
    mesh.attributes.insert(
        "vertex_groups".to_owned(),
        json!({
            "A":{"v0":0.17,"v1":0.39,"v2":0.68,"v3":0.82},
            "B":{"v0":0.71,"v1":0.24,"v2":0.45,"v3":0.16}
        }),
    );
    Ok(mesh)
}

fn weighted_normal_mesh(sharp_edge: bool) -> Result<Mesh, Box<dyn Error>> {
    let mut mesh = Mesh::from_positions_and_faces(
        vec![
            DVec3::ZERO,
            DVec3::new(4.0, 0.0, 0.0),
            DVec3::Y,
            DVec3::new(0.0, 0.0, 3.0),
            DVec3::new(-2.0, 0.0, 0.0),
        ],
        vec![vec![0, 1, 2], vec![0, 2, 3], vec![0, 3, 4]],
    )?;
    mesh.faces[2].material_index = 1;
    if sharp_edge {
        let edge = mesh
            .edges
            .iter()
            .find(|edge| {
                (edge.vertices[0] == 0 && edge.vertices[1] == 2)
                    || (edge.vertices[0] == 2 && edge.vertices[1] == 0)
            })
            .ok_or("weighted-normal sharp edge missing")?;
        mesh.attributes
            .insert("sharp_edges".to_owned(), json!([edge.id]));
    }
    Ok(mesh)
}

fn uv_project_mesh_for_parity() -> Result<Mesh, Box<dyn Error>> {
    let mut mesh = Mesh::from_positions_and_faces(
        vec![
            DVec3::ZERO,
            DVec3::new(2.0, 0.0, 0.0),
            DVec3::new(2.0, 1.0, 0.0),
            DVec3::Y,
            DVec3::new(3.0, 0.0, 0.0),
            DVec3::new(3.0, 2.0, 0.0),
            DVec3::new(3.0, 2.0, 1.0),
            DVec3::new(3.0, 0.0, 1.0),
        ],
        vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
    )?;
    mesh.attributes.insert(
        "uv_map".to_owned(),
        json!([
            {"face_id":0,"layer":"UVMap","uv":[[0,0],[1,0],[1,1],[0,1]]},
            {"face_id":1,"layer":"UVMap","uv":[[0,0],[1,0],[1,1],[0,1]]}
        ]),
    );
    Ok(mesh)
}

fn corner_normals(mesh: &Mesh) -> Result<Value, Box<dyn Error>> {
    let values = mesh
        .attributes
        .get("custom_normal")
        .and_then(|attribute| attribute.get("values"))
        .and_then(Value::as_object)
        .ok_or("custom corner normals missing")?;
    let mut result = Vec::new();
    for face in &mesh.faces {
        let corners = values
            .get(&format!("f{}", face.id))
            .and_then(Value::as_array)
            .ok_or("custom face corner normals missing")?;
        result.extend(corners.iter().cloned());
    }
    Ok(Value::Array(result))
}

fn transfer_domain_destination_mesh() -> Result<Mesh, Box<dyn Error>> {
    let mut mesh = Mesh::from_positions_and_faces(
        vec![
            DVec3::ZERO,
            DVec3::new(1.2, 0.1, 0.0),
            DVec3::new(1.6, 1.3, 0.0),
            DVec3::Y,
        ],
        vec![vec![0, 1, 2], vec![0, 2, 3]],
    )?;
    mesh.attributes.insert(
        "uv_map".to_owned(),
        json!([
            {"face_id":0,"layer":"UVMap","uv":[[0,0],[1,0],[1,1]]},
            {"face_id":1,"layer":"UVMap","uv":[[0,0],[1,1],[0,1]]}
        ]),
    );
    mesh.attributes.insert("sharp_edges".to_owned(), json!([]));
    mesh.attributes.insert(
        "uv_seam".to_owned(),
        json!({"domain":"edge","type":"bool","values":{"e0":false,"e1":false,"e2":false,"e3":false,"e4":false}}),
    );
    mesh.attributes.insert(
        "crease_edge".to_owned(),
        json!({"domain":"edge","type":"float","values":{"e0":0.0,"e1":0.0,"e2":0.0,"e3":0.0,"e4":0.0}}),
    );
    mesh.attributes.insert(
        "custom_normal".to_owned(),
        json!({"domain":"corner","type":"float3","values":{
            "f0":[[1,0,0],[1,0,0],[1,0,0]],
            "f1":[[1,0,0],[1,0,0],[1,0,0]]
        }}),
    );
    mesh.attributes.insert(
        "color".to_owned(),
        json!({"domain":"corner","type":"color","values":{
            "f0":[[0,0,0,1],[0,0,0,1],[0,0,0,1]],
            "f1":[[0,0,0,1],[0,0,0,1],[0,0,0,1]]
        }}),
    );
    mesh.attributes.insert(
        "smooth".to_owned(),
        json!({"domain":"face","type":"bool","values":{"f0":false,"f1":false}}),
    );
    mesh.attributes.insert(
        "freestyle_face".to_owned(),
        json!({"domain":"face","type":"bool","values":{"f0":false,"f1":false}}),
    );
    Ok(mesh)
}

fn transfer_domain_values(mesh: &Mesh) -> Result<Value, Box<dyn Error>> {
    let sharp_ids = mesh
        .attributes
        .get("sharp_edges")
        .and_then(Value::as_array)
        .ok_or("sharp edge attribute missing")?
        .iter()
        .filter_map(Value::as_u64)
        .filter_map(|id| u32::try_from(id).ok())
        .collect::<std::collections::BTreeSet<_>>();
    let edge_attributes = |name: &str| {
        mesh.attributes
            .get(name)
            .and_then(|attribute| attribute.get("values"))
            .and_then(Value::as_object)
    };
    let seam_values = edge_attributes("uv_seam").ok_or("seam edge values missing")?;
    let crease_values = edge_attributes("crease_edge").ok_or("crease edge values missing")?;
    let mut sharp = serde_json::Map::new();
    let mut seam = serde_json::Map::new();
    let mut crease = serde_json::Map::new();
    for edge in &mesh.edges {
        let first = edge.vertices[0].min(edge.vertices[1]);
        let second = edge.vertices[0].max(edge.vertices[1]);
        let key = format!("{first}-{second}");
        sharp.insert(key.clone(), Value::Bool(sharp_ids.contains(&edge.id)));
        seam.insert(
            key.clone(),
            Value::Bool(
                seam_values
                    .get(&format!("e{}", edge.id))
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            ),
        );
        crease.insert(
            key,
            json!(
                crease_values
                    .get(&format!("e{}", edge.id))
                    .and_then(Value::as_f64)
                    .unwrap_or_default()
            ),
        );
    }
    let mut uv = Vec::new();
    let uv_entries = mesh
        .attributes
        .get("uv_map")
        .and_then(Value::as_array)
        .ok_or("transferred UV layers missing")?;
    for face in &mesh.faces {
        let entry = uv_entries
            .iter()
            .find(|entry| {
                entry.get("face_id").and_then(Value::as_u64) == Some(u64::from(face.id))
                    && entry.get("layer").and_then(Value::as_str) == Some("UVMap")
            })
            .and_then(|entry| entry.get("uv"))
            .and_then(Value::as_array)
            .ok_or("transferred face UVs missing")?;
        uv.extend(entry.iter().cloned());
    }
    let corner_values = |name: &str| -> Result<Value, Box<dyn Error>> {
        let values = mesh
            .attributes
            .get(name)
            .and_then(|attribute| attribute.get("values"))
            .and_then(Value::as_object)
            .ok_or("transferred corner attribute missing")?;
        let mut result = Vec::new();
        for face in &mesh.faces {
            let corners = values
                .get(&format!("f{}", face.id))
                .and_then(Value::as_array)
                .ok_or("transferred face corner values missing")?;
            result.extend(corners.iter().cloned());
        }
        Ok(Value::Array(result))
    };
    let face_values = |name: &str| -> Result<Value, Box<dyn Error>> {
        let values = mesh
            .attributes
            .get(name)
            .and_then(|attribute| attribute.get("values"))
            .and_then(Value::as_object)
            .ok_or("transferred face attribute missing")?;
        mesh.faces
            .iter()
            .map(|face| {
                values
                    .get(&format!("f{}", face.id))
                    .cloned()
                    .ok_or_else(|| "transferred face value missing".into())
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()
            .map(Value::Array)
    };
    Ok(json!({
        "sharp":sharp,
        "seam":seam,
        "crease":crease,
        "uv":uv,
        "color":corner_values("color")?,
        "normal":corner_normals(mesh)?,
        "smooth":face_values("smooth")?,
        "freestyle":face_values("freestyle_face")?
    }))
}

fn assert_normal_parity(
    label: &str,
    actual: &Value,
    expected: &Value,
    tolerance: f64,
) -> Result<(), Box<dyn Error>> {
    let mut actual_values = Vec::new();
    let mut expected_values = Vec::new();
    flatten_numbers(actual, &mut actual_values)?;
    flatten_numbers(expected, &mut expected_values)?;
    assert_eq!(
        actual_values.len(),
        expected_values.len(),
        "{label} normal count"
    );
    let actual_chunks = actual_values.as_chunks::<3>().0;
    let expected_chunks = expected_values.as_chunks::<3>().0;
    let mut max_angle = 0.0_f64;
    let mut worst_actual = DVec3::ZERO;
    let mut worst_expected = DVec3::ZERO;
    for (actual, expected) in actual_chunks.iter().zip(expected_chunks.iter()) {
        let actual = DVec3::from_array(*actual).normalize_or_zero();
        let expected = DVec3::from_array(*expected).normalize_or_zero();
        let angle = actual.angle_between(expected);
        if angle > max_angle {
            max_angle = angle;
            worst_actual = actual;
            worst_expected = expected;
        }
    }
    eprintln!("Blender parity {label}: max normal angle {max_angle:.8e} radians");
    assert!(
        max_angle <= tolerance,
        "{label} normal angle {max_angle}; actual={worst_actual:?}, expected={worst_expected:?}"
    );
    Ok(())
}
