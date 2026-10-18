use std::{error::Error, fs, path::Path, process::Command};

use serde_json::{Value, json};
use tempfile::tempdir;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn init(scene: &Path) -> TestResult {
    let output = pot().arg("init").arg(scene).arg("--json").output()?;
    assert!(
        output.status.success(),
        "scene init failed: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    Ok(())
}

fn apply_response(scene: &Path, base_revision: u64, operations: Value) -> TestResult<Value> {
    let batch_path = scene.join("operations.json");
    let mut batch = json!({"schema_version": 1, "base_revision": base_revision});
    batch["operations"] = operations;
    fs::write(&batch_path, serde_json::to_vec(&batch)?)?;
    let output = pot()
        .arg("apply")
        .arg(scene)
        .arg("--file")
        .arg(batch_path)
        .arg("--json")
        .output()?;
    assert!(
        output.status.success(),
        "apply failed ({}):\nstdout={}\nstderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn mesh(scene: &Path, node_id: &str) -> TestResult<Value> {
    let document: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let data_id = document["nodes"][node_id]["data"]
        .as_str()
        .ok_or("node mesh data ID missing")?;
    let mesh = document["data_blocks"][data_id]["mesh"].clone();
    if !mesh.is_object() {
        return Err("persisted mesh is missing".into());
    }
    Ok(mesh)
}

fn mesh_positions(mesh: &Value) -> TestResult<Vec<Value>> {
    mesh["vertices"]
        .as_array()
        .ok_or("mesh vertices are missing")?
        .iter()
        .map(|vertex| {
            vertex["co"]
                .as_array()
                .map(|_| vertex["co"].clone())
                .ok_or_else(|| "mesh vertex coordinates are missing".into())
        })
        .collect()
}

#[test]
fn modifier_apply_forwards_array_fallback_warnings_and_keeps_fallback_geometry() -> TestResult {
    let cases = [
        (
            "fixed_count_plain",
            "modifier.array.count_too_high",
            false,
            json!({"fit_type":"FIXED_COUNT","count":2_147_483_647}),
        ),
        (
            "fixed_count_scene_context",
            "modifier.array.count_too_high",
            true,
            json!({"fit_type":"FIXED_COUNT","count":2_147_483_647}),
        ),
        (
            "tiny_offset_plain",
            "modifier.array.offset_too_small",
            false,
            json!({
                "fit_type":"FIT_LENGTH",
                "fit_length":4.0,
                "use_relative_offset":false,
                "use_constant_offset":true,
                "constant_offset_displace":[0.000_000_1,0.0,0.0]
            }),
        ),
        (
            "tiny_offset_scene_context",
            "modifier.array.offset_too_small",
            true,
            json!({
                "fit_type":"FIT_LENGTH",
                "fit_length":4.0,
                "use_relative_offset":false,
                "use_constant_offset":true,
                "constant_offset_displace":[0.000_000_1,0.0,0.0]
            }),
        ),
    ];

    for (case, feature_id, scene_context, mut params) in cases {
        let directory = tempdir()?;
        let scene = directory.path().join("scene");
        init(&scene)?;
        let mut setup = vec![json!({
            "op":"node.create","id":"subject","kind":"box","params":{"size":2.0}
        })];
        if scene_context {
            setup.push(json!({
                "op":"node.create","id":"offset","kind":"box","params":{"size":1.0}
            }));
            let params = params
                .as_object_mut()
                .ok_or("array modifier params are not an object")?;
            params.insert("use_object_offset".to_owned(), json!(true));
            params.insert("offset_object".to_owned(), json!("offset"));
        }
        setup.push(json!({
            "op":"modifier.create",
            "target":{"id":"subject"},
            "id":"array",
            "type":"array",
            "params":params
        }));
        apply_response(&scene, 0, Value::Array(setup))?;
        let source = mesh(&scene, "subject")?;
        let source_positions = mesh_positions(&source)?;
        let source_face_count = source["faces"]
            .as_array()
            .ok_or("source mesh faces are missing")?
            .len();

        let response = apply_response(
            &scene,
            1,
            json!([{"op":"modifier.apply","target":{"id":"subject"},"modifier_id":"array"}]),
        )?;
        assert_eq!(response["ok"], true, "{case}: {response}");
        assert!(
            response["warnings"].as_array().is_some_and(|warnings| {
                warnings
                    .iter()
                    .any(|warning| warning["details"]["feature_id"] == feature_id)
            }),
            "{case}: expected warning {feature_id}, got {}",
            response["warnings"]
        );

        let output = mesh(&scene, "subject")?;
        assert_eq!(
            mesh_positions(&output)?,
            source_positions,
            "{case}: fallback positions"
        );
        assert_eq!(
            output["faces"]
                .as_array()
                .ok_or("output faces are missing")?
                .len(),
            source_face_count,
            "{case}: fallback face count"
        );
    }
    Ok(())
}

#[test]
fn screw_apply_uses_axis_transform_without_evaluating_unselected_downstream_modifier() -> TestResult
{
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init(&scene)?;
    apply_response(
        &scene,
        0,
        json!([
            {"op":"node.create","id":"subject","kind":"box","params":{"size":2.0}},
            {"op":"node.create","id":"axis","kind":"box","params":{"size":1.0}},
            {"op":"modifier.create","target":{"id":"subject"},"id":"screw","type":"screw","params":{"axis":"Z","steps":2,"angle":1.0,"object":"axis","use_object_screw_offset":false}},
            {"op":"modifier.create","target":{"id":"subject"},"id":"unselected_unsupported_bevel","type":"bevel","params":{"width":0.1,"segments":2,"harden_normals":true}}
        ]),
    )?;

    let response = apply_response(
        &scene,
        1,
        json!([{"op":"modifier.apply","target":{"id":"subject"},"modifier_id":"screw"}]),
    )?;
    assert_eq!(response["ok"], true, "{response}");
    let positions = mesh_positions(&mesh(&scene, "subject")?)?;
    // The middle ring rotates the box corner (1, 1, 1) by half of the 1-radian turn around Z.
    let expected_mid_ring_vertex = [0.398_157, 1.357_008_1, 1.0];
    assert!(
        positions.iter().any(|position| {
            position.as_array().is_some_and(|coordinates| {
                coordinates
                    .iter()
                    .zip(expected_mid_ring_vertex)
                    .all(|(actual, expected)| {
                        actual
                            .as_f64()
                            .is_some_and(|actual| (actual - expected).abs() <= 1.0e-5)
                    })
            })
        }),
        "expected the rotated box corner in applied geometry, got {positions:?}"
    );

    let document: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let modifiers = document["nodes"]["subject"]["modifiers"]
        .as_array()
        .ok_or("subject modifiers are missing")?;
    assert_eq!(modifiers.len(), 1);
    assert_eq!(modifiers[0]["id"], "unselected_unsupported_bevel");
    Ok(())
}
