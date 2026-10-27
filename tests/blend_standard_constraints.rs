use std::{error::Error, fs, path::Path, process::Command, sync::Mutex};

use potter_core::model::SceneDoc;
use serde_json::{Value, json};
use tempfile::tempdir;

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/process.rs"]
mod process;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const MATRIX_TOLERANCE: f64 = 1.0e-5;
static BLENDER_TEST_LOCK: Mutex<()> = Mutex::new(());

fn pot_output(arguments: &[String]) -> TestResult<std::process::Output> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_pot"));
    command.args(arguments).arg("--json");
    process::run_guarded(command)
}

fn pot_json(arguments: &[String]) -> TestResult<Value> {
    let output = pot_output(arguments)?;
    if !output.status.success() {
        return Err(format!(
            "pot command failed: {}\nstdout={}\nstderr={}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        )
        .into());
    }
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn path_text(path: &Path) -> TestResult<String> {
    path.to_str()
        .map(str::to_owned)
        .ok_or_else(|| format!("path is not UTF-8: {}", path.display()).into())
}

fn matrix_parity(actual: &Value, expected: &Value, context: &str) -> TestResult {
    let actual_values = actual.as_array().ok_or("Potter matrix is missing")?;
    let expected_values = expected.as_array().ok_or("Blender matrix is missing")?;
    if actual_values.len() != 16 || expected_values.len() != 16 {
        return Err(format!(
            "{context}: expected 16 matrix values, Potter has {} and Blender has {}",
            actual_values.len(),
            expected_values.len()
        )
        .into());
    }
    let mut maximum_error = 0.0_f64;
    for (index, (actual, expected)) in actual_values.iter().zip(expected_values).enumerate() {
        let actual = actual
            .as_f64()
            .ok_or("Potter matrix entry is not numeric")?;
        let expected = expected
            .as_f64()
            .ok_or("Blender matrix entry is not numeric")?;
        let error = (actual - expected).abs();
        maximum_error = maximum_error.max(error);
        if error > MATRIX_TOLERANCE {
            return Err(format!(
                "{context}: matrix[{index}] Potter {actual} != Blender {expected} (error {error:.6e}); Potter matrix={actual_values:?}; Blender matrix={expected_values:?}"
            )
            .into());
        }
    }
    eprintln!("{context}: pot inspect matrix max error {maximum_error:.6e}");
    Ok(())
}

fn assert_fixture_parity(fixture_name: &str, blender: &Path) -> TestResult {
    let fixture_dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/standard_constraints");
    let oracle: Value =
        serde_json::from_slice(&fs::read(fixture_dir.join(format!("{fixture_name}.json")))?)?;
    let project_dir = tempdir()?;
    let project = project_dir.path().join("project");
    let project_text = path_text(&project)?;
    pot_json(&["init".to_owned(), project_text.clone()])?;

    let source = path_text(&fixture_dir.join(format!("{fixture_name}.blend")))?;
    let blender = path_text(blender)?;
    let imported = pot_json(&[
        "import".to_owned(),
        project_text.clone(),
        "--file".to_owned(),
        source,
        "--format".to_owned(),
        "blend".to_owned(),
        "--mode".to_owned(),
        "replace".to_owned(),
        "--base-revision".to_owned(),
        "0".to_owned(),
        "--blender".to_owned(),
        blender,
    ])?;
    if !imported["result"]["losses"]
        .as_array()
        .is_some_and(Vec::is_empty)
    {
        return Err(format!("{fixture_name} import reported losses: {imported}").into());
    }
    let imported_doc: SceneDoc =
        serde_json::from_slice(&fs::read(project_dir.path().join("project/scene.json"))?)?;
    let mappings = imported["result"]["id_mappings"]
        .as_object()
        .ok_or("Blender import omitted ID mappings")?;
    let owners = oracle["owners"]
        .as_array()
        .ok_or("Blender oracle omitted owners")?;
    for frame in [1, 2, 3] {
        let frame_text = frame.to_string();
        let inspected = pot_json(&[
            "inspect".to_owned(),
            project_text.clone(),
            "--frame".to_owned(),
            frame_text.clone(),
        ])?;
        let items = inspected["result"]["items"]
            .as_array()
            .ok_or("pot inspect omitted scene items")?;
        let expected_frame = oracle["frames"]
            .get(&frame_text)
            .and_then(Value::as_object)
            .ok_or_else(|| format!("Blender oracle omitted frame {frame}"))?;
        for owner in owners {
            let kind = owner["kind"]
                .as_str()
                .ok_or("oracle owner kind is missing")?;
            let name = owner["name"]
                .as_str()
                .ok_or("oracle owner name is missing")?;
            let (mapping_key, capture_key, bone_name) = if kind == "object" {
                (format!("Object:{name}"), name.to_owned(), None)
            } else if kind == "bone" {
                let armature = owner["armature"]
                    .as_str()
                    .ok_or("oracle bone armature name is missing")?;
                (
                    format!("Object:{armature}"),
                    format!("Bone:{armature}:{name}"),
                    Some((armature, name)),
                )
            } else {
                return Err(format!("unknown oracle owner kind `{kind}`").into());
            };
            let node_id = mappings
                .get(&mapping_key)
                .and_then(Value::as_str)
                .ok_or_else(|| format!("Blender import omitted mapping {mapping_key}"))?;
            let item = items
                .iter()
                .find(|item| item["id"].as_str() == Some(node_id))
                .ok_or_else(|| format!("pot inspect omitted owner {mapping_key}"))?;
            let actual = if let Some((armature, bone)) = bone_name {
                let armature_node = imported_doc
                    .nodes
                    .values()
                    .find(|node| node.name == armature)
                    .ok_or_else(|| format!("imported armature {armature} is missing"))?;
                let armature_data = armature_node
                    .data
                    .as_ref()
                    .and_then(|id| imported_doc.data_blocks.get(id))
                    .and_then(|data| data.armature.as_ref())
                    .ok_or_else(|| format!("imported armature data for {armature} is missing"))?;
                let (bone_id, _) = armature_data
                    .bones
                    .iter()
                    .find(|(_, candidate)| candidate.name == bone)
                    .ok_or_else(|| format!("imported bone {armature}:{bone} is missing"))?;
                let bone_id = bone_id.as_str();
                item["evaluated_bones"].get(bone_id).ok_or_else(|| {
                    format!("pot inspect omitted evaluated bone {armature}:{bone}")
                })?
            } else {
                &item["transform"]["world"]["matrix"]
            };
            let expected = expected_frame
                .get(&capture_key)
                .ok_or_else(|| format!("Blender oracle omitted {capture_key} at frame {frame}"))?;
            matrix_parity(
                actual,
                expected,
                &format!("{fixture_name} {capture_key} frame {frame}"),
            )?;
        }
    }
    Ok(())
}

#[test]
fn standard_constraint_modes_match_blender_through_single_import_and_inspect() -> TestResult {
    let _guard = BLENDER_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(blender) = blender_checked::blender_executable() else {
        return Err("Blender is required for standard constraint parity".into());
    };
    assert_fixture_parity("combined_constraints", &blender)
}

#[test]
fn limit_distance_stretch_to_and_child_of_capture_blender_initial_state() -> TestResult {
    let fixture_dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/standard_constraints");
    let oracle: Value = serde_json::from_slice(&fs::read(
        fixture_dir.join("initial_constraint_captures.json"),
    )?)?;
    let constraint_records = oracle["constraints"]
        .as_array()
        .ok_or("Blender capture oracle omitted constraints")?;
    assert_eq!(constraint_records.len(), 9);
    for (key, constraint_type) in [
        ("bone_limit_distance", "limit_distance"),
        ("bone_stretch_to", "stretch_to"),
        ("bone_child_of", "child_of"),
    ] {
        let capture = constraint_records
            .iter()
            .find(|capture| capture["key"] == key)
            .ok_or_else(|| format!("Blender capture oracle omitted pose constraint {key}"))?;
        assert_eq!(capture["type"], constraint_type);
        assert_eq!(capture["owner_bone"], key);
        let influence = capture["influence"]
            .as_f64()
            .ok_or("pose-bone capture influence is missing")?;
        assert!(influence > 0.0 && influence < 1.0);
    }
    let directory = tempdir()?;
    let project = directory.path().join("project");
    let project_text = path_text(&project)?;
    pot_json(&["init".to_owned(), project_text.clone()])?;

    let mut setup_operations = Vec::new();
    let nodes = oracle["nodes"]
        .as_array()
        .ok_or("Blender capture oracle omitted node specs")?;
    for node in nodes {
        setup_operations.push(json!({
            "op":"node.create",
            "id":node["id"],
            "kind":node["kind"],
            "transform":node["transform"],
        }));
        if let Some(bones) = node["bones"].as_array() {
            for bone in bones {
                setup_operations.push(json!({
                    "op":"bone.create",
                    "target":{"id":bone["armature_id"]},
                    "id":bone["id"],
                    "name":bone["name"],
                    "head":bone["head"],
                    "tail":bone["tail"],
                }));
            }
        }
        if let Some(fcurves) = node["fcurves"]
            .as_array()
            .filter(|curves| !curves.is_empty())
        {
            let node_id = node["id"]
                .as_str()
                .ok_or("animated Blender capture node ID is missing")?;
            let action_id = format!("capture_animation_{node_id}");
            setup_operations.push(json!({
                "op":"action.create",
                "id":action_id,
                "name":action_id,
            }));
            setup_operations.push(json!({
                "op":"node.update",
                "target":{"id":node_id},
                "set":{"action":action_id},
            }));
            setup_operations.push(json!({
                "op":"action.update",
                "target":{"id":action_id},
                "set":{"fcurves":fcurves},
            }));
        }
    }
    setup_operations.push(json!({
        "op":"scene.update",
        "target":{"id":"scene_main"},
        "set":{"frame_current":oracle["frame"]},
    }));
    let setup_path = directory.path().join("setup.json");
    fs::write(
        &setup_path,
        serde_json::to_vec(&json!({
            "schema_version":1,
            "base_revision":0,
            "operations":setup_operations,
        }))?,
    )?;
    pot_json(&[
        "apply".to_owned(),
        project_text.clone(),
        "--file".to_owned(),
        path_text(&setup_path)?,
    ])?;
    let setup_doc: SceneDoc = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;

    let mut constraint_operations = Vec::new();
    for constraint in constraint_records {
        let mut create = json!({
            "op":"constraint.create",
            "target":{"id":constraint["owner_id"]},
            "id":constraint["key"],
            "type":constraint["type"],
            "constraint_target":constraint["target_id"],
        });
        if let Some(owner_bone) = constraint["owner_bone"].as_str() {
            create["owner_bone"] = json!(owner_bone);
        }
        if let Some(influence) = constraint["influence"].as_f64() {
            create["influence"] = json!(influence);
        }
        if !constraint["create_params"].is_null() {
            create["params"] = constraint["create_params"].clone();
        }
        constraint_operations.push(create);
        if !constraint["update_params"].is_null() {
            constraint_operations.push(json!({
                "op":"constraint.update",
                "target":{"id":constraint["owner_id"]},
                "id":constraint["key"],
                "set":{"params":constraint["update_params"]},
            }));
        }
    }
    let constraints_path = directory.path().join("constraints.json");
    fs::write(
        &constraints_path,
        serde_json::to_vec(&json!({
            "schema_version":1,
            "base_revision":setup_doc.revision,
            "operations":constraint_operations,
        }))?,
    )?;
    pot_json(&[
        "apply".to_owned(),
        project_text.clone(),
        "--file".to_owned(),
        path_text(&constraints_path)?,
    ])?;
    let doc: SceneDoc = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    assert_eq!(
        doc.scenes[&doc.active_scene].frame_current,
        oracle["frame"].as_f64().ok_or("capture frame is missing")?
    );
    let frame = oracle["frame"]
        .as_i64()
        .ok_or("capture frame is not an integer")?
        .to_string();
    for capture in constraint_records {
        let owner_id = capture["owner_id"]
            .as_str()
            .ok_or("capture owner ID is missing")?;
        let constraint_id = capture["key"]
            .as_str()
            .ok_or("capture constraint ID is missing")?;
        let frame_one_cache = capture["frame_one_cache"]
            .as_object()
            .ok_or("Blender capture oracle omitted frame-one cache")?;
        let frame_two_cache = capture["cache"]
            .as_object()
            .ok_or("Blender capture oracle omitted frame-two cache")?;
        if frame_one_cache == frame_two_cache {
            return Err(format!(
                "{constraint_id} capture fixture does not distinguish frame 1 from frame 2"
            )
            .into());
        }
        let owner = doc
            .nodes
            .get(&potter_core::model::Id::new(owner_id.to_owned())?)
            .ok_or_else(|| format!("captured owner `{owner_id}` is missing"))?;
        let stored = owner
            .constraints
            .iter()
            .find(|item| item.id.as_str() == constraint_id)
            .ok_or_else(|| format!("captured constraint `{constraint_id}` is missing"))?;
        if let Some(owner_bone) = capture["owner_bone"].as_str() {
            assert_eq!(
                stored
                    .owner_bone
                    .as_ref()
                    .map(potter_core::model::Id::as_str),
                Some(owner_bone)
            );
            let influence = capture["influence"]
                .as_f64()
                .ok_or("pose-bone capture influence is missing")?;
            assert!(influence > 0.0 && influence < 1.0);
            assert!((stored.influence - influence).abs() <= f64::EPSILON);
        }
        for (key, expected) in capture["cache"]
            .as_object()
            .ok_or("Blender capture cache is missing")?
        {
            if key == "inverse_matrix" {
                let actual = stored
                    .params
                    .get(key)
                    .ok_or_else(|| format!("{constraint_id} did not persist {key}"))?;
                matrix_parity(actual, expected, &format!("{constraint_id} {key}"))?;
            } else if expected.is_boolean() {
                assert_eq!(stored.params.get(key), Some(expected));
            } else {
                let actual = stored.params.get(key).and_then(Value::as_f64);
                let expected = expected
                    .as_f64()
                    .ok_or("Blender cache value is not numeric")?;
                let actual =
                    actual.ok_or_else(|| format!("{constraint_id} did not persist {key}"))?;
                assert!(
                    (actual - expected).abs() <= MATRIX_TOLERANCE,
                    "{constraint_id} {key}: Potter {actual} != Blender {expected}",
                );
            }
        }
        if capture["type"] == "child_of" {
            assert_eq!(stored.params["set_inverse_pending"], json!(false));
        }
        let inspected = pot_json(&[
            "inspect".to_owned(),
            project_text.clone(),
            "--id".to_owned(),
            owner_id.to_owned(),
            "--frame".to_owned(),
            frame.clone(),
        ])?;
        let item = inspected["result"]["items"]
            .as_array()
            .and_then(|items| items.first())
            .ok_or_else(|| format!("pot inspect omitted {owner_id}"))?;
        let actual_matrix = if let Some(owner_bone) = capture["owner_bone"].as_str() {
            item["evaluated_bones"].get(owner_bone).ok_or_else(|| {
                format!("pot inspect omitted evaluated bone {owner_id}:{owner_bone}")
            })?
        } else {
            &item["transform"]["world"]["matrix"]
        };
        matrix_parity(
            actual_matrix,
            &capture["world_matrix"],
            &format!("{constraint_id} frame {frame}"),
        )?;
    }
    Ok(())
}

#[test]
fn targetless_stretch_to_can_be_renamed_before_target_capture() -> TestResult {
    let directory = tempdir()?;
    let project = directory.path().join("project");
    let project_text = path_text(&project)?;
    pot_json(&["init".to_owned(), project_text.clone()])?;
    let operations_path = directory.path().join("operations.json");
    let operations = json!({
        "schema_version":1,
        "base_revision":0,
        "operations":[
            {"op":"node.create","id":"owner","kind":"empty"},
            {"op":"node.create","id":"target","kind":"empty",
             "transform":{"translation":[0.0,5.0,0.0]}},
            {"op":"constraint.create","target":{"id":"owner"},"id":"stretch",
             "type":"stretch_to"},
            {"op":"constraint.update","target":{"id":"owner"},"id":"stretch",
             "set":{"name":"Renamed Stretch"}},
            {"op":"constraint.update","target":{"id":"owner"},"id":"stretch",
             "set":{"target":"target"}}
        ]
    });
    fs::write(&operations_path, serde_json::to_vec(&operations)?)?;
    pot_json(&[
        "apply".to_owned(),
        project_text,
        "--file".to_owned(),
        path_text(&operations_path)?,
    ])?;

    let doc: SceneDoc = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let owner = doc
        .nodes
        .get(&potter_core::model::Id::new("owner".to_owned())?)
        .ok_or("targetless Stretch To owner is missing")?;
    let constraint = owner
        .constraints
        .iter()
        .find(|constraint| constraint.id.as_str() == "stretch")
        .ok_or("renamed Stretch To constraint is missing")?;
    assert_eq!(constraint.name, "Renamed Stretch");
    assert_eq!(
        constraint
            .target
            .as_ref()
            .map(potter_core::model::Id::as_str),
        Some("target")
    );
    let rest_length = constraint
        .params
        .get("rest_length")
        .and_then(Value::as_f64)
        .ok_or("targeted Stretch To did not capture rest_length")?;
    assert!(
        (rest_length - 5.0).abs() <= MATRIX_TOLERANCE,
        "targeted Stretch To rest_length {rest_length} != Blender distance 5.0"
    );
    Ok(())
}

#[test]
fn unrelated_constraint_evaluation_errors_do_not_block_initial_state_edits() -> TestResult {
    let directory = tempdir()?;
    let project = directory.path().join("project");
    let project_text = path_text(&project)?;
    pot_json(&["init".to_owned(), project_text.clone()])?;
    let operations_path = directory.path().join("operations.json");
    let operations = json!({"schema_version":1,"base_revision":0,"operations":[
        {"op":"node.create","id":"a_unrelated","kind":"empty"},
        {"op":"node.create","id":"clamp_space","kind":"empty"},
        {"op":"node.create","id":"z_owner","kind":"empty"},
        {"op":"node.create","id":"target","kind":"empty",
         "transform":{"translation":[0.0,4.0,0.0]}},
        {"op":"curve.create","id":"path","dimensions":"three_d","splines":[
            {"type":"poly","points":[
                {"co":[0.0,0.0,0.0]},
                {"co":[1.0,0.5,0.0]},
                {"co":[2.0,0.0,0.0]}
            ]}
        ]},
        {"op":"constraint.create","target":{"id":"a_unrelated"},"id":"custom_clamp",
         "type":"clamp_to","constraint_target":"path",
         "params":{"owner_space":"CUSTOM","target_space":"WORLD",
                   "space_object":"clamp_space"}},
        {"op":"constraint.create","target":{"id":"z_owner"},"id":"stretch",
         "type":"stretch_to","constraint_target":"target"},
        {"op":"constraint.update","target":{"id":"z_owner"},"id":"stretch",
         "set":{"name":"Updated Stretch"}},
        {"op":"constraint.create","target":{"id":"z_owner"},"id":"child",
         "type":"child_of","constraint_target":"target"},
        {"op":"constraint.update","target":{"id":"z_owner"},"id":"child",
         "set":{"name":"Updated Child Of"}},
        {"op":"constraint.create","target":{"id":"z_owner"},"id":"distance",
         "type":"limit_distance","constraint_target":"target"},
        {"op":"constraint.update","target":{"id":"z_owner"},"id":"distance",
         "set":{"name":"Updated Distance"}}
    ]});
    fs::write(&operations_path, serde_json::to_vec(&operations)?)?;
    pot_json(&[
        "apply".to_owned(),
        project_text,
        "--file".to_owned(),
        path_text(&operations_path)?,
    ])?;

    let doc: SceneDoc = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let owner = &doc.nodes[&potter_core::model::Id::new("z_owner".to_owned())?];
    for (id, name) in [
        ("stretch", "Updated Stretch"),
        ("child", "Updated Child Of"),
        ("distance", "Updated Distance"),
    ] {
        let constraint = owner
            .constraints
            .iter()
            .find(|constraint| constraint.id.as_str() == id)
            .ok_or_else(|| format!("constraint {id} was not committed"))?;
        assert_eq!(constraint.name, name);
    }
    Ok(())
}

#[test]
fn degenerate_limit_distance_recaptures_while_stretch_to_remains_refused() -> TestResult {
    for (constraint_type, expected_success, feature_id) in [
        (
            "limit_distance",
            true,
            "constraint.limit_distance.initial_distance",
        ),
        ("stretch_to", false, "constraint.stretch_to.initial_length"),
    ] {
        let directory = tempdir()?;
        let project = directory.path().join("project");
        let project_text = path_text(&project)?;
        pot_json(&["init".to_owned(), project_text.clone()])?;
        let operations_path = directory.path().join("operations.json");
        let operations = json!({"schema_version":1,"base_revision":0,"operations":[
            {"op":"node.create","id":"owner","kind":"empty"},
            {"op":"node.create","id":"target","kind":"empty"},
            {"op":"constraint.create","target":{"id":"owner"},"id":"degenerate",
             "type":constraint_type,"constraint_target":"target"}
        ]});
        fs::write(&operations_path, serde_json::to_vec(&operations)?)?;
        let output = pot_output(&[
            "apply".to_owned(),
            project_text,
            "--file".to_owned(),
            path_text(&operations_path)?,
        ])?;
        assert_eq!(
            output.status.success(),
            expected_success,
            "{constraint_type}: {}",
            String::from_utf8_lossy(&output.stdout)
        );
        if expected_success {
            let document: SceneDoc =
                serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
            let owner = document
                .nodes
                .get(&potter_core::model::Id::new("owner".to_owned())?)
                .ok_or("degenerate limit-distance owner is missing")?;
            let constraint = owner
                .constraints
                .iter()
                .find(|constraint| constraint.id.as_str() == "degenerate")
                .ok_or("degenerate limit-distance constraint was not committed")?;
            assert_eq!(constraint.params.get("distance"), Some(&json!(0.0)));
        } else {
            let response: Value = serde_json::from_slice(&output.stdout)?;
            assert_eq!(response["error"]["code"], "UNSUPPORTED_FEATURE");
            assert_eq!(response["error"]["details"]["feature_id"], feature_id);
        }
    }
    Ok(())
}

#[test]
fn custom_curve_constraint_spaces_are_refused_with_precise_features() -> TestResult {
    for (constraint_type, feature_id, owner_space, target_space) in [
        (
            "clamp_to",
            "constraint.clamp_to.custom_spaces",
            "CUSTOM",
            "WORLD",
        ),
        (
            "clamp_to",
            "constraint.clamp_to.custom_spaces",
            "WORLD",
            "CUSTOM",
        ),
        (
            "follow_path",
            "constraint.follow_path.custom_spaces",
            "CUSTOM",
            "WORLD",
        ),
        (
            "follow_path",
            "constraint.follow_path.custom_spaces",
            "WORLD",
            "CUSTOM",
        ),
    ] {
        let directory = tempdir()?;
        let project = directory.path().join("project");
        let project_text = path_text(&project)?;
        pot_json(&["init".to_owned(), project_text.clone()])?;

        let operations_path = directory.path().join("operations.json");
        let operations = json!({"schema_version":1,"base_revision":0,"operations":[
            {"op":"node.create","id":"owner","kind":"empty"},
            {"op":"node.create","id":"space","kind":"empty"},
            {"op":"curve.create","id":"path","dimensions":"three_d","splines":[
                {"type":"poly","points":[
                    {"co":[0.0,0.0,0.0]},
                    {"co":[1.0,0.5,0.0]},
                    {"co":[2.0,0.0,0.0]}
                ]}
            ]},
            {"op":"constraint.create","target":{"id":"owner"},"id":"custom_space",
             "type":constraint_type,"constraint_target":"path",
             "params":{"owner_space":owner_space,"target_space":target_space,
                       "space_object":"space"}}
        ]});
        fs::write(&operations_path, serde_json::to_vec(&operations)?)?;
        pot_json(&[
            "apply".to_owned(),
            project_text.clone(),
            "--file".to_owned(),
            path_text(&operations_path)?,
        ])?;

        let output = pot_output(&[
            "inspect".to_owned(),
            project_text,
            "--id".to_owned(),
            "owner".to_owned(),
        ])?;
        if output.status.success() {
            return Err(format!(
                "{constraint_type} with {owner_space}/{target_space} spaces unexpectedly evaluated"
            )
            .into());
        }
        let response: Value = serde_json::from_slice(&output.stdout)?;
        assert_eq!(response["error"]["code"], "UNSUPPORTED_FEATURE");
        assert_eq!(response["error"]["details"]["feature_id"], feature_id);
    }
    Ok(())
}

#[test]
fn imported_zero_limit_distance_persists_first_nonzero_capture() -> TestResult {
    let _guard = BLENDER_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(blender) = blender_checked::blender_executable() else {
        return Err("Blender is required for zero Limit Distance import parity".into());
    };
    let fixture_dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/standard_constraints");
    let directory = tempdir()?;
    let root_text = path_text(directory.path())?;
    let generator = path_text(&fixture_dir.join("generate_zero_limit_distance.py"))?;
    let mut command = Command::new(&blender);
    command
        .args(["--background", "--factory-startup", "--python"])
        .arg(generator)
        .arg("--")
        .arg(root_text);
    let generated = process::run_guarded(command)?;
    assert!(
        generated.status.success(),
        "Blender fixture generation failed: {}",
        String::from_utf8_lossy(&generated.stdout)
    );
    let source = path_text(&directory.path().join("zero_limit_distance.blend"))?;
    let blender_text = path_text(&blender)?;
    let project = directory.path().join("project");
    let project_text = path_text(&project)?;
    pot_json(&["init".to_owned(), project_text.clone()])?;
    let imported = pot_json(&[
        "import".to_owned(),
        project_text.clone(),
        "--file".to_owned(),
        source,
        "--format".to_owned(),
        "blend".to_owned(),
        "--mode".to_owned(),
        "replace".to_owned(),
        "--base-revision".to_owned(),
        "0".to_owned(),
        "--blender".to_owned(),
        blender_text,
    ])?;
    let losses = imported["result"]["losses"]
        .as_array()
        .ok_or("Blender import omitted its loss list")?;
    assert!(
        !losses
            .iter()
            .any(|loss| { loss["feature_id"] == "constraint.limit_distance.initial_distance" }),
        "zero Limit Distance import reported a loss: {imported}"
    );
    let scene_path = project.join("scene.json");
    let document: SceneDoc = serde_json::from_slice(&fs::read(&scene_path)?)?;
    let owner_id = document
        .nodes
        .iter()
        .find(|(_, node)| node.name == "ZeroDistanceOwner")
        .map(|(id, _)| id.clone())
        .ok_or("imported Limit Distance owner is missing")?;
    let target_id = document
        .nodes
        .iter()
        .find(|(_, node)| node.name == "ZeroDistanceTarget")
        .map(|(id, _)| id.clone())
        .ok_or("imported Limit Distance target is missing")?;
    assert_eq!(
        document.nodes[&owner_id].constraints[0].params["distance"],
        0.0
    );

    let operations_path = directory.path().join("operations.json");
    fs::write(
        &operations_path,
        serde_json::to_vec(&json!({
            "schema_version":1,
            "base_revision":1,
            "operations":[{
                "op":"node.update",
                "target":{"id":target_id.to_string()},
                "set":{"transform":{"translation":[5.0,0.0,0.0]}}
            }]
        }))?,
    )?;
    pot_json(&[
        "apply".to_owned(),
        project_text.clone(),
        "--file".to_owned(),
        path_text(&operations_path)?,
    ])?;
    let first_move: SceneDoc = serde_json::from_slice(&fs::read(&scene_path)?)?;
    assert_eq!(
        first_move.nodes[&owner_id].constraints[0].params["distance"],
        5.0
    );

    fs::write(
        &operations_path,
        serde_json::to_vec(&json!({
            "schema_version":1,
            "base_revision":2,
            "operations":[{
                "op":"node.update",
                "target":{"id":target_id.to_string()},
                "set":{"transform":{"translation":[10.0,0.0,0.0]}}
            }]
        }))?,
    )?;
    pot_json(&[
        "apply".to_owned(),
        project_text.clone(),
        "--file".to_owned(),
        path_text(&operations_path)?,
    ])?;
    let inspected = pot_json(&[
        "inspect".to_owned(),
        project_text,
        "--id".to_owned(),
        owner_id.to_string(),
    ])?;
    let item = inspected["result"]["items"]
        .as_array()
        .and_then(|items| items.first())
        .ok_or("inspect omitted the Limit Distance owner")?;
    let matrix = item["transform"]["world"]["matrix"]
        .as_array()
        .ok_or("Potter matrix is missing")?;
    let owner_x = matrix
        .get(12)
        .and_then(Value::as_f64)
        .ok_or("Potter matrix translation is missing")?;
    assert!(
        (owner_x - 5.0).abs() <= MATRIX_TOLERANCE,
        "owner did not stay on its captured Limit Distance: x={owner_x}"
    );
    Ok(())
}
