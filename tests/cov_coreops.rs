#![expect(
    clippy::unwrap_used,
    reason = "focused integration tests use deterministic fixtures and fail immediately on setup errors"
)]

use std::{
    fs,
    path::Path,
    process::{Command, Output},
};

use glam::DVec3;
use potter::{
    error::ErrorCode,
    eval::{EvaluationContext, Snapshot},
    geom::Mesh,
    model::{
        Action, DataBlock, Extrapolation, FCurve, Id, Interpolation, Keyframe, Material, Node,
        SceneDoc, World,
    },
    ops, validate,
};
use proptest::prelude::*;
use serde_json::{Value, json};
use tempfile::tempdir;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn response(output: &Output) -> Value {
    serde_json::from_slice(&output.stdout).unwrap()
}

fn init(scene: &Path) {
    let output = pot().arg("init").arg(scene).arg("--json").output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
}

fn apply(scene: &Path, base_revision: u64, operations: &Value) -> Output {
    let directory = tempdir().unwrap();
    let batch = directory.path().join("operations.json");
    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": base_revision,
            "operations": operations
        }))
        .unwrap(),
    )
    .unwrap();
    pot()
        .arg("apply")
        .arg(scene)
        .arg("--file")
        .arg(batch)
        .arg("--json")
        .output()
        .unwrap()
}

fn command_output(command: &str, scene: &Path, arguments: &[&str]) -> Output {
    pot()
        .arg(command)
        .arg(scene)
        .args(arguments)
        .arg("--json")
        .output()
        .unwrap()
}

fn apply_doc(doc: &SceneDoc, operations: &Value) -> potter::ops::ApplyOutcome {
    ops::apply_batch(
        doc,
        &json!({
            "schema_version": 1,
            "base_revision": doc.revision,
            "operations": operations
        }),
    )
    .unwrap()
}
fn assert_issue_at(report: &Value, code: &str, pointer: &str) {
    assert!(
        report["issues"]
            .as_array()
            .unwrap()
            .iter()
            .any(|issue| { issue["code"] == code && issue["pointer"] == pointer }),
        "missing {code} at {pointer}: {}",
        report["issues"]
    );
}

#[test]
fn history_undo_redo_cover_steps_revisions_and_branch_errors() {
    let directory = tempdir().unwrap();
    let scene = directory.path().join("scene");
    init(&scene);

    let initial = command_output("history", &scene, &[]);
    assert!(initial.status.success());
    let initial_value = response(&initial);
    assert!(
        initial_value["result"]["entries"].is_array(),
        "{initial_value}"
    );
    let initial_entries = initial_value["result"]["entries"].as_array().unwrap();
    assert_eq!(initial_entries.len(), 1);
    assert_eq!(initial_entries[0]["kind"], "init");

    assert!(
        apply(
            &scene,
            0,
            &json!([{"op":"node.create","id":"first","kind":"empty"}])
        )
        .status
        .success()
    );
    assert!(
        apply(
            &scene,
            1,
            &json!([{"op":"node.create","id":"second","kind":"empty"}])
        )
        .status
        .success()
    );

    let history = command_output("history", &scene, &[]);
    assert!(history.status.success());
    let history_value = response(&history);
    let entries = history_value["result"]["entries"].as_array().unwrap();
    assert_eq!(entries.len(), 3);
    let revision_two = entries.iter().find(|entry| entry["revision"] == 2).unwrap();
    assert_eq!(revision_two["kind"], "apply");
    assert_eq!(revision_two["operations"][0]["op"], "node.create");
    assert!(
        !revision_two["scene_hash"].as_str().unwrap().is_empty(),
        "scene hash should not be empty"
    );

    let undo = command_output("undo", &scene, &["--base-revision", "2"]);
    assert!(undo.status.success());
    let undo_value = response(&undo);
    assert_eq!(undo_value["scene"]["revision"], 3);
    assert_eq!(undo_value["result"]["committed"], true);
    assert_eq!(undo_value["result"]["changed"], true);
    assert_eq!(undo_value["result"]["base_revision"], 2);
    assert_eq!(undo_value["result"]["candidate_revision"], 3);
    assert!(
        entries
            .iter()
            .any(|entry| entry["id"] == undo_value["result"]["history_id"])
    );
    assert_eq!(
        response(&command_output("inspect", &scene, &[]))["result"]["summary"]["counts"]["nodes"],
        1
    );

    let redo = command_output("redo", &scene, &["--base-revision", "3"]);
    assert!(redo.status.success());
    assert_eq!(response(&redo)["scene"]["revision"], 4);
    let undo_two = command_output("undo", &scene, &["--base-revision", "4", "--steps", "2"]);
    assert!(undo_two.status.success());
    assert_eq!(response(&undo_two)["scene"]["revision"], 5);
    assert_eq!(
        response(&command_output("inspect", &scene, &[]))["result"]["summary"]["counts"]["nodes"],
        0
    );
    let redo_two = command_output("redo", &scene, &["--base-revision", "5", "--steps", "2"]);
    assert!(redo_two.status.success());
    assert_eq!(response(&redo_two)["scene"]["revision"], 6);
    assert_eq!(
        response(&command_output("inspect", &scene, &[]))["result"]["summary"]["counts"]["nodes"],
        2
    );

    let zero_steps = command_output("undo", &scene, &["--base-revision", "6", "--steps", "0"]);
    assert_eq!(zero_steps.status.code(), Some(2));
    assert_eq!(response(&zero_steps)["error"]["code"], "INVALID_ARGUMENT");
    let stale = command_output("undo", &scene, &["--base-revision", "5"]);
    assert_eq!(stale.status.code(), Some(5));
    assert_eq!(response(&stale)["error"]["code"], "REVISION_CONFLICT");
    let too_far = command_output("undo", &scene, &["--base-revision", "6", "--steps", "99"]);
    assert_eq!(too_far.status.code(), Some(3));
    assert_eq!(response(&too_far)["error"]["code"], "TARGET_NOT_FOUND");

    assert!(
        command_output("undo", &scene, &["--base-revision", "6"])
            .status
            .success()
    );
    assert!(
        apply(
            &scene,
            7,
            &json!([{"op":"node.create","id":"branch","kind":"empty"}])
        )
        .status
        .success()
    );
    let no_redo = command_output("redo", &scene, &["--base-revision", "8"]);
    assert_eq!(no_redo.status.code(), Some(3));
    assert_eq!(response(&no_redo)["error"]["code"], "TARGET_NOT_FOUND");
}

#[test]
fn inspect_reports_multiple_tags_data_blocks_and_evaluation_context() {
    let directory = tempdir().unwrap();
    let scene = directory.path().join("scene");
    init(&scene);
    let created = apply(
        &scene,
        0,
        &json!([
            {"op":"node.create","id":"body","kind":"box","params":{},"tags":["pick","shared"]},
            {"op":"node.create","id":"body_two","kind":"box","params":{},"tags":["pick"]},
            {"op":"scene.create","id":"scene_alt","root_collection":"collection_root","view_layer":"view_alt","active":true}
        ]),
    );
    assert!(
        created.status.success(),
        "{}",
        String::from_utf8_lossy(&created.stdout)
    );

    let matches = command_output("inspect", &scene, &["--tag", "pick"]);
    assert!(matches.status.success());
    let matches_value = response(&matches);
    let items = matches_value["result"]["items"].as_array().unwrap();
    assert_eq!(items.len(), 2);
    assert_eq!(items[0]["id"], "body");
    assert_eq!(items[1]["id"], "body_two");

    let data = command_output("inspect", &scene, &["--id", "body_mesh"]);
    assert!(data.status.success());
    let item = response(&data)["result"]["items"][0].clone();
    assert_eq!(item["type"], "data_block");
    assert_eq!(item["geometry"]["vertices"], 8);
    assert_eq!(item["users"], json!(["body"]));

    let contextual = command_output(
        "inspect",
        &scene,
        &[
            "--scene-id",
            "scene_alt",
            "--view-layer",
            "view_alt",
            "--frame",
            "12.5",
        ],
    );
    assert!(contextual.status.success());
    assert_eq!(
        response(&contextual)["result"]["evaluation"]["scene_id"],
        "scene_alt"
    );
    assert_eq!(
        response(&contextual)["result"]["evaluation"]["view_layer"],
        "view_alt"
    );
    assert_eq!(response(&contextual)["result"]["evaluation"]["frame"], 12.5);

    let unknown_tag = command_output("inspect", &scene, &["--tag", "absent"]);
    assert_eq!(unknown_tag.status.code(), Some(3));
    assert_eq!(response(&unknown_tag)["error"]["code"], "TARGET_NOT_FOUND");
    let invalid_id = command_output("inspect", &scene, &["--id", "Not_An_Id"]);
    assert_eq!(invalid_id.status.code(), Some(2));
    assert_eq!(response(&invalid_id)["error"]["code"], "INVALID_ARGUMENT");
    let missing_layer = command_output(
        "inspect",
        &scene,
        &["--scene-id", "scene_alt", "--view-layer", "missing_layer"],
    );
    assert_eq!(missing_layer.status.code(), Some(3));
    assert_eq!(
        response(&missing_layer)["error"]["code"],
        "TARGET_NOT_FOUND"
    );
    let missing_scene = command_output("inspect", &scene, &["--scene-id", "missing_scene"]);
    assert_eq!(missing_scene.status.code(), Some(3));
    assert_eq!(
        response(&missing_scene)["error"]["code"],
        "TARGET_NOT_FOUND"
    );
    let nonfinite_frame = command_output("inspect", &scene, &["--frame", "NaN"]);
    assert_eq!(nonfinite_frame.status.code(), Some(2));
    assert_eq!(
        response(&nonfinite_frame)["error"]["code"],
        "INVALID_ARGUMENT"
    );
}

#[test]
fn evaluator_rejects_missing_contexts_and_nonfinite_frames() {
    let doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
    let missing_scene = EvaluationContext {
        scene_id: Some(Id::new("missing_scene").unwrap()),
        ..EvaluationContext::default()
    };
    assert_eq!(
        Snapshot::evaluate(&doc, &missing_scene).unwrap_err().code,
        ErrorCode::TargetNotFound
    );
    let missing_layer = EvaluationContext {
        view_layer: Some(Id::new("missing_layer").unwrap()),
        ..EvaluationContext::default()
    };
    assert_eq!(
        Snapshot::evaluate(&doc, &missing_layer).unwrap_err().code,
        ErrorCode::TargetNotFound
    );
    let nonfinite = EvaluationContext {
        frame: Some(f64::INFINITY),
        ..EvaluationContext::default()
    };
    assert_eq!(
        Snapshot::evaluate(&doc, &nonfinite).unwrap_err().code,
        ErrorCode::InvalidArgument
    );
}

proptest! {
    #[test]
    fn evaluated_child_translation_composes_parent_scale(
        px in -10.0f64..10.0,
        py in -10.0f64..10.0,
        pz in -10.0f64..10.0,
        sx in 0.25f64..4.0,
        sy in 0.25f64..4.0,
        sz in 0.25f64..4.0,
        cx in -10.0f64..10.0,
        cy in -10.0f64..10.0,
        cz in -10.0f64..10.0,
    ) {
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        let parent_id = Id::new("parent").unwrap();
        let child_id = Id::new("child").unwrap();
        let mut parent = Node::default();
        parent.transform.translation = [px, py, pz];
        parent.transform.scale = [sx, sy, sz];
        let mut child = Node {
            parent: Some(parent_id.clone()),
            ..Node::default()
        };
        child.transform.translation = [cx, cy, cz];
        doc.nodes.insert(parent_id, parent);
        doc.nodes.insert(child_id.clone(), child);

        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default()).unwrap();
        let matrix = snapshot.nodes[&child_id].world_matrix;
        prop_assert!((matrix[12] - (px + sx * cx)).abs() < 1.0e-9);
        prop_assert!((matrix[13] - (py + sy * cy)).abs() < 1.0e-9);
        prop_assert!((matrix[14] - (pz + sz * cz)).abs() < 1.0e-9);
    }
}

#[test]
fn node_duplicate_delete_reparent_and_cascade_policies_preserve_state() {
    let original = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
    let created = apply_doc(
        &original,
        &json!([
            {"op":"node.create","id":"assembly","kind":"group"},
            {"op":"node.create","id":"body","kind":"box","params":{},"parent":"assembly"},
            {"op":"node.duplicate","target":{"id":"assembly"},"id":"linked_copy","mode":"linked","recursive":true},
            {"op":"node.duplicate","target":{"id":"body"},"id":"independent_copy","mode":"independent"}
        ]),
    );
    let linked_child = Id::new("linked_copy_body").unwrap();
    let body_id = Id::new("body").unwrap();
    assert_eq!(
        created.doc.nodes[&linked_child].parent,
        Some(Id::new("linked_copy").unwrap())
    );
    assert_eq!(
        created.doc.nodes[&linked_child].data,
        created.doc.nodes[&body_id].data
    );
    assert_ne!(
        created.doc.nodes[&Id::new("independent_copy").unwrap()].data,
        created.doc.nodes[&body_id].data
    );
    assert_eq!(created.id_mappings["nodes"]["assembly"], "linked_copy");

    let no_policy = ops::apply_batch(
        &original,
        &json!({"schema_version":1,"base_revision":0,"operations":[
            {"op":"node.create","id":"parent","kind":"empty"},
            {"op":"node.create","id":"child","kind":"empty","parent":"parent"},
            {"op":"node.delete","target":{"id":"parent"}}
        ]}),
    )
    .unwrap_err();
    assert_eq!(no_policy.code, ErrorCode::InvalidOperation);

    let hierarchy = apply_doc(
        &original,
        &json!([
            {"op":"node.create","id":"grandparent","kind":"empty","transform":{"translation":[10,0,0]}},
            {"op":"node.create","id":"parent","kind":"empty","parent":"grandparent","transform":{"translation":[2,0,0]}},
            {"op":"node.create","id":"child","kind":"empty","parent":"parent","transform":{"translation":[3,0,0]}}
        ]),
    );
    let child_id = Id::new("child").unwrap();
    let before = Snapshot::evaluate(&hierarchy.doc, &EvaluationContext::default())
        .unwrap()
        .nodes[&child_id]
        .world_matrix[12];
    let to_parent = apply_doc(
        &hierarchy.doc,
        &json!([{"op":"node.delete","target":{"id":"parent"},"reparent":"to_parent"}]),
    );
    assert_eq!(
        to_parent.doc.nodes[&child_id].parent,
        Some(Id::new("grandparent").unwrap())
    );
    let parent_world = Snapshot::evaluate(&to_parent.doc, &EvaluationContext::default())
        .unwrap()
        .nodes[&child_id]
        .world_matrix[12];
    assert!((parent_world - before).abs() < 1.0e-9);
    let to_root = apply_doc(
        &to_parent.doc,
        &json!([{"op":"node.delete","target":{"id":"grandparent"},"reparent":"to_root"}]),
    );
    assert_eq!(to_root.doc.nodes[&child_id].parent, None);
    assert_eq!(to_root.doc.nodes[&child_id].parent_inverse, None);
    assert!(
        to_root.doc.nodes[&child_id]
            .transform
            .translation
            .into_iter()
            .zip([15.0, 0.0, 0.0])
            .all(|(actual, expected)| (actual - expected).abs() < 1.0e-9)
    );
    let root_world = Snapshot::evaluate(&to_root.doc, &EvaluationContext::default())
        .unwrap()
        .nodes[&child_id]
        .world_matrix[12];
    assert!((root_world - before).abs() < 1.0e-9);

    let shear_hierarchy = apply_doc(
        &original,
        &json!([
            {"op":"node.create","id":"shear_parent","kind":"empty","transform":{"scale":[2,1,1]}},
            {"op":"node.create","id":"shear_child","kind":"empty","parent":"shear_parent","transform":{"translation":[1,0,0],"rotation":[0,0,0.382_683_432_365_089_8,0.923_879_532_511_286_7]}}
        ]),
    );
    let shear_child_id = Id::new("shear_child").unwrap();
    let shear_before = Snapshot::evaluate(&shear_hierarchy.doc, &EvaluationContext::default())
        .unwrap()
        .nodes[&shear_child_id]
        .world_matrix;
    let shear_root = apply_doc(
        &shear_hierarchy.doc,
        &json!([{"op":"node.delete","target":{"id":"shear_parent"},"reparent":"to_root"}]),
    );
    assert_eq!(shear_root.doc.nodes[&shear_child_id].parent, None);
    assert!(
        shear_root.doc.nodes[&shear_child_id]
            .parent_inverse
            .is_some()
    );
    let shear_after = Snapshot::evaluate(&shear_root.doc, &EvaluationContext::default())
        .unwrap()
        .nodes[&shear_child_id]
        .world_matrix;
    for (before, after) in shear_before.into_iter().zip(shear_after) {
        assert!((before - after).abs() < 1.0e-9 * before.abs().max(1.0));
    }

    let local_only = apply_doc(
        &hierarchy.doc,
        &json!([{"op":"node.delete","target":{"id":"parent"},"reparent":"to_root","keep_world":false}]),
    );
    assert_eq!(local_only.doc.nodes[&child_id].parent, None);
    assert_eq!(local_only.doc.nodes[&child_id].parent_inverse, None);
    for operation in [
        json!({"op":"node.delete","target":{"id":"parent"},"recursive":true,"reparent":"to_root"}),
        json!({"op":"node.delete","target":{"id":"parent"},"keep_world":false}),
        json!({"op":"node.delete","target":{"id":"parent"},"reparent":"somewhere"}),
    ] {
        let error = ops::apply_batch(
            &hierarchy.doc,
            &json!({"schema_version":1,"base_revision":hierarchy.doc.revision,"operations":[operation]}),
        )
        .unwrap_err();
        assert_eq!(error.code, ErrorCode::InvalidOperation);
    }

    let deleted_linked = apply_doc(
        &created.doc,
        &json!([{"op":"node.delete","target":{"id":"linked_copy"},"recursive":true,"cascade_data":true}]),
    );
    assert!(
        !deleted_linked
            .doc
            .nodes
            .contains_key(&Id::new("linked_copy").unwrap())
    );
    assert!(!deleted_linked.doc.nodes.contains_key(&linked_child));
    let shared_data = deleted_linked.doc.nodes[&body_id].data.as_ref().unwrap();
    assert!(deleted_linked.doc.data_blocks.contains_key(shared_data));
    let independent_data = deleted_linked.doc.nodes[&Id::new("independent_copy").unwrap()]
        .data
        .as_ref()
        .unwrap()
        .clone();
    let cascaded = apply_doc(
        &deleted_linked.doc,
        &json!([{"op":"node.delete","target":{"id":"independent_copy"},"cascade_data":true}]),
    );
    assert!(!cascaded.doc.data_blocks.contains_key(&independent_data));

    let camera = apply_doc(
        &created.doc,
        &json!([
            {"op":"node.create","id":"camera_node","kind":"camera"},
            {"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"camera_node"}}
        ]),
    );
    let camera_error = ops::apply_batch(
        &camera.doc,
        &json!({"schema_version":1,"base_revision":camera.doc.revision,"operations":[{"op":"node.delete","target":{"id":"camera_node"}}]}),
    )
    .unwrap_err();
    assert_eq!(camera_error.code, ErrorCode::InvalidOperation);
    let unlinked_camera = apply_doc(
        &camera.doc,
        &json!([{"op":"node.delete","target":{"id":"camera_node"},"unlink":true,"cascade_data":true}]),
    );
    assert_eq!(
        unlinked_camera.doc.scenes[&Id::new("scene_main").unwrap()].camera,
        None
    );

    let collection = apply_doc(
        &created.doc,
        &json!([
            {"op":"node.create","id":"held","kind":"empty"},
            {"op":"collection.create","id":"child_collection"},
            {"op":"collection.link","collection":"child_collection","object":"held"}
        ]),
    );
    let blocked_collection = ops::apply_batch(
        &collection.doc,
        &json!({"schema_version":1,"base_revision":collection.doc.revision,"operations":[{"op":"collection.delete","target":{"id":"child_collection"}}]}),
    )
    .unwrap_err();
    assert_eq!(blocked_collection.code, ErrorCode::InvalidOperation);
    let root_delete = ops::apply_batch(
        &collection.doc,
        &json!({"schema_version":1,"base_revision":collection.doc.revision,"operations":[{"op":"collection.delete","target":{"id":"collection_root"},"unlink":true}]}),
    )
    .unwrap_err();
    assert_eq!(root_delete.code, ErrorCode::InvalidOperation);
    let unlinked = apply_doc(
        &collection.doc,
        &json!([{"op":"collection.delete","target":{"id":"child_collection"},"unlink":true}]),
    );
    assert!(
        !unlinked
            .doc
            .collections
            .contains_key(&Id::new("child_collection").unwrap())
    );
    assert!(unlinked.doc.nodes.contains_key(&Id::new("held").unwrap()));
}

#[test]
fn validation_reports_all_check_statuses_issue_severities_and_format_losses() {
    let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
    let active_scene = doc.active_scene.clone();
    let missing = Id::new("missing_ref").unwrap();
    let root_id = doc.scenes[&active_scene].root_collection.clone();
    let scene = doc.scenes.get_mut(&active_scene).unwrap();
    scene.root_collection = missing.clone();
    scene.camera = Some(missing.clone());
    scene.world = Some(missing.clone());
    scene.unit.scale_length = 0.0;
    scene.frame_start = 5;
    scene.frame_end = 1;
    scene.fps = 0;
    scene
        .view_layers
        .get_mut(&Id::new("view_main").unwrap())
        .unwrap()
        .excluded_collections
        .push(missing.clone());

    let root = doc.collections.get_mut(&root_id).unwrap();
    root.children
        .extend([missing.clone(), missing.clone(), root_id.clone()]);
    root.objects.push(missing.clone());
    let mut parent = Node {
        kind: "mesh".to_owned(),
        parent: Some(Id::new("child").unwrap()),
        data: Some(Id::new("camera_data").unwrap()),
        materials: vec![missing.clone()],
        action: Some(missing.clone()),
        tags: vec!["Bad_Tag".to_owned()],
        ..Node::default()
    };
    parent.transform.translation[0] = f64::NAN;
    doc.nodes.insert(Id::new("parent").unwrap(), parent);
    doc.nodes.insert(
        Id::new("child").unwrap(),
        Node {
            parent: Some(Id::new("parent").unwrap()),
            ..Node::default()
        },
    );
    doc.nodes.insert(
        Id::new("orphan").unwrap(),
        Node {
            parent: Some(missing.clone()),
            ..Node::default()
        },
    );
    doc.nodes.insert(
        Id::new("missing_data").unwrap(),
        Node {
            data: Some(missing.clone()),
            ..Node::default()
        },
    );
    doc.data_blocks.insert(
        Id::new("camera_data").unwrap(),
        DataBlock {
            data_type: "camera".to_owned(),
            ..DataBlock::default()
        },
    );
    doc.data_blocks.insert(
        Id::new("unused_data").unwrap(),
        DataBlock {
            data_type: "mesh".to_owned(),
            mesh: None,
            ..DataBlock::default()
        },
    );
    doc.materials.insert(
        Id::new("bad_material").unwrap(),
        Material {
            metallic: f64::NAN,
            ..Material::default()
        },
    );
    doc.worlds.insert(
        Id::new("bad_world").unwrap(),
        World {
            strength: f64::INFINITY,
            ..World::default()
        },
    );
    doc.actions.insert(
        Id::new("bad_action").unwrap(),
        Action {
            name: "Invalid curve".to_owned(),
            fcurves: vec![FCurve {
                path: "location".to_owned(),
                index: 0,
                extrapolation: Extrapolation::Constant,
                keyframes: vec![Keyframe {
                    frame: f64::NAN,
                    value: 0.0,
                    interpolation: Interpolation::Linear,
                    ..Keyframe::default()
                }],
            }],
            slots: Vec::new(),
        },
    );

    let open_mesh =
        Mesh::from_positions_and_faces(vec![DVec3::ZERO, DVec3::X, DVec3::Y], vec![vec![0, 1, 2]])
            .unwrap();
    doc.data_blocks.insert(
        Id::new("open_mesh").unwrap(),
        DataBlock {
            data_type: "mesh".to_owned(),
            mesh: Some(open_mesh.clone()),
            ..DataBlock::default()
        },
    );
    let zero_area_mesh = Mesh::from_positions_and_faces(
        vec![DVec3::ZERO, DVec3::X, DVec3::X * 2.0],
        vec![vec![0, 1, 2]],
    )
    .unwrap();
    doc.data_blocks.insert(
        Id::new("zero_area_mesh").unwrap(),
        DataBlock {
            data_type: "mesh".to_owned(),
            mesh: Some(zero_area_mesh),
            ..DataBlock::default()
        },
    );
    let mut invalid_mesh = open_mesh;
    invalid_mesh.faces[0].vertices[0] = 999;
    doc.data_blocks.insert(
        Id::new("invalid_mesh").unwrap(),
        DataBlock {
            data_type: "mesh".to_owned(),
            mesh: Some(invalid_mesh),
            ..DataBlock::default()
        },
    );
    let winding_mesh = Mesh::from_positions_and_faces(
        vec![DVec3::ZERO, DVec3::X, DVec3::Y, DVec3::new(0.0, -1.0, 0.0)],
        vec![vec![0, 1, 2], vec![0, 1, 3]],
    )
    .unwrap();
    doc.data_blocks.insert(
        Id::new("winding_mesh").unwrap(),
        DataBlock {
            data_type: "mesh".to_owned(),
            mesh: Some(winding_mesh),
            ..DataBlock::default()
        },
    );
    let non_manifold_mesh = Mesh::from_positions_and_faces(
        vec![
            DVec3::ZERO,
            DVec3::X,
            DVec3::Y,
            DVec3::new(0.0, -1.0, 0.0),
            DVec3::new(0.0, 0.0, 1.0),
        ],
        vec![vec![0, 1, 2], vec![0, 1, 3], vec![0, 1, 4]],
    )
    .unwrap();
    doc.data_blocks.insert(
        Id::new("non_manifold_mesh").unwrap(),
        DataBlock {
            data_type: "mesh".to_owned(),
            mesh: Some(non_manifold_mesh),
            ..DataBlock::default()
        },
    );

    let singular_doc = {
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        let mut flat_node = Node::default();
        flat_node.transform.scale[0] = 0.0;
        doc.nodes.insert(Id::new("flat").unwrap(), flat_node);
        doc
    };
    let singular = validate::validate(&singular_doc, None).unwrap();
    assert!(singular.has_warnings);
    assert!(
        singular.value["issues"]
            .as_array()
            .unwrap()
            .iter()
            .any(|issue| issue["code"] == "SINGULAR_TRANSFORM")
    );

    let report = validate::validate(&doc, None).unwrap();
    for (code, pointer) in [
        ("MISSING_REFERENCE", "/scenes/scene_main/root_collection"),
        ("MISSING_REFERENCE", "/scenes/scene_main/camera"),
        ("MISSING_REFERENCE", "/scenes/scene_main/world"),
        ("INVALID_UNIT", "/scenes/scene_main/unit/scale_length"),
        (
            "MISSING_REFERENCE",
            "/scenes/scene_main/view_layers/view_main/excluded_collections",
        ),
        ("MISSING_REFERENCE", "/collections/collection_root/children"),
        (
            "DUPLICATE_REFERENCE",
            "/collections/collection_root/children",
        ),
        ("MISSING_REFERENCE", "/collections/collection_root/objects"),
        ("MISSING_REFERENCE", "/nodes/orphan/parent"),
        ("MISSING_REFERENCE", "/nodes/missing_data/data"),
        ("MISSING_REFERENCE", "/nodes/parent/materials"),
        ("MISSING_REFERENCE", "/nodes/parent/action"),
        ("INVALID_ID", "/nodes/parent/tags/0"),
        ("REFERENCE_TYPE_MISMATCH", "/nodes/parent/data"),
        ("NON_FINITE", "/nodes/parent/transform"),
        ("INVALID_MESH", "/data_blocks/invalid_mesh/mesh"),
    ] {
        assert_issue_at(&report.value, code, pointer);
    }
    assert!(report.has_errors);
    assert!(report.has_warnings);
    let issues = report.value["issues"].as_array().unwrap();
    for code in [
        "MISSING_REFERENCE",
        "DUPLICATE_REFERENCE",
        "PARENT_CYCLE",
        "COLLECTION_CYCLE",
        "INVALID_UNIT",
        "INVALID_SCENE_TIMING",
        "REFERENCE_TYPE_MISMATCH",
        "NON_FINITE",
        "INVALID_MESH",
        "INVALID_ID",
        "UNUSED_DATA_BLOCK",
        "OPEN_BOUNDARY",
        "ZERO_AREA_FACE",
        "WINDING_INCONSISTENT",
        "NON_MANIFOLD_EDGE",
    ] {
        assert!(
            issues.iter().any(|issue| issue["code"] == code),
            "missing issue {code}"
        );
    }
    assert!(issues.iter().any(|issue| issue["severity"] == "error"));
    assert!(issues.iter().any(|issue| issue["severity"] == "warning"));
    assert!(
        issues
            .iter()
            .any(|issue| issue["severity"] == "information")
    );
    assert_eq!(report.value["valid"], false);
    assert!(
        report.value["summary"]["by_severity"]["information"]
            .as_u64()
            .unwrap()
            > 0
    );

    let checks = report.value["checks"].as_array().unwrap();
    for name in [
        "schema",
        "id",
        "reference",
        "parent_cycle",
        "collection_cycle",
        "mesh_indices",
        "non_finite",
        "mesh_manifold",
        "mesh_winding",
        "mesh_area",
        "singular_transform",
        "unused_data_blocks",
        "rigid_body_simulation",
    ] {
        assert!(
            checks
                .iter()
                .any(|check| check["check"] == name && check["status"] == "supported")
        );
    }
    for name in [
        "asset_hash",
        "library_dependencies",
        "rig_graph",
        "node_graph",
        "driver_constraint",
        "animation",
        "simulation_cache",
        "shader_compositor_sequencer",
        "feature_compatibility",
    ] {
        assert!(
            checks
                .iter()
                .any(|check| check["check"] == name && check["status"] == "not_supported")
        );
    }

    let mut loss_doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
    loss_doc.nodes.insert(
        Id::new("group_node").unwrap(),
        Node {
            kind: "group".to_owned(),
            ..Node::default()
        },
    );
    let loss_report = validate::validate(&loss_doc, Some("obj")).unwrap();
    assert_eq!(loss_report.value["valid"], false);
    assert!(
        loss_report.value["losses"]
            .as_array()
            .unwrap()
            .iter()
            .any(|loss| loss["feature_id"] == "node.group")
    );
    assert!(
        loss_report.value["checks"]
            .as_array()
            .unwrap()
            .iter()
            .any(|check| check["check"] == "format_loss" && check["status"] == "not_supported")
    );
    let unknown = validate::validate(&loss_doc, Some("not_a_format")).unwrap();
    assert!(
        unknown.value["issues"]
            .as_array()
            .unwrap()
            .iter()
            .any(|issue| issue["code"] == "FORMAT_UNKNOWN")
    );
}

#[test]
fn validation_timing_and_unit_predicates_fail_independently() {
    let mut unit_zero = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
    let scene_id = unit_zero.active_scene.clone();
    unit_zero
        .scenes
        .get_mut(&scene_id)
        .unwrap()
        .unit
        .scale_length = 0.0;
    let report = validate::validate(&unit_zero, None).unwrap();
    assert_issue_at(
        &report.value,
        "INVALID_UNIT",
        "/scenes/scene_main/unit/scale_length",
    );

    let mut unit_nonfinite = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
    let scene_id = unit_nonfinite.active_scene.clone();
    unit_nonfinite
        .scenes
        .get_mut(&scene_id)
        .unwrap()
        .unit
        .scale_length = f64::NAN;
    let report = validate::validate(&unit_nonfinite, None).unwrap();
    assert_issue_at(
        &report.value,
        "INVALID_UNIT",
        "/scenes/scene_main/unit/scale_length",
    );

    let mut frame_nonfinite = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
    let scene_id = frame_nonfinite.active_scene.clone();
    frame_nonfinite
        .scenes
        .get_mut(&scene_id)
        .unwrap()
        .frame_current = f64::NAN;
    let report = validate::validate(&frame_nonfinite, None).unwrap();
    assert_issue_at(&report.value, "INVALID_SCENE_TIMING", "/scenes/scene_main");

    let mut fps_base_nonfinite = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
    let scene_id = fps_base_nonfinite.active_scene.clone();
    fps_base_nonfinite
        .scenes
        .get_mut(&scene_id)
        .unwrap()
        .fps_base = f64::INFINITY;
    let report = validate::validate(&fps_base_nonfinite, None).unwrap();
    assert_issue_at(&report.value, "INVALID_SCENE_TIMING", "/scenes/scene_main");

    let mut fps_zero = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
    let scene_id = fps_zero.active_scene.clone();
    fps_zero.scenes.get_mut(&scene_id).unwrap().fps = 0;
    let report = validate::validate(&fps_zero, None).unwrap();
    assert_issue_at(&report.value, "INVALID_SCENE_TIMING", "/scenes/scene_main");

    let mut inverted_range = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
    let scene_id = inverted_range.active_scene.clone();
    let scene = inverted_range.scenes.get_mut(&scene_id).unwrap();
    scene.frame_start = 2;
    scene.frame_end = 1;
    let report = validate::validate(&inverted_range, None).unwrap();
    assert_issue_at(&report.value, "INVALID_SCENE_TIMING", "/scenes/scene_main");
}

#[test]
fn validate_strict_allows_information_and_fails_on_format_losses() {
    let directory = tempdir().unwrap();
    let scene = directory.path().join("scene");
    init(&scene);
    let scene_file = scene.join("scene.json");
    let mut document: Value = serde_json::from_slice(&fs::read(&scene_file).unwrap()).unwrap();
    document["data_blocks"]["orphan"] = json!({"type":"mesh","mesh":null});
    fs::write(&scene_file, serde_json::to_vec_pretty(&document).unwrap()).unwrap();

    let information_only = command_output("validate", &scene, &["--strict"]);
    assert!(information_only.status.success());
    assert_eq!(response(&information_only)["result"]["valid"], true);
    assert!(
        response(&information_only)["result"]["issues"]
            .as_array()
            .unwrap()
            .iter()
            .any(|issue| issue["code"] == "UNUSED_DATA_BLOCK")
    );

    let created = apply(
        &scene,
        0,
        &json!([{"op":"node.create","id":"group_node","kind":"group"}]),
    );
    assert!(created.status.success());
    let strict_loss = command_output("validate", &scene, &["--format", "obj", "--strict"]);
    assert_eq!(strict_loss.status.code(), Some(4));
    let value = response(&strict_loss);
    assert_eq!(value["error"]["code"], "VALIDATION_FAILED");
    assert!(
        value["result"]["losses"]
            .as_array()
            .unwrap()
            .iter()
            .any(|loss| loss["feature_id"] == "node.empty")
    );
}

#[test]
fn assets_check_distinguishes_unchecked_missing_files_and_invalid_packed_bytes() {
    let directory = tempdir().unwrap();
    let scene = directory.path().join("scene");
    init(&scene);
    let scene_file = scene.join("scene.json");
    let mut document: Value = serde_json::from_slice(&fs::read(&scene_file).unwrap()).unwrap();
    document["resources"]["external"] =
        json!({"uri":"missing.bin","kind":"texture","packed":false});
    document["resources"]["packed"] =
        json!({"uri":"packed://payload","kind":"resource","packed":true,"bytes":[0,256]});
    fs::write(&scene_file, serde_json::to_vec_pretty(&document).unwrap()).unwrap();

    let unchecked = command_output("assets", &scene, &[]);
    assert!(unchecked.status.success());
    let unchecked_result = response(&unchecked)["result"].clone();
    let unchecked_asset = unchecked_result["assets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|asset| asset["uri"] == "missing.bin")
        .unwrap();
    assert_eq!(unchecked_asset["missing"], true);
    assert_eq!(unchecked_result["checked"], false);

    let invalid = command_output("assets", &scene, &["--check"]);
    assert_eq!(invalid.status.code(), Some(4));
    assert_eq!(response(&invalid)["error"]["code"], "SCENE_INVALID");
    assert_eq!(response(&invalid)["error"]["details"]["index"], 1);

    document["resources"]
        .as_object_mut()
        .unwrap()
        .remove("packed");
    fs::write(&scene_file, serde_json::to_vec_pretty(&document).unwrap()).unwrap();
    let checked = command_output("assets", &scene, &["--check"]);
    assert!(checked.status.success());
    let checked_result = response(&checked)["result"].clone();
    let missing = checked_result["assets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|asset| asset["uri"] == "missing.bin")
        .unwrap();
    assert_eq!(missing["missing"], true);
    assert_eq!(missing["changed"], false);
    assert_eq!(checked_result["summary"]["missing"], 1);
}

#[test]
fn camera_light_creation_accepts_common_node_fields_and_schemas_describe_them() {
    let doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
    let outcome = apply_doc(
        &doc,
        &json!([
            {"op":"node.create","id":"rig","kind":"empty"},
            {"op":"collection.create","id":"staging"},
            {"op":"camera.create","id":"camera_main","name":"Main","tags":["hero"],"parent":"rig","collection":"staging","transform":{"translation":[1,2,3]},"visible":false,"render_visible":false,"selectable":false},
            {"op":"light.create","id":"light_main","name":"Key","tags":["key_light"],"parent":"rig","collection":"staging","transform":{"translation":[4,5,6]},"render_visible":false,"selectable":false}
        ]),
    );
    let camera_id = Id::new("camera_main").unwrap();
    let light_id = Id::new("light_main").unwrap();
    let camera = &outcome.doc.nodes[&camera_id];
    assert_eq!(camera.name, "Main");
    assert_eq!(camera.tags, vec!["hero".to_owned()]);
    assert_eq!(camera.parent, Some(Id::new("rig").unwrap()));
    assert!(
        camera
            .transform
            .translation
            .into_iter()
            .zip([1.0, 2.0, 3.0])
            .all(|(actual, expected)| (actual - expected).abs() < 1.0e-12)
    );
    assert!(!camera.visible);
    assert!(!camera.render_visible);
    assert!(!camera.selectable);
    let light = &outcome.doc.nodes[&light_id];
    assert_eq!(light.name, "Key");
    assert_eq!(light.tags, vec!["key_light".to_owned()]);
    assert_eq!(light.parent, Some(Id::new("rig").unwrap()));
    assert!(
        light
            .transform
            .translation
            .into_iter()
            .zip([4.0, 5.0, 6.0])
            .all(|(actual, expected)| (actual - expected).abs() < 1.0e-12)
    );
    assert!(!light.render_visible);
    assert!(!light.selectable);
    let staging = &outcome.doc.collections[&Id::new("staging").unwrap()];
    assert!(staging.objects.contains(&camera_id));
    assert!(staging.objects.contains(&light_id));

    let schema = |operation: &str| {
        let output = pot()
            .args([
                "schema",
                "--kind",
                "operations",
                "--op",
                operation,
                "--json",
            ])
            .output()
            .unwrap();
        assert!(output.status.success());
        response(&output)["result"]["schema"].clone()
    };
    let camera_schema = schema("camera.create");
    let camera_validator = jsonschema::validator_for(&camera_schema).unwrap();
    assert!(
        camera_validator
            .is_valid(&json!({"op":"camera.create","id":"camera_main","tags":["hero"]}))
    );
    let light_schema = schema("light.create");
    let light_validator = jsonschema::validator_for(&light_schema).unwrap();
    assert!(
        light_validator
            .is_valid(&json!({"op":"light.create","id":"light_main","tags":["key_light"]}))
    );

    let delete_schema = schema("node.delete");
    assert_eq!(
        delete_schema["properties"]["reparent"]["enum"],
        json!(["to_parent", "to_root"])
    );
    assert_eq!(delete_schema["x-potter"]["defaults"]["keep_world"], true);
    let validator = jsonschema::validator_for(&delete_schema).unwrap();
    assert!(validator.is_valid(&json!({"op":"node.delete","target":{"id":"rig"}})));
    assert!(validator.is_valid(
        &json!({"op":"node.delete","target":{"id":"rig"},"reparent":"to_root","keep_world":false})
    ));
    assert!(!validator.is_valid(
        &json!({"op":"node.delete","target":{"id":"rig"},"recursive":true,"reparent":"to_root"})
    ));
    assert!(
        !validator.is_valid(&json!({"op":"node.delete","target":{"id":"rig"},"keep_world":false}))
    );
}
