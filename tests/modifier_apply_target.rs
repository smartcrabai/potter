//! `modifier.apply` follows Blender's `modifier_apply`: only the applied modifier is evaluated, on
//! the base data block, whatever its `show_viewport`. Every other modifier stays in the stack with
//! its order, flags and parameters.
//!
//! Blender 5.2.2 probes (background, factory startup) behind these expectations:
//! - cube + hidden ARRAY + SUBSURF(1), apply SUBSURF: 26 vertices, 24 faces, the same as SUBSURF
//!   alone on the cube; the stack becomes `[Array]` with `show_viewport` False.
//! - cube + visible ARRAY + DISPLACE, apply DISPLACE: 8 vertices, 6 faces, the same as DISPLACE
//!   alone; the stack becomes `[Array]` with its flags unchanged.
//! - cube + SUBSURF + hidden DISPLACE + SOLIDIFY, apply SUBSURF: 26/24 as alone; the stack becomes
//!   `[Displace (viewport off), Solidify]`.
//! - cube + hidden SUBSURF + hidden SOLIDIFY, apply SUBSURF: applied anyway (FINISHED), 26/24 as
//!   alone; the stack becomes `[Solidify]` with both flags False.

#![expect(
    clippy::float_cmp,
    reason = "the rigid-body check compares simulated matrices exactly to detect any motion"
)]

use std::{error::Error, fmt::Debug, fs, path::Path, process::Command};

use glam::{DQuat, EulerRot};
use potter_core::{
    error::{ErrorCode, PotError},
    model::{DataBlock, Id, Modifier, SceneDoc},
    ops,
};
use serde_json::{Value, json};
use tempfile::tempdir;

type TestResult<T> = Result<T, Box<dyn Error>>;

const SCENE_ID: &str = "00000000-0000-4000-8000-000000000000";
const SUBJECT: &str = "subject";
const OPERAND: &str = "operand";

fn ensure_eq<T: PartialEq + Debug>(actual: &T, expected: &T, what: &str) -> TestResult<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(format!("{what}: expected {expected:?}, got {actual:?}").into())
    }
}

fn run(doc: &SceneDoc, operation: &Value) -> Result<SceneDoc, PotError> {
    let batch = json!({
        "schema_version": 1,
        "base_revision": doc.revision,
        "operations": [operation],
    });
    Ok(ops::apply_batch(doc, &batch)?.doc)
}

/// A box node named `SUBJECT` followed by `extra`, each operation applied in turn.
fn scene_operations(extra: &[Value]) -> Vec<Value> {
    let mut operations =
        vec![json!({"op": "node.create", "id": SUBJECT, "kind": "box", "params": {"size": 2.0}})];
    operations.extend_from_slice(extra);
    operations
}

fn build(extra: &[Value]) -> TestResult<SceneDoc> {
    let operations = scene_operations(extra);
    let mut doc = SceneDoc::new(SCENE_ID.to_owned());
    for operation in &operations {
        doc = run(&doc, operation)?;
    }
    Ok(doc)
}

fn apply_op(modifier_id: &str) -> Value {
    json!({"op": "modifier.apply", "target": {"id": SUBJECT}, "modifier_id": modifier_id})
}

/// Builds the scene from `extra`, then applies the modifier `modifier_id`.
fn apply_target(extra: &[Value], modifier_id: &str) -> TestResult<SceneDoc> {
    let doc = build(extra)?;
    Ok(run(&doc, &apply_op(modifier_id))?)
}
fn all_vertex_weights(doc: &SceneDoc, node_id: &str) -> TestResult<Vec<Value>> {
    Ok(data_block_for(doc, node_id)?
        .mesh
        .as_ref()
        .ok_or("node mesh is missing")?
        .vertices
        .iter()
        .map(|vertex| json!({"vertex_id": vertex.id, "weight": 1.0}))
        .collect())
}

fn simulated_subject_operations() -> Vec<Value> {
    vec![
        json!({
            "op": "physics.world.update",
            "target": {"id": "scene_main"},
            "set": {
                "enabled": true,
                "gravity": [0.0, 0.0, 0.0],
                "substeps": 8,
                "solver_iterations": 16,
                "frame_start": 1,
                "frame_end": 120,
                "seed": 23,
            },
        }),
        json!({
            "op": "physics.rigid_body.create",
            "target": {"id": SUBJECT},
            "type": "active",
            "mass": 1.0,
            "shape": "box",
            "friction": 0.0,
            "restitution": 0.0,
            "linear_damping": 0.0,
            "angular_damping": 0.0,
            "initial_velocity": [1.0, 0.0, 0.0],
        }),
    ]
}

fn snapshot_after_subject_motion(
    doc: &SceneDoc,
    frame: f64,
    modifier_name: &str,
) -> TestResult<potter_core::geom::Mesh> {
    let subject_id = Id::new(SUBJECT)?;
    let final_snapshot = potter_core::eval::Snapshot::evaluate(
        doc,
        &potter_core::eval::EvaluationContext {
            frame: Some(frame),
            ..potter_core::eval::EvaluationContext::default()
        },
    )?;
    let initial_snapshot = potter_core::eval::Snapshot::evaluate(
        doc,
        &potter_core::eval::EvaluationContext {
            frame: Some(1.0),
            ..potter_core::eval::EvaluationContext::default()
        },
    )?;
    let final_world = &final_snapshot
        .nodes
        .get(&subject_id)
        .ok_or("simulated subject is missing from Snapshot")?
        .world_matrix;
    let initial_world = &initial_snapshot
        .nodes
        .get(&subject_id)
        .ok_or("initial subject is missing from Snapshot")?
        .world_matrix;
    if final_world == initial_world {
        return Err(format!("rigid-body subject did not move before {modifier_name} apply").into());
    }
    let final_mesh = final_snapshot
        .meshes
        .get(&subject_id)
        .ok_or("simulated subject mesh is missing from Snapshot")?;
    let initial_mesh = initial_snapshot
        .meshes
        .get(&subject_id)
        .ok_or("initial subject mesh is missing from Snapshot")?;
    if final_mesh == initial_mesh {
        return Err(format!("rigid-body motion did not change {modifier_name} output").into());
    }
    Ok(final_mesh.clone())
}

fn hide_target_modifier(operations: &[Value], modifier_id: &str) -> TestResult<Vec<Value>> {
    let mut hidden = operations.to_vec();
    let target = hidden
        .iter_mut()
        .find(|operation| {
            operation.get("op").and_then(Value::as_str) == Some("modifier.create")
                && operation.get("id").and_then(Value::as_str) == Some(modifier_id)
        })
        .ok_or("target modifier create operation is missing")?;
    target["show_viewport"] = json!(false);
    Ok(hidden)
}

fn run_cli(command: &mut Command, label: &str) -> TestResult<()> {
    let output = command.output()?;
    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "{label} failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        )
        .into())
    }
}

fn run_cli_batch(scene: &Path, batch_path: &Path, batch: &Value) -> TestResult<()> {
    fs::write(batch_path, serde_json::to_vec(batch)?)?;
    let mut command = Command::new(env!("CARGO_BIN_EXE_pot"));
    command
        .args(["apply"])
        .arg(scene)
        .args(["--file"])
        .arg(batch_path)
        .arg("--json");
    run_cli(&mut command, "pot apply")
}

fn apply_project_target_and_reopen(
    extra: &[Value],
    modifier_id: &str,
    frame: f64,
) -> TestResult<SceneDoc> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let mut init = Command::new(env!("CARGO_BIN_EXE_pot"));
    init.args(["init"]).arg(&scene).arg("--json");
    run_cli(&mut init, "pot init")?;

    let batch_path = directory.path().join("operations.json");
    run_cli_batch(
        &scene,
        &batch_path,
        &json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": scene_operations(extra),
        }),
    )?;
    let project = potter_core::store::Project::open(&scene)?;
    let base_revision = project.doc().revision;
    let expected_stack = without(&node_modifiers_for(project.doc(), SUBJECT)?, modifier_id);
    drop(project);

    run_cli_batch(
        &scene,
        &batch_path,
        &json!({
            "schema_version": 1,
            "base_revision": base_revision,
            "evaluation": {"frame": frame},
            "operations": [apply_op(modifier_id)],
        }),
    )?;
    let reopened = potter_core::store::Project::open(&scene)?;
    ensure_eq(
        &node_modifiers_for(reopened.doc(), SUBJECT)?,
        &expected_stack,
        "reopened target modifier stack",
    )?;
    Ok(reopened.doc().clone())
}

fn modifier(
    id: &str,
    modifier_type: &str,
    params: Value,
    show_viewport: bool,
    show_render: bool,
) -> Value {
    let mut value = json!({
        "op": "modifier.create",
        "target": {"id": SUBJECT},
        "id": id,
        "type": modifier_type,
        "show_viewport": show_viewport,
        "show_render": show_render,
    });
    value["params"] = params;
    value
}

fn node_modifiers_for(doc: &SceneDoc, node_id: &str) -> TestResult<Vec<Modifier>> {
    let node = doc.nodes.get(&Id::new(node_id)?).ok_or("node is missing")?;
    Ok(node.modifiers.clone())
}

fn node_modifiers(doc: &SceneDoc) -> TestResult<Vec<Modifier>> {
    node_modifiers_for(doc, SUBJECT)
}

fn data_block_for<'a>(doc: &'a SceneDoc, node_id: &str) -> TestResult<&'a DataBlock> {
    let node = doc.nodes.get(&Id::new(node_id)?).ok_or("node is missing")?;
    let data_id = node.data.as_ref().ok_or("node has no geometry data")?;
    doc.data_blocks
        .get(data_id)
        .ok_or_else(|| "node data block is missing".into())
}

fn data_block(doc: &SceneDoc) -> TestResult<&DataBlock> {
    data_block_for(doc, SUBJECT)
}

fn unsupported_bevel_source(transform: &Value) -> [Value; 2] {
    [
        json!({
            "op": "node.create",
            "id": OPERAND,
            "kind": "box",
            "params": {"size": 2.0},
            "transform": transform,
        }),
        json!({
            "op": "modifier.create",
            "target": {"id": OPERAND},
            "id": "unsupported_bevel",
            "type": "bevel",
            "show_viewport": true,
            "show_render": true,
            "params": {"width": 0.1, "segments": 2, "profile": 0.0, "limit_method": "NONE"},
        }),
    ]
}

fn add_unsupported_bevel_uv(doc: &mut SceneDoc) -> TestResult<()> {
    let operand_id = Id::new(OPERAND)?;
    let data_id = doc
        .nodes
        .get(&operand_id)
        .and_then(|node| node.data.as_ref())
        .ok_or("operand has no geometry data")?
        .clone();
    if let Some(node) = doc.nodes.get_mut(&operand_id) {
        node.primitive = None;
    }
    let data = doc
        .data_blocks
        .get_mut(&data_id)
        .ok_or("operand data block is missing")?;
    let mesh = data.mesh.as_mut().ok_or("operand mesh is missing")?;
    let values = mesh
        .faces
        .iter()
        .flat_map(|face| face.vertices.iter().map(|_| json!([0.0, 0.0])))
        .collect::<Vec<_>>();
    mesh.attributes.insert(
        "blender_uv_layers".to_owned(),
        json!([{"name": "UVMap", "values": values}]),
    );
    data.descriptor = None;
    Ok(())
}

fn vertex_group_names(doc: &SceneDoc) -> TestResult<Vec<String>> {
    Ok(data_block(doc)?
        .vertex_groups
        .iter()
        .map(|group| group.name.clone())
        .collect())
}

fn ids_and_flags(doc: &SceneDoc) -> TestResult<Vec<(String, bool, bool)>> {
    Ok(node_modifiers(doc)?
        .iter()
        .map(|modifier| {
            (
                modifier.id.to_string(),
                modifier.show_viewport,
                modifier.show_render,
            )
        })
        .collect())
}

/// `modifiers` without the modifier `removed`: what must remain after that modifier is applied.
fn without(modifiers: &[Modifier], removed: &str) -> Vec<Modifier> {
    modifiers
        .iter()
        .filter(|modifier| modifier.id.to_string() != removed)
        .cloned()
        .collect()
}

/// The applied scene stores the same base data as the scene where the target was alone: mesh,
/// vertex-group names and vertex weights.
fn assert_matches_alone(applied: &SceneDoc, alone: &SceneDoc) -> TestResult<()> {
    ensure_eq(
        &data_block(applied)?.mesh,
        &data_block(alone)?.mesh,
        "stored mesh",
    )?;
    ensure_eq(
        &data_block(applied)?.vertex_weights,
        &data_block(alone)?.vertex_weights,
        "vertex weights",
    )?;
    ensure_eq(
        &vertex_group_names(applied)?,
        &vertex_group_names(alone)?,
        "vertex group names",
    )
}

#[test]
fn hidden_earlier_array_stays_and_the_target_applies_to_base() -> TestResult<()> {
    let array = modifier(
        "array",
        "array",
        json!({"count": 3, "relative_offset_displace": [1.5, 0.0, 0.0], "use_relative_offset": true}),
        false,
        true,
    );
    let subdivision = modifier(
        "subdivision",
        "subdivision",
        json!({"levels": 1}),
        true,
        true,
    );
    let before = node_modifiers(&build(&[array.clone(), subdivision.clone()])?)?;

    let applied = apply_target(&[array, subdivision.clone()], "subdivision")?;
    let alone = apply_target(&[subdivision], "subdivision")?;

    assert_matches_alone(&applied, &alone)?;
    let mesh = data_block(&applied)?
        .mesh
        .as_ref()
        .ok_or("subject mesh is missing")?;
    // Catmull-Clark of a cube: 8 + 6 + 12 = 26 vertices, 6 * 4 = 24 faces (Blender agrees).
    ensure_eq(&mesh.vertices.len(), &26, "subdivided vertices")?;
    ensure_eq(&mesh.faces.len(), &24, "subdivided faces")?;
    ensure_eq(
        &ids_and_flags(&applied)?,
        &vec![("array".to_owned(), false, true)],
        "remaining flags",
    )?;
    ensure_eq(
        &node_modifiers(&applied)?,
        &without(&before, "subdivision"),
        "remaining modifiers",
    )
}

#[test]
fn visible_array_before_displace_is_not_applied_to_base() -> TestResult<()> {
    let array = modifier(
        "array",
        "array",
        json!({"count": 3, "relative_offset_displace": [1.5, 0.0, 0.0], "use_relative_offset": true}),
        true,
        true,
    );
    let displace = modifier(
        "displace",
        "displace",
        json!({"strength": 0.5, "mid_level": 0.0}),
        true,
        true,
    );
    let before = node_modifiers(&build(&[array.clone(), displace.clone()])?)?;

    let applied = apply_target(&[array, displace.clone()], "displace")?;
    let alone = apply_target(&[displace], "displace")?;

    assert_matches_alone(&applied, &alone)?;
    let vertices = data_block(&applied)?
        .mesh
        .as_ref()
        .map(|mesh| mesh.vertices.len());
    // Displace keeps the cube's 8 vertices; the array's copies were never evaluated.
    ensure_eq(&vertices, &Some(8), "displaced vertices")?;
    ensure_eq(
        &ids_and_flags(&applied)?,
        &vec![("array".to_owned(), true, true)],
        "remaining flags",
    )?;
    ensure_eq(
        &node_modifiers(&applied)?,
        &without(&before, "displace"),
        "remaining modifiers",
    )
}

#[test]
fn modifiers_after_the_target_keep_their_order_and_flags() -> TestResult<()> {
    let array = modifier(
        "array",
        "array",
        json!({"count": 3, "relative_offset_displace": [1.5, 0.0, 0.0], "use_relative_offset": true}),
        true,
        false,
    );
    let subdivision = modifier(
        "subdivision",
        "subdivision",
        json!({"levels": 1}),
        true,
        true,
    );
    let displace = modifier(
        "displace",
        "displace",
        json!({"strength": 0.5, "mid_level": 0.0}),
        false,
        true,
    );
    let solidify = modifier(
        "solidify",
        "solidify",
        json!({"thickness": 0.1}),
        true,
        false,
    );
    let extra = [array, subdivision.clone(), displace, solidify];
    let before = node_modifiers(&build(&extra)?)?;

    let applied = apply_target(&extra, "subdivision")?;
    let alone = apply_target(&[subdivision], "subdivision")?;

    assert_matches_alone(&applied, &alone)?;
    ensure_eq(
        &ids_and_flags(&applied)?,
        &vec![
            ("array".to_owned(), true, false),
            ("displace".to_owned(), false, true),
            ("solidify".to_owned(), true, false),
        ],
        "order and flags",
    )?;
    ensure_eq(
        &node_modifiers(&applied)?,
        &without(&before, "subdivision"),
        "remaining modifiers",
    )
}

#[test]
fn hidden_target_is_applied_and_its_viewport_flag_is_ignored() -> TestResult<()> {
    let subdivision = modifier(
        "subdivision",
        "subdivision",
        json!({"levels": 1}),
        false,
        true,
    );
    let solidify = modifier(
        "solidify",
        "solidify",
        json!({"thickness": 0.1}),
        false,
        false,
    );
    let extra = [subdivision.clone(), solidify];
    let before = node_modifiers(&build(&extra)?)?;

    let applied = apply_target(&extra, "subdivision")?;
    let alone = apply_target(&[subdivision], "subdivision")?;

    assert_matches_alone(&applied, &alone)?;
    ensure_eq(
        &ids_and_flags(&applied)?,
        &vec![("solidify".to_owned(), false, false)],
        "remaining flags",
    )?;
    ensure_eq(
        &node_modifiers(&applied)?,
        &without(&before, "subdivision"),
        "remaining modifiers",
    )
}

#[test]
fn array_offset_object_apply_uses_transform_without_evaluating_source_geometry() -> TestResult<()> {
    let mut extra = unsupported_bevel_source(&json!({"translation": [3.0, 0.0, 0.0]}))
        .into_iter()
        .collect::<Vec<_>>();
    let array = modifier(
        "array",
        "array",
        json!({
            "count": 3,
            "fit_type": "FIXED_COUNT",
            "use_relative_offset": false,
            "use_constant_offset": false,
            "use_object_offset": true,
            "offset_object": OPERAND,
        }),
        false,
        false,
    );
    let solidify = modifier(
        "solidify",
        "solidify",
        json!({"thickness": 0.1, "use_pin_to_last": true}),
        true,
        true,
    );
    extra.extend([array, solidify]);
    let mut doc = build(&extra)?;
    add_unsupported_bevel_uv(&mut doc)?;
    let before = node_modifiers(&doc)?;
    let operand_before = node_modifiers_for(&doc, OPERAND)?;

    let applied = run(&doc, &apply_op("array"))?;

    let mesh = data_block(&applied)?
        .mesh
        .as_ref()
        .ok_or("subject mesh is missing")?;
    ensure_eq(&mesh.vertices.len(), &24, "array vertices")?;
    let max_x = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co.x)
        .fold(f64::NEG_INFINITY, f64::max);
    ensure_eq(&max_x, &7.0, "array offset-object max X")?;
    ensure_eq(
        &node_modifiers(&applied)?,
        &without(&before, "array"),
        "remaining subject modifiers",
    )?;
    ensure_eq(
        &node_modifiers_for(&applied, OPERAND)?,
        &operand_before,
        "source modifiers",
    )
}

#[test]
fn screw_object_apply_avoids_unrelated_source_geometry() -> TestResult<()> {
    let mut extra = unsupported_bevel_source(&json!({"translation": [0.0, 0.0, 0.5]}))
        .into_iter()
        .collect::<Vec<_>>();
    let screw = modifier(
        "screw",
        "screw",
        json!({
            "steps": 2,
            "axis": "Z",
            "angle": std::f64::consts::FRAC_PI_2,
            "object": OPERAND,
            "use_object_screw_offset": true,
        }),
        false,
        false,
    );
    let solidify = modifier(
        "solidify",
        "solidify",
        json!({"thickness": 0.1, "use_pin_to_last": true}),
        true,
        true,
    );
    extra.extend([screw, solidify]);
    let mut doc = build(&extra)?;
    add_unsupported_bevel_uv(&mut doc)?;
    let before = node_modifiers(&doc)?;
    let operand_before = node_modifiers_for(&doc, OPERAND)?;

    let applied = run(&doc, &apply_op("screw"))?;

    let mesh = data_block(&applied)?
        .mesh
        .as_ref()
        .ok_or("subject mesh is missing")?;
    ensure_eq(&mesh.vertices.len(), &24, "screw vertices")?;
    ensure_eq(
        &node_modifiers(&applied)?,
        &without(&before, "screw"),
        "remaining subject modifiers",
    )?;
    ensure_eq(
        &node_modifiers_for(&applied, OPERAND)?,
        &operand_before,
        "source modifiers",
    )
}

#[test]
fn array_fit_curve_apply_uses_curve_data_and_world_transform() -> TestResult<()> {
    let mut extra = unsupported_bevel_source(&json!({"translation": [9.0, 0.0, 0.0]}))
        .into_iter()
        .collect::<Vec<_>>();
    extra.push(json!({
        "op": "curve.create",
        "id": "path",
        "dimensions": "three_d",
        "transform": {"scale": [2.0, 2.0, 2.0]},
        "splines": [{
            "type": "poly",
            "points": [{"co": [0.0, 0.0, 0.0]}, {"co": [2.0, 0.0, 0.0]}],
            "order": 2,
            "cyclic": false,
            "resolution": 1,
            "use_endpoint": true,
        }],
    }));
    let array = modifier(
        "array",
        "array",
        json!({
            "count": 1,
            "fit_type": "FIT_CURVE",
            "curve": "path",
            "use_relative_offset": true,
            "relative_offset_displace": [1.0, 0.0, 0.0],
            "use_constant_offset": false,
        }),
        false,
        false,
    );
    let solidify = modifier(
        "solidify",
        "solidify",
        json!({"thickness": 0.1, "use_pin_to_last": true}),
        true,
        true,
    );
    extra.extend([array, solidify]);
    let mut doc = build(&extra)?;
    add_unsupported_bevel_uv(&mut doc)?;
    let before = node_modifiers(&doc)?;

    let applied = run(&doc, &apply_op("array"))?;

    let mesh = data_block(&applied)?
        .mesh
        .as_ref()
        .ok_or("subject mesh is missing")?;
    ensure_eq(&mesh.vertices.len(), &24, "fit-curve array vertices")?;
    ensure_eq(
        &node_modifiers(&applied)?,
        &without(&before, "array"),
        "remaining subject modifiers",
    )
}

#[test]
fn array_geometry_operand_uses_source_viewport_stack_visibility() -> TestResult<()> {
    let apply_with_visibility = |source_visible| -> TestResult<SceneDoc> {
        let source = json!({
            "op": "node.create",
            "id": OPERAND,
            "kind": "box",
            "params": {"size": 2.0},
        });
        let source_displace = json!({
            "op": "modifier.create",
            "target": {"id": OPERAND},
            "id": "source_displace",
            "type": "displace",
            "show_viewport": source_visible,
            "show_render": true,
            "params": {"direction": "Z", "strength": 1.0, "mid_level": 0.0},
        });
        let array = modifier(
            "array",
            "array",
            json!({
                "count": 1,
                "fit_type": "FIXED_COUNT",
                "use_relative_offset": false,
                "use_constant_offset": false,
                "start_cap": OPERAND,
            }),
            false,
            false,
        );
        let solidify = modifier(
            "solidify",
            "solidify",
            json!({"thickness": 0.1, "use_pin_to_last": true}),
            true,
            true,
        );
        let doc = build(&[source, source_displace, array, solidify])?;
        let before = node_modifiers(&doc)?;
        let operand_before = node_modifiers_for(&doc, OPERAND)?;
        let applied = run(&doc, &apply_op("array"))?;
        ensure_eq(
            &node_modifiers(&applied)?,
            &without(&before, "array"),
            "remaining subject modifiers",
        )?;
        ensure_eq(
            &node_modifiers_for(&applied, OPERAND)?,
            &operand_before,
            "source modifier flags",
        )?;
        Ok(applied)
    };
    let hidden_source = apply_with_visibility(false)?;
    let visible_source = apply_with_visibility(true)?;
    let max_z = |doc: &SceneDoc| -> TestResult<f64> {
        let mesh = data_block(doc)?
            .mesh
            .as_ref()
            .ok_or("subject mesh is missing")?;
        Ok(mesh
            .vertices
            .iter()
            .map(|vertex| vertex.co.z)
            .fold(f64::NEG_INFINITY, f64::max))
    };

    ensure_eq(&max_z(&hidden_source)?, &1.0, "hidden source stack max Z")?;
    ensure_eq(&max_z(&visible_source)?, &2.0, "visible source stack max Z")
}

#[test]
fn boolean_apply_uses_simulated_geometry_operand_world_transform() -> TestResult<()> {
    let scene = |target_visible| -> TestResult<SceneDoc> {
        let operand = json!({
            "op": "node.create",
            "id": OPERAND,
            "kind": "box",
            "params": {"size": 1.0},
            "transform": {"translation": [0.0, 0.0, 5.0]},
        });
        let physics_world = json!({
            "op": "physics.world.update",
            "target": {"id": "scene_main"},
            "set": {
                "enabled": true,
                "gravity": [0.0, 0.0, -9.81],
                "substeps": 4,
                "solver_iterations": 8,
                "frame_start": 1,
                "frame_end": 120,
                "seed": 23,
            },
        });
        let active_operand = json!({
            "op": "physics.rigid_body.create",
            "target": {"id": OPERAND},
            "type": "active",
            "mass": 1.0,
            "shape": "box",
            "friction": 0.5,
            "restitution": 0.0,
            "linear_damping": 0.0,
            "angular_damping": 0.0,
        });
        let boolean = modifier(
            "boolean",
            "boolean",
            json!({"object": OPERAND, "operation": "DIFFERENCE", "solver": "EXACT"}),
            target_visible,
            true,
        );
        build(&[operand, physics_world, active_operand, boolean])
    };
    let visible_target = scene(true)?;
    let hidden_target = scene(false)?;
    let subject_id = Id::new(SUBJECT)?;
    let frame_context = potter_core::eval::EvaluationContext {
        frame: Some(24.0),
        ..potter_core::eval::EvaluationContext::default()
    };
    let expected_snapshot = potter_core::eval::Snapshot::evaluate(&visible_target, &frame_context)?;
    let expected_mesh = expected_snapshot
        .meshes
        .get(&subject_id)
        .ok_or("evaluated target mesh is missing")?;
    let initial_snapshot = potter_core::eval::Snapshot::evaluate(
        &visible_target,
        &potter_core::eval::EvaluationContext {
            frame: Some(1.0),
            ..potter_core::eval::EvaluationContext::default()
        },
    )?;
    let initial_mesh = initial_snapshot
        .meshes
        .get(&subject_id)
        .ok_or("initial target mesh is missing")?;
    if expected_mesh == initial_mesh {
        return Err("rigid-body operand did not move through the Boolean surface".into());
    }

    let batch = json!({
        "schema_version": 1,
        "base_revision": hidden_target.revision,
        "evaluation": {"frame": 24.0},
        "operations": [apply_op("boolean")],
    });
    let applied = ops::apply_batch(&hidden_target, &batch)?.doc;
    let actual_mesh = data_block(&applied)?
        .mesh
        .as_ref()
        .ok_or("applied target mesh is missing")?;
    ensure_eq(
        actual_mesh,
        expected_mesh,
        "applied Boolean uses the simulated operand world transform",
    )
}

#[test]
fn warp_apply_uses_simulated_transform_reference_world_matrix() -> TestResult<()> {
    let scene = |target_visible| -> TestResult<SceneDoc> {
        let from = json!({
            "op": "node.create",
            "id": "warp_from",
            "kind": "empty",
            "transform": {"translation": [-1.0, 0.0, 0.0]},
        });
        let operand = json!({
            "op": "node.create",
            "id": OPERAND,
            "kind": "box",
            "params": {"size": 1.0},
            "transform": {"translation": [1.0, 0.0, 5.0]},
        });
        let physics_world = json!({
            "op": "physics.world.update",
            "target": {"id": "scene_main"},
            "set": {
                "enabled": true,
                "gravity": [0.0, 0.0, -9.81],
                "substeps": 4,
                "solver_iterations": 8,
                "frame_start": 1,
                "frame_end": 120,
                "seed": 23,
            },
        });
        let active_operand = json!({
            "op": "physics.rigid_body.create",
            "target": {"id": OPERAND},
            "type": "active",
            "mass": 1.0,
            "shape": "box",
            "friction": 0.5,
            "restitution": 0.0,
            "linear_damping": 0.0,
            "angular_damping": 0.0,
        });
        let warp = modifier(
            "warp",
            "warp",
            json!({
                "object_from": "warp_from",
                "object_to": OPERAND,
                "strength": 1.0,
                "falloff_type": "NONE",
            }),
            target_visible,
            true,
        );
        build(&[from, operand, physics_world, active_operand, warp])
    };
    let visible_target = scene(true)?;
    let hidden_target = scene(false)?;
    let subject_id = Id::new(SUBJECT)?;
    let expected_snapshot = potter_core::eval::Snapshot::evaluate(
        &visible_target,
        &potter_core::eval::EvaluationContext {
            frame: Some(24.0),
            ..potter_core::eval::EvaluationContext::default()
        },
    )?;
    let expected_mesh = expected_snapshot
        .meshes
        .get(&subject_id)
        .ok_or("evaluated target mesh is missing")?;
    let initial_snapshot = potter_core::eval::Snapshot::evaluate(
        &visible_target,
        &potter_core::eval::EvaluationContext {
            frame: Some(1.0),
            ..potter_core::eval::EvaluationContext::default()
        },
    )?;
    let initial_mesh = initial_snapshot
        .meshes
        .get(&subject_id)
        .ok_or("initial target mesh is missing")?;
    if expected_mesh == initial_mesh {
        return Err("rigid-body transform reference did not move the Warp output".into());
    }

    let batch = json!({
        "schema_version": 1,
        "base_revision": hidden_target.revision,
        "evaluation": {"frame": 24.0},
        "operations": [apply_op("warp")],
    });
    let applied = ops::apply_batch(&hidden_target, &batch)?.doc;
    let actual_mesh = data_block(&applied)?
        .mesh
        .as_ref()
        .ok_or("applied target mesh is missing")?;
    ensure_eq(
        actual_mesh,
        expected_mesh,
        "applied Warp uses the simulated transform reference world matrix",
    )
}

#[test]
fn hook_apply_uses_simulated_transform_reference_world_matrix() -> TestResult<()> {
    let scene = |target_visible| -> TestResult<SceneDoc> {
        let operand = json!({
            "op": "node.create",
            "id": OPERAND,
            "kind": "box",
            "params": {"size": 1.0},
            "transform": {"translation": [0.0, 0.0, 5.0]},
        });
        let physics_world = json!({
            "op": "physics.world.update",
            "target": {"id": "scene_main"},
            "set": {
                "enabled": true,
                "gravity": [0.0, 0.0, -9.81],
                "substeps": 4,
                "solver_iterations": 8,
                "frame_start": 1,
                "frame_end": 120,
                "seed": 23,
            },
        });
        let active_operand = json!({
            "op": "physics.rigid_body.create",
            "target": {"id": OPERAND},
            "type": "active",
            "mass": 1.0,
            "shape": "box",
            "friction": 0.5,
            "restitution": 0.0,
            "linear_damping": 0.0,
            "angular_damping": 0.0,
        });
        // Blender leaves a Hook with no listed vertices and no group unchanged (oracle
        // `hook_noop_no_indices_no_group`), so this hook lists every vertex to move the mesh.
        let hook = modifier(
            "hook",
            "hook",
            json!({
                "object": OPERAND,
                "center": [0.0, 0.0, 0.0],
                "strength": 1.0,
                "falloff_radius": 0.0,
                "vertex_indices": [0, 1, 2, 3, 4, 5, 6, 7],
            }),
            target_visible,
            true,
        );
        build(&[operand, physics_world, active_operand, hook])
    };
    let visible_target = scene(true)?;
    let hidden_target = scene(false)?;
    let subject_id = Id::new(SUBJECT)?;
    let expected_snapshot = potter_core::eval::Snapshot::evaluate(
        &visible_target,
        &potter_core::eval::EvaluationContext {
            frame: Some(24.0),
            ..potter_core::eval::EvaluationContext::default()
        },
    )?;
    let expected_mesh = expected_snapshot
        .meshes
        .get(&subject_id)
        .ok_or("evaluated target mesh is missing")?;
    let initial_snapshot = potter_core::eval::Snapshot::evaluate(
        &visible_target,
        &potter_core::eval::EvaluationContext {
            frame: Some(1.0),
            ..potter_core::eval::EvaluationContext::default()
        },
    )?;
    let initial_mesh = initial_snapshot
        .meshes
        .get(&subject_id)
        .ok_or("initial target mesh is missing")?;
    if expected_mesh == initial_mesh {
        return Err("rigid-body transform reference did not move the Hook output".into());
    }

    let batch = json!({
        "schema_version": 1,
        "base_revision": hidden_target.revision,
        "evaluation": {"frame": 24.0},
        "operations": [apply_op("hook")],
    });
    let applied = ops::apply_batch(&hidden_target, &batch)?.doc;
    let actual_mesh = data_block(&applied)?
        .mesh
        .as_ref()
        .ok_or("applied target mesh is missing")?;
    ensure_eq(
        actual_mesh,
        expected_mesh,
        "applied Hook uses the simulated transform reference world matrix",
    )
}

#[test]
fn armature_apply_uses_the_simulated_subject_transform_and_persists() -> TestResult<()> {
    let mut operations = simulated_subject_operations();
    let unrigged = build(&operations)?;
    let weights = all_vertex_weights(&unrigged, SUBJECT)?;
    let rotation = DQuat::from_euler(EulerRot::XYZ, 0.45, -0.35, 0.55);
    operations.extend([
        json!({"op": "node.create", "id": "rig", "kind": "armature"}),
        json!({
            "op": "bone.create",
            "target": {"id": "rig"},
            "id": "root",
            "name": "Root",
            "head": [0.0, 0.0, -1.0],
            "tail": [0.0, 1.0, 1.0],
        }),
        json!({
            "op": "pose.set",
            "target": {"id": "rig"},
            "bone_id": "root",
            "set": {
                "translation": [0.2, -0.1, 0.3],
                "rotation": [rotation.x, rotation.y, rotation.z, rotation.w],
                "scale": [1.0, 1.0, 1.0],
            },
        }),
        json!({"op": "vertex_group.create", "target": {"id": SUBJECT}, "id": "root_group", "name": "Root"}),
        json!({
            "op": "vertex_group.assign",
            "target": {"id": SUBJECT},
            "group_id": "root_group",
            "weights": weights,
        }),
        modifier(
            "armature",
            "armature",
            json!({"object": "rig", "use_vertex_groups": true}),
            true,
            true,
        ),
        modifier("tail", "subdivision", json!({"levels": 1}), false, false),
    ]);
    let visible_target = build(&operations)?;
    let expected_mesh = snapshot_after_subject_motion(&visible_target, 24.0, "Armature")?;
    let hidden_target = hide_target_modifier(&operations, "armature")?;
    let reopened = apply_project_target_and_reopen(&hidden_target, "armature", 24.0)?;
    let actual_mesh = data_block(&reopened)?
        .mesh
        .as_ref()
        .ok_or("reopened Armature-applied mesh is missing")?;
    ensure_eq(
        actual_mesh,
        &expected_mesh,
        "reopened Armature apply uses the simulated subject transform",
    )
}

#[test]
fn lattice_apply_uses_the_simulated_subject_transform_and_persists() -> TestResult<()> {
    let mut operations = simulated_subject_operations();
    let lattice_points = [
        [-1.0, -1.0, -1.0],
        [1.4, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [1.4, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.4, -1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [1.4, 1.0, 1.0],
    ];
    operations.extend([
        json!({
            "op": "lattice.create",
            "id": "cage",
            "transform": {
                "translation": [0.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "scale": [2.0, 1.0, 1.0],
            },
        }),
        json!({
            "op": "lattice.update",
            "target": {"id": "cage"},
            "set": {"points": lattice_points},
        }),
        modifier("lattice", "lattice", json!({"object": "cage"}), true, true),
        modifier("tail", "subdivision", json!({"levels": 1}), false, false),
    ]);
    let visible_target = build(&operations)?;
    let expected_mesh = snapshot_after_subject_motion(&visible_target, 24.0, "Lattice")?;
    let hidden_target = hide_target_modifier(&operations, "lattice")?;
    let reopened = apply_project_target_and_reopen(&hidden_target, "lattice", 24.0)?;
    let actual_mesh = data_block(&reopened)?
        .mesh
        .as_ref()
        .ok_or("reopened Lattice-applied mesh is missing")?;
    ensure_eq(
        actual_mesh,
        &expected_mesh,
        "reopened Lattice apply uses the simulated subject transform",
    )
}

#[test]
fn array_geometry_operand_propagates_unsupported_source_stack_errors() -> TestResult<()> {
    let mut extra = unsupported_bevel_source(&json!({"translation": [3.0, 0.0, 0.0]}))
        .into_iter()
        .collect::<Vec<_>>();
    extra.push(modifier(
        "array",
        "array",
        json!({
            "count": 1,
            "fit_type": "FIXED_COUNT",
            "use_relative_offset": false,
            "use_constant_offset": false,
            "start_cap": OPERAND,
        }),
        false,
        false,
    ));
    let mut doc = build(&extra)?;
    add_unsupported_bevel_uv(&mut doc)?;

    let error = run(&doc, &apply_op("array"))
        .err()
        .ok_or("unsupported geometry operand must block apply")?;
    ensure_eq(
        &error.code,
        &ErrorCode::UnsupportedFeature,
        "operand error code",
    )
}
#[test]
fn vertex_groups_persist_and_match_the_target_alone() -> TestResult<()> {
    let groups = vec![
        json!({"op": "vertex_group.create", "target": {"id": SUBJECT}, "id": "anchor", "name": "Anchor"}),
        json!({
            "op": "vertex_group.assign",
            "target": {"id": SUBJECT},
            "group_id": "anchor",
            "weights": [{"vertex_id": 0, "weight": 1.0}, {"vertex_id": 1, "weight": 0.5}],
        }),
    ];
    let array = modifier(
        "array",
        "array",
        json!({"count": 3, "relative_offset_displace": [1.5, 0.0, 0.0], "use_relative_offset": true}),
        false,
        true,
    );
    let subdivision = modifier(
        "subdivision",
        "subdivision",
        json!({"levels": 1}),
        true,
        true,
    );

    let mut stacked = groups.clone();
    stacked.extend([array, subdivision.clone()]);
    let applied = apply_target(&stacked, "subdivision")?;
    let mut alone_operations = groups;
    alone_operations.push(subdivision);
    let alone = apply_target(&alone_operations, "subdivision")?;

    assert_matches_alone(&applied, &alone)?;
    ensure_eq(
        &vertex_group_names(&applied)?,
        &vec!["Anchor".to_owned()],
        "persisted vertex groups",
    )?;
    ensure_eq(
        &ids_and_flags(&applied)?,
        &vec![("array".to_owned(), false, true)],
        "remaining flags",
    )
}

#[test]
fn shape_keys_still_refuse_apply_when_a_predecessor_is_hidden() -> TestResult<()> {
    let shape_key = json!({
        "op": "shape_key.create",
        "target": {"id": SUBJECT},
        "id": "smile",
        "name": "Smile",
        "positions": {"0": [1.0, -1.0, 0.25]},
    });
    let array = modifier(
        "array",
        "array",
        json!({"count": 3, "relative_offset_displace": [1.5, 0.0, 0.0], "use_relative_offset": true}),
        false,
        true,
    );
    let subdivision = modifier(
        "subdivision",
        "subdivision",
        json!({"levels": 1}),
        true,
        true,
    );
    let doc = build(&[shape_key, array, subdivision])?;

    let error = run(&doc, &apply_op("subdivision"))
        .err()
        .ok_or("shape keys must refuse the apply")?;
    ensure_eq(&error.code, &ErrorCode::UnsupportedFeature, "error code")?;
    ensure_eq(
        &error.details.get("feature_id").and_then(Value::as_str),
        &Some("modifier.apply.shape_key_guard"),
        "feature id",
    )
}

#[test]
fn apply_as_shape_key_removes_only_the_target() -> TestResult<()> {
    let array = modifier(
        "array",
        "array",
        json!({"count": 3, "relative_offset_displace": [1.5, 0.0, 0.0], "use_relative_offset": true}),
        false,
        true,
    );
    let displace = modifier(
        "displace",
        "displace",
        json!({"strength": 0.5, "mid_level": 0.0}),
        true,
        true,
    );
    let bake = json!({
        "op": "modifier.apply_as_shape_key",
        "target": {"id": SUBJECT},
        "modifier_id": "displace",
        "id": "baked",
        "name": "Baked",
    });
    let before = node_modifiers(&build(&[array.clone(), displace.clone()])?)?;

    let applied = run(&build(&[array, displace.clone()])?, &bake)?;
    let alone = run(&build(&[displace])?, &bake)?;

    ensure_eq(
        &data_block(&applied)?.shape_keys,
        &data_block(&alone)?.shape_keys,
        "baked shape key",
    )?;
    let keys = data_block(&applied)?
        .shape_keys
        .as_ref()
        .ok_or("baked shape keys are missing")?;
    let baked = keys
        .keys
        .get(&Id::new("baked")?)
        .ok_or("baked shape key is missing")?;
    ensure_eq(&baked.positions.len(), &8, "baked shape-key vertices")?;
    ensure_eq(
        &ids_and_flags(&applied)?,
        &vec![("array".to_owned(), false, true)],
        "remaining flags",
    )?;
    ensure_eq(
        &node_modifiers(&applied)?,
        &without(&before, "displace"),
        "remaining modifiers",
    )
}

#[test]
fn apply_as_shape_key_single_user_preserves_linked_data() -> TestResult<()> {
    let wave = modifier(
        "wave",
        "wave",
        json!({"height": 0.8, "speed": 0.0}),
        true,
        true,
    );
    let doc = build(&[wave])?;
    let linked = run(
        &doc,
        &json!({"op": "node.duplicate", "target": {"id": SUBJECT}, "id": "linked", "mode": "linked"}),
    )?;
    let data_id = |doc: &SceneDoc, node_id: &str| -> TestResult<Id> {
        Ok(doc
            .nodes
            .get(&Id::new(node_id)?)
            .and_then(|node| node.data.clone())
            .ok_or("linked node data is missing")?)
    };
    ensure_eq(
        &data_id(&linked, SUBJECT)?,
        &data_id(&linked, "linked")?,
        "linked nodes share data before apply",
    )?;

    let applied = run(
        &linked,
        &json!({
            "op": "modifier.apply_as_shape_key",
            "target": {"id": SUBJECT},
            "modifier_id": "wave",
            "id": "baked",
            "scope": "single_user",
        }),
    )?;

    if data_id(&applied, SUBJECT)? == data_id(&applied, "linked")? {
        return Err("single-user shape-key apply retained shared mesh data".into());
    }
    ensure_eq(
        &data_block_for(&applied, SUBJECT)?.shape_keys.is_some(),
        &true,
        "subject shape keys",
    )?;
    ensure_eq(
        &data_block_for(&applied, "linked")?.shape_keys.is_none(),
        &true,
        "linked shape keys remain unchanged",
    )?;
    ensure_eq(
        &node_modifiers_for(&applied, SUBJECT)?.len(),
        &0,
        "applied subject modifier",
    )?;
    ensure_eq(
        &node_modifiers_for(&applied, "linked")?.len(),
        &1,
        "unselected linked modifier",
    )
}
