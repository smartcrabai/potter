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

use std::{error::Error, fmt::Debug};

use potter_core::{
    error::{ErrorCode, PotError},
    model::{DataBlock, Id, Modifier, SceneDoc},
    ops,
};
use serde_json::{Value, json};

type TestResult<T> = Result<T, Box<dyn Error>>;

const SCENE_ID: &str = "00000000-0000-4000-8000-000000000000";
const SUBJECT: &str = "subject";

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
fn build(extra: &[Value]) -> TestResult<SceneDoc> {
    let mut operations =
        vec![json!({"op": "node.create", "id": SUBJECT, "kind": "box", "params": {"size": 2.0}})];
    operations.extend_from_slice(extra);
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

fn modifier(
    id: &str,
    modifier_type: &str,
    params: Value,
    show_viewport: bool,
    show_render: bool,
) -> Value {
    json!({
        "op": "modifier.create",
        "target": {"id": SUBJECT},
        "id": id,
        "type": modifier_type,
        "params": params,
        "show_viewport": show_viewport,
        "show_render": show_render,
    })
}

fn node_modifiers(doc: &SceneDoc) -> TestResult<Vec<Modifier>> {
    let node = doc
        .nodes
        .get(&Id::new(SUBJECT)?)
        .ok_or("subject node is missing")?;
    Ok(node.modifiers.clone())
}

fn data_block(doc: &SceneDoc) -> TestResult<&DataBlock> {
    let node = doc
        .nodes
        .get(&Id::new(SUBJECT)?)
        .ok_or("subject node is missing")?;
    let data_id = node.data.as_ref().ok_or("subject node has no geometry data")?;
    doc.data_blocks
        .get(data_id)
        .ok_or_else(|| "subject data block is missing".into())
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
    let subdivision = modifier("subdivision", "subdivision", json!({"levels": 1}), true, true);
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
    let subdivision = modifier("subdivision", "subdivision", json!({"levels": 1}), true, true);
    let displace = modifier(
        "displace",
        "displace",
        json!({"strength": 0.5, "mid_level": 0.0}),
        false,
        true,
    );
    let solidify = modifier("solidify", "solidify", json!({"thickness": 0.1}), true, false);
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
    let subdivision = modifier("subdivision", "subdivision", json!({"levels": 1}), false, true);
    let solidify = modifier("solidify", "solidify", json!({"thickness": 0.1}), false, false);
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
    let subdivision = modifier("subdivision", "subdivision", json!({"levels": 1}), true, true);

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
    let subdivision = modifier("subdivision", "subdivision", json!({"levels": 1}), true, true);
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
