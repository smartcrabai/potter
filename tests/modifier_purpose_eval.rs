//! Purpose-resolved modifier evaluation through the public `Snapshot` API.
//!
//! Blender 5.2.2 reference (`/tmp/pr18l-fix/PurposeRender/probe.py`; factory cube of size 2 with a
//! Subdivision modifier of `levels=1`, `render_levels=2`): the viewport depsgraph and
//! `DAG_EVAL_VIEWPORT` OBJ export give 26 vertices, `DAG_EVAL_RENDER` gives 98. With
//! `show_render=False` the render export gives the base 8 vertices; with `show_viewport=False` the
//! viewport gives 8 and the render gives 98.

use std::{collections::BTreeMap, error::Error, fmt::Debug};

use glam::DVec3;
use potter_core::{
    error::{ErrorCode, PotError},
    eval::{EvaluationContext, EvaluationPurpose, Snapshot},
    model::{Id, SceneDoc},
    ops,
};
use serde_json::{Value, json};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const SCENE_ID: &str = "00000000-0000-4000-8000-000000000000";
const BASE_VERTICES: usize = 8;
const LEVEL_ONE_VERTICES: usize = 26;
const LEVEL_TWO_VERTICES: usize = 98;

fn ensure_eq<T: PartialEq + Debug>(actual: &T, expected: &T, what: &str) -> TestResult {
    if actual == expected {
        Ok(())
    } else {
        Err(format!("{what}: expected {expected:?}, got {actual:?}").into())
    }
}

fn apply(doc: &SceneDoc, operations: &[Value]) -> Result<SceneDoc, PotError> {
    let batch = json!({
        "schema_version": 1,
        "base_revision": doc.revision,
        "operations": operations,
    });
    Ok(ops::apply_batch(doc, &batch)?.doc)
}

/// A box named `cube` followed by `modifiers`, applied in order.
fn cube_with(modifiers: &[Value]) -> TestResult<SceneDoc> {
    let mut operations = vec![json!({
        "op": "node.create",
        "id": "cube",
        "kind": "box",
        "params": {"size": 2.0},
    })];
    operations.extend(modifiers.iter().cloned());
    Ok(apply(&SceneDoc::new(SCENE_ID.to_owned()), &operations)?)
}

fn subdivision(
    id: &str,
    levels: u64,
    render_levels: u64,
    show_viewport: bool,
    show_render: bool,
) -> Value {
    json!({
        "op": "modifier.create",
        "target": {"id": "cube"},
        "id": id,
        "type": "subdivision",
        "params": {"levels": levels, "render_levels": render_levels},
        "show_viewport": show_viewport,
        "show_render": show_render,
    })
}

fn purpose_context(purpose: EvaluationPurpose) -> EvaluationContext {
    EvaluationContext {
        purpose,
        ..EvaluationContext::default()
    }
}

fn cube_vertices(doc: &SceneDoc, purpose: EvaluationPurpose) -> TestResult<Vec<DVec3>> {
    let snapshot = Snapshot::evaluate(doc, &purpose_context(purpose))?;
    let mesh = snapshot
        .meshes
        .get(&Id::new("cube")?)
        .ok_or("evaluated cube mesh is missing")?;
    Ok(mesh.vertices.iter().map(|vertex| vertex.co).collect())
}

#[test]
fn render_levels_apply_only_to_render_purpose() -> TestResult {
    let doc = cube_with(&[subdivision("subdivide", 1, 2, true, true)])?;
    let preview = cube_vertices(&doc, EvaluationPurpose::Preview)?;
    let render = cube_vertices(&doc, EvaluationPurpose::Render)?;
    ensure_eq(&preview.len(), &LEVEL_ONE_VERTICES, "preview vertex count")?;
    ensure_eq(&render.len(), &LEVEL_TWO_VERTICES, "render vertex count")?;

    let level_two = cube_with(&[subdivision("subdivide", 2, 2, true, true)])?;
    ensure_eq(
        &render,
        &cube_vertices(&level_two, EvaluationPurpose::Preview)?,
        "render equals subdivision level 2",
    )?;
    Ok(())
}

#[test]
fn modifier_hidden_for_render_is_omitted_from_render_only() -> TestResult {
    let doc = cube_with(&[subdivision("viewport_only", 1, 2, true, false)])?;
    ensure_eq(
        &cube_vertices(&doc, EvaluationPurpose::Preview)?.len(),
        &LEVEL_ONE_VERTICES,
        "preview keeps the render-hidden modifier",
    )?;
    ensure_eq(
        &cube_vertices(&doc, EvaluationPurpose::Render)?.len(),
        &BASE_VERTICES,
        "render omits the render-hidden modifier",
    )?;
    Ok(())
}

#[test]
fn modifier_hidden_for_viewport_is_render_only() -> TestResult {
    let doc = cube_with(&[subdivision("render_only", 1, 2, false, true)])?;
    ensure_eq(
        &cube_vertices(&doc, EvaluationPurpose::Preview)?.len(),
        &BASE_VERTICES,
        "preview omits the viewport-hidden modifier",
    )?;
    ensure_eq(
        &cube_vertices(&doc, EvaluationPurpose::Render)?.len(),
        &LEVEL_TWO_VERTICES,
        "render applies the render-only modifier at render levels",
    )?;
    Ok(())
}

#[test]
fn failing_node_and_dependents_are_reported_for_both_purposes() -> TestResult {
    // `host` gets Boolean followed by Decimate, which is refused as an unsupported topology
    // combination. `follower` uses `host` as its Boolean operand, so it depends on the failure.
    let doc = apply(
        &SceneDoc::new(SCENE_ID.to_owned()),
        &[
            json!({"op": "node.create", "id": "host", "kind": "box", "params": {"size": 2.0}}),
            json!({"op": "node.create", "id": "cutter", "kind": "box", "params": {"size": 1.0}}),
            json!({
                "op": "modifier.create",
                "target": {"id": "host"},
                "id": "boolean",
                "type": "boolean",
                "params": {"object": "cutter", "operation": "DIFFERENCE", "solver": "EXACT"},
            }),
            json!({
                "op": "modifier.create",
                "target": {"id": "host"},
                "id": "decimate",
                "type": "decimate",
                "params": {"ratio": 0.6},
            }),
            json!({"op": "node.create", "id": "follower", "kind": "box", "params": {"size": 1.0}}),
            json!({
                "op": "modifier.create",
                "target": {"id": "follower"},
                "id": "follow_boolean",
                "type": "boolean",
                "params": {"object": "host", "operation": "UNION", "solver": "FLOAT"},
            }),
        ],
    )?;
    let host = Id::new("host")?;
    let follower = Id::new("follower")?;
    let cutter = Id::new("cutter")?;

    for purpose in [EvaluationPurpose::Preview, EvaluationPurpose::Render] {
        let (snapshot, node_errors): (Snapshot, BTreeMap<Id, PotError>) =
            Snapshot::evaluate_available_with_cache(&doc, &purpose_context(purpose), None)?;
        ensure_eq(&node_errors.len(), &2, "failing and dependent node count")?;

        let host_error = node_errors.get(&host).ok_or("host error is missing")?;
        ensure_eq(
            &host_error.code,
            &ErrorCode::UnsupportedFeature,
            "host error code",
        )?;
        ensure_eq(
            &host_error.details["feature_id"],
            &json!("modifier.boolean.downstream_topology"),
            "host feature id",
        )?;

        let follower_error = node_errors
            .get(&follower)
            .ok_or("follower error is missing")?;
        ensure_eq(
            &follower_error.code,
            &ErrorCode::UnsupportedFeature,
            "follower error code",
        )?;
        ensure_eq(
            &follower_error.details["dependency_node_id"],
            &json!("host"),
            "follower dependency",
        )?;

        ensure_eq(
            &snapshot.meshes.contains_key(&host),
            &false,
            "failed host mesh is withheld",
        )?;
        ensure_eq(
            &snapshot.meshes.contains_key(&follower),
            &false,
            "dependent mesh is withheld",
        )?;
        ensure_eq(
            &snapshot.meshes.contains_key(&cutter),
            &true,
            "independent mesh is kept",
        )?;
    }
    Ok(())
}
