//! EXACT and FLOAT Boolean material and operand behaviour seen by collection instances, by Boolean
//! operands that carry their own TRANSFER pool, and by FLOAT's own-object guard, through the public
//! `Snapshot` API.

use std::{error::Error, fmt::Debug};

use potter_core::{
    eval::{EvaluationContext, Snapshot},
    model::{Id, SceneDoc},
    ops,
};
use serde_json::{Value, json};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const SCENE_ID: &str = "00000000-0000-4000-8000-000000000000";

fn ensure_eq<T: PartialEq + Debug>(actual: &T, expected: &T, what: &str) -> TestResult {
    if actual == expected {
        Ok(())
    } else {
        Err(format!("{what}: expected {expected:?}, got {actual:?}").into())
    }
}

fn apply(operations: &[Value]) -> TestResult<SceneDoc> {
    let doc = SceneDoc::new(SCENE_ID.to_owned());
    let batch = json!({
        "schema_version": 1,
        "base_revision": doc.revision,
        "operations": operations,
    });
    Ok(ops::apply_batch(&doc, &batch)?.doc)
}

/// Resolved material of every face of `id`'s evaluated mesh, in face order.
fn face_materials(
    snapshot: &Snapshot,
    doc: &SceneDoc,
    id: &str,
) -> TestResult<Vec<Option<String>>> {
    let node_id = Id::new(id.to_owned())?;
    let node = doc.nodes.get(&node_id).ok_or("node is missing")?;
    let mesh = snapshot
        .meshes
        .get(&node_id)
        .ok_or("evaluated mesh is missing")?;
    Ok(mesh
        .faces
        .iter()
        .map(|face| {
            snapshot
                .slot_material(&node_id, node, face.material_index as usize)
                .map(|material| material.as_str().to_owned())
        })
        .collect())
}

#[test]
fn collection_instance_resolves_transfer_boolean_operand_materials() -> TestResult {
    // `host` sits in `members` and takes its cutter's material through a TRANSFER EXACT union.
    // An instance of `members` must resolve each face to the same material as `host` itself.
    let doc = apply(&[
        json!({"op":"material.create","id":"m_host","base_color":[0.9,0.05,0.02,1.0]}),
        json!({"op":"material.create","id":"m_cut","base_color":[0.02,0.1,0.9,1.0]}),
        json!({"op":"collection.create","id":"members"}),
        json!({"op":"node.create","id":"host","kind":"box","collection":"members","params":{"size":2.0},"material":"m_host"}),
        json!({"op":"node.create","id":"cutter","kind":"box","params":{"size":1.0},"transform":{"translation":[1.5,0.0,0.0]},"material":"m_cut"}),
        json!({"op":"modifier.create","target":{"id":"host"},"id":"bool","type":"boolean","params":{"object":"cutter","operation":"UNION","solver":"EXACT","material_mode":"TRANSFER"}}),
        json!({"op":"collection.instance_create","id":"inst","collection":"members","transform":{"translation":[0.0,5.0,0.0]}}),
    ])?;
    let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;

    let host = face_materials(&snapshot, &doc, "host")?;
    let instance = face_materials(&snapshot, &doc, "inst")?;
    assert!(
        host.contains(&Some("m_cut".to_owned())),
        "the TRANSFER Boolean must bring the cutter material onto host: {host:?}"
    );
    let mut host_sorted = host;
    let mut instance_sorted = instance;
    host_sorted.sort();
    instance_sorted.sort();
    ensure_eq(&instance_sorted, &host_sorted, "instance face materials")?;
    Ok(())
}

#[test]
fn boolean_operand_with_transfer_pool_keeps_its_evaluated_materials() -> TestResult {
    // `inner` is a TRANSFER union whose cutter's material is pooled after its own. `outer` cuts
    // with `inner` as its operand in TRANSFER mode, so the cutter face material must come from
    // `inner`'s evaluated pool, not from its document slots.
    let doc = apply(&[
        json!({"op":"material.create","id":"m_a","base_color":[0.9,0.05,0.02,1.0]}),
        json!({"op":"material.create","id":"m_c","base_color":[0.02,0.1,0.9,1.0]}),
        json!({"op":"material.create","id":"m_d","base_color":[0.1,0.9,0.1,1.0]}),
        json!({"op":"node.create","id":"inner","kind":"box","params":{"size":2.0},"material":"m_a"}),
        json!({"op":"node.create","id":"cutter","kind":"box","params":{"size":1.0},"transform":{"translation":[1.5,0.0,0.0]},"material":"m_c"}),
        json!({"op":"modifier.create","target":{"id":"inner"},"id":"inner_bool","type":"boolean","params":{"object":"cutter","operation":"UNION","solver":"EXACT","material_mode":"TRANSFER"}}),
        json!({"op":"node.create","id":"outer","kind":"box","params":{"size":3.0},"transform":{"translation":[0.0,0.0,4.0]},"material":"m_d"}),
        json!({"op":"modifier.create","target":{"id":"outer"},"id":"outer_bool","type":"boolean","params":{"object":"inner","operation":"DIFFERENCE","solver":"EXACT","material_mode":"TRANSFER"}}),
    ])?;
    let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;

    let inner = face_materials(&snapshot, &doc, "inner")?;
    assert!(
        inner.contains(&Some("m_c".to_owned())),
        "inner's TRANSFER pool must carry the cutter material: {inner:?}"
    );
    let outer = face_materials(&snapshot, &doc, "outer")?;
    assert!(
        outer.contains(&Some("m_c".to_owned())),
        "outer must take inner's evaluated cutter material: {outer:?}"
    );
    Ok(())
}

#[test]
fn float_boolean_skips_its_own_object_in_a_collection() -> TestResult {
    // FLOAT DIFFERENCE over a collection holding its own subject: Blender never takes the object
    // as its own operand, so the subject is returned unchanged rather than cut by its raw mesh.
    let doc = apply(&[
        json!({"op":"collection.create","id":"members"}),
        json!({"op":"node.create","id":"host","kind":"box","collection":"members","params":{"size":2.0}}),
        json!({"op":"modifier.create","target":{"id":"host"},"id":"bool","type":"boolean","params":{"collection":"members","operation":"DIFFERENCE","solver":"FLOAT"}}),
    ])?;
    let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
    let mesh = snapshot
        .meshes
        .get(&Id::new("host".to_owned())?)
        .ok_or("evaluated mesh is missing")?;
    ensure_eq(&mesh.vertices.len(), &8, "vertex count")?;
    ensure_eq(&mesh.faces.len(), &6, "face count")?;
    Ok(())
}

#[test]
fn float_boolean_with_itself_as_object_is_unchanged() -> TestResult {
    let doc = apply(&[
        json!({"op":"node.create","id":"host","kind":"box","params":{"size":2.0}}),
        json!({"op":"modifier.create","target":{"id":"host"},"id":"bool","type":"boolean","params":{"object":"host","operation":"DIFFERENCE","solver":"FLOAT"}}),
    ])?;
    let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
    let mesh = snapshot
        .meshes
        .get(&Id::new("host".to_owned())?)
        .ok_or("evaluated mesh is missing")?;
    ensure_eq(&mesh.vertices.len(), &8, "vertex count")?;
    ensure_eq(&mesh.faces.len(), &6, "face count")?;
    Ok(())
}
