//! Baked modifier apply on a node with an OBJECT-linked slot and DATA slots.
//!
//! Blender `modifier_apply_obdata` (editors/object/object_modifier.cc, v5.2.2) calls
//! `BKE_object_material_from_eval_data` (blenkernel/intern/material.cc:1353), which sets the mesh's
//! material array to the evaluated mesh's array and then `BKE_object_materials_sync_length`
//! (material.cc:1039) resizes the object's slots. A slot the resize adds has matbits 0, so it is
//! DATA-linked. An existing object override keeps its link and value. Slot lookup follows
//! `BKE_object_material_get_eval` (material.cc:772): an object slot whose matbit is set and holds a
//! material wins; otherwise the DATA slot is used.

use std::error::Error;

use potter_core::{
    eval::{EvaluationContext, Snapshot},
    model::{Id, MaterialLink, SceneDoc},
    ops,
};
use serde_json::{Value, json};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const SCENE_ID: &str = "00000000-0000-4000-8000-000000000000";

fn id(text: &str) -> TestResult<Id> {
    Ok(Id::new(text.to_owned())?)
}

fn apply_operations(doc: &SceneDoc, operations: &[Value]) -> TestResult<SceneDoc> {
    let batch = json!({
        "schema_version": 1,
        "base_revision": doc.revision,
        "operations": operations,
    });
    Ok(ops::apply_batch(doc, &batch)?.doc)
}

/// Material ID resolved for one slot of `node_id` through the public slot API.
fn slot_material(
    snapshot: &Snapshot,
    doc: &SceneDoc,
    node_id: &Id,
    slot: usize,
) -> TestResult<Option<String>> {
    let node = doc.nodes.get(node_id).ok_or("node is missing")?;
    let data = node
        .data
        .as_ref()
        .and_then(|data_id| doc.data_blocks.get(data_id));
    Ok(snapshot
        .slot_material(node_id, node, data, slot)
        .map(|material| material.as_str().to_owned()))
}

#[test]
fn baked_apply_keeps_object_override_and_links_added_slots_to_data() -> TestResult {
    // Subject: DATA slots [A, B]; object slots [none, Object override O] with links [Data, Object].
    // The EXACT TRANSFER Boolean unions a cutter whose material is C. The pool is the subject's DATA
    // slots, then the operand's effective slot C (MOD_boolean.cc `non_float_boolean_mesh`).
    let doc = SceneDoc::new(SCENE_ID.to_owned());
    let mut doc = apply_operations(
        &doc,
        &[
            json!({"op":"material.create","id":"m_data_a","base_color":[0.9,0.05,0.02,1.0]}),
            json!({"op":"material.create","id":"m_data_b","base_color":[0.1,0.8,0.1,1.0]}),
            json!({"op":"material.create","id":"m_object","base_color":[0.02,0.1,0.9,1.0]}),
            json!({"op":"material.create","id":"m_cut","base_color":[0.9,0.9,0.1,1.0]}),
            json!({"op":"node.create","id":"subject","kind":"box","params":{"size":2.0}}),
            json!({"op":"node.create","id":"cutter","kind":"box","params":{"size":1.0},"transform":{"translation":[1.5,0.0,0.0]},"material":"m_cut"}),
            json!({"op":"modifier.create","target":{"id":"subject"},"id":"bool","type":"boolean","params":{"object":"cutter","operation":"UNION","solver":"EXACT","material_mode":"TRANSFER"}}),
        ],
    )?;
    let subject_id = id("subject")?;
    let data_id = doc
        .nodes
        .get(&subject_id)
        .and_then(|node| node.data.clone())
        .ok_or("subject has no geometry data")?;
    doc.data_blocks
        .get_mut(&data_id)
        .ok_or("subject data is missing")?
        .materials = vec![Some(id("m_data_a")?), Some(id("m_data_b")?)];
    {
        let subject = doc.nodes.get_mut(&subject_id).ok_or("subject is missing")?;
        subject.materials = vec![None, Some(id("m_object")?)];
        subject.material_links = vec![MaterialLink::Data, MaterialLink::Object];
    }

    let applied = apply_operations(
        &doc,
        &[json!({"op":"modifier.apply","target":{"id":"subject"},"modifier_id":"bool"})],
    )?;

    // DATA takes the evaluated array: the pool [A, B, C].
    let data = applied
        .data_blocks
        .get(&data_id)
        .ok_or("subject data is missing after apply")?;
    assert_eq!(
        data.materials,
        vec![
            Some(id("m_data_a")?),
            Some(id("m_data_b")?),
            Some(id("m_cut")?)
        ]
    );

    // The object keeps its override at slot 1 with its OBJECT link. The added slot 2 is DATA-linked.
    let node = applied
        .nodes
        .get(&subject_id)
        .ok_or("subject is missing after apply")?;
    assert!(node.modifiers.is_empty(), "the applied modifier is removed");
    assert_eq!(node.materials, vec![None, Some(id("m_object")?), None]);
    assert_eq!(
        node.material_links,
        vec![MaterialLink::Data, MaterialLink::Object, MaterialLink::Data]
    );

    // Slot resolution afterwards: slot 0 reads DATA A; slot 1 reads the object override O, since its
    // matbit is set and holds a material; slot 2 reads DATA C.
    let snapshot = Snapshot::evaluate(&applied, &EvaluationContext::default())?;
    assert_eq!(
        slot_material(&snapshot, &applied, &subject_id, 0)?.as_deref(),
        Some("m_data_a")
    );
    assert_eq!(
        slot_material(&snapshot, &applied, &subject_id, 1)?.as_deref(),
        Some("m_object")
    );
    assert_eq!(
        slot_material(&snapshot, &applied, &subject_id, 2)?.as_deref(),
        Some("m_cut")
    );
    Ok(())
}
