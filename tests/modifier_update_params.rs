//! `modifier.update` merges `set.params` into the current params when the modifier type is unchanged,
//! as Blender's property setters do: keys not named by the update keep their values, and a JSON
//! `null` removes its key so the modifier uses that parameter's default.

use std::error::Error;

use potter_core::{
    model::{Id, SceneDoc},
    ops,
};
use serde_json::{Map, Value, json};

type TestResult<T> = Result<T, Box<dyn Error>>;

const SCENE_ID: &str = "00000000-0000-4000-8000-000000000000";
const OBJECT: &str = "cube";
const MODIFIER: &str = "subdiv";

fn apply_one(doc: &SceneDoc, operation: &Value) -> TestResult<SceneDoc> {
    let batch = json!({
        "schema_version": 1,
        "base_revision": doc.revision,
        "operations": [operation],
    });
    Ok(ops::apply_batch(doc, &batch)?.doc)
}

fn modifier_params(doc: &SceneDoc) -> TestResult<Map<String, Value>> {
    let node = doc
        .nodes
        .get(&Id::new(OBJECT.to_owned())?)
        .ok_or("object is missing")?;
    let modifier = node
        .modifiers
        .iter()
        .find(|modifier| modifier.id.as_str() == MODIFIER)
        .ok_or("modifier is missing")?;
    Ok(modifier.params.clone())
}

#[test]
fn partial_modifier_update_keeps_other_params_and_null_resets_one() -> TestResult<()> {
    let doc = apply_one(
        &SceneDoc::new(SCENE_ID.to_owned()),
        &json!({"op": "node.create", "id": OBJECT, "kind": "box", "params": {"size": 2.0}}),
    )?;
    let doc = apply_one(
        &doc,
        &json!({
            "op": "modifier.create",
            "target": {"id": OBJECT},
            "id": MODIFIER,
            "type": "subdivision",
            "params": {"levels": 2, "quality": 4},
        }),
    )?;

    // A partial update names only `quality`; `levels` must survive it.
    let doc = apply_one(
        &doc,
        &json!({
            "op": "modifier.update",
            "target": {"id": OBJECT},
            "id": MODIFIER,
            "set": {"params": {"quality": 6}},
        }),
    )?;
    let params = modifier_params(&doc)?;
    assert_eq!(params.get("levels"), Some(&json!(2)));
    assert_eq!(params.get("quality"), Some(&json!(6)));

    // A JSON null removes `quality`, restoring its default, and leaves `levels` untouched.
    let doc = apply_one(
        &doc,
        &json!({
            "op": "modifier.update",
            "target": {"id": OBJECT},
            "id": MODIFIER,
            "set": {"params": {"quality": null}},
        }),
    )?;
    let params = modifier_params(&doc)?;
    assert!(!params.contains_key("quality"), "null must remove the key");
    assert_eq!(params.get("levels"), Some(&json!(2)));
    Ok(())
}
