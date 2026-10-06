#![expect(clippy::unwrap_used, reason = "integration test setup")]
#![expect(
    clippy::float_cmp,
    reason = "asserting exact deterministic transform values"
)]

use serde_json::{Value, json};

use potter::{
    error::ErrorCode,
    eval::{EvaluationContext, Snapshot},
    model::{Id, SceneDoc},
    ops::apply_batch,
};

fn apply(doc: &SceneDoc, operations: &Value) -> potter::error::Result<SceneDoc> {
    let batch = json!({
        "schema_version": 1,
        "base_revision": doc.revision,
        "operations": operations,
    });
    apply_batch(doc, &batch).map(|outcome| outcome.doc)
}

fn collection_with_two_boxes() -> SceneDoc {
    let doc = SceneDoc::new("4ae99b7f-471e-4c30-95a1-d8bfce8d2576".to_owned());
    apply(&doc, &json!([
        {"op":"collection.create", "id":"source"},
        {"op":"node.create", "id":"box_a", "kind":"box", "collection":"source", "params":{}},
        {"op":"node.create", "id":"box_b", "kind":"box", "collection":"source", "params":{}, "transform":{"translation":[4.0,0.0,0.0]}}
    ]))
    .unwrap()
}

#[test]
fn collection_instance_duplicates_geometry_and_offsets_bounds_and_paths() {
    let doc = collection_with_two_boxes();
    let baseline = Snapshot::evaluate(&doc, &EvaluationContext::default()).unwrap();
    let original_triangles: usize = baseline
        .meshes
        .values()
        .map(|mesh| {
            mesh.faces
                .iter()
                .map(|face| face.vertices.len() - 2)
                .sum::<usize>()
        })
        .sum();
    assert_eq!(original_triangles, 24);

    let instanced_doc = apply(
        &doc,
        &json!([{
            "op":"collection.instance_create",
            "id":"instance",
            "collection":"source",
            "transform":{"translation":[10.0,0.0,0.0]}
        }]),
    )
    .unwrap();
    assert_eq!(
        instanced_doc.nodes[&"instance".parse().unwrap()].kind,
        "collection_instance"
    );
    let snapshot = Snapshot::evaluate(&instanced_doc, &EvaluationContext::default()).unwrap();
    let evaluated_triangles: usize = snapshot
        .meshes
        .values()
        .map(|mesh| {
            mesh.faces
                .iter()
                .map(|face| face.vertices.len() - 2)
                .sum::<usize>()
        })
        .sum();

    assert_eq!(evaluated_triangles, 2 * original_triangles);
    assert_eq!(
        snapshot.meshes[&"instance".parse().unwrap()].faces.len(),
        12
    );
    let instance = &snapshot.nodes[&"instance".parse().unwrap()];
    let bounds = instance.bounds.unwrap();
    assert_eq!(bounds.min.to_array(), [9.0, -1.0, -1.0]);
    assert_eq!(bounds.max.to_array(), [15.0, 1.0, 1.0]);
    let paths = &snapshot.instance_paths[&"instance".parse().unwrap()];
    assert_eq!(paths.len(), 2);
    assert!(
        paths
            .iter()
            .any(|path| path.last().is_some_and(|id| id.as_str() == "box_a"))
    );
    assert!(
        paths
            .iter()
            .any(|path| path.last().is_some_and(|id| id.as_str() == "box_b"))
    );
}

#[test]
fn collection_instance_rejects_missing_target_collection() {
    let doc = SceneDoc::new("4ae99b7f-471e-4c30-95a1-d8bfce8d2576".to_owned());
    let error = apply(
        &doc,
        &json!([{
            "op":"collection.instance_create",
            "id":"instance",
            "collection":"missing"
        }]),
    )
    .unwrap_err();
    assert_eq!(error.code, ErrorCode::TargetNotFound);
}
#[test]
fn collection_instance_cycles_are_rejected() {
    let doc = SceneDoc::new("4ae99b7f-471e-4c30-95a1-d8bfce8d2576".to_owned());
    let error = apply(
        &doc,
        &json!([
            {"op":"collection.create", "id":"collection_a"},
            {"op":"collection.create", "id":"collection_b"},
            {"op":"collection.instance_create", "id":"instance_a", "collection":"collection_b"},
            {"op":"collection.instance_create", "id":"instance_b", "collection":"collection_a"},
            {"op":"collection.link", "collection":"collection_a", "object":"instance_a"},
            {"op":"collection.link", "collection":"collection_b", "object":"instance_b"}
        ]),
    )
    .unwrap_err();
    assert_eq!(error.code, ErrorCode::InvalidOperation);

    let mut cyclic_doc = apply(
        &doc,
        &json!([
            {"op":"collection.create", "id":"collection_a"},
            {"op":"collection.create", "id":"collection_b"},
            {"op":"collection.instance_create", "id":"instance_a", "collection":"collection_b"},
            {"op":"collection.instance_create", "id":"instance_b", "collection":"collection_a"}
        ]),
    )
    .unwrap();
    cyclic_doc
        .collections
        .get_mut(&"collection_a".parse().unwrap())
        .unwrap()
        .objects
        .push("instance_a".parse().unwrap());
    cyclic_doc
        .collections
        .get_mut(&"collection_b".parse().unwrap())
        .unwrap()
        .objects
        .push("instance_b".parse().unwrap());
    let error = Snapshot::evaluate(&cyclic_doc, &EvaluationContext::default()).unwrap_err();
    assert_eq!(error.code, ErrorCode::EvaluationFailed);
}

#[test]
fn nested_collection_instances_keep_recursive_paths_and_transformed_bounds() {
    let base = SceneDoc::new("4ae99b7f-471e-4c30-95a1-d8bfce8d2576".to_owned());
    let doc = apply(&base, &json!([
        {"op":"collection.create","id":"source"},
        {"op":"node.create","id":"box","kind":"box","collection":"source","params":{}},
        {"op":"collection.create","id":"wrapper"},
        {"op":"collection.instance_create","id":"inner","collection":"source"},
        {"op":"collection.link","collection":"wrapper","object":"inner"},
        {"op":"collection.instance_create","id":"outer","collection":"wrapper","transform":{"translation":[5.0,0.0,0.0]}}
    ]))
    .unwrap();
    let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default()).unwrap();
    let outer = Id::new("outer").unwrap();
    let inner = Id::new("inner").unwrap();
    let box_id = Id::new("box").unwrap();
    assert_eq!(
        snapshot.instance_paths[&outer],
        vec![vec![outer.clone(), inner, box_id]]
    );
    let bounds = snapshot.nodes[&outer].bounds.unwrap();
    assert_eq!(bounds.min.to_array(), [4.0, -1.0, -1.0]);
    assert_eq!(bounds.max.to_array(), [6.0, 1.0, 1.0]);
}
