#![expect(clippy::unwrap_used, reason = "integration test setup")]

use std::collections::BTreeMap;

use serde_json::{Value, json};

use potter_core::{
    error::ErrorCode,
    eval::{EvaluationContext, Snapshot},
    model::{Id, SceneDoc},
    ops::apply_batch,
};

fn apply(doc: &SceneDoc, operations: &Value) -> potter_core::error::Result<SceneDoc> {
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

#[test]
fn instanced_custom_normals_rotate_with_the_instance_transform() {
    let doc = apply(
        &SceneDoc::new("4ae99b7f-471e-4c30-95a1-d8bfce8d2576".to_owned()),
        &json!([
            {"op":"collection.create","id":"source"},
            {"op":"node.create","id":"box","kind":"box","collection":"source","params":{"size":1.0},"transform":{"rotation_deg":[0.0,0.0,90.0]}},
            {"op":"mesh.set_custom_normals","target":{"id":"box"},"elements":{"domain":"face","ids":["f0"]},"normals":{"f0":[[1.0,0.0,0.0],[1.0,0.0,0.0],[1.0,0.0,0.0],[1.0,0.0,0.0]]}},
            {"op":"collection.instance_create","id":"instance","collection":"source"}
        ]),
    )
    .unwrap();
    let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default()).unwrap();
    let mesh = &snapshot.meshes[&"instance".parse::<Id>().unwrap()];
    let custom = mesh.attributes["custom_normal"]["values"]
        .as_object()
        .unwrap();
    assert_eq!(
        custom.len(),
        1,
        "only the authored face carries a custom normal"
    );
    // A 90 degree turn about Z maps the authored +X normal to +Y.
    for corner in custom.values().next().unwrap().as_array().unwrap() {
        let [x, y, z] = [0, 1, 2].map(|axis| corner[axis].as_f64().unwrap());
        assert!(
            (x).abs() < 1.0e-12 && (y - 1.0).abs() < 1.0e-12 && z.abs() < 1.0e-12,
            "{corner:?}"
        );
    }
}

#[test]
fn collection_instance_keeps_each_members_material_slots() {
    let doc = apply(
        &SceneDoc::new("4ae99b7f-471e-4c30-95a1-d8bfce8d2576".to_owned()),
        &json!([
            {"op":"material.create","id":"jacket"},
            {"op":"material.create","id":"skin"},
            {"op":"collection.create","id":"source"},
            {"op":"node.create","id":"box_a","kind":"box","collection":"source","params":{"size":1.0}},
            {"op":"node.update","target":{"id":"box_a"},"set":{"materials":["jacket","skin"]}},
            {"op":"mesh.set_attribute","target":{"id":"box_a"},"elements":{"domain":"face","ids":["f1"]},"material_index":1},
            {"op":"uv.unwrap","target":{"id":"box_a"},"method":"smart"},
            {"op":"node.create","id":"box_b","kind":"box","collection":"source","params":{"size":1.0},"transform":{"translation":[2.0,0.0,0.0]}},
            {"op":"node.update","target":{"id":"box_b"},"set":{"materials":["skin"]}},
            {"op":"collection.instance_create","id":"instance","collection":"source","transform":{"translation":[4.0,0.0,0.0]}}
        ]),
    )
    .unwrap();
    let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default()).unwrap();
    let instance = Id::new("instance").unwrap();
    let jacket = Id::new("jacket").unwrap();
    let skin = Id::new("skin").unwrap();
    // Each member's slots follow it: box_a's jacket and skin, then box_b's skin.
    let bindings = &snapshot.slot_bindings[&instance];
    assert_eq!(
        bindings
            .iter()
            .map(|slot| slot.material.clone())
            .collect::<Vec<_>>(),
        vec![Some(jacket), Some(skin.clone()), Some(skin)]
    );
    assert_eq!(
        bindings
            .iter()
            .map(|slot| slot.uv.texcoord0_covered)
            .collect::<Vec<_>>(),
        vec![true, true, false]
    );
    let slots: Vec<u32> = snapshot.meshes[&instance]
        .faces
        .iter()
        .map(|face| face.material_index)
        .collect();
    assert_eq!(slots.iter().filter(|slot| **slot == 0).count(), 5);
    assert_eq!(slots.iter().filter(|slot| **slot == 1).count(), 1);
    assert_eq!(slots.iter().filter(|slot| **slot == 2).count(), 6);
}

#[test]
fn collection_instance_keeps_each_members_uvs_after_face_gaps() {
    let doc = apply(
        &SceneDoc::new("4ae99b7f-471e-4c30-95a1-d8bfce8d2576".to_owned()),
        &json!([
            {"op":"collection.create","id":"source"},
            {"op":"node.create","id":"box_a","kind":"box","collection":"source","params":{"size":1.0}},
            {"op":"mesh.delete","target":{"id":"box_a"},"elements":{"domain":"face","ids":["f0"]}},
            {"op":"uv.unwrap","target":{"id":"box_a"},"method":"smart"},
            {"op":"node.create","id":"box_b","kind":"box","collection":"source","params":{"size":1.0},"transform":{"translation":[2.0,0.0,0.0]}},
            {"op":"uv.unwrap","target":{"id":"box_b"},"method":"smart"},
            {"op":"collection.instance_create","id":"instance","collection":"source","transform":{"translation":[4.0,0.0,0.0]}}
        ]),
    )
    .unwrap();
    let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default()).unwrap();
    let mesh = &snapshot.meshes[&Id::new("instance").unwrap()];
    // The joined canonical store has one real row per surviving face and corner.
    assert!(!mesh.attributes.contains_key("uv_map"));
    assert_eq!(mesh.faces.len(), 11);
    let layers = mesh.attributes["uv_layers"]["layers"].as_array().unwrap();
    assert_eq!(layers.len(), 1);
    let rows = layers[0]["faces"].as_array().unwrap();
    assert_eq!(rows.len(), mesh.faces.len());
    let corners: usize = mesh.faces.iter().map(|face| face.vertices.len()).sum();
    assert_eq!(
        rows.iter()
            .map(|row| row["uv"].as_array().unwrap().len())
            .sum::<usize>(),
        corners
    );
    for (face, row) in mesh.faces.iter().zip(rows) {
        assert_eq!(row["face_id"], json!(face.id));
        assert_eq!(row["uv"].as_array().unwrap().len(), face.vertices.len());
    }
    // Both members have a UV value on every face, so both slots are covered.
    let bindings = &snapshot.slot_bindings[&Id::new("instance").unwrap()];
    assert_eq!(bindings.len(), 2);
    assert!(bindings.iter().all(|slot| slot.uv.texcoord0_covered));
}

#[test]
fn nested_collection_instances_carry_member_materials_upward() {
    let base = SceneDoc::new("4ae99b7f-471e-4c30-95a1-d8bfce8d2576".to_owned());
    let doc = apply(&base, &json!([
        {"op":"material.create","id":"jacket"},
        {"op":"material.create","id":"skin"},
        {"op":"collection.create","id":"source"},
        {"op":"node.create","id":"box","kind":"box","collection":"source","params":{}},
        {"op":"node.update","target":{"id":"box"},"set":{"materials":["jacket"]}},
        {"op":"collection.create","id":"wrapper"},
        {"op":"node.create","id":"side","kind":"box","collection":"wrapper","params":{}},
        {"op":"node.update","target":{"id":"side"},"set":{"materials":["skin"]}},
        {"op":"collection.instance_create","id":"inner","collection":"source"},
        {"op":"collection.link","collection":"wrapper","object":"inner"},
        {"op":"collection.instance_create","id":"outer","collection":"wrapper","transform":{"translation":[5.0,0.0,0.0]}}
    ]))
    .unwrap();
    let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default()).unwrap();
    let outer = Id::new("outer").unwrap();
    let table: Vec<Option<Id>> = snapshot.slot_bindings[&outer]
        .iter()
        .map(|slot| slot.material.clone())
        .collect();
    let jacket = Id::new("jacket").unwrap();
    let skin = Id::new("skin").unwrap();
    assert_eq!(table.len(), 2);
    assert!(table.contains(&Some(jacket)) && table.contains(&Some(skin)));
    // The inner instance's jacket faces and the side's skin faces each keep their material.
    let mut counts = BTreeMap::<Option<Id>, usize>::new();
    for face in &snapshot.meshes[&outer].faces {
        let slot = usize::try_from(face.material_index).unwrap();
        *counts.entry(table[slot].clone()).or_default() += 1;
    }
    assert_eq!(counts.values().sum::<usize>(), 12);
    assert!(counts.values().all(|count| *count == 6));
}
