#![expect(clippy::unwrap_used, reason = "integration tests")]

use potter_core::{
    error::ErrorCode,
    eval::{EvaluationContext, Snapshot},
    model::{DataBlock, Id, SceneDoc},
    ops,
};
use serde_json::{Value, json};

fn apply(doc: &SceneDoc, operations: &[Value]) -> SceneDoc {
    ops::apply_batch(
        doc,
        &json!({"schema_version":1,"base_revision":doc.revision,"operations":operations}),
    )
    .unwrap()
    .doc
}

fn curve_with_nodes(nodes: &[Value], links: &[Value], inputs: &[Value]) -> SceneDoc {
    let mut doc = SceneDoc::new("c8a8b9d4-2cbf-4bc4-9b17-6297b72e7d22".to_owned());
    let mut operations = vec![json!({
        "op":"graph.create",
        "id":"geo",
        "name":"Geometry",
        "kind":"geometry"
    })];
    operations.extend(nodes.iter().cloned().map(|mut node| {
        node["op"] = json!("graph.node_add");
        node["graph"] = json!("geo");
        node
    }));
    operations.push(json!({
        "op":"graph.interface_update",
        "graph":"geo",
        "set":{
            "inputs":inputs,
            "outputs":[{"id":"Geometry","name":"Geometry","socket_type":"geometry"}]
        }
    }));
    operations.extend(links.iter().cloned().map(|mut link| {
        link["op"] = json!("graph.link");
        link["graph"] = json!("geo");
        link
    }));
    doc = apply(&doc, &operations);
    doc.data_blocks.insert(
        Id::new("curve_data").unwrap(),
        serde_json::from_value::<DataBlock>(json!({
            "type":"curve",
            "curve":{
                "resolution_u":1,
                "splines":[{"type":"poly","points":[
                    {"co":[0.0,0.0,0.0]},
                    {"co":[1.0,0.0,0.0]}
                ]}]
            }
        }))
        .unwrap(),
    );
    doc.nodes.insert(
        Id::new("curve_object").unwrap(),
        serde_json::from_value(json!({
            "name":"Curve",
            "kind":"curve",
            "data":"curve_data"
        }))
        .unwrap(),
    );
    apply(
        &doc,
        &[json!({
            "op":"modifier.create",
            "target":{"id":"curve_object"},
            "id":"nodes_modifier",
            "type":"nodes",
            "params":{"node_group":"geo"}
        })],
    )
}

#[test]
fn curve_geometry_nodes_refuses_linked_modifier_input_geometry() {
    let doc = curve_with_nodes(
        &[
            json!({"id":"input","type":"NodeGroupInput"}),
            json!({"id":"output","type":"NodeGroupOutput"}),
        ],
        &[json!({
            "from_node":"input",
            "from_socket":"Geometry",
            "to_node":"output",
            "to_socket":"Geometry"
        })],
        &[json!({"id":"Geometry","name":"Geometry","socket_type":"geometry"})],
    );

    let error = Snapshot::evaluate(&doc, &EvaluationContext::default()).unwrap_err();
    assert_eq!(error.code, ErrorCode::UnsupportedFeature);
    assert_eq!(
        error.details["feature_id"],
        "modifier.geometry_nodes.curve_input_geometry"
    );
}

#[test]
fn curve_geometry_nodes_refuses_implicit_modifier_input_geometry() {
    let output_passthrough =
        curve_with_nodes(&[json!({"id":"output","type":"NodeGroupOutput"})], &[], &[]);
    let error = Snapshot::evaluate(&output_passthrough, &EvaluationContext::default()).unwrap_err();
    assert_eq!(error.code, ErrorCode::UnsupportedFeature);
    assert_eq!(
        error.details["feature_id"],
        "modifier.geometry_nodes.curve_input_geometry"
    );

    let implicit_transform_input = curve_with_nodes(
        &[
            json!({"id":"transform","type":"GeometryNodeTransform"}),
            json!({"id":"output","type":"NodeGroupOutput"}),
        ],
        &[json!({
            "from_node":"transform",
            "from_socket":"Geometry",
            "to_node":"output",
            "to_socket":"Geometry"
        })],
        &[],
    );
    let error =
        Snapshot::evaluate(&implicit_transform_input, &EvaluationContext::default()).unwrap_err();
    assert_eq!(error.code, ErrorCode::UnsupportedFeature);
    assert_eq!(
        error.details["feature_id"],
        "modifier.geometry_nodes.curve_input_geometry"
    );
}

#[test]
fn curve_geometry_nodes_allows_groups_that_do_not_consume_modifier_input() {
    let doc = curve_with_nodes(
        &[
            json!({"id":"input","type":"NodeGroupInput"}),
            json!({"id":"cube","type":"GeometryNodeMeshCube"}),
            json!({"id":"output","type":"NodeGroupOutput"}),
        ],
        &[json!({
            "from_node":"cube",
            "from_socket":"Mesh",
            "to_node":"output",
            "to_socket":"Geometry"
        })],
        &[json!({"id":"Geometry","name":"Geometry","socket_type":"geometry"})],
    );

    let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default()).unwrap();
    let vertex_count = snapshot
        .meshes
        .get(&Id::new("curve_object").unwrap())
        .unwrap()
        .vertices
        .len();
    assert_ne!(vertex_count, 0);
}
