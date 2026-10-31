#![expect(clippy::unwrap_used, reason = "integration tests")]
#![expect(
    clippy::float_cmp,
    reason = "asserting exact deterministic geometry values"
)]

use potter_core::{
    error::ErrorCode,
    eval::{EvaluationContext, Snapshot},
    geom::primitive,
    graph::{self, GraphEvaluation},
    model::{DataBlock, Id, SceneDoc},
    ops,
};
use serde_json::{Map, Value, json};

fn assert_close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-12);
}

fn apply(doc: &SceneDoc, operations: &[Value]) -> SceneDoc {
    ops::apply_batch(
        doc,
        &json!({"schema_version": 1, "base_revision": doc.revision, "operations": operations}),
    )
    .unwrap()
    .doc
}

fn graph_with_nodes(nodes: Vec<Value>, links: Vec<Value>) -> SceneDoc {
    let mut doc = SceneDoc::new("c8a8b9d4-2cbf-4bc4-9b17-6297b72e7d22".to_owned());
    let mut operations =
        vec![json!({"op":"graph.create","id":"geo","name":"Geometry","kind":"geometry"})];
    operations.extend(nodes.into_iter().map(|node| {
        let mut operation = node;
        operation["op"] = json!("graph.node_add");
        operation["graph"] = json!("geo");
        operation
    }));
    operations.extend(links.into_iter().map(|link| {
        let mut operation = link;
        operation["op"] = json!("graph.link");
        operation["graph"] = json!("geo");
        operation
    }));
    doc = apply(&doc, &operations);
    doc
}

fn evaluate(doc: &SceneDoc) -> GraphEvaluation {
    graph::evaluate(
        doc.node_groups.get(&Id::new("geo").unwrap()).unwrap(),
        None,
        &Map::new(),
        "Geometry",
    )
    .unwrap()
}

fn evaluate_primitive_node(node_type: &str, inputs: Value) -> potter_core::geom::Mesh {
    let mut primitive = json!({"id": "primitive", "type": node_type});
    primitive["inputs"] = inputs;
    let doc = graph_with_nodes(
        vec![primitive, json!({"id":"output","type":"NodeGroupOutput"})],
        vec![
            json!({"from_node":"primitive","from_socket":"Mesh","to_node":"output","to_socket":"Geometry"}),
        ],
    );
    evaluate(&doc).mesh
}

#[test]
fn cube_transform_graph_evaluates_and_set_position_applies_offset_field() {
    let doc = graph_with_nodes(
        vec![
            json!({"id":"cube","type":"GeometryNodeMeshCube","properties":{"size":[2.0,2.0,2.0]}}),
            json!({"id":"transform","type":"GeometryNodeTransform","inputs":{"Scale":[2.0,2.0,2.0]}}),
            json!({"id":"output","type":"NodeGroupOutput"}),
        ],
        vec![
            json!({"from_node":"cube","from_socket":"Mesh","to_node":"transform","to_socket":"Geometry"}),
            json!({"from_node":"transform","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
        ],
    );
    let result = evaluate(&doc);
    let size = result.mesh.bounds().unwrap().size();
    assert_close(size.x, 4.0);
    assert_close(size.y, 4.0);
    assert_close(size.z, 4.0);

    let shifted = graph_with_nodes(
        vec![
            json!({"id":"cube","type":"GeometryNodeMeshCube"}),
            json!({"id":"position_input","type":"GeometryNodeInputPosition"}),
            json!({"id":"offset","type":"ShaderNodeVectorMath","properties":{"operation":"SCALE"},"inputs":{"Scale":0.5}}),
            json!({"id":"position","type":"GeometryNodeSetPosition"}),
            json!({"id":"output","type":"NodeGroupOutput"}),
        ],
        vec![
            json!({"from_node":"cube","from_socket":"Mesh","to_node":"position","to_socket":"Geometry"}),
            json!({"from_node":"position_input","from_socket":"Position","to_node":"offset","to_socket":"Vector"}),
            json!({"from_node":"offset","from_socket":"Vector","to_node":"position","to_socket":"Offset"}),
            json!({"from_node":"position","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
        ],
    );
    let bounds = evaluate(&shifted).mesh.bounds().unwrap();
    assert_close(bounds.min.x, -0.75);
    assert_close(bounds.max.x, 0.75);
}

#[test]
fn deep_mixed_geometry_chain_preserves_order_and_fields() {
    const STEPS: usize = 80;
    let mut nodes = vec![
        json!({"id":"line","type":"GeometryNodeMeshLine","inputs":{"Count":1}}),
        json!({"id":"output","type":"NodeGroupOutput"}),
    ];
    let mut links = Vec::new();
    let mut previous_node = "line".to_owned();
    let mut previous_socket = "Mesh".to_owned();
    for index in 0..STEPS {
        let transform_id = format!("transform_{index}");
        nodes.push(json!({
            "id": transform_id.clone(),
            "type": "GeometryNodeTransform",
            "inputs": {"Translation": [1.0, 0.0, 0.0]}
        }));
        links.push(json!({
            "from_node": previous_node.clone(),
            "from_socket": previous_socket.clone(),
            "to_node": transform_id.clone(),
            "to_socket": "Geometry"
        }));
        previous_node = transform_id;
        previous_socket = "Geometry".to_owned();

        let position_id = format!("position_{index}");
        nodes.push(json!({
            "id": position_id.clone(),
            "type": "GeometryNodeSetPosition",
            "inputs": {"Offset": [0.0, 1.0, 0.0]}
        }));
        links.push(json!({
            "from_node": previous_node.clone(),
            "from_socket": previous_socket.clone(),
            "to_node": position_id.clone(),
            "to_socket": "Geometry"
        }));
        previous_node = position_id;

        let store_id = format!("store_{index}");
        nodes.push(json!({
            "id": store_id.clone(),
            "type": "GeometryNodeStoreNamedAttribute",
            "properties": {"domain": "POINT", "data_type": "INT"},
            "inputs": {
                "Name": format!("mixed_step_{index}"),
                "Value": i64::try_from(index).unwrap()
            }
        }));
        links.push(json!({
            "from_node": previous_node.clone(),
            "from_socket": previous_socket.clone(),
            "to_node": store_id.clone(),
            "to_socket": "Geometry"
        }));
        previous_node = store_id;
    }
    links.push(json!({
        "from_node": previous_node,
        "from_socket": previous_socket,
        "to_node": "output",
        "to_socket": "Geometry"
    }));

    let evaluation = evaluate(&graph_with_nodes(nodes, links));
    let expected_translation = f64::from(u32::try_from(STEPS).unwrap());
    assert_eq!(evaluation.mesh.vertices.len(), 1);
    assert_eq!(evaluation.mesh.vertices[0].co.x, expected_translation);
    assert_eq!(evaluation.mesh.vertices[0].co.y, expected_translation);
    for index in [0, STEPS - 1] {
        let name = format!("mixed_step_{index}");
        let values = evaluation.mesh.attributes[&name]["data"]
            .as_array()
            .unwrap();
        assert_eq!(values.len(), 1);
        assert_eq!(values[0].as_i64().unwrap(), i64::try_from(index).unwrap());
    }
}

#[test]
fn deep_join_geometry_worklist_reuses_empty_branch() {
    const STEPS: usize = 256;
    let mut nodes = vec![
        json!({"id":"point","type":"GeometryNodeMeshLine","inputs":{"Count":1}}),
        json!({"id":"empty_source","type":"GeometryNodeMeshLine","inputs":{"Count":1}}),
        json!({"id":"empty_points","type":"GeometryNodeMeshToPoints","inputs":{"Selection":false}}),
        json!({"id":"output","type":"NodeGroupOutput"}),
    ];
    let mut links = Vec::new();
    links.push(json!({
        "from_node": "empty_source",
        "from_socket": "Mesh",
        "to_node": "empty_points",
        "to_socket": "Mesh"
    }));
    let mut previous_node = "point".to_owned();
    let mut previous_socket = "Mesh".to_owned();
    for index in 0..STEPS {
        let join_id = format!("join_{index}");
        nodes.push(json!({"id":join_id.clone(),"type":"GeometryNodeJoinGeometry"}));
        links.push(json!({
            "from_node": previous_node.clone(),
            "from_socket": previous_socket.clone(),
            "to_node": join_id.clone(),
            "to_socket": "Geometry"
        }));
        links.push(json!({
            "from_node": "empty_points",
            "from_socket": "Points",
            "to_node": join_id.clone(),
            "to_socket": "Geometry"
        }));
        previous_node = join_id;
        previous_socket = "Geometry".to_owned();
    }
    links.push(json!({
        "from_node": previous_node,
        "from_socket": previous_socket,
        "to_node": "output",
        "to_socket": "Geometry"
    }));

    let mesh = evaluate(&graph_with_nodes(nodes, links)).mesh;
    assert_eq!(mesh.vertices.len(), 1);
    assert_eq!(mesh.vertices[0].co.x, 0.0);
    assert_eq!(mesh.vertices[0].co.y, 0.0);
    assert_eq!(mesh.vertices[0].co.z, 0.0);
}

#[test]
fn graph_interface_defaults_accept_input_overrides() {
    let doc = apply(
        &SceneDoc::new("c8a8b9d4-2cbf-4bc4-9b17-6297b72e7d22".to_owned()),
        &[
            json!({"op":"graph.create","id":"geo","kind":"geometry"}),
            json!({"op":"graph.node_add","graph":"geo","id":"input","type":"NodeGroupInput"}),
            json!({"op":"graph.node_add","graph":"geo","id":"cube","type":"GeometryNodeMeshCube","properties":{"size":[1.0,1.0,1.0]}}),
            json!({"op":"graph.node_add","graph":"geo","id":"transform","type":"GeometryNodeTransform"}),
            json!({"op":"graph.node_add","graph":"geo","id":"output","type":"NodeGroupOutput"}),
            json!({"op":"graph.interface_update","graph":"geo","set":{"inputs":[{"id":"scale","name":"Scale","socket_type":"vector","default":[1.0,1.0,1.0]}],"outputs":[]}}),
            json!({"op":"graph.link","graph":"geo","from_node":"cube","from_socket":"Mesh","to_node":"transform","to_socket":"Geometry"}),
            json!({"op":"graph.link","graph":"geo","from_node":"input","from_socket":"Scale","to_node":"transform","to_socket":"Scale"}),
            json!({"op":"graph.link","graph":"geo","from_node":"transform","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
        ],
    );
    let group = doc.node_groups.get(&Id::new("geo").unwrap()).unwrap();
    let default = graph::evaluate(group, None, &Map::new(), "Geometry").unwrap();
    let overridden = graph::evaluate(
        group,
        None,
        &Map::from_iter([("scale".to_owned(), json!([2.0, 2.0, 2.0]))]),
        "Geometry",
    )
    .unwrap();
    assert_close(default.mesh.bounds().unwrap().size().x, 1.0);
    assert_close(overridden.mesh.bounds().unwrap().size().x, 2.0);
}

#[test]
fn seeded_distribution_is_repeatable_and_instances_are_realized_with_paths() {
    let doc = graph_with_nodes(
        vec![
            json!({"id":"surface","type":"GeometryNodeMeshGrid","properties":{"size_x":2.0,"size_y":2.0,"vertices_x":3,"vertices_y":3}}),
            json!({"id":"points","type":"GeometryNodeDistributePointsOnFaces","inputs":{"Density":2.0,"Seed":17}}),
            json!({"id":"cube","type":"GeometryNodeMeshCube","properties":{"size":[0.1,0.1,0.1]}}),
            json!({"id":"instances","type":"GeometryNodeInstanceOnPoints"}),
            json!({"id":"output","type":"NodeGroupOutput"}),
        ],
        vec![
            json!({"from_node":"surface","from_socket":"Mesh","to_node":"points","to_socket":"Mesh"}),
            json!({"from_node":"points","from_socket":"Points","to_node":"instances","to_socket":"Points"}),
            json!({"from_node":"cube","from_socket":"Mesh","to_node":"instances","to_socket":"Instance"}),
            json!({"from_node":"instances","from_socket":"Instances","to_node":"output","to_socket":"Geometry"}),
        ],
    );
    let first = evaluate(&doc);
    let second = evaluate(&doc);
    assert_eq!(first.mesh, second.mesh);
    assert!(
        !first.instance_paths.is_empty(),
        "point distribution produced no instances"
    );
    assert_eq!(first.instance_paths.len(), first.mesh.faces.len() / 6);
}

#[test]
fn graph_rejects_socket_type_mismatch_and_evaluation_rejects_cycles() {
    let mut doc = SceneDoc::new("c8a8b9d4-2cbf-4bc4-9b17-6297b72e7d22".to_owned());
    let mismatch = ops::apply_batch(
        &doc,
        &json!({"schema_version":1,"base_revision":0,"operations":[
            {"op":"graph.create","id":"geo","kind":"geometry"},
            {"op":"graph.node_add","graph":"geo","id":"cube","type":"GeometryNodeMeshCube"},
            {"op":"graph.node_add","graph":"geo","id":"transform","type":"GeometryNodeTransform"},
            {"op":"graph.link","graph":"geo","from_node":"cube","from_socket":"Mesh","to_node":"transform","to_socket":"Scale"}
        ]}),
    )
    .unwrap_err();
    assert_eq!(mismatch.code, ErrorCode::InvalidOperation);

    doc = graph_with_nodes(
        vec![
            json!({"id":"a","type":"GeometryNodeTransform"}),
            json!({"id":"b","type":"GeometryNodeTransform"}),
            json!({"id":"output","type":"NodeGroupOutput"}),
        ],
        vec![
            json!({"from_node":"a","from_socket":"Geometry","to_node":"b","to_socket":"Geometry"}),
            json!({"from_node":"b","from_socket":"Geometry","to_node":"a","to_socket":"Geometry"}),
            json!({"from_node":"a","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
        ],
    );
    let error = graph::evaluate(
        doc.node_groups.get(&Id::new("geo").unwrap()).unwrap(),
        None,
        &Map::new(),
        "Geometry",
    )
    .unwrap_err();
    assert_eq!(error.code, ErrorCode::EvaluationFailed);
}
#[test]
fn nested_group_io_is_lazy_and_uses_socket_identifier_defaults() {
    let doc = apply(
        &SceneDoc::new("c8a8b9d4-2cbf-4bc4-9b17-6297b72e7d22".to_owned()),
        &[
            json!({"op":"graph.create","id":"child","kind":"geometry"}),
            json!({"op":"graph.interface_update","graph":"child","set":{"inputs":[
                {"id":"geometry","name":"Geometry","socket_type":"geometry"},
                {"id":"unused","name":"Unused","socket_type":"float","default":"not-a-number"}
            ]}}),
            json!({"op":"graph.node_add","graph":"child","id":"input","type":"NodeGroupInput"}),
            json!({"op":"graph.node_add","graph":"child","id":"output","type":"NodeGroupOutput"}),
            json!({"op":"graph.link","graph":"child","from_node":"input","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
            json!({"op":"graph.create","id":"geo","kind":"geometry"}),
            json!({"op":"graph.node_add","graph":"geo","id":"line","type":"GeometryNodeMeshLine","inputs":{"Count":2}}),
            json!({"op":"graph.node_add","graph":"geo","id":"nested","type":"GeometryNodeGroup","properties":{"node_tree":"child"}}),
            json!({"op":"graph.node_add","graph":"geo","id":"output","type":"NodeGroupOutput"}),
            json!({"op":"graph.link","graph":"geo","from_node":"line","from_socket":"Mesh","to_node":"nested","to_socket":"geometry"}),
            json!({"op":"graph.link","graph":"geo","from_node":"nested","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
        ],
    );
    let result = graph::evaluate_with_node_groups(
        &Id::new("geo").unwrap(),
        &doc.node_groups,
        None,
        &Map::new(),
        "NestedGroup",
    )
    .unwrap();
    assert_eq!(result.mesh.vertices.len(), 2);
}

#[test]
fn group_callsite_name_cannot_shadow_another_socket_identifier() {
    let doc = apply(
        &SceneDoc::new("c8a8b9d4-2cbf-4bc4-9b17-6297b72e7d22".to_owned()),
        &[
            json!({"op":"graph.create","id":"child","kind":"geometry"}),
            json!({"op":"graph.interface_update","graph":"child","set":{
                "inputs":[
                    {"id":"Socket_0","name":"Geometry","socket_type":"geometry"},
                    {"id":"Socket_1","name":"Amount","socket_type":"vector","default":[0.0,0.0,0.0]},
                    {"id":"Socket_2","name":"Socket_1","socket_type":"vector","default":[0.0,2.0,0.0]}
                ],
                "outputs":[{"id":"Result","name":"Geometry","socket_type":"geometry"}]
            }}),
            json!({"op":"graph.node_add","graph":"child","id":"input","type":"NodeGroupInput"}),
            json!({"op":"graph.node_add","graph":"child","id":"offset","type":"ShaderNodeVectorMath","properties":{"operation":"ADD"}}),
            json!({"op":"graph.node_add","graph":"child","id":"position","type":"GeometryNodeSetPosition"}),
            json!({"op":"graph.node_add","graph":"child","id":"output","type":"NodeGroupOutput"}),
            json!({"op":"graph.link","graph":"child","from_node":"input","from_socket":"Socket_1","to_node":"offset","to_socket":"Vector"}),
            json!({"op":"graph.link","graph":"child","from_node":"input","from_socket":"Socket_2","to_node":"offset","to_socket":"Vector_001"}),
            json!({"op":"graph.link","graph":"child","from_node":"input","from_socket":"Socket_0","to_node":"position","to_socket":"Geometry"}),
            json!({"op":"graph.link","graph":"child","from_node":"offset","from_socket":"Vector","to_node":"position","to_socket":"Offset"}),
            json!({"op":"graph.link","graph":"child","from_node":"position","from_socket":"Geometry","to_node":"output","to_socket":"Result"}),
            json!({"op":"graph.create","id":"geo","kind":"geometry"}),
            json!({"op":"graph.node_add","graph":"geo","id":"line","type":"GeometryNodeMeshLine","inputs":{"Count":1}}),
            json!({"op":"graph.node_add","graph":"geo","id":"nested_override","type":"GeometryNodeGroup","properties":{"node_tree":"child"},"inputs":{"Socket_1":[3.0,0.0,0.0]}}),
            json!({"op":"graph.node_add","graph":"geo","id":"source","type":"ShaderNodeCombineXYZ","inputs":{"X":1.0}}),
            json!({"op":"graph.node_add","graph":"geo","id":"nested_link","type":"GeometryNodeGroup","properties":{"node_tree":"child"}}),
            json!({"op":"graph.node_add","graph":"geo","id":"position","type":"GeometryNodeInputPosition"}),
            json!({"op":"graph.node_add","graph":"geo","id":"store_override","type":"GeometryNodeStoreNamedAttribute","properties":{"domain":"POINT","data_type":"FLOAT_VECTOR"},"inputs":{"Name":"override_value"}}),
            json!({"op":"graph.node_add","graph":"geo","id":"store_link","type":"GeometryNodeStoreNamedAttribute","properties":{"domain":"POINT","data_type":"FLOAT_VECTOR"},"inputs":{"Name":"linked_value"}}),
            json!({"op":"graph.node_add","graph":"geo","id":"output","type":"NodeGroupOutput"}),
            json!({"op":"graph.link","graph":"geo","from_node":"line","from_socket":"Mesh","to_node":"nested_override","to_socket":"Socket_0"}),
            json!({"op":"graph.link","graph":"geo","from_node":"nested_override","from_socket":"Result","to_node":"store_override","to_socket":"Geometry"}),
            json!({"op":"graph.link","graph":"geo","from_node":"position","from_socket":"Position","to_node":"store_override","to_socket":"Value"}),
            json!({"op":"graph.link","graph":"geo","from_node":"store_override","from_socket":"Geometry","to_node":"nested_link","to_socket":"Socket_0"}),
            json!({"op":"graph.link","graph":"geo","from_node":"source","from_socket":"Vector","to_node":"nested_link","to_socket":"Socket_1"}),
            json!({"op":"graph.link","graph":"geo","from_node":"nested_link","from_socket":"Result","to_node":"store_link","to_socket":"Geometry"}),
            json!({"op":"graph.link","graph":"geo","from_node":"position","from_socket":"Position","to_node":"store_link","to_socket":"Value"}),
            json!({"op":"graph.link","graph":"geo","from_node":"store_link","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
        ],
    );
    let result = graph::evaluate_with_node_groups(
        &Id::new("geo").unwrap(),
        &doc.node_groups,
        None,
        &Map::new(),
        "GroupSocketIdentifierCollision",
    )
    .unwrap();
    let override_value = &result.mesh.attributes["override_value"]["data"][0];
    assert_close(override_value[0].as_f64().unwrap(), 3.0);
    assert_close(override_value[1].as_f64().unwrap(), 2.0);
    assert_close(override_value[2].as_f64().unwrap(), 0.0);
    let linked_value = &result.mesh.attributes["linked_value"]["data"][0];
    assert_close(linked_value[0].as_f64().unwrap(), 4.0);
    assert_close(linked_value[1].as_f64().unwrap(), 4.0);
    assert_close(linked_value[2].as_f64().unwrap(), 0.0);
}

#[test]
fn group_output_name_alias_cannot_shadow_another_socket_identifier() {
    let doc = apply(
        &SceneDoc::new("c8a8b9d4-2cbf-4bc4-9b17-6297b72e7d22".to_owned()),
        &[
            json!({"op":"graph.create","id":"child","kind":"geometry"}),
            json!({"op":"graph.interface_update","graph":"child","set":{
                "inputs":[{"id":"GeometryInput","name":"Geometry","socket_type":"geometry"}],
                "outputs":[
                    {"id":"GeometryOutput","name":"Geometry","socket_type":"geometry"},
                    {"id":"Out1","name":"Amount","socket_type":"vector","default":[1.0,0.0,0.0]},
                    {"id":"Amount","name":"Other","socket_type":"vector","default":[0.0,0.0,0.0]}
                ]
            }}),
            json!({"op":"graph.node_add","graph":"child","id":"input","type":"NodeGroupInput"}),
            json!({"op":"graph.node_add","graph":"child","id":"value","type":"ShaderNodeCombineXYZ","inputs":{"X":9.0}}),
            json!({"op":"graph.node_add","graph":"child","id":"output","type":"NodeGroupOutput"}),
            json!({"op":"graph.link","graph":"child","from_node":"input","from_socket":"GeometryInput","to_node":"output","to_socket":"GeometryOutput"}),
            json!({"op":"graph.link","graph":"child","from_node":"value","from_socket":"Vector","to_node":"output","to_socket":"Amount"}),
            json!({"op":"graph.create","id":"geo","kind":"geometry"}),
            json!({"op":"graph.node_add","graph":"geo","id":"line","type":"GeometryNodeMeshLine","inputs":{"Count":1}}),
            json!({"op":"graph.node_add","graph":"geo","id":"nested","type":"GeometryNodeGroup","properties":{"node_tree":"child"}}),
            json!({"op":"graph.node_add","graph":"geo","id":"store","type":"GeometryNodeStoreNamedAttribute","properties":{"domain":"POINT","data_type":"FLOAT_VECTOR"},"inputs":{"Name":"alias_value"}}),
            json!({"op":"graph.node_add","graph":"geo","id":"output","type":"NodeGroupOutput"}),
            json!({"op":"graph.link","graph":"geo","from_node":"line","from_socket":"Mesh","to_node":"nested","to_socket":"GeometryInput"}),
            json!({"op":"graph.link","graph":"geo","from_node":"nested","from_socket":"GeometryOutput","to_node":"store","to_socket":"Geometry"}),
            json!({"op":"graph.link","graph":"geo","from_node":"nested","from_socket":"Out1","to_node":"store","to_socket":"Value"}),
            json!({"op":"graph.link","graph":"geo","from_node":"store","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
        ],
    );
    let result = graph::evaluate_with_node_groups(
        &Id::new("geo").unwrap(),
        &doc.node_groups,
        None,
        &Map::new(),
        "GroupOutputSocketIdentifierCollision",
    )
    .unwrap();
    let stored = &result.mesh.attributes["alias_value"]["data"][0];
    assert_close(stored[0].as_f64().unwrap(), 1.0);
    assert_close(stored[1].as_f64().unwrap(), 0.0);
    assert_close(stored[2].as_f64().unwrap(), 0.0);
}
#[test]
fn recursive_geometry_group_calls_return_deterministic_errors() {
    let doc = apply(
        &SceneDoc::new("c8a8b9d4-2cbf-4bc4-9b17-6297b72e7d22".to_owned()),
        &[
            json!({"op":"graph.create","id":"geo","kind":"geometry"}),
            json!({"op":"graph.node_add","graph":"geo","id":"recursive","type":"GeometryNodeGroup","properties":{"node_tree":"geo"}}),
            json!({"op":"graph.node_add","graph":"geo","id":"output","type":"NodeGroupOutput"}),
            json!({"op":"graph.link","graph":"geo","from_node":"recursive","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
        ],
    );
    let group_id = Id::new("geo").unwrap();
    let first = graph::evaluate_with_node_groups(
        &group_id,
        &doc.node_groups,
        None,
        &Map::new(),
        "RecursiveGroup",
    )
    .unwrap_err();
    let second = graph::evaluate_with_node_groups(
        &group_id,
        &doc.node_groups,
        None,
        &Map::new(),
        "RecursiveGroup",
    )
    .unwrap_err();
    assert_eq!(first.code, ErrorCode::EvaluationFailed);
    assert!(first.message.contains("recursive cycle"));
    assert_eq!(format!("{first:?}"), format!("{second:?}"));
}

#[test]
fn nodes_modifier_evaluates_graph_and_updates_snapshot_bounds() {
    let mut doc = graph_with_nodes(
        vec![
            json!({"id":"cube","type":"GeometryNodeMeshCube","properties":{"size":[1.0,1.0,1.0]}}),
            json!({"id":"transform","type":"GeometryNodeTransform","inputs":{"Scale":[2.0,2.0,2.0]}}),
            json!({"id":"output","type":"NodeGroupOutput"}),
        ],
        vec![
            json!({"from_node":"cube","from_socket":"Mesh","to_node":"transform","to_socket":"Geometry"}),
            json!({"from_node":"transform","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
        ],
    );
    let mesh = primitive("box", &json!({"size":1.0})).unwrap();
    let data_id = Id::new("mesh_body").unwrap();
    doc.data_blocks.insert(
        data_id,
        serde_json::from_value::<DataBlock>(json!({"type":"mesh","mesh":mesh})).unwrap(),
    );
    doc.nodes.insert(
        Id::new("body").unwrap(),
        serde_json::from_value(json!({"name":"Body","kind":"mesh","data":"mesh_body"})).unwrap(),
    );
    doc = apply(
        &doc,
        &[json!({
            "op":"modifier.create",
            "target":{"id":"body"},
            "id":"nodes_modifier",
            "type":"nodes",
            "params":{"node_group":"geo"}
        })],
    );
    let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default()).unwrap();
    let dimensions = snapshot
        .nodes
        .get(&Id::new("body").unwrap())
        .unwrap()
        .dimensions
        .unwrap();
    assert_close(dimensions[0], 2.0);
    assert_close(dimensions[1], 2.0);
    assert_close(dimensions[2], 2.0);
    let size = snapshot
        .meshes
        .get(&Id::new("body").unwrap())
        .unwrap()
        .bounds()
        .unwrap()
        .size();
    assert_close(size.x, 2.0);
    assert_close(size.y, 2.0);
    assert_close(size.z, 2.0);
}

#[test]
fn graph_operations_update_interface_and_remove_links_with_nodes() {
    let doc = apply(
        &SceneDoc::new("c8a8b9d4-2cbf-4bc4-9b17-6297b72e7d22".to_owned()),
        &[
            json!({"op":"graph.create","id":"geo","kind":"geometry"}),
            json!({"op":"graph.node_add","graph":"geo","id":"input","type":"NodeGroupInput"}),
            json!({"op":"graph.node_add","graph":"geo","id":"math","type":"ShaderNodeMath"}),
            json!({"op":"graph.interface_update","graph":"geo","set":{"inputs":[{"id":"amount","name":"Amount","socket_type":"float","default":2.0}],"outputs":[]}}),
            json!({"op":"graph.link","graph":"geo","from_node":"input","from_socket":"Amount","to_node":"math","to_socket":"Value"}),
            json!({"op":"graph.unlink","graph":"geo","from_node":"input","from_socket":"Amount","to_node":"math","to_socket":"Value"}),
            json!({"op":"graph.node_update","graph":"geo","node_id":"input","set":{"name":"Input"}}),
            json!({"op":"graph.node_remove","graph":"geo","node_id":"input"}),
            json!({"op":"graph.delete","target":{"id":"geo"}}),
        ],
    );
    assert!(doc.node_groups.is_empty(), "{:?}", doc.node_groups);
}

#[test]
fn listed_primitive_nodes_match_geometry_nodes_defaults() {
    let cube = evaluate_primitive_node("GeometryNodeMeshCube", json!({}));
    assert_eq!(
        (cube.vertices.len(), cube.edges.len(), cube.faces.len()),
        (8, 12, 6)
    );
    let bounds = cube.bounds().unwrap();
    assert_close(bounds.min.x, -0.5);
    assert_close(bounds.min.y, -0.5);
    assert_close(bounds.min.z, -0.5);
    assert_close(bounds.max.x, 0.5);
    assert_close(bounds.max.y, 0.5);
    assert_close(bounds.max.z, 0.5);

    let grid = evaluate_primitive_node("GeometryNodeMeshGrid", json!({}));
    assert_eq!(
        (grid.vertices.len(), grid.edges.len(), grid.faces.len()),
        (9, 12, 4)
    );
    let bounds = grid.bounds().unwrap();
    assert_close(bounds.min.x, -0.5);
    assert_close(bounds.min.y, -0.5);
    assert_close(bounds.max.x, 0.5);
    assert_close(bounds.max.y, 0.5);

    let sphere = evaluate_primitive_node("GeometryNodeMeshUVSphere", json!({}));
    assert_eq!(
        (
            sphere.vertices.len(),
            sphere.edges.len(),
            sphere.faces.len()
        ),
        (482, 992, 512)
    );
    assert_close(sphere.vertices[1].co.x, 0.191_341_716_182_544_9);
    assert_close(sphere.vertices[1].co.y, 0.038_060_233_744_356_63);

    let cylinder = evaluate_primitive_node("GeometryNodeMeshCylinder", json!({}));
    assert_eq!(
        (
            cylinder.vertices.len(),
            cylinder.edges.len(),
            cylinder.faces.len()
        ),
        (64, 96, 34)
    );
    let bounds = cylinder.bounds().unwrap();
    assert_close(bounds.min.z, -1.0);
    assert_close(bounds.max.z, 1.0);

    let cone = evaluate_primitive_node("GeometryNodeMeshCone", json!({}));
    assert_eq!((cone.vertices.len(), cone.faces.len()), (33, 33));
    let bounds = cone.bounds().unwrap();
    assert_close(bounds.min.z, 0.0);
    assert_close(bounds.max.z, 2.0);

    let line = evaluate_primitive_node("GeometryNodeMeshLine", json!({}));
    assert_eq!(
        (line.vertices.len(), line.edges.len(), line.faces.len()),
        (10, 9, 0)
    );

    let circle = evaluate_primitive_node("GeometryNodeMeshCircle", json!({}));
    assert_eq!(
        (
            circle.vertices.len(),
            circle.edges.len(),
            circle.faces.len()
        ),
        (32, 32, 0)
    );
    assert_close(circle.vertices[0].co.x, 1.0);
    assert_close(circle.vertices[0].co.y, 0.0);

    let ico_sphere = evaluate_primitive_node("GeometryNodeMeshIcoSphere", json!({}));
    assert_eq!(
        (
            ico_sphere.vertices.len(),
            ico_sphere.edges.len(),
            ico_sphere.faces.len()
        ),
        (12, 30, 20)
    );
}

#[test]
fn cube_and_grid_nodes_use_their_socket_parameters() {
    let cube = evaluate_primitive_node(
        "GeometryNodeMeshCube",
        json!({
            "Size": [2.0, 4.0, 6.0],
            "Vertices X": 3,
            "Vertices Y": 2,
            "Vertices Z": 2
        }),
    );
    assert_eq!((cube.vertices.len(), cube.faces.len()), (12, 10));
    let bounds = cube.bounds().unwrap();
    assert_close(bounds.min.x, -1.0);
    assert_close(bounds.min.y, -2.0);
    assert_close(bounds.min.z, -3.0);
    assert_close(bounds.max.x, 1.0);
    assert_close(bounds.max.y, 2.0);
    assert_close(bounds.max.z, 3.0);

    let grid = evaluate_primitive_node(
        "GeometryNodeMeshGrid",
        json!({
            "Size X": 4.0,
            "Size Y": 2.0,
            "Vertices X": 2,
            "Vertices Y": 3
        }),
    );
    assert_eq!((grid.vertices.len(), grid.faces.len()), (6, 2));
    let bounds = grid.bounds().unwrap();
    assert_close(bounds.min.x, -2.0);
    assert_close(bounds.min.y, -1.0);
    assert_close(bounds.max.x, 2.0);
    assert_close(bounds.max.y, 1.0);
}

#[test]
fn geometry_node_primitive_dimensions_follow_blender_sign_and_zero_rules() {
    let cube = evaluate_primitive_node("GeometryNodeMeshCube", json!({"Size": [-2.0, 0.0, 3.0]}));
    let bounds = cube.bounds().unwrap();
    assert_close(bounds.min.x, -1.0);
    assert_close(bounds.max.x, 1.0);
    assert_close(bounds.min.y, 0.0);
    assert_close(bounds.max.y, 0.0);
    assert_close(bounds.min.z, -1.5);
    assert_close(bounds.max.z, 1.5);

    let grid = evaluate_primitive_node(
        "GeometryNodeMeshGrid",
        json!({"Size X": -4.0, "Size Y": 0.0}),
    );
    let bounds = grid.bounds().unwrap();
    assert_close(bounds.min.x, -2.0);
    assert_close(bounds.max.x, 2.0);
    assert_close(bounds.min.y, 0.0);
    assert_close(bounds.max.y, 0.0);

    let sphere = evaluate_primitive_node("GeometryNodeMeshUVSphere", json!({"Radius": -2.0}));
    let bounds = sphere.bounds().unwrap();
    assert_close(bounds.min.x, -2.0);
    assert_close(bounds.max.x, 2.0);
    assert_close(bounds.min.z, -2.0);
    assert_close(bounds.max.z, 2.0);

    let cylinder = evaluate_primitive_node(
        "GeometryNodeMeshCylinder",
        json!({"Radius": 0.0, "Depth": 2.0}),
    );
    assert_eq!((cylinder.vertices.len(), cylinder.edges.len()), (2, 1));
    assert_close(cylinder.bounds().unwrap().min.z, -1.0);
    assert_close(cylinder.bounds().unwrap().max.z, 1.0);

    let cone = evaluate_primitive_node(
        "GeometryNodeMeshCone",
        json!({"Radius Top": 0.0, "Radius Bottom": 0.0, "Depth": -2.0}),
    );
    assert_eq!((cone.vertices.len(), cone.edges.len()), (2, 1));
    assert_close(cone.bounds().unwrap().min.z, -2.0);
    assert_close(cone.bounds().unwrap().max.z, 0.0);

    let circle = evaluate_primitive_node("GeometryNodeMeshCircle", json!({"Radius": 0.0}));
    assert_eq!((circle.vertices.len(), circle.edges.len()), (32, 32));
    assert!(
        circle
            .vertices
            .iter()
            .all(|vertex| vertex.co == glam::DVec3::ZERO)
    );
}

#[test]
fn cylinder_and_cone_nodes_honor_side_and_fill_segments() {
    let cylinder = graph_with_nodes(
        vec![
            json!({"id":"primitive","type":"GeometryNodeMeshCylinder","inputs":{"Vertices":8,"Side Segments":2,"Fill Segments":2}}),
            json!({"id":"output","type":"NodeGroupOutput"}),
        ],
        vec![
            json!({"from_node":"primitive","from_socket":"Mesh","to_node":"output","to_socket":"Geometry"}),
        ],
    );
    let cylinder = evaluate(&cylinder).mesh;
    assert_eq!(cylinder.vertices.len(), 40);
    assert_eq!(cylinder.faces.len(), 34);

    let cone = graph_with_nodes(
        vec![
            json!({"id":"primitive","type":"GeometryNodeMeshCone","inputs":{"Vertices":8,"Side Segments":2,"Fill Segments":2}}),
            json!({"id":"output","type":"NodeGroupOutput"}),
        ],
        vec![
            json!({"from_node":"primitive","from_socket":"Mesh","to_node":"output","to_socket":"Geometry"}),
        ],
    );
    let cone = evaluate(&cone).mesh;
    assert_eq!(cone.vertices.len(), 25);
    assert_eq!(cone.faces.len(), 25);
}

#[test]
fn mesh_to_points_supports_point_edge_face_and_corner_domains() {
    for (mode, expected_count) in [
        ("VERTICES", 8),
        ("EDGES", 12),
        ("FACES", 6),
        ("CORNERS", 24),
    ] {
        let doc = graph_with_nodes(
            vec![
                json!({"id":"cube","type":"GeometryNodeMeshCube"}),
                json!({"id":"points","type":"GeometryNodeMeshToPoints","properties":{"mode":mode}}),
                json!({"id":"output","type":"NodeGroupOutput"}),
            ],
            vec![
                json!({"from_node":"cube","from_socket":"Mesh","to_node":"points","to_socket":"Mesh"}),
                json!({"from_node":"points","from_socket":"Points","to_node":"output","to_socket":"Geometry"}),
            ],
        );
        assert_eq!(evaluate(&doc).mesh.vertices.len(), expected_count, "{mode}");
    }
}
#[test]
fn joined_mesh_to_points_preserves_nonempty_radius_attributes_and_fields() {
    let doc = graph_with_nodes(
        vec![
            json!({"id":"line_a","type":"GeometryNodeMeshLine","inputs":{"Count":2}}),
            json!({"id":"line_b","type":"GeometryNodeMeshLine","inputs":{"Count":2}}),
            json!({"id":"points_a","type":"GeometryNodeMeshToPoints","properties":{"mode":"VERTICES"},"inputs":{"Radius":0.25}}),
            json!({"id":"points_b","type":"GeometryNodeMeshToPoints","properties":{"mode":"VERTICES"},"inputs":{"Radius":0.75}}),
            json!({"id":"join","type":"GeometryNodeJoinGeometry"}),
            json!({"id":"radius","type":"GeometryNodeInputNamedAttribute","properties":{"data_type":"FLOAT"},"inputs":{"Name":"radius"}}),
            json!({"id":"store","type":"GeometryNodeStoreNamedAttribute","properties":{"domain":"POINT","data_type":"FLOAT"},"inputs":{"Name":"joined_radius"}}),
            json!({"id":"output","type":"NodeGroupOutput"}),
        ],
        vec![
            json!({"from_node":"line_a","from_socket":"Mesh","to_node":"points_a","to_socket":"Mesh"}),
            json!({"from_node":"line_b","from_socket":"Mesh","to_node":"points_b","to_socket":"Mesh"}),
            json!({"from_node":"points_a","from_socket":"Points","to_node":"join","to_socket":"Geometry"}),
            json!({"from_node":"points_b","from_socket":"Points","to_node":"join","to_socket":"Geometry"}),
            json!({"from_node":"join","from_socket":"Geometry","to_node":"store","to_socket":"Geometry"}),
            json!({"from_node":"radius","from_socket":"Attribute","to_node":"store","to_socket":"Value"}),
            json!({"from_node":"store","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
        ],
    );
    let mesh = evaluate(&doc).mesh;
    let expected = json!([0.25, 0.25, 0.75, 0.75]);
    assert_eq!(mesh.attributes["radius"]["domain"], "POINT");
    assert_eq!(mesh.attributes["radius"]["data_type"], "FLOAT");
    assert_eq!(mesh.attributes["radius"]["data"], expected);
    assert_eq!(mesh.attributes["joined_radius"]["data"], expected);
}
#[test]
fn random_value_boolean_probability_drives_fields() {
    let doc = graph_with_nodes(
        vec![
            json!({"id":"line","type":"GeometryNodeMeshLine","inputs":{"Count":5}}),
            json!({"id":"random","type":"FunctionNodeRandomValue","properties":{"data_type":"BOOLEAN"},"inputs":{"Probability":1.0,"Seed":17}}),
            json!({"id":"position","type":"GeometryNodeSetPosition","inputs":{"Offset":[1.0,0.0,0.0]}}),
            json!({"id":"output","type":"NodeGroupOutput"}),
        ],
        vec![
            json!({"from_node":"line","from_socket":"Mesh","to_node":"position","to_socket":"Geometry"}),
            json!({"from_node":"random","from_socket":"Value","to_node":"position","to_socket":"Selection"}),
            json!({"from_node":"position","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
        ],
    );
    let mesh = evaluate(&doc).mesh;
    assert_eq!(mesh.vertices.len(), 5);
    assert!(mesh.vertices.iter().all(|vertex| vertex.co.x == 1.0));
}
#[test]
fn point_distribution_stores_boolean_attributes_from_mesh_domains() {
    let nodes = vec![
        json!({
            "id":"points",
            "type":"GeometryNodeDistributePointsOnFaces",
            "properties":{"distribute_method":"RANDOM"},
            "inputs":{"Density":10.0,"Seed":17}
        }),
        json!({"id":"point_attr","type":"GeometryNodeInputNamedAttribute","properties":{"data_type":"BOOLEAN"},"inputs":{"Name":"point_flag"}}),
        json!({"id":"corner_attr","type":"GeometryNodeInputNamedAttribute","properties":{"data_type":"BOOLEAN"},"inputs":{"Name":"corner_flag"}}),
        json!({"id":"face_attr","type":"GeometryNodeInputNamedAttribute","properties":{"data_type":"BOOLEAN"},"inputs":{"Name":"face_flag"}}),
        json!({"id":"store_point","type":"GeometryNodeStoreNamedAttribute","properties":{"domain":"POINT","data_type":"BOOLEAN"},"inputs":{"Name":"sampled_point_flag"}}),
        json!({"id":"store_corner","type":"GeometryNodeStoreNamedAttribute","properties":{"domain":"POINT","data_type":"BOOLEAN"},"inputs":{"Name":"sampled_corner_flag"}}),
        json!({"id":"store_face","type":"GeometryNodeStoreNamedAttribute","properties":{"domain":"POINT","data_type":"BOOLEAN"},"inputs":{"Name":"sampled_face_flag"}}),
        json!({"id":"output","type":"NodeGroupOutput"}),
    ];
    let links = vec![
        json!({"from_node":"points","from_socket":"Points","to_node":"store_point","to_socket":"Geometry"}),
        json!({"from_node":"point_attr","from_socket":"Attribute","to_node":"store_point","to_socket":"Value"}),
        json!({"from_node":"store_point","from_socket":"Geometry","to_node":"store_corner","to_socket":"Geometry"}),
        json!({"from_node":"corner_attr","from_socket":"Attribute","to_node":"store_corner","to_socket":"Value"}),
        json!({"from_node":"store_corner","from_socket":"Geometry","to_node":"store_face","to_socket":"Geometry"}),
        json!({"from_node":"face_attr","from_socket":"Attribute","to_node":"store_face","to_socket":"Value"}),
        json!({"from_node":"store_face","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
    ];
    let doc = graph_with_nodes(nodes, links);
    let mut input = evaluate_primitive_node("GeometryNodeMeshCube", json!({}));
    let point_flags = (0..input.vertices.len())
        .map(|index| json!(index % 2 == 0))
        .collect::<Vec<_>>();
    let corner_flags = (0..input
        .faces
        .iter()
        .map(|face| face.vertices.len())
        .sum::<usize>())
        .map(|index| json!(index % 2 == 0))
        .collect::<Vec<_>>();
    let face_flags = (0..input.faces.len())
        .map(|index| json!(index % 2 == 0))
        .collect::<Vec<_>>();
    input.attributes.insert(
        "point_flag".to_owned(),
        json!({"domain":"POINT","data_type":"BOOLEAN","data":point_flags}),
    );
    input.attributes.insert(
        "corner_flag".to_owned(),
        json!({"domain":"CORNER","data_type":"BOOLEAN","data":corner_flags}),
    );
    input.attributes.insert(
        "face_flag".to_owned(),
        json!({"domain":"FACE","data_type":"BOOLEAN","data":face_flags}),
    );
    let mesh = graph::evaluate(
        doc.node_groups.get(&Id::new("geo").unwrap()).unwrap(),
        Some(&input),
        &Map::new(),
        "DistributionAttributePropagation",
    )
    .unwrap()
    .mesh;
    for name in [
        "sampled_point_flag",
        "sampled_corner_flag",
        "sampled_face_flag",
    ] {
        let attribute = &mesh.attributes[name];
        assert_eq!(attribute["domain"], "POINT", "{name} domain");
        assert_eq!(attribute["data_type"], "BOOLEAN", "{name} type");
        let values = attribute["data"].as_array().unwrap();
        assert_eq!(values.len(), mesh.vertices.len(), "{name} length");
        assert!(
            values.iter().any(|value| value.as_bool() == Some(true)),
            "{name} true samples"
        );
        assert!(
            values.iter().any(|value| value.as_bool() == Some(false)),
            "{name} false samples"
        );
    }
}

#[test]
fn absent_math_family_inputs_use_declaration_defaults() {
    let math_doc = graph_with_nodes(
        vec![
            json!({"id":"line","type":"GeometryNodeMeshLine","inputs":{"Count":1}}),
            json!({"id":"math","type":"ShaderNodeMath","properties":{"operation":"ADD"}}),
            json!({"id":"combine","type":"ShaderNodeCombineXYZ"}),
            json!({"id":"position","type":"GeometryNodeSetPosition"}),
            json!({"id":"output","type":"NodeGroupOutput"}),
        ],
        vec![
            json!({"from_node":"line","from_socket":"Mesh","to_node":"position","to_socket":"Geometry"}),
            json!({"from_node":"math","from_socket":"Value","to_node":"combine","to_socket":"X"}),
            json!({"from_node":"combine","from_socket":"Vector","to_node":"position","to_socket":"Offset"}),
            json!({"from_node":"position","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
        ],
    );
    assert_close(evaluate(&math_doc).mesh.vertices[0].co.x, 1.0);

    let vector_doc = graph_with_nodes(
        vec![
            json!({"id":"line","type":"GeometryNodeMeshLine","inputs":{"Count":1}}),
            json!({"id":"vector","type":"ShaderNodeVectorMath","properties":{"operation":"MULTIPLY"},"inputs":{"Vector":[2.0,3.0,4.0]}}),
            json!({"id":"position","type":"GeometryNodeSetPosition"}),
            json!({"id":"output","type":"NodeGroupOutput"}),
        ],
        vec![
            json!({"from_node":"line","from_socket":"Mesh","to_node":"position","to_socket":"Geometry"}),
            json!({"from_node":"vector","from_socket":"Vector","to_node":"position","to_socket":"Offset"}),
            json!({"from_node":"position","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
        ],
    );
    assert_close(evaluate(&vector_doc).mesh.vertices[0].co.x, 0.0);

    let scale_doc = graph_with_nodes(
        vec![
            json!({"id":"line","type":"GeometryNodeMeshLine","inputs":{"Count":1}}),
            json!({"id":"vector","type":"ShaderNodeVectorMath","properties":{"operation":"SCALE"},"inputs":{"Vector":[2.0,0.0,0.0]}}),
            json!({"id":"position","type":"GeometryNodeSetPosition"}),
            json!({"id":"output","type":"NodeGroupOutput"}),
        ],
        vec![
            json!({"from_node":"line","from_socket":"Mesh","to_node":"position","to_socket":"Geometry"}),
            json!({"from_node":"vector","from_socket":"Vector","to_node":"position","to_socket":"Offset"}),
            json!({"from_node":"position","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
        ],
    );
    assert_close(evaluate(&scale_doc).mesh.vertices[0].co.x, 2.0);

    let integer_doc = graph_with_nodes(
        vec![
            json!({"id":"line","type":"GeometryNodeMeshLine","inputs":{"Count":1}}),
            json!({"id":"integer","type":"FunctionNodeIntegerMath","properties":{"operation":"MULTIPLY"},"inputs":{"Value":3}}),
            json!({"id":"store","type":"GeometryNodeStoreNamedAttribute","properties":{"domain":"POINT","data_type":"INT"},"inputs":{"Name":"default_integer"}}),
            json!({"id":"output","type":"NodeGroupOutput"}),
        ],
        vec![
            json!({"from_node":"line","from_socket":"Mesh","to_node":"store","to_socket":"Geometry"}),
            json!({"from_node":"integer","from_socket":"Value","to_node":"store","to_socket":"Value"}),
            json!({"from_node":"store","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
        ],
    );
    let integer_mesh = evaluate(&integer_doc).mesh;
    let integer_values = integer_mesh.attributes["default_integer"]["data"]
        .as_array()
        .unwrap();
    assert_eq!(integer_values.len(), 1);
    assert_eq!(integer_values[0].as_i64().unwrap(), 0);
}

#[test]
fn absent_float_color_store_and_capture_values_use_socket_default() {
    let store_doc = graph_with_nodes(
        vec![
            json!({"id":"line","type":"GeometryNodeMeshLine","inputs":{"Count":4}}),
            json!({"id":"store","type":"GeometryNodeStoreNamedAttribute","properties":{"domain":"POINT","data_type":"FLOAT_COLOR"},"inputs":{"Name":"default_color"}}),
            json!({"id":"output","type":"NodeGroupOutput"}),
        ],
        vec![
            json!({"from_node":"line","from_socket":"Mesh","to_node":"store","to_socket":"Geometry"}),
            json!({"from_node":"store","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
        ],
    );
    let stored = evaluate(&store_doc).mesh;
    let stored_colors = stored.attributes["default_color"]["data"]
        .as_array()
        .unwrap();
    let expected = [0.8_f32, 0.8, 0.8, 1.0].map(f64::from);
    assert_eq!(stored_colors.len(), 4);
    for color in stored_colors {
        for (channel, expected) in color.as_array().unwrap().iter().zip(expected) {
            assert_close(channel.as_f64().unwrap(), expected);
        }
    }

    let capture_doc = graph_with_nodes(
        vec![
            json!({"id":"line","type":"GeometryNodeMeshLine","inputs":{"Count":4}}),
            json!({"id":"capture","type":"GeometryNodeCaptureAttribute","properties":{"domain":"POINT","data_type":"FLOAT_COLOR"}}),
            json!({"id":"store","type":"GeometryNodeStoreNamedAttribute","properties":{"domain":"POINT","data_type":"FLOAT_COLOR"},"inputs":{"Name":"captured_color"}}),
            json!({"id":"output","type":"NodeGroupOutput"}),
        ],
        vec![
            json!({"from_node":"line","from_socket":"Mesh","to_node":"capture","to_socket":"Geometry"}),
            json!({"from_node":"capture","from_socket":"Geometry","to_node":"store","to_socket":"Geometry"}),
            json!({"from_node":"capture","from_socket":"Attribute","to_node":"store","to_socket":"Value"}),
            json!({"from_node":"store","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
        ],
    );
    let captured = evaluate(&capture_doc).mesh;
    let captured_colors = captured.attributes["captured_color"]["data"]
        .as_array()
        .unwrap();
    assert_eq!(captured_colors.len(), 4);
    for color in captured_colors {
        for (channel, expected) in color.as_array().unwrap().iter().zip(expected) {
            assert_close(channel.as_f64().unwrap(), expected);
        }
    }

    let unselected_doc = graph_with_nodes(
        vec![
            json!({"id":"line","type":"GeometryNodeMeshLine","inputs":{"Count":1}}),
            json!({"id":"store","type":"GeometryNodeStoreNamedAttribute","properties":{"domain":"POINT","data_type":"FLOAT_COLOR"},"inputs":{"Name":"unselected_color","Selection":false}}),
            json!({"id":"output","type":"NodeGroupOutput"}),
        ],
        vec![
            json!({"from_node":"line","from_socket":"Mesh","to_node":"store","to_socket":"Geometry"}),
            json!({"from_node":"store","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
        ],
    );
    assert_eq!(
        evaluate(&unselected_doc).mesh.attributes["unselected_color"]["data"],
        json!([[0.0, 0.0, 0.0, 0.0]])
    );
}

#[test]
fn quaternion_store_preserves_finite_values_verbatim() {
    let doc = graph_with_nodes(
        vec![
            json!({"id":"line","type":"GeometryNodeMeshLine","inputs":{"Count":2}}),
            json!({"id":"non_unit","type":"GeometryNodeStoreNamedAttribute","properties":{"domain":"POINT","data_type":"QUATERNION"},"inputs":{"Name":"non_unit","Value":[0.0,0.0,0.0,2.0]}}),
            json!({"id":"zero","type":"GeometryNodeStoreNamedAttribute","properties":{"domain":"POINT","data_type":"QUATERNION"},"inputs":{"Name":"zero","Value":[0.0,0.0,0.0,0.0]}}),
            json!({"id":"output","type":"NodeGroupOutput"}),
        ],
        vec![
            json!({"from_node":"line","from_socket":"Mesh","to_node":"non_unit","to_socket":"Geometry"}),
            json!({"from_node":"non_unit","from_socket":"Geometry","to_node":"zero","to_socket":"Geometry"}),
            json!({"from_node":"zero","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
        ],
    );
    let mesh = evaluate(&doc).mesh;
    assert_eq!(
        mesh.attributes["non_unit"]["data"],
        json!([[0.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 2.0]])
    );
    assert_eq!(
        mesh.attributes["zero"]["data"],
        json!([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    );
}

#[test]
fn math_vector_math_and_xyz_nodes_evaluate_as_fields() {
    let math_doc = graph_with_nodes(
        vec![
            json!({"id":"line","type":"GeometryNodeMeshLine","inputs":{"Count":1}}),
            json!({"id":"math","type":"ShaderNodeMath","properties":{"operation":"MULTIPLY_ADD"},"inputs":{"Value":2.0,"Value_001":3.0,"Value_002":4.0}}),
            json!({"id":"combine","type":"ShaderNodeCombineXYZ"}),
            json!({"id":"position","type":"GeometryNodeSetPosition"}),
            json!({"id":"output","type":"NodeGroupOutput"}),
        ],
        vec![
            json!({"from_node":"line","from_socket":"Mesh","to_node":"position","to_socket":"Geometry"}),
            json!({"from_node":"math","from_socket":"Value","to_node":"combine","to_socket":"X"}),
            json!({"from_node":"combine","from_socket":"Vector","to_node":"position","to_socket":"Offset"}),
            json!({"from_node":"position","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
        ],
    );
    assert_close(evaluate(&math_doc).mesh.vertices[0].co.x, 10.0);

    let vector_doc = graph_with_nodes(
        vec![
            json!({"id":"line","type":"GeometryNodeMeshLine","inputs":{"Count":1}}),
            json!({"id":"vector","type":"ShaderNodeVectorMath","properties":{"operation":"CROSS_PRODUCT"},"inputs":{"Vector":[1.0,0.0,0.0],"Vector_001":[0.0,2.0,0.0]}}),
            json!({"id":"separate","type":"ShaderNodeSeparateXYZ"}),
            json!({"id":"combine","type":"ShaderNodeCombineXYZ"}),
            json!({"id":"position","type":"GeometryNodeSetPosition"}),
            json!({"id":"output","type":"NodeGroupOutput"}),
        ],
        vec![
            json!({"from_node":"line","from_socket":"Mesh","to_node":"position","to_socket":"Geometry"}),
            json!({"from_node":"vector","from_socket":"Vector","to_node":"separate","to_socket":"Vector"}),
            json!({"from_node":"separate","from_socket":"Z","to_node":"combine","to_socket":"X"}),
            json!({"from_node":"combine","from_socket":"Vector","to_node":"position","to_socket":"Offset"}),
            json!({"from_node":"position","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
        ],
    );
    assert_close(evaluate(&vector_doc).mesh.vertices[0].co.x, 2.0);
}

#[test]
fn repeat_zone_runs_body_for_each_iteration() {
    let doc = graph_with_nodes(
        vec![
            json!({"id":"cube","type":"GeometryNodeMeshCube","properties":{"size":[1.0,1.0,1.0]}}),
            json!({"id":"repeat_input","type":"GeometryNodeRepeatInput","inputs":{"Iterations":3}}),
            json!({"id":"translate","type":"GeometryNodeTransform","inputs":{"Translation":[1.0,0.0,0.0]}}),
            json!({"id":"repeat_output","type":"GeometryNodeRepeatOutput"}),
            json!({"id":"output","type":"NodeGroupOutput"}),
        ],
        vec![
            json!({"from_node":"cube","from_socket":"Mesh","to_node":"repeat_input","to_socket":"Geometry"}),
            json!({"from_node":"repeat_input","from_socket":"Geometry","to_node":"translate","to_socket":"Geometry"}),
            json!({"from_node":"translate","from_socket":"Geometry","to_node":"repeat_output","to_socket":"Geometry"}),
            json!({"from_node":"repeat_output","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
        ],
    );
    let bounds = evaluate(&doc).mesh.bounds().unwrap();
    assert_close(bounds.min.x, 2.5);
    assert_close(bounds.max.x, 3.5);
}

#[test]
fn distribute_points_minimum_distance_is_repeatable_and_enforced() {
    let doc = graph_with_nodes(
        vec![
            json!({"id":"surface","type":"GeometryNodeMeshGrid","properties":{"size_x":4.0,"size_y":4.0,"vertices_x":8,"vertices_y":8}}),
            json!({"id":"points","type":"GeometryNodeDistributePointsOnFaces","properties":{"distribute_method":"POISSON"},"inputs":{"Density":3.0,"Density Max":3.0,"Density Factor":1.0,"Distance Min":0.6,"Seed":73}}),
            json!({"id":"output","type":"NodeGroupOutput"}),
        ],
        vec![
            json!({"from_node":"surface","from_socket":"Mesh","to_node":"points","to_socket":"Mesh"}),
            json!({"from_node":"points","from_socket":"Points","to_node":"output","to_socket":"Geometry"}),
        ],
    );
    let first = evaluate(&doc).mesh;
    let second = evaluate(&doc).mesh;
    assert_eq!(first, second);
    assert!(
        !first.vertices.is_empty(),
        "point distribution produced no vertices"
    );
    for (index, point) in first.vertices.iter().enumerate() {
        for other in &first.vertices[index + 1..] {
            assert!(point.co.distance(other.co) >= 0.6);
        }
    }
}

#[test]
fn simulation_zone_evaluates_and_caches_each_frame_state() {
    let mut doc = graph_with_nodes(
        vec![
            json!({"id":"cube","type":"GeometryNodeMeshCube","properties":{"size":[1.0,1.0,1.0]}}),
            json!({"id":"simulation_input","type":"GeometryNodeSimulationInput"}),
            json!({"id":"translate","type":"GeometryNodeTransform","inputs":{"Translation":[0.0,0.0,1.0]}}),
            json!({"id":"simulation_output","type":"GeometryNodeSimulationOutput"}),
            json!({"id":"output","type":"NodeGroupOutput"}),
        ],
        vec![
            json!({"from_node":"cube","from_socket":"Mesh","to_node":"simulation_input","to_socket":"Geometry"}),
            json!({"from_node":"simulation_input","from_socket":"Geometry","to_node":"translate","to_socket":"Geometry"}),
            json!({"from_node":"translate","from_socket":"Geometry","to_node":"simulation_output","to_socket":"Geometry"}),
            json!({"from_node":"simulation_output","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
        ],
    );
    let group = doc.node_groups.get(&Id::new("geo").unwrap()).unwrap();
    let frame_three = graph::evaluate_at_frame(group, None, &Map::new(), 3.0, "Geometry").unwrap();
    let frame_two = graph::evaluate_at_frame(group, None, &Map::new(), 2.0, "Geometry").unwrap();
    assert_close(frame_three.mesh.bounds().unwrap().min.z, 2.5);
    assert_close(frame_two.mesh.bounds().unwrap().min.z, 1.5);
    let group_id = Id::new("geo").unwrap();
    let translate_id = Id::new("translate").unwrap();
    doc.node_groups
        .get_mut(&group_id)
        .unwrap()
        .nodes
        .get_mut(&translate_id)
        .unwrap()
        .inputs
        .insert("Translation".to_owned(), json!([0.0, 0.0, 2.0]));
    let changed = graph::evaluate_at_frame(
        doc.node_groups.get(&group_id).unwrap(),
        None,
        &Map::new(),
        3.0,
        "Geometry",
    )
    .unwrap();
    assert_close(changed.mesh.bounds().unwrap().min.z, 5.5);
}

#[test]
fn stored_named_attribute_drives_geometry() {
    let doc = graph_with_nodes(
        vec![
            json!({"id":"cube","type":"GeometryNodeMeshCube","properties":{"size":[1.0,1.0,1.0]}}),
            json!({"id":"position_input","type":"GeometryNodeInputPosition"}),
            json!({"id":"store","type":"GeometryNodeStoreNamedAttribute","properties":{"domain":"POINT","data_type":"FLOAT_VECTOR"},"inputs":{"Name":"captured_position"}}),
            json!({"id":"attribute","type":"GeometryNodeInputNamedAttribute","properties":{"data_type":"FLOAT_VECTOR"},"inputs":{"Name":"captured_position"}}),
            json!({"id":"position","type":"GeometryNodeSetPosition"}),
            json!({"id":"output","type":"NodeGroupOutput"}),
        ],
        vec![
            json!({"from_node":"cube","from_socket":"Mesh","to_node":"store","to_socket":"Geometry"}),
            json!({"from_node":"position_input","from_socket":"Position","to_node":"store","to_socket":"Value"}),
            json!({"from_node":"store","from_socket":"Geometry","to_node":"position","to_socket":"Geometry"}),
            json!({"from_node":"attribute","from_socket":"Attribute","to_node":"position","to_socket":"Offset"}),
            json!({"from_node":"position","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
        ],
    );
    let bounds = evaluate(&doc).mesh.bounds().unwrap();
    assert_close(bounds.size().x, 2.0);
    assert_close(bounds.size().y, 2.0);
    assert_close(bounds.size().z, 2.0);
}

#[test]
fn capture_attribute_keeps_values_from_capture_geometry() {
    let doc = graph_with_nodes(
        vec![
            json!({"id":"cube","type":"GeometryNodeMeshCube","properties":{"size":[1.0,1.0,1.0]}}),
            json!({"id":"position_input","type":"GeometryNodeInputPosition"}),
            json!({"id":"capture","type":"GeometryNodeCaptureAttribute","properties":{"domain":"POINT","data_type":"FLOAT_VECTOR"}}),
            json!({"id":"transform","type":"GeometryNodeTransform","inputs":{"Translation":[1.0,0.0,0.0]}}),
            json!({"id":"position","type":"GeometryNodeSetPosition"}),
            json!({"id":"output","type":"NodeGroupOutput"}),
        ],
        vec![
            json!({"from_node":"cube","from_socket":"Mesh","to_node":"capture","to_socket":"Geometry"}),
            json!({"from_node":"position_input","from_socket":"Position","to_node":"capture","to_socket":"Value"}),
            json!({"from_node":"capture","from_socket":"Geometry","to_node":"transform","to_socket":"Geometry"}),
            json!({"from_node":"transform","from_socket":"Geometry","to_node":"position","to_socket":"Geometry"}),
            json!({"from_node":"capture","from_socket":"Attribute","to_node":"position","to_socket":"Offset"}),
            json!({"from_node":"position","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
        ],
    );
    let bounds = evaluate(&doc).mesh.bounds().unwrap();
    assert_close(bounds.size().x, 2.0);
    assert_close(bounds.min.x, 0.0);
    assert_close(bounds.max.x, 2.0);
}

#[test]
fn capture_attribute_supports_same_type_int_and_boolean_fields() {
    let int_doc = graph_with_nodes(
        vec![
            json!({"id":"line","type":"GeometryNodeMeshLine","inputs":{"Count":4}}),
            json!({"id":"index","type":"GeometryNodeInputIndex"}),
            json!({"id":"capture","type":"GeometryNodeCaptureAttribute","properties":{"domain":"POINT","data_type":"INT"}}),
            json!({"id":"store","type":"GeometryNodeStoreNamedAttribute","properties":{"domain":"POINT","data_type":"INT"},"inputs":{"Name":"captured_int"}}),
            json!({"id":"output","type":"NodeGroupOutput"}),
        ],
        vec![
            json!({"from_node":"line","from_socket":"Mesh","to_node":"capture","to_socket":"Geometry"}),
            json!({"from_node":"index","from_socket":"Index","to_node":"capture","to_socket":"Value"}),
            json!({"from_node":"capture","from_socket":"Geometry","to_node":"store","to_socket":"Geometry"}),
            json!({"from_node":"capture","from_socket":"Attribute","to_node":"store","to_socket":"Value"}),
            json!({"from_node":"store","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
        ],
    );
    let int_mesh = evaluate(&int_doc).mesh;
    assert_eq!(int_mesh.vertices.len(), 4);
    assert_eq!(
        int_mesh.attributes["captured_int"]["data"],
        json!([0, 1, 2, 3])
    );

    let boolean_doc = graph_with_nodes(
        vec![
            json!({"id":"line","type":"GeometryNodeMeshLine","inputs":{"Count":4}}),
            json!({"id":"random","type":"FunctionNodeRandomValue","properties":{"data_type":"BOOLEAN"},"inputs":{"Probability":0.35,"Seed":29}}),
            json!({"id":"capture","type":"GeometryNodeCaptureAttribute","properties":{"domain":"POINT","data_type":"BOOLEAN"}}),
            json!({"id":"store","type":"GeometryNodeStoreNamedAttribute","properties":{"domain":"POINT","data_type":"BOOLEAN"},"inputs":{"Name":"captured_boolean"}}),
            json!({"id":"output","type":"NodeGroupOutput"}),
        ],
        vec![
            json!({"from_node":"line","from_socket":"Mesh","to_node":"capture","to_socket":"Geometry"}),
            json!({"from_node":"random","from_socket":"Value","to_node":"capture","to_socket":"Value"}),
            json!({"from_node":"capture","from_socket":"Geometry","to_node":"store","to_socket":"Geometry"}),
            json!({"from_node":"capture","from_socket":"Attribute","to_node":"store","to_socket":"Value"}),
            json!({"from_node":"store","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
        ],
    );
    let boolean_mesh = evaluate(&boolean_doc).mesh;
    assert_eq!(boolean_mesh.vertices.len(), 4);
    let boolean_values = boolean_mesh
        .attributes
        .get("captured_boolean")
        .and_then(|attribute| attribute.get("data"))
        .and_then(|data| data.as_array());
    assert_eq!(boolean_values.map_or(0, Vec::len), 4);
    assert!(
        boolean_values.is_some_and(|values| values.iter().all(serde_json::Value::is_boolean)),
        "captured BOOLEAN fields must remain booleans",
    );
}

#[test]
fn random_value_uses_seeded_per_element_ids() {
    let doc = graph_with_nodes(
        vec![
            json!({"id":"line","type":"GeometryNodeMeshLine","inputs":{"Count":8}}),
            json!({"id":"random","type":"FunctionNodeRandomValue","properties":{"data_type":"FLOAT"},"inputs":{"Seed":17,"Min":-1.0,"Max":1.0}}),
            json!({"id":"combine","type":"ShaderNodeCombineXYZ"}),
            json!({"id":"position","type":"GeometryNodeSetPosition"}),
            json!({"id":"output","type":"NodeGroupOutput"}),
        ],
        vec![
            json!({"from_node":"line","from_socket":"Mesh","to_node":"position","to_socket":"Geometry"}),
            json!({"from_node":"random","from_socket":"Value","to_node":"combine","to_socket":"X"}),
            json!({"from_node":"combine","from_socket":"Vector","to_node":"position","to_socket":"Offset"}),
            json!({"from_node":"position","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
        ],
    );
    let first = evaluate(&doc).mesh;
    let second = evaluate(&doc).mesh;
    assert_eq!(first, second);
    assert!(
        first
            .vertices
            .iter()
            .all(|vertex| (-1.0..=1.0).contains(&vertex.co.x))
    );
    assert!(
        first
            .vertices
            .iter()
            .skip(1)
            .any(|vertex| vertex.co.x != first.vertices[0].co.x)
    );
}

#[test]
fn normal_input_produces_a_vector_field() {
    let doc = graph_with_nodes(
        vec![
            json!({"id":"cube","type":"GeometryNodeMeshCube","properties":{"size":[1.0,1.0,1.0]}}),
            json!({"id":"normal","type":"GeometryNodeInputNormal"}),
            json!({"id":"scale","type":"ShaderNodeVectorMath","properties":{"operation":"SCALE"},"inputs":{"Scale":0.1}}),
            json!({"id":"position","type":"GeometryNodeSetPosition"}),
            json!({"id":"output","type":"NodeGroupOutput"}),
        ],
        vec![
            json!({"from_node":"cube","from_socket":"Mesh","to_node":"position","to_socket":"Geometry"}),
            json!({"from_node":"normal","from_socket":"Normal","to_node":"scale","to_socket":"Vector"}),
            json!({"from_node":"scale","from_socket":"Vector","to_node":"position","to_socket":"Offset"}),
            json!({"from_node":"position","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
        ],
    );
    let bounds = evaluate(&doc).mesh.bounds().unwrap();
    assert!(bounds.size().x > 1.0);
    assert!(bounds.size().y > 1.0);
    assert!(bounds.size().z > 1.0);
}
