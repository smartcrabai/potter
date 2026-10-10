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
    let default = graph::evaluate(group, None, &Map::new()).unwrap();
    let overridden = graph::evaluate(
        group,
        None,
        &Map::from_iter([("scale".to_owned(), json!([2.0, 2.0, 2.0]))]),
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
    )
    .unwrap_err();
    assert_eq!(error.code, ErrorCode::EvaluationFailed);
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
        (2, 1, 0)
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
    for (domain, expected_count) in [("POINT", 8), ("EDGE", 12), ("FACE", 6), ("CORNER", 24)] {
        let doc = graph_with_nodes(
            vec![
                json!({"id":"cube","type":"GeometryNodeMeshCube"}),
                json!({"id":"points","type":"GeometryNodeMeshToPoints","properties":{"domain":domain}}),
                json!({"id":"output","type":"NodeGroupOutput"}),
            ],
            vec![
                json!({"from_node":"cube","from_socket":"Mesh","to_node":"points","to_socket":"Mesh"}),
                json!({"from_node":"points","from_socket":"Points","to_node":"output","to_socket":"Geometry"}),
            ],
        );
        assert_eq!(
            evaluate(&doc).mesh.vertices.len(),
            expected_count,
            "{domain}"
        );
    }
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
            json!({"id":"points","type":"GeometryNodeDistributePointsOnFaces","inputs":{"Density":3.0,"Distance Min":0.6,"Seed":73}}),
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
    let frame_three = graph::evaluate_at_frame(group, None, &Map::new(), 3.0).unwrap();
    let frame_two = graph::evaluate_at_frame(group, None, &Map::new(), 2.0).unwrap();
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
