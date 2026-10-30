use num_traits::ToPrimitive;
use std::error::Error;

use potter_core::{
    geom::Mesh,
    graph::{self, GraphEvaluation},
    model::{Id, SceneDoc},
    ops,
};
use serde_json::{Map, Value, json};

const ORACLE: &str = include_str!("fixtures/geometry_nodes_mesh_ops/expected.json");

#[derive(Clone, Copy)]
struct FieldSpec {
    name: &'static str,
    node_type: &'static str,
    output: &'static str,
    domain: &'static str,
    data_type: &'static str,
    input_index: Option<&'static str>,
    offset: Option<i64>,
}

const FIELDS: &[FieldSpec] = &[
    FieldSpec {
        name: "edge_unsigned_angle",
        node_type: "GeometryNodeInputMeshEdgeAngle",
        output: "Unsigned Angle",
        domain: "EDGE",
        data_type: "FLOAT",
        input_index: None,
        offset: None,
    },
    FieldSpec {
        name: "edge_signed_angle",
        node_type: "GeometryNodeInputMeshEdgeAngle",
        output: "Signed Angle",
        domain: "EDGE",
        data_type: "FLOAT",
        input_index: None,
        offset: None,
    },
    FieldSpec {
        name: "edge_unsigned_angle_corner",
        node_type: "GeometryNodeInputMeshEdgeAngle",
        output: "Unsigned Angle",
        domain: "CORNER",
        data_type: "FLOAT",
        input_index: None,
        offset: None,
    },
    FieldSpec {
        name: "edge_signed_angle_corner",
        node_type: "GeometryNodeInputMeshEdgeAngle",
        output: "Signed Angle",
        domain: "CORNER",
        data_type: "FLOAT",
        input_index: None,
        offset: None,
    },
    FieldSpec {
        name: "edge_face_count",
        node_type: "GeometryNodeInputMeshEdgeNeighbors",
        output: "Face Count",
        domain: "EDGE",
        data_type: "INT",
        input_index: None,
        offset: None,
    },
    FieldSpec {
        name: "edge_vertex_1",
        node_type: "GeometryNodeInputMeshEdgeVertices",
        output: "Vertex Index 1",
        domain: "EDGE",
        data_type: "INT",
        input_index: None,
        offset: None,
    },
    FieldSpec {
        name: "edge_vertex_2",
        node_type: "GeometryNodeInputMeshEdgeVertices",
        output: "Vertex Index 2",
        domain: "EDGE",
        data_type: "INT",
        input_index: None,
        offset: None,
    },
    FieldSpec {
        name: "edge_position_1",
        node_type: "GeometryNodeInputMeshEdgeVertices",
        output: "Position 1",
        domain: "EDGE",
        data_type: "FLOAT_VECTOR",
        input_index: None,
        offset: None,
    },
    FieldSpec {
        name: "edge_position_2",
        node_type: "GeometryNodeInputMeshEdgeVertices",
        output: "Position 2",
        domain: "EDGE",
        data_type: "FLOAT_VECTOR",
        input_index: None,
        offset: None,
    },
    FieldSpec {
        name: "face_area",
        node_type: "GeometryNodeInputMeshFaceArea",
        output: "Area",
        domain: "FACE",
        data_type: "FLOAT",
        input_index: None,
        offset: None,
    },
    FieldSpec {
        name: "face_vertex_count",
        node_type: "GeometryNodeInputMeshFaceNeighbors",
        output: "Vertex Count",
        domain: "FACE",
        data_type: "INT",
        input_index: None,
        offset: None,
    },
    FieldSpec {
        name: "face_neighbor_count",
        node_type: "GeometryNodeInputMeshFaceNeighbors",
        output: "Face Count",
        domain: "FACE",
        data_type: "INT",
        input_index: None,
        offset: None,
    },
    FieldSpec {
        name: "vertex_neighbor_vertex_count",
        node_type: "GeometryNodeInputMeshVertexNeighbors",
        output: "Vertex Count",
        domain: "POINT",
        data_type: "INT",
        input_index: None,
        offset: None,
    },
    FieldSpec {
        name: "vertex_neighbor_face_count",
        node_type: "GeometryNodeInputMeshVertexNeighbors",
        output: "Face Count",
        domain: "POINT",
        data_type: "INT",
        input_index: None,
        offset: None,
    },
    FieldSpec {
        name: "corners_of_edge_index",
        node_type: "GeometryNodeCornersOfEdge",
        output: "Corner Index",
        domain: "EDGE",
        data_type: "INT",
        input_index: Some("Edge Index"),
        offset: None,
    },
    FieldSpec {
        name: "corners_of_edge_total",
        node_type: "GeometryNodeCornersOfEdge",
        output: "Total",
        domain: "EDGE",
        data_type: "INT",
        input_index: Some("Edge Index"),
        offset: None,
    },
    FieldSpec {
        name: "corners_of_face_index",
        node_type: "GeometryNodeCornersOfFace",
        output: "Corner Index",
        domain: "FACE",
        data_type: "INT",
        input_index: Some("Face Index"),
        offset: None,
    },
    FieldSpec {
        name: "corners_of_face_total",
        node_type: "GeometryNodeCornersOfFace",
        output: "Total",
        domain: "FACE",
        data_type: "INT",
        input_index: Some("Face Index"),
        offset: None,
    },
    FieldSpec {
        name: "corners_of_face_default_index",
        node_type: "GeometryNodeCornersOfFace",
        output: "Corner Index",
        domain: "FACE",
        data_type: "INT",
        input_index: Some("Face Index"),
        offset: None,
    },
    FieldSpec {
        name: "corners_of_vertex_index",
        node_type: "GeometryNodeCornersOfVertex",
        output: "Corner Index",
        domain: "POINT",
        data_type: "INT",
        input_index: Some("Vertex Index"),
        offset: None,
    },
    FieldSpec {
        name: "corners_of_vertex_total",
        node_type: "GeometryNodeCornersOfVertex",
        output: "Total",
        domain: "POINT",
        data_type: "INT",
        input_index: Some("Vertex Index"),
        offset: None,
    },
    FieldSpec {
        name: "edges_of_vertex_index",
        node_type: "GeometryNodeEdgesOfVertex",
        output: "Edge Index",
        domain: "POINT",
        data_type: "INT",
        input_index: Some("Vertex Index"),
        offset: None,
    },
    FieldSpec {
        name: "edges_of_vertex_total",
        node_type: "GeometryNodeEdgesOfVertex",
        output: "Total",
        domain: "POINT",
        data_type: "INT",
        input_index: Some("Vertex Index"),
        offset: None,
    },
    FieldSpec {
        name: "edges_of_corner_next",
        node_type: "GeometryNodeEdgesOfCorner",
        output: "Next Edge Index",
        domain: "CORNER",
        data_type: "INT",
        input_index: Some("Corner Index"),
        offset: None,
    },
    FieldSpec {
        name: "edges_of_corner_previous",
        node_type: "GeometryNodeEdgesOfCorner",
        output: "Previous Edge Index",
        domain: "CORNER",
        data_type: "INT",
        input_index: Some("Corner Index"),
        offset: None,
    },
    FieldSpec {
        name: "face_of_corner_index",
        node_type: "GeometryNodeFaceOfCorner",
        output: "Face Index",
        domain: "CORNER",
        data_type: "INT",
        input_index: Some("Corner Index"),
        offset: None,
    },
    FieldSpec {
        name: "face_of_corner_offset",
        node_type: "GeometryNodeFaceOfCorner",
        output: "Index in Face",
        domain: "CORNER",
        data_type: "INT",
        input_index: Some("Corner Index"),
        offset: None,
    },
    FieldSpec {
        name: "offset_corner_next",
        node_type: "GeometryNodeOffsetCornerInFace",
        output: "Corner Index",
        domain: "CORNER",
        data_type: "INT",
        input_index: Some("Corner Index"),
        offset: Some(1),
    },
    FieldSpec {
        name: "vertex_of_corner_index",
        node_type: "GeometryNodeVertexOfCorner",
        output: "Vertex Index",
        domain: "CORNER",
        data_type: "INT",
        input_index: Some("Corner Index"),
        offset: None,
    },
];

const OPERATION_CASES: &[&str] = &[
    "split_edges",
    "delete_faces",
    "delete_all_points",
    "delete_points_edge_face",
    "delete_points_only_face",
    "delete_edges_edge_face",
    "delete_edges_only_face",
    "separate_faces",
    "separate_faces_inverted",
    "triangulate",
    "scale_elements",
    "scale_elements_single_axis",
    "scale_elements_single_axis_zero_axis",
    "subdivide",
    "subdivide_attributes",
    "subdivide_level_2",
    "extrude",
    "extrude_face_fields",
    "extrude_face_individual_fields",
    "extrude_face_top_on_points",
    "extrude_edge_fields",
    "extrude_vertex_fields",
    "merge_by_distance",
    "merge_by_distance_connected",
    "flip_faces",
    "dual_mesh",
    "dual_mesh_drop_boundaries",
];

type TestResult<T> = Result<T, Box<dyn Error>>;

fn fixture() -> TestResult<Value> {
    Ok(serde_json::from_str(ORACLE)?)
}

fn input_mesh(oracle: &Value) -> TestResult<Mesh> {
    Ok(serde_json::from_value(oracle["input"].clone())?)
}

fn graph_with_nodes(nodes: Vec<Value>, links: Vec<Value>) -> TestResult<SceneDoc> {
    let doc = SceneDoc::new("8e8fa7b0-a854-4cde-8d77-fec9bbfd62c5".to_owned());
    let mut operations =
        vec![json!({"op":"graph.create","id":"geo","name":"Geometry","kind":"geometry"})];
    operations.extend(nodes.into_iter().map(|mut node| {
        node["op"] = json!("graph.node_add");
        node["graph"] = json!("geo");
        node
    }));
    operations.extend(links.into_iter().map(|mut link| {
        link["op"] = json!("graph.link");
        link["graph"] = json!("geo");
        link
    }));
    Ok(ops::apply_batch(
        &doc,
        &json!({"schema_version":1,"base_revision":doc.revision,"operations":operations}),
    )?
    .doc)
}
fn nested_top_group_doc() -> TestResult<SceneDoc> {
    let doc = SceneDoc::new("95f6921b-f835-4f6c-9271-c1c098d9a823".to_owned());
    let operations = json!([
        {"op":"graph.create","id":"child","kind":"geometry"},
        {"op":"graph.interface_update","graph":"child","set":{
            "inputs":[
                {"id":"geometry","name":"Geometry","socket_type":"geometry"},
                {"id":"selection","name":"Selection","socket_type":"boolean","default":true}
            ],
            "outputs":[
                {"id":"out_geometry","name":"Geometry","socket_type":"geometry"},
                {"id":"out_top","name":"Top","socket_type":"boolean","default":false}
            ]
        }},
        {"op":"graph.node_add","graph":"child","id":"input","type":"NodeGroupInput"},
        {"op":"graph.node_add","graph":"child","id":"extrude","type":"GeometryNodeExtrudeMesh",
         "properties":{"mode":"FACES"},"inputs":{"Offset Scale":1.0,"Individual":false}},
        {"op":"graph.node_add","graph":"child","id":"subdivide","type":"GeometryNodeSubdivideMesh",
         "inputs":{"Level":1}},
        {"op":"graph.node_add","graph":"child","id":"output","type":"NodeGroupOutput"},
        {"op":"graph.link","graph":"child","from_node":"input","from_socket":"Geometry","to_node":"extrude","to_socket":"Mesh"},
        {"op":"graph.link","graph":"child","from_node":"input","from_socket":"Selection","to_node":"extrude","to_socket":"Selection"},
        {"op":"graph.link","graph":"child","from_node":"extrude","from_socket":"Mesh","to_node":"subdivide","to_socket":"Mesh"},
        {"op":"graph.link","graph":"child","from_node":"subdivide","from_socket":"Mesh","to_node":"output","to_socket":"Geometry"},
        {"op":"graph.link","graph":"child","from_node":"extrude","from_socket":"Top","to_node":"output","to_socket":"Top"},
        {"op":"graph.create","id":"geo","kind":"geometry"},
        {"op":"graph.interface_update","graph":"geo","set":{
            "inputs":[
                {"id":"geometry","name":"Geometry","socket_type":"geometry"}
            ],
            "outputs":[
                {"id":"out_geometry","name":"Geometry","socket_type":"geometry"}
            ]
        }},
        {"op":"graph.node_add","graph":"geo","id":"input","type":"NodeGroupInput"},
        {"op":"graph.node_add","graph":"geo","id":"call_a","type":"GeometryNodeGroup",
         "properties":{"node_tree":"child"},"inputs":{"Selection":true}},
        {"op":"graph.node_add","graph":"geo","id":"call_b","type":"GeometryNodeGroup",
         "properties":{"node_tree":"child"},"inputs":{"Selection":false}},
        {"op":"graph.node_add","graph":"geo","id":"call_dead","type":"GeometryNodeGroup",
         "properties":{"node_tree":"child"},"inputs":{"Selection":true}},
        {"op":"graph.node_add","graph":"geo","id":"store_a","type":"GeometryNodeStoreNamedAttribute",
         "properties":{"domain":"FACE","data_type":"BOOLEAN"},"inputs":{"Name":"top_a"}},
        {"op":"graph.node_add","graph":"geo","id":"store_b","type":"GeometryNodeStoreNamedAttribute",
         "properties":{"domain":"FACE","data_type":"BOOLEAN"},"inputs":{"Name":"cross_call_top_b"}},
        {"op":"graph.node_add","graph":"geo","id":"dead_store","type":"GeometryNodeStoreNamedAttribute",
         "properties":{"domain":"FACE","data_type":"BOOLEAN"},"inputs":{"Name":"unused_top"}},
        {"op":"graph.node_add","graph":"geo","id":"distribute","type":"GeometryNodeDistributePointsOnFaces",
         "inputs":{"Selection":true,"Density":0.1,"Density Factor":1.0,"Density Max":1.0,"Seed":17}},
        {"op":"graph.node_add","graph":"geo","id":"join","type":"GeometryNodeJoinGeometry"},
        {"op":"graph.node_add","graph":"geo","id":"output","type":"NodeGroupOutput"},
        {"op":"graph.link","graph":"geo","from_node":"input","from_socket":"Geometry","to_node":"call_a","to_socket":"Geometry"},
        {"op":"graph.link","graph":"geo","from_node":"input","from_socket":"Geometry","to_node":"call_b","to_socket":"Geometry"},
        {"op":"graph.link","graph":"geo","from_node":"input","from_socket":"Geometry","to_node":"call_dead","to_socket":"Geometry"},
        {"op":"graph.link","graph":"geo","from_node":"call_a","from_socket":"Geometry","to_node":"store_a","to_socket":"Geometry"},
        {"op":"graph.link","graph":"geo","from_node":"call_a","from_socket":"Top","to_node":"store_a","to_socket":"Value"},
        {"op":"graph.link","graph":"geo","from_node":"store_a","from_socket":"Geometry","to_node":"store_b","to_socket":"Geometry"},
        {"op":"graph.link","graph":"geo","from_node":"call_b","from_socket":"Top","to_node":"store_b","to_socket":"Value"},
        {"op":"graph.link","graph":"geo","from_node":"call_dead","from_socket":"Geometry","to_node":"dead_store","to_socket":"Geometry"},
        {"op":"graph.link","graph":"geo","from_node":"call_dead","from_socket":"Top","to_node":"dead_store","to_socket":"Value"},
        {"op":"graph.link","graph":"geo","from_node":"call_dead","from_socket":"Geometry","to_node":"distribute","to_socket":"Mesh"},
        {"op":"graph.link","graph":"geo","from_node":"distribute","from_socket":"Points","to_node":"join","to_socket":"Geometry"},
        {"op":"graph.link","graph":"geo","from_node":"store_b","from_socket":"Geometry","to_node":"join","to_socket":"Geometry"},
        {"op":"graph.link","graph":"geo","from_node":"join","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}
    ]);
    Ok(ops::apply_batch(
        &doc,
        &json!({
            "schema_version": 1,
            "base_revision": doc.revision,
            "operations": operations,
        }),
    )?
    .doc)
}


fn evaluate(doc: &SceneDoc, mesh: &Mesh) -> TestResult<GraphEvaluation> {
    let graph_id = Id::new("geo")?;
    let group = doc.node_groups.get(&graph_id).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::NotFound, "geometry graph missing")
    })?;
    Ok(graph::evaluate(
        group,
        Some(mesh),
        &Map::new(),
        "GNMeshOpsFixture",
    )?)
}

fn normalize_uv_layers(mesh: &Mesh, attributes: &mut Map<String, Value>) {
    let Some(layers) = attributes
        .get_mut("uv_layers")
        .and_then(|store| store.get_mut("layers"))
        .and_then(Value::as_array_mut)
    else {
        return;
    };
    for layer in layers {
        let rows = layer
            .get("faces")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let by_face: std::collections::HashMap<_, _> = rows
            .iter()
            .filter_map(|row| {
                Some((
                    u32::try_from(row.get("face_id")?.as_u64()?).ok()?,
                    row.get("uv")?.clone(),
                ))
            })
            .collect();
        layer["faces"] = json!(
            mesh.faces
                .iter()
                .map(|face| by_face.get(&face.id).cloned().unwrap_or(Value::Null))
                .collect::<Vec<_>>()
        );
    }
}

fn normalize_color_layers(attributes: &mut Map<String, Value>, mesh: &Mesh) {
    let Some(names) = attributes
        .get("color_layers")
        .and_then(|store| store.get("layers"))
        .and_then(Value::as_array)
        .cloned()
    else {
        return;
    };
    for name in names.iter().filter_map(Value::as_str) {
        if let Some(attribute) = attributes.get_mut(name)
            && let Some(values) = attribute.get("values").and_then(Value::as_object).cloned()
        {
            let domain = attribute.get("domain").and_then(Value::as_str);
            let keys: Vec<_> = match domain {
                Some("point") => mesh
                    .vertices
                    .iter()
                    .map(|vertex| format!("v{}", vertex.id))
                    .collect(),
                Some("face" | "corner") => mesh
                    .faces
                    .iter()
                    .map(|face| format!("f{}", face.id))
                    .collect(),
                _ => Vec::new(),
            };
            attribute["values"] = json!(
                keys.iter()
                    .map(|key| values.get(key).cloned().unwrap_or(Value::Null))
                    .collect::<Vec<_>>()
            );
        }
    }
}

fn normalize_id_attribute(
    attributes: &mut Map<String, Value>,
    name: &str,
    prefix: &str,
    ids: impl Iterator<Item = u32>,
    default: &Value,
) {
    if let Some(attribute) = attributes.get_mut(name)
        && let Some(values) = attribute.get("values").and_then(Value::as_object).cloned()
    {
        attribute["values"] = json!(
            ids.map(|id| {
                values
                    .get(&format!("{prefix}{id}"))
                    .cloned()
                    .unwrap_or_else(|| default.clone())
            })
            .collect::<Vec<_>>()
        );
    }
}

fn normalize_mesh_flags(mesh: &Mesh, attributes: &mut Map<String, Value>) {
    normalize_id_attribute(
        attributes,
        "crease_edge",
        "e",
        mesh.edges.iter().map(|edge| edge.id),
        &json!(0.0),
    );
    normalize_id_attribute(
        attributes,
        "crease_vert",
        "v",
        mesh.vertices.iter().map(|vertex| vertex.id),
        &json!(0.0),
    );
    normalize_id_attribute(
        attributes,
        "sharp",
        "e",
        mesh.edges.iter().map(|edge| edge.id),
        &json!(false),
    );
    normalize_id_attribute(
        attributes,
        "seam",
        "e",
        mesh.edges.iter().map(|edge| edge.id),
        &json!(false),
    );
    normalize_id_attribute(
        attributes,
        "shade_smooth",
        "f",
        mesh.faces.iter().map(|face| face.id),
        &json!(false),
    );
}

fn mesh_snapshot_attributes(mesh: &Mesh) -> Map<String, Value> {
    let mut attributes = mesh.attributes.clone();
    normalize_uv_layers(mesh, &mut attributes);
    normalize_color_layers(&mut attributes, mesh);
    normalize_mesh_flags(mesh, &mut attributes);
    attributes
}

fn mesh_snapshot(mesh: &Mesh) -> Value {
    let vertex_indices: std::collections::HashMap<_, _> = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let edges: Vec<_> = mesh
        .edges
        .iter()
        .map(|edge| {
            let mut endpoints = [
                vertex_indices[&edge.vertices[0]],
                vertex_indices[&edge.vertices[1]],
            ];
            endpoints.sort_unstable();
            json!(endpoints)
        })
        .collect();
    let faces: Vec<_> = mesh
        .faces
        .iter()
        .map(|face| {
            json!(
                face.vertices
                    .iter()
                    .map(|id| vertex_indices[id])
                    .collect::<Vec<_>>()
            )
        })
        .collect();
    let vertices: Vec<_> = mesh
        .vertices
        .iter()
        .map(|vertex| json!([vertex.co.x, vertex.co.y, vertex.co.z]))
        .collect();
    json!({
        "vertices":vertices,
        "edges":edges,
        "faces":faces,
        "attributes":mesh_snapshot_attributes(mesh),
    })
}

fn ordered_f32_bits(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits & 0x8000_0000 == 0 {
        bits | 0x8000_0000
    } else {
        !bits
    }
}

fn assert_json_close(actual: &Value, expected: &Value, path: &str) -> TestResult<()> {
    match (actual, expected) {
        (Value::Number(actual), Value::Number(expected)) => {
            if let (Some(actual), Some(expected)) = (actual.as_i64(), expected.as_i64()) {
                assert_eq!(actual, expected, "{path}");
            } else if let (Some(actual), Some(expected)) = (actual.as_u64(), expected.as_u64()) {
                assert_eq!(actual, expected, "{path}");
            } else {
                match (actual.as_f64(), expected.as_f64()) {
                    (Some(actual), Some(expected)) => {
                        let actual_f32 = actual.to_f32().ok_or_else(|| {
                            std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                format!("{path}: actual {actual:?} is not representable as f32"),
                            )
                        })?;
                        let expected_f32 = expected.to_f32().ok_or_else(|| {
                            std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                format!(
                                    "{path}: expected {expected:?} is not representable as f32"
                                ),
                            )
                        })?;
                        let ulp_distance =
                            ordered_f32_bits(actual_f32).abs_diff(ordered_f32_bits(expected_f32));
                        assert!(
                            ulp_distance <= 1,
                            "{path}: actual {actual:?}, expected {expected:?}, f32 ULP distance {ulp_distance}"
                        );
                    }
                    _ => assert_eq!(actual, expected, "{path}"),
                }
            }
        }
        (Value::Array(actual), Value::Array(expected)) => {
            assert_eq!(actual.len(), expected.len(), "{path} length");
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                assert_json_close(actual, expected, &format!("{path}[{index}]"))?;
            }
        }
        (Value::Object(actual), Value::Object(expected)) => {
            assert_eq!(
                actual.keys().collect::<Vec<_>>(),
                expected.keys().collect::<Vec<_>>(),
                "{path} keys"
            );
            for (key, expected) in expected {
                assert_json_close(&actual[key], expected, &format!("{path}.{key}"))?;
            }
        }
        _ => assert_eq!(actual, expected, "{path}"),
    }
    Ok(())
}

fn topology_graph() -> TestResult<SceneDoc> {
    let mut nodes = vec![json!({"id":"index","type":"GeometryNodeInputIndex"})];
    let mut links = Vec::new();
    let mut previous_store = None;
    for (index, field) in FIELDS.iter().enumerate() {
        let field_id = format!("field_{index}");
        let store_id = format!("store_{index}");
        let mut field_node = json!({"id":field_id,"type":field.node_type});
        if field.name == "corners_of_face_default_index" {
            field_node["inputs"] = json!({"Face Index":0,"Sort Index":0});
        } else if let Some(offset) = field.offset {
            field_node["inputs"] = json!({"Offset":offset});
        } else if matches!(
            field.node_type,
            "GeometryNodeCornersOfEdge"
                | "GeometryNodeCornersOfFace"
                | "GeometryNodeCornersOfVertex"
                | "GeometryNodeEdgesOfVertex"
        ) {
            field_node["inputs"] = json!({"Sort Index":0});
        }
        nodes.push(field_node);
        nodes.push(json!({
            "id":store_id,
            "type":"GeometryNodeStoreNamedAttribute",
            "properties":{"domain":field.domain,"data_type":field.data_type},
            "inputs":{"Name":field.name},
        }));
        links.push(json!({
            "from_node":field_id,
            "from_socket":field.output,
            "to_node":store_id,
            "to_socket":"Value",
        }));
        if let Some(input_index) = field.input_index {
            links.push(json!({
                "from_node":"index",
                "from_socket":"Index",
                "to_node":field_id,
                "to_socket":input_index,
            }));
        }
        if let Some(previous) = previous_store {
            links.push(json!({
                "from_node":previous,
                "from_socket":"Geometry",
                "to_node":store_id,
                "to_socket":"Geometry",
            }));
        }
        previous_store = Some(store_id);
    }
    nodes.push(json!({"id":"output","type":"NodeGroupOutput"}));
    links.push(json!({
        "from_node":format!("store_{}", FIELDS.len() - 1),
        "from_socket":"Geometry",
        "to_node":"output",
        "to_socket":"Geometry",
    }));
    graph_with_nodes(nodes, links)
}

fn extrusion_field_spec(
    name: &str,
) -> Option<(
    &'static str,
    bool,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
)> {
    match name {
        "extrude_face_fields" => Some((
            "FACES",
            false,
            "extrude_face_top",
            "FACE",
            "extrude_face_side",
            "FACE",
        )),
        "extrude_face_top_on_points" => Some((
            "FACES",
            false,
            "extrude_face_top_on_points",
            "POINT",
            "extrude_face_side_on_points",
            "FACE",
        )),
        "extrude_face_individual_fields" => Some((
            "FACES",
            true,
            "extrude_face_individual_top",
            "FACE",
            "extrude_face_individual_side",
            "FACE",
        )),
        "extrude_edge_fields" => Some((
            "EDGES",
            false,
            "extrude_edge_top",
            "EDGE",
            "extrude_edge_side",
            "FACE",
        )),
        "extrude_vertex_fields" => Some((
            "VERTICES",
            false,
            "extrude_vertex_top",
            "POINT",
            "extrude_vertex_side",
            "EDGE",
        )),
        _ => None,
    }
}

fn extrusion_fields_graph(name: &str) -> TestResult<SceneDoc> {
    let (mode, individual, top_name, top_domain, side_name, side_domain) =
        extrusion_field_spec(name).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "unknown extrusion field case",
            )
        })?;
    let mut inputs = json!({"Selection":true,"Offset Scale":1.0});
    if mode == "FACES" {
        inputs["Individual"] = json!(individual);
    }
    let nodes = vec![
        json!({
            "id":"extrude",
            "type":"GeometryNodeExtrudeMesh",
            "properties":{"mode":mode},
            "inputs":inputs,
        }),
        json!({
            "id":"top_store",
            "type":"GeometryNodeStoreNamedAttribute",
            "properties":{"domain":top_domain,"data_type":"BOOLEAN"},
            "inputs":{"Name":top_name},
        }),
        json!({
            "id":"side_store",
            "type":"GeometryNodeStoreNamedAttribute",
            "properties":{"domain":side_domain,"data_type":"BOOLEAN"},
            "inputs":{"Name":side_name},
        }),
        json!({"id":"output","type":"NodeGroupOutput"}),
    ];
    let links = vec![
        json!({"from_node":"extrude","from_socket":"Mesh","to_node":"top_store","to_socket":"Geometry"}),
        json!({"from_node":"extrude","from_socket":"Top","to_node":"top_store","to_socket":"Value"}),
        json!({"from_node":"top_store","from_socket":"Geometry","to_node":"side_store","to_socket":"Geometry"}),
        json!({"from_node":"extrude","from_socket":"Side","to_node":"side_store","to_socket":"Value"}),
        json!({"from_node":"side_store","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
    ];
    graph_with_nodes(nodes, links)
}
fn top_side_subdivide_graph() -> TestResult<SceneDoc> {
    let nodes = vec![
        json!({
            "id":"selection",
            "type":"GeometryNodeInputNamedAttribute",
            "properties":{"data_type":"BOOLEAN"},
            "inputs":{"Name":"face_selected"},
        }),
        json!({
            "id":"extrude",
            "type":"GeometryNodeExtrudeMesh",
            "properties":{"mode":"FACES"},
            "inputs":{"Offset Scale":1.0,"Individual":false},
        }),
        json!({
            "id":"subdivide",
            "type":"GeometryNodeSubdivideMesh",
            "inputs":{"Level":1},
        }),
        json!({
            "id":"top_store",
            "type":"GeometryNodeStoreNamedAttribute",
            "properties":{"domain":"FACE","data_type":"BOOLEAN"},
            "inputs":{"Name":"top_after_subdivide"},
        }),
        json!({
            "id":"side_store",
            "type":"GeometryNodeStoreNamedAttribute",
            "properties":{"domain":"FACE","data_type":"BOOLEAN"},
            "inputs":{"Name":"side_after_subdivide"},
        }),
        json!({"id":"output","type":"NodeGroupOutput"}),
    ];
    let links = vec![
        json!({"from_node":"selection","from_socket":"Attribute","to_node":"extrude","to_socket":"Selection"}),
        json!({"from_node":"extrude","from_socket":"Mesh","to_node":"subdivide","to_socket":"Mesh"}),
        json!({"from_node":"subdivide","from_socket":"Mesh","to_node":"top_store","to_socket":"Geometry"}),
        json!({"from_node":"extrude","from_socket":"Top","to_node":"top_store","to_socket":"Value"}),
        json!({"from_node":"top_store","from_socket":"Geometry","to_node":"side_store","to_socket":"Geometry"}),
        json!({"from_node":"extrude","from_socket":"Side","to_node":"side_store","to_socket":"Value"}),
        json!({"from_node":"side_store","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
    ];
    graph_with_nodes(nodes, links)
}

fn vertex_top_edge_graph() -> TestResult<SceneDoc> {
    let nodes = vec![
        json!({"id":"offset","type":"GeometryNodeInputPosition"}),
        json!({
            "id":"extrude",
            "type":"GeometryNodeExtrudeMesh",
            "properties":{"mode":"VERTICES"},
            "inputs":{"Selection":true,"Offset Scale":1.0},
        }),
        json!({
            "id":"store",
            "type":"GeometryNodeStoreNamedAttribute",
            "properties":{"domain":"EDGE","data_type":"BOOLEAN"},
            "inputs":{"Name":"top_on_edge"},
        }),
        json!({"id":"output","type":"NodeGroupOutput"}),
    ];
    let links = vec![
        json!({"from_node":"offset","from_socket":"Position","to_node":"extrude","to_socket":"Offset"}),
        json!({"from_node":"extrude","from_socket":"Mesh","to_node":"store","to_socket":"Geometry"}),
        json!({"from_node":"extrude","from_socket":"Top","to_node":"store","to_socket":"Value"}),
        json!({"from_node":"store","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
    ];
    graph_with_nodes(nodes, links)
}
fn top_on_original_graph() -> TestResult<SceneDoc> {
    let nodes = vec![
        json!({
            "id":"extrude",
            "type":"GeometryNodeExtrudeMesh",
            "properties":{"mode":"FACES"},
            "inputs":{"Offset Scale":1.0,"Individual":false},
        }),
        json!({
            "id":"store",
            "type":"GeometryNodeStoreNamedAttribute",
            "properties":{"domain":"FACE","data_type":"BOOLEAN"},
            "inputs":{"Name":"top_on_original"},
        }),
        json!({"id":"output","type":"NodeGroupOutput"}),
    ];
    let links = vec![
        json!({"from_node":"extrude","from_socket":"Top","to_node":"store","to_socket":"Value"}),
        json!({"from_node":"store","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
    ];
    graph_with_nodes(nodes, links)
}
fn unconsumed_top_distribute_graph() -> TestResult<SceneDoc> {
    let nodes = vec![
        json!({
            "id":"extrude",
            "type":"GeometryNodeExtrudeMesh",
            "properties":{"mode":"FACES"},
            "inputs":{"Offset Scale":1.0,"Individual":false},
        }),
        json!({
            "id":"dead_store",
            "type":"GeometryNodeStoreNamedAttribute",
            "properties":{"domain":"FACE","data_type":"BOOLEAN"},
            "inputs":{"Name":"unused_top"},
        }),
        json!({
            "id":"distribute",
            "type":"GeometryNodeDistributePointsOnFaces",
            "inputs":{
                "Selection":true,"Distance Min":0.1,"Density":0.1,
                "Density Factor":1.0,"Density Max":1.0,"Seed":17
            },
        }),
        json!({"id":"output","type":"NodeGroupOutput"}),
    ];
    let links = vec![
        json!({"from_node":"extrude","from_socket":"Mesh","to_node":"dead_store","to_socket":"Geometry"}),
        json!({"from_node":"extrude","from_socket":"Top","to_node":"dead_store","to_socket":"Value"}),
        json!({"from_node":"extrude","from_socket":"Mesh","to_node":"distribute","to_socket":"Mesh"}),
        json!({"from_node":"distribute","from_socket":"Points","to_node":"output","to_socket":"Geometry"}),
    ];
    graph_with_nodes(nodes, links)
}

fn assert_boolean_attribute(mesh: &Mesh, name: &str, domain: &str) -> TestResult<()> {
    let attribute = mesh.attributes.get(name).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("stored extrusion attribute {name} is missing"),
        )
    })?;
    assert_eq!(attribute["domain"], domain, "{name} domain");
    assert_eq!(attribute["data_type"], "BOOLEAN", "{name} type");
    let values = invalid_array(&attribute["data"], name)?;
    assert!(
        values.iter().any(|value| value.as_bool() == Some(true)),
        "{name} contains no true elements"
    );
    Ok(())
}

struct OperationCase {
    node_type: &'static str,
    result_socket: &'static str,
    properties: Value,
    inputs: Value,
    selection_attribute: Option<&'static str>,
}

fn selection_operation_case(name: &str) -> Option<OperationCase> {
    let (node_type, result_socket, properties, selection_attribute) = match name {
        "delete_faces" => (
            "GeometryNodeDeleteGeometry",
            "Geometry",
            json!({"domain":"FACE","mode":"ALL"}),
            "face_selected",
        ),
        "delete_points_edge_face" => (
            "GeometryNodeDeleteGeometry",
            "Geometry",
            json!({"domain":"POINT","mode":"EDGE_FACE"}),
            "point_selected",
        ),
        "delete_points_only_face" => (
            "GeometryNodeDeleteGeometry",
            "Geometry",
            json!({"domain":"POINT","mode":"ONLY_FACE"}),
            "point_selected",
        ),
        "delete_edges_edge_face" => (
            "GeometryNodeDeleteGeometry",
            "Geometry",
            json!({"domain":"EDGE","mode":"EDGE_FACE"}),
            "edge_selected",
        ),
        "delete_edges_only_face" => (
            "GeometryNodeDeleteGeometry",
            "Geometry",
            json!({"domain":"EDGE","mode":"ONLY_FACE"}),
            "edge_selected",
        ),
        "separate_faces" => (
            "GeometryNodeSeparateGeometry",
            "Selection",
            json!({"domain":"FACE"}),
            "face_selected",
        ),
        "separate_faces_inverted" => (
            "GeometryNodeSeparateGeometry",
            "Inverted",
            json!({"domain":"FACE"}),
            "face_selected",
        ),
        _ => return None,
    };
    Some(OperationCase {
        node_type,
        result_socket,
        properties,
        inputs: json!({}),
        selection_attribute: Some(selection_attribute),
    })
}

fn standard_operation_case(name: &str) -> Option<OperationCase> {
    let (node_type, result_socket, properties, inputs) = match name {
        "delete_all_points" => (
            "GeometryNodeDeleteGeometry",
            "Geometry",
            json!({"domain":"POINT","mode":"ALL"}),
            json!({"Selection":true}),
        ),
        "split_edges" => (
            "GeometryNodeSplitEdges",
            "Mesh",
            json!({}),
            json!({"Selection":true}),
        ),
        "triangulate" => (
            "GeometryNodeTriangulate",
            "Mesh",
            json!({}),
            json!({"Selection":true,"Quad Method":"Beauty","N-gon Method":"Beauty"}),
        ),
        "scale_elements" => (
            "GeometryNodeScaleElements",
            "Geometry",
            json!({"domain":"FACE"}),
            json!({"Selection":true,"Scale":0.65,"Center":[0.25,0.25,0.0],"Scale Mode":"Uniform","Axis":[0.0,0.0,1.0]}),
        ),
        "scale_elements_single_axis" => (
            "GeometryNodeScaleElements",
            "Geometry",
            json!({"domain":"FACE"}),
            json!({"Selection":true,"Scale":0.65,"Center":[0.25,0.25,0.0],"Scale Mode":"Single Axis","Axis":[0.0,0.0,1.0]}),
        ),
        "scale_elements_single_axis_zero_axis" => (
            "GeometryNodeScaleElements",
            "Geometry",
            json!({"domain":"EDGE"}),
            json!({"Selection":true,"Scale":2.0,"Center":[0.0,0.0,0.0],"Scale Mode":"Single Axis","Axis":[0.0,0.0,0.0]}),
        ),
        "subdivide" | "subdivide_attributes" => (
            "GeometryNodeSubdivideMesh",
            "Mesh",
            json!({}),
            json!({"Level":1}),
        ),
        "subdivide_level_2" => (
            "GeometryNodeSubdivideMesh",
            "Mesh",
            json!({}),
            json!({"Level":2}),
        ),
        "extrude" => (
            "GeometryNodeExtrudeMesh",
            "Mesh",
            json!({"mode":"FACES"}),
            json!({"Selection":true,"Offset Scale":1.0,"Individual":false}),
        ),
        "merge_by_distance" => (
            "GeometryNodeMergeByDistance",
            "Geometry",
            json!({}),
            json!({"Selection":true,"Mode":"All","Distance":0.001}),
        ),
        "merge_by_distance_connected" => (
            "GeometryNodeMergeByDistance",
            "Geometry",
            json!({}),
            json!({"Selection":true,"Mode":"Connected","Distance":0.001}),
        ),
        "flip_faces" => (
            "GeometryNodeFlipFaces",
            "Mesh",
            json!({}),
            json!({"Selection":true}),
        ),
        "dual_mesh" => (
            "GeometryNodeDualMesh",
            "Dual Mesh",
            json!({}),
            json!({"Keep Boundaries":true}),
        ),
        "dual_mesh_drop_boundaries" => (
            "GeometryNodeDualMesh",
            "Dual Mesh",
            json!({}),
            json!({"Keep Boundaries":false}),
        ),
        _ => return None,
    };
    Some(OperationCase {
        node_type,
        result_socket,
        properties,
        inputs,
        selection_attribute: None,
    })
}

fn operation_graph(name: &str) -> TestResult<SceneDoc> {
    if extrusion_field_spec(name).is_some() {
        return extrusion_fields_graph(name);
    }
    let case = selection_operation_case(name)
        .or_else(|| standard_operation_case(name))
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("unknown mesh operation case {name}"),
            )
        })?;
    operation_case_graph(&case)
}

fn operation_case_graph(case: &OperationCase) -> TestResult<SceneDoc> {
    let operation = json!({
        "id":"operation",
        "type":case.node_type,
        "properties":case.properties,
        "inputs":case.inputs,
    });
    let mut nodes = vec![operation, json!({"id":"output","type":"NodeGroupOutput"})];
    let mut links = vec![json!({
        "from_node":"operation",
        "from_socket":case.result_socket,
        "to_node":"output",
        "to_socket":"Geometry",
    })];
    if case.node_type == "GeometryNodeScaleElements" {
        let center = case
            .inputs
            .get("Center")
            .and_then(Value::as_array)
            .filter(|values| values.len() == 3)
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Scale Elements case has no three-component center",
                )
            })?;
        nodes.push(json!({
            "id":"center",
            "type":"ShaderNodeCombineXYZ",
            "inputs":{"X":center[0],"Y":center[1],"Z":center[2]},
        }));
        links.push(json!({
            "from_node":"center",
            "from_socket":"Vector",
            "to_node":"operation",
            "to_socket":"Center",
        }));
    }
    if let Some(name) = case.selection_attribute {
        nodes.push(json!({
            "id":"selection",
            "type":"GeometryNodeInputNamedAttribute",
            "properties":{"data_type":"BOOLEAN"},
            "inputs":{"Name":name},
        }));
        links.push(json!({
            "from_node":"selection",
            "from_socket":"Attribute",
            "to_node":"operation",
            "to_socket":"Selection",
        }));
    }
    graph_with_nodes(nodes, links)
}

fn closed_dual_graph() -> TestResult<SceneDoc> {
    graph_with_nodes(
        vec![
            json!({
                "id":"dual",
                "type":"GeometryNodeDualMesh",
                "inputs":{"Keep Boundaries":false},
            }),
            json!({"id":"output","type":"NodeGroupOutput"}),
        ],
        vec![json!({
            "from_node":"dual",
            "from_socket":"Dual Mesh",
            "to_node":"output",
            "to_socket":"Geometry",
        })],
    )
}

fn invalid_array<'a>(value: &'a Value, path: &str) -> TestResult<&'a [Value]> {
    value.as_array().map(Vec::as_slice).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{path} is not an array"),
        )
        .into()
    })
}
fn json_number(value: &Value, path: &str) -> TestResult<f64> {
    value.as_f64().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{path} is not numeric"),
        )
        .into()
    })
}

fn json_index(value: &Value, path: &str) -> TestResult<usize> {
    let index = value.as_u64().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{path} is not a nonnegative index"),
        )
    })?;
    Ok(usize::try_from(index)?)
}

fn assert_corner_edge_angle_adaptation(oracle: &Value) -> TestResult<()> {
    let attributes = &oracle["topology_fields"]["attributes"];
    let edge_angles = invalid_array(&attributes["edge_unsigned_angle"]["data"], "edge angles")?;
    let corner_angles = invalid_array(
        &attributes["edge_unsigned_angle_corner"]["data"],
        "corner angles",
    )?;
    let next_edges = invalid_array(&attributes["edges_of_corner_next"]["data"], "next edges")?;
    let previous_edges = invalid_array(
        &attributes["edges_of_corner_previous"]["data"],
        "previous edges",
    )?;
    assert_eq!(corner_angles.len(), next_edges.len());
    assert_eq!(corner_angles.len(), previous_edges.len());
    for index in 0..corner_angles.len() {
        let next_edge = json_index(&next_edges[index], "next edge index")?;
        let previous_edge = json_index(&previous_edges[index], "previous edge index")?;
        let next_angle = edge_angles.get(next_edge).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "next edge index is out of range",
            )
        })?;
        let previous_angle = edge_angles.get(previous_edge).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "previous edge index is out of range",
            )
        })?;
        let average = f64::midpoint(
            json_number(next_angle, "next edge angle")?,
            json_number(previous_angle, "previous edge angle")?,
        );
        assert_json_close(
            &corner_angles[index],
            &json!(average),
            &format!("corner edge angle[{index}]"),
        )?;
    }
    assert_eq!(
        attributes["corners_of_face_default_index"]["data"],
        attributes["corners_of_face_index"]["data"],
        "unconnected Corners of Face Sort Index defaults to index zero"
    );
    Ok(())
}

#[test]
fn topology_input_fields_and_index_functions_match_blender() -> TestResult<()> {
    let oracle = fixture()?;
    assert_eq!(oracle["blender_version"], "5.2.2 LTS");
    let mesh = input_mesh(&oracle)?;
    let result = evaluate(&topology_graph()?, &mesh)?;
    assert_json_close(
        &mesh_snapshot(&result.mesh),
        &oracle["topology_fields"],
        "topology_fields",
    )?;
    assert_corner_edge_angle_adaptation(&oracle)?;

    assert_eq!(mesh.vertices.len(), 11);
    assert_eq!(mesh.edges.len(), 13);
    assert_eq!(mesh.faces.len(), 4);
    assert_eq!(mesh.attributes["weight"]["data_type"], "FLOAT");
    assert_eq!(mesh.attributes["vector_tag"]["data_type"], "FLOAT_VECTOR");
    assert_eq!(mesh.attributes["face_tag"]["data_type"], "INT");
    assert_eq!(mesh.attributes["edge_flag"]["data_type"], "BOOLEAN");
    let blender_attributes =
        invalid_array(&mesh.attributes["blender_attributes"], "blender_attributes")?;
    assert_eq!(blender_attributes.len(), 2);
    assert_eq!(blender_attributes[0]["name"], "edge_int16_2d");
    assert_eq!(blender_attributes[0]["data_type"], "INT16_2D");
    assert_ne!(
        blender_attributes[0]["values"][0],
        blender_attributes[0]["values"][1]
    );
    assert_eq!(blender_attributes[1]["name"], "edge_float4x4");
    assert_eq!(blender_attributes[1]["data_type"], "FLOAT4X4");
    assert_ne!(
        blender_attributes[1]["values"][0],
        blender_attributes[1]["values"][1]
    );
    assert_eq!(mesh.attributes["face_selected"]["domain"], "FACE");
    assert_eq!(mesh.attributes["face_selected"]["data_type"], "BOOLEAN");
    assert_eq!(mesh.attributes["uv_layers"]["active"], "UVMap");
    assert_eq!(mesh.attributes["uv_layers"]["render"], "UVMap");
    assert_eq!(mesh.attributes["color_layers"]["layers"], json!(["Paint"]));
    assert_eq!(mesh.attributes["color_layers"]["active"], "Paint");
    assert_eq!(mesh.attributes["color_layers"]["render"], "Paint");
    assert_eq!(mesh.attributes["shade_smooth"]["values"]["f0"], true);
    assert_eq!(mesh.attributes["shade_smooth"]["values"]["f1"], false);
    assert_eq!(mesh.attributes["sharp"]["values"]["e3"], true);
    assert_eq!(mesh.attributes["seam"]["values"]["e1"], true);
    assert!(
        mesh.attributes["crease_edge"]["values"]["e3"]
            .as_f64()
            .is_some_and(|value| value > 0.0)
    );
    assert!(
        mesh.attributes["crease_vert"]["values"]["v4"]
            .as_f64()
            .is_some_and(|value| value > 0.0)
    );
    let edge_faces = invalid_array(
        &oracle["topology_fields"]["attributes"]["edge_face_count"]["data"],
        "edge_face_count",
    )?;
    assert!(edge_faces.iter().any(|count| count == 0));
    assert!(edge_faces.iter().any(|count| count == 1));
    assert!(edge_faces.iter().any(|count| count == 2));
    assert!(edge_faces.iter().any(|count| count == 3));
    assert_eq!(mesh.vertices[0].co, mesh.vertices[10].co);
    Ok(())
}

#[test]
fn mesh_operation_transitions_match_blender_topology_and_attributes() -> TestResult<()> {
    let oracle = fixture()?;
    let mesh = input_mesh(&oracle)?;
    for &name in OPERATION_CASES {
        let result = evaluate(&operation_graph(name)?, &mesh)?;
        assert_json_close(
            &mesh_snapshot(&result.mesh),
            &oracle["operations"][name],
            name,
        )?;
        if let Some((_, _, top_name, top_domain, side_name, side_domain)) =
            extrusion_field_spec(name)
        {
            assert_boolean_attribute(&result.mesh, top_name, top_domain)?;
            assert_boolean_attribute(&result.mesh, side_name, side_domain)?;
        }
        if name == "delete_all_points" {
            assert_eq!(result.mesh, Mesh::default());
        }
    }
    assert!(
        invalid_array(
            &oracle["operations"]["subdivide_level_2"]["faces"],
            "subdivide_level_2",
        )?
        .len()
            > invalid_array(&oracle["operations"]["subdivide"]["faces"], "subdivide")?.len(),
        "Subdivide Mesh Level 2 refines beyond Level 1"
    );

    assert_eq!(
        invalid_array(
            &oracle["operations"]["delete_faces"]["faces"],
            "delete_faces"
        )?
        .len(),
        1
    );
    assert_eq!(
        invalid_array(
            &oracle["operations"]["separate_faces"]["faces"],
            "separate_faces"
        )?
        .len(),
        3
    );
    assert_eq!(
        invalid_array(
            &oracle["operations"]["separate_faces_inverted"]["faces"],
            "separate_faces_inverted",
        )?
        .len(),
        1
    );
    assert_eq!(
        invalid_array(
            &oracle["operations"]["merge_by_distance"]["vertices"],
            "merge_by_distance",
        )?
        .len(),
        10
    );
    for name in ["delete_points_edge_face", "delete_points_only_face"] {
        assert_eq!(
            invalid_array(&oracle["operations"][name]["faces"], name)?.len(),
            0,
            "{name} removes the selected center-vertex faces"
        );
    }
    for name in ["delete_edges_edge_face", "delete_edges_only_face"] {
        assert_eq!(
            invalid_array(&oracle["operations"][name]["faces"], name)?.len(),
            1,
            "{name} preserves only the face outside the selected shared edge"
        );
    }
    assert_eq!(
        invalid_array(
            &oracle["operations"]["merge_by_distance_connected"]["vertices"],
            "merge_by_distance_connected",
        )?
        .len(),
        11
    );
    assert_eq!(
        invalid_array(
            &oracle["operations"]["dual_mesh_drop_boundaries"]["edges"],
            "dual_mesh_drop_boundaries"
        )?
        .len(),
        0
    );
    assert_eq!(
        invalid_array(
            &oracle["operations"]["dual_mesh_drop_boundaries"]["faces"],
            "dual_mesh_drop_boundaries"
        )?
        .len(),
        0
    );
    for domain in ["vertices", "edges", "faces"] {
        assert_eq!(
            invalid_array(&oracle["operations"]["delete_all_points"][domain], domain)?.len(),
            0,
            "delete_all_points leaves no {domain}"
        );
    }
    let region_vertices = invalid_array(
        &oracle["operations"]["extrude_face_fields"]["vertices"],
        "extrude_face_fields",
    )?
    .len();
    let individual_vertices = invalid_array(
        &oracle["operations"]["extrude_face_individual_fields"]["vertices"],
        "extrude_face_individual_fields",
    )?
    .len();
    assert!(individual_vertices > region_vertices);
    Ok(())
}
#[test]
fn subdivide_mesh_propagates_creases_uv_colors_and_typed_attributes() -> TestResult<()> {
    let oracle = fixture()?;
    let mesh = input_mesh(&oracle)?;
    let result = evaluate(&operation_graph("subdivide_attributes")?, &mesh)?;
    assert_json_close(
        &mesh_snapshot(&result.mesh),
        &oracle["operations"]["subdivide_attributes"],
        "subdivide_attributes",
    )?;

    assert_eq!(result.mesh.attributes["weight"]["data_type"], "FLOAT");
    assert_eq!(
        result.mesh.attributes["vector_tag"]["data_type"],
        "FLOAT_VECTOR"
    );
    assert_eq!(result.mesh.attributes["face_tag"]["data_type"], "INT");
    assert_eq!(result.mesh.attributes["edge_flag"]["data_type"], "BOOLEAN");
    assert_eq!(result.mesh.attributes["crease_edge"]["domain"], "edge");
    assert_eq!(result.mesh.attributes["crease_vert"]["domain"], "point");
    let crease_edges = result.mesh.attributes["crease_edge"]["values"].as_object();
    assert!(crease_edges.is_some_and(|values| {
        values
            .values()
            .any(|value| value.as_f64().is_some_and(|crease| crease > 0.0))
    }));
    let crease_vertices = result.mesh.attributes["crease_vert"]["values"].as_object();
    assert!(crease_vertices.is_some_and(|values| {
        values
            .values()
            .any(|value| value.as_f64().is_some_and(|crease| crease > 0.0))
    }));
    let uv_layers = invalid_array(&result.mesh.attributes["uv_layers"]["layers"], "UV layers")?;
    assert_eq!(uv_layers.len(), 1);
    assert_eq!(uv_layers[0]["name"], "UVMap");
    assert_eq!(
        result.mesh.attributes["color_layers"]["layers"],
        json!(["Paint"])
    );
    assert_eq!(result.mesh.attributes["Paint"]["domain"], "corner");
    assert_eq!(result.mesh.attributes["Paint"]["type"], "color");
    let paint = result.mesh.attributes["Paint"]["values"].as_object();
    assert!(paint.is_some_and(|values| !values.is_empty()));
    Ok(())
}

#[test]
fn extrusion_fields_follow_topology_and_boolean_domain_adaptation() -> TestResult<()> {
    let oracle = fixture()?;
    let mesh = input_mesh(&oracle)?;
    let result = evaluate(&top_side_subdivide_graph()?, &mesh)?;
    assert_json_close(
        &mesh_snapshot(&result.mesh),
        &oracle["top_side_subdivide"],
        "extrude Top and Side through Subdivide Mesh",
    )?;
    assert_boolean_attribute(&result.mesh, "top_after_subdivide", "FACE")?;
    assert_boolean_attribute(&result.mesh, "side_after_subdivide", "FACE")?;
    let result = evaluate(&top_on_original_graph()?, &mesh)?;
    assert_json_close(
        &mesh_snapshot(&result.mesh),
        &oracle["top_on_original"],
        "Extrude Mesh Top on geometry without its anonymous attribute",
    )?;
    assert!(
        invalid_array(
            &result.mesh.attributes["top_on_original"]["data"],
            "top_on_original",
        )?
        .iter()
        .all(|value| value.as_bool() == Some(false)),
        "missing anonymous attribute defaults to false"
    );
    let result = evaluate(&unconsumed_top_distribute_graph()?, &mesh)?;
    assert_json_close(
        &mesh_snapshot(&result.mesh),
        &oracle["unconsumed_top_distribute"],
        "dead Top consumer does not affect Extrude Mesh to Distribute Points",
    )?;
    assert!(
        result
            .mesh
            .attributes
            .get("blender_attributes")
            .and_then(Value::as_array)
            .is_none_or(|layers| layers.iter().all(|layer| {
                layer.get("name").and_then(Value::as_str) != Some("unused_top")
                    && layer.get("potter_gn_anonymous").and_then(Value::as_bool) != Some(true)
            })),
        "unused Top does not create private anonymous layers"
    );

    let isolated: Mesh = serde_json::from_value(oracle["single_vertex_input"].clone())?;
    let result = evaluate(&vertex_top_edge_graph()?, &isolated)?;
    assert_json_close(
        &mesh_snapshot(&result.mesh),
        &oracle["vertex_extrude_top_on_edge"],
        "vertex Extrude Mesh Top adapted to EDGE",
    )?;
    assert_eq!(
        invalid_array(
            &result.mesh.attributes["top_on_edge"]["data"],
            "top_on_edge"
        )?,
        &[json!(false)],
        "POINT→EDGE boolean adaptation requires all endpoints"
    );
    Ok(())
}
#[test]
fn nested_group_top_fields_survive_subdivision_and_are_callsite_scoped() -> TestResult<()> {
    let oracle = fixture()?;
    let mesh = input_mesh(&oracle)?;
    let doc = nested_top_group_doc()?;
    let result = graph::evaluate_with_node_groups(
        &Id::new("geo")?,
        &doc.node_groups,
        Some(&mesh),
        &Map::new(),
        "GNMeshOpsNestedTop",
    )?;
    assert_json_close(
        &mesh_snapshot(&result.mesh),
        &oracle["nested_top_group"],
        "nested Geometry Node Group Top fields through Subdivide Mesh and unused output links",
    )?;
    assert_boolean_attribute(&result.mesh, "top_a", "FACE")?;
    assert_boolean_attribute(&result.mesh, "cross_call_top_b", "FACE")?;
    assert!(
        invalid_array(&result.mesh.attributes["top_a"]["data"], "top_a")?
            .iter()
            .any(|value| value.as_bool() == Some(true)),
        "call A Top selection should survive the nested Subdivide Mesh"
    );
    assert!(
        invalid_array(
            &result.mesh.attributes["cross_call_top_b"]["data"],
            "cross_call_top_b",
        )?
        .iter()
        .all(|value| value.as_bool() == Some(false)),
        "call B Top must not read call A's anonymous attribute"
    );
    assert!(!result.mesh.attributes.contains_key("unused_top"));
    assert!(
        result
            .mesh
            .attributes
            .get("blender_attributes")
            .and_then(Value::as_array)
            .is_none_or(|layers| layers
                .iter()
                .all(|layer| layer.get("potter_gn_anonymous").and_then(Value::as_bool)
                    != Some(true))),
        "unused group output must not leak anonymous layers"
    );
    Ok(())
}


#[test]
fn authored_anonymous_prefix_layer_survives_extrude() -> TestResult<()> {
    let oracle = fixture()?;
    let mut mesh = input_mesh(&oracle)?;
    let vertex_count = mesh.vertices.len();
    let name = ".potter_gn_anonymous_user_layer";
    let layers = mesh
        .attributes
        .entry("blender_attributes".to_owned())
        .or_insert_with(|| json!([]))
        .as_array_mut()
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Blender attribute layers are malformed",
            )
        })?;
    layers.push(json!({
        "name": name,
        "domain": "POINT",
        "data_type": "BOOLEAN",
        "values": vec![json!(false); vertex_count],
    }));

    let result = evaluate(&operation_graph("extrude")?, &mesh)?;
    let output_layers = result
        .mesh
        .attributes
        .get("blender_attributes")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "extrude discarded all Blender attribute layers",
            )
        })?;
    let layer = output_layers
        .iter()
        .find(|layer| layer.get("name").and_then(Value::as_str) == Some(name))
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "extrude removed an authored anonymous-prefix layer",
            )
        })?;
    assert!(layer.get("potter_gn_anonymous").is_none());
    assert_eq!(
        invalid_array(&layer["values"], name)?.len(),
        result.mesh.vertices.len()
    );
    Ok(())
}
#[test]
fn named_attribute_nodes_cannot_access_anonymous_geometry_layers() -> TestResult<()> {
    let oracle = fixture()?;
    let mut mesh = input_mesh(&oracle)?;
    let name = ".potter_gn_anonymous_private";
    let values = vec![json!(true); mesh.vertices.len()];
    mesh.attributes
        .entry("blender_attributes".to_owned())
        .or_insert_with(|| json!([]))
        .as_array_mut()
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Blender attribute layers are malformed",
            )
        })?
        .push(json!({
            "name": name,
            "domain": "POINT",
            "data_type": "BOOLEAN",
            "values": values.clone(),
            "potter_gn_anonymous": true,
        }));
    let doc = graph_with_nodes(
        vec![
            json!({
                "id":"extrude",
                "type":"GeometryNodeExtrudeMesh",
                "properties":{"mode":"FACES"},
                "inputs":{"Offset Scale":1.0,"Individual":false},
            }),
            json!({
                "id":"named_private",
                "type":"GeometryNodeInputNamedAttribute",
                "properties":{"data_type":"BOOLEAN"},
                "inputs":{"Name":name},
            }),
            json!({
                "id":"named_context",
                "type":"GeometryNodeInputNamedAttribute",
                "properties":{"data_type":"BOOLEAN"},
                "inputs":{"Name":".potter_gn_evaluation_context"},
            }),
            json!({
                "id":"store_private",
                "type":"GeometryNodeStoreNamedAttribute",
                "properties":{"domain":"POINT","data_type":"BOOLEAN"},
                "inputs":{"Name":"private_exists"},
            }),
            json!({
                "id":"store_context",
                "type":"GeometryNodeStoreNamedAttribute",
                "properties":{"domain":"POINT","data_type":"BOOLEAN"},
                "inputs":{"Name":"context_exists"},
            }),
            json!({"id":"output","type":"NodeGroupOutput"}),
        ],
        vec![
            json!({"from_node":"extrude","from_socket":"Mesh","to_node":"store_private","to_socket":"Geometry"}),
            json!({"from_node":"named_private","from_socket":"Exists","to_node":"store_private","to_socket":"Value"}),
            json!({"from_node":"store_private","from_socket":"Geometry","to_node":"store_context","to_socket":"Geometry"}),
            json!({"from_node":"named_context","from_socket":"Exists","to_node":"store_context","to_socket":"Value"}),
            json!({"from_node":"store_context","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
        ],
    )?;
    let result = evaluate(&doc, &mesh)?;
    for name in ["private_exists", "context_exists"] {
        let exists = invalid_array(&result.mesh.attributes[name]["data"], name)?;
        assert_eq!(exists.len(), result.mesh.vertices.len());
        assert!(
            exists.iter().all(|value| value.as_bool() == Some(false)),
            "{name} must not expose internal procedural attributes"
        );
    }
    let layer = result.mesh.attributes["blender_attributes"]
        .as_array()
        .and_then(|layers| {
            layers
                .iter()
                .find(|layer| layer.get("name").and_then(Value::as_str) == Some(name))
        })
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "input anonymous layer was discarded",
            )
        })?;
    let private_values = invalid_array(&layer["values"], name)?;
    assert_eq!(private_values.len(), result.mesh.vertices.len());
    assert!(
        private_values.iter().all(|value| value.as_bool() == Some(true)),
        "anonymous input layer must remain attached to the geometry"
    );
    Ok(())
}


#[test]
fn empty_mesh_topology_and_operations_match_blender() -> TestResult<()> {
    let oracle = fixture()?;
    let empty_mesh: Mesh = serde_json::from_value(oracle["empty_input"].clone())?;
    assert_eq!(empty_mesh, Mesh::default());

    let topology = evaluate(&topology_graph()?, &empty_mesh)?;
    assert_json_close(
        &mesh_snapshot(&topology.mesh),
        &oracle["empty_topology_fields"],
        "empty topology fields",
    )?;

    let expected_operations = oracle["empty_operations"].as_object().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "empty operation oracle is not an object",
        )
    })?;
    let mut expected_names = OPERATION_CASES.to_vec();
    expected_names.sort_unstable();
    assert_eq!(
        expected_operations
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        expected_names
    );
    for &name in OPERATION_CASES {
        let result = evaluate(&operation_graph(name)?, &empty_mesh)?;
        assert_json_close(
            &mesh_snapshot(&result.mesh),
            &oracle["empty_operations"][name],
            &format!("empty {name}"),
        )?;
    }
    Ok(())
}

#[test]
fn closed_dual_without_boundaries_preserves_order_and_named_domains() -> TestResult<()> {
    let oracle = fixture()?;
    let input: Mesh = serde_json::from_value(oracle["closed_dual_input"].clone())?;
    assert_eq!(input.vertices.len(), 4);
    assert_eq!(input.edges.len(), 6);
    assert_eq!(input.faces.len(), 4);
    for (name, domain, data_type, count) in [
        ("closed_point_value", "POINT", "FLOAT", 4),
        ("closed_face_tag", "FACE", "INT", 4),
        ("closed_edge_mark", "EDGE", "BOOLEAN", 6),
        ("closed_corner_vector", "CORNER", "FLOAT_VECTOR", 12),
    ] {
        let attribute = &input.attributes[name];
        assert_eq!(attribute["domain"], domain, "{name} input domain");
        assert_eq!(attribute["data_type"], data_type, "{name} input type");
        assert_eq!(
            invalid_array(&attribute["data"], name)?.len(),
            count,
            "{name} input length"
        );
    }
    assert_eq!(
        input.attributes["color_layers"]["layers"],
        json!(["ClosedFacePaint"])
    );
    assert!(input.attributes["color_layers"]["active"].is_null());
    assert!(input.attributes["color_layers"]["render"].is_null());
    assert_eq!(input.attributes["ClosedFacePaint"]["domain"], "face");
    assert_eq!(input.attributes["ClosedFacePaint"]["type"], "color");
    let input_face_colors = input.attributes["ClosedFacePaint"]["values"].as_object();
    assert!(
        input_face_colors
            .is_some_and(|values| { values.len() == 4 && values.get("f0") != values.get("f1") })
    );
    let result = evaluate(&closed_dual_graph()?, &input)?;
    assert_json_close(
        &mesh_snapshot(&result.mesh),
        &oracle["closed_dual"],
        "closed_dual",
    )?;
    assert_eq!(result.mesh.vertices.len(), 4);
    assert_eq!(result.mesh.edges.len(), 6);
    assert_eq!(result.mesh.faces.len(), 4);

    for (name, domain, data_type, count) in [
        ("closed_point_value", "FACE", "FLOAT", 4),
        ("closed_face_tag", "POINT", "INT", 4),
        ("closed_edge_mark", "EDGE", "BOOLEAN", 6),
        ("closed_corner_vector", "CORNER", "FLOAT_VECTOR", 12),
    ] {
        let attribute = &result.mesh.attributes[name];
        assert_eq!(attribute["domain"], domain, "{name} domain");
        assert_eq!(attribute["data_type"], data_type, "{name} type");
        assert_eq!(
            invalid_array(&attribute["data"], name)?.len(),
            count,
            "{name} length"
        );
    }
    assert_eq!(result.mesh.attributes["ClosedFacePaint"]["domain"], "point");
    assert_eq!(result.mesh.attributes["ClosedFacePaint"]["type"], "color");
    assert_eq!(
        result.mesh.attributes["color_layers"]["layers"],
        json!(["ClosedFacePaint"])
    );
    assert_eq!(
        result.mesh.attributes["color_layers"]["active"],
        "ClosedFacePaint"
    );
    assert_eq!(
        result.mesh.attributes["color_layers"]["render"],
        "ClosedFacePaint"
    );
    assert_eq!(
        result.mesh.attributes["blender_metadata"]["color_data_types"]["ClosedFacePaint"],
        "FLOAT_COLOR"
    );
    let transferred_colors = result.mesh.attributes["ClosedFacePaint"]["values"].as_object();
    assert!(transferred_colors.is_some_and(|values| values.len() == 4));
    Ok(())
}
