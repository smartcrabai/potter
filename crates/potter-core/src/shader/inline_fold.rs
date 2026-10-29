use crate::{
    graph::{GraphNode, NodeGroup},
    model::Id,
};
use serde_json::Value;

const SHADER_NODE_MULTI_FUNCTIONS: &[&str] = &[
    "ShaderNodeValue",
    "ShaderNodeMath",
    "ShaderNodeVectorMath",
    "ShaderNodeMapRange",
    "ShaderNodeClamp",
    "ShaderNodeMix",
    "ShaderNodeMixRGB",
    "ShaderNodeValToRGB",
    "ShaderNodeFloatCurve",
    "ShaderNodeVectorCurve",
    "ShaderNodeRGBCurve",
    "ShaderNodeSeparateXYZ",
    "ShaderNodeCombineXYZ",
    "ShaderNodeTexVoronoi",
    "ShaderNodeTexNoise",
    "ShaderNodeTexChecker",
    "ShaderNodeTexGradient",
    "ShaderNodeTexWhiteNoise",
    "ShaderNodeTexMagic",
    "ShaderNodeTexWave",
    "ShaderNodeTexBrick",
    "ShaderNodeTexGabor",
    "ShaderNodeGamma",
    "ShaderNodeVectorRotate",
    "ShaderNodeRadialTiling",
    "ShaderNodeBlackbody",
];

/// Whether this node and its linked inputs fold to primitive values in Blender's shader-tree inliner.
pub(crate) fn node_inputs_all_primitive(graph: &NodeGroup, node_id: &Id) -> bool {
    node_folds_to_primitive(graph, node_id, 0)
}

fn node_folds_to_primitive(graph: &NodeGroup, node_id: &Id, depth: usize) -> bool {
    if depth >= graph.nodes.len() {
        return false;
    }
    let Some(node) = graph.nodes.get(node_id) else {
        return false;
    };
    if !SHADER_NODE_MULTI_FUNCTIONS.contains(&node.node_type.as_str()) {
        return false;
    }
    for link in graph.links.iter().filter(|link| link.to_node == *node_id) {
        if !input_is_available(node, &link.to_socket) {
            continue;
        }
        if !node_folds_to_primitive(graph, &link.from_node, depth + 1) {
            return false;
        }
    }
    if has_unlinked_position_field_vector(graph, node_id, node) {
        return false;
    }
    node.inputs.iter().all(|(socket, value)| {
        !input_is_available(node, socket)
            || is_linked(graph, node_id, socket)
            || primitive_socket_value(value)
    })
}

fn is_linked(graph: &NodeGroup, node_id: &Id, socket: &str) -> bool {
    graph
        .links
        .iter()
        .any(|link| link.to_node == *node_id && link.to_socket == socket)
}

fn primitive_socket_value(value: &Value) -> bool {
    match value {
        Value::Bool(_) | Value::Number(_) | Value::String(_) => true,
        Value::Array(values) => values.iter().all(Value::is_number),
        Value::Null | Value::Object(_) => false,
    }
}

fn has_unlinked_position_field_vector(graph: &NodeGroup, node_id: &Id, node: &GraphNode) -> bool {
    if is_linked(graph, node_id, "Vector") || !input_is_available(node, "Vector") {
        return false;
    }
    matches!(
        node.node_type.as_str(),
        "ShaderNodeTexVoronoi"
            | "ShaderNodeTexNoise"
            | "ShaderNodeTexChecker"
            | "ShaderNodeTexGradient"
            | "ShaderNodeTexWhiteNoise"
            | "ShaderNodeTexMagic"
            | "ShaderNodeTexWave"
            | "ShaderNodeTexBrick"
            | "ShaderNodeTexGabor"
    )
}

fn input_is_available(node: &GraphNode, socket: &str) -> bool {
    let dimensions = match node.node_type.as_str() {
        "ShaderNodeTexVoronoi" => node
            .properties
            .get("voronoi_dimensions")
            .and_then(Value::as_str)
            .unwrap_or("3D"),
        "ShaderNodeTexNoise" | "ShaderNodeTexWhiteNoise" => node
            .properties
            .get("noise_dimensions")
            .and_then(Value::as_str)
            .unwrap_or("3D"),
        _ => "",
    };
    let feature = node
        .properties
        .get("feature")
        .and_then(Value::as_str)
        .unwrap_or("F1");
    let metric = node
        .properties
        .get("distance")
        .and_then(Value::as_str)
        .unwrap_or("EUCLIDEAN");
    let noise_type = node
        .properties
        .get("noise_type")
        .and_then(Value::as_str)
        .unwrap_or("FBM");
    match (node.node_type.as_str(), socket) {
        ("ShaderNodeTexVoronoi" | "ShaderNodeTexNoise" | "ShaderNodeTexWhiteNoise", "Vector") => {
            dimensions != "1D"
        }
        ("ShaderNodeTexVoronoi" | "ShaderNodeTexNoise" | "ShaderNodeTexWhiteNoise", "W") => {
            matches!(dimensions, "1D" | "4D")
        }
        ("ShaderNodeTexVoronoi", "Detail" | "Roughness" | "Lacunarity") => {
            feature != "N_SPHERE_RADIUS"
        }
        ("ShaderNodeTexVoronoi", "Smoothness") => feature == "SMOOTH_F1",
        ("ShaderNodeTexVoronoi", "Exponent") => {
            metric == "MINKOWSKI"
                && dimensions != "1D"
                && !matches!(feature, "DISTANCE_TO_EDGE" | "N_SPHERE_RADIUS")
        }
        ("ShaderNodeTexNoise", "Offset") => !matches!(noise_type, "MULTIFRACTAL" | "FBM"),
        ("ShaderNodeTexNoise", "Gain") => {
            matches!(noise_type, "HYBRID_MULTIFRACTAL" | "RIDGED_MULTIFRACTAL")
        }
        _ => true,
    }
}
