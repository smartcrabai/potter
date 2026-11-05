mod bake;
mod evaluator;
mod model;
mod rotation_matrix;
mod sockets;

pub(crate) use bake::{
    MAX_NODE_BAKE_FRAME_COUNT, MAX_NODE_BAKE_PACKED_BYTES, NODE_BAKE_JSON_MEMORY_MULTIPLIER,
    bake_frame_window, node_bake_directory, node_bake_frame_memory_bytes,
    packed_node_bake_resource, read_node_bake_payload_limited, same_bake_frame,
    uses_custom_node_bake_path, validate_node_bake_config, validate_node_bake_frame,
    validate_node_bake_record, validate_node_bake_resource, validate_node_bake_settings,
};
pub use bake::{NodeBakeFrame, NodeBakePayloadTransaction, NodeBakeRecord, NodeBakeValue};
pub(crate) use evaluator::{EvaluatedInstances, evaluate_at_frame_with_bakes_and_instances};
pub use evaluator::{
    GraphEvaluation, evaluate, evaluate_at_frame, evaluate_at_frame_with_bakes,
    evaluate_with_node_groups,
};
pub use model::{GraphInterface, GraphKind, GraphLink, GraphNode, GraphSocket, NodeGroup};
pub use sockets::{validate_link, validate_link_with_groups};

fn is_intrinsic_shader_socket_metadata(
    properties: &serde_json::Map<String, serde_json::Value>,
    allow_simple_material_marker: bool,
) -> bool {
    properties.iter().all(|(key, value)| match key.as_str() {
        "potter_simple_material" => {
            allow_simple_material_marker && value == &serde_json::Value::Bool(true)
        }
        "internal_links" => value.as_array().is_some_and(Vec::is_empty),
        "input_socket_flags" => value.as_object().is_some_and(|flags| {
            flags.iter().all(|(socket, value)| {
                !socket.is_empty()
                    && value.as_object().is_some_and(|flags| {
                        flags.len() == 2
                            && flags
                                .get("enabled")
                                .and_then(serde_json::Value::as_bool)
                                .is_some()
                            && flags
                                .get("hide_value")
                                .and_then(serde_json::Value::as_bool)
                                .is_some()
                    })
            })
        }),
        _ => false,
    })
}

pub(crate) fn is_generated_simple_material_graph(group: &NodeGroup) -> bool {
    if group.kind != GraphKind::Shader
        || !group.interface.inputs.is_empty()
        || !group.interface.outputs.is_empty()
        || group.nodes.len() != 2
        || group.links.len() != 1
    {
        return false;
    }
    let principled = group
        .nodes
        .iter()
        .find(|(id, _)| id.as_str() == "principled");
    let output = group.nodes.iter().find(|(id, _)| id.as_str() == "output");
    let Some((_, principled)) = principled else {
        return false;
    };
    let Some((_, output)) = output else {
        return false;
    };
    let link = &group.links[0];
    principled.node_type == "ShaderNodeBsdfPrincipled"
        && principled.name == "ShaderNodeBsdfPrincipled"
        && principled.location.map(f64::to_bits) == [0; 2]
        && principled.inputs.is_empty()
        && principled.properties.get("potter_simple_material")
            == Some(&serde_json::Value::Bool(true))
        && is_intrinsic_shader_socket_metadata(&principled.properties, true)
        && output.node_type == "OutputMaterial"
        && output.name == "OutputMaterial"
        && output.location.map(f64::to_bits) == [0; 2]
        && output.inputs.is_empty()
        && is_intrinsic_shader_socket_metadata(&output.properties, false)
        && link.from_node.as_str() == "principled"
        && link.from_socket == "BSDF"
        && link.to_node.as_str() == "output"
        && link.to_socket == "Surface"
}
