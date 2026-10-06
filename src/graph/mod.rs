mod evaluator;
mod model;
mod sockets;

pub use evaluator::{GraphEvaluation, evaluate, evaluate_at_frame};
pub use model::{GraphInterface, GraphKind, GraphLink, GraphNode, GraphSocket, NodeGroup};
pub use sockets::validate_link;

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
        && principled.properties.len() == 1
        && principled.properties.get("potter_simple_material")
            == Some(&serde_json::Value::Bool(true))
        && output.node_type == "OutputMaterial"
        && output.name == "OutputMaterial"
        && output.location.map(f64::to_bits) == [0; 2]
        && output.inputs.is_empty()
        && output.properties.is_empty()
        && link.from_node.as_str() == "principled"
        && link.from_socket == "BSDF"
        && link.to_node.as_str() == "output"
        && link.to_socket == "Surface"
}
