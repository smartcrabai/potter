use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use crate::{
    graph::{GraphKind, GraphNode, NodeGroup},
    model::{Id, Registry},
};
use serde_json::Value;

const MAX_GROUP_DEPTH: usize = 64;

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

/// The per-invocation shader links and primitive folds prepared with a material.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct FoldIndex {
    graph: Option<Id>,
    pub(super) incoming: Arc<BTreeMap<Id, Vec<usize>>>,
    pub(super) primitive_nodes: BTreeSet<Id>,
    pub(super) children: BTreeMap<Id, FoldIndex>,
}

impl FoldIndex {
    pub(super) fn new(graph_id: &Id, graph: &NodeGroup, node_groups: &Registry<NodeGroup>) -> Self {
        let mut incoming_by_graph = BTreeMap::new();
        let mut active_groups = BTreeSet::from([graph_id.clone()]);
        let mut index = Self::build(
            graph_id,
            graph,
            node_groups,
            &mut incoming_by_graph,
            &mut active_groups,
            0,
        );
        let mut resolver = FoldResolver::default();
        resolver.prepare(&mut index, graph, node_groups);
        index
    }

    fn build(
        graph_id: &Id,
        graph: &NodeGroup,
        node_groups: &Registry<NodeGroup>,
        incoming_by_graph: &mut BTreeMap<Id, Arc<BTreeMap<Id, Vec<usize>>>>,
        active_groups: &mut BTreeSet<Id>,
        depth: usize,
    ) -> Self {
        let incoming = Arc::clone(
            incoming_by_graph
                .entry(graph_id.clone())
                .or_insert_with(|| {
                    let mut incoming = BTreeMap::<Id, Vec<usize>>::new();
                    for (link_index, link) in graph.links.iter().enumerate() {
                        incoming
                            .entry(link.to_node.clone())
                            .or_default()
                            .push(link_index);
                    }
                    Arc::new(incoming)
                }),
        );
        let mut children = BTreeMap::new();
        if depth < MAX_GROUP_DEPTH {
            for (node_id, node) in &graph.nodes {
                if node.node_type != "ShaderNodeGroup" {
                    continue;
                }
                let Some(child_id) = node.properties.get("node_tree").and_then(Value::as_str)
                else {
                    continue;
                };
                let Ok(child_id) = Id::new(child_id.to_owned()) else {
                    continue;
                };
                let Some(child_graph) = node_groups.get(child_id.as_str()) else {
                    continue;
                };
                if child_graph.kind != GraphKind::Shader || !active_groups.insert(child_id.clone())
                {
                    continue;
                }
                let child = Self::build(
                    &child_id,
                    child_graph,
                    node_groups,
                    incoming_by_graph,
                    active_groups,
                    depth + 1,
                );
                active_groups.remove(&child_id);
                children.insert(node_id.clone(), child);
            }
        }
        Self {
            graph: Some(graph_id.clone()),
            incoming,
            primitive_nodes: BTreeSet::new(),
            children,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct InvocationKey(Vec<(Id, Id)>);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct OutputKey(InvocationKey, Id, String);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct NodeKey(InvocationKey, Id);

#[derive(Default)]
struct FoldResolver {
    outputs: BTreeMap<OutputKey, bool>,
    nodes: BTreeMap<NodeKey, bool>,
    active_outputs: BTreeSet<OutputKey>,
    active_nodes: BTreeSet<NodeKey>,
}

impl FoldResolver {
    fn prepare(
        &mut self,
        root: &mut FoldIndex,
        root_graph: &NodeGroup,
        node_groups: &Registry<NodeGroup>,
    ) {
        let mut invocations = Vec::new();
        collect_invocations(root, &InvocationKey::default(), &mut invocations);
        let mut primitive_nodes = BTreeMap::<InvocationKey, BTreeSet<Id>>::new();
        for invocation in invocations {
            let Some(graph) = graph_at(root, root_graph, node_groups, &invocation) else {
                continue;
            };
            for (node_id, node) in &graph.nodes {
                if SHADER_NODE_MULTI_FUNCTIONS.contains(&node.node_type.as_str())
                    && self.node_inputs_are_primitive(
                        root,
                        root_graph,
                        node_groups,
                        &invocation,
                        node_id,
                    )
                {
                    primitive_nodes
                        .entry(invocation.clone())
                        .or_default()
                        .insert(node_id.clone());
                }
            }
        }
        assign_primitive_nodes(root, &InvocationKey::default(), &mut primitive_nodes);
    }

    fn node_inputs_are_primitive(
        &mut self,
        root: &FoldIndex,
        root_graph: &NodeGroup,
        node_groups: &Registry<NodeGroup>,
        invocation: &InvocationKey,
        node_id: &Id,
    ) -> bool {
        let node_key = NodeKey(invocation.clone(), node_id.clone());
        if let Some(result) = self.nodes.get(&node_key) {
            return *result;
        }
        if !self.active_nodes.insert(node_key.clone()) {
            return false;
        }
        let result = (|| {
            let Some(index) = index_at(root, invocation) else {
                return false;
            };
            let Some(graph) = graph_at(root, root_graph, node_groups, invocation) else {
                return false;
            };
            let Some(node) = graph.nodes.get(node_id) else {
                return false;
            };
            if !SHADER_NODE_MULTI_FUNCTIONS.contains(&node.node_type.as_str())
                || has_unlinked_position_field_vector(index, graph, node_id, node)
            {
                return false;
            }
            if let Some(links) = index.incoming.get(node_id) {
                for &link_index in links {
                    let Some(link) = graph.links.get(link_index) else {
                        continue;
                    };
                    if link.to_node == *node_id
                        && input_is_available(node, &link.to_socket)
                        && !self.output_is_primitive(
                            root,
                            root_graph,
                            node_groups,
                            invocation,
                            &link.from_node,
                            &link.from_socket,
                        )
                    {
                        return false;
                    }
                }
            }
            node.inputs.iter().all(|(socket, value)| {
                !input_is_available(node, socket)
                    || is_linked_index(index, graph, node_id, socket)
                    || primitive_socket_value(value)
            })
        })();
        self.active_nodes.remove(&node_key);
        self.nodes.insert(node_key, result);
        result
    }

    fn output_is_primitive(
        &mut self,
        root: &FoldIndex,
        root_graph: &NodeGroup,
        node_groups: &Registry<NodeGroup>,
        invocation: &InvocationKey,
        node_id: &Id,
        socket: &str,
    ) -> bool {
        let output_key = OutputKey(invocation.clone(), node_id.clone(), socket.to_owned());
        if let Some(result) = self.outputs.get(&output_key) {
            return *result;
        }
        if !self.active_outputs.insert(output_key.clone()) {
            return false;
        }
        let result = (|| {
            let Some(graph) = graph_at(root, root_graph, node_groups, invocation) else {
                return false;
            };
            let Some(node) = graph.nodes.get(node_id) else {
                return false;
            };
            match node.node_type.as_str() {
                "NodeGroupInput" => self.group_input_output_is_primitive(
                    root,
                    root_graph,
                    node_groups,
                    invocation,
                    socket,
                ),
                "ShaderNodeGroup" => self.group_node_output_is_primitive(
                    root,
                    root_graph,
                    node_groups,
                    invocation,
                    node_id,
                    socket,
                ),
                _ => {
                    SHADER_NODE_MULTI_FUNCTIONS.contains(&node.node_type.as_str())
                        && self.node_inputs_are_primitive(
                            root,
                            root_graph,
                            node_groups,
                            invocation,
                            node_id,
                        )
                }
            }
        })();
        self.active_outputs.remove(&output_key);
        self.outputs.insert(output_key, result);
        result
    }

    fn group_input_output_is_primitive(
        &mut self,
        root: &FoldIndex,
        root_graph: &NodeGroup,
        node_groups: &Registry<NodeGroup>,
        invocation: &InvocationKey,
        socket: &str,
    ) -> bool {
        let Some(graph) = graph_at(root, root_graph, node_groups, invocation) else {
            return false;
        };
        let Some(input_socket) = graph
            .interface
            .inputs
            .iter()
            .find(|candidate| candidate.id == socket || candidate.name == socket)
        else {
            return false;
        };
        let Some((parent_graph_id, parent_node_id)) = invocation.0.last() else {
            return false;
        };
        let parent_invocation = InvocationKey(invocation.0[..invocation.0.len() - 1].to_vec());
        let Some(parent_index) = index_at(root, &parent_invocation) else {
            return false;
        };
        let Some(parent_graph) = graph_at(root, root_graph, node_groups, &parent_invocation) else {
            return false;
        };
        if parent_index.graph.as_ref() != Some(parent_graph_id) {
            return false;
        }
        let Some(parent_node) = parent_graph.nodes.get(parent_node_id) else {
            return false;
        };
        if let Some(links) = parent_index.incoming.get(parent_node_id) {
            for &link_index in links {
                let Some(link) = parent_graph.links.get(link_index) else {
                    continue;
                };
                if link.to_node == *parent_node_id
                    && socket_matches(&link.to_socket, &input_socket.id, &input_socket.name)
                {
                    return self.output_is_primitive(
                        root,
                        root_graph,
                        node_groups,
                        &parent_invocation,
                        &link.from_node,
                        &link.from_socket,
                    );
                }
            }
        }
        parent_node
            .inputs
            .iter()
            .find(|(name, value)| {
                !value.is_null() && socket_matches(name, &input_socket.id, &input_socket.name)
            })
            .map_or_else(
                || primitive_socket_value(&input_socket.default),
                |(_, value)| primitive_socket_value(value),
            )
    }

    fn group_node_output_is_primitive(
        &mut self,
        root: &FoldIndex,
        root_graph: &NodeGroup,
        node_groups: &Registry<NodeGroup>,
        invocation: &InvocationKey,
        node_id: &Id,
        socket: &str,
    ) -> bool {
        let Some(parent_index) = index_at(root, invocation) else {
            return false;
        };
        let Some(child_index) = parent_index.children.get(node_id) else {
            return false;
        };
        let Some(child_graph) = graph_at(
            root,
            root_graph,
            node_groups,
            &append_invocation(invocation, parent_index, node_id),
        ) else {
            return false;
        };
        let Some(output_socket) = child_graph
            .interface
            .outputs
            .iter()
            .find(|candidate| candidate.id == socket || candidate.name == socket)
        else {
            return false;
        };
        let child_invocation = append_invocation(invocation, parent_index, node_id);
        let Some(output_node_id) = active_group_output(child_graph) else {
            return false;
        };
        let Some(output_node) = child_graph.nodes.get(output_node_id) else {
            return false;
        };
        if let Some(links) = child_index.incoming.get(output_node_id) {
            for &link_index in links {
                let Some(link) = child_graph.links.get(link_index) else {
                    continue;
                };
                if link.to_node == *output_node_id
                    && socket_matches(&link.to_socket, &output_socket.id, &output_socket.name)
                {
                    return self.output_is_primitive(
                        root,
                        root_graph,
                        node_groups,
                        &child_invocation,
                        &link.from_node,
                        &link.from_socket,
                    );
                }
            }
        }
        output_node
            .inputs
            .iter()
            .find(|(name, value)| {
                !value.is_null() && socket_matches(name, &output_socket.id, &output_socket.name)
            })
            .is_some_and(|(_, value)| primitive_socket_value(value))
    }
}

fn collect_invocations(
    index: &FoldIndex,
    invocation: &InvocationKey,
    output: &mut Vec<InvocationKey>,
) {
    output.push(invocation.clone());
    for (node_id, child) in &index.children {
        let child_invocation = append_invocation(invocation, index, node_id);
        collect_invocations(child, &child_invocation, output);
    }
}

fn append_invocation(
    invocation: &InvocationKey,
    parent: &FoldIndex,
    node_id: &Id,
) -> InvocationKey {
    let mut child = invocation.clone();
    if let Some(graph_id) = &parent.graph {
        child.0.push((graph_id.clone(), node_id.clone()));
    }
    child
}

fn index_at<'a>(root: &'a FoldIndex, invocation: &InvocationKey) -> Option<&'a FoldIndex> {
    let mut index = root;
    for (graph_id, node_id) in &invocation.0 {
        if index.graph.as_ref() != Some(graph_id) {
            return None;
        }
        index = index.children.get(node_id)?;
    }
    Some(index)
}

fn graph_at<'a>(
    root: &FoldIndex,
    root_graph: &'a NodeGroup,
    node_groups: &'a Registry<NodeGroup>,
    invocation: &InvocationKey,
) -> Option<&'a NodeGroup> {
    let index = index_at(root, invocation)?;
    if invocation.0.is_empty() {
        Some(root_graph)
    } else {
        node_groups.get(index.graph.as_ref()?.as_str())
    }
}

fn assign_primitive_nodes(
    index: &mut FoldIndex,
    invocation: &InvocationKey,
    primitive_nodes: &mut BTreeMap<InvocationKey, BTreeSet<Id>>,
) {
    index.primitive_nodes = primitive_nodes.remove(invocation).unwrap_or_default();
    let child_nodes = index.children.keys().cloned().collect::<Vec<_>>();
    for node_id in child_nodes {
        let child_invocation = append_invocation(invocation, index, &node_id);
        if let Some(child) = index.children.get_mut(&node_id) {
            assign_primitive_nodes(child, &child_invocation, primitive_nodes);
        }
    }
}

/// Whether this node and its linked inputs fold to primitive values in Blender's shader-tree inliner.
pub(crate) fn node_inputs_all_primitive(graph: &NodeGroup, node_id: &Id) -> bool {
    let incoming = incoming_index(graph);
    let mut memo = BTreeMap::new();
    direct_node_folds(graph, node_id, &incoming, &mut memo, &mut BTreeSet::new())
}

fn incoming_index(graph: &NodeGroup) -> BTreeMap<Id, Vec<usize>> {
    let mut incoming = BTreeMap::<Id, Vec<usize>>::new();
    for (link_index, link) in graph.links.iter().enumerate() {
        incoming
            .entry(link.to_node.clone())
            .or_default()
            .push(link_index);
    }
    incoming
}

fn direct_node_folds(
    graph: &NodeGroup,
    node_id: &Id,
    incoming: &BTreeMap<Id, Vec<usize>>,
    memo: &mut BTreeMap<Id, bool>,
    active: &mut BTreeSet<Id>,
) -> bool {
    if let Some(result) = memo.get(node_id) {
        return *result;
    }
    if !active.insert(node_id.clone()) {
        return false;
    }
    let result = (|| {
        let Some(node) = graph.nodes.get(node_id) else {
            return false;
        };
        if !SHADER_NODE_MULTI_FUNCTIONS.contains(&node.node_type.as_str())
            || has_unlinked_position_field_vector_indexed(incoming, graph, node_id, node)
        {
            return false;
        }
        if let Some(links) = incoming.get(node_id) {
            for &link_index in links {
                let Some(link) = graph.links.get(link_index) else {
                    continue;
                };
                if link.to_node == *node_id
                    && input_is_available(node, &link.to_socket)
                    && !direct_node_folds(graph, &link.from_node, incoming, memo, active)
                {
                    return false;
                }
            }
        }
        node.inputs.iter().all(|(socket, value)| {
            !input_is_available(node, socket)
                || is_linked_incoming(incoming, graph, node_id, socket)
                || primitive_socket_value(value)
        })
    })();
    active.remove(node_id);
    memo.insert(node_id.clone(), result);
    result
}

fn is_linked_index(index: &FoldIndex, graph: &NodeGroup, node_id: &Id, socket: &str) -> bool {
    index
        .incoming
        .get(node_id)
        .into_iter()
        .flatten()
        .filter_map(|&link_index| graph.links.get(link_index))
        .any(|link| link.to_node == *node_id && link.to_socket == socket)
}

fn has_unlinked_position_field_vector(
    index: &FoldIndex,
    graph: &NodeGroup,
    node_id: &Id,
    node: &GraphNode,
) -> bool {
    has_unlinked_position_field_vector_indexed(&index.incoming, graph, node_id, node)
}

fn has_unlinked_position_field_vector_indexed(
    incoming: &BTreeMap<Id, Vec<usize>>,
    graph: &NodeGroup,
    node_id: &Id,
    node: &GraphNode,
) -> bool {
    if is_linked_incoming(incoming, graph, node_id, "Vector") || !input_is_available(node, "Vector")
    {
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

fn is_linked_incoming(
    incoming: &BTreeMap<Id, Vec<usize>>,
    graph: &NodeGroup,
    node_id: &Id,
    socket: &str,
) -> bool {
    incoming
        .get(node_id)
        .into_iter()
        .flatten()
        .filter_map(|&link_index| graph.links.get(link_index))
        .any(|link| link.to_node == *node_id && link.to_socket == socket)
}

fn primitive_socket_value(value: &Value) -> bool {
    match value {
        Value::Bool(_) | Value::Number(_) | Value::String(_) => true,
        Value::Array(values) => values.iter().all(Value::is_number),
        Value::Null | Value::Object(_) => false,
    }
}

fn active_group_output(group: &NodeGroup) -> Option<&Id> {
    let mut fallback = None;
    for (id, node) in &group.nodes {
        if node.node_type != "NodeGroupOutput" {
            continue;
        }
        if node
            .properties
            .get("is_active_output")
            .and_then(Value::as_bool)
            == Some(true)
        {
            return Some(id);
        }
        fallback.get_or_insert(id);
    }
    fallback
}

fn socket_matches(socket: &str, id: &str, name: &str) -> bool {
    socket == id || socket == name
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

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "fold index unit test setup")]
    use super::*;
    use crate::graph::{GraphLink, GraphSocket};
    use serde_json::json;

    fn grouped_math_folds(default: Value) -> bool {
        let root_id = Id::new("fold_root".to_owned()).unwrap();
        let child_id = Id::new("fold_child".to_owned()).unwrap();
        let mut child = NodeGroup::new("Fold Child", GraphKind::Shader);
        child.interface.inputs.push(GraphSocket {
            id: "input".to_owned(),
            name: "Input".to_owned(),
            socket_type: "NodeSocketFloat".to_owned(),
            default,
        });
        child.nodes.insert(
            Id::new("input_node".to_owned()).unwrap(),
            GraphNode::new("NodeGroupInput"),
        );
        let mut math = GraphNode::new("ShaderNodeMath");
        math.properties.insert("operation".to_owned(), json!("ADD"));
        math.inputs.insert("Value_001".to_owned(), json!(0.5));
        child
            .nodes
            .insert(Id::new("math".to_owned()).unwrap(), math);
        child.links.push(GraphLink {
            from_node: Id::new("input_node".to_owned()).unwrap(),
            from_socket: "Input".to_owned(),
            to_node: Id::new("math".to_owned()).unwrap(),
            to_socket: "Value".to_owned(),
        });

        let mut root = NodeGroup::new("Fold Root", GraphKind::Shader);
        let mut call = GraphNode::new("ShaderNodeGroup");
        call.properties
            .insert("node_tree".to_owned(), json!(child_id.as_str()));
        root.nodes.insert(Id::new("call".to_owned()).unwrap(), call);
        let mut node_groups = Registry::new();
        node_groups.insert(root_id.clone(), root.clone());
        node_groups.insert(child_id, child);

        let folds = FoldIndex::new(&root_id, &root, &node_groups);
        folds
            .children
            .get(&Id::new("call".to_owned()).unwrap())
            .is_some_and(|child| {
                child
                    .primitive_nodes
                    .contains(&Id::new("math".to_owned()).unwrap())
            })
    }

    #[test]
    fn group_interface_primitive_defaults_fold_but_field_defaults_do_not() {
        assert!(grouped_math_folds(json!(0.25)));
        assert!(!grouped_math_folds(json!({"field":"position"})));
    }
}
