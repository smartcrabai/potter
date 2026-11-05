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

type UnknownFoldSources = BTreeMap<Id, BTreeMap<String, Id>>;

/// The per-invocation shader links and primitive folds prepared with a material.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct FoldIndex {
    graph: Option<Id>,
    pub(super) incoming: Arc<BTreeMap<Id, Vec<usize>>>,
    pub(super) primitive_nodes: BTreeSet<Id>,
    pub(super) primitive_outputs: BTreeSet<(Id, String)>,
    pub(super) dangling_outputs: BTreeMap<Id, BTreeSet<String>>,
    pub(super) unknown_outputs: BTreeSet<(Id, String)>,
    unknown_fold_sources: UnknownFoldSources,
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
    pub(super) fn unknown_fold_source(&self, node_id: &Id, socket: &str) -> Option<&Id> {
        self.unknown_fold_sources.get(node_id)?.get(socket)
    }

    pub(super) fn has_unlinked_position_field_vector(
        &self,
        graph: &NodeGroup,
        node_id: &Id,
        node: &GraphNode,
    ) -> bool {
        has_unlinked_position_field_vector(self, graph, node_id, node)
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
            primitive_outputs: BTreeSet::new(),
            dangling_outputs: BTreeMap::new(),
            unknown_outputs: BTreeSet::new(),
            unknown_fold_sources: BTreeMap::new(),
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
#[derive(Clone, Debug, PartialEq, Eq)]
enum FoldOutputState {
    Primitive,
    Dangling,
    Dynamic,
    Unknown(Id),
}

#[derive(Default)]
struct FoldResolver {
    outputs: BTreeMap<OutputKey, FoldOutputState>,
    nodes: BTreeMap<NodeKey, FoldOutputState>,
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
        let mut primitive_outputs = BTreeMap::<InvocationKey, BTreeSet<(Id, String)>>::new();
        let mut dangling_outputs = BTreeMap::<InvocationKey, BTreeMap<Id, BTreeSet<String>>>::new();
        let mut unknown_outputs = BTreeMap::<InvocationKey, BTreeSet<(Id, String)>>::new();
        let mut unknown_fold_sources = BTreeMap::<InvocationKey, UnknownFoldSources>::new();
        for invocation in invocations {
            let Some(graph) = graph_at(root, root_graph, node_groups, &invocation) else {
                continue;
            };
            for (node_id, node) in &graph.nodes {
                if SHADER_NODE_MULTI_FUNCTIONS.contains(&node.node_type.as_str())
                    && self.node_inputs_state(root, root_graph, node_groups, &invocation, node_id)
                        == FoldOutputState::Primitive
                {
                    primitive_nodes
                        .entry(invocation.clone())
                        .or_default()
                        .insert(node_id.clone());
                }
            }
            for link in &graph.links {
                let output = self.output_state(
                    root,
                    root_graph,
                    node_groups,
                    &invocation,
                    &link.from_node,
                    &link.from_socket,
                );
                match output {
                    FoldOutputState::Primitive => {
                        primitive_outputs
                            .entry(invocation.clone())
                            .or_default()
                            .insert((link.from_node.clone(), link.from_socket.clone()));
                    }
                    FoldOutputState::Dangling => {
                        dangling_outputs
                            .entry(invocation.clone())
                            .or_default()
                            .entry(link.from_node.clone())
                            .or_default()
                            .insert(link.from_socket.clone());
                    }
                    FoldOutputState::Unknown(source_id) => {
                        unknown_fold_sources
                            .entry(invocation.clone())
                            .or_default()
                            .entry(link.from_node.clone())
                            .or_default()
                            .insert(link.from_socket.clone(), source_id.clone());
                        if source_id == link.from_node
                            || graph.nodes.get(&link.from_node).is_some_and(|node| {
                                node.node_type == "NodeReroute"
                                    || node.properties.get("mute").and_then(Value::as_bool)
                                        == Some(true)
                                    || matches!(
                                        node.node_type.as_str(),
                                        "NodeGroupInput" | "ShaderNodeGroup"
                                    )
                            })
                        {
                            unknown_outputs
                                .entry(invocation.clone())
                                .or_default()
                                .insert((link.from_node.clone(), link.from_socket.clone()));
                        }
                    }
                    FoldOutputState::Dynamic => {}
                }
            }
        }
        assign_primitive_nodes(
            root,
            &InvocationKey::default(),
            &mut primitive_nodes,
            &mut primitive_outputs,
            &mut dangling_outputs,
            &mut unknown_outputs,
            &mut unknown_fold_sources,
        );
    }

    fn node_inputs_state(
        &mut self,
        root: &FoldIndex,
        root_graph: &NodeGroup,
        node_groups: &Registry<NodeGroup>,
        invocation: &InvocationKey,
        node_id: &Id,
    ) -> FoldOutputState {
        let node_key = NodeKey(invocation.clone(), node_id.clone());
        if let Some(result) = self.nodes.get(&node_key) {
            return result.clone();
        }
        if !self.active_nodes.insert(node_key.clone()) {
            return FoldOutputState::Dynamic;
        }
        let result = (|| {
            let Some(index) = index_at(root, invocation) else {
                return FoldOutputState::Dynamic;
            };
            let Some(graph) = graph_at(root, root_graph, node_groups, invocation) else {
                return FoldOutputState::Dynamic;
            };
            let Some(node) = graph.nodes.get(node_id) else {
                return FoldOutputState::Dynamic;
            };
            if !SHADER_NODE_MULTI_FUNCTIONS.contains(&node.node_type.as_str())
                || has_unlinked_position_field_vector(index, graph, node_id, node)
            {
                return FoldOutputState::Dynamic;
            }
            let mut unknown_source = None;
            if let Some(links) = index.incoming.get(node_id) {
                for &link_index in links {
                    let Some(link) = graph.links.get(link_index) else {
                        continue;
                    };
                    if link.to_node != *node_id || !input_is_available(node, &link.to_socket) {
                        continue;
                    }
                    match self.output_state(
                        root,
                        root_graph,
                        node_groups,
                        invocation,
                        &link.from_node,
                        &link.from_socket,
                    ) {
                        FoldOutputState::Primitive => {}
                        FoldOutputState::Dangling
                            if node
                                .inputs
                                .get(&link.to_socket)
                                .is_some_and(primitive_socket_value) => {}
                        FoldOutputState::Unknown(source_id) => {
                            unknown_source.get_or_insert(source_id);
                        }
                        FoldOutputState::Dangling | FoldOutputState::Dynamic => {
                            return FoldOutputState::Dynamic;
                        }
                    }
                }
            }
            if !node.inputs.iter().all(|(socket, value)| {
                !input_is_available(node, socket)
                    || is_linked_index(index, graph, node_id, socket)
                    || primitive_socket_value(value)
            }) {
                return FoldOutputState::Dynamic;
            }
            unknown_source.map_or(FoldOutputState::Primitive, FoldOutputState::Unknown)
        })();
        self.active_nodes.remove(&node_key);
        self.nodes.insert(node_key, result.clone());
        result
    }

    fn output_state(
        &mut self,
        root: &FoldIndex,
        root_graph: &NodeGroup,
        node_groups: &Registry<NodeGroup>,
        invocation: &InvocationKey,
        node_id: &Id,
        socket: &str,
    ) -> FoldOutputState {
        let output_key = OutputKey(invocation.clone(), node_id.clone(), socket.to_owned());
        if let Some(result) = self.outputs.get(&output_key) {
            return result.clone();
        }
        if !self.active_outputs.insert(output_key.clone()) {
            return FoldOutputState::Dynamic;
        }
        let result = (|| {
            let Some(graph) = graph_at(root, root_graph, node_groups, invocation) else {
                return FoldOutputState::Dynamic;
            };
            let Some(node) = graph.nodes.get(node_id) else {
                return FoldOutputState::Dynamic;
            };
            if node.node_type == "NodeReroute"
                || node.properties.get("mute").and_then(Value::as_bool) == Some(true)
            {
                return self.passthrough_output_state(
                    root,
                    root_graph,
                    node_groups,
                    invocation,
                    node_id,
                    node,
                    socket,
                );
            }
            match node.node_type.as_str() {
                "NodeGroupInput" => {
                    self.group_input_output_state(root, root_graph, node_groups, invocation, socket)
                }
                "ShaderNodeGroup" => self.group_node_output_state(
                    root,
                    root_graph,
                    node_groups,
                    invocation,
                    node_id,
                    socket,
                ),
                _ if SHADER_NODE_MULTI_FUNCTIONS.contains(&node.node_type.as_str()) => {
                    self.node_inputs_state(root, root_graph, node_groups, invocation, node_id)
                }
                _ => FoldOutputState::Dynamic,
            }
        })();
        self.active_outputs.remove(&output_key);
        self.outputs.insert(output_key, result.clone());
        result
    }

    fn passthrough_output_state(
        &mut self,
        root: &FoldIndex,
        root_graph: &NodeGroup,
        node_groups: &Registry<NodeGroup>,
        invocation: &InvocationKey,
        node_id: &Id,
        node: &GraphNode,
        output_socket: &str,
    ) -> FoldOutputState {
        let Some(index) = index_at(root, invocation) else {
            return FoldOutputState::Dynamic;
        };
        let Some(graph) = graph_at(root, root_graph, node_groups, invocation) else {
            return FoldOutputState::Dynamic;
        };
        if source_muted_route_is_unknown(node, output_socket, graph, node_id) {
            return FoldOutputState::Unknown(node_id.clone());
        }
        let Some(input_socket) =
            source_passthrough_input_socket(node, output_socket, graph, node_id)
        else {
            return FoldOutputState::Dangling;
        };
        if let Some(link) = index
            .incoming
            .get(node_id)
            .into_iter()
            .flatten()
            .filter_map(|&link_index| graph.links.get(link_index))
            .find(|link| {
                link.to_node == *node_id
                    && source_input_socket_matches(node, &link.to_socket, input_socket)
            })
        {
            return self.output_state(
                root,
                root_graph,
                node_groups,
                invocation,
                &link.from_node,
                &link.from_socket,
            );
        }
        FoldOutputState::Dangling
    }

    fn group_input_output_state(
        &mut self,
        root: &FoldIndex,
        root_graph: &NodeGroup,
        node_groups: &Registry<NodeGroup>,
        invocation: &InvocationKey,
        socket: &str,
    ) -> FoldOutputState {
        let Some(graph) = graph_at(root, root_graph, node_groups, invocation) else {
            return FoldOutputState::Dynamic;
        };
        let Some(input_socket) = graph
            .interface
            .inputs
            .iter()
            .find(|candidate| candidate.id == socket || candidate.name == socket)
        else {
            return FoldOutputState::Dynamic;
        };
        let Some((parent_graph_id, parent_node_id)) = invocation.0.last() else {
            return FoldOutputState::Dynamic;
        };
        let parent_invocation = InvocationKey(invocation.0[..invocation.0.len() - 1].to_vec());
        let Some(parent_index) = index_at(root, &parent_invocation) else {
            return FoldOutputState::Dynamic;
        };
        let Some(parent_graph) = graph_at(root, root_graph, node_groups, &parent_invocation) else {
            return FoldOutputState::Dynamic;
        };
        if parent_index.graph.as_ref() != Some(parent_graph_id) {
            return FoldOutputState::Dynamic;
        }
        let Some(parent_node) = parent_graph.nodes.get(parent_node_id) else {
            return FoldOutputState::Dynamic;
        };
        let default_value = parent_node
            .inputs
            .iter()
            .find(|(name, value)| {
                !value.is_null() && socket_matches(name, &input_socket.id, &input_socket.name)
            })
            .map_or(&input_socket.default, |(_, value)| value);
        if let Some(links) = parent_index.incoming.get(parent_node_id) {
            for &link_index in links {
                let Some(link) = parent_graph.links.get(link_index) else {
                    continue;
                };
                if link.to_node == *parent_node_id
                    && socket_matches(&link.to_socket, &input_socket.id, &input_socket.name)
                {
                    let source = self.output_state(
                        root,
                        root_graph,
                        node_groups,
                        &parent_invocation,
                        &link.from_node,
                        &link.from_socket,
                    );
                    return if source == FoldOutputState::Dangling {
                        if primitive_socket_value(default_value) {
                            FoldOutputState::Primitive
                        } else {
                            FoldOutputState::Dynamic
                        }
                    } else {
                        source
                    };
                }
            }
        }
        if primitive_socket_value(default_value) {
            FoldOutputState::Primitive
        } else {
            FoldOutputState::Dynamic
        }
    }

    fn group_node_output_state(
        &mut self,
        root: &FoldIndex,
        root_graph: &NodeGroup,
        node_groups: &Registry<NodeGroup>,
        invocation: &InvocationKey,
        node_id: &Id,
        socket: &str,
    ) -> FoldOutputState {
        let Some(parent_index) = index_at(root, invocation) else {
            return FoldOutputState::Dynamic;
        };
        let Some(child_index) = parent_index.children.get(node_id) else {
            return FoldOutputState::Dynamic;
        };
        let child_invocation = append_invocation(invocation, parent_index, node_id);
        let Some(child_graph) = graph_at(root, root_graph, node_groups, &child_invocation) else {
            return FoldOutputState::Dynamic;
        };
        let Some(output_socket) = child_graph
            .interface
            .outputs
            .iter()
            .find(|candidate| candidate.id == socket || candidate.name == socket)
        else {
            return FoldOutputState::Dynamic;
        };
        let Some(output_node_id) = active_group_output(child_graph) else {
            return FoldOutputState::Dynamic;
        };
        let Some(output_node) = child_graph.nodes.get(output_node_id) else {
            return FoldOutputState::Dynamic;
        };
        let default_value = output_node
            .inputs
            .iter()
            .find(|(name, value)| {
                !value.is_null() && socket_matches(name, &output_socket.id, &output_socket.name)
            })
            .map_or(&output_socket.default, |(_, value)| value);
        if let Some(links) = child_index.incoming.get(output_node_id) {
            for &link_index in links {
                let Some(link) = child_graph.links.get(link_index) else {
                    continue;
                };
                if link.to_node == *output_node_id
                    && socket_matches(&link.to_socket, &output_socket.id, &output_socket.name)
                {
                    let source = self.output_state(
                        root,
                        root_graph,
                        node_groups,
                        &child_invocation,
                        &link.from_node,
                        &link.from_socket,
                    );
                    return if source == FoldOutputState::Dangling {
                        if primitive_socket_value(default_value) {
                            FoldOutputState::Primitive
                        } else {
                            FoldOutputState::Dynamic
                        }
                    } else {
                        source
                    };
                }
            }
        }
        if primitive_socket_value(default_value) {
            FoldOutputState::Primitive
        } else {
            FoldOutputState::Dynamic
        }
    }
}

/// Blender forwards a reroute's sole input and muted outputs through `internal_links`
/// (`shader_nodes_inline.cc:477-482, 508-526, 585-623`).
pub(super) fn source_passthrough_input_socket<'a>(
    node: &'a GraphNode,
    output_socket: &str,
    graph: &NodeGroup,
    node_id: &Id,
) -> Option<&'a str> {
    if node.node_type == "NodeReroute" {
        return node
            .inputs
            .keys()
            .next()
            .map(String::as_str)
            .or(Some("Input"));
    }
    if node.properties.get("mute").and_then(Value::as_bool) != Some(true) {
        return None;
    }
    if let Some(links) = node.properties.get("internal_links") {
        return links
            .as_array()?
            .iter()
            .find(|link| link.get("to_socket").and_then(Value::as_str) == Some(output_socket))
            .and_then(|link| link.get("from_socket").and_then(Value::as_str));
    }
    source_muted_default_input_socket(node, output_socket, graph, node_id)
}

fn source_muted_route_is_unknown(
    node: &GraphNode,
    output_socket: &str,
    graph: &NodeGroup,
    node_id: &Id,
) -> bool {
    if node.node_type == "NodeReroute"
        || node.properties.get("mute").and_then(Value::as_bool) != Some(true)
    {
        return false;
    }
    match node.properties.get("internal_links") {
        Some(links) => links.as_array().is_none(),
        None => source_muted_default_input_socket(node, output_socket, graph, node_id).is_none(),
    }
}

/// Blender prefers linked compatible inputs in the tree update; Mix factors opt out of muted links
/// (`blender_d13_exact_node_tree_update.cc@d4ff69f677bcc21a93a4e9f5570287f36aadc41f:860-890`,
/// `node_shader_mix.cc:50,58`).
fn source_muted_default_input_socket(
    node: &GraphNode,
    output_socket: &str,
    graph: &NodeGroup,
    node_id: &Id,
) -> Option<&'static str> {
    if node.node_type != "ShaderNodeMix" {
        return None;
    }
    let (result, candidates): (&'static str, &'static [&'static str]) = match node
        .properties
        .get("data_type")
        .and_then(Value::as_str)
        .unwrap_or("FLOAT")
    {
        "FLOAT" => ("Result_Float", &["A_Float", "B_Float"]),
        "VECTOR" => ("Result_Vector", &["A_Vector", "B_Vector"]),
        "RGBA" => ("Result_Color", &["A_Color", "B_Color"]),
        "ROTATION" => ("Result_Rotation", &["A_Rotation", "B_Rotation"]),
        _ => return None,
    };
    if output_socket != result {
        return None;
    }

    let mut selected = None;
    let mut selected_is_linked = false;
    for &candidate in candidates {
        if !input_is_available(node, candidate) {
            continue;
        }
        let is_linked = graph.links.iter().any(|link| {
            link.to_node == *node_id
                && source_input_socket_matches(node, &link.to_socket, candidate)
        });
        if selected.is_none() || (is_linked && !selected_is_linked) {
            selected = Some(candidate);
            selected_is_linked = is_linked;
        }
    }
    selected
}
fn source_input_socket_matches(node: &GraphNode, linked: &str, candidate: &str) -> bool {
    linked == candidate
        || (node.node_type == "ShaderNodeMix"
            && ((candidate == "A_Color" && linked == "A")
                || (candidate == "B_Color" && linked == "B")))
}

/// RNA input flags retained by the bridge in `GraphNode.properties`.
pub(super) fn source_input_socket_flag(node: &GraphNode, socket: &str, flag: &str) -> Option<bool> {
    node.properties
        .get("input_socket_flags")?
        .get(socket)?
        .get(flag)?
        .as_bool()
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
    primitive_outputs: &mut BTreeMap<InvocationKey, BTreeSet<(Id, String)>>,
    dangling_outputs: &mut BTreeMap<InvocationKey, BTreeMap<Id, BTreeSet<String>>>,
    unknown_outputs: &mut BTreeMap<InvocationKey, BTreeSet<(Id, String)>>,
    unknown_fold_sources: &mut BTreeMap<InvocationKey, UnknownFoldSources>,
) {
    index.primitive_nodes = primitive_nodes.remove(invocation).unwrap_or_default();
    index.primitive_outputs = primitive_outputs.remove(invocation).unwrap_or_default();
    index.dangling_outputs = dangling_outputs.remove(invocation).unwrap_or_default();
    index.unknown_outputs = unknown_outputs.remove(invocation).unwrap_or_default();
    index.unknown_fold_sources = unknown_fold_sources.remove(invocation).unwrap_or_default();
    let child_nodes = index.children.keys().cloned().collect::<Vec<_>>();
    for node_id in child_nodes {
        let child_invocation = append_invocation(invocation, index, &node_id);
        if let Some(child) = index.children.get_mut(&node_id) {
            assign_primitive_nodes(
                child,
                &child_invocation,
                primitive_nodes,
                primitive_outputs,
                dangling_outputs,
                unknown_outputs,
                unknown_fold_sources,
            );
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

pub(super) fn primitive_socket_value(value: &Value) -> bool {
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
    if let Some(enabled) = source_input_socket_flag(node, socket, "enabled") {
        return enabled;
    }
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
    let math_operation = node
        .properties
        .get("operation")
        .and_then(Value::as_str)
        .unwrap_or("ADD");
    let vector_operation = node
        .properties
        .get("operation")
        .and_then(Value::as_str)
        .unwrap_or("ADD");
    let mix_data_type = node
        .properties
        .get("data_type")
        .and_then(Value::as_str)
        .unwrap_or("FLOAT");
    let mix_factor_mode = node
        .properties
        .get("factor_mode")
        .and_then(Value::as_str)
        .unwrap_or("UNIFORM");
    let map_range_data_type = node
        .properties
        .get("data_type")
        .and_then(Value::as_str)
        .unwrap_or("FLOAT");
    match (node.node_type.as_str(), socket) {
        ("ShaderNodeMath", "Value_001") => !matches!(
            math_operation,
            "SQRT"
                | "SIGN"
                | "CEIL"
                | "SINE"
                | "ROUND"
                | "FLOOR"
                | "COSINE"
                | "ARCSINE"
                | "TANGENT"
                | "ABSOLUTE"
                | "RADIANS"
                | "DEGREES"
                | "FRACTION"
                | "ARCCOSINE"
                | "ARCTANGENT"
                | "INV_SQRT"
                | "INVERSE_SQRT"
                | "TRUNC"
                | "EXPONENT"
                | "COSH"
                | "SINH"
                | "TANH"
        ),
        ("ShaderNodeMath", "Value_002") => matches!(
            math_operation,
            "COMPARE" | "MULTIPLY_ADD" | "WRAP" | "SMOOTH_MIN" | "SMOOTH_MAX"
        ),
        ("ShaderNodeVectorMath", "Vector_001") => !matches!(
            vector_operation,
            "SINE"
                | "COSINE"
                | "TANGENT"
                | "CEIL"
                | "SCALE"
                | "FLOOR"
                | "LENGTH"
                | "ABSOLUTE"
                | "FRACTION"
                | "NORMALIZE"
                | "SIGN"
                | "ROUND"
        ),
        ("ShaderNodeVectorMath", "Vector_002") => {
            matches!(vector_operation, "WRAP" | "FACEFORWARD" | "MULTIPLY_ADD")
        }
        ("ShaderNodeVectorMath", "Scale") => {
            matches!(vector_operation, "SCALE" | "REFRACT")
        }
        ("ShaderNodeMix", "Factor_Float") => {
            mix_data_type != "VECTOR" || mix_factor_mode == "UNIFORM"
        }
        ("ShaderNodeMix", "Factor_Vector") => {
            mix_data_type == "VECTOR" && mix_factor_mode != "UNIFORM"
        }
        ("ShaderNodeMix", "A_Float" | "B_Float") => mix_data_type == "FLOAT",
        ("ShaderNodeMix", "A_Vector" | "B_Vector") => mix_data_type == "VECTOR",
        ("ShaderNodeMix", "A_Color" | "B_Color") => mix_data_type == "RGBA",
        ("ShaderNodeMix", "A_Rotation" | "B_Rotation") => mix_data_type == "ROTATION",
        ("ShaderNodeMapRange", socket) => match map_range_data_type {
            "FLOAT" => matches!(
                socket,
                "Value" | "From Min" | "From Max" | "To Min" | "To Max" | "Steps"
            ),
            "FLOAT_VECTOR" => matches!(
                socket,
                "Vector"
                    | "From_Min_FLOAT3"
                    | "From_Max_FLOAT3"
                    | "To_Min_FLOAT3"
                    | "To_Max_FLOAT3"
                    | "Steps_FLOAT3"
            ),
            _ => true,
        },
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
    fn muted_rgba_mix_graph(
        input_socket: &str,
    ) -> crate::error::Result<(Id, NodeGroup, Registry<NodeGroup>, crate::model::World)> {
        let graph_id = Id::new("world".to_owned())?;
        let source_id = Id::new("source_mix".to_owned())?;
        let mix_id = Id::new("muted_mix".to_owned())?;
        let background_id = Id::new("background".to_owned())?;
        let output_id = Id::new("output".to_owned())?;
        let mut source = GraphNode::new("ShaderNodeMix");
        source
            .properties
            .insert("data_type".to_owned(), json!("RGBA"));
        source.inputs.insert("Factor_Float".to_owned(), json!(1.0));
        source
            .inputs
            .insert("A_Color".to_owned(), json!([0.0, 0.0, 0.0, 1.0]));
        source
            .inputs
            .insert("B_Color".to_owned(), json!([0.25, 0.5, 0.75, 1.0]));
        let mut mix = GraphNode::new("ShaderNodeMix");
        mix.properties.insert("data_type".to_owned(), json!("RGBA"));
        mix.properties.insert("mute".to_owned(), json!(true));
        mix.inputs.insert("Factor_Float".to_owned(), json!(1.0));
        if input_socket == "B" {
            mix.inputs
                .insert("A".to_owned(), json!([0.0, 0.0, 0.0, 1.0]));
            mix.inputs
                .insert("B".to_owned(), json!([0.0, 0.0, 0.0, 1.0]));
        } else {
            mix.inputs
                .insert("A_Color".to_owned(), json!([0.0, 0.0, 0.0, 1.0]));
            mix.inputs
                .insert("B_Color".to_owned(), json!([0.0, 0.0, 0.0, 1.0]));
        }
        let mut graph = NodeGroup::new("World alias fold", GraphKind::Shader);
        graph.nodes.insert(source_id.clone(), source);
        graph.nodes.insert(mix_id, mix);
        graph.nodes.insert(
            background_id.clone(),
            GraphNode::new("ShaderNodeBackground"),
        );
        graph
            .nodes
            .insert(output_id.clone(), GraphNode::new("ShaderNodeOutputWorld"));
        graph.links.extend([
            GraphLink {
                from_node: source_id,
                from_socket: "Result_Color".to_owned(),
                to_node: Id::new("muted_mix".to_owned())?,
                to_socket: input_socket.to_owned(),
            },
            GraphLink {
                from_node: Id::new("muted_mix".to_owned())?,
                from_socket: "Result_Color".to_owned(),
                to_node: background_id.clone(),
                to_socket: "Color".to_owned(),
            },
            GraphLink {
                from_node: background_id,
                from_socket: "Background".to_owned(),
                to_node: output_id,
                to_socket: "Surface".to_owned(),
            },
        ]);
        let mut groups = Registry::new();
        groups.insert(graph_id.clone(), graph.clone());
        let world = crate::model::World {
            node_tree: Some(graph_id.clone()),
            ..Default::default()
        };
        Ok((graph_id, graph, groups, world))
    }

    #[test]
    fn muted_rgba_mix_preserves_world_b_socket_alias() -> crate::error::Result<()> {
        for input_socket in ["B_Color", "B"] {
            let (graph_id, graph, groups, world) = muted_rgba_mix_graph(input_socket)?;
            let mix_id = Id::new("muted_mix".to_owned())?;
            let folds = FoldIndex::new(&graph_id, &graph, &groups);
            assert_eq!(
                source_muted_default_input_socket(
                    &graph.nodes[&mix_id],
                    "Result_Color",
                    &graph,
                    &mix_id
                ),
                Some("B_Color")
            );
            assert!(
                folds
                    .primitive_outputs
                    .contains(&(mix_id.clone(), "Result_Color".to_owned()))
            );
            assert!(!folds.dangling_outputs.contains_key(&mix_id));
            assert!(
                !folds
                    .unknown_outputs
                    .iter()
                    .any(|(node, _)| node == &mix_id)
            );
            assert_eq!(
                crate::shader::world::evaluate_world(
                    &world,
                    &groups,
                    &std::collections::BTreeMap::new(),
                    glam::DVec3::Z,
                    true
                )?,
                glam::DVec3::new(0.25, 0.5, 0.75)
            );
        }
        Ok(())
    }
}
