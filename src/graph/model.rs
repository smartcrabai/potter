use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::model::{Id, Registry};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphKind {
    Geometry,
    Shader,
    Compositor,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphSocket {
    pub id: String,
    pub name: String,
    pub socket_type: String,
    #[serde(default)]
    pub default: Value,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphInterface {
    #[serde(default)]
    pub inputs: Vec<GraphSocket>,
    #[serde(default)]
    pub outputs: Vec<GraphSocket>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphNode {
    #[serde(rename = "type")]
    pub node_type: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub location: [f64; 2],
    #[serde(default)]
    pub properties: Map<String, Value>,
    #[serde(default)]
    pub inputs: BTreeMap<String, Value>,
}

impl GraphNode {
    #[must_use]
    pub fn new(node_type: impl Into<String>) -> Self {
        let node_type = node_type.into();
        Self {
            name: node_type.clone(),
            node_type,
            location: [0.0; 2],
            properties: Map::new(),
            inputs: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphLink {
    pub from_node: Id,
    pub from_socket: String,
    pub to_node: Id,
    pub to_socket: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeGroup {
    #[serde(default)]
    pub name: String,
    pub kind: GraphKind,
    #[serde(default)]
    pub interface: GraphInterface,
    #[serde(default)]
    pub nodes: Registry<GraphNode>,
    #[serde(default)]
    pub links: Vec<GraphLink>,
}

impl NodeGroup {
    #[must_use]
    pub fn new(name: impl Into<String>, kind: GraphKind) -> Self {
        Self {
            name: name.into(),
            kind,
            interface: GraphInterface::default(),
            nodes: Registry::new(),
            links: Vec::new(),
        }
    }
}
