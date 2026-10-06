use std::collections::BTreeMap;

use serde_json::{Map, Value};

use crate::{
    error::{ErrorCode, Result},
    graph::{GraphKind, GraphLink, GraphNode, GraphSocket, NodeGroup},
    model::Id,
};

use super::{
    ChangeKind, Engine, check_fields, operation_pointer, parse_id, pointer_escape, read_id,
    read_string,
};

pub(super) fn apply(
    engine: &mut Engine<'_>,
    name: &str,
    operation: &Map<String, Value>,
) -> Result<bool> {
    match name {
        "graph.create" => create(engine, operation),
        "graph.delete" => delete(engine, operation),
        "graph.node_add" => node_add(engine, operation),
        "graph.node_update" => node_update(engine, operation),
        "graph.node_remove" => node_remove(engine, operation),
        "graph.link" => link(engine, operation),
        "graph.unlink" => unlink(engine, operation),
        "graph.interface_update" => interface_update(engine, operation),
        _ => Err(engine.error(
            ErrorCode::InternalError,
            "invalid graph operation dispatch",
            &operation_pointer(engine.operation_index, "op"),
        )),
    }
}

fn create(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "id", "name", "kind"],
        &["id", "kind"],
    )?;
    let id = read_id(engine, operation, "id")?;
    if engine.doc.node_groups.contains_key(&id) {
        return Err(engine.error(
            ErrorCode::IdExists,
            format!("graph ID `{id}` already exists"),
            &operation_pointer(engine.operation_index, "id"),
        ));
    }
    let kind_text = read_string(engine, operation, "kind")?;
    let kind = match kind_text.as_str() {
        "geometry" => GraphKind::Geometry,
        "shader" => GraphKind::Shader,
        "compositor" => GraphKind::Compositor,
        _ => {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("unsupported graph kind `{kind_text}`"),
                &operation_pointer(engine.operation_index, "kind"),
            ));
        }
    };
    let name = if operation.contains_key("name") {
        read_string(engine, operation, "name")?
    } else {
        id.to_string()
    };
    engine
        .doc
        .node_groups
        .insert(id.clone(), NodeGroup::new(name, kind));
    engine.mark("node_groups", &id, ChangeKind::Created);
    Ok(true)
}

fn delete(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(engine, operation, &["op", "id", "target"], &[])?;
    let (id, pointer) = match (operation.get("id"), operation.get("target")) {
        (Some(_), Some(_)) => {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "graph.delete accepts either id or target, not both",
                &operation_pointer(engine.operation_index, "target"),
            ));
        }
        (Some(_), None) => (
            read_id(engine, operation, "id")?,
            operation_pointer(engine.operation_index, "id"),
        ),
        (None, Some(target)) => {
            let target = target.as_object().ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "target must be an object",
                    &operation_pointer(engine.operation_index, "target"),
                )
            })?;
            check_object_fields(
                engine,
                target,
                &["id"],
                &operation_pointer(engine.operation_index, "target"),
                "target",
            )?;
            if !target.contains_key("id") {
                return Err(engine.error(
                    ErrorCode::InvalidOperation,
                    "target must contain id",
                    &operation_pointer(engine.operation_index, "target/id"),
                ));
            }
            let text = read_object_string(
                engine,
                target,
                "id",
                &operation_pointer(engine.operation_index, "target/id"),
            )?;
            let pointer = operation_pointer(engine.operation_index, "target/id");
            (parse_id(engine, &text, &pointer)?, pointer)
        }
        (None, None) => {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "graph.delete requires id or target",
                &operation_pointer(engine.operation_index, "target"),
            ));
        }
    };
    if engine.doc.node_groups.remove(&id).is_none() {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            format!("graph `{id}` was not found"),
            &pointer,
        ));
    }
    engine.mark("node_groups", &id, ChangeKind::Deleted);
    Ok(true)
}

fn node_add(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &[
            "op",
            "graph",
            "id",
            "type",
            "name",
            "location",
            "properties",
            "inputs",
        ],
        &["graph", "id", "type"],
    )?;
    let graph_id = read_id(engine, operation, "graph")?;
    let node_id = read_id(engine, operation, "id")?;
    let graph = get_graph(engine, &graph_id)?;
    if graph.nodes.contains_key(&node_id) {
        return Err(engine.error(
            ErrorCode::IdExists,
            format!("node ID `{node_id}` already exists in graph `{graph_id}`"),
            &operation_pointer(engine.operation_index, "id"),
        ));
    }

    let node_type = read_string(engine, operation, "type")?;
    let mut node = GraphNode::new(node_type);
    if operation.contains_key("name") {
        node.name = read_string(engine, operation, "name")?;
    }
    if let Some(value) = operation.get("location") {
        node.location = read_location(
            engine,
            value,
            &operation_pointer(engine.operation_index, "location"),
        )?;
    }
    if let Some(value) = operation.get("properties") {
        node.properties = read_map(
            engine,
            value,
            &operation_pointer(engine.operation_index, "properties"),
            "properties",
        )?;
    }
    if let Some(value) = operation.get("inputs") {
        node.inputs = read_inputs(
            engine,
            value,
            &operation_pointer(engine.operation_index, "inputs"),
        )?;
    }
    if let Some(graph) = engine.doc.node_groups.get_mut(&graph_id) {
        graph.nodes.insert(node_id, node);
    }
    engine.mark("node_groups", &graph_id, ChangeKind::Updated);
    Ok(true)
}

fn node_update(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "graph", "node_id", "set"],
        &["graph", "node_id", "set"],
    )?;
    let graph_id = read_id(engine, operation, "graph")?;
    let node_id = read_id(engine, operation, "node_id")?;
    let graph = get_graph(engine, &graph_id)?;
    if !graph.nodes.contains_key(&node_id) {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            format!("node `{node_id}` was not found in graph `{graph_id}`"),
            &operation_pointer(engine.operation_index, "node_id"),
        ));
    }
    let set = operation
        .get("set")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "set must be an object",
                &operation_pointer(engine.operation_index, "set"),
            )
        })?;
    if set.is_empty() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "set must not be empty",
            &operation_pointer(engine.operation_index, "set"),
        ));
    }
    check_object_fields(
        engine,
        set,
        &["name", "location", "properties", "inputs"],
        &operation_pointer(engine.operation_index, "set"),
        "node set",
    )?;

    let name = set
        .get("name")
        .map(|value| {
            read_value_string(
                engine,
                value,
                &operation_pointer(engine.operation_index, "set/name"),
                "name",
            )
        })
        .transpose()?;
    let location = set
        .get("location")
        .map(|value| {
            read_location(
                engine,
                value,
                &operation_pointer(engine.operation_index, "set/location"),
            )
        })
        .transpose()?;
    let properties = set
        .get("properties")
        .map(|value| {
            read_map(
                engine,
                value,
                &operation_pointer(engine.operation_index, "set/properties"),
                "properties",
            )
        })
        .transpose()?;
    let inputs = set
        .get("inputs")
        .map(|value| {
            read_inputs(
                engine,
                value,
                &operation_pointer(engine.operation_index, "set/inputs"),
            )
        })
        .transpose()?;

    let mut changed = false;
    if let Some(graph) = engine.doc.node_groups.get_mut(&graph_id)
        && let Some(node) = graph.nodes.get_mut(&node_id)
    {
        if let Some(name) = name
            && node.name != name
        {
            node.name = name;
            changed = true;
        }
        if let Some(location) = location
            && !crate::float::equal_f64_array(&node.location, &location)
        {
            node.location = location;
            changed = true;
        }
        if let Some(properties) = properties
            && node.properties != properties
        {
            node.properties = properties;
            changed = true;
        }
        if let Some(inputs) = inputs
            && node.inputs != inputs
        {
            node.inputs = inputs;
            changed = true;
        }
    }
    if changed {
        engine.mark("node_groups", &graph_id, ChangeKind::Updated);
    }
    Ok(changed)
}

fn node_remove(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "graph", "node_id"],
        &["graph", "node_id"],
    )?;
    let graph_id = read_id(engine, operation, "graph")?;
    let node_id = read_id(engine, operation, "node_id")?;
    let graph = get_graph(engine, &graph_id)?;
    if !graph.nodes.contains_key(&node_id) {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            format!("node `{node_id}` was not found in graph `{graph_id}`"),
            &operation_pointer(engine.operation_index, "node_id"),
        ));
    }
    if let Some(graph) = engine.doc.node_groups.get_mut(&graph_id) {
        graph.nodes.remove(&node_id);
        graph
            .links
            .retain(|link| link.from_node != node_id && link.to_node != node_id);
    }
    engine.mark("node_groups", &graph_id, ChangeKind::Updated);
    Ok(true)
}

fn link(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_link_fields(engine, operation)?;
    let graph_id = read_id(engine, operation, "graph")?;
    let graph = get_graph(engine, &graph_id)?;
    let graph_link = read_link(engine, operation)?;
    ensure_link_nodes(engine, graph, &graph_link)?;
    crate::graph::validate_link(graph, &graph_link).map_err(|error| {
        engine.error(
            error.code,
            error.message,
            &operation_pointer(engine.operation_index, "to_socket"),
        )
    })?;
    if let Some(graph) = engine.doc.node_groups.get_mut(&graph_id) {
        graph.links.push(graph_link);
    }
    engine.mark("node_groups", &graph_id, ChangeKind::Updated);
    Ok(true)
}

fn unlink(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_link_fields(engine, operation)?;
    let graph_id = read_id(engine, operation, "graph")?;
    let graph_link = read_link(engine, operation)?;
    let graph = get_graph(engine, &graph_id)?;
    ensure_link_nodes(engine, graph, &graph_link)?;
    let changed = if let Some(graph) = engine.doc.node_groups.get_mut(&graph_id) {
        let previous_len = graph.links.len();
        graph.links.retain(|link| link != &graph_link);
        graph.links.len() != previous_len
    } else {
        false
    };
    if changed {
        engine.mark("node_groups", &graph_id, ChangeKind::Updated);
    }
    Ok(changed)
}

fn interface_update(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "graph", "set"],
        &["graph", "set"],
    )?;
    let graph_id = read_id(engine, operation, "graph")?;
    get_graph(engine, &graph_id)?;
    let set = operation
        .get("set")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "set must be an object",
                &operation_pointer(engine.operation_index, "set"),
            )
        })?;
    if set.is_empty() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "set must not be empty",
            &operation_pointer(engine.operation_index, "set"),
        ));
    }
    check_object_fields(
        engine,
        set,
        &["inputs", "outputs"],
        &operation_pointer(engine.operation_index, "set"),
        "interface set",
    )?;
    let inputs = set
        .get("inputs")
        .map(|value| {
            read_sockets(
                engine,
                value,
                &operation_pointer(engine.operation_index, "set/inputs"),
            )
        })
        .transpose()?;
    let outputs = set
        .get("outputs")
        .map(|value| {
            read_sockets(
                engine,
                value,
                &operation_pointer(engine.operation_index, "set/outputs"),
            )
        })
        .transpose()?;

    let mut changed = false;
    if let Some(graph) = engine.doc.node_groups.get_mut(&graph_id) {
        if let Some(inputs) = inputs
            && graph.interface.inputs != inputs
        {
            graph.interface.inputs = inputs;
            changed = true;
        }
        if let Some(outputs) = outputs
            && graph.interface.outputs != outputs
        {
            graph.interface.outputs = outputs;
            changed = true;
        }
    }
    if changed {
        engine.mark("node_groups", &graph_id, ChangeKind::Updated);
    }
    Ok(changed)
}

fn check_link_fields(engine: &Engine<'_>, operation: &Map<String, Value>) -> Result<()> {
    check_fields(
        engine,
        operation,
        &[
            "op",
            "graph",
            "from_node",
            "from_socket",
            "to_node",
            "to_socket",
        ],
        &["graph", "from_node", "from_socket", "to_node", "to_socket"],
    )
}

fn read_link(engine: &Engine<'_>, operation: &Map<String, Value>) -> Result<GraphLink> {
    Ok(GraphLink {
        from_node: read_id(engine, operation, "from_node")?,
        from_socket: read_string(engine, operation, "from_socket")?,
        to_node: read_id(engine, operation, "to_node")?,
        to_socket: read_string(engine, operation, "to_socket")?,
    })
}

fn ensure_link_nodes(engine: &Engine<'_>, graph: &NodeGroup, link: &GraphLink) -> Result<()> {
    for (node_id, field) in [(&link.from_node, "from_node"), (&link.to_node, "to_node")] {
        if !graph.nodes.contains_key(node_id) {
            return Err(engine.error(
                ErrorCode::TargetNotFound,
                format!("node `{node_id}` was not found in graph"),
                &operation_pointer(engine.operation_index, field),
            ));
        }
    }
    Ok(())
}

fn get_graph<'a>(engine: &'a Engine<'_>, graph_id: &Id) -> Result<&'a NodeGroup> {
    engine.doc.node_groups.get(graph_id).ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            format!("graph `{graph_id}` was not found"),
            &operation_pointer(engine.operation_index, "graph"),
        )
    })
}

fn check_object_fields(
    engine: &Engine<'_>,
    object: &Map<String, Value>,
    allowed: &[&str],
    pointer: &str,
    context: &str,
) -> Result<()> {
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("unknown {context} field `{key}`"),
                &format!("{pointer}/{}", pointer_escape(key)),
            ));
        }
    }
    Ok(())
}

fn read_object_string(
    engine: &Engine<'_>,
    object: &Map<String, Value>,
    key: &str,
    pointer: &str,
) -> Result<String> {
    object
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                format!("{key} must be a string"),
                pointer,
            )
        })
}

fn read_value_string(
    engine: &Engine<'_>,
    value: &Value,
    pointer: &str,
    field: &str,
) -> Result<String> {
    value.as_str().map(str::to_owned).ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("{field} must be a string"),
            pointer,
        )
    })
}

fn read_location(engine: &Engine<'_>, value: &Value, pointer: &str) -> Result<[f64; 2]> {
    let array = value.as_array().ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "location must be an array of two numbers",
            pointer,
        )
    })?;
    if array.len() != 2 {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "location must contain exactly two numbers",
            pointer,
        ));
    }
    let mut location = [0.0; 2];
    for (index, coordinate) in array.iter().enumerate() {
        location[index] = coordinate.as_f64().ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "location coordinates must be numbers",
                &format!("{pointer}/{index}"),
            )
        })?;
    }
    Ok(location)
}

fn read_map(
    engine: &Engine<'_>,
    value: &Value,
    pointer: &str,
    field: &str,
) -> Result<Map<String, Value>> {
    value.as_object().cloned().ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("{field} must be an object"),
            pointer,
        )
    })
}

fn read_inputs(
    engine: &Engine<'_>,
    value: &Value,
    pointer: &str,
) -> Result<BTreeMap<String, Value>> {
    let values = value.as_object().ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "inputs must be an object",
            pointer,
        )
    })?;
    Ok(values
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect())
}

fn read_sockets(engine: &Engine<'_>, value: &Value, pointer: &str) -> Result<Vec<GraphSocket>> {
    let values = value.as_array().ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "interface sockets must be an array",
            pointer,
        )
    })?;
    let mut sockets = Vec::with_capacity(values.len());
    for (index, value) in values.iter().enumerate() {
        let socket_pointer = format!("{pointer}/{index}");
        let socket = value.as_object().ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "interface socket must be an object",
                &socket_pointer,
            )
        })?;
        check_object_fields(
            engine,
            socket,
            &["id", "name", "socket_type", "default"],
            &socket_pointer,
            "interface socket",
        )?;
        let required = |key: &str| {
            let field_pointer = format!("{socket_pointer}/{key}");
            read_object_string(engine, socket, key, &field_pointer)
        };
        let id = required("id")?;
        let name = required("name")?;
        let socket_type = required("socket_type")?;
        let default = socket.get("default").cloned().unwrap_or(Value::Null);
        sockets.push(GraphSocket {
            id,
            name,
            socket_type,
            default,
        });
    }
    Ok(sockets)
}
