use serde_json::Value;

use crate::{
    error::{ErrorCode, Result},
    graph::{GraphKind, GraphNode, NodeGroup},
    hash::{canonicalize, sha256},
    model::{Id, Material},
};

use super::{ChangeKind, Engine, operation_pointer};

pub(super) fn resolve_graph(engine: &mut Engine<'_>, value: &Value) -> Result<Id> {
    if let Some(value) = value.as_str() {
        let graph_id = Id::new(value).map_err(|_| {
            engine.error(
                ErrorCode::InvalidOperation,
                "graph must reference a valid node-group ID",
                &operation_pointer(engine.operation_index, "graph"),
            )
        })?;
        let Some(graph) = engine.doc.node_groups.get(&graph_id) else {
            return Err(engine.error(
                ErrorCode::TargetNotFound,
                format!("shader graph `{graph_id}` was not found"),
                &operation_pointer(engine.operation_index, "graph"),
            ));
        };
        if graph.kind != GraphKind::Shader {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("node group `{graph_id}` is not a shader graph"),
                &operation_pointer(engine.operation_index, "graph"),
            ));
        }
        return Ok(graph_id);
    }

    let Some(object) = value.as_object() else {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "graph must be a shader graph object or node-group ID",
            &operation_pointer(engine.operation_index, "graph"),
        ));
    };
    let mut graph_value = value.clone();
    let graph_object = graph_value.as_object_mut().ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "graph must be an object",
            &operation_pointer(engine.operation_index, "graph"),
        )
    })?;
    if !graph_object.contains_key("kind") {
        graph_object.insert("kind".to_owned(), Value::String("shader".to_owned()));
    }
    let graph: NodeGroup = serde_json::from_value(graph_value.clone()).map_err(|error| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("invalid inline shader graph: {error}"),
            &operation_pointer(engine.operation_index, "graph"),
        )
    })?;
    if graph.kind != GraphKind::Shader {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "material graph must have kind `shader`",
            &operation_pointer(engine.operation_index, "graph/kind"),
        ));
    }
    let content = canonicalize(&graph_value)?;
    let graph_id = content_id(&content)?;
    if let Some(existing) = engine.doc.node_groups.get(&graph_id) {
        if existing != &graph {
            return Err(engine.error(
                ErrorCode::IdExists,
                format!("derived shader graph ID `{graph_id}` already exists"),
                &operation_pointer(engine.operation_index, "graph"),
            ));
        }
    } else {
        engine.doc.node_groups.insert(graph_id.clone(), graph);
        engine.mark("node_groups", &graph_id, ChangeKind::Created);
    }
    let _ = object;
    Ok(graph_id)
}

pub(super) fn attach_default_graph(
    engine: &mut Engine<'_>,
    material_id: &Id,
    material: &mut Material,
) -> Result<()> {
    let graph_id = content_id(material_id.as_str().as_bytes())?;
    let mut graph = NodeGroup::new(format!("{} Shader", material.name), GraphKind::Shader);
    let mut principled = GraphNode::new("ShaderNodeBsdfPrincipled");
    principled
        .properties
        .insert("potter_simple_material".to_owned(), Value::Bool(true));
    graph.nodes.insert(Id::new("principled")?, principled);
    graph
        .nodes
        .insert(Id::new("output")?, GraphNode::new("OutputMaterial"));
    graph.links.push(crate::graph::GraphLink {
        from_node: Id::new("principled")?,
        from_socket: "BSDF".to_owned(),
        to_node: Id::new("output")?,
        to_socket: "Surface".to_owned(),
    });
    if let Some(existing) = engine.doc.node_groups.get(&graph_id) {
        if existing != &graph {
            return Err(engine.error(
                ErrorCode::IdExists,
                format!("generated shader graph ID `{graph_id}` already exists"),
                &operation_pointer(engine.operation_index, "id"),
            ));
        }
    } else {
        engine.doc.node_groups.insert(graph_id.clone(), graph);
        engine.mark("node_groups", &graph_id, ChangeKind::Created);
    }
    material.node_tree = Some(graph_id);
    Ok(())
}

fn content_id(content: &[u8]) -> Result<Id> {
    let hash = sha256(content);
    Id::new(format!("shader_{}", &hash[7..31]))
}

#[cfg(test)]
mod tests {
    use super::content_id;
    use crate::model::Id;

    #[test]
    fn content_graph_ids_are_stable_and_valid() -> crate::error::Result<()> {
        let first = content_id(b"shader")?;
        let second = content_id(b"shader")?;
        assert_eq!(first, second);
        assert!(Id::is_valid(first.as_str()));
        Ok(())
    }
}
