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

/// The Principled inputs that the Material fields differing between `before` and `after` set, for an authored graph
/// (`blend_bridge.py` `_set_principled_values` names the same inputs). Alpha is written only when the alpha a Material
/// derives changes, so a colour edit does not rewrite an authored transparency.
pub(super) fn principled_writes(before: &Material, after: &Material) -> Vec<(&'static str, Value)> {
    use crate::float::{equal_f64, equal_f64_array};
    let mut writes = Vec::new();
    if !equal_f64_array(&before.base_color, &after.base_color) {
        writes.push(("Base Color", serde_json::json!(after.base_color)));
    }
    if !equal_f64(before.metallic, after.metallic) {
        writes.push(("Metallic", serde_json::json!(after.metallic)));
    }
    if !equal_f64(before.roughness, after.roughness) {
        writes.push(("Roughness", serde_json::json!(after.roughness)));
    }
    if !equal_f64_array(&before.emission_color, &after.emission_color) {
        let [red, green, blue] = after.emission_color;
        writes.push(("Emission Color", serde_json::json!([red, green, blue, 1.0])));
    }
    if !equal_f64(before.emission_strength, after.emission_strength) {
        writes.push((
            "Emission Strength",
            serde_json::json!(after.emission_strength),
        ));
    }
    if !equal_f64(before.transmission, after.transmission) {
        writes.push(("Transmission Weight", serde_json::json!(after.transmission)));
    }
    if !equal_f64(before.ior, after.ior) {
        writes.push(("IOR", serde_json::json!(after.ior)));
    }
    if !equal_f64(derived_alpha(before), derived_alpha(after)) {
        writes.push(("Alpha", serde_json::json!(derived_alpha(after))));
    }
    writes
}

/// The Principled Alpha a Material derives: 1 when opaque, its base alpha when blended, and binary at the threshold
/// when clipped (`blend_bridge.py` `_constant_alpha`).
fn derived_alpha(material: &Material) -> f64 {
    let base = material.base_color[3];
    match material.alpha_mode.as_str() {
        "blend" => base,
        "clip" => {
            if base >= material.alpha_threshold {
                1.0
            } else {
                0.0
            }
        }
        _ => 1.0,
    }
}

/// The Principled BSDF that feeds the Surface of a Material Output in `graph`, or None.
fn surface_principled(graph: &NodeGroup) -> Option<Id> {
    graph.links.iter().find_map(|link| {
        let output = graph.nodes.get(&link.to_node)?;
        let principled = graph.nodes.get(&link.from_node)?;
        let is_output = matches!(
            output.node_type.as_str(),
            "OutputMaterial" | "ShaderNodeOutputMaterial"
        );
        let is_principled = principled.node_type == "ShaderNodeBsdfPrincipled";
        (is_output && is_principled && link.from_socket == "BSDF" && link.to_socket == "Surface")
            .then(|| link.from_node.clone())
    })
}

/// Writes the Material field values in `writes` into the authored Principled BSDF that feeds `material_id`'s Surface, so
/// the graph Potter renders carries the edit, and export writes that graph as it is. A generated Principled (marked
/// `potter_simple_material`) takes the Material's fields itself, so it needs no write. The graph is content-addressed,
/// so the edit is a new graph under its own ID; the stored graph that other materials share is left unchanged.
pub(super) fn write_authored_inputs(
    engine: &mut Engine<'_>,
    material_id: &Id,
    writes: &[(&str, Value)],
) -> Result<()> {
    let Some(graph_id) = engine
        .doc
        .materials
        .get(material_id)
        .and_then(|material| material.node_tree.clone())
    else {
        return Ok(());
    };
    let Some(graph) = engine.doc.node_groups.get(&graph_id) else {
        return Ok(());
    };
    let Some(principled_id) = surface_principled(graph) else {
        return Ok(());
    };
    let generated = graph.nodes.get(&principled_id).is_some_and(|node| {
        node.properties.get("potter_simple_material") == Some(&Value::Bool(true))
    });
    if generated || writes.is_empty() {
        return Ok(());
    }
    let mut edited = graph.clone();
    if let Some(node) = edited.nodes.get_mut(&principled_id) {
        for (input, value) in writes {
            node.inputs.insert((*input).to_owned(), value.clone());
        }
    }
    let value = serde_json::to_value(&edited).map_err(crate::error::PotError::internal_json)?;
    let edited_id = content_id(&canonicalize(&value)?)?;
    if edited_id == graph_id {
        return Ok(());
    }
    match engine.doc.node_groups.get(&edited_id) {
        Some(existing) if existing != &edited => {
            return Err(engine.error(
                ErrorCode::IdExists,
                format!("derived shader graph ID `{edited_id}` already exists"),
                &operation_pointer(engine.operation_index, "set"),
            ));
        }
        Some(_) => {}
        None => {
            engine.doc.node_groups.insert(edited_id.clone(), edited);
            engine.mark("node_groups", &edited_id, ChangeKind::Created);
        }
    }
    if let Some(material) = engine.doc.materials.get_mut(material_id) {
        material.node_tree = Some(edited_id);
    }
    Ok(())
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
