use serde_json::{Value, json};

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

/// The Material fields Potter's Principled setter writes, which the bridge compares with an authored graph's baseline
/// (`blend_bridge.py` `_MATERIAL_FIELDS`).
const BASELINE_FIELDS: [&str; 9] = [
    "base_color",
    "metallic",
    "roughness",
    "alpha_mode",
    "alpha_threshold",
    "emission_color",
    "emission_strength",
    "transmission",
    "ior",
];
/// The node property holding Blender custom properties, which `blend_bridge.py` reads back (`_model_group_tree`).
pub(super) const CUSTOM_PROPERTIES: &str = "_blender_custom_properties";
/// The custom property holding the baseline (`blend_bridge.py` `_IMPORT_BASELINE_PROPERTY`).
const BASELINE_PROPERTY: &str = "potter_import_baseline";
const PRINCIPLED_NODE: &str = "ShaderNodeBsdfPrincipled";

/// Whether a graph node is a Principled BSDF, the node whose inputs a Material field sets.
pub(super) fn is_principled(node_type: &str) -> bool {
    node_type == PRINCIPLED_NODE
}

/// The Material fields a Principled input sets (`blend_bridge.py` `_PRINCIPLED_INPUT_FIELDS`); empty for other inputs.
pub(super) fn principled_input_fields(input: &str) -> &'static [&'static str] {
    match input {
        "Base Color" => &["base_color"],
        "Alpha" => &["base_color", "alpha_mode", "alpha_threshold"],
        "Metallic" => &["metallic"],
        "Roughness" => &["roughness"],
        "Emission Color" | "Emission" => &["emission_color"],
        "Emission Strength" => &["emission_strength"],
        "Transmission Weight" | "Transmission" => &["transmission"],
        "IOR" => &["ior"],
        _ => &[],
    }
}

fn material_value(material: &Material, field: &str) -> Value {
    match field {
        "base_color" => json!(material.base_color),
        "metallic" => json!(material.metallic),
        "roughness" => json!(material.roughness),
        "alpha_mode" => json!(material.alpha_mode),
        "alpha_threshold" => json!(material.alpha_threshold),
        "emission_color" => json!(material.emission_color),
        "emission_strength" => json!(material.emission_strength),
        "transmission" => json!(material.transmission),
        "ior" => json!(material.ior),
        _ => Value::Null,
    }
}

/// Records the Material's current values as the baseline of `graph`'s Principled nodes, for `fields` (every Material
/// field when `None`). The bridge writes a Material field over an authored graph input only when that field differs
/// from this baseline (`blend_bridge.py` `export_doc`). So a Material edit after this point reaches the graph, and a
/// field the Material has not changed since keeps the graph's own input.
pub(super) fn stamp_principled_baseline(
    graph: &mut NodeGroup,
    material: &Material,
    fields: Option<&[&str]>,
) -> Result<()> {
    let fields: Vec<&str> = fields.map_or_else(|| BASELINE_FIELDS.to_vec(), <[&str]>::to_vec);
    for node in graph
        .nodes
        .values_mut()
        .filter(|node| is_principled(&node.node_type))
    {
        let mut custom = match node.properties.remove(CUSTOM_PROPERTIES) {
            Some(Value::Object(map)) => map,
            _ => serde_json::Map::new(),
        };
        let mut baseline = match custom.get(BASELINE_PROPERTY).and_then(Value::as_str) {
            Some(text) => serde_json::from_str::<serde_json::Map<String, Value>>(text)
                .map_err(crate::error::PotError::internal_json)?,
            None => BASELINE_FIELDS
                .iter()
                .map(|field| ((*field).to_owned(), material_value(material, field)))
                .collect(),
        };
        for field in &fields {
            baseline.insert((*field).to_owned(), material_value(material, field));
        }
        let text =
            serde_json::to_string(&baseline).map_err(crate::error::PotError::internal_json)?;
        custom.insert(BASELINE_PROPERTY.to_owned(), Value::String(text));
        node.properties
            .insert(CUSTOM_PROPERTIES.to_owned(), Value::Object(custom));
    }
    Ok(())
}

/// Stamps the baseline on `graph_id`'s Principled nodes from the Material that uses it, when one does.
pub(super) fn stamp_graph_baseline(
    doc: &mut crate::model::SceneDoc,
    graph_id: &Id,
    fields: Option<&[&str]>,
) -> Result<()> {
    let Some(material) = doc
        .materials
        .values()
        .find(|material| material.node_tree.as_ref() == Some(graph_id))
        .cloned()
    else {
        return Ok(());
    };
    let Some(graph) = doc.node_groups.get_mut(graph_id) else {
        return Ok(());
    };
    stamp_principled_baseline(graph, &material, fields)
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
