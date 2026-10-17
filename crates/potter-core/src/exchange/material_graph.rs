//! Material node graphs for glTF textures that Potter's texture fields cannot hold.
//!
//! Blender's glTF importer builds a node graph for every material with a transformed texture, an
//! emissive texture, an occlusion texture, or a normal map scale. This module builds the same shapes:
//!
//! - an Image Texture whose `Vector` comes from a Mapping node (POINT) fed by Texture Coordinate `UV`
//!   when the slot has a `KHR_texture_transform`, with Blender's Location, Rotation and Scale;
//! - the base color, metallic-roughness and emissive colors into the Principled BSDF, multiplied by
//!   their factors as Blender does;
//! - the normal map through a Normal Map node whose Strength is the `normalTexture.scale` and whose
//!   UV Map is the normal texture's layer (the render layer when it names none);
//! - occlusion through a Separate Color (Red) into the `glTF Material Output` group, which PR10's
//!   resolver reads back as glTF occlusion.

use serde_json::{Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    exchange::material_surface::TextureTransform,
    graph::{GraphKind, GraphLink, GraphNode, GraphSocket, NodeGroup},
    image::ImageInterpolation,
    model::{Id, Material, SceneDoc, TextureRef},
};

/// Name of the group Blender's importer uses for glTF occlusion; PR10's resolver finds it by name.
const OUTPUT_GROUP_NAME: &str = "glTF Material Output";
/// Potter ID of the group that holds the `Occlusion` input.
const OUTPUT_GROUP_ID: &str = "gltf_material_output";

/// A texture of one glTF slot and its `KHR_texture_transform`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SlotTexture {
    pub(crate) texture: TextureRef,
    pub(crate) transform: Option<TextureTransform>,
}

/// Every texture slot of one material that a graph can hold.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GraphTextures {
    pub(crate) base_color: Option<SlotTexture>,
    /// Packed glTF metallic (B) and roughness (G).
    pub(crate) metallic_roughness: Option<SlotTexture>,
    pub(crate) normal: Option<SlotTexture>,
    /// `normalTexture.scale`, the Normal Map Strength.
    pub(crate) normal_scale: f64,
    pub(crate) emissive: Option<SlotTexture>,
    /// Packed occlusion in R.
    pub(crate) occlusion: Option<SlotTexture>,
}

impl GraphTextures {
    /// The textures of a material whose fields hold them, with `base_transform` on the base color.
    pub(crate) fn from_fields(
        material: &Material,
        base_transform: Option<TextureTransform>,
    ) -> Self {
        let plain = |texture: &Option<TextureRef>| {
            texture.clone().map(|texture| SlotTexture {
                texture,
                transform: None,
            })
        };
        Self {
            base_color: material
                .base_color_texture
                .clone()
                .map(|texture| SlotTexture {
                    texture,
                    transform: base_transform,
                }),
            metallic_roughness: plain(&material.metallic_texture),
            normal: plain(&material.normal_texture),
            normal_scale: 1.0,
            emissive: None,
            occlusion: None,
        }
    }

    /// Whether any slot needs a graph: a transform, an emissive or occlusion texture, or a normal
    /// scale that a Potter texture field cannot hold.
    pub(crate) fn needs_graph(&self) -> bool {
        let transformed =
            |slot: &Option<SlotTexture>| slot.as_ref().is_some_and(|slot| slot.transform.is_some());
        transformed(&self.base_color)
            || transformed(&self.metallic_roughness)
            || transformed(&self.normal)
            || transformed(&self.emissive)
            || transformed(&self.occlusion)
            || self.emissive.is_some()
            || self.occlusion.is_some()
            || !crate::float::equal_f64(self.normal_scale, 1.0)
    }
}

/// Mapping Location, Rotation and Scale for a `KHR_texture_transform`, as Blender's glTF importer sets
/// them (`io_scene_gltf2` `texture_transform_gltf_to_blender`).
fn mapping_values(transform: &TextureTransform) -> ([f64; 3], [f64; 3], [f64; 3]) {
    let (sin, cos) = transform.rotation.sin_cos();
    let location = [
        transform.offset[0] + transform.scale[1] * sin,
        1.0 - transform.offset[1] - transform.scale[1] * cos,
        0.0,
    ];
    (
        location,
        [0.0, 0.0, transform.rotation],
        [transform.scale[0], transform.scale[1], 1.0],
    )
}

fn graph_error(message: impl Into<String>) -> PotError {
    PotError::new(ErrorCode::ImportFailed, message)
}

/// Replaces the texture fields of `material_id` with a shader graph that holds `textures`.
///
/// The Principled BSDF keeps the material's factors. The texture fields are cleared, since the graph
/// now holds every texture. Callers check `GraphTextures::needs_graph` first.
pub(crate) fn move_textures_to_graph(
    doc: &mut SceneDoc,
    material_id: &Id,
    textures: &GraphTextures,
) -> Result<()> {
    let material = doc
        .materials
        .get(material_id)
        .cloned()
        .ok_or_else(|| graph_error(format!("material `{material_id}` is missing")))?;
    if material.node_tree.is_some() {
        return Err(graph_error(format!(
            "material `{material_id}` already has a shader graph"
        )));
    }
    let output_group = if textures.occlusion.is_some() {
        Some(ensure_output_group(doc)?)
    } else {
        None
    };
    let graph = build_graph(&material, textures, output_group.is_some())?;
    let graph_id = Id::new(format!(
        "shader_{}",
        &crate::hash::sha256(format!("texture-graph:{material_id}").as_bytes())[7..31]
    ))?;
    doc.node_groups.insert(graph_id.clone(), graph);
    let target = doc
        .materials
        .get_mut(material_id)
        .ok_or_else(|| graph_error(format!("material `{material_id}` is missing")))?;
    target.node_tree = Some(graph_id);
    target.base_color_texture = None;
    target.metallic_texture = None;
    target.roughness_texture = None;
    target.normal_texture = None;
    Ok(())
}

/// The `glTF Material Output` group with its `Occlusion` input, created once per document.
fn ensure_output_group(doc: &mut SceneDoc) -> Result<Id> {
    let id = Id::new(OUTPUT_GROUP_ID)?;
    if !doc.node_groups.contains_key(&id) {
        let mut group = NodeGroup::new(OUTPUT_GROUP_NAME, GraphKind::Shader);
        group.interface.inputs.push(GraphSocket {
            id: "Occlusion".to_owned(),
            name: "Occlusion".to_owned(),
            socket_type: "NodeSocketFloat".to_owned(),
            default: json!(1.0),
        });
        doc.node_groups.insert(id.clone(), group);
    }
    Ok(id)
}

fn build_graph(
    material: &Material,
    textures: &GraphTextures,
    occlusion_group: bool,
) -> Result<NodeGroup> {
    let mut graph = NodeGroup::new(format!("{} Shader", material.name), GraphKind::Shader);
    let mut principled = GraphNode::new("ShaderNodeBsdfPrincipled");
    principled
        .inputs
        .insert("IOR".to_owned(), json!(material.ior));
    principled.inputs.insert(
        "Transmission Weight".to_owned(),
        json!(material.transmission),
    );
    principled.inputs.insert(
        "Emission Color".to_owned(),
        json!([
            material.emission_color[0],
            material.emission_color[1],
            material.emission_color[2],
            1.0,
        ]),
    );
    principled.inputs.insert(
        "Emission Strength".to_owned(),
        json!(material.emission_strength),
    );
    principled
        .inputs
        .insert("Alpha".to_owned(), json!(material.base_color[3]));
    principled
        .inputs
        .insert("Base Color".to_owned(), json!(material.base_color));
    principled
        .inputs
        .insert("Metallic".to_owned(), json!(material.metallic));
    principled
        .inputs
        .insert("Roughness".to_owned(), json!(material.roughness));
    insert(&mut graph, "principled", principled)?;

    let mut output = GraphNode::new("OutputMaterial");
    output
        .properties
        .insert("is_active_output".to_owned(), json!(true));
    insert(&mut graph, "output", output)?;
    connect(&mut graph, "principled", "BSDF", "output", "Surface")?;
    insert(&mut graph, "coord", GraphNode::new("ShaderNodeTexCoord"))?;

    if let Some(slot) = &textures.base_color {
        let color = texture_node(&mut graph, "base", slot)?;
        let factor = material.base_color;
        if crate::float::equal_f64_array(&[factor[0], factor[1], factor[2]], &[1.0; 3]) {
            connect(&mut graph, &color, "Color", "principled", "Base Color")?;
        } else {
            let rgb = [factor[0], factor[1], factor[2]];
            multiply_color(
                &mut graph,
                &color,
                "Color",
                "base",
                rgb,
                "principled",
                "Base Color",
            )?;
        }
        if material.alpha_mode != "opaque" {
            connect(&mut graph, &color, "Alpha", "principled", "Alpha")?;
        }
    }

    if let Some(slot) = &textures.metallic_roughness {
        let color = texture_node(&mut graph, "metal", slot)?;
        insert(
            &mut graph,
            "metal_split",
            GraphNode::new("ShaderNodeSeparateColor"),
        )?;
        connect(&mut graph, &color, "Color", "metal_split", "Color")?;
        multiply_channel(
            &mut graph,
            "metal_split",
            "Blue",
            "metal",
            material.metallic,
            "principled",
            "Metallic",
        )?;
        multiply_channel(
            &mut graph,
            "metal_split",
            "Green",
            "rough",
            material.roughness,
            "principled",
            "Roughness",
        )?;
    }

    if let Some(slot) = &textures.normal {
        let color = texture_node(&mut graph, "normal", slot)?;
        let mut normal_map = GraphNode::new("ShaderNodeNormalMap");
        normal_map
            .properties
            .insert("space".to_owned(), json!("TANGENT"));
        normal_map
            .properties
            .insert("potter_gltf_imported_normal_map".to_owned(), json!(true));
        // The node reads the normal texture's layer, as Blender's importer sets it; the render
        // layer (no name, or an empty one) is the node default.
        if let Some(layer) = slot
            .texture
            .uv_layer
            .as_deref()
            .filter(|layer| !layer.is_empty())
        {
            normal_map
                .properties
                .insert("uv_map".to_owned(), json!(layer));
        }
        normal_map
            .inputs
            .insert("Strength".to_owned(), json!(textures.normal_scale));
        insert(&mut graph, "normal_strength", normal_map)?;
        connect(&mut graph, &color, "Color", "normal_strength", "Color")?;
        connect(
            &mut graph,
            "normal_strength",
            "Normal",
            "principled",
            "Normal",
        )?;
    }

    if let Some(slot) = &textures.emissive {
        let color = texture_node(&mut graph, "emissive", slot)?;
        // Blender folds a grey emissive factor into the strength and multiplies a coloured one into
        // the texture color. A zero grey factor therefore turns the emission off.
        let factor = material.emission_color;
        if crate::float::equal_f64_array(&factor, &[factor[0]; 3]) {
            connect(&mut graph, &color, "Color", "principled", "Emission Color")?;
            set_input(
                &mut graph,
                "principled",
                "Emission Strength",
                json!(material.emission_strength * factor[0]),
            )?;
        } else {
            multiply_color(
                &mut graph,
                &color,
                "Color",
                "emissive",
                factor,
                "principled",
                "Emission Color",
            )?;
        }
    }

    if let Some(slot) = &textures.occlusion {
        let color = texture_node(&mut graph, "occlusion", slot)?;
        insert(
            &mut graph,
            "occlusion_split",
            GraphNode::new("ShaderNodeSeparateColor"),
        )?;
        connect(&mut graph, &color, "Color", "occlusion_split", "Color")?;
        if occlusion_group {
            let mut group = GraphNode::new("ShaderNodeGroup");
            group
                .properties
                .insert("node_tree".to_owned(), json!({ "name": OUTPUT_GROUP_NAME }));
            insert(&mut graph, "output_group", group)?;
            connect(
                &mut graph,
                "occlusion_split",
                "Red",
                "output_group",
                "Occlusion",
            )?;
        }
    }
    Ok(graph)
}

/// An Image Texture node for `slot`, fed by a Mapping node when the slot has a transform, and returns
/// the name of the Image Texture node. A named UV layer is read through a UV Map node; the render
/// layer (no name) through Texture Coordinate `UV`.
fn texture_node(graph: &mut NodeGroup, name: &str, slot: &SlotTexture) -> Result<String> {
    let tex = format!("{name}_tex");
    let mut source = match &slot.texture.uv_layer {
        Some(layer) => {
            let uv_node = format!("{name}_uv");
            let mut node = GraphNode::new("ShaderNodeUVMap");
            node.properties.insert("uv_map".to_owned(), json!(layer));
            insert(graph, &uv_node, node)?;
            (uv_node, "UV".to_owned())
        }
        None => ("coord".to_owned(), "UV".to_owned()),
    };
    if let Some(transform) = &slot.transform {
        let map = format!("{name}_map");
        let (location, rotation, scale) = mapping_values(transform);
        let mut mapping = GraphNode::new("ShaderNodeMapping");
        mapping
            .properties
            .insert("vector_type".to_owned(), json!("POINT"));
        mapping
            .inputs
            .insert("Location".to_owned(), json!(location));
        mapping
            .inputs
            .insert("Rotation".to_owned(), json!(rotation));
        mapping.inputs.insert("Scale".to_owned(), json!(scale));
        insert(graph, &map, mapping)?;
        connect(graph, &source.0, &source.1, &map, "Vector")?;
        source = (map, "Vector".to_owned());
    }
    let mut image = GraphNode::new("ShaderNodeTexImage");
    image
        .properties
        .insert("image".to_owned(), json!(slot.texture.image.as_str()));
    let interpolation = match slot.texture.interpolation {
        ImageInterpolation::Closest => "Closest",
        ImageInterpolation::Linear => "Linear",
    };
    image
        .properties
        .insert("interpolation".to_owned(), json!(interpolation));
    insert(graph, &tex, image)?;
    connect(graph, &source.0, &source.1, &tex, "Vector")?;
    Ok(tex)
}

/// Links `color` (a Color output of `node`) into `to_socket`, multiplied by the RGB `factor`.
fn multiply_color(
    graph: &mut NodeGroup,
    node: &str,
    socket: &str,
    name: &str,
    factor: [f64; 3],
    to: &str,
    to_socket: &str,
) -> Result<()> {
    let mix = format!("{name}_mix");
    let mut mix_node = GraphNode::new("ShaderNodeMixRGB");
    mix_node
        .properties
        .insert("blend_type".to_owned(), json!("MULTIPLY"));
    mix_node.inputs.insert("Fac".to_owned(), json!(1.0));
    mix_node.inputs.insert(
        "Color1".to_owned(),
        json!([factor[0], factor[1], factor[2], 1.0]),
    );
    insert(graph, &mix, mix_node)?;
    connect(graph, node, socket, &mix, "Color2")?;
    connect(graph, &mix, "Color", to, to_socket)
}

/// Links `channel` of the Separate Color `split` into `to_socket`, multiplied by `factor` (a Math node
/// unless the factor is one).
fn multiply_channel(
    graph: &mut NodeGroup,
    split: &str,
    channel: &str,
    name: &str,
    factor: f64,
    to: &str,
    to_socket: &str,
) -> Result<()> {
    if crate::float::equal_f64(factor, 1.0) {
        return connect(graph, split, channel, to, to_socket);
    }
    let math = format!("{name}_factor");
    let mut node = GraphNode::new("ShaderNodeMath");
    node.properties
        .insert("operation".to_owned(), json!("MULTIPLY"));
    node.inputs.insert("Value_001".to_owned(), json!(factor));
    insert(graph, &math, node)?;
    connect(graph, split, channel, &math, "Value")?;
    connect(graph, &math, "Value", to, to_socket)
}

fn insert(graph: &mut NodeGroup, name: &str, node: GraphNode) -> Result<()> {
    let id = Id::new(name)?;
    if graph.nodes.contains_key(&id) {
        return Err(graph_error(format!("graph node `{name}` is defined twice")));
    }
    graph.nodes.insert(id, node);
    Ok(())
}

fn set_input(graph: &mut NodeGroup, node: &str, socket: &str, value: Value) -> Result<()> {
    let node = graph
        .nodes
        .get_mut(&Id::new(node)?)
        .ok_or_else(|| graph_error(format!("graph node `{node}` is missing")))?;
    node.inputs.insert(socket.to_owned(), value);
    Ok(())
}

fn connect(
    graph: &mut NodeGroup,
    from: &str,
    from_socket: &str,
    to: &str,
    to_socket: &str,
) -> Result<()> {
    graph.links.push(GraphLink {
        from_node: Id::new(from)?,
        from_socket: from_socket.to_owned(),
        to_node: Id::new(to)?,
        to_socket: to_socket.to_owned(),
    });
    Ok(())
}

/// Restores the internal glTF tangent-frame marker from the exact green-flip graph emitted by the
/// Blender bridge. The graph uses standard Blender nodes; this keeps Potter's V-down representation
/// canonical after a `.blend` round trip.
pub(crate) fn restore_gltf_normal_map_handedness(group: &mut NodeGroup) {
    if group.kind != GraphKind::Shader {
        return;
    }
    let mut restored = Vec::new();
    for (normal_id, normal) in &group.nodes {
        if normal.node_type != "ShaderNodeNormalMap" {
            continue;
        }
        let Some(color_link) = only_link(&group.links, |link| {
            link.to_node == *normal_id && link.to_socket == "Color"
        }) else {
            continue;
        };
        let add_id = color_link.from_node.clone();
        let Some(add) = group.nodes.get(&add_id) else {
            continue;
        };
        // Only collapse the bridge's named helpers; user-authored vector math remains untouched.
        if !add.name.starts_with("Potter GLTF Normal Green Add")
            || !is_vector_math(add, "ADD", [0.0, 1.0, 0.0])
            || group
                .links
                .iter()
                .filter(|link| link.to_node == add_id)
                .count()
                != 1
            || group
                .links
                .iter()
                .filter(|link| link.from_node == add_id)
                .count()
                != 1
        {
            continue;
        }
        let Some(multiply_link) = only_link(&group.links, |link| {
            link.to_node == add_id && link.to_socket == "Vector"
        }) else {
            continue;
        };
        let multiply_id = multiply_link.from_node.clone();
        let Some(multiply) = group.nodes.get(&multiply_id) else {
            continue;
        };
        if !multiply
            .name
            .starts_with("Potter GLTF Normal Green Multiply")
            || !is_vector_math(multiply, "MULTIPLY", [1.0, -1.0, 1.0])
            || group
                .links
                .iter()
                .filter(|link| link.to_node == multiply_id)
                .count()
                != 1
            || group
                .links
                .iter()
                .filter(|link| link.from_node == multiply_id)
                .count()
                != 1
        {
            continue;
        }
        let Some(source_link) = only_link(&group.links, |link| {
            link.to_node == multiply_id && link.to_socket == "Vector"
        }) else {
            continue;
        };
        let source_id = source_link.from_node.clone();
        if source_link.from_socket != "Color"
            || group
                .nodes
                .get(&source_id)
                .is_none_or(|source| source.node_type != "ShaderNodeTexImage")
        {
            continue;
        }
        restored.push((
            normal_id.clone(),
            add_id,
            multiply_id,
            GraphLink {
                from_node: source_id,
                from_socket: source_link.from_socket.clone(),
                to_node: normal_id.clone(),
                to_socket: "Color".to_owned(),
            },
        ));
    }
    if restored.is_empty() {
        return;
    }
    let mut removed = std::collections::BTreeSet::new();
    let mut links = Vec::with_capacity(restored.len());
    for (normal_id, add_id, multiply_id, link) in restored {
        removed.insert(add_id);
        removed.insert(multiply_id);
        if let Some(normal) = group.nodes.get_mut(&normal_id) {
            normal
                .properties
                .insert("potter_gltf_imported_normal_map".to_owned(), json!(true));
        }
        links.push(link);
    }
    group
        .links
        .retain(|link| !removed.contains(&link.from_node) && !removed.contains(&link.to_node));
    for node_id in &removed {
        group.nodes.remove(node_id);
    }
    group.links.extend(links);
}

fn only_link<'a>(
    links: &'a [GraphLink],
    predicate: impl Fn(&GraphLink) -> bool,
) -> Option<&'a GraphLink> {
    let mut matching = links.iter().filter(|link| predicate(link));
    let first = matching.next()?;
    matching.next().is_none().then_some(first)
}

fn is_vector_math(node: &GraphNode, operation: &str, second: [f64; 3]) -> bool {
    node.node_type == "ShaderNodeVectorMath"
        && node.properties.get("operation").and_then(Value::as_str) == Some(operation)
        && node
            .inputs
            .get("Vector_001")
            .and_then(Value::as_array)
            .is_some_and(|values| {
                values.len() == second.len()
                    && values
                        .iter()
                        .zip(second)
                        .all(|(value, expected)| value.as_f64() == Some(expected))
            })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_refuses_a_node_name_that_is_already_used() {
        let mut graph = NodeGroup::new("test", GraphKind::Shader);
        assert!(insert(&mut graph, "node", GraphNode::new("ShaderNodeMapping")).is_ok());
        assert!(insert(&mut graph, "node", GraphNode::new("ShaderNodeNormalMap")).is_err());
        assert_eq!(graph.nodes.len(), 1);
    }

    fn normal_map_properties(uv_layer: Option<&str>) -> Result<(Option<Value>, Option<Value>)> {
        let textures = GraphTextures {
            base_color: None,
            metallic_roughness: None,
            normal: Some(SlotTexture {
                texture: TextureRef {
                    image: Id::new("normal_image")?,
                    uv_layer: uv_layer.map(str::to_owned),
                    interpolation: ImageInterpolation::Linear,
                },
                transform: None,
            }),
            normal_scale: 1.0,
            emissive: None,
            occlusion: None,
        };
        let graph = build_graph(&Material::default(), &textures, false)?;
        let properties = &graph
            .nodes
            .get(&Id::new("normal_strength")?)
            .ok_or_else(|| graph_error("normal map node is missing"))?
            .properties;
        Ok((
            properties.get("uv_map").cloned(),
            properties.get("potter_gltf_imported_normal_map").cloned(),
        ))
    }

    #[test]
    fn importer_generated_normal_map_is_marked_without_changing_uv_selection() -> Result<()> {
        assert_eq!(
            normal_map_properties(Some("UVMap.001"))?,
            (Some(json!("UVMap.001")), Some(json!(true)))
        );
        assert_eq!(normal_map_properties(None)?, (None, Some(json!(true))));
        assert_eq!(normal_map_properties(Some(""))?, (None, Some(json!(true))));
        Ok(())
    }
}
