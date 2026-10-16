//! Material node graphs for glTF texture slots that carry a `KHR_texture_transform`.
//!
//! Blender's glTF importer maps such a texture through a Mapping node: Image Texture `Vector` <-
//! Mapping (POINT) `Vector` <- Texture Coordinate `UV`, with the transform in Mapping's Location,
//! Rotation and Scale. This module builds the same shape, feeding the Principled BSDF in place of the
//! material's texture fields, so the transform survives into the shader and back out of the exporter.

use serde_json::{Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    graph::{GraphKind, GraphLink, GraphNode, NodeGroup},
    image::ImageInterpolation,
    model::{Id, Material, SceneDoc, TextureRef},
};

/// A glTF `KHR_texture_transform`: texture coordinates are `T * R * S` applied to the UV.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct TextureTransform {
    pub(crate) offset: [f64; 2],
    pub(crate) rotation: f64,
    pub(crate) scale: [f64; 2],
}

impl TextureTransform {
    /// Whether the transform leaves every UV unchanged, so it needs no Mapping node.
    pub(crate) fn is_identity(&self) -> bool {
        crate::float::equal_f64_array(&self.offset, &[0.0; 2])
            && crate::float::equal_f64(self.rotation, 0.0)
            && crate::float::equal_f64_array(&self.scale, &[1.0; 2])
    }

    /// Mapping Location, Rotation and Scale, as Blender's glTF importer sets them
    /// (`io_scene_gltf2` `texture_transform_gltf_to_blender`).
    fn mapping(&self) -> ([f64; 3], [f64; 3], [f64; 3]) {
        let (sin, cos) = self.rotation.sin_cos();
        let location = [
            self.offset[0] + self.scale[1] * sin,
            1.0 - self.offset[1] - self.scale[1] * cos,
            0.0,
        ];
        (
            location,
            [0.0, 0.0, self.rotation],
            [self.scale[0], self.scale[1], 1.0],
        )
    }
}

/// The transforms of one material's texture slots. A slot without a transform has `None`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct SlotTransforms {
    pub(crate) base_color: Option<TextureTransform>,
    pub(crate) metallic_roughness: Option<TextureTransform>,
    pub(crate) normal: Option<TextureTransform>,
}

impl SlotTransforms {
    /// Whether no slot has a transform, so the material keeps its texture fields.
    pub(crate) fn is_empty(&self) -> bool {
        self.base_color.is_none() && self.metallic_roughness.is_none() && self.normal.is_none()
    }
}

fn graph_error(message: impl Into<String>) -> PotError {
    PotError::new(ErrorCode::ImportFailed, message)
}

/// Moves the texture slots of `material_id` into a shader graph when a slot has a transform.
///
/// The Principled BSDF takes the material's factors and each texture's color (or channel) through
/// the node links; the texture fields are cleared, since the graph now holds them. Does nothing when
/// every slot is untransformed or when the material already has a graph.
pub(crate) fn move_textures_to_graph(
    doc: &mut SceneDoc,
    material_id: &Id,
    transforms: &SlotTransforms,
) -> Result<()> {
    if transforms.is_empty() {
        return Ok(());
    }
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
    let graph = build_graph(&material, transforms)?;
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

fn build_graph(material: &Material, transforms: &SlotTransforms) -> Result<NodeGroup> {
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
            1.0
        ]),
    );
    principled.inputs.insert(
        "Emission Strength".to_owned(),
        json!(material.emission_strength),
    );
    principled
        .inputs
        .insert("Alpha".to_owned(), json!(material.base_color[3]));
    let factor = material.base_color;
    insert(&mut graph, "principled", principled)?;

    let mut output = GraphNode::new("OutputMaterial");
    output
        .properties
        .insert("is_active_output".to_owned(), json!(true));
    insert(&mut graph, "output", output)?;
    connect(&mut graph, "principled", "BSDF", "output", "Surface")?;
    insert(&mut graph, "coord", GraphNode::new("ShaderNodeTexCoord"))?;

    // Base color: the texture color times the factor, as Blender's Mix (multiply) does.
    if let Some(texture) = &material.base_color_texture {
        let color = texture_node(&mut graph, "base", texture, transforms.base_color)?;
        if crate::float::equal_f64_array(&[factor[0], factor[1], factor[2]], &[1.0; 3]) {
            connect(&mut graph, &color, "Color", "principled", "Base Color")?;
        } else {
            let mut mix = GraphNode::new("ShaderNodeMixRGB");
            mix.properties
                .insert("blend_type".to_owned(), json!("MULTIPLY"));
            mix.inputs.insert("Fac".to_owned(), json!(1.0));
            mix.inputs.insert(
                "Color1".to_owned(),
                json!([factor[0], factor[1], factor[2], 1.0]),
            );
            insert(&mut graph, "base_mix", mix)?;
            connect(&mut graph, &color, "Color", "base_mix", "Color2")?;
            connect(&mut graph, "base_mix", "Color", "principled", "Base Color")?;
        }
        if material.alpha_mode != "opaque" {
            connect(&mut graph, &color, "Alpha", "principled", "Alpha")?;
        }
    } else {
        set_input(&mut graph, "principled", "Base Color", json!(factor))?;
    }

    // Metallic (blue) and roughness (green) share one texture; each factor multiplies its channel.
    if let Some(texture) = material
        .metallic_texture
        .as_ref()
        .or(material.roughness_texture.as_ref())
    {
        let color = texture_node(&mut graph, "metal", texture, transforms.metallic_roughness)?;
        insert(&mut graph, "metal_split", separate_color())?;
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
    } else {
        set_input(
            &mut graph,
            "principled",
            "Metallic",
            json!(material.metallic),
        )?;
        set_input(
            &mut graph,
            "principled",
            "Roughness",
            json!(material.roughness),
        )?;
    }

    if let Some(texture) = &material.normal_texture {
        let color = texture_node(&mut graph, "normal", texture, transforms.normal)?;
        let mut normal_map = GraphNode::new("ShaderNodeNormalMap");
        normal_map
            .properties
            .insert("space".to_owned(), json!("TANGENT"));
        normal_map.inputs.insert("Strength".to_owned(), json!(1.0));
        insert(&mut graph, "normal_map", normal_map)?;
        connect(&mut graph, &color, "Color", "normal_map", "Color")?;
        connect(&mut graph, "normal_map", "Normal", "principled", "Normal")?;
    }
    Ok(graph)
}

/// An Image Texture node for `texture`, fed by Mapping when the slot has a transform, and the node
/// id whose `Color` and `Alpha` outputs carry the texture. The UVs are Texture Coordinate `UV`.
fn texture_node(
    graph: &mut NodeGroup,
    name: &str,
    texture: &TextureRef,
    transform: Option<TextureTransform>,
) -> Result<String> {
    let tex = format!("{name}_tex");
    let mut source = ("coord".to_owned(), "UV".to_owned());
    if let Some(transform) = transform.filter(|transform| !transform.is_identity()) {
        let map = format!("{name}_map");
        let (location, rotation, scale) = transform.mapping();
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
        .insert("image".to_owned(), json!(texture.image.as_str()));
    let interpolation = match texture.interpolation {
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

fn separate_color() -> GraphNode {
    GraphNode::new("ShaderNodeSeparateColor")
}

/// Links `channel` of `split` into `to_socket`, multiplied by `factor` (a Math node when the factor
/// is not one).
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
    graph.nodes.insert(Id::new(name)?, node);
    Ok(())
}

fn set_input(
    graph: &mut NodeGroup,
    node: &str,
    socket: &str,
    value: serde_json::Value,
) -> Result<()> {
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

/// Whether a glTF file has a `KHR_texture_transform` on an emissive or occlusion texture: slots Potter
/// has no texture field for, so the transform cannot be kept. The base color, metallic-roughness and
/// normal transforms are imported into the graph; VRM 1.0 `MToon` transforms are read by the VRM
/// importer.
pub(crate) fn unhandled_texture_transforms(root: &Value) -> bool {
    root["materials"].as_array().is_some_and(|materials| {
        materials.iter().any(|material| {
            ["emissiveTexture", "occlusionTexture"]
                .iter()
                .any(|slot| material[*slot]["extensions"]["KHR_texture_transform"].is_object())
        })
    })
}
