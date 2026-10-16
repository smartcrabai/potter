//! Material node graphs for textures that Potter's texture fields cannot hold, and for the texture fields written to .blend.
//!
//! Blender's glTF importer builds a node graph for every material with a transformed texture, an
//! emissive texture, an occlusion texture, or a normal map scale. This module builds the same shapes:
//!
//! - an Image Texture whose `Vector` comes from a Mapping node (POINT) fed by Texture Coordinate `UV`
//!   when the slot has a `KHR_texture_transform`, with Blender's Location, Rotation and Scale;
//! - the base color, metallic-roughness and emissive colors into the Principled BSDF, multiplied by
//!   their factors as Blender does;
//! - the normal map through a Normal Map node whose Strength is the `normalTexture.scale`;
//! - occlusion through a Separate Color (Red) into the `glTF Material Output` group, which PR10's
//!   resolver reads back as glTF occlusion.

use std::{borrow::Cow, collections::BTreeSet};

use serde_json::{Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    exchange::{
        Loss, is_default_material_graph,
        material_surface::{TextureTransform, feature_id},
    },
    geom::{Mesh, uv_layers},
    graph::{GraphKind, GraphLink, GraphNode, GraphSocket, NodeGroup},
    image::ImageInterpolation,
    model::{Id, ImageColorspace, Material, SceneDoc, TextureRef},
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
    /// Separate roughness (G) and metallic (B) textures of a Potter material's fields.
    pub(crate) roughness: Option<SlotTexture>,
    pub(crate) metallic: Option<SlotTexture>,
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
            roughness: None,
            metallic: None,
            normal: plain(&material.normal_texture),
            normal_scale: 1.0,
            emissive: None,
            occlusion: None,
        }
    }

    /// The texture fields of a material, for a .blend export. Roughness and metallic each read their
    /// own image (G and B); one texture that both fields name is the packed texture glTF uses.
    pub(crate) fn from_field_refs(
        base_color: Option<TextureRef>,
        roughness: Option<TextureRef>,
        metallic: Option<TextureRef>,
        normal: Option<TextureRef>,
    ) -> Self {
        let plain = |texture: Option<TextureRef>| {
            texture.map(|texture| SlotTexture {
                texture,
                transform: None,
            })
        };
        let packed = roughness.is_some() && roughness == metallic;
        Self {
            base_color: plain(base_color),
            metallic_roughness: if packed {
                plain(roughness.clone())
            } else {
                None
            },
            roughness: if packed { None } else { plain(roughness) },
            metallic: if packed { None } else { plain(metallic) },
            normal: plain(normal),
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

/// The document a .blend export sends to the bridge, and the losses that the export reports.
///
/// A field material (texture fields, no custom graph) gets the shader graph its fields describe,
/// built by `build_graph`: an Image Texture per field image, read through a UV Map node that names
/// the layer (the render layer when the field names none and every user mesh agrees on it), linked
/// into the Principled BSDF. Each image a data map (roughness, metallic, normal) reads must be
/// Non-Color in Blender, which is per image datablock: such an image is exported as Non-Color unless
/// a color map also reads it, which is a loss. A texture field on a material with a custom graph is
/// not written and is a loss. The native document is never changed; `Cow::Borrowed` is returned when
/// the export needs no change.
///
/// # Errors
///
/// Returns a graph error when a field material cannot take its generated graph.
pub(crate) fn blend_export_document(doc: &SceneDoc) -> Result<(Cow<'_, SceneDoc>, Vec<Loss>)> {
    let mut losses = Vec::new();
    let mut plans = Vec::<(Id, GraphTextures)>::new();
    let mut data_images = BTreeSet::<Id>::new();
    let mut color_images = BTreeSet::<Id>::new();
    for (material_id, material) in &doc.materials {
        if !has_texture_field(material) {
            continue;
        }
        if material
            .node_tree
            .as_ref()
            .is_some_and(|graph_id| !is_default_material_graph(doc, graph_id))
        {
            losses.push(Loss {
                feature_id: feature_id::MATERIAL_TEXTURES.to_owned(),
                data_id: Some(material_id.as_str().to_owned()),
                reason: "the material's shader graph is authoritative; its texture fields are not written to .blend"
                    .to_owned(),
                suggestion: Some(
                    "connect the textures in the shader graph instead of the texture fields"
                        .to_owned(),
                ),
            });
            continue;
        }
        let users = material_users(doc, material_id);
        let [base, roughness, metallic, normal] = [
            ("base color", &material.base_color_texture),
            ("roughness", &material.roughness_texture),
            ("metallic", &material.metallic_texture),
            ("normal", &material.normal_texture),
        ]
        .map(|(role, field)| {
            field
                .as_ref()
                .map(|reference| export_reference(&users, material_id, role, reference, &mut losses))
        });
        color_images.extend(base.iter().map(|texture| texture.image.clone()));
        data_images.extend(
            [&roughness, &metallic, &normal]
                .into_iter()
                .flatten()
                .map(|texture| texture.image.clone()),
        );
        plans.push((
            material_id.clone(),
            GraphTextures::from_field_refs(base, roughness, metallic, normal),
        ));
    }
    for graph in doc.node_groups.values() {
        color_images.extend(graph.nodes.values().filter_map(|node| {
            let image = node.properties.get("image")?;
            let name = image
                .as_str()
                .or_else(|| image.get("name").and_then(Value::as_str))?;
            Id::new(name).ok()
        }));
    }
    let mut recolored = Vec::<Id>::new();
    for image_id in &data_images {
        let Some(image) = doc.images.get(image_id) else {
            continue;
        };
        if image.colorspace == ImageColorspace::NonColor {
            continue;
        }
        if color_images.contains(image_id) {
            losses.push(Loss {
                feature_id: feature_id::MATERIAL_TEXTURES.to_owned(),
                data_id: Some(image_id.as_str().to_owned()),
                reason: "the image is a data map (roughness, metallic, or normal) and also a color map; Blender holds one colorspace per image, so one of the two reads the wrong colorspace"
                    .to_owned(),
                suggestion: Some("use separate images for the data maps and the color maps".to_owned()),
            });
        } else {
            recolored.push(image_id.clone());
        }
    }
    if plans.is_empty() && recolored.is_empty() {
        return Ok((Cow::Borrowed(doc), losses));
    }
    let mut export = doc.clone();
    for image_id in &recolored {
        if let Some(image) = export.images.get_mut(image_id) {
            image.colorspace = ImageColorspace::NonColor;
        }
    }
    for (material_id, textures) in &plans {
        let material = export
            .materials
            .get_mut(material_id)
            .ok_or_else(|| graph_error(format!("material `{material_id}` is missing")))?;
        // The generated default graph (if any) is replaced; drop it when nothing else uses it.
        let previous = material.node_tree.take();
        move_textures_to_graph(&mut export, material_id, textures)?;
        if let Some(graph_id) = previous
            && !export
                .materials
                .values()
                .any(|material| material.node_tree.as_ref() == Some(&graph_id))
        {
            export.node_groups.remove(&graph_id);
        }
    }
    Ok((Cow::Owned(export), losses))
}

fn has_texture_field(material: &Material) -> bool {
    material.base_color_texture.is_some()
        || material.roughness_texture.is_some()
        || material.metallic_texture.is_some()
        || material.normal_texture.is_some()
}

/// The meshes of the objects whose slots hold `material_id`.
fn material_users<'a>(doc: &'a SceneDoc, material_id: &Id) -> Vec<&'a Mesh> {
    doc.nodes
        .values()
        .filter(|node| node.materials.contains(material_id))
        .filter_map(|node| node.data.as_ref().and_then(|data| doc.data_blocks.get(data)))
        .filter_map(|data| data.mesh.as_ref())
        .collect()
}

/// A texture reference with the layer Blender should sample. A named layer stays. The render layer
/// (no name) becomes the layer that the users' meshes render with, when they agree; when they do not,
/// it stays unnamed, so Blender's Texture Coordinate `UV` samples each object's own render layer. A
/// name that a user mesh lacks is a loss there, as Potter's own resolver reports it.
fn export_reference(
    users: &[&Mesh],
    material_id: &Id,
    role: &str,
    reference: &TextureRef,
    losses: &mut Vec<Loss>,
) -> TextureRef {
    let mut names = BTreeSet::<Option<String>>::new();
    let mut missing = false;
    for mesh in users {
        let store = uv_layers::read(mesh);
        let (index, unknown) = store.resolve(reference.uv_layer.as_deref());
        missing |= unknown.is_some();
        names.insert(
            index
                .and_then(|index| store.layers.get(index))
                .map(|layer| layer.name.clone()),
        );
    }
    if missing {
        losses.push(Loss {
            feature_id: feature_id::UV_LAYER_MISSING.to_owned(),
            data_id: Some(material_id.as_str().to_owned()),
            reason: format!(
                "the {role} texture names a UV layer that a mesh using the material does not have; that mesh samples its render layer"
            ),
            suggestion: None,
        });
    }
    let uv_layer = if names.len() <= 1 {
        names
            .into_iter()
            .next()
            .flatten()
            .or_else(|| reference.uv_layer.clone())
    } else if reference.uv_layer.is_some() {
        losses.push(Loss {
            feature_id: feature_id::MESH_UV_LAYER.to_owned(),
            data_id: Some(material_id.as_str().to_owned()),
            reason: format!(
                "meshes using the material resolve the {role} texture to different UV layers; the texture names one layer"
            ),
            suggestion: None,
        });
        reference.uv_layer.clone()
    } else {
        None
    };
    TextureRef {
        uv_layer,
        ..reference.clone()
    }
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
    let emission = material.emission_color;
    principled.inputs.insert(
        "Emission Color".to_owned(),
        json!([emission[0], emission[1], emission[2], 1.0]),
    );
    principled.inputs.insert(
        "Emission Strength".to_owned(),
        json!(material.emission_strength),
    );
    principled.inputs.insert(
        "Alpha".to_owned(),
        json!(if material.alpha_mode == "opaque" {
            1.0
        } else {
            material.base_color[3]
        }),
    );
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
            multiply_value(
                &mut graph,
                &color,
                "Alpha",
                "base_alpha",
                material.base_color[3],
                "principled",
                "Alpha",
            )?;
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

    if textures.metallic_roughness.is_some()
        && (textures.roughness.is_some() || textures.metallic.is_some())
    {
        return Err(graph_error(
            "a packed metallicRoughness texture cannot be combined with separate roughness or metallic textures",
        ));
    }
    if let Some(slot) = &textures.roughness {
        channel_texture(
            &mut graph,
            "roughness",
            slot,
            "Green",
            material.roughness,
            "Roughness",
        )?;
    }
    if let Some(slot) = &textures.metallic {
        channel_texture(
            &mut graph,
            "metallic",
            slot,
            "Blue",
            material.metallic,
            "Metallic",
        )?;
    }

    if let Some(slot) = &textures.normal {
        let color = texture_node(&mut graph, "normal", slot)?;
        let mut normal_map = GraphNode::new("ShaderNodeNormalMap");
        normal_map
            .properties
            .insert("space".to_owned(), json!("TANGENT"));
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

/// Links `channel` (a Separate Color output) of `slot`'s texture into the Principled BSDF
/// `to_socket`, multiplied by `factor`. Potter's scalar fields are factors of their textures.
fn channel_texture(
    graph: &mut NodeGroup,
    name: &str,
    slot: &SlotTexture,
    channel: &str,
    factor: f64,
    to_socket: &str,
) -> Result<()> {
    let color = texture_node(graph, name, slot)?;
    let split = format!("{name}_split");
    insert(graph, &split, GraphNode::new("ShaderNodeSeparateColor"))?;
    connect(graph, &color, "Color", &split, "Color")?;
    multiply_channel(graph, &split, channel, name, factor, "principled", to_socket)
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

/// Links a scalar output into `to_socket`, multiplied by `factor` unless the factor is one.
fn multiply_value(
    graph: &mut NodeGroup,
    from_node: &str,
    from_socket: &str,
    name: &str,
    factor: f64,
    to: &str,
    to_socket: &str,
) -> Result<()> {
    if crate::float::equal_f64(factor, 1.0) {
        return connect(graph, from_node, from_socket, to, to_socket);
    }
    let math = format!("{name}_factor");
    let mut math_node = GraphNode::new("ShaderNodeMath");
    math_node
        .properties
        .insert("operation".to_owned(), json!("MULTIPLY"));
    math_node.inputs.insert("Value_001".to_owned(), json!(factor));
    insert(graph, &math, math_node)?;
    connect(graph, from_node, from_socket, &math, "Value")?;
    connect(graph, &math, "Value", to, to_socket)
}

/// Links `channel` of the Separate Color `split` into `to_socket`, multiplied by `factor`.
fn multiply_channel(
    graph: &mut NodeGroup,
    split: &str,
    channel: &str,
    name: &str,
    factor: f64,
    to: &str,
    to_socket: &str,
) -> Result<()> {
    multiply_value(graph, split, channel, name, factor, to, to_socket)
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

    #[test]
    fn field_base_alpha_multiplies_texture_alpha_by_material_factor() {
        let texture = TextureRef {
            image: Id::new("paint").unwrap(),
            uv_layer: None,
            interpolation: ImageInterpolation::Linear,
        };
        let textures = GraphTextures::from_field_refs(Some(texture), None, None, None);
        let material = Material {
            base_color: [1.0, 1.0, 1.0, 0.35],
            alpha_mode: "blend".to_owned(),
            ..Material::default()
        };
        let graph = build_graph(&material, &textures, false).expect("graph");
        let math = graph
            .nodes
            .get(&Id::new("base_alpha_factor").unwrap())
            .expect("base alpha multiply node");
        assert_eq!(math.node_type, "ShaderNodeMath");
        assert_eq!(math.properties.get("operation"), Some(&json!("MULTIPLY")));
        assert_eq!(math.inputs.get("Value_001"), Some(&json!(0.35)));
        assert!(graph.links.contains(&GraphLink {
            from_node: Id::new("base_tex").unwrap(),
            from_socket: "Alpha".to_owned(),
            to_node: Id::new("base_alpha_factor").unwrap(),
            to_socket: "Value".to_owned(),
        }));
        assert!(graph.links.contains(&GraphLink {
            from_node: Id::new("base_alpha_factor").unwrap(),
            from_socket: "Value".to_owned(),
            to_node: Id::new("principled").unwrap(),
            to_socket: "Alpha".to_owned(),
        }));
    }
}
