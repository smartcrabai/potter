//! Material node graphs for glTF textures and Potter texture fields that the graph must hold, including the texture fields written to .blend.
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

use std::collections::BTreeSet;

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
    /// Separate roughness and metallic textures of a Potter material, each read through its Red channel.
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

    /// The native texture fields of a Potter material, each read through its Red channel as the field is, including when two fields reference one image.
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
        Self {
            base_color: plain(base_color),
            metallic_roughness: None,
            roughness: plain(roughness),
            metallic: plain(metallic),
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
    let graph_id = texture_graph_id(material_id)?;
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
            multiply_channel(
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

    if textures.metallic_roughness.is_some()
        && (textures.roughness.is_some() || textures.metallic.is_some())
    {
        return Err(graph_error(
            "a packed metallicRoughness texture cannot be combined with separate roughness or metallic textures",
        ));
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

    if let Some(slot) = &textures.roughness {
        channel_texture(
            &mut graph,
            "roughness",
            slot,
            "Red",
            material.roughness,
            "Roughness",
        )?;
    }
    if let Some(slot) = &textures.metallic {
        channel_texture(
            &mut graph,
            "metallic",
            slot,
            "Red",
            material.metallic,
            "Metallic",
        )?;
    }

    if let Some(slot) = &textures.normal {
        let color = texture_node(&mut graph, "normal", slot)?;
        insert(
            &mut graph,
            "normal_strength",
            normal_map_node(slot, textures.normal_scale),
        )?;
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
        None => {
            let coord = format!("{name}_coord");
            insert(graph, &coord, GraphNode::new("ShaderNodeTexCoord"))?;
            (coord, "UV".to_owned())
        }
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

/// Links `channel` of a `split` output (a Separate Color channel, or a texture's Alpha) into `to_socket`,
/// multiplied by `factor` (a Math node unless the factor is one).
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

fn only_link(links: &[GraphLink], predicate: impl Fn(&GraphLink) -> bool) -> Option<&GraphLink> {
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

/// Material texture work of a .blend export: the shader graphs that the export writes in place of each material's
/// graph, the default graphs they replace, and the losses for shapes no graph can hold. The source document is not
/// changed; the export document takes the edits.
#[derive(Default)]
pub(crate) struct BlendTextureEdits {
    /// Material, its export graph's ID, and the graph. Each becomes the material's `node_tree` in the export.
    graphs: Vec<(Id, Id, NodeGroup)>,
    /// Default graphs that generated graphs replace and that no other material uses.
    removed_graphs: Vec<Id>,
}

/// The texture fields of a material, which the export document clears once a graph holds them.
const TEXTURE_FIELDS: [&str; 4] = [
    "base_color_texture",
    "roughness_texture",
    "metallic_texture",
    "normal_texture",
];

/// Plans the .blend texture export of every material with texture fields.
///
/// A material with no custom graph gets the graph its fields describe. A material with an authored graph keeps that
/// graph: the export writes a copy of it with the fields overlaid on its Principled BSDF (`overlay_authored`). A graph
/// shape that takes no overlay is a loss. Each field reads its own UV layer; each image keeps its declared colorspace.
pub(crate) fn plan_blend_textures(doc: &SceneDoc) -> Result<(Vec<Loss>, BlendTextureEdits)> {
    let mut losses = Vec::new();
    let mut plans = Vec::<(&Id, &Material, GraphTextures)>::new();
    for (material_id, material) in &doc.materials {
        if !has_texture_field(material) {
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
            field.as_ref().map(|reference| {
                export_reference(&users, material_id, role, reference, &mut losses)
            })
        });
        plans.push((
            material_id,
            material,
            GraphTextures::from_field_refs(base, roughness, metallic, normal),
        ));
    }
    let planned = plans
        .iter()
        .map(|(material_id, _, _)| *material_id)
        .collect::<BTreeSet<&Id>>();
    let mut graphs = Vec::with_capacity(plans.len());
    let mut replaced = Vec::<Id>::new();
    for (material_id, material, textures) in plans {
        let (graph, graph_id) = match material.node_tree.as_ref() {
            Some(authored) if !is_default_material_graph(doc, authored) => {
                let group = doc
                    .node_groups
                    .get(authored)
                    .ok_or_else(|| graph_error(format!("shader graph `{authored}` is missing")))?;
                match overlay_authored(group, material_id, material, &textures, &mut losses)? {
                    Some(graph) => (graph, export_graph_id("texture-overlay", material_id)?),
                    None => {
                        losses.push(Loss {
                            feature_id: feature_id::MATERIAL_TEXTURES.to_owned(),
                            data_id: Some(material_id.as_str().to_owned()),
                            reason: "the material's shader graph does not drive its active output from a Principled BSDF, so its texture fields cannot be written to .blend"
                                .to_owned(),
                            suggestion: Some(
                                "drive the active output from a Principled BSDF"
                                    .to_owned(),
                            ),
                        });
                        continue;
                    }
                }
            }
            previous => {
                if let Some(previous) = previous {
                    replaced.push(previous.clone());
                }
                (
                    build_graph(material, &textures, false)?,
                    texture_graph_id(material_id)?,
                )
            }
        };
        if material.node_tree.as_ref() != Some(&graph_id) && doc.node_groups.contains_key(&graph_id)
        {
            return Err(graph_error(format!(
                "generated shader graph `{graph_id}` collides with an existing graph"
            )));
        }
        graphs.push((material_id.clone(), graph_id, graph));
    }
    let removed_graphs = replaced
        .into_iter()
        .filter(|graph_id| {
            !graphs.iter().any(|(_, id, _)| id == graph_id)
                && !graph_used_outside(doc, &planned, graph_id)
        })
        .collect();
    Ok((
        losses,
        BlendTextureEdits {
            graphs,
            removed_graphs,
        },
    ))
}

impl BlendTextureEdits {
    /// Writes the graphs and removals into the serialized export document, which the bridge reads.
    pub(crate) fn apply(self, document: &mut Value) -> Result<()> {
        for (material_id, graph_id, graph) in self.graphs {
            let group = serde_json::to_value(&graph).map_err(PotError::internal_json)?;
            object_entry(document, "node_groups")?.insert(graph_id.as_str().to_owned(), group);
            let material = object_entry(document, "materials")?
                .get_mut(material_id.as_str())
                .and_then(Value::as_object_mut)
                .ok_or_else(|| {
                    graph_error(format!(
                        "material `{material_id}` is missing from the export document"
                    ))
                })?;
            material.insert("node_tree".to_owned(), json!(graph_id.as_str()));
            for field in TEXTURE_FIELDS {
                material.insert(field.to_owned(), Value::Null);
            }
        }
        for graph_id in &self.removed_graphs {
            object_entry(document, "node_groups")?.remove(graph_id.as_str());
        }
        Ok(())
    }
}

fn object_entry<'a>(
    document: &'a mut Value,
    key: &str,
) -> Result<&'a mut serde_json::Map<String, Value>> {
    document
        .as_object_mut()
        .and_then(|root| root.entry(key).or_insert_with(|| json!({})).as_object_mut())
        .ok_or_else(|| graph_error(format!("export document field `{key}` is not an object")))
}

fn has_texture_field(material: &Material) -> bool {
    material.base_color_texture.is_some()
        || material.roughness_texture.is_some()
        || material.metallic_texture.is_some()
        || material.normal_texture.is_some()
}

/// The meshes of the objects that use `material_id` in a slot.
fn material_users<'a>(doc: &'a SceneDoc, material_id: &Id) -> Vec<&'a Mesh> {
    doc.nodes
        .values()
        .filter(|node| node.materials.contains(material_id))
        .filter_map(|node| {
            node.data
                .as_ref()
                .and_then(|data_id| doc.data_blocks.get(data_id))
        })
        .filter_map(|data| data.mesh.as_ref())
        .collect()
}

/// A texture reference with the layer Blender should sample. A named layer stays. The render layer (no name) becomes
/// the layer the users' meshes render with when they agree. When they differ it stays unnamed, and Texture Coordinate
/// `UV` samples each object's own render layer. A name that a user mesh lacks is a loss, as Potter's resolver reports it.
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

/// Whether a material outside `planned` still uses the graph `graph_id`.
fn graph_used_outside(doc: &SceneDoc, planned: &BTreeSet<&Id>, graph_id: &Id) -> bool {
    for (material_id, material) in &doc.materials {
        if !planned.contains(material_id) && material.node_tree.as_ref() == Some(graph_id) {
            return true;
        }
    }
    false
}

/// The ID of a graph the export generates for a material, from its purpose.
fn export_graph_id(purpose: &str, material_id: &Id) -> Result<Id> {
    Id::new(format!(
        "shader_{}",
        &crate::hash::sha256(format!("{purpose}:{material_id}").as_bytes())[7..31]
    ))
}

/// The ID of the graph generated for a material's texture fields.
fn texture_graph_id(material_id: &Id) -> Result<Id> {
    export_graph_id("texture-graph", material_id)
}

/// The Normal Map node of a normal texture. It carries the importer's marker, so the bridge writes Blender's V-up
/// green flip and the evaluator reads the canonical frame that native normal textures use. The node reads the
/// texture's layer; the render layer (no name, or an empty one) is the node default.
fn normal_map_node(slot: &SlotTexture, strength: f64) -> GraphNode {
    let mut normal_map = GraphNode::new("ShaderNodeNormalMap");
    normal_map
        .properties
        .insert("space".to_owned(), json!("TANGENT"));
    normal_map
        .properties
        .insert("potter_gltf_imported_normal_map".to_owned(), json!(true));
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
        .insert("Strength".to_owned(), json!(strength));
    normal_map
}

/// The authored graph of a material with native texture fields, copied for export with the fields overlaid on its
/// Principled BSDF as Potter's evaluator applies them after the graph. Base Color multiplies its RGB, Roughness and
/// Metallic multiply their Red channel, and Alpha multiplies its alpha on a non-opaque material (the renderer ignores
/// alpha on an opaque one). A normal texture drives an unlinked Normal input through a Normal Map node. A linked Normal
/// input stays as authored, and its normal texture is a loss: the native normal perturbs the linked vector, which no
/// Blender node reproduces. Every generated node ID starts with a prefix that no authored node ID starts with. Returns
/// `None` when the active output is not driven by a Principled BSDF.
fn overlay_authored(
    group: &NodeGroup,
    material_id: &Id,
    material: &Material,
    textures: &GraphTextures,
    losses: &mut Vec<Loss>,
) -> Result<Option<NodeGroup>> {
    let Some(principled) = surface_principled(group) else {
        return Ok(None);
    };
    let prefix = fresh_prefix(group, "potter_texture");
    let mut graph = group.clone();
    if let Some(slot) = &textures.base_color {
        let base = format!("{prefix}_base");
        let texel = texture_node(&mut graph, &base, slot)?;
        let source = take_input_link(&mut graph, &principled, "Base Color");
        let mix_name = format!("{prefix}_base_mix");
        let mut mix = GraphNode::new("ShaderNodeMixRGB");
        mix.properties
            .insert("blend_type".to_owned(), json!("MULTIPLY"));
        mix.inputs.insert("Fac".to_owned(), json!(1.0));
        insert(&mut graph, &mix_name, mix)?;
        match source {
            Some(link) => connect_link(&mut graph, &link, &mix_name, "Color1")?,
            None => set_input(
                &mut graph,
                &mix_name,
                "Color1",
                input_value(
                    group,
                    &principled,
                    "Base Color",
                    json!([0.8, 0.8, 0.8, 1.0]),
                ),
            )?,
        }
        connect(&mut graph, &texel, "Color", &mix_name, "Color2")?;
        connect(
            &mut graph,
            &mix_name,
            "Color",
            principled.as_str(),
            "Base Color",
        )?;
        if material.alpha_mode != "opaque" {
            let source = take_input_link(&mut graph, &principled, "Alpha");
            let value = input_scalar(group, &principled, "Alpha", 1.0);
            multiply_input(
                &mut graph,
                source,
                value,
                (texel.as_str(), "Alpha"),
                &format!("{prefix}_alpha"),
                (principled.as_str(), "Alpha"),
            )?;
        }
    }
    if let Some(slot) = &textures.roughness {
        overlay_scalar(
            &mut graph,
            group,
            &principled,
            slot,
            "Roughness",
            0.5,
            &format!("{prefix}_roughness"),
        )?;
    }
    if let Some(slot) = &textures.metallic {
        overlay_scalar(
            &mut graph,
            group,
            &principled,
            slot,
            "Metallic",
            0.0,
            &format!("{prefix}_metallic"),
        )?;
    }
    if let Some(slot) = &textures.normal {
        let normal_linked = group
            .links
            .iter()
            .any(|link| link.to_node == principled && link.to_socket == "Normal");
        if normal_linked {
            losses.push(Loss {
                feature_id: feature_id::MATERIAL_TEXTURES.to_owned(),
                data_id: Some(material_id.as_str().to_owned()),
                reason: "the material's Principled Normal input is linked, and its normal texture perturbs that linked vector, which no Blender node reproduces; the normal texture is not written to .blend"
                    .to_owned(),
                suggestion: Some(
                    "leave the Normal input unlinked so the normal texture drives it".to_owned(),
                ),
            });
        } else {
            let color = texture_node(&mut graph, &format!("{prefix}_normal"), slot)?;
            let map_name = format!("{prefix}_normal_map");
            insert(&mut graph, &map_name, normal_map_node(slot, 1.0))?;
            connect(&mut graph, &color, "Color", &map_name, "Color")?;
            connect(
                &mut graph,
                &map_name,
                "Normal",
                principled.as_str(),
                "Normal",
            )?;
        }
    }
    Ok(Some(graph))
}

/// A name prefix that no node ID of `group` starts with: `base`, else `base_2`, `base_3`, and so on. The overlay's
/// generated IDs all extend it, so none can equal an authored ID (the suffix convention of `unique_shape_key_id`).
fn fresh_prefix(group: &NodeGroup, base: &str) -> String {
    let mut candidate = base.to_owned();
    let mut suffix = 2_u32;
    while group
        .nodes
        .keys()
        .any(|id| id.as_str().starts_with(candidate.as_str()))
    {
        candidate = format!("{base}_{suffix}");
        suffix += 1;
    }
    candidate
}

/// Multiplies a Principled scalar input by the Red channel of `slot`'s texture, as the evaluator does. `name` is the
/// overlay's own node-name stem.
fn overlay_scalar(
    graph: &mut NodeGroup,
    group: &NodeGroup,
    principled: &Id,
    slot: &SlotTexture,
    socket: &str,
    default: f64,
    name: &str,
) -> Result<()> {
    let texel = texture_node(graph, name, slot)?;
    let split = format!("{name}_split");
    insert(graph, &split, GraphNode::new("ShaderNodeSeparateColor"))?;
    connect(graph, &texel, "Color", &split, "Color")?;
    let source = take_input_link(graph, principled, socket);
    let value = input_scalar(group, principled, socket, default);
    multiply_input(
        graph,
        source,
        value,
        (split.as_str(), "Red"),
        name,
        (principled.as_str(), socket),
    )
}

/// The Principled BSDF that drives the active output's Surface, when the output is one.
fn surface_principled(group: &NodeGroup) -> Option<Id> {
    let outputs = group
        .nodes
        .iter()
        .filter(|(_, node)| {
            matches!(
                node.node_type.as_str(),
                "OutputMaterial" | "ShaderNodeOutputMaterial"
            )
        })
        .collect::<Vec<_>>();
    let output = outputs
        .iter()
        .find(|(_, node)| {
            node.properties
                .get("is_active_output")
                .and_then(Value::as_bool)
                == Some(true)
        })
        .or_else(|| outputs.first())?
        .0
        .clone();
    let link = group
        .links
        .iter()
        .find(|link| link.to_node == output && link.to_socket == "Surface")?;
    let node = group.nodes.get(&link.from_node)?;
    (node.node_type == "ShaderNodeBsdfPrincipled" && link.from_socket == "BSDF")
        .then(|| link.from_node.clone())
}

/// Removes and returns the link into `node`'s input `socket`, if one exists.
fn take_input_link(graph: &mut NodeGroup, node: &Id, socket: &str) -> Option<GraphLink> {
    let index = graph
        .links
        .iter()
        .position(|link| link.to_node == *node && link.to_socket == socket)?;
    Some(graph.links.remove(index))
}

fn connect_link(graph: &mut NodeGroup, link: &GraphLink, to: &str, to_socket: &str) -> Result<()> {
    connect(
        graph,
        link.from_node.as_str(),
        &link.from_socket,
        to,
        to_socket,
    )
}

/// An input's authored default, or `default` when the input has none.
fn input_value(group: &NodeGroup, node: &Id, socket: &str, default: Value) -> Value {
    group
        .nodes
        .get(node)
        .and_then(|node| node.inputs.get(socket))
        .cloned()
        .unwrap_or(default)
}

fn input_scalar(group: &NodeGroup, node: &Id, socket: &str, default: f64) -> f64 {
    input_value(group, node, socket, json!(default))
        .as_f64()
        .unwrap_or(default)
}

/// Writes a Math MULTIPLY into `to`: `source` (a link, else `value`) times the `texel` output.
fn multiply_input(
    graph: &mut NodeGroup,
    source: Option<GraphLink>,
    value: f64,
    texel: (&str, &str),
    name: &str,
    to: (&str, &str),
) -> Result<()> {
    let math = format!("{name}_factor");
    let mut node = GraphNode::new("ShaderNodeMath");
    node.properties
        .insert("operation".to_owned(), json!("MULTIPLY"));
    insert(graph, &math, node)?;
    match source {
        Some(link) => connect_link(graph, &link, &math, "Value")?,
        None => set_input(graph, &math, "Value", json!(value))?,
    }
    connect(graph, texel.0, texel.1, &math, "Value_001")?;
    connect(graph, &math, "Value", to.0, to.1)
}

/// Links the Red channel of `slot`'s texture into the Principled BSDF `to_socket`, multiplied by `factor`: a native
/// scalar field is its factor times the texture's Red channel, as Potter's evaluator reads it.
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
    multiply_channel(
        graph,
        &split,
        channel,
        name,
        factor,
        "principled",
        to_socket,
    )
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
            roughness: None,
            metallic: None,
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
