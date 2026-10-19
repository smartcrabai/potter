//! Glue between Potter materials and the glTF/VRM material model.
//!
//! A Potter material is either explicit PBR fields or a shader node graph (imported from
//! `.blend` or authored with node ops). [`resolve_surface`] turns either into one
//! [`ResolvedSurface`]; the glTF writer and the VRM writers (1.0 and 0.x) read only that
//! value. Conversion rules follow Blender 5.2's glTF exporter (`io_scene_gltf2` 5.2.40,
//! `blender/exp/material/*.py`). Where glTF can represent the graph but Blender drops or
//! mis-exports it, the Potter value is exported and the divergence is documented in
//! `docs/spec.md`. Where glTF cannot represent a graph part, a loss is reported with a
//! `material.shader_graph.*` feature id, gated by `--allow-lossy`.

use serde_json::{Value, json};

use crate::{
    error::{ErrorCode, PotError},
    graph::{GraphKind, GraphLink, GraphNode, NodeGroup},
    image::{ImageExtension, ImageInterpolation},
    model::{Id, Material, SceneDoc, TextureRef},
};

/// Source channel of a Potter image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TextureChannel {
    Red,
    Green,
    Blue,
    Alpha,
}

/// One glTF texel channel fed from one channel of one Potter image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ChannelSource {
    pub(crate) image: Id,
    pub(crate) channel: TextureChannel,
}

/// glTF sampler wrap mode. Blender: EXTEND/CLIP -> `CLAMP_TO_EDGE`, MIRROR -> `MIRRORED_REPEAT`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TextureWrap {
    Repeat,
    ClampToEdge,
    MirroredRepeat,
}

/// `KHR_texture_transform` values already converted from Blender's Mapping node.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TextureTransform {
    pub(crate) offset: [f64; 2],
    pub(crate) rotation: f64,
    pub(crate) scale: [f64; 2],
}

impl TextureTransform {
    /// Whether this transform leaves texture coordinates unchanged.
    pub(crate) fn is_identity(&self) -> bool {
        crate::float::equal_f64_array(&self.offset, &[0.0; 2])
            && crate::float::equal_f64(self.rotation, 0.0)
            && crate::float::equal_f64_array(&self.scale, &[1.0; 2])
    }
}

/// One glTF texture: four output channels (R, G, B, A) plus sampling state.
///
/// `channels[i] == None` means the glTF channel is the constant 1.0 (Blender fills unlinked
/// packed channels with white). The writer reuses an image without re-encoding only when all
/// four channels are `Some` and select the same image's R, G, B, A in order.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SampledTexture {
    pub(crate) channels: [Option<ChannelSource>; 4],
    /// UV layer name. `None` is the render layer, which exports as the texture's `texCoord`.
    pub(crate) uv_layer: Option<String>,
    pub(crate) interpolation: ImageInterpolation,
    /// Wrap modes for S and T.
    pub(crate) wrap: [TextureWrap; 2],
    pub(crate) transform: Option<TextureTransform>,
}

/// A glTF factor that multiplies an optional texture.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TexturedValue<T> {
    pub(crate) factor: T,
    pub(crate) texture: Option<SampledTexture>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TexturedNormal {
    pub(crate) texture: SampledTexture,
    /// `normalTexture.scale` (Blender `NormalMap` Strength).
    pub(crate) scale: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TexturedOcclusion {
    /// Occlusion lives in the R channel of `texture`.
    pub(crate) texture: SampledTexture,
    /// `occlusionTexture.strength`; 1.0 unless a Blender Mix-with-white strength was found.
    pub(crate) strength: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum AlphaMode {
    Opaque,
    /// `alphaCutoff` is the threshold; Blender's clip detection yields these.
    Mask {
        cutoff: f64,
    },
    Blend,
}

/// `KHR_materials_clearcoat`. Factors are glTF values; textures carry the channel named in
/// their doc comment, and the writer emits them only when present.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Clearcoat {
    /// Weight: texture R.
    pub(crate) factor: f64,
    pub(crate) texture: Option<SampledTexture>,
    /// Roughness: texture G.
    pub(crate) roughness_factor: f64,
    pub(crate) roughness_texture: Option<SampledTexture>,
    pub(crate) normal: Option<TexturedNormal>,
}

/// `KHR_materials_sheen`. Color: texture RGB. Roughness: texture A.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Sheen {
    pub(crate) color_factor: [f64; 3],
    pub(crate) color_texture: Option<SampledTexture>,
    pub(crate) roughness_factor: f64,
    pub(crate) roughness_texture: Option<SampledTexture>,
}

/// `KHR_materials_specular`. Level: texture A. Color: texture RGB.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Specular {
    pub(crate) factor: f64,
    pub(crate) texture: Option<SampledTexture>,
    pub(crate) color_factor: [f64; 3],
    pub(crate) color_texture: Option<SampledTexture>,
}

/// `KHR_materials_anisotropy` without textures (textured anisotropy reports a loss).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Anisotropy {
    pub(crate) strength: f64,
    /// Radians. Blender's rotation is in turns; 1 turn = 2π.
    pub(crate) rotation: f64,
}

/// The vertex color a base color multiplies in: the primitive's `COLOR_0`, read from the named color
/// layer. `None` is the render layer, Blender's active color.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct VertexColorRead {
    pub(crate) layer: Option<String>,
}

/// The glTF material for one Potter material, independent of the writer.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResolvedSurface {
    pub(crate) double_sided: bool,
    pub(crate) alpha_mode: AlphaMode,
    /// `KHR_materials_unlit`: the base color carries the emitted color.
    pub(crate) unlit: bool,
    /// `pbrMetallicRoughness.baseColorFactor` (RGBA, alpha factor included) and texture (RGBA).
    /// glTF multiplies the vertex color read by `vertex_color` into this value.
    pub(crate) base_color: TexturedValue<[f64; 4]>,
    /// The vertex color the base color multiplies (`COLOR_0`). `None` when the graph reads none.
    pub(crate) vertex_color: Option<VertexColorRead>,
    /// A vertex color the graph reads where glTF cannot multiply it (a `VERTEX_COLOR` loss). The
    /// layer is still written as `COLOR_0` so its data is kept; `vertex_color` takes precedence.
    pub(crate) unmultiplied_vertex_color: Option<VertexColorRead>,
    pub(crate) metallic: f64,
    pub(crate) roughness: f64,
    /// glTF metallicRoughnessTexture: G = roughness, B = metallic.
    pub(crate) metallic_roughness_texture: Option<SampledTexture>,
    pub(crate) occlusion: Option<TexturedOcclusion>,
    pub(crate) normal: Option<TexturedNormal>,
    /// Emitted color multiplied by emission strength, unnormalized. The writer applies
    /// `KHR_materials_emissive_strength` when a channel exceeds 1.
    pub(crate) emissive: TexturedValue<[f64; 3]>,
    /// `KHR_materials_ior` value; the writer emits it only with transmission or specular.
    pub(crate) ior: f64,
    /// `KHR_materials_transmission` factor with texture R.
    pub(crate) transmission: Option<TexturedValue<f64>>,
    pub(crate) clearcoat: Option<Clearcoat>,
    pub(crate) sheen: Option<Sheen>,
    pub(crate) specular: Option<Specular>,
    pub(crate) anisotropy: Option<Anisotropy>,
}

/// Feature ids for material conversion losses and errors (shared by resolver and writers).
pub(crate) mod feature_id {
    /// Error: the graph is structurally invalid (dangling link, missing image, cycle, no output).
    pub(crate) const INVALID: &str = "material.shader_graph.invalid";
    /// Loss: a node type, socket or channel path with no glTF representation.
    pub(crate) const UNSUPPORTED_NODE: &str = "material.shader_graph.unsupported_node";
    /// Loss: a Principled input or BSDF parameter with no glTF representation.
    pub(crate) const UNSUPPORTED_INPUT: &str = "material.shader_graph.unsupported_input";
    /// Loss: a surface shader combination glTF cannot express (two surfaces, non-Principled BSDF).
    pub(crate) const UNREPRESENTABLE_SURFACE: &str =
        "material.shader_graph.unrepresentable_surface";
    /// Loss: bump, object/world-space normal maps, or a texture wired straight into Normal.
    pub(crate) const NORMAL_MAP: &str = "material.shader_graph.normal_map";
    /// Loss: Mapping rotation about X/Y or a non-invertible TEXTURE mapping.
    pub(crate) const TRANSFORM: &str = "material.shader_graph.transform";
    /// Loss: a vertex color read where glTF cannot take it (anything but the base color's `Color`).
    pub(crate) const VERTEX_COLOR: &str = "material.shader_graph.vertex_color";
    /// Loss: displacement output.
    pub(crate) const DISPLACEMENT: &str = "material.shader_graph.displacement";
    /// Loss: volume output.
    pub(crate) const VOLUME: &str = "material.shader_graph.volume";
    /// Loss: sampler state glTF cannot express exactly (Cubic/Smart interpolation, CLIP extension).
    pub(crate) const SAMPLER: &str = "material.shader_graph.sampler";
    /// Loss: a material's texture fields that the output cannot hold. glTF: the graph is authoritative and ignores them.
    /// `.blend`: the authored graph does not drive its output from a Principled BSDF, or a linked Normal input takes a normal texture.
    pub(crate) const MATERIAL_TEXTURES: &str = "material.shader_graph.material_textures";
    /// Loss: a mesh UV layer the format cannot hold, or a texture set that samples layers that
    /// cannot share one `texCoord` (a packed texture, or a material shared by meshes that order
    /// their layers differently).
    pub(crate) const MESH_UV_LAYER: &str = "mesh.uv_layer";
    /// Loss: a stored UV layer reference names no layer on the mesh, so it falls back to the render layer.
    pub(crate) const UV_LAYER_MISSING: &str = "uv.layer_missing";
    /// Loss: a VRM 0.x `MToon` texture samples a non-zero UV layer; VRM 0.x has no `texCoord`, so it exports from `TEXCOORD_0`.
    pub(crate) const VRM_V0_MTOON_UV_LAYER: &str = "vrm.v0.mtoon_uv_layer";
}

/// Whether a material is assigned to any node's effective source-aware slot.
pub(crate) fn material_is_used(doc: &crate::model::SceneDoc, material_id: &Id) -> bool {
    doc.nodes.values().any(|node| {
        let data = node
            .data
            .as_ref()
            .and_then(|data_id| doc.data_blocks.get(data_id));
        (0..node.material_slot_count(data))
            .any(|slot| node.material_slot(data, slot) == Some(material_id))
    })
}

/// Resolves a Potter material to its glTF surface and the losses the conversion reports.
///
/// Materials without a custom graph (no node tree, or the generated default graph) resolve
/// from their explicit fields. Custom graphs resolve from the graph, which is authoritative
/// for every channel (the material's mirror fields and texture references are ignored and
/// reported as losses).
///
/// # Errors
///
/// Returns `SCENE_INVALID` / `INVALID_OPERATION` with `feature_id::INVALID` for a structurally
/// invalid graph or a missing image. Unsupported graph parts are losses, never errors.
pub(crate) fn resolve_surface(
    doc: &crate::model::SceneDoc,
    material_id: &Id,
) -> crate::error::Result<(ResolvedSurface, Vec<super::Loss>)> {
    let material = material_of(doc, material_id)?;
    match &material.node_tree {
        Some(graph_id) if !super::is_default_material_graph(doc, graph_id) => {
            graph_surface(doc, material_id, material, graph_id)
        }
        _ => fields_surface(doc, material_id),
    }
}

/// Resolves only the explicit fields of a material (ignores any custom graph). Used for
/// unused materials whose graph is invalid, so they never block export.
///
/// # Errors
///
/// Returns an error only for a missing texture image or an invalid alpha mode.
pub(crate) fn resolve_fields_surface(
    doc: &crate::model::SceneDoc,
    material_id: &Id,
) -> crate::error::Result<ResolvedSurface> {
    fields_surface(doc, material_id).map(|(surface, _)| surface)
}

/// Looks up a material, reporting a missing id as `TARGET_NOT_FOUND`.
fn material_of<'a>(doc: &'a SceneDoc, material_id: &Id) -> crate::error::Result<&'a Material> {
    doc.materials.get(material_id).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::TargetNotFound,
            format!("material `{material_id}` was not found"),
            json!({ "material_id": material_id.as_str() }),
        )
    })
}

/// Builds the `SCENE_INVALID` error for a structurally invalid material or graph.
fn invalid_error(material_id: &Id, node_id: Option<&Id>, message: impl Into<String>) -> PotError {
    PotError::with_details(
        ErrorCode::SceneInvalid,
        message,
        json!({
            "feature_id": feature_id::INVALID,
            "material_id": material_id.as_str(),
            "node_id": node_id.map(Id::as_str),
        }),
    )
}

/// Fails when a texture refers to an image that the document does not contain.
fn image_exists(doc: &SceneDoc, material_id: &Id, image_id: &Id) -> crate::error::Result<()> {
    if doc.images.contains_key(image_id) {
        return Ok(());
    }
    Err(PotError::with_details(
        ErrorCode::SceneInvalid,
        format!("material `{material_id}` references missing image `{image_id}`"),
        json!({
            "feature_id": feature_id::INVALID,
            "material_id": material_id.as_str(),
            "image_id": image_id.as_str(),
        }),
    ))
}

fn push_loss(
    losses: &mut Vec<super::Loss>,
    feature: &str,
    material_id: &Id,
    reason: String,
    suggestion: Option<&str>,
) {
    losses.push(super::Loss {
        feature_id: feature.to_owned(),
        data_id: Some(material_id.as_str().to_owned()),
        reason,
        suggestion: suggestion.map(str::to_owned),
    });
}

/// The layer a shader `ShaderNodeUVMap` node names: an empty name is the render layer (`None`).
fn uv_layer_name(name: Option<&str>) -> Option<String> {
    name.filter(|name| !name.is_empty()).map(str::to_owned)
}

/// Channels R, G, B, A each read the same channel of `image`.
pub(crate) fn identity_channels(image: &Id) -> [Option<ChannelSource>; 4] {
    [
        TextureChannel::Red,
        TextureChannel::Green,
        TextureChannel::Blue,
        TextureChannel::Alpha,
    ]
    .map(|channel| {
        Some(ChannelSource {
            image: image.clone(),
            channel,
        })
    })
}

/// A Potter texture reference as a glTF texture: the image's own R, G, B, A channels.
pub(crate) fn reference_sampled(reference: &TextureRef) -> SampledTexture {
    sampled_from_ref(reference, identity_channels(&reference.image))
}

/// One channel of a Potter texture reference, as grey RGB with opaque alpha (VRM `MToon` masks).
pub(crate) fn remapped_reference_sampled(
    reference: &TextureRef,
    channel: TextureChannel,
) -> SampledTexture {
    let source = || {
        Some(ChannelSource {
            image: reference.image.clone(),
            channel,
        })
    };
    sampled_from_ref(reference, [source(), source(), source(), None])
}

/// Sampling state of a Potter texture reference (no transform). Its extension sets the wrap of both axes.
fn sampled_from_ref(
    reference: &TextureRef,
    channels: [Option<ChannelSource>; 4],
) -> SampledTexture {
    let wrap = texture_wrap_of(reference.extension);
    SampledTexture {
        channels,
        uv_layer: reference.uv_layer.clone(),
        interpolation: reference.interpolation,
        wrap: [wrap; 2],
        transform: None,
    }
}

/// The glTF sampler wrap of a Potter extension, as Blender's exporter writes it (`exp/sampler.py`): EXTEND and CLIP are
/// `CLAMP_TO_EDGE` (CLIP is the closest, since glTF has no clip), MIRROR is `MIRRORED_REPEAT`, and REPEAT is the default.
pub(crate) fn texture_wrap_of(extension: ImageExtension) -> TextureWrap {
    match extension {
        ImageExtension::Repeat => TextureWrap::Repeat,
        ImageExtension::Extend | ImageExtension::Clip => TextureWrap::ClampToEdge,
        ImageExtension::Mirror => TextureWrap::MirroredRepeat,
    }
}

/// The metallicRoughness texture of a material's explicit fields: G from the roughness texture's
/// Red channel, B from the metallic texture's Red channel. A native scalar field is read through
/// its Red channel (`material_graph::GraphTextures::from_field_refs`), so each field packs its Red
/// into its glTF channel. Each channel reads its own image; the texture's sampling state comes
/// from the roughness reference when present.
fn metallic_roughness_from_refs(
    material_id: &Id,
    roughness: Option<&TextureRef>,
    metallic: Option<&TextureRef>,
    losses: &mut Vec<super::Loss>,
) -> Option<SampledTexture> {
    let base = roughness.or(metallic)?;
    if let (Some(rough), Some(metal)) = (roughness, metallic) {
        if rough.uv_layer != metal.uv_layer {
            push_loss(
                losses,
                feature_id::MESH_UV_LAYER,
                material_id,
                "roughness and metallic textures sample different UV layers; the packed \
                 metallicRoughness texture uses the roughness texture's layer"
                    .to_owned(),
                None,
            );
        }
        if rough.interpolation != metal.interpolation {
            push_loss(
                losses,
                feature_id::SAMPLER,
                material_id,
                "roughness and metallic textures differ in interpolation; the packed \
                 metallicRoughness texture uses the roughness texture's interpolation"
                    .to_owned(),
                None,
            );
        }
    }
    let roughness_source = roughness.map(|rough| ChannelSource {
        image: rough.image.clone(),
        channel: TextureChannel::Red,
    });
    let metallic_source = metallic.map(|metal| ChannelSource {
        image: metal.image.clone(),
        channel: TextureChannel::Red,
    });
    Some(sampled_from_ref(
        base,
        [None, roughness_source, metallic_source, None],
    ))
}

/// Resolves the explicit fields of a material. Used for materials without a custom graph.
fn fields_surface(
    doc: &SceneDoc,
    material_id: &Id,
) -> crate::error::Result<(ResolvedSurface, Vec<super::Loss>)> {
    let material = material_of(doc, material_id)?;
    for reference in [
        material.base_color_texture.as_ref(),
        material.roughness_texture.as_ref(),
        material.metallic_texture.as_ref(),
        material.normal_texture.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        image_exists(doc, material_id, &reference.image)?;
    }
    let alpha_mode = match material.alpha_mode.as_str() {
        "opaque" => AlphaMode::Opaque,
        "clip" => AlphaMode::Mask {
            cutoff: material.alpha_threshold,
        },
        "blend" => AlphaMode::Blend,
        other => {
            return Err(invalid_error(
                material_id,
                None,
                format!("material `{material_id}` has unknown alpha mode `{other}`"),
            ));
        }
    };
    let mut losses = Vec::new();
    for (property, reference) in [
        ("base colour", material.base_color_texture.as_ref()),
        ("roughness", material.roughness_texture.as_ref()),
        ("metallic", material.metallic_texture.as_ref()),
        ("normal", material.normal_texture.as_ref()),
    ] {
        if reference.is_some_and(|reference| reference.extension == ImageExtension::Clip) {
            push_loss(
                &mut losses,
                feature_id::SAMPLER,
                material_id,
                format!(
                    "texture field `{property}` of material `{material_id}` uses `clip` extension; it is sampled as clamp to edge"
                ),
                None,
            );
        }
    }
    let base_color = TexturedValue {
        factor: material.base_color,
        texture: material
            .base_color_texture
            .as_ref()
            .map(|reference| sampled_from_ref(reference, identity_channels(&reference.image))),
    };
    let metallic_roughness_texture = metallic_roughness_from_refs(
        material_id,
        material.roughness_texture.as_ref(),
        material.metallic_texture.as_ref(),
        &mut losses,
    );
    let normal = material
        .normal_texture
        .as_ref()
        .map(|reference| TexturedNormal {
            texture: sampled_from_ref(reference, identity_channels(&reference.image)),
            scale: 1.0,
        });
    let surface = ResolvedSurface {
        double_sided: material.double_sided,
        alpha_mode,
        unlit: false,
        vertex_color: None,
        unmultiplied_vertex_color: None,
        base_color,
        metallic: material.metallic,
        roughness: material.roughness,
        metallic_roughness_texture,
        occlusion: None,
        normal,
        emissive: TexturedValue {
            factor: material
                .emission_color
                .map(|c| c * material.emission_strength),
            texture: None,
        },
        ior: material.ior,
        transmission: (material.transmission != 0.0).then_some(TexturedValue {
            factor: material.transmission,
            texture: None,
        }),
        clearcoat: None,
        sheen: None,
        specular: None,
        anisotropy: None,
    };
    Ok((surface, losses))
}

// ---------------------------------------------------------------------------
// Graph path: reduction of a shader node graph to a `ResolvedSurface`.
// ---------------------------------------------------------------------------

/// Graph recursion limit; deeper graphs are treated as cyclic and rejected.
const MAX_GRAPH_DEPTH: usize = 128;

/// Texture sampling state of one `ShaderNodeTexImage`. `wrap` is the wrap of each UV axis (S, T).
#[derive(Clone, Debug, PartialEq)]
struct TexBase {
    image: Id,
    uv_layer: Option<String>,
    interpolation: ImageInterpolation,
    wrap: [TextureWrap; 2],
    transform: Option<TextureTransform>,
}

/// The UV an image texture reads: its layer, its transform, and the wrap of each axis that Blender's per-axis wrap
/// pattern sets (`None` where the pattern passes the axis through, or where there is no pattern).
#[derive(Clone, Debug, Default, PartialEq)]
struct UvSource {
    uv_layer: Option<String>,
    transform: Option<TextureTransform>,
    wrap: [Option<TextureWrap>; 2],
}

/// One texture read. `channel == None` is the Color output: the consuming slot decides which
/// channel(s) it reads. `Some(c)` is a single channel (Alpha output or `SeparateColor`).
#[derive(Clone, Debug, PartialEq)]
struct Sample {
    base: TexBase,
    channel: Option<TextureChannel>,
}

/// A color-valued socket: `factor` times the sampled texture (or the factor alone).
#[derive(Clone, Debug, PartialEq)]
struct ColorIn {
    factor: [f64; 3],
    tex: Option<Sample>,
    /// Vertex color multiplied in (`COLOR_0`). Only a base color takes it; other sockets drop it.
    vertex: Option<VertexColorRead>,
}

/// A scalar-valued socket: `factor` times the sampled channel (or the factor alone).
#[derive(Clone, Debug, PartialEq)]
struct ScalarIn {
    factor: f64,
    tex: Option<Sample>,
}

/// Alpha expression of a socket.
#[derive(Clone, Debug, PartialEq)]
enum Alpha {
    Const(f64),
    /// Texture alpha times a factor (BLEND).
    Tex {
        sample: Sample,
        factor: f64,
    },
    /// Texture alpha thresholded: alpha >= `cutoff` is kept (MASK).
    Clip {
        sample: Sample,
        cutoff: f64,
    },
}

/// Reduced Principled BSDF (or a non-Principled approximation).
#[derive(Clone, Debug, PartialEq)]
struct Core {
    base: ColorIn,
    alpha: Alpha,
    metallic: ScalarIn,
    roughness: ScalarIn,
    normal: Option<TexturedNormal>,
    ior: f64,
    transmission: Option<TexturedValue<f64>>,
    clearcoat: Option<Clearcoat>,
    sheen: Option<Sheen>,
    specular: Option<Specular>,
    anisotropy: Option<Anisotropy>,
}

/// A reduced surface shader. Empty parts contribute nothing.
#[derive(Clone, Debug, Default, PartialEq)]
struct Shade {
    /// PBR core from a Principled BSDF (or a non-Principled approximation).
    core: Option<Core>,
    /// Emitted color (color times strength), summed over additive contributions.
    emission: Option<TexturedValue<[f64; 3]>>,
    /// `Background` color (strength folded in); an unlit surface.
    unlit: Option<ColorIn>,
    /// Alpha from `Mix(Fac, Transparent, X)`.
    mix_alpha: Option<Alpha>,
    /// This shade is a `Transparent BSDF`.
    transparent: bool,
}

impl Shade {
    fn is_empty(&self) -> bool {
        self.core.is_none() && self.emission.is_none() && self.unlit.is_none() && !self.transparent
    }
}

fn sampled_texture(base: &TexBase, channels: [Option<ChannelSource>; 4]) -> SampledTexture {
    SampledTexture {
        channels,
        uv_layer: base.uv_layer.clone(),
        interpolation: base.interpolation,
        wrap: base.wrap,
        transform: base.transform.clone(),
    }
}

fn channel_of(image: &Id, channel: TextureChannel) -> ChannelSource {
    ChannelSource {
        image: image.clone(),
        channel,
    }
}

/// RGB slots from a color sample; a single channel is replicated; A is unfilled.
fn rgb_texture(sample: &Sample) -> SampledTexture {
    let channels = match sample.channel {
        None => [
            TextureChannel::Red,
            TextureChannel::Green,
            TextureChannel::Blue,
        ]
        .map(|channel| Some(channel_of(&sample.base.image, channel))),
        Some(channel) => [channel; 3].map(|channel| Some(channel_of(&sample.base.image, channel))),
    };
    sampled_texture(
        &sample.base,
        [
            channels[0].clone(),
            channels[1].clone(),
            channels[2].clone(),
            None,
        ],
    )
}

/// One glTF slot fed from `sample`; the other slots are unfilled.
fn slot_texture(sample: &Sample, slot: usize, default: TextureChannel) -> SampledTexture {
    let mut channels: [Option<ChannelSource>; 4] = [None, None, None, None];
    channels[slot] = Some(channel_of(
        &sample.base.image,
        sample.channel.unwrap_or(default),
    ));
    sampled_texture(&sample.base, channels)
}

/// Identity RGBA of a normal map's image, sampled with the normal's sampling state.
fn normal_texture(sample: &Sample) -> SampledTexture {
    sampled_texture(&sample.base, identity_channels(&sample.base.image))
}

/// Describes why two texture reads cannot share one glTF texture.
fn sampling_mismatch(first: &TexBase, second: &TexBase) -> Option<(&'static str, &'static str)> {
    if first.uv_layer != second.uv_layer {
        Some((feature_id::MESH_UV_LAYER, "UV layer"))
    } else if first.interpolation != second.interpolation || first.wrap != second.wrap {
        Some((feature_id::SAMPLER, "sampler state"))
    } else if first.transform != second.transform {
        Some((feature_id::TRANSFORM, "texture transform"))
    } else {
        None
    }
}

fn value_array(value: &Value) -> Option<Vec<f64>> {
    value.as_array()?.iter().map(Value::as_f64).collect()
}

/// Three components from an array, or from Blender's `Euler((x, y, z), 'XYZ')` string.
fn value_vec3(value: &Value) -> Option<[f64; 3]> {
    let parts = if let Some(text) = value.as_str() {
        let inner = text.strip_prefix("Euler((")?.split(')').next()?;
        inner
            .split(',')
            .map(|part| part.trim().parse::<f64>().ok())
            .collect::<Option<Vec<f64>>>()?
    } else {
        value_array(value)?
    };
    Some([*parts.first()?, *parts.get(1)?, *parts.get(2)?])
}

/// RGB from an array of at least three numbers (alpha is ignored).
fn value_rgb(value: &Value) -> Option<[f64; 3]> {
    value_vec3(value)
}

fn prop_str<'n>(node: &'n GraphNode, key: &str) -> Option<&'n str> {
    node.properties.get(key).and_then(Value::as_str)
}

fn uv_map_layer(node: &GraphNode) -> Option<String> {
    uv_layer_name(prop_str(node, "uv_map"))
}

/// glTF `KHR_texture_transform` from a Blender point mapping (verified against Blender 5.2:
/// P02 gives offset (-1.4, -1.798) for location (0.1, 0.2), rotation 30 degrees, scale (2, 3)).
fn point_transform(location: [f64; 2], scale: [f64; 2], turn: f64) -> TextureTransform {
    let (sin, cos) = turn.sin_cos();
    TextureTransform {
        offset: [
            location[0] - scale[1] * sin,
            1.0 - location[1] - scale[1] * cos,
        ],
        rotation: turn,
        scale,
    }
}

/// Longest chain of Reroute nodes followed upstream; longer chains are reported as unlinked.
const MAX_REROUTE_DEPTH: usize = 64;

/// Reduces the graph of one material. Losses accumulate; structural problems are errors.
struct GraphReducer<'a> {
    doc: &'a SceneDoc,
    material_id: &'a Id,
    group: &'a NodeGroup,
    losses: Vec<super::Loss>,
    /// The first vertex color a non-base socket dropped (see `plain_color_input`).
    dropped_vertex: Option<VertexColorRead>,
}

impl<'a> GraphReducer<'a> {
    fn node(&self, node_id: &Id) -> crate::error::Result<&'a GraphNode> {
        let group: &'a NodeGroup = self.group;
        group
            .nodes
            .get(node_id)
            .ok_or_else(|| self.invalid(node_id, format!("shader graph has no node `{node_id}`")))
    }

    /// The link into `node_id.socket`, if any.
    /// The link feeding `node_id.socket`, with Reroute nodes stepped over: Blender's exporter
    /// follows a Reroute to its input (`search_node_tree.py`). An unlinked Reroute reads as unlinked.
    fn link_into(&self, node_id: &Id, socket: &str) -> Option<&'a GraphLink> {
        let group: &'a NodeGroup = self.group;
        let mut link = group
            .links
            .iter()
            .find(|link| link.to_node == *node_id && link.to_socket == socket)?;
        for _ in 0..MAX_REROUTE_DEPTH {
            let is_reroute = group
                .nodes
                .get(&link.from_node)
                .is_some_and(|node| node.node_type == "NodeReroute");
            if !is_reroute {
                break;
            }
            link = group
                .links
                .iter()
                .find(|feed| feed.to_node == link.from_node)?;
        }
        Some(link)
    }

    fn invalid(&self, node_id: &Id, message: String) -> PotError {
        invalid_error(self.material_id, Some(node_id), message)
    }

    fn loss(&mut self, feature: &str, reason: String, suggestion: Option<&str>) {
        push_loss(
            &mut self.losses,
            feature,
            self.material_id,
            reason,
            suggestion,
        );
    }

    /// A node the reduction does not model: loss, and the caller uses the socket default.
    fn unsupported(&mut self, node_id: &Id, node_type: &str, feeding: &str) {
        self.loss(
            feature_id::UNSUPPORTED_NODE,
            format!("node `{node_id}` ({node_type}) feeding {feeding} is not supported; the socket default is used"),
            None,
        );
    }

    fn deeper(&self, node_id: &Id, depth: usize) -> crate::error::Result<usize> {
        if depth >= MAX_GRAPH_DEPTH {
            return Err(self.invalid(
                node_id,
                format!(
                    "shader graph is deeper than {MAX_GRAPH_DEPTH} nodes at `{node_id}` (cycle?)"
                ),
            ));
        }
        Ok(depth + 1)
    }

    /// A vector socket's constant value; a linked socket is a loss and yields `default`.
    fn constant_vec(
        &mut self,
        node_id: &Id,
        node: &GraphNode,
        socket: &str,
        default: [f64; 3],
    ) -> [f64; 3] {
        if self.link_into(node_id, socket).is_some() {
            self.loss(
                feature_id::UNSUPPORTED_INPUT,
                format!(
                    "linked `{socket}` on `{node_id}` is not supported; the socket default is used"
                ),
                None,
            );
            return default;
        }
        node.inputs
            .get(socket)
            .and_then(value_vec3)
            .unwrap_or(default)
    }

    /// UV layer, transform and per-axis wrap of an image texture's `Vector` input.
    fn tex_vector(&mut self, tex_id: &Id, depth: usize) -> crate::error::Result<UvSource> {
        let Some(link) = self.link_into(tex_id, "Vector") else {
            return Ok(UvSource::default());
        };
        self.uv_source(tex_id, link, depth)
    }

    /// The UV source a link feeds: a UV Map, a Texture Coordinate, a Mapping, or Blender's per-axis wrap pattern
    /// (`exp/sampler.py` `detect_manual_uv_wrapping`), which wraps the source's coordinates on each axis.
    fn uv_source(
        &mut self,
        tex_id: &Id,
        link: &GraphLink,
        depth: usize,
    ) -> crate::error::Result<UvSource> {
        let source = self.node(&link.from_node)?;
        match (source.node_type.as_str(), link.from_socket.as_str()) {
            ("ShaderNodeUVMap", "UV") => Ok(UvSource {
                uv_layer: uv_map_layer(source),
                ..UvSource::default()
            }),
            ("ShaderNodeTexCoord", "UV") => Ok(UvSource::default()),
            ("ShaderNodeMapping", "Vector") => {
                let (uv_layer, transform) = self.mapping(&link.from_node, depth)?;
                Ok(UvSource {
                    uv_layer,
                    transform,
                    ..UvSource::default()
                })
            }
            ("ShaderNodeCombineXYZ", "Vector") => {
                if let Some((feed, wrap)) = self.uv_wrap_pattern(&link.from_node) {
                    let depth = self.deeper(&link.from_node, depth)?;
                    let mut uv = self.uv_source(tex_id, feed, depth)?;
                    uv.wrap = wrap;
                    Ok(uv)
                } else {
                    self.unsupported(
                        &link.from_node,
                        &source.node_type,
                        &format!("the Vector of `{tex_id}`"),
                    );
                    Ok(UvSource::default())
                }
            }
            _ => {
                self.unsupported(
                    &link.from_node,
                    &source.node_type,
                    &format!("the Vector of `{tex_id}`"),
                );
                Ok(UvSource::default())
            }
        }
    }

    /// Blender's per-axis UV wrap: a Combine XYZ whose X and Y are each the matching axis of one Separate XYZ, passed
    /// through or wrapped by a Math node. A `WRAP` with max 0 and min 1 is REPEAT, and a `PINGPONG` of scale 1 is
    /// `MIRRORED_REPEAT`, as the exporter recognises them. Returns the link that feeds the Separate XYZ, and each axis's
    /// wrap, with `None` for an axis passed through. Any other shape is not the pattern.
    fn uv_wrap_pattern(
        &self,
        combine_id: &Id,
    ) -> Option<(&'a GraphLink, [Option<TextureWrap>; 2])> {
        let mut separate: Option<Id> = None;
        let mut axes = [None; 2];
        for (index, axis) in ["X", "Y"].into_iter().enumerate() {
            let link = self.link_into(combine_id, axis)?;
            let source = self.node(&link.from_node).ok()?;
            let (feed_axis, wrap, sep_id) = match source.node_type.as_str() {
                "ShaderNodeSeparateXYZ" => {
                    (link.from_socket.as_str(), None, link.from_node.clone())
                }
                "ShaderNodeMath" => {
                    let constant = |socket: &str| -> Option<f64> {
                        if self.link_into(&link.from_node, socket).is_some() {
                            return None;
                        }
                        source.inputs.get(socket).and_then(Value::as_f64)
                    };
                    let wrap = match prop_str(source, "operation")? {
                        "PINGPONG" if crate::float::equal_f64(constant("Value_001")?, 1.0) => {
                            TextureWrap::MirroredRepeat
                        }
                        "WRAP"
                            if crate::float::equal_f64(constant("Value_001")?, 0.0)
                                && crate::float::equal_f64(constant("Value_002")?, 1.0) =>
                        {
                            TextureWrap::Repeat
                        }
                        _ => return None,
                    };
                    let feed = self.link_into(&link.from_node, "Value")?;
                    if self.node(&feed.from_node).ok()?.node_type != "ShaderNodeSeparateXYZ" {
                        return None;
                    }
                    (
                        feed.from_socket.as_str(),
                        Some(wrap),
                        feed.from_node.clone(),
                    )
                }
                _ => return None,
            };
            // X must come from X and Y from Y, and both from one Separate XYZ.
            if feed_axis != axis {
                return None;
            }
            match &separate {
                None => separate = Some(sep_id),
                Some(first) if *first == sep_id => {}
                Some(_) => return None,
            }
            axes[index] = wrap;
        }
        let separate = separate?;
        let feed = self.link_into(&separate, "Vector")?;
        Some((feed, axes))
    }

    /// `ShaderNodeMapping` as `KHR_texture_transform`, plus the UV layer of its input.
    fn mapping(
        &mut self,
        map_id: &Id,
        depth: usize,
    ) -> crate::error::Result<(Option<String>, Option<TextureTransform>)> {
        self.deeper(map_id, depth)?;
        let mut uv_layer = None;
        if let Some(link) = self.link_into(map_id, "Vector") {
            let source = self.node(&link.from_node)?;
            match (source.node_type.as_str(), link.from_socket.as_str()) {
                ("ShaderNodeUVMap", "UV") => uv_layer = uv_map_layer(source),
                ("ShaderNodeTexCoord", "UV") => {}
                ("ShaderNodeMapping", _) => self.loss(
                    feature_id::UNSUPPORTED_NODE,
                    format!("Mapping `{map_id}` is fed by another Mapping; only the outer Mapping is applied"),
                    None,
                ),
                _ => self.unsupported(
                    &link.from_node,
                    &source.node_type,
                    &format!("the Vector of Mapping `{map_id}`"),
                ),
            }
        }
        let node = self.node(map_id)?;
        let vector_type = prop_str(node, "vector_type").unwrap_or("POINT");
        let location = self.constant_vec(map_id, node, "Location", [0.0; 3]);
        let scale = self.constant_vec(map_id, node, "Scale", [1.0; 3]);
        let mut rotation = self.constant_vec(map_id, node, "Rotation", [0.0; 3]);
        if rotation[0] != 0.0 || rotation[1] != 0.0 {
            self.loss(
                feature_id::TRANSFORM,
                format!("Mapping `{map_id}` rotates about X or Y, which glTF cannot express; the whole rotation is dropped"),
                None,
            );
            rotation = [0.0; 3];
        }
        let turn = rotation[2];
        let (sx, sy) = (scale[0], scale[1]);
        let transform = match vector_type {
            "POINT" => Some(point_transform([location[0], location[1]], [sx, sy], turn)),
            "VECTOR" => Some(point_transform([0.0, 0.0], [sx, sy], turn)),
            "TEXTURE" => {
                if crate::float::equal_f64(sx, 0.0)
                    || crate::float::equal_f64(sy, 0.0)
                    || (!crate::float::equal_f64(turn, 0.0) && !crate::float::equal_f64(sx, sy))
                {
                    self.loss(
                        feature_id::TRANSFORM,
                        format!("Mapping `{map_id}` (TEXTURE) cannot be inverted for glTF; the transform is dropped"),
                        None,
                    );
                    None
                } else {
                    // Inverse TRS (Blender inverted_trs_mapping_node): rotate (-loc) by -turn, then divide by scale.
                    let (sin, cos) = turn.sin_cos();
                    let (vx, vy) = (-location[0], -location[1]);
                    let offset = [(cos * vx + sin * vy) / sx, (-sin * vx + cos * vy) / sy];
                    Some(point_transform(offset, [1.0 / sx, 1.0 / sy], -turn))
                }
            }
            other => {
                self.loss(
                    feature_id::UNSUPPORTED_NODE,
                    format!(
                        "Mapping `{map_id}` has vector type `{other}`; its transform is dropped"
                    ),
                    None,
                );
                None
            }
        };
        Ok((uv_layer, transform.filter(|t| !t.is_identity())))
    }

    /// An image texture read, with its channel selection.
    fn tex_sample(
        &mut self,
        tex_id: &Id,
        channel: Option<TextureChannel>,
        depth: usize,
    ) -> crate::error::Result<Sample> {
        let node = self.node(tex_id)?;
        let Some(image_name) = prop_str(node, "image").filter(|name| !name.is_empty()) else {
            return Err(self.invalid(tex_id, format!("image texture `{tex_id}` has no image")));
        };
        let image = Id::new(image_name).map_err(|_| {
            self.invalid(
                tex_id,
                format!("image texture `{tex_id}` has an invalid image id"),
            )
        })?;
        if !self.doc.images.contains_key(&image) {
            return Err(self.invalid(
                tex_id,
                format!("image texture `{tex_id}` references missing image `{image}`"),
            ));
        }
        let interpolation = match prop_str(node, "interpolation") {
            None | Some("Linear") => ImageInterpolation::Linear,
            Some("Closest") => ImageInterpolation::Closest,
            Some(other) => {
                self.loss(
                    feature_id::SAMPLER,
                    format!("image texture `{tex_id}` uses `{other}` interpolation; it is sampled as Linear"),
                    None,
                );
                ImageInterpolation::Linear
            }
        };
        let wrap = match prop_str(node, "extension") {
            None | Some("REPEAT") => TextureWrap::Repeat,
            Some("EXTEND") => TextureWrap::ClampToEdge,
            Some("MIRROR") => TextureWrap::MirroredRepeat,
            Some(other) => {
                self.loss(
                    feature_id::SAMPLER,
                    format!("image texture `{tex_id}` uses `{other}` extension; it is sampled as clamp to edge"),
                    None,
                );
                TextureWrap::ClampToEdge
            }
        };
        match prop_str(node, "projection") {
            None | Some("FLAT") => {}
            Some(other) => self.loss(
                feature_id::UNSUPPORTED_NODE,
                format!(
                    "image texture `{tex_id}` uses `{other}` projection; it is sampled as flat"
                ),
                None,
            ),
        }
        let uv = self.tex_vector(tex_id, depth)?;
        Ok(Sample {
            base: TexBase {
                image,
                uv_layer: uv.uv_layer,
                interpolation,
                // An axis the wrap pattern passes through keeps the node's extension, as Blender's exporter reads it.
                wrap: uv.wrap.map(|axis| axis.unwrap_or(wrap)),
                transform: uv.transform,
            },
            channel,
        })
    }
}

/// Clip predicate `x >= cutoff` (`ge`) or `x < cutoff` (`!ge`) on a texture channel.
struct Predicate {
    sample: Sample,
    cutoff: f64,
    ge: bool,
}

impl ColorIn {
    fn constant(factor: [f64; 3]) -> Self {
        Self {
            factor,
            tex: None,
            vertex: None,
        }
    }
}

impl ScalarIn {
    fn constant(factor: f64) -> Self {
        Self { factor, tex: None }
    }
}

/// Whether a node reads vertex colors on `output`: a Vertex Color node, or an Attribute node on the
/// geometry whose `Color` or `Alpha` output reads a named attribute as a color (Blender's glTF
/// exporter takes both as the color layer).
fn is_vertex_color(node: &GraphNode, output: &str) -> bool {
    match node.node_type.as_str() {
        "ShaderNodeVertexColor" => true,
        "ShaderNodeAttribute" => {
            prop_str(node, "attribute_type") == Some("GEOMETRY")
                && matches!(output, "Color" | "Alpha")
        }
        _ => false,
    }
}

/// The vertex color a `Color` output reads, when glTF can take it. Only an unnamed Vertex Color
/// selects the active/render layer; an empty geometry Attribute remains a missing named read.
fn vertex_color_read(node: &GraphNode, output: &str) -> Option<VertexColorRead> {
    if output != "Color" || !is_vertex_color(node, output) {
        return None;
    }
    let layer = if node.node_type == "ShaderNodeVertexColor" {
        prop_str(node, "layer_name")
            .filter(|name| !name.is_empty())
            .map(str::to_owned)
    } else {
        Some(
            prop_str(node, "attribute_name")
                .or_else(|| prop_str(node, "name"))
                .unwrap_or("")
                .to_owned(),
        )
    };
    Some(VertexColorRead { layer })
}

/// The product of two color inputs when glTF can hold it: factors multiply, with at most one
/// texture and at most one vertex color between them.
fn multiply(a: &ColorIn, b: &ColorIn) -> Option<ColorIn> {
    let tex = match (&a.tex, &b.tex) {
        (None, tex) | (tex, None) => tex.clone(),
        _ => return None,
    };
    let vertex = match (&a.vertex, &b.vertex) {
        (None, vertex) | (vertex, None) => vertex.clone(),
        _ => return None,
    };
    Some(ColorIn {
        factor: [
            a.factor[0] * b.factor[0],
            a.factor[1] * b.factor[1],
            a.factor[2] * b.factor[2],
        ],
        tex,
        vertex,
    })
}

/// Maps a `SeparateColor` output name to its channel index and channel.
fn separate_channel(output: &str) -> Option<(usize, TextureChannel)> {
    match output {
        "Red" => Some((0, TextureChannel::Red)),
        "Green" => Some((1, TextureChannel::Green)),
        "Blue" => Some((2, TextureChannel::Blue)),
        _ => None,
    }
}

impl GraphReducer<'_> {
    /// Constant or linked input in a color socket. `default` is the value used when the
    /// source is unsupported.
    fn color_input(
        &mut self,
        node_id: &Id,
        socket: &str,
        default: [f64; 3],
        depth: usize,
    ) -> crate::error::Result<ColorIn> {
        let node = self.node(node_id)?;
        let fallback = node
            .inputs
            .get(socket)
            .and_then(value_rgb)
            .unwrap_or(default);
        match self.link_into(node_id, socket) {
            Some(link) => self.color_from(&link.from_node, &link.from_socket, fallback, depth),
            None => Ok(ColorIn::constant(fallback)),
        }
    }

    /// A color socket whose value a constant or texture carries. A vertex color read in it is a
    /// loss: only the base color multiplies `COLOR_0`. The read's layer is still kept as the
    /// material's `COLOR_0` (`unmultiplied_vertex_color`), so the color data is exported.
    fn plain_color_input(
        &mut self,
        node_id: &Id,
        socket: &str,
        default: [f64; 3],
        depth: usize,
    ) -> crate::error::Result<ColorIn> {
        let mut color = self.color_input(node_id, socket, default, depth)?;
        if let Some(read) = color.vertex.take() {
            self.loss(
                feature_id::VERTEX_COLOR,
                format!("node `{node_id}` reads vertex color into socket `{socket}`; glTF multiplies vertex color only into the base color, so this read is not expressed; its layer is still written as COLOR_0"),
                None,
            );
            self.dropped_vertex = self.dropped_vertex.take().or(Some(read));
        }
        Ok(color)
    }

    fn color_from(
        &mut self,
        source: &Id,
        output: &str,
        fallback: [f64; 3],
        depth: usize,
    ) -> crate::error::Result<ColorIn> {
        let depth = self.deeper(source, depth)?;
        let node = self.node(source)?;
        if let Some(vertex_color) = vertex_color_read(node, output) {
            return Ok(ColorIn {
                factor: [1.0; 3],
                tex: None,
                vertex: Some(vertex_color),
            });
        }
        if is_vertex_color(node, output) {
            self.vertex_color_loss(source, &node.node_type);
            return Ok(ColorIn::constant([1.0; 3]));
        }
        match (node.node_type.as_str(), output) {
            ("ShaderNodeTexImage", "Color") => Ok(ColorIn {
                factor: [1.0; 3],
                tex: Some(self.tex_sample(source, None, depth)?),
                vertex: None,
            }),
            ("ShaderNodeTexImage", "Alpha") => Ok(ColorIn {
                factor: [1.0; 3],
                tex: Some(self.tex_sample(source, Some(TextureChannel::Alpha), depth)?),
                vertex: None,
            }),
            ("ShaderNodeSeparateColor", _) => {
                let scalar = self.separated(source, output, fallback[0], depth)?;
                Ok(ColorIn {
                    factor: [scalar.factor; 3],
                    tex: scalar.tex,
                    vertex: None,
                })
            }
            ("ShaderNodeMix" | "ShaderNodeMixRGB", _) => self.mix_color(source, depth),
            _ => {
                self.unsupported(source, &node.node_type, "a color input");
                Ok(ColorIn::constant(fallback))
            }
        }
    }

    fn vertex_color_loss(&mut self, node_id: &Id, node_type: &str) {
        self.loss(
            feature_id::VERTEX_COLOR,
            format!("node `{node_id}` ({node_type}) reads vertex color where glTF takes none (only the base color multiplies COLOR_0); it is treated as white"),
            None,
        );
    }

    /// One channel of a `SeparateColor` input (RGB mode only).
    fn separated(
        &mut self,
        source: &Id,
        output: &str,
        fallback: f64,
        depth: usize,
    ) -> crate::error::Result<ScalarIn> {
        let depth = self.deeper(source, depth)?;
        let node = self.node(source)?;
        let Some((index, channel)) = separate_channel(output) else {
            self.unsupported(source, &node.node_type, "a scalar input");
            return Ok(ScalarIn::constant(fallback));
        };
        if let Some(mode) = prop_str(node, "mode").filter(|mode| *mode != "RGB") {
            self.loss(
                feature_id::UNSUPPORTED_NODE,
                format!("SeparateColor `{source}` uses mode `{mode}`; only RGB is supported"),
                None,
            );
            return Ok(ScalarIn::constant(fallback));
        }
        let color = self.plain_color_input(source, "Color", [fallback; 3], depth)?;
        Ok(ScalarIn {
            factor: color.factor[index],
            tex: color.tex.map(|sample| Sample {
                channel: sample.channel.or(Some(channel)),
                ..sample
            }),
        })
    }

    /// `Mix` (RGBA) and legacy `MixRGB` in color context. Blender MULTIPLY with a constant
    /// factor f computes A * ((1 - f) + f * B), which is a factor on A when B is constant.
    fn mix_color(&mut self, node_id: &Id, depth: usize) -> crate::error::Result<ColorIn> {
        let node = self.node(node_id)?;
        let (factor_socket, a_socket, b_socket) = match node.node_type.as_str() {
            "ShaderNodeMix" => ("Factor_Float", "A_Color", "B_Color"),
            _ => ("Fac", "Color1", "Color2"),
        };
        let blend = prop_str(node, "blend_type").unwrap_or("MIX");
        let rgba_mode =
            node.node_type != "ShaderNodeMix" || prop_str(node, "data_type") == Some("RGBA");
        let factor = node
            .inputs
            .get(factor_socket)
            .and_then(Value::as_f64)
            .unwrap_or(0.5);
        let a = self.color_input(node_id, a_socket, [0.5; 3], depth)?;
        let b = self.color_input(node_id, b_socket, [0.5; 3], depth)?;
        if !rgba_mode || blend != "MULTIPLY" {
            self.loss(
                feature_id::UNSUPPORTED_NODE,
                format!("mix `{node_id}` ({blend}) is not supported; its first input is used"),
                None,
            );
            return Ok(a);
        }
        if self.link_into(node_id, factor_socket).is_some() {
            self.loss(
                feature_id::UNSUPPORTED_NODE,
                format!("mix `{node_id}` has a linked factor; its first input is used"),
                None,
            );
            return Ok(a);
        }
        // A constant `b` scales `a`; `a` keeps its texture and vertex color.
        if b.tex.is_none() && b.vertex.is_none() {
            let scale = b.factor.map(|value| (1.0 - factor) + factor * value);
            return Ok(ColorIn {
                factor: [
                    a.factor[0] * scale[0],
                    a.factor[1] * scale[1],
                    a.factor[2] * scale[2],
                ],
                tex: a.tex,
                vertex: a.vertex,
            });
        }
        if crate::float::equal_f64(factor, 1.0)
            && let Some(product) = multiply(&a, &b)
        {
            return Ok(product);
        }
        self.loss(
            feature_id::UNSUPPORTED_NODE,
            format!("mix `{node_id}` multiplies two textures or vertex colors (or uses a factor other than 1); its first input is used"),
            None,
        );
        Ok(a)
    }

    /// Scalar input (metallic, roughness, weights, ...). Constant when unlinked.
    fn scalar_input(
        &mut self,
        node_id: &Id,
        socket: &str,
        default: f64,
        depth: usize,
    ) -> crate::error::Result<ScalarIn> {
        let node = self.node(node_id)?;
        let fallback = node
            .inputs
            .get(socket)
            .and_then(Value::as_f64)
            .unwrap_or(default);
        match self.link_into(node_id, socket) {
            Some(link) => self.scalar_from(&link.from_node, &link.from_socket, fallback, depth),
            None => Ok(ScalarIn::constant(fallback)),
        }
    }

    fn scalar_from(
        &mut self,
        source: &Id,
        output: &str,
        fallback: f64,
        depth: usize,
    ) -> crate::error::Result<ScalarIn> {
        let depth = self.deeper(source, depth)?;
        let node = self.node(source)?;
        if is_vertex_color(node, output) {
            self.vertex_color_loss(source, &node.node_type);
            return Ok(ScalarIn::constant(1.0));
        }
        match (node.node_type.as_str(), output) {
            ("ShaderNodeTexImage", "Color") => Ok(ScalarIn {
                factor: 1.0,
                tex: Some(self.tex_sample(source, None, depth)?),
            }),
            ("ShaderNodeTexImage", "Alpha") => Ok(ScalarIn {
                factor: 1.0,
                tex: Some(self.tex_sample(source, Some(TextureChannel::Alpha), depth)?),
            }),
            ("ShaderNodeSeparateColor", _) => self.separated(source, output, fallback, depth),
            ("ShaderNodeMath", "Value") => self.math_scalar(source, depth, fallback),
            _ => {
                self.unsupported(source, &node.node_type, "a scalar input");
                Ok(ScalarIn::constant(fallback))
            }
        }
    }

    /// `Math` in scalar context: only MULTIPLY by a constant is a factor.
    fn math_scalar(
        &mut self,
        math_id: &Id,
        depth: usize,
        fallback: f64,
    ) -> crate::error::Result<ScalarIn> {
        let node = self.node(math_id)?;
        let operation = prop_str(node, "operation").unwrap_or("ADD");
        if operation != "MULTIPLY" {
            self.loss(
                feature_id::UNSUPPORTED_NODE,
                format!("Math `{math_id}` uses operation `{operation}`; only MULTIPLY is supported in a scalar input"),
                None,
            );
            return Ok(ScalarIn::constant(fallback));
        }
        let (linked0, linked1, c0, c1) = self.math_operands(math_id)?;
        match (linked0, linked1) {
            (true, true) => {
                self.loss(
                    feature_id::UNSUPPORTED_NODE,
                    format!("Math `{math_id}` multiplies two linked values; the product of the defaults is used"),
                    None,
                );
                Ok(ScalarIn::constant(c0 * c1))
            }
            (true, false) => {
                let x = self.scalar_input(math_id, "Value", c0, depth)?;
                Ok(ScalarIn {
                    factor: x.factor * c1,
                    tex: x.tex,
                })
            }
            (false, true) => {
                let x = self.scalar_input(math_id, "Value_001", c1, depth)?;
                Ok(ScalarIn {
                    factor: x.factor * c0,
                    tex: x.tex,
                })
            }
            (false, false) => Ok(ScalarIn::constant(c0 * c1)),
        }
    }

    /// Linked flags and constant values of a Math node's two operands (`Value`, `Value_001`).
    /// Blender's default operand value is 0.5.
    fn math_operands(&self, math_id: &Id) -> crate::error::Result<(bool, bool, f64, f64)> {
        let node = self.node(math_id)?;
        let linked0 = self.link_into(math_id, "Value").is_some();
        let linked1 = self.link_into(math_id, "Value_001").is_some();
        let c0 = node
            .inputs
            .get("Value")
            .and_then(Value::as_f64)
            .unwrap_or(0.5);
        let c1 = node
            .inputs
            .get("Value_001")
            .and_then(Value::as_f64)
            .unwrap_or(0.5);
        Ok((linked0, linked1, c0, c1))
    }

    /// Alpha socket: constant when unlinked, otherwise the linked alpha expression.
    fn alpha_input(
        &mut self,
        node_id: &Id,
        socket: &str,
        default: f64,
        depth: usize,
    ) -> crate::error::Result<Alpha> {
        let node = self.node(node_id)?;
        let fallback = node
            .inputs
            .get(socket)
            .and_then(Value::as_f64)
            .unwrap_or(default);
        match self.link_into(node_id, socket) {
            Some(link) => self.alpha_from(&link.from_node, &link.from_socket, fallback, depth),
            None => Ok(Alpha::Const(fallback)),
        }
    }

    fn alpha_from(
        &mut self,
        source: &Id,
        output: &str,
        fallback: f64,
        depth: usize,
    ) -> crate::error::Result<Alpha> {
        let depth = self.deeper(source, depth)?;
        let node = self.node(source)?;
        if is_vertex_color(node, output) {
            self.vertex_color_loss(source, &node.node_type);
            return Ok(Alpha::Const(1.0));
        }
        match (node.node_type.as_str(), output) {
            ("ShaderNodeTexImage", "Alpha") => Ok(Alpha::Tex {
                sample: self.tex_sample(source, Some(TextureChannel::Alpha), depth)?,
                factor: 1.0,
            }),
            ("ShaderNodeTexImage", _) => {
                self.loss(
                    feature_id::UNSUPPORTED_NODE,
                    format!("the Color output of image texture `{source}` drives an alpha input; alpha is 1"),
                    None,
                );
                Ok(Alpha::Const(1.0))
            }
            ("ShaderNodeSeparateColor", _) => {
                let scalar = self.separated(source, output, fallback, depth)?;
                Ok(match scalar.tex {
                    Some(sample) => Alpha::Tex {
                        sample,
                        factor: scalar.factor,
                    },
                    None => Alpha::Const(scalar.factor),
                })
            }
            ("ShaderNodeMath", "Value") => self.math_alpha(source, depth),
            _ => {
                self.unsupported(source, &node.node_type, "an alpha input");
                Ok(Alpha::Const(fallback))
            }
        }
    }

    /// Alpha through a Math node: MULTIPLY by a constant, or a clip comparison.
    fn math_alpha(&mut self, math_id: &Id, depth: usize) -> crate::error::Result<Alpha> {
        let node = self.node(math_id)?;
        let operation = prop_str(node, "operation").unwrap_or("ADD");
        match operation {
            "MULTIPLY" => {
                let (linked0, linked1, c0, c1) = self.math_operands(math_id)?;
                match (linked0, linked1) {
                    (true, false) => {
                        let x = self.alpha_input(math_id, "Value", c0, depth)?;
                        Ok(self.combine_alpha(math_id, x, Alpha::Const(c1)))
                    }
                    (false, true) => {
                        let x = self.alpha_input(math_id, "Value_001", c1, depth)?;
                        Ok(self.combine_alpha(math_id, x, Alpha::Const(c0)))
                    }
                    (false, false) => Ok(Alpha::Const(c0 * c1)),
                    (true, true) => {
                        self.loss(
                            feature_id::UNSUPPORTED_NODE,
                            format!(
                                "Math `{math_id}` multiplies two linked alpha values; alpha is 1"
                            ),
                            None,
                        );
                        Ok(Alpha::Const(1.0))
                    }
                }
            }
            "ROUND" | "GREATER_THAN" | "LESS_THAN" | "SUBTRACT" => {
                if let Some(predicate) = self.predicate(math_id, depth)? {
                    if !predicate.ge {
                        self.loss(
                            feature_id::UNSUPPORTED_NODE,
                            format!("Math `{math_id}` is an inverted alpha clip; MASK keeps the non-inverted region"),
                            Some("invert the texture or the comparison"),
                        );
                    }
                    Ok(Alpha::Clip {
                        sample: predicate.sample,
                        cutoff: predicate.cutoff,
                    })
                } else {
                    self.loss(
                        feature_id::UNSUPPORTED_NODE,
                        format!("Math `{math_id}` ({operation}) is not an alpha clip pattern; alpha is 1"),
                        None,
                    );
                    Ok(Alpha::Const(1.0))
                }
            }
            other => {
                self.loss(
                    feature_id::UNSUPPORTED_NODE,
                    format!(
                        "Math `{math_id}` uses operation `{other}` in an alpha input; alpha is 1"
                    ),
                    None,
                );
                Ok(Alpha::Const(1.0))
            }
        }
    }

    /// Clip comparison on a texture alpha: ROUND, `GREATER_THAN`, `LESS_THAN` and
    /// SUBTRACT(1, comparison), following Blender's `detect_alpha_clip`.
    fn predicate(&mut self, math_id: &Id, depth: usize) -> crate::error::Result<Option<Predicate>> {
        let depth = self.deeper(math_id, depth)?;
        let node = self.node(math_id)?;
        let operation = prop_str(node, "operation").unwrap_or("ADD");
        let (linked0, linked1, c0, c1) = self.math_operands(math_id)?;
        let (socket, cutoff, ge) = match (operation, linked0, linked1) {
            ("ROUND", true, _) => ("Value", 0.5, true),
            ("GREATER_THAN", true, false) => ("Value", c1, true),
            ("LESS_THAN", false, true) => ("Value_001", c0, true),
            ("LESS_THAN", true, false) => ("Value", c1, false),
            ("GREATER_THAN", false, true) => ("Value_001", c0, false),
            ("SUBTRACT", false, true) if crate::float::equal_f64(c0, 1.0) => {
                let Some(link) = self.link_into(math_id, "Value_001") else {
                    return Ok(None);
                };
                let inner = self.predicate(&link.from_node, depth)?;
                return Ok(inner.map(|p| Predicate { ge: !p.ge, ..p }));
            }
            _ => return Ok(None),
        };
        match self.alpha_input(math_id, socket, 1.0, depth)? {
            Alpha::Tex { sample, .. } => Ok(Some(Predicate { sample, cutoff, ge })),
            _ => Ok(None),
        }
    }

    /// Product of two alpha expressions. Textures are multiplied only when at most one is
    /// texture-based; otherwise `primary` is kept and a loss is reported.
    fn combine_alpha(&mut self, owner: &Id, primary: Alpha, secondary: Alpha) -> Alpha {
        match (primary, secondary) {
            (Alpha::Const(a), Alpha::Const(b)) => Alpha::Const(a * b),
            (Alpha::Const(k), Alpha::Tex { sample, factor })
            | (Alpha::Tex { sample, factor }, Alpha::Const(k)) => Alpha::Tex {
                sample,
                factor: factor * k,
            },
            (Alpha::Const(k), Alpha::Clip { sample, cutoff })
            | (Alpha::Clip { sample, cutoff }, Alpha::Const(k)) => {
                if !crate::float::equal_f64(k, 1.0) {
                    self.loss(
                        feature_id::UNSUPPORTED_NODE,
                        format!("alpha factor {k} on a clipped texture at `{owner}` is dropped"),
                        None,
                    );
                }
                Alpha::Clip { sample, cutoff }
            }
            (primary, _) => {
                self.loss(
                    feature_id::UNSUPPORTED_NODE,
                    format!("alpha at `{owner}` multiplies two textures; only the first is used"),
                    None,
                );
                primary
            }
        }
    }
}

/// Emitted color = color times strength; `None` when nothing is emitted.
fn emission_value(color: &ColorIn, strength: f64) -> Option<TexturedValue<[f64; 3]>> {
    let factor = color.factor.map(|channel| channel * strength);
    if crate::float::equal_f64_array(&factor, &[0.0; 3]) {
        return None;
    }
    Some(TexturedValue {
        factor,
        texture: color.tex.as_ref().map(rgb_texture),
    })
}

/// Plain PBR core with Principled defaults for everything not given.
fn plain_core(base: ColorIn, metallic: ScalarIn, roughness: ScalarIn) -> Core {
    Core {
        base,
        alpha: Alpha::Const(1.0),
        metallic,
        roughness,
        normal: None,
        ior: 1.5,
        transmission: None,
        clearcoat: None,
        sheen: None,
        specular: None,
        anisotropy: None,
    }
}

impl GraphReducer<'_> {
    /// A numeric socket that glTF cannot express as a value: linked sockets are a loss and
    /// yield `linked`; unlinked sockets yield their constant value (or `default`).
    fn constant_input(
        &mut self,
        node_id: &Id,
        node: &GraphNode,
        socket: &str,
        default: f64,
        linked: f64,
    ) -> f64 {
        if self.link_into(node_id, socket).is_some() {
            self.loss(
                feature_id::UNSUPPORTED_INPUT,
                format!("linked `{socket}` on `{node_id}` is not representable in glTF; {linked} is used"),
                None,
            );
            return linked;
        }
        node.inputs
            .get(socket)
            .and_then(Value::as_f64)
            .unwrap_or(default)
    }

    /// A Principled input with no glTF representation: a loss when linked or non-default.
    fn reject_input(&mut self, node_id: &Id, node: &GraphNode, socket: &str, default: f64) {
        let linked = self.link_into(node_id, socket).is_some();
        let value = node
            .inputs
            .get(socket)
            .and_then(Value::as_f64)
            .unwrap_or(default);
        if linked || !crate::float::equal_f64(value, default) {
            self.loss(
                feature_id::UNSUPPORTED_INPUT,
                format!("Principled `{node_id}` input `{socket}` has no glTF representation and is ignored"),
                None,
            );
        }
    }

    /// Shader feeding a Surface-like socket. Unlinked sockets yield an empty shade.
    fn shader_input(
        &mut self,
        node_id: &Id,
        socket: &str,
        depth: usize,
    ) -> crate::error::Result<Shade> {
        match self.link_into(node_id, socket) {
            Some(link) => self.surface(&link.from_node, depth),
            None => Ok(Shade::default()),
        }
    }

    /// Reduces one shader node. Unsupported shaders are losses and yield an empty shade.
    fn surface(&mut self, source: &Id, depth: usize) -> crate::error::Result<Shade> {
        let depth = self.deeper(source, depth)?;
        let node = self.node(source)?;
        match node.node_type.as_str() {
            "ShaderNodeBsdfPrincipled" => self.principled(source, depth),
            "ShaderNodeEmission" => self.emission_shade(source, node, depth),
            "ShaderNodeBackground" => self.background_shade(source, node, depth),
            "ShaderNodeBsdfTransparent" => Ok(Shade {
                transparent: true,
                ..Shade::default()
            }),
            "ShaderNodeMixShader" => self.mix_shader(source, depth),
            "ShaderNodeAddShader" => self.add_shader(source, depth),
            other if other.starts_with("ShaderNodeBsdf") => self.other_bsdf(source, node, depth),
            other => {
                self.loss(
                    feature_id::UNSUPPORTED_NODE,
                    format!("surface node `{source}` ({other}) is not supported; Principled defaults are used"),
                    None,
                );
                Ok(Shade::default())
            }
        }
    }

    fn emission_shade(
        &mut self,
        source: &Id,
        node: &GraphNode,
        depth: usize,
    ) -> crate::error::Result<Shade> {
        let color = self.plain_color_input(source, "Color", [1.0; 3], depth)?;
        let strength = self.constant_input(source, node, "Strength", 1.0, 1.0);
        Ok(Shade {
            emission: emission_value(&color, strength),
            ..Shade::default()
        })
    }

    /// `Background` is unlit. Its strength is folded into the color when every product is <= 1.
    fn background_shade(
        &mut self,
        source: &Id,
        node: &GraphNode,
        depth: usize,
    ) -> crate::error::Result<Shade> {
        let color = self.color_input(source, "Color", [1.0; 3], depth)?;
        let strength = self.constant_input(source, node, "Strength", 1.0, 1.0);
        let product = color.factor.map(|channel| channel * strength);
        let factor = if product.iter().all(|channel| *channel <= 1.0) {
            product
        } else {
            self.loss(
                feature_id::UNSUPPORTED_NODE,
                format!("Background `{source}` has color times strength above 1; glTF unlit color is clamped to 1"),
                None,
            );
            product.map(|channel| channel.clamp(0.0, 1.0))
        };
        Ok(Shade {
            unlit: Some(ColorIn {
                factor,
                tex: color.tex,
                vertex: color.vertex,
            }),
            ..Shade::default()
        })
    }

    /// `Mix Shader`: `Fac` against a Transparent arm is alpha; otherwise glTF cannot blend.
    fn mix_shader(&mut self, source: &Id, depth: usize) -> crate::error::Result<Shade> {
        let fac = self.alpha_input(source, "Fac", 0.5, depth)?;
        let first = self.shader_input(source, "Shader", depth)?;
        let second = self.shader_input(source, "Shader_001", depth)?;
        Ok(self.mix_shades(source, fac, first, second))
    }

    fn mix_shades(&mut self, owner: &Id, fac: Alpha, first: Shade, second: Shade) -> Shade {
        if first.is_empty() {
            return second;
        }
        if second.is_empty() {
            return first;
        }
        if first.transparent || second.transparent {
            if first.transparent && second.transparent {
                return first;
            }
            // Fac weights the Transparent arm: alpha = Fac (Shader 1) or 1 - Fac (Shader 2).
            let transparent_first = first.transparent;
            let mut other = if transparent_first { second } else { first };
            let alpha = if transparent_first {
                fac
            } else {
                self.invert_alpha(owner, fac)
            };
            let existing = other.mix_alpha.take();
            other.mix_alpha = Some(match existing {
                Some(inner) => self.combine_alpha(owner, inner, alpha),
                None => alpha,
            });
            return other;
        }
        if first.unlit.is_some() && second.core.is_some() {
            self.loss(
                feature_id::UNREPRESENTABLE_SURFACE,
                format!("Mix `{owner}` combines a Background with a Principled BSDF; the Principled BSDF is kept and the surface is not unlit"),
                None,
            );
            return Shade {
                unlit: None,
                ..second
            };
        }
        if second.unlit.is_some() && first.core.is_some() {
            self.loss(
                feature_id::UNREPRESENTABLE_SURFACE,
                format!("Mix `{owner}` combines a Background with a Principled BSDF; the Principled BSDF is kept and the surface is not unlit"),
                None,
            );
            return Shade {
                unlit: None,
                ..first
            };
        }
        // A PBR surface mixed with an emission-only surface: the PBR core is kept and the
        // emitter's radiance is weighted by its mix factor (Blender keeps the PBR base and
        // adds the emission at full weight; the factor is applied here).
        let emission_only = |shade: &Shade| {
            shade.core.is_none() && shade.unlit.is_none() && shade.emission.is_some()
        };
        if first.core.is_some() && emission_only(&second)
            || second.core.is_some() && emission_only(&first)
        {
            let emitter_is_second = emission_only(&second);
            let weight = match (fac, emitter_is_second) {
                (Alpha::Const(weight), true) => Some(weight),
                (Alpha::Const(weight), false) => Some(1.0 - weight),
                _ => None,
            };
            if weight.is_none() {
                self.loss(
                    feature_id::UNSUPPORTED_NODE,
                    format!("Mix `{owner}` weights an emission arm by a texture; the emission is added at full weight"),
                    None,
                );
            }
            let (core_side, mut emitter) = if emitter_is_second {
                (first, second)
            } else {
                (second, first)
            };
            if let (Some(emission), Some(weight)) = (emitter.emission.as_mut(), weight) {
                emission
                    .factor
                    .iter_mut()
                    .for_each(|channel| *channel *= weight);
            }
            let merged = self.sum_emission(owner, core_side.emission.clone(), emitter.emission);
            self.loss(
                feature_id::UNREPRESENTABLE_SURFACE,
                format!("Mix `{owner}` blends a Principled BSDF with an emission shader; glTF keeps the PBR base and weights the emission"),
                None,
            );
            return Shade {
                emission: merged,
                ..core_side
            };
        }
        // glTF cannot blend two surfaces: the arm with weight >= 0.5 (Shader 2) wins, else Shader 1.
        let take_second = matches!(fac, Alpha::Const(weight) if weight >= 0.5);
        self.loss(
            feature_id::UNREPRESENTABLE_SURFACE,
            format!(
                "Mix `{owner}` blends two surfaces, which glTF cannot express; the {} arm is used",
                if take_second { "second" } else { "first" }
            ),
            Some("split the materials or use a single surface"),
        );
        if take_second { second } else { first }
    }

    /// Fac weight of a Transparent arm inverted (Shader 2). Only constant Fac is exact.
    fn invert_alpha(&mut self, owner: &Id, fac: Alpha) -> Alpha {
        match fac {
            Alpha::Const(value) => Alpha::Const(1.0 - value),
            other => {
                self.loss(
                    feature_id::UNSUPPORTED_NODE,
                    format!("Mix `{owner}` inverts a texture-driven Fac for a Transparent second arm; glTF cannot invert a texture, so the Fac is used as is"),
                    None,
                );
                other
            }
        }
    }

    fn add_shader(&mut self, source: &Id, depth: usize) -> crate::error::Result<Shade> {
        let first = self.shader_input(source, "Shader", depth)?;
        let second = self.shader_input(source, "Shader_001", depth)?;
        Ok(self.add_shades(source, first, second))
    }

    fn add_shades(&mut self, owner: &Id, first: Shade, second: Shade) -> Shade {
        if first.is_empty() {
            return second;
        }
        if second.is_empty() {
            return first;
        }
        if first.transparent || second.transparent {
            if first.transparent && second.transparent {
                return first;
            }
            self.loss(
                feature_id::UNREPRESENTABLE_SURFACE,
                format!("Add `{owner}` adds a Transparent BSDF; it contributes nothing to the glTF material"),
                None,
            );
            return if first.transparent { second } else { first };
        }
        let two_cores = first.core.is_some() && second.core.is_some();
        let two_unlit = first.unlit.is_some() && second.unlit.is_some();
        let unlit_and_core = (first.unlit.is_some() || second.unlit.is_some())
            && (first.core.is_some() || second.core.is_some());
        if two_cores {
            self.loss(
                feature_id::UNREPRESENTABLE_SURFACE,
                format!("Add `{owner}` adds two Principled-like surfaces; glTF has one PBR surface, so the first is kept"),
                None,
            );
        } else if two_unlit {
            self.loss(
                feature_id::UNREPRESENTABLE_SURFACE,
                format!("Add `{owner}` adds two Background surfaces; the first is kept"),
                None,
            );
        } else if unlit_and_core {
            self.loss(
                feature_id::UNREPRESENTABLE_SURFACE,
                format!("Add `{owner}` adds a Background to a Principled BSDF; the Background is dropped and the surface is not unlit"),
                None,
            );
        }
        let core = first.core.or(second.core);
        let unlit = if core.is_some() {
            None
        } else {
            first.unlit.or(second.unlit)
        };
        let emission = self.sum_emission(owner, first.emission, second.emission);
        let mix_alpha = match (first.mix_alpha, second.mix_alpha) {
            (Some(a), Some(b)) => Some(self.combine_alpha(owner, a, b)),
            (a, b) => a.or(b),
        };
        Shade {
            core,
            emission,
            unlit,
            mix_alpha,
            transparent: false,
        }
    }

    /// Adds two emission contributions. Two textured emissions keep the first (loss).
    fn sum_emission(
        &mut self,
        owner: &Id,
        first: Option<TexturedValue<[f64; 3]>>,
        second: Option<TexturedValue<[f64; 3]>>,
    ) -> Option<TexturedValue<[f64; 3]>> {
        let (a, b) = match (first, second) {
            (None, other) | (other, None) => return other,
            (Some(a), Some(b)) => (a, b),
        };
        match (a.texture.is_some(), b.texture.is_some()) {
            (false, false) => Some(TexturedValue {
                factor: [
                    a.factor[0] + b.factor[0],
                    a.factor[1] + b.factor[1],
                    a.factor[2] + b.factor[2],
                ],
                texture: None,
            }),
            (true, true) => {
                self.loss(
                    feature_id::UNSUPPORTED_NODE,
                    format!(
                        "Add `{owner}` adds two textured emissions; the second texture is dropped"
                    ),
                    None,
                );
                Some(a)
            }
            (true, false) => {
                self.loss(
                    feature_id::UNSUPPORTED_NODE,
                    format!("Add `{owner}` adds a constant emission to a textured one; the constant is dropped"),
                    None,
                );
                Some(a)
            }
            (false, true) => {
                self.loss(
                    feature_id::UNSUPPORTED_NODE,
                    format!("Add `{owner}` adds a constant emission to a textured one; the constant is dropped"),
                    None,
                );
                Some(b)
            }
        }
    }

    /// Non-Principled BSDF as the surface: its color and roughness, with glTF PBR defaults.
    fn other_bsdf(
        &mut self,
        source: &Id,
        node: &GraphNode,
        depth: usize,
    ) -> crate::error::Result<Shade> {
        self.loss(
            feature_id::UNREPRESENTABLE_SURFACE,
            format!("surface `{source}` ({}) is not a Principled BSDF; its color is exported with glTF PBR defaults", node.node_type),
            None,
        );
        let color = self.color_input(source, "Color", [0.8; 3], depth)?;
        let roughness = if node.node_type == "ShaderNodeBsdfDiffuse" {
            ScalarIn::constant(1.0)
        } else {
            self.scalar_input(source, "Roughness", 0.5, depth)?
        };
        Ok(Shade {
            core: Some(plain_core(color, ScalarIn::constant(0.0), roughness)),
            ..Shade::default()
        })
    }

    /// Normal input: `NormalMap` (tangent space) with an image color, or a loss.
    fn normal_input(
        &mut self,
        node_id: &Id,
        socket: &str,
        depth: usize,
    ) -> crate::error::Result<Option<TexturedNormal>> {
        let Some(link) = self.link_into(node_id, socket) else {
            return Ok(None);
        };
        let source = &link.from_node;
        let depth = self.deeper(source, depth)?;
        let node = self.node(source)?;
        match node.node_type.as_str() {
            "ShaderNodeNormalMap" => {
                let space = prop_str(node, "space").unwrap_or("TANGENT");
                if space != "TANGENT" {
                    self.loss(
                        feature_id::NORMAL_MAP,
                        format!("NormalMap `{source}` uses {space} space; glTF normal maps are tangent space, so it is exported as tangent space"),
                        None,
                    );
                }
                let scale = self.constant_input(source, node, "Strength", 1.0, 1.0);
                if self.link_into(source, "Color").is_none() {
                    return Ok(None);
                }
                let color = self.plain_color_input(source, "Color", [0.5, 0.5, 1.0], depth)?;
                match color.tex {
                    Some(sample)
                        if crate::float::equal_f64_array(&color.factor, &[1.0; 3])
                            && sample.channel.is_none() =>
                    {
                        Ok(Some(TexturedNormal {
                            texture: normal_texture(&sample),
                            scale,
                        }))
                    }
                    Some(_) => {
                        self.loss(
                            feature_id::UNSUPPORTED_NODE,
                            format!("NormalMap `{source}` input is not a plain image color; no normal texture is exported"),
                            None,
                        );
                        Ok(None)
                    }
                    None => Ok(None),
                }
            }
            "ShaderNodeBump" => {
                self.loss(
                    feature_id::NORMAL_MAP,
                    format!("Bump `{source}` drives the normal; glTF has no height-to-normal input, so the normal is dropped"),
                    None,
                );
                Ok(None)
            }
            "ShaderNodeTexImage" => {
                self.loss(
                    feature_id::NORMAL_MAP,
                    format!("image texture `{source}` drives Normal directly; glTF needs a NormalMap node, so the normal is dropped"),
                    None,
                );
                Ok(None)
            }
            other => {
                self.unsupported(source, other, &format!("the Normal of `{node_id}`"));
                Ok(None)
            }
        }
    }

    /// `Principled BSDF`, input by input (research §4.2).
    fn principled(&mut self, node_id: &Id, depth: usize) -> crate::error::Result<Shade> {
        let depth = self.deeper(node_id, depth)?;
        let node = self.node(node_id)?;
        let base = self.color_input(node_id, "Base Color", [0.8; 3], depth)?;
        let alpha = self.alpha_input(node_id, "Alpha", 1.0, depth)?;
        let metallic = self.scalar_input(node_id, "Metallic", 0.0, depth)?;
        let roughness = self.scalar_input(node_id, "Roughness", 0.5, depth)?;
        let normal = self.normal_input(node_id, "Normal", depth)?;
        let ior = self.constant_input(node_id, node, "IOR", 1.5, 1.5);
        let transmission = self.scalar_input(node_id, "Transmission Weight", 0.0, depth)?;
        let clearcoat = self.clearcoat(node_id, node, depth)?;
        let sheen = self.sheen(node_id, depth)?;
        let specular = self.specular(node_id, depth)?;
        let anisotropy = self.anisotropy(node_id, node);
        self.reject_input(node_id, node, "Subsurface Weight", 0.0);
        self.reject_input(node_id, node, "Thin Film Thickness", 0.0);
        self.reject_input(node_id, node, "Diffuse Roughness", 0.0);
        self.reject_input(node_id, node, "Weight", 0.0);
        if self.link_into(node_id, "Tangent").is_some() {
            self.loss(
                feature_id::UNSUPPORTED_INPUT,
                format!(
                    "linked `Tangent` on `{node_id}` is not representable in glTF; it is ignored"
                ),
                None,
            );
        }
        if node.inputs.get("Thin Wall").and_then(Value::as_bool) == Some(true) {
            self.loss(
                feature_id::UNSUPPORTED_INPUT,
                format!("Principled `{node_id}` has Thin Wall enabled, which glTF cannot express; it is ignored"),
                None,
            );
        }
        let emission_color = self.plain_color_input(node_id, "Emission Color", [1.0; 3], depth)?;
        let emission_strength = self.constant_input(node_id, node, "Emission Strength", 0.0, 1.0);
        let transmission =
            (transmission.factor != 0.0 || transmission.tex.is_some()).then(|| TexturedValue {
                factor: transmission.factor,
                texture: transmission
                    .tex
                    .as_ref()
                    .map(|sample| slot_texture(sample, 0, TextureChannel::Red)),
            });
        Ok(Shade {
            core: Some(Core {
                base,
                alpha,
                metallic,
                roughness,
                normal,
                ior,
                transmission,
                clearcoat,
                sheen,
                specular,
                anisotropy,
            }),
            emission: emission_value(&emission_color, emission_strength),
            ..Shade::default()
        })
    }

    /// `Coat Weight` (R), `Coat Roughness` (G), `Coat Normal`. Coat IOR and tint are losses.
    fn clearcoat(
        &mut self,
        node_id: &Id,
        node: &GraphNode,
        depth: usize,
    ) -> crate::error::Result<Option<Clearcoat>> {
        let weight = self.scalar_input(node_id, "Coat Weight", 0.0, depth)?;
        let roughness = self.scalar_input(node_id, "Coat Roughness", 0.03, depth)?;
        let normal = self.normal_input(node_id, "Coat Normal", depth)?;
        let ior = self.constant_input(node_id, node, "Coat IOR", 1.5, 1.5);
        if !crate::float::equal_f64(ior, 1.5) {
            self.loss(
                feature_id::UNSUPPORTED_INPUT,
                format!("Coat IOR {ior} on `{node_id}`: glTF clearcoat has a fixed IOR of 1.5; it is ignored"),
                None,
            );
        }
        let tint = self.plain_color_input(node_id, "Coat Tint", [1.0; 3], depth)?;
        if tint.tex.is_some() || !crate::float::equal_f64_array(&tint.factor, &[1.0; 3]) {
            self.loss(
                feature_id::UNSUPPORTED_INPUT,
                format!(
                    "Coat Tint on `{node_id}` is not representable in glTF clearcoat; it is ignored"
                ),
                None,
            );
        }
        let has_texture = weight.tex.is_some() || roughness.tex.is_some() || normal.is_some();
        if weight.factor == 0.0 && !has_texture {
            return Ok(None);
        }
        Ok(Some(Clearcoat {
            factor: weight.factor,
            texture: weight
                .tex
                .as_ref()
                .map(|sample| slot_texture(sample, 0, TextureChannel::Red)),
            roughness_factor: roughness.factor,
            roughness_texture: roughness
                .tex
                .as_ref()
                .map(|sample| slot_texture(sample, 1, TextureChannel::Green)),
            normal,
        }))
    }

    /// `Sheen Weight` scales `Sheen Tint`; `Sheen Roughness` (A) is the sheen roughness.
    fn sheen(&mut self, node_id: &Id, depth: usize) -> crate::error::Result<Option<Sheen>> {
        let weight = self.scalar_input(node_id, "Sheen Weight", 0.0, depth)?;
        if weight.tex.is_some() {
            self.loss(
                feature_id::UNSUPPORTED_INPUT,
                format!("textured Sheen Weight on `{node_id}` has no glTF representation; the weight is used as a factor"),
                None,
            );
        }
        let tint = self.plain_color_input(node_id, "Sheen Tint", [1.0; 3], depth)?;
        let roughness = self.scalar_input(node_id, "Sheen Roughness", 0.5, depth)?;
        if weight.factor == 0.0 && tint.tex.is_none() && roughness.tex.is_none() {
            return Ok(None);
        }
        Ok(Some(Sheen {
            color_factor: tint.factor.map(|channel| channel * weight.factor),
            color_texture: tint.tex.as_ref().map(rgb_texture),
            roughness_factor: roughness.factor,
            roughness_texture: roughness
                .tex
                .as_ref()
                .map(|sample| slot_texture(sample, 3, TextureChannel::Alpha)),
        }))
    }

    /// `Specular IOR Level` (A): Blender's `2 * level` splits between `specularFactor`, capped at 1,
    /// and the tint, which takes the excess (Blender specular.py). A textured level reads the raw
    /// texture A, as Blender's exporter does, and its doubling moves into the tint.
    fn specular(&mut self, node_id: &Id, depth: usize) -> crate::error::Result<Option<Specular>> {
        let level = self.scalar_input(node_id, "Specular IOR Level", 0.5, depth)?;
        let tint = self.plain_color_input(node_id, "Specular Tint", [1.0; 3], depth)?;
        let doubled = 2.0 * level.factor;
        let factor = doubled.min(1.0);
        let color_factor = tint.factor.map(|channel| channel * doubled.max(1.0));
        if crate::float::equal_f64(factor, 1.0)
            && crate::float::equal_f64_array(&color_factor, &[1.0; 3])
            && level.tex.is_none()
            && tint.tex.is_none()
        {
            return Ok(None);
        }
        Ok(Some(Specular {
            factor,
            texture: level
                .tex
                .as_ref()
                .map(|sample| slot_texture(sample, 3, TextureChannel::Alpha)),
            color_factor,
            color_texture: tint.tex.as_ref().map(rgb_texture),
        }))
    }

    /// `Anisotropic` and `Anisotropic Rotation` (turns, 1 turn = 2 pi).
    fn anisotropy(&mut self, node_id: &Id, node: &GraphNode) -> Option<Anisotropy> {
        let strength = self.constant_input(node_id, node, "Anisotropic", 0.0, 0.0);
        let turns = self.constant_input(node_id, node, "Anisotropic Rotation", 0.0, 0.0);
        if strength == 0.0 {
            return None;
        }
        Some(Anisotropy {
            strength,
            rotation: turns * std::f64::consts::TAU,
        })
    }
}

/// The output node that drives the surface: the active one, else the first (evaluator rule).
fn output_node(group: &NodeGroup) -> Option<&Id> {
    let mut fallback = None;
    for (id, node) in &group.nodes {
        if matches!(
            node.node_type.as_str(),
            "OutputMaterial" | "ShaderNodeOutputMaterial"
        ) {
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
    }
    fallback
}

/// Node-group name of a `ShaderNodeGroup` (`node_tree.name`, or a plain string).
fn group_tree_name(node: &GraphNode) -> Option<&str> {
    let value = node.properties.get("node_tree")?;
    value
        .get("name")
        .and_then(Value::as_str)
        .or_else(|| value.as_str())
}

/// Interface name of input `socket` of the node group named `tree`.
fn group_input_name<'d>(doc: &'d SceneDoc, tree: &str, socket: &str) -> Option<&'d str> {
    doc.node_groups
        .values()
        .find(|group| group.name == tree)?
        .interface
        .inputs
        .iter()
        .find(|input| input.id == socket)
        .map(|input| input.name.as_str())
}

impl GraphReducer<'_> {
    fn check_sampling(&mut self, slot: &str, expected: &TexBase, actual: &TexBase) {
        if let Some((feature, what)) = sampling_mismatch(expected, actual) {
            self.loss(
                feature,
                format!("the {slot} texture differs in {what} from the packed texture; the packed texture's {what} is used"),
                None,
            );
        }
    }

    /// Native texture references are ignored for graph materials: the graph holds the textures and
    /// is authoritative. `alpha_mode` is not a field here: it is Blender's render method (what
    /// Potter's renderer uses), and glTF takes `alphaMode` from the graph's Alpha path.
    fn material_field_losses(&mut self, material: &Material) {
        let mut fields: Vec<String> = Vec::new();
        if material.base_color_texture.is_some() {
            fields.push("base_color_texture".to_owned());
        }
        if material.metallic_texture.is_some() {
            fields.push("metallic_texture".to_owned());
        }
        if material.roughness_texture.is_some() {
            fields.push("roughness_texture".to_owned());
        }
        if material.normal_texture.is_some() {
            fields.push("normal_texture".to_owned());
        }
        if fields.is_empty() {
            return;
        }
        self.loss(
            feature_id::MATERIAL_TEXTURES,
            format!(
                "graph material ignores Material fields ({}); the graph is authoritative",
                fields.join(", ")
            ),
            Some("remove the Material texture references and use image texture nodes in the graph"),
        );
    }

    /// Occlusion from the `glTF Material Output` group's `Occlusion` input (R channel). The
    /// group node need not be linked to the output, so every node of the graph is scanned.
    fn occlusion(&mut self) -> crate::error::Result<Option<ScalarIn>> {
        let group = self.group;
        let doc = self.doc;
        let mut found = None;
        for (node_id, node) in &group.nodes {
            if node.node_type != "ShaderNodeGroup" {
                continue;
            }
            let Some(tree) = group_tree_name(node) else {
                continue;
            };
            if !tree.starts_with("glTF Material Output") {
                continue;
            }
            for link in group.links.iter().filter(|link| link.to_node == *node_id) {
                let name = group_input_name(doc, tree, link.to_socket.as_str())
                    .unwrap_or(link.to_socket.as_str());
                if name != "Occlusion" {
                    self.loss(
                        feature_id::UNSUPPORTED_NODE,
                        format!("glTF Material Output `{node_id}` input `{name}` is not supported; it is ignored"),
                        None,
                    );
                    continue;
                }
                if found.is_some() {
                    self.loss(
                        feature_id::UNSUPPORTED_NODE,
                        format!("glTF Material Output `{node_id}` has a second Occlusion link; it is ignored"),
                        None,
                    );
                    continue;
                }
                let occlusion = self.scalar_from(&link.from_node, &link.from_socket, 1.0, 0)?;
                if occlusion.tex.is_none() {
                    self.loss(
                        feature_id::UNSUPPORTED_INPUT,
                        format!("constant occlusion on `{node_id}` has no glTF representation; it is ignored"),
                        None,
                    );
                } else {
                    if !crate::float::equal_f64(occlusion.factor, 1.0) {
                        self.loss(
                            feature_id::UNSUPPORTED_NODE,
                            format!("occlusion on `{node_id}` is multiplied by {}; glTF occlusion strength is not a multiplier, so the factor is ignored", occlusion.factor),
                            None,
                        );
                    }
                    found = Some(occlusion);
                }
            }
        }
        Ok(found)
    }

    /// Base color texture: RGB from the color sample, A from the alpha sample.
    fn base_texture(
        &mut self,
        color: Option<&Sample>,
        alpha: Option<&Sample>,
    ) -> Option<SampledTexture> {
        let base = color.or(alpha)?;
        if let (Some(color), Some(alpha)) = (color, alpha) {
            self.check_sampling("alpha", &color.base, &alpha.base);
        }
        let mut channels: [Option<ChannelSource>; 4] = [None, None, None, None];
        if let Some(color) = color {
            let rgb = match color.channel {
                None => [
                    TextureChannel::Red,
                    TextureChannel::Green,
                    TextureChannel::Blue,
                ],
                Some(channel) => [channel; 3],
            };
            for (slot, channel) in rgb.into_iter().enumerate() {
                channels[slot] = Some(channel_of(&color.base.image, channel));
            }
        }
        if let Some(alpha) = alpha {
            channels[3] = Some(channel_of(
                &alpha.base.image,
                alpha.channel.unwrap_or(TextureChannel::Alpha),
            ));
        }
        Some(sampled_texture(&base.base, channels))
    }

    /// The packed metallicRoughness texture (G roughness, B metallic). Occlusion (R) joins it
    /// when it reads the same sampling state; the flag says whether it did.
    fn packed_texture(
        &mut self,
        roughness: Option<&Sample>,
        metallic: Option<&Sample>,
        occlusion: Option<&Sample>,
    ) -> (Option<SampledTexture>, bool) {
        let Some(group_base) = roughness.or(metallic) else {
            return (None, false);
        };
        let mut channels: [Option<ChannelSource>; 4] = [None, None, None, None];
        if let Some(rough) = roughness {
            self.check_sampling("roughness", &group_base.base, &rough.base);
            channels[1] = Some(channel_of(
                &rough.base.image,
                rough.channel.unwrap_or(TextureChannel::Green),
            ));
        }
        if let Some(metal) = metallic {
            self.check_sampling("metallic", &group_base.base, &metal.base);
            channels[2] = Some(channel_of(
                &metal.base.image,
                metal.channel.unwrap_or(TextureChannel::Blue),
            ));
        }
        let packed = occlusion.is_some_and(|occ| occ.base == group_base.base);
        if let (true, Some(occ)) = (packed, occlusion) {
            channels[0] = Some(channel_of(
                &occ.base.image,
                occ.channel.unwrap_or(TextureChannel::Red),
            ));
        }
        (Some(sampled_texture(&group_base.base, channels)), packed)
    }

    /// Builds the glTF material from the reduced shade and the occlusion source.
    fn assemble(
        &mut self,
        material: &Material,
        shade: Shade,
        occlusion: Option<ScalarIn>,
    ) -> ResolvedSurface {
        let material_id = self.material_id;
        let Shade {
            core,
            mut emission,
            unlit,
            mix_alpha,
            transparent,
        } = shade;
        let (core, is_unlit) = match (core, unlit) {
            (Some(core), _) => (core, false),
            (None, Some(color)) => (
                plain_core(color, ScalarIn::constant(0.0), ScalarIn::constant(0.9)),
                true,
            ),
            (None, None) if emission.is_some() => (
                plain_core(
                    ColorIn::constant([0.0; 3]),
                    ScalarIn::constant(1.0),
                    ScalarIn::constant(1.0),
                ),
                false,
            ),
            (None, None) => (
                plain_core(
                    ColorIn::constant([0.8; 3]),
                    ScalarIn::constant(0.0),
                    ScalarIn::constant(0.5),
                ),
                false,
            ),
        };
        if is_unlit && emission.is_some() {
            self.loss(
                feature_id::UNREPRESENTABLE_SURFACE,
                format!("Background with an emission term on material `{material_id}`: glTF unlit ignores emission; the emission is dropped"),
                None,
            );
            emission = None;
        }
        let mix_alpha = if transparent {
            Some(Alpha::Const(0.0))
        } else {
            mix_alpha
        };
        let Core {
            base,
            alpha: core_alpha,
            metallic,
            roughness,
            normal,
            ior,
            transmission,
            clearcoat,
            sheen,
            specular,
            anisotropy,
        } = core;
        let alpha = match mix_alpha {
            Some(mix) => self.combine_alpha(material_id, core_alpha, mix),
            None => core_alpha,
        };
        let (alpha_factor, alpha_sample, alpha_mode) = match &alpha {
            Alpha::Const(value) => (
                *value,
                None,
                if crate::float::equal_f64(*value, 1.0) {
                    AlphaMode::Opaque
                } else {
                    AlphaMode::Blend
                },
            ),
            Alpha::Tex { sample, factor } => (*factor, Some(sample), AlphaMode::Blend),
            Alpha::Clip { sample, cutoff } => {
                (1.0, Some(sample), AlphaMode::Mask { cutoff: *cutoff })
            }
        };
        let base_texture = self.base_texture(base.tex.as_ref(), alpha_sample);
        let (metallic_roughness_texture, packed) = self.packed_texture(
            roughness.tex.as_ref(),
            metallic.tex.as_ref(),
            occlusion.as_ref().and_then(|occ| occ.tex.as_ref()),
        );
        let occlusion = occlusion.and_then(|occ| {
            let texture = if packed {
                metallic_roughness_texture.clone()
            } else {
                occ.tex
                    .as_ref()
                    .map(|sample| slot_texture(sample, 0, TextureChannel::Red))
            };
            // Blender's exporter writes strength only for its Mix-with-white pattern, which this
            // reducer does not model; a multiplied occlusion factor is a loss (see `occlusion`).
            texture.map(|texture| TexturedOcclusion {
                texture,
                strength: 1.0,
            })
        });
        ResolvedSurface {
            double_sided: material.double_sided,
            alpha_mode,
            unlit: is_unlit,
            base_color: TexturedValue {
                factor: [base.factor[0], base.factor[1], base.factor[2], alpha_factor],
                texture: base_texture,
            },
            vertex_color: base.vertex,
            unmultiplied_vertex_color: self.dropped_vertex.take(),
            metallic: metallic.factor,
            roughness: roughness.factor,
            metallic_roughness_texture,
            occlusion,
            normal,
            emissive: emission.unwrap_or(TexturedValue {
                factor: [0.0; 3],
                texture: None,
            }),
            ior,
            transmission,
            clearcoat,
            sheen,
            specular,
            anisotropy,
        }
    }
}

/// Resolves a graph material: the Surface input of the output node, plus the occlusion group.
fn graph_surface(
    doc: &SceneDoc,
    material_id: &Id,
    material: &Material,
    graph_id: &Id,
) -> crate::error::Result<(ResolvedSurface, Vec<super::Loss>)> {
    let Some(group) = doc.node_groups.get(graph_id) else {
        return Err(invalid_error(
            material_id,
            None,
            format!("shader graph `{graph_id}` was not found"),
        ));
    };
    if group.kind != GraphKind::Shader {
        return Err(invalid_error(
            material_id,
            None,
            format!("material graph `{graph_id}` is not a shader graph"),
        ));
    }
    let Some(output) = output_node(group) else {
        return Err(invalid_error(
            material_id,
            None,
            format!("shader graph `{graph_id}` has no OutputMaterial node"),
        ));
    };
    let mut reducer = GraphReducer {
        doc,
        material_id,
        group,
        losses: Vec::new(),
        dropped_vertex: None,
    };
    reducer.material_field_losses(material);
    let shade = if let Some(link) = reducer.link_into(output, "Surface") {
        reducer.surface(&link.from_node, 0)?
    } else {
        reducer.loss(
            feature_id::UNREPRESENTABLE_SURFACE,
            format!("material output `{output}` has no Surface link; Principled defaults are used"),
            None,
        );
        Shade::default()
    };
    if reducer.link_into(output, "Displacement").is_some() {
        reducer.loss(
            feature_id::DISPLACEMENT,
            format!("displacement on `{output}` is not exported; glTF has no displacement"),
            None,
        );
    }
    if reducer.link_into(output, "Volume").is_some() {
        reducer.loss(
            feature_id::VOLUME,
            format!(
                "volume on `{output}` is not exported; the glTF volume extension is not implemented"
            ),
            None,
        );
    }
    let occlusion = reducer.occlusion()?;
    let surface = reducer.assemble(material, shade, occlusion);
    Ok((surface, reducer.losses))
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests")]
    #![expect(clippy::expect_used, reason = "tests")]

    use serde_json::{Value, json};

    use super::*;
    use crate::{
        exchange::Loss,
        model::{DataBlock, Image, ImageAlphaMode, ImageColorspace, ImageSource, Node},
    };

    fn id(value: &str) -> Id {
        Id::new(value).unwrap()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    fn image() -> Image {
        Image {
            name: "img".to_owned(),
            source: ImageSource::default(),
            colorspace: ImageColorspace::default(),
            width: 4,
            height: 4,
            tiles: Vec::new(),
            blob: None,
            source_path: None,
            source_hash: None,
            alpha_mode: ImageAlphaMode::default(),
            tiled: false,
        }
    }

    /// Document with image `img` and material `mat`, driven by graph `graph` when given.
    fn document(graph: Option<NodeGroup>) -> SceneDoc {
        let mut doc = SceneDoc::default();
        doc.images.insert(id("img"), image());
        let mut material = Material {
            name: "mat".to_owned(),
            ..Material::default()
        };
        if let Some(group) = graph {
            doc.node_groups.insert(id("graph"), group);
            material.node_tree = Some(id("graph"));
        }
        doc.materials.insert(id("mat"), material);
        doc
    }

    #[test]
    fn material_use_includes_data_linked_slots() {
        let mut doc = SceneDoc::default();
        let material_id = id("data_material");
        let data_id = id("mesh_data");
        doc.materials
            .insert(material_id.clone(), Material::default());
        doc.data_blocks.insert(
            data_id.clone(),
            DataBlock {
                data_type: "mesh".to_owned(),
                materials: vec![Some(material_id.clone())],
                ..DataBlock::default()
            },
        );
        doc.nodes.insert(
            id("mesh"),
            Node {
                data: Some(data_id),
                ..Node::default()
            },
        );

        assert!(material_is_used(&doc, &material_id));
    }

    fn add_node(
        group: &mut NodeGroup,
        name: &str,
        node_type: &str,
        properties: &Value,
        inputs: &Value,
    ) {
        let mut node = GraphNode::new(node_type);
        node.properties = properties.as_object().cloned().unwrap_or_default();
        node.inputs = inputs
            .as_object()
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect();
        group.nodes.insert(id(name), node);
    }

    fn link(group: &mut NodeGroup, from: &str, from_socket: &str, to: &str, to_socket: &str) {
        group.links.push(GraphLink {
            from_node: id(from),
            from_socket: from_socket.to_owned(),
            to_node: id(to),
            to_socket: to_socket.to_owned(),
        });
    }

    /// Shader graph: output `out`, Principled `bsdf` (with `bsdf_inputs`) wired to Surface.
    fn principled_graph(bsdf_inputs: &Value) -> NodeGroup {
        let mut group = NodeGroup::new("graph", GraphKind::Shader);
        add_node(
            &mut group,
            "out",
            "OutputMaterial",
            &json!({"is_active_output": true}),
            &json!({}),
        );
        add_node(
            &mut group,
            "bsdf",
            "ShaderNodeBsdfPrincipled",
            &json!({}),
            bsdf_inputs,
        );
        link(&mut group, "bsdf", "BSDF", "out", "Surface");
        group
    }

    fn resolved(doc: &SceneDoc) -> (ResolvedSurface, Vec<Loss>) {
        resolve_surface(doc, &id("mat")).unwrap()
    }

    #[test]
    fn fields_path_resolves_a_material_without_graph() {
        let doc = document(None);
        let (surface, losses) = resolved(&doc);
        assert!(losses.is_empty());
        assert_eq!(surface.alpha_mode, AlphaMode::Opaque);
        assert!(surface.base_color.texture.is_none());
        assert_eq!(surface.base_color.factor, [0.6, 0.6, 0.6, 1.0]);
        assert!(!surface.unlit);
        let fields = resolve_fields_surface(&doc, &id("mat")).unwrap();
        assert_eq!(fields, surface);
    }
    #[test]
    fn clip_texture_field_reports_sampler_loss() {
        let mut doc = document(None);
        doc.materials
            .get_mut(&id("mat"))
            .unwrap()
            .base_color_texture = Some(TextureRef {
            image: id("img"),
            uv_layer: None,
            interpolation: ImageInterpolation::Linear,
            extension: ImageExtension::Clip,
        });
        let (_, losses) = resolved(&doc);
        assert!(
            losses
                .iter()
                .any(|loss| loss.feature_id == feature_id::SAMPLER)
        );
    }

    #[test]
    fn point_mapping_becomes_texture_transform() {
        let mut group = principled_graph(&json!({}));
        add_node(
            &mut group,
            "uv",
            "ShaderNodeUVMap",
            &json!({"uv_map": "UVMap"}),
            &json!({}),
        );
        add_node(
            &mut group,
            "map",
            "ShaderNodeMapping",
            &json!({"vector_type": "POINT"}),
            &json!({
                "Location": [0.1, 0.2, 0.0],
                "Rotation": "Euler((0.0, 0.0, 0.5235987901687622), 'XYZ')",
                "Scale": [2.0, 3.0, 1.0],
            }),
        );
        add_node(
            &mut group,
            "tex",
            "ShaderNodeTexImage",
            &json!({"image": "img", "interpolation": "Linear", "extension": "REPEAT"}),
            &json!({}),
        );
        link(&mut group, "uv", "UV", "map", "Vector");
        link(&mut group, "map", "Vector", "tex", "Vector");
        link(&mut group, "tex", "Color", "bsdf", "Base Color");
        let (surface, losses) = resolved(&document(Some(group)));
        assert!(losses.is_empty(), "{losses:?}");
        let texture = surface.base_color.texture.unwrap();
        assert_eq!(texture.uv_layer.as_deref(), Some("UVMap"));
        let transform = texture.transform.unwrap();
        assert!(close(transform.offset[0], -1.4));
        assert!(close(transform.offset[1], -1.798_076_2));
        assert!(close(transform.rotation, 0.523_598_790_168_762_1));
        assert_eq!(transform.scale, [2.0, 3.0]);
    }

    #[test]
    fn texture_mapping_is_inverted() {
        let mut group = principled_graph(&json!({}));
        add_node(
            &mut group,
            "map",
            "ShaderNodeMapping",
            &json!({"vector_type": "TEXTURE"}),
            &json!({"Location": [0.25, 0.0, 0.0], "Scale": [2.0, 2.0, 1.0]}),
        );
        add_node(
            &mut group,
            "tex",
            "ShaderNodeTexImage",
            &json!({"image": "img"}),
            &json!({}),
        );
        link(&mut group, "map", "Vector", "tex", "Vector");
        link(&mut group, "tex", "Color", "bsdf", "Base Color");
        let (surface, losses) = resolved(&document(Some(group)));
        assert!(losses.is_empty(), "{losses:?}");
        let transform = surface.base_color.texture.unwrap().transform.unwrap();
        assert!(close(transform.offset[0], -0.125));
        assert!(close(transform.offset[1], 0.5));
        assert_eq!(transform.scale, [0.5, 0.5]);
    }

    #[test]
    fn greater_than_alpha_becomes_mask() {
        let mut group = principled_graph(&json!({"Base Color": [0.2, 0.4, 0.8, 1.0]}));
        add_node(
            &mut group,
            "tex",
            "ShaderNodeTexImage",
            &json!({"image": "img"}),
            &json!({}),
        );
        add_node(
            &mut group,
            "math",
            "ShaderNodeMath",
            &json!({"operation": "GREATER_THAN"}),
            &json!({"Value_001": 0.5}),
        );
        link(&mut group, "tex", "Alpha", "math", "Value");
        link(&mut group, "math", "Value", "bsdf", "Alpha");
        let (surface, losses) = resolved(&document(Some(group)));
        assert!(losses.is_empty(), "{losses:?}");
        assert_eq!(surface.alpha_mode, AlphaMode::Mask { cutoff: 0.5 });
        let texture = surface.base_color.texture.unwrap();
        assert_eq!(
            texture.channels[3],
            Some(ChannelSource {
                image: id("img"),
                channel: TextureChannel::Alpha,
            })
        );
    }

    #[test]
    fn subtracted_less_than_alpha_becomes_mask() {
        let mut group = principled_graph(&json!({}));
        add_node(
            &mut group,
            "tex",
            "ShaderNodeTexImage",
            &json!({"image": "img"}),
            &json!({}),
        );
        add_node(
            &mut group,
            "less",
            "ShaderNodeMath",
            &json!({"operation": "LESS_THAN"}),
            &json!({"Value_001": 0.5}),
        );
        add_node(
            &mut group,
            "sub",
            "ShaderNodeMath",
            &json!({"operation": "SUBTRACT"}),
            &json!({"Value": 1.0}),
        );
        link(&mut group, "tex", "Alpha", "less", "Value");
        link(&mut group, "less", "Value", "sub", "Value_001");
        link(&mut group, "sub", "Value", "bsdf", "Alpha");
        let (surface, losses) = resolved(&document(Some(group)));
        assert!(losses.is_empty(), "{losses:?}");
        assert_eq!(surface.alpha_mode, AlphaMode::Mask { cutoff: 0.5 });
    }

    #[test]
    fn transparent_mix_uses_fac_as_alpha() {
        let mut group = NodeGroup::new("graph", GraphKind::Shader);
        add_node(
            &mut group,
            "out",
            "OutputMaterial",
            &json!({"is_active_output": true}),
            &json!({}),
        );
        add_node(
            &mut group,
            "bsdf",
            "ShaderNodeBsdfPrincipled",
            &json!({}),
            &json!({"Base Color": [0.8, 0.1, 0.1, 1.0]}),
        );
        add_node(
            &mut group,
            "tex",
            "ShaderNodeTexImage",
            &json!({"image": "img"}),
            &json!({}),
        );
        add_node(
            &mut group,
            "clear",
            "ShaderNodeBsdfTransparent",
            &json!({}),
            &json!({}),
        );
        add_node(
            &mut group,
            "mix",
            "ShaderNodeMixShader",
            &json!({}),
            &json!({}),
        );
        link(&mut group, "tex", "Alpha", "mix", "Fac");
        link(&mut group, "clear", "BSDF", "mix", "Shader");
        link(&mut group, "bsdf", "BSDF", "mix", "Shader_001");
        link(&mut group, "mix", "Shader", "out", "Surface");
        let (surface, losses) = resolved(&document(Some(group)));
        assert!(losses.is_empty(), "{losses:?}");
        assert_eq!(surface.alpha_mode, AlphaMode::Blend);
        assert_eq!(surface.base_color.factor, [0.8, 0.1, 0.1, 1.0]);
        let texture = surface.base_color.texture.unwrap();
        assert_eq!(
            texture.channels[3],
            Some(ChannelSource {
                image: id("img"),
                channel: TextureChannel::Alpha,
            })
        );
    }

    #[test]
    fn native_scalar_fields_pack_red_into_gltf_channels() {
        let texture = TextureRef {
            image: id("img"),
            uv_layer: None,
            interpolation: ImageInterpolation::Closest,
            extension: crate::image::ImageExtension::Repeat,
        };
        let mut doc = document(None);
        let material = doc.materials.get_mut(&id("mat")).expect("material");
        material.roughness_texture = Some(texture.clone());
        material.metallic_texture = Some(texture);
        let (surface, losses) = resolved(&doc);
        assert!(losses.is_empty(), "{losses:?}");
        let packed = surface
            .metallic_roughness_texture
            .expect("native scalar textures pack");
        assert_eq!(
            packed.channels[1],
            Some(ChannelSource {
                image: id("img"),
                channel: TextureChannel::Red,
            })
        );
        assert_eq!(
            packed.channels[2],
            Some(ChannelSource {
                image: id("img"),
                channel: TextureChannel::Red,
            })
        );
    }
    #[test]
    fn separate_color_channels_pack_into_metallic_roughness() {
        let mut group = principled_graph(&json!({}));
        add_node(
            &mut group,
            "tex",
            "ShaderNodeTexImage",
            &json!({"image": "img"}),
            &json!({}),
        );
        add_node(
            &mut group,
            "sep",
            "ShaderNodeSeparateColor",
            &json!({"mode": "RGB"}),
            &json!({}),
        );
        link(&mut group, "tex", "Color", "sep", "Color");
        link(&mut group, "sep", "Green", "bsdf", "Roughness");
        link(&mut group, "sep", "Blue", "bsdf", "Metallic");
        let (surface, losses) = resolved(&document(Some(group)));
        assert!(losses.is_empty(), "{losses:?}");
        assert_eq!(surface.metallic, 1.0);
        assert_eq!(surface.roughness, 1.0);
        let packed = surface.metallic_roughness_texture.unwrap();
        assert_eq!(packed.channels[0], None);
        assert_eq!(
            packed.channels[1],
            Some(ChannelSource {
                image: id("img"),
                channel: TextureChannel::Green,
            })
        );
        assert_eq!(
            packed.channels[2],
            Some(ChannelSource {
                image: id("img"),
                channel: TextureChannel::Blue,
            })
        );
        assert_eq!(packed.channels[3], None);
    }

    #[test]
    fn vertex_color_multiplies_the_base_color_as_a_named_layer() {
        let mut group = principled_graph(&json!({}));
        add_node(
            &mut group,
            "paint",
            "ShaderNodeVertexColor",
            &json!({"layer_name": "Paint"}),
            &json!({}),
        );
        link(&mut group, "paint", "Color", "bsdf", "Base Color");
        let (surface, losses) = resolved(&document(Some(group)));
        assert!(losses.is_empty(), "{losses:?}");
        assert_eq!(
            surface.vertex_color,
            Some(VertexColorRead {
                layer: Some("Paint".to_owned())
            })
        );
        assert_eq!(surface.base_color.factor, [1.0; 4]);
        assert!(surface.base_color.texture.is_none());
    }

    #[test]
    fn empty_names_read_the_render_layer_and_geometry_attributes_read_their_name() {
        let mut group = principled_graph(&json!({}));
        add_node(
            &mut group,
            "paint",
            "ShaderNodeVertexColor",
            &json!({"layer_name": ""}),
            &json!({}),
        );
        link(&mut group, "paint", "Color", "bsdf", "Base Color");
        let (surface, _) = resolved(&document(Some(group)));
        assert_eq!(surface.vertex_color, Some(VertexColorRead { layer: None }));

        let mut group = principled_graph(&json!({}));
        add_node(
            &mut group,
            "paint",
            "ShaderNodeVertexColor",
            &json!({}),
            &json!({}),
        );
        link(&mut group, "paint", "Color", "bsdf", "Base Color");
        let (surface, _) = resolved(&document(Some(group)));
        assert_eq!(surface.vertex_color, Some(VertexColorRead { layer: None }));

        let mut group = principled_graph(&json!({}));
        add_node(
            &mut group,
            "wear",
            "ShaderNodeAttribute",
            &json!({"attribute_type": "GEOMETRY", "attribute_name": "Wear"}),
            &json!({}),
        );
        link(&mut group, "wear", "Color", "bsdf", "Base Color");
        let (surface, losses) = resolved(&document(Some(group)));
        assert!(losses.is_empty(), "{losses:?}");
        assert_eq!(
            surface.vertex_color,
            Some(VertexColorRead {
                layer: Some("Wear".to_owned())
            })
        );
        let mut group = principled_graph(&json!({}));
        add_node(
            &mut group,
            "empty_attribute",
            "ShaderNodeAttribute",
            &json!({"attribute_type": "GEOMETRY", "attribute_name": ""}),
            &json!({}),
        );
        link(&mut group, "empty_attribute", "Color", "bsdf", "Base Color");
        let (surface, _) = resolved(&document(Some(group)));
        assert_eq!(
            surface.vertex_color,
            Some(VertexColorRead {
                layer: Some(String::new())
            })
        );
    }

    #[test]
    fn vertex_color_times_texture_keeps_both_in_the_base_color() {
        let mut group = principled_graph(&json!({}));
        add_node(
            &mut group,
            "paint",
            "ShaderNodeVertexColor",
            &json!({"layer_name": "Paint"}),
            &json!({}),
        );
        add_node(
            &mut group,
            "tex",
            "ShaderNodeTexImage",
            &json!({"image": "img"}),
            &json!({}),
        );
        add_node(
            &mut group,
            "mix",
            "ShaderNodeMix",
            &json!({"blend_type": "MULTIPLY", "data_type": "RGBA"}),
            &json!({"Factor_Float": 1.0}),
        );
        link(&mut group, "paint", "Color", "mix", "A_Color");
        link(&mut group, "tex", "Color", "mix", "B_Color");
        link(&mut group, "mix", "Result_Color", "bsdf", "Base Color");
        let (surface, losses) = resolved(&document(Some(group)));
        assert!(losses.is_empty(), "{losses:?}");
        assert_eq!(
            surface.vertex_color,
            Some(VertexColorRead {
                layer: Some("Paint".to_owned())
            })
        );
        assert!(surface.base_color.texture.is_some());
    }

    #[test]
    fn vertex_color_outside_the_base_color_is_a_loss() {
        let mut group = principled_graph(&json!({}));
        add_node(
            &mut group,
            "paint",
            "ShaderNodeVertexColor",
            &json!({"layer_name": "Paint"}),
            &json!({}),
        );
        link(&mut group, "paint", "Color", "bsdf", "Emission Color");
        let (surface, losses) = resolved(&document(Some(group)));
        assert!(surface.vertex_color.is_none());
        assert!(
            losses
                .iter()
                .any(|loss| loss.feature_id == feature_id::VERTEX_COLOR),
            "{losses:?}"
        );
    }

    #[test]
    fn vertex_alpha_is_a_loss() {
        let mut group = principled_graph(&json!({}));
        add_node(
            &mut group,
            "paint",
            "ShaderNodeVertexColor",
            &json!({"layer_name": "Paint"}),
            &json!({}),
        );
        link(&mut group, "paint", "Alpha", "bsdf", "Alpha");
        let (surface, losses) = resolved(&document(Some(group)));
        assert!(surface.vertex_color.is_none());
        assert!(
            losses
                .iter()
                .any(|loss| loss.feature_id == feature_id::VERTEX_COLOR),
            "{losses:?}"
        );
    }

    #[test]
    fn principled_emission_is_color_times_strength() {
        let group = principled_graph(&json!({
            "Emission Color": [1.0, 0.5, 0.2, 1.0],
            "Emission Strength": 5.0,
        }));
        let (surface, losses) = resolved(&document(Some(group)));
        assert!(losses.is_empty(), "{losses:?}");
        assert!(close(surface.emissive.factor[0], 5.0));
        assert!(close(surface.emissive.factor[1], 2.5));
        assert!(close(surface.emissive.factor[2], 1.0));
        assert!(surface.emissive.texture.is_none());
    }

    #[test]
    fn background_strength_folds_into_unlit_color() {
        let mut group = NodeGroup::new("graph", GraphKind::Shader);
        add_node(
            &mut group,
            "out",
            "OutputMaterial",
            &json!({"is_active_output": true}),
            &json!({}),
        );
        add_node(
            &mut group,
            "bg",
            "ShaderNodeBackground",
            &json!({}),
            &json!({"Color": [0.5, 0.5, 0.5, 1.0], "Strength": 2.0}),
        );
        link(&mut group, "bg", "Background", "out", "Surface");
        let (surface, losses) = resolved(&document(Some(group)));
        assert!(losses.is_empty(), "{losses:?}");
        assert!(surface.unlit);
        assert_eq!(surface.base_color.factor, [1.0, 1.0, 1.0, 1.0]);
        assert!(close(surface.roughness, 0.9));
    }

    #[test]
    fn unsupported_texture_node_is_a_loss() {
        let mut group = principled_graph(&json!({}));
        add_node(
            &mut group,
            "noise",
            "ShaderNodeTexNoise",
            &json!({}),
            &json!({}),
        );
        link(&mut group, "noise", "Color", "bsdf", "Base Color");
        let (_, losses) = resolved(&document(Some(group)));
        let loss = losses
            .iter()
            .find(|loss| loss.feature_id == feature_id::UNSUPPORTED_NODE)
            .unwrap();
        assert_eq!(loss.data_id.as_deref(), Some("mat"));
        assert!(loss.reason.contains("noise"));
    }

    #[test]
    fn missing_image_is_an_error() {
        let mut group = principled_graph(&json!({}));
        add_node(
            &mut group,
            "tex",
            "ShaderNodeTexImage",
            &json!({"image": "missing"}),
            &json!({}),
        );
        link(&mut group, "tex", "Color", "bsdf", "Base Color");
        let error = resolve_surface(&document(Some(group)), &id("mat")).unwrap_err();
        assert_eq!(error.code, ErrorCode::SceneInvalid);
        assert_eq!(error.details["feature_id"], feature_id::INVALID);
    }

    #[test]
    fn material_texture_fields_are_losses_on_graph_materials() {
        let mut doc = document(Some(principled_graph(&json!({}))));
        doc.materials
            .get_mut(&id("mat"))
            .unwrap()
            .base_color_texture = Some(TextureRef {
            image: id("img"),
            uv_layer: None,
            interpolation: ImageInterpolation::Linear,
            extension: crate::image::ImageExtension::Repeat,
        });
        let (_, losses) = resolved(&doc);
        assert!(
            losses
                .iter()
                .any(|loss| loss.feature_id == feature_id::MATERIAL_TEXTURES)
        );
    }

    #[test]
    fn add_of_principled_and_emission_keeps_both() {
        let mut group = NodeGroup::new("graph", GraphKind::Shader);
        add_node(
            &mut group,
            "out",
            "OutputMaterial",
            &json!({"is_active_output": true}),
            &json!({}),
        );
        add_node(
            &mut group,
            "bsdf",
            "ShaderNodeBsdfPrincipled",
            &json!({}),
            &json!({"Base Color": [0.5, 0.5, 0.5, 1.0]}),
        );
        add_node(
            &mut group,
            "em",
            "ShaderNodeEmission",
            &json!({}),
            &json!({"Color": [1.0, 0.8, 0.4, 1.0], "Strength": 4.0}),
        );
        add_node(
            &mut group,
            "add",
            "ShaderNodeAddShader",
            &json!({}),
            &json!({}),
        );
        link(&mut group, "bsdf", "BSDF", "add", "Shader");
        link(&mut group, "em", "Emission", "add", "Shader_001");
        link(&mut group, "add", "Shader", "out", "Surface");
        let (surface, losses) = resolved(&document(Some(group)));
        assert!(losses.is_empty(), "{losses:?}");
        assert!(close(surface.emissive.factor[0], 4.0));
        assert!(close(surface.emissive.factor[1], 3.2));
        assert!(close(surface.emissive.factor[2], 1.6));
        assert_eq!(surface.base_color.factor, [0.5, 0.5, 0.5, 1.0]);
    }

    #[test]
    fn reroute_is_stepped_over_to_its_texture() {
        let mut group = principled_graph(&json!({}));
        add_node(
            &mut group,
            "tex",
            "ShaderNodeTexImage",
            &json!({"image": "img", "interpolation": "Linear", "extension": "REPEAT"}),
            &json!({}),
        );
        add_node(&mut group, "dot", "NodeReroute", &json!({}), &json!({}));
        link(&mut group, "tex", "Color", "dot", "Input");
        link(&mut group, "dot", "Output", "bsdf", "Base Color");
        let (surface, losses) = resolved(&document(Some(group)));
        assert!(losses.is_empty(), "{losses:?}");
        assert!(surface.base_color.texture.is_some());
    }

    #[test]
    fn unlinked_reroute_reads_as_unlinked() {
        let mut group = principled_graph(&json!({}));
        add_node(&mut group, "dot", "NodeReroute", &json!({}), &json!({}));
        link(&mut group, "dot", "Output", "bsdf", "Base Color");
        let (surface, losses) = resolved(&document(Some(group)));
        assert!(losses.is_empty(), "{losses:?}");
        assert!(surface.base_color.texture.is_none());
        assert_eq!(surface.base_color.factor[0], 0.8);
    }

    #[test]
    fn textured_specular_level_reads_raw_texture_and_moves_doubling_to_tint() {
        let mut group = principled_graph(&json!({"Specular IOR Level": 0.5}));
        add_node(
            &mut group,
            "tex",
            "ShaderNodeTexImage",
            &json!({"image": "img", "interpolation": "Linear", "extension": "REPEAT"}),
            &json!({}),
        );
        link(&mut group, "tex", "Alpha", "bsdf", "Specular IOR Level");
        let (surface, losses) = resolved(&document(Some(group)));
        assert!(losses.is_empty(), "{losses:?}");
        let specular = surface.specular.unwrap();
        assert!(close(specular.factor, 1.0));
        assert!(close(specular.color_factor[0], 2.0));
        assert!(specular.texture.is_some());
    }

    #[test]
    fn specular_factor_never_exceeds_one() {
        let group = principled_graph(&json!({"Specular IOR Level": 0.9}));
        let (surface, _) = resolved(&document(Some(group)));
        let specular = surface.specular.unwrap();
        assert!(close(specular.factor, 1.0));
        assert!(close(specular.color_factor[0], 1.8));
    }
}
