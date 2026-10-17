use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    fs,
    path::Path,
};

use super::material_graph;
use glam::{DMat4, DQuat, DVec3};
use serde_json::{Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    eval::{EvaluationPurpose, Snapshot, mesh_vertex_weights},
    geom::{
        Mesh, MeshError, attribute_keys, clnor,
        color_layers::{self, ColorDomain, ColorLayer, ColorStore},
        edge_key,
        normals::{close, corner_normals, sharp_edge_flags, smooth_face_flags},
        uv_layers::{self, UvLayer, UvStore},
    },
    image::{ImageData, ImageInterpolation, load_image_data},
    model::{
        Action, ArmatureData, AvatarSettings, AvatarVersion, Bone, CameraData, CameraProjection,
        DataBlock, Extrapolation, FCurve, Id, Image, ImageAlphaMode, ImageColorspace, ImageSource,
        Interpolation, Keyframe, LightData, LightType, Material, Modifier, Node, SceneDoc,
        ShapeKey, ShapeKeyData, TextureRef, Transform, VertexGroup,
    },
};

use super::{
    Loss, import_error, is_default_material_graph,
    material_surface::{
        AlphaMode, ChannelSource, ResolvedSurface, SampledTexture, TextureChannel,
        TextureTransform, TextureWrap, TexturedNormal, feature_id, material_is_used,
        resolve_fields_surface, resolve_surface,
    },
    shading,
};

const POT_TO_GLTF: DMat4 = DMat4::from_cols_array(&[
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
]);

fn pot_to_gltf_rotation() -> DQuat {
    DQuat::from_rotation_x(-std::f64::consts::FRAC_PI_2)
}

pub(crate) struct TextureExport {
    images: Vec<Value>,
    textures: Vec<Value>,
    samplers: Vec<Value>,
    image_indices: BTreeMap<Id, usize>,
    packed_image_indices: BTreeMap<String, usize>,
    sampler_indices: BTreeMap<(bool, u32, u32), usize>,
    texture_indices: BTreeMap<(usize, usize), usize>,
    uses_texture_transform: bool,
}

impl TextureExport {
    fn new() -> Self {
        Self {
            images: Vec::new(),
            textures: Vec::new(),
            samplers: Vec::new(),
            image_indices: BTreeMap::new(),
            packed_image_indices: BTreeMap::new(),
            sampler_indices: BTreeMap::new(),
            texture_indices: BTreeMap::new(),
            uses_texture_transform: false,
        }
    }

    /// Whether any texture info written so far uses `KHR_texture_transform`.
    pub(crate) fn uses_texture_transform(&self) -> bool {
        self.uses_texture_transform
    }

    /// Texture info for a sampled texture: `index`, `texCoord` (the TEXCOORD index its UV layer
    /// resolves to for the material, see [`TexCoords`]), and `KHR_texture_transform` when the texture
    /// has a transform. `reads` names the channels the slot samples; the rest are free to hold the
    /// image's own values.
    pub(crate) fn sampled_info(
        &mut self,
        texture: &SampledTexture,
        reads: [bool; 4],
        tex_coords: &TexCoords,
        doc: &SceneDoc,
        root_path: &Path,
        binary: &mut Vec<u8>,
        views: &mut Vec<Value>,
    ) -> Result<Value> {
        let mut filled = texture.clone();
        fill_unread(&mut filled.channels, reads);
        let index = self.sampled_index(&filled, doc, root_path, binary, views)?;
        let tex_coord = tex_coords.get(&texture.uv_layer).copied().unwrap_or(0);
        let mut info = json!({ "index": index, "texCoord": tex_coord });
        if let Some(transform) = &texture.transform {
            self.uses_texture_transform = true;
            info["extensions"] = json!({ "KHR_texture_transform": transform_json(transform) });
        }
        Ok(info)
    }

    /// glTF texture index for a sampled texture. Textures with the same source and sampler share one.
    pub(crate) fn sampled_index(
        &mut self,
        texture: &SampledTexture,
        doc: &SceneDoc,
        root_path: &Path,
        binary: &mut Vec<u8>,
        views: &mut Vec<Value>,
    ) -> Result<usize> {
        let image_index = match identity_image(&texture.channels) {
            Some(image) => self.export_image_index(
                image,
                texture.interpolation,
                doc,
                root_path,
                binary,
                views,
            )?,
            None => self.packed_image_index(texture, doc, root_path, binary, views)?,
        };
        let sampler_index = self.sampler_index(texture.interpolation, texture.wrap);
        if let Some(index) = self.texture_indices.get(&(image_index, sampler_index)) {
            return Ok(*index);
        }
        let index = self.textures.len();
        self.textures
            .push(json!({ "source": image_index, "sampler": sampler_index }));
        self.texture_indices
            .insert((image_index, sampler_index), index);
        Ok(index)
    }

    /// Sampler for one interpolation and wrap pair. Linear uses LINEAR(_`MIPMAP_LINEAR`) and Closest
    /// NEAREST(_`MIPMAP_NEAREST`), as Blender's exporter does.
    fn sampler_index(
        &mut self,
        interpolation: ImageInterpolation,
        wrap: [TextureWrap; 2],
    ) -> usize {
        let nearest = interpolation == ImageInterpolation::Closest;
        let key = (nearest, wrap_code(wrap[0]), wrap_code(wrap[1]));
        if let Some(index) = self.sampler_indices.get(&key) {
            return *index;
        }
        let (mag, min) = if nearest { (9728, 9984) } else { (9729, 9987) };
        let index = self.samplers.len();
        self.samplers.push(json!({
            "magFilter": mag,
            "minFilter": min,
            "wrapS": key.1,
            "wrapT": key.2
        }));
        self.sampler_indices.insert(key, index);
        index
    }

    /// Image index of a derived (packed) texture, encoded once per channel list.
    fn packed_image_index(
        &mut self,
        texture: &SampledTexture,
        doc: &SceneDoc,
        root_path: &Path,
        binary: &mut Vec<u8>,
        views: &mut Vec<Value>,
    ) -> Result<usize> {
        let key = format!("{:?}", texture.channels);
        if let Some(index) = self.packed_image_indices.get(&key) {
            return Ok(*index);
        }
        let packed = pack_channels(&texture.channels, texture.interpolation, doc, root_path)?;
        let encoded = encode_png_with_colorspace(ImageColorspace::Linear, &packed.pixels)?;
        align4(binary);
        let offset = binary.len();
        binary.extend_from_slice(&encoded);
        let view = push_view(views, offset, encoded.len(), None);
        let mut potter = json!({ "packed": packed.sources });
        // A grey copy of one source channel (VRM 0.x masks) also names that source and channel, so the
        // VRM importer can recover the original image; `remapped_channel` is the glTF component index.
        if let Some((image, component)) = Self::single_channel_copy(&texture.channels) {
            potter["id"] = json!(image.as_str());
            potter["remapped_channel"] = json!(component);
            // The copy's name is derived from its source; the importer restores that source name.
            potter["source_name"] = json!(texture_image_for_export(doc, image)?.name);
        }
        let index = self.images.len();
        self.images.push(json!({
            "name": packed.name,
            "bufferView": view,
            "mimeType": "image/png",
            "extras": { "potter": potter }
        }));
        self.packed_image_indices.insert(key, index);
        Ok(index)
    }

    /// The source image and glTF component (0 = R, 1 = G, 2 = B) when a texture's RGB all read one
    /// channel of one image and its alpha is constant: the grey copy the VRM masks use.
    fn single_channel_copy(channels: &[Option<ChannelSource>; 4]) -> Option<(&Id, usize)> {
        let [Some(red), Some(green), Some(blue), None] = channels else {
            return None;
        };
        let component = |source: &ChannelSource| match source.channel {
            TextureChannel::Red => 0,
            TextureChannel::Green => 1,
            TextureChannel::Blue => 2,
            TextureChannel::Alpha => 3,
        };
        let same = red.image == green.image
            && red.image == blue.image
            && red.channel == green.channel
            && red.channel == blue.channel;
        (same && component(red) < 3).then_some((&red.image, component(red)))
    }

    pub(crate) fn export_image_index(
        &mut self,
        image_id: &Id,
        interpolation: ImageInterpolation,
        doc: &SceneDoc,
        root_path: &Path,
        binary: &mut Vec<u8>,
        views: &mut Vec<Value>,
    ) -> Result<usize> {
        if let Some(index) = self.image_indices.get(image_id) {
            return Ok(*index);
        }
        let image = texture_image_for_export(doc, image_id)?;
        let pixels = load_image_data(image, root_path, interpolation)?;
        let encoded = encode_png(image, &pixels)?;
        align4(binary);
        let offset = binary.len();
        binary.extend_from_slice(&encoded);
        let view = push_view(views, offset, encoded.len(), None);
        let index = self.images.len();
        self.images.push(json!({
            "name": image.name,
            "bufferView": view,
            "mimeType": "image/png",
            "extras": { "potter": { "id": image_id } }
        }));
        self.image_indices.insert(image_id.clone(), index);
        Ok(index)
    }
}

/// The image a texture reads when its four channels are that image's R, G, B, A in order. Such a
/// texture reuses the image without re-encoding.
fn identity_image(channels: &[Option<ChannelSource>; 4]) -> Option<&Id> {
    let [Some(red), Some(green), Some(blue), Some(alpha)] = channels else {
        return None;
    };
    let same_image =
        red.image == green.image && red.image == blue.image && red.image == alpha.image;
    let in_order = red.channel == TextureChannel::Red
        && green.channel == TextureChannel::Green
        && blue.channel == TextureChannel::Blue
        && alpha.channel == TextureChannel::Alpha;
    (same_image && in_order).then_some(&red.image)
}

/// glTF channels (R, G, B, A) a texture slot samples. Unread channels may hold any value, so a
/// texture whose read channels already sit in place can reuse its image without re-encoding.
pub(crate) mod reads {
    pub(crate) const RGBA: [bool; 4] = [true, true, true, true];
    pub(crate) const RGB: [bool; 4] = [true, true, true, false];
    pub(crate) const R: [bool; 4] = [true, false, false, false];
    pub(crate) const G: [bool; 4] = [false, true, false, false];
    pub(crate) const A: [bool; 4] = [false, false, false, true];
    /// metallicRoughness: G roughness, B metallic.
    pub(crate) const MR: [bool; 4] = [false, true, true, false];
}

/// Channels a base color slot samples: alpha only when the material is not opaque (glTF ignores
/// `baseColorTexture` alpha under OPAQUE).
fn base_reads(surface: &ResolvedSurface) -> [bool; 4] {
    if matches!(surface.alpha_mode, AlphaMode::Opaque) {
        reads::RGB
    } else {
        reads::RGBA
    }
}

/// Fills unread channels that are constant 1.0 with the one image the texture reads, at their own
/// channel. glTF never samples those channels, so the output is unchanged; a texture whose read
/// channels already sit in place then reuses its image whole, as Blender's exporter does when it
/// fills only the channels it needs (`encode_image.py`, `__on_happy_path`).
fn fill_unread(channels: &mut [Option<ChannelSource>; 4], reads: [bool; 4]) {
    let mut sources = channels.iter().flatten().map(|source| &source.image);
    let Some(image) = sources.next().cloned() else {
        return;
    };
    if sources.any(|other| *other != image) {
        return;
    }
    let own = [
        TextureChannel::Red,
        TextureChannel::Green,
        TextureChannel::Blue,
        TextureChannel::Alpha,
    ];
    for (slot, read) in reads.iter().enumerate() {
        if !read && channels[slot].is_none() {
            channels[slot] = Some(ChannelSource {
                image: image.clone(),
                channel: own[slot],
            });
        }
    }
}

/// Whether a `blender_attributes` entry is a Blender color layer (`FLOAT_COLOR` or `BYTE_COLOR`).
pub(crate) fn is_color_attribute(entry: &Value) -> bool {
    matches!(
        entry["data_type"].as_str(),
        Some("FLOAT_COLOR" | "BYTE_COLOR")
    )
}

/// Whether a mesh attribute key stores one of the mesh's native colour layers. The writer exports
/// those as `COLOR_n`, so the key is neither a loss nor an unsupported attribute.
fn color_layer_storage_key(mesh: &Mesh, key: &str) -> bool {
    color_layers::read(mesh)
        .layers
        .iter()
        .any(|layer| attribute_keys::storage_key(&layer.name).as_ref() == key)
}

/// A derived glTF image: its pixels plus the Potter images it was packed from.
struct PackedImage {
    name: String,
    pixels: ImageData,
    sources: Vec<Id>,
}

/// Packs a texture's four glTF channels from their source channels (`None` is the constant 1.0).
///
/// Each source contributes its raw glTF value: sRGB color channels are re-encoded from the loaded
/// linear values. The output has the largest source's size; smaller or larger sources are sampled
/// nearest.
fn pack_channels(
    channels: &[Option<ChannelSource>; 4],
    interpolation: ImageInterpolation,
    doc: &SceneDoc,
    root_path: &Path,
) -> Result<PackedImage> {
    let mut sources = Vec::<Id>::new();
    for source in channels.iter().flatten() {
        if !sources.contains(&source.image) {
            sources.push(source.image.clone());
        }
    }
    let mut inputs = Vec::with_capacity(sources.len());
    for id in &sources {
        let image = texture_image_for_export(doc, id)?;
        inputs.push((image, load_image_data(image, root_path, interpolation)?));
    }
    let name = match inputs.first() {
        Some((image, _)) => format!("{}_packed", image.name),
        None => {
            return Err(PotError::new(
                ErrorCode::InternalError,
                "packed glTF texture has no channel source",
            ));
        }
    };
    let (width, height) = inputs.iter().fold((0_u32, 0_u32), |best, (_, input)| {
        let area = u64::from(input.width) * u64::from(input.height);
        if area > u64::from(best.0) * u64::from(best.1) {
            (input.width, input.height)
        } else {
            best
        }
    });
    let mut plan = Vec::with_capacity(channels.len());
    for channel in channels {
        let entry = match channel {
            None => None,
            Some(source) => {
                let input = sources
                    .iter()
                    .position(|id| *id == source.image)
                    .ok_or_else(|| {
                        PotError::new(ErrorCode::InternalError, "packed channel source is missing")
                    })?;
                let component: usize = match source.channel {
                    TextureChannel::Red => 0,
                    TextureChannel::Green => 1,
                    TextureChannel::Blue => 2,
                    TextureChannel::Alpha => 3,
                };
                let srgb = component < 3 && inputs[input].0.colorspace == ImageColorspace::Srgb;
                Some((input, component, srgb))
            }
        };
        plan.push(entry);
    }
    let mut pixels = Vec::with_capacity(width as usize * height as usize);
    for y in 0..height {
        for x in 0..width {
            let mut pixel = [1.0_f64; 4];
            for (output, source) in pixel.iter_mut().zip(&plan) {
                if let Some((input, component, srgb)) = *source {
                    let source_image = &inputs[input].1;
                    let sample_x = nearest_index(x, width, source_image.width);
                    let sample_y = nearest_index(y, height, source_image.height);
                    let raw = source_image
                        .pixels
                        .get(sample_y * source_image.width as usize + sample_x)
                        .ok_or_else(|| {
                            PotError::new(
                                ErrorCode::SceneInvalid,
                                "image pixel count does not match its dimensions",
                            )
                        })?[component];
                    *output = if srgb {
                        crate::color::linear_to_srgb_unclamped(raw)
                    } else {
                        raw
                    };
                }
            }
            pixels.push(pixel);
        }
    }
    Ok(PackedImage {
        name,
        pixels: ImageData {
            width,
            height,
            pixels,
            tiles: BTreeMap::new(),
            interpolation,
        },
        sources,
    })
}

/// Nearest source index along one axis for output sample `index` of `output` samples.
fn nearest_index(index: u32, output: u32, input: u32) -> usize {
    let scaled = (f64::from(index) + 0.5) / f64::from(output) * f64::from(input);
    (scaled.floor().max(0.0) as usize).min(input.saturating_sub(1) as usize)
}

fn transform_json(transform: &TextureTransform) -> Value {
    let mut value = serde_json::Map::new();
    if !crate::float::equal_f64_array(&transform.offset, &[0.0; 2]) {
        value.insert("offset".to_owned(), json!(transform.offset));
    }
    if !crate::float::equal_f64(transform.rotation, 0.0) {
        value.insert("rotation".to_owned(), json!(transform.rotation));
    }
    if !crate::float::equal_f64_array(&transform.scale, &[1.0; 2]) {
        value.insert("scale".to_owned(), json!(transform.scale));
    }
    Value::Object(value)
}

fn wrap_code(wrap: TextureWrap) -> u32 {
    match wrap {
        TextureWrap::Repeat => 10497,
        TextureWrap::ClampToEdge => 33071,
        TextureWrap::MirroredRepeat => 33648,
    }
}

fn encode_png(image: &Image, pixels: &ImageData) -> Result<Vec<u8>> {
    encode_png_with_colorspace(image.colorspace, pixels)
}

fn encode_png_with_colorspace(colorspace: ImageColorspace, pixels: &ImageData) -> Result<Vec<u8>> {
    let pixel_count = usize::try_from(pixels.width)
        .ok()
        .and_then(|width| {
            usize::try_from(pixels.height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, "image dimensions overflow"))?;
    if pixels.pixels.len() != pixel_count {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "image pixel count does not match its dimensions",
        ));
    }
    let mut raw = Vec::with_capacity(pixel_count.saturating_mul(4));
    for pixel in &pixels.pixels {
        for (index, component) in pixel.iter().copied().enumerate() {
            if !component.is_finite() {
                return Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    "image contains a non-finite pixel",
                ));
            }
            let component = if index < 3 && colorspace == ImageColorspace::Srgb {
                crate::color::linear_to_srgb_unclamped(component)
            } else {
                component
            };
            raw.push((component.clamp(0.0, 1.0) * 255.0).round() as u8);
        }
    }
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, pixels.width, pixels.height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().map_err(|error| {
            PotError::new(
                ErrorCode::ExportFailed,
                format!("PNG texture header encoding failed: {error}"),
            )
        })?;
        writer.write_image_data(&raw).map_err(|error| {
            PotError::new(
                ErrorCode::ExportFailed,
                format!("PNG texture encoding failed: {error}"),
            )
        })?;
        writer.finish().map_err(|error| {
            PotError::new(
                ErrorCode::ExportFailed,
                format!("PNG texture finalization failed: {error}"),
            )
        })?;
    }
    Ok(bytes)
}

pub(crate) fn unsupported_export(feature_id: &str, reason: &str) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        reason,
        json!({ "feature_id": feature_id, "status": "not_supported", "reason": reason }),
    )
}

fn texture_image_for_export<'a>(doc: &'a SceneDoc, image_id: &Id) -> Result<&'a Image> {
    let image = doc.images.get(image_id).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::SceneInvalid,
            "material texture references a missing image",
            json!({ "image": image_id }),
        )
    })?;
    if !image.tiles.is_empty() {
        return Err(unsupported_export(
            "material.texture_udim",
            "glTF export does not represent Potter UDIM image tiles",
        ));
    }
    Ok(image)
}

#[derive(Clone)]
struct SkinBinding {
    armature_node: Id,
    joint_ids: Vec<Id>,
}

fn skin_binding(
    doc: &SceneDoc,
    node: &Node,
    purpose: EvaluationPurpose,
) -> Result<Option<SkinBinding>> {
    let modifiers = node
        .modifiers
        .iter()
        .filter(|modifier| modifier.is_active(purpose) && modifier.modifier_type == "armature")
        .collect::<Vec<_>>();
    if modifiers.is_empty() {
        return Ok(None);
    }
    if modifiers.len() != 1
        || node
            .modifiers
            .iter()
            .any(|modifier| modifier.is_active(purpose) && modifier.modifier_type != "armature")
    {
        return Err(unsupported_export(
            "rig.skin_modifier_stack",
            "glTF skin export requires exactly one enabled armature modifier and no other enabled modifiers",
        ));
    }
    let target = modifiers[0]
        .params
        .get("object")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::SceneInvalid,
                "armature modifier has no armature target",
            )
        })?;
    let armature_node = Id::new(target).map_err(|_| {
        PotError::new(
            ErrorCode::SceneInvalid,
            "armature modifier target is not a valid ID",
        )
    })?;
    let armature_object = doc.nodes.get(&armature_node).ok_or_else(|| {
        PotError::new(
            ErrorCode::SceneInvalid,
            "armature modifier target does not exist",
        )
    })?;
    let armature_data_id = armature_object.data.as_ref().ok_or_else(|| {
        PotError::new(ErrorCode::SceneInvalid, "armature object has no Data-Block")
    })?;
    let armature_data = doc.data_blocks.get(armature_data_id).ok_or_else(|| {
        PotError::new(
            ErrorCode::SceneInvalid,
            "armature Data-Block does not exist",
        )
    })?;
    let armature = armature_data.armature.as_ref().ok_or_else(|| {
        PotError::new(
            ErrorCode::SceneInvalid,
            "armature object Data-Block is not an armature",
        )
    })?;
    Ok(Some(SkinBinding {
        armature_node,
        joint_ids: armature.bones.keys().cloned().collect(),
    }))
}
fn validate_skin_shape_key_rest(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    binding: &SkinBinding,
    rest_world_cache: &mut BTreeMap<Id, BTreeMap<Id, DMat4>>,
) -> Result<()> {
    if !rest_world_cache.contains_key(&binding.armature_node) {
        let armature = doc
            .nodes
            .get(&binding.armature_node)
            .and_then(|node| node.data.as_ref())
            .and_then(|data_id| doc.data_blocks.get(data_id))
            .and_then(|data| data.armature.as_ref())
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::SceneInvalid,
                    "shape-key skin armature data is missing",
                )
            })?;
        let armature_world = snapshot
            .nodes
            .get(&binding.armature_node)
            .map(|node| DMat4::from_cols_array(&node.world_matrix))
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::EvaluationFailed,
                    "shape-key skin armature world transform is missing",
                )
            })?;
        let rest = crate::eval::rig::evaluate_bone_matrices(armature, &BTreeMap::new())?;
        let rest_world = rest
            .into_iter()
            .map(|(bone_id, matrices)| (bone_id, armature_world * matrices.rest))
            .collect();
        rest_world_cache.insert(binding.armature_node.clone(), rest_world);
    }
    let expected = &rest_world_cache[&binding.armature_node];
    let actual = snapshot
        .bone_matrices
        .get(&binding.armature_node)
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::EvaluationFailed,
                "shape-key skin evaluated bones are missing",
            )
        })?;
    for (bone_id, expected_world) in expected {
        let actual_world = actual
            .get(bone_id)
            .map(DMat4::from_cols_array)
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::EvaluationFailed,
                    "shape-key skin evaluated bone is missing",
                )
            })?;
        let at_rest = expected_world
            .to_cols_array()
            .into_iter()
            .zip(actual_world.to_cols_array())
            .all(|(expected, actual)| {
                (expected - actual).abs() <= 1.0e-5 * expected.abs().max(actual.abs()).max(1.0)
            });
        if !at_rest {
            return Err(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                "glTF shape-key skin export requires evaluated bones to match the armature rest pose.",
                json!({
                    "feature_id": "rig.skin_shape_key_pose",
                    "status": "not_supported",
                    "armature": binding.armature_node,
                    "bone": bone_id
                }),
            ));
        }
    }
    Ok(())
}

/// Vertex-group weights keyed by `mesh`'s vertex IDs. Shape-key exports pass the source mesh, whose
/// IDs key `DataBlock.vertex_weights`; every other export passes the evaluated mesh, whose remapped
/// `vertex_groups` attribute is the only valid source.
fn export_vertex_weights<'a>(
    mesh: &Mesh,
    data_block: &'a DataBlock,
) -> Cow<'a, BTreeMap<u32, BTreeMap<Id, f64>>> {
    let shape_keyed = data_block
        .shape_keys
        .as_ref()
        .is_some_and(|shape_keys| !shape_keys.keys.is_empty());
    if shape_keyed {
        Cow::Borrowed(&data_block.vertex_weights)
    } else {
        Cow::Owned(mesh_vertex_weights(mesh, &data_block.vertex_groups))
    }
}

/// Deform-bone influences per exported vertex as (joint index, weight), sorted by joint index.
/// The glTF writer and VRM preflight share this so `validate --format vrm|vrm0` reports the same
/// failures: every skinned vertex needs weight, and VRM also caps influences at four.
fn skin_influences(
    node_id: &Id,
    data_block: &DataBlock,
    mesh: &Mesh,
    source_vertex_indices: &[usize],
    binding: &SkinBinding,
    doc: &SceneDoc,
    vrm: bool,
) -> Result<Vec<Vec<(usize, f64)>>> {
    let armature_node = doc
        .nodes
        .get(&binding.armature_node)
        .ok_or_else(|| PotError::new(ErrorCode::SceneInvalid, "skin armature node is missing"))?;
    let armature_data_id = armature_node.data.as_ref().ok_or_else(|| {
        PotError::new(
            ErrorCode::SceneInvalid,
            "skin armature Data-Block is missing",
        )
    })?;
    let armature = doc
        .data_blocks
        .get(armature_data_id)
        .and_then(|data| data.armature.as_ref())
        .ok_or_else(|| PotError::new(ErrorCode::SceneInvalid, "skin armature data is missing"))?;
    let joint_indices = binding
        .joint_ids
        .iter()
        .enumerate()
        .map(|(index, id)| (id.clone(), index))
        .collect::<BTreeMap<_, _>>();
    let mut group_bones = BTreeMap::<Id, usize>::new();
    for group in &data_block.vertex_groups {
        if let Some((bone_id, _)) = armature
            .bones
            .iter()
            .find(|(_, bone)| bone.deform && bone.name == group.name)
            && let Some(joint_index) = joint_indices.get(bone_id)
        {
            group_bones.insert(group.id.clone(), *joint_index);
        }
    }
    let group_ids = data_block
        .vertex_groups
        .iter()
        .map(|group| group.id.clone())
        .collect::<BTreeSet<_>>();
    let mut vertex_influences =
        Vec::<Vec<(usize, f64)>>::with_capacity(source_vertex_indices.len());
    // Keyed by source vertex: one source vertex can appear in several split output vertices.
    let mut unweighted = BTreeSet::<u32>::new();
    let mut over_limit = BTreeMap::<u32, usize>::new();
    let weights_by_vertex = export_vertex_weights(mesh, data_block);
    for source_index in source_vertex_indices {
        let vertex = mesh.vertices.get(*source_index).ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "glTF source vertex index is invalid",
            )
        })?;
        let mut influences = Vec::<(usize, f64)>::new();
        if let Some(vertex_weights) = weights_by_vertex.get(&vertex.id) {
            for (group_id, weight) in vertex_weights {
                if !group_ids.contains(group_id) {
                    return Err(PotError::new(
                        ErrorCode::SceneInvalid,
                        "skin weights reference a missing vertex group",
                    ));
                }
                if !weight.is_finite() || *weight < 0.0 {
                    return Err(PotError::new(
                        ErrorCode::SceneInvalid,
                        "skin weight is non-finite or negative",
                    ));
                }
                if *weight > 0.0
                    && let Some(joint_index) = group_bones.get(group_id)
                {
                    influences.push((*joint_index, *weight));
                }
            }
        }
        influences.sort_by_key(|(joint_index, _)| *joint_index);
        if influences.is_empty() {
            unweighted.insert(vertex.id);
        }
        if vrm && influences.len() > 4 {
            over_limit.insert(vertex.id, influences.len());
        }
        vertex_influences.push(influences);
    }
    if let Some(first) = unweighted.first() {
        let reason = format!(
            "skinned mesh `{node_id}` has {} vertices with no weight on a deform bone (first vertex {first}); weight every vertex to a deform bone or remove the armature modifier",
            unweighted.len()
        );
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            &reason,
            json!({
                "feature_id": "mesh.skin_unweighted_vertex",
                "status": "not_supported",
                "reason": &reason,
                "node": node_id,
                "count": unweighted.len(),
                "first_vertex": first,
            }),
        ));
    }
    if let Some((first, _)) = over_limit.first_key_value() {
        let max_influences = over_limit.values().copied().max().unwrap_or_default();
        let reason = format!(
            "VRM skins allow at most 4 bone influences per vertex; skinned mesh `{node_id}` has {} vertices with up to {max_influences} (first vertex {first}); limit each vertex to its 4 strongest deform-bone weights",
            over_limit.len()
        );
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            &reason,
            json!({
                "feature_id": "vrm.skin_influence_limit",
                "status": "not_supported",
                "reason": &reason,
                "node": node_id,
                "count": over_limit.len(),
                "first_vertex": first,
                "max_influences": max_influences,
            }),
        ));
    }
    Ok(vertex_influences)
}

fn append_skin_attributes(
    vertex_influences: &[Vec<(usize, f64)>],
    binary: &mut Vec<u8>,
    views: &mut Vec<Value>,
    accessors: &mut Vec<Value>,
) -> Result<Vec<(usize, usize)>> {
    let max_influences = vertex_influences.iter().map(Vec::len).max().unwrap_or(0);
    let set_count = max_influences.div_ceil(4).max(1);
    let mut joint_sets = (0..set_count)
        .map(|_| Vec::<[u16; 4]>::with_capacity(vertex_influences.len()))
        .collect::<Vec<_>>();
    let mut weight_sets = (0..set_count)
        .map(|_| Vec::<[f32; 4]>::with_capacity(vertex_influences.len()))
        .collect::<Vec<_>>();
    for influences in vertex_influences {
        let total = influences.iter().map(|(_, weight)| *weight).sum::<f64>();
        if !total.is_finite() {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "sum of skin weights is non-finite",
            ));
        }
        for set_index in 0..set_count {
            let mut joint_values = [0_u16; 4];
            let mut weight_values = [0.0_f32; 4];
            for slot in 0..4 {
                if let Some((joint_index, weight)) = influences.get(set_index * 4 + slot).copied() {
                    joint_values[slot] = u16::try_from(joint_index).map_err(|_| {
                        PotError::new(ErrorCode::LimitExceeded, "glTF skin has too many joints")
                    })?;
                    weight_values[slot] = to_f32(weight / total)?;
                }
            }
            joint_sets[set_index].push(joint_values);
            weight_sets[set_index].push(weight_values);
        }
    }
    let mut accessors_for_sets = Vec::with_capacity(set_count);
    for (joints, weights) in joint_sets.iter().zip(&weight_sets) {
        align4(binary);
        let joints_offset = binary.len();
        for joint in joints {
            for component in joint {
                binary.extend_from_slice(&component.to_le_bytes());
            }
        }
        let joints_view = push_view(
            views,
            joints_offset,
            binary.len() - joints_offset,
            Some(34962),
        );
        let joints_accessor = accessors.len();
        accessors.push(json!({
            "bufferView": joints_view,
            "componentType": 5123,
            "count": joints.len(),
            "type": "VEC4"
        }));
        align4(binary);
        let weights_offset = binary.len();
        for weight in weights {
            for component in weight {
                binary.extend_from_slice(&component.to_le_bytes());
            }
        }
        let weights_view = push_view(
            views,
            weights_offset,
            binary.len() - weights_offset,
            Some(34962),
        );
        let weights_accessor = accessors.len();
        accessors.push(json!({
            "bufferView": weights_view,
            "componentType": 5126,
            "count": weights.len(),
            "type": "VEC4"
        }));
        accessors_for_sets.push((joints_accessor, weights_accessor));
    }
    Ok(accessors_for_sets)
}

/// Values of one logical layer assigned to a dense `COLOR_n` accessor: one per face corner (mesh
/// face order, then corner order), and one per vertex, which loose vertices (no corner) use.
struct CarriedColor {
    corners: Vec<[f64; 4]>,
    vertices: Vec<[f64; 4]>,
}

/// The value a corner-domain layer gives a loose vertex, which has no corner to read.
const LOOSE_VERTEX_COLOR: [f64; 4] = [0.0, 0.0, 0.0, 1.0];

impl CarriedColor {
    fn new(mesh: &Mesh, layer: &ColorLayer) -> Result<Self> {
        match layer.domain {
            ColorDomain::Corner => Ok(Self {
                corners: layer.values.clone(),
                vertices: vec![LOOSE_VERTEX_COLOR; mesh.vertices.len()],
            }),
            ColorDomain::Point => {
                let vertex_indices = mesh
                    .vertices
                    .iter()
                    .enumerate()
                    .map(|(index, vertex)| (vertex.id, index))
                    .collect::<BTreeMap<_, _>>();
                let mut corners = Vec::new();
                for face in &mesh.faces {
                    for vertex_id in &face.vertices {
                        let value = vertex_indices
                            .get(vertex_id)
                            .and_then(|index| layer.values.get(*index))
                            .copied()
                            .ok_or_else(|| {
                                PotError::new(
                                    ErrorCode::SceneInvalid,
                                    format!(
                                        "color layer `{}` has no value for a face vertex",
                                        layer.name
                                    ),
                                )
                            })?;
                        corners.push(value);
                    }
                }
                Ok(Self {
                    corners,
                    vertices: layer.values.clone(),
                })
            }
        }
    }
}
/// The canonical color store in Potter extras, keyed to the source mesh's persistent element IDs.
fn potter_color_attributes(mesh: &Mesh) -> Value {
    let store = color_layers::read(mesh);
    let layers = store
        .layers
        .iter()
        .map(|layer| {
            let values = match layer.domain {
                ColorDomain::Point => json!(layer.values),
                ColorDomain::Corner => {
                    let mut offset = 0;
                    let rows = mesh
                        .faces
                        .iter()
                        .map(|face| {
                            let end = offset + face.vertices.len();
                            let row = &layer.values[offset..end];
                            offset = end;
                            json!(row)
                        })
                        .collect::<Vec<_>>();
                    json!(rows)
                }
            };
            json!({
                "name": layer.name,
                "domain": layer.domain.name(),
                "values": values
            })
        })
        .collect::<Vec<_>>();
    json!({
        "version": 1,
        "active": store.active,
        "render": store.render,
        "vertex_ids": mesh.vertices.iter().map(|vertex| vertex.id).collect::<Vec<_>>(),
        "face_ids": mesh.faces.iter().map(|face| face.id).collect::<Vec<_>>(),
        "layers": layers
    })
}

/// Adds the selected color layer without replacing other primitive extras.
fn put_potter_primitive_color_layer(primitive: &mut Value, name: &str) -> Result<()> {
    let extras = primitive
        .as_object_mut()
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "glTF primitive is not an object"))?
        .entry("extras")
        .or_insert_with(|| json!({}));
    let extras = extras.as_object_mut().ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "glTF primitive extras are not an object",
        )
    })?;
    let potter = extras.entry("potter").or_insert_with(|| json!({}));
    let potter = potter.as_object_mut().ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "glTF primitive Potter extras are not an object",
        )
    })?;
    potter.insert("color_layer".to_owned(), json!(name));
    Ok(())
}

/// How a mesh's primitives read their color layers. A colored primitive has its selected layer as
/// `COLOR_0`; following layers use dense `COLOR_1` … `COLOR_n`. Potter extras preserve the
/// logical names and domains, including layers on colorless primitives.
#[derive(Default)]
struct ColorPlan {
    /// Store index of the layer each material slot holds as `COLOR_0`.
    slot_layers: BTreeMap<u32, usize>,
    /// Source-member layers available to each colored primitive slot.
    slot_available_layers: BTreeMap<u32, BTreeSet<usize>>,
    /// Material slots the mesh's faces use, ascending.
    used_slots: Vec<u32>,
    /// Slots whose material reads a layer the mesh does not have. `None` names the render layer.
    missing: BTreeMap<u32, Option<String>>,
    /// Slots whose translucent material reads a layer; glTF multiplies that layer's alpha into the
    /// base alpha, which Blender ignores.
    translucent: BTreeSet<u32>,
}

impl ColorPlan {
    /// Dense `COLOR_n` mappings for a primitive: selected layer first, then its other available layers.
    fn attributes(&self, slot: u32) -> Vec<(usize, usize)> {
        let Some(selected) = self.slot_layers.get(&slot).copied() else {
            return Vec::new();
        };
        let mut attributes = vec![(0, selected)];
        if let Some(available) = self.slot_available_layers.get(&slot) {
            for &layer in available {
                if layer != selected {
                    attributes.push((attributes.len(), layer));
                }
            }
        }
        attributes
    }

    /// Unique logical layers actually written by at least one colored primitive.
    fn carried_layers(&self) -> Vec<usize> {
        self.slot_available_layers
            .values()
            .flat_map(|layers| layers.iter().copied())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }
}

fn color_layer_available(binding: Option<&crate::eval::SlotBinding>, layer: usize) -> bool {
    binding.is_none_or(|binding| {
        binding
            .color_available_mask
            .get(layer / u64::BITS as usize)
            .is_some_and(|word| word & (1_u64 << (layer % u64::BITS as usize)) != 0)
    })
}

fn color_plan(
    node_id: &Id,
    node: &Node,
    snapshot: &Snapshot,
    data: Option<&DataBlock>,
    mesh: &Mesh,
    surfaces: &BTreeMap<Id, ResolvedSurface>,
) -> ColorPlan {
    let store = color_layers::read(mesh);
    let mut plan = ColorPlan {
        used_slots: mesh
            .faces
            .iter()
            .map(|face| face.material_index)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect(),
        ..ColorPlan::default()
    };
    for &slot in &plan.used_slots {
        let Some(slot_index) = usize::try_from(slot).ok() else {
            continue;
        };
        let binding = snapshot.slot_binding(node_id, node, slot_index);
        let Some(material_id) = snapshot.slot_material(node_id, node, data, slot_index) else {
            continue;
        };
        let surface = surfaces.get(material_id);
        let Some(reference) = surface
            .and_then(|surface| surface.vertex_color.as_ref())
            .map(|read| read.layer.clone())
        else {
            continue;
        };
        let index = match reference.as_deref() {
            Some(name) => store.index_of(name),
            None => binding
                .and_then(|binding| binding.color_render_layer)
                .or_else(|| binding.is_none().then(|| store.render_index()).flatten()),
        };
        let available = (0..store.layers.len())
            .filter(|layer| color_layer_available(binding, *layer))
            .collect::<BTreeSet<_>>();
        match index.filter(|index| available.contains(index)) {
            Some(index) => {
                plan.slot_layers.insert(slot, index);
                plan.slot_available_layers.insert(slot, available);
                if surface.is_some_and(|surface| !matches!(surface.alpha_mode, AlphaMode::Opaque)) {
                    plan.translucent.insert(slot);
                }
            }
            None => {
                plan.missing.insert(slot, reference);
            }
        }
    }
    plan
}

/// glTF vertices after splitting by source vertex, every UV set, carried color layers, and every
/// shading layer's normal.
struct ExportVertexLayout {
    source_vertex_indices: Vec<usize>,
    /// One entry per UV set (`TEXCOORD_n`), each with one coordinate per output vertex.
    uvs: Vec<Vec<[f64; 2]>>,
    /// Layer zero is base shading; each later layer is one shape key's morphed shading.
    normals: Vec<Vec<DVec3>>,
    /// One entry per carried color layer, with one linear RGBA value per output vertex.
    colors: Vec<Vec<[f64; 4]>>,
    face_indices: BTreeMap<u32, BTreeMap<u32, u32>>,
}

fn export_vertex_layout(
    mesh: &Mesh,
    layers: &[Vec<DVec3>],
    fallback_normal: DVec3,
    colors: &[CarriedColor],
) -> Result<ExportVertexLayout> {
    let uv_sets = uv_layers::read_layers(mesh);
    let vertex_indices = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<BTreeMap<_, _>>();
    let corner_count = mesh
        .faces
        .iter()
        .map(|face| face.vertices.len())
        .sum::<usize>();
    if layers.iter().any(|layer| layer.len() != corner_count)
        || colors
            .iter()
            .any(|color| color.corners.len() != corner_count)
    {
        return Err(PotError::new(
            ErrorCode::InternalError,
            "glTF corner normals or colors do not match mesh corners",
        ));
    }
    // Corners grouped by source vertex; corner indices follow mesh.faces order then face corner order.
    let mut corners_by_vertex =
        vec![Vec::<(u32, u32, usize, Vec<[f64; 2]>)>::new(); mesh.vertices.len()];
    let mut corner = 0;
    for (face_index, face) in mesh.faces.iter().enumerate() {
        for (position, vertex_id) in face.vertices.iter().enumerate() {
            let source_index = vertex_indices.get(vertex_id).copied().ok_or_else(|| {
                PotError::new(
                    ErrorCode::SceneInvalid,
                    "UV face references a missing vertex",
                )
            })?;
            let uvs: Vec<[f64; 2]> = uv_sets
                .iter()
                .map(|layer| {
                    layer
                        .faces
                        .get(face_index)
                        .and_then(Option::as_ref)
                        .and_then(|values| values.get(position))
                        .copied()
                        .unwrap_or([0.0, 1.0])
                })
                .collect();
            corners_by_vertex[source_index].push((face.id, *vertex_id, corner, uvs));
            corner += 1;
        }
    }
    let mut source_vertex_indices = Vec::<usize>::new();
    let mut uvs = vec![Vec::<[f64; 2]>::new(); uv_sets.len()];
    let mut normals = vec![Vec::<DVec3>::new(); layers.len()];
    let mut colors_out = vec![Vec::<[f64; 4]>::new(); colors.len()];
    let mut outputs = BTreeMap::<(usize, Vec<[u64; 2]>, Vec<[u64; 3]>, Vec<[u64; 4]>), u32>::new();
    let mut face_indices = BTreeMap::<u32, BTreeMap<u32, u32>>::new();
    for (source_index, corners) in corners_by_vertex.iter().enumerate() {
        if corners.is_empty() {
            // Loose vertices keep an output slot so positions and vertex IDs stay aligned.
            source_vertex_indices.push(source_index);
            for output in &mut uvs {
                output.push([0.0, 0.0]);
            }
            for output in &mut normals {
                output.push(fallback_normal);
            }
            for (output, color) in colors_out.iter_mut().zip(colors) {
                output.push(color.vertices[source_index]);
            }
            continue;
        }
        for (face_id, vertex_id, corner, corner_uvs) in corners {
            // Normal and color bits are normalized so -0.0 and 0.0 share one vertex.
            let key = (
                source_index,
                corner_uvs
                    .iter()
                    .map(|uv| uv.map(|value| (value + 0.0).to_bits()))
                    .collect::<Vec<_>>(),
                layers
                    .iter()
                    .map(|layer| {
                        layer[*corner]
                            .to_array()
                            .map(|value| (value + 0.0).to_bits())
                    })
                    .collect::<Vec<_>>(),
                colors
                    .iter()
                    .map(|color| color.corners[*corner].map(|value| (value + 0.0).to_bits()))
                    .collect::<Vec<_>>(),
            );
            let index = if let Some(index) = outputs.get(&key) {
                *index
            } else {
                let index = u32::try_from(source_vertex_indices.len()).map_err(|_| {
                    PotError::new(ErrorCode::LimitExceeded, "glTF mesh has too many vertices")
                })?;
                source_vertex_indices.push(source_index);
                for (output, uv) in uvs.iter_mut().zip(corner_uvs) {
                    output.push(*uv);
                }
                for (output, layer) in normals.iter_mut().zip(layers) {
                    output.push(layer[*corner]);
                }
                for (output, color) in colors_out.iter_mut().zip(colors) {
                    output.push(color.corners[*corner]);
                }
                outputs.insert(key, index);
                index
            };
            face_indices
                .entry(*face_id)
                .or_default()
                .insert(*vertex_id, index);
        }
    }
    Ok(ExportVertexLayout {
        source_vertex_indices,
        uvs,
        normals,
        colors: colors_out,
        face_indices,
    })
}

/// Triangles for glTF export, `len - 2` per face in `mesh.faces` order. A face without area has no
/// ear to clip, so it fans into degenerate triangles, as Blender's glTF exporter writes it.
fn export_triangles(mesh: &Mesh) -> std::result::Result<Vec<[u32; 3]>, MeshError> {
    if let Ok(triangles) = mesh.triangulate() {
        return Ok(triangles);
    }
    let mut subset = mesh.clone();
    let mut degenerate = BTreeSet::<u32>::new();
    let triangles = loop {
        match subset.triangulate() {
            Ok(triangles) => break triangles,
            Err(MeshError::TriangulationFailed(id))
                if subset.faces.iter().any(|face| face.id == id) =>
            {
                degenerate.insert(id);
                subset.faces.retain(|face| face.id != id);
            }
            Err(error) => return Err(error),
        }
    };
    let mut regular = triangles.into_iter();
    let mut output = Vec::new();
    for face in &mesh.faces {
        if degenerate.contains(&face.id) {
            for pair in face.vertices[1..].windows(2) {
                output.push([face.vertices[0], pair[0], pair[1]]);
            }
        } else {
            output.extend(regular.by_ref().take(face.vertices.len().saturating_sub(2)));
        }
    }
    Ok(output)
}

/// Bytes per element of a packed `extras.potter.polygons` field whose values stay at or below
/// `limit`. The reader derives the same width from the same limit, so the record needs no type tags.
fn packed_width(limit: usize) -> usize {
    if limit <= 0xff {
        1
    } else if limit <= 0xffff {
        2
    } else {
        4
    }
}

/// Appends `values` as tightly packed little-endian unsigned ints and returns their bufferView index.
fn append_packed(
    binary: &mut Vec<u8>,
    views: &mut Vec<Value>,
    values: &[u32],
    width: usize,
) -> usize {
    align4(binary);
    let offset = binary.len();
    for value in values {
        binary.extend_from_slice(&value.to_le_bytes()[..width]);
    }
    push_view(views, offset, binary.len() - offset, None)
}

/// `extras.potter.polygons`, stored as bufferViews of packed unsigned ints: `sizes` (corners per
/// face), `corners` (global glTF vertex indices, face corner order, mirrored bakes reversed whole),
/// `primitives` (the primitive holding each face's triangles), `smooth` (one byte per face) and
/// `sharp_edges` (glTF vertex pairs, one per sharp edge; absent when none). Faces are in
/// `mesh.faces` order and `face_primitives` gives each face's primitive index.
fn export_polygons(
    mesh: &Mesh,
    layout: &ExportVertexLayout,
    mirrored: bool,
    face_primitives: &[u32],
    primitive_count: usize,
    binary: &mut Vec<u8>,
    views: &mut Vec<Value>,
) -> Result<Value> {
    let sharp_keys = mesh
        .edges
        .iter()
        .zip(sharp_edge_flags(mesh)?)
        .filter(|(_, sharp)| *sharp)
        .map(|(edge, _)| edge_key(edge.vertices[0], edge.vertices[1]))
        .collect::<BTreeSet<_>>();
    let mut emitted = BTreeSet::new();
    let mut sharp_pairs = Vec::<u32>::new();
    let mut sizes = Vec::with_capacity(mesh.faces.len());
    let mut corners_flat = Vec::<u32>::new();
    let mut smooth = Vec::with_capacity(mesh.faces.len());
    for (face, smooth_flag) in mesh.faces.iter().zip(smooth_face_flags(mesh)?) {
        let indices = layout.face_indices.get(&face.id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "face has no output vertices")
        })?;
        let mut corners = face
            .vertices
            .iter()
            .map(|id| {
                indices.get(id).copied().ok_or_else(|| {
                    PotError::new(
                        ErrorCode::SceneInvalid,
                        "polygon references missing output vertex",
                    )
                })
            })
            .collect::<Result<Vec<u32>>>()?;
        let count = corners.len();
        for corner in 0..count {
            let next = (corner + 1) % count;
            let pair = edge_key(face.vertices[corner], face.vertices[next]);
            if sharp_keys.contains(&pair) && emitted.insert(pair) {
                sharp_pairs.extend([corners[corner], corners[next]]);
            }
        }
        if mirrored {
            corners.reverse();
        }
        sizes.push(
            u32::try_from(count).map_err(|_| {
                PotError::new(ErrorCode::SceneInvalid, "polygon has too many corners")
            })?,
        );
        corners_flat.extend(corners);
        smooth.push(u32::from(smooth_flag));
    }
    let vertex_limit = layout.source_vertex_indices.len();
    let sizes_view = append_packed(binary, views, &sizes, packed_width(vertex_limit));
    let corners_view = append_packed(
        binary,
        views,
        &corners_flat,
        packed_width(vertex_limit.saturating_sub(1)),
    );
    let primitives_view = append_packed(
        binary,
        views,
        face_primitives,
        packed_width(primitive_count.saturating_sub(1)),
    );
    let smooth_view = append_packed(binary, views, &smooth, 1);
    let mut record = json!({
        "version": 1,
        "sizes": sizes_view,
        "corners": corners_view,
        "primitives": primitives_view,
        "smooth": smooth_view
    });
    if !sharp_pairs.is_empty() {
        record["sharp_edges"] = json!(append_packed(
            binary,
            views,
            &sharp_pairs,
            packed_width(vertex_limit.saturating_sub(1)),
        ));
    }
    Ok(record)
}

/// Maps pot-space normals into the glTF output frame: inverse transpose of diag(scale) * basis.
/// A zero normal becomes the unit up vector first, as Blender's glTF exporter writes it.
fn output_normals(normals: &[DVec3], geometry_scale: DVec3, basis: DMat4) -> Vec<DVec3> {
    normals
        .iter()
        .map(|normal| {
            let unit = normal.try_normalize().unwrap_or(DVec3::Z);
            (basis.transform_vector3(unit) / geometry_scale).normalize_or_zero()
        })
        .collect()
}

/// Base positions per storage vertex: the shape-key basis when keys exist, otherwise the vertex coordinates.
fn shape_key_base_positions(mesh: &Mesh, data_block: Option<&DataBlock>) -> Vec<DVec3> {
    let basis = data_block
        .and_then(|data| data.shape_keys.as_ref())
        .filter(|shape_keys| !shape_keys.keys.is_empty())
        .map(|shape_keys| &shape_keys.basis);
    mesh.vertices
        .iter()
        .map(|vertex| {
            basis
                .and_then(|basis| basis.get(&vertex.id))
                .map_or(vertex.co, |point| DVec3::from_array(*point))
        })
        .collect()
}

/// Pot-space morph displacement per storage vertex, one vector per shape key in `shape_keys.keys` order.
fn shape_key_deltas(mesh: &Mesh, data_block: &DataBlock) -> Result<Vec<Vec<DVec3>>> {
    let Some(shape_keys) = data_block
        .shape_keys
        .as_ref()
        .filter(|shape_keys| !shape_keys.keys.is_empty())
    else {
        return Ok(Vec::new());
    };
    let basis = shape_keys
        .basis
        .iter()
        .map(|(vertex_id, point)| (*vertex_id, DVec3::from_array(*point)))
        .collect::<BTreeMap<_, _>>();
    let mut result = Vec::with_capacity(shape_keys.keys.len());
    for key in shape_keys.keys.values() {
        if !key.value.is_finite() {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "shape-key default value is non-finite",
            ));
        }
        let relative_key = key
            .relative_key
            .as_ref()
            .and_then(|relative_id| shape_keys.keys.get(relative_id));
        if key.relative_key.is_some() && relative_key.is_none() {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "shape key references a missing relative key",
            ));
        }
        let mut deltas = Vec::with_capacity(mesh.vertices.len());
        for vertex in &mesh.vertices {
            let base = basis.get(&vertex.id).copied().unwrap_or(vertex.co);
            let relative = relative_key
                .and_then(|relative| relative.positions.get(&vertex.id))
                .map_or(base, |point| DVec3::from_array(*point));
            let destination = key
                .positions
                .get(&vertex.id)
                .map_or(relative, |point| DVec3::from_array(*point));
            let mask = if let Some(group_id) = &key.vertex_group {
                let group_exists = data_block
                    .vertex_groups
                    .iter()
                    .any(|group| &group.id == group_id);
                if !group_exists {
                    return Err(PotError::new(
                        ErrorCode::SceneInvalid,
                        "shape-key vertex group does not exist",
                    ));
                }
                let weight = data_block
                    .vertex_weights
                    .get(&vertex.id)
                    .and_then(|weights| weights.get(group_id))
                    .copied()
                    .unwrap_or(0.0);
                if !weight.is_finite() || weight < 0.0 {
                    return Err(PotError::new(
                        ErrorCode::SceneInvalid,
                        "shape-key vertex group weight is invalid",
                    ));
                }
                weight
            } else {
                1.0
            };
            let delta = (destination - relative) * mask;
            if !delta.is_finite() {
                return Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    "shape-key delta is non-finite",
                ));
            }
            deltas.push(delta);
        }
        result.push(deltas);
    }
    Ok(result)
}

/// Output-frame layout and shape-key displacement for one exported mesh.
struct MeshExportLayout {
    layout: ExportVertexLayout,
    key_deltas: Vec<Vec<DVec3>>,
}

fn mesh_export_layout(
    mesh: &Mesh,
    data_block: Option<&DataBlock>,
    geometry_scale: DVec3,
    basis: DMat4,
    carried_color_layers: &[usize],
) -> Result<MeshExportLayout> {
    let key_deltas = match data_block {
        Some(block) => shape_key_deltas(mesh, block)?,
        None => Vec::new(),
    };
    let base = shape_key_base_positions(mesh, data_block);
    let mut layers = vec![output_normals(
        &corner_normals(mesh, &base)?,
        geometry_scale,
        basis,
    )];
    for deltas in &key_deltas {
        let morphed = base
            .iter()
            .zip(deltas)
            .map(|(point, delta)| *point + *delta)
            .collect::<Vec<_>>();
        // Morph layers follow the base rules, so a collapsed face gets the same normals as Blender.
        layers.push(output_normals(
            &corner_normals(mesh, &morphed)?,
            geometry_scale,
            basis,
        ));
    }
    let fallback = output_normals(&[DVec3::Z], geometry_scale, basis)[0];
    let store = color_layers::read(mesh);
    let carried = carried_color_layers
        .iter()
        .map(|index| {
            store
                .layers
                .get(*index)
                .ok_or_else(|| {
                    PotError::new(ErrorCode::InternalError, "selected color layer is missing")
                })
                .and_then(|layer| CarriedColor::new(mesh, layer))
        })
        .collect::<Result<Vec<_>>>()?;
    let layout = export_vertex_layout(mesh, &layers, fallback, &carried)?;
    Ok(MeshExportLayout { layout, key_deltas })
}

/// Material slots bound by faces, with each effective material resolved from the snapshot.
fn used_material_bindings<'a>(
    node_id: &Id,
    node: &'a Node,
    snapshot: &'a Snapshot,
    data: Option<&'a DataBlock>,
    mesh: &Mesh,
) -> Result<Vec<(usize, &'a Id)>> {
    let mut used = BTreeSet::new();
    let mut bindings = Vec::new();
    for face in &mesh.faces {
        let slot = usize::try_from(face.material_index).map_err(|_| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "mesh face material slot exceeds platform range",
                json!({ "face_id": face.id, "slot": face.material_index }),
            )
        })?;
        if !used.insert(slot) {
            continue;
        }
        if let Some(material) = snapshot.slot_material(node_id, node, data, slot) {
            bindings.push((slot, material));
        }
    }
    Ok(bindings)
}
fn material_slot_key(
    node_id: &Id,
    node: &Node,
    snapshot: &Snapshot,
    data: Option<&DataBlock>,
) -> Vec<Option<Id>> {
    (0..snapshot.slot_count(node_id, node, data))
        .map(|slot| snapshot.slot_material(node_id, node, data, slot).cloned())
        .collect()
}

/// Whether a `vertex_groups` attribute is a structurally valid export of `data_block`'s groups:
/// every group name exists on the block and every entry is `v{vertex id}` with a finite, nonnegative
/// weight. Weights are not compared with the source, because modifiers may change them.
fn valid_vertex_groups_attribute(attribute: &Value, data_block: &DataBlock) -> bool {
    if data_block.vertex_groups.is_empty() {
        return false;
    }
    let Some(groups) = attribute.as_object() else {
        return false;
    };
    groups.iter().all(|(name, entries)| {
        data_block
            .vertex_groups
            .iter()
            .any(|group| group.name == *name)
            && entries.as_object().is_some_and(|weights| {
                weights.iter().all(|(key, weight)| {
                    key.strip_prefix('v')
                        .is_some_and(|id| id.parse::<u32>().is_ok())
                        && weight
                            .as_f64()
                            .is_some_and(|weight| weight.is_finite() && weight >= 0.0)
                })
            })
    })
}

/// Blender per-element attributes that carry no exported data: editor selection, cached
/// topology, the vertex positions, and the shading flag already kept in `blender_polygon_smooth`.
/// Blender 5.2 attributes the writer skips. Each one's data is carried elsewhere (face slots in
/// `Face.material_index`, edge flags in `blender_edge_flags`, custom normals in the decoded
/// `custom_normal` attribute), is editor state, or is topology kept in the mesh; hidden geometry is
/// written, as Blender's exporter writes it (`primitive_extract.py` reads no hide flags).
const BLENDER_BOOKKEEPING_ATTRIBUTES: [&str; 18] = [
    "material_index",
    "sharp_edge",
    "uv_seam",
    ".hide_poly",
    ".select_vert",
    ".select_edge",
    ".select_poly",
    ".uv_select_vert",
    ".uv_select_edge",
    ".uv_select_face",
    ".edge_verts",
    ".corner_vert",
    ".corner_edge",
    "position",
    "sharp_face",
    ".hide_vert",
    ".hide_edge",
    "custom_normal",
];
/// Whether a `blender_attributes` entry is ignored by the writer. Color layers are exported as
/// `COLOR_n` attributes; this predicate handles attribute validation separately.
fn blender_attribute_exportable(entry: &Value) -> bool {
    let name = entry["name"].as_str().unwrap_or_default();
    BLENDER_BOOKKEEPING_ATTRIBUTES.contains(&name) || is_color_attribute(entry)
}
fn validate_mesh_export_attributes(mesh: &Mesh, data_block: Option<&DataBlock>) -> Result<()> {
    for (name, value) in &mesh.attributes {
        let supported = match name.as_str() {
            "uv_layers"
            | "blender_polygon_smooth"
            | "blender_edge_flags"
            | "seam"
            | "shade_smooth"
            | "sharp"
            | "color_layers"
            | "custom_normal" => true,
            "blender_metadata" => value["custom_properties"]
                .as_object()
                .is_some_and(serde_json::Map::is_empty),
            "blender_attributes" => value
                .as_array()
                .is_some_and(|entries| entries.iter().all(blender_attribute_exportable)),
            // An instance mesh has no data block to hold groups. Blender's exporter writes vertex
            // groups only as skin weights, so an instance's member groups are not written either.
            "vertex_groups" => {
                data_block.is_none_or(|block| valid_vertex_groups_attribute(value, block))
            }
            _ if color_layer_storage_key(mesh, name) => true,
            _ => false,
        };
        if !supported {
            let message = if name == "vertex_groups" {
                "glTF export does not preserve Potter mesh attribute `vertex_groups`: it does not match the data block's vertex groups".to_owned()
            } else {
                format!("glTF export does not preserve Potter mesh attribute `{name}`")
            };
            return Err(unsupported_export("mesh.vertex_attributes", &message));
        }
    }
    Ok(())
}

fn source_mesh_for_export<'a>(
    node: &Node,
    data_block: &'a DataBlock,
    evaluated: &'a Mesh,
    purpose: EvaluationPurpose,
) -> Result<&'a Mesh> {
    let Some(shape_keys) = data_block
        .shape_keys
        .as_ref()
        .filter(|shape_keys| !shape_keys.keys.is_empty())
    else {
        return Ok(evaluated);
    };
    if node
        .modifiers
        .iter()
        .any(|modifier| modifier.is_active(purpose) && modifier.modifier_type != "armature")
    {
        return Err(unsupported_export(
            "mesh.shape_key_modifier_stack",
            "glTF shape-key export cannot preserve shape keys through non-armature enabled modifiers",
        ));
    }
    let _ = shape_keys;
    data_block.mesh.as_ref().ok_or_else(|| {
        PotError::new(
            ErrorCode::SceneInvalid,
            "shape-key mesh Data-Block has no base mesh",
        )
    })
}

fn append_child(nodes: &mut [Value], parent_index: usize, child_index: usize) {
    let children = nodes[parent_index]["children"].as_array_mut();
    if let Some(children) = children {
        children.push(json!(child_index));
    } else {
        nodes[parent_index]["children"] = json!([child_index]);
    }
}

fn finite_matrix(matrix: DMat4) -> bool {
    matrix.to_cols_array().iter().all(|value| value.is_finite())
}
struct NodeFrameMaps {
    node_local_matrices: BTreeMap<Id, DMat4>,
    node_world_matrices: BTreeMap<Id, DMat4>,
    mesh_geometry_scales: BTreeMap<Id, DVec3>,
    node_offset_matrices: BTreeMap<Id, DMat4>,
}

struct BoneFrameMaps {
    local: BTreeMap<(Id, Id), DMat4>,
    world: BTreeMap<(Id, Id), DMat4>,
    offset: BTreeMap<(Id, Id), DMat4>,
}

pub(crate) struct VrmFrameMaps {
    // Each matrix maps a source-local metadata point into the emitted local frame.
    pub(crate) node_offset_matrices: BTreeMap<Id, DMat4>,
    pub(crate) bone_offset_matrices: BTreeMap<(Id, Id), DMat4>,
    node_local_matrices: BTreeMap<Id, DMat4>,
    node_world_matrices: BTreeMap<Id, DMat4>,
    mesh_geometry_scales: BTreeMap<Id, DVec3>,
    bone_local_matrices: BTreeMap<(Id, Id), DMat4>,
    bone_world_matrices: BTreeMap<(Id, Id), DMat4>,
}

pub(crate) fn avatar_basis(version: AvatarVersion) -> DMat4 {
    match version {
        AvatarVersion::V1 => POT_TO_GLTF,
        AvatarVersion::V0 => DMat4::from_rotation_y(std::f64::consts::PI) * POT_TO_GLTF,
    }
}

fn matrices_close(left: DMat4, right: DMat4) -> bool {
    left.to_cols_array()
        .into_iter()
        .zip(right.to_cols_array())
        .all(|(left, right)| (left - right).abs() <= 1.0e-7 * left.abs().max(right.abs()).max(1.0))
}

fn positive_uniform_scale(scale: DVec3) -> bool {
    let average = (scale.x + scale.y + scale.z) / 3.0;
    average.is_finite()
        && average > 0.0
        && [scale.x, scale.y, scale.z]
            .into_iter()
            .all(|value| (value - average).abs() <= 1.0e-7 * average.max(1.0))
}

fn normalized_v1_local(matrix: DMat4, can_bake_scale: bool) -> Result<(DMat4, DVec3)> {
    if !finite_matrix(matrix) {
        return Err(unsupported_export(
            "vrm.v1.node_transform",
            "VRM 1.0 requires finite node transforms.",
        ));
    }
    let (scale, rotation, translation) = matrix.to_scale_rotation_translation();
    if !scale.is_finite() || !rotation.is_finite() || !translation.is_finite() {
        return Err(unsupported_export(
            "vrm.v1.node_transform",
            "VRM 1.0 cannot represent a singular or non-finite node transform.",
        ));
    }
    let reconstructed = DMat4::from_scale_rotation_translation(scale, rotation, translation);
    if !matrices_close(matrix, reconstructed) {
        return Err(unsupported_export(
            "vrm.v1.node_transform",
            "VRM 1.0 cannot preserve a sheared node transform; apply transforms so each node has only rotation and positive uniform scale, or export VRM 0.x.",
        ));
    }
    if positive_uniform_scale(scale) {
        let uniform_scale = DVec3::splat((scale.x + scale.y + scale.z) / 3.0);
        return Ok((
            DMat4::from_scale_rotation_translation(uniform_scale, rotation, translation),
            DVec3::ONE,
        ));
    }
    if can_bake_scale {
        return Ok((
            DMat4::from_rotation_translation(rotation, translation),
            scale,
        ));
    }
    Err(unsupported_export(
        "vrm.v1.node_transform",
        "VRM 1.0 requires positive uniform scale on every node; non-uniform or negative scale on a skin joint cannot be baked into geometry faithfully.",
    ))
}

fn exportable_mesh_nodes(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    visible: &BTreeSet<Id>,
) -> BTreeSet<Id> {
    doc.nodes
        .iter()
        .filter(|(id, node)| {
            node.visible
                && node.render_visible
                && visible.contains(*id)
                && snapshot
                    .meshes
                    .get(*id)
                    .is_some_and(|mesh| !mesh.vertices.is_empty())
                && (node
                    .data
                    .as_ref()
                    .and_then(|data_id| doc.data_blocks.get(data_id))
                    .is_some_and(|data| data.data_type == "mesh")
                    || node.properties.contains_key("instance_collection"))
        })
        .map(|(id, _)| id.clone())
        .collect()
}
fn node_source_world_matrices(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    frame: f64,
) -> Result<BTreeMap<Id, DMat4>> {
    fn visit(
        id: &Id,
        doc: &SceneDoc,
        snapshot: &Snapshot,
        frame: f64,
        worlds: &mut BTreeMap<Id, DMat4>,
        active: &mut BTreeSet<Id>,
    ) -> Result<DMat4> {
        if let Some(world) = worlds.get(id) {
            return Ok(*world);
        }
        if let Some(evaluated) = snapshot.nodes.get(id) {
            let world = DMat4::from_cols_array(&evaluated.world_matrix);
            if !finite_matrix(world) {
                return Err(PotError::new(
                    ErrorCode::EvaluationFailed,
                    "VRM node world transform is non-finite",
                ));
            }
            worlds.insert(id.clone(), world);
            return Ok(world);
        }
        if !active.insert(id.clone()) {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "glTF node hierarchy contains a cycle",
            ));
        }
        let node = doc.nodes.get(id).ok_or_else(|| {
            PotError::new(ErrorCode::SceneInvalid, "node transform target is missing")
        })?;
        let transform = crate::eval::animation::animated_transform(node, doc, frame)?;
        let mut local = transform.matrix();
        if let Some(parent_inverse) = &node.parent_inverse {
            local = DMat4::from_cols_array(parent_inverse) * local;
        }
        let parent_world = if let Some(parent) = &node.parent {
            visit(parent, doc, snapshot, frame, worlds, active)?
        } else {
            DMat4::IDENTITY
        };
        let world = parent_world * local;
        if !finite_matrix(world) {
            return Err(PotError::new(
                ErrorCode::EvaluationFailed,
                "VRM node world transform is non-finite",
            ));
        }
        worlds.insert(id.clone(), world);
        active.remove(id);
        Ok(world)
    }

    let mut worlds = BTreeMap::new();
    let mut active = BTreeSet::new();
    for id in doc.nodes.keys() {
        visit(id, doc, snapshot, frame, &mut worlds, &mut active)?;
    }
    Ok(worlds)
}

fn node_frame_maps(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    version: AvatarVersion,
    mesh_nodes: &BTreeSet<Id>,
    bone_world_matrices: Option<&BTreeMap<(Id, Id), DMat4>>,
) -> Result<NodeFrameMaps> {
    fn visit(
        id: &Id,
        doc: &SceneDoc,
        source_world_matrices: &BTreeMap<Id, DMat4>,
        version: AvatarVersion,
        mesh_nodes: &BTreeSet<Id>,
        bone_world_matrices: Option<&BTreeMap<(Id, Id), DMat4>>,
        basis: DMat4,
        inverse_basis: DMat4,
        local_matrices: &mut BTreeMap<Id, DMat4>,
        world_matrices: &mut BTreeMap<Id, DMat4>,
        geometry_scales: &mut BTreeMap<Id, DVec3>,
        offset_matrices: &mut BTreeMap<Id, DMat4>,
        active: &mut BTreeSet<Id>,
    ) -> Result<DMat4> {
        if let Some(matrix) = world_matrices.get(id) {
            return Ok(*matrix);
        }
        if !active.insert(id.clone()) {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "glTF node hierarchy contains a cycle",
            ));
        }
        let node = doc.nodes.get(id).ok_or_else(|| {
            PotError::new(ErrorCode::SceneInvalid, "node transform target is missing")
        })?;
        let source_world = source_world_matrices.get(id).copied().ok_or_else(|| {
            PotError::new(
                ErrorCode::EvaluationFailed,
                "VRM node world transform is missing",
            )
        })?;
        if !finite_matrix(source_world) {
            return Err(PotError::new(
                ErrorCode::EvaluationFailed,
                "VRM node world transform is non-finite",
            ));
        }
        let output_target_world = basis * source_world * inverse_basis;
        let parent_node_world = if let Some(parent) = &node.parent {
            visit(
                parent,
                doc,
                source_world_matrices,
                version,
                mesh_nodes,
                bone_world_matrices,
                basis,
                inverse_basis,
                local_matrices,
                world_matrices,
                geometry_scales,
                offset_matrices,
                active,
            )?
        } else {
            DMat4::IDENTITY
        };
        let parent_world = if node.parent_type == crate::model::ParentType::Bone {
            if let Some(bone_world_matrices) = bone_world_matrices {
                let parent = node.parent.as_ref().ok_or_else(|| {
                    PotError::new(ErrorCode::SceneInvalid, "bone parent node is missing")
                })?;
                let bone = node.parent_bone.as_ref().ok_or_else(|| {
                    PotError::new(ErrorCode::SceneInvalid, "bone parent ID is missing")
                })?;
                bone_world_matrices
                    .get(&(parent.clone(), bone.clone()))
                    .copied()
                    .ok_or_else(|| {
                        PotError::with_details(
                            ErrorCode::SceneInvalid,
                            "normalized parent bone frame is missing",
                            json!({ "parent": parent, "bone": bone, "node": id }),
                        )
                    })?
            } else {
                parent_node_world
            }
        } else {
            parent_node_world
        };
        let parent_inverse = parent_world.inverse();
        if !finite_matrix(parent_inverse) {
            return Err(unsupported_export(
                "vrm.node_singular_parent",
                "VRM export requires invertible parent node transforms.",
            ));
        }
        let source_local = parent_inverse * output_target_world;
        let (output_local, baked_scale) = if version == AvatarVersion::V1 {
            normalized_v1_local(source_local, true)?
        } else {
            (source_local, DVec3::ONE)
        };
        let geometry_scale = if mesh_nodes.contains(id) {
            baked_scale
        } else {
            DVec3::ONE
        };
        let output_world = parent_world * output_local;
        let source_to_output = output_world.inverse() * basis * source_world;
        if !finite_matrix(output_world) || !finite_matrix(source_to_output) {
            return Err(unsupported_export(
                "vrm.node_singular_transform",
                "VRM export requires invertible node transforms.",
            ));
        }
        local_matrices.insert(id.clone(), output_local);
        world_matrices.insert(id.clone(), output_world);
        geometry_scales.insert(id.clone(), geometry_scale);
        offset_matrices.insert(id.clone(), source_to_output);
        active.remove(id);
        Ok(output_world)
    }

    let basis = avatar_basis(version);
    let inverse_basis = basis.inverse();
    let source_world_matrices = node_source_world_matrices(doc, snapshot, snapshot.frame)?;
    let mut local_matrices = BTreeMap::new();
    let mut world_matrices = BTreeMap::new();
    let mut geometry_scales = BTreeMap::new();
    let mut offset_matrices = BTreeMap::new();
    let mut active = BTreeSet::new();
    for id in doc.nodes.keys() {
        visit(
            id,
            doc,
            &source_world_matrices,
            version,
            mesh_nodes,
            bone_world_matrices,
            basis,
            inverse_basis,
            &mut local_matrices,
            &mut world_matrices,
            &mut geometry_scales,
            &mut offset_matrices,
            &mut active,
        )?;
    }
    Ok(NodeFrameMaps {
        node_local_matrices: local_matrices,
        node_world_matrices: world_matrices,
        mesh_geometry_scales: geometry_scales,
        node_offset_matrices: offset_matrices,
    })
}
fn bone_parent_armatures(doc: &SceneDoc) -> BTreeSet<Id> {
    doc.nodes
        .values()
        .filter(|node| node.parent_type == crate::model::ParentType::Bone)
        .filter_map(|node| node.parent.clone())
        .collect()
}

fn normalized_bone_frames(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    version: AvatarVersion,
    node_world_matrices: &BTreeMap<Id, DMat4>,
    armature_ids: &BTreeSet<Id>,
) -> Result<BoneFrameMaps> {
    fn visit(
        armature_id: &Id,
        bone_id: &Id,
        armature: &ArmatureData,
        source_world_bones: &BTreeMap<Id, [f64; 16]>,
        version: AvatarVersion,
        basis: DMat4,
        inverse_basis: DMat4,
        output_armature_world: DMat4,
        output_armature_inverse: DMat4,
        local_matrices: &mut BTreeMap<(Id, Id), DMat4>,
        world_matrices: &mut BTreeMap<(Id, Id), DMat4>,
        offset_matrices: &mut BTreeMap<(Id, Id), DMat4>,
    ) -> Result<DMat4> {
        let key = (armature_id.clone(), bone_id.clone());
        if let Some(matrix) = world_matrices.get(&key) {
            return Ok(*matrix);
        }
        let bone = armature.bones.get(bone_id).ok_or_else(|| {
            PotError::new(
                ErrorCode::SceneInvalid,
                "VRM bone transform target is missing",
            )
        })?;
        let source_bone_world = source_world_bones
            .get(bone_id)
            .map(DMat4::from_cols_array)
            .ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::EvaluationFailed,
                    "VRM bone world transform is missing",
                    json!({ "armature": armature_id, "bone": bone_id }),
                )
            })?;
        let (parent_world, output_local) = if version == AvatarVersion::V0 {
            let parent_world = if let Some(parent_id) = &bone.parent {
                visit(
                    armature_id,
                    parent_id,
                    armature,
                    source_world_bones,
                    version,
                    basis,
                    inverse_basis,
                    output_armature_world,
                    output_armature_inverse,
                    local_matrices,
                    world_matrices,
                    offset_matrices,
                )?
            } else {
                output_armature_world
            };
            let head = output_armature_inverse
                .transform_point3(basis.transform_point3(source_bone_world.w_axis.truncate()));
            let parent_head = if let Some(parent_id) = &bone.parent {
                let parent_source = source_world_bones
                    .get(parent_id)
                    .map(DMat4::from_cols_array)
                    .ok_or_else(|| {
                        PotError::new(
                            ErrorCode::EvaluationFailed,
                            "VRM parent bone world transform is missing",
                        )
                    })?;
                output_armature_inverse
                    .transform_point3(basis.transform_point3(parent_source.w_axis.truncate()))
            } else {
                DVec3::ZERO
            };
            (parent_world, DMat4::from_translation(head - parent_head))
        } else {
            let source_parent_world = if let Some(parent_id) = &bone.parent {
                visit(
                    armature_id,
                    parent_id,
                    armature,
                    source_world_bones,
                    version,
                    basis,
                    inverse_basis,
                    output_armature_world,
                    output_armature_inverse,
                    local_matrices,
                    world_matrices,
                    offset_matrices,
                )?
            } else {
                output_armature_world
            };
            let desired_world = basis * source_bone_world * inverse_basis;
            let inverse_parent = source_parent_world.inverse();
            if !finite_matrix(inverse_parent) {
                return Err(unsupported_export(
                    "vrm.v1.singular_bone_parent",
                    "VRM 1.0 bone transforms require an invertible parent frame.",
                ));
            }
            let (local, _) = normalized_v1_local(inverse_parent * desired_world, false)?;
            (source_parent_world, local)
        };
        let output_world = parent_world * output_local;
        let source_to_output = output_world.inverse() * basis * source_bone_world;
        if !finite_matrix(output_world) || !finite_matrix(source_to_output) {
            return Err(unsupported_export(
                "vrm.bone_singular_transform",
                "VRM bone transforms must be finite and invertible.",
            ));
        }
        local_matrices.insert(key.clone(), output_local);
        world_matrices.insert(key.clone(), output_world);
        offset_matrices.insert(key, source_to_output);
        Ok(output_world)
    }

    let basis = avatar_basis(version);
    let inverse_basis = basis.inverse();
    let mut local_matrices = BTreeMap::new();
    let mut world_matrices = BTreeMap::new();
    let mut offset_matrices = BTreeMap::new();
    for armature_id in armature_ids {
        let armature = doc
            .nodes
            .get(armature_id)
            .and_then(|node| node.data.as_ref())
            .and_then(|data_id| doc.data_blocks.get(data_id))
            .and_then(|data| data.armature.as_ref())
            .ok_or_else(|| {
                PotError::new(ErrorCode::SceneInvalid, "VRM armature data is missing")
            })?;
        let source_world_bones = snapshot.bone_matrices.get(armature_id).ok_or_else(|| {
            PotError::new(
                ErrorCode::EvaluationFailed,
                "VRM armature evaluated bones are missing",
            )
        })?;
        let output_armature_world =
            node_world_matrices
                .get(armature_id)
                .copied()
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::EvaluationFailed,
                        "VRM armature output frame is missing",
                    )
                })?;
        let output_armature_inverse = if version == AvatarVersion::V0 {
            let inverse = output_armature_world.inverse();
            if !finite_matrix(inverse) {
                return Err(unsupported_export(
                    "vrm.v0.singular_armature",
                    "VRM 0.x bone normalization requires an invertible armature node transform.",
                ));
            }
            inverse
        } else {
            DMat4::IDENTITY
        };
        for bone_id in armature.bones.keys() {
            visit(
                armature_id,
                bone_id,
                armature,
                source_world_bones,
                version,
                basis,
                inverse_basis,
                output_armature_world,
                output_armature_inverse,
                &mut local_matrices,
                &mut world_matrices,
                &mut offset_matrices,
            )?;
        }
    }
    Ok(BoneFrameMaps {
        local: local_matrices,
        world: world_matrices,
        offset: offset_matrices,
    })
}

fn build_vrm_frame_maps(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    version: AvatarVersion,
    mesh_nodes: &BTreeSet<Id>,
    armature_ids: &BTreeSet<Id>,
) -> Result<VrmFrameMaps> {
    let mut node_frames = node_frame_maps(doc, snapshot, version, mesh_nodes, None)?;
    let mut bone_frames = normalized_bone_frames(
        doc,
        snapshot,
        version,
        &node_frames.node_world_matrices,
        armature_ids,
    )?;
    let bone_parented_armatures = armature_ids
        .iter()
        .filter(|id| {
            doc.nodes
                .get(*id)
                .is_some_and(|node| node.parent_type == crate::model::ParentType::Bone)
        })
        .count();
    if bone_parented_armatures == 0 {
        node_frames =
            node_frame_maps(doc, snapshot, version, mesh_nodes, Some(&bone_frames.world))?;
    } else {
        for _ in 0..=bone_parented_armatures {
            node_frames =
                node_frame_maps(doc, snapshot, version, mesh_nodes, Some(&bone_frames.world))?;
            bone_frames = normalized_bone_frames(
                doc,
                snapshot,
                version,
                &node_frames.node_world_matrices,
                armature_ids,
            )?;
        }
    }
    Ok(VrmFrameMaps {
        node_offset_matrices: node_frames.node_offset_matrices,
        bone_offset_matrices: bone_frames.offset,
        node_local_matrices: node_frames.node_local_matrices,
        node_world_matrices: node_frames.node_world_matrices,
        mesh_geometry_scales: node_frames.mesh_geometry_scales,
        bone_local_matrices: bone_frames.local,
        bone_world_matrices: bone_frames.world,
    })
}

pub(crate) fn vrm_frame_maps(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    avatar: &AvatarSettings,
    version: AvatarVersion,
) -> Result<VrmFrameMaps> {
    let visible = super::render_visible_nodes(doc, snapshot)?;
    let mesh_nodes = exportable_mesh_nodes(doc, snapshot, &visible);
    let mut armature_ids = super::vrm::referenced_armatures(avatar);
    armature_ids.extend(bone_parent_armatures(doc));
    for (node_id, node) in &doc.nodes {
        if !node.visible || !node.render_visible || !visible.contains(node_id) {
            continue;
        }
        if node
            .data
            .as_ref()
            .and_then(|data_id| doc.data_blocks.get(data_id))
            .is_some_and(|data| data.data_type == "mesh")
            && let Some(binding) = skin_binding(doc, node, snapshot.purpose)?
        {
            armature_ids.insert(binding.armature_node);
        }
    }
    build_vrm_frame_maps(doc, snapshot, version, &mesh_nodes, &armature_ids)
}

fn append_matrix_accessor(
    matrices: &[DMat4],
    binary: &mut Vec<u8>,
    views: &mut Vec<Value>,
    accessors: &mut Vec<Value>,
) -> Result<usize> {
    let mut values = Vec::with_capacity(matrices.len().saturating_mul(16));
    for matrix in matrices {
        for component in matrix.to_cols_array() {
            values.push(to_f32(component)?);
        }
    }
    Ok(append_float_accessor(
        binary,
        views,
        accessors,
        &values,
        matrices.len(),
        "MAT4",
        None,
        None,
    ))
}

struct SkinDefinitions {
    skins: Vec<Value>,
    node_skins: BTreeMap<Id, usize>,
    bone_indices: BTreeMap<Id, BTreeMap<Id, usize>>,
}

fn skin_definitions(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    node_indices: &BTreeMap<Id, usize>,
    skin_bindings: &BTreeMap<Id, SkinBinding>,
    avatar_armature_nodes: Option<&BTreeSet<Id>>,
    frame_maps: Option<&VrmFrameMaps>,
    nodes: &mut Vec<Value>,
    binary: &mut Vec<u8>,
    views: &mut Vec<Value>,
    accessors: &mut Vec<Value>,
) -> Result<SkinDefinitions> {
    let inverse_basis = POT_TO_GLTF.inverse();
    let mut armature_ids = skin_bindings
        .values()
        .map(|binding| binding.armature_node.clone())
        .collect::<BTreeSet<_>>();
    if let Some(avatar_armature_nodes) = avatar_armature_nodes {
        armature_ids.extend(avatar_armature_nodes.iter().cloned());
    }
    armature_ids.extend(bone_parent_armatures(doc));
    let mut bone_indices = BTreeMap::<Id, BTreeMap<Id, usize>>::new();
    for armature_node_id in armature_ids {
        let armature = doc
            .nodes
            .get(&armature_node_id)
            .and_then(|node| node.data.as_ref())
            .and_then(|data_id| doc.data_blocks.get(data_id))
            .and_then(|data| data.armature.as_ref())
            .ok_or_else(|| {
                PotError::new(ErrorCode::SceneInvalid, "skin armature data is missing")
            })?;
        let armature_world = snapshot
            .nodes
            .get(&armature_node_id)
            .map(|node| DMat4::from_cols_array(&node.world_matrix))
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::EvaluationFailed,
                    "skin armature world transform is missing",
                )
            })?;
        if !finite_matrix(armature_world) {
            return Err(PotError::new(
                ErrorCode::EvaluationFailed,
                "skin armature world transform is non-finite",
            ));
        }
        let armature_inverse = armature_world.inverse();
        if !finite_matrix(armature_inverse) {
            return Err(unsupported_export(
                "rig.skin_singular_armature",
                "glTF skin export requires an invertible armature world transform",
            ));
        }
        let world_bones = snapshot
            .bone_matrices
            .get(&armature_node_id)
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::EvaluationFailed,
                    "skin armature evaluated bones are missing",
                )
            })?;
        let indices = armature
            .bones
            .keys()
            .enumerate()
            .map(|(offset, bone_id)| {
                Ok((
                    bone_id.clone(),
                    nodes.len().checked_add(offset).ok_or_else(|| {
                        PotError::new(ErrorCode::LimitExceeded, "glTF node index overflow")
                    })?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        for (bone_id, bone) in &armature.bones {
            let local = if let Some(frame_maps) = frame_maps {
                frame_maps
                    .bone_local_matrices
                    .get(&(armature_node_id.clone(), bone_id.clone()))
                    .copied()
                    .ok_or_else(|| {
                        PotError::new(
                            ErrorCode::InternalError,
                            "normalized VRM bone transform is missing",
                        )
                    })?
            } else {
                let bone_world = world_bones
                    .get(bone_id)
                    .map(DMat4::from_cols_array)
                    .ok_or_else(|| {
                        PotError::with_details(
                            ErrorCode::EvaluationFailed,
                            "skin bone world transform is missing",
                            json!({ "armature": armature_node_id, "bone": bone_id }),
                        )
                    })?;
                let parent_world = if let Some(parent_id) = &bone.parent {
                    if !armature.bones.contains_key(parent_id) {
                        return Err(PotError::new(
                            ErrorCode::SceneInvalid,
                            "skin bone parent does not exist",
                        ));
                    }
                    world_bones
                        .get(parent_id)
                        .map(DMat4::from_cols_array)
                        .ok_or_else(|| {
                            PotError::new(
                                ErrorCode::EvaluationFailed,
                                "skin parent bone world transform is missing",
                            )
                        })?
                } else {
                    armature_world
                };
                let parent_inverse = parent_world.inverse();
                if !finite_matrix(parent_inverse) {
                    return Err(unsupported_export(
                        "rig.skin_singular_bone",
                        "glTF skin export requires invertible bone parent transforms",
                    ));
                }
                let local = parent_inverse * bone_world;
                POT_TO_GLTF * local * inverse_basis
            };
            if !finite_matrix(local) {
                return Err(unsupported_export(
                    "rig.skin_singular_bone",
                    "glTF skin export requires finite bone transforms",
                ));
            }
            let matrix = local
                .to_cols_array()
                .into_iter()
                .map(to_f32)
                .collect::<Result<Vec<_>>>()?;
            let bone_value = serde_json::to_value(bone)
                .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))?;
            nodes.push(json!({
                "name": bone.name,
                "matrix": matrix,
                "extras": { "potter": {
                    "bone_id": bone_id,
                    "armature_id": armature_node_id,
                    "bone": bone_value
                } }
            }));
        }
        for (bone_id, bone) in &armature.bones {
            let child_index = indices[bone_id];
            if let Some(parent_id) = &bone.parent {
                let parent_index = indices.get(parent_id).copied().ok_or_else(|| {
                    PotError::new(ErrorCode::SceneInvalid, "skin bone parent does not exist")
                })?;
                append_child(nodes, parent_index, child_index);
            } else {
                let parent_index =
                    node_indices
                        .get(&armature_node_id)
                        .copied()
                        .ok_or_else(|| {
                            PotError::new(
                                ErrorCode::InternalError,
                                "skin armature node index is missing",
                            )
                        })?;
                append_child(nodes, parent_index, child_index);
            }
        }
        bone_indices.insert(armature_node_id, indices);
    }

    let mut skins = Vec::new();
    let mut node_skins = BTreeMap::new();
    for (mesh_node_id, binding) in skin_bindings {
        let armature = doc
            .nodes
            .get(&binding.armature_node)
            .and_then(|node| node.data.as_ref())
            .and_then(|id| doc.data_blocks.get(id))
            .and_then(|data| data.armature.as_ref())
            .ok_or_else(|| {
                PotError::new(ErrorCode::SceneInvalid, "skin armature data is missing")
            })?;
        let matrix_map = snapshot
            .bone_matrices
            .get(&binding.armature_node)
            .ok_or_else(|| PotError::new(ErrorCode::EvaluationFailed, "skin bones are missing"))?;
        let source_mesh_world = snapshot
            .nodes
            .get(mesh_node_id)
            .map(|node| DMat4::from_cols_array(&node.world_matrix))
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::EvaluationFailed,
                    "skinned mesh transform is missing",
                )
            })?;
        let mesh_world = if let Some(frames) = frame_maps {
            frames
                .node_world_matrices
                .get(mesh_node_id)
                .copied()
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::EvaluationFailed,
                        "normalized skin mesh transform is missing",
                    )
                })?
        } else {
            source_mesh_world
        };
        let node_bones = bone_indices.get(&binding.armature_node).ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "skin bone-node indices are missing",
            )
        })?;
        let mut inverse_bind_matrices = Vec::with_capacity(binding.joint_ids.len());
        let mut joints = Vec::with_capacity(binding.joint_ids.len());
        for bone_id in &binding.joint_ids {
            let joint_world = if let Some(frames) = frame_maps {
                frames
                    .bone_world_matrices
                    .get(&(binding.armature_node.clone(), bone_id.clone()))
                    .copied()
                    .ok_or_else(|| {
                        PotError::new(
                            ErrorCode::EvaluationFailed,
                            "normalized skin joint transform is missing",
                        )
                    })?
            } else {
                matrix_map
                    .get(bone_id)
                    .map(DMat4::from_cols_array)
                    .ok_or_else(|| {
                        PotError::new(
                            ErrorCode::EvaluationFailed,
                            "skin joint world transform is missing",
                        )
                    })?
            };
            let joint_inverse = joint_world.inverse();
            let inverse_bind = joint_inverse * mesh_world;
            if !finite_matrix(inverse_bind) {
                return Err(unsupported_export(
                    "rig.skin_singular_bind",
                    "glTF inverse bind matrices require invertible joint transforms",
                ));
            }
            inverse_bind_matrices.push(if frame_maps.is_some() {
                inverse_bind
            } else {
                POT_TO_GLTF * inverse_bind * inverse_basis
            });
            joints.push(node_bones.get(bone_id).copied().ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "skin joint node is missing")
            })?);
        }
        let inverse_bind_accessor =
            append_matrix_accessor(&inverse_bind_matrices, binary, views, accessors)?;
        let roots = armature
            .bones
            .iter()
            .filter(|(_, bone)| bone.parent.is_none())
            .filter_map(|(id, _)| node_bones.get(id).copied())
            .collect::<Vec<_>>();
        let mut skin = json!({
            "name": format!("{mesh_node_id}_skin"),
            "joints": joints,
            "inverseBindMatrices": inverse_bind_accessor,
            "extras": { "potter": { "armature_id": binding.armature_node } }
        });
        if roots.len() == 1 {
            skin["skeleton"] = json!(roots[0]);
        }
        let skin_index = skins.len();
        skins.push(skin);
        node_skins.insert(mesh_node_id.clone(), skin_index);
    }
    Ok(SkinDefinitions {
        skins,
        node_skins,
        bone_indices,
    })
}

pub(crate) struct VrmWriterMappings<'a> {
    pub(crate) node_indices: &'a BTreeMap<Id, usize>,
    pub(crate) bone_indices: &'a BTreeMap<Id, BTreeMap<Id, usize>>,
    pub(crate) material_indices: &'a BTreeMap<Id, usize>,
    pub(crate) surfaces: &'a BTreeMap<Id, ResolvedSurface>,
    pub(crate) tex_coords: &'a TexCoordPlan,
    pub(crate) node_mesh_indices: &'a mut BTreeMap<Id, usize>,
    pub(crate) node_offset_matrices: &'a BTreeMap<Id, DMat4>,
    pub(crate) bone_offset_matrices: &'a BTreeMap<(Id, Id), DMat4>,
}

pub(crate) fn export_with_root(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    root_path: &Path,
    bin_uri: Option<&str>,
) -> Result<(Vec<u8>, Vec<u8>)> {
    export_with_root_inner(doc, snapshot, root_path, bin_uri, None, None)
}

pub(crate) fn export_vrm_with_frames(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    root_path: &Path,
    version: AvatarVersion,
    frames: &VrmFrameMaps,
) -> Result<Vec<u8>> {
    let avatar = super::vrm::required_avatar(doc)?;
    let (json_bytes, binary) = export_with_root_inner(
        doc,
        snapshot,
        root_path,
        None,
        Some((avatar, version)),
        Some(frames),
    )?;
    to_glb(&json_bytes, &binary)
}

/// Resolves every material of `doc` to its glTF surface, keyed by material ID.
///
/// A material used by a mesh must resolve. An unused material whose graph does not resolve falls
/// back to its explicit fields, so it never blocks export. Every referenced image must exist.
/// Every material resolved to its glTF surface once: used materials must resolve, unused ones fall
/// back to their fields so an invalid graph on an unused material never blocks export.
pub(crate) fn resolve_material_surfaces(doc: &SceneDoc) -> Result<BTreeMap<Id, ResolvedSurface>> {
    resolve_material_surfaces_using(doc, |id| material_is_used(doc, id))
}

fn resolve_material_surfaces_with_snapshot(
    doc: &SceneDoc,
    snapshot: &Snapshot,
) -> Result<BTreeMap<Id, ResolvedSurface>> {
    resolve_material_surfaces_using(doc, |material_id| {
        doc.nodes.iter().any(|(node_id, node)| {
            let data = node
                .data
                .as_ref()
                .and_then(|data_id| doc.data_blocks.get(data_id));
            (0..snapshot.slot_count(node_id, node, data))
                .any(|slot| snapshot.slot_material(node_id, node, data, slot) == Some(material_id))
        })
    })
}

fn resolve_material_surfaces_using(
    doc: &SceneDoc,
    mut is_used: impl FnMut(&Id) -> bool,
) -> Result<BTreeMap<Id, ResolvedSurface>> {
    let mut surfaces = BTreeMap::new();
    for id in doc.materials.keys() {
        let surface = if is_used(id) {
            resolve_surface(doc, id)?.0
        } else {
            match resolve_surface(doc, id) {
                Ok((surface, _)) => surface,
                Err(_) => resolve_fields_surface(doc, id)?,
            }
        };
        for texture in surface_textures(&surface) {
            for source in texture.channels.iter().flatten() {
                texture_image_for_export(doc, &source.image)?;
            }
        }
        surfaces.insert(id.clone(), surface);
    }
    Ok(surfaces)
}

/// Every sampled texture a resolved surface references, in a fixed order.
pub(crate) fn surface_textures(surface: &ResolvedSurface) -> Vec<&SampledTexture> {
    let mut textures = Vec::new();
    textures.extend(surface.base_color.texture.as_ref());
    textures.extend(surface.metallic_roughness_texture.as_ref());
    textures.extend(
        surface
            .occlusion
            .as_ref()
            .map(|occlusion| &occlusion.texture),
    );
    textures.extend(surface.normal.as_ref().map(|normal| &normal.texture));
    textures.extend(surface.emissive.texture.as_ref());
    textures.extend(
        surface
            .transmission
            .as_ref()
            .and_then(|transmission| transmission.texture.as_ref()),
    );
    if let Some(clearcoat) = &surface.clearcoat {
        textures.extend(clearcoat.texture.as_ref());
        textures.extend(clearcoat.roughness_texture.as_ref());
        textures.extend(clearcoat.normal.as_ref().map(|normal| &normal.texture));
    }
    if let Some(sheen) = &surface.sheen {
        textures.extend(sheen.color_texture.as_ref());
        textures.extend(sheen.roughness_texture.as_ref());
    }
    if let Some(specular) = &surface.specular {
        textures.extend(specular.texture.as_ref());
        textures.extend(specular.color_texture.as_ref());
    }
    textures
}

/// The glTF material for one resolved surface. `extras.potter` keeps the Potter values that glTF
/// cannot express, so Potter round trips keep them.
fn material_value(
    id: &Id,
    material: &Material,
    surface: &ResolvedSurface,
    tex_coords: &TexCoords,
    omitted: &BTreeSet<Option<String>>,
    doc: &SceneDoc,
    root_path: &Path,
    texture_export: &mut TextureExport,
    binary: &mut Vec<u8>,
    views: &mut Vec<Value>,
    extensions_used: &mut BTreeSet<String>,
) -> Result<Value> {
    let mut extensions = serde_json::Map::new();
    // glTF colors stay in [0, 1]; a peak above 1 moves into KHR_materials_emissive_strength.
    let mut emissive_factor = surface.emissive.factor;
    let peak = emissive_factor.iter().copied().fold(0.0_f64, f64::max);
    if peak > 1.0 {
        for component in &mut emissive_factor {
            *component /= peak;
        }
        add_extension(
            &mut extensions,
            extensions_used,
            "KHR_materials_emissive_strength",
            json!({ "emissiveStrength": peak }),
        );
    }
    let mut pbr = json!({
        "baseColorFactor": surface.base_color.factor,
        "metallicFactor": surface.metallic,
        "roughnessFactor": surface.roughness
    });
    if let Some(texture) = &surface.base_color.texture
        && !omitted.contains(&texture.uv_layer)
    {
        pbr["baseColorTexture"] = texture_export.sampled_info(
            texture,
            base_reads(surface),
            tex_coords,
            doc,
            root_path,
            binary,
            views,
        )?;
    }
    if let Some(texture) = &surface.metallic_roughness_texture
        && !omitted.contains(&texture.uv_layer)
    {
        pbr["metallicRoughnessTexture"] = texture_export.sampled_info(
            texture,
            reads::MR,
            tex_coords,
            doc,
            root_path,
            binary,
            views,
        )?;
    }
    let mut value = json!({
        "name": material.name,
        "pbrMetallicRoughness": pbr,
        "emissiveFactor": emissive_factor,
        "doubleSided": surface.double_sided,
        "extras": { "potter": { "id": id } }
    });
    // These fields are the material's own values. A graph material takes them from its graph, so
    // writing them would contradict the glTF extensions on re-import.
    if material
        .node_tree
        .as_ref()
        .is_none_or(|graph| is_default_material_graph(doc, graph))
    {
        value["extras"]["potter"]["emission_strength"] = json!(material.emission_strength);
        value["extras"]["potter"]["transmission"] = json!(material.transmission);
        value["extras"]["potter"]["ior"] = json!(material.ior);
    }
    if let Some(texture) = &surface.emissive.texture
        && !omitted.contains(&texture.uv_layer)
    {
        value["emissiveTexture"] = texture_export.sampled_info(
            texture,
            reads::RGB,
            tex_coords,
            doc,
            root_path,
            binary,
            views,
        )?;
    }
    if let Some(normal) = &surface.normal
        && !omitted.contains(&normal.texture.uv_layer)
    {
        value["normalTexture"] = textured_normal_info(
            normal,
            tex_coords,
            doc,
            root_path,
            texture_export,
            binary,
            views,
        )?;
    }
    if let Some(occlusion) = &surface.occlusion
        && !omitted.contains(&occlusion.texture.uv_layer)
    {
        let mut info = texture_export.sampled_info(
            &occlusion.texture,
            reads::R,
            tex_coords,
            doc,
            root_path,
            binary,
            views,
        )?;
        if !crate::float::equal_f64(occlusion.strength, 1.0) {
            info["strength"] = json!(occlusion.strength);
        }
        value["occlusionTexture"] = info;
    }
    match &surface.alpha_mode {
        AlphaMode::Opaque => {}
        AlphaMode::Blend => value["alphaMode"] = json!("BLEND"),
        AlphaMode::Mask { cutoff } => {
            value["alphaMode"] = json!("MASK");
            value["alphaCutoff"] = json!(cutoff);
        }
    }
    if surface.unlit {
        add_extension(
            &mut extensions,
            extensions_used,
            "KHR_materials_unlit",
            json!({}),
        );
    }
    if let Some(transmission) = &surface.transmission {
        let mut extension = serde_json::Map::new();
        if !crate::float::equal_f64(transmission.factor, 0.0) {
            extension.insert("transmissionFactor".to_owned(), json!(transmission.factor));
        }
        if let Some(texture) = &transmission.texture
            && !omitted.contains(&texture.uv_layer)
        {
            extension.insert(
                "transmissionTexture".to_owned(),
                texture_export.sampled_info(
                    texture,
                    reads::R,
                    tex_coords,
                    doc,
                    root_path,
                    binary,
                    views,
                )?,
            );
        }
        add_extension(
            &mut extensions,
            extensions_used,
            "KHR_materials_transmission",
            Value::Object(extension),
        );
    }
    // IOR is written only with the extensions that use it (Blender's rule).
    if (surface.transmission.is_some() || surface.specular.is_some())
        && !crate::float::equal_f64(surface.ior, 1.5)
    {
        add_extension(
            &mut extensions,
            extensions_used,
            "KHR_materials_ior",
            json!({ "ior": surface.ior }),
        );
    }
    if let Some(clearcoat) = &surface.clearcoat {
        let mut extension = serde_json::Map::new();
        if !crate::float::equal_f64(clearcoat.factor, 0.0) {
            extension.insert("clearcoatFactor".to_owned(), json!(clearcoat.factor));
        }
        if let Some(texture) = &clearcoat.texture
            && !omitted.contains(&texture.uv_layer)
        {
            extension.insert(
                "clearcoatTexture".to_owned(),
                texture_export.sampled_info(
                    texture,
                    reads::R,
                    tex_coords,
                    doc,
                    root_path,
                    binary,
                    views,
                )?,
            );
        }
        if !crate::float::equal_f64(clearcoat.roughness_factor, 0.0) {
            extension.insert(
                "clearcoatRoughnessFactor".to_owned(),
                json!(clearcoat.roughness_factor),
            );
        }
        if let Some(texture) = &clearcoat.roughness_texture
            && !omitted.contains(&texture.uv_layer)
        {
            extension.insert(
                "clearcoatRoughnessTexture".to_owned(),
                texture_export.sampled_info(
                    texture,
                    reads::G,
                    tex_coords,
                    doc,
                    root_path,
                    binary,
                    views,
                )?,
            );
        }
        if let Some(normal) = &clearcoat.normal
            && !omitted.contains(&normal.texture.uv_layer)
        {
            extension.insert(
                "clearcoatNormalTexture".to_owned(),
                textured_normal_info(
                    normal,
                    tex_coords,
                    doc,
                    root_path,
                    texture_export,
                    binary,
                    views,
                )?,
            );
        }
        add_extension(
            &mut extensions,
            extensions_used,
            "KHR_materials_clearcoat",
            Value::Object(extension),
        );
    }
    if let Some(sheen) = &surface.sheen {
        let mut extension = serde_json::Map::new();
        if !crate::float::equal_f64_array(&sheen.color_factor, &[0.0; 3]) {
            extension.insert("sheenColorFactor".to_owned(), json!(sheen.color_factor));
        }
        if let Some(texture) = &sheen.color_texture
            && !omitted.contains(&texture.uv_layer)
        {
            extension.insert(
                "sheenColorTexture".to_owned(),
                texture_export.sampled_info(
                    texture,
                    reads::RGB,
                    tex_coords,
                    doc,
                    root_path,
                    binary,
                    views,
                )?,
            );
        }
        if !crate::float::equal_f64(sheen.roughness_factor, 0.0) {
            extension.insert(
                "sheenRoughnessFactor".to_owned(),
                json!(sheen.roughness_factor),
            );
        }
        if let Some(texture) = &sheen.roughness_texture
            && !omitted.contains(&texture.uv_layer)
        {
            extension.insert(
                "sheenRoughnessTexture".to_owned(),
                texture_export.sampled_info(
                    texture,
                    reads::A,
                    tex_coords,
                    doc,
                    root_path,
                    binary,
                    views,
                )?,
            );
        }
        add_extension(
            &mut extensions,
            extensions_used,
            "KHR_materials_sheen",
            Value::Object(extension),
        );
    }
    if let Some(specular) = &surface.specular {
        let mut extension = serde_json::Map::new();
        if !crate::float::equal_f64(specular.factor, 1.0) {
            extension.insert("specularFactor".to_owned(), json!(specular.factor));
        }
        if let Some(texture) = &specular.texture
            && !omitted.contains(&texture.uv_layer)
        {
            extension.insert(
                "specularTexture".to_owned(),
                texture_export.sampled_info(
                    texture,
                    reads::A,
                    tex_coords,
                    doc,
                    root_path,
                    binary,
                    views,
                )?,
            );
        }
        if !crate::float::equal_f64_array(&specular.color_factor, &[1.0; 3]) {
            extension.insert(
                "specularColorFactor".to_owned(),
                json!(specular.color_factor),
            );
        }
        if let Some(texture) = &specular.color_texture
            && !omitted.contains(&texture.uv_layer)
        {
            extension.insert(
                "specularColorTexture".to_owned(),
                texture_export.sampled_info(
                    texture,
                    reads::RGB,
                    tex_coords,
                    doc,
                    root_path,
                    binary,
                    views,
                )?,
            );
        }
        add_extension(
            &mut extensions,
            extensions_used,
            "KHR_materials_specular",
            Value::Object(extension),
        );
    }
    if let Some(anisotropy) = &surface.anisotropy {
        let mut extension = serde_json::Map::new();
        if !crate::float::equal_f64(anisotropy.strength, 0.0) {
            extension.insert("anisotropyStrength".to_owned(), json!(anisotropy.strength));
        }
        if !crate::float::equal_f64(anisotropy.rotation, 0.0) {
            extension.insert("anisotropyRotation".to_owned(), json!(anisotropy.rotation));
        }
        add_extension(
            &mut extensions,
            extensions_used,
            "KHR_materials_anisotropy",
            Value::Object(extension),
        );
    }
    if !extensions.is_empty() {
        value["extensions"] = Value::Object(extensions);
    }
    Ok(value)
}

fn add_extension(
    extensions: &mut serde_json::Map<String, Value>,
    extensions_used: &mut BTreeSet<String>,
    name: &str,
    value: Value,
) {
    extensions.insert(name.to_owned(), value);
    extensions_used.insert(name.to_owned());
}

/// `normalTexture` or `clearcoatNormalTexture` info; `scale` is written only when it is not 1.
fn textured_normal_info(
    normal: &TexturedNormal,
    tex_coords: &TexCoords,
    doc: &SceneDoc,
    root_path: &Path,
    texture_export: &mut TextureExport,
    binary: &mut Vec<u8>,
    views: &mut Vec<Value>,
) -> Result<Value> {
    let mut info = texture_export.sampled_info(
        &normal.texture,
        reads::RGB,
        tex_coords,
        doc,
        root_path,
        binary,
        views,
    )?;
    if !crate::float::equal_f64(normal.scale, 1.0) {
        info["scale"] = json!(normal.scale);
    }
    Ok(info)
}

/// `texCoord` map from each UV layer reference to its glTF TEXCOORD index.
pub(crate) type TexCoords = BTreeMap<Option<String>, usize>;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct TexCoordVariant {
    coords: TexCoords,
    omitted: BTreeSet<Option<String>>,
}

#[derive(Clone, Debug)]
struct SlotTexCoord {
    variant: TexCoordVariant,
    /// Member-present UV layers, or `None` when the mesh layout is not member-local.
    uv_available_mask: Option<u8>,
}

/// Empty `texCoord` map for a material no mesh uses: its references read `TEXCOORD_0`.
static NO_TEX_COORDS: TexCoords = BTreeMap::new();

/// One material's per-face-slot UV map; `coords` remains its first-used base variant for VRM.
pub(crate) struct TexCoordPlan {
    coords: BTreeMap<Id, TexCoords>,
    slot_coords: BTreeMap<Id, BTreeMap<usize, (Id, SlotTexCoord)>>,
    losses: Vec<Loss>,
    warnings: Vec<PotError>,
}

/// Losses of a glTF or VRM export of `doc` for mesh data glTF cannot write losslessly: generic and
/// Blender-only attributes, out-of-range carried colors, and vertex-color reads that name no layer.
pub(crate) fn mesh_export_losses(doc: &SceneDoc, snapshot: &Snapshot) -> Result<Vec<Loss>> {
    let surfaces = resolve_material_surfaces_with_snapshot(doc, snapshot)?;
    let visible = super::render_visible_nodes(doc, snapshot)?;
    let mut losses = Vec::new();
    let mut reported = BTreeSet::<(String, String)>::new();
    for (node_id, node) in &doc.nodes {
        if !node.visible || !node.render_visible || !visible.contains(node_id) {
            continue;
        }
        let Some(evaluated) = snapshot.meshes.get(node_id) else {
            continue;
        };
        if evaluated.vertices.is_empty() {
            continue;
        }
        let (mesh, owner, data) = if let Some(data_id) = &node.data {
            let Some(data_block) = doc
                .data_blocks
                .get(data_id)
                .filter(|data| data.data_type == "mesh")
            else {
                continue;
            };
            (
                source_mesh_for_export(node, data_block, evaluated, snapshot.purpose)?,
                data_id.to_string(),
                Some(data_block),
            )
        } else if instance_collection_target(node)?.is_some() {
            (evaluated, node_id.to_string(), None)
        } else {
            continue;
        };
        let found = unwritten_attribute_losses(mesh, &owner)
            .into_iter()
            .chain(explicit_zero_normal_loss(mesh, &owner)?)
            .chain(color_losses(
                node_id, node, snapshot, data, mesh, &owner, &surfaces,
            ));
        for loss in found {
            if reported.insert((owner.clone(), loss.reason.clone())) {
                losses.push(loss);
            }
        }
    }
    Ok(losses)
}

/// One loss of a mesh's data, owned by `owner` (a mesh data-block or an instancing node).
fn export_loss(owner: &str, feature: &str, reason: String, suggestion: &str) -> Loss {
    Loss {
        feature_id: feature.to_owned(),
        data_id: Some(owner.to_owned()),
        reason,
        suggestion: Some(suggestion.to_owned()),
    }
}

/// The `mesh.normals` loss of a mesh with an explicit-zero custom normal. glTF writes a zero normal
/// as the up-vector fallback, so the authored zero is not preserved. Automatic and authored codes
/// are written normally and are not losses.
fn explicit_zero_normal_loss(mesh: &Mesh, owner: &str) -> Result<Option<Loss>> {
    const SUGGESTION: &str =
        "export with --allow-lossy, or export as blend only if its loss is accepted";
    let codes = clnor::corner_codes(mesh)?.unwrap_or_default();
    if !codes.contains(&clnor::CornerNormalCode::ExplicitZero) {
        return Ok(None);
    }
    Ok(Some(export_loss(
        owner,
        "mesh.normals",
        format!(
            "glTF output writes a zero custom normal on `{owner}` as the up-vector fallback, so the authored explicit zero is not preserved"
        ),
        SUGGESTION,
    )))
}

/// Losses for the attributes of one mesh that glTF does not write: generic element attributes,
/// Blender custom properties, and Blender-only `blender_attributes` entries. Color data is reported
/// by [`color_losses`].
pub(crate) fn unwritten_attribute_losses(mesh: &Mesh, owner: &str) -> Vec<Loss> {
    const SUGGESTION: &str = "export as blend to keep mesh attributes";
    let store = color_layers::read(mesh);
    let uv_names = uv_layers::read_layers(mesh)
        .into_iter()
        .map(|layer| layer.name)
        .collect::<BTreeSet<_>>();
    let mut losses = Vec::new();
    for (key, value) in &mesh.attributes {
        match key.as_str() {
            "blender_metadata" => {
                if value["custom_properties"]
                    .as_object()
                    .is_some_and(|properties| !properties.is_empty())
                {
                    losses.push(export_loss(
                        owner,
                        "mesh.attribute",
                        format!(
                            "glTF export does not write Blender custom properties on mesh `{owner}`"
                        ),
                        SUGGESTION,
                    ));
                }
            }
            "blender_attributes" => {
                for entry in value.as_array().into_iter().flatten() {
                    let Some(entry_name) = entry["name"].as_str() else {
                        continue;
                    };
                    // Dot-prefixed entries are Blender selection and hide state. The bookkeeping
                    // entries mirror native data, and UV layers are stored as UV layers.
                    if entry_name.starts_with('.')
                        || BLENDER_BOOKKEEPING_ATTRIBUTES.contains(&entry_name)
                        || uv_names.contains(entry_name)
                        || (is_color_attribute(entry)
                            && store.layers.iter().any(|layer| layer.name == entry_name))
                    {
                        continue;
                    }
                    let (feature, kind) = if is_color_attribute(entry) {
                        ("mesh.color_attribute", "color attribute")
                    } else {
                        ("mesh.attribute", "attribute")
                    };
                    losses.push(export_loss(
                        owner,
                        feature,
                        format!("glTF export does not write Blender {kind} `{entry_name}` on mesh `{owner}`"),
                        SUGGESTION,
                    ));
                }
            }
            _ => {
                let Some(name) = attribute_keys::user_name(key) else {
                    if matches!(
                        key.as_str(),
                        "uv_layers"
                            | "color_layers"
                            | "custom_normal"
                            | "shade_smooth"
                            | "sharp"
                            | "seam"
                            | "vertex_groups"
                    ) || color_layer_storage_key(mesh, key)
                    {
                        continue;
                    }
                    losses.push(export_loss(
                        owner,
                        "mesh.attribute",
                        format!(
                            "glTF export does not write mesh attribute `{key}` on mesh `{owner}`"
                        ),
                        SUGGESTION,
                    ));
                    continue;
                };
                if store.layers.iter().any(|layer| layer.name == name) {
                    continue;
                }
                losses.push(export_loss(
                    owner,
                    "mesh.attribute",
                    format!("glTF export does not write mesh attribute `{name}` on mesh `{owner}`"),
                    SUGGESTION,
                ));
            }
        }
    }
    losses
}

fn slot_has_nonopaque_color_alpha(
    mesh: &Mesh,
    layer: &ColorLayer,
    slot: u32,
    vertex_indices: &mut Option<BTreeMap<u32, usize>>,
) -> bool {
    match layer.domain {
        ColorDomain::Corner => {
            let mut corner_index = 0;
            for face in &mesh.faces {
                for _ in &face.vertices {
                    let nonopaque = layer
                        .values
                        .get(corner_index)
                        .is_some_and(|color| !crate::float::equal_f64(color[3], 1.0));
                    if face.material_index == slot && nonopaque {
                        return true;
                    }
                    corner_index += 1;
                }
            }
            false
        }
        ColorDomain::Point => {
            let indices = vertex_indices.get_or_insert_with(|| {
                mesh.vertices
                    .iter()
                    .enumerate()
                    .map(|(index, vertex)| (vertex.id, index))
                    .collect()
            });
            mesh.faces
                .iter()
                .filter(|face| face.material_index == slot)
                .flat_map(|face| face.vertices.iter())
                .any(|vertex_id| {
                    indices
                        .get(vertex_id)
                        .and_then(|&index| layer.values.get(index))
                        .is_some_and(|color| !crate::float::equal_f64(color[3], 1.0))
                })
        }
    }
}

/// Color losses of one exported mesh. Application extras preserve every named layer and its
/// POINT/CORNER domain. glTF's core `COLOR_n` range is narrower than canonical HDR color values.
fn color_losses(
    node_id: &Id,
    node: &Node,
    snapshot: &Snapshot,
    data: Option<&DataBlock>,
    mesh: &Mesh,
    owner: &str,
    surfaces: &BTreeMap<Id, ResolvedSurface>,
) -> Vec<Loss> {
    const SUGGESTION: &str =
        "export with --allow-lossy, or export as blend to keep the original color values";
    let store = color_layers::read(mesh);
    let plan = color_plan(node_id, node, snapshot, data, mesh, surfaces);
    let mut losses = Vec::new();
    for index in plan.carried_layers() {
        let Some(layer) = store.layers.get(index) else {
            continue;
        };
        let out_of_range = layer
            .values
            .iter()
            .flatten()
            .filter(|component| **component < 0.0 || **component > 1.0)
            .count();
        if out_of_range > 0 {
            losses.push(export_loss(
                owner,
                "mesh.color_attribute",
                format!(
                    "layer `{}` has {out_of_range} RGBA component(s) outside glTF COLOR_n's [0, 1] range; its color accessor is clamped, while the original values remain in Potter extras",
                    layer.name
                ),
                SUGGESTION,
            ));
        }
    }
    let mut vertex_indices = None;
    for slot in &plan.translucent {
        let Some(layer) = plan
            .slot_layers
            .get(slot)
            .and_then(|&index| store.layers.get(index))
        else {
            continue;
        };
        if slot_has_nonopaque_color_alpha(mesh, layer, *slot, &mut vertex_indices) {
            losses.push(export_loss(
                owner,
                feature_id::VERTEX_COLOR,
                format!(
                    "glTF multiplies the alpha of `{}` into the base alpha of `{owner}` (slot {slot}); Blender ignores that alpha, so the output differs",
                    layer.name
                ),
                SUGGESTION,
            ));
        }
    }
    for (slot, reference) in &plan.missing {
        let layer = reference
            .as_deref()
            .map_or_else(|| "the render layer".to_owned(), |name| format!("`{name}`"));
        losses.push(export_loss(
            owner,
            feature_id::VERTEX_COLOR,
            format!("a material of `{owner}` (slot {slot}) reads {layer}, which the mesh does not have; its COLOR_0 is not written"),
            "read a color layer the mesh has, or give the mesh that layer",
        ));
    }
    losses
}

impl TexCoordPlan {
    /// The base material's `texCoord` map, retained for VRM material-index mappings.
    pub(crate) fn for_material(&self, material: &Id) -> &TexCoords {
        self.coords.get(material).unwrap_or(&NO_TEX_COORDS)
    }
    fn for_slot(&self, node_id: &Id, slot: usize) -> Option<&TexCoordVariant> {
        self.slot_coords
            .get(node_id)
            .and_then(|slots| slots.get(&slot))
            .map(|(_, slot_coords)| &slot_coords.variant)
    }
    fn slot_uv_available_mask(&self, node_id: &Id, slot: usize) -> Option<u8> {
        self.slot_coords
            .get(node_id)
            .and_then(|slots| slots.get(&slot))
            .and_then(|(_, slot_coords)| slot_coords.uv_available_mask)
    }
}

/// Resolves one texture selector using a collection member's layer availability when present.
fn binding_tex_coord(
    layers: &uv_layers::UvStore,
    reference: Option<&str>,
    binding: Option<&crate::eval::SlotBinding>,
) -> (Option<usize>, bool) {
    let Some(binding) = binding else {
        let (index, missing) = layers.resolve(reference);
        return (index, missing.is_some() || index.is_none());
    };
    let render_index = binding.uv.render_index.map(usize::from);
    match reference {
        None | Some("") => (render_index, render_index.is_none()),
        Some(name) => {
            let index = layers.index_of(name).filter(|index| {
                u8::try_from(*index)
                    .ok()
                    .and_then(|index| 1_u8.checked_shl(u32::from(index)))
                    .is_some_and(|bit| binding.uv.available_mask & bit != 0)
            });
            match index {
                Some(index) => (Some(index), false),
                None => (render_index, true),
            }
        }
    }
}

fn set_mtoon_texture_coord(
    extension: &mut Value,
    field: &str,
    texture: &TextureRef,
    coords: &TexCoords,
    omitted: &BTreeSet<Option<String>>,
) {
    if omitted.contains(&texture.uv_layer) {
        if let Some(fields) = extension.as_object_mut() {
            fields.remove(field);
        }
    } else if extension[field].is_object() {
        extension[field]["texCoord"] = json!(coords.get(&texture.uv_layer).copied().unwrap_or(0));
    }
}

fn set_mtoon_variant_uvs(
    extension: &mut Value,
    mtoon: &crate::model::AvatarMtoon,
    coords: &TexCoords,
    omitted: &BTreeSet<Option<String>>,
) {
    if let Some(texture) = &mtoon.shade_multiply_texture {
        set_mtoon_texture_coord(extension, "shadeMultiplyTexture", texture, coords, omitted);
    }
    if let Some(shift) = &mtoon.shading_shift_texture {
        set_mtoon_texture_coord(
            extension,
            "shadingShiftTexture",
            &shift.texture,
            coords,
            omitted,
        );
    }
    for (field, texture) in [
        ("rimMultiplyTexture", mtoon.rim_multiply_texture.as_ref()),
        (
            "outlineWidthMultiplyTexture",
            mtoon.outline_width_multiply_texture.as_ref(),
        ),
        (
            "uvAnimationMaskTexture",
            mtoon.uv_animation_mask_texture.as_ref(),
        ),
    ] {
        if let Some(texture) = texture {
            set_mtoon_texture_coord(extension, field, texture, coords, omitted);
        }
    }
}

/// Resolves UV layer references per used face slot and reports only actual missing-layer fallbacks.
pub(crate) fn tex_coord_plan(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    surfaces: &BTreeMap<Id, ResolvedSurface>,
    vrm: Option<(&AvatarSettings, AvatarVersion)>,
) -> Result<TexCoordPlan> {
    let mut references = BTreeMap::<Id, BTreeSet<Option<String>>>::new();
    for (material_id, surface) in surfaces {
        references.insert(
            material_id.clone(),
            surface_textures(surface)
                .into_iter()
                .map(|texture| texture.uv_layer.clone())
                .collect(),
        );
    }
    if let Some((avatar, _)) = vrm {
        for (material_id, mtoon) in &avatar.mtoon {
            let Some(set) = references.get_mut(material_id) else {
                continue;
            };
            set.extend(super::vrm::uv_textures(mtoon).map(|texture| texture.uv_layer.clone()));
        }
    }
    let visible = super::render_visible_nodes(doc, snapshot)?;
    let mut plan = TexCoordPlan {
        coords: BTreeMap::new(),
        slot_coords: BTreeMap::new(),
        losses: Vec::new(),
        warnings: Vec::new(),
    };
    let mut reported = BTreeSet::<(Id, Option<String>, Id, usize, &'static str)>::new();
    for (node_id, node) in &doc.nodes {
        if !node.visible || !node.render_visible || !visible.contains(node_id) {
            continue;
        }
        let Some(evaluated) = snapshot.meshes.get(node_id) else {
            continue;
        };
        if evaluated.vertices.is_empty() {
            continue;
        }
        let (mesh, data_block) = if let Some(data_id) = &node.data {
            let Some(data_block) = doc
                .data_blocks
                .get(data_id)
                .filter(|data| data.data_type == "mesh")
            else {
                continue;
            };
            (
                source_mesh_for_export(node, data_block, evaluated, snapshot.purpose)?,
                Some(data_block),
            )
        } else if instance_collection_target(node)?.is_some() {
            (evaluated, None)
        } else {
            continue;
        };
        let layers = uv_layers::read(mesh);
        let mut node_coords = BTreeMap::new();
        for (slot, material_id) in
            used_material_bindings(node_id, node, snapshot, data_block, mesh)?
        {
            let texture_refs = references.get(material_id).ok_or_else(|| {
                PotError::new(
                    ErrorCode::SceneInvalid,
                    "node material reference is missing",
                )
            })?;
            let binding = snapshot.slot_binding(node_id, node, slot);
            let uv_available_mask = binding.map(|binding| binding.uv.available_mask);
            let mut coords = TexCoords::new();
            let mut omitted = BTreeSet::new();
            for reference in texture_refs {
                let (index, missing) = binding_tex_coord(&layers, reference.as_deref(), binding);
                let index = index.and_then(|index| {
                    uv_available_mask.map_or(Some(index), |mask| {
                        let bit = u32::try_from(index)
                            .ok()
                            .and_then(|index| 1_u8.checked_shl(index))?;
                        (mask & bit != 0).then(|| (mask & (bit - 1)).count_ones() as usize)
                    })
                });
                if missing
                    && reported.insert((
                        material_id.clone(),
                        reference.clone(),
                        node_id.clone(),
                        slot,
                        feature_id::UV_LAYER_MISSING,
                    ))
                {
                    let layer = reference
                        .as_deref()
                        .filter(|name| !name.is_empty())
                        .map_or_else(|| "the render layer".to_owned(), |name| format!("`{name}`"));
                    if let Some(index) = index {
                        plan.warnings.push(PotError::with_details(
                            ErrorCode::UnrepresentableFeature,
                            format!(
                                "material `{material_id}` samples UV layer {layer}, which is unavailable to mesh `{node_id}` for this slot; the texture falls back to TEXCOORD_{index}"
                            ),
                            json!({
                                "feature_id": feature_id::UV_LAYER_MISSING,
                                "status": "fallback",
                                "material_id": material_id,
                                "node_id": node_id,
                                "uv_layer": reference,
                                "texcoord": index
                            }),
                        ));
                    } else {
                        plan.losses.push(Loss {
                            feature_id: feature_id::UV_LAYER_MISSING.to_owned(),
                            data_id: Some(material_id.to_string()),
                            reason: format!(
                                "material `{material_id}` samples UV layer {layer}, but mesh `{node_id}` has no UV layer available to this texture for the slot"
                            ),
                            suggestion: Some(
                                "create a UV layer for the textured mesh or choose a material texture layer that exists".to_owned(),
                            ),
                        });
                    }
                }
                if let Some(index) = index {
                    coords.insert(reference.clone(), index);
                } else {
                    omitted.insert(reference.clone());
                }
            }
            if !plan.coords.contains_key(material_id) {
                plan.coords.insert(material_id.clone(), coords.clone());
            }
            node_coords.insert(
                slot,
                (
                    material_id.clone(),
                    SlotTexCoord {
                        variant: TexCoordVariant { coords, omitted },
                        uv_available_mask,
                    },
                ),
            );
        }
        if !node_coords.is_empty() {
            plan.slot_coords.insert(node_id.clone(), node_coords);
        }
    }
    // VRM 0.x MToon textures carry no `texCoord`, so they sample TEXCOORD_0.
    let mut v0_losses = Vec::new();
    if let Some((avatar, AvatarVersion::V0)) = vrm {
        for (material_id, mtoon) in &avatar.mtoon {
            let Some(surface) = surfaces.get(material_id) else {
                continue;
            };
            let coords = plan.for_material(material_id);
            let layers = surface_textures(surface)
                .into_iter()
                .map(|texture| texture.uv_layer.clone())
                .chain(super::vrm::uv_textures(mtoon).map(|texture| texture.uv_layer.clone()))
                .collect::<BTreeSet<_>>();
            for layer in layers {
                let index = plan
                    .slot_coords
                    .values()
                    .flat_map(|slots| slots.values())
                    .filter(|(bound_material, _)| bound_material == material_id)
                    .filter_map(|(_, slot_coords)| slot_coords.variant.coords.get(&layer).copied())
                    .find(|index| *index != 0)
                    .or_else(|| coords.get(&layer).copied())
                    .unwrap_or(0);
                if index != 0 {
                    let name = layer
                        .as_deref()
                        .map_or_else(|| "the render layer".to_owned(), |name| format!("`{name}`"));
                    v0_losses.push(Loss {
                        feature_id: feature_id::VRM_V0_MTOON_UV_LAYER.to_owned(),
                        data_id: Some(material_id.to_string()),
                        reason: format!(
                            "VRM 0.x MToon textures sample TEXCOORD_0; material `{material_id}` samples UV layer {name} (TEXCOORD_{index}), which exports from TEXCOORD_0"
                        ),
                        suggestion: Some(
                            "export VRM 1.0, or sample these textures from the first UV layer".to_owned(),
                        ),
                    });
                }
            }
        }
    }
    plan.losses.extend(v0_losses);
    Ok(plan)
}

/// UV losses and fallback warnings from one resolved material/slot plan.
pub(crate) fn uv_layer_diagnostics(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    vrm: Option<(&AvatarSettings, AvatarVersion)>,
) -> Result<(Vec<Loss>, Vec<PotError>)> {
    let surfaces = resolve_material_surfaces_with_snapshot(doc, snapshot)?;
    let plan = tex_coord_plan(doc, snapshot, &surfaces, vrm)?;
    Ok((plan.losses, plan.warnings))
}

fn remove_vrm0_texture_property(
    property: &mut Value,
    field: &str,
    uv_layer: Option<&Option<String>>,
    omitted: &BTreeSet<Option<String>>,
) {
    if uv_layer.is_some_and(|uv_layer| omitted.contains(uv_layer))
        && let Some(textures) = property["textureProperties"].as_object_mut()
    {
        textures.remove(field);
    }
}

fn remove_vrm0_variant_textures(
    property: &mut Value,
    surface: &ResolvedSurface,
    mtoon: Option<&crate::model::AvatarMtoon>,
    omitted: &BTreeSet<Option<String>>,
) {
    for (field, uv_layer) in [
        (
            "_MainTex",
            surface
                .base_color
                .texture
                .as_ref()
                .map(|texture| &texture.uv_layer),
        ),
        (
            "_BumpMap",
            surface
                .normal
                .as_ref()
                .map(|normal| &normal.texture.uv_layer),
        ),
        (
            "_EmissionMap",
            surface
                .emissive
                .texture
                .as_ref()
                .map(|texture| &texture.uv_layer),
        ),
    ] {
        remove_vrm0_texture_property(property, field, uv_layer, omitted);
    }
    if let Some(mtoon) = mtoon {
        for (field, uv_layer) in [
            (
                "_ShadeTexture",
                mtoon
                    .shade_multiply_texture
                    .as_ref()
                    .map(|texture| &texture.uv_layer),
            ),
            (
                "_RimTexture",
                mtoon
                    .rim_multiply_texture
                    .as_ref()
                    .map(|texture| &texture.uv_layer),
            ),
            (
                "_OutlineWidthTexture",
                mtoon
                    .outline_width_multiply_texture
                    .as_ref()
                    .map(|texture| &texture.uv_layer),
            ),
            (
                "_UvAnimMaskTexture",
                mtoon
                    .uv_animation_mask_texture
                    .as_ref()
                    .map(|texture| &texture.uv_layer),
            ),
        ] {
            remove_vrm0_texture_property(property, field, uv_layer, omitted);
        }
    }
}

fn preserve_vrm0_expression_variants(
    root: &mut Value,
    variant_names: &BTreeMap<String, Vec<String>>,
) -> Result<()> {
    let groups = root["extensions"]["VRM"]["blendShapeMaster"]["blendShapeGroups"]
        .as_array_mut()
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "VRM 0 expression groups are missing",
            )
        })?;
    for group in groups {
        let Some(values) = group["materialValues"].as_array_mut() else {
            continue;
        };
        let original = values.clone();
        for value in original {
            let Some(name) = value["materialName"].as_str() else {
                continue;
            };
            let Some(names) = variant_names.get(name) else {
                continue;
            };
            for name in names {
                let mut variant_value = value.clone();
                variant_value["materialName"] = json!(name);
                values.push(variant_value);
            }
        }
    }
    Ok(())
}
fn preserve_vrm_material_variants(
    version: AvatarVersion,
    surfaces: &BTreeMap<Id, ResolvedSurface>,
    avatar: &AvatarSettings,
    material_indices: &BTreeMap<Id, usize>,
    material_variant_indices: &BTreeMap<Id, BTreeMap<TexCoordVariant, usize>>,
    root: &mut Value,
    base_material_count: usize,
) -> Result<()> {
    let mut variants = material_variant_indices
        .iter()
        .flat_map(|(material_id, uv_variants)| {
            uv_variants
                .iter()
                .map(move |(coords, index)| (index, material_id, coords))
        })
        .filter(|(index, _, _)| **index >= base_material_count)
        .collect::<Vec<_>>();
    variants.sort_by_key(|(index, _, _)| **index);
    if variants.is_empty() {
        return Ok(());
    }
    match version {
        AvatarVersion::V0 => {
            let mut used_names = root["materials"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|material| material["name"].as_str().map(str::to_owned))
                .collect::<BTreeSet<_>>();
            let mut variant_names = BTreeMap::<String, Vec<String>>::new();
            for (index, material_id, variant) in variants {
                let base_index = material_indices.get(material_id).copied().ok_or_else(|| {
                    PotError::new(
                        ErrorCode::InternalError,
                        "VRM base material index is missing",
                    )
                })?;
                let base_name = root["materials"][base_index]["name"]
                    .as_str()
                    .map_or_else(|| material_id.to_string(), str::to_owned);
                let mut variant_name = format!("{base_name}__potter_uv_variant_{index}");
                while !used_names.insert(variant_name.clone()) {
                    variant_name.push('_');
                }
                let surface = surfaces.get(material_id).ok_or_else(|| {
                    PotError::new(ErrorCode::SceneInvalid, "node material surface is missing")
                })?;
                let mut property = root["extensions"]["VRM"]["materialProperties"]
                    .get(base_index)
                    .cloned()
                    .ok_or_else(|| {
                        PotError::new(
                            ErrorCode::InternalError,
                            "VRM base material properties are missing",
                        )
                    })?;
                remove_vrm0_variant_textures(
                    &mut property,
                    surface,
                    avatar.mtoon.get(material_id),
                    &variant.omitted,
                );
                property["name"] = json!(variant_name.clone());
                {
                    let properties = root["extensions"]["VRM"]["materialProperties"]
                        .as_array_mut()
                        .ok_or_else(|| {
                            PotError::new(
                                ErrorCode::InternalError,
                                "VRM 0 material properties are missing",
                            )
                        })?;
                    if properties.len() != *index {
                        return Err(PotError::new(
                            ErrorCode::InternalError,
                            "VRM variant material properties are not contiguous",
                        ));
                    }
                    properties.push(property);
                }
                root["materials"][*index]["name"] = json!(variant_name.clone());
                variant_names
                    .entry(base_name)
                    .or_default()
                    .push(variant_name);
            }
            preserve_vrm0_expression_variants(root, &variant_names)?;
        }
        AvatarVersion::V1 => {
            for (material_id, mtoon) in &avatar.mtoon {
                let base_index = material_indices.get(material_id).copied().ok_or_else(|| {
                    PotError::new(
                        ErrorCode::InternalError,
                        "VRM base material index is missing",
                    )
                })?;
                let base_extension =
                    root["materials"][base_index]["extensions"]["VRMC_materials_mtoon"].clone();
                if !base_extension.is_object() {
                    continue;
                }
                for (index, variant_material, coords) in &variants {
                    if *variant_material != material_id || **index == base_index {
                        continue;
                    }
                    let mut extension = base_extension.clone();
                    set_mtoon_variant_uvs(&mut extension, mtoon, &coords.coords, &coords.omitted);
                    root["materials"][**index]["extensions"]["VRMC_materials_mtoon"] = extension;
                }
            }
        }
    }
    if matches!(version, AvatarVersion::V1) {
        preserve_vrm_v1_expression_variants(root, material_indices, material_variant_indices);
    }

    Ok(())
}

fn preserve_vrm_v1_expression_variants(
    root: &mut Value,
    material_indices: &BTreeMap<Id, usize>,
    material_variant_indices: &BTreeMap<Id, BTreeMap<TexCoordVariant, usize>>,
) {
    for expression_group in ["preset", "custom"] {
        let Some(expressions) =
            root["extensions"]["VRMC_vrm"]["expressions"][expression_group].as_object_mut()
        else {
            continue;
        };
        for expression in expressions.values_mut() {
            for field in ["materialColorBinds", "textureTransformBinds"] {
                let Some(binds) = expression[field].as_array_mut() else {
                    continue;
                };
                let original = binds.clone();
                for bind in original {
                    let Some(source_index) = bind["material"]
                        .as_u64()
                        .and_then(|value| usize::try_from(value).ok())
                    else {
                        continue;
                    };
                    let Some((material_id, _)) = material_indices
                        .iter()
                        .find(|(_, index)| **index == source_index)
                    else {
                        continue;
                    };
                    let Some(variants) = material_variant_indices.get(material_id) else {
                        continue;
                    };
                    let mut indexes = variants
                        .values()
                        .copied()
                        .filter(|index| *index != source_index)
                        .collect::<Vec<_>>();
                    indexes.sort_unstable();
                    indexes.dedup();
                    for index in indexes {
                        let mut variant_bind = bind.clone();
                        variant_bind["material"] = json!(index);
                        binds.push(variant_bind);
                    }
                }
            }
        }
    }
}

fn validate_shape_key_export(shape_keys: &ShapeKeyData) -> Result<()> {
    if shape_keys.absolute && !shape_keys.keys.is_empty() {
        return Err(unsupported_export(
            "mesh.shape_key_absolute",
            "glTF morph targets cannot preserve Potter absolute shape-key timing",
        ));
    }
    Ok(())
}

fn validate_skin_binding_export(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    data_id: &Id,
    data: &DataBlock,
    binding: &SkinBinding,
    data_skin_bindings: &mut BTreeMap<Id, SkinBinding>,
    rest_world_cache: &mut BTreeMap<Id, BTreeMap<Id, DMat4>>,
) -> Result<()> {
    if binding.joint_ids.is_empty() {
        return Err(unsupported_export(
            "rig.skin_empty_armature",
            "glTF skins require at least one armature bone",
        ));
    }
    if binding.joint_ids.len() > usize::from(u16::MAX) + 1 {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "glTF skin has more joints than JOINTS_0 can index",
        ));
    }
    if let Some(previous) = data_skin_bindings.get(data_id) {
        if previous.armature_node != binding.armature_node
            || previous.joint_ids != binding.joint_ids
        {
            return Err(unsupported_export(
                "rig.shared_mesh_multiple_skeletons",
                "one shared mesh Data-Block cannot use different glTF skin joint layouts",
            ));
        }
    } else {
        data_skin_bindings.insert(data_id.clone(), binding.clone());
    }
    if data
        .shape_keys
        .as_ref()
        .is_some_and(|shape_keys| !shape_keys.keys.is_empty())
    {
        validate_skin_shape_key_rest(doc, snapshot, binding, rest_world_cache)?;
    }
    Ok(())
}

fn validate_vrm_texture_images(doc: &SceneDoc, avatar: &AvatarSettings) -> Result<()> {
    for mtoon in avatar.mtoon.values() {
        for texture in super::vrm::uv_textures(mtoon).chain(mtoon.matcap_texture.iter()) {
            texture_image_for_export(doc, &texture.image)?;
        }
    }
    Ok(())
}

fn instance_collection_target(node: &Node) -> Result<Option<Id>> {
    let Some(value) = node.properties.get("instance_collection") else {
        return Ok(None);
    };
    let target = value.as_str().ok_or_else(|| {
        PotError::new(
            ErrorCode::SceneInvalid,
            "collection instance target must be a collection ID",
        )
    })?;
    let target = Id::new(target).map_err(|_| {
        PotError::new(
            ErrorCode::SceneInvalid,
            "collection instance target is not a valid ID",
        )
    })?;
    Ok(Some(target))
}
pub(crate) fn validate_vrm_mesh_references(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    avatar: &AvatarSettings,
    frames: &VrmFrameMaps,
    version: AvatarVersion,
) -> Result<()> {
    validate_vrm_texture_images(doc, avatar)?;
    let visible = super::render_visible_nodes(doc, snapshot)?;
    let mut rest_world_cache = BTreeMap::<Id, BTreeMap<Id, DMat4>>::new();
    let mut data_skin_bindings = BTreeMap::<Id, SkinBinding>::new();
    for (node_id, node) in &doc.nodes {
        if !node.visible || !node.render_visible || !visible.contains(node_id) {
            continue;
        }
        let Some(data_id) = &node.data else {
            continue;
        };
        let Some(data) = doc
            .data_blocks
            .get(data_id)
            .filter(|data| data.data_type == "mesh")
        else {
            continue;
        };
        let Some(binding) = skin_binding(doc, node, snapshot.purpose)? else {
            continue;
        };
        validate_skin_binding_export(
            doc,
            snapshot,
            data_id,
            data,
            &binding,
            &mut data_skin_bindings,
            &mut rest_world_cache,
        )?;
    }
    let mut exported_mesh_nodes = BTreeSet::new();
    for (node_id, node) in &doc.nodes {
        if !node.visible || !node.render_visible || !visible.contains(node_id) {
            continue;
        }
        let Some(evaluated) = snapshot.meshes.get(node_id) else {
            continue;
        };
        if evaluated.vertices.is_empty() {
            continue;
        }
        let (mesh, data_block) = if let Some(data_id) = &node.data {
            let Some(data_block) = doc
                .data_blocks
                .get(data_id)
                .filter(|data| data.data_type == "mesh")
            else {
                continue;
            };
            (
                source_mesh_for_export(node, data_block, evaluated, snapshot.purpose)?,
                Some(data_block),
            )
        } else if instance_collection_target(node)?.is_some() {
            (evaluated, None)
        } else {
            continue;
        };
        validate_mesh_export_attributes(mesh, data_block)?;
        if let Some(shape_keys) = data_block
            .and_then(|data| data.shape_keys.as_ref())
            .filter(|shape_keys| !shape_keys.keys.is_empty())
        {
            validate_shape_key_export(shape_keys)?;
        }
        let scale = frames
            .mesh_geometry_scales
            .get(node_id)
            .copied()
            .unwrap_or(DVec3::ONE);
        let export = mesh_export_layout(mesh, data_block, scale, avatar_basis(version), &[])?;
        if let (Some(data_id), Some(data_block)) = (node.data.as_ref(), data_block)
            && let Some(binding) = data_skin_bindings.get(data_id)
        {
            skin_influences(
                node_id,
                data_block,
                mesh,
                &export.layout.source_vertex_indices,
                binding,
                doc,
                true,
            )?;
        }
        exported_mesh_nodes.insert(node_id.clone());
    }
    for annotation in avatar
        .first_person
        .iter()
        .flat_map(|first_person| &first_person.mesh_annotations)
    {
        if !exported_mesh_nodes.contains(&annotation.node) {
            return Err(unsupported_export(
                "vrm.first_person_mesh_export",
                "First-person mesh annotation target is not emitted as a glTF mesh; ensure it is visible and renderable.",
            ));
        }
    }
    for bind in avatar
        .expressions
        .values()
        .flat_map(|expression| &expression.morph_target_binds)
    {
        if !exported_mesh_nodes.contains(&bind.node) {
            return Err(unsupported_export(
                "vrm.expression_morph_mesh_export",
                "Expression morph target node is not emitted as a glTF mesh; ensure it is visible and renderable.",
            ));
        }
    }
    Ok(())
}

fn export_with_root_inner(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    root_path: &Path,
    bin_uri: Option<&str>,
    vrm: Option<(&AvatarSettings, AvatarVersion)>,
    precomputed_frame_maps: Option<&VrmFrameMaps>,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let inverse_basis = POT_TO_GLTF.inverse();
    let mut binary = Vec::new();
    let mut views = Vec::<Value>::new();
    let mut accessors = Vec::<Value>::new();
    let mut extensions_used = BTreeSet::new();
    let mut material_indices = BTreeMap::<Id, usize>::new();
    let mut materials = Vec::<Value>::new();
    let mut texture_export = TextureExport::new();
    let surfaces = resolve_material_surfaces_with_snapshot(doc, snapshot)?;
    let tex_coords = tex_coord_plan(doc, snapshot, &surfaces, vrm)?;
    let no_omitted_texture_layers = BTreeSet::new();
    for (id, material) in &doc.materials {
        material_indices.insert(id.clone(), materials.len());
        let surface = surfaces.get(id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "material surface is missing")
        })?;
        let value = material_value(
            id,
            material,
            surface,
            tex_coords.for_material(id),
            &no_omitted_texture_layers,
            doc,
            root_path,
            &mut texture_export,
            &mut binary,
            &mut views,
            &mut extensions_used,
        )?;
        materials.push(value);
    }
    let mut material_variant_indices = BTreeMap::<Id, BTreeMap<TexCoordVariant, usize>>::new();
    for id in doc.materials.keys() {
        let index = material_indices.get(id).copied().ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "base material index is missing")
        })?;
        material_variant_indices
            .entry(id.clone())
            .or_default()
            .insert(
                TexCoordVariant {
                    coords: tex_coords.for_material(id).clone(),
                    omitted: BTreeSet::new(),
                },
                index,
            );
    }
    for slots in tex_coords.slot_coords.values() {
        for (material_id, slot_coords) in slots.values() {
            let variant = &slot_coords.variant;
            if material_variant_indices
                .get(material_id)
                .is_some_and(|variants| variants.contains_key(variant))
            {
                continue;
            }
            let material = doc.materials.get(material_id).ok_or_else(|| {
                PotError::new(
                    ErrorCode::SceneInvalid,
                    "node material reference is missing",
                )
            })?;
            let surface = surfaces.get(material_id).ok_or_else(|| {
                PotError::new(ErrorCode::SceneInvalid, "node material surface is missing")
            })?;
            let value = material_value(
                material_id,
                material,
                surface,
                &variant.coords,
                &variant.omitted,
                doc,
                root_path,
                &mut texture_export,
                &mut binary,
                &mut views,
                &mut extensions_used,
            )?;
            let index = materials.len();
            materials.push(value);
            material_variant_indices
                .entry(material_id.clone())
                .or_default()
                .insert(variant.clone(), index);
        }
    }
    let visible_node_ids = super::render_visible_nodes(doc, snapshot)?;
    let node_ids = doc.nodes.keys().cloned().collect::<Vec<_>>();
    let node_indices = node_ids
        .iter()
        .enumerate()
        .map(|(index, id)| (id.clone(), index))
        .collect::<BTreeMap<_, _>>();
    let mut meshes = Vec::<Value>::new();
    let mut mesh_indices = BTreeMap::<(Id, Option<[u64; 3]>, Vec<Option<Id>>), usize>::new();
    let mut instance_mesh_indices =
        BTreeMap::<(Id, Option<[u64; 3]>, Vec<Option<Id>>), usize>::new();
    let mut node_mesh_indices = BTreeMap::<Id, usize>::new();
    let mut skin_bindings = BTreeMap::<Id, SkinBinding>::new();
    let mut data_skin_bindings = BTreeMap::<Id, SkinBinding>::new();
    let mut rest_world_cache = BTreeMap::<Id, BTreeMap<Id, DMat4>>::new();
    for (node_id, node) in &doc.nodes {
        if !node.visible || !node.render_visible || !visible_node_ids.contains(node_id) {
            continue;
        }
        let Some(data_id) = &node.data else {
            continue;
        };
        let Some(data) = doc
            .data_blocks
            .get(data_id)
            .filter(|data| data.data_type == "mesh")
        else {
            continue;
        };
        let Some(binding) = skin_binding(doc, node, snapshot.purpose)? else {
            continue;
        };
        validate_skin_binding_export(
            doc,
            snapshot,
            data_id,
            data,
            &binding,
            &mut data_skin_bindings,
            &mut rest_world_cache,
        )?;
        skin_bindings.insert(node_id.clone(), binding);
    }
    let avatar_armature_nodes = vrm.map(|(avatar, _)| super::vrm::referenced_armatures(avatar));
    let frame_maps = if vrm.is_some() {
        Some(precomputed_frame_maps.ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "VRM output frame maps are missing",
            )
        })?)
    } else {
        None
    };
    let geometry_basis = vrm.map_or(POT_TO_GLTF, |(_, version)| avatar_basis(version));
    for (node_id, node) in &doc.nodes {
        if !node.visible || !node.render_visible || !visible_node_ids.contains(node_id) {
            continue;
        }
        let Some(evaluated) = snapshot.meshes.get(node_id) else {
            continue;
        };
        if evaluated.vertices.is_empty() {
            continue;
        }
        let (mesh, data_id, instance_target, data_block) = if let Some(data_id) = &node.data {
            let Some(data_block) = doc
                .data_blocks
                .get(data_id)
                .filter(|data| data.data_type == "mesh")
            else {
                continue;
            };
            let mesh = source_mesh_for_export(node, data_block, evaluated, snapshot.purpose)?;
            (mesh, Some(data_id), None, Some(data_block))
        } else if let Some(target) = instance_collection_target(node)? {
            (evaluated, None, Some(target), None)
        } else {
            continue;
        };
        validate_mesh_export_attributes(mesh, data_block)?;
        let geometry_scale = frame_maps
            .and_then(|frames| frames.mesh_geometry_scales.get(node_id))
            .copied()
            .unwrap_or(DVec3::ONE);
        let mesh_variant = frame_maps.map(|_| {
            [
                geometry_scale.x.to_bits(),
                geometry_scale.y.to_bits(),
                geometry_scale.z.to_bits(),
            ]
        });
        let material_slots = material_slot_key(node_id, node, snapshot, data_block);
        if let Some(data_id) = data_id
            && let Some(index) =
                mesh_indices.get(&(data_id.clone(), mesh_variant, material_slots.clone()))
        {
            node_mesh_indices.insert(node_id.clone(), *index);
            continue;
        }
        if let Some(target) = &instance_target
            && let Some(index) =
                instance_mesh_indices.get(&(target.clone(), mesh_variant, material_slots.clone()))
        {
            node_mesh_indices.insert(node_id.clone(), *index);
            continue;
        }
        let mesh_name = data_id.map_or_else(
            || {
                format!(
                    "instance_{}",
                    instance_target.as_ref().map_or("", Id::as_str)
                )
            },
            std::string::ToString::to_string,
        );
        let colors = color_plan(node_id, node, snapshot, data_block, mesh, &surfaces);
        let color_metadata = potter_color_attributes(mesh);
        let carried_color_layers = colors.carried_layers();
        let MeshExportLayout { layout, key_deltas } = mesh_export_layout(
            mesh,
            data_block,
            geometry_scale,
            geometry_basis,
            &carried_color_layers,
        )?;
        let uv_sets = &layout.uvs;
        if let Some(shape_keys) = data_block
            .and_then(|data| data.shape_keys.as_ref())
            .filter(|shape_keys| !shape_keys.keys.is_empty())
        {
            validate_shape_key_export(shape_keys)?;
        }
        let mut positions = Vec::with_capacity(layout.source_vertex_indices.len());
        let mut minimum = [f32::INFINITY; 3];
        let mut maximum = [f32::NEG_INFINITY; 3];
        for source_index in &layout.source_vertex_indices {
            let vertex = &mesh.vertices[*source_index];
            let source_point = data_block
                .and_then(|data| data.shape_keys.as_ref())
                .filter(|shape_keys| !shape_keys.keys.is_empty())
                .and_then(|shape_keys| shape_keys.basis.get(&vertex.id))
                .map_or(vertex.co, |point| DVec3::from_array(*point));
            let point = geometry_scale * geometry_basis.transform_point3(source_point);
            let value = [to_f32(point.x)?, to_f32(point.y)?, to_f32(point.z)?];
            for axis in 0..3 {
                minimum[axis] = minimum[axis].min(value[axis]);
                maximum[axis] = maximum[axis].max(value[axis]);
            }
            positions.push(value);
        }
        align4(&mut binary);
        let position_offset = binary.len();
        for position in &positions {
            for component in position {
                binary.extend_from_slice(&component.to_le_bytes());
            }
        }
        let position_view = push_view(
            &mut views,
            position_offset,
            binary.len() - position_offset,
            Some(34962),
        );
        let position_accessor = accessors.len();
        accessors.push(json!({
            "bufferView": position_view,
            "componentType": 5126,
            "count": positions.len(),
            "type": "VEC3",
            "min": minimum,
            "max": maximum
        }));
        let mut attributes = json!({ "POSITION": position_accessor });
        for (set_index, uv_values) in uv_sets.iter().enumerate() {
            let mut floats = Vec::with_capacity(uv_values.len().saturating_mul(2));
            for uv in uv_values {
                floats.push(to_f32(uv[0])?);
                floats.push(to_f32(uv[1])?);
            }
            let accessor = append_float_accessor(
                &mut binary,
                &mut views,
                &mut accessors,
                &floats,
                uv_values.len(),
                "VEC2",
                None,
                None,
            );
            attributes[format!("TEXCOORD_{set_index}").as_str()] = json!(accessor);
        }
        let mut normal_floats = Vec::with_capacity(layout.normals[0].len().saturating_mul(3));
        for normal in &layout.normals[0] {
            for component in normal.to_array() {
                normal_floats.push(to_f32(component)?);
            }
        }
        let normal_accessor = append_float_accessor(
            &mut binary,
            &mut views,
            &mut accessors,
            &normal_floats,
            layout.normals[0].len(),
            "VEC3",
            None,
            None,
        );
        attributes["NORMAL"] = json!(normal_accessor);
        // Carried logical layers use the same dense order as their COLOR_n attributes.
        let mut color_accessors = BTreeMap::<usize, usize>::new();
        for (layer, values) in carried_color_layers.iter().copied().zip(&layout.colors) {
            let mut floats = Vec::with_capacity(values.len().saturating_mul(4));
            for color in values {
                for component in color {
                    floats.push(to_f32((*component).clamp(0.0, 1.0))?);
                }
            }
            let accessor = append_float_accessor(
                &mut binary,
                &mut views,
                &mut accessors,
                &floats,
                values.len(),
                "VEC4",
                None,
                None,
            );
            color_accessors.insert(layer, accessor);
        }
        if let (Some(data_id), Some(data_block)) = (data_id, data_block)
            && let Some(binding) = data_skin_bindings.get(data_id)
        {
            let influences = skin_influences(
                node_id,
                data_block,
                mesh,
                &layout.source_vertex_indices,
                binding,
                doc,
                vrm.is_some(),
            )?;
            let accessors_for_sets =
                append_skin_attributes(&influences, &mut binary, &mut views, &mut accessors)?;
            for (set_index, (joints_accessor, weights_accessor)) in
                accessors_for_sets.into_iter().enumerate()
            {
                attributes[format!("JOINTS_{set_index}")] = json!(joints_accessor);
                attributes[format!("WEIGHTS_{set_index}")] = json!(weights_accessor);
            }
        }
        let mirrored = geometry_scale.x * geometry_scale.y * geometry_scale.z < 0.0;
        let triangles = export_triangles(mesh).map_err(|error| {
            PotError::with_details(
                ErrorCode::ExportFailed,
                "mesh triangulation failed",
                json!({ "reason": error.to_string(), "data_id": data_id }),
            )
        })?;
        let mut slot_indices = BTreeMap::<u32, Vec<u32>>::new();
        let mut face_slots = Vec::with_capacity(mesh.faces.len());
        let mut triangle_offset = 0;
        for face in &mesh.faces {
            let triangle_count = face.vertices.len().saturating_sub(2);
            let face_triangles = triangles
                .get(triangle_offset..triangle_offset + triangle_count)
                .ok_or_else(|| {
                    PotError::new(ErrorCode::InternalError, "mesh triangulation is incomplete")
                })?;
            let face_indices = layout.face_indices.get(&face.id);
            let slot = face.material_index;
            face_slots.push(slot);
            let indices = slot_indices.entry(slot).or_default();
            for triangle in face_triangles {
                let mut corners = [0_u32; 3];
                for (corner, id) in corners.iter_mut().zip(triangle) {
                    *corner = face_indices
                        .and_then(|indices| indices.get(id))
                        .copied()
                        .ok_or_else(|| {
                            PotError::new(
                                ErrorCode::SceneInvalid,
                                "triangle references missing output vertex",
                            )
                        })?;
                }
                // A negative baked determinant mirrors the surface; reverse winding so front faces match NORMAL.
                if mirrored {
                    corners.swap(1, 2);
                }
                indices.extend_from_slice(&corners);
            }
            triangle_offset += triangle_count;
        }
        if triangle_offset != triangles.len() {
            return Err(PotError::new(
                ErrorCode::InternalError,
                "mesh triangulation has unexpected triangles",
            ));
        }
        // A face's primitive is its slot's position among the primitives written below.
        let primitive_of_slot = slot_indices
            .keys()
            .enumerate()
            .map(|(index, slot)| (*slot, index))
            .collect::<BTreeMap<_, _>>();
        let face_primitives = face_slots
            .iter()
            .map(|slot| {
                primitive_of_slot
                    .get(slot)
                    .and_then(|index| u32::try_from(*index).ok())
                    .ok_or_else(|| {
                        PotError::new(
                            ErrorCode::InternalError,
                            "face material slot has no primitive",
                        )
                    })
            })
            .collect::<Result<Vec<u32>>>()?;
        let polygons = if mesh.faces.is_empty() {
            None
        } else {
            Some(export_polygons(
                mesh,
                &layout,
                mirrored,
                &face_primitives,
                slot_indices.len(),
                &mut binary,
                &mut views,
            )?)
        };
        if slot_indices.is_empty() {
            slot_indices.insert(0, Vec::new());
        }
        let mut primitives = Vec::with_capacity(slot_indices.len());
        for (slot, indices) in &slot_indices {
            align4(&mut binary);
            let index_offset = binary.len();
            for index in indices {
                binary.extend_from_slice(&index.to_le_bytes());
            }
            let index_view = push_view(
                &mut views,
                index_offset,
                binary.len() - index_offset,
                Some(34963),
            );
            let index_accessor = accessors.len();
            accessors.push(json!({
                "bufferView": index_view,
                "componentType": 5125,
                "count": indices.len(),
                "type": "SCALAR"
            }));
            let mut primitive =
                json!({ "attributes": attributes.clone(), "indices": index_accessor, "mode": 4 });
            // The primitive's selected layer is COLOR_0; its source name identifies the Potter material read.
            for (set, layer) in colors.attributes(*slot) {
                let accessor = color_accessors.get(&layer).copied().ok_or_else(|| {
                    PotError::new(ErrorCode::InternalError, "color accessor is missing")
                })?;
                primitive["attributes"][format!("COLOR_{set}")] = json!(accessor);
                if set == 0 {
                    let layer_name = color_metadata["layers"][layer]["name"]
                        .as_str()
                        .ok_or_else(|| {
                            PotError::new(ErrorCode::InternalError, "color layer name is missing")
                        })?;
                    put_potter_primitive_color_layer(&mut primitive, layer_name)?;
                }
            }
            let slot = usize::try_from(*slot).map_err(|_| {
                PotError::new(
                    ErrorCode::SceneInvalid,
                    "mesh material slot exceeds platform range",
                )
            })?;
            if let Some(uv_available_mask) = tex_coords.slot_uv_available_mask(node_id, slot) {
                let source_attributes = attributes.as_object().ok_or_else(|| {
                    PotError::new(ErrorCode::InternalError, "glTF mesh attributes are missing")
                })?;
                let primitive_attributes =
                    primitive["attributes"].as_object_mut().ok_or_else(|| {
                        PotError::new(ErrorCode::InternalError, "primitive attributes are missing")
                    })?;
                primitive_attributes.retain(|name, _| !name.starts_with("TEXCOORD_"));
                let mut target = 0;
                for (source, _) in uv_sets.iter().enumerate() {
                    if uv_available_mask & (1_u8 << source) == 0 {
                        continue;
                    }
                    let source_name = format!("TEXCOORD_{source}");
                    let target_name = format!("TEXCOORD_{target}");
                    let accessor =
                        source_attributes
                            .get(&source_name)
                            .cloned()
                            .ok_or_else(|| {
                                PotError::new(
                                    ErrorCode::InternalError,
                                    "glTF UV accessor is missing for a bound collection layer",
                                )
                            })?;
                    primitive_attributes.insert(target_name, accessor);
                    target += 1;
                }
            }
            if let Some(material_id) = snapshot.slot_material(node_id, node, data_block, slot) {
                let material_index = if let Some(variant) = tex_coords.for_slot(node_id, slot) {
                    material_variant_indices
                        .get(material_id)
                        .and_then(|variants| variants.get(variant))
                        .copied()
                } else {
                    material_indices.get(material_id).copied()
                }
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::SceneInvalid,
                        "node material UV variant is missing",
                    )
                })?;
                primitive["material"] = json!(material_index);
            }
            primitives.push(primitive);
        }
        let mut target_names = Vec::new();
        let mut target_metadata = Vec::new();
        let mut mesh_weights = Vec::new();
        let mut targets = Vec::new();
        if let Some(shape_keys) = data_block
            .and_then(|data| data.shape_keys.as_ref())
            .filter(|shape_keys| !shape_keys.keys.is_empty())
        {
            let count = layout.source_vertex_indices.len();
            for (key_index, (key_id, key)) in shape_keys.keys.iter().enumerate() {
                let deltas = &key_deltas[key_index];
                let mut position_values = Vec::with_capacity(count.saturating_mul(3));
                let mut normal_values = Vec::with_capacity(count.saturating_mul(3));
                for (output, source_index) in layout.source_vertex_indices.iter().enumerate() {
                    let delta =
                        geometry_scale * geometry_basis.transform_vector3(deltas[*source_index]);
                    if !delta.is_finite() {
                        return Err(PotError::new(
                            ErrorCode::SceneInvalid,
                            "shape-key delta is non-finite",
                        ));
                    }
                    for component in delta.to_array() {
                        position_values.push(to_f32(component)?);
                    }
                    let normal_delta =
                        layout.normals[key_index + 1][output] - layout.normals[0][output];
                    for component in normal_delta.to_array() {
                        normal_values.push(to_f32(component)?);
                    }
                }
                let mut minimum = [f32::INFINITY; 3];
                let mut maximum = [f32::NEG_INFINITY; 3];
                for delta in position_values.as_chunks::<3>().0 {
                    for axis in 0..3 {
                        minimum[axis] = minimum[axis].min(delta[axis]);
                        maximum[axis] = maximum[axis].max(delta[axis]);
                    }
                }
                let position_accessor = append_float_accessor(
                    &mut binary,
                    &mut views,
                    &mut accessors,
                    &position_values,
                    count,
                    "VEC3",
                    Some(json!(minimum)),
                    Some(json!(maximum)),
                );
                let normal_accessor = append_float_accessor(
                    &mut binary,
                    &mut views,
                    &mut accessors,
                    &normal_values,
                    count,
                    "VEC3",
                    None,
                    None,
                );
                targets.push(json!({ "POSITION": position_accessor, "NORMAL": normal_accessor }));
                target_names.push(key.name.clone());
                mesh_weights.push(key.value);
                target_metadata.push(json!({
                    "id": key_id,
                    "name": key.name,
                    "value": key.value,
                    "slider_min": key.slider_min,
                    "slider_max": key.slider_max,
                    "frame": key.frame,
                    "relative_key": key.relative_key,
                    "vertex_group": key.vertex_group
                }));
            }
        }
        for primitive in &mut primitives {
            if !targets.is_empty() {
                primitive["targets"] = json!(targets);
            }
        }
        let mesh_index = meshes.len();
        let mut mesh_value = json!({
            "name": mesh_name,
            "primitives": primitives,
            "extras": { "potter": {
                "data_id": data_id,
                "instance_collection": instance_target,
                "shape_keys": target_metadata,
                "shape_key_data": data_block.and_then(|block| block.shape_keys.as_ref()),
                "vertex_ids": layout.source_vertex_indices.iter().map(|index| mesh.vertices[*index].id).collect::<Vec<_>>(),
                "vertex_groups": data_block.map(|block| &block.vertex_groups),
                "vertex_weights": data_block.map(|block| export_vertex_weights(mesh, block)),
            } }
        });
        if color_metadata["layers"]
            .as_array()
            .is_some_and(|layers| !layers.is_empty())
        {
            mesh_value["extras"]["potter"]["color_attributes"] = color_metadata;
        }
        if !target_names.is_empty() {
            mesh_value["weights"] = json!(mesh_weights);
            mesh_value["extras"]["targetNames"] = json!(target_names);
        }
        if let Some(polygons) = polygons {
            mesh_value["extras"]["potter"]["polygons"] = polygons;
        }
        meshes.push(mesh_value);
        if let Some(data_id) = data_id {
            mesh_indices.insert(
                (data_id.clone(), mesh_variant, material_slots.clone()),
                mesh_index,
            );
        }
        if let Some(target) = instance_target {
            instance_mesh_indices.insert((target, mesh_variant, material_slots), mesh_index);
        }
        node_mesh_indices.insert(node_id.clone(), mesh_index);
    }

    let (cameras, camera_indices) = camera_definitions(doc);
    let (lights, light_indices) = light_definitions(doc);
    let mut nodes = Vec::<Value>::with_capacity(node_ids.len());
    for id in &node_ids {
        let node = doc.nodes.get(id).ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "node registry changed during export",
            )
        })?;
        let mut gltf_node = if let Some(frame_maps) = frame_maps {
            let matrix = frame_maps
                .node_local_matrices
                .get(id)
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::InternalError,
                        "normalized node transform is missing",
                    )
                })?
                .to_cols_array()
                .into_iter()
                .map(to_f32)
                .collect::<Result<Vec<_>>>()?;
            json!({ "name": node.name, "matrix": matrix })
        } else if node.parent_type == crate::model::ParentType::Bone {
            let armature_id = node.parent.as_ref().ok_or_else(|| {
                PotError::new(ErrorCode::SceneInvalid, "bone parent node is missing")
            })?;
            let bone_id = node.parent_bone.as_ref().ok_or_else(|| {
                PotError::new(ErrorCode::SceneInvalid, "bone parent ID is missing")
            })?;
            let bone_world = snapshot
                .bone_matrices
                .get(armature_id)
                .and_then(|bones| bones.get(bone_id))
                .map(DMat4::from_cols_array)
                .ok_or_else(|| {
                    PotError::with_details(
                        ErrorCode::EvaluationFailed,
                        "evaluated parent bone transform is missing",
                        json!({ "parent": armature_id, "bone": bone_id, "node": id }),
                    )
                })?;
            let bone_inverse = bone_world.inverse();
            let node_world = snapshot
                .nodes
                .get(id)
                .map(|node| DMat4::from_cols_array(&node.world_matrix))
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::EvaluationFailed,
                        "bone-parented node world transform is missing",
                    )
                })?;
            if !finite_matrix(bone_inverse) || !finite_matrix(node_world) {
                return Err(unsupported_export(
                    "gltf.bone_parent_transform",
                    "glTF bone-parented nodes require finite invertible parent bone transforms.",
                ));
            }
            let converted = POT_TO_GLTF * bone_inverse * node_world * inverse_basis;
            let matrix = converted
                .to_cols_array()
                .into_iter()
                .map(to_f32)
                .collect::<Result<Vec<_>>>()?;
            json!({ "name": node.name, "matrix": matrix })
        } else {
            let transform = crate::eval::animation::animated_transform(node, doc, snapshot.frame)?;
            if node.action.is_some() && node.parent_inverse.is_none() {
                let basis_rotation = pot_to_gltf_rotation();
                let converted_rotation =
                    basis_rotation * transform.rotation_quat() * basis_rotation.conjugate();
                json!({
                    "name": node.name,
                    "translation": axis_to_gltf(transform.translation),
                    "rotation": [converted_rotation.x, converted_rotation.y, converted_rotation.z, converted_rotation.w],
                    "scale": transform.scale
                })
            } else {
                let mut local = transform.matrix();
                if let Some(parent_inverse) = &node.parent_inverse {
                    local = DMat4::from_cols_array(parent_inverse) * local;
                }
                let converted = POT_TO_GLTF * local * inverse_basis;
                let matrix = converted
                    .to_cols_array()
                    .into_iter()
                    .map(to_f32)
                    .collect::<Result<Vec<_>>>()?;
                json!({ "name": node.name, "matrix": matrix })
            }
        };
        let mut potter_extras = json!({
            "id": id,
            "kind": node.kind,
            "data_id": node.data,
            "tags": node.tags,
            "scene_hash": snapshot.scene_hash,
            "revision": snapshot.revision,
            "properties": node.properties
        });
        if node.kind == "armature"
            && let Some(armature) = node
                .data
                .as_ref()
                .and_then(|data_id| doc.data_blocks.get(data_id))
                .and_then(|data| data.armature.as_ref())
        {
            potter_extras["armature"] = serde_json::to_value(armature)
                .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))?;
            potter_extras["pose"] = serde_json::to_value(&node.pose)
                .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))?;
        }
        if let Some(scale) = frame_maps
            .and_then(|frames| frames.mesh_geometry_scales.get(id))
            .filter(|scale| **scale != DVec3::ONE)
        {
            potter_extras["vrm_baked_scale"] = json!(scale.to_array());
        }
        gltf_node["extras"] = json!({ "potter": potter_extras });
        if let Some(parent) = &node.parent {
            let _ = node_indices
                .get(parent)
                .ok_or_else(|| PotError::new(ErrorCode::SceneInvalid, "node parent is missing"))?;
        }
        if visible_node_ids.contains(id) && node.visible && node.render_visible {
            if let Some(index) = node_mesh_indices.get(id) {
                gltf_node["mesh"] = json!(index);
            }
            if let Some(data_id) = &node.data {
                if let Some(index) = camera_indices.get(data_id) {
                    gltf_node["camera"] = json!(index);
                }
                if let Some(index) = light_indices.get(data_id) {
                    gltf_node["extensions"] = json!({ "KHR_lights_punctual": { "light": index } });
                }
            }
        }
        nodes.push(gltf_node);
    }
    let SkinDefinitions {
        skins,
        node_skins,
        bone_indices,
    } = skin_definitions(
        doc,
        snapshot,
        &node_indices,
        &skin_bindings,
        avatar_armature_nodes.as_ref(),
        frame_maps,
        &mut nodes,
        &mut binary,
        &mut views,
        &mut accessors,
    )?;
    for (node_id, skin_index) in &node_skins {
        let node_index = node_indices.get(node_id).copied().ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "skinned mesh node index is missing",
            )
        })?;
        nodes[node_index]["skin"] = json!(skin_index);
    }
    for (id, node) in &doc.nodes {
        let Some(parent) = &node.parent else {
            continue;
        };
        let parent_index = if node.parent_type == crate::model::ParentType::Bone {
            let bone = node.parent_bone.as_ref().ok_or_else(|| {
                PotError::new(ErrorCode::SceneInvalid, "bone parent ID is missing")
            })?;
            bone_indices
                .get(parent)
                .and_then(|indices| indices.get(bone))
                .copied()
                .ok_or_else(|| {
                    PotError::with_details(
                        ErrorCode::SceneInvalid,
                        "emitted parent bone node is missing",
                        json!({ "parent": parent, "bone": bone, "node": id }),
                    )
                })?
        } else {
            *node_indices
                .get(parent)
                .ok_or_else(|| PotError::new(ErrorCode::SceneInvalid, "node parent is missing"))?
        };
        let child_index = *node_indices
            .get(id)
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "node index is missing"))?;
        let children = nodes[parent_index]["children"].as_array_mut();
        if let Some(children) = children {
            children.push(json!(child_index));
        } else {
            nodes[parent_index]["children"] = json!([child_index]);
        }
    }
    let animations = animation_definitions(
        doc,
        snapshot,
        &node_indices,
        &mut binary,
        &mut views,
        &mut accessors,
    )?;
    let roots = doc
        .nodes
        .iter()
        .filter(|(_, node)| node.parent.is_none())
        .filter_map(|(id, _)| node_indices.get(id).copied())
        .collect::<Vec<_>>();
    let buffer = if let Some(uri) = bin_uri {
        json!({ "byteLength": binary.len(), "uri": uri })
    } else {
        json!({ "byteLength": binary.len() })
    };
    let scene = doc
        .scenes
        .get(&snapshot.scene_id)
        .ok_or_else(|| PotError::new(ErrorCode::SceneInvalid, "export scene is missing"))?;
    let mut root = json!({
        "asset": { "version": "2.0", "generator": "potter" },
        "scene": 0,
        "scenes": [{ "nodes": roots }],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "bufferViews": views,
        "accessors": accessors,
        "buffers": [buffer],
        "extras": { "potter": { "scene_id": snapshot.scene_id, "scene_hash": snapshot.scene_hash, "revision": snapshot.revision, "frame": snapshot.frame, "fps": scene.fps, "fps_base": scene.fps_base } }
    });
    if !skins.is_empty() {
        root["skins"] = json!(skins);
    }
    if !texture_export.images.is_empty() {
        root["images"] = json!(texture_export.images);
    }
    if !texture_export.textures.is_empty() {
        root["textures"] = json!(texture_export.textures);
    }
    if !texture_export.samplers.is_empty() {
        root["samplers"] = json!(texture_export.samplers);
    }
    if !cameras.is_empty() {
        root["cameras"] = json!(cameras);
    }
    if !lights.is_empty() {
        extensions_used.insert("KHR_lights_punctual".to_owned());
        root["extensions"] = json!({ "KHR_lights_punctual": { "lights": lights } });
    }
    if texture_export.uses_texture_transform() {
        extensions_used.insert("KHR_texture_transform".to_owned());
        root["extensionsRequired"] = json!(["KHR_texture_transform"]);
    }
    if !extensions_used.is_empty() {
        root["extensionsUsed"] = json!(extensions_used.into_iter().collect::<Vec<_>>());
    }
    if let Some((avatar, version)) = vrm {
        let frames = frame_maps.ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "VRM output frame maps are missing",
            )
        })?;
        super::vrm::append_extensions(
            doc,
            avatar,
            version,
            VrmWriterMappings {
                node_indices: &node_indices,
                bone_indices: &bone_indices,
                material_indices: &material_indices,
                surfaces: &surfaces,
                tex_coords: &tex_coords,
                node_mesh_indices: &mut node_mesh_indices,
                node_offset_matrices: &frames.node_offset_matrices,
                bone_offset_matrices: &frames.bone_offset_matrices,
            },
            &mut root,
            root_path,
            &mut texture_export,
            &mut binary,
            &mut views,
        )?;
        preserve_vrm_material_variants(
            version,
            &surfaces,
            avatar,
            &material_indices,
            &material_variant_indices,
            &mut root,
            material_indices.len(),
        )?;
        root["bufferViews"] = json!(views);
        root["buffers"][0]["byteLength"] = json!(binary.len());
        if !texture_export.images.is_empty() {
            root["images"] = json!(texture_export.images);
        }
        if !texture_export.textures.is_empty() {
            root["textures"] = json!(texture_export.textures);
        }
        if !texture_export.samplers.is_empty() {
            root["samplers"] = json!(texture_export.samplers);
        }
    }
    if !animations.is_empty() {
        root["animations"] = json!(animations);
    }
    omit_empty_containers(&mut root, binary.is_empty());
    let json_bytes = serde_json::to_vec(&root)
        .map_err(|error| PotError::new(ErrorCode::ExportFailed, error.to_string()))?;
    Ok((json_bytes, binary))
}

/// Removes top-level containers that would be empty. glTF requires `accessors`, `bufferViews`,
/// `materials`, and `meshes` to hold at least one entry when present, and a buffer to hold at
/// least one byte, so empty geometry omits them instead of serializing empty containers.
fn omit_empty_containers(root: &mut Value, binary_is_empty: bool) {
    let Some(object) = root.as_object_mut() else {
        return;
    };
    for key in ["accessors", "bufferViews", "materials", "meshes"] {
        if object
            .get(key)
            .and_then(Value::as_array)
            .is_some_and(Vec::is_empty)
        {
            object.remove(key);
        }
    }
    if binary_is_empty {
        object.remove("buffers");
    }
}

fn camera_definitions(doc: &SceneDoc) -> (Vec<Value>, BTreeMap<Id, usize>) {
    let mut cameras = Vec::new();
    let mut indices = BTreeMap::new();
    for (data_id, data) in &doc.data_blocks {
        let Some(camera) = &data.camera else {
            continue;
        };
        let value = match camera.projection {
            CameraProjection::Perspective => {
                let field_of_view = (camera.sensor_width_mm / (2.0 * camera.lens_mm)).atan() * 2.0;
                json!({
                    "name": data_id,
                    "type": "perspective",
                    "perspective": {
                        "yfov": field_of_view,
                        "znear": camera.clip_start,
                        "zfar": camera.clip_end
                    }
                })
            }
            CameraProjection::Orthographic => json!({
                "name": data_id,
                "type": "orthographic",
                "orthographic": {
                    "xmag": camera.ortho_scale,
                    "ymag": camera.ortho_scale,
                    "znear": camera.clip_start,
                    "zfar": camera.clip_end
                }
            }),
            CameraProjection::Panorama | CameraProjection::Fisheye => {
                let field_of_view = (camera.sensor_width_mm / (2.0 * camera.lens_mm)).atan() * 2.0;
                json!({
                    "name": data_id,
                    "type": "perspective",
                    "perspective": {
                        "yfov": field_of_view,
                        "znear": camera.clip_start,
                        "zfar": camera.clip_end
                    },
                    "extras": { "potter": { "camera_data": camera } }
                })
            }
        };
        indices.insert(data_id.clone(), cameras.len());
        cameras.push(value);
    }
    (cameras, indices)
}

fn light_definitions(doc: &SceneDoc) -> (Vec<Value>, BTreeMap<Id, usize>) {
    let mut lights = Vec::new();
    let mut indices = BTreeMap::new();
    for (data_id, data) in &doc.data_blocks {
        let Some(light) = &data.light else {
            continue;
        };
        let (kind, spot) = match light.light_type {
            LightType::Sun => ("directional", None),
            LightType::Point => ("point", None),
            LightType::Spot => (
                "spot",
                Some(json!({
                    "innerConeAngle": light.spot_size * (1.0 - light.spot_blend) * 0.5,
                    "outerConeAngle": light.spot_size * 0.5
                })),
            ),
            LightType::Area => continue,
        };
        let mut value = json!({
            "name": data_id,
            "type": kind,
            "color": light.color,
            "intensity": light.energy
        });
        if kind == "point" && light.radius.is_finite() && light.radius > 0.0 {
            value["range"] = json!(light.radius);
        }
        if let Some(spot) = spot {
            value["spot"] = spot;
        }
        indices.insert(data_id.clone(), lights.len());
        lights.push(value);
    }
    (lights, indices)
}

fn animation_definitions(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    node_indices: &BTreeMap<Id, usize>,
    binary: &mut Vec<u8>,
    views: &mut Vec<Value>,
    accessors: &mut Vec<Value>,
) -> Result<Vec<Value>> {
    let scene = doc
        .scenes
        .get(&snapshot.scene_id)
        .ok_or_else(|| PotError::new(ErrorCode::SceneInvalid, "animation scene is missing"))?;
    let frames_per_second = f64::from(scene.fps) / scene.fps_base;
    let mut animations = Vec::new();
    for (node_id, node) in &doc.nodes {
        let Some(action_id) = &node.action else {
            continue;
        };
        if node.parent_inverse.is_some() {
            continue;
        }
        let action = doc.actions.get(action_id).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "node action reference is missing",
                json!({ "node_id": node_id, "action_id": action_id }),
            )
        })?;
        let mut frames = action
            .fcurves
            .iter()
            .flat_map(|curve| curve.keyframes.iter().map(|keyframe| keyframe.frame))
            .collect::<Vec<_>>();
        frames.sort_by(f64::total_cmp);
        frames.dedup_by(|left, right| crate::float::equal_f64(*left, *right));
        if frames.is_empty() {
            continue;
        }
        let all_keys = action
            .fcurves
            .iter()
            .flat_map(|curve| curve.keyframes.iter())
            .collect::<Vec<_>>();
        let interpolation = if all_keys
            .iter()
            .all(|key| key.interpolation == Interpolation::Constant)
        {
            "STEP"
        } else if all_keys
            .iter()
            .all(|key| key.interpolation == Interpolation::Bezier)
        {
            "CUBICSPLINE"
        } else {
            "LINEAR"
        };
        let start_frame = frames[0];
        let times = frames
            .iter()
            .map(|frame| to_f32((frame - start_frame) / frames_per_second))
            .collect::<Result<Vec<_>>>()?;
        let input = append_float_accessor(
            binary,
            views,
            accessors,
            &times,
            times.len(),
            "SCALAR",
            Some(json!([times[0]])),
            Some(json!([times[times.len() - 1]])),
        );
        let mut samplers = Vec::new();
        let mut channels = Vec::new();
        for path in ["translation", "rotation", "scale"] {
            let mut values = Vec::new();
            for frame in &frames {
                let sample = transform_values(node, doc, *frame, path)?;
                if interpolation == "CUBICSPLINE" {
                    let tangent =
                        transform_tangent(node, doc, &frames, *frame, path, frames_per_second)?;
                    for value in tangent.iter().chain(sample.iter()).chain(tangent.iter()) {
                        values.push(to_f32(*value)?);
                    }
                } else {
                    values.extend(
                        sample
                            .iter()
                            .map(|value| to_f32(*value))
                            .collect::<Result<Vec<_>>>()?,
                    );
                }
            }
            let dimensions = if path == "rotation" { 4 } else { 3 };
            let output_count = frames
                .len()
                .checked_mul(if interpolation == "CUBICSPLINE" { 3 } else { 1 })
                .ok_or_else(|| {
                    PotError::new(ErrorCode::LimitExceeded, "animation output count overflow")
                })?;
            let output = append_float_accessor(
                binary,
                views,
                accessors,
                &values,
                output_count,
                if dimensions == 4 { "VEC4" } else { "VEC3" },
                None,
                None,
            );
            let sampler_index = samplers.len();
            samplers
                .push(json!({ "input": input, "output": output, "interpolation": interpolation }));
            let target_node = node_indices.get(node_id).copied().ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "animated node index is missing")
            })?;
            channels.push(json!({
                "sampler": sampler_index,
                "target": { "node": target_node, "path": path }
            }));
        }
        animations.push(json!({
            "name": action.name,
            "samplers": samplers,
            "channels": channels,
            "extras": { "potter": { "start_frame": start_frame } }
        }));
    }
    Ok(animations)
}

fn append_float_accessor(
    binary: &mut Vec<u8>,
    views: &mut Vec<Value>,
    accessors: &mut Vec<Value>,
    values: &[f32],
    count: usize,
    kind: &str,
    minimum: Option<Value>,
    maximum: Option<Value>,
) -> usize {
    align4(binary);
    let offset = binary.len();
    for value in values {
        binary.extend_from_slice(&value.to_le_bytes());
    }
    let view = push_view(views, offset, binary.len() - offset, None);
    let index = accessors.len();
    let mut accessor = json!({
        "bufferView": view,
        "componentType": 5126,
        "count": count,
        "type": kind
    });
    if let Some(minimum) = minimum {
        accessor["min"] = minimum;
    }
    if let Some(maximum) = maximum {
        accessor["max"] = maximum;
    }
    accessors.push(accessor);
    index
}

fn transform_values(
    node: &crate::model::Node,
    doc: &SceneDoc,
    frame: f64,
    path: &str,
) -> Result<Vec<f64>> {
    let transform = crate::eval::animation::animated_transform(node, doc, frame)?;
    match path {
        "translation" => Ok(axis_to_gltf(transform.translation).to_vec()),
        "scale" => Ok(transform.scale.to_vec()),
        "rotation" => {
            let basis = pot_to_gltf_rotation();
            let rotation = basis * transform.rotation_quat() * basis.conjugate();
            Ok(vec![rotation.x, rotation.y, rotation.z, rotation.w])
        }
        _ => Err(PotError::new(
            ErrorCode::InternalError,
            "unsupported glTF animation path",
        )),
    }
}

fn transform_tangent(
    node: &crate::model::Node,
    doc: &SceneDoc,
    frames: &[f64],
    frame: f64,
    path: &str,
    frames_per_second: f64,
) -> Result<Vec<f64>> {
    let first = frames.first().copied().unwrap_or(frame);
    let last = frames.last().copied().unwrap_or(frame);
    let before = (frame - 0.001).max(first);
    let after = (frame + 0.001).min(last);
    if crate::float::equal_f64(before, after) {
        return Ok(vec![0.0; if path == "rotation" { 4 } else { 3 }]);
    }
    let left = transform_values(node, doc, before, path)?;
    let right = transform_values(node, doc, after, path)?;
    let elapsed_seconds = (after - before) / frames_per_second;
    Ok(left
        .iter()
        .zip(right)
        .map(|(left, right)| (right - left) / elapsed_seconds)
        .collect())
}

fn push_view(views: &mut Vec<Value>, offset: usize, length: usize, target: Option<u32>) -> usize {
    let index = views.len();
    let mut view = json!({ "buffer": 0, "byteOffset": offset, "byteLength": length });
    if let Some(target) = target {
        view["target"] = json!(target);
    }
    views.push(view);
    index
}

fn align4(bytes: &mut Vec<u8>) {
    while !bytes.len().is_multiple_of(4) {
        bytes.push(0);
    }
}

fn to_f32(value: f64) -> Result<f32> {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "glTF binary attributes are float32 by specification"
    )]
    let value = value as f32;
    if !value.is_finite() {
        return Err(PotError::new(
            ErrorCode::ExportFailed,
            "coordinate exceeds glTF float32 range",
        ));
    }
    Ok(value)
}

/// Packs the JSON chunk and, when there are buffer bytes, the BIN chunk into one GLB container.
/// glTF forbids zero-length buffers, so an export without buffer bytes has no BIN chunk.
pub(crate) fn to_glb(json_bytes: &[u8], binary: &[u8]) -> Result<Vec<u8>> {
    let json_length = json_bytes
        .len()
        .checked_add((4 - json_bytes.len() % 4) % 4)
        .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, "GLB JSON chunk is too large"))?;
    let binary_length = binary
        .len()
        .checked_add((4 - binary.len() % 4) % 4)
        .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, "GLB binary chunk is too large"))?;
    let binary_chunk_length = if binary.is_empty() {
        0
    } else {
        binary_length.checked_add(8).ok_or_else(|| {
            PotError::new(ErrorCode::LimitExceeded, "GLB binary chunk is too large")
        })?
    };
    let total_length = 12_usize
        .checked_add(8)
        .and_then(|value| value.checked_add(json_length))
        .and_then(|value| value.checked_add(binary_chunk_length))
        .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, "GLB file is too large"))?;
    let total_length = u32::try_from(total_length)
        .map_err(|_| PotError::new(ErrorCode::LimitExceeded, "GLB file exceeds u32 length"))?;
    let mut output = Vec::with_capacity(total_length as usize);
    output.extend_from_slice(b"glTF");
    output.extend_from_slice(&2_u32.to_le_bytes());
    output.extend_from_slice(&total_length.to_le_bytes());
    output.extend_from_slice(
        &u32::try_from(json_length)
            .map_err(|_| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "GLB JSON chunk exceeds u32 length",
                )
            })?
            .to_le_bytes(),
    );
    output.extend_from_slice(&0x4E4F_534A_u32.to_le_bytes());
    output.extend_from_slice(json_bytes);
    output.resize(output.len() + json_length - json_bytes.len(), b' ');
    if !binary.is_empty() {
        output.extend_from_slice(
            &u32::try_from(binary_length)
                .map_err(|_| {
                    PotError::new(
                        ErrorCode::LimitExceeded,
                        "GLB binary chunk exceeds u32 length",
                    )
                })?
                .to_le_bytes(),
        );
        output.extend_from_slice(&0x004E_4942_u32.to_le_bytes());
        output.extend_from_slice(binary);
        output.resize(output.len() + binary_length - binary.len(), 0);
    }
    Ok(output)
}

fn read_buffer_view_bytes(root: &Value, buffers: &[Vec<u8>], view_index: usize) -> Result<Vec<u8>> {
    let view = root["bufferViews"]
        .get(view_index)
        .ok_or_else(|| import_error("glTF image bufferView is missing"))?;
    let buffer_index = usize::try_from(view["buffer"].as_u64().unwrap_or(0))
        .map_err(|_| import_error("glTF image buffer index exceeds platform range"))?;
    let buffer = buffers
        .get(buffer_index)
        .ok_or_else(|| import_error("glTF image buffer is missing"))?;
    let start = usize::try_from(view["byteOffset"].as_u64().unwrap_or(0))
        .map_err(|_| import_error("glTF image buffer offset exceeds platform range"))?;
    let length = usize::try_from(
        view["byteLength"]
            .as_u64()
            .ok_or_else(|| import_error("glTF image byteLength is missing"))?,
    )
    .map_err(|_| import_error("glTF image byteLength exceeds platform range"))?;
    let end = start
        .checked_add(length)
        .ok_or_else(|| import_error("glTF image buffer range overflows"))?;
    buffer
        .get(start..end)
        .map(<[u8]>::to_vec)
        .ok_or_else(|| import_error("glTF image buffer data is truncated"))
}

fn import_images(
    root: &Value,
    buffers: &[Vec<u8>],
    source_path: &Path,
    doc: &mut SceneDoc,
    assets: &mut Vec<(String, Vec<u8>)>,
) -> Result<Vec<Id>> {
    let image_values = root["images"].as_array().map_or(&[][..], Vec::as_slice);
    let mut image_ids = Vec::with_capacity(image_values.len());
    let mut used_ids = BTreeSet::new();
    let mut known_assets = BTreeSet::new();
    for (index, image_value) in image_values.iter().enumerate() {
        let mime = image_value["mimeType"].as_str().unwrap_or_else(|| {
            image_value["uri"]
                .as_str()
                .filter(|uri| {
                    uri.starts_with("data:image/png") || uri.to_ascii_lowercase().ends_with(".png")
                })
                .map_or("", |_| "image/png")
        });
        if mime != "image/png" {
            return Err(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                "glTF texture import supports PNG images only",
                json!({ "feature_id": "material.texture_image_encoding", "status": "not_supported", "mime_type": mime }),
            ));
        }
        let (encoded, source_name) = if let Some(uri) = image_value["uri"].as_str() {
            if uri.starts_with("data:") {
                let (header, _) = uri
                    .split_once(',')
                    .ok_or_else(|| import_error("glTF image data URI is invalid"))?;
                if !header.starts_with("data:image/png;base64") {
                    return Err(import_error("glTF image data URI is not a PNG"));
                }
                (decode_data_uri(uri)?, format!("gltf_image_{index}.png"))
            } else {
                let image_path = source_path
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .join(uri);
                let bytes = fs::read(&image_path).map_err(|error| PotError::io(&error))?;
                let name = image_path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .map_or_else(|| format!("gltf_image_{index}.png"), str::to_owned);
                (bytes, name)
            }
        } else if let Some(view_index) = image_value["bufferView"].as_u64() {
            (
                read_buffer_view_bytes(
                    root,
                    buffers,
                    usize::try_from(view_index).map_err(|_| {
                        import_error("glTF image bufferView exceeds platform range")
                    })?,
                )?,
                format!("gltf_image_{index}.png"),
            )
        } else {
            return Err(import_error(
                "glTF image has neither a URI nor a bufferView",
            ));
        };
        let (width, height, _) = crate::image::decode_pixels(&encoded, ImageColorspace::Srgb)
            .map_err(|error| {
                PotError::with_details(
                    ErrorCode::ImportFailed,
                    "glTF PNG texture is invalid",
                    json!({ "image_index": index, "reason": error.to_string() }),
                )
            })?;
        let digest = crate::hash::sha256(&encoded);
        if known_assets.insert(digest.clone()) {
            assets.push((digest.clone(), encoded));
        }
        let suggested = image_value["extras"]["potter"]["id"]
            .as_str()
            .or_else(|| image_value["name"].as_str())
            .unwrap_or("image");
        let id = unique_import_id(suggested, "image", index, &mut used_ids)?;
        // A single-channel copy (VRM masks) is named `<source>_packed`; the Potter image keeps the source name.
        let name = image_value["extras"]["potter"]["source_name"]
            .as_str()
            .or_else(|| image_value["name"].as_str())
            .map_or(source_name, str::to_owned);
        doc.images.insert(
            id.clone(),
            Image {
                name,
                source: ImageSource::Packed,
                colorspace: ImageColorspace::Srgb,
                width,
                height,
                tiles: Vec::new(),
                blob: Some(digest),
                source_path: None,
                source_hash: None,
                alpha_mode: ImageAlphaMode::Straight,
            },
        );
        image_ids.push(id);
    }
    Ok(image_ids)
}
/// The `KHR_texture_transform` of a texture info, or `None` when it is absent or the identity.
fn texture_transform(info: &Value) -> Result<Option<super::material_surface::TextureTransform>> {
    let Some(extension) = info["extensions"]["KHR_texture_transform"].as_object() else {
        return Ok(None);
    };
    let numbers = |key: &str, default: &[f64]| -> Result<Vec<f64>> {
        match extension.get(key) {
            None | Some(Value::Null) => Ok(default.to_vec()),
            Some(value) => value
                .as_array()
                .and_then(|items| {
                    items
                        .iter()
                        .map(Value::as_f64)
                        .collect::<Option<Vec<f64>>>()
                })
                .filter(|items| items.len() == default.len())
                .ok_or_else(|| {
                    import_error(format!("glTF KHR_texture_transform `{key}` is invalid"))
                }),
        }
    };
    let offset = numbers("offset", &[0.0, 0.0])?;
    let scale = numbers("scale", &[1.0, 1.0])?;
    let rotation = match extension.get("rotation") {
        None | Some(Value::Null) => 0.0,
        Some(value) => value
            .as_f64()
            .ok_or_else(|| import_error("glTF KHR_texture_transform rotation is invalid"))?,
    };
    let transform = super::material_surface::TextureTransform {
        offset: [offset[0], offset[1]],
        rotation,
        scale: [scale[0], scale[1]],
    };
    Ok((!transform.is_identity()).then_some(transform))
}

/// The UV set a texture info samples. A `KHR_texture_transform` `texCoord` overrides `texCoord` for the
/// texture it transforms, as the extension specifies.
pub(crate) fn texture_info_set(info: &Value) -> u64 {
    info["extensions"]["KHR_texture_transform"]["texCoord"]
        .as_u64()
        .unwrap_or_else(|| info["texCoord"].as_u64().unwrap_or(0))
}

/// The layer a texture info samples on import. `None` is the render layer, which is layer 0 after import.
/// Sets past the layer limit fall back to it; `gltf_import_losses` reports them.
pub(crate) fn imported_uv_layer(info: &Value) -> Option<String> {
    let set = usize::try_from(texture_info_set(info)).ok()?;
    (set > 0 && set < uv_layers::MAX_LAYERS).then(|| uv_set_layer_name(set))
}

/// Every texture slot of a glTF material that a graph can hold, each with its transform.
fn material_textures(
    material: &Value,
    root: &Value,
    image_ids: &[Id],
) -> Result<material_graph::GraphTextures> {
    let slot = |info: &Value| -> Result<Option<material_graph::SlotTexture>> {
        let transform = texture_transform(info)?;
        Ok(import_texture_ref(info, root, image_ids)?
            .map(|texture| material_graph::SlotTexture { texture, transform }))
    };
    let pbr = &material["pbrMetallicRoughness"];
    Ok(material_graph::GraphTextures {
        base_color: slot(&pbr["baseColorTexture"])?,
        metallic_roughness: slot(&pbr["metallicRoughnessTexture"])?,
        roughness: None,
        metallic: None,
        normal: slot(&material["normalTexture"])?,
        normal_scale: material["normalTexture"]["scale"].as_f64().unwrap_or(1.0),
        emissive: slot(&material["emissiveTexture"])?,
        occlusion: slot(&material["occlusionTexture"])?,
    })
}

/// Layer name of glTF `TEXCOORD_n` on import: `UVMap` for set 0 and `UVMap.00n` after it, as
/// Blender names its layers.
fn uv_set_layer_name(set: usize) -> String {
    if set == 0 {
        "UVMap".to_owned()
    } else {
        format!("UVMap.{set:03}")
    }
}

fn import_texture_ref(info: &Value, root: &Value, image_ids: &[Id]) -> Result<Option<TextureRef>> {
    if !info.is_object() {
        return Ok(None);
    }
    // Sets past the layer limit fall back to the render layer; `gltf_import_losses` reports them.
    let uv_layer = imported_uv_layer(info);
    let texture_index = usize::try_from(
        info["index"]
            .as_u64()
            .ok_or_else(|| import_error("glTF material texture index is missing"))?,
    )
    .map_err(|_| import_error("glTF texture index exceeds platform range"))?;
    let texture = root["textures"]
        .get(texture_index)
        .ok_or_else(|| import_error("glTF material references a missing texture"))?;
    let image_index = usize::try_from(
        texture["source"]
            .as_u64()
            .ok_or_else(|| import_error("glTF texture source is missing"))?,
    )
    .map_err(|_| import_error("glTF image index exceeds platform range"))?;
    let image = image_ids
        .get(image_index)
        .cloned()
        .ok_or_else(|| import_error("glTF texture references a missing image"))?;
    let interpolation = if texture["sampler"]
        .as_u64()
        .and_then(|sampler_index| {
            usize::try_from(sampler_index)
                .ok()
                .and_then(|index| root["samplers"].get(index))
        })
        .is_some_and(|sampler| {
            sampler["magFilter"].as_u64() == Some(9728)
                || sampler["minFilter"].as_u64() == Some(9728)
        }) {
        ImageInterpolation::Closest
    } else {
        ImageInterpolation::Linear
    };
    Ok(Some(TextureRef {
        image,
        uv_layer,
        interpolation,
    }))
}

#[derive(Clone)]
struct ImportedSkin {
    armature_node: Id,
    armature_data_id: Id,
    joint_ids: Vec<Id>,
    joint_node_indices: Vec<usize>,
}
type SkinVertexWeights = (Vec<[u16; 4]>, Vec<[f64; 4]>);

fn node_world_matrices(nodes: &[Value], parents: &[Option<usize>]) -> Result<Vec<DMat4>> {
    fn visit(
        index: usize,
        nodes: &[Value],
        parents: &[Option<usize>],
        output: &mut [Option<DMat4>],
        active: &mut BTreeSet<usize>,
    ) -> Result<DMat4> {
        if let Some(matrix) = output.get(index).copied().flatten() {
            return Ok(matrix);
        }
        if !active.insert(index) {
            return Err(import_error("glTF node hierarchy contains a cycle"));
        }
        let local = node_matrix(&nodes[index])?;
        let world = if let Some(parent) = parents[index] {
            visit(parent, nodes, parents, output, active)? * local
        } else {
            local
        };
        if !finite_matrix(world) {
            return Err(import_error("glTF node world matrix is non-finite"));
        }
        active.remove(&index);
        output[index] = Some(world);
        Ok(world)
    }
    let mut output = vec![None; nodes.len()];
    let mut active = BTreeSet::new();
    for index in 0..nodes.len() {
        visit(index, nodes, parents, &mut output, &mut active)?;
    }
    output
        .into_iter()
        .map(|matrix| matrix.ok_or_else(|| import_error("glTF node world matrix is missing")))
        .collect()
}

/// The rest-world matrix of each imported skin bone, using the armature and glTF node frames.
fn imported_bone_world_matrices(
    doc: &SceneDoc,
    skins: &[ImportedSkin],
    node_ids: &[Id],
    node_worlds: &[DMat4],
    basis: DMat4,
) -> Result<BTreeMap<(Id, Id), DMat4>> {
    let mut worlds = BTreeMap::new();
    let mut armatures = BTreeSet::new();
    let inverse_basis = basis.inverse();
    for skin in skins {
        if !armatures.insert(skin.armature_node.clone()) {
            continue;
        }
        let armature = doc
            .data_blocks
            .get(&skin.armature_data_id)
            .and_then(|data| data.armature.as_ref())
            .ok_or_else(|| import_error("imported skin armature data is missing"))?;
        let armature_world = node_ids
            .iter()
            .position(|id| id == &skin.armature_node)
            .map_or(DMat4::IDENTITY, |index| {
                inverse_basis * node_worlds[index] * basis
            });
        for (bone_id, matrices) in
            crate::eval::rig::evaluate_bone_matrices(armature, &BTreeMap::new())?
        {
            let world = armature_world * matrices.rest;
            if !finite_matrix(world) {
                return Err(import_error("imported skin bone transform is non-finite"));
            }
            worlds.insert((skin.armature_node.clone(), bone_id), world);
        }
    }
    Ok(worlds)
}

type ImportedSkinGraph = (
    Vec<ImportedSkin>,
    BTreeSet<usize>,
    BTreeMap<usize, Id>,
    Vec<Id>,
);

/// Which armature a skin belongs to, chosen as Blender's glTF importer does. A Potter export names its
/// armature node; any other skin takes the deepest common ancestor of its joints and skeleton (`None`
/// for the scene root), or that node's parent when the ancestor is itself a joint.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ArmatureKey {
    Node(usize),
    Ancestor(Option<usize>),
}

/// Skins that share one armature: one armature whose bones are the union of the member skins' joints.
struct SkinGroup {
    armature_node_index: Option<usize>,
    first_skin: usize,
    /// Joint nodes of every member skin, in first-seen order.
    joints: Vec<usize>,
    /// Raw inverse bind matrix of each joint, from the first member skin that lists it.
    inverse_binds: BTreeMap<usize, Option<DMat4>>,
}

/// The glTF node indices from the scene root down to `node`, `node` included.
fn ancestor_path(parents: &[Option<usize>], node: usize) -> Vec<usize> {
    let mut path = vec![node];
    let mut current = node;
    while let Some(parent) = parents[current] {
        path.push(parent);
        current = parent;
    }
    path.reverse();
    path
}

/// Blender's armature node for a skin: the deepest common ancestor of `nodes` (`None` for the scene
/// root), or its parent when that ancestor is one of `joints`.
fn armature_ancestor(
    parents: &[Option<usize>],
    nodes: &[usize],
    joints: &[usize],
) -> Option<usize> {
    let mut common: Option<Vec<usize>> = None;
    for node in nodes {
        let path = ancestor_path(parents, *node);
        common = Some(match common {
            None => path,
            Some(previous) => previous
                .iter()
                .zip(&path)
                .take_while(|(left, right)| left == right)
                .map(|(left, _)| *left)
                .collect(),
        });
    }
    let ancestor = common.and_then(|path| path.last().copied());
    match ancestor {
        Some(node) if joints.contains(&node) => parents[node],
        other => other,
    }
}

/// The nearest ancestor of `node` that is one of `joints`.
fn nearest_joint_ancestor(
    parents: &[Option<usize>],
    node: usize,
    joints: &[usize],
) -> Option<usize> {
    let mut current = parents[node];
    while let Some(parent) = current {
        if joints.contains(&parent) {
            return Some(parent);
        }
        current = parents[parent];
    }
    None
}

fn import_skins(
    root: &Value,
    buffers: &[Vec<u8>],
    node_values: &[Value],
    parents: &[Option<usize>],
    source_node_worlds: &[DMat4],
    node_ids: &[Id],
    doc: &mut SceneDoc,
    used_data_ids: &mut BTreeSet<Id>,
    used_node_ids: &mut BTreeSet<Id>,
    basis: DMat4,
) -> Result<ImportedSkinGraph> {
    let skin_values = root["skins"].as_array().map_or(&[][..], Vec::as_slice);
    let world_matrices = source_node_worlds;
    // Skins with the same `ArmatureKey` form one armature.
    let mut groups = Vec::<SkinGroup>::new();
    let mut group_by_key = BTreeMap::<ArmatureKey, usize>::new();
    let mut skin_groups = Vec::with_capacity(skin_values.len());
    let mut skin_joints = Vec::with_capacity(skin_values.len());
    for (skin_index, skin) in skin_values.iter().enumerate() {
        let joints = skin["joints"]
            .as_array()
            .ok_or_else(|| import_error("glTF skin joints are missing"))?;
        if joints.is_empty() {
            return Err(import_error("glTF skin has no joints"));
        }
        let joint_indices = joints
            .iter()
            .map(|joint| {
                usize::try_from(
                    joint
                        .as_u64()
                        .ok_or_else(|| import_error("glTF skin joint index is invalid"))?,
                )
                .map_err(|_| import_error("glTF skin joint index exceeds platform range"))
            })
            .collect::<Result<Vec<_>>>()?;
        if joint_indices
            .iter()
            .any(|index| *index >= node_values.len())
        {
            return Err(import_error("glTF skin references a missing joint node"));
        }
        let inverse_bind_matrices = skin["inverseBindMatrices"]
            .as_u64()
            .map(|accessor| {
                read_matrix_accessor(
                    root,
                    buffers,
                    usize::try_from(accessor).map_err(|_| {
                        import_error("glTF inverse bind accessor exceeds platform range")
                    })?,
                )
            })
            .transpose()?;
        if inverse_bind_matrices
            .as_ref()
            .is_some_and(|matrices| matrices.len() != joint_indices.len())
        {
            return Err(import_error(
                "glTF inverse bind matrix count does not match skin joints",
            ));
        }
        let armature_node_index = skin["extras"]["potter"]["armature_id"]
            .as_str()
            .and_then(|raw_id| node_ids.iter().position(|id| id.as_str() == raw_id));
        let key = if let Some(index) = armature_node_index {
            ArmatureKey::Node(index)
        } else {
            let mut nodes = joint_indices.clone();
            if let Some(skeleton) = skin["skeleton"].as_u64() {
                let skeleton = usize::try_from(skeleton)
                    .ok()
                    .filter(|index| *index < node_values.len())
                    .ok_or_else(|| import_error("glTF skin skeleton is missing"))?;
                nodes.push(skeleton);
            }
            ArmatureKey::Ancestor(armature_ancestor(parents, &nodes, &joint_indices))
        };
        let group = *group_by_key.entry(key).or_insert_with(|| {
            groups.push(SkinGroup {
                armature_node_index,
                first_skin: skin_index,
                joints: Vec::new(),
                inverse_binds: BTreeMap::new(),
            });
            groups.len() - 1
        });
        for (order, joint_index) in joint_indices.iter().enumerate() {
            let entry = &mut groups[group];
            if !entry.inverse_binds.contains_key(joint_index) {
                entry.joints.push(*joint_index);
                entry.inverse_binds.insert(
                    *joint_index,
                    inverse_bind_matrices
                        .as_ref()
                        .and_then(|matrices| matrices.get(order))
                        .copied(),
                );
            }
        }
        skin_groups.push(group);
        skin_joints.push(joint_indices);
    }
    let joint_nodes = skin_joints
        .iter()
        .flatten()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut armature_data_ids = BTreeMap::new();
    let mut synthetic_roots = Vec::new();
    // Per group: its armature node, its armature Data-Block, and the bone ID of each joint node.
    let mut group_armatures = Vec::with_capacity(groups.len());
    for group in &groups {
        let skin = &skin_values[group.first_skin];
        let armature_node_id = if let Some(index) = group.armature_node_index {
            node_ids[index].clone()
        } else {
            let suggested = format!("armature_skin_{}", group.first_skin);
            let id = unique_import_id(&suggested, "armature", group.first_skin, used_node_ids)?;
            doc.nodes.insert(
                id.clone(),
                Node {
                    name: skin["name"]
                        .as_str()
                        .map_or_else(|| id.to_string(), str::to_owned),
                    kind: "armature".to_owned(),
                    ..Node::default()
                },
            );
            synthetic_roots.push(id.clone());
            id
        };
        let mut bone_id_by_joint = BTreeMap::new();
        let mut used_bone_ids = BTreeSet::new();
        for (order, joint_index) in group.joints.iter().enumerate() {
            let joint = &node_values[*joint_index];
            let raw_id = joint["extras"]["potter"]["bone_id"]
                .as_str()
                .or_else(|| joint["extras"]["potter"]["id"].as_str())
                .or_else(|| joint["name"].as_str())
                .unwrap_or("bone");
            let id = unique_import_id(raw_id, "bone", order, &mut used_bone_ids)?;
            bone_id_by_joint.insert(*joint_index, id);
        }
        let armature_metadata = group
            .armature_node_index
            .and_then(|index| node_values[index]["extras"]["potter"]["armature"].as_object())
            .map(|_| {
                let index = group.armature_node_index.unwrap_or_default();
                serde_json::from_value::<ArmatureData>(
                    node_values[index]["extras"]["potter"]["armature"].clone(),
                )
                .map_err(|error| {
                    import_error(format!("glTF armature metadata is invalid: {error}"))
                })
            })
            .transpose()?;
        let armature = if let Some(armature) = armature_metadata {
            armature
        } else {
            let bone_ids = bone_id_by_joint.values().cloned().collect::<BTreeSet<_>>();
            let mut bones = BTreeMap::new();
            for joint_index in &group.joints {
                let joint_node = &node_values[*joint_index];
                let extras = &joint_node["extras"]["potter"];
                let id = bone_id_by_joint[joint_index].clone();
                if let Some(metadata) = extras.get("bone").filter(|value| value.is_object()) {
                    let bone: Bone = serde_json::from_value(metadata.clone()).map_err(|error| {
                        import_error(format!("glTF bone metadata is invalid: {error}"))
                    })?;
                    if let Some(parent_id) = bone.parent.as_ref()
                        && !bone_ids.contains(parent_id)
                    {
                        return Err(import_error("glTF bone parent metadata is missing"));
                    }
                    bones.insert(id, bone);
                    continue;
                }
                // The rest matrix is the inverse of the inverse bind matrix, or the node's world
                // matrix when the skin has none.
                let joint_bind = group
                    .inverse_binds
                    .get(joint_index)
                    .and_then(Option::as_ref)
                    .map_or_else(|| world_matrices[*joint_index], glam::DMat4::inverse);
                if !finite_matrix(joint_bind) {
                    return Err(import_error("glTF inverse bind matrix is singular"));
                }
                let converted = basis.inverse() * joint_bind * basis;
                let head = converted.transform_point3(DVec3::ZERO);
                let mut tail = converted.transform_point3(DVec3::Y);
                if head.distance_squared(tail) <= f64::EPSILON {
                    tail = head + DVec3::Y;
                }
                let parent = nearest_joint_ancestor(parents, *joint_index, &group.joints)
                    .and_then(|parent| bone_id_by_joint.get(&parent).cloned());
                let name = joint_node["name"]
                    .as_str()
                    .map_or_else(|| id.to_string(), str::to_owned);
                bones.insert(
                    id,
                    Bone {
                        name,
                        parent,
                        head: head.to_array(),
                        tail: tail.to_array(),
                        roll: 0.0,
                        deform: true,
                        inherit_rotation: true,
                        use_connect: false,
                        custom_shape: None,
                        envelope_distance: 0.25,
                        envelope_weight: 1.0,
                        head_radius: 0.1,
                        tail_radius: 0.1,
                        bbone_settings: BTreeMap::new(),
                    },
                );
            }
            ArmatureData {
                bones,
                ..ArmatureData::default()
            }
        };
        let data_suggested = group
            .armature_node_index
            .and_then(|index| node_values[index]["extras"]["potter"]["data_id"].as_str())
            .map_or_else(
                || format!("armature_skin_{}_data", group.first_skin),
                str::to_owned,
            );
        let armature_data_id = unique_import_id(
            &data_suggested,
            "armature_data",
            group.first_skin,
            used_data_ids,
        )?;
        doc.data_blocks.insert(
            armature_data_id.clone(),
            DataBlock {
                data_type: "armature".to_owned(),
                mesh: None,
                armature: Some(armature),
                ..DataBlock::default()
            },
        );
        if let Some(index) = group.armature_node_index {
            armature_data_ids.insert(index, armature_data_id.clone());
        } else if let Some(node) = doc.nodes.get_mut(&armature_node_id) {
            node.data = Some(armature_data_id.clone());
        }
        group_armatures.push((armature_node_id, armature_data_id, bone_id_by_joint));
    }
    let mut skins = Vec::with_capacity(skin_values.len());
    for (group_index, joint_indices) in skin_groups.into_iter().zip(skin_joints) {
        let (armature_node, armature_data_id, bone_id_by_joint) = &group_armatures[group_index];
        skins.push(ImportedSkin {
            armature_node: armature_node.clone(),
            armature_data_id: armature_data_id.clone(),
            joint_ids: joint_indices
                .iter()
                .map(|joint| bone_id_by_joint[joint].clone())
                .collect(),
            joint_node_indices: joint_indices,
        });
    }
    Ok((skins, joint_nodes, armature_data_ids, synthetic_roots))
}
fn read_matrix_accessor(
    root: &Value,
    buffers: &[Vec<u8>],
    accessor_index: usize,
) -> Result<Vec<DMat4>> {
    let (values, dimensions) = read_float_accessor(root, buffers, accessor_index)?;
    if dimensions != 16 {
        return Err(import_error("glTF inverse bind accessor must be MAT4"));
    }
    let (matrices, remainder) = values.as_chunks::<16>();
    if !remainder.is_empty() {
        return Err(import_error("glTF inverse bind matrix is truncated"));
    }
    matrices
        .iter()
        .map(|values| Ok(DMat4::from_cols_array(values)))
        .collect()
}

fn restore_skin_weights(
    doc: &mut SceneDoc,
    mesh_index: usize,
    skin: &ImportedSkin,
    mesh_values: &[Value],
    data_ids: &[Id],
    mesh_skin_weights: &[Option<Vec<SkinVertexWeights>>],
    ids_stale: bool,
) -> Result<Id> {
    let data_id = data_ids
        .get(mesh_index)
        .cloned()
        .ok_or_else(|| import_error("glTF skin references a missing mesh"))?;
    let joint_weights = mesh_skin_weights
        .get(mesh_index)
        .and_then(Option::as_ref)
        .ok_or_else(|| import_error("glTF skin has no JOINTS_0 and WEIGHTS_0 attributes"))?;
    let armature = doc
        .data_blocks
        .get(&skin.armature_data_id)
        .and_then(|data| data.armature.clone())
        .ok_or_else(|| import_error("glTF skin armature Data-Block is missing"))?;
    let mesh_value = mesh_values
        .get(mesh_index)
        .ok_or_else(|| import_error("glTF skin references a missing mesh"))?;
    let source_block = doc
        .data_blocks
        .get(&data_id)
        .ok_or_else(|| import_error("glTF skin mesh Data-Block is missing"))?;
    let vertex_order = source_block
        .mesh
        .as_ref()
        .ok_or_else(|| import_error("glTF skin mesh has no topology"))?
        .vertices
        .iter()
        .map(|vertex| vertex.id)
        .collect::<Vec<_>>();
    let source_vertex_ids = match potter_vertex_ids(mesh_value)? {
        // Copies of one Potter vertex rejoin, so the skin has one entry per welded vertex.
        Some(ids) if !ids_stale => {
            let mut seen = BTreeSet::new();
            ids.into_iter().filter(|id| seen.insert(*id)).collect()
        }
        _ => vertex_order.clone(),
    };
    if vertex_order.len() != joint_weights.len() || source_vertex_ids.len() != vertex_order.len() {
        return Err(import_error(
            "glTF skin weight count does not match mesh vertices",
        ));
    }
    let source_vertex_id_set = source_vertex_ids.iter().copied().collect::<BTreeSet<_>>();
    let has_group_metadata = mesh_value["extras"]["potter"]["vertex_groups"].is_array();
    let mut groups = source_block.vertex_groups.clone();
    if !has_group_metadata {
        for joint_id in &skin.joint_ids {
            let bone = armature
                .bones
                .get(joint_id)
                .ok_or_else(|| import_error("glTF skin joint is not in its armature"))?;
            if !groups.iter().any(|group| group.name == bone.name) {
                groups.push(VertexGroup {
                    id: joint_id.clone(),
                    name: bone.name.clone(),
                });
            }
        }
    }
    let group_ids = groups
        .iter()
        .map(|group| group.id.clone())
        .collect::<BTreeSet<_>>();
    let exact_weight_metadata =
        !ids_stale && mesh_value["extras"]["potter"]["vertex_weights"].is_object();
    let exact_vertex_weights = exact_weight_metadata.then(|| source_block.vertex_weights.clone());
    let vertex_weights = if let Some(exact) = exact_vertex_weights {
        for (vertex_id, row) in &exact {
            if !source_vertex_id_set.contains(vertex_id) {
                return Err(import_error(
                    "glTF vertex weight metadata references a missing vertex",
                ));
            }
            for (group_id, weight) in row {
                if !group_ids.contains(group_id) || !weight.is_finite() || *weight < 0.0 {
                    return Err(import_error("glTF vertex weight metadata is invalid"));
                }
            }
        }
        let mut output = BTreeMap::new();
        for (vertex_id, source_id) in vertex_order.iter().zip(&source_vertex_ids) {
            if let Some(weights) = exact.get(source_id) {
                output.insert(*vertex_id, weights.clone());
            }
        }
        output
    } else {
        let mut group_for_bone = BTreeMap::<Id, Id>::new();
        for group in &groups {
            if let Some((bone_id, _)) = armature
                .bones
                .iter()
                .find(|(_, bone)| bone.name == group.name)
            {
                group_for_bone.insert(bone_id.clone(), group.id.clone());
            }
        }
        let mut output = BTreeMap::new();
        for (vertex_id, (joint_sets, weight_sets)) in vertex_order.iter().zip(joint_weights) {
            let mut row = BTreeMap::new();
            if joint_sets.len() != weight_sets.len() {
                return Err(import_error(
                    "glTF JOINTS_n and WEIGHTS_n set counts do not match",
                ));
            }
            for (joints, weights) in joint_sets.iter().zip(weight_sets) {
                for slot in 0..4 {
                    let weight = weights[slot];
                    if weight <= 0.0 {
                        continue;
                    }
                    let joint_id =
                        skin.joint_ids
                            .get(usize::from(joints[slot]))
                            .ok_or_else(|| {
                                import_error("glTF JOINTS_n references a missing skin joint")
                            })?;
                    if let Some(group_id) = group_for_bone.get(joint_id) {
                        row.insert(group_id.clone(), weight);
                    }
                }
            }
            if !row.is_empty() {
                output.insert(*vertex_id, row);
            }
        }
        output
    };
    let block = doc
        .data_blocks
        .get_mut(&data_id)
        .ok_or_else(|| import_error("glTF skin mesh Data-Block is missing"))?;
    block.vertex_groups = groups;
    block.vertex_weights = vertex_weights;
    Ok(data_id)
}

/// glTF identities behind an import; the VRM decoder resolves glTF indices through these.
pub(crate) struct GltfIndex {
    /// glTF JSON root, read by the VRM decoder for extensions and accessors.
    pub(crate) root: Value,
    /// Potter-to-glTF axis basis of this import (`POT_TO_GLTF` for glTF, the avatar basis for VRM).
    pub(crate) basis: DMat4,
    pub(crate) node_ids: Vec<Id>,
    /// World matrices of glTF nodes in glTF space, indexed like `nodes`.
    pub(crate) node_worlds: Vec<DMat4>,
    pub(crate) material_ids: Vec<Id>,
    /// Color variants keyed by the source material and their selected Blender color-layer names.
    pub(crate) color_material_variants: BTreeMap<Id, BTreeMap<String, Id>>,
    pub(crate) image_ids: Vec<Id>,
    /// Per glTF mesh: the Potter shape-key ID of each morph target, in target order.
    pub(crate) mesh_shape_key_ids: Vec<Vec<Id>>,
    /// glTF joint node index to (armature node, bone) for every skinned joint.
    pub(crate) joint_bones: BTreeMap<usize, (Id, Id)>,
}

pub(crate) fn import(path: &Path, scene_id: String) -> Result<crate::exchange::ImportedGraph> {
    import_indexed(path, scene_id, POT_TO_GLTF).map(|(graph, _)| graph)
}

/// Imports glTF or GLB with Potter axes mapped through `basis` (VRM 0.x adds a Ry(180) turn).
pub(crate) fn import_indexed(
    path: &Path,
    scene_id: String,
    basis: DMat4,
) -> Result<(crate::exchange::ImportedGraph, GltfIndex)> {
    fn import_shape_keys(
        mesh_value: &Value,
        welded_source_ids: Option<&[u32]>,
        mesh: &Mesh,
        morph_deltas: &[Vec<[f64; 3]>],
    ) -> Result<(ShapeKeyData, Vec<Id>)> {
        if welded_source_ids.is_some()
            && mesh_value["extras"]["potter"]["shape_key_data"].is_object()
        {
            let mut shape_keys = serde_json::from_value::<ShapeKeyData>(
                mesh_value["extras"]["potter"]["shape_key_data"].clone(),
            )
            .map_err(|error| {
                import_error(format!("glTF shape-key metadata is invalid: {error}"))
            })?;
            let source_vertex_ids = welded_source_ids
                .ok_or_else(|| import_error("glTF shape-key vertex IDs are missing"))?;
            // One entry per welded vertex, so each source vertex maps to exactly one mesh vertex.
            let vertex_id_map = source_vertex_ids
                .iter()
                .zip(&mesh.vertices)
                .map(|(source_id, vertex)| (*source_id, vertex.id))
                .collect::<BTreeMap<_, _>>();
            let remap_positions = |positions: BTreeMap<u32, [f64; 3]>| {
                positions
                    .into_iter()
                    .map(|(source_id, position)| {
                        let vertex_id = vertex_id_map.get(&source_id).ok_or_else(|| {
                            import_error("glTF shape-key metadata references a missing vertex")
                        })?;
                        Ok::<_, PotError>((*vertex_id, position))
                    })
                    .collect::<Result<BTreeMap<_, _>>>()
            };
            shape_keys.basis = remap_positions(std::mem::take(&mut shape_keys.basis))?;
            for key in shape_keys.keys.values_mut() {
                key.positions = remap_positions(std::mem::take(&mut key.positions))?;
            }
            let ids = mesh_value["extras"]["potter"]["shape_keys"]
                .as_array()
                .map_or_else(Vec::new, |entries| {
                    entries
                        .iter()
                        .filter_map(|entry| entry["id"].as_str().and_then(|id| Id::new(id).ok()))
                        .collect()
                });
            return Ok((shape_keys, ids));
        }
        let metadata = mesh_value["extras"]["potter"]["shape_keys"]
            .as_array()
            .map_or(&[][..], Vec::as_slice);
        let names = mesh_value["extras"]["targetNames"]
            .as_array()
            .map_or(&[][..], Vec::as_slice);
        let weights = mesh_value["weights"]
            .as_array()
            .map_or(&[][..], Vec::as_slice);
        let basis = mesh
            .vertices
            .iter()
            .map(|vertex| (vertex.id, vertex.co.to_array()))
            .collect::<BTreeMap<_, _>>();
        let mut used_ids = BTreeSet::new();
        let mut key_ids = Vec::with_capacity(morph_deltas.len());
        let mut raw_to_id = BTreeMap::new();
        for index in 0..morph_deltas.len() {
            let raw_id = metadata
                .get(index)
                .and_then(|value| value["id"].as_str())
                .or_else(|| names.get(index).and_then(Value::as_str))
                .unwrap_or("shape_key");
            let id = unique_import_id(raw_id, "shape_key", index, &mut used_ids)?;
            raw_to_id.insert(raw_id.to_owned(), id.clone());
            key_ids.push(id);
        }
        let mut key_values = Vec::with_capacity(morph_deltas.len());
        for (index, id) in key_ids.iter().enumerate() {
            let entry = metadata.get(index).unwrap_or(&Value::Null);
            let raw_relative = entry["relative_key"].as_str();
            let relative_key = raw_relative
                .map(|raw| {
                    raw_to_id
                        .get(raw)
                        .cloned()
                        .ok_or_else(|| import_error("glTF shape key relative reference is missing"))
                })
                .transpose()?;
            let value = weights
                .get(index)
                .and_then(Value::as_f64)
                .or_else(|| entry["value"].as_f64())
                .unwrap_or(0.0);
            let name = entry["name"]
                .as_str()
                .or_else(|| names.get(index).and_then(Value::as_str))
                .map_or_else(|| id.to_string(), str::to_owned);
            let vertex_group = entry["vertex_group"]
                .as_str()
                .map(Id::new)
                .transpose()
                .map_err(|_| import_error("glTF shape-key vertex group ID is invalid"))?;
            key_values.push(ShapeKey {
                id: id.clone(),
                name,
                value,
                mute: false,
                slider_min: entry["slider_min"].as_f64().unwrap_or(0.0),
                frame: entry["frame"].as_f64().unwrap_or(0.0),
                slider_max: entry["slider_max"].as_f64().unwrap_or(1.0),
                relative_key,
                vertex_group,
                positions: BTreeMap::new(),
            });
        }
        let mut result = ShapeKeyData {
            basis,
            absolute: false,
            evaluation_time: 0.0,
            action: None,
            action_slot: None,
            muted_action_curves: BTreeSet::new(),
            keys: BTreeMap::new(),
        };
        let mut pending = (0..key_values.len()).collect::<BTreeSet<_>>();
        while !pending.is_empty() {
            let mut completed = Vec::new();
            for index in &pending {
                let key = &key_values[*index];
                let relative = if let Some(relative_id) = &key.relative_key {
                    let Some(relative_key) = result.keys.get(relative_id) else {
                        continue;
                    };
                    &relative_key.positions
                } else {
                    &result.basis
                };
                let mut positions = BTreeMap::new();
                for (vertex_index, vertex) in mesh.vertices.iter().enumerate() {
                    let base = DVec3::from_array(
                        result
                            .basis
                            .get(&vertex.id)
                            .copied()
                            .unwrap_or(vertex.co.to_array()),
                    );
                    let reference = relative
                        .get(&vertex.id)
                        .copied()
                        .map_or(base, DVec3::from_array);
                    let delta = morph_deltas
                        .get(*index)
                        .and_then(|values| values.get(vertex_index))
                        .ok_or_else(|| import_error("glTF morph target count is invalid"))?;
                    let position = reference + DVec3::from_array(*delta);
                    if !position.is_finite() {
                        return Err(import_error("glTF morph target position is non-finite"));
                    }
                    positions.insert(vertex.id, position.to_array());
                }
                let mut imported = key.clone();
                imported.positions = positions;
                result.keys.insert(imported.id.clone(), imported);
                completed.push(*index);
            }
            if completed.is_empty() {
                return Err(import_error(
                    "glTF shape-key relative references contain a cycle",
                ));
            }
            for index in completed {
                pending.remove(&index);
            }
        }
        Ok((result, key_ids))
    }
    let bytes = fs::read(path).map_err(|error| PotError::io(&error))?;
    let is_glb = bytes.starts_with(b"glTF");
    let (json_bytes, glb_binary) = if is_glb {
        split_glb(&bytes)?
    } else {
        (bytes.as_slice(), None)
    };
    let root: Value = serde_json::from_slice(json_bytes).map_err(|error| {
        PotError::with_details(
            ErrorCode::ImportFailed,
            "invalid glTF JSON",
            json!({ "line": error.line(), "column": error.column() }),
        )
    })?;
    if root["asset"]["version"].as_str() != Some("2.0") {
        return Err(import_error("only glTF 2.0 is supported"));
    }
    let buffer_values = root["buffers"].as_array().map_or(&[][..], Vec::as_slice);
    let mut binaries = Vec::<Vec<u8>>::with_capacity(buffer_values.len());
    let mut compat_blobs = Vec::new();
    let mut assets = Vec::<(String, Vec<u8>)>::new();
    for (index, buffer) in buffer_values.iter().enumerate() {
        if index == 0 && glb_binary.is_some() {
            binaries.push(glb_binary.clone().unwrap_or_default());
            continue;
        }
        let uri = buffer["uri"]
            .as_str()
            .ok_or_else(|| import_error("glTF buffer URI is missing"))?;
        if uri.starts_with("data:") {
            binaries.push(decode_data_uri(uri)?);
        } else {
            let dependency = path.parent().unwrap_or_else(|| Path::new(".")).join(uri);
            let dependency_bytes = fs::read(&dependency).map_err(|error| PotError::io(&error))?;
            let dependency_name = dependency
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("buffer.bin")
                .to_owned();
            compat_blobs.push((dependency_name, dependency_bytes.clone()));
            binaries.push(dependency_bytes);
        }
    }

    let mut doc = SceneDoc::new(scene_id);
    let image_ids = import_images(&root, &binaries, path, &mut doc, &mut assets)?;
    if let Some(scene) = doc.scenes.get_mut(&doc.active_scene) {
        if let Some(fps) = root["extras"]["potter"]["fps"].as_u64() {
            scene.fps = u32::try_from(fps)
                .map_err(|_| import_error("glTF frame rate exceeds u32 range"))?;
        }
        scene.fps_base = root["extras"]["potter"]["fps_base"]
            .as_f64()
            .unwrap_or(scene.fps_base);
        scene.frame_current = root["extras"]["potter"]["frame"]
            .as_f64()
            .unwrap_or(scene.frame_current);
    }
    let material_values = root["materials"].as_array().map_or(&[][..], Vec::as_slice);
    let mut material_ids = Vec::with_capacity(material_values.len());
    let mut used_material_ids = BTreeSet::new();
    let mut graph_materials = Vec::<(Id, material_graph::GraphTextures)>::new();
    for (index, material) in material_values.iter().enumerate() {
        let suggested = material["extras"]["potter"]["id"]
            .as_str()
            .or_else(|| material["name"].as_str())
            .unwrap_or("material");
        let id = unique_import_id(suggested, "material", index, &mut used_material_ids)?;
        let mut color = [1.0; 4];
        if let Some(values) = material["pbrMetallicRoughness"]["baseColorFactor"].as_array() {
            for (target, source) in color.iter_mut().zip(values) {
                *target = source
                    .as_f64()
                    .ok_or_else(|| import_error("glTF material color is invalid"))?;
            }
        }
        let mut emission_color = [0.0; 3];
        if let Some(values) = material["emissiveFactor"].as_array() {
            for (target, source) in emission_color.iter_mut().zip(values) {
                *target = source
                    .as_f64()
                    .ok_or_else(|| import_error("glTF emission color is invalid"))?;
            }
        }
        // glTF's factor is Potter's color times strength, divided by the emissive-strength peak. Potter's
        // own strength (extras) wins so round trips keep the Potter split; the product stays the same.
        let gltf_peak =
            material["extensions"]["KHR_materials_emissive_strength"]["emissiveStrength"].as_f64();
        let emission_strength = material["extras"]["potter"]["emission_strength"]
            .as_f64()
            .or(gltf_peak)
            .unwrap_or(
                if crate::float::equal_f64_array(&emission_color, &[0.0; 3]) {
                    0.0
                } else {
                    1.0
                },
            );
        if !crate::float::equal_f64(emission_strength, 0.0) {
            let peak = gltf_peak.unwrap_or(1.0);
            for component in &mut emission_color {
                *component *= peak / emission_strength;
            }
        }
        let textures = material_textures(material, &root, &image_ids)?;
        // A texture a Potter texture field cannot hold (transform, emissive, occlusion, normal scale)
        // moves into the material graph, as Blender's importer builds one; the fields stay empty.
        let graph_needed = textures.needs_graph();
        let field = |slot: &Option<material_graph::SlotTexture>| {
            (!graph_needed)
                .then(|| slot.as_ref().map(|slot| slot.texture.clone()))
                .flatten()
        };
        let base_color_texture = field(&textures.base_color);
        let metallic_roughness_texture = field(&textures.metallic_roughness);
        let normal_texture = field(&textures.normal);
        let alpha_mode = match material["alphaMode"].as_str() {
            None | Some("OPAQUE") => "opaque",
            Some("BLEND") => "blend",
            Some("MASK") => "clip",
            _ => return Err(import_error("glTF material alphaMode is invalid")),
        };
        let value = Material {
            name: material["name"]
                .as_str()
                .map_or_else(|| id.to_string(), str::to_owned),
            base_color: color,
            alpha_mode: alpha_mode.to_owned(),
            alpha_threshold: material["alphaCutoff"].as_f64().unwrap_or(0.5),
            double_sided: material["doubleSided"].as_bool().unwrap_or(false),
            metallic: material["pbrMetallicRoughness"]["metallicFactor"]
                .as_f64()
                .unwrap_or(0.0),
            roughness: material["pbrMetallicRoughness"]["roughnessFactor"]
                .as_f64()
                .unwrap_or(1.0),
            emission_color,
            emission_strength,
            transmission:
                material["extensions"]["KHR_materials_transmission"]["transmissionFactor"]
                    .as_f64()
                    .unwrap_or(0.0),
            ior: material["extras"]["potter"]["ior"]
                .as_f64()
                .or_else(|| material["extensions"]["KHR_materials_ior"]["ior"].as_f64())
                .unwrap_or(1.5),
            normal_texture,
            base_color_texture,
            metallic_texture: metallic_roughness_texture.clone(),
            roughness_texture: metallic_roughness_texture,
            ..Material::default()
        };
        if graph_needed {
            graph_materials.push((id.clone(), textures));
        }
        doc.materials.insert(id.clone(), value);
        material_ids.push(id);
    }
    // Materials whose textures need a graph move them now; their texture fields are cleared.
    for (material_id, textures) in &graph_materials {
        material_graph::move_textures_to_graph(&mut doc, material_id, textures)?;
    }
    let mut used_graph_ids = doc.node_groups.keys().cloned().collect::<BTreeSet<_>>();
    let mut color_material_variants = BTreeMap::<(Id, String), Id>::new();
    let mut gltf_default_material = None::<Id>;

    let mut data_ids = Vec::<Id>::new();
    let mut mesh_skin_weights = Vec::<Option<Vec<SkinVertexWeights>>>::new();
    let mut mesh_ids_stale = Vec::<bool>::new();
    let mesh_values = root["meshes"].as_array().map_or(&[][..], Vec::as_slice);
    let mut used_data_ids = BTreeSet::new();
    let mut morph_normal_loss = false;
    let mut polygon_loss = false;
    let mut vertex_ids_loss = false;
    let mut color_attributes_loss = false;
    let mut color_attributes_core_fallback = false;
    let inverse_basis = basis.inverse();
    let baked_mesh_scales = baked_mesh_scales(&root)?;
    let mut mesh_shape_key_ids = Vec::<Vec<Id>>::new();
    for (mesh_index, mesh_value) in mesh_values.iter().enumerate() {
        let baked = baked_mesh_scales
            .get(&mesh_index)
            .copied()
            .unwrap_or(DVec3::ONE);
        let suggested = mesh_value["extras"]["potter"]["data_id"]
            .as_str()
            .or_else(|| mesh_value["name"].as_str())
            .unwrap_or("mesh");
        let data_id = unique_import_id(suggested, "mesh", mesh_index, &mut used_data_ids)?;
        let primitives = mesh_value["primitives"]
            .as_array()
            .ok_or_else(|| import_error("glTF mesh primitives are missing"))?;
        // Stored UV layers: one per `TEXCOORD_n` the primitives use, up to the layer limit.
        let uv_set_count = primitives
            .iter()
            .flat_map(|primitive| primitive["attributes"].as_object().into_iter().flatten())
            .filter_map(|(name, _)| name.strip_prefix("TEXCOORD_")?.parse::<usize>().ok())
            .map(|set| set.saturating_add(1))
            .max()
            .unwrap_or(0)
            .min(uv_layers::MAX_LAYERS);
        let mut positions = Vec::<DVec3>::new();
        let mut polygons = Vec::<Vec<usize>>::new();
        let mut face_material_indices = Vec::<u32>::new();
        let mut face_uvs = Vec::<Vec<CornerUvs>>::new();
        let mut slots = Vec::<(Id, Option<String>)>::new();
        let mut morph_target_count = None;
        let mut morph_deltas = Vec::<Vec<[f64; 3]>>::new();
        let mut joint_data = Vec::<Option<SkinVertexWeights>>::new();
        let mut normals = Vec::<DVec3>::new();
        let mut has_missing_normals = false;
        let mut morph_normal_deltas = Vec::<Vec<[f64; 3]>>::new();
        let mut morph_normals_missing = false;
        let mut primitive_starts = Vec::<usize>::new();
        let mut vertex_bases = BTreeMap::<VertexPayload, usize>::new();
        // Color layers: one per `COLOR_n` set the primitives use.
        let color_set_numbers = primitives
            .iter()
            .flat_map(|primitive| primitive["attributes"].as_object().into_iter().flatten())
            .filter_map(|(name, _)| name.strip_prefix("COLOR_")?.parse::<usize>().ok())
            .collect::<BTreeSet<_>>();
        let mut color_sets = BTreeMap::<usize, Vec<Option<[f64; 4]>>>::new();
        let mut face_color_sets = BTreeMap::<usize, Vec<Vec<Option<[f64; 4]>>>>::new();
        for primitive in primitives {
            let mode = primitive["mode"].as_u64().unwrap_or(4);
            let accessor_index = usize::try_from(
                primitive["attributes"]["POSITION"]
                    .as_u64()
                    .ok_or_else(|| import_error("glTF primitive POSITION is missing"))?,
            )
            .map_err(|_| import_error("glTF POSITION accessor exceeds platform range"))?;
            let decoded = read_vec3(&root, &binaries, accessor_index)?;
            // Primitives share vertices only when their whole vertex payload matches.
            let payload = vertex_payload(primitive);
            let first_use = !vertex_bases.contains_key(&payload);
            let base = *vertex_bases.entry(payload).or_insert(positions.len());
            if first_use {
                positions.extend(
                    decoded
                        .iter()
                        .map(|point| DVec3::from_array(potter_point(inverse_basis, baked, *point))),
                );
            }
            // Base NORMAL values, kept for the morph NORMAL deltas of this primitive.
            let mut base_normal_values = None;
            if let Some(accessor) = primitive["attributes"]["NORMAL"].as_u64() {
                let accessor = usize::try_from(accessor)
                    .map_err(|_| import_error("glTF NORMAL accessor exceeds platform range"))?;
                let values = read_vec3(&root, &binaries, accessor)?;
                if values.len() != decoded.len() {
                    return Err(import_error("glTF NORMAL count does not match POSITION"));
                }
                if first_use {
                    normals.extend(values.iter().map(|normal| {
                        DVec3::from_array(potter_normal(inverse_basis, baked, *normal))
                    }));
                }
                base_normal_values = Some(values);
            } else {
                has_missing_normals = true;
                if first_use {
                    normals.extend(std::iter::repeat_n(DVec3::ZERO, decoded.len()));
                }
            }
            // Each stored set is read per vertex; a primitive without the set reads `[0, 0]`.
            let mut primitive_uvs = Vec::<Vec<[f64; 2]>>::with_capacity(uv_set_count);
            for set in 0..uv_set_count {
                let Some(accessor) = primitive["attributes"][format!("TEXCOORD_{set}")].as_u64()
                else {
                    primitive_uvs.push(Vec::new());
                    continue;
                };
                let accessor = usize::try_from(accessor)
                    .map_err(|_| import_error("glTF TEXCOORD accessor exceeds platform range"))?;
                let values = read_vec2(&root, &binaries, accessor)?;
                if values.len() != decoded.len() {
                    return Err(import_error("glTF TEXCOORD count does not match POSITION"));
                }
                primitive_uvs.push(values);
            }
            // Each COLOR_n set is read per vertex. A primitive without the set leaves its vertices unset.
            let mut primitive_colors = BTreeMap::<usize, Vec<[f64; 4]>>::new();
            for set in &color_set_numbers {
                let Some(accessor) = primitive["attributes"][format!("COLOR_{set}")].as_u64()
                else {
                    continue;
                };
                let accessor = usize::try_from(accessor)
                    .map_err(|_| import_error("glTF COLOR accessor exceeds platform range"))?;
                let values = read_color(&root, &binaries, accessor)?;
                if values.len() != decoded.len() {
                    return Err(import_error("glTF COLOR count does not match POSITION"));
                }
                primitive_colors.insert(*set, values);
            }
            // Point values fill loose vertices; face rows below retain primitive-specific colors.
            for set in &color_set_numbers {
                let values_by_vertex = color_sets.entry(*set).or_default();
                values_by_vertex.resize(positions.len(), None);
                if let Some(values) = primitive_colors.get(set) {
                    for (target, value) in values_by_vertex[base..base + values.len()]
                        .iter_mut()
                        .zip(values)
                    {
                        if target.is_none() {
                            *target = Some(*value);
                        }
                    }
                }
            }
            let target_values = primitive["targets"]
                .as_array()
                .map_or(&[][..], Vec::as_slice);
            if let Some(expected) = morph_target_count {
                if expected != target_values.len() {
                    return Err(import_error(
                        "glTF mesh primitives have different morph target counts",
                    ));
                }
            } else {
                morph_target_count = Some(target_values.len());
                morph_deltas.resize_with(target_values.len(), Vec::new);
                morph_normal_deltas.resize_with(target_values.len(), Vec::new);
            }
            for (target_index, target) in target_values.iter().enumerate() {
                if target["TANGENT"].is_object() {
                    return Err(unsupported_export(
                        "mesh.morph_tangents",
                        "glTF morph tangent targets are not represented by Potter shape keys",
                    ));
                }
                let accessor = usize::try_from(
                    target["POSITION"]
                        .as_u64()
                        .ok_or_else(|| import_error("glTF morph POSITION target is missing"))?,
                )
                .map_err(|_| import_error("glTF morph accessor exceeds platform range"))?;
                let values = read_vec3(&root, &binaries, accessor)?;
                if values.len() != decoded.len() {
                    return Err(import_error(
                        "glTF morph target count does not match POSITION",
                    ));
                }
                if first_use {
                    morph_deltas[target_index].extend(
                        values
                            .iter()
                            .map(|delta| potter_point(inverse_basis, baked, *delta)),
                    );
                }
                if let Some(accessor) = target["NORMAL"].as_u64() {
                    let accessor = usize::try_from(accessor).map_err(|_| {
                        import_error("glTF morph NORMAL accessor exceeds platform range")
                    })?;
                    let values = read_vec3(&root, &binaries, accessor)?;
                    if values.len() != decoded.len() {
                        return Err(import_error(
                            "glTF morph NORMAL count does not match POSITION",
                        ));
                    }
                    if first_use {
                        morph_normal_deltas[target_index].extend(values.iter().enumerate().map(
                            |(vertex, delta)| {
                                potter_normal_delta(
                                    inverse_basis,
                                    baked,
                                    base_normal_values.as_deref().map(|base| base[vertex]),
                                    *delta,
                                )
                            },
                        ));
                    }
                } else {
                    morph_normals_missing = true;
                    if first_use {
                        morph_normal_deltas[target_index]
                            .extend(std::iter::repeat_n([0.0; 3], decoded.len()));
                    }
                }
            }
            if let Some(values) = read_skin_attributes(primitive, &root, &binaries, decoded.len())?
            {
                if first_use {
                    joint_data.extend(values.into_iter().map(Some));
                }
            } else if first_use {
                joint_data.extend(std::iter::repeat_n(None, decoded.len()));
            }
            let indices = if let Some(accessor) = primitive["indices"].as_u64() {
                read_indices(
                    &root,
                    &binaries,
                    usize::try_from(accessor)
                        .map_err(|_| import_error("glTF index accessor exceeds platform range"))?,
                )?
            } else {
                (0..decoded.len()).collect()
            };
            let color_layer = primitive_colors.get(&0).map(|_| {
                primitive["extras"]["potter"]["color_layer"]
                    .as_str()
                    .unwrap_or("Col")
                    .to_owned()
            });
            let source_material_id = if let Some(index) = primitive["material"].as_u64() {
                let material_index = usize::try_from(index)
                    .map_err(|_| import_error("glTF material index exceeds platform range"))?;
                material_ids
                    .get(material_index)
                    .cloned()
                    .ok_or_else(|| import_error("glTF primitive references a missing material"))?
            } else if let Some(id) = &gltf_default_material {
                id.clone()
            } else {
                let id = unique_import_id(
                    "gltf_default_material",
                    "material",
                    used_material_ids.len(),
                    &mut used_material_ids,
                )?;
                let material = Material {
                    name: "glTF Default".to_owned(),
                    base_color: [1.0; 4],
                    alpha_mode: "opaque".to_owned(),
                    ..Material::default()
                };
                doc.materials.insert(id.clone(), material);
                gltf_default_material = Some(id.clone());
                id
            };
            let binding = (source_material_id, color_layer);
            let material_slot =
                if let Some(slot) = slots.iter().position(|candidate| candidate == &binding) {
                    u32::try_from(slot).map_err(|_| import_error("too many glTF material slots"))?
                } else {
                    let slot = u32::try_from(slots.len())
                        .map_err(|_| import_error("too many glTF material slots"))?;
                    slots.push(binding);
                    slot
                };
            let face_start = polygons.len();
            primitive_starts.push(face_start);
            append_triangles(
                &indices,
                mode,
                decoded.len(),
                base,
                &mut polygons,
                &mut face_material_indices,
            )?;
            face_material_indices[face_start..].fill(material_slot);
            for polygon in &polygons[face_start..] {
                let uv: Vec<CornerUvs> = polygon
                    .iter()
                    .map(|vertex_index| {
                        primitive_uvs
                            .iter()
                            .map(|values| {
                                values
                                    .get(vertex_index.saturating_sub(base))
                                    .copied()
                                    .unwrap_or([0.0, 0.0])
                            })
                            .collect::<CornerUvs>()
                    })
                    .collect();
                face_uvs.push(uv);
                for set in &color_set_numbers {
                    let row = primitive_colors.get(set).map_or_else(
                        || vec![None; polygon.len()],
                        |values| {
                            polygon
                                .iter()
                                .map(|vertex_index| {
                                    values.get(vertex_index.saturating_sub(base)).copied()
                                })
                                .collect()
                        },
                    );
                    face_color_sets.entry(*set).or_default().push(row);
                }
            }
        }
        primitive_starts.push(polygons.len());
        // Copies of one Potter vertex, split per UV and normal, rejoin here; corners keep their own normals.
        // Potter's vertex IDs must describe these vertices. Another tool may have changed them (the
        // VRM Add-on keeps mesh extras but drops root ones), so stale IDs are ignored and reported.
        let parsed_ids = potter_vertex_ids(mesh_value)?;
        let ids_stale = parsed_ids
            .as_ref()
            .is_some_and(|ids| ids.len() != positions.len());
        vertex_ids_loss |= ids_stale;
        let source_ids = parsed_ids.filter(|_| !ids_stale);
        let (welded_of, first) = weld_copies(positions.len(), source_ids.as_deref())?;
        if welded_of
            .iter()
            .zip(&positions)
            .any(|(welded, point)| *point != positions[first[*welded]])
        {
            return Err(import_error(
                "glTF copies of one Potter vertex disagree on position",
            ));
        }
        let source_ids_welded =
            source_ids.map(|ids| first.iter().map(|index| ids[*index]).collect::<Vec<_>>());
        let welded_positions = first
            .iter()
            .map(|index| positions[*index])
            .collect::<Vec<_>>();
        // Potter's stored polygons replace the triangles, which carry only materials and UVs here.
        // A record that does not describe these triangles falls back to them and reports a loss.
        let parsed_polygons = potter_polygons(
            mesh_value,
            &root,
            &binaries,
            positions.len(),
            primitives.len(),
        );
        // A record written before another tool changed the vertices is stale: its fields are sized for
        // the old vertex count, so it no longer parses. Stale vertex IDs make that case a loss.
        let polygon_record = match parsed_polygons {
            Err(_) if ids_stale => PotterPolygons::UnknownVersion,
            parsed => parsed?,
        };
        let restored = match polygon_record {
            PotterPolygons::Absent => None,
            PotterPolygons::UnknownVersion => {
                polygon_loss = true;
                None
            }
            PotterPolygons::Faces(mut faces) => {
                // A mirrored bake reversed each face's corners at export; the import undoes the bake, so restore them.
                if baked.element_product() < 0.0 {
                    for corners in &mut faces.corners {
                        corners.reverse();
                    }
                }
                let restored = restore_polygons(
                    &faces,
                    &polygons,
                    &face_material_indices,
                    &face_uvs,
                    &face_color_sets,
                    &primitive_starts,
                );
                if restored.is_none() {
                    polygon_loss = true;
                }
                restored.map(|restored| (faces, restored))
            }
        };
        let faces_restored = restored.is_some();
        let (polygons, face_material_indices, face_uvs, face_color_rows, exact) = match restored {
            Some((faces, (materials, uvs, colors))) => {
                let exact = shading::ExactShading {
                    smooth: faces.smooth,
                    sharp: faces
                        .sharp
                        .iter()
                        .map(|[first, second]| [welded_of[*first], welded_of[*second]])
                        .collect(),
                };
                (faces.corners, materials, uvs, colors, Some(exact))
            }
            None => (
                polygons,
                face_material_indices,
                face_uvs,
                face_color_sets,
                None,
            ),
        };
        let welded_polygons = polygons
            .iter()
            .map(|polygon| {
                polygon
                    .iter()
                    .map(|index| welded_of[*index])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut mesh =
            Mesh::from_positions_and_faces(welded_positions, welded_polygons).map_err(|error| {
                PotError::with_details(
                    ErrorCode::ImportFailed,
                    "glTF mesh topology is invalid",
                    json!({ "reason": error.to_string(), "data_id": data_id }),
                )
            })?;
        for (face, material_index) in mesh.faces.iter_mut().zip(face_material_indices) {
            face.material_index = material_index;
        }
        if uv_set_count > 0 {
            let layers = (0..uv_set_count)
                .map(|set| {
                    UvLayer::new(
                        uv_set_layer_name(set),
                        face_uvs
                            .iter()
                            .map(|corners| Some(corners.iter().map(|corner| corner[set]).collect()))
                            .collect(),
                    )
                })
                .collect();
            let first = uv_set_layer_name(0);
            uv_layers::write(
                &mut mesh,
                &UvStore {
                    layers,
                    active: Some(first.clone()),
                    render: Some(first),
                },
            );
        }
        let has_core_color_data = color_sets
            .values()
            .any(|values| values.iter().any(Option::is_some))
            || face_color_rows
                .values()
                .flatten()
                .flatten()
                .any(Option::is_some);
        if !import_potter_color_attributes(
            &mut mesh,
            &mesh_value["extras"]["potter"]["color_attributes"],
            source_ids_welded.as_deref(),
            faces_restored,
            has_core_color_data,
        )? {
            import_color_layers(
                &mut mesh,
                &polygons,
                &color_sets,
                &face_color_rows,
                &welded_of,
                &first,
            )?;
            if !mesh_value["extras"]["potter"]["color_attributes"].is_null() {
                color_attributes_loss = true;
                color_attributes_core_fallback |= !color_set_numbers.is_empty();
            }
        }
        let morph_normals_complete = !morph_deltas.is_empty() && !morph_normals_missing;
        // Morph deltas, like positions, come from each welded vertex's first copy.
        let welded_morphs = morph_deltas
            .iter()
            .map(|deltas| first.iter().map(|index| deltas[*index]).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        if !has_missing_normals && !normals.is_empty() {
            let corner_normals = polygons
                .iter()
                .flatten()
                .map(|index| normals[*index])
                .collect::<Vec<_>>();
            let morph_targets = if morph_normals_complete {
                welded_morphs
                    .iter()
                    .zip(&morph_normal_deltas)
                    .map(|(positions, normal_values)| {
                        let corner_deltas = polygons
                            .iter()
                            .flatten()
                            .map(|index| normal_values[*index])
                            .collect::<Vec<_>>();
                        (positions.clone(), corner_deltas)
                    })
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            shading::import_shading_flags(&mut mesh, &corner_normals, exact.as_ref())?;
            if morph_normals_disagree(&mesh, &corner_normals, &morph_targets) {
                morph_normal_loss = true;
            }
        }
        let (shape_keys, morph_key_ids) = if morph_deltas.is_empty() {
            (None, Vec::new())
        } else {
            let (keys, ids) = import_shape_keys(
                mesh_value,
                source_ids_welded.as_deref(),
                &mesh,
                &welded_morphs,
            )?;
            (Some(keys), ids)
        };
        let joint_data = first
            .iter()
            .map(|index| joint_data[*index].clone())
            .collect::<Vec<_>>();
        let imported_joint_data = if joint_data.iter().any(Option::is_some) {
            if joint_data.iter().any(Option::is_none) {
                return Err(import_error(
                    "glTF mesh mixes skinned and unskinned primitives",
                ));
            }
            Some(
                joint_data
                    .into_iter()
                    .map(|value| {
                        value.ok_or_else(|| import_error("glTF skin attribute is missing"))
                    })
                    .collect::<Result<Vec<_>>>()?,
            )
        } else {
            None
        };
        let vertex_groups = mesh_value["extras"]["potter"]["vertex_groups"]
            .as_array()
            .map(|_| {
                serde_json::from_value::<Vec<VertexGroup>>(
                    mesh_value["extras"]["potter"]["vertex_groups"].clone(),
                )
                .map_err(|error| {
                    import_error(format!("glTF vertex group metadata is invalid: {error}"))
                })
            })
            .transpose()?
            .unwrap_or_default();
        let vertex_weights = mesh_value["extras"]["potter"]["vertex_weights"]
            .as_object()
            .map(|_| {
                serde_json::from_value::<BTreeMap<u32, BTreeMap<Id, f64>>>(
                    mesh_value["extras"]["potter"]["vertex_weights"].clone(),
                )
                .map_err(|error| {
                    import_error(format!("glTF vertex weight metadata is invalid: {error}"))
                })
            })
            .transpose()?
            .unwrap_or_default();
        let color_store = color_layers::read(&mesh);
        let data_materials = slots
            .into_iter()
            .map(|(source_material, source_layer)| {
                let Some(source_layer) = source_layer else {
                    return Ok(source_material);
                };
                let layer = if color_store.index_of(&source_layer).is_some() {
                    source_layer
                } else {
                    color_store
                        .layers
                        .first()
                        .map(|layer| layer.name.clone())
                        .ok_or_else(|| import_error("glTF COLOR_0 has no imported color layer"))?
                };
                imported_color_material(
                    &mut doc,
                    &source_material,
                    &layer,
                    &mut color_material_variants,
                    &mut used_material_ids,
                    &mut used_graph_ids,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        doc.data_blocks.insert(
            data_id.clone(),
            DataBlock {
                data_type: "mesh".to_owned(),
                descriptor: None,
                mesh: Some(mesh),
                materials: data_materials.into_iter().map(Some).collect(),
                shape_keys,
                vertex_groups,
                vertex_weights,
                light: None,
                ..DataBlock::default()
            },
        );
        data_ids.push(data_id);
        mesh_skin_weights.push(imported_joint_data);
        mesh_ids_stale.push(ids_stale);
        mesh_shape_key_ids.push(morph_key_ids);
    }

    let camera_ids = import_cameras(&root, &mut doc, &mut used_data_ids)?;
    let light_ids = import_lights(&root, &mut doc, &mut used_data_ids)?;
    let node_values = root["nodes"].as_array().map_or(&[][..], Vec::as_slice);
    let parents = node_parent_indices(node_values)?;
    let source_node_worlds = node_world_matrices(node_values, &parents)?;
    let inverse_basis = basis.inverse();
    let mut used_node_ids = BTreeSet::new();
    let node_ids = node_values
        .iter()
        .enumerate()
        .map(|(index, node)| {
            let suggested = node["extras"]["potter"]["id"]
                .as_str()
                .or_else(|| node["name"].as_str())
                .unwrap_or("node");
            unique_import_id(suggested, "node", index, &mut used_node_ids)
        })
        .collect::<Result<Vec<_>>>()?;
    let (imported_skins, joint_nodes, armature_data_ids, synthetic_roots) = import_skins(
        &root,
        &binaries,
        node_values,
        &parents,
        &source_node_worlds,
        &node_ids,
        &mut doc,
        &mut used_data_ids,
        &mut used_node_ids,
        basis,
    )?;
    let mut joint_bones = BTreeMap::<usize, (Id, Id)>::new();
    for skin in &imported_skins {
        for (joint_index, joint_id) in skin.joint_node_indices.iter().zip(&skin.joint_ids) {
            joint_bones.insert(*joint_index, (skin.armature_node.clone(), joint_id.clone()));
        }
    }
    let bone_worlds =
        imported_bone_world_matrices(&doc, &imported_skins, &node_ids, &source_node_worlds, basis)?;
    let mut restored_skins = BTreeMap::<Id, Id>::new();
    for (index, node_value) in node_values.iter().enumerate() {
        if joint_nodes.contains(&index) {
            continue;
        }
        let bone_parent =
            parents[index].and_then(|parent_index| joint_bones.get(&parent_index).cloned());
        let mut gltf_local = node_matrix(node_value)?;
        if let Some(scale) = baked_node_scale(node_value)? {
            gltf_local *= DMat4::from_scale(scale);
        }
        let local = if let Some((armature, bone)) = &bone_parent {
            let bone_world = bone_worlds
                .get(&(armature.clone(), bone.clone()))
                .copied()
                .ok_or_else(|| {
                    import_error("glTF bone-parented node has no imported bone frame")
                })?;
            let gltf_parent_world =
                parents[index].map_or(DMat4::IDENTITY, |parent| source_node_worlds[parent]);
            let source_world = inverse_basis * gltf_parent_world * gltf_local * basis;
            let inverse_bone = bone_world.inverse();
            if !finite_matrix(inverse_bone) {
                return Err(import_error("glTF bone-parent frame is singular"));
            }
            inverse_bone * source_world
        } else {
            inverse_basis * gltf_local * basis
        };
        let (scale, rotation, translation) = local.to_scale_rotation_translation();
        if !scale.is_finite() || !rotation.is_finite() || !translation.is_finite() {
            return Err(import_error(
                "glTF node transform contains non-finite values",
            ));
        }
        let transform = Transform::from_rotation_quat(
            translation.to_array(),
            [rotation.x, rotation.y, rotation.z, rotation.w],
            scale.to_array(),
        )?;
        let mesh_index = node_value["mesh"]
            .as_u64()
            .map(|value| {
                usize::try_from(value)
                    .map_err(|_| import_error("glTF mesh index exceeds platform range"))
            })
            .transpose()?;
        let skin_index = node_value["skin"]
            .as_u64()
            .map(|value| {
                usize::try_from(value)
                    .map_err(|_| import_error("glTF skin index exceeds platform range"))
            })
            .transpose()?;
        let skin = skin_index
            .map(|index| {
                imported_skins
                    .get(index)
                    .ok_or_else(|| import_error("glTF node references a missing skin"))
            })
            .transpose()?;
        let camera_index = node_value["camera"]
            .as_u64()
            .map(|value| {
                usize::try_from(value)
                    .map_err(|_| import_error("glTF camera index exceeds platform range"))
            })
            .transpose()?;
        let light_index = node_value["extensions"]["KHR_lights_punctual"]["light"]
            .as_u64()
            .map(|value| {
                usize::try_from(value)
                    .map_err(|_| import_error("glTF light index exceeds platform range"))
            })
            .transpose()?;
        let data = if let Some(mesh_index) = mesh_index {
            Some(
                data_ids
                    .get(mesh_index)
                    .cloned()
                    .ok_or_else(|| import_error("glTF node references a missing mesh"))?,
            )
        } else if let Some(camera_index) = camera_index {
            Some(
                camera_ids
                    .get(camera_index)
                    .cloned()
                    .ok_or_else(|| import_error("glTF node references a missing camera"))?,
            )
        } else if let Some(light_index) = light_index {
            Some(
                light_ids
                    .get(light_index)
                    .cloned()
                    .ok_or_else(|| import_error("glTF node references a missing light"))?,
            )
        } else {
            armature_data_ids.get(&index).cloned()
        };
        let mut modifiers = Vec::new();
        if let Some(skin) = skin {
            let mesh_index =
                mesh_index.ok_or_else(|| import_error("glTF node has a skin but no mesh"))?;
            let data_id = data_ids
                .get(mesh_index)
                .cloned()
                .ok_or_else(|| import_error("glTF skin node references a missing mesh"))?;
            if let Some(previous) = restored_skins.get(&data_id) {
                if previous != &skin.armature_node {
                    return Err(unsupported_export(
                        "rig.shared_mesh_multiple_skeletons",
                        "glTF skin import cannot bind one shared mesh Data-Block to multiple armatures",
                    ));
                }
            } else {
                restore_skin_weights(
                    &mut doc,
                    mesh_index,
                    skin,
                    mesh_values,
                    &data_ids,
                    &mesh_skin_weights,
                    mesh_ids_stale[mesh_index],
                )?;
                restored_skins.insert(data_id, skin.armature_node.clone());
            }
            let mut params = serde_json::Map::new();
            params.insert("object".to_owned(), json!(skin.armature_node));
            params.insert("use_vertex_groups".to_owned(), Value::Bool(true));
            modifiers.push(Modifier {
                id: Id::new("armature_skin")?,
                modifier_type: "armature".to_owned(),
                name: "Armature".to_owned(),
                enabled: true,
                show_viewport: true,
                show_render: true,
                params,
                binding_data: None,
                runtime: crate::model::ModifierRuntime::default(),
            });
        }
        let tags = node_value["extras"]["potter"]["tags"]
            .as_array()
            .map_or_else(Vec::new, |tags| {
                tags.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            });
        let kind = if armature_data_ids.contains_key(&index) {
            "armature".to_owned()
        } else {
            node_value["extras"]["potter"]["kind"].as_str().map_or_else(
                || {
                    if mesh_index.is_some() {
                        "mesh"
                    } else if camera_index.is_some() {
                        "camera"
                    } else if light_index.is_some() {
                        "light"
                    } else {
                        "group"
                    }
                    .to_owned()
                },
                str::to_owned,
            )
        };
        let (parent, parent_type, parent_bone) = if let Some((armature, bone)) = bone_parent {
            (Some(armature), crate::model::ParentType::Bone, Some(bone))
        } else {
            (
                parents[index]
                    .filter(|parent| !joint_nodes.contains(parent))
                    .map(|parent| node_ids[parent].clone()),
                crate::model::ParentType::Object,
                None,
            )
        };
        let pose = if node_value["extras"]["potter"]["pose"].is_object() {
            serde_json::from_value(node_value["extras"]["potter"]["pose"].clone())
                .map_err(|error| import_error(format!("glTF armature pose is invalid: {error}")))?
        } else {
            BTreeMap::new()
        };
        let properties = node_value["extras"]["potter"]["properties"]
            .as_object()
            .cloned()
            .unwrap_or_default();
        doc.nodes.insert(
            node_ids[index].clone(),
            Node {
                name: node_value["name"]
                    .as_str()
                    .map_or_else(|| node_ids[index].to_string(), str::to_owned),
                kind,
                primitive: None,
                tags,
                parent,
                parent_type,
                parent_bone,
                parent_inverse: None,
                transform,
                data,
                modifiers,
                visible: true,
                render_visible: true,
                selectable: true,
                action: None,
                pose,
                properties,
                ..Node::default()
            },
        );
        if doc.scenes.contains_key(&doc.active_scene)
            && camera_index.is_some()
            && let Some(scene) = doc.scenes.get_mut(&doc.active_scene)
            && scene.camera.is_none()
        {
            scene.camera = Some(node_ids[index].clone());
        }
    }
    let root_collection = Id::new("collection_root")?;
    let mut scene_roots = parents
        .iter()
        .enumerate()
        .filter(|&(index, parent)| {
            !joint_nodes.contains(&index)
                && parent.is_none_or(|parent| joint_nodes.contains(&parent))
        })
        .map(|(index, _parent)| node_ids[index].clone())
        .collect::<Vec<_>>();
    scene_roots.extend(synthetic_roots);
    doc.collections
        .get_mut(&root_collection)
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "root collection is missing"))?
        .objects = scene_roots;
    let mut losses = import_animations(&root, &binaries, &node_ids, &mut doc, basis)?;
    losses.extend(gltf_import_losses(&root));
    if morph_normal_loss {
        losses.push(crate::exchange::Loss {
            feature_id: "mesh.morph_normals".to_owned(),
            data_id: None,
            reason:
                "glTF morph NORMAL targets disagree with normals derived from Potter shape keys"
                    .to_owned(),
            suggestion: Some(
                "use --allow-lossy to import the file with derived shading".to_owned(),
            ),
        });
    }
    if polygon_loss {
        losses.push(crate::exchange::Loss {
            feature_id: "mesh.polygon_topology".to_owned(),
            data_id: None,
            reason: "glTF Potter polygon topology has an unknown version and imports as triangles"
                .to_owned(),
            suggestion: Some(
                "use --allow-lossy to import the file with triangulated faces".to_owned(),
            ),
        });
    }
    if vertex_ids_loss {
        losses.push(crate::exchange::Loss {
            feature_id: "mesh.vertex_ids".to_owned(),
            data_id: None,
            reason: "glTF Potter vertex IDs do not match the mesh vertices; the file was edited by another tool, so its Potter vertex, shape-key, weight and group metadata are ignored".to_owned(),
            suggestion: Some(
                "use --allow-lossy to import the geometry without the stale Potter metadata".to_owned(),
            ),
        });
    }
    if color_attributes_loss {
        losses.push(crate::exchange::Loss {
            feature_id: "mesh.color_attribute".to_owned(),
            data_id: None,
            reason: if color_attributes_core_fallback {
                "Potter's named color layers could not be restored because their source vertex IDs or polygon topology no longer match; only glTF COLOR_n values remain".to_owned()
            } else {
                "Potter's named color layers could not be restored because their source vertex IDs or polygon topology no longer match; no core COLOR_n values were available".to_owned()
            },
            suggestion: Some(
                "use --allow-lossy to import the file with the available glTF color data".to_owned(),
            ),
        });
    }
    doc.validate()?;
    let node_worlds = source_node_worlds;
    let mut material_variants = BTreeMap::<Id, BTreeMap<String, Id>>::new();
    for ((material_id, layer), variant_id) in color_material_variants {
        material_variants
            .entry(material_id)
            .or_default()
            .insert(layer, variant_id);
    }
    let index = GltfIndex {
        root,
        basis,
        node_ids,
        node_worlds,
        material_ids,
        color_material_variants: material_variants,
        image_ids,
        mesh_shape_key_ids,
        joint_bones,
    };
    let graph = crate::exchange::ImportedGraph {
        doc,
        losses,
        id_mappings: json!({}),
        compat_blobs,
        assets,
        source: json!({ "format": if is_glb { "glb" } else { "gltf" }, "path": path.display().to_string() }),
    };
    Ok((graph, index))
}

/// Morph position deltas per welded vertex and morph NORMAL deltas per face corner, in Potter axes.
type MorphNormalTarget = (Vec<[f64; 3]>, Vec<[f64; 3]>);

/// Vertex-domain accessors of a glTF primitive. `TEXCOORD_n` and `COLOR_n` are left out because both are
/// read per face corner before UV storage and POINT/CORNER color-domain inference.
type VertexPayload = (
    BTreeMap<String, Option<u64>>,
    Vec<BTreeMap<String, Option<u64>>>,
);

fn vertex_payload(primitive: &Value) -> VertexPayload {
    fn accessors(value: &Value) -> BTreeMap<String, Option<u64>> {
        value
            .as_object()
            .into_iter()
            .flatten()
            .filter(|(name, _)| !name.starts_with("TEXCOORD_") && !name.starts_with("COLOR_"))
            .map(|(name, accessor)| (name.clone(), accessor.as_u64()))
            .collect()
    }
    let targets = primitive["targets"]
        .as_array()
        .map_or_else(Vec::new, |targets| targets.iter().map(accessors).collect());
    (accessors(&primitive["attributes"]), targets)
}

/// Potter's `vertex_ids`: the source vertex of each glTF vertex. Third-party files have none.
fn potter_vertex_ids(mesh_value: &Value) -> Result<Option<Vec<u32>>> {
    match &mesh_value["extras"]["potter"]["vertex_ids"] {
        Value::Null => Ok(None),
        Value::Array(values) => values
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|id| u32::try_from(id).ok())
                    .ok_or_else(|| import_error("glTF source vertex ID is invalid"))
            })
            .collect::<Result<Vec<_>>>()
            .map(Some),
        _ => Err(import_error("glTF source vertex IDs are invalid")),
    }
}

/// Imports `COLOR_n` values per primitive face corner, then welds agreeing values to the POINT domain.
/// A welded vertex whose corners disagree makes the layer CORNER-domain; missing primitive sets use
/// glTF's opaque-white default.
fn import_color_layers(
    mesh: &mut Mesh,
    polygons: &[Vec<usize>],
    sets: &BTreeMap<usize, Vec<Option<[f64; 4]>>>,
    face_sets: &BTreeMap<usize, Vec<Vec<Option<[f64; 4]>>>>,
    welded_of: &[usize],
    first: &[usize],
) -> Result<()> {
    const DEFAULT: [f64; 4] = [1.0; 4];
    let mut layers = Vec::<ColorLayer>::new();
    for (&set, values) in sets {
        let face_rows = face_sets.get(&set);
        let has_face_values =
            face_rows.is_some_and(|rows| rows.iter().flatten().any(Option::is_some));
        if values.iter().all(Option::is_none) && !has_face_values {
            continue;
        }
        if face_rows.is_some_and(|rows| rows.len() != polygons.len()) {
            return Err(import_error(
                "glTF color face values do not match mesh polygons",
            ));
        }
        let mut corner_values = Vec::new();
        let mut welded_values = vec![None::<[f64; 4]>; first.len()];
        let mut disagree = false;
        for (face_index, polygon) in polygons.iter().enumerate() {
            let row = face_rows.and_then(|rows| rows.get(face_index));
            if row.is_none() && !polygon.is_empty() {
                return Err(import_error("glTF color face values are missing"));
            }
            if row.is_some_and(|row| row.len() != polygon.len()) {
                return Err(import_error("glTF color face row has the wrong length"));
            }
            for (corner, &vertex) in polygon.iter().enumerate() {
                let value = row
                    .and_then(|row| row.get(corner))
                    .copied()
                    .flatten()
                    .unwrap_or(DEFAULT);
                corner_values.push(value);
                let welded = welded_of[vertex];
                match welded_values[welded] {
                    None => welded_values[welded] = Some(value),
                    Some(seen) => disagree |= !crate::float::equal_f64_array(&seen, &value),
                }
            }
        }
        let (domain, data) = if disagree {
            (ColorDomain::Corner, corner_values)
        } else {
            let per_vertex = first
                .iter()
                .zip(&welded_values)
                .map(|(copy, value)| {
                    value.unwrap_or_else(|| values.get(*copy).copied().flatten().unwrap_or(DEFAULT))
                })
                .collect::<Vec<_>>();
            (ColorDomain::Point, per_vertex)
        };
        let name = if layers.is_empty() {
            "Col".to_owned()
        } else {
            format!("Col.{:03}", layers.len())
        };
        layers.push(ColorLayer {
            name,
            domain,
            values: data,
        });
    }
    let Some(active) = layers.first().map(|layer| layer.name.clone()) else {
        return Ok(());
    };
    color_layers::write(
        mesh,
        &ColorStore {
            layers,
            active: Some(active.clone()),
            render: Some(active),
        },
    );
    Ok(())
}

fn potter_color_rgba(value: &Value) -> Result<[f64; 4]> {
    let values = value
        .as_array()
        .filter(|values| values.len() == 4)
        .ok_or_else(|| import_error("glTF Potter color value is invalid"))?;
    let color = [
        values[0]
            .as_f64()
            .ok_or_else(|| import_error("glTF Potter color value is invalid"))?,
        values[1]
            .as_f64()
            .ok_or_else(|| import_error("glTF Potter color value is invalid"))?,
        values[2]
            .as_f64()
            .ok_or_else(|| import_error("glTF Potter color value is invalid"))?,
        values[3]
            .as_f64()
            .ok_or_else(|| import_error("glTF Potter color value is invalid"))?,
    ];
    if color.iter().any(|channel| !channel.is_finite()) {
        return Err(import_error("glTF Potter color value is non-finite"));
    }
    Ok(color)
}

fn potter_color_element_ids(value: &Value, field: &str) -> Result<Option<Vec<u32>>> {
    let Some(values) = value[field].as_array() else {
        if value[field].is_null() {
            return Ok(None);
        }
        return Err(import_error(format!(
            "glTF Potter color {field} are invalid"
        )));
    };
    values
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .ok_or_else(|| import_error(format!("glTF Potter color {field} are invalid")))
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

/// Restores Potter's exact named/domain-aware colors when its element and topology IDs still match.
fn import_potter_color_attributes(
    mesh: &mut Mesh,
    metadata: &Value,
    source_vertex_ids: Option<&[u32]>,
    faces_restored: bool,
    has_core_color_data: bool,
) -> Result<bool> {
    if metadata.is_null() || metadata["version"].as_u64() != Some(1) {
        return Ok(false);
    }
    let Some(vertex_ids) = potter_color_element_ids(metadata, "vertex_ids")? else {
        return Ok(false);
    };
    let face_ids = potter_color_element_ids(metadata, "face_ids")?;
    if vertex_ids.iter().collect::<BTreeSet<_>>().len() != vertex_ids.len()
        || face_ids
            .as_ref()
            .is_some_and(|ids| ids.iter().collect::<BTreeSet<_>>().len() != ids.len())
    {
        return Err(import_error("glTF Potter color element IDs are duplicated"));
    }
    let raw_layers = metadata["layers"]
        .as_array()
        .ok_or_else(|| import_error("glTF Potter color layers are missing"))?;
    if raw_layers.is_empty() && has_core_color_data {
        return Ok(false);
    }
    let mut names = BTreeSet::new();
    let mut layers = Vec::with_capacity(raw_layers.len());
    for raw in raw_layers {
        let name = raw["name"]
            .as_str()
            .filter(|name| !name.is_empty())
            .ok_or_else(|| import_error("glTF Potter color layer name is invalid"))?;
        if !names.insert(name.to_owned()) {
            return Err(import_error("glTF Potter color layer names are duplicated"));
        }
        let domain = match raw["domain"].as_str() {
            Some("point") => ColorDomain::Point,
            Some("corner") => ColorDomain::Corner,
            _ => return Err(import_error("glTF Potter color layer domain is invalid")),
        };
        let values = match domain {
            ColorDomain::Point => {
                let Some(source_ids) = source_vertex_ids else {
                    return Ok(false);
                };
                let raw_values = raw["values"]
                    .as_array()
                    .ok_or_else(|| import_error("glTF Potter point colors are missing"))?;
                if vertex_ids.len() != raw_values.len() {
                    return Err(import_error(
                        "glTF Potter point colors do not match their IDs",
                    ));
                }
                let source_set = source_ids.iter().copied().collect::<BTreeSet<_>>();
                let vertex_set = vertex_ids.iter().copied().collect::<BTreeSet<_>>();
                if source_ids.len() != mesh.vertices.len()
                    || source_set.len() != source_ids.len()
                    || source_set != vertex_set
                {
                    return Ok(false);
                }
                let values_by_id = vertex_ids
                    .iter()
                    .copied()
                    .zip(
                        raw_values
                            .iter()
                            .map(potter_color_rgba)
                            .collect::<Result<Vec<_>>>()?,
                    )
                    .collect::<BTreeMap<_, _>>();
                source_ids
                    .iter()
                    .map(|id| {
                        values_by_id
                            .get(id)
                            .copied()
                            .ok_or_else(|| import_error("glTF Potter point color ID is missing"))
                    })
                    .collect::<Result<Vec<_>>>()?
            }
            ColorDomain::Corner => {
                if !faces_restored {
                    return Ok(false);
                }
                let raw_rows = raw["values"]
                    .as_array()
                    .ok_or_else(|| import_error("glTF Potter corner colors are missing"))?;
                let Some(face_ids) = face_ids.as_ref() else {
                    return Ok(false);
                };
                if face_ids.len() != mesh.faces.len() || raw_rows.len() != mesh.faces.len() {
                    return Ok(false);
                }
                let mut values = Vec::new();
                for (face, row) in mesh.faces.iter().zip(raw_rows) {
                    let row = row
                        .as_array()
                        .ok_or_else(|| import_error("glTF Potter color face row is invalid"))?;
                    if row.len() != face.vertices.len() {
                        return Ok(false);
                    }
                    values.extend(
                        row.iter()
                            .map(potter_color_rgba)
                            .collect::<Result<Vec<_>>>()?,
                    );
                }
                values
            }
        };
        layers.push(ColorLayer {
            name: name.to_owned(),
            domain,
            values,
        });
    }
    let designation = |key: &str| -> Result<Option<String>> {
        match &metadata[key] {
            Value::Null => Ok(None),
            Value::String(name) => Ok(Some(name.clone())),
            _ => Err(import_error(format!(
                "glTF Potter color {key} designation is invalid"
            ))),
        }
    };
    let store = ColorStore {
        layers,
        active: designation("active")?,
        render: designation("render")?,
    };
    for name in [store.active.as_deref(), store.render.as_deref()]
        .into_iter()
        .flatten()
    {
        if store.index_of(name).is_none() {
            return Err(import_error(
                "glTF Potter color designation names a missing layer",
            ));
        }
    }
    color_layers::write(mesh, &store);
    color_layers::validate(mesh)?;
    Ok(true)
}

fn available_graph_node_id(graph: &crate::graph::NodeGroup, stem: &str) -> Result<Id> {
    let mut suffix = 0_usize;
    loop {
        let name = if suffix == 0 {
            stem.to_owned()
        } else {
            format!("{stem}_{suffix}")
        };
        let id = Id::new(name)?;
        if !graph.nodes.contains_key(&id) {
            return Ok(id);
        }
        suffix = suffix.saturating_add(1);
    }
}

pub(crate) fn bind_imported_vertex_color(
    graph: &mut crate::graph::NodeGroup,
    layer: &str,
    material: &Material,
) -> Result<()> {
    use crate::graph::{GraphLink, GraphNode};

    let bsdf_id = graph
        .nodes
        .iter()
        .find(|(_, node)| node.node_type == "ShaderNodeBsdfPrincipled")
        .map(|(id, _)| id.clone())
        .ok_or_else(|| import_error("glTF material graph has no Principled BSDF"))?;
    let bsdf = graph
        .nodes
        .get(&bsdf_id)
        .ok_or_else(|| import_error("glTF material Principled BSDF is missing"))?;
    let base_input = bsdf
        .inputs
        .get("Base Color")
        .cloned()
        .unwrap_or_else(|| json!(material.base_color));
    let base_link = graph
        .links
        .iter()
        .find(|link| link.to_node == bsdf_id && link.to_socket == "Base Color")
        .cloned();
    let color_id = available_graph_node_id(graph, "vertex_color")?;
    let mix_id = available_graph_node_id(graph, "vertex_color_multiply")?;
    let alpha_id = if material.alpha_mode == "opaque" {
        None
    } else {
        Some(available_graph_node_id(graph, "vertex_alpha_multiply")?)
    };
    let mut color = GraphNode::new("ShaderNodeVertexColor");
    color
        .properties
        .insert("layer_name".to_owned(), json!(layer));
    let mut mix = GraphNode::new("ShaderNodeMixRGB");
    mix.properties
        .insert("blend_type".to_owned(), json!("MULTIPLY"));
    mix.inputs.insert("Fac".to_owned(), json!(1.0));
    mix.inputs.insert("Color1".to_owned(), base_input);
    mix.inputs
        .insert("Color2".to_owned(), json!([1.0, 1.0, 1.0, 1.0]));

    graph
        .links
        .retain(|link| !(link.to_node == bsdf_id && link.to_socket == "Base Color"));
    graph.nodes.insert(color_id.clone(), color);
    graph.nodes.insert(mix_id.clone(), mix);
    if let Some(link) = base_link {
        graph.links.push(GraphLink {
            from_node: link.from_node,
            from_socket: link.from_socket,
            to_node: mix_id.clone(),
            to_socket: "Color1".to_owned(),
        });
    }
    graph.links.extend([
        GraphLink {
            from_node: color_id.clone(),
            from_socket: "Color".to_owned(),
            to_node: mix_id.clone(),
            to_socket: "Color2".to_owned(),
        },
        GraphLink {
            from_node: mix_id,
            from_socket: "Color".to_owned(),
            to_node: bsdf_id.clone(),
            to_socket: "Base Color".to_owned(),
        },
    ]);
    if let Some(alpha_id) = alpha_id {
        let bsdf = graph
            .nodes
            .get(&bsdf_id)
            .ok_or_else(|| import_error("glTF material Principled BSDF is missing"))?;
        let alpha_input = bsdf
            .inputs
            .get("Alpha")
            .cloned()
            .unwrap_or_else(|| json!(material.base_color[3]));
        let alpha_link = graph
            .links
            .iter()
            .find(|link| link.to_node == bsdf_id && link.to_socket == "Alpha")
            .cloned();
        let mut alpha = GraphNode::new("ShaderNodeMath");
        alpha
            .properties
            .insert("operation".to_owned(), json!("MULTIPLY"));
        alpha.inputs.insert("Value".to_owned(), alpha_input);
        alpha.inputs.insert("Value_001".to_owned(), json!(1.0));
        graph
            .links
            .retain(|link| !(link.to_node == bsdf_id && link.to_socket == "Alpha"));
        graph.nodes.insert(alpha_id.clone(), alpha);
        if let Some(link) = alpha_link {
            graph.links.push(GraphLink {
                from_node: link.from_node,
                from_socket: link.from_socket,
                to_node: alpha_id.clone(),
                to_socket: "Value".to_owned(),
            });
        }
        graph.links.extend([
            GraphLink {
                from_node: color_id,
                from_socket: "Alpha".to_owned(),
                to_node: alpha_id.clone(),
                to_socket: "Value_001".to_owned(),
            },
            GraphLink {
                from_node: alpha_id,
                from_socket: "Value".to_owned(),
                to_node: bsdf_id,
                to_socket: "Alpha".to_owned(),
            },
        ]);
    }
    Ok(())
}

fn imported_color_material(
    doc: &mut SceneDoc,
    material_id: &Id,
    layer: &str,
    variants: &mut BTreeMap<(Id, String), Id>,
    used_material_ids: &mut BTreeSet<Id>,
    used_graph_ids: &mut BTreeSet<Id>,
) -> Result<Id> {
    let key = (material_id.clone(), layer.to_owned());
    if let Some(id) = variants.get(&key) {
        return Ok(id.clone());
    }
    let mut material = doc
        .materials
        .get(material_id)
        .cloned()
        .ok_or_else(|| import_error("glTF material is missing"))?;
    let variant_id = unique_import_id(
        &format!("{material_id}_vertex_color_{}", variants.len()),
        "material",
        variants.len(),
        used_material_ids,
    )?;
    let graph_id = if let Some(source_graph_id) = material.node_tree.clone() {
        let mut graph = doc
            .node_groups
            .get(&source_graph_id)
            .cloned()
            .ok_or_else(|| import_error("glTF material shader graph is missing"))?;
        let graph_id = unique_import_id(
            &format!("{variant_id}_vertex_color"),
            "shader",
            used_graph_ids.len(),
            used_graph_ids,
        )?;
        material.node_tree = Some(graph_id.clone());
        bind_imported_vertex_color(&mut graph, layer, &material)?;
        doc.node_groups.insert(graph_id.clone(), graph);
        graph_id
    } else {
        let textures = material_graph::GraphTextures::from_fields(&material, None);
        doc.materials.insert(variant_id.clone(), material.clone());
        material_graph::move_textures_to_graph(doc, &variant_id, &textures)?;
        let graph_id = doc
            .materials
            .get(&variant_id)
            .and_then(|material| material.node_tree.clone())
            .ok_or_else(|| import_error("glTF color material graph was not created"))?;
        used_graph_ids.insert(graph_id.clone());
        let graph = doc
            .node_groups
            .get_mut(&graph_id)
            .ok_or_else(|| import_error("glTF color material graph is missing"))?;
        bind_imported_vertex_color(graph, layer, &material)?;
        variants.insert(key, variant_id.clone());
        return Ok(variant_id);
    };
    material.node_tree = Some(graph_id.clone());
    doc.materials.insert(variant_id.clone(), material);
    used_graph_ids.insert(graph_id);
    variants.insert(key, variant_id.clone());
    Ok(variant_id)
}

/// Groups glTF vertices that are copies of one Potter vertex. Returns each glTF vertex's welded
/// index and the first glTF copy of each welded vertex; without IDs every vertex stays separate.
fn weld_copies(
    vertex_count: usize,
    source_ids: Option<&[u32]>,
) -> Result<(Vec<usize>, Vec<usize>)> {
    let Some(ids) = source_ids else {
        return Ok(((0..vertex_count).collect(), (0..vertex_count).collect()));
    };
    if ids.len() != vertex_count {
        return Err(import_error("glTF vertex IDs do not match mesh vertices"));
    }
    let mut welded_by_id = BTreeMap::<u32, usize>::new();
    let mut welded_of = Vec::with_capacity(vertex_count);
    let mut first = Vec::new();
    for (index, id) in ids.iter().enumerate() {
        let welded = *welded_by_id.entry(*id).or_insert_with(|| {
            first.push(index);
            first.len() - 1
        });
        welded_of.push(welded);
    }
    Ok((welded_of, first))
}

/// Whether morph NORMAL deltas disagree with normals derived from the morphed geometry.
/// Runs after `shading::import_shading`, so `mesh` already carries the imported shading.
/// `normals` and each target's NORMAL deltas hold one value per face corner in `mesh.faces` order.
fn morph_normals_disagree(
    mesh: &Mesh,
    normals: &[DVec3],
    morph_targets: &[MorphNormalTarget],
) -> bool {
    let mut disagree = false;
    for (position_deltas, normal_deltas) in morph_targets {
        let morphed = mesh
            .vertices
            .iter()
            .zip(position_deltas)
            .map(|(vertex, delta)| vertex.co + DVec3::from_array(*delta))
            .collect::<Vec<_>>();
        let Ok(expected) = corner_normals(mesh, &morphed) else {
            disagree = true;
            continue;
        };
        // glTF stores zero normals as the unit up vector, so the comparison uses that fallback.
        for (corner, expected) in expected.iter().enumerate() {
            let expected = expected.try_normalize().unwrap_or(DVec3::Z);
            let implied =
                (normals[corner] + DVec3::from_array(normal_deltas[corner])).normalize_or_zero();
            if !close(implied, expected) {
                disagree = true;
            }
        }
    }
    disagree
}

/// Polygons a Potter writer stores in `extras.potter.polygons`, over global glTF vertex indices.
struct PotterFaces {
    corners: Vec<Vec<usize>>,
    smooth: Vec<bool>,
    /// Primitive holding each face's triangles.
    primitives: Vec<usize>,
    /// Endpoint vertex pairs of sharp edges.
    sharp: Vec<[usize; 2]>,
}

enum PotterPolygons {
    Absent,
    /// A version this importer does not read; the mesh imports as triangles and reports a loss.
    UnknownVersion,
    Faces(PotterFaces),
}

/// Bytes of the bufferView a polygon field names. Dangling or out-of-range references are errors.
fn potter_view<'a>(root: &Value, buffers: &'a [Vec<u8>], index: &Value) -> Result<&'a [u8]> {
    let invalid = || import_error("glTF Potter polygon bufferView is invalid");
    let view = index
        .as_u64()
        .and_then(|index| usize::try_from(index).ok())
        .and_then(|index| root["bufferViews"].get(index))
        .ok_or_else(invalid)?;
    let buffer = buffers
        .get(usize::try_from(view["buffer"].as_u64().unwrap_or(0)).map_err(|_| invalid())?)
        .ok_or_else(invalid)?;
    let offset =
        usize::try_from(view["byteOffset"].as_u64().unwrap_or(0)).map_err(|_| invalid())?;
    let length =
        usize::try_from(view["byteLength"].as_u64().ok_or_else(invalid)?).map_err(|_| invalid())?;
    buffer
        .get(offset..offset.checked_add(length).ok_or_else(invalid)?)
        .ok_or_else(invalid)
}

/// Decodes packed little-endian unsigned ints, each as wide as the writer's `packed_width(limit)`.
fn packed_values(bytes: &[u8], limit: usize) -> Result<Vec<usize>> {
    let width = packed_width(limit);
    if !bytes.len().is_multiple_of(width) {
        return Err(import_error("glTF Potter polygon field length is invalid"));
    }
    Ok(bytes
        .chunks_exact(width)
        .map(|chunk| {
            chunk
                .iter()
                .rev()
                .fold(0, |value, byte| (value << 8) | usize::from(*byte))
        })
        .collect())
}

/// Reads `extras.potter.polygons`. Structural faults (bad references, lengths, indices, or
/// duplicate corners) are errors, as the other Potter extras are. An unknown version is reported by
/// the caller.
fn potter_polygons(
    mesh_value: &Value,
    root: &Value,
    buffers: &[Vec<u8>],
    vertex_count: usize,
    primitive_count: usize,
) -> Result<PotterPolygons> {
    let value = &mesh_value["extras"]["potter"]["polygons"];
    if value.is_null() {
        return Ok(PotterPolygons::Absent);
    }
    let invalid = || import_error("glTF Potter polygon topology is invalid");
    if !value.is_object() {
        return Err(invalid());
    }
    if value["version"].as_u64() != Some(1) {
        return Ok(PotterPolygons::UnknownVersion);
    }
    let field = |name: &str, limit: usize| -> Result<Vec<usize>> {
        packed_values(potter_view(root, buffers, &value[name])?, limit)
    };
    let smooth = field("smooth", 1)?;
    let face_count = smooth.len();
    let sizes = field("sizes", vertex_count)?;
    let primitives = field("primitives", primitive_count.saturating_sub(1))?;
    let corners_flat = field("corners", vertex_count.saturating_sub(1))?;
    if smooth.iter().any(|flag| *flag > 1)
        || sizes.len() != face_count
        || primitives.len() != face_count
        || sizes.iter().any(|size| *size < 3)
        || corners_flat.len() != sizes.iter().sum::<usize>()
        || corners_flat.iter().any(|corner| *corner >= vertex_count)
        || primitives
            .iter()
            .any(|primitive| *primitive >= primitive_count)
    {
        return Err(invalid());
    }
    let mut corners = Vec::with_capacity(face_count);
    let mut rest = corners_flat.as_slice();
    for size in &sizes {
        let (face, tail) = rest.split_at_checked(*size).ok_or_else(invalid)?;
        let mut sorted = face.to_vec();
        sorted.sort_unstable();
        if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(invalid());
        }
        corners.push(face.to_vec());
        rest = tail;
    }
    let sharp_flat = if value["sharp_edges"].is_null() {
        Vec::new()
    } else {
        packed_values(
            potter_view(root, buffers, &value["sharp_edges"])?,
            vertex_count.saturating_sub(1),
        )?
    };
    if sharp_flat.len() % 2 != 0 {
        return Err(invalid());
    }
    let sharp = sharp_flat
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| match *pair {
            [first, second] if first != second && first < vertex_count && second < vertex_count => {
                Ok([first, second])
            }
            _ => Err(invalid()),
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(PotterPolygons::Faces(PotterFaces {
        corners,
        smooth: smooth.into_iter().map(|flag| flag == 1).collect(),
        primitives,
        sharp,
    }))
}

/// One polygon corner's UV values, one per stored UV layer in layer order.
type CornerUvs = Vec<[f64; 2]>;

/// The corner UVs of each face, in face corner order.
type FaceUvs = Vec<Vec<CornerUvs>>;
/// `COLOR_n` values for each restored face in face-corner order.
type FaceColorSets = BTreeMap<usize, Vec<Vec<Option<[f64; 4]>>>>;

/// Restores the stored faces onto the triangles, which the importer read per primitive in order.
/// Faces take their primitive's next `corners - 2` triangles in face order, and those triangles
/// must cover exactly the face's corners with every triangle used. Materials, UVs and color rows
/// are restored together in stored face order.
fn restore_polygons(
    faces: &PotterFaces,
    triangles: &[Vec<usize>],
    slots: &[u32],
    uvs: &[Vec<CornerUvs>],
    colors: &FaceColorSets,
    primitive_starts: &[usize],
) -> Option<(Vec<u32>, FaceUvs, FaceColorSets)> {
    // Cursors into each primitive's triangles; `primitive_starts` ends with the total count.
    let mut cursors = primitive_starts
        .iter()
        .take(primitive_starts.len().saturating_sub(1))
        .copied()
        .collect::<Vec<_>>();
    let mut materials = Vec::with_capacity(faces.corners.len());
    let mut corner_uvs = Vec::with_capacity(faces.corners.len());
    let mut face_colors = colors
        .keys()
        .map(|&set| (set, Vec::with_capacity(faces.corners.len())))
        .collect::<FaceColorSets>();
    for (corners, primitive) in faces.corners.iter().zip(&faces.primitives) {
        let start = *cursors.get(*primitive)?;
        let end = start.checked_add(corners.len().checked_sub(2)?)?;
        if end > *primitive_starts.get(*primitive + 1)? {
            return None;
        }
        *cursors.get_mut(*primitive)? = end;
        let owned = triangles.get(start..end)?;
        let owned_uvs = uvs.get(start..end)?;
        let mut covered = owned.iter().flatten().copied().collect::<Vec<_>>();
        covered.sort_unstable();
        covered.dedup();
        let mut sorted = corners.clone();
        sorted.sort_unstable();
        if covered != sorted {
            return None;
        }
        // Every triangle corner carries its glTF vertex's UV, so any triangle holding the corner gives it.
        let mut corner_values = owned
            .iter()
            .zip(owned_uvs)
            .flat_map(|(triangle, triangle_uvs)| {
                triangle.iter().copied().zip(triangle_uvs.iter().cloned())
            })
            .collect::<Vec<_>>();
        corner_values.sort_unstable_by_key(|(corner, _)| *corner);
        let face_uvs = corners
            .iter()
            .map(|corner| {
                let index = corner_values
                    .binary_search_by_key(corner, |(value, _)| *value)
                    .ok()?;
                corner_values.get(index).map(|(_, uv)| uv.clone())
            })
            .collect::<Option<Vec<_>>>()?;
        materials.push(*slots.get(start)?);
        corner_uvs.push(face_uvs);

        for (&set, rows) in colors {
            let owned_rows = rows.get(start..end)?;
            if owned_rows.len() != owned.len()
                || owned
                    .iter()
                    .zip(owned_rows)
                    .any(|(triangle, row)| triangle.len() != row.len())
            {
                return None;
            }
            let mut values = owned
                .iter()
                .zip(owned_rows)
                .flat_map(|(triangle, row)| triangle.iter().copied().zip(row.iter().copied()))
                .collect::<Vec<_>>();
            values.sort_unstable_by_key(|(corner, _)| *corner);
            if values
                .windows(2)
                .any(|pair| pair[0].0 == pair[1].0 && pair[0].1 != pair[1].1)
            {
                return None;
            }
            let face_values = corners
                .iter()
                .map(|corner| {
                    let index = values
                        .binary_search_by_key(corner, |(value, _)| *value)
                        .ok()?;
                    values.get(index).map(|(_, color)| *color)
                })
                .collect::<Option<Vec<_>>>()?;
            face_colors.get_mut(&set)?.push(face_values);
        }
    }
    if cursors
        .iter()
        .zip(primitive_starts.iter().skip(1))
        .any(|(cursor, end)| cursor != end)
    {
        return None;
    }
    // Each sharp pair must be a polygon edge; the shading import relies on that.
    if !faces.sharp.is_empty() {
        let mut edges = faces
            .corners
            .iter()
            .flat_map(|corners| {
                corners
                    .iter()
                    .zip(corners.iter().cycle().skip(1))
                    .map(|(first, second)| edge_key(*first, *second))
            })
            .collect::<Vec<_>>();
        edges.sort_unstable();
        edges.dedup();
        if faces
            .sharp
            .iter()
            .any(|[first, second]| edges.binary_search(&edge_key(*first, *second)).is_err())
        {
            return None;
        }
    }
    Some((materials, corner_uvs, face_colors))
}

fn append_triangles(
    indices: &[usize],
    mode: u64,
    vertex_count: usize,
    base: usize,
    faces: &mut Vec<Vec<usize>>,
    material_indices: &mut Vec<u32>,
) -> Result<()> {
    let mut triangles = Vec::<[usize; 3]>::new();
    match mode {
        4 => {
            if !indices.len().is_multiple_of(3) {
                return Err(import_error(
                    "glTF triangle index count is not divisible by three",
                ));
            }
            for triangle in indices.as_chunks::<3>().0 {
                triangles.push([triangle[0], triangle[1], triangle[2]]);
            }
        }
        5 => {
            for index in 2..indices.len() {
                triangles.push(if index % 2 == 0 {
                    [indices[index - 2], indices[index - 1], indices[index]]
                } else {
                    [indices[index - 1], indices[index - 2], indices[index]]
                });
            }
        }
        6 => {
            if let Some(first) = indices.first().copied() {
                for index in 2..indices.len() {
                    triangles.push([first, indices[index - 1], indices[index]]);
                }
            }
        }
        _ => return Err(import_error("only glTF triangle primitives are supported")),
    }
    for triangle in triangles {
        let mut face = Vec::with_capacity(3);
        for index in triangle {
            if index >= vertex_count {
                return Err(import_error("glTF primitive index is out of range"));
            }
            face.push(
                base.checked_add(index)
                    .ok_or_else(|| import_error("glTF primitive index overflow"))?,
            );
        }
        faces.push(face);
        material_indices.push(0);
    }
    Ok(())
}

fn import_cameras(root: &Value, doc: &mut SceneDoc, used: &mut BTreeSet<Id>) -> Result<Vec<Id>> {
    let cameras = root["cameras"].as_array().map_or(&[][..], Vec::as_slice);
    let mut ids = Vec::with_capacity(cameras.len());
    for (index, value) in cameras.iter().enumerate() {
        let suggested = value["name"].as_str().unwrap_or("camera");
        let id = unique_import_id(suggested, "camera", index, used)?;
        let mut camera = CameraData::default();
        match value["type"].as_str() {
            Some("perspective") => {
                camera.projection = CameraProjection::Perspective;
                let perspective = &value["perspective"];
                if let Some(yfov) = perspective["yfov"].as_f64()
                    && yfov.is_finite()
                    && yfov > 0.0
                    && yfov < std::f64::consts::PI
                {
                    camera.lens_mm = camera.sensor_width_mm / (2.0 * (yfov * 0.5).tan());
                }
                camera.clip_start = perspective["znear"].as_f64().unwrap_or(camera.clip_start);
                camera.clip_end = perspective["zfar"].as_f64().unwrap_or(camera.clip_end);
            }
            Some("orthographic") => {
                camera.projection = CameraProjection::Orthographic;
                let orthographic = &value["orthographic"];
                camera.ortho_scale = orthographic["ymag"].as_f64().unwrap_or(camera.ortho_scale);
                camera.clip_start = orthographic["znear"].as_f64().unwrap_or(camera.clip_start);
                camera.clip_end = orthographic["zfar"].as_f64().unwrap_or(camera.clip_end);
            }
            _ => return Err(import_error("glTF camera type is invalid")),
        }
        if let Some(camera_data) = value["extras"]["potter"]["camera_data"].as_object() {
            camera = serde_json::from_value(Value::Object(camera_data.clone()))
                .map_err(|_| import_error("glTF Potter camera metadata is invalid"))?;
        }
        doc.data_blocks.insert(
            id.clone(),
            DataBlock {
                data_type: "camera".to_owned(),
                descriptor: None,
                mesh: None,
                camera: Some(camera),
                light: None,
                ..DataBlock::default()
            },
        );
        ids.push(id);
    }
    Ok(ids)
}

fn import_lights(root: &Value, doc: &mut SceneDoc, used: &mut BTreeSet<Id>) -> Result<Vec<Id>> {
    let lights = root["extensions"]["KHR_lights_punctual"]["lights"]
        .as_array()
        .map_or(&[][..], Vec::as_slice);
    let mut ids = Vec::with_capacity(lights.len());
    for (index, value) in lights.iter().enumerate() {
        let suggested = value["name"].as_str().unwrap_or("light");
        let id = unique_import_id(suggested, "light", index, used)?;
        let mut light = LightData {
            light_type: match value["type"].as_str() {
                Some("directional") => LightType::Sun,
                Some("point") => LightType::Point,
                Some("spot") => LightType::Spot,
                _ => return Err(import_error("glTF punctual light type is invalid")),
            },
            ..LightData::default()
        };
        if let Some(color) = value["color"].as_array() {
            for (target, source) in light.color.iter_mut().zip(color) {
                *target = source
                    .as_f64()
                    .ok_or_else(|| import_error("glTF light color is invalid"))?;
            }
        }
        light.energy = value["intensity"].as_f64().unwrap_or(light.energy);
        light.radius = value["range"].as_f64().unwrap_or(light.radius);
        if light.light_type == LightType::Spot {
            let outer = value["spot"]["outerConeAngle"]
                .as_f64()
                .unwrap_or(light.spot_size * 0.5);
            let inner = value["spot"]["innerConeAngle"].as_f64().unwrap_or(0.0);
            light.spot_size = outer * 2.0;
            light.spot_blend = if outer > 0.0 {
                (1.0 - inner / outer).clamp(0.0, 1.0)
            } else {
                0.0
            };
        }
        doc.data_blocks.insert(
            id.clone(),
            DataBlock {
                data_type: "light".to_owned(),
                descriptor: None,
                mesh: None,
                camera: None,
                light: Some(light),
                ..DataBlock::default()
            },
        );
        ids.push(id);
    }
    Ok(ids)
}

fn node_parent_indices(nodes: &[Value]) -> Result<Vec<Option<usize>>> {
    let mut parents = vec![None; nodes.len()];
    for (parent_index, node) in nodes.iter().enumerate() {
        if let Some(children) = node["children"].as_array() {
            for child in children {
                let child = usize::try_from(
                    child
                        .as_u64()
                        .ok_or_else(|| import_error("glTF child index is invalid"))?,
                )
                .map_err(|_| import_error("glTF child index exceeds platform range"))?;
                let entry = parents
                    .get_mut(child)
                    .ok_or_else(|| import_error("glTF child index is out of range"))?;
                if entry.replace(parent_index).is_some() {
                    return Err(import_error("glTF node has multiple parents"));
                }
            }
        }
    }
    Ok(parents)
}

fn unique_import_id(raw: &str, prefix: &str, index: usize, used: &mut BTreeSet<Id>) -> Result<Id> {
    let mut base = String::with_capacity(raw.len().min(64));
    for byte in raw.bytes() {
        let value = char::from(byte).to_ascii_lowercase();
        if value.is_ascii_lowercase() || value.is_ascii_digit() || value == '_' || value == '-' {
            base.push(value);
        } else {
            base.push('_');
        }
    }
    if base.is_empty() || !base.as_bytes()[0].is_ascii_lowercase() {
        base.insert_str(0, &format!("{prefix}{index}_"));
    }
    base.truncate(64);
    if base.is_empty() {
        let _ = write!(base, "{prefix}{index}");
    }
    let first = Id::new(base.clone())?;
    if let Some(id) = super::claim_import_id(first, used) {
        return Ok(id);
    }
    let mut attempt = index;
    loop {
        let suffix = format!("_{attempt}");
        let max_base = 64_usize.saturating_sub(suffix.len());
        let value = format!(
            "{}{suffix}",
            base.chars().take(max_base).collect::<String>()
        );
        let candidate = Id::new(value)?;
        if let Some(id) = super::claim_import_id(candidate, used) {
            return Ok(id);
        }
        attempt = attempt.saturating_add(1);
    }
}

fn import_animations(
    root: &Value,
    buffers: &[Vec<u8>],
    node_ids: &[Id],
    doc: &mut SceneDoc,
    basis: DMat4,
) -> Result<Vec<crate::exchange::Loss>> {
    let animation_values = root["animations"].as_array().map_or(&[][..], Vec::as_slice);
    let mut losses = Vec::new();
    let scene = doc
        .scenes
        .get(&doc.active_scene)
        .ok_or_else(|| PotError::new(ErrorCode::SceneInvalid, "animation scene is missing"))?;
    let frames_per_second = f64::from(scene.fps) / scene.fps_base;
    let mut actions = BTreeMap::<Id, Action>::new();
    for animation in animation_values {
        let samplers = animation["samplers"]
            .as_array()
            .ok_or_else(|| import_error("glTF animation samplers are missing"))?;
        let channels = animation["channels"]
            .as_array()
            .ok_or_else(|| import_error("glTF animation channels are missing"))?;
        let start_frame = animation["extras"]["potter"]["start_frame"]
            .as_f64()
            .unwrap_or(1.0);
        let animation_name = animation["name"].as_str().unwrap_or("Animation");
        for channel in channels {
            let target = &channel["target"];
            let node_index = usize::try_from(
                target["node"]
                    .as_u64()
                    .ok_or_else(|| import_error("glTF animation target node is missing"))?,
            )
            .map_err(|_| import_error("glTF animation node exceeds platform range"))?;
            let node_id = node_ids
                .get(node_index)
                .cloned()
                .ok_or_else(|| import_error("glTF animation references a missing node"))?;
            let path = match target["path"].as_str() {
                Some("translation") => "transform.translation",
                Some("scale") => "transform.scale",
                Some("rotation") => "transform.rotation_quaternion",
                _ => {
                    losses.push(crate::exchange::Loss {
                        feature_id: "animation.channel".to_owned(),
                        data_id: Some(node_id.to_string()),
                        reason: "glTF animation target path is not supported".to_owned(),
                        suggestion: Some(
                            "use --allow-lossy or remove the unsupported channel".to_owned(),
                        ),
                    });
                    continue;
                }
            };
            let sampler_index = usize::try_from(
                channel["sampler"]
                    .as_u64()
                    .ok_or_else(|| import_error("glTF animation sampler index is missing"))?,
            )
            .map_err(|_| import_error("glTF animation sampler exceeds platform range"))?;
            let sampler = samplers
                .get(sampler_index)
                .ok_or_else(|| import_error("glTF animation references a missing sampler"))?;
            let input_index = usize::try_from(
                sampler["input"]
                    .as_u64()
                    .ok_or_else(|| import_error("glTF animation input is missing"))?,
            )
            .map_err(|_| import_error("glTF animation input exceeds platform range"))?;
            let output_index = usize::try_from(
                sampler["output"]
                    .as_u64()
                    .ok_or_else(|| import_error("glTF animation output is missing"))?,
            )
            .map_err(|_| import_error("glTF animation output exceeds platform range"))?;
            let (times, time_dimensions) = read_float_accessor(root, buffers, input_index)?;
            let (values, output_dimensions) = read_float_accessor(root, buffers, output_index)?;
            if time_dimensions != 1
                || output_dimensions
                    != if path.ends_with("rotation_quaternion") {
                        4
                    } else {
                        3
                    }
            {
                return Err(import_error(
                    "glTF animation accessor dimensions do not match target path",
                ));
            }
            let interpolation_name = sampler["interpolation"].as_str().unwrap_or("LINEAR");
            let (interpolation, cubic) = match interpolation_name {
                "STEP" => (Interpolation::Constant, false),
                "LINEAR" => (Interpolation::Linear, false),
                "CUBICSPLINE" => (Interpolation::Bezier, true),
                _ => return Err(import_error("glTF animation interpolation is invalid")),
            };
            if cubic {
                losses.push(crate::exchange::Loss {
                    feature_id: "animation.cubic_tangents".to_owned(),
                    data_id: Some(node_id.to_string()),
                    reason: "glTF cubic tangent handles are approximated by the potter Bezier curve model".to_owned(),
                    suggestion: Some("use --allow-lossy or export linear/step animation".to_owned()),
                });
            }
            let dimensions = output_dimensions;
            let output_key_count = times.len();
            let expected_count = output_key_count
                .checked_mul(dimensions)
                .and_then(|count| count.checked_mul(if cubic { 3 } else { 1 }))
                .ok_or_else(|| import_error("glTF animation output count overflow"))?;
            if values.len() != expected_count {
                return Err(import_error(
                    "glTF animation output count does not match input keys",
                ));
            }
            if times.iter().any(|time| !time.is_finite())
                || times.windows(2).any(|pair| pair[0] >= pair[1])
            {
                return Err(import_error(
                    "glTF animation times must be finite and strictly increasing",
                ));
            }
            let internal_path = path.to_owned();
            let mut component_values = vec![Vec::<f64>::with_capacity(times.len()); dimensions];
            let inverse_basis = basis.inverse();
            let basis_rotation = DQuat::from_mat4(&basis);
            for key_index in 0..times.len() {
                let source_offset = key_index * dimensions * if cubic { 3 } else { 1 };
                let value_offset = source_offset + if cubic { dimensions } else { 0 };
                let mut value = values[value_offset..value_offset + dimensions].to_vec();
                if path == "transform.translation" {
                    let position = inverse_basis
                        .transform_vector3(DVec3::new(value[0], value[1], value[2]))
                        .to_array();
                    value.copy_from_slice(&position);
                } else if path == "transform.rotation_quaternion" {
                    let rotation = DQuat::from_xyzw(value[0], value[1], value[2], value[3]);
                    let potter_rotation = basis_rotation.conjugate() * rotation * basis_rotation;
                    // F-Curve rotation components use Blender's (w, x, y, z) order.
                    value.copy_from_slice(&[
                        potter_rotation.w,
                        potter_rotation.x,
                        potter_rotation.y,
                        potter_rotation.z,
                    ]);
                }
                for (component, value) in value.into_iter().enumerate() {
                    component_values[component].push(value);
                }
            }
            let action = actions.entry(node_id.clone()).or_insert_with(|| Action {
                name: animation_name.to_owned(),
                fcurves: Vec::new(),
                ..Action::default()
            });
            if !action.fcurves.is_empty() && action.name != animation_name {
                action.name.push_str(", ");
                action.name.push_str(animation_name);
            }
            for (component, component_values) in component_values.into_iter().enumerate() {
                let keyframes = times
                    .iter()
                    .zip(component_values)
                    .map(|(time, value)| Keyframe {
                        frame: start_frame + *time * frames_per_second,
                        value,
                        interpolation,
                        ..Keyframe::default()
                    })
                    .collect::<Vec<_>>();
                action.fcurves.push(FCurve {
                    path: internal_path.clone(),
                    index: u32::try_from(component)
                        .map_err(|_| import_error("animation component index exceeds u32 range"))?,
                    keyframes,
                    extrapolation: Extrapolation::Constant,
                });
            }
        }
    }
    let mut used = doc.actions.keys().cloned().collect::<BTreeSet<_>>();
    for (node_id, action) in actions {
        let suggested = format!("action_{node_id}");
        let action_id = unique_import_id(&suggested, "action", used.len(), &mut used)?;
        let node = doc
            .nodes
            .get_mut(&node_id)
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "animated node is missing"))?;
        node.action = Some(action_id.clone());
        doc.actions.insert(action_id, action);
    }
    Ok(losses)
}

fn gltf_byte_offset(value: &Value) -> Result<usize> {
    match value.get("byteOffset") {
        None => Ok(0),
        Some(offset) => usize::try_from(
            offset
                .as_u64()
                .ok_or_else(|| import_error("glTF byteOffset is invalid"))?,
        )
        .map_err(|_| import_error("glTF byteOffset exceeds platform range")),
    }
}

fn buffer_view_data<'a>(
    root: &'a Value,
    buffers: &'a [Vec<u8>],
    view_index: usize,
) -> Result<(&'a Value, &'a [u8])> {
    let view = root["bufferViews"]
        .get(view_index)
        .ok_or_else(|| import_error("glTF accessor bufferView is missing"))?;
    let buffer_index = usize::try_from(
        view["buffer"]
            .as_u64()
            .ok_or_else(|| import_error("glTF accessor buffer index is missing"))?,
    )
    .map_err(|_| import_error("glTF accessor buffer index exceeds platform range"))?;
    let buffer = buffers
        .get(buffer_index)
        .ok_or_else(|| import_error("glTF accessor buffer is missing"))?;
    let start = gltf_byte_offset(view)?;
    let length = usize::try_from(
        view["byteLength"]
            .as_u64()
            .ok_or_else(|| import_error("glTF accessor bufferView byteLength is missing"))?,
    )
    .map_err(|_| import_error("glTF accessor bufferView length exceeds platform range"))?;
    let end = start
        .checked_add(length)
        .ok_or_else(|| import_error("glTF accessor bufferView range overflows"))?;
    let data = buffer
        .get(start..end)
        .ok_or_else(|| import_error("glTF accessor bufferView data is truncated"))?;
    Ok((view, data))
}

fn read_accessor_component(bytes: &[u8], component_type: u64, normalized: bool) -> Result<f64> {
    match component_type {
        5126 => Ok(f64::from(f32::from_le_bytes(bytes.try_into().map_err(
            |_| import_error("glTF accessor component is truncated"),
        )?))),
        5121 if normalized => Ok(f64::from(bytes[0]) / f64::from(u8::MAX)),
        5123 if normalized => Ok(f64::from(u16::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| import_error("glTF accessor component is truncated"))?,
        )) / f64::from(u16::MAX)),
        _ => Err(import_error("glTF accessor component type is unsupported")),
    }
}

fn read_sparse_index(bytes: &[u8], component_type: u64) -> Result<usize> {
    match component_type {
        5121 => Ok(usize::from(bytes[0])),
        5123 => {
            Ok(usize::from(u16::from_le_bytes(bytes.try_into().map_err(
                |_| import_error("glTF sparse index is truncated"),
            )?)))
        }
        5125 => usize::try_from(u32::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| import_error("glTF sparse index is truncated"))?,
        ))
        .map_err(|_| import_error("glTF sparse index exceeds platform range")),
        _ => Err(import_error(
            "glTF sparse index component type is unsupported",
        )),
    }
}

fn read_accessor_values(
    root: &Value,
    buffers: &[Vec<u8>],
    index: usize,
    allow_normalized_unsigned: bool,
) -> Result<(Vec<f64>, usize)> {
    let accessor = root["accessors"]
        .get(index)
        .ok_or_else(|| import_error("glTF accessor is missing"))?;
    let component_type = accessor["componentType"]
        .as_u64()
        .ok_or_else(|| import_error("glTF accessor component type is missing"))?;
    let normalized = match accessor.get("normalized") {
        None => false,
        Some(value) => value
            .as_bool()
            .ok_or_else(|| import_error("glTF accessor normalized flag is invalid"))?,
    };
    let component_size = match component_type {
        5126 if !normalized => 4,
        5121 | 5123 if allow_normalized_unsigned && normalized => {
            if component_type == 5121 {
                1
            } else {
                2
            }
        }
        _ => return Err(import_error("glTF accessor component type is unsupported")),
    };
    let dimensions: usize = match accessor["type"].as_str() {
        Some("SCALAR") => 1,
        Some("VEC2") => 2,
        Some("VEC3") => 3,
        Some("VEC4") => 4,
        Some("MAT4") => 16,
        _ => return Err(import_error("glTF accessor type is unsupported")),
    };
    let count = usize::try_from(
        accessor["count"]
            .as_u64()
            .ok_or_else(|| import_error("glTF accessor count is missing"))?,
    )
    .map_err(|_| import_error("glTF accessor count exceeds platform range"))?;
    let capacity = count
        .checked_mul(dimensions)
        .ok_or_else(|| import_error("glTF accessor value count overflows"))?;
    let element_size = dimensions
        .checked_mul(component_size)
        .ok_or_else(|| import_error("glTF accessor element size overflows"))?;
    let mut values = vec![0.0; capacity];

    let sparse = accessor
        .get("sparse")
        .map(|value| {
            value
                .as_object()
                .ok_or_else(|| import_error("glTF sparse accessor is invalid"))
        })
        .transpose()?;
    let buffer_view_index = accessor
        .get("bufferView")
        .map(|value| {
            usize::try_from(
                value
                    .as_u64()
                    .ok_or_else(|| import_error("glTF accessor bufferView is invalid"))?,
            )
            .map_err(|_| import_error("glTF accessor bufferView exceeds platform range"))
        })
        .transpose()?;
    if let Some(view_index) = buffer_view_index {
        let (view, data) = buffer_view_data(root, buffers, view_index)?;
        let byte_offset = gltf_byte_offset(accessor)?;
        if !byte_offset.is_multiple_of(component_size) {
            return Err(import_error("glTF accessor offset is misaligned"));
        }
        let stride = match view.get("byteStride") {
            None => element_size,
            Some(value) => usize::try_from(
                value
                    .as_u64()
                    .ok_or_else(|| import_error("glTF accessor byteStride is invalid"))?,
            )
            .map_err(|_| import_error("glTF accessor stride exceeds platform range"))?,
        };
        if stride < element_size || !stride.is_multiple_of(component_size) {
            return Err(import_error(
                "glTF accessor byteStride is too small or misaligned",
            ));
        }
        for element in 0..count {
            let start = byte_offset
                .checked_add(
                    element
                        .checked_mul(stride)
                        .ok_or_else(|| import_error("glTF accessor stride overflows"))?,
                )
                .ok_or_else(|| import_error("glTF accessor offset overflows"))?;
            for component in 0..dimensions {
                let offset = start
                    .checked_add(component * component_size)
                    .ok_or_else(|| import_error("glTF accessor offset overflows"))?;
                let end = offset
                    .checked_add(component_size)
                    .ok_or_else(|| import_error("glTF accessor offset overflows"))?;
                let raw = data
                    .get(offset..end)
                    .ok_or_else(|| import_error("glTF accessor data is truncated"))?;
                values[element * dimensions + component] =
                    read_accessor_component(raw, component_type, normalized)?;
            }
        }
    }

    if let Some(sparse) = sparse {
        let sparse_count = usize::try_from(
            sparse
                .get("count")
                .and_then(Value::as_u64)
                .ok_or_else(|| import_error("glTF sparse accessor count is missing"))?,
        )
        .map_err(|_| import_error("glTF sparse accessor count exceeds platform range"))?;
        if sparse_count == 0 || sparse_count > count {
            return Err(import_error(
                "glTF sparse accessor count is outside accessor count",
            ));
        }
        let indices_value = sparse
            .get("indices")
            .ok_or_else(|| import_error("glTF sparse accessor indices are missing"))?;
        let indices = indices_value
            .as_object()
            .ok_or_else(|| import_error("glTF sparse accessor indices are invalid"))?;
        let index_type = indices
            .get("componentType")
            .and_then(Value::as_u64)
            .ok_or_else(|| import_error("glTF sparse index component type is missing"))?;
        let index_size = match index_type {
            5121 => 1,
            5123 => 2,
            5125 => 4,
            _ => {
                return Err(import_error(
                    "glTF sparse index component type is unsupported",
                ));
            }
        };
        let index_view = usize::try_from(
            indices
                .get("bufferView")
                .and_then(Value::as_u64)
                .ok_or_else(|| import_error("glTF sparse indices bufferView is missing"))?,
        )
        .map_err(|_| import_error("glTF sparse indices bufferView exceeds platform range"))?;
        let (_, index_data) = buffer_view_data(root, buffers, index_view)?;
        let index_offset = gltf_byte_offset(indices_value)?;
        if !index_offset.is_multiple_of(index_size) {
            return Err(import_error("glTF sparse indices offset is misaligned"));
        }
        let values_info_value = sparse
            .get("values")
            .ok_or_else(|| import_error("glTF sparse accessor values are missing"))?;
        let values_info = values_info_value
            .as_object()
            .ok_or_else(|| import_error("glTF sparse accessor values are invalid"))?;
        let value_view = usize::try_from(
            values_info
                .get("bufferView")
                .and_then(Value::as_u64)
                .ok_or_else(|| import_error("glTF sparse values bufferView is missing"))?,
        )
        .map_err(|_| import_error("glTF sparse values bufferView exceeds platform range"))?;
        let (_, value_data) = buffer_view_data(root, buffers, value_view)?;
        let value_offset = gltf_byte_offset(values_info_value)?;
        if !value_offset.is_multiple_of(component_size) {
            return Err(import_error("glTF sparse values offset is misaligned"));
        }
        let mut previous_index = None;
        for sparse_element in 0..sparse_count {
            let index_start = index_offset
                .checked_add(
                    sparse_element
                        .checked_mul(index_size)
                        .ok_or_else(|| import_error("glTF sparse index offset overflows"))?,
                )
                .ok_or_else(|| import_error("glTF sparse index offset overflows"))?;
            let index_end = index_start
                .checked_add(index_size)
                .ok_or_else(|| import_error("glTF sparse index offset overflows"))?;
            let source_index = read_sparse_index(
                index_data
                    .get(index_start..index_end)
                    .ok_or_else(|| import_error("glTF sparse indices are truncated"))?,
                index_type,
            )?;
            if source_index >= count
                || previous_index.is_some_and(|previous| source_index <= previous)
            {
                return Err(import_error(
                    "glTF sparse indices must be unique, increasing, and in range",
                ));
            }
            previous_index = Some(source_index);
            let value_start = value_offset
                .checked_add(
                    sparse_element
                        .checked_mul(element_size)
                        .ok_or_else(|| import_error("glTF sparse value offset overflows"))?,
                )
                .ok_or_else(|| import_error("glTF sparse value offset overflows"))?;
            for component in 0..dimensions {
                let offset = value_start
                    .checked_add(component * component_size)
                    .ok_or_else(|| import_error("glTF sparse value offset overflows"))?;
                let end = offset
                    .checked_add(component_size)
                    .ok_or_else(|| import_error("glTF sparse value offset overflows"))?;
                let raw = value_data
                    .get(offset..end)
                    .ok_or_else(|| import_error("glTF sparse values are truncated"))?;
                values[source_index * dimensions + component] =
                    read_accessor_component(raw, component_type, normalized)?;
            }
        }
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(import_error("glTF accessor contains non-finite values"));
    }
    Ok((values, dimensions))
}

fn read_float_accessor(
    root: &Value,
    buffers: &[Vec<u8>],
    index: usize,
) -> Result<(Vec<f64>, usize)> {
    let accessor = root["accessors"]
        .get(index)
        .ok_or_else(|| import_error("glTF animation accessor is missing"))?;
    if accessor["componentType"].as_u64() != Some(5126) {
        return Err(import_error(
            "glTF animation accessors must use float components",
        ));
    }
    read_accessor_values(root, buffers, index, false)
}

/// Whether a glTF attribute name is a `TEXCOORD_n` set past the stored UV layer limit.
fn texcoord_past_limit(name: &str) -> bool {
    name.strip_prefix("TEXCOORD_")
        .and_then(|set| set.parse::<usize>().ok())
        .is_some_and(|set| set >= uv_layers::MAX_LAYERS)
}

fn gltf_import_losses(root: &Value) -> Vec<crate::exchange::Loss> {
    let mut losses = Vec::new();
    if root["nodes"].as_array().is_some_and(|nodes| {
        nodes.iter().any(|node| {
            let Some(node_weights) = node["weights"].as_array() else {
                return false;
            };
            let Some(mesh_index) = node["mesh"]
                .as_u64()
                .and_then(|index| usize::try_from(index).ok())
            else {
                return true;
            };
            root["meshes"]
                .get(mesh_index)
                .and_then(|mesh| mesh["weights"].as_array())
                != Some(node_weights)
        })
    }) {
        losses.push(crate::exchange::Loss {
            feature_id: "mesh.shape_key_instance_weights".to_owned(),
            data_id: None,
            reason: "glTF per-node morph weights cannot be represented by Potter shared shape-key values".to_owned(),
            suggestion: Some("use --allow-lossy or use identical mesh-level morph weights".to_owned()),
        });
    }
    if root["materials"].as_array().is_some_and(|materials| {
        materials.iter().any(|material| {
            material["occlusionTexture"]["strength"]
                .as_f64()
                .is_some_and(|strength| !crate::float::equal_f64(strength, 1.0))
        })
    }) {
        // Blender's importer ignores occlusion strength too; Potter's graph has no input for it.
        losses.push(crate::exchange::Loss {
            feature_id: "material.occlusion_strength".to_owned(),
            data_id: None,
            reason: "glTF occlusion strength is not represented by Potter; occlusion is applied at full strength".to_owned(),
            suggestion: Some("use --allow-lossy to import occlusion at full strength".to_owned()),
        });
    }
    // Each stored UV layer is one TEXCOORD set, so sets past the layer limit, and texture references
    // to them, cannot be kept.
    let texcoord_beyond = root["meshes"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|mesh| mesh["primitives"].as_array().into_iter().flatten())
        .filter_map(|primitive| primitive["attributes"].as_object())
        .flat_map(|attributes| attributes.keys())
        .any(|name| texcoord_past_limit(name));
    let texture_past_limit = root["materials"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|material| {
            let pbr = &material["pbrMetallicRoughness"];
            [
                &pbr["baseColorTexture"],
                &pbr["metallicRoughnessTexture"],
                &material["normalTexture"],
                &material["emissiveTexture"],
                &material["occlusionTexture"],
            ]
        })
        .any(|info| {
            usize::try_from(texture_info_set(info)).is_ok_and(|set| set >= uv_layers::MAX_LAYERS)
        });
    if texcoord_beyond || texture_past_limit {
        losses.push(crate::exchange::Loss {
            feature_id: feature_id::MESH_UV_LAYER.to_owned(),
            data_id: None,
            reason: format!(
                "glTF UV sets past the {} layer limit cannot be stored; they import as absent and texture references to them sample the render layer",
                uv_layers::MAX_LAYERS
            ),
            suggestion: Some(
                "use --allow-lossy or keep each mesh to the layer limit's TEXCOORD sets".to_owned(),
            ),
        });
    }
    if root["extensionsUsed"].as_array().is_some_and(|extensions| {
        extensions.iter().any(|extension| {
            extension.as_str().is_some_and(|name| {
                !matches!(
                    name,
                    "KHR_lights_punctual"
                        | "KHR_materials_emissive_strength"
                        | "KHR_materials_transmission"
                        | "KHR_materials_ior"
                        | "KHR_texture_transform"
                )
            })
        })
    }) {
        losses.push(crate::exchange::Loss {
            feature_id: "gltf.extension".to_owned(),
            data_id: None,
            reason: "one or more glTF extensions are not imported by this adapter".to_owned(),
            suggestion: Some("use --allow-lossy or remove unsupported extensions".to_owned()),
        });
    }
    losses
}

fn split_glb(bytes: &[u8]) -> Result<(&[u8], Option<Vec<u8>>)> {
    if bytes.len() < 20 || bytes.get(0..4) != Some(b"glTF") {
        return Err(import_error("GLB header is invalid"));
    }
    let version = u32::from_le_bytes(
        bytes[4..8]
            .try_into()
            .map_err(|_| import_error("GLB version is missing"))?,
    );
    let declared = u32::from_le_bytes(
        bytes[8..12]
            .try_into()
            .map_err(|_| import_error("GLB length is missing"))?,
    ) as usize;
    if version != 2 || declared != bytes.len() || !declared.is_multiple_of(4) {
        return Err(import_error("GLB version, length, or alignment is invalid"));
    }
    let json_len = u32::from_le_bytes(
        bytes[12..16]
            .try_into()
            .map_err(|_| import_error("GLB JSON chunk is missing"))?,
    ) as usize;
    if bytes.get(16..20) != Some(&0x4E4F_534A_u32.to_le_bytes()) {
        return Err(import_error("GLB JSON chunk type is invalid"));
    }
    let json_end = 20_usize
        .checked_add(json_len)
        .ok_or_else(|| import_error("GLB JSON chunk length overflow"))?;
    let json = bytes
        .get(20..json_end)
        .ok_or_else(|| import_error("GLB JSON chunk is truncated"))?;
    if !json_len.is_multiple_of(4) {
        return Err(import_error("GLB JSON chunk is not aligned"));
    }
    if json_end == bytes.len() {
        return Ok((json, None));
    }
    let binary_header_end = json_end
        .checked_add(8)
        .ok_or_else(|| import_error("GLB BIN chunk is truncated"))?;
    let header = bytes
        .get(json_end..binary_header_end)
        .ok_or_else(|| import_error("GLB BIN chunk is truncated"))?;
    let binary_len = u32::from_le_bytes(
        header[..4]
            .try_into()
            .map_err(|_| import_error("GLB BIN length is missing"))?,
    ) as usize;
    if header.get(4..8) != Some(&0x004E_4942_u32.to_le_bytes()) || !binary_len.is_multiple_of(4) {
        return Err(import_error("GLB BIN chunk header is invalid"));
    }
    let binary_end = binary_header_end
        .checked_add(binary_len)
        .ok_or_else(|| import_error("GLB BIN length overflow"))?;
    let binary = bytes
        .get(binary_header_end..binary_end)
        .ok_or_else(|| import_error("GLB BIN chunk is truncated"))?;
    if binary_end != bytes.len() {
        return Err(import_error("GLB has trailing chunks"));
    }
    Ok((json, Some(binary.to_vec())))
}

fn node_matrix(node: &Value) -> Result<DMat4> {
    if let Some(matrix) = node["matrix"].as_array() {
        if matrix.len() != 16 {
            return Err(import_error("glTF node matrix must have 16 values"));
        }
        let mut values = [0.0; 16];
        for (target, value) in values.iter_mut().zip(matrix) {
            *target = value
                .as_f64()
                .ok_or_else(|| import_error("glTF node matrix value is invalid"))?;
        }
        return Ok(DMat4::from_cols_array(&values));
    }
    let vec3 = |field: &str, default: [f64; 3]| -> Result<[f64; 3]> {
        if let Some(values) = node[field].as_array() {
            if values.len() != 3 {
                return Err(import_error("glTF node TRS vector must have three values"));
            }
            let mut result = default;
            for (target, value) in result.iter_mut().zip(values) {
                *target = value
                    .as_f64()
                    .ok_or_else(|| import_error("glTF node TRS value is invalid"))?;
            }
            Ok(result)
        } else {
            Ok(default)
        }
    };
    let translation = vec3("translation", [0.0; 3])?;
    let scale = vec3("scale", [1.0; 3])?;
    let rotation = if let Some(values) = node["rotation"].as_array() {
        if values.len() != 4 {
            return Err(import_error("glTF node quaternion must have four values"));
        }
        DQuat::from_xyzw(
            values[0]
                .as_f64()
                .ok_or_else(|| import_error("invalid quaternion"))?,
            values[1]
                .as_f64()
                .ok_or_else(|| import_error("invalid quaternion"))?,
            values[2]
                .as_f64()
                .ok_or_else(|| import_error("invalid quaternion"))?,
            values[3]
                .as_f64()
                .ok_or_else(|| import_error("invalid quaternion"))?,
        )
    } else {
        DQuat::IDENTITY
    };
    Ok(DMat4::from_scale_rotation_translation(
        DVec3::from_array(scale),
        rotation,
        DVec3::from_array(translation),
    ))
}

fn read_vec3(root: &Value, buffers: &[Vec<u8>], accessor_index: usize) -> Result<Vec<[f64; 3]>> {
    let accessor = root["accessors"]
        .get(accessor_index)
        .ok_or_else(|| import_error("glTF position accessor is missing"))?;
    if accessor["componentType"].as_u64() != Some(5126) || accessor["type"].as_str() != Some("VEC3")
    {
        return Err(import_error("glTF POSITION accessor must be float VEC3"));
    }
    let count = accessor["count"]
        .as_u64()
        .ok_or_else(|| import_error("glTF accessor count is missing"))? as usize;
    let view_index = accessor["bufferView"]
        .as_u64()
        .ok_or_else(|| import_error("glTF accessor bufferView is missing"))?
        as usize;
    let view = root["bufferViews"]
        .get(view_index)
        .ok_or_else(|| import_error("glTF bufferView is missing"))?;
    let buffer_index = view["buffer"].as_u64().unwrap_or(0) as usize;
    let buffer = buffers
        .get(buffer_index)
        .ok_or_else(|| import_error("glTF buffer is missing"))?;
    let view_offset = view["byteOffset"].as_u64().unwrap_or(0) as usize;
    let accessor_offset = accessor["byteOffset"].as_u64().unwrap_or(0) as usize;
    let stride = view["byteStride"]
        .as_u64()
        .map_or(12, |value| value as usize);
    let base = view_offset
        .checked_add(accessor_offset)
        .ok_or_else(|| import_error("glTF accessor offset overflow"))?;
    let mut values = Vec::with_capacity(count);
    for index in 0..count {
        let start = base
            .checked_add(
                index
                    .checked_mul(stride)
                    .ok_or_else(|| import_error("glTF accessor stride overflow"))?,
            )
            .ok_or_else(|| import_error("glTF accessor offset overflow"))?;
        let mut point = [0.0; 3];
        for (axis, target) in point.iter_mut().enumerate() {
            let offset = start + axis * 4;
            let chunk = buffer
                .get(offset..offset + 4)
                .ok_or_else(|| import_error("glTF POSITION data is truncated"))?;
            *target = f64::from(f32::from_le_bytes(
                chunk
                    .try_into()
                    .map_err(|_| import_error("glTF POSITION is truncated"))?,
            ));
        }
        values.push(point);
    }
    Ok(values)
}

fn read_vec2(root: &Value, buffers: &[Vec<u8>], accessor_index: usize) -> Result<Vec<[f64; 2]>> {
    let (values, dimensions) = read_accessor_values(root, buffers, accessor_index, true)?;
    if dimensions != 2 {
        return Err(import_error(
            "glTF TEXCOORD accessor must be VEC2 with float or normalized unsigned components",
        ));
    }
    Ok(values
        .as_chunks::<2>()
        .0
        .iter()
        .map(|value| [value[0], value[1]])
        .collect())
}

/// Reads a `COLOR_n` accessor as linear RGBA. glTF COLOR holds float or normalized unsigned byte or
/// short components, in VEC3 (alpha 1) or VEC4.
fn read_color(root: &Value, buffers: &[Vec<u8>], accessor_index: usize) -> Result<Vec<[f64; 4]>> {
    let accessor = root["accessors"]
        .get(accessor_index)
        .ok_or_else(|| import_error("glTF COLOR accessor is missing"))?;
    let components = match accessor["type"].as_str() {
        Some("VEC3") => 3,
        Some("VEC4") => 4,
        _ => return Err(import_error("glTF COLOR accessor must be VEC3 or VEC4")),
    };
    let normalized = match accessor.get("normalized") {
        None => false,
        Some(value) => value
            .as_bool()
            .ok_or_else(|| import_error("glTF accessor normalized flag is invalid"))?,
    };
    match accessor["componentType"].as_u64() {
        Some(5126) if !normalized => {}
        Some(5121 | 5123) if normalized => {}
        _ => {
            return Err(import_error(
                "glTF COLOR accessor must be float or normalized unsigned byte or short",
            ));
        }
    }
    let (values, dimensions) = read_accessor_values(root, buffers, accessor_index, true)?;
    if dimensions != components {
        return Err(import_error("glTF COLOR accessor must be VEC3 or VEC4"));
    }
    let mut colors = Vec::with_capacity(values.len() / components);
    for channels in values.chunks_exact(components) {
        let mut color = [0.0, 0.0, 0.0, 1.0];
        color[..components].copy_from_slice(channels);
        colors.push(color);
    }
    Ok(colors)
}

fn read_joint_indices(
    root: &Value,
    buffers: &[Vec<u8>],
    accessor_index: usize,
) -> Result<Vec<[u16; 4]>> {
    let accessor = root["accessors"]
        .get(accessor_index)
        .ok_or_else(|| import_error("glTF JOINTS_0 accessor is missing"))?;
    if accessor["type"].as_str() != Some("VEC4") {
        return Err(import_error("glTF JOINTS_0 accessor must be VEC4"));
    }
    let component_type = accessor["componentType"]
        .as_u64()
        .ok_or_else(|| import_error("glTF JOINTS_0 component type is missing"))?;
    let component_size: usize = match component_type {
        5121 => 1,
        5123 => 2,
        _ => {
            return Err(import_error(
                "glTF JOINTS_0 must use unsigned byte or short",
            ));
        }
    };
    let count = usize::try_from(
        accessor["count"]
            .as_u64()
            .ok_or_else(|| import_error("glTF JOINTS_0 count is missing"))?,
    )
    .map_err(|_| import_error("glTF JOINTS_0 count exceeds platform range"))?;
    let view_index = usize::try_from(
        accessor["bufferView"]
            .as_u64()
            .ok_or_else(|| import_error("glTF JOINTS_0 bufferView is missing"))?,
    )
    .map_err(|_| import_error("glTF JOINTS_0 bufferView exceeds platform range"))?;
    let view = root["bufferViews"]
        .get(view_index)
        .ok_or_else(|| import_error("glTF JOINTS_0 bufferView is missing"))?;
    let buffer_index = usize::try_from(view["buffer"].as_u64().unwrap_or(0))
        .map_err(|_| import_error("glTF JOINTS_0 buffer index exceeds platform range"))?;
    let buffer = buffers
        .get(buffer_index)
        .ok_or_else(|| import_error("glTF JOINTS_0 buffer is missing"))?;
    let base = usize::try_from(view["byteOffset"].as_u64().unwrap_or(0))
        .map_err(|_| import_error("glTF JOINTS_0 view offset exceeds platform range"))?
        .checked_add(
            usize::try_from(accessor["byteOffset"].as_u64().unwrap_or(0)).map_err(|_| {
                import_error("glTF JOINTS_0 accessor offset exceeds platform range")
            })?,
        )
        .ok_or_else(|| import_error("glTF JOINTS_0 offset overflows"))?;
    let element_size = component_size
        .checked_mul(4)
        .ok_or_else(|| import_error("glTF JOINTS_0 element size overflows"))?;
    let stride = usize::try_from(view["byteStride"].as_u64().unwrap_or(element_size as u64))
        .map_err(|_| import_error("glTF JOINTS_0 stride exceeds platform range"))?;
    if stride < element_size {
        return Err(import_error("glTF JOINTS_0 stride is too small"));
    }
    let mut result = Vec::with_capacity(count);
    for element in 0..count {
        let start = base
            .checked_add(
                element
                    .checked_mul(stride)
                    .ok_or_else(|| import_error("glTF JOINTS_0 stride overflows"))?,
            )
            .ok_or_else(|| import_error("glTF JOINTS_0 offset overflows"))?;
        let mut values = [0_u16; 4];
        for (component, value) in values.iter_mut().enumerate() {
            let offset = start
                .checked_add(component * component_size)
                .ok_or_else(|| import_error("glTF JOINTS_0 offset overflows"))?;
            let bytes = buffer
                .get(offset..offset + component_size)
                .ok_or_else(|| import_error("glTF JOINTS_0 data is truncated"))?;
            *value = if component_type == 5121 {
                u16::from(bytes[0])
            } else {
                u16::from_le_bytes(
                    bytes
                        .try_into()
                        .map_err(|_| import_error("glTF JOINTS_0 data is truncated"))?,
                )
            };
        }
        result.push(values);
    }
    Ok(result)
}

fn read_skin_attributes(
    primitive: &Value,
    root: &Value,
    buffers: &[Vec<u8>],
    vertex_count: usize,
) -> Result<Option<Vec<SkinVertexWeights>>> {
    let attributes = &primitive["attributes"];
    let mut joint_sets = Vec::<Vec<[u16; 4]>>::new();
    let mut weight_sets = Vec::<Vec<[f64; 4]>>::new();
    for set_index in 0_usize.. {
        let joint_name = format!("JOINTS_{set_index}");
        let weight_name = format!("WEIGHTS_{set_index}");
        let joint_accessor = attributes[&joint_name].as_u64();
        let weight_accessor = attributes[&weight_name].as_u64();
        match (joint_accessor, weight_accessor) {
            (None, None) => {
                let later_set_exists = attributes.as_object().is_some_and(|values| {
                    values.keys().any(|name| {
                        ["JOINTS_", "WEIGHTS_"].iter().any(|prefix| {
                            name.strip_prefix(prefix)
                                .and_then(|suffix| suffix.parse::<usize>().ok())
                                .is_some_and(|index| index > set_index)
                        })
                    })
                });
                if later_set_exists {
                    return Err(import_error(
                        "glTF skin attribute sets must be contiguous from index zero",
                    ));
                }
                break;
            }
            (Some(joints), Some(weights)) => {
                let joints = read_joint_indices(
                    root,
                    buffers,
                    usize::try_from(joints).map_err(|_| {
                        import_error("glTF JOINTS_n accessor exceeds platform range")
                    })?,
                )?;
                let (weights, dimensions) = read_float_accessor(
                    root,
                    buffers,
                    usize::try_from(weights).map_err(|_| {
                        import_error("glTF WEIGHTS_n accessor exceeds platform range")
                    })?,
                )?;
                if dimensions != 4
                    || joints.len() != vertex_count
                    || weights.len() != vertex_count.saturating_mul(4)
                {
                    return Err(import_error(
                        "glTF skin attribute counts do not match POSITION",
                    ));
                }
                if weights.iter().any(|weight| *weight < 0.0) {
                    return Err(import_error("glTF skin weight is negative"));
                }
                let weights = weights
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|weight| [weight[0], weight[1], weight[2], weight[3]])
                    .collect::<Vec<_>>();
                joint_sets.push(joints);
                weight_sets.push(weights);
            }
            _ => {
                return Err(import_error(
                    "glTF skinned primitive requires matching JOINTS_n and WEIGHTS_n attributes",
                ));
            }
        }
    }
    if joint_sets.is_empty() {
        return Ok(None);
    }
    Ok(Some(
        (0..vertex_count)
            .map(|vertex_index| {
                (
                    joint_sets.iter().map(|set| set[vertex_index]).collect(),
                    weight_sets.iter().map(|set| set[vertex_index]).collect(),
                )
            })
            .collect(),
    ))
}

fn read_indices(root: &Value, buffers: &[Vec<u8>], accessor_index: usize) -> Result<Vec<usize>> {
    let accessor = root["accessors"]
        .get(accessor_index)
        .ok_or_else(|| import_error("glTF index accessor is missing"))?;
    if accessor["type"].as_str() != Some("SCALAR") {
        return Err(import_error("glTF index accessor must be SCALAR"));
    }
    let count = accessor["count"]
        .as_u64()
        .ok_or_else(|| import_error("glTF index count is missing"))? as usize;
    let component_type = accessor["componentType"]
        .as_u64()
        .ok_or_else(|| import_error("glTF index type is missing"))?;
    let size = match component_type {
        5121 => 1,
        5123 => 2,
        5125 => 4,
        _ => return Err(import_error("unsupported glTF index component type")),
    };
    let view_index = accessor["bufferView"]
        .as_u64()
        .ok_or_else(|| import_error("glTF index bufferView is missing"))?
        as usize;
    let view = root["bufferViews"]
        .get(view_index)
        .ok_or_else(|| import_error("glTF index bufferView is missing"))?;
    let buffer_index = view["buffer"].as_u64().unwrap_or(0) as usize;
    let buffer = buffers
        .get(buffer_index)
        .ok_or_else(|| import_error("glTF index buffer is missing"))?;
    let start = (view["byteOffset"].as_u64().unwrap_or(0) as usize)
        .checked_add(accessor["byteOffset"].as_u64().unwrap_or(0) as usize)
        .ok_or_else(|| import_error("glTF index offset overflow"))?;
    let stride = view["byteStride"]
        .as_u64()
        .map_or(size, |value| value as usize);
    let mut output = Vec::with_capacity(count);
    for index in 0..count {
        let offset = start
            .checked_add(
                index
                    .checked_mul(stride)
                    .ok_or_else(|| import_error("glTF index stride overflow"))?,
            )
            .ok_or_else(|| import_error("glTF index offset overflow"))?;
        let bytes = buffer
            .get(offset..offset + size)
            .ok_or_else(|| import_error("glTF index data is truncated"))?;
        let value = match size {
            1 => usize::from(bytes[0]),
            2 => usize::from(u16::from_le_bytes(
                bytes
                    .try_into()
                    .map_err(|_| import_error("glTF index is truncated"))?,
            )),
            _ => usize::try_from(u32::from_le_bytes(
                bytes
                    .try_into()
                    .map_err(|_| import_error("glTF index is truncated"))?,
            ))
            .map_err(|_| import_error("glTF index exceeds platform range"))?,
        };
        output.push(value);
    }
    Ok(output)
}

fn decode_data_uri(uri: &str) -> Result<Vec<u8>> {
    let (_, data) = uri
        .split_once(',')
        .ok_or_else(|| import_error("glTF data URI is malformed"))?;
    decode_base64(data)
}

fn decode_base64(input: &str) -> Result<Vec<u8>> {
    let mut output = Vec::with_capacity(input.len() * 3 / 4);
    let mut accumulator = 0_u32;
    let mut bits = 0_u8;
    for byte in input.bytes() {
        if byte == b'=' {
            break;
        }
        let value = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'\n' | b'\r' | b' ' => continue,
            _ => return Err(import_error("glTF data URI is invalid base64")),
        };
        accumulator = (accumulator << 6) | u32::from(value);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            output.push(((accumulator >> bits) & 0xFF) as u8);
        }
    }
    Ok(output)
}

pub(crate) fn axis_to_gltf(point: [f64; 3]) -> [f64; 3] {
    [point[0], point[2], -point[1]]
}

pub(crate) fn axis_from_gltf(point: [f64; 3]) -> [f64; 3] {
    [point[0], -point[2], point[1]]
}

/// Maps a glTF point or delta to Potter axes, undoing the VRM baked geometry scale first.
fn potter_point(inverse_basis: DMat4, baked: DVec3, point: [f64; 3]) -> [f64; 3] {
    inverse_basis
        .transform_vector3(DVec3::from_array(point) / baked)
        .to_array()
}

/// Maps a glTF normal to Potter axes; the baked scale acts on normals as its inverse transpose.
fn potter_normal(inverse_basis: DMat4, baked: DVec3, normal: [f64; 3]) -> [f64; 3] {
    let mapped = inverse_basis.transform_vector3(DVec3::from_array(normal) * baked);
    if baked == DVec3::ONE {
        mapped.to_array()
    } else {
        mapped.normalize_or_zero().to_array()
    }
}
/// Maps a morph target's NORMAL delta to Potter axes. glTF stores the delta as a difference of two
/// output normals (`output_normals`), so the morphed normal is `base + delta`. The Potter delta is the
/// difference of the mapped morphed and base normals, which keeps a non-uniform baked scale exact.
/// Without the base normal (the check is then skipped) the delta is zero.
fn potter_normal_delta(
    inverse_basis: DMat4,
    baked: DVec3,
    base: Option<[f64; 3]>,
    delta: [f64; 3],
) -> [f64; 3] {
    let Some(base) = base else {
        return [0.0; 3];
    };
    let morphed = potter_normal(
        inverse_basis,
        baked,
        std::array::from_fn(|axis| base[axis] + delta[axis]),
    );
    let original = potter_normal(inverse_basis, baked, base);
    std::array::from_fn(|axis| morphed[axis] - original[axis])
}

/// The VRM baked scale recorded on a glTF node: non-uniform or mirrored mesh scale moved into geometry.
fn baked_node_scale(node: &Value) -> Result<Option<DVec3>> {
    let value = &node["extras"]["potter"]["vrm_baked_scale"];
    if value.is_null() {
        return Ok(None);
    }
    let scale = serde_json::from_value::<[f64; 3]>(value.clone())
        .map(DVec3::from_array)
        .map_err(|_| import_error("glTF VRM baked scale is invalid"))?;
    if scale
        .to_array()
        .iter()
        .any(|component| !component.is_finite() || *component == 0.0)
    {
        return Err(import_error(
            "glTF VRM baked scale must be finite and nonzero",
        ));
    }
    Ok(Some(scale))
}

/// Baked scale per glTF mesh index, taken from the first node that instances the mesh.
fn baked_mesh_scales(root: &Value) -> Result<BTreeMap<usize, DVec3>> {
    let mut scales = BTreeMap::new();
    for node in root["nodes"].as_array().map_or(&[][..], Vec::as_slice) {
        let Some(scale) = baked_node_scale(node)? else {
            continue;
        };
        if let Some(mesh) = node["mesh"]
            .as_u64()
            .and_then(|mesh| usize::try_from(mesh).ok())
        {
            scales.entry(mesh).or_insert(scale);
        }
    }
    Ok(scales)
}

#[cfg(test)]
mod tests {
    use glam::DVec3;
    use proptest::{prelude::*, test_runner::TestCaseError};
    use serde_json::{Value, json};

    use std::collections::{BTreeMap, BTreeSet};

    use crate::geom::{
        Mesh,
        normals::{sharp_edge_flags, smooth_face_flags},
        uv_layers::{self, UvLayer, UvStore},
    };
    use crate::{
        eval::{EvaluationContext, Snapshot},
        exchange::ImportedGraph,
        model::SceneDoc,
        ops::{apply_batch, apply_batch_with_project_context},
    };

    use super::{
        axis_from_gltf, axis_to_gltf, export_with_root, gltf_import_losses, import, to_glb,
        uv_layer_diagnostics,
    };

    type TestResult<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;
    /// Rewrites an exported glTF document and its binary buffer.
    type Edit = fn(&mut Value, &mut Vec<u8>) -> TestResult<()>;

    fn scene(operations: &Value) -> TestResult<SceneDoc> {
        let initial = SceneDoc::new(uuid::Uuid::new_v4().to_string());
        let applied = apply_batch(
            &initial,
            &json!({
                "schema_version": 1,
                "base_revision": 0,
                "operations": operations
            }),
        )?;
        Ok(applied.doc)
    }

    fn scene_with_asset_root(root: &std::path::Path, operations: &Value) -> TestResult<SceneDoc> {
        let initial = SceneDoc::new(uuid::Uuid::new_v4().to_string());
        let applied = apply_batch_with_project_context(
            &initial,
            &json!({
                "schema_version": 1,
                "base_revision": 0,
                "operations": operations
            }),
            root,
            None,
        )?;
        for (digest, bytes) in applied.asset_blobs {
            let hex = digest
                .strip_prefix("sha256:")
                .ok_or("image blob digest is malformed")?;
            let blob = root.join("assets").join("sha256").join(hex).join("blob");
            std::fs::create_dir_all(blob.parent().ok_or("image blob parent path is missing")?)?;
            std::fs::write(blob, bytes)?;
        }
        Ok(applied.doc)
    }

    fn round_trip(doc: &SceneDoc) -> TestResult<(ImportedGraph, Value)> {
        let directory = tempfile::tempdir()?;
        round_trip_in(doc, directory.path())
    }

    fn round_trip_in(doc: &SceneDoc, root: &std::path::Path) -> TestResult<(ImportedGraph, Value)> {
        let snapshot = Snapshot::evaluate(doc, &EvaluationContext::default())?;
        let path = root.join("scene.gltf");
        let (document, binary) = export_with_root(doc, &snapshot, root, Some("scene.bin"))?;
        let value = serde_json::from_slice(&document)?;
        std::fs::write(&path, document)?;
        std::fs::write(root.join("scene.bin"), binary)?;
        let imported = import(&path, uuid::Uuid::new_v4().to_string())?;
        Ok((imported, value))
    }

    fn mesh_positions(doc: &SceneDoc) -> TestResult<Vec<[f64; 3]>> {
        let snapshot = Snapshot::evaluate(doc, &EvaluationContext::default())?;
        Ok(crate::exchange::evaluated_meshes(doc, &snapshot)?
            .into_iter()
            .flat_map(|mesh| mesh.positions)
            .collect())
    }

    fn as_case<T, E: std::fmt::Display>(
        result: std::result::Result<T, E>,
    ) -> std::result::Result<T, TestCaseError> {
        result.map_err(|error| TestCaseError::fail(error.to_string()))
    }

    #[test]
    fn read_vec2_defaults_missing_buffer_view_to_zero() -> TestResult {
        let document = json!({
            "accessors": [{ "componentType": 5126, "count": 3, "type": "VEC2" }]
        });
        assert_eq!(super::read_vec2(&document, &[], 0)?, vec![[0.0, 0.0]; 3]);
        Ok(())
    }

    #[test]
    fn malformed_sparse_accessor_fields_return_import_errors() -> TestResult {
        let sparse_cases = [
            json!({}),
            json!({"count": 1}),
            json!({"count": 1, "indices": {}}),
            json!({"count": 1, "indices": {"componentType": 5121}}),
            json!({"count": 1, "indices": {"componentType": 5121, "bufferView": 0}}),
            json!({"count": 1, "indices": {"componentType": 5121, "bufferView": 0}, "values": {}}),
        ];
        for sparse in sparse_cases {
            let document = json!({
                "accessors": [{
                    "componentType": 5126,
                    "count": 1,
                    "type": "VEC2",
                    "sparse": sparse
                }],
                "bufferViews": [{"buffer": 0, "byteLength": 1}]
            });
            let error = super::read_vec2(&document, &[vec![0_u8]], 0)
                .err()
                .ok_or("malformed sparse accessor was accepted")?;
            assert_eq!(error.code, crate::error::ErrorCode::ImportFailed);
        }
        Ok(())
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 16, .. ProptestConfig::default() })]

        #[test]
        fn axis_conversion_round_trip_is_identity(point in any::<[f64; 3]>()) {
            prop_assume!(point.iter().all(|value| value.is_finite()));
            let converted = axis_from_gltf(axis_to_gltf(point));
            for axis in 0..3 { prop_assert_eq!(converted[axis], point[axis]); }
        }

        #[test]
        fn small_scene_geometry_round_trip_preserves_world_positions(
            size in prop::array::uniform3(0.1_f64..4.0),
            translation in prop::array::uniform3(-5.0_f64..5.0),
        ) {
            let source = as_case(scene(&json!([{
                "op": "node.create",
                "id": "box",
                "kind": "box",
                "params": { "size": size[0] },
                "transform": {
                    "translation": translation,
                    "scale": [1.0, size[1] / size[0], size[2] / size[0]]
                }
            }])))?;
            let (imported, _) = as_case(round_trip(&source))?;
            let expected = as_case(mesh_positions(&source))?;
            let actual = as_case(mesh_positions(&imported.doc))?;
            let order = |left: &[f64; 3], right: &[f64; 3]| {
                left[0].total_cmp(&right[0])
                    .then(left[1].total_cmp(&right[1]))
                    .then(left[2].total_cmp(&right[2]))
            };
            // Normal splits duplicate a vertex per corner; compare the distinct world positions.
            let distinct = |mut points: Vec<[f64; 3]>| {
                points.sort_by(order);
                points.dedup_by(|left, right| order(left, right).is_eq());
                points
            };
            let expected = distinct(expected);
            let actual = distinct(actual);
            prop_assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(&expected) {
                for axis in 0..3 {
                    prop_assert!(
                        (actual[axis] - expected[axis]).abs() <= 1.0e-5,
                        "position mismatch on axis {axis}: {} vs {}",
                        actual[axis],
                        expected[axis]
                    );
                }
            }
        }

        #[test]
        fn ascii_fbx_and_usda_round_trip_random_scenes(
            size in prop::array::uniform3(0.1_f64..4.0),
            translation in prop::array::uniform3(-5.0_f64..5.0),
        ) {
            let source = as_case(scene(&json!([{
                "op": "node.create",
                "id": "box",
                "kind": "box",
                "params": { "size": size[0] },
                "transform": {
                    "translation": translation,
                    "scale": [1.0, size[1] / size[0], size[2] / size[0]]
                }
            }])))?;
            let snapshot = as_case(Snapshot::evaluate(&source, &EvaluationContext::default()))?;
            let directory = tempfile::tempdir().map_err(|error| TestCaseError::fail(error.to_string()))?;
            let expected = as_case(mesh_positions(&source))?;
            let order = |left: &[f64; 3], right: &[f64; 3]| {
                left[0].total_cmp(&right[0])
                    .then(left[1].total_cmp(&right[1]))
                    .then(left[2].total_cmp(&right[2]))
            };

            let fbx_path = directory.path().join("scene.fbx");
            let fbx_bytes = as_case(crate::exchange::fbx::export(&source, &snapshot))?;
            std::fs::write(&fbx_path, fbx_bytes).map_err(|error| TestCaseError::fail(error.to_string()))?;
            let fbx = as_case(crate::exchange::fbx::import(&fbx_path, uuid::Uuid::new_v4().to_string()))?;
            let mut fbx_positions = as_case(mesh_positions(&fbx.doc))?;
            fbx_positions.sort_by(order);
            let mut expected_fbx = expected.clone();
            expected_fbx.sort_by(order);
            prop_assert_eq!(fbx_positions.len(), expected_fbx.len());
            for (actual, expected) in fbx_positions.iter().zip(&expected_fbx) {
                for axis in 0..3 {
                    prop_assert!((actual[axis] - expected[axis]).abs() <= 1.0e-5);
                }
            }

            let usda_path = directory.path().join("scene.usda");
            let usda_bytes = as_case(crate::exchange::usd::export_usda(&source, &snapshot))?;
            std::fs::write(&usda_path, usda_bytes).map_err(|error| TestCaseError::fail(error.to_string()))?;
            let usda = as_case(crate::exchange::usd::import_usda(&usda_path, uuid::Uuid::new_v4().to_string()))?;
            let mut usda_positions = as_case(mesh_positions(&usda.doc))?;
            usda_positions.sort_by(order);
            let mut expected_usda = expected;
            expected_usda.sort_by(order);
            prop_assert_eq!(usda_positions.len(), expected_usda.len());
            for (actual, expected) in usda_positions.iter().zip(&expected_usda) {
                for axis in 0..3 {
                    prop_assert!((actual[axis] - expected[axis]).abs() <= 1.0e-5);
                }
            }
        }
    }

    #[test]
    fn coordinate_export_and_animation_rotation_match_authoring_axes() -> TestResult {
        let directory = tempfile::tempdir()?;
        let quarter_turn = std::f64::consts::FRAC_1_SQRT_2;
        let source = scene(&json!([
            {
                "op": "node.create",
                "id": "box",
                "kind": "box",
                "params": { "size": 2.0 },
                "transform": {
                    "translation": [3.0, 4.0, 5.0],
                    "rotation": [0.0, 0.0, quarter_turn, quarter_turn]
                }
            },
            {
                "op": "node.create",
                "id": "animated",
                "kind": "empty",
                "transform": {
                    "translation": [1.0, 2.0, 3.0],
                    "rotation": [0.0, 0.0, quarter_turn, quarter_turn]
                }
            },
            { "op": "action.create", "id": "turn", "name": "Turn" },
            { "op": "node.update", "target": { "id": "animated" }, "set": { "action": "turn" } },
            // Animation components use Blender's WXYZ order; this is the same quarter turn about Z.
            { "op": "keyframe.insert", "target": { "id": "animated" }, "path": "transform.rotation_quaternion", "index": 0, "frame": 1.0, "value": quarter_turn },
            { "op": "keyframe.insert", "target": { "id": "animated" }, "path": "transform.rotation_quaternion", "index": 0, "frame": 3.0, "value": quarter_turn },
            { "op": "keyframe.insert", "target": { "id": "animated" }, "path": "transform.rotation_quaternion", "index": 1, "frame": 1.0, "value": 0.0 },
            { "op": "keyframe.insert", "target": { "id": "animated" }, "path": "transform.rotation_quaternion", "index": 1, "frame": 3.0, "value": 0.0 },
            { "op": "keyframe.insert", "target": { "id": "animated" }, "path": "transform.rotation_quaternion", "index": 2, "frame": 1.0, "value": 0.0 },
            { "op": "keyframe.insert", "target": { "id": "animated" }, "path": "transform.rotation_quaternion", "index": 2, "frame": 3.0, "value": 0.0 },
            { "op": "keyframe.insert", "target": { "id": "animated" }, "path": "transform.rotation_quaternion", "index": 3, "frame": 1.0, "value": quarter_turn },
            { "op": "keyframe.insert", "target": { "id": "animated" }, "path": "transform.rotation_quaternion", "index": 3, "frame": 3.0, "value": quarter_turn }
        ]))?;
        let (imported, document) = round_trip_in(&source, directory.path())?;
        let binary = vec![std::fs::read(directory.path().join("scene.bin"))?];

        let box_node = document["nodes"]
            .as_array()
            .and_then(|nodes| {
                nodes
                    .iter()
                    .find(|node| node["extras"]["potter"]["id"] == "box")
            })
            .ok_or("transformed box node is missing")?;
        let box_matrix = box_node["matrix"]
            .as_array()
            .ok_or("transformed box matrix is missing")?
            .iter()
            .map(|value| {
                value
                    .as_f64()
                    .ok_or_else(|| "transformed box matrix is invalid".into())
            })
            .collect::<TestResult<Vec<_>>>()?;
        let box_matrix = glam::DMat4::from_cols_array(
            &box_matrix
                .as_slice()
                .try_into()
                .map_err(|_| "transformed box matrix must contain 16 values")?,
        );
        let mesh_index = usize::try_from(
            box_node["mesh"]
                .as_u64()
                .ok_or("transformed box mesh index is missing")?,
        )?;
        let position_accessor = usize::try_from(
            document["meshes"][mesh_index]["primitives"][0]["attributes"]["POSITION"]
                .as_u64()
                .ok_or("transformed box positions are missing")?,
        )?;
        let exported_positions = super::read_vec3(&document, &binary, position_accessor)?
            .into_iter()
            .map(|position| box_matrix.transform_point3(glam::DVec3::from_array(position)))
            .collect::<Vec<_>>();
        let expected_positions = mesh_positions(&source)?
            .into_iter()
            .map(|position| glam::DVec3::from_array(axis_to_gltf(position)))
            .collect::<Vec<_>>();
        let close = |left: glam::DVec3, right: glam::DVec3| left.distance(right) <= 1.0e-5;
        assert!(
            exported_positions.iter().all(|position| {
                expected_positions
                    .iter()
                    .any(|expected| close(*position, *expected))
            }) && expected_positions.iter().all(|position| {
                exported_positions
                    .iter()
                    .any(|actual| close(*position, *actual))
            }),
            "exported transformed geometry must use the authoring-to-glTF frame"
        );
        let imported_positions = mesh_positions(&imported.doc)?;
        assert!(
            mesh_positions(&source)?.iter().all(|position| {
                imported_positions.iter().any(|imported| {
                    glam::DVec3::from_array(*position).distance(glam::DVec3::from_array(*imported))
                        <= 1.0e-5
                })
            }),
            "import must restore transformed source geometry"
        );

        let animated_node_index = document["nodes"]
            .as_array()
            .and_then(|nodes| {
                nodes
                    .iter()
                    .position(|node| node["extras"]["potter"]["id"] == "animated")
            })
            .ok_or("animated node is missing")?;
        let animated_node = &document["nodes"][animated_node_index];
        let translation = animated_node["translation"]
            .as_array()
            .ok_or("animated node translation is missing")?
            .iter()
            .map(|value| {
                value
                    .as_f64()
                    .ok_or_else(|| "animated translation is invalid".into())
            })
            .collect::<TestResult<Vec<_>>>()?;
        let expected_translation = axis_to_gltf([1.0, 2.0, 3.0]);
        for (actual, expected) in translation.iter().zip(expected_translation) {
            assert!((actual - expected).abs() <= 1.0e-6);
        }
        let exported_rotation = glam::DQuat::from_xyzw(
            animated_node["rotation"][0]
                .as_f64()
                .ok_or("animated node rotation is missing")?,
            animated_node["rotation"][1]
                .as_f64()
                .ok_or("animated node rotation is missing")?,
            animated_node["rotation"][2]
                .as_f64()
                .ok_or("animated node rotation is missing")?,
            animated_node["rotation"][3]
                .as_f64()
                .ok_or("animated node rotation is missing")?,
        );
        let gltf_forward = exported_rotation * glam::DVec3::X;
        assert!(gltf_forward.distance(-glam::DVec3::Z) <= 1.0e-6);

        let animation = document["animations"]
            .as_array()
            .and_then(|animations| animations.first())
            .ok_or("rotation animation is missing")?;
        let channel = animation["channels"]
            .as_array()
            .and_then(|channels| {
                channels.iter().find(|channel| {
                    channel["target"]["node"].as_u64() == Some(animated_node_index as u64)
                        && channel["target"]["path"] == "rotation"
                })
            })
            .ok_or("rotation animation channel is missing")?;
        let sampler_index = usize::try_from(
            channel["sampler"]
                .as_u64()
                .ok_or("rotation animation sampler is missing")?,
        )?;
        let output_accessor = usize::try_from(
            animation["samplers"][sampler_index]["output"]
                .as_u64()
                .ok_or("rotation animation output is missing")?,
        )?;
        let (rotation_values, dimensions) =
            super::read_float_accessor(&document, &binary, output_accessor)?;
        assert_eq!(dimensions, 4);
        let animated_forward = glam::DQuat::from_xyzw(
            rotation_values[4],
            rotation_values[5],
            rotation_values[6],
            rotation_values[7],
        ) * glam::DVec3::X;
        assert!(animated_forward.distance(-glam::DVec3::Z) <= 1.0e-6);

        let imported_node = imported
            .doc
            .nodes
            .get(&"animated".parse()?)
            .ok_or("animated node was not imported")?;
        let imported_action = imported_node
            .action
            .as_ref()
            .and_then(|action_id| imported.doc.actions.get(action_id))
            .ok_or("rotation animation was not imported")?;
        for (component, expected) in [quarter_turn, 0.0, 0.0, quarter_turn]
            .into_iter()
            .enumerate()
        {
            let curve = imported_action
                .fcurves
                .iter()
                .find(|curve| {
                    curve.path == "transform.rotation_quaternion" && curve.index == component as u32
                })
                .ok_or("rotation component was not imported")?;
            assert!(curve.keyframes.iter().any(|keyframe| {
                (keyframe.frame - 3.0).abs() <= 1.0e-6
                    && (keyframe.value - expected).abs() <= 1.0e-6
            }));
        }
        Ok(())
    }

    #[test]
    fn export_import_round_trips_skin_morph_texture_animations_cameras_lights_and_instances()
    -> TestResult {
        let directory = tempfile::tempdir()?;
        let source = scene_with_asset_root(
            directory.path(),
            &json!([
                {"op":"image.create","id":"paint","width":1,"height":1,"colorspace":"srgb","fill_color":[0.25,0.5,0.75,1.0]},
                {"op":"material.create","id":"surface","base_color_texture":{"image":"paint","interpolation":"closest"}},
                {"op":"node.create","id":"skin_mesh","kind":"box","params":{"size":2.0},"material":"surface"},
                {"op":"uv.unwrap","target":{"id":"skin_mesh"},"method":"smart"},
                {"op":"uv.pin","target":{"id":"skin_mesh"}},
                {"op":"shape_key.create","target":{"id":"skin_mesh"},"id":"lift","name":"Lift","positions":{"0":[0.0,0.0,1.0]}},
                {"op":"node.create","id":"arm","kind":"armature"},
                {"op":"bone.create","target":{"id":"arm"},"id":"root","name":"Root","head":[0.0,0.0,0.0],"tail":[0.0,1.0,0.0]},
                {"op":"vertex_group.create","target":{"id":"skin_mesh"},"id":"root_group","name":"Root"},
                {"op":"vertex_group.assign","target":{"id":"skin_mesh"},"group_id":"root_group","weights":[
                    {"vertex_id":0,"weight":1.0},
                    {"vertex_id":1,"weight":0.5},
                    {"vertex_id":2,"weight":1.0},
                    {"vertex_id":3,"weight":0.25},
                    {"vertex_id":4,"weight":1.0},
                    {"vertex_id":5,"weight":0.5},
                    {"vertex_id":6,"weight":1.0},
                    {"vertex_id":7,"weight":0.25}
                ]},
                {"op":"modifier.create","target":{"id":"skin_mesh"},"id":"skin","type":"armature","params":{"object":"arm","use_vertex_groups":true}},
                {"op":"node.create","id":"group_mesh","kind":"box","params":{"size":1.0}},
                {"op":"vertex_group.create","target":{"id":"group_mesh"},"id":"extra_group","name":"Extra"},
                {"op":"vertex_group.assign","target":{"id":"group_mesh"},"group_id":"extra_group","weights":[{"vertex_id":0,"weight":0.75}]},
                {"op":"camera.create","id":"camera","name":"Camera","projection":"orthographic","ortho_scale":4.0,"lens_mm":35.0},
                {"op":"light.create","id":"light","name":"Light","light_type":"spot","energy":25.0,"spot_size":0.75},
                {"op":"collection.create","id":"source_collection"},
                {"op":"node.create","id":"collection_box","kind":"box","collection":"source_collection","params":{"size":1.0},"transform":{"scale":[1.0,2.0,3.0]}},
                {"op":"collection.instance_create","id":"collection_instance","collection":"source_collection","transform":{"translation":[4.0,0.0,0.0]}},
                {"op":"node.create","id":"step_node","kind":"empty"},
                {"op":"action.create","id":"step_action","name":"Step"},
                {"op":"node.update","target":{"id":"step_node"},"set":{"action":"step_action"}},
                {"op":"keyframe.insert","target":{"id":"step_node"},"path":"transform.translation","index":0,"frame":1.0,"value":0.0,"interpolation":"constant"},
                {"op":"keyframe.insert","target":{"id":"step_node"},"path":"transform.translation","index":0,"frame":3.0,"value":2.0,"interpolation":"constant"},
                {"op":"node.create","id":"linear_node","kind":"empty"},
                {"op":"action.create","id":"linear_action","name":"Linear"},
                {"op":"node.update","target":{"id":"linear_node"},"set":{"action":"linear_action"}},
                {"op":"keyframe.insert","target":{"id":"linear_node"},"path":"transform.translation","index":0,"frame":1.0,"value":0.0,"interpolation":"linear"},
                {"op":"keyframe.insert","target":{"id":"linear_node"},"path":"transform.translation","index":0,"frame":3.0,"value":2.0,"interpolation":"linear"},
                {"op":"node.create","id":"cubic_node","kind":"empty"},
                {"op":"action.create","id":"cubic_action","name":"Cubic"},
                {"op":"node.update","target":{"id":"cubic_node"},"set":{"action":"cubic_action"}},
                {"op":"keyframe.insert","target":{"id":"cubic_node"},"path":"transform.translation","index":0,"frame":1.0,"value":0.0,"interpolation":"bezier"},
                {"op":"keyframe.insert","target":{"id":"cubic_node"},"path":"transform.translation","index":0,"frame":3.0,"value":2.0,"interpolation":"bezier"}
            ]),
        )?;

        let (imported, document) = round_trip_in(&source, directory.path())?;
        let skin_mesh_value = document["meshes"]
            .as_array()
            .and_then(|meshes| {
                meshes
                    .iter()
                    .find(|mesh| mesh["primitives"][0]["attributes"]["JOINTS_0"].is_number())
            })
            .ok_or("skinned mesh was not exported")?;
        let primitive = &skin_mesh_value["primitives"][0];
        let position_accessor = usize::try_from(
            primitive["attributes"]["POSITION"]
                .as_u64()
                .ok_or("mesh POSITION accessor is missing")?,
        )?;
        let uv_accessor = usize::try_from(
            primitive["attributes"]["TEXCOORD_0"]
                .as_u64()
                .ok_or("default UV0 accessor is missing")?,
        )?;
        let position_count = usize::try_from(
            document["accessors"][position_accessor]["count"]
                .as_u64()
                .ok_or("mesh POSITION count is missing")?,
        )?;
        let uv_count = usize::try_from(
            document["accessors"][uv_accessor]["count"]
                .as_u64()
                .ok_or("mesh UV0 count is missing")?,
        )?;
        assert!(
            position_count > 8,
            "box UV seams must split export vertices"
        );
        assert_eq!(uv_count, position_count);
        let buffers = vec![std::fs::read(directory.path().join("scene.bin"))?];
        let morph_targets = primitive["targets"]
            .as_array()
            .ok_or("morph targets are missing")?;
        for target in morph_targets {
            let accessor_index = usize::try_from(
                target["POSITION"]
                    .as_u64()
                    .ok_or("morph POSITION accessor is missing")?,
            )?;
            let deltas = super::read_vec3(&document, &buffers, accessor_index)?;
            let mut minimum = [f64::INFINITY; 3];
            let mut maximum = [f64::NEG_INFINITY; 3];
            for delta in &deltas {
                for axis in 0..3 {
                    minimum[axis] = minimum[axis].min(delta[axis]);
                    maximum[axis] = maximum[axis].max(delta[axis]);
                }
            }
            let accessor = &document["accessors"][accessor_index];
            for axis in 0..3 {
                let actual_minimum = accessor["min"][axis]
                    .as_f64()
                    .ok_or("morph POSITION minimum is missing")?;
                let actual_maximum = accessor["max"][axis]
                    .as_f64()
                    .ok_or("morph POSITION maximum is missing")?;
                assert_eq!(actual_minimum, minimum[axis]);
                assert_eq!(actual_maximum, maximum[axis]);
                assert!(actual_minimum <= 0.0 && actual_maximum >= 0.0);
            }
        }
        let exported_uvs = super::read_vec2(&document, &buffers, uv_accessor)?;
        let source_uvs = source
            .nodes
            .get(&"skin_mesh".parse()?)
            .and_then(|node| node.data.as_ref())
            .and_then(|data_id| source.data_blocks.get(data_id))
            .and_then(|data| data.mesh.as_ref())
            .and_then(|mesh| uv_layers::read_layers(mesh).into_iter().next())
            .ok_or("source UV layer is missing")?
            .faces
            .into_iter()
            .flatten()
            .flatten()
            .collect::<Vec<[f64; 2]>>();
        assert!(exported_uvs.iter().all(|actual| {
            source_uvs.iter().any(|expected| {
                (actual[0] - expected[0]).abs() <= f64::from(f32::EPSILON)
                    && (actual[1] - expected[1]).abs() <= f64::from(f32::EPSILON)
            })
        }));
        assert_eq!(document["buffers"][0]["uri"], "scene.bin");
        assert!(
            document["skins"]
                .as_array()
                .is_some_and(|skins| skins.len() == 1)
        );
        assert!(document["meshes"].as_array().is_some_and(|meshes| {
            meshes.iter().any(|mesh| {
                mesh["primitives"][0]["targets"]
                    .as_array()
                    .is_some_and(|targets| targets.len() == 1)
            })
        }));
        assert!(
            document["images"]
                .as_array()
                .is_some_and(|images| images.len() == 1)
        );
        assert!(
            document["cameras"]
                .as_array()
                .is_some_and(|cameras| cameras.len() == 1)
        );
        assert!(
            document["extensions"]["KHR_lights_punctual"]["lights"]
                .as_array()
                .is_some_and(|lights| lights.len() == 1)
        );
        assert_eq!(
            document["animations"]
                .as_array()
                .ok_or("animation exports are missing")?
                .iter()
                .filter_map(|animation| animation["samplers"][0]["interpolation"].as_str())
                .collect::<std::collections::BTreeSet<_>>(),
            ["CUBICSPLINE", "LINEAR", "STEP"].into_iter().collect()
        );
        assert!(document["nodes"].as_array().is_some_and(|nodes| {
            nodes.iter().any(|node| {
                node["extras"]["potter"]["properties"]["instance_collection"] == "source_collection"
            })
        }));
        assert_eq!(imported.doc.images.len(), 1);
        assert!(!imported.assets.is_empty(), "image asset was not imported");
        let mesh = imported
            .doc
            .nodes
            .get(&"skin_mesh".parse()?)
            .ok_or("skinned mesh was not imported")?;
        assert!(
            mesh.modifiers
                .iter()
                .any(|modifier| modifier.modifier_type == "armature")
        );
        let mesh_data = mesh
            .data
            .as_ref()
            .and_then(|id| imported.doc.data_blocks.get(id))
            .ok_or("skinned mesh data is missing")?;
        let lift_id = "lift".parse()?;
        let shape_keys = mesh_data
            .shape_keys
            .as_ref()
            .ok_or("shape keys are missing")?;
        let lift = shape_keys
            .keys
            .get(&lift_id)
            .ok_or("morph target was not imported")?;
        assert!(
            lift.positions
                .values()
                .any(|position| (position[2] - 1.0).abs() < 1.0e-6)
        );
        let surface_id = "surface".parse()?;
        let material = imported
            .doc
            .materials
            .get(&surface_id)
            .ok_or("textured material was not imported")?;
        let texture = material
            .base_color_texture
            .as_ref()
            .ok_or("base-color texture was not imported")?;
        assert_eq!(texture.image, "paint".parse()?);
        assert_eq!(
            texture.interpolation,
            crate::image::ImageInterpolation::Closest
        );
        let (width, height, pixels) = crate::image::decode_pixels(
            &imported.assets[0].1,
            crate::model::ImageColorspace::Srgb,
        )?;
        assert_eq!((width, height), (1, 1));
        for (actual, expected) in pixels[0].iter().zip([0.25, 0.5, 0.75, 1.0]) {
            assert!((actual - expected).abs() <= 0.01);
        }
        let camera = imported
            .doc
            .nodes
            .get(&"camera".parse()?)
            .and_then(|node| node.data.as_ref())
            .and_then(|id| imported.doc.data_blocks.get(id))
            .and_then(|data| data.camera.as_ref())
            .ok_or("camera settings were not imported")?;
        assert_eq!(
            camera.projection,
            crate::model::CameraProjection::Orthographic
        );
        assert!((camera.ortho_scale - 4.0).abs() <= 1.0e-6);
        let light = imported
            .doc
            .nodes
            .get(&"light".parse()?)
            .and_then(|node| node.data.as_ref())
            .and_then(|id| imported.doc.data_blocks.get(id))
            .and_then(|data| data.light.as_ref())
            .ok_or("light settings were not imported")?;
        assert_eq!(light.light_type, crate::model::LightType::Spot);
        assert!((light.energy - 25.0).abs() <= 1.0e-6);
        let instance = imported
            .doc
            .nodes
            .get(&"collection_instance".parse()?)
            .ok_or("collection instance was not imported")?;
        assert_eq!(
            instance.properties["instance_collection"],
            "source_collection"
        );
        assert!(instance.data.is_some());
        assert!(
            !mesh_data.vertex_groups.is_empty(),
            "joint vertex groups were not imported"
        );
        assert!(!mesh_data.vertex_weights.is_empty());
        let root_group_id = "root_group".parse()?;
        assert!(
            mesh_data
                .vertex_weights
                .values()
                .filter_map(|weights| weights.get(&root_group_id))
                .any(|weight| (weight - 0.5).abs() < 1.0e-6)
        );
        assert_eq!(imported.doc.actions.len(), 3);
        for (node_id, interpolation) in [
            ("step_node", crate::model::Interpolation::Constant),
            ("linear_node", crate::model::Interpolation::Linear),
            ("cubic_node", crate::model::Interpolation::Bezier),
        ] {
            let node = imported
                .doc
                .nodes
                .get(&node_id.parse()?)
                .ok_or("animated node was not imported")?;
            let action_id = node
                .action
                .as_ref()
                .ok_or("animation action was not imported")?;
            let action = imported
                .doc
                .actions
                .get(action_id)
                .ok_or("animation action data is missing")?;
            let curve = action
                .fcurves
                .iter()
                .find(|curve| curve.path == "transform.translation" && curve.index == 0)
                .ok_or("translation curve was not imported")?;
            assert_eq!(curve.keyframes.len(), 2);
            assert!((curve.keyframes[1].frame - 3.0).abs() <= 1.0e-6);
            assert!((curve.keyframes[1].value - 2.0).abs() <= 1.0e-6);
            assert!(
                curve
                    .keyframes
                    .iter()
                    .all(|key| key.interpolation == interpolation)
            );
        }
        assert!(
            imported
                .losses
                .iter()
                .any(|loss| loss.feature_id == "animation.cubic_tangents")
        );
        Ok(())
    }
    #[test]
    fn gltf_reports_unrepresented_mesh_attributes_as_losses() -> TestResult {
        let mut mesh = crate::geom::Mesh::default();
        mesh.attributes
            .insert("color".to_owned(), json!({"domain":"point","values":{}}));
        super::validate_mesh_export_attributes(&mesh, None)?;
        let losses = super::unwritten_attribute_losses(&mesh, "mesh");
        assert!(
            losses
                .iter()
                .any(|loss| loss.feature_id == "mesh.attribute" && loss.reason.contains("`color`")),
            "{losses:?}"
        );

        mesh.attributes.insert(
            "seam".to_owned(),
            json!({"domain":"edge","type":"bool","values":{}}),
        );
        mesh.attributes.insert(
            "crease_edge".to_owned(),
            json!({"domain":"edge","type":"float","values":{}}),
        );
        let losses = super::unwritten_attribute_losses(&mesh, "mesh");
        for name in ["seam", "crease_edge"] {
            assert!(
                losses
                    .iter()
                    .any(|loss| loss.feature_id == "mesh.attribute" && loss.reason.contains(name)),
                "{name} must be a mesh.attribute loss: {losses:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn explicit_zero_custom_normal_is_a_gltf_normals_loss() -> TestResult {
        let mut mesh = two_triangles()?;
        let set_codes = |mesh: &mut crate::geom::Mesh, rows: [serde_json::Value; 2]| {
            let mut values = serde_json::Map::new();
            for (face, row) in mesh.faces.iter().zip(rows) {
                values.insert(format!("f{}", face.id), row);
            }
            mesh.attributes.insert(
                "custom_normal".to_owned(),
                json!({"domain": "corner", "type": "normal_code", "values": values}),
            );
        };

        set_codes(
            &mut mesh,
            [
                json!([null, "explicit_zero", null]),
                json!(["explicit_zero", null, null]),
            ],
        );
        let loss = super::explicit_zero_normal_loss(&mesh, "mesh")?
            .ok_or("explicit zero must be reported as a loss")?;
        assert_eq!(loss.feature_id, "mesh.normals");
        assert_eq!(loss.data_id.as_deref(), Some("mesh"));
        assert!(loss.reason.contains("up-vector fallback"), "{loss:?}");
        assert!(
            loss.suggestion
                .as_deref()
                .is_some_and(|text| text.contains("--allow-lossy")),
            "{loss:?}"
        );

        set_codes(
            &mut mesh,
            [json!([null, null, null]), json!([null, null, null])],
        );
        assert!(super::explicit_zero_normal_loss(&mesh, "mesh")?.is_none());

        set_codes(
            &mut mesh,
            [
                json!([[12000, 7000], null, [-3, 4]]),
                json!([null, [0, 9], null]),
            ],
        );
        assert!(super::explicit_zero_normal_loss(&mesh, "mesh")?.is_none());

        mesh.attributes.remove("custom_normal");
        assert!(super::explicit_zero_normal_loss(&mesh, "mesh")?.is_none());
        Ok(())
    }

    /// Two triangles sharing the edge 0-2, so vertices 0 and 2 each have two corners.
    fn two_triangles() -> TestResult<crate::geom::Mesh> {
        Ok(crate::geom::Mesh::from_positions_and_faces(
            vec![
                glam::DVec3::new(0.0, 0.0, 0.0),
                glam::DVec3::new(1.0, 0.0, 0.0),
                glam::DVec3::new(1.0, 1.0, 0.0),
                glam::DVec3::new(0.0, 1.0, 0.0),
            ],
            vec![vec![0, 1, 2], vec![0, 2, 3]],
        )?)
    }

    /// Points `material` at a shader graph whose Principled BSDF takes its base color from a Vertex
    /// Color node reading `layer` (`""` reads the render layer).
    fn read_vertex_color(doc: &mut SceneDoc, material: &str, layer: &str) -> TestResult<()> {
        use crate::graph::{GraphKind, GraphLink, GraphNode, NodeGroup};
        let mut group = NodeGroup::new("paint", GraphKind::Shader);
        let mut output = GraphNode::new("OutputMaterial");
        output
            .properties
            .insert("is_active_output".to_owned(), json!(true));
        let mut vertex = GraphNode::new("ShaderNodeVertexColor");
        vertex
            .properties
            .insert("layer_name".to_owned(), json!(layer));
        group.nodes.insert(crate::model::Id::new("out")?, output);
        group.nodes.insert(
            crate::model::Id::new("bsdf")?,
            GraphNode::new("ShaderNodeBsdfPrincipled"),
        );
        group.nodes.insert(crate::model::Id::new("paint")?, vertex);
        group.links.push(GraphLink {
            from_node: crate::model::Id::new("bsdf")?,
            from_socket: "BSDF".to_owned(),
            to_node: crate::model::Id::new("out")?,
            to_socket: "Surface".to_owned(),
        });
        group.links.push(GraphLink {
            from_node: crate::model::Id::new("paint")?,
            from_socket: "Color".to_owned(),
            to_node: crate::model::Id::new("bsdf")?,
            to_socket: "Base Color".to_owned(),
        });
        let graph = crate::model::Id::new(format!("{material}_graph"))?;
        doc.node_groups.insert(graph.clone(), group);
        doc.materials
            .get_mut(&crate::model::Id::new(material)?)
            .ok_or("material is missing")?
            .node_tree = Some(graph);
        Ok(())
    }

    #[test]
    fn adding_the_primitive_color_layer_preserves_other_extras() -> TestResult {
        let mut primitive = json!({
            "extras": {
                "author": {"tag": "keep"},
                "potter": {"color_attributes": {"version": 1}, "other": "keep"}
            }
        });
        super::put_potter_primitive_color_layer(&mut primitive, "Paint")?;
        assert_eq!(primitive["extras"]["author"]["tag"], json!("keep"));
        assert_eq!(
            primitive["extras"]["potter"]["color_attributes"]["version"],
            json!(1)
        );
        assert_eq!(primitive["extras"]["potter"]["other"], json!("keep"));
        assert_eq!(primitive["extras"]["potter"]["color_layer"], json!("Paint"));
        Ok(())
    }

    #[test]
    fn vertex_color_exports_the_selected_layer_as_linear_color0_per_corner() -> TestResult {
        let root = tempfile::tempdir()?;
        let mut doc = scene_with_asset_root(
            root.path(),
            &json!([
                {"op":"material.create","id":"paint"},
                {"op":"node.create","id":"sheet","kind":"plane","params":{"size":2.0},"material":"paint"}
            ]),
        )?;
        for data in doc.data_blocks.values_mut() {
            if data.mesh.is_some() {
                data.mesh = Some(two_triangles()?);
            }
        }
        let point_values = vec![
            [0.125, 0.5, 0.25, 1.0],
            [0.5, 0.25, 0.75, 0.5],
            [0.875, 0.0625, 0.5, 1.0],
            [0.0, 0.25, 0.5, 1.0],
        ];
        let corner_values = vec![
            [0.25, 0.5, 0.75, 1.0],
            [0.5, 0.25, 0.125, 0.5],
            [0.875, 0.0625, 0.5, 1.0],
            [0.75, 0.75, 0.25, 1.0],
            [0.0625, 0.125, 0.875, 0.25],
            [0.5, 0.875, 0.0625, 1.0],
        ];
        let mesh = doc
            .data_blocks
            .values_mut()
            .find_map(|data| data.mesh.as_mut())
            .ok_or("mesh is missing")?;
        crate::geom::color_layers::write(
            mesh,
            &crate::geom::color_layers::ColorStore {
                layers: vec![
                    crate::geom::color_layers::ColorLayer {
                        name: "Point".to_owned(),
                        domain: crate::geom::color_layers::ColorDomain::Point,
                        values: point_values.clone(),
                    },
                    crate::geom::color_layers::ColorLayer {
                        name: "Corner".to_owned(),
                        domain: crate::geom::color_layers::ColorDomain::Corner,
                        values: corner_values.clone(),
                    },
                ],
                active: Some("Point".to_owned()),
                render: Some("Point".to_owned()),
            },
        );
        read_vertex_color(&mut doc, "paint", "Corner")?;

        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        let losses = super::mesh_export_losses(&doc, &snapshot)?;
        assert!(
            !losses
                .iter()
                .any(|loss| loss.feature_id == "mesh.color_attribute"),
            "every layer is written, so none is a loss: {losses:?}"
        );

        let (document, binary) = export_root_and_binary_at_root(&doc, root.path())?;
        let buffers = vec![binary];
        let mesh_value = &document["meshes"][0];
        let primitives = mesh_value["primitives"]
            .as_array()
            .ok_or("primitives are missing")?;
        assert_eq!(primitives.len(), 1);
        let accessor = primitives[0]["attributes"]["COLOR_0"]
            .as_u64()
            .ok_or("COLOR_0 is missing")?;
        let (colors, dimensions) =
            super::read_float_accessor(&document, &buffers, usize::try_from(accessor)?)?;
        assert_eq!(dimensions, 4);
        let vertex_ids = mesh_value["extras"]["potter"]["vertex_ids"]
            .as_array()
            .ok_or("vertex IDs are missing")?;
        let exported = vertex_ids
            .iter()
            .zip(colors.as_chunks::<4>().0)
            .map(|(id, color)| (id.as_u64().unwrap_or(u64::MAX), color.map(f64::to_bits)))
            .collect::<BTreeSet<_>>();
        // Each corner's vertex with its corner value. Corners that share a value and a vertex
        // share an output vertex, so the sets compare exactly.
        let mut expected = BTreeSet::new();
        let mut corner = 0;
        for face in &doc
            .data_blocks
            .values()
            .find_map(|data| data.mesh.as_ref())
            .ok_or("mesh is missing")?
            .faces
        {
            for vertex in &face.vertices {
                expected.insert((u64::from(*vertex), corner_values[corner].map(f64::to_bits)));
                corner += 1;
            }
        }
        assert_eq!(exported, expected);

        // The other named layer uses the next dense set and remains per vertex.
        let point_accessor = primitives[0]["attributes"]["COLOR_1"]
            .as_u64()
            .ok_or("COLOR_1 is missing")?;
        let (point_colors, _) =
            super::read_float_accessor(&document, &buffers, usize::try_from(point_accessor)?)?;
        let exported_points = vertex_ids
            .iter()
            .zip(point_colors.as_chunks::<4>().0)
            .map(|(id, color)| (id.as_u64().unwrap_or(u64::MAX), color.map(f64::to_bits)))
            .collect::<BTreeSet<_>>();
        let expected_points = doc
            .data_blocks
            .values()
            .find_map(|data| data.mesh.as_ref())
            .ok_or("mesh is missing")?
            .vertices
            .iter()
            .zip(&point_values)
            .map(|(vertex, color)| (u64::from(vertex.id), color.map(f64::to_bits)))
            .collect::<BTreeSet<_>>();
        assert_eq!(exported_points, expected_points);
        let metadata = &mesh_value["extras"]["potter"]["color_attributes"];
        assert_eq!(metadata["active"], json!("Point"));
        assert_eq!(metadata["render"], json!("Point"));
        assert_eq!(metadata["layers"][0]["name"], json!("Point"));
        assert_eq!(metadata["layers"][0]["domain"], json!("point"));
        assert_eq!(metadata["layers"][0]["values"], json!(point_values));
        assert_eq!(metadata["layers"][1]["name"], json!("Corner"));
        assert_eq!(metadata["layers"][1]["domain"], json!("corner"));
        assert_eq!(
            metadata["layers"][1]["values"],
            json!([corner_values[..3].to_vec(), corner_values[3..].to_vec()])
        );

        let (imported, _) = round_trip_in(&doc, root.path())?;
        let store = crate::geom::color_layers::read(first_mesh(&imported.doc)?);
        assert_eq!(store.layers.len(), 2);
        assert_eq!(store.layers[0].name, "Point");
        assert_eq!(
            store.layers[0].domain,
            crate::geom::color_layers::ColorDomain::Point
        );
        assert_eq!(store.layers[0].values, point_values);
        assert_eq!(store.layers[1].name, "Corner");
        assert_eq!(
            store.layers[1].domain,
            crate::geom::color_layers::ColorDomain::Corner
        );
        assert_eq!(store.layers[1].values, corner_values);
        assert_eq!(store.active.as_deref(), Some("Point"));
        assert_eq!(store.render.as_deref(), Some("Point"));
        let imported_node = &imported.doc.nodes[&crate::model::Id::new("sheet")?];
        let material_id = imported_node
            .data
            .as_ref()
            .and_then(|data_id| imported.doc.data_blocks.get(data_id))
            .and_then(|data| data.materials.first())
            .and_then(Option::as_ref)
            .ok_or("material variant is missing")?;
        assert_eq!(imported_node.materials.len(), 0);
        assert_eq!(
            imported_node.material_link(0),
            crate::model::MaterialLink::Data
        );
        let surfaces = super::resolve_material_surfaces(&imported.doc)?;
        assert_eq!(
            surfaces[material_id]
                .vertex_color
                .as_ref()
                .and_then(|read| read.layer.as_deref()),
            Some("Corner")
        );

        Ok(())
    }

    #[test]
    fn unused_color_layer_is_preserved_without_a_color_semantic_or_material_tint() -> TestResult {
        let root = tempfile::tempdir()?;
        let mut doc = scene_with_asset_root(
            root.path(),
            &json!([
                {"op":"material.create","id":"plain"},
                {"op":"node.create","id":"sheet","kind":"plane","params":{"size":2.0},"material":"plain"}
            ]),
        )?;
        let mesh = doc
            .data_blocks
            .values_mut()
            .find_map(|block| block.mesh.as_mut())
            .ok_or("mesh is missing")?;
        *mesh = two_triangles()?;
        let values = vec![
            [0.125, 0.5, 0.875, 0.25],
            [0.5, 0.25, 0.125, 0.5],
            [0.875, 0.0625, 0.5, 1.0],
            [0.75, 0.75, 0.25, 1.0],
            [0.0625, 0.125, 0.875, 0.25],
            [0.5, 0.875, 0.0625, 1.0],
        ];
        crate::geom::color_layers::write(
            mesh,
            &crate::geom::color_layers::ColorStore {
                layers: vec![crate::geom::color_layers::ColorLayer {
                    name: "Unused".to_owned(),
                    domain: crate::geom::color_layers::ColorDomain::Corner,
                    values: values.clone(),
                }],
                active: Some("Unused".to_owned()),
                render: Some("Unused".to_owned()),
            },
        );
        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        let losses = super::mesh_export_losses(&doc, &snapshot)?;
        assert!(
            !losses
                .iter()
                .any(|loss| loss.feature_id == "mesh.color_attribute"),
            "Potter extras preserve unused color data: {losses:?}"
        );
        let (document, _) = export_root_and_binary_at_root(&doc, root.path())?;
        let attributes = document["meshes"][0]["primitives"][0]["attributes"]
            .as_object()
            .ok_or("primitive attributes are missing")?;
        assert!(
            attributes.keys().all(|name| !name.starts_with("COLOR_")),
            "an unconsumed layer must not tint the plain material or leave a COLOR_n gap: {attributes:?}"
        );
        let metadata = &document["meshes"][0]["extras"]["potter"]["color_attributes"];
        assert_eq!(metadata["layers"][0]["name"], json!("Unused"));
        assert_eq!(metadata["layers"][0]["domain"], json!("corner"));
        assert_eq!(
            metadata["layers"][0]["values"],
            json!([values[..3].to_vec(), values[3..].to_vec()])
        );
        let (imported, _) = round_trip_in(&doc, root.path())?;
        let restored = crate::geom::color_layers::read(first_mesh(&imported.doc)?);
        assert_eq!(restored.layers[0].name, "Unused");
        assert_eq!(restored.layers[0].values, values);
        let imported_material = &imported.doc.materials[&crate::model::Id::new("plain")?];
        let surfaces = super::resolve_material_surfaces(&imported.doc)?;
        assert!(
            surfaces[&crate::model::Id::new("plain")?]
                .vertex_color
                .is_none()
        );
        assert_eq!(
            imported_material.base_color,
            doc.materials[&crate::model::Id::new("plain")?].base_color
        );
        Ok(())
    }

    #[test]
    fn unassigned_color_layers_are_preserved_without_core_semantics() -> TestResult {
        let root = tempfile::tempdir()?;
        let mut doc = scene(&json!([
            {"op":"node.create","id":"sheet","kind":"plane","params":{"size":2.0}}
        ]))?;
        let mesh = doc
            .data_blocks
            .values_mut()
            .find_map(|block| block.mesh.as_mut())
            .ok_or("mesh is missing")?;
        *mesh = two_triangles()?;
        let values = vec![
            [0.125, 0.5, 0.875, 0.25],
            [0.5, 0.25, 0.125, 0.5],
            [0.875, 0.0625, 0.5, 1.0],
            [0.75, 0.75, 0.25, 1.0],
            [0.0625, 0.125, 0.875, 0.25],
            [0.5, 0.875, 0.0625, 1.0],
        ];
        crate::geom::color_layers::write(
            mesh,
            &crate::geom::color_layers::ColorStore {
                layers: vec![crate::geom::color_layers::ColorLayer {
                    name: "Unassigned".to_owned(),
                    domain: crate::geom::color_layers::ColorDomain::Corner,
                    values: values.clone(),
                }],
                active: Some("Unassigned".to_owned()),
                render: Some("Unassigned".to_owned()),
            },
        );
        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        let (document, _) = export_root_and_binary_at_root(&doc, root.path())?;
        let attributes = document["meshes"][0]["primitives"][0]["attributes"]
            .as_object()
            .ok_or("primitive attributes are missing")?;
        assert!(attributes.keys().all(|name| !name.starts_with("COLOR_")));
        assert_eq!(
            document["meshes"][0]["extras"]["potter"]["color_attributes"]["layers"][0]["values"],
            json!([values[..3].to_vec(), values[3..].to_vec()])
        );
        assert!(
            !super::mesh_export_losses(&doc, &snapshot)?
                .iter()
                .any(|loss| loss.feature_id == "mesh.color_attribute")
        );
        let (imported, _) = round_trip_in(&doc, root.path())?;
        let restored = crate::geom::color_layers::read(first_mesh(&imported.doc)?);
        assert_eq!(restored.layers[0].name, "Unassigned");
        assert_eq!(restored.layers[0].values, values);
        Ok(())
    }

    #[test]
    fn vertex_color_losses_name_missing_layers_and_translucent_alpha() -> TestResult {
        let root = tempfile::tempdir()?;
        let mut doc = scene_with_asset_root(
            root.path(),
            &json!([
                {"op":"material.create","id":"paint"},
                {"op":"material.create","id":"opaque"},
                {"op":"node.create","id":"sheet","kind":"plane","params":{"size":2.0},"material":"paint"}
            ]),
        )?;
        for data in doc.data_blocks.values_mut() {
            if data.mesh.is_some() {
                data.mesh = Some(two_triangles()?);
            }
        }
        let mesh = doc
            .data_blocks
            .values_mut()
            .find_map(|data| data.mesh.as_mut())
            .ok_or("mesh is missing")?;
        crate::geom::color_layers::write(
            mesh,
            &crate::geom::color_layers::ColorStore {
                layers: vec![crate::geom::color_layers::ColorLayer {
                    name: "Render".to_owned(),
                    domain: crate::geom::color_layers::ColorDomain::Point,
                    values: vec![[0.2, 0.4, 0.8, 0.35]; 4],
                }],
                active: Some("Render".to_owned()),
                render: Some("Render".to_owned()),
            },
        );
        read_vertex_color(&mut doc, "paint", "Missing")?;
        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        let losses = super::mesh_export_losses(&doc, &snapshot)?;
        assert!(
            losses.iter().any(|loss| {
                loss.feature_id == "material.shader_graph.vertex_color"
                    && loss.reason.contains("`Missing`")
            }),
            "a read of a layer the mesh lacks is a loss: {losses:?}"
        );

        let (document, _) = export_root_and_binary_at_root(&doc, root.path())?;
        assert!(
            document["meshes"][0]["primitives"][0]["attributes"]["COLOR_0"].is_null(),
            "a missing named layer does not fall back to the available render layer"
        );
        let sheet_id = crate::model::Id::new("sheet")?;
        doc.nodes
            .get_mut(&sheet_id)
            .ok_or("sheet node is missing")?
            .materials
            .push(Some(crate::model::Id::new("opaque")?));
        let mesh = doc
            .data_blocks
            .values_mut()
            .find_map(|data| data.mesh.as_mut())
            .ok_or("mesh is missing")?;
        mesh.faces[1].material_index = 1;
        crate::geom::color_layers::write(
            mesh,
            &crate::geom::color_layers::ColorStore {
                layers: vec![crate::geom::color_layers::ColorLayer {
                    name: "Paint".to_owned(),
                    domain: crate::geom::color_layers::ColorDomain::Corner,
                    values: vec![
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 0.5],
                        [1.0, 1.0, 1.0, 0.5],
                        [1.0, 1.0, 1.0, 0.5],
                    ],
                }],
                active: Some("Paint".to_owned()),
                render: Some("Paint".to_owned()),
            },
        );
        read_vertex_color(&mut doc, "paint", "Paint")?;
        doc.materials
            .get_mut(&crate::model::Id::new("paint")?)
            .ok_or("material is missing")?
            .alpha_mode = "blend".to_owned();
        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        let losses = super::mesh_export_losses(&doc, &snapshot)?;
        assert!(
            !losses.iter().any(|loss| {
                loss.feature_id == "material.shader_graph.vertex_color"
                    && loss.reason.contains("`Paint`")
            }),
            "alpha on the opaque slot does not change the translucent slot: {losses:?}"
        );
        let mesh = doc
            .data_blocks
            .values_mut()
            .find_map(|data| data.mesh.as_mut())
            .ok_or("mesh is missing")?;
        let mut colors = crate::geom::color_layers::read(mesh);
        colors.layers[0].values[..3].fill([1.0, 1.0, 1.0, 0.5]);
        crate::geom::color_layers::write(mesh, &colors);
        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        let losses = super::mesh_export_losses(&doc, &snapshot)?;
        assert!(
            losses.iter().any(|loss| {
                loss.feature_id == "material.shader_graph.vertex_color"
                    && loss.reason.contains("`Paint`")
            }),
            "translucent COLOR_0 alpha is a loss: {losses:?}"
        );
        let mesh = doc
            .data_blocks
            .values_mut()
            .find_map(|data| data.mesh.as_mut())
            .ok_or("mesh is missing")?;
        crate::geom::color_layers::write(
            mesh,
            &crate::geom::color_layers::ColorStore {
                layers: vec![crate::geom::color_layers::ColorLayer {
                    name: "Paint".to_owned(),
                    domain: crate::geom::color_layers::ColorDomain::Point,
                    values: vec![
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 0.5],
                    ],
                }],
                active: Some("Paint".to_owned()),
                render: Some("Paint".to_owned()),
            },
        );
        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        let losses = super::mesh_export_losses(&doc, &snapshot)?;
        assert!(
            !losses.iter().any(|loss| {
                loss.feature_id == "material.shader_graph.vertex_color"
                    && loss.reason.contains("`Paint`")
            }),
            "a nonopaque point value used only by the opaque slot is not a loss: {losses:?}"
        );
        let mesh = doc
            .data_blocks
            .values_mut()
            .find_map(|data| data.mesh.as_mut())
            .ok_or("mesh is missing")?;
        let mut colors = crate::geom::color_layers::read(mesh);
        colors.layers[0].values[0][3] = 0.5;
        crate::geom::color_layers::write(mesh, &colors);
        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        let losses = super::mesh_export_losses(&doc, &snapshot)?;
        assert!(
            losses.iter().any(|loss| {
                loss.feature_id == "material.shader_graph.vertex_color"
                    && loss.reason.contains("`Paint`")
            }),
            "a nonopaque point value used by the translucent slot is a loss: {losses:?}"
        );
        Ok(())
    }

    #[test]
    fn empty_vertex_color_uses_member_render_and_availability_without_named_fallback() -> TestResult
    {
        let root = tempfile::tempdir()?;
        let mut doc = scene(&json!([
            {"op":"material.create","id":"paint"},
            {"op":"node.create","id":"sheet","kind":"plane","params":{"size":2.0},"material":"paint"}
        ]))?;
        let data_id = doc.nodes[&crate::model::Id::new("sheet")?]
            .data
            .clone()
            .ok_or("sheet mesh data is missing")?;
        let mesh = doc
            .data_blocks
            .get_mut(&data_id)
            .and_then(|block| block.mesh.as_mut())
            .ok_or("sheet mesh is missing")?;
        *mesh = two_triangles()?;
        let colors = [
            [0.875, 0.125, 0.25, 0.375],
            [0.125, 0.25, 0.5, 0.75],
            [0.75, 0.5, 0.25, 0.125],
            [0.25, 0.375, 0.625, 0.875],
        ];
        crate::geom::color_layers::write(
            mesh,
            &crate::geom::color_layers::ColorStore {
                layers: colors
                    .iter()
                    .enumerate()
                    .map(|(index, color)| crate::geom::color_layers::ColorLayer {
                        name: ["A", "B", "C", "D"][index].to_owned(),
                        domain: crate::geom::color_layers::ColorDomain::Point,
                        values: vec![*color; 4],
                    })
                    .collect(),
                active: Some("A".to_owned()),
                render: Some("A".to_owned()),
            },
        );
        read_vertex_color(&mut doc, "paint", "")?;
        let mut snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        let node_id = crate::model::Id::new("sheet")?;
        snapshot.slot_bindings.insert(
            node_id.clone(),
            vec![crate::eval::SlotBinding {
                material: Some(crate::model::Id::new("paint")?),
                uv: crate::eval::UvBinding::default(),
                color_render_layer: Some(3),
                color_available_mask: std::sync::Arc::from(vec![0b1010_u64]),
            }],
        );
        {
            let surfaces = super::resolve_material_surfaces(&doc)?;
            let node = &doc.nodes[&node_id];
            let mesh = doc.data_blocks[&data_id]
                .mesh
                .as_ref()
                .ok_or("sheet mesh is missing")?;
            let plan = super::color_plan(
                &node_id,
                node,
                &snapshot,
                Some(&doc.data_blocks[&data_id]),
                mesh,
                &surfaces,
            );
            assert_eq!(plan.attributes(0), vec![(0, 3), (1, 1)]);
            assert_eq!(plan.carried_layers(), vec![1, 3]);

            let (document, binary) =
                export_with_root(&doc, &snapshot, root.path(), Some("scene.bin"))?;
            let document: Value = serde_json::from_slice(&document)?;
            let buffers = [binary];
            let primitive = &document["meshes"][0]["primitives"][0];
            assert!(primitive["attributes"]["COLOR_0"].is_u64());
            assert!(primitive["attributes"]["COLOR_1"].is_u64());
            assert!(primitive["attributes"]["COLOR_2"].is_null());
            for (set, expected) in [(0, colors[3]), (1, colors[1])] {
                let accessor = usize::try_from(
                    primitive["attributes"][format!("COLOR_{set}")]
                        .as_u64()
                        .ok_or("color accessor is missing")?,
                )?;
                let (actual, dimensions) =
                    super::read_float_accessor(&document, &buffers, accessor)?;
                assert_eq!(dimensions, 4);
                assert!(
                    actual
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .all(|color| *color == expected)
                );
            }
        }

        read_vertex_color(&mut doc, "paint", "Missing")?;
        let surfaces = super::resolve_material_surfaces(&doc)?;
        let node = &doc.nodes[&node_id];
        let mesh = doc.data_blocks[&data_id]
            .mesh
            .as_ref()
            .ok_or("sheet mesh is missing")?;
        let plan = super::color_plan(
            &node_id,
            node,
            &snapshot,
            Some(&doc.data_blocks[&data_id]),
            mesh,
            &surfaces,
        );
        assert_eq!(plan.attributes(0).len(), 0);
        assert_eq!(plan.missing.get(&0), Some(&Some("Missing".to_owned())));
        Ok(())
    }

    #[test]
    fn material_slots_select_distinct_color0_layers_without_indexed_gaps_or_losses() -> TestResult {
        let root = tempfile::tempdir()?;
        let mut doc = scene(&json!([
            {"op":"material.create","id":"first"},
            {"op":"material.create","id":"second"},
            {"op":"node.create","id":"sheet","kind":"plane","params":{"size":2.0}},
            {"op":"node.update","target":{"id":"sheet"},"set":{"materials":["first","second"]}}
        ]))?;
        let node_id = crate::model::Id::new("sheet")?;
        let data_id = doc.nodes[&node_id]
            .data
            .clone()
            .ok_or("sheet mesh data is missing")?;
        let mut mesh = two_triangles()?;
        mesh.faces[1].material_index = 1;
        crate::geom::color_layers::write(
            &mut mesh,
            &crate::geom::color_layers::ColorStore {
                layers: vec![
                    crate::geom::color_layers::ColorLayer {
                        name: "First".to_owned(),
                        domain: crate::geom::color_layers::ColorDomain::Point,
                        values: vec![[0.1, 0.2, 0.3, 1.0]; 4],
                    },
                    crate::geom::color_layers::ColorLayer {
                        name: "Second".to_owned(),
                        domain: crate::geom::color_layers::ColorDomain::Point,
                        values: vec![[0.7, 0.8, 0.9, 1.0]; 4],
                    },
                ],
                active: Some("First".to_owned()),
                render: Some("First".to_owned()),
            },
        );
        doc.data_blocks
            .get_mut(&data_id)
            .ok_or("sheet mesh data block is missing")?
            .mesh = Some(mesh);
        read_vertex_color(&mut doc, "first", "First")?;
        read_vertex_color(&mut doc, "second", "Second")?;

        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        let surfaces = super::resolve_material_surfaces(&doc)?;
        let node = &doc.nodes[&node_id];
        let mesh = doc.data_blocks[&data_id]
            .mesh
            .as_ref()
            .ok_or("sheet mesh is missing")?;
        let plan = super::color_plan(
            &node_id,
            node,
            &snapshot,
            Some(&doc.data_blocks[&data_id]),
            mesh,
            &surfaces,
        );
        for (slot, selected, other) in [(0, 0, 1), (1, 1, 0)] {
            assert_eq!(
                plan.attributes(slot),
                vec![(0, selected), (1, other)],
                "slot {slot} has dense COLOR_0 and COLOR_1 sets"
            );
        }

        let losses = super::mesh_export_losses(&doc, &snapshot)?;
        assert!(
            !losses
                .iter()
                .any(|loss| loss.feature_id == "mesh.color_attribute"),
            "Potter extras restore both logical layers: {losses:?}"
        );
        let (document, binary) = export_root_and_binary_at_root(&doc, root.path())?;
        let primitives = document["meshes"][0]["primitives"]
            .as_array()
            .ok_or("mesh primitives are missing")?;
        assert_eq!(primitives.len(), 2);
        let buffers = vec![binary];
        for (primitive, (name, expected, other)) in primitives.iter().zip([
            ("First", [0.1, 0.2, 0.3, 1.0], [0.7, 0.8, 0.9, 1.0]),
            ("Second", [0.7, 0.8, 0.9, 1.0], [0.1, 0.2, 0.3, 1.0]),
        ]) {
            assert!(primitive["attributes"]["COLOR_0"].is_u64());
            assert!(primitive["attributes"]["COLOR_1"].is_u64());
            assert!(primitive["attributes"]["COLOR_2"].is_null());
            assert_eq!(primitive["extras"]["potter"]["color_layer"], json!(name));
            for (set, expected) in [(0, expected), (1, other)] {
                let accessor = usize::try_from(
                    primitive["attributes"][format!("COLOR_{set}")]
                        .as_u64()
                        .ok_or("color accessor is missing")?,
                )?;
                let (colors, dimensions) =
                    super::read_float_accessor(&document, &buffers, accessor)?;
                assert_eq!(dimensions, 4);
                assert!(colors.as_chunks::<4>().0.iter().all(|color| {
                    color
                        .iter()
                        .zip(expected)
                        .all(|(actual, expected)| (*actual - expected).abs() < 1e-6)
                }));
            }
        }
        let metadata = &document["meshes"][0]["extras"]["potter"]["color_attributes"];
        assert_eq!(metadata["layers"][0]["name"], json!("First"));
        assert_eq!(metadata["layers"][1]["name"], json!("Second"));
        let reimport_root = tempfile::tempdir()?;
        let (imported, _) = round_trip_in(&doc, reimport_root.path())?;
        let restored = crate::geom::color_layers::read(first_mesh(&imported.doc)?);
        assert_eq!(restored.layers[0].name, "First");
        assert_eq!(restored.layers[1].name, "Second");
        assert_eq!(restored.layers[0].values, vec![[0.1, 0.2, 0.3, 1.0]; 4]);
        assert_eq!(restored.layers[1].values, vec![[0.7, 0.8, 0.9, 1.0]; 4]);
        Ok(())
    }

    #[test]
    fn hdr_color_is_clamped_for_color0_and_preserved_raw_in_potter_extras() -> TestResult {
        let root = tempfile::tempdir()?;
        let mut doc = scene_with_asset_root(
            root.path(),
            &json!([
                {"op":"material.create","id":"paint"},
                {"op":"node.create","id":"sheet","kind":"plane","params":{"size":2.0},"material":"paint"}
            ]),
        )?;
        let mesh = doc
            .data_blocks
            .values_mut()
            .find_map(|block| block.mesh.as_mut())
            .ok_or("mesh is missing")?;
        *mesh = two_triangles()?;
        let values = vec![
            [2.0, 0.25, 0.5, 1.0],
            [0.125, 0.75, 1.5, 0.25],
            [0.5, 0.25, 0.125, 1.0],
            [0.25, 0.5, 0.75, 1.0],
        ];
        crate::geom::color_layers::write(
            mesh,
            &crate::geom::color_layers::ColorStore {
                layers: vec![crate::geom::color_layers::ColorLayer {
                    name: "HDR".to_owned(),
                    domain: crate::geom::color_layers::ColorDomain::Point,
                    values: values.clone(),
                }],
                active: Some("HDR".to_owned()),
                render: Some("HDR".to_owned()),
            },
        );
        read_vertex_color(&mut doc, "paint", "HDR")?;
        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        let losses = super::mesh_export_losses(&doc, &snapshot)?;
        assert!(
            losses.iter().any(|loss| {
                loss.feature_id == "mesh.color_attribute"
                    && loss.reason.contains("outside glTF COLOR_n's [0, 1] range")
            }),
            "strict export must report the core range limitation: {losses:?}"
        );
        let (document, binary) = export_root_and_binary_at_root(&doc, root.path())?;
        let buffers = vec![binary];
        let primitive = &document["meshes"][0]["primitives"][0];
        let accessor = usize::try_from(
            primitive["attributes"]["COLOR_0"]
                .as_u64()
                .ok_or("COLOR_0 is missing")?,
        )?;
        let (colors, dimensions) = super::read_float_accessor(&document, &buffers, accessor)?;
        assert_eq!(dimensions, 4);
        assert!(
            colors
                .as_chunks::<4>()
                .0
                .iter()
                .flatten()
                .all(|component| (0.0..=1.0).contains(component)),
            "lossy COLOR_0 remains legal"
        );
        let ids = document["meshes"][0]["extras"]["potter"]["vertex_ids"]
            .as_array()
            .ok_or("vertex IDs are missing")?;
        let mesh = first_mesh(&doc)?;
        let exported = ids
            .iter()
            .zip(colors.as_chunks::<4>().0)
            .map(|(id, color)| (id.as_u64().unwrap_or(u64::MAX), *color))
            .collect::<BTreeMap<_, _>>();
        for (vertex, value) in mesh.vertices.iter().zip(&values) {
            assert_eq!(
                exported[&u64::from(vertex.id)],
                value.map(|component| component.clamp(0.0, 1.0))
            );
        }
        assert_eq!(
            document["meshes"][0]["extras"]["potter"]["color_attributes"]["layers"][0]["values"],
            json!(values)
        );
        let (imported, _) = round_trip_in(&doc, root.path())?;
        let restored = crate::geom::color_layers::read(first_mesh(&imported.doc)?);
        assert_eq!(restored.layers[0].name, "HDR");
        assert_eq!(restored.layers[0].values, values);
        Ok(())
    }

    #[test]
    fn color_accessor_fallback_restores_quad_rows_with_potter_polygons() -> TestResult {
        let mut doc = scene(&json!([
            {"op":"material.create","id":"paint"},
            {"op":"node.create","id":"sheet","kind":"plane","params":{"size":2.0},"material":"paint"}
        ]))?;
        let expected = [
            [0.125, 0.25, 0.375, 0.5],
            [0.25, 0.375, 0.5, 0.625],
            [0.375, 0.5, 0.625, 0.75],
            [0.5, 0.625, 0.75, 0.875],
        ];
        let mesh = doc
            .data_blocks
            .values_mut()
            .find_map(|block| block.mesh.as_mut())
            .ok_or("mesh is missing")?;
        *mesh = Mesh::from_positions_and_faces(
            vec![DVec3::ZERO, DVec3::X, DVec3::ONE, DVec3::Y],
            vec![vec![0, 1, 2, 3]],
        )?;
        crate::geom::color_layers::write(
            mesh,
            &crate::geom::color_layers::ColorStore {
                layers: vec![crate::geom::color_layers::ColorLayer {
                    name: "Paint".to_owned(),
                    domain: crate::geom::color_layers::ColorDomain::Point,
                    values: expected.to_vec(),
                }],
                active: Some("Paint".to_owned()),
                render: Some("Paint".to_owned()),
            },
        );
        read_vertex_color(&mut doc, "paint", "Paint")?;

        let imported = import_edited(&doc, |root, _| {
            root["meshes"][0]["extras"]["potter"]
                .as_object_mut()
                .ok_or("Potter mesh extras are missing")?
                .remove("color_attributes");
            Ok(())
        })??;
        let mesh = first_mesh(&imported.doc)?;
        assert_eq!(mesh.faces.len(), 1);
        assert_eq!(mesh.faces[0].vertices.len(), 4);
        let store = crate::geom::color_layers::read(mesh);
        assert_eq!(store.layers[0].name, "Col");
        assert_eq!(
            store.layers[0].domain,
            crate::geom::color_layers::ColorDomain::Point
        );
        assert_eq!(store.layers[0].values, expected);
        Ok(())
    }

    #[test]
    fn imported_color0_uses_a_material_variant_for_colorless_primitives() -> TestResult {
        let root = tempfile::tempdir()?;
        let mut doc = scene_with_asset_root(
            root.path(),
            &json!([
                {"op":"material.create","id":"paint"},
                {"op":"node.create","id":"sheet","kind":"plane","params":{"size":2.0},"material":"paint"}
            ]),
        )?;
        let mesh = doc
            .data_blocks
            .values_mut()
            .find_map(|block| block.mesh.as_mut())
            .ok_or("mesh is missing")?;
        *mesh = two_triangles()?;
        crate::geom::color_layers::write(
            mesh,
            &crate::geom::color_layers::ColorStore {
                layers: vec![crate::geom::color_layers::ColorLayer {
                    name: "Paint".to_owned(),
                    domain: crate::geom::color_layers::ColorDomain::Point,
                    values: vec![
                        [0.125, 0.5, 0.875, 0.35],
                        [0.25, 0.75, 0.125, 1.0],
                        [0.875, 0.0625, 0.5, 1.0],
                        [0.5, 0.25, 0.75, 1.0],
                    ],
                }],
                active: Some("Paint".to_owned()),
                render: Some("Paint".to_owned()),
            },
        );
        doc.materials
            .get_mut(&crate::model::Id::new("paint")?)
            .ok_or("paint material is missing")?
            .alpha_mode = "blend".to_owned();
        read_vertex_color(&mut doc, "paint", "Paint")?;
        let imported = import_edited(&doc, |root, _| {
            let primitive = root["meshes"][0]["primitives"][0].clone();
            let mut colorless = primitive.clone();
            colorless["attributes"]
                .as_object_mut()
                .ok_or("primitive attributes are missing")?
                .remove("COLOR_0");
            let primitives = root["meshes"][0]["primitives"]
                .as_array_mut()
                .ok_or("mesh primitives are missing")?;
            for primitive in primitives.iter_mut() {
                if let Some(extras) = primitive["extras"]["potter"].as_object_mut() {
                    extras.remove("color_layer");
                }
            }
            primitives.push(colorless);
            let potter = root["meshes"][0]["extras"]["potter"]
                .as_object_mut()
                .ok_or("Potter mesh extras are missing")?;
            potter.remove("vertex_ids");
            potter.remove("polygons");
            potter.remove("color_attributes");
            Ok(())
        })??;
        let node = &imported.doc.nodes[&crate::model::Id::new("sheet")?];
        let data = imported
            .doc
            .data_blocks
            .get(node.data.as_ref().ok_or("mesh data is missing")?)
            .ok_or("mesh data is missing")?;
        let first_material = data.materials[0]
            .as_ref()
            .ok_or("color material is missing")?;
        let second_material = data.materials[1]
            .as_ref()
            .ok_or("colorless material is missing")?;
        assert_eq!(node.materials.len(), 0);
        assert_eq!(data.materials.len(), 2);
        assert_ne!(first_material, second_material);
        let color_material = &imported.doc.materials[first_material];
        assert_eq!(color_material.alpha_mode, "blend");
        let graph_id = color_material
            .node_tree
            .as_ref()
            .ok_or("color material graph is missing")?;
        let graph = &imported.doc.node_groups[graph_id];
        assert!(graph.nodes.values().any(|node| {
            node.node_type == "ShaderNodeMath"
                && node.properties.get("operation") == Some(&json!("MULTIPLY"))
        }));
        let surfaces = super::resolve_material_surfaces(&imported.doc)?;
        assert_eq!(
            surfaces[first_material]
                .vertex_color
                .as_ref()
                .and_then(|read| read.layer.as_deref()),
            Some("Col")
        );
        assert!(surfaces[second_material].vertex_color.is_none());
        let mesh = imported.doc.data_blocks[node.data.as_ref().ok_or("mesh data is missing")?]
            .mesh
            .as_ref()
            .ok_or("mesh is missing")?;
        let color_store = crate::geom::color_layers::read(mesh);
        assert_eq!(
            color_store.layers[0].domain,
            crate::geom::color_layers::ColorDomain::Corner
        );
        let expected_color_values = [
            [0.125, 0.5, 0.875, 0.35],
            [0.25, 0.75, 0.125, 1.0],
            [0.875, 0.0625, 0.5, 1.0],
            [0.125, 0.5, 0.875, 0.35],
            [0.875, 0.0625, 0.5, 1.0],
            [0.5, 0.25, 0.75, 1.0],
            [1.0; 4],
            [1.0; 4],
            [1.0; 4],
            [1.0; 4],
            [1.0; 4],
            [1.0; 4],
        ];
        assert_eq!(
            color_store.layers[0].values.len(),
            expected_color_values.len()
        );
        for (actual, expected) in color_store.layers[0]
            .values
            .iter()
            .zip(expected_color_values)
        {
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((actual - expected).abs() <= 1.0e-6);
            }
        }
        assert!(mesh.faces.iter().any(|face| face.material_index == 0));
        assert!(mesh.faces.iter().any(|face| face.material_index == 1));

        let snapshot = Snapshot::evaluate(&imported.doc, &EvaluationContext::default())?;
        let (document, _) = export_root_and_binary(&imported.doc)?;
        let primitives = document["meshes"][0]["primitives"]
            .as_array()
            .ok_or("re-exported primitives are missing")?;
        assert_eq!(
            primitives
                .iter()
                .filter(|primitive| primitive["attributes"]["COLOR_0"].is_u64())
                .count(),
            1
        );
        assert!(
            super::mesh_export_losses(&imported.doc, &snapshot)?
                .iter()
                .all(|loss| loss.feature_id != "mesh.color_attribute")
        );
        Ok(())
    }

    /// Imports two copies of welded vertex 2: glTF vertex 4 (in the second triangle) takes `copy`.
    fn import_copies(copy: [f64; 4]) -> TestResult<crate::geom::color_layers::ColorStore> {
        let red = [1.0, 0.0, 0.0, 1.0];
        let green = [0.0, 1.0, 0.0, 1.0];
        let blue = [0.0, 0.0, 1.0, 1.0];
        let white = [1.0; 4];
        let mut mesh = crate::geom::Mesh::from_positions_and_faces(
            vec![
                glam::DVec3::ZERO,
                glam::DVec3::X,
                glam::DVec3::Y,
                glam::DVec3::Z,
            ],
            vec![vec![0, 1, 2], vec![0, 2, 3]],
        )?;
        let sets = BTreeMap::from([(
            0_usize,
            vec![Some(red), Some(green), Some(blue), Some(white), Some(copy)],
        )]);
        let face_sets = BTreeMap::from([(
            0_usize,
            vec![
                vec![Some(red), Some(green), Some(blue)],
                vec![Some(red), Some(copy), Some(white)],
            ],
        )]);
        super::import_color_layers(
            &mut mesh,
            &[vec![0, 1, 2], vec![0, 4, 3]],
            &sets,
            &face_sets,
            &[0, 1, 2, 3, 2],
            &[0, 1, 2, 3],
        )?;
        Ok(crate::geom::color_layers::read(&mesh))
    }

    #[test]
    fn imported_color_copies_keep_per_corner_differences() -> TestResult {
        let agreeing = import_copies([0.0, 0.0, 1.0, 1.0])?;
        assert_eq!(
            agreeing.layers[0].domain,
            crate::geom::color_layers::ColorDomain::Point
        );
        assert_eq!(agreeing.layers[0].values.len(), 4);

        let disagreeing = import_copies([0.0, 0.0, 0.0, 1.0])?;
        assert_eq!(disagreeing.layers[0].name, "Col");
        assert_eq!(
            disagreeing.layers[0].domain,
            crate::geom::color_layers::ColorDomain::Corner
        );
        assert_eq!(
            disagreeing.layers[0].values,
            vec![
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0; 4],
            ]
        );
        assert_eq!(disagreeing.active.as_deref(), Some("Col"));
        Ok(())
    }

    #[test]
    fn color_accessors_decode_float_and_normalized_integer_components() -> TestResult {
        let mut binary = Vec::new();
        for value in [0.5_f32, 0.25, 0.75] {
            binary.extend_from_slice(&value.to_le_bytes());
        }
        binary.extend_from_slice(&[255, 128, 0, 255]);
        for value in [0_u16, 32768, u16::MAX, u16::MAX] {
            binary.extend_from_slice(&value.to_le_bytes());
        }
        let root = json!({
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": 12},
                {"buffer": 0, "byteOffset": 12, "byteLength": 4},
                {"buffer": 0, "byteOffset": 16, "byteLength": 8}
            ],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": 1, "type": "VEC3"},
                {"bufferView": 1, "componentType": 5121, "normalized": true, "count": 1, "type": "VEC4"},
                {"bufferView": 2, "componentType": 5123, "normalized": true, "count": 1, "type": "VEC4"}
            ]
        });
        let buffers = vec![binary];
        assert_eq!(
            super::read_color(&root, &buffers, 0)?,
            vec![[0.5, 0.25, 0.75, 1.0]]
        );
        assert_eq!(
            super::read_color(&root, &buffers, 1)?,
            vec![[1.0, 128.0 / 255.0, 0.0, 1.0]]
        );
        assert_eq!(
            super::read_color(&root, &buffers, 2)?,
            vec![[0.0, 32768.0 / 65535.0, 1.0, 1.0]]
        );
        Ok(())
    }

    #[test]
    fn sparse_color_accessors_apply_overrides_and_zero_base() -> TestResult {
        let mut binary = Vec::new();
        for value in [0.0_f32, 0.25, 0.5, 0.75, 1.0, 0.0] {
            binary.extend_from_slice(&value.to_le_bytes());
        }
        binary.push(1);
        binary.extend_from_slice(&[0; 3]);
        for value in [1.0_f32, 0.5, 0.25] {
            binary.extend_from_slice(&value.to_le_bytes());
        }
        binary.push(1);
        binary.extend_from_slice(&[0; 3]);
        for value in [0.0_f32, 0.5, 1.0, 0.25] {
            binary.extend_from_slice(&value.to_le_bytes());
        }
        let root = json!({
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": 24},
                {"buffer": 0, "byteOffset": 24, "byteLength": 1},
                {"buffer": 0, "byteOffset": 28, "byteLength": 12},
                {"buffer": 0, "byteOffset": 40, "byteLength": 1},
                {"buffer": 0, "byteOffset": 44, "byteLength": 16}
            ],
            "accessors": [
                {
                    "bufferView": 0,
                    "componentType": 5126,
                    "count": 2,
                    "type": "VEC3",
                    "sparse": {
                        "count": 1,
                        "indices": {"bufferView": 1, "componentType": 5121},
                        "values": {"bufferView": 2}
                    }
                },
                {
                    "componentType": 5126,
                    "count": 2,
                    "type": "VEC4",
                    "sparse": {
                        "count": 1,
                        "indices": {"bufferView": 3, "componentType": 5121},
                        "values": {"bufferView": 4}
                    }
                }
            ]
        });
        let buffers = vec![binary];
        assert_eq!(
            super::read_color(&root, &buffers, 0)?,
            vec![[0.0, 0.25, 0.5, 1.0], [1.0, 0.5, 0.25, 1.0]]
        );
        assert_eq!(
            super::read_color(&root, &buffers, 1)?,
            vec![[0.0, 0.0, 0.0, 0.0], [0.0, 0.5, 1.0, 0.25]]
        );
        Ok(())
    }

    #[test]
    fn gltf_import_reports_each_unsupported_loss_class() {
        let losses = gltf_import_losses(&json!({
            "nodes": [{"mesh": 0, "weights": [1.0]}],
            "meshes": [{"weights": [0.0]}],
            "materials": [{"occlusionTexture": {"index": 0, "strength": 0.5}}],
            "extensionsUsed": ["KHR_draco_mesh_compression"]
        }));
        let features = losses
            .iter()
            .map(|loss| loss.feature_id.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            features,
            [
                "gltf.extension",
                "material.occlusion_strength",
                "mesh.shape_key_instance_weights"
            ]
            .into_iter()
            .collect()
        );
    }
    #[test]
    fn bone_parented_vrm_spring_keeps_hierarchy_and_world_position() -> TestResult {
        fn node_index(nodes: &[Value], key: &str, expected: &str) -> TestResult<usize> {
            nodes
                .iter()
                .position(|node| node["extras"]["potter"][key] == expected)
                .ok_or_else(|| format!("glTF node with {key}={expected} is missing").into())
        }

        let directory = tempfile::tempdir()?;
        let mut source = scene(&json!([
            {"op":"node.create","id":"arm","kind":"armature","transform":{"translation":[1.0,0.25,-0.5]}},
            {"op":"rig.generate_basic_human","target":{"id":"arm"},"scale":1.0},
            {"op":"bone.create","target":{"id":"arm"},"id":"hair_root","name":"Hair Root","parent":"head","head":[0.0,0.0,1.9],"tail":[0.0,0.0,2.0]},
            {"op":"node.create","id":"skin_mesh","kind":"box","params":{"size":1.0}},
            {"op":"vertex_group.create","target":{"id":"skin_mesh"},"id":"hair_group","name":"Hair Root"},
            {"op":"vertex_group.assign","target":{"id":"skin_mesh"},"group_id":"hair_group","weights":[
                {"vertex_id":0,"weight":1.0},{"vertex_id":1,"weight":1.0},{"vertex_id":2,"weight":1.0},{"vertex_id":3,"weight":1.0},
                {"vertex_id":4,"weight":1.0},{"vertex_id":5,"weight":1.0},{"vertex_id":6,"weight":1.0},{"vertex_id":7,"weight":1.0}
            ]},
            {"op":"modifier.create","target":{"id":"skin_mesh"},"id":"hair_skin","type":"armature","params":{"object":"arm","use_vertex_groups":true}},
            {"op":"node.create","id":"hair","kind":"box","params":{"size":0.25},"transform":{"translation":[0.2,0.25,0.4]}}
        ]))?;
        let arm_id = crate::model::Id::new("arm")?;
        let bone_id = crate::model::Id::new("hair_root")?;
        let hair_id = crate::model::Id::new("hair")?;
        let hair = source
            .nodes
            .get_mut(&hair_id)
            .ok_or("hair node is missing")?;
        hair.parent = Some(arm_id.clone());
        hair.parent_type = crate::model::ParentType::Bone;
        hair.parent_bone = Some(bone_id.clone());
        hair.parent_inverse = Some([
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.15, 0.0, 0.0, 1.0,
        ]);
        source.avatar = Some(serde_json::from_value(json!({
            "armature": "arm",
            "humanoid": {
                "hips": "hips", "spine": "spine", "head": "head",
                "leftUpperArm": "upper_arm_l", "leftLowerArm": "forearm_l", "leftHand": "hand_l",
                "rightUpperArm": "upper_arm_r", "rightLowerArm": "forearm_r", "rightHand": "hand_r",
                "leftUpperLeg": "thigh_l", "leftLowerLeg": "shin_l", "leftFoot": "foot_l",
                "rightUpperLeg": "thigh_r", "rightLowerLeg": "shin_r", "rightFoot": "foot_r"
            },
            "meta": {
                "name": "Bone Parent Test",
                "authors": ["Potter"],
                "license_url": "https://vrm.dev/licenses/1.0/",
                "avatar_permission": "everyone",
                "commercial_usage": "personal_non_profit",
                "credit_notation": "required",
                "modification": "allow_modification"
            },
            "spring_bone": {
                "springs": [{
                    "joints": [
                        { "node": { "node": "arm", "bone": "hair_root" } },
                        { "node": { "node": "hair" } }
                    ]
                }]
            }
        }))?);
        let snapshot = Snapshot::evaluate(&source, &EvaluationContext::default())?;
        let frames = super::vrm_frame_maps(
            &source,
            &snapshot,
            source
                .avatar
                .as_ref()
                .ok_or("avatar settings are missing")?,
            crate::model::AvatarVersion::V1,
        )?;
        let v0_frames = super::vrm_frame_maps(
            &source,
            &snapshot,
            source
                .avatar
                .as_ref()
                .ok_or("avatar settings are missing")?,
            crate::model::AvatarVersion::V0,
        )?;
        let v0_basis = super::avatar_basis(crate::model::AvatarVersion::V0);
        let v0_source_world = glam::DMat4::from_cols_array(&snapshot.nodes[&hair_id].world_matrix);
        assert!(super::matrices_close(
            v0_frames.node_world_matrices[&hair_id],
            v0_basis * v0_source_world * v0_basis.inverse()
        ));
        let native_glb = super::export_vrm_with_frames(
            &source,
            &snapshot,
            directory.path(),
            crate::model::AvatarVersion::V1,
            &frames,
        )?;
        let (native_json, _) = super::split_glb(&native_glb)?;
        let native_root: Value = serde_json::from_slice(native_json)?;
        let native_nodes = native_root["nodes"]
            .as_array()
            .ok_or("native VRM nodes are missing")?;
        let native_bone_index = node_index(native_nodes, "bone_id", "hair_root")?;
        let native_hair_index = node_index(native_nodes, "id", "hair")?;
        let native_parents = super::node_parent_indices(native_nodes)?;
        assert_eq!(native_parents[native_hair_index], Some(native_bone_index));
        assert!(
            native_nodes[native_bone_index]["children"]
                .as_array()
                .is_some_and(|children| children.contains(&json!(native_hair_index)))
        );
        let native_joints = native_root["extensions"]["VRMC_springBone"]["springs"][0]["joints"]
            .as_array()
            .ok_or("VRM 1.0 spring joints are missing")?;
        assert_eq!(
            native_joints[0]["node"].as_u64(),
            Some(native_bone_index as u64)
        );
        assert_eq!(
            native_joints[1]["node"].as_u64(),
            Some(native_hair_index as u64)
        );
        let native_worlds = super::node_world_matrices(native_nodes, &native_parents)?;
        let source_world = glam::DMat4::from_cols_array(&snapshot.nodes[&hair_id].world_matrix);
        let basis = super::avatar_basis(crate::model::AvatarVersion::V1);
        assert!(super::matrices_close(
            native_worlds[native_hair_index],
            basis * source_world * basis.inverse()
        ));

        let mut ordinary = source.clone();
        ordinary.avatar = None;
        let ordinary_snapshot = Snapshot::evaluate(&ordinary, &EvaluationContext::default())?;
        let (ordinary_json, ordinary_binary) = export_with_root(
            &ordinary,
            &ordinary_snapshot,
            directory.path(),
            Some("ordinary.bin"),
        )?;
        let ordinary_root: Value = serde_json::from_slice(&ordinary_json)?;
        let ordinary_nodes = ordinary_root["nodes"]
            .as_array()
            .ok_or("ordinary glTF nodes are missing")?;
        let ordinary_bone_index = node_index(ordinary_nodes, "bone_id", "hair_root")?;
        let ordinary_hair_index = node_index(ordinary_nodes, "id", "hair")?;
        let ordinary_parents = super::node_parent_indices(ordinary_nodes)?;
        assert_eq!(
            ordinary_parents[ordinary_hair_index],
            Some(ordinary_bone_index)
        );
        let ordinary_skin_mesh_index = node_index(ordinary_nodes, "id", "skin_mesh")?;
        assert!(ordinary_nodes[ordinary_skin_mesh_index]["skin"].is_number());
        let skin_index = ordinary_nodes[ordinary_skin_mesh_index]["skin"]
            .as_u64()
            .ok_or("weighted mesh skin is missing")? as usize;
        assert!(
            ordinary_root["skins"][skin_index]["joints"]
                .as_array()
                .is_some_and(|joints| joints.contains(&json!(ordinary_bone_index)))
        );
        let ordinary_worlds = super::node_world_matrices(ordinary_nodes, &ordinary_parents)?;
        assert!(super::matrices_close(
            ordinary_worlds[ordinary_hair_index],
            super::POT_TO_GLTF
                * glam::DMat4::from_cols_array(&ordinary_snapshot.nodes[&hair_id].world_matrix)
                * super::POT_TO_GLTF.inverse()
        ));
        let mut unskinned = ordinary.clone();
        let skin_mesh_id = crate::model::Id::new("skin_mesh")?;
        unskinned
            .nodes
            .remove(&skin_mesh_id)
            .ok_or("skin mesh node is missing")?;
        let unskinned_snapshot = Snapshot::evaluate(&unskinned, &EvaluationContext::default())?;
        let (unskinned_json, _) =
            export_with_root(&unskinned, &unskinned_snapshot, directory.path(), None)?;
        let unskinned_root: Value = serde_json::from_slice(&unskinned_json)?;
        let unskinned_nodes = unskinned_root["nodes"]
            .as_array()
            .ok_or("unskinned glTF nodes are missing")?;
        let unskinned_bone_index = node_index(unskinned_nodes, "bone_id", "hair_root")?;
        let unskinned_hair_index = node_index(unskinned_nodes, "id", "hair")?;
        assert_eq!(
            super::node_parent_indices(unskinned_nodes)?[unskinned_hair_index],
            Some(unskinned_bone_index)
        );
        let ordinary_path = directory.path().join("ordinary.gltf");
        std::fs::write(&ordinary_path, ordinary_json)?;
        std::fs::write(directory.path().join("ordinary.bin"), ordinary_binary)?;
        let imported = super::import(&ordinary_path, uuid::Uuid::new_v4().to_string())?;
        let imported_hair = imported
            .doc
            .nodes
            .get(&hair_id)
            .ok_or("imported hair node is missing")?;
        assert_eq!(imported_hair.parent.as_ref(), Some(&arm_id));
        assert_eq!(imported_hair.parent_type, crate::model::ParentType::Bone);
        assert_eq!(imported_hair.parent_bone.as_ref(), Some(&bone_id));
        let imported_snapshot = Snapshot::evaluate(&imported.doc, &EvaluationContext::default())?;
        let imported_world =
            glam::DMat4::from_cols_array(&imported_snapshot.nodes[&hair_id].world_matrix);
        let source_position =
            glam::DMat4::from_cols_array(&ordinary_snapshot.nodes[&hair_id].world_matrix)
                .w_axis
                .truncate();
        assert!((imported_world.w_axis.truncate() - source_position).length() < 1.0e-5);
        Ok(())
    }
    fn export_root_and_binary(doc: &SceneDoc) -> TestResult<(Value, Vec<u8>)> {
        let directory = tempfile::tempdir()?;
        let snapshot = Snapshot::evaluate(doc, &EvaluationContext::default())?;
        let (document, binary) =
            export_with_root(doc, &snapshot, directory.path(), Some("scene.bin"))?;
        Ok((serde_json::from_slice(&document)?, binary))
    }

    fn export_root_and_binary_at_root(
        doc: &SceneDoc,
        root: &std::path::Path,
    ) -> TestResult<(Value, Vec<u8>)> {
        let snapshot = Snapshot::evaluate(doc, &EvaluationContext::default())?;
        let (document, binary) = export_with_root(doc, &snapshot, root, Some("scene.bin"))?;
        Ok((serde_json::from_slice(&document)?, binary))
    }

    fn accessor_vec3(root: &Value, binary: &[u8], accessor: &Value) -> TestResult<Vec<[f64; 3]>> {
        let index = usize::try_from(accessor.as_u64().ok_or("accessor index is missing")?)?;
        Ok(super::read_vec3(root, &[binary.to_vec()], index)?)
    }

    #[test]
    fn shade_smooth_exports_averaged_normals_and_flat_faces_keep_face_normals() -> TestResult {
        let smooth = scene(&json!([
            {"op":"node.create","id":"cube","kind":"box","params":{"size":2.0}},
            {"op":"mesh.shade_smooth","target":{"id":"cube"},"elements":{"domain":"face","ids":["f0","f1","f2","f3","f4","f5"]}}
        ]))?;
        let (root, binary) = export_root_and_binary(&smooth)?;
        let primitive = &root["meshes"][0]["primitives"][0];
        let diagonal = 3.0_f64.sqrt().recip();
        for normal in accessor_vec3(&root, &binary, &primitive["attributes"]["NORMAL"])? {
            for component in normal {
                assert!(
                    (component.abs() - diagonal).abs() < 1.0e-6,
                    "smooth cube corner normal {normal:?} is not the vertex diagonal"
                );
            }
        }
        let flat = scene(&json!([
            {"op":"node.create","id":"cube","kind":"box","params":{"size":2.0}}
        ]))?;
        let (root, binary) = export_root_and_binary(&flat)?;
        let primitive = &root["meshes"][0]["primitives"][0];
        let flat_normals = accessor_vec3(&root, &binary, &primitive["attributes"]["NORMAL"])?;
        assert_eq!(flat_normals.len(), 24, "flat cube splits each face corner");
        for normal in flat_normals {
            let axis_aligned = normal
                .iter()
                .filter(|component| (component.abs() - 1.0).abs() < 1.0e-6)
                .count();
            assert_eq!(
                axis_aligned, 1,
                "flat cube normal {normal:?} is not a face normal"
            );
        }
        Ok(())
    }

    #[test]
    fn face_material_slots_emit_one_primitive_per_slot_sharing_attributes() -> TestResult {
        let mut doc = scene(&json!([
            {"op":"material.create","id":"jacket"},
            {"op":"material.create","id":"skin"},
            {"op":"node.create","id":"body","kind":"box","params":{"size":1.0}},
            {"op":"node.update","target":{"id":"body"},"set":{"materials":["jacket","skin"]}},
            {"op":"mesh.set_attribute","target":{"id":"body"},"elements":{"domain":"face","ids":["f1"]},"material_index":1}
        ]))?;
        let (root, _) = export_root_and_binary(&doc)?;
        let primitives = root["meshes"][0]["primitives"]
            .as_array()
            .ok_or("mesh primitives are missing")?;
        assert_eq!(primitives.len(), 2, "one primitive per used material slot");
        for attribute in ["POSITION", "NORMAL"] {
            assert_eq!(
                primitives[0]["attributes"][attribute], primitives[1]["attributes"][attribute],
                "{attribute} accessor is shared across material primitives"
            );
        }
        assert_ne!(primitives[0]["indices"], primitives[1]["indices"]);
        let material_id = |primitive: &Value| -> TestResult<String> {
            let index = usize::try_from(
                primitive["material"]
                    .as_u64()
                    .ok_or("material is missing")?,
            )?;
            Ok(root["materials"][index]["extras"]["potter"]["id"]
                .as_str()
                .ok_or("material ID is missing")?
                .to_owned())
        };
        assert_eq!(material_id(&primitives[0])?, "jacket");
        assert_eq!(material_id(&primitives[1])?, "skin");
        let index_count = |primitive: &Value| -> TestResult<u64> {
            let accessor =
                usize::try_from(primitive["indices"].as_u64().ok_or("indices are missing")?)?;
            Ok(root["accessors"][accessor]["count"]
                .as_u64()
                .ok_or("index count is missing")?)
        };
        assert_eq!(
            index_count(&primitives[0])?,
            30,
            "five jacket faces, two triangles each"
        );
        assert_eq!(
            index_count(&primitives[1])?,
            6,
            "one skin face, two triangles"
        );
        let node_id = crate::model::Id::new("body")?;
        let data_id = doc.nodes[&node_id]
            .data
            .clone()
            .ok_or("body mesh data is missing")?;
        doc.data_blocks
            .get_mut(&data_id)
            .and_then(|data| data.mesh.as_mut())
            .ok_or("body mesh is missing")?
            .faces[0]
            .material_index = 2;
        let directory = tempfile::tempdir()?;
        let (clamped_root, _) = export_root_and_binary_at_root(&doc, directory.path())?;
        let clamped_primitives = clamped_root["meshes"][0]["primitives"]
            .as_array()
            .ok_or("clamped mesh primitives are missing")?;
        assert_eq!(clamped_primitives.len(), 3);
        let clamped_material_ids = clamped_primitives
            .iter()
            .map(|primitive| -> TestResult<Option<String>> {
                let Some(index) = primitive["material"].as_u64() else {
                    return Ok(None);
                };
                let index = usize::try_from(index)?;
                Ok(Some(
                    clamped_root["materials"][index]["extras"]["potter"]["id"]
                        .as_str()
                        .ok_or("clamped material ID is missing")?
                        .to_owned(),
                ))
            })
            .collect::<TestResult<Vec<_>>>()?;
        assert_eq!(
            clamped_material_ids,
            [
                Some("jacket".to_owned()),
                Some("skin".to_owned()),
                Some("skin".to_owned())
            ]
        );
        assert_ne!(
            clamped_primitives[0]["indices"],
            clamped_primitives[2]["indices"]
        );
        Ok(())
    }

    #[test]
    fn baked_scale_normals_use_inverse_transpose() {
        let normal = glam::DVec3::new(1.0, 1.0, 0.0).normalize();
        let stretched = super::output_normals(
            &[normal],
            glam::DVec3::new(2.0, 1.0, 1.0),
            glam::DMat4::IDENTITY,
        )[0];
        let expected = glam::DVec3::new(0.5, 1.0, 0.0).normalize();
        assert!((stretched - expected).length() < 1.0e-12, "{stretched:?}");
        let mirrored = super::output_normals(
            &[glam::DVec3::Y],
            glam::DVec3::new(-1.0, 1.0, 1.0),
            glam::DMat4::IDENTITY,
        )[0];
        assert!(
            (mirrored - glam::DVec3::Y).length() < 1.0e-12,
            "{mirrored:?}"
        );
    }

    #[test]
    fn export_import_shading_roundtrip_has_no_spurious_custom_normals() -> TestResult {
        let smooth = scene(&json!([
            {"op":"node.create","id":"cube","kind":"box","params":{"size":2.0}},
            {"op":"mesh.shade_smooth","target":{"id":"cube"},"elements":{"domain":"face","ids":["f0","f1","f2","f3","f4","f5"]}}
        ]))?;
        let directory = tempfile::tempdir()?;
        let (imported, _) = round_trip_in(&smooth, directory.path())?;
        let mesh = imported
            .doc
            .data_blocks
            .values()
            .find_map(|block| block.mesh.as_ref())
            .ok_or("imported mesh is missing")?;
        assert!(!mesh.attributes.contains_key("custom_normal"));
        let smooth_faces = mesh.attributes["shade_smooth"]["values"]
            .as_object()
            .ok_or("imported shade_smooth is missing")?
            .values()
            .filter(|value| value.as_bool() == Some(true))
            .count();
        // Potter polygons keep the cube's quads, so every imported face is still a smooth quad.
        assert_eq!(smooth_faces, mesh.faces.len());
        let flat = scene(&json!([
            {"op":"node.create","id":"cube","kind":"box","params":{"size":2.0}}
        ]))?;
        let (imported, _) = round_trip_in(&flat, directory.path())?;
        let mesh = imported
            .doc
            .data_blocks
            .values()
            .find_map(|block| block.mesh.as_ref())
            .ok_or("imported mesh is missing")?;
        assert!(!mesh.attributes.contains_key("custom_normal"));
        assert!(!mesh.attributes.contains_key("shade_smooth"));
        Ok(())
    }

    #[test]
    fn morph_normal_deltas_are_exported_and_roundtrip_without_loss() -> TestResult {
        // Potter polygons keep the quads, so morph normals derived on them agree after the roundtrip.
        let source = scene(&json!([
            {"op":"node.create","id":"cube","kind":"box","params":{"size":2.0}},
            {"op":"shape_key.create","target":{"id":"cube"},"id":"lift","name":"Lift","positions":{"0":[0.0,0.0,1.0]}}
        ]))?;
        let (root, binary) = export_root_and_binary(&source)?;
        let target = &root["meshes"][0]["primitives"][0]["targets"][0];
        let deltas = accessor_vec3(&root, &binary, &target["NORMAL"])?;
        assert!(
            deltas
                .iter()
                .any(|delta| delta.iter().any(|value| value.abs() > 1.0e-3)),
            "morph NORMAL deltas must move with the shape key"
        );
        let directory = tempfile::tempdir()?;
        let (imported, _) = round_trip_in(&source, directory.path())?;
        assert!(
            !imported
                .losses
                .iter()
                .any(|loss| loss.feature_id == "mesh.morph_normals"),
            "derived morph normals must agree with imported shading: {:?}",
            imported.losses
        );
        Ok(())
    }

    fn apply_operations(doc: &SceneDoc, operations: &Value) -> TestResult<SceneDoc> {
        let applied = apply_batch(
            doc,
            &json!({
                "schema_version": 1,
                "base_revision": doc.revision,
                "operations": operations
            }),
        )?;
        Ok(applied.doc)
    }

    /// Box skinned to one bone with a shape key. `multi_material` moves face f1 to a second slot.
    fn rigged_box_scene(multi_material: bool) -> TestResult<SceneDoc> {
        let weights: Vec<Value> = (0..8)
            .map(|vertex_id| json!({"vertex_id": vertex_id, "weight": 1.0}))
            .collect();
        let mut operations = vec![
            json!({"op":"material.create","id":"jacket"}),
            json!({"op":"material.create","id":"skin"}),
            json!({"op":"node.create","id":"body","kind":"box","params":{"size":2.0}}),
            json!({"op":"node.create","id":"rig","name":"Rig","kind":"armature"}),
            json!({"op":"bone.create","target":{"id":"rig"},"id":"root","name":"Root","head":[0.0,0.0,0.0],"tail":[0.0,0.0,1.0]}),
            json!({"op":"vertex_group.create","target":{"id":"body"},"id":"root_group","name":"Root"}),
            json!({"op":"vertex_group.assign","target":{"id":"body"},"group_id":"root_group","weights":weights}),
            json!({"op":"modifier.create","target":{"id":"body"},"id":"armature","type":"armature","params":{"object":"rig","use_vertex_groups":true}}),
            json!({"op":"shape_key.create","target":{"id":"body"},"id":"lift","name":"Lift","positions":{"0":[-1.0,-1.0,1.5]}}),
        ];
        if multi_material {
            operations.push(json!({"op":"node.update","target":{"id":"body"},"set":{"materials":["jacket","skin"]}}));
            operations.push(json!({"op":"mesh.set_attribute","target":{"id":"body"},"elements":{"domain":"face","ids":["f1"]},"material_index":1}));
        }
        scene(&Value::Array(operations))
    }

    fn import_without_potter_metadata(doc: &SceneDoc) -> TestResult<ImportedGraph> {
        let directory = tempfile::tempdir()?;
        let snapshot = Snapshot::evaluate(doc, &EvaluationContext::default())?;
        let (document, binary) =
            export_with_root(doc, &snapshot, directory.path(), Some("scene.bin"))?;
        let mut root: Value = serde_json::from_slice(&document)?;
        for mesh in root["meshes"].as_array_mut().ok_or("meshes are missing")? {
            if let Some(extras) = mesh["extras"].as_object_mut() {
                extras.remove("potter");
            }
        }
        let path = directory.path().join("third_party.gltf");
        std::fs::write(&path, serde_json::to_vec(&root)?)?;
        std::fs::write(directory.path().join("scene.bin"), binary)?;
        Ok(import(&path, uuid::Uuid::new_v4().to_string())?)
    }

    #[test]
    fn multi_material_skinned_shape_key_mesh_reimports_with_single_material_vertices() -> TestResult
    {
        let directory = tempfile::tempdir()?;
        let (single, _) = round_trip_in(&rigged_box_scene(false)?, directory.path())?;
        let (multi, _) = round_trip_in(&rigged_box_scene(true)?, directory.path())?;
        let summary = |imported: &ImportedGraph| -> TestResult<(usize, usize)> {
            let block = imported
                .doc
                .data_blocks
                .values()
                .find(|block| block.mesh.is_some())
                .ok_or("imported mesh is missing")?;
            let vertices = block.mesh.as_ref().map_or(0, |mesh| mesh.vertices.len());
            let keys = block.shape_keys.as_ref().map_or(0, |keys| keys.keys.len());
            Ok((vertices, keys))
        };
        let single_summary = summary(&single)?;
        assert_eq!(single_summary.1, 1, "shape key survives the round trip");
        assert_eq!(
            summary(&multi)?,
            single_summary,
            "material primitives must share one vertex set"
        );
        Ok(())
    }

    #[test]
    fn shape_key_collapsing_a_face_exports_with_zero_morph_normal_deltas() -> TestResult {
        let base =
            scene(&json!([{"op":"node.create","id":"body","kind":"plane","params":{"size":2.0}}]))?;
        // Every plane vertex moves to the origin, so the face has no area in the morph layer.
        let doc = apply_operations(
            &base,
            &json!([{"op":"shape_key.create","target":{"id":"body"},"id":"blink","name":"Blink","positions":{"0":[0.0,0.0,0.0],"1":[0.0,0.0,0.0],"2":[0.0,0.0,0.0],"3":[0.0,0.0,0.0]}}]),
        )?;
        let (root, binary) = export_root_and_binary(&doc)?;
        let target = &root["meshes"][0]["primitives"][0]["targets"][0];
        let deltas = accessor_vec3(&root, &binary, &target["NORMAL"])?;
        assert!(
            deltas
                .iter()
                .all(|delta| delta.iter().all(|value| value.abs() < 1.0e-6)),
            "collapsed corners keep the base normal: {deltas:?}"
        );
        Ok(())
    }

    #[test]
    fn collection_instance_exports_one_primitive_per_member_material() -> TestResult {
        // Members keep their own slot lists, as Blender's flattened instance nodes do: box_a
        // contributes jacket (5 faces) and skin (1 face), box_b its skin (6 faces).
        let doc = scene(&json!([
            {"op":"material.create","id":"jacket"},
            {"op":"material.create","id":"skin"},
            {"op":"collection.create","id":"source"},
            {"op":"node.create","id":"box_a","kind":"box","collection":"source","params":{"size":1.0}},
            {"op":"node.update","target":{"id":"box_a"},"set":{"materials":["jacket","skin"]}},
            {"op":"mesh.set_attribute","target":{"id":"box_a"},"elements":{"domain":"face","ids":["f1"]},"material_index":1},
            {"op":"node.create","id":"box_b","kind":"box","collection":"source","params":{"size":1.0},"transform":{"translation":[2.0,0.0,0.0]}},
            {"op":"node.update","target":{"id":"box_b"},"set":{"materials":["skin"]}},
            {"op":"collection.instance_create","id":"instance","collection":"source","transform":{"translation":[4.0,0.0,0.0]}}
        ]))?;
        let (root, _) = export_root_and_binary(&doc)?;
        let instance = root["meshes"]
            .as_array()
            .ok_or("meshes are missing")?
            .iter()
            .find(|mesh| mesh["name"] == "instance_source")
            .ok_or("instance mesh is missing")?;
        let primitives = instance["primitives"]
            .as_array()
            .ok_or("primitives are missing")?;
        let materials: Vec<Value> = primitives
            .iter()
            .map(|primitive| primitive["material"].clone())
            .collect();
        assert_eq!(materials, [json!(0), json!(1), json!(1)]);
        let index_counts: Vec<u64> = primitives
            .iter()
            .map(|primitive| {
                let accessor = primitive["indices"].as_u64().unwrap_or(u64::MAX);
                root["accessors"][usize::try_from(accessor).unwrap_or(usize::MAX)]["count"]
                    .as_u64()
                    .unwrap_or(0)
            })
            .collect();
        assert_eq!(index_counts, [30, 6, 36]);
        Ok(())
    }

    #[test]
    fn collection_instance_keeps_shading_from_every_source() -> TestResult {
        // The first source is flat and the second smooth: the smooth normals must survive instancing.
        let doc = scene(&json!([
            {"op":"collection.create","id":"source"},
            {"op":"node.create","id":"flat","kind":"box","collection":"source","params":{"size":1.0}},
            {"op":"node.create","id":"smooth","kind":"box","collection":"source","params":{"size":1.0},"transform":{"translation":[2.0,0.0,0.0]}},
            {"op":"mesh.shade_smooth","target":{"id":"smooth"},"elements":{"domain":"face","ids":["f0","f1","f2","f3","f4","f5"]}},
            {"op":"collection.instance_create","id":"instance","collection":"source","transform":{"translation":[10.0,0.0,0.0]}}
        ]))?;
        let (root, binary) = export_root_and_binary(&doc)?;
        let instance = root["meshes"]
            .as_array()
            .ok_or("meshes are missing")?
            .iter()
            .find(|mesh| mesh["name"] == "instance_source")
            .ok_or("instance mesh is missing")?;
        let normals = accessor_vec3(
            &root,
            &binary,
            &instance["primitives"][0]["attributes"]["NORMAL"],
        )?;
        let axis_aligned =
            |normal: &[f64; 3]| normal.iter().filter(|value| value.abs() > 1.0e-6).count() == 1;
        assert!(
            normals.iter().any(|normal| !axis_aligned(normal)),
            "the smooth source keeps diagonal normals inside the instance"
        );
        assert!(
            normals.iter().any(axis_aligned),
            "the flat source keeps axis-aligned normals"
        );
        Ok(())
    }

    #[test]
    fn set_attribute_sharp_writes_canonical_attribute_that_exports() -> TestResult {
        let base = scene(&json!([
            {"op":"node.create","id":"cube","kind":"box","params":{"size":2.0}},
            {"op":"mesh.shade_smooth","target":{"id":"cube"},"elements":{"domain":"face","ids":["f0","f1","f2","f3","f4","f5"]}}
        ]))?;
        let node_id = crate::model::Id::new("cube")?;
        let data_id = base.nodes[&node_id]
            .data
            .clone()
            .ok_or("cube mesh data is missing")?;
        let edge_ids: Vec<String> = base.data_blocks[&data_id]
            .mesh
            .as_ref()
            .ok_or("cube mesh is missing")?
            .edges
            .iter()
            .map(|edge| format!("e{}", edge.id))
            .collect();
        let doc = apply_operations(
            &base,
            &json!([{"op":"mesh.set_attribute","target":{"id":"cube"},"elements":{"domain":"edge","ids":edge_ids},"name":"sharp","value":true}]),
        )?;
        let mesh = doc.data_blocks[&data_id]
            .mesh
            .as_ref()
            .ok_or("cube mesh is missing")?;
        assert_eq!(mesh.attributes["sharp"]["domain"], json!("edge"));
        assert_eq!(mesh.attributes["sharp"]["type"], json!("bool"));
        let (root, binary) = export_root_and_binary(&doc)?;
        let normals = accessor_vec3(
            &root,
            &binary,
            &root["meshes"][0]["primitives"][0]["attributes"]["NORMAL"],
        )?;
        assert!(
            normals
                .iter()
                .all(|normal| normal.iter().filter(|value| value.abs() > 1.0e-6).count() == 1),
            "every edge sharp: corner normals are face normals"
        );
        Ok(())
    }

    #[test]
    fn potter_smooth_seamed_shape_key_mesh_reports_no_morph_normal_loss() -> TestResult {
        // Potter polygons keep the quads the morph normals were derived on; UV seams rejoin the box.
        let doc = scene(&json!([
            {"op":"node.create","id":"body","kind":"box","params":{"size":2.0}},
            {"op":"mesh.shade_smooth","target":{"id":"body"},"elements":{"domain":"face","ids":["f0","f1","f2","f3","f4","f5"]}},
            {"op":"uv.unwrap","target":{"id":"body"},"method":"smart"},
            {"op":"shape_key.create","target":{"id":"body"},"id":"lift","name":"Lift","positions":{"0":[-1.0,-1.0,1.5]}}
        ]))?;
        let directory = tempfile::tempdir()?;
        let (imported, _) = round_trip_in(&doc, directory.path())?;
        assert!(
            !imported
                .losses
                .iter()
                .any(|loss| loss.feature_id == "mesh.morph_normals"),
            "Potter-derived morph normals are not a loss: {:?}",
            imported.losses
        );
        let mesh = first_mesh(&imported.doc)?;
        assert_eq!(
            mesh.vertices.len(),
            8,
            "UV seams must rejoin the source vertices"
        );
        assert!(
            mesh.faces.iter().all(|face| face.vertices.len() == 4),
            "quads must survive the roundtrip"
        );
        assert!(
            !mesh.attributes.contains_key("custom_normal"),
            "UV seams must not create custom normals"
        );
        // Without Potter polygons the quads arrive as triangles, whose morph normals disagree.
        let third_party = import_without_potter_metadata(&doc)?;
        assert!(
            third_party
                .losses
                .iter()
                .any(|loss| loss.feature_id == "mesh.morph_normals"),
            "third-party morph normals are still compared: {:?}",
            third_party.losses
        );
        Ok(())
    }

    #[test]
    fn potter_polygons_roundtrip_restores_face_order_corners_materials_and_uvs() -> TestResult {
        // Pentagonal caps, quad sides, and one triangle over two material slots.
        let doc = scene(&json!([
            {"op":"material.create","id":"jacket"},
            {"op":"material.create","id":"skin"},
            {"op":"node.create","id":"body","kind":"cylinder","params":{"vertices":5,"radius":1.0,"depth":2.0}},
            {"op":"node.update","target":{"id":"body"},"set":{"materials":["jacket","skin"]}},
            {"op":"mesh.triangulate","target":{"id":"body"},"elements":{"domain":"face","ids":["f2"]}},
            {"op":"mesh.set_attribute","target":{"id":"body"},"elements":{"domain":"face","ids":["f1","f4"]},"material_index":1},
            {"op":"uv.unwrap","target":{"id":"body"},"method":"smart"}
        ]))?;
        let directory = tempfile::tempdir()?;
        let (imported, _) = round_trip_in(&doc, directory.path())?;
        let source = first_mesh(&doc)?;
        let mesh = first_mesh(&imported.doc)?;
        let sizes = |mesh: &Mesh| {
            mesh.faces
                .iter()
                .map(|face| face.vertices.len())
                .collect::<Vec<_>>()
        };
        assert_eq!(sizes(mesh), sizes(source));
        assert!(
            [3, 4, 5].iter().all(|size| sizes(mesh).contains(size)),
            "{:?}",
            sizes(mesh)
        );
        let materials = |mesh: &Mesh| {
            mesh.faces
                .iter()
                .map(|face| face.material_index)
                .collect::<Vec<_>>()
        };
        assert_eq!(materials(mesh), materials(source));
        assert_close("positions", &face_points(mesh), &face_points(source));
        assert_close("uvs", &face_uvs(mesh)?, &face_uvs(source)?);
        Ok(())
    }

    #[test]
    fn potter_shading_flags_roundtrip_exactly() -> TestResult {
        let base = scene(&json!([
            {"op":"node.create","id":"cube","kind":"box","params":{"size":2.0}},
            {"op":"mesh.shade_smooth","target":{"id":"cube"},"elements":{"domain":"face","ids":["f0"]}}
        ]))?;
        let edge_ids = first_mesh(&base)?
            .edges
            .iter()
            .take(3)
            .map(|edge| format!("e{}", edge.id))
            .collect::<Vec<_>>();
        let doc = apply_operations(
            &base,
            &json!([{"op":"mesh.set_attribute","target":{"id":"cube"},"elements":{"domain":"edge","ids":edge_ids},"name":"sharp","value":true}]),
        )?;
        let directory = tempfile::tempdir()?;
        let (imported, _) = round_trip_in(&doc, directory.path())?;
        let source = first_mesh(&doc)?;
        let mesh = first_mesh(&imported.doc)?;
        assert!(!sharp_edge_points(source)?.is_empty());
        assert_eq!(smooth_face_flags(mesh)?, smooth_face_flags(source)?);
        assert_eq!(sharp_edge_points(mesh)?, sharp_edge_points(source)?);
        assert!(
            !mesh.attributes.contains_key("custom_normal"),
            "exact flags must derive no custom normals"
        );
        Ok(())
    }

    #[test]
    fn malformed_potter_polygons_are_rejected() -> TestResult {
        let doc = scene(&json!([
            {"op":"node.create","id":"cube","kind":"box","params":{"size":2.0}},
            {"op":"mesh.shade_smooth","target":{"id":"cube"},"elements":{"domain":"face","ids":["f0"]}}
        ]))?;
        let cases: [(&str, Edit); 7] = [
            ("field bufferView out of range", |root, _| {
                root["meshes"][0]["extras"]["potter"]["polygons"]["corners"] = json!(999);
                Ok(())
            }),
            ("field length that does not match the faces", |root, _| {
                let smooth = root["meshes"][0]["extras"]["potter"]["polygons"]["smooth"].clone();
                root["meshes"][0]["extras"]["potter"]["polygons"]["corners"] = smooth;
                Ok(())
            }),
            ("corner beyond the vertex list", |root, binary| {
                let offset = polygon_field_offset(root, "corners")?;
                binary[offset] = 200;
                Ok(())
            }),
            ("repeated corner in one face", |root, binary| {
                let offset = polygon_field_offset(root, "corners")?;
                binary[offset + 1] = binary[offset];
                Ok(())
            }),
            ("primitive beyond the primitive list", |root, binary| {
                let offset = polygon_field_offset(root, "primitives")?;
                binary[offset] = 3;
                Ok(())
            }),
            ("face with fewer than three corners", |root, binary| {
                let offset = polygon_field_offset(root, "sizes")?;
                binary[offset] = 2;
                Ok(())
            }),
            ("smooth flag that is not a bool", |root, binary| {
                let offset = polygon_field_offset(root, "smooth")?;
                binary[offset] = 2;
                Ok(())
            }),
        ];
        for (name, edit) in cases {
            let error = import_edited(&doc, edit)?
                .err()
                .ok_or(format!("{name} must be rejected"))?;
            assert_eq!(
                error.code,
                crate::error::ErrorCode::ImportFailed,
                "{name}: {error}"
            );
        }
        Ok(())
    }

    #[test]
    fn unknown_potter_polygon_version_imports_triangles_with_loss() -> TestResult {
        let doc = scene(&json!([
            {"op":"node.create","id":"cube","kind":"box","params":{"size":2.0}}
        ]))?;
        let imported = import_edited(&doc, |root, _| {
            root["meshes"][0]["extras"]["potter"]["polygons"]["version"] = json!(2);
            Ok(())
        })??;
        assert!(
            imported
                .losses
                .iter()
                .any(|loss| loss.feature_id == "mesh.polygon_topology"),
            "unknown polygon versions are reported: {:?}",
            imported.losses
        );
        let mesh = first_mesh(&imported.doc)?;
        assert_eq!(mesh.faces.len(), 12, "the cube imports as its triangles");
        assert!(mesh.faces.iter().all(|face| face.vertices.len() == 3));
        Ok(())
    }

    #[test]
    fn stale_potter_polygons_from_another_edit_fall_back_to_triangles_with_loss() -> TestResult {
        // Another tool moved one corner of a face and kept the extras: the stored polygon no longer
        // matches the triangles that cover it, so the mesh imports as triangles and reports a loss.
        let doc = scene(&json!([
            {"op":"node.create","id":"cube","kind":"box","params":{"size":2.0}}
        ]))?;
        let imported = import_edited(&doc, |root, binary| {
            let vertex_count = root["accessors"][usize::try_from(
                root["meshes"][0]["primitives"][0]["attributes"]["POSITION"]
                    .as_u64()
                    .ok_or("POSITION is missing")?,
            )?]["count"]
                .as_u64()
                .ok_or("POSITION count is missing")?;
            let offset = polygon_field_offset(root, "corners")?;
            let face = binary
                .get(offset..offset + 4)
                .ok_or("cube face corners are missing")?
                .to_vec();
            let unused = (0..vertex_count)
                .find(|index| !face.iter().any(|corner| u64::from(*corner) == *index))
                .ok_or("every vertex is on the face")?;
            binary[offset + 3] = u8::try_from(unused)?;
            Ok(())
        })??;
        assert!(
            imported
                .losses
                .iter()
                .any(|loss| loss.feature_id == "mesh.polygon_topology"),
            "a record that no longer matches its triangles is reported: {:?}",
            imported.losses
        );
        let mesh = first_mesh(&imported.doc)?;
        assert_eq!(mesh.faces.len(), 12, "the cube imports as its triangles");
        Ok(())
    }

    #[test]
    fn smooth_two_material_box_without_uvs_restores_faces_across_shared_vertices() -> TestResult {
        let doc = scene(&json!([
            {"op":"material.create","id":"jacket"},
            {"op":"material.create","id":"skin"},
            {"op":"node.create","id":"box","kind":"box","params":{"size":2.0}},
            {"op":"node.update","target":{"id":"box"},"set":{"materials":["jacket","skin"]}},
            {"op":"mesh.shade_smooth","target":{"id":"box"},"elements":{"domain":"face","ids":["f0","f1","f2","f3","f4","f5"]}},
            {"op":"mesh.set_attribute","target":{"id":"box"},"elements":{"domain":"face","ids":["f1","f4"]},"material_index":1}
        ]))?;
        let directory = tempfile::tempdir()?;
        let (imported, _) = round_trip_in(&doc, directory.path())?;
        let source = first_mesh(&doc)?;
        let mesh = first_mesh(&imported.doc)?;
        assert_eq!(
            mesh.vertices.len(),
            8,
            "smooth faces share the eight box corners"
        );
        assert_eq!(face_materials(mesh), face_materials(source));
        assert_close("positions", &face_points(mesh), &face_points(source));
        assert!(!mesh.attributes.contains_key("custom_normal"));
        Ok(())
    }

    #[test]
    fn concave_l_face_beside_a_quad_it_wraps_keeps_its_corners_and_materials() -> TestResult {
        // The L-tromino's reflex corner sits on the fourth quad, whose corners all lie on the L's
        // boundary, so triangle corners alone cannot tell which face owns each triangle.
        let doc = scene(&json!([
            {"op":"material.create","id":"jacket"},
            {"op":"material.create","id":"skin"},
            {"op":"node.create","id":"grid","kind":"grid","params":{"x_subdivisions":2,"y_subdivisions":2}},
            {"op":"node.update","target":{"id":"grid"},"set":{"materials":["jacket","skin"]}},
            {"op":"mesh.set_attribute","target":{"id":"grid"},"elements":{"domain":"face","ids":["f0","f1","f2"]},"material_index":0},
            {"op":"mesh.set_attribute","target":{"id":"grid"},"elements":{"domain":"face","ids":["f3"]},"material_index":1},
            {"op":"mesh.dissolve","target":{"id":"grid"},"elements":{"domain":"edge","ids":["e1","e2"]}}
        ]))?;
        let directory = tempfile::tempdir()?;
        let (imported, _) = round_trip_in(&doc, directory.path())?;
        let source = first_mesh(&doc)?;
        let mesh = first_mesh(&imported.doc)?;
        assert!(
            source.faces.iter().any(|face| face.vertices.len() == 8),
            "the L face is an octagon"
        );
        assert_eq!(face_sizes(mesh), face_sizes(source));
        assert_eq!(face_materials(mesh), face_materials(source));
        assert_close("positions", &face_points(mesh), &face_points(source));
        assert!(
            !imported
                .losses
                .iter()
                .any(|loss| loss.feature_id == "mesh.polygon_topology")
        );
        Ok(())
    }

    #[test]
    fn double_sided_quad_keeps_each_face_material() -> TestResult {
        // Two faces over the same four corners, wound oppositely, each in its own material slot.
        let mut doc = scene(&json!([
            {"op":"material.create","id":"jacket"},
            {"op":"material.create","id":"skin"},
            {"op":"node.create","id":"sheet","kind":"plane","params":{"size":2.0}},
            {"op":"node.update","target":{"id":"sheet"},"set":{"materials":["jacket","skin"]}}
        ]))?;
        let node_id = crate::model::Id::new("sheet")?;
        let data_id = doc.nodes[&node_id]
            .data
            .clone()
            .ok_or("sheet mesh data is missing")?;
        let mut mesh = Mesh::from_positions_and_faces(
            vec![
                glam::DVec3::new(-1.0, -1.0, 0.0),
                glam::DVec3::new(1.0, -1.0, 0.0),
                glam::DVec3::new(1.0, 1.0, 0.0),
                glam::DVec3::new(-1.0, 1.0, 0.0),
            ],
            vec![vec![0, 1, 2, 3], vec![3, 2, 1, 0]],
        )?;
        mesh.faces[0].material_index = 1;
        doc.data_blocks
            .get_mut(&data_id)
            .ok_or("sheet data block is missing")?
            .mesh = Some(mesh);
        let doc = apply_operations(
            &doc,
            &json!([{"op":"mesh.shade_smooth","target":{"id":"sheet"},"elements":{"domain":"face","ids":["f0","f1"]}}]),
        )?;
        let directory = tempfile::tempdir()?;
        let (imported, _) = round_trip_in(&doc, directory.path())?;
        let source = first_mesh(&doc)?;
        let mesh = first_mesh(&imported.doc)?;
        assert_eq!(face_materials(source), vec![1, 0]);
        assert_eq!(face_materials(mesh), vec![1, 0]);
        assert_close("positions", &face_points(mesh), &face_points(source));
        assert!(
            !imported
                .losses
                .iter()
                .any(|loss| loss.feature_id == "mesh.polygon_topology")
        );
        Ok(())
    }

    /// Exports `doc`, lets `edit` rewrite the glTF JSON and binary buffer, and imports the result.
    fn import_edited(
        doc: &SceneDoc,
        edit: impl FnOnce(&mut Value, &mut Vec<u8>) -> TestResult<()>,
    ) -> TestResult<crate::error::Result<ImportedGraph>> {
        let directory = tempfile::tempdir()?;
        let snapshot = Snapshot::evaluate(doc, &EvaluationContext::default())?;
        let (document, mut binary) =
            export_with_root(doc, &snapshot, directory.path(), Some("scene.bin"))?;
        let mut root: Value = serde_json::from_slice(&document)?;
        edit(&mut root, &mut binary)?;
        let path = directory.path().join("edited.gltf");
        std::fs::write(&path, serde_json::to_vec(&root)?)?;
        std::fs::write(directory.path().join("scene.bin"), binary)?;
        Ok(import(&path, uuid::Uuid::new_v4().to_string()))
    }

    fn append_test_buffer_view(
        root: &mut Value,
        binary: &mut Vec<u8>,
        bytes: &[u8],
        alignment: usize,
    ) -> TestResult<usize> {
        while !binary.len().is_multiple_of(alignment) {
            binary.push(0);
        }
        let offset = binary.len();
        binary.extend_from_slice(bytes);
        let views = root["bufferViews"]
            .as_array_mut()
            .ok_or("bufferViews are missing")?;
        let index = views.len();
        views.push(json!({
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": bytes.len()
        }));
        Ok(index)
    }

    /// Byte offset in the binary buffer of the bufferView that holds one `extras.potter.polygons` field.
    fn polygon_field_offset(root: &Value, field: &str) -> TestResult<usize> {
        let view = root["meshes"][0]["extras"]["potter"]["polygons"][field]
            .as_u64()
            .ok_or("polygon field is missing")?;
        let offset = root["bufferViews"][usize::try_from(view)?]["byteOffset"]
            .as_u64()
            .ok_or("polygon bufferView has no offset")?;
        Ok(usize::try_from(offset)?)
    }

    fn face_materials(mesh: &Mesh) -> Vec<u32> {
        mesh.faces.iter().map(|face| face.material_index).collect()
    }

    fn face_sizes(mesh: &Mesh) -> Vec<usize> {
        mesh.faces.iter().map(|face| face.vertices.len()).collect()
    }

    fn first_mesh(doc: &SceneDoc) -> TestResult<&Mesh> {
        Ok(doc
            .data_blocks
            .values()
            .find_map(|block| block.mesh.as_ref())
            .ok_or("mesh is missing")?)
    }

    /// Corner positions per face, in face order, as flat `x, y, z` runs.
    fn face_points(mesh: &Mesh) -> Vec<Vec<f64>> {
        let points = mesh
            .vertices
            .iter()
            .map(|vertex| (vertex.id, vertex.co))
            .collect::<BTreeMap<_, _>>();
        mesh.faces
            .iter()
            .map(|face| {
                face.vertices
                    .iter()
                    .flat_map(|id| points[id].to_array())
                    .collect()
            })
            .collect()
    }

    /// Corner UVs of the first UV layer per face, in face order, as flat `u, v` runs.
    fn face_uvs(mesh: &Mesh) -> TestResult<Vec<Vec<f64>>> {
        let layer = uv_layers::read_layers(mesh)
            .into_iter()
            .next()
            .ok_or("uv layer is missing")?;
        Ok(layer
            .faces
            .iter()
            .map(|values| values.iter().flatten().flat_map(|corner| *corner).collect())
            .collect())
    }

    /// Sharp edges as sorted endpoint positions, rounded so f32 storage compares equal.
    fn sharp_edge_points(mesh: &Mesh) -> TestResult<BTreeSet<[[i64; 3]; 2]>> {
        let positions = mesh
            .vertices
            .iter()
            .map(|vertex| (vertex.id, vertex.co))
            .collect::<BTreeMap<_, _>>();
        let rounded = |id: u32| {
            let point = positions[&id];
            point.to_array().map(|value| (value * 1.0e6).round() as i64)
        };
        Ok(mesh
            .edges
            .iter()
            .zip(sharp_edge_flags(mesh)?)
            .filter(|(_, sharp)| *sharp)
            .map(|(edge, _)| {
                let mut pair = [rounded(edge.vertices[0]), rounded(edge.vertices[1])];
                pair.sort_unstable();
                pair
            })
            .collect())
    }

    fn assert_close(what: &str, left: &[Vec<f64>], right: &[Vec<f64>]) {
        assert_eq!(left.len(), right.len(), "{what}: face count");
        for (left, right) in left.iter().zip(right) {
            assert_eq!(left.len(), right.len(), "{what}: corner count");
            for (left, right) in left.iter().zip(right) {
                assert!((left - right).abs() < 1.0e-5, "{what}: {left} != {right}");
            }
        }
    }

    #[test]
    fn custom_normals_survive_topology_modifier_export() -> TestResult {
        let doc = scene(&json!([
            {"op":"node.create","id":"cube","kind":"box","params":{"size":2.0}},
            {"op":"mesh.set_custom_normals","target":{"id":"cube"},"elements":{"domain":"face","ids":["f0"]},"normals":{"f0":[[0.0,0.0,1.0],[0.0,0.0,1.0],[0.0,0.0,1.0],[0.0,0.0,1.0]]}},
            {"op":"modifier.create","target":{"id":"cube"},"id":"subdivide","type":"subdivision","params":{"levels":1}}
        ]))?;
        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        let directory = tempfile::tempdir()?;
        export_with_root(&doc, &snapshot, directory.path(), Some("scene.bin"))?;
        Ok(())
    }

    #[test]
    fn applying_topology_modifier_over_shading_keeps_shading() -> TestResult {
        let base = scene(&json!([
            {"op":"node.create","id":"cube","kind":"box","params":{"size":2.0}},
            {"op":"mesh.shade_smooth","target":{"id":"cube"},"elements":{"domain":"face","ids":["f0","f1","f2","f3","f4","f5"]}},
            {"op":"modifier.create","target":{"id":"cube"},"id":"subdivide","type":"subdivision","params":{"levels":1}}
        ]))?;
        let applied = apply_batch(
            &base,
            &json!({
                "schema_version": 1,
                "base_revision": base.revision,
                "operations": [{"op":"modifier.apply","target":{"id":"cube"},"modifier_id":"subdivide"}]
            }),
        )?;
        let mesh = applied
            .doc
            .data_blocks
            .values()
            .find_map(|data| data.mesh.as_ref())
            .ok_or("applied mesh is missing")?;
        assert_eq!(mesh.faces.len(), 24);
        let smooth = mesh.attributes["shade_smooth"]["values"]
            .as_object()
            .ok_or("smooth flags are missing")?;
        assert_eq!(
            smooth.values().filter(|flag| **flag == json!(true)).count(),
            24
        );
        Ok(())
    }

    #[test]
    fn collection_instance_member_with_topology_modifier_keeps_shading() -> TestResult {
        // The hidden source's subdivision adds faces; its smooth flags must follow those faces.
        let doc = scene(&json!([
            {"op":"collection.create","id":"source"},
            {"op":"node.create","id":"member","kind":"box","collection":"source","params":{"size":1.0}},
            {"op":"mesh.shade_smooth","target":{"id":"member"},"elements":{"domain":"face","ids":["f0","f1","f2","f3","f4","f5"]}},
            {"op":"modifier.create","target":{"id":"member"},"id":"subdivide","type":"subdivision","params":{"levels":1}},
            {"op":"collection.update","target":{"id":"source"},"set":{"exclude":true}},
            {"op":"collection.instance_create","id":"instance","collection":"source","transform":{"translation":[4.0,0.0,0.0]}}
        ]))?;
        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        let instance = snapshot
            .meshes
            .get(&crate::model::Id::new("instance")?)
            .ok_or("instance mesh is missing")?;
        assert_eq!(instance.faces.len(), 24);
        let smooth = instance.attributes["shade_smooth"]["values"]
            .as_object()
            .ok_or("instance smooth flags are missing")?;
        assert_eq!(
            smooth.values().filter(|flag| **flag == json!(true)).count(),
            24
        );
        Ok(())
    }

    #[test]
    fn primitives_sharing_positions_keep_their_own_morph_targets() -> TestResult {
        fn vec3(values: &[[f32; 3]]) -> Vec<u8> {
            values
                .iter()
                .flatten()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        }
        fn indices(values: &[u32]) -> Vec<u8> {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        }
        // Both triangles use vertex 0; each primitive's morph target moves it along glTF Y in opposite directions.
        let positions: [[f32; 3]; 5] = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ];
        let lift: [[f32; 3]; 5] = [[0.0, 1.0, 0.0], [0.0; 3], [0.0; 3], [0.0; 3], [0.0; 3]];
        let drop: [[f32; 3]; 5] = [[0.0, -1.0, 0.0], [0.0; 3], [0.0; 3], [0.0; 3], [0.0; 3]];
        let mut binary = Vec::<u8>::new();
        let mut views = Vec::<Value>::new();
        let mut accessors = Vec::<Value>::new();
        let mut push = |bytes: Vec<u8>, component: u32, kind: &str, count: usize| {
            views.push(json!({"buffer": 0, "byteOffset": binary.len(), "byteLength": bytes.len()}));
            binary.extend_from_slice(&bytes);
            accessors.push(json!({"bufferView": views.len() - 1, "componentType": component, "count": count, "type": kind}));
            accessors.len() - 1
        };
        let position = push(vec3(&positions), 5126, "VEC3", 5);
        let lift_accessor = push(vec3(&lift), 5126, "VEC3", 5);
        let drop_accessor = push(vec3(&drop), 5126, "VEC3", 5);
        let front = push(indices(&[0, 1, 2]), 5125, "SCALAR", 3);
        let back = push(indices(&[0, 3, 4]), 5125, "SCALAR", 3);
        let root = json!({
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "buffers": [{"uri": "scene.bin", "byteLength": binary.len()}],
            "bufferViews": views,
            "accessors": accessors,
            "meshes": [{"primitives": [
                {"attributes": {"POSITION": position}, "indices": front, "targets": [{"POSITION": lift_accessor}]},
                {"attributes": {"POSITION": position}, "indices": back, "targets": [{"POSITION": drop_accessor}]}
            ]}]
        });
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("shared_positions.gltf");
        std::fs::write(&path, serde_json::to_vec(&root)?)?;
        std::fs::write(directory.path().join("scene.bin"), &binary)?;
        let imported = import(&path, uuid::Uuid::new_v4().to_string())?;
        let block = imported
            .doc
            .data_blocks
            .values()
            .find(|block| block.mesh.is_some())
            .ok_or("imported mesh is missing")?;
        let mesh = block.mesh.as_ref().ok_or("imported mesh is missing")?;
        let key = block
            .shape_keys
            .as_ref()
            .and_then(|keys| keys.keys.values().next())
            .ok_or("imported shape key is missing")?;
        // Each triangle's copy of vertex 0 must keep its own morph target: one lifted, one dropped.
        let mut origin_keys = mesh
            .faces
            .iter()
            .flat_map(|face| face.vertices.iter())
            .filter(|id| {
                mesh.vertices
                    .iter()
                    .any(|vertex| vertex.id == **id && vertex.co == glam::DVec3::ZERO)
            })
            .map(|id| key.positions[id])
            .collect::<Vec<_>>();
        origin_keys.sort_by(|left, right| left[2].total_cmp(&right[2]));
        assert_eq!(
            origin_keys,
            vec![[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
            "a shared POSITION must not discard the second primitive's morph target"
        );
        Ok(())
    }
    /// Appends `bytes` to the GLB binary chunk as a buffer view and returns its index.
    fn push_view(bin: &mut Vec<u8>, views: &mut Vec<Value>, bytes: &[u8]) -> usize {
        while !bin.len().is_multiple_of(4) {
            bin.push(0);
        }
        views.push(json!({ "buffer": 0, "byteOffset": bin.len(), "byteLength": bytes.len() }));
        bin.extend_from_slice(bytes);
        views.len() - 1
    }

    /// A GLB with one skinned triangle per skin. `nodes` and `skins` are glTF arrays; each skin gets
    /// identity inverse bind matrices, and skin `i` is drawn by mesh `i`. Scene roots are the nodes no
    /// other node lists as a child.
    fn skinned_glb(nodes: &Value, skins: &Value) -> TestResult<Vec<u8>> {
        let skins = skins.as_array().ok_or("skins must be an array")?;
        let mut bin = Vec::<u8>::new();
        let mut views = Vec::<Value>::new();
        let positions = [0.0_f32, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0];
        let position_view = push_view(
            &mut bin,
            &mut views,
            &positions
                .iter()
                .copied()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        );
        let joints_view = push_view(&mut bin, &mut views, &[0_u8; 12]);
        let weights = [1.0_f32, 0.0, 0.0, 0.0].repeat(3);
        let weights_view = push_view(
            &mut bin,
            &mut views,
            &weights
                .iter()
                .copied()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        );
        let mut accessors = vec![
            json!({ "bufferView": position_view, "componentType": 5126, "count": 3, "type": "VEC3" }),
            json!({ "bufferView": joints_view, "componentType": 5121, "count": 3, "type": "VEC4" }),
            json!({ "bufferView": weights_view, "componentType": 5126, "count": 3, "type": "VEC4" }),
        ];
        let identity = [
            1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let mut skin_values = Vec::with_capacity(skins.len());
        for skin in skins {
            let count = skin["joints"].as_array().map_or(0, Vec::len);
            let bytes = identity
                .iter()
                .copied()
                .cycle()
                .take(16 * count)
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<u8>>();
            let view = push_view(&mut bin, &mut views, &bytes);
            accessors.push(json!({ "bufferView": view, "componentType": 5126, "count": count, "type": "MAT4" }));
            let mut skin = skin.clone();
            skin["inverseBindMatrices"] = json!(accessors.len() - 1);
            skin_values.push(skin);
        }
        let meshes = skins
            .iter()
            .map(|_| json!({ "primitives": [{ "attributes": { "POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2 } }] }))
            .collect::<Vec<_>>();
        let node_values = nodes.as_array().ok_or("nodes must be an array")?;
        let listed = node_values
            .iter()
            .flat_map(|node| node["children"].as_array().cloned().unwrap_or_default())
            .filter_map(|child| child.as_u64())
            .collect::<BTreeSet<_>>();
        let roots = (0..node_values.len())
            .filter(|index| !listed.contains(&(*index as u64)))
            .collect::<Vec<_>>();
        let document = json!({
            "asset": { "version": "2.0" },
            "scene": 0,
            "scenes": [{ "nodes": roots }],
            "nodes": nodes,
            "meshes": meshes,
            "skins": skin_values,
            "accessors": accessors,
            "bufferViews": views,
            "buffers": [{ "byteLength": bin.len() }]
        });
        Ok(to_glb(&serde_json::to_vec(&document)?, &bin)?)
    }

    /// The armatures a GLB imports as, each as its bones mapped to their parent bones, sorted.
    fn imported_armatures(glb: &[u8]) -> TestResult<Vec<BTreeMap<String, Option<String>>>> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("skins.glb");
        std::fs::write(&path, glb)?;
        let graph = import(&path, uuid::Uuid::new_v4().to_string())?;
        let mut armatures = graph
            .doc
            .data_blocks
            .values()
            .filter(|block| block.data_type == "armature")
            .map(|block| {
                block.armature.as_ref().map_or_else(BTreeMap::new, |data| {
                    data.bones
                        .iter()
                        .map(|(id, bone)| {
                            (
                                id.to_string(),
                                bone.parent.as_ref().map(ToString::to_string),
                            )
                        })
                        .collect()
                })
            })
            .collect::<Vec<_>>();
        armatures.sort();
        Ok(armatures)
    }

    fn bones(pairs: &[(&str, Option<&str>)]) -> BTreeMap<String, Option<String>> {
        pairs
            .iter()
            .map(|(id, parent)| ((*id).to_owned(), parent.map(str::to_owned)))
            .collect()
    }

    /// Two skins whose joint sets share `hips` form one armature with the union of their joints, as
    /// Blender's importer builds it: both skins' deepest common ancestor is `Armature`.
    #[test]
    fn skins_sharing_a_joint_tree_import_as_one_armature() -> TestResult {
        let nodes = json!([
            { "name": "Armature", "children": [1] },
            { "name": "hips", "translation": [0.0, 1.0, 0.0], "children": [2, 3] },
            { "name": "spine", "translation": [0.0, 1.0, 0.0] },
            { "name": "leg", "translation": [0.2, -0.5, 0.0] },
            { "name": "meshA", "mesh": 0, "skin": 0 },
            { "name": "meshB", "mesh": 1, "skin": 1 }
        ]);
        let skins = json!([{ "joints": [1, 2] }, { "joints": [1, 3] }]);
        let glb = skinned_glb(&nodes, &skins)?;
        assert_eq!(
            imported_armatures(&glb)?,
            vec![bones(&[
                ("hips", None),
                ("leg", Some("hips")),
                ("spine", Some("hips")),
            ])],
            "the armature holds the union of both skins' joints under their glTF parents"
        );
        Ok(())
    }

    /// Two characters under one empty root: each skin's deepest common ancestor is its own character
    /// node, so Blender imports two armatures and Potter must too.
    #[test]
    fn skins_of_separate_characters_under_a_shared_root_stay_separate() -> TestResult {
        let nodes = json!([
            { "name": "Root", "children": [1, 5] },
            { "name": "Char0", "children": [2, 4] },
            { "name": "hipsA", "children": [3] },
            { "name": "spineA" },
            { "name": "meshA", "mesh": 0, "skin": 0 },
            { "name": "Char1", "children": [6, 8] },
            { "name": "hipsB", "children": [7] },
            { "name": "spineB" },
            { "name": "meshB", "mesh": 1, "skin": 1 }
        ]);
        let skins = json!([{ "joints": [2, 3] }, { "joints": [6, 7] }]);
        let glb = skinned_glb(&nodes, &skins)?;
        assert_eq!(
            imported_armatures(&glb)?,
            vec![
                bones(&[("hipsa", None), ("spinea", Some("hipsa"))]),
                bones(&[("hipsb", None), ("spineb", Some("hipsb"))]),
            ],
            "each character keeps its own armature"
        );
        Ok(())
    }

    /// Stores three UV layers on `mesh`, `UVMap`, `UVMap.001`, and `UVMap.002`, rendering `render`.
    /// Layer `k` holds `k / 4 + corner / 8` in U and `1 / 2 + k / 4 - corner / 8` in V, exact in f32.
    fn store_three_uv_layers(mesh: &mut Mesh, render: &str) {
        let layers = ["UVMap", "UVMap.001", "UVMap.002"]
            .into_iter()
            .enumerate()
            .map(|(set, name)| {
                let faces = mesh
                    .faces
                    .iter()
                    .map(|face| {
                        Some(
                            (0..face.vertices.len())
                                .map(|corner| {
                                    [
                                        set as f64 * 0.25 + corner as f64 * 0.125,
                                        0.5 + set as f64 * 0.25 - corner as f64 * 0.125,
                                    ]
                                })
                                .collect(),
                        )
                    })
                    .collect();
                UvLayer::new(name, faces)
            })
            .collect();
        uv_layers::write(
            mesh,
            &UvStore {
                layers,
                active: Some("UVMap".to_owned()),
                render: Some(render.to_owned()),
            },
        );
    }

    /// The three-layer store on every mesh data block of `doc`.
    fn store_three_uv_layers_everywhere(doc: &mut SceneDoc, render: &str) {
        for data in doc.data_blocks.values_mut() {
            if let Some(mesh) = data.mesh.as_mut() {
                store_three_uv_layers(mesh, render);
            }
        }
    }

    fn store_single_uv_layer(
        doc: &mut SceneDoc,
        node_name: &str,
        layer_name: &str,
        has_data: bool,
    ) -> TestResult {
        let node_id = crate::model::Id::new(node_name)?;
        let data_id = doc
            .nodes
            .get(&node_id)
            .and_then(|node| node.data.clone())
            .ok_or("node mesh data is missing")?;
        let mesh = doc
            .data_blocks
            .get_mut(&data_id)
            .and_then(|data| data.mesh.as_mut())
            .ok_or("node mesh is missing")?;
        let faces = mesh
            .faces
            .iter()
            .map(|face| {
                has_data.then(|| {
                    (0..face.vertices.len())
                        .map(|corner| [corner as f64 + 1.0, corner as f64 + 2.0])
                        .collect()
                })
            })
            .collect();
        uv_layers::write(
            mesh,
            &UvStore {
                layers: vec![UvLayer::new(layer_name, faces)],
                active: Some(layer_name.to_owned()),
                render: Some(layer_name.to_owned()),
            },
        );
        Ok(())
    }

    /// Asserts two UV lists hold the same values in any order.
    fn assert_same_uvs(actual: &[[f64; 2]], expected: &[[f64; 2]]) {
        let sorted = |values: &[[f64; 2]]| {
            let mut values = values.to_vec();
            values.sort_by(|left, right| {
                left[0]
                    .total_cmp(&right[0])
                    .then(left[1].total_cmp(&right[1]))
            });
            values
        };
        assert_eq!(sorted(actual), sorted(expected));
    }

    #[test]
    fn three_uv_layers_export_as_texcoord_sets_and_import_by_name() -> TestResult {
        let root = tempfile::tempdir()?;
        let mut doc = scene_with_asset_root(
            root.path(),
            &json!([
                {"op":"image.create","id":"paint","width":1,"height":1,"colorspace":"srgb","fill_color":[0.25,0.5,0.75,1.0]},
                {"op":"material.create","id":"named","base_color_texture":{"image":"paint","interpolation":"linear"}},
                {"op":"material.create","id":"default","base_color_texture":{"image":"paint","interpolation":"linear"}},
                {"op":"node.create","id":"named_quad","kind":"plane","params":{"size":2.0},"material":"named"},
                {"op":"node.create","id":"default_quad","kind":"plane","params":{"size":2.0},"material":"default"}
            ]),
        )?;
        store_three_uv_layers_everywhere(&mut doc, "UVMap");
        let named = doc
            .materials
            .get_mut(&crate::model::Id::new("named")?)
            .ok_or("named material is missing")?;
        named
            .base_color_texture
            .as_mut()
            .ok_or("named texture is missing")?
            .uv_layer = Some("UVMap.002".to_owned());

        let (document, binary) = export_root_and_binary_at_root(&doc, root.path())?;
        let buffers = vec![binary];
        let stored = uv_layers::read_layers(first_mesh(&doc)?);
        assert_eq!(stored.len(), 3);
        for mesh_value in document["meshes"].as_array().ok_or("meshes are missing")? {
            let attributes = &mesh_value["primitives"][0]["attributes"];
            for (set, layer) in stored.iter().enumerate() {
                let accessor = attributes[format!("TEXCOORD_{set}")]
                    .as_u64()
                    .ok_or("TEXCOORD set is missing")?;
                let exported = super::read_vec2(&document, &buffers, usize::try_from(accessor)?)?;
                let values = layer
                    .faces
                    .iter()
                    .flatten()
                    .flatten()
                    .copied()
                    .collect::<Vec<_>>();
                assert_same_uvs(&exported, &values);
            }
            assert!(
                attributes["TEXCOORD_3"].is_null(),
                "only the stored layers export"
            );
        }
        for material in document["materials"]
            .as_array()
            .ok_or("materials are missing")?
        {
            let coord = material["pbrMetallicRoughness"]["baseColorTexture"]["texCoord"].as_u64();
            match material["extras"]["potter"]["id"].as_str() {
                Some("named") => assert_eq!(coord, Some(2), "a named layer takes its TEXCOORD"),
                Some("default") => {
                    assert_eq!(coord, Some(0), "a null layer takes the render layer");
                }
                _ => {}
            }
        }

        let (imported, _) = round_trip_in(&doc, root.path())?;
        let imported_mesh = first_mesh(&imported.doc)?;
        let store = uv_layers::read(imported_mesh);
        let names = store
            .layers
            .iter()
            .map(|layer| layer.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names, ["UVMap", "UVMap.001", "UVMap.002"]);
        assert_eq!(store.active.as_deref(), Some("UVMap"));
        assert_eq!(store.render.as_deref(), Some("UVMap"));
        for (imported_layer, stored_layer) in store.layers.iter().zip(&stored) {
            assert_eq!(imported_layer.faces, stored_layer.faces);
        }
        let mut texture_layers = imported
            .doc
            .materials
            .values()
            .map(|material| {
                material
                    .base_color_texture
                    .as_ref()
                    .and_then(|texture| texture.uv_layer.clone())
            })
            .collect::<Vec<_>>();
        texture_layers.sort();
        assert_eq!(texture_layers, vec![None, Some("UVMap.002".to_owned())]);
        Ok(())
    }

    #[test]
    fn shared_material_exports_uv_index_variants_per_mesh() -> TestResult {
        let root = tempfile::tempdir()?;
        let mut doc = scene_with_asset_root(
            root.path(),
            &json!([
                {"op":"image.create","id":"paint","width":1,"height":1,"colorspace":"srgb","fill_color":[0.25,0.5,0.75,1.0]},
                {"op":"material.create","id":"shared","base_color_texture":{"image":"paint","interpolation":"linear"}},
                {"op":"node.create","id":"left","kind":"plane","params":{"size":2.0},"material":"shared"},
                {"op":"node.create","id":"right","kind":"plane","params":{"size":2.0},"material":"shared"}
            ]),
        )?;
        store_three_uv_layers_everywhere(&mut doc, "UVMap");
        doc.materials
            .get_mut(&crate::model::Id::new("shared")?)
            .and_then(|material| material.base_color_texture.as_mut())
            .ok_or("shared material texture is missing")?
            .uv_layer = Some("UVMap.002".to_owned());
        let right_id = crate::model::Id::new("right")?;
        let data_id = doc
            .nodes
            .get(&right_id)
            .and_then(|node| node.data.clone())
            .ok_or("right mesh data is missing")?;
        let mesh = doc
            .data_blocks
            .get_mut(&data_id)
            .and_then(|data| data.mesh.as_mut())
            .ok_or("right mesh is missing")?;
        let mut store = uv_layers::read(mesh);
        store.layers.swap(0, 2);
        uv_layers::write(mesh, &store);

        let (document, _) = export_root_and_binary_at_root(&doc, root.path())?;
        let mesh_index = |name: &str| -> TestResult<usize> {
            let node = document["nodes"]
                .as_array()
                .ok_or("nodes are missing")?
                .iter()
                .find(|node| node["extras"]["potter"]["id"] == name)
                .ok_or("mesh node is missing")?;
            Ok(usize::try_from(
                node["mesh"].as_u64().ok_or("node mesh index is missing")?,
            )?)
        };
        let material_index = |name: &str| -> TestResult<usize> {
            let mesh = mesh_index(name)?;
            Ok(usize::try_from(
                document["meshes"][mesh]["primitives"][0]["material"]
                    .as_u64()
                    .ok_or("primitive material index is missing")?,
            )?)
        };
        let left_material = material_index("left")?;
        let right_material = material_index("right")?;
        assert_ne!(left_material, right_material);
        assert_eq!(
            document["materials"][left_material]["pbrMetallicRoughness"]["baseColorTexture"]["texCoord"]
                .as_u64(),
            Some(2)
        );
        assert_eq!(
            document["materials"][right_material]["pbrMetallicRoughness"]["baseColorTexture"]["texCoord"]
                .as_u64(),
            Some(0)
        );
        Ok(())
    }

    #[test]
    fn texture_naming_a_missing_uv_layer_reports_uv_layer_missing() -> TestResult {
        let root = tempfile::tempdir()?;
        let mut doc = scene_with_asset_root(
            root.path(),
            &json!([
                {"op":"image.create","id":"paint","width":1,"height":1,"colorspace":"srgb","fill_color":[0.25,0.5,0.75,1.0]},
                {"op":"material.create","id":"paint_mat","base_color_texture":{"image":"paint","interpolation":"linear"}},
                {"op":"node.create","id":"quad","kind":"plane","params":{"size":2.0},"material":"paint_mat"}
            ]),
        )?;
        store_three_uv_layers_everywhere(&mut doc, "UVMap");
        let material = doc
            .materials
            .get_mut(&crate::model::Id::new("paint_mat")?)
            .ok_or("material is missing")?;
        material
            .base_color_texture
            .as_mut()
            .ok_or("texture is missing")?
            .uv_layer = Some("Missing".to_owned());

        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        let (losses, warnings) = uv_layer_diagnostics(&doc, &snapshot, None)?;
        assert!(
            warnings.iter().any(|warning| {
                warning.details["feature_id"] == "uv.layer_missing"
                    && warning.details["material_id"] == "paint_mat"
                    && warning.details["status"] == "fallback"
            }),
            "{warnings:?}"
        );
        assert!(losses.is_empty());
        // The texture still exports, sampling the render layer.
        let (document, _) = export_root_and_binary_at_root(&doc, root.path())?;
        assert_eq!(
            document["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["texCoord"]
                .as_u64(),
            Some(0)
        );
        Ok(())
    }

    #[test]
    fn uvless_texture_reports_a_loss_without_fabricating_texcoord_attributes() -> TestResult {
        let root = tempfile::tempdir()?;
        let doc = scene_with_asset_root(
            root.path(),
            &json!([
                {"op":"image.create","id":"paint","width":1,"height":1,"colorspace":"srgb","fill_color":[0.25,0.5,0.75,1.0]},
                {"op":"material.create","id":"paint_mat","base_color_texture":{"image":"paint"}},
                {"op":"node.create","id":"quad","kind":"plane","params":{"size":2.0},"material":"paint_mat"}
            ]),
        )?;
        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        let (losses, warnings) = uv_layer_diagnostics(&doc, &snapshot, None)?;
        assert!(
            losses
                .iter()
                .any(|loss| loss.feature_id == "uv.layer_missing")
        );
        assert!(
            warnings.is_empty(),
            "an absent UV layer is a loss, not fallback: {warnings:?}"
        );

        let (document, _) = export_root_and_binary_at_root(&doc, root.path())?;
        let node = document["nodes"]
            .as_array()
            .and_then(|nodes| {
                nodes
                    .iter()
                    .find(|node| node["extras"]["potter"]["id"] == "quad")
            })
            .ok_or("textured mesh node is missing")?;
        let mesh_index = usize::try_from(node["mesh"].as_u64().ok_or("mesh index is missing")?)?;
        let attributes = document["meshes"][mesh_index]["primitives"][0]["attributes"]
            .as_object()
            .ok_or("primitive attributes are missing")?;
        assert!(!attributes.keys().any(|name| name.starts_with("TEXCOORD_")));
        Ok(())
    }

    #[test]
    fn higher_texcoord_sets_import_sparse_and_normalized_integer_accessors() -> TestResult {
        let doc = scene(&json!([
            {"op":"node.create","id":"plane","kind":"plane","params":{"size":2.0}},
            {"op":"uv.unwrap","target":{"id":"plane"},"method":"smart"}
        ]))?;
        let imported = import_edited(&doc, |root, binary| {
            let position_accessor = usize::try_from(
                root["meshes"][0]["primitives"][0]["attributes"]["POSITION"]
                    .as_u64()
                    .ok_or("POSITION accessor is missing")?,
            )?;
            let count = usize::try_from(
                root["accessors"][position_accessor]["count"]
                    .as_u64()
                    .ok_or("POSITION count is missing")?,
            )?;
            assert!(count >= 2, "fixture plane needs at least two vertices");
            let last_index = u8::try_from(count - 1)?;
            let indices_view = append_test_buffer_view(root, binary, &[0, last_index], 4)?;
            let sparse_values = [255_u8, 0, 128, 255];
            let sparse_values_view = append_test_buffer_view(root, binary, &sparse_values, 4)?;
            let sparse_accessor = {
                let accessors = root["accessors"]
                    .as_array_mut()
                    .ok_or("accessors are missing")?;
                let index = accessors.len();
                accessors.push(json!({
                    "componentType": 5121,
                    "count": count,
                    "type": "VEC2",
                    "normalized": true,
                    "sparse": {
                        "count": 2,
                        "indices": {"bufferView": indices_view, "componentType": 5121},
                        "values": {"bufferView": sparse_values_view}
                    }
                }));
                index
            };

            let mut unsigned_short_values = Vec::with_capacity(count * 4);
            for _ in 0..count {
                for value in [0_u16, u16::MAX] {
                    unsigned_short_values.extend_from_slice(&value.to_le_bytes());
                }
            }
            let unsigned_short_view =
                append_test_buffer_view(root, binary, &unsigned_short_values, 4)?;
            let unsigned_short_sparse_indices = append_test_buffer_view(root, binary, &[1_u8], 4)?;
            let mut unsigned_short_sparse_values = Vec::with_capacity(4);
            for value in [32_768_u16, 16_384_u16] {
                unsigned_short_sparse_values.extend_from_slice(&value.to_le_bytes());
            }
            let unsigned_short_sparse_values_view =
                append_test_buffer_view(root, binary, &unsigned_short_sparse_values, 4)?;
            let unsigned_short_accessor = {
                let accessors = root["accessors"]
                    .as_array_mut()
                    .ok_or("accessors are missing")?;
                let index = accessors.len();
                accessors.push(json!({
                    "bufferView": unsigned_short_view,
                    "componentType": 5123,
                    "count": count,
                    "type": "VEC2",
                    "normalized": true,
                    "sparse": {
                        "count": 1,
                        "indices": {"bufferView": unsigned_short_sparse_indices, "componentType": 5121},
                        "values": {"bufferView": unsigned_short_sparse_values_view}
                    }
                }));
                index
            };
            root["meshes"][0]["primitives"][0]["attributes"]["TEXCOORD_1"] = json!(sparse_accessor);
            root["meshes"][0]["primitives"][0]["attributes"]["TEXCOORD_2"] =
                json!(unsigned_short_accessor);
            root["buffers"][0]["byteLength"] = json!(binary.len());
            Ok(())
        })??;

        let layers = uv_layers::read_layers(first_mesh(&imported.doc)?);
        assert_eq!(
            layers
                .iter()
                .map(|layer| layer.name.as_str())
                .collect::<Vec<_>>(),
            ["UVMap", "UVMap.001", "UVMap.002"]
        );
        let values = |set: usize| {
            layers[set]
                .faces
                .iter()
                .flatten()
                .flatten()
                .copied()
                .collect::<Vec<_>>()
        };
        let sparse_uvs = values(1);
        assert!(sparse_uvs.contains(&[1.0, 0.0]));
        assert!(sparse_uvs.contains(&[f64::from(128_u8) / 255.0, 1.0]));
        assert!(sparse_uvs.contains(&[0.0, 0.0]));
        let normalized_uvs = values(2);
        assert!(normalized_uvs.contains(&[0.0, 1.0]));
        assert!(normalized_uvs.contains(&[32_768.0 / 65_535.0, 16_384.0 / 65_535.0]));
        Ok(())
    }

    #[test]
    fn higher_texcoord_without_storage_imports_zero_uv_rows() -> TestResult {
        let doc = scene(&json!([
            {"op":"node.create","id":"plane","kind":"plane","params":{"size":2.0}},
            {"op":"uv.unwrap","target":{"id":"plane"},"method":"smart"}
        ]))?;
        let imported = import_edited(&doc, |root, _| {
            let position_accessor = usize::try_from(
                root["meshes"][0]["primitives"][0]["attributes"]["POSITION"]
                    .as_u64()
                    .ok_or("POSITION accessor is missing")?,
            )?;
            let count = root["accessors"][position_accessor]["count"]
                .as_u64()
                .ok_or("POSITION count is missing")?;
            let accessor = {
                let accessors = root["accessors"]
                    .as_array_mut()
                    .ok_or("accessors are missing")?;
                let index = accessors.len();
                accessors.push(json!({
                    "componentType": 5126,
                    "count": count,
                    "type": "VEC2"
                }));
                index
            };
            root["meshes"][0]["primitives"][0]["attributes"]["TEXCOORD_1"] = json!(accessor);
            Ok(())
        })??;

        let mesh = first_mesh(&imported.doc)?;
        let layers = uv_layers::read_layers(mesh);
        let layer = layers
            .iter()
            .find(|layer| layer.name == "UVMap.001")
            .ok_or("zero-filled TEXCOORD_1 did not create UVMap.001")?;
        assert_eq!(layer.faces.len(), mesh.faces.len());
        let mut corner_count = 0;
        for (row, face) in layer.faces.iter().zip(&mesh.faces) {
            let row = row.as_ref().ok_or("TEXCOORD_1 face row is missing")?;
            assert_eq!(row.len(), face.vertices.len());
            assert!(row.iter().all(|uv| *uv == [0.0, 0.0]));
            corner_count += row.len();
        }
        assert_eq!(
            corner_count,
            mesh.faces
                .iter()
                .map(|face| face.vertices.len())
                .sum::<usize>()
        );
        assert!(corner_count > 0, "fixture must exercise real UV corners");
        Ok(())
    }

    #[test]
    fn maximum_texcoord_index_imports_with_layer_limit_loss() -> TestResult {
        let doc = scene(&json!([
            {"op":"node.create","id":"plane","kind":"plane","params":{"size":2.0}},
            {"op":"uv.unwrap","target":{"id":"plane"},"method":"smart"}
        ]))?;
        let imported = import_edited(&doc, |root, _| {
            let uv_accessor = root["meshes"][0]["primitives"][0]["attributes"]["TEXCOORD_0"]
                .as_u64()
                .ok_or("TEXCOORD_0 accessor is missing")?;
            let name = format!("TEXCOORD_{}", usize::MAX);
            root["meshes"][0]["primitives"][0]["attributes"][name.as_str()] = json!(uv_accessor);
            Ok(())
        })??;

        assert!(
            imported
                .losses
                .iter()
                .any(|loss| loss.feature_id == super::feature_id::MESH_UV_LAYER),
            "out-of-limit TEXCOORD set should report a layer-limit loss: {:?}",
            imported.losses
        );
        let mesh = first_mesh(&imported.doc)?;
        assert!(
            !mesh.faces.is_empty(),
            "mesh should remain usable after importing an out-of-limit TEXCOORD set"
        );
        assert!(
            uv_layers::read_layers(mesh)
                .iter()
                .any(|layer| layer.name == "UVMap"),
            "lower TEXCOORD layer should remain present"
        );
        Ok(())
    }

    #[test]
    fn gltf_import_tags_normal_map_only_when_other_textures_force_a_graph() -> TestResult {
        let root = tempfile::tempdir()?;
        let doc = scene_with_asset_root(
            root.path(),
            &json!([
                {"op":"image.create","id":"normal","width":1,"height":1,"colorspace":"linear","fill_color":[0.5,0.5,1.0,1.0]},
                {"op":"material.create","id":"normal_mat","normal_texture":{"image":"normal"}},
                {"op":"node.create","id":"surface","kind":"plane","params":{"size":1.0},"material":"normal_mat"}
            ]),
        )?;
        let (mut document, binary) = export_root_and_binary_at_root(&doc, root.path())?;
        std::fs::write(root.path().join("scene.bin"), &binary)?;
        let direct_path = root.path().join("direct.gltf");
        std::fs::write(&direct_path, serde_json::to_vec(&document)?)?;
        let direct = import(&direct_path, uuid::Uuid::new_v4().to_string())?;
        let material_id = crate::model::Id::new("normal_mat")?;
        let direct_material = direct
            .doc
            .materials
            .get(&material_id)
            .ok_or("direct normal material is missing")?;
        assert!(direct_material.normal_texture.is_some());
        assert!(direct_material.node_tree.is_none());

        let texture_index = document["materials"][0]["normalTexture"]["index"].clone();
        document["materials"][0]["emissiveTexture"] = json!({"index": texture_index});
        document["materials"][0]["emissiveFactor"] = json!([1.0, 1.0, 1.0]);
        let graph_path = root.path().join("graph.gltf");
        std::fs::write(&graph_path, serde_json::to_vec(&document)?)?;
        let graph_import = import(&graph_path, uuid::Uuid::new_v4().to_string())?;
        let graph_material = graph_import
            .doc
            .materials
            .get(&material_id)
            .ok_or("graph-converted normal material is missing")?;
        let graph_id = graph_material
            .node_tree
            .as_ref()
            .ok_or("normal graph is missing")?;
        let graph = graph_import
            .doc
            .node_groups
            .get(graph_id)
            .ok_or("normal graph group is missing")?;
        let normal_map = graph
            .nodes
            .values()
            .find(|node| node.node_type == "ShaderNodeNormalMap")
            .ok_or("imported NormalMap node is missing")?;
        assert_eq!(
            normal_map.properties["potter_gltf_imported_normal_map"],
            json!(true)
        );
        Ok(())
    }

    #[test]
    fn uvless_collection_slot_omits_union_uv_attrs_and_unsupported_textures() -> TestResult {
        let root = tempfile::tempdir()?;
        let mut doc = scene_with_asset_root(
            root.path(),
            &json!([
                {"op":"image.create","id":"paint","width":1,"height":1,"colorspace":"srgb","fill_color":[0.25,0.5,0.75,1.0]},
                {"op":"material.create","id":"shared","base_color_texture":{"image":"paint"}},
                {"op":"collection.create","id":"source"},
                {"op":"node.create","id":"a_member","kind":"plane","collection":"source","params":{"size":2.0}},
                {"op":"node.update","target":{"id":"a_member"},"set":{"materials":["shared"]}},
                {"op":"node.create","id":"b_member","kind":"plane","collection":"source","params":{"size":2.0}},
                {"op":"node.update","target":{"id":"b_member"},"set":{"materials":["shared"]}},
                {"op":"collection.instance_create","id":"instance","collection":"source"}
            ]),
        )?;
        store_single_uv_layer(&mut doc, "b_member", "MemberUV", true)?;
        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        let (losses, warnings) = uv_layer_diagnostics(&doc, &snapshot, None)?;
        assert!(
            losses
                .iter()
                .any(|loss| loss.feature_id == "uv.layer_missing")
        );
        assert_eq!(
            warnings,
            Vec::new(),
            "unexpected UV-layer fallback warnings"
        );

        let (document, _) = export_root_and_binary_at_root(&doc, root.path())?;
        let mesh = document["meshes"]
            .as_array()
            .and_then(|meshes| meshes.iter().find(|mesh| mesh["name"] == "instance_source"))
            .ok_or("collection instance mesh is missing")?;
        let primitives = mesh["primitives"]
            .as_array()
            .ok_or("collection instance primitives are missing")?;
        assert_eq!(primitives.len(), 2);
        let missing_attributes = primitives[0]["attributes"]
            .as_object()
            .ok_or("UV-less member attributes are missing")?;
        assert!(
            !missing_attributes
                .keys()
                .any(|name| name.starts_with("TEXCOORD_"))
        );
        let missing_material = usize::try_from(
            primitives[0]["material"]
                .as_u64()
                .ok_or("UV-less material variant is missing")?,
        )?;
        let missing_pbr = &document["materials"][missing_material]["pbrMetallicRoughness"];
        assert!(missing_pbr["baseColorTexture"].is_null());
        assert!(missing_pbr["baseColorFactor"].is_array());

        let mapped_attributes = primitives[1]["attributes"]
            .as_object()
            .ok_or("UV-mapped member attributes are missing")?;
        assert!(mapped_attributes["TEXCOORD_0"].is_number());
        assert!(!mapped_attributes.contains_key("TEXCOORD_1"));
        let mapped_material = usize::try_from(
            primitives[1]["material"]
                .as_u64()
                .ok_or("UV-mapped material variant is missing")?,
        )?;
        assert!(
            document["materials"][mapped_material]["pbrMetallicRoughness"]["baseColorTexture"]
                .is_object()
        );
        Ok(())
    }
}

#[cfg(test)]
mod material_texture_tests {
    #![expect(clippy::unwrap_used, reason = "tests")]
    use crate::model::Id;

    use super::{ChannelSource, TextureChannel, fill_unread, identity_image, reads};

    fn source(image: &str, channel: TextureChannel) -> ChannelSource {
        ChannelSource {
            image: Id::new(image).unwrap(),
            channel,
        }
    }

    #[test]
    fn unread_channels_are_filled_from_the_read_image_and_reuse_it() {
        // Opaque base color: RGB read, alpha unread.
        let mut channels = [
            Some(source("img", TextureChannel::Red)),
            Some(source("img", TextureChannel::Green)),
            Some(source("img", TextureChannel::Blue)),
            None,
        ];
        fill_unread(&mut channels, reads::RGB);
        assert_eq!(identity_image(&channels), Some(&Id::new("img").unwrap()));
    }

    #[test]
    fn a_read_channel_that_is_constant_forces_packing() {
        // Metallic from the image, roughness constant: G is read, so it cannot be left to the image.
        let mut channels = [None, None, Some(source("img", TextureChannel::Blue)), None];
        fill_unread(&mut channels, reads::MR);
        assert!(identity_image(&channels).is_none());
        assert!(channels[1].is_none());
    }

    #[test]
    fn channels_from_two_images_are_not_filled() {
        let mut channels = [
            Some(source("a", TextureChannel::Red)),
            Some(source("b", TextureChannel::Green)),
            None,
            None,
        ];
        fill_unread(&mut channels, reads::RGBA);
        assert!(channels[3].is_none());
    }
}
