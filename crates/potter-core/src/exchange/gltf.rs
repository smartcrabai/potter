use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    fs,
    path::Path,
};

use glam::{DMat4, DQuat, DVec3};
use serde_json::{Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    eval::{Snapshot, mesh_vertex_weights},
    geom::{
        Face, Mesh,
        normals::{corner_normals, morph_corner_normals, shading_keys_stale},
    },
    image::{ImageData, ImageInterpolation, load_image_data},
    model::{
        Action, ArmatureData, AvatarSettings, AvatarVersion, Bone, CameraData, CameraProjection,
        DataBlock, Extrapolation, FCurve, Id, Image, ImageAlphaMode, ImageColorspace, ImageSource,
        Interpolation, Keyframe, LightData, LightType, Material, Modifier, Node, SceneDoc,
        ShapeKey, ShapeKeyData, TextureRef, Transform, VertexGroup,
    },
};

use super::import_error;

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
    remapped_image_indices: BTreeMap<(Id, usize), usize>,
    texture_indices: BTreeMap<(Id, Option<usize>, bool), usize>,
    sampler_indices: BTreeMap<bool, usize>,
}
impl TextureExport {
    fn new() -> Self {
        Self {
            images: Vec::new(),
            textures: Vec::new(),
            samplers: Vec::new(),
            image_indices: BTreeMap::new(),
            remapped_image_indices: BTreeMap::new(),
            texture_indices: BTreeMap::new(),
            sampler_indices: BTreeMap::new(),
        }
    }

    pub(crate) fn texture_info(
        &mut self,
        reference: &TextureRef,
        doc: &SceneDoc,
        root_path: &Path,
        binary: &mut Vec<u8>,
        views: &mut Vec<Value>,
    ) -> Result<Value> {
        self.texture_info_channel(reference, None, doc, root_path, binary, views)
    }

    pub(crate) fn texture_info_channel(
        &mut self,
        reference: &TextureRef,
        source_channel: Option<usize>,
        doc: &SceneDoc,
        root_path: &Path,
        binary: &mut Vec<u8>,
        views: &mut Vec<Value>,
    ) -> Result<Value> {
        let image_index = if let Some(channel) = source_channel {
            if !matches!(channel, 1 | 2) {
                return Err(PotError::new(
                    ErrorCode::InvalidArgument,
                    "VRM mask remapping supports only source channels G and B",
                ));
            }
            let image_key = (reference.image.clone(), channel);
            if let Some(index) = self.remapped_image_indices.get(&image_key) {
                *index
            } else {
                let image = texture_image_for_export(doc, &reference.image)?;
                let source_pixels = load_image_data(image, root_path, reference.interpolation)?;
                let derived_name = format!("{}_channel_{channel}", image.name);
                let derived_pixels = ImageData {
                    width: source_pixels.width,
                    height: source_pixels.height,
                    pixels: source_pixels
                        .pixels
                        .iter()
                        .map(|pixel| {
                            // The loader linearizes sRGB, but packed mask channels are numeric values.
                            let value = if image.colorspace == ImageColorspace::Srgb {
                                crate::color::linear_to_srgb_unclamped(pixel[channel])
                            } else {
                                pixel[channel]
                            };
                            [value, value, value, 1.0]
                        })
                        .collect(),
                    tiles: BTreeMap::new(),
                    interpolation: reference.interpolation,
                };
                let encoded = encode_png_with_colorspace(ImageColorspace::Linear, &derived_pixels)?;
                align4(binary);
                let offset = binary.len();
                binary.extend_from_slice(&encoded);
                let view = push_view(views, offset, encoded.len(), None);
                let index = self.images.len();
                self.images.push(json!({
                    "name": derived_name,
                    "bufferView": view,
                    "mimeType": "image/png",
                    "extras": { "potter": {
                        "id": reference.image,
                        "remapped_channel": channel
                    } }
                }));
                self.remapped_image_indices.insert(image_key, index);
                index
            }
        } else {
            self.export_image_index(
                &reference.image,
                reference.interpolation,
                doc,
                root_path,
                binary,
                views,
            )?
        };
        let nearest = reference.interpolation == ImageInterpolation::Closest;
        let texture_key = (reference.image.clone(), source_channel, nearest);
        let texture_index = if let Some(index) = self.texture_indices.get(&texture_key) {
            *index
        } else {
            let sampler_index = if let Some(index) = self.sampler_indices.get(&nearest) {
                *index
            } else {
                let index = self.samplers.len();
                let filter = if nearest { 9728 } else { 9729 };
                self.samplers.push(json!({
                    "magFilter": filter,
                    "minFilter": filter,
                    "wrapS": 10497,
                    "wrapT": 10497
                }));
                self.sampler_indices.insert(nearest, index);
                index
            };
            let index = self.textures.len();
            self.textures
                .push(json!({ "source": image_index, "sampler": sampler_index }));
            self.texture_indices.insert(texture_key, index);
            index
        };
        Ok(json!({ "index": texture_index, "texCoord": 0 }))
    }
}

impl TextureExport {
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

    pub(crate) fn texture_index(
        &mut self,
        reference: &TextureRef,
        doc: &SceneDoc,
        root_path: &Path,
        binary: &mut Vec<u8>,
        views: &mut Vec<Value>,
    ) -> Result<usize> {
        let info = self.texture_info(reference, doc, root_path, binary, views)?;
        info["index"]
            .as_u64()
            .and_then(|index| usize::try_from(index).ok())
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "glTF texture index is invalid"))
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

fn skin_binding(doc: &SceneDoc, node: &Node) -> Result<Option<SkinBinding>> {
    let modifiers = node
        .modifiers
        .iter()
        .filter(|modifier| modifier.enabled && modifier.modifier_type == "armature")
        .collect::<Vec<_>>();
    if modifiers.is_empty() {
        return Ok(None);
    }
    if modifiers.len() != 1
        || node
            .modifiers
            .iter()
            .any(|modifier| modifier.enabled && modifier.modifier_type != "armature")
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

/// glTF vertices after splitting by source vertex, UV, and every shading layer's normal.
struct ExportVertexLayout {
    source_vertex_indices: Vec<usize>,
    uvs: Option<Vec<[f64; 2]>>,
    /// Layer zero is base shading; each later layer is one shape key's morphed shading.
    normals: Vec<Vec<DVec3>>,
    face_indices: BTreeMap<u32, BTreeMap<u32, u32>>,
}

fn export_vertex_layout(
    mesh: &Mesh,
    layers: &[Vec<DVec3>],
    fallback_normal: DVec3,
) -> Result<ExportVertexLayout> {
    let uv_entries = mesh.attributes.get("uv_map").and_then(Value::as_array);
    let vertex_indices = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<BTreeMap<_, _>>();
    let faces = mesh
        .faces
        .iter()
        .map(|face| (face.id, face))
        .collect::<BTreeMap<_, _>>();
    let mut face_uvs = BTreeMap::<u32, Vec<[f64; 2]>>::new();
    let mut preferred_uvs = vec![None; mesh.vertices.len()];
    for entry in uv_entries.into_iter().flatten() {
        let face_id = entry["face_id"]
            .as_u64()
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| PotError::new(ErrorCode::SceneInvalid, "UV face ID is invalid"))?;
        let face = faces.get(&face_id).ok_or_else(|| {
            PotError::new(ErrorCode::SceneInvalid, "UV map references a missing face")
        })?;
        let coordinates = entry["uv"].as_array().ok_or_else(|| {
            PotError::new(ErrorCode::SceneInvalid, "UV face coordinates are missing")
        })?;
        if coordinates.len() != face.vertices.len() {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "UV corner count does not match its face",
            ));
        }
        let mut values = Vec::with_capacity(coordinates.len());
        for (vertex_id, coordinate) in face.vertices.iter().zip(coordinates) {
            let uv_values = coordinate
                .as_array()
                .filter(|values| values.len() == 2)
                .ok_or_else(|| {
                    PotError::new(ErrorCode::SceneInvalid, "UV coordinate is invalid")
                })?;
            let uv = [
                uv_values[0].as_f64().ok_or_else(|| {
                    PotError::new(ErrorCode::SceneInvalid, "UV coordinate is invalid")
                })?,
                uv_values[1].as_f64().ok_or_else(|| {
                    PotError::new(ErrorCode::SceneInvalid, "UV coordinate is invalid")
                })?,
            ];
            if !uv[0].is_finite() || !uv[1].is_finite() {
                return Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    "UV coordinate is non-finite",
                ));
            }
            let vertex_index = vertex_indices.get(vertex_id).copied().ok_or_else(|| {
                PotError::new(
                    ErrorCode::SceneInvalid,
                    "UV face references a missing vertex",
                )
            })?;
            preferred_uvs[vertex_index].get_or_insert(uv);
            values.push(uv);
        }
        if let Some(previous) = face_uvs.get(&face_id) {
            if previous != &values {
                return Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    "UV map repeats a face with conflicting coordinates",
                ));
            }
        } else {
            face_uvs.insert(face_id, values);
        }
    }
    let corner_count = mesh
        .faces
        .iter()
        .map(|face| face.vertices.len())
        .sum::<usize>();
    if layers.iter().any(|layer| layer.len() != corner_count) {
        return Err(PotError::new(
            ErrorCode::InternalError,
            "glTF corner normals do not match mesh corners",
        ));
    }
    // Corners grouped by source vertex; corner indices follow mesh.faces order then face corner order.
    let mut corners_by_vertex =
        vec![Vec::<(u32, u32, usize, [f64; 2])>::new(); mesh.vertices.len()];
    let mut corner = 0;
    for face in &mesh.faces {
        let coordinates = face_uvs.get(&face.id);
        for (position, vertex_id) in face.vertices.iter().enumerate() {
            let source_index = vertex_indices.get(vertex_id).copied().ok_or_else(|| {
                PotError::new(
                    ErrorCode::SceneInvalid,
                    "UV face references a missing vertex",
                )
            })?;
            let uv = coordinates
                .and_then(|values| values.get(position))
                .copied()
                .unwrap_or([0.0, 0.0]);
            corners_by_vertex[source_index].push((face.id, *vertex_id, corner, uv));
            corner += 1;
        }
    }
    let has_uv = uv_entries.is_some();
    let mut source_vertex_indices = Vec::<usize>::new();
    let mut uvs = Vec::<[f64; 2]>::new();
    let mut normals = vec![Vec::<DVec3>::new(); layers.len()];
    let mut outputs = BTreeMap::<(usize, [u64; 2], Vec<[u64; 3]>), u32>::new();
    let mut face_indices = BTreeMap::<u32, BTreeMap<u32, u32>>::new();
    for (source_index, corners) in corners_by_vertex.iter().enumerate() {
        if corners.is_empty() {
            // Loose vertices keep an output slot so positions and vertex IDs stay aligned.
            source_vertex_indices.push(source_index);
            uvs.push(preferred_uvs[source_index].unwrap_or([0.0, 0.0]));
            for output in &mut normals {
                output.push(fallback_normal);
            }
            continue;
        }
        for (face_id, vertex_id, corner, uv) in corners {
            let uv = if has_uv { *uv } else { [0.0, 0.0] };
            // Normal bits are normalized so -0.0 and 0.0 share one vertex.
            let key = (
                source_index,
                uv.map(|value| (value + 0.0).to_bits()),
                layers
                    .iter()
                    .map(|layer| {
                        layer[*corner]
                            .to_array()
                            .map(|value| (value + 0.0).to_bits())
                    })
                    .collect::<Vec<_>>(),
            );
            let index = if let Some(index) = outputs.get(&key) {
                *index
            } else {
                let index = u32::try_from(source_vertex_indices.len()).map_err(|_| {
                    PotError::new(ErrorCode::LimitExceeded, "glTF mesh has too many vertices")
                })?;
                source_vertex_indices.push(source_index);
                uvs.push(uv);
                for (output, layer) in normals.iter_mut().zip(layers) {
                    output.push(layer[*corner]);
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
        uvs: has_uv.then_some(uvs),
        normals,
        face_indices,
    })
}

/// Maps pot-space normals into the glTF output frame: inverse transpose of diag(scale) * basis.
fn output_normals(normals: &[DVec3], geometry_scale: DVec3, basis: DMat4) -> Result<Vec<DVec3>> {
    normals
        .iter()
        .map(|normal| {
            let unit = (basis.transform_vector3(*normal) / geometry_scale).normalize();
            if unit.is_finite() {
                Ok(unit)
            } else {
                Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    "mesh normal cannot be transformed for glTF export",
                ))
            }
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
) -> Result<MeshExportLayout> {
    let key_deltas = match data_block {
        Some(block) => shape_key_deltas(mesh, block)?,
        None => Vec::new(),
    };
    let base = shape_key_base_positions(mesh, data_block);
    let base_corners = corner_normals(mesh, &base)?;
    let mut layers = vec![output_normals(&base_corners, geometry_scale, basis)?];
    for deltas in &key_deltas {
        let morphed = base
            .iter()
            .zip(deltas)
            .map(|(point, delta)| *point + *delta)
            .collect::<Vec<_>>();
        // A morph that collapses a face has no normal there, so its NORMAL delta is zero.
        layers.push(output_normals(
            &morph_corner_normals(mesh, &morphed, &base_corners)?,
            geometry_scale,
            basis,
        )?);
    }
    let fallback = output_normals(&[DVec3::Z], geometry_scale, basis)?[0];
    let layout = export_vertex_layout(mesh, &layers, fallback)?;
    Ok(MeshExportLayout { layout, key_deltas })
}

/// Material slot a face exports to. Collection-instance meshes carry no per-source material
/// lists, so all their faces use the instancing node's slot 0.
// shortcut: instance meshes drop per-face material slots, resolve source material lists during evaluation to upgrade
fn export_material_slot(face: &Face, instanced: bool) -> u32 {
    if instanced { 0 } else { face.material_index }
}

/// Face material slots actually bound by faces, validated against the node material list.
fn used_material_ids<'a>(node: &'a Node, mesh: &Mesh, instanced: bool) -> Result<Vec<&'a Id>> {
    let mut slots = BTreeSet::new();
    for face in &mesh.faces {
        let raw_slot = export_material_slot(face, instanced);
        let slot = usize::try_from(raw_slot)
            .ok()
            .filter(|slot| *slot < node.materials.len().max(1))
            .ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "mesh face material slot is outside the node material list",
                    json!({ "face_id": face.id, "slot": raw_slot, "material_count": node.materials.len() }),
                )
            })?;
        slots.insert(slot);
    }
    Ok(slots
        .into_iter()
        .filter_map(|slot| node.materials.get(slot))
        .collect())
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

fn validate_mesh_export_attributes(mesh: &Mesh, data_block: Option<&DataBlock>) -> Result<()> {
    for (name, value) in &mesh.attributes {
        let supported = match name.as_str() {
            "uv_map" | "seam" | "shade_smooth" | "sharp" | "custom_normal" => true,
            "vertex_groups" => {
                data_block.is_some_and(|block| valid_vertex_groups_attribute(value, block))
            }
            _ => false,
        };
        if !supported {
            return Err(unsupported_export(
                "mesh.vertex_attributes",
                &format!("glTF export does not preserve Potter mesh attribute `{name}`"),
            ));
        }
    }
    Ok(())
}

fn source_mesh_for_export<'a>(
    node: &Node,
    data_block: &'a DataBlock,
    evaluated: &'a Mesh,
) -> Result<&'a Mesh> {
    let Some(shape_keys) = data_block
        .shape_keys
        .as_ref()
        .filter(|shape_keys| !shape_keys.keys.is_empty())
    else {
        validate_shading_topology(data_block, evaluated)?;
        return Ok(evaluated);
    };
    if node
        .modifiers
        .iter()
        .any(|modifier| modifier.enabled && modifier.modifier_type != "armature")
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

/// Shading attributes are keyed by face and edge IDs; a modifier stack that changes the faces
/// leaves those keys stale, so export refuses rather than shading with the wrong normals.
fn validate_shading_topology(data_block: &DataBlock, evaluated: &Mesh) -> Result<()> {
    let Some(source) = data_block.mesh.as_ref() else {
        return Ok(());
    };
    if shading_keys_stale(source, evaluated) {
        return Err(unsupported_export(
            "mesh.shading_topology",
            "the modifier stack changes this mesh's faces while it has custom normals or smooth shading; apply the modifiers before authoring shading, or remove the modifiers",
        ));
    }
    Ok(())
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

fn avatar_basis(version: AvatarVersion) -> DMat4 {
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
            && let Some(binding) = skin_binding(doc, node)?
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
fn validate_material_export(doc: &SceneDoc, material: &Material) -> Result<()> {
    if material
        .node_tree
        .as_ref()
        .is_some_and(|graph_id| !super::is_default_material_graph(doc, graph_id))
    {
        return Err(unsupported_export(
            "material.shader_graph",
            "glTF export supports explicit Potter material fields and texture references, not node-tree shader graphs",
        ));
    }
    if material.metallic_texture.is_some() || material.roughness_texture.is_some() {
        let (Some(metallic), Some(roughness)) =
            (&material.metallic_texture, &material.roughness_texture)
        else {
            return Err(unsupported_export(
                "material.metallic_roughness_texture",
                "glTF metallic-roughness textures require Potter metallic and roughness references to share one image",
            ));
        };
        if metallic.image != roughness.image
            || metallic.uv_map != roughness.uv_map
            || metallic.interpolation != roughness.interpolation
        {
            return Err(unsupported_export(
                "material.metallic_roughness_texture",
                "glTF cannot represent separate Potter metallic and roughness image references",
            ));
        }
    }
    for texture in [
        material.base_color_texture.as_ref(),
        material.metallic_texture.as_ref(),
        material.roughness_texture.as_ref(),
        material.normal_texture.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        texture_image_for_export(doc, &texture.image)?;
    }
    Ok(())
}

fn validate_texture_uv_map(texture: &TextureRef, has_uv: bool, mtoon: bool) -> Result<()> {
    let (map_reason, missing_uv_reason) = if mtoon {
        (
            "VRM MToon textures support only Potter's default UV map.",
            "A VRM MToon texture is assigned to an exported mesh without UV coordinates.",
        )
    } else {
        (
            "glTF export supports only the Potter default UV map",
            "textured materials require UV coordinates on the exported mesh",
        )
    };
    if texture
        .uv_map
        .as_deref()
        .is_some_and(|name| name != "uv_map")
    {
        return Err(unsupported_export("material.texture_uv_map", map_reason));
    }
    if !has_uv {
        return Err(unsupported_export(
            "material.texture_uv_map",
            missing_uv_reason,
        ));
    }
    Ok(())
}

fn validate_mesh_material_uv(
    doc: &SceneDoc,
    material_id: &Id,
    avatar: Option<&AvatarSettings>,
    has_uv: bool,
) -> Result<()> {
    let material = doc.materials.get(material_id).ok_or_else(|| {
        PotError::new(
            ErrorCode::SceneInvalid,
            "node material reference is missing",
        )
    })?;
    for texture in [
        material.base_color_texture.as_ref(),
        material.metallic_texture.as_ref(),
        material.roughness_texture.as_ref(),
        material.normal_texture.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        validate_texture_uv_map(texture, has_uv, false)?;
    }
    if let Some(mtoon) = avatar.and_then(|avatar| avatar.mtoon.get(material_id)) {
        for texture in super::vrm::uv_textures(mtoon) {
            validate_texture_uv_map(texture, has_uv, true)?;
        }
    }
    Ok(())
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
    for material in doc.materials.values() {
        validate_material_export(doc, material)?;
    }
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
        let Some(binding) = skin_binding(doc, node)? else {
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
                source_mesh_for_export(node, data_block, evaluated)?,
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
        let export = mesh_export_layout(mesh, data_block, scale, avatar_basis(version))?;
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
        for material_id in used_material_ids(node, mesh, data_block.is_none())? {
            validate_mesh_material_uv(doc, material_id, Some(avatar), export.layout.uvs.is_some())?;
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
    for (id, material) in &doc.materials {
        validate_material_export(doc, material)?;
        material_indices.insert(id.clone(), materials.len());
        let mut extensions = serde_json::Map::new();
        // Inert when the emissive factor is zero or the strength is 1; extras.potter keeps the Potter value.
        if !crate::float::equal_f64_array(&material.emission_color, &[0.0; 3])
            && !crate::float::equal_f64(material.emission_strength, 1.0)
        {
            extensions.insert(
                "KHR_materials_emissive_strength".to_owned(),
                json!({ "emissiveStrength": material.emission_strength }),
            );
            extensions_used.insert("KHR_materials_emissive_strength".to_owned());
        }
        if !crate::float::equal_f64(material.transmission, 0.0) {
            extensions.insert(
                "KHR_materials_transmission".to_owned(),
                json!({ "transmissionFactor": material.transmission }),
            );
            extensions_used.insert("KHR_materials_transmission".to_owned());
        }
        if !crate::float::equal_f64(material.ior, 1.5)
            && !vrm.is_some_and(|(avatar, _)| avatar.mtoon.contains_key(id))
        {
            extensions.insert(
                "KHR_materials_ior".to_owned(),
                json!({ "ior": material.ior }),
            );
            extensions_used.insert("KHR_materials_ior".to_owned());
        }
        let mut pbr = json!({
            "baseColorFactor": material.base_color,
            "metallicFactor": material.metallic,
            "roughnessFactor": material.roughness
        });
        if let Some(texture) = &material.base_color_texture {
            pbr["baseColorTexture"] =
                texture_export.texture_info(texture, doc, root_path, &mut binary, &mut views)?;
        }
        if let Some(metallic) = &material.metallic_texture {
            pbr["metallicRoughnessTexture"] =
                texture_export.texture_info(metallic, doc, root_path, &mut binary, &mut views)?;
        }
        let mut normal_texture = Value::Null;
        if let Some(texture) = &material.normal_texture {
            normal_texture =
                texture_export.texture_info(texture, doc, root_path, &mut binary, &mut views)?;
        }
        let mut value = json!({
            "name": material.name,
            "pbrMetallicRoughness": pbr,
            "emissiveFactor": material.emission_color,
            "doubleSided": material.double_sided,
            "extras": { "potter": {
                "id": id,
                "emission_strength": material.emission_strength,
                "transmission": material.transmission,
                "ior": material.ior
            } }
        });
        if !normal_texture.is_null() {
            value["normalTexture"] = normal_texture;
        }
        if vrm.is_some() {
            match material.alpha_mode.as_str() {
                "opaque" => {}
                "clip" => {
                    value["alphaMode"] = json!("MASK");
                    value["alphaCutoff"] = json!(material.alpha_threshold);
                }
                "blend" => value["alphaMode"] = json!("BLEND"),
                _ => {
                    return Err(PotError::new(
                        ErrorCode::SceneInvalid,
                        "VRM material alpha_mode must be opaque, clip, or blend",
                    ));
                }
            }
        }
        if !extensions.is_empty() {
            value["extensions"] = Value::Object(extensions);
        }
        materials.push(value);
    }
    let visible_node_ids = super::render_visible_nodes(doc, snapshot)?;
    let node_ids = doc.nodes.keys().cloned().collect::<Vec<_>>();
    let node_indices = node_ids
        .iter()
        .enumerate()
        .map(|(index, id)| (id.clone(), index))
        .collect::<BTreeMap<_, _>>();
    let mut meshes = Vec::<Value>::new();
    let mut mesh_indices = BTreeMap::<(Id, Option<[u64; 3]>, Vec<Id>), usize>::new();
    let mut instance_mesh_indices = BTreeMap::<(Id, Option<[u64; 3]>, Vec<Id>), usize>::new();
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
        let Some(binding) = skin_binding(doc, node)? else {
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
            let mesh = source_mesh_for_export(node, data_block, evaluated)?;
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
        if let Some(data_id) = data_id
            && let Some(index) =
                mesh_indices.get(&(data_id.clone(), mesh_variant, node.materials.clone()))
        {
            node_mesh_indices.insert(node_id.clone(), *index);
            continue;
        }
        if let Some(target) = &instance_target
            && let Some(index) =
                instance_mesh_indices.get(&(target.clone(), mesh_variant, node.materials.clone()))
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
        let MeshExportLayout { layout, key_deltas } =
            mesh_export_layout(mesh, data_block, geometry_scale, geometry_basis)?;
        let uv_values = layout.uvs.as_ref();
        for material_id in used_material_ids(node, mesh, data_block.is_none())? {
            validate_mesh_material_uv(
                doc,
                material_id,
                vrm.map(|(avatar, _)| avatar),
                uv_values.is_some(),
            )?;
        }
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
        if let Some(uv_values) = uv_values {
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
            attributes["TEXCOORD_0"] = json!(accessor);
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
        let triangles = mesh.triangulate().map_err(|error| {
            PotError::with_details(
                ErrorCode::ExportFailed,
                "mesh triangulation failed",
                json!({ "reason": error.to_string(), "data_id": data_id }),
            )
        })?;
        let mut slot_indices = BTreeMap::<u32, Vec<u32>>::new();
        let mut triangle_offset = 0;
        for face in &mesh.faces {
            let triangle_count = face.vertices.len().saturating_sub(2);
            let face_triangles = triangles
                .get(triangle_offset..triangle_offset + triangle_count)
                .ok_or_else(|| {
                    PotError::new(ErrorCode::InternalError, "mesh triangulation is incomplete")
                })?;
            let face_indices = layout.face_indices.get(&face.id);
            let indices = slot_indices
                .entry(export_material_slot(face, data_block.is_none()))
                .or_default();
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
            if !node.materials.is_empty() {
                let material_id = usize::try_from(*slot)
                    .ok()
                    .and_then(|slot| node.materials.get(slot))
                    .ok_or_else(|| {
                        PotError::new(
                            ErrorCode::SceneInvalid,
                            "mesh face material slot is outside the node material list",
                        )
                    })?;
                let material_index =
                    material_indices.get(material_id).copied().ok_or_else(|| {
                        PotError::new(
                            ErrorCode::SceneInvalid,
                            "node material reference is missing",
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
                "vertex_weights": data_block.map(|block| export_vertex_weights(mesh, block))
            } }
        });
        if !target_names.is_empty() {
            mesh_value["weights"] = json!(mesh_weights);
            mesh_value["extras"]["targetNames"] = json!(target_names);
        }
        meshes.push(mesh_value);
        if let Some(data_id) = data_id {
            mesh_indices.insert(
                (data_id.clone(), mesh_variant, node.materials.clone()),
                mesh_index,
            );
        }
        if let Some(target) = instance_target {
            instance_mesh_indices
                .insert((target, mesh_variant, node.materials.clone()), mesh_index);
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
    let json_bytes = serde_json::to_vec(&root)
        .map_err(|error| PotError::new(ErrorCode::ExportFailed, error.to_string()))?;
    Ok((json_bytes, binary))
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

pub(crate) fn to_glb(json_bytes: &[u8], binary: &[u8]) -> Result<Vec<u8>> {
    let json_length = json_bytes
        .len()
        .checked_add((4 - json_bytes.len() % 4) % 4)
        .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, "GLB JSON chunk is too large"))?;
    let binary_length = binary
        .len()
        .checked_add((4 - binary.len() % 4) % 4)
        .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, "GLB binary chunk is too large"))?;
    let total_length = 12_usize
        .checked_add(8)
        .and_then(|value| value.checked_add(json_length))
        .and_then(|value| value.checked_add(8))
        .and_then(|value| value.checked_add(binary_length))
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
        let name = image_value["name"]
            .as_str()
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

fn import_texture_ref(info: &Value, root: &Value, image_ids: &[Id]) -> Result<Option<TextureRef>> {
    if !info.is_object() {
        return Ok(None);
    }
    if info["texCoord"].as_u64().unwrap_or(0) != 0 {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "glTF material texture uses an unsupported UV set",
            json!({ "feature_id": "material.texture_uv_set", "status": "not_supported" }),
        ));
    }
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
        uv_map: Some("uv_map".to_owned()),
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

type ImportedSkinGraph = (
    Vec<ImportedSkin>,
    BTreeSet<usize>,
    BTreeMap<usize, Id>,
    Vec<Id>,
);

fn import_skins(
    root: &Value,
    buffers: &[Vec<u8>],
    node_values: &[Value],
    parents: &[Option<usize>],
    node_ids: &[Id],
    doc: &mut SceneDoc,
    used_data_ids: &mut BTreeSet<Id>,
    used_node_ids: &mut BTreeSet<Id>,
) -> Result<ImportedSkinGraph> {
    let skin_values = root["skins"].as_array().map_or(&[][..], Vec::as_slice);
    let world_matrices = node_world_matrices(node_values, parents)?;
    let mut skins = Vec::with_capacity(skin_values.len());
    let mut joint_nodes = BTreeSet::new();
    let mut armature_data_ids = BTreeMap::new();
    let mut synthetic_roots = Vec::new();
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
        joint_nodes.extend(joint_indices.iter().copied());
        let mut bone_ids = Vec::with_capacity(joint_indices.len());
        let mut bone_id_by_raw = BTreeMap::new();
        let mut used_bone_ids = BTreeSet::new();
        for (joint_order, joint_index) in joint_indices.iter().enumerate() {
            let joint = &node_values[*joint_index];
            let raw_id = joint["extras"]["potter"]["bone_id"]
                .as_str()
                .or_else(|| joint["extras"]["potter"]["id"].as_str())
                .or_else(|| joint["name"].as_str())
                .unwrap_or("bone");
            let id = unique_import_id(raw_id, "bone", joint_order, &mut used_bone_ids)?;
            bone_id_by_raw.insert(raw_id.to_owned(), id.clone());
            bone_ids.push(id);
        }
        let armature_raw_id = skin["extras"]["potter"]["armature_id"].as_str();
        let armature_node_index = armature_raw_id.and_then(|raw_id| {
            node_ids
                .iter()
                .position(|node_id| node_id.as_str() == raw_id)
        });
        let armature_node_id = if let Some(index) = armature_node_index {
            node_ids[index].clone()
        } else {
            let suggested = format!("armature_skin_{skin_index}");
            let id = unique_import_id(&suggested, "armature", skin_index, used_node_ids)?;
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
        let armature_metadata = armature_node_index
            .and_then(|index| node_values.get(index))
            .and_then(|node| node["extras"]["potter"]["armature"].as_object())
            .map(|_| {
                serde_json::from_value::<ArmatureData>(
                    node_values[armature_node_index.unwrap_or_default()]["extras"]["potter"]
                        ["armature"]
                        .clone(),
                )
                .map_err(|error| import_error(format!("glTF armature metadata is invalid: {error}")))
            })
            .transpose()?;
        let armature = if let Some(armature) = armature_metadata {
            armature
        } else {
            let joint_set = joint_indices.iter().copied().collect::<BTreeSet<_>>();
            let mut bones = BTreeMap::new();
            for (joint_order, joint_index) in joint_indices.iter().enumerate() {
                let joint_node = &node_values[*joint_index];
                let extras = &joint_node["extras"]["potter"];
                let id = bone_ids[joint_order].clone();
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
                let joint_bind = inverse_bind_matrices
                    .as_ref()
                    .and_then(|matrices| matrices.get(joint_order))
                    .map_or(world_matrices[*joint_index], glam::DMat4::inverse);
                if !finite_matrix(joint_bind) {
                    return Err(import_error("glTF inverse bind matrix is singular"));
                }
                let converted = POT_TO_GLTF.inverse() * joint_bind * POT_TO_GLTF;
                let head = converted.transform_point3(DVec3::ZERO);
                let mut tail = converted.transform_point3(DVec3::Y);
                if head.distance_squared(tail) <= f64::EPSILON {
                    tail = head + DVec3::Y;
                }
                let parent = parents[*joint_index]
                    .filter(|parent| joint_set.contains(parent))
                    .and_then(|parent| joint_indices.iter().position(|value| *value == parent))
                    .map(|parent_order| bone_ids[parent_order].clone());
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
        let data_suggested = armature_node_index
            .and_then(|index| node_values[index]["extras"]["potter"]["data_id"].as_str())
            .map_or_else(|| format!("armature_skin_{skin_index}_data"), str::to_owned);
        let armature_data_id =
            unique_import_id(&data_suggested, "armature_data", skin_index, used_data_ids)?;
        doc.data_blocks.insert(
            armature_data_id.clone(),
            DataBlock {
                data_type: "armature".to_owned(),
                mesh: None,
                armature: Some(armature),
                ..DataBlock::default()
            },
        );
        let imported_armature_data_id = armature_data_id.clone();
        if let Some(index) = armature_node_index {
            armature_data_ids.insert(index, armature_data_id);
        } else if let Some(node) = doc.nodes.get_mut(&armature_node_id) {
            node.data = Some(armature_data_id);
        }
        skins.push(ImportedSkin {
            armature_node: armature_node_id,
            armature_data_id: imported_armature_data_id,
            joint_ids: bone_ids,
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
        Some(ids) => {
            let mut seen = BTreeSet::new();
            ids.into_iter().filter(|id| seen.insert(*id)).collect()
        }
        None => vertex_order.clone(),
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
    let exact_weight_metadata = mesh_value["extras"]["potter"]["vertex_weights"].is_object();
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

pub(crate) fn import(path: &Path, scene_id: String) -> Result<crate::exchange::ImportedGraph> {
    fn import_shape_keys(
        mesh_value: &Value,
        welded_source_ids: Option<&[u32]>,
        mesh: &Mesh,
        morph_deltas: &[Vec<[f64; 3]>],
    ) -> Result<ShapeKeyData> {
        if mesh_value["extras"]["potter"]["shape_key_data"].is_object() {
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
            return Ok(shape_keys);
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
        Ok(result)
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
    for (index, material) in material_values.iter().enumerate() {
        let suggested = material["extras"]["potter"]["id"]
            .as_str()
            .or_else(|| material["name"].as_str())
            .unwrap_or("material");
        let id = unique_import_id(suggested, "material", index, &mut used_material_ids)?;
        let mut color = [0.6, 0.6, 0.6, 1.0];
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
        let base_color_texture = import_texture_ref(
            &material["pbrMetallicRoughness"]["baseColorTexture"],
            &root,
            &image_ids,
        )?;
        let metallic_roughness_texture = import_texture_ref(
            &material["pbrMetallicRoughness"]["metallicRoughnessTexture"],
            &root,
            &image_ids,
        )?;
        let normal_texture = import_texture_ref(&material["normalTexture"], &root, &image_ids)?;
        let value = Material {
            name: material["name"]
                .as_str()
                .map_or_else(|| id.to_string(), str::to_owned),
            base_color: color,
            metallic: material["pbrMetallicRoughness"]["metallicFactor"]
                .as_f64()
                .unwrap_or(0.0),
            roughness: material["pbrMetallicRoughness"]["roughnessFactor"]
                .as_f64()
                .unwrap_or(1.0),
            emission_color,
            emission_strength:
                material["extensions"]["KHR_materials_emissive_strength"]["emissiveStrength"]
                    .as_f64()
                    .or_else(|| material["extras"]["potter"]["emission_strength"].as_f64())
                    .unwrap_or(
                        if crate::float::equal_f64_array(&emission_color, &[0.0; 3]) {
                            0.0
                        } else {
                            1.0
                        },
                    ),
            transmission:
                material["extensions"]["KHR_materials_transmission"]["transmissionFactor"]
                    .as_f64()
                    .unwrap_or(0.0),
            ior: material["extensions"]["KHR_materials_ior"]["ior"]
                .as_f64()
                .unwrap_or(1.5),
            normal_texture,
            base_color_texture,
            metallic_texture: metallic_roughness_texture.clone(),
            roughness_texture: metallic_roughness_texture,
            ..Material::default()
        };
        doc.materials.insert(id.clone(), value);
        material_ids.push(id);
    }

    let mut data_ids = Vec::<Id>::new();
    let mut mesh_materials = Vec::<Vec<Id>>::new();
    let mut mesh_skin_weights = Vec::<Option<Vec<SkinVertexWeights>>>::new();
    let mesh_values = root["meshes"].as_array().map_or(&[][..], Vec::as_slice);
    let mut used_data_ids = BTreeSet::new();
    let mut morph_normal_loss = false;
    for (mesh_index, mesh_value) in mesh_values.iter().enumerate() {
        let suggested = mesh_value["extras"]["potter"]["data_id"]
            .as_str()
            .or_else(|| mesh_value["name"].as_str())
            .unwrap_or("mesh");
        let data_id = unique_import_id(suggested, "mesh", mesh_index, &mut used_data_ids)?;
        let primitives = mesh_value["primitives"]
            .as_array()
            .ok_or_else(|| import_error("glTF mesh primitives are missing"))?;
        let mut positions = Vec::<DVec3>::new();
        let mut polygons = Vec::<Vec<usize>>::new();
        let mut face_material_indices = Vec::<u32>::new();
        let mut face_uvs = Vec::<Vec<[f64; 2]>>::new();
        let mut has_uvs = false;
        let mut slots = Vec::<Id>::new();
        let mut morph_target_count = None;
        let mut morph_deltas = Vec::<Vec<[f64; 3]>>::new();
        let mut joint_data = Vec::<Option<SkinVertexWeights>>::new();
        let mut normals = Vec::<DVec3>::new();
        let mut has_missing_normals = false;
        let mut morph_normal_deltas = Vec::<Vec<[f64; 3]>>::new();
        let mut morph_normals_missing = false;
        let mut vertex_bases = BTreeMap::<VertexPayload, usize>::new();
        for primitive in primitives {
            if primitive["attributes"]["TEXCOORD_1"].is_object() {
                return Err(unsupported_export(
                    "material.texture_uv_set",
                    "glTF import supports only TEXCOORD_0",
                ));
            }
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
                        .map(|point| DVec3::from_array(axis_from_gltf(*point))),
                );
            }
            if let Some(accessor) = primitive["attributes"]["NORMAL"].as_u64() {
                let accessor = usize::try_from(accessor)
                    .map_err(|_| import_error("glTF NORMAL accessor exceeds platform range"))?;
                let values = read_vec3(&root, &binaries, accessor)?;
                if values.len() != decoded.len() {
                    return Err(import_error("glTF NORMAL count does not match POSITION"));
                }
                if first_use {
                    normals.extend(
                        values
                            .iter()
                            .map(|normal| DVec3::from_array(axis_from_gltf(*normal))),
                    );
                }
            } else {
                has_missing_normals = true;
                if first_use {
                    normals.extend(std::iter::repeat_n(DVec3::ZERO, decoded.len()));
                }
            }
            let primitive_uvs = if let Some(accessor) =
                primitive["attributes"]["TEXCOORD_0"].as_u64()
            {
                has_uvs = true;
                let accessor = usize::try_from(accessor)
                    .map_err(|_| import_error("glTF TEXCOORD accessor exceeds platform range"))?;
                let values = read_vec2(&root, &binaries, accessor)?;
                if values.len() != decoded.len() {
                    return Err(import_error(
                        "glTF TEXCOORD_0 count does not match POSITION",
                    ));
                }
                Some(values)
            } else {
                None
            };
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
                    morph_deltas[target_index]
                        .extend(values.iter().map(|delta| axis_from_gltf(*delta)));
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
                        morph_normal_deltas[target_index]
                            .extend(values.iter().map(|delta| axis_from_gltf(*delta)));
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
            let material_slot = if let Some(index) = primitive["material"].as_u64() {
                let material_index = usize::try_from(index)
                    .map_err(|_| import_error("glTF material index exceeds platform range"))?;
                let id = material_ids
                    .get(material_index)
                    .cloned()
                    .ok_or_else(|| import_error("glTF primitive references a missing material"))?;
                if let Some(slot) = slots.iter().position(|candidate| candidate == &id) {
                    u32::try_from(slot).map_err(|_| import_error("too many glTF material slots"))?
                } else {
                    let slot = u32::try_from(slots.len())
                        .map_err(|_| import_error("too many glTF material slots"))?;
                    slots.push(id);
                    slot
                }
            } else {
                0
            };
            let face_start = polygons.len();
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
                let uv = polygon
                    .iter()
                    .map(|vertex_index| {
                        primitive_uvs
                            .as_ref()
                            .and_then(|values| values.get(vertex_index.saturating_sub(base)))
                            .copied()
                            .unwrap_or([0.0, 0.0])
                    })
                    .collect();
                face_uvs.push(uv);
            }
        }
        // Copies of one Potter vertex, split per UV and normal, rejoin here; corners keep their own normals.
        let source_ids = potter_vertex_ids(mesh_value)?;
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
        if has_uvs {
            mesh.attributes.insert(
                "uv_map".to_owned(),
                json!(
                    mesh.faces
                        .iter()
                        .zip(face_uvs)
                        .map(|(face, uv)| json!({ "face_id": face.id, "uv": uv }))
                        .collect::<Vec<_>>()
                ),
            );
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
            if import_shading(&mut mesh, &corner_normals, &morph_targets)? {
                morph_normal_loss = true;
            }
        }
        let shape_keys = if morph_deltas.is_empty() {
            None
        } else {
            Some(import_shape_keys(
                mesh_value,
                source_ids_welded.as_deref(),
                &mesh,
                &welded_morphs,
            )?)
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
        doc.data_blocks.insert(
            data_id.clone(),
            DataBlock {
                data_type: "mesh".to_owned(),
                descriptor: None,
                mesh: Some(mesh),
                shape_keys,
                vertex_groups,
                vertex_weights,
                light: None,
                ..DataBlock::default()
            },
        );
        data_ids.push(data_id);
        mesh_materials.push(slots);
        mesh_skin_weights.push(imported_joint_data);
    }

    let camera_ids = import_cameras(&root, &mut doc, &mut used_data_ids)?;
    let light_ids = import_lights(&root, &mut doc, &mut used_data_ids)?;
    let node_values = root["nodes"].as_array().map_or(&[][..], Vec::as_slice);
    let parents = node_parent_indices(node_values)?;
    let mut used_node_ids = BTreeSet::new();
    let node_ids = node_values
        .iter()
        .enumerate()
        .map(|(index, node)| {
            let suggested = node["extras"]["potter"]["id"].as_str().unwrap_or("node");
            unique_import_id(suggested, "node", index, &mut used_node_ids)
        })
        .collect::<Result<Vec<_>>>()?;
    let (imported_skins, joint_nodes, armature_data_ids, synthetic_roots) = import_skins(
        &root,
        &binaries,
        node_values,
        &parents,
        &node_ids,
        &mut doc,
        &mut used_data_ids,
        &mut used_node_ids,
    )?;
    let mut restored_skins = BTreeMap::<Id, Id>::new();
    for (index, node_value) in node_values.iter().enumerate() {
        if joint_nodes.contains(&index) {
            continue;
        }
        let local = POT_TO_GLTF.inverse() * node_matrix(node_value)? * POT_TO_GLTF;
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
                params,
                binding_data: None,
                runtime: crate::model::ModifierRuntime::default(),
            });
        }
        let materials = mesh_index
            .and_then(|mesh_index| mesh_materials.get(mesh_index))
            .cloned()
            .unwrap_or_default();
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
        let bone_parent = parents[index].and_then(|parent_index| {
            imported_skins.iter().find_map(|skin| {
                skin.joint_node_indices
                    .iter()
                    .position(|joint_index| *joint_index == parent_index)
                    .map(|joint_offset| {
                        (
                            skin.armature_node.clone(),
                            skin.joint_ids[joint_offset].clone(),
                        )
                    })
            })
        });
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
                materials,
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
    let mut losses = import_animations(&root, &binaries, &node_ids, &mut doc)?;
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
    doc.validate()?;
    Ok(crate::exchange::ImportedGraph {
        doc,
        losses,
        id_mappings: json!({}),
        compat_blobs,
        assets,
        source: json!({ "format": if is_glb { "glb" } else { "gltf" }, "path": path.display().to_string() }),
    })
}

/// Morph position deltas per welded vertex and morph NORMAL deltas per face corner, in Potter axes.
type MorphNormalTarget = (Vec<[f64; 3]>, Vec<[f64; 3]>);

/// Vertex-domain accessors of a glTF primitive. TEXCOORD is left out: UVs are stored per corner and
/// read from each primitive, so sharing a vertex cannot lose them.
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
            .filter(|(name, _)| !name.starts_with("TEXCOORD_"))
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

/// Derives per-face smooth flags and custom normals from imported glTF NORMAL data.
/// `normals` and each target's NORMAL deltas hold one value per face corner in `mesh.faces` order.
/// Returns true when morph NORMAL deltas disagree with normals derived from the morphed geometry.
fn import_shading(
    mesh: &mut Mesh,
    normals: &[DVec3],
    morph_targets: &[MorphNormalTarget],
) -> Result<bool> {
    const TOLERANCE: f64 = 1.0e-4;
    fn close(left: DVec3, right: DVec3) -> bool {
        (left - right).abs().max_element() <= TOLERANCE
    }
    let index_of = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<BTreeMap<_, _>>();
    let corners = mesh
        .faces
        .iter()
        .map(|face| {
            face.vertices
                .iter()
                .map(|id| {
                    index_of
                        .get(id)
                        .copied()
                        .ok_or_else(|| import_error("glTF face references a missing vertex"))
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    let points = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    let mut smooth = Vec::with_capacity(corners.len());
    let mut corner = 0;
    for face_corners in &corners {
        let mut area = DVec3::ZERO;
        for (offset, vertex) in face_corners.iter().enumerate() {
            let next = face_corners[(offset + 1) % face_corners.len()];
            area += points[*vertex].cross(points[next]);
        }
        let flat = area.normalize_or_zero();
        let end = corner + face_corners.len();
        smooth.push(
            normals[corner..end]
                .iter()
                .any(|normal| !close(*normal, flat)),
        );
        corner = end;
    }
    if smooth.iter().any(|value| *value) {
        let values = mesh
            .faces
            .iter()
            .zip(&smooth)
            .map(|(face, value)| (format!("f{}", face.id), json!(value)))
            .collect::<serde_json::Map<String, Value>>();
        mesh.attributes.insert(
            "shade_smooth".to_owned(),
            json!({ "domain": "face", "type": "bool", "values": values }),
        );
    }
    let computed = corner_normals(mesh, &points).ok();
    let mut custom_values = serde_json::Map::new();
    let mut corner = 0;
    for (face, face_corners) in mesh.faces.iter().zip(&corners) {
        let end = corner + face_corners.len();
        let mismatch = match &computed {
            Some(expected) => (corner..end).any(|index| !close(expected[index], normals[index])),
            None => true,
        };
        if mismatch {
            let values = normals[corner..end]
                .iter()
                .map(|normal| normal.normalize_or_zero().to_array())
                .collect::<Vec<_>>();
            custom_values.insert(format!("f{}", face.id), json!(values));
        }
        corner = end;
    }
    if !custom_values.is_empty() {
        mesh.attributes.insert(
            "custom_normal".to_owned(),
            json!({ "domain": "corner", "type": "float3", "values": custom_values }),
        );
    }
    let mut disagree = false;
    // The exporter's base layer includes the imported custom normals, so morph fallbacks come from it.
    let base = if morph_targets.is_empty() {
        None
    } else {
        corner_normals(mesh, &points).ok()
    };
    for (position_deltas, normal_deltas) in morph_targets {
        let morphed = mesh
            .vertices
            .iter()
            .zip(position_deltas)
            .map(|(vertex, delta)| vertex.co + DVec3::from_array(*delta))
            .collect::<Vec<_>>();
        let Some(expected) = base
            .as_deref()
            .and_then(|base| morph_corner_normals(mesh, &morphed, base).ok())
        else {
            disagree = true;
            continue;
        };
        for (corner, expected) in expected.iter().enumerate() {
            let implied =
                (normals[corner] + DVec3::from_array(normal_deltas[corner])).normalize_or_zero();
            if !close(implied, *expected) {
                disagree = true;
            }
        }
    }
    Ok(disagree)
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
            for key_index in 0..times.len() {
                let source_offset = key_index * dimensions * if cubic { 3 } else { 1 };
                let value_offset = source_offset + if cubic { dimensions } else { 0 };
                let mut value = values[value_offset..value_offset + dimensions].to_vec();
                if path == "transform.translation" {
                    let position = axis_from_gltf([value[0], value[1], value[2]]);
                    value.copy_from_slice(&position);
                } else if path == "transform.rotation_quaternion" {
                    let rotation = DQuat::from_xyzw(value[0], value[1], value[2], value[3]);
                    let basis = pot_to_gltf_rotation();
                    let potter_rotation = basis.conjugate() * rotation * basis;
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
    let dimensions: usize = match accessor["type"].as_str() {
        Some("SCALAR") => 1,
        Some("VEC2") => 2,
        Some("VEC3") => 3,
        Some("VEC4") => 4,
        Some("MAT4") => 16,
        _ => return Err(import_error("glTF animation accessor type is unsupported")),
    };
    let count = usize::try_from(
        accessor["count"]
            .as_u64()
            .ok_or_else(|| import_error("glTF animation accessor count is missing"))?,
    )
    .map_err(|_| import_error("glTF animation accessor count exceeds platform range"))?;
    let view_index = usize::try_from(
        accessor["bufferView"]
            .as_u64()
            .ok_or_else(|| import_error("glTF animation accessor bufferView is missing"))?,
    )
    .map_err(|_| import_error("glTF animation bufferView exceeds platform range"))?;
    let view = root["bufferViews"]
        .get(view_index)
        .ok_or_else(|| import_error("glTF animation bufferView is missing"))?;
    let buffer_index = usize::try_from(view["buffer"].as_u64().unwrap_or(0))
        .map_err(|_| import_error("glTF animation buffer index exceeds platform range"))?;
    let buffer = buffers
        .get(buffer_index)
        .ok_or_else(|| import_error("glTF animation buffer is missing"))?;
    let base = usize::try_from(view["byteOffset"].as_u64().unwrap_or(0))
        .map_err(|_| import_error("glTF animation buffer offset exceeds platform range"))?
        .checked_add(
            usize::try_from(accessor["byteOffset"].as_u64().unwrap_or(0)).map_err(|_| {
                import_error("glTF animation accessor offset exceeds platform range")
            })?,
        )
        .ok_or_else(|| import_error("glTF animation buffer offset overflow"))?;
    let element_size = dimensions
        .checked_mul(4)
        .ok_or_else(|| import_error("glTF animation element size overflow"))?;
    let stride = view["byteStride"]
        .as_u64()
        .map(|value| {
            usize::try_from(value)
                .map_err(|_| import_error("glTF animation stride exceeds platform range"))
        })
        .transpose()?
        .unwrap_or(element_size);
    if stride < element_size {
        return Err(import_error("glTF animation byteStride is too small"));
    }
    let capacity = count
        .checked_mul(dimensions)
        .ok_or_else(|| import_error("glTF animation value count overflow"))?;
    let mut values = Vec::with_capacity(capacity);
    for element in 0..count {
        let start = base
            .checked_add(
                element
                    .checked_mul(stride)
                    .ok_or_else(|| import_error("glTF animation offset overflow"))?,
            )
            .ok_or_else(|| import_error("glTF animation offset overflow"))?;
        for component in 0..dimensions {
            let offset = start
                .checked_add(component * 4)
                .ok_or_else(|| import_error("glTF animation offset overflow"))?;
            let raw = buffer
                .get(offset..offset + 4)
                .ok_or_else(|| import_error("glTF animation data is truncated"))?;
            let value = f64::from(f32::from_le_bytes(
                raw.try_into()
                    .map_err(|_| import_error("glTF animation data is truncated"))?,
            ));
            if !value.is_finite() {
                return Err(import_error("glTF animation contains non-finite values"));
            }
            values.push(value);
        }
    }
    Ok((values, dimensions))
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
            material["occlusionTexture"].is_object() || material["emissiveTexture"].is_object()
        })
    }) {
        losses.push(crate::exchange::Loss {
            feature_id: "material.texture".to_owned(),
            data_id: None,
            reason: "glTF occlusion and emissive textures are not represented by Potter materials"
                .to_owned(),
            suggestion: Some("use --allow-lossy or remove unsupported texture roles".to_owned()),
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
    let (values, dimensions) = read_float_accessor(root, buffers, accessor_index)?;
    if dimensions != 2 {
        return Err(import_error("glTF TEXCOORD accessor must be float VEC2"));
    }
    Ok(values
        .as_chunks::<2>()
        .0
        .iter()
        .map(|value| [value[0], value[1]])
        .collect())
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

#[cfg(test)]
mod tests {
    use proptest::{prelude::*, test_runner::TestCaseError};
    use serde_json::{Value, json};

    use crate::{
        eval::{EvaluationContext, Snapshot},
        exchange::ImportedGraph,
        model::SceneDoc,
        ops::{apply_batch, apply_batch_with_asset_root},
    };

    use super::{axis_from_gltf, axis_to_gltf, export_with_root, gltf_import_losses, import};

    type TestResult<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

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
        let applied = apply_batch_with_asset_root(
            &initial,
            &json!({
                "schema_version": 1,
                "base_revision": 0,
                "operations": operations
            }),
            root,
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
                {"op":"material.create","id":"surface","base_color_texture":{"image":"paint","uv_map":"uv_map","interpolation":"closest"}},
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
            .and_then(|mesh| mesh.attributes.get("uv_map"))
            .and_then(Value::as_array)
            .ok_or("source UV map is missing")?
            .iter()
            .flat_map(|face| face["uv"].as_array().into_iter().flatten())
            .map(|coordinate| -> TestResult<[f64; 2]> {
                Ok([
                    coordinate[0]
                        .as_f64()
                        .ok_or("source U coordinate is missing")?,
                    coordinate[1]
                        .as_f64()
                        .ok_or("source V coordinate is missing")?,
                ])
            })
            .collect::<TestResult<Vec<_>>>()?;
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
    fn gltf_rejects_unrepresented_mesh_attributes_by_name() -> TestResult {
        let mut mesh = crate::geom::Mesh::default();
        mesh.attributes
            .insert("color".to_owned(), json!({"domain":"point","values":{}}));
        let error = super::validate_mesh_export_attributes(&mesh, None)
            .err()
            .ok_or("unrepresented consumer attributes must remain unsupported")?;
        assert!(error.to_string().contains("color"));
        Ok(())
    }

    #[test]
    fn gltf_import_reports_each_unsupported_loss_class() {
        let losses = gltf_import_losses(&json!({
            "nodes": [{"mesh": 0, "weights": [1.0]}],
            "meshes": [{"weights": [0.0]}],
            "materials": [{"occlusionTexture": {"index": 0}}],
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
                "material.texture",
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
        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        let directory = tempfile::tempdir()?;
        assert!(
            export_with_root(&doc, &snapshot, directory.path(), Some("scene.bin")).is_err(),
            "out-of-range face material slot must not clamp"
        );
        Ok(())
    }

    #[test]
    fn baked_scale_normals_use_inverse_transpose() -> TestResult {
        let normal = glam::DVec3::new(1.0, 1.0, 0.0).normalize();
        let stretched = super::output_normals(
            &[normal],
            glam::DVec3::new(2.0, 1.0, 1.0),
            glam::DMat4::IDENTITY,
        )?[0];
        let expected = glam::DVec3::new(0.5, 1.0, 0.0).normalize();
        assert!((stretched - expected).length() < 1.0e-12, "{stretched:?}");
        let mirrored = super::output_normals(
            &[glam::DVec3::Y],
            glam::DVec3::new(-1.0, 1.0, 1.0),
            glam::DMat4::IDENTITY,
        )?[0];
        assert!(
            (mirrored - glam::DVec3::Y).length() < 1.0e-12,
            "{mirrored:?}"
        );
        Ok(())
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
        // The writer triangulates, so every imported face is a triangle of the smooth cube.
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
        // Quad faces are exported as triangles, and morph normals derived on quad topology differ
        // from triangle topology; a triangle source keeps the topology identical across the roundtrip.
        let source = scene(&json!([
            {"op":"node.create","id":"cube","kind":"box","params":{"size":2.0}},
            {"op":"mesh.triangulate","target":{"id":"cube"},"elements":{"domain":"face","ids":["f0","f1","f2","f3","f4","f5"]}},
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
    fn collection_instance_with_multi_material_faces_exports_one_primitive() -> TestResult {
        let doc = scene(&json!([
            {"op":"material.create","id":"jacket"},
            {"op":"material.create","id":"skin"},
            {"op":"collection.create","id":"source"},
            {"op":"node.create","id":"box_a","kind":"box","collection":"source","params":{"size":1.0}},
            {"op":"node.update","target":{"id":"box_a"},"set":{"materials":["jacket","skin"]}},
            {"op":"mesh.set_attribute","target":{"id":"box_a"},"elements":{"domain":"face","ids":["f1"]},"material_index":1},
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
        assert_eq!(primitives.len(), 1, "instances export every face on slot 0");
        assert!(
            primitives[0].get("material").is_none(),
            "the instancing node has no material slot"
        );
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
        // Triangles keep the topology the morph normals were derived on, so only the UV seams can differ.
        let base = scene(&json!([
            {"op":"node.create","id":"body","kind":"box","params":{"size":2.0}},
            {"op":"mesh.triangulate","target":{"id":"body"},"elements":{"domain":"face","ids":["f0","f1","f2","f3","f4","f5"]}}
        ]))?;
        let node_id = crate::model::Id::new("body")?;
        let data_id = base.nodes[&node_id]
            .data
            .clone()
            .ok_or("body mesh data is missing")?;
        let face_ids = base.data_blocks[&data_id]
            .mesh
            .as_ref()
            .ok_or("body mesh is missing")?
            .faces
            .iter()
            .map(|face| format!("f{}", face.id))
            .collect::<Vec<_>>();
        let doc = apply_operations(
            &base,
            &json!([
                {"op":"mesh.shade_smooth","target":{"id":"body"},"elements":{"domain":"face","ids":face_ids}},
                {"op":"uv.unwrap","target":{"id":"body"},"method":"smart"},
                {"op":"shape_key.create","target":{"id":"body"},"id":"lift","name":"Lift","positions":{"0":[-1.0,-1.0,1.5]}}
            ]),
        )?;
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
        let mesh = imported
            .doc
            .data_blocks
            .values()
            .find_map(|block| block.mesh.as_ref())
            .ok_or("imported mesh is missing")?;
        assert_eq!(
            mesh.vertices.len(),
            8,
            "UV seams must rejoin the source vertices"
        );
        assert!(
            !mesh.attributes.contains_key("custom_normal"),
            "UV seams must not create custom normals"
        );
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
    fn quad_smooth_seamed_shape_key_mesh_reports_triangulated_morph_normal_loss() -> TestResult {
        // Quads are exported as triangles, and their morph normals are derived on the quads.
        let doc = scene(&json!([
            {"op":"node.create","id":"body","kind":"box","params":{"size":2.0}},
            {"op":"mesh.shade_smooth","target":{"id":"body"},"elements":{"domain":"face","ids":["f0","f1","f2","f3","f4","f5"]}},
            {"op":"uv.unwrap","target":{"id":"body"},"method":"smart"},
            {"op":"shape_key.create","target":{"id":"body"},"id":"lift","name":"Lift","positions":{"0":[-1.0,-1.0,1.5]}}
        ]))?;
        let directory = tempfile::tempdir()?;
        let (imported, _) = round_trip_in(&doc, directory.path())?;
        assert!(
            imported
                .losses
                .iter()
                .any(|loss| loss.feature_id == "mesh.morph_normals"),
            "quad morph normals cannot be re-derived from triangles: {:?}",
            imported.losses
        );
        Ok(())
    }

    #[test]
    fn stale_custom_normals_after_topology_modifier_fail_export() -> TestResult {
        let doc = scene(&json!([
            {"op":"node.create","id":"cube","kind":"box","params":{"size":2.0}},
            {"op":"mesh.set_custom_normals","target":{"id":"cube"},"elements":{"domain":"face","ids":["f0"]},"normals":{"f0":[[0.0,0.0,1.0],[0.0,0.0,1.0],[0.0,0.0,1.0],[0.0,0.0,1.0]]}},
            {"op":"modifier.create","target":{"id":"cube"},"id":"subdivide","type":"subdivision","params":{"levels":1}}
        ]))?;
        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        let directory = tempfile::tempdir()?;
        let error = export_with_root(&doc, &snapshot, directory.path(), Some("scene.bin"))
            .err()
            .ok_or("export must reject stale custom normals")?;
        assert_eq!(
            error.code,
            crate::error::ErrorCode::UnsupportedFeature,
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn applying_topology_modifier_over_shading_is_refused() -> TestResult {
        let base = scene(&json!([
            {"op":"node.create","id":"cube","kind":"box","params":{"size":2.0}},
            {"op":"mesh.shade_smooth","target":{"id":"cube"},"elements":{"domain":"face","ids":["f0","f1","f2","f3","f4","f5"]}},
            {"op":"modifier.create","target":{"id":"cube"},"id":"subdivide","type":"subdivision","params":{"levels":1}}
        ]))?;
        let error = apply_batch(
            &base,
            &json!({
                "schema_version": 1,
                "base_revision": base.revision,
                "operations": [{"op":"modifier.apply","target":{"id":"cube"},"modifier_id":"subdivide"}]
            }),
        )
        .err()
        .ok_or("applying the modifier must be refused")?;
        assert_eq!(
            error.code,
            crate::error::ErrorCode::UnsupportedFeature,
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn collection_instance_member_with_topology_modifier_refuses_stale_shading() -> TestResult {
        // The hidden source's subdivision adds faces, so its face-keyed smooth flags would land on other faces.
        let doc = scene(&json!([
            {"op":"collection.create","id":"source"},
            {"op":"node.create","id":"member","kind":"box","collection":"source","params":{"size":1.0}},
            {"op":"mesh.shade_smooth","target":{"id":"member"},"elements":{"domain":"face","ids":["f0","f1","f2","f3","f4","f5"]}},
            {"op":"modifier.create","target":{"id":"member"},"id":"subdivide","type":"subdivision","params":{"levels":1}},
            {"op":"collection.update","target":{"id":"source"},"set":{"exclude":true}},
            {"op":"collection.instance_create","id":"instance","collection":"source","transform":{"translation":[4.0,0.0,0.0]}}
        ]))?;
        let error = Snapshot::evaluate(&doc, &EvaluationContext::default())
            .err()
            .ok_or("instancing stale shading must be refused")?;
        assert_eq!(
            error.code,
            crate::error::ErrorCode::UnsupportedFeature,
            "{error}"
        );
        assert_eq!(error.details["feature_id"], "mesh.shading_topology");
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
}
