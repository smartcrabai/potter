use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    fs,
    path::Path,
};

use glam::{DMat4, DQuat, DVec3};
use serde_json::{Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    eval::Snapshot,
    geom::Mesh,
    image::{ImageData, ImageInterpolation, load_image_data},
    model::{
        Action, ArmatureData, Bone, CameraData, CameraProjection, DataBlock, Extrapolation, FCurve,
        Id, Image, ImageAlphaMode, ImageColorspace, ImageSource, Interpolation, Keyframe,
        LightData, LightType, Material, Modifier, Node, PoseBone, SceneDoc, ShapeKey, ShapeKeyData,
        TextureRef, Transform, VertexGroup,
    },
};

use super::import_error;

const POT_TO_GLTF: DMat4 = DMat4::from_cols_array(&[
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
]);

struct TextureExport {
    images: Vec<Value>,
    textures: Vec<Value>,
    samplers: Vec<Value>,
    image_indices: BTreeMap<Id, usize>,
    texture_indices: BTreeMap<(Id, bool), usize>,
    sampler_indices: BTreeMap<bool, usize>,
}

impl TextureExport {
    fn new() -> Self {
        Self {
            images: Vec::new(),
            textures: Vec::new(),
            samplers: Vec::new(),
            image_indices: BTreeMap::new(),
            texture_indices: BTreeMap::new(),
            sampler_indices: BTreeMap::new(),
        }
    }

    fn texture_info(
        &mut self,
        reference: &TextureRef,
        doc: &SceneDoc,
        root_path: &Path,
        binary: &mut Vec<u8>,
        views: &mut Vec<Value>,
    ) -> Result<Value> {
        let image_index = if let Some(index) = self.image_indices.get(&reference.image) {
            *index
        } else {
            let image = doc.images.get(&reference.image).ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "material texture references a missing image",
                    json!({ "image": reference.image }),
                )
            })?;
            if !image.tiles.is_empty() {
                return Err(unsupported_export(
                    "material.texture_udim",
                    "glTF export does not represent Potter UDIM image tiles",
                ));
            }
            let pixels = load_image_data(image, root_path, reference.interpolation)?;
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
                "extras": { "potter": { "id": reference.image } }
            }));
            self.image_indices.insert(reference.image.clone(), index);
            index
        };
        let nearest = reference.interpolation == ImageInterpolation::Closest;
        let texture_key = (reference.image.clone(), nearest);
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

fn encode_png(image: &Image, pixels: &ImageData) -> Result<Vec<u8>> {
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
            let component = if index < 3 && image.colorspace == ImageColorspace::Srgb {
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

fn unsupported_export(feature_id: &str, reason: &str) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        reason,
        json!({ "feature_id": feature_id, "status": "not_supported", "reason": reason }),
    )
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

fn append_skin_attributes(
    data_block: &DataBlock,
    mesh: &Mesh,
    binding: &SkinBinding,
    doc: &SceneDoc,
    binary: &mut Vec<u8>,
    views: &mut Vec<Value>,
    accessors: &mut Vec<Value>,
) -> Result<Vec<(usize, usize)>> {
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
    let mut vertex_influences = Vec::<Vec<(usize, f64)>>::with_capacity(mesh.vertices.len());
    let mut max_influences = 0_usize;
    for vertex in &mesh.vertices {
        let mut influences = Vec::<(usize, f64)>::new();
        if let Some(vertex_weights) = data_block.vertex_weights.get(&vertex.id) {
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
        max_influences = max_influences.max(influences.len());
        vertex_influences.push(influences);
    }
    let set_count = max_influences.div_ceil(4).max(1);
    let mut joint_sets = (0..set_count)
        .map(|_| Vec::<[u16; 4]>::with_capacity(mesh.vertices.len()))
        .collect::<Vec<_>>();
    let mut weight_sets = (0..set_count)
        .map(|_| Vec::<[f32; 4]>::with_capacity(mesh.vertices.len()))
        .collect::<Vec<_>>();
    for influences in &vertex_influences {
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

fn vertex_uvs(mesh: &Mesh) -> Result<Option<Vec<[f64; 2]>>> {
    let Some(entries) = mesh.attributes.get("uv_map").and_then(Value::as_array) else {
        return Ok(None);
    };
    let faces = mesh
        .faces
        .iter()
        .map(|face| (face.id, face))
        .collect::<BTreeMap<_, _>>();
    let mut values = BTreeMap::<u32, [f64; 2]>::new();
    for entry in entries {
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
            if let Some(previous) = values.get(vertex_id) {
                if !crate::float::equal_f64_array(previous, &uv) {
                    return Err(unsupported_export(
                        "mesh.uv_seams",
                        "glTF texture export does not yet split vertices at UV seams",
                    ));
                }
            } else {
                values.insert(*vertex_id, uv);
            }
        }
    }
    Ok(Some(
        mesh.vertices
            .iter()
            .map(|vertex| values.get(&vertex.id).copied().unwrap_or([0.0, 0.0]))
            .collect(),
    ))
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

fn skin_definitions(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    node_indices: &BTreeMap<Id, usize>,
    skin_bindings: &BTreeMap<Id, SkinBinding>,
    nodes: &mut Vec<Value>,
    binary: &mut Vec<u8>,
    views: &mut Vec<Value>,
    accessors: &mut Vec<Value>,
) -> Result<(Vec<Value>, BTreeMap<Id, usize>)> {
    let inverse_basis = POT_TO_GLTF.inverse();
    let armature_ids = skin_bindings
        .values()
        .map(|binding| binding.armature_node.clone())
        .collect::<BTreeSet<_>>();
    let mut bone_indices = BTreeMap::<Id, BTreeMap<Id, usize>>::new();
    for armature_node_id in armature_ids {
        let armature_node = doc.nodes.get(&armature_node_id).ok_or_else(|| {
            PotError::new(ErrorCode::SceneInvalid, "skin armature object is missing")
        })?;
        let data_id = armature_node.data.as_ref().ok_or_else(|| {
            PotError::new(
                ErrorCode::SceneInvalid,
                "skin armature Data-Block is missing",
            )
        })?;
        let armature_data = doc.data_blocks.get(data_id).ok_or_else(|| {
            PotError::new(
                ErrorCode::SceneInvalid,
                "skin armature Data-Block is missing",
            )
        })?;
        let armature = armature_data.armature.as_ref().ok_or_else(|| {
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
            let local = parent_inverse * bone_world;
            if !finite_matrix(local) {
                return Err(unsupported_export(
                    "rig.skin_singular_bone",
                    "glTF skin export requires invertible bone parent transforms",
                ));
            }
            let converted = POT_TO_GLTF * local * inverse_basis;
            let matrix = converted
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
                append_child(nodes, indices[parent_id], child_index);
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
        let mesh_world = snapshot
            .nodes
            .get(mesh_node_id)
            .map(|node| DMat4::from_cols_array(&node.world_matrix))
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::EvaluationFailed,
                    "skinned mesh transform is missing",
                )
            })?;
        let node_bones = bone_indices.get(&binding.armature_node).ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "skin bone-node indices are missing",
            )
        })?;
        let mut inverse_bind_matrices = Vec::with_capacity(binding.joint_ids.len());
        let mut joints = Vec::with_capacity(binding.joint_ids.len());
        for bone_id in &binding.joint_ids {
            let joint_world = matrix_map
                .get(bone_id)
                .map(DMat4::from_cols_array)
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::EvaluationFailed,
                        "skin joint world transform is missing",
                    )
                })?;
            let joint_inverse = joint_world.inverse();
            let inverse_bind = joint_inverse * mesh_world;
            if !finite_matrix(inverse_bind) {
                return Err(unsupported_export(
                    "rig.skin_singular_bind",
                    "glTF inverse bind matrices require invertible joint transforms",
                ));
            }
            inverse_bind_matrices.push(POT_TO_GLTF * inverse_bind * inverse_basis);
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
    Ok((skins, node_skins))
}

pub(crate) fn export_with_root(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    root_path: &Path,
    bin_uri: Option<&str>,
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
        material_indices.insert(id.clone(), materials.len());
        let mut extensions = serde_json::Map::new();
        extensions.insert(
            "KHR_materials_emissive_strength".to_owned(),
            json!({ "emissiveStrength": material.emission_strength }),
        );
        extensions_used.insert("KHR_materials_emissive_strength".to_owned());
        if !crate::float::equal_f64(material.transmission, 0.0) {
            extensions.insert(
                "KHR_materials_transmission".to_owned(),
                json!({ "transmissionFactor": material.transmission }),
            );
            extensions_used.insert("KHR_materials_transmission".to_owned());
        }
        if !crate::float::equal_f64(material.ior, 1.5) {
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
        value["extensions"] = Value::Object(extensions);
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
    let mut mesh_indices = BTreeMap::<Id, usize>::new();
    let mut instance_mesh_indices = BTreeMap::<Id, usize>::new();
    let mut node_mesh_indices = BTreeMap::<Id, usize>::new();
    let mut skin_bindings = BTreeMap::<Id, SkinBinding>::new();
    let mut data_skin_bindings = BTreeMap::<Id, SkinBinding>::new();
    for (node_id, node) in &doc.nodes {
        if !node.visible || !node.render_visible || !visible_node_ids.contains(node_id) {
            continue;
        }
        let Some(data_id) = &node.data else {
            continue;
        };
        let Some(data_block) = doc
            .data_blocks
            .get(data_id)
            .filter(|data| data.data_type == "mesh")
        else {
            continue;
        };
        let Some(binding) = skin_binding(doc, node)? else {
            continue;
        };
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
        if data_block
            .shape_keys
            .as_ref()
            .is_some_and(|shape_keys| !shape_keys.keys.is_empty())
        {
            let armature = doc.nodes.get(&binding.armature_node).ok_or_else(|| {
                PotError::new(ErrorCode::SceneInvalid, "skin armature node is missing")
            })?;
            let rest_pose = armature.action.is_none()
                && armature.nla_tracks.is_empty()
                && armature.constraints.is_empty()
                && armature.drivers.is_empty()
                && armature
                    .pose
                    .values()
                    .all(|pose| *pose == PoseBone::default());
            if !rest_pose {
                return Err(unsupported_export(
                    "rig.skin_shape_key_pose",
                    "glTF shape-key skin export currently requires an unanimated armature at rest",
                ));
            }
        }
        if let Some(previous) = data_skin_bindings.get(data_id)
            && (previous.armature_node != binding.armature_node
                || previous.joint_ids != binding.joint_ids)
        {
            return Err(unsupported_export(
                "rig.shared_mesh_multiple_skeletons",
                "one shared mesh Data-Block cannot use different glTF skin joint layouts",
            ));
        }
        data_skin_bindings
            .entry(data_id.clone())
            .or_insert_with(|| binding.clone());
        skin_bindings.insert(node_id.clone(), binding);
    }
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
        } else {
            let Some(value) = node.properties.get("instance_collection") else {
                continue;
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
            (evaluated, None, Some(target), None)
        };
        if mesh.attributes.keys().any(|name| name != "uv_map") {
            return Err(unsupported_export(
                "mesh.vertex_attributes",
                "glTF export does not preserve this Potter mesh attribute",
            ));
        }
        if let Some(data_id) = data_id
            && let Some(index) = mesh_indices.get(data_id)
        {
            node_mesh_indices.insert(node_id.clone(), *index);
            continue;
        }
        if let Some(target) = &instance_target
            && let Some(index) = instance_mesh_indices.get(target)
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
        let vertex_indices = mesh
            .vertices
            .iter()
            .enumerate()
            .map(|(index, vertex)| (vertex.id, index))
            .collect::<BTreeMap<_, _>>();
        let mut positions = Vec::with_capacity(mesh.vertices.len());
        let mut minimum = [f32::INFINITY; 3];
        let mut maximum = [f32::NEG_INFINITY; 3];
        for vertex in &mesh.vertices {
            let source_point = data_block
                .and_then(|data| data.shape_keys.as_ref())
                .filter(|shape_keys| !shape_keys.keys.is_empty())
                .and_then(|shape_keys| shape_keys.basis.get(&vertex.id))
                .map_or(vertex.co, |point| DVec3::from_array(*point));
            let point = POT_TO_GLTF.transform_point3(source_point);
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
        let uv_values = vertex_uvs(mesh)?;
        let material_id = node.materials.first().cloned();
        if let Some(material_id) = &material_id {
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
                if texture
                    .uv_map
                    .as_deref()
                    .is_some_and(|name| name != "uv_map")
                {
                    return Err(unsupported_export(
                        "material.texture_uv_map",
                        "glTF export supports only the Potter default UV map",
                    ));
                }
                if uv_values.is_none() {
                    return Err(unsupported_export(
                        "material.texture_uv_map",
                        "textured materials require UV coordinates on the exported mesh",
                    ));
                }
            }
        }
        let mut attributes = json!({ "POSITION": position_accessor });
        if let Some(uv_values) = &uv_values {
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
        if let (Some(data_id), Some(data_block)) = (data_id, data_block)
            && let Some(binding) = data_skin_bindings.get(data_id)
        {
            let accessors_for_sets = append_skin_attributes(
                data_block,
                mesh,
                binding,
                doc,
                &mut binary,
                &mut views,
                &mut accessors,
            )?;
            for (set_index, (joints_accessor, weights_accessor)) in
                accessors_for_sets.into_iter().enumerate()
            {
                attributes[format!("JOINTS_{set_index}")] = json!(joints_accessor);
                attributes[format!("WEIGHTS_{set_index}")] = json!(weights_accessor);
            }
        }
        let triangles = mesh.triangulate().map_err(|error| {
            PotError::with_details(
                ErrorCode::ExportFailed,
                "mesh triangulation failed",
                json!({ "reason": error.to_string(), "data_id": data_id }),
            )
        })?;
        align4(&mut binary);
        let index_offset = binary.len();
        for triangle in &triangles {
            for id in triangle {
                let index = vertex_indices.get(id).copied().ok_or_else(|| {
                    PotError::new(
                        ErrorCode::SceneInvalid,
                        "triangle references missing vertex",
                    )
                })?;
                let index = u32::try_from(index).map_err(|_| {
                    PotError::new(ErrorCode::LimitExceeded, "glTF mesh has too many vertices")
                })?;
                binary.extend_from_slice(&index.to_le_bytes());
            }
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
            "count": triangles.len().checked_mul(3).ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, "glTF index count overflow"))?,
            "type": "SCALAR"
        }));
        let mut primitive =
            json!({ "attributes": attributes, "indices": index_accessor, "mode": 4 });
        if let Some(material_id) = &material_id {
            let material_index = material_indices.get(material_id).copied().ok_or_else(|| {
                PotError::new(
                    ErrorCode::SceneInvalid,
                    "node material reference is missing",
                )
            })?;
            primitive["material"] = json!(material_index);
        }
        let mut target_names = Vec::new();
        let mut target_metadata = Vec::new();
        let mut mesh_weights = Vec::new();
        if let Some(shape_keys) = data_block
            .and_then(|data| data.shape_keys.as_ref())
            .filter(|shape_keys| !shape_keys.keys.is_empty())
        {
            if shape_keys.absolute {
                return Err(unsupported_export(
                    "mesh.shape_key_absolute",
                    "glTF morph targets cannot preserve Potter absolute shape-key timing",
                ));
            }
            let basis = shape_keys
                .basis
                .iter()
                .map(|(vertex_id, point)| (*vertex_id, DVec3::from_array(*point)))
                .collect::<BTreeMap<_, _>>();
            let mut targets = Vec::with_capacity(shape_keys.keys.len());
            for (key_id, key) in &shape_keys.keys {
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
                let mut deltas = Vec::with_capacity(mesh.vertices.len().saturating_mul(3));
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
                        let group_exists = data_block.is_some_and(|block| {
                            block
                                .vertex_groups
                                .iter()
                                .any(|group| &group.id == group_id)
                        });
                        if !group_exists {
                            return Err(PotError::new(
                                ErrorCode::SceneInvalid,
                                "shape-key vertex group does not exist",
                            ));
                        }
                        let weight = data_block
                            .and_then(|block| block.vertex_weights.get(&vertex.id))
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
                    let delta = POT_TO_GLTF.transform_vector3(destination - relative) * mask;
                    if !delta.is_finite() {
                        return Err(PotError::new(
                            ErrorCode::SceneInvalid,
                            "shape-key delta is non-finite",
                        ));
                    }
                    deltas.extend([to_f32(delta.x)?, to_f32(delta.y)?, to_f32(delta.z)?]);
                }
                let accessor = append_float_accessor(
                    &mut binary,
                    &mut views,
                    &mut accessors,
                    &deltas,
                    mesh.vertices.len(),
                    "VEC3",
                    None,
                    None,
                );
                targets.push(json!({ "POSITION": accessor }));
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
            primitive["targets"] = json!(targets);
        }
        let mesh_index = meshes.len();
        let mut mesh_value = json!({
            "name": mesh_name,
            "primitives": [primitive],
            "extras": { "potter": {
                "data_id": data_id,
                "instance_collection": instance_target,
                "shape_keys": target_metadata,
                "shape_key_data": data_block.and_then(|block| block.shape_keys.as_ref()),
                "vertex_ids": mesh.vertices.iter().map(|vertex| vertex.id).collect::<Vec<_>>(),
                "vertex_groups": data_block.map(|block| &block.vertex_groups),
                "vertex_weights": data_block.map(|block| &block.vertex_weights)
            } }
        });
        if !target_names.is_empty() {
            mesh_value["weights"] = json!(mesh_weights);
            mesh_value["extras"]["targetNames"] = json!(target_names);
        }
        meshes.push(mesh_value);
        if let Some(data_id) = data_id {
            mesh_indices.insert(data_id.clone(), mesh_index);
        }
        if let Some(target) = instance_target {
            instance_mesh_indices.insert(target, mesh_index);
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
        let transform = crate::eval::animation::animated_transform(node, doc, snapshot.frame)?;
        let mut gltf_node = if node.action.is_some() && node.parent_inverse.is_none() {
            let basis_rotation = DQuat::from_rotation_x(std::f64::consts::FRAC_PI_2);
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
    let (skins, node_skins) = skin_definitions(
        doc,
        snapshot,
        &node_indices,
        &skin_bindings,
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
        let parent_index = *node_indices
            .get(parent)
            .ok_or_else(|| PotError::new(ErrorCode::SceneInvalid, "node parent is missing"))?;
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
        root["textures"] = json!(texture_export.textures);
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
            let basis = DQuat::from_rotation_x(std::f64::consts::FRAC_PI_2);
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
    let vertex_ids = vertex_order.iter().copied().collect::<BTreeSet<_>>();
    if vertex_ids.len() != joint_weights.len() {
        return Err(import_error(
            "glTF skin weight count does not match mesh vertices",
        ));
    }
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
            if !vertex_ids.contains(vertex_id) {
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
        exact
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
            let source_vertex_ids = mesh_value["extras"]["potter"]["vertex_ids"]
                .as_array()
                .ok_or_else(|| import_error("glTF shape-key vertex IDs are missing"))?
                .iter()
                .map(|value| {
                    value
                        .as_u64()
                        .and_then(|id| u32::try_from(id).ok())
                        .ok_or_else(|| import_error("glTF shape-key vertex ID is invalid"))
                })
                .collect::<Result<Vec<_>>>()?;
            if source_vertex_ids.len() != mesh.vertices.len() {
                return Err(import_error(
                    "glTF shape-key vertex ID count does not match mesh positions",
                ));
            }
            let vertex_id_map = source_vertex_ids
                .into_iter()
                .zip(&mesh.vertices)
                .map(|(source_id, vertex)| (source_id, vertex.id))
                .collect::<BTreeMap<_, _>>();
            let remap_positions = |positions: BTreeMap<u32, [f64; 3]>| {
                positions
                    .into_iter()
                    .map(|(source_id, position)| {
                        vertex_id_map
                            .get(&source_id)
                            .copied()
                            .map(|vertex_id| (vertex_id, position))
                            .ok_or_else(|| {
                                import_error("glTF shape-key metadata references a missing vertex")
                            })
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
            let base = positions.len();
            positions.extend(
                decoded
                    .iter()
                    .map(|point| DVec3::from_array(axis_from_gltf(*point))),
            );
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
            }
            for (target_index, target) in target_values.iter().enumerate() {
                if target["NORMAL"].is_object() || target["TANGENT"].is_object() {
                    return Err(unsupported_export(
                        "mesh.morph_normals",
                        "glTF morph normal and tangent targets are not represented by Potter shape keys",
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
                morph_deltas[target_index]
                    .extend(values.iter().map(|delta| axis_from_gltf(*delta)));
            }
            if let Some(values) = read_skin_attributes(primitive, &root, &binaries, decoded.len())?
            {
                joint_data.extend(values.into_iter().map(Some));
            } else {
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
        let mut mesh = Mesh::from_positions_and_faces(positions, polygons).map_err(|error| {
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
        let shape_keys = if morph_deltas.is_empty() {
            None
        } else {
            Some(import_shape_keys(mesh_value, &mesh, &morph_deltas)?)
        };
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
        let parent = parents[index]
            .filter(|parent| !joint_nodes.contains(parent))
            .map(|parent| node_ids[parent].clone());
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
                    let basis = DQuat::from_rotation_x(std::f64::consts::FRAC_PI_2);
                    let potter_rotation = basis.conjugate() * rotation * basis;
                    value.copy_from_slice(&[
                        potter_rotation.x,
                        potter_rotation.y,
                        potter_rotation.z,
                        potter_rotation.w,
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
            let mut expected = as_case(mesh_positions(&source))?;
            let mut actual = as_case(mesh_positions(&imported.doc))?;
            let order = |left: &[f64; 3], right: &[f64; 3]| {
                left[0].total_cmp(&right[0])
                    .then(left[1].total_cmp(&right[1]))
                    .then(left[2].total_cmp(&right[2]))
            };
            expected.sort_by(order);
            actual.sort_by(order);
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
    fn export_import_round_trips_skin_morph_texture_animations_cameras_lights_and_instances()
    -> TestResult {
        let directory = tempfile::tempdir()?;
        let source = scene_with_asset_root(
            directory.path(),
            &json!([
                {"op":"image.create","id":"paint","width":1,"height":1,"colorspace":"srgb","fill_color":[0.25,0.5,0.75,1.0]},
                {"op":"material.create","id":"surface","base_color_texture":{"image":"paint","uv_map":"uv_map","interpolation":"closest"}},
                {"op":"node.create","id":"skin_mesh","kind":"plane","params":{"size":2.0},"material":"surface"},
                {"op":"uv.unwrap","target":{"id":"skin_mesh"},"method":"smart"},
                {"op":"shape_key.create","target":{"id":"skin_mesh"},"id":"lift","name":"Lift","positions":{"0":[0.0,0.0,1.0]}},
                {"op":"node.create","id":"arm","kind":"armature"},
                {"op":"bone.create","target":{"id":"arm"},"id":"root","name":"Root","head":[0.0,0.0,0.0],"tail":[0.0,1.0,0.0]},
                {"op":"vertex_group.create","target":{"id":"skin_mesh"},"id":"root_group","name":"Root"},
                {"op":"vertex_group.assign","target":{"id":"skin_mesh"},"group_id":"root_group","weights":[
                    {"vertex_id":0,"weight":1.0},
                    {"vertex_id":1,"weight":0.5},
                    {"vertex_id":2,"weight":1.0},
                    {"vertex_id":3,"weight":0.25}
                ]},
                {"op":"modifier.create","target":{"id":"skin_mesh"},"id":"skin","type":"armature","params":{"object":"arm","use_vertex_groups":true}},
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
}
