//! VRM import: decodes the VRM 1.0 (`VRMC_vrm` family) or VRM 0.x (`VRM`) extension over the glTF
//! importer. The glTF importer restores meshes, skins, morph targets, materials and textures with
//! Potter axes mapped through the avatar basis; this module maps the avatar extension back into
//! `AvatarSettings`, undoing the exporter's frame transforms, and reports what Potter cannot represent.

mod expression;
mod humanoid;
mod look;
mod meta;
mod mtoon;
mod spring;

use std::{collections::BTreeMap, path::Path};

use glam::{DMat4, DVec3};
use serde_json::{Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    eval::{EvaluationContext, Snapshot},
    exchange::{ImportedGraph, Loss, gltf::GltfIndex},
    model::{
        AvatarMeta, AvatarNodeRef, AvatarSettings, AvatarSpringBone, AvatarSpringColliderGroup,
        AvatarVersion, Id, SceneDoc, validate_avatar_settings, validate_spring,
    },
};

use super::super::gltf;
use super::uniform_scale;

/// Tolerance of the importer's uniform-frame check, relative to the scale. glTF frames are f32, and
/// Blender and its VRM Add-on re-export them with axis errors up to about 1e-5; 1e-4 accepts those
/// and still rejects real shear or non-uniform scale.
const F32_FRAME_TOLERANCE: f64 = 1e-4;

/// glTF extensions that the glTF importer reads as Potter material or light data.
const GLTF_SUPPORTED_EXTENSIONS: &[&str] = &[
    "KHR_lights_punctual",
    "KHR_materials_emissive_strength",
    "KHR_materials_transmission",
    "KHR_materials_ior",
    "KHR_texture_transform",
];

/// Imports a `.vrm` file as the given VRM version into a new scene graph.
pub fn import(path: &Path, scene_id: String, version: AvatarVersion) -> Result<ImportedGraph> {
    let (mut graph, index) = gltf::import_indexed(
        path,
        scene_id,
        gltf::avatar_basis(version),
        |root, material_index| match version {
            AvatarVersion::V0 => mtoon::v0_main_texture_transform(root, material_index),
            AvatarVersion::V1 => None,
        },
    )?;
    let extension_name = match version {
        AvatarVersion::V1 => "VRMC_vrm",
        AvatarVersion::V0 => "VRM",
    };
    let extension = index.root["extensions"][extension_name].clone();
    if !extension.is_object() {
        return Err(PotError::with_details(
            ErrorCode::ImportFailed,
            format!("glTF file has no {extension_name} extension"),
            json!({ "extension": extension_name }),
        ));
    }
    // The generic glTF extension loss would name VRM extensions Potter decodes; replace it with precise reports.
    let mut losses = graph
        .losses
        .drain(..)
        .filter(|loss| loss.feature_id != "gltf.extension")
        .collect::<Vec<_>>();
    losses.extend(unsupported_extension_losses(&index.root, version));
    apply_material_render_state(&mut graph.doc, &index);
    let snapshot = Snapshot::evaluate(&graph.doc, &EvaluationContext::default())?;
    let mut decoder = Decoder::new(version, &index, &graph.doc, &snapshot, &extension);
    let avatar = decode_avatar(&mut decoder)?;
    losses.append(&mut decoder.losses);
    if let Some(avatar) = avatar {
        graph.doc.avatar = accept_components(&graph.doc, avatar, &mut losses);
    }
    graph.doc.validate()?;
    graph.losses = losses;
    if let Some(source) = graph.source.as_object_mut() {
        source.insert(
            "format".to_owned(),
            json!(match version {
                AvatarVersion::V1 => "vrm",
                AvatarVersion::V0 => "vrm0",
            }),
        );
    }
    Ok(graph)
}

/// Decodes the avatar; `None` means humanoid mapping failed, which is reported as a loss.
fn decode_avatar(decoder: &mut Decoder<'_>) -> Result<Option<AvatarSettings>> {
    let meta: AvatarMeta = meta::decode(decoder)?;
    let Some((armature, humanoid)) = humanoid::decode(decoder)? else {
        return Ok(None);
    };
    decoder.armature = Some(armature.clone());
    decoder.humanoid = humanoid.clone();
    let first_person = look::first_person(decoder)?;
    let look_at = look::look_at(decoder)?;
    let mtoon = mtoon::decode(decoder)?;
    let expressions = expression::decode(decoder, &mtoon)?;
    let spring_bone = spring::decode(decoder)?;
    Ok(Some(AvatarSettings {
        armature,
        humanoid,
        meta,
        expressions,
        first_person,
        look_at,
        spring_bone,
        mtoon,
    }))
}

/// One loss per glTF extension that neither the glTF importer nor the VRM decoder represents.
fn unsupported_extension_losses(root: &Value, version: AvatarVersion) -> Vec<Loss> {
    let decoded: &[&str] = match version {
        AvatarVersion::V1 => &["VRMC_vrm", "VRMC_springBone", "VRMC_materials_mtoon"],
        AvatarVersion::V0 => &["VRM"],
    };
    let mut losses = root["extensionsUsed"]
        .as_array()
        .map_or(&[][..], Vec::as_slice)
        .iter()
        .filter_map(Value::as_str)
        .filter(|name| {
            !GLTF_SUPPORTED_EXTENSIONS.contains(name)
                && !decoded.contains(name)
                && *name != "KHR_materials_unlit"
        })
        .map(|name| Loss {
            feature_id: format!("vrm.extension.{name}"),
            data_id: None,
            reason: format!("glTF extension `{name}` is not represented by Potter"),
            suggestion: Some(
                "use --allow-lossy to import the file without this extension".to_owned(),
            ),
        })
        .collect::<Vec<_>>();
    // The VRM Add-on writes KHR_materials_unlit as a fallback on MToon materials, which Potter reads as MToon.
    if has_unlit_material_without_mtoon(root, version) {
        losses.push(Loss {
            feature_id: "vrm.extension.KHR_materials_unlit".to_owned(),
            data_id: None,
            reason: "glTF extension `KHR_materials_unlit` on a material without MToon is not represented by Potter".to_owned(),
            suggestion: Some("use --allow-lossy to import the material as lit".to_owned()),
        });
    }
    losses
}

fn has_unlit_material_without_mtoon(root: &Value, version: AvatarVersion) -> bool {
    let materials = root["materials"].as_array().map_or(&[][..], Vec::as_slice);
    let mtoon_shaders = root["extensions"]["VRM"]["materialProperties"]
        .as_array()
        .map_or(&[][..], Vec::as_slice);
    materials.iter().enumerate().any(|(index, material)| {
        if !material["extensions"]["KHR_materials_unlit"].is_object() {
            return false;
        }
        match version {
            AvatarVersion::V1 => !material["extensions"]["VRMC_materials_mtoon"].is_object(),
            AvatarVersion::V0 => {
                mtoon_shaders
                    .get(index)
                    .and_then(|property| property["shader"].as_str())
                    != Some("VRM/MToon")
            }
        }
    })
}

/// Copies glTF alpha and double-sided state onto the imported materials; VRM files carry them for `MToon`.
fn apply_material_render_state(doc: &mut SceneDoc, index: &GltfIndex) {
    let materials = index.root["materials"]
        .as_array()
        .map_or(&[][..], Vec::as_slice);
    for (material_index, id) in index.material_ids.iter().enumerate() {
        let Some(value) = materials.get(material_index) else {
            continue;
        };
        for id in std::iter::once(id).chain(
            index
                .color_material_variants
                .get(id)
                .into_iter()
                .flat_map(|variants| variants.values()),
        ) {
            let Some(material) = doc.materials.get_mut(id) else {
                continue;
            };
            match value["alphaMode"].as_str() {
                Some("MASK") => {
                    "clip".clone_into(&mut material.alpha_mode);
                    material.alpha_threshold = value["alphaCutoff"].as_f64().unwrap_or(0.5);
                }
                Some("BLEND") => "blend".clone_into(&mut material.alpha_mode),
                _ => "opaque".clone_into(&mut material.alpha_mode),
            }
            material.double_sided = value["doubleSided"].as_bool().unwrap_or(false);
        }
    }
}

/// Shared state while decoding one VRM extension over the glTF import.
///
/// Decoders report unrepresentable data through [`Decoder::loss`] and skip that element; they return
/// `Err` only when the file itself is malformed.
pub(super) struct Decoder<'a> {
    pub(super) version: AvatarVersion,
    pub(super) index: &'a GltfIndex,
    /// The imported scene before the avatar is attached.
    pub(super) doc: &'a SceneDoc,
    /// Evaluated `doc`: Potter-space world matrices of nodes and bones.
    pub(super) snapshot: &'a Snapshot,
    /// The VRM extension object: `VRMC_vrm` for VRM 1.0 or `VRM` for VRM 0.x.
    pub(super) extension: &'a Value,
    /// Armature node that owns the humanoid bones; set once humanoid mapping succeeds.
    pub(super) armature: Option<Id>,
    /// Humanoid role to armature bone; set once humanoid mapping succeeds.
    pub(super) humanoid: BTreeMap<String, Id>,
    pub(super) losses: Vec<Loss>,
    /// Potter node ID to the glTF node index that carries it.
    potter_node_index: BTreeMap<Id, usize>,
    /// (armature node, bone) to the glTF joint index.
    joint_index: BTreeMap<(Id, Id), usize>,
}

impl<'a> Decoder<'a> {
    fn new(
        version: AvatarVersion,
        index: &'a GltfIndex,
        doc: &'a SceneDoc,
        snapshot: &'a Snapshot,
        extension: &'a Value,
    ) -> Self {
        let potter_node_index = index
            .node_ids
            .iter()
            .enumerate()
            .map(|(node_index, id)| (id.clone(), node_index))
            .collect();
        let joint_index = index
            .joint_bones
            .iter()
            .map(|(node_index, bone)| (bone.clone(), *node_index))
            .collect();
        Self {
            version,
            index,
            doc,
            snapshot,
            extension,
            armature: None,
            humanoid: BTreeMap::new(),
            losses: Vec::new(),
            potter_node_index,
            joint_index,
        }
    }

    /// Records a loss for data Potter cannot represent. `feature_id` names the VRM feature.
    pub(super) fn loss(&mut self, feature_id: &str, reason: impl Into<String>, suggestion: &str) {
        self.losses.push(Loss {
            feature_id: feature_id.to_owned(),
            data_id: None,
            reason: reason.into(),
            suggestion: Some(suggestion.to_owned()),
        });
    }

    /// The glTF node JSON at `index`, or a malformed-file error.
    pub(super) fn gltf_node(&self, index: usize) -> Result<&Value> {
        self.index.root["nodes"]
            .get(index)
            .ok_or_else(|| import_error("VRM references a missing glTF node"))
    }

    /// The Potter reference for a glTF node: a bone when the node is a skinned joint, else a node.
    pub(super) fn node_ref(&self, node_index: usize) -> Result<AvatarNodeRef> {
        if let Some((armature, bone)) = self.index.joint_bones.get(&node_index) {
            return Ok(AvatarNodeRef {
                node: armature.clone(),
                bone: Some(bone.clone()),
            });
        }
        let node = self
            .index
            .node_ids
            .get(node_index)
            .ok_or_else(|| import_error("VRM references a missing glTF node"))?;
        Ok(AvatarNodeRef {
            node: node.clone(),
            bone: None,
        })
    }

    /// Potter material for a glTF material index.
    pub(super) fn material_id(&self, index: usize) -> Result<Id> {
        self.index
            .material_ids
            .get(index)
            .cloned()
            .ok_or_else(|| import_error("VRM references a missing glTF material"))
    }

    /// Original Potter material followed by its color-layer variants for a glTF material index.
    pub(super) fn material_ids(&self, index: usize) -> Result<Vec<Id>> {
        let original = self.material_id(index)?;
        let mut materials = Vec::with_capacity(
            1 + self
                .index
                .color_material_variants
                .get(&original)
                .map_or(0, BTreeMap::len),
        );
        materials.push(original.clone());
        if let Some(variants) = self.index.color_material_variants.get(&original) {
            materials.extend(variants.values().cloned());
        }
        Ok(materials)
    }

    /// Potter image for a glTF image index.
    pub(super) fn image_id(&self, index: usize) -> Result<Id> {
        self.index
            .image_ids
            .get(index)
            .cloned()
            .ok_or_else(|| import_error("VRM references a missing glTF image"))
    }

    /// Potter shape key for morph target `target` of glTF mesh `mesh`.
    pub(super) fn morph_shape_key(&self, mesh: usize, target: usize) -> Result<Id> {
        self.index
            .mesh_shape_key_ids
            .get(mesh)
            .and_then(|keys| keys.get(target))
            .cloned()
            .ok_or_else(|| import_error("VRM references a morph target that Potter did not import"))
    }

    /// glTF node indices that instance glTF mesh `mesh`.
    pub(super) fn nodes_using_mesh(&self, mesh: usize) -> Vec<usize> {
        self.index.root["nodes"]
            .as_array()
            .map_or(&[][..], Vec::as_slice)
            .iter()
            .enumerate()
            .filter(|(_, node)| node["mesh"].as_u64() == u64::try_from(mesh).ok())
            .map(|(node_index, _)| node_index)
            .collect()
    }

    /// The matrix that maps a glTF-space point expressed in `reference`'s local frame back into the
    /// Potter source frame: `source⁻¹ · basis⁻¹ · emitted`. The exporter's frame offset is its inverse.
    pub(super) fn source_frame(&self, reference: &AvatarNodeRef) -> Result<DMat4> {
        let (source, emitted) = if let Some(bone) = &reference.bone {
            let source = self
                .snapshot
                .bone_matrices
                .get(&reference.node)
                .and_then(|bones| bones.get(bone))
                .map(DMat4::from_cols_array)
                .ok_or_else(|| {
                    import_error("VRM bone reference is missing from the imported armature")
                })?;
            let node = self
                .joint_index
                .get(&(reference.node.clone(), bone.clone()))
                .copied()
                .ok_or_else(|| import_error("VRM bone reference is not a skinned joint"))?;
            (source, self.index.node_worlds[node])
        } else {
            let source = self
                .snapshot
                .nodes
                .get(&reference.node)
                .map(|node| DMat4::from_cols_array(&node.world_matrix))
                .ok_or_else(|| {
                    import_error("VRM node reference is missing from the imported scene")
                })?;
            let node = self
                .potter_node_index
                .get(&reference.node)
                .copied()
                .ok_or_else(|| import_error("VRM node reference is missing from the glTF file"))?;
            (source, self.index.node_worlds[node])
        };
        let matrix = source.inverse() * self.index.basis.inverse() * emitted;
        if !matrix.to_cols_array().iter().all(|value| value.is_finite()) {
            return Err(import_error("VRM frame is singular"));
        }
        Ok(matrix)
    }

    /// A point stored in `reference`'s emitted local frame, in Potter source coordinates.
    pub(super) fn source_point(
        &self,
        reference: &AvatarNodeRef,
        point: [f64; 3],
    ) -> Result<[f64; 3]> {
        Ok(self
            .source_frame(reference)?
            .transform_point3(DVec3::from_array(point))
            .to_array())
    }

    /// A vector stored in `reference`'s emitted local frame, in Potter source coordinates.
    pub(super) fn source_vector(
        &self,
        reference: &AvatarNodeRef,
        vector: [f64; 3],
    ) -> Result<[f64; 3]> {
        Ok(self
            .source_frame(reference)?
            .transform_vector3(DVec3::from_array(vector))
            .to_array())
    }

    /// A length stored in `reference`'s emitted local frame, in Potter source units. The frame must be a
    /// uniform scale (`vrm::uniform_scale`), read back at f32 precision.
    pub(super) fn source_radius(&self, reference: &AvatarNodeRef, radius: f64) -> Result<f64> {
        if radius == 0.0 {
            return Ok(0.0);
        }
        let matrix = self.source_frame(reference)?;
        let scale = uniform_scale(&matrix, F32_FRAME_TOLERANCE).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::ImportFailed,
                "VRM radius frame is not uniformly scaled",
                json!({ "node": reference.node, "bone": reference.bone }),
            )
        })?;
        Ok(radius * scale)
    }
}

pub(super) fn import_error(message: impl Into<String>) -> PotError {
    PotError::new(ErrorCode::ImportFailed, message)
}

/// Keeps each avatar component that validates and drops the others, each with its own loss. Meta,
/// armature and humanoid are required: when one fails there is no avatar. Optional components (first
/// person, look-at, expressions, spring bones and `MToon` materials) are dropped alone, and a spring
/// bone keeps every collider, group and chain that validates on its own.
fn accept_components(
    doc: &SceneDoc,
    avatar: AvatarSettings,
    losses: &mut Vec<Loss>,
) -> Option<AvatarSettings> {
    let mut accepted = AvatarSettings {
        expressions: BTreeMap::new(),
        first_person: None,
        look_at: None,
        spring_bone: None,
        mtoon: BTreeMap::new(),
        ..avatar.clone()
    };
    if let Err(error) = validate_avatar_settings(doc, &accepted) {
        losses.push(Loss {
            feature_id: required_feature(&error),
            data_id: None,
            reason: format!(
                "required VRM avatar component is not representable: {}",
                error.message
            ),
            suggestion: Some(
                "use --allow-lossy to import the geometry without avatar settings".to_owned(),
            ),
        });
        return None;
    }
    if let Some(first_person) = avatar.first_person {
        let mut candidate = accepted.clone();
        candidate.first_person = Some(first_person);
        keep_if_valid(doc, &mut accepted, candidate, "vrm.first_person", losses);
    }
    if let Some(look_at) = avatar.look_at {
        let mut candidate = accepted.clone();
        candidate.look_at = Some(look_at);
        keep_if_valid(doc, &mut accepted, candidate, "vrm.look_at", losses);
    }
    // MToon materials first: an expression that binds a material colour needs that material's settings.
    for (material, mtoon) in avatar.mtoon {
        let mut candidate = accepted.clone();
        let feature = format!("vrm.mtoon.{material}");
        candidate.mtoon.insert(material, mtoon);
        keep_if_valid(doc, &mut accepted, candidate, &feature, losses);
    }
    for (name, expression) in avatar.expressions {
        let mut candidate = accepted.clone();
        candidate.expressions.insert(name.clone(), expression);
        keep_if_valid(
            doc,
            &mut accepted,
            candidate,
            &format!("vrm.expression.{name}"),
            losses,
        );
    }
    if let Some(spring) = avatar.spring_bone {
        let mut candidate = accepted.clone();
        candidate.spring_bone = Some(spring.clone());
        if validate_avatar_settings(doc, &candidate).is_ok() {
            accepted = candidate;
        } else {
            accepted.spring_bone = Some(keep_spring_parts(doc, spring, losses));
        }
    }
    Some(accepted)
}

/// Adopts `candidate` when the whole avatar still validates; otherwise reports `feature` and keeps
/// `accepted` as it was.
fn keep_if_valid(
    doc: &SceneDoc,
    accepted: &mut AvatarSettings,
    candidate: AvatarSettings,
    feature: &str,
    losses: &mut Vec<Loss>,
) {
    match validate_avatar_settings(doc, &candidate) {
        Ok(()) => *accepted = candidate,
        Err(error) => losses.push(Loss {
            feature_id: feature.to_owned(),
            data_id: None,
            reason: format!(
                "VRM avatar component is not representable: {}",
                error.message
            ),
            suggestion: Some(
                "use --allow-lossy to import the file without this component".to_owned(),
            ),
        }),
    }
}

/// The spring bone without its invalid parts: colliders that fail alone are dropped, then collider
/// groups left empty, then each chain that fails with the parts kept so far. Losses name each part.
fn keep_spring_parts(
    doc: &SceneDoc,
    spring: AvatarSpringBone,
    losses: &mut Vec<Loss>,
) -> AvatarSpringBone {
    let mut kept = AvatarSpringBone {
        colliders: Vec::new(),
        collider_groups: Vec::new(),
        springs: Vec::new(),
    };
    let mut collider_map = Vec::with_capacity(spring.colliders.len());
    for (index, collider) in spring.colliders.iter().enumerate() {
        let alone = AvatarSpringBone {
            colliders: vec![collider.clone()],
            collider_groups: Vec::new(),
            springs: Vec::new(),
        };
        if validate_spring(doc, &alone).is_ok() {
            collider_map.push(Some(kept.colliders.len()));
            kept.colliders.push(collider.clone());
        } else {
            collider_map.push(None);
            spring_loss(
                losses,
                &format!("vrm.spring.collider.{index}"),
                validate_spring(doc, &alone),
            );
        }
    }
    let mut group_map = Vec::with_capacity(spring.collider_groups.len());
    for group in &spring.collider_groups {
        let members: Vec<usize> = group
            .colliders
            .iter()
            .filter_map(|index| collider_map.get(*index).copied().flatten())
            .collect();
        let remapped = AvatarSpringColliderGroup {
            name: group.name.clone(),
            colliders: members,
        };
        let alone = AvatarSpringBone {
            colliders: kept.colliders.clone(),
            collider_groups: vec![remapped.clone()],
            springs: Vec::new(),
        };
        if validate_spring(doc, &alone).is_ok() {
            group_map.push(Some(kept.collider_groups.len()));
            kept.collider_groups.push(remapped);
        } else {
            group_map.push(None);
            spring_loss(
                losses,
                &format!("vrm.spring.collider_group.{}", group.name),
                validate_spring(doc, &alone),
            );
        }
    }
    for chain in spring.springs {
        let mut candidate = chain.clone();
        candidate.collider_groups = chain
            .collider_groups
            .iter()
            .filter_map(|index| group_map.get(*index).copied().flatten())
            .collect();
        let mut trial = kept.clone();
        trial.springs.push(candidate.clone());
        let feature = match &chain.name {
            Some(name) => format!("vrm.spring.{name}"),
            None => "vrm.spring".to_owned(),
        };
        match validate_spring(doc, &trial) {
            Ok(()) => kept = trial,
            Err(error) => spring_loss(losses, &feature, Err(error)),
        }
    }
    kept
}

fn spring_loss(losses: &mut Vec<Loss>, feature: &str, outcome: crate::error::Result<()>) {
    if let Err(error) = outcome {
        losses.push(Loss {
            feature_id: feature.to_owned(),
            data_id: None,
            reason: format!(
                "VRM spring bone part is not representable: {}",
                error.message
            ),
            suggestion: Some(
                "use --allow-lossy to import the file without this spring bone part".to_owned(),
            ),
        });
    }
}

/// The feature a failed required component names, from the validation pointer.
fn required_feature(error: &PotError) -> String {
    let pointer = error
        .details
        .get("pointer")
        .and_then(Value::as_str)
        .unwrap_or_default();
    ["meta", "armature", "humanoid"]
        .iter()
        .find(|component| pointer.starts_with(&format!("/avatar/{component}")))
        .map_or_else(
            || "vrm.avatar".to_owned(),
            |component| format!("vrm.{component}"),
        )
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use serde_json::json;

    use crate::model::Material;

    use super::*;

    fn shared_material_fixture(root: Value) -> Result<(SceneDoc, GltfIndex, Id, Id)> {
        let source = Id::new("shared")?;
        let variant = Id::new("shared_color")?;
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        doc.materials.insert(
            source.clone(),
            Material {
                name: "shared".to_owned(),
                ..Material::default()
            },
        );
        doc.materials.insert(
            variant.clone(),
            Material {
                name: "shared".to_owned(),
                ..Material::default()
            },
        );
        let index = GltfIndex {
            root,
            basis: DMat4::IDENTITY,
            node_ids: Vec::new(),
            node_worlds: Vec::new(),
            material_ids: vec![source.clone()],
            color_material_variants: BTreeMap::from([(
                source.clone(),
                BTreeMap::from([("Paint".to_owned(), variant.clone())]),
            )]),
            image_ids: Vec::new(),
            mesh_shape_key_ids: Vec::new(),
            joint_bones: BTreeMap::new(),
            main_texture_transforms: BTreeSet::new(),
            discarded_base_transforms: BTreeSet::new(),
        };
        Ok((doc, index, source, variant))
    }

    fn shared_material_root(mtoon: &Value) -> Value {
        json!({
            "materials": [{
                "name": "shared",
                "alphaMode": "BLEND",
                "doubleSided": true,
                "extensions": {"VRMC_materials_mtoon": mtoon}
            }],
            "meshes": [{"primitives": [
                {"material": 0, "attributes": {"COLOR_0": 1}},
                {"material": 0, "attributes": {"POSITION": 0}}
            ]}]
        })
    }

    #[test]
    fn vrm_v1_material_metadata_reaches_color_variants() -> Result<()> {
        let root = shared_material_root(&json!({}));
        assert_eq!(root["meshes"][0]["primitives"][0]["material"], json!(0));
        assert_eq!(root["meshes"][0]["primitives"][1]["material"], json!(0));
        assert!(root["meshes"][0]["primitives"][0]["attributes"]["COLOR_0"].is_number());
        assert!(root["meshes"][0]["primitives"][1]["attributes"]["COLOR_0"].is_null());
        let (mut doc, index, source, variant) = shared_material_fixture(root)?;
        let extension = json!({
            "expressions": {
                "custom": {"smile": {
                    "materialColorBinds": [{
                        "material": 0,
                        "type": "shadeColor",
                        "targetValue": [0.2, 0.3, 0.4, 1.0]
                    }],
                    "textureTransformBinds": [{
                        "material": 0,
                        "scale": [2.0, 3.0],
                        "offset": [0.1, 0.2]
                    }]
                }}
            }
        });
        apply_material_render_state(&mut doc, &index);
        assert_eq!(doc.materials[&variant].alpha_mode, "blend");
        assert!(doc.materials[&variant].double_sided);
        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        let mut decoder = Decoder::new(AvatarVersion::V1, &index, &doc, &snapshot, &extension);
        assert_eq!(decoder.material_id(0)?, source);
        assert_eq!(
            decoder.material_ids(0)?,
            vec![source.clone(), variant.clone()]
        );
        let mtoon = mtoon::decode(&mut decoder)?;
        assert_eq!(mtoon[&source], mtoon[&variant]);
        let expressions = expression::decode(&mut decoder, &mtoon)?;
        let expression = &expressions["smile"];
        assert_eq!(
            expression
                .material_color_binds
                .iter()
                .map(|bind| bind.material.clone())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([source.clone(), variant.clone()])
        );
        assert_eq!(
            expression
                .texture_transform_binds
                .iter()
                .map(|bind| bind.material.clone())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([source, variant])
        );
        Ok(())
    }

    #[test]
    fn vrm_v0_material_names_and_binds_use_originals_and_variants() -> Result<()> {
        let root = shared_material_root(&json!({}));
        let (doc, index, source, variant) = shared_material_fixture(root)?;
        let extension = json!({
            "materialProperties": [{
                "shader": "VRM/MToon",
                "floatProperties": {},
                "vectorProperties": {},
                "textureProperties": {}
            }],
            "blendShapeMaster": {"blendShapeGroups": [{
                "name": "smile",
                "binds": [],
                "materialValues": [
                    {
                        "propertyName": "_RimColor",
                        "materialName": "shared",
                        "targetValue": [0.2, 0.3, 0.4, 1.0]
                    },
                    {
                        "propertyName": "_MainTex_ST",
                        "materialName": "shared",
                        "targetValue": [2.0, 3.0, 0.1, 0.2]
                    }
                ]
            }]}
        });
        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        let mut decoder = Decoder::new(AvatarVersion::V0, &index, &doc, &snapshot, &extension);
        let mtoon = mtoon::decode(&mut decoder)?;
        assert_eq!(mtoon[&source], mtoon[&variant]);
        let expressions = expression::decode(&mut decoder, &mtoon)?;
        let expression = &expressions["smile"];
        assert_eq!(
            expression
                .material_color_binds
                .iter()
                .map(|bind| bind.material.clone())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([source.clone(), variant.clone()])
        );
        assert_eq!(
            expression
                .texture_transform_binds
                .iter()
                .map(|bind| bind.material.clone())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([source, variant])
        );
        Ok(())
    }
}
