mod mtoon;
pub(crate) use mtoon::uv_textures;

use glam::{DMat4, DVec3};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

use crate::{
    error::{ErrorCode, PotError, Result},
    float::equal_f64,
    image::ImageInterpolation,
    model::{
        AvatarCommercialUsage, AvatarCreditNotation, AvatarExpression, AvatarExpressionOverride,
        AvatarLegacyAllowedUser, AvatarLegacyBinaryPermission, AvatarLegacyLicenseName,
        AvatarLookAt, AvatarLookAtType, AvatarMaterialColorType, AvatarMeta, AvatarModification,
        AvatarNodeRef, AvatarPermission, AvatarSettings, AvatarSpring, AvatarSpringBone,
        AvatarSpringColliderShape, AvatarVersion, Id, SceneDoc, TextureRef,
        validate_avatar_settings,
    },
};

use super::gltf::{TextureExport, VrmWriterMappings, reference_texture, unsupported_export};
use super::material_surface::{ResolvedSurface, resolve_surface};

pub fn export(
    doc: &SceneDoc,
    snapshot: &crate::eval::Snapshot,
    project_root: &Path,
    version: AvatarVersion,
) -> Result<Vec<u8>> {
    let frames = validate_output_with_frames(doc, snapshot, version)?;
    super::gltf::export_vrm_with_frames(doc, snapshot, project_root, version, &frames)
}

pub(crate) fn required_avatar(doc: &SceneDoc) -> Result<&AvatarSettings> {
    doc.avatar.as_ref().ok_or_else(|| {
        super::gltf::unsupported_export(
            "vrm.avatar_settings",
            "VRM export requires avatar settings; set them with avatar.update.",
        )
    })
}

pub fn validate_settings(doc: &SceneDoc, version: AvatarVersion) -> Result<()> {
    let avatar = required_avatar(doc)?;
    validate_avatar_settings(doc, avatar)?;
    validate_animation_channels(doc)?;
    for material in doc.materials.values() {
        if !matches!(material.alpha_mode.as_str(), "opaque" | "clip" | "blend") {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "VRM material alpha_mode must be opaque, clip, or blend",
            ));
        }
    }
    match version {
        AvatarVersion::V1 => {
            if avatar.meta.license_url != "https://vrm.dev/licenses/1.0/" {
                return Err(unsupported_export(
                    "vrm.v1.license_url",
                    "VRM 1.0 requires license_url `https://vrm.dev/licenses/1.0/`; put custom license links in other_license_url.",
                ));
            }
            if let Some(thumbnail) = &avatar.meta.thumbnail
                && doc
                    .images
                    .get(thumbnail)
                    .is_some_and(|image| image.width != image.height)
            {
                return Err(unsupported_export(
                    "vrm.v1.thumbnail_aspect",
                    "VRM 1.0 requires a square thumbnail image; resize the thumbnail or export without one.",
                ));
            }
            for expression in avatar.expressions.values() {
                for bind in &expression.material_color_binds {
                    if matches!(
                        bind.color_type,
                        AvatarMaterialColorType::ShadeColor
                            | AvatarMaterialColorType::MatcapColor
                            | AvatarMaterialColorType::RimColor
                            | AvatarMaterialColorType::OutlineColor
                    ) && !avatar.mtoon.contains_key(&bind.material)
                    {
                        return Err(unsupported_export(
                            "vrm.expression_mtoon_color",
                            "VRM material color binds for shade, matcap, rim, and outline require MToon settings on the target material.",
                        ));
                    }
                }
            }
            for material_id in avatar.mtoon.keys() {
                if !doc.materials.contains_key(material_id) {
                    return Err(PotError::new(
                        ErrorCode::SceneInvalid,
                        "MToon material is missing",
                    ));
                }
                let (surface, _) = resolve_surface(doc, material_id)?;
                mtoon::validate_v1(&surface)?;
            }
        }
        AvatarVersion::V0 => validate_v0_settings(doc, avatar)?,
    }
    Ok(())
}
pub fn validate_output(
    doc: &SceneDoc,
    snapshot: &crate::eval::Snapshot,
    version: AvatarVersion,
) -> Result<()> {
    validate_output_with_frames(doc, snapshot, version).map(|_| ())
}

fn validate_output_with_frames(
    doc: &SceneDoc,
    snapshot: &crate::eval::Snapshot,
    version: AvatarVersion,
) -> Result<super::gltf::VrmFrameMaps> {
    validate_settings(doc, version)?;
    let avatar = required_avatar(doc)?;
    let frames = super::gltf::vrm_frame_maps(doc, snapshot, avatar, version)?;
    super::gltf::validate_vrm_mesh_references(doc, snapshot, avatar, &frames, version)?;
    validate_vrm_frame_maps(avatar, version, &frames)?;
    Ok(frames)
}
fn validate_vrm_frame_maps(
    avatar: &AvatarSettings,
    version: AvatarVersion,
    frames: &super::gltf::VrmFrameMaps,
) -> Result<()> {
    if let Some(spring) = &avatar.spring_bone {
        for collider in &spring.colliders {
            let matrix = frame_matrix(
                &frames.node_offset_matrices,
                &frames.bone_offset_matrices,
                &collider.node,
            )?;
            let (offset, radius) = match &collider.shape {
                AvatarSpringColliderShape::Sphere { offset, radius }
                | AvatarSpringColliderShape::Capsule { offset, radius, .. } => (offset, radius),
            };
            if !matrix
                .transform_point3(DVec3::from_array(*offset))
                .is_finite()
            {
                return Err(unsupported_export(
                    "vrm.collider_frame",
                    "Spring collider offsets cannot be represented in the normalized VRM frame.",
                ));
            }
            if *radius > 0.0 {
                uniform_frame_scale(&matrix, "spring collider radius")?;
            }
            if let AvatarSpringColliderShape::Capsule { tail, .. } = &collider.shape
                && !matrix
                    .transform_point3(DVec3::from_array(*tail))
                    .is_finite()
            {
                return Err(unsupported_export(
                    "vrm.collider_frame",
                    "Spring collider tails cannot be represented in the normalized VRM frame.",
                ));
            }
        }
        for chain in &spring.springs {
            let mut chain_scale = None;
            for joint in &chain.joints {
                let matrix = frame_matrix(
                    &frames.node_offset_matrices,
                    &frames.bone_offset_matrices,
                    &joint.node,
                )?;
                if joint.hit_radius > 0.0 {
                    let scale = uniform_frame_scale(&matrix, "spring joint hit radius")?;
                    if version == AvatarVersion::V0
                        && chain_scale.is_some_and(|previous: f64| !nearly_equal(previous, scale))
                    {
                        return Err(unsupported_export(
                            "vrm.v0.spring_hit_radius_frame",
                            "VRM 0.x uses one hitRadius for the whole spring subtree, but the normalized joint frames require different radii.",
                        ));
                    }
                    chain_scale = Some(scale);
                }
            }
        }
    }
    if let Some(look_at) = &avatar.look_at {
        let head = avatar
            .humanoid
            .get("head")
            .ok_or_else(|| PotError::new(ErrorCode::SceneInvalid, "avatar head bone is missing"))?;
        let matrix = frame_matrix(
            &frames.node_offset_matrices,
            &frames.bone_offset_matrices,
            &AvatarNodeRef {
                node: avatar.armature.clone(),
                bone: Some(head.clone()),
            },
        )?;
        if !matrix
            .transform_vector3(DVec3::from_array(look_at.offset_from_head_bone))
            .is_finite()
        {
            return Err(unsupported_export(
                "vrm.look_at_frame",
                "The look-at offset cannot be represented in the normalized head-bone frame.",
            ));
        }
    }
    Ok(())
}

fn frame_matrix(
    node_matrices: &BTreeMap<Id, DMat4>,
    bone_matrices: &BTreeMap<(Id, Id), DMat4>,
    reference: &AvatarNodeRef,
) -> Result<DMat4> {
    let matrix = if let Some(bone) = &reference.bone {
        bone_matrices.get(&(reference.node.clone(), bone.clone()))
    } else {
        node_matrices.get(&reference.node)
    };
    matrix.copied().ok_or_else(|| {
        unsupported_export(
            "vrm.local_frame",
            "A VRM attachment references a node or bone without a normalized local-frame mapping.",
        )
    })
}

fn map_local_vector(
    mappings: &VrmWriterMappings<'_>,
    reference: &AvatarNodeRef,
    vector: [f64; 3],
) -> Result<[f64; 3]> {
    Ok(frame_matrix(
        mappings.node_offset_matrices,
        mappings.bone_offset_matrices,
        reference,
    )?
    .transform_vector3(DVec3::from_array(vector))
    .to_array())
}

fn map_local_point(
    mappings: &VrmWriterMappings<'_>,
    reference: &AvatarNodeRef,
    point: [f64; 3],
) -> Result<[f64; 3]> {
    Ok(frame_matrix(
        mappings.node_offset_matrices,
        mappings.bone_offset_matrices,
        reference,
    )?
    .transform_point3(DVec3::from_array(point))
    .to_array())
}

fn map_local_radius(
    mappings: &VrmWriterMappings<'_>,
    reference: &AvatarNodeRef,
    radius: f64,
    label: &str,
) -> Result<f64> {
    if radius == 0.0 {
        return Ok(0.0);
    }
    let matrix = frame_matrix(
        mappings.node_offset_matrices,
        mappings.bone_offset_matrices,
        reference,
    )?;
    let scale = uniform_frame_scale(&matrix, label)?;
    let mapped_radius = radius * scale;
    if !mapped_radius.is_finite() {
        return Err(unsupported_export(
            "vrm.collider_frame_scale",
            &format!("{label} overflows under the normalized frame scale."),
        ));
    }
    Ok(mapped_radius)
}

fn map_head_offset(avatar: &AvatarSettings, mappings: &VrmWriterMappings<'_>) -> Result<[f64; 3]> {
    let head = avatar.humanoid.get("head").ok_or_else(|| {
        unsupported_export(
            "vrm.look_at_frame",
            "Look-at settings require a mapped humanoid head bone.",
        )
    })?;
    let reference = AvatarNodeRef {
        node: avatar.armature.clone(),
        bone: Some(head.clone()),
    };
    let offset = avatar
        .look_at
        .as_ref()
        .map_or([0.0; 3], |look_at| look_at.offset_from_head_bone);
    map_local_vector(mappings, &reference, offset)
}

fn uniform_frame_scale(matrix: &DMat4, label: &str) -> Result<f64> {
    let axes = [
        matrix.transform_vector3(DVec3::X),
        matrix.transform_vector3(DVec3::Y),
        matrix.transform_vector3(DVec3::Z),
    ];
    let lengths = axes.map(DVec3::length);
    let scale = lengths[0];
    let length_tolerance = scale * 1e-8;
    if !scale.is_finite()
        || scale <= f64::EPSILON
        || lengths
            .iter()
            .any(|length| !length.is_finite() || (*length - scale).abs() > length_tolerance)
        || {
            let axes = axes.map(|axis| axis / scale);
            axes[0].dot(axes[1]).abs() > 1e-8
                || axes[0].dot(axes[2]).abs() > 1e-8
                || axes[1].dot(axes[2]).abs() > 1e-8
        }
    {
        return Err(unsupported_export(
            "vrm.collider_frame_scale",
            &format!(
                "{label} cannot be represented by the normalized frame's uniform scalar scale."
            ),
        ));
    }
    Ok(scale)
}

fn nearly_equal(left: f64, right: f64) -> bool {
    (left - right).abs() <= left.abs().max(right.abs()).max(f64::EPSILON) * 1e-8
}

fn validate_animation_channels(doc: &SceneDoc) -> Result<()> {
    for (node_id, node) in &doc.nodes {
        if node
            .action
            .as_ref()
            .and_then(|action_id| doc.actions.get(action_id))
            .is_some_and(action_has_transform_channels)
        {
            return Err(unsupported_export(
                "vrm.animation_transform",
                &format!(
                    "Node `{node_id}` has active transform animation channels that VRM cannot preserve after coordinate normalization; remove or bake the action before export."
                ),
            ));
        }
        let has_solo = node
            .nla_tracks
            .iter()
            .any(|track| track.solo && !track.mute);
        if node.nla_tracks.iter().any(|track| {
            !track.mute
                && (!has_solo || track.solo)
                && track.strips.iter().any(|strip| {
                    strip.influence > 0.0
                        && doc
                            .actions
                            .get(&strip.action)
                            .is_some_and(action_has_transform_channels)
                })
        }) {
            return Err(unsupported_export(
                "vrm.animation_transform",
                &format!(
                    "Node `{node_id}` has active NLA transform animation channels that VRM cannot preserve after coordinate normalization; mute, remove, or bake the strips before export."
                ),
            ));
        }
        if node
            .drivers
            .iter()
            .any(|driver| is_transform_channel(&driver.path))
        {
            return Err(unsupported_export(
                "vrm.animation_transform",
                &format!(
                    "Node `{node_id}` has an active transform driver that VRM cannot preserve after coordinate normalization; remove or bake the driver before export."
                ),
            ));
        }
    }
    Ok(())
}

fn action_has_transform_channels(action: &crate::model::Action) -> bool {
    action
        .fcurves
        .iter()
        .any(|curve| !curve.keyframes.is_empty() && is_transform_channel(&curve.path))
}

fn is_transform_channel(path: &str) -> bool {
    if matches!(
        path,
        "transform.translation"
            | "transform.rotation"
            | "transform.rotation_euler"
            | "transform.rotation_quaternion"
            | "transform.scale"
    ) {
        return true;
    }
    let channel = if let Some(end) = path.find(']')
        && path.starts_with("pose.bones[")
    {
        path[end + 1..].trim_start_matches('.')
    } else if let Some(rest) = path.strip_prefix("pose.")
        && let Some((_, channel)) = rest.split_once('.')
    {
        channel
    } else {
        return false;
    };
    matches!(
        channel,
        "location" | "translation" | "rotation" | "rotation_euler" | "scale"
    )
}

fn validate_v0_settings(doc: &SceneDoc, avatar: &AvatarSettings) -> Result<()> {
    validate_v0_meta_settings(&avatar.meta)?;
    for role in ["chest", "neck"] {
        if !avatar.humanoid.contains_key(role) {
            return Err(unsupported_export(
                "vrm.v0.humanoid_required_bone",
                &format!("VRM 0.x requires the `{role}` humanoid bone; map it in avatar.humanoid."),
            ));
        }
    }
    for expression in avatar.expressions.values() {
        if expression.override_blink != AvatarExpressionOverride::None
            || expression.override_mouth != AvatarExpressionOverride::None
            || expression.override_look_at != AvatarExpressionOverride::None
        {
            return Err(unsupported_export(
                "vrm.v0.expression_override",
                "VRM 0.x has no expression procedural override fields; clear overrides or export VRM 1.0.",
            ));
        }
        for bind in &expression.material_color_binds {
            if bind.color_type == AvatarMaterialColorType::MatcapColor {
                return Err(unsupported_export(
                    "vrm.v0.expression_matcap_color",
                    "VRM 0.x MToon exposes matcap as a texture, not a material color factor; remove this bind or export VRM 1.0.",
                ));
            }
            if !avatar.mtoon.contains_key(&bind.material) {
                return Err(unsupported_export(
                    "vrm.v0.expression_material_bind",
                    "VRM 0.x material value binds require MToon settings on the target material.",
                ));
            }
            ensure_unique_legacy_material_name(doc, &bind.material)?;
        }
        for bind in &expression.texture_transform_binds {
            if !doc.materials.contains_key(&bind.material) {
                return Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    "expression target material is missing",
                ));
            }
            if !avatar.mtoon.contains_key(&bind.material) {
                return Err(unsupported_export(
                    "vrm.v0.expression_texture_transform",
                    "VRM 0.x texture-transform binds require MToon settings on the target material.",
                ));
            }
            ensure_legacy_texture_transform(doc, &bind.material)?;
            ensure_unique_legacy_material_name(doc, &bind.material)?;
        }
    }
    for (material_id, mtoon) in &avatar.mtoon {
        if !doc.materials.contains_key(material_id) {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "MToon material is missing",
            ));
        }
        let (surface, _) = resolve_surface(doc, material_id)?;
        mtoon::validate_v0(&surface, mtoon)?;
    }
    if let Some(spring) = &avatar.spring_bone {
        validate_v0_spring(doc, spring)?;
    }
    Ok(())
}

struct V0MetaPolicy<'a> {
    license_name: AvatarLegacyLicenseName,
    allowed_user: AvatarLegacyAllowedUser,
    violent_usage: AvatarLegacyBinaryPermission,
    sexual_usage: AvatarLegacyBinaryPermission,
    commercial_usage: AvatarLegacyBinaryPermission,
    title: &'a str,
    author: Option<&'a str>,
    reference: Option<&'a str>,
    other_permission_url: Option<&'a str>,
    other_license_url: Option<&'a str>,
}

fn validate_v0_meta_settings(meta: &AvatarMeta) -> Result<V0MetaPolicy<'_>> {
    let legacy = meta.legacy.as_ref();
    if meta.copyright_information.is_some() || meta.third_party_licenses.is_some() {
        return Err(unsupported_export(
            "vrm.v0.metadata",
            "VRM 0.x has no copyrightInformation or thirdPartyLicenses field; remove those fields or export VRM 1.0.",
        ));
    }
    if meta.credit_notation == AvatarCreditNotation::Unnecessary {
        return Err(unsupported_export(
            "vrm.v0.metadata_credit",
            "VRM 0.x cannot encode credit_notation `unnecessary`; export VRM 1.0.",
        ));
    }
    if legacy
        .and_then(|legacy| legacy.other_permission_url.as_deref())
        .is_none_or(|url| url.trim().is_empty())
    {
        return Err(unsupported_export(
            "vrm.v0.meta_permissions",
            "VRM 0.x cannot encode political/religious or antisocial/hate usage permissions; supply meta.legacy.other_permission_url.",
        ));
    }
    if meta.commercial_usage != AvatarCommercialUsage::PersonalNonProfit
        && legacy.and_then(|legacy| legacy.commercial_usage).is_none()
    {
        return Err(unsupported_export(
            "vrm.v0.meta_permissions",
            "VRM 0.x commercial permission is binary; set meta.legacy.commercial_usage explicitly.",
        ));
    }
    if (meta.allow_redistribution || meta.modification != AvatarModification::Prohibited)
        && legacy.and_then(|legacy| legacy.license_name).is_none()
    {
        return Err(unsupported_export(
            "vrm.v0.meta_license",
            "Set meta.legacy.license_name to map redistribution and modification rights to VRM 0.x.",
        ));
    }
    let license = legacy
        .and_then(|legacy| legacy.license_name)
        .unwrap_or(AvatarLegacyLicenseName::RedistributionProhibited);
    let creative_commons = matches!(
        license,
        AvatarLegacyLicenseName::Cc0
            | AvatarLegacyLicenseName::CcBy
            | AvatarLegacyLicenseName::CcByNc
            | AvatarLegacyLicenseName::CcBySa
            | AvatarLegacyLicenseName::CcByNcSa
            | AvatarLegacyLicenseName::CcByNd
            | AvatarLegacyLicenseName::CcByNcNd
    );
    if creative_commons != meta.allow_redistribution && license != AvatarLegacyLicenseName::Other {
        return Err(unsupported_export(
            "vrm.v0.meta_license",
            "The selected VRM 0.x license does not match allow_redistribution; choose a compatible legacy license.",
        ));
    }
    if matches!(license, AvatarLegacyLicenseName::Cc0)
        && meta.credit_notation == AvatarCreditNotation::Required
    {
        return Err(unsupported_export(
            "vrm.v0.meta_credit_license",
            "VRM 0.x CC0 waives attribution and cannot preserve credit_notation `required`.",
        ));
    }
    if creative_commons && meta.modification == AvatarModification::Prohibited {
        return Err(unsupported_export(
            "vrm.v0.meta_license_modification",
            "Creative Commons licenses grant adaptation (CC BY-ND grants private adaptation), so modification `prohibited` cannot preserve them; use modification `allow_modification` with a BY-ND or BY-NC-ND license, or export VRM 1.0.",
        ));
    }
    let shares_adapted_material = matches!(
        license,
        AvatarLegacyLicenseName::Cc0
            | AvatarLegacyLicenseName::CcBy
            | AvatarLegacyLicenseName::CcBySa
            | AvatarLegacyLicenseName::CcByNc
            | AvatarLegacyLicenseName::CcByNcSa
    );
    if shares_adapted_material && meta.modification == AvatarModification::AllowModification {
        return Err(unsupported_export(
            "vrm.v0.meta_license_modification",
            "This Creative Commons license permits sharing adapted material, which modification `allow_modification` (private use only) cannot preserve; use modification `allow_modification_redistribution` or a BY-ND or BY-NC-ND license.",
        ));
    }
    if matches!(
        license,
        AvatarLegacyLicenseName::Cc0
            | AvatarLegacyLicenseName::CcBy
            | AvatarLegacyLicenseName::CcBySa
            | AvatarLegacyLicenseName::CcByNd
    ) && meta.commercial_usage != AvatarCommercialUsage::Corporation
    {
        return Err(unsupported_export(
            "vrm.v0.meta_license_commercial",
            "This Creative Commons license permits commercial use by anyone, so commercial_usage must be `corporation` to preserve it.",
        ));
    }
    let expected_allowed_user = match meta.avatar_permission {
        AvatarPermission::OnlyAuthor => AvatarLegacyAllowedUser::OnlyAuthor,
        AvatarPermission::OnlySeparatelyLicensedPerson => {
            AvatarLegacyAllowedUser::ExplicitlyLicensedPerson
        }
        AvatarPermission::Everyone => AvatarLegacyAllowedUser::Everyone,
    };
    let allowed_user = legacy
        .and_then(|legacy| legacy.allowed_user)
        .unwrap_or(expected_allowed_user);
    if legacy_user_rank(allowed_user) > legacy_user_rank(expected_allowed_user) {
        return Err(unsupported_export(
            "vrm.v0.meta_permissions",
            "The selected VRM 0.x allowed_user permits a broader audience than avatar_permission; choose an equally or more restrictive legacy value.",
        ));
    }
    let violent = legacy.and_then(|legacy| legacy.violent_usage).unwrap_or(
        if meta.allow_excessively_violent_usage {
            AvatarLegacyBinaryPermission::Allow
        } else {
            AvatarLegacyBinaryPermission::Disallow
        },
    );
    let sexual = legacy.and_then(|legacy| legacy.sexual_usage).unwrap_or(
        if meta.allow_excessively_sexual_usage {
            AvatarLegacyBinaryPermission::Allow
        } else {
            AvatarLegacyBinaryPermission::Disallow
        },
    );
    let commercial = legacy.and_then(|legacy| legacy.commercial_usage).unwrap_or(
        if meta.commercial_usage == AvatarCommercialUsage::PersonalNonProfit {
            AvatarLegacyBinaryPermission::Disallow
        } else {
            AvatarLegacyBinaryPermission::Allow
        },
    );
    if (!meta.allow_excessively_violent_usage && violent == AvatarLegacyBinaryPermission::Allow)
        || (!meta.allow_excessively_sexual_usage && sexual == AvatarLegacyBinaryPermission::Allow)
    {
        return Err(unsupported_export(
            "vrm.v0.meta_permissions",
            "The selected VRM 0.x violent/sexual permission is broader than the corresponding VRM 1.0 setting.",
        ));
    }
    if commercial == AvatarLegacyBinaryPermission::Allow
        && meta.commercial_usage != AvatarCommercialUsage::Corporation
    {
        return Err(unsupported_export(
            "vrm.v0.meta_permissions",
            "The selected VRM 0.x commercial permission allows a broader class of users than the VRM 1.0 commercial_usage setting.",
        ));
    }
    if commercial == AvatarLegacyBinaryPermission::Allow
        && matches!(
            license,
            AvatarLegacyLicenseName::CcByNc
                | AvatarLegacyLicenseName::CcByNcSa
                | AvatarLegacyLicenseName::CcByNcNd
        )
    {
        return Err(unsupported_export(
            "vrm.v0.meta_license_commercial",
            "The selected Creative Commons NonCommercial license prohibits commercial use, so it cannot be paired with VRM 0.x commercialUsage `Allow`.",
        ));
    }
    let title = legacy
        .and_then(|legacy| legacy.title.as_deref())
        .unwrap_or(&meta.name);
    let author = legacy.and_then(|legacy| legacy.author.as_deref());
    let author_is_empty = author.map_or_else(
        || meta.authors.iter().all(|author| author.trim().is_empty()),
        |author| author.trim().is_empty(),
    );
    if title.trim().is_empty() || author_is_empty {
        return Err(unsupported_export(
            "vrm.v0.metadata",
            "VRM 0.x title and author must be non-empty; correct the legacy title/author overrides.",
        ));
    }
    if legacy.and_then(|legacy| legacy.license_name) != Some(AvatarLegacyLicenseName::Other)
        && (legacy
            .and_then(|legacy| legacy.other_license_url.as_ref())
            .is_some()
            || meta.other_license_url.is_some())
    {
        return Err(unsupported_export(
            "vrm.v0.meta_license",
            "VRM 0.x other_license_url is valid only when legacy license_name is `other`.",
        ));
    }
    if license == AvatarLegacyLicenseName::Other
        && legacy
            .and_then(|legacy| legacy.other_license_url.as_deref())
            .or(meta.other_license_url.as_deref())
            .is_none_or(|url| url.trim().is_empty())
    {
        return Err(unsupported_export(
            "vrm.v0.meta_license",
            "VRM 0.x license_name `other` requires a non-empty other_license_url.",
        ));
    }
    Ok(V0MetaPolicy {
        license_name: license,
        allowed_user,
        violent_usage: violent,
        sexual_usage: sexual,
        commercial_usage: commercial,
        title,
        author,
        reference: legacy.and_then(|legacy| legacy.reference.as_deref()),
        other_permission_url: legacy.and_then(|legacy| legacy.other_permission_url.as_deref()),
        other_license_url: legacy
            .and_then(|legacy| legacy.other_license_url.as_deref())
            .or(meta.other_license_url.as_deref()),
    })
}

fn validate_v0_spring(doc: &SceneDoc, spring: &AvatarSpringBone) -> Result<()> {
    if spring
        .colliders
        .iter()
        .any(|collider| matches!(collider.shape, AvatarSpringColliderShape::Capsule { .. }))
    {
        return Err(unsupported_export(
            "vrm.v0.spring_capsule",
            "VRM 0.x supports sphere colliders only; replace this capsule with spheres or export VRM 1.0.",
        ));
    }
    let mut used = BTreeSet::new();
    for group in &spring.collider_groups {
        if group.colliders.is_empty() {
            return Err(unsupported_export(
                "vrm.v0.spring_collider_group",
                "VRM 0.x collider groups must contain at least one sphere collider.",
            ));
        }
        for collider in &group.colliders {
            if *collider >= spring.colliders.len() {
                return Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    "spring collider group index is invalid",
                ));
            }
            used.insert(*collider);
        }
    }
    if used.len() != spring.colliders.len() {
        return Err(unsupported_export(
            "vrm.v0.spring_unreferenced_collider",
            "VRM 0.x cannot preserve standalone colliders; add each collider to a collider group or remove it.",
        ));
    }
    for chain in &spring.springs {
        let Some(first) = chain.joints.first() else {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "spring chain has no joints",
            ));
        };
        if chain.joints.iter().skip(1).any(|joint| {
            !equal_f64(joint.hit_radius, first.hit_radius)
                || !equal_f64(joint.stiffness, first.stiffness)
                || !equal_f64(joint.gravity_power, first.gravity_power)
                || !joint
                    .gravity_dir
                    .iter()
                    .zip(first.gravity_dir.iter())
                    .all(|(left, right)| equal_f64(*left, *right))
                || !equal_f64(joint.drag_force, first.drag_force)
        }) {
            return Err(unsupported_export(
                "vrm.v0.spring_joint_settings",
                "VRM 0.x uses one physics parameter set per spring chain; make every joint setting identical or export VRM 1.0.",
            ));
        }
        validate_v0_chain_topology(doc, chain)?;
        validate_v0_terminal_rotation(doc, spring, chain)?;
    }
    for (index, chain) in spring.springs.iter().enumerate() {
        let Some(root) = chain.joints.first().map(|joint| &joint.node) else {
            continue;
        };
        if spring.springs.iter().take(index).any(|previous| {
            previous
                .joints
                .first()
                .is_some_and(|other| v0_spring_roots_overlap(doc, root, &other.node))
        }) {
            return Err(unsupported_export(
                "vrm.v0.spring_duplicate_root",
                "VRM 0.x recursively simulates every spring-root subtree; configure disjoint roots so no joint is simulated by more than one spring.",
            ));
        }
    }
    Ok(())
}

fn validate_v0_chain_topology(doc: &SceneDoc, chain: &AvatarSpring) -> Result<()> {
    let first = chain
        .joints
        .first()
        .ok_or_else(|| PotError::new(ErrorCode::SceneInvalid, "spring chain has no joints"))?;
    let bone_mode = first.node.bone.is_some();
    if chain
        .joints
        .iter()
        .any(|joint| joint.node.bone.is_some() != bone_mode)
        || (bone_mode
            && chain
                .joints
                .iter()
                .any(|joint| joint.node.node != first.node.node))
    {
        return Err(unsupported_export(
            "vrm.v0.spring_hierarchy",
            "VRM 0.x spring roots animate their complete node subtree; configure one complete linear subtree per spring or export VRM 1.0.",
        ));
    }
    let expected = chain
        .joints
        .iter()
        .map(|joint| {
            joint
                .node
                .bone
                .clone()
                .unwrap_or_else(|| joint.node.node.clone())
        })
        .collect::<BTreeSet<_>>();
    if bone_mode {
        let armature_node = doc.nodes.get(&first.node.node).ok_or_else(|| {
            PotError::new(ErrorCode::SceneInvalid, "spring armature node is missing")
        })?;
        let armature = armature_node
            .data
            .as_ref()
            .and_then(|id| doc.data_blocks.get(id))
            .and_then(|data| data.armature.as_ref())
            .ok_or_else(|| {
                PotError::new(ErrorCode::SceneInvalid, "spring armature data is missing")
            })?;
        for pair in chain.joints.windows(2) {
            let (Some(parent_bone), Some(child_bone)) =
                (pair[0].node.bone.as_ref(), pair[1].node.bone.as_ref())
            else {
                return Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    "spring chain mixes bone and node references",
                ));
            };
            if armature
                .bones
                .get(child_bone)
                .and_then(|bone| bone.parent.as_ref())
                != Some(parent_bone)
                || armature
                    .bones
                    .values()
                    .filter(|bone| bone.parent.as_ref() == Some(parent_bone))
                    .count()
                    != 1
                || doc.nodes.values().any(|node| {
                    node.parent_type == crate::model::ParentType::Bone
                        && node.parent.as_ref() == Some(&first.node.node)
                        && node.parent_bone.as_ref() == Some(parent_bone)
                })
            {
                return Err(unsupported_export(
                    "vrm.v0.spring_hierarchy",
                    "VRM 0.x recursively simulates every spring-root branch, and each joint uses its first child as its tail; remove extra children and bone-parented accessories from nonterminal joints or export VRM 1.0.",
                ));
            }
        }
        let Some(root) = first.node.bone.as_ref() else {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "spring chain mixes bone and node references",
            ));
        };
        let mut actual = BTreeSet::from([root.clone()]);
        for candidate in armature.bones.keys() {
            if bone_descends_from(armature, root, candidate) {
                actual.insert(candidate.clone());
            }
        }
        if !expected.is_subset(&actual) {
            return Err(unsupported_export(
                "vrm.v0.spring_hierarchy",
                "VRM 0.x spring roots recursively simulate their whole descendant subtree; use one direct-child chain and ensure unlisted terminal descendants are inert, or export VRM 1.0.",
            ));
        }
    } else {
        let root = doc
            .nodes
            .get(&first.node.node)
            .ok_or_else(|| PotError::new(ErrorCode::SceneInvalid, "spring root node is missing"))?;
        if root.kind == "armature" {
            return Err(unsupported_export(
                "vrm.v0.spring_hierarchy",
                "VRM 0.x spring roots on an armature object include its generated bones; reference the intended bones explicitly or export VRM 1.0.",
            ));
        }
        for pair in chain.joints.windows(2) {
            if doc
                .nodes
                .get(&pair[1].node.node)
                .and_then(|node| node.parent.as_ref())
                != Some(&pair[0].node.node)
            {
                return Err(unsupported_export(
                    "vrm.v0.spring_hierarchy",
                    "VRM 0.x spring roots animate a complete direct-child subtree; configure one direct-child joint path or export VRM 1.0.",
                ));
            }
            if doc
                .nodes
                .get(&pair[0].node.node)
                .is_some_and(|node| node.kind == "armature")
                || doc
                    .nodes
                    .values()
                    .filter(|node| node.parent.as_ref() == Some(&pair[0].node.node))
                    .count()
                    != 1
            {
                return Err(unsupported_export(
                    "vrm.v0.spring_hierarchy",
                    "VRM 0.x recursively simulates every spring-root branch, and each joint uses its first child as its tail; remove extra children from nonterminal joints or export VRM 1.0.",
                ));
            }
        }
        let mut actual = BTreeSet::from([first.node.node.clone()]);
        for candidate in doc.nodes.keys() {
            if node_descends_from(doc, &first.node.node, candidate) {
                actual.insert(candidate.clone());
            }
        }
        if !expected.is_subset(&actual) {
            return Err(unsupported_export(
                "vrm.v0.spring_hierarchy",
                "VRM 0.x spring roots recursively simulate their whole descendant subtree; use one direct-child chain and ensure unlisted terminal descendants are inert, or export VRM 1.0.",
            ));
        }
    }
    Ok(())
}
fn validate_v0_terminal_rotation(
    doc: &SceneDoc,
    spring: &AvatarSpringBone,
    chain: &AvatarSpring,
) -> Result<()> {
    let Some(first) = chain.joints.first() else {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "spring chain has no joints",
        ));
    };
    let Some(last) = chain.joints.last() else {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "spring chain has no joints",
        ));
    };
    let terminal = &last.node;
    let observable = if let Some(root_bone) = &first.node.bone {
        let armature_node = doc.nodes.get(&first.node.node).ok_or_else(|| {
            PotError::new(ErrorCode::SceneInvalid, "spring armature node is missing")
        })?;
        let armature = armature_node
            .data
            .as_ref()
            .and_then(|data_id| doc.data_blocks.get(data_id))
            .and_then(|data| data.armature.as_ref())
            .ok_or_else(|| {
                PotError::new(ErrorCode::SceneInvalid, "spring armature data is missing")
            })?;
        let Some(terminal_bone) = terminal.bone.as_ref() else {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "spring chain mixes bone and node references",
            ));
        };
        armature.bones.iter().any(|(bone_id, bone)| {
            (bone_id == root_bone || bone_descends_from(armature, root_bone, bone_id))
                && (bone_id == terminal_bone
                    || !chain
                        .joints
                        .iter()
                        .any(|joint| joint.node.bone.as_ref() == Some(bone_id)))
                && v0_ref_rotation_observable(
                    doc,
                    spring,
                    chain,
                    &AvatarNodeRef {
                        node: first.node.node.clone(),
                        bone: Some(bone_id.clone()),
                    },
                    Some(bone),
                )
        })
    } else {
        doc.nodes.iter().any(|(node_id, _)| {
            (node_id == &first.node.node || node_descends_from(doc, &first.node.node, node_id))
                && (node_id == &terminal.node
                    || !chain
                        .joints
                        .iter()
                        .any(|joint| joint.node.bone.is_none() && joint.node.node == *node_id))
                && v0_ref_rotation_observable(
                    doc,
                    spring,
                    chain,
                    &AvatarNodeRef {
                        node: node_id.clone(),
                        bone: None,
                    },
                    None,
                )
        })
    };
    if observable {
        return Err(unsupported_export(
            "vrm.v0.spring_terminal_rotation",
            "VRM 0.x simulates the final spring leaf and recursively includes unlisted descendants, while VRM 1.0 treats its final joint as tail-only; make those extra rotations inert or export VRM 1.0.",
        ));
    }
    Ok(())
}

fn v0_ref_rotation_observable(
    doc: &SceneDoc,
    spring: &AvatarSpringBone,
    chain: &AvatarSpring,
    reference: &AvatarNodeRef,
    bone: Option<&crate::model::Bone>,
) -> bool {
    let node_payload_is_visible = || {
        doc.nodes.get(&reference.node).is_some_and(|node| {
            node.kind != "empty"
                || node.data.is_some()
                || node.primitive.is_some()
                || node.properties.contains_key("instance_collection")
        })
    };
    let bone_is_observable = match (bone, reference.bone.as_ref()) {
        (Some(bone), Some(bone_id)) => {
            avatar_bone_is_mapped(doc, reference, bone_id)
                || doc.nodes.values().any(|node| {
                    node.parent_type == crate::model::ParentType::Bone
                        && node.parent.as_ref() == Some(&reference.node)
                        && node.parent_bone.as_ref() == Some(bone_id)
                })
                || spring_terminal_has_skin_weights(doc, &reference.node, bone)
        }
        (Some(_), None) | (None, Some(_)) => true,
        (None, None) => false,
    };
    (bone.is_none() && node_payload_is_visible())
        || bone_is_observable
        || spring.colliders.iter().any(|collider| {
            collider.node == *reference
                && match &collider.shape {
                    AvatarSpringColliderShape::Sphere { offset, .. } => !is_zero_vec(offset),
                    AvatarSpringColliderShape::Capsule { offset, tail, .. } => {
                        !is_zero_vec(offset) || !is_zero_vec(tail)
                    }
                }
        })
        || spring.springs.iter().any(|other| {
            other.center.as_ref() == Some(reference)
                || (!std::ptr::eq(other, chain)
                    && other.joints.iter().any(|joint| joint.node == *reference))
        })
}

fn avatar_bone_is_mapped(doc: &SceneDoc, terminal: &AvatarNodeRef, bone_id: &Id) -> bool {
    doc.avatar.as_ref().is_some_and(|avatar| {
        avatar.armature == terminal.node && avatar.humanoid.values().any(|mapped| mapped == bone_id)
    })
}

fn spring_terminal_has_skin_weights(
    doc: &SceneDoc,
    armature_node_id: &Id,
    bone: &crate::model::Bone,
) -> bool {
    if !bone.deform {
        return false;
    }
    doc.nodes.values().any(|node| {
        node.modifiers.iter().any(|modifier| {
            modifier.enabled
                && modifier.modifier_type == "armature"
                && modifier.params.get("object").and_then(Value::as_str)
                    == Some(armature_node_id.as_str())
        }) && node
            .data
            .as_ref()
            .and_then(|data_id| doc.data_blocks.get(data_id))
            .filter(|data| data.data_type == "mesh")
            .is_some_and(|data| {
                data.vertex_weights.values().any(|weights| {
                    data.vertex_groups.iter().any(|group| {
                        group.name == bone.name
                            && weights.get(&group.id).is_some_and(|weight| *weight > 0.0)
                    })
                })
            })
    })
}

fn is_zero_vec(value: &[f64; 3]) -> bool {
    value.iter().all(|component| *component == 0.0)
}

fn v0_spring_roots_overlap(doc: &SceneDoc, left: &AvatarNodeRef, right: &AvatarNodeRef) -> bool {
    match (&left.bone, &right.bone) {
        (Some(left_bone), Some(right_bone)) if left.node == right.node => {
            let armature = doc
                .nodes
                .get(&left.node)
                .and_then(|node| node.data.as_ref())
                .and_then(|data_id| doc.data_blocks.get(data_id))
                .and_then(|data| data.armature.as_ref());
            left_bone == right_bone
                || armature.is_some_and(|armature| {
                    bone_descends_from(armature, left_bone, right_bone)
                        || bone_descends_from(armature, right_bone, left_bone)
                })
        }
        (Some(left_bone), Some(right_bone)) => {
            object_root_overlaps_bone_subtree(doc, &left.node, &right.node, right_bone)
                || object_root_overlaps_bone_subtree(doc, &right.node, &left.node, left_bone)
        }
        (None, None) => {
            left.node == right.node
                || node_descends_from(doc, &left.node, &right.node)
                || node_descends_from(doc, &right.node, &left.node)
        }
        (Some(bone), None) => object_root_overlaps_bone_subtree(doc, &right.node, &left.node, bone),
        (None, Some(bone)) => object_root_overlaps_bone_subtree(doc, &left.node, &right.node, bone),
    }
}

fn object_root_overlaps_bone_subtree(
    doc: &SceneDoc,
    object_root: &Id,
    armature_node: &Id,
    root_bone: &Id,
) -> bool {
    object_root == armature_node
        || node_descends_from(doc, armature_node, object_root)
        || doc.nodes.iter().any(|(node_id, node)| {
            (node_id == object_root || node_descends_from(doc, object_root, node_id))
                && node.parent_type == crate::model::ParentType::Bone
                && node.parent.as_ref() == Some(armature_node)
                && node.parent_bone.as_ref().is_some_and(|bone| {
                    bone == root_bone
                        || doc
                            .nodes
                            .get(armature_node)
                            .and_then(|node| node.data.as_ref())
                            .and_then(|data_id| doc.data_blocks.get(data_id))
                            .and_then(|data| data.armature.as_ref())
                            .is_some_and(|armature| bone_descends_from(armature, root_bone, bone))
                })
        })
}

fn bone_descends_from(armature: &crate::model::ArmatureData, root: &Id, candidate: &Id) -> bool {
    let mut current = armature
        .bones
        .get(candidate)
        .and_then(|bone| bone.parent.as_ref());
    while let Some(id) = current {
        if id == root {
            return true;
        }
        current = armature.bones.get(id).and_then(|bone| bone.parent.as_ref());
    }
    false
}

fn node_descends_from(doc: &SceneDoc, root: &Id, candidate: &Id) -> bool {
    let mut current = doc
        .nodes
        .get(candidate)
        .and_then(|node| node.parent.as_ref());
    while let Some(id) = current {
        if id == root {
            return true;
        }
        current = doc.nodes.get(id).and_then(|node| node.parent.as_ref());
    }
    false
}
fn legacy_spring_root(
    doc: &SceneDoc,
    chain: &AvatarSpring,
    mappings: &VrmWriterMappings<'_>,
) -> Result<usize> {
    let root = chain
        .joints
        .first()
        .ok_or_else(|| PotError::new(ErrorCode::SceneInvalid, "spring chain has no joints"))?;
    resolve_node_ref(doc, &root.node, mappings)
}

pub(crate) fn append_extensions(
    doc: &SceneDoc,
    avatar: &AvatarSettings,
    version: AvatarVersion,
    mut mappings: VrmWriterMappings<'_>,
    root: &mut Value,
    root_path: &Path,
    texture_export: &mut TextureExport,
    binary: &mut Vec<u8>,
    views: &mut Vec<Value>,
) -> Result<()> {
    let extension = match version {
        AvatarVersion::V1 => vrm1_extension(
            doc,
            avatar,
            &mappings,
            root,
            root_path,
            texture_export,
            binary,
            views,
        )?,
        AvatarVersion::V0 => {
            split_legacy_mesh_references(avatar, &mut mappings, root)?;
            vrm0_extension(
                doc,
                avatar,
                &mappings,
                root,
                root_path,
                texture_export,
                binary,
                views,
            )?
        }
    };
    let extension_name = match version {
        AvatarVersion::V1 => "VRMC_vrm",
        AvatarVersion::V0 => "VRM",
    };
    set_root_extension(root, extension_name, extension)?;
    mark_extension_used(root, extension_name)?;
    Ok(())
}

pub(crate) fn referenced_armatures(avatar: &AvatarSettings) -> BTreeSet<Id> {
    let mut armatures = BTreeSet::from([avatar.armature.clone()]);
    if let Some(spring) = &avatar.spring_bone {
        for reference in spring
            .colliders
            .iter()
            .map(|collider| &collider.node)
            .chain(
                spring
                    .springs
                    .iter()
                    .flat_map(|chain| chain.joints.iter().map(|joint| &joint.node)),
            )
            .chain(
                spring
                    .springs
                    .iter()
                    .filter_map(|chain| chain.center.as_ref()),
            )
        {
            if reference.bone.is_some() {
                armatures.insert(reference.node.clone());
            }
        }
    }
    armatures
}
fn split_legacy_mesh_references(
    avatar: &AvatarSettings,
    mappings: &mut VrmWriterMappings<'_>,
    root: &mut Value,
) -> Result<()> {
    let mut targeted_nodes = BTreeSet::new();
    if let Some(first_person) = &avatar.first_person {
        targeted_nodes.extend(
            first_person
                .mesh_annotations
                .iter()
                .map(|annotation| annotation.node.clone()),
        );
    }
    for expression in avatar.expressions.values() {
        targeted_nodes.extend(
            expression
                .morph_target_binds
                .iter()
                .map(|bind| bind.node.clone()),
        );
    }
    for node_id in targeted_nodes {
        let mesh_index = index_for(
            mappings.node_mesh_indices,
            &node_id,
            "legacy node-specific mesh reference",
        )?;
        if mappings
            .node_mesh_indices
            .values()
            .filter(|index| **index == mesh_index)
            .count()
            < 2
        {
            continue;
        }
        let node_index = index_for(
            mappings.node_indices,
            &node_id,
            "legacy node-specific mesh reference",
        )?;
        let meshes = root["meshes"]
            .as_array_mut()
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "glTF mesh table is missing"))?;
        let mesh = meshes
            .get(mesh_index)
            .cloned()
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "glTF mesh index is missing"))?;
        let split_index = meshes.len();
        meshes.push(mesh);
        let node = root["nodes"]
            .as_array_mut()
            .and_then(|nodes| nodes.get_mut(node_index))
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "glTF mesh node is missing"))?;
        node["mesh"] = json!(split_index);
        mappings.node_mesh_indices.insert(node_id, split_index);
    }
    Ok(())
}

fn vrm1_extension(
    doc: &SceneDoc,
    avatar: &AvatarSettings,
    mappings: &VrmWriterMappings<'_>,
    root: &mut Value,
    root_path: &Path,
    texture_export: &mut TextureExport,
    binary: &mut Vec<u8>,
    views: &mut Vec<Value>,
) -> Result<Value> {
    let human_bones = human_bones(avatar, mappings, AvatarVersion::V1)?;
    let meta = vrm1_meta(doc, &avatar.meta, root_path, texture_export, binary, views)?;
    let mut extension = json!({
        "specVersion": "1.0",
        "humanoid": { "humanBones": human_bones },
        "meta": meta
    });
    if let Some(first_person) = &avatar.first_person {
        extension["firstPerson"] = vrm1_first_person(first_person, mappings)?;
    }
    if let Some(look_at) = &avatar.look_at {
        extension["lookAt"] = vrm1_look_at(look_at, avatar, mappings)?;
    }
    if !avatar.expressions.is_empty() {
        extension["expressions"] = vrm1_expressions(doc, avatar, mappings, root)?;
    }
    if let Some(spring) = &avatar.spring_bone {
        let spring_extension = vrm1_spring_bone(doc, spring, mappings)?;
        if spring_extension
            .as_object()
            .is_some_and(|value| value.len() > 1)
        {
            set_root_extension(root, "VRMC_springBone", spring_extension)?;
            mark_extension_used(root, "VRMC_springBone")?;
        }
    }
    for (material_id, mtoon) in &avatar.mtoon {
        let material_index = index_for(mappings.material_indices, material_id, "MToon material")?;
        let surface = mappings.surfaces.get(material_id).ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "MToon material surface is missing",
            )
        })?;
        let mtoon_extension = mtoon::v1(
            surface,
            mtoon,
            doc,
            root_path,
            texture_export,
            binary,
            views,
        )?;
        set_material_extension(
            root,
            material_index,
            "VRMC_materials_mtoon",
            mtoon_extension,
        )?;
        mark_extension_used(root, "VRMC_materials_mtoon")?;
    }
    Ok(extension)
}

fn vrm1_meta(
    doc: &SceneDoc,
    meta: &AvatarMeta,
    root_path: &Path,
    texture_export: &mut TextureExport,
    binary: &mut Vec<u8>,
    views: &mut Vec<Value>,
) -> Result<Value> {
    let mut value = json!({
        "name": meta.name,
        "authors": meta.authors,
        "licenseUrl": meta.license_url,
        "avatarPermission": match meta.avatar_permission {
            AvatarPermission::OnlyAuthor => "onlyAuthor",
            AvatarPermission::OnlySeparatelyLicensedPerson => "onlySeparatelyLicensedPerson",
            AvatarPermission::Everyone => "everyone",
        },
        "allowExcessivelyViolentUsage": meta.allow_excessively_violent_usage,
        "allowExcessivelySexualUsage": meta.allow_excessively_sexual_usage,
        "commercialUsage": match meta.commercial_usage {
            AvatarCommercialUsage::PersonalNonProfit => "personalNonProfit",
            AvatarCommercialUsage::PersonalProfit => "personalProfit",
            AvatarCommercialUsage::Corporation => "corporation",
        },
        "allowPoliticalOrReligiousUsage": meta.allow_political_or_religious_usage,
        "allowAntisocialOrHateUsage": meta.allow_antisocial_or_hate_usage,
        "creditNotation": match meta.credit_notation {
            AvatarCreditNotation::Required => "required",
            AvatarCreditNotation::Unnecessary => "unnecessary",
        },
        "allowRedistribution": meta.allow_redistribution,
        "modification": match meta.modification {
            AvatarModification::Prohibited => "prohibited",
            AvatarModification::AllowModification => "allowModification",
            AvatarModification::AllowModificationRedistribution => "allowModificationRedistribution",
        },
    });
    insert_optional(&mut value, "version", meta.version.as_deref());
    insert_optional(
        &mut value,
        "copyrightInformation",
        meta.copyright_information.as_deref(),
    );
    insert_optional(
        &mut value,
        "contactInformation",
        meta.contact_information.as_deref(),
    );
    insert_optional(
        &mut value,
        "thirdPartyLicenses",
        meta.third_party_licenses.as_deref(),
    );
    insert_optional(
        &mut value,
        "otherLicenseUrl",
        meta.other_license_url.as_deref(),
    );
    if !meta.references.is_empty() {
        value["references"] = json!(meta.references);
    }
    if let Some(thumbnail) = &meta.thumbnail
        && !doc.images.contains_key(thumbnail)
    {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "VRM thumbnail image is missing",
        ));
    }
    if let Some(thumbnail) = &meta.thumbnail {
        let image = texture_export.export_image_index(
            thumbnail,
            ImageInterpolation::Linear,
            doc,
            root_path,
            binary,
            views,
        )?;
        value["thumbnailImage"] = json!(image);
    }
    Ok(value)
}

fn human_bones(
    avatar: &AvatarSettings,
    mappings: &VrmWriterMappings<'_>,
    version: AvatarVersion,
) -> Result<Value> {
    let mut bones = serde_json::Map::new();
    for (role, bone_id) in &avatar.humanoid {
        let target_role = if version == AvatarVersion::V0 {
            legacy_humanoid_role(role)
        } else {
            role.as_str()
        };
        let index = index_for_bone(mappings, &avatar.armature, bone_id, target_role)?;
        if version == AvatarVersion::V1 {
            bones.insert(target_role.to_owned(), json!({ "node": index }));
        } else {
            bones.insert(
                target_role.to_owned(),
                json!({ "bone": target_role, "node": index, "useDefaultValues": true }),
            );
        }
    }
    if version == AvatarVersion::V0 {
        Ok(Value::Array(
            bones.into_iter().map(|(_, value)| value).collect(),
        ))
    } else {
        Ok(Value::Object(bones))
    }
}

fn legacy_humanoid_role(role: &str) -> &str {
    match role {
        "leftThumbMetacarpal" => "leftThumbProximal",
        "leftThumbProximal" => "leftThumbIntermediate",
        "leftThumbDistal" => "leftThumbDistal",
        "rightThumbMetacarpal" => "rightThumbProximal",
        "rightThumbProximal" => "rightThumbIntermediate",
        "rightThumbDistal" => "rightThumbDistal",
        _ => role,
    }
}

fn index_for_bone(
    mappings: &VrmWriterMappings<'_>,
    armature: &Id,
    bone_id: &Id,
    role: &str,
) -> Result<usize> {
    mappings
        .bone_indices
        .get(armature)
        .and_then(|bones| bones.get(bone_id))
        .copied()
        .ok_or_else(|| {
            unsupported_export(
                "vrm.humanoid_bone_export",
                &format!("Humanoid role `{role}` references a bone that was not emitted by the glTF skin writer; ensure the avatar armature deforms an exported skinned mesh."),
            )
        })
}

fn vrm1_first_person(
    first_person: &crate::model::AvatarFirstPerson,
    mappings: &VrmWriterMappings<'_>,
) -> Result<Value> {
    let annotations = first_person
        .mesh_annotations
        .iter()
        .map(|annotation| {
            let node = mesh_node_index(mappings, &annotation.node, "first-person mesh annotation")?;
            Ok(json!({
                "node": node,
                "type": match annotation.annotation_type {
                    crate::model::AvatarMeshAnnotationType::Auto => "auto",
                    crate::model::AvatarMeshAnnotationType::Both => "both",
                    crate::model::AvatarMeshAnnotationType::ThirdPersonOnly => "thirdPersonOnly",
                    crate::model::AvatarMeshAnnotationType::FirstPersonOnly => "firstPersonOnly",
                }
            }))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut value = json!({});
    if !annotations.is_empty() {
        value["meshAnnotations"] = json!(annotations);
    }
    Ok(value)
}

fn vrm1_look_at(
    look_at: &AvatarLookAt,
    avatar: &AvatarSettings,
    mappings: &VrmWriterMappings<'_>,
) -> Result<Value> {
    let offset = map_head_offset(avatar, mappings)?;
    Ok(json!({
        "type": match look_at.look_at_type {
            AvatarLookAtType::Bone => "bone",
            AvatarLookAtType::Expression => "expression",
        },
        "offsetFromHeadBone": offset,
        "rangeMapHorizontalInner": range_map_v1(look_at.range_map_horizontal_inner),
        "rangeMapHorizontalOuter": range_map_v1(look_at.range_map_horizontal_outer),
        "rangeMapVerticalDown": range_map_v1(look_at.range_map_vertical_down),
        "rangeMapVerticalUp": range_map_v1(look_at.range_map_vertical_up),
    }))
}

fn range_map_v1(map: crate::model::AvatarRangeMap) -> Value {
    json!({ "inputMaxValue": map.input_max_value, "outputScale": map.output_scale })
}

fn vrm1_expressions(
    doc: &SceneDoc,
    avatar: &AvatarSettings,
    mappings: &VrmWriterMappings<'_>,
    root: &Value,
) -> Result<Value> {
    let mut preset = serde_json::Map::new();
    let mut custom = serde_json::Map::new();
    for (name, expression) in &avatar.expressions {
        let value = expression_v1(doc, expression, mappings, root)?;
        if is_vrm1_preset(name) {
            preset.insert(name.clone(), value);
        } else {
            custom.insert(name.clone(), value);
        }
    }
    let mut expressions = json!({});
    if !preset.is_empty() {
        expressions["preset"] = Value::Object(preset);
    }
    if !custom.is_empty() {
        expressions["custom"] = Value::Object(custom);
    }
    Ok(expressions)
}

fn expression_v1(
    doc: &SceneDoc,
    expression: &AvatarExpression,
    mappings: &VrmWriterMappings<'_>,
    root: &Value,
) -> Result<Value> {
    let mut value = json!({
        "isBinary": expression.is_binary,
        "overrideBlink": override_name(expression.override_blink),
        "overrideMouth": override_name(expression.override_mouth),
        "overrideLookAt": override_name(expression.override_look_at),
    });
    let morphs = expression
        .morph_target_binds
        .iter()
        .map(|bind| {
            let node_index = mesh_node_index(mappings, &bind.node, "expression morph target node")?;
            let target_index = morph_index(doc, mappings, root, &bind.node, &bind.shape_key)?;
            Ok(json!({ "node": node_index, "index": target_index, "weight": bind.weight }))
        })
        .collect::<Result<Vec<_>>>()?;
    if !morphs.is_empty() {
        value["morphTargetBinds"] = json!(morphs);
    }
    let colors = expression
        .material_color_binds
        .iter()
        .map(|bind| {
            let material = index_for(
                mappings.material_indices,
                &bind.material,
                "expression material color bind",
            )?;
            Ok(json!({
                "material": material,
                "type": material_color_type(bind.color_type),
                "targetValue": bind.target_value
            }))
        })
        .collect::<Result<Vec<_>>>()?;
    if !colors.is_empty() {
        value["materialColorBinds"] = json!(colors);
    }
    let transforms = expression
        .texture_transform_binds
        .iter()
        .map(|bind| {
            let material = index_for(
                mappings.material_indices,
                &bind.material,
                "expression texture transform bind",
            )?;
            Ok(json!({
                "material": material,
                "scale": bind.scale,
                "offset": bind.offset
            }))
        })
        .collect::<Result<Vec<_>>>()?;
    if !transforms.is_empty() {
        value["textureTransformBinds"] = json!(transforms);
    }
    Ok(value)
}

fn vrm1_spring_bone(
    doc: &SceneDoc,
    spring: &AvatarSpringBone,
    mappings: &VrmWriterMappings<'_>,
) -> Result<Value> {
    let colliders = spring
        .colliders
        .iter()
        .map(|collider| {
            let node = resolve_node_ref(doc, &collider.node, mappings)?;
            let shape = match &collider.shape {
                AvatarSpringColliderShape::Sphere { offset, radius } => {
                    let offset = map_local_point(mappings, &collider.node, *offset)?;
                    let radius = map_local_radius(
                        mappings,
                        &collider.node,
                        *radius,
                        "spring collider radius",
                    )?;
                    json!({ "sphere": { "offset": offset, "radius": radius } })
                }
                AvatarSpringColliderShape::Capsule {
                    offset,
                    tail,
                    radius,
                } => {
                    let offset = map_local_point(mappings, &collider.node, *offset)?;
                    let tail = map_local_point(mappings, &collider.node, *tail)?;
                    let radius = map_local_radius(
                        mappings,
                        &collider.node,
                        *radius,
                        "spring collider radius",
                    )?;
                    json!({ "capsule": { "offset": offset, "tail": tail, "radius": radius } })
                }
            };
            Ok(json!({ "node": node, "shape": shape }))
        })
        .collect::<Result<Vec<_>>>()?;
    let collider_groups = spring
        .collider_groups
        .iter()
        .map(|group| {
            json!({
                "name": group.name,
                "colliders": group.colliders
            })
        })
        .collect::<Vec<_>>();
    let springs = spring
        .springs
        .iter()
        .map(|chain| spring_chain_v1(doc, chain, mappings))
        .collect::<Result<Vec<_>>>()?;
    let mut value = json!({ "specVersion": "1.0" });
    if !colliders.is_empty() {
        value["colliders"] = json!(colliders);
    }
    if !collider_groups.is_empty() {
        value["colliderGroups"] = json!(collider_groups);
    }
    if !springs.is_empty() {
        value["springs"] = json!(springs);
    }
    Ok(value)
}

fn spring_chain_v1(
    doc: &SceneDoc,
    chain: &AvatarSpring,
    mappings: &VrmWriterMappings<'_>,
) -> Result<Value> {
    let joints = chain
        .joints
        .iter()
        .map(|joint| {
            let node = resolve_node_ref(doc, &joint.node, mappings)?;
            let hit_radius = map_local_radius(
                mappings,
                &joint.node,
                joint.hit_radius,
                "spring joint hit radius",
            )?;
            Ok(json!({
                "node": node,
                "hitRadius": hit_radius,
                "stiffness": joint.stiffness,
                "gravityPower": joint.gravity_power,
                "gravityDir": super::gltf::axis_to_gltf(joint.gravity_dir),
                "dragForce": joint.drag_force
            }))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut value = json!({ "joints": joints });
    if let Some(name) = &chain.name {
        value["name"] = json!(name);
    }
    if let Some(center) = &chain.center {
        value["center"] = json!(resolve_node_ref(doc, center, mappings)?);
    }
    if !chain.collider_groups.is_empty() {
        value["colliderGroups"] = json!(chain.collider_groups);
    }
    Ok(value)
}

fn vrm0_extension(
    doc: &SceneDoc,
    avatar: &AvatarSettings,
    mappings: &VrmWriterMappings<'_>,
    root: &mut Value,
    root_path: &Path,
    texture_export: &mut TextureExport,
    binary: &mut Vec<u8>,
    views: &mut Vec<Value>,
) -> Result<Value> {
    let meta = vrm0_meta(doc, &avatar.meta, root_path, texture_export, binary, views)?;
    let human_bones = human_bones(avatar, mappings, AvatarVersion::V0)?;
    let mut extension = json!({
        "exporterVersion": "potter",
        "meta": meta,
        "humanoid": { "humanBones": human_bones }
    });
    if avatar.first_person.is_some() || avatar.look_at.is_some() {
        extension["firstPerson"] = vrm0_first_person(avatar, mappings)?;
    }
    if !avatar.expressions.is_empty() {
        extension["blendShapeMaster"] = vrm0_expressions(doc, avatar, mappings, root)?;
    }
    if let Some(spring) = &avatar.spring_bone {
        let secondary = vrm0_spring_bone(doc, spring, mappings)?;
        if secondary.as_object().is_some_and(|value| !value.is_empty()) {
            extension["secondaryAnimation"] = secondary;
        }
    }
    if !doc.materials.is_empty() {
        extension["materialProperties"] = Value::Array(vrm0_material_properties(
            doc,
            avatar,
            mappings.surfaces,
            root_path,
            texture_export,
            binary,
            views,
        )?);
    }
    Ok(extension)
}

fn vrm0_meta(
    doc: &SceneDoc,
    meta: &AvatarMeta,
    root_path: &Path,
    texture_export: &mut TextureExport,
    binary: &mut Vec<u8>,
    views: &mut Vec<Value>,
) -> Result<Value> {
    let policy = validate_v0_meta_settings(meta)?;
    let joined_authors = policy.author.is_none().then(|| meta.authors.join("\n"));
    let author = policy.author.or(joined_authors.as_deref()).unwrap_or("");
    let mut value = json!({
        "title": policy.title,
        "author": author,
        "allowedUserName": match policy.allowed_user {
            AvatarLegacyAllowedUser::OnlyAuthor => "OnlyAuthor",
            AvatarLegacyAllowedUser::ExplicitlyLicensedPerson => "ExplicitlyLicensedPerson",
            AvatarLegacyAllowedUser::Everyone => "Everyone",
        },
        "violentUssageName": permission_v0(policy.violent_usage),
        "sexualUssageName": permission_v0(policy.sexual_usage),
        "commercialUssageName": permission_v0(policy.commercial_usage),
        "licenseName": legacy_license_name(policy.license_name),
    });
    insert_optional(&mut value, "version", meta.version.as_deref());
    insert_optional(
        &mut value,
        "contactInformation",
        meta.contact_information.as_deref(),
    );
    let joined_reference = policy
        .reference
        .is_none()
        .then(|| meta.references.join("\n"));
    let reference = policy
        .reference
        .or(joined_reference.as_deref())
        .unwrap_or("");
    insert_optional(
        &mut value,
        "reference",
        (!reference.is_empty()).then_some(reference),
    );
    insert_optional(
        &mut value,
        "otherPermissionUrl",
        policy.other_permission_url,
    );
    if policy.license_name == AvatarLegacyLicenseName::Other {
        insert_optional(&mut value, "otherLicenseUrl", policy.other_license_url);
    }
    if let Some(thumbnail) = &meta.thumbnail {
        let texture = TextureRef {
            image: thumbnail.clone(),
            uv_map: None,
            interpolation: ImageInterpolation::Linear,
        };
        let index = texture_export.sampled_index(
            &reference_texture(&texture),
            doc,
            root_path,
            binary,
            views,
        )?;
        value["texture"] = json!(index);
    }
    Ok(value)
}

fn vrm0_first_person(avatar: &AvatarSettings, mappings: &VrmWriterMappings<'_>) -> Result<Value> {
    let head = avatar.humanoid.get("head").ok_or_else(|| {
        unsupported_export(
            "vrm.v0.first_person",
            "VRM 0.x first-person settings require a mapped head bone.",
        )
    })?;
    let first_person_bone = index_for_bone(mappings, &avatar.armature, head, "head")?;
    let offset = if avatar.look_at.is_some() {
        map_head_offset(avatar, mappings)?
    } else {
        [0.0; 3]
    };
    let mut value = json!({
        "firstPersonBone": first_person_bone,
        "firstPersonBoneOffset": legacy_z_reflected_vec3(offset)
    });
    if let Some(first_person) = &avatar.first_person {
        let annotations = first_person
            .mesh_annotations
            .iter()
            .map(|annotation| {
                let mesh = index_for(
                    mappings.node_mesh_indices,
                    &annotation.node,
                    "first-person mesh annotation",
                )?;
                Ok(json!({
                    "mesh": mesh,
                    "firstPersonFlag": match annotation.annotation_type {
                        crate::model::AvatarMeshAnnotationType::Auto => "Auto",
                        crate::model::AvatarMeshAnnotationType::Both => "Both",
                        crate::model::AvatarMeshAnnotationType::ThirdPersonOnly => "ThirdPersonOnly",
                        crate::model::AvatarMeshAnnotationType::FirstPersonOnly => "FirstPersonOnly",
                    }
                }))
            })
            .collect::<Result<Vec<_>>>()?;
        if !annotations.is_empty() {
            value["meshAnnotations"] = json!(annotations);
        }
    }
    if let Some(look_at) = &avatar.look_at {
        value["lookAtTypeName"] = json!(match look_at.look_at_type {
            AvatarLookAtType::Bone => "Bone",
            AvatarLookAtType::Expression => "BlendShape",
        });
        value["lookAtHorizontalInner"] = range_map_v0(look_at.range_map_horizontal_inner);
        value["lookAtHorizontalOuter"] = range_map_v0(look_at.range_map_horizontal_outer);
        value["lookAtVerticalDown"] = range_map_v0(look_at.range_map_vertical_down);
        value["lookAtVerticalUp"] = range_map_v0(look_at.range_map_vertical_up);
    }
    Ok(value)
}

fn range_map_v0(map: crate::model::AvatarRangeMap) -> Value {
    json!({ "xRange": map.input_max_value, "yRange": map.output_scale })
}

fn legacy_vec3(value: [f64; 3]) -> Value {
    json!({ "x": value[0], "y": value[1], "z": value[2] })
}

fn legacy_z_reflected_vec3(value: [f64; 3]) -> Value {
    legacy_vec3([value[0], value[1], -value[2]])
}

fn vrm0_expressions(
    doc: &SceneDoc,
    avatar: &AvatarSettings,
    mappings: &VrmWriterMappings<'_>,
    root: &Value,
) -> Result<Value> {
    let mut groups = Vec::with_capacity(avatar.expressions.len());
    for (name, expression) in &avatar.expressions {
        let mut group = json!({
            "name": name,
            "presetName": legacy_expression_preset(name),
            "isBinary": expression.is_binary
        });
        let binds = expression
            .morph_target_binds
            .iter()
            .map(|bind| {
                let mesh = index_for(
                    mappings.node_mesh_indices,
                    &bind.node,
                    "legacy expression mesh",
                )?;
                let index = morph_index(doc, mappings, root, &bind.node, &bind.shape_key)?;
                Ok(json!({ "mesh": mesh, "index": index, "weight": bind.weight * 100.0 }))
            })
            .collect::<Result<Vec<_>>>()?;
        if !binds.is_empty() {
            group["binds"] = json!(binds);
        }
        let mut material_values = Vec::new();
        for bind in &expression.material_color_binds {
            let material = doc.materials.get(&bind.material).ok_or_else(|| {
                PotError::new(
                    ErrorCode::SceneInvalid,
                    "expression target material is missing",
                )
            })?;
            material_values.push(json!({
                "materialName": material.name,
                "propertyName": legacy_material_color_property(bind.color_type),
                "targetValue": bind.target_value
            }));
        }
        for bind in &expression.texture_transform_binds {
            let material = doc.materials.get(&bind.material).ok_or_else(|| {
                PotError::new(
                    ErrorCode::SceneInvalid,
                    "expression target material is missing",
                )
            })?;
            material_values.push(json!({
                "materialName": material.name,
                "propertyName": "_MainTex_ST",
                "targetValue": [
                    bind.scale[0],
                    bind.scale[1],
                    bind.offset[0],
                    1.0 - bind.offset[1] - bind.scale[1]
                ]
            }));
        }
        if !material_values.is_empty() {
            group["materialValues"] = json!(material_values);
        }
        groups.push(group);
    }
    Ok(json!({ "blendShapeGroups": groups }))
}

/// VRM 0.x `_MainTex_ST` holds one transform, so a material whose base texture already has a KHR
/// texture transform cannot also take an expression texture-transform bind.
fn ensure_legacy_texture_transform(doc: &SceneDoc, material_id: &Id) -> Result<()> {
    let (surface, _) = resolve_surface(doc, material_id)?;
    if surface
        .base_color
        .texture
        .as_ref()
        .is_some_and(|texture| texture.transform.is_some())
    {
        return Err(unsupported_export(
            "vrm.v0.expression_texture_transform",
            "VRM 0.x MToon `_MainTex_ST` holds one transform; this material's base texture already has a texture transform, so an expression texture-transform bind cannot also target it. Remove one or export VRM 1.0.",
        ));
    }
    Ok(())
}

fn ensure_unique_legacy_material_name(doc: &SceneDoc, material_id: &Id) -> Result<()> {
    let name = doc
        .materials
        .get(material_id)
        .map(|material| material.name.as_str())
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::SceneInvalid,
                "legacy expression material is missing",
            )
        })?;
    if doc
        .materials
        .iter()
        .any(|(id, material)| id != material_id && material.name == name)
    {
        return Err(unsupported_export(
            "vrm.v0.expression_material_name",
            "VRM 0.x material binds resolve by name and this document has duplicate material names; rename the target material before export.",
        ));
    }
    Ok(())
}

fn vrm0_spring_bone(
    doc: &SceneDoc,
    spring: &AvatarSpringBone,
    mappings: &VrmWriterMappings<'_>,
) -> Result<Value> {
    let (collider_groups, group_expansions) = legacy_collider_groups(doc, spring, mappings)?;
    let mut bone_groups = Vec::with_capacity(spring.springs.len());
    for chain in &spring.springs {
        let Some(first) = chain.joints.first() else {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "spring chain has no joints",
            ));
        };
        let bones = vec![legacy_spring_root(doc, chain, mappings)?];
        let collider_group_indices = chain
            .collider_groups
            .iter()
            .flat_map(|index| group_expansions[*index].iter().copied())
            .collect::<Vec<_>>();
        let hit_radius = map_local_radius(
            mappings,
            &first.node,
            first.hit_radius,
            "spring joint hit radius",
        )?;
        // VRM 0.x vectors are Unity-space and the model faces -Z: store z-reflect(Rot_Y180(glTF dir)).
        let g = super::gltf::axis_to_gltf(first.gravity_dir);
        let mut group = json!({
            "stiffiness": first.stiffness,
            "gravityPower": first.gravity_power,
            "gravityDir": legacy_z_reflected_vec3([-g[0], g[1], -g[2]]),
            "dragForce": first.drag_force,
            "hitRadius": hit_radius,
            "bones": bones,
            "colliderGroups": collider_group_indices
        });
        if let Some(name) = &chain.name {
            group["comment"] = json!(name);
        }
        if let Some(center) = &chain.center {
            group["center"] = json!(resolve_node_ref(doc, center, mappings)?);
        }
        bone_groups.push(group);
    }
    let mut value = json!({});
    if !collider_groups.is_empty() {
        value["colliderGroups"] = json!(collider_groups);
    }
    if !bone_groups.is_empty() {
        value["boneGroups"] = json!(bone_groups);
    }
    Ok(value)
}
fn legacy_collider_groups(
    doc: &SceneDoc,
    spring: &AvatarSpringBone,
    mappings: &VrmWriterMappings<'_>,
) -> Result<(Vec<Value>, Vec<Vec<usize>>)> {
    let mut output = Vec::new();
    let mut expansions = Vec::with_capacity(spring.collider_groups.len());
    for group in &spring.collider_groups {
        let mut by_node = Vec::<(usize, Vec<Value>)>::new();
        for collider_index in &group.colliders {
            let collider = spring.colliders.get(*collider_index).ok_or_else(|| {
                PotError::new(
                    ErrorCode::SceneInvalid,
                    "spring collider group index is invalid",
                )
            })?;
            let AvatarSpringColliderShape::Sphere { offset, radius } = &collider.shape else {
                return Err(unsupported_export(
                    "vrm.v0.spring_capsule",
                    "VRM 0.x supports sphere colliders only; replace this capsule with spheres or export VRM 1.0.",
                ));
            };
            let node = resolve_node_ref(doc, &collider.node, mappings)?;
            let offset = map_local_point(mappings, &collider.node, *offset)?;
            let radius =
                map_local_radius(mappings, &collider.node, *radius, "spring collider radius")?;
            let item = json!({
                "offset": legacy_z_reflected_vec3(offset),
                "radius": radius
            });
            if let Some((_, colliders)) = by_node.iter_mut().find(|(index, _)| *index == node) {
                colliders.push(item);
            } else {
                by_node.push((node, vec![item]));
            }
        }
        let mut expanded = Vec::with_capacity(by_node.len());
        for (node, colliders) in by_node {
            expanded.push(output.len());
            output.push(json!({ "node": node, "colliders": colliders }));
        }
        expansions.push(expanded);
    }
    Ok((output, expansions))
}

fn vrm0_material_properties(
    doc: &SceneDoc,
    avatar: &AvatarSettings,
    surfaces: &BTreeMap<Id, ResolvedSurface>,
    root_path: &Path,
    texture_export: &mut TextureExport,
    binary: &mut Vec<u8>,
    views: &mut Vec<Value>,
) -> Result<Vec<Value>> {
    doc.materials
        .iter()
        .map(|(material_id, material)| {
            if let Some(mtoon) = avatar.mtoon.get(material_id) {
                let surface = surfaces.get(material_id).ok_or_else(|| {
                    PotError::new(
                        ErrorCode::InternalError,
                        "MToon material surface is missing",
                    )
                })?;
                mtoon::v0(
                    &material.name,
                    surface,
                    mtoon,
                    doc,
                    root_path,
                    texture_export,
                    binary,
                    views,
                )
            } else {
                Ok(json!({
                    "name": material.name,
                    "shader": "VRM_USE_GLTFSHADER",
                    "renderQueue": -1,
                    "floatProperties": {},
                    "vectorProperties": {},
                    "textureProperties": {},
                    "keywordMap": {},
                    "tagMap": {}
                }))
            }
        })
        .collect()
}

fn resolve_node_ref(
    doc: &SceneDoc,
    reference: &AvatarNodeRef,
    mappings: &VrmWriterMappings<'_>,
) -> Result<usize> {
    if let Some(bone) = &reference.bone {
        return index_for_bone(mappings, &reference.node, bone, "spring bone reference");
    }
    let _ = doc.nodes.get(&reference.node).ok_or_else(|| {
        PotError::new(ErrorCode::SceneInvalid, "avatar node reference is missing")
    })?;
    index_for(
        mappings.node_indices,
        &reference.node,
        "avatar node reference",
    )
}

fn morph_index(
    doc: &SceneDoc,
    mappings: &VrmWriterMappings<'_>,
    root: &Value,
    node_id: &Id,
    shape_key_id: &Id,
) -> Result<usize> {
    let data_id = doc
        .nodes
        .get(node_id)
        .and_then(|node| node.data.as_ref())
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::SceneInvalid,
                "morph target node has no mesh data",
            )
        })?;
    let shape_keys = doc
        .data_blocks
        .get(data_id)
        .and_then(|data| data.shape_keys.as_ref())
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::SceneInvalid,
                "morph target node has no shape keys",
            )
        })?;
    let index = shape_keys
        .keys
        .keys()
        .position(|id| id == shape_key_id)
        .ok_or_else(|| {
            PotError::new(ErrorCode::SceneInvalid, "morph target shape key is missing")
        })?;
    let node_index = index_for(mappings.node_indices, node_id, "morph target node")?;
    let expected_mesh_index = index_for(mappings.node_mesh_indices, node_id, "morph target mesh")?;
    let mesh_index = root["nodes"]
        .as_array()
        .and_then(|nodes| nodes.get(node_index))
        .and_then(|node| node["mesh"].as_u64())
        .and_then(|mesh| usize::try_from(mesh).ok())
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::SceneInvalid,
                "morph target node was not exported as a mesh",
            )
        })?;
    if mesh_index != expected_mesh_index {
        return Err(PotError::new(
            ErrorCode::InternalError,
            "VRM morph target mesh mapping is inconsistent",
        ));
    }
    let target_count = root["meshes"][mesh_index]["primitives"][0]["targets"]
        .as_array()
        .map(Vec::len)
        .unwrap_or_default();
    if index >= target_count {
        return Err(unsupported_export(
            "vrm.expression_morph_target",
            "A referenced shape key was not emitted as a glTF morph target.",
        ));
    }
    Ok(index)
}

fn mesh_node_index(mappings: &VrmWriterMappings<'_>, node_id: &Id, what: &str) -> Result<usize> {
    let _ = index_for(mappings.node_mesh_indices, node_id, what)?;
    index_for(mappings.node_indices, node_id, what)
}

fn index_for<T>(mapping: &std::collections::BTreeMap<Id, T>, id: &Id, what: &str) -> Result<T>
where
    T: Copy,
{
    mapping.get(id).copied().ok_or_else(|| {
        unsupported_export(
            "vrm.reference_export",
            &format!("The {what} `{id}` has no corresponding exported glTF index."),
        )
    })
}

fn is_vrm1_preset(name: &str) -> bool {
    matches!(
        name,
        "neutral"
            | "aa"
            | "ih"
            | "ou"
            | "ee"
            | "oh"
            | "blink"
            | "blinkLeft"
            | "blinkRight"
            | "happy"
            | "angry"
            | "sad"
            | "relaxed"
            | "surprised"
            | "lookUp"
            | "lookDown"
            | "lookLeft"
            | "lookRight"
    )
}

fn legacy_expression_preset(name: &str) -> &'static str {
    match name {
        "neutral" => "neutral",
        "aa" => "a",
        "ih" => "i",
        "ou" => "u",
        "ee" => "e",
        "oh" => "o",
        "blink" => "blink",
        "blinkLeft" => "blink_l",
        "blinkRight" => "blink_r",
        "happy" => "joy",
        "angry" => "angry",
        "sad" => "sorrow",
        "relaxed" => "fun",
        "lookUp" => "lookup",
        "lookDown" => "lookdown",
        "lookLeft" => "lookleft",
        "lookRight" => "lookright",
        _ => "unknown",
    }
}

fn override_name(value: AvatarExpressionOverride) -> &'static str {
    match value {
        AvatarExpressionOverride::None => "none",
        AvatarExpressionOverride::Block => "block",
        AvatarExpressionOverride::Blend => "blend",
    }
}

fn material_color_type(value: AvatarMaterialColorType) -> &'static str {
    match value {
        AvatarMaterialColorType::Color => "color",
        AvatarMaterialColorType::EmissionColor => "emissionColor",
        AvatarMaterialColorType::ShadeColor => "shadeColor",
        AvatarMaterialColorType::MatcapColor => "matcapColor",
        AvatarMaterialColorType::RimColor => "rimColor",
        AvatarMaterialColorType::OutlineColor => "outlineColor",
    }
}

fn legacy_material_color_property(value: AvatarMaterialColorType) -> &'static str {
    match value {
        AvatarMaterialColorType::Color => "_Color",
        AvatarMaterialColorType::EmissionColor => "_EmissionColor",
        AvatarMaterialColorType::ShadeColor => "_ShadeColor",
        AvatarMaterialColorType::MatcapColor => "_SphereAdd",
        AvatarMaterialColorType::RimColor => "_RimColor",
        AvatarMaterialColorType::OutlineColor => "_OutlineColor",
    }
}

fn permission_v0(value: AvatarLegacyBinaryPermission) -> &'static str {
    match value {
        AvatarLegacyBinaryPermission::Disallow => "Disallow",
        AvatarLegacyBinaryPermission::Allow => "Allow",
    }
}
fn legacy_user_rank(value: AvatarLegacyAllowedUser) -> u8 {
    match value {
        AvatarLegacyAllowedUser::OnlyAuthor => 0,
        AvatarLegacyAllowedUser::ExplicitlyLicensedPerson => 1,
        AvatarLegacyAllowedUser::Everyone => 2,
    }
}

fn legacy_license_name(value: AvatarLegacyLicenseName) -> &'static str {
    match value {
        AvatarLegacyLicenseName::RedistributionProhibited => "Redistribution_Prohibited",
        AvatarLegacyLicenseName::Cc0 => "CC0",
        AvatarLegacyLicenseName::CcBy => "CC_BY",
        AvatarLegacyLicenseName::CcByNc => "CC_BY_NC",
        AvatarLegacyLicenseName::CcBySa => "CC_BY_SA",
        AvatarLegacyLicenseName::CcByNcSa => "CC_BY_NC_SA",
        AvatarLegacyLicenseName::CcByNd => "CC_BY_ND",
        AvatarLegacyLicenseName::CcByNcNd => "CC_BY_NC_ND",
        AvatarLegacyLicenseName::Other => "Other",
    }
}

fn insert_optional(value: &mut Value, key: &str, item: Option<&str>) {
    if let Some(item) = item {
        value[key] = json!(item);
    }
}

fn set_root_extension(root: &mut Value, name: &str, extension: Value) -> Result<()> {
    let extensions = root
        .as_object_mut()
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "glTF root is not an object"))?
        .entry("extensions")
        .or_insert_with(|| json!({}));
    let extensions = extensions.as_object_mut().ok_or_else(|| {
        PotError::new(ErrorCode::InternalError, "glTF extensions is not an object")
    })?;
    extensions.insert(name.to_owned(), extension);
    Ok(())
}

fn set_material_extension(
    root: &mut Value,
    material_index: usize,
    name: &str,
    extension: Value,
) -> Result<()> {
    let material = root["materials"]
        .as_array_mut()
        .and_then(|materials| materials.get_mut(material_index))
        .ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "MToon material index is missing")
        })?;
    let extensions = material
        .as_object_mut()
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "glTF material is not an object"))?
        .entry("extensions")
        .or_insert_with(|| json!({}));
    extensions
        .as_object_mut()
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "glTF material extensions is not an object",
            )
        })?
        .insert(name.to_owned(), extension);
    Ok(())
}

fn mark_extension_used(root: &mut Value, name: &str) -> Result<()> {
    let extensions_used = root
        .as_object_mut()
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "glTF root is not an object"))?
        .entry("extensionsUsed")
        .or_insert_with(|| json!([]));
    let extensions_used = extensions_used.as_array_mut().ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "glTF extensionsUsed is not an array",
        )
    })?;
    if !extensions_used
        .iter()
        .any(|value| value.as_str() == Some(name))
    {
        extensions_used.push(json!(name));
        extensions_used.sort_by(|left, right| left.as_str().cmp(&right.as_str()));
    }
    Ok(())
}
