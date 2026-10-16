use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{
    error::{ErrorCode, PotError, Result},
    float::equal_f64,
    model::{Id, SceneDoc, TextureRef},
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AvatarSettings {
    pub armature: Id,
    pub humanoid: BTreeMap<String, Id>,
    pub meta: AvatarMeta,
    #[serde(default)]
    pub expressions: BTreeMap<String, AvatarExpression>,
    #[serde(default)]
    pub first_person: Option<AvatarFirstPerson>,
    #[serde(default)]
    pub look_at: Option<AvatarLookAt>,
    #[serde(default)]
    pub spring_bone: Option<AvatarSpringBone>,
    #[serde(default)]
    pub mtoon: BTreeMap<Id, AvatarMtoon>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AvatarMeta {
    pub name: String,
    #[serde(default)]
    pub version: Option<String>,
    pub authors: Vec<String>,
    #[serde(default)]
    pub copyright_information: Option<String>,
    #[serde(default)]
    pub contact_information: Option<String>,
    #[serde(default)]
    pub references: Vec<String>,
    #[serde(default)]
    pub third_party_licenses: Option<String>,
    #[serde(default)]
    pub thumbnail: Option<Id>,
    pub license_url: String,
    #[serde(default)]
    pub avatar_permission: AvatarPermission,
    #[serde(default)]
    pub allow_excessively_violent_usage: bool,
    #[serde(default)]
    pub allow_excessively_sexual_usage: bool,
    #[serde(default)]
    pub commercial_usage: AvatarCommercialUsage,
    #[serde(default)]
    pub allow_political_or_religious_usage: bool,
    #[serde(default)]
    pub allow_antisocial_or_hate_usage: bool,
    #[serde(default)]
    pub credit_notation: AvatarCreditNotation,
    #[serde(default)]
    pub allow_redistribution: bool,
    #[serde(default)]
    pub modification: AvatarModification,
    #[serde(default)]
    pub other_license_url: Option<String>,
    #[serde(default)]
    pub legacy: Option<AvatarLegacyMeta>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct AvatarLegacyMeta {
    pub title: Option<String>,
    pub author: Option<String>,
    pub reference: Option<String>,
    pub allowed_user: Option<AvatarLegacyAllowedUser>,
    pub violent_usage: Option<AvatarLegacyBinaryPermission>,
    pub sexual_usage: Option<AvatarLegacyBinaryPermission>,
    pub commercial_usage: Option<AvatarLegacyBinaryPermission>,
    pub other_permission_url: Option<String>,
    pub license_name: Option<AvatarLegacyLicenseName>,
    pub other_license_url: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AvatarPermission {
    #[default]
    OnlyAuthor,
    OnlySeparatelyLicensedPerson,
    Everyone,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AvatarCommercialUsage {
    #[default]
    PersonalNonProfit,
    PersonalProfit,
    Corporation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AvatarCreditNotation {
    #[default]
    Required,
    Unnecessary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AvatarModification {
    #[default]
    Prohibited,
    AllowModification,
    AllowModificationRedistribution,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AvatarLegacyAllowedUser {
    OnlyAuthor,
    ExplicitlyLicensedPerson,
    Everyone,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AvatarLegacyBinaryPermission {
    Disallow,
    Allow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AvatarLegacyLicenseName {
    RedistributionProhibited,
    Cc0,
    CcBy,
    CcByNc,
    CcBySa,
    CcByNcSa,
    CcByNd,
    CcByNcNd,
    Other,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AvatarExpression {
    pub is_binary: bool,
    pub override_blink: AvatarExpressionOverride,
    pub override_mouth: AvatarExpressionOverride,
    pub override_look_at: AvatarExpressionOverride,
    pub morph_target_binds: Vec<AvatarMorphTargetBind>,
    pub material_color_binds: Vec<AvatarMaterialColorBind>,
    pub texture_transform_binds: Vec<AvatarTextureTransformBind>,
}

impl Default for AvatarExpression {
    fn default() -> Self {
        Self {
            is_binary: false,
            override_blink: AvatarExpressionOverride::None,
            override_mouth: AvatarExpressionOverride::None,
            override_look_at: AvatarExpressionOverride::None,
            morph_target_binds: Vec::new(),
            material_color_binds: Vec::new(),
            texture_transform_binds: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AvatarExpressionOverride {
    #[default]
    None,
    Block,
    Blend,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AvatarMorphTargetBind {
    pub node: Id,
    pub shape_key: Id,
    pub weight: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AvatarMaterialColorBind {
    pub material: Id,
    pub color_type: AvatarMaterialColorType,
    pub target_value: [f64; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AvatarMaterialColorType {
    Color,
    EmissionColor,
    ShadeColor,
    MatcapColor,
    RimColor,
    OutlineColor,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AvatarTextureTransformBind {
    pub material: Id,
    pub scale: [f64; 2],
    pub offset: [f64; 2],
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AvatarFirstPerson {
    pub mesh_annotations: Vec<AvatarMeshAnnotation>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AvatarMeshAnnotation {
    pub node: Id,
    pub annotation_type: AvatarMeshAnnotationType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AvatarMeshAnnotationType {
    Auto,
    Both,
    ThirdPersonOnly,
    FirstPersonOnly,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AvatarLookAt {
    pub look_at_type: AvatarLookAtType,
    pub offset_from_head_bone: [f64; 3],
    pub range_map_horizontal_inner: AvatarRangeMap,
    pub range_map_horizontal_outer: AvatarRangeMap,
    pub range_map_vertical_down: AvatarRangeMap,
    pub range_map_vertical_up: AvatarRangeMap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AvatarLookAtType {
    Bone,
    Expression,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AvatarRangeMap {
    pub input_max_value: f64,
    pub output_scale: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AvatarSpringBone {
    pub colliders: Vec<AvatarSpringCollider>,
    pub collider_groups: Vec<AvatarSpringColliderGroup>,
    pub springs: Vec<AvatarSpring>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AvatarNodeRef {
    pub node: Id,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bone: Option<Id>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AvatarSpringCollider {
    pub node: AvatarNodeRef,
    pub shape: AvatarSpringColliderShape,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum AvatarSpringColliderShape {
    Sphere {
        offset: [f64; 3],
        radius: f64,
    },
    Capsule {
        offset: [f64; 3],
        tail: [f64; 3],
        radius: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AvatarSpringColliderGroup {
    pub name: String,
    pub colliders: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AvatarSpring {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub center: Option<AvatarNodeRef>,
    pub joints: Vec<AvatarSpringJoint>,
    #[serde(default)]
    pub collider_groups: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AvatarSpringJoint {
    pub node: AvatarNodeRef,
    #[serde(default)]
    pub hit_radius: f64,
    #[serde(default = "spring_default_stiffness")]
    pub stiffness: f64,
    #[serde(default)]
    pub gravity_power: f64,
    #[serde(default = "spring_default_gravity_dir")]
    pub gravity_dir: [f64; 3],
    #[serde(default = "spring_default_drag_force")]
    pub drag_force: f64,
}

fn spring_default_stiffness() -> f64 {
    1.0
}

fn spring_default_gravity_dir() -> [f64; 3] {
    [0.0, 0.0, -1.0]
}

fn spring_default_drag_force() -> f64 {
    0.5
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AvatarMtoon {
    pub transparent_with_z_write: bool,
    pub render_queue_offset_number: i32,
    pub shade_color_factor: [f64; 3],
    pub shade_multiply_texture: Option<TextureRef>,
    pub shading_shift_factor: f64,
    pub shading_shift_texture: Option<AvatarMtoonShadingShiftTexture>,
    pub shading_toony_factor: f64,
    pub gi_equalization_factor: f64,
    pub matcap_factor: [f64; 3],
    pub matcap_texture: Option<TextureRef>,
    pub parametric_rim_color_factor: [f64; 3],
    pub rim_multiply_texture: Option<TextureRef>,
    pub rim_lighting_mix_factor: f64,
    pub parametric_rim_fresnel_power_factor: f64,
    pub parametric_rim_lift_factor: f64,
    pub outline_width_mode: AvatarOutlineWidthMode,
    pub outline_width_factor: f64,
    pub outline_width_multiply_texture: Option<TextureRef>,
    pub outline_color_factor: [f64; 3],
    pub outline_lighting_mix_factor: f64,
    pub uv_animation_mask_texture: Option<TextureRef>,
    pub uv_animation_scroll_x_speed_factor: f64,
    pub uv_animation_scroll_y_speed_factor: f64,
    pub uv_animation_rotation_speed_factor: f64,
    /// Transforms of `MToon` texture slots, keyed by the VRM 1.0 slot field name (for example
    /// `shadeMultiplyTexture`). Each key must name a slot that holds a texture.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub texture_transforms: BTreeMap<String, AvatarTextureTransform>,
}

impl Default for AvatarMtoon {
    fn default() -> Self {
        Self {
            transparent_with_z_write: false,
            render_queue_offset_number: 0,
            shade_color_factor: [1.0; 3],
            shade_multiply_texture: None,
            shading_shift_factor: 0.0,
            shading_shift_texture: None,
            shading_toony_factor: 0.9,
            gi_equalization_factor: 0.9,
            matcap_factor: [1.0; 3],
            matcap_texture: None,
            parametric_rim_color_factor: [0.0; 3],
            rim_multiply_texture: None,
            rim_lighting_mix_factor: 1.0,
            parametric_rim_fresnel_power_factor: 5.0,
            parametric_rim_lift_factor: 0.0,
            outline_width_mode: AvatarOutlineWidthMode::None,
            outline_width_factor: 0.0,
            outline_width_multiply_texture: None,
            outline_color_factor: [0.0; 3],
            outline_lighting_mix_factor: 1.0,
            uv_animation_mask_texture: None,
            uv_animation_scroll_x_speed_factor: 0.0,
            uv_animation_scroll_y_speed_factor: 0.0,
            uv_animation_rotation_speed_factor: 0.0,
            texture_transforms: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AvatarMtoonShadingShiftTexture {
    pub texture: TextureRef,
    pub scale: f64,
}

/// A VRM 1.0 `KHR_texture_transform` on one `MToon` texture slot (`texture_transforms`). The default
/// is the identity, which a stored entry must not be.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AvatarTextureTransform {
    pub offset: [f64; 2],
    pub rotation: f64,
    pub scale: [f64; 2],
}

impl Default for AvatarTextureTransform {
    fn default() -> Self {
        Self {
            offset: [0.0; 2],
            rotation: 0.0,
            scale: [1.0; 2],
        }
    }
}

impl AvatarTextureTransform {
    /// Whether this transform leaves texture coordinates unchanged.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.offset.iter().all(|value| equal_f64(*value, 0.0))
            && equal_f64(self.rotation, 0.0)
            && self.scale.iter().all(|value| equal_f64(*value, 1.0))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AvatarOutlineWidthMode {
    #[default]
    None,
    WorldCoordinates,
    ScreenCoordinates,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AvatarVersion {
    V1,
    V0,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AvatarFeatureDescriptor {
    pub feature_id: &'static str,
    pub limitation: &'static str,
}

pub const AVATAR_FEATURE_DESCRIPTORS: &[AvatarFeatureDescriptor] = &[
    AvatarFeatureDescriptor {
        feature_id: "humanoid",
        limitation: "VRM 0.x uses its legacy human-bone array.",
    },
    AvatarFeatureDescriptor {
        feature_id: "metadata",
        limitation: "VRM 0.x has fewer permission and licensing fields; explicit legacy overrides are available.",
    },
    AvatarFeatureDescriptor {
        feature_id: "expressions",
        limitation: "VRM 0.x texture-transform binds require MToon settings and no metallic/roughness maps or shader graph.",
    },
    AvatarFeatureDescriptor {
        feature_id: "first_person",
        limitation: "VRM 0.x uses firstPersonFlag mesh annotations.",
    },
    AvatarFeatureDescriptor {
        feature_id: "look_at",
        limitation: "VRM 0.x stores look-at controls in legacy firstPerson fields.",
    },
    AvatarFeatureDescriptor {
        feature_id: "spring_bone",
        limitation: "VRM 0.x supports sphere colliders and group-wide joint settings only.",
    },
    AvatarFeatureDescriptor {
        feature_id: "mtoon",
        limitation: "VRM 0.x uses legacy materialProperties.",
    },
];

const REQUIRED_HUMANOID_ROLES: &[&str] = &[
    "hips",
    "spine",
    "head",
    "leftUpperLeg",
    "leftLowerLeg",
    "leftFoot",
    "rightUpperLeg",
    "rightLowerLeg",
    "rightFoot",
    "leftUpperArm",
    "leftLowerArm",
    "leftHand",
    "rightUpperArm",
    "rightLowerArm",
    "rightHand",
];

const HUMANOID_ROLES: &[&str] = &[
    "hips",
    "spine",
    "chest",
    "upperChest",
    "neck",
    "head",
    "leftEye",
    "rightEye",
    "jaw",
    "leftUpperLeg",
    "leftLowerLeg",
    "leftFoot",
    "leftToes",
    "rightUpperLeg",
    "rightLowerLeg",
    "rightFoot",
    "rightToes",
    "leftShoulder",
    "leftUpperArm",
    "leftLowerArm",
    "leftHand",
    "rightShoulder",
    "rightUpperArm",
    "rightLowerArm",
    "rightHand",
    "leftThumbMetacarpal",
    "leftThumbProximal",
    "leftThumbDistal",
    "leftIndexProximal",
    "leftIndexIntermediate",
    "leftIndexDistal",
    "leftMiddleProximal",
    "leftMiddleIntermediate",
    "leftMiddleDistal",
    "leftRingProximal",
    "leftRingIntermediate",
    "leftRingDistal",
    "leftLittleProximal",
    "leftLittleIntermediate",
    "leftLittleDistal",
    "rightThumbMetacarpal",
    "rightThumbProximal",
    "rightThumbDistal",
    "rightIndexProximal",
    "rightIndexIntermediate",
    "rightIndexDistal",
    "rightMiddleProximal",
    "rightMiddleIntermediate",
    "rightMiddleDistal",
    "rightRingProximal",
    "rightRingIntermediate",
    "rightRingDistal",
    "rightLittleProximal",
    "rightLittleIntermediate",
    "rightLittleDistal",
];

/// Whether `role` is an official VRM 1.0 humanoid bone name that Potter can map.
#[must_use]
pub fn is_humanoid_role(role: &str) -> bool {
    HUMANOID_ROLES.contains(&role)
}

pub fn validate_avatar_settings(doc: &SceneDoc, settings: &AvatarSettings) -> Result<()> {
    validate_meta(doc, &settings.meta)?;
    let armature_node = doc
        .nodes
        .get(&settings.armature)
        .ok_or_else(|| avatar_error("Avatar armature node does not exist.", "/avatar/armature"))?;
    if armature_node.kind != "armature" {
        return Err(avatar_error(
            "Avatar armature must reference an armature node.",
            "/avatar/armature",
        ));
    }
    let armature_data = armature_node
        .data
        .as_ref()
        .and_then(|id| doc.data_blocks.get(id))
        .filter(|data| data.data_type == "armature")
        .and_then(|data| data.armature.as_ref())
        .ok_or_else(|| {
            avatar_error(
                "Avatar armature node has no armature data.",
                "/avatar/armature",
            )
        })?;

    for role in settings.humanoid.keys() {
        if !HUMANOID_ROLES.contains(&role.as_str()) {
            return Err(avatar_error(
                format!("Unknown VRM humanoid role `{role}`."),
                "/avatar/humanoid",
            ));
        }
    }
    for role in REQUIRED_HUMANOID_ROLES {
        if !settings.humanoid.contains_key(*role) {
            return Err(avatar_error(
                format!("Required humanoid role `{role}` is missing."),
                "/avatar/humanoid",
            ));
        }
    }
    let mut mapped = BTreeSet::new();
    for (role, bone) in &settings.humanoid {
        if !armature_data.bones.contains_key(bone) {
            return Err(avatar_error(
                format!("Humanoid role `{role}` references a missing armature bone `{bone}`."),
                "/avatar/humanoid",
            ));
        }
        if !mapped.insert(bone) {
            return Err(avatar_error(
                format!("Armature bone `{bone}` is assigned to more than one humanoid role."),
                "/avatar/humanoid",
            ));
        }
    }
    validate_humanoid_hierarchy(settings, armature_data)?;

    for (name, expression) in &settings.expressions {
        if name.trim().is_empty() {
            return Err(avatar_error(
                "Avatar expression names must not be empty.",
                "/avatar/expressions",
            ));
        }
        for bind in &expression.morph_target_binds {
            let node = doc.nodes.get(&bind.node).ok_or_else(|| {
                avatar_error(
                    "Morph target bind node does not exist.",
                    "/avatar/expressions",
                )
            })?;
            let shape_key_exists = node.kind == "mesh"
                && node
                    .data
                    .as_ref()
                    .and_then(|id| doc.data_blocks.get(id))
                    .is_some_and(|data| {
                        data.mesh.is_some()
                            && data
                                .shape_keys
                                .as_ref()
                                .is_some_and(|keys| keys.keys.contains_key(&bind.shape_key))
                    });
            if !shape_key_exists {
                return Err(avatar_error(
                    format!(
                        "Expression `{name}` references a missing shape key `{}` on node `{}`.",
                        bind.shape_key, bind.node
                    ),
                    "/avatar/expressions",
                ));
            }
            finite_range(
                bind.weight,
                0.0,
                1.0,
                "Expression morph weight must be finite and between 0 and 1.",
                "/avatar/expressions",
            )?;
        }
        for bind in &expression.material_color_binds {
            if !doc.materials.contains_key(&bind.material) {
                return Err(avatar_error(
                    format!(
                        "Expression `{name}` references missing material `{}`.",
                        bind.material
                    ),
                    "/avatar/expressions",
                ));
            }
            if matches!(
                bind.color_type,
                AvatarMaterialColorType::ShadeColor
                    | AvatarMaterialColorType::MatcapColor
                    | AvatarMaterialColorType::RimColor
                    | AvatarMaterialColorType::OutlineColor
            ) && !settings.mtoon.contains_key(&bind.material)
            {
                return Err(avatar_error(
                    format!(
                        "Expression `{name}` binds an MToon color on material `{}`, but that material has no MToon settings.",
                        bind.material
                    ),
                    "/avatar/expressions",
                ));
            }
            finite_array(
                &bind.target_value,
                "Expression material color must contain finite values.",
                "/avatar/expressions",
            )?;
        }
        for bind in &expression.texture_transform_binds {
            if !doc.materials.contains_key(&bind.material) {
                return Err(avatar_error(
                    format!(
                        "Expression `{name}` references missing material `{}`.",
                        bind.material
                    ),
                    "/avatar/expressions",
                ));
            }
            finite_array(
                &bind.scale,
                "Expression texture scale must contain finite values.",
                "/avatar/expressions",
            )?;
            finite_array(
                &bind.offset,
                "Expression texture offset must contain finite values.",
                "/avatar/expressions",
            )?;
        }
    }

    if let Some(first_person) = &settings.first_person {
        let mut seen = BTreeSet::new();
        for annotation in &first_person.mesh_annotations {
            let node = doc.nodes.get(&annotation.node).ok_or_else(|| {
                avatar_error(
                    "First-person mesh annotation node does not exist.",
                    "/avatar/first_person/mesh_annotations",
                )
            })?;
            if node.kind != "mesh"
                || node
                    .data
                    .as_ref()
                    .and_then(|id| doc.data_blocks.get(id))
                    .and_then(|data| data.mesh.as_ref())
                    .is_none()
            {
                return Err(avatar_error(
                    "First-person mesh annotation must reference a node with mesh data.",
                    "/avatar/first_person/mesh_annotations",
                ));
            }
            if !seen.insert(&annotation.node) {
                return Err(avatar_error(
                    format!(
                        "Mesh `{}` has duplicate first-person annotations.",
                        annotation.node
                    ),
                    "/avatar/first_person/mesh_annotations",
                ));
            }
        }
    }

    if let Some(look_at) = &settings.look_at {
        finite_array(
            &look_at.offset_from_head_bone,
            "Look-at offset must contain finite values.",
            "/avatar/look_at/offset_from_head_bone",
        )?;
        for range in [
            &look_at.range_map_horizontal_inner,
            &look_at.range_map_horizontal_outer,
            &look_at.range_map_vertical_down,
            &look_at.range_map_vertical_up,
        ] {
            finite_range(
                range.input_max_value,
                0.0,
                180.0,
                "Look-at input_max_value must be between 0 and 180 degrees.",
                "/avatar/look_at",
            )?;
            let max = if look_at.look_at_type == AvatarLookAtType::Expression {
                1.0
            } else {
                180.0
            };
            finite_range(
                range.output_scale,
                0.0,
                max,
                "Look-at output_scale is outside the supported range.",
                "/avatar/look_at",
            )?;
        }
    }

    if let Some(spring) = &settings.spring_bone {
        validate_spring(doc, spring)?;
    }
    for (material, mtoon) in &settings.mtoon {
        if !doc.materials.contains_key(material) {
            return Err(avatar_error(
                format!("MToon settings reference missing material `{material}`."),
                "/avatar/mtoon",
            ));
        }
        validate_mtoon(doc, material, mtoon)?;
    }
    Ok(())
}

fn validate_meta(doc: &SceneDoc, meta: &AvatarMeta) -> Result<()> {
    if meta.name.trim().is_empty()
        || meta.authors.is_empty()
        || meta.authors.iter().any(|author| author.trim().is_empty())
        || meta.license_url.trim().is_empty()
    {
        return Err(avatar_error(
            "Avatar metadata requires a non-empty name, at least one non-empty author, and a non-empty license_url.",
            "/avatar/meta",
        ));
    }
    if let Some(thumbnail) = &meta.thumbnail
        && !doc.images.contains_key(thumbnail)
    {
        return Err(avatar_error(
            format!("Avatar thumbnail image `{thumbnail}` does not exist."),
            "/avatar/meta/thumbnail",
        ));
    }
    if let Some(legacy) = &meta.legacy {
        for (field, value) in [
            ("title", legacy.title.as_deref()),
            ("author", legacy.author.as_deref()),
            ("reference", legacy.reference.as_deref()),
            (
                "other_permission_url",
                legacy.other_permission_url.as_deref(),
            ),
            ("other_license_url", legacy.other_license_url.as_deref()),
        ] {
            if value.is_some_and(|value| value.trim().is_empty()) {
                return Err(avatar_error(
                    format!("Legacy metadata `{field}` must not be empty when supplied."),
                    "/avatar/meta/legacy",
                ));
            }
        }
        if legacy.license_name == Some(AvatarLegacyLicenseName::Other)
            && legacy
                .other_license_url
                .as_deref()
                .is_none_or(|value| value.trim().is_empty())
        {
            return Err(avatar_error(
                "Legacy license_name `other` requires a non-empty other_license_url.",
                "/avatar/meta/legacy/other_license_url",
            ));
        }
    }
    Ok(())
}

const HUMANOID_PARENT_ROLES: &[(&str, Option<&str>, bool)] = &[
    ("hips", None, false),
    ("spine", Some("hips"), false),
    ("chest", Some("spine"), false),
    ("upperChest", Some("chest"), true),
    ("neck", Some("upperChest"), false),
    ("head", Some("neck"), false),
    ("leftEye", Some("head"), false),
    ("rightEye", Some("head"), false),
    ("jaw", Some("head"), false),
    ("leftUpperLeg", Some("hips"), false),
    ("leftLowerLeg", Some("leftUpperLeg"), false),
    ("leftFoot", Some("leftLowerLeg"), false),
    ("leftToes", Some("leftFoot"), false),
    ("rightUpperLeg", Some("hips"), false),
    ("rightLowerLeg", Some("rightUpperLeg"), false),
    ("rightFoot", Some("rightLowerLeg"), false),
    ("rightToes", Some("rightFoot"), false),
    ("leftShoulder", Some("upperChest"), false),
    ("leftUpperArm", Some("leftShoulder"), false),
    ("leftLowerArm", Some("leftUpperArm"), false),
    ("leftHand", Some("leftLowerArm"), false),
    ("rightShoulder", Some("upperChest"), false),
    ("rightUpperArm", Some("rightShoulder"), false),
    ("rightLowerArm", Some("rightUpperArm"), false),
    ("rightHand", Some("rightLowerArm"), false),
    ("leftThumbMetacarpal", Some("leftHand"), false),
    ("leftThumbProximal", Some("leftThumbMetacarpal"), true),
    ("leftThumbDistal", Some("leftThumbProximal"), true),
    ("leftIndexProximal", Some("leftHand"), false),
    ("leftIndexIntermediate", Some("leftIndexProximal"), true),
    ("leftIndexDistal", Some("leftIndexIntermediate"), true),
    ("leftMiddleProximal", Some("leftHand"), false),
    ("leftMiddleIntermediate", Some("leftMiddleProximal"), true),
    ("leftMiddleDistal", Some("leftMiddleIntermediate"), true),
    ("leftRingProximal", Some("leftHand"), false),
    ("leftRingIntermediate", Some("leftRingProximal"), true),
    ("leftRingDistal", Some("leftRingIntermediate"), true),
    ("leftLittleProximal", Some("leftHand"), false),
    ("leftLittleIntermediate", Some("leftLittleProximal"), true),
    ("leftLittleDistal", Some("leftLittleIntermediate"), true),
    ("rightThumbMetacarpal", Some("rightHand"), false),
    ("rightThumbProximal", Some("rightThumbMetacarpal"), true),
    ("rightThumbDistal", Some("rightThumbProximal"), true),
    ("rightIndexProximal", Some("rightHand"), false),
    ("rightIndexIntermediate", Some("rightIndexProximal"), true),
    ("rightIndexDistal", Some("rightIndexIntermediate"), true),
    ("rightMiddleProximal", Some("rightHand"), false),
    ("rightMiddleIntermediate", Some("rightMiddleProximal"), true),
    ("rightMiddleDistal", Some("rightMiddleIntermediate"), true),
    ("rightRingProximal", Some("rightHand"), false),
    ("rightRingIntermediate", Some("rightRingProximal"), true),
    ("rightRingDistal", Some("rightRingIntermediate"), true),
    ("rightLittleProximal", Some("rightHand"), false),
    ("rightLittleIntermediate", Some("rightLittleProximal"), true),
    ("rightLittleDistal", Some("rightLittleIntermediate"), true),
];

fn validate_humanoid_hierarchy(
    settings: &AvatarSettings,
    armature: &crate::model::ArmatureData,
) -> Result<()> {
    for &(role, parent_role, requires_parent) in HUMANOID_PARENT_ROLES {
        let Some(child_bone) = settings.humanoid.get(role) else {
            continue;
        };
        let Some(parent_role) = parent_role else {
            if nearest_mapped_bone_ancestor(armature, &settings.humanoid, child_bone).is_some() {
                return Err(avatar_error(
                    "The hips bone must not have another mapped humanoid bone as an ancestor.",
                    "/avatar/humanoid",
                ));
            }
            continue;
        };
        if requires_parent && !settings.humanoid.contains_key(parent_role) {
            return Err(avatar_error(
                format!("Humanoid role `{role}` requires its parent role `{parent_role}`."),
                "/avatar/humanoid",
            ));
        }
        let mut ancestor_role = Some(parent_role);
        let mut expected_parent = None;
        while let Some(expected_role) = ancestor_role {
            if let Some(parent_bone) = settings.humanoid.get(expected_role) {
                expected_parent = Some((expected_role, parent_bone));
                break;
            }
            ancestor_role = HUMANOID_PARENT_ROLES
                .iter()
                .find(|(candidate, _, _)| *candidate == expected_role)
                .and_then(|(_, parent, _)| *parent);
        }
        let actual_parent = nearest_mapped_bone_ancestor(armature, &settings.humanoid, child_bone);
        if actual_parent != expected_parent.map(|(_, bone)| bone) {
            let expected_role = expected_parent.map_or("no humanoid parent", |(role, _)| role);
            return Err(avatar_error(
                format!(
                    "Humanoid role `{role}` must have nearest mapped parent `{expected_role}`; intermediate bones must not be mapped humanoid roles."
                ),
                "/avatar/humanoid",
            ));
        }
    }
    Ok(())
}

fn nearest_mapped_bone_ancestor<'a>(
    armature: &crate::model::ArmatureData,
    humanoid: &'a BTreeMap<String, Id>,
    bone: &Id,
) -> Option<&'a Id> {
    let mut current = armature
        .bones
        .get(bone)
        .and_then(|bone| bone.parent.as_ref());
    let mut visited = BTreeSet::new();
    while let Some(id) = current {
        if let Some(mapped) = humanoid.values().find(|mapped| *mapped == id) {
            return Some(mapped);
        }
        if !visited.insert(id) {
            return None;
        }
        current = armature.bones.get(id).and_then(|bone| bone.parent.as_ref());
    }
    None
}

fn validate_node_ref(doc: &SceneDoc, reference: &AvatarNodeRef, pointer: &str) -> Result<()> {
    let node = doc.nodes.get(&reference.node).ok_or_else(|| {
        avatar_error(
            format!("Referenced scene node `{}` does not exist.", reference.node),
            pointer,
        )
    })?;
    if let Some(bone) = &reference.bone {
        if node.kind != "armature" {
            return Err(avatar_error(
                "A bone reference must point through an armature node.",
                pointer,
            ));
        }
        let armature = node
            .data
            .as_ref()
            .and_then(|id| doc.data_blocks.get(id))
            .filter(|data| data.data_type == "armature")
            .and_then(|data| data.armature.as_ref())
            .ok_or_else(|| {
                avatar_error("Referenced armature node has no armature data.", pointer)
            })?;
        if !armature.bones.contains_key(bone) {
            return Err(avatar_error(
                format!(
                    "Referenced bone `{bone}` does not exist in armature node `{}`.",
                    reference.node
                ),
                pointer,
            ));
        }
    }
    Ok(())
}
fn spring_ref_is_ancestor(doc: &SceneDoc, ancestor: &AvatarNodeRef, child: &AvatarNodeRef) -> bool {
    let target = (&ancestor.node, ancestor.bone.as_ref());
    let mut current = (&child.node, child.bone.as_ref());
    let mut visited = BTreeSet::new();
    let mut found = false;

    while visited.insert(current) {
        let Some(parent) = spring_ref_parent(doc, current.0, current.1) else {
            return found;
        };
        found |= parent == target;
        current = parent;
    }
    false
}

fn spring_ref_parent<'a>(
    doc: &'a SceneDoc,
    node_id: &'a Id,
    bone_id: Option<&'a Id>,
) -> Option<(&'a Id, Option<&'a Id>)> {
    let node = doc.nodes.get(node_id)?;
    if let Some(bone_id) = bone_id {
        if node.kind != "armature" {
            return None;
        }
        let armature = node
            .data
            .as_ref()
            .and_then(|id| doc.data_blocks.get(id))
            .filter(|data| data.data_type == "armature")
            .and_then(|data| data.armature.as_ref())?;
        let bone = armature.bones.get(bone_id)?;
        if let Some(parent_bone) = bone.parent.as_ref() {
            armature.bones.get(parent_bone)?;
            return Some((node_id, Some(parent_bone)));
        }
        return Some((node_id, None));
    }

    let parent_id = node.parent.as_ref()?;
    let parent_node = doc.nodes.get(parent_id)?;
    match node.parent_type {
        crate::model::ParentType::Object if node.parent_bone.is_none() => Some((parent_id, None)),
        crate::model::ParentType::Bone => {
            if parent_node.kind != "armature" {
                return None;
            }
            let parent_bone = node.parent_bone.as_ref()?;
            let armature = parent_node
                .data
                .as_ref()
                .and_then(|id| doc.data_blocks.get(id))
                .filter(|data| data.data_type == "armature")
                .and_then(|data| data.armature.as_ref())?;
            armature.bones.get(parent_bone)?;
            Some((parent_id, Some(parent_bone)))
        }
        crate::model::ParentType::Object => None,
    }
}

pub(crate) fn validate_spring(doc: &SceneDoc, spring: &AvatarSpringBone) -> Result<()> {
    for (index, collider) in spring.colliders.iter().enumerate() {
        let pointer = format!("/avatar/spring_bone/colliders/{index}");
        validate_node_ref(doc, &collider.node, &pointer)?;
        match &collider.shape {
            AvatarSpringColliderShape::Sphere { offset, radius } => {
                finite_array(
                    offset,
                    "Collider offset must contain finite values.",
                    &pointer,
                )?;
                finite_range(
                    *radius,
                    0.0,
                    f64::MAX,
                    "Collider radius must be finite and non-negative.",
                    &pointer,
                )?;
            }
            AvatarSpringColliderShape::Capsule {
                offset,
                tail,
                radius,
            } => {
                finite_array(
                    offset,
                    "Collider offset must contain finite values.",
                    &pointer,
                )?;
                finite_array(tail, "Collider tail must contain finite values.", &pointer)?;
                finite_range(
                    *radius,
                    0.0,
                    f64::MAX,
                    "Collider radius must be finite and non-negative.",
                    &pointer,
                )?;
            }
        }
    }
    for (group_index, group) in spring.collider_groups.iter().enumerate() {
        if group.name.trim().is_empty() || group.colliders.is_empty() {
            return Err(avatar_error(
                "Collider groups require a non-empty name and at least one collider index.",
                "/avatar/spring_bone/collider_groups",
            ));
        }
        let mut indices = BTreeSet::new();
        for collider in &group.colliders {
            if *collider >= spring.colliders.len() {
                return Err(avatar_error(
                    format!(
                        "Collider group {group_index} references missing collider index {collider} ({} colliders configured).",
                        spring.colliders.len()
                    ),
                    "/avatar/spring_bone/collider_groups",
                ));
            }
            if !indices.insert(collider) {
                return Err(avatar_error(
                    format!("Collider group {group_index} repeats collider index {collider}."),
                    "/avatar/spring_bone/collider_groups",
                ));
            }
        }
    }
    for (spring_index, chain) in spring.springs.iter().enumerate() {
        let pointer = format!("/avatar/spring_bone/springs/{spring_index}");
        if chain.joints.is_empty() {
            return Err(avatar_error(
                "Each spring must contain at least one joint.",
                &pointer,
            ));
        }
        if let Some(center) = &chain.center {
            validate_node_ref(doc, center, &pointer)?;
        }
        let mut previous: Option<&AvatarNodeRef> = None;
        for (joint_index, joint) in chain.joints.iter().enumerate() {
            let joint_pointer = format!("{pointer}/joints/{joint_index}");
            validate_node_ref(doc, &joint.node, &joint_pointer)?;
            finite_range(
                joint.hit_radius,
                0.0,
                f64::MAX,
                "Spring joint hit_radius must be finite and non-negative.",
                &joint_pointer,
            )?;
            finite_range(
                joint.stiffness,
                0.0,
                f64::MAX,
                "Spring joint stiffness must be finite and non-negative.",
                &joint_pointer,
            )?;
            finite_range(
                joint.gravity_power,
                0.0,
                f64::MAX,
                "Spring joint gravity_power must be finite and non-negative.",
                &joint_pointer,
            )?;
            finite_array(
                &joint.gravity_dir,
                "Spring joint gravity_dir must contain finite values.",
                &joint_pointer,
            )?;
            finite_range(
                joint.drag_force,
                0.0,
                1.0,
                "Spring joint drag_force must be between 0 and 1.",
                &joint_pointer,
            )?;
            if previous.is_some_and(|previous| !spring_ref_is_ancestor(doc, previous, &joint.node))
            {
                return Err(avatar_error(
                    "Each spring joint after the first must descend from the previous joint.",
                    &joint_pointer,
                ));
            }
            previous = Some(&joint.node);
        }
        let mut indices = BTreeSet::new();
        for group in &chain.collider_groups {
            if *group >= spring.collider_groups.len() {
                return Err(avatar_error(
                    format!(
                        "Spring references missing collider group index {group} ({} groups configured).",
                        spring.collider_groups.len()
                    ),
                    &pointer,
                ));
            }
            if !indices.insert(group) {
                return Err(avatar_error(
                    format!("Spring repeats collider group index {group}."),
                    &pointer,
                ));
            }
        }
    }
    for (spring_index, chain) in spring.springs.iter().enumerate() {
        let pointer = format!("/avatar/spring_bone/springs/{spring_index}");
        if let Some(center) = &chain.center {
            if !spring_ref_is_same_or_ancestor(doc, center, &chain.joints[0].node) {
                return Err(avatar_error(
                    "Spring center must be the first joint or one of its ancestors.",
                    &pointer,
                ));
            }
            for (other_index, other) in spring.springs.iter().enumerate() {
                if other_index != spring_index
                    && other
                        .joints
                        .iter()
                        .any(|joint| spring_ref_is_same_or_ancestor(doc, &joint.node, center))
                {
                    return Err(avatar_error(
                        format!(
                            "Spring center must not be a joint or descendant of joint in spring {other_index}."
                        ),
                        &pointer,
                    ));
                }
            }
        }
        for (other_index, other) in spring.springs.iter().enumerate().take(spring_index) {
            if spring_chains_overlap(doc, chain, other) {
                return Err(avatar_error(
                    format!(
                        "Spring chain overlaps a joint path already owned by spring {other_index}."
                    ),
                    &pointer,
                ));
            }
        }
    }
    Ok(())
}

fn spring_ref_is_same_or_ancestor(
    doc: &SceneDoc,
    ancestor: &AvatarNodeRef,
    child: &AvatarNodeRef,
) -> bool {
    ancestor == child || spring_ref_is_ancestor(doc, ancestor, child)
}

fn spring_ref_on_segment(
    doc: &SceneDoc,
    start: &AvatarNodeRef,
    end: &AvatarNodeRef,
    point: &AvatarNodeRef,
) -> bool {
    spring_ref_is_same_or_ancestor(doc, start, point)
        && spring_ref_is_same_or_ancestor(doc, point, end)
}

fn spring_chain_contains(doc: &SceneDoc, chain: &AvatarSpring, point: &AvatarNodeRef) -> bool {
    chain.joints.iter().any(|joint| &joint.node == point)
        || chain
            .joints
            .windows(2)
            .any(|pair| spring_ref_on_segment(doc, &pair[0].node, &pair[1].node, point))
}

fn spring_chains_overlap(doc: &SceneDoc, left: &AvatarSpring, right: &AvatarSpring) -> bool {
    left.joints
        .iter()
        .any(|joint| spring_chain_contains(doc, right, &joint.node))
        || right
            .joints
            .iter()
            .any(|joint| spring_chain_contains(doc, left, &joint.node))
        || left.joints.windows(2).any(|left_pair| {
            right.joints.windows(2).any(|right_pair| {
                spring_ref_on_segment(
                    doc,
                    &left_pair[0].node,
                    &left_pair[1].node,
                    &right_pair[0].node,
                ) || spring_ref_on_segment(
                    doc,
                    &left_pair[0].node,
                    &left_pair[1].node,
                    &right_pair[1].node,
                ) || spring_ref_on_segment(
                    doc,
                    &right_pair[0].node,
                    &right_pair[1].node,
                    &left_pair[0].node,
                ) || spring_ref_on_segment(
                    doc,
                    &right_pair[0].node,
                    &right_pair[1].node,
                    &left_pair[1].node,
                )
            })
        })
}

fn validate_mtoon(doc: &SceneDoc, material_id: &Id, material: &AvatarMtoon) -> Result<()> {
    let pointer = |field: &str| format!("/avatar/mtoon/{material_id}/{field}");
    if !(-9..=9).contains(&material.render_queue_offset_number) {
        return Err(avatar_error(
            "MToon render_queue_offset_number must be between -9 and 9.",
            &pointer("render_queue_offset_number"),
        ));
    }
    for (values, label) in [
        (&material.shade_color_factor, "shade_color_factor"),
        (&material.matcap_factor, "matcap_factor"),
        (
            &material.parametric_rim_color_factor,
            "parametric_rim_color_factor",
        ),
        (&material.outline_color_factor, "outline_color_factor"),
    ] {
        finite_color(values, &pointer(label))?;
    }
    for (value, min, max, field) in [
        (
            material.shading_toony_factor,
            0.0,
            1.0,
            "shading_toony_factor",
        ),
        (
            material.gi_equalization_factor,
            0.0,
            1.0,
            "gi_equalization_factor",
        ),
        (
            material.rim_lighting_mix_factor,
            0.0,
            1.0,
            "rim_lighting_mix_factor",
        ),
        (
            material.parametric_rim_fresnel_power_factor,
            0.0,
            f64::MAX,
            "parametric_rim_fresnel_power_factor",
        ),
        (
            material.outline_width_factor,
            0.0,
            f64::MAX,
            "outline_width_factor",
        ),
        (
            material.outline_lighting_mix_factor,
            0.0,
            1.0,
            "outline_lighting_mix_factor",
        ),
    ] {
        finite_range(
            value,
            min,
            max,
            "MToon setting is outside its official range.",
            &pointer(field),
        )?;
    }
    for (value, field) in [
        (material.shading_shift_factor, "shading_shift_factor"),
        (
            material.parametric_rim_lift_factor,
            "parametric_rim_lift_factor",
        ),
        (
            material.uv_animation_scroll_x_speed_factor,
            "uv_animation_scroll_x_speed_factor",
        ),
        (
            material.uv_animation_scroll_y_speed_factor,
            "uv_animation_scroll_y_speed_factor",
        ),
        (
            material.uv_animation_rotation_speed_factor,
            "uv_animation_rotation_speed_factor",
        ),
    ] {
        finite(value, "MToon setting must be finite.", &pointer(field))?;
    }
    if let Some(texture) = &material.shade_multiply_texture {
        validate_texture(doc, texture, &pointer("shade_multiply_texture"))?;
    }
    if let Some(texture) = &material.shading_shift_texture {
        validate_texture(doc, &texture.texture, &pointer("shading_shift_texture"))?;
        finite(
            texture.scale,
            "MToon shading shift texture scale must be finite.",
            &pointer("shading_shift_texture/scale"),
        )?;
    }
    if let Some(texture) = &material.matcap_texture {
        validate_texture(doc, texture, &pointer("matcap_texture"))?;
    }
    if let Some(texture) = &material.rim_multiply_texture {
        validate_texture(doc, texture, &pointer("rim_multiply_texture"))?;
    }
    if let Some(texture) = &material.outline_width_multiply_texture {
        validate_texture(doc, texture, &pointer("outline_width_multiply_texture"))?;
    }
    if let Some(texture) = &material.uv_animation_mask_texture {
        validate_texture(doc, texture, &pointer("uv_animation_mask_texture"))?;
    }
    for (slot, transform) in &material.texture_transforms {
        let location = pointer(&format!("texture_transforms/{slot}"));
        match mtoon_texture_slot_is_set(material, slot) {
            None => {
                return Err(avatar_error(
                    format!("MToon texture transform `{slot}` is not a VRM 1.0 texture slot."),
                    &location,
                ));
            }
            Some(false) => {
                return Err(avatar_error(
                    format!("MToon texture transform `{slot}` needs a texture in that slot."),
                    &location,
                ));
            }
            Some(true) => {}
        }
        if transform.is_identity() {
            return Err(avatar_error(
                "MToon texture transform is the identity; remove the entry instead.",
                &location,
            ));
        }
        finite_array(
            &transform.offset,
            "MToon texture transform offset must be finite.",
            &location,
        )?;
        finite_array(
            &transform.scale,
            "MToon texture transform scale must be finite.",
            &location,
        )?;
        finite(
            transform.rotation,
            "MToon texture transform rotation must be finite.",
            &location,
        )?;
    }
    Ok(())
}

fn validate_texture(doc: &SceneDoc, texture: &TextureRef, pointer: &str) -> Result<()> {
    if !doc.images.contains_key(&texture.image) {
        return Err(avatar_error(
            format!("MToon texture image `{}` does not exist.", texture.image),
            pointer,
        ));
    }
    Ok(())
}

/// Whether the VRM 1.0 `MToon` texture slot `slot` holds a texture; `None` when `slot` is not a slot.
fn mtoon_texture_slot_is_set(material: &AvatarMtoon, slot: &str) -> Option<bool> {
    let set = match slot {
        "shadeMultiplyTexture" => material.shade_multiply_texture.is_some(),
        "shadingShiftTexture" => material.shading_shift_texture.is_some(),
        "matcapTexture" => material.matcap_texture.is_some(),
        "rimMultiplyTexture" => material.rim_multiply_texture.is_some(),
        "outlineWidthMultiplyTexture" => material.outline_width_multiply_texture.is_some(),
        "uvAnimationMaskTexture" => material.uv_animation_mask_texture.is_some(),
        _ => return None,
    };
    Some(set)
}

fn finite_color(values: &[f64; 3], pointer: &str) -> Result<()> {
    for value in values {
        finite_range(
            *value,
            0.0,
            1.0,
            "MToon color factors must be finite and between 0 and 1.",
            pointer,
        )?;
    }
    Ok(())
}

fn finite_array<const N: usize>(values: &[f64; N], message: &str, pointer: &str) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(avatar_error(message, pointer));
    }
    Ok(())
}

fn finite(value: f64, message: &str, pointer: &str) -> Result<()> {
    if !value.is_finite() {
        return Err(avatar_error(message, pointer));
    }
    Ok(())
}

fn finite_range(value: f64, min: f64, max: f64, message: &str, pointer: &str) -> Result<()> {
    if !value.is_finite() || value < min || value > max {
        return Err(avatar_error(message, pointer));
    }
    Ok(())
}

fn avatar_error(message: impl Into<String>, pointer: &str) -> PotError {
    PotError::with_details(
        ErrorCode::InvalidArgument,
        message,
        serde_json::json!({"pointer":pointer}),
    )
}
