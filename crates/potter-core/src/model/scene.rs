use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet};

use crate::{
    color::ColorManagement,
    error::{ErrorCode, PotError, Result},
    float::equal_f64,
    geom::{
        Mesh, lattice::LatticeData, metaball::MetaballData, pointcloud::PointCloudData,
        volume::VolumeData,
    },
    graph::NodeGroup,
    image::{ImageExtension, ImageInterpolation},
    mask::Mask,
    sequencer::Sequencer,
};

use super::{
    AvatarSettings, Id, Registry, Transform,
    rig::{
        ArmatureData, Constraint, ConstraintType, Driver, ParentType, PoseBone, ShapeKeyData,
        VertexGroup,
    },
};

pub const MAX_REVISION: u64 = (1_u64 << 53) - 1;

#[must_use]
pub const fn revision_in_range(revision: u64) -> bool {
    revision <= MAX_REVISION
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneDoc {
    pub schema_version: u32,
    pub scene_id: String,
    pub revision: u64,
    pub active_scene: Id,
    #[serde(default)]
    pub profile: Profile,
    #[serde(default)]
    pub scenes: Registry<Scene>,
    #[serde(default)]
    pub collections: Registry<Collection>,
    #[serde(default)]
    pub nodes: Registry<Node>,
    #[serde(default)]
    pub data_blocks: Registry<DataBlock>,
    #[serde(default)]
    pub materials: Registry<Material>,
    #[serde(default)]
    pub images: Registry<Image>,
    #[serde(default)]
    pub movie_clips: Registry<MovieClip>,
    #[serde(default)]
    pub masks: Registry<Mask>,
    #[serde(default)]
    pub worlds: Registry<World>,
    #[serde(default)]
    pub node_groups: Registry<NodeGroup>,
    #[serde(default)]
    pub actions: Registry<Action>,
    #[serde(default)]
    pub resources: Registry<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub avatar: Option<AvatarSettings>,
    #[serde(default)]
    pub libraries: Registry<Library>,
    #[serde(default)]
    pub compatibility: Map<String, Value>,
    /// Loss ids (`feature_id`) that an `--allow-lossy` import reported and the user accepted. Evaluation
    /// does not refuse the settings they name; it warns and leaves them unevaluated.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub accepted_losses: BTreeSet<String>,
    #[serde(default)]
    pub history: HistoryState,
}
impl SceneDoc {
    #[must_use]
    pub fn new(scene_id: String) -> Self {
        let active_scene = Id::from_static("scene_main");
        let root_collection = Id::from_static("collection_root");
        let view_layer = Id::from_static("view_main");
        let mut scenes = Registry::new();
        scenes.insert(
            active_scene.clone(),
            Scene {
                name: "Scene".to_owned(),
                root_collection: root_collection.clone(),
                view_layers: Registry::from([(
                    view_layer,
                    ViewLayer {
                        name: "View Layer".to_owned(),
                        excluded_collections: Vec::new(),
                    },
                )]),
                eevee_has_local_volume_probe_grids: Some(false),
                cycles_hair_shadow_groups: None,
                frame_current: 1.0,
                frame_start: 1,
                frame_end: 250,
                use_custom_simulation_range: false,
                simulation_frame_start: 1,
                simulation_frame_end: 250,
                fps: 24,
                fps_base: 1.0,
                use_gravity: true,
                sequencer: Sequencer::default(),
                camera: None,
                world: None,
                active_clip: None,
                unit: UnitSettings::default(),
                render: RenderSettings::default(),
                use_compositing: false,
                compositor: None,
                color_management: ColorManagement::default(),
                markers: Vec::new(),
                rigid_body_world: None,
            },
        );
        let collections = Registry::from([(
            root_collection,
            Collection {
                name: "Collection".to_owned(),
                children: Vec::new(),
                objects: Vec::new(),
            },
        )]);
        Self {
            schema_version: 18,
            scene_id,
            revision: 0,
            active_scene,
            profile: Profile {
                blender: "5.2.2".to_owned(),
            },
            scenes,
            collections,
            nodes: Registry::new(),
            data_blocks: Registry::new(),
            materials: Registry::new(),
            images: Registry::new(),
            movie_clips: Registry::new(),
            worlds: Registry::new(),
            masks: Registry::new(),
            node_groups: Registry::new(),
            actions: Registry::new(),
            resources: Registry::new(),
            libraries: Registry::new(),
            compatibility: Map::new(),
            accepted_losses: BTreeSet::new(),
            avatar: None,
            history: HistoryState::default(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != 18 {
            return Err(PotError::with_details(
                ErrorCode::UnsupportedVersion,
                format!("unsupported scene schema_version {}", self.schema_version),
                json!({ "schema_version": self.schema_version }),
            ));
        }
        if !revision_in_range(self.revision) {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "revision exceeds the interoperable integer range",
            ));
        }
        if !uuid::Uuid::parse_str(&self.scene_id)
            .is_ok_and(|scene_id| scene_id.get_version() == Some(uuid::Version::Random))
        {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "scene_id must be a UUID v4",
            ));
        }
        if !self.scenes.contains_key(&self.active_scene) {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "active_scene does not reference a scene",
            ));
        }
        for (resource_id, resource) in &self.resources {
            let resource_kind = crate::model::migrate::inline_resource_kind(resource)?;
            if resource.get("storage").is_some()
                && resource.get("storage").and_then(Value::as_str) != Some("inline")
            {
                return Err(PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "typed resources must use storage `inline`",
                    json!({ "resource": resource_id }),
                ));
            }
            if resource_kind == Some(crate::geom::legacy_texture::LEGACY_TEXTURE_KIND) {
                crate::geom::legacy_texture::validate_record_in_scene(self, resource_id, resource)?;
            }
            if resource_kind == Some("node_bake") {
                crate::graph::validate_node_bake_resource(resource)?;
            }
        }
        for clip in self.movie_clips.values() {
            clip.validate()?;
        }
        for (data_id, block) in &self.data_blocks {
            if let Some(mesh) = &block.mesh {
                crate::geom::uv_layers::validate(mesh)?;
                crate::geom::color_layers::validate(mesh)?;
                crate::geom::clnor::validate(mesh)?;
            }
            if let Some(hair) = &block.hair_curves {
                if !matches!(hair.selection_domain.as_str(), "POINT" | "CURVE") {
                    return Err(PotError::with_details(
                        ErrorCode::SceneInvalid,
                        "hair Curves selection domain is invalid",
                        json!({"data_id":data_id,"selection_domain":hair.selection_domain}),
                    ));
                }
                if let Some(surface_id) = &hair.surface
                    && !self.nodes.get(surface_id).is_some_and(|node| {
                        node.kind == "mesh"
                            && node
                                .data
                                .as_ref()
                                .and_then(|data_id| self.data_blocks.get(data_id))
                                .is_some_and(|data| data.mesh.is_some())
                    })
                {
                    return Err(PotError::with_details(
                        ErrorCode::SceneInvalid,
                        "hair Curves surface must reference a mesh object",
                        json!({"data_id":data_id,"surface":surface_id}),
                    ));
                }
            }
        }
        for (world_id, world) in &self.worlds {
            if world.eevee_sun_threshold.is_some_and(|threshold| {
                !threshold.is_finite() || !(0.0..=f64::from(f32::MAX)).contains(&threshold)
            }) {
                return Err(PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "world eevee_sun_threshold must be in Blender's 0..=FLT_MAX range",
                    json!({"world_id":world_id}),
                ));
            }
            for (setting, value, maximum) in [
                (
                    "eevee_sun_angle",
                    world.eevee_sun_angle,
                    f64::from(std::f32::consts::PI),
                ),
                (
                    "eevee_sun_shadow_maximum_resolution",
                    world.eevee_sun_shadow_maximum_resolution,
                    f64::from(f32::MAX),
                ),
                (
                    "eevee_sun_shadow_jitter_overblur",
                    world.eevee_sun_shadow_jitter_overblur,
                    100.0,
                ),
                (
                    "eevee_sun_shadow_filter_radius",
                    world.eevee_sun_shadow_filter_radius,
                    f64::from(f32::MAX),
                ),
            ] {
                if value
                    .is_some_and(|value| !value.is_finite() || !(0.0..=maximum).contains(&value))
                {
                    return Err(PotError::with_details(
                        ErrorCode::SceneInvalid,
                        format!("world {setting} is outside Blender's RNA range"),
                        json!({"world_id":world_id}),
                    ));
                }
            }
        }
        for (scene_id, scene) in &self.scenes {
            scene.render.validate()?;
            if let Some(groups) = &scene.cycles_hair_shadow_groups {
                let mut group_keys = Vec::new();
                for group in groups {
                    let stable_geometry_key = !matches!(
                        &group.geometry_key,
                        CyclesHairShadowGeometryKey::Data { data_id: None }
                    );
                    if stable_geometry_key {
                        let group_key = (group.view_layer_id.clone(), group.geometry_key.clone());
                        if group_keys.contains(&group_key) {
                            return Err(PotError::with_details(
                                ErrorCode::SceneInvalid,
                                "Cycles Hair shadow groups duplicate a stable view-layer GeometryKey",
                                json!({"scene_id":scene_id,"view_layer_id":group.view_layer_id}),
                            ));
                        }
                        group_keys.push(group_key);
                    }
                    if !scene.view_layers.contains_key(&group.view_layer_id) {
                        return Err(PotError::with_details(
                            ErrorCode::SceneInvalid,
                            "Cycles Hair shadow group references a missing view layer",
                            json!({"scene_id":scene_id,"view_layer_id":group.view_layer_id}),
                        ));
                    }
                    let mut member_nodes = Vec::new();
                    let mut member_node_set = BTreeSet::new();
                    let mut instance_keys = BTreeSet::new();
                    let mut invalid_member = false;
                    for member in &group.members {
                        let instance = &member.instance;
                        if !self.nodes.contains_key(&instance.real_object_node_id)
                            || instance
                                .parent_node_id
                                .as_ref()
                                .is_some_and(|parent| !self.nodes.contains_key(parent))
                            || (!instance.is_instance
                                && (instance.parent_node_id.as_ref()
                                    != Some(&instance.real_object_node_id)
                                    || instance.persistent_id != [0; 8]))
                            || !instance_keys.insert(instance)
                        {
                            invalid_member = true;
                        }
                        if member_node_set.insert(instance.real_object_node_id.clone()) {
                            member_nodes.push(instance.real_object_node_id.clone());
                        }
                        if member
                            .object_to_world
                            .iter()
                            .flatten()
                            .any(|value| !value.is_finite())
                            || !member.object_color.iter().all(|value| value.is_finite())
                            || member
                                .slot_0_material_id
                                .as_ref()
                                .is_some_and(|material_id| {
                                    !self.materials.contains_key(material_id)
                                })
                        {
                            invalid_member = true;
                        }
                    }
                    if group.members.is_empty()
                        || invalid_member
                        || group.member_node_ids.first() != Some(&group.representative_node_id)
                        || group.member_node_ids != member_nodes
                        || group.members.first().is_none_or(|member| {
                            member.instance.real_object_node_id != group.representative_node_id
                        })
                    {
                        return Err(PotError::with_details(
                            ErrorCode::SceneInvalid,
                            "Cycles Hair shadow group has invalid representative or member records",
                            json!({"scene_id":scene_id,"view_layer_id":group.view_layer_id}),
                        ));
                    }
                    let representative_node = &self.nodes[&group.representative_node_id];
                    let source_hair_curves = representative_node
                        .data
                        .as_ref()
                        .and_then(|data_id| self.data_blocks.get(data_id))
                        .and_then(|data| data.hair_curves.as_ref());
                    let curve_indices_invalid =
                        group.source_curve_indices.as_ref().is_some_and(|indices| {
                            source_hair_curves.is_none_or(|hair| {
                                indices.iter().any(|index| {
                                    usize::try_from(*index)
                                        .ok()
                                        .is_none_or(|index| index >= hair.curves.len())
                                })
                            })
                        });
                    let source_input_fingerprint_invalid = group
                        .source_input_fingerprint
                        .as_deref()
                        .is_some_and(|fingerprint| {
                            !fingerprint.strip_prefix("sha256:").is_some_and(|hex| {
                                hex.len() == 64 && hex.bytes().all(|byte| byte.is_ascii_hexdigit())
                            })
                        });
                    let geometry_key_valid = match &group.geometry_key {
                        CyclesHairShadowGeometryKey::Object { node_id } => {
                            node_id == &group.representative_node_id
                        }
                        CyclesHairShadowGeometryKey::Data { data_id } => {
                            data_id.as_ref().is_none_or(|data_id| {
                                self.data_blocks.contains_key(data_id)
                                    && representative_node.data.as_ref() == Some(data_id)
                            })
                        }
                    };
                    if group.evaluated_hair_curves.as_ref().is_some_and(|hair| {
                        !matches!(hair.selection_domain.as_str(), "POINT" | "CURVE")
                            || hair
                                .surface
                                .as_ref()
                                .is_some_and(|surface| !self.nodes.contains_key(surface))
                    }) || curve_indices_invalid
                        || source_input_fingerprint_invalid
                        || !geometry_key_valid
                    {
                        return Err(PotError::with_details(
                            ErrorCode::SceneInvalid,
                            "Cycles Hair shadow group geometry context is invalid",
                            json!({"scene_id":scene_id,"view_layer_id":group.view_layer_id}),
                        ));
                    }
                }
            }
            if let Some(camera_id) = &scene.render.dicing_camera
                && !self.is_camera_object(camera_id)
            {
                return Err(PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "render dicing_camera must reference a camera object",
                    json!({"scene_id":scene_id,"camera_id":camera_id}),
                ));
            }
            if let Some(curve) = &scene.color_management.curve {
                curve.validate()?;
            }
            if let Some(world) = &scene.rigid_body_world
                && (!world.gravity.iter().all(|value| value.is_finite())
                    || world.substeps == 0
                    || world.substeps > 128
                    || world.solver_iterations == 0
                    || world.solver_iterations > 128
                    || world.frame_start > world.frame_end)
            {
                return Err(PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "rigid body world settings are invalid",
                    json!({"scene_id":scene_id}),
                ));
            }
        }
        for (node_id, node) in &self.nodes {
            if let Some(index) = node.active_vertex_group_index {
                let valid = node
                    .data
                    .as_ref()
                    .and_then(|data_id| self.data_blocks.get(data_id))
                    .is_some_and(|data| index < data.vertex_groups.len());
                if !valid {
                    return Err(PotError::with_details(
                        ErrorCode::SceneInvalid,
                        "active vertex-group index is out of range",
                        json!({"node_id":node_id,"active_vertex_group_index":index}),
                    ));
                }
            }
            if let Some(captured) = &node.blender_matrix_basis
                && !captured.matrix_basis.iter().all(|value| value.is_finite())
            {
                return Err(PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "captured Blender matrix_basis values must be finite",
                    json!({"node_id":node_id}),
                ));
            }
            if let Some(body) = &node.rigid_body
                && (!body.mass.is_finite()
                    || body.mass < 0.0
                    || (body.body_type == RigidBodyType::Active && body.mass == 0.0)
                    || !body.friction.is_finite()
                    || body.friction < 0.0
                    || !body.restitution.is_finite()
                    || !(0.0..=1.0).contains(&body.restitution)
                    || !body.linear_damping.is_finite()
                    || body.linear_damping < 0.0
                    || !body.angular_damping.is_finite()
                    || body.angular_damping < 0.0
                    || !body.initial_velocity.iter().all(|value| value.is_finite()))
            {
                return Err(PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "rigid body settings are invalid",
                    json!({"node_id":node_id}),
                ));
            }
            if node.rigid_body.is_some()
                && (node.kind != "mesh"
                    || node
                        .data
                        .as_ref()
                        .and_then(|data_id| self.data_blocks.get(data_id))
                        .and_then(|data| data.mesh.as_ref())
                        .is_none())
            {
                return Err(PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "rigid bodies require a mesh Object with mesh data",
                    json!({"node_id":node_id}),
                ));
            }
            if let Some(force) = &node.force_field
                && (node.kind != "empty"
                    || !force.strength.is_finite()
                    || !force.falloff.is_finite()
                    || force.falloff < 0.0)
            {
                return Err(PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "force fields require an Empty and finite non-negative settings",
                    json!({"node_id":node_id}),
                ));
            }
            if node.kind == "armature"
                && node
                    .data
                    .as_ref()
                    .and_then(|data_id| self.data_blocks.get(data_id))
                    .and_then(|data| data.armature.as_ref())
                    .is_none()
            {
                return Err(PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "armature Object requires an armature Data-Block",
                    json!({"node_id":node_id}),
                ));
            }
            if node.kind == "grease_pencil"
                && node
                    .data
                    .as_ref()
                    .and_then(|data_id| self.data_blocks.get(data_id))
                    .and_then(|data| data.grease_pencil.as_ref())
                    .is_none()
            {
                return Err(PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "Grease Pencil Object requires a Grease Pencil Data-Block",
                    json!({"node_id":node_id}),
                ));
            }
            match (
                node.parent_type,
                node.parent.as_ref(),
                node.parent_bone.as_ref(),
            ) {
                (ParentType::Object, _, Some(_)) | (ParentType::Bone, None, _) => {
                    return Err(PotError::with_details(
                        ErrorCode::SceneInvalid,
                        "bone parenting fields are inconsistent",
                        json!({"node_id":node_id}),
                    ));
                }
                (ParentType::Bone, Some(parent_id), Some(bone_id)) => {
                    let valid_bone = self
                        .nodes
                        .get(parent_id)
                        .filter(|parent| parent.kind == "armature")
                        .and_then(|parent| parent.data.as_ref())
                        .and_then(|data_id| self.data_blocks.get(data_id))
                        .and_then(|data| data.armature.as_ref())
                        .is_some_and(|armature| armature.bones.contains_key(bone_id));
                    if !valid_bone {
                        return Err(PotError::with_details(
                            ErrorCode::SceneInvalid,
                            "bone parent reference does not exist",
                            json!({"node_id":node_id,"parent":parent_id,"bone_id":bone_id}),
                        ));
                    }
                }
                (ParentType::Bone, Some(_), None) => {
                    return Err(PotError::with_details(
                        ErrorCode::SceneInvalid,
                        "bone parenting requires a bone ID",
                        json!({"node_id":node_id}),
                    ));
                }
                _ => {}
            }
            for constraint in &node.constraints {
                if let Some(owner_bone) = &constraint.owner_bone {
                    let valid_owner = node.kind == "armature"
                        && node
                            .data
                            .as_ref()
                            .and_then(|data_id| self.data_blocks.get(data_id))
                            .and_then(|data| data.armature.as_ref())
                            .is_some_and(|armature| armature.bones.contains_key(owner_bone));
                    if !valid_owner {
                        return Err(PotError::with_details(
                            ErrorCode::SceneInvalid,
                            "constraint owner_bone does not reference a bone on its armature node",
                            json!({"node_id":node_id,"constraint_id":constraint.id,"owner_bone":owner_bone}),
                        ));
                    }
                }
                if constraint.inverse_frame.is_some()
                    && (constraint.constraint_type != ConstraintType::ObjectSolver
                        || constraint.inverse_matrix.is_none())
                    || constraint.inverse_matrix.is_some_and(|inverse| {
                        constraint.constraint_type != ConstraintType::ObjectSolver
                            || inverse.iter().any(|component| !component.is_finite())
                    })
                    || constraint
                        .inverse_frame
                        .is_some_and(|frame| !frame.is_finite())
                {
                    return Err(PotError::with_details(
                        ErrorCode::SceneInvalid,
                        "constraint inverse_matrix must be a finite Object Solver matrix",
                        json!({"node_id":node_id,"constraint_id":constraint.id}),
                    ));
                }
            }
        }
        for (image_id, image) in &self.images {
            let dimensions_valid = |width: u32, height: u32| {
                u64::from(width)
                    .checked_mul(u64::from(height))
                    .is_some_and(|count| count > 0 && count <= crate::image::MAX_IMAGE_PIXELS)
            };
            let valid_hash = |digest: &str| {
                digest.strip_prefix("sha256:").is_some_and(|hex| {
                    hex.len() == 64 && hex.bytes().all(|byte| byte.is_ascii_hexdigit())
                })
            };
            if image.base_is_missing() {
                if (image.width, image.height) != (0, 0) {
                    return Err(PotError::with_details(
                        ErrorCode::SceneInvalid,
                        "a missing image base has no size",
                        json!({"image_id":image_id,"width":image.width,"height":image.height}),
                    ));
                }
            } else {
                if !dimensions_valid(image.width, image.height) {
                    return Err(PotError::with_details(
                        ErrorCode::SceneInvalid,
                        "image dimensions are invalid or exceed the pixel limit",
                        json!({"image_id":image_id,"width":image.width,"height":image.height}),
                    ));
                }
                match image.source {
                    ImageSource::Generated | ImageSource::Packed if image.blob.is_none() => {
                        return Err(PotError::with_details(
                            ErrorCode::SceneInvalid,
                            "generated or packed image requires a content-addressed blob",
                            json!({"image_id":image_id}),
                        ));
                    }
                    ImageSource::File
                        if image.blob.is_none()
                            && (image.source_path.is_none() || image.source_hash.is_none()) =>
                    {
                        return Err(PotError::with_details(
                            ErrorCode::SceneInvalid,
                            "linked image requires source_path and source_hash",
                            json!({"image_id":image_id}),
                        ));
                    }
                    _ => {}
                }
            }
            if image
                .blob
                .as_deref()
                .is_some_and(|digest| !valid_hash(digest))
                || image
                    .source_hash
                    .as_deref()
                    .is_some_and(|digest| !valid_hash(digest))
            {
                return Err(PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "image hashes must be sha256 digests",
                    json!({"image_id":image_id}),
                ));
            }
            let mut tile_numbers = std::collections::BTreeSet::new();
            for tile in &image.tiles {
                let content_valid = match tile.blob.as_deref() {
                    Some(blob) => dimensions_valid(tile.width, tile.height) && valid_hash(blob),
                    None => tile.width == 0 && tile.height == 0,
                };
                if tile.number <= 1001 || !tile_numbers.insert(tile.number) || !content_valid {
                    return Err(PotError::with_details(
                        ErrorCode::SceneInvalid,
                        "UDIM tile metadata is invalid",
                        json!({"image_id":image_id,"tile":tile.number}),
                    ));
                }
            }
        }
        for (material_id, material) in &self.materials {
            for texture in [
                material.base_color_texture.as_ref(),
                material.roughness_texture.as_ref(),
                material.metallic_texture.as_ref(),
                material.normal_texture.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                if !self.images.contains_key(&texture.image) {
                    return Err(PotError::with_details(
                        ErrorCode::SceneInvalid,
                        "material texture references a missing image",
                        json!({"material_id":material_id,"image_id":texture.image}),
                    ));
                }
            }
        }
        if let Some(avatar) = &self.avatar {
            crate::model::validate_avatar_settings(self, avatar).map_err(|error| {
                PotError::with_details(ErrorCode::SceneInvalid, error.message, error.details)
            })?;
        }
        Ok(())
    }

    pub(crate) fn cycles_hair_shadow_source_fingerprint(
        &self,
        scene_id: &Id,
        group: &CyclesHairShadowGroup,
    ) -> Result<Option<String>> {
        cycles_hair_shadow_source_fingerprint(self, scene_id, group)
    }

    pub(crate) fn is_camera_object(&self, camera_id: &Id) -> bool {
        self.nodes
            .get(camera_id)
            .filter(|node| node.kind == "camera")
            .and_then(|node| node.data.as_ref())
            .and_then(|data_id| self.data_blocks.get(data_id))
            .is_some_and(|data| data.camera.is_some())
    }
}

fn cycles_hair_shadow_source_fingerprint(
    doc: &SceneDoc,
    scene_id: &Id,
    group: &CyclesHairShadowGroup,
) -> Result<Option<String>> {
    let Some(scene) = doc.scenes.get(scene_id) else {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "Cycles Hair provenance names a missing Scene",
        ));
    };
    let Some(view_layer) = scene.view_layers.get(&group.view_layer_id) else {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "Cycles Hair provenance names a missing View Layer",
        ));
    };
    let mut closure = CyclesHairSourceClosure {
        complete: true,
        ..CyclesHairSourceClosure::default()
    };
    let member_node_ids = group
        .member_node_ids
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let instance_member_node_ids = group
        .members
        .iter()
        .filter(|member| member.instance.is_instance)
        .map(|member| member.instance.real_object_node_id.clone())
        .collect::<BTreeSet<_>>();
    closure
        .instance_context_nodes
        .clone_from(&instance_member_node_ids);
    let instance_data_ids = group
        .members
        .iter()
        .filter(|member| member.instance.is_instance)
        .filter_map(|member| {
            doc.nodes
                .get(&member.instance.real_object_node_id)
                .and_then(|node| node.data.clone())
        })
        .collect::<BTreeSet<_>>();
    for member in &group.members {
        closure.visit_node(
            doc,
            &member.instance.real_object_node_id,
            member.instance.is_instance,
            0,
        );
        if member.instance.is_instance
            && let Some(parent) = &member.instance.parent_node_id
        {
            closure.visit_node(doc, parent, true, 0);
        }
    }
    closure.visit_scene_collection(
        doc,
        &scene.root_collection,
        &view_layer.excluded_collections,
        &member_node_ids,
        &instance_member_node_ids,
        0,
    );
    if !closure.complete {
        return Ok(None);
    }

    let nodes = closure
        .nodes
        .iter()
        .map(|id| -> Result<_> {
            let node = &doc.nodes[id];
            Ok((
                id.to_string(),
                cycles_hair_node_geometry_state(
                    node,
                    closure.transforms.contains(id),
                    instance_member_node_ids.contains(id),
                )?,
            ))
        })
        .collect::<Result<serde_json::Map<_, _>>>()?;
    let data_blocks = closure
        .data_blocks
        .iter()
        .map(|id| {
            let data = &doc.data_blocks[id];
            let state = serde_json::json!({
                "type": data.data_type,
                "descriptor": data.descriptor,
                "mesh": data.mesh,
                "grease_pencil": data.grease_pencil,
                "armature": data.armature,
                "shape_keys": data.shape_keys,
                "vertex_groups": data.vertex_groups,
                "vertex_weights": data.vertex_weights,
                "curve": data.curve,
                "surface": data.surface,
                "text": data.text,
                "hair_curves": data.hair_curves,
                "metaball": data.metaball,
                "lattice": data.lattice,
                "pointcloud": data.pointcloud,
                "volume": data.volume,
                "material_slots": instance_data_ids.contains(id).then_some(&data.materials),
                "material_slot_count": data.materials.len(),
            });
            (id.to_string(), state)
        })
        .collect::<serde_json::Map<_, _>>();
    let scene_membership = closure
        .scene_membership
        .iter()
        .map(|(id, objects)| (id.to_string(), serde_json::json!(objects)))
        .collect::<serde_json::Map<_, _>>();

    let collections = closure
        .collections
        .iter()
        .map(|id| {
            let collection = &doc.collections[id];
            (
                id.to_string(),
                serde_json::json!({
                    "children": collection.children,
                    "objects": collection.objects,
                }),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    let node_groups = closure
        .node_groups
        .iter()
        .map(|id| {
            (
                id.to_string(),
                cycles_hair_node_group_geometry_state(&doc.node_groups[id]),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    let actions = closure
        .actions
        .iter()
        .map(|(id, (include_all, include_instance_context))| {
            (
                id.to_string(),
                cycles_hair_action_geometry_state(
                    &doc.actions[id],
                    *include_all,
                    *include_instance_context,
                    &closure.nodes,
                ),
            )
        })
        .collect::<serde_json::Map<_, _>>();

    let images = closure
        .images
        .iter()
        .map(|id| {
            serde_json::to_value(&doc.images[id])
                .map(|state| (id.to_string(), state))
                .map_err(PotError::internal_json)
        })
        .collect::<Result<serde_json::Map<_, _>>>()?;
    let member_instance_keys = group
        .members
        .iter()
        .map(|member| &member.instance)
        .collect::<Vec<_>>();
    let instance_contexts = group
        .members
        .iter()
        .filter(|member| member.instance.is_instance)
        .collect::<Vec<_>>();
    let state = serde_json::json!({
        "scene": {
            "frame_current": scene.frame_current,
            "root_collection": scene.root_collection,
            "use_gravity": scene.use_gravity,
        },
        "view_layer": {
            "id": group.view_layer_id,
            "excluded_collections": view_layer.excluded_collections,
            "candidate_membership": scene_membership,
        },
        "geometry_key": group.geometry_key,
        "representative_node_id": group.representative_node_id,
        "member_node_ids": group.member_node_ids,
        "member_instance_keys": member_instance_keys,
        "instance_contexts": instance_contexts,
        "source_curve_indices": group.source_curve_indices,
        "nodes": nodes,
        "data_blocks": data_blocks,
        "collections": collections,
        "node_groups": node_groups,
        "actions": actions,
        "images": images,
    });
    let bytes = crate::hash::canonicalize(&state)?;
    Ok(Some(crate::hash::sha256(&bytes)))
}

#[derive(Default)]
struct CyclesHairSourceClosure {
    nodes: BTreeSet<Id>,
    transforms: BTreeSet<Id>,
    data_blocks: BTreeSet<Id>,
    collections: BTreeSet<Id>,
    scene_membership: BTreeMap<Id, serde_json::Value>,
    node_groups: BTreeSet<Id>,
    instance_context_nodes: BTreeSet<Id>,
    actions: BTreeMap<Id, (bool, bool)>,
    images: BTreeSet<Id>,
    complete: bool,
}

impl CyclesHairSourceClosure {
    fn visit_node(&mut self, doc: &SceneDoc, id: &Id, with_transform: bool, depth: usize) {
        if depth > 64 {
            self.complete = false;
            return;
        }
        if with_transform {
            self.transforms.insert(id.clone());
        }
        if !self.nodes.insert(id.clone()) {
            return;
        }
        let Some(node) = doc.nodes.get(id) else {
            self.complete = false;
            return;
        };
        let include_instance_context = self.instance_context_nodes.contains(id);
        if let Some(action) = &node.action {
            self.visit_action(
                doc,
                action,
                !matches!(node.kind.as_str(), "light" | "camera"),
                include_instance_context,
                depth + 1,
            );
        }
        for track in &node.nla_tracks {
            for strip in &track.strips {
                self.visit_action(
                    doc,
                    &strip.action,
                    !matches!(node.kind.as_str(), "light" | "camera"),
                    include_instance_context,
                    depth + 1,
                );
            }
        }
        if let Some(data_id) = &node.data {
            self.visit_data_block(doc, data_id, depth + 1);
        }
        for modifier in &node.modifiers {
            if modifier.is_active(crate::eval::EvaluationPurpose::Render) {
                for value in modifier.params.values() {
                    self.visit_value(doc, value, depth + 1);
                }
            }
        }
        for key in ["instance_collection", "physics_particle_systems"] {
            if let Some(value) = node.properties.get(key) {
                self.visit_value(doc, value, depth + 1);
            }
        }
        for constraint in &node.constraints {
            if let Some(target) = &constraint.target {
                self.visit_node(doc, target, true, depth + 1);
            }
            for value in constraint.params.values() {
                self.visit_value(doc, value, depth + 1);
            }
        }
        for driver in &node.drivers {
            if let Ok(value) = serde_json::to_value(driver) {
                self.visit_value(doc, &value, depth + 1);
            } else {
                self.complete = false;
            }
        }
        if with_transform && let Some(parent) = &node.parent {
            self.visit_node(doc, parent, true, depth + 1);
        }
    }

    fn visit_data_block(&mut self, doc: &SceneDoc, id: &Id, depth: usize) {
        if depth > 64 {
            self.complete = false;
            return;
        }
        let Some(data) = doc.data_blocks.get(id) else {
            self.complete = false;
            return;
        };
        if !self.data_blocks.insert(id.clone()) {
            return;
        }
        self.visit_serializable(doc, &data.descriptor, depth + 1);
        self.visit_serializable(doc, &data.mesh, depth + 1);
        self.visit_serializable(doc, &data.grease_pencil, depth + 1);
        self.visit_serializable(doc, &data.armature, depth + 1);
        self.visit_serializable(doc, &data.shape_keys, depth + 1);
        self.visit_serializable(doc, &data.vertex_groups, depth + 1);
        self.visit_serializable(doc, &data.vertex_weights, depth + 1);
        self.visit_serializable(doc, &data.curve, depth + 1);
        self.visit_serializable(doc, &data.surface, depth + 1);
        self.visit_serializable(doc, &data.text, depth + 1);
        self.visit_serializable(doc, &data.hair_curves, depth + 1);
        self.visit_serializable(doc, &data.metaball, depth + 1);
        self.visit_serializable(doc, &data.lattice, depth + 1);
        self.visit_serializable(doc, &data.pointcloud, depth + 1);
        self.visit_serializable(doc, &data.volume, depth + 1);
        if !data.materials.is_empty() {
            self.visit_value(
                doc,
                &serde_json::json!({"material_slot_count": data.materials.len()}),
                depth + 1,
            );
        }
        let has_geometry = data.mesh.is_some()
            || data.grease_pencil.is_some()
            || data.armature.is_some()
            || data.shape_keys.is_some()
            || !data.vertex_groups.is_empty()
            || !data.vertex_weights.is_empty()
            || data.curve.is_some()
            || data.surface.is_some()
            || data.text.is_some()
            || data.hair_curves.is_some()
            || data.metaball.is_some()
            || data.lattice.is_some()
            || data.pointcloud.is_some()
            || data.volume.is_some();
        if !has_geometry && !matches!(data.data_type.as_str(), "camera" | "light") {
            self.complete = false;
        }
    }

    fn visit_serializable<T: serde::Serialize>(&mut self, doc: &SceneDoc, value: &T, depth: usize) {
        match serde_json::to_value(value) {
            Ok(value) => self.visit_value(doc, &value, depth + 1),
            Err(_) => self.complete = false,
        }
    }

    fn visit_scene_collection(
        &mut self,
        doc: &SceneDoc,
        id: &Id,
        excluded: &[Id],
        member_node_ids: &BTreeSet<Id>,
        instance_member_node_ids: &BTreeSet<Id>,
        depth: usize,
    ) {
        if depth > 64 {
            self.complete = false;
            return;
        }
        if excluded.contains(id) || self.scene_membership.contains_key(id) {
            return;
        }
        let Some(collection) = doc.collections.get(id) else {
            self.complete = false;
            return;
        };
        let mut candidates = Vec::new();
        for object_id in &collection.objects {
            if cycles_hair_source_candidate(doc, object_id) {
                candidates.push(object_id.clone());
                let include_transform = !member_node_ids.contains(object_id)
                    || instance_member_node_ids.contains(object_id);
                self.visit_node(doc, object_id, include_transform, depth + 1);
            }
        }
        self.scene_membership.insert(
            id.clone(),
            serde_json::json!({
                "children": collection.children,
                "candidate_objects": candidates,
            }),
        );
        for child in &collection.children {
            self.visit_scene_collection(
                doc,
                child,
                excluded,
                member_node_ids,
                instance_member_node_ids,
                depth + 1,
            );
        }
    }

    fn visit_collection(&mut self, doc: &SceneDoc, id: &Id, depth: usize) {
        if depth > 64 {
            self.complete = false;
            return;
        }
        if !self.collections.insert(id.clone()) {
            return;
        }
        let Some(collection) = doc.collections.get(id) else {
            self.complete = false;
            return;
        };
        for object_id in &collection.objects {
            self.visit_node(doc, object_id, true, depth + 1);
        }
        for child in &collection.children {
            self.visit_collection(doc, child, depth + 1);
        }
    }

    fn visit_value(&mut self, doc: &SceneDoc, value: &serde_json::Value, depth: usize) {
        if depth > 64 {
            self.complete = false;
            return;
        }
        match value {
            serde_json::Value::Array(values) => {
                for value in values {
                    self.visit_value(doc, value, depth + 1);
                }
            }
            serde_json::Value::Object(object) => {
                if let (Some(kind), Some(name)) = (
                    object.get("id_type").and_then(serde_json::Value::as_str),
                    object.get("name").and_then(serde_json::Value::as_str),
                ) {
                    match kind {
                        "Material" | "ShaderNodeTree" | "CompositorNodeTree" => return,
                        "Object" => {
                            if let Some(id) = cycles_hair_mapped_id(doc, "Object", name) {
                                self.visit_node(doc, &id, true, depth + 1);
                            } else {
                                self.complete = false;
                            }
                        }
                        "Collection" => {
                            let id = object
                                .get("bridge_id")
                                .and_then(serde_json::Value::as_str)
                                .and_then(|value| Id::new(value.to_owned()).ok())
                                .or_else(|| cycles_hair_mapped_id(doc, "Collection", name));
                            if let Some(id) = id {
                                self.visit_collection(doc, &id, depth + 1);
                            } else {
                                self.complete = false;
                            }
                        }
                        "GeometryNodeTree" | "NodeTree" => {
                            if let Some(id) = cycles_hair_mapped_id(doc, "NodeGroup", name) {
                                self.visit_node_group(doc, &id, depth + 1);
                            } else {
                                self.complete = false;
                            }
                        }
                        "Image" => {
                            if let Some(id) = cycles_hair_mapped_id(doc, "Image", name) {
                                self.images.insert(id);
                            } else {
                                self.complete = false;
                            }
                        }
                        _ => self.complete = false,
                    }
                    return;
                }
                for value in object.values() {
                    self.visit_value(doc, value, depth + 1);
                }
            }
            serde_json::Value::String(value) => {
                if let Ok(id) = Id::new(value.clone()) {
                    if doc.nodes.contains_key(&id) {
                        self.visit_node(doc, &id, true, depth + 1);
                    } else if doc.data_blocks.contains_key(&id) {
                        self.visit_data_block(doc, &id, depth + 1);
                    } else if doc.collections.contains_key(&id) {
                        self.visit_collection(doc, &id, depth + 1);
                    } else if doc.node_groups.contains_key(&id) {
                        self.visit_node_group(doc, &id, depth + 1);
                    } else if doc.images.contains_key(&id) {
                        self.images.insert(id);
                    }
                }
            }
            _ => {}
        }
    }

    fn visit_action(
        &mut self,
        doc: &SceneDoc,
        id: &Id,
        include_all: bool,
        include_instance_context: bool,
        depth: usize,
    ) {
        if depth > 64 || !doc.actions.contains_key(id) {
            self.complete = false;
            return;
        }
        self.actions
            .entry(id.clone())
            .and_modify(|scope| {
                scope.0 |= include_all;
                scope.1 |= include_instance_context;
            })
            .or_insert((include_all, include_instance_context));
    }

    fn visit_node_group(&mut self, doc: &SceneDoc, id: &Id, depth: usize) {
        if depth > 64 {
            self.complete = false;
            return;
        }
        let Some(group) = doc.node_groups.get(id) else {
            self.complete = false;
            return;
        };
        if group.kind != crate::graph::GraphKind::Geometry {
            self.complete = false;
            return;
        }
        if !self.node_groups.insert(id.clone()) {
            return;
        }
        for node in group.nodes.values() {
            for value in node.properties.values().chain(node.inputs.values()) {
                self.visit_value(doc, value, depth + 1);
            }
        }
        for socket in group
            .interface
            .inputs
            .iter()
            .chain(group.interface.outputs.iter())
        {
            self.visit_value(doc, &socket.default, depth + 1);
        }
    }
}

fn cycles_hair_node_group_geometry_state(group: &NodeGroup) -> Value {
    let nodes = group
        .nodes
        .iter()
        .map(|(id, node)| {
            (
                id.to_string(),
                json!({
                    "type": node.node_type,
                    "properties": node.properties,
                    "inputs": node.inputs,
                }),
            )
        })
        .collect::<Map<_, _>>();
    json!({
        "kind": group.kind,
        "interface": group.interface,
        "nodes": nodes,
        "links": group.links,
    })
}

fn cycles_hair_action_geometry_state(
    action: &Action,
    include_all: bool,
    include_instance_context: bool,
    node_ids: &BTreeSet<Id>,
) -> Value {
    let fcurves = action
        .fcurves
        .iter()
        .filter(|curve| {
            cycles_hair_action_fcurve(&curve.path, include_all, include_instance_context)
        })
        .collect::<Vec<_>>();
    let slots = action
        .slots
        .iter()
        .filter(|slot| node_ids.contains(&slot.node))
        .collect::<Vec<_>>();
    let active_slots = action
        .active_slots
        .iter()
        .filter(|(node_id, _)| node_ids.contains(*node_id))
        .map(|(node_id, slot)| ((*node_id).clone(), (*slot).clone()))
        .collect::<BTreeMap<_, _>>();
    let modifier_bindings = action
        .modifier_bindings
        .iter()
        .filter_map(|(index, bindings)| {
            let bindings = bindings
                .iter()
                .filter(|binding| node_ids.contains(&binding.node))
                .collect::<Vec<_>>();
            (!bindings.is_empty()).then_some((*index, bindings))
        })
        .collect::<BTreeMap<_, _>>();
    json!({
        "fcurves": fcurves,
        "slots": slots,
        "active_slots": active_slots,
        "modifier_bindings": modifier_bindings,
    })
}

fn cycles_hair_action_fcurve(
    path: &str,
    include_all: bool,
    include_instance_context: bool,
) -> bool {
    let path = path.to_ascii_lowercase();
    let object_context = [
        "color",
        "pass_index",
        "random_id",
        "material_slots[",
        "materials[",
    ]
    .iter()
    .any(|prefix| path.starts_with(prefix));
    if object_context {
        return include_instance_context;
    }
    include_all || cycles_hair_transform_fcurve(&path)
}

fn cycles_hair_transform_fcurve(path: &str) -> bool {
    let path = path.to_ascii_lowercase();
    [
        "location",
        "rotation",
        "scale",
        "delta_",
        "matrix_",
        "parent",
        "constraints[",
        "modifiers[",
        "data.",
        "pose.bones[",
        "hide_render",
        "visible_",
        "instance_collection",
        "particle_systems[",
    ]
    .iter()
    .any(|prefix| path.starts_with(prefix))
}

fn cycles_hair_source_candidate(doc: &SceneDoc, node_id: &Id) -> bool {
    let Some(node) = doc.nodes.get(node_id) else {
        return false;
    };
    if node.kind == "hair_curves"
        || node
            .data
            .as_ref()
            .and_then(|data_id| doc.data_blocks.get(data_id))
            .is_some_and(|data| data.hair_curves.is_some())
    {
        return true;
    }
    if node.modifiers.iter().any(|modifier| {
        modifier.is_active(crate::eval::EvaluationPurpose::Render)
            && matches!(
                modifier.modifier_type.as_str(),
                "nodes" | "particle_instance" | "particle_system"
            )
    }) {
        return true;
    }
    node.properties
        .get("instance_collection")
        .is_some_and(|value| !value.is_null())
        || node
            .properties
            .get("physics_particle_systems")
            .is_some_and(|value| match value {
                serde_json::Value::Array(values) => !values.is_empty(),
                serde_json::Value::Object(values) => !values.is_empty(),
                _ => !value.is_null(),
            })
}

fn cycles_hair_mapped_id(doc: &SceneDoc, kind: &str, name: &str) -> Option<Id> {
    doc.compatibility
        .get("blender_adapter")?
        .get("id_mappings")?
        .get(format!("{kind}:{name}"))?
        .as_str()
        .and_then(|value| Id::new(value.to_owned()).ok())
}

fn cycles_hair_node_geometry_state(
    node: &Node,
    include_transform: bool,
    include_instance_context: bool,
) -> Result<serde_json::Value> {
    let modifiers = node
        .modifiers
        .iter()
        .filter(|modifier| modifier.is_active(crate::eval::EvaluationPurpose::Render))
        .map(serde_json::to_value)
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(PotError::internal_json)?;
    let metadata = node.properties.get("blender_object_metadata");
    let rna_properties = metadata.and_then(|value| value.get("rna_properties"));
    Ok(serde_json::json!({
        "name": node.name,
        "kind": node.kind,
        "data": node.data,
        "modifiers": modifiers,
        "render_visible": node.render_visible,
        "ray_visibility": node.ray_visibility,
        "material_links": node.material_links,
        "material_slot_count": node.materials.len(),
        "material_slots": include_instance_context.then_some(&node.materials),
        "instance_collection": node.properties.get("instance_collection"),
        "physics_particle_systems": node.properties.get("physics_particle_systems"),
        "action": node.action,
        "nla_tracks": node.nla_tracks,
        "action_blend_type": node.action_blend_type,
        "action_influence": node.action_influence,
        "action_extrapolation": node.action_extrapolation,
        "transform": include_transform.then_some(&node.transform),
        "captured_matrix_basis": include_transform.then_some(&node.blender_matrix_basis),
        "parent": include_transform.then_some(&node.parent),
        "parent_inverse": include_transform.then_some(&node.parent_inverse),
        "parent_type": include_transform.then_some(&node.parent_type),
        "parent_bone": include_transform.then_some(&node.parent_bone),
        "constraints": include_transform.then_some(&node.constraints),
        "drivers": node.drivers,
        "object_info": include_instance_context.then(|| serde_json::json!({
            "color": rna_properties.and_then(|properties| properties.get("color")),
            "pass_index": rna_properties.and_then(|properties| properties.get("pass_index")),
        })),
        "object_transform_metadata": include_transform.then(|| serde_json::json!({
            "matrix_basis": metadata.and_then(|value| value.get("matrix_basis")),
            "matrix_local": metadata.and_then(|value| value.get("matrix_local")),
            "matrix_world": metadata.and_then(|value| value.get("matrix_world")),
            "delta_location": metadata.and_then(|value| value.get("delta_location")),
            "delta_rotation_euler": metadata.and_then(|value| value.get("delta_rotation_euler")),
            "delta_rotation_quaternion": metadata.and_then(|value| value.get("delta_rotation_quaternion")),
            "delta_scale": metadata.and_then(|value| value.get("delta_scale")),
        })),
    }))
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LibraryKind {
    #[default]
    PotterProject,
    Blend,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LibraryStatus {
    #[default]
    Ok,
    Missing,
    Changed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Library {
    pub name: String,
    pub kind: LibraryKind,
    pub uri: String,
    pub resolved_path: String,
    pub resource: Option<Id>,
    pub hash: String,
    pub status: LibraryStatus,
    pub linked_ids: BTreeMap<String, Vec<Id>>,
    pub overrides: Vec<LibraryOverride>,
    /// Source IDs keyed by `registry:local_id`, retained for Potter reloads.
    pub items: BTreeMap<String, String>,
    /// Original project path retained when reading pre-typed library records.
    pub source_project: Option<String>,
}

impl Default for Library {
    fn default() -> Self {
        Self {
            name: String::new(),
            kind: LibraryKind::PotterProject,
            uri: String::new(),
            resolved_path: String::new(),
            resource: None,
            hash: String::new(),
            status: LibraryStatus::Ok,
            linked_ids: BTreeMap::new(),
            overrides: Vec::new(),
            items: BTreeMap::new(),
            source_project: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LibraryOverride {
    pub registry: String,
    pub id: Id,
    pub reference_id: Id,
    pub properties: Vec<LibraryOverrideProperty>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LibraryOverrideProperty {
    pub path: String,
    pub operation: String,
    pub value: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    #[serde(default = "default_blender")]
    pub blender: String,
}

impl Default for Profile {
    fn default() -> Self {
        Self {
            blender: default_blender(),
        }
    }
}

fn default_blender() -> String {
    "5.2.2".to_owned()
}

impl Default for SceneDoc {
    fn default() -> Self {
        Self::new(String::new())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scene {
    pub name: String,
    pub root_collection: Id,
    pub view_layers: Registry<ViewLayer>,
    /// Marker for visible, render-enabled Grid `LightProbes` in the source view layers. Their caches are not exposed through RNA.
    #[serde(default)]
    pub eevee_has_local_volume_probe_grids: Option<bool>,
    /// Exact Cycles Hair geometry-key groups captured from each Blender render view layer.
    /// `None` means this document has no captured Blender provenance; `Some([])` means the
    /// Cycles render callback captured no Hair geometry groups.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cycles_hair_shadow_groups: Option<Vec<CyclesHairShadowGroup>>,
    pub frame_current: f64,
    pub frame_start: i32,
    pub frame_end: i32,
    /// Blender `SCE_CUSTOM_SIMULATION_RANGE`: bakes without their own range use
    /// `simulation_frame_start..=simulation_frame_end` instead of the scene frame range.
    #[serde(default)]
    pub use_custom_simulation_range: bool,
    #[serde(default = "default_simulation_frame_start")]
    pub simulation_frame_start: i32,
    #[serde(default = "default_simulation_frame_end")]
    pub simulation_frame_end: i32,
    pub fps: u32,
    pub fps_base: f64,
    #[serde(default = "default_true")]
    pub use_gravity: bool,
    pub camera: Option<Id>,
    pub world: Option<Id>,
    /// Active Movie Clip used by constraints with `use_active_clip`.
    #[serde(default)]
    pub active_clip: Option<Id>,
    pub unit: UnitSettings,
    #[serde(default)]
    pub render: RenderSettings,
    #[serde(default)]
    pub use_compositing: bool,
    #[serde(default)]
    pub compositor: Option<Id>,
    #[serde(default)]
    pub color_management: ColorManagement,
    #[serde(default)]
    pub markers: Vec<TimelineMarker>,
    #[serde(default)]
    pub rigid_body_world: Option<RigidBodyWorld>,
    #[serde(default)]
    pub sequencer: Sequencer,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CyclesHairShadowGroup {
    pub view_layer_id: Id,
    pub geometry_key: CyclesHairShadowGeometryKey,
    pub representative_node_id: Id,
    pub member_node_ids: Vec<Id>,
    /// Each exact Cycles Scene Object, in source order; the first is the representative.
    pub members: Vec<CyclesHairShadowMember>,
    /// Evaluated Hair Curves index to source curve index; `None` when no exact mapping exists.
    pub source_curve_indices: Option<Vec<u32>>,
    /// Render-evaluated raw controls when the Cycles iterator's Curves ID differs from source data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluated_hair_curves: Option<HairCurvesData>,
    /// Fingerprint of the geometry-affecting source dependency closure; `None` means it was unmodeled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_input_fingerprint: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CyclesHairShadowMember {
    pub instance: CyclesHairShadowInstanceKey,
    /// Blender row-major evaluated `Object.matrix_world`.
    pub object_to_world: [[f64; 4]; 4],
    pub object_color: [f64; 4],
    pub object_pass_index: i32,
    pub object_random_id: i32,
    pub slot_0_material_id: Option<Id>,
    pub ray_visibility: RayVisibility,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CyclesHairShadowInstanceKey {
    pub real_object_node_id: Id,
    pub parent_node_id: Option<Id>,
    pub persistent_id: [i32; 8],
    pub is_instance: bool,
    pub use_particle_hair: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CyclesHairShadowGeometryKey {
    Data { data_id: Option<Id> },
    Object { node_id: Id },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MovieClip {
    pub name: String,
    /// Image-sequence resource ID or source path.
    pub source: Option<String>,
    /// Optional content hash of the source sequence.
    pub source_hash: Option<String>,
    pub frame_start: i32,
    /// Source image width in pixels; both dimensions remain zero when unavailable.
    #[serde(default)]
    pub width: u32,
    /// Source image height in pixels; both dimensions remain zero when unavailable.
    #[serde(default)]
    pub height: u32,
    pub fps: f64,
    pub tracking: MovieTracking,
}

impl Default for MovieClip {
    fn default() -> Self {
        Self {
            name: String::new(),
            source: None,
            source_hash: None,
            frame_start: 1,
            width: 0,
            height: 0,
            fps: 24.0,
            tracking: MovieTracking::default(),
        }
    }
}

impl MovieClip {
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty()
            || !self.fps.is_finite()
            || self.fps <= 0.0
            || (self.width == 0) != (self.height == 0)
        {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "movie clip name, frame rate, and optional dimensions must be valid",
            ));
        }
        self.tracking.validate()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MovieTracking {
    pub tracks: Vec<TrackingTrack>,
    pub plane_tracks: Vec<PlaneTrack>,
    pub camera: CameraIntrinsics,
    pub reconstruction: Reconstruction,
    #[serde(default)]
    pub objects: Vec<TrackingObject>,
}

impl MovieTracking {
    pub fn validate(&self) -> Result<()> {
        let mut track_ids = BTreeSet::new();
        for track in &self.tracks {
            if track.id.is_empty() || track.name.is_empty() || !track_ids.insert(&track.id) {
                return Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    "movie tracking track IDs and names must be non-empty and unique",
                ));
            }
            let mut previous_frame = None;
            for marker in &track.markers {
                if !marker.frame.is_finite()
                    || marker.co.iter().any(|value| !value.is_finite())
                    || marker
                        .pattern_corners
                        .iter()
                        .flatten()
                        .any(|value| !value.is_finite())
                    || marker
                        .search_area
                        .iter()
                        .flatten()
                        .any(|value| !value.is_finite())
                    || previous_frame.is_some_and(|previous| marker.frame <= previous)
                {
                    return Err(PotError::new(
                        ErrorCode::SceneInvalid,
                        "movie tracking markers must have finite, strictly ordered coordinates",
                    ));
                }
                previous_frame = Some(marker.frame);
            }
        }
        let mut plane_track_ids = BTreeSet::new();
        for plane_track in &self.plane_tracks {
            if plane_track.id.is_empty()
                || plane_track.name.is_empty()
                || !plane_track_ids.insert(&plane_track.id)
                || plane_track.track_ids.len() < 4
                || !plane_track.reference_frame.is_finite()
                || plane_track
                    .track_ids
                    .iter()
                    .any(|track_id| !track_ids.contains(track_id))
                || plane_track.track_ids.iter().any(|track_id| {
                    self.tracks
                        .iter()
                        .find(|track| &track.id == track_id)
                        .is_none_or(|track| {
                            !track.markers.iter().any(|marker| {
                                crate::float::equal_f64(marker.frame, plane_track.reference_frame)
                            })
                        })
                })
            {
                return Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    "plane tracks must reference existing tracks with enabled reference markers",
                ));
            }
            let mut previous_frame = None;
            for homography in &plane_track.homographies {
                if !homography.frame.is_finite()
                    || previous_frame.is_some_and(|previous| homography.frame <= previous)
                    || homography
                        .matrix
                        .iter()
                        .flatten()
                        .any(|value| !value.is_finite())
                {
                    return Err(PotError::new(
                        ErrorCode::SceneInvalid,
                        "plane track homographies must be finite and strictly ordered",
                    ));
                }
                previous_frame = Some(homography.frame);
            }
        }
        let mut object_ids = BTreeSet::new();
        for object in &self.objects {
            let object_tracks = object.tracks.iter().collect::<BTreeSet<_>>();
            let mut previous_frame = None;
            if object.id.is_empty()
                || object.name.is_empty()
                || !object.scale.is_finite()
                || object.scale <= 0.0
                || !object.reconstruction_average_error.is_finite()
                || !object_ids.insert(&object.id)
                || object_tracks.len() != object.tracks.len()
                || object.tracks.iter().any(|track| !track_ids.contains(track))
            {
                return Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    "tracking objects must have unique IDs, valid tracks, and a positive finite scale",
                ));
            }
            for pose in &object.reconstruction {
                if !pose.frame.is_finite()
                    || !pose.average_error.is_finite()
                    || previous_frame.is_some_and(|previous| pose.frame <= previous)
                    || pose.matrix.iter().any(|value| !value.is_finite())
                {
                    return Err(PotError::new(
                        ErrorCode::SceneInvalid,
                        "tracking object poses must be finite and strictly ordered by frame",
                    ));
                }
                previous_frame = Some(pose.frame);
            }
        }
        self.camera.validate()?;
        if !self.reconstruction.average_error.is_finite() {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "reconstruction average error must be finite",
            ));
        }
        for camera in &self.reconstruction.cameras {
            if !camera.average_error.is_finite()
                || camera
                    .matrix
                    .iter()
                    .flatten()
                    .any(|value| !value.is_finite())
            {
                return Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    "reconstructed camera matrices and errors must be finite",
                ));
            }
        }
        for point in &self.reconstruction.points {
            if !track_ids.contains(&point.track) || point.co.iter().any(|value| !value.is_finite())
            {
                return Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    "reconstructed points must reference existing tracks and finite coordinates",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrackingTrack {
    pub id: String,
    pub name: String,
    pub markers: Vec<TrackingMarker>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrackingMarker {
    pub frame: f64,
    pub co: [f64; 2],
    pub pattern_corners: [[f64; 2]; 4],
    pub search_area: [[f64; 2]; 4],
    pub disabled: bool,
}

impl Default for TrackingMarker {
    fn default() -> Self {
        Self {
            frame: 1.0,
            co: [0.0; 2],
            pattern_corners: [[0.0; 2]; 4],
            search_area: [[0.0; 2]; 4],
            disabled: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PlaneTrack {
    pub id: String,
    pub name: String,
    pub track_ids: Vec<String>,
    pub reference_frame: f64,
    pub homographies: Vec<PlaneHomography>,
}

impl Default for PlaneTrack {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: String::new(),
            track_ids: Vec::new(),
            reference_frame: 1.0,
            homographies: Vec::new(),
        }
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrackingObjectPose {
    pub frame: f64,
    /// Column-major camera-to-object transform for Blender Object Solver evaluation.
    pub matrix: [f64; 16],
    #[serde(default)]
    pub average_error: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrackingObject {
    pub id: String,
    pub name: String,
    pub tracks: Vec<String>,
    pub reconstruction: Vec<TrackingObjectPose>,
    pub reconstruction_is_valid: bool,
    pub reconstruction_average_error: f64,
    pub scale: f64,
}

impl Default for TrackingObject {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: String::new(),
            tracks: Vec::new(),
            reconstruction: Vec::new(),
            reconstruction_is_valid: false,
            reconstruction_average_error: 0.0,
            scale: 1.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlaneHomography {
    pub frame: f64,
    pub matrix: [[f64; 3]; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CameraIntrinsics {
    pub focal_mm: f64,
    pub sensor_width_mm: f64,
    pub principal: [f64; 2],
    pub units: String,
    pub pixel_aspect: f64,
    pub distortion_model: String,
    pub k1: f64,
    pub k2: f64,
    pub k3: f64,
    pub division_k1: f64,
    pub division_k2: f64,
    pub nuke_k1: f64,
    pub nuke_k2: f64,
    pub nuke_p1: f64,
    pub nuke_p2: f64,
    pub brown_k1: f64,
    pub brown_k2: f64,
    pub brown_k3: f64,
    pub brown_k4: f64,
    pub brown_p1: f64,
    pub brown_p2: f64,
}

impl Default for CameraIntrinsics {
    fn default() -> Self {
        Self {
            focal_mm: 50.0,
            sensor_width_mm: 36.0,
            principal: [0.5, 0.5],
            units: "MILLIMETERS".to_owned(),
            pixel_aspect: 1.0,
            distortion_model: "POLYNOMIAL".to_owned(),
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            division_k1: 0.0,
            division_k2: 0.0,
            nuke_k1: 0.0,
            nuke_k2: 0.0,
            nuke_p1: 0.0,
            nuke_p2: 0.0,
            brown_k1: 0.0,
            brown_k2: 0.0,
            brown_k3: 0.0,
            brown_k4: 0.0,
            brown_p1: 0.0,
            brown_p2: 0.0,
        }
    }
}

impl CameraIntrinsics {
    pub fn validate(&self) -> Result<()> {
        let coefficients = [
            self.k1,
            self.k2,
            self.k3,
            self.division_k1,
            self.division_k2,
            self.nuke_k1,
            self.nuke_k2,
            self.nuke_p1,
            self.nuke_p2,
            self.brown_k1,
            self.brown_k2,
            self.brown_k3,
            self.brown_k4,
            self.brown_p1,
            self.brown_p2,
        ];
        if !self.focal_mm.is_finite()
            || self.focal_mm <= 0.0
            || !self.sensor_width_mm.is_finite()
            || self.sensor_width_mm <= 0.0
            || !self.pixel_aspect.is_finite()
            || self.pixel_aspect <= 0.0
            || self.principal.iter().any(|value| !value.is_finite())
            || coefficients.iter().any(|value| !value.is_finite())
            || !matches!(self.units.as_str(), "PIXELS" | "MILLIMETERS")
            || !matches!(
                self.distortion_model.as_str(),
                "POLYNOMIAL" | "DIVISION" | "NUKE" | "BROWN"
            )
        {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "camera intrinsics and lens distortion must be finite and valid",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Reconstruction {
    pub cameras: Vec<SolvedCamera>,
    pub points: Vec<ReconstructedPoint>,
    pub is_valid: bool,
    pub average_error: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolvedCamera {
    pub frame: i32,
    pub matrix: [[f64; 4]; 3],
    #[serde(default)]
    pub average_error: f64,
    /// Whether `matrix` is Blender's reconstructed camera-to-world transform.
    #[serde(default)]
    pub matrix_is_camera_to_world: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReconstructedPoint {
    pub track: String,
    pub co: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TimelineMarker {
    pub id: Id,
    pub name: String,
    pub frame: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NlaTrack {
    pub id: Id,
    pub name: String,
    #[serde(default)]
    pub mute: bool,
    #[serde(default)]
    pub solo: bool,
    #[serde(default)]
    pub strips: Vec<NlaStrip>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NlaStrip {
    pub id: Id,
    pub action: Id,
    /// Action slot assigned to this strip; `None` only for single-slot or slotless Actions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action_slot: Option<Id>,
    pub frame_start: f64,
    pub frame_end: f64,
    pub action_frame_start: f64,
    pub action_frame_end: f64,
    pub scale: f64,
    pub repeat: f64,
    pub blend_type: NlaBlendType,
    pub influence: f64,
    /// Blender's `NlaStrip.use_animated_influence`.
    #[serde(default, skip_serializing_if = "is_false")]
    pub use_animated_influence: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub influence_curve: Option<FCurve>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub reverse: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub use_sync_length: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub use_auto_blend: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub mute: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub use_animated_time: bool,
    #[serde(default)]
    pub strip_time: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strip_time_curve: Option<FCurve>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub use_animated_time_cyclic: bool,
    pub extrapolation: NlaExtrapolation,
    pub blend_in: f64,
    pub blend_out: f64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NlaBlendType {
    #[default]
    Replace,
    Add,
    Subtract,
    Multiply,
    Combine,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NlaExtrapolation {
    #[default]
    Hold,
    HoldForward,
    Nothing,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionSlot {
    pub id: Id,
    pub node: Id,
    /// Blender `ActionSlot` identity, kept only to map linked Blender slots back to stable IDs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blender_identifier: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GreasePencilData {
    #[serde(default)]
    pub layers: Vec<GreasePencilLayer>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GreasePencilLayer {
    pub id: Id,
    pub name: String,
    pub opacity: f64,
    #[serde(default = "default_true")]
    pub visible: bool,
    #[serde(default)]
    pub frames: Vec<GreasePencilFrame>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GreasePencilFrame {
    pub frame: f64,
    #[serde(default)]
    pub strokes: Vec<GreasePencilStroke>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GreasePencilStroke {
    pub id: Id,
    #[serde(default)]
    pub points: Vec<GreasePencilPoint>,
    #[serde(default)]
    pub material: Option<Id>,
    #[serde(default)]
    pub cyclic: bool,
    #[serde(default)]
    pub fill: Option<[f64; 4]>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GreasePencilPoint {
    pub position: [f64; 3],
    pub pressure: f64,
    pub radius: f64,
    pub opacity: f64,
    pub time: f64,
}

/// Cycles' film pixel filter (`intern/cycles/scene/film.h`, `FilterType`). Blender's default is
/// Blackman-Harris, serialized as Blender's enum identifiers (`BOX`, `GAUSSIAN`, `BLACKMAN_HARRIS`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PixelFilterType {
    #[serde(rename = "BOX")]
    Box,
    #[serde(rename = "GAUSSIAN")]
    Gaussian,
    #[serde(rename = "BLACKMAN_HARRIS")]
    BlackmanHarris,
}

/// Blender's `filter_width` default (`properties.py`): 1.5 pixels.
pub const DEFAULT_FILTER_WIDTH: f64 = 1.5;
/// Blender's `filter_width` RNA range. Blender stores the value as float32, so the lower bound is
/// `0.01_f32` widened: a `.blend` holding the clamped minimum reads back as this value exactly.
pub const FILTER_WIDTH_MIN: f64 = 0.01_f32 as f64;
pub const FILTER_WIDTH_MAX: f64 = 10.0;

fn default_pixel_filter_type() -> PixelFilterType {
    PixelFilterType::BlackmanHarris
}

/// serde's `skip_serializing_if` passes a reference. The predicates are generic so that the
/// by-reference signature needs no lint suppression for a small `Copy` type.
fn is_default_pixel_filter_type<T: PartialEq<PixelFilterType>>(value: &T) -> bool {
    PartialEq::eq(value, &PixelFilterType::BlackmanHarris)
}

fn default_filter_width() -> f64 {
    DEFAULT_FILTER_WIDTH
}

fn is_default_filter_width<T: Copy + Into<f64>>(value: &T) -> bool {
    equal_f64((*value).into(), DEFAULT_FILTER_WIDTH)
}

/// Blender's `SceneEEVEE.gi_cubemap_resolution` enum values.
pub const EEVEE_GI_CUBEMAP_RESOLUTIONS: [u32; 6] = [128, 256, 512, 1024, 2048, 4096];
pub const EEVEE_RAYTRACING_RESOLUTION_SCALES: [u32; 5] = [1, 2, 4, 8, 16];

fn default_eevee_volumetric_tile_size() -> u32 {
    8
}

fn is_default_eevee_volumetric_tile_size<T: Copy + Into<f64>>(value: &T) -> bool {
    equal_f64((*value).into(), 8.0)
}

fn default_eevee_volumetric_samples() -> u32 {
    64
}

fn is_default_eevee_volumetric_samples<T: Copy + Into<f64>>(value: &T) -> bool {
    equal_f64((*value).into(), 64.0)
}

fn default_eevee_volumetric_sample_distribution() -> f64 {
    0.8
}

// Blender stores these RNA factors and distances as floats.
fn is_default_eevee_volumetric_sample_distribution<T: Copy + Into<f64>>(value: &T) -> bool {
    equal_f64(f64::from((*value).into() as f32), f64::from(0.8_f32))
}

fn default_eevee_volumetric_start() -> f64 {
    0.1
}

fn is_default_eevee_volumetric_start<T: Copy + Into<f64>>(value: &T) -> bool {
    equal_f64(f64::from((*value).into() as f32), f64::from(0.1_f32))
}

fn default_eevee_volumetric_end() -> f64 {
    100.0
}

fn is_default_eevee_volumetric_end<T: Copy + Into<f64>>(value: &T) -> bool {
    equal_f64(f64::from((*value).into() as f32), f64::from(100.0_f32))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CyclesCurveShape {
    Ribbons,
    Thick,
    ThickLinear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EeveeRayTracingMethod {
    Probe,
    Screen,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EeveeFastGiMethod {
    AmbientOcclusionOnly,
    GlobalIllumination,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RenderSettings {
    pub resolution_x: u32,
    pub resolution_y: u32,
    pub resolution_percentage: u32,
    pub samples: u32,
    pub seed: u32,
    pub max_bounces: u32,
    /// Cycles' `volume_bounces`: a volume path takes `volume_bounces + 1` scatter vertices with direct light, and each
    /// later segment only attenuates (`max_volume_bounce`, `intern/cycles/scene/integrator.cpp`). Omitted when equal to
    /// Blender's default of 0.
    #[serde(default, skip_serializing_if = "is_default_volume_bounces")]
    pub volume_bounces: u32,
    /// Cycles' `transparent_max_bounces`. Omitted when equal to Blender's default of 8.
    #[serde(
        default = "default_transparent_max_bounces",
        skip_serializing_if = "is_default_transparent_max_bounces"
    )]
    pub transparent_max_bounces: u32,
    /// Blender's Filter Glossy (`scene.cycles.blur_glossy`): the widening of the ray differentials after a bounce,
    /// which bump nodes read. Omitted when equal to Blender's default of 1.0.
    #[serde(
        default = "default_filter_glossy",
        skip_serializing_if = "is_default_filter_glossy"
    )]
    pub filter_glossy: f64,
    /// Cycles' `pixel_filter_type` (`scene/film.h`). Omitted when equal to Blender's default.
    #[serde(
        default = "default_pixel_filter_type",
        skip_serializing_if = "is_default_pixel_filter_type"
    )]
    pub pixel_filter_type: PixelFilterType,
    /// Cycles' `filter_width` in pixels, Blender's RNA range 0.01..=10. Omitted at Blender's default.
    #[serde(
        default = "default_filter_width",
        skip_serializing_if = "is_default_filter_width"
    )]
    pub filter_width: f64,
    pub film_transparent: bool,
    /// Blender's `render.filter_size` (`scene.r.gauss`, `rna_scene.cc`): the reconstruction radius, in pixels, of
    /// EEVEE's film (`eevee_film.cc`). Realtime Beauty uses it; Solid and the path tracer do not.
    #[serde(default = "default_filter_size")]
    pub filter_size: f64,
    pub engine: String,
    /// Blender's `RenderSettings.use_high_quality_normals`; unknown source values stay `None`.
    #[serde(default)]
    pub use_high_quality_normals: Option<bool>,
    pub use_sequencer: bool,
    pub audio_codec: String,
    #[serde(default)]
    pub passes: Vec<String>,
    /// Blender's view-layer cryptomatte accuracy setting (`ViewLayer.use_pass_cryptomatte_accurate`,
    /// `rna_scene.cc`). It changes how the cryptomatte pass is generated and is not a pass itself.
    #[serde(default, skip_serializing_if = "is_false")]
    pub cryptomatte_accurate: bool,
    /// Blender's `eevee.light_threshold` (`DNA_scene_types.h`): the power at which an EEVEE local light's influence ends,
    /// where `light_influence` reads `eevee_light.cc` `attenuation_radius_get`. Omitted at Blender's default of 0.01.
    #[serde(
        default = "default_light_threshold",
        skip_serializing_if = "is_default_light_threshold"
    )]
    pub eevee_light_threshold: f64,
    /// Blender's `SceneEEVEE.gi_cubemap_resolution`; EEVEE sphere probes capture at half this extent.
    pub eevee_gi_cubemap_resolution: Option<u32>,
    /// Blender's `SceneEEVEE.use_raytracing`; `None` is unknown or retains the legacy native renderer path.
    #[serde(default)]
    pub eevee_use_raytracing: Option<bool>,
    #[serde(default)]
    pub eevee_use_fast_gi: Option<bool>,
    #[serde(default)]
    pub eevee_raytracing_method: Option<EeveeRayTracingMethod>,
    #[serde(default)]
    pub eevee_raytracing_resolution_scale: Option<u32>,
    #[serde(default)]
    pub eevee_raytracing_screen_trace_quality: Option<f64>,
    #[serde(default)]
    pub eevee_raytracing_screen_trace_thickness: Option<f64>,
    #[serde(default)]
    pub eevee_raytracing_trace_max_roughness: Option<f64>,
    #[serde(default)]
    pub eevee_raytracing_use_backface_hit: Option<bool>,
    #[serde(default)]
    pub eevee_raytracing_backface_radiance_scale: Option<f64>,
    #[serde(default)]
    pub eevee_raytracing_use_denoise: Option<bool>,
    #[serde(default)]
    pub eevee_raytracing_denoise_spatial: Option<bool>,
    #[serde(default)]
    pub eevee_raytracing_denoise_temporal: Option<bool>,
    #[serde(default)]
    pub eevee_raytracing_denoise_bilateral: Option<bool>,
    #[serde(default)]
    pub eevee_fast_gi_method: Option<EeveeFastGiMethod>,
    #[serde(default)]
    pub eevee_fast_gi_bias: Option<f64>,
    #[serde(default)]
    pub eevee_fast_gi_resolution: Option<u32>,
    #[serde(default)]
    pub eevee_fast_gi_step_count: Option<u32>,
    #[serde(default)]
    pub eevee_fast_gi_ray_count: Option<u32>,
    #[serde(default)]
    pub eevee_fast_gi_quality: Option<f64>,
    #[serde(default)]
    pub eevee_fast_gi_distance: Option<f64>,
    #[serde(default)]
    pub eevee_fast_gi_thickness_near: Option<f64>,
    /// Blender's `SceneEEVEE.clamp_surface_indirect`; zero disables indirect-light clamping in EEVEE.
    #[serde(default)]
    pub eevee_clamp_surface_indirect: Option<f64>,

    /// EEVEE's volume resolve settings (`scene.eevee`), omitted at Blender defaults.
    #[serde(
        default = "default_eevee_volumetric_tile_size",
        skip_serializing_if = "is_default_eevee_volumetric_tile_size"
    )]
    pub eevee_volumetric_tile_size: u32,
    #[serde(
        default = "default_eevee_volumetric_samples",
        skip_serializing_if = "is_default_eevee_volumetric_samples"
    )]
    pub eevee_volumetric_samples: u32,
    #[serde(
        default = "default_eevee_volumetric_sample_distribution",
        skip_serializing_if = "is_default_eevee_volumetric_sample_distribution"
    )]
    pub eevee_volumetric_sample_distribution: f64,
    #[serde(default, skip_serializing_if = "is_false")]
    pub eevee_use_volume_custom_range: bool,
    #[serde(
        default = "default_eevee_volumetric_start",
        skip_serializing_if = "is_default_eevee_volumetric_start"
    )]
    pub eevee_volumetric_start: f64,
    #[serde(
        default = "default_eevee_volumetric_end",
        skip_serializing_if = "is_default_eevee_volumetric_end"
    )]
    pub eevee_volumetric_end: f64,
    /// Blender's `Scene.eevee.use_volumetric_shadows`; disabled by default.
    #[serde(default, skip_serializing_if = "is_false")]
    pub eevee_use_volumetric_shadows: bool,
    #[serde(default)]
    pub motion_blur: bool,
    #[serde(default = "default_shutter")]
    pub shutter: f64,
    #[serde(default = "default_motion_blur_samples")]
    pub motion_blur_samples: u32,
    pub dicing_rate: f64,
    pub preview_dicing_rate: f64,
    pub max_subdivisions: u32,
    pub offscreen_dicing_scale: f64,
    #[serde(default)]
    pub dicing_camera: Option<Id>,
    pub cycles_curves_shape: Option<CyclesCurveShape>,
    pub cycles_curves_subdivisions: Option<u32>,
}

impl RenderSettings {
    fn validate(&self) -> Result<()> {
        if !self.dicing_rate.is_finite()
            || !(0.1..=1000.0).contains(&self.dicing_rate)
            || !self.preview_dicing_rate.is_finite()
            || !(0.1..=1000.0).contains(&self.preview_dicing_rate)
            || self.max_subdivisions > 16
            || !self.offscreen_dicing_scale.is_finite()
            || !(1.0..=f32::MAX as f64).contains(&self.offscreen_dicing_scale)
        {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "Cycles adaptive dicing settings are outside Blender RNA ranges",
            ));
        }
        if !(FILTER_WIDTH_MIN..=FILTER_WIDTH_MAX).contains(&self.filter_width) {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "Cycles filter_width is outside Blender RNA range 0.01..=10",
            ));
        }
        if self
            .cycles_curves_subdivisions
            .is_some_and(|subdivisions| subdivisions > 24)
        {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "Cycles curves subdivisions must be in Blender RNA range 0..=24",
            ));
        }
        if self
            .eevee_gi_cubemap_resolution
            .is_some_and(|resolution| !EEVEE_GI_CUBEMAP_RESOLUTIONS.contains(&resolution))
        {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "EEVEE gi_cubemap_resolution must be one of 128, 256, 512, 1024, 2048, or 4096",
            ));
        }

        let raytracing = [
            (self.eevee_raytracing_screen_trace_quality, 0.0, 1.0),
            (
                self.eevee_raytracing_screen_trace_thickness,
                1e-6,
                f32::MAX as f64,
            ),
            (self.eevee_raytracing_trace_max_roughness, 0.0, 1.0),
            (self.eevee_raytracing_backface_radiance_scale, 0.0, 1.0),
            (self.eevee_fast_gi_bias, 0.0, 1.0),
            (self.eevee_fast_gi_quality, 0.0, 1.0),
            (self.eevee_fast_gi_distance, 0.0, 100_000.0),
            (self.eevee_fast_gi_thickness_near, 0.0, 100_000.0),
            (self.eevee_clamp_surface_indirect, 0.0, f64::from(f32::MAX)),
        ];
        if raytracing.iter().any(|(value, min, max)| {
            value.is_some_and(|value| !value.is_finite() || value < *min || value > *max)
        }) {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "EEVEE ray-tracing and Fast GI settings are outside Blender RNA ranges",
            ));
        }
        if self
            .eevee_raytracing_resolution_scale
            .is_some_and(|value| !EEVEE_RAYTRACING_RESOLUTION_SCALES.contains(&value))
            || self
                .eevee_fast_gi_resolution
                .is_some_and(|value| !EEVEE_RAYTRACING_RESOLUTION_SCALES.contains(&value))
            || self
                .eevee_fast_gi_ray_count
                .is_some_and(|value| !(1..=16).contains(&value))
            || self
                .eevee_fast_gi_step_count
                .is_some_and(|value| !(1..=64).contains(&value))
        {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "EEVEE ray-tracing resolution and Fast GI integer settings are outside Blender RNA ranges",
            ));
        }
        if !matches!(self.eevee_volumetric_tile_size, 1 | 2 | 4 | 8 | 16)
            || !(1..=256).contains(&self.eevee_volumetric_samples)
            || !self.eevee_volumetric_sample_distribution.is_finite()
            || !(0.0..=1.0).contains(&self.eevee_volumetric_sample_distribution)
            || !self.eevee_volumetric_start.is_finite()
            || self.eevee_volumetric_start <= 0.0
            || !self.eevee_volumetric_end.is_finite()
            || self.eevee_volumetric_end <= 0.0
        {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "EEVEE volumetric settings are outside Blender RNA ranges",
            ));
        }
        Ok(())
    }
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            resolution_x: 1920,
            resolution_y: 1080,
            resolution_percentage: 100,
            samples: 64,
            seed: 0,
            max_bounces: 4,
            volume_bounces: 0,
            transparent_max_bounces: 8,
            filter_glossy: 1.0,
            pixel_filter_type: PixelFilterType::BlackmanHarris,
            filter_width: DEFAULT_FILTER_WIDTH,
            film_transparent: false,
            filter_size: default_filter_size(),
            engine: "path".to_owned(),
            use_high_quality_normals: Some(false),
            use_sequencer: false,
            audio_codec: "wav".to_owned(),
            passes: vec!["combined".to_owned()],
            cryptomatte_accurate: false,
            eevee_light_threshold: default_light_threshold(),
            eevee_gi_cubemap_resolution: Some(512),
            eevee_use_raytracing: None,
            eevee_use_fast_gi: None,
            eevee_raytracing_method: None,
            eevee_raytracing_resolution_scale: None,
            eevee_raytracing_screen_trace_quality: None,
            eevee_raytracing_screen_trace_thickness: None,
            eevee_raytracing_trace_max_roughness: None,
            eevee_raytracing_use_backface_hit: None,
            eevee_raytracing_backface_radiance_scale: None,
            eevee_raytracing_use_denoise: None,
            eevee_raytracing_denoise_spatial: None,
            eevee_raytracing_denoise_temporal: None,
            eevee_raytracing_denoise_bilateral: None,
            eevee_fast_gi_bias: None,
            eevee_fast_gi_resolution: None,
            eevee_fast_gi_step_count: None,
            eevee_fast_gi_ray_count: None,
            eevee_fast_gi_quality: None,
            eevee_fast_gi_distance: None,
            eevee_fast_gi_thickness_near: None,
            eevee_fast_gi_method: None,
            eevee_clamp_surface_indirect: Some(10.0),

            eevee_volumetric_tile_size: default_eevee_volumetric_tile_size(),
            eevee_volumetric_samples: default_eevee_volumetric_samples(),
            eevee_volumetric_sample_distribution: default_eevee_volumetric_sample_distribution(),
            eevee_use_volume_custom_range: false,
            eevee_volumetric_start: default_eevee_volumetric_start(),
            eevee_volumetric_end: default_eevee_volumetric_end(),
            eevee_use_volumetric_shadows: false,
            motion_blur: false,
            shutter: default_shutter(),
            motion_blur_samples: default_motion_blur_samples(),
            dicing_rate: 1.0,
            preview_dicing_rate: 8.0,
            max_subdivisions: 12,
            offscreen_dicing_scale: 4.0,
            dicing_camera: None,
            cycles_curves_shape: Some(CyclesCurveShape::Thick),
            cycles_curves_subdivisions: None,
        }
    }
}

fn default_shutter() -> f64 {
    0.5
}

fn default_filter_size() -> f64 {
    1.5
}

fn default_motion_blur_samples() -> u32 {
    8
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ViewLayer {
    pub name: String,
    #[serde(default)]
    pub excluded_collections: Vec<Id>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnitSettings {
    pub system: String,
    pub scale_length: f64,
}

impl Default for UnitSettings {
    fn default() -> Self {
        Self {
            system: "metric".to_owned(),
            scale_length: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Collection {
    pub name: String,
    #[serde(default)]
    pub children: Vec<Id>,
    #[serde(default)]
    pub objects: Vec<Id>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CameraProjection {
    #[default]
    Perspective,
    Orthographic,
    Panorama,
}

/// Blender's `M_PI_F` (`rna_camera.cc`): the float32 π that Blender's panorama limits and fisheye field of view are built
/// from. Doubling and halving are exact in float32, so the derived constants below are the values Blender holds.
pub const BLENDER_PI: f64 = std::f32::consts::PI as f64;
/// A full turn, Blender's float32 longitude span. Blender's full-sphere default is ±π, so its span is this turn, a hair
/// beyond the f64 `TAU`; the bounds Potter checks use it so Blender's own defaults validate.
pub const BLENDER_TURN: f64 = 2.0 * BLENDER_PI;
/// A right angle, Blender's float32 latitude limit.
pub const BLENDER_RIGHT_ANGLE: f64 = 0.5 * BLENDER_PI;
/// Blender's fisheye field of view hard range (`rna_camera.cc`): 10 degrees to 10π. Blender clamps a value outside it, so
/// Potter refuses one rather than store a field of view its Blender camera would not keep. The soft limit is 2π.
pub const FISHEYE_FOV_MIN: f64 = 0.1745;
pub const FISHEYE_FOV_MAX: f64 = 31.415_925_979_614_258;
/// Blender's `fisheye_lens` hard range (`rna_camera.cc`): 0.01 to 100 mm. Blender clamps a value outside it.
pub const FISHEYE_LENS_MIN: f64 = 0.01;
pub const FISHEYE_LENS_MAX: f64 = 100.0;
/// Blender's default `fisheye_lens`, in millimetres.
const FISHEYE_LENS_DEFAULT: f64 = 10.5;

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde skip_serializing_if passes a reference"
)]
fn is_default_longitude_min(value: &f64) -> bool {
    equal_f64(*value, -BLENDER_PI)
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde skip_serializing_if passes a reference"
)]
fn is_default_longitude_max(value: &f64) -> bool {
    equal_f64(*value, BLENDER_PI)
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde skip_serializing_if passes a reference"
)]
fn is_default_latitude_min(value: &f64) -> bool {
    equal_f64(*value, -BLENDER_RIGHT_ANGLE)
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde skip_serializing_if passes a reference"
)]
fn is_default_latitude_max(value: &f64) -> bool {
    equal_f64(*value, BLENDER_RIGHT_ANGLE)
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde skip_serializing_if passes a reference"
)]
fn is_default_fisheye_fov(value: &f64) -> bool {
    equal_f64(*value, BLENDER_PI)
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde skip_serializing_if passes a reference"
)]
fn is_default_fisheye_lens(value: &f64) -> bool {
    equal_f64(*value, FISHEYE_LENS_DEFAULT)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CameraData {
    pub projection: CameraProjection,
    /// Blender's `lens` in millimetres, the perspective focal length. A perspective camera uses it, and so does a panorama
    /// camera in the engines that draw it as perspective (EEVEE and Workbench); the path tracer ignores it for panoramas.
    pub lens_mm: f64,
    pub sensor_height_mm: f64,
    pub sensor_fit: String,
    pub sensor_width_mm: f64,
    pub ortho_scale: f64,
    pub clip_start: f64,
    pub clip_end: f64,
    pub shift: [f64; 2],
    pub panorama_type: String,
    /// Blender's equirectangular longitude and latitude limits, in radians; the full sphere by default.
    #[serde(skip_serializing_if = "is_default_longitude_min")]
    pub longitude_min: f64,
    #[serde(skip_serializing_if = "is_default_longitude_max")]
    pub longitude_max: f64,
    #[serde(skip_serializing_if = "is_default_latitude_min")]
    pub latitude_min: f64,
    #[serde(skip_serializing_if = "is_default_latitude_max")]
    pub latitude_max: f64,
    /// The fisheye field of view in radians, for the fisheye panoramas; Blender's default is 180 degrees.
    #[serde(skip_serializing_if = "is_default_fisheye_fov")]
    pub fisheye_fov: f64,
    /// Blender's `fisheye_lens` in millimetres: the focal length of an equisolid fisheye, separate from `lens_mm`.
    #[serde(skip_serializing_if = "is_default_fisheye_lens")]
    pub fisheye_lens: f64,
    pub dof_enabled: bool,
    pub focus_distance: f64,
    pub f_stop: f64,
    pub aperture_blades: u32,
    pub stereo_mode: String,
    pub interocular_distance: f64,
}

impl Default for CameraData {
    fn default() -> Self {
        Self {
            projection: CameraProjection::Perspective,
            lens_mm: 50.0,
            sensor_height_mm: 24.0,
            sensor_fit: "AUTO".to_owned(),
            sensor_width_mm: 36.0,
            ortho_scale: 6.0,
            clip_start: 0.1,
            clip_end: 1000.0,
            shift: [0.0, 0.0],
            panorama_type: "equirectangular".to_owned(),
            longitude_min: -BLENDER_PI,
            longitude_max: BLENDER_PI,
            latitude_min: -BLENDER_RIGHT_ANGLE,
            latitude_max: BLENDER_RIGHT_ANGLE,
            fisheye_fov: BLENDER_PI,
            fisheye_lens: FISHEYE_LENS_DEFAULT,
            dof_enabled: false,
            focus_distance: 10.0,
            f_stop: 2.8,
            aperture_blades: 0,
            stereo_mode: "none".to_owned(),
            interocular_distance: 0.065,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LightType {
    #[default]
    Point,
    Sun,
    Spot,
    Area,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LightData {
    pub light_type: LightType,
    /// Whether this light casts shadows. Defaults to `Some(true)` to preserve native behavior.
    pub use_shadow: Option<bool>,
    #[serde(default = "white_rgb")]
    pub color: [f64; 3],
    pub energy: f64,
    pub radius: f64,
    /// Blender's `use_soft_falloff` (Cycles `is_sphere = !soft_falloff`). When `true`, a point or
    /// spot light with a radius is a disk oriented to the receiver; when `false`, a sphere.
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub soft_falloff: bool,
    pub spot_size: f64,
    pub spot_blend: f64,
    pub area_shape: String,
    pub area_size: f64,
    pub area_size_y: f64,
    /// Blender's area light `spread` (Cycles `AreaLight::spread`), in radians within [0, π]: the
    /// emission is confined to the cone of half-angle `spread / 2` about the emitting normal. The
    /// default π is the whole hemisphere and is not stored.
    #[serde(default = "default_spread", skip_serializing_if = "is_default_spread")]
    pub spread: f64,
    /// Blender's `cycles.use_multiple_importance_sampling` (Cycles `SHADER_USE_MIS`). When `true`,
    /// a BSDF-sampled ray can hit an area light, or a point or spot light with a radius, and that
    /// hit is weighted against light sampling. When `false`, the light is only light-sampled.
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub use_multiple_importance_sampling: bool,
    /// Blender's sun `angle` (`light.sun_angle`, `rna_light.cc`): the sun's angular diameter, in radians. EEVEE draws the
    /// sun's disc from it (`eevee_light.cc`, `shape_radius = tan(angle / 2)`). Omitted at Blender's default.
    #[serde(
        default = "default_sun_angle",
        skip_serializing_if = "is_default_sun_angle"
    )]
    pub sun_angle: f64,
    /// Blender's `use_custom_distance` (`LA_CUSTOM_ATTENUATION`): the light's influence ends at `cutoff_distance` instead of
    /// where its power falls to the scene's light threshold (`eevee_light.cc`, `attenuation_radius_get`).
    #[serde(default, skip_serializing_if = "is_false")]
    pub use_custom_distance: bool,
    /// Blender's `cutoff_distance` (`att_dist`), the influence radius of a light with `use_custom_distance`. Omitted at
    /// Blender's default of 40 m.
    #[serde(
        default = "default_cutoff_distance",
        skip_serializing_if = "is_default_cutoff_distance"
    )]
    pub cutoff_distance: f64,
    /// Blender's `diffuse_factor` (`Light.diff_fac`, `rna_light.cc`): the multiplier of the light's diffuse closures
    /// (diffuse, sheen and subsurface). EEVEE applies it (`eevee_light.cc`, `power[LIGHT_DIFFUSE]`); Cycles does not.
    #[serde(
        default = "default_light_factor",
        skip_serializing_if = "is_default_light_factor"
    )]
    pub diffuse_factor: f64,
    /// Blender's `specular_factor` (`Light.spec_fac`): the multiplier of the specular and coat closures, and of the glass
    /// reflection. EEVEE applies it (`power[LIGHT_SPECULAR]`); Cycles does not.
    #[serde(
        default = "default_light_factor",
        skip_serializing_if = "is_default_light_factor"
    )]
    pub specular_factor: f64,
    /// Blender's `transmission_factor` (`Light.transmission_fac`): the multiplier of the transmission closures. EEVEE
    /// applies it (`power[LIGHT_TRANSMISSION]`); Cycles does not.
    #[serde(
        default = "default_light_factor",
        skip_serializing_if = "is_default_light_factor"
    )]
    pub transmission_factor: f64,
    /// Blender's `volume_factor` (`Light.volume_fac`): the multiplier of the light's volume scattering (`power[LIGHT_VOLUME]`).
    /// EEVEE applies it; Cycles does not.
    #[serde(
        default = "default_light_factor",
        skip_serializing_if = "is_default_light_factor"
    )]
    pub volume_factor: f64,
}

impl Default for LightData {
    fn default() -> Self {
        Self {
            light_type: LightType::Point,
            use_shadow: Some(true),
            color: white_rgb(),
            energy: 1000.0,
            radius: 0.1,
            soft_falloff: true,
            spot_size: std::f64::consts::FRAC_PI_4,
            spot_blend: 0.15,
            area_shape: "SQUARE".to_owned(),
            area_size: 0.25,
            area_size_y: 0.25,
            spread: std::f64::consts::PI,
            use_multiple_importance_sampling: true,
            sun_angle: default_sun_angle(),
            use_custom_distance: false,
            cutoff_distance: default_cutoff_distance(),
            diffuse_factor: default_light_factor(),
            specular_factor: default_light_factor(),
            transmission_factor: default_light_factor(),
            volume_factor: default_light_factor(),
        }
    }
}

fn white_rgb() -> [f64; 3] {
    [1.0, 1.0, 1.0]
}

/// Blender's sun angle default, `DNA_light_types.h`: `DEG2RADF(0.526f)`.
fn default_sun_angle() -> f64 {
    0.526_f64.to_radians()
}

fn is_default_sun_angle<T: Into<f64> + Copy>(value: &T) -> bool {
    ((*value).into() - default_sun_angle()).abs() < 1.0e-6
}

/// Blender's light closure multipliers default to one (`DNA_light_types.h`: `diff_fac`, `spec_fac`, `transmission_fac`,
/// `volume_fac`).
fn default_light_factor() -> f64 {
    1.0
}

fn is_default_light_factor<T: Into<f64> + Copy>(value: &T) -> bool {
    ((*value).into() - 1.0).abs() < 1.0e-6
}

/// Blender's light cutoff distance default, `DNA_light_types.h`: 40 m.
fn default_cutoff_distance() -> f64 {
    40.0
}

fn is_default_cutoff_distance<T: Into<f64> + Copy>(value: &T) -> bool {
    ((*value).into() - default_cutoff_distance()).abs() < 1.0e-9
}

/// Blender's `eevee.light_threshold` default, `DNA_scene_types.h`: 0.01.
fn default_light_threshold() -> f64 {
    0.01
}

fn is_default_light_threshold<T: Into<f64> + Copy>(value: &T) -> bool {
    ((*value).into() - default_light_threshold()).abs() < 1.0e-6
}
fn default_volume_color() -> [f64; 3] {
    [0.5; 3]
}

/// Blender's `cycles.volume_sampling` for a material: distance, equiangular, or multiple importance sampling of
/// both (Blender's default). A volume's direct light is sampled along its ray in the chosen way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VolumeSampling {
    Distance,
    Equiangular,
    #[default]
    MultipleImportance,
}

/// Skips a volume setting in serde output when it is at its default, which is then read back as the default.
fn is_serialised_default<T: Default + PartialEq>(value: &T) -> bool {
    *value == T::default()
}

/// Blender's `cycles.volume_interpolation` for a material: trilinear (Blender's default) or the cubic B-spline of
/// Cycles' `kernel_image_interp_tricubic_nanovdb`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VolumeInterpolation {
    #[default]
    Linear,
    Cubic,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct World {
    /// Blender's non-node viewport/display color.
    pub color: [f64; 3],
    /// Background shader color, when a world node tree defines one.
    pub background_color: Option<[f64; 3]>,
    pub strength: f64,
    pub node_tree: Option<Id>,
    /// Blender's `World.sun_threshold`, which clamps world radiance during EEVEE sphere-probe capture.
    pub eevee_sun_threshold: Option<f64>,
    /// Blender's `World.sun_angle`, the extracted sun's angular diameter.
    pub eevee_sun_angle: Option<f64>,
    /// Blender's `World.use_sun_shadow`, controlling shadows from the extracted sun.
    pub eevee_use_sun_shadow: Option<bool>,
    /// Blender's `World.sun_shadow_maximum_resolution`, the extracted sun's shadow-map pixel limit.
    pub eevee_sun_shadow_maximum_resolution: Option<f64>,
    /// Blender's `World.sun_shadow_jitter_overblur`, applied to jittered extracted-sun shadows.
    pub eevee_sun_shadow_jitter_overblur: Option<f64>,
    /// Blender's `World.sun_shadow_filter_radius`, the extracted sun's PCF filter radius.
    pub eevee_sun_shadow_filter_radius: Option<f64>,
    /// Blender's `World.use_sun_shadow_jitter`, enabling jittered extracted-sun shadows.
    pub eevee_use_sun_shadow_jitter: Option<bool>,
}

impl Default for World {
    fn default() -> Self {
        Self {
            color: white_rgb(),
            background_color: None,
            strength: 1.0,
            node_tree: None,
            eevee_sun_threshold: Some(10.0),
            eevee_sun_angle: Some(f64::from(0.526_f32.to_radians())),
            eevee_use_sun_shadow: Some(true),
            eevee_sun_shadow_maximum_resolution: Some(f64::from(0.001_f32)),
            eevee_sun_shadow_jitter_overblur: Some(10.0),
            eevee_sun_shadow_filter_radius: Some(1.0),
            eevee_use_sun_shadow_jitter: Some(false),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RigidBodyType {
    #[default]
    Active,
    Passive,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RigidBodyShape {
    #[default]
    Box,
    Sphere,
    ConvexHull,
    Mesh,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RigidBodyWorld {
    pub enabled: bool,
    pub gravity: [f64; 3],
    pub substeps: u32,
    pub solver_iterations: u32,
    pub frame_start: i32,
    pub frame_end: i32,
    pub seed: u32,
}

impl Default for RigidBodyWorld {
    fn default() -> Self {
        Self {
            enabled: true,
            gravity: [0.0, 0.0, -9.81],
            substeps: 4,
            solver_iterations: 10,
            frame_start: 1,
            frame_end: 250,
            seed: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RigidBody {
    #[serde(rename = "type")]
    pub body_type: RigidBodyType,
    pub mass: f64,
    pub friction: f64,
    pub restitution: f64,
    pub shape: RigidBodyShape,
    pub linear_damping: f64,
    pub angular_damping: f64,
    pub initial_velocity: [f64; 3],
}

impl Default for RigidBody {
    fn default() -> Self {
        Self {
            body_type: RigidBodyType::Active,
            mass: 1.0,
            friction: 0.5,
            restitution: 0.0,
            shape: RigidBodyShape::Box,
            linear_damping: 0.04,
            angular_damping: 0.1,
            initial_velocity: [0.0, 0.0, 0.0],
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForceFieldType {
    Wind,
    Vortex,
    #[default]
    Force,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ForceField {
    #[serde(rename = "type")]
    pub field_type: ForceFieldType,
    pub strength: f64,
    pub falloff: f64,
}

impl Default for ForceField {
    fn default() -> Self {
        Self {
            field_type: ForceFieldType::Force,
            strength: 1.0,
            falloff: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OceanBakeCache {
    pub frame_start: i32,
    pub frame_end: i32,
    /// Square EXR width in pixels (the modifier's render resolution squared).
    pub resolution: u32,
    pub manifest: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Modifier {
    pub id: Id,
    #[serde(rename = "type")]
    pub modifier_type: String,
    pub name: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Blender `show_viewport`: active in viewport previews (`EvaluationPurpose::Preview`).
    /// Serialized only when false; a missing key reads as true.
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub show_viewport: bool,
    /// Blender `show_render`: active in saved renders and the exports whose Blender writer evaluates the render stack (`EvaluationPurpose::Render`).
    /// Serialized only when false; a missing key reads as true.
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub show_render: bool,
    #[serde(default)]
    pub params: Map<String, Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baked_ocean_cache: Option<OceanBakeCache>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[doc(hidden)]
    pub binding_data: Option<Value>,
    #[serde(skip)]
    #[doc(hidden)]
    pub runtime: ModifierRuntime,
}

impl Modifier {
    /// Whether the modifier takes part in an evaluation for `purpose`. `enabled` is the global
    /// switch; the purpose's `show_*` flag selects the viewport or render stack.
    #[must_use]
    pub fn is_active(&self, purpose: crate::eval::EvaluationPurpose) -> bool {
        self.enabled
            && match purpose {
                crate::eval::EvaluationPurpose::Preview => self.show_viewport,
                crate::eval::EvaluationPurpose::Render => self.show_render,
            }
    }

    /// Copy whose `enabled` means "active for `purpose`", so evaluators that read only `enabled`
    /// see that purpose's stack.
    #[must_use]
    pub fn for_purpose(&self, purpose: crate::eval::EvaluationPurpose) -> Self {
        let mut modifier = self.clone();
        modifier.enabled = self.is_active(purpose);
        modifier
    }
}

/// Blender's ordered f32 texture-coordinate stages, retained without matrix composition.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextureCoordinateMapping {
    pub subject_to_world_f32: [f32; 16],
    pub world_to_mapping_f32: Option<[f32; 16]>,
}

#[doc(hidden)]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModifierRuntime {
    pub target_mesh: Option<Mesh>,
    #[doc(hidden)]
    pub target_to_subject: Option<[f64; 16]>,
    /// Shrinkwrap `local2target`: the subject's object-local space in the target's, in f32.
    #[doc(hidden)]
    pub shrinkwrap_subject_to_target: Option<[f64; 16]>,
    /// Shrinkwrap `local2target` for the auxiliary target, in f32.
    #[doc(hidden)]
    pub shrinkwrap_auxiliary_subject_to_target: Option<[f64; 16]>,
    pub shrinkwrap_target_mesh: Option<Mesh>,
    pub shrinkwrap_auxiliary_target_mesh: Option<Mesh>,
    /// Boolean operands in the subject's space (subject excluded), in Blender order.
    pub boolean_operands: Vec<crate::geom::boolean::BooleanOperand>,
    /// `inv(subject_world) * reference_world` (column-major) for Hook, Cast, Simple Deform origin,
    /// Curve, and Wave's start object. A bone reference is `armature_world * pose_mat`.
    pub object_matrix: Option<[f64; 16]>,
    /// `inv(subject_world) * source_world` for generic Warp source object or pose bone.
    pub warp_from_matrix: Option<[f64; 16]>,
    /// `inv(subject_world) * destination_world` for generic Warp destination object or pose bone.
    pub warp_to_matrix: Option<[f64; 16]>,
    /// Complete Blender f32 UV Warp matrix in column-major order.
    #[doc(hidden)]
    pub uv_warp_matrix: Option<[[f32; 4]; 4]>,
    /// Raw subject-to-world and optional world-to-mapping matrices for ordered f32 texture mapping.
    #[doc(hidden)]
    pub texture_mapping: Option<TextureCoordinateMapping>,
    /// Bake-node records for this modifier's node group, keyed by Potter node id.
    pub node_bakes: BTreeMap<String, crate::graph::NodeBakeRecord>,
    /// Bake nodes whose frame the evaluation captures (explicit bake action).
    pub capture_node_bakes: BTreeSet<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Action {
    pub name: String,
    #[serde(default)]
    pub fcurves: Vec<FCurve>,
    #[serde(default)]
    pub slots: Vec<ActionSlot>,
    /// Action slot assigned to each Object that uses this Action as its active Action.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub active_slots: BTreeMap<Id, Id>,
    /// Scalar modifier-parameter bindings per flattened F-Curve index.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub modifier_bindings: BTreeMap<u32, Vec<ModifierParameterBinding>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModifierParameterBinding {
    pub node: Id,
    pub modifier: Id,
    pub parameter: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action_slot: Option<Id>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FCurve {
    pub path: String,
    pub index: u32,
    #[serde(default)]
    pub keyframes: Vec<Keyframe>,
    #[serde(default)]
    pub extrapolation: Extrapolation,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto_smoothing: Option<FCurveAutoSmoothing>,
    #[serde(default)]
    pub modifiers: Vec<FCurveModifier>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FCurveModifier {
    #[serde(rename = "type")]
    pub modifier_type: String,
    #[serde(default)]
    pub properties: Map<String, Value>,
}

impl FCurveModifier {
    #[must_use]
    pub(crate) fn is_supported(&self) -> bool {
        Self::is_supported_type(&self.modifier_type)
    }

    #[must_use]
    pub(crate) fn is_supported_type(modifier_type: &str) -> bool {
        matches!(
            modifier_type,
            "GENERATOR" | "FNGENERATOR" | "ENVELOPE" | "CYCLES" | "NOISE" | "LIMITS" | "STEPPED"
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FCurveAutoSmoothing {
    #[default]
    None,
    ContAccel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FCurveHandleType {
    #[default]
    Auto,
    AutoClamped,
    Vector,
    Aligned,
    Free,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Extrapolation {
    #[default]
    Constant,
    Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Interpolation {
    Constant,
    #[default]
    Linear,
    Bezier,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Keyframe {
    pub frame: f64,
    pub value: f64,
    #[serde(default)]
    pub interpolation: Interpolation,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handle_left: Option<[f64; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handle_right: Option<[f64; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handle_left_type: Option<FCurveHandleType>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handle_right_type: Option<FCurveHandleType>,
}

/// Blender-authored local `matrix_basis` and the fingerprint of the static state it was read from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapturedMatrixBasis {
    /// Column-major local Blender `matrix_basis` values.
    pub matrix_basis: [f64; 16],
    /// SHA-256 fingerprint of the static authoring state used to capture the matrix.
    pub fingerprint: String,
}

/// Source of a node material slot, matching Blender's per-slot material link setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaterialLink {
    Data,
    Object,
}

/// Blender `Object.track_axis` (`trackflag`): the axis Particle Instance orients along. Blender
/// initializes new objects to `POS_Y`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrackAxis {
    #[serde(rename = "POS_X")]
    PosX,
    #[default]
    #[serde(rename = "POS_Y")]
    PosY,
    #[serde(rename = "POS_Z")]
    PosZ,
    #[serde(rename = "NEG_X")]
    NegX,
    #[serde(rename = "NEG_Y")]
    NegY,
    #[serde(rename = "NEG_Z")]
    NegZ,
}

/// Blender `Object.up_axis` (`upflag`). Blender initializes new objects to `Z`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum UpAxis {
    X,
    Y,
    #[default]
    Z,
}

fn is_default_axis<T: Default + PartialEq>(value: &T) -> bool {
    *value == T::default()
}

/// A Cycles ray class that Blender's object visibility gates (`PATH_RAY_*` in Cycles' kernel).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RayKind {
    Camera,
    Diffuse,
    Glossy,
    Transmission,
    VolumeScatter,
    Shadow,
}

/// Blender's per-object ray visibility (`visible_camera`, `visible_diffuse`, ...): one bit per `RayKind`. Cycles masks
/// each object with these bits and skips it for rays whose kind is switched off. Stored as the mask a triangle carries,
/// and serialised as the six named flags (`{"camera": true, ...}`), all required.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RayVisibility {
    mask: u8,
}

impl Default for RayVisibility {
    /// Every ray kind is on: Blender's default for an object that is not a light.
    fn default() -> Self {
        Self {
            mask: RayKind::ALL_MASK,
        }
    }
}

impl RayVisibility {
    /// Blender's flags for a new object of `kind`. Lights are hidden from camera rays
    /// (`BKE_object_init`, object.cc:2177); every other object starts fully visible.
    #[must_use]
    pub fn default_for_kind(kind: &str) -> Self {
        let mut visibility = Self::default();
        if kind == "light" {
            visibility.set(RayKind::Camera, false);
        }
        visibility
    }

    #[must_use]
    pub fn allows(self, ray: RayKind) -> bool {
        self.mask & ray.bit() != 0
    }

    /// Sets one flag.
    pub fn set(&mut self, ray: RayKind, value: bool) {
        if value {
            self.mask |= ray.bit();
        } else {
            self.mask &= !ray.bit();
        }
    }

    /// One bit per ray kind, the mask a triangle carries so a BVH query tests a single `u8`.
    #[must_use]
    pub const fn mask(self) -> u8 {
        self.mask
    }
}

impl serde::Serialize for RayVisibility {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(Some(RayKind::ALL.len()))?;
        for ray in RayKind::ALL {
            map.serialize_entry(ray.name(), &self.allows(ray))?;
        }
        map.end()
    }
}

impl<'de> serde::Deserialize<'de> for RayVisibility {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error as _;
        let flags = BTreeMap::<String, bool>::deserialize(deserializer)?;
        if let Some(unknown) = flags
            .keys()
            .find(|key| !RayKind::ALL.iter().any(|ray| ray.name() == key.as_str()))
        {
            return Err(D::Error::unknown_field(unknown, &RAY_NAMES));
        }
        let mut visibility = Self { mask: 0 };
        for ray in RayKind::ALL {
            let value = flags
                .get(ray.name())
                .copied()
                .ok_or_else(|| D::Error::missing_field(ray.name()))?;
            visibility.set(ray, value);
        }
        Ok(visibility)
    }
}

/// The six flag names, as the error message lists the fields a `RayVisibility` takes.
const RAY_NAMES: [&str; 6] = [
    "camera",
    "diffuse",
    "glossy",
    "transmission",
    "volume_scatter",
    "shadow",
];

impl RayKind {
    /// Every ray kind, in the order of their bits.
    pub const ALL: [Self; 6] = [
        Self::Camera,
        Self::Diffuse,
        Self::Glossy,
        Self::Transmission,
        Self::VolumeScatter,
        Self::Shadow,
    ];
    /// Every ray kind's bit: the mask of a surface that no ray kind hides (`RayVisibility::default`).
    pub const ALL_MASK: u8 = (1 << 6) - 1;

    /// The kind's bit in a ray mask (`PATH_RAY_*` in Cycles' kernel).
    #[must_use]
    pub const fn bit(self) -> u8 {
        1 << self as u8
    }

    /// The kind's name in Potter's JSON, as `ray_visibility.camera`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Camera => "camera",
            Self::Diffuse => "diffuse",
            Self::Glossy => "glossy",
            Self::Transmission => "transmission",
            Self::VolumeScatter => "volume_scatter",
            Self::Shadow => "shadow",
        }
    }

    /// Blender's RNA property for the kind's visibility flag, as `visible_camera`.
    #[must_use]
    pub const fn rna_property(self) -> &'static str {
        match self {
            Self::Camera => "visible_camera",
            Self::Diffuse => "visible_diffuse",
            Self::Glossy => "visible_glossy",
            Self::Transmission => "visible_transmission",
            Self::VolumeScatter => "visible_volume_scatter",
            Self::Shadow => "visible_shadow",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Node {
    pub name: String,
    pub kind: String,
    #[serde(default)]
    pub primitive: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub parent: Option<Id>,
    #[serde(default)]
    pub parent_inverse: Option<[f64; 16]>,
    #[serde(default)]
    pub transform: Transform,
    /// Blender-authored local basis; omitted for ordinary Potter-authored nodes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blender_matrix_basis: Option<CapturedMatrixBasis>,
    /// Blender's `matrix_parent_inverse` for a root that came from Blender. Blender ignores it without a
    /// parent, so evaluation and the captured basis never apply it; it is round-trip data that export
    /// writes back unchanged. Omitted when identity or when the node is parented.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blender_parent_inverse: Option<[f64; 16]>,
    #[serde(default)]
    pub data: Option<Id>,
    /// Blender's active vertex-group index; must name one of the data block's vertex groups.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_vertex_group_index: Option<usize>,
    /// Material slots in slot order; OBJECT-linked entries are overrides of the data material.
    /// `null` is an empty slot: a face index on it has no material, as Blender's empty material
    /// slot does. Old files holding only ids still load unchanged.
    #[serde(default)]
    pub materials: Vec<Option<Id>>,
    /// Per-slot source; missing entries use legacy `OBJECT` for existing node slots.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub material_links: Vec<MaterialLink>,
    #[serde(default)]
    pub modifiers: Vec<Modifier>,
    #[serde(default = "default_true")]
    pub visible: bool,
    #[serde(default = "default_true")]
    pub render_visible: bool,
    #[serde(default = "default_true")]
    pub selectable: bool,
    #[serde(default)]
    pub action: Option<Id>,
    #[serde(default)]
    pub nla_tracks: Vec<NlaTrack>,
    #[serde(default)]
    pub action_blend_type: NlaBlendType,
    #[serde(default = "default_action_influence")]
    pub action_influence: f64,
    #[serde(default)]
    pub action_extrapolation: NlaExtrapolation,
    #[serde(default)]
    pub properties: Map<String, Value>,
    #[serde(default)]
    pub rigid_body: Option<RigidBody>,
    #[serde(default)]
    pub force_field: Option<ForceField>,
    #[serde(default)]
    pub parent_type: ParentType,
    #[serde(default)]
    pub parent_bone: Option<Id>,
    /// Blender `Object.track_axis`; omitted from serialized documents at the default `POS_Y`.
    #[serde(default, skip_serializing_if = "is_default_axis")]
    pub track_axis: TrackAxis,
    /// Blender `Object.up_axis`; omitted from serialized documents at the default `Z`.
    #[serde(default, skip_serializing_if = "is_default_axis")]
    pub up_axis: UpAxis,
    #[serde(default)]
    pub pose: BTreeMap<Id, PoseBone>,
    #[serde(default)]
    pub constraints: Vec<Constraint>,
    #[serde(default)]
    pub drivers: Vec<Driver>,
    /// Blender's object ray visibility. `None` means the default for the node's kind; see
    /// [`Node::resolved_ray_visibility`]. Omitted from JSON when unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ray_visibility: Option<RayVisibility>,
}

impl Default for Node {
    fn default() -> Self {
        Self {
            name: "Node".to_owned(),
            kind: "empty".to_owned(),
            primitive: None,
            tags: Vec::new(),
            parent: None,
            parent_inverse: None,
            transform: Transform::default(),
            blender_matrix_basis: None,
            blender_parent_inverse: None,
            data: None,
            active_vertex_group_index: None,
            materials: Vec::new(),
            material_links: Vec::new(),
            modifiers: Vec::new(),
            visible: true,
            render_visible: true,
            selectable: true,
            action: None,
            nla_tracks: Vec::new(),
            action_blend_type: NlaBlendType::default(),
            action_influence: default_action_influence(),
            action_extrapolation: NlaExtrapolation::default(),
            properties: Map::new(),
            rigid_body: None,
            force_field: None,
            parent_type: ParentType::default(),
            parent_bone: None,
            track_axis: TrackAxis::default(),
            up_axis: UpAxis::default(),
            pose: BTreeMap::new(),
            constraints: Vec::new(),
            drivers: Vec::new(),
            ray_visibility: None,
        }
    }
}

/// Pinned Blender evaluated material count: max(Object.totcol, evaluated DATA length).
pub(crate) fn material_slot_count(object_count: usize, data_count: usize) -> usize {
    object_count.max(data_count)
}

/// Link state at one slot; legacy absent entries preserve old node-local object slots.
pub(crate) fn material_link_at(
    object_count: usize,
    links: &[MaterialLink],
    slot: usize,
) -> MaterialLink {
    links.get(slot).copied().unwrap_or(if slot < object_count {
        MaterialLink::Object
    } else {
        MaterialLink::Data
    })
}

/// Borrowed effective material lookup shared by model, evaluation, and material-aware callers.
pub(crate) fn resolve_material_slot<'a>(
    object_materials: &'a [Option<Id>],
    data_materials: &'a [Option<Id>],
    links: &[MaterialLink],
    index: usize,
) -> Option<&'a Id> {
    let count = material_slot_count(object_materials.len(), data_materials.len());
    let slot = index.min(count.checked_sub(1)?);
    match material_link_at(object_materials.len(), links, slot) {
        MaterialLink::Data => data_materials.get(slot).and_then(Option::as_ref),
        MaterialLink::Object => object_materials
            .get(slot)
            .and_then(Option::as_ref)
            .or_else(|| data_materials.get(slot).and_then(Option::as_ref)),
    }
}

impl Node {
    /// Link source for slot `index`, including the legacy default.
    #[must_use]
    pub fn material_link(&self, index: usize) -> MaterialLink {
        material_link_at(self.materials.len(), &self.material_links, index)
    }

    /// Evaluated material slot count before modifier-specific changes.
    #[must_use]
    pub fn material_slot_count(&self, data: Option<&DataBlock>) -> usize {
        material_slot_count(
            self.materials.len(),
            data.map_or(0, |data| data.materials.len()),
        )
    }

    /// Blender's ray visibility for this node: the stored flags, or the default for its kind.
    #[must_use]
    pub fn resolved_ray_visibility(&self) -> RayVisibility {
        self.ray_visibility
            .unwrap_or_else(|| RayVisibility::default_for_kind(&self.kind))
    }

    /// Effective material for one indexed slot; out-of-range access clamps like Blender.
    #[must_use]
    pub fn material_slot<'a>(
        &'a self,
        data: Option<&'a DataBlock>,
        index: usize,
    ) -> Option<&'a Id> {
        resolve_material_slot(
            &self.materials,
            data.map_or(&[][..], |data| data.materials.as_slice()),
            &self.material_links,
            index,
        )
    }

    /// Fingerprints the fixed authoring fields that determine a captured Blender local basis.
    pub fn matrix_basis_fingerprint(&self, effective_transform: &Transform) -> Result<String> {
        let digest = self.matrix_basis_fingerprint_digest(effective_transform)?;
        Ok(format_matrix_basis_fingerprint(&digest))
    }

    fn matrix_basis_fingerprint_digest(&self, effective_transform: &Transform) -> Result<[u8; 32]> {
        use sha2::Digest as _;

        let mut hasher = sha2::Sha256::new();
        hasher.update(b"potter-captured-matrix-basis-v2\0");
        update_matrix_basis_transform(&mut hasher, effective_transform)?;
        // Only whether a parent exists is hashed: a parent id changes on import remap or duplicate
        // without changing the local basis. A root's parent_inverse, parent_type and parent_bone do
        // not change the captured basis, so they are hashed only under a parent (bone parents
        // change the world matrix). A root with a non-identity parent_inverse never uses the capture
        // (see `captured_matrix_basis`).
        hasher.update([u8::from(self.parent.is_some())]);
        if self.parent.is_some() {
            match &self.parent_inverse {
                Some(matrix) => {
                    hasher.update([1]);
                    update_matrix_basis_values(&mut hasher, matrix);
                }
                None => hasher.update([0]),
            }
            hasher.update([match self.parent_type {
                ParentType::Object => 0,
                ParentType::Bone => 1,
            }]);
            update_matrix_basis_optional_id(&mut hasher, self.parent_bone.as_ref());
        }
        update_matrix_basis_delta_transforms(&mut hasher, &self.properties);
        // Dynamic state is part of the fingerprint, so removing it leaves a capture stale.
        hasher.update([
            u8::from(self.action.is_some()),
            u8::from(!self.nla_tracks.is_empty()),
            u8::from(!self.drivers.is_empty()),
            u8::from(self.rigid_body.is_some()),
        ]);
        let digest = hasher.finalize();
        let mut result = [0; 32];
        result.copy_from_slice(&digest);
        Ok(result)
    }

    /// Returns the captured local basis only while its authoring and dynamic state remain current.
    pub fn captured_matrix_basis(
        &self,
        effective_transform: &Transform,
    ) -> Result<Option<[f64; 16]>> {
        let Some(captured) = &self.blender_matrix_basis else {
            return Ok(None);
        };
        if !captured.matrix_basis.iter().all(|value| value.is_finite()) {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "captured Blender matrix_basis values must be finite",
            ));
        }
        if self.action.is_some()
            || !self.nla_tracks.is_empty()
            || !self.drivers.is_empty()
            || self.rigid_body.is_some()
        {
            return Ok(None);
        }
        // The transform path applies a root's parent_inverse (keep-world unparenting keeps shear,
        // spec.md). Blender's captured basis is local and ignores it for a root, so such a root
        // takes the transform path in both states and its world does not jump after an edit.
        if self.parent.is_none()
            && self
                .parent_inverse
                .is_some_and(|matrix| !is_identity_matrix(&matrix))
        {
            return Ok(None);
        }
        let digest = self.matrix_basis_fingerprint_digest(effective_transform)?;
        Ok(
            matrix_basis_fingerprint_matches(&captured.fingerprint, &digest)
                .then_some(captured.matrix_basis),
        )
    }
}

fn update_matrix_basis_transform(hasher: &mut sha2::Sha256, transform: &Transform) -> Result<()> {
    use sha2::Digest as _;

    for value in &transform.translation {
        hasher.update(value.to_bits().to_le_bytes());
    }
    let rotation = crate::model::normalize_rotation(transform.rotation)?;
    for value in rotation {
        hasher.update(value.to_bits().to_le_bytes());
    }
    for value in &transform.scale {
        hasher.update(value.to_bits().to_le_bytes());
    }
    update_matrix_basis_string(hasher, &transform.rotation_mode);
    Ok(())
}

/// Whether a column-major matrix is the identity, within 1e-12 per element.
pub(crate) fn is_identity_matrix(matrix: &[f64; 16]) -> bool {
    // The diagonal is indices 0, 5, 10 and 15 of a column-major matrix.
    matrix.iter().enumerate().all(|(index, &value)| {
        let expected = if index % 5 == 0 { 1.0 } else { 0.0 };
        (value - expected).abs() <= 1.0e-12
    })
}

fn update_matrix_basis_values(hasher: &mut sha2::Sha256, values: &[f64; 16]) {
    use sha2::Digest as _;

    for value in values {
        hasher.update(value.to_bits().to_le_bytes());
    }
}

fn update_matrix_basis_optional_id(hasher: &mut sha2::Sha256, id: Option<&Id>) {
    use sha2::Digest as _;

    if let Some(id) = id {
        hasher.update([1]);
        update_matrix_basis_string(hasher, id.as_str());
    } else {
        hasher.update([0]);
    }
}

fn update_matrix_basis_delta_transforms(
    hasher: &mut sha2::Sha256,
    properties: &Map<String, Value>,
) {
    use sha2::Digest as _;

    let metadata = properties
        .get("blender_object_metadata")
        .and_then(Value::as_object);
    for (name, length) in [
        ("delta_location", 3),
        ("delta_rotation_euler", 3),
        ("delta_rotation_quaternion", 4),
        ("delta_scale", 3),
    ] {
        match metadata.and_then(|metadata| metadata.get(name)) {
            None | Some(Value::Null) => hasher.update([0]),
            Some(Value::Array(values)) if values.len() == length => {
                hasher.update([1]);
                for value in values {
                    if let Some(value) = value.as_f64() {
                        hasher.update([1]);
                        hasher.update(value.to_bits().to_le_bytes());
                    } else {
                        hasher.update([0]);
                    }
                }
            }
            Some(Value::Array(values)) => {
                hasher.update([2]);
                hasher.update((values.len() as u64).to_le_bytes());
            }
            Some(_) => hasher.update([3]),
        }
    }
}

fn update_matrix_basis_string(hasher: &mut sha2::Sha256, value: &str) {
    use sha2::Digest as _;

    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn format_matrix_basis_fingerprint(digest: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(71);
    result.push_str("sha256:");
    for &byte in digest {
        result.push(char::from(HEX[usize::from(byte >> 4)]));
        result.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    result
}

fn matrix_basis_fingerprint_matches(fingerprint: &str, digest: &[u8; 32]) -> bool {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let Some(hex) = fingerprint.strip_prefix("sha256:") else {
        return false;
    };
    let hex = hex.as_bytes();
    hex.len() == 64
        && digest.iter().enumerate().all(|(index, byte)| {
            hex[index * 2] == HEX[usize::from(byte >> 4)]
                && hex[index * 2 + 1] == HEX[usize::from(byte & 0x0f)]
        })
}

fn default_simulation_frame_start() -> i32 {
    1
}

fn default_simulation_frame_end() -> i32 {
    250
}

fn default_true() -> bool {
    true
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde skip_serializing_if passes a reference"
)]
fn is_true(value: &bool) -> bool {
    *value
}

fn default_spread() -> f64 {
    std::f64::consts::PI
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde skip_serializing_if passes a reference"
)]
fn is_default_spread(value: &f64) -> bool {
    *value >= std::f64::consts::PI
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde skip_serializing_if passes a reference"
)]
fn is_default_transparent_max_bounces(value: &u32) -> bool {
    *value == 8
}

fn default_transparent_max_bounces() -> u32 {
    8
}

fn default_filter_glossy() -> f64 {
    1.0
}
fn default_action_influence() -> f64 {
    1.0
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde skip_serializing_if passes a reference"
)]
fn is_default_filter_glossy(value: &f64) -> bool {
    value.to_bits() == 1.0_f64.to_bits()
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde skip_serializing_if passes a reference"
)]
fn is_default_volume_bounces(value: &u32) -> bool {
    *value == 0
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde skip_serializing_if passes a reference"
)]
fn is_false(value: &bool) -> bool {
    !*value
}

fn default_text_resolution() -> u32 {
    12
}

fn default_text_fill_mode() -> CurveFillMode {
    CurveFillMode::Both
}

fn default_text_fill_solver() -> CurveFillSolver {
    CurveFillSolver::Cdt
}

fn default_text_fill_rule() -> CurveFillRule {
    CurveFillRule::Nonzero
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CurveDimensions {
    TwoD,
    #[default]
    ThreeD,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CurveSplineType {
    Bezier,
    #[default]
    Poly,
    Nurbs,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CurveHandleType {
    #[default]
    Auto,
    Vector,
    Aligned,
    Free,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CurveFillMode {
    #[default]
    None,
    Front,
    Back,
    Both,
}

// Blender 5.2.2 DNA_curve_types.h:282-286 defines Curve::fill_solver/fill_rule and defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CurveFillSolver {
    #[default]
    SweepLine,
    Cdt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CurveFillRule {
    #[default]
    EvenOdd,
    Nonzero,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CurveTaperRadiusMode {
    #[default]
    Override,
    Multiply,
    Add,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CurvePoint {
    pub co: [f64; 3],
    pub handle_left: [f64; 3],
    pub handle_right: [f64; 3],
    pub handle_type: CurveHandleType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub handle_left_type: Option<CurveHandleType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub handle_right_type: Option<CurveHandleType>,
    pub weight: f64,
    pub radius: f64,
    pub tilt: f64,
}

impl Default for CurvePoint {
    fn default() -> Self {
        Self {
            co: [0.0; 3],
            handle_left: [0.0; 3],
            handle_right: [0.0; 3],
            handle_type: CurveHandleType::Auto,
            handle_left_type: None,
            handle_right_type: None,
            weight: 1.0,
            radius: 1.0,
            tilt: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CurveSpline {
    #[serde(rename = "type")]
    pub spline_type: CurveSplineType,
    pub points: Vec<CurvePoint>,
    pub order: u32,
    pub cyclic: bool,
    pub resolution: u32,
    pub use_endpoint: bool,
    pub use_bezier: bool,
    pub custom_knots: bool,
}

impl Default for CurveSpline {
    fn default() -> Self {
        Self {
            spline_type: CurveSplineType::Poly,
            points: Vec::new(),
            order: 3,
            cyclic: false,
            resolution: 12,
            use_endpoint: false,
            use_bezier: false,
            custom_knots: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CurveData {
    pub resolution_u: u32,
    pub splines: Vec<CurveSpline>,
    pub dimensions: CurveDimensions,
    pub bevel_depth: f64,
    pub bevel_resolution: u32,
    pub extrude: f64,
    /// True when the Blender Curve has a `bevel_object` pointer; the target is not represented.
    #[serde(default, skip_serializing_if = "is_false")]
    pub has_bevel_object: bool,
    pub taper: Option<Id>,
    pub taper_radius_mode: CurveTaperRadiusMode,
    pub use_map_taper: bool,
    pub fill_mode: CurveFillMode,
    pub fill_solver: CurveFillSolver,
    pub fill_rule: CurveFillRule,
    pub twist_mode: String,
    pub use_path: bool,
    pub path_duration: u32,
    pub eval_time: f64,
    pub eval_time_fcurves: Vec<FCurve>,
}

impl Default for CurveData {
    fn default() -> Self {
        Self {
            resolution_u: 12,
            splines: Vec::new(),
            dimensions: CurveDimensions::ThreeD,
            bevel_depth: 0.0,
            bevel_resolution: 0,
            extrude: 0.0,
            has_bevel_object: false,
            taper: None,
            taper_radius_mode: CurveTaperRadiusMode::Override,
            use_map_taper: false,
            fill_mode: CurveFillMode::None,
            fill_solver: CurveFillSolver::SweepLine,
            fill_rule: CurveFillRule::EvenOdd,
            twist_mode: "MINIMUM".to_owned(),
            use_path: false,
            path_duration: 100,
            eval_time: 0.0,
            eval_time_fcurves: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SurfacePoint {
    pub co: [f64; 3],
    pub weight: f64,
}

impl Default for SurfacePoint {
    fn default() -> Self {
        Self {
            co: [0.0; 3],
            weight: 1.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SurfaceData {
    pub points: Vec<Vec<SurfacePoint>>,
    pub order_u: u32,
    pub order_v: u32,
    pub resolution: [u32; 2],
    pub cyclic_u: bool,
    pub cyclic_v: bool,
    pub use_endpoint_u: bool,
    pub use_endpoint_v: bool,
}

impl Default for SurfaceData {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            order_u: 3,
            order_v: 3,
            resolution: [12, 12],
            cyclic_u: false,
            cyclic_v: false,
            use_endpoint_u: true,
            use_endpoint_v: true,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TextCharacterFormat {
    #[serde(default, skip_serializing_if = "is_false")]
    pub use_bold: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub use_italic: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub use_underline: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub use_small_caps: bool,
    pub material_index: u32,
    pub kerning: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TextObjectData {
    pub body: String,
    /// Built-in sentinel, registered resource URI, or external font path; unresolved fonts are refused during evaluation.
    pub font: String,
    pub font_name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub font_bold: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub font_bold_italic: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub font_italic: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub font_bold_name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub font_bold_italic_name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub font_italic_name: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub character_formats: Vec<TextCharacterFormat>,
    #[serde(default = "default_text_resolution")]
    pub resolution_u: u32,
    pub size: f64,
    pub align_x: String,
    pub align_y: String,
    #[serde(default = "default_text_fill_mode")]
    pub fill_mode: CurveFillMode,
    #[serde(default = "default_text_fill_solver")]
    pub fill_solver: CurveFillSolver,
    #[serde(default = "default_text_fill_rule")]
    pub fill_rule: CurveFillRule,
    pub extrude: f64,
    pub bevel_depth: f64,
    pub character_spacing: f64,
    pub word_spacing: f64,
    pub line_spacing: f64,
    pub shear: f64,
    pub offset_x: f64,
    pub offset_y: f64,
    pub small_caps_scale: f64,
}

impl Default for TextObjectData {
    fn default() -> Self {
        Self {
            body: String::new(),
            font: "builtin".to_owned(),
            font_name: "Bfont".to_owned(),
            font_bold: String::new(),
            font_bold_italic: String::new(),
            font_italic: String::new(),
            font_bold_name: String::new(),
            font_bold_italic_name: String::new(),
            font_italic_name: String::new(),
            character_formats: Vec::new(),
            resolution_u: default_text_resolution(),
            size: 1.0,
            align_x: "left".to_owned(),
            align_y: "baseline".to_owned(),
            fill_mode: default_text_fill_mode(),
            fill_solver: default_text_fill_solver(),
            fill_rule: default_text_fill_rule(),
            extrude: 0.0,
            bevel_depth: 0.0,
            character_spacing: 1.0,
            word_spacing: 1.0,
            line_spacing: 1.0,
            shear: 0.0,
            offset_x: 0.0,
            offset_y: 0.0,
            small_caps_scale: 0.75,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HairCurvesData {
    pub curves: Vec<HairCurve>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub attributes: BTreeMap<String, HairCurveAttribute>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub surface: Option<Id>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub surface_uv_map: String,
    #[serde(default, skip_serializing_if = "is_false")]
    pub use_mirror_x: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub use_mirror_y: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub use_mirror_z: bool,
    #[serde(default = "default_hair_curves_selection_domain")]
    pub selection_domain: String,
}

impl Default for HairCurvesData {
    fn default() -> Self {
        Self {
            curves: Vec::new(),
            attributes: BTreeMap::new(),
            surface: None,
            surface_uv_map: String::new(),
            use_mirror_x: false,
            use_mirror_y: false,
            use_mirror_z: false,
            selection_domain: default_hair_curves_selection_domain(),
        }
    }
}

fn default_hair_curves_selection_domain() -> String {
    "POINT".to_owned()
}

/// A Blender Curves attribute stored in its native domain, data type, and array order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct HairCurveAttribute {
    pub domain: String,
    pub data_type: String,
    pub values: Vec<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HairCurve {
    pub points: Vec<[f64; 3]>,
    pub radius: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataBlock {
    #[serde(rename = "type")]
    pub data_type: String,
    #[serde(default)]
    pub descriptor: Option<PrimitiveDescriptor>,
    #[serde(default)]
    pub mesh: Option<Mesh>,
    /// Ordered raw DATA-linked material slots for this Data-Block.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub materials: Vec<Option<Id>>,
    #[serde(default)]
    pub camera: Option<CameraData>,
    #[serde(default)]
    pub light: Option<LightData>,
    #[serde(default)]
    pub grease_pencil: Option<GreasePencilData>,
    #[serde(default)]
    pub armature: Option<ArmatureData>,
    #[serde(default)]
    pub shape_keys: Option<ShapeKeyData>,
    #[serde(default)]
    pub vertex_groups: Vec<VertexGroup>,
    #[serde(default)]
    pub vertex_weights: BTreeMap<u32, BTreeMap<Id, f64>>,
    #[serde(default)]
    pub curve: Option<CurveData>,
    #[serde(default)]
    pub surface: Option<SurfaceData>,
    #[serde(default)]
    pub text: Option<TextObjectData>,
    #[serde(default)]
    pub hair_curves: Option<HairCurvesData>,
    #[serde(default)]
    pub metaball: Option<MetaballData>,
    #[serde(default)]
    pub lattice: Option<LatticeData>,
    #[serde(default)]
    pub pointcloud: Option<PointCloudData>,
    #[serde(default)]
    pub volume: Option<VolumeData>,
}

impl Default for DataBlock {
    fn default() -> Self {
        Self {
            data_type: "mesh".to_owned(),
            descriptor: None,
            mesh: Some(Mesh::default()),
            materials: Vec::new(),
            camera: None,
            light: None,
            grease_pencil: None,
            armature: None,
            shape_keys: None,
            vertex_groups: Vec::new(),
            vertex_weights: BTreeMap::new(),
            curve: None,
            surface: None,
            text: None,
            hair_curves: None,
            metaball: None,
            lattice: None,
            pointcloud: None,
            volume: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrimitiveDescriptor {
    pub primitive: String,
    #[serde(default)]
    pub params: Map<String, Value>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageSource {
    #[default]
    Generated,
    File,
    Packed,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageColorspace {
    #[default]
    Srgb,
    Linear,
    NonColor,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageAlphaMode {
    #[default]
    Straight,
    Premultiplied,
    None,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageTile {
    pub number: u32,
    pub width: u32,
    pub height: u32,
    /// `None` is the missing state: the tile stays listed, as Blender keeps it, but its file was not found, so it has
    /// no size or pixels and samples as Blender's missing colour.
    #[serde(default)]
    pub blob: Option<String>,
    /// Blender's tile label, carried through import and export. Omitted when empty.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub label: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Image {
    pub name: String,
    #[serde(default)]
    pub source: ImageSource,
    #[serde(default)]
    pub colorspace: ImageColorspace,
    pub width: u32,
    pub height: u32,
    #[serde(default)]
    pub tiles: Vec<ImageTile>,
    #[serde(default)]
    pub blob: Option<String>,
    #[serde(default)]
    pub source_path: Option<String>,
    #[serde(default)]
    pub source_hash: Option<String>,
    #[serde(default)]
    pub alpha_mode: ImageAlphaMode,
    /// Blender's TILED source: the image routes UDIM lookups even when only its base tile 1001 is stored, so a lookup
    /// into any other tile is missing, as Blender renders it. Omitted when false.
    #[serde(default, skip_serializing_if = "is_false")]
    pub tiled: bool,
}

impl Image {
    /// A UDIM image routes its lookups by tile: a TILED source, or any stored tile beyond the base.
    #[must_use]
    pub fn is_udim(&self) -> bool {
        self.tiled || !self.tiles.is_empty()
    }

    /// A UDIM file image whose base tile (1001) file was missing at import: no blob and no source, with its listed tiles
    /// kept. Blender keeps such an image and renders the base's lookups as its missing colour, so Potter keeps the record.
    /// A packed or generated image with no blob is still invalid.
    #[must_use]
    pub fn base_is_missing(&self) -> bool {
        self.source == ImageSource::File
            && self.blob.is_none()
            && self.source_path.is_none()
            && self.source_hash.is_none()
            && self.is_udim()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextureRef {
    pub image: Id,
    #[serde(default)]
    pub uv_layer: Option<String>,
    #[serde(default)]
    pub interpolation: ImageInterpolation,
    /// Blender's Image Texture `extension`: how a lookup past the edge repeats, extends, or mirrors. Omitted when
    /// REPEAT, which is the default for scene JSON written before this field existed.
    #[serde(default, skip_serializing_if = "ImageExtension::is_repeat")]
    pub extension: ImageExtension,
}

/// Blender's viewport display values for a `.blend` material: `Material.diffuse_color` (RGBA), `roughness`, and
/// `metallic`. Workbench's Solid shading reads these in place of the material's own values
/// (`workbench_engine.cc` `get_material`); Beauty never does.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ViewportDisplay {
    pub diffuse_color: [f64; 4],
    pub roughness: f64,
    pub metallic: f64,
}

/// Blender's `Material.thickness_mode` (`MA_THICKNESS_SPHERE`, `MA_THICKNESS_SLAB`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThicknessMode {
    #[default]
    Sphere,
    Slab,
}

impl ThicknessMode {
    /// The name the operation schema and the Blender bridge use.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sphere => "sphere",
            Self::Slab => "slab",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Material {
    pub name: String,
    /// Blender's Material Index render pass (`Material.pass_index`, when recorded, in 0..=32767).
    #[serde(default = "default_material_pass_index")]
    pub pass_index: Option<u32>,
    #[serde(default = "default_color")]
    pub base_color: [f64; 4],
    #[serde(default = "default_alpha_mode")]
    pub alpha_mode: String,
    #[serde(default = "default_alpha_threshold")]
    pub alpha_threshold: f64,
    #[serde(default)]
    pub metallic: f64,
    #[serde(default = "default_roughness")]
    pub roughness: f64,
    #[serde(default)]
    pub emission_color: [f64; 3],
    #[serde(default)]
    pub emission_strength: f64,
    #[serde(default)]
    pub transmission: f64,
    /// Blender's EEVEE refraction toggle; `None` means imported data did not record it.
    #[serde(default)]
    pub use_raytrace_refraction: Option<bool>,
    #[serde(default = "default_ior")]
    pub ior: f64,
    /// Blender's `Material.thickness_mode`: the refraction thickness model EEVEE uses when the Thickness socket is not
    /// linked (`eevee_thickness_lib.bsl.hh`). Sphere is Blender's default.
    #[serde(default)]
    pub thickness_mode: ThicknessMode,
    /// Blender's `use_backface_culling` is off for new materials (`MA_BL_CULL_BACKFACE` unset in `DNA_material_types.h`).
    #[serde(default = "default_true")]
    pub double_sided: bool,
    /// Blender's `use_transparent_shadow` (`MA_BL_TRANSPARENT_SHADOW`, on by default): a transparent surface casts a
    /// partly transparent shadow, as EEVEE's shadow pass does (`eevee_surf_shadow.bsl.hh`). Off, it casts an opaque one.
    #[serde(default = "default_true")]
    pub use_transparent_shadow: bool,
    /// Blender's `cycles.use_bump_map_correction` (on by default, `intern/cycles/blender/addon/properties.py`): the shader
    /// flag `SD_USE_BUMP_MAP_CORRECTION` (`scene/shader.cpp`) that enables the terminator corrections for bump-mapped
    /// normals: the grazing diffuse smoothing and the lifted specular reflection (`kernel/closure/bsdf_util.h`).
    #[serde(default = "default_true")]
    pub use_bump_map_correction: bool,
    #[serde(default)]
    pub node_tree: Option<Id>,
    #[serde(default)]
    pub base_color_texture: Option<TextureRef>,
    #[serde(default)]
    pub roughness_texture: Option<TextureRef>,
    #[serde(default)]
    pub metallic_texture: Option<TextureRef>,
    #[serde(default)]
    pub normal_texture: Option<TextureRef>,
    #[serde(default = "default_displacement_method")]
    pub displacement_method: String,
    #[serde(default = "default_displacement_scale")]
    pub displacement_scale: f64,
    #[serde(default = "default_displacement_midlevel")]
    pub displacement_midlevel: f64,
    #[serde(default)]
    pub volume_density: f64,
    #[serde(default = "default_volume_color")]
    pub volume_color: [f64; 3],
    #[serde(default)]
    pub volume_anisotropy: f64,
    /// Blender's `cycles.volume_sampling`: how a volume's direct light is sampled (Cycles' `volume_stack_sample_method`).
    #[serde(default, skip_serializing_if = "is_serialised_default")]
    pub volume_sampling: VolumeSampling,
    /// Blender's `cycles.volume_interpolation`: how a voxel volume's density is interpolated between its samples.
    #[serde(default, skip_serializing_if = "is_serialised_default")]
    pub volume_interpolation: VolumeInterpolation,
    /// Blender's viewport display values, which Solid shading reads. Set by `.blend` import; omitted when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub viewport_display: Option<ViewportDisplay>,
}
impl Default for Material {
    fn default() -> Self {
        Self {
            name: String::new(),
            pass_index: DEFAULT_MATERIAL_PASS_INDEX,
            base_color: default_color(),
            alpha_mode: default_alpha_mode(),
            alpha_threshold: default_alpha_threshold(),
            metallic: 0.0,
            roughness: default_roughness(),
            emission_color: [0.0; 3],
            emission_strength: 0.0,
            transmission: 0.0,
            use_raytrace_refraction: Some(false),
            ior: default_ior(),
            thickness_mode: ThicknessMode::Sphere,
            double_sided: default_true(),
            use_transparent_shadow: default_true(),
            use_bump_map_correction: default_true(),
            node_tree: None,
            base_color_texture: None,
            roughness_texture: None,
            metallic_texture: None,
            normal_texture: None,
            displacement_method: default_displacement_method(),
            displacement_scale: default_displacement_scale(),
            displacement_midlevel: default_displacement_midlevel(),
            volume_density: 0.0,
            volume_color: default_volume_color(),
            volume_anisotropy: 0.0,
            volume_sampling: VolumeSampling::default(),
            volume_interpolation: VolumeInterpolation::default(),
            viewport_display: None,
        }
    }
}

const DEFAULT_MATERIAL_PASS_INDEX: Option<u32> = Some(0);

fn default_material_pass_index() -> Option<u32> {
    DEFAULT_MATERIAL_PASS_INDEX
}

fn default_color() -> [f64; 4] {
    [0.6, 0.6, 0.6, 1.0]
}

fn default_alpha_mode() -> String {
    "opaque".to_owned()
}

fn default_alpha_threshold() -> f64 {
    0.5
}

fn default_roughness() -> f64 {
    0.8
}
fn default_ior() -> f64 {
    1.5
}

fn default_displacement_method() -> String {
    "bump".to_owned()
}

fn default_displacement_scale() -> f64 {
    1.0
}

fn default_displacement_midlevel() -> f64 {
    0.5
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct HistoryState {
    #[serde(default)]
    pub head: Option<String>,
    #[serde(default)]
    pub undo: Vec<String>,
    #[serde(default)]
    pub redo: Vec<String>,
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests")]
    use super::{
        Action, CameraData, Id, Interpolation, LightData, MAX_REVISION, Material, Modifier,
        SceneDoc, revision_in_range,
    };

    #[test]
    fn default_scene_has_contract_initial_state_and_empty_registries() {
        let doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        let render = &doc.scenes[&Id::from_static("scene_main")].render;
        assert_eq!(
            render.cycles_curves_shape,
            Some(super::CyclesCurveShape::Thick)
        );
        assert_eq!(render.cycles_curves_subdivisions, None);
        assert_eq!(doc.schema_version, 18);
        doc.validate().unwrap();
        let value = serde_json::to_value(doc).unwrap();
        assert_eq!(value["revision"], 0);
        assert_eq!(value["scenes"]["scene_main"]["frame_current"], 1.0);
        assert_eq!(value["scenes"]["scene_main"]["fps"], 24);
        assert_eq!(
            value["collections"]["collection_root"]["objects"],
            serde_json::json!([])
        );
    }
    #[test]
    fn eevee_sun_threshold_uses_blender_default_and_validates_source_range() {
        assert_eq!(super::World::default().eevee_sun_threshold, Some(10.0));
        let world_id = Id::new("test_world").unwrap();
        let valid = super::World {
            eevee_sun_threshold: Some(f64::from(0.125_f32)),
            ..super::World::default()
        };
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        doc.worlds.insert(world_id.clone(), valid.clone());
        doc.validate().unwrap();
        let serialized = serde_json::to_value(&valid).unwrap();
        assert_eq!(serialized["eevee_sun_threshold"], f64::from(0.125_f32));
        let round_trip: super::World = serde_json::from_value(serialized).unwrap();
        assert_eq!(round_trip, valid);

        for threshold in [None, Some(0.0), Some(f64::from(f32::MAX))] {
            let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
            doc.worlds.insert(
                world_id.clone(),
                super::World {
                    eevee_sun_threshold: threshold,
                    ..super::World::default()
                },
            );
            doc.validate().unwrap();
        }
        for threshold in [
            Some(-f64::EPSILON),
            Some(f64::from_bits(f64::from(f32::MAX).to_bits() + 1)),
            Some(f64::INFINITY),
        ] {
            let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
            doc.worlds.insert(
                world_id.clone(),
                super::World {
                    eevee_sun_threshold: threshold,
                    ..super::World::default()
                },
            );
            assert!(doc.validate().is_err());
        }
    }
    #[test]
    fn eevee_world_sun_shadow_settings_round_trip_defaults_and_validate_rna_ranges() {
        let defaults = super::World::default();
        assert_eq!(
            defaults.eevee_sun_angle,
            Some(f64::from(0.526_f32.to_radians()))
        );
        assert_eq!(defaults.eevee_use_sun_shadow, Some(true));
        assert_eq!(
            defaults.eevee_sun_shadow_maximum_resolution,
            Some(f64::from(0.001_f32))
        );
        assert_eq!(defaults.eevee_sun_shadow_jitter_overblur, Some(10.0));
        assert_eq!(defaults.eevee_sun_shadow_filter_radius, Some(1.0));
        assert_eq!(defaults.eevee_use_sun_shadow_jitter, Some(false));

        let world_id = Id::new("sun_world").unwrap();
        let valid = super::World {
            eevee_sun_angle: Some(f64::from(std::f32::consts::PI)),
            eevee_use_sun_shadow: Some(false),
            eevee_sun_shadow_maximum_resolution: Some(f64::from(f32::MAX)),
            eevee_sun_shadow_jitter_overblur: Some(100.0),
            eevee_sun_shadow_filter_radius: Some(f64::from(f32::MAX)),
            eevee_use_sun_shadow_jitter: Some(true),
            ..super::World::default()
        };
        let serialized = serde_json::to_value(&valid).unwrap();
        let round_trip: super::World = serde_json::from_value(serialized.clone()).unwrap();
        assert_eq!(round_trip, valid);
        assert_eq!(serialized["eevee_use_sun_shadow"], false);

        for invalid in [
            super::World {
                eevee_sun_angle: Some(-f64::EPSILON),
                ..super::World::default()
            },
            super::World {
                eevee_sun_angle: Some(f64::from(std::f32::consts::PI) + 1.0),
                ..super::World::default()
            },
            super::World {
                eevee_sun_shadow_maximum_resolution: Some(f64::INFINITY),
                ..super::World::default()
            },
            super::World {
                eevee_sun_shadow_jitter_overblur: Some(f64::from_bits(100.0_f64.to_bits() + 1)),
                ..super::World::default()
            },
            super::World {
                eevee_sun_shadow_filter_radius: Some(-f64::EPSILON),
                ..super::World::default()
            },
        ] {
            let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
            doc.worlds.insert(world_id.clone(), invalid);
            assert!(doc.validate().is_err());
        }

        let unknown = super::World {
            eevee_sun_angle: None,
            eevee_use_sun_shadow: None,
            eevee_sun_shadow_maximum_resolution: None,
            eevee_sun_shadow_jitter_overblur: None,
            eevee_sun_shadow_filter_radius: None,
            eevee_use_sun_shadow_jitter: None,
            ..super::World::default()
        };
        let serialized = serde_json::to_value(&unknown).unwrap();
        for field in [
            "eevee_sun_angle",
            "eevee_use_sun_shadow",
            "eevee_sun_shadow_maximum_resolution",
            "eevee_sun_shadow_jitter_overblur",
            "eevee_sun_shadow_filter_radius",
            "eevee_use_sun_shadow_jitter",
        ] {
            assert!(serialized[field].is_null(), "{field}");
        }
        assert_eq!(
            serde_json::from_value::<super::World>(serialized).unwrap(),
            unknown
        );
    }

    #[test]
    fn cycles_hair_shadow_groups_round_trip_and_remain_absent_for_old_scenes() {
        let doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        let mut value = serde_json::to_value(doc).unwrap();
        let scene = &value["scenes"]["scene_main"];
        assert!(scene.get("cycles_hair_shadow_groups").is_none());

        value["scenes"]["scene_main"]["cycles_hair_shadow_groups"] = serde_json::json!([{
            "view_layer_id": "view_main",
            "geometry_key": {"kind": "data", "data_id": null},
            "representative_node_id": "hair_first",
            "member_node_ids": ["hair_first", "hair_second"],
            "members": [
                {
                    "instance": {
                        "real_object_node_id": "hair_first",
                        "parent_node_id": "hair_first",
                        "persistent_id": [0, 0, 0, 0, 0, 0, 0, 0],
                        "is_instance": false,
                        "use_particle_hair": false
                    },
                    "object_to_world": [[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],
                                        [0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]],
                    "object_color": [1.0,0.0,0.0,1.0],
                    "object_pass_index": 0,
                    "object_random_id": 1,
                    "slot_0_material_id": null,
                    "ray_visibility": {
                        "camera": true, "diffuse": true, "glossy": true,
                        "transmission": true, "volume_scatter": true, "shadow": true
                    }
                },
                {
                    "instance": {
                        "real_object_node_id": "hair_first",
                        "parent_node_id": "hair_second",
                        "persistent_id": [1, 0, 0, 0, 0, 0, 0, 0],
                        "is_instance": true,
                        "use_particle_hair": false
                    },
                    "object_to_world": [[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,2.0],
                                        [0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]],
                    "object_color": [0.0,0.0,1.0,1.0],
                    "object_pass_index": 22,
                    "object_random_id": -947_307_473,
                    "slot_0_material_id": "mat_hair",
                    "ray_visibility": {
                        "camera": false, "diffuse": true, "glossy": true,
                        "transmission": true, "volume_scatter": true, "shadow": true
                    }
                },
                {
                    "instance": {
                        "real_object_node_id": "hair_second",
                        "parent_node_id": "hair_second",
                        "persistent_id": [0, 0, 0, 0, 0, 0, 0, 0],
                        "is_instance": false,
                        "use_particle_hair": false
                    },
                    "object_to_world": [[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],
                                        [0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]],
                    "object_color": [0.0,0.0,1.0,1.0],
                    "object_pass_index": 0,
                    "object_random_id": 3,
                    "slot_0_material_id": null,
                    "ray_visibility": {
                        "camera": true, "diffuse": true, "glossy": true,
                        "transmission": true, "volume_scatter": true, "shadow": true
                    }
                }
            ],
            "source_curve_indices": null,
            "evaluated_hair_curves": null,
            "source_input_fingerprint": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
        }]);
        value["materials"]["mat_hair"] = serde_json::json!({"name": "Hair"});
        let loaded: SceneDoc = serde_json::from_value(value).unwrap();
        let groups = loaded.scenes[&Id::from_static("scene_main")]
            .cycles_hair_shadow_groups
            .as_ref()
            .unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].view_layer_id, Id::from_static("view_main"));
        assert_eq!(
            groups[0].geometry_key,
            super::CyclesHairShadowGeometryKey::Data { data_id: None }
        );
        assert!(groups[0].source_curve_indices.is_none());
        assert_eq!(groups[0].members.len(), 3);
        assert!(groups[0].members[1].instance.is_instance);
        assert_eq!(
            groups[0].members[1].instance.real_object_node_id,
            Id::from_static("hair_first")
        );
        assert_eq!(
            groups[0].members[0].instance.real_object_node_id,
            Id::from_static("hair_first")
        );
        assert_eq!(
            groups[0].members[2].instance.real_object_node_id,
            Id::from_static("hair_second")
        );
        assert_eq!(groups[0].members[1].object_pass_index, 22);
        assert_eq!(groups[0].members[1].object_color, [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(
            groups[0].members[1].slot_0_material_id,
            Some(Id::from_static("mat_hair"))
        );
        assert_eq!(groups[0].members[1].object_random_id, -947_307_473);
        assert_eq!(groups[0].members[1].object_to_world[1][3], 2.0);
        assert!(
            !groups[0].members[1]
                .ray_visibility
                .allows(super::RayKind::Camera)
        );
        assert_eq!(
            groups[0].source_input_fingerprint.as_deref(),
            Some(format!("sha256:{}", "0".repeat(64)).as_str())
        );
        assert_eq!(
            groups[0].representative_node_id,
            Id::from_static("hair_first")
        );
        assert_eq!(
            groups[0].member_node_ids,
            [
                Id::from_static("hair_first"),
                Id::from_static("hair_second")
            ]
        );
        let mut empty_value = serde_json::to_value(SceneDoc::new(
            "00000000-0000-4000-8000-000000000001".to_owned(),
        ))
        .unwrap();
        empty_value["scenes"]["scene_main"]["cycles_hair_shadow_groups"] = serde_json::json!([]);
        let empty: SceneDoc = serde_json::from_value(empty_value).unwrap();
        assert_eq!(
            empty.scenes[&Id::from_static("scene_main")].cycles_hair_shadow_groups,
            Some(Vec::new())
        );
        assert_eq!(
            serde_json::to_value(empty).unwrap()["scenes"]["scene_main"]["cycles_hair_shadow_groups"],
            serde_json::json!([])
        );
    }

    #[test]
    fn cycles_hair_source_fingerprint_tracks_geometry_node_edits_in_place() {
        let scene_id = Id::from_static("scene_main");
        let object_id = Id::new("hair_object").unwrap();
        let data_id = Id::new("hair_data").unwrap();
        let group_id = Id::new("geometry_group").unwrap();
        let mat_a = Id::new("material_a").unwrap();
        let mat_b = Id::new("material_b").unwrap();
        let graph_node_id = Id::new("set_position").unwrap();
        let mut doc = SceneDoc::new("fingerprint-test".to_owned());
        doc.materials
            .insert(mat_a.clone(), super::Material::default());
        doc.materials
            .insert(mat_b.clone(), super::Material::default());
        let mut graph = crate::graph::NodeGroup::new("Geometry", crate::graph::GraphKind::Geometry);
        let mut set_position = crate::graph::GraphNode::new("GeometryNodeSetPosition");
        set_position
            .inputs
            .insert("Offset".to_owned(), serde_json::json!([0.0, 0.0, 0.0]));
        graph.nodes.insert(graph_node_id.clone(), set_position);
        doc.node_groups.insert(group_id.clone(), graph);
        doc.data_blocks.insert(
            data_id.clone(),
            super::DataBlock {
                data_type: "hair_curves".to_owned(),
                mesh: None,
                hair_curves: Some(super::HairCurvesData {
                    curves: vec![super::HairCurve {
                        points: vec![[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                        radius: 0.1,
                    }],
                    ..super::HairCurvesData::default()
                }),
                materials: vec![Some(mat_a.clone())],
                ..super::DataBlock::default()
            },
        );
        let mut modifier = Modifier {
            id: Id::new("geometry_modifier").unwrap(),
            modifier_type: "nodes".to_owned(),
            name: "Geometry Nodes".to_owned(),
            enabled: true,
            show_viewport: true,
            show_render: true,
            params: serde_json::Map::new(),
            baked_ocean_cache: None,
            binding_data: None,
            runtime: super::ModifierRuntime::default(),
        };
        modifier
            .params
            .insert("node_group".to_owned(), serde_json::json!(group_id));
        let mut node = super::Node {
            name: "Hair".to_owned(),
            kind: "hair_curves".to_owned(),
            data: Some(data_id.clone()),
            materials: vec![Some(mat_a.clone())],
            material_links: vec![super::MaterialLink::Data],
            modifiers: vec![modifier],
            ..super::Node::default()
        };
        node.properties.insert(
            "blender_object_metadata".to_owned(),
            serde_json::json!({
                "rna_properties": {"color": [1.0, 1.0, 1.0, 1.0], "pass_index": 4}
            }),
        );
        doc.nodes.insert(object_id.clone(), node);
        doc.collections
            .get_mut(&Id::from_static("collection_root"))
            .unwrap()
            .objects
            .push(object_id.clone());
        let group = super::CyclesHairShadowGroup {
            view_layer_id: Id::from_static("view_main"),
            geometry_key: super::CyclesHairShadowGeometryKey::Data {
                data_id: Some(data_id.clone()),
            },
            representative_node_id: object_id.clone(),
            member_node_ids: vec![object_id.clone()],
            members: vec![super::CyclesHairShadowMember {
                instance: super::CyclesHairShadowInstanceKey {
                    real_object_node_id: object_id.clone(),
                    parent_node_id: Some(Id::from_static("hair_object")),
                    persistent_id: [0; 8],
                    is_instance: false,
                    use_particle_hair: false,
                },
                object_to_world: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                object_color: [1.0; 4],
                object_pass_index: 4,
                object_random_id: -1,
                slot_0_material_id: Some(mat_a.clone()),
                ray_visibility: super::RayVisibility::default(),
            }],
            source_curve_indices: None,
            // An identity Geometry Nodes offset is represented by the authored keys, so no
            // evaluated payload exists; the dependency fingerprint must still cover the graph.
            evaluated_hair_curves: None,
            source_input_fingerprint: None,
        };
        let first = doc
            .cycles_hair_shadow_source_fingerprint(&scene_id, &group)
            .unwrap()
            .unwrap();
        {
            let node = doc.nodes.get_mut(&object_id).unwrap();
            node.transform.translation[0] = 3.0;
            let rna = node
                .properties
                .get_mut("blender_object_metadata")
                .and_then(|metadata| metadata.get_mut("rna_properties"))
                .and_then(serde_json::Value::as_object_mut)
                .unwrap();
            rna.insert("color".to_owned(), serde_json::json!([0.1, 0.2, 0.3, 1.0]));
            rna.insert("pass_index".to_owned(), serde_json::json!(99));
            node.materials[0] = Some(mat_b.clone());
        }
        doc.data_blocks.get_mut(&data_id).unwrap().materials[0] = Some(mat_b.clone());
        let live_context = doc
            .cycles_hair_shadow_source_fingerprint(&scene_id, &group)
            .unwrap()
            .unwrap();
        assert_eq!(first, live_context);

        let parent_id = Id::new("hair_parent").unwrap();
        doc.nodes.insert(
            parent_id.clone(),
            super::Node {
                name: "Instancer".to_owned(),
                ..super::Node::default()
            },
        );
        let mut instance_group = group.clone();
        let instance = &mut instance_group.members[0];
        instance.instance.parent_node_id = Some(parent_id);
        instance.instance.persistent_id = [1, 0, 0, 0, 0, 0, 0, 0];
        instance.instance.is_instance = true;
        instance.object_to_world[0][3] = 3.0;
        instance.object_color = [0.1, 0.2, 0.3, 1.0];
        instance.object_pass_index = 99;
        instance.object_random_id = -947_307_473;
        instance.slot_0_material_id = Some(mat_b.clone());
        let instance_fingerprint = doc
            .cycles_hair_shadow_source_fingerprint(&scene_id, &instance_group)
            .unwrap()
            .unwrap();
        {
            let node = doc.nodes.get_mut(&object_id).unwrap();
            node.transform.translation[0] = 4.0;
            let rna = node
                .properties
                .get_mut("blender_object_metadata")
                .and_then(|metadata| metadata.get_mut("rna_properties"))
                .and_then(serde_json::Value::as_object_mut)
                .unwrap();
            rna.insert("color".to_owned(), serde_json::json!([0.9, 0.8, 0.7, 1.0]));
            rna.insert("pass_index".to_owned(), serde_json::json!(100));
            node.materials[0] = Some(mat_a.clone());
        }
        doc.data_blocks.get_mut(&data_id).unwrap().materials[0] = Some(mat_a);
        let changed_instance = doc
            .cycles_hair_shadow_source_fingerprint(&scene_id, &instance_group)
            .unwrap()
            .unwrap();
        assert_ne!(instance_fingerprint, changed_instance);
        let before_graph_edit = doc
            .cycles_hair_shadow_source_fingerprint(&scene_id, &group)
            .unwrap()
            .unwrap();
        doc.node_groups
            .get_mut(&group_id)
            .unwrap()
            .nodes
            .get_mut(&graph_node_id)
            .unwrap()
            .inputs
            .insert("Offset".to_owned(), serde_json::json!([1.0, 0.0, 0.0]));
        let changed = doc
            .cycles_hair_shadow_source_fingerprint(&scene_id, &group)
            .unwrap()
            .unwrap();
        assert_ne!(before_graph_edit, changed);
    }

    #[test]
    fn render_settings_keep_eevee_volumetric_defaults_implicit_for_old_projects() {
        let render: super::RenderSettings = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(render.eevee_volumetric_tile_size, 8);
        assert_eq!(render.eevee_volumetric_samples, 64);
        assert_eq!(render.eevee_volumetric_sample_distribution, 0.8);
        assert!(!render.eevee_use_volume_custom_range);
        assert_eq!(render.eevee_volumetric_start, 0.1);
        assert_eq!(render.eevee_volumetric_end, 100.0);

        let value = serde_json::to_value(render).unwrap();
        for field in [
            "eevee_volumetric_tile_size",
            "eevee_volumetric_samples",
            "eevee_volumetric_sample_distribution",
            "eevee_use_volume_custom_range",
            "eevee_volumetric_start",
            "eevee_volumetric_end",
        ] {
            assert!(value.get(field).is_none(), "{field} should be omitted");
        }
    }
    #[test]
    fn eevee_raytracing_setting_round_trips_true_false_and_unknown() {
        for value in [Some(true), Some(false), None] {
            let render = super::RenderSettings {
                eevee_use_raytracing: value,
                ..super::RenderSettings::default()
            };
            let serialized = serde_json::to_value(&render).unwrap();
            assert_eq!(serialized["eevee_use_raytracing"].as_bool(), value);
            let round_trip: super::RenderSettings = serde_json::from_value(serialized).unwrap();
            assert_eq!(round_trip.eevee_use_raytracing, value);
        }
    }
    #[test]
    fn eevee_local_volume_probe_marker_round_trips_unknown_and_native_default() {
        let mut legacy = serde_json::to_value(SceneDoc::new("scene-test".to_owned())).unwrap();
        assert_eq!(
            legacy["scenes"]["scene_main"]["eevee_has_local_volume_probe_grids"].as_bool(),
            Some(false)
        );
        legacy["scenes"]["scene_main"]
            .as_object_mut()
            .unwrap()
            .remove("eevee_has_local_volume_probe_grids");
        let legacy: SceneDoc = serde_json::from_value(legacy).unwrap();
        assert_eq!(
            legacy.scenes[&Id::from_static("scene_main")].eevee_has_local_volume_probe_grids,
            None
        );

        for value in [Some(true), Some(false), None] {
            let mut source = serde_json::to_value(SceneDoc::new("scene-test".to_owned())).unwrap();
            source["scenes"]["scene_main"]["eevee_has_local_volume_probe_grids"] =
                value.map_or(serde_json::Value::Null, serde_json::Value::Bool);
            let round_trip: SceneDoc = serde_json::from_value(source).unwrap();
            assert_eq!(
                round_trip.scenes[&Id::from_static("scene_main")]
                    .eevee_has_local_volume_probe_grids,
                value
            );
        }
    }
    #[test]
    fn eevee_raytracing_and_fast_gi_options_round_trip_and_validate() {
        assert_eq!(
            super::RenderSettings::default().eevee_clamp_surface_indirect,
            Some(10.0)
        );
        let render = super::RenderSettings {
            eevee_use_raytracing: Some(true),
            eevee_use_fast_gi: Some(true),
            eevee_raytracing_method: Some(super::EeveeRayTracingMethod::Screen),
            eevee_raytracing_resolution_scale: Some(4),
            eevee_raytracing_screen_trace_quality: Some(0.75),
            eevee_raytracing_screen_trace_thickness: Some(0.2),
            eevee_raytracing_trace_max_roughness: Some(0.8),
            eevee_raytracing_use_backface_hit: Some(false),
            eevee_raytracing_backface_radiance_scale: Some(0.4),
            eevee_raytracing_use_denoise: Some(true),
            eevee_raytracing_denoise_spatial: Some(false),
            eevee_raytracing_denoise_temporal: Some(true),
            eevee_raytracing_denoise_bilateral: Some(false),
            eevee_fast_gi_bias: Some(0.2),
            eevee_fast_gi_resolution: Some(8),
            eevee_fast_gi_step_count: Some(32),
            eevee_fast_gi_ray_count: Some(5),
            eevee_fast_gi_quality: Some(0.7),
            eevee_fast_gi_distance: Some(125.0),
            eevee_fast_gi_thickness_near: Some(0.3),
            eevee_fast_gi_method: Some(super::EeveeFastGiMethod::AmbientOcclusionOnly),
            eevee_clamp_surface_indirect: Some(f64::from(18.75_f32)),
            ..super::RenderSettings::default()
        };
        assert!(render.validate().is_ok());
        let serialized = serde_json::to_value(&render).unwrap();
        assert_eq!(serialized["eevee_raytracing_method"], "screen");
        assert_eq!(serialized["eevee_fast_gi_method"], "ambient_occlusion_only");
        assert_eq!(
            serialized["eevee_clamp_surface_indirect"],
            f64::from(18.75_f32)
        );
        let round_trip: super::RenderSettings = serde_json::from_value(serialized).unwrap();
        assert_eq!(round_trip, render);

        for invalid in [
            super::RenderSettings {
                eevee_raytracing_resolution_scale: Some(3),
                ..super::RenderSettings::default()
            },
            super::RenderSettings {
                eevee_raytracing_screen_trace_thickness: Some(0.0),
                ..super::RenderSettings::default()
            },
            super::RenderSettings {
                eevee_fast_gi_ray_count: Some(17),
                ..super::RenderSettings::default()
            },
            super::RenderSettings {
                eevee_fast_gi_step_count: Some(65),
                ..super::RenderSettings::default()
            },
            super::RenderSettings {
                eevee_clamp_surface_indirect: Some(-f64::EPSILON),
                ..super::RenderSettings::default()
            },
            super::RenderSettings {
                eevee_clamp_surface_indirect: Some(f64::from_bits(
                    f64::from(f32::MAX).to_bits() + 1,
                )),
                ..super::RenderSettings::default()
            },
        ] {
            assert!(invalid.validate().is_err());
        }
    }
    #[test]
    fn eevee_cubemap_resolution_round_trips_only_blender_enum_values() {
        let render: super::RenderSettings = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(render.eevee_gi_cubemap_resolution, Some(512));
        for resolution in super::EEVEE_GI_CUBEMAP_RESOLUTIONS {
            let render = super::RenderSettings {
                eevee_gi_cubemap_resolution: Some(resolution),
                ..super::RenderSettings::default()
            };
            assert!(render.validate().is_ok(), "{resolution}");
            let value = serde_json::to_value(&render).unwrap();
            assert_eq!(value["eevee_gi_cubemap_resolution"], resolution);
            let round_trip: super::RenderSettings = serde_json::from_value(value).unwrap();
            assert_eq!(round_trip.eevee_gi_cubemap_resolution, Some(resolution));
        }
        let unknown = super::RenderSettings {
            eevee_gi_cubemap_resolution: None,
            ..super::RenderSettings::default()
        };
        let value = serde_json::to_value(&unknown).unwrap();
        assert!(value["eevee_gi_cubemap_resolution"].is_null());
        let round_trip: super::RenderSettings = serde_json::from_value(value).unwrap();
        assert_eq!(round_trip.eevee_gi_cubemap_resolution, None);

        let invalid = super::RenderSettings {
            eevee_gi_cubemap_resolution: Some(2049),
            ..super::RenderSettings::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn cycles_curve_render_settings_round_trip_all_shapes_and_subdivisions() {
        for (shape, serialized) in [
            (super::CyclesCurveShape::Ribbons, "ribbons"),
            (super::CyclesCurveShape::Thick, "thick"),
            (super::CyclesCurveShape::ThickLinear, "thick_linear"),
        ] {
            let render = super::RenderSettings {
                cycles_curves_shape: Some(shape),
                cycles_curves_subdivisions: Some(2),
                ..super::RenderSettings::default()
            };
            let value = serde_json::to_value(&render).unwrap();
            assert_eq!(value["cycles_curves_shape"], serialized);
            assert_eq!(value["cycles_curves_subdivisions"], 2);
            let round_trip: super::RenderSettings = serde_json::from_value(value).unwrap();
            assert_eq!(round_trip.cycles_curves_shape, Some(shape));
            assert_eq!(round_trip.cycles_curves_subdivisions, Some(2));
        }

        let unknown = super::RenderSettings {
            cycles_curves_shape: None,
            cycles_curves_subdivisions: None,
            ..super::RenderSettings::default()
        };
        let value = serde_json::to_value(&unknown).unwrap();
        assert!(value["cycles_curves_shape"].is_null());
        assert!(value["cycles_curves_subdivisions"].is_null());
        let round_trip: super::RenderSettings = serde_json::from_value(value).unwrap();
        assert_eq!(round_trip.cycles_curves_shape, None);
        assert_eq!(round_trip.cycles_curves_subdivisions, None);

        for subdivisions in [0, 24] {
            let settings = super::RenderSettings {
                cycles_curves_subdivisions: Some(subdivisions),
                ..super::RenderSettings::default()
            };
            assert!(settings.validate().is_ok());
        }
        let invalid = super::RenderSettings {
            cycles_curves_subdivisions: Some(25),
            ..super::RenderSettings::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn document_validation_rejects_deserialized_malformed_color_curve() {
        let mut value = serde_json::to_value(SceneDoc::new(
            "00000000-0000-4000-8000-000000000000".to_owned(),
        ))
        .unwrap();
        let mut curve = serde_json::to_value(crate::color::curve::CurveMapping::default()).unwrap();
        curve["curves"][0]["points"] = serde_json::json!([{
            "location": [0.0, 0.0],
            "handle": "auto"
        }]);
        value["scenes"]["scene_main"]["color_management"]["curve"] = curve;
        let doc: SceneDoc = serde_json::from_value(value).unwrap();

        let error = doc.validate().unwrap_err();
        assert_eq!(error.code, crate::error::ErrorCode::InvalidArgument);
        assert!(error.message.contains("curve 0 needs at least two points"));
    }

    #[test]
    fn scene_id_must_be_uuid_v4_and_revision_must_fit_safe_integer() {
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        doc.validate().unwrap();
        doc.scene_id = "00000000-0000-1000-8000-000000000000".to_owned();
        assert!(doc.validate().is_err());
        doc.scene_id = "00000000-0000-4000-8000-000000000000".to_owned();
        doc.revision = MAX_REVISION;
        assert!(doc.validate().is_ok());
        doc.revision = MAX_REVISION + 1;
        assert!(doc.validate().is_err());
    }

    #[test]
    fn revision_validation_matches_the_json_safe_integer_limit() {
        assert_eq!(MAX_REVISION, 9_007_199_254_740_991_u64);
        assert!(revision_in_range(9_007_199_254_740_991_u64));
        assert!(!revision_in_range(9_007_199_254_740_992_u64));
    }

    #[test]
    fn typed_camera_light_modifier_and_action_payloads_have_schema_defaults() {
        let camera: CameraData =
            serde_json::from_value(serde_json::json!({"projection":"orthographic"})).unwrap();
        assert_eq!(camera.lens_mm, 50.0);
        assert_eq!(camera.sensor_width_mm, 36.0);
        assert_eq!(camera.ortho_scale, 6.0);
        assert_eq!(camera.clip_start, 0.1);
        assert_eq!(camera.clip_end, 1000.0);
        assert_eq!(camera.shift, [0.0, 0.0]);

        let light: LightData = serde_json::from_value(serde_json::json!({
            "light_type":"area","energy":10.0,"radius":0.1,"spot_size":std::f64::consts::FRAC_PI_4,"spot_blend":0.15
        })).unwrap();
        assert_eq!(light.color, [1.0, 1.0, 1.0]);
        assert_eq!(light.use_shadow, Some(true));

        let light_without_shadows: LightData =
            serde_json::from_value(serde_json::json!({"use_shadow":false})).unwrap();
        let roundtrip: LightData =
            serde_json::from_value(serde_json::to_value(light_without_shadows).unwrap()).unwrap();
        assert_eq!(roundtrip.use_shadow, Some(false));

        let modifier: Modifier = serde_json::from_value(serde_json::json!({
            "id":"bevel","type":"bevel","name":"Bevel"
        }))
        .unwrap();
        assert!(modifier.enabled);
        assert!(modifier.params.is_empty());
        assert!(
            serde_json::from_value::<Modifier>(serde_json::json!({
                "id":"bevel","type":"bevel","name":"Bevel","unknown":true
            }))
            .is_err()
        );

        let action: Action = serde_json::from_value(serde_json::json!({
            "name":"Move",
            "fcurves":[{"path":"transform.translation","index":0,"keyframes":[{"frame":1.0,"value":0.0}]}]
        })).unwrap();
        assert_eq!(
            action.fcurves[0].keyframes[0].interpolation,
            Interpolation::Linear
        );
    }

    #[test]
    fn captured_matrix_basis_tracks_static_state_and_rejects_dynamic_state() {
        let effective = super::Transform::default();
        let mut node = super::Node::default();
        assert!(
            serde_json::to_value(&node)
                .unwrap()
                .get("blender_matrix_basis")
                .is_none()
        );
        let fingerprint = node.matrix_basis_fingerprint(&effective).unwrap();
        let identity = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        node.blender_matrix_basis = Some(super::CapturedMatrixBasis {
            matrix_basis: identity,
            fingerprint,
        });
        assert_eq!(
            node.captured_matrix_basis(&effective).unwrap(),
            Some(identity)
        );
        let changed_transform = super::Transform {
            translation: [1.0, 0.0, 0.0],
            ..effective.clone()
        };
        assert_eq!(
            node.captured_matrix_basis(&changed_transform).unwrap(),
            None
        );
        node.properties
            .insert("custom".to_owned(), serde_json::json!(1));
        node.pose
            .insert(super::Id::from_static("bone"), super::PoseBone::default());
        node.constraints.push(super::Constraint::default());
        assert_eq!(
            node.captured_matrix_basis(&effective).unwrap(),
            Some(identity)
        );
        node.properties.clear();
        node.pose.clear();
        node.constraints.clear();
        node.properties.insert(
            "blender_object_metadata".to_owned(),
            serde_json::json!({"delta_location": [1.0, 0.0, 0.0]}),
        );
        assert_eq!(node.captured_matrix_basis(&effective).unwrap(), None);
        node.properties.clear();
        // Without a parent, parent_inverse, parent_type and parent_bone do not affect the capture.
        node.parent_inverse = Some(identity);
        node.parent_type = super::ParentType::Bone;
        node.parent_bone = Some(super::Id::from_static("bone"));
        assert_eq!(
            node.captured_matrix_basis(&effective).unwrap(),
            Some(identity)
        );
        node.parent_inverse = None;
        node.parent_type = super::ParentType::Object;
        node.parent_bone = None;
        node.parent = Some(super::Id::from_static("parent"));
        assert_eq!(node.captured_matrix_basis(&effective).unwrap(), None);
        let fingerprint = node.matrix_basis_fingerprint(&effective).unwrap();
        node.blender_matrix_basis = Some(super::CapturedMatrixBasis {
            matrix_basis: identity,
            fingerprint,
        });
        assert_eq!(
            node.captured_matrix_basis(&effective).unwrap(),
            Some(identity)
        );
        // A parent id change keeps the capture; parent_inverse under a parent does not.
        node.parent = Some(super::Id::from_static("remapped_parent"));
        assert_eq!(
            node.captured_matrix_basis(&effective).unwrap(),
            Some(identity)
        );
        node.parent_inverse = Some(identity);
        assert_eq!(node.captured_matrix_basis(&effective).unwrap(), None);
        node.parent_inverse = None;
        node.parent_type = super::ParentType::Bone;
        assert_eq!(node.captured_matrix_basis(&effective).unwrap(), None);
        node.parent_type = super::ParentType::Object;
        node.parent_bone = Some(super::Id::from_static("bone"));
        assert_eq!(node.captured_matrix_basis(&effective).unwrap(), None);
        node.parent_bone = None;
        node.transform.translation[0] = 1.0;
        assert_eq!(node.captured_matrix_basis(&node.transform).unwrap(), None);
        node.transform.translation[0] = 0.0;
        node.action = Some(super::Id::from_static("action"));
        assert_eq!(node.captured_matrix_basis(&effective).unwrap(), None);
        node.action = None;
        node.nla_tracks =
            vec![serde_json::from_value(serde_json::json!({"id":"track","name":"Track"})).unwrap()];
        assert_eq!(node.captured_matrix_basis(&effective).unwrap(), None);
        node.nla_tracks.clear();
        node.drivers = vec![
            serde_json::from_value(serde_json::json!({
                "id":"driver","path":"transform.translation","index":0,"type":"scripted_expression"
            }))
            .unwrap(),
        ];
        assert_eq!(node.captured_matrix_basis(&effective).unwrap(), None);
        node.drivers.clear();
        node.rigid_body = Some(super::RigidBody::default());
        assert_eq!(node.captured_matrix_basis(&effective).unwrap(), None);
        node.rigid_body = None;
        let mut animated = super::Node {
            action: Some(super::Id::from_static("action")),
            ..super::Node::default()
        };
        let fingerprint = animated.matrix_basis_fingerprint(&effective).unwrap();
        animated.blender_matrix_basis = Some(super::CapturedMatrixBasis {
            matrix_basis: identity,
            fingerprint,
        });
        assert_eq!(animated.captured_matrix_basis(&effective).unwrap(), None);
        animated.action = None;
        assert_eq!(animated.captured_matrix_basis(&effective).unwrap(), None);
        node.blender_matrix_basis.as_mut().unwrap().matrix_basis[0] = f64::INFINITY;
        assert!(node.captured_matrix_basis(&effective).is_err());
    }

    #[test]
    fn scene_document_rejects_unknown_top_level_fields() {
        let mut value = serde_json::to_value(SceneDoc::new(String::new())).unwrap();
        value["unexpected"] = serde_json::json!(true);
        assert!(serde_json::from_value::<SceneDoc>(value).is_err());
    }

    #[test]
    fn material_pass_index_keeps_unknown_distinct_from_legacy_default() {
        assert_eq!(Material::default().pass_index, Some(0));
        let legacy: Material =
            serde_json::from_value(serde_json::json!({"name":"Legacy"})).unwrap();
        assert_eq!(legacy.pass_index, Some(0));

        let unknown: Material = serde_json::from_value(serde_json::json!({
            "name":"Unknown",
            "pass_index":null
        }))
        .unwrap();
        assert_eq!(unknown.pass_index, None);
        assert_eq!(
            serde_json::to_value(unknown).unwrap()["pass_index"],
            serde_json::Value::Null
        );
    }

    #[test]
    fn legacy_material_without_displacement_method_loads_as_bump() {
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        doc.materials.insert(
            Id::new("mat_legacy").unwrap(),
            Material {
                name: "Legacy".to_owned(),
                ..Material::default()
            },
        );
        let mut value = serde_json::to_value(&doc).unwrap();
        value["materials"]["mat_legacy"]
            .as_object_mut()
            .unwrap()
            .remove("displacement_method");
        let loaded: SceneDoc = serde_json::from_value(value).unwrap();
        assert_eq!(
            loaded.materials[&Id::new("mat_legacy").unwrap()].displacement_method,
            "bump"
        );
        let bare: Material = serde_json::from_value(serde_json::json!({"name": "Bare"})).unwrap();
        assert_eq!(bare.displacement_method, "bump");
    }
}

#[cfg(kani)]
#[kani::proof]
fn kani_revision_bound_is_exact() {
    let revision: u64 = kani::any();
    kani::assert(
        MAX_REVISION == 9_007_199_254_740_991_u64,
        "the scene revision limit must be the largest JSON-safe integer",
    );
    kani::assert(
        revision_in_range(revision) == (revision <= 9_007_199_254_740_991_u64),
        "revision range must include exactly the interoperable integer range",
    );
}
