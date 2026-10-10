use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet};

use crate::{
    color::ColorManagement,
    error::{ErrorCode, PotError, Result},
    geom::{
        Mesh, lattice::LatticeData, metaball::MetaballData, pointcloud::PointCloudData,
        volume::VolumeData,
    },
    graph::NodeGroup,
    image::ImageInterpolation,
    mask::Mask,
    sequencer::Sequencer,
};

use super::{
    Id, Registry, Transform,
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
    #[serde(default)]
    pub libraries: Registry<Library>,
    #[serde(default)]
    pub compatibility: Map<String, Value>,
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
                frame_current: 1.0,
                frame_start: 1,
                frame_end: 250,
                fps: 24,
                fps_base: 1.0,
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
            schema_version: 1,
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
            history: HistoryState::default(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != 1 {
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
        for clip in self.movie_clips.values() {
            clip.validate()?;
        }
        for (scene_id, scene) in &self.scenes {
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
                if tile.number <= 1001
                    || !tile_numbers.insert(tile.number)
                    || !dimensions_valid(tile.width, tile.height)
                    || !valid_hash(&tile.blob)
                {
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
        Ok(())
    }
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
    pub frame_current: f64,
    pub frame_start: i32,
    pub frame_end: i32,
    pub fps: u32,
    pub fps_base: f64,
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
    pub frame_start: f64,
    pub frame_end: f64,
    pub action_frame_start: f64,
    pub action_frame_end: f64,
    pub scale: f64,
    pub repeat: f64,
    pub blend_type: NlaBlendType,
    pub influence: f64,
    pub extrapolation: NlaExtrapolation,
    pub blend_in: f64,
    pub blend_out: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NlaBlendType {
    Replace,
    Add,
    Combine,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NlaExtrapolation {
    Hold,
    HoldForward,
    Nothing,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionSlot {
    pub id: Id,
    pub node: Id,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RenderSettings {
    pub resolution_x: u32,
    pub resolution_y: u32,
    pub resolution_percentage: u32,
    pub samples: u32,
    pub seed: u32,
    pub max_bounces: u32,
    pub film_transparent: bool,
    pub engine: String,
    pub use_sequencer: bool,
    pub audio_codec: String,
    #[serde(default)]
    pub passes: Vec<String>,
    #[serde(default)]
    pub motion_blur: bool,
    #[serde(default = "default_shutter")]
    pub shutter: f64,
    #[serde(default = "default_motion_blur_samples")]
    pub motion_blur_samples: u32,
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
            film_transparent: false,
            engine: "path".to_owned(),
            use_sequencer: false,
            audio_codec: "wav".to_owned(),
            passes: vec!["combined".to_owned()],
            motion_blur: false,
            shutter: default_shutter(),
            motion_blur_samples: default_motion_blur_samples(),
        }
    }
}

fn default_shutter() -> f64 {
    0.5
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
    Fisheye,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CameraData {
    pub projection: CameraProjection,
    pub lens_mm: f64,
    pub sensor_height_mm: f64,
    pub sensor_fit: String,
    pub sensor_width_mm: f64,
    pub ortho_scale: f64,
    pub clip_start: f64,
    pub clip_end: f64,
    pub shift: [f64; 2],
    pub panorama_type: String,
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
    #[serde(default = "white_rgb")]
    pub color: [f64; 3],
    pub energy: f64,
    pub radius: f64,
    pub spot_size: f64,
    pub spot_blend: f64,
    pub area_shape: String,
    pub area_size: f64,
    pub area_size_y: f64,
}

impl Default for LightData {
    fn default() -> Self {
        Self {
            light_type: LightType::Point,
            color: white_rgb(),
            energy: 1000.0,
            radius: 0.1,
            spot_size: std::f64::consts::FRAC_PI_4,
            spot_blend: 0.15,
            area_shape: "SQUARE".to_owned(),
            area_size: 0.25,
            area_size_y: 0.25,
        }
    }
}

fn white_rgb() -> [f64; 3] {
    [1.0, 1.0, 1.0]
}
fn default_volume_color() -> [f64; 3] {
    [0.5; 3]
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
}

impl Default for World {
    fn default() -> Self {
        Self {
            color: white_rgb(),
            background_color: None,
            strength: 1.0,
            node_tree: None,
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
pub struct Modifier {
    pub id: Id,
    #[serde(rename = "type")]
    pub modifier_type: String,
    pub name: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub params: Map<String, Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[doc(hidden)]
    pub binding_data: Option<Value>,
    #[serde(skip)]
    #[doc(hidden)]
    pub runtime: ModifierRuntime,
}

#[doc(hidden)]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModifierRuntime {
    pub target_mesh: Option<Value>,
    #[doc(hidden)]
    pub target_to_subject: Option<[f64; 16]>,
    pub operand_mesh: Option<Value>,
    pub warp_from: Option<[f64; 3]>,
    pub warp_to: Option<[f64; 3]>,
    pub hook_target_position: Option<[f64; 3]>,
    pub use_render_levels: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Action {
    pub name: String,
    #[serde(default)]
    pub fcurves: Vec<FCurve>,
    #[serde(default)]
    pub slots: Vec<ActionSlot>,
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
    #[serde(default)]
    pub data: Option<Id>,
    #[serde(default)]
    pub materials: Vec<Id>,
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
    pub properties: Map<String, Value>,
    #[serde(default)]
    pub rigid_body: Option<RigidBody>,
    #[serde(default)]
    pub force_field: Option<ForceField>,
    #[serde(default)]
    pub parent_type: ParentType,
    #[serde(default)]
    pub parent_bone: Option<Id>,
    #[serde(default)]
    pub pose: BTreeMap<Id, PoseBone>,
    #[serde(default)]
    pub constraints: Vec<Constraint>,
    #[serde(default)]
    pub drivers: Vec<Driver>,
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
            data: None,
            materials: Vec::new(),
            modifiers: Vec::new(),
            visible: true,
            render_visible: true,
            selectable: true,
            action: None,
            nla_tracks: Vec::new(),
            properties: Map::new(),
            rigid_body: None,
            force_field: None,
            parent_type: ParentType::default(),
            parent_bone: None,
            pose: BTreeMap::new(),
            constraints: Vec::new(),
            drivers: Vec::new(),
        }
    }
}

fn default_true() -> bool {
    true
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CurvePoint {
    pub co: [f64; 3],
    pub handle_left: [f64; 3],
    pub handle_right: [f64; 3],
    pub handle_type: CurveHandleType,
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
    pub taper: Option<Id>,
    pub fill_mode: CurveFillMode,
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
            taper: None,
            fill_mode: CurveFillMode::None,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TextObjectData {
    pub body: String,
    /// Resource URI or external file path; geometry uses the built-in font when unavailable.
    pub font: String,
    pub font_name: String,
    pub size: f64,
    pub align_x: String,
    pub align_y: String,
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
            size: 1.0,
            align_x: "left".to_owned(),
            align_y: "baseline".to_owned(),
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct HairCurvesData {
    pub curves: Vec<HairCurve>,
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
    pub blob: String,
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
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextureRef {
    pub image: Id,
    #[serde(default)]
    pub uv_map: Option<String>,
    #[serde(default)]
    pub interpolation: ImageInterpolation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Material {
    pub name: String,
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
    #[serde(default = "default_ior")]
    pub ior: f64,
    #[serde(default)]
    pub double_sided: bool,
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
    #[serde(default)]
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
}
impl Default for Material {
    fn default() -> Self {
        Self {
            name: String::new(),
            base_color: default_color(),
            alpha_mode: default_alpha_mode(),
            alpha_threshold: default_alpha_threshold(),
            metallic: 0.0,
            roughness: default_roughness(),
            emission_color: [0.0; 3],
            emission_strength: 0.0,
            transmission: 0.0,
            ior: default_ior(),
            double_sided: false,
            node_tree: None,
            base_color_texture: None,
            roughness_texture: None,
            metallic_texture: None,
            normal_texture: None,
            displacement_method: "bump".to_owned(),
            displacement_scale: default_displacement_scale(),
            displacement_midlevel: default_displacement_midlevel(),
            volume_density: 0.0,
            volume_color: default_volume_color(),
            volume_anisotropy: 0.0,
        }
    }
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
    1.45
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
        Action, CameraData, Interpolation, LightData, MAX_REVISION, Modifier, SceneDoc,
        revision_in_range,
    };

    #[test]
    fn default_scene_has_contract_initial_state_and_empty_registries() {
        let doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
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
    fn scene_document_rejects_unknown_top_level_fields() {
        let mut value = serde_json::to_value(SceneDoc::new(String::new())).unwrap();
        value["unexpected"] = serde_json::json!(true);
        assert!(serde_json::from_value::<SceneDoc>(value).is_err());
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
