use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::{Id, Registry};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArmatureData {
    #[serde(default)]
    pub bones: Registry<Bone>,
    #[serde(default)]
    pub bone_collections: Registry<BoneCollection>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BoneCollection {
    pub name: String,
    pub bones: Vec<Id>,
    pub visible: bool,
}

impl Default for BoneCollection {
    fn default() -> Self {
        Self {
            name: String::new(),
            bones: Vec::new(),
            visible: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bone {
    pub name: String,
    #[serde(default)]
    pub parent: Option<Id>,
    pub head: [f64; 3],
    pub tail: [f64; 3],
    #[serde(default)]
    pub roll: f64,
    #[serde(default = "default_true")]
    pub deform: bool,
    #[serde(default = "default_true")]
    pub inherit_rotation: bool,
    #[serde(default)]
    pub use_connect: bool,
    #[serde(default)]
    pub custom_shape: Option<Id>,
    #[serde(default = "default_envelope_distance")]
    pub envelope_distance: f64,
    #[serde(default = "one")]
    pub envelope_weight: f64,
    #[serde(default = "default_envelope_radius")]
    pub head_radius: f64,
    #[serde(default = "default_envelope_radius")]
    pub tail_radius: f64,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub bbone_settings: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PoseBone {
    pub translation: [f64; 3],
    pub rotation: [f64; 4],
    pub scale: [f64; 3],
    pub lock_ik: [bool; 3],
    pub use_ik_limit: [bool; 3],
    pub ik_min: [f64; 3],
    pub ik_max: [f64; 3],
    pub ik_stiffness: [f64; 3],
    pub ik_stretch: f64,
}

impl Default for PoseBone {
    fn default() -> Self {
        Self {
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
            lock_ik: [false; 3],
            use_ik_limit: [false; 3],
            ik_min: [-std::f64::consts::PI; 3],
            ik_max: [std::f64::consts::PI; 3],
            ik_stiffness: [0.0; 3],
            ik_stretch: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParentType {
    #[default]
    Object,
    Bone,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintType {
    CopyLocation,
    CopyRotation,
    CopyScale,
    TrackTo,
    DampedTrack,
    LockedTrack,
    StretchTo,
    Transformation,
    MaintainVolume,
    Floor,
    Pivot,
    Shrinkwrap,
    SplineIk,
    LimitLocation,
    LimitRotation,
    LimitScale,
    ChildOf,
    Action,
    Armature,
    CameraSolver,
    ClampTo,
    CopyTransforms,
    FollowPath,
    FollowTrack,
    GeometryAttribute,
    LimitDistance,
    ObjectSolver,
    TransformCache,
    Ik,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Constraint {
    pub id: Id,
    #[serde(rename = "type")]
    pub constraint_type: ConstraintType,
    pub name: String,
    #[serde(default)]
    pub target: Option<Id>,
    #[serde(default)]
    pub subtarget: Option<Id>,
    /// Pose bone that owns this constraint; `None` denotes an object constraint.
    #[serde(default)]
    pub owner_bone: Option<Id>,
    #[serde(default = "one")]
    pub influence: f64,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub params: Map<String, Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inverse_matrix: Option<[f64; 16]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inverse_frame: Option<f64>,
}

impl Default for Constraint {
    fn default() -> Self {
        Self {
            id: Id::from_static("constraint"),
            constraint_type: ConstraintType::CopyLocation,
            name: String::new(),
            target: None,
            subtarget: None,
            owner_bone: None,
            influence: 1.0,
            enabled: true,
            inverse_matrix: None,
            inverse_frame: None,
            params: Map::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VertexGroup {
    pub id: Id,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShapeKey {
    pub id: Id,
    pub name: String,
    #[serde(default)]
    pub value: f64,
    #[serde(default)]
    pub mute: bool,
    #[serde(default)]
    pub slider_min: f64,
    #[serde(default = "one")]
    pub slider_max: f64,
    #[serde(default)]
    pub relative_key: Option<Id>,
    #[serde(default)]
    pub vertex_group: Option<Id>,
    #[serde(default)]
    pub frame: f64,
    #[serde(default)]
    pub positions: BTreeMap<u32, [f64; 3]>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShapeKeyData {
    #[serde(default)]
    pub basis: BTreeMap<u32, [f64; 3]>,
    #[serde(default)]
    pub keys: Registry<ShapeKey>,
    #[serde(default)]
    pub absolute: bool,
    #[serde(default)]
    pub evaluation_time: f64,
    #[serde(default)]
    pub action: Option<Id>,
    /// Blender 5 action-slot display name for the Key datablock's action.
    #[serde(default)]
    pub action_slot: Option<String>,
    #[serde(default)]
    pub muted_action_curves: BTreeSet<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DriverType {
    #[default]
    Average,
    Sum,
    Min,
    Max,
    ScriptedExpression,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DriverVariableType {
    #[default]
    SingleProp,
    Transforms,
    #[serde(rename = "loc_diff", alias = "location_difference")]
    LocationDifference,
    #[serde(
        rename = "rotation_diff",
        alias = "rot_diff",
        alias = "rotation_difference"
    )]
    RotationDifference,
    #[serde(rename = "context_prop")]
    ContextProperty,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DriverVariable {
    pub name: String,
    #[serde(rename = "type")]
    pub variable_type: DriverVariableType,
    pub target: Id,
    pub path: String,
    #[serde(default)]
    pub index: u32,
    #[serde(default)]
    pub target_2: Option<Id>,
    #[serde(default)]
    pub transform_space: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Driver {
    pub id: Id,
    pub path: String,
    pub index: u32,
    #[serde(rename = "type")]
    pub driver_type: DriverType,
    #[serde(default)]
    pub variables: Vec<DriverVariable>,
    #[serde(default)]
    pub expression: Option<String>,
}

fn default_true() -> bool {
    true
}

fn one() -> f64 {
    1.0
}
fn default_envelope_distance() -> f64 {
    0.25
}

fn default_envelope_radius() -> f64 {
    0.1
}
