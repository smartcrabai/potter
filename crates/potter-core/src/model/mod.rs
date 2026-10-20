mod avatar;
mod id;
pub mod migrate;
mod rig;
pub use migrate::{ParsedScene, parse_scene, upgrade_scene};
mod scene;
mod transform;

pub use crate::geom::{
    lattice::{LatticeData, LatticeInterpolation},
    metaball::{ElementType, MetaballData, MetaballElement},
    pointcloud::{PointCloudAttribute, PointCloudData, PointCloudPoint},
    volume::{VolumeData, VolumeFileSource, VolumeGeneratedSource, VolumeGrid, VolumeSource},
};
pub(crate) use avatar::validate_spring;
pub use avatar::{
    AVATAR_FEATURE_DESCRIPTORS, AvatarCommercialUsage, AvatarCreditNotation, AvatarExpression,
    AvatarExpressionOverride, AvatarFeatureDescriptor, AvatarFirstPerson, AvatarLegacyAllowedUser,
    AvatarLegacyBinaryPermission, AvatarLegacyLicenseName, AvatarLegacyMeta, AvatarLookAt,
    AvatarLookAtType, AvatarMaterialColorBind, AvatarMaterialColorType, AvatarMeshAnnotation,
    AvatarMeshAnnotationType, AvatarMeta, AvatarModification, AvatarMorphTargetBind, AvatarMtoon,
    AvatarMtoonShadingShiftTexture, AvatarNodeRef, AvatarOutlineWidthMode, AvatarPermission,
    AvatarRangeMap, AvatarSettings, AvatarSpring, AvatarSpringBone, AvatarSpringCollider,
    AvatarSpringColliderGroup, AvatarSpringColliderShape, AvatarSpringJoint,
    AvatarTextureTransform, AvatarTextureTransformBind, AvatarVersion, is_humanoid_role,
    validate_avatar_settings,
};
pub use id::{Id, is_valid as is_valid_id};
pub use rig::{
    ArmatureData, Bone, BoneCollection, Constraint, ConstraintType, Driver, DriverType,
    DriverVariable, DriverVariableType, ParentType, PoseBone, ShapeKey, ShapeKeyData, VertexGroup,
};
pub use scene::{
    Action, ActionSlot, BLENDER_PI, BLENDER_RIGHT_ANGLE, BLENDER_TURN, CameraData,
    CameraIntrinsics, CameraProjection, CapturedMatrixBasis, Collection, CurveData,
    CurveDimensions, CurveFillMode, CurveHandleType, CurvePoint, CurveSpline, CurveSplineType,
    DEFAULT_FILTER_WIDTH, DataBlock, Extrapolation, FCurve, FISHEYE_FOV_MAX, FISHEYE_FOV_MIN,
    FISHEYE_LENS_MAX, FISHEYE_LENS_MIN, ForceField, ForceFieldType, GreasePencilData,
    GreasePencilFrame, GreasePencilLayer, GreasePencilPoint, GreasePencilStroke, HairCurve,
    HairCurvesData, HistoryState, Image, ImageAlphaMode, ImageColorspace, ImageSource, ImageTile,
    Interpolation, Keyframe, Library, LibraryKind, LibraryOverride, LibraryOverrideProperty,
    LibraryStatus, LightData, LightType, MAX_REVISION, Material, MaterialLink, Modifier,
    ModifierRuntime, MovieClip, MovieTracking, NlaBlendType, NlaExtrapolation, NlaStrip, NlaTrack,
    Node, PixelFilterType, PlaneHomography, PlaneTrack, PrimitiveDescriptor, Profile,
    ReconstructedPoint, Reconstruction, RenderSettings, RigidBody, RigidBodyShape, RigidBodyType,
    RigidBodyWorld, Scene, SceneDoc, SolvedCamera, SurfaceData, SurfacePoint, TextObjectData,
    TextureRef, TimelineMarker, TrackingMarker, TrackingObject, TrackingObjectPose, TrackingTrack,
    UnitSettings, ViewLayer, ViewportDisplay, World,
};
pub(crate) use scene::{
    is_identity_matrix, material_link_at, material_slot_count, resolve_material_slot,
};
pub use transform::{Transform, canonicalize_quaternion, normalize_rotation};

pub type Registry<T> = std::collections::BTreeMap<Id, T>;
