mod id;
mod rig;
mod scene;
mod transform;

pub use crate::geom::{
    lattice::{LatticeData, LatticeInterpolation},
    metaball::{ElementType, MetaballData, MetaballElement},
    pointcloud::{PointCloudAttribute, PointCloudData, PointCloudPoint},
    volume::{VolumeData, VolumeFileSource, VolumeGeneratedSource, VolumeGrid, VolumeSource},
};
pub use id::{Id, is_valid as is_valid_id};
pub use rig::{
    ArmatureData, Bone, BoneCollection, Constraint, ConstraintType, Driver, DriverType,
    DriverVariable, DriverVariableType, ParentType, PoseBone, ShapeKey, ShapeKeyData, VertexGroup,
};
pub use scene::{
    Action, ActionSlot, CameraData, CameraIntrinsics, CameraProjection, Collection, CurveData,
    CurveDimensions, CurveFillMode, CurveHandleType, CurvePoint, CurveSpline, CurveSplineType,
    DataBlock, Extrapolation, FCurve, ForceField, ForceFieldType, GreasePencilData,
    GreasePencilFrame, GreasePencilLayer, GreasePencilPoint, GreasePencilStroke, HairCurve,
    HairCurvesData, HistoryState, Image, ImageAlphaMode, ImageColorspace, ImageSource, ImageTile,
    Interpolation, Keyframe, Library, LibraryKind, LibraryOverride, LibraryOverrideProperty,
    LibraryStatus, LightData, LightType, MAX_REVISION, Material, Modifier, ModifierRuntime,
    MovieClip, MovieTracking, NlaBlendType, NlaExtrapolation, NlaStrip, NlaTrack, Node,
    PlaneHomography, PlaneTrack, PrimitiveDescriptor, Profile, ReconstructedPoint, Reconstruction,
    RenderSettings, RigidBody, RigidBodyShape, RigidBodyType, RigidBodyWorld, Scene, SceneDoc,
    SolvedCamera, SurfaceData, SurfacePoint, TextObjectData, TextureRef, TimelineMarker,
    TrackingMarker, TrackingObject, TrackingObjectPose, TrackingTrack, UnitSettings, ViewLayer,
    World,
};
pub use transform::{Transform, canonicalize_quaternion, normalize_rotation};

pub type Registry<T> = std::collections::BTreeMap<Id, T>;
