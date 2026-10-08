#![recursion_limit = "512"]
#![expect(
    clippy::unwrap_used,
    reason = "the Blender fixture uses fixed valid identifiers and generated fixture data"
)]

use glam::DVec3;
use std::{
    collections::{BTreeMap, HashMap},
    fs, io,
    path::Path,
    process::Command,
    sync::Mutex,
};

use potter::{
    eval::{EvaluationContext, Snapshot},
    model::{Id, SceneDoc},
};
use serde_json::{Value, json};
use tempfile::tempdir;
#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/blender_script_guarded.rs"]
mod blender_script_guarded;
mod common;
#[path = "common/pot_json_guarded.rs"]
mod pot_json_guarded;
#[path = "common/process.rs"]
mod process;

use blender_checked::blender_executable as blender;
use blender_script_guarded::run_blender_script;
use common::TestResult;
use pot_json_guarded::pot_json;
use process::run_guarded;

const CONSTRAINT_TYPES: &[&str] = &[
    "copy_location",
    "copy_rotation",
    "copy_scale",
    "copy_transforms",
    "limit_location",
    "limit_rotation",
    "limit_scale",
    "limit_distance",
    "child_of",
    "damped_track",
    "locked_track",
    "track_to",
    "stretch_to",
    "ik",
    "spline_ik",
    "clamp_to",
    "floor",
    "follow_path",
    "pivot",
    "shrinkwrap",
    "maintain_volume",
    "transformation",
    "transform_cache",
    "armature",
    "action",
    "geometry_attribute",
    "camera_solver",
    "follow_track",
    "object_solver",
];
const FRAMES: [i32; 3] = [1, 2, 3];
static BLENDER_CONSTRAINT_TEST_LOCK: Mutex<()> = Mutex::new(());

const BLENDER_FIXTURE: &str = r#"
import bpy
import json
import math
import os
import sys
from mathutils import Matrix

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
constraint_types = json.load(open(os.path.join(root, "constraint_types.json"), encoding="utf-8"))
scene = bpy.context.scene
scene.render.resolution_x = 160
scene.render.resolution_y = 90
scene.frame_start = 1
scene.frame_end = 3

# A solved movie clip gives Camera Solver, Follow Track, and Object Solver real
# tracking and reconstruction data, rather than inert references.
for frame in range(1, 4):
    image = bpy.data.images.new("TrackingImage%d" % frame, 160, 90)
    image.filepath_raw = os.path.join(root, "frame_%04d.png" % frame)
    image.file_format = "PNG"
    image.save()
clip = bpy.data.movieclips.load(os.path.join(root, "frame_0001.png"), check_existing=False)
clip.name = "RoundtripClip"
clip.tracking.camera.units = "MILLIMETERS"
clip.tracking.camera.focal_length = 50.0
clip.tracking.camera.sensor_width = 36.0
scene.active_clip = clip

background_points = [
    (-2,-1,10),(1,-1,11),(2,2,12),(-1,3,13),
    (3,-2,14),(-3,2,15),(0.3,0.7,10.8),(1.7,0.2,13.5)
]
object_points = [
    (-1,-1,-1),(1,-1,-0.5),(-1,1,0.2),(1,1,0.8),
    (-0.4,0.3,1.4),(0.7,-0.2,1.8),(-0.6,0.8,2.3),(0.2,-0.7,2.6)
]
focal_pixels = 50.0 / 36.0 * 160
background_tracks = []
for index, point in enumerate(background_points):
    track = clip.tracking.tracks.new(name="Static%d" % index)
    background_tracks.append(track)
    for frame, camera_x in ((1,0.0),(2,0.35),(3,1.0)):
        x,y,z = point
        track.markers.insert_frame(
            frame,
            co=(0.5 + focal_pixels * (x-camera_x) / z / 160.0,
                0.5 - focal_pixels * y / z / 90.0))
camera_object = clip.tracking.objects.active
camera_object.keyframe_a = 1
camera_object.keyframe_b = 3
area = next(area for area in bpy.context.screen.areas if area.type == "VIEW_3D")
area.type = "CLIP_EDITOR"
area.spaces.active.clip = clip
with bpy.context.temp_override(area=area, space_data=area.spaces.active):
    bpy.ops.clip.solve_camera()
moving_object = clip.tracking.objects.new(name="Moving")
clip.tracking.objects.active = moving_object
moving_object.keyframe_a = 1
moving_object.keyframe_b = 3
for index, point in enumerate(object_points):
    track = moving_object.tracks.new(name="Object%d" % index)
    for frame, object_x in ((1,0.1),(2,0.3),(3,0.5)):
        camera_x = (0.0,0.35,1.0)[frame-1]
        x,y,z = point
        track.markers.insert_frame(
            frame,
            co=(0.5 + focal_pixels * (x+object_x-camera_x) / (z+8.0) / 160.0,
                0.5 - focal_pixels * y / (z+8.0) / 90.0))
with bpy.context.temp_override(area=area, space_data=area.spaces.active):
    bpy.ops.clip.solve_camera()

# Real target bones allow bone references, head/tail offsets, and CUSTOM spaces.
def make_armature(name, bone_name):
    data = bpy.data.armatures.new(name + "Data")
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bone = data.edit_bones.new(bone_name)
    bone.head = (0.0, 0.0, 0.0)
    bone.tail = (0.0, 1.0, 0.0)
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.select_set(False)
    return obj

reference = make_armature("ReferenceRig", "TargetBone")
reference.location = (2.1, -0.7, 0.8)
for frame, x in ((1,2.1),(2,2.6),(3,3.4)):
    reference.location.x = x
    reference.keyframe_insert(data_path="location", frame=frame)

camera_data = bpy.data.cameras.new("SolvedCameraData")
camera = bpy.data.objects.new("SolvedCamera", camera_data)
scene.collection.objects.link(camera)
scene.camera = camera
# Curve targets are animated through their actual path evaluation time.
curve_data = bpy.data.curves.new("AnimatedPathData", type="CURVE")
curve_data.dimensions = "3D"
curve_data.path_duration = 2
curve_data.use_path = True
curve_data.resolution_u = 12
curve_data.bevel_depth = 0.12
curve_data.bevel_resolution = 2
spline = curve_data.splines.new("BEZIER")
spline.bezier_points.add(2)
for index, (point, coordinate) in enumerate(
    zip(spline.bezier_points, ((-2,0,0),(0,2,1),(3,0,2)))):
    point.co = coordinate
    point.radius = (0.65, 1.35, 0.9)[index]
    point.tilt = (0.12, -0.28, 0.41)[index]
    point.handle_left_type = "AUTO"
    point.handle_right_type = "AUTO"
curve = bpy.data.objects.new("AnimatedPath", curve_data)
scene.collection.objects.link(curve)
for frame, value in ((1,0.2),(2,1.1),(3,2.0)):
    curve_data.eval_time = value
    curve_data.keyframe_insert(data_path="eval_time", frame=frame)

# Mesh data and a named vector attribute are used by Geometry Attribute and Shrinkwrap.
mesh = bpy.data.meshes.new("ConstraintTargetMeshData")
mesh.from_pydata([(-1,-1,0),(1,-1,0),(1,1,0),(-1,1,0)], [], [(0,1,2,3)])
mesh.update()
attribute = mesh.attributes.new(name="ConstraintVector", type="FLOAT_VECTOR", domain="POINT")
attribute.data[0].vector = (0.8, -0.3, 0.6)
mesh_target = bpy.data.objects.new("ConstraintTargetMesh", mesh)
scene.collection.objects.link(mesh_target)
# A non-identity space object makes CUSTOM owner/target spaces observable.
custom_space = bpy.data.objects.new("ConstraintCustomSpace", None)
scene.collection.objects.link(custom_space)
custom_space.location = (1.7, -0.8, 0.55)
custom_space.rotation_euler = (0.17, -0.32, 0.24)

# An actual Alembic cache is required by Transform Cache constraints.
cache_mesh = bpy.data.meshes.new("CacheMeshData")
cache_mesh.from_pydata([(0,0,0),(1,0,0),(0,1,0)], [], [(0,1,2)])
cache_mesh.update()
cache_source = bpy.data.objects.new("CacheSource", cache_mesh)
scene.collection.objects.link(cache_source)
for frame, x in ((1,0.0),(2,0.5),(3,1.4)):
    cache_source.location.x = x
    cache_source.keyframe_insert(data_path="location", frame=frame)
cache_source.select_set(True)
bpy.context.view_layer.objects.active = cache_source
cache_path = os.path.join(root, "constraint_cache.abc")
bpy.ops.wm.alembic_export(
    filepath=cache_path, start=1, end=3, selected=True, flatten=False,
    uvs=False, normals=False, as_background_job=False)
cache_source.select_set(False)
bpy.ops.cachefile.open(filepath=cache_path)
bpy.ops.wm.alembic_import(filepath=cache_path, as_background_job=False)
cache_file = bpy.data.cache_files.get(os.path.basename(cache_path))
if cache_file is None:
    cache_file = next(iter(bpy.data.cache_files))
cache_file.frame_offset = 0.25
cache_file.scale = 1.1
cache_object_path = next(item.path for item in cache_file.object_paths
                         if item.path.startswith("/CacheSource/"))

# Action constraints retain an actual named action reference.
action = bpy.data.actions.new("ConstraintAction")

# Use one independent object owner and one independent pose-bone owner per type.
rig_data = bpy.data.armatures.new("ConstraintOwnersData")
rig = bpy.data.objects.new("PoseConstraintOwners", rig_data)
scene.collection.objects.link(rig)
for frame, delta in ((1,0.0),(2,0.45),(3,1.1)):
    rig.location.x = delta
    rig.keyframe_insert(data_path="location", frame=frame)
rig.select_set(True)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="EDIT")
pose_names = {}
for index, kind in enumerate(constraint_types):
    stem = kind.upper()
    root_bone = rig_data.edit_bones.new("Root_" + stem)
    root_bone.head = (index * 2.0, 0.0, 0.0)
    root_bone.tail = (index * 2.0, 0.8, 0.0)
    tip_bone = rig_data.edit_bones.new("Tip_" + stem)
    tip_bone.head = root_bone.tail
    tip_bone.tail = (index * 2.0, 1.6, 0.2)
    tip_bone.parent = root_bone
    tip_bone.use_connect = True
    pose_names[kind] = tip_bone.name
bpy.ops.object.mode_set(mode="OBJECT")
rig.select_set(False)

pole_rig = make_armature("PoleRig", "PoleBone")
pole_rig.location = (0.2, 0.4, 1.5)
mix_mode_defaults = {}
for index, kind in enumerate(constraint_types):
    stem = kind.upper()
    native_kind = "TRANSFORM" if kind == "transformation" else stem
    owner = bpy.data.objects.new("Object_" + stem, None)
    scene.collection.objects.link(owner)
    owner.location = (0.25 + index * 0.11, 0.35, -0.2)
    for frame, delta in ((1,0.0),(2,0.45),(3,1.1)):
        owner.location.x = 0.25 + index * 0.11 + delta
        owner.rotation_euler.y = 0.08 + delta * 0.1
        owner.keyframe_insert(data_path="location", frame=frame)
        owner.keyframe_insert(data_path="rotation_euler", frame=frame)

    constraint = owner.constraints.new(native_kind)
    constraint.name = "Object_" + stem
    pose_constraint = rig.pose.bones[pose_names[kind]].constraints.new(native_kind)
    pose_constraint.name = "Pose_" + stem
    for current in (constraint, pose_constraint):
        current.influence = 0.65
        current.mute = False
        current.owner_space = "CUSTOM" if index % 2 == 0 else "LOCAL"
        current.target_space = "LOCAL" if index % 2 == 0 else "CUSTOM"
        if hasattr(current, "space_object"):
            current.space_object = custom_space if kind == "transform_cache" else mesh_target
        if hasattr(current, "space_subtarget"):
            current.space_subtarget = ""
        if hasattr(current, "head_tail"):
            current.head_tail = 0.37
        if hasattr(current, "mix_mode"):
            prop = current.bl_rna.properties["mix_mode"]
            initial_mix_mode = current.mix_mode
            alternatives = [item.identifier for item in prop.enum_items
                            if item.identifier != initial_mix_mode]
            if alternatives:
                current.mix_mode = alternatives[-1]
                mix_mode_defaults[current.name] = initial_mix_mode
        if hasattr(current, "use_x"):
            current.use_x = True
            current.use_y = False
            current.use_z = True
        if hasattr(current, "target"):
            current.target = (curve if kind in ("follow_path", "clamp_to", "spline_ik")
                              else mesh_target if kind in ("shrinkwrap", "geometry_attribute")
                              else reference)
        if hasattr(current, "subtarget") and current.target == reference:
            current.subtarget = "TargetBone"

        if kind == "child_of":
            current.inverse_matrix = Matrix.Translation((-0.23, 0.17, -0.31))
            for axis in "xyz":
                setattr(current, "use_location_" + axis, axis != "y")
        elif kind == "follow_path":
            current.use_curve_follow = True
            current.forward_axis = "FORWARD_X"
            current.up_axis = "UP_Z"
            if current is constraint:
                current.use_fixed_location = False
                current.offset = 0.1
            else:
                current.use_fixed_location = True
                current.offset_factor = 0.35
        elif kind == "ik":
            current.chain_count = 2
            current.pole_target = pole_rig
            current.pole_subtarget = "PoleBone"
            current.pole_angle = 0.23
        elif kind == "spline_ik":
            current.chain_count = 2
            current.use_chain_offset = True
        elif kind == "armature":
            target = current.targets.new()
            target.target = reference
            target.subtarget = "TargetBone"
            target.weight = 0.7
            current.use_deform_preserve_volume = True
            current.use_bone_envelopes = True
            current.use_current_location = False
        elif kind == "action":
            current.action = action
            if hasattr(current, "target"):
                current.target = reference
            if hasattr(current, "transform_channel"):
                current.transform_channel = "LOCATION_X"
            if hasattr(current, "min"):
                current.min = 0.0
            if hasattr(current, "max"):
                current.max = 1.0
        elif kind == "geometry_attribute":
            current.attribute_name = "ConstraintVector"
        elif kind == "camera_solver":
            current.clip = clip
            current.use_active_clip = False
        elif kind == "follow_track":
            current.clip = clip
            current.track = background_tracks[0].name
            current.camera = camera
            current.use_active_clip = False
            current.use_3d_position = True
        elif kind == "object_solver":
            current.clip = clip
            current.object = "Moving"
            current.camera = camera
            current.use_active_clip = False
            current.set_inverse_pending = True
        elif kind == "transform_cache":
            current.cache_file = cache_file
            current.object_path = cache_object_path

        if kind == "follow_path":
            if current is constraint:
                current.owner_space = "WORLD"
                current.target_space = "WORLD"
            else:
                current.owner_space = "CUSTOM"
                current.target_space = "LOCAL"
        elif current is pose_constraint:
            current.owner_space = "CUSTOM"
            current.target_space = "CUSTOM"
        else:
            current.owner_space = "CUSTOM" if index % 2 == 0 else "LOCAL"
            current.target_space = "LOCAL" if index % 2 == 0 else "CUSTOM"
        if hasattr(current, "space_object"):
            current.space_object = (
                custom_space if kind == "transform_cache"
                else None if kind == "follow_path" and current is constraint
                else mesh_target)
        if hasattr(current, "space_subtarget"):
            current.space_subtarget = ""
    pose_bone = rig.pose.bones[pose_names[kind]]
    pose_bone.rotation_mode = "XYZ"
    for frame, delta in ((1,0.0),(2,0.2),(3,0.6)):
        pose_bone.location.x = delta
        pose_bone.rotation_euler.z = delta * 0.2
        pose_bone.keyframe_insert(data_path="location", frame=frame)
        pose_bone.keyframe_insert(data_path="rotation_euler", frame=frame)

def set_object_solver_inverse(owner, constraint_name, owner_type, bone_name=None):
    view_layer = bpy.context.view_layer
    previous_active = view_layer.objects.active
    previous_selected = list(bpy.context.selected_objects)
    previous_mode = owner.mode
    try:
        if owner.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        for selected in previous_selected:
            selected.select_set(False)
        owner.select_set(True)
        view_layer.objects.active = owner
        if owner_type == "BONE":
            owner.data.bones.active = owner.data.bones[bone_name]
            bpy.ops.object.mode_set(mode="POSE")
            constraint = owner.pose.bones[bone_name].constraints[constraint_name]
        else:
            constraint = owner.constraints[constraint_name]
        with bpy.context.temp_override(
                object=owner, active_object=owner, selected_objects=[owner],
                selected_editable_objects=[owner], scene=scene, view_layer=view_layer):
            result = bpy.ops.constraint.objectsolver_set_inverse(
                constraint=constraint_name, owner=owner_type)
        assert "FINISHED" in result, "Blender did not set the Object Solver inverse"
        if owner.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        tracking_object = constraint.clip.tracking.objects[constraint.object]
        depsgraph = bpy.context.evaluated_depsgraph_get()
        depsgraph.update()
        camera_world = constraint.camera.evaluated_get(depsgraph).matrix_world
        sample = min(
            tracking_object.reconstruction.cameras,
            key=lambda camera: abs(float(camera.frame) - scene.frame_current))
        inverse = (camera_world @ sample.matrix.inverted()).inverted()
        values = [
            float(inverse[row][column])
            for column in range(4) for row in range(4)
        ]
        inverse_map = json.loads(owner.get("potter.object_solver_inverses_json", "{}"))
        entry = {"matrix": values, "frame": float(scene.frame_current)}
        if owner_type == "BONE":
            inverse_map.setdefault("bones", {}).setdefault(bone_name, {})[
                constraint_name] = entry
        else:
            inverse_map.setdefault("objects", {})[constraint_name] = entry
        owner["potter.object_solver_inverses_json"] = json.dumps(
            inverse_map, sort_keys=True)
    finally:
        if owner.mode != previous_mode:
            bpy.ops.object.mode_set(mode=previous_mode)
        owner.select_set(False)
        for selected in previous_selected:
            if selected.name in bpy.data.objects:
                selected.select_set(True)
        view_layer.objects.active = previous_active

if "object_solver" in constraint_types:
    object_solver_owner = bpy.data.objects["Object_OBJECT_SOLVER"]
    set_object_solver_inverse(
        object_solver_owner, "Object_OBJECT_SOLVER", "OBJECT")
    set_object_solver_inverse(
        rig, "Pose_OBJECT_SOLVER", "BONE", pose_names["object_solver"])

bpy.context.view_layer.update()

def stable(value):
    if isinstance(value, bpy.types.ID):
        return value.name
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return [stable(item) for item in value]
    except TypeError:
        return str(value)

def settings(constraint):
    ignored = {"rna_type", "is_override_data", "is_valid", "error_location",
               "error_rotation", "active", "show_expanded"}
    result = {}
    for prop in constraint.bl_rna.properties:
        name = prop.identifier
        if name in ignored or prop.type in ("POINTER", "COLLECTION"):
            continue
        try:
            result[name] = stable(getattr(constraint, name))
        except (AttributeError, TypeError, ValueError):
            continue
    for name in ("target", "space_object", "camera", "depth_object", "pole_target",
                 "action", "clip", "cache_file"):
        if hasattr(constraint, name):
            result["ref_" + name] = stable(getattr(constraint, name))
    if hasattr(constraint, "targets"):
        result["armature_targets"] = [
            {"target": stable(item.target), "subtarget": item.subtarget,
             "weight": float(item.weight)} for item in constraint.targets]
    if hasattr(constraint, "cache_file") and constraint.cache_file:
        cache_filepath = bpy.path.abspath(constraint.cache_file.filepath)
        result["cache_filepath"] = os.path.basename(cache_filepath)
        result["cache_file_exists"] = os.path.isfile(cache_filepath)
        result["cache_object_path"] = constraint.object_path
        result["cache_file_settings"] = {
            key: stable(getattr(constraint.cache_file, key))
            for key in ("is_sequence", "override_frame", "frame", "frame_offset",
                        "forward_axis", "up_axis", "scale", "velocity_name",
                        "velocity_unit")
        }
    return result

def matrix_values(matrix):
    return [float(matrix[row][column]) for column in range(4) for row in range(4)]

def evaluated_matrices():
    values = {"objects": {}, "bones": {}}
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    for kind in constraint_types:
        stem = kind.upper()
        obj = bpy.data.objects["Object_" + stem]
        values["objects"][obj.name] = matrix_values(obj.evaluated_get(depsgraph).matrix_world)
        evaluated_rig = rig.evaluated_get(depsgraph)
        bone_matrix = evaluated_rig.matrix_world @ evaluated_rig.pose.bones[pose_names[kind]].matrix
        values["bones"][pose_names[kind]] = matrix_values(bone_matrix)
    return values

expected = {"frames": {}, "settings": {}, "mix_mode_defaults": mix_mode_defaults}
expected["follow_path_diagnostics"] = {}
for frame in (1, 2, 3):
    scene.frame_set(frame)
    expected["frames"][str(frame)] = evaluated_matrices()
    if "follow_path" in constraint_types:
        object_constraint = bpy.data.objects["Object_FOLLOW_PATH"].constraints["Object_FOLLOW_PATH"]
        pose_constraint = rig.pose.bones[pose_names["follow_path"]].constraints["Pose_FOLLOW_PATH"]
        path_duration = max(float(curve.data.path_duration), 1.0)
        diagnostics = {
            "eval_time": float(curve.data.eval_time),
            "use_path": bool(curve.data.use_path),
            "path_duration": float(curve.data.path_duration),
        }
        for owner_name, current in (
            ("object", object_constraint),
            ("pose", pose_constraint),
        ):
            path_factor = (
                float(current.offset_factor)
                if current.use_fixed_location
                else (float(curve.data.eval_time) - float(current.offset)) / path_duration
            )
            diagnostics[owner_name] = {
                "is_valid": bool(current.is_valid),
                "error_location": float(current.error_location),
                "error_rotation": float(current.error_rotation),
                "owner_space": current.owner_space,
                "target_space": current.target_space,
                "space_object": current.space_object.name_full if current.space_object else None,
                "target": current.target.name_full if current.target else None,
                "influence": float(current.influence),
                "mute": bool(current.mute),
                "use_fixed_location": bool(current.use_fixed_location),
                "path_factor": path_factor,
            }
        expected["follow_path_diagnostics"][str(frame)] = diagnostics
for kind in constraint_types:
    stem = kind.upper()
    object_owner = bpy.data.objects["Object_" + stem]
    object_constraint = object_owner.constraints["Object_" + stem]
    expected["settings"]["Object_" + stem] = settings(object_constraint)
    pose_bone = rig.pose.bones[pose_names[kind]]
    pose_constraint = pose_bone.constraints["Pose_" + stem]
    expected["settings"]["Pose_" + stem] = settings(pose_constraint)
scene.frame_set(2)
curve_depsgraph = bpy.context.evaluated_depsgraph_get()
curve_depsgraph.update()
evaluated_curve = curve.evaluated_get(curve_depsgraph)
curve_mesh = evaluated_curve.to_mesh()
expected["curve_mesh"] = [
    [float(vertex.co[0]), float(vertex.co[1]), float(vertex.co[2])]
    for vertex in curve_mesh.vertices
]
evaluated_curve.to_mesh_clear()
scene.frame_set(3)
def camera_values(camera):
    return {
        "focal_length": float(camera.focal_length),
        "sensor_width": float(camera.sensor_width),
        "principal_point": [float(value) for value in camera.principal_point],
        "units": camera.units,
        "pixel_aspect": float(camera.pixel_aspect),
        "distortion_model": camera.distortion_model,
        "k1": float(camera.k1), "k2": float(camera.k2), "k3": float(camera.k3),
        "division_k1": float(camera.division_k1),
        "division_k2": float(camera.division_k2),
        "nuke_k1": float(camera.nuke_k1), "nuke_k2": float(camera.nuke_k2),
        "nuke_p1": float(camera.nuke_p1), "nuke_p2": float(camera.nuke_p2),
        "brown_k1": float(camera.brown_k1), "brown_k2": float(camera.brown_k2),
        "brown_k3": float(camera.brown_k3), "brown_k4": float(camera.brown_k4),
        "brown_p1": float(camera.brown_p1), "brown_p2": float(camera.brown_p2),
    }

def track_bundles(tracking_object):
    return [
        {"name": track.name, "has_bundle": bool(track.has_bundle),
         "bundle": [float(value) for value in track.bundle] if track.has_bundle else None}
        for track in sorted(tracking_object.tracks, key=lambda track: track.name)
    ]
expected["clip"] = {
    "name": clip.name,
    "filepath_exists": bool(os.path.isfile(bpy.path.abspath(clip.filepath))),
    "filepath": bpy.path.abspath(clip.filepath),
    "camera_tracks": sorted(track.name for track in clip.tracking.objects["Camera"].tracks),
    "object_tracks": sorted(track.name for track in clip.tracking.objects["Moving"].tracks),
    "reconstruction_frames": sorted(int(camera.frame) for camera in clip.tracking.reconstruction.cameras),
    "camera_reconstruction_valid": bool(clip.tracking.reconstruction.is_valid),
    "camera_reconstruction_average_error": float(
        clip.tracking.reconstruction.average_error),
    "moving_reconstruction_valid": bool(
        clip.tracking.objects["Moving"].reconstruction.is_valid),
    "moving_reconstruction_average_error": float(
        clip.tracking.objects["Moving"].reconstruction.average_error),
    "camera": camera_values(clip.tracking.camera),
    "camera_track_bundles": track_bundles(clip.tracking.objects["Camera"]),
    "moving_track_bundles": track_bundles(clip.tracking.objects["Moving"]),
}
expected["reconstruction_cameras"] = [
    {
        "frame": int(sample.frame),
        "matrix": [[float(sample.matrix[row][column]) for column in range(4)]
                   for row in range(3)],
        "average_error": float(sample.average_error),
    }
    for sample in clip.tracking.reconstruction.cameras
]
expected["moving_reconstruction"] = [
    {"frame": float(sample.frame), "matrix": matrix_values(sample.matrix),
     "average_error": float(sample.average_error)}
    for sample in clip.tracking.objects["Moving"].reconstruction.cameras
]
expected["scene_camera"] = scene.camera.name if scene.camera else None
with open(os.path.join(root, "blender_before.json"), "w", encoding="utf-8") as output:
    json.dump(expected, output)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, "source.blend"))
"#;

const BLENDER_REOPEN: &str = r#"
import bpy
import json
import os
import sys

root, output_path = sys.argv[sys.argv.index("--") + 1:sys.argv.index("--") + 3]
constraint_types = json.load(open(os.path.join(root, "constraint_types.json"), encoding="utf-8"))
scene = bpy.context.scene
rig = bpy.data.objects["PoseConstraintOwners"]
clip = scene.active_clip

def stable(value):
    if isinstance(value, bpy.types.ID):
        return value.name
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return [stable(item) for item in value]
    except TypeError:
        return str(value)

def settings(constraint):
    ignored = {"rna_type", "is_override_data", "is_valid", "error_location",
               "error_rotation", "active", "show_expanded"}
    result = {}
    for prop in constraint.bl_rna.properties:
        name = prop.identifier
        if name in ignored or prop.type in ("POINTER", "COLLECTION"):
            continue
        try:
            result[name] = stable(getattr(constraint, name))
        except (AttributeError, TypeError, ValueError):
            continue
    for name in ("target", "space_object", "camera", "depth_object", "pole_target",
                 "action", "clip", "cache_file"):
        if hasattr(constraint, name):
            result["ref_" + name] = stable(getattr(constraint, name))
    if hasattr(constraint, "targets"):
        result["armature_targets"] = [
            {"target": stable(item.target), "subtarget": item.subtarget,
             "weight": float(item.weight)} for item in constraint.targets]
    if hasattr(constraint, "cache_file") and constraint.cache_file:
        cache_filepath = bpy.path.abspath(constraint.cache_file.filepath)
        result["cache_filepath"] = os.path.basename(cache_filepath)
        result["cache_file_exists"] = os.path.isfile(cache_filepath)
        result["cache_object_path"] = constraint.object_path
        result["cache_file_settings"] = {
            key: stable(getattr(constraint.cache_file, key))
            for key in ("is_sequence", "override_frame", "frame", "frame_offset",
                        "forward_axis", "up_axis", "scale", "velocity_name",
                        "velocity_unit")
        }
    return result

def matrix_values(matrix):
    return [float(matrix[row][column]) for column in range(4) for row in range(4)]
def camera_values(camera):
    return {
        "focal_length": float(camera.focal_length),
        "sensor_width": float(camera.sensor_width),
        "principal_point": [float(value) for value in camera.principal_point],
        "units": camera.units,
        "pixel_aspect": float(camera.pixel_aspect),
        "distortion_model": camera.distortion_model,
        "k1": float(camera.k1), "k2": float(camera.k2), "k3": float(camera.k3),
        "division_k1": float(camera.division_k1),
        "division_k2": float(camera.division_k2),
        "nuke_k1": float(camera.nuke_k1), "nuke_k2": float(camera.nuke_k2),
        "nuke_p1": float(camera.nuke_p1), "nuke_p2": float(camera.nuke_p2),
        "brown_k1": float(camera.brown_k1), "brown_k2": float(camera.brown_k2),
        "brown_k3": float(camera.brown_k3), "brown_k4": float(camera.brown_k4),
        "brown_p1": float(camera.brown_p1), "brown_p2": float(camera.brown_p2),
    }

def track_bundles(tracking_object):
    return [
        {"name": track.name, "has_bundle": bool(track.has_bundle),
         "bundle": [float(value) for value in track.bundle] if track.has_bundle else None}
        for track in sorted(tracking_object.tracks, key=lambda track: track.name)
    ]

expected = {"frames": {}, "settings": {}}
for frame in (1, 2, 3):
    scene.frame_set(frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    values = {"objects": {}, "bones": {}}
    evaluated_rig = rig.evaluated_get(depsgraph)
    for kind in constraint_types:
        stem = kind.upper()
        obj = bpy.data.objects["Object_" + stem]
        values["objects"][obj.name] = matrix_values(obj.evaluated_get(depsgraph).matrix_world)
        bone_name = "Tip_" + stem
        values["bones"][bone_name] = matrix_values(
            evaluated_rig.matrix_world @ evaluated_rig.pose.bones[bone_name].matrix)
    expected["frames"][str(frame)] = values
for kind in constraint_types:
    stem = kind.upper()
    object_owner = bpy.data.objects["Object_" + stem]
    object_constraint = object_owner.constraints["Object_" + stem]
    expected["settings"]["Object_" + stem] = settings(object_constraint)
    pose_bone = rig.pose.bones["Tip_" + stem]
    pose_constraint = pose_bone.constraints["Pose_" + stem]
    expected["settings"]["Pose_" + stem] = settings(pose_constraint)
expected["clip"] = {
    "name": clip.name if clip else None,
    "filepath_exists": bool(clip and os.path.isfile(bpy.path.abspath(clip.filepath))),
    "filepath": bpy.path.abspath(clip.filepath) if clip else None,
    "camera_tracks": sorted(track.name for track in clip.tracking.objects["Camera"].tracks) if clip else [],
    "object_tracks": sorted(track.name for track in clip.tracking.objects["Moving"].tracks) if clip else [],
    "reconstruction_frames": sorted(int(camera.frame) for camera in clip.tracking.reconstruction.cameras) if clip else [],
    "camera_reconstruction_valid": bool(clip and clip.tracking.reconstruction.is_valid),
    "camera_reconstruction_average_error": (
        float(clip.tracking.reconstruction.average_error) if clip else None),
    "moving_reconstruction_valid": bool(
        clip and clip.tracking.objects["Moving"].reconstruction.is_valid),
    "moving_reconstruction_average_error": (
        float(clip.tracking.objects["Moving"].reconstruction.average_error)
        if clip else None),
    "camera": camera_values(clip.tracking.camera) if clip else None,
    "camera_track_bundles": (
        track_bundles(clip.tracking.objects["Camera"]) if clip else []),
    "moving_track_bundles": (
        track_bundles(clip.tracking.objects["Moving"]) if clip else []),
}
expected["reconstruction_cameras"] = [
    {
        "frame": int(sample.frame),
        "matrix": [[float(sample.matrix[row][column]) for column in range(4)]
                   for row in range(3)],
        "average_error": float(sample.average_error),
    }
    for sample in (clip.tracking.reconstruction.cameras if clip else [])
]
expected["moving_reconstruction"] = [
    {"frame": float(sample.frame), "matrix": matrix_values(sample.matrix),
     "average_error": float(sample.average_error)}
    for sample in (
        clip.tracking.objects["Moving"].reconstruction.cameras if clip else [])
]
expected["scene_camera"] = scene.camera.name if scene.camera else None
with open(output_path, "w", encoding="utf-8") as output:
    json.dump(expected, output)
"#;

const BLENDER_IK_CHAIN_FIXTURE: &str = r#"
import bpy
import json
import os
import sys

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
fixture = json.load(open(os.path.join(root, "ik_chain.json"), encoding="utf-8"))
chain_count = fixture["chain_count"]
swing_ellipse_limit = fixture["swing_ellipse_limit"]
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 3
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

prefix = "IK%d" % chain_count
rig_data = bpy.data.armatures.new(prefix + "RigData")
rig = bpy.data.objects.new(prefix + "Rig", rig_data)
scene.collection.objects.link(rig)
rig.select_set(True)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="EDIT")
points = (
    [(-0.6,0,0),(-0.6,0.8,0),(0.15,1.25,0)]
    if chain_count == 2
    else [(-0.9,0,0),(-0.25,0.45,0),(0.25,1.05,0),(1.0,1.35,0)]
)
bone_names = []
previous = None
for index in range(chain_count):
    bone = rig_data.edit_bones.new("%s_Bone%d" % (prefix, index))
    bone.head = points[index]
    bone.tail = points[index + 1]
    if previous:
        bone.parent = previous
        bone.use_connect = True
    previous = bone
    bone_names.append(bone.name)
bpy.ops.object.mode_set(mode="OBJECT")
rig.select_set(False)

target = bpy.data.objects.new(prefix + "Target", None)
pole = bpy.data.objects.new(prefix + "Pole", None)
scene.collection.objects.link(target)
scene.collection.objects.link(pole)
pole.location = (0.2,0.4,1.5)
for frame, location in (
    (1,(0.25,1.4,0.45)),
    (2,(0.55,1.8,0.7)),
    (3,(0.1,1.6,0.35)),
):
    target.location = location
    target.keyframe_insert(data_path="location", frame=frame)

for index, name in enumerate(bone_names):
    pose_bone = rig.pose.bones[name]
    if chain_count == 2:
        pose_bone.lock_ik_x = index == 0
        pose_bone.lock_ik_y = index == 1
        pose_bone.lock_ik_z = False
        pose_bone.use_ik_limit_x = index != chain_count - 1 or swing_ellipse_limit
        pose_bone.use_ik_limit_y = True
        pose_bone.use_ik_limit_z = index != chain_count - 1 or swing_ellipse_limit
        pose_bone.ik_min_x = -0.7 + 0.04 * index
        pose_bone.ik_max_x = 0.8 - 0.03 * index
        pose_bone.ik_min_y = -0.6 + 0.02 * index
        pose_bone.ik_max_y = 0.7 - 0.02 * index
        pose_bone.ik_min_z = -0.5 + 0.03 * index
        pose_bone.ik_max_z = 0.65 - 0.02 * index
        pose_bone.ik_stiffness_x = 0.1 + 0.05 * index
        pose_bone.ik_stiffness_y = 0.2 + 0.04 * index
        pose_bone.ik_stiffness_z = 0.3 + 0.03 * index
        pose_bone.ik_stretch = 0.15 + 0.12 * index
    else:
        pose_bone.lock_ik_x = index == 0
        pose_bone.lock_ik_y = index == 2
        pose_bone.lock_ik_z = index == 1
        pose_bone.use_ik_limit_x = index != chain_count - 1 or swing_ellipse_limit
        pose_bone.use_ik_limit_y = True
        pose_bone.use_ik_limit_z = index != chain_count - 1 or swing_ellipse_limit
        pose_bone.ik_min_x = -0.7 + 0.04 * index
        pose_bone.ik_max_x = 0.8 - 0.03 * index
        pose_bone.ik_min_y = -0.6 + 0.02 * index
        pose_bone.ik_max_y = 0.7 - 0.02 * index
        pose_bone.ik_min_z = -0.5 + 0.03 * index
        pose_bone.ik_max_z = 0.65 - 0.02 * index
        pose_bone.ik_stiffness_x = 0.1 + 0.05 * index
        pose_bone.ik_stiffness_y = 0.2 + 0.04 * index
        pose_bone.ik_stiffness_z = 0.3 + 0.03 * index
        pose_bone.ik_stretch = 0.15 + 0.12 * index
constraint = rig.pose.bones[bone_names[-1]].constraints.new("IK")
constraint.name = prefix + "Constraint"
constraint.influence = 0.65 if chain_count == 2 else 1.0
constraint.target = target
constraint.pole_target = pole
constraint.chain_count = chain_count
constraint.pole_angle = 0.31
constraint.use_stretch = True
constraint.use_location = True
constraint.use_rotation = True
constraint.weight = 0.7
constraint.orient_weight = 0.35

def matrix_values(matrix):
    return [float(matrix[row][column]) for column in range(4) for row in range(4)]

def pose_settings(pose_bone):
    return {
        "lock_ik": [bool(pose_bone.lock_ik_x), bool(pose_bone.lock_ik_y),
                    bool(pose_bone.lock_ik_z)],
        "use_ik_limit": [bool(pose_bone.use_ik_limit_x), bool(pose_bone.use_ik_limit_y),
                         bool(pose_bone.use_ik_limit_z)],
        "ik_min": [float(pose_bone.ik_min_x), float(pose_bone.ik_min_y),
                   float(pose_bone.ik_min_z)],
        "ik_max": [float(pose_bone.ik_max_x), float(pose_bone.ik_max_y),
                   float(pose_bone.ik_max_z)],
        "ik_stiffness": [float(pose_bone.ik_stiffness_x),
                         float(pose_bone.ik_stiffness_y),
                         float(pose_bone.ik_stiffness_z)],
        "ik_stretch": float(pose_bone.ik_stretch),
    }

def collect(output_path):
    result = {"frames": {}, "rest_frames": {}, "pose_bones": {}, "constraint": {}}
    for frame in (1,2,3):
        scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        depsgraph.update()
        evaluated_rig = rig.evaluated_get(depsgraph)
        result["frames"][str(frame)] = {
            name: matrix_values(evaluated_rig.matrix_world
                                @ evaluated_rig.pose.bones[name].matrix)
            for name in bone_names
        }
    constraint.mute = True
    for frame in (1,2,3):
        scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        depsgraph.update()
        evaluated_rig = rig.evaluated_get(depsgraph)
        result["rest_frames"][str(frame)] = {
            name: matrix_values(evaluated_rig.matrix_world
                                @ evaluated_rig.pose.bones[name].matrix)
            for name in bone_names
        }
    constraint.mute = False
    bpy.context.view_layer.update()
    for name in bone_names:
        result["pose_bones"][name] = pose_settings(rig.pose.bones[name])
    result["constraint"] = {
        "target": constraint.target.name,
        "pole_target": constraint.pole_target.name if constraint.pole_target else None,
        "chain_count": int(constraint.chain_count),
        "pole_angle": float(constraint.pole_angle),
        "influence": float(constraint.influence),
        "use_stretch": bool(constraint.use_stretch),
        "use_location": bool(constraint.use_location),
        "use_rotation": bool(constraint.use_rotation),
        "weight": float(constraint.weight),
        "orient_weight": float(constraint.orient_weight),
    }
    with open(output_path, "w", encoding="utf-8") as output:
        json.dump(result, output)

collect(os.path.join(root, "ik_before.json"))
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, "ik_source.blend"))

"#;

const BLENDER_IK_CHAIN_REOPEN: &str = r#"
import bpy
import json
import os
import sys

root, output_path = sys.argv[sys.argv.index("--") + 1:sys.argv.index("--") + 3]
chain_count = json.load(open(os.path.join(root, "ik_chain.json"), encoding="utf-8"))["chain_count"]
scene = bpy.context.scene
prefix = "IK%d" % chain_count
rig = bpy.data.objects[prefix + "Rig"]
bone_names = ["%s_Bone%d" % (prefix, index) for index in range(chain_count)]
constraint = rig.pose.bones[bone_names[-1]].constraints[prefix + "Constraint"]

def matrix_values(matrix):
    return [float(matrix[row][column]) for column in range(4) for row in range(4)]

def pose_settings(pose_bone):
    return {
        "lock_ik": [bool(pose_bone.lock_ik_x), bool(pose_bone.lock_ik_y),
                    bool(pose_bone.lock_ik_z)],
        "use_ik_limit": [bool(pose_bone.use_ik_limit_x), bool(pose_bone.use_ik_limit_y),
                         bool(pose_bone.use_ik_limit_z)],
        "ik_min": [float(pose_bone.ik_min_x), float(pose_bone.ik_min_y),
                   float(pose_bone.ik_min_z)],
        "ik_max": [float(pose_bone.ik_max_x), float(pose_bone.ik_max_y),
                   float(pose_bone.ik_max_z)],
        "ik_stiffness": [float(pose_bone.ik_stiffness_x),
                         float(pose_bone.ik_stiffness_y),
                         float(pose_bone.ik_stiffness_z)],
        "ik_stretch": float(pose_bone.ik_stretch),
    }

result = {"frames": {}, "pose_bones": {}, "constraint": {}}
for frame in (1,2,3):
    scene.frame_set(frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    evaluated_rig = rig.evaluated_get(depsgraph)
    result["frames"][str(frame)] = {
        name: matrix_values(evaluated_rig.matrix_world @ evaluated_rig.pose.bones[name].matrix)
        for name in bone_names
    }
for name in bone_names:
    result["pose_bones"][name] = pose_settings(rig.pose.bones[name])
result["constraint"] = {
    "target": constraint.target.name,
    "pole_target": constraint.pole_target.name if constraint.pole_target else None,
    "chain_count": int(constraint.chain_count),
    "pole_angle": float(constraint.pole_angle),
    "influence": float(constraint.influence),
    "use_stretch": bool(constraint.use_stretch),
    "use_location": bool(constraint.use_location),
    "use_rotation": bool(constraint.use_rotation),
    "weight": float(constraint.weight),
    "orient_weight": float(constraint.orient_weight),
}
with open(output_path, "w", encoding="utf-8") as output:
    json.dump(result, output)
"#;

const BLENDER_TARGET_PROJECT_FIXTURE: &str = r#"
import bpy
import json
import os
import sys

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
scene = bpy.context.scene
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

mesh = bpy.data.meshes.new("BoundarySurfaceData")
mesh.from_pydata(
    [(0,0,0),(1,0,0),(1.4,1,0),(0,1,0)], [],
    [(0,1,2),(0,2,3)])
mesh.update()
target = bpy.data.objects.new("BoundarySurface", mesh)
scene.collection.objects.link(target)
owner = bpy.data.objects.new("BoundaryOwner", None)
scene.collection.objects.link(owner)
# The source is above the plane but outside the normal triangle; its nearest
# projection is the open boundary edge from vertex 0 to vertex 1.
owner.location = (0.4,-0.25,0.6)
constraint = owner.constraints.new("SHRINKWRAP")
constraint.name = "BoundaryShrinkwrap"
constraint.target = target
constraint.shrinkwrap_type = "TARGET_PROJECT"
constraint.wrap_mode = "ON_SURFACE"
constraint.distance = 0.0

def matrix_values(matrix):
    return [float(matrix[row][column]) for column in range(4) for row in range(4)]

def boundary_edges(mesh):
    counts = {}
    for polygon in mesh.polygons:
        vertices = list(polygon.vertices)
        for index, first in enumerate(vertices):
            second = vertices[(index + 1) % len(vertices)]
            edge = tuple(sorted((int(first), int(second))))
            counts[edge] = counts.get(edge, 0) + 1
    return [list(edge) for edge, count in sorted(counts.items()) if count == 1]

def collect():
    scene.frame_set(1)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    evaluated_owner = owner.evaluated_get(depsgraph)
    return {
        "input_location": [float(value) for value in owner.location],
        "matrix": matrix_values(evaluated_owner.matrix_world),
        "constraint": {
            "target": constraint.target.name,
            "shrinkwrap_type": constraint.shrinkwrap_type,
            "wrap_mode": constraint.wrap_mode,
            "distance": float(constraint.distance),
        },
        "boundary_edges": boundary_edges(target.data),
    }

with open(os.path.join(root, "target_project_before.json"), "w", encoding="utf-8") as output:
    json.dump(collect(), output)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, "target_project_source.blend"))
"#;

const BLENDER_TARGET_PROJECT_REOPEN: &str = r#"
import bpy
import json
import sys

output_path = sys.argv[sys.argv.index("--") + 2]
scene = bpy.context.scene
owner = bpy.data.objects["BoundaryOwner"]
target = bpy.data.objects["BoundarySurface"]
constraint = owner.constraints["BoundaryShrinkwrap"]
def matrix_values(matrix):
    return [float(matrix[row][column]) for column in range(4) for row in range(4)]

def boundary_edges(mesh):
    counts = {}
    for polygon in mesh.polygons:
        vertices = list(polygon.vertices)
        for index, first in enumerate(vertices):
            second = vertices[(index + 1) % len(vertices)]
            edge = tuple(sorted((int(first), int(second))))
            counts[edge] = counts.get(edge, 0) + 1
    return [list(edge) for edge, count in sorted(counts.items()) if count == 1]


scene.frame_set(1)
depsgraph = bpy.context.evaluated_depsgraph_get()
depsgraph.update()
evaluated_owner = owner.evaluated_get(depsgraph)
result = {
    "input_location": [float(value) for value in owner.location],
    "matrix": matrix_values(evaluated_owner.matrix_world),
    "constraint": {
        "target": constraint.target.name,
        "shrinkwrap_type": constraint.shrinkwrap_type,
        "wrap_mode": constraint.wrap_mode,
        "distance": float(constraint.distance),
    },
    "boundary_edges": boundary_edges(target.data),
}
with open(output_path, "w", encoding="utf-8") as output:
    json.dump(result, output)
"#;

fn matrix_max_error(actual: &[Value], expected: &[Value]) -> TestResult<f64> {
    if actual.len() != 16 || expected.len() != 16 {
        return Err(format!(
            "matrix sizes differ: Potter={}, Blender={}",
            actual.len(),
            expected.len()
        )
        .into());
    }
    actual
        .iter()
        .zip(expected)
        .try_fold(0.0_f64, |maximum, (actual, expected)| {
            let actual = actual
                .as_f64()
                .ok_or("Potter matrix component is not numeric")?;
            let expected = expected
                .as_f64()
                .ok_or("Blender matrix component is not numeric")?;
            Ok(maximum.max((actual - expected).abs()))
        })
}

fn assert_blender_matrices(actual: &[Value], expected: &[Value], context: &str) {
    assert_eq!(actual.len(), 16, "{context}: Potter matrix length");
    assert_eq!(expected.len(), 16, "{context}: Blender matrix length");
    let actual_matrix = actual;
    let expected_matrix = expected;
    for (component, (actual, expected)) in actual_matrix.iter().zip(expected_matrix).enumerate() {
        let actual = actual.as_f64().unwrap();
        let expected = expected.as_f64().unwrap();
        let error = (actual - expected).abs();
        assert!(
            error <= 1.0e-5,
            "{context}: matrix component {component} differs by {error:e}; Potter={actual}, Blender={expected}; Potter matrix={actual_matrix:?}; Blender matrix={expected_matrix:?}"
        );
    }
}
fn assert_numeric_values(actual: &[Value], expected: &[Value], context: &str) -> TestResult {
    assert_eq!(actual.len(), expected.len(), "{context}: value count");
    for (component, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let actual = actual
            .as_f64()
            .ok_or("actual matrix component is not numeric")?;
        let expected = expected
            .as_f64()
            .ok_or("expected matrix component is not numeric")?;
        let error = (actual - expected).abs();
        assert!(
            error <= 1.0e-5,
            "{context}: matrix component {component} differs by {error:e}; actual={actual}, expected={expected}"
        );
    }
    Ok(())
}
fn flatten_matrix_rows(matrix: &Value, context: &str) -> TestResult<Vec<Value>> {
    let rows = matrix
        .as_array()
        .ok_or_else(|| format!("{context}: matrix rows are missing"))?;
    let mut values = Vec::new();
    for row in rows {
        values.extend(
            row.as_array()
                .ok_or_else(|| format!("{context}: matrix row is missing"))?
                .iter()
                .cloned(),
        );
    }
    Ok(values)
}
fn assert_curve_point_sets(
    actual: &potter::geom::Mesh,
    expected: &Value,
    context: &str,
) -> TestResult {
    let expected_values = expected
        .as_array()
        .ok_or_else(|| format!("{context}: Blender curve points are missing"))?;
    let mut expected_points = Vec::with_capacity(expected_values.len());
    for point in expected_values {
        let coordinates = point
            .as_array()
            .filter(|coordinates| coordinates.len() == 3)
            .ok_or_else(|| format!("{context}: Blender curve point is not 3D"))?;
        expected_points.push(DVec3::new(
            coordinates[0]
                .as_f64()
                .ok_or_else(|| format!("{context}: Blender curve X is invalid"))?,
            coordinates[1]
                .as_f64()
                .ok_or_else(|| format!("{context}: Blender curve Y is invalid"))?,
            coordinates[2]
                .as_f64()
                .ok_or_else(|| format!("{context}: Blender curve Z is invalid"))?,
        ));
    }
    let actual_points: Vec<_> = actual.vertices.iter().map(|vertex| vertex.co).collect();
    if actual_points.is_empty() || expected_points.is_empty() {
        return Err(format!("{context}: curve point set is empty").into());
    }
    let mut maximum_error = 0.0_f64;
    let mut potter_index = 0;
    let mut blender_index = 0;
    for (index, point) in actual_points.iter().enumerate() {
        let (nearest_index, distance) = expected_points
            .iter()
            .enumerate()
            .map(|(index, candidate)| (index, point.distance(*candidate)))
            .min_by(|left, right| left.1.total_cmp(&right.1))
            .ok_or("Blender curve point set is empty")?;
        if distance > maximum_error {
            maximum_error = distance;
            potter_index = index;
            blender_index = nearest_index;
        }
    }
    for (index, point) in expected_points.iter().enumerate() {
        let (nearest_index, distance) = actual_points
            .iter()
            .enumerate()
            .map(|(index, candidate)| (index, point.distance(*candidate)))
            .min_by(|left, right| left.1.total_cmp(&right.1))
            .ok_or("Potter curve point set is empty")?;
        if distance > maximum_error {
            maximum_error = distance;
            potter_index = nearest_index;
            blender_index = index;
        }
    }
    assert!(
        maximum_error <= 1.0e-5,
        "{context}: max point-set distance {maximum_error:e} at Potter vertex {potter_index} {:?} vs Blender vertex {blender_index} {:?} (counts Potter={}, Blender={})",
        actual_points[potter_index],
        expected_points[blender_index],
        actual_points.len(),
        expected_points.len()
    );
    Ok(())
}

fn assert_matrix_changed(first: &[Value], last: &[Value], context: &str) {
    let maximum_delta = first
        .iter()
        .zip(last)
        .map(|(first, last)| (first.as_f64().unwrap() - last.as_f64().unwrap()).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        maximum_delta > 1.0e-5,
        "{context} did not respond to the animated fixture: {maximum_delta:e}"
    );
}

fn assert_constraint_options(
    doc: &SceneDoc,
    constraint: &potter::model::Constraint,
    settings: &Value,
    context: &str,
) -> TestResult {
    if let Some(head_tail) = settings["head_tail"].as_f64() {
        let actual = constraint.params["head_tail"]
            .as_f64()
            .ok_or_else(|| format!("{context} head_tail is missing"))?;
        assert!(
            (actual - head_tail).abs() < 1.0e-6,
            "{context} head_tail: {actual} != {head_tail}"
        );
    }
    if let Some(mix_mode) = settings["mix_mode"].as_str() {
        assert_eq!(
            constraint.params["mix_mode"].as_str(),
            Some(mix_mode),
            "{context} mix mode"
        );
    }
    if let Some(target_name) = settings["ref_target"].as_str() {
        let (target_id, target) = node_named(doc, target_name)?;
        assert_eq!(
            constraint.target.as_ref(),
            Some(target_id),
            "{context} target"
        );
        if settings["subtarget"] == "TargetBone" {
            let data_id = target
                .data
                .as_ref()
                .ok_or("bone target has no data block")?;
            let data = doc
                .data_blocks
                .get(data_id)
                .ok_or("bone target data block missing")?;
            let armature = data
                .armature
                .as_ref()
                .ok_or("bone target armature missing")?;
            let (bone_id, _) = armature
                .bones
                .iter()
                .find(|(_, bone)| bone.name == "TargetBone")
                .ok_or("constraint target bone is missing")?;
            assert_eq!(
                constraint.subtarget.as_ref(),
                Some(bone_id),
                "{context} subtarget"
            );
        }
    }
    for (settings_key, parameter_key) in [
        ("ref_camera", "camera"),
        ("ref_space_object", "space_object"),
    ] {
        if let Some(name) = settings[settings_key].as_str() {
            let (reference_id, _) = node_named(doc, name)?;
            assert_eq!(
                constraint.params.get(parameter_key).and_then(Value::as_str),
                Some(reference_id.as_str()),
                "{context} {parameter_key} reference"
            );
        }
    }
    if let Some(name) = settings["ref_clip"].as_str() {
        let (clip_id, _) = doc
            .movie_clips
            .iter()
            .find(|(_, clip)| clip.name == name)
            .ok_or_else(|| format!("{context} MovieClip {name} is missing"))?;
        assert_eq!(
            constraint.params.get("clip").and_then(Value::as_str),
            Some(clip_id.as_str()),
            "{context} MovieClip reference"
        );
    }
    for flag in [
        "use_deform_preserve_volume",
        "use_bone_envelopes",
        "use_current_location",
    ] {
        if !settings[flag].is_null() {
            assert_eq!(
                constraint.params.get(flag),
                Some(&settings[flag]),
                "{context} {flag}"
            );
        }
    }
    if let Some(expected_targets) = settings["armature_targets"].as_array() {
        let actual_targets = constraint.params["targets"]
            .as_array()
            .ok_or_else(|| format!("{context} armature targets are missing"))?;
        assert_eq!(
            actual_targets.len(),
            expected_targets.len(),
            "{context} armature target count"
        );
        for (index, (expected_target, actual_target)) in
            expected_targets.iter().zip(actual_targets).enumerate()
        {
            let target_name = expected_target["target"]
                .as_str()
                .ok_or_else(|| format!("{context} armature target {index} has no object"))?;
            let (target_id, _) = node_named(doc, target_name)?;
            assert_eq!(
                actual_target["target"].as_str(),
                Some(target_id.as_str()),
                "{context} armature target {index} object"
            );
            assert_eq!(
                actual_target["subtarget"], expected_target["subtarget"],
                "{context} armature target {index} subtarget"
            );
            let actual_weight = actual_target["weight"]
                .as_f64()
                .ok_or_else(|| format!("{context} armature target {index} weight is missing"))?;
            let expected_weight = expected_target["weight"].as_f64().ok_or_else(|| {
                format!("{context} Blender armature target {index} weight is missing")
            })?;
            assert!(
                (actual_weight - expected_weight).abs() <= 1.0e-6,
                "{context} armature target {index} weight: {actual_weight} != {expected_weight}"
            );
        }
    }
    for key in [
        "attribute_name",
        "data_type",
        "domain",
        "sample_index",
        "apply_target_transform",
        "mix_loc",
        "mix_rot",
        "mix_scl",
        "use_fixed_location",
        "offset",
        "offset_factor",
        "use_curve_follow",
        "forward_axis",
        "up_axis",
    ] {
        if !settings[key].is_null() {
            assert_eq!(
                constraint.params.get(key),
                Some(&settings[key]),
                "{context} {key}"
            );
        }
    }
    Ok(())
}

fn node_named<'a>(doc: &'a SceneDoc, name: &str) -> TestResult<(&'a Id, &'a potter::model::Node)> {
    doc.nodes
        .iter()
        .find(|(_, node)| node.name == name)
        .ok_or_else(|| format!("imported Blender object {name} is missing").into())
}

fn assert_imported_evaluation(
    doc: &SceneDoc,
    project: &Path,
    expected: &Value,
    target_kind: &str,
    evaluation_failures: &mut Vec<String>,
) -> TestResult<BTreeMap<String, f64>> {
    let mut object_ids = HashMap::new();
    let mut bone_ids = HashMap::new();
    for kind in CONSTRAINT_TYPES
        .iter()
        .filter(|candidate| **candidate == target_kind)
    {
        let stem = kind.to_uppercase();
        let object_name = format!("Object_{stem}");
        let (object_id, object) = node_named(doc, &object_name)?;
        assert_eq!(
            object.constraints.len(),
            1,
            "{object_name} constraint count"
        );
        assert_eq!(
            serde_json::to_value(object.constraints[0].constraint_type).unwrap(),
            json!(*kind),
            "{object_name} type"
        );
        let object_settings = &expected["settings"][&object_name];
        let expected_object_influence = object_settings["influence"]
            .as_f64()
            .ok_or_else(|| format!("{object_name} Blender influence is missing"))?;
        assert!(
            (object.constraints[0].influence - expected_object_influence).abs() <= 1.0e-6,
            "{object_name} influence"
        );
        let owner_space = object_settings["owner_space"]
            .as_str()
            .ok_or_else(|| format!("{object_name} Blender owner space is missing"))?;
        let target_space = object_settings["target_space"]
            .as_str()
            .ok_or_else(|| format!("{object_name} Blender target space is missing"))?;
        assert_eq!(
            object.constraints[0]
                .params
                .get("owner_space")
                .and_then(Value::as_str),
            Some(owner_space),
            "{object_name} owner space: {:?}",
            object.constraints[0].params
        );
        assert_eq!(
            object.constraints[0]
                .params
                .get("target_space")
                .and_then(Value::as_str),
            Some(target_space),
            "{object_name} target space: {:?}",
            object.constraints[0].params
        );
        assert_constraint_options(doc, &object.constraints[0], object_settings, &object_name)?;
        object_ids.insert(object_name, object_id.clone());

        let bone_name = format!("Tip_{stem}");
        let (rig_id, rig) = node_named(doc, "PoseConstraintOwners")?;
        assert_eq!(
            rig.constraints.len(),
            1,
            "pose constraints stored on armature"
        );
        let data_id = rig
            .data
            .as_ref()
            .ok_or("imported pose rig has no armature data")?;
        let data = doc
            .data_blocks
            .get(data_id)
            .ok_or("armature data block missing")?;
        let armature = data.armature.as_ref().ok_or("armature model missing")?;
        let (bone_id, _) = armature
            .bones
            .iter()
            .find(|(_, bone)| bone.name == bone_name)
            .ok_or_else(|| format!("imported pose bone {bone_name} is missing"))?;
        let owner_constraint = rig
            .constraints
            .iter()
            .find(|constraint| constraint.owner_bone.as_ref() == Some(bone_id))
            .ok_or_else(|| format!("constraint on pose bone {bone_name} is missing"))?;
        let pose_settings = &expected["settings"][&format!("Pose_{stem}")];
        let expected_pose_influence = pose_settings["influence"]
            .as_f64()
            .ok_or_else(|| format!("{bone_name} Blender influence is missing"))?;
        assert!(
            (owner_constraint.influence - expected_pose_influence).abs() <= 1.0e-6,
            "{bone_name} influence"
        );
        let owner_space = pose_settings["owner_space"]
            .as_str()
            .ok_or_else(|| format!("{bone_name} Blender owner space is missing"))?;
        let target_space = pose_settings["target_space"]
            .as_str()
            .ok_or_else(|| format!("{bone_name} Blender target space is missing"))?;
        assert_eq!(
            owner_constraint
                .params
                .get("owner_space")
                .and_then(Value::as_str),
            Some(owner_space),
            "{bone_name} owner space: {:?}",
            owner_constraint.params
        );
        assert_eq!(
            owner_constraint
                .params
                .get("target_space")
                .and_then(Value::as_str),
            Some(target_space),
            "{bone_name} target space: {:?}",
            owner_constraint.params
        );
        assert_constraint_options(doc, owner_constraint, pose_settings, &bone_name)?;
        bone_ids.insert((rig_id.clone(), bone_id.clone()), bone_name);
    }

    let before_frames = expected["frames"]
        .as_object()
        .ok_or("Blender frame matrices missing")?;
    let mut max_errors = BTreeMap::new();
    for kind in CONSTRAINT_TYPES.iter().filter(|kind| **kind == target_kind) {
        let stem = kind.to_uppercase();
        let mut type_max_error: f64 = 0.0;
        for (owner_kind, pose_owner) in [("object", false), ("pose bone", true)] {
            let mut isolated = doc.clone();
            let expected_type = json!(*kind);
            for node in isolated.nodes.values_mut() {
                node.constraints.retain(|constraint| {
                    constraint.owner_bone.is_some() == pose_owner
                        && serde_json::to_value(constraint.constraint_type)
                            .is_ok_and(|actual_type| actual_type == expected_type)
                });
            }
            for frame in FRAMES {
                let snapshot = match Snapshot::evaluate_with_cache(
                    &isolated,
                    &EvaluationContext {
                        frame: Some(f64::from(frame)),
                        ..EvaluationContext::default()
                    },
                    Some(project),
                ) {
                    Ok(snapshot) => snapshot,
                    Err(error) => {
                        evaluation_failures
                            .push(format!("{kind} {owner_kind} at frame {frame}: {error}"));
                        break;
                    }
                };
                let frame_key = frame.to_string();
                let frame_values = &before_frames[&frame_key];
                if pose_owner {
                    let name = format!("Tip_{stem}");
                    let ((rig_id, bone_id), _) = bone_ids
                        .iter()
                        .find(|(_, bone_name)| **bone_name == name)
                        .ok_or_else(|| format!("Potter evaluation omitted pose bone {name}"))?;
                    let actual = snapshot
                        .bone_matrices
                        .get(rig_id)
                        .and_then(|bones| bones.get(bone_id))
                        .ok_or_else(|| format!("Potter evaluation omitted pose bone {name}"))?;
                    let actual = actual.map(|value| json!(value));
                    let expected = frame_values["bones"][&name]
                        .as_array()
                        .ok_or("Blender bone matrix missing")?;
                    let error = matrix_max_error(&actual, expected)?;
                    type_max_error = type_max_error.max(error);
                    if error > 1.0e-5 {
                        evaluation_failures.push(format!(
                            "{kind} pose bone {name} at frame {frame}: max error {error:e}; Potter={actual:?}; Blender={expected:?}"
                        ));
                    }
                } else {
                    let name = format!("Object_{stem}");
                    let id = object_ids
                        .get(&name)
                        .ok_or_else(|| format!("Potter evaluation omitted object {name}"))?;
                    let actual = snapshot
                        .nodes
                        .get(id)
                        .ok_or_else(|| format!("Potter evaluation omitted object {name}"))?;
                    let actual = actual.world_matrix.map(|value| json!(value));
                    let expected = frame_values["objects"][&name]
                        .as_array()
                        .ok_or("Blender object matrix missing")?;
                    let error = matrix_max_error(&actual, expected)?;
                    type_max_error = type_max_error.max(error);
                    if error > 1.0e-5 {
                        evaluation_failures.push(format!(
                            "{kind} object {name} at frame {frame}: max error {error:e}; Potter={actual:?}; Blender={expected:?}"
                        ));
                    }
                }
            }
        }
        eprintln!("constraint parity {kind}: import=ok eval_max_error={type_max_error:.3e}");
        max_errors.insert((*kind).to_owned(), type_max_error);
    }
    Ok(max_errors)
}

#[expect(
    clippy::too_many_lines,
    reason = "the integration test exercises one Blender constraint through evaluated export/reopen parity"
)]
fn round_trip_constraint(kind: &str) -> TestResult {
    let _guard = BLENDER_CONSTRAINT_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(blender) = blender() else {
        eprintln!("skipping Blender constraint roundtrip test: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    fs::write(
        root.join("constraint_types.json"),
        serde_json::to_vec(&[kind])?,
    )?;
    run_blender_script(
        &blender,
        "make_constraints.py",
        BLENDER_FIXTURE,
        root,
        &[],
        "Blender script make_constraints.py failed",
        run_guarded,
    )?;
    let before: Value = serde_json::from_slice(&fs::read(root.join("blender_before.json"))?)?;
    assert_eq!(before["clip"]["name"], "RoundtripClip");
    assert_eq!(
        before["clip"]["filepath_exists"], true,
        "source MovieClip filepath does not resolve"
    );
    assert_eq!(before["clip"]["camera_reconstruction_valid"], true);
    assert_eq!(before["clip"]["moving_reconstruction_valid"], true);
    assert!(
        !before["clip"]["camera_tracks"]
            .as_array()
            .unwrap()
            .is_empty(),
        "Blender camera tracks are missing"
    );
    assert!(
        !before["clip"]["reconstruction_frames"]
            .as_array()
            .unwrap()
            .is_empty(),
        "Blender reconstruction frames are missing"
    );
    let mix_mode_defaults = before["mix_mode_defaults"]
        .as_object()
        .ok_or("Blender mix mode defaults are missing")?;
    let has_mix_mode = before["settings"]
        .as_object()
        .ok_or("Blender constraint settings are missing")?
        .values()
        .any(|settings| settings["mix_mode"].is_string());
    assert_eq!(
        !mix_mode_defaults.is_empty(),
        has_mix_mode,
        "mix-mode fixture settings do not match the selected constraint type"
    );
    for (owner, default) in mix_mode_defaults {
        let default = default
            .as_str()
            .ok_or("constraint mix mode default is invalid")?;
        let actual = before["settings"][owner]["mix_mode"]
            .as_str()
            .ok_or_else(|| format!("{owner} mix mode is missing"))?;
        assert_ne!(actual, default, "{owner} mix mode remained at its default");
    }
    if kind == "follow_path" {
        for (frame, object_factor, eval_time) in [(1, 0.05, 0.2), (2, 0.5, 1.1), (3, 0.95, 2.0)] {
            let diagnostics = &before["follow_path_diagnostics"][frame.to_string()];
            assert_eq!(
                diagnostics["use_path"], true,
                "curve use_path at frame {frame}"
            );
            assert_eq!(diagnostics["path_duration"], 2.0, "curve path duration");
            assert!(
                (diagnostics["eval_time"].as_f64().unwrap() - eval_time).abs() <= 1.0e-6,
                "curve eval_time at frame {frame}: {:?}",
                diagnostics["eval_time"]
            );
            for (owner, fixed_location, expected_factor, expected_valid) in [
                ("object", false, object_factor, true),
                ("pose", true, 0.35, false),
            ] {
                let owner_diagnostics = &diagnostics[owner];
                assert_eq!(
                    owner_diagnostics["is_valid"], expected_valid,
                    "{owner} Follow Path validity at frame {frame}"
                );
                assert_eq!(
                    owner_diagnostics["use_fixed_location"], fixed_location,
                    "{owner} Follow Path location mode"
                );
                let path_factor = owner_diagnostics["path_factor"]
                    .as_f64()
                    .ok_or_else(|| format!("{owner} Follow Path factor is missing"))?;
                assert!(
                    (path_factor - expected_factor).abs() <= 1.0e-6,
                    "{owner} Follow Path factor at frame {frame}: {path_factor} != {expected_factor}"
                );
            }
        }
    }
    for kind in [kind] {
        let stem = kind.to_uppercase();
        for (owner_kind, name) in [
            ("objects", format!("Object_{stem}")),
            ("bones", format!("Tip_{stem}")),
        ] {
            assert_matrix_changed(
                before["frames"]["1"][owner_kind][name.as_str()]
                    .as_array()
                    .ok_or("frame-1 Blender matrix missing")?,
                before["frames"]["3"][owner_kind][name.as_str()]
                    .as_array()
                    .ok_or("frame-3 Blender matrix missing")?,
                &format!("{owner_kind} {name}"),
            );
        }
    }
    if kind == "camera_solver" {
        for owner in ["Object_CAMERA_SOLVER", "Pose_CAMERA_SOLVER"] {
            assert_eq!(
                before["settings"][owner]["use_active_clip"], false,
                "{owner} use_active_clip"
            );
            assert_eq!(
                before["settings"][owner]["ref_clip"], "RoundtripClip",
                "{owner} clip"
            );
        }
    }
    if kind == "transform_cache" {
        for owner in ["Object_TRANSFORM_CACHE", "Pose_TRANSFORM_CACHE"] {
            assert_eq!(
                before["settings"][owner]["ref_cache_file"], "constraint_cache.abc",
                "{owner} original CacheFile name"
            );
        }
    }

    let project = root.join("project");
    pot_json(&["init", project.to_str().unwrap()])?;
    let imported = pot_json(&[
        "import",
        project.to_str().unwrap(),
        "--file",
        root.join("source.blend").to_str().unwrap(),
        "--format",
        "blend",
        "--mode",
        "replace",
        "--base-revision",
        "0",
        "--blender",
        blender.to_str().unwrap(),
    ])?;
    assert!(
        imported["result"]["losses"].as_array().unwrap().is_empty(),
        "constraint fixture import reported losses: {imported}"
    );
    let doc: SceneDoc = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let mut evaluation_failures = Vec::new();
    let evaluation_max_errors =
        assert_imported_evaluation(&doc, &project, &before, kind, &mut evaluation_failures)?;
    if !evaluation_failures.is_empty() {
        eprintln!(
            "{kind} constraint frame mismatches:\n{}",
            evaluation_failures.join("\n")
        );
    }
    let (clip_id, _) = doc
        .movie_clips
        .iter()
        .find(|(_, clip)| clip.name == before["clip"]["name"].as_str().unwrap())
        .ok_or("imported MovieClip is missing")?;
    let active_scene = doc
        .scenes
        .get(&doc.active_scene)
        .ok_or("imported active scene is missing")?;
    let camera_name = before["scene_camera"]
        .as_str()
        .ok_or("Blender scene camera is missing")?;
    let (camera_id, _) = node_named(&doc, camera_name)?;
    assert_eq!(
        active_scene.camera.as_ref(),
        Some(camera_id),
        "imported Scene camera relationship"
    );
    assert_eq!(
        active_scene.active_clip.as_ref(),
        Some(clip_id),
        "imported Scene.active_clip"
    );
    let imported_clip = doc
        .movie_clips
        .get(clip_id)
        .ok_or("imported active MovieClip data is missing")?;

    let camera = &imported_clip.tracking.camera;
    let expected_camera = &before["clip"]["camera"];
    assert_eq!(camera.units, expected_camera["units"]);
    assert_eq!(camera.distortion_model, expected_camera["distortion_model"]);
    for (key, actual) in [
        ("focal_length", camera.focal_mm),
        ("sensor_width", camera.sensor_width_mm),
        ("pixel_aspect", camera.pixel_aspect),
        ("k1", camera.k1),
        ("k2", camera.k2),
        ("k3", camera.k3),
        ("division_k1", camera.division_k1),
        ("division_k2", camera.division_k2),
        ("nuke_k1", camera.nuke_k1),
        ("nuke_k2", camera.nuke_k2),
        ("nuke_p1", camera.nuke_p1),
        ("nuke_p2", camera.nuke_p2),
        ("brown_k1", camera.brown_k1),
        ("brown_k2", camera.brown_k2),
        ("brown_k3", camera.brown_k3),
        ("brown_k4", camera.brown_k4),
        ("brown_p1", camera.brown_p1),
        ("brown_p2", camera.brown_p2),
    ] {
        let expected = expected_camera[key]
            .as_f64()
            .ok_or_else(|| format!("Blender camera {key} is missing"))?;
        assert!(
            (actual - expected).abs() <= 1.0e-6,
            "imported MovieClip camera {key}: {actual} != {expected}"
        );
    }
    let actual_principal: Vec<_> = camera.principal.iter().map(|value| json!(value)).collect();
    let expected_principal = expected_camera["principal_point"]
        .as_array()
        .ok_or("Blender MovieClip principal point is missing")?;
    assert_numeric_values(
        &actual_principal,
        expected_principal,
        "imported MovieClip principal point",
    )?;
    assert_eq!(
        imported_clip.tracking.reconstruction.is_valid,
        before["clip"]["camera_reconstruction_valid"],
        "imported camera reconstruction validity"
    );
    let expected_camera_error = before["clip"]["camera_reconstruction_average_error"]
        .as_f64()
        .ok_or("Blender camera reconstruction error is missing")?;
    assert!(
        (imported_clip.tracking.reconstruction.average_error - expected_camera_error).abs()
            <= 1.0e-6,
        "imported camera reconstruction average error changed"
    );
    for (object_name, bundles_key) in [
        ("Camera", "camera_track_bundles"),
        ("Moving", "moving_track_bundles"),
    ] {
        let expected_bundles = before["clip"][bundles_key]
            .as_array()
            .ok_or_else(|| format!("Blender {object_name} track bundles are missing"))?;
        for bundle in expected_bundles {
            let track_name = bundle["name"]
                .as_str()
                .ok_or("Blender tracking bundle name is missing")?;
            let track_id = format!("{object_name}:{track_name}");
            let point = imported_clip
                .tracking
                .reconstruction
                .points
                .iter()
                .find(|point| point.track == track_id);
            if bundle["has_bundle"] == true {
                let point =
                    point.ok_or_else(|| format!("imported bundle {track_id} is missing"))?;
                let actual: Vec<_> = point.co.iter().map(|value| json!(value)).collect();
                let expected = bundle["bundle"]
                    .as_array()
                    .ok_or("Blender tracking bundle position is missing")?;
                assert_numeric_values(
                    &actual,
                    expected,
                    &format!("imported tracking bundle {track_id}"),
                )?;
            } else {
                assert!(
                    point.is_none(),
                    "imported track {track_id} unexpectedly has a bundle"
                );
            }
        }
    }
    let clip_source = imported_clip
        .source
        .as_ref()
        .ok_or("imported MovieClip source resource is missing")?;
    let clip_source = Id::new(clip_source.clone())
        .map_err(|error| format!("imported MovieClip source ID is invalid: {error}"))?;
    let clip_resource = doc
        .resources
        .get(&clip_source)
        .ok_or("imported MovieClip source resource record is missing")?;
    let clip_uri = clip_resource["uri"]
        .as_str()
        .filter(|uri| !uri.is_empty())
        .ok_or("imported MovieClip source URI is missing")?;
    let clip_uri = clip_uri.strip_prefix("file://").unwrap_or(clip_uri);
    let clip_path = Path::new(clip_uri);
    let clip_path = if clip_path.is_absolute() {
        clip_path.to_path_buf()
    } else {
        project.join(clip_path)
    };
    assert!(
        clip_path.is_file(),
        "imported MovieClip source is unavailable: {}",
        clip_path.display()
    );
    let mut imported_track_names: Vec<_> = imported_clip
        .tracking
        .tracks
        .iter()
        .map(|track| track.name.clone())
        .collect();
    let mut expected_track_names: Vec<_> = before["clip"]["camera_tracks"]
        .as_array()
        .ok_or("Blender camera track names are missing")?
        .iter()
        .chain(
            before["clip"]["object_tracks"]
                .as_array()
                .ok_or("Blender object track names are missing")?,
        )
        .map(|name| name.as_str().unwrap().to_owned())
        .collect();
    imported_track_names.sort();
    expected_track_names.sort();
    assert_eq!(
        imported_track_names, expected_track_names,
        "imported MovieClip tracks"
    );
    let imported_camera_frames: Vec<_> = imported_clip
        .tracking
        .reconstruction
        .cameras
        .iter()
        .map(|camera| camera.frame)
        .collect();
    let expected_camera_frames: Vec<_> = before["clip"]["reconstruction_frames"]
        .as_array()
        .ok_or("Blender reconstruction frames are missing")?
        .iter()
        .map(|frame| i32::try_from(frame.as_i64().unwrap()).unwrap())
        .collect();
    assert_eq!(
        imported_camera_frames, expected_camera_frames,
        "imported camera reconstruction frames"
    );
    let expected_camera_matrices = before["reconstruction_cameras"]
        .as_array()
        .ok_or("Blender reconstruction camera matrices are missing")?;
    assert_eq!(
        imported_clip.tracking.reconstruction.cameras.len(),
        expected_camera_matrices.len(),
        "imported reconstruction camera count"
    );
    for (camera, expected_camera) in imported_clip
        .tracking
        .reconstruction
        .cameras
        .iter()
        .zip(expected_camera_matrices)
    {
        assert_eq!(
            camera.frame,
            i32::try_from(expected_camera["frame"].as_i64().unwrap()).unwrap(),
            "imported reconstruction camera frame"
        );
        let actual_matrix: Vec<_> = camera
            .matrix
            .iter()
            .flatten()
            .map(|value| json!(value))
            .collect();
        let mut expected_matrix = Vec::new();
        for row in expected_camera["matrix"]
            .as_array()
            .ok_or("Blender reconstruction camera matrix rows are missing")?
        {
            expected_matrix.extend(
                row.as_array()
                    .ok_or("Blender reconstruction camera matrix row is missing")?
                    .iter()
                    .cloned(),
            );
        }
        assert_numeric_values(
            &actual_matrix,
            &expected_matrix,
            "imported MovieClip reconstruction camera",
        )?;
        let expected_error = expected_camera["average_error"]
            .as_f64()
            .ok_or("Blender reconstruction camera error is missing")?;
        assert!(
            (camera.average_error - expected_error).abs() <= 1.0e-6,
            "imported reconstruction camera error differs"
        );
    }
    let moving_object = imported_clip
        .tracking
        .objects
        .iter()
        .find(|object| object.name == "Moving")
        .ok_or("imported MovieClip moving tracking object is missing")?;
    assert_eq!(
        moving_object.reconstruction_is_valid, before["clip"]["moving_reconstruction_valid"],
        "imported moving-object reconstruction validity"
    );
    let expected_moving_error = before["clip"]["moving_reconstruction_average_error"]
        .as_f64()
        .ok_or("Blender moving reconstruction error is missing")?;
    assert!(
        (moving_object.reconstruction_average_error - expected_moving_error).abs() <= 1.0e-6,
        "imported moving-object reconstruction average error changed"
    );
    let track_names_by_id: HashMap<_, _> = imported_clip
        .tracking
        .tracks
        .iter()
        .map(|track| (track.id.as_str(), track.name.as_str()))
        .collect();
    let mut imported_object_track_names: Vec<_> = moving_object
        .tracks
        .iter()
        .map(|track_id| {
            track_names_by_id
                .get(track_id.as_str())
                .map(|name| (*name).to_owned())
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "moving tracking object references a missing track",
                    )
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut expected_object_track_names: Vec<_> = before["clip"]["object_tracks"]
        .as_array()
        .ok_or("Blender object track names are missing")?
        .iter()
        .map(|name| name.as_str().unwrap().to_owned())
        .collect();
    imported_object_track_names.sort();
    expected_object_track_names.sort();
    assert_eq!(
        imported_object_track_names, expected_object_track_names,
        "imported moving object tracks"
    );
    assert_eq!(
        moving_object
            .reconstruction
            .iter()
            .map(|pose| pose.frame)
            .collect::<Vec<_>>(),
        FRAMES.map(f64::from).to_vec(),
        "imported moving object reconstruction frames"
    );
    let expected_moving_reconstruction = before["moving_reconstruction"]
        .as_array()
        .ok_or("Blender moving-object reconstruction matrices are missing")?;
    assert_eq!(
        moving_object.reconstruction.len(),
        expected_moving_reconstruction.len(),
        "imported moving-object reconstruction count"
    );
    for (pose, expected_pose) in moving_object
        .reconstruction
        .iter()
        .zip(expected_moving_reconstruction)
    {
        let expected_frame = expected_pose["frame"]
            .as_f64()
            .ok_or("Blender moving-object reconstruction frame is missing")?;
        assert!(
            (pose.frame - expected_frame).abs() <= 1.0e-6,
            "imported moving-object reconstruction frame differs: {} != {expected_frame}",
            pose.frame
        );
        let actual_matrix: Vec<_> = pose.matrix.iter().map(|value| json!(value)).collect();
        let expected_matrix = expected_pose["matrix"]
            .as_array()
            .ok_or("Blender moving-object reconstruction matrix is missing")?;
        assert_numeric_values(
            &actual_matrix,
            expected_matrix,
            "imported MovieClip moving-object reconstruction",
        )?;
        let expected_error = expected_pose["average_error"]
            .as_f64()
            .ok_or("Blender moving reconstruction camera error is missing")?;
        assert!(
            (pose.average_error - expected_error).abs() <= 1.0e-6,
            "imported moving-object reconstruction camera error differs"
        );
    }

    let exported = root.join("roundtrip.blend");
    pot_json(&[
        "export",
        project.to_str().unwrap(),
        "--format",
        "blend",
        "--out",
        exported.to_str().unwrap(),
        "--blender",
        blender.to_str().unwrap(),
    ])?;
    let reopened_path = root.join("blender_after.json");
    let script_path = root.join("reopen_constraints.py");
    fs::write(&script_path, BLENDER_REOPEN)?;
    let mut command = Command::new(&blender);
    command
        .args(["--background"])
        .arg(&exported)
        .arg("--python")
        .arg(&script_path)
        .arg("--")
        .arg(root)
        .arg(&reopened_path);
    let output = run_guarded(command)?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success() && !stdout.contains("Traceback") && !stderr.contains("Traceback"),
        "Blender reopen failed: stdout={stdout} stderr={stderr}"
    );
    let after: Value = serde_json::from_slice(&fs::read(reopened_path)?)?;
    assert_eq!(
        after["settings"], before["settings"],
        "constraint RNA settings changed on reopen"
    );
    let mut before_clip = before["clip"]
        .as_object()
        .ok_or("source MovieClip summary is missing")?
        .clone();
    let mut after_clip = after["clip"]
        .as_object()
        .ok_or("reopened MovieClip summary is missing")?
        .clone();
    for diagnostic in ["filepath", "filepath_exists"] {
        before_clip.remove(diagnostic);
        after_clip.remove(diagnostic);
    }
    assert_eq!(
        after_clip, before_clip,
        "MovieClip tracking/reconstruction changed on reopen"
    );
    assert!(
        after["clip"]["filepath_exists"].as_bool().unwrap_or(false),
        "reopened MovieClip media path does not resolve: {}",
        after["clip"]["filepath"].as_str().unwrap_or("<missing>")
    );
    assert_eq!(
        after["scene_camera"], before["scene_camera"],
        "Scene camera relationship changed on reopen"
    );
    // ObjectSolverConstraint exposes no inverse_matrix RNA in Blender 5.2.2;
    // evaluated matrices below verify the inverse's consumer-visible result.
    let expected_camera_matrices = before["reconstruction_cameras"]
        .as_array()
        .ok_or("source reconstruction camera matrices are missing")?;
    let reopened_camera_matrices = after["reconstruction_cameras"]
        .as_array()
        .ok_or("reopened reconstruction camera matrices are missing")?;
    assert_eq!(
        reopened_camera_matrices.len(),
        expected_camera_matrices.len(),
        "reopened reconstruction camera count"
    );
    for (actual_camera, expected_camera) in reopened_camera_matrices
        .iter()
        .zip(expected_camera_matrices)
    {
        assert_eq!(actual_camera["frame"], expected_camera["frame"]);
        let actual_matrix =
            flatten_matrix_rows(&actual_camera["matrix"], "reopened camera matrix")?;
        let expected_matrix =
            flatten_matrix_rows(&expected_camera["matrix"], "source camera matrix")?;
        assert_numeric_values(
            &actual_matrix,
            &expected_matrix,
            "reopened MovieClip reconstruction camera",
        )?;
        let actual_error = actual_camera["average_error"]
            .as_f64()
            .ok_or("reopened reconstruction camera error is missing")?;
        let expected_error = expected_camera["average_error"]
            .as_f64()
            .ok_or("source reconstruction camera error is missing")?;
        assert!(
            (actual_error - expected_error).abs() <= 1.0e-6,
            "reopened MovieClip reconstruction camera error changed"
        );
    }
    let expected_object_reconstruction = before["moving_reconstruction"]
        .as_array()
        .ok_or("source moving-object reconstruction is missing")?;
    let reopened_object_reconstruction = after["moving_reconstruction"]
        .as_array()
        .ok_or("reopened moving-object reconstruction is missing")?;
    assert_eq!(
        reopened_object_reconstruction.len(),
        expected_object_reconstruction.len(),
        "reopened moving-object reconstruction count"
    );
    for (actual_pose, expected_pose) in reopened_object_reconstruction
        .iter()
        .zip(expected_object_reconstruction)
    {
        assert_eq!(actual_pose["frame"], expected_pose["frame"]);
        assert_numeric_values(
            actual_pose["matrix"]
                .as_array()
                .ok_or("reopened moving-object reconstruction matrix is missing")?,
            expected_pose["matrix"]
                .as_array()
                .ok_or("source moving-object reconstruction matrix is missing")?,
            "reopened MovieClip moving-object reconstruction",
        )?;
        let actual_error = actual_pose["average_error"]
            .as_f64()
            .ok_or("reopened moving reconstruction error is missing")?;
        let expected_error = expected_pose["average_error"]
            .as_f64()
            .ok_or("source moving reconstruction error is missing")?;
        assert!(
            (actual_error - expected_error).abs() <= 1.0e-6,
            "reopened moving-object reconstruction camera error changed"
        );
    }
    for frame in FRAMES {
        for owner_kind in ["objects", "bones"] {
            let expected = before["frames"][frame.to_string()][owner_kind]
                .as_object()
                .ok_or("Blender owner matrices are missing")?;
            let actual = after["frames"][frame.to_string()][owner_kind]
                .as_object()
                .ok_or("reopened Blender owner matrices are missing")?;
            assert_eq!(
                actual.len(),
                expected.len(),
                "reopened {owner_kind} count at frame {frame}"
            );
            for (name, matrix) in expected {
                assert_blender_matrices(
                    actual[name]
                        .as_array()
                        .ok_or("reopened Blender matrix is missing")?,
                    matrix
                        .as_array()
                        .ok_or("source Blender matrix is missing")?,
                    &format!("reopened {owner_kind} {name} at frame {frame}"),
                );
            }
        }
    }
    let maximum_error = evaluation_max_errors
        .get(kind)
        .ok_or_else(|| format!("{kind} evaluation report is missing"))?;
    println!(
        "constraint parity {kind}: import=ok eval_max_error={maximum_error:.3e} export=ok reopen=ok"
    );
    if !evaluation_failures.is_empty() {
        return Err(format!(
            "constraint evaluation mismatches after export/reopen assertions:\n{}",
            evaluation_failures.join("\n")
        )
        .into());
    }
    Ok(())
}

macro_rules! constraint_roundtrip_tests {
    ($($test_name:ident => $constraint_type:literal),+ $(,)?) => {
        $(
            #[test]
            fn $test_name() -> TestResult {
                round_trip_constraint($constraint_type)
            }
        )+
    };
}

constraint_roundtrip_tests! {
    copy_location => "copy_location",
    copy_rotation => "copy_rotation",
    copy_scale => "copy_scale",
    copy_transforms => "copy_transforms",
    limit_location => "limit_location",
    limit_rotation => "limit_rotation",
    limit_scale => "limit_scale",
    limit_distance => "limit_distance",
    child_of => "child_of",
    damped_track => "damped_track",
    locked_track => "locked_track",
    track_to => "track_to",
    stretch_to => "stretch_to",
    ik => "ik",
    spline_ik => "spline_ik",
    clamp_to => "clamp_to",
    floor => "floor",
    follow_path => "follow_path",
    pivot => "pivot",
    shrinkwrap => "shrinkwrap",
    maintain_volume => "maintain_volume",
    transformation => "transformation",
    transform_cache => "transform_cache",
    armature => "armature",
    action => "action",
    geometry_attribute => "geometry_attribute",
    camera_solver => "camera_solver",
    follow_track => "follow_track",
    object_solver => "object_solver",
}
#[test]
fn blender_follow_path_curve_mesh_parity() -> TestResult {
    let _guard = BLENDER_CONSTRAINT_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(blender) = blender() else {
        eprintln!("skipping Blender Follow Path curve mesh test: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    fs::write(
        root.join("constraint_types.json"),
        serde_json::to_vec(&["follow_path"])?,
    )?;
    run_blender_script(
        &blender,
        "make_constraints.py",
        BLENDER_FIXTURE,
        root,
        &[],
        "Blender script make_constraints.py failed",
        run_guarded,
    )?;
    let before: Value = serde_json::from_slice(&fs::read(root.join("blender_before.json"))?)?;
    let project = root.join("project");
    pot_json(&["init", project.to_str().unwrap()])?;
    let imported = pot_json(&[
        "import",
        project.to_str().unwrap(),
        "--file",
        root.join("source.blend").to_str().unwrap(),
        "--format",
        "blend",
        "--mode",
        "replace",
        "--base-revision",
        "0",
        "--blender",
        blender.to_str().unwrap(),
    ])?;
    assert!(
        imported["result"]["losses"].as_array().unwrap().is_empty(),
        "curve geometry import reported losses: {imported}"
    );
    let mut doc: SceneDoc = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    for node in doc.nodes.values_mut() {
        node.constraints.clear();
    }
    let (curve_id, _) = node_named(&doc, "AnimatedPath")?;
    let snapshot = Snapshot::evaluate_with_cache(
        &doc,
        &EvaluationContext {
            frame: Some(2.0),
            ..EvaluationContext::default()
        },
        Some(&project),
    )?;
    let curve_mesh = snapshot
        .meshes
        .get(curve_id)
        .ok_or("Potter evaluated curve mesh is missing")?;
    assert_curve_point_sets(
        curve_mesh,
        &before["curve_mesh"],
        "AnimatedPath evaluated bevel/radius/tilt geometry",
    )?;
    Ok(())
}

fn assert_ik_pose_settings(actual: &Value, expected: &Value, context: &str) -> TestResult {
    for key in ["lock_ik", "use_ik_limit"] {
        assert_eq!(actual[key], expected[key], "{context} {key}");
    }
    for key in ["ik_min", "ik_max", "ik_stiffness"] {
        let actual = actual[key]
            .as_array()
            .ok_or_else(|| format!("{context} {key} is missing"))?;
        let expected = expected[key]
            .as_array()
            .ok_or_else(|| format!("{context} expected {key} is missing"))?;
        assert_numeric_values(actual, expected, &format!("{context} {key}"))?;
    }
    let actual = actual["ik_stretch"]
        .as_f64()
        .ok_or_else(|| format!("{context} ik_stretch is missing"))?;
    let expected = expected["ik_stretch"]
        .as_f64()
        .ok_or_else(|| format!("{context} expected ik_stretch is missing"))?;
    assert!(
        (actual - expected).abs() <= 1.0e-6,
        "{context} ik_stretch: {actual} != {expected}"
    );
    Ok(())
}

fn assert_boundary_edge_result(matrix: &[Value], context: &str) -> TestResult {
    assert_eq!(matrix.len(), 16, "{context}: matrix length");
    let x = matrix[12].as_f64().ok_or("boundary x is missing")?;
    let y = matrix[13].as_f64().ok_or("boundary y is missing")?;
    let z = matrix[14].as_f64().ok_or("boundary z is missing")?;
    assert!(
        (x - 0.446_383_6).abs() <= 1.0e-5 && y.abs() <= 1.0e-5 && z.abs() <= 1.0e-5,
        "{context}: Shrinkwrap did not match Blender's projected boundary-edge hit: ({x}, {y}, {z})"
    );
    Ok(())
}

fn assert_open_boundary_edge(summary: &Value, context: &str) -> TestResult {
    let edges = summary["boundary_edges"]
        .as_array()
        .ok_or_else(|| format!("{context}: boundary edge list is missing"))?;
    assert!(
        edges.iter().any(|edge| edge == &json!([0, 1])),
        "{context}: expected target edge (0, 1) to be an open boundary, found {edges:?}"
    );
    Ok(())
}

fn round_trip_ik_chain(chain_count: usize, swing_ellipse_limit: bool) -> TestResult {
    let _guard = BLENDER_CONSTRAINT_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(blender) = blender() else {
        eprintln!(
            "skipping Blender {chain_count}-bone IK chain roundtrip test: Blender is unavailable"
        );
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    fs::write(
        root.join("ik_chain.json"),
        serde_json::to_vec(&json!({
            "chain_count": chain_count,
            "swing_ellipse_limit": swing_ellipse_limit,
        }))?,
    )?;
    run_blender_script(
        &blender,
        "make_ik_chain.py",
        BLENDER_IK_CHAIN_FIXTURE,
        root,
        &[],
        "Blender script make_ik_chain.py failed",
        run_guarded,
    )?;
    let before: Value = serde_json::from_slice(&fs::read(root.join("ik_before.json"))?)?;
    let prefix = format!("IK{chain_count}");
    let rig_name = format!("{prefix}Rig");
    let bone_names: Vec<_> = (0..chain_count)
        .map(|index| format!("{prefix}_Bone{index}"))
        .collect();
    assert_eq!(before["constraint"]["chain_count"], json!(chain_count));
    assert_eq!(
        before["constraint"]["target"],
        json!(format!("{prefix}Target"))
    );
    assert_eq!(
        before["constraint"]["pole_target"],
        json!(format!("{prefix}Pole"))
    );
    assert_eq!(before["constraint"]["use_stretch"], json!(true));
    assert_eq!(before["constraint"]["use_location"], json!(true));
    assert_eq!(before["constraint"]["use_rotation"], json!(true));
    let expected_weight = 0.7;
    let expected_orient_weight = 0.35;
    let source_weight = before["constraint"]["weight"]
        .as_f64()
        .ok_or("Blender IK weight is missing")?;
    let source_orient_weight = before["constraint"]["orient_weight"]
        .as_f64()
        .ok_or("Blender IK orient weight is missing")?;
    assert!((source_weight - expected_weight).abs() <= 1.0e-6);
    assert!((source_orient_weight - expected_orient_weight).abs() <= 1.0e-6);
    let expected_influence = if chain_count == 2 { 0.65 } else { 1.0 };
    let source_influence = before["constraint"]["influence"]
        .as_f64()
        .ok_or("Blender IK influence is missing")?;
    assert!(
        (source_influence - expected_influence).abs() <= 1.0e-6,
        "{prefix} source IK influence"
    );
    for name in &bone_names {
        let settings = &before["pose_bones"][name.as_str()];
        let locks = settings["lock_ik"]
            .as_array()
            .ok_or_else(|| format!("{name} source IK locks are missing"))?;
        assert!(
            locks.iter().any(|axis| axis.as_bool() == Some(true)),
            "{name} source fixture must enable an IK lock"
        );
        assert_eq!(
            settings["use_ik_limit"],
            if !swing_ellipse_limit && name == bone_names.last().unwrap() {
                json!([false, true, false])
            } else {
                json!([true, true, true])
            }
        );
        for key in ["ik_min", "ik_max", "ik_stiffness"] {
            let values = settings[key]
                .as_array()
                .ok_or_else(|| format!("{name} source {key} values are missing"))?;
            assert!(
                values
                    .iter()
                    .any(|value| value.as_f64().is_some_and(|number| number != 0.0)),
                "{name} source fixture must set non-default {key}"
            );
        }
        assert!(
            settings["ik_stretch"]
                .as_f64()
                .is_some_and(|value| value > 0.0),
            "{name} source fixture must set IK stretch"
        );
    }
    assert_matrix_changed(
        before["frames"]["1"][bone_names.last().unwrap()]
            .as_array()
            .ok_or("frame-1 Blender IK bone matrix missing")?,
        before["frames"]["3"][bone_names.last().unwrap()]
            .as_array()
            .ok_or("frame-3 Blender IK bone matrix missing")?,
        &format!("{prefix} evaluated tip"),
    );

    let project = root.join("project");
    pot_json(&["init", project.to_str().unwrap()])?;
    let imported = pot_json(&[
        "import",
        project.to_str().unwrap(),
        "--file",
        root.join("ik_source.blend").to_str().unwrap(),
        "--format",
        "blend",
        "--mode",
        "replace",
        "--base-revision",
        "0",
        "--blender",
        blender.to_str().unwrap(),
    ])?;
    assert_eq!(
        imported["result"]["losses"],
        json!([]),
        "{prefix} IK import losses: {imported}"
    );
    let doc: SceneDoc = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let (rig_id, rig) = node_named(&doc, &rig_name)?;
    let data_id = rig
        .data
        .as_ref()
        .ok_or("imported IK rig has no armature data")?;
    let armature = doc
        .data_blocks
        .get(data_id)
        .and_then(|data| data.armature.as_ref())
        .ok_or("imported IK armature model is missing")?;
    let (target_id, _) = node_named(&doc, &format!("{prefix}Target"))?;
    let (pole_id, _) = node_named(&doc, &format!("{prefix}Pole"))?;
    let mut bone_ids = Vec::new();
    for name in &bone_names {
        let (bone_id, _) = armature
            .bones
            .iter()
            .find(|(_, bone)| bone.name == *name)
            .ok_or_else(|| format!("imported IK bone {name} is missing"))?;
        let pose = rig
            .pose
            .get(bone_id)
            .ok_or_else(|| format!("imported IK pose-bone settings for {name} are missing"))?;
        let settings = serde_json::to_value(pose)?;
        assert_ik_pose_settings(
            &settings,
            &before["pose_bones"][name.as_str()],
            &format!("{prefix} imported {name}"),
        )?;
        bone_ids.push(bone_id.clone());
    }
    let tip_id = bone_ids.last().ok_or("IK chain has no terminal bone")?;
    let constraint = rig
        .constraints
        .iter()
        .find(|constraint| constraint.owner_bone.as_ref() == Some(tip_id))
        .ok_or("imported IK constraint on terminal pose bone is missing")?;
    assert_eq!(
        serde_json::to_value(constraint.constraint_type)?,
        json!("ik"),
        "{prefix} imported constraint type"
    );
    assert_eq!(
        constraint.target.as_ref(),
        Some(target_id),
        "{prefix} target"
    );
    assert_eq!(constraint.params["pole_target"], json!(pole_id.as_str()));
    assert!(
        (constraint.influence
            - before["constraint"]["influence"]
                .as_f64()
                .ok_or("Blender IK influence is missing")?)
        .abs()
            <= 1.0e-6,
        "{prefix} imported IK influence"
    );
    for key in [
        "chain_count",
        "pole_angle",
        "use_stretch",
        "use_location",
        "use_rotation",
        "weight",
        "orient_weight",
    ] {
        assert_eq!(
            constraint.params.get(key),
            Some(&before["constraint"][key]),
            "{prefix} imported IK {key}"
        );
    }
    for key in ["use_location", "use_rotation", "weight", "orient_weight"] {
        assert_eq!(
            constraint.params.get(key),
            Some(&before["constraint"][key]),
            "{prefix} imported IK {key}"
        );
    }
    if swing_ellipse_limit {
        let error = Snapshot::evaluate_with_cache(
            &doc,
            &EvaluationContext {
                frame: Some(1.0),
                ..EvaluationContext::default()
            },
            Some(&project),
        )
        .err()
        .ok_or("IK X/Z Swing ellipse limits unexpectedly evaluated")?;
        assert_eq!(error.code, potter::error::ErrorCode::UnsupportedFeature);
        assert_eq!(
            error.details["feature_id"],
            json!("constraint.ik.swing_ellipse_limit")
        );
        assert_eq!(error.details["status"], json!("not_supported"));
        assert_eq!(
            error.details["bone_id"],
            json!(bone_ids.last().unwrap().as_str())
        );
        return Ok(());
    }

    let mut rest_doc = doc.clone();
    let rest_rig = rest_doc
        .nodes
        .get_mut(rig_id)
        .ok_or("imported armature disappeared from cloned scene")?;
    let rest_ik = rest_rig
        .constraints
        .iter_mut()
        .find(|item| item.id == constraint.id)
        .ok_or("IK constraint is missing from cloned scene")?;
    rest_ik.enabled = false;
    let mut rest_pose_max_error = 0.0_f64;
    for frame in FRAMES {
        let snapshot = Snapshot::evaluate_with_cache(
            &rest_doc,
            &EvaluationContext {
                frame: Some(f64::from(frame)),
                ..EvaluationContext::default()
            },
            Some(&project),
        )?;
        for (name, bone_id) in bone_names.iter().zip(&bone_ids) {
            let actual = snapshot
                .bone_matrices
                .get(rig_id)
                .and_then(|bones| bones.get(bone_id))
                .ok_or_else(|| format!("Potter rest-pose evaluation omitted {name}"))?;
            let actual: Vec<_> = actual.iter().map(|value| json!(value)).collect();
            let expected = before["rest_frames"][frame.to_string()][name.as_str()]
                .as_array()
                .ok_or_else(|| format!("Blender rest-pose matrix for {name} is missing"))?;
            let error = matrix_max_error(&actual, expected)?;
            rest_pose_max_error = rest_pose_max_error.max(error);
            assert_blender_matrices(
                &actual,
                expected,
                &format!("{prefix} rest pose {name} at frame {frame}"),
            );
        }
    }
    println!(
        "Blender IK chain {chain_count}-bone rest-pose parity: max_error={rest_pose_max_error:.3e}"
    );
    let mut maximum_error = 0.0_f64;
    let mut evaluation_failures = Vec::new();
    for frame in FRAMES {
        let snapshot = Snapshot::evaluate_with_cache(
            &doc,
            &EvaluationContext {
                frame: Some(f64::from(frame)),
                ..EvaluationContext::default()
            },
            Some(&project),
        )?;
        for (name, bone_id) in bone_names.iter().zip(&bone_ids) {
            let actual = snapshot
                .bone_matrices
                .get(rig_id)
                .and_then(|bones| bones.get(bone_id))
                .ok_or_else(|| format!("Potter evaluation omitted {name} at frame {frame}"))?;
            let actual: Vec<_> = actual.iter().map(|value| json!(value)).collect();
            let expected = before["frames"][frame.to_string()][name.as_str()]
                .as_array()
                .ok_or_else(|| format!("Blender matrix for {name} at frame {frame} is missing"))?;
            let error = matrix_max_error(&actual, expected)?;
            maximum_error = maximum_error.max(error);
            if error > 1.0e-5 {
                let mut rotation_error = 0.0_f64;
                for (potter, blender) in actual[..12].iter().zip(&expected[..12]) {
                    let potter = potter
                        .as_f64()
                        .ok_or("IK matrix component is not numeric")?;
                    let blender = blender
                        .as_f64()
                        .ok_or("Blender matrix component is not numeric")?;
                    rotation_error = rotation_error.max((potter - blender).abs());
                }
                eprintln!(
                    "{prefix} {name} frame {frame}: matrix={error:e}, rotation={rotation_error:e}, Potter translation={:?}, Blender translation={:?}",
                    &actual[12..15],
                    &expected[12..15]
                );
                evaluation_failures.push(format!("{name} at frame {frame}: {error:e}"));
            }
        }
    }
    println!(
        "Blender IK chain {chain_count}-bone parity: import=ok eval_max_error={maximum_error:.3e}"
    );

    let exported = root.join("ik_roundtrip.blend");
    pot_json(&[
        "export",
        project.to_str().unwrap(),
        "--format",
        "blend",
        "--out",
        exported.to_str().unwrap(),
        "--blender",
        blender.to_str().unwrap(),
    ])?;
    let reopened_path = root.join("ik_after.json");
    let script_path = root.join("reopen_ik_chain.py");
    fs::write(&script_path, BLENDER_IK_CHAIN_REOPEN)?;
    let mut command = Command::new(&blender);
    command
        .args(["--background"])
        .arg(&exported)
        .arg("--python")
        .arg(&script_path)
        .arg("--")
        .arg(root)
        .arg(&reopened_path);
    let output = run_guarded(command)?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success() && !stdout.contains("Traceback") && !stderr.contains("Traceback"),
        "{prefix} Blender reopen failed: stdout={stdout} stderr={stderr}"
    );
    let after: Value = serde_json::from_slice(&fs::read(reopened_path)?)?;
    assert_eq!(
        after["constraint"], before["constraint"],
        "{prefix} IK constraint settings changed on reopen"
    );
    for name in &bone_names {
        assert_ik_pose_settings(
            &after["pose_bones"][name.as_str()],
            &before["pose_bones"][name.as_str()],
            &format!("{prefix} reopened {name}"),
        )?;
    }
    for frame in FRAMES {
        for name in &bone_names {
            assert_blender_matrices(
                after["frames"][frame.to_string()][name.as_str()]
                    .as_array()
                    .ok_or("reopened Blender IK matrix is missing")?,
                before["frames"][frame.to_string()][name.as_str()]
                    .as_array()
                    .ok_or("source Blender IK matrix is missing")?,
                &format!("{prefix} reopened {name} at frame {frame}"),
            );
        }
    }
    println!(
        "Blender IK chain {chain_count}-bone parity: import=ok eval_max_error={maximum_error:.3e} export=ok reopen=ok"
    );
    if !evaluation_failures.is_empty() {
        return Err(evaluation_failures.join("\n").into());
    }
    Ok(())
}

#[test]
fn blender_ik_chain_2_pose_settings_roundtrip() -> TestResult {
    round_trip_ik_chain(2, false)
}

#[test]
fn blender_ik_chain_3_pose_settings_roundtrip() -> TestResult {
    round_trip_ik_chain(3, false)
}

#[test]
fn blender_ik_chain_swing_ellipse_limits_are_gated() -> TestResult {
    round_trip_ik_chain(2, true)?;
    round_trip_ik_chain(3, true)
}

#[test]
fn blender_target_project_boundary_edge_fallback_roundtrip() -> TestResult {
    let _guard = BLENDER_CONSTRAINT_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(blender) = blender() else {
        eprintln!(
            "skipping Blender TARGET_PROJECT boundary fallback roundtrip test: Blender is unavailable"
        );
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    run_blender_script(
        &blender,
        "make_target_project_boundary.py",
        BLENDER_TARGET_PROJECT_FIXTURE,
        root,
        &[],
        "Blender script make_target_project_boundary.py failed",
        run_guarded,
    )?;
    let before: Value =
        serde_json::from_slice(&fs::read(root.join("target_project_before.json"))?)?;
    let expected_location = json!([0.4, -0.25, 0.6]);
    assert_numeric_values(
        before["input_location"]
            .as_array()
            .ok_or("source input location is missing")?,
        expected_location
            .as_array()
            .ok_or("expected input location is invalid")?,
        "source point must project from outside the target triangles",
    )?;
    assert_eq!(
        before["constraint"]["shrinkwrap_type"],
        json!("TARGET_PROJECT")
    );
    assert_eq!(before["constraint"]["wrap_mode"], json!("ON_SURFACE"));
    assert_open_boundary_edge(&before, "Blender source mesh")?;
    let before_matrix = before["matrix"]
        .as_array()
        .ok_or("Blender source Shrinkwrap matrix is missing")?;
    assert_boundary_edge_result(before_matrix, "Blender source fallback")?;

    let project = root.join("target_project");
    pot_json(&["init", project.to_str().unwrap()])?;
    let imported = pot_json(&[
        "import",
        project.to_str().unwrap(),
        "--file",
        root.join("target_project_source.blend").to_str().unwrap(),
        "--format",
        "blend",
        "--mode",
        "replace",
        "--base-revision",
        "0",
        "--blender",
        blender.to_str().unwrap(),
    ])?;
    assert_eq!(
        imported["result"]["losses"],
        json!([]),
        "TARGET_PROJECT import losses: {imported}"
    );
    let doc: SceneDoc = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let (owner_id, owner) = node_named(&doc, "BoundaryOwner")?;
    let (target_id, _) = node_named(&doc, "BoundarySurface")?;
    assert_eq!(
        owner.constraints.len(),
        1,
        "TARGET_PROJECT constraint count"
    );
    let constraint = &owner.constraints[0];
    assert_eq!(
        serde_json::to_value(constraint.constraint_type)?,
        json!("shrinkwrap")
    );
    assert_eq!(constraint.target.as_ref(), Some(target_id));
    assert_eq!(
        constraint.params["shrinkwrap_type"],
        before["constraint"]["shrinkwrap_type"]
    );
    let snapshot =
        Snapshot::evaluate_with_cache(&doc, &EvaluationContext::default(), Some(&project))?;
    let actual = snapshot
        .nodes
        .get(owner_id)
        .ok_or("Potter evaluation omitted BoundaryOwner")?
        .world_matrix
        .map(|value| json!(value));
    assert_blender_matrices(
        &actual,
        before_matrix,
        "TARGET_PROJECT boundary fallback import evaluation",
    );
    let evaluation_error = matrix_max_error(&actual, before_matrix)?;
    assert_boundary_edge_result(&actual, "Potter import fallback")?;
    println!(
        "Blender TARGET_PROJECT boundary fallback: import=ok eval_max_error={evaluation_error:.3e}"
    );

    let exported = root.join("target_project_roundtrip.blend");
    pot_json(&[
        "export",
        project.to_str().unwrap(),
        "--format",
        "blend",
        "--out",
        exported.to_str().unwrap(),
        "--blender",
        blender.to_str().unwrap(),
    ])?;
    let reopened_path = root.join("target_project_after.json");
    let script_path = root.join("reopen_target_project_boundary.py");
    fs::write(&script_path, BLENDER_TARGET_PROJECT_REOPEN)?;
    let mut command = Command::new(&blender);
    command
        .args(["--background"])
        .arg(&exported)
        .arg("--python")
        .arg(&script_path)
        .arg("--")
        .arg(root)
        .arg(&reopened_path);
    let output = run_guarded(command)?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success() && !stdout.contains("Traceback") && !stderr.contains("Traceback"),
        "TARGET_PROJECT Blender reopen failed: stdout={stdout} stderr={stderr}"
    );
    let after: Value = serde_json::from_slice(&fs::read(reopened_path)?)?;
    assert_eq!(
        after["constraint"], before["constraint"],
        "TARGET_PROJECT constraint settings changed on reopen"
    );
    assert_eq!(
        after["input_location"], before["input_location"],
        "TARGET_PROJECT source transform changed on reopen"
    );
    assert_open_boundary_edge(&after, "reopened Blender mesh")?;
    assert_blender_matrices(
        after["matrix"]
            .as_array()
            .ok_or("reopened Blender Shrinkwrap matrix is missing")?,
        before_matrix,
        "TARGET_PROJECT reopened Blender fallback",
    );
    assert_boundary_edge_result(
        after["matrix"]
            .as_array()
            .ok_or("reopened Blender Shrinkwrap matrix is missing")?,
        "Blender reopened fallback",
    )?;
    println!(
        "Blender TARGET_PROJECT boundary fallback: import=ok eval_max_error={evaluation_error:.3e} export=ok reopen=ok"
    );
    Ok(())
}
