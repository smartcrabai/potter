import json
import math
from pathlib import Path

import bpy
from mathutils import Euler, Matrix, Vector


if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"fixture requires Blender 5.2.2, found {bpy.app.version_string}")

DIRECTORY = Path(__file__).resolve().parent
BLEND_PATH = DIRECTORY / "rotation_parenting.blend"
JSON_PATH = DIRECTORY / "rotation_parenting.json"


def set_transform(obj, location, rotation, scale, rotation_mode="XYZ"):
    obj.rotation_mode = rotation_mode
    obj.location = location
    obj.rotation_euler = Euler(rotation, rotation_mode)
    obj.scale = scale


def empty(name, location=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0),
          scale=(1.0, 1.0, 1.0), rotation_mode="XYZ"):
    obj = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(obj)
    set_transform(obj, location, rotation, scale, rotation_mode)
    return obj


def animate_object(obj, keys, rotation_mode="XYZ"):
    obj.rotation_mode = rotation_mode
    for frame, location, rotation, scale in keys:
        set_transform(obj, location, rotation, scale, rotation_mode)
        for channel in ("location", "rotation_euler", "scale"):
            obj.keyframe_insert(data_path=channel, frame=frame)


def make_rig(name, location, rotation, scale, bones):
    data = bpy.data.armatures.new(f"{name}Data")
    rig = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(rig)
    set_transform(rig, location, rotation, scale)
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    for bone_name, head, tail in bones:
        bone = data.edit_bones.new(bone_name)
        bone.head = head
        bone.tail = tail
    bpy.ops.object.mode_set(mode="OBJECT")
    return rig


def animate_bone(bone, keys, rotation_mode):
    bone.rotation_mode = rotation_mode
    for frame, location, rotation, scale in keys:
        bone.location = location
        bone.rotation_euler = Euler(rotation, rotation_mode)
        bone.scale = scale
        for channel in ("location", "rotation_euler", "scale"):
            bone.keyframe_insert(data_path=channel, frame=frame)


def copy_rotation(owner, name, target, *, euler_order, mix_mode,
                  owner_space, target_space, axes, invert, influence):
    constraint = owner.constraints.new("COPY_ROTATION")
    constraint.name = name
    constraint.target = target
    constraint.euler_order = euler_order
    constraint.mix_mode = mix_mode
    constraint.owner_space = owner_space
    constraint.target_space = target_space
    constraint.use_x, constraint.use_y, constraint.use_z = axes
    constraint.invert_x, constraint.invert_y, constraint.invert_z = invert
    constraint.influence = influence
    return constraint


def limit_rotation(owner, name, *, euler_order, axes, minimum, maximum,
                   legacy, influence):
    constraint = owner.constraints.new("LIMIT_ROTATION")
    constraint.name = name
    constraint.euler_order = euler_order
    constraint.use_limit_x, constraint.use_limit_y, constraint.use_limit_z = axes
    constraint.min_x, constraint.min_y, constraint.min_z = minimum
    constraint.max_x, constraint.max_y, constraint.max_z = maximum
    constraint.use_legacy_behavior = legacy
    constraint.influence = influence
    return constraint


def child_of(owner, name, target, *, owner_space, target_space, channels,
             inverse, influence):
    constraint = owner.constraints.new("CHILD_OF")
    constraint.name = name
    constraint.target = target
    constraint.owner_space = owner_space
    constraint.target_space = target_space
    for channel, enabled in zip(
        ("location", "rotation", "scale"), channels, strict=True
    ):
        for axis, value in zip("xyz", enabled, strict=True):
            setattr(constraint, f"use_{channel}_{axis}", value)
    if inverse is None:
        constraint.set_inverse_pending = True
    else:
        constraint.inverse_matrix = inverse
        constraint.set_inverse_pending = False
    constraint.influence = influence
    return constraint


def inverse_matrix(location, rotation, scale):
    return (
        Matrix.Translation(Vector(location))
        @ Euler(rotation, "XYZ").to_matrix().to_4x4()
        @ Matrix.Diagonal((*scale, 1.0))
    )


def column_major(matrix):
    values = [float(matrix[row][column]) for column in range(4) for row in range(4)]
    return [0.0 if round(value, 7) == 0 else round(value, 7) for value in values]


bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 3

# Each target has an animated local transform under an independently transformed parent.
rotation_parent_a = empty("RotationTargetParentA", (0.35, -0.8, 0.6), (0.3, -0.4, 0.2), (1.1, 1.1, 1.1))
rotation_parent_b = empty("RotationTargetParentB", (-0.7, 0.25, 0.9), (-0.25, 0.45, -0.35), (0.9, 0.9, 0.9))
rotation_target_a = empty("RotationTargetA")
rotation_target_a.parent = rotation_parent_a
animate_object(rotation_target_a, [
    (1, (0.8, -0.35, 0.5), (0.35, -0.25, 0.6), (1.1, 0.8, 1.3)),
    (2, (-0.45, 0.9, -0.2), (-0.4, 0.55, -0.15), (0.75, 1.4, 0.9)),
    (3, (0.55, 0.15, 1.0), (0.7, -0.1, -0.45), (1.35, 0.95, 0.7)),
])
rotation_target_b = empty("RotationTargetB")
rotation_target_b.parent = rotation_parent_b
animate_object(rotation_target_b, [
    (1, (-0.6, 0.4, 0.75), (-0.2, 0.5, -0.3), (0.9, 1.25, 0.7)),
    (2, (0.7, -0.55, 0.1), (0.6, -0.3, 0.45), (1.4, 0.7, 1.15)),
    (3, (0.15, 0.8, -0.65), (-0.5, 0.2, 0.75), (0.65, 1.5, 0.85)),
])

child_target_rig = make_rig(
    "Rig", (0.45, -0.8, 0.65), (0.18, -0.27, 0.34), (1.0, 1.0, 1.0),
    [("Ctrl", (-0.6, 0.0, 0.0), (0.2, 0.75, 0.18))],
)
child_target_ctrl = child_target_rig.pose.bones["Ctrl"]
child_target_ctrl.rotation_mode = "ZXY"
child_target_ctrl.location = (0.1, -0.2, 0.3)
child_target_ctrl.rotation_euler = Euler((0.45, 0.72, -0.28), "ZXY")
child_target_ctrl.scale = (1.0, 1.0, 1.0)

owner_parent = empty("RotationOwnerParent", (0.4, -0.3, 0.7), (0.2, -0.35, 0.25), (1.15, 1.15, 1.15))
copy_replace = empty("CopyRotationExplicitReplace", rotation_mode="XYZ")
copy_replace.parent = owner_parent
animate_object(copy_replace, [
    (1, (-0.25, 0.4, 0.8), (0.25, -0.55, 0.35), (0.9, 1.2, 0.8)),
    (2, (0.65, -0.3, 0.2), (-0.4, 0.25, 0.7), (1.25, 0.75, 1.1)),
    (3, (-0.1, 0.75, -0.45), (0.5, -0.2, -0.35), (0.7, 1.35, 0.95)),
])
copy_replace_constraint = copy_rotation(
    copy_replace, "Explicit XYZ Selected Inverted Replace", rotation_target_a,
    euler_order="XYZ", mix_mode="REPLACE", owner_space="LOCAL", target_space="LOCAL",
    axes=(True, False, True), invert=(False, False, True), influence=0.68,
)

copy_auto = empty("CopyRotationAutoAdd", rotation_mode="XYZ")
copy_auto.parent = owner_parent
animate_object(copy_auto, [
    (1, (0.5, -0.7, 0.15), (-0.2, 0.35, -0.45), (1.1, 0.8, 1.3)),
    (2, (-0.6, 0.25, 0.9), (0.45, -0.15, 0.55), (0.85, 1.3, 0.75)),
    (3, (0.2, 0.8, -0.3), (-0.35, 0.65, 0.1), (1.4, 0.7, 1.05)),
], "XYZ")
copy_auto_constraint = copy_rotation(
    copy_auto, "AUTO Owner XYZ Add", rotation_target_b,
    euler_order="AUTO", mix_mode="ADD", owner_space="WORLD", target_space="WORLD",
    axes=(True, True, False), invert=(True, False, False), influence=1.0,
)

copy_offset = empty("CopyRotationOffset", rotation_mode="XYZ")
copy_offset.parent = owner_parent
animate_object(copy_offset, [
    (1, (-0.8, 0.2, 0.45), (0.35, 0.1, -0.5), (0.8, 1.25, 0.9)),
    (2, (0.35, -0.45, 0.7), (-0.15, 0.5, 0.25), (1.3, 0.85, 1.15)),
    (3, (0.75, 0.6, -0.15), (0.6, -0.35, 0.4), (0.7, 1.4, 0.8)),
], "XYZ")
copy_offset_constraint = copy_rotation(
    copy_offset, "Explicit XYZ Offset", rotation_target_a,
    euler_order="XYZ", mix_mode="OFFSET", owner_space="LOCAL", target_space="WORLD",
    axes=(True, True, True), invert=(False, True, False), influence=0.82,
)
# Blender's normalized XYZ decomposition selects the lowest-magnitude solution. Both
# targets have a middle axis above 90 degrees and opposite X/Z signs so selecting the
# canonical middle-angle solution changes the copied X/Z channels.
euler_alt_target_positive_x = empty("EulerAlternateTargetPositiveX")
animate_object(euler_alt_target_positive_x, [
    (1, (0.2, -0.1, 0.3), (2.4, 2.1, -0.6), (1.0, 1.0, 1.0)),
    (2, (0.4, 0.2, -0.1), (2.2, 2.2, -0.45), (1.0, 1.0, 1.0)),
    (3, (-0.1, 0.3, 0.2), (2.6, 2.05, -0.8), (1.0, 1.0, 1.0)),
])
euler_alt_owner_positive_x = empty("CopyRotationEulerAlternatePositiveX")
animate_object(euler_alt_owner_positive_x, [
    (1, (-0.2, 0.1, 0.3), (0.2, -0.5, 0.45), (1.0, 1.0, 1.0)),
    (2, (0.3, -0.2, 0.1), (0.35, -0.4, 0.5), (1.0, 1.0, 1.0)),
    (3, (0.1, 0.25, -0.15), (0.15, -0.6, 0.35), (1.0, 1.0, 1.0)),
])
euler_alt_positive_constraint = copy_rotation(
    euler_alt_owner_positive_x, "XYZ Lowest-Magnitude Solution Positive X",
    euler_alt_target_positive_x, euler_order="XYZ", mix_mode="REPLACE",
    owner_space="WORLD", target_space="WORLD", axes=(True, False, True),
    invert=(False, False, False), influence=1.0,
)

euler_alt_target_negative_x = empty("EulerAlternateTargetNegativeX")
animate_object(euler_alt_target_negative_x, [
    (1, (-0.3, 0.1, -0.2), (-2.4, 2.1, 0.6), (1.0, 1.0, 1.0)),
    (2, (0.1, 0.3, -0.2), (-2.2, 2.2, 0.45), (1.0, 1.0, 1.0)),
    (3, (0.25, -0.1, 0.3), (-2.6, 2.05, 0.8), (1.0, 1.0, 1.0)),
])
euler_alt_owner_negative_x = empty("CopyRotationEulerAlternateNegativeX")
animate_object(euler_alt_owner_negative_x, [
    (1, (0.2, -0.3, 0.1), (-0.25, 0.45, -0.35), (1.0, 1.0, 1.0)),
    (2, (-0.1, 0.25, 0.2), (-0.35, 0.4, -0.5), (1.0, 1.0, 1.0)),
    (3, (0.3, 0.15, -0.25), (-0.15, 0.6, -0.45), (1.0, 1.0, 1.0)),
])
euler_alt_negative_constraint = copy_rotation(
    euler_alt_owner_negative_x, "XYZ Lowest-Magnitude Solution Negative X",
    euler_alt_target_negative_x, euler_order="XYZ", mix_mode="REPLACE",
    owner_space="WORLD", target_space="WORLD", axes=(True, False, True),
    invert=(False, False, False), influence=1.0,
)
assert euler_alt_target_positive_x.rotation_euler.y > math.pi / 2
assert euler_alt_target_positive_x.rotation_euler.x > 0
assert euler_alt_target_positive_x.rotation_euler.z < 0
assert euler_alt_target_negative_x.rotation_euler.y > math.pi / 2
assert euler_alt_target_negative_x.rotation_euler.x < 0
assert euler_alt_target_negative_x.rotation_euler.z > 0
limit_euler_positive_x = empty("LimitRotationEulerLowestPositiveX", rotation_mode="ZYX")
animate_object(limit_euler_positive_x, [
    (1, (-0.3, 0.2, 0.5), (0.6, 2.0, -0.8), (1.0, 1.0, 1.0)),
    (2, (0.15, -0.25, 0.4), (0.7, 2.1, -0.7), (1.0, 1.0, 1.0)),
    (3, (0.25, 0.1, -0.3), (0.8, 2.15, -0.9), (1.0, 1.0, 1.0)),
], "ZYX")
limit_euler_positive_constraint = limit_rotation(
    limit_euler_positive_x, "ZYX Lowest-Magnitude +X -Z",
    euler_order="ZYX", axes=(True, False, True),
    minimum=(-0.4, 0.0, -0.3), maximum=(0.4, 0.0, 0.3),
    legacy=False, influence=1.0,
)

limit_euler_negative_x = empty("LimitRotationEulerLowestNegativeX", rotation_mode="ZYX")
animate_object(limit_euler_negative_x, [
    (1, (0.2, -0.1, 0.35), (-0.7, 2.0, 0.9), (1.0, 1.0, 1.0)),
    (2, (-0.15, 0.3, -0.2), (-0.6, 2.1, 0.8), (1.0, 1.0, 1.0)),
    (3, (0.3, 0.1, 0.2), (-0.8, 2.15, 0.7), (1.0, 1.0, 1.0)),
], "ZYX")
limit_euler_negative_constraint = limit_rotation(
    limit_euler_negative_x, "ZYX Lowest-Magnitude -X +Z",
    euler_order="ZYX", axes=(True, False, True),
    minimum=(-0.4, 0.0, -0.3), maximum=(0.4, 0.0, 0.3),
    legacy=False, influence=1.0,
)
assert limit_euler_positive_x.rotation_mode == "ZYX"
assert limit_euler_positive_x.rotation_euler.y > math.pi / 2
assert limit_euler_positive_x.rotation_euler.x > 0
assert limit_euler_positive_x.rotation_euler.z < 0
assert limit_euler_negative_x.rotation_mode == "ZYX"
assert limit_euler_negative_x.rotation_euler.y > math.pi / 2
assert limit_euler_negative_x.rotation_euler.x < 0
assert limit_euler_negative_x.rotation_euler.z > 0

limit_gimbal = empty("LimitRotationEulerGimbal90", rotation_mode="XYZ")
animate_object(limit_gimbal, [
    (1, (-0.25, 0.15, 0.3), (0.3, math.pi / 2, 0.2), (1.0, 1.0, 1.0)),
    (2, (0.2, -0.35, 0.15), (-0.4, math.pi / 2, 0.25), (1.0, 1.0, 1.0)),
    (3, (0.35, 0.1, -0.2), (0.55, math.pi / 2, -0.15), (1.0, 1.0, 1.0)),
], "XYZ")
limit_gimbal_constraint = limit_rotation(
    limit_gimbal, "XYZ Gimbal-Lock Z Limit",
    euler_order="XYZ", axes=(False, False, True),
    minimum=(0.0, 0.0, -0.1), maximum=(0.0, 0.0, 0.1),
    legacy=False, influence=1.0,
)




# Rotation limits include a wrapped non-legacy interval, legacy clamping, and a bone order.
limit_wrap = empty("LimitRotationWrapZ")
animate_object(limit_wrap, [
    (1, (-0.4, 0.2, 0.65), (0.15, -0.25, math.radians(-179.0)), (1.1, 0.85, 1.2)),
    (2, (0.55, -0.35, 0.1), (-0.3, 0.4, math.radians(-175.0)), (0.8, 1.3, 0.9)),
    (3, (-0.15, 0.7, -0.45), (0.45, -0.1, math.radians(179.0)), (1.25, 0.75, 1.1)),
])
limit_wrap_constraint = limit_rotation(
    limit_wrap, "Nonlegacy Wrapped Z 170 To 190 Degrees",
    euler_order="XYZ", axes=(False, False, True),
    minimum=(0.0, 0.0, math.radians(170.0)),
    maximum=(0.0, 0.0, math.radians(190.0)), legacy=False, influence=1.0,
)

limit_legacy = empty("LimitRotationLegacyClamp", rotation_mode="XYZ")
animate_object(limit_legacy, [
    (1, (0.25, -0.55, 0.4), (0.9, -0.75, 0.2), (0.9, 1.15, 0.8)),
    (2, (-0.65, 0.3, 0.75), (-0.7, 0.8, -0.4), (1.3, 0.7, 1.2)),
    (3, (0.8, 0.1, -0.35), (0.65, -0.9, 0.5), (0.75, 1.4, 0.95)),
], "XYZ")
limit_legacy_constraint = limit_rotation(
    limit_legacy, "Legacy Clamp XY", euler_order="XYZ", axes=(True, True, False),
    minimum=(math.radians(-20.0), math.radians(-35.0), 0.0),
    maximum=(math.radians(25.0), math.radians(40.0), 0.0), legacy=True, influence=0.72,
)

# A single armature holds the pose-bone owners for rotation and limits.
rig = make_rig(
    "RotationParentRig", (0.3, 0.85, -0.55), (0.3, -0.25, 0.45), (1.1, 1.1, 1.1),
    [
        ("CopyRotationBone", (-0.9, 0.0, 0.0), (-0.25, 0.65, 0.25)),
        ("LimitRotationBone", (-0.1, 0.0, 0.0), (0.55, 0.7, 0.15)),
        ("ChildOfPoseBone", (0.75, 0.0, 0.0), (1.3, 0.8, 0.2)),
    ],
)
copy_bone = rig.pose.bones["CopyRotationBone"]
animate_bone(copy_bone, [
    (1, (0.15, -0.3, 0.4), (0.25, -0.45, 0.15), (0.9, 1.15, 0.8)),
    (2, (-0.35, 0.25, 0.1), (-0.3, 0.4, 0.55), (1.25, 0.75, 1.1)),
    (3, (0.4, 0.45, -0.2), (0.5, 0.1, -0.35), (0.8, 1.35, 0.95)),
], "XYZ")
copy_bone_constraint = copy_rotation(
    copy_bone, "Explicit XYZ Bone Before", rotation_target_b,
    euler_order="XYZ", mix_mode="BEFORE", owner_space="POSE", target_space="WORLD",
    axes=(False, True, True), invert=(False, True, False), influence=0.47,
)

limit_bone = rig.pose.bones["LimitRotationBone"]
animate_bone(limit_bone, [
    (1, (-0.2, 0.25, 0.35), (0.7, -0.9, 0.8), (1.1, 0.8, 1.2)),
    (2, (0.3, -0.4, 0.15), (-0.65, 0.75, -0.7), (0.85, 1.3, 0.75)),
    (3, (0.45, 0.1, -0.3), (0.85, -0.55, 0.65), (1.35, 0.7, 1.0)),
], "XYZ")
limit_bone_constraint = limit_rotation(
    limit_bone, "Bone YZ Partial Nonlegacy", euler_order="XYZ", axes=(False, True, True),
    minimum=(0.0, math.radians(-15.0), math.radians(-30.0)),
    maximum=(0.0, math.radians(25.0), math.radians(40.0)), legacy=False, influence=0.58,
)

child_of_bone = rig.pose.bones["ChildOfPoseBone"]
animate_bone(child_of_bone, [
    (1, (0.12, -0.18, 0.25), (0.25, 0.6, -0.35), (1.0, 1.0, 1.0)),
    (2, (-0.2, 0.3, 0.1), (-0.4, 0.75, 0.2), (1.0, 1.0, 1.0)),
    (3, (0.35, 0.1, -0.22), (0.5, 0.85, -0.45), (1.0, 1.0, 1.0)),
], "XYZ")
child_of_bone_constraint = child_of(
    child_of_bone, "Pose Child Of POSE Local Partial", child_target_rig,
    owner_space="POSE", target_space="LOCAL",
    channels=((True, True, True), (True, False, True), (True, True, True)),
    inverse=inverse_matrix((0.25, -0.4, 0.15), (0.2, 0.3, -0.25), (1.0, 1.0, 1.0)),
    influence=0.43,
)
child_of_bone_constraint.subtarget = "Ctrl"


# Child Of targets combine rotated parents and animated local rotations, so their transforms do
# not commute. Explicit inverses and disabled channels cover per-channel evaluation.
child_target_parent_a = empty("ChildTargetParentA", (-0.6, 0.8, 0.25), (0.55, -0.25, 0.4), (1.2, 1.2, 1.2))
child_target_a = empty("ChildTargetA")
child_target_a.parent = child_target_parent_a
animate_object(child_target_a, [
    (1, (0.65, -0.2, 0.45), (-0.35, 0.6, 0.1), (0.8, 1.3, 0.9)),
    (2, (-0.4, 0.75, -0.15), (0.45, -0.2, 0.7), (1.35, 0.7, 1.1)),
    (3, (0.2, 0.15, 0.8), (-0.1, 0.75, -0.4), (0.95, 1.2, 0.75)),
])
child_target_parent_b = empty("ChildTargetParentB", (0.9, -0.5, 0.4), (-0.3, 0.65, -0.2), (0.85, 0.85, 0.85))
child_target_b = empty("ChildTargetB")
child_target_b.parent = child_target_parent_b
animate_object(child_target_b, [
    (1, (-0.25, 0.55, 0.3), (0.2, -0.5, 0.75), (1.1, 0.8, 1.3)),
    (2, (0.7, -0.35, 0.6), (-0.55, 0.25, -0.1), (0.75, 1.4, 0.9)),
    (3, (-0.5, 0.15, -0.4), (0.4, 0.35, 0.55), (1.25, 0.7, 1.15)),
])

child_world = empty("ChildOfWorldChannels", rotation_mode="XYZ")
animate_object(child_world, [
    (1, (-0.7, 0.35, 0.5), (0.3, -0.4, 0.25), (0.9, 1.2, 0.75)),
    (2, (0.4, -0.6, 0.2), (-0.25, 0.5, -0.35), (1.3, 0.8, 1.1)),
    (3, (0.8, 0.1, -0.45), (0.55, -0.15, 0.6), (0.75, 1.35, 0.9)),
])
child_world_constraint = child_of(
    child_world, "World XZ Location Y Rotation XZ Scale", child_target_a,
    owner_space="WORLD", target_space="WORLD",
    channels=((True, False, True), (False, True, False), (True, False, True)),
    inverse=inverse_matrix((0.35, -0.6, 0.25), (0.2, -0.35, 0.5), (1.1, 0.85, 1.25)),
    influence=0.72,
)

child_world_full = empty("ChildOfWorldFullChannels", rotation_mode="ZYX")
animate_object(child_world_full, [
    (1, (0.3, -0.5, 0.75), (-0.35, 0.15, 0.45), (1.2, 0.75, 1.05)),
    (2, (-0.55, 0.4, -0.1), (0.5, -0.3, 0.2), (0.8, 1.25, 0.9)),
    (3, (0.15, 0.85, 0.35), (-0.2, 0.6, -0.5), (1.35, 0.7, 1.2)),
], "ZYX")
child_world_full.parent = owner_parent
child_world_full_constraint = child_of(
    child_world_full, "Local Full-Channel Composition", child_target_b,
    owner_space="LOCAL", target_space="LOCAL",
    channels=((True, True, True), (True, True, True), (True, True, True)),
    inverse=None,
    influence=0.46,
)

child_of_filtered_owner = empty("ChildOfFilteredTargetBone", rotation_mode="XYZ")
animate_object(child_of_filtered_owner, [
    (1, (-0.3, 0.25, 0.55), (0.35, -0.25, 0.4), (1.0, 1.0, 1.0)),
    (2, (0.25, -0.4, 0.2), (-0.45, 0.35, -0.3), (1.0, 1.0, 1.0)),
    (3, (0.15, 0.3, -0.25), (0.55, -0.15, 0.25), (1.0, 1.0, 1.0)),
], "XYZ")
child_of_filtered_constraint = child_of(
    child_of_filtered_owner, "Filtered Child Of Rig Ctrl ZXY", child_target_rig,
    owner_space="WORLD", target_space="WORLD",
    channels=((True, True, True), (False, True, True), (True, True, True)),
    inverse=inverse_matrix((0.1, -0.2, 0.3), (0.25, -0.15, 0.35), (1.0, 1.0, 1.0)),
    influence=0.67,
)
child_of_filtered_constraint.subtarget = "Ctrl"



# Leave rotation_range unset: Blender's native Pivot constraint default is NX.
pivot_target = empty("PivotTarget")
animate_object(pivot_target, [
    (1, (1.4, -0.6, 0.35), (0.25, -0.4, 0.3), (1.1, 0.8, 1.2)),
    (2, (-0.5, 1.2, -0.25), (-0.35, 0.55, -0.1), (0.75, 1.35, 0.9)),
    (3, (0.7, 0.15, 0.95), (0.6, -0.2, 0.45), (1.3, 0.7, 1.1)),
])
pivot_owner = empty("PivotNXOwnerZYX", rotation_mode="ZYX")
animate_object(pivot_owner, [
    (1, (-0.8, 0.45, 0.25), (-0.35, 0.25, -0.55), (0.9, 1.15, 0.8)),
    (2, (0.35, -0.65, 0.7), (0.45, -0.3, 0.2), (1.25, 0.75, 1.1)),
    (3, (0.65, 0.2, -0.4), (-0.2, 0.6, -0.35), (0.8, 1.3, 0.95)),
], "ZYX")
pivot_constraint = pivot_owner.constraints.new("PIVOT")
pivot_constraint.name = "Default NX Pivot Range"
pivot_constraint.target = pivot_target
pivot_constraint.owner_space = "WORLD"
pivot_constraint.target_space = "WORLD"
pivot_constraint.use_relative_location = True
pivot_constraint.influence = 0.74

# The target's X samples pass from above to below a deliberately reversed source interval.
transform_target = empty("TransformReversedRangeTarget")
animate_object(transform_target, [
    (1, (2.0, -0.7, 0.45), (0.25, -0.4, 0.35), (0.9, 1.2, 0.8)),
    (2, (0.0, 0.55, -0.25), (-0.3, 0.5, -0.15), (1.3, 0.75, 1.15)),
    (3, (-2.0, -0.2, 0.7), (0.6, -0.1, 0.45), (0.75, 1.35, 0.9)),
])
transform_owner = empty("TransformReversedRangeOwner", (0.2, -0.35, 0.5), (0.3, -0.45, 0.15), (1.1, 0.85, 1.25))
animate_object(transform_owner, [
    (1, (0.5, -0.2, 0.3), (-0.2, 0.35, 0.55), (0.9, 1.15, 0.8)),
    (2, (-0.45, 0.6, -0.15), (0.4, -0.25, -0.35), (1.25, 0.75, 1.05)),
    (3, (0.7, 0.1, 0.65), (-0.35, 0.5, 0.2), (0.8, 1.3, 0.9)),
])
transform_constraint = transform_owner.constraints.new("TRANSFORM")
transform_constraint.name = "Reversed X Location Range"
transform_constraint.target = transform_target
transform_constraint.owner_space = "WORLD"
transform_constraint.target_space = "WORLD"
transform_constraint.map_from = "LOCATION"
transform_constraint.map_to = "LOCATION"
transform_constraint.map_to_x_from = "X"
transform_constraint.from_min_x = 1.5
transform_constraint.from_max_x = -1.5
transform_constraint.to_min_x = -1.25
transform_constraint.to_max_x = 1.75
transform_constraint.from_min_y = -1.0
transform_constraint.from_max_y = 1.0
transform_constraint.to_min_y = -1.0
transform_constraint.to_max_y = 1.0
transform_constraint.from_min_z = -1.0
transform_constraint.from_max_z = 1.0
transform_constraint.to_min_z = -1.0
transform_constraint.to_max_z = 1.0
transform_constraint.influence = 0.8

object_owners = [
    copy_replace, copy_auto, copy_offset,
    limit_wrap, limit_legacy, limit_gimbal,
    child_world, child_world_full, child_of_filtered_owner,
    euler_alt_owner_positive_x, euler_alt_owner_negative_x,
    limit_euler_positive_x, limit_euler_negative_x,
    pivot_owner, transform_owner,
]
bone_owners = [copy_bone, limit_bone, child_of_bone]
owners = [{"kind": "object", "name": owner.name} for owner in object_owners] + [
    {"kind": "bone", "armature": rig.name, "name": bone.name} for bone in bone_owners
]

# Keep the authored settings observable in the saved native scene.
assert [copy_replace_constraint.euler_order, copy_auto_constraint.euler_order,
        copy_offset_constraint.euler_order, copy_bone_constraint.euler_order] == ["XYZ", "AUTO", "XYZ", "XYZ"]
assert {copy_replace_constraint.mix_mode, copy_auto_constraint.mix_mode,
        copy_offset_constraint.mix_mode, copy_bone_constraint.mix_mode} == {"REPLACE", "ADD", "OFFSET", "BEFORE"}
assert copy_auto.rotation_mode == "XYZ" and copy_auto_constraint.influence == 1.0
assert any(constraint.influence < 1.0 for constraint in (
    copy_replace_constraint, copy_offset_constraint, copy_bone_constraint,
))
assert limit_wrap_constraint.use_legacy_behavior is False
assert limit_wrap_constraint.use_limit_z and math.isclose(
    limit_wrap_constraint.min_z, math.radians(170.0), abs_tol=1e-6
)
assert limit_legacy_constraint.use_legacy_behavior is True
assert limit_bone.rotation_mode == "XYZ" and limit_bone_constraint.euler_order == "XYZ"
assert limit_bone_constraint.use_limit_y and limit_bone_constraint.use_limit_z
assert child_world_constraint.inverse_matrix != Matrix.Identity(4)
assert any(
    not getattr(child_world_constraint, f"use_{kind}_{axis}")
    for kind in ("location", "rotation", "scale")
    for axis in "xyz"
)
assert pivot_owner.rotation_mode == "ZYX" and pivot_constraint.rotation_range == "NX"
assert transform_constraint.from_min_x > transform_constraint.from_max_x
assert limit_gimbal.rotation_mode == "XYZ" and limit_gimbal_constraint.use_limit_z
assert child_target_rig.rotation_mode == "XYZ"
assert child_target_ctrl.rotation_mode == "ZXY"
assert child_of_filtered_constraint.subtarget == "Ctrl"
assert child_of_bone_constraint.owner_space == "POSE"
assert child_of_bone_constraint.target_space == "LOCAL"
assert child_of_bone_constraint.influence < 1.0

assert len(object_owners) == 15 and len(bone_owners) == 3 and len(owners) == 18
captured = {}
for frame in (1, 2, 3):
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    frame_values = {}
    for owner in object_owners:
        frame_values[owner.name] = column_major(owner.evaluated_get(depsgraph).matrix_world)
    evaluated_rig = rig.evaluated_get(depsgraph)
    for bone in bone_owners:
        evaluated_bone = evaluated_rig.pose.bones[bone.name]
        key = f"Bone:{rig.name}:{bone.name}"
        frame_values[key] = column_major(evaluated_rig.matrix_world @ evaluated_bone.matrix)
    assert len(frame_values) == len(owners)
    assert all(len(matrix) == 16 for matrix in frame_values.values())
    captured[str(frame)] = frame_values

for key in [owner["name"] for owner in owners if owner["kind"] == "object"] + [
    f"Bone:{owner['armature']}:{owner['name']}" for owner in owners if owner["kind"] == "bone"
]:
    assert len({tuple(captured[str(frame)][key]) for frame in (1, 2, 3)}) == 3, key
assert not child_world_full_constraint.set_inverse_pending
assert child_world_full_constraint.inverse_matrix != Matrix.Identity(4)

scene.frame_set(1)
bpy.context.view_layer.update()
bpy.context.preferences.filepaths.save_version = 0
bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))
with JSON_PATH.open("w", encoding="utf-8", newline="\n") as output:
    json.dump({"owners": owners, "frames": captured}, output, separators=(",", ":"), allow_nan=False)
    output.write("\n")
