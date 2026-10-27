import json
from pathlib import Path

import bpy
from mathutils import Euler


if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"fixture requires Blender 5.2.2, found {bpy.app.version_string}")

DIRECTORY = Path(__file__).resolve().parent
BLEND_PATH = DIRECTORY / "floor_volume.blend"
JSON_PATH = DIRECTORY / "floor_volume.json"


def set_transform(owner, location, rotation, scale):
    owner.location = location
    owner.rotation_mode = "XYZ"
    owner.rotation_euler = Euler(rotation, "XYZ")
    owner.scale = scale


def empty(name, location, rotation, scale):
    owner = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(owner)
    set_transform(owner, location, rotation, scale)
    return owner


def action_fcurves(owner):
    return (
        curve
        for layer in owner.animation_data.action.layers
        for strip in layer.strips
        for bag in strip.channelbags
        for curve in bag.fcurves
    )


def animate(owner, keys):
    for frame, location, rotation, scale in keys:
        set_transform(owner, location, rotation, scale)
        for channel in ("location", "rotation_euler", "scale"):
            owner.keyframe_insert(data_path=channel, frame=frame)
    for curve in action_fcurves(owner):
        for point in curve.keyframe_points:
            point.interpolation = "LINEAR"


def animate_bone(armature, bone, keys):
    for frame, location, rotation, scale in keys:
        bone.location = location
        bone.rotation_mode = "XYZ"
        bone.rotation_euler = Euler(rotation, "XYZ")
        bone.scale = scale
        for channel in ("location", "rotation_euler", "scale"):
            bone.keyframe_insert(data_path=channel, frame=frame)
    for curve in action_fcurves(armature):
        for point in curve.keyframe_points:
            point.interpolation = "LINEAR"


def floor(owner, name, target, *, location, use_rotation, offset,
          owner_space, target_space, influence=1.0):
    constraint = owner.constraints.new("FLOOR")
    constraint.name = name
    constraint.target = target
    constraint.floor_location = location
    constraint.use_rotation = use_rotation
    constraint.offset = offset
    constraint.owner_space = owner_space
    constraint.target_space = target_space
    constraint.influence = influence
    return constraint


def maintain_volume(owner, name, *, mode, volume, free_axis="SAMEVOL_X", influence=1.0):
    constraint = owner.constraints.new("MAINTAIN_VOLUME")
    constraint.name = name
    constraint.mode = mode
    constraint.volume = volume
    constraint.free_axis = free_axis
    constraint.influence = influence
    return constraint


def column_major(matrix):
    values = [float(matrix[row][column]) for column in range(4) for row in range(4)]
    return [0.0 if round(value, 7) == 0 else round(value, 7) for value in values]


bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 3

owner_parent = empty("FloorOwnerParent", (0.65, -0.35, 0.8), (0.2, -0.3, 0.4), (1.2, 1.2, 1.2))
target_parent = empty("FloorTargetParent", (-0.45, 0.7, -0.2), (-0.25, 0.35, -0.15), (0.9, 0.9, 0.9))

floor_targets = [
    empty("FloorTargetWorld", (0, 0, 0), (0, 0, 0), (1, 1, 1)),
    empty("FloorTargetLocal", (0, 0, 0), (0, 0, 0), (1, 1, 1)),
]
floor_targets[1].parent = target_parent
animate(floor_targets[0], [
    (1, (1.1, -0.6, 0.3), (0.28, -0.42, 0.36), (1.1, 0.8, 1.25)),
    (2, (-0.4, 0.9, 1.0), (-0.37, 0.19, 0.72), (0.8, 1.4, 0.65)),
    (3, (0.7, 0.2, -0.8), (0.58, -0.16, -0.45), (1.5, 0.75, 1.1)),
])
animate(floor_targets[1], [
    (1, (-0.8, 0.3, 0.6), (-0.22, 0.48, -0.31), (0.9, 1.15, 0.7)),
    (2, (0.5, -1.0, 0.2), (0.41, -0.27, 0.63), (1.35, 0.65, 1.2)),
    (3, (0.25, 0.75, -0.4), (-0.52, 0.14, 0.29), (0.7, 1.3, 0.95)),
])

floor_owners = []
for name, location, rotation, scale in [
    ("FloorNoRotation", (-0.4, 0.6, 1.1), (0.18, -0.33, 0.27), (0.9, 1.2, 0.75)),
    ("FloorRotatedNegative", (0.8, -0.5, 0.4), (-0.31, 0.26, 0.45), (1.3, 0.7, 1.1)),
    ("FloorLocalParented", (-0.2, 0.9, -0.3), (0.24, 0.11, -0.38), (0.8, 1.15, 1.4)),
]:
    owner = empty(name, location, rotation, scale)
    if name != "FloorRotatedNegative":
        owner.parent = owner_parent
    floor_owners.append(owner)

animate(floor_owners[0], [
    (1, (-0.4, 0.6, 1.1), (0.18, -0.33, 0.27), (0.9, 1.2, 0.75)),
    (2, (-0.1, 0.2, 0.6), (0.31, -0.22, 0.12), (1.1, 0.85, 1.3)),
    (3, (0.35, -0.4, 0.9), (-0.14, 0.37, -0.26), (0.75, 1.4, 0.95)),
])
animate(floor_owners[1], [
    (1, (0.8, -0.5, 0.4), (-0.31, 0.26, 0.45), (1.3, 0.7, 1.1)),
    (2, (0.45, 0.25, 0.9), (0.16, -0.39, 0.58), (0.85, 1.25, 0.95)),
    (3, (-0.3, 0.75, 0.2), (-0.42, 0.18, -0.11), (1.15, 0.9, 1.35)),
])
animate(floor_owners[2], [
    (1, (-0.2, 0.9, -0.3), (0.24, 0.11, -0.38), (0.8, 1.15, 1.4)),
    (2, (0.4, 0.3, 0.5), (-0.15, 0.32, 0.2), (1.25, 0.9, 0.7)),
    (3, (-0.6, -0.1, 0.8), (0.35, -0.24, 0.41), (0.7, 1.35, 1.05)),
])
floor(floor_owners[0], "Ignore Rotated Target", floor_targets[0],
      location="FLOOR_X", use_rotation=False, offset=0.2,
      owner_space="WORLD", target_space="WORLD")
floor(floor_owners[1], "Rotated Negative Y Floor", floor_targets[1],
      location="FLOOR_NEGATIVE_Y", use_rotation=True, offset=-0.35,
      owner_space="LOCAL", target_space="LOCAL", influence=0.65)
floor(floor_owners[2], "Local Parent Negative Z Floor", floor_targets[0],
      location="FLOOR_NEGATIVE_Z", use_rotation=True, offset=0.45,
      owner_space="LOCAL", target_space="WORLD")

volume_owners = []
for name, location, rotation, scale in [
    ("VolumeStrictObject", (-1.1, 0.3, 0.7), (0.22, -0.15, 0.31), (1.2, 0.8, 1.45)),
    ("VolumeUniformObject", (0.5, -0.9, 0.2), (-0.32, 0.4, -0.18), (0.7, 1.3, 0.95)),
    ("VolumeSingleXObject", (0.9, 0.4, -0.6), (0.16, 0.23, -0.37), (1.45, 0.75, 1.1)),
    ("VolumeSingleZObject", (-0.6, -0.2, 1.0), (-0.27, -0.35, 0.29), (0.8, 1.5, 0.65)),
]:
    owner = empty(name, location, rotation, scale)
    volume_owners.append(owner)

for owner, keys in zip(volume_owners, [
    [
        (1, (-1.1, 0.3, 0.7), (0.22, -0.15, 0.31), (1.2, 0.8, 1.45)),
        (2, (-0.7, 0.8, 0.2), (-0.1, 0.32, -0.2), (1.55, 0.65, 1.05)),
        (3, (-0.2, -0.4, 0.9), (0.35, -0.26, 0.14), (0.85, 1.4, 1.25)),
    ],
    [
        (1, (0.5, -0.9, 0.2), (-0.32, 0.4, -0.18), (0.7, 1.3, 0.95)),
        (2, (0.2, -0.3, 0.7), (0.26, -0.12, 0.37), (1.25, 0.8, 1.4)),
        (3, (0.8, 0.4, -0.1), (-0.21, 0.35, -0.44), (0.9, 1.55, 0.75)),
    ],
    [
        (1, (0.9, 0.4, -0.6), (0.16, 0.23, -0.37), (1.45, 0.75, 1.1)),
        (2, (0.1, 0.75, -0.2), (-0.3, 0.18, 0.25), (0.75, 1.3, 1.6)),
        (3, (-0.4, 0.2, 0.65), (0.28, -0.41, 0.19), (1.2, 0.9, 0.7)),
    ],
    [
        (1, (-0.6, -0.2, 1.0), (-0.27, -0.35, 0.29), (0.8, 1.5, 0.65)),
        (2, (-0.1, -0.7, 0.3), (0.32, 0.2, -0.16), (1.4, 0.7, 1.15)),
        (3, (0.35, -0.25, 0.6), (-0.19, 0.38, 0.43), (0.65, 1.25, 1.5)),
    ],
]):
    animate(owner, keys)

maintain_volume(volume_owners[0], "Strict Preserve Volume", mode="STRICT", volume=0.8)
maintain_volume(volume_owners[1], "Uniform Preserve Volume", mode="UNIFORM", volume=0.45)
maintain_volume(volume_owners[2], "Single Axis X Free", mode="SINGLE_AXIS", volume=1.0, free_axis="SAMEVOL_X")
maintain_volume(volume_owners[3], "Single Axis Z Free", mode="SINGLE_AXIS", volume=0.7, free_axis="SAMEVOL_Z")

armature_data = bpy.data.armatures.new("OwnerRigData")
armature = bpy.data.objects.new("OwnerRig", armature_data)
scene.collection.objects.link(armature)
set_transform(armature, (0.3, 0.7, -0.5), (0.25, -0.2, 0.4), (1.1, 1.1, 1.1))
bpy.context.view_layer.objects.active = armature
armature.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
for name, head, tail in [
    ("FloorBone", (-0.8, -0.1, 0.0), (-0.2, 0.5, 0.25)),
    ("VolumeStrictBone", (0.25, -0.2, 0.35), (0.85, 0.35, 0.7)),
    ("VolumeSingleYBone", (0.95, 0.1, -0.25), (1.5, 0.55, 0.1)),
]:
    edit_bone = armature_data.edit_bones.new(name)
    edit_bone.head = head
    edit_bone.tail = tail
bpy.ops.object.mode_set(mode="OBJECT")

floor_bone = armature.pose.bones["FloorBone"]
volume_strict_bone = armature.pose.bones["VolumeStrictBone"]
volume_single_bone = armature.pose.bones["VolumeSingleYBone"]
for bone, keys in zip((floor_bone, volume_strict_bone, volume_single_bone), [
    [
        (1, (0.15, -0.2, 0.35), (0.18, -0.31, 0.24), (0.9, 1.2, 0.75)),
        (2, (-0.25, 0.35, 0.1), (-0.28, 0.16, 0.43), (1.25, 0.8, 1.1)),
        (3, (0.4, 0.05, -0.3), (0.32, -0.19, -0.22), (0.7, 1.35, 0.95)),
    ],
    [
        (1, (0.1, 0.2, -0.15), (-0.2, 0.35, 0.17), (1.25, 0.8, 1.4)),
        (2, (-0.3, 0.1, 0.25), (0.31, -0.18, 0.4), (0.75, 1.45, 0.85)),
        (3, (0.2, -0.25, 0.4), (-0.12, 0.29, -0.35), (1.5, 0.7, 1.1)),
    ],
    [
        (1, (-0.2, 0.3, 0.1), (0.27, -0.14, 0.38), (0.8, 1.35, 0.9)),
        (2, (0.35, -0.1, 0.25), (-0.22, 0.36, -0.16), (1.4, 0.75, 1.2)),
        (3, (0.1, 0.45, -0.2), (0.39, 0.18, -0.3), (0.7, 1.5, 0.8)),
    ],
]):
    animate_bone(armature, bone, keys)

floor(floor_bone, "Pose Bone Negative X Floor", floor_targets[1],
      location="FLOOR_NEGATIVE_X", use_rotation=False, offset=0.3,
      owner_space="LOCAL", target_space="LOCAL")
maintain_volume(volume_strict_bone, "Strict Bone Preserve Volume", mode="STRICT", volume=0.6)
maintain_volume(volume_single_bone, "Single Axis Y Free Partial", mode="SINGLE_AXIS", volume=0.9,
                free_axis="SAMEVOL_Y", influence=0.55)

owners = [
    {"kind": "object", "name": owner.name}
    for owner in floor_owners + volume_owners
] + [
    {"kind": "bone", "armature": armature.name, "name": bone.name}
    for bone in (floor_bone, volume_strict_bone, volume_single_bone)
]
assert len(owners) == 10
assert all(owner.constraints and owner.constraints[0].type == "FLOOR" for owner in floor_owners)
assert all(owner.constraints and owner.constraints[0].type == "MAINTAIN_VOLUME" for owner in volume_owners)
assert floor_bone.constraints[0].type == "FLOOR"
assert all(bone.constraints[0].type == "MAINTAIN_VOLUME" for bone in (volume_strict_bone, volume_single_bone))

captured = {}
for frame in (1, 2, 3):
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    frame_values = {}
    for owner in floor_owners + volume_owners:
        frame_values[owner.name] = column_major(owner.evaluated_get(depsgraph).matrix_world)
    evaluated_armature = armature.evaluated_get(depsgraph)
    for bone in (floor_bone, volume_strict_bone, volume_single_bone):
        key = f"Bone:{armature.name}:{bone.name}"
        evaluated_bone = evaluated_armature.pose.bones[bone.name]
        frame_values[key] = column_major(evaluated_armature.matrix_world @ evaluated_bone.matrix)
    captured[str(frame)] = frame_values

scene.frame_set(1)
bpy.context.view_layer.update()
bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))
with JSON_PATH.open("w", encoding="utf-8", newline="\n") as output:
    json.dump({"owners": owners, "frames": captured}, output, separators=(",", ":"), allow_nan=False)
    output.write("\n")
