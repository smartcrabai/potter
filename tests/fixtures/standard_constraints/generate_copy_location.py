import json
from pathlib import Path

import bpy
from mathutils import Euler


if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"fixture requires Blender 5.2.2, found {bpy.app.version_string}")

DIRECTORY = Path(__file__).resolve().parent
BLEND_PATH = DIRECTORY / "copy_location.blend"
JSON_PATH = DIRECTORY / "copy_location.json"


def set_transform(owner, location, rotation, scale):
    owner.location = location
    owner.rotation_euler = Euler(rotation, "XYZ")
    owner.scale = scale


def empty(name, location, rotation, scale):
    owner = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(owner)
    set_transform(owner, location, rotation, scale)
    return owner


def animate(owner, keys):
    for frame, location, rotation, scale in keys:
        set_transform(owner, location, rotation, scale)
        for channel in ("location", "rotation_euler", "scale"):
            owner.keyframe_insert(data_path=channel, frame=frame)


def animate_bone(bone, keys):
    bone.rotation_mode = "XYZ"
    for frame, location, rotation, scale in keys:
        bone.location = location
        bone.rotation_euler = rotation
        bone.scale = scale
        for channel in ("location", "rotation_euler", "scale"):
            bone.keyframe_insert(data_path=channel, frame=frame)


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
        edit_bone = data.edit_bones.new(bone_name)
        edit_bone.head = head
        edit_bone.tail = tail
    bpy.ops.object.mode_set(mode="OBJECT")
    return rig


def copy_location(owner, name, target, *, owner_space, target_space, axes,
                  invert, offset, influence, subtarget=""):
    constraint = owner.constraints.new("COPY_LOCATION")
    constraint.name = name
    constraint.target = target
    constraint.subtarget = subtarget
    constraint.owner_space = owner_space
    constraint.target_space = target_space
    constraint.use_x, constraint.use_y, constraint.use_z = axes
    constraint.invert_x, constraint.invert_y, constraint.invert_z = invert
    constraint.use_offset = offset
    constraint.influence = influence
    return constraint


def column_major(matrix):
    values = [float(matrix[row][column]) for column in range(4) for row in range(4)]
    return [0.0 if round(value, 7) == 0 else round(value, 7) for value in values]


bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 3

owner_parent = empty("LocationOwnerParent", (0.65, -0.4, 1.1), (0.25, -0.35, 0.2), (1.2, 1.2, 1.2))
target_parent = empty("LocationTargetParent", (-0.55, 0.75, -0.3), (-0.2, 0.3, -0.15), (0.9, 0.9, 0.9))

target = empty("LocationTarget", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
target.parent = target_parent
animate(target, [
    (1, (1.25, -0.7, 0.45), (0.25, -0.4, 0.35), (1.3, 0.75, 1.6)),
    (2, (-0.6, 1.15, 0.8), (-0.35, 0.55, -0.2), (0.7, 1.8, 0.9)),
    (3, (0.9, 0.3, -0.65), (0.6, -0.15, 0.75), (1.9, 0.65, 1.25)),
])

target_rig = make_rig(
    "LocationTargetRig", (-0.3, 0.8, 0.55), (0.3, -0.2, 0.45), (1.1, 1.1, 1.1),
    [("LocationTargetBone", (-0.35, 0.0, 0.0), (0.1, 0.65, 0.25))],
)
target_bone = target_rig.pose.bones["LocationTargetBone"]
animate_bone(target_bone, [
    (1, (0.2, -0.15, 0.3), (0.2, -0.35, 0.15), (1.0, 1.2, 0.8)),
    (2, (-0.25, 0.35, 0.1), (-0.3, 0.25, 0.5), (1.25, 0.75, 1.1)),
    (3, (0.4, 0.1, -0.2), (0.45, 0.15, -0.4), (0.85, 1.4, 0.95)),
])

object_owners = []
for name, keys in [
    ("LocationAllAxesInverted", [
        (1, (-0.35, 0.6, 0.85), (0.25, -0.45, 0.3), (0.9, 1.25, 0.75)),
        (2, (0.55, -0.2, 0.4), (-0.3, 0.2, 0.6), (1.2, 0.8, 1.35)),
        (3, (-0.1, 0.75, -0.45), (0.5, -0.1, -0.35), (0.7, 1.4, 1.05)),
    ]),
    ("LocationSelectedAxesOffset", [
        (1, (0.8, -0.45, 0.2), (-0.2, 0.35, -0.4), (1.15, 0.9, 1.3)),
        (2, (-0.4, 0.65, 0.75), (0.4, -0.25, 0.15), (0.85, 1.3, 0.7)),
        (3, (0.3, -0.8, 0.55), (-0.15, 0.5, 0.25), (1.4, 0.75, 1.1)),
    ]),
    ("LocationPoseTarget", [
        (1, (-0.7, 0.25, 0.5), (0.35, 0.15, -0.2), (0.8, 1.1, 1.45)),
        (2, (0.15, -0.55, 0.95), (-0.25, 0.45, 0.3), (1.3, 0.7, 0.9)),
        (3, (0.65, 0.4, -0.35), (0.55, -0.3, 0.1), (0.95, 1.5, 0.8)),
    ]),
]:
    owner = empty(name, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    owner.parent = owner_parent
    animate(owner, keys)
    object_owners.append(owner)

constraints = [
    copy_location(
        object_owners[0], "All Axes Inverted Partial", target,
        owner_space="LOCAL", target_space="LOCAL", axes=(True, True, True),
        invert=(True, True, True), offset=False, influence=0.45,
    ),
    copy_location(
        object_owners[1], "Selected XZ Additive Offset", target,
        owner_space="LOCAL", target_space="WORLD", axes=(True, False, True),
        invert=(False, False, True), offset=True, influence=0.8,
    ),
    copy_location(
        object_owners[2], "Pose Bone Target Y Inverted", target_rig,
        owner_space="WORLD", target_space="LOCAL", axes=(False, True, False),
        invert=(False, True, False), offset=False, influence=0.65,
        subtarget=target_bone.name,
    ),
]

owner_rig = make_rig(
    "LocationOwnerRig", (0.4, -0.6, 0.9), (0.25, 0.35, -0.3), (1.1, 1.1, 1.1),
    [("LocationOwnerBone", (-0.45, 0.0, 0.0), (0.0, 0.7, 0.3))],
)
owner_bone = owner_rig.pose.bones["LocationOwnerBone"]
animate_bone(owner_bone, [
    (1, (0.15, -0.3, 0.45), (0.2, -0.4, 0.25), (0.9, 1.2, 0.8)),
    (2, (-0.35, 0.2, 0.1), (-0.3, 0.35, 0.5), (1.25, 0.75, 1.1)),
    (3, (0.5, 0.45, -0.25), (0.45, 0.15, -0.2), (0.8, 1.35, 0.95)),
])
constraints.append(copy_location(
    owner_bone, "Pose Owner Object Target Offset", target,
    owner_space="LOCAL", target_space="WORLD", axes=(True, False, True),
    invert=(True, False, False), offset=True, influence=0.6,
))

owners = [{"kind": "object", "name": owner.name} for owner in object_owners] + [
    {"kind": "bone", "armature": owner_rig.name, "name": owner_bone.name}
]
assert len(object_owners) >= 2 and len(owners) == 4
assert all(constraint.type == "COPY_LOCATION" for constraint in constraints)
assert constraints[2].target == target_rig and constraints[2].subtarget == target_bone.name
assert constraints[3].target == target and constraints[3].subtarget == ""
assert any(constraint.use_offset for constraint in constraints)
assert any(not constraint.use_offset for constraint in constraints)
assert any(constraint.influence < 1.0 for constraint in constraints)

captured = {}
for frame in (1, 2, 3):
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    frame_values = {}
    for owner in object_owners:
        frame_values[owner.name] = column_major(owner.evaluated_get(depsgraph).matrix_world)
    evaluated_armature = owner_rig.evaluated_get(depsgraph)
    evaluated_bone = evaluated_armature.pose.bones[owner_bone.name]
    key = f"Bone:{owner_rig.name}:{owner_bone.name}"
    frame_values[key] = column_major(evaluated_armature.matrix_world @ evaluated_bone.matrix)
    assert len(frame_values) == len(owners)
    captured[str(frame)] = frame_values

scene.frame_set(1)
bpy.context.view_layer.update()
bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))
with JSON_PATH.open("w", encoding="utf-8", newline="\n") as output:
    json.dump({"owners": owners, "frames": captured}, output, separators=(",", ":"), allow_nan=False)
    output.write("\n")
