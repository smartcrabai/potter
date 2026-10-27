import json
from pathlib import Path

import bpy
from mathutils import Euler


if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"fixture requires Blender 5.2.2, found {bpy.app.version_string}")

DIRECTORY = Path(__file__).resolve().parent
BLEND_PATH = DIRECTORY / "copy_scale.blend"
JSON_PATH = DIRECTORY / "copy_scale.json"


def set_transform(obj, location, rotation, scale):
    obj.location = location
    obj.rotation_euler = Euler(rotation, "XYZ")
    obj.scale = scale


def empty(name, location, rotation, scale):
    obj = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(obj)
    set_transform(obj, location, rotation, scale)
    return obj


def animate_target(obj, keys):
    for frame, location, rotation, scale in keys:
        set_transform(obj, location, rotation, scale)
        for channel in ("location", "rotation_euler", "scale"):
            obj.keyframe_insert(data_path=channel, frame=frame)


def copy_scale(owner, name, target, *, owner_space, target_space, power, uniform,
               offset, add, axes, influence):
    constraint = owner.constraints.new("COPY_SCALE")
    constraint.name = name
    constraint.target = target
    constraint.owner_space = owner_space
    constraint.target_space = target_space
    constraint.power = power
    constraint.use_make_uniform = uniform
    constraint.use_offset = offset
    constraint.use_add = add
    constraint.use_x, constraint.use_y, constraint.use_z = axes
    constraint.influence = influence
    return constraint


def column_major(matrix):
    values = [float(matrix[row][column]) for column in range(4) for row in range(4)]
    return [0.0 if round(value, 7) == 0 else round(value, 7) for value in values]


bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 3

owner_parent = empty("OwnerParent", (0.7, -0.4, 1.2), (0.25, -0.3, 0.45), (1.2, 1.2, 1.2))
target_parent_a = empty("TargetParentA", (-0.6, 0.9, 0.3), (-0.2, 0.35, -0.15), (0.9, 0.9, 0.9))
target_parent_b = empty("TargetParentB", (1.1, 0.2, -0.7), (0.4, -0.15, 0.3), (1.15, 1.15, 1.15))

target_a = empty("TargetA", (0, 0, 0), (0, 0, 0), (1, 1, 1))
target_a.parent = target_parent_a
animate_target(target_a, [
    (1, (1.4, -0.8, 0.6), (0.2, -0.35, 0.4), (1.7, 0.65, 1.3)),
    (2, (-0.3, 1.1, 0.8), (-0.4, 0.25, 0.9), (0.8, 2.1, 0.5)),
    (3, (0.9, 0.2, -0.6), (0.65, -0.1, -0.5), (2.4, 1.1, 0.9)),
])
target_b = empty("TargetB", (0, 0, 0), (0, 0, 0), (1, 1, 1))
target_b.parent = target_parent_b
animate_target(target_b, [
    (1, (-0.9, 0.5, 1.3), (-0.3, 0.55, 0.1), (0.55, 1.6, 2.2)),
    (2, (0.4, -1.2, 0.2), (0.5, -0.2, 0.75), (1.9, 0.8, 0.7)),
    (3, (1.2, 0.7, -0.4), (0.1, 0.35, -0.6), (1.1, 2.3, 1.4)),
])

object_owners = []
for name, location, rotation, scale in [
    ("ObjectPowerAxes", (-0.2, 0.6, 0.9), (0.3, -0.5, 0.2), (0.8, 1.4, 1.1)),
    ("ObjectUniformOffsetMul", (1.0, -0.3, 0.5), (-0.4, 0.2, 0.7), (1.3, 0.6, 1.7)),
    ("ObjectLegacyAdd", (-1.1, 0.4, -0.2), (0.15, 0.6, -0.35), (0.7, 1.2, 1.5)),
]:
    owner = empty(name, location, rotation, scale)
    owner.parent = owner_parent
    object_owners.append(owner)

copy_scale(
    object_owners[0], "Power XZ in World", target_a,
    owner_space="WORLD", target_space="WORLD", power=2.0, uniform=False,
    offset=False, add=False, axes=(True, False, True), influence=1.0,
)
copy_scale(
    object_owners[1], "Uniform Y Multiplicative Offset", target_a,
    owner_space="LOCAL", target_space="LOCAL", power=1.4, uniform=True,
    offset=True, add=False, axes=(False, True, False), influence=0.65,
)
copy_scale(
    object_owners[2], "Legacy Additive XZ Offset", target_b,
    owner_space="LOCAL", target_space="WORLD", power=1.6, uniform=False,
    offset=True, add=True, axes=(True, False, True), influence=0.8,
)

armature_data = bpy.data.armatures.new("OwnerRigData")
armature = bpy.data.objects.new("OwnerRig", armature_data)
scene.collection.objects.link(armature)
set_transform(armature, (0.3, 1.0, -0.8), (0.35, -0.25, 0.5), (1.2, 1.2, 1.2))
bpy.context.view_layer.objects.active = armature
armature.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
for name, head, tail in [
    ("ScaleBoneA", (-0.8, 0.0, 0.0), (-0.1, 0.65, 0.3)),
    ("ScaleBoneB", (0.35, -0.2, 0.45), (0.9, 0.55, 0.8)),
]:
    bone = armature_data.edit_bones.new(name)
    bone.head = head
    bone.tail = tail
bpy.ops.object.mode_set(mode="OBJECT")

bone_a = armature.pose.bones["ScaleBoneA"]
bone_a.location = (0.2, -0.15, 0.35)
bone_a.rotation_mode = "XYZ"
bone_a.rotation_euler = (0.25, -0.4, 0.3)
bone_a.scale = (0.85, 1.25, 0.7)
copy_scale(
    bone_a, "Uniform Additive XZ Offset", target_b,
    owner_space="POSE", target_space="WORLD", power=0.75, uniform=True,
    offset=True, add=True, axes=(True, False, True), influence=0.55,
)
bone_b = armature.pose.bones["ScaleBoneB"]
bone_b.location = (-0.3, 0.4, 0.1)
bone_b.rotation_mode = "XYZ"
bone_b.rotation_euler = (-0.2, 0.3, -0.45)
bone_b.scale = (1.35, 0.75, 1.1)
copy_scale(
    bone_b, "Local Y Multiplicative Offset", target_a,
    owner_space="LOCAL_WITH_PARENT", target_space="LOCAL", power=1.8, uniform=False,
    offset=True, add=False, axes=(False, True, False), influence=0.7,
)

owners = [
    {"kind": "object", "name": owner.name} for owner in object_owners
] + [
    {"kind": "bone", "armature": armature.name, "name": bone.name}
    for bone in (bone_a, bone_b)
]
assert len(object_owners) == 3 and len(owners) == 5
assert all(owner.constraints and owner.constraints[0].type == "COPY_SCALE" for owner in object_owners)
assert all(bone.constraints and bone.constraints[0].type == "COPY_SCALE" for bone in (bone_a, bone_b))

captured = {}
for frame in (1, 2, 3):
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    frame_values = {}
    for owner in object_owners:
        evaluated = owner.evaluated_get(depsgraph)
        frame_values[owner.name] = column_major(evaluated.matrix_world)
    evaluated_armature = armature.evaluated_get(depsgraph)
    for bone in (bone_a, bone_b):
        evaluated_bone = evaluated_armature.pose.bones[bone.name]
        key = f"Bone:{armature.name}:{bone.name}"
        frame_values[key] = column_major(evaluated_armature.matrix_world @ evaluated_bone.matrix)
    captured[str(frame)] = frame_values

scene.frame_set(1)
bpy.context.view_layer.update()
bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))
with JSON_PATH.open("w", encoding="utf-8", newline="\n") as output:
    json.dump({"owners": owners, "frames": captured}, output, separators=(",", ":"), allow_nan=False)
    output.write("\n")
