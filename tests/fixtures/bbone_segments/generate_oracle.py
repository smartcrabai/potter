import bpy
import json
import os
import sys

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
case = sys.argv[sys.argv.index("--") + 2]
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

armature = bpy.data.armatures.new("BboneRigData")
rig = bpy.data.objects.new("Rig", armature)
scene.collection.objects.link(rig)
rig.select_set(True)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="EDIT")
root_bone = armature.edit_bones.new("Root")
root_bone.head = (0.0, -1.0, 0.0)
root_bone.tail = (0.0, 0.0, 0.0)
bendy = armature.edit_bones.new("Bendy")
bendy.head = (0.0, 0.0, 0.0)
bendy.tail = (0.0, 2.0, 0.0)
bendy.parent = root_bone
start_handle = armature.edit_bones.new("StartHandle")
start_handle.head = (-0.6, 0.2, 0.0)
start_handle.tail = (-0.6, 0.7, 0.0)
end_handle = armature.edit_bones.new("EndHandle")
end_handle.head = (0.6, 1.3, 0.0)
end_handle.tail = (0.6, 1.8, 0.0)
bpy.ops.object.mode_set(mode="OBJECT")
rig.select_set(False)

bone = armature.bones["Bendy"]
bone.bbone_segments = 5
bone.bbone_mapping_mode = "CURVED"
bone.bbone_x = 0.17
bone.bbone_z = 0.23
bone.bbone_handle_type_start = "TANGENT"
bone.bbone_custom_handle_start = armature.bones["StartHandle"]
bone.bbone_handle_use_scale_start = (True, False, True)
bone.bbone_handle_use_ease_start = True
bone.bbone_handle_type_end = "ABSOLUTE"
bone.bbone_custom_handle_end = armature.bones["EndHandle"]
bone.bbone_handle_use_scale_end = (False, True, False)
bone.bbone_handle_use_ease_end = False
bone.bbone_rollin = 0.13
bone.bbone_rollout = -0.21
bone.bbone_curveinx = 0.11
bone.bbone_curveinz = 0.24
bone.bbone_curveoutx = -0.14
bone.bbone_curveoutz = -0.27
bone.bbone_easein = 0.31
bone.bbone_easeout = 0.42
bone.bbone_scalein = (1.1, 0.8, 1.2)
bone.bbone_scaleout = (0.9, 1.3, 0.7)

rig.location = (0.31, -0.22, 0.18)
rig.rotation_euler = (0.11, -0.16, 0.09)
bendy_pose = rig.pose.bones["Bendy"]
bendy_pose.rotation_mode = "XYZ"
bendy_pose.rotation_euler = (0.19, -0.23, 0.14)
bendy_pose.location = (0.07, 0.11, -0.05)
bendy_pose.scale = (1.08, 0.93, 1.12)
bendy_pose.bbone_rollin = 0.27
start_handle_pose = rig.pose.bones["StartHandle"]
start_handle_pose.rotation_mode = "XYZ"
start_handle_pose.rotation_euler = (0.12, -0.18, 4.0)
end_handle_pose = rig.pose.bones["EndHandle"]
end_handle_pose.rotation_mode = "XYZ"
end_handle_pose.rotation_euler = (-0.14, 0.1, -0.12)

def armature_point_with_segment_blend(pose_bone, target_segment, target_blend):
    best_error = float("inf")
    best_point = None
    best_blend = None
    for segment in range(pose_bone.bone.bbone_segments):
        first = pose_bone.bbone_segment_matrix(segment).translation
        second = pose_bone.bbone_segment_matrix(segment + 1).translation
        for step in range(101):
            local_point = first.lerp(second, step / 100.0)
            armature_point = pose_bone.matrix @ local_point
            index, blend = pose_bone.bbone_segment_index(armature_point)
            error = abs(blend - target_blend) if index == target_segment else float("inf")
            if error < best_error:
                best_error = error
                best_point = armature_point
                best_blend = blend
    if best_point is None or best_error > 0.02:
        raise RuntimeError("could not find an owner point at the requested B-Bone segment blend")
    return best_point, best_blend


owners = {}
armature_target_blend = None
if case == "target_shape":
    head_tails = (0.0, 0.13, 0.35, 0.61, 0.83, 1.0)
    for index, head_tail in enumerate(head_tails):
        name = "CaseOwner" if head_tail == 0.35 else f"ShapeSweep_{index}"
        owner = bpy.data.objects.new(name, None)
        scene.collection.objects.link(owner)
        owner.location = (0.2, -0.4, 0.1)
        owner.rotation_euler = (0.07, -0.13, 0.17)
        owner.scale = (1.05, 0.9, 1.15)
        constraint = owner.constraints.new("COPY_LOCATION")
        constraint.name = "BboneShapeTarget" if name == "CaseOwner" else f"ShapeTarget_{index}"
        constraint.target = rig
        constraint.subtarget = "Bendy"
        constraint.head_tail = head_tail
        constraint.use_bbone_shape = True
        owners[name] = owner
    for index, head_tail in enumerate(head_tails):
        name = f"TransformSweep_{index}"
        owner = bpy.data.objects.new(name, None)
        scene.collection.objects.link(owner)
        owner.location = (0.2, -0.4, 0.1)
        owner.rotation_euler = (0.07, -0.13, 0.17)
        owner.scale = (1.05, 0.9, 1.15)
        constraint = owner.constraints.new("COPY_TRANSFORMS")
        constraint.mix_mode = "REPLACE"
        constraint.name = f"TransformTarget_{index}"
        constraint.target = rig
        constraint.subtarget = "Bendy"
        constraint.head_tail = head_tail
        constraint.use_bbone_shape = True
        owners[name] = owner
elif case in (
    "armature_constraint",
    "armature_constraint_scale_2",
    "armature_constraint_scale_0_01",
):
    scale = {
        "armature_constraint_scale_2": 2.0,
        "armature_constraint_scale_0_01": 0.01,
    }.get(case)
    if scale is not None:
        rig.scale = (scale, scale, scale)
    scene.frame_set(1)
    bpy.context.view_layer.update()
    armature_point, armature_target_blend = armature_point_with_segment_blend(
        rig.pose.bones["Bendy"], 2, 0.5
    )
    world_point = rig.matrix_world @ armature_point
    positions = (
        (-0.2, -0.1, 0.15),
        (0.15, 0.45, -0.2),
        tuple(world_point),
        (0.28, 1.55, -0.24),
        (0.1, 2.2, 0.1),
    )
    for index, position in enumerate(positions):
        name = "CaseOwner" if index == 2 else f"ArmatureSweep_{index}"
        owner = bpy.data.objects.new(name, None)
        scene.collection.objects.link(owner)
        owner.location = position
        owner.rotation_euler = (0.12, -0.17, 0.21)
        owner.scale = (1.1, 0.9, 1.04)
        constraint = owner.constraints.new("ARMATURE")
        constraint.name = "WeightedBboneTarget" if name == "CaseOwner" else f"ArmatureTarget_{index}"
        constraint.use_deform_preserve_volume = True
        plain_target = constraint.targets.new()
        plain_target.target = rig
        plain_target.subtarget = "Root"
        plain_target.weight = 0.35
        segmented_target = constraint.targets.new()
        segmented_target.target = rig
        segmented_target.subtarget = "Bendy"
        segmented_target.weight = 0.65
        owners[name] = owner
elif case == "armature_constraint_zero_weight":
    bone.use_envelope_multiply = True
    bone.head_radius = 0.2
    bone.tail_radius = 0.2
    bone.envelope_distance = 0.2
    bone.envelope_weight = 1.0
    scene.frame_set(1)
    bpy.context.view_layer.update()
    owner = bpy.data.objects.new("CaseOwner", None)
    scene.collection.objects.link(owner)
    owner.location = (100.0, 100.0, 100.0)
    owner.rotation_euler = (0.12, -0.17, 0.21)
    owner.scale = (1.1, 0.9, 1.04)
    constraint = owner.constraints.new("ARMATURE")
    constraint.name = "WeightedBboneTarget"
    constraint.use_bone_envelopes = False
    constraint.use_deform_preserve_volume = True
    target = constraint.targets.new()
    target.target = rig
    target.subtarget = "Bendy"
    target.weight = 1.0
    owners["CaseOwner"] = owner
else:
    raise ValueError("unknown B-Bone oracle case: " + case)

scene.frame_set(1)
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()

def world_matrix(obj):
    matrix = obj.evaluated_get(depsgraph).matrix_world
    return [float(matrix[row][column]) for column in range(4) for row in range(4)]

oracle = {"owners": {name: world_matrix(owner) for name, owner in owners.items()}}
if armature_target_blend is not None:
    oracle["armature_target_blend"] = armature_target_blend
with open(os.path.join(root, "oracle.json"), "w", encoding="utf-8") as output:
    json.dump(oracle, output)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, "source.blend"), check_existing=False)
