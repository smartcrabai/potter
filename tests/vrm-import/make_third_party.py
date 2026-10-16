# Generates a third-party-style VRM with the VRM Add-on for Blender (4.10.2) through its own exporter.
#
# Run (the add-on must already be installed into BLENDER_USER_RESOURCES, see tests/vrm_import.rs):
#   blender -b --factory-startup --python-exit-code 1 --python tests/vrm-import/make_third_party.py -- <out.vrm> <1.0|0.0>
#
# The scene is fixed so the Rust test can assert the values it imports:
#   humanoid bones (snake_case Blender names, VRM role per bone), one skinned mesh "body" with shape keys
#   "smile", "blink", "wink", the add-on's MToon material "skin", a native Copy Rotation bone constraint
#   (exported as VRMC_node_constraint), a hair spring chain hair_root -> hair_mid -> hair_tip with one sphere
#   collider on "head" in a collider group, and the VRM 0.x equivalents (blend shape groups, secondary animation).
import sys
from uuid import uuid4

import bpy
from mathutils import Matrix, Vector

MODULE = "bl_ext.user_default.vrm"
ARGS = sys.argv[sys.argv.index("--") + 1 :]
OUT_PATH = ARGS[0]
SPEC_VERSION = ARGS[1]

# VRM role (camelCase, as in both VRM specs) -> Blender bone name (snake_case; Potter ids are lowercased names).
HUMANOID = {
    "hips": "hips",
    "spine": "spine",
    "chest": "chest",
    "neck": "neck",
    "head": "head",
    "leftUpperArm": "left_upper_arm",
    "leftLowerArm": "left_lower_arm",
    "leftHand": "left_hand",
    "rightUpperArm": "right_upper_arm",
    "rightLowerArm": "right_lower_arm",
    "rightHand": "right_hand",
    "leftUpperLeg": "left_upper_leg",
    "leftLowerLeg": "left_lower_leg",
    "leftFoot": "left_foot",
    "rightUpperLeg": "right_upper_leg",
    "rightLowerLeg": "right_lower_leg",
    "rightFoot": "right_foot",
}
HAIR_CHAIN = ["hair_root", "hair_mid", "hair_tip"]
# bone name -> (head, tail, parent)
BONES = {
    "hips": ((0.0, 0.0, 1.0), (0.0, 0.0, 1.1), None),
    "spine": ((0.0, 0.0, 1.1), (0.0, 0.0, 1.3), "hips"),
    "chest": ((0.0, 0.0, 1.3), (0.0, 0.0, 1.5), "spine"),
    "neck": ((0.0, 0.0, 1.5), (0.0, 0.0, 1.6), "chest"),
    "head": ((0.0, 0.0, 1.6), (0.0, 0.0, 1.8), "neck"),
    "left_upper_arm": ((0.2, 0.0, 1.45), (0.5, 0.0, 1.45), "chest"),
    "left_lower_arm": ((0.5, 0.0, 1.45), (0.7, 0.0, 1.45), "left_upper_arm"),
    "left_hand": ((0.7, 0.0, 1.45), (0.8, 0.0, 1.45), "left_lower_arm"),
    "right_upper_arm": ((-0.2, 0.0, 1.45), (-0.5, 0.0, 1.45), "chest"),
    "right_lower_arm": ((-0.5, 0.0, 1.45), (-0.7, 0.0, 1.45), "right_upper_arm"),
    "right_hand": ((-0.7, 0.0, 1.45), (-0.8, 0.0, 1.45), "right_lower_arm"),
    "left_upper_leg": ((0.1, 0.0, 1.0), (0.1, 0.0, 0.5), "hips"),
    "left_lower_leg": ((0.1, 0.0, 0.5), (0.1, 0.0, 0.1), "left_upper_leg"),
    "left_foot": ((0.1, 0.0, 0.1), (0.1, 0.1, 0.0), "left_lower_leg"),
    "right_upper_leg": ((-0.1, 0.0, 1.0), (-0.1, 0.0, 0.5), "hips"),
    "right_lower_leg": ((-0.1, 0.0, 0.5), (-0.1, 0.0, 0.1), "right_upper_leg"),
    "right_foot": ((-0.1, 0.0, 0.1), (-0.1, 0.1, 0.0), "right_lower_leg"),
    "hair_root": ((0.0, 0.0, 1.8), (0.0, -0.05, 1.9), "head"),
    "hair_mid": ((0.0, -0.05, 1.9), (0.0, -0.1, 2.0), "hair_root"),
    "hair_tip": ((0.0, -0.1, 2.0), (0.0, -0.15, 2.1), "hair_mid"),
    "accessory": ((0.1, 0.0, 1.7), (0.1, 0.0, 1.9), "head"),
}
COLLIDER_RADIUS = 0.05
# |offset| = 0.1: head-bone space for VRM 1.0 (the add-on writes it unchanged), world space offset from the
# head bone for VRM 0.x (the add-on writes the Unity-space form of that offset).
COLLIDER_OFFSET = (0.0, 0.0, 0.1)
JOINT_GRAVITY_DIR = (0.6, 0.0, -0.8)


def snake_attribute(role):
    out = ""
    for ch in role:
        if ch.isupper():
            out += "_" + ch.lower()
        else:
            out += ch
    return out


def build_armature():
    arm_data = bpy.data.armatures.new("Armature")
    arm = bpy.data.objects.new("Armature", arm_data)
    bpy.context.scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = arm_data.edit_bones
    for name, (head, tail, parent) in BONES.items():
        bone = edit_bones.new(name)
        bone.head = Vector(head)
        bone.tail = Vector(tail)
        if parent:
            bone.parent = edit_bones[parent]
    bpy.ops.object.mode_set(mode="OBJECT")
    return arm


def build_body(arm):
    bpy.ops.mesh.primitive_cube_add(size=0.2, location=(0.0, 0.0, 1.2))
    body = bpy.context.active_object
    body.name = "body"
    body.data.name = "body"
    body.parent = arm
    modifier = body.modifiers.new("Armature", "ARMATURE")
    modifier.object = arm
    group = body.vertex_groups.new(name="chest")
    group.add([v.index for v in body.data.vertices], 1.0, "REPLACE")
    body.shape_key_add(name="Basis", from_mix=False)
    for key, delta in (("smile", (0.0, 0.0, 0.02)), ("blink", (0.0, 0.02, 0.0)), ("wink", (0.02, 0.0, 0.0))):
        block = body.shape_key_add(name=key, from_mix=False)
        for vertex in block.data:
            vertex.co += Vector(delta)
    material = bpy.data.materials.new("skin")
    material.vrm_addon_extension.mtoon1.enabled = True
    body.data.materials.append(material)
    return body


def build_constraint(arm):
    pose_bone = arm.pose.bones["accessory"]
    constraint = pose_bone.constraints.new("COPY_ROTATION")
    constraint.target = arm
    constraint.subtarget = "neck"
    constraint.mix_mode = "ADD"
    constraint.owner_space = "LOCAL"
    constraint.target_space = "LOCAL"
    constraint.use_x = True
    constraint.use_y = True
    constraint.use_z = True


def build_vrm1(arm, ext):
    human = ext.vrm1.humanoid.human_bones
    human.initial_automatic_bone_assignment = False
    for role, bone in HUMANOID.items():
        getattr(human, snake_attribute(role)).node.bone_name = bone

    meta = ext.vrm1.meta
    meta.vrm_name = "Third Party Fixture"
    meta.version = "1.2.3"
    meta.authors.add().value = "Third Party Author"

    presets = ext.vrm1.expressions.preset
    for preset, key in (("happy", "smile"), ("blink", "blink")):
        bind = getattr(presets, preset).morph_target_binds.add()
        bind.node.mesh_object_name = "body"
        bind.index = key
        bind.weight = 1.0
    custom = ext.vrm1.expressions.custom.add()
    custom.custom_name = "wink"
    bind = custom.morph_target_binds.add()
    bind.node.mesh_object_name = "body"
    bind.index = "wink"
    bind.weight = 1.0

    spring = ext.spring_bone1
    collider = spring.add_collider(bpy.context, arm)
    collider.node.bone_name = "head"
    collider.shape_type = "Sphere"
    collider.shape.sphere.radius = COLLIDER_RADIUS
    collider.shape.sphere.offset = COLLIDER_OFFSET
    group = spring.add_collider_group()
    group.vrm_name = "Head"
    group.colliders.add().collider_uuid = collider.uuid
    chain = spring.add_spring()
    chain.vrm_name = "Hair"
    for bone in HAIR_CHAIN:
        joint = chain.joints.add()
        joint.node.bone_name = bone
        joint.hit_radius = 0.02
        joint.stiffness = 0.8
        joint.gravity_power = 1.1
        joint.gravity_dir = JOINT_GRAVITY_DIR
        joint.drag_force = 0.3
    chain.collider_groups.add().collider_group_uuid = group.uuid


def build_vrm0(arm, ext):
    human = ext.vrm0.humanoid.human_bones
    ext.vrm0.humanoid.initial_automatic_bone_assignment = False
    by_bone = {item.bone: item for item in human}
    for role, bone in HUMANOID.items():
        item = by_bone.get(role)
        if item is None:
            item = human.add()
            item.bone = role
        item.node.bone_name = bone

    meta = ext.vrm0.meta
    meta.title = "Third Party Fixture"
    meta.version = "1.2.3"
    meta.author = "Third Party Author"
    meta.reference = "https://example.test/reference"
    meta.allowed_user_name = "Everyone"
    meta.violent_ussage_name = "Disallow"
    meta.sexual_ussage_name = "Disallow"
    meta.commercial_ussage_name = "Disallow"
    meta.license_name = "CC_BY"
    meta.other_permission_url = "https://example.test/permissions"

    groups = ext.vrm0.blend_shape_master.blend_shape_groups
    for name, preset, key in (("Joy", "joy", "smile"), ("Blink", "blink", "blink"), ("wink", "unknown", "wink")):
        group = groups.add()
        group.name = name
        group.preset_name = preset
        bind = group.binds.add()
        bind.mesh.mesh_object_name = "body"
        bind.index = key
        bind.weight = 1.0

    secondary = ext.vrm0.secondary_animation
    head_world = arm.matrix_world @ arm.data.bones["head"].head_local
    collider_group = secondary.collider_groups.add()
    collider_group.uuid = uuid4().hex
    collider_group.node.bone_name = "head"
    collider_object = bpy.data.objects.new("head Collider", None)
    bpy.context.scene.collection.objects.link(collider_object)
    collider_object.empty_display_size = COLLIDER_RADIUS
    collider_object.parent = arm
    collider_object.parent_type = "BONE"
    collider_object.parent_bone = "head"
    bpy.context.view_layer.update()
    collider_object.matrix_world = Matrix.Translation(head_world + Vector(COLLIDER_OFFSET))
    bpy.context.view_layer.update()
    collider_group.colliders.add().bpy_object = collider_object

    bone_group = secondary.bone_groups.add()
    bone_group.comment = "Hair"
    bone_group.stiffiness = 0.8
    bone_group.gravity_power = 1.1
    bone_group.gravity_dir = JOINT_GRAVITY_DIR
    bone_group.drag_force = 0.3
    bone_group.hit_radius = 0.02
    for bone in HAIR_CHAIN:
        bone_group.bones.add().bone_name = bone
    bone_group.collider_groups.add().collider_group_uuid = collider_group.uuid


bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.preferences.addon_enable(module=MODULE)
arm = build_armature()
build_body(arm)
if SPEC_VERSION == "1.0":
    build_constraint(arm)
ext = arm.data.vrm_addon_extension
ext.spec_version = SPEC_VERSION
if SPEC_VERSION == "1.0":
    build_vrm1(arm, ext)
else:
    build_vrm0(arm, ext)
bpy.context.view_layer.update()
result = bpy.ops.export_scene.vrm(filepath=OUT_PATH)
if result != {"FINISHED"}:
    raise RuntimeError(f"export_scene.vrm returned {sorted(result)}")
print("POTTER_VRM_GENERATED", OUT_PATH)
