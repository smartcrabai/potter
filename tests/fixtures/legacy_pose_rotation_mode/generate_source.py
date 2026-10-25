import bpy
import os
import sys


if bpy.app.version != (5, 2, 2):
    raise RuntimeError("legacy pose-mode fixture requires Blender 5.2.2")

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 9

armature_data = bpy.data.armatures.new("LegacyPoseRigData")
rig = bpy.data.objects.new("LegacyPoseRig", armature_data)
scene.collection.objects.link(rig)
bpy.context.view_layer.objects.active = rig
rig.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
bone = armature_data.edit_bones.new("EulerBone")
bone.head = (0.0, 0.0, 0.0)
bone.tail = (0.0, 1.0, 0.0)
bpy.ops.object.mode_set(mode="OBJECT")

pose_bone = rig.pose.bones["EulerBone"]
pose_bone.rotation_mode = "XYZ"
for frame, rotation in (
    (1, (0.0, 0.0, 0.0)),
    (5, (0.25, -0.18, 0.55)),
    (9, (0.4, -0.3, 1.1)),
):
    pose_bone.rotation_euler = rotation
    pose_bone.keyframe_insert(data_path="rotation_euler", frame=frame)

assert rig.animation_data is not None and rig.animation_data.action is not None
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, "legacy_pose_source.blend"))
