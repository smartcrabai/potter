import bpy
import json
import os
import sys


root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
rig = bpy.data.objects.get("LegacyPoseRig")
assert rig is not None and rig.type == "ARMATURE"
pose_bone = rig.pose.bones.get("EulerBone")
assert pose_bone is not None


def evaluated_euler(frame):
    bpy.context.scene.frame_set(frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = rig.evaluated_get(depsgraph)
    return [float(value) for value in evaluated.pose.bones["EulerBone"].rotation_euler]


result = {
    "rotation_mode": pose_bone.rotation_mode,
    "action_present": bool(rig.animation_data and rig.animation_data.action),
    "frame_1": evaluated_euler(1),
    "frame_9": evaluated_euler(9),
}
with open(os.path.join(root, "legacy_pose_reopened.json"), "w", encoding="utf-8") as output:
    json.dump(result, output)
