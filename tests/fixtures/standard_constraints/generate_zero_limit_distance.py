import os
import sys

import bpy

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"fixture requires Blender 5.2.2, found {bpy.app.version_string}")

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
owner = bpy.data.objects.new("ZeroDistanceOwner", None)
target = bpy.data.objects.new("ZeroDistanceTarget", None)
scene.collection.objects.link(owner)
scene.collection.objects.link(target)
constraint = owner.constraints.new("LIMIT_DISTANCE")
constraint.target = target
constraint.distance = 0.0
bpy.context.view_layer.update()
assert constraint.distance == 0.0
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, "zero_limit_distance.blend"))
print("ZERO_LIMIT_DISTANCE_FIXTURE_OK")
