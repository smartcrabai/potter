# Imports a VRM with the pinned VRM Add-on for Blender and exports it again, unchanged.
#
# Run (the add-on must already be installed into BLENDER_USER_RESOURCES, see tests/vrm_import.rs):
#   blender -b --factory-startup --python-exit-code 1 --python tests/vrm-import/addon_round_trip.py -- <in.vrm> <out.vrm>
#
# The add-on keeps mesh-level extras but drops root-level ones, which is how Potter's stale vertex
# metadata reaches the importer after this round trip.
import sys

import bpy

MODULE = "bl_ext.user_default.vrm"
ARGS = sys.argv[sys.argv.index("--") + 1 :]
INPUT_PATH, OUTPUT_PATH = ARGS[0], ARGS[1]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.preferences.addon_enable(module=MODULE)
bpy.ops.import_scene.vrm(filepath=INPUT_PATH)
armature = next(obj for obj in bpy.data.objects if obj.type == "ARMATURE")
bpy.context.view_layer.objects.active = armature
result = bpy.ops.export_scene.vrm(filepath=OUTPUT_PATH)
if "FINISHED" not in result:
    raise RuntimeError(f"VRM Add-on export did not finish: {result}")
print("POTTER_VRM_GENERATED")
