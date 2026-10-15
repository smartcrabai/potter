"""Exports POTTER_PROBE_BLEND to POTTER_PROBE_OUT_GLB with Blender's glTF exporter (GLB, default options).

Run headless:
    Blender -b --factory-startup --python export_gltf.py
"""

import os

import bpy

bpy.ops.wm.open_mainfile(filepath=os.environ["POTTER_PROBE_BLEND"])
bpy.ops.export_scene.gltf(filepath=os.environ["POTTER_PROBE_OUT_GLB"], export_format="GLB")
