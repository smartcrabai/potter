"""Dumps the six ray visibility flags of every object in the open .blend (Blender 5.2.2).

Usage:
  Blender --factory-startup -b FILE.blend --python ray_visibility_dump.py -- OUT.json

The output maps each object name to its `visible_*` flags, read through Blender's own API.
"""

import json
import sys

import bpy

OUT = sys.argv[sys.argv.index("--") + 1]
FLAGS = (
    "visible_camera",
    "visible_diffuse",
    "visible_glossy",
    "visible_transmission",
    "visible_volume_scatter",
    "visible_shadow",
)

result = {
    obj.name: {flag: bool(getattr(obj, flag)) for flag in FLAGS} for obj in bpy.data.objects
}
with open(OUT, "w", encoding="utf-8") as handle:
    json.dump(result, handle, sort_keys=True)
