"""Lists the lights of the open .blend with their type and area spread, for the round-trip test.

Arguments after `--`: json_out. A light without a spread (every type but AREA) reports Blender's
default π.
"""

import json
import math
import sys

import bpy

json_out = sys.argv[sys.argv.index("--") + 1]

lights = [
    {
        "name": light.name,
        "type": light.type,
        "spread": float(getattr(light, "spread", math.pi)),
    }
    for light in bpy.data.lights
]
with open(json_out, "w", encoding="utf-8") as handle:
    json.dump({"lights": lights}, handle)
