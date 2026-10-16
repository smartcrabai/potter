"""PR9 alpha reader: dumps every material of the open .blend as JSON (Blender 5.2.2).

Usage:
  Blender --factory-startup -b FILE.blend --python material_alpha.py -- OUT.json

`surface_render_method` and `alpha_threshold` are Blender's own. `principled_alpha` is the
Principled BSDF's Alpha default, `principled_base_alpha` the fourth component of its Base Color,
and `alpha_mode_property` the `potter.alpha_mode` marker (None when absent).
"""

import json
import sys

import bpy

OUT = sys.argv[sys.argv.index("--") + 1]


def principled(material):
    if material.node_tree is None:
        return None
    return next((node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"), None)


def material_record(material):
    shader = principled(material)
    return {
        "name": material.name,
        "surface_render_method": material.surface_render_method,
        "alpha_threshold": float(material.alpha_threshold),
        "principled_alpha": float(shader.inputs["Alpha"].default_value) if shader else None,
        "principled_base_alpha": float(shader.inputs["Base Color"].default_value[3]) if shader else None,
        "alpha_mode_property": material.get("potter.alpha_mode"),
    }


with open(OUT, "w") as handle:
    json.dump({"materials": [material_record(item) for item in bpy.data.materials]}, handle)
