"""PR9 alpha fixture: writes a .blend whose materials were authored in Blender 5.2.2, with no
`potter.alpha_mode` marker. Each material keeps the fake user so it is saved unassigned.

Usage:
  Blender --factory-startup -b --python author_alpha_materials.py -- OUT.blend
"""

import sys

import bpy

OUT = sys.argv[sys.argv.index("--") + 1]

# (name, surface_render_method, Principled Alpha)
SPECS = [
    ("Authored Blended", "BLENDED", 1.0),
    ("Authored Dithered Alpha", "DITHERED", 0.4),
    ("Authored Dithered Opaque", "DITHERED", 1.0),
]

for name, method, alpha in SPECS:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.surface_render_method = method
    shader = next(node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    shader.inputs["Alpha"].default_value = alpha
    material.use_fake_user = True

bpy.ops.wm.save_as_mainfile(filepath=OUT)
