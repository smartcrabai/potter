"""Render the canonical volume-factor scene with EEVEE volumetric shadows enabled.

Run in Blender 5.2.2 after `eevee_volume_factor.py` saves a source blend:

    Blender -b --factory-startup -P eevee_volume_shadowed.py -- SOURCE.blend OUTPUT.blend RENDER.exr
"""

import sys

import bpy

source_blend, blend_out, exr_out = sys.argv[sys.argv.index("--") + 1 :][:3]
bpy.ops.wm.open_mainfile(filepath=source_blend)
scene = bpy.context.scene
scene.eevee.use_volumetric_shadows = True
scene.render.filepath = exr_out
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
bpy.ops.wm.save_as_mainfile(filepath=blend_out)

if exr_out == "-":
    sys.exit(0)

bpy.ops.render.render(write_still=True)
