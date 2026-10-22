"""Blender 5.2.2 EEVEE renders of a world alone, for the realtime world background oracle.

Run by Blender, with the case, the .blend to save and the EEVEE render (EXR) to write:

    Blender -b --factory-startup -P eevee_world_background.py -- CASE OUT.blend RENDER.exr

RENDER.exr is `-` to save the .blend and stop before rendering. The render is exported as f32le by
`export_eevee_fixture.py` for the committed `eevee_macos/` captures.

Each case takes the world's colour from its Background node, not from `world.color`, which Blender's viewport shows:
- `node_black`: a black Background node over a viewport colour of 0.05 (Blender's default). EEVEE shows black, so a
  background that reads `world.color` shows 0.05 instead.
- `node_colour`: a Background node colour of (0.2, 0.4, 0.1) at strength 0.5, over a viewport colour of 0.9. EEVEE shows
  (0.1, 0.2, 0.05).

The camera is orthographic and looks down the -Z axis at an empty scene, so every pixel is the world.
"""

import sys

import bpy

RESOLUTION = 64
SAMPLES = 16

argv = sys.argv[sys.argv.index("--") + 1 :][:3]
case, blend_out, render_exr = argv

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.eevee.taa_render_samples = SAMPLES
scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"

camera_data = bpy.data.cameras.new("Camera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 8.0
camera = bpy.data.objects.new("Camera", camera_data)
scene.collection.objects.link(camera)
camera.location = (0.0, 0.0, 4.0)
scene.camera = camera

world = bpy.data.worlds.new("World")
world.use_nodes = True
scene.world = world
background = world.node_tree.nodes["Background"]
if case == "node_black":
    world.color = (0.05, 0.05, 0.05)
    background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    background.inputs["Strength"].default_value = 1.0
elif case == "node_colour":
    world.color = (0.9, 0.9, 0.9)
    background.inputs["Color"].default_value = (0.2, 0.4, 0.1, 1.0)
    background.inputs["Strength"].default_value = 0.5
else:
    raise SystemExit(f"unknown case {case}")

bpy.ops.wm.save_as_mainfile(filepath=blend_out)
if render_exr == "-":
    raise SystemExit(0)
scene.render.filepath = render_exr
bpy.ops.render.render(write_still=True)
print("EEVEE_WORLD_BACKGROUND", case, render_exr)
