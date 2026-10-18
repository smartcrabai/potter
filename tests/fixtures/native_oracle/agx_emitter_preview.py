"""AgX end-to-end oracle: Blender's render of the oracle emitter under the AgX view (5.2.2).

Usage:
  Blender --factory-startup -b oracle.blend --python agx_emitter_preview.py -- AGX.blend OUT.png

The emitter's `Ink` colour layer is set to a constant 0.18 grey (linear), which the emitter
material reads through its Vertex Color node. The scene is saved as AGX.blend, with the AgX view
transform on the sRGB display (look None, exposure 0, gamma 1), for Potter to import. Every other
object is hidden, and a Cycles orthographic camera frames the emitter with one sample and no
bounces, so the render pixel equals the emission. OUT.png is the 8-bit RGB render made through the
same view transform.
"""

import sys

import bpy

ARGUMENTS = sys.argv[sys.argv.index("--") + 1 :]
AGX_BLEND, OUT_PNG = ARGUMENTS[0], ARGUMENTS[1]
GREY = 0.18

scene = bpy.context.scene
emitter = bpy.data.objects["Emitter"]

view = scene.view_settings
view.view_transform = "AgX"
view.look = "None"
view.exposure = 0.0
view.gamma = 1.0
scene.display_settings.display_device = "sRGB"

for item in emitter.data.color_attributes["Ink"].data:
    item.color = (GREY, GREY, GREY, 1.0)
bpy.ops.wm.save_as_mainfile(filepath=AGX_BLEND)

for obj in scene.objects:
    if obj is not emitter:
        obj.hide_render = True
camera_data = bpy.data.cameras.new("AgxCamera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 4.0
camera = bpy.data.objects.new("AgxCamera", camera_data)
scene.collection.objects.link(camera)
camera.location = (0.0, 0.0, 6.0)
scene.camera = camera
scene.world = None

scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 1
scene.cycles.use_denoising = False
scene.cycles.max_bounces = 0
scene.cycles.pixel_filter_type = "BOX"
scene.cycles.filter_width = 0.01
scene.render.resolution_x = 16
scene.render.resolution_y = 16
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.render.image_settings.color_depth = "8"
bpy.ops.render.render(write_still=False)
bpy.data.images["Render Result"].save_render(OUT_PNG)
