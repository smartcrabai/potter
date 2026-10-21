"""EEVEE oracle for the realtime film filter (`render.filter_size`, Blender's Filter Size; 5.2.2).

A 64 px orthographic frame of 8 m (0.125 m per pixel) looks down at a white emitter that fills the world right of
x = EDGE_X. The edge sits at sub-pixel column EDGE_PIXEL = 32.3, so each filter size shows a different blur of a
known edge. The world is black, so the only light is the emitter, with Standard view transform and gamma 1.

Each FILTER_SIZES value renders once in EEVEE with `taa_render_samples = SAMPLES`. The output JSON holds, for each
size, the red channel averaged over the rows of the frame (the edge is vertical, so the rows are identical up to
noise): one value per column.

Arguments after `--`: blend_out, json_out, render_dir. `blend_out` is the scene with the default Filter Size, for
Potter to import; the EEVEE renders go to `render_dir`.
"""

import json
import os
import sys

import bpy
import numpy as np

RESOLUTION = 64
ORTHO_SCALE = 8.0
EDGE_PIXEL = 32.3
EDGE_X = (EDGE_PIXEL / RESOLUTION - 0.5) * ORTHO_SCALE
SAMPLES = 256
FILTER_SIZES = [0.01, 0.5, 1.5, 2.5]

blend_out, json_out, render_dir = sys.argv[sys.argv.index("--") + 1:][:3]
os.makedirs(render_dir, exist_ok=True)

for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

scene = bpy.context.scene
world = bpy.data.worlds.new("Black")
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
scene.world = world
# Potter's realtime Beauty background reads `World.color` (`render/preview.rs` `background_color`), not the Background node
# that EEVEE renders. Both are black, so the oracle's world is black on either reading.
world.color = (0.0, 0.0, 0.0)

scene.render.engine = "BLENDER_EEVEE"
scene.eevee.taa_render_samples = SAMPLES
scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
view = scene.view_settings
view.view_transform = "Standard"
view.look = "None"
view.exposure = 0.0
view.gamma = 1.0
scene.display_settings.display_device = "sRGB"

camera_data = bpy.data.cameras.new("OracleCamera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = ORTHO_SCALE
camera = bpy.data.objects.new("OracleCamera", camera_data)
scene.collection.objects.link(camera)
camera.location = (0.0, 0.0, 6.0)
scene.camera = camera

bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, 0.0))
plane = bpy.context.object
plane.name = "Emitter"
EMITTER_WIDTH = 4.5 - EDGE_X
plane.scale = (EMITTER_WIDTH, 8.0, 1.0)
plane.location = (EDGE_X + EMITTER_WIDTH / 2.0, 0.0, 0.0)
material = bpy.data.materials.new("Emitter")
material.use_nodes = True
bsdf = material.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
bsdf.inputs["Specular IOR Level"].default_value = 0.0
bsdf.inputs["Emission Color"].default_value = (1.0, 1.0, 1.0, 1.0)
bsdf.inputs["Emission Strength"].default_value = 1.0
plane.data.materials.append(material)

# The scene the test imports keeps Blender's default Filter Size.
scene.render.filter_size = 1.5
bpy.ops.wm.save_as_mainfile(filepath=blend_out)


def row_profile(path):
    image = bpy.data.images.load(path, check_existing=False)
    width, height = image.size
    pixels = np.array(image.pixels[:]).reshape(height, width, 4)[:, :, 0]
    bpy.data.images.remove(image)
    return pixels.mean(axis=0).tolist()


result = {"resolution": RESOLUTION, "ortho_scale": ORTHO_SCALE, "edge_pixel": EDGE_PIXEL,
          "samples": SAMPLES, "profiles": {}}
for size in FILTER_SIZES:
    scene.render.filter_size = size
    path = os.path.join(render_dir, f"filter_{size}.exr")
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    result["profiles"][str(size)] = row_profile(path)

with open(json_out, "w") as handle:
    json.dump(result, handle, indent=2)
print("WROTE", json_out)
