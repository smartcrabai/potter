"""Blender 5.2.2's colour at the two pixels that tests/render_world.rs
equirectangular_panorama_pixels_follow_their_camera_ray_directions checks, rendered with the same scene: an 8 x 1 texture
of eight longitude colours (red, green, blue, yellow, cyan, magenta, white, black) on the world, a PANO camera at the
origin with the EQUIRECTANGULAR type and default ranges, and no rotation, at 64 x 32 with Closest interpolation.

Regenerate from this directory with Blender 5.2.2, one thread, one run:

    Blender -b --factory-startup --threads 1 -P generate_render_world_panorama_oracle.py -- OUT.json

Each record is [x, top_row, r, g, b] where top_row is the image row counted from the top (as the test's PNG pixel
coordinates are), so the EXR row used is HEIGHT - 1 - top_row.
"""

import json
import os
import sys
import tempfile

import bpy

argv = sys.argv[sys.argv.index("--") + 1 :]
out_path = os.path.abspath(argv[0])
work = tempfile.mkdtemp(prefix="render_world_panorama_")
WIDTH, HEIGHT = 64, 32
CHECKS = [(48, 12), (16, 12)]  # (x, top_row), as the test's pixel() calls

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
render = scene.render
render.engine = "CYCLES"
render.resolution_x = WIDTH
render.resolution_y = HEIGHT
render.resolution_percentage = 100
render.film_transparent = False
scene.cycles.device = "CPU"
scene.cycles.samples = 1
scene.cycles.use_denoising = False
if hasattr(scene.cycles, "pixel_filter_type"):
    scene.cycles.pixel_filter_type = "BOX"
else:
    scene.cycles.filter_type = "BOX"
scene.cycles.filter_width = 0.0
scene.view_settings.view_transform = "Raw"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0
image_settings = render.image_settings
image_settings.file_format = "OPEN_EXR"
image_settings.color_depth = "32"
image_settings.color_mode = "RGBA"
image_settings.exr_codec = "NONE"

camera_data = bpy.data.cameras.new("pano")
camera_data.type = "PANO"
camera_data.panorama_type = "EQUIRECTANGULAR"
camera = bpy.data.objects.new("pano", camera_data)
scene.collection.objects.link(camera)
camera.location = (0.0, 0.0, 0.0)
camera.rotation_euler = (0.0, 0.0, 0.0)
scene.camera = camera

colors = [
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (0.0, 0.0, 0.0),
]
texture = bpy.data.images.new("panorama", 8, 1, alpha=True, float_buffer=True)
texture.colorspace_settings.name = "Non-Color"
texture.pixels.foreach_set([v for c in colors for v in (c[0], c[1], c[2], 1.0)])

world = bpy.data.worlds.new("world")
world.use_nodes = True
scene.world = world
tree = world.node_tree
for node in list(tree.nodes):
    tree.nodes.remove(node)
env = tree.nodes.new("ShaderNodeTexEnvironment")
env.image = texture
env.interpolation = "Closest"
background = tree.nodes.new("ShaderNodeBackground")
background.inputs["Strength"].default_value = 1.0
output = tree.nodes.new("ShaderNodeOutputWorld")
tree.links.new(env.outputs["Color"], background.inputs["Color"])
tree.links.new(background.outputs["Background"], output.inputs["Surface"])

render.filepath = os.path.join(work, "panorama.exr")
bpy.ops.render.render(write_still=True)
loaded = bpy.data.images.load(render.filepath, check_existing=False)
pixels = list(loaded.pixels)
bpy.data.images.remove(loaded)

records = []
for x, top_row in CHECKS:
    row = HEIGHT - 1 - top_row
    offset = (row * WIDTH + x) * 4
    records.append([x, top_row, round(pixels[offset], 6), round(pixels[offset + 1], 6), round(pixels[offset + 2], 6)])

with open(out_path, "w") as handle:
    json.dump({"width": WIDTH, "height": HEIGHT, "records": records}, handle, indent=1)
print("ORACLE_OK render_world", records)
