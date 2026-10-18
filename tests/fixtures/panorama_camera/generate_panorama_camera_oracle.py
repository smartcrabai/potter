"""Blender 5.2.2's equirectangular panorama camera ray directions, as the oracle for Potter's panorama ray generation
(render/raster.rs ProjectionFrame::Panorama, and the path renderer's camera rays).

Regenerate from this directory with Blender 5.2.2, one thread, one run:

    Blender -b --factory-startup --threads 1 -P generate_panorama_camera_oracle.py -- OUT.json

The camera is a PANO camera with the EQUIRECTANGULAR panorama type, non-default longitude and latitude ranges (Blender's
`longitude_min` and friends, in radians), and a rotated, translated object. A world whose Background shows the ray
direction (Texture Coordinate Generated, mapped to 0..1) records each pixel's world-space ray direction. The filter width
is zero, so each pixel is sampled at its centre.

Each record is [x, row, dx, dy, dz] where x is the column and row is the EXR row, which is bottom-up (row 0 is the bottom
of the image). The camera transform and ranges are stored in the file too.
"""

import json
import math
import os
import sys
import tempfile

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1 :]
out_path = os.path.abspath(argv[0])
work = tempfile.mkdtemp(prefix="panorama_camera_")
WIDTH, HEIGHT = 32, 16
LONGITUDE = (-1.2, 2.0)
LATITUDE = (-0.7, 1.1)
LOCATION = (0.3, -0.2, 0.5)
ROTATION = (1.0, 0.3, 0.6)

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
camera_data.longitude_min, camera_data.longitude_max = LONGITUDE
camera_data.latitude_min, camera_data.latitude_max = LATITUDE
camera = bpy.data.objects.new("pano", camera_data)
scene.collection.objects.link(camera)
camera.location = LOCATION
camera.rotation_euler = ROTATION
scene.camera = camera


def new_world(name):
    world = bpy.data.worlds.new(name)
    world.use_nodes = True
    scene.world = world
    for node in list(world.node_tree.nodes):
        world.node_tree.nodes.remove(node)
    return world.node_tree


tree = new_world("direction")
coord = tree.nodes.new("ShaderNodeTexCoord")
mapped = tree.nodes.new("ShaderNodeVectorMath")
mapped.operation = "MULTIPLY_ADD"
mapped.inputs[1].default_value = (0.5, 0.5, 0.5)
mapped.inputs[2].default_value = (0.5, 0.5, 0.5)
background = tree.nodes.new("ShaderNodeBackground")
background.inputs["Strength"].default_value = 1.0
output = tree.nodes.new("ShaderNodeOutputWorld")
tree.links.new(coord.outputs["Generated"], mapped.inputs[0])
tree.links.new(mapped.outputs["Vector"], background.inputs["Color"])
tree.links.new(background.outputs["Background"], output.inputs["Surface"])

render.filepath = os.path.join(work, "direction.exr")
bpy.ops.render.render(write_still=True)
loaded = bpy.data.images.load(render.filepath, check_existing=False)
pixels = list(loaded.pixels)
bpy.data.images.remove(loaded)

records = []
for row in range(HEIGHT):
    for x in range(WIDTH):
        offset = (row * WIDTH + x) * 4
        d = [2.0 * pixels[offset + c] - 1.0 for c in range(3)]
        length = math.sqrt(sum(v * v for v in d))
        records.append(
            [x, row]
            + [round(v / length, 7) for v in d]
        )

matrix = [list(row) for row in camera.matrix_world]
with open(out_path, "w") as handle:
    json.dump(
        {
            "width": WIDTH,
            "height": HEIGHT,
            "longitude": list(LONGITUDE),
            "latitude": list(LATITUDE),
            "camera_to_world": matrix,
            "records": records,
        },
        handle,
        indent=1,
    )
print("ORACLE_OK panorama", len(records))
