"""Blender 5.2.2's environment-texture colour around the equirectangular u = 0 seam and the poles, as the oracle for Potter's
world reader (shader/world.rs, environment_colour).

Regenerate from this directory with Blender 5.2.2, one thread, one run per case:

    Blender -b --factory-startup --threads 1 -P generate_world_env_oracle.py -- INTERPOLATION OUT.json

INTERPOLATION is Closest or Linear.

The world is a 4 x 2 float equirectangular texture whose texel at Blender row y (bottom-up) and column x holds
(x / 4, y / 2, 0.5), so every texel is distinct. The world's Background shows the Environment Texture (no Vector input, so
Cycles looks up the ray direction). A 16 x 16 camera with a 160 degree field looks at (-1, 0, 0.6), so its centre lies on
the u = 0 seam and its top rows reach the zenith. A second render of the same camera, with the world's Background set to
the ray direction (Texture Coordinate Generated, mapped to 0..1), records the direction of each pixel exactly.

Each record is [dx, dy, dz, r, g, b]: the ray direction Blender used and the rendered environment colour.
"""

import json
import math
import os
import sys
import tempfile

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1 :]
interpolation, out_path = argv[:2]
out_path = os.path.abspath(out_path)
work = tempfile.mkdtemp(prefix="world_env_")
SIZE = 16
W, H = 4, 2

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
render = scene.render
render.engine = "CYCLES"
render.resolution_x = SIZE
render.resolution_y = SIZE
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

direction = Vector((-1.0, 0.0, 0.6)).normalized()
camera_data = bpy.data.cameras.new("cam")
camera_data.angle = math.radians(160.0)
camera = bpy.data.objects.new("cam", camera_data)
scene.collection.objects.link(camera)
camera.location = (0.0, 0.0, 0.0)
camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
scene.camera = camera

texels = []
for y in range(H):
    for x in range(W):
        texels.extend([x / W, y / H, 0.5, 1.0])
environment = bpy.data.images.new("env", W, H, alpha=False, float_buffer=True)
environment.colorspace_settings.name = "Non-Color"
environment.pixels.foreach_set(texels)


def new_world(name):
    world = bpy.data.worlds.new(name)
    world.use_nodes = True
    scene.world = world
    for node in list(world.node_tree.nodes):
        world.node_tree.nodes.remove(node)
    return world.node_tree


def render_to(path):
    render.filepath = path
    bpy.ops.render.render(write_still=True)
    loaded = bpy.data.images.load(path, check_existing=False)
    values = list(loaded.pixels)
    bpy.data.images.remove(loaded)
    return values


# First render: the Background shows the ray direction mapped to 0..1, so each pixel records its direction.
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
direction_pixels = render_to(os.path.join(work, "direction.exr"))

# Second render: the Background shows the Environment Texture.
tree = new_world("environment")
env = tree.nodes.new("ShaderNodeTexEnvironment")
env.image = environment
env.interpolation = "Closest" if interpolation == "Closest" else "Linear"
background = tree.nodes.new("ShaderNodeBackground")
background.inputs["Strength"].default_value = 1.0
output = tree.nodes.new("ShaderNodeOutputWorld")
tree.links.new(env.outputs["Color"], background.inputs["Color"])
tree.links.new(background.outputs["Background"], output.inputs["Surface"])
colour_pixels = render_to(os.path.join(work, "colour.exr"))

records = []
for y in range(SIZE):
    for x in range(SIZE):
        offset = (y * SIZE + x) * 4
        d = [2.0 * direction_pixels[offset + c] - 1.0 for c in range(3)]
        records.append(
            [
                round(d[0], 7),
                round(d[1], 7),
                round(d[2], 7),
                round(colour_pixels[offset], 6),
                round(colour_pixels[offset + 1], 6),
                round(colour_pixels[offset + 2], 6),
            ]
        )

with open(out_path, "w") as handle:
    json.dump(
        {
            "interpolation": interpolation,
            "width": W,
            "height": H,
            "texels_blender_rows": [
                [x / W, y / H, 0.5, 1.0] for y in range(H) for x in range(W)
            ],
            "records": records,
        },
        handle,
        indent=1,
    )
print("ORACLE_OK world", interpolation, len(records))
