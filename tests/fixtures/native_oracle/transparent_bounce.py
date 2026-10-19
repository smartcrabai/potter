"""A stack of alpha-0.5 planes in front of a coloured world, rendered by Blender's Cycles (v5.2.2).

Writes the scene as a .blend (Potter imports it), renders it to a linear EXR, and records the mean
RGB of a central patch. The patch grid must match `PATCH` in `tests/native_transparent_bounce.rs`.

Each plane is a black diffuse Principled BSDF with Alpha 0.5: Cycles mixes a Transparent closure of
weight `1 - alpha` with the Principled of weight `alpha`, so a ray passes each plane with
probability 0.5 and is absorbed otherwise. With six planes the world shows through with
probability 1/64. The regular `max_bounces` is 4, which is fewer than the six passes: transparent
bounces must not consume the regular budget (`transparent_max_bounces` stays at Blender's default
8), or the world behind the stack disappears entirely.

Arguments after `--`: blend_out, exr_out, json_out.
"""

import json
import sys

import bpy

blend_out, exr_out, json_out = sys.argv[sys.argv.index("--") + 1:][:3]

WIDTH = 32
HEIGHT = 32
PLANE_COUNT = 6
PLANE_SPACING = 1.0
ALPHA = 0.5
MAX_BOUNCES = 4
WORLD = (1.0, 0.5, 0.0)
# Pixel rows and columns of the patch (pixel centres, fractions of the frame): the central 16 x 16 of
# 32 x 32, the same window as `tests/native_transparent_bounce.rs`.
PATCH = [(index + 0.5) / WIDTH for index in range(8, 24)]

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 64
scene.cycles.use_denoising = False
scene.cycles.max_bounces = MAX_BOUNCES
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.render.resolution_x = WIDTH
scene.render.resolution_y = HEIGHT
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"

world = bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
background = world.node_tree.nodes["Background"]
background.inputs["Color"].default_value = (*WORLD, 1.0)
background.inputs["Strength"].default_value = 1.0

material = bpy.data.materials.new("Alpha")
material.use_nodes = True
# BLENDED keeps the sub-one Principled Alpha as an alpha blend (Potter's `blend` alpha mode); the
# default DITHERED method would import as `clip`, which binarizes the alpha at the threshold.
material.surface_render_method = "BLENDED"
bsdf = material.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
bsdf.inputs["Metallic"].default_value = 0.0
bsdf.inputs["Roughness"].default_value = 1.0
bsdf.inputs["Specular IOR Level"].default_value = 0.0
bsdf.inputs["Alpha"].default_value = ALPHA

for index in range(PLANE_COUNT):
    bpy.ops.mesh.primitive_plane_add(size=10.0, location=(0.0, 0.0, index * PLANE_SPACING))
    plane = bpy.context.object
    plane.data.materials.append(material)

camera_data = bpy.data.cameras.new("Camera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 4.0
camera = bpy.data.objects.new("Camera", camera_data)
scene.collection.objects.link(camera)
camera.location = (0.0, 0.0, PLANE_COUNT * PLANE_SPACING + 5.0)
scene.camera = camera

bpy.ops.wm.save_as_mainfile(filepath=blend_out)

scene.render.filepath = exr_out
bpy.ops.render.render(write_still=True)
image = bpy.data.images.load(exr_out, check_existing=False)
width, height = image.size
channels = image.channels
pixels = list(image.pixels)
sums = [0.0, 0.0, 0.0]
count = 0
for v in PATCH:
    for u in PATCH:
        x = min(width - 1, int(u * width))
        y = min(height - 1, int(v * height))
        index = (y * width + x) * channels
        for channel in range(3):
            sums[channel] += pixels[index + channel]
        count += 1

with open(json_out, "w", encoding="utf-8") as handle:
    json.dump({"patch_mean": [value / count for value in sums]}, handle)
