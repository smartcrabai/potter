"""A diffuse plane under a square area light with a spread, rendered by Blender's Cycles (v5.2.2).

The light is a 1 m square 2 m above the plane's centre, emitting along -Z. Its spread confines the
emission to a cone about -Z, so the plane away from the light's axis sees only the part of the light
that the cone reaches. The scene is saved as a .blend (Potter imports it), and the render is repeated
for `seeds` seeds at a fixed sample count with adaptive sampling off. For each seed the mean RGB of
each column strip of the frame is recorded, so the test can use Blender's own seed-to-seed spread as
the tolerance.

Arguments after `--`: blend_out, json_out, spread, samples, seeds.
  spread: the light's spread in radians (0.3 and 1.0 in the tests)

The strips must match `STRIPS` in `tests/native_area_light_spread.rs`.
"""

import json
import math
import os
import sys

import bpy

blend_out, json_out, spread, samples, seeds = sys.argv[sys.argv.index("--") + 1:][:5]
SPREAD = float(spread)
SAMPLES = int(samples)
SEEDS = int(seeds)

WIDTH = 16
# Inclusive pixel columns of 16 across the 3 m frame (0.19 m a pixel): on the light's axis, about
# 0.4 m off it, and about 0.8 m off it.
STRIPS = [(7, 8), (10, 11), (13, 14)]
# Radiance 5 W/(sr m²) over the 1 m² light: power = radiance · π · area.
LIGHT_RADIANCE = 5.0

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = SAMPLES
scene.cycles.use_adaptive_sampling = False
scene.cycles.use_denoising = False
scene.cycles.max_bounces = 1
scene.cycles.pixel_filter_type = "BOX"
scene.cycles.filter_width = 1.0
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.render.resolution_x = WIDTH
scene.render.resolution_y = WIDTH
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"

bpy.ops.mesh.primitive_plane_add(size=10.0, location=(0.0, 0.0, 0.0))
plane = bpy.context.object
material = bpy.data.materials.new("Diffuse")
material.use_nodes = True
bsdf = material.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
bsdf.inputs["Metallic"].default_value = 0.0
bsdf.inputs["Roughness"].default_value = 1.0
plane.data.materials.append(material)

light_data = bpy.data.lights.new("Square", "AREA")
light_data.shape = "SQUARE"
light_data.size = 1.0
light_data.spread = SPREAD
light_data.energy = LIGHT_RADIANCE * math.pi * light_data.size * light_data.size
light = bpy.data.objects.new("Square", light_data)
scene.collection.objects.link(light)
light.location = (0.0, 0.0, 2.0)

camera_data = bpy.data.cameras.new("Camera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 3.0
camera = bpy.data.objects.new("Camera", camera_data)
scene.collection.objects.link(camera)
# Looking straight down -Z from above the light, so the frame shows the plane.
camera.location = (0.0, 0.0, 5.0)
scene.camera = camera

bpy.ops.wm.save_as_mainfile(filepath=blend_out)

strip_means = []
frame_base = os.path.splitext(json_out)[0]
for seed in range(SEEDS):
    scene.cycles.seed = seed
    scene.render.filepath = f"{frame_base}_seed{seed}.exr"
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(scene.render.filepath, check_existing=False)
    width, height = image.size
    channels = image.channels
    pixels = list(image.pixels)
    means = []
    for first, last in STRIPS:
        sums = [0.0, 0.0, 0.0]
        count = 0
        for y in range(height):
            for x in range(first, last + 1):
                index = (y * width + x) * channels
                for channel in range(3):
                    sums[channel] += pixels[index + channel]
                count += 1
        means.append([value / count for value in sums])
    strip_means.append(means)
    bpy.data.images.remove(image)

with open(json_out, "w", encoding="utf-8") as handle:
    json.dump({"spread": SPREAD, "samples": SAMPLES, "strip_means": strip_means}, handle)
