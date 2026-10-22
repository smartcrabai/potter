"""A rough metal sphere under one or more lights and a world, rendered by Blender's Cycles (v5.2.2).

The camera looks at the sphere from the front, so the central patch of the frame is sphere surface
that reflects the lights and the world. The scene is saved as a .blend (Potter imports it), and the
render is repeated for `seeds` seeds at a fixed sample count with adaptive sampling and denoising
off. For each seed the mean RGB of the central patch is recorded, so the test can use Blender's own
seed-to-seed spread as its tolerance.

Arguments after `--`: blend_out, json_out, scenario, samples, seeds, frame.
  scenario: which lights and world the scene has. `point`, `rect`, and `sun` are one light and no
            world; `world` is the world alone; `all` has the point, the rectangle, the sun, and the world.
  frame:    square frame size in pixels (32 for the reference, 16 for the runtime-reduced scene). The
            camera's focal length scales with 32 / frame, so a 16 px frame covers the central half of
            the 32 px frame's view with the same pixel grid. The patch is the central 16 × 16 pixels.

The committed reference (`metal_sphere_lights_reference.json`) is the 32 px frame at 1024 samples.
"""

import json
import math
import os
import sys

import bpy
from mathutils import Vector

blend_out, json_out, scenario, samples, seeds, frame = sys.argv[sys.argv.index("--") + 1:][:6]
SCENARIO = scenario
SAMPLES = int(samples)
SEEDS = int(seeds)
FRAME = int(frame)
LIGHTS = {
    "point": ["point"],
    "rect": ["rect"],
    "sun": ["sun"],
    "world": [],
    "all": ["point", "rect", "sun"],
}
if SCENARIO not in LIGHTS:
    raise SystemExit(f"unknown scenario {SCENARIO}")
USE_WORLD = SCENARIO in ("world", "all")

WIDTH = FRAME
HEIGHT = FRAME
# Texel centres of the central 16×16 block. The block is symmetric under a row flip, so its mean does
# not depend on whether the renderer's row 0 is the top or the bottom.
PATCH = [(index + 0.5) / WIDTH for index in range(WIDTH // 2 - 8, WIDTH // 2 + 8)]

ROUGHNESS = 0.3
BASE = (0.9, 0.55, 0.25, 1.0)
# Radiance of the rectangle (W/(sr m²)); the rectangle's power is radiance · π · area.
RECT_RADIANCE = 2.0
RECT_SIZE = 1.5
POINT_POWER = 300.0  # W; radiant intensity P / (4π)
SUN_STRENGTH = 2.0  # W/m²
WORLD_COLOUR = (0.3, 0.35, 0.4, 1.0)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = SAMPLES
scene.cycles.use_adaptive_sampling = False
scene.cycles.use_denoising = False
scene.cycles.max_bounces = 4
scene.cycles.pixel_filter_type = "BOX"
scene.cycles.filter_width = 1.0
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.render.resolution_x = WIDTH
scene.render.resolution_y = HEIGHT
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"


def aim(obj, target):
    # Area and spot lights emit along local -Z; the sun does too.
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


world = bpy.data.worlds.new("World")
world.use_nodes = True
background = world.node_tree.nodes["Background"]
background.inputs["Color"].default_value = WORLD_COLOUR if USE_WORLD else (0, 0, 0, 1)
background.inputs["Strength"].default_value = 1.0
scene.world = world

bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=64, ring_count=32, location=(0.0, 0.0, 0.0))
sphere = bpy.context.object
sphere.name = "Sphere"
bpy.ops.object.shade_smooth()
material = bpy.data.materials.new("Metal")
material.use_nodes = True
bsdf = material.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = BASE
bsdf.inputs["Metallic"].default_value = 1.0
bsdf.inputs["Roughness"].default_value = ROUGHNESS
sphere.data.materials.append(material)

for kind in LIGHTS[SCENARIO]:
    if kind == "point":
        data = bpy.data.lights.new("Point", "POINT")
        data.energy = POINT_POWER
        data.shadow_soft_size = 0.0
        obj = bpy.data.objects.new("Point", data)
        scene.collection.objects.link(obj)
        obj.location = (-2.5, -2.5, 3.0)
    elif kind == "rect":
        data = bpy.data.lights.new("Rect", "AREA")
        data.shape = "RECTANGLE"
        data.size = RECT_SIZE
        data.size_y = RECT_SIZE
        data.energy = RECT_RADIANCE * math.pi * RECT_SIZE * RECT_SIZE
        obj = bpy.data.objects.new("Rect", data)
        scene.collection.objects.link(obj)
        obj.location = (2.5, -2.5, 2.5)
        aim(obj, (0.0, 0.0, 0.0))
    elif kind == "sun":
        data = bpy.data.lights.new("Sun", "SUN")
        data.energy = SUN_STRENGTH
        obj = bpy.data.objects.new("Sun", data)
        scene.collection.objects.link(obj)
        obj.location = (0.0, 0.0, 5.0)
        aim(obj, (-1.0, 0.5, 0.0))

camera_data = bpy.data.cameras.new("Camera")
# The 32 px reference has the default 50 mm lens; a smaller frame zooms by 32 / frame.
camera_data.lens = 50.0 * 32 / FRAME
camera = bpy.data.objects.new("Camera", camera_data)
scene.collection.objects.link(camera)
camera.location = (0.0, -4.0, 0.0)
aim(camera, (0.0, 0.0, 0.0))
scene.camera = camera

bpy.ops.wm.save_as_mainfile(filepath=blend_out)

patch_means = []
frame_base = os.path.splitext(json_out)[0]
for seed in range(SEEDS):
    scene.cycles.seed = seed
    scene.render.filepath = f"{frame_base}_seed{seed}.exr"
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(scene.render.filepath, check_existing=False)
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
    patch_means.append([value / count for value in sums])
    bpy.data.images.remove(image)

with open(json_out, "w", encoding="utf-8") as handle:
    json.dump(
        {"scenario": SCENARIO, "samples": SAMPLES, "patch_means": patch_means},
        handle,
    )
