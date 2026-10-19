"""A glossy plane under a disk area light, rendered by Blender's Cycles (v5.2.2).

The camera looks at the mirror image of the light, so the plane's centre shows the specular
highlight. The scene is saved as a .blend (Potter imports it), and the render is repeated for
`seeds` seeds at a fixed sample count with adaptive sampling off. For each seed the mean RGB of
the central patch is recorded, so the test can use Blender's own seed-to-seed spread as the
tolerance.

Arguments after `--`: blend_out, json_out, roughness, light_size, samples, seeds.
  roughness:  principled roughness of the plane (0.05 or 0.24 in the tests)
  light_size: diameter of the disk light in metres (0.2 small, 1.0 large)
The light's power scales with the square of its size, so its radiance is the same in both.

The patch grid must match `PATCH` in `tests/native_area_light_mis.rs`.
"""

import json
import math
import os
import sys

import bpy
from mathutils import Vector

blend_out, json_out, roughness, light_size, samples, seeds = sys.argv[
    sys.argv.index("--") + 1:
][:6]
ROUGHNESS = float(roughness)
LIGHT_SIZE = float(light_size)
SAMPLES = int(samples)
SEEDS = int(seeds)

WIDTH = 16
HEIGHT = 16
# Fractions of the frame: [0.25, 0.75] spans the central half of the plane's image.
PATCH = [0.25 + 0.5 * index / 8 for index in range(9)]
# Radiance 506 W/(sr m²) for every size: power = radiance · π · area.
LIGHT_RADIANCE = 506.0

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
scene.render.resolution_y = HEIGHT
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"

bpy.ops.mesh.primitive_plane_add(size=10.0, location=(0.0, 0.0, 0.0))
plane = bpy.context.object
material = bpy.data.materials.new("Glossy")
material.use_nodes = True
bsdf = material.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
bsdf.inputs["Metallic"].default_value = 0.0
bsdf.inputs["Roughness"].default_value = ROUGHNESS
plane.data.materials.append(material)

# The disk's area is π r², and Cycles' radiance is P / (π · area), so P = L · π · π r².
light_data = bpy.data.lights.new("Disk", "AREA")
light_data.shape = "DISK"
light_data.size = LIGHT_SIZE
light_data.energy = LIGHT_RADIANCE * math.pi * math.pi * (LIGHT_SIZE / 2.0) ** 2
light_data.cycles.use_multiple_importance_sampling = True
light = bpy.data.objects.new("Disk", light_data)
scene.collection.objects.link(light)
# An area light emits along its local -Z, which is -Z in world space at the default rotation.
light.location = (0.0, 0.0, 2.0)

camera_data = bpy.data.cameras.new("Camera")
camera = bpy.data.objects.new("Camera", camera_data)
scene.collection.objects.link(camera)
camera.location = (0.0, -3.0, 1.0)
# Look at (0, -2, 0), where the plane mirrors the light, so the highlight is at the frame centre.
camera.rotation_euler = Vector((0.0, 1.0, -1.0)).normalized().to_track_quat("-Z", "Y").to_euler()
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
        {
            "roughness": ROUGHNESS,
            "light_size": LIGHT_SIZE,
            "samples": SAMPLES,
            "patch_means": patch_means,
        },
        handle,
    )
