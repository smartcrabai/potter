"""Transparent bounce cap and shadow budget, rendered by Blender's Cycles (v5.2.2).

A lit diffuse floor sits under N alpha-0.5 planes, and a point light sits above them. Each plane is
a black BLENDED Principled with Alpha 0.5, so a camera ray passes it with probability 0.5 and is
absorbed otherwise. `transparent_max_bounces` is C.

Cycles' rules (`intern/cycles/kernel/integrator/path_state.h`, `intersect_filter.h`):
  - the pass that reaches C sets TERMINATE_ON_NEXT_SURFACE, so the next surface, the floor included,
    ends the path after its emission (no direct light);
  - a shadow ray from a vertex with t transparent passes may cross at most C - t transparent
    surfaces, and is blocked past that.

For the floor to be lit, the camera path passes all N planes (probability 2^-N). The floor is
terminated when N >= C. Its shadow ray crosses N planes against the budget C - N, so it is blocked
when N > C - N, and otherwise transmits 2^-N. Expected patch means, with L0 the lit floor radiance:
  n2c4: 2^-2 * 2^-2 * L0    n2c3: 0 (shadow blocked)    n2c2: 0 (terminated)    n1c2: 2^-1 * 2^-1 * L0

Arguments after `--`: blend_out, exr_out, json_out, kind (n2c4 | n2c3 | n2c2 | n1c2).
"""

import json
import sys

import bpy

blend_out, exr_out, json_out, kind = sys.argv[sys.argv.index("--") + 1:][:4]

PLANES, CAP = {"n2c4": (2, 4), "n2c3": (2, 3), "n2c2": (2, 2), "n1c2": (1, 2)}[kind]
WIDTH = 32
HEIGHT = 32
SAMPLES = 256
PLANE_SPACING = 1.0
LIGHT_HEIGHT = 5.0
LIGHT_ENERGY = 1200.0
LIGHT_RADIUS = 0.5
# Fractions of the frame: the same patch grid as `tests/native_transparent_cap.rs`.
PATCH = [0.25 + 0.5 * index / 8 for index in range(9)]

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = SAMPLES
scene.cycles.use_denoising = False
scene.cycles.max_bounces = 4
scene.cycles.transparent_max_bounces = CAP
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.render.resolution_x = WIDTH
scene.render.resolution_y = HEIGHT
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"

floor_mesh = bpy.data.materials.new("Floor")
floor_mesh.use_nodes = True
floor = floor_mesh.node_tree.nodes["Principled BSDF"]
floor.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
floor.inputs["Metallic"].default_value = 0.0
floor.inputs["Roughness"].default_value = 1.0
floor.inputs["Specular IOR Level"].default_value = 0.0
bpy.ops.mesh.primitive_plane_add(size=10.0, location=(0.0, 0.0, 0.0))
bpy.context.object.data.materials.append(floor_mesh)

plane_material = bpy.data.materials.new("Alpha")
plane_material.use_nodes = True
# BLENDED keeps the sub-one Alpha as an alpha blend; the default DITHERED method imports as `clip`.
plane_material.surface_render_method = "BLENDED"
plane = plane_material.node_tree.nodes["Principled BSDF"]
plane.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
plane.inputs["Metallic"].default_value = 0.0
plane.inputs["Roughness"].default_value = 1.0
plane.inputs["Specular IOR Level"].default_value = 0.0
plane.inputs["Alpha"].default_value = 0.5
for index in range(PLANES):
    bpy.ops.mesh.primitive_plane_add(size=10.0, location=(0.0, 0.0, 1.0 + index * PLANE_SPACING))
    bpy.context.object.data.materials.append(plane_material)

light_data = bpy.data.lights.new("Point", "POINT")
light_data.energy = LIGHT_ENERGY
light_data.shadow_soft_size = LIGHT_RADIUS
light = bpy.data.objects.new(light_data.name, light_data)
scene.collection.objects.link(light)
light.location = (0.0, 0.0, LIGHT_HEIGHT)

camera_data = bpy.data.cameras.new("Camera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 4.0
camera = bpy.data.objects.new("Camera", camera_data)
scene.collection.objects.link(camera)
camera.location = (0.0, 0.0, 10.0)
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
    json.dump({"kind": kind, "patch_mean": [value / count for value in sums]}, handle)
