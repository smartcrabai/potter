"""A point or spot light over a diffuse plane, rendered by Blender's Cycles (v5.2.2).

Writes the scene as a .blend (Potter imports it), renders it to a linear EXR, and records the mean
RGB of a central patch of the plane. The patch grid must match `PATCH` in
`tests/native_point_spot_light.rs`.

Arguments after `--`: blend_out, exr_out, json_out, kind (point_soft | point_sphere | spot), and an
optional samples per pixel (default 64). The point kinds are tested at 8 samples: their patch-mean
noise there is about 6 x 10^-4 per seed (measured on 8 seeds), against a tolerance of 0.01, a margin
of about 12 standard errors. The spot light keeps 64 samples: at 8 its noise gives a margin of 3.
  point_soft:   point light, radius, soft falloff on (Blender's default, a disk facing the receiver)
  point_sphere: point light, radius, soft falloff off (a sphere)
  spot:         spot light, radius, soft falloff on, spot size 1.0 rad, blend 0.3
"""

import json
import sys

import bpy

blend_out, exr_out, json_out, kind = sys.argv[sys.argv.index("--") + 1:][:4]
extra = sys.argv[sys.argv.index("--") + 1:][4:]
SAMPLES = int(extra[0]) if extra else 64

WIDTH = 64
HEIGHT = 64
LIGHT_HEIGHT = 1.5
LIGHT_ENERGY = 100.0
LIGHT_RADIUS = 0.5
# Fractions of the frame: [0.25, 0.75] spans +-1 m on the plane under the 4 m orthographic frame.
PATCH = [0.25 + 0.5 * index / 8 for index in range(9)]

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = SAMPLES
scene.cycles.use_denoising = False
scene.cycles.max_bounces = 0
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
material = bpy.data.materials.new("Lambert")
material.use_nodes = True
bsdf = material.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
bsdf.inputs["Metallic"].default_value = 0.0
bsdf.inputs["Roughness"].default_value = 1.0
bsdf.inputs["Specular IOR Level"].default_value = 0.0
plane.data.materials.append(material)

if kind == "spot":
    light_data = bpy.data.lights.new("Spot", "SPOT")
    light_data.spot_size = 1.0
    light_data.spot_blend = 0.3
    light_data.use_soft_falloff = True
else:
    light_data = bpy.data.lights.new("Point", "POINT")
    light_data.use_soft_falloff = kind == "point_soft"
light_data.energy = LIGHT_ENERGY
light_data.shadow_soft_size = LIGHT_RADIUS
light = bpy.data.objects.new(light_data.name, light_data)
scene.collection.objects.link(light)
# A spot light points down its local -Z, which is -Z in world space at the default rotation.
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
