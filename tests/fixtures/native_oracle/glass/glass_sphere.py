"""Rough glass sphere under a point light and a disk light, rendered by Blender's Cycles (v5.2.2).

A unit sphere with a Principled BSDF (Base Color white, Transmission Weight 1, IOR 1.5, Roughness 0.2) sits on
the origin in an empty black world. A point light and a disk light (radius 0.75) sit above and in front of it;
an orthographic camera looks along +Y at the sphere. `max_bounces` is 3: camera, front surface, back surface,
then the exit ray can reach a light. Shadow rays from inside the sphere are blocked by the glass, as Cycles
blocks every closure without transparency (`kernel/integrator/surface_shader.h`, `surface_shader_transparency`).

Eight seeds are rendered, and each whole-frame mean is written per seed, so the Potter test can take the
seed-to-seed spread as the Monte Carlo error.

Arguments after `--`: blend_out, json_out.
"""

import json
import os
import sys
import tempfile

import bpy
from mathutils import Vector

blend_out, json_out = sys.argv[sys.argv.index("--") + 1:][:2]
ROUGHNESS = 0.2
SEEDS = list(range(8))
RESOLUTION = 16
SAMPLES = 256
ORTHO_SCALE = 2.6
CAMERA = Vector((0.0, -6.0, 0.0))
POINT_LOCATION = Vector((-3.0, -2.0, 4.0))
POINT_POWER = 120.0
DISK_LOCATION = Vector((2.5, -2.5, 3.5))
DISK_SIZE = 1.5
DISK_POWER = 300.0


def aim(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = SAMPLES
scene.cycles.use_denoising = False
scene.cycles.max_bounces = 3
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0
scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"

world = bpy.data.worlds.new("Black")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
scene.world = world

bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=64, ring_count=32, location=(0.0, 0.0, 0.0))
sphere = bpy.context.object
bpy.ops.object.shade_smooth()
glass = bpy.data.materials.new("Glass")
glass.use_nodes = True
shader = glass.node_tree.nodes["Principled BSDF"]
shader.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
shader.inputs["Metallic"].default_value = 0.0
shader.inputs["Roughness"].default_value = ROUGHNESS
shader.inputs["IOR"].default_value = 1.5
shader.inputs["Transmission Weight"].default_value = 1.0
sphere.data.materials.append(glass)

point_data = bpy.data.lights.new("Point", "POINT")
point_data.energy = POINT_POWER
point_data.shadow_soft_size = 0.0
point = bpy.data.objects.new("Point", point_data)
scene.collection.objects.link(point)
point.location = POINT_LOCATION

disk_data = bpy.data.lights.new("Disk", "AREA")
disk_data.shape = "DISK"
disk_data.size = DISK_SIZE
disk_data.energy = DISK_POWER
disk = bpy.data.objects.new("Disk", disk_data)
scene.collection.objects.link(disk)
disk.location = DISK_LOCATION
aim(disk, (0.0, 0.0, 0.0))

camera_data = bpy.data.cameras.new("Camera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = ORTHO_SCALE
camera = bpy.data.objects.new("Camera", camera_data)
scene.collection.objects.link(camera)
camera.location = CAMERA
aim(camera, (0.0, 0.0, 0.0))
scene.camera = camera

bpy.ops.wm.save_as_mainfile(filepath=blend_out)

work = tempfile.mkdtemp()


def frame_mean(path):
    image = bpy.data.images.load(path, check_existing=False)
    width, height = image.size
    pixels = list(image.pixels)
    totals = [0.0, 0.0, 0.0]
    for index in range(width * height):
        for channel in range(3):
            totals[channel] += pixels[4 * index + channel]
    bpy.data.images.remove(image)
    return [total / (width * height) for total in totals]


seed_means = []
for seed in SEEDS:
    scene.cycles.seed = seed
    scene.render.filepath = os.path.join(work, f"seed_{seed}.exr")
    bpy.ops.render.render(write_still=True)
    seed_means.append(frame_mean(scene.render.filepath))

with open(json_out, "w") as handle:
    json.dump({"roughness": ROUGHNESS, "samples": SAMPLES, "seed_means": seed_means}, handle, indent=2)
print("DONE", flush=True)
