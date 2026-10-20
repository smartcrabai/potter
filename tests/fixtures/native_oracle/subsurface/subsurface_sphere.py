"""Principled subsurface scattering on a sphere under a point light and a disk light, rendered by Blender's
Cycles (v5.2.2), one render per (method, subsurface weight) case.

A unit sphere with a Principled BSDF (Base Color (0.8, 0.2, 0.1), Roughness 0.5, IOR 1.4, Subsurface Radius
(1.0, 0.2, 0.1) with Subsurface Scale 6.0, Subsurface Anisotropy 0) sits on the origin in an empty black world.
A point light and a disk light (radius 0.75) sit above and in front of it; an orthographic camera looks along +Y.
`max_bounces` is 8 so the subsurface walk has room. Each case renders eight seeds at SAMPLES samples per pixel and
writes the whole-frame mean RGB per seed, so the Potter test can take the seed-to-seed spread as the Monte Carlo
error.

Cases: every subsurface method (burley, random_walk, random_walk_skin, random_walk_legacy) with Subsurface Weight
1.0, and random_walk with Subsurface Weight 0.0 (no subsurface: the reference for the subsurface effect).

Arguments after `--`: blend_out, json_out, and optionally a case name (burley, random_walk, random_walk_skin,
random_walk_legacy, no_subsurface).
"""

import json
import os
import sys
import tempfile

import bpy
from mathutils import Vector

blend_out, json_out = sys.argv[sys.argv.index("--") + 1:][:2]
# An optional case name renders that case alone (the Potter test runs the Burley case); no name renders every case.
CASE = sys.argv[sys.argv.index("--") + 3:][:1]
SEEDS = list(range(8))
RESOLUTION = 16
SAMPLES = 256
MAX_BOUNCES = 8
ORTHO_SCALE = 2.6
CAMERA = Vector((0.0, -6.0, 0.0))
POINT_LOCATION = Vector((-3.0, -2.0, 4.0))
POINT_POWER = 120.0
DISK_LOCATION = Vector((2.5, -2.5, 3.5))
DISK_SIZE = 1.5
DISK_POWER = 300.0
# (name, subsurface method, subsurface weight, max_bounces). `burley_bounce0` is the Burley sphere with a zero budget:
# Cycles still enters the surface and shades its exit (the extra vertex beyond the budget), so the result is not black.
CASES = [
    ("burley", "BURLEY", 1.0, MAX_BOUNCES),
    ("burley_bounce0", "BURLEY", 1.0, 0),
    ("random_walk", "RANDOM_WALK", 1.0, MAX_BOUNCES),
    ("random_walk_skin", "RANDOM_WALK_SKIN", 1.0, MAX_BOUNCES),
    ("random_walk_legacy", "RANDOM_WALK_LEGACY", 1.0, MAX_BOUNCES),
    ("no_subsurface", "RANDOM_WALK", 0.0, MAX_BOUNCES),
]


CASES = [case for case in CASES if not CASE or case[0] == CASE[0]]
if not CASES:
    raise SystemExit(f"unknown case {CASE}")


def aim(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = SAMPLES
scene.cycles.use_denoising = False
scene.cycles.max_bounces = MAX_BOUNCES
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
material = bpy.data.materials.new("Skin")
material.use_nodes = True
shader = material.node_tree.nodes["Principled BSDF"]
shader.inputs["Base Color"].default_value = (0.8, 0.2, 0.1, 1.0)
shader.inputs["Metallic"].default_value = 0.0
shader.inputs["Roughness"].default_value = 0.5
shader.inputs["IOR"].default_value = 1.4
shader.inputs["Subsurface Radius"].default_value = (1.0, 0.2, 0.1)
shader.inputs["Subsurface Scale"].default_value = 6.0
shader.inputs["Subsurface Anisotropy"].default_value = 0.0
sphere.data.materials.append(material)

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

# The saved file carries the first selected case's method and weight, so an import reproduces that case.
shader.subsurface_method = CASES[0][1]
shader.inputs["Subsurface Weight"].default_value = CASES[0][2]
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


results = {}
for name, method, weight, budget in CASES:
    shader.subsurface_method = method
    shader.inputs["Subsurface Weight"].default_value = weight
    scene.cycles.max_bounces = budget
    seed_means = []
    for seed in SEEDS:
        scene.cycles.seed = seed
        scene.render.filepath = os.path.join(work, f"{name}_seed_{seed}.exr")
        bpy.ops.render.render(write_still=True)
        seed_means.append(frame_mean(scene.render.filepath))
    results[name] = {
        "method": method,
        "subsurface_weight": weight,
        "max_bounces": budget,
        "seed_means": seed_means,
    }
    print("CASE", name, flush=True)

with open(json_out, "w") as handle:
    json.dump({"samples": SAMPLES, "max_bounces": MAX_BOUNCES, "cases": results}, handle, indent=2)
print("DONE", flush=True)
