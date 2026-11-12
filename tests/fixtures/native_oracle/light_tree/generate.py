"""Generate Cycles light-tree oracle scenes and eight-seed patch means.

Arguments after --: blend_out, json_out, scenario, samples, seeds.
"""
import json
import math
import os
import sys

import bpy
from mathutils import Vector

blend_out, json_out, scenario, samples, seeds = sys.argv[sys.argv.index("--") + 1:][:5]
samples = int(samples)
seeds = int(seeds)
width = 16
height = 16
patch = range(6, 10)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = samples
scene.cycles.use_adaptive_sampling = False
scene.cycles.use_denoising = False
scene.cycles.use_light_tree = True
scene.cycles.max_bounces = 4
scene.cycles.diffuse_bounces = 4
scene.cycles.glossy_bounces = 4
scene.cycles.transmission_bounces = 4
scene.cycles.volume_bounces = 0
scene.cycles.sample_clamp_indirect = 10.0
scene.cycles.sample_clamp_direct = 5.0 if scenario == "volume_segment" else 0.0
scene.cycles.pixel_filter_type = "BOX"
scene.cycles.filter_width = 1.0
scene.render.resolution_x = width
scene.render.resolution_y = height
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0

world = bpy.data.worlds.new("TreeWorld")
world.use_nodes = True
background = world.node_tree.nodes.get("Background")
if scenario == "sun_local_world":
    background.inputs["Color"].default_value = (0.15, 0.2, 0.3, 1.0)
    background.inputs["Strength"].default_value = 0.4
else:
    background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    background.inputs["Strength"].default_value = 0.0
scene.world = world


def material(name, kind):
    result = bpy.data.materials.new(name)
    result.use_nodes = True
    nodes = result.node_tree.nodes
    links = result.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    if kind == "mirror":
        shader = nodes.new("ShaderNodeBsdfGlossy")
        shader.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1.0)
        shader.inputs["Roughness"].default_value = 0.0
        links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    elif kind == "diffuse":
        shader = nodes.new("ShaderNodeBsdfDiffuse")
        shader.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1.0)
        links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    elif kind == "volume":
        shader = nodes.new("ShaderNodeVolumeScatter")
        shader.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        shader.inputs["Density"].default_value = 0.18
        shader.inputs["Anisotropy"].default_value = 0.2
        links.new(shader.outputs["Volume"], output.inputs["Volume"])
    return result


def plane(name, z, size, mat, flip=False):
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0.0, 0.0, z))
    obj = bpy.context.object
    obj.name = name
    if flip:
        obj.rotation_euler.x = math.pi
    obj.data.materials.append(mat)
    return obj


def point_light(name, energy, location, radius=0.0):
    data = bpy.data.lights.new(name, "POINT")
    data.use_nodes = False
    data.energy = energy
    data.shadow_soft_size = radius
    data.use_soft_falloff = False
    data.use_shadow = False
    data.cycles.use_multiple_importance_sampling = True
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    obj.location = location
    return obj


def area_light(name, energy, location, size, direction=(0.0, 0.0, -1.0)):
    data = bpy.data.lights.new(name, "AREA")
    data.use_nodes = False
    data.use_shadow = False
    data.shape = "DISK"
    data.size = size
    data.energy = energy
    data.cycles.use_multiple_importance_sampling = True
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    obj.location = location
    obj.rotation_euler = Vector(direction).to_track_quat("-Z", "Y").to_euler()
    return obj


def sun_light(name, energy, direction, angle):
    data = bpy.data.lights.new(name, "SUN")
    data.use_nodes = False
    data.use_shadow = False
    data.energy = energy
    data.angle = angle
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    obj.rotation_euler = Vector(direction).to_track_quat("-Z", "Y").to_euler()
    return obj

if scenario in ("multi_light_clamp", "sun_local_world"):
    plane("Mirror", 0.0, 12.0, material("MirrorMaterial", "mirror"))
    plane("Ceiling", 2.0, 12.0, material("CeilingMaterial", "diffuse"), flip=True)
    point_light("NearPowerful", 12000.0, (0.0, 0.0, -0.2), 0.15)
    point_light("FarMedium", 500.0, (3.0, 1.0, -0.2), 0.05)
    point_light("FarWeak", 80.0, (-3.0, -1.0, -0.2), 0.0)
    if scenario == "sun_local_world":
        sun_light("Sun", 2.0, (0.3, 0.2, 1.0), 0.0)
    camera_data = bpy.data.cameras.new("Camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 1.5
    camera = bpy.data.objects.new("Camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0.0, 0.0, 1.0)
    camera.rotation_euler = (0.0, 0.0, 0.0)
    scene.camera = camera
else:
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
    volume = bpy.context.object
    volume.name = "ScatteringVolume"
    volume.data.materials.append(material("VolumeMaterial", "volume"))
    plane("Ground", -1.2, 8.0, material("GroundMaterial", "diffuse"))
    point_light("NearPowerful", 900.0, (0.0, 0.0, 1.5), 0.1)
    point_light("FarMedium", 250.0, (2.0, 1.0, 0.0), 0.05)
    area_light("Area", 300.0, (-1.5, 0.0, 1.0), 0.5)
    camera_data = bpy.data.cameras.new("Camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 1.5
    camera = bpy.data.objects.new("Camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0.0, 0.0, 3.0)
    camera.rotation_euler = (0.0, 0.0, 0.0)
    scene.camera = camera

bpy.ops.wm.save_as_mainfile(filepath=blend_out)
patch_means = []
frame_base = os.path.splitext(json_out)[0]
for seed in range(seeds):
    scene.cycles.seed = seed
    scene.render.filepath = f"{frame_base}_seed{seed}.exr"
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(scene.render.filepath, check_existing=False)
    channels = image.channels
    pixels = list(image.pixels)
    sums = [0.0, 0.0, 0.0]
    count = 0
    for y in patch:
        for x in patch:
            index = (y * width + x) * channels
            for channel in range(3):
                sums[channel] += pixels[index + channel]
            count += 1
    patch_means.append([value / count for value in sums])
    bpy.data.images.remove(image)
with open(json_out, "w", encoding="utf-8") as handle:
    json.dump({"scenario": scenario, "samples": samples, "patch_means": patch_means}, handle)
