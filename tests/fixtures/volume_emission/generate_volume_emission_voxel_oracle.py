"""Blender 5.2.2's emitted radiance of an emissive VDB volume, for tests/volume_emission.rs.

The grid: a 16 x 16 x 16 VDB density grid of ones, voxel size 0.125, placed so that index (i, j, k) sits at world
(-0.9375 + 0.125 i, -0.9375 + 0.125 j, -0.9375 + 0.125 k), the node convention of Potter's inline grids. Its material is a
Principled Volume with Density 2, Emission Color 1 and Emission Strength 1, and a Color of 0 (absorption only) or 0.5 (half
scatter). The world is black and there is no light, so a camera ray's radiance is the emission integral through the grid,
taken from the camera side, plus what a scattered path collects along its terminal ray:

    L = integral emission(s) * exp(-sigma_t * s) ds,  sigma_t = Density * grid value = 2, emission 1.

With Color 0 this is the closed form for a unit-density grid ((1 - e^-4) / 2 = 0.4908 for a length of 2), which the grid's
boundary fade raises slightly. Color 0.5 is stochastic (the grid scatters), so this exercises the stochastic-grid emission:
its collisions end the emission, and it takes no extra transmittance weight. The generator records the mean and standard
error of the 4 x 4 pixels at the image centre, per channel. The bounce budget is Blender's default for volumes (max_bounces
0, volume_bounces 0).

Regenerate from this directory with Blender 5.2.2, one thread, one run (openvdb must be importable):

    Blender -b --factory-startup --threads 1 -P generate_volume_emission_voxel_oracle.py -- OUT.json
"""

import json
import math
import os
import sys
import tempfile

import bpy
import openvdb

argv = sys.argv[sys.argv.index("--") + 1 :]
out_path = os.path.abspath(argv[0])
work = tempfile.mkdtemp(prefix="volume_emission_voxel_")
SIZE = 24
SAMPLES = 1024
BLOCK = range(10, 14)
VOXEL = 0.125
ORIGIN = -0.9375
DIMS = 16
DENSITY = 2.0
COLORS = {"color0": 0.0, "color05": 0.5}


def write_grid():
    grid = openvdb.FloatGrid()
    grid.name = "density"
    grid.transform = openvdb.createLinearTransform(VOXEL)
    grid.transform.postTranslate((ORIGIN, ORIGIN, ORIGIN))
    grid.fill((0, 0, 0), (DIMS - 1, DIMS - 1, DIMS - 1), 1.0, True)
    path = os.path.join(work, "emissive_grid.vdb")
    openvdb.write(path, grids=[grid])
    return path


def build_scene(vdb_path, color):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    render = scene.render
    render.engine = "CYCLES"
    render.resolution_x = SIZE
    render.resolution_y = SIZE
    render.resolution_percentage = 100
    render.film_transparent = False
    scene.cycles.device = "CPU"
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = 0
    scene.cycles.volume_bounces = 0
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    image_settings = render.image_settings
    image_settings.file_format = "OPEN_EXR"
    image_settings.color_depth = "32"
    image_settings.color_mode = "RGBA"
    image_settings.exr_codec = "NONE"

    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    scene.world = world
    background = world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    background.inputs["Strength"].default_value = 1.0

    volume_data = bpy.data.volumes.new("EmissiveGrid")
    volume_data.filepath = vdb_path
    volume = bpy.data.objects.new("EmissiveGrid", volume_data)
    scene.collection.objects.link(volume)
    volume.location = (0.0, 0.0, 0.0)

    material = bpy.data.materials.new("emitter")
    material.use_nodes = True
    tree = material.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    principled = tree.nodes.new("ShaderNodeVolumePrincipled")
    principled.inputs["Color"].default_value = (color, color, color, 1.0)
    principled.inputs["Density"].default_value = DENSITY
    principled.inputs["Emission Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    principled.inputs["Emission Strength"].default_value = 1.0
    tree.links.new(principled.outputs["Volume"], output.inputs["Volume"])
    volume.data.materials.append(material)

    camera_data = bpy.data.cameras.new("camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 2.0
    camera = bpy.data.objects.new("camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0.0, 0.0, 3.0)
    camera.rotation_euler = (0.0, 0.0, 0.0)
    scene.camera = camera


def render_block(name):
    path = os.path.join(work, name + ".exr")
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    loaded = bpy.data.images.load(path, check_existing=False)
    pixels = list(loaded.pixels)
    bpy.data.images.remove(loaded)
    channels = [[], [], []]
    for y in BLOCK:
        for x in BLOCK:
            offset = (y * SIZE + x) * 4
            for channel in range(3):
                channels[channel].append(pixels[offset + channel])
    means, errors = [], []
    for values in channels:
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        means.append(round(mean, 6))
        errors.append(round(math.sqrt(variance / len(values)), 6))
    return means, errors


vdb_path = write_grid()
records = {}
for name, color in COLORS.items():
    build_scene(vdb_path, color)
    means, errors = render_block("emissive_voxel_" + name)
    records["emissive_voxel_" + name] = {"mean": means, "stderr": errors}
    print("ORACLE_CASE emissive_voxel_" + name, means, errors)

with open(out_path, "w") as handle:
    json.dump(
        {
            "blender": bpy.app.version_string,
            "samples": SAMPLES,
            "density": DENSITY,
            "max_bounces": 0,
            "volume_bounces": 0,
            "cases": records,
        },
        handle,
        indent=1,
    )
print("ORACLE_OK volume_emission_voxel", json.dumps(records))
