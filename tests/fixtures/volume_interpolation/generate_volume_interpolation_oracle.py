"""Blender 5.2.2 single-scatter oracle for a voxel grid under LINEAR and CUBIC volume interpolation.

The grid is 16 x 16 x 16 nodes at index (i, j, k), with value v = h(i, j, k) / 1023 where h = ((i * 73856093) ^
(j * 19349663) ^ (k * 83492791)) & 1023, a hash that is noisy enough for the cubic B-spline to differ from the
trilinear sample. It is written to a VDB named "density" (background 0, every node active), imported as a Volume
object placed so that index (i, j, k) sits at world origin + (i, j, k) / 16, with origin (-0.5, -0.5, -0.5), the
same node convention as Potter's inline grids. The material is a Volume Scatter (Color 1, Anisotropy 0) with
Density = 0.05 * Volume Info density, so the optical depth is small and the render is single scatter, as in
generate_volume_nee_oracle.py: volume_bounces = max_bounces = 1, a point light of 100 W at (1.6, 0.6, 0.9),
an orthographic camera at (0, 0, 3) with ortho scale 2, and a black world. Each case records the mean and the
standard error of the 4 x 4 pixels at the image centre.

Regenerate from this directory with Blender 5.2.2 (openvdb must be importable), one thread, one run:

    Blender -b --factory-startup --threads 1 -P generate_volume_interpolation_oracle.py -- OUT.json
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
work = tempfile.mkdtemp(prefix="volume_interpolation_")
SIZE = 24
SAMPLES = 65536
BLOCK = range(10, 14)
ENERGY = 100.0
POSITION = (1.6, 0.6, 0.9)
DENSITY_SCALE = 0.05
N = 16
ORIGIN = -0.5
VOXEL = 1.0 / N


def hashed(i, j, k):
    return ((i * 73856093) ^ (j * 19349663) ^ (k * 83492791)) & 1023


def write_vdb(path):
    grid = openvdb.FloatGrid()
    grid.name = "density"
    accessor = grid.getAccessor()
    for i in range(N):
        for j in range(N):
            for k in range(N):
                accessor.setValueOn((i, j, k), hashed(i, j, k) / 1023.0)
    openvdb.write(path, grids=[grid])


def reset(interpolation):
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
    scene.cycles.max_bounces = 1
    scene.cycles.volume_bounces = 1
    scene.cycles.transparent_max_bounces = 64
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    image = render.image_settings
    image.file_format = "OPEN_EXR"
    image.color_depth = "32"
    image.color_mode = "RGBA"
    image.exr_codec = "NONE"
    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    scene.world = world
    background = world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    background.inputs["Strength"].default_value = 0.0
    camera_data = bpy.data.cameras.new("camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 2.0
    camera = bpy.data.objects.new("camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0.0, 0.0, 3.0)
    camera.rotation_euler = (0.0, 0.0, 0.0)
    scene.camera = camera
    light_data = bpy.data.lights.new("point", "POINT")
    light_data.energy = ENERGY
    light_data.shadow_soft_size = 0.0
    light = bpy.data.objects.new("point", light_data)
    scene.collection.objects.link(light)
    light.location = POSITION
    return scene


def voxel_volume(vdb_path, interpolation):
    bpy.ops.object.volume_import(filepath=vdb_path)
    volume = bpy.context.active_object
    volume.location = (ORIGIN, ORIGIN, ORIGIN)
    volume.scale = (VOXEL, VOXEL, VOXEL)
    material = bpy.data.materials.new("voxels")
    material.use_nodes = True
    material.cycles.volume_interpolation = interpolation
    tree = material.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    scatter = tree.nodes.new("ShaderNodeVolumeScatter")
    scatter.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    scatter.inputs["Anisotropy"].default_value = 0.0
    info = tree.nodes.new("ShaderNodeVolumeInfo")
    scale = tree.nodes.new("ShaderNodeMath")
    scale.operation = "MULTIPLY"
    scale.inputs[1].default_value = DENSITY_SCALE
    tree.links.new(info.outputs["Density"], scale.inputs[0])
    tree.links.new(scale.outputs["Value"], scatter.inputs["Density"])
    tree.links.new(scatter.outputs["Volume"], output.inputs["Volume"])
    volume.data.materials.append(material)


def render_block(name):
    path = os.path.join(work, name + ".exr")
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    loaded = bpy.data.images.load(path, check_existing=False)
    pixels = list(loaded.pixels)
    bpy.data.images.remove(loaded)
    values = []
    for y in BLOCK:
        for x in BLOCK:
            values.append(pixels[(y * SIZE + x) * 4])
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    # The whole image's red channel, row by row from the top, so Potter's per-pixel values can be compared with
    # Blender's. The block mean alone barely separates LINEAR from CUBIC: the B-spline preserves the field's mean.
    image = [round(pixels[(y * SIZE + x) * 4], 6) for y in range(SIZE) for x in range(SIZE)]
    return round(mean, 6), round(math.sqrt(variance / len(values)), 6), image


vdb_path = os.path.join(work, "density.vdb")
write_vdb(vdb_path)
records = {}
for name, interpolation in (("linear_noise", "LINEAR"), ("cubic_noise", "CUBIC")):
    reset(interpolation)
    voxel_volume(vdb_path, interpolation)
    mean, stderr, image = render_block(name)
    records[name] = {"mean": mean, "stderr": stderr, "pixels": image}
    print("ORACLE_CASE", name, mean, stderr)

with open(out_path, "w") as handle:
    json.dump(
        {
            "blender": bpy.app.version_string,
            "samples": SAMPLES,
            "energy": ENERGY,
            "position": list(POSITION),
            "density_scale": DENSITY_SCALE,
            "image_size": SIZE,
            "grid": {"dims": [N, N, N], "voxel_size": VOXEL, "origin": [ORIGIN] * 3},
            "cases": records,
        },
        handle,
        indent=1,
    )
print("ORACLE_OK volume_interpolation", json.dumps(records))
