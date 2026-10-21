"""Blender 5.2.2's transmittance through a 2 x 2 x 2 density grid of 0.7 (absorption only), for tests/render_world.rs.

The grid: nodes at -0.5 and 0.5 on each axis, voxel size 1, every node 0.7, as Potter's inline grid of the same name
(`tests/render_world.rs`, homogeneous_volume_cube). It is a VDB with those eight active voxels, in a Principled Volume with
Color 0 (absorption only) and Density 1, so the extinction is 0.7 where the grid is full. An orthographic camera at z = 3
looks through the centre. The film is transparent and there is no light, so the centre alpha is 1 - transmittance.

Absorption in a heterogeneous Cycles volume is ratio tracked, so each render is noisy. The generator renders several
seeds at 4096 samples and records their mean and standard error, which the test compares with a three-standard-error
band. A 64-sample render, which earlier runs used, is not converged: its mean moved by about 0.03.

Regenerate from this directory with Blender 5.2.2, one thread, one run (openvdb must be importable):

    Blender -b --factory-startup --threads 1 -P generate_inline_grid_transmittance_oracle.py -- OUT.json
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
work = tempfile.mkdtemp(prefix="inline_grid_transmittance_")
SIZE = 24
SAMPLES = 4096
SEEDS = [0, 1, 2, 3]
BLOCK = range(10, 14)
DENSITY = 0.7


def write_grid():
    grid = openvdb.FloatGrid()
    grid.name = "density"
    grid.transform = openvdb.createLinearTransform(1.0)
    grid.transform.postTranslate((-0.5, -0.5, -0.5))
    grid.fill((0, 0, 0), (1, 1, 1), DENSITY, True)
    path = os.path.join(work, "grid.vdb")
    openvdb.write(path, grids=[grid])
    return path


def build_scene(vdb_path, seed):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    render = scene.render
    render.engine = "CYCLES"
    render.resolution_x = SIZE
    render.resolution_y = SIZE
    render.resolution_percentage = 100
    render.film_transparent = True
    scene.cycles.device = "CPU"
    scene.cycles.samples = SAMPLES
    scene.cycles.seed = seed
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

    volume_data = bpy.data.volumes.new("Grid")
    volume_data.filepath = vdb_path
    volume = bpy.data.objects.new("Grid", volume_data)
    scene.collection.objects.link(volume)
    volume.location = (0.0, 0.0, 0.0)

    material = bpy.data.materials.new("absorber")
    material.use_nodes = True
    tree = material.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    principled = tree.nodes.new("ShaderNodeVolumePrincipled")
    principled.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    principled.inputs["Density"].default_value = 1.0
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


def centre_transmittance():
    path = os.path.join(work, "render.exr")
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    loaded = bpy.data.images.load(path, check_existing=False)
    pixels = list(loaded.pixels)
    bpy.data.images.remove(loaded)
    alphas = [pixels[(y * SIZE + x) * 4 + 3] for y in BLOCK for x in BLOCK]
    return 1.0 - sum(alphas) / len(alphas)


vdb_path = write_grid()
per_seed = []
for seed in SEEDS:
    build_scene(vdb_path, seed)
    per_seed.append(centre_transmittance())
mean = sum(per_seed) / len(per_seed)
variance = sum((value - mean) ** 2 for value in per_seed) / (len(per_seed) - 1)
standard_error = math.sqrt(variance / len(per_seed))
print("ORACLE_CASE inline_grid_transmittance", round(mean, 6), round(standard_error, 6), per_seed)

with open(out_path, "w") as handle:
    json.dump(
        {
            "blender": bpy.app.version_string,
            "samples": SAMPLES,
            "seeds": SEEDS,
            "density": DENSITY,
            "cases": {
                "inline_grid_transmittance": {
                    "transmittance": round(mean, 6),
                    "stderr": round(standard_error, 6),
                    "per_seed": [round(value, 6) for value in per_seed],
                }
            },
        },
        handle,
        indent=1,
    )
print("ORACLE_OK inline_grid_transmittance")
