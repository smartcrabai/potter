"""Blender 5.2.2's transmittance through uniform VDB grids, for the padded-fade convention (absorption only).

Each case is an n^3 grid of ones at voxel size v, whose first node sits at origin (index i at origin + i * v, the transform
carries both). The density is `density`. A ray through the grid's centre crosses an effective length n * v under Cycles'
volume mesh, which pads the active box by one voxel: (n - 1) * v of nodes plus one padded voxel on the far side, so the
padded-model transmittance of a plateau ray is exp(-density * n * v). Both cases have density * n * v = 0.7, so both
predict exp(-0.7) = 0.4966 for a pixel whose footprint lies inside the node span.

Absorption in a heterogeneous Cycles volume is ratio tracked, so each render is noisy. The generator renders several seeds
at 4096 samples and records their mean and standard error. A 64-sample render is not converged: runs at 64 samples
differed by up to 0.03 between seeds. The measured block is a 4 x 4 pixel block: its outer pixels' footprints reach the
padded fade, so their expectation is the footprint integral, not the centre value. Converged results: 16^3 transmits
0.4932 +- 0.0007 against 0.4966 (plateau, -0.7 %); 4^3 transmits 0.5607 +- 0.0023 against a footprint-integrated
padded 0.5447 (+3 %, not explained; centre-pixel 2 x 2 at density 0.7: 0.4906 against 0.4966).

Regenerate from this directory with Blender 5.2.2, one thread, one run (openvdb must be importable):

    Blender -b --factory-startup --threads 1 -P generate_padded_fade_check.py -- OUT.json
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
work = tempfile.mkdtemp(prefix="padded_fade_check_")
SIZE = 8
SAMPLES = 4096
SEEDS = [0, 1, 2, 3]
BLOCK = range(2, 6)
CASES = {
    "uniform_16cubed": {"dims": 16, "voxel": 0.125, "origin": -0.9375, "density": 0.35},
    "uniform_4cubed": {"dims": 4, "voxel": 0.25, "origin": -0.375, "density": 0.7},
}


def write_grid(case_name, case):
    grid = openvdb.FloatGrid()
    grid.name = "density"
    grid.transform = openvdb.createLinearTransform(case["voxel"])
    grid.transform.postTranslate((case["origin"],) * 3)
    accessor = grid.getAccessor()
    n = case["dims"]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                accessor.setValueOn((i, j, k), case["density"])
    path = os.path.join(work, case_name + ".vdb")
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


records = {}
for name, case in CASES.items():
    vdb_path = write_grid(name, case)
    per_seed = []
    for seed in SEEDS:
        build_scene(vdb_path, seed)
        per_seed.append(centre_transmittance())
    mean = sum(per_seed) / len(per_seed)
    variance = sum((value - mean) ** 2 for value in per_seed) / (len(per_seed) - 1)
    standard_error = math.sqrt(variance / len(per_seed))
    records[name] = {
        "transmittance": round(mean, 6),
        "stderr": round(standard_error, 6),
        "per_seed": [round(value, 6) for value in per_seed],
        "padded_model": round(math.exp(-case["density"] * case["dims"] * case["voxel"]), 6),
    }
    print("ORACLE_CASE", name, records[name])

with open(out_path, "w") as handle:
    json.dump({"blender": bpy.app.version_string, "samples": SAMPLES, "seeds": SEEDS, "cases": records}, handle, indent=1)
print("ORACLE_OK padded_fade_check")
