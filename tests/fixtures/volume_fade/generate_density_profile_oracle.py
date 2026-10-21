"""Blender 5.2.2's effective density profile of a uniform VDB grid along one axis, for the padded-fade convention.

The grid: n^3 active voxels of value 1 at voxel size v, with the first node at `origin` on each axis (index i at
origin + i * v, the convention of Potter's inline grids and of openvdb's voxel-centre transform). A Principled Volume
with Color 0 (absorption only) and Density 1 makes the extinction equal the grid value. The camera is orthographic
on the +X axis looking along -X, so each pixel column integrates the density along X at one (y, z) position. The centre
row (y = 0) therefore gives, per column, -ln(T) proportional to the density's Z profile, since the grid is separable:
-ln T(y, z) = c * f(y) * f(z) with f the 1-D effective profile of Cycles' volume mesh and sampler.

Output: the column centre positions in world Z and the measured -ln T per column (centre row), per case. The comparison
with the padded model (integral ratio and per-column residual) is an offline analysis of this JSON, not a test.

Regenerate from this directory with Blender 5.2.2, one thread, one run (openvdb must be importable):

    Blender -b --factory-startup --threads 1 -P generate_density_profile_oracle.py -- OUT.json
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
work = tempfile.mkdtemp(prefix="density_profile_")

CASES = {
    "n2_v05": {"dims": 2, "voxel": 0.5, "origin": -0.25},
    "n4_v025": {"dims": 4, "voxel": 0.25, "origin": -0.375},
}
WIDTH = 384  # columns along Z
HEIGHT = 1  # one row at y = 0: a column's -ln T is one ray's integral
SAMPLES = 4096
ORTHO = 2.0 * 1.6


def write_grid(dims, voxel, origin):
    grid = openvdb.FloatGrid()
    grid.name = "density"
    grid.transform = openvdb.createLinearTransform(voxel)
    grid.transform.postTranslate((origin, origin, origin))
    grid.fill((0, 0, 0), (dims - 1, dims - 1, dims - 1), 1.0, True)
    path = os.path.join(work, "grid_%d.vdb" % dims)
    openvdb.write(path, grids=[grid])
    return path


def build_scene(vdb_path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    render = scene.render
    render.engine = "CYCLES"
    render.resolution_x = WIDTH
    render.resolution_y = HEIGHT
    render.resolution_percentage = 100
    render.film_transparent = True
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
    camera_data.ortho_scale = ORTHO
    camera = bpy.data.objects.new("camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (3.0, 0.0, 0.0)
    camera.rotation_euler = (0.0, math.pi / 2.0, 0.0)
    scene.camera = camera


def centre_row_profile():
    path = os.path.join(work, "render.exr")
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    loaded = bpy.data.images.load(path, check_existing=False)
    pixels = list(loaded.pixels)
    bpy.data.images.remove(loaded)
    row = HEIGHT // 2
    profile = []
    for column in range(WIDTH):
        alpha = pixels[(row * WIDTH + column) * 4 + 3]
        transmittance = max(1.0 - alpha, 1.0e-6)
        profile.append(-math.log(transmittance))
    return profile


results = {}
for name, case in CASES.items():
    vdb_path = write_grid(case["dims"], case["voxel"], case["origin"])
    build_scene(vdb_path)
    profile = centre_row_profile()
    results[name] = {
        "dims": case["dims"],
        "voxel": case["voxel"],
        "origin": case["origin"],
        "ortho": ORTHO,
        "width": WIDTH,
        "neg_log_transmittance_centre_row": [round(value, 6) for value in profile],
    }
    print("PROFILE", name, "max", round(max(profile), 6))

with open(out_path, "w") as handle:
    json.dump({"blender": bpy.app.version_string, "samples": SAMPLES, "cases": results}, handle, indent=1)
print("PROFILE_OK")
