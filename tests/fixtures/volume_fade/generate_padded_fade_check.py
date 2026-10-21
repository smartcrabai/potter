"""Blender 5.2.2 transmittance of a 2 x 2 x 2 VDB density grid, voxel size 0.5, density 1.4 (absorption only).

Box convention (nodes span the box): e^(-1.4 * 0.5) = 0.4966. Padded-active convention (one voxel of fade each side of
the node span, as Cycles' create_volume_mesh dilates the active voxels and takes corners max+1): the optical depth is
1.4 * (0.25 + 0.5 + 0.25) = 1.4 -> 0.2466 of the centre ray, i.e. the same as the 2-unit case in render_world.
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
work = tempfile.mkdtemp(prefix="grid_alpha_")
SIZE = 24
SAMPLES = 64
BLOCK = range(10, 14)
VOXEL = 0.125
ORIGIN = -0.9375
DENSITY = 0.35

grid = openvdb.FloatGrid()
grid.name = "density"
grid.transform = openvdb.createLinearTransform(VOXEL)
grid.transform.postTranslate((ORIGIN, ORIGIN, ORIGIN))
grid.fill((0, 0, 0), (15, 15, 15), 1.0, True)
vdb_path = os.path.join(work, "grid.vdb")
openvdb.write(vdb_path, grids=[grid])

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
principled.inputs["Density"].default_value = DENSITY
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

path = os.path.join(work, "render.exr")
render.filepath = path
bpy.ops.render.render(write_still=True)
loaded = bpy.data.images.load(path, check_existing=False)
pixels = list(loaded.pixels)
bpy.data.images.remove(loaded)
alphas = [pixels[(y * SIZE + x) * 4 + 3] for y in BLOCK for x in BLOCK]
mean_alpha = sum(alphas) / len(alphas)
transmittance = 1.0 - mean_alpha
print("ORACLE_CASE grid_alpha transmittance", transmittance, "alpha", mean_alpha)
with open(out_path, "w") as handle:
    json.dump({"transmittance": transmittance, "mean_alpha": mean_alpha}, handle)
print("BOX_nodes", math.exp(-0.35*1.875), "PADDED_1voxel", math.exp(-0.35*2.0))
