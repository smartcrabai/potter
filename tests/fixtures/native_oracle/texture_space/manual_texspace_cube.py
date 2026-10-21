"""Texture Coordinate Generated on a cube with a manual texture space, rendered by Blender 5.2.2 Cycles.

The cube's mesh has its automatic texture space turned off and a stored texture space of location (0.5, -0.25, 0.1)
and size (2, 1, 1.5), which differs from the cube's own bounds. The Generated coordinate is then
`(P - location) * 0.5 / size + 0.5` per axis (Cycles' `mesh_texture_space`), so the frame mean measures the stored
texture space rather than the bounds. The material emits the Generated coordinate as a colour (Emission) and the
world is black, so the frame mean is the mean of the coordinate over the cube's visible pixels, with the camera
looking along +Y at the face at local y = -1.

Arguments after `--`: blend_out and json_out.
"""

import json
import os
import sys
import tempfile

import bpy
from mathutils import Vector

blend_out, json_out = sys.argv[sys.argv.index("--") + 1:][:2]

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 64
scene.cycles.use_denoising = False
scene.cycles.max_bounces = 0
scene.cycles.seed = 0
scene.render.resolution_x = 16
scene.render.resolution_y = 16
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.view_settings.view_transform = "Standard"

bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
cube = bpy.context.object
cube.data.use_auto_texspace = False
cube.data.texspace_location = (0.5, -0.25, 0.1)
cube.data.texspace_size = (2.0, 1.0, 1.5)

material = bpy.data.materials.new("Generated")
material.use_nodes = True
tree = material.node_tree
for node in list(tree.nodes):
    tree.nodes.remove(node)
output = tree.nodes.new("ShaderNodeOutputMaterial")
emission = tree.nodes.new("ShaderNodeEmission")
emission.inputs["Strength"].default_value = 1.0
texcoord = tree.nodes.new("ShaderNodeTexCoord")
tree.links.new(texcoord.outputs["Generated"], emission.inputs["Color"])
tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
cube.data.materials.append(material)

camera_data = bpy.data.cameras.new("Camera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 2.4
camera = bpy.data.objects.new("Camera", camera_data)
scene.collection.objects.link(camera)
camera.location = Vector((0.0, -10.0, 0.0))
camera.rotation_euler = (1.5707963267948966, 0.0, 0.0)
scene.camera = camera

bpy.ops.wm.save_as_mainfile(filepath=blend_out)


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


work = tempfile.mkdtemp()
seed_means = []
for seed in range(8):
    scene.cycles.seed = seed
    scene.render.filepath = os.path.join(work, f"manual_seed_{seed}.exr")
    bpy.ops.render.render(write_still=True)
    seed_means.append(frame_mean(scene.render.filepath))

with open(json_out, "w") as handle:
    json.dump({"samples": 64, "seed_means": seed_means}, handle, indent=2)
print("DONE", flush=True)
