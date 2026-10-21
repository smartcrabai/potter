"""Bump node slope on a plane, rendered by Blender 5.2.2 Cycles.

The plane's height is the Generated X coordinate, a ramp from 0 to 1 across its two-unit width, bumped with Distance
0.5 and the node's default Filter Width. The material emits the bumped normal, mapped from [-1, 1] to [0, 1] (Vector
Math multiply then add), so the frame mean measures the slope the bump draws at the footprint of each pixel. The
camera looks straight down, and the world is black.

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
# Potter samples each pixel with a one-pixel box, so the oracle uses Cycles' box filter of width 1, as the other oracles do.
scene.cycles.pixel_filter_type = "BOX"
scene.cycles.filter_width = 1.0
scene.cycles.seed = 0
scene.render.resolution_x = 16
scene.render.resolution_y = 16
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.view_settings.view_transform = "Standard"

bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, 0.0))
plane = bpy.context.object

material = bpy.data.materials.new("Bump")
material.use_nodes = True
tree = material.node_tree
for node in list(tree.nodes):
    tree.nodes.remove(node)
output = tree.nodes.new("ShaderNodeOutputMaterial")
emission = tree.nodes.new("ShaderNodeEmission")
emission.inputs["Strength"].default_value = 1.0
texcoord = tree.nodes.new("ShaderNodeTexCoord")
separate = tree.nodes.new("ShaderNodeSeparateXYZ")
bump = tree.nodes.new("ShaderNodeBump")
bump.inputs["Distance"].default_value = 0.5
multiply = tree.nodes.new("ShaderNodeVectorMath")
multiply.operation = "MULTIPLY"
multiply.inputs[1].default_value = (0.5, 0.5, 0.5)
add = tree.nodes.new("ShaderNodeVectorMath")
add.operation = "ADD"
add.inputs[1].default_value = (0.5, 0.5, 0.5)

tree.links.new(texcoord.outputs["Generated"], separate.inputs["Vector"])
tree.links.new(separate.outputs["X"], bump.inputs["Height"])
tree.links.new(bump.outputs["Normal"], multiply.inputs[0])
tree.links.new(multiply.outputs["Vector"], add.inputs[0])
tree.links.new(add.outputs["Vector"], emission.inputs["Color"])
tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
plane.data.materials.append(material)

camera_data = bpy.data.cameras.new("Camera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 2.4
camera = bpy.data.objects.new("Camera", camera_data)
scene.collection.objects.link(camera)
camera.location = Vector((0.0, 0.0, 5.0))
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
    scene.render.filepath = os.path.join(work, f"bump_seed_{seed}.exr")
    bpy.ops.render.render(write_still=True)
    seed_means.append(frame_mean(scene.render.filepath))

with open(json_out, "w") as handle:
    json.dump({"samples": 64, "seed_means": seed_means}, handle, indent=2)
print("DONE", flush=True)
