"""Bump at a secondary hit with Filter Glossy on, rendered by Blender 5.2.2 Cycles.

A diffuse floor fills the frame. Outside the frame, to the left, a bumped emitter faces down onto the floor: its height
is the Generated X ramp, bumped with Distance 0.5 and the default Filter Width, and it emits the bumped normal mapped to
[0, 1]. The camera sees only the floor, lit by the emitter in one diffuse bounce (max bounces 1), so the emitter's bump
is evaluated at a secondary hit. Filter Glossy widens the ray differentials after the floor's diffuse bounce, so the
emitter's bump footprint depends on it. The world is black.

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
scene.cycles.max_bounces = 1
scene.cycles.seed = 0
scene.cycles.blur_glossy = 1.0
scene.render.resolution_x = 16
scene.render.resolution_y = 16
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.view_settings.view_transform = "Standard"

bpy.ops.mesh.primitive_plane_add(size=10.0, location=(0.0, 0.0, 0.0))
floor = bpy.context.object
floor_material = bpy.data.materials.new("Floor")
floor_material.use_nodes = True
floor_tree = floor_material.node_tree
for node in list(floor_tree.nodes):
    floor_tree.nodes.remove(node)
floor_output = floor_tree.nodes.new("ShaderNodeOutputMaterial")
diffuse = floor_tree.nodes.new("ShaderNodeBsdfDiffuse")
diffuse.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1.0)
floor_tree.links.new(diffuse.outputs["BSDF"], floor_output.inputs["Surface"])
floor.data.materials.append(floor_material)

bpy.ops.mesh.primitive_plane_add(size=4.0, location=(-4.0, 0.0, 0.8))
emitter = bpy.context.object
emitter.rotation_euler = (3.141592653589793, 0.0, 0.0)

material = bpy.data.materials.new("BumpEmitter")
material.use_nodes = True
tree = material.node_tree
for node in list(tree.nodes):
    tree.nodes.remove(node)
output = tree.nodes.new("ShaderNodeOutputMaterial")
emission = tree.nodes.new("ShaderNodeEmission")
emission.inputs["Strength"].default_value = 20.0
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
square = tree.nodes.new("ShaderNodeMath")
square.operation = "MULTIPLY"
tree.links.new(separate.outputs["X"], square.inputs[0])
tree.links.new(separate.outputs["X"], square.inputs[1])
tree.links.new(square.outputs["Value"], bump.inputs["Height"])
tree.links.new(bump.outputs["Normal"], multiply.inputs[0])
tree.links.new(multiply.outputs["Vector"], add.inputs[0])
tree.links.new(add.outputs["Vector"], emission.inputs["Color"])
tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
emitter.data.materials.append(material)

camera_data = bpy.data.cameras.new("Camera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 4.4
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
    scene.render.filepath = os.path.join(work, f"indirect_seed_{seed}.exr")
    bpy.ops.render.render(write_still=True)
    seed_means.append(frame_mean(scene.render.filepath))

with open(json_out, "w") as handle:
    json.dump({"samples": 64, "filter_glossy": 1.0, "seed_means": seed_means}, handle, indent=2)
print("DONE", flush=True)
