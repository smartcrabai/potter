"""Bump at the end of a glossy reflection, with Filter Glossy, rendered by Blender 5.2.2 Cycles.

A perspective camera looks down at a metallic floor of roughness 0.3, so the floor is a glossy reflector and its
reflection is the only thing it shows. The reflected ray reaches a bumped emitter that stands on the mirror image of the
camera's view. The emitter's material turns light sampling off (`emission_sampling` NONE), so the emitter is reached only
through the floor's BSDF samples, which are glossy, non-specular bounces: Filter Glossy widens the ray differential at
that bounce (`bsdf_widen_dD`, `shade_surface.h`), and the emitter's bump footprint depends on it. The emitter's height is
a sawtooth, `FRACT` of the Generated X ramp with 125 teeth across the emitter, so the widened footprint, which spans many
teeth, averages the bump into a different mean normal, while the narrow default footprint keeps the sharp teeth. Blender's
Filter Glossy 1.0 and 0.0 differ by 15.7 and 40 times the test tolerance in red and green (seed spread about 1e-3). The
emitter emits the bumped normal mapped to [0, 1]. The world is black.

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

bpy.ops.mesh.primitive_plane_add(size=20.0, location=(0.0, 0.0, 0.0))
floor = bpy.context.object
floor_material = bpy.data.materials.new("GlossyFloor")
floor_material.use_nodes = True
floor_principled = floor_material.node_tree.nodes["Principled BSDF"]
floor_principled.inputs["Metallic"].default_value = 1.0
floor_principled.inputs["Roughness"].default_value = 0.3
floor_principled.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
floor.data.materials.append(floor_material)

bpy.ops.mesh.primitive_plane_add(size=6.0, location=(0.0, 4.0, 2.0))
emitter = bpy.context.object
emitter.rotation_euler = (1.5707963267948966, 0.0, 0.0)

material = bpy.data.materials.new("BumpEmitter")
material.use_nodes = True
material.cycles.emission_sampling = "NONE"
tree = material.node_tree
for node in list(tree.nodes):
    tree.nodes.remove(node)
output = tree.nodes.new("ShaderNodeOutputMaterial")
emission = tree.nodes.new("ShaderNodeEmission")
emission.inputs["Strength"].default_value = 1.0
texcoord = tree.nodes.new("ShaderNodeTexCoord")
separate = tree.nodes.new("ShaderNodeSeparateXYZ")
teeth = tree.nodes.new("ShaderNodeMath")
teeth.operation = "MULTIPLY"
teeth.inputs[1].default_value = 125.0
sawtooth = tree.nodes.new("ShaderNodeMath")
sawtooth.operation = "FRACT"
bump = tree.nodes.new("ShaderNodeBump")
bump.inputs["Distance"].default_value = 0.3
multiply = tree.nodes.new("ShaderNodeVectorMath")
multiply.operation = "MULTIPLY"
multiply.inputs[1].default_value = (0.5, 0.5, 0.5)
add = tree.nodes.new("ShaderNodeVectorMath")
add.operation = "ADD"
add.inputs[1].default_value = (0.5, 0.5, 0.5)

tree.links.new(texcoord.outputs["Generated"], separate.inputs["Vector"])
tree.links.new(separate.outputs["X"], teeth.inputs[0])
tree.links.new(teeth.outputs["Value"], sawtooth.inputs[0])
tree.links.new(sawtooth.outputs["Value"], bump.inputs["Height"])
tree.links.new(bump.outputs["Normal"], multiply.inputs[0])
tree.links.new(multiply.outputs["Vector"], add.inputs[0])
tree.links.new(add.outputs["Vector"], emission.inputs["Color"])
tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
emitter.data.materials.append(material)

camera_data = bpy.data.cameras.new("Camera")
camera = bpy.data.objects.new("Camera", camera_data)
scene.collection.objects.link(camera)
camera.location = Vector((0.0, -6.0, 3.0))
camera.rotation_euler = (Vector((0.0, 6.0, -3.0)).to_track_quat("-Z", "Y")).to_euler()
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
result = {"samples": 64}
for key, filter_glossy in (("seed_means", 1.0), ("seed_means_filter_glossy_0", 0.0)):
    scene.cycles.blur_glossy = filter_glossy
    seed_means = []
    for seed in range(8):
        scene.cycles.seed = seed
        scene.render.filepath = os.path.join(work, f"reflection_{key}_{seed}.exr")
        bpy.ops.render.render(write_still=True)
        seed_means.append(frame_mean(scene.render.filepath))
    result[key] = seed_means

with open(json_out, "w") as handle:
    json.dump(result, handle, indent=2)
print("DONE", flush=True)
