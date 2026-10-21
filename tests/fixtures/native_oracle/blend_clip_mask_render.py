# SPDX-License-Identifier: GPL-2.0-or-later
# Blender 5.2.2 oracle for the binary alpha clip that Potter's .blend export writes.
#
# A glTF MASK material imports as DITHERED with Principled Alpha = 1 - (x < cutoff) (Math LESS_THAN, then Math
# SUBTRACT(1, .); `io_scene_gltf2/blender/imp/pbrMetallicRoughness.py` 668-700). Here x is a gradient, so the
# surface is a hard cutout: each pixel of the plane is fully opaque or fully transparent, except the pixels that
# straddle the cutoff edge.
#
# The scene is saved to `blend_out` (Potter imports it), then rendered at eight Cycles seeds. For each seed the
# script writes the window statistics of the alpha channel: the fractions of pixels that are opaque (alpha at least
# 0.99), transparent (at most 0.01), or in between. The Potter test compares those seed means with its own.
#
# Arguments after `--`: blend_out, json_out.

import json
import os
import sys
import tempfile

import bpy

blend_out, json_out = sys.argv[sys.argv.index("--") + 1:][:2]

RESOLUTION = 64
WINDOW = range(16, 48)
CUTOFF = 0.5
SAMPLES = 16
SEEDS = range(8)

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = SAMPLES
# Potter samples each pixel with a one-pixel box, so the oracle uses Cycles' box filter of width 1.
scene.cycles.pixel_filter_type = "BOX"
scene.cycles.filter_width = 1.0
scene.cycles.device = "CPU"
scene.render.film_transparent = True
scene.render.resolution_x = scene.render.resolution_y = RESOLUTION
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.render.image_settings.color_mode = "RGBA"

for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)
bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, 0, 0))
plane = bpy.context.active_object

material = bpy.data.materials.new("clip_gradient")
material.use_nodes = True
material.surface_render_method = "DITHERED"
plane.data.materials.append(material)
tree = material.node_tree
for node in list(tree.nodes):
    tree.nodes.remove(node)
output = tree.nodes.new("ShaderNodeOutputMaterial")
principled = tree.nodes.new("ShaderNodeBsdfPrincipled")
principled.inputs["Base Color"].default_value = (0.9, 0.2, 0.1, 1.0)
tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])

coord = tree.nodes.new("ShaderNodeTexCoord")
gradient = tree.nodes.new("ShaderNodeTexGradient")
gradient.gradient_type = "LINEAR"
# The gradient runs along U (0 to 1 across the plane), so the cutoff falls on the plane's middle column.
tree.links.new(coord.outputs["UV"], gradient.inputs["Vector"])

# The importer's clip stage: Alpha = SUBTRACT(1, LESS_THAN(x, cutoff)).
compare = tree.nodes.new("ShaderNodeMath")
compare.operation = "LESS_THAN"
compare.inputs[1].default_value = CUTOFF
invert = tree.nodes.new("ShaderNodeMath")
invert.operation = "SUBTRACT"
invert.inputs[0].default_value = 1.0
tree.links.new(compare.inputs[0], gradient.outputs["Fac"])
tree.links.new(invert.inputs[1], compare.outputs["Value"])
tree.links.new(principled.inputs["Alpha"], invert.outputs["Value"])

camera_data = bpy.data.cameras.new("camera")
camera = bpy.data.objects.new("camera", camera_data)
bpy.context.collection.objects.link(camera)
camera.location = (0, 0, 3)
camera.rotation_euler = (0, 0, 0)
scene.camera = camera

light_data = bpy.data.lights.new("sun", "SUN")
light = bpy.data.objects.new("sun", light_data)
bpy.context.collection.objects.link(light)

bpy.ops.wm.save_as_mainfile(filepath=blend_out)

work = tempfile.mkdtemp()


def window_fractions(path):
    image = bpy.data.images.load(path, check_existing=False)
    width, height = image.size
    pixels = list(image.pixels)
    bpy.data.images.remove(image)
    alphas = [pixels[(y * width + x) * 4 + 3] for y in WINDOW for x in WINDOW]
    count = float(len(alphas))
    opaque = sum(1 for value in alphas if value >= 0.99) / count
    transparent = sum(1 for value in alphas if value <= 0.01) / count
    return {"opaque": opaque, "transparent": transparent, "fractional": 1.0 - opaque - transparent}


seeds = []
for seed in SEEDS:
    scene.cycles.seed = seed
    scene.render.filepath = os.path.join(work, f"seed_{seed}.exr")
    bpy.ops.render.render(write_still=True)
    seeds.append(window_fractions(scene.render.filepath))

with open(json_out, "w") as handle:
    json.dump({"samples": SAMPLES, "resolution": RESOLUTION, "window": [WINDOW[0], WINDOW[-1] + 1],
               "seeds": seeds}, handle, indent=2)
