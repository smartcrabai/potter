# SPDX-License-Identifier: GPL-2.0-or-later
#
# Blender 5.2.2 oracle for the binary alpha clip that Potter's .blend export writes.
#
# A glTF MASK material imports as DITHERED with Principled Alpha = 1 - (x < cutoff) (Math LESS_THAN, then Math
# SUBTRACT(1, .); `io_scene_gltf2/blender/imp/pbrMetallicRoughness.py` 668-700). Here x is a gradient, so the
# surface is a hard cutout: each pixel is fully opaque or fully transparent, except the pixels that straddle the
# cutoff edge. The window statistics below are recorded in `blend_clip_mask_render.json`; Potter's render of the same
# graph is not compared against them yet (see the report for the render gate).
#
# Run in background mode, from the repository root:
#   /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
#       --python tests/fixtures/native_oracle/blend_clip_mask_render.py -- <output_dir>
#
# The camera looks straight down at the plane, so the window (pixels 16..48) sits inside the plane's silhouette.

import os
import sys

import bpy

OUTPUT_DIR = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else "."
RESOLUTION = 64
WINDOW = range(16, 48)
CUTOFF = 0.5

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = 8
scene.cycles.device = "CPU"
scene.render.film_transparent = True
scene.render.resolution_x = scene.render.resolution_y = RESOLUTION
scene.render.resolution_percentage = 100

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
tree.links.new(coord.outputs["Generated"], gradient.inputs["Vector"])

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

image_path = os.path.join(OUTPUT_DIR, "clip_mask_cutout.png")
scene.render.filepath = image_path
bpy.ops.render.render(write_still=True)

image = bpy.data.images.load(image_path)
pixels = image.pixels[:]


def alpha(x, y):
    return pixels[(y * RESOLUTION + x) * 4 + 3]


window = [alpha(x, y) for y in WINDOW for x in WINDOW]
fractional = [value for value in window if 0.01 < value < 0.99]
opaque = sum(1 for value in window if value >= 0.99)
transparent = sum(1 for value in window if value <= 0.01)
print("ORACLE window_pixels", len(window))
print("ORACLE window_opaque", opaque)
print("ORACLE window_transparent", transparent)
print("ORACLE window_fractional", len(fractional))
print("ORACLE window_fraction_fractional", round(len(fractional) / len(window), 4))
