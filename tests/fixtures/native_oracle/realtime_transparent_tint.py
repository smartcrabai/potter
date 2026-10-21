"""A Transparent BSDF pane's Color tints what it passes, as EEVEE 5.2.2 composites it.

EEVEE's Transparent node (`gpu_shader_material_transparent.glsl`) sets the closure's transmittance to its Color.
Which buffers keep it depends on the material's render method:

- BLENDED: `ForwardPipeline::use_colored_transparency` is set by `GPU_MATFLAG_TRANSPARENT_MAYBE_COLORED`, so the forward
  pass keeps the transmittance per channel and `eevee_forward_resolve.bsl.hh` multiplies the background by it channel
  by channel. A pixel behind the pane is the background times the Color.
- DITHERED (Blender's default): the deferred pass dithers the surface and stores one alpha, `1 - average(transmittance)`
  (`eevee_surf_deferred.bsl.hh`, `out_color.a`). A pixel behind the pane is the background times the Color's mean.

The backdrop is an emitter of strength 1 with a black base, so its radiance is exactly 1 and needs no light, and no
shadow tints it a second time. The pane is a Transparent BSDF with the Color `TINT` as its only closure, covering the
whole frame one unit above the backdrop.

Arguments after `--`: blend_out, image_out, json_out, method (BLENDED or DITHERED), and an optional render flag
(default 1). With the render flag 0 the generator writes only the .blend, so the test needs no render for the scene.
"""

import json
import sys

import bpy

ARGS = sys.argv[sys.argv.index("--") + 1:]
blend_out, image_out, json_out, method = ARGS[:4]
RENDER = ARGS[4] != "0" if len(ARGS) > 4 else True

WIDTH = 64
HEIGHT = 64
SAMPLES = 16
TINT = (0.2, 0.8, 0.4)
# Fractions of the frame: the same grid as the other native oracles.
PATCH = [0.25 + 0.5 * index / 8 for index in range(9)]

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.eevee.taa_render_samples = SAMPLES
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.render.resolution_x = WIDTH
scene.render.resolution_y = HEIGHT
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"

world = bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0

backdrop = bpy.data.materials.new("Backdrop")
backdrop.use_nodes = True
emitter = backdrop.node_tree.nodes["Principled BSDF"]
emitter.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
emitter.inputs["Specular IOR Level"].default_value = 0.0
emitter.inputs["Emission Color"].default_value = (1.0, 1.0, 1.0, 1.0)
emitter.inputs["Emission Strength"].default_value = 1.0
bpy.ops.mesh.primitive_plane_add(size=20.0, location=(0.0, 0.0, 0.0))
bpy.context.object.data.materials.append(backdrop)

pane = bpy.data.materials.new("Pane")
pane.use_nodes = True
if method == "BLENDED":
    pane.surface_render_method = "BLENDED"
elif method == "DITHERED":
    pane.surface_render_method = "DITHERED"
else:
    raise SystemExit("unknown method " + method)
tree = pane.node_tree
for node in list(tree.nodes):
    if node.type != "OUTPUT_MATERIAL":
        tree.nodes.remove(node)
output = next(node for node in tree.nodes if node.type == "OUTPUT_MATERIAL")
transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
transparent.inputs["Color"].default_value = (TINT[0], TINT[1], TINT[2], 1.0)
tree.links.new(transparent.outputs["BSDF"], output.inputs["Surface"])
bpy.ops.mesh.primitive_plane_add(size=20.0, location=(0.0, 0.0, 1.0))
bpy.context.object.data.materials.append(pane)

camera_data = bpy.data.cameras.new("Camera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 2.0
camera = bpy.data.objects.new("Camera", camera_data)
scene.collection.objects.link(camera)
camera.location = (-1.5, 0.0, 6.0)
scene.camera = camera

bpy.ops.wm.save_as_mainfile(filepath=blend_out)
if not RENDER:
    raise SystemExit(0)

scene.render.filepath = image_out
bpy.ops.render.render(write_still=True)
image = bpy.data.images.load(image_out, check_existing=False)
width, height = image.size
channels = image.channels
pixels = list(image.pixels)
sums = [0.0, 0.0, 0.0]
count = 0
for v in PATCH:
    for u in PATCH:
        x = min(width - 1, int(u * width))
        y = min(height - 1, int(v * height))
        index = (y * width + x) * channels
        for channel in range(3):
            sums[channel] += pixels[index + channel]
        count += 1

with open(json_out, "w", encoding="utf-8") as handle:
    json.dump(
        {"method": method, "patch_mean": [value / count for value in sums], "tint": list(TINT)},
        handle,
    )
