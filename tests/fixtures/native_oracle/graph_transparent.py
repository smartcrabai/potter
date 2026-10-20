"""A DITHERED material whose graph mixes a Transparent BSDF 50% over a diffuse surface, in front of a coloured
emitter, rendered by Blender (v5.2.2) with the engine given on the command line (CYCLES or BLENDER_EEVEE).

The surface plane's material is DITHERED (the default render method) and its only transparency is the graph:
`Mix Shader(Fac 0.5, Transparent, Principled diffuse)`. Cycles ignores the render method and honours the graph,
so a ray passes the plane with probability 0.5 and otherwise shades the diffuse. EEVEE dithers the same graph
(its Transparent node sets `GPU_MATFLAG_TRANSPARENT`, `source/blender/nodes/shader/nodes/node_shader_bsdf_transparent.cc`), so its expected colour is
the same. The emitter plane sits behind the surface plane, and the world is a uniform grey that lights the diffuse.

The expected mean is `0.5 * emitter + 0.5 * albedo * world`. The patch grid must match `PATCH` in
`tests/native_graph_transparent.rs`.

Arguments after `--`: blend_out, image_out, json_out, engine (CYCLES or BLENDER_EEVEE).
"""

import json
import sys

import bpy

blend_out, image_out, json_out, engine = sys.argv[sys.argv.index("--") + 1:][:4]

WIDTH = 64
HEIGHT = 64
SAMPLES = 64
EMITTER = (0.1, 0.6, 0.2)
ALBEDO = (0.8, 0.3, 0.1)
WORLD = (0.5, 0.5, 0.5)
FAC = 0.5
# Fractions of the frame: the same patch grid as `tests/native_point_spot_light.rs`.
PATCH = [0.25 + 0.5 * index / 8 for index in range(9)]

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = engine
if engine == "CYCLES":
    scene.cycles.device = "CPU"
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = 4
else:
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
background = world.node_tree.nodes["Background"]
background.inputs["Color"].default_value = (*WORLD, 1.0)
background.inputs["Strength"].default_value = 1.0

# The surface: Mix(Fac, Transparent, Principled). Its render method stays DITHERED.
surface = bpy.data.materials.new("Graph Transparent")
surface.use_nodes = True
tree = surface.node_tree
output = next(node for node in tree.nodes if node.type == "OUTPUT_MATERIAL")
principled = tree.nodes["Principled BSDF"]
principled.inputs["Base Color"].default_value = (*ALBEDO, 1.0)
principled.inputs["Metallic"].default_value = 0.0
principled.inputs["Roughness"].default_value = 1.0
principled.inputs["Specular IOR Level"].default_value = 0.0
mix = tree.nodes.new("ShaderNodeMixShader")
mix.inputs["Fac"].default_value = FAC
transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
tree.links.new(transparent.outputs["BSDF"], mix.inputs[1])
tree.links.new(principled.outputs["BSDF"], mix.inputs[2])
tree.links.new(mix.outputs["Shader"], output.inputs["Surface"])

# The emitter: a coloured plane behind the surface.
emitter = bpy.data.materials.new("Emitter")
emitter.use_nodes = True
emitter_tree = emitter.node_tree
emitter_output = next(node for node in emitter_tree.nodes if node.type == "OUTPUT_MATERIAL")
emission = emitter_tree.nodes.new("ShaderNodeEmission")
emission.inputs["Color"].default_value = (*EMITTER, 1.0)
emission.inputs["Strength"].default_value = 1.0
emitter_tree.links.new(emission.outputs["Emission"], emitter_output.inputs["Surface"])

# Planes face the camera at -Y (the camera at y = -6 looks along +Y).
bpy.ops.mesh.primitive_plane_add(size=10.0, rotation=(1.5707963, 0.0, 0.0), location=(0.0, 0.0, 0.0))
bpy.context.object.data.materials.append(surface)
bpy.ops.mesh.primitive_plane_add(size=10.0, rotation=(1.5707963, 0.0, 0.0), location=(0.0, 2.0, 0.0))
bpy.context.object.data.materials.append(emitter)

camera_data = bpy.data.cameras.new("Camera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 4.0
camera = bpy.data.objects.new("Camera", camera_data)
scene.collection.objects.link(camera)
camera.location = (0.0, -6.0, 0.0)
camera.rotation_euler = (1.5707963, 0.0, 0.0)
scene.camera = camera

bpy.ops.wm.save_as_mainfile(filepath=blend_out)

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
    json.dump({"patch_mean": [value / count for value in sums]}, handle)
