"""Graph alpha cases rendered by Blender (v5.2.2), with the engine given on the command line.

Arguments after `--`: blend_out, image_out, json_out, engine (CYCLES or BLENDER_EEVEE), case.

`viewport_alpha`: a DITHERED material whose graph is a Diffuse BSDF alone, with the viewport colour (`diffuse_color`)
at alpha 0.5, in front of a white world. The graph has no transparency, so the plane is opaque and the patch shows the
diffuse lit by the world: albedo 0.8 under a unit uniform environment. Blender ignores the viewport alpha for a graph
material; Potter must too (`shader/evaluator.rs` `closure_params`).

`tinted_transparent`: a Transparent BSDF tinted (0.9, 0.4, 0.2) on a plane at z = 1, above a white Diffuse floor at
z = 0 lit by a sun of strength 1 pointing down. A camera ray passes the tinted plane once, and the floor's shadow ray
passes it once more, so the patch is `tint^2 * albedo / pi` per channel: Cycles weights each transparent crossing by
the closure colour (`intern/cycles/kernel/closure/bsdf_transparent.h`), for camera and shadow rays alike.

The patch grid must match `PATCH` in `tests/native_graph_transparent.rs`.
"""

import json
import sys

import bpy

blend_out, image_out, json_out, engine, case = sys.argv[sys.argv.index("--") + 1:][:5]

WIDTH = 64
HEIGHT = 64
SAMPLES = 64
ALBEDO = 0.8
TINT = (0.9, 0.4, 0.2)
# Fractions of the frame: the same patch grid as `tests/native_graph_transparent.rs`.
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


def graph_material(name, shader_builder):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.surface_render_method = "DITHERED"
    tree = material.node_tree
    output = next(node for node in tree.nodes if node.type == "OUTPUT_MATERIAL")
    for node in list(tree.nodes):
        if node.type != "OUTPUT_MATERIAL":
            tree.nodes.remove(node)
    shader = shader_builder(tree)
    tree.links.new(shader.outputs[0], output.inputs["Surface"])
    return material


def diffuse(tree):
    node = tree.nodes.new("ShaderNodeBsdfDiffuse")
    node.inputs["Color"].default_value = (ALBEDO, ALBEDO, ALBEDO, 1.0)
    node.inputs["Roughness"].default_value = 0.0
    return node


def tinted_transparent(tree):
    node = tree.nodes.new("ShaderNodeBsdfTransparent")
    node.inputs["Color"].default_value = (*TINT, 1.0)
    return node


camera_data = bpy.data.cameras.new("Camera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 4.0
camera = bpy.data.objects.new("Camera", camera_data)
scene.collection.objects.link(camera)
scene.camera = camera

if case == "viewport_alpha":
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
    material = graph_material("Viewport Alpha", diffuse)
    material.diffuse_color = (ALBEDO, ALBEDO, ALBEDO, 0.5)
    bpy.ops.mesh.primitive_plane_add(size=10.0, rotation=(1.5707963, 0.0, 0.0), location=(0.0, 0.0, 0.0))
    bpy.context.object.data.materials.append(material)
    camera.location = (0.0, -6.0, 0.0)
    camera.rotation_euler = (1.5707963, 0.0, 0.0)
elif case == "tinted_transparent":
    floor = graph_material("Floor", diffuse)
    bpy.ops.mesh.primitive_plane_add(size=10.0, location=(0.0, 0.0, 0.0))
    bpy.context.object.data.materials.append(floor)
    glass = graph_material("Tinted", tinted_transparent)
    bpy.ops.mesh.primitive_plane_add(size=10.0, location=(0.0, 0.0, 1.0))
    bpy.context.object.data.materials.append(glass)
    sun_data = bpy.data.lights.new("Sun", "SUN")
    sun_data.energy = 1.0
    sun_data.angle = 0.0
    sun = bpy.data.objects.new("Sun", sun_data)
    scene.collection.objects.link(sun)
    sun.rotation_euler = (0.0, 0.0, 0.0)
    camera.location = (0.0, 0.0, 5.0)
else:
    raise SystemExit("unknown case " + case)

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
