"""Blender 5.2.2's transmitted radiance through a unit slab, for tests/volume_extinction.rs.

The scene: a unit cube (1 unit thick along the view axis) with a volume-only material, in front of a
white world of strength 1, seen by an orthographic camera at z = 3 with ortho scale 2, Cycles CPU, Raw
view transform, 32-bit EXR output. Each case is a shader graph whose output feeds the Material Output's
Volume socket:

- volume_absorption: Volume Absorption, Color (0.2, 0.5, 0.8), Density 1. Cycles' absorption closure
  weight is 1 - Color (intern/cycles/kernel/svm/closure.h, svm_node_closure_volume), so the
  transmittance is exp(-(1 - Color)).
- volume_absorption_default: Volume Absorption with Color unset (Blender's default 0.8), Density 1.
  The transmittance is exp(-0.2) in every channel.
- principled_absorption: Principled Volume, Color 0, Absorption Color (0.36, 0.25, 0.04), Density 1.
  The absorption weight is (1 - Color) * (1 - sqrt(Absorption Color)), so the transmittance is
  exp(-(0.4, 0.5, 0.8)).
- volume_scatter_white: Volume Scatter, Color (0.2, 0.5, 0.8), Density 1. A pure scatterer (albedo 1)
  in a uniform white world conserves energy, so the radiance is 1 in every channel.
- add_absorption: Add Shader of Volume Absorption (Color (0.2, 0.5, 0.8), Density 1) and Volume
  Absorption (Color 0.5, Density 0.5). Cycles adds the closures' extinctions, so the extinction is
  (0.8, 0.5, 0.2) + (0.25, 0.25, 0.25) = (1.05, 0.75, 0.45).
- mix_absorption: Mix Shader, Fac 0.25, of the same two closures. Cycles weights each closure by its
  mix factor, so the extinction is 0.75 * (0.8, 0.5, 0.2) + 0.25 * (0.25, 0.25, 0.25) =
  (0.6625, 0.4375, 0.2125).

The record of each case is the mean and standard error of the 4 x 4 pixels at the image centre.

Regenerate from this directory with Blender 5.2.2, one thread, one run:

    Blender -b --factory-startup --threads 1 -P generate_volume_extinction_oracle.py -- OUT.json
"""

import json
import math
import os
import sys
import tempfile

import bpy

argv = sys.argv[sys.argv.index("--") + 1 :]
out_path = os.path.abspath(argv[0])
work = tempfile.mkdtemp(prefix="volume_extinction_")
SIZE = 24
SAMPLES = 1024
BLOCK = range(10, 14)

ABSORPTION = "ShaderNodeVolumeAbsorption"
# A node's sockets are addressed by index for the combiners, whose two shader inputs share a name.
CASES = {
    "volume_absorption": {
        "nodes": [("volume", ABSORPTION, {"Color": (0.2, 0.5, 0.8, 1.0), "Density": 1.0})],
        "links": [],
        "output": ("volume", 0),
    },
    "volume_absorption_default": {
        "nodes": [("volume", ABSORPTION, {"Density": 1.0})],
        "links": [],
        "output": ("volume", 0),
    },
    "principled_absorption": {
        "nodes": [
            (
                "volume",
                "ShaderNodeVolumePrincipled",
                {"Color": (0.0, 0.0, 0.0, 1.0), "Absorption Color": (0.36, 0.25, 0.04, 1.0), "Density": 1.0},
            )
        ],
        "links": [],
        "output": ("volume", 0),
    },
    "volume_scatter_white": {
        "nodes": [("volume", "ShaderNodeVolumeScatter", {"Color": (0.2, 0.5, 0.8, 1.0), "Density": 1.0})],
        "links": [],
        "output": ("volume", 0),
    },
    "add_absorption": {
        "nodes": [
            ("first", ABSORPTION, {"Color": (0.2, 0.5, 0.8, 1.0), "Density": 1.0}),
            ("second", ABSORPTION, {"Color": (0.5, 0.5, 0.5, 1.0), "Density": 0.5}),
            ("add", "ShaderNodeAddShader", {}),
        ],
        # Add Shader inputs: 0 = Shader, 1 = Shader_001.
        "links": [("first", 0, "add", 0), ("second", 0, "add", 1)],
        "output": ("add", 0),
    },
    "mix_absorption": {
        "nodes": [
            ("first", ABSORPTION, {"Color": (0.2, 0.5, 0.8, 1.0), "Density": 1.0}),
            ("second", ABSORPTION, {"Color": (0.5, 0.5, 0.5, 1.0), "Density": 0.5}),
            ("mix", "ShaderNodeMixShader", {"Fac": 0.25}),
        ],
        # Mix Shader inputs: 0 = Fac, 1 = Shader, 2 = Shader_001.
        "links": [("first", 0, "mix", 1), ("second", 0, "mix", 2)],
        "output": ("mix", 0),
    },
}


def build_scene(case):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    render = scene.render
    render.engine = "CYCLES"
    render.resolution_x = SIZE
    render.resolution_y = SIZE
    render.resolution_percentage = 100
    render.film_transparent = False
    scene.cycles.device = "CPU"
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = 64
    scene.cycles.volume_bounces = 64
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
    background.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    background.inputs["Strength"].default_value = 1.0

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
    cube = bpy.context.active_object
    material = bpy.data.materials.new("slab")
    material.use_nodes = True
    tree = material.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    created = {}
    for name, node_type, inputs in case["nodes"]:
        node = tree.nodes.new(node_type)
        for socket, value in inputs.items():
            node.inputs[socket].default_value = value
        created[name] = node
    for source, source_index, target, target_index in case["links"]:
        tree.links.new(created[source].outputs[source_index], created[target].inputs[target_index])
    feeding, feeding_index = case["output"]
    tree.links.new(created[feeding].outputs[feeding_index], output.inputs["Volume"])
    cube.data.materials.append(material)

    camera_data = bpy.data.cameras.new("camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 2.0
    camera = bpy.data.objects.new("camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0.0, 0.0, 3.0)
    camera.rotation_euler = (0.0, 0.0, 0.0)
    scene.camera = camera


def render_block(name):
    path = os.path.join(work, name + ".exr")
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    loaded = bpy.data.images.load(path, check_existing=False)
    pixels = list(loaded.pixels)
    bpy.data.images.remove(loaded)
    channels = [[], [], []]
    for y in BLOCK:
        for x in BLOCK:
            offset = (y * SIZE + x) * 4
            for channel in range(3):
                channels[channel].append(pixels[offset + channel])
    means, errors = [], []
    for values in channels:
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        means.append(round(mean, 6))
        errors.append(round(math.sqrt(variance / len(values)), 6))
    return means, errors


records = {}
for name, case in CASES.items():
    build_scene(case)
    means, errors = render_block(name)
    records[name] = {"mean": means, "stderr": errors}
    print("ORACLE_CASE", name, means, errors)

with open(out_path, "w") as handle:
    json.dump({"blender": bpy.app.version_string, "samples": SAMPLES, "cases": records}, handle, indent=1)
print("ORACLE_OK volume_extinction", json.dumps(records))
