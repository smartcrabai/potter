"""Blender 5.2.2's single-scatter radiance of a pure scatterer in a white world, for tests/volume_extinction.rs.

The scene: a unit cube with a Volume Scatter material (Color 1, albedo 1, Density 1) in front of a white world of
strength 1, seen by an orthographic camera at z = 3 with ortho scale 2, Cycles CPU, Raw view transform, 32-bit EXR
output. The bounce budget is Blender's default for volumes, `max_bounces = 0` and `volume_bounces = 0`, so each path
takes one scatter vertex with direct light and the scattered ray is not followed past it (Cycles'
`max_volume_bounce = volume_bounces + 1`, intern/cycles/scene/integrator.cpp:227). The environment is the only light, so
the direct light of the vertex is the environment's single-scatter estimate, MIS weighted against the phase sample
that Cycles traces on to the environment.

The record is the mean and standard error of the 4 x 4 pixels at the image centre, per channel.

Regenerate from this directory with Blender 5.2.2, one thread, one run:

    Blender -b --factory-startup --threads 1 -P generate_volume_single_scatter_oracle.py -- OUT.json
"""

import json
import math
import os
import sys
import tempfile

import bpy

argv = sys.argv[sys.argv.index("--") + 1 :]
out_path = os.path.abspath(argv[0])
work = tempfile.mkdtemp(prefix="volume_single_scatter_")
SIZE = 24
SAMPLES = 4096
BLOCK = range(10, 14)


def build_scene():
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
    scene.cycles.max_bounces = 0
    scene.cycles.volume_bounces = 0
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
    material = bpy.data.materials.new("scatterer")
    material.use_nodes = True
    tree = material.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    scatter = tree.nodes.new("ShaderNodeVolumeScatter")
    scatter.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    scatter.inputs["Density"].default_value = 1.0
    tree.links.new(scatter.outputs[0], output.inputs["Volume"])
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


build_scene()
means, errors = render_block("volume_scatter_white_max0")
record = {"mean": means, "stderr": errors}
print("ORACLE_CASE volume_scatter_white_max0", means, errors)

with open(out_path, "w") as handle:
    json.dump(
        {
            "blender": bpy.app.version_string,
            "samples": SAMPLES,
            "max_bounces": 0,
            "volume_bounces": 0,
            "cases": {"volume_scatter_white_max0": record},
        },
        handle,
        indent=1,
    )
print("ORACLE_OK volume_single_scatter", json.dumps(record))
