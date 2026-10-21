"""Blender 5.2.2's emitted radiance of an emissive absorbing or scattering mesh cube, for tests/volume_emission.rs.

The scene: a unit cube with a Principled Volume material (Density 2, Emission Color 1, Emission Strength 1, and a Color of
0, 0.5 or 1) in a black world, seen by an orthographic camera at z = 3 with ortho scale 2, Cycles CPU, Raw view transform,
32-bit EXR output. Color sets the albedo: 0 is absorption only, so the radiance is the closed form of the emission integral
through the cube,

    L = E * integral_0^1 exp(-sigma_t * s) ds = E * (1 - exp(-sigma_t)) / sigma_t,  sigma_t = Density = 2, E = 1,

with E unscaled by the Density (Blender's emission coefficient is the Emission Colour times its strength). Colour 0.5 and 1
scatter as well: a scattered path continues on along a terminal ray, which still collects the cube's emission, so their
means are larger than the closed form. Each case records the mean and standard error of the 4 x 4 pixels at the image
centre, per channel. The bounce budget is Blender's default for volumes (max_bounces 0, volume_bounces 0).

Regenerate from this directory with Blender 5.2.2, one thread, one run:

    Blender -b --factory-startup --threads 1 -P generate_volume_emission_oracle.py -- OUT.json
"""

import json
import math
import os
import sys
import tempfile

import bpy

argv = sys.argv[sys.argv.index("--") + 1 :]
out_path = os.path.abspath(argv[0])
work = tempfile.mkdtemp(prefix="volume_emission_")
SIZE = 24
SAMPLES = 256
BLOCK = range(10, 14)
DENSITY = 2.0
COLORS = {"color0": 0.0, "color05": 0.5, "color1": 1.0}


def build_scene(color):
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
    background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    background.inputs["Strength"].default_value = 1.0

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
    cube = bpy.context.active_object
    material = bpy.data.materials.new("emitter")
    material.use_nodes = True
    tree = material.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    volume = tree.nodes.new("ShaderNodeVolumePrincipled")
    volume.inputs["Color"].default_value = (color, color, color, 1.0)
    volume.inputs["Density"].default_value = DENSITY
    volume.inputs["Emission Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    volume.inputs["Emission Strength"].default_value = 1.0
    tree.links.new(volume.outputs["Volume"], output.inputs["Volume"])
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
for name, color in COLORS.items():
    build_scene(color)
    means, errors = render_block("emissive_cube_" + name)
    records["emissive_cube_" + name] = {"mean": means, "stderr": errors}
    print("ORACLE_CASE emissive_cube_" + name, means, errors)

with open(out_path, "w") as handle:
    json.dump(
        {
            "blender": bpy.app.version_string,
            "samples": SAMPLES,
            "density": DENSITY,
            "max_bounces": 0,
            "volume_bounces": 0,
            "cases": records,
        },
        handle,
        indent=1,
    )
print("ORACLE_OK volume_emission", json.dumps(records))
