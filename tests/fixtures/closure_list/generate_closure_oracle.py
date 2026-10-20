"""Blender 5.2.2 Cycles references for tests/closure_list_oracle.rs (CPU, fixed seeds).

Three scenes, each a 2 x 2 unit plane facing an orthographic camera at z = 5 (ortho scale 1.8, so the plane fills the
frame). A uniform world (0.25, 0.4, 0.6, strength 1) and a Sun (energy 3, tilted 40 degrees from the plane normal toward
+X) light it. Standard view transform, 32-bit linear EXR, no denoising, no adaptive sampling, max_bounces 8.

- mix_principled: Mix Shader, Factor 0.3 (the second arm weighs 0.3), of two Principled BSDFs. The first arm has
  Roughness 0.1 and the geometric normal. The second has Roughness 0.6 and a constant normal (0.6, 0.0, 0.8) from a
  Combine XYZ node: a unit vector about 37 degrees from the plane normal toward +X. The two arms differ in roughness and
  in normal, so the mix must keep each closure's own normal.
- add_diffuse_glossy: Add Shader (weights 1 and 1) of a Diffuse BSDF (Color (0.8, 0.3, 0.1)) and a Glossy BSDF
  (Color 0.5, Roughness 0.3).
- mix_glass_diffuse: Mix Shader, Factor 0.5, of a Glass BSDF (Color 1, Roughness 0, IOR 1.45) and the same Diffuse. The
  glass share transmits straight through the plane to the world.

Principled, Glossy and Glass keep Blender's default MULTI_GGX distribution.

Each scene writes <name>.blend (imported by the Rust test), <name>.exr (the Cycles render at seed 0), and <name>.json.
The JSON holds the 16 x 16 region at (8, 8) as per-pixel RGB means (top-left origin, row-major), the region mean, and
the mean absolute error between two Cycles renders at different seeds, which is the reference's own noise.

Regenerate from this directory:

    Blender -b --factory-startup -P generate_closure_oracle.py -- .
"""

import json
import math
import os
import sys
import tempfile

import bpy

out_dir = os.path.abspath(sys.argv[sys.argv.index("--") + 1])

WIDTH = 32
HEIGHT = 32
SAMPLES = 16384
MAX_BOUNCES = 8
REFERENCE_SEED = 0
NOISE_SEED = 1
# x, y, width, height of the region, from the top-left corner. The same window as the Rust test's patch.
REGION = (8, 8, 16, 16)
WORLD = (0.25, 0.4, 0.6)
SUN_ENERGY = 3.0
SUN_TILT = math.radians(40.0)
DIFFUSE = (0.8, 0.3, 0.1)


def setup_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.filepaths.save_version = 0
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = SAMPLES
    scene.cycles.seed = REFERENCE_SEED
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = MAX_BOUNCES
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

    sun = bpy.data.lights.new("Sun", "SUN")
    sun.energy = SUN_ENERGY
    sun_object = bpy.data.objects.new("Sun", sun)
    scene.collection.objects.link(sun_object)
    sun_object.rotation_euler = (0.0, SUN_TILT, 0.0)

    camera_data = bpy.data.cameras.new("Camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 1.8
    camera = bpy.data.objects.new("Camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0.0, 0.0, 5.0)
    scene.camera = camera

    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, 0.0))
    return bpy.context.object


def principled(tree, color, roughness):
    node = tree.nodes.new("ShaderNodeBsdfPrincipled")
    node.inputs["Base Color"].default_value = (*color, 1.0)
    node.inputs["Roughness"].default_value = roughness
    return node


def graph_mix_principled(tree):
    flat_arm = principled(tree, (0.8, 0.2, 0.2), 0.1)
    tilted_arm = principled(tree, (0.2, 0.6, 0.3), 0.6)
    # A constant unit vector, about 37 degrees from the geometric normal toward +X.
    normal = tree.nodes.new("ShaderNodeCombineXYZ")
    normal.inputs["X"].default_value = 0.6
    normal.inputs["Y"].default_value = 0.0
    normal.inputs["Z"].default_value = 0.8
    tree.links.new(normal.outputs["Vector"], tilted_arm.inputs["Normal"])
    mix = tree.nodes.new("ShaderNodeMixShader")
    mix.inputs["Factor"].default_value = 0.3
    tree.links.new(flat_arm.outputs["BSDF"], mix.inputs[1])
    tree.links.new(tilted_arm.outputs["BSDF"], mix.inputs[2])
    return mix


def graph_add_diffuse_glossy(tree):
    diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = (*DIFFUSE, 1.0)
    glossy = tree.nodes.new("ShaderNodeBsdfGlossy")
    glossy.inputs["Color"].default_value = (0.5, 0.5, 0.5, 1.0)
    glossy.inputs["Roughness"].default_value = 0.3
    add = tree.nodes.new("ShaderNodeAddShader")
    tree.links.new(diffuse.outputs["BSDF"], add.inputs[0])
    tree.links.new(glossy.outputs["BSDF"], add.inputs[1])
    return add


def graph_mix_glass_diffuse(tree):
    glass = tree.nodes.new("ShaderNodeBsdfGlass")
    glass.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    glass.inputs["Roughness"].default_value = 0.0
    glass.inputs["IOR"].default_value = 1.45
    diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = (*DIFFUSE, 1.0)
    mix = tree.nodes.new("ShaderNodeMixShader")
    mix.inputs["Factor"].default_value = 0.5
    tree.links.new(glass.outputs["BSDF"], mix.inputs[1])
    tree.links.new(diffuse.outputs["BSDF"], mix.inputs[2])
    return mix


def assign_material(plane, name, build):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.remove(tree.nodes["Principled BSDF"])
    output = next(node for node in tree.nodes if node.type == "OUTPUT_MATERIAL")
    root = build(tree)
    tree.links.new(root.outputs[0], output.inputs["Surface"])
    plane.data.materials.append(material)


def render_to(path, seed):
    scene = bpy.context.scene
    scene.cycles.seed = seed
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(path, check_existing=False)
    width, height = image.size
    channels = image.channels
    pixels = list(image.pixels)
    bpy.data.images.remove(image)
    return width, height, channels, pixels


def region_rows(width, height, channels, pixels):
    x0, y0, w, h = REGION
    rows = []
    for top in range(y0, y0 + h):
        # Blender stores the bottom image row first.
        bottom = height - 1 - top
        for x in range(x0, x0 + w):
            start = (bottom * width + x) * channels
            rows.append([pixels[start], pixels[start + 1], pixels[start + 2]])
    return rows


def mean_abs_error(actual, wanted):
    total = sum(abs(a - w) for pa, pw in zip(actual, wanted) for a, w in zip(pa, pw))
    return total / (len(actual) * 3)


SCENES = {
    "mix_principled": graph_mix_principled,
    "add_diffuse_glossy": graph_add_diffuse_glossy,
    "mix_glass_diffuse": graph_mix_glass_diffuse,
}

with tempfile.TemporaryDirectory(prefix="closure_list_") as work:
    for name, build in SCENES.items():
        plane = setup_scene()
        assign_material(plane, name, build)
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, name + ".blend"))
        reference = region_rows(*render_to(os.path.join(out_dir, name + ".exr"), REFERENCE_SEED))
        other = region_rows(*render_to(os.path.join(work, name + "_noise.exr"), NOISE_SEED))
        record = {
            "blender": bpy.app.version_string,
            "scene": name,
            "resolution": [WIDTH, HEIGHT],
            "region": {"x": REGION[0], "y": REGION[1], "width": REGION[2], "height": REGION[3]},
            "origin": "top-left: pixel row 0 is the top image row; pixels are row-major over the region",
            "cycles": {
                "device": "CPU",
                "samples": SAMPLES,
                "seed": REFERENCE_SEED,
                "max_bounces": MAX_BOUNCES,
                "adaptive_sampling": False,
                "denoising": False,
                "view_transform": "Standard",
            },
            "noise": {"seed": NOISE_SEED, "mean_abs_error_between_seeds": mean_abs_error(reference, other)},
            "mean": [sum(pixel[c] for pixel in reference) / len(reference) for c in range(3)],
            "pixels": reference,
        }
        with open(os.path.join(out_dir, name + ".json"), "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=1)
        print("CLOSURE_ORACLE", name, record["mean"], record["noise"])
