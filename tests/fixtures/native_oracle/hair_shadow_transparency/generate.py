"""Build a pinned Cycles hair-curve shadow-transparency scene and capture."""
import json
import os
import sys
from array import array

import bpy

SIZE = 64
SAMPLES = 64
SEED = 29
ORTHO = 8.0
PATCH_RADIUS = 1
SHADOW_PROBE = (SIZE // 2, SIZE // 2, PATCH_RADIUS)
LIT_PROBE = (SIZE // 2, 20, PATCH_RADIUS)
CURVE_POINTS = [(-3.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
                (1.0, 0.0, 0.0), (3.0, 0.0, 0.0)]
CURVE_RADIUS = 0.35
TRANSPARENT_COLOR = (0.2, 0.7, 0.4, 1.0)
MIX_FACTOR = 0.35
EXPECTED_BAKED_SHADOW_TRANSPARENCY = (1.0 - MIX_FACTOR) * sum(TRANSPARENT_COLOR[:3]) / 3.0
RECEIVER_COLOR = (0.8, 0.55, 0.35, 1.0)
SUN_ENERGY = 2.0

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"expected Blender 5.2.2, got {bpy.app.version_string}")

args = sys.argv[sys.argv.index("--") + 1:]
if len(args) != 1:
    raise RuntimeError("usage: generate.py -- <output-directory>")
out_dir = os.path.abspath(args[0])
os.makedirs(out_dir, exist_ok=True)
bpy.context.preferences.filepaths.save_version = 0

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
for block in bpy.data.materials:
    bpy.data.materials.remove(block)

scene = bpy.context.scene
scene.name = "Hair Shadow Transparency"
scene.render.engine = "CYCLES"
scene.render.resolution_x = SIZE
scene.render.resolution_y = SIZE
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_mode = "RGBA"
scene.render.image_settings.color_depth = "32"
scene.render.image_settings.exr_codec = "ZIP"
scene.render.threads_mode = "FIXED"
scene.render.threads = 1
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0
scene.cycles.device = "CPU"
scene.cycles.samples = SAMPLES
scene.cycles.seed = SEED
scene.cycles.pixel_filter_type = "BOX"
scene.cycles.filter_width = 1.0
scene.cycles.use_denoising = False
scene.cycles.max_bounces = 1
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 0
scene.cycles.transmission_bounces = 0
scene.cycles.volume_bounces = 0
scene.cycles.transparent_max_bounces = 8

world = bpy.data.worlds.new("Black World")
world.use_nodes = True
world.node_tree.nodes.clear()
background = world.node_tree.nodes.new("ShaderNodeBackground")
background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
background.inputs["Strength"].default_value = 0.0
world_output = world.node_tree.nodes.new("ShaderNodeOutputWorld")
world.node_tree.links.new(background.outputs["Background"], world_output.inputs["Surface"])
scene.world = world

camera_data = bpy.data.cameras.new("Camera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = ORTHO
camera = bpy.data.objects.new("Camera", camera_data)
scene.collection.objects.link(camera)
camera.location = (0.0, 0.0, 10.0)
camera.rotation_euler = (0.0, 0.0, 0.0)
scene.camera = camera

receiver_mesh = bpy.data.meshes.new("Diffuse Receiver Mesh")
receiver_mesh.from_pydata(
    [(-8.0, -8.0, 0.0), (8.0, -8.0, 0.0), (8.0, 8.0, 0.0), (-8.0, 8.0, 0.0)],
    [], [(0, 1, 2, 3)],
)
receiver = bpy.data.objects.new("Diffuse Receiver", receiver_mesh)
scene.collection.objects.link(receiver)
receiver_material = bpy.data.materials.new("Diffuse Receiver Material")
receiver_material.use_nodes = True
receiver_material.node_tree.nodes.clear()
receiver_diffuse = receiver_material.node_tree.nodes.new("ShaderNodeBsdfDiffuse")
receiver_diffuse.inputs["Color"].default_value = RECEIVER_COLOR
receiver_output = receiver_material.node_tree.nodes.new("ShaderNodeOutputMaterial")
receiver_material.node_tree.links.new(receiver_diffuse.outputs["BSDF"], receiver_output.inputs["Surface"])
receiver.data.materials.append(receiver_material)

sun_data = bpy.data.lights.new("Zero Angle Sun", type="SUN")
sun_data.energy = SUN_ENERGY
sun_data.angle = 0.0
sun = bpy.data.objects.new("Zero Angle Sun", sun_data)
scene.collection.objects.link(sun)
sun.rotation_euler = (0.0, 0.0, 0.0)

curve_data = bpy.data.hair_curves.new("Catmull Rom Hair Curves")
curve_data.add_curves([len(CURVE_POINTS)])
curve_data.set_types(type="CATMULL_ROM")
for point, position in zip(curve_data.points, CURVE_POINTS):
    point.position = position
    point.radius = CURVE_RADIUS
hair = bpy.data.objects.new("Camera Hidden Hair", curve_data)
scene.collection.objects.link(hair)
hair.location = (0.0, 0.0, 0.8)
hair.visible_camera = False
hair.visible_diffuse = False
hair.visible_glossy = False
hair.visible_transmission = False
hair.visible_volume_scatter = False
hair.visible_shadow = True
hair_material = bpy.data.materials.new("Tinted Transparent Diffuse Mix")
hair_material.use_nodes = True
hair_material.node_tree.nodes.clear()
hair_nodes = hair_material.node_tree.nodes
hair_links = hair_material.node_tree.links
transparent = hair_nodes.new("ShaderNodeBsdfTransparent")
transparent.inputs["Color"].default_value = TRANSPARENT_COLOR
opaque_diffuse = hair_nodes.new("ShaderNodeBsdfDiffuse")
mix = hair_nodes.new("ShaderNodeMixShader")
mix.inputs[0].default_value = MIX_FACTOR
hair_output = hair_nodes.new("ShaderNodeOutputMaterial")
hair_links.new(transparent.outputs["BSDF"], mix.inputs[1])
hair_links.new(opaque_diffuse.outputs["BSDF"], mix.inputs[2])
hair_links.new(mix.outputs["Shader"], hair_output.inputs["Surface"])
hair.data.materials.append(hair_material)

blend_path = os.path.join(out_dir, "hair_shadow_transparency.blend")
exr_path = os.path.join(out_dir, "hair_shadow_transparency.exr")
scene.render.filepath = os.path.join(out_dir, "hair_shadow_transparency")
bpy.ops.wm.save_as_mainfile(filepath=blend_path)
bpy.ops.render.render(write_still=True)

image = bpy.data.images.load(exr_path, check_existing=False)
width, height = image.size[:]
pixels = array("f", [0.0]) * (width * height * 4)
image.pixels.foreach_get(pixels)

def patch_mean(probe):
    x, y, radius = probe
    total = [0.0, 0.0, 0.0, 0.0]
    count = 0
    for py in range(y - radius, y + radius + 1):
        for px in range(x - radius, x + radius + 1):
            index = ((height - 1 - py) * width + px) * 4
            for channel in range(4):
                total[channel] += float(pixels[index + channel])
            count += 1
    return [value / count for value in total]

cases = [
    {"name": "shadow_patch", "probe": list(SHADOW_PROBE), "linear_rgba": patch_mean(SHADOW_PROBE)},
    {"name": "lit_patch", "probe": list(LIT_PROBE), "linear_rgba": patch_mean(LIT_PROBE)},
]
build_hash = bpy.app.build_hash
if isinstance(build_hash, bytes):
    build_hash = build_hash.decode("ascii")
capture = {
    "blender_version": bpy.app.version_string,
    "blender_build_hash": build_hash,
    "engine": "CYCLES",
    "resolution": [width, height],
    "samples": SAMPLES,
    "seed": SEED,
    "pixel_filter": "BOX",
    "filter_width": 1.0,
    "view_transform": "Standard",
    "linear_exr": os.path.basename(exr_path),
    "cases": cases,
    "inputs": {
        "attribute_source": "ATTR_STD_SHADOW_TRANSPARENCY",
        "attribute_name": "shadow_transparency",
        "expected_baked_shadow_transparency": EXPECTED_BAKED_SHADOW_TRANSPARENCY,
        "curve": {
            "name": "Camera Hidden Hair",
            "spline_type": "CATMULL_ROM",
            "key_positions": CURVE_POINTS,
            "key_radii": [CURVE_RADIUS] * len(CURVE_POINTS),
            "visible_camera": False,
            "visible_diffuse": False,
            "visible_glossy": False,
            "visible_transmission": False,
            "visible_volume_scatter": False,
            "visible_shadow": True,
        },
        "hair_material": {
            "nodes": ["ShaderNodeBsdfTransparent", "ShaderNodeBsdfDiffuse", "ShaderNodeMixShader"],
            "transparent_color": TRANSPARENT_COLOR,
            "mix_factor": MIX_FACTOR,
            "transparent_weight": 1.0 - MIX_FACTOR,
            "opaque_closure": "ShaderNodeBsdfDiffuse",
            "dynamic_hair_model": None,
        },
        "receiver": {"diffuse_color": RECEIVER_COLOR},
        "sun": {"type": "SUN", "energy": SUN_ENERGY, "angle": 0.0, "rotation_euler": [0.0, 0.0, 0.0]},
        "probes": {
            "shadow_patch": {"pixel": list(SHADOW_PROBE[:2]), "radius": PATCH_RADIUS},
            "lit_patch": {"pixel": list(LIT_PROBE[:2]), "radius": PATCH_RADIUS},
        },
    },
}
with open(os.path.join(out_dir, "hair_shadow_transparency.json"), "w", encoding="utf-8") as stream:
    json.dump(capture, stream, indent=2)
    stream.write("\n")
bpy.data.images.remove(image)
