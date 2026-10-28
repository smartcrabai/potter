"""Blender 5.2.2 EEVEE oracle for a VDB volume's realtime shader context."""

import json
import os
import tempfile
from pathlib import Path

import bpy

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"expected Blender 5.2.2, got {bpy.app.version_string}")

ROOT = Path(__file__).resolve().parent
VDB = ROOT.parent / "openvdb_transform_types" / "density.vdb"
BLEND = ROOT / "realtime_volume_context.blend"
ORACLE = ROOT / "realtime_volume_context.json"
SIZE = 16
PROBE = (8, 8)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.eevee.taa_render_samples = 8
if hasattr(scene.eevee, "volumetric_samples"):
    scene.eevee.volumetric_samples = 32
scene.render.resolution_x = SIZE
scene.render.resolution_y = SIZE
scene.render.resolution_percentage = 100
scene.render.filter_size = 0.0
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.render.image_settings.color_mode = "RGBA"
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0

world = bpy.data.worlds.new("Black World")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
scene.world = world

bpy.ops.object.volume_import(filepath=str(VDB))
volume = bpy.context.object
if volume is None or volume.type != "VOLUME":
    raise RuntimeError("Blender did not import the committed density VDB as a Volume object")
volume.name = "Random Target"
volume.data.filepath = "//../openvdb_transform_types/density.vdb"
volume.pass_index = 29
volume.color = (0.2, 0.4, 0.6, 0.7)

material = bpy.data.materials.new("Realtime Volume Context")
material.use_nodes = True
material.pass_index = 11
volume.data.materials.append(material)
tree = material.node_tree
nodes = tree.nodes
links = tree.links
output = nodes.get("Material Output")
for node in list(nodes):
    if node != output:
        nodes.remove(node)

info = nodes.new("ShaderNodeObjectInfo")
falloff = nodes.new("ShaderNodeLightFalloff")
falloff.inputs["Smooth"].default_value = 0.0


def weighted(source, scale):
    multiply = nodes.new("ShaderNodeMath")
    multiply.operation = "MULTIPLY"
    multiply.inputs[1].default_value = scale
    links.new(source, multiply.inputs[0])
    return multiply.outputs["Value"]


terms = [
    weighted(info.outputs["Alpha"], 1.5),
    weighted(info.outputs["Object Index"], 0.015),
    weighted(info.outputs["Random"], 1.0),
    weighted(info.outputs["Material Index"], 0.07),
]
strength = terms[0]
for term in terms[1:]:
    add = nodes.new("ShaderNodeMath")
    add.operation = "ADD"
    links.new(strength, add.inputs[0])
    links.new(term, add.inputs[1])
    strength = add.outputs["Value"]
links.new(strength, falloff.inputs["Strength"])

principled = nodes.new("ShaderNodeVolumePrincipled")
principled.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
principled.inputs["Density"].default_value = 1.0
principled.inputs["Anisotropy"].default_value = 0.0
links.new(info.outputs["Color"], principled.inputs["Emission Color"])
links.new(falloff.outputs["Linear"], principled.inputs["Emission Strength"])
links.new(principled.outputs["Volume"], output.inputs["Volume"])

camera_data = bpy.data.cameras.new("Volume Context Camera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 1.5
camera = bpy.data.objects.new("Volume Context Camera", camera_data)
scene.collection.objects.link(camera)
camera.location = (0.0, 0.0, 3.0)
scene.camera = camera

with tempfile.TemporaryDirectory(prefix="realtime_volume_context_") as temp_dir:
    exr = os.path.join(temp_dir, "realtime_volume_context.exr")
    scene.render.filepath = exr
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND), check_existing=False)
    bpy.ops.render.render(write_still=True)
    rendered = bpy.data.images.load(exr, check_existing=False)
    pixels = list(rendered.pixels)
    x, y = PROBE
    offset = ((SIZE - 1 - y) * SIZE + x) * 4
    rgba = pixels[offset : offset + 4]
    bpy.data.images.remove(rendered)

result = {
    "engine": "Blender 5.2.2 EEVEE",
    "fixture": BLEND.name,
    "vdb": "../openvdb_transform_types/density.vdb",
    "resolution": [SIZE, SIZE],
    "samples": scene.eevee.taa_render_samples,
    "object": {
        "name": volume.name,
        "color": list(volume.color),
        "pass_index": volume.pass_index,
        "material_pass_index": material.pass_index,
    },
    "probe": {
        "pixel": list(PROBE),
        "uv": [(PROBE[0] + 0.5) / SIZE, (PROBE[1] + 0.5) / SIZE],
        "rgba": rgba,
    },
}
ORACLE.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print("ORACLE_NATIVE Blender 5.2.2 EEVEE center rgba", json.dumps(rgba))
print("ORACLE_OK realtime_volume_context", ORACLE)
