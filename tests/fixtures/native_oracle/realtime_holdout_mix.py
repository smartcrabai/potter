"""Generate the Blender 5.2.2 EEVEE mixed-Holdout scene and raw RGBA oracle capture.

From the repository root, run:

    /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 1 \
        --python-exit-code 3 --python tests/fixtures/native_oracle/realtime_holdout_mix.py -- \
        tests/fixtures/native_oracle/eevee_macos/holdout_mix.blend \
        tests/fixtures/native_oracle/eevee_macos/holdout_mix.exr \
        tests/fixtures/native_oracle/eevee_macos/holdout_mix.json

The JSON pixel values are read back from the rendered EXR; the script does not derive expectations from Potter.
"""

import json
import os
import sys

import bpy

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"fixture requires Blender 5.2.2, found {bpy.app.version_string}")

arguments = sys.argv[sys.argv.index("--") + 1 :]
if len(arguments) != 3:
    raise RuntimeError("expected output .blend, .exr, and .json paths after --")
blend_out, exr_out, json_out = arguments

scene = bpy.context.scene
engine_ids = [item.identifier for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
scene.render.engine = next(engine for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE") if engine in engine_ids)
scene.render.resolution_x = 64
scene.render.resolution_y = 64
scene.render.resolution_percentage = 100
scene.render.film_transparent = True
scene.eevee.taa_render_samples = 16
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_mode = "RGBA"
scene.render.image_settings.color_depth = "32"
scene.render.image_settings.exr_codec = "NONE"
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0
scene.render.filepath = "//holdout_mix.exr"
bpy.context.preferences.filepaths.save_version = 0

for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)
for material in list(bpy.data.materials):
    bpy.data.materials.remove(material)

world = bpy.data.worlds.new("BlackWorld")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
scene.world = world

camera_data = bpy.data.cameras.new("HoldoutMixCamera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 8.0
camera = bpy.data.objects.new("HoldoutMixCamera", camera_data)
scene.collection.objects.link(camera)
camera.location = (0.0, 0.0, 10.0)
scene.camera = camera

backplate = bpy.data.materials.new("BACKPLATE_OPAQUE_EMISSION")
backplate.use_nodes = True
backplate_tree = backplate.node_tree
backplate_tree.nodes.clear()
backplate_output = backplate_tree.nodes.new("ShaderNodeOutputMaterial")
backplate_emission = backplate_tree.nodes.new("ShaderNodeEmission")
backplate_color = (0.08, 0.36, 0.72)
backplate_emission.inputs["Color"].default_value = (*backplate_color, 1.0)
backplate_emission.inputs["Strength"].default_value = 1.0
backplate_tree.links.new(backplate_emission.outputs["Emission"], backplate_output.inputs["Surface"])
bpy.ops.mesh.primitive_plane_add(size=40.0, location=(0.0, 0.0, -1.0))
backplate_obj = bpy.context.object
backplate_obj.name = "OPAQUE_BACKPLATE"
backplate_obj.data.materials.append(backplate)


def mixed_material(name, closure_type):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.surface_render_method = "BLENDED"
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    output.name = "MATERIAL_OUTPUT"
    holdout = tree.nodes.new("ShaderNodeHoldout")
    holdout.name = "HOLDOUT_CLOSURE"
    if closure_type == "DIFFUSE":
        closure = tree.nodes.new("ShaderNodeBsdfDiffuse")
        closure.name = "OPAQUE_DIFFUSE_CLOSURE"
        closure.inputs["Color"].default_value = (0.8, 0.32, 0.12, 1.0)
        closure.inputs["Roughness"].default_value = 0.0
    elif closure_type == "TRANSPARENT":
        closure = tree.nodes.new("ShaderNodeBsdfTransparent")
        closure.name = "TRANSPARENT_CLOSURE"
        closure.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    else:
        raise RuntimeError(f"unknown closure type {closure_type}")
    mix = tree.nodes.new("ShaderNodeMixShader")
    mix.name = "MIX_HOLDOUT_AND_" + closure_type
    mix.inputs["Fac"].default_value = 0.5
    tree.links.new(holdout.outputs["Holdout"], mix.inputs[1])
    tree.links.new(closure.outputs[0], mix.inputs[2])
    tree.links.new(mix.outputs["Shader"], output.inputs["Surface"])
    return material


# Row 32 targets are centered on the pair of central image rows; x=20 and x=44 are pixel-centered.
for name, x, closure_type in (
    ("PROBE_HOLDOUT_DIFFUSE_50_50", -1.4375, "DIFFUSE"),
    ("PROBE_HOLDOUT_TRANSPARENT_50_50", 1.5625, "TRANSPARENT"),
):
    bpy.ops.mesh.primitive_plane_add(size=1.5, location=(x, 0.0, 0.0))
    probe = bpy.context.object
    probe.name = name
    probe.data.materials.append(mixed_material(name + "_MATERIAL", closure_type))

light_data = bpy.data.lights.new("DIFFUSE_PROBE_POINT_LIGHT", "POINT")
light_data.energy = 100.0
light_data.color = (1.0, 1.0, 1.0)
light = bpy.data.objects.new("DIFFUSE_PROBE_POINT_LIGHT", light_data)
scene.collection.objects.link(light)
light.location = (-1.4375, 0.0, 4.0)

for output_path in (blend_out, exr_out, json_out):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=blend_out)
bpy.ops.render.render(write_still=False)
bpy.data.images["Render Result"].save_render(exr_out, scene=scene)
image = bpy.data.images.load(exr_out, check_existing=False)
if tuple(image.size) != (64, 64) or image.channels != 4:
    raise RuntimeError(f"unexpected oracle EXR shape/channels: {image.size[:]} / {image.channels}")


def rgba_at(x, y):
    offset = (y * 64 + x) * image.channels
    return [float(channel) for channel in image.pixels[offset : offset + 4]]


probes = {
    "holdout_diffuse": {
        "contract_id": "EEVEE_HOLDOUT_DIFFUSE_50_50",
        "target": {"xy": [20, 32], "rgba": rgba_at(20, 32)},
        "backplate_neighbor": {"xy": [10, 32], "rgba": rgba_at(10, 32)},
    },
    "holdout_transparent": {
        "contract_id": "EEVEE_HOLDOUT_TRANSPARENT_50_50",
        "target": {"xy": [44, 32], "rgba": rgba_at(44, 32)},
        "backplate_neighbor": {"xy": [54, 32], "rgba": rgba_at(54, 32)},
    },
}
capture = {
    "blender_version": bpy.app.version_string,
    "engine": scene.render.engine,
    "resolution": [scene.render.resolution_x, scene.render.resolution_y],
    "samples": scene.eevee.taa_render_samples,
    "film_transparent": scene.render.film_transparent,
    "view_transform": scene.view_settings.view_transform,
    "backplate_rgb": list(backplate_color),
    "probe_size_world_units": 1.5,
    "probes": probes,
    "artifacts": {"blend": os.path.basename(blend_out), "exr": os.path.basename(exr_out)},
}
with open(json_out, "w", encoding="utf-8") as capture_file:
    json.dump(capture, capture_file, indent=2)
    capture_file.write("\n")
bpy.data.images.remove(image)
print("HOLDOUT_MIX_CAPTURE", json.dumps(probes, sort_keys=True), flush=True)
