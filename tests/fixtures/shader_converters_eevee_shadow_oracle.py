"""Blender 5.2.2 EEVEE oracle for a noise-alpha caster with Bump on its shading normal."""

import json
import sys

import bpy

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"expected Blender 5.2.2, got {bpy.app.version_string}")

arguments = sys.argv[sys.argv.index("--") + 1 :]
blend_out, exr_out, json_out = arguments[:3]
mode = arguments[3] if len(arguments) > 3 else "normal_only"
if mode not in ("normal_only", "opacity_bump"):
    raise ValueError(f"unknown alpha+Bump shadow mode `{mode}`")

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.eevee.taa_render_samples = 16
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.render.resolution_x = 64
scene.render.resolution_y = 64
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.render.filepath = exr_out

world = bpy.data.worlds.new("Black World")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
scene.world = world

floor = bpy.data.materials.new("Shadow Oracle Floor")
floor.use_nodes = True
floor_shader = floor.node_tree.nodes["Principled BSDF"]
floor_shader.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
floor_shader.inputs["Roughness"].default_value = 1.0
floor_shader.inputs["Specular IOR Level"].default_value = 0.0
bpy.ops.mesh.primitive_plane_add(size=20.0, location=(0.0, 0.0, 0.0))
bpy.context.object.data.materials.append(floor)

caster = bpy.data.materials.new("Noise Alpha Bump Shadow Caster")
caster.use_nodes = True
caster.surface_render_method = "BLENDED"
caster.use_transparent_shadow = True
caster_tree = caster.node_tree
principled = caster_tree.nodes["Principled BSDF"]
principled.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
coordinates = caster_tree.nodes.new("ShaderNodeTexCoord")
noise = caster_tree.nodes.new("ShaderNodeTexNoise")
noise.inputs["Scale"].default_value = 8.0
noise.inputs["Detail"].default_value = 0.0
bump = caster_tree.nodes.new("ShaderNodeBump")
bump.inputs["Distance"].default_value = 0.2
caster_tree.links.new(coordinates.outputs["UV"], noise.inputs["Vector"])
caster_tree.links.new(noise.outputs["Fac"], bump.inputs["Height"])
if mode == "normal_only":
    caster_tree.links.new(noise.outputs["Fac"], principled.inputs["Alpha"])
    caster_tree.links.new(bump.outputs["Normal"], principled.inputs["Normal"])
else:
    separate_normal = caster_tree.nodes.new("ShaderNodeSeparateXYZ")
    caster_tree.links.new(bump.outputs["Normal"], separate_normal.inputs["Vector"])
    caster_tree.links.new(separate_normal.outputs["Z"], principled.inputs["Alpha"])
bpy.ops.mesh.primitive_plane_add(size=0.9, location=(-3.0, 0.0, 1.0))
bpy.context.object.data.materials.append(caster)

sun_data = bpy.data.lights.new("Shadow Oracle Sun", "SUN")
sun_data.energy = 1.0
sun = bpy.data.objects.new("Shadow Oracle Sun", sun_data)
scene.collection.objects.link(sun)
sun.rotation_euler = (0.0, -0.7853981634, 0.0)

camera_data = bpy.data.cameras.new("Shadow Oracle Camera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 2.0
camera = bpy.data.objects.new("Shadow Oracle Camera", camera_data)
scene.collection.objects.link(camera)
camera.location = (-1.5, 0.0, 6.0)
scene.camera = camera

bpy.ops.wm.save_as_mainfile(filepath=blend_out, check_existing=False)
bpy.ops.render.render(write_still=True)
rendered = bpy.data.images.load(exr_out, check_existing=False)
pixels = list(rendered.pixels)
width, height = (int(value) for value in rendered.size)

def patch_mean(center_x):
    totals = [0.0, 0.0, 0.0]
    count = 0
    for y in range(29, 36):
        for x in range(center_x - 3, center_x + 4):
            offset = (y * width + x) * 4
            for channel in range(3):
                totals[channel] += pixels[offset + channel]
            count += 1
    return [value / count for value in totals]

with open(json_out, "w", encoding="utf-8") as stream:
    json.dump(
        {
            "engine": "Blender 5.2.2 EEVEE",
            "resolution": [width, height],
            "shadow_patch_rgb": patch_mean(16),
            "lit_patch_rgb": patch_mean(38),
        },
        stream,
        indent=2,
    )
