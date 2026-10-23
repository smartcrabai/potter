"""Render a native EEVEE point-light shape through a spatially varying mesh volume.

Run with Blender 5.2.2:
    Blender -b --factory-startup -P eevee_mesh_volume_gradient.py -- SCENE.blend CAPTURE.f32le

The script writes bottom-up RGB float32 pixels from the 64x64 EEVEE render. It also renders a clear-light
control and requires attenuation in the central 2x2 pixel window. The volume's Generated Z coordinate is
zero-density at the camera-entry face and increases in density toward the interior.
"""

import os
import struct
import sys
import tempfile

import bpy

RESOLUTION = 64
CENTER_WINDOW = 2

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"fixture requires Blender 5.2.2, found {bpy.app.version_string}")

blend_out, raw_out = sys.argv[sys.argv.index("--") + 1 :][:2]
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.eevee.taa_render_samples = 64
scene.eevee.volumetric_tile_size = "2"
scene.eevee.volumetric_samples = 64
scene.eevee.volumetric_start = 0.1
scene.eevee.volumetric_end = 10.0
scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_mode = "RGBA"
scene.render.image_settings.color_depth = "32"
scene.render.filter_size = 0.0
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0
scene.display_settings.display_device = "sRGB"

world = bpy.data.worlds.new("BlackWorld")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
scene.world = world

camera_data = bpy.data.cameras.new("OracleCamera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 8.0
camera = bpy.data.objects.new("OracleCamera", camera_data)
camera.location = (0.0, 0.0, 10.0)
scene.collection.objects.link(camera)
scene.camera = camera

light_data = bpy.data.lights.new("PointLight", "POINT")
light_data.energy = 5.0
light_data.shadow_soft_size = 1.0
light_data.use_soft_falloff = False
light = bpy.data.objects.new("PointLight", light_data)
light.location = (0.0, 0.0, 0.0)
light.visible_camera = True
scene.collection.objects.link(light)

bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 5.0))
volume_object = bpy.context.object
volume_object.name = "GradientVolume"
volume_object.scale = (4.0, 4.0, 1.0)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

material = bpy.data.materials.new("EntryZeroGradientAbsorption")
material.use_nodes = True
nodes = material.node_tree.nodes
nodes.clear()
output = nodes.new("ShaderNodeOutputMaterial")
coordinates = nodes.new("ShaderNodeTexCoord")
separate = nodes.new("ShaderNodeSeparateXYZ")
subtract = nodes.new("ShaderNodeMath")
subtract.operation = "SUBTRACT"
subtract.inputs[0].default_value = 1.0
multiply = nodes.new("ShaderNodeMath")
multiply.operation = "MULTIPLY"
multiply.inputs[1].default_value = 1.5
absorption = nodes.new("ShaderNodeVolumeAbsorption")
absorption.inputs["Color"].default_value = (0.5, 0.5, 0.5, 1.0)
material.node_tree.links.new(coordinates.outputs["Generated"], separate.inputs["Vector"])
material.node_tree.links.new(separate.outputs["Z"], subtract.inputs[1])
material.node_tree.links.new(subtract.outputs["Value"], multiply.inputs[0])
material.node_tree.links.new(multiply.outputs["Value"], absorption.inputs["Density"])
material.node_tree.links.new(absorption.outputs["Volume"], output.inputs["Volume"])
volume_object.data.materials.append(material)


def render_pixels(filepath):
    scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(filepath, check_existing=False)
    rgba = [0.0] * (RESOLUTION * RESOLUTION * 4)
    image.pixels.foreach_get(rgba)
    bpy.data.images.remove(image)
    return rgba


def center_mean(rgb, channels):
    values = []
    start = RESOLUTION // 2 - CENTER_WINDOW // 2
    for y in range(start, start + CENTER_WINDOW):
        for x in range(start, start + CENTER_WINDOW):
            offset = (y * RESOLUTION + x) * channels
            values.extend(rgb[offset : offset + channels])
    return sum(values) / len(values)


scene.render.filepath = "//mesh_volume_gradient"
bpy.ops.wm.save_as_mainfile(filepath=blend_out)

with tempfile.TemporaryDirectory(prefix="eevee_mesh_volume_gradient_") as temp_dir:
    volume_object.hide_render = True
    clear_rgba = render_pixels(os.path.join(temp_dir, "clear.exr"))
    volume_object.hide_render = False
    clear_rgb = [component for i in range(0, len(clear_rgba), 4) for component in clear_rgba[i : i + 3]]
    clear_mean = center_mean(clear_rgb, 3)

    volume_rgba = render_pixels(os.path.join(temp_dir, "volume.exr"))
    volume_rgb = [component for i in range(0, len(volume_rgba), 4) for component in volume_rgba[i : i + 3]]
    volume_mean = center_mean(volume_rgb, 3)
    if not (0.0 < volume_mean < clear_mean):
        raise RuntimeError(f"volume must attenuate the clear center window: volume={volume_mean}, clear={clear_mean}")

with open(raw_out, "wb") as handle:
    handle.write(struct.pack("<" + "f" * len(volume_rgb), *volume_rgb))
print(f"WROTE {raw_out} ({RESOLUTION}x{RESOLUTION} RGB f32le)")
print(f"center_window={CENTER_WINDOW}x{CENTER_WINDOW} clear_mean={clear_mean:.9g} volume_mean={volume_mean:.9g}")
