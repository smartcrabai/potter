"""Render a native EEVEE point-light shape through a spatially varying mesh volume.

Run with Blender 5.2.2:
    Blender -b --factory-startup -P eevee_mesh_volume_gradient.py -- SCENE.blend CAPTURE.f32le [PERSP|ORTHO TILE POINT_X]

The script writes bottom-up RGB float32 pixels from the 64x64 EEVEE render. It also renders a clear-light
control and requires attenuation in the point-shape center window. The volume's Generated Z coordinate is
zero-density at the camera-entry face and increases in density toward the interior.
"""

import os
import struct
import sys
import tempfile

import bpy

RESOLUTION = 64
CENTER_WINDOW = 2
ORTHO_SCALE = 8.0
CAMERA_LENS = 20.0
CAMERA_SENSOR_WIDTH = 36.0
CAMERA_DISTANCE = 10.0

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"fixture requires Blender 5.2.2, found {bpy.app.version_string}")

arguments = sys.argv[sys.argv.index("--") + 1 :]
blend_out, raw_out = arguments[:2]
camera_type = arguments[2] if len(arguments) > 2 else "ORTHO"
tile_size = arguments[3] if len(arguments) > 3 else "2"
point_x = float(arguments[4]) if len(arguments) > 4 else 0.0
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.eevee.taa_render_samples = 64
scene.eevee.volumetric_tile_size = tile_size
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
camera_data.type = camera_type
if camera_type == "ORTHO":
    camera_data.ortho_scale = ORTHO_SCALE
else:
    camera_data.lens = CAMERA_LENS
    camera_data.sensor_width = CAMERA_SENSOR_WIDTH
    camera_data.sensor_fit = "HORIZONTAL"
camera = bpy.data.objects.new("OracleCamera", camera_data)
camera.location = (0.0, 0.0, 10.0)
scene.collection.objects.link(camera)
scene.camera = camera
if camera_type == "ORTHO":
    center_column = round((point_x / ORTHO_SCALE + 0.5) * RESOLUTION)
else:
    frame_width = CAMERA_DISTANCE * CAMERA_SENSOR_WIDTH / CAMERA_LENS
    center_column = round((point_x / frame_width + 0.5) * RESOLUTION)

light_data = bpy.data.lights.new("PointLight", "POINT")
light_data.energy = 5.0
light_data.shadow_soft_size = 1.0
light_data.use_soft_falloff = False
light = bpy.data.objects.new("PointLight", light_data)
light.location = (point_x, 0.0, 0.0)
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


def center_mean(rgb, channels, column):
    values = []
    start_x = column - CENTER_WINDOW // 2
    start_y = RESOLUTION // 2 - CENTER_WINDOW // 2
    for y in range(start_y, start_y + CENTER_WINDOW):
        for x in range(start_x, start_x + CENTER_WINDOW):
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
    clear_mean = center_mean(clear_rgb, 3, center_column)

    volume_rgba = render_pixels(os.path.join(temp_dir, "volume.exr"))
    volume_rgb = [component for i in range(0, len(volume_rgba), 4) for component in volume_rgba[i : i + 3]]
    volume_mean = center_mean(volume_rgb, 3, center_column)
    if not (0.0 < volume_mean < clear_mean):
        raise RuntimeError(f"volume must attenuate the clear center window: volume={volume_mean}, clear={clear_mean}")

with open(raw_out, "wb") as handle:
    handle.write(struct.pack("<" + "f" * len(volume_rgb), *volume_rgb))
print(f"WROTE {raw_out} ({RESOLUTION}x{RESOLUTION} RGB f32le)")
print(f"center_column={center_column} center_window={CENTER_WINDOW}x{CENTER_WINDOW} clear_mean={clear_mean:.9g} volume_mean={volume_mean:.9g}")
