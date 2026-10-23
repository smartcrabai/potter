"""EEVEE 5.2.2 captures for odd-viewport volume, scaled spot, and small-area boundaries.

Run with Blender 5.2.2:
    Blender -b --factory-startup -P eevee_light_shape_boundaries.py -- \
        ODD.blend SPOT.blend AREA.blend ODD.f32le SPOT.f32le AREA.f32le

Use `-` for a capture path to save only the corresponding scene. Captures are bottom-up RGB f32le.
"""

import math
import os
import struct
import sys
import tempfile

import bpy
from mathutils import Euler

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"fixture requires Blender 5.2.2, found {bpy.app.version_string}")

odd_blend, spot_blend, area_blend, odd_capture, spot_capture, area_capture = sys.argv[
    sys.argv.index("--") + 1 :
]


def new_scene(width=64, height=64, tile_size="8"):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = 64
    scene.eevee.volumetric_tile_size = tile_size
    scene.eevee.volumetric_samples = 64
    scene.eevee.volumetric_start = 0.1
    scene.eevee.volumetric_end = 10.0
    scene.render.resolution_x = width
    scene.render.resolution_y = height
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
    return scene


def add_camera(scene, camera_type="ORTHO", ortho_scale=8.0):
    data = bpy.data.cameras.new("OracleCamera")
    data.type = camera_type
    if camera_type == "ORTHO":
        data.ortho_scale = ortho_scale
    else:
        data.lens = 20.0
        data.sensor_width = 36.0
        data.sensor_fit = "HORIZONTAL"
    camera = bpy.data.objects.new("OracleCamera", data)
    camera.location = (0.0, 0.0, 10.0)
    scene.collection.objects.link(camera)
    scene.camera = camera


def add_capture(scene, output):
    if output == "-":
        return
    width, height = scene.render.resolution_x, scene.render.resolution_y
    with tempfile.TemporaryDirectory(prefix="eevee_light_shape_boundary_") as temp_dir:
        exr = os.path.join(temp_dir, "capture.exr")
        scene.render.filepath = exr
        bpy.ops.render.render(write_still=True)
        image = bpy.data.images.load(exr, check_existing=False)
        rgba = [0.0] * (width * height * 4)
        image.pixels.foreach_get(rgba)
        bpy.data.images.remove(image)
    rgb = [component for index in range(width * height) for component in rgba[index * 4 : index * 4 + 3]]
    with open(output, "wb") as handle:
        handle.write(struct.pack("<" + "f" * len(rgb), *rgb))
    print(f"WROTE {output} ({width}x{height} RGB f32le)")


# Perspective point emitter through a heterogeneous mesh volume at a 101x65 viewport and tile size 16.
scene = new_scene(101, 65, "16")
add_camera(scene, "PERSP")
light_data = bpy.data.lights.new("PointLight", "POINT")
light_data.energy = 5.0
light_data.shadow_soft_size = 1.0
light_data.use_soft_falloff = False
light = bpy.data.objects.new("PointLight", light_data)
light.location = (-6.0, 0.0, 0.0)
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
scene.render.filepath = "//eevee_light_shape_odd_viewport"
bpy.ops.wm.save_as_mainfile(filepath=odd_blend)
add_capture(scene, odd_capture)

# The orthographic camera is seven degrees outside the spot axis. Inverse-transforming its direction
# through the light's [1, 1, 2] matrix puts it beyond the 10-degree outer cone.
scene = new_scene()
add_camera(scene)
spot_data = bpy.data.lights.new("ScaledSpot", "SPOT")
spot_data.energy = 5.0
spot_data.shadow_soft_size = 0.5
spot_data.spot_size = math.radians(20.0)
spot_data.spot_blend = 0.15
spot_data.use_soft_falloff = False
spot = bpy.data.objects.new("ScaledSpot", spot_data)
spot.rotation_euler = Euler((math.pi, math.radians(7.0), 0.0))
spot.scale = (1.0, 1.0, 2.0)
spot.visible_camera = True
scene.collection.objects.link(spot)
scene.render.filepath = "//eevee_light_shape_scaled_spot"
bpy.ops.wm.save_as_mainfile(filepath=spot_blend)
add_capture(scene, spot_capture)

# A 0.004 square area light has 0.002 half-extents, whose product is below EEVEE's 1e-5 cutoff.
scene = new_scene()
add_camera(scene, "ORTHO", 0.25)
area_data = bpy.data.lights.new("SmallArea", "AREA")
area_data.energy = 10.0
area_data.shape = "SQUARE"
area_data.size = 0.004
area = bpy.data.objects.new("SmallArea", area_data)
area.rotation_euler = Euler((math.pi, 0.0, 0.0))
area.visible_camera = True
scene.collection.objects.link(area)
scene.render.filepath = "//eevee_light_shape_small_area"
bpy.ops.wm.save_as_mainfile(filepath=area_blend)
add_capture(scene, area_capture)
