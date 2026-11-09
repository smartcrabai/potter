"""Render Blender 5.2.2 EEVEE offsets in the realtime direct-light plane/sphere scene."""

import bpy
import json
import math
import sys

case, blend_out, exr_out, json_out = sys.argv[sys.argv.index("--") + 1 :][:4]
CASES = {
    "zero": (0.0, 0.1),
    "normal": (0.35, 2.0),
    "geometry": (0.35, 1.5),
}
normal_offset, geometry_offset = CASES[case]
SIZE = 128
ORTHO_SCALE = 12.0
ALBEDO = 0.8

scene = bpy.context.scene
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

world = bpy.data.worlds.new("Oracle")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
scene.world = world

view = scene.view_settings
view.view_transform = "Standard"
view.look = "None"
view.exposure = 0.0
view.gamma = 1.0
scene.display_settings.display_device = "sRGB"

material = bpy.data.materials.new("Lambert")
material.use_nodes = True
nodes = material.node_tree.nodes
links = material.node_tree.links
for node in list(nodes):
    if node.type != "OUTPUT_MATERIAL":
        nodes.remove(node)
diffuse = nodes.new("ShaderNodeBsdfDiffuse")
diffuse.inputs["Color"].default_value = (ALBEDO, ALBEDO, ALBEDO, 1.0)
links.new(diffuse.outputs["BSDF"], nodes["Material Output"].inputs["Surface"])

bpy.ops.mesh.primitive_plane_add(size=40.0, location=(0.0, 0.0, 0.0))
plane = bpy.context.object
plane.name = "Plane"
plane.data.materials.append(material)
plane.shadow_terminator_normal_offset = normal_offset
plane.shadow_terminator_geometry_offset = geometry_offset

light_data = bpy.data.lights.new("Oracle Sun", type="SUN")
light_data.energy = 3.0
light_data.color = (1.0, 1.0, 1.0)
light_data.use_shadow = True
light_data.angle = math.radians(10.0)
light = bpy.data.objects.new("Grazing Sun", light_data)
scene.collection.objects.link(light)
light.location = (0.0, 0.0, 10.0)
light.rotation_euler = (0.0, 0.0, 0.0)

bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, segments=64, ring_count=32, location=(0.0, 0.0, 1.0))
occluder = bpy.context.object
occluder.name = "Occluder"
bpy.ops.object.shade_smooth()
occluder_material = bpy.data.materials.new("Occluder")
occluder_material.use_nodes = True
occluder.data.materials.append(occluder_material)

scene.eevee.light_threshold = 0.01

camera_data = bpy.data.cameras.new("OracleCamera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = ORTHO_SCALE
camera = bpy.data.objects.new("OracleCamera", camera_data)
scene.collection.objects.link(camera)
camera.location = (0.0, 0.0, 20.0)
scene.camera = camera

scene.render.engine = "BLENDER_EEVEE"
scene.eevee.taa_render_samples = 64
scene.render.resolution_x = SIZE
scene.render.resolution_y = SIZE
scene.render.resolution_percentage = 100
scene.render.filter_size = 0.0
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.render.filepath = exr_out

bpy.ops.wm.save_as_mainfile(filepath=blend_out)
bpy.ops.render.render(write_still=True)
image = bpy.data.images.load(exr_out, check_existing=False)
pixels = list(image.pixels)
with open(json_out, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "version": bpy.app.version_string,
            "width": image.size[0],
            "height": image.size[1],
            "pixels": [pixels[index : index + 3] for index in range(0, len(pixels), 4)],
        },
        handle,
    )
