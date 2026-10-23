"""EEVEE oracle for a point shape behind an opaque camera surface (Blender 5.2.2).

The point's physical sphere front is in front of the plane, but EEVEE draws its camera shape at
its center plane, behind the plane. The image must therefore contain no point shape.

Arguments after `--`: blend_out, exr_out (or `-` to save only the `.blend`).
"""

import sys

import bpy

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"fixture requires Blender 5.2.2, found {bpy.app.version_string}")

blend_out, exr_out = sys.argv[sys.argv.index("--") + 1 :][:2]
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.eevee.taa_render_samples = 64
scene.render.resolution_x = 64
scene.render.resolution_y = 64
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_mode = "RGBA"
scene.render.image_settings.color_depth = "32"
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0

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

point_data = bpy.data.lights.new("PointLight", "POINT")
point_data.energy = 5.0
point_data.shadow_soft_size = 1.0
point_data.use_soft_falloff = False
point = bpy.data.objects.new("PointLight", point_data)
point.location = (0.0, 0.0, 0.0)
point.visible_camera = True
scene.collection.objects.link(point)

bpy.ops.mesh.primitive_plane_add(size=8.0, location=(0.0, 0.0, 0.5))
occluder = bpy.context.object
occluder.name = "OpaqueOccluder"
material = bpy.data.materials.new("BlackOccluder")
material.use_nodes = True
material.node_tree.nodes.clear()
output = material.node_tree.nodes.new("ShaderNodeOutputMaterial")
emission = material.node_tree.nodes.new("ShaderNodeEmission")
emission.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
emission.inputs["Strength"].default_value = 0.0
material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
occluder.data.materials.append(material)

bpy.ops.wm.save_as_mainfile(filepath=blend_out)
if exr_out != "-":
    scene.render.filepath = exr_out
    bpy.ops.render.render(write_still=True)
    print("WROTE", exr_out)
