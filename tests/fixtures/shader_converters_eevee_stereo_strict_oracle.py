"""Blender 5.2.2 EEVEE stereo Camera Data oracle with point-sampled, exact distances.

Arguments after `--`: blend_out, left_exr, right_exr, json_out.
"""

import json
import sys

import bpy

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"expected Blender 5.2.2, got {bpy.app.version_string}")

blend_out, left_exr, right_exr, json_out = sys.argv[sys.argv.index("--") + 1 :][:4]

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
world = bpy.data.worlds.new("Black")
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
scene.world = world
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0
scene.render.engine = "BLENDER_EEVEE"
scene.render.filter_size = 0.0
scene.eevee.taa_render_samples = 1
scene.render.resolution_x = 49
scene.render.resolution_y = 49
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"

material = bpy.data.materials.new("Stereo Camera Data")
material.use_nodes = True
tree = material.node_tree
tree.nodes.clear()
output = tree.nodes.new("ShaderNodeOutputMaterial")
emission = tree.nodes.new("ShaderNodeEmission")
camera_data = tree.nodes.new("ShaderNodeCameraData")
tree.links.new(camera_data.outputs["View Distance"], emission.inputs["Strength"])
tree.links.new(emission.outputs[0], output.inputs["Surface"])

bpy.ops.mesh.primitive_plane_add(size=8.0, location=(0.0, 0.0, 0.0))
plane = bpy.context.object
plane.name = "Stereo Camera Data Plane"
plane.data.materials.append(material)

camera_data_block = bpy.data.cameras.new("Stereo Strict Oracle Camera")
camera_data_block.type = "ORTHO"
camera_data_block.ortho_scale = 24.5
camera = bpy.data.objects.new("Stereo Strict Oracle Camera", camera_data_block)
scene.collection.objects.link(camera)
camera.location = (-2.0, 0.0, 3.0)
scene.camera = camera
bpy.ops.wm.save_as_mainfile(filepath=blend_out, check_existing=False)


def render_eye(camera_x, exr_path, pixel_x):
    camera.location.x = camera_x
    scene.render.filepath = exr_path
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(exr_path, check_existing=False)
    start = (24 * 49 + pixel_x) * 4
    return list(image.pixels[start : start + 4])


# The left eye is at x=-4 and sees the point (0, 0, 0) at a 3-4-5 distance;
# the right eye is at x=0 and sees the same point at distance 3. Odd width and
# half-unit texels place both probes exactly at pixel centers.
left_rgba = render_eye(-4.0, left_exr, 32)
right_rgba = render_eye(0.0, right_exr, 24)
if abs(left_rgba[0] - 5.0) > 1.0e-3 or abs(right_rgba[0] - 3.0) > 1.0e-3:
    raise RuntimeError(f"unexpected stereo Camera Data samples: {left_rgba}, {right_rgba}")

with open(json_out, "w", encoding="utf-8") as stream:
    json.dump(
        {
            "engine": "Blender 5.2.2 EEVEE",
            "resolution": [98, 49],
            "left": {"uv": [32.5 / 98.0, 24.5 / 49.0], "rgba": left_rgba},
            "right": {"uv": [73.5 / 98.0, 24.5 / 49.0], "rgba": right_rgba},
        },
        stream,
        indent=2,
    )
