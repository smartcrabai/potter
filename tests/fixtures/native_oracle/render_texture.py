"""PR9 texture orientation oracle: renders the `TexturePlane` texture in Blender (5.2.2).

Usage:
  Blender --factory-startup -b FILE.blend --python render_texture.py -- OUT.json RENDER.exr

The texture that feeds Base Color is connected to an Emission (strength 1) that drives the
material output, so the render shows the texture alone. Every other object is hidden. A
Cycles orthographic camera frames the plane exactly, with the render size equal to the image
size and a box pixel filter of width 0.01, so each render pixel samples one texel.

Writes JSON with the image's own `pixels` and the render, both listed bottom row first (Blender's
row order), as linear RGB. Nothing is inferred from Potter: the file's own nodes and layers are
used as they are.
"""

import json
import sys

import bpy

ARGUMENTS = sys.argv[sys.argv.index("--") + 1 :]
OUT_JSON, OUT_EXR = ARGUMENTS[0], ARGUMENTS[1]

scene = bpy.context.scene
plane = bpy.data.objects["TexturePlane"]
for obj in scene.objects:
    if obj is not plane:
        obj.hide_render = True

tree = plane.active_material.node_tree
output = next(node for node in tree.nodes if node.type == "OUTPUT_MATERIAL")
base_link = next(
    link
    for link in tree.links
    if link.to_node.type == "BSDF_PRINCIPLED" and link.to_socket.name == "Base Color"
)
texture = base_link.from_node
if texture.type != "TEX_IMAGE" or texture.image is None:
    raise SystemExit("Base Color is not fed by an image texture")
uv_source = None
for link in tree.links:
    if link.to_node.name == texture.name and link.to_socket.name == "Vector":
        uv_source = link.from_node.uv_map if link.from_node.type == "UVMAP" else None

for link in list(output.inputs["Surface"].links):
    tree.links.remove(link)
emission = tree.nodes.new("ShaderNodeEmission")
emission.inputs["Strength"].default_value = 1.0
tree.links.new(texture.outputs["Color"], emission.inputs["Color"])
tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])


def rows_bottom_up(pixels, width, height):
    return [
        [[pixels[(y * width + x) * 4 + c] for c in range(3)] for x in range(width)]
        for y in range(height)
    ]


image = texture.image
image_width, image_height = image.size
image_pixels = rows_bottom_up(list(image.pixels), image_width, image_height)

camera_data = bpy.data.cameras.new("TextureCamera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 2.0
camera = bpy.data.objects.new("TextureCamera", camera_data)
scene.collection.objects.link(camera)
camera.location = (plane.location.x, plane.location.y, 5.0)
scene.camera = camera
scene.world = None

scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 16
scene.cycles.use_denoising = False
scene.cycles.max_bounces = 0
scene.render.resolution_x = image_width
scene.render.resolution_y = image_height
scene.render.resolution_percentage = 100
scene.cycles.pixel_filter_type = "BOX"
scene.cycles.filter_width = 0.01
scene.view_settings.view_transform = "Standard"
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"

bpy.ops.render.render(write_still=False)
bpy.data.images["Render Result"].save_render(OUT_EXR)
rendered = bpy.data.images.load(OUT_EXR, check_existing=False)
render_width, render_height = rendered.size
render_pixels = rows_bottom_up(list(rendered.pixels), render_width, render_height)

with open(OUT_JSON, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "image_name": image.name,
            "image_size": [image_width, image_height],
            "interpolation": texture.interpolation,
            "uv_map": uv_source,
            "image_pixels_bottom_up": image_pixels,
            "render_size": [render_width, render_height],
            "render_bottom_up": render_pixels,
        },
        handle,
    )
