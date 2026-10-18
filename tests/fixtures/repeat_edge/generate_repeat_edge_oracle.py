"""Blender 5.2.2's Image Texture colour at exact integer UVs, as the oracle for Potter's Repeat sampling at texel edges.

Regenerate from this directory with Blender 5.2.2, one thread, one run per case:

    Blender -b --factory-startup --threads 1 -P generate_repeat_edge_oracle.py -- INTERPOLATION OUT.json

INTERPOLATION is Closest or Linear. Extension is Repeat.

The scene is a 4 x 4 frame covered by a 4 x 4 plane with UV = 2 x vertex position, so each pixel centre has an exact
integer UV: u and v are each one of -3, -1, 1, 3. The texture is a 4 x 4 float image whose texel at Blender row y
(bottom-up) and column x holds (x / 4, y / 4, 0.5), so every texel is distinct. Emission reads the texture colour, and a
second render of the same geometry reads the UV that Blender interpolated (a float32 EXR, stored exactly).

Each record is [u, v, r, g, b, sample_u, sample_v]: the intended pixel-centre UV, the rendered colour, and the UV the
Image Texture node received. The texel grid is recorded in Blender's row order.
"""

import json
import os
import sys
import tempfile

import bpy

argv = sys.argv[sys.argv.index("--") + 1 :]
interpolation, out_path = argv[:2]
out_path = os.path.abspath(out_path)
work = tempfile.mkdtemp(prefix="repeat_edge_")
SIZE = 4
TEXELS = SIZE


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    render = scene.render
    render.engine = "CYCLES"
    render.resolution_x = SIZE
    render.resolution_y = SIZE
    render.resolution_percentage = 100
    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    if hasattr(scene.cycles, "pixel_filter_type"):
        scene.cycles.pixel_filter_type = "BOX"
    else:
        scene.cycles.filter_type = "BOX"
    scene.cycles.filter_width = 0.0
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    camera_data = bpy.data.cameras.new("cam")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = float(SIZE)
    camera = bpy.data.objects.new("cam", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0.0, 0.0, 2.0)
    scene.camera = camera
    image_settings = render.image_settings
    image_settings.file_format = "OPEN_EXR"
    image_settings.color_depth = "32"
    image_settings.color_mode = "RGBA"
    image_settings.exr_codec = "NONE"
    return scene


# The plane spans -2..2, so its vertices are at (+-2, +-2) and carry UV (+-4, +-4) = 2 x position.
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = reset_scene()
bpy.ops.mesh.primitive_plane_add(size=float(SIZE), location=(0.0, 0.0, 0.0))
plane = bpy.context.active_object
plane.name = "Plane"
mesh = plane.data
layer = mesh.uv_layers.get("UVMap") or mesh.uv_layers.new(name="UVMap")
mesh.uv_layers.active = layer
for polygon in mesh.polygons:
    for loop_index in polygon.loop_indices:
        co = mesh.vertices[mesh.loops[loop_index].vertex_index].co
        layer.data[loop_index].uv = (2.0 * co.x, 2.0 * co.y)

# The texture: texel (x, y) in Blender's bottom-up rows is (x / 4, y / 4, 0.5), all distinct.
texture = bpy.data.images.new("edge_texture", TEXELS, TEXELS, alpha=False, float_buffer=True)
texture.colorspace_settings.name = "Non-Color"
pixels = []
for y in range(TEXELS):
    for x in range(TEXELS):
        pixels.extend([x / TEXELS, y / TEXELS, 0.5, 1.0])
texture.pixels.foreach_set(pixels)

material = bpy.data.materials.new("edge")
material.use_nodes = True
tree = material.node_tree
for node in list(tree.nodes):
    tree.nodes.remove(node)
uv = tree.nodes.new("ShaderNodeUVMap")
uv.uv_map = "UVMap"
tex = tree.nodes.new("ShaderNodeTexImage")
tex.image = texture
tex.interpolation = "Closest" if interpolation == "Closest" else "Linear"
tex.extension = "REPEAT"
emission = tree.nodes.new("ShaderNodeEmission")
emission.inputs["Strength"].default_value = 1.0
output = tree.nodes.new("ShaderNodeOutputMaterial")
split = tree.nodes.new("ShaderNodeSeparateXYZ")
tree.links.new(uv.outputs["UV"], split.inputs["Vector"])
round_u = tree.nodes.new("ShaderNodeMath")
round_u.operation = "ROUND"
round_v = tree.nodes.new("ShaderNodeMath")
round_v.operation = "ROUND"
tree.links.new(split.outputs["X"], round_u.inputs[0])
tree.links.new(split.outputs["Y"], round_v.inputs[0])
snapped = tree.nodes.new("ShaderNodeCombineXYZ")
tree.links.new(round_u.outputs["Value"], snapped.inputs["X"])
tree.links.new(round_v.outputs["Value"], snapped.inputs["Y"])
tree.links.new(snapped.outputs["Vector"], tex.inputs["Vector"])
tree.links.new(tex.outputs["Color"], emission.inputs["Color"])
tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
plane.data.materials.append(material)


def render_to(path):
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    loaded = bpy.data.images.load(path, check_existing=False)
    values = list(loaded.pixels)
    bpy.data.images.remove(loaded)
    return values


colour_pixels = render_to(os.path.join(work, "repeat_edge_%s_colour.exr" % interpolation.lower()))

# Second render: the emission reads the UV itself, so the file holds the UV the Image Texture received.
for link in list(tree.links):
    if link.from_node == tex and link.to_node == emission:
        tree.links.remove(link)
tree.links.new(snapped.outputs["Vector"], emission.inputs["Color"])
uv_pixels = render_to(os.path.join(work, "repeat_edge_%s_uv.exr" % interpolation.lower()))

records = []
for y in range(SIZE):
    v = -3.0 + 2.0 * y
    for x in range(SIZE):
        u = -3.0 + 2.0 * x
        offset = (y * SIZE + x) * 4
        records.append(
            [
                u,
                v,
                round(colour_pixels[offset], 6),
                round(colour_pixels[offset + 1], 6),
                round(colour_pixels[offset + 2], 6),
                uv_pixels[offset],
                uv_pixels[offset + 1],
            ]
        )

with open(out_path, "w") as handle:
    json.dump(
        {
            "colorspace": "Non-Color",
            "interpolation": interpolation,
            "extension": "REPEAT",
            "width": TEXELS,
            "height": TEXELS,
            "texels_blender_rows": [[x / TEXELS, y / TEXELS, 0.5, 1.0] for y in range(TEXELS) for x in range(TEXELS)],
            "records": records,
        },
        handle,
        indent=1,
    )
print("ORACLE_OK", interpolation, len(records))
