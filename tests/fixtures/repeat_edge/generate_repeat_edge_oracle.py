"""Blender 5.2.2's Image Texture colour at exact and near-edge UVs, as the oracle for Potter's full-image sampling.

Regenerate from this directory with Blender 5.2.2, one thread, one run per case:

    Blender -b --factory-startup --threads 1 -P generate_repeat_edge_oracle.py -- \
        INTERPOLATION EXTENSION OUT.json

INTERPOLATION is Closest or Linear. EXTENSION is REPEAT, EXTEND, CLIP, or MIRROR.

The scene is a 16 x 16 frame covered by a 16 x 16 plane, so each pixel centre is at X = x - 7.5. The plane's UV is
u = 0.125 X + 0.1875 (and v likewise), so pixel-centre UVs run in eighths from -0.75 to 1.125: they include the texel
edges 0 and 1, the exact integers -1 and 2, and the bands just outside 0 and 1 (-0.125 and 1.125), where a linear sample
is partly inside the image. The texture is a 2 x 2 float image whose texel at Blender row y (bottom-up) and column x holds
(x / 2, y / 2, 0.5), so every texel is distinct. The texture input is snapped to eighths (Round(8 u) / 8, exact), which
removes the sub-pixel float offset of the camera. Emission reads the texture colour, and a second render of the same
geometry reads the snapped UV the Image Texture received.

Each record is [u, v, r, g, b, sample_u, sample_v]: the intended pixel-centre UV, the rendered colour, and the UV the
Image Texture node received. The texel grid is recorded in Blender's row order.
"""

import json
import os
import sys
import tempfile

import bpy

argv = sys.argv[sys.argv.index("--") + 1 :]
interpolation, extension, out_path = argv[:3]
out_path = os.path.abspath(out_path)
work = tempfile.mkdtemp(prefix="edge_oracle_")
SIZE = 16
TEXELS = 2


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
        layer.data[loop_index].uv = (0.125 * co.x + 0.1875, 0.125 * co.y + 0.1875)

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
split = tree.nodes.new("ShaderNodeSeparateXYZ")
tree.links.new(uv.outputs["UV"], split.inputs["Vector"])


def snap(source_output):
    """Round(8 x) / 8 for one component: an exact eighth-step coordinate."""
    scaled = tree.nodes.new("ShaderNodeMath")
    scaled.operation = "MULTIPLY"
    scaled.inputs[1].default_value = 8.0
    rounded = tree.nodes.new("ShaderNodeMath")
    rounded.operation = "ROUND"
    unscaled = tree.nodes.new("ShaderNodeMath")
    unscaled.operation = "DIVIDE"
    unscaled.inputs[1].default_value = 8.0
    tree.links.new(source_output, scaled.inputs[0])
    tree.links.new(scaled.outputs["Value"], rounded.inputs[0])
    tree.links.new(rounded.outputs["Value"], unscaled.inputs[0])
    return unscaled.outputs["Value"]


combined = tree.nodes.new("ShaderNodeCombineXYZ")
tree.links.new(snap(split.outputs["X"]), combined.inputs["X"])
tree.links.new(snap(split.outputs["Y"]), combined.inputs["Y"])
tex = tree.nodes.new("ShaderNodeTexImage")
tex.image = texture
tex.interpolation = "Closest" if interpolation == "Closest" else "Linear"
tex.extension = extension
emission = tree.nodes.new("ShaderNodeEmission")
emission.inputs["Strength"].default_value = 1.0
output = tree.nodes.new("ShaderNodeOutputMaterial")
tree.links.new(combined.outputs["Vector"], tex.inputs["Vector"])
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


colour_pixels = render_to(os.path.join(work, "colour.exr"))

# Second render: the emission reads the snapped vector the Image Texture received.
for link in list(tree.links):
    if link.from_node == tex and link.to_node == emission:
        tree.links.remove(link)
tree.links.new(combined.outputs["Vector"], emission.inputs["Color"])
uv_pixels = render_to(os.path.join(work, "uv.exr"))

records = []
for y in range(SIZE):
    v = 0.125 * (y - 6)
    for x in range(SIZE):
        u = 0.125 * (x - 6)
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
            "extension": extension,
            "width": TEXELS,
            "height": TEXELS,
            "texels_blender_rows": [
                [x / TEXELS, y / TEXELS, 0.5, 1.0] for y in range(TEXELS) for x in range(TEXELS)
            ],
            "records": records,
        },
        handle,
        indent=1,
    )
print("ORACLE_OK", interpolation, extension, len(records))
