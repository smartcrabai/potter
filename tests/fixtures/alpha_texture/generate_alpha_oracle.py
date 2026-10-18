"""Render Blender 5.2.2's Image Texture colour for a straight-alpha 2x1 image, as the oracle for Potter's alpha rule.

Regenerate from this directory with Blender 5.2.2, one thread, one run per case:

    Blender -b --factory-startup --threads 1 -P generate_alpha_oracle.py -- sRGB Closest EXTEND unlinked out.json

Arguments: colour space (sRGB or Non-Color), interpolation (Closest or Linear), extension (EXTEND or REPEAT), and
whether the Image Texture's Alpha output is linked (linked or unlinked). The image is two texels, [128, 40, 0, 128]
and [64, 0, 255, 255], the alpha-128 texel of the blend round-trip test. A linked Alpha feeds the Fac of a Mix whose two
colour inputs are both the texture's Color, so the rendered colour is the texture's Color but Cycles must compile the
Alpha link. A quad spans Blender UVs -0.5..1.5, covering both texels and both wraps. The Raw view transform and a
near-zero box filter give each pixel centre's exact value. Each record is
[u, v, r, g, b, sample_u, sample_v]: the assumed pixel-centre UV (V-up), the rendered colour, and the UV the Image Texture
receives, read from a second render of the same UV Map. Colour is rounded to six decimals and the sample UV to seven.
"""

import json
import os
import struct
import sys
import tempfile
import zlib

import bpy

argv = sys.argv[sys.argv.index("--") + 1 :]
colorspace, interpolation, extension, linked_name, out_path = argv[:5]
linked = linked_name == "linked"
out_path = os.path.abspath(out_path)
size = 32


def write_png(path, width, height, rgba):
    raw = b"".join(b"\x00" + bytes(rgba[y * width * 4 : (y + 1) * width * 4]) for y in range(height))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    with open(path, "wb") as handle:
        handle.write(
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b"")
        )


png_path = os.path.join(tempfile.gettempdir(), "alpha_oracle_%s.png" % colorspace.replace(" ", "_"))
write_png(png_path, 2, 1, [128, 40, 0, 128, 64, 0, 255, 255])

bpy.ops.wm.read_factory_settings(use_empty=True)
image = bpy.data.images.load(png_path, check_existing=False)
image.colorspace_settings.name = colorspace

mat = bpy.data.materials.new("alpha_oracle")
mat.use_nodes = True
nt = mat.node_tree
for node in list(nt.nodes):
    nt.nodes.remove(node)
tex = nt.nodes.new("ShaderNodeTexImage")
tex.image = image
tex.interpolation = interpolation
tex.extension = extension
uv = nt.nodes.new("ShaderNodeUVMap")
uv.uv_map = "Lightmap"
emission = nt.nodes.new("ShaderNodeEmission")
emission.inputs["Strength"].default_value = 1.0
output = nt.nodes.new("ShaderNodeOutputMaterial")
nt.links.new(uv.outputs["UV"], tex.inputs["Vector"])
# Two Emission nodes read the same colour, and a Mix Shader weights them by the texture's Alpha. The weights sum to one,
# so the rendered colour is the texture's Color, and the Alpha link is kept in the compiled graph. (A Mix RGB whose two
# colour inputs are the same socket is simplified away, which would hide the link.)
emission_b = nt.nodes.new("ShaderNodeEmission")
emission_b.inputs["Strength"].default_value = 1.0
if linked:
    shader_mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(tex.outputs["Alpha"], shader_mix.inputs["Fac"])
    nt.links.new(emission.outputs["Emission"], shader_mix.inputs[1])
    nt.links.new(emission_b.outputs["Emission"], shader_mix.inputs[2])
    nt.links.new(tex.outputs["Color"], emission_b.inputs["Color"])
    nt.links.new(shader_mix.outputs["Shader"], output.inputs["Surface"])
else:
    nt.links.new(emission.outputs["Emission"], output.inputs["Surface"])
nt.links.new(tex.outputs["Color"], emission.inputs["Color"])

bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, 0.0))
plane = bpy.context.active_object
mesh = plane.data
layer = mesh.uv_layers.new(name="Lightmap")
mesh.uv_layers.active = layer
for polygon in mesh.polygons:
    for loop_index in polygon.loop_indices:
        co = mesh.vertices[mesh.loops[loop_index].vertex_index].co
        layer.data[loop_index].uv = (2.0 * co.x + 0.5, 2.0 * co.y + 0.5)
mesh.materials.clear()
mesh.materials.append(mat)

camera_data = bpy.data.cameras.new("cam")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 1.0
camera = bpy.data.objects.new("cam", camera_data)
bpy.context.scene.collection.objects.link(camera)
camera.location = (0.0, 0.0, 2.0)
scene = bpy.context.scene
scene.camera = camera
render = scene.render
render.engine = "CYCLES"
render.resolution_x = size
render.resolution_y = size
render.resolution_percentage = 100
render.film_transparent = False
scene.cycles.device = "CPU"
scene.cycles.samples = 1
scene.cycles.use_denoising = False
if hasattr(scene.cycles, "pixel_filter_type"):
    scene.cycles.pixel_filter_type = "BOX"
else:
    scene.cycles.filter_type = "BOX"
scene.cycles.filter_width = 0.0001
scene.view_settings.view_transform = "Raw"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0
render.image_settings.file_format = "OPEN_EXR"
render.image_settings.color_depth = "32"
render.image_settings.exr_codec = "NONE"


def render_to(path):
    render.filepath = path
    bpy.ops.render.render(write_still=True)
    loaded = bpy.data.images.load(path, check_existing=False)
    loaded.reload()
    pixels = list(loaded.pixels)
    bpy.data.images.remove(loaded)
    return pixels


work = tempfile.gettempdir()
tag = "%s_%s_%s_%s" % (colorspace.replace(" ", "_"), interpolation, extension, linked_name)
color_pixels = render_to(os.path.join(work, "alpha_colour_%s.exr" % tag))
for link in list(emission.inputs["Color"].links):
    nt.links.remove(link)
nt.links.new(uv.outputs["UV"], emission.inputs["Color"])
for link in list(emission_b.inputs["Color"].links):
    nt.links.remove(link)
nt.links.new(uv.outputs["UV"], emission_b.inputs["Color"])
uv_pixels = render_to(os.path.join(work, "alpha_uv_%s.exr" % tag))

records = []
for y in range(size):
    v = -0.5 + 2.0 * (y + 0.5) / size
    for x in range(size):
        u = -0.5 + 2.0 * (x + 0.5) / size
        offset = (y * size + x) * 4
        records.append(
            [round(u, 6), round(v, 6)]
            + [round(color_pixels[offset + c], 6) for c in range(3)]
            + [round(uv_pixels[offset], 7), round(uv_pixels[offset + 1], 7)]
        )
with open(out_path, "w") as handle:
    json.dump(
        {
            "colorspace": colorspace,
            "interpolation": interpolation,
            "extension": extension,
            "alpha_output": linked_name,
            "size": size,
            "records": records,
        },
        handle,
    )
print("ORACLE_OK", colorspace, interpolation, extension, linked_name, len(records))
