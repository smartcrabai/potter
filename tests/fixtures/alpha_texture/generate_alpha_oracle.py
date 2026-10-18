"""Render Blender 5.2.2's Image Texture colour for a 2x1 image with straight or premultiplied alpha, as the oracle for
Potter's alpha rule.

Regenerate from this directory with Blender 5.2.2, one thread, one run per case:

    Blender -b --factory-startup --threads 1 -P generate_alpha_oracle.py -- \
        COLOURSPACE INTERPOLATION EXTENSION LINKED OUT [TEXELS] [DEPTH] [ALPHA] [FORMAT]

Arguments:
  COLOURSPACE    sRGB, Non-Color, or Linear (Blender's scene linear, Linear Rec.709).
  INTERPOLATION  Closest or Linear.
  EXTENSION      EXTEND or REPEAT.
  LINKED         linked or unlinked: whether the Image Texture's Alpha output is linked.
  OUT            the JSON file to write.
  TEXELS         base (default), floor, or half. base is the blend round-trip image, [128, 40, 0, 128] and
                 [64, 0, 255, 255]. floor puts [200, 100, 0, 200] in texel 0: c * a / 255 = 156.86, so Blender's integer
                 truncation (156) and a rounding premultiply (157) differ. half uses the float texels
                 [0.7, 0.35, 0.0, 0.6] and [0.3, 0.0, 1.0, 1.0], which Blender stores as half float and premultiplies in
                 half precision.
  DEPTH          8 (default) or 16. A 16-bit PNG uses the 16-bit texels [50000, 20000, 0, 50000] and
                 [20000, 0, 65535, 65535]; the `floor` texel set applies to 16-bit too.
  ALPHA          STRAIGHT (default) or PREMUL. PREMUL sets the image's Blender alpha mode to Premultiplied, so the file's
                 values are taken as already premultiplied.
  FORMAT         png (default), tiff (an 8-bit RGBA TIFF whose bytes this script writes, with ExtraSamples = 2, so the file
                 holds exactly the texels), exr (a 16-bit half EXR written by Blender), or exr32 (a single-layer 32-bit
                 float EXR written by Blender). tiff uses the 8-bit texels, exr the half texels, exr32 the float texels
                 [0.8, 0.4, 0.2, 0.5] and [0.3, 0.0, 1.0, 1.0]. A multilayer EXR is not generated here (see CHANGELOG).

A quad spans Blender UVs -0.5..1.5, covering both texels and both wraps. The Raw view transform and a near-zero box filter
give each pixel centre's exact value. A linked Alpha feeds the Fac of a Mix Shader whose two Emission inputs both read the
texture's Color, so the colour is unchanged but the Alpha link is kept in the compiled graph (a Mix RGB with identical
inputs is simplified away). Each record is [u, v, r, g, b, sample_u, sample_v]: the assumed pixel-centre UV (V-up), the
rendered colour, and the UV the Image Texture receives, read from a second render of the same UV Map. Colour is rounded to
six decimals and the sample UV to seven.
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
texel_set = argv[5] if len(argv) > 5 else "base"
depth = int(argv[6]) if len(argv) > 6 else 8
alpha_mode = argv[7] if len(argv) > 7 else "STRAIGHT"
container = argv[8] if len(argv) > 8 else "png"
linked = linked_name == "linked"
out_path = os.path.abspath(out_path)
size = 32

INTEGER_TEXELS = {
    ("base", 8): [128, 40, 0, 128, 64, 0, 255, 255],
    ("floor", 8): [200, 100, 0, 200, 64, 0, 255, 255],
    ("floor", 16): [50000, 20000, 0, 50000, 20000, 0, 65535, 65535],
}
HALF_TEXELS = [0.7, 0.35, 0.0, 0.6, 0.3, 0.0, 1.0, 1.0]
# Float texels with alpha 0.5 for the single-layer and multilayer EXR cases.
FLOAT_TEXELS = [0.8, 0.4, 0.2, 0.5, 0.3, 0.0, 1.0, 1.0]
work = tempfile.gettempdir()
tag = "%s_%s_%s_%s_%s_%d_%s_%s" % (
    colorspace.replace(" ", "_").replace("-", ""),
    interpolation,
    extension,
    linked_name,
    texel_set,
    depth,
    alpha_mode,
    container,
)


def write_png(path, width, height, values, depth):
    """A PNG with 8-bit or 16-bit RGBA samples (colour type 6, no interlace, no filtering)."""
    if depth == 8:
        raw = b"".join(b"\x00" + bytes(values[y * width * 4 : (y + 1) * width * 4]) for y in range(height))
        bit_depth = 8
    else:
        raw = b"".join(
            b"\x00" + b"".join(struct.pack(">H", v) for v in values[y * width * 4 : (y + 1) * width * 4])
            for y in range(height)
        )
        bit_depth = 16

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    with open(path, "wb") as handle:
        handle.write(
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, bit_depth, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b"")
        )


def write_tiff_rgba8(path, width, height, values):
    """A little-endian baseline TIFF: one uncompressed strip of 8-bit RGBA with straight alpha (ExtraSamples = 2)."""
    data = bytes(values[: width * height * 4])
    strip_offset = 8
    bits_offset = strip_offset + len(data)
    ifd_offset = bits_offset + 8
    entries = [
        (256, 3, 1, width),
        (257, 3, 1, height),
        (258, 3, 4, bits_offset),
        (259, 3, 1, 1),
        (262, 3, 1, 2),
        (273, 4, 1, strip_offset),
        (277, 3, 1, 4),
        (278, 3, 1, height),
        (279, 4, 1, len(data)),
        (338, 3, 1, 2),
    ]
    ifd = struct.pack("<H", len(entries))
    for tag, kind, count, value in entries:
        if kind == 3 and count == 1:
            field = struct.pack("<HH", value, 0)
        else:
            field = struct.pack("<I", value)
        ifd += struct.pack("<HHI", tag, kind, count) + field.ljust(4, b"\x00")
    ifd += struct.pack("<I", 0)
    header = b"II" + struct.pack("<HI", 42, ifd_offset)
    bits = struct.pack("<HHHH", 8, 8, 8, 8)
    with open(path, "wb") as handle:
        handle.write(header + data + bits + ifd)


bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.view_settings.view_transform = "Raw"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0

if container == "png":
    samples = INTEGER_TEXELS[(texel_set, depth)]
    source_path = os.path.join(work, "alpha_oracle_%s.png" % tag)
    write_png(source_path, 2, 1, samples, depth)
    image = bpy.data.images.load(source_path, check_existing=False)
elif container == "tiff":
    # A baseline uncompressed 8-bit RGBA TIFF written byte for byte, with ExtraSamples = 2 (unassociated alpha), so the
    # file holds exactly these texels and Blender's colour-managed TIFF save is not involved.
    samples = INTEGER_TEXELS[("floor", 8)] if texel_set == "floor" else INTEGER_TEXELS[("base", 8)]
    source_path = os.path.join(work, "alpha_oracle_%s.tif" % tag)
    write_tiff_rgba8(source_path, 2, 1, samples)
    image = bpy.data.images.load(source_path, check_existing=False)
elif container == "exr":
    # Blender writes a 16-bit half EXR from the float texels, rounding each to half precision.
    source_path = os.path.join(work, "alpha_oracle_%s.exr" % tag)
    staged = bpy.data.images.new("alpha_stage", width=2, height=1, alpha=True, float_buffer=True)
    staged.pixels = HALF_TEXELS
    settings = scene.render.image_settings
    settings.file_format = "OPEN_EXR"
    settings.color_depth = "16"
    settings.color_mode = "RGBA"
    settings.exr_codec = "NONE"
    staged.save_render(source_path)
    bpy.data.images.remove(staged)
    image = bpy.data.images.load(source_path, check_existing=False)
elif container in ("exr32", "exrml"):
    # Blender writes a 32-bit float EXR from the float texels: a single layer (exr32) or a multilayer file (exrml).
    source_path = os.path.join(work, "alpha_oracle_%s.exr" % tag)
    staged = bpy.data.images.new("alpha_stage", width=2, height=1, alpha=True, float_buffer=True)
    staged.pixels = FLOAT_TEXELS
    settings = scene.render.image_settings
    settings.file_format = "OPEN_EXR_MULTILAYER" if container == "exrml" else "OPEN_EXR"
    settings.color_depth = "32"
    settings.color_mode = "RGBA"
    settings.exr_codec = "NONE"
    staged.save_render(source_path)
    bpy.data.images.remove(staged)
    image = bpy.data.images.load(source_path, check_existing=False)
else:
    raise SystemExit("unknown container: %s" % container)

image.colorspace_settings.name = {"sRGB": "sRGB", "Non-Color": "Non-Color", "Linear": "Linear Rec.709"}[colorspace]
image.alpha_mode = alpha_mode

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
emission_b = nt.nodes.new("ShaderNodeEmission")
emission_b.inputs["Strength"].default_value = 1.0
output = nt.nodes.new("ShaderNodeOutputMaterial")
nt.links.new(uv.outputs["UV"], tex.inputs["Vector"])
nt.links.new(tex.outputs["Color"], emission.inputs["Color"])
nt.links.new(tex.outputs["Color"], emission_b.inputs["Color"])
if linked:
    shader_mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(tex.outputs["Alpha"], shader_mix.inputs["Fac"])
    nt.links.new(emission.outputs["Emission"], shader_mix.inputs[1])
    nt.links.new(emission_b.outputs["Emission"], shader_mix.inputs[2])
    nt.links.new(shader_mix.outputs["Shader"], output.inputs["Surface"])
else:
    nt.links.new(emission.outputs["Emission"], output.inputs["Surface"])

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


color_pixels = render_to(os.path.join(work, "alpha_colour_%s.exr" % tag))
for link in list(emission.inputs["Color"].links):
    nt.links.remove(link)
for link in list(emission_b.inputs["Color"].links):
    nt.links.remove(link)
nt.links.new(uv.outputs["UV"], emission.inputs["Color"])
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
            "texels": texel_set,
            "depth": depth,
            "alpha_mode": alpha_mode,
            "container": container,
            "size": size,
            "records": records,
        },
        handle,
    )
print("ORACLE_OK", colorspace, interpolation, extension, linked_name, texel_set, depth, alpha_mode, container, len(records))
