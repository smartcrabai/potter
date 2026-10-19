"""Blender 5.2.2's Image Texture colour across UDIM tiles at exact and near-edge UVs, as the oracle for Potter's tile lookup.

Regenerate from this directory with Blender 5.2.2, one thread, one run per case:

    Blender -b --factory-startup --threads 1 -P generate_udim_edge_oracle.py -- \
        INTERPOLATION EXTENSION OUT.json

INTERPOLATION is Closest or Linear. EXTENSION is REPEAT, EXTEND, CLIP, or MIRROR.

The image is a tiled (UDIM) image with two tiles, 1001 and 1002, each a 2 x 2 float texture whose texel at row y (bottom-up)
and column x holds (x / 2, y / 2, b), with b = 0.25 for 1001 and 0.75 for 1002. Tile 1003 and the row above tile 1001 are
absent. The scene is a 16 x 16 frame covered by a 16 x 16 plane with UV = (0.125 X + 1.1875, 0.125 Y + 0.1875), so the
pixel-centre UVs run from u = 0.25 to 2.125 (across tile 1001, the tile-1002 boundary at exactly u = 1.0, and the missing
tile 1003) and from v = -0.75 to 1.125 (negative rows and the missing row above). The Image Texture input is snapped to
eighths (Round(8 x) / 8, exact). Emission reads the texture colour; a second render reads the snapped UV.

Each record is [u, v, r, g, b, sample_u, sample_v]. Tile images are stored with Blender's bottom-up rows.
"""

import json
import os
import sys
import tempfile

import bpy

argv = sys.argv[sys.argv.index("--") + 1 :]
interpolation, extension, out_path = argv[:3]
out_path = os.path.abspath(out_path)
work = tempfile.mkdtemp(prefix="udim_oracle_")
SIZE = 16
TILE_TEXELS = 2
TILES = {1001: 0.25, 1002: 0.75}


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


def tile_texels(blue):
    values = []
    for y in range(TILE_TEXELS):
        for x in range(TILE_TEXELS):
            values.extend([x / TILE_TEXELS, y / TILE_TEXELS, blue, 1.0])
    return values


# Write each tile as a file, then load the set as one tiled image (the UDIM filename pattern picks up both tiles).
folder = os.path.join(work, "tiles")
os.makedirs(folder, exist_ok=True)
for number, blue in TILES.items():
    tile = bpy.data.images.new("tile%d" % number, TILE_TEXELS, TILE_TEXELS, alpha=False, float_buffer=True)
    tile.colorspace_settings.name = "Non-Color"
    tile.pixels.foreach_set(tile_texels(blue))
    tile.filepath_raw = os.path.join(folder, "edge.%d.exr" % number)
    tile.file_format = "OPEN_EXR"
    tile.save()
    bpy.data.images.remove(tile)

scene = reset_scene()
texture = bpy.data.images.load(os.path.join(folder, "edge.1001.exr"), check_existing=False)
texture.source = "TILED"
texture.filepath = os.path.join(folder, "edge.<UDIM>.exr")
texture.reload()
assert sorted(t.number for t in texture.tiles) == [1001, 1002], [t.number for t in texture.tiles]
texture.colorspace_settings.name = "Non-Color"

bpy.ops.mesh.primitive_plane_add(size=float(SIZE), location=(0.0, 0.0, 0.0))
plane = bpy.context.active_object
plane.name = "Plane"
mesh = plane.data
layer = mesh.uv_layers.get("UVMap") or mesh.uv_layers.new(name="UVMap")
mesh.uv_layers.active = layer
for polygon in mesh.polygons:
    for loop_index in polygon.loop_indices:
        co = mesh.vertices[mesh.loops[loop_index].vertex_index].co
        layer.data[loop_index].uv = (0.125 * co.x + 1.1875, 0.125 * co.y + 0.1875)

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
for link in list(tree.links):
    if link.from_node == tex and link.to_node == emission:
        tree.links.remove(link)
tree.links.new(combined.outputs["Vector"], emission.inputs["Color"])
uv_pixels = render_to(os.path.join(work, "uv.exr"))

records = []
for y in range(SIZE):
    v = 0.125 * (y - 6)
    for x in range(SIZE):
        u = 0.125 * (x - 6) + 1.0
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

tile_grids = {
    str(number): {
        "width": TILE_TEXELS,
        "height": TILE_TEXELS,
        "texels_blender_rows": [
            [x / TILE_TEXELS, y / TILE_TEXELS, blue, 1.0]
            for y in range(TILE_TEXELS)
            for x in range(TILE_TEXELS)
        ],
    }
    for number, blue in TILES.items()
}
with open(out_path, "w") as handle:
    json.dump(
        {
            "kind": "udim",
            "colorspace": "Non-Color",
            "interpolation": interpolation,
            "extension": extension,
            "tiles": tile_grids,
            "records": records,
        },
        handle,
        indent=1,
    )
print("ORACLE_OK udim", interpolation, extension, len(records))
