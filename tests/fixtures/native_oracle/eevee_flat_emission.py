"""EEVEE oracle for material emission on flat, smooth and graph-per-pixel realtime surfaces (5.2.2).

Builds three emissive Principled planes side by side under a black world with no lights, framed by
an orthographic camera of scale 6 (x and y in [-3, 3], 16 px per unit at 96 px):

- `FlatEmitter` (x in [-3, -1]): flat faces, Emission Color (0.6, 0.3, 0.1) at strength 1.5.
- `SmoothEmitter` (x in [-1, 1]): smooth shading, the same emission.
- `GraphEmitter` (x in [1, 3]): flat faces, Emission Color linked to a Vertex Color node reading
  the layer `Ink` (constant linear (0.4, 0.2, 0.6)) at strength 1.5. Potter evaluates it per pixel.

The expected radiance is colour times strength: (0.9, 0.45, 0.15), and (0.6, 0.3, 0.9) for the graph
plane. Standard view transform, look None, exposure 0 and gamma 1 keep the render scene-linear.

Arguments after `--`: blend_out, exr_out, json_out.

`blend_out` is the scene for Potter to import. The render is written to `exr_out` as 32-bit float
and reloaded, because the in-memory Render Result is empty under `--background`. `json_out` holds,
for each plane, the EEVEE pixel at its pixel centre, the largest deviation of its interior pixels
from its expected radiance, and the interior's per-channel spread.
"""

import json
import sys

import bpy

blend_out, exr_out, json_out = sys.argv[sys.argv.index("--") + 1 :][:3]

STRENGTH = 1.5
SIZE = 96
PLANE_PIXELS = 32
INK = (0.4, 0.2, 0.6)

scene = bpy.context.scene
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

# A new world's node tree starts with a 0.05 grey Background that still lights the planes.
world = bpy.data.worlds.new("Black")
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
scene.world = world

view = scene.view_settings
view.view_transform = "Standard"
view.look = "None"
view.exposure = 0.0
view.gamma = 1.0
scene.display_settings.display_device = "sRGB"


def principled(material, emission_color=None):
    tree = material.node_tree
    bsdf = tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.2, 0.4, 0.8, 1.0)
    bsdf.inputs["Emission Strength"].default_value = STRENGTH
    if emission_color is None:
        bsdf.inputs["Emission Color"].default_value = (0.6, 0.3, 0.1, 1.0)
    else:
        bsdf.inputs["Emission Color"].default_value = (*emission_color, 1.0)
    return bsdf


def plane(name, x_min, x_max, smooth, graph):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(
        [
            (x_min, -3.0, 0.0),
            (x_max, -3.0, 0.0),
            (x_max, 3.0, 0.0),
            (x_min, 3.0, 0.0),
        ],
        [],
        [(0, 1, 2, 3)],
    )
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = smooth
    material = bpy.data.materials.new(name + "Material")
    if graph:
        ink = mesh.color_attributes.new("Ink", "FLOAT_COLOR", "CORNER")
        for item in ink.data:
            item.color = (*INK, 1.0)
        bsdf = principled(material)
        vertex_color = material.node_tree.nodes.new("ShaderNodeVertexColor")
        vertex_color.layer_name = "Ink"
        material.node_tree.links.new(
            vertex_color.outputs["Color"], bsdf.inputs["Emission Color"]
        )
    else:
        principled(material)
    mesh.materials.append(material)
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)


plane("FlatEmitter", -3.0, -1.0, False, False)
plane("SmoothEmitter", -1.0, 1.0, True, False)
plane("GraphEmitter", 1.0, 3.0, False, True)

camera_data = bpy.data.cameras.new("OracleCamera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 6.0
camera = bpy.data.objects.new("OracleCamera", camera_data)
scene.collection.objects.link(camera)
camera.location = (0.0, 0.0, 6.0)
scene.camera = camera

scene.render.engine = "BLENDER_EEVEE"
scene.eevee.taa_render_samples = 16
scene.render.resolution_x = SIZE
scene.render.resolution_y = SIZE
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.render.filepath = exr_out

bpy.ops.wm.save_as_mainfile(filepath=blend_out)
bpy.ops.render.render(write_still=True)

result = bpy.data.images.load(exr_out, check_existing=False)
width, height = result.size
pixels = list(result.pixels)


def pixel(x, y):
    start = (y * width + x) * 4
    return pixels[start : start + 4]


def summarize(index, expected):
    left = index * PLANE_PIXELS
    centre_x = left + PLANE_PIXELS // 2
    centre = pixel(centre_x, height // 2)
    samples = [
        pixel(x, y)
        for y in range(4, height - 4)
        for x in range(left + 4, left + PLANE_PIXELS - 4)
    ]
    deviation = max(
        abs(sample[channel] - expected[channel])
        for sample in samples
        for channel in range(3)
    )
    spread = [
        max(sample[channel] for sample in samples)
        - min(sample[channel] for sample in samples)
        for channel in range(3)
    ]
    return {
        "uv": [(centre_x + 0.5) / width, 0.5],
        "expected_rgb": expected,
        "rgba": centre,
        "interior_max_deviation": deviation,
        "interior_spread": spread,
    }


flat_rgb = [channel * STRENGTH for channel in (0.6, 0.3, 0.1)]
graph_rgb = [channel * STRENGTH for channel in INK]
with open(json_out, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "flat": summarize(0, flat_rgb),
            "smooth": summarize(1, flat_rgb),
            "graph": summarize(2, graph_rgb),
        },
        handle,
        indent=2,
    )
