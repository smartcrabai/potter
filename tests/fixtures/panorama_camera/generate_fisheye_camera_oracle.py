"""Blender 5.2.2's fisheye panorama camera ray directions and out-of-circle pixels, as the oracle for Potter's fisheye
ray generation (render/raster.rs, ProjectionFrame::FisheyeEquidistant and ProjectionFrame::FisheyeEquisolid).

Regenerate from this directory with Blender 5.2.2, one thread, one run:

    Blender -b --factory-startup --threads 1 -P generate_fisheye_camera_oracle.py -- OUT.json

Three PANO cameras share one non-square 40 x 24 frame and one rotated, translated transform:

- FISHEYE_EQUIDISTANT with a 2.0 rad field of view,
- FISHEYE_EQUISOLID with a 2.5 rad field of view, a 12 mm lens and a 20 mm sensor width, AUTO sensor fit,
- FISHEYE_EQUISOLID with the same lens and field of view, a 15 mm sensor height and VERTICAL sensor fit.

The world's Background shows 1 + d / 2 for the world-space ray direction d, so a pixel whose ray exists has every
channel above 0.5, and a pixel outside the projection circle (no ray) is black. Each record is
[x, top_row, true, dx, dy, dz] for a pixel with a ray, where top_row counts from the top as Potter's pixel coordinates
do, and [x, top_row, false, null, null, null, r, g, b, a] for a pixel without one, with the stored colour and alpha.
The filter width is zero, so each pixel is sampled at its centre.
"""

import json
import math
import os
import sys
import tempfile

import bpy

argv = sys.argv[sys.argv.index("--") + 1 :]
out_path = os.path.abspath(argv[0])
work = tempfile.mkdtemp(prefix="fisheye_camera_")
WIDTH, HEIGHT = 40, 24
LOCATION = (0.3, -0.2, 0.5)
ROTATION = (1.0, 0.3, 0.6)
CAMERAS = [
    {
        "name": "equidistant",
        "panorama_type": "FISHEYE_EQUIDISTANT",
        "fisheye_fov": 2.0,
    },
    {
        "name": "equisolid_auto",
        "panorama_type": "FISHEYE_EQUISOLID",
        "fisheye_fov": 2.5,
        "fisheye_lens": 12.0,
        "sensor_width": 20.0,
        "sensor_height": 15.0,
        "sensor_fit": "AUTO",
    },
    {
        "name": "equisolid_vertical",
        "panorama_type": "FISHEYE_EQUISOLID",
        "fisheye_fov": 2.5,
        "fisheye_lens": 12.0,
        "sensor_width": 20.0,
        "sensor_height": 15.0,
        "sensor_fit": "VERTICAL",
    },
]

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
render = scene.render
render.engine = "CYCLES"
render.resolution_x = WIDTH
render.resolution_y = HEIGHT
render.resolution_percentage = 100
render.film_transparent = False
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
image_settings = render.image_settings
image_settings.file_format = "OPEN_EXR"
image_settings.color_depth = "32"
image_settings.color_mode = "RGBA"
image_settings.exr_codec = "NONE"

tree = bpy.data.worlds.new("direction")
tree.use_nodes = True
scene.world = tree
nodes = tree.node_tree.nodes
links = tree.node_tree.links
for node in list(nodes):
    nodes.remove(node)
coord = nodes.new("ShaderNodeTexCoord")
mapped = nodes.new("ShaderNodeVectorMath")
mapped.operation = "MULTIPLY_ADD"
mapped.inputs[1].default_value = (0.5, 0.5, 0.5)
mapped.inputs[2].default_value = (1.0, 1.0, 1.0)
background = nodes.new("ShaderNodeBackground")
background.inputs["Strength"].default_value = 1.0
output = nodes.new("ShaderNodeOutputWorld")
links.new(coord.outputs["Generated"], mapped.inputs[0])
links.new(mapped.outputs["Vector"], background.inputs["Color"])
links.new(background.outputs["Background"], output.inputs["Surface"])

results = []
for config in CAMERAS:
    camera_data = bpy.data.cameras.new(config["name"])
    camera_data.type = "PANO"
    camera_data.panorama_type = config["panorama_type"]
    camera_data.fisheye_fov = config["fisheye_fov"]
    if "fisheye_lens" in config:
        camera_data.fisheye_lens = config["fisheye_lens"]
        camera_data.sensor_width = config["sensor_width"]
        camera_data.sensor_height = config["sensor_height"]
        camera_data.sensor_fit = config["sensor_fit"]
    camera = bpy.data.objects.new(config["name"], camera_data)
    scene.collection.objects.link(camera)
    camera.location = LOCATION
    camera.rotation_euler = ROTATION
    scene.camera = camera
    bpy.context.view_layer.update()

    render.filepath = os.path.join(work, config["name"] + ".exr")
    bpy.ops.render.render(write_still=True)
    loaded = bpy.data.images.load(render.filepath, check_existing=False)
    pixels = list(loaded.pixels)
    bpy.data.images.remove(loaded)

    records = []
    for row in range(HEIGHT):
        for x in range(WIDTH):
            top_row = HEIGHT - 1 - row
            # Rows in the EXR are bottom-up; Potter's pixel rows count from the top.
            pixel_offset = (row * WIDTH + x) * 4
            r, g, b, a = pixels[pixel_offset : pixel_offset + 4]
            if min(r, g, b) > 0.25:
                d = [2.0 * c - 2.0 for c in (r, g, b)]
                length = math.sqrt(sum(v * v for v in d))
                records.append([x, top_row, True] + [round(v / length, 7) for v in d])
            else:
                records.append([x, top_row, False, None, None, None] + [round(c, 7) for c in (r, g, b, a)])

    matrix = [list(row_) for row_ in camera.matrix_world]
    results.append(
        {
            "name": config["name"],
            "panorama_type": config["panorama_type"],
            "fisheye_fov": config["fisheye_fov"],
            "fisheye_lens": config.get("fisheye_lens"),
            "sensor_width": config.get("sensor_width"),
            "sensor_height": config.get("sensor_height"),
            "sensor_fit": config.get("sensor_fit"),
            "camera_to_world": matrix,
            "records": records,
        }
    )
    scene.collection.objects.unlink(camera)
    bpy.data.objects.remove(camera)
    bpy.data.cameras.remove(camera_data)

with open(out_path, "w") as handle:
    json.dump({"width": WIDTH, "height": HEIGHT, "cameras": results}, handle, separators=(",", ":"))
print("ORACLE_OK fisheye", sum(len(c["records"]) for c in results))
