"""Blender 5.2.2 oracle: Workbench Solid draws back faces unless the scene's viewport culls them.

A 2x2 plane at z = 0 (normal +Z) is framed by an orthographic camera of scale 2.2, the framing
Potter's preset `top` and `bottom` views use, and rendered in Workbench Solid from above and from
below. Each value is the fraction of pixels that differ from the corner, which is background.

Run: blender -b --factory-startup --python backface_coverage.py -- <out.json> <png_dir>
"""

import json
import math
import os
import sys

import bpy

out_json, png_dir = sys.argv[sys.argv.index("--") + 1 :][:2]

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, 0.0))
# Workbench outlines the selected object; the outline is not part of the plane's coverage.
bpy.context.view_layer.objects.active = None
bpy.ops.object.select_all(action="DESELECT")

camera_data = bpy.data.cameras.new("Camera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 2.2
camera = bpy.data.objects.new("Camera", camera_data)
scene.collection.objects.link(camera)
scene.camera = camera

scene.render.engine = "BLENDER_WORKBENCH"
scene.render.resolution_x = 64
scene.render.resolution_y = 64
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.dither_intensity = 0.0
scene.display.render_aa = "OFF"
scene.display.viewport_aa = "OFF"


def coverage(path):
    image = bpy.data.images.load(path, check_existing=False)
    width, height = image.size
    pixels = image.pixels[:]
    corner = pixels[0:3]
    lit = 0
    for index in range(width * height):
        offset = index * 4
        if any(abs(pixels[offset + k] - corner[k]) > 0.5 / 255.0 for k in range(3)):
            lit += 1
    bpy.data.images.remove(image)
    return lit / (width * height)


VIEWS = {
    "top": ((0.0, 0.0, 5.0), (0.0, 0.0, 0.0)),
    "bottom": ((0.0, 0.0, -5.0), (math.pi, 0.0, 0.0)),
}
CASES = [
    ("top", "top", False),
    ("bottom", "bottom", False),
    ("bottom_scene_culling", "bottom", True),
]
result = {}
for label, view, culling in CASES:
    scene.display.shading.show_backface_culling = culling
    location, rotation = VIEWS[view]
    camera.location = location
    camera.rotation_euler = rotation
    scene.render.filepath = os.path.join(png_dir, f"{label}.png")
    bpy.ops.render.render(write_still=True)
    result[label] = coverage(scene.render.filepath)

with open(out_json, "w") as handle:
    json.dump(result, handle, indent=2)
print("BACKFACE_COVERAGE", json.dumps(result))
