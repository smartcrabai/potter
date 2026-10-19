"""Blender 5.2.2 oracle: which side of a plane each renderer draws.

A 2x2 plane at z = 0 (normal +Z) is framed by an orthographic camera of scale 2.2 at z = +5 (top)
or z = -5 (bottom, rotated to look up), the framing Potter's `top` and `bottom` views use. Each
case renders one PNG and records the fraction of pixels that differ from the corner, which is
background.

- Workbench (Solid): the scene's `show_backface_culling` decides. Off draws the back face; on
  hides it.
- EEVEE (Beauty): the material's `use_backface_culling` decides. The material is diffuse white and
  a sun above the plane (identity rotation, shining down -Z) lights its front face; the back face
  is unlit, so a drawn back face covers pixels that differ from the background.

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
plane = bpy.context.active_object
# Workbench outlines the selected object; the outline is not part of the plane's coverage.
bpy.context.view_layer.objects.active = None
bpy.ops.object.select_all(action="DESELECT")

material = bpy.data.materials.new("Single")
material.use_nodes = True
principled = material.node_tree.nodes["Principled BSDF"]
principled.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
plane.data.materials.append(material)

sun_data = bpy.data.lights.new("Sun", "SUN")
sun_data.energy = 3.0
sun = bpy.data.objects.new("Sun", sun_data)
scene.collection.objects.link(sun)
sun.location = (0.0, 0.0, 5.0)
sun.rotation_euler = (0.0, 0.0, 0.0)

camera_data = bpy.data.cameras.new("Camera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 2.2
camera = bpy.data.objects.new("Camera", camera_data)
scene.collection.objects.link(camera)
scene.camera = camera

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
# label, view, engine, scene Workbench culling, material culling
CASES = [
    ("top", "top", "BLENDER_WORKBENCH", False, False),
    ("bottom", "bottom", "BLENDER_WORKBENCH", False, False),
    ("bottom_scene_culling", "bottom", "BLENDER_WORKBENCH", True, False),
    ("eevee_bottom_material_culling", "bottom", "BLENDER_EEVEE", False, True),
]
result = {}
for label, view, engine, scene_culling, material_culling in CASES:
    scene.render.engine = engine
    scene.display.shading.show_backface_culling = scene_culling
    material.use_backface_culling = material_culling
    location, rotation = VIEWS[view]
    camera.location = location
    camera.rotation_euler = rotation
    scene.render.filepath = os.path.join(png_dir, f"{label}.png")
    bpy.ops.render.render(write_still=True)
    result[label] = coverage(scene.render.filepath)

with open(out_json, "w") as handle:
    json.dump(result, handle, indent=2)
print("BACKFACE_COVERAGE", json.dumps(result))
