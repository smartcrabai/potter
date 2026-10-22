"""EEVEE oracle for the realtime film filter (`render.filter_size`, Blender's Filter Size; 5.2.2).

A 64 px orthographic frame of 8 m (0.125 m per pixel) looks down at a white emitter. Two scenes are rendered:

- `vertical`: the emitter fills the world right of x = EDGE_X, so its edge is a column edge at pixel column EDGE_PIXEL.
- `horizontal`: the emitter fills the world below y = EDGE_Y, so its edge is a row edge at pixel row EDGE_PIXEL from the
  top.

Each FILTER_SIZES value renders once per scene, in EEVEE at `taa_render_samples = SAMPLES`. The output JSON holds, for
each size, the red channel averaged over the other axis: `profiles` (one value per column, from the vertical scene) and
`horizontal_profiles` (one value per row from the top, from the horizontal scene).

Arguments after `--`: blend_out, json_out, render_dir. `blend_out` is the vertical scene with the default Filter Size, for
Potter to import; the horizontal scene is saved beside it with `_horizontal` before the extension. The EEVEE renders go to
`render_dir`. With `render_dir` set to `0` the generator only saves both .blend files; the test then reads the committed
profiles in `eevee_film_filter.json`, which this script wrote with the renders.
"""

import json
import os
import sys

import bpy
import numpy as np

RESOLUTION = 64
ORTHO_SCALE = 8.0
EDGE_PIXEL = 32.3
EDGE_X = (EDGE_PIXEL / RESOLUTION - 0.5) * ORTHO_SCALE
EDGE_Y = (0.5 - EDGE_PIXEL / RESOLUTION) * ORTHO_SCALE
SAMPLES = 256
FILTER_SIZES = [0.01, 0.5, 1.5, 2.5]

blend_out, json_out, render_dir = sys.argv[sys.argv.index("--") + 1:][:3]
blend_out_horizontal = os.path.splitext(blend_out)[0] + "_horizontal" + os.path.splitext(blend_out)[1]
if render_dir != "0":
    os.makedirs(render_dir, exist_ok=True)

for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

scene = bpy.context.scene
world = bpy.data.worlds.new("Black")
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
scene.world = world
# Potter's realtime Beauty background reads `World.color` (`render/preview.rs` `background_color`), not the Background node
# that EEVEE renders. Both are black, so the oracle's world is black on either reading.
world.color = (0.0, 0.0, 0.0)

scene.render.engine = "BLENDER_EEVEE"
scene.eevee.taa_render_samples = SAMPLES
scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
view = scene.view_settings
view.view_transform = "Standard"
view.look = "None"
view.exposure = 0.0
view.gamma = 1.0
scene.display_settings.display_device = "sRGB"

camera_data = bpy.data.cameras.new("OracleCamera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = ORTHO_SCALE
camera = bpy.data.objects.new("OracleCamera", camera_data)
scene.collection.objects.link(camera)
camera.location = (0.0, 0.0, 6.0)
scene.camera = camera

material = bpy.data.materials.new("Emitter")
material.use_nodes = True
bsdf = material.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
bsdf.inputs["Specular IOR Level"].default_value = 0.0
bsdf.inputs["Emission Color"].default_value = (1.0, 1.0, 1.0, 1.0)
bsdf.inputs["Emission Strength"].default_value = 1.0


def build_emitter(orientation):
    """Replaces the emitter with a plane whose edge is at the pixel-edge position of `orientation`."""
    for obj in list(bpy.data.objects):
        if obj.name == "Emitter":
            bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, 0.0))
    plane = bpy.context.object
    plane.name = "Emitter"
    if orientation == "vertical":
        width = 4.5 - EDGE_X
        plane.scale = (width, 8.0, 1.0)
        plane.location = (EDGE_X + width / 2.0, 0.0, 0.0)
    else:
        # Covers y from -4.5 up to EDGE_Y, the rows below the edge.
        width = EDGE_Y + 4.5
        plane.scale = (8.0, width, 1.0)
        plane.location = (0.0, (-4.5 + EDGE_Y) / 2.0, 0.0)
    plane.data.materials.append(material)


def per_column(path):
    image = bpy.data.images.load(path, check_existing=False)
    width, height = image.size
    pixels = np.array(image.pixels[:]).reshape(height, width, 4)[:, :, 0]
    bpy.data.images.remove(image)
    return pixels.mean(axis=0).tolist()


def per_row(path):
    image = bpy.data.images.load(path, check_existing=False)
    width, height = image.size
    pixels = np.array(image.pixels[:]).reshape(height, width, 4)[:, :, 0]
    bpy.data.images.remove(image)
    # Blender stores row 0 at the bottom; Potter's rows start at the top.
    return pixels.mean(axis=1)[::-1].tolist()


build_emitter("vertical")
scene.render.filter_size = 1.5
bpy.ops.wm.save_as_mainfile(filepath=blend_out)
build_emitter("horizontal")
scene.render.filter_size = 1.5
bpy.ops.wm.save_as_mainfile(filepath=blend_out_horizontal)
if render_dir == "0":
    sys.exit(0)

result = {
    "resolution": RESOLUTION,
    "ortho_scale": ORTHO_SCALE,
    "edge_pixel": EDGE_PIXEL,
    "samples": SAMPLES,
    "profiles": {},
    "horizontal_profiles": {},
}
for orientation, key, profile in [
    ("vertical", "profiles", per_column),
    ("horizontal", "horizontal_profiles", per_row),
]:
    build_emitter(orientation)
    for size in FILTER_SIZES:
        scene.render.filter_size = size
        path = os.path.join(render_dir, f"{orientation}_{size}.exr")
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        result[key][str(size)] = profile(path)

with open(json_out, "w") as handle:
    json.dump(result, handle, indent=2)
print("WROTE", json_out)
