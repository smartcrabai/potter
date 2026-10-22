"""EEVEE oracle for camera-visible light shapes in the realtime Beauty image (Blender 5.2.2).

A black world and an orthographic camera of 8 m (64 px, 0.125 m per pixel) look down at two lights with
`visible_camera` on, and nothing else in the scene:

- `AreaLight`: a square area light of side 2 m at x = -1.5 m, rotated to face the camera (EEVEE discards the back of an
  area light's shape, `eevee_light_shape_display.bsl.hh`).
- `PointLight`: a point light of radius 1 m at x = +1.5 m.

EEVEE draws each emitter as its radiance (`light.color` times `1 / (π · area)`, `shape_display_light_radiance_get`).
The output JSON holds the mean red value over a 7 x 7 pixel window at each light's centre and over a background window,
for a fixed scene and no sampling noise beyond EEVEE's own. Pass `-` for `render_exr` to save the `.blend` without
rendering; test expectations come from the committed macOS f32le capture.

Arguments after `--`: blend_out, json_out, render_exr. `blend_out` is the scene for Potter to import.
"""

import json
import math
import sys

import bpy
import numpy as np

RESOLUTION = 64
ORTHO_SCALE = 8.0
AREA_SIZE = 2.0
AREA_CENTRE_X = -1.5
POINT_RADIUS = 1.0
POINT_CENTRE_X = 1.5
AREA_ENERGY = 10.0
POINT_ENERGY = 5.0
WINDOW = 3

blend_out, json_out, render_exr = sys.argv[sys.argv.index("--") + 1:][:3]

for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

scene = bpy.context.scene
world = bpy.data.worlds.new("Black")
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
scene.world = world

scene.render.engine = "BLENDER_EEVEE"
scene.eevee.taa_render_samples = 64
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
camera.location = (0.0, 0.0, 10.0)
scene.camera = camera

area_data = bpy.data.lights.new("AreaLight", "AREA")
area_data.shape = "SQUARE"
area_data.size = AREA_SIZE
area_data.energy = AREA_ENERGY
area_data.color = (1.0, 1.0, 1.0)
area = bpy.data.objects.new("AreaLight", area_data)
scene.collection.objects.link(area)
area.location = (AREA_CENTRE_X, 0.0, 0.0)
area.rotation_euler = (math.pi, 0.0, 0.0)
area.visible_camera = True

point_data = bpy.data.lights.new("PointLight", "POINT")
point_data.shadow_soft_size = POINT_RADIUS
point_data.energy = POINT_ENERGY
point_data.color = (1.0, 1.0, 1.0)
point = bpy.data.objects.new("PointLight", point_data)
scene.collection.objects.link(point)
point.location = (POINT_CENTRE_X, 0.0, 0.0)
point.visible_camera = True

bpy.ops.wm.save_as_mainfile(filepath=blend_out)

if render_exr == "-":
    print("SAVED", blend_out, "(EEVEE render skipped)")
    sys.exit(0)

scene.render.filepath = render_exr
bpy.ops.render.render(write_still=True)

image = bpy.data.images.load(render_exr, check_existing=False)
width, height = image.size
pixels = np.array(image.pixels[:]).reshape(height, width, 4)[:, :, 0]
bpy.data.images.remove(image)


def window_mean(column, row):
    return float(pixels[row - WINDOW:row + WINDOW + 1, column - WINDOW:column + WINDOW + 1].mean())


def world_to_column(x):
    return int(math.floor((x / ORTHO_SCALE + 0.5) * RESOLUTION))


centre_row = RESOLUTION // 2
result = {
    "resolution": RESOLUTION,
    "ortho_scale": ORTHO_SCALE,
    "area_centre_column": world_to_column(AREA_CENTRE_X),
    "point_centre_column": world_to_column(POINT_CENTRE_X),
    "window": WINDOW,
    "area_mean_red": window_mean(world_to_column(AREA_CENTRE_X), centre_row),
    "point_mean_red": window_mean(world_to_column(POINT_CENTRE_X), centre_row),
    "background_mean_red": window_mean(RESOLUTION // 4, RESOLUTION // 4),
}
with open(json_out, "w") as handle:
    json.dump(result, handle, indent=2)
print("WROTE", json_out, json.dumps(result))
