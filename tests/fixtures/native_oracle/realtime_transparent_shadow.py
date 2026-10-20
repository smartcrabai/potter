"""A transparent occluder's shadow on a diffuse floor, rendered by Blender's EEVEE (v5.2.2).

Writes the scene as a .blend (Potter imports it), renders it to a linear EXR with the EEVEE engine, and records the mean
RGB of a 9 by 9 grid over the whole frame. The grid must match `PATCH` in `tests/realtime_transparent_shadow.rs`.

The floor is a Principled diffuse of albedo 0.8 under a sun (or a point light), so a lit floor is 0.8 / pi times the sun
strength for a sun of strength 1 pointing straight down. The occluder is a horizontal plane outside the frame, one
unit above the floor, whose shadow lands inside it:

- `alpha`: a BLENDED Principled with Alpha 0.5, with `use_transparent_shadow` on (EEVEE's default). Its shadow is half
  as dense, as EEVEE's stochastic transparent shadow pass gives (`eevee_surf_shadow.bsl.hh`).
- `alpha_off`: the same material with `use_transparent_shadow` off. EEVEE casts an opaque shadow.
- `none`: no occluder, the lit floor alone.

Arguments after `--`: blend_out, image_out, json_out, light (SUN or POINT), occluder (ALPHA, ALPHA_OFF or NONE), and an
optional shift (default 0). EEVEE has no seed; its transparent-shadow noise is hashed from world position (`pcg4d(P, ...)`
in `eevee_surf_shadow.bsl.hh`), so shifting the whole scene along y by a sub-millimetre amount gives an independent noise
realisation of the same expectation. `tests/fixtures/native_oracle/measure_realtime_transparent_shadow_spread.py` runs
several shifts and records their spread.
"""

import json
import sys

import bpy

blend_out, image_out, json_out, light_kind, occluder = sys.argv[sys.argv.index("--") + 1:][:5]
args = sys.argv[sys.argv.index("--") + 1:]
SHIFT_Y = float(args[5]) if len(args) > 5 else 0.0

WIDTH = 64
HEIGHT = 64
SAMPLES = 64
ALBEDO = 0.8
OCCLUDER_ALPHA = 0.5
# Fractions of the frame: the same grid as the other native oracles.
PATCH = [0.25 + 0.5 * index / 8 for index in range(9)]

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.eevee.taa_render_samples = SAMPLES
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.render.resolution_x = WIDTH
scene.render.resolution_y = HEIGHT
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"

world = bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0

floor = bpy.data.materials.new("Floor")
floor.use_nodes = True
principled = floor.node_tree.nodes["Principled BSDF"]
principled.inputs["Base Color"].default_value = (ALBEDO, ALBEDO, ALBEDO, 1.0)
principled.inputs["Roughness"].default_value = 1.0
principled.inputs["Specular IOR Level"].default_value = 0.0
bpy.ops.mesh.primitive_plane_add(size=20.0, location=(0.0, SHIFT_Y, 0.0))
bpy.context.object.data.materials.append(floor)
# Smooth shading gives Potter corner normals, so it shades the floor per pixel. Flat faces are baked once per
# subdivided triangle (`render/scene.rs` `subdivide_realtime_triangles`), which would decide each shadow per cell.
# A flat plane has the same normals either way, so EEVEE's render is unchanged.
bpy.ops.object.shade_smooth()

if occluder != "NONE":
    glass = bpy.data.materials.new("Occluder")
    glass.use_nodes = True
    glass.surface_render_method = "BLENDED"
    glass.use_transparent_shadow = occluder == "ALPHA"
    shader = glass.node_tree.nodes["Principled BSDF"]
    shader.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
    shader.inputs["Alpha"].default_value = OCCLUDER_ALPHA
    bpy.ops.mesh.primitive_plane_add(size=0.9, location=(-3.0, SHIFT_Y, 1.0))
    bpy.context.object.data.materials.append(glass)

if light_kind == "SUN":
    light_data = bpy.data.lights.new("Sun", "SUN")
    light_data.energy = 1.0
    light_data.angle = 0.0
    light = bpy.data.objects.new("Sun", light_data)
    scene.collection.objects.link(light)
    # Sun rays travel along local -Z; this rotation sends them toward +x and down, so the shadow falls to +x.
    light.rotation_euler = (0.0, -0.7853981634, 0.0)
elif light_kind == "POINT":
    light_data = bpy.data.lights.new("Point", "POINT")
    light_data.energy = 100.0
    light_data.shadow_soft_size = 0.0
    light = bpy.data.objects.new("Point", light_data)
    scene.collection.objects.link(light)
    # Above and to the left of the occluder, so its shadow falls inside the frame toward +x.
    light.location = (-4.0, SHIFT_Y, 2.5)
else:
    raise SystemExit("unknown light " + light_kind)

camera_data = bpy.data.cameras.new("Camera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 2.0
camera = bpy.data.objects.new("Camera", camera_data)
scene.collection.objects.link(camera)
camera.location = (-1.5, SHIFT_Y, 6.0)
scene.camera = camera

bpy.ops.wm.save_as_mainfile(filepath=blend_out)

scene.render.filepath = image_out
bpy.ops.render.render(write_still=True)
image = bpy.data.images.load(image_out, check_existing=False)
width, height = image.size
channels = image.channels
pixels = list(image.pixels)
sums = [0.0, 0.0, 0.0]
count = 0
for v in PATCH:
    for u in PATCH:
        x = min(width - 1, int(u * width))
        y = min(height - 1, int(v * height))
        index = (y * width + x) * channels
        for channel in range(3):
            sums[channel] += pixels[index + channel]
        count += 1

with open(json_out, "w", encoding="utf-8") as handle:
    json.dump({"patch_mean": [value / count for value in sums], "shift_y": SHIFT_Y}, handle)
