"""EEVEE oracle for glass over a coloured, non-uniform world, with default EEVEE settings (5.2.2).

The world is an Environment Texture (Linear) of an equirectangular EXR written here. Its colour depends only on
elevation, `z`, so the texel row alone decides it and the horizontal mapping does not matter:
    t = (z + 1) / 2,  colour = (0.25 + 0.5 t, 0.5, 0.75 - 0.5 t).
The top of the world is therefore red-heavy and the bottom blue-heavy. The camera looks down -Z at three glass spheres
(IOR 1.45, Transmission 1, no lights) in a row:
- `clear`: Base Color white, Roughness 0. Its centre reflects the top and refracts the bottom of the world.
- `tinted`: Base Color (0.9, 0.5, 0.2), Roughness 0, so the transmission tint is the square root of the base.
- `rough`: Base Color white, Roughness 0.5, so the probe is blurred.
Off-centre pixels see the world at tilted directions, so the refracted and reflected probe directions decide them.

The JSON lists the pixels whose centres lie within 0.75 of each sphere's radius, with their EEVEE RGB, the sphere centre
and radius in pixels, and the world function, so the Python mirror can evaluate the same model.

Arguments after `--`: blend_out, exr_out, json_out, env_exr_out.
"""

import json
import math
import sys

import bpy

blend_out, exr_out, json_out, env_out = sys.argv[sys.argv.index("--") + 1 :][:4]

SIZE = 96
ORTHO_SCALE = 7.6
ENV_WIDTH = 8
ENV_HEIGHT = 128
IOR = 1.45
INTERIOR = 0.75
SPHERES = [
    ("clear", -2.4, (1.0, 1.0, 1.0, 1.0), 0.0),
    ("tinted", 0.0, (0.9, 0.5, 0.2, 1.0), 0.0),
    ("rough", 2.4, (1.0, 1.0, 1.0, 1.0), 0.5),
]


def world_colour(z):
    t = (z + 1.0) / 2.0
    return (0.25 + 0.5 * t, 0.5, 0.75 - 0.5 * t)


scene = bpy.context.scene
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

# The equirectangular environment: row v = (y + 0.5) / H is elevation (v - 0.5) * pi, so z = sin(elevation).
env = bpy.data.images.new("OracleEnv", ENV_WIDTH, ENV_HEIGHT, alpha=False, float_buffer=True)
pixels = []
for row in range(ENV_HEIGHT):
    elevation = ((row + 0.5) / ENV_HEIGHT - 0.5) * math.pi
    r, g, b = world_colour(math.sin(elevation))
    for _ in range(ENV_WIDTH):
        pixels.extend((r, g, b, 1.0))
env.pixels.foreach_set(pixels)
env.filepath_raw = env_out
env.file_format = "OPEN_EXR"
env.save()

world = bpy.data.worlds.new("Oracle")
world.use_nodes = True
nodes = world.node_tree.nodes
links = world.node_tree.links
tex = nodes.new("ShaderNodeTexEnvironment")
tex.image = bpy.data.images.load(env_out, check_existing=False)
tex.interpolation = "Linear"
links.new(tex.outputs["Color"], nodes["Background"].inputs["Color"])
nodes["Background"].inputs["Strength"].default_value = 1.0
scene.world = world

view = scene.view_settings
view.view_transform = "Standard"
view.look = "None"
view.exposure = 0.0
view.gamma = 1.0
scene.display_settings.display_device = "sRGB"


def glass(name, base, roughness):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = base
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["IOR"].default_value = IOR
    bsdf.inputs["Transmission Weight"].default_value = 1.0
    return material


for name, x, base, roughness in SPHERES:
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=64, ring_count=32, location=(x, 0.0, 0.0))
    sphere = bpy.context.object
    sphere.name = name
    bpy.ops.object.shade_smooth()
    sphere.data.materials.append(glass(f"Glass_{name}", base, roughness))

camera_data = bpy.data.cameras.new("OracleCamera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = ORTHO_SCALE
camera = bpy.data.objects.new("OracleCamera", camera_data)
scene.collection.objects.link(camera)
camera.location = (0.0, 0.0, 10.0)
scene.camera = camera

scene.render.engine = "BLENDER_EEVEE"
scene.eevee.taa_render_samples = 64
scene.render.resolution_x = SIZE
scene.render.resolution_y = SIZE
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.render.filepath = exr_out

bpy.ops.wm.save_as_mainfile(filepath=blend_out)
# `-` as the EXR path saves the scene and stops: the tests import the .blend and compare with the committed EEVEE render.
if exr_out == "-":
    sys.exit(0)
bpy.ops.render.render(write_still=True)

result = bpy.data.images.load(exr_out, check_existing=False)
width, height = result.size
pixel_values = list(result.pixels)
scale = width / ORTHO_SCALE


def pixel(x, y):
    start = (y * width + x) * 4
    return pixel_values[start : start + 3]


spheres = {}
for name, x, _, roughness in SPHERES:
    # Image x grows with world X; the centre row is the sphere's centre line. Rows are Blender's (bottom first).
    centre_x = (x + ORTHO_SCALE / 2.0) * scale
    centre_y = height / 2.0
    radius = scale
    samples = []
    for row in range(height):
        for column in range(width):
            dx = (column + 0.5 - centre_x) / radius
            dy = (row + 0.5 - centre_y) / radius
            if dx * dx + dy * dy > INTERIOR * INTERIOR:
                continue
            samples.append(
                {
                    "uv": [(column + 0.5) / width, (row + 0.5) / height],
                    "rgb": pixel(column, row),
                    "x_offset": dx,
                    "y_offset": dy,
                }
            )
    spheres[name] = {"roughness": roughness, "samples": samples}

with open(json_out, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "size": SIZE,
            "ortho_scale": ORTHO_SCALE,
            "ior": IOR,
            "world": "t=(z+1)/2, colour (0.25+0.5t, 0.5, 0.75-0.5t) by elevation z",
            "spheres": spheres,
        },
        handle,
    )
