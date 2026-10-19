"""EEVEE oracle for material emission on flat and smooth realtime surfaces (Blender 5.2.2).

Builds two emissive Principled planes side by side (`FlatEmitter` with flat faces, `SmoothEmitter`
with smooth shading), no lights, a black world, and an orthographic camera that frames both. Both
Principled materials use Emission Color (0.6, 0.3, 0.1) at Emission Strength 1.5, so the emitted
radiance is colour times strength: (0.9, 0.45, 0.15). Standard view transform, look None, exposure
0 and gamma 1 keep the render scene-linear.

Arguments after `--`: blend_out, exr_out, json_out.

`blend_out` is the scene for Potter to import. The render is written to `exr_out` as 32-bit float
and reloaded, because the in-memory Render Result is empty under `--background`. `json_out` holds,
for each plane, the EEVEE pixel at its UV centre, the largest deviation of the plane's interior
pixels from colour times strength, and the interior's per-channel spread.
"""

import json
import sys

import bpy

blend_out, exr_out, json_out = sys.argv[sys.argv.index("--") + 1 :][:3]

EMISSION_COLOR = (0.6, 0.3, 0.1)
EMISSION_STRENGTH = 1.5
EXPECTED = [channel * EMISSION_STRENGTH for channel in EMISSION_COLOR]
SIZE = 64

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


def emitter_material(name):
    material = bpy.data.materials.new(name)
    principled = material.node_tree.nodes["Principled BSDF"]
    principled.inputs["Base Color"].default_value = (0.2, 0.4, 0.8, 1.0)
    principled.inputs["Emission Color"].default_value = (*EMISSION_COLOR, 1.0)
    principled.inputs["Emission Strength"].default_value = EMISSION_STRENGTH
    return material


def plane(name, x_min, x_max, smooth):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(
        [
            (x_min, -2.0, 0.0),
            (x_max, -2.0, 0.0),
            (x_max, 2.0, 0.0),
            (x_min, 2.0, 0.0),
        ],
        [],
        [(0, 1, 2, 3)],
    )
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = smooth
    mesh.materials.append(emitter_material(name + "Material"))
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)


plane("FlatEmitter", -2.0, 0.0, False)
plane("SmoothEmitter", 0.0, 2.0, True)

camera_data = bpy.data.cameras.new("OracleCamera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 4.0
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


def summarize(uv, x_range):
    centre = pixel(int(uv[0] * width), int(uv[1] * height))
    samples = [
        pixel(x, y)
        for y in range(4, height - 4)
        for x in range(x_range[0], x_range[1])
    ]
    deviation = max(
        abs(sample[channel] - EXPECTED[channel])
        for sample in samples
        for channel in range(3)
    )
    spread = [
        max(sample[channel] for sample in samples)
        - min(sample[channel] for sample in samples)
        for channel in range(3)
    ]
    return {
        "uv": uv,
        "rgba": centre,
        "interior_max_deviation": deviation,
        "interior_spread": spread,
    }


with open(json_out, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "expected_rgb": EXPECTED,
            "flat": summarize([0.25, 0.5], (4, width // 2 - 4)),
            "smooth": summarize([0.75, 0.5], (width // 2 + 4, width - 4)),
        },
        handle,
        indent=2,
    )
