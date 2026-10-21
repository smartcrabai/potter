"""Pixel filter oracle: thin dark lines on a white floor, rendered by Blender's Cycles (v5.2.2).

The floor is a 2 x 2 white plane lit by a white world, seen from straight above by an orthographic
camera whose 32 x 32 frame covers it exactly. Eight dark lines run each way, one pixel wide, every
four pixels, and each pair straddles a pixel boundary. The scene is symmetric under mirroring in x
and in y, so the image is the same whichever way round an image reader lays out its rows.

Cycles importance-samples its pixel filter (`intern/cycles/scene/film.cpp`, `filter_table`), so a
filter with a support wider than a pixel blurs these lines, and a box filter does not. The mean of
the whole frame is nearly the same for each filter; the per-pixel values are what differ.

The script renders `seeds` seeds of the same scene at `SAMPLES` samples per pixel. It writes the
scene (with the filter type and width set) as a `.blend`, which Potter imports, and writes per
pixel the mean and standard error of the red channel over the seeds (the floor and lines are grey).
With `seeds` 0 the script only saves the `.blend` and writes no JSON, so a test that needs just the
scene does not pay for renders.

Arguments after `--`: blend_out, json_out, kind (box | gaussian | gaussian_wide | blackman_harris),
seeds. `gaussian_wide` is a Gaussian of width 2.5, so it differs from the default-width kinds and
shows whether Potter reads `filter_width`.
"""

import json
import math
import sys

import bpy

blend_out, json_out, kind, seed_count = sys.argv[sys.argv.index("--") + 1:][:4]
SEEDS = int(seed_count)
WIDTH = 32
HEIGHT = 32
SAMPLES = 256
FRAME = 2.0
PIXEL = FRAME / WIDTH
LINE_WIDTH = PIXEL
# Line centres: symmetric about 0, one every four pixels, each straddling a pixel boundary.
LINE_CENTRES = [sign * (0.125 + 0.25 * index) for index in range(4) for sign in (-1.0, 1.0)]
FLOOR_ALBEDO = 0.8
FILTERS = {
    "box": ("BOX", 1.5),
    "gaussian": ("GAUSSIAN", 1.5),
    "gaussian_wide": ("GAUSSIAN", 2.5),
    "blackman_harris": ("BLACKMAN_HARRIS", 1.5),
}
filter_type, filter_width = FILTERS[kind]

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = SAMPLES
scene.cycles.use_denoising = False
scene.cycles.max_bounces = 4
scene.cycles.pixel_filter_type = filter_type
scene.cycles.filter_width = filter_width
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
background = world.node_tree.nodes["Background"]
background.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
background.inputs["Strength"].default_value = 1.0


def material(name, albedo):
    result = bpy.data.materials.new(name)
    result.use_nodes = True
    shader = result.node_tree.nodes["Principled BSDF"]
    shader.inputs["Base Color"].default_value = (albedo, albedo, albedo, 1.0)
    shader.inputs["Roughness"].default_value = 1.0
    shader.inputs["Specular IOR Level"].default_value = 0.0
    return result


floor_material = material("Floor", FLOOR_ALBEDO)
line_material = material("Line", 0.0)

bpy.ops.mesh.primitive_plane_add(size=FRAME, location=(0.0, 0.0, 0.0))
bpy.context.object.data.materials.append(floor_material)

for centre in LINE_CENTRES:
    for vertical in (True, False):
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        line = bpy.context.object
        # A thin bar across the whole frame, at the given centre along one axis.
        if vertical:
            line.location = (centre, 0.0, 0.0)
            line.scale = (LINE_WIDTH, FRAME * 2.0, 0.01)
        else:
            line.location = (0.0, centre, 0.0)
            line.scale = (FRAME * 2.0, LINE_WIDTH, 0.01)
        line.data.materials.append(line_material)

camera_data = bpy.data.cameras.new("Camera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = FRAME
camera = bpy.data.objects.new("Camera", camera_data)
scene.collection.objects.link(camera)
camera.location = (0.0, 0.0, 5.0)
scene.camera = camera

if 0 < SEEDS < 2:
    raise ValueError("pixel_filter.py needs 0 seeds (save only) or at least 2 seeds")

bpy.ops.wm.save_as_mainfile(filepath=blend_out)

if SEEDS > 0:
    sums = [0.0] * (WIDTH * HEIGHT)
    squares = [0.0] * (WIDTH * HEIGHT)
    for seed in range(SEEDS):
        scene.cycles.seed = seed
        scene.render.filepath = blend_out + f".{seed}.exr"
        bpy.ops.render.render(write_still=True)
        image = bpy.data.images.load(scene.render.filepath, check_existing=False)
        pixels = list(image.pixels)
        channels = image.channels
        for index in range(WIDTH * HEIGHT):
            value = pixels[index * channels]
            sums[index] += value
            squares[index] += value * value
        bpy.data.images.remove(image)

    means = [value / SEEDS for value in sums]
    # Unbiased sample standard deviation over the seeds, then the standard error of the mean.
    errors = [
        math.sqrt(max(squares[index] / SEEDS - means[index] ** 2, 0.0) * SEEDS / (SEEDS - 1)) / math.sqrt(SEEDS)
        for index in range(WIDTH * HEIGHT)
    ]

    with open(json_out, "w", encoding="utf-8") as handle:
        json.dump({"kind": kind, "seeds": SEEDS, "width": WIDTH, "height": HEIGHT,
                   "mean": means, "stderr": errors}, handle)
