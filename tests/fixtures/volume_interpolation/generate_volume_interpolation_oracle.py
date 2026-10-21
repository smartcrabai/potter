"""Blender 5.2.2 single-scatter oracle for a voxel grid under LINEAR and CUBIC volume interpolation.

The grid is 16 x 16 x 16 nodes at index (i, j, k), with value v = 1 where h(i, j, k) >= 512 and 0 otherwise, where
h = ((i * 73856093) ^ (j * 19349663) ^ (k * 83492791)) & 1023: a hashed binary grid, whose sharp edges make LINEAR and
CUBIC differ far more than a smooth noise field does. It is written to a VDB named "density" (background 0, every node active)
whose transform carries the voxel size 1/16 and the origin (-0.5, -0.5, -0.5), so index (i, j, k) sits at world origin +
(i, j, k) / 16, the same node convention as Potter's inline grids. The volume object stays at identity: a uniform cube of
this grid transmits 0.201 with the VDB transform alone and 0.028 with an object scale of 1/16 on top of it (the padded
model gives 0.164), so the object scale is not Potter's layout and is not used.
The material is a Volume Scatter (Color 1,
Anisotropy 0) with Density = DENSITY_SCALE * Volume Info density. At a bounce budget of 0 Blender renders exact single
scatter at any density (`volume_bounces = max_bounces = 0`, verified against physics for a uniform cube), so the density
is raised for signal: at 0.2 the LINEAR-to-CUBIC difference is about 1 %, below the seed noise, and at 2.0 it is about 12 %.
A point light of 100 W at (1.6, 0.6, 0.9), an orthographic camera at (0, 0, 3) with ortho scale 1, so the 4 x 4 frame is the
unit cube, a black world, and a one-pixel BOX filter. Every pixel is averaged.

Pixels are written top-down, the row order of the Potter test's EXR reader (`position.y()` from the top). Blender's
`image.pixels` is bottom-up, so Blender row r is top row SIZE - 1 - r, the same convention as the panorama oracle
(`generate_render_world_panorama_oracle.py`: the EXR row is HEIGHT - 1 - top_row).

Noise derivation. Each case is rendered at SEEDS, and the oracle image is the per-pixel mean over them. The per-render noise,
relative to the mean of the image, is sigma = sqrt(mean over seed pairs of mean_pixels((a - b)^2) / 2) / mean(image):
the RMS of one render about its expectation, since two independent renders differ with variance 2 sigma^2. The
mean of K seeds has noise sigma / sqrt(K). The image the test compares with is that mean, so its noise is
sigma_blender = sigma / sqrt(K). For LINEAR against CUBIC, delta = rms(linear - cubic) / mean(cubic) with the
means. The signal, the part of delta that is not noise, is
signal = sqrt(max(0, delta^2 - sigma_blender_linear^2 - sigma_blender_cubic^2)).

The test derives its tolerances from these. Each parity tolerance is three noise floors: Potter's single-render frame-mean
spread (the test's POTTER_SD constants, measured over four seeds at its sample count) combined with Blender's seed standard
error (sigma_blender above, for the mean of K seeds). A Potter render matches Blender's when its frame mean is within that
tolerance, for LINEAR and for CUBIC separately. The test does not compare the difference between the two modes against
Blender's: at Potter's render budget that difference sits inside the noise, so the deterministic cubic and edge checks in
`geom/volume.rs` carry that discrimination.

Regenerate from this directory with Blender 5.2.2 (openvdb must be importable), one thread:

    Blender -b --factory-startup --threads 1 -P generate_volume_interpolation_oracle.py -- OUT.json
"""

import json
import math
import os
import sys
import tempfile

import bpy
import openvdb

argv = sys.argv[sys.argv.index("--") + 1 :]
out_path = os.path.abspath(argv[0])
work = tempfile.mkdtemp(prefix="volume_interpolation_")
SIZE = 4
SAMPLES = 8192
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)
ENERGY = 100.0
POSITION = (1.6, 0.6, 0.9)
DENSITY_SCALE = 2.0
N = 16
ORIGIN = -0.5
VOXEL = 1.0 / N


def hashed(i, j, k):
    return ((i * 73856093) ^ (j * 19349663) ^ (k * 83492791)) & 1023


def node_value(i, j, k):
    return 1.0 if hashed(i, j, k) >= 512 else 0.0


def write_vdb(path):
    grid = openvdb.FloatGrid()
    grid.name = "density"
    grid.transform = openvdb.createLinearTransform(VOXEL)
    grid.transform.postTranslate((ORIGIN, ORIGIN, ORIGIN))
    accessor = grid.getAccessor()
    for i in range(N):
        for j in range(N):
            for k in range(N):
                accessor.setValueOn((i, j, k), node_value(i, j, k))
    openvdb.write(path, grids=[grid])


def reset(interpolation, seed):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    render = scene.render
    render.engine = "CYCLES"
    render.resolution_x = SIZE
    render.resolution_y = SIZE
    render.resolution_percentage = 100
    render.film_transparent = False
    scene.cycles.device = "CPU"
    scene.cycles.samples = SAMPLES
    # Potter integrates each pixel uniformly over its footprint (a one-pixel box). Cycles' BOX filter of width 1 px is
    # that box. The frame spans the unit cube, so every pixel footprint lies inside it and the default 1.5 px filter's
    # outside-cube zeros cannot enter a pixel.
    if hasattr(scene.cycles, "pixel_filter_type"):
        scene.cycles.pixel_filter_type = "BOX"
    else:
        scene.cycles.filter_type = "BOX"
    scene.cycles.filter_width = 1.0
    scene.cycles.seed = seed
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = 0
    scene.cycles.volume_bounces = 0
    scene.cycles.transparent_max_bounces = 64
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    image = render.image_settings
    image.file_format = "OPEN_EXR"
    image.color_depth = "32"
    image.color_mode = "RGBA"
    image.exr_codec = "NONE"
    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    scene.world = world
    background = world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    background.inputs["Strength"].default_value = 0.0
    camera_data = bpy.data.cameras.new("camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 1.0
    camera = bpy.data.objects.new("camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0.0, 0.0, 3.0)
    camera.rotation_euler = (0.0, 0.0, 0.0)
    scene.camera = camera
    light_data = bpy.data.lights.new("point", "POINT")
    light_data.energy = ENERGY
    light_data.shadow_soft_size = 0.0
    light = bpy.data.objects.new("point", light_data)
    scene.collection.objects.link(light)
    light.location = POSITION
    return scene


def voxel_volume(vdb_path, interpolation):
    bpy.ops.object.volume_import(filepath=vdb_path)
    volume = bpy.context.active_object
    material = bpy.data.materials.new("voxels")
    material.use_nodes = True
    material.cycles.volume_interpolation = interpolation
    tree = material.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    scatter = tree.nodes.new("ShaderNodeVolumeScatter")
    scatter.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    scatter.inputs["Anisotropy"].default_value = 0.0
    info = tree.nodes.new("ShaderNodeVolumeInfo")
    scale = tree.nodes.new("ShaderNodeMath")
    scale.operation = "MULTIPLY"
    scale.inputs[1].default_value = DENSITY_SCALE
    tree.links.new(info.outputs["Density"], scale.inputs[0])
    tree.links.new(scale.outputs["Value"], scatter.inputs["Density"])
    tree.links.new(scatter.outputs["Volume"], output.inputs["Volume"])
    volume.data.materials.append(material)


def render_red(name):
    """The red channel, top-down: row y of the output is Blender's bottom-up row SIZE - 1 - y."""
    path = os.path.join(work, name + ".exr")
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    loaded = bpy.data.images.load(path, check_existing=False)
    pixels = list(loaded.pixels)
    bpy.data.images.remove(loaded)
    return [
        round(pixels[((SIZE - 1 - y) * SIZE + x) * 4], 6)
        for y in range(SIZE)
        for x in range(SIZE)
    ]


def mean(values):
    return sum(values) / len(values)


def relative_noise(seed_images):
    """sigma: the RMS of one render about its expectation, relative to the image mean."""
    pairs = [(a, b) for index, a in enumerate(seed_images) for b in seed_images[index + 1 :]]
    squares = [
        mean([(x - y) ** 2 for x, y in zip(a, b)]) / 2.0 for a, b in pairs
    ]
    return math.sqrt(mean(squares)) / mean(seed_images[0])


vdb_path = os.path.join(work, "density.vdb")
write_vdb(vdb_path)
images = {"linear_noise": [], "cubic_noise": []}
for name, interpolation in (("linear_noise", "LINEAR"), ("cubic_noise", "CUBIC")):
    for seed in SEEDS:
        reset(interpolation, seed)
        voxel_volume(vdb_path, interpolation)
        images[name].append(render_red(f"{name}_{seed}"))
        print("ORACLE_RENDER", name, seed)

sigma_single = {name: relative_noise(seeds) for name, seeds in images.items()}
averaged = {
    name: [mean([seed[index] for seed in seeds]) for index in range(SIZE * SIZE)]
    for name, seeds in images.items()
}
sigma = {name: sigma_single[name] / math.sqrt(len(SEEDS)) for name in images}
linear, cubic = averaged["linear_noise"], averaged["cubic_noise"]
delta = math.sqrt(mean([(x - y) ** 2 for x, y in zip(linear, cubic)])) / mean(cubic)
signal = math.sqrt(max(0.0, delta**2 - sigma["linear_noise"] ** 2 - sigma["cubic_noise"] ** 2))
records = {
    name: {
        "mean": round(mean(averaged[name]), 6),
        "pixels": [round(value, 6) for value in averaged[name]],
        "seed_pixels": images[name],
    }
    for name in images
}
noise = {
    "seeds": list(SEEDS),
    "sigma_single_render": sigma_single,
    "sigma_blender": sigma,
    "linear_vs_cubic_delta": delta,
    "linear_vs_cubic_signal": signal,
}
print("ORACLE_NOISE", json.dumps(noise))

with open(out_path, "w") as handle:
    json.dump(
        {
            "blender": bpy.app.version_string,
            "samples": SAMPLES,
            "energy": ENERGY,
            "position": list(POSITION),
            "density_scale": DENSITY_SCALE,
            "image_size": SIZE,
            "row_order": "top_down",
            "grid": {"dims": [N, N, N], "voxel_size": VOXEL, "origin": [ORIGIN] * 3},
            "noise": noise,
            "cases": records,
        },
        handle,
        indent=1,
    )
print("ORACLE_OK volume_interpolation")
