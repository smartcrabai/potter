"""Blender 5.2.2 Cycles references for tests/volume_spectral_extinction.rs.

This generator writes the committed 4 x 4, 32-bit linear EXR capture
`volume_absorption_cycles.exr` into the output directory. It uses a 17^3 unit-density VDB at 1/16 spacing with its
node origin at (-0.5, -0.5, -0.5), an orthographic camera at (0, 0, 3) with ortho scale 1, a one-pixel BOX filter,
Raw view transform, CPU Cycles, 65536 samples, no denoising, and Cycles seed 0.

- `volume_absorption_cycles.exr`: Volume Absorption, Color (0.1, 0.4, 0.9), Density 1, viewed through a white world of
  strength 1. This is the per-channel camera transmittance.

The committed EXR frame mean (linear RGB, all 16 pixels) is absorption
`(0.3803185, 0.5260214, 0.8986703)`.

The native Cycles anchors are `intern/cycles/kernel/svm/closure.h:svm_node_closure_volume`
(per-channel absorption and scatter closure coefficients),
`intern/cycles/kernel/closure/volume.h:volume_phase_sample`, and
`intern/cycles/kernel/integrator/shade_volume.h:volume_sample_indirect_scatter`,
`integrate_volume_direct_light`, and `volume_direct_scatter_mis` (volume event sampling,
next-event estimation, and the distance/equiangular MIS pair).
The realtime absorption check uses the same channel transmittance; Blender EEVEE accumulates per-channel froxel extinction as `exp(-extinction * step_len)` (`source/blender/draw/engines/eevee/shaders/eevee_volume.bsl.hh:335-355`).

Regenerate with Blender 5.2.2, one thread, from the repository root:

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --threads 1 \
      -P tests/fixtures/volume_spectral_extinction/generate_volume_spectral_extinction.py -- \
      tests/fixtures/volume_spectral_extinction

The multi-seed Principled Volume captures are written to `volume_spectral_extinction.json` with `--new-only`; that
mode never renders or overwrites the committed absorption EXR:

    Blender -b --factory-startup --threads 1 \
      -P tests/fixtures/volume_spectral_extinction/generate_volume_spectral_extinction.py -- \
      --new-only tests/fixtures/volume_spectral_extinction

The new oracles use a hashed 17×17×49 density VDB spanning 3 m along Z, a 4 x 4 Raw EXR frame, and Cycles CPU at
16384 samples and seeds 0–3.
"""

import json
import math
import os
import shutil
import sys
import tempfile

import bpy
import openvdb

SIZE = 4
N = 17
HASHED_Z = 49
SEED = 0
CYCLES_SAMPLES = 65536
ABSORPTION_COLOR = (0.1, 0.4, 0.9, 1.0)

argv = sys.argv[sys.argv.index("--") + 1 :]
new_only = bool(argv and argv[0] == "--new-only")
if new_only:
    argv = argv[1:]
if len(argv) != 1:
    raise SystemExit("expected [--new-only] and an output directory after --")
out_dir = os.path.abspath(argv[0])
os.makedirs(out_dir, exist_ok=True)
work = tempfile.mkdtemp(prefix="volume_spectral_extinction_")
vdb_path = os.path.join(work, "unit_density.vdb")

if not new_only:
    grid = openvdb.FloatGrid()
    grid.name = "density"
    grid.transform = openvdb.createLinearTransform(1.0 / 16.0)
    grid.transform.postTranslate((-0.5, -0.5, -0.5))
    accessor = grid.getAccessor()
    for z in range(N):
        for y in range(N):
            for x in range(N):
                accessor.setValueOn((x, y, z), 1.0)
    openvdb.write(vdb_path, grids=[grid])


def setup_scene(name, world_color, samples):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    render = scene.render
    render.engine = "CYCLES"
    render.resolution_x = SIZE
    render.resolution_y = SIZE
    render.resolution_percentage = 100
    render.film_transparent = False
    render.image_settings.file_format = "OPEN_EXR"
    render.image_settings.color_depth = "32"
    render.image_settings.color_mode = "RGBA"
    render.image_settings.exr_codec = "NONE"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.seed = SEED
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = 0
    scene.cycles.volume_bounces = 0
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 1.0
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    world = bpy.data.worlds.new(name + "World")
    world.use_nodes = True
    scene.world = world
    background = world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (*world_color, 1.0)
    background.inputs["Strength"].default_value = 1.0 if any(world_color) else 0.0

    camera_data = bpy.data.cameras.new(name + "Camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 1.0
    camera = bpy.data.objects.new(name + "Camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0.0, 0.0, 3.0)
    camera.rotation_euler = (0.0, 0.0, 0.0)
    scene.camera = camera

    bpy.ops.object.volume_import(filepath=vdb_path)
    volume = bpy.context.active_object
    return scene, volume


def material_with_volume(volume, node_type, color, density):
    material = bpy.data.materials.new(node_type + "Material")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    closure = tree.nodes.new(node_type)
    closure.inputs["Color"].default_value = color
    info = tree.nodes.new("ShaderNodeVolumeInfo")
    scale = tree.nodes.new("ShaderNodeMath")
    scale.operation = "MULTIPLY"
    scale.inputs[1].default_value = density
    tree.links.new(info.outputs["Density"], scale.inputs[0])
    tree.links.new(scale.outputs["Value"], closure.inputs["Density"])
    tree.links.new(closure.outputs["Volume"], output.inputs["Volume"])
    volume.data.materials.append(material)

def render_reference(filename):
    path = os.path.join(out_dir, filename)
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print("CYCLES_REFERENCE", filename, os.path.getsize(path))

def hashed_density(x, y, z):
    h = (x * 0x9E3779B1 + y * 0x85EBCA77 + z * 0xC2B2AE3D + 0x27D4EB2F) & 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 0x7FEB352D) & 0xFFFFFFFF
    h ^= h >> 15
    h = (h * 0x846CA68B) & 0xFFFFFFFF
    h ^= h >> 16
    return 0.25 + 0.75 * ((h >> 8) & 0xFFFFFF) / 0xFFFFFF


def build_hashed_vdb(path):
    grid = openvdb.FloatGrid()
    grid.name = "density"
    grid.transform = openvdb.createLinearTransform(1.0 / 16.0)
    grid.transform.postTranslate((-0.5, -0.5, -1.5))
    accessor = grid.getAccessor()
    for z in range(HASHED_Z):
        for y in range(N):
            for x in range(N):
                accessor.setValueOn((x, y, z), hashed_density(x, y, z))
    openvdb.write(path, grids=[grid])


CAPTURE_SEEDS = (0, 1, 2, 3)
CYCLES_CAPTURE_SAMPLES = 16384
PRINCIPLED_COLOR = (0.05, 0.4, 0.9, 1.0)
ABSORPTION_COLOR = (0.0, 0.8, 0.8, 1.0)
MESH_ABSORPTION_COLOR = (0.5, 0.5, 0.5, 1.0)
POINT_LIGHT_POSITION = (4.0, 2.0, -1.9)
POINT_LIGHT_ENERGY = 1000.0


def material_with_principled_volume(volume):
    material = bpy.data.materials.new("PrincipledHeterogeneousVolume")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    principled = tree.nodes.new("ShaderNodeVolumePrincipled")
    principled.inputs["Color"].default_value = PRINCIPLED_COLOR
    principled.inputs["Absorption Color"].default_value = ABSORPTION_COLOR
    principled.inputs["Density"].default_value = 1.2
    principled.inputs["Anisotropy"].default_value = 0.0
    tree.links.new(principled.outputs["Volume"], output.inputs["Volume"])
    volume.data.materials.append(material)


def material_with_mesh_absorption(mesh):
    material = bpy.data.materials.new("ClosedMeshAbsorptionVolume")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    absorption = tree.nodes.new("ShaderNodeVolumeAbsorption")
    absorption.inputs["Color"].default_value = MESH_ABSORPTION_COLOR
    absorption.inputs["Density"].default_value = 0.3
    tree.links.new(absorption.outputs["Volume"], output.inputs["Volume"])
    mesh.data.materials.append(material)


def add_capture_light(scene):
    light_data = bpy.data.lights.new("Point", "POINT")
    light_data.energy = POINT_LIGHT_ENERGY
    light_data.shadow_soft_size = 0.0
    light = bpy.data.objects.new("Point", light_data)
    scene.collection.objects.link(light)
    light.location = POINT_LIGHT_POSITION

def setup_capture_scene(name):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    render = scene.render
    render.engine = "CYCLES"
    render.resolution_x = SIZE
    render.resolution_y = SIZE
    render.resolution_percentage = 100
    render.film_transparent = False
    render.image_settings.file_format = "OPEN_EXR"
    render.image_settings.color_depth = "32"
    render.image_settings.color_mode = "RGBA"
    render.image_settings.exr_codec = "NONE"
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.cycles.device = "CPU"
    scene.cycles.samples = CYCLES_CAPTURE_SAMPLES
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = 4
    scene.cycles.volume_bounces = 3
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 1.0

    world = bpy.data.worlds.new(name + "World")
    world.use_nodes = True
    scene.world = world
    background = world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    background.inputs["Strength"].default_value = 0.0

    camera_data = bpy.data.cameras.new(name + "Camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 1.0
    camera = bpy.data.objects.new(name + "Camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0.0, 0.0, 3.0)
    camera.rotation_euler = (0.0, 0.0, 0.0)
    scene.camera = camera

    bpy.ops.object.volume_import(filepath=hashed_vdb_path)
    voxel_volume = bpy.context.active_object
    material_with_principled_volume(voxel_volume)
    add_capture_light(scene)
    return scene, voxel_volume


def mean_from_exr(path):
    loaded = bpy.data.images.load(path, check_existing=False)
    pixels = list(loaded.pixels)
    bpy.data.images.remove(loaded)
    return [
        sum(pixels[offset + channel] for offset in range(0, len(pixels), 4)) / (SIZE * SIZE)
        for channel in range(3)
    ]


def capture_case(name, mixed_mesh=False):
    scene, _ = setup_capture_scene(name)
    if mixed_mesh:
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
        material_with_mesh_absorption(bpy.context.object)
    seed_means = []
    for seed in CAPTURE_SEEDS:
        scene.cycles.seed = seed
        path = os.path.join(work, "{}_seed_{}.exr".format(name, seed))
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        seed_means.append(mean_from_exr(path))
        os.remove(path)
    means = [sum(values[channel] for values in seed_means) / len(seed_means) for channel in range(3)]
    errors = []
    for channel in range(3):
        variance = sum((values[channel] - means[channel]) ** 2 for values in seed_means) / (
            len(seed_means) - 1
        )
        errors.append(math.sqrt(variance / len(seed_means)))
    record = {"mean": means, "stderr": errors, "seed_means": seed_means}
    print("ORACLE_CASE", name, json.dumps(record))
    return record


def generate_new_oracles():
    global hashed_vdb_path
    hashed_vdb_path = os.path.join(work, "hashed_density.vdb")
    build_hashed_vdb(hashed_vdb_path)
    records = {
        "principled_heterogeneous": capture_case("principled_heterogeneous"),
        "mixed_voxel_mesh": capture_case("mixed_voxel_mesh", mixed_mesh=True),
    }
    path = os.path.join(out_dir, "volume_spectral_extinction.json")
    with open(path, "w") as handle:
        json.dump(
            {
                "blender": bpy.app.version_string,
                "resolution": [SIZE, SIZE],
                "seeds": list(CAPTURE_SEEDS),
                "cycles_samples": CYCLES_CAPTURE_SAMPLES,
                "cases": records,
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    print("ORACLE_OK volume_spectral_extinction", path)


hashed_vdb_path = None
if new_only:
    generate_new_oracles()
    shutil.rmtree(work)
else:
    scene, volume = setup_scene("Absorption", (1.0, 1.0, 1.0), CYCLES_SAMPLES)
    material_with_volume(volume, "ShaderNodeVolumeAbsorption", ABSORPTION_COLOR, 1.0)
    render_reference("volume_absorption_cycles.exr")



