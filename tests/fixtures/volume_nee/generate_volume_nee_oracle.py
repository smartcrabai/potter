"""Blender 5.2.2 single-scatter oracle for volume direct lighting under a point light.

Every case is rendered with `volume_bounces = 0` and `max_bounces = 0`, a single scatter. At one bounce Cycles also
adds the direct light of a second scatter vertex (about 12 % more at density 0.25 for a sun, measured with
Blender 5.2.2), so a bounce budget of 1 would not be single scatter; the budget of 0 is the integral the analytic
and Potter's tests compute. The density is low (optical depth
0.05 across the cube), so multiple scattering is about 0.2 % and the render is the single-scatter integral
that `analytic_volume_nee.py` evaluates. At density 2 Blender's render exceeds single scatter by 60 %
(multiple scattering), so a high-density oracle would not test the single-scatter estimator.

Scene: a unit cube at the origin (its extent is [-0.5, 0.5]^3) with a volume-only material, an
orthographic camera at (0, 0, 3) looking down -Z (ortho scale 1, so the frame is the unit cube), a point light of radius 0 at
POSITION, and a black world. The measured value is the mean over the whole 4 x 4 frame, whose pixel centres are at +-0.125 and +-0.375 with footprints of +-0.25 about them, all inside the cube.

Cases:
- diffuse: a white Diffuse plane at z = 0 with no cube. It calibrates the light unit, since the diffuse
  radiance is albedo / pi times the irradiance, and that irradiance is the same quantity the volume
  integral uses.
- homogeneous_g0 / homogeneous_g05: Volume Scatter, Color 1, Density DENSITY, Anisotropy 0 or 0.5.
- gradient_g0 / gradient_g05: Density = GRADIENT_SCALE * x, where x is the Generated coordinate (0 at
  the cube's -X face, 1 at +X), from a Linear Gradient texture.

The frame is 4 x 4 (SIZE) and the measured block is all of it (indices 0..3), so the cube fills the
block. The pixel filter is a one-pixel BOX (see the comment in reset()), which is Potter's sampling, so the Blender
mean is the pixel-averaged value that Potter renders.

Regenerate from this directory with Blender 5.2.2, one thread, one run:

    Blender -b --factory-startup --threads 1 -P generate_volume_nee_oracle.py -- OUT.json
"""

import json
import math
import os
import sys
import tempfile

import bpy

argv = sys.argv[sys.argv.index("--") + 1 :]
out_path = os.path.abspath(argv[0])
work = tempfile.mkdtemp(prefix="volume_nee_")
SIZE = 4
SAMPLES = 65536
BLOCK = range(0, 4)
ENERGY = 100.0
POSITION = (1.6, 0.6, 0.9)
DENSITY = 0.05
GRADIENT_SCALE = 0.05
PIXEL_CENTRES = [-0.5 + (index + 0.5) / SIZE for index in BLOCK]


def reset():
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
    # Potter integrates each pixel uniformly over its whole footprint (render/path.rs draws the sub-pixel offset
    # uniformly in [0, 1)^2, a box filter of one pixel). Cycles' BOX filter of width 1 px is the same box. The 4 px frame
    # spans the unit cube (ortho scale 1), so every pixel (width 0.25, centres at +-0.125 and +-0.375) lies
    # inside the cube, and the outside-cube zero of Cycles' default 1.5 px Blackman-Harris filter cannot enter any pixel.
    # A point filter (filter_width 0.01) would average the pixel centre only.
    if hasattr(scene.cycles, "pixel_filter_type"):
        scene.cycles.pixel_filter_type = "BOX"
    else:
        scene.cycles.filter_type = "BOX"
    scene.cycles.filter_width = 1.0
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


def cube_with_volume(density_value, anisotropy, gradient):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
    cube = bpy.context.active_object
    material = bpy.data.materials.new("medium")
    material.use_nodes = True
    tree = material.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    scatter = tree.nodes.new("ShaderNodeVolumeScatter")
    scatter.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    scatter.inputs["Anisotropy"].default_value = anisotropy
    if gradient:
        coords = tree.nodes.new("ShaderNodeTexCoord")
        ramp = tree.nodes.new("ShaderNodeTexGradient")
        ramp.gradient_type = "LINEAR"
        scale = tree.nodes.new("ShaderNodeMath")
        scale.operation = "MULTIPLY"
        scale.inputs[1].default_value = GRADIENT_SCALE
        tree.links.new(coords.outputs["Generated"], ramp.inputs["Vector"])
        tree.links.new(ramp.outputs["Fac"], scale.inputs[0])
        tree.links.new(scale.outputs["Value"], scatter.inputs["Density"])
    else:
        scatter.inputs["Density"].default_value = density_value
    tree.links.new(scatter.outputs["Volume"], output.inputs["Volume"])
    cube.data.materials.append(material)


def diffuse_plane():
    bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0.0, 0.0, 0.0))
    plane = bpy.context.active_object
    material = bpy.data.materials.new("white")
    material.use_nodes = True
    tree = material.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    tree.links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])
    plane.data.materials.append(material)


def render_block(name):
    path = os.path.join(work, name + ".exr")
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    loaded = bpy.data.images.load(path, check_existing=False)
    pixels = list(loaded.pixels)
    bpy.data.images.remove(loaded)
    values = []
    for y in BLOCK:
        for x in BLOCK:
            offset = (y * SIZE + x) * 4
            values.append(pixels[offset])
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return round(mean, 6), round(math.sqrt(variance / len(values)), 6)


CASES = [
    ("diffuse", None),
    ("homogeneous_g0", ("volume", DENSITY, 0.0, False)),
    ("homogeneous_g05", ("volume", DENSITY, 0.5, False)),
    ("gradient_g0", ("volume", None, 0.0, True)),
    ("gradient_g05", ("volume", None, 0.5, True)),
]

records = {}
for name, spec in CASES:
    reset()
    if spec is None:
        diffuse_plane()
    else:
        _, density_value, anisotropy, gradient = spec
        cube_with_volume(density_value, anisotropy, gradient)
    mean, stderr = render_block(name)
    records[name] = {"mean": mean, "stderr": stderr}
    print("ORACLE_CASE", name, mean, stderr)

with open(out_path, "w") as handle:
    json.dump(
        {
            "blender": bpy.app.version_string,
            "samples": SAMPLES,
            "energy": ENERGY,
            "position": list(POSITION),
            "density": DENSITY,
            "gradient_scale": GRADIENT_SCALE,
            "pixel_centres": PIXEL_CENTRES,
            "cases": records,
        },
        handle,
        indent=1,
    )
print("ORACLE_OK volume_nee", json.dumps(records))
