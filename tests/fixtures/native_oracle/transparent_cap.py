"""Transparent bounce cap, shadow budget and volume film alpha, rendered by Blender's Cycles (v5.2.2).

Plane kinds: a lit diffuse floor sits under N alpha-0.5 planes, and a point light sits above them.
Each plane is a black BLENDED Principled with Alpha 0.5, so a camera ray passes it with probability
0.5 and is absorbed otherwise. `transparent_max_bounces` is C.

Cycles' rules (`intern/cycles/kernel/integrator/path_state.h`, `intersect_filter.h`):
  - the pass that reaches C sets TERMINATE_ON_NEXT_SURFACE, so the next surface, the floor included,
    ends the path after its emission (no direct light);
  - a shadow ray from a vertex with t transparent passes may cross at most C - t transparent
    surfaces, and is blocked past that.

For the floor to be lit, the camera path passes all N planes (probability 2^-N). The floor is
terminated when N >= C. Its shadow ray crosses N planes against the budget C - N, so it is blocked
when N > C - N, and otherwise transmits 2^-N. Expected patch means, with L0 the lit floor radiance:
  n2c4: 2^-2 * 2^-2 * L0    n2c3: 0 (shadow blocked)    n2c2: 0 (terminated)    n1c2: 2^-1 * 2^-1 * L0

Volume kind `volume_alpha`: a homogeneous absorbing slab (a Principled Volume with Density SIGMA,
Color and Absorption Color black, so its extinction is SIGMA per unit length in Cycles' closure
model) over a transparent card, with film_transparent and an empty world. Each camera ray crosses
the slab's full thickness, so the film alpha is 1 - exp(-SIGMA * SLAB_SIZE) for every pixel.

Every kind writes the patch mean RGB and the patch mean alpha.

Arguments after `--`: blend_out, exr_out, json_out, kind (n2c4 | n2c3 | n2c2 | n1c2 | volume_alpha).
"""

import json
import math
import sys

import bpy

blend_out, exr_out, json_out, kind = sys.argv[sys.argv.index("--") + 1:][:4]

PLANE_CASES = {"n2c4": (2, 4), "n2c3": (2, 3), "n2c2": (2, 2), "n1c2": (1, 2)}
WIDTH = 32
HEIGHT = 32
SAMPLES = 256
PLANE_SPACING = 1.0
LIGHT_HEIGHT = 5.0
LIGHT_ENERGY = 1200.0
LIGHT_RADIUS = 0.5
SIGMA = 0.25
SLAB_SIZE = 4.0
CARD_HEIGHT = -2.5
# Fractions of the frame: the same patch grid as `tests/native_transparent_cap.rs`.
PATCH = [0.25 + 0.5 * index / 8 for index in range(9)]


def black_principled(material, alpha):
    shader = material.node_tree.nodes["Principled BSDF"]
    shader.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    shader.inputs["Metallic"].default_value = 0.0
    shader.inputs["Roughness"].default_value = 1.0
    shader.inputs["Specular IOR Level"].default_value = 0.0
    shader.inputs["Alpha"].default_value = alpha
    return shader


def build_plane_stack(planes, cap, scene):
    floor_material = bpy.data.materials.new("Floor")
    floor_material.use_nodes = True
    floor = floor_material.node_tree.nodes["Principled BSDF"]
    floor.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
    floor.inputs["Metallic"].default_value = 0.0
    floor.inputs["Roughness"].default_value = 1.0
    floor.inputs["Specular IOR Level"].default_value = 0.0
    bpy.ops.mesh.primitive_plane_add(size=10.0, location=(0.0, 0.0, 0.0))
    bpy.context.object.data.materials.append(floor_material)

    plane_material = bpy.data.materials.new("Alpha")
    plane_material.use_nodes = True
    # BLENDED keeps the sub-one Alpha as an alpha blend; the default DITHERED method imports as `clip`.
    plane_material.surface_render_method = "BLENDED"
    black_principled(plane_material, 0.5)
    for index in range(planes):
        bpy.ops.mesh.primitive_plane_add(
            size=10.0, location=(0.0, 0.0, 1.0 + index * PLANE_SPACING)
        )
        bpy.context.object.data.materials.append(plane_material)

    light_data = bpy.data.lights.new("Point", "POINT")
    light_data.energy = LIGHT_ENERGY
    light_data.shadow_soft_size = LIGHT_RADIUS
    light = bpy.data.objects.new(light_data.name, light_data)
    scene.collection.objects.link(light)
    light.location = (0.0, 0.0, LIGHT_HEIGHT)


def build_volume_slab(scene):
    scene.render.film_transparent = True

    slab_material = bpy.data.materials.new("Slab")
    # BLENDED: Potter imports a DITHERED material whose Principled Alpha is unauthored as `opaque`,
    # which forces alpha 1 and hides the Transparent BSDF; the card needs the same for the same reason.
    slab_material.surface_render_method = "BLENDED"
    slab_material.use_nodes = True
    tree = slab_material.node_tree
    output = tree.nodes["Material Output"]
    surface = tree.nodes.new("ShaderNodeBsdfTransparent")
    tree.links.new(surface.outputs[0], output.inputs["Surface"])
    volume = tree.nodes.new("ShaderNodeVolumePrincipled")
    volume.inputs["Density"].default_value = SIGMA
    volume.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    volume.inputs["Absorption Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    tree.links.new(volume.outputs[0], output.inputs["Volume"])
    bpy.ops.mesh.primitive_cube_add(size=SLAB_SIZE, location=(0.0, 0.0, 0.0))
    bpy.context.object.data.materials.append(slab_material)

    card_material = bpy.data.materials.new("Card")
    card_material.surface_render_method = "BLENDED"
    card_material.use_nodes = True
    card_tree = card_material.node_tree
    card_tree.links.new(
        card_tree.nodes.new("ShaderNodeBsdfTransparent").outputs[0],
        card_tree.nodes["Material Output"].inputs["Surface"],
    )
    bpy.ops.mesh.primitive_plane_add(size=10.0, location=(0.0, 0.0, CARD_HEIGHT))
    bpy.context.object.data.materials.append(card_material)


bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = SAMPLES
scene.cycles.use_denoising = False
scene.cycles.max_bounces = 4
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.render.resolution_x = WIDTH
scene.render.resolution_y = HEIGHT
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_mode = "RGBA"
scene.render.image_settings.color_depth = "32"

if kind == "volume_alpha":
    build_volume_slab(scene)
else:
    planes, cap = PLANE_CASES[kind]
    scene.cycles.transparent_max_bounces = cap
    build_plane_stack(planes, cap, scene)

camera_data = bpy.data.cameras.new("Camera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 4.0
camera = bpy.data.objects.new("Camera", camera_data)
scene.collection.objects.link(camera)
camera.location = (0.0, 0.0, 10.0)
scene.camera = camera

bpy.ops.wm.save_as_mainfile(filepath=blend_out)

scene.render.filepath = exr_out
bpy.ops.render.render(write_still=True)
image = bpy.data.images.load(exr_out, check_existing=False)
width, height = image.size
channels = image.channels
pixels = list(image.pixels)
sums = [0.0, 0.0, 0.0, 0.0]
count = 0
for v in PATCH:
    for u in PATCH:
        x = min(width - 1, int(u * width))
        y = min(height - 1, int(v * height))
        index = (y * width + x) * channels
        for channel in range(4):
            sums[channel] += pixels[index + channel]
        count += 1
means = [value / count for value in sums]

with open(json_out, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "kind": kind,
            "patch_mean": means[:3],
            "patch_alpha_mean": means[3],
            "beer_lambert_alpha": 1.0 - math.exp(-SIGMA * SLAB_SIZE),
        },
        handle,
    )
