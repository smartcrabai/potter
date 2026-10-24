"""EEVEE oracle for Principled Thin Wall transmission and backlit subsurface (Blender 5.2.2).

Run:

    Blender -b --factory-startup -P eevee_thin_wall.py -- CASE OUT.blend OUT.exr [CAPTURE.f32le]

`-` for the EXR path saves only the scene. A capture path writes RGB from Blender's rendered `image.pixels` as
little-endian f32 in row order. The four cases use a zero-depth object (Thickness::None) or a shallow, closed
object (Thickness::Sphere); Thin Wall must keep each material's single-sheet response in either case.
"""

import math
import struct
import sys

import bpy
from mathutils import Vector

arguments = sys.argv[sys.argv.index("--") + 1 :]
CASE, BLEND_OUT, EXR_OUT = arguments[:3]
CAPTURE_OUT = arguments[3] if len(arguments) > 3 else None
SIZE = 96
ORTHO_SCALE = 2.4

CASES = {
    "pane_none": ("pane", False),
    "pane_sphere": ("pane", True),
    "leaf_none": ("leaf", False),
    "leaf_sphere": ("leaf", True),
}
if CASE not in CASES:
    raise SystemExit(f"unknown thin-wall case {CASE!r}")
KIND, HAS_SPHERE_THICKNESS = CASES[CASE]

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.eevee.taa_render_samples = 64
scene.render.resolution_x = SIZE
scene.render.resolution_y = SIZE
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.render.filepath = EXR_OUT
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0
scene.view_layers[0].use_pass_cryptomatte_accurate = False
scene.display_settings.display_device = "sRGB"
scene.render.film_transparent = False

world = bpy.data.worlds.new("BlackWorld")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
scene.world = world

if KIND == "pane":
    if HAS_SPHERE_THICKNESS:
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
        surface = bpy.context.object
        surface.dimensions = (4.0, 4.0, 0.2)
        surface.name = "ThinWallPane"
    else:
        bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0.0, 0.0, 0.0))
        surface = bpy.context.object
        surface.name = "ThinWallPane"
    base_color = (0.78, 0.9, 1.0, 1.0)
    roughness = 0.18
else:
    count = 32
    depth = 0.2 if HAS_SPHERE_THICKNESS else 0.0
    vertices = []
    for index in range(count):
        angle = 2.0 * math.pi * index / count
        vertices.append((1.05 * math.cos(angle), 0.48 * math.sin(angle), depth * 0.5))
    faces = [list(range(count))]
    if HAS_SPHERE_THICKNESS:
        vertices += [(x, y, -depth * 0.5) for x, y, _ in vertices]
        faces.append(list(reversed(range(count, 2 * count))))
        faces.extend(
            (index, (index + 1) % count, (index + 1) % count + count, index + count)
            for index in range(count)
        )
    mesh = bpy.data.meshes.new("ThinWallLeafMesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    surface = bpy.data.objects.new("ThinWallLeaf", mesh)
    scene.collection.objects.link(surface)
    base_color = (0.08, 0.42, 0.12, 1.0)
    roughness = 0.65

material = bpy.data.materials.new("PrincipledThinWall")
material.use_nodes = True
material.thickness_mode = "SPHERE"
principled = material.node_tree.nodes["Principled BSDF"]
principled.inputs["Base Color"].default_value = base_color
principled.inputs["Roughness"].default_value = roughness
principled.inputs["IOR"].default_value = 1.45
principled.inputs["Thin Wall"].default_value = True
if KIND == "pane":
    principled.inputs["Transmission Weight"].default_value = 1.0
else:
    principled.inputs["Subsurface Weight"].default_value = 0.85
    principled.inputs["Subsurface Radius"].default_value = (1.0, 0.45, 0.2)
    principled.inputs["Subsurface Scale"].default_value = 0.16
surface.data.materials.append(material)

light_data = bpy.data.lights.new("Backlight", "AREA")
light_data.shape = "SQUARE"
light_data.size = 1.0
light_data.energy = 1.0 if KIND == "pane" else 100.0
light = bpy.data.objects.new("Backlight", light_data)
scene.collection.objects.link(light)
light.location = (0.0, 0.0, -1.5)
light.rotation_euler = (Vector((0.0, 0.0, 0.0)) - light.location).to_track_quat("-Z", "Y").to_euler()

camera_data = bpy.data.cameras.new("Camera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = ORTHO_SCALE
camera = bpy.data.objects.new("Camera", camera_data)
scene.collection.objects.link(camera)
camera.location = (0.0, 0.0, 5.0)
camera.rotation_euler = (Vector((0.0, 0.0, 0.0)) - camera.location).to_track_quat("-Z", "Y").to_euler()
scene.camera = camera

bpy.ops.wm.save_as_mainfile(filepath=BLEND_OUT)
if EXR_OUT == "-":
    raise SystemExit(0)
bpy.ops.render.render(write_still=True)
if CAPTURE_OUT is not None:
    image = bpy.data.images.load(EXR_OUT, check_existing=False)
    width, height = image.size
    pixels = list(image.pixels)
    if len(pixels) != width * height * 4:
        raise RuntimeError(f"unexpected render pixel count: {len(pixels)}")
    with open(CAPTURE_OUT, "wb") as capture:
        for offset in range(0, len(pixels), 4):
            capture.write(struct.pack("<fff", pixels[offset], pixels[offset + 1], pixels[offset + 2]))
