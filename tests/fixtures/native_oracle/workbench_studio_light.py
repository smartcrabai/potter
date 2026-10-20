"""Blender 5.2.2 oracle: Workbench STUDIO lighting of a sphere over a plane.

A smooth-shaded UV sphere (radius 1, 32 segments, 16 rings) hangs over a 4 x 4 plane at z = -1.
Both are plain Workbench MATERIAL surfaces with the factory `Default` studio light, render
engine BLENDER_WORKBENCH, anti-aliasing off, and the Standard view transform, so each PNG holds
the sRGB value of the Workbench result. The world colour is set to the linear value of sRGB 242,
the Solid preview background. Two cameras frame the scene: a perspective camera and an
orthographic one. Their transforms are the same literals Potter's test uses.

Run: blender -b --factory-startup --python workbench_studio_light.py -- <png_dir>
"""

import os
import sys

import bpy

png_dir = sys.argv[sys.argv.index("--") + 1]

# sRGB 242 -> linear, so that the Standard view transform writes the Solid preview background.
BACKGROUND_LINEAR = 0.887923

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

scene.render.engine = "BLENDER_WORKBENCH"
scene.render.resolution_x = 128
scene.render.resolution_y = 128
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.render.dither_intensity = 0.0
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.render.image_settings.color_depth = "8"
scene.display.render_aa = "OFF"
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0

shading = scene.display.shading
shading.light = "STUDIO"
shading.studio_light = "Default"
shading.color_type = "MATERIAL"
shading.show_specular_highlight = True
shading.show_cavity = False
shading.show_shadows = False
shading.show_object_outline = False
shading.show_backface_culling = False
shading.show_xray = False
shading.use_world_space_lighting = False

world = bpy.data.worlds.new("World")
world.color = (BACKGROUND_LINEAR, BACKGROUND_LINEAR, BACKGROUND_LINEAR)
scene.world = world


def material(name, color, roughness, metallic):
    # Workbench's MATERIAL colour type reads the viewport colour, roughness and metallic.
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1.0)
    mat.roughness = roughness
    mat.metallic = metallic
    return mat


bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=1.0, location=(0.0, 0.0, 0.0))
sphere = bpy.context.active_object
sphere.name = "Ball"
bpy.ops.object.shade_smooth()
sphere.data.materials.append(material("Ball", (0.8, 0.3, 0.2), 0.25, 0.5))

bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0.0, 0.0, -1.0))
floor = bpy.context.active_object
floor.name = "Floor"
floor.data.materials.append(material("Floor", (0.8, 0.8, 0.8), 0.4, 0.0))

# Quaternions are (w, x, y, z) for Blender; each camera looks at the origin with +Z up.
CAMERAS = {
    "perspective": {
        "ortho": False,
        "location": (3.2, -4.6, 2.6),
        "rotation": (0.804255397, 0.513445087, 0.161024448, 0.252227132),
    },
    "orthographic": {
        "ortho": True,
        "location": (-2.6, -3.4, 3.9),
        "rotation": (0.866441905, 0.382697454, -0.129555881, -0.293319548),
    },
}

for label, spec in CAMERAS.items():
    camera_data = bpy.data.cameras.new(label)
    camera_data.type = "ORTHO" if spec["ortho"] else "PERSP"
    if spec["ortho"]:
        camera_data.ortho_scale = 4.4
    camera = bpy.data.objects.new(label, camera_data)
    scene.collection.objects.link(camera)
    camera.location = spec["location"]
    camera.rotation_mode = "QUATERNION"
    camera.rotation_quaternion = spec["rotation"]
    scene.camera = camera
    scene.render.filepath = os.path.join(png_dir, f"{label}.png")
    bpy.ops.render.render(write_still=True)

print("WORKBENCH_STUDIO_RENDERED", sorted(CAMERAS))
