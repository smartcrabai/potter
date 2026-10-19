"""Blender 5.2.2 oracle: Workbench Solid of a `.blend` whose viewport display differs from its node tree.

A smooth-shaded UV sphere and a 4 x 4 plane each have a Principled node tree whose Base Color,
Roughness and Metallic differ from the material's viewport display (`diffuse_color`, `roughness`,
`metallic`). Workbench's MATERIAL colour type reads the viewport values, so the render shows them
and not the node tree. The scene is saved as a `.blend`, which the test imports into Potter, and
rendered with the requested view transform from a perspective and an orthographic camera.

Run: blender -b --factory-startup --python workbench_viewport_display.py -- <png_dir> <blend_path> <view_transform>
"""

import os
import sys

import bpy

png_dir, blend_path, view_transform = sys.argv[sys.argv.index("--") + 1:][:3]

# sRGB 242 -> linear, the Solid preview background that the studio oracle also uses.
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
scene.view_settings.view_transform = view_transform
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


def material(name, node_base, node_roughness, node_metallic, viewport_colour, viewport_roughness, viewport_metallic):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    principled = next(node for node in mat.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    principled.inputs["Base Color"].default_value = (*node_base, 1.0)
    principled.inputs["Roughness"].default_value = node_roughness
    principled.inputs["Metallic"].default_value = node_metallic
    # The viewport display is what Workbench's MATERIAL colour type reads; it is set apart from the node tree.
    mat.diffuse_color = (*viewport_colour, 1.0)
    mat.roughness = viewport_roughness
    mat.metallic = viewport_metallic
    return mat


bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=1.0, location=(0.0, 0.0, 0.0))
sphere = bpy.context.active_object
sphere.name = "Ball"
bpy.ops.object.shade_smooth()
sphere.data.materials.append(material("Ball", (0.1, 0.6, 0.9), 0.8, 0.0, (0.8, 0.3, 0.2), 0.25, 0.5))

bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0.0, 0.0, -1.0))
floor = bpy.context.active_object
floor.name = "Floor"
floor.data.materials.append(material("Floor", (0.9, 0.1, 0.1), 0.1, 1.0, (0.8, 0.8, 0.8), 0.4, 0.0))

# The same cameras as workbench_studio_light.py; quaternions are (w, x, y, z) for Blender.
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

bpy.ops.wm.save_as_mainfile(filepath=blend_path, check_existing=False)

for label in CAMERAS:
    scene.camera = bpy.data.objects[label]
    scene.render.filepath = os.path.join(png_dir, f"{label}.png")
    bpy.ops.render.render(write_still=True)

print("WORKBENCH_VIEWPORT_RENDERED", sorted(CAMERAS), view_transform)
