"""EEVEE oracle for direct light through glass, with default EEVEE settings (5.2.2) and a black world.

Two glass spheres sit in a row (unit radius, Principled BSDF, Transmission Weight 1, IOR 1.45), seen from an orthographic
camera above them (the default camera looks down -Z):
- `clear`: Roughness 0 at X = -2.4;
- `rough`: Roughness 0.3 at X = +2.4.
The world is black, so every pixel is direct light: the glass reflection and the transmitted light of the scene's
light. Each scene has one light, which the argument names:
- `point_above`: a point light of radius 0.5 above the spheres (soft falloff on, as Blender's default), so the glass
  reflects it and transmits none (a light on the viewer's side is outside the transmission lobe);
- `area_above`: a 2 m square area light above the spheres, facing down;
- `point_below`: the point light below the spheres, so the light refracts through them to the viewer;
- `area_below`: the 2 m square area light below the spheres, facing up.

Each sphere keeps Blender's defaults for everything else: no Raytrace Transmission (`use_raytrace_refraction` off),
the Sphere thickness mode, and no Thickness socket, so each refraction takes the object's bounds as its thickness
(the diameter, 2). The film filter is 0 px, so each pixel is one sample at its centre and the oracle has no filter
blur at the highlight edges.

The JSON lists the pixels whose centres lie within 0.75 of each sphere's radius (the interior, where the silhouette
and the shadow terminator do not reach), with their EEVEE RGB and their world-space X and Y, so the Python mirror
(`eevee_glass_direct_mirror.py`) and the Potter test compare the same pixels.

Arguments after `--`: scene, blend_out, exr_out, json_out.
"""

import json
import math
import sys

import bpy

scene_name, blend_out, exr_out, json_out = sys.argv[sys.argv.index("--") + 1 :][:4]

SIZE = 96
ORTHO_SCALE = 7.6
IOR = 1.45
INTERIOR = 0.75
LIGHT_ENERGY = 40.0
SPHERES = [("clear", -2.4, 0.0), ("rough", 2.4, 0.3)]
LIGHTS = {
    "point_above": ("POINT", (0.0, 0.0, 4.0), (0.0, 0.0, 0.0)),
    "area_above": ("AREA", (0.0, 0.0, 4.0), (0.0, 0.0, 0.0)),
    "point_below": ("POINT", (0.0, 0.0, -4.0), (0.0, 0.0, 0.0)),
    "area_below": ("AREA", (0.0, 0.0, -4.0), (math.pi, 0.0, 0.0)),
}

scene = bpy.context.scene
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

world = bpy.data.worlds.new("Oracle")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
scene.world = world

view = scene.view_settings
view.view_transform = "Standard"
view.look = "None"
view.exposure = 0.0
view.gamma = 1.0
scene.display_settings.display_device = "sRGB"


def glass(name, roughness):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["IOR"].default_value = IOR
    bsdf.inputs["Transmission Weight"].default_value = 1.0
    return material


for name, x, roughness in SPHERES:
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=64, ring_count=32, location=(x, 0.0, 0.0))
    sphere = bpy.context.object
    sphere.name = name
    bpy.ops.object.shade_smooth()
    sphere.data.materials.append(glass(f"Glass_{name}", roughness))

kind, location, rotation = LIGHTS[scene_name]
light_data = bpy.data.lights.new(f"Light_{scene_name}", kind)
light_data.energy = LIGHT_ENERGY
light_data.color = (1.0, 1.0, 1.0)
if kind == "POINT":
    light_data.shadow_soft_size = 0.5
    light_data.use_soft_falloff = True
else:
    light_data.shape = "SQUARE"
    light_data.size = 2.0
light = bpy.data.objects.new(f"Light_{scene_name}", light_data)
scene.collection.objects.link(light)
light.location = location
light.rotation_euler = rotation

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
scene.render.filter_size = 0.0
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.render.filepath = exr_out

bpy.ops.wm.save_as_mainfile(filepath=blend_out)
bpy.ops.render.render(write_still=True)

result = bpy.data.images.load(exr_out, check_existing=False)
width, height = result.size
pixels = list(result.pixels)
pixel_size = ORTHO_SCALE / width


def pixel(x, y):
    start = (y * width + x) * 4
    return pixels[start : start + 3]


spheres = {}
for name, x, roughness in SPHERES:
    centre_x = (x + ORTHO_SCALE / 2.0) * width / ORTHO_SCALE
    centre_y = height / 2.0
    samples = []
    for row in range(height):
        for column in range(width):
            px = column + 0.5
            py = row + 0.5
            dx = (px - centre_x) / (width / ORTHO_SCALE)
            dy = (py - centre_y) / (width / ORTHO_SCALE)
            radius = (dx * dx + dy * dy) ** 0.5
            if radius > INTERIOR:
                continue
            # Blender's image rows run bottom-up, so row `r` is world Y `-ORTHO/2 + (r + 0.5) * pixel_size`.
            world_x = -ORTHO_SCALE / 2.0 + (column + 0.5) * pixel_size
            world_y = -ORTHO_SCALE / 2.0 + (row + 0.5) * pixel_size
            samples.append(
                {
                    "x": column,
                    "y": row,
                    "uv": [(column + 0.5) / width, (row + 0.5) / height],
                    "world": [world_x, world_y],
                    "rgb": pixel(column, row)[:3],
                    "normalized_radius": radius,
                }
            )
    spheres[name] = {"roughness": roughness, "centre": [x, 0.0, 0.0], "samples": samples}

with open(json_out, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "scene": scene_name,
            "size": SIZE,
            "ortho_scale": ORTHO_SCALE,
            "ior": IOR,
            "light": {
                "type": kind,
                "location": list(location),
                "rotation": list(rotation),
                "energy": LIGHT_ENERGY,
                "color": [1.0, 1.0, 1.0],
                "radius": light_data.shadow_soft_size,
                "soft_falloff": light_data.use_soft_falloff if kind == "POINT" else None,
                "area_size": light_data.size if kind == "AREA" else None,
                "light_threshold": scene.eevee.light_threshold,
            },
            "object_thickness": 2.0,
            "spheres": spheres,
        },
        handle,
    )
