"""EEVEE oracle for Principled transmission over a known world, with default EEVEE settings (5.2.2).

Three unit spheres sit in a row under a white world of strength 1 and no lights, seen from an orthographic camera above
them (the default camera looks down -Z):
- `clear`: Base Color white, Roughness 0, so the glass is a sharp reflection and refraction;
- `tinted`: Base Color (0.9, 0.5, 0.2), Roughness 0, so the transmission tint is the square root of the base;
- `rough`: Base Color white, Roughness 0.5, so the GGX split and the probe blur are rough.
Every sphere has Transmission Weight 1 and IOR 1.45. The material keeps Blender's defaults: no Raytrace
Transmission (`use_raytrace_refraction` off), the Sphere thickness mode, and no Thickness socket, so each refraction
takes the object's bounds as its thickness.

The world is white because EEVEE's coloured-world output is not linear in the world colour (green and blue come out about
1% below the glass model at 0.6 and 0.3, while the white world and the background itself are exact), so a coloured world
would test EEVEE's colour path rather than the transmission model.

The JSON lists the pixels whose centres lie within 0.75 of each sphere's radius (the interior, where the sphere's
silhouette and EEVEE's 1.5 pixel filter do not reach), with their EEVEE RGB, so the Potter test compares the same
pixels and can measure its deviation from the model.

Arguments after `--`: blend_out, exr_out, json_out.
"""

import json
import sys

import bpy

blend_out, exr_out, json_out = sys.argv[sys.argv.index("--") + 1 :][:3]

SIZE = 96
ORTHO_SCALE = 7.6
WORLD = (1.0, 1.0, 1.0)
IOR = 1.45
INTERIOR = 0.75
SPHERES = [
    ("clear", -2.4, (1.0, 1.0, 1.0, 1.0), 0.0),
    ("tinted", 0.0, (0.9, 0.5, 0.2, 1.0), 0.0),
    ("rough", 2.4, (1.0, 1.0, 1.0, 1.0), 0.5),
]

scene = bpy.context.scene
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

world = bpy.data.worlds.new("Oracle")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (*WORLD, 1.0)
scene.world = world

view = scene.view_settings
view.view_transform = "Standard"
view.look = "None"
view.exposure = 0.0
view.gamma = 1.0
scene.display_settings.display_device = "sRGB"


def glass(name, base, roughness):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = base
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["IOR"].default_value = IOR
    bsdf.inputs["Transmission Weight"].default_value = 1.0
    return material


for name, x, base, roughness in SPHERES:
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=64, ring_count=32, location=(x, 0.0, 0.0))
    sphere = bpy.context.object
    sphere.name = name
    bpy.ops.object.shade_smooth()
    sphere.data.materials.append(glass(f"Glass_{name}", base, roughness))

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
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.render.filepath = exr_out

bpy.ops.wm.save_as_mainfile(filepath=blend_out)
bpy.ops.render.render(write_still=True)

result = bpy.data.images.load(exr_out, check_existing=False)
width, height = result.size
pixels = list(result.pixels)
scale = width / ORTHO_SCALE


def pixel(x, y):
    start = (y * width + x) * 4
    return pixels[start : start + 3]


spheres = {}
for name, x, _, _ in SPHERES:
    # Image x grows with world X; the centre row is the sphere's centre line.
    centre_x = (x + ORTHO_SCALE / 2.0) * scale
    centre_y = height / 2.0
    radius = scale
    samples = []
    for row in range(height):
        for column in range(width):
            px = column + 0.5
            py = row + 0.5
            dx = (px - centre_x) / radius
            dy = (py - centre_y) / radius
            if dx * dx + dy * dy > INTERIOR * INTERIOR:
                continue
            rgb = pixel(column, row)
            samples.append(
                {
                    "x": column,
                    "y": row,
                    "uv": [(column + 0.5) / width, (row + 0.5) / height],
                    "rgb": rgb,
                    "normalized_radius": (dx * dx + dy * dy) ** 0.5,
                }
            )
    spheres[name] = {"roughness": next(r for n, _, _, r in SPHERES if n == name), "samples": samples}

with open(json_out, "w", encoding="utf-8") as handle:
    json.dump({"size": SIZE, "ortho_scale": ORTHO_SCALE, "spheres": spheres}, handle)
