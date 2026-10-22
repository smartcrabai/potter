"""EEVEE oracle for the light closure factors and the sun's disc, with EEVEE 5.2.2's default settings.

Each case renders one scene under a black world with an orthographic camera looking down, so every pixel is the direct light of
one surface. The argument names the case:
- `diffuse_base` / `diffuse_half`: a Diffuse BSDF plane (albedo 0.8) under a point light at height 2; `diffuse_half` sets the
  light's Diffuse Factor to 0.5 (`diffuse_factor`, EEVEE's `power[LIGHT_DIFFUSE]`);
- `specular_base` / `specular_half`: a Principled plane (Base Color black, Roughness 0.3, Specular IOR Level 0.5) under the
  same light; `specular_half` sets the Specular Factor to 0.5 (`specular_factor`, `power[LIGHT_SPECULAR]`);
- `glass_base` / `glass_zero` / `glass_half`: the unit glass sphere (IOR 1.45, Transmission 1, Roughness 0) under a point light
  2 m below it, so its transmitted light reaches the camera; `glass_zero` sets the Transmission Factor to 0 (the reflection
  alone) and `glass_half` to 0.5 (`transmission_factor`, `power[LIGHT_TRANSMISSION]`);
- `sun_delta` / `sun_disc`: a sun of strength 3 over the Principled plane (Roughness 0.3, Specular IOR Level 0.5). `sun_delta`
  has an angle of 0 (EEVEE's minimum disc, a delta light), and `sun_disc` one of 10 degrees, whose disc the specular lobe
  integrates over (`eevee_light.cc` `shape_radius`).

The JSON lists every pixel with its world position and RGB, and the case's parameters. The Python mirror
(`eevee_light_factors_mirror.py`) checks the factors as ratios and the disc as the ratio of the lobe over the disc to its centre.

Arguments after `--`: case, blend_out, exr_out, json_out.
"""

import json
import math
import sys

import bpy

case, blend_out, exr_out, json_out = sys.argv[sys.argv.index("--") + 1 :][:4]

SIZE = 128
ORTHO_SCALE = 12.0
POINT_ENERGY = 40.0
CASES = {
    "diffuse_base": {"scene": "diffuse", "light": "POINT", "location": (0.0, 0.0, 2.0)},
    "diffuse_half": {"scene": "diffuse", "light": "POINT", "location": (0.0, 0.0, 2.0), "diffuse_factor": 0.5},
    "specular_base": {"scene": "specular", "light": "POINT", "location": (0.0, 0.0, 2.0)},
    "specular_half": {"scene": "specular", "light": "POINT", "location": (0.0, 0.0, 2.0), "specular_factor": 0.5},
    "glass_base": {"scene": "glass", "light": "POINT", "location": (0.0, 0.0, -2.0)},
    "glass_zero": {"scene": "glass", "light": "POINT", "location": (0.0, 0.0, -2.0), "transmission_factor": 0.0},
    "glass_half": {"scene": "glass", "light": "POINT", "location": (0.0, 0.0, -2.0), "transmission_factor": 0.5},
    "sun_delta": {"scene": "specular", "light": "SUN", "location": (0.0, 0.0, 10.0), "angle_degrees": 0.0},
    "sun_disc": {"scene": "specular", "light": "SUN", "location": (0.0, 0.0, 10.0), "angle_degrees": 10.0},
}
spec = CASES[case]

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


def diffuse_material(name):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    for node in list(nodes):
        if node.type != "OUTPUT_MATERIAL":
            nodes.remove(node)
    diffuse = nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1.0)
    links.new(diffuse.outputs["BSDF"], nodes["Material Output"].inputs["Surface"])
    return material


def principled_material(name, base, roughness, transmission):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = base
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["IOR"].default_value = 1.45
    bsdf.inputs["Specular IOR Level"].default_value = 0.5
    bsdf.inputs["Transmission Weight"].default_value = transmission
    return material


if spec["scene"] == "diffuse":
    bpy.ops.mesh.primitive_plane_add(size=40.0, location=(0.0, 0.0, 0.0))
    plane = bpy.context.object
    plane.data.materials.append(diffuse_material("Lambert"))
elif spec["scene"] == "specular":
    bpy.ops.mesh.primitive_plane_add(size=40.0, location=(0.0, 0.0, 0.0))
    plane = bpy.context.object
    plane.data.materials.append(principled_material("Glossy", (0.0, 0.0, 0.0, 1.0), 0.3, 0.0))
else:
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=64, ring_count=32, location=(0.0, 0.0, 0.0))
    sphere = bpy.context.object
    bpy.ops.object.shade_smooth()
    sphere.data.materials.append(principled_material("Glass", (1.0, 1.0, 1.0, 1.0), 0.0, 1.0))

light_data = bpy.data.lights.new(f"Light_{case}", spec["light"])
light_data.color = (1.0, 1.0, 1.0)
light_data.diffuse_factor = spec.get("diffuse_factor", 1.0)
light_data.specular_factor = spec.get("specular_factor", 1.0)
light_data.transmission_factor = spec.get("transmission_factor", 1.0)
if spec["light"] == "POINT":
    light_data.energy = POINT_ENERGY
    light_data.shadow_soft_size = 0.0
    light_data.use_soft_falloff = True
else:
    light_data.energy = 3.0
    light_data.angle = math.radians(spec["angle_degrees"])
light = bpy.data.objects.new(f"Light_{case}", light_data)
scene.collection.objects.link(light)
light.location = spec["location"]

scene.eevee.light_threshold = 0.01

camera_data = bpy.data.cameras.new("OracleCamera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = ORTHO_SCALE
camera = bpy.data.objects.new("OracleCamera", camera_data)
scene.collection.objects.link(camera)
camera.location = (0.0, 0.0, 20.0)
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
# `-` as the EXR path saves the scene and stops: the tests import the .blend and compare with the committed EEVEE render.
if exr_out == "-":
    sys.exit(0)
bpy.ops.render.render(write_still=True)

result = bpy.data.images.load(exr_out, check_existing=False)
width, height = result.size
pixels = list(result.pixels)
pixel_size = ORTHO_SCALE / width
samples = []
for row in range(height):
    for column in range(width):
        start = (row * width + column) * 4
        world_x = -ORTHO_SCALE / 2.0 + (column + 0.5) * pixel_size
        world_y = -ORTHO_SCALE / 2.0 + (row + 0.5) * pixel_size
        samples.append({"world": [world_x, world_y], "rgb": pixels[start : start + 3]})

with open(json_out, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "case": case,
            "size": SIZE,
            "ortho_scale": ORTHO_SCALE,
            "light": {
                "type": spec["light"],
                "location": list(spec["location"]),
                "energy": POINT_ENERGY if spec["light"] == "POINT" else 3.0,
                "diffuse_factor": spec.get("diffuse_factor", 1.0),
                "specular_factor": spec.get("specular_factor", 1.0),
                "transmission_factor": spec.get("transmission_factor", 1.0),
                "angle_degrees": spec.get("angle_degrees"),
            },
            "scene": spec["scene"],
            "samples": samples,
        },
        handle,
    )
