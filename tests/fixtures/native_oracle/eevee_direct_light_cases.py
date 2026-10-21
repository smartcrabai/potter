"""EEVEE oracle for the realtime direct light's cut-offs, spot cone and sun disc, with EEVEE 5.2.2's default settings.

A 40 m white-albedo Diffuse BSDF plane (albedo 0.8, Lambert) lies at Z = 0 under a black world. The camera looks down
from above, orthographic, so every pixel is the plane's direct light at its world position. Each case has one light,
which the argument names:
- `influence_point`: a point light of 40 W at height 2, the default `eevee.light_threshold` (0.01): its influence ends
  at R = sqrt(max_power / threshold), with max_power = colour · 40 / 100 = 0.4, so R = 6.3 m and the cut-off is soft;
- `influence_threshold`: the light at height 1 with `eevee.light_threshold` 0.1, so R = 2 m and the plane is lit only inside
  a 1.7 m radius;
- `influence_custom`: the same light with `use_custom_distance` on and `cutoff_distance` 3 m;
- `influence_area`: a 2 m square area light at height 2 facing down, 40 W, default threshold;
- `spot_plain`: a spot light at height 3, 60 degrees, blend 0.3, scale 1: the cone on the plane;
- `spot_scaled`: the same spot with its object scaled (1, 1, 2), which stretches the cone in EEVEE;
- `sun_angle`: a sun of strength 3 with a 10 degree angle, and a 0.5 m sphere 1 m above the plane, so the sphere's
  shadow has a penumbra whose width is set by the angle.

The JSON lists every pixel (all are inside the plane), with its world position, its RGB and the case's parameters.
The Python mirror (`eevee_direct_light_mirror.py`) predicts the same pixels, and `tests/realtime_direct_light_cases.rs`
compares Potter's realtime pixels with them.

Arguments after `--`: case, blend_out, exr_out, json_out.
"""

import json
import math
import sys

import bpy

case, blend_out, exr_out, json_out = sys.argv[sys.argv.index("--") + 1 :][:4]

SIZE = 128
ORTHO_SCALE = 12.0
ALBEDO = 0.8
CASES = {
    "influence_point": {"light": "POINT", "location": (0.0, 0.0, 2.0), "energy": 40.0, "threshold": 0.01},
    "influence_threshold": {"light": "POINT", "location": (0.0, 0.0, 1.0), "energy": 40.0, "threshold": 0.1},
    "influence_custom": {"light": "POINT", "location": (0.0, 0.0, 2.0), "energy": 40.0, "threshold": 0.01, "custom": 3.0},
    "influence_area": {"light": "AREA", "location": (0.0, 0.0, 2.0), "energy": 40.0, "threshold": 0.01},
    "spot_plain": {"light": "SPOT", "location": (0.0, 0.0, 3.0), "energy": 400.0, "threshold": 0.01, "scale": (1.0, 1.0, 1.0)},
    "spot_scaled": {"light": "SPOT", "location": (0.0, 0.0, 3.0), "energy": 400.0, "threshold": 0.01, "scale": (1.0, 1.0, 2.0)},
    "sun_angle": {"light": "SUN", "location": (0.0, 0.0, 10.0), "energy": 3.0, "threshold": 0.01, "angle_degrees": 10.0},
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

material = bpy.data.materials.new("Lambert")
material.use_nodes = True
nodes = material.node_tree.nodes
links = material.node_tree.links
for node in list(nodes):
    if node.type != "OUTPUT_MATERIAL":
        nodes.remove(node)
diffuse = nodes.new("ShaderNodeBsdfDiffuse")
diffuse.inputs["Color"].default_value = (ALBEDO, ALBEDO, ALBEDO, 1.0)
links.new(diffuse.outputs["BSDF"], nodes["Material Output"].inputs["Surface"])

bpy.ops.mesh.primitive_plane_add(size=40.0, location=(0.0, 0.0, 0.0))
plane = bpy.context.object
plane.name = "Plane"
plane.data.materials.append(material)

light_data = bpy.data.lights.new(f"Light_{case}", spec["light"])
light_data.energy = spec["energy"]
light_data.color = (1.0, 1.0, 1.0)
light_data.use_shadow = True
if spec["light"] == "POINT":
    light_data.shadow_soft_size = 0.0
    light_data.use_soft_falloff = True
    light_data.use_custom_distance = "custom" in spec
    if "custom" in spec:
        light_data.cutoff_distance = spec["custom"]
elif spec["light"] == "AREA":
    light_data.shape = "SQUARE"
    light_data.size = 2.0
elif spec["light"] == "SPOT":
    light_data.shadow_soft_size = 0.0
    light_data.use_soft_falloff = True
    light_data.spot_size = math.radians(60.0)
    light_data.spot_blend = 0.3
elif spec["light"] == "SUN":
    light_data.angle = math.radians(spec["angle_degrees"])
light = bpy.data.objects.new(f"Light_{case}", light_data)
scene.collection.objects.link(light)
light.location = spec["location"]
light.scale = spec.get("scale", (1.0, 1.0, 1.0))
if spec["light"] == "SUN":
    light.rotation_euler = (0.0, 0.0, 0.0)

if spec["light"] == "SUN":
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, segments=64, ring_count=32, location=(0.0, 0.0, 1.0))
    occluder = bpy.context.object
    occluder.name = "Occluder"
    bpy.ops.object.shade_smooth()
    occluder_material = bpy.data.materials.new("Occluder")
    occluder_material.use_nodes = True
    occluder.data.materials.append(occluder_material)

scene.eevee.light_threshold = spec["threshold"]

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
bpy.ops.render.render(write_still=True)

result = bpy.data.images.load(exr_out, check_existing=False)
width, height = result.size
pixels = list(result.pixels)
pixel_size = ORTHO_SCALE / width
samples = []
for row in range(height):
    for column in range(width):
        start = (row * width + column) * 4
        # Blender's image rows run bottom-up, so row `r` is world Y `-ORTHO/2 + (r + 0.5) * pixel_size`.
        world_x = -ORTHO_SCALE / 2.0 + (column + 0.5) * pixel_size
        world_y = -ORTHO_SCALE / 2.0 + (row + 0.5) * pixel_size
        samples.append({"world": [world_x, world_y], "rgb": pixels[start : start + 3]})

with open(json_out, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "case": case,
            "albedo": ALBEDO,
            "size": SIZE,
            "ortho_scale": ORTHO_SCALE,
            "light": {
                "type": spec["light"],
                "location": list(spec["location"]),
                "scale": list(spec.get("scale", (1.0, 1.0, 1.0))),
                "energy": spec["energy"],
                "threshold": spec["threshold"],
                "custom_distance": spec.get("custom"),
                "angle_degrees": spec.get("angle_degrees"),
                "radius": 0.1 if spec["light"] in ("POINT", "SPOT") else None,
                "soft_falloff": True if spec["light"] in ("POINT", "SPOT") else None,
                "area_size": 2.0 if spec["light"] == "AREA" else None,
                "spot_size_degrees": 60.0 if spec["light"] == "SPOT" else None,
                "spot_blend": 0.3 if spec["light"] == "SPOT" else None,
            },
            "occluder": {"centre": [0.0, 0.0, 1.0], "radius": 0.5} if spec["light"] == "SUN" else None,
            "samples": samples,
        },
        handle,
    )
