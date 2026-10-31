"""Build and capture physical Hair Curve collection instances with Blender Cycles."""
import array
import json
import math
import os
import sys

import bpy

WIDTH = 96
HEIGHT = 96
SAMPLES = 64
SEED = 29
ORTHO_SCALE = 6.0
HAIR_Z = 1.0
SUN_ANGLE_X = math.radians(30.0)
SHADOW_Y = HAIR_Z * math.tan(SUN_ANGLE_X)
PATCH_RADIUS = 2
MEMBERS = [
    {
        "name": "HairMemberRed",
        "collection": "HairSourceRed",
        "instance": "RedCollectionInstance",
        "location": [-1.3, 0.0, 0.0],
        "color": [0.95, 0.08, 0.04, 1.0],
        "pass_index": 11,
        "visible_camera": True,
        "visible_diffuse": True,
        "visible_glossy": True,
        "visible_transmission": True,
        "visible_volume_scatter": True,
        "visible_shadow": True,
    },
    {
        "name": "HairMemberBlue",
        "collection": "HairSourceBlue",
        "instance": "BlueCollectionInstance",
        "location": [1.3, 0.0, 0.0],
        "color": [0.04, 0.62, 0.98, 1.0],
        "pass_index": 22,
        "visible_camera": True,
        "visible_diffuse": True,
        "visible_glossy": True,
        "visible_transmission": True,
        "visible_volume_scatter": True,
        "visible_shadow": False,
    },
]
CURVE_POINTS = [
    [-0.55, -0.2, 0.0], [-0.2, -0.2, 0.0], [0.2, -0.2, 0.0], [0.55, -0.2, 0.0],
    [-0.55, 0.0, 0.0], [-0.2, 0.0, 0.0], [0.2, 0.0, 0.0], [0.55, 0.0, 0.0],
    [-0.55, 0.2, 0.0], [-0.2, 0.2, 0.0], [0.2, 0.2, 0.0], [0.55, 0.2, 0.0],
]
CURVE_RADIUS = 0.075

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"expected Blender 5.2.2, got {bpy.app.version_string}")

args = sys.argv[sys.argv.index("--") + 1:]
if len(args) != 1:
    raise RuntimeError("usage: generate.py -- <output-directory>")
out_dir = os.path.abspath(args[0])
os.makedirs(out_dir, exist_ok=True)
bpy.context.preferences.filepaths.save_version = 0

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
for collection in list(bpy.data.collections):
    if collection.name != bpy.context.scene.collection.name:
        bpy.data.collections.remove(collection)
for material in list(bpy.data.materials):
    bpy.data.materials.remove(material)

scene = bpy.context.scene
scene.name = "Hair Instance Context 18"
scene.render.engine = "CYCLES"
scene.render.resolution_x = WIDTH
scene.render.resolution_y = HEIGHT
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_mode = "RGBA"
scene.render.image_settings.color_depth = "32"
scene.render.image_settings.exr_codec = "ZIP"
scene.render.threads_mode = "FIXED"
scene.render.threads = 1
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0
scene.cycles.device = "CPU"
scene.cycles.samples = SAMPLES
scene.cycles.seed = SEED
scene.cycles.pixel_filter_type = "BOX"
scene.cycles.filter_width = 1.0
scene.cycles.use_denoising = False
scene.cycles.max_bounces = 2
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transmission_bounces = 0
scene.cycles.volume_bounces = 0

world = bpy.data.worlds.new("Black Hair Instance World")
world.use_nodes = True
world.node_tree.nodes.clear()
background = world.node_tree.nodes.new("ShaderNodeBackground")
background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
background.inputs["Strength"].default_value = 0.0
world_output = world.node_tree.nodes.new("ShaderNodeOutputWorld")
world.node_tree.links.new(background.outputs["Background"], world_output.inputs["Surface"])
scene.world = world

camera_data = bpy.data.cameras.new("Hair Instance Camera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = ORTHO_SCALE
camera = bpy.data.objects.new("Hair Instance Camera", camera_data)
scene.collection.objects.link(camera)
camera.location = (0.0, 0.0, 8.0)
camera.rotation_euler = (0.0, 0.0, 0.0)
scene.camera = camera

floor_mesh = bpy.data.meshes.new("Hair Instance Receiver Mesh")
floor_mesh.from_pydata(
    [(-3.5, -3.5, -0.05), (3.5, -3.5, -0.05), (3.5, 3.5, -0.05), (-3.5, 3.5, -0.05)],
    [],
    [(0, 1, 2, 3)],
)
floor = bpy.data.objects.new("Hair Instance Receiver", floor_mesh)
scene.collection.objects.link(floor)
floor_mat = bpy.data.materials.new("Hair Instance Diffuse Receiver")
floor_mat.use_nodes = True
floor_mat.node_tree.nodes.clear()
floor_diffuse = floor_mat.node_tree.nodes.new("ShaderNodeBsdfDiffuse")
floor_diffuse.inputs["Color"].default_value = (0.72, 0.72, 0.72, 1.0)
floor_output = floor_mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
floor_mat.node_tree.links.new(floor_diffuse.outputs["BSDF"], floor_output.inputs["Surface"])
floor.data.materials.append(floor_mat)

sun_data = bpy.data.lights.new("Hair Instance Shadow Sun", type="SUN")
sun_data.energy = 2.0
sun_data.angle = 0.0
sun = bpy.data.objects.new("Hair Instance Shadow Sun", sun_data)
scene.collection.objects.link(sun)
sun.rotation_euler = (SUN_ANGLE_X, 0.0, 0.0)

curve_data = bpy.data.hair_curves.new("Shared Physical Hair Curves")
curve_data.add_curves([4, 4, 4])
curve_data.set_types(type="CATMULL_ROM")
for point, position in zip(curve_data.points, CURVE_POINTS):
    point.position = position
    point.radius = CURVE_RADIUS

hair_mat = bpy.data.materials.new("Object Info Colored Cycles Hair")
hair_mat.use_nodes = True
hair_mat.node_tree.nodes.clear()
hair_nodes = hair_mat.node_tree.nodes
hair_links = hair_mat.node_tree.links
object_info = hair_nodes.new("ShaderNodeObjectInfo")
map_range = hair_nodes.new("ShaderNodeMapRange")
map_range.inputs["From Min"].default_value = float(MEMBERS[0]["pass_index"])
map_range.inputs["From Max"].default_value = float(MEMBERS[1]["pass_index"])
map_range.inputs["To Min"].default_value = 0.0
map_range.inputs["To Max"].default_value = 1.0
color_ramp = hair_nodes.new("ShaderNodeValToRGB")
color_ramp.color_ramp.interpolation = "CONSTANT"
color_ramp.color_ramp.elements[0].color = (0.95, 0.08, 0.04, 1.0)
color_ramp.color_ramp.elements[1].color = (0.04, 0.62, 0.98, 1.0)
hair_bsdf = hair_nodes.new("ShaderNodeBsdfHair")
hair_bsdf.component = "Reflection"
hair_output = hair_nodes.new("ShaderNodeOutputMaterial")
hair_links.new(object_info.outputs["Object Index"], map_range.inputs["Value"])
hair_links.new(map_range.outputs["Result"], color_ramp.inputs["Fac"])
hair_links.new(color_ramp.outputs["Color"], hair_bsdf.inputs["Color"])
hair_links.new(object_info.outputs["Random"], hair_bsdf.inputs["RoughnessU"])
hair_links.new(hair_bsdf.outputs["BSDF"], hair_output.inputs["Surface"])

for member in MEMBERS:
    source = bpy.data.objects.new(member["name"], curve_data)
    source.location = (0.0, 0.0, HAIR_Z)
    source.color = member["color"]
    source.pass_index = member["pass_index"]
    for flag in ("visible_camera", "visible_diffuse", "visible_glossy", "visible_transmission",
                 "visible_volume_scatter", "visible_shadow"):
        setattr(source, flag, member[flag])
    source.data.materials.append(hair_mat)
    source_collection = bpy.data.collections.new(member["collection"])
    source_collection.objects.link(source)
    instancer = bpy.data.objects.new(member["instance"], None)
    scene.collection.objects.link(instancer)
    instancer.instance_type = "COLLECTION"
    instancer.instance_collection = source_collection
    instancer.location = member["location"]
# Deterministic patch centres: the centre spline is directly visible; its shadow is offset by the
# analytically computed sun displacement. No pixel search or post-render probe selection is used.
def pixel_for_world(value):
    return int(round((value / ORTHO_SCALE + 0.5) * WIDTH))

probes = [
    {"name": "red_hair_hit", "pixel": [pixel_for_world(-1.3), HEIGHT // 2], "radius": PATCH_RADIUS},
    {"name": "blue_hair_hit", "pixel": [pixel_for_world(1.3), HEIGHT // 2], "radius": PATCH_RADIUS},
    {"name": "red_shadow_visible", "pixel": [pixel_for_world(-1.3), pixel_for_world(SHADOW_Y)], "radius": PATCH_RADIUS},
    {"name": "blue_shadow_masked", "pixel": [pixel_for_world(1.3), pixel_for_world(SHADOW_Y)], "radius": PATCH_RADIUS},
    {"name": "lit_receiver", "pixel": [WIDTH // 2, pixel_for_world(SHADOW_Y)], "radius": PATCH_RADIUS},
]

blend_path = os.path.join(out_dir, "hair_instance_context18.blend")
exr_path = os.path.join(out_dir, "hair_instance_context18.exr")
scene.render.filepath = os.path.join(out_dir, "hair_instance_context18")
member_names = {member["name"] for member in MEMBERS}

def members_from_render_depsgraph(depsgraph):
    if depsgraph.mode != "RENDER":
        raise RuntimeError(f"expected Cycles RENDER depsgraph, got {depsgraph.mode}")
    members = []
    for instance in depsgraph.object_instances:
        obj = instance.object
        if obj.type != "CURVES" or not instance.is_instance:
            continue
        real_object = instance.instance_object
        if real_object.name not in member_names:
            continue
        members.append({
            "name": real_object.name,
            "parent": instance.parent.name if instance.parent else None,
            "persistent_id": [int(value) for value in instance.persistent_id],
            "is_instance": bool(instance.is_instance),
            "use_particle_hair": False,
            "object_to_world": [[float(instance.matrix_world[row][col]) for col in range(4)] for row in range(4)],
            "object_color": [float(value) for value in obj.color],
            "source_object_color": [float(value) for value in real_object.color],
            "object_pass_index": int(obj.pass_index),
            "object_random_id": int(instance.random_id),
            "slot_0_material": obj.material_slots[0].material.name if obj.material_slots and obj.material_slots[0].material else None,
            "ray_visibility": {
                "camera": bool(obj.visible_camera),
                "diffuse": bool(obj.visible_diffuse),
                "glossy": bool(obj.visible_glossy),
                "transmission": bool(obj.visible_transmission),
                "volume_scatter": bool(obj.visible_volume_scatter),
                "shadow": bool(obj.visible_shadow),
            },
        })
    if len(members) != 2 or {member["name"] for member in members} != member_names:
        raise RuntimeError(f"expected exactly the two real collection-instanced Hair objects, got {members}")
    if len({member["object_random_id"] for member in members}) != 2:
        raise RuntimeError(f"physical collection instances did not expose distinct Cycles random IDs: {members}")
    return members

bpy.ops.wm.save_as_mainfile(filepath=blend_path)
from cycles import engine as cycles_engine
captured_members = []
original_cycles_render = cycles_engine.render
def capture_render(_engine, depsgraph):
    captured_members.append(members_from_render_depsgraph(depsgraph))
try:
    cycles_engine.render = capture_render
    view_layer = bpy.context.view_layer
    with bpy.context.temp_override(scene=scene, view_layer=view_layer):
        result = bpy.ops.render.render(write_still=False, layer=view_layer.name)
    if "FINISHED" not in result:
        raise RuntimeError("Cycles render-depsgraph capture did not finish")
finally:
    cycles_engine.render = original_cycles_render
if not captured_members or any(members != captured_members[0] for members in captured_members[1:]):
    raise RuntimeError("Cycles render depsgraph did not produce consistent Hair instance members")
actual_members = captured_members[0]
bpy.ops.render.render(write_still=True)

image = bpy.data.images.load(exr_path, check_existing=False)
pixels = array.array("f", [0.0]) * (WIDTH * HEIGHT * 4)
image.pixels.foreach_get(pixels)

def patch_mean(probe):
    x, y = probe["pixel"]
    radius = probe["radius"]
    total = [0.0] * 4
    count = 0
    for py in range(y - radius, y + radius + 1):
        for px in range(x - radius, x + radius + 1):
            offset = (py * WIDTH + px) * 4
            for channel in range(4):
                total[channel] += pixels[offset + channel]
            count += 1
    return [value / count for value in total]



for probe in probes:
    probe["linear_rgba"] = patch_mean(probe)
probe_by_name = {probe["name"]: probe["linear_rgba"] for probe in probes}
if not (probe_by_name["red_hair_hit"][0] > probe_by_name["red_hair_hit"][2]):
    raise RuntimeError(f"red Hair hit lost Object Info pass-index context: {probe_by_name}")
if not (probe_by_name["blue_hair_hit"][2] > probe_by_name["blue_hair_hit"][0]):
    raise RuntimeError(f"blue Hair hit lost Object Info pass-index context: {probe_by_name}")
if not (probe_by_name["red_shadow_visible"][0] < 0.8 * probe_by_name["lit_receiver"][0]):
    raise RuntimeError(f"shadow-visible Hair member did not darken its fixed receiver patch: {probe_by_name}")
if not (probe_by_name["blue_shadow_masked"][0] > 0.8 * probe_by_name["lit_receiver"][0]):
    raise RuntimeError(f"shadow-masked Hair member unexpectedly darkened its fixed receiver patch: {probe_by_name}")

build_hash = bpy.app.build_hash
if isinstance(build_hash, bytes):
    build_hash = build_hash.decode("ascii")
manifest = {
    "blender_version": bpy.app.version_string,
    "blender_build_hash": build_hash,
    "engine": scene.render.engine,
    "device": scene.cycles.device,
    "resolution": [WIDTH, HEIGHT],
    "resolution_percentage": scene.render.resolution_percentage,
    "samples": SAMPLES,
    "seed": SEED,
    "denoising": scene.cycles.use_denoising,
    "max_bounces": scene.cycles.max_bounces,
    "diffuse_bounces": scene.cycles.diffuse_bounces,
    "glossy_bounces": scene.cycles.glossy_bounces,
    "transmission_bounces": scene.cycles.transmission_bounces,
    "volume_bounces": scene.cycles.volume_bounces,
    "transparent_max_bounces": scene.cycles.transparent_max_bounces,
    "pixel_filter": scene.cycles.pixel_filter_type,
    "filter_width": scene.cycles.filter_width,
    "cycles_curves": {
        "shape": scene.cycles_curves.shape.lower(),
        "subdivisions": int(scene.cycles_curves.subdivisions),
    },
    "view_transform": scene.view_settings.view_transform,
    "view_look": scene.view_settings.look,
    "exposure": scene.view_settings.exposure,
    "gamma": scene.view_settings.gamma,
    "render": {
        "film_transparent": scene.render.film_transparent,
        "threads_mode": scene.render.threads_mode,
        "threads": scene.render.threads,
        "image_format": scene.render.image_settings.file_format,
        "image_color_mode": scene.render.image_settings.color_mode,
        "image_color_depth": scene.render.image_settings.color_depth,
        "exr_codec": scene.render.image_settings.exr_codec,
    },
    "protocol": "CPU render; one fixed seed/sample count; 32-bit linear RGBA OPEN_EXR ZIP; no denoising; patch coordinates are analytically fixed from camera projection and sun displacement.",
    "world": {
        "background_color": list(background.inputs["Color"].default_value),
        "background_strength": background.inputs["Strength"].default_value,
    },
    "linear_exr": "hair_instance_context18.exr",
    "scene": {
        "camera": {"type": camera_data.type, "ortho_scale": camera_data.ortho_scale, "location": list(camera.location), "rotation_euler": list(camera.rotation_euler)},
        "receiver": {"name": floor.name, "diffuse_color": [0.72, 0.72, 0.72, 1.0], "z": -0.05},
        "sun": {"name": sun.name, "energy": sun_data.energy, "angle": sun_data.angle, "rotation_euler": list(sun.rotation_euler), "shadow_offset_y": SHADOW_Y},
        "shared_hair_data": {
            "name": curve_data.name,
            "spline_type": "CATMULL_ROM",
            "curve_count": 3,
            "key_positions": CURVE_POINTS,
            "radius": CURVE_RADIUS,
            "material": hair_mat.name,
            "closure": "ShaderNodeBsdfHair",
            "closure_component": hair_bsdf.component,
            "object_info_output_to_hair_color": "Object Index -> Map Range -> constant Color Ramp",
            "pass_index_range": [MEMBERS[0]["pass_index"], MEMBERS[1]["pass_index"]],
            "object_info_color_used": False,
            "object_info_random_used": True,
            "object_info_random_to_hair_roughness_u": True,
            "hair_bsdf_inputs": {
                socket.name: (
                    [float(value) for value in socket.default_value]
                    if hasattr(socket.default_value, "__iter__")
                    else float(socket.default_value)
                )
                for socket in hair_bsdf.inputs
                if hasattr(socket, "default_value")
            },
            "map_range": {
                "from_min": map_range.inputs["From Min"].default_value,
                "from_max": map_range.inputs["From Max"].default_value,
                "to_min": map_range.inputs["To Min"].default_value,
                "to_max": map_range.inputs["To Max"].default_value,
            },
            "color_ramp": {
                "interpolation": color_ramp.color_ramp.interpolation,
                "elements": [
                    {
                        "position": element.position,
                        "color": [float(value) for value in element.color],
                    }
                    for element in color_ramp.color_ramp.elements
                ],
            },
        },
        "members": actual_members,
    },
    "probes": probes,
}
with open(os.path.join(out_dir, "hair_instance_context18.json"), "w", encoding="utf-8") as stream:
    json.dump(manifest, stream, indent=2)
    stream.write("\n")
bpy.data.images.remove(image)
