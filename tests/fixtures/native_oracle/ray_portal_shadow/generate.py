"""Generate Blender 5.2.2 Cycles mesh-shadow references for Ray Portal closures.

Run with: blender --background --factory-startup --python generate.py -- <output-directory>
"""

import json
import math
import os
import sys

import bpy

SIZE = 4
SAMPLES = 32
SEEDS = (31, 43)
PORTAL_COLOR = (0.8, 0.4, 0.2, 1.0)

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"expected Blender 5.2.2, got {bpy.app.version_string}")

build_hash = bpy.app.build_hash
if isinstance(build_hash, bytes):
    build_hash = build_hash.decode("ascii")
if build_hash != "d13f752e3b9c":
    raise RuntimeError(f"expected Blender build d13f752e3b9c, got {build_hash}")

arguments = sys.argv[sys.argv.index("--") + 1 :]
if len(arguments) != 1:
    raise RuntimeError("usage: generate.py -- <output-directory>")
out_dir = os.path.abspath(arguments[0])
os.makedirs(out_dir, exist_ok=True)
bpy.context.preferences.filepaths.save_version = 0
bpy.ops.wm.read_factory_settings(use_empty=True)

scene = bpy.context.scene
scene.name = "Ray Portal Mesh Shadow"
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = SAMPLES
scene.cycles.seed = SEEDS[0]
scene.cycles.use_denoising = False
scene.cycles.max_bounces = 1
scene.render.resolution_x = SIZE
scene.render.resolution_y = SIZE
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_mode = "RGBA"
scene.render.image_settings.color_depth = "32"
scene.render.image_settings.exr_codec = "ZIP"
scene.render.use_file_extension = True
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0

world = bpy.data.worlds.new("BlackWorld")
world.use_nodes = True
background = world.node_tree.nodes.get("Background")
background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
background.inputs["Strength"].default_value = 0.0
scene.world = world

floor_material = bpy.data.materials.new("WhiteDiffuseFloor")
floor_material.use_nodes = True
floor_material.use_transparent_shadow = True
floor_tree = floor_material.node_tree
floor_tree.nodes.clear()
floor_diffuse = floor_tree.nodes.new("ShaderNodeBsdfDiffuse")
floor_diffuse.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
floor_diffuse.inputs["Roughness"].default_value = 0.0
floor_output = floor_tree.nodes.new("ShaderNodeOutputMaterial")
floor_tree.links.new(floor_diffuse.outputs["BSDF"], floor_output.inputs["Surface"])
bpy.ops.mesh.primitive_plane_add(size=8.0, location=(0.0, 0.0, 0.0))
floor = bpy.context.object
floor.name = "Floor"
floor.data.materials.append(floor_material)

sun_data = bpy.data.lights.new("Sun", type="SUN")
sun_data.energy = 1.0
sun_data.angle = 0.0
sun = bpy.data.objects.new("Sun", sun_data)
scene.collection.objects.link(sun)
sun.location = (0.0, 0.0, 4.0)
sun.rotation_euler = (0.0, 0.0, 0.0)

camera_data = bpy.data.cameras.new("Camera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 1.5
camera = bpy.data.objects.new("Camera", camera_data)
scene.collection.objects.link(camera)
camera.location = (0.0, 0.0, 3.0)
camera.rotation_euler = (0.0, 0.0, 0.0)
scene.camera = camera

portal_material = bpy.data.materials.new("SingleRayPortal")
portal_material.use_nodes = True
portal_material.use_transparent_shadow = True
portal_tree = portal_material.node_tree
portal_tree.nodes.clear()
portal = portal_tree.nodes.new("ShaderNodeBsdfRayPortal")
portal.inputs["Color"].default_value = PORTAL_COLOR
portal.inputs["Direction"].default_value = (0.0, 0.0, -1.0)
portal_output = portal_tree.nodes.new("ShaderNodeOutputMaterial")
portal_tree.links.new(portal.outputs[0], portal_output.inputs["Surface"])

mixed_material = bpy.data.materials.new("HalfTransparentHalfRayPortal")
mixed_material.use_nodes = True
mixed_material.use_transparent_shadow = True
mixed_tree = mixed_material.node_tree
mixed_tree.nodes.clear()
transparent = mixed_tree.nodes.new("ShaderNodeBsdfTransparent")
transparent.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
mixed_portal = mixed_tree.nodes.new("ShaderNodeBsdfRayPortal")
mixed_portal.inputs["Color"].default_value = PORTAL_COLOR
mixed_portal.inputs["Direction"].default_value = (0.0, 0.0, -1.0)
mix = mixed_tree.nodes.new("ShaderNodeMixShader")
mix.inputs[0].default_value = 0.5
mixed_output = mixed_tree.nodes.new("ShaderNodeOutputMaterial")
mixed_tree.links.new(transparent.outputs[0], mix.inputs[1])
mixed_tree.links.new(mixed_portal.outputs[0], mix.inputs[2])
mixed_tree.links.new(mix.outputs[0], mixed_output.inputs["Surface"])

bpy.ops.mesh.primitive_plane_add(size=8.0, location=(0.0, 0.0, 1.0))
occluder = bpy.context.object
occluder.name = "RayPortalOccluder"
occluder.visible_camera = False
occluder.visible_diffuse = False
occluder.visible_glossy = False
occluder.visible_transmission = False
occluder.visible_volume_scatter = False
occluder.visible_shadow = True


def summarize(path):
    image = bpy.data.images.load(path, check_existing=False)
    width, height = image.size[:]
    pixels = list(image.pixels[:])
    rows = [pixels[index : index + 4] for index in range(0, len(pixels), 4)]
    means = [sum(pixel[channel] for pixel in rows) / len(rows) for channel in range(4)]
    stderrs = []
    for channel in range(4):
        variance = sum((pixel[channel] - means[channel]) ** 2 for pixel in rows) / (len(rows) - 1)
        stderrs.append(math.sqrt(variance / len(rows)))
    bpy.data.images.remove(image)
    return {
        "linear_exr": os.path.basename(path),
        "pixel_count": width * height,
        "mean_linear_associated_rgba": means,
        "stderr_linear_associated_rgba": stderrs,
    }


cases = {}
for name, material in (("baseline", None), ("pure_portal", portal_material), ("mixed", mixed_material)):
    occluder.hide_render = material is None
    if material is not None:
        occluder.data.materials.clear()
        occluder.data.materials.append(material)
    scene.cycles.seed = SEEDS[0]
    blend_name = f"{name}.blend"
    blend_path = os.path.join(out_dir, blend_name)
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    renders = []
    for seed in SEEDS:
        scene.cycles.seed = seed
        exr_path = os.path.join(out_dir, f"{name}-seed{seed}.exr")
        scene.render.filepath = os.path.splitext(exr_path)[0]
        bpy.ops.render.render(write_still=True)
        renders.append({"seed": seed, **summarize(exr_path)})
    cases[name] = {"blend": blend_name, "renders": renders}

ratios = []
for seed in SEEDS:
    baseline = next(item for item in cases["baseline"]["renders"] if item["seed"] == seed)
    baseline_mean = baseline["mean_linear_associated_rgba"]
    baseline_stderr = baseline["stderr_linear_associated_rgba"]
    for name in ("pure_portal", "mixed"):
        tested = next(item for item in cases[name]["renders"] if item["seed"] == seed)
        mean = tested["mean_linear_associated_rgba"]
        stderr = tested["stderr_linear_associated_rgba"]
        rgb = [mean[channel] / baseline_mean[channel] for channel in range(3)]
        ratio_stderr = [
            math.hypot(
                stderr[channel] / baseline_mean[channel],
                mean[channel] * baseline_stderr[channel] / baseline_mean[channel] ** 2,
            )
            for channel in range(3)
        ]
        ratios.append({
            "seed": seed,
            "case": name,
            "rgb": rgb,
            "stderr_rgb": ratio_stderr,
            "alpha_ratio": mean[3] / baseline_mean[3],
        })

capture = {
    "blender_version": bpy.app.version_string,
    "blender_build_hash": build_hash,
    "engine": scene.render.engine,
    "device": scene.cycles.device,
    "resolution": [SIZE, SIZE],
    "samples": SAMPLES,
    "seeds": list(SEEDS),
    "denoising": scene.cycles.use_denoising,
    "max_bounces": scene.cycles.max_bounces,
    "pixel_filter": scene.cycles.pixel_filter_type,
    "filter_width": scene.cycles.filter_width,
    "view_transform": scene.view_settings.view_transform,
    "world_color": [0.0, 0.0, 0.0],
    "world_strength": 0.0,
    "sun_energy": sun_data.energy,
    "sun_angle": sun_data.angle,
    "sun_rotation_euler": [float(value) for value in sun.rotation_euler],
    "floor_dimensions": [8.0, 8.0],
    "camera_type": camera_data.type,
    "camera_ortho_scale": camera_data.ortho_scale,
    "occluder": {
        "name": occluder.name,
        "visible_camera": occluder.visible_camera,
        "visible_shadow": occluder.visible_shadow,
    },
    "portal_color": list(PORTAL_COLOR),
    "portal_direction": [0.0, 0.0, -1.0],
    "mixed_factor": 0.5,
    "transparent_color": [1.0, 1.0, 1.0, 1.0],
    "transparent_shadow_gate": True,
    "reference_semantics": "Means and standard errors are measured across all 16 linear EXR pixels; ratio uncertainty uses first-order independent-error propagation.",
    "cases": cases,
    "ratios": ratios,
}
json_path = os.path.join(out_dir, "ray_portal_shadow.json")
with open(json_path, "w", encoding="utf-8") as stream:
    json.dump(capture, stream, indent=2)
    stream.write("\n")
print(json.dumps({
    "blender_version": capture["blender_version"],
    "blender_build_hash": capture["blender_build_hash"],
    "settings": {
        "engine": capture["engine"],
        "device": capture["device"],
        "resolution": capture["resolution"],
        "samples": capture["samples"],
        "seeds": capture["seeds"],
        "max_bounces": capture["max_bounces"],
    },
    "ratios": ratios,
}, indent=2))
