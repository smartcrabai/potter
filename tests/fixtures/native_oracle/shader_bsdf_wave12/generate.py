"""Generate the committed Blender 5.2.2 native BSDF scenes and linear EXR references."""
import bpy
import hashlib
import json
import math
import os
import sys
from array import array
from mathutils import Vector

SIZE = 192
ORTHO = 8.0
CELL = 1.25
COLS = 6
PATCH_RADIUS = 1
CYCLES_SAMPLES = 256
EEVEE_SAMPLES = 64
SEED = 17
CLI_ARGS = sys.argv[sys.argv.index("--") + 1 :]
SUN_ONLY = "--sun-only" in CLI_ARGS
EEVEE_SHEEN_ONLY = "--eevee-sheen-only" in CLI_ARGS
EEVEE_PREVIEW_ONLY = "--eevee-preview-only" in CLI_ARGS
SKIN_CROP_ONLY = "--skin-crop-only" in CLI_ARGS
TOON_SUN_ONLY = "--toon-sun-only" in CLI_ARGS
EEVEE_SSS_TRANSMISSION_ONLY = "--eevee-sss-transmission-only" in CLI_ARGS
EEVEE_REFRACTION_LAYERS_ONLY = "--eevee-refraction-layers-only" in CLI_ARGS
PATH_ONLY = "--path-only" in CLI_ARGS
PATH_SEEDS_ONLY = "--path-seeds-only" in CLI_ARGS
OUT = CLI_ARGS[-1]
SKIN_CROP_SIZE = 4
SKIN_SAMPLES = 4096
PATH_EXTRA_SEEDS = (53, 61, 71, 83, 97, 113)
PATH_SEEDS = (SEED, 29) + PATH_EXTRA_SEEDS

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"expected Blender 5.2.2, got {bpy.app.version_string}")
bpy.context.preferences.filepaths.save_version = 0
bpy.context.preferences.view.use_translate_new_dataname = False

CASES = [
    ("sheen_ashikhmin", "ShaderNodeBsdfSheen", {"distribution": "ASHIKHMIN"}),
    ("sheen_microfiber", "ShaderNodeBsdfSheen", {"distribution": "MICROFIBER"}),
    ("subsurface_burley", "ShaderNodeSubsurfaceScattering", {"falloff": "BURLEY"}),
    ("subsurface_random_walk", "ShaderNodeSubsurfaceScattering", {"falloff": "RANDOM_WALK"}),
    ("subsurface_random_walk_skin", "ShaderNodeSubsurfaceScattering", {"falloff": "RANDOM_WALK_SKIN"}),
    ("subsurface_random_walk_legacy", "ShaderNodeSubsurfaceScattering", {"falloff": "RANDOM_WALK_LEGACY"}),
    ("translucent", "ShaderNodeBsdfTranslucent", {}),
    ("refraction_beckmann", "ShaderNodeBsdfRefraction", {"distribution": "BECKMANN"}),
    ("refraction_ggx", "ShaderNodeBsdfRefraction", {"distribution": "GGX"}),
    ("toon_diffuse", "ShaderNodeBsdfToon", {"component": "DIFFUSE"}),
    ("toon_glossy", "ShaderNodeBsdfToon", {"component": "GLOSSY"}),
    ("metallic_beckmann_f82", "ShaderNodeBsdfMetallic", {"distribution": "BECKMANN", "fresnel_type": "F82"}),
    ("metallic_ggx_f82", "ShaderNodeBsdfMetallic", {"distribution": "GGX", "fresnel_type": "F82"}),
    ("metallic_multi_ggx_f82", "ShaderNodeBsdfMetallic", {"distribution": "MULTI_GGX", "fresnel_type": "F82"}),
    ("metallic_beckmann_conductor", "ShaderNodeBsdfMetallic", {"distribution": "BECKMANN", "fresnel_type": "PHYSICAL_CONDUCTOR"}),
    ("metallic_ggx_conductor", "ShaderNodeBsdfMetallic", {"distribution": "GGX", "fresnel_type": "PHYSICAL_CONDUCTOR"}),
    ("metallic_multi_ggx_conductor", "ShaderNodeBsdfMetallic", {"distribution": "MULTI_GGX", "fresnel_type": "PHYSICAL_CONDUCTOR"}),
    ("holdout", "ShaderNodeHoldout", {}),
    ("hair_reflection", "ShaderNodeBsdfHair", {"component": "Reflection"}),
    ("hair_transmission", "ShaderNodeBsdfHair", {"component": "Transmission"}),
    ("principled_hair_chiang_reflectance", "ShaderNodeBsdfHairPrincipled", {"model": "CHIANG", "parametrization": "COLOR"}),
    ("principled_hair_chiang_melanin", "ShaderNodeBsdfHairPrincipled", {"model": "CHIANG", "parametrization": "MELANIN"}),
    ("principled_hair_chiang_absorption", "ShaderNodeBsdfHairPrincipled", {"model": "CHIANG", "parametrization": "ABSORPTION"}),
    ("principled_hair_huang_reflectance", "ShaderNodeBsdfHairPrincipled", {"model": "HUANG", "parametrization": "COLOR"}),
    ("principled_hair_huang_melanin", "ShaderNodeBsdfHairPrincipled", {"model": "HUANG", "parametrization": "MELANIN"}),
    ("principled_hair_huang_absorption", "ShaderNodeBsdfHairPrincipled", {"model": "HUANG", "parametrization": "ABSORPTION"}),
    ("ray_portal", "ShaderNodeBsdfRayPortal", {}),
]
EEVEE_CASES = [case for case in CASES if case[1] in {
    "ShaderNodeSubsurfaceScattering", "ShaderNodeBsdfTranslucent",
    "ShaderNodeBsdfRefraction", "ShaderNodeBsdfMetallic", "ShaderNodeHoldout",
}]

EEVEE_PREVIEW_CASES = [
    ("toon_diffuse_eevee", "ShaderNodeBsdfToon", {"component": "DIFFUSE"}),
    ("toon_glossy_eevee", "ShaderNodeBsdfToon", {"component": "GLOSSY"}),
    ("hair_eevee", "ShaderNodeBsdfHair", {"component": "Transmission"}),
    ("principled_hair_eevee", "ShaderNodeBsdfHairPrincipled",
     {"model": "HUANG", "parametrization": "ABSORPTION"}),
    ("ray_portal_eevee", "ShaderNodeBsdfRayPortal", {}),
]


def set_input(node, name, value):
    socket = node.inputs.get(name)
    if socket is not None and hasattr(socket, "default_value"):
        socket.default_value = value


def set_material_inputs(node, node_type):
    color = (0.72, 0.31, 0.12, 1.0)
    if node_type == "ShaderNodeBsdfSheen":
        values = {"Color": color, "Roughness": 0.43, "Normal": (0.06, 0.02, 0.998)}
    elif node_type == "ShaderNodeSubsurfaceScattering":
        values = {"Color": color, "Radius": (0.85, 0.31, 0.12), "Scale": 0.24,
                  "Roughness": 0.36, "IOR": 1.38, "Anisotropy": 0.22,
                  "Normal": (0.06, 0.02, 0.998)}
    elif node_type == "ShaderNodeBsdfTranslucent":
        values = {"Color": color, "Normal": (0.06, 0.02, 0.998)}
    elif node_type == "ShaderNodeBsdfRefraction":
        values = {"Color": (0.74, 0.88, 0.96, 1.0), "Roughness": 0.13, "IOR": 1.43,
                  "Normal": (0.06, 0.02, 0.998)}
    elif node_type == "ShaderNodeBsdfToon":
        values = {"Color": color, "Size": 0.38, "Smooth": 0.17,
                  "Normal": (0.06, 0.02, 0.998)}
    elif node_type == "ShaderNodeBsdfMetallic":
        values = {"Base Color": (0.72, 0.31, 0.12, 1.0), "Edge Tint": (0.84, 0.63, 0.43, 1.0),
                  "Roughness": 0.27, "Anisotropy": 0.32, "Rotation": 0.21,
                  "Thin Film Thickness": 280.0, "Thin Film IOR": 1.34,
                  "IOR": (2.4, 2.8, 3.1), "Extinction": (3.2, 2.5, 2.1),
                  "Normal": (0.06, 0.02, 0.998)}
    elif node_type == "ShaderNodeBsdfHair":
        values = {"Color": color, "Offset": 0.037, "RoughnessU": 0.24,
                  "RoughnessV": 0.51, "Tangent": (1.0, 0.15, 0.0)}
    elif node_type == "ShaderNodeBsdfHairPrincipled":
        values = {"Color": (0.34, 0.12, 0.07, 1.0), "Tint": (0.88, 0.54, 0.31, 1.0),
                  "Melanin": 0.52, "Melanin Redness": 0.64,
                  "Absorption Coefficient": (0.31, 0.73, 1.27),
                  "Aspect Ratio": 0.73, "Roughness": 0.34, "Radial Roughness": 0.29,
                  "Coat": 0.23, "IOR": 1.48, "Offset": 0.031,
                  "Random Color": 0.12, "Random Roughness": 0.14, "Random": 0.37,
                  "Reflection": 0.83, "Transmission": 0.71, "Secondary Reflection": 0.62,
                  "Tangent": (1.0, 0.15, 0.0)}
    elif node_type == "ShaderNodeBsdfRayPortal":
        values = {"Color": (0.92, 0.87, 0.73, 1.0), "Position": (0.0, 0.0, 2.0),
                  "Direction": (0.684, 0.0, -0.73)}
    else:
        values = {}
    for name, value in values.items():
        set_input(node, name, value)
    return values


def new_material(case):
    name, node_type, properties = case
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    bsdf = tree.nodes.new(node_type)
    bsdf.location = (0, 0)
    output.location = (260, 0)
    for key, value in properties.items():
        if hasattr(bsdf, key):
            setattr(bsdf, key, value)
        else:
            raise RuntimeError(f"{node_type} lacks mode property {key}")
    inputs = set_material_inputs(bsdf, node_type)
    tree.links.new(bsdf.outputs[0], output.inputs["Surface"])
    return material, inputs


def clear_scene(scene):
    bpy.context.window.scene = scene
    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for material in list(bpy.data.materials):
        bpy.data.materials.remove(material)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def camera_and_lighting(scene):
    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0, 0, 12)
    camera.rotation_euler = (0, 0, 0)
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = ORTHO
    scene.camera = camera

    world = bpy.data.worlds.new("World") if scene.world is None else scene.world
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes.clear()
    background = world.node_tree.nodes.new("ShaderNodeBackground")
    background.inputs["Color"].default_value = (0.035, 0.055, 0.11, 1.0)
    background.inputs["Strength"].default_value = 0.45
    output = world.node_tree.nodes.new("ShaderNodeOutputWorld")
    world.node_tree.links.new(background.outputs["Background"], output.inputs["Surface"])

    lights = [((-3.2, -1.5, 5.0), (1.0, 0.22, 0.08), 850.0),
              ((2.8, 2.5, 4.4), (0.10, 0.48, 1.0), 1050.0),
              ((0.1, 3.8, 3.7), (1.0, 0.82, 0.31), 620.0)]
    for number, (location, color, energy) in enumerate(lights):
        data = bpy.data.lights.new(f"Area{number}", type="AREA")
        data.energy = energy
        data.shape = "RECTANGLE"
        data.size = 2.0
        data.size_y = 0.8
        data.color = color
        obj = bpy.data.objects.new(f"Area{number}", data)
        scene.collection.objects.link(obj)
        obj.location = location
        obj.rotation_euler = (Vector((0, 0, 0)) - obj.location).to_track_quat("-Z", "Y").to_euler()


def tile_material(index):
    colors = [(0.10, 0.24, 0.68, 1.0), (0.72, 0.11, 0.06, 1.0),
              (0.09, 0.61, 0.23, 1.0), (0.67, 0.40, 0.07, 1.0),
              (0.41, 0.10, 0.65, 1.0), (0.06, 0.51, 0.56, 1.0)]
    mat = bpy.data.materials.new(f"Backdrop{index}")
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    emission = mat.node_tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = colors[index % len(colors)]
    emission.inputs["Strength"].default_value = 0.72 + 0.13 * (index % 3)
    output = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
    mat.node_tree.links.new(emission.outputs[0], output.inputs["Surface"])
    return mat


def make_hair_curve(name, location, material):
    data = bpy.data.hair_curves.new(name)
    data.add_curves([4])
    data.set_types(type="CATMULL_ROM")
    for point, co in zip(data.points, [(-0.38, 0.0, 0.30), (-0.14, 0.05, 0.48),
                                       (0.14, -0.03, 0.52), (0.38, 0.0, 0.30)]):
        point.position = co
        point.radius = 0.11
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location
    obj.data.materials.append(material)
    return obj


def build_scene(name, cases, engine, hair_on_mesh=False, link_normal_input=False,
                independent_backdrop_layout=False):
    scene = bpy.context.scene
    clear_scene(scene)
    scene.name = name
    scene.render.engine = engine
    scene.render.resolution_x = SIZE
    scene.render.resolution_y = SIZE
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "32"
    scene.render.image_settings.exr_codec = "ZIP"
    scene.render.resolution_percentage = 100
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 1
    scene.render.use_file_extension = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.display_settings.display_device = "sRGB"
    scene.camera = None
    camera_and_lighting(scene)
    if engine == "CYCLES":
        scene.cycles.device = "CPU"
        scene.cycles.samples = CYCLES_SAMPLES
        scene.cycles.seed = SEED
        scene.cycles.pixel_filter_type = "BOX"
        scene.cycles.filter_width = 1.0
        scene.cycles.use_denoising = False
        scene.cycles.max_bounces = 8
        scene.cycles.transparent_max_bounces = 8
        # Potter has no light tree; with it on, Blender adds a one-sided excess versus Potter's direct light sampling.
        scene.cycles.use_light_tree = False
    else:
        scene.eevee.taa_render_samples = EEVEE_SAMPLES
        if hasattr(scene.eevee, "use_raytracing"):
            scene.eevee.use_raytracing = True

    rows = math.ceil(len(cases) / COLS)
    probes = []
    for index, case in enumerate(cases):
        column, row = index % COLS, index // COLS
        x = (column - (COLS - 1) / 2.0) * CELL
        y = ((rows - 1) / 2.0 - row) * CELL
        tile = bpy.data.meshes.new(f"BackdropTile{index}")
        tile.from_pydata([(-0.58, -0.58, 0.0), (0.58, -0.58, 0.0),
                          (0.58, 0.58, 0.0), (-0.58, 0.58, 0.0)], [], [(0, 1, 2, 3)])
        tile_obj = bpy.data.objects.new(f"BackdropTile{index}", tile)
        scene.collection.objects.link(tile_obj)
        tile_obj.location = (x, y, 0.0)
        tile_obj.data.materials.append(tile_material(index))
        if case[0] == "ray_portal":
            if not independent_backdrop_layout:
                tile_obj.location = (0.0, 0.0, 0.0)
            x, y = 0.0, 0.0

        material, inputs = new_material(case)
        case_name, node_type, properties = case
        normal_linked = False
        normal_value = inputs.get("Normal")
        if link_normal_input and normal_value is not None:
            bsdf = next(node for node in material.node_tree.nodes if node.bl_idname == node_type)
            normal_socket = bsdf.inputs.get("Normal")
            if normal_socket is not None:
                normal_node = material.node_tree.nodes.new("ShaderNodeCombineXYZ")
                normal_node.inputs["X"].default_value = normal_value[0]
                normal_node.inputs["Y"].default_value = normal_value[1]
                normal_node.inputs["Z"].default_value = normal_value[2]
                material.node_tree.links.new(normal_node.outputs["Vector"], normal_socket)
                normal_linked = True
        if not hair_on_mesh and node_type in {"ShaderNodeBsdfHair", "ShaderNodeBsdfHairPrincipled"}:
            obj = make_hair_curve(case_name, (x, y, 0.0), material)
        else:
            bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=16, radius=0.47,
                                                   location=(x, y, 0.48))
            obj = bpy.context.object
            obj.name = case_name
            obj.data.materials.append(material)
            for polygon in obj.data.polygons:
                polygon.use_smooth = True
        px = round((x / ORTHO + 0.5) * SIZE - 0.5)
        py = round((0.5 - y / ORTHO) * SIZE - 0.5)
        probe = {"name": case_name, "x": px, "y": py, "patch_radius": PATCH_RADIUS,
                 "mode": properties, "inputs": inputs}
        if normal_linked:
            probe["normal_linked"] = True
        probes.append(probe)

    bpy.context.view_layer.objects.active = scene.camera
    for obj in scene.objects:
        obj.select_set(False)
    return scene, probes


def patch_means(pixels, width, height, probes):
    means = []
    for probe in probes:
        totals = [0.0, 0.0, 0.0, 0.0]
        count = 0
        for dy in range(-probe["patch_radius"], probe["patch_radius"] + 1):
            for dx in range(-probe["patch_radius"], probe["patch_radius"] + 1):
                x, y = probe["x"] + dx, probe["y"] + dy
                source_index = ((height - 1 - y) * width + x) * 4
                for channel in range(4):
                    totals[channel] += float(pixels[source_index + channel])
                count += 1
        means.append([value / count for value in totals])
    return means


def load_pixels(path, width, height):
    image = bpy.data.images.load(path, check_existing=False)
    if tuple(image.size) != (width, height):
        raise RuntimeError(f"{path} is {tuple(image.size)}, expected {(width, height)}")
    pixels = array("f", [0.0]) * (width * height * 4)
    image.pixels.foreach_get(pixels)
    bpy.data.images.remove(image)
    return pixels


def make_scene(name, cases, engine, output_prefix, hair_on_mesh=False, link_normal_input=False,
               independent_backdrop_layout=False):
    scene, probes = build_scene(name, cases, engine, hair_on_mesh, link_normal_input,
                                independent_backdrop_layout)
    scene.render.filepath = os.path.join(OUT, f"{output_prefix}.exr")
    blend_path = os.path.join(OUT, f"{output_prefix}.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    bpy.ops.render.render(write_still=True)

    image = bpy.data.images.load(scene.render.filepath, check_existing=False)
    width, height = image.size[:]
    pixels = array("f", [0.0]) * (width * height * 4)
    image.pixels.foreach_get(pixels)
    cases_json = []
    for probe, mean in zip(probes, patch_means(pixels, width, height, probes)):
        case = {"name": probe["name"], "mode": probe["mode"], "inputs": probe["inputs"],
                "probe": [probe["x"], probe["y"], probe["patch_radius"]],
                "linear_rgba": mean}
        if probe.get("normal_linked"):
            case["normal_linked"] = True
        if probe["name"] == "holdout":
            source_index = ((height - 1 - probe["y"]) * width + probe["x"] + 13) * 4
            case["neighbor_rgba"] = [float(pixels[source_index + channel]) for channel in range(4)]
        cases_json.append(case)
    capture = {"blender_version": bpy.app.version_string, "engine": engine,
               "resolution": [width, height], "samples": CYCLES_SAMPLES if engine == "CYCLES" else EEVEE_SAMPLES,
               "seed": SEED, "linear_exr": os.path.basename(scene.render.filepath),
               "cases": cases_json}
    with open(os.path.join(OUT, f"{output_prefix}.json"), "w", encoding="utf-8") as stream:
        json.dump(capture, stream, indent=2)
        stream.write("\n")
    bpy.data.images.remove(image)

    if engine == "CYCLES":
        scene.cycles.seed = 29
        scene.render.filepath = os.path.join(OUT, "path_seed29.exr")
        bpy.ops.render.render(write_still=True)
        second = bpy.data.images.load(scene.render.filepath, check_existing=False)
        second_pixels = array("f", [0.0]) * (width * height * 4)
        second.pixels.foreach_get(second_pixels)
        noise_cases = []
        second_means = patch_means(second_pixels, width, height, probes)
        for probe, first_case, second_mean in zip(probes, cases_json, second_means):
            noise_cases.append({"name": probe["name"],
                                "mean_absolute_error": sum(abs(a - b) for a, b in
                                                           zip(first_case["linear_rgba"], second_mean)) / 4.0})
        with open(os.path.join(OUT, "path_noise.json"), "w", encoding="utf-8") as stream:
            json.dump({"blender_version": bpy.app.version_string, "cases": noise_cases}, stream, indent=2)
            stream.write("\n")
        bpy.data.images.remove(second)


def make_path_seeds_capture(scratch):
    scene, probes = build_scene("Shader BSDF Path Wave12", CASES, "CYCLES",
                                independent_backdrop_layout=True)
    width, height = SIZE, SIZE
    cycles = scene.cycles
    settings = {"device": cycles.device, "samples": cycles.samples,
                "use_adaptive_sampling": cycles.use_adaptive_sampling,
                "pixel_filter_type": cycles.pixel_filter_type, "filter_width": cycles.filter_width,
                "use_denoising": cycles.use_denoising, "max_bounces": cycles.max_bounces,
                "transparent_max_bounces": cycles.transparent_max_bounces,
                "threads_mode": scene.render.threads_mode, "threads": scene.render.threads}
    seed_pixels = {}
    for seed in PATH_SEEDS:
        scene.cycles.seed = seed
        scene.render.filepath = os.path.join(scratch, f"path_seed{seed}.exr")
        bpy.ops.render.render(write_still=True)
        seed_pixels[seed] = load_pixels(scene.render.filepath, width, height)
    seed_means = {seed: patch_means(seed_pixels[seed], width, height, probes) for seed in PATH_SEEDS}

    # The committed seed 17/29 captures must be reproduced exactly before the extra seeds are trusted.
    fixture_pixels = {SEED: load_pixels(os.path.join(OUT, "path.exr"), width, height),
                      29: load_pixels(os.path.join(OUT, "path_seed29.exr"), width, height)}
    with open(os.path.join(OUT, "path.json"), encoding="utf-8") as stream:
        committed = [case["linear_rgba"] for case in json.load(stream)["cases"]]
    fixture_means = {SEED: committed, 29: patch_means(fixture_pixels[29], width, height, probes)}
    reproduction = {}
    for seed in (SEED, 29):
        exact = seed_means[seed] == fixture_means[seed]
        if not exact:
            raise RuntimeError(f"seed {seed} patch means do not reproduce the committed capture")
        reproduction[str(seed)] = {
            "patch_means_exact": exact,
            "max_abs_pixel_difference": max(abs(a - b) for a, b in zip(seed_pixels[seed], fixture_pixels[seed])),
        }

    build_hash = bpy.app.build_hash
    if isinstance(build_hash, bytes):
        build_hash = build_hash.decode("ascii")
    cases_json = [{"name": probe["name"], "probe": [probe["x"], probe["y"], probe["patch_radius"]],
                   "seed_captures": [{"seed": seed, "linear_rgba": seed_means[seed][index]} for seed in PATH_SEEDS]}
                  for index, probe in enumerate(probes)]
    capture = {
        "blender_version": bpy.app.version_string,
        "blender_build_hash": build_hash,
        "blender_binary": bpy.app.binary_path,
        "blender_binary_sha256": file_sha256(bpy.app.binary_path),
        "generator": "blender --background --factory-startup --python generate.py -- --path-seeds-only <scratch-directory> <output-directory>",
        "generator_sha256": file_sha256(os.path.abspath(__file__)),
        "engine": "CYCLES",
        "resolution": [width, height],
        "samples": CYCLES_SAMPLES,
        "seeds": list(PATH_SEEDS),
        "cycles": settings,
        "reproduction": {"seeds": [SEED, 29],
                         "compared_against": {"17": "path.json linear_rgba", "29": "path_seed29.exr patch means"},
                         "results": reproduction},
        "cases": cases_json,
    }
    with open(os.path.join(OUT, "path_seeds.json"), "w", encoding="utf-8") as stream:
        json.dump(capture, stream, indent=2)
        stream.write("\n")


def make_eevee_sheen_scenes():
    for distribution in ("ASHIKHMIN", "MICROFIBER"):
        case = ("sheen_eevee", "ShaderNodeBsdfSheen", {"distribution": distribution})
        output_prefix = f"eevee_sheen_{distribution.lower()}"
        make_scene(
            "Shader BSDF EEVEE Sheen",
            [case],
            "BLENDER_EEVEE",
            output_prefix,
            link_normal_input=True,
        )

def make_eevee_preview_scene():
    make_scene(
        "Shader BSDF EEVEE GPU Previews",
        EEVEE_PREVIEW_CASES,
        "BLENDER_EEVEE",
        "eevee_preview",
        hair_on_mesh=True,
        link_normal_input=True,
    )

def make_eevee_sss_transmission_scene():
    scene = bpy.context.scene
    clear_scene(scene)
    scene.name = "Shader BSDF EEVEE Burley SSS Backface Direct"
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 64
    scene.render.resolution_y = 64
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "32"
    scene.render.image_settings.exr_codec = "ZIP"
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 1
    scene.render.use_file_extension = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.display_settings.display_device = "sRGB"
    scene.eevee.taa_render_samples = EEVEE_SAMPLES
    if hasattr(scene.eevee, "use_raytracing"):
        scene.eevee.use_raytracing = True

    world = bpy.data.worlds.new("Black World")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes.clear()
    background = world.node_tree.nodes.new("ShaderNodeBackground")
    background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    background.inputs["Strength"].default_value = 0.0
    output = world.node_tree.nodes.new("ShaderNodeOutputWorld")
    world.node_tree.links.new(background.outputs["Background"], output.inputs["Surface"])

    camera_data = bpy.data.cameras.new("Camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 4.0
    camera = bpy.data.objects.new("Camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0.0, 0.0, 4.0)
    scene.camera = camera

    material = bpy.data.materials.new("Burley SSS Transmission")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    sss = tree.nodes.new("ShaderNodeSubsurfaceScattering")
    sss.falloff = "BURLEY"
    node_inputs = {
        "Color": (0.8, 0.8, 0.8, 1.0),
        "Weight": 1.0,
        "Scale": 0.24,
        "Radius": (1.0, 0.7, 0.4),
        "IOR": 1.4,
        "Roughness": 0.36,
        "Anisotropy": 0.0,
    }
    for name, value in node_inputs.items():
        set_input(sss, name, value)
    tree.links.new(sss.outputs[0], output.inputs["Surface"])

    thickness = tree.nodes.new("ShaderNodeValue")
    thickness.outputs["Value"].default_value = 0.0
    tree.links.new(thickness.outputs["Value"], output.inputs["Thickness"])

    mesh = bpy.data.meshes.new("Thin Surface")
    mesh.from_pydata([(-1.4, -1.4, 0.0), (1.4, -1.4, 0.0),
                      (1.4, 1.4, 0.0), (-1.4, 1.4, 0.0)], [], [(0, 1, 2, 3)])
    surface = bpy.data.objects.new("Burley SSS Surface", mesh)
    scene.collection.objects.link(surface)
    surface.data.materials.append(material)

    sun_data = bpy.data.lights.new("Backside Sun", type="SUN")
    sun_data.energy = 4.0
    sun_data.color = (1.0, 1.0, 1.0)
    sun_data.angle = 0.0
    sun = bpy.data.objects.new("Backside Sun", sun_data)
    scene.collection.objects.link(sun)
    sun.rotation_euler = (math.pi, 0.0, 0.0)

    output_path = os.path.join(OUT, "eevee_sss_transmission.exr")
    scene.render.filepath = output_path
    blend_path = os.path.join(OUT, "eevee_sss_transmission.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    bpy.ops.render.render(write_still=True)

    image = bpy.data.images.load(output_path, check_existing=False)
    width, height = image.size[:]
    pixels = array("f", [0.0]) * (width * height * 4)
    image.pixels.foreach_get(pixels)
    probe = [width // 2, height // 2, 1]
    totals = [0.0, 0.0, 0.0, 0.0]
    count = 0
    for dy in range(-probe[2], probe[2] + 1):
        for dx in range(-probe[2], probe[2] + 1):
            source_index = ((height - 1 - (probe[1] + dy)) * width + probe[0] + dx) * 4
            for channel in range(4):
                totals[channel] += float(pixels[source_index + channel])
            count += 1
    corner_probe = [2, 2]
    corner_index = ((height - 1 - corner_probe[1]) * width + corner_probe[0]) * 4
    corner_rgba = [float(pixels[corner_index + channel]) for channel in range(4)]
    build_hash = bpy.app.build_hash
    if isinstance(build_hash, bytes):
        build_hash = build_hash.decode("ascii")
    capture = {
        "blender_version": bpy.app.version_string,
        "blender_build_hash": build_hash,
        "engine": scene.render.engine,
        "resolution": [width, height],
        "samples": EEVEE_SAMPLES,
        "seed": SEED,
        "linear_exr": os.path.basename(output_path),
        "source": "Blender 5.2.2 d13 eevee_deferred_eval.bsl.hh:183-201",
        "world": {"background_color": [0.0, 0.0, 0.0, 1.0], "strength": 0.0},
        "light": {"type": "SUN", "energy": sun_data.energy, "rotation_euler": [math.pi, 0.0, 0.0]},
        "surface": {"normal": [0.0, 0.0, 1.0], "thickness": 0.0,
                    "thickness_node": {"type": "ShaderNodeValue", "value": 0.0}},
        "cases": [{
            "name": "burley_sss_backface_direct",
            "mode": {"node_type": sss.bl_idname, "falloff": sss.falloff},
            "inputs": node_inputs,
            "probe": probe,
            "linear_rgba": [value / count for value in totals],
            "corner_probe": corner_probe,
            "corner_rgba": corner_rgba,
        }],
    }
    with open(os.path.join(OUT, "eevee_sss_transmission.json"), "w", encoding="utf-8") as stream:
        json.dump(capture, stream, indent=2)
        stream.write("\n")
    bpy.data.images.remove(image)



def make_eevee_refraction_layers_scene():
    scene = bpy.context.scene
    clear_scene(scene)
    scene.name = "EEVEE Screen Refraction Layer Feedback"
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 128
    scene.render.resolution_y = 64
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "32"
    scene.render.image_settings.exr_codec = "ZIP"
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 1
    scene.render.use_file_extension = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.display_settings.display_device = "sRGB"
    scene.eevee.taa_render_samples = EEVEE_SAMPLES
    scene.eevee.use_raytracing = True
    scene.eevee.ray_tracing_method = "SCREEN"
    ray_tracing = scene.eevee.ray_tracing_options
    ray_tracing.resolution_scale = "1"
    ray_tracing.screen_trace_quality = 1.0
    ray_tracing.screen_trace_thickness = 0.1
    ray_tracing.trace_max_roughness = 0.5

    world = bpy.data.worlds.new("Black World")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes.clear()
    background = world.node_tree.nodes.new("ShaderNodeBackground")
    background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    background.inputs["Strength"].default_value = 0.0
    output = world.node_tree.nodes.new("ShaderNodeOutputWorld")
    world.node_tree.links.new(background.outputs["Background"], output.inputs["Surface"])

    camera_data = bpy.data.cameras.new("Camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 8.0
    camera = bpy.data.objects.new("Camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0.0, 0.0, 4.0)
    scene.camera = camera

    def emission_material(name, color, screen_refraction=False):
        material = bpy.data.materials.new(name)
        material.use_nodes = True
        material.use_raytrace_refraction = screen_refraction
        tree = material.node_tree
        tree.nodes.clear()
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.inputs["Color"].default_value = (*color, 1.0)
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
        return material

    def rectangle(name, bounds, z, material, x_slope=0.0):
        left, bottom, right, top = bounds
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(
            [
                (left, bottom, z + left * x_slope),
                (right, bottom, z + right * x_slope),
                (right, top, z + right * x_slope),
                (left, top, z + left * x_slope),
            ],
            [],
            [(0, 1, 2, 3)],
        )
        obj = bpy.data.objects.new(name, mesh)
        scene.collection.objects.link(obj)
        obj.data.materials.append(material)
        return obj

    base = emission_material("Opaque red emissive backplate", (0.8, 0.025, 0.01))
    rectangle("Opaque red backplate", (-4.0, -2.0, 4.0, 2.0), -0.5, base)

    emission = emission_material(
        "Flagged emission without transmission", (0.02, 0.8, 0.08), True
    )
    rectangle("Flagged emission under enabled glass", (-3.35, -0.5, -2.45, 0.5), 0.4, emission)
    rectangle("Visible flagged emission control", (-0.45, 0.55, 0.45, 1.45), 0.6, emission)

    principled = bpy.data.materials.new("Flagged Principled with zero transmission")
    principled.use_nodes = True
    principled.use_raytrace_refraction = True
    shader = principled.node_tree.nodes.get("Principled BSDF")
    if shader is None:
        raise RuntimeError("Principled BSDF node is missing")
    set_input(shader, "Base Color", (0.12, 0.04, 0.8, 1.0))
    set_input(shader, "Roughness", 1.0)
    set_input(shader, "Transmission Weight", 0.0)
    set_input(shader, "Emission Color", (0.12, 0.04, 0.8, 1.0))
    set_input(shader, "Emission Strength", 1.0)
    rectangle(
        "Flagged Principled under enabled glass",
        (-2.35, -0.5, -1.45, 0.5),
        0.4,
        principled,
    )
    rectangle("Visible flagged Principled control", (-0.45, -1.45, 0.45, -0.55), 0.6, principled)

    sss = bpy.data.materials.new("Opaque Burley SSS feedback target")
    sss.use_nodes = True
    sss.use_raytrace_refraction = False
    sss.node_tree.nodes.clear()
    sss_output = sss.node_tree.nodes.new("ShaderNodeOutputMaterial")
    sss_node = sss.node_tree.nodes.new("ShaderNodeSubsurfaceScattering")
    sss_node.falloff = "BURLEY"
    sss_inputs = {
        "Color": (0.12, 0.45, 0.9, 1.0),
        "Weight": 1.0,
        "Scale": 0.24,
        "Radius": (0.25, 0.65, 1.0),
        "IOR": 1.4,
        "Roughness": 0.36,
        "Anisotropy": 0.0,
    }
    for name, value in sss_inputs.items():
        set_input(sss_node, name, value)
    sss.node_tree.links.new(sss_node.outputs[0], sss_output.inputs["Surface"])
    rectangle("Resolved opaque SSS target", (-1.25, -0.7, -0.55, 0.7), 0.4, sss)
    rectangle("Visible opaque SSS control", (-0.45, -0.35, 0.45, 0.35), 0.4, sss)

    sun_data = bpy.data.lights.new("Front Sun", type="SUN")
    sun_data.energy = 4.0
    sun_data.color = (1.0, 1.0, 1.0)
    sun_data.angle = 0.0
    sun_data.use_shadow = False
    sun = bpy.data.objects.new("Front Sun", sun_data)
    scene.collection.objects.link(sun)
    sun.rotation_euler = (0.0, 0.0, 0.0)

    def glass_material(name, screen_refraction):
        material = bpy.data.materials.new(name)
        material.use_nodes = True
        material.use_raytrace_refraction = screen_refraction
        material.thickness_mode = "SLAB"
        material.node_tree.nodes.clear()
        glass = material.node_tree.nodes.new("ShaderNodeBsdfRefraction")
        glass.distribution = "GGX"
        glass.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        glass.inputs["Roughness"].default_value = 0.05
        glass.inputs["IOR"].default_value = 1.45
        glass.inputs["Normal"].default_value = (0.0, 0.0, 1.0)
        output = material.node_tree.nodes.new("ShaderNodeOutputMaterial")
        output.inputs["Thickness"].default_value = 0.25
        material.node_tree.links.new(glass.outputs[0], output.inputs["Surface"])
        return material

    glass_enabled = glass_material("Enabled screen-refraction glass", True)
    glass_disabled = glass_material("Disabled screen-refraction glass", False)
    rectangle("Enabled glass pane", (-3.8, -1.8, -0.5, 1.8), 1.5, glass_enabled, 0.25)
    rectangle("Disabled glass pane", (0.5, -1.8, 3.8, 1.8), 1.5, glass_disabled, 0.25)

    output_path = os.path.join(OUT, "eevee_refraction_layers.exr")
    blend_path = os.path.join(OUT, "eevee_refraction_layers.blend")
    scene.render.filepath = output_path
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    bpy.ops.render.render(write_still=True)

    image = bpy.data.images.load(output_path, check_existing=False)
    width, height = image.size[:]
    pixels = array("f", [0.0]) * (width * height * 4)
    image.pixels.foreach_get(pixels)
    cases = []
    probes = [
        ("enabled_glass_over_flagged_emission", [17, 32, 1], "enabled glass must see red opaque BACK"),
        ("enabled_glass_over_principled_transmission_zero", [33, 32, 1], "enabled glass must see red opaque BACK"),
        ("enabled_glass_over_resolved_sss", [49, 32, 1], "enabled glass must see the resolved opaque SSS feedback"),
        ("disabled_glass_uses_world", [96, 32, 1], "disabled glass uses the black World probe"),
        ("flagged_emission_visible", [64, 17, 1], "flagged nontransmissive emission remains visible"),
        ("flagged_principled_visible", [64, 46, 1], "flagged Principled transmission=0 remains visible"),
        ("opaque_sss_visible", [64, 32, 1], "opaque SSS source target is visibly distinct"),
        ("red_backplate_visible", [126, 32, 1], "unoccluded emissive backplate color"),
    ]
    for name, probe, behavior in probes:
        x, y, radius = probe
        total = [0.0, 0.0, 0.0, 0.0]
        count = 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                index = ((height - 1 - (y + dy)) * width + x + dx) * 4
                for channel in range(4):
                    total[channel] += float(pixels[index + channel])
                count += 1
        cases.append({
            "name": name,
            "probe": probe,
            "world_location": [
                -4.0 + (x + 0.5) * 8.0 / width,
                2.0 - (y + 0.5) * 4.0 / height,
            ],
            "behavior": behavior,
            "linear_rgba": [value / count for value in total],
        })
    build_hash = bpy.app.build_hash
    if isinstance(build_hash, bytes):
        build_hash = build_hash.decode("ascii")
    capture = {
        "blender_version": bpy.app.version_string,
        "blender_build_hash": build_hash,
        "blender_binary": bpy.app.binary_path,
        "blender_binary_sha256": file_sha256(bpy.app.binary_path),
        "generator": "blender --background --factory-startup --python generate.py -- --eevee-refraction-layers-only <output-directory>",
        "engine": scene.render.engine,
        "resolution": [width, height],
        "samples": scene.eevee.taa_render_samples,
        "seed": SEED,
        "linear_exr": os.path.basename(output_path),
        "pixel_format": "32-bit RGBA OpenEXR ZIP",
        "render": {
            "resolution_percentage": scene.render.resolution_percentage,
            "film_transparent": scene.render.film_transparent,
            "threads_mode": scene.render.threads_mode,
            "threads": scene.render.threads,
            "frame": scene.frame_current,
        },
        "view": {
            "transform": scene.view_settings.view_transform,
            "look": scene.view_settings.look,
            "exposure": scene.view_settings.exposure,
            "gamma": scene.view_settings.gamma,
        },
        "raytracing": {
            "enabled": scene.eevee.use_raytracing,
            "method": scene.eevee.ray_tracing_method,
            "resolution_scale": ray_tracing.resolution_scale,
            "screen_trace_quality": ray_tracing.screen_trace_quality,
            "screen_trace_thickness": ray_tracing.screen_trace_thickness,
            "trace_max_roughness": ray_tracing.trace_max_roughness,
        },
        "world": {"background_color": [0.0, 0.0, 0.0, 1.0], "strength": 0.0},
        "camera": {
            "projection": "ORTHO",
            "location": [0.0, 0.0, 4.0],
            "ortho_scale": 8.0,
            "bounds": [-4.0, 4.0, -2.0, 2.0],
        },
        "geometry": {
            "enabled_glass_bounds": [-3.8, -1.8, -0.5, 1.8],
            "disabled_glass_bounds": [0.5, -1.8, 3.8, 1.8],
            "glass_z": 1.5,
            "glass_x_slope": 0.25,
            "backplate_bounds": [-4.0, -2.0, 4.0, 2.0],
            "backplate_z": -0.5,
            "flagged_emission_under_glass_bounds": [-3.35, -0.5, -2.45, 0.5],
            "flagged_principled_under_glass_bounds": [-2.35, -0.5, -1.45, 0.5],
            "resolved_sss_bounds": [-1.25, -0.7, -0.55, 0.7],
            "visible_opaque_sss_bounds": [-0.45, -0.35, 0.45, 0.35],
            "unoccluded_backplate_bounds": [3.8, -2.0, 4.0, 2.0],
            "opaque_target_z": 0.4,
        },
        "materials": {
            "enabled_glass": {"name": glass_enabled.name, "node": "ShaderNodeBsdfRefraction", "use_raytrace_refraction": True, "ior": 1.45, "roughness": 0.05, "normal": [0.0, 0.0, 1.0], "thickness_mode": "SLAB", "thickness": 0.25},
            "disabled_glass": {"name": glass_disabled.name, "node": "ShaderNodeBsdfRefraction", "use_raytrace_refraction": False, "ior": 1.45, "roughness": 0.05, "normal": [0.0, 0.0, 1.0], "thickness_mode": "SLAB", "thickness": 0.25},
            "flagged_emission": {"name": emission.name, "node": "ShaderNodeEmission", "use_raytrace_refraction": True, "transmission": 0.0},
            "flagged_principled": {"name": principled.name, "node": "ShaderNodeBsdfPrincipled", "use_raytrace_refraction": True, "transmission_weight": 0.0},
            "opaque_sss": {"name": sss.name, "use_raytrace_refraction": False, "node": "BURLEY"},
        },
        "light": {"type": "SUN", "energy": sun_data.energy, "angle": 0.0, "rotation_euler": [0.0, 0.0, 0.0], "use_shadow": False},
        "cases": cases,
    }
    with open(os.path.join(OUT, "eevee_refraction_layers.json"), "w", encoding="utf-8") as stream:
        json.dump(capture, stream, indent=2)
        stream.write("\n")
    bpy.data.images.remove(image)


def frame_bounds(camera, scene):
    frame = camera.data.view_frame(scene=scene)
    return (min(point.x for point in frame), max(point.x for point in frame),
            min(point.y for point in frame), max(point.y for point in frame), frame[0].z)


def pixel_local_origin(bounds, width, height, x, y):
    left, right, bottom, top, depth = bounds
    return Vector((left + (x + 0.5) * (right - left) / width,
                   top - (y + 0.5) * (top - bottom) / height, depth))


def world_ray(matrix_world, local_origin):
    origin = matrix_world @ local_origin
    direction = matrix_world.to_3x3() @ Vector((0.0, 0.0, -1.0))
    direction.normalize()
    return origin, direction


def render_skin_crop():
    blend_path = os.path.join(OUT, "path.blend")
    json_path = os.path.join(OUT, "path.json")
    with open(json_path, encoding="utf-8") as stream:
        source_capture = json.load(stream)
    source_case = next(case for case in source_capture["cases"]
                       if case["name"] == "subsurface_random_walk_skin")

    bpy.ops.wm.open_mainfile(filepath=blend_path)
    scene = bpy.context.scene
    camera = scene.camera
    if camera is None or camera.data.type != "ORTHO":
        raise RuntimeError("path scene must have an orthographic camera")
    if scene.render.engine != "CYCLES" or scene.cycles.max_bounces != 8 or \
            scene.cycles.transparent_max_bounces != 8:
        raise RuntimeError("path scene must use Cycles with 8 maximum and transparent bounces")

    original_width = source_capture["resolution"][0]
    original_height = source_capture["resolution"][1]
    probe_x, probe_y, patch_radius = source_case["probe"]
    bpy.context.view_layer.update()
    source_bounds = frame_bounds(camera, scene)
    source_matrix = camera.matrix_world.copy()
    source_scale = camera.data.ortho_scale

    scene.render.resolution_x = SKIN_CROP_SIZE
    scene.render.resolution_y = SKIN_CROP_SIZE
    camera.data.ortho_scale = source_scale * SKIN_CROP_SIZE / original_width
    crop_bounds = frame_bounds(camera, scene)
    source_first = pixel_local_origin(source_bounds, original_width, original_height,
                                      probe_x - patch_radius, probe_y - patch_radius)
    crop_first = pixel_local_origin(crop_bounds, SKIN_CROP_SIZE, SKIN_CROP_SIZE, 0, 0)
    shift = camera.matrix_world.to_3x3() @ Vector(
        (source_first.x - crop_first.x, source_first.y - crop_first.y, 0.0))
    camera.location += shift
    bpy.context.view_layer.update()

    max_origin_delta = 0.0
    for dy in range(patch_radius * 2 + 1):
        for dx in range(patch_radius * 2 + 1):
            source_local = Vector((
                source_first.x + dx * (source_bounds[1] - source_bounds[0]) / original_width,
                source_first.y - dy * (source_bounds[3] - source_bounds[2]) / original_height,
                source_bounds[4]))
            source_ray = world_ray(source_matrix, source_local)
            crop_local = pixel_local_origin(crop_bounds, SKIN_CROP_SIZE, SKIN_CROP_SIZE, dx, dy)
            crop_ray = world_ray(camera.matrix_world, crop_local)
            delta = max(abs(a - b) for a, b in zip(source_ray[0], crop_ray[0]))
            max_origin_delta = max(max_origin_delta, delta)
            if tuple(source_ray[1]) != tuple(crop_ray[1]):
                raise RuntimeError("skin crop changed a primary ray direction")
    if max_origin_delta > 3e-7:
        raise RuntimeError(f"skin crop ray origin delta {max_origin_delta} exceeds 3e-7")

    scene.cycles.samples = SKIN_SAMPLES
    scene.cycles.seed = 31
    blend_path = os.path.join(OUT, "path_skin_crop.blend")
    scene.render.filepath = os.path.join(OUT, "path_skin_crop_seed31.exr")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    for seed in (31, 43):
        exr_name = f"path_skin_crop_seed{seed}.exr"
        scene.cycles.seed = seed
        scene.render.filepath = os.path.join(OUT, exr_name)
        bpy.ops.render.render(write_still=True)
        image = bpy.data.images.load(scene.render.filepath, check_existing=False)
        width, height = image.size[:]
        pixels = array("f", [0.0]) * (width * height * 4)
        image.pixels.foreach_get(pixels)
        totals = [0.0, 0.0, 0.0, 0.0]
        for y in range(patch_radius * 2 + 1):
            for x in range(patch_radius * 2 + 1):
                source_index = ((height - 1 - y) * width + x) * 4
                for channel in range(4):
                    totals[channel] += float(pixels[source_index + channel])
        count = (patch_radius * 2 + 1) ** 2
        capture_case = {
            "name": source_case["name"],
            "mode": source_case["mode"],
            "inputs": source_case["inputs"],
            "probe": [1, 1, 1],
            "linear_rgba": [value / count for value in totals],
        }
        capture = {
            "blender_version": bpy.app.version_string,
            "engine": "CYCLES",
            "resolution": [width, height],
            "samples": SKIN_SAMPLES,
            "seed": seed,
            "linear_exr": exr_name,
            "cases": [capture_case],
        }
        with open(os.path.join(OUT, f"path_skin_crop_seed{seed}.json"), "w",
                  encoding="utf-8") as stream:
            json.dump(capture, stream, indent=2)
            stream.write("\n")
        bpy.data.images.remove(image)
    print(f"skin crop ray proof max world-origin delta: {max_origin_delta:.9g}")



def make_sheen_sun_scene():
    scene = bpy.context.scene
    clear_scene(scene)
    scene.name = "Shader BSDF Ashikhmin Direct Control"
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = SIZE
    scene.render.resolution_y = SIZE
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "32"
    scene.render.image_settings.exr_codec = "ZIP"
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 1
    scene.render.use_file_extension = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.cycles.device = "CPU"
    scene.cycles.samples = CYCLES_SAMPLES
    scene.cycles.seed = SEED
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 1.0
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = 8
    scene.cycles.transparent_max_bounces = 8

    world = bpy.data.worlds.new("Black World")
    world.use_nodes = True
    world.node_tree.nodes.clear()
    background = world.node_tree.nodes.new("ShaderNodeBackground")
    background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    background.inputs["Strength"].default_value = 0.0
    output = world.node_tree.nodes.new("ShaderNodeOutputWorld")
    world.node_tree.links.new(background.outputs["Background"], output.inputs["Surface"])
    scene.world = world

    mesh = bpy.data.meshes.new("Direct Control Plane")
    mesh.from_pydata(
        [(-32.0, -32.0, 0.0), (32.0, -32.0, 0.0), (32.0, 32.0, 0.0), (-32.0, 32.0, 0.0)],
        [],
        [(0, 1, 2, 3)],
    )
    plane = bpy.data.objects.new("Direct Control Plane", mesh)
    scene.collection.objects.link(plane)
    plane.rotation_euler[1] = math.radians(60.0)
    case = ("sheen_ashikhmin_sun", "ShaderNodeBsdfSheen", {"distribution": "ASHIKHMIN"})
    material, inputs = new_material(case)
    plane.data.materials.append(material)

    sun_data = bpy.data.lights.new("Sun", type="SUN")
    sun_data.energy = 100.0
    sun_data.angle = 0.0
    sun = bpy.data.objects.new("Sun", sun_data)
    scene.collection.objects.link(sun)
    sun.rotation_euler = (0.0, 0.0, 0.0)

    camera_data = bpy.data.cameras.new("Camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = ORTHO
    camera = bpy.data.objects.new("Camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0.0, 0.0, 12.0)
    camera.rotation_euler = (0.0, 0.0, 0.0)
    scene.camera = camera

    output_prefix = "sheen_ashikhmin_sun"
    scene.render.filepath = os.path.join(OUT, output_prefix)
    bpy.ops.wm.save_as_mainfile(filepath=scene.render.filepath + ".blend")
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(scene.render.filepath + ".exr", check_existing=False)
    width, height = image.size[:]
    pixels = array("f", [0.0]) * (width * height * 4)
    image.pixels.foreach_get(pixels)
    probe = {"name": output_prefix, "x": width // 2, "y": height // 2, "patch_radius": 1}
    totals = [0.0, 0.0, 0.0, 0.0]
    count = 0
    for dy in range(-probe["patch_radius"], probe["patch_radius"] + 1):
        for dx in range(-probe["patch_radius"], probe["patch_radius"] + 1):
            x, y = probe["x"] + dx, probe["y"] + dy
            source_index = ((height - 1 - y) * width + x) * 4
            for channel in range(4):
                totals[channel] += float(pixels[source_index + channel])
            count += 1
    capture = {
        "blender_version": bpy.app.version_string,
        "engine": "CYCLES",
        "resolution": [width, height],
        "samples": CYCLES_SAMPLES,
        "seed": SEED,
        "linear_exr": output_prefix + ".exr",
        "cases": [{
            "name": output_prefix,
            "mode": case[2],
            "inputs": inputs,
            "probe": [probe["x"], probe["y"], probe["patch_radius"]],
            "linear_rgba": [value / count for value in totals],
        }],
    }
    with open(os.path.join(OUT, output_prefix + ".json"), "w", encoding="utf-8") as stream:
        json.dump(capture, stream, indent=2)
        stream.write("\n")
    bpy.data.images.remove(image)

def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_toon_sun_scenes():
    canonical_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "path.blend")
    canonical_sha256 = file_sha256(canonical_path)
    blender_binary = bpy.app.binary_path
    regimes = [
        ("plateau", 0.0),
        ("mid_smooth", (0.38 + 0.17 * 0.5) * math.pi / 2.0),
        ("outside", (0.38 + 0.17 + 0.1) * math.pi / 2.0),
    ]
    seeds = (31, 43)
    cases = []
    for regime, theta in regimes:
        bpy.ops.wm.open_mainfile(filepath=canonical_path)
        scene = bpy.context.scene
        source_obj = bpy.data.objects.get("toon_diffuse")
        if source_obj is None or not source_obj.data.materials:
            raise RuntimeError("canonical path.blend has no toon_diffuse material object")
        material = source_obj.data.materials[0]
        toon = next((node for node in material.node_tree.nodes
                     if node.bl_idname == "ShaderNodeBsdfToon"), None)
        if toon is None or toon.component != "DIFFUSE":
            raise RuntimeError("canonical Toon Diffuse node is missing or not DIFFUSE")
        expected_inputs = {
            "Color": (0.72, 0.31, 0.12, 1.0),
            "Size": 0.38,
            "Smooth": 0.17,
            "Normal": (0.06, 0.02, 0.998),
        }
        inputs = {}
        for name, expected in expected_inputs.items():
            socket = toon.inputs.get(name)
            if socket is None:
                raise RuntimeError(f"canonical Toon Diffuse input {name} is missing")
            if name == "Normal" and socket.is_linked:
                raise RuntimeError("canonical Toon Diffuse Normal input must remain unlinked")
            value = socket.default_value
            inputs[name] = ([float(channel) for channel in value]
                            if hasattr(value, "__len__") else float(value))
            if isinstance(expected, tuple):
                if any(abs(actual - target) > 1e-6 for actual, target in zip(inputs[name], expected)):
                    raise RuntimeError(f"canonical Toon Diffuse input {name} differs: {inputs[name]!r}")
            elif abs(inputs[name] - expected) > 1e-6:
                raise RuntimeError(f"canonical Toon Diffuse input {name} differs: {inputs[name]!r}")

        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        for old_material in list(bpy.data.materials):
            if old_material != material:
                bpy.data.materials.remove(old_material)
        for old_world in list(bpy.data.worlds):
            bpy.data.worlds.remove(old_world)
        world = bpy.data.worlds.new("Diagnostic Black World")
        world.use_nodes = True
        world.node_tree.nodes.clear()
        background = world.node_tree.nodes.new("ShaderNodeBackground")
        background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
        background.inputs["Strength"].default_value = 0.0
        world_output = world.node_tree.nodes.new("ShaderNodeOutputWorld")
        world.node_tree.links.new(background.outputs["Background"], world_output.inputs["Surface"])
        scene.world = world

        mesh = bpy.data.meshes.new("Diagnostic Toon Plane Mesh")
        mesh.from_pydata(
            [(-4.0, -4.0, 0.0), (4.0, -4.0, 0.0), (4.0, 4.0, 0.0), (-4.0, 4.0, 0.0)],
            [],
            [(0, 1, 2, 3)],
        )
        mesh.update()
        plane = bpy.data.objects.new("Diagnostic Toon Plane", mesh)
        scene.collection.objects.link(plane)
        plane.data.materials.append(material)
        sun_data = bpy.data.lights.new("Diagnostic Sun", type="SUN")
        sun_data.energy = 1.0
        sun_data.angle = 0.0
        sun = bpy.data.objects.new("Diagnostic Sun", sun_data)
        scene.collection.objects.link(sun)
        sun.rotation_euler = (0.0, theta, 0.0)
        camera_data = bpy.data.cameras.new("Diagnostic Orthographic Camera")
        camera_data.type = "ORTHO"
        camera_data.ortho_scale = 4.0
        camera = bpy.data.objects.new("Diagnostic Orthographic Camera", camera_data)
        scene.collection.objects.link(camera)
        camera.location = (0.0, 0.0, 12.0)
        camera.rotation_euler = (0.0, 0.0, 0.0)
        scene.camera = camera
        bpy.context.view_layer.update()

        scene.name = f"Toon Diffuse Direct SUN {regime}"
        scene.render.engine = "CYCLES"
        scene.render.resolution_x = 4
        scene.render.resolution_y = 4
        scene.render.resolution_percentage = 100
        scene.render.film_transparent = False
        scene.render.image_settings.file_format = "OPEN_EXR"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.image_settings.color_depth = "32"
        scene.render.image_settings.exr_codec = "ZIP"
        scene.render.threads_mode = "FIXED"
        scene.render.threads = 2
        scene.render.use_file_extension = True
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
        scene.cycles.device = "CPU"
        scene.cycles.samples = 32
        scene.cycles.seed = seeds[0]
        scene.cycles.pixel_filter_type = "BOX"
        scene.cycles.filter_width = 1.0
        scene.cycles.use_denoising = False
        scene.cycles.max_bounces = 8
        scene.cycles.transparent_max_bounces = 8
        ray = sun.matrix_world.to_3x3() @ Vector((0.0, 0.0, -1.0))
        expected_ray = Vector((-math.sin(theta), 0.0, -math.cos(theta)))
        if (ray - expected_ray).length > 1e-7:
            raise RuntimeError(f"Sun orientation mismatch for {regime}: {tuple(ray)}")
        if len(scene.objects) != 3 or sorted(obj.type for obj in scene.objects) != ["CAMERA", "LIGHT", "MESH"]:
            raise RuntimeError(f"unexpected objects in Toon SUN scene: {[obj.name for obj in scene.objects]}")
        if tuple(plane.data.polygons[0].normal) != (0.0, 0.0, 1.0):
            raise RuntimeError("Toon SUN plane normal is not +Z")

        name = f"toon_diffuse_sun_{regime}"
        blend_name = name + ".blend"
        blend_path = os.path.join(OUT, blend_name)
        scene.render.filepath = os.path.join(OUT, name + "_seed31")
        bpy.ops.wm.save_as_mainfile(filepath=blend_path)
        seed_captures = []
        for seed in seeds:
            scene.cycles.seed = seed
            exr_name = f"{name}_seed{seed}.exr"
            exr_path = os.path.join(OUT, exr_name)
            scene.render.filepath = os.path.join(OUT, name + f"_seed{seed}")
            bpy.ops.render.render(write_still=True)
            image = bpy.data.images.load(exr_path, check_existing=False)
            width, height = image.size[:]
            pixels = array("f", [0.0]) * (width * height * 4)
            image.pixels.foreach_get(pixels)
            rgba = [[float(pixels[index + channel]) for channel in range(4)]
                    for index in range(0, len(pixels), 4)]
            mean = [sum(pixel[channel] for pixel in rgba) / len(rgba) for channel in range(4)]
            field_range = [max(pixel[channel] for pixel in rgba) - min(pixel[channel] for pixel in rgba)
                           for channel in range(4)]
            seed_captures.append({
                "seed": seed,
                "exr": exr_name,
                "linear_rgba": mean,
                "field_range_rgba": field_range,
            })
            bpy.data.images.remove(image)
        cases.append({
            "name": name,
            "regime": regime,
            "theta_radians": theta,
            "theta_degrees": math.degrees(theta),
            "mode": {"component": toon.component},
            "inputs": inputs,
            "normal_linked": bool(toon.inputs["Normal"].is_linked),
            "probe": [1, 1, 1],
            "blend": blend_name,
            "blend_sha256": file_sha256(blend_path),
            "seed_captures": seed_captures,
        })

    metadata = {
        "blender_version": bpy.app.version_string,
        "blender_build_hash": (bpy.app.build_hash.decode("ascii")
                               if isinstance(bpy.app.build_hash, bytes) else str(bpy.app.build_hash)),
        "blender_binary": blender_binary,
        "blender_sha256": file_sha256(blender_binary),
        "generator": "generate.py --toon-sun-only",
        "engine": "CYCLES",
        "resolution": [4, 4],
        "samples": 32,
        "seeds": list(seeds),
        "cycles": {
            "device": "CPU",
            "threads_mode": "FIXED",
            "threads": 2,
            "denoising": False,
            "pixel_filter": "BOX",
            "filter_width": 1.0,
            "max_bounces": 8,
            "transparent_max_bounces": 8,
        },
        "pixel_filter": "BOX",
        "filter_width": 1.0,
        "view_transform": "Standard",
        "look": "None",
        "exposure": 0.0,
        "gamma": 1.0,
        "linear_exr": "32-bit RGBA OpenEXR",
        "canonical_material_source": {"blend": "path.blend", "sha256": canonical_sha256, "case": "toon_diffuse"},
        "world": {"background_color": [0.0, 0.0, 0.0, 1.0], "strength": 0.0},
        "sun": {"type": "SUN", "energy": 1.0, "angular_diameter_radians": 0.0},
        "camera": {"projection": "ORTHO", "location": [0.0, 0.0, 12.0], "ortho_scale": 4.0},
        "plane": {"size": [8.0, 8.0], "location": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, 1.0]},
        "cases": cases,
    }
    with open(os.path.join(OUT, "toon_diffuse_sun.json"), "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")


def remove_blend_backups(names):
    for name in names:
        backup_path = os.path.join(OUT, name + ".blend1")
        if os.path.exists(backup_path):
            os.remove(backup_path)


os.makedirs(OUT, exist_ok=True)
if PATH_ONLY:
    make_scene("Shader BSDF Path Wave12", CASES, "CYCLES", "path",
               independent_backdrop_layout=True)
    remove_blend_backups(("path",))
elif PATH_SEEDS_ONLY:
    make_path_seeds_capture(CLI_ARGS[-2])
elif EEVEE_REFRACTION_LAYERS_ONLY:
    make_eevee_refraction_layers_scene()
    remove_blend_backups(("eevee_refraction_layers",))
elif EEVEE_PREVIEW_ONLY:
    make_eevee_preview_scene()
    remove_blend_backups(("eevee_preview",))
elif EEVEE_SHEEN_ONLY:
    make_eevee_sheen_scenes()
    remove_blend_backups(("eevee_sheen_ashikhmin", "eevee_sheen_microfiber"))
elif SKIN_CROP_ONLY:
    render_skin_crop()
elif TOON_SUN_ONLY:
    make_toon_sun_scenes()
    remove_blend_backups(("toon_diffuse_sun_plateau", "toon_diffuse_sun_mid_smooth", "toon_diffuse_sun_outside"))
elif SUN_ONLY:
    make_sheen_sun_scene()
else:
    make_scene("Shader BSDF Path Wave12", CASES, "CYCLES", "path",
               independent_backdrop_layout=True)
    render_skin_crop()
    make_scene("Shader BSDF EEVEE Wave12", EEVEE_CASES, "BLENDER_EEVEE", "eevee")
    make_eevee_sheen_scenes()
    make_eevee_preview_scene()
    make_sheen_sun_scene()
    make_toon_sun_scenes()
    remove_blend_backups((
        "path", "eevee", "sheen_ashikhmin_sun", "eevee_sheen_ashikhmin",
        "eevee_sheen_microfiber", "eevee_preview", "toon_diffuse_sun_plateau",
        "toon_diffuse_sun_mid_smooth", "toon_diffuse_sun_outside",
    ))
