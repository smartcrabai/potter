"""Generate the committed Blender 5.2.2 native BSDF scenes and linear EXR references."""
import bpy
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
OUT = sys.argv[sys.argv.index("--") + 1]

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"expected Blender 5.2.2, got {bpy.app.version_string}")
bpy.context.preferences.filepaths.save_version = 0

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


def make_scene(name, cases, engine, output_prefix):
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
        if case[0] == "ray_portal":
            x, y = 0.0, 0.0
        tile = bpy.data.meshes.new(f"BackdropTile{index}")
        tile.from_pydata([(-0.58, -0.58, 0.0), (0.58, -0.58, 0.0),
                          (0.58, 0.58, 0.0), (-0.58, 0.58, 0.0)], [], [(0, 1, 2, 3)])
        tile_obj = bpy.data.objects.new(f"BackdropTile{index}", tile)
        scene.collection.objects.link(tile_obj)
        tile_obj.location = (x, y, 0.0)
        tile_obj.data.materials.append(tile_material(index))

        material, inputs = new_material(case)
        case_name, node_type, properties = case
        if node_type in {"ShaderNodeBsdfHair", "ShaderNodeBsdfHairPrincipled"}:
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
        probes.append({"name": case_name, "x": px, "y": py, "patch_radius": PATCH_RADIUS,
                       "mode": properties, "inputs": inputs})

    bpy.context.view_layer.objects.active = scene.camera
    for obj in scene.objects:
        obj.select_set(False)
    scene.render.filepath = os.path.join(OUT, f"{output_prefix}.exr")
    blend_path = os.path.join(OUT, f"{output_prefix}.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    bpy.ops.render.render(write_still=True)

    image = bpy.data.images.load(scene.render.filepath, check_existing=False)
    width, height = image.size[:]
    pixels = array("f", [0.0]) * (width * height * 4)
    image.pixels.foreach_get(pixels)
    cases_json = []
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
        case = {"name": probe["name"], "mode": probe["mode"], "inputs": probe["inputs"],
                "probe": [probe["x"], probe["y"], probe["patch_radius"]],
                "linear_rgba": [value / count for value in totals]}
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
        for probe, first_case in zip(probes, cases_json):
            second_total = [0.0, 0.0, 0.0, 0.0]
            count = 0
            for dy in range(-probe["patch_radius"], probe["patch_radius"] + 1):
                for dx in range(-probe["patch_radius"], probe["patch_radius"] + 1):
                    x, y = probe["x"] + dx, probe["y"] + dy
                    source_index = ((height - 1 - y) * width + x) * 4
                    for channel in range(4):
                        second_total[channel] += float(second_pixels[source_index + channel])
                    count += 1
            second_mean = [value / count for value in second_total]
            noise_cases.append({"name": probe["name"],
                                "mean_absolute_error": sum(abs(a - b) for a, b in
                                                           zip(first_case["linear_rgba"], second_mean)) / 4.0})
        with open(os.path.join(OUT, "path_noise.json"), "w", encoding="utf-8") as stream:
            json.dump({"blender_version": bpy.app.version_string, "cases": noise_cases}, stream, indent=2)
            stream.write("\n")
        bpy.data.images.remove(second)


os.makedirs(OUT, exist_ok=True)
make_scene("Shader BSDF Path Wave12", CASES, "CYCLES", "path")
make_scene("Shader BSDF EEVEE Wave12", EEVEE_CASES, "BLENDER_EEVEE", "eevee")
for backup in ("path.blend1", "eevee.blend1"):
    backup_path = os.path.join(OUT, backup)
    if os.path.exists(backup_path):
        os.remove(backup_path)
