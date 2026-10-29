"""Generate the Blender 5.2.2 ShaderP1 native-render oracle.

Source references: Blender's ``source/blender/nodes/shader/nodes/`` implementations
(``node_shader_tex_checker.cc``, ``node_shader_tex_gradient.cc``, ``node_shader_tex_noise.cc``,
``node_shader_tex_voronoi.cc``, ``node_shader_tex_coord.cc``, ``node_shader_geometry.cc``,
``node_shader_volume_principled.cc``, ``node_shader_layer_weight.cc``, ``node_shader_fresnel.cc``,
and ``node_shader_attribute.cc``), EEVEE material shaders, Cycles' ``intern/cycles/scene/shader_nodes.cpp``,
and shader evaluation in ``intern/cycles/kernel/svm/``. Run with Blender 5.2.2 only:

    /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup \\
      --threads 1 --python-exit-code 3 --python generate.py -- FIXTURE_DIR

To generate only the Voronoi fold invariant fixture:

    /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup \\
      --threads 1 --python-exit-code 3 --python generate.py -- \\
      --only-voronoi-fold-invariants FIXTURE_DIR

The committed JSON values come from Blender's rendered pixels; they are not Potter output.
"""

import json
import math
import os
import sys
import tempfile

import bpy


GRID = 9
TILE_PIXELS = 16
WIDTH = GRID * TILE_PIXELS
HEIGHT = GRID * TILE_PIXELS
TILE = 1.0


def argv_output():
    try:
        args = sys.argv[sys.argv.index("--") + 1:]
    except ValueError as error:
        raise RuntimeError("expected output directory after --") from error
    if args and args[0] == "--only-voronoi-fold-invariants":
        args = args[1:]
    try:
        return args[0]
    except IndexError as error:
        raise RuntimeError("expected output directory after --") from error


def configure_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = 0
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.001
    scene.render.resolution_x = WIDTH
    scene.render.resolution_y = HEIGHT
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    camera_data = bpy.data.cameras.new("Oracle Camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = float(GRID)
    camera = bpy.data.objects.new("Oracle Camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0.0, 0.0, 20.0)
    camera.rotation_euler = (0.0, 0.0, 0.0)
    scene.camera = camera
    return scene


def make_chart_plane(name, index, material, attributes=None, reverse=False,
                     rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0)):
    col = index % GRID
    row = index // GRID
    cx = (col + 0.5) * TILE - GRID * TILE / 2.0
    cy = GRID * TILE / 2.0 - (row + 0.5) * TILE
    vertices = [(-0.5, -0.5, 0.0), (0.5, -0.5, 0.0),
                (0.5, 0.5, 0.0), (-0.5, 0.5, 0.0)]
    face = (3, 2, 1, 0) if reverse else (0, 1, 2, 3)
    mesh = bpy.data.meshes.new(name + " Mesh")
    mesh.from_pydata(vertices, [], [face])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = (cx, cy, 0.0)
    obj.rotation_euler = rotation
    obj.scale = scale
    obj.data.materials.append(material)
    if attributes:
        for attr_name, attr_type, values in attributes:
            attr = mesh.attributes.new(attr_name, attr_type, "POINT")
            for datum, value in zip(attr.data, values):
                if attr_type in {"FLOAT", "INT", "BOOLEAN"}:
                    datum.value = value
                elif attr_type == "FLOAT_VECTOR":
                    datum.vector = value
                else:
                    datum.color = value
    return obj


def emission_material(name, node_type, output_socket, configure=None):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0
    source = tree.nodes.new(node_type)
    if configure:
        configure(source, tree)
    source_output = next(socket for socket in source.outputs if socket.name == output_socket)
    emission_color = next(socket for socket in emission.inputs if socket.name == "Color")
    tree.links.new(source_output, emission_color)
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material, source


def image_texture(name, size=8, height=None):
    image_height = size if height is None else height
    image = bpy.data.images.new(name, width=size, height=image_height, alpha=True, float_buffer=True)
    pixels = []
    for y in range(image_height):
        for x in range(size):
            pixels.extend([
                0.05 + 0.12 * x,
                0.08 + 0.11 * y,
                0.9 if (x + 2 * y) % 3 == 0 else 0.18,
                1.0,
            ])
    declared = image.colorspace_settings.name
    descriptor, path = tempfile.mkstemp(suffix=".exr")
    os.close(descriptor)
    previous_path = image.filepath_raw
    previous_format = image.file_format
    try:
        # Blender 5.2.2's bare pack() can leave a generated float image's EXR line offsets empty.
        image.colorspace_settings.name = "sRGB"
        image.pixels.foreach_set(pixels)
        image.update()
        image.filepath_raw = path
        image.file_format = "OPEN_EXR"
        image.save()
        with open(path, "rb") as handle:
            data = handle.read()
        image.pack(data=data, data_len=len(data))
    finally:
        image.filepath_raw = previous_path
        image.file_format = previous_format
        image.colorspace_settings.name = declared
        os.remove(path)
    return image


def set_input(node, name, value):
    socket = next((socket for socket in node.inputs if socket.name == name), None)
    if socket is None:
        raise RuntimeError(f"{node.bl_idname} lacks input {name}")
    if name == "Vector" and isinstance(value, (list, tuple)) and len(value) == 3:
        tree = node.id_data
        combine = tree.nodes.new("ShaderNodeCombineXYZ")
        for axis, component in zip("XYZ", value):
            next(item for item in combine.inputs if item.name == axis).default_value = component
        vector = next(item for item in combine.outputs if item.name == "Vector")
        tree.links.new(vector, socket)
    else:
        socket.default_value = value


def vector_config(value, generated=False):
    def configure(node, tree):
        if generated:
            texcoord = tree.nodes.new("ShaderNodeTexCoord")
            generated_socket = next(socket for socket in texcoord.outputs if socket.name == "Generated")
            vector_socket = next(socket for socket in node.inputs if socket.name == "Vector")
            tree.links.new(generated_socket, vector_socket)
        else:
            set_input(node, "Vector", value)
    return configure


def add_case(cases, name, node_type, output, configure, index, extra_probe=False, attrs=None,
             reverse=False, rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0)):
    material, _ = emission_material(name, node_type, output, configure)
    make_chart_plane(name, index, material, attrs, reverse, rotation, scale)
    centre_x = TILE_PIXELS // 2 + (index % GRID) * TILE_PIXELS
    centre_y = TILE_PIXELS // 2 + (index // GRID) * TILE_PIXELS
    probes = [{"x": centre_x, "y": centre_y}]
    if extra_probe:
        inner = (TILE_PIXELS // 4, TILE_PIXELS * 7 // 16)
        probes = [
            {"x": x + (index % GRID) * TILE_PIXELS, "y": y + (index // GRID) * TILE_PIXELS}
            for y in inner
            for x in inner
        ]
    cases.append({"name": name, "probes": probes})


def checker_cases(cases):
    # These two leave the Blender Color1, Color2, and Scale socket defaults untouched.
    add_case(cases, "checker_blender_defaults", "ShaderNodeTexChecker", "Color",
             vector_config((0.1, 0.1, 0.1)), 0)

    def parity1(node, _tree):
        set_input(node, "Vector", (0.0, 0.0, 0.0))
        set_input(node, "Color1", (0.91, 0.17, 0.29, 1.0))
        set_input(node, "Color2", (0.13, 0.71, 0.37, 1.0))
    add_case(cases, "checker_parity_even", "ShaderNodeTexChecker", "Color", parity1, 1)

    def parity2(node, _tree):
        set_input(node, "Vector", (0.31, 0.0, 0.0))
        set_input(node, "Color1", (0.91, 0.17, 0.29, 1.0))
        set_input(node, "Color2", (0.13, 0.71, 0.37, 1.0))
    add_case(cases, "checker_parity_odd", "ShaderNodeTexChecker", "Color", parity2, 2)
    add_case(cases, "checker_generated_coordinates", "ShaderNodeTexChecker", "Color",
             vector_config(None, generated=True), 3, extra_probe=True)


def gradient_cases(cases):
    for index, mode in enumerate(("LINEAR", "QUADRATIC", "EASING", "DIAGONAL", "RADIAL", "SPHERICAL"), 4):
        def configure(node, _tree, mode=mode):
            node.gradient_type = mode
            set_input(node, "Vector", (0.27, 0.39, 0.61))
        add_case(cases, "gradient_" + mode.lower(), "ShaderNodeTexGradient", "Factor", configure, index)


def view_node_cases(cases):
    def layer_config(node, _tree):
        set_input(node, "Blend", 0.37)

    add_case(cases, "layer_weight_fresnel", "ShaderNodeLayerWeight", "Fresnel", layer_config, 10)
    add_case(cases, "layer_weight_facing", "ShaderNodeLayerWeight", "Facing", layer_config, 11)

    def fresnel_front(node, _tree):
        set_input(node, "IOR", 1.33)

    def fresnel_back(node, tree):
        set_input(node, "IOR", 1.33)
        normal = tree.nodes.new("ShaderNodeCombineXYZ")
        set_input(normal, "X", 0.8660254)
        set_input(normal, "Z", -0.5)
        connect_named(tree, normal, "Vector", node, "Normal")
    add_case(cases, "fresnel_front", "ShaderNodeFresnel", "Factor", fresnel_front, 12)
    add_case(cases, "fresnel_backface", "ShaderNodeFresnel", "Factor", fresnel_back, 13, reverse=True)
    backface_rotation = (math.radians(37.0), math.radians(-22.0), 0.0)
    add_case(
        cases,
        "fresnel_backface_geometry",
        "ShaderNodeFresnel",
        "Factor",
        fresnel_front,
        61,
        reverse=True,
        rotation=backface_rotation,
    )
    add_case(
        cases,
        "layer_weight_fresnel_backface",
        "ShaderNodeLayerWeight",
        "Fresnel",
        layer_config,
        62,
        reverse=True,
        rotation=backface_rotation,
    )
    add_case(
        cases,
        "layer_weight_facing_backface",
        "ShaderNodeLayerWeight",
        "Facing",
        layer_config,
        63,
        reverse=True,
        rotation=backface_rotation,
    )


def noise_cases(cases):
    for index, dimensions in enumerate(("1D", "2D", "3D", "4D"), 14):
        def configure(node, _tree, dimensions=dimensions):
            node.noise_dimensions = dimensions
            set_input(node, "Vector", (0.23, 0.41, 0.67))
            set_input(node, "W", 0.29)
            set_input(node, "Scale", 2.3)
            set_input(node, "Detail", 2.1)
            set_input(node, "Roughness", 0.63)
            set_input(node, "Lacunarity", 2.5)
            set_input(node, "Distortion", 0.7)
        add_case(cases, "noise_" + dimensions.lower(), "ShaderNodeTexNoise", "Factor", configure, index)

    def lacunarity(node, _tree):
        node.noise_dimensions = "4D"
        node.normalize = False
        set_input(node, "Vector", (0.23, 0.41, 0.67))
        set_input(node, "W", 0.73)
        set_input(node, "Scale", 1.7)
        set_input(node, "Detail", 3.0)
        set_input(node, "Roughness", 0.52)
        set_input(node, "Lacunarity", 1.37)
        set_input(node, "Distortion", 0.0)
    add_case(cases, "noise_w_lacunarity_unnormalized", "ShaderNodeTexNoise", "Factor", lacunarity, 18)

    def normalization(node, _tree):
        node.noise_dimensions = "3D"
        node.normalize = True
        set_input(node, "Vector", (0.23, 0.41, 0.67))
        set_input(node, "Scale", 1.7)
        set_input(node, "Detail", 3.0)
        set_input(node, "Roughness", 0.52)
        set_input(node, "Lacunarity", 2.0)
        set_input(node, "Distortion", 0.83)
    add_case(cases, "noise_distortion_normalized", "ShaderNodeTexNoise", "Factor", normalization, 19)


def voronoi_cases(cases):
    for index, dimensions in enumerate(("1D", "2D", "3D", "4D"), 20):
        def configure(node, _tree, dimensions=dimensions):
            node.voronoi_dimensions = dimensions
            node.feature = "F1"
            node.distance = "EUCLIDEAN"
            set_input(node, "Vector", (0.23, 0.41, 0.67))
            set_input(node, "W", 0.29)
            set_input(node, "Scale", 2.3)
            set_input(node, "Randomness", 0.73)
        add_case(cases, "voronoi_" + dimensions.lower(), "ShaderNodeTexVoronoi", "Distance", configure, index)

    for offset, feature in enumerate(("F1", "F2", "SMOOTH_F1", "DISTANCE_TO_EDGE", "N_SPHERE_RADIUS"), 24):
        def configure(node, _tree, feature=feature):
            node.voronoi_dimensions = "3D"
            node.feature = feature
            node.distance = "EUCLIDEAN"
            set_input(node, "Vector", (0.23, 0.41, 0.67))
            set_input(node, "Scale", 2.3)
            set_input(node, "Smoothness", 0.31)
            set_input(node, "Randomness", 0.73)
        output = "Radius" if feature == "N_SPHERE_RADIUS" else "Color" if feature == "F1" else "Distance"
        add_case(cases, "voronoi_feature_" + feature.lower(), "ShaderNodeTexVoronoi", output, configure, offset)

    for offset, metric in enumerate(("EUCLIDEAN", "MANHATTAN", "CHEBYCHEV", "MINKOWSKI"), 29):
        def configure(node, _tree, metric=metric):
            node.voronoi_dimensions = "3D"
            node.feature = "F1"
            node.distance = metric
            set_input(node, "Exponent", 2.6)
            set_input(node, "Vector", (0.23, 0.41, 0.67))
            set_input(node, "Scale", 2.3)
            set_input(node, "Randomness", 0.73)
        add_case(cases, "voronoi_metric_" + metric.lower(), "ShaderNodeTexVoronoi", "Distance", configure, offset)

    def randomless(node, _tree):
        node.voronoi_dimensions = "3D"
        node.feature = "SMOOTH_F1"
        node.distance = "MANHATTAN"
        set_input(node, "Vector", (0.23, 0.41, 0.67))
        set_input(node, "Scale", 2.3)
        set_input(node, "Smoothness", 0.67)
        set_input(node, "Randomness", 0.0)
    add_case(cases, "voronoi_smoothness_randomness", "ShaderNodeTexVoronoi", "Distance", randomless, 33)

    def fractal(node, _tree):
        node.voronoi_dimensions = "4D"
        node.feature = "F1"
        node.distance = "MINKOWSKI"
        set_input(node, "Exponent", 2.6)
        node.normalize = False
        set_input(node, "Vector", (0.23, 0.41, 0.67))
        set_input(node, "W", 0.29)
        set_input(node, "Scale", 2.3)
        set_input(node, "Detail", 2.4)
        set_input(node, "Roughness", 0.59)
        set_input(node, "Lacunarity", 1.8)
        set_input(node, "Randomness", 0.73)
    add_case(cases, "voronoi_fractal_4d", "ShaderNodeTexVoronoi", "Distance", fractal, 34)



def image_cases(cases):
    image = image_texture("ShaderP1 Texture Source")
    projections = ("FLAT", "BOX", "SPHERE", "TUBE")
    for index, projection in enumerate(projections, 35):
        def configure(node, _tree, projection=projection):
            node.image = image
            node.projection = projection
            node.interpolation = "Linear"
            node.extension = "REPEAT"
            set_input(node, "Vector", (0.32, 0.61, 0.44))
        add_case(cases, "image_projection_" + projection.lower(), "ShaderNodeTexImage", "Color", configure, index)

    for offset, interpolation in enumerate(("Closest", "Linear", "Cubic", "Smart"), 39):
        def configure(node, _tree, interpolation=interpolation):
            node.image = image
            node.projection = "FLAT"
            node.interpolation = interpolation
            node.extension = "REPEAT"
            set_input(node, "Vector", (0.47, 0.58, 0.2))
        add_case(cases, "image_interpolation_" + interpolation.lower(), "ShaderNodeTexImage", "Color", configure, offset)

    for offset, extension in enumerate(("REPEAT", "EXTEND", "CLIP"), 43):
        def configure(node, _tree, extension=extension):
            node.image = image
            node.projection = "FLAT"
            node.interpolation = "Closest"
            node.extension = extension
            set_input(node, "Vector", (1.27, -0.19, 0.2))
        add_case(cases, "image_extension_" + extension.lower(), "ShaderNodeTexImage", "Color", configure, offset)

    projections = ("EQUIRECTANGULAR", "MIRROR_BALL")
    interpolations = ("Closest", "Linear", "Cubic", "Smart")
    for projection_index, projection in enumerate(projections):
        for interpolation_index, interpolation in enumerate(interpolations):
            slot = 46 + projection_index * len(interpolations) + interpolation_index
            def configure(node, _tree, projection=projection, interpolation=interpolation):
                node.image = image
                node.projection = projection
                node.interpolation = interpolation
                set_input(node, "Vector", (0.37, 0.2, 0.91))
            add_case(cases, "environment_" + projection.lower() + "_" + interpolation.lower(),
                     "ShaderNodeTexEnvironment", "Color", configure, slot)

    def rotated_box(node, tree):
        node.image = image
        node.projection = "BOX"
        node.interpolation = "Linear"
        node.extension = "REPEAT"
        texcoord = tree.nodes.new("ShaderNodeTexCoord")
        normal = next(socket for socket in texcoord.outputs if socket.name == "Normal")
        vector = next(socket for socket in node.inputs if socket.name == "Vector")
        tree.links.new(normal, vector)

    rotation = (math.radians(23.0), math.radians(37.0), 0.0)
    scale = (1.7, 0.7, 1.3)
    add_case(cases, "image_box_rotated_scaled_front", "ShaderNodeTexImage",
             "Color", rotated_box, 59, rotation=rotation, scale=scale)
    add_case(cases, "image_box_rotated_scaled_backface", "ShaderNodeTexImage",
             "Color", rotated_box, 60, reverse=True, rotation=rotation, scale=scale)

    for index, extension in enumerate(("REPEAT", "EXTEND", "CLIP", "MIRROR"), 64):
        def cubic_edge(node, _tree, extension=extension):
            node.image = image
            node.projection = "FLAT"
            node.interpolation = "Cubic"
            node.extension = extension
            set_input(node, "Vector", (1.02, 0.42, 0.2))
        add_case(cases, "image_cubic_extension_" + extension.lower() + "_edge",
                 "ShaderNodeTexImage", "Color", cubic_edge, index)
def default_coordinate_cases(cases):
    add_case(
        cases,
        "checker_default_unlinked_generated",
        "ShaderNodeTexChecker",
        "Color",
        None,
        68,
        extra_probe=True,
    )

    def gradient(node, _tree):
        node.gradient_type = "LINEAR"
    add_case(cases, "gradient_default_unlinked_generated", "ShaderNodeTexGradient",
             "Factor", gradient, 69)

    def noise(node, _tree):
        node.noise_dimensions = "3D"
        set_input(node, "Scale", 2.3)
        set_input(node, "Detail", 2.1)
        set_input(node, "Roughness", 0.63)
        set_input(node, "Lacunarity", 2.5)
    add_case(cases, "noise_default_unlinked_generated", "ShaderNodeTexNoise",
             "Factor", noise, 70)

    def voronoi(node, _tree):
        node.voronoi_dimensions = "3D"
        node.feature = "F1"
        node.distance = "EUCLIDEAN"
        set_input(node, "Scale", 2.3)
        set_input(node, "Randomness", 0.73)
    add_case(cases, "voronoi_default_unlinked_generated", "ShaderNodeTexVoronoi",
             "Distance", voronoi, 71)

    image = image_texture("ShaderP1 Default Generated Image")

    def image_default(node, _tree):
        node.image = image
        node.projection = "FLAT"
        node.interpolation = "Cubic"
        node.extension = "REPEAT"
    add_case(cases, "image_default_unlinked_generated", "ShaderNodeTexImage",
             "Color", image_default, 72)


def attribute_cases(cases):
    specifications = [
        ("attribute_float", "Float", 0.73, "Factor"),
        ("attribute_int", "Int", 3, "Factor"),
        ("attribute_vector", "FloatVector", (0.17, 0.53, 0.89), "Vector"),
        ("attribute_float_color", "FloatColor", (0.91, 0.23, 0.61, 1.0), "Color"),
        ("attribute_byte_color", "ByteColor", (0.31, 0.67, 0.13, 1.0), "Color"),
    ]
    for index, (name, attr_type, value, output) in enumerate(specifications, 54):
        def configure(node, _tree):
            node.attribute_type = "GEOMETRY"
            node.attribute_name = "Probe"
        values = [value] * 4
        attr_type = {"FloatVector": "FLOAT_VECTOR", "FloatColor": "FLOAT_COLOR",
                     "ByteColor": "BYTE_COLOR"}.get(attr_type, attr_type.upper())
        attrs = [("Probe", attr_type, values)]
        add_case(cases, name, "ShaderNodeAttribute", output, configure, index, attrs=attrs)




def collect_pixels(cases, out_dir):
    image_path = os.path.join(out_dir, "oracle_render.exr")
    scene = bpy.context.scene
    scene.render.filepath = image_path
    bpy.ops.render.render(write_still=True)
    result = bpy.data.images.load(image_path, check_existing=False)
    if tuple(result.size) != (WIDTH, HEIGHT):
        raise RuntimeError(f"rendered reference has unexpected dimensions: {tuple(result.size)}")
    pixels = result.pixels[:]
    for case in cases:
        for probe in case["probes"]:
            x, y = probe["x"], probe["y"]
            bottom_y = HEIGHT - 1 - y
            offset = (bottom_y * WIDTH + x) * 4
            probe["rgb"] = [float(pixels[offset + channel]) for channel in range(3)]
    bpy.data.images.remove(result)
    os.remove(image_path)
    return {"resolution": [WIDTH, HEIGHT], "renderer": "Blender 5.2.2 Cycles CPU; 1 sample; Raw; box filter 0.001",
            "source": "Blender 5.2.2 source/blender/nodes/shader/nodes and intern/cycles/kernel/svm",
            "cases": cases}


def connect_named(tree, source, output_name, target, input_name):
    output = next(socket for socket in source.outputs if socket.name == output_name)
    input_socket = next(socket for socket in target.inputs if socket.name == input_name)
    tree.links.new(output, input_socket)


def set_black_world(scene):
    world = bpy.data.worlds.new("ShaderP1 Black World")
    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputWorld")
    background = tree.nodes.new("ShaderNodeBackground")
    set_input(background, "Color", (0.0, 0.0, 0.0, 1.0))
    set_input(background, "Strength", 0.0)
    connect_named(tree, background, "Background", output, "Surface")
    scene.world = world


def set_white_world(scene):
    world = bpy.data.worlds.new("ShaderP1 White World")
    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputWorld")
    background = tree.nodes.new("ShaderNodeBackground")
    set_input(background, "Color", (1.0, 1.0, 1.0, 1.0))
    set_input(background, "Strength", 1.0)
    connect_named(tree, background, "Background", output, "Surface")
    scene.world = world


def render_image(out_dir, name, width, height):
    image_path = os.path.join(out_dir, name + ".exr")
    bpy.context.scene.render.filepath = image_path
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(image_path, check_existing=False)
    if tuple(image.size) != (width, height):
        raise RuntimeError(f"{name} reference has unexpected dimensions: {tuple(image.size)}")
    pixels = image.pixels[:]
    bpy.data.images.remove(image)
    os.remove(image_path)
    return pixels


def pixel_rgb(pixels, width, height, x, y):
    offset = ((height - 1 - y) * width + x) * 4
    return [float(pixels[offset + channel]) for channel in range(3)]


def region_mean(pixels, width, height, region):
    x0, y0, region_width, region_height = region
    total = [0.0, 0.0, 0.0]
    for y in range(y0, y0 + region_height):
        for x in range(x0, x0 + region_width):
            for channel, value in enumerate(pixel_rgb(pixels, width, height, x, y)):
                total[channel] += value
    count = float(region_width * region_height)
    return [value / count for value in total]


def save_blend(path):
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=path, relative_remap=False)



def build_glass_distribution_fixture(out_dir):
    width = 64
    height = 64
    samples = 4096
    scene = configure_scene()
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.cycles.samples = samples
    scene.cycles.seed = 0
    scene.cycles.max_bounces = 4
    scene.camera.data.ortho_scale = 4.0
    set_white_world(scene)

    cases = []
    glass_variants = [
        ("ggx", "GGX", -1.0),
        ("multi_ggx", "MULTI_GGX", 1.0),
        ("default", None, 0.0),
    ]
    for name, distribution, centre_x in glass_variants:
        glass = bpy.data.materials.new("Glass " + name)
        glass.use_nodes = True
        tree = glass.node_tree
        tree.nodes.clear()
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        node = tree.nodes.new("ShaderNodeBsdfGlass")
        if distribution is not None:
            node.distribution = distribution
        set_input(node, "Color", (1.0, 1.0, 1.0, 1.0))
        set_input(node, "Roughness", 0.9)
        set_input(node, "IOR", 1.45)
        connect_named(tree, node, "BSDF", output, "Surface")
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(centre_x, 0.9, 0.0))
        bpy.context.object.name = "glass_" + name
        bpy.context.object.data.materials.append(glass)
        x_centre = round(width / 2 + centre_x * width / 4)
        y_centre = round(height / 2 - 0.9 * height / 4)
        cases.append({
            "name": "glass_" + name,
            "region": [x_centre - 4, y_centre - 4, 8, 8],
        })

    path = os.path.join(out_dir, "glass_distribution.blend")
    save_blend(path)
    pixels = render_image(out_dir, "glass_distribution_reference", width, height)
    for case in cases:
        case["mean"] = region_mean(pixels, width, height, case["region"])
    document = {
        "resolution": [width, height],
        "samples": samples,
        "seed": 0,
        "max_bounces": 4,
        "renderer": "Blender 5.2.2 Cycles CPU; Raw; box filter 0.001",
        "source": "Blender 5.2.2 intern/cycles/kernel/closure/bsdf_microfacet.h",
        "cases": cases,
    }
    with open(os.path.join(out_dir, "glass_distribution.json"), "w", encoding="utf-8") as output:
        json.dump(document, output, indent=2, allow_nan=False)
        output.write("\n")


def build_physics_fixture(out_dir):
    width = 64
    height = 64
    samples = 1024
    scene = configure_scene()
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.cycles.samples = samples
    scene.cycles.seed = 0
    scene.cycles.max_bounces = 4
    scene.camera.data.ortho_scale = 4.0
    set_black_world(scene)

    cases = []
    for temperature in (1800.0, 9000.0):
        volume_material = bpy.data.materials.new(f"Blackbody {temperature:g} K")
        volume_material.use_nodes = True
        tree = volume_material.node_tree
        tree.nodes.clear()
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        volume = tree.nodes.new("ShaderNodeVolumePrincipled")
        set_input(volume, "Density", 0.6)
        set_input(volume, "Emission Strength", 0.0)
        set_input(volume, "Blackbody Intensity", 0.2)
        set_input(volume, "Temperature", temperature)
        connect_named(tree, volume, "Volume", output, "Volume")
        centre_x = -1.0 if temperature == 1800.0 else 1.0
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(centre_x, -0.9, 0.0))
        bpy.context.object.name = f"blackbody_{temperature:g}k"
        bpy.context.object.data.materials.append(volume_material)
        x_centre = round(width / 2 + centre_x * width / 4)
        y_centre = round(height / 2 + 0.9 * height / 4)
        cases.append({
            "name": f"volume_blackbody_{temperature:g}k",
            "region": [x_centre - 4, y_centre - 4, 8, 8],
        })

    path = os.path.join(out_dir, "shader_physics.blend")
    save_blend(path)
    pixels = render_image(out_dir, "shader_physics_reference", width, height)
    for case in cases:
        case["mean"] = region_mean(pixels, width, height, case["region"])
    document = {
        "resolution": [width, height],
        "samples": samples,
        "seed": 0,
        "max_bounces": 4,
        "renderer": "Blender 5.2.2 Cycles CPU; Raw; box filter 0.001",
        "source": "Blender 5.2.2 source/blender/nodes/shader/nodes/node_shader_volume_principled.cc",
        "cases": cases,
    }
    with open(os.path.join(out_dir, "shader_physics.json"), "w", encoding="utf-8") as output:
        json.dump(document, output, indent=2, allow_nan=False)
        output.write("\n")


def voronoi_clamp_material(name, detail, roughness):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    emission = tree.nodes.new("ShaderNodeEmission")
    voronoi = tree.nodes.new("ShaderNodeTexVoronoi")
    voronoi.voronoi_dimensions = "3D"
    voronoi.feature = "SMOOTH_F1"
    voronoi.distance = "EUCLIDEAN"
    set_input(voronoi, "Vector", (0.23, 0.41, 0.67))
    set_input(voronoi, "Scale", 2.3)
    set_input(voronoi, "Detail", detail)
    set_input(voronoi, "Roughness", roughness)
    set_input(voronoi, "Lacunarity", 1.8)
    set_input(voronoi, "Smoothness", 0.31)
    set_input(voronoi, "Randomness", 0.73)
    connect_named(tree, voronoi, "Distance", emission, "Color")
    connect_named(tree, emission, "Emission", output, "Surface")
    return material


def build_voronoi_clamp_fixture(out_dir):
    width = 64
    height = 32
    ortho_scale = 4.0
    scene = configure_scene()
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.camera.data.ortho_scale = ortho_scale
    set_black_world(scene)
    variants = [
        ("detail_at_15", -1.6, 15.0, 0.59, "Detail", 15.0),
        ("detail_over_15", -0.8, 20.0, 0.59, "Detail", 15.0),
        ("smooth_f1_lower_detail", 0.0, 4.0, 0.59, "Detail", 4.0),
        ("roughness_at_1", 0.8, 2.4, 1.0, "Roughness", 1.0),
        ("roughness_over_1", 1.6, 2.4, 1.4, "Roughness", 1.0),
    ]
    cases = []
    for name, centre_x, detail, roughness, clamped_property, clamp_value in variants:
        material = voronoi_clamp_material(name, detail, roughness)
        bpy.ops.mesh.primitive_plane_add(size=0.8, location=(centre_x, 0.0, 0.0))
        bpy.context.object.name = "voronoi_" + name
        bpy.context.object.data.materials.append(material)
        cases.append({
            "name": name,
            "probe": [round(width / 2 + centre_x * width / ortho_scale), height // 2],
            "input_value": detail if clamped_property == "Detail" else roughness,
            "clamped_value": clamp_value,
            "property": clamped_property,
        })
    save_blend(os.path.join(out_dir, "voronoi_clamps.blend"))
    pixels = render_image(out_dir, "voronoi_clamps_reference", width, height)
    for case in cases:
        x, y = case["probe"]
        case["rgb"] = pixel_rgb(pixels, width, height, x, y)
    with open(os.path.join(out_dir, "voronoi_clamps.json"), "w", encoding="utf-8") as output:
        json.dump({
            "resolution": [width, height],
            "renderer": "Blender 5.2.2 Cycles CPU; 1 sample; Raw; box filter 0.001",
            "source": "Blender 5.2.2 intern/cycles/kernel/svm/voronoi.h",
            "cases": cases,
        }, output, indent=2, allow_nan=False)
        output.write("\n")

def build_voronoi_fold_fields_fixture(out_dir):
    width = 64
    height = 32
    scene = configure_scene()
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.camera.data.ortho_scale = 4.0
    set_black_world(scene)
    cases = []
    for name, centre_x, coordinate_source in (
        ("voronoi_constant_folded", 0.0, "constant"),
        ("voronoi_unlinked_generated", 1.5, "unlinked"),
        ("voronoi_texcoord_generated", -1.5, "generated"),
    ):
        def configure(node, tree, coordinate_source=coordinate_source):
            node.voronoi_dimensions = "3D"
            node.feature = "F1"
            node.distance = "EUCLIDEAN"
            set_input(node, "Scale", 2.3)
            set_input(node, "Detail", 20.0)
            set_input(node, "Roughness", 1.4)
            set_input(node, "Randomness", 0.73)
            vector = next(socket for socket in node.inputs if socket.name == "Vector")
            if coordinate_source == "constant":
                combine = tree.nodes.new("ShaderNodeCombineXYZ")
                for axis, value in zip("XYZ", (0.23, 0.41, 0.67)):
                    set_input(combine, axis, value)
                tree.links.new(combine.outputs["Vector"], vector)
            elif coordinate_source == "generated":
                texcoord = tree.nodes.new("ShaderNodeTexCoord")
                tree.links.new(texcoord.outputs["Generated"], vector)

        material, _ = emission_material(
            name, "ShaderNodeTexVoronoi", "Distance", configure
        )
        bpy.ops.mesh.primitive_plane_add(size=0.8, location=(centre_x, 0.0, 0.0))
        bpy.context.object.name = name
        bpy.context.object.data.materials.append(material)
        probe_x = round(width / 2 + centre_x * width / 4)
        if name == "voronoi_unlinked_generated":
            probe_x -= 2
        cases.append({
            "name": name,
            "detail": 20.0,
            "roughness": 1.4,
            "probe": [probe_x, height // 2],
        })

    save_blend(os.path.join(out_dir, "voronoi_fold_fields.blend"))
    references = {}
    for engine in ("eevee", "cycles"):
        pixels = render_engine_image(
            scene, out_dir, "voronoi_fold_fields_" + engine, width, height, engine
        )
        references[engine] = pixels
    for case in cases:
        x, y = case["probe"]
        for engine, key in (("eevee", "eevee_rgb"), ("cycles", "cycles_rgb")):
            case[key] = pixel_rgb(references[engine], width, height, x, y)
    for key in ("eevee_rgb", "cycles_rgb"):
        if len({tuple(case[key]) for case in cases}) != len(cases):
            raise RuntimeError(f"Voronoi input probes collapsed in {key}: {[(case['name'], case[key]) for case in cases]}")
    with open(os.path.join(out_dir, "voronoi_fold_fields.json"), "w", encoding="utf-8") as output:
        json.dump({
            "resolution": [width, height],
            "samples": {"eevee": 1, "cycles": 1},
            "seed": 0,
            "renderer": {
                "eevee": "Blender 5.2.2 EEVEE Next; 1 TAA sample; Raw",
                "cycles": "Blender 5.2.2 Cycles CPU; 1 sample; Raw; box filter 0.001",
            },
            "source": "Blender 5.2.2: unlinked EEVEE Voronoi Vector uses CD_ORCO, the same attribute as TexCoord.Generated (node_shader_util.cc, node_shader_tex_coord.cc); Cycles lowers it to LINK_TEXTURE_GENERATED (intern/cycles/scene/shader_nodes.cpp); node_shader_tex_voronoi.cc, gpu_shader_material_tex_voronoi.glsl, intern/cycles/kernel/svm/voronoi.h",
            "cases": cases,
        }, output, indent=2, allow_nan=False)
        output.write("\n")


def build_voronoi_fold_invariants_fixture(out_dir):
    width = 64
    height = 32
    scene = configure_scene()
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.camera.data.ortho_scale = 4.0
    set_black_world(scene)

    def configure_voronoi(node, detail, roughness):
        node.voronoi_dimensions = "3D"
        node.feature = "F1"
        node.distance = "EUCLIDEAN"
        set_input(node, "Scale", 2.3)
        set_input(node, "Detail", detail)
        set_input(node, "Roughness", roughness)
        set_input(node, "Lacunarity", 1.8)
        set_input(node, "Smoothness", 0.31)
        set_input(node, "Randomness", 0.73)

    def constant_material():
        def configure(node, tree):
            configure_voronoi(node, 5.0, 0.5)
            combine = tree.nodes.new("ShaderNodeCombineXYZ")
            for axis, value in zip("XYZ", (0.23, 0.41, 0.67)):
                set_input(combine, axis, value)
            tree.links.new(combine.outputs["Vector"], node.inputs["Vector"])

        return emission_material(
            "voronoi_constant_low_detail_absolute",
            "ShaderNodeTexVoronoi",
            "Distance",
            configure,
        )[0]

    def difference_material(name, left_detail, left_roughness,
                            right_detail, right_roughness, unlinked_left=False):
        material = bpy.data.materials.new(name)
        material.use_nodes = True
        tree = material.node_tree
        tree.nodes.clear()
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        emission = tree.nodes.new("ShaderNodeEmission")
        left = tree.nodes.new("ShaderNodeTexVoronoi")
        right = tree.nodes.new("ShaderNodeTexVoronoi")
        configure_voronoi(left, left_detail, left_roughness)
        configure_voronoi(right, right_detail, right_roughness)
        texcoord = tree.nodes.new("ShaderNodeTexCoord")
        if not unlinked_left:
            tree.links.new(texcoord.outputs["Generated"], left.inputs["Vector"])
        tree.links.new(texcoord.outputs["Generated"], right.inputs["Vector"])
        subtract = tree.nodes.new("ShaderNodeMath")
        subtract.operation = "SUBTRACT"
        tree.links.new(left.outputs["Distance"], subtract.inputs[0])
        tree.links.new(right.outputs["Distance"], subtract.inputs[1])
        tree.links.new(subtract.outputs["Value"], emission.inputs["Color"])
        tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
        return material

    def add_probe_plane(name, material, location):
        bpy.ops.mesh.primitive_plane_add(size=0.8, location=location)
        obj = bpy.context.object
        obj.name = name
        obj.data.materials.append(material)

    def probe(center_x, center_y):
        return [
            round(width / 2 + center_x * width / 4),
            round(height / 2 - center_y * width / 4),
        ]

    constant_name = "voronoi_constant_low_detail_absolute"
    constant_x, constant_y = -1.4, -0.5
    add_probe_plane(constant_name, constant_material(), (constant_x, constant_y, 0.0))

    clamp_name = "voronoi_kernel_clamp_difference_zero"
    clamp_x, clamp_y = -0.1, -0.5
    add_probe_plane(
        clamp_name,
        difference_material(clamp_name, 20.0, 1.4, 15.0, 1.0),
        (clamp_x, clamp_y, 0.0),
    )

    generated_name = "voronoi_generated_default_difference_zero"
    generated_material = difference_material(
        generated_name, 5.0, 0.5, 5.0, 0.5, unlinked_left=True
    )
    generated_mesh = bpy.data.meshes.new(generated_name + " Mesh")
    generated_mesh.from_pydata(
        [(0.0, 0.0, 0.5), (1.0, 0.0, 0.5), (1.0, 1.0, 0.5), (0.0, 1.0, 0.5)],
        [],
        [(0, 1, 2, 3)],
    )
    generated_mesh.update()
    generated_plane = bpy.data.objects.new(generated_name, generated_mesh)
    scene.collection.objects.link(generated_plane)
    generated_plane.data.materials.append(generated_material)

    cases = [
        {"name": constant_name, "probe": probe(constant_x, constant_y)},
        {"name": clamp_name, "probe": probe(clamp_x, clamp_y)},
        {"name": generated_name, "probe": probe(0.5, 0.5)},
    ]
    save_blend(os.path.join(out_dir, "voronoi_fold_invariants.blend"))
    references = {}
    for engine in ("eevee", "cycles"):
        references[engine] = render_engine_image(
            scene, out_dir, "voronoi_fold_invariants_" + engine, width, height, engine
        )
    zero_cases = {clamp_name, generated_name}
    for case in cases:
        x, y = case["probe"]
        for engine, key in (("eevee", "eevee_rgb"), ("cycles", "cycles_rgb")):
            rgb = pixel_rgb(references[engine], width, height, x, y)
            if case["name"] in zero_cases:
                if rgb != [0.0, 0.0, 0.0]:
                    raise RuntimeError(
                        f"Voronoi invariant {case['name']} was nonzero in {engine}: {rgb}"
                    )
                rgb = [0, 0, 0]
            case[key] = rgb
    with open(os.path.join(out_dir, "voronoi_fold_invariants.json"), "w", encoding="utf-8") as output:
        json.dump({
            "resolution": [width, height],
            "samples": {"eevee": 1, "cycles": 1},
            "seed": 0,
            "renderer": {
                "eevee": "Blender 5.2.2 EEVEE Next; 1 TAA sample; Raw",
                "cycles": "Blender 5.2.2 Cycles CPU; 1 sample; Raw; box filter 0.001",
            },
            "source": "Blender 5.2.2 source/blender/nodes/shader/nodes/node_shader_tex_voronoi.cc, node_shader_tex_coord.cc, source/blender/gpu/shaders/material/gpu_shader_material_tex_voronoi.glsl, and intern/cycles/kernel/svm/voronoi.h",
            "cases": cases,
        }, output, indent=2, allow_nan=False)
        output.write("\n")


def build_texcoord_normal_fixture(out_dir):
    width = 64
    height = 32
    scene = configure_scene()
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.filter_size = 0.0
    scene.eevee.taa_render_samples = 1
    scene.camera.data.ortho_scale = 4.0
    scene.camera.location = (10.0, 0.0, 10.0)
    scene.camera.rotation_euler = (0.0, math.radians(45.0), 0.0)
    set_black_world(scene)
    cases = []
    for name, centre_x, rotation in (
        ("normal_identity", -1.0, (0.0, 0.0, 0.0)),
        ("normal_rotated_90", 1.0, (0.0, math.radians(90.0), 0.0)),
    ):
        material, _ = emission_material(name, "ShaderNodeTexCoord", "Normal")
        bpy.ops.mesh.primitive_plane_add(size=0.8, location=(centre_x, 0.0, 0.0))
        obj = bpy.context.object
        obj.name = name
        obj.rotation_euler = rotation
        obj.data.materials.append(material)
        cases.append({
            "name": name,
            "probe": [
                round(width / 2 + centre_x * math.cos(math.radians(45.0)) * width / 4),
                height // 2,
            ],
        })

    save_blend(os.path.join(out_dir, "texcoord_normal_eevee.blend"))
    pixels = render_engine_image(
        scene, out_dir, "texcoord_normal_eevee_reference", width, height, "eevee"
    )
    for case in cases:
        x, y = case["probe"]
        case["rgb"] = pixel_rgb(pixels, width, height, x, y)
    with open(os.path.join(out_dir, "texcoord_normal_eevee.json"), "w", encoding="utf-8") as output:
        json.dump({
            "resolution": [width, height],
            "samples": 1,
            "seed": 0,
            "renderer": "Blender 5.2.2 EEVEE Next; 1 TAA sample; Raw",
            "source": "Blender 5.2.2 source/blender/nodes/shader/nodes/node_shader_tex_coord.cc and source/blender/gpu/shaders/material/gpu_shader_material_texture_coordinates.glsl",
            "cases": cases,
        }, output, indent=2, allow_nan=False)
        output.write("\n")


def build_negative_scale_facing_fixture(out_dir):
    width = 64
    height = 32
    scene = configure_scene()
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.camera.data.ortho_scale = 4.0
    set_black_world(scene)
    cases = []
    for name, centre_x, scale in (
        ("facing_positive_scale", -1.0, (1.0, 1.0, 1.0)),
        ("facing_negative_scale", 1.0, (1.0, 1.0, -1.0)),
    ):
        material, _ = emission_material(name, "ShaderNodeNewGeometry", "Backfacing")
        bpy.ops.mesh.primitive_plane_add(size=0.8, location=(centre_x, 0.0, 0.0))
        obj = bpy.context.object
        obj.name = name
        obj.scale = scale
        obj.data.materials.append(material)
        cases.append({
            "name": name,
            "probe": [round(width / 2 + centre_x * width / 4), height // 2],
        })

    save_blend(os.path.join(out_dir, "negative_scale_facing.blend"))
    references = {}
    for engine in ("eevee", "cycles"):
        pixels = render_engine_image(
            scene, out_dir, "negative_scale_facing_" + engine, width, height, engine
        )
        references[engine] = pixels
    for case in cases:
        x, y = case["probe"]
        case["eevee_rgb"] = pixel_rgb(references["eevee"], width, height, x, y)
        case["cycles_rgb"] = pixel_rgb(references["cycles"], width, height, x, y)
    for key in ("eevee_rgb", "cycles_rgb"):
        if cases[0][key] == cases[1][key]:
            raise RuntimeError(f"positive and negative object scales collapsed in {key}")
    with open(os.path.join(out_dir, "negative_scale_facing.json"), "w", encoding="utf-8") as output:
        json.dump({
            "resolution": [width, height],
            "samples": {"eevee": 1, "cycles": 1},
            "seed": 0,
            "renderer": {
                "eevee": "Blender 5.2.2 EEVEE Next; 1 TAA sample; Raw",
                "cycles": "Blender 5.2.2 Cycles CPU; 1 sample; Raw; box filter 0.001",
            },
            "source": "Blender 5.2.2 source/blender/nodes/shader/nodes/node_shader_geometry.cc, source/blender/gpu/shaders/material/gpu_shader_material_geometry.glsl, and intern/cycles/kernel/svm/geometry.h",
            "cases": cases,
        }, output, indent=2, allow_nan=False)
        output.write("\n")


def build_realtime_glass_fixture(out_dir):
    width = 96
    height = 96
    samples = 64
    ortho_scale = 4.0
    scene = configure_scene()
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.filter_size = 0.0
    scene.eevee.taa_render_samples = samples
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.camera.data.ortho_scale = ortho_scale
    scene.camera.location = (0.0, 0.0, 10.0)
    set_black_world(scene)

    light_data = bpy.data.lights.new("Glass Distribution Area", "AREA")
    light_data.shape = "SQUARE"
    light_data.size = 2.0
    light_data.energy = 400.0
    light = bpy.data.objects.new("Glass Distribution Area", light_data)
    scene.collection.objects.link(light)
    light.location = (0.0, 0.0, 4.0)

    cases = []
    for name, distribution, centre_x in (
        ("ggx", "GGX", -0.75),
        ("multi_ggx", "MULTI_GGX", 0.75),
    ):
        material = bpy.data.materials.new("Realtime Glass " + name)
        material.use_nodes = True
        tree = material.node_tree
        tree.nodes.clear()
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        glass = tree.nodes.new("ShaderNodeBsdfGlass")
        glass.distribution = distribution
        set_input(glass, "Color", (1.0, 1.0, 1.0, 1.0))
        set_input(glass, "Roughness", 0.9)
        set_input(glass, "IOR", 1.45)
        connect_named(tree, glass, "BSDF", output, "Surface")
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=0.65, segments=64, ring_count=32, location=(centre_x, 0.0, 0.0)
        )
        sphere = bpy.context.object
        sphere.name = "realtime_glass_" + name
        bpy.ops.object.shade_smooth()
        sphere.data.materials.append(material)
        x_centre = round((centre_x + ortho_scale / 2.0) * width / ortho_scale)
        y_centre = height // 2
        cases.append({
            "name": "glass_" + name,
            "region": [x_centre - 4, y_centre - 4, 8, 8],
        })

    save_blend(os.path.join(out_dir, "glass_realtime.blend"))
    pixels = render_image(out_dir, "glass_realtime_reference", width, height)
    for case in cases:
        case["mean"] = region_mean(pixels, width, height, case["region"])
    with open(os.path.join(out_dir, "glass_realtime.json"), "w", encoding="utf-8") as output:
        json.dump({
            "resolution": [width, height],
            "samples": samples,
            "renderer": "Blender 5.2.2 EEVEE Next; 64 TAA samples; Standard",
            "source": "Blender 5.2.2 source/blender/nodes/shader/nodes/node_shader_bsdf_glass.cc and source/blender/gpu/shaders/material/gpu_shader_material_glass.glsl",
            "cases": cases,
        }, output, indent=2, allow_nan=False)
        output.write("\n")


def build_realtime_colored_glass_fixture(out_dir):
    width = 96
    height = 96
    samples = 64
    ortho_scale = 4.0
    scene = configure_scene()
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.filter_size = 0.0
    scene.eevee.taa_render_samples = samples
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.camera.data.ortho_scale = ortho_scale
    scene.camera.location = (0.0, 0.0, 10.0)
    set_black_world(scene)

    light_data = bpy.data.lights.new("Colored Glass Distribution Area", "AREA")
    light_data.shape = "SQUARE"
    light_data.size = 2.0
    light_data.energy = 400.0
    light = bpy.data.objects.new("Colored Glass Distribution Area", light_data)
    scene.collection.objects.link(light)
    light.location = (0.0, 0.0, 4.0)

    cases = []
    for name, distribution, centre_x in (
        ("ggx", "GGX", -0.75),
        ("multi_ggx", "MULTI_GGX", 0.75),
    ):
        material = bpy.data.materials.new("Realtime Colored Glass " + name)
        material.use_nodes = True
        tree = material.node_tree
        tree.nodes.clear()
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        glass = tree.nodes.new("ShaderNodeBsdfGlass")
        glass.distribution = distribution
        set_input(glass, "Color", (0.82, 0.18, 0.07, 1.0))
        set_input(glass, "Roughness", 0.9)
        set_input(glass, "IOR", 1.45)
        connect_named(tree, glass, "BSDF", output, "Surface")
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=0.65, segments=64, ring_count=32, location=(centre_x, 0.0, 0.0)
        )
        sphere = bpy.context.object
        sphere.name = "realtime_colored_glass_" + name
        bpy.ops.object.shade_smooth()
        sphere.data.materials.append(material)
        x_centre = round((centre_x + ortho_scale / 2.0) * width / ortho_scale)
        cases.append({
            "name": "glass_" + name,
            "probe": [x_centre, height // 2],
        })

    save_blend(os.path.join(out_dir, "glass_realtime_colored.blend"))
    pixels = render_image(out_dir, "glass_realtime_colored_reference", width, height)
    for case in cases:
        x, y = case["probe"]
        case["rgb"] = pixel_rgb(pixels, width, height, x, y)
    if not any(
        abs(channel_a - channel_b) > 1.0e-3
        for channel_a, channel_b in zip(cases[0]["rgb"], cases[1]["rgb"])
    ):
        raise RuntimeError("colored Glass GGX and MULTI_GGX probes did not distinguish")
    with open(os.path.join(out_dir, "glass_realtime_colored.json"), "w", encoding="utf-8") as output:
        json.dump({
            "resolution": [width, height],
            "samples": samples,
            "seed": 0,
            "renderer": "Blender 5.2.2 EEVEE Next; 64 TAA samples; Standard",
            "source": "Blender 5.2.2 source/blender/nodes/shader/nodes/node_shader_bsdf_glass.cc and source/blender/gpu/shaders/material/gpu_shader_material_glass.glsl",
            "cases": cases,
        }, output, indent=2, allow_nan=False)
        output.write("\n")


def displacement_material(name, space):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    emission = tree.nodes.new("ShaderNodeEmission")
    displacement = tree.nodes.new("ShaderNodeDisplacement")
    displacement.space = space
    set_input(displacement, "Height", 0.83)
    set_input(displacement, "Midlevel", 0.27)
    set_input(displacement, "Scale", 1.7)
    local_normal = tree.nodes.new("ShaderNodeCombineXYZ")
    set_input(local_normal, "Z", 1.0)
    connect_named(tree, local_normal, "Vector", displacement, "Normal")
    connect_named(tree, displacement, "Displacement", emission, "Color")
    connect_named(tree, emission, "Emission", output, "Surface")
    return material


def build_displacement_fixture(out_dir):
    width = 64
    height = 64
    scene = configure_scene()
    scene.render.engine = "BLENDER_EEVEE"
    scene.cycles.seed = 0
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.eevee.taa_render_samples = 1
    scene.camera.data.ortho_scale = 4.0
    set_black_world(scene)

    cases = []
    for index, space in enumerate(("OBJECT", "WORLD")):
        name = "displacement_" + space.lower()
        centre_x = -1.0 if index == 0 else 1.0
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(centre_x, 0.0, 0.0))
        obj = bpy.context.object
        obj.name = name
        obj.rotation_euler = (math.radians(27.0), math.radians(34.0), 0.0)
        obj.scale = (2.0, 1.0, 0.5)
        obj.data.materials.append(displacement_material(name, space))
        x_centre = round(width / 2 + centre_x * width / 4)
        cases.append({
            "name": name,
            "probes": [
                {"x": x_centre, "y": height // 2},
                {"x": x_centre - 2, "y": height // 2},
                {"x": x_centre + 2, "y": height // 2},
            ],
        })

    path = os.path.join(out_dir, "displacement_space.blend")
    save_blend(path)
    eevee_pixels = render_image(out_dir, "displacement_space_eevee_reference", width, height)
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    scene.cycles.max_bounces = 0
    cycles_pixels = render_image(out_dir, "displacement_space_cycles_reference", width, height)
    for case in cases:
        for probe in case["probes"]:
            probe["eevee_rgb"] = pixel_rgb(
                eevee_pixels, width, height, probe["x"], probe["y"]
            )
            probe["cycles_rgb"] = pixel_rgb(
                cycles_pixels, width, height, probe["x"], probe["y"]
            )
    if cases[0]["probes"][0]["eevee_rgb"] == cases[1]["probes"][0]["eevee_rgb"]:
        raise RuntimeError("OBJECT and WORLD displacement did not produce different EEVEE probes")
    if cases[0]["probes"][0]["cycles_rgb"] == cases[1]["probes"][0]["cycles_rgb"]:
        raise RuntimeError("OBJECT and WORLD displacement did not produce different Cycles probes")
    document = {
        "resolution": [width, height],
        "samples": {"eevee": 1, "cycles": 1},
        "seed": 0,
        "renderer": {
            "eevee": "Blender 5.2.2 EEVEE Next; 1 TAA sample; Raw",
            "cycles": "Blender 5.2.2 Cycles CPU; 1 sample; Raw",
        },
        "source": "Blender 5.2.2 source/blender/nodes/shader/nodes/node_shader_displacement.cc (GPU space-specific shader)",
        "cases": cases,
    }
    with open(os.path.join(out_dir, "displacement_space.json"), "w", encoding="utf-8") as output:
        json.dump(document, output, indent=2, allow_nan=False)
        output.write("\n")

def build_environment_world_fixture(out_dir):
    width = 64
    height = 64
    cases = []
    variants = [
        ("EQUIRECTANGULAR", "Closest", False),
        ("MIRROR_BALL", "Cubic", True),
    ]
    for projection, interpolation, linked_vector in variants:
        name = f"world_environment_{projection.lower()}_{interpolation.lower()}"
        scene = configure_scene()
        scene.render.resolution_x = width
        scene.render.resolution_y = height
        world = bpy.data.worlds.new(name)
        world.use_nodes = True
        tree = world.node_tree
        tree.nodes.clear()
        output = tree.nodes.new("ShaderNodeOutputWorld")
        background = tree.nodes.new("ShaderNodeBackground")
        environment = tree.nodes.new("ShaderNodeTexEnvironment")
        environment.image = image_texture(name + " Image")
        environment.projection = projection
        environment.interpolation = interpolation
        if linked_vector:
            set_input(environment, "Vector", (0.37, 0.2, 0.91))
        connect_named(tree, environment, "Color", background, "Color")
        connect_named(tree, background, "Background", output, "Surface")
        scene.world = world
        save_blend(os.path.join(out_dir, name + ".blend"))
        pixels = {}
        for engine in ("eevee", "cycles"):
            pixels[engine] = render_engine_image(
                scene, out_dir, name + "_" + engine + "_reference", width, height, engine
            )
        x = width // 2
        y = height // 2
        cases.append({
            "name": name,
            "blend": name + ".blend",
            "probe": [x, y],
            "eevee_rgb": pixel_rgb(pixels["eevee"], width, height, x, y),
            "cycles_rgb": pixel_rgb(pixels["cycles"], width, height, x, y),
        })
    if cases[0]["eevee_rgb"] == cases[1]["eevee_rgb"]:
        raise RuntimeError("EEVEE world probes did not distinguish environment projection/filtering")
    with open(os.path.join(out_dir, "environment_world.json"), "w", encoding="utf-8") as output:
        json.dump({
            "resolution": [width, height],
            "samples": {"eevee": 1, "cycles": 1},
            "seed": 0,
            "renderer": {
                "eevee": "Blender 5.2.2 EEVEE Next; 1 TAA sample; Raw",
                "cycles": "Blender 5.2.2 Cycles CPU; 1 sample; Raw; box filter 0.001",
            },
            "source": "Blender 5.2.2 source/blender/nodes/shader/nodes/node_shader_tex_environment.cc, source/blender/gpu/shaders/material/gpu_shader_material_tex_environment.glsl, and intern/cycles/kernel/svm/image.h",
            "cases": cases,
        }, output, indent=2, allow_nan=False)
        output.write("\n")

def build_realtime_volume_world_ambient_fixture(out_dir):
    width = 64
    height = 64
    samples = 16
    scene = configure_scene()
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.filter_size = 0.0
    scene.eevee.taa_render_samples = samples
    scene.camera.data.ortho_scale = 4.0
    scene.camera.location = (0.0, 0.0, 10.0)

    world = bpy.data.worlds.new("Realtime Volume World Ambient")
    world.use_nodes = True
    world_tree = world.node_tree
    world_tree.nodes.clear()
    world_output = world_tree.nodes.new("ShaderNodeOutputWorld")
    background = world_tree.nodes.new("ShaderNodeBackground")
    environment = world_tree.nodes.new("ShaderNodeTexEnvironment")
    environment.image = image_texture("Volume Ambient Equirectangular Image", size=16, height=8)
    environment.projection = "EQUIRECTANGULAR"
    environment.interpolation = "Linear"
    set_input(background, "Strength", 1.0)
    connect_named(world_tree, environment, "Color", background, "Color")
    connect_named(world_tree, background, "Background", world_output, "Surface")
    scene.world = world

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
    volume_object = bpy.context.object
    volume_object.name = "Realtime World Ambient Mesh Volume"
    material = bpy.data.materials.new("Realtime World Ambient Mesh Volume Material")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    volume = tree.nodes.new("ShaderNodeVolumePrincipled")
    set_input(volume, "Density", 0.6)
    set_input(volume, "Color", (1.0, 1.0, 1.0, 1.0))
    set_input(volume, "Anisotropy", 0.0)
    set_input(volume, "Emission Strength", 0.0)
    set_input(volume, "Blackbody Intensity", 0.0)
    connect_named(tree, volume, "Volume", output, "Volume")
    volume_object.data.materials.append(material)

    blend_name = "volume_world_ambient_eevee.blend"
    save_blend(os.path.join(out_dir, blend_name))
    probe = [width // 2, height // 2]
    volume_pixels = render_engine_image(
        scene, out_dir, "volume_world_ambient_eevee_reference", width, height,
        "eevee", samples=samples,
    )
    volume_rgb = pixel_rgb(volume_pixels, width, height, *probe)
    volume_object.hide_render = True
    background_pixels = render_engine_image(
        scene, out_dir, "volume_world_ambient_background_reference", width, height,
        "eevee", samples=samples,
    )
    background_rgb = pixel_rgb(background_pixels, width, height, *probe)
    if all(abs(a - b) <= 1.0e-3 for a, b in zip(volume_rgb, background_rgb)):
        raise RuntimeError("world environment ambient did not change the EEVEE volume probe")
    with open(os.path.join(out_dir, "volume_world_ambient_eevee.json"), "w", encoding="utf-8") as output:
        json.dump({
            "resolution": [width, height],
            "samples": samples,
            "seed": 0,
            "renderer": "Blender 5.2.2 EEVEE Next; 16 TAA samples; Raw",
            "environment_projection": "EQUIRECTANGULAR",
            "environment_interpolation": "Linear",
            "ambient_direction": [0.0, 0.0, 1.0],
            "source": "Blender 5.2.2 source/blender/nodes/shader/nodes/node_shader_tex_environment.cc, node_shader_volume_principled.cc, source/blender/gpu/shaders/material/gpu_shader_material_tex_environment.glsl, and gpu_shader_material_volume_principled.glsl",
            "case": {
                "name": "world_ambient_volume_equirectangular_v_seam",
                "blend": blend_name,
                "probe": probe,
                "rgb": volume_rgb,
                "background_rgb": background_rgb,
            },
        }, output, indent=2, allow_nan=False)
        output.write("\n")

def build_realtime_anisotropic_world_volume_refusal_fixture(out_dir):
    width = 16
    height = 16
    samples = 1
    scene = configure_scene()
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.eevee.taa_render_samples = samples
    scene.camera.data.ortho_scale = 4.0
    scene.camera.location = (0.0, 0.0, 10.0)

    world = bpy.data.worlds.new("Anisotropic Volume Nonuniform World")
    world.use_nodes = True
    world_tree = world.node_tree
    world_tree.nodes.clear()
    world_output = world_tree.nodes.new("ShaderNodeOutputWorld")
    background = world_tree.nodes.new("ShaderNodeBackground")
    environment = world_tree.nodes.new("ShaderNodeTexEnvironment")
    environment.image = image_texture("Anisotropic Volume Nonuniform Environment")
    environment.projection = "EQUIRECTANGULAR"
    environment.interpolation = "Linear"
    set_input(background, "Strength", 1.0)
    connect_named(world_tree, environment, "Color", background, "Color")
    connect_named(world_tree, background, "Background", world_output, "Surface")
    scene.world = world

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
    volume_object = bpy.context.object
    volume_object.name = "Anisotropic Volume Refusal Mesh"
    material = bpy.data.materials.new("Anisotropic Volume Refusal Material")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    volume = tree.nodes.new("ShaderNodeVolumePrincipled")
    set_input(volume, "Density", 0.6)
    set_input(volume, "Color", (1.0, 1.0, 1.0, 1.0))
    set_input(volume, "Anisotropy", 0.5)
    connect_named(tree, volume, "Volume", output, "Volume")
    volume_object.data.materials.append(material)

    blend_name = "volume_world_ambient_anisotropic.blend"
    save_blend(os.path.join(out_dir, blend_name))
    with open(
        os.path.join(out_dir, "volume_world_ambient_refusal.json"), "w", encoding="utf-8"
    ) as output:
        json.dump({
            "samples": samples,
            "seed": 0,
            "renderer": "Blender 5.2.2 EEVEE Next; 1 TAA sample",
            "world": {
                "environment_image_nonuniform": True,
                "projection": "EQUIRECTANGULAR",
                "interpolation": "Linear",
            },
            "cases": [{
                "name": "anisotropic_volume_nonuniform_world",
                "blend": blend_name,
                "engine": "realtime",
                "anisotropy": 0.5,
                "error_code": "UNSUPPORTED_FEATURE",
                "feature_id": "shader.volume.world_probe_l1.realtime_scattering",
            }],
            "source": "Blender 5.2.2 node_shader_tex_environment.cc and node_shader_volume_principled.cc; EEVEE realtime volume scattering under a nonuniform world environment",
        }, output, indent=2, allow_nan=False)
        output.write("\n")


def build_mapping_refusal_fixtures(out_dir):
    cases = []
    for setting in ("texture_mapping", "color_mapping"):
        scene = configure_scene()
        material = bpy.data.materials.new("ShaderP1 " + setting)
        material.use_nodes = True
        tree = material.node_tree
        tree.nodes.clear()
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        emission = tree.nodes.new("ShaderNodeEmission")
        noise = tree.nodes.new("ShaderNodeTexNoise")
        noise.name = "Mapped Noise"
        if setting == "texture_mapping":
            noise.texture_mapping.translation = (0.25, -0.125, 0.0)
            translation = [0.25, -0.125, 0.0]
            brightness = 1.0
            feature_id = "shader.node.ShaderNodeTexNoise.texture_mapping"
        else:
            noise.color_mapping.brightness = 1.25
            translation = [0.0, 0.0, 0.0]
            brightness = 1.25
            feature_id = "shader.node.ShaderNodeTexNoise.color_mapping"
        connect_named(tree, noise, "Color", emission, "Color")
        connect_named(tree, emission, "Emission", output, "Surface")
        bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, 0.0))
        bpy.context.object.name = "mapped_" + setting
        bpy.context.object.data.materials.append(material)
        blend_name = "mapped_" + setting + ".blend"
        save_blend(os.path.join(out_dir, blend_name))
        cases.append({
            "name": setting,
            "blend": blend_name,
            "loss_feature_id": feature_id,
            "error_code": "UNSUPPORTED_FEATURE",
            "feature_id": feature_id,
            "texture_mapping": {
                "vector_type": "POINT",
                "translation": translation,
            },
            "color_mapping": {
                "brightness": brightness,
            },
        })
    with open(os.path.join(out_dir, "mapping_refusals.json"), "w", encoding="utf-8") as output:
        json.dump({"cases": cases}, output, indent=2, allow_nan=False)
        output.write("\n")


def build_spatial_volume_fixture(out_dir):
    scene = configure_scene()
    vdb_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../openvdb_transform_types/frustum.vdb")
    )
    bpy.ops.object.volume_import(filepath=vdb_path)
    volume_object = bpy.context.object
    volume_object.name = "Spatially Varying VDB"
    volume_object.data.filepath = "//../../openvdb_transform_types/frustum.vdb"

    material = bpy.data.materials.new("Spatially Varying Volume Material")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    principled = tree.nodes.new("ShaderNodeVolumePrincipled")
    texcoord = tree.nodes.new("ShaderNodeTexCoord")
    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.noise_dimensions = "3D"
    set_input(noise, "Scale", 3.7)
    generated = next(socket for socket in texcoord.outputs if socket.name == "Generated")
    vector = next(socket for socket in noise.inputs if socket.name == "Vector")
    tree.links.new(generated, vector)
    connect_named(tree, noise, "Factor", principled, "Density")
    connect_named(tree, principled, "Volume", output, "Volume")
    volume_object.data.materials.append(material)
    scene.world = bpy.data.worlds.new("Spatial Volume Black World")
    save_blend(os.path.join(out_dir, "volume_spatially_varying.blend"))


def build_temperature_volume_fixture(out_dir):
    import openvdb

    vdb_path = os.path.join(out_dir, "temperature_grid.vdb")
    transform = openvdb.createLinearTransform(0.125)
    density = openvdb.FloatGrid()
    density.name = "density"
    density.gridClass = "fog volume"
    density.transform = transform
    density.fill((0, 0, 0), (7, 7, 7), 0.7)
    temperature = openvdb.FloatGrid()
    temperature.name = "temperature"
    temperature.gridClass = "fog volume"
    temperature.transform = transform
    accessor = temperature.getAccessor()
    for z in range(8):
        for y in range(8):
            for x in range(8):
                accessor.setValueOn((x, y, z), 1.8 + 0.6 * x + 0.6 * z)
    openvdb.write(vdb_path, [density, temperature])

    width = 32
    height = 32
    samples = 4096
    scene = configure_scene()
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.cycles.samples = samples
    scene.cycles.seed = 0
    scene.cycles.max_bounces = 4
    scene.camera.data.ortho_scale = 2.5
    set_black_world(scene)
    vdb_path = os.path.join(out_dir, "temperature_grid.vdb")
    volume_object_path = os.path.abspath(vdb_path)
    bpy.ops.object.volume_import(filepath=volume_object_path)
    volume_object = bpy.context.object
    volume_object.name = "Temperature Grid Volume"
    volume_object.location = (-0.4375, -0.4375, -0.4375)
    volume_object.data.filepath = volume_object_path
    volume_object.data.grids.load()
    material = bpy.data.materials.new("Volume Temperature Grid Material")
    material.use_nodes = True
    material.cycles.volume_interpolation = "LINEAR"
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    volume = tree.nodes.new("ShaderNodeVolumePrincipled")
    # Density zero removes stochastic scattering while Cycles still samples blackbody emission attributes.
    set_input(volume, "Density", 0.0)
    set_input(volume, "Blackbody Intensity", 0.2)
    set_input(volume, "Temperature Attribute", "temperature")
    connect_named(tree, volume, "Volume", output, "Volume")
    volume_object.data.materials.append(material)
    volume_object.data.filepath = "//temperature_grid.vdb"
    save_blend(os.path.join(out_dir, "volume_temperature_grid.blend"))
    pixels = render_image(out_dir, "volume_temperature_grid_reference", width, height)
    eevee_pixels = render_engine_image(
        scene, out_dir, "volume_temperature_grid_eevee_reference", width, height, "eevee"
    )
    cases = [
        {"name": "temperature_grid_cool", "region": [11, 12, 4, 8]},
        {"name": "temperature_grid_hot", "region": [18, 12, 4, 8]},
    ]
    for case in cases:
        case["mean"] = region_mean(pixels, width, height, case["region"])
        x = case["region"][0] + case["region"][2] // 2
        y = case["region"][1] + case["region"][3] // 2
        case["eevee_probe"] = [x, y]
        case["eevee_rgb"] = pixel_rgb(eevee_pixels, width, height, x, y)
    with open(os.path.join(out_dir, "volume_temperature_grid.json"), "w", encoding="utf-8") as output:
        json.dump({
            "resolution": [width, height],
            "samples": samples,
            "eevee_samples": 1,
            "seed": 0,
            "max_bounces": 4,
            "renderer": {
                "cycles": "Blender 5.2.2 Cycles CPU; 4096 samples; Raw; box filter 0.001",
                "eevee": "Blender 5.2.2 EEVEE Next; 1 TAA sample; Raw",
            },
            "source": "Blender 5.2.2 source/blender/nodes/shader/nodes/node_shader_volume_principled.cc, source/blender/gpu/shaders/material/gpu_shader_material_volume_principled.glsl, and intern/cycles/scene/shader_nodes.cpp; EEVEE VDB temperature attributes use half-float volume-grid taps",
            "cases": cases,
        }, output, indent=2, allow_nan=False)
        output.write("\n")


def build_refusal_fixtures(out_dir):
    scene = configure_scene()
    material = bpy.data.materials.new("Unsupported Attribute Target")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    emission = tree.nodes.new("ShaderNodeEmission")
    attribute = tree.nodes.new("ShaderNodeAttribute")
    attribute.attribute_type = "OBJECT"
    attribute.attribute_name = "Probe"
    connect_named(tree, attribute, "Color", emission, "Color")
    connect_named(tree, emission, "Emission", output, "Surface")
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, 0.0))
    bpy.context.object.name = "attribute_object_target"
    bpy.context.object["Probe"] = 0.75
    bpy.context.object.data.materials.append(material)
    save_blend(os.path.join(out_dir, "attribute_object.blend"))

    scene = configure_scene()
    world = bpy.data.worlds.new("Non-default Sky World")
    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputWorld")
    background = tree.nodes.new("ShaderNodeBackground")
    sky = tree.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "PREETHAM"
    connect_named(tree, sky, "Color", background, "Color")
    connect_named(tree, background, "Background", output, "Surface")
    scene.world = world
    save_blend(os.path.join(out_dir, "sky_nondefault.blend"))

    scene = configure_scene()
    world = bpy.data.worlds.new("Default Multiple Scattering Sky")
    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputWorld")
    background = tree.nodes.new("ShaderNodeBackground")
    sky = tree.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "MULTIPLE_SCATTERING"
    connect_named(tree, sky, "Color", background, "Color")
    connect_named(tree, background, "Background", output, "Surface")
    scene.world = world
    save_blend(os.path.join(out_dir, "sky_default.blend"))
    refusals = {
        "cases": [
            {
                "name": "attribute_object",
                "blend": "attribute_object.blend",
                "loss_feature_id": "blender.material.shader_node.attribute_type",
                "error_code": "UNSUPPORTED_FEATURE",
                "feature_id": "shader.node.ShaderNodeAttribute.attribute_type.OBJECT",
            },
            {
                "name": "sky_nondefault",
                "blend": "sky_nondefault.blend",
                "loss_feature_id": "blender.world.sky_settings",
                "error_code": "UNSUPPORTED_FEATURE",
                "feature_id": "world.node.ShaderNodeTexSky.sky_type",
            },
            {
                "name": "sky_default_multiple_scattering",
                "blend": "sky_default.blend",
                "loss_feature_id": "blender.world.sky_model",
                "error_code": "UNSUPPORTED_FEATURE",
                "feature_id": "world.node.ShaderNodeTexSky.multiple_scattering_model",
            },
            {
                "name": "volume_spatially_varying",
                "blend": "volume_spatially_varying.blend",
                "loss_feature_id": "shader.volume.spatially_varying_graph",
                "error_code": "UNSUPPORTED_FEATURE",
                "feature_id": "shader.volume.spatially_varying_graph",
            },
        ],
    }
    with open(os.path.join(out_dir, "refusals.json"), "w", encoding="utf-8") as output:
        json.dump(refusals, output, indent=2)
        output.write("\n")

def build_geometry_attribute_import_fixtures(out_dir):
    cases = []
    for output_socket, blender_socket_name in (("Color", "Color"), ("Fac", "Factor")):
        name = "geometry_attribute_point_float_" + output_socket.lower()
        scene = configure_scene()
        material = bpy.data.materials.new(name + " Material")
        material.use_nodes = True
        tree = material.node_tree
        tree.nodes.clear()
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        emission = tree.nodes.new("ShaderNodeEmission")
        attribute = tree.nodes.new("ShaderNodeAttribute")
        attribute.attribute_type = "GEOMETRY"
        attribute.attribute_name = "Probe"
        connect_named(tree, attribute, blender_socket_name, emission, "Color")
        connect_named(tree, emission, "Emission", output, "Surface")
        bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, 0.0))
        mesh = bpy.context.object
        mesh.name = name + " Mesh"
        probe = mesh.data.attributes.new("Probe", "FLOAT", "POINT")
        for datum in probe.data:
            datum.value = 0.75
        mesh.data.materials.append(material)
        blend_name = name + ".blend"
        save_blend(os.path.join(out_dir, blend_name))
        cases.append({
            "name": name,
            "blend": blend_name,
            "attribute_name": "Probe",
            "attribute_type": "GEOMETRY",
            "data_type": "FLOAT",
            "domain": "POINT",
            "output_socket": output_socket,
            "expect_loss": output_socket == "Color",
            "loss_feature_id": "shader.node.ShaderNodeAttribute.geometry_attribute.unsupported",
        })
    with open(
        os.path.join(out_dir, "attribute_geometry_import_losses.json"), "w", encoding="utf-8"
    ) as output:
        json.dump({
            "source": "Blender 5.2.2 source/blender/nodes/shader/nodes/node_shader_attribute.cc; Mesh POINT FLOAT attribute `Probe` through GEOMETRY Attribute outputs",
            "cases": cases,
        }, output, indent=2)
        output.write("\n")


def render_engine_image(scene, out_dir, name, width, height, engine, samples=1):
    if engine == "eevee":
        scene.render.engine = "BLENDER_EEVEE"
        scene.eevee.taa_render_samples = samples
    else:
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = samples
        scene.cycles.seed = 0
        scene.cycles.max_bounces = 4
    return render_image(out_dir, name, width, height)


def engine_reference(scene, out_dir, name, width, height, regions):
    references = {}
    for engine in ("eevee", "cycles"):
        pixels = render_engine_image(scene, out_dir, name + "_" + engine, width, height, engine)
        references[engine] = [
            region_mean(pixels, width, height, region) for region in regions
        ]
    return references


def build_volume_blackbody_engine_fixture(out_dir):
    import openvdb

    width = 64
    height = 64
    samples = 1
    scene = configure_scene()
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.camera.data.ortho_scale = 4.0
    set_black_world(scene)

    cases = []
    for temperature, centre_x in ((1800.0, -1.0), (10200.0, 1.0)):
        name = f"volume_blackbody_{temperature:g}k"
        grid_name = name + ".vdb"
        vdb_path = os.path.join(out_dir, grid_name)
        transform = openvdb.createLinearTransform(0.125)
        density_grid = openvdb.FloatGrid()
        density_grid.name = "density"
        density_grid.gridClass = "fog volume"
        density_grid.transform = transform
        density_grid.fill((0, 0, 0), (7, 7, 7), 0.7)
        temperature_grid = openvdb.FloatGrid()
        temperature_grid.name = "temperature"
        temperature_grid.gridClass = "fog volume"
        temperature_grid.transform = transform
        temperature_grid.fill((0, 0, 0), (7, 7, 7), temperature / 1000.0)
        openvdb.write(vdb_path, [density_grid, temperature_grid])

        material = bpy.data.materials.new(name)
        material.use_nodes = True
        tree = material.node_tree
        tree.nodes.clear()
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        volume = tree.nodes.new("ShaderNodeVolumePrincipled")
        set_input(volume, "Density", 0.0)
        set_input(volume, "Emission Strength", 0.0)
        set_input(volume, "Blackbody Intensity", 0.2)
        set_input(volume, "Temperature", 1000.0)
        set_input(volume, "Temperature Attribute", "temperature")
        connect_named(tree, volume, "Volume", output, "Volume")
        bpy.ops.object.volume_import(filepath=os.path.abspath(vdb_path))
        volume_object = bpy.context.object
        volume_object.name = name
        volume_object.location = (centre_x - 0.4375, -0.4375, -0.4375)
        volume_object.data.filepath = os.path.abspath(vdb_path)
        volume_object.data.grids.load()
        volume_object.data.materials.append(material)
        volume_object.data.filepath = "//" + grid_name
        x = round(width / 2 + centre_x * width / 4)
        cases.append({
            "name": name,
            "region": [x - 3, height // 2 - 3, 6, 6],
            "temperature_attribute": temperature / 1000.0,
            "volume_grid": grid_name,
        })

    save_blend(os.path.join(out_dir, "volume_blackbody_engines.blend"))
    references = engine_reference(
        scene, out_dir, "volume_blackbody_engines_reference", width, height,
        [case["region"] for case in cases],
    )
    for index, case in enumerate(cases):
        case["eevee_mean"] = references["eevee"][index]
        case["cycles_mean"] = references["cycles"][index]
    with open(os.path.join(out_dir, "volume_blackbody_engines.json"), "w", encoding="utf-8") as output:
        json.dump({
            "resolution": [width, height],
            "samples": {"eevee": samples, "cycles": samples},
            "seed": 0,
            "density": 0.0,
            "temperature_socket": 1000.0,
            "blackbody_intensity": 0.2,
            "renderer": {
                "eevee": "Blender 5.2.2 EEVEE Next; 1 TAA sample; Raw",
                "cycles": "Blender 5.2.2 Cycles CPU; 1 sample; Raw",
            },
            "source": "Blender 5.2.2 VolumePrincipled blackbody emission from a temperature VDB; Density=0 isolates blackbody from scattering",
            "cases": cases,
        }, output, indent=2, allow_nan=False)
        output.write("\n")

def build_volume_attribute_engine_fixture(out_dir):
    width = 64
    height = 32
    samples = 64
    scene = configure_scene()
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.camera.data.ortho_scale = 4.0
    scene.cycles.samples = samples
    scene.cycles.seed = 0
    scene.eevee.taa_render_samples = samples
    set_black_world(scene)

    bpy.ops.mesh.primitive_cube_add(size=0.875, location=(-1.0, 0.0, 0.0))
    missing_volume_object = bpy.context.object
    missing_volume_object.name = "Missing Temperature Attribute Mesh Volume"
    material = bpy.data.materials.new("Missing Temperature Attribute Mesh Volume Material")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    volume = tree.nodes.new("ShaderNodeVolumePrincipled")
    set_input(volume, "Density", 0.0)
    set_input(volume, "Emission Strength", 0.0)
    set_input(volume, "Blackbody Intensity", 0.2)
    set_input(volume, "Temperature", 3000.0)
    set_input(volume, "Temperature Attribute", "missing_temperature")
    connect_named(tree, volume, "Volume", output, "Volume")
    missing_volume_object.data.materials.append(material)

    bpy.ops.mesh.primitive_cube_add(size=0.875, location=(1.0, 0.0, 0.0))
    mesh_object = bpy.context.object
    mesh_object.name = "Mesh Volume Color Attribute"
    colors = mesh_object.data.attributes.new("mesh_tint", "FLOAT_COLOR", "POINT")
    for datum in colors.data:
        datum.color = (0.1, 0.8, 0.2, 1.0)
    mesh_material = bpy.data.materials.new("Mesh Volume Color Attribute Material")
    mesh_material.use_nodes = True
    mesh_tree = mesh_material.node_tree
    mesh_tree.nodes.clear()
    mesh_output = mesh_tree.nodes.new("ShaderNodeOutputMaterial")
    mesh_volume = mesh_tree.nodes.new("ShaderNodeVolumePrincipled")
    set_input(mesh_volume, "Density", 0.8)
    set_input(mesh_volume, "Color", (1.0, 1.0, 1.0, 1.0))
    set_input(mesh_volume, "Color Attribute", "mesh_tint")
    set_input(mesh_volume, "Emission Strength", 0.0)
    set_input(mesh_volume, "Blackbody Intensity", 0.0)
    connect_named(mesh_tree, mesh_volume, "Volume", mesh_output, "Volume")
    mesh_object.data.materials.append(mesh_material)

    light_data = bpy.data.lights.new("Mesh Volume Attribute Area", "AREA")
    light_data.shape = "DISK"
    light_data.size = 1.5
    light_data.energy = 500.0
    light = bpy.data.objects.new("Mesh Volume Attribute Area", light_data)
    scene.collection.objects.link(light)
    light.location = (0.0, -1.0, 3.0)

    cases = [
        {
            "name": "missing_temperature_attribute",
            "region": [13, 13, 6, 6],
            "object_type": "MESH_VOLUME",
            "attribute": "missing_temperature",
            "temperature_socket": 3000.0,
        },
        {
            "name": "mesh_volume_color_attribute",
            "region": [45, 13, 6, 6],
            "object_type": "MESH_VOLUME",
            "attribute": "mesh_tint",
            "mesh_color": [0.1, 0.8, 0.2, 1.0],
        },
    ]
    save_blend(os.path.join(out_dir, "volume_attribute_engines.blend"))
    references = {}
    for engine in ("eevee", "cycles"):
        pixels = render_engine_image(
            scene, out_dir, "volume_attribute_engines_" + engine, width, height, engine,
            samples=samples,
        )
        references[engine] = pixels
    for case in cases:
        region = case["region"]
        case["eevee_rgb"] = region_mean(
            references["eevee"], width, height, region
        )
        case["cycles_rgb"] = region_mean(
            references["cycles"], width, height, region
        )
    if any(cases[0]["eevee_rgb"]) or not any(cases[0]["cycles_rgb"]):
        raise RuntimeError("missing Temperature Attribute did not distinguish EEVEE and Cycles")
    if cases[1]["eevee_rgb"] == cases[1]["cycles_rgb"]:
        raise RuntimeError("mesh-volume Color Attribute references did not distinguish engines")
    with open(os.path.join(out_dir, "volume_attribute_engines.json"), "w", encoding="utf-8") as output:
        json.dump({
            "resolution": [width, height],
            "samples": {"eevee": samples, "cycles": samples},
            "seed": 0,
            "renderer": {
                "eevee": "Blender 5.2.2 EEVEE Next; 64 TAA samples; Raw",
                "cycles": "Blender 5.2.2 Cycles CPU; 64 samples; Raw",
            },
            "source": "Blender 5.2.2 source/blender/nodes/shader/nodes/node_shader_volume_principled.cc, source/blender/gpu/shaders/material/gpu_shader_material_volume_principled.glsl, and intern/cycles/scene/shader_nodes.cpp",
            "cases": cases,
        }, output, indent=2, allow_nan=False)
        output.write("\n")


def build_environment_axis_seam_fixture(out_dir):
    width = 65
    height = 65
    scene = configure_scene()
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.camera.data.type = "PERSP"
    scene.camera.data.lens = 24.0
    # Keep the seam rays just off the exact-axis GLSL fallback so hardware interpolation ULPs cannot change texels.
    scene.camera.data.shift_x = 0.0006666667
    scene.camera.data.shift_y = 0.0006666667
    scene.eevee.taa_render_samples = 1
    image = image_texture("Environment Axis Seam Image", size=16, height=8)
    world = bpy.data.worlds.new("Environment Axis Seam World")
    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputWorld")
    background = tree.nodes.new("ShaderNodeBackground")
    environment = tree.nodes.new("ShaderNodeTexEnvironment")
    environment.image = image
    environment.projection = "EQUIRECTANGULAR"
    environment.interpolation = "Closest"
    connect_named(tree, environment, "Color", background, "Color")
    connect_named(tree, background, "Background", output, "Surface")
    scene.world = world

    cases = [
        {"name": "x_near_zero_positive_y", "near_zero_component": "x", "probe": [32, 20]},
        {"name": "x_near_zero_negative_y", "near_zero_component": "x", "probe": [32, 44]},
        {"name": "y_near_zero_positive_x", "near_zero_component": "y", "probe": [44, 32]},
        {"name": "y_near_zero_negative_x", "near_zero_component": "y", "probe": [20, 32]},
    ]
    save_blend(os.path.join(out_dir, "environment_axis_seam.blend"))
    pixels = render_engine_image(
        scene, out_dir, "environment_axis_seam_reference", width, height, "eevee"
    )
    for case in cases:
        x, y = case["probe"]
        case["rgb"] = pixel_rgb(pixels, width, height, x, y)
    with open(os.path.join(out_dir, "environment_axis_seam.json"), "w", encoding="utf-8") as output:
        json.dump({
            "resolution": [width, height],
            "samples": 1,
            "renderer": "Blender 5.2.2 EEVEE Next; 1 TAA sample; Raw",
            "source": "Blender 5.2.2 EEVEE EQUIRECTANGULAR near-axis rays, 0.001 rad camera shift",
            "cases": cases,
        }, output, indent=2, allow_nan=False)
        output.write("\n")


def offset_texture_material(name, node_type):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    emission = tree.nodes.new("ShaderNodeEmission")
    texture = tree.nodes.new(node_type)
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    offset = tree.nodes.new("ShaderNodeVectorMath")
    offset.operation = "ADD"
    offset.inputs[1].default_value = (10000.125, -9999.625, 12345.375)
    tree.links.new(coordinates.outputs["Generated"], offset.inputs[0])
    tree.links.new(offset.outputs["Vector"], texture.inputs["Vector"])
    if node_type == "ShaderNodeTexNoise":
        texture.noise_dimensions = "3D"
        set_input(texture, "Scale", 1.7)
        set_input(texture, "Detail", 2.3)
        set_input(texture, "Roughness", 0.53)
        set_input(texture, "Lacunarity", 2.0)
        set_input(texture, "Distortion", 0.55)
        output_name = "Factor"
    else:
        texture.voronoi_dimensions = "3D"
        texture.feature = "F1"
        texture.distance = "EUCLIDEAN"
        set_input(texture, "Scale", 2.3)
        set_input(texture, "Randomness", 0.73)
        output_name = "Distance"
    connect_named(tree, texture, output_name, emission, "Color")
    connect_named(tree, emission, "Emission", output, "Surface")
    return material


def build_shader_gpu_precision_fixture(out_dir):
    width = 64
    height = 64
    scene = configure_scene()
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.camera.data.ortho_scale = 4.0
    scene.eevee.taa_render_samples = 1
    set_black_world(scene)
    cases = []
    for name, node_type, centre_x in (
        ("noise_large_offset", "ShaderNodeTexNoise", -1.0),
        ("voronoi_large_offset", "ShaderNodeTexVoronoi", 1.0),
    ):
        material = offset_texture_material(name, node_type)
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(centre_x, 0.0, 0.0))
        bpy.context.object.name = name
        bpy.context.object.data.materials.append(material)
        x = round(width / 2 + centre_x * width / 4)
        cases.append({
            "name": name,
            "probes": [
                {"x": x - 3, "y": height // 2},
                {"x": x, "y": height // 2},
                {"x": x + 3, "y": height // 2},
            ],
        })

    save_blend(os.path.join(out_dir, "shader_gpu_precision.blend"))
    references = {}
    for engine in ("eevee", "cycles"):
        pixels = render_engine_image(
            scene, out_dir, "shader_gpu_precision_" + engine, width, height, engine
        )
        references[engine] = pixels
    for case in cases:
        for probe in case["probes"]:
            probe["eevee_rgb"] = pixel_rgb(
                references["eevee"], width, height, probe["x"], probe["y"]
            )
            probe["cycles_rgb"] = pixel_rgb(
                references["cycles"], width, height, probe["x"], probe["y"]
            )
    with open(os.path.join(out_dir, "shader_gpu_precision.json"), "w", encoding="utf-8") as output:
        json.dump({
            "resolution": [width, height],
            "samples": {"eevee": 1, "cycles": 1},
            "seed": 0,
            "vector_offset": [10000.125, -9999.625, 12345.375],
            "renderer": {
                "eevee": "Blender 5.2.2 EEVEE Next; 1 TAA sample; Raw",
                "cycles": "Blender 5.2.2 Cycles CPU; 1 sample; Raw",
            },
            "source": "Blender 5.2.2 source/blender/nodes/shader/nodes/node_shader_tex_noise.cc, node_shader_tex_voronoi.cc, and EEVEE GPU shader implementations",
            "cases": cases,
        }, output, indent=2, allow_nan=False)
        output.write("\n")


def build_image_minification_refusal_fixture(out_dir):
    width = 16
    height = 16
    scene = configure_scene()
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.camera.data.ortho_scale = 4.0
    scene.eevee.taa_render_samples = 1
    image = image_texture("Minified Linear Image", size=32)
    material = bpy.data.materials.new("Minified Linear Image Material")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    emission = tree.nodes.new("ShaderNodeEmission")
    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.image = image
    texture.projection = "FLAT"
    texture.interpolation = "Linear"
    texture.extension = "REPEAT"
    connect_named(tree, texture, "Color", emission, "Color")
    connect_named(tree, emission, "Emission", output, "Surface")
    bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0.0, 0.0, 0.0))
    bpy.context.object.name = "image_linear_minifying_plane"
    bpy.context.object.data.materials.append(material)
    save_blend(os.path.join(out_dir, "image_linear_minification.blend"))
    with open(os.path.join(out_dir, "image_linear_minification.json"), "w", encoding="utf-8") as output:
        json.dump({
            "resolution": [width, height],
            "samples": 1,
            "renderer": "Blender 5.2.2 EEVEE Next source fixture; LINEAR Image Texture minified 2:1",
            "cases": [{
                "name": "image_linear_minification",
                "blend": "image_linear_minification.blend",
                "engine": "realtime",
                "error_code": "UNSUPPORTED_FEATURE",
                "feature_id": "shader.node.ShaderNodeTexImage.linear_mip_filtering",
            }],
        }, output, indent=2, allow_nan=False)
        output.write("\n")


def main():
    if bpy.app.version[:3] != (5, 2, 2):
        raise RuntimeError(f"oracle requires Blender 5.2.2, got {bpy.app.version_string}")
    out_dir = os.path.abspath(argv_output())
    args = sys.argv[sys.argv.index("--") + 1:]
    only_voronoi_fold_invariants = (
        bool(args) and args[0] == "--only-voronoi-fold-invariants"
    )
    os.makedirs(out_dir, exist_ok=True)
    if only_voronoi_fold_invariants:
        build_voronoi_fold_invariants_fixture(out_dir)
        print("SHADER_P1_VORONOI_FOLD_INVARIANTS", json.dumps({"output": out_dir}))
        return
    scene = configure_scene()
    cases = []
    checker_cases(cases)
    gradient_cases(cases)
    view_node_cases(cases)
    noise_cases(cases)
    voronoi_cases(cases)
    image_cases(cases)
    attribute_cases(cases)
    default_coordinate_cases(cases)
    if len(cases) > GRID * GRID:
        raise RuntimeError(f"chart has {len(cases)} cases but only {GRID * GRID} tiles")
    save_blend(os.path.join(out_dir, "shader_nodes.blend"))
    reference = collect_pixels(cases, out_dir)
    with open(os.path.join(out_dir, "shader_nodes.json"), "w", encoding="utf-8") as output:
        json.dump(reference, output, indent=2, allow_nan=False)
        output.write("\n")
    build_glass_distribution_fixture(out_dir)
    build_physics_fixture(out_dir)
    build_voronoi_clamp_fixture(out_dir)
    build_voronoi_fold_fields_fixture(out_dir)
    build_texcoord_normal_fixture(out_dir)
    build_negative_scale_facing_fixture(out_dir)
    build_realtime_glass_fixture(out_dir)
    build_realtime_colored_glass_fixture(out_dir)
    build_displacement_fixture(out_dir)
    build_environment_world_fixture(out_dir)
    build_realtime_volume_world_ambient_fixture(out_dir)
    build_realtime_anisotropic_world_volume_refusal_fixture(out_dir)
    build_mapping_refusal_fixtures(out_dir)
    build_spatial_volume_fixture(out_dir)
    build_temperature_volume_fixture(out_dir)
    build_volume_blackbody_engine_fixture(out_dir)
    build_volume_attribute_engine_fixture(out_dir)
    build_environment_axis_seam_fixture(out_dir)
    build_shader_gpu_precision_fixture(out_dir)
    build_image_minification_refusal_fixture(out_dir)
    build_refusal_fixtures(out_dir)
    build_geometry_attribute_import_fixtures(out_dir)
    print("SHADER_P1_ORACLE", json.dumps({"cases": len(cases), "output": out_dir}))


if __name__ == "__main__":
    main()
