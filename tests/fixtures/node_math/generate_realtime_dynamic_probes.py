"""Blender 5.2.2 per-pixel EEVEE oracles for shader-node math kernels."""
import bpy
import json
import math
import os
import sys
import tempfile

if bpy.app.version != (5, 2, 2):
    raise RuntimeError("dynamic node math oracle requires Blender 5.2.2, found " + bpy.app.version_string)

arguments = sys.argv[sys.argv.index("--") + 1:]
blend_path = arguments[0]
capture_path = arguments[1] if len(arguments) > 1 else None

CASES = [
    ("dynamic_math_arcsine_in_domain", "scalar"),
    ("dynamic_math_arccosine_in_domain", "scalar"),
    ("dynamic_math_arcsine_out_of_range", "scalar"),
    ("constant_math_arcsine_out_of_range", "scalar"),
    ("dynamic_math_arccosine_out_of_range", "scalar"),
    ("constant_math_arccosine_out_of_range", "scalar"),
    ("constant_map_range_arcsine_out_of_range", "scalar"),
    ("dynamic_math_compare_defined", "scalar"),
    ("dynamic_math_compare_sub_half_float_epsilon", "scalar"),
    ("constant_math_compare_sub_half_float_epsilon", "scalar"),
    ("dynamic_math_inverse_sqrt_positive", "scalar"),
    ("dynamic_math_power_near_integer", "scalar"),
    ("constant_math_power_near_integer", "scalar"),
    ("dynamic_math_power_negative_noninteger", "scalar"),
    ("constant_math_power_negative_noninteger", "scalar"),
    ("dynamic_math_modulo", "scalar"),
    ("dynamic_math_modulo_precision_boundary", "scalar"),
    ("constant_math_modulo_precision_boundary", "scalar"),
    ("constant_math_modulo_quotient_out_of_range", "scalar"),
    ("dynamic_vector_sign", "vector"),
    ("dynamic_vector_power_integer", "vector"),
    ("dynamic_vector_power_near_integer", "vector"),
    ("dynamic_vector_modulo", "vector"),
    ("dynamic_vector_modulo_precision_boundary", "vector"),
    ("constant_vector_modulo_precision_boundary", "vector"),
    ("constant_vector_modulo_quotient_out_of_range", "vector"),
    ("dynamic_vector_reflect_tiny_normal", "vector"),
    ("dynamic_vector_refract_tiny_normal", "vector"),
    ("dynamic_map_range_linear_zero_span", "scalar"),
    ("dynamic_map_range_stepped_zero_steps", "scalar"),
    ("dynamic_color_ramp_constant_table", "color"),
    ("dynamic_color_ramp_linear_two_stop", "color"),
    ("dynamic_color_ramp_linear_table_near_start", "color"),
    ("dynamic_color_ramp_linear_table_near_end", "color"),
    ("dynamic_float_curve_clip", "scalar"),
]


def set_input(tree, socket, value):
    if isinstance(value, bpy.types.NodeSocket):
        tree.links.new(value, socket)
    else:
        socket.default_value = value


def math_output(tree, operation, first, second=0.0, third=0.0):
    node = tree.nodes.new("ShaderNodeMath")
    node.operation = operation
    for index, value in enumerate((first, second, third)):
        set_input(tree, node.inputs[index], value)
    return node.outputs["Value"]


def vector_output(tree, operation, first, second=(0.0, 0.0, 0.0), third=(0.0, 0.0, 0.0), scale=1.0):
    node = tree.nodes.new("ShaderNodeVectorMath")
    node.operation = operation
    for index, value in enumerate((first, second, third, scale)):
        set_input(tree, node.inputs[index], value)
    return node.outputs["Vector"]


def combine_vector(tree, x, y, z):
    node = tree.nodes.new("ShaderNodeCombineXYZ")
    for name, value in (("X", x), ("Y", y), ("Z", z)):
        set_input(tree, node.inputs[name], value)
    return node.outputs["Vector"]


def texture_x(tree):
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    separate = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(coordinates.outputs["UV"], separate.inputs["Vector"])
    return separate.outputs["X"]


def build_probe(tree, identifier):

    x = texture_x(tree)
    if identifier in ("dynamic_math_arcsine_in_domain", "dynamic_math_arccosine_in_domain"):
        value = math_output(
            tree,
            "ADD",
            0.25,
            math_output(tree, "MULTIPLY", math_output(tree, "SUBTRACT", x, 0.5), 1.0e-4),
        )
        operation = "ARCSINE" if identifier.endswith("arcsine_in_domain") else "ARCCOSINE"
        return math_output(tree, operation, value)
    if identifier == "dynamic_math_arcsine_out_of_range":
        value = math_output(tree, "ADD", 1.25, math_output(tree, "MULTIPLY", x, 0.5))
        return math_output(tree, "ARCSINE", value)
    if identifier == "constant_math_arcsine_out_of_range":
        return math_output(tree, "ARCSINE", 1.5)
    if identifier == "dynamic_math_arccosine_out_of_range":
        value = math_output(tree, "SUBTRACT", -1.25, math_output(tree, "MULTIPLY", x, 0.5))
        return math_output(tree, "ARCCOSINE", value)
    if identifier == "constant_math_arccosine_out_of_range":
        return math_output(tree, "ARCCOSINE", -1.5)
    if identifier == "constant_map_range_arcsine_out_of_range":
        map_range = tree.nodes.new("ShaderNodeMapRange")
        map_range.clamp = False
        map_range.inputs["Value"].default_value = 5.0
        map_range.inputs["From Min"].default_value = 1.0
        map_range.inputs["From Max"].default_value = 1.0
        map_range.inputs["To Min"].default_value = 2.0
        return math_output(tree, "ARCSINE", map_range.outputs["Result"])
    if identifier == "dynamic_math_compare_sub_half_float_epsilon":
        difference = math_output(
            tree,
            "ADD",
            5.0e-6,
            math_output(tree, "MULTIPLY", x, 1.0e-7),
        )
        return math_output(tree, "COMPARE", difference, 0.0, 0.0)
    if identifier == "constant_math_compare_sub_half_float_epsilon":
        return math_output(tree, "COMPARE", 5.0e-6, 0.0, 0.0)
    if identifier == "constant_math_power_near_integer":
        return math_output(tree, "POWER", -2.0, 2.0001)
    if identifier == "dynamic_math_power_negative_noninteger":
        exponent = math_output(
            tree,
            "ADD",
            0.25,
            math_output(tree, "MULTIPLY", x, 0.1),
        )
        return math_output(tree, "POWER", -2.0, exponent)
    if identifier == "constant_math_power_negative_noninteger":
        return math_output(tree, "POWER", -2.0, 0.25)
    if identifier == "dynamic_math_modulo_precision_boundary":
        value = math_output(
            tree,
            "ADD",
            1.0,
            math_output(tree, "MULTIPLY", x, 0.2),
        )
        return math_output(tree, "MODULO", value, 0.1)
    if identifier == "constant_math_modulo_precision_boundary":
        return math_output(tree, "MODULO", 1.2, 0.1)
    if identifier == "constant_math_modulo_quotient_out_of_range":
        large = math_output(
            tree,
            "MULTIPLY",
            math_output(tree, "MULTIPLY", 10000.0, 10000.0),
            100.0,
        )
        return math_output(tree, "MODULO", large, 1.0)
    if identifier == "dynamic_vector_modulo_precision_boundary":
        vector = combine_vector(
            tree,
            math_output(tree, "ADD", 1.0, math_output(tree, "MULTIPLY", x, 0.2)),
            math_output(tree, "ADD", 0.7, math_output(tree, "MULTIPLY", x, 0.05)),
            math_output(tree, "ADD", 0.4, math_output(tree, "MULTIPLY", x, 0.03)),
        )
        return vector_output(tree, "MODULO", vector, (0.1, 0.2, 0.3))
    if identifier == "constant_vector_modulo_precision_boundary":
        return vector_output(tree, "MODULO", (1.2, 0.7, 0.4), (0.1, 0.2, 0.3))
    if identifier == "constant_vector_modulo_quotient_out_of_range":
        x_value = math_output(
            tree,
            "MULTIPLY",
            math_output(tree, "MULTIPLY", 10000.0, 10000.0),
            100.0,
        )
        y_value = math_output(tree, "MULTIPLY", 10000.0, 10000.0)
        vector = combine_vector(tree, x_value, y_value, 0.0)
        return vector_output(tree, "MODULO", vector, (1.0, 1.0, 1.0))
    if identifier == "dynamic_math_compare_defined":
        difference = math_output(
            tree,
            "ADD",
            0.0005,
            math_output(tree, "MULTIPLY", math_output(tree, "SUBTRACT", x, 0.5), 1.0e-4),
        )
        return math_output(tree, "COMPARE", difference, 0.0, 0.001)
    if identifier == "dynamic_math_inverse_sqrt_positive":
        value = math_output(
            tree,
            "ADD",
            1.0,
            math_output(tree, "MULTIPLY", math_output(tree, "SUBTRACT", x, 0.5), 1.0e-4),
        )
        return math_output(tree, "INVERSE_SQRT", value)
    if identifier == "dynamic_math_power_integer":
        exponent = math_output(
            tree,
            "ROUND",
            math_output(tree, "ADD", 2.1, math_output(tree, "MULTIPLY", math_output(tree, "SUBTRACT", x, 0.5), 0.1)),
        )
        return math_output(tree, "POWER", -2.0, exponent)
    if identifier == "dynamic_math_power_near_integer":
        exponent = math_output(
            tree,
            "ADD",
            2.0001,
            math_output(tree, "MULTIPLY", x, 0.0001),
        )
        return math_output(tree, "POWER", -2.0, exponent)
    if identifier == "dynamic_math_modulo":
        centered = math_output(tree, "SUBTRACT", x, 0.5)
        inside = math_output(tree, "COMPARE", centered, 0.0, 0.1)
        value = math_output(
            tree,
            "ADD",
            1.23,
            math_output(tree, "MULTIPLY", math_output(tree, "SUBTRACT", 1.0, inside), 0.1),
        )
        return math_output(tree, "MODULO", value, 0.1)
    if identifier == "dynamic_vector_sign":
        centered = math_output(tree, "SUBTRACT", x, 0.5)
        inside = math_output(tree, "COMPARE", centered, 0.0, 0.1)
        zero_at_center = math_output(tree, "SUBTRACT", 1.0, inside)
        vector = combine_vector(tree, zero_at_center, 2.0, 3.0)
        signed = vector_output(tree, "SIGN", vector)
        return signed
    if identifier == "dynamic_vector_power_integer":
        exponent = math_output(
            tree,
            "ROUND",
            math_output(tree, "ADD", 2.1, math_output(tree, "MULTIPLY", math_output(tree, "SUBTRACT", x, 0.5), 0.1)),
        )
        exponent_vector = combine_vector(tree, exponent, 2.0, 2.0)
        power = vector_output(tree, "POWER", (-0.5, -0.5, -0.5), exponent_vector)
        return vector_output(tree, "ABSOLUTE", power)
    if identifier == "dynamic_vector_power_near_integer":
        exponent = combine_vector(
            tree,
            math_output(tree, "ADD", 1.0, math_output(tree, "MULTIPLY", x, 0.0001)),
            2.0,
            0.0,
        )
        power = vector_output(tree, "POWER", (-0.5, -0.5, -0.5), exponent)
        return vector_output(tree, "ABSOLUTE", power)
    if identifier == "dynamic_vector_modulo":
        centered = math_output(tree, "SUBTRACT", x, 0.5)
        inside = math_output(tree, "COMPARE", centered, 0.0, 0.1)
        first = math_output(
            tree,
            "ADD",
            1.23,
            math_output(tree, "MULTIPLY", math_output(tree, "SUBTRACT", 1.0, inside), 0.1),
        )
        vector = combine_vector(tree, first, 0.7, 0.4)
        return vector_output(tree, "MODULO", vector, (0.1, 0.2, 0.3))
    if identifier in ("dynamic_vector_reflect_tiny_normal", "dynamic_vector_refract_tiny_normal"):
        normal = combine_vector(tree, math_output(tree, "MULTIPLY", x, 2.0e-20), 0.0, 0.0)
        operation = "REFLECT" if identifier.endswith("reflect_tiny_normal") else "REFRACT"
        return vector_output(tree, operation, (1.0, 0.0, 0.0), normal, scale=1.0)
    if identifier == "dynamic_map_range_linear_zero_span":
        node = tree.nodes.new("ShaderNodeMapRange")
        node.data_type = "FLOAT"
        node.interpolation_type = "LINEAR"
        node.clamp = False
        value = math_output(
            tree,
            "ADD",
            0.5,
            math_output(tree, "MULTIPLY", math_output(tree, "SUBTRACT", x, 0.5), 1.0e-4),
        )
        set_input(tree, node.inputs["Value"], value)
        node.inputs["From Min"].default_value = 0.5
        node.inputs["From Max"].default_value = 0.5
        node.inputs["To Min"].default_value = 0.25
        node.inputs["To Max"].default_value = 0.75
        return node.outputs["Result"]
    if identifier == "dynamic_map_range_stepped_zero_steps":
        node = tree.nodes.new("ShaderNodeMapRange")
        node.data_type = "FLOAT"
        node.interpolation_type = "STEPPED"
        node.clamp = False
        set_input(tree, node.inputs["Value"], x)
        node.inputs["From Min"].default_value = 0.0
        node.inputs["From Max"].default_value = 1.0
        node.inputs["To Min"].default_value = 0.25
        node.inputs["To Max"].default_value = 0.75
        node.inputs["Steps"].default_value = 0.0
        return node.outputs["Result"]
    if identifier == "dynamic_color_ramp_constant_table":
        node = tree.nodes.new("ShaderNodeValToRGB")
        ramp = node.color_ramp
        ramp.interpolation = "CONSTANT"
        ramp.color_mode = "RGB"
        ramp.elements[0].position = 0.0
        ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
        ramp.elements[1].position = 1.0
        ramp.elements[1].color = (0.0, 0.0, 1.0, 1.0)
        middle = ramp.elements.new(0.501)
        middle.color = (1.0, 0.0, 0.0, 1.0)
        factor = math_output(
            tree,
            "ADD",
            0.5011,
            math_output(tree, "MULTIPLY", math_output(tree, "SUBTRACT", x, 0.5), 1.0e-4),
        )
        set_input(tree, node.inputs["Fac"], factor)
        return node.outputs["Color"]
    if identifier == "dynamic_color_ramp_linear_two_stop":
        node = tree.nodes.new("ShaderNodeValToRGB")
        ramp = node.color_ramp
        ramp.interpolation = "LINEAR"
        ramp.color_mode = "RGB"
        ramp.elements[0].position = 0.2
        ramp.elements[0].color = (0.9, 0.1, 0.2, 0.3)
        ramp.elements[1].position = 0.8
        ramp.elements[1].color = (0.1, 0.8, 0.4, 0.7)
        factor = math_output(
            tree,
            "ADD",
            0.2011,
            math_output(tree, "MULTIPLY", math_output(tree, "SUBTRACT", x, 0.5), 1.0e-4),
        )
        set_input(tree, node.inputs["Fac"], factor)
        return node.outputs["Color"]
    if identifier in (
        "dynamic_color_ramp_linear_table_near_start",
        "dynamic_color_ramp_linear_table_near_end",
    ):
        node = tree.nodes.new("ShaderNodeValToRGB")
        ramp = node.color_ramp
        ramp.interpolation = "LINEAR"
        ramp.color_mode = "RGB"
        ramp.elements[0].position = 0.2
        ramp.elements[0].color = (0.9, 0.1, 0.2, 0.3)
        ramp.elements[1].position = 0.8
        ramp.elements[1].color = (0.1, 0.8, 0.4, 0.7)
        middle = ramp.elements.new(0.5)
        middle.color = (0.15, 0.25, 0.95, 0.55)
        factor_base = 0.2025 if identifier.endswith("near_start") else 0.7975
        factor = math_output(
            tree,
            "ADD",
            factor_base,
            math_output(tree, "MULTIPLY", math_output(tree, "SUBTRACT", x, 0.5), 1.0e-4),
        )
        set_input(tree, node.inputs["Fac"], factor)
        return node.outputs["Color"]
    if identifier == "dynamic_float_curve_clip":
        node = tree.nodes.new("ShaderNodeFloatCurve")
        mapping = node.mapping
        mapping.extend = "EXTRAPOLATED"
        mapping.use_clip = True
        mapping.clip_min_x, mapping.clip_min_y = 0.1, 0.12
        mapping.clip_max_x, mapping.clip_max_y = 0.9, 0.35
        curve = mapping.curves[0]
        curve.points[0].location = (0.0, 0.08)
        curve.points[-1].location = (1.0, 0.92)
        curve.points[0].handle_type = "VECTOR"
        curve.points[-1].handle_type = "VECTOR"
        mapping.update()
        value = math_output(
            tree,
            "ADD",
            1.0,
            math_output(tree, "MULTIPLY", math_output(tree, "SUBTRACT", x, 0.5), 1.0e-4),
        )
        set_input(tree, node.inputs["Value"], value)
        return node.outputs["Value"]
    raise RuntimeError("unknown dynamic shader probe " + identifier)


def make_material(identifier, kind):
    material = bpy.data.materials.new(identifier)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    result = build_probe(tree, identifier)
    emission = tree.nodes.new("ShaderNodeEmission")
    if kind == "scalar":
        emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        tree.links.new(result, emission.inputs["Strength"])
    else:
        emission.inputs["Strength"].default_value = 1.0
        tree.links.new(result, emission.inputs["Color"])
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def build_scene(probes):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.resolution_x = len(probes) * 8
    scene.render.resolution_y = 8
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    scene.render.image_settings.color_mode = "RGBA"

    materials = [make_material(identifier, kind) for identifier, kind in probes]
    vertices, faces = [], []
    for column in range(len(probes)):
        x = float(column)
        base = len(vertices)
        vertices.extend([(x, 0.0, 0.0), (x + 1.0, 0.0, 0.0),
                         (x + 1.0, 1.0, 0.0), (x, 1.0, 0.0)])
        faces.append((base, base + 1, base + 2, base + 3))
    mesh = bpy.data.meshes.new("Per-pixel EEVEE math probes")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for polygon in mesh.polygons:
        column = polygon.index
        polygon.material_index = column
        for loop_index in polygon.loop_indices:
            vertex = mesh.vertices[mesh.loops[loop_index].vertex_index].co
            uv_layer.data[loop_index].uv = (vertex.x - column - 0.0625, vertex.y - 0.0625)
    for material in materials:
        mesh.materials.append(material)
    obj = bpy.data.objects.new("Per-pixel EEVEE math probes", mesh)
    scene.collection.objects.link(obj)

    camera_data = bpy.data.cameras.new("Per-pixel EEVEE math probe camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = float(len(probes))
    camera = bpy.data.objects.new("Per-pixel EEVEE math probe camera", camera_data)
    camera.location = (len(probes) / 2.0, 0.5, 10.0)
    scene.collection.objects.link(camera)
    scene.camera = camera


def capture(probes):
    scene = bpy.context.scene
    with tempfile.TemporaryDirectory(prefix="potter-node-math-dynamic-") as directory:
        image_path = os.path.join(directory, "eevee_dynamic.exr")
        scene.render.filepath = image_path
        scene.render.use_file_extension = False
        bpy.ops.render.render(write_still=True)
        image = bpy.data.images.load(image_path, check_existing=False)
        try:
            pixels = [0.0] * (image.size[0] * image.size[1] * 4)
            image.pixels.foreach_get(pixels)
            results = {}
            for column, (identifier, kind) in enumerate(probes):
                offset = (4 * image.size[0] + column * 8 + 4) * 4
                raw = pixels[offset:offset + 3]
                if not all(math.isfinite(component) for component in raw):
                    raise RuntimeError("non-finite EEVEE output for " + identifier)
                results[identifier] = raw[0] if kind == "scalar" else raw
            return results
        finally:
            bpy.data.images.remove(image)


build_scene(CASES)
eevee_results = capture(CASES)
bpy.ops.wm.save_as_mainfile(filepath=blend_path)
scene = bpy.context.scene
result = {
    "engine": scene.render.engine,
    "resolution": [scene.render.resolution_x, scene.render.resolution_y],
    "cases": [{"id": identifier, "kind": kind} for identifier, kind in CASES],
    "eevee": eevee_results,
}
if capture_path:
    with open(capture_path, "w", encoding="utf-8") as capture_file:
        json.dump(result, capture_file, sort_keys=True, indent=2, allow_nan=False)
        capture_file.write("\n")
print("POTTER_NODE_MATH_REALTIME_DYNAMIC=" + json.dumps(
    result, sort_keys=True, allow_nan=False,
))
