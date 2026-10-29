"""Blender 5.2.2 per-pixel shader graphs for realtime math guard fixtures."""
import bpy
import json
import os
import sys

if bpy.app.version != (5, 2, 2):
    raise RuntimeError("realtime node math fixtures require Blender 5.2.2, found " + bpy.app.version_string)

output_directory = sys.argv[sys.argv.index("--") + 1]

CASES = [
    ("shader_math_modulo_quotient_out_of_range", "ShaderNodeMath", "MODULO",
     "math_modulo_overflow", "Value", "scalar"),
    ("shader_vector_math_modulo_quotient_out_of_range", "ShaderNodeVectorMath", "MODULO",
     "vector_modulo_overflow", "Vector", "vector"),
    ("shader_vector_math_normalize_subnormal_input", "ShaderNodeVectorMath", "NORMALIZE",
     "vector_normalize_subnormal", "Vector", "vector"),
    ("shader_math_inverse_sqrt_nonpositive", "ShaderNodeMath", "INVERSE_SQRT",
     "math_inverse_sqrt_domain", "Value", "scalar"),
    ("shader_math_logarithm_base_one", "ShaderNodeMath", "LOGARITHM",
     "math_log_domain", "Value", "scalar"),
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


def combine_vector(tree, x, y, z):
    node = tree.nodes.new("ShaderNodeCombineXYZ")
    for name, value in (("X", x), ("Y", y), ("Z", z)):
        set_input(tree, node.inputs[name], value)
    return node.outputs["Vector"]


def shader_input(tree, node, mode):
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    separate = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(coordinates.outputs["UV"], separate.inputs["Vector"])
    x = separate.outputs["X"]
    large = 4294967296.0
    if mode == "math_modulo_overflow":
        set_input(tree, node.inputs[0], math_output(tree, "MULTIPLY", x, large))
        node.inputs[1].default_value = 1.0
    elif mode == "math_inverse_sqrt_domain":
        centered = math_output(tree, "SUBTRACT", x, 0.5)
        inside = math_output(tree, "COMPARE", centered, 0.0, 0.1)
        value = math_output(tree, "SUBTRACT", 0.5, math_output(tree, "MULTIPLY", inside, 0.5))
        set_input(tree, node.inputs[0], value)
    elif mode == "math_log_domain":
        centered = math_output(tree, "SUBTRACT", x, 0.5)
        inside = math_output(tree, "COMPARE", centered, 0.0, 0.1)
        base = math_output(
            tree,
            "ADD",
            1.0,
            math_output(tree, "MULTIPLY", math_output(tree, "SUBTRACT", 1.0, inside), 0.1),
        )
        node.inputs[0].default_value = 2.0
        set_input(tree, node.inputs[1], base)
    elif mode == "vector_modulo_overflow":
        vector = combine_vector(tree, math_output(tree, "MULTIPLY", x, large), 0.0, 0.0)
        set_input(tree, node.inputs[0], vector)
        node.inputs[1].default_value = (1.0, 1.0, 1.0)
    elif mode == "vector_normalize_subnormal":
        vector = combine_vector(tree, math_output(tree, "MULTIPLY", x, 2.0e-20), 0.0, 0.0)
        set_input(tree, node.inputs[0], vector)


def save_case(identifier, node_type, operation, mode, output, output_kind):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.view_settings.view_transform = "Raw"
    scene.render.resolution_x = 8
    scene.render.resolution_y = 8
    scene.render.resolution_percentage = 100

    material = bpy.data.materials.new(identifier)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    operation_node = tree.nodes.new(node_type)
    operation_node.operation = operation
    shader_input(tree, operation_node, mode)

    emission = tree.nodes.new("ShaderNodeEmission")
    if output_kind == "scalar":
        emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        tree.links.new(operation_node.outputs[output], emission.inputs["Strength"])
    else:
        emission.inputs["Strength"].default_value = 1.0
        tree.links.new(operation_node.outputs[output], emission.inputs["Color"])
    output_node = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(emission.outputs["Emission"], output_node.inputs["Surface"])

    mesh = bpy.data.meshes.new(identifier + "_mesh")
    mesh.from_pydata(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)],
        [],
        [(0, 1, 2, 3)],
    )
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for loop_index in mesh.polygons[0].loop_indices:
        vertex = mesh.vertices[mesh.loops[loop_index].vertex_index].co
        uv_layer.data[loop_index].uv = (vertex.x - 0.0625, vertex.y - 0.0625)
    mesh.materials.append(material)
    obj = bpy.data.objects.new(identifier, mesh)
    scene.collection.objects.link(obj)

    camera_data = bpy.data.cameras.new(identifier + "_camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 1.0
    camera = bpy.data.objects.new(identifier + "_camera", camera_data)
    camera.location = (0.5, 0.5, 10.0)
    scene.collection.objects.link(camera)
    scene.camera = camera
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(output_directory, identifier + ".blend"))


os.makedirs(output_directory, exist_ok=True)
for case in CASES:
    save_case(*case)
print("POTTER_NODE_MATH_REALTIME_REFUSALS=" + json.dumps([case[0] for case in CASES]))
