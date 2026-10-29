"""Blender 5.2.2 Geometry Nodes and Cycles shader node-math references.

The script accepts a JSON array of math/vector cases and a destination .blend path (or `-` for a geometry-only run). It
evaluates one generated mesh through Geometry Nodes and reads back stored attributes, then renders one exact pixel per
shader case through Cycles. Cases marked "shader_cli" are also saved in a minimal blend scene for import and rendering by
the Potter CLI.
"""
import bpy
import json
import math
import sys
import os
import tempfile

if bpy.app.version != (5, 2, 2):
    raise RuntimeError("node math oracle requires Blender 5.2.2, found " + bpy.app.version_string)

with open(sys.argv[sys.argv.index("--") + 1], encoding="utf-8") as source:
    cases = json.load(source)
shader_blend_path = sys.argv[sys.argv.index("--") + 2]
eevee_blend_path = sys.argv[sys.argv.index("--") + 3]


def assert_complete_catalogs():
    material = bpy.data.materials.new("Potter Math Catalog")
    material.use_nodes = True
    math_node = material.node_tree.nodes.new("ShaderNodeMath")
    vector_node = material.node_tree.nodes.new("ShaderNodeVectorMath")
    for kind, node in (("math", math_node), ("vector", vector_node)):
        available = {item.identifier for item in node.bl_rna.properties["operation"].enum_items}
        covered = {case["operation"] for case in cases if case["kind"] == kind}
        if available != covered:
            missing = sorted(available - covered)
            unknown = sorted(covered - available)
            raise RuntimeError("%s catalog mismatch: missing=%s unknown=%s" % (kind, missing, unknown))
    bpy.data.materials.remove(material)


assert_complete_catalogs()


# Case JSON stays finite; Blender float sockets round values outside f32 range to infinity.
F32_MAX = 3.4028234663852886e38


def blender_input(value):
    if isinstance(value, list):
        return [blender_input(component) for component in value]
    if value > F32_MAX:
        return float("inf")
    if value < -F32_MAX:
        return float("-inf")
    return value


def set_inputs(node, case):
    for index, value in enumerate(case["inputs"]):
        node.inputs[index].default_value = blender_input(value)


def build_geometry_reference():
    tree = bpy.data.node_groups.new("Potter Math Oracle", "GeometryNodeTree")
    tree.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes = tree.nodes
    links = tree.links
    mesh_line = nodes.new("GeometryNodeMeshLine")
    mesh_line.inputs["Count"].default_value = 1
    geometry_output = mesh_line.outputs["Mesh"]

    for case in cases:
        operation = nodes.new("ShaderNodeMath" if case["kind"] == "math" else "ShaderNodeVectorMath")
        operation.operation = case["operation"]
        if case["kind"] == "math":
            operation.use_clamp = case.get("clamp", False)
        set_inputs(operation, case)
        store = nodes.new("GeometryNodeStoreNamedAttribute")
        store.domain = "POINT"
        store.data_type = case["data_type"]
        store.inputs["Name"].default_value = case["id"]
        links.new(geometry_output, store.inputs["Geometry"])
        links.new(operation.outputs[case["output"]], store.inputs["Value"])
        geometry_output = store.outputs["Geometry"]

    group_output = nodes.new("NodeGroupOutput")
    links.new(geometry_output, group_output.inputs["Geometry"])
    mesh = bpy.data.meshes.new("Potter Math Oracle Mesh")
    mesh.from_pydata([(0.0, 0.0, 0.0)], [], [])
    obj = bpy.data.objects.new("Potter Math Oracle", mesh)
    bpy.context.collection.objects.link(obj)
    modifier = obj.modifiers.new("Potter Math Oracle", "NODES")
    modifier.node_group = tree
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    evaluated_mesh = evaluated.to_mesh()
    results = {}
    for case in cases:
        attribute = evaluated_mesh.attributes.get(case["id"])
        if attribute is None or len(attribute.data) != 1:
            raise RuntimeError("missing expected evaluated attribute: " + case["id"])
        item = attribute.data[0]
        is_vector = case["data_type"] == "FLOAT_VECTOR"
        raw = list(item.vector) if is_vector else item.value
        finite = all(math.isfinite(value) for value in raw) if is_vector else math.isfinite(raw)
        results[case["id"]] = raw if finite else "non-finite"
    evaluated.to_mesh_clear()
    return results


def make_material(case):
    material = bpy.data.materials.new("Math " + case["id"])
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    operation = tree.nodes.new("ShaderNodeMath" if case["kind"] == "math" else "ShaderNodeVectorMath")
    operation.operation = case["operation"]
    if case["kind"] == "math":
        operation.use_clamp = case.get("clamp", False)
    set_inputs(operation, case)
    if case.get("value_input") is not None:
        value = tree.nodes.new("ShaderNodeValue")
        value.outputs["Value"].default_value = blender_input(case["value_input"])
        tree.links.new(value.outputs["Value"], operation.inputs["Value"])
    emission = tree.nodes.new("ShaderNodeEmission")
    if case["output"] == "Value":
        emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        tree.links.new(operation.outputs[case["output"]], emission.inputs["Strength"])
    else:
        emission.inputs["Strength"].default_value = 1.0
        tree.links.new(operation.outputs[case["output"]], emission.inputs["Color"])
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def build_shader_scene(shader_cases):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    columns = len(shader_cases)
    materials = [make_material(case) for case in shader_cases]
    vertices = []
    faces = []
    for column in range(columns):
        x = float(column)
        base = len(vertices)
        vertices.extend([(x, 0.0, 0.0), (x + 1.0, 0.0, 0.0), (x + 1.0, 1.0, 0.0), (x, 1.0, 0.0)])
        faces.append((base, base + 1, base + 2, base + 3))
    mesh = bpy.data.meshes.new("Potter Math Shader Probes")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    for material in materials:
        mesh.materials.append(material)
    for column, polygon in enumerate(mesh.polygons):
        polygon.material_index = column
    obj = bpy.data.objects.new("Potter Math Shader Probes", mesh)
    scene.collection.objects.link(obj)

    camera_data = bpy.data.cameras.new("Potter Math Probe Camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = float(columns)
    camera = bpy.data.objects.new("Potter Math Probe Camera", camera_data)
    camera.location = (columns / 2.0, 0.5, 10.0)
    scene.collection.objects.link(camera)
    scene.camera = camera

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    scene.cycles.seed = 0
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = 0
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.0001
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.resolution_x = columns * 8
    scene.render.resolution_y = 8
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    scene.render.image_settings.color_mode = "RGBA"
    if scene.world is not None:
        scene.world.color = (0.0, 0.0, 0.0)


def read_shader_reference(shader_cases, image_name):
    scene = bpy.context.scene
    results = {}
    with tempfile.TemporaryDirectory(prefix="potter-math-oracle-") as directory:
        image_path = os.path.join(directory, image_name)
        scene.render.filepath = image_path
        scene.render.use_file_extension = False
        bpy.ops.render.render(write_still=True)
        rendered = bpy.data.images.load(image_path, check_existing=False)
        try:
            if tuple(rendered.size) != (len(shader_cases) * 8, 8):
                raise RuntimeError("unexpected shader probe image size %s" % (tuple(rendered.size),))
            pixels = [0.0] * (rendered.size[0] * rendered.size[1] * 4)
            rendered.pixels.foreach_get(pixels)
            for column, case in enumerate(shader_cases):
                offset = (4 * rendered.size[0] + column * 8 + 4) * 4
                raw = pixels[offset:offset + 3]
                results[case["id"]] = (
                    raw if case["data_type"] == "FLOAT_VECTOR" else raw[0]
                ) if all(math.isfinite(value) for value in raw) else "non-finite"
        finally:
            bpy.data.images.remove(rendered)
    return results


def shader_reference():
    shader_cases = cases
    build_shader_scene(shader_cases)
    results = read_shader_reference(shader_cases, "cycles_probe.exr")

    path_cases = [case for case in cases if case.get("shader_cli", False)]
    if not path_cases:
        raise RuntimeError("no Cycles shader CLI probes were requested")
    build_shader_scene(path_cases)
    bpy.ops.wm.save_as_mainfile(filepath=shader_blend_path)

    eevee_cases = [case for case in cases if case.get("eevee_cli", False)]
    if not eevee_cases:
        raise RuntimeError("no realtime shader probes were requested")
    build_shader_scene(eevee_cases)
    bpy.context.scene.render.engine = "BLENDER_EEVEE"
    eevee_results = read_shader_reference(eevee_cases, "eevee_probe.exr")
    finite_eevee_cases = [
        case for case in eevee_cases
        if not isinstance(eevee_results[case["id"]], str)
    ]
    if not finite_eevee_cases:
        raise RuntimeError("no finite realtime shader probes were captured")
    build_shader_scene(finite_eevee_cases)
    bpy.context.scene.render.engine = "BLENDER_EEVEE"
    bpy.ops.wm.save_as_mainfile(filepath=eevee_blend_path)
    return results, eevee_results

bpy.ops.wm.read_factory_settings(use_empty=True)
geometry_results = build_geometry_reference()
if shader_blend_path == "-":
    shader_results, eevee_results = {}, {}
else:
    shader_results, eevee_results = shader_reference()
print("POTTER_NODE_MATH=" + json.dumps(
    {"geometry": geometry_results, "shader": shader_results, "eevee": eevee_results},
    sort_keys=True,
    allow_nan=False,
))
