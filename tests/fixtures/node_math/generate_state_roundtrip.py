"""Generate and inspect Blender 5.2.2 node-state fixtures for the .blend bridge."""
import bpy
import json
import sys

if bpy.app.version != (5, 2, 2):
    raise RuntimeError("node state round-trip requires Blender 5.2.2, found " + bpy.app.version_string)


def curve_state(node):
    mapping = node.mapping
    state = {
        "extend": mapping.extend,
        "use_clip": bool(mapping.use_clip),
        "clip": [float(mapping.clip_min_x), float(mapping.clip_min_y),
                 float(mapping.clip_max_x), float(mapping.clip_max_y)],
        "black": [float(value) for value in mapping.black_level],
        "white": [float(value) for value in mapping.white_level],
        "curves": [
            [{"location": [float(point.location[0]), float(point.location[1])],
              "handle": point.handle_type} for point in curve.points]
            for curve in mapping.curves
        ],
    }
    if hasattr(mapping, "tone"):
        state["tone"] = mapping.tone
    return state


def set_curve_mapping(node, use_clip, clip, black, white, curves):
    mapping = node.mapping
    if hasattr(mapping, "tone"):
        mapping.tone = "FILMLIKE"
    mapping.extend = "HORIZONTAL"
    mapping.use_clip = use_clip
    mapping.clip_min_x, mapping.clip_min_y = clip[0], clip[1]
    mapping.clip_max_x, mapping.clip_max_y = clip[2], clip[3]
    mapping.black_level = black
    mapping.white_level = white
    for target, points in zip(mapping.curves, curves):
        for point in list(target.points)[1:-1]:
            target.points.remove(point)
        first, last = target.points[0], target.points[-1]
        first.location = points[0][:2]
        first.handle_type = points[0][2]
        for x, y, handle in points[1:-1]:
            middle = target.points.new(x, y)
            middle.handle_type = handle
        last.location = points[-1][:2]
        last.handle_type = points[-1][2]
    mapping.update()


def ramp_state(node):
    ramp = node.color_ramp
    return {
        "color_mode": ramp.color_mode,
        "hue_interpolation": ramp.hue_interpolation,
        "interpolation": ramp.interpolation,
        "elements": [
            {"position": float(element.position),
             "color": [float(value) for value in element.color]}
            for element in ramp.elements
        ],
    }


def read_state():
    material = bpy.data.materials.get("NodeMathState")
    if material is None or not material.use_nodes:
        raise RuntimeError("NodeMathState material is missing")
    nodes = material.node_tree.nodes
    curves = {
        node.bl_idname: node
        for node in nodes
        if node.bl_idname in {
            "ShaderNodeFloatCurve", "ShaderNodeVectorCurve", "ShaderNodeRGBCurve"
        }
    }
    ramp = next((node for node in nodes if node.bl_idname == "ShaderNodeValToRGB"), None)
    if len(curves) != 3 or ramp is None:
        raise RuntimeError("Float, Vector, and RGB Curve or ColorRamp node is missing")
    return {
        "float_curve": curve_state(curves["ShaderNodeFloatCurve"]),
        "vector_curve": curve_state(curves["ShaderNodeVectorCurve"]),
        "rgb_curve": curve_state(curves["ShaderNodeRGBCurve"]),
        "ramp": ramp_state(ramp),
    }


def generate(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    material = bpy.data.materials.new("NodeMathState")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()

    float_curve = tree.nodes.new("ShaderNodeFloatCurve")
    float_curve.name = "OracleFloatCurve"
    set_curve_mapping(
        float_curve, False, (0.0625, 0.125, 0.9375, 0.875),
        (0.125, 0.0625, 0.25), (0.875, 0.9375, 0.75),
        [[(0.0625, 0.125, "VECTOR"), (0.5, 0.375, "AUTO_CLAMPED"),
          (0.9375, 0.875, "VECTOR")]],
    )

    vector_curve = tree.nodes.new("ShaderNodeVectorCurve")
    vector_curve.name = "OracleVectorCurve"
    set_curve_mapping(
        vector_curve, False, (0.0625, 0.0625, 0.9375, 0.9375),
        (0.0625, 0.1875, 0.125), (0.9375, 0.8125, 0.875),
        [
            [(0.0625, 0.125, "VECTOR"), (0.5, 0.75, "AUTO_CLAMPED"),
             (0.9375, 0.875, "VECTOR")],
            [(0.125, 0.1875, "AUTO_CLAMPED"), (0.5625, 0.4375, "VECTOR"),
             (0.875, 0.8125, "AUTO_CLAMPED")],
            [(0.1875, 0.0625, "VECTOR"), (0.4375, 0.625, "AUTO_CLAMPED"),
             (0.8125, 0.9375, "VECTOR")],
        ],
    )

    curve = tree.nodes.new("ShaderNodeRGBCurve")
    curve.name = "OracleCurve"
    set_curve_mapping(
        curve, True, (0.125, 0.0625, 0.875, 0.9375),
        (0.0625, 0.125, 0.1875), (0.8125, 0.875, 0.9375),
        [
            [(0.125, 0.125, "VECTOR"), (0.375, 0.6875, "AUTO_CLAMPED"),
             (0.875, 0.875, "VECTOR")],
            [(0.125, 0.1875, "AUTO_CLAMPED"), (0.5, 0.3125, "VECTOR"),
             (0.875, 0.8125, "AUTO_CLAMPED")],
            [(0.125, 0.0625, "VECTOR"), (0.5625, 0.75, "AUTO_CLAMPED"),
             (0.875, 0.9375, "VECTOR")],
            [(0.125, 0.125, "VECTOR"), (0.375, 0.6875, "AUTO_CLAMPED"),
             (0.875, 0.875, "VECTOR")],
        ],
    )

    ramp_node = tree.nodes.new("ShaderNodeValToRGB")
    ramp_node.name = "OracleRamp"
    ramp = ramp_node.color_ramp
    ramp.color_mode = "HSL"
    ramp.hue_interpolation = "CW"
    ramp.interpolation = "EASE"
    first, last = ramp.elements[0], ramp.elements[-1]
    first.position = 0.125
    first.color = (0.125, 0.25, 0.5, 1.0)
    extra = ramp.elements.new(0.4375)
    extra.color = (0.75, 0.125, 0.375, 0.8)
    last.position = 0.875
    last.color = (0.25, 0.875, 0.625, 0.6)

    principled = tree.nodes.new("ShaderNodeBsdfPrincipled")
    principled.inputs["Emission Strength"].default_value = 1.0
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(float_curve.outputs["Value"], principled.inputs["Roughness"])
    tree.links.new(vector_curve.outputs["Vector"], principled.inputs["Normal"])
    tree.links.new(curve.outputs["Color"], principled.inputs["Emission Color"])
    tree.links.new(ramp_node.outputs["Color"], principled.inputs["Base Color"])
    tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    mesh = bpy.data.meshes.new("NodeMathStateMesh")
    mesh.from_pydata(
        [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)],
        [],
        [(0, 1, 2, 3)],
    )
    obj = bpy.data.objects.new("NodeMathStateObject", mesh)
    scene.collection.objects.link(obj)
    mesh.materials.append(material)
    bpy.ops.wm.save_as_mainfile(filepath=path)
    bpy.ops.wm.open_mainfile(filepath=path)
    print("POTTER_NODE_STATE=" + json.dumps(read_state(), sort_keys=True))


arguments = sys.argv[sys.argv.index("--") + 1:]
if arguments[0] == "generate":
    generate(arguments[1])
elif arguments[0] == "read":
    bpy.ops.wm.open_mainfile(filepath=arguments[1])
    print("POTTER_NODE_STATE=" + json.dumps(read_state(), sort_keys=True))
else:
    raise RuntimeError("expected generate or read mode")
