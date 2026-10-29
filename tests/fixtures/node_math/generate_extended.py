"""Blender 5.2.2 oracles for the remaining math, compare, mix, ramp, and curve nodes."""
import bpy
from mathutils import Quaternion
import json
import math
import re
import sys
import os
import tempfile

if bpy.app.version != (5, 2, 2):
    raise RuntimeError("extended node math oracle requires Blender 5.2.2, found " + bpy.app.version_string)

arguments = sys.argv[sys.argv.index("--") + 1:]
blend_path = arguments[0]
capture_path = arguments[1] if len(arguments) > 1 else None
F32_MAX = 3.4028234663852886e38


def blender_input(value):
    if isinstance(value, list):
        return [blender_input(component) for component in value]
    if isinstance(value, float) and abs(value) > F32_MAX:
        return math.copysign(float("inf"), value)
    return value


cases = []


def add_case(label, node_type, inputs, properties, output, attribute_type,
             shader_probe=False, render_kind="color", realtime_probe=False):
    identifier = re.sub(r"[^a-z0-9_]+", "_", label.lower()).strip("_")
    case = {
        "id": identifier,
        "label": label,
        "node_type": node_type,
        "inputs": inputs,
        "properties": properties,
        "output": output,
        "attribute_type": attribute_type,
        "shader_probe": shader_probe,
        "render_kind": render_kind,
        "realtime_probe": realtime_probe,
    }
    cases.append(case)
    return case


def compare_case(data_type, operation, mode="ELEMENT", first=None, second=None,
                 epsilon=None, parameter=None, suffix=None):
    inputs = {"A": first, "B": second}
    if epsilon is not None:
        inputs["Epsilon"] = epsilon
    if parameter is not None:
        inputs["C" if mode == "DOT_PRODUCT" else "Angle"] = parameter
    label = "compare_%s_%s_%s" % (data_type.lower(), mode.lower(), operation.lower())
    if suffix:
        label += "_" + suffix
    add_case(
        label,
        "FunctionNodeCompare",
        inputs,
        {"data_type": data_type, "operation": operation, "mode": mode},
        "Result",
        "BOOLEAN",
    )


def scalar_compare_pair(operation):
    return {
        "LESS_THAN": (0.25, 0.75, None),
        "LESS_EQUAL": (0.75, 0.75, None),
        "GREATER_THAN": (0.8, 0.2, None),
        "GREATER_EQUAL": (0.75, 0.75, None),
        "EQUAL": (0.3, 0.34, 0.05),
        "NOT_EQUAL": (0.3, 0.42, 0.05),
    }[operation]


def vector_pair(mode, operation):
    if mode == "ELEMENT":
        a, b = ([0.2, 0.4, 0.6], [0.7, 0.8, 0.9])
        if operation in ("GREATER_THAN", "GREATER_EQUAL"):
            a, b = b, a
        elif operation in ("EQUAL", "NOT_EQUAL"):
            delta = 0.02 if operation == "EQUAL" else 0.2
            a, b = [0.2, 0.4, 0.6], [0.2 + delta, 0.4 + delta, 0.6 + delta]
        elif operation == "LESS_EQUAL":
            a, b = [0.2, 0.4, 0.6], [0.2, 0.4, 0.6]
        return a, b, None, 0.05 if operation in ("EQUAL", "NOT_EQUAL") else None
    if mode == "LENGTH":
        a, b = ([1.0, 2.0, 2.0], [2.0, 3.0, 6.0])
        if operation in ("GREATER_THAN", "GREATER_EQUAL"):
            a, b = b, a
        elif operation in ("EQUAL", "NOT_EQUAL"):
            a, b = ([3.0, 0.0, 0.0], [3.02, 0.0, 0.0] if operation == "EQUAL" else [3.2, 0.0, 0.0])
        elif operation == "LESS_EQUAL":
            a, b = [3.0, 0.0, 0.0], [3.0, 0.0, 0.0]
        return a, b, None, 0.05 if operation in ("EQUAL", "NOT_EQUAL") else None
    if mode == "AVERAGE":
        a, b = ([0.1, 0.2, 0.3], [0.7, 0.8, 0.9])
        if operation in ("GREATER_THAN", "GREATER_EQUAL"):
            a, b = b, a
        elif operation in ("EQUAL", "NOT_EQUAL"):
            delta = 0.02 if operation == "EQUAL" else 0.2
            a, b = [0.2, 0.4, 0.6], [0.2 + delta, 0.4 + delta, 0.6 + delta]
        elif operation == "LESS_EQUAL":
            a, b = [0.2, 0.4, 0.6], [0.2, 0.4, 0.6]
        return a, b, None, 0.05 if operation in ("EQUAL", "NOT_EQUAL") else None
    if mode == "DOT_PRODUCT":
        a, b = [1.0, 0.0, 0.0], [0.6, 0.8, 0.0]
        threshold = 0.25
        if operation in ("GREATER_THAN", "GREATER_EQUAL"):
            threshold = 0.4
        elif operation == "EQUAL":
            threshold = 0.6
        elif operation == "NOT_EQUAL":
            threshold = 0.3
        elif operation == "LESS_EQUAL":
            threshold = 0.6
        return a, b, threshold, 0.05 if operation in ("EQUAL", "NOT_EQUAL") else None
    if mode == "DIRECTION":
        a, b = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
        threshold = 1.0
        if operation in ("GREATER_THAN", "GREATER_EQUAL"):
            threshold = 2.0
        elif operation == "EQUAL":
            threshold = math.pi / 2.0
        elif operation == "NOT_EQUAL":
            threshold = 0.2
        elif operation == "LESS_EQUAL":
            threshold = math.pi / 2.0
        return a, b, threshold, 0.05 if operation in ("EQUAL", "NOT_EQUAL") else None
    raise RuntimeError("unknown vector comparison mode " + mode)


def new_material_node(name, node_type):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    return material.node_tree.nodes.new(node_type)


def generate_cases():
    int_node = bpy.data.node_groups.new("integer_catalog", "GeometryNodeTree").nodes.new("FunctionNodeIntegerMath")
    integer_operations = sorted(item.identifier for item in int_node.bl_rna.properties["operation"].enum_items)
    for operation in integer_operations:
        a, b, c = -17, 5, 4
        if operation == "POWER":
            a, b = 3, 4
        elif operation in ("GCD", "LCM"):
            a, b = 84, 30
        elif operation == "MULTIPLY_ADD":
            a, b, c = 7, -3, 5
        inputs = {"Value": a, "Value_001": b}
        if operation == "MULTIPLY_ADD":
            inputs["Value_002"] = c
        add_case("integer_" + operation.lower(), "FunctionNodeIntegerMath", inputs,
                 {"operation": operation}, "Value", "INT")

    bool_node = bpy.data.node_groups.new("boolean_catalog", "GeometryNodeTree").nodes.new("FunctionNodeBooleanMath")
    boolean_operations = sorted(item.identifier for item in bool_node.bl_rna.properties["operation"].enum_items)
    for operation in boolean_operations:
        inputs = {"Boolean": True}
        if operation != "NOT":
            inputs["Boolean_001"] = False
        add_case("boolean_" + operation.lower(), "FunctionNodeBooleanMath", inputs,
                 {"operation": operation}, "Boolean", "BOOLEAN")

    compare_ops = ("LESS_THAN", "LESS_EQUAL", "GREATER_THAN", "GREATER_EQUAL", "EQUAL", "NOT_EQUAL")
    for operation in compare_ops:
        first, second, epsilon = scalar_compare_pair(operation)
        compare_case("FLOAT", operation, first=first, second=second, epsilon=epsilon)
        int_pair = {
            "LESS_THAN": (2, 5), "LESS_EQUAL": (5, 5),
            "GREATER_THAN": (8, 3), "GREATER_EQUAL": (5, 5),
            "EQUAL": (4, 4), "NOT_EQUAL": (4, 7),
        }[operation]
        compare_case("INT", operation, first=int_pair[0], second=int_pair[1])
    for operation in ("EQUAL", "NOT_EQUAL"):
        compare_case("STRING", operation, first="copper", second="sapphire")
        compare_case("STRING", operation, first="copper", second="copper", suffix="same")
    for operation in ("BRIGHTER", "DARKER"):
        compare_case("RGBA", operation, first=[0.78, 0.3, 0.2, 0.2],
                     second=[0.12, 0.35, 0.6, 0.85])
    for operation in ("EQUAL", "NOT_EQUAL"):
        compare_case("RGBA", operation, first=[0.25, 0.4, 0.6, 0.1],
                     second=[0.27, 0.41, 0.61, 0.9], epsilon=0.04)

    vector_modes = ("ELEMENT", "LENGTH", "AVERAGE", "DOT_PRODUCT", "DIRECTION")
    for mode in vector_modes:
        for operation in compare_ops:
            first, second, parameter, epsilon = vector_pair(mode, operation)
            compare_case("VECTOR", operation, mode, first, second, epsilon, parameter)
    compare_case("FLOAT", "EQUAL", first=0.0, second=1.0e-8, epsilon=0.0,
                 suffix="zero_epsilon_small_difference")
    compare_case("FLOAT", "EQUAL", first=F32_MAX, second=F32_MAX, epsilon=0.0,
                 suffix="equal_max_float")

    id_types = ("OBJECT", "IMAGE", "COLLECTION", "MATERIAL", "FONT", "SOUND")
    for data_type in id_types:
        a = {"id_type": data_type, "name": data_type.lower() + "_compare_a"}
        b = {"id_type": data_type, "name": data_type.lower() + "_compare_b"}
        for operation in ("EQUAL", "NOT_EQUAL"):
            compare_case(data_type, operation, first=a, second=b)
        compare_case(data_type, "EQUAL", first=a, second=a, suffix="same")
        compare_case(data_type, "NOT_EQUAL", first=a, second=a, suffix="same")

    map_node = new_material_node("map_catalog", "ShaderNodeMapRange")
    map_interpolations = sorted(item.identifier for item in map_node.bl_rna.properties["interpolation_type"].enum_items)
    for data_type in ("FLOAT", "FLOAT_VECTOR"):
        for interpolation in map_interpolations:
            for clamp in (False, True):
                if data_type == "FLOAT":
                    inputs = {"Value": 1.3, "From Min": 0.0, "From Max": 1.0,
                              "To Min": 0.25, "To Max": 1.6, "Steps": 3.0}
                    output, attribute_type, render_kind = "Result", "FLOAT", "scalar"
                else:
                    inputs = {"Vector": [1.3, 0.68, 0.41],
                              "From_Min_FLOAT3": [0.0, 0.0, 0.0],
                              "From_Max_FLOAT3": [1.0, 1.0, 1.0],
                              "To_Min_FLOAT3": [0.25, 0.4, 0.6],
                              "To_Max_FLOAT3": [1.6, 1.8, 2.0],
                              "Steps_FLOAT3": [3.0, 4.0, 2.0]}
                    output, attribute_type, render_kind = "Vector", "FLOAT_VECTOR", "color"
                add_case("map_%s_%s_clamp_%s" % (data_type.lower(), interpolation.lower(), str(clamp).lower()),
                         "ShaderNodeMapRange", inputs,
                         {"data_type": data_type, "interpolation_type": interpolation, "clamp": clamp},
                         output, attribute_type, True, render_kind)
    add_case(
        "map_float_linear_zero_span_distinct_to_min",
        "ShaderNodeMapRange",
        {"Value": 0.5, "From Min": 0.5, "From Max": 0.5,
         "To Min": 0.375, "To Max": 1.625, "Steps": 3.0},
        {"data_type": "FLOAT", "interpolation_type": "LINEAR", "clamp": False},
        "Result", "FLOAT", True, "scalar", True)
    add_case(
        "map_float_linear_zero_span_clamped",
        "ShaderNodeMapRange",
        {"Value": 0.5, "From Min": 0.5, "From Max": 0.5,
         "To Min": 0.375, "To Max": 1.625, "Steps": 3.0},
        {"data_type": "FLOAT", "interpolation_type": "LINEAR", "clamp": True},
        "Result", "FLOAT", True, "scalar", True)
    add_case(
        "map_float_smoothstep_zero_span_clamped",
        "ShaderNodeMapRange",
        {"Value": 0.5, "From Min": 0.5, "From Max": 0.5,
         "To Min": 0.375, "To Max": 1.625, "Steps": 3.0},
        {"data_type": "FLOAT", "interpolation_type": "SMOOTHSTEP", "clamp": True},
        "Result", "FLOAT", True, "scalar", True)
    add_case(
        "map_vector_stepped_nonpositive_steps",
        "ShaderNodeMapRange",
        {"Vector": [0.6, 0.6, 0.6],
         "From_Min_FLOAT3": [0.0, 0.0, 0.0],
         "From_Max_FLOAT3": [1.0, 1.0, 1.0],
         "To_Min_FLOAT3": [0.25, 0.5, 0.75],
         "To_Max_FLOAT3": [1.25, 1.5, 1.75],
         "Steps_FLOAT3": [-2.0, 0.0, -3.0]},
        {"data_type": "FLOAT_VECTOR", "interpolation_type": "STEPPED", "clamp": False},
        "Vector", "FLOAT_VECTOR", True, "color", True)
    add_case(
        "map_float_linear_default_clamp",
        "ShaderNodeMapRange",
        {"Value": 2.0, "From Min": 0.0, "From Max": 1.0,
         "To Min": 0.0, "To Max": 1.0, "Steps": 3.0},
        {"data_type": "FLOAT", "interpolation_type": "LINEAR"},
        "Result", "FLOAT", True, "scalar", True)
    add_case(
        "map_float_stepped_zero_steps",
        "ShaderNodeMapRange",
        {"Value": 0.5, "From Min": 0.0, "From Max": 1.0,
         "To Min": 0.375, "To Max": 1.625, "Steps": 0.0},
        {"data_type": "FLOAT", "interpolation_type": "STEPPED", "clamp": False},
        "Result", "FLOAT", True, "scalar", True)

    clamp_node = new_material_node("clamp_catalog", "ShaderNodeClamp")
    clamp_modes = sorted(item.identifier for item in clamp_node.bl_rna.properties["clamp_type"].enum_items)
    for mode in clamp_modes:
        add_case("clamp_" + mode.lower(), "ShaderNodeClamp",
                 {"Value": 0.5, "Min": 0.8, "Max": 0.2},
                 {"clamp_type": mode}, "Result", "FLOAT", True, "scalar")

    mix_node = new_material_node("mix_catalog", "ShaderNodeMix")
    blend_modes = sorted(item.identifier for item in mix_node.bl_rna.properties["blend_type"].enum_items)
    factor_modes = sorted(item.identifier for item in mix_node.bl_rna.properties["factor_mode"].enum_items)
    clamp_flags = (False, True)
    rotation_axis = [component / math.sqrt(14.0) for component in (1.0, 2.0, 3.0)]
    rotation_sine = math.sin(math.radians(85.0))
    rotation_cosine = math.cos(math.radians(85.0))
    rotation_a = [rotation_sine * component for component in rotation_axis] + [rotation_cosine]
    rotation_b = [-rotation_sine * component for component in rotation_axis] + [rotation_cosine]
    for factor in (-0.25, 1.25):
        for clamp_factor in clamp_flags:
            add_case("mix_rotation_factor_%s_clamp_%s" %
                     ("negative" if factor < 0.0 else "above_one", str(clamp_factor).lower()),
                     "ShaderNodeMix",
                     {"Factor_Float": factor, "A_Rotation": rotation_a, "B_Rotation": rotation_b},
                     {"data_type": "ROTATION", "factor_mode": "UNIFORM", "blend_type": "MIX",
                      "clamp_factor": clamp_factor, "clamp_result": False},
                     "Result_Rotation", "QUATERNION")
    for clamp_factor in clamp_flags:
        for clamp_result in clamp_flags:
            add_case("mix_float_factor_%s_result_%s" % (str(clamp_factor).lower(), str(clamp_result).lower()),
                     "ShaderNodeMix",
                     {"Factor_Float": 1.4, "A_Float": 0.25, "B_Float": 0.75},
                     {"data_type": "FLOAT", "factor_mode": "UNIFORM", "blend_type": "MIX",
                      "clamp_factor": clamp_factor, "clamp_result": clamp_result},
                     "Result_Float", "FLOAT", True, "scalar")
    for factor_mode in factor_modes:
        if factor_mode == "UNIFORM":
            factor_socket, factor = "Factor_Float", 1.4
        else:
            factor_socket, factor = "Factor_Vector", [1.4, -0.3, 0.65]
        for clamp_factor in clamp_flags:
            for clamp_result in clamp_flags:
                add_case("mix_vector_%s_factor_%s_result_%s" %
                         (factor_mode.lower(), str(clamp_factor).lower(), str(clamp_result).lower()),
                         "ShaderNodeMix",
                         {factor_socket: factor, "A_Vector": [0.15, 0.35, 0.55],
                          "B_Vector": [0.8, 0.7, 0.95]},
                         {"data_type": "VECTOR", "factor_mode": factor_mode, "blend_type": "MIX",
                          "clamp_factor": clamp_factor, "clamp_result": clamp_result},
                         "Result_Vector", "FLOAT_VECTOR", True, "color")
    for blend_mode in blend_modes:
        for clamp_factor in clamp_flags:
            for clamp_result in clamp_flags:
                add_case("mix_rgba_%s_factor_%s_result_%s" %
                         (blend_mode.lower(), str(clamp_factor).lower(), str(clamp_result).lower()),
                         "ShaderNodeMix",
                         {"Factor_Float": 1.4, "A_Color": [0.8, 0.2, 0.4, 0.3],
                          "B_Color": [0.7, 0.9, 0.1, 0.85]},
                         {"data_type": "RGBA", "factor_mode": "UNIFORM", "blend_type": blend_mode,
                          "clamp_factor": clamp_factor, "clamp_result": clamp_result},
                         "Result_Color", "FLOAT_COLOR", True, "color")

    legacy_node = new_material_node("mixrgb_catalog", "ShaderNodeMixRGB")
    legacy_modes = sorted(item.identifier for item in legacy_node.bl_rna.properties["blend_type"].enum_items)
    for blend_mode in legacy_modes:
        add_case("mixrgb_" + blend_mode.lower(), "ShaderNodeMixRGB",
                 {"Fac": 0.63, "Color1": [0.82, 0.23, 0.46, 0.3],
                  "Color2": [0.19, 0.74, 0.31, 0.85]},
                 {"blend_type": blend_mode}, "Color", "FLOAT_COLOR", True, "color")

    ramp_node = new_material_node("ramp_catalog", "ShaderNodeValToRGB")
    ramp_interpolations = sorted(item.identifier for item in ramp_node.color_ramp.bl_rna.properties["interpolation"].enum_items)
    ramp_color_modes = sorted(item.identifier for item in ramp_node.color_ramp.bl_rna.properties["color_mode"].enum_items)
    ramp_hue_modes = sorted(item.identifier for item in ramp_node.color_ramp.bl_rna.properties["hue_interpolation"].enum_items)
    ramp_elements = [
        {"position": 0.0, "color": [0.95, 0.15, 0.25, 0.2]},
        {"position": 0.38, "color": [0.15, 0.85, 0.3, 0.65]},
        {"position": 1.0, "color": [0.1, 0.25, 0.95, 0.9]},
    ]
    for interpolation in ramp_interpolations:
        for color_mode in ramp_color_modes:
            for hue_mode in ramp_hue_modes:
                ramp = {"interpolation": interpolation, "color_mode": color_mode,
                        "hue_interpolation": hue_mode, "elements": ramp_elements}
                label = "ramp_%s_%s_%s" % (interpolation.lower(), color_mode.lower(), hue_mode.lower())
                add_case(label + "_color", "ShaderNodeValToRGB", {"Fac": 0.67},
                         {"color_ramp": ramp}, "Color", "FLOAT_COLOR", True, "color")
                add_case(label + "_alpha", "ShaderNodeValToRGB", {"Fac": 0.67},
                         {"color_ramp": ramp}, "Alpha", "FLOAT", True, "scalar")

    constant_ramp = {
        "interpolation": "CONSTANT",
        "color_mode": "RGB",
        "hue_interpolation": "NEAR",
        "elements": [
            {"position": 0.0, "color": [0.9, 0.1, 0.2, 0.3]},
            {"position": 0.3, "color": [0.1, 0.8, 0.4, 0.7]},
        ],
    }
    add_case("ramp_constant_factor_after_stop", "ShaderNodeValToRGB", {"Fac": 0.3005},
             {"color_ramp": constant_ramp}, "Color", "FLOAT_COLOR", True, "color", True)
    off_table_ramp = {
        "interpolation": "LINEAR",
        "color_mode": "RGB",
        "hue_interpolation": "NEAR",
        "elements": [
            {"position": 0.2, "color": [0.9, 0.1, 0.2, 0.3]},
            {"position": 0.8, "color": [0.1, 0.8, 0.4, 0.7]},
        ],
    }
    for label, factor in (("near_start", 0.201), ("near_end", 0.801)):
        add_case("ramp_linear_two_stop_" + label, "ShaderNodeValToRGB", {"Fac": factor},
                 {"color_ramp": off_table_ramp}, "Color", "FLOAT_COLOR", True, "color", True)
    texture_ramp = {
        "interpolation": "LINEAR",
        "color_mode": "RGB",
        "hue_interpolation": "NEAR",
        "elements": [
            {"position": 0.2, "color": [0.9, 0.1, 0.2, 0.3]},
            {"position": 0.5, "color": [0.15, 0.25, 0.95, 0.55]},
            {"position": 0.8, "color": [0.1, 0.8, 0.4, 0.7]},
        ],
    }
    for label, factor in (("near_start", 0.2025), ("near_end", 0.7975)):
        add_case("ramp_linear_texture_" + label, "ShaderNodeValToRGB", {"Fac": factor},
                 {"color_ramp": texture_ramp}, "Color", "FLOAT_COLOR", True, "color", True)
    constant_table_ramp = {
        "interpolation": "CONSTANT",
        "color_mode": "RGB",
        "hue_interpolation": "NEAR",
        "elements": [
            {"position": 0.0, "color": [0.0, 0.0, 0.0, 1.0]},
            {"position": 0.501, "color": [1.0, 0.0, 0.0, 1.0]},
            {"position": 1.0, "color": [0.0, 0.0, 1.0, 1.0]},
        ],
    }
    add_case("ramp_constant_table_constant_factor", "ShaderNodeValToRGB", {"Fac": 0.5011},
             {"color_ramp": constant_table_ramp}, "Color", "FLOAT_COLOR", True, "color", True)
    curve_definitions = {
        "ShaderNodeFloatCurve": ("FLOAT", "Value", "Value", "FLOAT", "scalar"),
        "ShaderNodeVectorCurve": ("VECTOR", "Vector", "Vector", "FLOAT_VECTOR", "color"),
        "ShaderNodeRGBCurve": ("RGBA", "Color", "Color", "FLOAT_COLOR", "color"),
    }
    channel_points = [
        [[0.0, 0.08], [0.5, 0.78], [1.0, 0.92]],
        [[0.0, 0.12], [0.5, 0.42], [1.0, 0.86]],
        [[0.0, 0.05], [0.5, 0.7], [1.0, 0.75]],
        [[0.0, 0.02], [0.5, 0.63], [1.0, 0.9]],
    ]
    for node_type, (kind, input_socket, output, attribute_type, render_kind) in curve_definitions.items():
        for extend in (False, True):
            for use_clip in (False, True):
                mapping = {
                    "tone": "film_like" if kind == "RGBA" and extend else "standard",
                    "extend_extrapolate": extend,
                    "use_clip": use_clip,
                    "clip": [0.1, 0.12, 0.9, 0.88],
                    "black": [0.05, 0.08, 0.12],
                    "white": [0.9, 0.88, 0.95],
                    "curves": [
                        {"points": [
                            {"location": point, "handle": "vector"}
                            for point in points
                        ]}
                        for points in channel_points
                    ],
                }
                if kind == "FLOAT":
                    value = 1.15
                    inputs = {input_socket: value}
                elif kind == "VECTOR":
                    value = [0.22, 0.53, 1.15]
                    inputs = {"Fac": 0.73, input_socket: value}
                else:
                    value = [0.22, 0.53, 1.15, 0.47]
                    inputs = {"Fac": 0.73, input_socket: value}
                add_case("curve_%s_extend_%s_clip_%s" %
                         (kind.lower(), str(extend).lower(), str(use_clip).lower()),
                         node_type, inputs, {"mapping": mapping}, output,
                         attribute_type, True, render_kind,
                         kind == "FLOAT" and extend and use_clip)
    float_clip_mapping = {
        "tone": "standard",
        "extend_extrapolate": True,
        "use_clip": True,
        "clip": [0.1, 0.12, 0.9, 0.35],
        "black": [0.05, 0.08, 0.12],
        "white": [0.9, 0.88, 0.95],
        "curves": [{
            "points": [
                {"location": [0.0, 0.08], "handle": "vector"},
                {"location": [0.5, 0.78], "handle": "vector"},
                {"location": [1.0, 0.92], "handle": "vector"},
            ],
        }],
    }
    add_case("curve_float_clip_constant_input", "ShaderNodeFloatCurve",
             {"Value": 1.0}, {"mapping": float_clip_mapping},
             "Value", "FLOAT", True, "scalar", True)
    add_case("curve_float_input_beyond_clip_gn_control", "ShaderNodeFloatCurve",
             {"Value": 1.15}, {"mapping": float_clip_mapping},
             "Value", "FLOAT", True, "scalar")

    random_node = bpy.data.node_groups.new("random_catalog", "GeometryNodeTree").nodes.new("FunctionNodeRandomValue")
    random_types = {item.identifier for item in random_node.bl_rna.properties["data_type"].enum_items}
    if "BOOLEAN" not in random_types:
        raise RuntimeError("Blender 5.2.2 Random Value has no BOOLEAN data type")
    for probability in (0.0, 1.0):
        add_case("random_boolean_probability_%s" % str(probability).replace(".", "_"),
                 "FunctionNodeRandomValue", {"Probability": probability, "Seed": 29},
                 {"data_type": "BOOLEAN"}, "Value", "BOOLEAN")


generate_cases()

for case in cases:
    if not case["shader_probe"]:
        continue
    if case["node_type"] == "ShaderNodeValToRGB":
        case["shader_varying_input"] = "Fac"
    elif case["node_type"] == "ShaderNodeFloatCurve":
        case["shader_varying_input"] = "Value"
    elif case["node_type"] == "ShaderNodeMapRange" and case["id"] in (
        "map_float_linear_zero_span_distinct_to_min",
        "map_float_linear_zero_span_clamped",
        "map_float_smoothstep_zero_span_clamped",
        "map_float_linear_default_clamp",
    ):
        case["shader_varying_input"] = "Value"
    elif case["node_type"] == "ShaderNodeMapRange" and case["id"] == (
        "map_vector_stepped_nonpositive_steps"
    ):
        case["shader_varying_input"] = "Vector"

def ids_for_compare():
    ids = {}
    font_path = os.path.join(bpy.utils.resource_path("LOCAL"), "datafiles", "fonts", "Inter.woff2")
    sound_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare.wav")
    for data_type in ("OBJECT", "IMAGE", "COLLECTION", "MATERIAL", "FONT", "SOUND"):
        for suffix in ("a", "b"):
            name = data_type.lower() + "_compare_" + suffix
            if data_type == "OBJECT":
                block = bpy.data.objects.new(name, None)
                bpy.context.scene.collection.objects.link(block)
            elif data_type == "IMAGE":
                block = bpy.data.images.new(name, 1, 1)
            elif data_type == "COLLECTION":
                block = bpy.data.collections.new(name)
            elif data_type == "MATERIAL":
                block = bpy.data.materials.new(name)
            elif data_type == "FONT":
                block = bpy.data.fonts.load(font_path, check_existing=False)
            else:
                block = bpy.data.sounds.load(sound_path, check_existing=False)
            block.name = name
            ids[(data_type, name)] = block
    for data_type in ("OBJECT", "IMAGE", "COLLECTION", "MATERIAL", "FONT", "SOUND"):
        first = ids[(data_type, data_type.lower() + "_compare_a")]
        second = ids[(data_type, data_type.lower() + "_compare_b")]
        if first.as_pointer() == second.as_pointer():
            raise RuntimeError("compare oracle requires distinct " + data_type + " IDs")
    return ids


def resolve_input(value, ids):
    if isinstance(value, dict) and "id_type" in value:
        return ids[(value["id_type"], value["name"])]
    return value


def socket_by_identifier(node, identifier):
    for socket in node.inputs:
        if socket.identifier == identifier:
            return socket
    return None
def output_socket_by_identifier(node, identifier):
    for socket in node.outputs:
        if socket.identifier == identifier:
            return socket
    return None




def set_curve_mapping(node, state):
    mapping = node.mapping
    if hasattr(mapping, "tone"):
        mapping.tone = "FILMLIKE" if state["tone"] == "film_like" else "STANDARD"
    mapping.extend = "EXTRAPOLATED" if state["extend_extrapolate"] else "HORIZONTAL"
    mapping.use_clip = state["use_clip"]
    mapping.clip_min_x, mapping.clip_min_y, mapping.clip_max_x, mapping.clip_max_y = state["clip"]
    mapping.black_level = state["black"]
    mapping.white_level = state["white"]
    handles = {"auto": "AUTO", "auto_clamped": "AUTO_CLAMPED", "vector": "VECTOR"}
    for target, source in zip(mapping.curves, state["curves"]):
        while len(target.points) > 2:
            target.points.remove(target.points[1])
        source_points = source["points"]
        target.points[0].location = source_points[0]["location"]
        target.points[-1].location = source_points[-1]["location"]
        for point in source_points[1:-1]:
            target.points.new(*point["location"])
        for point, source_point in zip(target.points, source_points):
            point.location = source_point["location"]
            point.handle_type = handles[source_point["handle"]]
    mapping.update()

def set_color_ramp(node, state):
    ramp = node.color_ramp
    ramp.interpolation = state["interpolation"]
    ramp.color_mode = state["color_mode"]
    ramp.hue_interpolation = state["hue_interpolation"]
    while len(ramp.elements) > 2:
        ramp.elements.remove(ramp.elements[-1])
    elements = state["elements"]
    ramp.elements[0].position = elements[0]["position"]
    ramp.elements[0].color = elements[0]["color"]
    ramp.elements[1].position = elements[-1]["position"]
    ramp.elements[1].color = elements[-1]["color"]
    for element in elements[1:-1]:
        created = ramp.elements.new(element["position"])
        created.color = element["color"]


def set_node(case, node, ids):
    for key, value in case["properties"].items():
        if key == "mapping":
            set_curve_mapping(node, value)
        elif key == "color_ramp":
            set_color_ramp(node, value)
        else:
            setattr(node, key, value)
    for identifier, value in case["inputs"].items():
        socket = socket_by_identifier(node, identifier)
        if socket is None:
            continue
        if case["properties"].get("data_type") == "ROTATION" and identifier in ("A_Rotation", "B_Rotation"):
            quaternion = Quaternion((value[3], value[0], value[1], value[2]))
            socket.default_value = quaternion.to_euler()
        else:
            socket.default_value = blender_input(resolve_input(value, ids))


def new_geometry_oracle():
    tree = bpy.data.node_groups.new("Potter Extended Node Math Oracle", "GeometryNodeTree")
    tree.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes, links = tree.nodes, tree.links
    line = nodes.new("GeometryNodeMeshLine")
    line.inputs["Count"].default_value = 1
    geometry = line.outputs["Mesh"]
    for case in cases:
        operation = nodes.new(case["node_type"])
        set_node(case, operation, GENERATED_IDS)
        store = nodes.new("GeometryNodeStoreNamedAttribute")
        store.domain = "POINT"
        store.data_type = case["attribute_type"]
        store.inputs["Name"].default_value = case["id"]
        links.new(geometry, store.inputs["Geometry"])
        output_socket = output_socket_by_identifier(operation, case["output"])
        if output_socket is None:
            raise RuntimeError("missing output socket " + case["output"] + " on " + case["node_type"])
        links.new(output_socket, store.inputs["Value"])
        geometry = store.outputs["Geometry"]
    output = nodes.new("NodeGroupOutput")
    links.new(geometry, output.inputs["Geometry"])

    mesh = bpy.data.meshes.new("Potter Extended Oracle Mesh")
    mesh.from_pydata([(0.0, 0.0, 0.0)], [], [])
    obj = bpy.data.objects.new("Potter Extended Oracle", mesh)
    bpy.context.collection.objects.link(obj)
    modifier = obj.modifiers.new("Potter Extended Oracle", "NODES")
    modifier.node_group = tree
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    evaluated_mesh = evaluated.to_mesh()
    result = {}
    for case in cases:
        attribute = evaluated_mesh.attributes.get(case["id"])
        if attribute is None or len(attribute.data) != 1:
            raise RuntimeError("missing evaluated geometry attribute " + case["id"])
        item = attribute.data[0]
        attribute_type = case["attribute_type"]
        if attribute_type == "FLOAT_VECTOR":
            value = [float(component) for component in item.vector]
        elif attribute_type == "FLOAT_COLOR":
            value = [float(component) for component in item.color]
        elif attribute_type == "QUATERNION":
            quaternion = item.value
            value = [float(quaternion[index]) for index in (1, 2, 3, 0)]
        elif attribute_type == "BOOLEAN":
            value = bool(item.value)
        elif attribute_type == "INT":
            value = int(item.value)
        else:
            value = float(item.value)
        if isinstance(value, list):
            finite = all(math.isfinite(component) for component in value)
        elif isinstance(value, bool):
            finite = True
        else:
            finite = math.isfinite(value)
        if not finite:
            raise RuntimeError("non-finite Geometry Nodes result " + case["id"])
        result[case["id"]] = value
    evaluated.to_mesh_clear()
    return result



def wire_varying_shader_input(tree, case, node):
    identifier = case.get("shader_varying_input")
    if identifier is None:
        return
    value = case["inputs"][identifier]
    if not isinstance(value, (int, float, list)) or (
        isinstance(value, list) and len(value) != 3
    ):
        raise RuntimeError("shader probe varying input must be scalar or vector: " + case["id"])
    socket = socket_by_identifier(node, identifier)
    if socket is None:
        raise RuntimeError("missing varying input " + identifier + " on " + case["node_type"])
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    separate = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(coordinates.outputs["UV"], separate.inputs["Vector"])
    centered = tree.nodes.new("ShaderNodeMath")
    centered.operation = "SUBTRACT"
    tree.links.new(separate.outputs["X"], centered.inputs[0])
    centered.inputs[1].default_value = 0.5625
    scaled = tree.nodes.new("ShaderNodeMath")
    scaled.operation = "MULTIPLY"
    tree.links.new(centered.outputs["Value"], scaled.inputs[0])
    scaled.inputs[1].default_value = 1.0e-6
    if isinstance(value, list):
        offset = tree.nodes.new("ShaderNodeCombineXYZ")
        for axis in ("X", "Y", "Z"):
            tree.links.new(scaled.outputs["Value"], offset.inputs[axis])
        shifted = tree.nodes.new("ShaderNodeVectorMath")
        shifted.operation = "ADD"
        shifted.inputs[0].default_value = tuple(value)
        tree.links.new(offset.outputs["Vector"], shifted.inputs[1])
        tree.links.new(shifted.outputs["Vector"], socket)
    else:
        shifted = tree.nodes.new("ShaderNodeMath")
        shifted.operation = "ADD"
        shifted.inputs[0].default_value = float(value)
        tree.links.new(scaled.outputs["Value"], shifted.inputs[1])
        tree.links.new(shifted.outputs["Value"], socket)

def make_material(case):
    material = bpy.data.materials.new("Extended " + case["id"])
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    operation = tree.nodes.new(case["node_type"])
    set_node(case, operation, {})
    wire_varying_shader_input(tree, case, operation)
    emission = tree.nodes.new("ShaderNodeEmission")
    output_socket = output_socket_by_identifier(operation, case["output"])
    if output_socket is None:
        raise RuntimeError("missing output socket " + case["output"] + " on " + case["node_type"])
    if case["render_kind"] == "scalar":
        emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        tree.links.new(output_socket, emission.inputs["Strength"])
    else:
        emission.inputs["Strength"].default_value = 1.0
        tree.links.new(output_socket, emission.inputs["Color"])
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def build_shader_scene(shader_cases):
    scene = bpy.context.scene
    columns = len(shader_cases)
    materials = [make_material(case) for case in shader_cases]
    vertices, faces = [], []
    for column in range(columns):
        x = float(column)
        base = len(vertices)
        vertices.extend([(x, 0.0, 0.0), (x + 1.0, 0.0, 0.0),
                         (x + 1.0, 1.0, 0.0), (x, 1.0, 0.0)])
        faces.append((base, base + 1, base + 2, base + 3))
    mesh = bpy.data.meshes.new("Potter Extended Shader Probes")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    uv_layer = mesh.uv_layers.new(name="UVMap")
    uv_coordinates = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    for polygon in mesh.polygons:
        for loop_index, uv in zip(polygon.loop_indices, uv_coordinates):
            uv_layer.data[loop_index].uv = uv
    for material in materials:
        mesh.materials.append(material)
    for column, polygon in enumerate(mesh.polygons):
        polygon.material_index = column
    obj = bpy.data.objects.new("Potter Extended Shader Probes", mesh)
    scene.collection.objects.link(obj)

    camera_data = bpy.data.cameras.new("Potter Extended Probe Camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = float(columns)
    camera = bpy.data.objects.new("Potter Extended Probe Camera", camera_data)
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
    with tempfile.TemporaryDirectory(prefix="potter-math-extended-") as directory:
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
                if len(raw) != 3:
                    raise RuntimeError("missing shader output " + case["id"])
                results[case["id"]] = (
                    raw[0] if case["render_kind"] == "scalar" else raw
                ) if all(math.isfinite(component) for component in raw) else "non-finite"
        finally:
            bpy.data.images.remove(rendered)
    return results


def assert_catalogs():
    def enum_values(node, prop):
        return {item.identifier for item in node.bl_rna.properties[prop].enum_items}

    by_type = {}
    for case in cases:
        by_type.setdefault(case["node_type"], []).append(case)
    integer = bpy.data.node_groups.new("integer_catalog_check", "GeometryNodeTree").nodes.new("FunctionNodeIntegerMath")
    boolean = bpy.data.node_groups.new("boolean_catalog_check", "GeometryNodeTree").nodes.new("FunctionNodeBooleanMath")
    assert {case["properties"]["operation"] for case in by_type["FunctionNodeIntegerMath"]} == enum_values(integer, "operation")
    assert {case["properties"]["operation"] for case in by_type["FunctionNodeBooleanMath"]} == enum_values(boolean, "operation")

    compare = bpy.data.node_groups.new("compare_catalog_check", "GeometryNodeTree").nodes.new("FunctionNodeCompare")
    compare_types = {"FLOAT", "INT", "VECTOR", "RGBA", "STRING",
                     "OBJECT", "IMAGE", "COLLECTION", "MATERIAL", "FONT", "SOUND"}
    assert compare_types == {case["properties"]["data_type"] for case in by_type["FunctionNodeCompare"]}
    assert enum_values(compare, "mode") == {"ELEMENT", "LENGTH", "AVERAGE", "DOT_PRODUCT", "DIRECTION"}
    compare_ops = {case["properties"]["operation"] for case in by_type["FunctionNodeCompare"]}
    assert compare_ops == enum_values(compare, "operation")
    scalar_ops = {"LESS_THAN", "LESS_EQUAL", "GREATER_THAN", "GREATER_EQUAL", "EQUAL", "NOT_EQUAL"}
    identity_ops = {"EQUAL", "NOT_EQUAL"}
    expected_compare = {
        (data_type, "ELEMENT", operation)
        for data_type in ("FLOAT", "INT")
        for operation in scalar_ops
    }
    expected_compare.update(
        (data_type, "ELEMENT", operation)
        for data_type in ("STRING", "OBJECT", "IMAGE", "COLLECTION",
                          "MATERIAL", "FONT", "SOUND")
        for operation in identity_ops
    )
    expected_compare.update(
        ("RGBA", "ELEMENT", operation)
        for operation in ("EQUAL", "NOT_EQUAL", "BRIGHTER", "DARKER")
    )
    expected_compare.update(
        ("VECTOR", mode, operation)
        for mode in enum_values(compare, "mode")
        for operation in scalar_ops
    )
    actual_compare = {
        (case["properties"]["data_type"], case["properties"]["mode"], case["properties"]["operation"])
        for case in by_type["FunctionNodeCompare"]
    }
    assert actual_compare == expected_compare
    for case in by_type["FunctionNodeCompare"]:
        compare.data_type = case["properties"]["data_type"]
        compare.mode = case["properties"]["mode"]
        compare.operation = case["properties"]["operation"]

    map_node = new_material_node("map_catalog_check", "ShaderNodeMapRange")
    map_data = {
        (
            case["properties"]["data_type"],
            case["properties"]["interpolation_type"],
            case["properties"].get("clamp", True),
        )
        for case in by_type["ShaderNodeMapRange"]
    }
    assert map_data == {(data_type, interpolation, clamp)
                        for data_type in ("FLOAT", "FLOAT_VECTOR")
                        for interpolation in enum_values(map_node, "interpolation_type")
                        for clamp in (False, True)}
    clamp_node = new_material_node("clamp_catalog_check", "ShaderNodeClamp")
    assert {case["properties"]["clamp_type"] for case in by_type["ShaderNodeClamp"]} == enum_values(clamp_node, "clamp_type")

    mix_node = new_material_node("mix_catalog_check", "ShaderNodeMix")
    assert enum_values(mix_node, "data_type") == {"FLOAT", "VECTOR", "RGBA", "ROTATION"}
    store_node = bpy.data.node_groups.new(
        "rotation_attribute_catalog_check", "GeometryNodeTree"
    ).nodes.new("GeometryNodeStoreNamedAttribute")
    store_types = enum_values(store_node, "data_type")
    assert "QUATERNION" in store_types and "ROTATION" not in store_types
    store_node.data_type = "QUATERNION"
    mix_data = {(case["properties"]["data_type"], case["properties"]["factor_mode"],
                 case["properties"]["blend_type"], case["properties"]["clamp_factor"],
                 case["properties"]["clamp_result"]) for case in by_type["ShaderNodeMix"]}
    expected_mix = {
        ("FLOAT", "UNIFORM", "MIX", factor, result)
        for factor in (False, True) for result in (False, True)
    }
    expected_mix.update(
        ("VECTOR", factor_mode, "MIX", factor, result)
        for factor_mode in enum_values(mix_node, "factor_mode")
        for factor in (False, True) for result in (False, True)
    )
    expected_mix.update(
        ("RGBA", "UNIFORM", blend, factor, result)
        for blend in enum_values(mix_node, "blend_type")
        for factor in (False, True) for result in (False, True)
    )
    expected_mix.update(
        ("ROTATION", "UNIFORM", "MIX", factor, False)
        for factor in (False, True)
    )
    assert {case["properties"]["data_type"] for case in by_type["ShaderNodeMix"]} == enum_values(mix_node, "data_type")
    assert {
        (case["inputs"]["Factor_Float"], case["properties"]["clamp_factor"])
        for case in by_type["ShaderNodeMix"]
        if case["properties"]["data_type"] == "ROTATION"
    } == {(-0.25, False), (-0.25, True), (1.25, False), (1.25, True)}
    assert mix_data == expected_mix
    legacy = new_material_node("mixrgb_catalog_check", "ShaderNodeMixRGB")
    assert {case["properties"]["blend_type"] for case in by_type["ShaderNodeMixRGB"]} == enum_values(legacy, "blend_type")

    ramp_node = new_material_node("ramp_catalog_check", "ShaderNodeValToRGB")
    ramp_catalog_cases = [
        case for case in by_type["ShaderNodeValToRGB"]
        if case["id"].endswith("_color") or case["id"].endswith("_alpha")
    ]
    ramp_settings = [case["properties"]["color_ramp"] for case in ramp_catalog_cases]
    assert {(ramp["interpolation"], ramp["color_mode"], ramp["hue_interpolation"])
            for ramp in ramp_settings} == {
                (interpolation, color_mode, hue_mode)
                for interpolation in enum_values(ramp_node.color_ramp, "interpolation")
                for color_mode in enum_values(ramp_node.color_ramp, "color_mode")
                for hue_mode in enum_values(ramp_node.color_ramp, "hue_interpolation")
            }
    assert len([case for case in ramp_catalog_cases if case["output"] == "Color"]) == len(ramp_settings) // 2
    assert len([case for case in ramp_catalog_cases if case["output"] == "Alpha"]) == len(ramp_settings) // 2

    random_node = bpy.data.node_groups.new("random_catalog_check", "GeometryNodeTree").nodes.new("FunctionNodeRandomValue")
    assert {case["inputs"]["Probability"] for case in by_type["FunctionNodeRandomValue"]} == {0.0, 1.0}
    assert "BOOLEAN" in enum_values(random_node, "data_type")


def main():
    assert_catalogs()
    global GENERATED_IDS
    GENERATED_IDS = ids_for_compare()
    geometry_results = new_geometry_oracle()

    bpy.ops.wm.read_factory_settings(use_empty=True)
    shader_cases = [case for case in cases if case["shader_probe"]]
    build_shader_scene(shader_cases)
    shader_results = read_shader_reference(shader_cases, "cycles_probe.exr")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)

    realtime_cases = [case for case in cases if case["realtime_probe"]]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    build_shader_scene(realtime_cases)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    eevee_results = read_shader_reference(realtime_cases, "eevee_probe.exr")
    capture = {
        "engine": scene.render.engine,
        "resolution": [scene.render.resolution_x, scene.render.resolution_y],
        "cases": [
            {"id": case["id"], "render_kind": case["render_kind"]}
            for case in realtime_cases
        ],
        "eevee": eevee_results,
    }
    if capture_path:
        with open(capture_path, "w", encoding="utf-8") as capture_file:
            json.dump(capture, capture_file, sort_keys=True, indent=2, allow_nan=False)
            capture_file.write("\n")
    print("POTTER_NODE_MATH_EXTENDED=" + json.dumps(
        {
            "cases": cases,
            "geometry": geometry_results,
            "shader": shader_results,
            "eevee_engine": capture["engine"],
            "eevee_resolution": capture["resolution"],
            "eevee": eevee_results,
        },
        sort_keys=True, allow_nan=False,
    ))


main()
