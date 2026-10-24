"""Generate the Blender 5.2.2 F-Curve modifier evaluation oracle.

Run with:
    blender -b --factory-startup --python generate_oracle.py -- OUTPUT_DIRECTORY

Writes `fcurve_modifiers.blend` and `expected.json` into the output directory.
"""

import json
import math
import os
import sys

import bpy


def configure_modifier(curve, source):
    modifier = curve.modifiers.new(source["type"])
    properties = dict(source.get("properties", {}))
    control_points = properties.pop("control_points", None)
    for name, value in properties.items():
        setattr(modifier, name, value)
    if control_points is not None:
        for point in control_points:
            envelope_point = modifier.control_points.add(point["frame"])
            envelope_point.min = point["min"]
            envelope_point.max = point["max"]
    return modifier


CASES = [
    {
        "name": "generator_expanded",
        "keys": [[0.0, 0.0], [4.0, 1.0]],
        "modifiers": [{"type": "GENERATOR", "properties": {
            "mode": "POLYNOMIAL", "poly_order": 2,
            "coefficients": [0.25, 0.75, -0.125], "use_additive": False,
        }}],
        "frames": [-1.25, 0.0, 0.5, 2.25, 4.0, 4.75],
    },
    {
        "name": "generator_additive",
        "keys": [[0.0, 0.0], [4.0, 4.0]],
        "modifiers": [{"type": "GENERATOR", "properties": {
            "mode": "POLYNOMIAL", "poly_order": 1,
            "coefficients": [0.5, 0.25], "use_additive": True,
        }}],
        "frames": [-0.5, 0.0, 1.25, 2.5, 4.0, 4.75],
    },
    {
        "name": "generator_factorised",
        "keys": [[0.0, 1.0], [4.0, 2.0]],
        "modifiers": [{"type": "GENERATOR", "properties": {
            "mode": "POLYNOMIAL_FACTORISED", "poly_order": 2,
            "coefficients": [0.5, 1.0, -0.25, 2.0], "use_additive": False,
        }}],
        "frames": [-0.5, 0.0, 0.75, 2.25, 4.0, 4.5],
    },
    {
        "name": "generator_factorised_defaults",
        "keys": [[0.0, 1.0], [4.0, 2.0]],
        "modifiers": [{"type": "GENERATOR", "properties": {
            "mode": "POLYNOMIAL_FACTORISED", "poly_order": 2,
        }}],
        "frames": [-0.5, 0.0, 0.75, 2.25, 4.0, 4.5],
    },
    {
        "name": "function_generator",
        "keys": [[0.0, -1.0], [4.0, 3.0]],
        "modifiers": [{"type": "FNGENERATOR", "properties": {
            "function_type": "COS", "amplitude": 0.75,
            "phase_multiplier": 0.4, "phase_offset": 0.2,
            "value_offset": 0.1, "use_additive": False,
        }}],
        "frames": [-0.75, 0.0, 0.625, 2.25, 4.0, 4.5],
    },
    {
        "name": "envelope",
        "keys": [[0.0, 0.5], [4.0, 3.0]],
        "modifiers": [{"type": "ENVELOPE", "properties": {
            "reference_value": 1.0, "default_min": -2.0, "default_max": 2.0,
            "control_points": [
                {"frame": 0.0, "min": -1.0, "max": 1.0},
                {"frame": 2.0, "min": -0.5, "max": 2.0},
                {"frame": 4.0, "min": 0.5, "max": 3.0},
            ],
        }}],
        "frames": [-0.5, 0.0, 0.75, 2.25, 4.0, 4.5],
    },
    {
        "name": "cycles_repeat",
        "keys": [[2.0, 1.0], [6.0, 5.0]],
        "modifiers": [{"type": "CYCLES", "properties": {
            "mode_before": "REPEAT", "cycles_before": 0,
            "mode_after": "REPEAT", "cycles_after": 0,
        }}],
        "frames": [-6.0, -2.0, -0.5, 2.0, 3.25, 6.0, 6.25, 10.0, 14.25],
    },
    {
        "name": "cycles_offset_mirror_finite",
        "keys": [[2.0, -1.0], [6.0, 3.0]],
        "modifiers": [{"type": "CYCLES", "properties": {
            "mode_before": "REPEAT_OFFSET", "cycles_before": 1,
            "mode_after": "MIRROR", "cycles_after": 2,
        }}],
        "frames": [-3.0, 0.0, 1.5, 2.0, 6.0, 6.5, 10.0, 14.0, 15.0],
    },
    {
        "name": "cycles_mirror_before_offset_after_finite",
        "keys": [[2.0, -1.0], [6.0, 3.0]],
        "modifiers": [{"type": "CYCLES", "properties": {
            "mode_before": "MIRROR", "cycles_before": 2,
            "mode_after": "REPEAT_OFFSET", "cycles_after": 1,
        }}],
        "frames": [-6.0, -5.25, -3.0, 6.0, 6.5, 10.0, 11.0],
    },
    {
        "name": "cycles_none",
        "keys": [[2.0, 0.5], [6.0, 2.5]],
        "modifiers": [{"type": "CYCLES", "properties": {
            "mode_before": "NONE", "cycles_before": 0,
            "mode_after": "NONE", "cycles_after": 0,
        }}],
        "frames": [-1.25, 0.0, 2.0, 4.25, 6.0, 8.5],
    },
    *[
        {
            "name": "noise_" + blend_type.lower(),
            "keys": [[0.0, -0.5], [6.0, 1.5]],
            "modifiers": [{"type": "NOISE", "properties": {
                "blend_type": blend_type, "scale": 2.3, "strength": 0.45,
                "phase": 0.25, "offset": 0.5, "depth": 3,
                "lacunarity": 2.1, "roughness": 0.55,
            }}],
            "frames": [0.25, 1.5, 2.75, 5.5, 7.25],
        }
        for blend_type in ("ADD", "SUBTRACT", "MULTIPLY", "REPLACE")
    ],
    {
        "name": "noise_legacy",
        "keys": [[0.0, 0.25], [6.0, 1.25]],
        "modifiers": [{"type": "NOISE", "properties": {
            "blend_type": "REPLACE", "scale": 2.0, "strength": 0.4,
            "phase": 0.65, "offset": 1.0, "depth": 3,
            "use_legacy_noise": True,
        }}],
        "frames": [0.25, 1.5, 2.75, 5.5, 7.25],
    },
    {
        "name": "limits",
        "keys": [[0.0, -2.0], [6.0, 3.0]],
        "modifiers": [{"type": "LIMITS", "properties": {
            "use_min_x": True, "min_x": 1.0,
            "use_max_x": True, "max_x": 4.0,
            "use_min_y": True, "min_y": -0.5,
            "use_max_y": True, "max_y": 1.5,
        }}],
        "frames": [-1.0, 0.5, 1.0, 2.25, 4.5, 6.5],
    },
    {
        "name": "stepped",
        "keys": [[0.0, 0.0], [10.0, 10.0]],
        "modifiers": [{"type": "STEPPED", "properties": {
            "frame_step": 2.5, "frame_offset": 0.25,
            "use_frame_start": True, "frame_start": 2.5,
            "use_frame_end": True, "frame_end": 8.0,
        }}],
        "frames": [0.75, 2.25, 2.5, 3.75, 5.25, 8.0, 8.5],
    },
    {
        "name": "muted_modifier",
        "keys": [[0.0, 0.0], [4.0, 4.0]],
        "modifiers": [{"type": "GENERATOR", "properties": {
            "mode": "POLYNOMIAL", "poly_order": 1,
            "coefficients": [10.0, 4.0], "mute": True,
        }}],
        "frames": [-0.25, 0.0, 1.5, 3.75, 4.25],
    },
    {
        "name": "restricted_influenced_stack",
        "keys": [[0.0, 0.0], [10.0, 10.0]],
        "modifiers": [
            {"type": "STEPPED", "properties": {
                "frame_step": 1.5, "frame_offset": 0.25,
                "use_restricted_range": True, "frame_start": 2.0,
                "frame_end": 8.0, "blend_in": 2.0, "blend_out": 2.0,
                "use_influence": True, "influence": 0.6,
            }},
            {"type": "NOISE", "properties": {
                "blend_type": "ADD", "scale": 3.25, "strength": 0.35,
                "phase": 0.5, "offset": 0.0, "depth": 2,
                "use_restricted_range": True, "frame_start": 2.0,
                "frame_end": 8.0, "blend_in": 2.0, "blend_out": 2.0,
                "use_influence": True, "influence": 0.4,
            }},
            {"type": "LIMITS", "properties": {
                "use_min_y": True, "min_y": 1.25,
                "use_max_y": True, "max_y": 7.75,
            }},
        ],
        "frames": [0.75, 2.25, 3.75, 5.25, 6.75, 8.5, 9.25],
    },
    {
        "name": "cycles_stacked",
        "keys": [[2.0, 1.0], [6.0, 5.0]],
        "modifiers": [
            {"type": "CYCLES", "properties": {
                "mode_before": "REPEAT", "mode_after": "REPEAT_OFFSET",
            }},
            {"type": "CYCLES", "properties": {
                "mode_before": "REPEAT", "mode_after": "REPEAT_OFFSET",
            }},
        ],
        "frames": [2.0, 4.0, 6.0, 6.5, 7.0, 8.0, 10.0],
    },
    {
        "name": "empty_noise_static",
        "location": 7.0,
        "keys": [],
        "modifiers": [{"type": "NOISE", "properties": {
            "blend_type": "REPLACE", "scale": 1.0, "strength": 1.0,
            "phase": 0.0, "offset": 0.0, "depth": 1,
        }}],
        "frames": [0.0, 1.25, 5.0],
    },
]


def main(output_directory):
    if bpy.app.version != (5, 2, 2):
        raise RuntimeError("the F-Curve oracle requires Blender 5.2.2, found %s" %
                           bpy.app.version_string)
    os.makedirs(output_directory, exist_ok=True)
    scene = bpy.context.scene
    scene.frame_start = -10
    scene.frame_end = 20
    expected = {"blender_version": bpy.app.version_string, "tolerance": 2e-6, "cases": []}
    for case in CASES:
        obj = bpy.data.objects.new(case["name"], None)
        scene.collection.objects.link(obj)
        obj.location.x = case.get("location", 0.0)
        action = bpy.data.actions.new(case["name"] + " Action")
        slot = action.slots.new("OBJECT", obj.name)
        layer = action.layers.new("Layer")
        strip = layer.strips.new(type="KEYFRAME")
        channelbag = strip.channelbag(slot, ensure=True)
        curve = channelbag.fcurves.new(data_path="location", index=0)
        for frame, value in case["keys"]:
            point = curve.keyframe_points.insert(frame, value)
            point.interpolation = "LINEAR"
        for modifier_source in case["modifiers"]:
            configure_modifier(curve, modifier_source)
        curve.update()
        obj.animation_data_create()
        obj.animation_data.action = action
        obj.animation_data.action_slot = slot
        samples = []
        for frame in case["frames"]:
            integral_frame = math.floor(frame)
            scene.frame_set(integral_frame, subframe=frame - integral_frame)
            evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
            samples.append({"frame": frame, "value": float(evaluated.location.x)})
        expected["cases"].append({"name": case["name"], "samples": samples})
    bpy.ops.wm.save_as_mainfile(
        filepath=os.path.join(output_directory, "fcurve_modifiers.blend"))
    with open(os.path.join(output_directory, "expected.json"), "w", encoding="utf-8") as output:
        json.dump(expected, output, separators=(",", ":"), sort_keys=True)
        output.write("\n")


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    main(args[0] if args else os.path.dirname(os.path.abspath(__file__)))
