"""Generate Blender 5.2.2 F-Curve handle and evaluation expectations.

Run with:
    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
        --python tests/fixtures/fcurve_handles/generate_oracle.py -- OUTPUT_DIRECTORY

Writes `expected.json` and `expected.blend` into the output directory. The integration test runs this
generator and feeds its key positions, handle types, authored free/aligned handles, smoothing, and
Cycles modifier to the real Potter CLI.
"""
import json
import os
import sys

import bpy


if bpy.app.version != (5, 2, 2):
    raise RuntimeError("need Blender 5.2.2, got " + str(bpy.app.version_string))
if "--" not in sys.argv or len(sys.argv) <= sys.argv.index("--") + 1:
    raise RuntimeError("pass an output directory after --")

OUTPUT = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
os.makedirs(OUTPUT, exist_ok=True)
INITIAL_KEYS = [[0.0, 3.0], [3.0, -1.0], [7.0, 2.0], [12.0, 1.0], [18.0, 4.0]]
FINAL_KEYS = [[1.0, -2.0], [4.0, 5.0], [8.0, 1.0], [13.0, 8.0], [19.0, -1.0]]
SAMPLE_FRAMES = [2.5, 6.0, 10.5, 16.0]
CASES = [
    {"name": "auto", "handle_type": "AUTO", "auto_smoothing": "NONE"},
    {"name": "auto_clamped", "handle_type": "AUTO_CLAMPED", "auto_smoothing": "NONE"},
    {"name": "vector", "handle_type": "VECTOR", "auto_smoothing": "NONE"},
    {"name": "aligned", "handle_type": "ALIGNED", "auto_smoothing": "NONE"},
    {"name": "free", "handle_type": "FREE", "auto_smoothing": "NONE"},
    {"name": "smoothing_none", "handle_type": "AUTO", "auto_smoothing": "NONE"},
    {"name": "smoothing_cont_accel", "handle_type": "AUTO", "auto_smoothing": "CONT_ACCEL"},
    {
        "name": "cyclic",
        "handle_type": "AUTO_CLAMPED",
        "auto_smoothing": "NONE",
        "modifiers": [
            {
                "type": "CYCLES",
                "properties": {
                    "mode_before": "REPEAT",
                    "cycles_before": 0,
                    "mode_after": "REPEAT",
                    "cycles_after": 0,
                },
            }
        ],
    },
    {
        "name": "cyclic_cont_accel_auto_clamped",
        "handle_type": "AUTO_CLAMPED",
        "auto_smoothing": "CONT_ACCEL",
        "initial_keys": [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, -1.0], [4.0, 0.0]],
        "final_keys": [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, -1.0], [4.0, 0.0]],
        "sample_frames": [-3.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 7.5],
        "modifiers": [
            {
                "type": "CYCLES",
                "properties": {
                    "mode_before": "REPEAT",
                    "cycles_before": 0,
                    "mode_after": "REPEAT",
                    "cycles_after": 0,
                },
            }
        ],
    },
    {
        "name": "mixed_first_auto_vector_constant",
        "handle_type": "AUTO",
        "auto_smoothing": "NONE",
        "extrapolation": "CONSTANT",
        "key_handle_types": [["AUTO", "VECTOR"], ["AUTO", "AUTO"], ["AUTO", "AUTO"],
                             ["AUTO", "AUTO"], ["AUTO", "AUTO"]],
        "sample_frames": [-2.0, 2.5, 6.0, 10.5, 16.0, 21.0],
    },
    {
        "name": "vector_auto_endpoint_cont_accel_linear",
        "handle_type": "AUTO",
        "auto_smoothing": "CONT_ACCEL",
        "extrapolation": "LINEAR",
        "key_handle_types": [["VECTOR", "AUTO"], ["AUTO", "AUTO"], ["AUTO", "AUTO"],
                             ["AUTO", "AUTO"], ["AUTO", "AUTO"]],
        "sample_frames": [1.0, 2.5, 6.0, 10.5, 16.0, 19.0],
    },
]


def create_location_curve(obj):
    action = bpy.data.actions.new(obj.name + "Action")
    slot = action.slots.new("OBJECT", obj.name)
    obj.animation_data_create()
    obj.animation_data.action = action
    obj.animation_data.action_slot = slot
    layer = action.layers.new("Animation")
    strip = layer.strips.new(type="KEYFRAME")
    channelbag = strip.channelbag(slot, ensure=True)
    return channelbag.fcurves.new(data_path="location", index=0)


def authored_handles(handle_type, frame, value, index):
    if handle_type == "FREE":
        return (
            [frame - 0.7 - index * 0.03, value - 0.25 - index * 0.08],
            [frame + 1.2 + index * 0.04, value + 0.35 + index * 0.06],
        )
    if handle_type == "ALIGNED":
        return (
            [frame - 0.7 - index * 0.03, value - 0.6 - index * 0.1],
            [frame + 1.2 + index * 0.04, value + 0.25 + index * 0.08],
        )
    return None, None


def generate_case(spec):
    name = spec["name"]
    obj = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(obj)
    curve = create_location_curve(obj)
    initial_keys = spec.get("initial_keys", INITIAL_KEYS)
    final_keys = spec.get("final_keys", FINAL_KEYS)
    for frame, value in initial_keys:
        point = curve.keyframe_points.insert(frame, value)
        point.interpolation = "BEZIER"

    curve.auto_smoothing = spec["auto_smoothing"]
    curve.extrapolation = spec.get("extrapolation", "CONSTANT")
    handle_type = spec["handle_type"]
    key_handle_types = spec.get("key_handle_types")
    key_records = []
    for index, (point, (frame, value)) in enumerate(zip(curve.keyframe_points, final_keys)):
        point.co = (frame, value)
        point.interpolation = "BEZIER"
        left, right = authored_handles(handle_type, frame, value, index)
        if left is not None:
            point.handle_left_type = "FREE"
            point.handle_right_type = "FREE"
            point.handle_left = left
            point.handle_right = right
        left_type, right_type = (
            key_handle_types[index] if key_handle_types else (handle_type, handle_type)
        )
        point.handle_left_type = left_type
        point.handle_right_type = right_type
        key_records.append(
            {
                "frame": float(frame),
                "value": float(value),
                "handle_left_type": point.handle_left_type.lower(),
                "handle_right_type": point.handle_right_type.lower(),
                "input_handle_left": left,
                "input_handle_right": right,
            }
        )

    modifiers = spec.get("modifiers", [])
    for modifier_spec in modifiers:
        modifier = curve.modifiers.new(modifier_spec["type"])
        for property_name, property_value in modifier_spec.get("properties", {}).items():
            setattr(modifier, property_name, property_value)
    curve.update()

    for point, key_record in zip(curve.keyframe_points, key_records):
        key_record["handle_left"] = [float(point.handle_left[0]), float(point.handle_left[1])]
        key_record["handle_right"] = [float(point.handle_right[0]), float(point.handle_right[1])]

    frames = list(spec.get("sample_frames", SAMPLE_FRAMES))
    if modifiers and "sample_frames" not in spec:
        frames.extend([-3.5, 24.5])
    samples = [
        {"frame": float(frame), "value": float(curve.evaluate(frame))}
        for frame in frames
    ]
    return {
        "name": name,
        "handle_type": handle_type.lower(),
        "auto_smoothing": spec["auto_smoothing"].lower(),
        "extrapolation": spec.get("extrapolation", "CONSTANT").lower(),
        "modifiers": modifiers,
        "initial_keys": initial_keys,
        "keys": key_records,
        "samples": samples,
    }

def point_record(point):
    return {
        "frame": float(point.co[0]),
        "value": float(point.co[1]),
        "interpolation": point.interpolation.lower(),
        "handle_left_type": point.handle_left_type.lower(),
        "handle_right_type": point.handle_right_type.lower(),
        "handle_left": [float(point.handle_left[0]), float(point.handle_left[1])],
        "handle_right": [float(point.handle_right[0]), float(point.handle_right[1])],
    }


def generate_subdivide_case():
    name = "subdivide_nonauto_handles"
    obj = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(obj)
    curve = create_location_curve(obj)
    curve.auto_smoothing = "NONE"
    authored_keys = [
        {
            "frame": 1.0,
            "value": -1.0,
            "handle_left": [0.0, -1.5],
            "handle_right": [3.0, 1.25],
        },
        {
            "frame": 9.0,
            "value": 5.0,
            "handle_left": [7.0, 7.5],
            "handle_right": [10.0, 4.0],
        },
    ]
    for key in authored_keys:
        point = curve.keyframe_points.insert(key["frame"], key["value"])
        point.interpolation = "BEZIER"
        point.handle_left_type = "FREE"
        point.handle_right_type = "FREE"
        point.handle_left = key["handle_left"]
        point.handle_right = key["handle_right"]
    curve.update()
    seed_keys = [point_record(point) for point in curve.keyframe_points]

    inserted_frame = 5.0
    inserted_value = float(curve.evaluate(inserted_frame))
    inserted = curve.keyframe_points.insert(inserted_frame, inserted_value)
    inserted.interpolation = "BEZIER"
    curve.update()
    keys = [point_record(point) for point in curve.keyframe_points]
    frames = [0.5, 2.0, 3.5, inserted_frame, 6.5, 8.5, 9.5]
    samples = [
        {"frame": float(frame), "value": float(curve.evaluate(frame))}
        for frame in frames
    ]
    return {
        "name": name,
        "scenario": "subdivide_nonauto_handles",
        "handle_type": "free",
        "auto_smoothing": "none",
        "extrapolation": "constant",
        "modifiers": [],
        "seed_keys": seed_keys,
        "inserted_key": keys[1],
        "keys": keys,
        "samples": samples,
    }

def generate_subdivide_auto_free_case():
    name = "subdivide_auto_free_between_auto"
    obj = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(obj)
    curve = create_location_curve(obj)
    curve.auto_smoothing = "NONE"
    for frame, value in [[1.0, -1.0], [9.0, 5.0]]:
        point = curve.keyframe_points.insert(frame, value)
        point.interpolation = "BEZIER"
        point.handle_left_type = "AUTO"
        point.handle_right_type = "AUTO"
    curve.update()
    seed_keys = [point_record(point) for point in curve.keyframe_points]

    inserted_frame = 5.0
    inserted_value = float(curve.evaluate(inserted_frame))
    inserted = curve.keyframe_points.insert(inserted_frame, inserted_value)
    inserted.interpolation = "BEZIER"
    inserted.handle_left_type = "AUTO"
    inserted.handle_right_type = "AUTO"
    curve.update()
    inserted = curve.keyframe_points[1]
    inserted.handle_right_type = "FREE"
    curve.update()
    keys = [point_record(point) for point in curve.keyframe_points]
    frames = [0.5, 2.0, 3.5, inserted_frame, 6.5, 8.5, 9.5]
    samples = [
        {"frame": float(frame), "value": float(curve.evaluate(frame))}
        for frame in frames
    ]
    return {
        "name": name,
        "scenario": "subdivide_auto_free_between_auto",
        "handle_type": "auto",
        "auto_smoothing": "none",
        "extrapolation": "constant",
        "modifiers": [],
        "seed_keys": seed_keys,
        "inserted_key": keys[1],
        "keys": keys,
        "samples": samples,
    }


result = {
    "generator": "tests/fixtures/fcurve_handles/generate_oracle.py",
    "blender_version": bpy.app.version_string,
    "tolerance": 1.0e-5,
    "cases": (
        [generate_case(spec) for spec in CASES]
        + [generate_subdivide_case(), generate_subdivide_auto_free_case()]
    ),
}
with open(os.path.join(OUTPUT, "expected.json"), "w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2, allow_nan=False)
    handle.write("\n")
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUTPUT, "expected.blend"))
print("wrote Blender 5.2.2 F-Curve handle oracle:", os.path.join(OUTPUT, "expected.json"))
