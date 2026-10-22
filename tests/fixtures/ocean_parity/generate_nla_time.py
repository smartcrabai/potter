"""Regenerate the Blender 5.2 NLA oracle for an Ocean modifier's `time` binding.

Run from any directory with:
    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
        --python tests/fixtures/ocean_parity/generate_nla_time.py

Each case animates `modifiers["Ocean"].time` in an Action (linear keys), moves that Action into
one NLA strip with a given blend type and influence, then records the value Blender writes to the
modifier at every frame. The property is read after `scene.frame_set`, so it is the NLA-evaluated
value that Blender flushes to the original data.

The expected values are Blender's, not Potter's. `tests/ocean_nla_time_oracle.rs` builds the same
scene in Potter and checks `animated_modifier_scalar` against them.
"""

import json
import os

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(HERE, "nla_time_oracle.json")

# Keys and the stored `time` are chosen so that the default (1.0), the stored value (0.5), and the
# strip value differ, which separates REPLACE, ADD and COMBINE.
KEYS = ((1, 1.5), (3, 2.5))
STORED_TIME = 0.5
FRAMES = range(0, 7)

CASES = [
    {"name": "replace_half", "blend": "REPLACE", "influence": 0.5, "in_nla": True},
    {"name": "add_half", "blend": "ADD", "influence": 0.5, "in_nla": True},
    {"name": "combine_half", "blend": "COMBINE", "influence": 0.5, "in_nla": True},
    {"name": "replace_full", "blend": "REPLACE", "influence": 1.0, "in_nla": True},
    {"name": "add_full", "blend": "ADD", "influence": 1.0, "in_nla": True},
    {"name": "combine_full", "blend": "COMBINE", "influence": 1.0, "in_nla": True},
    {"name": "combine_zero", "blend": "COMBINE", "influence": 0.0, "in_nla": True},
    {"name": "not_in_nla", "blend": None, "influence": None, "in_nla": False},
]


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def add_ocean_object(name):
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=2, y_subdivisions=2, size=1.0)
    obj = bpy.context.object
    obj.name = name
    modifier = obj.modifiers.new("Ocean", "OCEAN")
    modifier.time = STORED_TIME
    return obj, modifier


def linear_fcurve_points(obj, modifier, action_slot):
    action = obj.animation_data.action
    for layer in action.layers:
        for strip in layer.strips:
            channelbag = strip.channelbag(action_slot)
            if channelbag is None:
                continue
            for curve in channelbag.fcurves:
                for point in curve.keyframe_points:
                    point.interpolation = "LINEAR"


def run_case(case):
    reset_scene()
    scene = bpy.context.scene
    obj, modifier = add_ocean_object(case["name"])
    if case["in_nla"]:
        for frame, value in KEYS:
            scene.frame_set(frame)
            modifier.time = value
            modifier.keyframe_insert(data_path="time", frame=frame)
        animation = obj.animation_data
        action = animation.action
        action_slot = animation.action_slot
        linear_fcurve_points(obj, modifier, action_slot)
        # Move the Action into one NLA strip and clear the active Action, so only the strip drives `time`.
        track = animation.nla_tracks.new()
        strip = track.strips.new(case["name"], int(KEYS[0][0]), action)
        strip.blend_type = case["blend"]
        # Blender applies `strip.influence` only through an animated influence (`use_animated_influence`);
        # otherwise the strip fade replaces it. Key the same influence at both keys, so it stays constant.
        strip.use_animated_influence = True
        animation.action = action
        strip.influence = case["influence"]
        for frame, _ in KEYS:
            strip.keyframe_insert(data_path="influence", frame=frame)
        animation.action = None
        strip.extrapolation = "NOTHING"
        modifier.time = STORED_TIME
    expected = {}
    for frame in FRAMES:
        scene.frame_set(frame)
        expected[str(frame)] = float(modifier.time)
    # The NLA domain is what decides whether Blender writes the default back; record the strip facts too.
    strip_facts = None
    if case["in_nla"]:
        strip_facts = {
            "frame_start": strip.frame_start,
            "frame_end": strip.frame_end,
            "action_frame_start": strip.action_frame_start,
            "action_frame_end": strip.action_frame_end,
            "blend_type": strip.blend_type,
            "influence": strip.influence,
            "use_animated_influence": strip.use_animated_influence,
            "extrapolation": strip.extrapolation,
        }
    return {
        "name": case["name"],
        "stored_time": STORED_TIME,
        "keys": [{"frame": frame, "value": value} for frame, value in KEYS] if case["in_nla"] else [],
        "strip": strip_facts,
        "expected_by_frame": expected,
    }


def main():
    cases = [run_case(case) for case in CASES]
    document = {
        "format": "potter-ocean-nla-time-oracle",
        "format_version": 1,
        "blender_version": bpy.app.version_string,
        "default_time": 1.0,
        "cases": cases,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("wrote", OUTPUT_PATH)


main()
