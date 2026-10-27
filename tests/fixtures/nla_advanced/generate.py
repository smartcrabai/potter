import bpy
import json
import sys
import math
from pathlib import Path


if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"NLA oracle requires Blender 5.2.2, found {bpy.app.version_string}")

root = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
root.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 28
scene.render.fps = 24


def make_owner(name, kind="EMPTY"):
    if kind == "ARMATURE":
        data = bpy.data.armatures.new(name + "Data")
        owner = bpy.data.objects.new(name, data)
        scene.collection.objects.link(owner)
        bpy.context.view_layer.objects.active = owner
        owner.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        bone = data.edit_bones.new("OracleBone")
        bone.head = (0.0, 0.0, 0.0)
        bone.tail = (0.0, 1.0, 0.0)
        bpy.ops.object.mode_set(mode="OBJECT")
        owner.select_set(False)
    else:
        owner = bpy.data.objects.new(name, None)
        scene.collection.objects.link(owner)
    return owner


def make_action(owner, name, channels):
    action = bpy.data.actions.new(name)
    slot = action.slots.new("OBJECT", owner.name)
    layer = action.layers.new("Animation")
    action_strip = layer.strips.new(type="KEYFRAME")
    channelbag = action_strip.channelbag(slot, ensure=True)
    for data_path, index, keys in channels:
        curve = channelbag.fcurves.new(data_path=data_path, index=index)
        for frame, value in keys:
            point = curve.keyframe_points.insert(frame, value, options={"FAST"})
            point.interpolation = "LINEAR"
        curve.update()
    return action, slot


def scalar_action(owner, name, values, channels=None):
    curves = [("location", 0, values)] if channels is None else channels
    return make_action(owner, name, curves)


def add_track(owner, name, strips):
    animation = owner.animation_data_create()
    track = animation.nla_tracks.new()
    track.name = name
    for spec in strips:
        action, slot = spec["action"]
        strip = track.strips.new(spec["name"], spec["start"], action)
        strip.action_slot = slot
        for property_name, value in spec.get("settings", {}).items():
            setattr(strip, property_name, value)
        if "action_range" in spec:
            strip.action_frame_start, strip.action_frame_end = spec["action_range"]
        if "scale" in spec:
            strip.scale = spec["scale"]
        if "repeat" in spec:
            strip.repeat = spec["repeat"]
        if "end" in spec:
            strip.frame_end = spec["end"]
        spec["created"] = strip
    return track




def make_clip(owner, name, values, start=2, settings=None, action_range=None,
              scale=None, repeat=None, blend="REPLACE", influence=1.0,
              extrapolation="HOLD", channels=None):
    action, slot = scalar_action(owner, name + "Action", values, channels)
    spec = {"name": name, "start": start, "action": (action, slot),
            "settings": {"blend_type": blend, "influence": influence,
                         "extrapolation": extrapolation}}
    if settings:
        spec["settings"].update(settings)
    if action_range is not None:
        spec["action_range"] = action_range
    if scale is not None:
        spec["scale"] = scale
    if repeat is not None:
        spec["repeat"] = repeat
    return spec


# Timing, restricted action ranges, scale/repeat, and reverse playback.
timing = make_owner("NLA_TimingRangeReverse")
timing.location = (0.25, -0.5, 0.75)
timing_clip = make_clip(
    timing, "RangeRepeatReverse", [(1, 0.0), (4, 6.0), (8, 2.0)],
    start=3, action_range=(2.0, 7.0), scale=1.25, repeat=1.5,
    settings={"use_reverse": True, "use_sync_length": True},
    channels=[("location", 0, [(1, 0.0), (4, 6.0), (8, 2.0)]),
              ("rotation_euler", 2, [(1, 0.0), (8, 0.9)])])
add_track(timing, "Timing", [timing_clip])

# Exercise each scalar/transform blend mode from the same non-neutral basis.
blend_specs = (
    ("Replace", "REPLACE", [(2, 8.0), (8, 8.0)], [(2, 0.6), (8, 0.6)], [(2, 0.4), (8, 0.4)]),
    ("Add", "ADD", [(2, 1.5), (8, 1.5)], [(2, 1.2), (8, 1.2)], [(2, 0.25), (8, 0.25)]),
    ("Combine", "COMBINE", [(2, 0.75), (8, 0.75)], [(2, 1.4), (8, 1.4)], [(2, 0.7), (8, 0.7)]),
    ("Subtract", "SUBTRACT", [(2, 0.5), (8, 0.5)], [(2, 0.6), (8, 0.6)], [(2, 0.2), (8, 0.2)]),
    ("Multiply", "MULTIPLY", [(2, 1.5), (8, 1.5)], [(2, 1.3), (8, 1.3)], [(2, 0.8), (8, 0.8)]),
)
blend_names = []
for label, mode, locations, scales, rotations in blend_specs:
    owner = make_owner("NLA_Blend" + label)
    owner.location = (2.0, -0.75, 0.5)
    owner.scale = (1.7, 0.8, 1.1)
    owner.rotation_euler = (0.1, -0.2, 0.35)
    clip = make_clip(
        owner, label, locations, start=3, action_range=(2.0, 8.0),
        blend=mode, extrapolation="NOTHING",
        channels=[("location", 0, locations), ("scale", 0, scales),
                  ("rotation_euler", 2, rotations)])
    add_track(owner, mode, [clip])
    blend_names.append(owner.name)

quaternion = make_owner("NLA_Quaternion270")
quaternion.rotation_mode = "QUATERNION"
angle = math.radians(270.0)
quaternion_values = (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0))
quaternion_clip = make_clip(
    quaternion, "Quaternion270", [(1, 0.0), (8, 0.0)], start=2,
    action_range=(1.0, 8.0), blend="COMBINE", influence=0.25,
    settings={"use_animated_influence": True},
    channels=[("rotation_quaternion", index, [(1, value), (8, value)])
              for index, value in enumerate(quaternion_values)])
add_track(quaternion, "QuaternionCombine", [quaternion_clip])

quaternion_replace = make_owner("NLA_Quaternion270Replace")
quaternion_replace.rotation_mode = "QUATERNION"
quaternion_replace_clip = make_clip(
    quaternion_replace, "Quaternion270Replace", [(1, 0.0), (8, 0.0)], start=2,
    action_range=(1.0, 8.0), blend="REPLACE", influence=0.25,
    settings={"use_animated_influence": True},
    channels=[("rotation_quaternion", index, [(1, value), (8, value)])
              for index, value in enumerate(quaternion_values)])
add_track(quaternion_replace, "QuaternionReplace", [quaternion_replace_clip])

euler = make_owner("NLA_EulerAccumulation")
euler.rotation_mode = "XYZ"
euler_base = make_clip(
    euler, "Euler100", [(1, 0.0), (8, 0.0)], start=2,
    blend="REPLACE",
    channels=[("rotation_euler", 2,
               [(1, math.radians(100.0)), (8, math.radians(100.0))])])
euler_add = make_clip(
    euler, "Euler10", [(1, 0.0), (8, 0.0)], start=2,
    blend="ADD",
    channels=[("rotation_euler", 2,
               [(1, math.radians(10.0)), (8, math.radians(10.0))])])
add_track(euler, "EulerLower", [euler_base])
add_track(euler, "EulerUpper", [euler_add])

# Track order uses non-commuting additions and multiplications; mute and solo are separate controls.
ordered = make_owner("NLA_TrackOrder")
ordered.location.x = 2.0
first = make_clip(ordered, "OrderAdd", [(1, 3.0), (8, 3.0)], start=2,
                  blend="ADD", extrapolation="HOLD")
second = make_clip(ordered, "OrderMultiply", [(1, 4.0), (8, 4.0)], start=2,
                   blend="MULTIPLY", extrapolation="HOLD")
add_track(ordered, "01_Add", [first])
add_track(ordered, "02_Multiply", [second])

muted = make_owner("NLA_TrackMute")
muted.location.x = 2.0
add_track(muted, "VisibleAdd", [make_clip(muted, "VisibleAdd", [(1, 3.0), (8, 3.0)],
                                          start=2, blend="ADD")])
muted_track = add_track(muted, "MutedReplace", [make_clip(
    muted, "MutedReplace", [(1, 50.0), (8, 50.0)], start=2, blend="REPLACE")])
muted_track.mute = True

strip_muted = make_owner("NLA_StripMute")
strip_muted.location.x = 2.0
add_track(strip_muted, "StripMute", [make_clip(
    strip_muted, "MutedStrip", [(1, 50.0), (8, 50.0)], start=2,
    settings={"mute": True}, blend="REPLACE")])

muted_domain = make_owner("NLA_MutedStripDomainReset")
muted_domain.location = (2.0, 3.0, 4.0)
muted_domain_clip = make_clip(
    muted_domain, "MutedDomain", [(1, 50.0), (4, 50.0)], start=19,
    action_range=(1.0, 4.0), settings={"mute": True},
    extrapolation="HOLD")
add_track(muted_domain, "MutedDomain", [muted_domain_clip])

solo = make_owner("NLA_TrackSolo")
solo.location.x = 2.0
add_track(solo, "NonSoloAdd", [make_clip(solo, "NonSoloAdd", [(1, 30.0), (8, 30.0)],
                                         start=2, blend="ADD")])
solo_track = add_track(solo, "SoloReplace", [make_clip(
    solo, "SoloReplace", [(1, 7.0), (8, 7.0)], start=2, blend="REPLACE")])
solo_track.is_solo = True

solo_action = make_owner("NLA_SoloWithActiveAction")
solo_action.location.x = 2.0
add_track(solo_action, "NonSoloAdd", [make_clip(
    solo_action, "NonSoloAdd", [(1, 30.0), (8, 30.0)], start=2, blend="ADD")])
solo_action_track = add_track(solo_action, "SoloReplace", [make_clip(
    solo_action, "SoloReplace", [(1, 7.0), (8, 7.0)], start=2)])
solo_action_track.is_solo = True
solo_action_data = solo_action.animation_data
solo_action_action, solo_action_slot = make_action(
    solo_action, "SoloActiveAction", [("location", 0, [(2, 4.0), (8, 4.0)])])
solo_action_data.action = solo_action_action
solo_action_data.action_slot = solo_action_slot
solo_action_data.action_blend_type = "ADD"
solo_action_data.action_influence = 0.5
solo_action_data.action_extrapolation = "HOLD"
solo_domain = make_owner("NLA_SoloStripDomainReset")
solo_domain.location.x = 2.0
solo_domain_track = add_track(solo_domain, "SoloRotationOnly", [make_clip(
    solo_domain, "SoloRotationOnly", [], start=1,
    channels=[("rotation_euler", 2, [(1, 0.4), (8, 0.4)])])])
solo_domain_track.is_solo = True
solo_domain_data = solo_domain.animation_data
solo_domain_action, solo_domain_slot = make_action(
    solo_domain, "SoloDomainActiveAction", [("location", 0, [(1, 4.0), (8, 4.0)])])
solo_domain_data.action = solo_domain_action
solo_domain_data.action_slot = solo_domain_slot
solo_domain_data.action_blend_type = "REPLACE"
solo_domain_data.action_influence = 1.0
solo_domain_data.action_extrapolation = "HOLD"


# Hold, hold-forward, nothing, and a gap between neighboring clips.
hold_names = []
for label, extrapolation in (("Hold", "HOLD"), ("HoldForward", "HOLD_FORWARD"),
                             ("Nothing", "NOTHING")):
    owner = make_owner("NLA_Extrapolation" + label)
    owner.location.x = 1.25
    clip = make_clip(owner, label, [(1, 2.0), (4, 5.0)], start=4,
                     action_range=(1.0, 4.0), extrapolation=extrapolation)
    add_track(owner, label, [clip])
    hold_names.append(owner.name)

gapped = make_owner("NLA_Gap")
gapped.location.x = -2.0
gap_a = make_clip(gapped, "GapA", [(1, 1.0), (4, 4.0)], start=3,
                 action_range=(1.0, 4.0), extrapolation="NOTHING")
gap_b = make_clip(gapped, "GapB", [(1, 8.0), (4, 5.0)], start=10,
                 action_range=(1.0, 4.0), extrapolation="NOTHING")
add_track(gapped, "SeparatedClips", [gap_a, gap_b])

gap_hold = make_owner("NLA_GapHold")
gap_hold.location.x = -1.5
gap_hold_a = make_clip(gap_hold, "GapHoldA", [(1, 1.0), (4, 4.0)], start=3,
                       action_range=(1.0, 4.0), extrapolation="HOLD")
gap_hold_b = make_clip(gap_hold, "GapHoldB", [(1, 8.0), (4, 5.0)], start=10,
                       action_range=(1.0, 4.0), extrapolation="NOTHING")
add_track(gap_hold, "HeldClipGap", [gap_hold_a, gap_hold_b])

# Explicit animated influence, explicit blend ramps, and cross-track automatic blends.
blended = make_owner("NLA_InfluenceBlend")
blended.location.x = 1.5
blend_clip = make_clip(blended, "InfluenceBlend", [(1, 8.0), (9, 8.0)], start=3,
                       action_range=(1.0, 9.0), influence=0.65,
                       settings={"use_animated_influence": True,
                                 "blend_in": 2.0, "blend_out": 2.0},
                       extrapolation="NOTHING")
add_track(blended, "BlendRamp", [blend_clip])

fade = make_owner("NLA_BlendFade")
fade.location.x = 1.5
fade_clip = make_clip(fade, "BlendFade", [(1, 8.0), (9, 8.0)], start=3,
                      action_range=(1.0, 9.0), settings={"blend_in": 2.0, "blend_out": 2.0},
                      extrapolation="NOTHING")
add_track(fade, "FadeInOut", [fade_clip])

animated_influence = make_owner("NLA_AnimatedInfluence")
animated_influence.location.x = 1.5
influence_clip = make_clip(
    animated_influence, "AnimatedInfluence", [(1, 8.0), (9, 8.0)], start=3,
    action_range=(1.0, 9.0), settings={"use_animated_influence": True},
    extrapolation="NOTHING")
add_track(animated_influence, "AnimatedInfluence", [influence_clip])
influence_strip = influence_clip["created"]
for frame, value in ((3.0, 0.25), (7.0, 0.75)):
    influence_strip.influence = value
    influence_strip.keyframe_insert(data_path="influence", frame=frame)
influence_curve = influence_strip.fcurves.find("influence", index=0)
for point in influence_curve.keyframe_points:
    point.interpolation = "BEZIER"
    point.handle_left_type = "FREE"
    point.handle_right_type = "FREE"
influence_curve.keyframe_points[0].handle_left = (2.0, 0.25)
influence_curve.keyframe_points[0].handle_right = (4.0, -2.0)
influence_curve.keyframe_points[1].handle_left = (6.0, 4.0)
influence_curve.keyframe_points[1].handle_right = (8.0, 0.75)
influence_curve.update()

auto = make_owner("NLA_AutoBlend")
auto.location.x = -1.0
auto_a = make_clip(auto, "AutoBlendA", [(1, 2.0), (5, 2.0)], start=3,
                   action_range=(1.0, 5.0), extrapolation="NOTHING",
                   settings={"use_auto_blend": True})
auto_b = make_clip(auto, "AutoBlendB", [(1, 7.0), (5, 7.0)], start=5,
                   action_range=(1.0, 5.0), extrapolation="NOTHING",
                   settings={"use_auto_blend": True})
add_track(auto, "AutomaticBlendLower", [auto_a])
add_track(auto, "AutomaticBlendUpper", [auto_b])

auto_three = make_owner("NLA_AutoBlendThreeTrack")
auto_three.location.x = -1.0
for index, (start, value) in enumerate(((2, 2.0), (4, 6.0), (6, 10.0)), start=1):
    clip = make_clip(
        auto_three, f"AutoBlendThree{index}", [(1, value), (9, value)], start=start,
        action_range=(1.0, 9.0), extrapolation="NOTHING",
        settings={"use_auto_blend": True})
    add_track(auto_three, f"AutomaticBlend{index}", [clip])

# Constant strip-time curve samples two source times and wraps cyclically.
time_owner = make_owner("NLA_AnimatedStripTime")
time_action = make_clip(time_owner, "AnimatedTime", [(1, 0.0), (9, 8.0)], start=2,
                        action_range=(1.0, 9.0), extrapolation="NOTHING")
add_track(time_owner, "AnimatedTime", [time_action])
time_strip = time_action["created"]
time_strip.use_animated_time = True
time_strip.use_animated_time_cyclic = True
for frame, value in ((2.0, 1.0), (6.0, 5.0), (10.0, 9.0)):
    time_strip.strip_time = value
    time_strip.keyframe_insert(data_path="strip_time", frame=frame)
time_curve = time_strip.fcurves.find("strip_time", index=0)
for point in time_curve.keyframe_points:
    point.interpolation = "CONSTANT"
time_curve.update()

# The active Action has its own blend, influence, and extrapolation controls and is applied last.
active = make_owner("NLA_ActiveActionControls")
active.location = (2.0, 0.5, -0.25)
base_clip = make_clip(
    active, "ActiveBase", [(1, 2.0), (8, 2.0)], start=1,
    action_range=(1.0, 8.0),
    channels=[("location", 0, [(1, 2.0), (8, 2.0)]),
              ("scale", 1, [(1, 1.0), (8, 1.0)]),
              ("rotation_euler", 2, [(1, 0.25), (8, 0.25)])])
add_track(active, "ActiveBase", [base_clip])
action, slot = make_action(active, "ActiveAction", [
    ("location", 0, [(2, 6.0), (8, 6.0)]),
    ("scale", 1, [(2, 1.8), (8, 1.8)]),
    ("rotation_euler", 2, [(2, 0.8), (8, 0.8)]),
])
active.animation_data.action = action
active.animation_data.action_slot = slot
active.animation_data.action_blend_type = "ADD"
active.animation_data.action_influence = 0.4
active.animation_data.action_extrapolation = "NOTHING"
pushdown = make_owner("NLA_PushDownActiveAction")
pushdown.location = (0.0, 0.5, -0.25)
pushdown_action, pushdown_slot = make_action(pushdown, "PushDownActiveAction", [
    ("location", 0, [(3, 2.0), (11, 10.0)]),
])
pushdown_animation = pushdown.animation_data_create()
pushdown_animation.action = pushdown_action
pushdown_animation.action_slot = pushdown_slot
pushdown_animation.action_blend_type = "ADD"
pushdown_animation.action_influence = 0.5
pushdown_animation.action_extrapolation = "NOTHING"
if pushdown_animation.nla_tracks:
    raise RuntimeError("push-down input must not already have NLA tracks")


active_action_names = []
for label, extrapolation in (("Hold", "HOLD"),
                             ("HoldForward", "HOLD_FORWARD"),
                             ("Nothing", "NOTHING")):
    owner = make_owner("NLA_ActiveAction" + label)
    owner.location = (2.5, -0.5, 0.75)
    base_clip = make_clip(
        owner, label + "Base", [(1, 2.5), (12, 2.5)], start=1,
        action_range=(1.0, 12.0), extrapolation="HOLD")
    add_track(owner, label + "Base", [base_clip])
    action, slot = make_action(owner, label + "Action", [
        ("location", 0, [(5, 4.0), (8, 8.0)]),
        ("rotation_euler", 2, [(5, 0.2), (8, 0.8)]),
    ])
    action_fcurves = action.layers[0].strips[0].channelbag(slot).fcurves
    for curve in action_fcurves:
        curve.extrapolation = "LINEAR"
    if label == "Nothing":
        action_fcurves.find("location", index=0).modifiers.new("CYCLES")
    animation = owner.animation_data_create()
    animation.action = action
    animation.action_slot = slot
    animation.action_blend_type = "REPLACE"
    animation.action_influence = 1.0
    animation.action_extrapolation = extrapolation
    active_action_names.append(owner.name)
static_euler = make_owner("NLA_StaticEulerMatrixBasis")
static_euler.rotation_mode = "XYZ"
static_euler.rotation_euler = tuple(math.radians(value) for value in (30.0, 20.0, 10.0))



# One pose Action verifies action slots and evaluated pose-bone matrices through the public CLI.
rig = make_owner("NLA_PoseTransforms", "ARMATURE")
pose_action, pose_slot = make_action(rig, "PoseTransforms", [
    ('pose.bones["OracleBone"].location', 0, [(1, 0.0), (8, 1.0)]),
    ('pose.bones["OracleBone"].rotation_euler', 2, [(1, 0.0), (8, 0.6)]),
    ('pose.bones["OracleBone"].scale', 1, [(1, 1.0), (8, 1.5)]),
])
pose_clip = {"name": "PoseTransforms", "start": 2, "action": (pose_action, pose_slot),
             "action_range": (1.0, 8.0), "settings": {
                 "blend_type": "REPLACE", "influence": 1.0,
                 "extrapolation": "NOTHING"}}
add_track(rig, "PoseTransforms", [pose_clip])



def matrix_values(matrix):
    return [float(matrix[row][column]) for column in range(4) for row in range(4)]


frames = [1, 3, 4, 5, 6, 7, 9, 11, 13, 16, 20, 24]
object_names = [
    timing.name, *blend_names, quaternion.name, quaternion_replace.name, euler.name, ordered.name, muted.name,
    strip_muted.name, muted_domain.name, solo.name, solo_action.name, solo_domain.name, *hold_names,
    gapped.name, blended.name, fade.name, animated_influence.name, auto.name,
    auto_three.name, time_owner.name, gap_hold.name, active.name, pushdown.name,
    *active_action_names, rig.name, static_euler.name,
]
expected = {"blender_version": bpy.app.version_string, "frames": {}, "objects": object_names}
for frame in frames:
    scene.frame_set(frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    values = {}
    for name in object_names:
        evaluated = bpy.data.objects[name].evaluated_get(depsgraph)
        record = {"matrix_world": matrix_values(evaluated.matrix_world)}
        if name == static_euler.name:
            record["matrix_basis"] = matrix_values(bpy.data.objects[name].matrix_basis)
        if evaluated.type == "ARMATURE":
            record["bones"] = {
                bone.name: matrix_values(evaluated.matrix_world @ evaluated.pose.bones[bone.name].matrix)
                for bone in evaluated.data.bones
            }
        values[name] = record
    expected["frames"][str(frame)] = values

# Frame evaluation flushes active-Action reset channels to RNA; restore authored inputs before save.
solo_domain.location.x = 2.0
pushdown.location = (0.0, 0.5, -0.25)
bpy.context.preferences.filepaths.save_version = 0
# Preserve the active Action as Potter's input; only the oracle below runs push-down.
bpy.ops.wm.save_as_mainfile(filepath=str(root / "nla_advanced.blend"))

for selected in bpy.context.selected_objects:
    selected.select_set(False)
pushdown.select_set(True)
bpy.context.view_layer.objects.active = pushdown
window = bpy.context.window
area = next(area for area in window.screen.areas if area.type == "VIEW_3D")
area.type = "DOPESHEET_EDITOR"
area.spaces.active.mode = "ACTION"
region = next(region for region in area.regions if region.type == "WINDOW")
with bpy.context.temp_override(window=window, area=area, region=region,
                               object=pushdown, active_object=pushdown,
                               selected_objects=[pushdown], selected_editable_objects=[pushdown]):
    result = bpy.ops.action.push_down()
if "FINISHED" not in result:
    raise RuntimeError(f"Blender did not push down the active Action: {result}")
if pushdown_animation.action is not None:
    raise RuntimeError(
        f"Blender push-down kept active Action {pushdown_animation.action.name!r}; "
        f"track_count={len(pushdown_animation.nla_tracks)}"
    )
if pushdown_animation.action_slot is not None:
    raise RuntimeError("Blender push-down kept the active Action slot assigned")
if (len(pushdown_animation.nla_tracks) != 1
        or len(pushdown_animation.nla_tracks[0].strips) != 1):
    raise RuntimeError("Blender did not create one push-down track and strip")
pushdown_track = pushdown_animation.nla_tracks[0]
pushdown_strip = pushdown_track.strips[0]
influence_curve = pushdown_strip.fcurves.find("influence", index=0)
if (not pushdown_strip.use_animated_influence or influence_curve is None
        or len(influence_curve.keyframe_points) != 1
        or influence_curve.keyframe_points[0].co.y != 0.5):
    raise RuntimeError("Blender push-down did not create the constant influence key")
expected["push_down"] = {
    "action": None if pushdown_animation.action is None else pushdown_animation.action.name,
    "action_slot": (None if pushdown_animation.action_slot is None
                    else pushdown_animation.action_slot.identifier),
    "action_blend_type": pushdown_animation.action_blend_type.lower(),
    "action_influence": float(pushdown_animation.action_influence),
    "action_extrapolation": pushdown_animation.action_extrapolation.lower(),
    "strip": {
        "blend_type": pushdown_strip.blend_type.lower(),
        "influence": float(pushdown_strip.influence),
        "extrapolation": pushdown_strip.extrapolation.lower(),
        "use_animated_influence": pushdown_strip.use_animated_influence,
        "frame_start": float(pushdown_strip.frame_start),
        "frame_end": float(pushdown_strip.frame_end),
        "action_frame_start": float(pushdown_strip.action_frame_start),
        "action_frame_end": float(pushdown_strip.action_frame_end),
        "scale": float(pushdown_strip.scale),
        "repeat": float(pushdown_strip.repeat),
        "influence_curve": {
            "keyframes": [
                {"frame": float(point.co.x), "value": float(point.co.y),
                 "interpolation": point.interpolation.lower()}
                for point in influence_curve.keyframe_points
            ],
        },
    },
}
for frame in frames:
    scene.frame_set(frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    evaluated = pushdown.evaluated_get(depsgraph)
    expected["frames"][str(frame)][pushdown.name] = {
        "matrix_world": matrix_values(evaluated.matrix_world),
    }

with (root / "blender_expected.json").open("w", encoding="utf-8") as output:
    json.dump(expected, output, indent=2, sort_keys=True)

# Blender exposes non-CLIP strip creation through the NLA editor operators.
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
negative = make_owner("NLA_UnsupportedTransition")
track = negative.animation_data_create().nla_tracks.new()
track.name = "UnsupportedTransitionTrack"
for name, start, value in (("BeforeTransition", 1, 1.0), ("AfterTransition", 7, 4.0)):
    action, slot = make_action(negative, name, [("location", 0, [(1, value), (4, value)])])
    strip = track.strips.new(name, start, action)
    strip.action_slot = slot
    strip.select = True
negative.select_set(True)
bpy.context.view_layer.objects.active = negative
window = bpy.context.window
area = next(area for area in window.screen.areas if area.type == "VIEW_3D")
area.type = "NLA_EDITOR"
region = next(region for region in area.regions if region.type == "WINDOW")
with bpy.context.temp_override(window=window, area=area, region=region,
                               object=negative, active_object=negative,
                               selected_objects=[negative], selected_editable_objects=[negative]):
    result = bpy.ops.nla.transition_add()
if "FINISHED" not in result or not any(strip.type == "TRANSITION" for strip in track.strips):
    raise RuntimeError("Blender did not create the unsupported NLA transition fixture")
bpy.context.preferences.filepaths.save_version = 0
bpy.ops.wm.save_as_mainfile(filepath=str(root / "unsupported_transition.blend"))

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
tweak = make_owner("NLA_UnsupportedTweakMode")
tweak.animation_data_create()
action, slot = make_action(tweak, "TweakAction", [("location", 0, [(1, 0.0), (8, 1.0)])])
tweak.animation_data.action = action
tweak.animation_data.action_slot = slot
track = tweak.animation_data.nla_tracks.new()
track.name = "TweakTrack"
strip = track.strips.new("TweakStrip", 1, action)
strip.action_slot = slot
strip.select = True
tweak.animation_data.use_tweak_mode = True
if not tweak.animation_data.use_tweak_mode:
    raise RuntimeError("Blender did not enable unsupported NLA tweak mode")
bpy.context.preferences.filepaths.save_version = 0
bpy.ops.wm.save_as_mainfile(filepath=str(root / "unsupported_tweak_mode.blend"))