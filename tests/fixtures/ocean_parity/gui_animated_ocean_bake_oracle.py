"""Run in a fresh, separate Blender 5.2.x GUI Python Console to verify animated Ocean bake time.

This assigns one real Ocean Modifier time F-Curve through a shared Action slot to two Ocean objects,
then runs the asynchronous GUI bake job for one object. It reports the modifier's RNA
`path_from_id("time")`, the assigned Action slot, per-frame `time` values for both objects, cached
frame outputs, EXR checksums, and cache behavior after removing the Action and changing static time.
It never resets the startup scene or removes any pre-existing object, file, or cache directory.

Run in the Blender GUI Python Console with:
    exec(compile(open('/Users/admin/apps/potter-pr18k/tests/fixtures/ocean_parity/gui_animated_ocean_bake_oracle.py').read(), 'gui_animated_ocean_bake_oracle.py', 'exec'))

The script refuses to overwrite existing non-empty output targets. It writes only:
    /tmp/potter-pr18k-ocean-animated-gui-cache/
    /tmp/potter-pr18k-ocean-animated-gui-cache.json
"""

import bpy
import hashlib
import json
import os
import time

CACHE_DIR = "/tmp/potter-pr18k-ocean-animated-gui-cache"
REPORT_PATH = "/tmp/potter-pr18k-ocean-animated-gui-cache.json"
BAKE_START = 1
BAKE_END = 2
CHANNELS = ("disp", "foam", "spray", "spray_inverse", "normal")
TIMEOUT_SECONDS = 300.0

if bpy.app.version[:2] != (5, 2):
    raise RuntimeError("animated Ocean bake oracle requires Blender 5.2.x")
if os.path.exists(REPORT_PATH):
    raise FileExistsError("refusing to overwrite existing oracle report: " + REPORT_PATH)
if os.path.exists(CACHE_DIR) and os.listdir(CACHE_DIR):
    raise FileExistsError("refusing to overwrite non-empty Ocean cache directory: " + CACHE_DIR)
os.makedirs(CACHE_DIR, exist_ok=True)

original_scene = bpy.context.scene
original_frame = original_scene.frame_current
original_view_layer = bpy.context.view_layer
original_active = original_view_layer.objects.active
original_selected = tuple(bpy.context.selected_objects)
mesh = bpy.data.meshes.new("OceanAnimatedCacheOracleMesh")
mesh.from_pydata(
    [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)],
    [],
    [(0, 1, 2, 3)],
)
obj = bpy.data.objects.new("OceanAnimatedCacheOracle", mesh)
original_scene.collection.objects.link(obj)
original_view_layer.objects.active = obj
obj.select_set(True)
modifier = obj.modifiers.new("OceanAnimatedCacheOracle", "OCEAN")
modifier.geometry_mode = "GENERATE"
modifier.resolution = 3
modifier.viewport_resolution = 2
modifier.spatial_size = 12
modifier.size = 1.0
modifier.repeat_x = 1
modifier.repeat_y = 1
modifier.time = 0.5
modifier.wave_scale = 1.0
modifier.choppiness = 1.0
modifier.random_seed = 7
modifier.spectrum = "PHILLIPS"
modifier.use_foam = True
modifier.use_spray = True
modifier.use_normals = True
modifier.invert_spray = False
modifier.foam_layer_name = "OceanFoam"
modifier.spray_layer_name = "OceanSpray"
modifier.foam_coverage = 0.0
modifier.bake_foam_fade = 0.5
modifier.frame_start = BAKE_START
modifier.frame_end = BAKE_END
modifier.filepath = CACHE_DIR + os.sep

mesh_b = bpy.data.meshes.new("OceanAnimatedCacheOracleMeshB")
mesh_b.from_pydata(
    [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)],
    [],
    [(0, 1, 2, 3)],
)
obj_b = bpy.data.objects.new("OceanAnimatedCacheOracleB", mesh_b)
original_scene.collection.objects.link(obj_b)
modifier_b = obj_b.modifiers.new("OceanAnimatedCacheOracle", "OCEAN")
modifier_b.geometry_mode = "GENERATE"
modifier_b.resolution = 3
modifier_b.viewport_resolution = 2
modifier_b.time = 0.5

time_path = modifier.path_from_id("time")
modifier.time = 0.5
inserted_start = modifier.keyframe_insert(data_path="time", frame=BAKE_START)
modifier.time = 1.5
inserted_end = modifier.keyframe_insert(data_path="time", frame=BAKE_END)
action = obj.animation_data.action if obj.animation_data else None
animation_data = obj.animation_data
active_slot = getattr(animation_data, "action_slot", None) if animation_data else None

obj_b.animation_data_create()
obj_b.animation_data.action = action
if active_slot is not None:
    obj_b.animation_data.action_slot = active_slot
time_path_b = modifier_b.path_from_id("time")
shared_slot_assignment = {
    "same_action": obj_b.animation_data.action is action,
    "same_action_slot": (
        active_slot is not None
        and obj_b.animation_data.action_slot.as_pointer() == active_slot.as_pointer()
    ),
    "object_a": obj.name_full,
    "object_b": obj_b.name_full,
    "time_path_a": time_path,
    "time_path_b": time_path_b,
    "modifier_instances_are_distinct": (
        modifier.as_pointer() != modifier_b.as_pointer()
    ),
}
action_slot = None
if active_slot is not None:
    action_slot = {
        "identifier": getattr(active_slot, "identifier", None),
        "name": getattr(active_slot, "name_display", getattr(active_slot, "name", None)),
        "handle": getattr(active_slot, "handle", None),
        "target_id_type": getattr(active_slot, "target_id_type", None),
    }
curve_paths = []
if action is not None:
    for layer in getattr(action, "layers", ()):
        for strip in getattr(layer, "strips", ()):
            if active_slot is None or not hasattr(strip, "channelbag"):
                continue
            try:
                channelbag = strip.channelbag(active_slot)
            except (TypeError, RuntimeError):
                continue
            if channelbag is not None:
                curve_paths.extend(
                    {"data_path": curve.data_path, "array_index": curve.array_index}
                    for curve in channelbag.fcurves
                )

scene = original_scene
scene.frame_set(BAKE_START)
frame_time_values = {}
shared_object_time_values = {}
for frame in range(BAKE_START, BAKE_END + 1):
    scene.frame_set(frame)
    frame_time_values[str(frame)] = float(modifier.time)
    shared_object_time_values[str(frame)] = float(modifier_b.time)
scene.frame_set(BAKE_START)
bake_result = bpy.ops.object.ocean_bake(modifier=modifier.name, free=False)
started = time.monotonic()


def mesh_dump(frame):
    scene.frame_set(frame)
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    result_mesh = evaluated.to_mesh()
    try:
        colors = {}
        for attribute in result_mesh.color_attributes:
            colors[attribute.name] = {
                "domain": attribute.domain,
                "data_type": attribute.data_type,
                "values": [list(item.color) for item in attribute.data],
            }
        return {
            "frame": frame,
            "scene_frame": scene.frame_current,
            "modifier_time": float(modifier.time),
            "is_cached": bool(modifier.is_cached),
            "positions": [list(vertex.co) for vertex in result_mesh.vertices],
            "polygons": [list(poly.vertices) for poly in result_mesh.polygons],
            "color_attributes": colors,
        }
    finally:
        evaluated.to_mesh_clear()


def cache_files_complete(filenames):
    for filename in filenames:
        path = os.path.join(CACHE_DIR, filename)
        if not os.path.isfile(path) or os.path.getsize(path) <= 4:
            return False
        with open(path, "rb") as stream:
            if stream.read(4) != b"\x76\x2f\x31\x01":
                return False
    return True


def restore_context():
    original_scene.frame_set(original_frame)
    for selected in bpy.context.selected_objects:
        selected.select_set(False)
    for selected in original_selected:
        if selected.name in bpy.data.objects:
            selected.select_set(True)
    if original_active is not None and original_active.name in bpy.data.objects:
        original_view_layer.objects.active = original_active
    else:
        original_view_layer.objects.active = None
    if obj_b.name in bpy.data.objects:
        bpy.data.objects.remove(obj_b, do_unlink=True)
    if obj.name in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    if mesh_b.name in bpy.data.meshes and mesh_b.users == 0:
        bpy.data.meshes.remove(mesh_b)
    if mesh.name in bpy.data.meshes and mesh.users == 0:
        bpy.data.meshes.remove(mesh)
    if action is not None and action.name in bpy.data.actions and action.users == 0:
        bpy.data.actions.remove(action)


def finish_report(report):
    try:
        with open(REPORT_PATH, "x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=True)
            stream.write("\n")
        print("OCEAN_ANIMATED_CACHE_GUI_ORACLE " + REPORT_PATH)
    finally:
        restore_context()


def poll_bake_job():
    if time.monotonic() - started > TIMEOUT_SECONDS:
        finish_report({
            "status": "timeout",
            "is_cached": bool(modifier.is_cached),
            "bake_operator_result": sorted(bake_result),
            "cache_directory": CACHE_DIR,
            "time_path": time_path,
            "action_slot": action_slot,
            "curve_paths": curve_paths,
            "frame_time_values": frame_time_values,
            "shared_slot_assignment": shared_slot_assignment,
            "shared_object_time_values": shared_object_time_values,
            "keyframe_inserted": [bool(inserted_start), bool(inserted_end)],
        })
        return None
    if not modifier.is_cached:
        return 0.25

    expected_files = [
        "%s_%04d.exr" % (channel, frame)
        for frame in range(BAKE_START, BAKE_END + 1)
        for channel in CHANNELS
    ]
    if not cache_files_complete(expected_files):
        return 0.25

    frame_samples = [mesh_dump(frame) for frame in (0, 1, 2, 3)]
    baseline = frame_samples[1]
    baked_frame_one = frame_samples[1]
    baked_frame_two = frame_samples[2]
    shared_slot_assignment_valid = (
        shared_slot_assignment["same_action"]
        and shared_slot_assignment["same_action_slot"]
        and shared_slot_assignment["modifier_instances_are_distinct"]
        and shared_slot_assignment["time_path_a"] == shared_slot_assignment["time_path_b"]
        and shared_object_time_values[str(BAKE_START)]
        == frame_time_values[str(BAKE_START)]
        and shared_object_time_values[str(BAKE_END)]
        == frame_time_values[str(BAKE_END)]
        and shared_object_time_values[str(BAKE_START)]
        != shared_object_time_values[str(BAKE_END)]
    )
    animated_time_curve_observed = (
        inserted_start
        and inserted_end
        and action_slot is not None
        and time_path in [curve["data_path"] for curve in curve_paths]
        and frame_time_values[str(BAKE_START)] != frame_time_values[str(BAKE_END)]
        and baked_frame_one["modifier_time"] == frame_time_values[str(BAKE_START)]
        and baked_frame_two["modifier_time"] == frame_time_values[str(BAKE_END)]
        and shared_slot_assignment_valid
    )
    animated_cache_output_changed = (
        baked_frame_one["positions"] != baked_frame_two["positions"]
        or baked_frame_one["color_attributes"] != baked_frame_two["color_attributes"]
    )
    obj.animation_data_clear()
    old_time = modifier.time
    modifier.time = old_time + 7.75
    after_time_update = mesh_dump(1)
    time_update_kept_cache = bool(modifier.is_cached)
    time_update_preserved_output = (
        after_time_update["positions"] == baseline["positions"]
        and after_time_update["color_attributes"] == baseline["color_attributes"]
    )
    old_choppiness = modifier.choppiness
    modifier.choppiness = 0.0
    chop_change_cleared_cache = not bool(modifier.is_cached)
    after_choppiness_change = mesh_dump(1)
    cache_files = {}
    for name in expected_files:
        path = os.path.join(CACHE_DIR, name)
        with open(path, "rb") as stream:
            payload = stream.read()
        cache_files[name] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "exr_magic": payload[:4].hex(),
        }
    animation_time_valid = bool(
        animated_time_curve_observed and animated_cache_output_changed
    )
    finish_report({
        "status": "complete" if animation_time_valid and time_update_kept_cache and time_update_preserved_output and chop_change_cleared_cache else "mismatch",
        "blender_version": bpy.app.version_string,
        "bake_operator_result": sorted(bake_result),
        "is_cached_after_job_completion": True,
        "cache_directory": CACHE_DIR,
        "cache_files": cache_files,
        "time_path": time_path,
        "action_slot": action_slot,
        "curve_paths": curve_paths,
        "frame_time_values": frame_time_values,
        "shared_slot_assignment": shared_slot_assignment,
        "shared_slot_assignment_valid": bool(shared_slot_assignment_valid),
        "shared_object_time_values": shared_object_time_values,
        "keyframe_inserted": [bool(inserted_start), bool(inserted_end)],
        "animated_time_curve_observed": bool(animated_time_curve_observed),
        "animated_cache_output_changed": animated_cache_output_changed,
        "frame_samples": frame_samples,
        "time_update": {
            "from": old_time,
            "to": modifier.time,
            "is_cached_after_change": time_update_kept_cache,
            "output_unchanged": time_update_preserved_output,
        },
        "choppiness_crossing": {
            "from": old_choppiness,
            "to": modifier.choppiness,
            "is_cached_after_change": not chop_change_cleared_cache,
            "cleared_cache": chop_change_cleared_cache,
            "output": after_choppiness_change,
        },
    })
    return None


bpy.app.timers.register(poll_bake_job, first_interval=0.25)
