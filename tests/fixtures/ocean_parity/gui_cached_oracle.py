"""Run in a Blender 5.2.x GUI Python Console to verify cached Ocean evaluation.

This deliberately uses the asynchronous GUI bake job. Do not replace it with background EXR
creation: `is_cached` is set by the window-manager job's end callback.

Run in a fresh, separate Blender 5.2.x GUI instance through its Python Console with:
    exec(compile(open('/Users/admin/apps/potter-pr18k/tests/fixtures/ocean_parity/gui_cached_oracle.py').read(), 'gui_cached_oracle.py', 'exec'))

The script preserves the starting scene's frame, selection, and active object. It deletes only
the temporary Ocean object and mesh it creates; it never resets the startup scene or removes cache
files/reports.
It writes only the exact cache/report targets below, refuses existing non-empty targets, and
records cached evaluation before/after an ordinary time update plus cache invalidation on a
zero/nonzero choppiness crossing.
"""

import bpy
import hashlib
import json
import os
import time

CACHE_DIR = "/tmp/potter-pr18k-ocean-gui-cache"
REPORT_PATH = "/tmp/potter-pr18k-ocean-gui-cache.json"
BAKE_START = 1
BAKE_END = 2
CHANNELS = ("disp", "foam", "spray", "spray_inverse", "normal")
TIMEOUT_SECONDS = 300.0

if bpy.app.version[:2] != (5, 2):
    raise RuntimeError("Ocean cache GUI oracle requires Blender 5.2.x")
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
mesh = bpy.data.meshes.new("OceanCacheOracleMesh")
mesh.from_pydata(
    [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)],
    [],
    [(0, 1, 2, 3)],
)
obj = bpy.data.objects.new("OceanCacheOracle", mesh)
original_scene.collection.objects.link(obj)
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
modifier = obj.modifiers.new("OceanCacheOracle", "OCEAN")
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

scene = bpy.context.scene
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
    if obj.name in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    if mesh.name in bpy.data.meshes and mesh.users == 0:
        bpy.data.meshes.remove(mesh)


def finish_report(report):
    try:
        with open(REPORT_PATH, "x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=True)
            stream.write("\n")
        print("OCEAN_CACHE_GUI_ORACLE " + REPORT_PATH)
    finally:
        restore_context()


def poll_bake_job():
    if time.monotonic() - started > TIMEOUT_SECONDS:
        finish_report({
            "status": "timeout",
            "is_cached": bool(modifier.is_cached),
            "bake_operator_result": sorted(bake_result),
            "cache_directory": CACHE_DIR,
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
    old_time = modifier.time
    modifier.time = old_time + 1.0
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
    finish_report({
        "status": "complete" if time_update_kept_cache and time_update_preserved_output and chop_change_cleared_cache else "mismatch",
        "blender_version": bpy.app.version_string,
        "bake_operator_result": sorted(bake_result),
        "is_cached_after_job_completion": True,
        "cache_directory": CACHE_DIR,
        "cache_files": cache_files,
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
