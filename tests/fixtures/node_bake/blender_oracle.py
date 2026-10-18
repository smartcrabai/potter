"""Blender 5.2.2 oracle for Geometry Nodes Bake semantics.

Run headless:
    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --python blender_oracle.py

Writes expected.json next to this file. Scratch data lives under /tmp/pr18n/oracle/.
Reopen checks run in fresh Blender child processes so that nothing is cached in memory.
"""

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import bmesh
import bpy

HERE = Path(__file__).resolve().parent
EXPECTED = HERE / "expected.json"
SCRATCH = Path("/tmp/pr18n/oracle")
WORK = SCRATCH / "work"
OUT = SCRATCH / "out"

FRAMES_CORE = [0, 1, 2, 2.5, 3, 4, 5, 6]
FRAMES_SUB = [0, 1, 2, 2.25, 2.5, 2.75, 3, 4, 5, 6]
FRAMES_BAKED = [1, 2, 3, 4, 5]
TOL = 1e-5


# ---------------------------------------------------------------- filesystem


def fresh_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def walk(root):
    if not root.exists():
        return None
    entries = []
    for dirpath, dirnames, filenames in os.walk(root):
        for name in filenames:
            entries.append(str(Path(dirpath, name).relative_to(root)))
        for name in dirnames:
            entries.append(str(Path(dirpath, name).relative_to(root)) + "/")
    return sorted(entries)


def meta_example(root):
    if not root.exists():
        return None
    metas = sorted(p for p in root.rglob("*.json") if "meta" in p.parts)
    if not metas:
        return None
    text = metas[0].read_text()
    record = {"path": str(metas[0].relative_to(root)), "head_300": text[:300]}
    try:
        record["top_level_keys"] = sorted(json.loads(text).keys())
    except ValueError as exc:
        record["parse_error"] = str(exc)
    return record


# ---------------------------------------------------------------- node graph


def find_socket(sockets, name, socket_type=None):
    for socket in sockets:
        if socket.enabled and socket.name == name and (socket_type is None or socket.type == socket_type):
            return socket
    raise KeyError(f"socket {name!r} ({socket_type}) missing from {[(s.name, s.type) for s in sockets]}")


def build_group(variant):
    """variant: 'geo' (Transform by Shift, then Bake), 'float' (Shift baked as a FLOAT item that
    drives Transform after the bake), 'topo' (Switch between Cube and IcoSphere on Shift > 1.25,
    then Transform and Bake)."""
    ng = bpy.data.node_groups.new("BakeOracle", "GeometryNodeTree")
    ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket(name="Shift", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 0.0
    ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    nodes, links = ng.nodes, ng.links
    group_in = nodes.new("NodeGroupInput")
    group_out = nodes.new("NodeGroupOutput")
    combine = nodes.new("ShaderNodeCombineXYZ")
    transform = nodes.new("GeometryNodeTransform")
    bake = nodes.new("GeometryNodeBake")
    bake.bake_items.new("GEOMETRY", "Geometry")
    if variant == "float":
        bake.bake_items.new("FLOAT", "Shift")
    bake_geo_in = find_socket(bake.inputs, "Geometry", "GEOMETRY")
    bake_geo_out = find_socket(bake.outputs, "Geometry", "GEOMETRY")
    geo_in = group_in.outputs["Geometry"]
    shift_in = group_in.outputs["Shift"]

    if variant == "float":
        links.new(geo_in, bake_geo_in)
        links.new(shift_in, find_socket(bake.inputs, "Shift", "VALUE"))
        links.new(bake_geo_out, transform.inputs["Geometry"])
        links.new(find_socket(bake.outputs, "Shift", "VALUE"), combine.inputs["X"])
        links.new(transform.outputs["Geometry"], group_out.inputs["Geometry"])
    else:
        links.new(shift_in, combine.inputs["X"])
        if variant == "topo":
            compare = nodes.new("ShaderNodeMath")
            compare.operation = "GREATER_THAN"
            compare.inputs[1].default_value = 1.25
            switch = nodes.new("GeometryNodeSwitch")
            switch.input_type = "GEOMETRY"
            cube = nodes.new("GeometryNodeMeshCube")
            ico = nodes.new("GeometryNodeMeshIcoSphere")
            ico.inputs["Subdivisions"].default_value = 2
            links.new(shift_in, compare.inputs[0])
            links.new(compare.outputs[0], find_socket(switch.inputs, "Switch"))
            links.new(cube.outputs["Mesh"], find_socket(switch.inputs, "False"))
            links.new(ico.outputs["Mesh"], find_socket(switch.inputs, "True"))
            links.new(switch.outputs[0], transform.inputs["Geometry"])
        elif variant == "geo":
            links.new(geo_in, transform.inputs["Geometry"])
        else:
            raise ValueError(variant)
        links.new(transform.outputs["Geometry"], bake_geo_in)
        links.new(bake_geo_out, group_out.inputs["Geometry"])
    links.new(combine.outputs[0], transform.inputs["Translation"])
    return ng


def shift_identifier(ng):
    for item in ng.interface.items_tree:
        if item.item_type == "SOCKET" and item.name == "Shift" and item.in_out == "INPUT":
            return item.identifier
    raise KeyError("Shift input missing")


# ---------------------------------------------------------------- scene setup


def input_value_path(mod, ident):
    return f'modifiers["{mod.name}"].properties.inputs["{ident}"]["value"]'


def get_input(mod, ident):
    return mod.properties.inputs[ident]["value"]


def set_input(mod, ident, value):
    mod.properties.inputs[ident]["value"] = value
    mod.id_data.update_tag()


def set_keyframes_linear(ob):
    action = ob.animation_data.action if ob.animation_data else None
    if action is None:
        return
    for layer in action.layers:
        for strip in layer.strips:
            for channelbag in strip.channelbags:
                for fcurve in channelbag.fcurves:
                    for point in fcurve.keyframe_points:
                        point.interpolation = "LINEAR"

def setup(variant="geo"):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    ng = build_group(variant)
    bpy.ops.mesh.primitive_cube_add()
    ob = bpy.context.active_object
    ob.name = "Cube"
    mod = ob.modifiers.new("Nodes", "NODES")
    mod.node_group = ng
    bpy.context.view_layer.update()
    ident = shift_identifier(ng)

    scene.frame_start = 1
    scene.frame_end = 5
    bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
    for frame in FRAMES_BAKED:
        set_input(mod, ident, 0.5 * frame)
        ob.keyframe_insert(data_path=input_value_path(mod, ident), frame=frame)
    set_keyframes_linear(ob)
    return ob, mod, ident


def the_bake(mod):
    return mod.bakes[0]


def bake_snapshot(mod):
    bake = the_bake(mod)
    return {
        "bake_count": len(mod.bakes),
        "bake_id": bake.bake_id,
        "node": bake.node.name,
        "bake_mode": bake.bake_mode,
        "bake_target": bake.bake_target,
        "frame_start": bake.frame_start,
        "frame_end": bake.frame_end,
        "use_custom_simulation_frame_range": bake.use_custom_simulation_frame_range,
        "use_custom_path": bake.use_custom_path,
        "directory": bake.directory,
        "data_blocks": len(bake.data_blocks),
        "modifier_bake_target": mod.bake_target,
        "modifier_bake_directory": mod.bake_directory,
    }


def run_bake(op, ob, mod):
    bake = the_bake(mod)
    with bpy.context.temp_override(object=ob, active_object=ob):
        result = getattr(bpy.ops.object, op)(
            session_uid=ob.session_uid, modifier_name=mod.name, bake_id=bake.bake_id
        )
    if result != {"FINISHED"}:
        raise RuntimeError(f"{op} returned {result}")


def replace_mesh_with_icosphere(ob):
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=2, radius=1.0)
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()


def save(path):
    bpy.ops.wm.save_as_mainfile(filepath=str(path), check_existing=False)


# ---------------------------------------------------------------- sampling


def sample(ob, mod, ident, t, phase):
    scene = bpy.context.scene
    frame = math.floor(t)
    sub = round(t - frame, 6)
    scene.frame_set(frame, subframe=sub)
    ob.update_tag()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = ob.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    points = [tuple(v.co) for v in mesh.vertices]
    evaluated.to_mesh_clear()
    xs = [p[0] for p in points]
    return {
        "phase": phase,
        "frame": frame,
        "subframe": sub,
        "shift_input": get_input(mod, ident),
        "vertex_count": len(points),
        "min_x": min(xs) if xs else None,
        "max_x": max(xs) if xs else None,
        "mean_x": sum(xs) / len(xs) if xs else None,
        "positions": [[round(c, 6) for c in p] for p in points],
    }


def key(s):
    return s["frame"] + s["subframe"]


def pick(samples, t, phase):
    for s in samples:
        if s["phase"] == phase and abs(key(s) - t) < 1e-9:
            return s
    raise KeyError((t, phase))


def positions_equal(a, b):
    if a["vertex_count"] != b["vertex_count"]:
        return False
    return all(abs(x - y) <= TOL for pa, pb in zip(a["positions"], b["positions"]) for x, y in zip(pa, pb))



def passthrough_matches(s, base_min, base_max):
    shift = s["shift_input"]
    return abs(s["min_x"] - (base_min + shift)) <= TOL and abs(s["max_x"] - (base_max + shift)) <= TOL


def mix_report(a, b, mid, frac):
    report = {
        "between_frames": [a["frame"], b["frame"]],
        "at": key(mid),
        "vertex_counts_a_mid_b": [a["vertex_count"], mid["vertex_count"], b["vertex_count"]],
        "equals_frame_a": positions_equal(mid, a),
        "equals_frame_b": positions_equal(mid, b),
        "max_abs_err_vs_per_vertex_lerp": None,
    }
    if a["vertex_count"] == b["vertex_count"] == mid["vertex_count"]:
        err = 0.0
        for pm, pa, pb in zip(mid["positions"], a["positions"], b["positions"]):
            for m, x, y in zip(pm, pa, pb):
                err = max(err, abs(m - ((1 - frac) * x + frac * y)))
        report["max_abs_err_vs_per_vertex_lerp"] = round(err, 6)
    return report


# ---------------------------------------------------------------- child processes


def run_child(blend, phase, frames, tag):
    out_json = WORK / f"child_{tag}.json"
    cmd = [
        bpy.app.binary_path, "-b", "--factory-startup", "--python", str(Path(__file__).resolve()),
        "--", "child", str(blend), str(out_json), phase, ",".join(str(f) for f in frames),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(f"--- child {tag} stdout ---")
    print(proc.stdout)
    print(f"--- child {tag} stderr ---")
    print(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"child {tag} failed with {proc.returncode}")
    return json.loads(out_json.read_text())


def child_main(args):
    blend, out_json, phase, frames_csv = args
    frames = [float(f) for f in frames_csv.split(",")]
    bpy.ops.wm.open_mainfile(filepath=blend)
    ob = bpy.data.objects["Cube"]
    mod = ob.modifiers["Nodes"]
    ident = shift_identifier(mod.node_group)
    samples = [sample(ob, mod, ident, t, phase) for t in frames]
    result = {
        "samples": samples,
        "bake": bake_snapshot(mod),
        "walk_blend_dir": walk(Path(blend).parent),
    }
    Path(out_json).write_text(json.dumps(result))


# ---------------------------------------------------------------- scenarios


def scenario(sid, title):
    def wrap(fn):
        fn.sid = sid
        fn.title = title
        return fn
    return wrap


@scenario("S1_unbaked", "No bake node state: passthrough of the input geometry")
def s1_unbaked():
    ob, mod, ident = setup()
    obs = {"initial_bake": bake_snapshot(mod)}
    samples = [sample(ob, mod, ident, t, "unbaked") for t in FRAMES_SUB]
    obs["passthrough_matches_shift_on_cube"] = all(passthrough_matches(s, -1.0, 1.0) for s in samples)
    return {
        "samples": samples,
        "observations": obs,
        "files": [],
        "notes": [
            "Cube base vertices are at x=+-1; passthrough means min_x=-1+Shift, max_x=1+Shift.",
            "Shift is keyed 0.5*f on frames 1..5 with LINEAR interpolation; frames 0 and 6 use constant extrapolation.",
        ],
    }


@scenario("S2_animation_packed_input_change", "ANIMATION bake, PACKED, input change and delete")
def s2_animation_input_change():
    ob, mod, ident = setup()
    the_bake(mod).bake_mode = "ANIMATION"
    obs = {"before_bake": bake_snapshot(mod)}
    run_bake("geometry_node_bake_single", ob, mod)
    obs["after_bake"] = bake_snapshot(mod)

    samples = [sample(ob, mod, ident, t, "after_bake") for t in FRAMES_SUB]

    replace_mesh_with_icosphere(ob)
    ob.animation_data_clear()
    set_input(mod, ident, 100.0)
    samples += [sample(ob, mod, ident, t, "after_input_change") for t in FRAMES_BAKED]

    run_bake("geometry_node_bake_delete_single", ob, mod)
    obs["after_delete"] = bake_snapshot(mod)
    samples += [sample(ob, mod, ident, t, "after_delete") for t in FRAMES_BAKED]

    obs["after_input_change_equals_after_bake_frames_1_to_5"] = {
        str(t): positions_equal(pick(samples, t, "after_bake"), pick(samples, t, "after_input_change"))
        for t in FRAMES_BAKED
    }
    input_min_x = min(v.co.x for v in ob.data.vertices)
    input_max_x = max(v.co.x for v in ob.data.vertices)
    obs["after_delete_is_input_passthrough_shift100"] = {
        str(s["frame"]): passthrough_matches(s, input_min_x, input_max_x) and s["vertex_count"] == len(ob.data.vertices)
        for s in samples if s["phase"] == "after_delete"
    }
    obs["clamp_frame0_equals_frame1"] = positions_equal(pick(samples, 0, "after_bake"), pick(samples, 1, "after_bake"))
    obs["clamp_frame6_equals_frame5"] = positions_equal(pick(samples, 6, "after_bake"), pick(samples, 5, "after_bake"))
    obs["mix_2_to_3"] = {
        str(t): mix_report(
            pick(samples, 2, "after_bake"),
            pick(samples, 3, "after_bake"),
            pick(samples, t, "after_bake"),
            t - 2,
        )
        for t in (2.25, 2.5, 2.75)
    }
    return {
        "samples": samples,
        "observations": obs,
        "files": [],
        "notes": [
            "Input change: mesh replaced with a 42-vertex icosphere, Shift keys removed, Shift=100.",
            "Delete: geometry_node_bake_delete_single, then the same frames are sampled.",
        ],
    }


@scenario("S3_still", "STILL bake at frame 3 held across frame changes")
def s3_still():
    ob, mod, ident = setup()
    obs = {"default_bake": bake_snapshot(mod)}
    bpy.context.scene.frame_set(3)
    run_bake("geometry_node_bake_single", ob, mod)
    obs["after_still_bake"] = bake_snapshot(mod)
    samples = [sample(ob, mod, ident, t, "after_still_bake_at_3") for t in [1, 2, 2.5, 3, 4, 5]]
    bpy.context.scene.frame_end = 20
    samples += [sample(ob, mod, ident, t, "after_scene_range_change") for t in [1, 2.5, 3, 5, 6, 10, 20]]
    obs["after_scene_range_change"] = bake_snapshot(mod)
    return {
        "samples": samples,
        "observations": obs,
        "files": [],
        "notes": [
            "Default bake_mode on a freshly created bake is STILL; the bake was made while the scene was on frame 3.",
            "Frame changes are sampled before and after changing scene.frame_end from 5 to 20.",
        ],
    }


@scenario("S4_packed_save_reopen", "PACKED bake saved to out/packed.blend and reopened in a fresh process")
def s4_packed_save_reopen():
    ob, mod, ident = setup()
    the_bake(mod).bake_mode = "ANIMATION"
    run_bake("geometry_node_bake_single", ob, mod)
    obs = {"before_save": bake_snapshot(mod)}
    samples = [sample(ob, mod, ident, t, "before_save") for t in FRAMES_CORE]

    before = walk(OUT)
    blend = OUT / "packed.blend"
    save(blend)
    after = walk(OUT)
    obs["out_dir_before_save"] = before
    obs["out_dir_after_save"] = after
    obs["extra_directories_after_save"] = [e for e in after if e.endswith("/")]

    reopened = run_child(blend, "after_reopen", FRAMES_CORE, "s4")
    samples += reopened["samples"]
    obs["after_reopen_bake"] = reopened["bake"]
    obs["after_reopen_equals_before_save"] = {
        str(t): positions_equal(pick(samples, t, "before_save"), pick(samples, t, "after_reopen"))
        for t in FRAMES_CORE
    }
    return {
        "samples": samples,
        "observations": obs,
        "files": [str(blend)] + walk(OUT),
        "notes": ["Reopened in a fresh Blender child process; no bake data was re-created."],
    }


@scenario("S5_disk_relative", "DISK bake into //oracle_disk_bake: layout, reopen, and directory deletion")
def s5_disk_relative():
    work = fresh_dir(WORK / "s5")
    blend = work / "s5.blend"
    ob, mod, ident = setup()
    save(blend)
    mod.bake_target = "DISK"
    mod.bake_directory = "//oracle_disk_bake"
    the_bake(mod).bake_mode = "ANIMATION"
    run_bake("geometry_node_bake_single", ob, mod)
    obs = {"after_bake": bake_snapshot(mod)}
    samples = [sample(ob, mod, ident, t, "after_bake") for t in FRAMES_CORE]

    disk_root = work / "oracle_disk_bake"
    obs["disk_root_absolute"] = str(disk_root.resolve())
    obs["disk_tree"] = walk(disk_root)
    obs["meta_example"] = meta_example(disk_root)

    ob.animation_data_clear()
    set_input(mod, ident, 100.0)
    samples += [sample(ob, mod, ident, t, "after_input_change") for t in FRAMES_CORE]
    obs["after_input_change_equals_after_bake"] = {
        str(t): positions_equal(pick(samples, t, "after_bake"), pick(samples, t, "after_input_change"))
        for t in FRAMES_CORE
    }

    save(blend)
    reopened = run_child(blend, "after_reopen", FRAMES_CORE, "s5_reopen")
    samples += reopened["samples"]
    obs["after_reopen_bake"] = reopened["bake"]
    obs["after_reopen_equals_after_input_change"] = {
        str(t): positions_equal(pick(samples, t, "after_input_change"), pick(samples, t, "after_reopen"))
        for t in FRAMES_CORE
    }

    shutil.rmtree(disk_root)
    deleted = run_child(blend, "after_dir_deleted", FRAMES_CORE, "s5_deleted")
    samples += deleted["samples"]
    obs["after_dir_deleted_bake"] = deleted["bake"]
    obs["after_dir_deleted_equals_after_input_change"] = {
        str(t): positions_equal(pick(samples, t, "after_input_change"), pick(samples, t, "after_dir_deleted"))
        for t in FRAMES_CORE
    }
    return {
        "samples": samples,
        "observations": obs,
        "files": [str(blend)] + (obs["disk_tree"] or []),
        "notes": [
            "Relative directory resolved against the saved .blend location (work/s5/).",
            "Before save/reopen the animation was removed and Shift set to 100, distinguishing baked output from live input.",
        ],
    }


@scenario("S6_disk_default_directory", "DISK bake with empty bake_directory: default location and value")
def s6_disk_default_directory():
    work = fresh_dir(WORK / "s6")
    blend = work / "s6_default.blend"
    ob, mod, ident = setup()
    save(blend)
    mod.bake_target = "DISK"
    mod.bake_directory = ""
    the_bake(mod).bake_mode = "ANIMATION"
    run_bake("geometry_node_bake_single", ob, mod)
    obs = {"after_bake": bake_snapshot(mod)}
    samples = [sample(ob, mod, ident, t, "after_bake") for t in FRAMES_CORE]
    obs["walk_blend_dir_after_bake"] = walk(work)
    obs["meta_example"] = meta_example(work)
    obs["default_root_absolute"] = str((work / "blendcache_s6_default").resolve())
    obs["default_bake_absolute"] = str(
        (work / "blendcache_s6_default" / "Cube_Nodes" / str(the_bake(mod).bake_id)).resolve()
    )

    save(blend)
    reopened = run_child(blend, "after_reopen", FRAMES_CORE, "s6")
    samples += reopened["samples"]
    obs["after_reopen_bake"] = reopened["bake"]
    obs["walk_blend_dir_after_save"] = walk(work)
    return {
        "samples": samples,
        "observations": obs,
        "files": [str(blend)] + (obs["walk_blend_dir_after_save"] or []),
        "notes": ["Blend saved as s6_default.blend so the blend name is 's6_default'."],
    }


@scenario("S7_disk_directory_change", "DISK bake, bake_directory changed to an empty dir in-session and after reopen")
def s7_disk_directory_change():
    work = fresh_dir(WORK / "s7")
    blend = work / "s7.blend"
    ob, mod, ident = setup()
    save(blend)
    mod.bake_target = "DISK"
    mod.bake_directory = "//s7_disk"
    the_bake(mod).bake_mode = "ANIMATION"
    run_bake("geometry_node_bake_single", ob, mod)
    obs = {"after_bake": bake_snapshot(mod)}
    samples = [sample(ob, mod, ident, t, "after_bake") for t in FRAMES_CORE]

    (work / "empty_dir").mkdir()
    mod.bake_directory = "//empty_dir"
    ob.animation_data_clear()
    set_input(mod, ident, 100.0)
    obs["after_dir_change_snapshot"] = bake_snapshot(mod)
    samples += [sample(ob, mod, ident, t, "after_dir_change_same_session") for t in FRAMES_CORE]

    save(blend)
    reopened = run_child(blend, "after_reopen_empty_dir", FRAMES_CORE, "s7")
    samples += reopened["samples"]
    obs["after_reopen_bake"] = reopened["bake"]
    obs["s7_disk_tree"] = walk(work / "s7_disk")
    obs["empty_dir_tree"] = walk(work / "empty_dir")
    obs["same_session_equals_after_bake"] = {
        str(t): positions_equal(pick(samples, t, "after_bake"), pick(samples, t, "after_dir_change_same_session"))
        for t in FRAMES_CORE
    }
    obs["reopen_equals_after_bake"] = {
        str(t): positions_equal(pick(samples, t, "after_bake"), pick(samples, t, "after_reopen_empty_dir"))
        for t in FRAMES_CORE
    }
    return {
        "samples": samples,
        "observations": obs,
        "files": [str(blend)] + (obs["s7_disk_tree"] or []),
        "notes": [
            "//empty_dir was created empty before the change; //s7_disk holds the bake files.",
            "After the directory change, animation was removed and Shift set to 100 to distinguish cache output from live evaluation.",
        ],
    }


@scenario("S8_frame_range", "ANIMATION bake with custom simulation frame range 2..4")
def s8_frame_range():
    ob, mod, ident = setup()
    bake = the_bake(mod)
    bake.bake_mode = "ANIMATION"
    bake.use_custom_simulation_frame_range = True
    bake.frame_start = 2
    bake.frame_end = 4
    run_bake("geometry_node_bake_single", ob, mod)
    obs = {"after_bake": bake_snapshot(mod)}
    ts = [0, 1, 2, 2.5, 3, 3.5, 4, 5, 6]
    samples = [sample(ob, mod, ident, t, "after_bake") for t in ts]
    obs["clamp_frame1_equals_frame2"] = positions_equal(pick(samples, 1, "after_bake"), pick(samples, 2, "after_bake"))
    obs["clamp_frame5_equals_frame4"] = positions_equal(pick(samples, 5, "after_bake"), pick(samples, 4, "after_bake"))
    obs["mix_2_to_3"] = {
        str(t): mix_report(pick(samples, 2, "after_bake"), pick(samples, 3, "after_bake"),
                           pick(samples, t, "after_bake"), t - 2)
        for t in (2.5,)
    }
    obs["mix_3_to_4"] = {
        str(t): mix_report(pick(samples, 3, "after_bake"), pick(samples, 4, "after_bake"),
                           pick(samples, t, "after_bake"), t - 3)
        for t in (3.5,)
    }
    return {
        "samples": samples,
        "observations": obs,
        "files": [],
        "notes": ["Scene range stays 1..5; the bake uses its own custom range 2..4."],
    }


@scenario("S9_float_item", "Bake node with a FLOAT item feeding Transform after the bake")
def s9_float_item():
    ob, mod, ident = setup("float")
    bake = the_bake(mod)
    bake.bake_mode = "ANIMATION"
    obs = {"bake_items": [(i.name, i.socket_type) for i in bake.node.bake_items]}
    run_bake("geometry_node_bake_single", ob, mod)
    obs["after_bake"] = bake_snapshot(mod)
    samples = [sample(ob, mod, ident, t, "after_bake") for t in FRAMES_CORE]

    ob.animation_data_clear()
    set_input(mod, ident, 100.0)
    samples += [sample(ob, mod, ident, t, "after_input_change") for t in FRAMES_CORE]
    obs["after_input_change_equals_after_bake"] = {
        str(t): positions_equal(pick(samples, t, "after_bake"), pick(samples, t, "after_input_change"))
        for t in FRAMES_CORE
    }
    obs["baked_translation_x_at_frames"] = {
        str(s["frame"]): round(s["min_x"] + 1.0, 6)
        for s in samples if s["phase"] == "after_bake" and s["subframe"] == 0
    }
    return {
        "samples": samples,
        "observations": obs,
        "files": [],
        "notes": [
            "Geometry item and FLOAT item (named Shift) are both baked; Transform consumes the baked float.",
            "Shift input is set to 100 after bake; baked float should keep 0.5*f at baked frames.",
        ],
    }


@scenario("S10_topology_mix", "ANIMATION bake whose geometry topology changes between baked frames")
def s10_topology_mix():
    ob, mod, ident = setup("topo")
    the_bake(mod).bake_mode = "ANIMATION"
    run_bake("geometry_node_bake_single", ob, mod)
    obs = {"after_bake": bake_snapshot(mod)}
    ts = [1, 1.5, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 5, 6]
    samples = [sample(ob, mod, ident, t, "after_bake") for t in ts]
    obs["vertex_count_by_frame"] = {str(s["frame"]) + "+" + str(s["subframe"]): s["vertex_count"] for s in samples}
    obs["mix_2_to_3"] = {
        str(t): mix_report(pick(samples, 2, "after_bake"), pick(samples, 3, "after_bake"),
                           pick(samples, t, "after_bake"), t - 2)
        for t in (2.25, 2.5, 2.75)
    }
    return {
        "samples": samples,
        "observations": obs,
        "files": [],
        "notes": [
            "Shift > 1.25 selects a 42-vertex IcoSphere; otherwise an 8-vertex Cube. Frame 2 (Shift 1.0) is Cube, frame 3 (Shift 1.5) is IcoSphere.",
        ],
    }


SCENARIOS = [
    s1_unbaked,
    s2_animation_input_change,
    s3_still,
    s4_packed_save_reopen,
    s5_disk_relative,
    s6_disk_default_directory,
    s7_disk_directory_change,
    s8_frame_range,
    s9_float_item,
    s10_topology_mix,
]


def main():
    fresh_dir(OUT)
    fresh_dir(WORK)
    results = []
    for fn in SCENARIOS:
        print(f"=== {fn.sid}: {fn.title}")
        data = fn()
        results.append({"scenario": fn.sid, "title": fn.title, **data})
    doc = {
        "blender_version": bpy.app.version_string,
        "blender_build_hash": bpy.app.build_hash.decode("ascii"),
        "blender_binary": bpy.app.binary_path,
        "script": str(Path(__file__).resolve()),
        "scenarios": results,
    }
    EXPECTED.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"WROTE {EXPECTED}")


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if args and args[0] == "child":
        child_main(args[1:])
    else:
        main()
