"""PR18L modifier-stack oracle, executed inside Blender 5.2 (headless) by generate.py.

Usage:
    Blender -b --factory-startup --python blender_oracle.py -- <raw_output.json> <artifact_dir>

Writes raw observations to <raw_output.json>. generate.py turns them into expected.json.
Hash contract (mirrored by tests/modifier_stack_semantics.rs):
    each vertex position is formatted as f"{v:.5f}" (with "-0.00000" normalized to "0.00000"),
    the "x y z" strings are sorted bytewise, each followed by "\\n", and the SHA-256 hex digest
    of that UTF-8 payload is recorded.
"""

import hashlib
import json
import os
import shutil
import sys
import tempfile

import bpy
from mathutils import Vector

ARGS = sys.argv[sys.argv.index("--") + 1 :]
RAW_OUTPUT = os.path.realpath(ARGS[0])
ARTIFACT_DIR = os.path.realpath(ARGS[1])
WORK_DIR = tempfile.mkdtemp(prefix="pr18l_oracle_")
OBJECT_NAME = "cube"
EXPORT_COUNTER = 0

# Kind names are shared with the Rust test through expected.json.
KINDS = {
    "taper": {
        "blender_type": "SIMPLE_DEFORM",
        "potter_type": "simple_deform",
        "params": {"deform_method": "TAPER", "deform_axis": "Z", "factor": 0.5},
    },
    "smooth": {
        "blender_type": "SMOOTH",
        "potter_type": "smooth",
        "params": {"factor": 0.5},
    },
    "array": {
        "blender_type": "ARRAY",
        "potter_type": "array",
        "params": {
            "count": 3,
            "use_relative_offset": True,
            "relative_offset_displace": [1.5, 0.0, 0.0],
        },
    },
}


def create(name, kind, **flags):
    return {"action": "create", "name": name, "kind": kind, **flags}


def pin(name, value):
    return {"action": "pin", "name": name, "value": value}


def visibility(name, **flags):
    return {"action": "visibility", "name": name, **flags}


def move(name, to_index):
    return {"action": "move", "name": name, "to_index": to_index}


def delete(name):
    return {"action": "delete", "name": name}


CASES = [
    {
        "name": "S_unpinned",
        "group": "stack",
        "description": "taper then array, no pins: default append order",
        "steps": [create("taper", "taper"), create("array", "array")],
    },
    {
        "name": "S_pin_first",
        "group": "stack",
        "description": "taper pinned to last: moves past array",
        "steps": [create("taper", "taper"), create("array", "array"), pin("taper", True)],
    },
    {
        "name": "S_pin_then_unpin",
        "group": "stack",
        "description": "taper pinned then unpinned: unpin does not move it forward",
        "steps": [
            create("taper", "taper"),
            create("array", "array"),
            pin("taper", True),
            pin("taper", False),
        ],
    },
    {
        "name": "S_create_before_pinned",
        "group": "stack",
        "description": "array pinned, then taper created: new modifier inserts before trailing pinned",
        "steps": [create("array", "array"), pin("array", True), create("taper", "taper")],
    },
    {
        "name": "S_create_pinned",
        "group": "stack",
        "description": "array then taper, taper pinned afterwards",
        "steps": [create("array", "array"), create("taper", "taper"), pin("taper", True)],
    },
    {
        "name": "S_delete_middle",
        "group": "stack",
        "description": "taper, array, smooth; delete the middle array",
        "steps": [
            create("taper", "taper"),
            create("array", "array"),
            create("third", "smooth"),
            delete("array"),
        ],
    },
    {
        "name": "S_move_blocked",
        "group": "stack",
        "description": "array then pinned taper; moving array past pinned taper is refused",
        "steps": [
            create("array", "array"),
            create("taper", "taper"),
            pin("taper", True),
            move("array", 1),
        ],
    },
    {
        "name": "S_move_plain",
        "group": "stack",
        "description": "taper, array, third; move third to index 0",
        "steps": [
            create("taper", "taper"),
            create("array", "array"),
            create("third", "smooth"),
            move("third", 0),
        ],
    },
    {
        "name": "V_viewport_only",
        "group": "visibility",
        "description": "array visible in viewport, hidden for render",
        "steps": [create("taper", "taper"), create("array", "array", show_render=False)],
    },
    {
        "name": "V_render_only",
        "group": "visibility",
        "description": "array hidden in viewport, visible for render",
        "steps": [create("taper", "taper"), create("array", "array", show_viewport=False)],
    },
    {
        "name": "V_both_off",
        "group": "visibility",
        "description": "array hidden for both purposes",
        "steps": [
            create("taper", "taper"),
            create("array", "array", show_viewport=False, show_render=False),
        ],
    },
    {
        "name": "V_enabled_off_potter_equivalent",
        "group": "visibility",
        "description": (
            "array with both show flags false; Blender has no enabled flag, so this is the "
            "Blender form of Potter enabled=false. The Rust test also runs the enabled=false form."
        ),
        "steps": [
            create("taper", "taper"),
            create("array", "array", show_viewport=False, show_render=False),
        ],
        "potter_equivalent_variant": "enabled_false",
    },
    {
        "name": "V_pinned_with_hidden",
        "group": "visibility",
        "description": "taper pinned and hidden for render; array unpinned and visible",
        "steps": [
            create("taper", "taper"),
            create("array", "array"),
            pin("taper", True),
            visibility("taper", show_render=False),
        ],
    },
]


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0.0, 0.0, 0.0))
    ob = bpy.context.active_object
    ob.name = OBJECT_NAME
    ob.data.name = OBJECT_NAME
    bpy.context.view_layer.objects.active = ob
    return ob


def apply_visibility(modifier, step):
    for key in ("show_viewport", "show_render"):
        if key in step:
            setattr(modifier, key, step[key])


def apply_step(ob, step):
    action = step["action"]
    if action == "create":
        kind = KINDS[step["kind"]]
        modifier = ob.modifiers.new(step["name"], kind["blender_type"])
        for key, value in kind["params"].items():
            setattr(modifier, key, value)
        apply_visibility(modifier, step)
        return None
    if action == "pin":
        ob.modifiers[step["name"]].use_pin_to_last = step["value"]
        return None
    if action == "visibility":
        apply_visibility(ob.modifiers[step["name"]], step)
        return None
    if action == "move":
        result = bpy.ops.object.modifier_move_to_index(
            modifier=step["name"], index=step["to_index"]
        )
        return next(iter(result))
    if action == "delete":
        ob.modifiers.remove(ob.modifiers[step["name"]])
        return None
    raise ValueError(f"unknown step action: {action}")


def fixed5(value):
    text = f"{value:.5f}"
    return "0.00000" if text == "-0.00000" else text


def position_sha(positions):
    keys = sorted(" ".join(fixed5(c) for c in point) for point in positions)
    payload = "".join(key + "\n" for key in keys)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def depsgraph_mesh(ob):
    """Viewport depsgraph (Blender's default evaluated depsgraph); Blender stores float32 positions."""
    bpy.context.view_layer.update()
    evaluated = ob.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        positions = [tuple(vertex.co) for vertex in mesh.vertices]
        faces = len(mesh.polygons)
    finally:
        evaluated.to_mesh_clear()
    return {"vertices": len(positions), "faces": faces, "position_sha": position_sha(positions)}


def obj_mesh(eval_mode):
    """Evaluated mesh through wm.obj_export; DAG_EVAL_RENDER uses the render depsgraph.

    Positions are written with 6 decimals by the OBJ writer; see the notes in expected.json.
    """
    global EXPORT_COUNTER
    EXPORT_COUNTER += 1
    path = os.path.join(WORK_DIR, f"export_{EXPORT_COUNTER:03d}_{eval_mode}.obj")
    bpy.context.view_layer.update()
    bpy.ops.wm.obj_export(
        filepath=path,
        check_existing=False,
        export_selected_objects=False,
        export_eval_mode=eval_mode,
        apply_modifiers=True,
        apply_transform=False,
        global_scale=1.0,
        forward_axis="Y",
        up_axis="Z",
    )
    positions = []
    faces = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                positions.append(tuple(float(token) for token in line.split()[1:4]))
            elif line.startswith("f "):
                faces += 1
    return {"vertices": len(positions), "faces": faces, "position_sha": position_sha(positions)}


def measure(ob):
    viewport = depsgraph_mesh(ob)
    obj_viewport = obj_mesh("DAG_EVAL_VIEWPORT")
    if obj_viewport != viewport:
        raise RuntimeError(f"OBJ viewport export {obj_viewport} != depsgraph {viewport}")
    render = obj_mesh("DAG_EVAL_RENDER")
    return viewport, render


def run_case(case):
    ob = reset_scene()
    step_results = [apply_step(ob, step) for step in case["steps"]]
    viewport, render = measure(ob)
    return {
        "name": case["name"],
        "group": case["group"],
        "description": case["description"],
        "steps": case["steps"],
        "step_results": step_results,
        "stack": [modifier.name for modifier in ob.modifiers],
        "viewport_stack": [modifier.name for modifier in ob.modifiers if modifier.show_viewport],
        "render_stack": [modifier.name for modifier in ob.modifiers if modifier.show_render],
        "viewport": viewport,
        "render": render,
        "obj_viewport_matches_depsgraph": True,
        "potter_equivalent_variant": case.get("potter_equivalent_variant"),
    }


def apply_probe(modifier_name):
    ob = reset_scene()
    apply_step(ob, create("taper", "taper"))
    apply_step(ob, create("array", "array"))
    modifier = ob.modifiers[modifier_name]
    modifier.show_viewport = False
    base_positions = [tuple(vertex.co) for vertex in ob.data.vertices]
    base_count = len(base_positions)
    result = bpy.ops.object.modifier_apply(modifier=modifier_name, report=True)
    applied_positions = [tuple(vertex.co) for vertex in ob.data.vertices]
    operator_result = next(iter(result))
    return {
        "modifier": modifier_name,
        "show_viewport": False,
        "show_render": True,
        "operator_result": operator_result,
        "refused": operator_result != "FINISHED",
        "vertices_before": base_count,
        "vertices_after": len(applied_positions),
        "data_changed": applied_positions != base_positions,
        "stack_after": [entry.name for entry in ob.modifiers],
    }


def render_artifacts(directory):
    os.makedirs(directory, exist_ok=True)
    written = []
    for label, show_render in (("render_array_off", False), ("render_array_on", True)):
        ob = reset_scene()
        apply_step(ob, create("taper", "taper"))
        apply_step(ob, create("array", "array", show_render=show_render))
        scene = bpy.context.scene
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.display.shading.type = "SOLID"
        scene.render.resolution_x = 480
        scene.render.resolution_y = 360
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"
        target = Vector((3.0, 0.0, 0.0))
        camera_data = bpy.data.cameras.new("PR18L_Camera")
        camera = bpy.data.objects.new("PR18L_Camera", camera_data)
        scene.collection.objects.link(camera)
        camera.location = (10.0, -16.0, 9.0)
        camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
        scene.camera = camera
        path = os.path.join(directory, f"V_viewport_only_{label}.png")
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        if not os.path.isfile(path):
            raise RuntimeError(f"render did not write {path}")
        written.append(path)
    return written


def main():
    cases = [run_case(case) for case in CASES]
    by_name = {case["name"]: case for case in cases}

    stack_s = by_name["S_unpinned"]["viewport"]["position_sha"]
    checks = {
        "taper_array_order_changes_geometry": stack_s != by_name["S_pin_first"]["viewport"]["position_sha"],
        "S_pin_first_stack_is_array_taper": by_name["S_pin_first"]["stack"] == ["array", "taper"],
        "S_pin_then_unpin_stack_is_array_taper": by_name["S_pin_then_unpin"]["stack"] == ["array", "taper"],
        "S_create_before_pinned_stack": by_name["S_create_before_pinned"]["stack"] == ["taper", "array"],
        "S_create_pinned_stack": by_name["S_create_pinned"]["stack"] == ["array", "taper"],
        "S_delete_middle_stack": by_name["S_delete_middle"]["stack"] == ["taper", "third"],
        "S_move_blocked_refused": by_name["S_move_blocked"]["step_results"][-1] == "CANCELLED",
        "S_move_blocked_stack_unchanged": by_name["S_move_blocked"]["stack"] == ["array", "taper"],
        "S_move_plain_stack": by_name["S_move_plain"]["stack"] == ["third", "taper", "array"],
        "V_viewport_only_render_drops_array": by_name["V_viewport_only"]["render"]["vertices"]
        == by_name["V_viewport_only"]["viewport"]["vertices"] // 3,
        "V_render_only_viewport_drops_array": by_name["V_render_only"]["viewport"]["vertices"]
        == by_name["V_render_only"]["render"]["vertices"] // 3,
        "V_enabled_off_equals_both_off": by_name["V_enabled_off_potter_equivalent"]["viewport"]
        == by_name["V_both_off"]["viewport"]
        and by_name["V_enabled_off_potter_equivalent"]["render"] == by_name["V_both_off"]["render"],
        "V_pinned_with_hidden_render_stack": by_name["V_pinned_with_hidden"]["render_stack"] == ["array"],
        "V_pinned_with_hidden_viewport_stack": by_name["V_pinned_with_hidden"]["viewport_stack"]
        == ["array", "taper"],
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(f"semantic checks failed: {failed}")

    apply_results = {
        "array": apply_probe("array"),
        "taper": apply_probe("taper"),
    }
    artifacts = render_artifacts(ARTIFACT_DIR)

    raw = {
        "blender": {
            "version": bpy.app.version_string,
            "build_hash": bpy.app.build_hash.decode("ascii") if isinstance(bpy.app.build_hash, bytes) else str(bpy.app.build_hash),
        },
        "kinds": KINDS,
        "position_hash": {
            "algorithm": "sha256",
            "decimals": 5,
            "serialization": (
                "for each vertex: f'{x:.5f} {y:.5f} {z:.5f}' with '-0.00000' -> '0.00000'; "
                "sort the lines bytewise; payload = concatenation of line + '\\n'; "
                "hash = lowercase hex sha256(payload as UTF-8)"
            ),
        },
        "render_method": {
            "viewport": "bpy.context.evaluated_depsgraph_get() + Object.evaluated_get().to_mesh(); positions are Blender float32 values exposed as Python floats",
            "render": (
                "bpy.ops.wm.obj_export(export_eval_mode='DAG_EVAL_RENDER', apply_modifiers=True, "
                "forward_axis='Y', up_axis='Z', global_scale=1.0); counts from v/f lines; "
                "positions as written by the OBJ writer (6 decimals)"
            ),
        },
        "cases": {case["name"]: case for case in cases},
        "checks": checks,
        "apply_viewport_disabled": apply_results,
        "artifacts": artifacts,
        "notes": [
            "Python exposes only the viewport depsgraph (Depsgraph.mode is read-only); the render depsgraph is read through wm.obj_export with export_eval_mode='DAG_EVAL_RENDER'.",
            "Every case asserts that the OBJ DAG_EVAL_VIEWPORT export has the same counts and hash as the viewport depsgraph to_mesh() result.",
            "OBJ positions are written with 6 decimals and the hash rounds to 5. SMOOTH averages neighbour positions, so stack positions are not all decimal-exact; the render hash equals the viewport hash in every case where visibility does not differ (V_viewport_only, V_render_only and V_pinned_with_hidden differ by design). Irrational deformers (TWIST, BEND) were not used: float32 storage makes 5-decimal hashes of irrational positions unstable.",
            "The third modifier is SMOOTH (factor 0.5, deterministic) rather than MIRROR: Potter's Mirror diverges from Blender in merge (use_mirror_merge, PR-B), so MIRROR cannot be matched by these cases.",
            "Render-time handlers (bpy.app.handlers.render_pre) receive depsgraph=None and bpy.context.evaluated_depsgraph_get() is VIEWPORT there, so no in-process RENDER depsgraph was reachable; wm.obj_export is the render path.",
            "Blender 5.2 modifiers have no enabled flag; Potter enabled=false is represented in Blender by show_viewport=False and show_render=False. The Rust test also runs the enabled=false form for V_enabled_off_potter_equivalent.",
            "Blender's stack list is evaluation order: index 0 is evaluated first.",
            "modifier_apply does not refuse a viewport-disabled (or render-disabled) modifier in background mode; it applies it. The refusal message 'Modifier is disabled' was not printed.",
            "Move refusal is observed through the modifier_move_to_index operator result (CANCELLED); the stack is unchanged.",
            "Only modifiers.new() was probed for insertion order; the RequiresOriginalData/Hook insertion rule of BKE_modifiers_add_at_end_if_possible is not exercised by any case and is not encoded.",
            "Pin semantics use the use_pin_to_last RNA setter (rna_Modifier_use_pin_to_last_set), not the move operator.",
        ],
    }
    with open(RAW_OUTPUT, "w", encoding="utf-8") as output:
        json.dump(raw, output, indent=2, sort_keys=True)
        output.write("\n")


try:
    main()
finally:
    shutil.rmtree(WORK_DIR, ignore_errors=True)
