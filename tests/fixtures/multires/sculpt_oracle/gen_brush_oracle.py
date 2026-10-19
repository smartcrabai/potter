"""Blender 5.2.2 oracle for the Potter Draw-brush contract on the driven, shrinkwrapped multires body.

Derivation: every Potter input -> Blender input, with the Blender 5.2.2 source (tag v5.2.2) citation.
File refs: paint_stroke.cc = editors/sculpt_paint/paint_stroke.cc; sculpt.cc = editors/sculpt_paint/mesh/sculpt.cc;
brush.cc = blenkernel/intern/brush.cc; rna_sculpt_paint.cc / rna_brush.cc = makesrna/intern/.

Stroke element
  pressure 1.0      -> OperatorStrokeElement.pressure (rna_brush.cc:4066-4069); read to cache.pressure
                       (sculpt.cc:5852-5854); enters bstrength via brush_strength (sculpt.cc:2340-2342).
  location          -> element.location is OBJECT space (paint_stroke.cc:555 writes the object-space location).
                       sculpt.cc:5831-5837 copies it verbatim to cache.location on the first step, and
                       override_location defaults to False (sculpt.cc:6217-6218), so no raycast replaces it.
                       Potter's (2.5,0,0) is WORLD space; the body is translated (0,0,1), so the Blender value is
                       inverse(matrix_world) @ (2.5,0,0) = (2.5,0,-1). Passing world coordinates places the brush
                       one unit above the surface (probe_world_location_AREA_R1.5 below: measured zero change).
  mouse_event       -> region pixel of the world point in the top ortho view (paint_stroke.cc:1651-1652). It drives
                       test_start / over_mesh (sculpt.cc:5784-5787); it does not set the brush centre.
  mouse             -> output only (paint_stroke.cc:1671).
  size              -> output only (paint_stroke.cc:554).
  x_tilt, y_tilt    -> 0 (rna_brush.cc:4076-4079); tilt_strength_factor = 0 (rna_brush.cc:3482-3484) so
                       tilt_effective_normal_get = sculpt_normal_symm (sculpt.cc:2871-2873).
  time              -> 0.0 (rna_brush.cc:4087-4089). Not read by stroke_cache_update (sculpt.cc:5821-5856).
  is_start          -> True (rna_brush.cc:4092-4094). The exec path constructs PaintStroke (sculpt.cc:6152); its
                       ctor sets overlap_factor = 1.0 (paint_stroke.cc:911); the first dab is added at
                       paint_stroke.cc:1613-1615 for a SPACE stroke.
  pen_flip          -> not an element member in 5.2.2 (rna_brush.cc:4061-4095); operator property read at
                       paint_stroke.cc:873. Not applicable.

Pressure dependence (off, so the contract is pressure-independent)
  strength pressure -> Brush.use_pressure_strength = BRUSH_ALPHA_PRESSURE (rna_brush.cc:3682-3683), tested by
                       BKE_brush_use_alpha_pressure (brush.cc:1308-1310) at sculpt.cc:2340-2342. Off.
  size pressure     -> Brush.use_pressure_size = BRUSH_SIZE_PRESSURE (rna_brush.cc:3703-3704), tested at
                       sculpt.cc:5885-5894. Off.
  offset pressure   -> Brush.use_offset_pressure (rna_brush.cc:3689-3690), plane offset only. Off.
  unified           -> UnifiedPaintSettings exposes no pressure flags (rna_sculpt_paint.cc).

Radius (world units; locked size)
  use_locked_size 'SCENE' -> UNIFIED_PAINT_BRUSH_LOCK_SIZE (rna_sculpt_paint.cc:834-837) -> BKE_brush_use_locked_size
                       (brush.cc:1295-1301) -> object_space_radius_get returns
                       BKE_brush_unprojected_radius_get * scale_factor (sculpt.cc:120-131).
  unprojected_size  -> DIAMETER (rna_sculpt_paint.cc:894, PROP_DISTANCE_DIAMETER); radius = size / 2
                       (brush.cc:1335-1337). So unprojected_size = 2 * R_world (3.0 for R=1.5).
  unified size      -> UNIFIED_PAINT_SIZE (rna_sculpt_paint.cc:850-851), read by BKE_brush_unprojected_size_get
                       (brush.cc:1326-1333).
  cache.radius      -> initial_radius (sculpt.cc:5859-5860, 5892); radius_squared (sculpt.cc:5898).
  distance          -> 3D from location_symm to the vertex for SPHERE falloff (sculpt.cc:7451-7476), body-local
                       space, equal to world distance here (translation only).

Strength (scales with radius)
  root_alpha        -> unified strength via UNIFIED_PAINT_ALPHA (rna_sculpt_paint.cc:856-857); strength maps to
                       ups.alpha (rna_sculpt_paint.cc:903-904); BKE_brush_alpha_get (brush.cc:1377-1385).
  alpha             -> root_alpha^2 (sculpt.cc:2338-2339). Strength 0.5 -> 0.25. Potter does NOT square strength.
  bstrength (DRAW)  -> alpha * flip * pressure * overlap * feather (sculpt.cc:2357-2360); flip = 1 (no invert,
                       sculpt.cc:2316-2320); feather = 1 (no symmetry feather, sculpt.cc:1393-1394); overlap = 1
                       (paint_stroke.cc:696-697 returns 1 when Brush.use_space_attenuation is off,
                       rna_brush.cc:3781-3782; set off).
  cache.bstrength   -> sculpt.cc:3859.

Draw displacement
  offset            -> effective_normal * cache.radius * cache.scale * cache.bstrength (draw.cc:194-197).
                       RADIUS-SCALED. Potter's Draw rule is aligned to this in crates/potter-core/src/geom/sculpt.rs (Draw arm).
  translation       -> offset * factor (draw.cc:66, translations_from_offset_and_factors).
  factor            -> BRUSH_CURVE_LIN: 1 - d/R, zero at d >= R (brush.cc:1548-1558; sculpt.cc:7490-7500,
                       7571-7580). Preset string 'LIN' (rna_brush.cc:86). Potter Falloff::Linear = 1-t (sculpt.rs:1083).
  normal            -> sculpt_plane AREA (Blender default for Draw): calc_sculpt_normal -> calc_area_normal(...)
                       .value_or(0) (sculpt.cc:2704-2706, 2741-2751). The area normal averages CCG normals within
                       normal_radius = cache.radius * Brush.normal_radius_factor (sculpt.cc:1438-1442, 1701-1716).
                       Default factor 0.5 (read back, see brush_configured). For the FIRST dab the original-position
                       sample (sculpt.cc:1636-1646, Accumulate off) equals the current positions, so Accumulate on and
                       off give identical results (measured). Potter uses the per-vertex normal (sculpt.rs:127); on
                       this flat patch both are +Z (measured: Z plane equals AREA).
  masks             -> none configured (sculpt.cc:6905-6935).

Spacing (single dab)
  one element, is_start -> single first dab (paint_stroke.cc:1613-1615). With spacing attenuation off the overlap is
                       1 (paint_stroke.cc:696-697), so spacing does not scale the dab. use_smooth_stroke off
                       (first dab not deferred, paint_stroke.cc:1613).

Potter-only inputs
  seed 7            -> no Blender equivalent. Potter's Draw has no random term (the former deterministic
                       scale was removed; Blender's Draw has none, draw.cc:196-197). The seed is a no-op for Draw.
  symmetry off      -> tool_settings.sculpt.use_symmetry_x/y/z = False.
  dyntopo off       -> Object.use_dynamic_topology_sculpting must be False (asserted).
  level 1           -> MultiresModifier.sculpt_levels = 1 (asserted total_levels == 1).

Run (headful, NOT -b):
  BRUSH_ORACLE_OUT=/tmp/brush_oracle.json \
  /Applications/Blender.app/Contents/MacOS/Blender --factory-startup --python gen_brush_oracle.py
"""

import json
import math
import os
import traceback

import bpy
from bpy_extras.view3d_utils import location_3d_to_region_2d
from mathutils import Vector

OUT = os.environ.get("BRUSH_ORACLE_OUT", "/tmp/brush_oracle.json")
STROKE_POINT_WORLD = (2.5, 0.0, 0.0)
STRENGTH = 0.5
PRESSURE = 1.0
TIME_FIRST = 0.0
VIEW_DISTANCE = 4.0
CHANGE_EPS = 1.0e-7
CHECK_EPS = 1.0e-4
SETTLE_SECONDS = 0.3
POLL_TRIES = 30
WINDOW_TRIES = 300
DEFAULT_NORMAL_RADIUS_FACTOR = 0.5  # Blender default, read back on the first attempt
ATTEMPTS = [
    # Source-predicted validation: R=6 covers the whole 2x2 body from the contract point.
    # Predicted centre displacement (sculpt level) = R * alpha^2 = 6 * 0.25 = 1.5 along +Z.
    {"name": "validate_R6_AREA", "radius": 6.0, "plane": "AREA", "accumulate": False, "location": "object",
     "role": "validation"},
    # CONTRACT: Blender defaults (AREA plane, Accumulate off) with the object-space location.
    # Predicted centre displacement (sculpt level) = 1.5 * 0.25 = 0.375 along +Z.
    {"name": "contract_AREA_R1.5", "radius": 1.5, "plane": "AREA", "accumulate": False, "location": "object",
     "role": "contract"},
    # Evidence: Accumulate on (current-position sampling) must equal the contract for a single dab.
    {"name": "evidence_AREA_accum_on_R1.5", "radius": 1.5, "plane": "AREA", "accumulate": True, "location": "object",
     "role": "evidence"},
    # Evidence: Z plane (exact +Z, sculpt.cc:2713-2714) must equal the AREA displacement on this flat patch.
    {"name": "evidence_Z_R1.5", "radius": 1.5, "plane": "Z", "accumulate": False, "location": "object",
     "role": "evidence"},
    # Probe: the harness bug. World coordinates passed as object-space location; centre 1.0 above the surface.
    # Predicted zero: normal radius 0.75 < 1.0 distance to the nearest surface sample (sculpt.cc:1441-1442, 2706).
    {"name": "probe_world_location_AREA_R1.5", "radius": 1.5, "plane": "AREA", "accumulate": False,
     "location": "world", "role": "probe"},
]
_S = {"ticks": 0}
CONFIG_ERRORS = []


def write_and_quit(payload):
    try:
        with open(OUT, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print("ORACLE_JSON_WRITTEN", OUT, payload.get("status"), flush=True)
    finally:
        bpy.ops.wm.quit_blender()
    return None


def find_view3d():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != "VIEW_3D":
                continue
            for region in area.regions:
                if region.type == "WINDOW" and region.width > 1 and region.height > 1:
                    return window, area, region
    return None


def tick():
    _S["ticks"] += 1
    if "gen" not in _S:
        found = find_view3d()
        if found is None:
            if _S["ticks"] > WINDOW_TRIES:
                return write_and_quit(
                    {"status": "error", "error": f"no VIEW_3D WINDOW region after {WINDOW_TRIES} tries"}
                )
            return 0.2
        _S["gen"] = oracle_steps(*found)
    try:
        return next(_S["gen"])
    except StopIteration as stop:
        return write_and_quit(stop.value)
    except Exception:
        return write_and_quit({"status": "error", "traceback": traceback.format_exc()})


def view_context(view):
    window, area, region = view
    return bpy.context.temp_override(window=window, area=area, region=region)


def clear_scene():
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)
    for armature in list(bpy.data.armatures):
        bpy.data.armatures.remove(armature)


def build_scene():
    """Scene from multires_sculpt_uses_driven_armature_and_shrinkwrap_prefix_for_brush_base."""
    clear_scene()
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, 1.0))
    body = bpy.context.active_object
    body.name = "body"
    body.data.name = "body_mesh"
    if len(body.data.vertices) != 4:
        raise RuntimeError(f"body plane has {len(body.data.vertices)} vertices, expected 4")

    bpy.ops.mesh.primitive_plane_add(size=8.0, location=(0.0, 0.0, 0.0))
    surface = bpy.context.active_object
    surface.name = "surface"

    bpy.ops.object.armature_add(location=(0.0, 0.0, 0.0))
    arm = bpy.context.active_object
    arm.name = "arm"
    bpy.ops.object.mode_set(mode="EDIT")
    bone = arm.data.edit_bones[0]
    bone.name = "Root"
    bone.head = (0.0, 0.0, 0.0)
    bone.tail = (0.0, 1.0, 0.0)
    bone.roll = 0.0
    bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.empty_add(type="PLAIN_AXES", location=STROKE_POINT_WORLD)
    controller = bpy.context.active_object
    controller.name = "controller"

    group = body.vertex_groups.new(name="Root")
    group.add([0, 1, 2, 3], 1.0, "REPLACE")

    armature_mod = body.modifiers.new("armature_deform", "ARMATURE")
    armature_mod.object = arm
    armature_mod.use_vertex_groups = True

    driver_fcurve = arm.pose.bones["Root"].driver_add("location", 0)
    driver = driver_fcurve.driver
    driver.type = "SCRIPTED"
    driver.expression = "var"
    for variable in list(driver.variables):
        driver.variables.remove(variable)
    variable = driver.variables.new()
    variable.name = "var"
    variable.type = "TRANSFORMS"
    target = variable.targets[0]
    target.id = controller
    target.transform_type = "LOC_X"
    target.transform_space = "WORLD_SPACE"

    shrink_mod = body.modifiers.new("shrink", "SHRINKWRAP")
    shrink_mod.target = surface
    shrink_mod.wrap_method = "NEAREST_SURFACEPOINT"
    shrink_mod.wrap_mode = "ON_SURFACE"

    bpy.context.view_layer.update()
    body.select_set(True)
    bpy.context.view_layer.objects.active = body
    # multires_subdivide only subdivides an EXISTING multires modifier; it does not add one.
    multires = body.modifiers.new("Multires", "MULTIRES")
    bpy.ops.object.multires_subdivide(modifier=multires.name, mode="CATMULL_CLARK")
    multires.levels = 1
    multires.sculpt_levels = 1
    multires.render_levels = 1
    multires.use_sculpt_base_mesh = False
    if multires.total_levels != 1:
        raise RuntimeError(f"multires total_levels is {multires.total_levels}, expected 1")

    subsurf = body.modifiers.new("after_multires", "SUBSURF")
    subsurf.levels = 1

    stack = [(mod.name, mod.type) for mod in body.modifiers]
    expected = [
        ("armature_deform", "ARMATURE"),
        ("shrink", "SHRINKWRAP"),
        ("Multires", "MULTIRES"),
        ("after_multires", "SUBSURF"),
    ]
    if stack != expected:
        raise RuntimeError(f"modifier stack {stack} != expected {expected}")
    if body.use_dynamic_topology_sculpting:
        raise RuntimeError("dyntopo unexpectedly enabled on fresh body")
    return body, arm, surface, controller, multires


def to_json(value):
    if isinstance(value, (set, frozenset)):
        return sorted(str(item) for item in value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    if hasattr(value, "__len__"):
        return [to_json(item) for item in value]
    return str(value)


def set_prop(owner, name, value):
    try:
        setattr(owner, name, value)
    except Exception as exc:  # recorded in the output, never silent
        CONFIG_ERRORS.append({"prop": name, "value": to_json(value), "error": str(exc)})


def read_prop(owner, name):
    try:
        return to_json(getattr(owner, name))
    except Exception as exc:
        return f"<unreadable: {exc}>"


BRUSH_PROPS = [
    "name",
    "sculpt_plane",
    "falloff_shape",
    "curve_distance_falloff_preset",
    "use_frontface",
    "use_pressure_size",
    "use_pressure_strength",
    "use_offset_pressure",
    "use_original_normal",
    "use_space_attenuation",
    "use_smooth_stroke",
    "use_accumulate",
    "normal_radius_factor",
    "use_locked_size",
    "unprojected_size",
    "strength",
    "tilt_strength_factor",
    "stroke_method",
    "spacing",
    "jitter",
    "direction",
]
UNIFIED_PROPS = [
    "use_unified_size",
    "use_locked_size",
    "unprojected_size",
    "size",
    "use_unified_strength",
    "strength",
]


def snapshot(sculpt):
    brush = sculpt.brush
    unified = sculpt.unified_paint_settings
    return {
        "brush": {name: read_prop(brush, name) for name in BRUSH_PROPS},
        "unified": {name: read_prop(unified, name) for name in UNIFIED_PROPS},
        "use_symmetry": [
            read_prop(sculpt, "use_symmetry_x"),
            read_prop(sculpt, "use_symmetry_y"),
            read_prop(sculpt, "use_symmetry_z"),
        ],
    }


def poll_diagnostics(view):
    with view_context(view):
        ctx = bpy.context
        brush = ctx.tool_settings.sculpt.brush
        workspace_tool = ctx.workspace.tools.from_space_view3d_mode("SCULPT", create=True)
        return {
            "mode": ctx.object.mode if ctx.object else None,
            "area": ctx.area.type if ctx.area else None,
            "region": ctx.region.type if ctx.region else None,
            "active_object": ctx.object.name if ctx.object else None,
            "sculpt_brush": brush.name if brush else None,
            "workspace_tool": workspace_tool.idname,
            "mode_set_poll": bool(bpy.ops.object.mode_set.poll()),
            "brush_stroke_poll": bool(bpy.ops.sculpt.brush_stroke.poll()),
            "sculptmode_toggle_poll": bool(bpy.ops.sculpt.sculptmode_toggle.poll()),
        }


def configure_brush(scene, radius, plane, accumulate):
    """Every contract-relevant brush/unified setting is set explicitly (see derivation)."""
    sculpt = scene.tool_settings.sculpt
    brush = sculpt.brush
    if brush is None or brush.name != "Draw":
        raise RuntimeError(f"sculpt brush is {brush}, expected Draw")
    unified = sculpt.unified_paint_settings
    diameter = 2.0 * radius  # unprojected_size is a DIAMETER (brush.cc:1335-1337)

    set_prop(unified, "use_unified_size", True)
    set_prop(unified, "use_locked_size", "SCENE")
    set_prop(unified, "unprojected_size", diameter)
    set_prop(unified, "use_unified_strength", True)
    set_prop(unified, "strength", STRENGTH)

    set_prop(brush, "use_locked_size", "SCENE")
    set_prop(brush, "unprojected_size", diameter)
    set_prop(brush, "strength", STRENGTH)
    set_prop(brush, "curve_distance_falloff_preset", "LIN")
    set_prop(brush, "falloff_shape", "SPHERE")
    set_prop(brush, "sculpt_plane", plane)
    set_prop(brush, "use_frontface", False)
    set_prop(brush, "use_pressure_size", False)
    set_prop(brush, "use_pressure_strength", False)
    set_prop(brush, "use_offset_pressure", False)
    set_prop(brush, "use_original_normal", False)
    set_prop(brush, "use_space_attenuation", False)
    set_prop(brush, "use_smooth_stroke", False)
    set_prop(brush, "use_accumulate", accumulate)
    set_prop(brush, "normal_radius_factor", DEFAULT_NORMAL_RADIUS_FACTOR)
    set_prop(brush, "tilt_strength_factor", 0.0)
    set_prop(brush, "jitter", 0.0)

    set_prop(sculpt, "use_symmetry_x", False)
    set_prop(sculpt, "use_symmetry_y", False)
    set_prop(sculpt, "use_symmetry_z", False)
    return brush, unified


def evaluate_local(body):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = body.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        return [[float(c) for c in vertex.co] for vertex in mesh.vertices]
    finally:
        evaluated.to_mesh_clear()


def stroke_elements(mouse, location):
    return [
        {
            "name": "oracle",
            "is_start": True,
            "location": [float(c) for c in location],
            "mouse": [float(mouse.x), float(mouse.y)],
            "mouse_event": [float(mouse.x), float(mouse.y)],
            "size": 0.0,
            "pressure": PRESSURE,
            "time": TIME_FIRST,
            "x_tilt": 0.0,
            "y_tilt": 0.0,
        }
    ]


def attempt_steps(view, mouse, spec):
    """Fresh scene, one single-element stroke. Generator: yields seconds between timer ticks."""
    radius = spec["radius"]
    body, arm, surface, controller, multires = build_scene()
    scene = bpy.context.scene
    bpy.context.view_layer.objects.active = body
    body.select_set(True)
    before_local = evaluate_local(body)
    object_location = body.matrix_world.inverted() @ Vector(STROKE_POINT_WORLD)
    element_location = STROKE_POINT_WORLD if spec["location"] == "world" else tuple(object_location)
    bpy.ops.object.mode_set(mode="SCULPT")
    sculpt = scene.tool_settings.sculpt
    CONFIG_ERRORS.clear()
    brush, unified = configure_brush(scene, radius, spec["plane"], spec["accumulate"])
    configured = snapshot(sculpt)
    config_errors = list(CONFIG_ERRORS)

    ready = False
    poll_tries = 0
    for poll_tries in range(1, POLL_TRIES + 1):
        with view_context(view):
            ready = bool(bpy.ops.sculpt.brush_stroke.poll())
        if ready:
            break
        yield SETTLE_SECONDS
    if not ready:
        raise RuntimeError(
            f"sculpt.brush_stroke poll never passed after {POLL_TRIES} settles; diag={poll_diagnostics(view)}"
        )

    elements = stroke_elements(mouse, element_location)
    with view_context(view):
        bpy.ops.ed.undo_push(message=f"oracle before stroke ({spec['name']})")
        result = bpy.ops.sculpt.brush_stroke(stroke=elements, mode="NORMAL")
    mode_at_stroke_end = body.mode
    with view_context(view):
        bpy.ops.object.mode_set(mode="OBJECT")
    after_local = evaluate_local(body)

    center_local = [float(c) for c in object_location]
    dists = [math.dist(p, center_local) for p in before_local]
    deltas = [[b[i] - a[i] for i in range(3)] for a, b in zip(before_local, after_local)]
    changed = [max(abs(d) for d in delta) > CHANGE_EPS for delta in deltas]
    nearest = min(range(len(dists)), key=lambda i: dists[i])
    changed_outside = [i for i, flag in enumerate(changed) if flag and dists[i] >= radius]
    unchanged_inside = [i for i, flag in enumerate(changed) if not flag and dists[i] < radius]
    return {
        "name": spec["name"],
        "role": spec["role"],
        "radius_world": radius,
        "sculpt_plane": spec["plane"],
        "accumulate": spec["accumulate"],
        "location_mode": spec["location"],
        "element_location_sent": [float(c) for c in element_location],
        "object_space_location_expected": center_local,
        "brush_configured": configured,
        "config_errors": config_errors,
        "poll_settles": poll_tries - 1,
        "stroke_operator_result": sorted(result),
        "mode_at_stroke_end": mode_at_stroke_end,
        "stroke_elements": elements,
        "predicted_centre_displacement_sculpt_level": radius * STRENGTH**2,
        "vertex_count": len(after_local),
        "changed_vertices": sum(changed),
        "max_abs_delta": max(max(abs(d) for d in delta) for delta in deltas),
        "nearest_vertex": {
            "index": nearest,
            "distance_local": dists[nearest],
            "before_local": before_local[nearest],
            "after_local": after_local[nearest],
            "delta_local": deltas[nearest],
        },
        "changed_with_distance_ge_radius": changed_outside,
        "unchanged_with_distance_lt_radius": unchanged_inside,
        "mean_before_local": [sum(p[i] for p in before_local) / len(before_local) for i in range(3)],
        "mean_after_local": [sum(p[i] for p in after_local) / len(after_local) for i in range(3)],
        "before_local": before_local,
        "after_local": after_local,
        "distance_local": dists,
        "changed": changed,
    }


def oracle_steps(window, area, region):
    view = (window, area, region)
    rv3d = area.spaces.active.region_3d
    with view_context(view):
        rv3d.view_perspective = "ORTHO"
        rv3d.view_rotation = (1.0, 0.0, 0.0, 0.0)  # identity quaternion = top view
        rv3d.view_location = Vector(STROKE_POINT_WORLD)
        rv3d.view_distance = VIEW_DISTANCE
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
    yield SETTLE_SECONDS

    mouse = location_3d_to_region_2d(region, rv3d, Vector(STROKE_POINT_WORLD))
    if mouse is None:
        raise RuntimeError("stroke point does not project into the region")

    attempts = []
    for spec in ATTEMPTS:
        attempt = yield from attempt_steps(view, mouse, spec)
        attempts.append(attempt)

    by_name = {a["name"]: a for a in attempts}
    contract = by_name["contract_AREA_R1.5"]
    validation = by_name["validate_R6_AREA"]
    checks_ok = (
        abs(contract["mean_before_local"][0] - 2.5) <= CHECK_EPS
        and abs(contract["mean_before_local"][2] + 1.0) <= CHECK_EPS
    )
    if not checks_ok:
        raise RuntimeError(f"driver/shrinkwrap check failed: before mean {contract['mean_before_local']}")
    status = (
        "ok"
        if contract["changed_vertices"] > 0 and validation["changed_vertices"] > 0
        else "no_geometry_change"
    )

    summary = [
        {
            k: a[k]
            for k in (
                "name", "role", "radius_world", "sculpt_plane", "accumulate", "location_mode",
                "element_location_sent", "changed_vertices", "max_abs_delta",
                "predicted_centre_displacement_sculpt_level", "changed_with_distance_ge_radius",
                "unchanged_with_distance_lt_radius", "nearest_vertex", "stroke_operator_result",
                "poll_settles",
            )
        }
        for a in attempts
    ]
    return {
        "status": status,
        "blender_version": bpy.app.version_string,
        "generator": "gen_brush_oracle.py",
        "region_size_px": [region.width, region.height],
        "mouse_region_px": [float(mouse.x), float(mouse.y)],
        "view": {
            "perspective": "ORTHO",
            "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "location_world": list(STROKE_POINT_WORLD),
            "distance": VIEW_DISTANCE,
        },
        "contract_attempt": "contract_AREA_R1.5",
        "contract_point_world": list(STROKE_POINT_WORLD),
        "contract_point_object_space": contract["object_space_location_expected"],
        "contract_radius_world": 1.5,
        "contract_strength": STRENGTH,
        "contract_pressure": PRESSURE,
        "contract_sculpt_plane": "AREA",
        "contract_accumulate": False,
        "summary": summary,
        "attempts": attempts,
    }


bpy.app.timers.register(tick, first_interval=0.5)
