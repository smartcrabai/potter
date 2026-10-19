# SPDX-License-Identifier: GPL-2.0-or-later
#
# Generates the Blender 5.2.2 oracle fixtures for Potter's dab-sequential sculpt brushes
# (Clay Strips, Inflate, Layer). This script needs a display: it must run in a normal (non-background)
# Blender session so that the 3D viewport has a real window, region and redraw cycle:
#
#   /Applications/Blender.app/Contents/MacOS/Blender --factory-startup --python generate_oracle.py -- <output_dir>
#
# Background mode (`-b`) does not refresh the region's view matrices, so Clay Strips derives its strip
# direction from a stale view. The GUI run sets an explicit top orthographic view, redraws it, and then
# strokes under a `temp_override` that names the real window, area and region.
#
# Each case replays a DOTS stroke with explicit dab `location`s through `sculpt.brush_stroke`. Mouse
# positions are the screen projection of each location in that top view, so the stroke's screen-space
# grab delta equals the world-space delta between dabs.
#
# Blender facts this script relies on (Blender 5.2.2 source):
# - `unified_paint_settings.use_unified_size` and `use_unified_strength` must be off, or brush sizes and
#   strengths come from the unified settings instead of the brush.
# - `use_locked_size = 'SCENE'` makes the dab radius `unprojected_size / 2` (`brush.cc`
#   `BKE_brush_unprojected_radius_get`).
# - Mirror cases set `mesh.use_mirror_x`, which sets the sculpt symmetry (`mesh_symmetry_xyz_get`).
# - Vertex normals are the stroke-start normals for every dab of a stroke, not recomputed after each dab.
#
# Writes one `<case>.json` per case: base positions, faces, dabs and Blender's result positions.
# The script quits Blender when it finishes, or exits with code 1 after writing the traceback to
# `<output_dir>/error.txt`.

import json
import os
import sys
import traceback

import bpy
from bpy_extras import view3d_utils
from mathutils import Quaternion, Vector

ESSENTIALS = os.path.join(
    bpy.utils.system_resource("DATAFILES"),
    "assets",
    "brushes",
    "essentials_brushes-mesh_sculpt.blend",
)
BRUSH_NAMES = {
    "clay_strips": "Clay Strips",
    "inflate": "Inflate/Deflate",
    "layer": "Layer",
    "draw": "Draw",
    "draw_sharp": "Draw Sharp",
    "blob": "Blob",
    "nudge": "Nudge",
    "crease": "Crease Polish",
    "pinch": "Pinch/Magnify",
    "smooth": "Smooth",
    "flatten": "Flatten/Contrast",
    "fill": "Fill/Deepen",
    "scrape": "Scrape/Fill",
    "grab": "Grab",
    "snake_hook": "Snake Hook",
    "thumb": "Thumb",
    "rotate": "Twist",
    "elastic_deform": "Elastic Grab",
    "pose": "Pose",
    "boundary": "Boundary",
}
STRIP_DIRECTION = Vector((0.8, 0.6, 0.0)).normalized()


def sphere_dabs(count, radius, strength, pressures):
    # Points on and slightly above the unit sphere, spread over the visible cap.
    base = [
        (0.0, 0.0, 1.0),
        (0.3, 0.2, 0.9),
        (-0.25, 0.35, 0.85),
        (0.1, -0.4, 0.9),
        (0.5, -0.3, 0.8),
        (-0.1, -0.1, 1.0),
    ]
    dabs = []
    for index in range(count):
        dabs.append(
            {
                "position": base[index % len(base)],
                "radius": radius,
                "strength": strength,
                "pressure": pressures[index % len(pressures)],
            }
        )
    return dabs


def strip_dabs(count, step, radius, strength, pressures):
    # Straight stroke on the z = 0 plane along STRIP_DIRECTION.
    dabs = []
    for index in range(count):
        position = tuple(STRIP_DIRECTION * (index * step))
        dabs.append(
            {
                "position": position,
                "radius": radius,
                "strength": strength,
                "pressure": pressures[index % len(pressures)],
            }
        )
    return dabs


def view_context():
    window = bpy.context.window_manager.windows[0]
    area = next(a for a in window.screen.areas if a.type == "VIEW_3D")
    region = next(r for r in area.regions if r.type == "WINDOW")
    return window, area, region, area.spaces.active.region_3d


def set_top_ortho_view():
    window, area, region, rv3d = view_context()
    rv3d.view_perspective = "ORTHO"
    # Blender's TOP view is the identity view rotation (look down -Z, X right, Y up).
    rv3d.view_rotation = Quaternion((1.0, 0.0, 0.0, 0.0))
    rv3d.view_location = Vector((0.0, 0.0, 0.0))
    rv3d.view_distance = 10.0
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
    rv3d.update()
    origin = view3d_utils.location_3d_to_region_2d(region, rv3d, Vector((0.0, 0.0, 0.0)))
    x_axis = view3d_utils.location_3d_to_region_2d(region, rv3d, Vector((1.0, 0.0, 0.0)))
    y_axis = view3d_utils.location_3d_to_region_2d(region, rv3d, Vector((0.0, 1.0, 0.0)))
    # X must point right and Y up on screen, so the mouse deltas match world deltas.
    assert x_axis.x > origin.x and abs(x_axis.y - origin.y) < 1e-3, (origin, x_axis)
    assert y_axis.y > origin.y and abs(y_axis.x - origin.x) < 1e-3, (origin, y_axis)


def make_object(kind):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    if kind == "grid":
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=20, y_subdivisions=20, size=2.0)
    else:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=1.0)
    ob = bpy.context.active_object
    ob.location = (0.0, 0.0, 0.0)
    ob.rotation_euler = (0.0, 0.0, 0.0)
    ob.scale = (1.0, 1.0, 1.0)
    return ob


def run_case(name, brush_key, kind, dabs, strength, radius, mirror=False, mask=None, deform=None):
    window, area, region, rv3d = view_context()
    ob = make_object(kind)
    bpy.context.view_layer.objects.active = ob
    base = [list(v.co) for v in ob.data.vertices]
    faces = [list(p.vertices) for p in ob.data.polygons]
    ob.data.use_mirror_x = mirror
    if mask is not None:
        attribute = ob.data.attributes.new(".sculpt_mask", "FLOAT", "POINT")
        for vertex in ob.data.vertices:
            attribute.data[vertex.index].value = mask(vertex.co)
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="SCULPT")
        ts = bpy.context.tool_settings
        ts.sculpt.use_symmetry_x = False
        ts.sculpt.use_symmetry_y = False
        ts.sculpt.use_symmetry_z = False
        bpy.ops.ed.undo_push(message="pre-stroke")
        bpy.ops.brush.asset_activate(
            asset_library_type="ESSENTIALS",
            relative_asset_identifier="brushes/essentials_brushes-mesh_sculpt.blend/Brush/"
            + BRUSH_NAMES[brush_key],
        )
        brush = ts.sculpt.brush
        assert brush.name == BRUSH_NAMES[brush_key], brush.name
        ups = ts.sculpt.unified_paint_settings
        ups.use_unified_size = False
        ups.use_unified_strength = False
        brush.stroke_method = "DOTS"
        brush.use_locked_size = "SCENE"
        brush.unprojected_size = 2.0 * radius
        brush.strength = strength
        # Pose and Boundary keep their deform type in the brush, so every case sets it explicitly.
        if brush_key == "pose":
            brush.pose_deform_type = deform or "ROTATE_TWIST"
        if brush_key == "boundary":
            brush.boundary_deform_type = deform or "BEND"
        items = []
        for index, dab in enumerate(dabs):
            location = Vector(dab["position"])
            mouse = view3d_utils.location_3d_to_region_2d(region, rv3d, ob.matrix_world @ location)
            items.append(
                {
                    "name": "dab",
                    "is_start": index == 0,
                    "location": tuple(location),
                    "mouse": tuple(mouse),
                    "mouse_event": tuple(mouse),
                    "pressure": float(dab["pressure"]),
                    "size": 10,
                    "x_tilt": 0.0,
                    "y_tilt": 0.0,
                    "time": float(index),
                }
            )
        result = bpy.ops.sculpt.brush_stroke(stroke=items)
        assert result == {"FINISHED"}, result
        assert brush.name == BRUSH_NAMES[brush_key]
    bpy.ops.object.mode_set(mode="OBJECT")
    result_positions = [list(v.co) for v in ob.data.vertices]
    moved = sum(1 for a, b in zip(base, result_positions) if a != b)
    assert moved > 0, f"{name}: stroke changed no vertices"
    case = {
        "name": name,
        "brush": brush_key,
        "strength": strength,
        "radius": radius,
        "mirror": mirror,
        "deform": deform,
        "dabs": dabs,
        "base": base,
        "faces": faces,
        "result": result_positions,
    }
    if mask is not None:
        case["mask"] = [mask(Vector(co)) for co in base]
    return case


# Per-case sculpt masks, as a function of the vertex position. Written to Blender's `.sculpt_mask` attribute
# and recorded in the fixture as the per-vertex mask values.
MASKS = {"layer_sphere_masked": lambda co: 0.6 if co.x > 0.0 else 0.25}


def cases():
    return [
        ("inflate_sphere", "inflate", "sphere", sphere_dabs(4, 0.4, 1.0, [1.0, 1.0, 0.5, 1.0]), 1.0, 0.4, False),
        ("inflate_sphere_strength", "inflate", "sphere", sphere_dabs(3, 0.5, 0.7, [0.6, 1.0, 1.0]), 0.7, 0.5, False),
        ("inflate_sphere_mirror", "inflate", "sphere", sphere_dabs(3, 0.4, 1.0, [1.0]), 1.0, 0.4, True),
        ("layer_sphere_clamp", "layer", "sphere", sphere_dabs(12, 0.4, 1.0, [1.0]), 1.0, 0.4, False),
        ("layer_sphere_mirror", "layer", "sphere", sphere_dabs(4, 0.45, 0.8, [1.0, 0.5, 0.8, 1.0]), 0.8, 0.45, True),
        ("layer_sphere", "layer", "sphere", sphere_dabs(5, 0.45, 0.8, [1.0, 0.5, 1.0, 0.8, 1.0]), 0.8, 0.45, False),
        ("layer_sphere_masked", "layer", "sphere", sphere_dabs(5, 0.45, 0.8, [1.0, 0.5, 1.0, 0.8, 1.0]), 0.8, 0.45, False),
        ("clay_strips_grid", "clay_strips", "grid", strip_dabs(4, 0.08, 0.5, 1.0, [1.0]), 1.0, 0.5, False),
        ("clay_strips_grid_mirror", "clay_strips", "grid", strip_dabs(4, 0.08, 0.5, 1.0, [1.0]), 1.0, 0.5, True),
        (
            "clay_strips_grid_pressure",
            "clay_strips",
            "grid",
            strip_dabs(4, 0.1, 0.4, 0.6, [1.0, 0.5, 0.8, 1.0]),
            0.6,
            0.4,
            False,
        ),
        ("draw_sphere", "draw", "sphere", sphere_dabs(4, 0.4, 0.5, [1.0, 1.0, 0.5, 1.0]), 0.5, 0.4, False),
        ("draw_sphere_mirror", "draw", "sphere", sphere_dabs(3, 0.4, 0.5, [1.0]), 0.5, 0.4, True),
        ("draw_sharp_sphere", "draw_sharp", "sphere", sphere_dabs(4, 0.4, 0.5, [1.0, 1.0, 0.5, 1.0]), 0.5, 0.4, False),
        ("blob_sphere", "blob", "sphere", sphere_dabs(4, 0.4, 0.5, [1.0, 0.8, 1.0, 1.0]), 0.5, 0.4, False),
        ("nudge_grid", "nudge", "grid", strip_dabs(4, 0.08, 0.5, 0.5, [1.0]), 0.5, 0.5, False),
        ("crease_sphere", "crease", "sphere", sphere_dabs(3, 0.5, 0.25, [1.0]), 0.25, 0.5, False),
        ("pinch_sphere", "pinch", "sphere", sphere_dabs(4, 0.4, 0.5, [1.0, 0.5, 1.0, 1.0]), 0.5, 0.4, False),
        ("smooth_sphere", "smooth", "sphere", sphere_dabs(3, 0.5, 0.7, [1.0]), 0.7, 0.5, False),
        ("smooth_sphere_four_dabs", "smooth", "sphere", sphere_dabs(4, 0.5, 1.0, [1.0]), 1.0, 0.5, False),
        ("smooth_sphere_mirror", "smooth", "sphere", sphere_dabs(3, 0.5, 0.7, [1.0]), 0.7, 0.5, True),
        (
            "smooth_sphere_mirror_two",
            "smooth",
            "sphere",
            [
                {"position": (0.3, 0.2, 0.9), "radius": 0.5, "strength": 0.7, "pressure": 1.0},
                {"position": (0.1, -0.4, 0.9), "radius": 0.5, "strength": 0.7, "pressure": 1.0},
            ],
            0.7,
            0.5,
            True,
        ),
        ("flatten_sphere", "flatten", "sphere", sphere_dabs(4, 0.4, 0.5, [1.0]), 0.5, 0.4, False),
        ("fill_sphere", "fill", "sphere", sphere_dabs(4, 0.4, 0.7, [1.0, 0.5, 1.0, 1.0]), 0.7, 0.4, False),
        ("scrape_sphere", "scrape", "sphere", sphere_dabs(4, 0.4, 0.7, [1.0, 0.5, 1.0, 1.0]), 0.7, 0.4, False),
        ("grab_grid", "grab", "grid", strip_dabs(4, 0.08, 0.5, 0.4, [1.0]), 0.4, 0.5, False),
        ("snake_hook_grid", "snake_hook", "grid", strip_dabs(4, 0.08, 0.5, 1.0, [1.0]), 1.0, 0.5, False),
        ("thumb_grid", "thumb", "grid", strip_dabs(4, 0.08, 0.5, 0.5, [1.0]), 0.5, 0.5, False),
        ("rotate_sphere", "rotate", "sphere", sphere_dabs(3, 0.5, 0.6, [1.0]), 0.6, 0.5, False),
        ("elastic_grab_grid", "elastic_deform", "grid", strip_dabs(4, 0.08, 0.5, 0.5, [1.0]), 0.5, 0.5, False),
    ]


def all_cases():
    # Pose and Boundary cases carry an explicit deform type; the other brushes carry None.
    return [(*case, None) for case in cases()] + deform_cases()


def path_dabs(points, radius, strength):
    # Dabs along a planar path on the z = 0 plane, one per (x, y) point.
    return [
        {"position": (x, y, 0.0), "radius": radius, "strength": strength, "pressure": 1.0}
        for x, y in points
    ]


def deform_cases():
    # Pose and Boundary are measured on the flat grid, so the top view's ray hits z = 0 at each
    # dab location. Blender's Pose origin comes from the stroke's topology.
    pose_path = [(0.85, 0.0), (0.9, 0.05), (0.95, 0.1), (0.97, 0.15)]
    # Boundary's bend reads Blender's cursor location at stroke start (`cache->initial_location`). A
    # fresh headless session leaves the cursor at the origin, so the first dab sits there.
    boundary_path = [(0.0, 0.0), (0.05, 0.0), (0.1, 0.02), (0.12, 0.04)]
    return [
        ("pose_grid", "pose", "grid", path_dabs(pose_path, 0.45, 0.5), 0.5, 0.45, False, "ROTATE_TWIST"),
        ("pose_grid_scale_translate", "pose", "grid", path_dabs(pose_path, 0.45, 0.5), 0.5, 0.45, False, "SCALE_TRANSLATE"),
        ("pose_grid_squash_stretch", "pose", "grid", path_dabs(pose_path, 0.45, 0.5), 0.5, 0.45, False, "SQUASH_STRETCH"),
        ("boundary_grid_bend", "boundary", "grid", path_dabs(boundary_path, 1.05, 0.5), 0.5, 1.05, False, "BEND"),
        ("boundary_grid_grab", "boundary", "grid", path_dabs(boundary_path, 1.05, 0.5), 0.5, 1.05, False, "GRAB"),
        ("boundary_grid_expand", "boundary", "grid", path_dabs(boundary_path, 1.05, 0.5), 0.5, 1.05, False, "EXPAND"),
        ("boundary_grid_inflate", "boundary", "grid", path_dabs(boundary_path, 1.05, 0.5), 0.5, 1.05, False, "INFLATE"),
        ("boundary_grid_twist", "boundary", "grid", path_dabs(boundary_path, 1.05, 0.5), 0.5, 1.05, False, "TWIST"),
        ("boundary_grid_smooth", "boundary", "grid", path_dabs(boundary_path, 1.05, 0.5), 0.5, 1.05, False, "SMOOTH"),
        ("pose_grid_mirror", "pose", "grid", path_dabs(pose_path, 0.45, 0.5), 0.5, 0.45, True, "ROTATE_TWIST"),
        ("pose_grid_scale_translate_mirror", "pose", "grid", path_dabs(pose_path, 0.45, 0.5), 0.5, 0.45, True, "SCALE_TRANSLATE"),
        ("boundary_grid_bend_mirror", "boundary", "grid", path_dabs(boundary_path, 1.05, 0.5), 0.5, 1.05, True, "BEND"),
        ("boundary_grid_grab_mirror", "boundary", "grid", path_dabs(boundary_path, 1.05, 0.5), 0.5, 1.05, True, "GRAB"),
    ]


def generate(output_dir):
    with bpy.data.libraries.load(ESSENTIALS, link=False) as (source, target):
        target.brushes = list(source.brushes)
    set_top_ortho_view()
    for name, brush_key, kind, dabs, strength, radius, mirror, deform in all_cases():
        # A failing case is recorded in its own error file, so the other brushes still generate.
        try:
            case = run_case(
                name,
                brush_key,
                kind,
                dabs,
                strength,
                radius,
                mirror,
                mask=MASKS.get(name),
                deform=deform,
            )
        except AssertionError:
            with open(os.path.join(output_dir, name + ".error.txt"), "w") as handle:
                handle.write(traceback.format_exc())
            print("FAILED", name, flush=True)
            continue
        path = os.path.join(output_dir, name + ".json")
        with open(path, "w") as handle:
            json.dump(case, handle)
        print("WROTE", path, len(case["base"]), "vertices", flush=True)


def run_once():
    output_dir = sys.argv[sys.argv.index("--") + 1]
    try:
        generate(output_dir)
    except Exception:
        with open(os.path.join(output_dir, "error.txt"), "w") as handle:
            handle.write(traceback.format_exc())
        print(traceback.format_exc(), flush=True)
        bpy.ops.wm.quit_blender()
        return None
    bpy.ops.wm.quit_blender()
    return None


# The window only has a usable VIEW_3D region after the first event-loop tick.
bpy.app.timers.register(run_once, first_interval=1.0)
