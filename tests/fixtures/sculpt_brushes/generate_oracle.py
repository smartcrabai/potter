# SPDX-License-Identifier: GPL-2.0-or-later
#
# Generates the Blender 5.2.2 oracle fixtures for Potter's dab-sequential sculpt brushes
# (Clay Strips, Inflate, Layer). Run headless:
#
#   Blender --background --factory-startup --python generate_oracle.py -- <output_dir>
#
# Each case replays a DOTS stroke with explicit dab `location`s through `sculpt.brush_stroke`,
# so Blender applies one `do_brush_action` per dab at the given world position. The brush
# settings below are set explicitly; Potter's stroke carries only radius, strength and pressure.
#
# Blender facts this script relies on (Blender 5.2.2 source):
# - `unified_paint_settings.use_unified_size` must be off, or brush sizes are read from the
#   unified settings instead of the brush (`BKE_brush_use_locked_size`, `BKE_paint_use_unified_size`).
# - `use_locked_size = 'SCENE'` makes the dab radius `unprojected_size / 2` (`brush.cc`
#   `BKE_brush_unprojected_radius_get`).
# - Clay Strips needs a non-zero grab delta between dabs (`clay_strips.cc` returns early otherwise).
#   The grab delta is the screen-space stroke delta unprojected at the first dab's depth
#   (`sculpt.cc` `update_cache_variants`). `SculptPaintStroke::test_start` recomputes the view
#   matrices from `view_rotation`, which `view3d.view_axis(type='TOP')` sets to identity, so the
#   stroke sees a top orthographic view. Mouse positions `centre + K * location` (K > 0) then
#   unproject to exactly `location` deltas up to the pixel scale, which only changes magnitudes that
#   the strip frame normalizes away. Python's pre-stroke projection is not used, because the
#   matrices it reads are stale in background mode.
# - Mirrored cases set `mesh.use_mirror_x`; Blender then runs one brush pass per mirror
#   combination, with the location (and Clay Strips grab delta and plane) flipped.
# - Vertex normals are the stroke-start normals for every dab of a stroke (verified against the
#   Inflate and Layer results), not recomputed after each dab.

import json
import math
import os
import sys

import bpy
from bpy_extras import view3d_utils
from mathutils import Vector

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
}


def sphere_dabs(count, radius, strength, pressures):
    # Points on and slightly above the unit sphere, spread over the visible cap so every dab
    # touches the mesh under the stale top-down projection.
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
        position = base[index % len(base)]
        dabs.append(
            {
                "position": position,
                "radius": radius,
                "strength": strength,
                "pressure": pressures[index % len(pressures)],
            }
        )
    return dabs


def strip_dabs(view_matrix, start, count, step, radius, strength, pressures):
    # Straight stroke on the z = 0 plane, perpendicular to the stale view's horizontal direction,
    # so that Blender's screen-space grab delta equals the world-space delta (see module note).
    back = view_matrix[2]
    direction = Vector((-back[1], back[0], 0.0)).normalized()
    dabs = []
    for index in range(count):
        position = tuple(Vector(start) + direction * (index * step))
        dabs.append(
            {
                "position": position,
                "radius": radius,
                "strength": strength,
                "pressure": pressures[index % len(pressures)],
            }
        )
    return dabs


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


def run_case(name, brush_key, kind, dab_factory, strength, radius, window, area, region, rv3d, mirror=False):
    ob = make_object(kind)
    bpy.context.view_layer.objects.active = ob
    base = [list(v.co) for v in ob.data.vertices]
    faces = [list(p.vertices) for p in ob.data.polygons]
    ob.data.use_mirror_x = mirror
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="SCULPT")
        ts = bpy.context.tool_settings
        ts.sculpt.use_symmetry_x = False
        ts.sculpt.use_symmetry_y = False
        ts.sculpt.use_symmetry_z = False
        bpy.ops.view3d.view_axis(type="TOP")
        bpy.context.evaluated_depsgraph_get().update()
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
        dabs = dab_factory(rv3d.view_matrix)
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
    return {
        "name": name,
        "brush": brush_key,
        "strength": strength,
        "radius": radius,
        "dabs": dabs,
        "base": base,
        "faces": faces,
        "result": result_positions,
    }


def main():
    output_dir = sys.argv[sys.argv.index("--") + 1]
    window = bpy.context.window_manager.windows[0]
    area = next(a for a in window.screen.areas if a.type == "VIEW_3D")
    region = next(r for r in area.regions if r.type == "WINDOW")
    rv3d = area.spaces[0].region_3d
    rv3d.view_perspective = "ORTHO"
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.view3d.view_axis(type="TOP")
    with bpy.data.libraries.load(ESSENTIALS, link=False) as (source, target):
        target.brushes = list(source.brushes)
    window = bpy.context.window_manager.windows[0]
    area = next(a for a in window.screen.areas if a.type == "VIEW_3D")
    region = next(r for r in area.regions if r.type == "WINDOW")
    rv3d = area.spaces[0].region_3d

    cases = [
        run_case(
            "inflate_sphere",
            "inflate",
            "sphere",
            lambda _vm: sphere_dabs(4, 0.4, 1.0, [1.0, 1.0, 0.5, 1.0]),
            1.0,
            0.4,
            window,
            area,
            region,
            rv3d,
        ),
        run_case(
            "inflate_sphere_strength",
            "inflate",
            "sphere",
            lambda _vm: sphere_dabs(3, 0.5, 0.7, [0.6, 1.0, 1.0]),
            0.7,
            0.5,
            window,
            area,
            region,
            rv3d,
        ),
        run_case(
            "layer_sphere_clamp",
            "layer",
            "sphere",
            lambda _vm: sphere_dabs(12, 0.4, 1.0, [1.0]),
            1.0,
            0.4,
            window,
            area,
            region,
            rv3d,
        ),
        run_case(
            "layer_sphere",
            "layer",
            "sphere",
            lambda _vm: sphere_dabs(5, 0.45, 0.8, [1.0, 0.5, 1.0, 0.8, 1.0]),
            0.8,
            0.45,
            window,
            area,
            region,
            rv3d,
        ),
        run_case(
            "clay_strips_grid",
            "clay_strips",
            "grid",
            lambda vm: strip_dabs(vm, (0.0, 0.0, 0.0), 4, 0.08, 0.5, 1.0, [1.0]),
            1.0,
            0.5,
            window,
            area,
            region,
            rv3d,
        ),
        run_case(
            "inflate_sphere_mirror",
            "inflate",
            "sphere",
            lambda _vm: sphere_dabs(3, 0.4, 1.0, [1.0]),
            1.0,
            0.4,
            window,
            area,
            region,
            rv3d,
            mirror=True,
        ),
        run_case(
            "clay_strips_grid_mirror",
            "clay_strips",
            "grid",
            lambda vm: strip_dabs(vm, (0.0, 0.0, 0.0), 4, 0.08, 0.5, 1.0, [1.0]),
            1.0,
            0.5,
            window,
            area,
            region,
            rv3d,
            mirror=True,
        ),
        run_case(
            "clay_strips_grid_pressure",
            "clay_strips",
            "grid",
            lambda vm: strip_dabs(vm, (0.0, 0.0, 0.0), 4, 0.1, 0.4, 0.6, [1.0, 0.5, 0.8, 1.0]),
            0.6,
            0.4,
            window,
            area,
            region,
            rv3d,
        ),
    ]
    for case in cases:
        path = os.path.join(output_dir, case["name"] + ".json")
        with open(path, "w") as handle:
            json.dump(case, handle)
        print("WROTE", path, len(case["base"]), "vertices")


main()
