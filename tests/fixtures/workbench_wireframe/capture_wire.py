"""GUI capture of the Blender 5.2.2 viewport in wireframe shading.

Run with the scene file as the first argument so no splash screen opens:
  Blender --factory-startup --window-geometry 0 0 1000 800 wire_scene.blend --python capture_wire.py -- <out.png> <out.json> [view_transform look]
(wire_scene.blend comes from build_scene.py: blender -b --factory-startup --python build_scene.py -- wire_scene.blend)
The 3D viewport is the only visible area: its toolbars, header, gizmos, grid, axes, floor, text and
extras are off. The view is fixed (orthographic, Blender's front-right-top-style rotation below), and
the theme is the factory default. The JSON records the theme values, the region size and the
viewport matrices the capture ran with.
"""
import json
import sys

import bpy
from mathutils import Matrix, Quaternion, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
png_path, json_path = argv[0], argv[1]
VIEW_TRANSFORM = argv[2] if len(argv) > 2 else "Standard"
LOOK = argv[3] if len(argv) > 3 else "None"
BACK_OVERRIDE = float(argv[4]) if len(argv) > 4 else None

# View: looking down -Y-ish at the scene, orthographic. Rotation quaternion (w, x, y, z) is Blender's
# view_rotation, and the view centre is the origin, so the scene sits in the viewport centre.
VIEW_ROTATION = Quaternion((0.7071068, 0.7071068, 0.0, 0.0))
VIEW_DISTANCE = 12.0
TICKS = 6

state = {"tick": 0}


def find_view3d():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                return window, area
    return None, None


def prepare(area):
    if BACK_OVERRIDE is not None:
        grad = bpy.context.preferences.themes[0].view_3d.space.gradients
        grad.high_gradient = (BACK_OVERRIDE, BACK_OVERRIDE, BACK_OVERRIDE)
    bpy.context.scene.view_settings.view_transform = VIEW_TRANSFORM
    bpy.context.scene.view_settings.look = LOOK
    space = area.spaces.active
    space.shading.type = "WIREFRAME"
    space.overlay.show_overlays = True
    space.overlay.show_floor = False
    space.overlay.show_axis_x = False
    space.overlay.show_axis_y = False
    space.overlay.show_axis_z = False
    space.overlay.show_text = False
    space.overlay.show_object_origins = False
    space.overlay.show_extras = False
    space.overlay.show_bones = False
    space.overlay.show_relationship_lines = False
    space.overlay.show_outline_selected = False
    space.overlay.show_cursor = False
    space.show_gizmo = False
    space.show_region_ui = False
    space.show_region_toolbar = False
    space.show_region_header = False
    space.show_region_tool_header = False
    space.show_region_hud = False
    rv3d = space.region_3d
    rv3d.view_perspective = "ORTHO"
    rv3d.view_rotation = VIEW_ROTATION
    rv3d.view_distance = VIEW_DISTANCE
    rv3d.view_location = Vector((0.0, 0.0, 0.0))


def record(window, area, region, path_json):
    rv3d = area.spaces.active.region_3d
    theme = bpy.context.preferences.themes[0].view_3d
    fields = [
        "back", "back_grad", "wire", "wire_edit", "select", "active", "transform",
        "object_select", "vertex", "edge", "face", "grid",
    ]
    theme_values = {}
    for field in fields:
        if hasattr(theme, field):
            theme_values[field] = list(getattr(theme, field))
    gradients = theme.space.gradients
    data = {
        "blender_version": bpy.app.version_string,
        "theme_background": {
            "high_gradient": list(gradients.high_gradient),
            "gradient": list(gradients.gradient),
            "background_type": gradients.background_type,
        },
        "region": [region.width, region.height],
        "area": [area.width, area.height],
        "view_matrix": [list(row) for row in rv3d.view_matrix],
        "window_matrix": [list(row) for row in rv3d.window_matrix],
        "view_perspective": rv3d.view_perspective,
        "theme_view_3d": theme_values,
        "shading_type": area.spaces.active.shading.type,
        "wireframe_threshold": area.spaces.active.overlay.wireframe_threshold,
        "wireframe_opacity": area.spaces.active.overlay.wireframe_opacity,
        "wireframe_color_type": area.spaces.active.shading.wireframe_color_type,
        "background_type": area.spaces.active.shading.background_type,
        "view_transform": bpy.context.scene.view_settings.view_transform,
        "look": bpy.context.scene.view_settings.look,
    }
    with open(path_json, "w") as handle:
        json.dump(data, handle, indent=2)


def tick():
    state["tick"] += 1
    window, area = find_view3d()
    if area is None:
        print("NO_VIEW3D", flush=True)
        bpy.ops.wm.quit_blender()
        return None
    if state["tick"] == 1:
        prepare(area)
        area.tag_redraw()
        return 0.3
    area.tag_redraw()
    if state["tick"] < TICKS:
        return 0.3
    region = [r for r in area.regions if r.type == "WINDOW"][0]
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.screen.screenshot_area(filepath=png_path)
    record(window, area, region, json_path)
    print("CAPTURE_DONE", region.width, region.height, flush=True)
    bpy.ops.wm.quit_blender()
    return None


bpy.app.timers.register(tick, first_interval=0.5)
