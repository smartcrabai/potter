"""Generate Blender 5.2.2 Dynamic Paint WEIGHT bake oracle."""
import json
import sys
from pathlib import Path

import bpy


arguments = sys.argv[sys.argv.index("--") + 1:]
OUTPUT = Path(arguments[0])
oracle_only = len(arguments) > 1 and arguments[1] == "--oracle-only"
if len(arguments) > 1 and not oracle_only:
    raise RuntimeError("the only optional argument is --oracle-only")
OUTPUT.mkdir(parents=True, exist_ok=True)
FRAME_START = 1
FRAME_END = 8
DISSOLVE_SPEED = 4
GROUPS = {
    "CanvasHistoryRaw": "HistoryRawWeight",
    "CanvasHistoryLinear": "HistoryLinearWeight",
    "CanvasHistoryLog": "HistoryLogWeight",
    "CanvasErase": "EraseWeight",
    "CanvasNoErase": "NoEraseWeight",
    "CanvasSmooth": "SmoothWeight",
    "CanvasUnitScale": "UnitScaleWeight",
}

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.frame_start = FRAME_START
scene.frame_end = FRAME_END

# Dense samples at and around the radius make CONSTANT and SMOOTH falloff diverge.
xs = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
ys = [-1.0, -0.5, 0.0, 0.5, 1.0]
vertices = [(x, y, 0.0) for y in ys for x in xs]
faces = []
for row in range(len(ys) - 1):
    for column in range(len(xs) - 1):
        a = row * len(xs) + column
        faces.append((a, a + 1, a + 1 + len(xs), a + len(xs)))
mesh = bpy.data.meshes.new("CanvasMesh")
mesh.from_pydata(vertices, [], faces)
mesh.update()


def new_collection(name):
    collection = bpy.data.collections.new(name)
    scene.collection.children.link(collection)
    return collection


constant_brushes = new_collection("ConstantBrushes")
point_only_brushes = new_collection("PointOnlyBrushes")
smooth_brushes = new_collection("SmoothBrushes")
history_brushes = new_collection("HistoryBrushes")


def activate(obj):
    for candidate in bpy.context.selected_objects:
        candidate.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def make_brush(name, collection, distance, falloff, erase=False):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=8, ring_count=4, radius=0.12)
    obj = bpy.context.object
    obj.name = name
    for candidate in list(obj.users_collection):
        candidate.objects.unlink(obj)
    collection.objects.link(obj)
    modifier = obj.modifiers.new("Dynamic Paint Brush", "DYNAMIC_PAINT")
    modifier.ui_type = "BRUSH"
    activate(obj)
    if bpy.ops.dpaint.type_toggle(type="BRUSH") != {"FINISHED"}:
        raise RuntimeError("Blender did not create a Dynamic Paint brush: " + name)
    settings = modifier.brush_settings
    settings.paint_source = "POINT"
    settings.paint_distance = distance
    settings.proximity_falloff = falloff
    settings.use_paint_erase = erase
    return obj, settings


def animate_location(obj, keyframes):
    for frame, location in keyframes:
        obj.location = location
        obj.keyframe_insert(data_path="location", frame=frame)


point_brush, point_settings = make_brush(
    "PointBrush", constant_brushes, 1.0, "CONSTANT"
)
point_parent = bpy.data.objects.new("BrushParent", None)
scene.collection.objects.link(point_parent)
point_parent.location = (0.0, 8.0, 0.0)
point_brush.parent = point_parent
point_brush.delta_location = (0.0, 2.0, 0.0)
animate_location(
    point_brush,
    (
        (FRAME_START, (-2.0, 0.0, 0.25)),
        (2, (-1.0, 0.0, 0.25)),
        (3, (0.0, 0.0, 0.25)),
        (4, (1.0, 0.0, 0.25)),
        (5, (2.0, 0.0, 0.25)),
        (FRAME_END, (2.0, 0.0, 0.25)),
    ),
)
point_only_brushes.objects.link(point_brush)
history_brush, history_settings = make_brush(
    "HistoryBrush", history_brushes, 1.0, "CONSTANT"
)
animate_location(
    history_brush,
    (
        (FRAME_START, (-2.0, 0.0, 0.25)),
        (2, (20.0, 0.0, 0.25)),
        (FRAME_END, (20.0, 0.0, 0.25)),
    ),
)

erase_brush, erase_settings = make_brush(
    "EraseBrush", constant_brushes, 0.4, "CONSTANT", erase=True
)
animate_location(
    erase_brush,
    (
        (FRAME_START, (-5.0, 0.0, 0.25)),
        (2, (-2.0, 0.0, 0.25)),
        (FRAME_END, (-2.0, 0.0, 0.25)),
    ),
)

smooth_brush, smooth_settings = make_brush(
    "SmoothBrush", smooth_brushes, 1.0, "SMOOTH"
)
animate_location(
    smooth_brush,
    (
        (FRAME_START, (-2.0, 0.0, 0.25)),
        (2, (-1.0, 0.0, 0.25)),
        (3, (0.0, 0.0, 0.25)),
        (4, (1.0, 0.0, 0.25)),
        (5, (2.0, 0.0, 0.25)),
        (FRAME_END, (2.0, 0.0, 0.25)),
    ),
)

scenario_specs = {
    "CanvasHistoryRaw": {
        "group": GROUPS["CanvasHistoryRaw"],
        "collection": history_brushes,
        "use_log": False,
        "radius_scale": 0.8,
        "influence_scale": 0.55,
    },
    "CanvasHistoryLinear": {
        "group": GROUPS["CanvasHistoryLinear"],
        "collection": history_brushes,
        "use_log": False,
        "radius_scale": 0.8,
        "influence_scale": 0.55,
        "weight_edit": True,
    },
    "CanvasHistoryLog": {
        "group": GROUPS["CanvasHistoryLog"],
        "collection": history_brushes,
        "use_log": True,
        "radius_scale": 0.8,
        "influence_scale": 0.55,
    },
    "CanvasErase": {
        "group": GROUPS["CanvasErase"],
        "collection": constant_brushes,
        "use_log": False,
        "radius_scale": 0.8,
        "influence_scale": 0.55,
    },
    "CanvasNoErase": {
        "group": GROUPS["CanvasNoErase"],
        "collection": point_only_brushes,
        "use_log": False,
        "radius_scale": 0.8,
        "influence_scale": 0.55,
    },
    "CanvasSmooth": {
        "group": GROUPS["CanvasSmooth"],
        "collection": smooth_brushes,
        "use_log": False,
        "radius_scale": 0.8,
        "influence_scale": 0.55,
    },
    "CanvasUnitScale": {
        "group": GROUPS["CanvasUnitScale"],
        "collection": point_only_brushes,
        "use_log": False,
        "radius_scale": 1.0,
        "influence_scale": 1.0,
    },
}


def make_canvas(name, spec):
    group_name = spec["group"]
    canvas_mesh = mesh.copy()
    canvas_mesh.name = name + "Mesh"
    canvas = bpy.data.objects.new(name, canvas_mesh)
    scene.collection.objects.link(canvas)
    output_group = canvas.vertex_groups.new(name=group_name)
    source_vertex = len(vertices) - 1
    output_group.add([source_vertex], 0.0, "REPLACE")

    modifier = canvas.modifiers.new("Dynamic Paint Canvas", "DYNAMIC_PAINT")
    modifier.ui_type = "CANVAS"
    activate(canvas)
    if bpy.ops.dpaint.type_toggle(type="CANVAS") != {"FINISHED"}:
        raise RuntimeError("Blender did not create a Dynamic Paint canvas: " + name)
    surface = modifier.canvas_settings.canvas_surfaces[0]
    surface.surface_type = "WEIGHT"
    surface.surface_format = "VERTEX"
    surface.output_name_a = group_name
    surface.frame_start = FRAME_START
    surface.frame_end = FRAME_END
    surface.frame_substeps = 0
    surface.use_dissolve = True
    surface.dissolve_speed = DISSOLVE_SPEED
    surface.use_dissolve_log = spec["use_log"]
    surface.brush_collection = spec["collection"]
    surface.brush_radius_scale = spec["radius_scale"]
    surface.brush_influence_scale = spec["influence_scale"]

    if spec.get("weight_edit", False):
        weight_edit = canvas.modifiers.new("Named Weight Falloff", "VERTEX_WEIGHT_EDIT")
        weight_edit.vertex_group = group_name
        weight_edit.falloff_type = "SHARP"
        weight_edit.normalize = False
        weight_edit.use_add = False
        weight_edit.use_remove = False
    return canvas, output_group, surface


canvases = {}
for name, spec in scenario_specs.items():
    canvas, output_group, surface = make_canvas(name, spec)
    canvases[name] = (canvas, output_group, surface, spec)

scene.frame_set(FRAME_START)
bpy.context.view_layer.update()
if not oracle_only:
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT / "source.blend"))


def group_weights(obj, name):
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    group = evaluated.vertex_groups.get(name)
    if group is None:
        raise RuntimeError("Dynamic Paint did not create the named vertex group: " + name)
    values = []
    for vertex in obj.data.vertices:
        try:
            values.append(float(group.weight(vertex.index)))
        except RuntimeError:
            values.append(0.0)
    return values


frames = {name: {} for name in GROUPS}
raw_canvas, raw_group = canvases["CanvasHistoryRaw"][:2]
linear_canvas, linear_group = canvases["CanvasHistoryLinear"][:2]
for frame in range(FRAME_START, FRAME_END + 1):
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    history_weights = group_weights(raw_canvas, raw_group.name)
    frames["CanvasHistoryRaw"][str(frame)] = {
        "weights": history_weights,
        "edited_weights": history_weights,
    }
    frames["CanvasHistoryLinear"][str(frame)] = {
        "weights": history_weights,
        "edited_weights": group_weights(linear_canvas, linear_group.name),
    }
    for name, (canvas, output_group, _surface, _spec) in canvases.items():
        if name in ("CanvasHistoryRaw", "CanvasHistoryLinear"):
            continue
        raw_weights = group_weights(canvas, output_group.name)
        frames[name][str(frame)] = {
            "weights": raw_weights,
            "edited_weights": raw_weights,
        }

source_group_weights = {}
for name, (canvas, output_group, _surface, _spec) in canvases.items():
    source_group_weights[name] = {
        f"v{vertex.index + 1}": float(assignment.weight)
        for vertex in canvas.data.vertices
        for assignment in vertex.groups
        if assignment.group == output_group.index
    }

capture = {
    "blender_version": bpy.app.version_string,
    "frame_start": FRAME_START,
    "frame_end": FRAME_END,
    "brushes": {
        "point": {
            "name": point_brush.name,
            "source": point_settings.paint_source,
            "distance": point_settings.paint_distance,
            "proximity_falloff": point_settings.proximity_falloff,
            "use_paint_erase": point_settings.use_paint_erase,
            "parent": point_brush.parent.name,
            "parent_location": list(point_parent.location),
            "delta_location": list(point_brush.delta_location),
        },
        "history": {
            "name": history_brush.name,
            "source": history_settings.paint_source,
            "distance": history_settings.paint_distance,
            "proximity_falloff": history_settings.proximity_falloff,
            "use_paint_erase": history_settings.use_paint_erase,
        },
        "erase": {
            "name": erase_brush.name,
            "source": erase_settings.paint_source,
            "distance": erase_settings.paint_distance,
            "proximity_falloff": erase_settings.proximity_falloff,
            "use_paint_erase": erase_settings.use_paint_erase,
        },
        "smooth": {
            "name": smooth_brush.name,
            "source": smooth_settings.paint_source,
            "distance": smooth_settings.paint_distance,
            "proximity_falloff": smooth_settings.proximity_falloff,
            "use_paint_erase": smooth_settings.use_paint_erase,
        },
    },
    "canvases": {
        name: {
            "canvas": name,
            "group": GROUPS[name],
            "surface": {
                "format": surface.surface_format,
                "type": surface.surface_type,
                "output_name": surface.output_name_a,
                "frame_start": surface.frame_start,
                "frame_end": surface.frame_end,
                "frame_substeps": surface.frame_substeps,
                "use_dissolve": surface.use_dissolve,
                "dissolve_speed": surface.dissolve_speed,
                "use_dissolve_log": spec["use_log"],
                "brush_radius_scale": surface.brush_radius_scale,
                "brush_influence_scale": surface.brush_influence_scale,
            },
            "downstream_modifier": (
                {
                    "type": "VERTEX_WEIGHT_EDIT",
                    "vertex_group": GROUPS[name],
                    "falloff_type": "SHARP",
                }
                if spec.get("weight_edit", False)
                else None
            ),
            "source_group_weights": source_group_weights[name],
            "frames": frames[name],
        }
        for name, (_canvas, _group, surface, spec) in canvases.items()
    },
}
with (OUTPUT / "oracle.json").open("w", encoding="utf-8") as stream:
    json.dump(capture, stream, indent=2, sort_keys=True)
print("WROTE", OUTPUT / "oracle.json")
