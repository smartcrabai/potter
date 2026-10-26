"""Generate Blender 5.2.2 oracle for ordered Dynamic Paint canvas surfaces."""
import json
import os
import sys

import bpy

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
os.makedirs(root, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 4
scene.use_gravity = False

coords = [-1.0, 0.0, 1.0]
vertices = [(x, y, 0.0) for y in coords for x in coords]
faces = []
for row in range(2):
    for column in range(2):
        lower_left = row * 3 + column
        faces.append((lower_left, lower_left + 1, lower_left + 4, lower_left + 3))
mesh = bpy.data.meshes.new("MultiSurfaceCanvasMesh")
mesh.from_pydata(vertices, [], faces)
mesh.update()
canvas = bpy.data.objects.new("MultiSurfaceCanvas", mesh)
scene.collection.objects.link(canvas)


def new_collection(name):
    collection = bpy.data.collections.new(name)
    scene.collection.children.link(collection)
    return collection


paint_collection = new_collection("PaintBrushes")
weight_collection = new_collection("WeightBrushes")
inactive_collection = new_collection("InactiveBrushes")


def activate(obj):
    for candidate in bpy.context.selected_objects:
        candidate.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def make_brush(name, collection, location, color):
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=8, ring_count=4, radius=0.12, location=location
    )
    obj = bpy.context.object
    obj.name = name
    for old_collection in list(obj.users_collection):
        old_collection.objects.unlink(obj)
    collection.objects.link(obj)
    modifier = obj.modifiers.new("Dynamic Paint Brush", "DYNAMIC_PAINT")
    modifier.ui_type = "BRUSH"
    activate(obj)
    if bpy.ops.dpaint.type_toggle(type="BRUSH") != {"FINISHED"}:
        raise RuntimeError("Blender did not create Dynamic Paint brush " + name)
    settings = modifier.brush_settings
    settings.paint_source = "POINT"
    settings.paint_color = color
    settings.paint_alpha = 1.0
    settings.paint_wetness = 1.0
    settings.paint_distance = 0.7
    settings.use_absolute_alpha = True
    return obj


make_brush("PaintBrush", paint_collection, (0.0, 0.0, 0.25), (1.0, 0.0, 0.0))
make_brush("WeightBrush", weight_collection, (1.0, 0.0, 0.25), (0.0, 1.0, 0.0))
make_brush("InactiveBrush", inactive_collection, (-1.0, 0.0, 0.25), (0.0, 0.0, 1.0))

modifier = canvas.modifiers.new("Dynamic Paint Canvas", "DYNAMIC_PAINT")
modifier.ui_type = "CANVAS"
activate(canvas)
if bpy.ops.dpaint.type_toggle(type="CANVAS") != {"FINISHED"}:
    raise RuntimeError("Blender did not create Dynamic Paint canvas")


def add_surface(name, surface_type, frame_start, frame_end, collection, output_a, output_b=None):
    settings = modifier.canvas_settings
    if len(settings.canvas_surfaces) == 0:
        raise RuntimeError("Dynamic Paint canvas has no initial surface")
    if len(settings.canvas_surfaces) > 0 and name != "PaintSurface":
        with bpy.context.temp_override(object=canvas, active_object=canvas):
            result = bpy.ops.dpaint.surface_slot_add()
        if result != {"FINISHED"}:
            raise RuntimeError("Blender did not add Dynamic Paint surface " + name)
    surface = settings.canvas_surfaces[-1]
    surface.name = name
    surface.surface_type = surface_type
    surface.surface_format = "VERTEX"
    surface.frame_start = frame_start
    surface.frame_end = frame_end
    surface.frame_substeps = 0
    surface.is_active = name != "InactiveSurface"
    surface.brush_collection = collection
    surface.output_name_a = output_a
    surface.use_output_a = True
    if output_b is not None:
        surface.output_name_b = output_b
        surface.use_output_b = True
        surface.use_drying = False
    if surface_type == "WEIGHT":
        surface.use_dissolve = False
    else:
        surface.use_dissolve = False
    if surface_type == "PAINT":
        mesh.color_attributes.new(name=output_a, type="BYTE_COLOR", domain="CORNER")
        if output_b is not None:
            mesh.color_attributes.new(name=output_b, type="BYTE_COLOR", domain="CORNER")
    return surface


paint_surface = add_surface(
    "PaintSurface", "PAINT", 1, 2, paint_collection, "MultiPaint", "MultiWetness"
)
weight_group = canvas.vertex_groups.new(name="MultiWeights")
weight_group.add([0], 0.0, "REPLACE")
weight_surface = add_surface(
    "WeightSurface", "WEIGHT", 2, 3, weight_collection, "MultiWeights"
)
inactive_surface = add_surface(
    "InactiveSurface", "PAINT", 1, 4, inactive_collection, "InactivePaint"
)

scene.frame_set(1)
bpy.context.view_layer.update()
bpy.ops.wm.save_as_mainfile(
    filepath=os.path.join(root, "DynamicPaintMultiSurface.blend"), relative_remap=True
)


def color_values(evaluated_mesh, name):
    layer = evaluated_mesh.color_attributes.get(name)
    if layer is None:
        raise RuntimeError("Dynamic Paint output layer is missing: " + name)
    values = [None] * len(evaluated_mesh.vertices)
    for loop in evaluated_mesh.loops:
        if values[loop.vertex_index] is None:
            values[loop.vertex_index] = list(layer.data[loop.index].color)
    return [value if value is not None else [0.0, 0.0, 0.0, 0.0] for value in values]


def weight_values(evaluated, name):
    group = evaluated.vertex_groups.get(name)
    if group is None:
        raise RuntimeError("Dynamic Paint output vertex group is missing: " + name)
    values = []
    for vertex in canvas.data.vertices:
        try:
            values.append(float(group.weight(vertex.index)))
        except RuntimeError:
            values.append(0.0)
    return values


frames = {}
for frame in range(1, 5):
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    evaluated = canvas.evaluated_get(bpy.context.evaluated_depsgraph_get())
    evaluated_mesh = evaluated.to_mesh()
    frames[str(frame)] = {
        "colors": {
            "MultiPaint": color_values(evaluated_mesh, "MultiPaint"),
            "MultiWetness": color_values(evaluated_mesh, "MultiWetness"),
            "InactivePaint": color_values(evaluated_mesh, "InactivePaint"),
        },
        "weights": weight_values(evaluated, "MultiWeights"),
    }
    evaluated.to_mesh_clear()

capture = {
    "blender_version": bpy.app.version_string,
    "surfaces": [
        {
            "name": surface.name,
            "surface_type": surface.surface_type,
            "surface_format": surface.surface_format,
            "is_active": surface.is_active,
            "frame_start": surface.frame_start,
            "frame_end": surface.frame_end,
            "brush_collection": surface.brush_collection.name,
            "output_name_a": surface.output_name_a,
            "output_name_b": surface.output_name_b,
            "use_output_a": surface.use_output_a,
            "use_output_b": surface.use_output_b,
        }
        for surface in modifier.canvas_settings.canvas_surfaces
    ],
    "frames": frames,
}
with open(os.path.join(root, "oracle.json"), "w", encoding="utf-8") as output:
    json.dump(capture, output, separators=(",", ":"))
