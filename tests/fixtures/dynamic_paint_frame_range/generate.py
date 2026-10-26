import bpy
import json
import os
import sys


FRAME_RANGE = range(1, 5)


def activate(obj):
    for candidate in bpy.context.selected_objects:
        candidate.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def make_grid(name, wave, shape_key=False):
    coordinates = [-1.0, 0.0, 1.0]
    vertices = [(x, y, 0.0) for y in coordinates for x in coordinates]
    faces = []
    for y in range(2):
        for x in range(2):
            lower_left = y * 3 + x
            faces.append((lower_left, lower_left + 1, lower_left + 4, lower_left + 3))
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    if shape_key:
        obj.shape_key_add(name="Basis")
        animated = obj.shape_key_add(name="AnimatedPose")
        animated.data[4].co.z = 2.0
        for frame, value in ((1, 0.0), (2, 0.5), (3, 1.0), (4, 0.25)):
            animated.value = value
            animated.keyframe_insert(data_path="value", frame=frame)
    if wave:
        modifier = obj.modifiers.new("UpstreamWave", "WAVE")
        modifier.height = 8.0
        modifier.width = 1.0
        modifier.narrowness = 2.0
        modifier.speed = 0.7
        modifier.time_offset = 1.0
    return obj


def make_brush(name, location):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=8, ring_count=4, radius=0.12, location=location)
    obj = bpy.context.object
    obj.name = name
    modifier = obj.modifiers.new("Brush", "DYNAMIC_PAINT")
    modifier.ui_type = "BRUSH"
    activate(obj)
    if bpy.ops.dpaint.type_toggle(type="BRUSH") != {"FINISHED"}:
        raise RuntimeError("Blender did not create a dynamic-paint brush")
    settings = modifier.brush_settings
    settings.paint_source = "POINT"
    settings.paint_color = (1.0, 0.0, 0.0)
    settings.paint_alpha = 1.0
    settings.paint_distance = 0.2
    settings.use_absolute_alpha = True
    return obj


def add_surface(canvas, modifier, name, start, end, output_name, first):
    activate(canvas)
    settings = modifier.canvas_settings
    if first:
        surface = settings.canvas_surfaces[0]
    else:
        with bpy.context.temp_override(object=canvas, active_object=canvas):
            result = bpy.ops.dpaint.surface_slot_add()
        if result != {"FINISHED"}:
            raise RuntimeError("Blender did not add a dynamic-paint surface")
        surface = settings.canvas_surfaces[-1]
    surface.name = name
    surface.surface_type = "PAINT"
    surface.surface_format = "VERTEX"
    surface.frame_start = start
    surface.frame_end = end
    surface.output_name_a = output_name
    surface.output_name_b = "dp_wetmap"
    surface.use_drying = True
    surface.use_dissolve = False
    canvas.data.color_attributes.new(name=output_name, type="BYTE_COLOR", domain="CORNER")
    return surface


def create_case(root, name, surfaces, wave, shape_key=False):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start = FRAME_RANGE.start
    scene.frame_end = FRAME_RANGE.stop - 1
    scene.use_gravity = False
    canvas = make_grid(name + "Canvas", wave, shape_key)
    brush_location = (0.0, 0.0, 0.0)
    if wave or shape_key:
        scene.frame_set(2)
        bpy.context.view_layer.update()
        evaluated = canvas.evaluated_get(bpy.context.evaluated_depsgraph_get())
        evaluated_mesh = evaluated.to_mesh()
        brush_location = tuple(evaluated_mesh.vertices[4].co)
        evaluated.to_mesh_clear()
    make_brush(name + "Brush", brush_location)

    modifier = canvas.modifiers.new("DynamicPaintCanvas", "DYNAMIC_PAINT")
    modifier.ui_type = "CANVAS"
    activate(canvas)
    if bpy.ops.dpaint.type_toggle(type="CANVAS") != {"FINISHED"}:
        raise RuntimeError("Blender did not create a dynamic-paint canvas")

    output_names = []
    for index, (surface_name, start, end, output_name) in enumerate(surfaces):
        add_surface(canvas, modifier, surface_name, start, end, output_name, index == 0)
        output_names.append(output_name)


    scene.frame_set(1)
    bpy.context.view_layer.update()
    blend_path = os.path.join(root, name + ".blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path, relative_remap=True)

    frames = {}
    for frame in FRAME_RANGE:
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        evaluated = canvas.evaluated_get(bpy.context.evaluated_depsgraph_get())
        evaluated_mesh = evaluated.to_mesh()
        colors = {}
        for layer_name in output_names:
            layer = evaluated_mesh.color_attributes.get(layer_name)
            if layer is None:
                raise RuntimeError("Blender did not produce color output " + layer_name)
            per_vertex = [None] * len(evaluated_mesh.vertices)
            for loop in evaluated_mesh.loops:
                if per_vertex[loop.vertex_index] is None:
                    per_vertex[loop.vertex_index] = list(layer.data[loop.index].color)
            colors[layer_name] = [
                value if value is not None else [0.0, 0.0, 0.0, 0.0]
                for value in per_vertex
            ]
        frames[str(frame)] = {
            "positions": [list(vertex.co) for vertex in evaluated_mesh.vertices],
            "colors": colors,
        }
        evaluated.to_mesh_clear()
    return {
        "surfaces": [
            {"name": surface_name, "frame_start": start, "frame_end": end, "output_name_a": output_name}
            for surface_name, start, end, output_name in surfaces
        ],
        "frames": frames,
    }


def main(root):
    os.makedirs(root, exist_ok=True)
    oracle = {
        "blender_version": bpy.app.version_string,
        "held_deformation": create_case(
            root,
            "HeldDynamicPaint",
            [("HeldSurface", 2, 2, "HeldPaint")],
            True,
        ),
        "animated_shape_key": create_case(
            root,
            "ShapeKeyDynamicPaint",
            [("ShapeKeySurface", 2, 2, "ShapeKeyPaint")],
            False,
            True,
        ),
        "multiple_surfaces": create_case(
            root,
            "MultipleDynamicPaint",
            [
                ("EarlySurface", 2, 2, "EarlyPaint"),
                ("LateSurface", 3, 4, "LatePaint"),
            ],
            False,
        ),
    }
    with open(os.path.join(root, "blender_expected.json"), "w", encoding="utf-8") as output:
        json.dump(oracle, output, separators=(",", ":"))


if __name__ == "__main__":
    main(os.path.realpath(sys.argv[sys.argv.index("--") + 1]))
