import bpy
import bmesh
import json
import math
import os
import sys


FRAMES = range(1, 6)
COORDINATES = [-1.0, 0.0, 1.0]
VERTICES = [(x, y, 0.0) for y in COORDINATES for x in COORDINATES]
FACES = []
for y in range(2):
    for x in range(2):
        lower_left = y * 3 + x
        FACES.append((lower_left, lower_left + 1, lower_left + 4, lower_left + 3))

VOLUME_COORDINATES = [index / 4.0 for index in range(-8, 9)]
VOLUME_VERTICES = [
    (x, y, 0.0) for y in VOLUME_COORDINATES for x in VOLUME_COORDINATES
]
VOLUME_FACES = []
for y in range(len(VOLUME_COORDINATES) - 1):
    for x in range(len(VOLUME_COORDINATES) - 1):
        lower_left = y * len(VOLUME_COORDINATES) + x
        VOLUME_FACES.append(
            (lower_left, lower_left + 1, lower_left + 1 + len(VOLUME_COORDINATES), lower_left + len(VOLUME_COORDINATES))
        )


def activate(obj):
    for candidate in bpy.context.selected_objects:
        candidate.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def make_brush(name, collection, color, absolute_alpha):
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata([(0.0, 0.0, 0.0)], [], [])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    modifier = obj.modifiers.new(name + "Settings", "DYNAMIC_PAINT")
    modifier.ui_type = "BRUSH"
    activate(obj)
    if bpy.ops.dpaint.type_toggle(type="BRUSH") != {"FINISHED"}:
        raise RuntimeError("Blender did not create a Dynamic Paint brush")
    brush = modifier.brush_settings
    brush.paint_source = "POINT"
    brush.paint_distance = 0.1
    brush.paint_color = color
    brush.paint_alpha = 0.5
    brush.paint_wetness = 0.8
    brush.use_absolute_alpha = absolute_alpha
    obj.location = (0.0, 0.0, 0.0) if name.endswith("Red") else (10.0, 0.0, 0.0)
    obj.keyframe_insert(data_path="location", frame=1)
    obj.location = (10.0, 0.0, 0.0) if name.endswith("Red") else (0.0, 0.0, 0.0)
    obj.keyframe_insert(data_path="location", frame=2)
    obj.location = (10.0, 0.0, 0.0)
    obj.keyframe_insert(data_path="location", frame=3)
    return obj, brush


def make_brush_collection(scene, name, absolute_alpha):
    collection = bpy.data.collections.new(name)
    scene.collection.children.link(collection)
    red, red_settings = make_brush(name + "Red", collection, (1.0, 0.0, 0.0), absolute_alpha)
    blue, blue_settings = make_brush(name + "Blue", collection, (0.0, 0.0, 1.0), absolute_alpha)
    return collection, [(red, red_settings), (blue, blue_settings)]


def make_volume_brush_collection(scene):
    collection = bpy.data.collections.new("VolumeBrushes")
    scene.collection.children.link(collection)
    bpy.ops.mesh.primitive_cube_add(size=2.6)
    obj = bpy.context.object
    obj.name = "VolumeBrush"
    for vertex in obj.data.vertices:
        vertex.co.x += 1.5
    obj.data.update()
    for candidate in list(obj.users_collection):
        candidate.objects.unlink(obj)
    collection.objects.link(obj)
    obj.modifiers.new("PrePaintBevel", "BEVEL").width = 0.05
    modifier = obj.modifiers.new("VolumeSettings", "DYNAMIC_PAINT")
    modifier.ui_type = "BRUSH"
    activate(obj)
    if bpy.ops.dpaint.type_toggle(type="BRUSH") != {"FINISHED"}:
        raise RuntimeError("Blender did not create a VOLUME Dynamic Paint brush")
    # Blender paints from the modifier stack prefix; this later Mirror must not affect the brush.
    mirror = obj.modifiers.new("PostPaintMirror", "MIRROR")
    mirror.use_axis[0] = True
    brush = modifier.brush_settings
    brush.paint_source = "VOLUME"
    brush.paint_distance = 0.001
    brush.paint_color = (0.2, 0.8, 0.35)
    brush.paint_alpha = 0.01
    brush.paint_wetness = 0.35
    brush.use_absolute_alpha = False
    brush.proximity_falloff = "RAMP"
    for frame, x in enumerate((-2.0, -1.0, 0.0, 8.5, 8.5), start=1):
        obj.location = (x, 0.0, 0.0)
        obj.keyframe_insert(data_path="location", frame=frame)
    return collection, [(obj, brush)]


def make_volume_shape_brush_collection(scene, name, shape):
    collection = bpy.data.collections.new(name + "Brushes")
    scene.collection.children.link(collection)
    if shape in ("rotated_cube", "open_cube"):
        bpy.ops.mesh.primitive_cube_add(size=2.0)
        obj = bpy.context.object
        if shape == "rotated_cube":
            obj.rotation_euler[2] = math.radians(35.0)
        else:
            bm = bmesh.new()
            bm.from_mesh(obj.data)
            top = max(bm.faces, key=lambda face: face.calc_center_median().z)
            bm.faces.remove(top)
            bm.to_mesh(obj.data)
            bm.free()
            obj.data.update()
    elif shape == "torus":
        bpy.ops.mesh.primitive_torus_add(
            major_segments=24,
            minor_segments=12,
            major_radius=1.0,
            minor_radius=0.35,
        )
        obj = bpy.context.object
    else:
        raise RuntimeError("unknown VOLUME shape: " + shape)
    obj.name = "Volume" + name
    for candidate in list(obj.users_collection):
        candidate.objects.unlink(obj)
    collection.objects.link(obj)
    modifier = obj.modifiers.new(name + "Settings", "DYNAMIC_PAINT")
    modifier.ui_type = "BRUSH"
    activate(obj)
    if bpy.ops.dpaint.type_toggle(type="BRUSH") != {"FINISHED"}:
        raise RuntimeError("Blender did not create a VOLUME Dynamic Paint brush")
    brush = modifier.brush_settings
    brush.paint_source = "VOLUME"
    brush.paint_distance = 0.001
    brush.paint_color = (0.35, 0.65, 0.9)
    brush.paint_alpha = 1.0
    brush.paint_wetness = 1.0
    brush.use_absolute_alpha = True
    brush.proximity_falloff = "RAMP"
    return collection, [(obj, brush)]


def volume_shape_aabb_mask(canvas, brush_obj):
    evaluated = brush_obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    points = [evaluated.matrix_world @ vertex.co for vertex in mesh.vertices]
    evaluated.to_mesh_clear()
    minimum = [min(point[axis] for point in points) for axis in range(3)]
    maximum = [max(point[axis] for point in points) for axis in range(3)]
    positions = [list(vertex.co) for vertex in canvas.data.vertices]
    mask = [
        all(minimum[axis] <= point[axis] <= maximum[axis] for axis in range(3))
        for point in positions
    ]
    return positions, mask


def add_canvas(
    scene,
    name,
    brush_collection,
    drying,
    dissolving,
    output_a,
    output_b,
    flag_only=False,
    output_a_toggle=True,
    vertices=VERTICES,
    faces=FACES,
):
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    canvas = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(canvas)
    modifier = canvas.modifiers.new(name + "Settings", "DYNAMIC_PAINT")
    modifier.ui_type = "CANVAS"
    activate(canvas)
    if bpy.ops.dpaint.type_toggle(type="CANVAS") != {"FINISHED"}:
        raise RuntimeError("Blender did not create the Dynamic Paint canvas")
    surface = modifier.canvas_settings.canvas_surfaces[0]
    surface.surface_type = "PAINT"
    surface.surface_format = "VERTEX"
    surface.frame_start = 1
    surface.frame_end = 4
    surface.output_name_a = output_a
    surface.output_name_b = output_b
    surface.brush_collection = brush_collection
    surface.use_drying = drying
    surface.dry_speed = 4
    surface.use_dry_log = True
    surface.color_dry_threshold = 0.5
    surface.use_dissolve = dissolving
    surface.dissolve_speed = 4
    surface.use_dissolve_log = True
    if flag_only:
        surface.use_output_b = True
    activate(canvas)
    if output_a_toggle:
        if bpy.ops.dpaint.output_toggle(output="A") != {"FINISHED"}:
            raise RuntimeError("Blender did not create Dynamic Paint output A")
        if not flag_only:
            if bpy.ops.dpaint.output_toggle(output="B") != {"FINISHED"}:
                raise RuntimeError("Blender did not create Dynamic Paint output B")
    return canvas, surface


def color_layer(mesh, name):
    layer = mesh.color_attributes.get(name)
    if layer is None:
        return None
    per_vertex = [None] * len(mesh.vertices)
    for loop in mesh.loops:
        if per_vertex[loop.vertex_index] is None:
            per_vertex[loop.vertex_index] = list(layer.data[loop.index].color)
    return [value if value is not None else [0.0, 0.0, 0.0, 0.0] for value in per_vertex]


def surface_values(surface):
    return {
        "type": surface.surface_type,
        "format": surface.surface_format,
        "frame_start": surface.frame_start,
        "frame_end": surface.frame_end,
        "output_name_a": surface.output_name_a,
        "output_name_b": surface.output_name_b,
        "use_output_b": surface.use_output_b,
        "use_drying": surface.use_drying,
        "dry_speed": surface.dry_speed,
        "use_dry_log": surface.use_dry_log,
        "color_dry_threshold": surface.color_dry_threshold,
        "use_dissolve": surface.use_dissolve,
        "dissolve_speed": surface.dissolve_speed,
        "use_dissolve_log": surface.use_dissolve_log,
    }


def capture_canvas(scene, canvas, output_a, output_b):
    frames = {}
    for frame in FRAMES:
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        evaluated = canvas.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh = evaluated.to_mesh()
        frames[str(frame)] = {
            "colors": {
                output_a: color_layer(mesh, output_a),
                output_b: color_layer(mesh, output_b),
            },
        }
        evaluated.to_mesh_clear()
    return frames


def brush_values(brushes):
    values = []
    for obj, settings in brushes:
        brush = {
            "name": obj.name,
            "source": settings.paint_source,
            "color": list(settings.paint_color),
            "alpha": settings.paint_alpha,
            "wetness": settings.paint_wetness,
            "absolute_alpha": settings.use_absolute_alpha,
        }
        if settings.paint_source == "VOLUME":
            brush["distance"] = settings.paint_distance
            brush["falloff"] = settings.proximity_falloff
            locations = {}
            for frame in FRAMES:
                bpy.context.scene.frame_set(frame)
                bpy.context.view_layer.update()
                locations[str(frame)] = list(obj.location)
            brush["locations"] = locations
        values.append(brush)
    return values


def create_scene(root):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 5
    scene.use_gravity = False

    dry_collection, dry_brushes = make_brush_collection(scene, "DryBrushes", True)
    dry_canvas, dry_surface = add_canvas(
        scene,
        "WetnessCanvas",
        dry_collection,
        drying=True,
        dissolving=False,
        output_a="PaintColor",
        output_b="PaintWetness",
    )
    if dry_surface.use_output_b:
        raise RuntimeError("the Output B layer toggle unexpectedly changed use_output_b")

    dissolve_collection, dissolve_brushes = make_brush_collection(scene, "DissolveBrushes", False)
    dissolve_canvas, dissolve_surface = add_canvas(
        scene,
        "DissolveOnlyCanvas",
        dissolve_collection,
        drying=False,
        dissolving=True,
        output_a="DissolveColor",
        output_b="DissolveWetness",
    )
    if dissolve_surface.use_output_b:
        raise RuntimeError("the Output B layer toggle unexpectedly changed use_output_b")

    flag_canvas, flag_surface = add_canvas(
        scene,
        "FlagOnlyCanvas",
        dry_collection,
        drying=False,
        dissolving=False,
        output_a="FlagOnlyColor",
        output_b="FlagOnlyWetness",
        flag_only=True,
    )
    if flag_canvas.data.color_attributes.get("FlagOnlyWetness") is not None:
        raise RuntimeError("the flag-only test unexpectedly created the wetness layer")
    no_output_canvas, no_output_surface = add_canvas(
        scene,
        "NoOutputCanvas",
        dry_collection,
        drying=False,
        dissolving=False,
        output_a="NoOutputColor",
        output_b="NoOutputWetness",
        output_a_toggle=False,
    )
    if no_output_canvas.data.color_attributes.get("NoOutputColor") is not None:
        raise RuntimeError("the no-output test unexpectedly created output A")

    volume_collection, volume_brushes = make_volume_brush_collection(scene)
    volume_canvas, volume_surface = add_canvas(
        scene,
        "VolumePaintCanvas",
        volume_collection,
        drying=False,
        dissolving=False,
        output_a="VolumePaintColor",
        output_b="VolumePaintWetness",
    )
    volume_surface.use_output_b = True
    volume_shape_cases = {}
    for case_key, shape in [
        ("rotated_cube", "rotated_cube"),
        ("torus", "torus"),
        ("open_cube", "open_cube"),
    ]:
        shape_collection, shape_brushes = make_volume_shape_brush_collection(
            scene, "Volume" + case_key.title(), shape
        )
        shape_canvas, shape_surface = add_canvas(
            scene,
            "Volume" + case_key.title() + "Canvas",
            shape_collection,
            drying=False,
            dissolving=False,
            output_a="Volume" + case_key.title() + "Color",
            output_b="Volume" + case_key.title() + "Wetness",
            vertices=VOLUME_VERTICES,
            faces=VOLUME_FACES,
        )
        positions, aabb_mask = volume_shape_aabb_mask(shape_canvas, shape_brushes[0][0])
        volume_shape_cases[case_key] = {
            "canvas": shape_canvas.name,
            "surface": surface_values(shape_surface),
            "brushes": brush_values(shape_brushes),
            "positions": positions,
            "aabb_mask": aabb_mask,
        }


    scene.frame_set(1)
    bpy.context.view_layer.update()
    bpy.ops.wm.save_as_mainfile(
        filepath=os.path.join(root, "DynamicPaintWetness.blend"), relative_remap=True
    )

    return {
        "paint_wetness": {
            "canvas": dry_canvas.name,
            "surface": surface_values(dry_surface),
            "brushes": brush_values(dry_brushes),
            "frames": capture_canvas(scene, dry_canvas, "PaintColor", "PaintWetness"),
        },
        "dissolve_only": {
            "canvas": dissolve_canvas.name,
            "surface": surface_values(dissolve_surface),
            "brushes": brush_values(dissolve_brushes),
            "frames": capture_canvas(scene, dissolve_canvas, "DissolveColor", "DissolveWetness"),
        },
        "flag_only": {
            "canvas": flag_canvas.name,
            "surface": surface_values(flag_surface),
            "frames": capture_canvas(scene, flag_canvas, "FlagOnlyColor", "FlagOnlyWetness"),
        },
        "volume_paint": {
            "canvas": volume_canvas.name,
            "surface": surface_values(volume_surface),
            "brushes": brush_values(volume_brushes),
            "frames": capture_canvas(
                scene, volume_canvas, "VolumePaintColor", "VolumePaintWetness"
            ),
        },
        "no_output": {
            "canvas": no_output_canvas.name,
            "surface": surface_values(no_output_surface),
            "brushes": brush_values(dry_brushes),
            "frames": capture_canvas(
                scene, no_output_canvas, "NoOutputColor", "NoOutputWetness"
            ),
        },
        "volume_shapes": {
            case_key: {
                **case,
                "frames": capture_canvas(
                    scene,
                    bpy.data.objects[case["canvas"]],
                    case["surface"]["output_name_a"],
                    case["surface"]["output_name_b"],
                ),
            }
            for case_key, case in volume_shape_cases.items()
        },

    }


def main(root, capture_path=None):
    os.makedirs(root, exist_ok=True)
    oracle = {
        "blender_version": bpy.app.version_string,
        **create_scene(root),
    }
    for path in [os.path.join(root, "blender_expected.json"), capture_path]:
        if path is not None:
            with open(path, "w", encoding="utf-8") as output:
                json.dump(oracle, output, separators=(",", ":"))


if __name__ == "__main__":
    arguments = sys.argv[sys.argv.index("--") + 1 :]
    main(
        os.path.realpath(arguments[0]),
        os.path.realpath(arguments[1]) if len(arguments) > 1 else None,
    )
