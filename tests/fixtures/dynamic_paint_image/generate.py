"""Generate Blender 5.2.2 Dynamic Paint IMAGE sequence oracles."""
import json
import os
import re
import struct
import sys
import time

import bpy


root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])

SCENARIOS = (
    {
        "id": "point",
        "directory": ".",
        "brush_source": "POINT",
        "antialiasing": False,
        "drying": False,
        "dissolve": False,
        "uv_islands": False,
        "output_name_a": "dp_paintmap",
    },
    {
        "id": "volume_antialias_on",
        "directory": "scenarios/volume_antialias_on",
        "brush_source": "VOLUME",
        "antialiasing": True,
        "drying": False,
        "dissolve": False,
        "uv_islands": False,
        "output_name_a": "dp_paintmap##",
    },
    {
        "id": "volume_antialias_off",
        "directory": "scenarios/volume_antialias_off",
        "brush_source": "VOLUME",
        "antialiasing": False,
        "drying": False,
        "dissolve": False,
        "uv_islands": False,
        "output_name_a": "dp_paintmap",
    },
    {
        "id": "drying_dissolve",
        "directory": "scenarios/drying_dissolve",
        "brush_source": "POINT",
        "antialiasing": False,
        "drying": True,
        "dissolve": True,
        "uv_islands": False,
        "output_name_a": "dp_paintmap",
    },
    {
        "id": "uv_islands_margin",
        "directory": "scenarios/uv_islands_margin",
        "brush_source": "POINT",
        "antialiasing": False,
        "drying": False,
        "dissolve": False,
        "uv_islands": True,
        "output_name_a": "dp_paintmap",
    },
)


def frame_filename(output_name, frame):
    matches = list(re.finditer(r"#+", output_name))
    if matches:
        match = matches[-1]
        frame_width = match.end() - match.start()
        frame_text = ("-" if frame < 0 else "") + str(abs(frame)).zfill(frame_width)
        return output_name[:match.start()] + frame_text + output_name[match.end():] + ".png"
    frame_text = ("-" if frame < 0 else "") + str(abs(frame)).zfill(4)
    return output_name + frame_text + ".png"


def make_mesh(scene, use_uv_islands):
    mesh = bpy.data.meshes.new("ImageCanvasMesh")
    if use_uv_islands:
        # Two disconnected canvas quads map to separate UV islands with a two-pixel margin.
        vertices = [
            (-1.0, -1.0, 0.0), (-0.1, -1.0, 0.0), (-0.1, 1.0, 0.0), (-1.0, 1.0, 0.0),
            (0.1, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (0.1, 1.0, 0.0),
        ]
        faces = [(0, 1, 2, 3), (4, 5, 6, 7)]
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        uv = mesh.uv_layers.new(name="UVMap")
        regions = ((0.20, 0.43), (0.57, 0.80))
        for polygon, (uv_min_x, uv_max_x) in zip(mesh.polygons, regions):
            for loop_index in polygon.loop_indices:
                x, y, _ = mesh.vertices[mesh.loops[loop_index].vertex_index].co
                x_min, x_max = ((-1.0, -0.1) if polygon.index == 0 else (0.1, 1.0))
                uv.data[loop_index].uv = (
                    uv_min_x + (x - x_min) * (uv_max_x - uv_min_x) / (x_max - x_min),
                    0.18 + (y + 1.0) * 0.32,
                )
        mesh_info = {
            "uv_layer": "UVMap",
            "uv_island_count": 2,
            "uv_margin_pixels": 2,
            "brush_touches_island_edges": True,
        }
    else:
        # A 16-by-16 quad grid gives the image surface a stable, explicit UV map.
        vertices = [
            (column / 8.0 - 1.0, row / 8.0 - 1.0, 0.0)
            for row in range(17)
            for column in range(17)
        ]
        faces = []
        for row in range(16):
            for column in range(16):
                lower_left = row * 17 + column
                faces.append((lower_left, lower_left + 1, lower_left + 18, lower_left + 17))
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        uv = mesh.uv_layers.new(name="UVMap")
        for polygon in mesh.polygons:
            for loop_index in polygon.loop_indices:
                x, y, _ = mesh.vertices[mesh.loops[loop_index].vertex_index].co
                uv.data[loop_index].uv = ((x + 1.0) * 0.5, (y + 1.0) * 0.5)
        mesh_info = {"grid_width": 16, "grid_height": 16, "uv_layer": "UVMap"}
    canvas = bpy.data.objects.new("ImageCanvas", mesh)
    scene.collection.objects.link(canvas)
    return canvas, mesh_info


def generate_scenario(spec):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.filepaths.save_version = 0
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 2
    scene.use_gravity = False

    fixture_dir = os.path.join(root, spec["directory"])
    image_dir = os.path.join(fixture_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(".png"):
            os.remove(os.path.join(image_dir, filename))

    canvas, mesh_info = make_mesh(scene, spec["uv_islands"])
    start_x, end_x = (-1.0, 1.0) if spec["uv_islands"] else (-0.5, 0.5)
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=12,
        ring_count=8,
        radius=0.18,
        location=(start_x, 0.0, 0.15),
    )
    brush = bpy.context.object
    brush.name = "ImageBrush"
    brush_modifier = brush.modifiers.new("ImageBrushSettings", "DYNAMIC_PAINT")
    brush_modifier.ui_type = "BRUSH"
    for selected in bpy.context.selected_objects:
        selected.select_set(False)
    brush.select_set(True)
    bpy.context.view_layer.objects.active = brush
    if bpy.ops.dpaint.type_toggle(type="BRUSH") != {"FINISHED"}:
        raise RuntimeError("Blender did not create the Dynamic Paint brush")
    brush_settings = brush_modifier.brush_settings
    brush_settings.paint_source = spec["brush_source"]
    brush_settings.paint_color = (0.9, 0.15, 0.05)
    brush_settings.paint_alpha = 1.0
    brush_settings.paint_wetness = 1.0
    brush_settings.paint_distance = 0.42
    brush_settings.use_absolute_alpha = True
    for frame, x in ((1, start_x), (2, end_x)):
        brush.location.x = x
        brush.keyframe_insert(data_path="location", frame=frame)

    canvas_modifier = canvas.modifiers.new("ImageCanvasSettings", "DYNAMIC_PAINT")
    canvas_modifier.ui_type = "CANVAS"
    for selected in bpy.context.selected_objects:
        selected.select_set(False)
    canvas.select_set(True)
    bpy.context.view_layer.objects.active = canvas
    if bpy.ops.dpaint.type_toggle(type="CANVAS") != {"FINISHED"}:
        raise RuntimeError("Blender did not create the Dynamic Paint canvas")
    surface = canvas_modifier.canvas_settings.canvas_surfaces[0]
    surface.surface_type = "PAINT"
    surface.surface_format = "IMAGE"
    surface.image_resolution = 16
    surface.image_output_path = image_dir
    surface.image_fileformat = "PNG"
    surface.frame_start = 1
    surface.frame_end = 2
    surface.frame_substeps = 0
    surface.use_antialiasing = spec["antialiasing"]
    surface.use_dissolve = spec["dissolve"]
    surface.dissolve_speed = 250
    surface.use_drying = spec["drying"]
    surface.dry_speed = 500
    surface.output_name_a = spec["output_name_a"]
    surface.use_output_a = True
    surface.output_name_b = "dp_wetmap"
    surface.use_output_b = True

    scene.frame_set(1)
    bpy.context.view_layer.update()
    surface.image_output_path = ""
    blend_path = os.path.join(fixture_dir, "dynamic_paint_image.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    surface.image_output_path = image_dir

    result = bpy.ops.dpaint.bake()
    if "FINISHED" not in result:
        raise RuntimeError("Blender Dynamic Paint IMAGE bake failed: %r" % (result,))

    output_names = (surface.output_name_a, surface.output_name_b)
    expected_paths = [
        os.path.join(image_dir, frame_filename(output_name, frame))
        for output_name in output_names
        for frame in range(1, 3)
    ]
    deadline = time.monotonic() + 30.0
    while not all(os.path.isfile(path) for path in expected_paths):
        if time.monotonic() >= deadline:
            missing = [path for path in expected_paths if not os.path.isfile(path)]
            raise RuntimeError("Blender Dynamic Paint IMAGE bake did not write: %r" % missing)
        time.sleep(0.05)

    outputs = []
    for output_name in output_names:
        for frame in range(1, 3):
            path = os.path.join(image_dir, frame_filename(output_name, frame))
            with open(path, "rb") as stream:
                data = stream.read()
            if data[:8] != b"\x89PNG\r\n\x1a\n":
                raise RuntimeError("Blender did not write a valid PNG: " + path)
            width, height = struct.unpack(">II", data[16:24])
            if (width, height) != (16, 16):
                raise RuntimeError("Unexpected Dynamic Paint image dimensions: %r" % ((width, height),))
            outputs.append(
                {
                    "name": output_name,
                    "frame": frame,
                    "file": os.path.basename(path),
                    "width": width,
                    "height": height,
                }
            )

    capture = {
        "id": spec["id"],
        "directory": spec["directory"],
        "blend": "dynamic_paint_image.blend",
        "image_dir": "images",
        "canvas": canvas.name,
        "brush": brush.name,
        "mesh": mesh_info,
        "brush_source": brush_settings.paint_source,
        "surface": {
            "surface_type": surface.surface_type,
            "surface_format": surface.surface_format,
            "is_active": bool(surface.is_active),
            "image_resolution": int(surface.image_resolution),
            "image_fileformat": surface.image_fileformat,
            "image_output_path": "",
            "frame_start": int(surface.frame_start),
            "frame_end": int(surface.frame_end),
            "use_antialiasing": bool(surface.use_antialiasing),
            "use_drying": bool(surface.use_drying),
            "dry_speed": int(surface.dry_speed),
            "use_dissolve": bool(surface.use_dissolve),
            "dissolve_speed": int(surface.dissolve_speed),
            "output_name_a": surface.output_name_a,
            "use_output_a": bool(surface.use_output_a),
            "output_name_b": surface.output_name_b,
            "use_output_b": bool(surface.use_output_b),
        },
        "outputs": outputs,
    }
    return capture


scenarios = [generate_scenario(spec) for spec in SCENARIOS]
capture = {"blender_version": bpy.app.version_string, "scenarios": scenarios}
with open(os.path.join(root, "capture.json"), "w", encoding="utf-8") as stream:
    json.dump(capture, stream, indent=2, sort_keys=True)
    stream.write("\n")
