#!/usr/bin/env python3
"""Blender 5.2.2 oracle fixtures for Build and Wireframe modifier parity."""
import json
import math
import os
import sys

import bpy

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else ROOT
VERSION = "5.2.2 LTS"
assert bpy.app.version == (5, 2, 2), f"need Blender 5.2.2, got {bpy.app.version}"
os.makedirs(OUT, exist_ok=True)

CLOSED = {
    "vertices": [[0, 0, 0], [3, 0, 0], [3, 2, 0], [0, 2, 0],
                 [0.5, 0.25, 2], [2.75, 0.25, 2.25], [2.75, 1.75, 2], [0.5, 1.75, 2.5],
                 [1.5, 1, 3.5]],
    "faces": [[0, 3, 2, 1], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
              [4, 5, 8], [5, 6, 8], [6, 7, 8], [7, 4, 8]],
    "edges": [], "sharp_edges": [[0, 1]], "materials": [0] * 9,
}
OPEN = {
    "vertices": [[0, 0, 0], [2, 0, 0], [3, 1, 0.25], [2, 2, 0.5], [0, 2, 0],
                 [1, 0.75, 1], [3.5, 2.5, 1.5]],
    "faces": [[0, 1, 5], [1, 2, 5], [2, 3, 5], [3, 4, 5], [4, 0, 5]],
    "edges": [[0, 6], [1, 3]], "sharp_edges": [[0, 1]], "materials": [0] * 5,
}
EDGE_ONLY = {
    "vertices": [[i * 0.25, (i % 5) * 0.125, ((i * 3) % 7) * 0.0625] for i in range(27)],
    "faces": [], "edges": [[i, i + 1] for i in range(26)], "materials": [],
}
POINTS = {
    "vertices": [[i * 0.25, (i % 4) * 0.125, ((i * 5) % 9) * 0.0625] for i in range(17)],
    "faces": [], "edges": [], "materials": [],
}
PARTIAL = {"Weights": {"0": 0.25, "1": 0.5, "2": 0.75, "4": 1.0, "5": 0.125},
           "Other": {"3": 0.5}}
EMPTY_NO_DVERTS = {"Empty": {}}
EMPTY_WITH_DVERTS = {"Empty": {}, "Other": {"2": 0.5}}
TRANSFORM = {"location": [3.25, -2.5, 1.75], "rotation": [0.25, -0.5, 0.75], "scale": [1.5, 0.75, 2.0]}


def attrs(spec):
    values = [[(corner + 1) / 32.0, ((corner * 3) % 11) / 16.0]
              for corner in range(sum(len(face) for face in spec["faces"]))]
    uv_faces = []
    start = 0
    for face_index, face in enumerate(spec["faces"]):
        end = start + len(face)
        uv_faces.append({
            "face_id": face_index,
            "uv": [[u, 1.0 - v] for u, v in values[start:end]],
        })
        start = end
    colors = [[((i + 1) % 7) / 8.0, ((i * 3 + 2) % 9) / 16.0,
               ((i * 5 + 1) % 11) / 16.0, 1.0] for i in range(len(values))]
    color_rows = {}
    start = 0
    for face_index, face in enumerate(spec["faces"]):
        end = start + len(face)
        color_rows[f"f{face_index}"] = colors[start:end]
        start = end
    return {
        "uv_layers": {
            "layers": [{"name": "UVMap", "faces": uv_faces}],
            "active": "UVMap",
            "render": "UVMap",
        },
        "color_layers": {"layers": ["Col"], "active": "Col", "render": "Col"},
        "blender_metadata": {"color_data_types": {"Col": "FLOAT_COLOR"}},
        "Col": {"domain": "corner", "type": "color", "values": color_rows},
    }


def reset():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for data in (bpy.data.meshes, bpy.data.materials):
        for item in list(data):
            if item.users == 0:
                data.remove(item)


def make_object(spec, groups, slot_count):
    mesh = bpy.data.meshes.new("InputMesh")
    mesh.from_pydata(spec["vertices"], spec["edges"], spec["faces"])
    mesh.update()
    for index in range(slot_count):
        mesh.materials.append(bpy.data.materials.new(name=f"WireframeSlot{index}"))
    sharp_lookup = {tuple(sorted(edge.vertices)): edge.index for edge in mesh.edges}
    for pair in spec.get("sharp_edges", []):
        mesh.edges[sharp_lookup[tuple(sorted(pair))]].use_edge_sharp = True
    mesh.update()
    attributes = attrs(spec)
    uv_layer_spec = attributes["uv_layers"]["layers"][0]
    uv_layer = mesh.uv_layers.new(name=uv_layer_spec["name"])
    for face_index, polygon in enumerate(mesh.polygons):
        for corner, value in enumerate(uv_layer_spec["faces"][face_index]["uv"]):
            uv_layer.data[polygon.loop_start + corner].uv = (value[0], 1.0 - value[1])
    mesh.uv_layers.active_index = 0
    mesh.uv_layers.active_render_index = 0
    color_data = attributes["Col"]["values"]
    color = mesh.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="CORNER")
    for face_index, polygon in enumerate(mesh.polygons):
        row = color_data[f"f{face_index}"]
        for corner, value in enumerate(row):
            color.data[polygon.loop_start + corner].color = value
    mesh.color_attributes.active_color_index = 0
    mesh.color_attributes.render_color_index = 0
    ob = bpy.data.objects.new("Subject", mesh)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = TRANSFORM["location"]
    ob.rotation_euler = TRANSFORM["rotation"]
    ob.scale = TRANSFORM["scale"]
    for name, weights in groups.items():
        group = ob.vertex_groups.new(name=name)
        for index, weight in weights.items():
            group.add([int(index)], weight, "REPLACE")
    return ob


def output_mesh(ob):
    evaluated = ob.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=bpy.context.evaluated_depsgraph_get())
    try:
        vertices = [[float(c) for c in v.co] for v in mesh.vertices]
        faces = [list(poly.vertices) for poly in mesh.polygons]
        materials = [int(poly.material_index) for poly in mesh.polygons]
        edges = [sorted([int(edge.vertices[0]), int(edge.vertices[1])]) for edge in mesh.edges]
        sharp_attr = mesh.attributes.get("sharp_edge")
        sharp = [bool(item.value) for item in sharp_attr.data] if sharp_attr else [False] * len(edges)
        crease_attr = mesh.attributes.get("crease_edge")
        crease = [float(item.value) for item in crease_attr.data] if crease_attr else [0.0] * len(edges)
        uv = mesh.uv_layers.get("UVMap")
        uv_values = [[float(x) for x in item.uv] for item in uv.data] if uv else []
        uv_layers = list(mesh.uv_layers)
        uv_layer_name = lambda index: uv_layers[index].name if 0 <= index < len(uv_layers) else None
        uv_metadata = {
            "layers": [layer.name for layer in uv_layers],
            "active": uv_layer_name(mesh.uv_layers.active_index),
            "render": uv_layer_name(mesh.uv_layers.active_render_index),
        }
        color_attr = mesh.color_attributes.get("Col")
        colors = [[float(x) for x in item.color] for item in color_attr.data] if color_attr else []
        color_attributes = list(mesh.color_attributes)
        color_layers = {
            "layers": [attr.name for attr in color_attributes],
            "active": color_attributes[mesh.color_attributes.active_color_index].name if color_attributes else None,
            "render": color_attributes[mesh.color_attributes.render_color_index].name if color_attributes else None,
        }
        groups_by_index = {group.index: group.name for group in ob.vertex_groups}
        groups = {name: [0.0] * len(mesh.vertices) for name in groups_by_index.values()}
        for vertex in mesh.vertices:
            for assignment in vertex.groups:
                name = groups_by_index.get(assignment.group)
                if name is not None:
                    groups[name][vertex.index] = float(assignment.weight)
        return {"vertices": vertices, "faces": faces, "materials": materials,
                "edges": edges, "sharp": sharp, "crease": crease,
                "uv": uv_values, "uv_metadata": uv_metadata,
                "colors": colors, "color_layers": color_layers,
                "color_data_type": color_attr.data_type if color_attr else None,
                "groups": groups,
                "counts": {"vertices": len(vertices), "edges": len(edges), "faces": len(faces),
                           "corners": len(mesh.loops)}}
    finally:
        evaluated.to_mesh_clear()


def run_build(name, spec, params, frames, groups=None):
    reset()
    groups = groups or {}
    ob = make_object(spec, groups, 0)
    mod = ob.modifiers.new("Build", "BUILD")
    for key, value in params.items():
        setattr(mod, key, value)
    readback = {key: getattr(mod, key) for key in
                ("frame_start", "frame_duration", "use_reverse", "use_random_order", "seed")}
    cases = []
    for frame in frames:
        base = math.floor(frame)
        bpy.context.scene.frame_set(base, subframe=frame - base)
        cases.append({"frame": frame, "output": output_mesh(ob)})
    return {"name": name, "modifier": {"type": "build", "params": readback},
            "mesh": dict(spec, attributes=attrs(spec), groups=groups), "groups": groups,
            "material_slots": 0, "transform": TRANSFORM, "cases": cases}


def run_wireframe(name, spec, params, groups=None, slot_count=3):
    reset()
    groups = groups or {}
    ob = make_object(spec, groups, slot_count)
    mod = ob.modifiers.new("Wireframe", "WIREFRAME")
    defaults = {"thickness": 0.02, "offset": 0.0, "use_boundary": False, "use_replace": True,
                "use_even_offset": True, "use_relative_offset": False, "use_crease": False,
                "crease_weight": 1.0, "material_offset": 0, "vertex_group": "",
                "invert_vertex_group": False, "thickness_vertex_group": 0.0}
    defaults.update(params)
    for key, value in defaults.items():
        setattr(mod, key, value)
    readback = {key: getattr(mod, key) for key in defaults}
    bpy.context.scene.frame_set(1)
    return {"name": name, "modifier": {"type": "wireframe", "params": readback},
            "mesh": dict(spec, attributes=attrs(spec), groups=groups), "groups": groups,
            "material_slots": slot_count, "transform": TRANSFORM,
            "cases": [{"frame": 1.0, "output": output_mesh(ob)}]}


BUILD_CASES = [
    run_build("face_sequential_frames", CLOSED,
              {"frame_start": 1.0, "frame_duration": 9.0, "use_random_order": False},
              [0.0, 1.0, 1.5, 2.0, 5.0, 9.0, 10.0, 11.0]),
    run_build("face_random_seed_1", CLOSED,
              {"frame_start": 1.0, "frame_duration": 8.0, "use_random_order": True, "seed": 1},
              [2.0, 3.75, 5.0, 9.0]),
    run_build("face_reverse_random_seed_7", CLOSED,
              {"frame_start": 1.0, "frame_duration": 8.0, "use_reverse": True,
               "use_random_order": True, "seed": 7}, [1.0, 3.75, 5.0, 9.0]),
    run_build("face_random_seed_max", CLOSED,
              {"frame_start": 1.0, "frame_duration": 8.0, "use_random_order": True,
               "seed": 1048574}, [4.0, 7.0]),
    run_build("edge_f32_boundary_seed_7", EDGE_ONLY,
              {"frame_start": 1.0, "frame_duration": 26.0, "use_random_order": True, "seed": 7},
              [1.0, 16.0, 27.0]),
    run_build("edge_reverse_random_seed_1048574", EDGE_ONLY,
              {"frame_start": 1.0, "frame_duration": 26.0, "use_reverse": True,
               "use_random_order": True, "seed": 1048574}, [1.0, 14.0, 27.0]),
    run_build("point_random_seed_7", POINTS,
              {"frame_start": 1.0, "frame_duration": 16.0, "use_random_order": True, "seed": 7},
              [1.0, 2.0, 9.5, 17.0]),
    run_build("point_reverse_random_seed_1", POINTS,
              {"frame_start": 1.0, "frame_duration": 16.0, "use_reverse": True,
               "use_random_order": True, "seed": 1}, [1.0, 9.0, 17.0]),
]

WIREFRAME_CASES = [
    run_wireframe("closed_default", CLOSED, {"thickness": 0.125}),
    run_wireframe("closed_offset_even_off", CLOSED,
                  {"thickness": 0.25, "offset": 0.375, "use_even_offset": False}),
    run_wireframe("closed_replace_off", CLOSED,
                  {"thickness": 0.125, "use_replace": False, "use_boundary": False}),
    run_wireframe("closed_negative_thickness", CLOSED,
                  {"thickness": -0.125, "offset": -0.25, "use_even_offset": True}),
    run_wireframe("open_boundary_even_crease", OPEN,
                  {"thickness": 0.125, "offset": -0.25, "use_boundary": True,
                   "use_even_offset": True, "use_crease": True, "crease_weight": 0.625}),
    run_wireframe("open_boundary_even_off", OPEN,
                  {"thickness": 0.1875, "offset": 0.25, "use_boundary": True,
                   "use_even_offset": False}),
    run_wireframe("open_relative_loose_edges", OPEN,
                  {"thickness": 0.125, "offset": 0.125, "use_replace": False,
                   "use_relative_offset": True, "use_boundary": True}),
    run_wireframe("group_partial", OPEN,
                  {"thickness": 0.125, "vertex_group": "Weights"}, PARTIAL),
    run_wireframe("group_partial_inverted", OPEN,
                  {"thickness": 0.125, "vertex_group": "Weights", "invert_vertex_group": True}, PARTIAL),
    run_wireframe("group_thickness_blend", OPEN,
                  {"thickness": 0.125, "vertex_group": "Weights", "thickness_vertex_group": 0.5}, PARTIAL),
    run_wireframe("group_empty_no_dverts_noninverted", OPEN,
                  {"thickness": 0.125, "vertex_group": "Empty"}, EMPTY_NO_DVERTS),
    run_wireframe("group_empty_with_dverts_noninverted", OPEN,
                  {"thickness": 0.125, "vertex_group": "Empty"}, EMPTY_WITH_DVERTS),
    run_wireframe("group_empty_no_dverts_inverted", OPEN,
                  {"thickness": 0.125, "vertex_group": "Empty", "invert_vertex_group": True}, EMPTY_NO_DVERTS),
    run_wireframe("group_empty_with_dverts_inverted", OPEN,
                  {"thickness": 0.125, "vertex_group": "Empty", "invert_vertex_group": True}, EMPTY_WITH_DVERTS),
    run_wireframe("group_missing_name", OPEN,
                  {"thickness": 0.125, "vertex_group": "Missing", "invert_vertex_group": True}, PARTIAL),
    run_wireframe("group_missing_name_noninverted", OPEN,
                  {"thickness": 0.125, "vertex_group": "Missing"}, PARTIAL),
    run_wireframe("group_relative_thickness_blend", OPEN,
                  {"thickness": 0.125, "use_relative_offset": True, "use_boundary": True,
                   "vertex_group": "Weights", "invert_vertex_group": True,
                   "thickness_vertex_group": 0.25}, PARTIAL),
    run_wireframe("group_all_wireframe_controls", OPEN,
                  {"thickness": 0.1875, "offset": -0.25, "use_boundary": True,
                   "use_replace": False, "use_even_offset": False, "use_relative_offset": True,
                   "use_crease": True, "crease_weight": 0.375, "vertex_group": "Weights",
                   "thickness_vertex_group": 0.25}, PARTIAL),
    run_wireframe("material_offset_plus_five", OPEN,
                  {"thickness": 0.125, "material_offset": 5}),
    run_wireframe("material_offset_minus_one", OPEN,
                  {"thickness": 0.125, "material_offset": -1}),
    run_wireframe("material_offset_zero_preserves", OPEN,
                  {"thickness": 0.125, "material_offset": 0}, slot_count=0),
    run_wireframe("material_offset_plus_five_no_slots", OPEN,
                  {"thickness": 0.125, "material_offset": 5}, slot_count=0),
]


def write(name, variants):
    document = {"blender_version": VERSION, "variants": variants}
    for path in {os.path.join(ROOT, name + ".json"), os.path.join(OUT, name + ".json")}:
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=1)
            stream.write("\n")
    for variant in variants:
        for case in variant["cases"]:
            output = case["output"]
            print(f"[PROBE] {name}/{variant['name']} frame={case['frame']}: "
                  f"verts={output['counts']['vertices']} edges={output['counts']['edges']} "
                  f"faces={output['counts']['faces']}")


write("build", BUILD_CASES)
write("wireframe", WIREFRAME_CASES)
