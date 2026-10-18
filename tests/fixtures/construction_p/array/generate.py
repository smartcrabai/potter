"""Generate Array modifier parity cases with Blender 5.2.2 as the oracle.

Run with Blender 5.2.2 in background mode:
  Blender -b --factory-startup -P generate.py -- <output-directory>
"""
import json
import os
import sys

import math
import struct

import bpy
from mathutils import Euler, Vector

assert bpy.app.version == (5, 2, 2), f"need Blender 5.2.2, got {bpy.app.version}"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else HERE
os.makedirs(OUT, exist_ok=True)

# All input coordinates and UVs are multiples of 1/1024; offset-boundary values are f32 exact.
BASE = {
    "vertices": [[0, 0, 0], [1, 0, 0], [1.25, 1, 0], [0.125, 1.125, 0],
                 [0.25, 0.125, 0.75]],
    "faces": [[0, 3, 2, 1], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
    "materials": [0, 1, 2, 1, 0],
    "uv_layers": [
        {"name": "UVMap", "values": [[0, 0], [0.125, 0], [0.25, 0.125], [0, 0.25],
                                         [0.125, 0.125], [0.375, 0.125], [0.5, 0.25],
                                         [0.625, 0.375], [0.75, 0.5], [0.875, 0.625],
                                         [1, 0.75], [1.125, 0.875], [1.25, 1], [1.375, 1.125],
                                         [1.5, 1.25], [1.625, 1.375]]},
        {"name": "UV2", "values": [[0.25, 0.125], [0.375, 0.25], [0.5, 0.375], [0.625, 0.5],
                                      [0.75, 0.625], [0.875, 0.75], [1, 0.875], [1.125, 1],
                                      [1.25, 1.125], [1.375, 1.25], [1.5, 1.375], [1.625, 1.5],
                                      [1.75, 1.625], [1.875, 1.75], [2, 1.875], [2.125, 2]]},
    ],
    "groups": {"Root": [1.0, 0.5, None, 0.25, 0.75],
               "Tip": [None, 0.25, 1.0, None, 0.5]},
    "colors": [
        {"name": "PointTint", "domain": "POINT", "data_type": "BYTE_COLOR",
         "values": [[(i + 1) / 8, (i + 2) / 16, (i + 3) / 32, (i + 4) / 8]
                    for i in range(5)]},
        {"name": "CornerTint", "domain": "CORNER", "data_type": "FLOAT_COLOR",
         "values": [[(i + 1) / 32, (i + 3) / 64, (i + 5) / 128, 1.0]
                    for i in range(16)]},
    ],
    "color_active": "CornerTint",
    "color_render": "PointTint",
}
NORM_BASE = dict(BASE)
NORM_BASE["face_smooth"] = [True, False, True, False, True]
NORM_BASE["custom_normals"] = [
    [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]],
    [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
    [[0, 0, 1], [1, 1, 0], [0, 1, 0]],
    [[1, 0, 0], [0, 0, 1], [1, 1, 0]],
    [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
]


BOUNDARY = {
    "vertices": [[0, 0, 0], [1.125, 0, 0], [1.375, 1, 0], [0.25, 1.25, 0.75]],
    "faces": [[0, 1, 2, 3]],
    "materials": [1],
    "uv_layers": [{"name": "UVMap", "values": [[0, 0], [0.125, 0], [0.25, 0.25], [0.375, 0.5]]}],
    "groups": {"Root": [1.0, 0.25, 0.5, None]},
}
SMOOTH_SPLIT = {
    "vertices": [[0, 0, 0], [-1, 1, 0], [-2, 0, 0], [0, 0, 0], [1, -1, 0], [2, 0, 0]],
    "faces": [[0, 1, 2, 3, 4, 5]],
    "materials": [0],
    "uv_layers": [],
    "groups": {},
    "face_smooth": [True],
}


EMPTY = {"vertices": [], "faces": [], "materials": [], "uv_layers": [], "groups": {}}
EMPTY_CAP = {
    "vertices": [[0, 0, 0], [0, 1, 0], [0, 0, 1]],
    "faces": [[0, 1, 2]],
    "materials": [0],
    "uv_layers": [{"name": "CapUV", "values": [[0, 0], [1, 0], [0, 1]]}],
    "groups": {},
}


def float32(value):
    return struct.unpack("=f", struct.pack("=f", value))[0]


def repeated_f32_translation(step, count):
    """Accumulate f32 translations exactly, skipping only power-of-two increments."""
    step = float32(step)
    assert step
    if math.frexp(abs(step))[0] != 0.5:
        value = 0.0
        for _ in range(count):
            next_value = float32(value + step)
            if next_value == value:
                break
            value = next_value
        return value
    value = 0.0
    remaining = count
    increment = abs(step)
    while remaining:
        if float32(value + step) == value:
            break
        magnitude = abs(value)
        boundary = increment if magnitude < increment else 2.0 ** (
            math.floor(math.log2(magnitude)) + 1
        )
        steps_to_boundary = round((boundary - magnitude) / increment)
        jump = min(remaining, steps_to_boundary)
        if jump == steps_to_boundary:
            value = math.copysign(boundary, step)
        else:
            value = float32(value + step * jump)
        remaining -= jump
    return value


def xform(translation=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1)):
    q = Euler(rotation, 'XYZ').to_quaternion()
    # Potter Transform uses xyzw quaternion ordering.
    return {"translation": list(translation), "rotation": [q.x, q.y, q.z, q.w], "scale": list(scale)}


def transform_dump(obj):
    quaternion = obj.rotation_quaternion
    return {
        "translation": [float(value) for value in obj.location],
        "rotation": [float(quaternion.x), float(quaternion.y), float(quaternion.z), float(quaternion.w)],
        "scale": [float(value) for value in obj.scale],
    }


def color_dump(mesh):
    layers = [
        {
            "name": attr.name,
            "domain": attr.domain,
            "data_type": attr.data_type,
            "values": [list(item.color) for item in attr.data],
        }
        for attr in mesh.color_attributes
    ]
    attributes = list(mesh.color_attributes)
    active = attributes[mesh.color_attributes.active_color_index].name if attributes else None
    render = attributes[mesh.color_attributes.render_color_index].name if attributes else None
    return {
        "colors": layers,
        "color_layers": {
            "layers": [attr.name for attr in attributes],
            "active": active,
            "render": render,
        },
    }

def uv_metadata(mesh):
    layers = list(mesh.uv_layers)
    active_index = mesh.uv_layers.active_index if layers else None
    render_index = mesh.uv_layers.active_render_index if layers else None
    return {
        "uv_active": layers[active_index].name if active_index is not None else None,
        "uv_render": layers[render_index].name if render_index is not None else None,
    }


def add_material_slots(obj, count):
    for index in range(count):
        mat = bpy.data.materials.new(f"{obj.name}_slot_{index}")
        obj.data.materials.append(mat)


def make_mesh_object(name, spec, transform=None, material_slots=3, groups_order=None):
    me = bpy.data.meshes.new(f"{name}_mesh")
    me.from_pydata(spec["vertices"], [], spec["faces"])
    me.update()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    if transform:
        obj.location = transform["translation"]
        obj.rotation_mode = 'QUATERNION'
        obj.rotation_quaternion = (transform["rotation"][3], *transform["rotation"][:3])
        obj.scale = transform["scale"]
    if "face_smooth" in spec:
        for poly, smooth in zip(me.polygons, spec["face_smooth"]):
            poly.use_smooth = smooth
    if spec.get("custom_normals") is not None:
        normals = [normal for face in spec["custom_normals"] for normal in face]
        me.normals_split_custom_set(normals)
    add_material_slots(obj, material_slots)
    for poly, material in zip(me.polygons, spec.get("materials", [])):
        poly.material_index = material
    for layer_spec in spec.get("uv_layers", []):
        layer = me.uv_layers.new(name=layer_spec["name"])
        for corner, value in zip(layer.data, layer_spec["values"]):
            corner.uv = value
    for color in spec.get("colors", []):
        attr = me.color_attributes.new(
            name=color["name"], type=color["data_type"], domain=color["domain"]
        )
        for item, value in zip(attr.data, color["values"]):
            item.color = value
    if me.color_attributes:
        names = [attr.name for attr in me.color_attributes]
        me.color_attributes.active_color_index = names.index(spec.get("color_active", names[0]))
        me.color_attributes.render_color_index = names.index(spec.get("color_render", names[0]))
    order = groups_order or list(spec.get("groups", {}).keys())
    groups = {}
    for name in order:
        groups[name] = obj.vertex_groups.new(name=name)
        weights = spec.get("groups", {}).get(name, [])
        for index, weight in enumerate(weights):
            if weight is not None:
                groups[name].add([index], float(weight), 'REPLACE')
    return obj


def make_empty(name, transform):
    obj = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = transform["translation"]
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = (transform["rotation"][3], *transform["rotation"][:3])
    obj.scale = transform["scale"]
    return obj


def output_mesh(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    try:
        layers = []
        for layer in mesh.uv_layers:
            values = []
            for poly in mesh.polygons:
                values.append([[float(mesh.uv_layers[layer.name].data[loop].uv[axis])
                                for axis in range(2)]
                               for loop in range(poly.loop_start, poly.loop_start + poly.loop_total)])
            layers.append({"name": layer.name, "faces": values})
        groups = {}
        group_names = {group.index: group.name for group in evaluated.vertex_groups}
        for name in group_names.values():
            groups[name] = []
        for vertex in mesh.vertices:
            weights = {group_names[item.group]: float(item.weight)
                       for item in vertex.groups if item.group in group_names}
            for name in groups:
                groups[name].append(weights.get(name))
        return {
            "vertices": [[float(component) for component in vertex.co] for vertex in mesh.vertices],
            "faces": [{"vertices": list(poly.vertices), "material": int(poly.material_index),
                       "smooth": bool(poly.use_smooth)} for poly in mesh.polygons],
            "edges": [list(edge.vertices) for edge in mesh.edges],
            "uv_layers": layers,
            **uv_metadata(mesh),
            "groups": groups,
            **color_dump(mesh),
            "normals": [[[float(component) for component in mesh.corner_normals[index].vector]
                         for index in range(poly.loop_start, poly.loop_start + poly.loop_total)]
                        for poly in mesh.polygons],
        }
    finally:
        evaluated.to_mesh_clear()


def configure_array(obj, params):
    mod = obj.modifiers.new("Array parity", 'ARRAY')
    for key, value in params.items():
        setattr(mod, key, value)
    return mod


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def run_case(case):
    reset()
    owner = make_mesh_object("Body", case["input"], case.get("owner_transform"),
                             case.get("owner_material_slots", 3), case.get("owner_groups_order"))
    objects = {}
    for name, spec in case.get("objects", {}).items():
        objects[name] = make_mesh_object(name, spec["mesh"], spec.get("transform"),
                                         spec.get("material_slots", 3), spec.get("groups_order"))
    for name, transform in case.get("empties", {}).items():
        objects[name] = make_empty(name, transform)
    if "curve" in case:
        curve_spec = case["curve"]
        curve = bpy.data.curves.new("Fit path data", 'CURVE')
        curve.dimensions = '3D'
        curve.use_path = curve_spec.get("use_path", False)
        curve.resolution_u = 1
        spline = curve.splines.new('POLY')
        points = curve_spec["points"]
        spline.points.add(len(points) - 1)
        for point, co in zip(spline.points, points):
            point.co = (*co, 1.0)
        path = bpy.data.objects.new("FitPath", curve)
        bpy.context.scene.collection.objects.link(path)
        transform = curve_spec.get("transform", xform())
        path.location = transform["translation"]
        path.rotation_mode = 'QUATERNION'
        path.rotation_quaternion = (transform["rotation"][3], *transform["rotation"][:3])
        path.scale = transform["scale"]
        objects["FitPath"] = path
    bpy.context.view_layer.update()
    params = dict(case["params"])
    if "probe_count" in case:
        params["count"] = case["probe_count"]
    for key in ("offset_object", "start_cap", "end_cap", "curve"):
        if key in params:
            params[key] = objects[params[key]]
    configure_array(owner, params)
    bpy.context.view_layer.update()
    output = output_mesh(owner)
    # Probe Blender safely; derive the huge pure-translation result from its f32 accumulation.
    if "fixture_count" in case:
        cap = case["objects"]["EndCap"]["mesh"]
        start = len(output["vertices"]) - len(cap["vertices"])
        probe_translation = repeated_f32_translation(
            case["params"]["constant_offset_displace"][0], case["probe_count"]
        )
        for index, vertex in enumerate(cap["vertices"]):
            expected_probe_x = float32(vertex[0] + probe_translation)
            assert output["vertices"][start + index][0] == expected_probe_x
        end_translation = repeated_f32_translation(
            case["params"]["constant_offset_displace"][0], case["fixture_count"]
        )
        for index, vertex in enumerate(cap["vertices"]):
            output["vertices"][start + index][0] = float32(vertex[0] + end_translation)
    input_spec = dict(case["input"])
    input_spec.update(color_dump(owner.data))
    input_spec.update(uv_metadata(owner.data))
    object_specs = {}
    for name, spec in case.get("objects", {}).items():
        object_spec = dict(spec)
        object_spec["transform"] = transform_dump(objects[name])
        mesh_spec = dict(object_spec["mesh"])
        mesh_spec.update(color_dump(objects[name].data))
        mesh_spec.update(uv_metadata(objects[name].data))
        object_spec["mesh"] = mesh_spec
        object_specs[name] = object_spec
    empty_specs = {
        name: transform_dump(objects[name])
        for name in case.get("empties", {})
    }
    curve_spec = case.get("curve")
    if curve_spec is not None:
        curve_spec = dict(curve_spec)
        curve_spec["transform"] = transform_dump(objects["FitPath"])
    return {"name": case["name"], "input": input_spec, "params": case["params"],
            "owner_transform": transform_dump(owner),
            "owner_material_slots": case.get("owner_material_slots", 3),
            "objects": object_specs, "empties": empty_specs,
            "curve": curve_spec, "expected": output}


# A stepped cap starts one offset before the first chunk and ends one step beyond the last.
CAP_STEP = 2.0
start_cap = {"mesh": {**BASE, "vertices": [[x + CAP_STEP, y, z] for x, y, z in BASE["vertices"]],
                       "uv_layers": [{"name": layer["name"],
                                      "values": [[u + 3, v + 2] for u, v in layer["values"]]}
                                     for layer in BASE["uv_layers"]],
                       "groups": {"Tip": [0.125, None, 0.5, 1.0, 0.25],
                                  "Root": [0.5, 1.0, None, 0.75, 0.125],
                                  "Unused": [1.0] * 5}},
             "material_slots": 3, "groups_order": ["Tip", "Unused", "Root"],
             "transform": xform((7.0, -5.0, 2.0), (0.25, -0.5, 0.125), (1.5, 0.75, 2.0))}
end_cap = {"mesh": {**BASE, "vertices": [[x - CAP_STEP, y, z] for x, y, z in BASE["vertices"]],
                     "uv_layers": [{"name": layer["name"],
                                    "values": [[u + 5, v + 4] for u, v in layer["values"]]}
                                   for layer in BASE["uv_layers"]],
                     "groups": {"Root": [0.375, 0.625, None, 0.875, 0.25],
                                "Tip": [None, 0.125, 0.75, 0.5, 1.0]}},
           "material_slots": 3, "groups_order": ["Root", "Tip"],
           "transform": xform((-3.0, 4.0, -1.0), (-0.375, 0.25, 0.5), (0.75, 1.25, 0.5))}

# Simulates a persisted source whose corner storage differs from its Blender loop basis.
ORDERED_NORMALS = dict(NORM_BASE)
ORDERED_NORMALS["custom_normal_corner_order"] = [
    [15, 14, 13, 12], [11, 10, 9], [8, 7, 6], [5, 4, 3], [2, 1, 0],
]
normal_start_cap = {
    **start_cap,
    "mesh": {
        **start_cap["mesh"],
        "face_smooth": NORM_BASE["face_smooth"],
        "custom_normals": NORM_BASE["custom_normals"],
    },
    "transform": xform(),
}
normal_end_cap = {
    **end_cap,
    "mesh": {
        **end_cap["mesh"],
        "face_smooth": NORM_BASE["face_smooth"],
        "custom_normals": NORM_BASE["custom_normals"],
    },
    "transform": xform(),
}

CASES = [
    {"name": "offset_composition_uv_layers", "input": BASE,
     "params": {"count": 3, "use_relative_offset": True,
                 "relative_offset_displace": [0.5, -0.25, 0.125],
                 "use_constant_offset": True, "constant_offset_displace": [0.25, 0.125, -0.125],
                 "offset_u": 0.125, "offset_v": -0.25}},
    {"name": "object_offset_owner_and_target_transforms", "input": BASE,
     "owner_transform": xform((1.25, -0.5, 2.0), (0.25, -0.5, 0.375), (1.5, 0.75, 2.0)),
     "empties": {"Offset": xform((-2.0, 1.25, 0.5), (-0.375, 0.25, 0.5), (0.75, 1.25, 0.5))},
     "params": {"count": 3, "use_relative_offset": False, "use_constant_offset": True,
                 "constant_offset_displace": [0.25, -0.125, 0.375],
                 "use_object_offset": True, "offset_object": "Offset"}},
    {"name": "custom_normals_follow_chunk_transform", "input": NORM_BASE,
     "owner_transform": xform((1.25, -0.5, 2.0), (0.25, -0.5, 0.375), (1.5, 0.75, 2.0)),
     "empties": {"Offset": xform((-2.0, 1.25, 0.5), (-0.375, 0.25, 0.5), (0.75, 1.25, 0.5))},
     "params": {"count": 3, "use_relative_offset": False, "use_constant_offset": True,
                 "constant_offset_displace": [0.25, -0.125, 0.375],
                 "use_object_offset": True, "offset_object": "Offset"}},
    {"name": "custom_normals_nonidentity_order_multiple_copies", "input": ORDERED_NORMALS,
     "params": {"count": 3, "use_relative_offset": False, "use_constant_offset": True,
                 "constant_offset_displace": [0.25, -0.125, 0.375]}},
    {"name": "custom_normals_nonidentity_order_caps", "input": ORDERED_NORMALS,
     "objects": {"StartCap": normal_start_cap, "EndCap": normal_end_cap},
     "params": {"count": 2, "use_relative_offset": False, "use_constant_offset": True,
                 "constant_offset_displace": [CAP_STEP, 0.0, 0.0],
                 "start_cap": "StartCap", "end_cap": "EndCap"}},
    {"name": "custom_normals_nonidentity_order_caps_merge", "input": ORDERED_NORMALS,
     "objects": {"StartCap": normal_start_cap, "EndCap": normal_end_cap},
     "params": {"count": 2, "use_relative_offset": False, "use_constant_offset": True,
                 "constant_offset_displace": [CAP_STEP, 0.0, 0.0], "use_merge_vertices": True,
                 "merge_threshold": 0.0, "start_cap": "StartCap", "end_cap": "EndCap",
                 "use_merge_vertices_cap": True}},
    {"name": "fit_length_exact", "input": BASE,
     "params": {"fit_type": "FIT_LENGTH", "fit_length": 2.0,
                 "use_relative_offset": False, "use_constant_offset": True,
                 "constant_offset_displace": [1.0, 0.0, 0.0]}},
    {"name": "fit_length_below_epsilon", "input": BASE,
     "params": {"fit_type": "FIT_LENGTH", "fit_length": 2.0 - 2.0 ** -20,
                 "use_relative_offset": False, "use_constant_offset": True,
                 "constant_offset_displace": [1.0, 0.0, 0.0]}},
    {"name": "fit_length_below_boundary", "input": BASE,
     "params": {"fit_type": "FIT_LENGTH", "fit_length": 2.0 - 2.0 ** -19,
                 "use_relative_offset": False, "use_constant_offset": True,
                 "constant_offset_displace": [1.0, 0.0, 0.0]}},
    {"name": "fit_length_above_epsilon", "input": BASE,
     "params": {"fit_type": "FIT_LENGTH", "fit_length": 2.0 + 2.0 ** -20,
                 "use_relative_offset": False, "use_constant_offset": True,
                 "constant_offset_displace": [1.0, 0.0, 0.0]}},
    {"name": "merge_threshold_below", "input": BOUNDARY,
     "params": {"count": 2, "use_relative_offset": False, "use_constant_offset": True,
                 "constant_offset_displace": [1.0, 0.0, 0.0], "use_merge_vertices": True,
                 "merge_threshold": 0.125 - 2.0 ** -20}},
    {"name": "merge_split_faces_keep_smooth_default", "input": SMOOTH_SPLIT,
     "params": {"count": 2, "use_relative_offset": False, "use_constant_offset": True,
                 "constant_offset_displace": [0.0, 0.0, 0.0], "use_merge_vertices": True,
                 "merge_threshold": 0.0}},
    {"name": "merge_threshold_at_boundary", "input": BOUNDARY,
     "params": {"count": 2, "use_relative_offset": False, "use_constant_offset": True,
                 "constant_offset_displace": [1.0, 0.0, 0.0], "use_merge_vertices": True,
                 "merge_threshold": 0.125}},
    {"name": "caps_keep_material_groups_and_uvs", "input": BASE,
     "objects": {"StartCap": start_cap, "EndCap": end_cap},
     "params": {"count": 2, "use_relative_offset": False, "use_constant_offset": True,
                 "constant_offset_displace": [CAP_STEP, 0.0, 0.0],
                 "start_cap": "StartCap", "end_cap": "EndCap",
                 "offset_u": 0.25, "offset_v": -0.125}},
    {"name": "caps_merge_material_groups_and_uvs", "input": BASE,
     "objects": {"StartCap": start_cap, "EndCap": end_cap},
     "params": {"count": 2, "use_relative_offset": False, "use_constant_offset": True,
                 "constant_offset_displace": [CAP_STEP, 0.0, 0.0], "use_merge_vertices": True,
                 "merge_threshold": 0.0, "start_cap": "StartCap", "end_cap": "EndCap",
                 "use_merge_vertices_cap": True, "offset_u": 0.25, "offset_v": -0.125}},
    {"name": "count_one", "input": BASE,
     "params": {"count": 1, "use_relative_offset": True,
                 "relative_offset_displace": [0.5, 0.25, -0.125]}},
    {"name": "fit_length_tiny_offset_fallback", "input": BASE,
     "params": {"fit_type": "FIT_LENGTH", "fit_length": 4.0,
                 "use_relative_offset": False, "use_constant_offset": True,
                 "constant_offset_displace": [0.000001, 0.0, 0.0]}},
    {"name": "maximum_rna_count_falls_back", "input": BASE,
     "params": {"count": 2147483647, "use_relative_offset": False, "use_constant_offset": True,
                 "constant_offset_displace": [1.0, 0.0, 0.0]}},
    {"name": "empty_source_huge_count_no_caps", "input": EMPTY,
     "params": {"count": 2147483647, "use_relative_offset": False,
                 "use_constant_offset": False}, "probe_count": 8},
    {"name": "empty_source_huge_count_with_caps", "input": EMPTY,
     "objects": {"StartCap": {"mesh": EMPTY_CAP}, "EndCap": {"mesh": EMPTY_CAP}},
     "params": {"count": 2147483647, "use_relative_offset": False,
                 "use_constant_offset": True, "constant_offset_displace": [1.0, 0.0, 0.0],
                 "use_merge_vertices": True, "merge_threshold": 0.0,
                 "use_merge_vertices_cap": True, "start_cap": "StartCap", "end_cap": "EndCap"},
     "probe_count": 8, "fixture_count": 2147483647},
    {"name": "empty_source_huge_count_with_caps_nonbinary_offset", "input": EMPTY,
     "objects": {"StartCap": {"mesh": EMPTY_CAP}, "EndCap": {"mesh": EMPTY_CAP}},
     "params": {"count": 2147483647, "use_relative_offset": False,
                 "use_constant_offset": True, "constant_offset_displace": [0.75, 0.0, 0.0],
                 "use_merge_vertices": True, "merge_threshold": 0.0,
                 "use_merge_vertices_cap": True, "start_cap": "StartCap", "end_cap": "EndCap"},
     "probe_count": 8, "fixture_count": 2147483647},
    {"name": "empty_source_object_offset_reflection_odd_count", "input": EMPTY,
     "objects": {"StartCap": {"mesh": EMPTY_CAP}, "EndCap": {"mesh": EMPTY_CAP}},
     "empties": {"Offset": xform((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (-1.0, 1.0, 1.0))},
     "params": {"count": 3, "use_relative_offset": False, "use_constant_offset": False,
                 "use_object_offset": True, "offset_object": "Offset",
                 "use_merge_vertices": True, "merge_threshold": 0.0,
                 "use_merge_vertices_cap": True, "start_cap": "StartCap", "end_cap": "EndCap"}},
    {"name": "empty_source_object_offset_reflection_even_count", "input": EMPTY,
     "objects": {"StartCap": {"mesh": EMPTY_CAP}, "EndCap": {"mesh": EMPTY_CAP}},
     "empties": {"Offset": xform((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (-1.0, 1.0, 1.0))},
     "params": {"count": 4, "use_relative_offset": False, "use_constant_offset": False,
                 "use_object_offset": True, "offset_object": "Offset",
                 "use_merge_vertices": True, "merge_threshold": 0.0,
                 "use_merge_vertices_cap": True, "start_cap": "StartCap", "end_cap": "EndCap"}},
    {"name": "empty_source_object_offset_rotated_count", "input": EMPTY,
     "objects": {"StartCap": {"mesh": EMPTY_CAP}, "EndCap": {"mesh": EMPTY_CAP}},
     "empties": {"Offset": xform((0.25, -0.5, 0.75), (0.25, -0.375, 0.5))},
     "params": {"count": 5, "use_relative_offset": False, "use_constant_offset": False,
                 "use_object_offset": True, "offset_object": "Offset",
                 "use_merge_vertices": True, "merge_threshold": 0.0,
                 "use_merge_vertices_cap": True, "start_cap": "StartCap", "end_cap": "EndCap"}},
    {"name": "fit_curve_poly_path", "input": BASE,
     "curve": {"points": [[0, 0, 0], [1, 0.25, 0], [1.25, 1, 0.5]],
               "use_path": True,
               "transform": xform((0.5, -1.0, 2.0), (0.25, 0.125, -0.375), (1.5, 0.75, 2.0))},
     "params": {"fit_type": "FIT_CURVE", "curve": "FitPath", "fit_length": 3.0,
                 "use_relative_offset": False, "use_constant_offset": True,
                 "constant_offset_displace": [0.75, 0.0, 0.0]}},
]

results = [run_case(case) for case in CASES]
path = os.path.join(OUT, "array_parity.json")
with open(path, "w", encoding="utf-8") as stream:
    json.dump({"blender_version": bpy.app.version_string, "cases": results}, stream,
              separators=(",", ":"), allow_nan=False)
print(f"Wrote {path}: {len(results)} cases")
