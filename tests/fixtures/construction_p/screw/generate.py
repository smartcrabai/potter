# Run with: /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup -P tests/fixtures/construction_p/screw/generate.py -- <scratch-dir>
"""Generate Blender 5.2.2 Screw modifier parity fixtures."""
import json
import math
import os
import struct
import sys

import bpy
from mathutils import Euler

OUT = sys.argv[sys.argv.index("--") + 1]
FIXTURES = os.path.dirname(os.path.abspath(__file__))
assert bpy.app.version == (5, 2, 2), f"need Blender 5.2.2, got {bpy.app.version}"
os.makedirs(OUT, exist_ok=True)


def f32(value):
    return struct.unpack("f", struct.pack("f", float(value)))[0]


def vec(value):
    return [f32(c) for c in value]


def reset():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)
    for material in list(bpy.data.materials):
        bpy.data.materials.remove(material)


def set_transform(obj, value):
    obj.location = value["translation"]
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Euler(
        tuple(math.radians(degree) for degree in value["rotation_deg"]), "XYZ"
    ).to_quaternion()
    obj.scale = value["scale"]
    return {
        "translation": vec(obj.location),
        "rotation": [
            f32(obj.rotation_quaternion.x),
            f32(obj.rotation_quaternion.y),
            f32(obj.rotation_quaternion.z),
            f32(obj.rotation_quaternion.w),
        ],
        "scale": vec(obj.scale),
    }


def color_registry(mesh):
    layers = list(mesh.color_attributes)
    return {
        "layers": [layer.name for layer in layers],
        "active": layers[mesh.color_attributes.active_color_index].name if layers else None,
        "render": layers[mesh.color_attributes.render_color_index].name if layers else None,
    }

def uv_metadata(mesh):
    layers = list(mesh.uv_layers)
    active_index = mesh.uv_layers.active_index if layers else None
    render_index = mesh.uv_layers.active_render_index if layers else None
    return {
        "uv_active": layers[active_index].name if active_index is not None else None,
        "uv_render": layers[render_index].name if render_index is not None else None,
    }


def color_input(mesh):
    return {
        "colors": [
            {
                "name": attr.name,
                "domain": attr.domain,
                "data_type": attr.data_type,
                "values": [vec(item.color) for item in attr.data],
            }
            for attr in mesh.color_attributes
        ],
        "color_layers": color_registry(mesh),
    }


def make_spec_mesh(name, spec):
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(spec["vertices"], spec["edges"], [face["vertices"] for face in spec["faces"]])
    for index in range(3):
        mesh.materials.append(bpy.data.materials.new(f"Slot{index}"))
    mesh.update()
    for poly, face in zip(mesh.polygons, spec["faces"]):
        poly.material_index = face.get("material", 0)
        poly.use_smooth = face.get("smooth", False)
    edge_lookup = {tuple(sorted(edge.vertices)): edge.index for edge in mesh.edges}
    sharp_flags = [False] * len(mesh.edges)
    for pair in spec.get("sharp_edges", []):
        sharp_flags[edge_lookup[tuple(sorted(pair))]] = True
    if sharp_flags:
        mesh.edges.foreach_set("use_edge_sharp", sharp_flags)
    for layer_spec in spec.get("uv_layers", []):
        layer = mesh.uv_layers.new(name=layer_spec["name"])
        cursor = 0
        for face_index, poly in enumerate(mesh.polygons):
            for corner in range(poly.loop_total):
                layer.data[poly.loop_start + corner].uv = layer_spec["faces"][face_index][corner]
                cursor += 1
        assert cursor == len(layer.data)
    color_specs = spec.get("colors", [])
    for color_spec in color_specs:
        attr = mesh.color_attributes.new(
            name=color_spec["name"], type=color_spec["data_type"], domain=color_spec["domain"]
        )
        for item, value in zip(attr.data, color_spec["values"]):
            item.color = value
    if color_specs:
        names = [attr.name for attr in mesh.color_attributes]
        mesh.color_attributes.active_color_index = names.index(spec.get("color_active", names[0]))
        mesh.color_attributes.render_color_index = names.index(spec.get("color_render", names[0]))
    if spec.get("custom_normals"):
        mesh.normals_split_custom_set(
            [normal for face in spec["custom_normals"] for normal in face]
        )
    return mesh


def add_groups(obj, spec):
    names = []
    for name, weights in spec.get("groups", {}).items():
        group = obj.vertex_groups.new(name=name)
        names.append(name)
        for index, weight in weights.items():
            group.add([int(index)], float(weight), "REPLACE")
    return names


def output_mesh(owner, depsgraph):
    evaluated = owner.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    try:
        vertices = [vec(vertex.co) for vertex in mesh.vertices]
        uv_layers = {}
        for layer in mesh.uv_layers:
            uv_layers[layer.name] = [vec(layer.data[loop].uv) for loop in range(len(mesh.loops))]
        faces = []
        for poly in mesh.polygons:
            loops = range(poly.loop_start, poly.loop_start + poly.loop_total)
            faces.append({
                "vertices": list(poly.vertices),
                "corners": [vertices[mesh.loops[loop].vertex_index] for loop in loops],
                "material": int(poly.material_index),
                "smooth": bool(poly.use_smooth),
                "normal": vec(poly.normal),
                "corner_normals": [vec(mesh.corner_normals[loop].vector) for loop in loops],
                "uv_layers": {
                    layer.name: [vec(layer.data[loop].uv) for loop in range(poly.loop_start, poly.loop_start + poly.loop_total)]
                    for layer in mesh.uv_layers
                },
            })
        edges = [
            {"vertices": list(edge.vertices), "sharp": bool(edge.use_edge_sharp)}
            for edge in mesh.edges
        ]
        colors = {}
        for attr in mesh.color_attributes:
            colors[attr.name] = {
                "domain": attr.domain,
                "data_type": attr.data_type,
                "values": [vec(item.color) for item in attr.data],
            }
        group_names = {index: group.name for index, group in enumerate(evaluated.vertex_groups)}
        weights = []
        for vertex in mesh.vertices:
            weights.append({
                group_names[item.group]: f32(item.weight)
                for item in vertex.groups
                if item.group in group_names
            })
        return {
            "counts": {
                "vertices": len(mesh.vertices), "edges": len(mesh.edges),
                "faces": len(mesh.polygons), "corners": len(mesh.loops),
            },
            "vertices": vertices,
            "edges": edges,
            "faces": faces,
            "uv_layers": uv_layers,
            **uv_metadata(mesh),
            "colors": colors,
            "color_layers": color_registry(mesh),
            "weights": weights,
        }
    finally:
        evaluated.to_mesh_clear()


CURRENT_OWNER = None
CURRENT_OUTPUT = None


class ScrewProbeEngine(bpy.types.RenderEngine):
    bl_idname = "POTTER_SCREW_PROBE"
    bl_label = "Potter Screw Probe"

    def render(self, depsgraph):
        global CURRENT_OUTPUT
        assert depsgraph.mode == "RENDER", depsgraph.mode
        CURRENT_OUTPUT = output_mesh(CURRENT_OWNER, depsgraph)


if not hasattr(bpy.types, "ScrewProbeEngine"):
    bpy.utils.register_class(ScrewProbeEngine)


BASE = [
    [0.375, -0.25, 0.125],
    [1.125, 0.125, 0.5],
    [1.5, 0.875, 0.375],
    [2.125, 1.0, 1.125],
    [2.625, -0.5, 0.875],
]
MIXED = {
    "vertices": [
        [0.125, 0.25, 0.375], [1.125, 0.125, 0.5], [1.25, 1.0, 0.625], [0.25, 1.125, 0.25],
        [2.125, 0.375, 0.875], [3.0, 0.25, 1.25], [3.125, 1.0, 1.125], [2.0, 1.125, 0.75],
        [0.625, 2.25, 0.375], [1.5, 2.5, 1.125],
    ],
    "edges": [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [8, 9]],
    "faces": [
        {"vertices": [0, 1, 2, 3], "material": 1, "smooth": False},
        {"vertices": [4, 5, 6, 7], "material": 2, "smooth": True},
    ],
    "sharp_edges": [[0, 1], [5, 6]],
    "uv_layers": [
        {"name": "UVMap", "faces": [[[0.125, 0.25], [0.375, 0.5], [0.625, 0.75], [0.875, 1.0]], [[1.125, 0.25], [1.375, 0.5], [1.625, 0.75], [1.875, 1.0]]]},
        {"name": "SecondaryUV", "faces": [[[2.125, 1.25], [2.375, 1.5], [2.625, 1.75], [2.875, 2.0]], [[3.125, 1.25], [3.375, 1.5], [3.625, 1.75], [3.875, 2.0]]]},
    ],
    "colors": [
        {"name": "PointTint", "domain": "POINT", "data_type": "FLOAT_COLOR", "values": [[0.125, 0.25, 0.375, 1.0], [0.25, 0.375, 0.5, 1.0], [0.375, 0.5, 0.625, 1.0], [0.5, 0.625, 0.75, 1.0], [0.625, 0.75, 0.875, 1.0], [0.75, 0.875, 1.0, 1.0], [0.875, 1.0, 0.125, 1.0], [1.0, 0.125, 0.25, 1.0], [0.125, 0.5, 0.25, 1.0], [0.375, 0.75, 0.5, 1.0]]},
        {"name": "CornerTint", "domain": "CORNER", "data_type": "FLOAT_COLOR", "values": [[0.125 + 0.0625 * i, 0.25 + 0.03125 * i, 0.375, 1.0] for i in range(8)]},
    ],
    "color_active": "CornerTint",
    "color_render": "PointTint",
    "custom_normals": [
        [[0.0, 0.0, -1.0], [0.1, 0.0, -0.99], [0.0, 0.1, -0.99], [0.1, 0.1, -0.98]],
        [[0.2, 0.0, 0.98], [0.1, 0.2, 0.97], [0.0, 0.2, 0.98], [0.1, 0.1, 0.98]],
    ],
    "groups": {"Partial": {"0": 0.25, "2": 0.75, "5": 0.5, "9": 1.0}, "Empty": {}},
}

RING = {
    "vertices": [[1.0, 0.125, 0.25], [0.5, 0.375, 1.0], [-0.25, 0.625, 0.75], [-0.875, 0.25, -0.375], [0.125, -0.125, -1.0]],
    "edges": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]],
    "faces": [],
    "uv_layers": [
        {"name": "UVMap", "faces": []},
        {"name": "SecondaryUV", "faces": []},
    ],
    "groups": {"RingWeights": {"0": 0.25, "2": 0.625}},
}

MERGE_PROFILE = {
    "vertices": [[0.25, 0.0, 0.125], [0.625, 0.0, 0.75], [1.125, 0.5, 0.375], [0.375, 0.75, 1.125]],
    "edges": [[0, 1], [1, 2], [2, 3], [3, 0]],
    "faces": [{"vertices": [0, 1, 2, 3], "material": 2, "smooth": False}],
    "uv_layers": [
        {"name": "UVMap", "faces": [[[0.125, 0.25], [0.375, 0.5], [0.625, 0.75], [0.875, 1.0]]]},
        {"name": "SecondaryUV", "faces": [[[1.125, 1.25], [1.375, 1.5], [1.625, 1.75], [1.875, 2.0]]]},
    ],
    "groups": {"Boundary": {"0": 0.375, "1": 0.875}},
}

CASES = [
    {"name": "open_zigzag_z", "spec": {"vertices": BASE, "edges": [[0, 1], [1, 2], [2, 3]], "faces": [], "uv_layers": [{"name": "UVMap", "faces": []}, {"name": "SecondaryUV", "faces": []}], "groups": {"Partial": {"0": 0.25, "2": 0.75}}}, "params": {"axis": "Z", "angle": 1.75, "screw_offset": 0.625, "steps": 3, "render_steps": 5, "iterations": 1, "use_stretch_u": True, "use_stretch_v": True, "use_normal_calculate": False, "use_normal_flip": False, "use_smooth_shade": True}},
    {"name": "loose_line_x_iter3", "spec": {"vertices": [[0.25, 0.375, -0.125], [0.875, 0.125, 0.625], [1.375, -0.25, 1.125]], "edges": [[0, 1], [1, 2]], "faces": [], "groups": {"LineWeights": {"0": 0.5}}}, "params": {"axis": "X", "angle": -1.25, "screw_offset": 0.375, "steps": 2, "render_steps": 4, "iterations": 3, "use_stretch_u": False, "use_stretch_v": True, "use_normal_calculate": True, "use_normal_flip": True, "use_smooth_shade": False}},
    {"name": "isolated_point_profile_z", "spec": {"vertices": [[0.375, -0.25, 0.625]], "edges": [], "faces": [], "uv_layers": [{"name": "UVMap", "faces": []}, {"name": "SecondaryUV", "faces": []}], "groups": {"PointWeight": {"0": 0.75}}}, "params": {"axis": "Z", "angle": 1.5, "screw_offset": 0.25, "steps": 2, "render_steps": 4, "iterations": 1, "use_normal_calculate": True, "use_normal_flip": False, "use_smooth_shade": True}},
    {"name": "closed_profile_ring_y", "spec": RING, "params": {"axis": "Y", "angle": 6.2831854820251465, "screw_offset": 0.0, "steps": 4, "render_steps": 6, "iterations": 1, "use_merge_vertices": False, "use_stretch_u": True, "use_stretch_v": False, "use_normal_calculate": True, "use_normal_flip": False, "use_smooth_shade": True}},
    {"name": "object_axis_scaled_owner", "spec": MIXED, "params": {"axis": "Z", "object": "screw_axis", "use_object_screw_offset": True, "angle": 1.25, "screw_offset": 0.5, "steps": 3, "render_steps": 5, "iterations": 2, "use_stretch_u": False, "use_stretch_v": True, "use_normal_calculate": True, "use_normal_flip": False, "use_smooth_shade": True}, "owner_transform": {"translation": [1.25, -0.75, 0.5], "rotation_deg": [0, 0, 90], "scale": [1.5, 0.75, 1.25]}, "axis_transform": {"translation": [2.0, 1.25, -0.5], "rotation_deg": [0, 90, 90], "scale": [0.75, 1.25, 1.5]}},
    {"name": "merge_threshold_boundary_on", "spec": MERGE_PROFILE, "params": {"axis": "Z", "angle": 6.2831854820251465, "screw_offset": 0.0, "steps": 4, "render_steps": 4, "iterations": 2, "use_merge_vertices": True, "merge_threshold": 0.25, "use_stretch_u": True, "use_stretch_v": False, "use_normal_calculate": False, "use_normal_flip": False, "use_smooth_shade": True}},
    {"name": "merge_threshold_boundary_off", "spec": MERGE_PROFILE, "params": {"axis": "Z", "angle": 6.2831854820251465, "screw_offset": 0.0, "steps": 4, "render_steps": 4, "iterations": 2, "use_merge_vertices": False, "merge_threshold": 0.25, "use_stretch_u": False, "use_stretch_v": True, "use_normal_calculate": False, "use_normal_flip": True, "use_smooth_shade": False}},
    {"name": "two_face_uv_normals_preview", "spec": MIXED, "params": {"axis": "X", "angle": 1.5, "screw_offset": 0.25, "steps": 2, "render_steps": 5, "iterations": 3, "use_stretch_u": True, "use_stretch_v": True, "use_normal_calculate": True, "use_normal_flip": False, "use_smooth_shade": False}},
    {"name": "two_face_uv_normals_render", "spec": MIXED, "params": {"axis": "X", "angle": 1.5, "screw_offset": 0.25, "steps": 2, "render_steps": 5, "iterations": 3, "use_stretch_u": True, "use_stretch_v": True, "use_normal_calculate": True, "use_normal_flip": False, "use_smooth_shade": False}, "render": True},
    {"name": "normal_calculate_asymmetric_radial_neighbors", "spec": {"vertices": [[2.0, 0.0, 0.0], [1.9, 0.0, 1.0], [0.0, 0.0, 2.0]], "edges": [[0, 1], [0, 2]], "faces": []}, "params": {"axis": "Z", "angle": 1.0, "screw_offset": 0.0, "steps": 2, "render_steps": 2, "iterations": 1, "use_normal_calculate": True, "use_normal_flip": False, "use_smooth_shade": True}},
    {"name": "large_pivot_preserves_first_ring_f32", "spec": {"vertices": [[0.0001, 0.0, 0.0], [0.0001, 0.0, 1.0]], "edges": [[0, 1]], "faces": []}, "params": {"axis": "Z", "object": "screw_axis", "angle": 1.0, "screw_offset": 0.0, "steps": 2, "render_steps": 2, "iterations": 1}, "axis_transform": {"translation": [10000, 0, 0], "rotation_deg": [0, 0, 0], "scale": [1, 1, 1]}},
    {"name": "large_radius_single_axis_matrix_f32", "spec": {"vertices": [[10_000.0, 0.0, 0.0], [10_000.0, 0.0, 1.0]], "edges": [[0, 1]], "faces": []}, "params": {"axis": "Z", "angle": 1.5707963705062866, "screw_offset": 0.0, "steps": 1, "render_steps": 1, "iterations": 1, "use_normal_calculate": False, "use_normal_flip": False, "use_smooth_shade": True}},
]


def make_camera():
    data = bpy.data.cameras.new("ProbeCamera")
    camera = bpy.data.objects.new("ProbeCamera", data)
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera


def run_case(case):
    global CURRENT_OWNER, CURRENT_OUTPUT
    reset()
    spec = case["spec"]
    mesh = make_spec_mesh(case["name"], spec)
    owner = bpy.data.objects.new("Owner", mesh)
    bpy.context.scene.collection.objects.link(owner)
    owner_transform = set_transform(owner, case.get("owner_transform", {"translation": [0, 0, 0], "rotation_deg": [0, 0, 0], "scale": [1, 1, 1]}))
    group_names = add_groups(owner, spec)
    axis_transform = None
    if "axis_transform" in case:
        axis = bpy.data.objects.new("Axis", None)
        bpy.context.scene.collection.objects.link(axis)
        axis_transform = set_transform(axis, case["axis_transform"])
    params = dict(case["params"])
    if "object" in params:
        params["object"] = "screw_axis"
    modifier = owner.modifiers.new("Screw", "SCREW")
    screw_fields = (
        "object", "use_object_screw_offset", "iterations", "steps", "render_steps",
        "angle", "axis", "screw_offset", "merge_threshold", "use_merge_vertices",
        "use_normal_calculate", "use_normal_flip", "use_smooth_shade",
        "use_stretch_u", "use_stretch_v",
    )
    modifier_defaults = {key: getattr(modifier, key) for key in screw_fields}
    for key, value in params.items():
        setattr(modifier, key, bpy.data.objects["Axis"] if key == "object" else value)
    modifier_readback = {}
    for key in params:
        value = getattr(modifier, key)
        modifier_readback[key] = "screw_axis" if key == "object" else value
    doc_input = {
        "vertices": [vec(vertex.co) for vertex in mesh.vertices],
        "edges": [list(edge.vertices) for edge in mesh.edges],
        "faces": [
            {"vertices": list(poly.vertices), "material": int(poly.material_index), "smooth": bool(poly.use_smooth)}
            for poly in mesh.polygons
        ],
        "sharp_edges": [list(edge.vertices) for edge in mesh.edges if edge.use_edge_sharp],
        "uv_layers": [
            {"name": layer.name, "values": [vec(item.uv) for item in layer.data]}
            for layer in mesh.uv_layers
        ],
        **uv_metadata(mesh),
        **color_input(mesh),
        "groups": {
            name: {str(index): f32(weight) for index, weight in weights.items()}
            for name, weights in spec.get("groups", {}).items()
        },
        "material_slots": len(owner.material_slots),
    }
    doc_input["custom_normals"] = spec.get("custom_normals", [])
    camera_needed = bool(case.get("render"))
    if camera_needed:
        make_camera()
    dg = bpy.context.evaluated_depsgraph_get()
    viewport = output_mesh(owner, dg)
    render_output = None
    if camera_needed:
        CURRENT_OWNER = owner
        CURRENT_OUTPUT = None
        bpy.context.scene.render.engine = "POTTER_SCREW_PROBE"
        bpy.context.scene.render.resolution_x = 16
        bpy.context.scene.render.resolution_y = 16
        bpy.ops.render.render(write_still=False)
        render_output = CURRENT_OUTPUT
        assert render_output is not None
        bpy.context.scene.render.engine = "BLENDER_EEVEE"
    return {
        "name": case["name"],
        "input": doc_input,
        "modifier": {"type": "screw", "defaults": modifier_defaults, "params": modifier_readback},
        "owner_transform": owner_transform,
        "axis_transform": axis_transform,
        "group_names": group_names,
        "render_mode": camera_needed,
        "viewport": viewport,
        "render": render_output,
    }


def main():
    cases = [run_case(case) for case in CASES]
    document = {"blender_version": "5.2.2", "cases": cases}
    path = os.path.join(FIXTURES, "screw.json")
    with open(path, "w", encoding="utf-8") as output:
        json.dump(document, output, indent=1, allow_nan=False)
        output.write("\n")
    print(f"wrote {path}")
    for case in cases:
        output = case["render"] if case["render_mode"] else case["viewport"]
        print(case["name"], output["counts"])


main()
