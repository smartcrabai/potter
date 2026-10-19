# Usage: /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup -P tests/fixtures/construction_p/weld/generate.py
"""Blender 5.2.2 oracle fixtures for Weld vertex selection and merge behavior."""
import json
import os

import bpy

FIXTURES = os.path.dirname(os.path.abspath(__file__))
BLENDER_VERSION = "5.2.2 LTS"
assert bpy.app.version == (5, 2, 2), f"need Blender 5.2.2, got {bpy.app.version}"

TRANSFORM = {
    "location": [2.0, -3.0, 1.0],
    "rotation_euler": [0.5, 0.25, 0.75],
    "scale": [1.5, 0.75, 2.0],
}


def uv_layers(faces):
    total = sum(len(face) for face in faces)
    return [
        {
            "name": "UV_A",
            "values": [[(i * 11 + 3) / 1024, (i * 7 + 5) / 1024] for i in range(total)],
        },
        {
            "name": "UV_B",
            "values": [[(i * 13 + 1) / 1024, (i * 5 + 9) / 1024] for i in range(total)],
        },
    ]

def uv_metadata(mesh):
    layers = list(mesh.uv_layers)
    active_index = mesh.uv_layers.active_index if layers else None
    render_index = mesh.uv_layers.active_render_index if layers else None
    return {
        "uv_active": layers[active_index].name if active_index is not None else None,
        "uv_render": layers[render_index].name if render_index is not None else None,
    }


def canonical_uv_store(mesh):
    layers = []
    for layer in mesh.uv_layers:
        faces = []
        for polygon in mesh.polygons:
            uv = [
                [float(layer.data[loop].uv[0]), 1.0 - float(layer.data[loop].uv[1])]
                for loop in range(polygon.loop_start, polygon.loop_start + polygon.loop_total)
            ]
            faces.append({"face_id": polygon.index, "uv": uv})
        layers.append({"name": layer.name, "faces": faces})
    if not layers:
        return None
    metadata = uv_metadata(mesh)
    return {"layers": layers, "active": metadata["uv_active"], "render": metadata["uv_render"]}


def color_specs(vertex_count, faces):
    corner_count = sum(len(face) for face in faces)
    return [
        {
            "name": "PointTint",
            "domain": "POINT",
            "data_type": "FLOAT_COLOR",
            "values": [[(i + 1) / 32, (i + 2) / 64, (i + 3) / 128, 1.0]
                       for i in range(vertex_count)],
        },
        {
            "name": "CornerTint",
            "domain": "CORNER",
            "data_type": "FLOAT_COLOR",
            "values": [[(i + 1) / 64, (i + 3) / 128, (i + 5) / 256, 1.0]
                       for i in range(corner_count)],
        },
    ]
def storage_key(name):
    internal = {
        "alembic_normals", "alembic_time_samples", "alembic_velocity", "blender_attributes",
        "blender_metadata", "blender_skin_vertices", "color_layers", "crease_edge",
        "custom_normal", "seam", "sharp", "shade_smooth", "skin_radii", "skin_roots",
        "uv_layers", "vertex_groups",
    }
    return f"user:{name}" if name.startswith("user:") or name in internal else name


def cube_spec():
    vertices = [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1.125, 0, 1], [1, 1, 1.25], [0, 1, 1],
        [0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
    ]
    faces = [
        [0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
        [3, 7, 6, 2], [0, 4, 7, 3], [1, 2, 6, 5], [8, 9, 10, 11],
    ]
    groups = {
        "keep": {0: 1.0, 1: 0.0, 2: 1 / 1048576, 3: 1.0,
                 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0,
                 8: 1.0, 9: 0.0, 10: 1 / 1048576},
        "other": {0: 1.0},
        "empty": {},
    }
    return {
        "vertices": vertices,
        "faces": faces,
        "edges": [[0, 8], [1, 11], [2, 10], [3, 9]],
        "materials": [0, 2, 1, 2, 0, 1, 3],
        "face_smooth": [False, True, False, True, False, True, False],
        "sharp_edges": [[0, 1], [2, 6], [8, 9], [0, 8]],
        "colors": color_specs(len(vertices), faces),
        "groups": groups,
        "uv_layers": uv_layers(faces),
        "material_slots": 4,
        "object_transform": TRANSFORM,
    }


def grid_spec():
    vertices = [[0, 0, 0], [1, 0, 0], [3, 0, 0],
                [0, 2, 0], [1, 2, 0], [3, 2, 0]]
    faces = [[0, 1, 4, 3], [1, 2, 5, 4]]
    return {
        "vertices": vertices,
        "faces": faces,
        "edges": [],
        "materials": [1, 2],
        "face_smooth": [True, False],
        "sharp_edges": [[0, 1], [1, 4]],
        "colors": color_specs(len(vertices), faces),
        "groups": {"selected": {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}},
        "uv_layers": uv_layers(faces),
        "material_slots": 3,
        "object_transform": TRANSFORM,
    }


def chain_spec(reverse=False):
    return {
        "vertices": [[0, 0, 0], [1 / 8, 0, 0], [1 / 4, 0, 0]],
        "faces": [],
        "edges": [[1, 2], [0, 1]] if reverse else [[0, 1], [1, 2]],
        "materials": [],
        "face_smooth": [],
        "sharp_edges": [],
        "colors": color_specs(3, []),
        "groups": {},
        "uv_layers": [],
        "material_slots": 0,
        "object_transform": TRANSFORM,
    }


def loose_spec():
    vertices = [[0, 0, 0], [1 / 8, 0, 0], [0, 1, 0], [2, 0, 0], [2 + 1 / 8, 0, 0]]
    faces = [[0, 1, 2]]
    return {
        "vertices": vertices,
        "faces": faces,
        "edges": [[3, 4]],
        "materials": [2],
        "face_smooth": [False],
        "sharp_edges": [[0, 1], [3, 4]],
        "colors": color_specs(len(vertices), faces),
        "groups": {},
        "uv_layers": uv_layers(faces),
        "material_slots": 3,
        "object_transform": TRANSFORM,
    }


def split_spec():
    vertices = [[0, 0, 0], [1, 0, 0], [3 / 2, 1, 0], [0, 0, 0], [-1 / 2, 5 / 4, 0], [-1 / 4, 1 / 4, 0]]
    faces = [[0, 1, 2, 3, 4, 5]]
    return {
        "vertices": vertices,
        "faces": faces,
        "edges": [[0, 3]],
        "materials": [2],
        "face_smooth": [False],
        "sharp_edges": [[0, 1], [0, 3]],
        "colors": color_specs(len(vertices), faces),
        "groups": {"subset": {0: 1.0, 3: 1.0, 1: 0.0}},
        "uv_layers": uv_layers(faces),
        "material_slots": 4,
        "object_transform": TRANSFORM,
    }



def custom_normal_spec():
    faces = [[0, 1, 2], [3, 4, 5]]
    return {
        "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 0], [1, -1, 0]],
        "faces": faces,
        "edges": [[0, 4], [1, 3]],
        "materials": [1, 2],
        "face_smooth": [True, True],
        "sharp_edges": [],
        "colors": color_specs(6, faces),
        "custom_normals": [
            [
                [0.22647890448570251, 0.7925845980644226, 0.5661421418190002],
                [0.3057146668434143, 0.7543173432350159, 0.580985426902771],
                [0.36850568652153015, 0.7186882495880127, 0.5896530747413635],
            ],
            [
                [0.4618031084537506, 0.7572927474975586, 0.4617851674556732],
                [0.5027193427085876, 0.7205271124839783, 0.4776129424571991],
                [0.5353772044181824, 0.6883115768432617, 0.4894878566265106],
            ],
        ],
        "groups": {"keep": {0: 1.0, 4: 1.0, 1: 0.0, 3: 0.0, 2: 0.0, 5: 0.0}},
        "uv_layers": [],
        "material_slots": 3,
        "object_transform": TRANSFORM,
    }

def no_deform_spec():
    spec = cube_spec()
    spec["groups"] = {"empty": {}}
    return spec



VARIANTS = [
    {
        "name": "asymmetric_cube_with_split_face",
        "mesh": cube_spec(),
        "cases": [
            {"mode": "ALL", "threshold": 1 / 1024, "vertex_group": "keep", "invert": False},
            {"mode": "ALL", "threshold": 1 / 1024, "vertex_group": "keep", "invert": True},
            {"mode": "CONNECTED", "threshold": 1 / 1024, "vertex_group": "keep", "invert": False, "loose_edges": False},
            {"mode": "CONNECTED", "threshold": 1 / 1024, "vertex_group": "keep", "invert": True, "loose_edges": True},
            {"mode": "ALL", "threshold": 1 / 1024, "vertex_group": "missing", "invert": False},
            {"mode": "ALL", "threshold": 1 / 1024, "vertex_group": "missing", "invert": True},
            {"mode": "ALL", "threshold": 1 / 1024, "vertex_group": "empty", "invert": False},
            {"mode": "ALL", "threshold": 1 / 1024, "vertex_group": "empty", "invert": True},
        ],
    },
    {
        "name": "empty_group_without_deform_weights",
        "mesh": no_deform_spec(),
        "cases": [
            {"mode": "ALL", "threshold": 1 / 1024, "vertex_group": "empty", "invert": False},
            {"mode": "ALL", "threshold": 1 / 1024, "vertex_group": "empty", "invert": True},
        ],
    },
    {
        "name": "asymmetric_grid_threshold",
        "mesh": grid_spec(),
        "cases": [
            {"mode": "ALL", "threshold": 1.0, "vertex_group": "", "invert": False},
            {"mode": "ALL", "threshold": 1023 / 1024, "vertex_group": "", "invert": False},
            {"mode": "CONNECTED", "threshold": 1.0, "vertex_group": "", "invert": False, "loose_edges": False},
            {"mode": "CONNECTED", "threshold": 1.0, "vertex_group": "selected", "invert": False, "loose_edges": True},
        ],
    },
    {
        "name": "connected_chain_forward",
        "mesh": chain_spec(False),
        "cases": [
            {"mode": "ALL", "threshold": 1 / 8, "vertex_group": "", "invert": False},
            {"mode": "CONNECTED", "threshold": 1 / 8, "vertex_group": "", "invert": False, "loose_edges": False},
        ],
    },
    {
        "name": "connected_chain_reverse",
        "mesh": chain_spec(True),
        "cases": [
            {"mode": "ALL", "threshold": 1 / 8, "vertex_group": "", "invert": False},
            {"mode": "CONNECTED", "threshold": 1 / 8, "vertex_group": "", "invert": False, "loose_edges": False},
        ],
    },
    {
        "name": "connected_loose_boundary",
        "mesh": loose_spec(),
        "cases": [
            {"mode": "ALL", "threshold": 1 / 8, "vertex_group": "", "invert": False, "loose_edges": False},
            {"mode": "ALL", "threshold": 1 / 8, "vertex_group": "", "invert": False, "loose_edges": True},
            {"mode": "CONNECTED", "threshold": 1 / 8, "vertex_group": "", "invert": False, "loose_edges": False},
            {"mode": "CONNECTED", "threshold": 1 / 8, "vertex_group": "", "invert": False, "loose_edges": True},
            {"mode": "CONNECTED", "threshold": 127 / 1024, "vertex_group": "", "invert": False, "loose_edges": True},
        ],
    },
    {
        "name": "split_ngon",
        "mesh": split_spec(),
        "cases": [
            {"mode": "ALL", "threshold": 1 / 1024, "vertex_group": "", "invert": False},
            {"mode": "ALL", "threshold": 1 / 1024, "vertex_group": "subset", "invert": False},
            {"mode": "ALL", "threshold": 1 / 1024, "vertex_group": "subset", "invert": True},
            {"mode": "CONNECTED", "threshold": 1 / 1024, "vertex_group": "", "invert": False, "loose_edges": False},
        ],
    },
    {
        "name": "custom_normals_selected_and_unselected_groups",
        "mesh": custom_normal_spec(),
        "cases": [
            {"mode": "ALL", "threshold": 1 / 1024, "vertex_group": "keep", "invert": False},
            {"mode": "ALL", "threshold": 1 / 1024, "vertex_group": "keep", "invert": True},
            {"mode": "ALL", "threshold": 1 / 1024, "vertex_group": "missing", "invert": False},
        ],
    },
]


def mesh_document(spec, mesh):
    attributes = {
        "shade_smooth": {
            "domain": "face",
            "type": "bool",
            "values": {f"f{i}": smooth for i, smooth in enumerate(spec["face_smooth"])},
        },
        "blender_polygon_smooth": spec["face_smooth"],
    }
    uv_store = canonical_uv_store(mesh)
    if uv_store is not None:
        attributes["uv_layers"] = uv_store
    encodings = {}
    names = []
    for layer in spec["colors"]:
        name = layer["name"]
        names.append(name)
        encodings[name] = layer["data_type"]
        if layer["domain"] == "POINT":
            values = {f"v{i}": color for i, color in enumerate(layer["values"])}
            domain = "point"
        else:
            values = {}
            start = 0
            for face_index, face in enumerate(spec["faces"]):
                end = start + len(face)
                values[f"f{face_index}"] = layer["values"][start:end]
                start = end
            domain = "corner"
        attributes[storage_key(name)] = {
            "domain": domain,
            "type": "color",
            "values": values,
        }
    if names:
        active = spec.get("color_active", "CornerTint" if "CornerTint" in names else names[0])
        render = spec.get("color_render", "PointTint" if "PointTint" in names else names[0])
        attributes["color_layers"] = {"layers": names, "active": active, "render": render}
        attributes["blender_metadata"] = {"color_data_types": encodings}
    return {
        "vertices": spec["vertices"],
        "faces": spec["faces"],
        "edges": spec["edges"],
        "materials": spec["materials"],
        "face_smooth": spec["face_smooth"],
        "sharp_edges": spec["sharp_edges"],
        "attributes": attributes,
        "custom_normals": spec.get("custom_normals", []),
    }


def build_object(name, spec):
    me = bpy.data.meshes.new(name + "_mesh")
    me.from_pydata(spec["vertices"], spec["edges"], spec["faces"])
    me.update()
    for slot in range(spec["material_slots"]):
        me.materials.append(bpy.data.materials.new(f"{name}_slot_{slot}"))
    for polygon, material in zip(me.polygons, spec["materials"]):
        polygon.material_index = material
    if spec["face_smooth"]:
        me.polygons.foreach_set("use_smooth", spec["face_smooth"])
    if spec.get("custom_normals"):
        me.normals_split_custom_set(
            [normal for face in spec["custom_normals"] for normal in face]
        )
        me.update()
    for layer_spec in spec["uv_layers"]:
        layer = me.uv_layers.new(name=layer_spec["name"])
        for item, uv in zip(layer.data, layer_spec["values"]):
            item.uv = uv
    color_specs = spec["colors"]
    for color_spec in color_specs:
        color = me.color_attributes.new(
            name=color_spec["name"], type=color_spec["data_type"], domain=color_spec["domain"]
        )
        for item, value in zip(color.data, color_spec["values"]):
            item.color = value
    if color_specs:
        names = [color.name for color in me.color_attributes]
        me.color_attributes.active_color_index = names.index(
            spec.get("color_active", "CornerTint" if "CornerTint" in names else names[0])
        )
        me.color_attributes.render_color_index = names.index(
            spec.get("color_render", "PointTint" if "PointTint" in names else names[0])
        )
    if spec["sharp_edges"]:
        edge_indices = {
            tuple(sorted(edge.vertices)): edge.index
            for edge in me.edges
        }
        sharp = [False] * len(me.edges)
        for pair in spec["sharp_edges"]:
            sharp[edge_indices[tuple(sorted(pair))]] = True
        me.edges.foreach_set("use_edge_sharp", sharp)
        me.update()
    ob = bpy.data.objects.new(name, me)
    ob.location = spec["object_transform"]["location"]
    ob.rotation_euler = spec["object_transform"]["rotation_euler"]
    ob.scale = spec["object_transform"]["scale"]
    bpy.context.scene.collection.objects.link(ob)
    for group_name, values in spec["groups"].items():
        group = ob.vertex_groups.new(name=group_name)
        for index, weight in values.items():
            group.add([index], weight, "REPLACE")
    return ob


def expected_output(ob):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = ob.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    try:
        faces = []
        for polygon in mesh.polygons:
            faces.append([mesh.loops[i].vertex_index for i in range(polygon.loop_start, polygon.loop_start + polygon.loop_total)])
        groups = {}
        for group in ob.vertex_groups:
            values = []
            for vertex in mesh.vertices:
                element = next((item for item in vertex.groups if item.group == group.index), None)
                values.append(float(element.weight) if element else 0.0)
            groups[group.name] = values
        sharp_edges = []
        for edge in mesh.edges:
            if edge.use_edge_sharp:
                sharp_edges.append(sorted(edge.vertices))
        colors = {
            attr.name: {
                "domain": attr.domain,
                "data_type": attr.data_type,
                "values": [list(item.color) for item in attr.data],
            }
            for attr in mesh.color_attributes
        }
        color_attributes = list(mesh.color_attributes)
        color_layers = {
            "layers": [attr.name for attr in color_attributes],
            "active": color_attributes[mesh.color_attributes.active_color_index].name if color_attributes else None,
            "render": color_attributes[mesh.color_attributes.render_color_index].name if color_attributes else None,
        }
        return {
            "counts": {
                "vertices": len(mesh.vertices),
                "edges": len(mesh.edges),
                "faces": len(mesh.polygons),
                "corners": len(mesh.loops),
            },
            "vertices": [[float(c) for c in vertex.co] for vertex in mesh.vertices],
            "faces": faces,
            "edges": [list(edge.vertices) for edge in mesh.edges],
            "materials": [int(face.material_index) for face in mesh.polygons],
            "face_smooth": [bool(face.use_smooth) for face in mesh.polygons],
            "sharp_edges": sharp_edges,
            "uv_layers": [
                {"name": layer.name, "values": [[float(c) for c in item.uv] for item in layer.data]}
                for layer in mesh.uv_layers
            ],
            **uv_metadata(mesh),
            "colors": colors,
            "color_layers": color_layers,
            "groups": groups,
            "corner_normals": (
                [[float(c) for c in normal.vector] for normal in mesh.corner_normals]
                if mesh.has_custom_normals
                else []
            ),
        }
    finally:
        evaluated.to_mesh_clear()


def main():
    variants = []
    for variant in VARIANTS:
        spec = variant["mesh"]
        output_variant = {
            "name": variant["name"],
            "groups": {name: {str(index): weight for index, weight in values.items()} for name, values in spec["groups"].items()},
            "material_slots": spec["material_slots"],
            "object_transform": spec["object_transform"],
            "cases": [],
        }
        for case_index, case in enumerate(variant["cases"]):
            ob = build_object(f"{variant['name']}_{case_index}", spec)
            if case_index == 0:
                output_variant["mesh"] = mesh_document(spec, ob.data)
            modifier = ob.modifiers.new("Weld oracle", "WELD")
            modifier.mode = case["mode"]
            modifier.merge_threshold = case["threshold"]
            modifier.vertex_group = case["vertex_group"]
            modifier.invert_vertex_group = case["invert"]
            modifier.loose_edges = case.get("loose_edges", False)
            output_variant["cases"].append({"params": case, "output": expected_output(ob)})
            bpy.data.objects.remove(ob, do_unlink=True)
            bpy.data.meshes.remove(spec_mesh := bpy.data.meshes.get(f"{variant['name']}_{case_index}_mesh"))
        variants.append(output_variant)
    with open(os.path.join(FIXTURES, "weld.json"), "w", encoding="utf-8") as handle:
        json.dump({"blender_version": BLENDER_VERSION, "variants": variants}, handle, indent=2)
        handle.write("\n")


main()
