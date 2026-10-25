"""Blender 5.2 Boolean parity corpus generator for tests/blend_boolean_parity.rs.

Run headless from the repository root (`--factory-startup` keeps the run reproducible):

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
        --python tests/fixtures/boolean_parity/generate_blender_expectations.py -- \
        tests/fixtures/boolean_parity

Outputs under the directory given after `--`:

    expected/<case>.json   one expectation per case (committed corpus)
    scene.blend            every case in one scene; the Potter import source (git-ignored)

Cases reproduce research/§3 of the PR17 design: §3.1 offset, §3.2 nested, §3.3 coplanar,
§3.4 face-touching and holes, §3.5 open operands, §3.6 self-overlapping operand, §3.7
collection operand, attributes, vertex groups, and material INDEX/TRANSFER. EXACT is asserted for
the standard cases; every recorded FLOAT row is compared to Blender's own output, including its
source-specific coplanar, touching, and material-index results.

Each case is a Boolean modifier on a host object named after the case. Every object keeps
the identity transform, so object-local and world coordinates coincide. All cases share one
scene, and the Rust driver evaluates each host with only its own Boolean enabled. Boolean
evaluation is independent per host, so the shared scene gives the same result as one scene
per case.

Expectations are canonical: face rings are rotated to start at their minimum vertex
(coordinates rounded to 1e-6), faces are sorted by (ring, material_index), and each
edge stores its world-space endpoints in `edge.vertices` order. Point, edge, face, and corner attributes retain
position/ring association, including face-local point payloads and every named UV layer/pin; Blender
element indices are not comparison keys.
Material fields keep four distinct sources: raw OBJECT assignments (`object_materials`, null for DATA links), effective slot materials (`effective_materials`), raw DATA slots (`data_materials`), and per-slot link modes (`material_links`).
"""

import json
import os
import sys

import bpy

OUTPUT = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
EXPECTED = os.path.join(OUTPUT, "expected")

OPERATIONS = ("DIFFERENCE", "UNION", "INTERSECT")
BOX_FACES = (
    (0, 3, 2, 1),  # bottom
    (4, 5, 6, 7),  # top
    (0, 1, 5, 4),  # front
    (2, 3, 7, 6),  # back
    (0, 4, 7, 3),  # left
    (1, 2, 6, 5),  # right
)


def box(x0, x1, y0, y1, z0, z1):
    verts = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    return verts, list(BOX_FACES)


def remove_face(mesh, face_index):
    verts, faces = mesh
    return verts, [face for index, face in enumerate(faces) if index != face_index]


def input_mesh_geometry(spec):
    verts, faces = spec["mesh"]
    return {
        "vertices": [list(vertex) for vertex in verts],
        "faces": [list(face) for face in faces],
    }


def join(*meshes):
    verts, faces = [], []
    for mesh_verts, mesh_faces in meshes:
        offset = len(verts)
        verts.extend(mesh_verts)
        faces.extend(tuple(index + offset for index in face) for face in mesh_faces)
    return verts, faces


def part(mesh, **options):
    return {"mesh": mesh, **options}


def point_attributes(seed):
    return {
        "probe_float": ("FLOAT", lambda index: seed * 0.1 + index * 0.01),
        "probe_int": ("INT", lambda index: seed * 100 + index),
        "probe_vector": (
            "FLOAT_VECTOR",
            lambda index: (seed + index * 0.1, -seed - index * 0.2, index * 0.3),
        ),
        "probe_color": (
            "FLOAT_COLOR",
            lambda index: (
                seed * 0.01 + index * 0.001,
                seed * 0.005 + index * 0.002,
                0.2 + index * 0.003,
                0.4 + index * 0.01,
            ),
        ),
        "probe_bool": ("BOOLEAN", lambda index: (int(seed) + index) % 2 == 0),
        "probe_string": ("STRING", lambda index: f"{seed}:{index}"),
    }


# INT8 is intentionally generic in Potter for FACE/EDGE.
def domain_attributes(seed):
    return [
        ("EDGE", "probe_edge", "INT8", lambda index: seed % 50 + index),
        ("FACE", "probe_face", "INT8", lambda index: seed % 60 + index),
        ("EDGE", "probe_edge_string", "STRING", lambda index: f"edge:{seed}:{index}"),
        ("FACE", "probe_face_string", "STRING", lambda index: f"face:{seed}:{index}"),
    ]


def split_by_x(center):
    return 0 if center[0] < 0.0 else 1


def split_by_z(center):
    return 0 if center[2] < 1.0 else 1


def face_center(verts, face):
    return [sum(verts[i][axis] for i in face) / len(face) for axis in range(3)]


def canonical(co):
    # Nine significant figures resolve f32 coordinates at any scale the corpus uses (Blender stores
    # f32); six decimal places collapsed the 2^-30-scale tiny cases to zero.
    return tuple(float(f"{c:.9g}") + 0.0 for c in co)

def attribute_value(item, data_type):
    if data_type == "QUATERNION":
        value = item.value
        if hasattr(value, "x"):
            return [float(value.x), float(value.y), float(value.z), float(value.w)]
        # Blender 5.2 returns a plain (w, x, y, z) array for quaternion attributes.
        return [float(value[1]), float(value[2]), float(value[3]), float(value[0])]
    if data_type == "FLOAT4X4":
        return [[float(component) for component in row] for row in item.value]
    if data_type in ("FLOAT_COLOR", "BYTE_COLOR"):
        value = item.color
    elif hasattr(item, "vector"):
        value = item.vector
    else:
        value = item.value
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, (tuple, list)) or (
        hasattr(value, "__getitem__") and not isinstance(value, str)
    ):
        return list(value)
    return value


def domain_attribute_values(mesh, domain):
    uv_names = {layer.name for layer in mesh.uv_layers}
    values = {}
    for attribute in mesh.attributes:
        if (attribute.domain != domain or attribute.name.startswith(".")
                or attribute.name in {"position", "material_index", "sharp_face", "sharp_edge",
                                      "uv_seam", "custom_normal"}
                or (domain == "FACE" and attribute.name == "fid")
                or attribute.name in uv_names):
            continue
        values[attribute.name] = [
            attribute_value(item, attribute.data_type) for item in attribute.data
        ]
    for attribute in mesh.color_attributes:
        if attribute.domain == domain:
            values[attribute.name] = [
                attribute_value(item, attribute.data_type) for item in attribute.data
            ]
    return values


def smooth_faces(mesh):
    # Blender stores flat faces in `sharp_face`; without that attribute every face is smooth.
    sharp = mesh.attributes.get("sharp_face")
    if sharp is None:
        return [True] * len(mesh.polygons)
    return [not item.value for item in sharp.data]


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def case(family, section, solver, operation, host, operands, *, collection=False,
         use_self=False, material_mode="INDEX", suffix="", hole_tolerant=None,
         record_attribute_order=False):
    result = {
        "id": f"{family}_{operation.lower()}_{solver.lower()}{suffix}",
        "research_section": section,
        "solver": solver,
        "operation": operation,
        "use_self": use_self,
        "material_mode": material_mode,
        "host": host,
        "operands": operands,
        "collection": collection,
    }
    if hole_tolerant is not None:
        result["hole_tolerant"] = hole_tolerant
    if record_attribute_order:
        result["record_attribute_order"] = True
    return result


def specs():
    host = part(box(-1, 1, -1, 1, -1, 1))
    offset = part(box(0.5, 2.5, -0.5, 1.5, 0, 2))
    cases = []

    # 3.1 offset cube, EXACT (FLOAT identical, not recorded).
    cases += [case("offset", "3.1", "EXACT", op, host, [offset]) for op in OPERATIONS]

    # 3.2 nested cavity, EXACT.
    nested = part(box(-0.5, 0.5, -0.5, 0.5, -0.5, 0.5))
    cases += [case("nested", "3.2", "EXACT", op, host, [nested]) for op in OPERATIONS]

    # 3.3 coplanar, 3.4 touching and holes: EXACT asserted, FLOAT recorded for PR17b.
    coplanar = part(box(0.5, 2.5, -0.5, 1.5, -1, 1))
    touch = part(box(1, 3, -1, 1, -1, 1))
    through_hole = part(box(-0.5, 0.5, -0.5, 0.5, -2, 2))
    for family, section, operand in (
        ("coplanar", "3.3", coplanar),
        ("touch", "3.4", touch),
        ("hole_through", "3.4", through_hole),
    ):
        for solver in ("EXACT", "FLOAT"):
            cases += [case(family, section, solver, op, host, [operand]) for op in OPERATIONS]
    top_hole = part(box(-0.5, 0.5, -0.5, 0.5, 0, 2))
    cases += [
        case("hole_top", "3.4", solver, "DIFFERENCE", host, [top_hole])
        for solver in ("EXACT", "FLOAT")
    ]

    # 3.5 open operands, EXACT: both hole-tolerant paths and all three operations.
    open_operands = (
        ("open_top_offset", part(remove_face(box(0.5, 2.5, -0.5, 1.5, 0, 2), 1))),
        ("open_inside", part(remove_face(box(-0.5, 0.5, -0.5, 0.5, -0.5, 0.5), 1))),
        ("open_bottom_offset", part(remove_face(box(0, 2, -0.5, 0.5, 0, 2), 0))),
    )
    for family, operand in open_operands:
        for hole_tolerant in (False, True):
            suffix = "_hole_tolerant_on" if hole_tolerant else "_hole_tolerant_off"
            cases += [
                case(family, "3.5", "EXACT", op, host, [operand],
                     suffix=suffix, hole_tolerant=hole_tolerant)
                for op in OPERATIONS
            ]

    # 3.6 self-overlapping operand (two overlapping boxes in one object), EXACT.
    big_host = part(box(-2, 2, -2, 2, -2, 2))
    self_operand = part(join(box(0, 2, 0, 2, 0, 2), box(1, 3, 1, 3, 1, 3)))
    for op in OPERATIONS:
        for use_self in (False, True):
            cases.append(case(
                "self_overlap", "3.6", "EXACT", op, big_host, [self_operand],
                use_self=use_self, suffix="_use_self" if use_self else "_no_self",
            ))

    # 3.7 collection operand, EXACT (A ∩ C1 ∩ C2 is an N-ary intersection, not A ∩ (C1 ∪ C2)).
    members = [
        part(box(0.5, 2.5, -0.5, 1.5, 0, 2)),
        part(box(1, 3, -0.5, 1.5, -0.5, 1)),
    ]
    for op in OPERATIONS:
        for use_self in (False, True):
            cases.append(case(
                "collection", "3.7", "EXACT", op, host, members, collection=True,
                use_self=use_self, suffix="_use_self" if use_self else "_no_self",
            ))

    # 3.7 custom face attribute (fid) and UV map, EXACT.
    cases += [
        case("attributes", "3.7", "EXACT", op,
             part(box(-1, 1, -1, 1, -1, 1), fid=100, uv=0.0),
             [part(box(0.5, 2.5, -0.5, 1.5, 0, 2), fid=200, uv=5.0)])
        for op in OPERATIONS
    ]

    # 3.7 vertex groups: host weights by index, operand constant 0.9, EXACT.
    cases += [
        case("vertex_group", "3.7", "EXACT", op,
             part(box(-1, 1, -1, 1, -1, 1), weight=lambda index: 0.1 * (index + 1)),
             [part(box(0.5, 2.5, -0.5, 1.5, 0, 2), weight=lambda index: 0.9)])
        for op in ("DIFFERENCE", "UNION")
    ]

    # EXACT generic POINT attributes use the source vertex's local index in joined storage.
    point_host = part(box(-1, 1, -1, 1, -1, 1), point_attributes=point_attributes(11))
    point_operand = part(box(0.5, 2.5, -0.5, 1.5, 0, 2), point_attributes=point_attributes(70))
    cases.append(case("point_attributes", "3.7", "EXACT", "UNION", point_host, [point_operand]))

    # Each member has 16 vertices while the host has 8: the second member's indices 8..15 read
    # the first member's values from joined storage.
    point_members = [
        part(
            join(box(0.5, 2.5, -0.5, 1.5, 0, 2), box(3.5, 4.5, -0.5, 1.5, 0, 2)),
            point_attributes=point_attributes(70),
            domain_attributes=domain_attributes(70),
        ),
        part(
            join(box(1, 3, -0.5, 1.5, -0.5, 1), box(5, 6, -0.5, 1.5, -0.5, 1)),
            point_attributes=point_attributes(90),
            domain_attributes=domain_attributes(90),
        ),
    ]
    cases.append(case(
        "point_attributes_collection", "3.7", "EXACT", "UNION", point_host, point_members,
        collection=True, record_attribute_order=True,
    ))

    point_self_operand = part(
        join(box(0, 2, 0, 2, 0, 2), box(1, 3, 1, 3, 1, 3)),
        point_attributes=point_attributes(70),
    )
    point_big_host = part(box(-2, 2, -2, 2, -2, 2), point_attributes=point_attributes(11))
    cases.append(case(
        "point_attributes_self_overlap", "3.7", "EXACT", "UNION", point_big_host,
        [point_self_operand], use_self=True,
    ))

    point_open_operand = part(
        remove_face(box(0.5, 2.5, -0.5, 1.5, 0, 2), 1),
        point_attributes=point_attributes(70),
    )
    cases.append(case(
        "point_attributes_hole_tolerant", "3.7", "EXACT", "UNION", point_host,
        [point_open_operand], hole_tolerant=True,
    ))

    # 3.7 materials: host split by x, operand split by z; INDEX and TRANSFER, EXACT.
    # The FLOAT INDEX row is recorded for PR17b.
    material_host = part(box(-1, 1, -1, 1, -1, 1), materials=split_by_x)
    material_operand = [part(box(0.5, 2.5, -0.5, 1.5, 0, 2), materials=split_by_z)]
    cases.append(case("material_index", "3.7", "EXACT", "UNION", material_host,
                      material_operand, material_mode="INDEX"))
    cases.append(case("material_transfer", "3.7", "EXACT", "UNION", material_host,
                      material_operand, material_mode="TRANSFER"))
    cases.append(case("material_index", "3.7", "FLOAT", "UNION", material_host,
                      material_operand, material_mode="INDEX"))

    ids = [entry["id"] for entry in cases]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate case ids")
    return cases


def make_object(name, collection, spec):
    verts, faces = spec["mesh"]
    data = bpy.data.meshes.new(name)
    data.from_pydata(verts, [], faces)
    data.update()
    obj = bpy.data.objects.new(name, data)
    collection.objects.link(obj)
    if "materials" in spec:
        for slot in range(2):
            data.materials.append(bpy.data.materials.new(f"{name}.slot{slot}"))
        for polygon in data.polygons:
            polygon.material_index = spec["materials"](face_center(verts, faces[polygon.index]))
    if "uv" in spec:
        layer = data.uv_layers.new(name="UVMap")
        for polygon in data.polygons:
            for loop_index in polygon.loop_indices:
                x, y, _ = verts[data.loops[loop_index].vertex_index]
                layer.data[loop_index].uv = (x + spec["uv"], y + spec["uv"])
    if "fid" in spec:
        attribute = data.attributes.new("fid", "INT", "FACE")
        for polygon in data.polygons:
            attribute.data[polygon.index].value = spec["fid"] + polygon.index
    if "weight" in spec:
        group = obj.vertex_groups.new(name="G")
        for index in range(len(verts)):
            group.add([index], spec["weight"](index), "REPLACE")
    if "point_attributes" in spec:
        for name, (data_type, value_for_index) in spec["point_attributes"].items():
            attribute = data.attributes.new(name, data_type, "POINT")
            for index, item in enumerate(attribute.data):
                value = value_for_index(index)
                if data_type == "FLOAT_VECTOR":
                    item.vector = value
                elif data_type == "FLOAT_COLOR":
                    item.color = value
                elif data_type == "STRING":
                    item.value = value.encode("utf-8")
                else:
                    item.value = value
    if "domain_attributes" in spec:
        for domain, name, data_type, value_for_index in spec["domain_attributes"]:
            attribute = data.attributes.new(name, data_type, domain)
            for index, item in enumerate(attribute.data):
                value = value_for_index(index)
                item.value = value.encode("utf-8") if data_type == "STRING" else value
    return obj


def build_case(spec):
    scene = bpy.context.scene.collection
    if spec["collection"]:
        members = bpy.data.collections.new(f"{spec['id']}.members")
        scene.children.link(members)
        for index, operand in enumerate(spec["operands"]):
            make_object(f"{spec['id']}.member{index}", members, operand)
    else:
        (operand,) = spec["operands"]
        operand_object = make_object(f"{spec['id']}.operand", scene, operand)
    host = make_object(spec["id"], scene, spec["host"])
    modifier = host.modifiers.new("Boolean", "BOOLEAN")
    modifier.operation = spec["operation"]
    modifier.solver = spec["solver"]
    modifier.use_self = spec["use_self"]
    modifier.use_hole_tolerant = spec.get("hole_tolerant", False)
    modifier.material_mode = spec["material_mode"]
    if spec["collection"]:
        modifier.operand_type = "COLLECTION"
        modifier.collection = members
    else:
        modifier.operand_type = "OBJECT"
        modifier.object = operand_object
    return host


def describe(host, spec):
    bpy.context.view_layer.update()
    evaluated = host.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        return describe_mesh(evaluated, mesh, spec)
    finally:
        evaluated.to_mesh_clear()


def describe_mesh(evaluated, mesh, spec):
    matrix = evaluated.matrix_world
    raw = [tuple(float(c) for c in matrix @ vertex.co) for vertex in mesh.vertices]
    edges = sorted(
        [canonical(raw[index]) for index in edge.vertices]
        for edge in mesh.edges
    )
    point_values = domain_attribute_values(mesh, "POINT")
    edge_values = domain_attribute_values(mesh, "EDGE")
    face_values = domain_attribute_values(mesh, "FACE")
    corner_values = domain_attribute_values(mesh, "CORNER")
    if any(edge.use_edge_sharp for edge in mesh.edges):
        edge_values["sharp"] = [bool(edge.use_edge_sharp) for edge in mesh.edges]
    if any(edge.use_seam for edge in mesh.edges):
        edge_values["seam"] = [bool(edge.use_seam) for edge in mesh.edges]

    point_attributes = []
    for index, co in enumerate(raw):
        point_attributes.append({
            "co": canonical(co),
            "values": {name: values[index] for name, values in point_values.items()},
        })
    point_attributes.sort(key=lambda row: (row["co"], json.dumps(row["values"], sort_keys=True)))

    edge_attributes = []
    for index, edge in enumerate(mesh.edges):
        edge_attributes.append({
            "edge": [canonical(raw[vertex]) for vertex in edge.vertices],
            "values": {name: values[index] for name, values in edge_values.items()},
        })
    edge_attributes.sort(key=lambda row: (
        row["edge"], json.dumps(row["values"], sort_keys=True)
    ))
    edge_points = []
    for edge in mesh.edges:
        edge_points.append({
            "edge": [canonical(raw[index]) for index in edge.vertices],
            "endpoints": [
                {
                    "co": canonical(raw[index]),
                    "values": {name: values[index] for name, values in point_values.items()},
                }
                for index in edge.vertices
            ],
        })
    edge_points.sort(key=lambda row: (
        row["edge"], json.dumps(row["endpoints"], sort_keys=True)
    ))


    color_attributes = mesh.color_attributes
    color_names = [attribute.name for attribute in color_attributes]
    color_layers = None
    if color_names:
        active = color_attributes.active_color_index
        render = color_attributes.render_color_index
        color_layers = {
            "active": color_names[active] if 0 <= active < len(color_names) else None,
            "render": color_names[render] if 0 <= render < len(color_names) else None,
            "layers": [
                {
                    "name": attribute.name,
                    "domain": attribute.domain,
                    "data_type": attribute.data_type,
                }
                for attribute in color_attributes
            ],
        }

    uv_layer_names = [layer.name for layer in mesh.uv_layers]
    uv_active_index = mesh.uv_layers.active_index
    uv_render_index = mesh.uv_layers.active_render_index
    uv_layer_metadata = {
        "layers": uv_layer_names,
        "active": uv_layer_names[uv_active_index] if 0 <= uv_active_index < len(uv_layer_names) else None,
        "render": uv_layer_names[uv_render_index] if 0 <= uv_render_index < len(uv_layer_names) else None,
    }

    uv_data = mesh.uv_layers["UVMap"].data if "UVMap" in mesh.uv_layers else None
    normal_layer = mesh.attributes.get("custom_normal")
    smooth = smooth_faces(mesh)
    faces = []
    sizes = {}
    volume = 0.0
    for polygon in mesh.polygons:
        indices = list(polygon.vertices)
        sizes[len(indices)] = sizes.get(len(indices), 0) + 1
        ring = [canonical(raw[i]) for i in indices]
        loops = None
        if uv_data is not None:
            loops = [[round(uv_data[li].uv[0], 6) + 0.0, round(uv_data[li].uv[1], 6) + 0.0]
                     for li in polygon.loop_indices]
        named_uv_layers = {
            layer.name: {
                "uv": [
                    [round(layer.data[loop_index].uv[0], 6) + 0.0,
                     round(layer.data[loop_index].uv[1], 6) + 0.0]
                    for loop_index in polygon.loop_indices
                ],
                "pins": [bool(layer.data[loop_index].pin_uv) for loop_index in polygon.loop_indices],
            }
            for layer in mesh.uv_layers
        }
        corner_attributes = {
            name: [values[loop_index] for loop_index in polygon.loop_indices]
            for name, values in corner_values.items()
        }
        normal_codes = None
        normal_vectors = None
        if normal_layer is not None:
            normal_codes = [
                [int(component) for component in normal_layer.data[loop_index].value]
                for loop_index in polygon.loop_indices
            ]
            normal_vectors = [
                [float(component) for component in mesh.corner_normals[loop_index].vector]
                for loop_index in polygon.loop_indices
            ]
        face_attributes = {
            name: values[polygon.index] for name, values in face_values.items()
        }
        face_attributes["shade_smooth"] = smooth[polygon.index]
        point_attribute_rows = [
            {name: values[index] for name, values in point_values.items()}
            for index in indices
        ]
        uv_pins = (
            [bool(uv_data[loop_index].pin_uv) for loop_index in polygon.loop_indices]
            if uv_data is not None else None
        )
        start = min(range(len(ring)), key=ring.__getitem__)
        ring = ring[start:] + ring[:start]
        record = {"ring": ring, "material_index": polygon.material_index}
        if loops is not None:
            record["uv"] = loops[start:] + loops[:start]
        record["uv_layers"] = {
            name: {
                "uv": values["uv"][start:] + values["uv"][:start],
                "pins": values["pins"][start:] + values["pins"][:start],
            }
            for name, values in named_uv_layers.items()
        }
        record["face_attributes"] = face_attributes
        if corner_attributes:
            record["corner_attributes"] = {
                name: values[start:] + values[:start]
                for name, values in corner_attributes.items()
            }
        if normal_codes is not None:
            record["custom_normal_codes"] = normal_codes[start:] + normal_codes[:start]
        if normal_vectors is not None:
            record["custom_normals"] = normal_vectors[start:] + normal_vectors[:start]
        if "fid" in mesh.attributes:
            record["fid"] = mesh.attributes["fid"].data[polygon.index].value
        record["point_attributes"] = (
            point_attribute_rows[start:] + point_attribute_rows[:start]
        )
        if uv_pins is not None:
            record["uv_pins"] = uv_pins[start:] + uv_pins[:start]
        faces.append(record)
        p0 = raw[indices[0]]
        for k in range(1, len(indices) - 1):
            volume += dot(p0, cross(raw[indices[k]], raw[indices[k + 1]])) / 6.0
    faces.sort(key=lambda face: (face["ring"], face["material_index"]))

    group_names = {group.index: group.name for group in evaluated.vertex_groups}
    groups = {name: [] for name in group_names.values()}
    for vertex, co in zip(mesh.vertices, raw):
        weights = {group_names[group.group]: round(group.weight, 6) for group in vertex.groups}
        for name, entries in groups.items():
            entries.append({"co": canonical(co), "weight": weights.get(name)})
    for entries in groups.values():
        entries.sort(key=lambda entry: entry["co"])

    # Blender `BKE_object_material_get_eval` resolves an OBJECT-null slot through DATA; compute the
    # effective table explicitly from the raw object table, links, and evaluated mesh's DATA table.
    source = bpy.data.objects[spec["id"]]
    source_slots = list(source.material_slots)
    object_materials = []
    material_links = []
    for index in range(len(evaluated.material_slots)):
        if index < len(source_slots):
            slot = source_slots[index]
            object_materials.append(
                slot.material.name if slot.link == "OBJECT" and slot.material is not None else None
            )
            material_links.append(slot.link.lower())
        else:
            object_materials.append(None)
            material_links.append("data")

    data_materials = [
        material.name if material is not None else None
        for material in mesh.materials
    ]
    # Effective names are Blender's evaluated resolution (BKE_object_material_get_eval on the
    # evaluated object), not a Python re-derivation from raw tables. The OBJECT-null-over-DATA
    # case falls through to the DATA slot here and is the source witness for that resolution.
    effective_materials = [
        slot.material.name if slot.material is not None else None
        for slot in evaluated.material_slots
    ]

    result = {
        "case": spec["id"],
        "research_section": spec["research_section"],
        "solver": spec["solver"],
        "operation": spec["operation"],
        "use_self": spec["use_self"],
        "material_mode": spec["material_mode"],
        "object_materials": object_materials,
        "effective_materials": effective_materials,
        "data_materials": data_materials,
        "material_links": material_links,
        "counts": {
            "vertices": len(mesh.vertices),
            "edges": len(mesh.edges),
            "faces": len(mesh.polygons),
        },
        "face_sizes": {str(size): count for size, count in sorted(sizes.items())},
        "volume": volume,
        "edges": edges,
        "edge_points": edge_points,
        "faces": faces,
        "uv_layers": uv_layer_metadata,
        "vertex_groups": groups,
        "blender_version": bpy.app.version_string,
    }
    # Always written: null means this mesh has no data in that domain, not an omitted section.
    result["edge_attributes"] = edge_attributes if edge_values else None
    result["point_attributes"] = point_attributes if point_values else None
    result["color_layers"] = color_layers
    if spec.get("record_attribute_order"):
        result["blender_attribute_order"] = [
            [attribute.domain, attribute.name, attribute.data_type]
            for attribute in mesh.attributes
            if attribute.name in {"probe_edge", "probe_face"}
        ]
    if "hole_tolerant" in spec:
        result["hole_tolerant"] = spec["hole_tolerant"]
        result["input_geometry"] = {
            "host": input_mesh_geometry(spec["host"]),
            "operands": [input_mesh_geometry(operand) for operand in spec["operands"]],
        }
    return result


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    cases = specs()
    hosts = [(spec, build_case(spec)) for spec in cases]
    os.makedirs(EXPECTED, exist_ok=True)
    for name in os.listdir(EXPECTED):
        if name.endswith(".json"):
            os.remove(os.path.join(EXPECTED, name))
    for spec, host in hosts:
        result = describe(host, spec)
        with open(os.path.join(EXPECTED, spec["id"] + ".json"), "w", encoding="utf-8") as output:
            json.dump(result, output, indent=1)
            output.write("\n")
        counts = result["counts"]
        print(
            f"{spec['id']}: V={counts['vertices']} E={counts['edges']} F={counts['faces']} "
            f"volume={result['volume']:.6f} sizes={result['face_sizes']}",
            flush=True,
        )
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUTPUT, "scene.blend"))
    print("WROTE", len(hosts), "cases to", EXPECTED, flush=True)


main()
