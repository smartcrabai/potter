"""Blender 5.2 Boolean parity corpus generator for tests/blend_boolean_parity.rs.

Run headless from the repository root (`--factory-startup` keeps the run reproducible):

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
        --python tests/fixtures/boolean_parity/generate_blender_expectations.py -- \
        tests/fixtures/boolean_parity

Outputs under the directory given after `--`:

    expected/<case>.json   one expectation per case (committed corpus)
    scene.blend            every case in one scene; the Potter import source (git-ignored)

Cases reproduce research/§3 of the PR17 design: §3.1 offset, §3.2 nested, §3.3 coplanar,
§3.4 face-touching and holes, §3.6 self-overlapping operand, §3.7 collection operand,
attributes, vertex groups, and material INDEX/TRANSFER. EXACT cases are the parity oracle.
FLOAT cases of §3.3, §3.4, and the §3.7 INDEX material row are recorded as Blender's
defective outputs for PR17b; the Rust driver skips them.

Each case is a Boolean modifier on a host object named after the case. Every object keeps
the identity transform, so object-local and world coordinates coincide. All cases share one
scene, and the Rust driver evaluates each host with only its own Boolean enabled. Boolean
evaluation is independent per host, so the shared scene gives the same result as one scene
per case.

Expectations are canonical: each face ring is rotated to start at its minimum vertex
(coordinates rounded to 1e-6), faces are sorted by (ring, material_index), and vertex groups
are sorted by rounded coordinates. Object-index order is not part of the contract, because
FLOAT vertex numbering varies between runs.
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


def join(*meshes):
    verts, faces = [], []
    for mesh_verts, mesh_faces in meshes:
        offset = len(verts)
        verts.extend(mesh_verts)
        faces.extend(tuple(index + offset for index in face) for face in mesh_faces)
    return verts, faces


def part(mesh, **options):
    return {"mesh": mesh, **options}


def split_by_x(center):
    return 0 if center[0] < 0.0 else 1


def split_by_z(center):
    return 0 if center[2] < 1.0 else 1


def face_center(verts, face):
    return [sum(verts[i][axis] for i in face) / len(face) for axis in range(3)]


def canonical(co):
    return tuple(round(c, 6) + 0.0 for c in co)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def case(family, section, solver, operation, host, operands, *, collection=False,
         use_self=False, material_mode="INDEX", suffix=""):
    return {
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
    modifier.use_hole_tolerant = False
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
    group_names = {group.index: group.name for group in evaluated.vertex_groups}
    uv_data = mesh.uv_layers["UVMap"].data if "UVMap" in mesh.uv_layers else None
    has_fid = "fid" in mesh.attributes
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
        start = min(range(len(ring)), key=ring.__getitem__)
        ring = ring[start:] + ring[:start]
        record = {"ring": ring, "material_index": polygon.material_index}
        if loops is not None:
            loops = loops[start:] + loops[:start]
            record["uv"] = loops
        if has_fid:
            record["fid"] = mesh.attributes["fid"].data[polygon.index].value
        faces.append(record)
        p0 = raw[indices[0]]
        for k in range(1, len(indices) - 1):
            volume += dot(p0, cross(raw[indices[k]], raw[indices[k + 1]])) / 6.0
    faces.sort(key=lambda face: (face["ring"], face["material_index"]))

    groups = {name: [] for name in group_names.values()}
    for vertex, co in zip(mesh.vertices, raw):
        weights = {group_names[group.group]: round(group.weight, 6) for group in vertex.groups}
        for name, entries in groups.items():
            entries.append({"co": canonical(co), "weight": weights.get(name)})
    for entries in groups.values():
        entries.sort(key=lambda entry: entry["co"])

    return {
        "case": spec["id"],
        "research_section": spec["research_section"],
        "solver": spec["solver"],
        "operation": spec["operation"],
        "use_self": spec["use_self"],
        "material_mode": spec["material_mode"],
        "counts": {
            "vertices": len(mesh.vertices),
            "edges": len(mesh.edges),
            "faces": len(mesh.polygons),
        },
        "face_sizes": {str(size): count for size, count in sorted(sizes.items())},
        "volume": volume,
        "faces": faces,
        "vertex_groups": groups,
        "blender_version": bpy.app.version_string,
    }


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
