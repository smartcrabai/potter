"""Blender 5.2.2 expectations for the EXACT Boolean regressions in tests/blend_boolean_regressions.rs.

Run headless from the repository root (`--factory-startup` keeps the run reproducible):

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
        --python tests/fixtures/boolean_parity/regressions/generate_regression_expectations.py -- \
        tests/fixtures/boolean_parity/regressions

Outputs under the directory given after `--`:

    expected/<case>.json   one expectation per case (committed corpus)
    scene.blend            every case in one scene; the Potter import source (git-ignored)

Each case is a host object named after the case. Hosts evaluate independently, so the shared
scene gives the same result as one scene per case. Expectations use the canonical form of the
parity corpus (tests/fixtures/boolean_parity/generate_blender_expectations.py): world-space face
rings rotated to start at their minimum vertex, faces sorted, coordinates rounded to 1e-6.

Kinds:
- `evaluated`: the host's Boolean result, with that Boolean as the only enabled modifier.
- `applied`: the host after `modifier.apply` of its Boolean, run on a copy. `materials` lists the
  applied material slots by name, `null` for an empty slot.

Face records carry `custom_normals` (per corner, world space) where the Blender mesh has custom
normals. Spheres record counts, face sizes, and volume only (`rings` is false): their topology
is pinned by the counts and the face-size histogram.
"""

import json
import math
import os
import sys

import bmesh
import bpy
from mathutils import Matrix, Vector

if bpy.app.version != (5, 2, 2):
    raise RuntimeError("Boolean regression expectations require Blender 5.2.2")

OUTPUT = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
EXPECTED = os.path.join(OUTPUT, "expected")

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


def material(name):
    return bpy.data.materials.get(name) or bpy.data.materials.new(name)


def new_collection(name, parent):
    collection = bpy.data.collections.new(name)
    parent.children.link(collection)
    return collection


def mesh_object(
    name, collection, geometry, *, slots=(), slot_of_face=None, custom_normals=False, edges=(),
):
    verts, faces = geometry
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, edges, faces)
    mesh.update()
    for slot in slots:
        mesh.materials.append(None if slot is None else material(slot))
    if slot_of_face is not None:
        for polygon in mesh.polygons:
            polygon.material_index = slot_of_face(polygon.index)
    if custom_normals:
        loop_normals = [None] * len(mesh.loops)
        for polygon in mesh.polygons:
            for loop_index in polygon.loop_indices:
                loop_normals[loop_index] = tuple(polygon.normal)
        mesh.normals_split_custom_set(loop_normals)
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    return obj


def primitive(operator, name, **arguments):
    operator(**arguments)
    obj = bpy.context.object
    obj.name = name
    return obj


def canonical_order(obj):
    """Renumber a mesh so its order does not depend on the Blender process.

    Blender 5.2.2's UV sphere primitive keeps the same vertex positions in every process but emits
    them, and its faces, in an order that changes from run to run. Vertices are sorted by exact
    position, each face starts at its lowest vertex index, and faces are sorted, so the Boolean
    input (and the edge order Blender builds from it) is one fixed mesh per run.
    """
    mesh = obj.data
    coords = [tuple(vertex.co) for vertex in mesh.vertices]
    order = sorted(range(len(coords)), key=lambda index: coords[index])
    renumber = {old: new for new, old in enumerate(order)}
    faces = []
    for polygon in mesh.polygons:
        ring = [renumber[index] for index in polygon.vertices]
        start = ring.index(min(ring))
        faces.append(tuple(ring[start:] + ring[:start]))
    faces.sort()
    mesh.clear_geometry()
    mesh.from_pydata([coords[index] for index in order], [], faces)
    mesh.update()


def uv_sphere(name, segments, open_top):
    obj = primitive(
        bpy.ops.mesh.primitive_uv_sphere_add, name,
        segments=segments, ring_count=segments // 2, radius=1.0,
    )
    canonical_order(obj)
    if open_top:
        # Same cut as the probes: faces with every vertex above z = 0.95 are removed.
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        cap = [face for face in bm.faces if all(vert.co.z > 0.95 for vert in face.verts)]
        bmesh.ops.delete(bm, geom=cap, context="FACES")
        bm.to_mesh(obj.data)
        bm.free()
    return obj


def boolean(host, operation, solver, *, operand=None, collection=None, use_self=False,
            hole_tolerant=False, material_mode="INDEX"):
    modifier = host.modifiers.new("Boolean", "BOOLEAN")
    modifier.operation = operation
    modifier.solver = solver
    modifier.use_self = use_self
    modifier.use_hole_tolerant = hole_tolerant
    modifier.material_mode = material_mode
    if collection is not None:
        modifier.operand_type = "COLLECTION"
        modifier.collection = collection
    elif operand is not None:
        modifier.operand_type = "OBJECT"
        modifier.object = operand
    return modifier


def rotated(values, start):
    return values[start:] + values[:start]


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


def smooth_faces(mesh):
    # Blender stores flat faces in `sharp_face`; without that attribute every face is smooth.
    sharp = mesh.attributes.get("sharp_face")
    if sharp is None:
        return [True] * len(mesh.polygons)
    return [not item.value for item in sharp.data]


def describe_mesh(case, regression, owner, mesh, *, rings):
    matrix = owner.matrix_world
    raw = [tuple(float(c) for c in matrix @ vertex.co) for vertex in mesh.vertices]
    edges = sorted(
        [canonical(raw[index]) for index in edge.vertices]
        for edge in mesh.edges
    )
    uv_layer = mesh.uv_layers["UVMap"] if "UVMap" in mesh.uv_layers else None
    custom = mesh.has_custom_normals
    smooth = smooth_faces(mesh)
    faces = []
    sizes = {}
    volume = 0.0
    for polygon in mesh.polygons:
        indices = list(polygon.vertices)
        corners = list(polygon.loop_indices)
        sizes[len(indices)] = sizes.get(len(indices), 0) + 1
        ring = [canonical(raw[i]) for i in indices]
        start = min(range(len(ring)), key=ring.__getitem__)
        if rings:
            record = {"ring": rotated(ring, start), "material_index": polygon.material_index}
            record["face_attributes"] = {"shade_smooth": smooth[polygon.index]}
            if uv_layer is not None:
                record["uv"] = rotated(
                    [[round(uv_layer.data[li].uv[0], 6) + 0.0,
                      round(uv_layer.data[li].uv[1], 6) + 0.0] for li in corners],
                    start,
                )
            if custom:
                normal_layer = mesh.attributes["custom_normal"]
                record["custom_normal_codes"] = rotated(
                    [[int(component) for component in normal_layer.data[li].value] for li in corners],
                    start,
                )
                record["custom_normals"] = rotated(
                    [[round(c, 6) + 0.0 for c in mesh.corner_normals[li].vector] for li in corners],
                    start,
                )
            faces.append(record)
        p0 = raw[indices[0]]
        for k in range(1, len(indices) - 1):
            volume += dot(p0, cross(raw[indices[k]], raw[indices[k + 1]])) / 6.0
    if rings:
        faces.sort(key=lambda face: (face["ring"], face["material_index"]))

    group_names = {group.index: group.name for group in owner.vertex_groups}
    groups = {name: [] for name in group_names.values()}
    for vertex, co in zip(mesh.vertices, raw):
        weights = {group_names[group.group]: round(group.weight, 6) for group in vertex.groups}
        for name, entries in groups.items():
            entries.append({"co": canonical(co), "weight": weights.get(name)})
    for entries in groups.values():
        entries.sort(key=lambda entry: entry["co"])

    result = {
        "case": case,
        "regression": regression,
        "counts": {
            "vertices": len(mesh.vertices),
            "edges": len(mesh.edges),
            "faces": len(mesh.polygons),
        },
        "face_sizes": {str(size): count for size, count in sorted(sizes.items())},
        "volume": volume,
        "vertex_groups": groups,
        "blender_version": bpy.app.version_string,
    }
    # Always written: the sphere cases (rings=False) record counts, face sizes, and volume only.
    result["faces"] = faces if rings else None
    result["edges"] = edges
    return result


def evaluate(case, regression, host, *, rings):
    bpy.context.view_layer.update()
    evaluated = host.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        result = describe_mesh(case, regression, evaluated, mesh, rings=rings)
    finally:
        evaluated.to_mesh_clear()
    result["kind"] = "evaluated"
    return result


def apply_copy(case, regression, host):
    copy = host.copy()
    copy.data = host.data.copy()
    copy.name = case + ".applied"
    bpy.context.scene.collection.objects.link(copy)
    bpy.context.view_layer.objects.active = copy
    with bpy.context.temp_override(object=copy, active_object=copy):
        bpy.ops.object.modifier_apply(modifier=copy.modifiers[0].name)
    bpy.context.view_layer.update()
    result = describe_mesh(case, regression, copy, copy.data, rings=True)
    result["kind"] = "applied"
    result["materials"] = [
        slot.material.name if slot.material is not None else None
        for slot in copy.material_slots
    ]
    bpy.data.objects.remove(copy, do_unlink=True)
    return result


def build_scene(scene):
    """Returns (case, regression, host, rings, applied) for every expectation."""
    cases = []

    def add(case, regression, host, *, rings=True, applied=False):
        cases.append((case, regression, host, rings, applied))

    # b: an unset OBJECT Boolean leaves a closed mesh unchanged.
    monkey = primitive(bpy.ops.mesh.primitive_monkey_add, "unset_object_monkey", size=2.0)
    boolean(monkey, "DIFFERENCE", "EXACT")
    add("unset_object_monkey", "b", monkey)
    cube = primitive(
        bpy.ops.mesh.primitive_cube_add, "unset_object_cube", size=2.0, location=(6.0, 0.0, 0.0)
    )
    boolean(cube, "DIFFERENCE", "EXACT")
    add("unset_object_cube", "b", cube)

    # c: an empty COLLECTION operand, EXACT, on a self-overlapping mesh.
    empty = new_collection("regress.empty.members", scene)
    for use_self in (False, True):
        case = "empty_collection_use_self" if use_self else "empty_collection_no_self"
        host = mesh_object(case, scene, join(box(0, 2, 0, 2, 0, 2), box(1, 3, 1, 3, 1, 3)))
        boolean(host, "DIFFERENCE", "EXACT", collection=empty, use_self=use_self)
        add(case, "c", host)

    # d: the subject is also a member of its operand collection. Blender skips the subject.
    for operation in ("DIFFERENCE", "UNION"):
        case = "subject_in_collection_" + operation.lower()
        members = new_collection(case + ".members", scene)
        host = mesh_object(case, scene, box(-1, 1, -1, 1, -1, 1))
        members.objects.link(host)
        mesh_object(case + ".member", members, box(0.5, 2.5, -0.5, 1.5, 0, 2))
        boolean(host, operation, "EXACT", collection=members)
        add(case, "d", host)

    # e: INDEX remap of operand slots onto the subject's slots.
    def indexed_pair(case, host_slots, cutter_slots):
        host = mesh_object(
            case, scene, box(-1, 1, -1, 1, -1, 1),
            slots=host_slots, slot_of_face=lambda index: index % len(host_slots),
        )
        cutter = mesh_object(
            case + ".cutter", scene, box(0.5, 2.5, -0.5, 1.5, 0, 2),
            slots=cutter_slots, slot_of_face=lambda index: index % len(cutter_slots),
        )
        boolean(host, "UNION", "EXACT", operand=cutter, material_mode="INDEX")
        return host

    prefix = "regress.index_one"
    add("index_remap_one_slot", "e",
        indexed_pair("index_remap_one_slot", [prefix + ".A"], [prefix + ".X", prefix + ".Y"]))
    prefix = "regress.index_keep"
    add("index_remap_same_material_keeps_slot", "e",
        indexed_pair("index_remap_same_material_keeps_slot",
                     [prefix + ".A", prefix + ".B", prefix + ".C"],
                     [prefix + ".X", prefix + ".B"]))
    prefix = "regress.index_first"
    add("index_remap_first_matching_slot", "e",
        indexed_pair("index_remap_first_matching_slot",
                     [prefix + ".A", prefix + ".B", prefix + ".A", prefix + ".D"],
                     [prefix + ".B", prefix + ".A", prefix + ".C", prefix + ".E"]))

    # f: TRANSFER into a subject without material slots, then modifier.apply.
    prefix = "regress.transfer_empty"
    transfer_host = mesh_object("transfer_empty_subject", scene, box(-1, 1, -1, 1, -1, 1))
    transfer_cutter = mesh_object(
        "transfer_empty_subject.cutter", scene, box(0.5, 2.5, -0.5, 1.5, 0, 2),
        slots=[prefix + ".M1", prefix + ".M2"], slot_of_face=lambda index: index % 2,
    )
    boolean(transfer_host, "UNION", "EXACT", operand=transfer_cutter, material_mode="TRANSFER")
    add("transfer_empty_subject", "f", transfer_host, applied=True)

    # g: touching cubes rotated about Z, with the operand placed in the subject's frame
    # (the rot.py probe; expectations are world space).
    for index, degrees in enumerate((10, 17, 23, 30, 37, 41, 53, 61)):
        location = (0.1 * index, 0.37 * index, 0.0)
        radians = math.radians(degrees)
        host = primitive(
            bpy.ops.mesh.primitive_cube_add, f"rotated_touch_{degrees}",
            size=2.0, location=location, rotation=(0, 0, radians),
        )
        rotation = Matrix.Rotation(radians, 4, "Z")
        cutter_location = Vector(location) + (rotation @ Vector((2.0, 0.5, 0.0)))
        cutter = primitive(
            bpy.ops.mesh.primitive_cube_add, f"rotated_touch_{degrees}.cutter",
            size=2.0, location=cutter_location, rotation=(0, 0, radians),
        )
        boolean(host, "UNION", "EXACT", operand=cutter)
        add(f"rotated_touch_{degrees}", "g", host)

    # h: a cavity wall keeps the cutter's custom normal, flipped. The normals are authored in Potter
    # by mesh.set_custom_normals with the same face normals (the regression test does that).
    host = mesh_object("cavity_wall_normal", scene, box(-1, 1, -1, 1, -1, 1), custom_normals=True)
    cutter = mesh_object(
        "cavity_wall_normal.cutter", scene, box(-0.5, 0.5, -0.5, 0.5, -0.5, 0.5),
        custom_normals=True,
    )
    boolean(host, "DIFFERENCE", "EXACT", operand=cutter)
    add("cavity_wall_normal", "h", host)

    # i: open and closed UV spheres with use_hole_tolerant.
    for case, segments, open_top, hole_tolerant in (
        ("sphere_open64_hole_tolerant_on", 64, True, True),
        ("sphere_open64_hole_tolerant_off", 64, True, False),
        ("sphere_open128_hole_tolerant_on", 128, True, True),
        ("sphere_closed64_hole_tolerant_on", 64, False, True),
    ):
        host = uv_sphere(case, segments, open_top)
        cutter = primitive(
            bpy.ops.mesh.primitive_cube_add, case + ".cutter", size=1.0, location=(0.9, 0.0, 0.0)
        )
        boolean(host, "DIFFERENCE", "EXACT", operand=cutter, hole_tolerant=hole_tolerant)
        add(case, "i", host, rings=False)
    # Raw Blender sphere orders (dump_sphere_input.py): one case per committed input file.
    inputs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs")
    for name in sorted(os.listdir(inputs)):
        if not name.endswith(".json"):
            continue
        case = name[: -len(".json")]
        with open(os.path.join(inputs, name)) as handle:
            data = json.load(handle)
        host = mesh_object(
            case, scene, ([tuple(vertex) for vertex in data["verts"]],
                          [tuple(face) for face in data["faces"]]),
            edges=[tuple(edge) for edge in data["edges"]],
        )
        cutter = primitive(
            bpy.ops.mesh.primitive_cube_add, case + ".cutter", size=1.0, location=(0.9, 0.0, 0.0)
        )
        boolean(host, "DIFFERENCE", "EXACT", operand=cutter, hole_tolerant=True)
        add(case, "i", host, rings=False)
    return cases


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    cases = build_scene(bpy.context.scene.collection)
    os.makedirs(EXPECTED, exist_ok=True)
    for name in os.listdir(EXPECTED):
        if name.endswith(".json"):
            os.remove(os.path.join(EXPECTED, name))
    for case, regression, host, rings, applied in cases:
        if applied:
            result = apply_copy(case, regression, host)
        else:
            result = evaluate(case, regression, host, rings=rings)
        with open(os.path.join(EXPECTED, case + ".json"), "w", encoding="utf-8") as output:
            json.dump(result, output, indent=1)
            output.write("\n")
        counts = result["counts"]
        print(
            f"{case}: V={counts['vertices']} E={counts['edges']} F={counts['faces']} "
            f"volume={result['volume']:.6f} sizes={result['face_sizes']} "
            f"materials={result.get('materials')}",
            flush=True,
        )
    # The import scene carries no custom normals: the regression test authors them in Potter with
    # mesh.set_custom_normals, from the same face normals. Expectations above used them in Blender.
    for obj in bpy.data.objects:
        if obj.type == "MESH" and "custom_normal" in obj.data.attributes:
            obj.data.attributes.remove(obj.data.attributes["custom_normal"])
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUTPUT, "scene.blend"))
    print("WROTE", len(cases), "cases to", EXPECTED, flush=True)


main()
