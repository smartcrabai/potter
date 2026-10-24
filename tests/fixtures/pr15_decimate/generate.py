# Usage: /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup \
#        --python tests/fixtures/pr15_decimate/generate.py -- tests/fixtures/pr15_decimate
#
# Writes one <case>.json per case into the output directory. Each file holds the input mesh exactly as
# Blender stores it (vertices, faces, edges in Blender's edge order with flags, material indices, UV
# corners, vertex-group weights) and the Blender 5.2.2 evaluated result of the Decimate modifier.
# `tests/modifier_decimate_parity.rs` rebuilds the input in Potter and asserts the same output.
# Pass `--only <substring>` before the output directory to regenerate a subset.
#
# `inputs/float1_edge_potter_order.json` is hand-committed and no generator writes it. Its positions and
# faces are Blender 5.2.2's Boolean FLOAT-1 result; its edge list is in the order Potter's FLOAT output
# emits (the `note` field in the file). The Potter command or commit that produced that edge order is not
# recorded in the repository. The case `float1_edge_potter_r0.6` decimates this mesh in Blender, so its
# expected output is Blender's. This script only reads the file.
import json
import math
import os
import sys

import bpy
import bmesh

ARGS = sys.argv[sys.argv.index("--") + 1:]
ONLY = None
if ARGS and ARGS[0] == "--only":
    ONLY = ARGS[1]
    ARGS = ARGS[2:]
OUT = ARGS[0]
os.makedirs(OUT, exist_ok=True)

BLENDER_VERSION = "5.2.2 LTS"


# ---------------------------------------------------------------- meshes
def cube_mesh():
    verts = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    verts = [(x * 2 - 1, y * 2 - 1, z * 2 - 1) for x, y, z in verts]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (3, 7, 6, 2), (0, 4, 7, 3), (1, 2, 6, 5)]
    return verts, faces


def uv_sphere(segments=16, rings=8):
    verts = [(0.0, 0.0, 1.0)]
    for r in range(1, rings):
        phi = math.pi * r / rings
        for s in range(segments):
            theta = 2.0 * math.pi * s / segments
            verts.append((math.sin(phi) * math.cos(theta), math.sin(phi) * math.sin(theta), math.cos(phi)))
    verts.append((0.0, 0.0, -1.0))
    bottom = len(verts) - 1
    faces = []
    for s in range(segments):
        faces.append((0, 1 + (s + 1) % segments, 1 + s))
    for r in range(rings - 2):
        a = 1 + r * segments
        b = a + segments
        for s in range(segments):
            faces.append((a + s, a + (s + 1) % segments, b + (s + 1) % segments, b + s))
    last = 1 + (rings - 2) * segments
    for s in range(segments):
        faces.append((last + s, bottom, last + (s + 1) % segments))
    return verts, faces


def grid_mesh(n, wave):
    verts = []
    for j in range(n):
        for i in range(n):
            x = -1.0 + 2.0 * i / (n - 1)
            y = -1.0 + 2.0 * j / (n - 1)
            z = 0.12 * math.cos(x * 2.0) * math.sin(y * 2.0) if wave else 0.0
            verts.append((x, y, z))
    faces = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            faces.append((a, a + 1, a + n + 1, a + n))
    return verts, faces


def cylinder_mesh(segments=16, rings=6, caps=True):
    verts = []
    for r in range(rings):
        z = -1.0 + 2.0 * r / (rings - 1)
        for s in range(segments):
            theta = 2.0 * math.pi * s / segments
            verts.append((0.8 * math.cos(theta), 0.8 * math.sin(theta) * (1.0 + 0.1 * z), z))
    faces = []
    for r in range(rings - 1):
        for s in range(segments):
            a = r * segments + s
            b = r * segments + (s + 1) % segments
            faces.append((a, b, b + segments, a + segments))
    if caps:
        faces.append(tuple(range(segments)))
        faces.append(tuple((rings - 1) * segments + s for s in reversed(range(segments))))
    return verts, faces


def subdivided_cube(levels):
    bm = bmesh.new()
    verts, faces = cube_mesh()
    vs = [bm.verts.new(v) for v in verts]
    for f in faces:
        bm.faces.new([vs[i] for i in f])
    for _ in range(levels):
        bmesh.ops.subdivide_edges(bm, edges=list(bm.edges), cuts=1, use_grid_fill=True)
    out_verts = [tuple(v.co) for v in bm.verts]
    out_faces = [tuple(v.index for v in f.verts) for f in bm.faces]
    bm.free()
    return out_verts, out_faces


def build_base(spec):
    kind = spec["mesh"]
    if kind == "cube":
        return cube_mesh()
    if kind == "sphere":
        return uv_sphere(spec.get("segments", 16), spec.get("rings", 8))
    if kind == "grid":
        return grid_mesh(spec["n"], spec.get("wave", True))
    if kind == "cylinder":
        return cylinder_mesh(spec.get("segments", 16), spec.get("rings", 6), spec.get("caps", True))
    if kind == "subdivided_cube":
        return subdivided_cube(spec["levels"])
    if kind == "explicit":
        data = explicit_mesh(spec)
        return [tuple(v) for v in data["positions"]], [tuple(f) for f in data["faces"]]
    if kind == "shared_vertex_fans":
        verts = [(0.0, 0.0, 0.0)]
        faces = []
        for fan in range(2):
            inner = []
            outer = []
            for i in range(3):
                angle = fan * math.pi + math.pi * 0.5 * i / 2
                inner_radius = 1.0 + 0.1 * math.sin((i + 1) * 0.7 + fan)
                outer_radius = 2.0 + 0.2 * math.cos((i + 1) * 0.8 + fan)
                inner.append(len(verts))
                verts.append((inner_radius * math.cos(angle),
                              inner_radius * math.sin(angle),
                              0.3 * math.sin((i + 1) * 0.6 + fan)))
                outer.append(len(verts))
                verts.append((outer_radius * math.cos(angle),
                              outer_radius * math.sin(angle),
                              0.4 * math.cos((i + 1) * 1.4 + fan)))
            for i in range(2):
                faces.append((0, inner[i], outer[i], outer[i + 1], inner[i + 1]))
        return verts, faces
    raise ValueError(kind)


def explicit_mesh(spec):
    # Mesh data from a committed JSON file; edges keep the file's order.
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), spec["file"])
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------- mesh decoration
def uv_value(kind, co, corner, layer):
    if kind == "random":
        a = 64 + (corner * 37 + layer * 11) % 800
        b = 64 + (corner * 53 + layer * 29 + 7) % 800
        return [a / 1024.0, b / 1024.0]
    if kind == "continuous":
        x, y = co[0], co[1]
        if layer == 0:
            return [(128 + 200 * x + 50 * y) / 1024.0, (300 + 100 * y + 30 * x * x) / 1024.0]
        return [(128 + 200 * y + 50 * x) / 1024.0, (300 + 100 * x + 30 * y * y) / 1024.0]
    if kind == "offset":
        return [co[0] + 1.0 + (4.0 if layer == 1 else 0.0), co[1] + 1.0]
    raise ValueError(kind)


def decorate(me, obj, spec):
    if spec.get("uv"):
        names = ["UVMap", "UV2"][: spec.get("uv_layers", 1)]
        for layer_index, name in enumerate(names):
            layer = me.uv_layers.new(name=name)
            corner = 0
            for poly in me.polygons:
                for vi in poly.vertices:
                    co = me.vertices[vi].co
                    layer.data[corner].uv = uv_value(spec["uv"], co, corner, layer_index)
                    corner += 1
    if spec.get("materials"):
        # `materials=True` is two slots; an integer is that many slots, assigned round-robin per face.
        count = 2 if spec["materials"] is True else spec["materials"]
        for slot in range(count):
            me.materials.append(bpy.data.materials.new(f"Mat{slot}"))
        for poly in me.polygons:
            poly.material_index = poly.index % count
    seam = spec.get("seam")
    sharp = spec.get("sharp")
    if seam or sharp:
        for edge in me.edges:
            a, b = edge.vertices
            if seam and seam(me.vertices[a].co, me.vertices[b].co):
                edge.use_seam = True
            if sharp and sharp(me.vertices[a].co, me.vertices[b].co):
                edge.use_edge_sharp = True
    if spec.get("group"):
        name, fn = spec["group"]
        vg = obj.vertex_groups.new(name=name)
        for vertex in me.vertices:
            weight = fn(vertex.co)
            if weight is not None:
                vg.add([vertex.index], weight, "REPLACE")


def boolean_cube_object(name):
    # A size-2 cube minus a size-1 cube inside it (EXACT solver, DIFFERENCE), applied. The result is a
    # hollow cube made of two closed shells: 16 vertices, 24 edges, 12 quads (Blender 5.2.2 V16 E24 F12).
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
    cutter = bpy.context.active_object
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
    host = bpy.context.active_object
    host.name = name
    boolean = host.modifiers.new("Boolean", "BOOLEAN")
    boolean.operation = "DIFFERENCE"
    boolean.solver = "EXACT"
    boolean.operand_type = "OBJECT"
    boolean.object = cutter
    bpy.context.view_layer.objects.active = host
    bpy.ops.object.modifier_apply(modifier="Boolean")
    bpy.data.objects.remove(cutter)
    return host


def make_object(name, spec):
    if spec.get("boolean"):
        return boolean_cube_object(name)
    verts, faces = build_base(spec)
    # Vertices no face uses: Blender's BMesh keeps them through every decimate mode.
    verts = verts + [(5.0, 5.0, 5.0), (-6.0, 4.0, 2.0)] if spec.get("isolated") else verts
    me = bpy.data.meshes.new(name + "_mesh")
    edges = explicit_mesh(spec)["edges"] if spec["mesh"] == "explicit" else []
    me.from_pydata(verts, [tuple(e) for e in edges], faces)
    me.update()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    decorate(me, obj, spec)
    if spec.get("attrs"):
        idx = me.attributes.new("idx", "INT", "POINT")
        flag = me.attributes.new("flag", "BOOLEAN", "POINT")
        col = me.attributes.new("col", "FLOAT_COLOR", "POINT")
        for i in range(len(me.vertices)):
            idx.data[i].value = (7 * i) % 13
            flag.data[i].value = i % 3 == 0
            col.data[i].color = ((i % 5) / 4.0, (i % 7) / 6.0, (i % 11) / 10.0, 1.0)
        edge_idx = me.attributes.new("edge_idx", "INT", "EDGE")
        edge_flag = me.attributes.new("edge_flag", "BOOLEAN", "EDGE")
        edge_col = me.attributes.new("edge_col", "FLOAT_COLOR", "EDGE")
        for i in range(len(me.edges)):
            edge_idx.data[i].value = (7 * i) % 19
            edge_flag.data[i].value = i % 3 == 0
            edge_col.data[i].color = ((i % 5) / 4.0, (i % 7) / 6.0, (i % 11) / 10.0, 1.0)
    if spec.get("edge_float"):
        # One FLOAT EDGE attribute with exact binary fractions, so each value identifies its edge's input.
        edge_w = me.attributes.new("edge_w", "FLOAT", "EDGE")
        for i in range(len(me.edges)):
            edge_w.data[i].value = ((5 * i) % 16) / 8.0
    if spec.get("bytecol"):
        # Bytes are written exactly: `color_srgb` = byte / 255 quantises back to the same byte in Blender.
        bcol = me.attributes.new("bcol", "BYTE_COLOR", "POINT")
        for vertex in me.vertices:
            bcol.data[vertex.index].color_srgb = [c / 255.0 for c in spec["bytecol"](vertex.index, vertex.co)]
    return obj


# Generic custom attributes Blender interpolates during collapse.
POINT_ATTRIBUTE_TYPES = {"idx": "INT", "flag": "BOOLEAN", "col": "FLOAT_COLOR", "bcol": "BYTE_COLOR"}
EDGE_ATTRIBUTE_TYPES = {"edge_idx": "INT", "edge_flag": "BOOLEAN", "edge_col": "FLOAT_COLOR", "edge_w": "FLOAT"}


def generic_attributes(me, types):
    out = {}
    for name in types:
        attr = me.attributes.get(name)
        if attr is None:
            continue
        # FLOAT_COLOR and BYTE_COLOR are both dumped as linear RGBA floats (`item.color`), the form Potter's
        # blender_attributes uses. BYTE_COLOR bytes are exact: the bytecol spec writes them via `color_srgb`
        # = byte / 255, so the dumped `color` is the linear value of those exact bytes.
        if attr.data_type in ("FLOAT_COLOR", "BYTE_COLOR"):
            out[name] = [list(item.color) for item in attr.data]
        else:
            out[name] = [item.value for item in attr.data]
    return out


def point_attributes(me):
    return generic_attributes(me, POINT_ATTRIBUTE_TYPES)


def edge_attributes(me):
    return generic_attributes(me, EDGE_ATTRIBUTE_TYPES)


 # ---------------------------------------------------------------- dumps
def dump_input(obj, native_edge_flags):
    me = obj.data
    out = {
        "native_edge_flags": native_edge_flags,
        "vertices": [list(v.co) for v in me.vertices],
        "faces": [list(p.vertices) for p in me.polygons],
        "edges": [list(e.vertices) for e in me.edges],
        "edge_flags": [[e.use_seam, e.use_edge_sharp] for e in me.edges],
        "material_index": [p.material_index for p in me.polygons],
        "uv_layers": [],
        "vertex_groups": {},
    }
    for layer in me.uv_layers:
        values = [list(item.uv) for item in layer.data]
        out["uv_layers"].append({"name": layer.name, "values": values})
    for group in obj.vertex_groups:
        members = {}
        for vertex in me.vertices:
            for g in vertex.groups:
                if g.group == group.index:
                    members[str(vertex.index)] = g.weight
        out["vertex_groups"][group.name] = members
    out["point_attributes"] = point_attributes(me)
    out["edge_attributes"] = edge_attributes(me)
    return out


def dump_output(obj, dg):
    ob_eval = obj.evaluated_get(dg)
    me = ob_eval.to_mesh()
    names = [layer.name for layer in me.uv_layers]
    out = {
        "vertices": [list(v.co) for v in me.vertices],
        "faces": [list(p.vertices) for p in me.polygons],
        "edges": [list(e.vertices) for e in me.edges],
        "material_index": [p.material_index for p in me.polygons],
        "edge_flags": [[e.use_seam, e.use_edge_sharp] for e in me.edges],
        "uv_layers": {},
        "vertex_groups": {},
    }
    for name in names:
        layer = me.uv_layers[name]
        values = []
        for poly in me.polygons:
            for li in range(poly.loop_start, poly.loop_start + poly.loop_total):
                values.append(list(layer.data[li].uv))
        out["uv_layers"][name] = values
    group_names = [g.name for g in ob_eval.vertex_groups]
    for gi, gname in enumerate(group_names):
        weights = [0.0] * len(me.vertices)
        for vertex in me.vertices:
            for g in vertex.groups:
                if g.group == gi:
                    weights[vertex.index] = g.weight
        out["vertex_groups"][gname] = weights
    out["point_attributes"] = point_attributes(me)
    out["edge_attributes"] = edge_attributes(me)
    ob_eval.to_mesh_clear()
    return out


def set_modifier(obj, props):
    mod = obj.modifiers.new("Decimate", "DECIMATE")
    for key, value in props.items():
        if key == "delimit":
            mod.delimit = set(value)
        else:
            setattr(mod, key, value)
    return mod


# ---------------------------------------------------------------- cases
def center_line_x(a, b):
    # Edges whose both ends lie on the grid's middle column x == 0 (a seam or sharp line).
    return abs(a[0]) < 1e-6 and abs(b[0]) < 1e-6


def center_line_y(a, b):
    return abs(a[1]) < 1e-6 and abs(b[1]) < 1e-6


def cylinder_weight(co):
    return (co[2] + 1.0) * 0.5


def grid_ramp(co):
    return (co[0] + 1.0) * 0.5


def ratio_cases():
    base = {
        "sphere": dict(mesh="sphere"),
        "grid9": dict(mesh="grid", n=9, wave=True),
        "avatar": dict(mesh="cylinder", segments=16, rings=6, caps=True,
                       seam=lambda a, b: abs(a[1]) < 1e-6 and abs(b[1]) < 1e-6 and a[0] > 0 and b[0] > 0,
                       group=("Torso", cylinder_weight)),
    }
    cases = {}
    for key, spec in base.items():
        for r in range(1, 10):
            cases[f"{key}_r{r / 10:.1f}"] = (dict(spec), {"decimate_type": "COLLAPSE", "ratio": r / 10})
    return cases


def on_plane(axis, value):
    # Edges whose both ends lie on the plane `co[axis] == value`.
    return lambda a, b: abs(a[axis] - value) < 1e-6 and abs(b[axis] - value) < 1e-6


def meridian_seam(a, b):
    return abs(a[1]) < 1e-6 and abs(b[1]) < 1e-6 and a[0] > 0 and b[0] > 0


def upper_cap(a, b):
    return a[2] > 0.3 and b[2] > 0.3


def bcol_hash(index, co):
    # Distinct, well-spread RGBA bytes per vertex.
    return ((index * 37) % 256, (index * 91 + 13) % 256, (index * 53 + 101) % 256, (index * 61 + 40) % 256)


def bcol_midpoint(index, co):
    # Vertex 7 is black and vertex 12 white: the ends of the interior edge that collapses to its midpoint.
    if index == 7:
        return (0, 0, 0, 255)
    if index == 12:
        return (255, 255, 255, 255)
    return bcol_hash(index, co)


def all_cases():
    cases = ratio_cases()
    # COLLAPSE: cube, UV and continuous-UV grids, symmetry per axis, vertex groups, triangulation.
    cases["cube_r0.5"] = (dict(mesh="cube"), {"decimate_type": "COLLAPSE", "ratio": 0.5})
    # Boolean FLOAT stack FLOAT-1 with Potter's edge order: Blender's COLLAPSE rings depend on edge order.
    cases["float1_edge_potter_r0.6"] = (dict(mesh="explicit", file="inputs/float1_edge_potter_order.json"),
                                        {"decimate_type": "COLLAPSE", "ratio": 0.6})
    cases["shared_vertex_fans_nonmanifold_r0.35"] = (
        dict(mesh="shared_vertex_fans"),
        {"decimate_type": "COLLAPSE", "ratio": 0.35})
    cases["cube_r0.7"] = (dict(mesh="cube"), {"decimate_type": "COLLAPSE", "ratio": 0.7})
    # Boolean-applied hollow cube (two closed shells). Its inner shell triangulates and joins back into a
    # face that doubles one already made; Blender keeps the existing face (`bm_decim_triangulate_end`).
    cases["boolean_cube_r0.6"] = (dict(boolean=True), {"decimate_type": "COLLAPSE", "ratio": 0.6})
    cases["subcube2_r0.4"] = (dict(mesh="subdivided_cube", levels=2), {"decimate_type": "COLLAPSE", "ratio": 0.4})
    cases["cube_uv_r0.5"] = (dict(mesh="cube", uv="random", uv_layers=2),
                             {"decimate_type": "COLLAPSE", "ratio": 0.5})
    cases["grid3_cont_r0.2"] = (dict(mesh="grid", n=3, wave=True, uv="continuous"),
                                {"decimate_type": "COLLAPSE", "ratio": 0.2})
    cases["grid3_cont_r0.5"] = (dict(mesh="grid", n=3, wave=True, uv="continuous"),
                                {"decimate_type": "COLLAPSE", "ratio": 0.5})
    cases["grid9_cont_r0.5"] = (dict(mesh="grid", n=9, wave=True, uv="continuous", uv_layers=2),
                                {"decimate_type": "COLLAPSE", "ratio": 0.5})
    cases["sphere_uv_r0.5"] = (dict(mesh="sphere", uv="random", uv_layers=2),
                               {"decimate_type": "COLLAPSE", "ratio": 0.5})
    cases["sphere_uv_materials_r0.3"] = (dict(mesh="sphere", uv="random", materials=True),
                                         {"decimate_type": "COLLAPSE", "ratio": 0.3})
    for axis in ("X", "Y", "Z"):
        cases[f"sphere_sym{axis}_r0.5"] = (dict(mesh="sphere"),
                                           {"decimate_type": "COLLAPSE", "ratio": 0.5,
                                            "use_symmetry": True, "symmetry_axis": axis})
        cases[f"cube_sym{axis}_r0.5"] = (dict(mesh="cube"),
                                         {"decimate_type": "COLLAPSE", "ratio": 0.5,
                                          "use_symmetry": True, "symmetry_axis": axis})
    cases["grid9_symX_r0.5"] = (dict(mesh="grid", n=9, wave=True),
                                {"decimate_type": "COLLAPSE", "ratio": 0.5,
                                 "use_symmetry": True, "symmetry_axis": "X"})
    cases["grid9_flat_symX_r0.6"] = (dict(mesh="grid", n=9, wave=False),
                                     {"decimate_type": "COLLAPSE", "ratio": 0.6,
                                      "use_symmetry": True, "symmetry_axis": "X"})
    cases["avatar_symX_r0.5"] = (dict(mesh="cylinder", segments=16, rings=6),
                                 {"decimate_type": "COLLAPSE", "ratio": 0.5,
                                  "use_symmetry": True, "symmetry_axis": "X"})
    for name, factor, invert in (("vg_f1", 1.0, False), ("vg_f0.5", 0.5, False),
                                 ("vg_inv_f1", 1.0, True), ("vg_inv_f0.5", 0.5, True),
                                 ("vg_f0", 0.0, False)):
        cases[f"grid9_{name}_r0.5"] = (
            dict(mesh="grid", n=9, wave=True, group=("Ramp", grid_ramp)),
            {"decimate_type": "COLLAPSE", "ratio": 0.5, "vertex_group": "Ramp",
             "vertex_group_factor": factor, "invert_vertex_group": invert})
        cases[f"avatar_{name}_r0.4"] = (
            dict(mesh="cylinder", segments=16, rings=6, caps=True, group=("Torso", cylinder_weight)),
            {"decimate_type": "COLLAPSE", "ratio": 0.4, "vertex_group": "Torso",
             "vertex_group_factor": factor, "invert_vertex_group": invert})
    cases["sphere_tri_r0.5"] = (dict(mesh="sphere"),
                                {"decimate_type": "COLLAPSE", "ratio": 0.5, "use_collapse_triangulate": True})
    cases["grid9_wave_tri_r0.5"] = (dict(mesh="grid", n=9, wave=True),
                                    {"decimate_type": "COLLAPSE", "ratio": 0.5, "use_collapse_triangulate": True})
    cases["cube_tri_r0.5"] = (dict(mesh="cube"),
                              {"decimate_type": "COLLAPSE", "ratio": 0.5, "use_collapse_triangulate": True})
    cases["avatar_tri_r0.5"] = (dict(mesh="cylinder", segments=16, rings=6, caps=True),
                                {"decimate_type": "COLLAPSE", "ratio": 0.5, "use_collapse_triangulate": True})
    # UNSUBDIV on subdivided inputs, 1-3 iterations.
    for levels in (1, 2, 3):
        for it in (1, 2, 3):
            if it <= levels + 1:
                cases[f"subcube{levels}_unsubdiv{it}"] = (
                    dict(mesh="subdivided_cube", levels=levels),
                    {"decimate_type": "UNSUBDIV", "iterations": it})
    cases["grid9_unsubdiv1"] = (dict(mesh="grid", n=9, wave=True), {"decimate_type": "UNSUBDIV", "iterations": 1})
    cases["sphere_unsubdiv1"] = (dict(mesh="sphere"), {"decimate_type": "UNSUBDIV", "iterations": 1})
    # DISSOLVE: angle, every delimit, and boundaries.
    cases["grid9_dissolve_flat"] = (dict(mesh="grid", n=9, wave=False),
                                    {"decimate_type": "DISSOLVE", "angle_limit": 0.5})
    cases["grid9_dissolve_wave_a0.3"] = (dict(mesh="grid", n=9, wave=True),
                                         {"decimate_type": "DISSOLVE", "angle_limit": 0.3})
    cases["grid9_dissolve_wave_a1.0"] = (dict(mesh="grid", n=9, wave=True),
                                         {"decimate_type": "DISSOLVE", "angle_limit": 1.0})
    cases["sphere_dissolve_a0.5"] = (dict(mesh="sphere"), {"decimate_type": "DISSOLVE", "angle_limit": 0.5})
    cases["cube_dissolve_a3.14"] = (dict(mesh="cube"), {"decimate_type": "DISSOLVE", "angle_limit": 3.14})
    cases["grid9_dissolve_normal"] = (dict(mesh="grid", n=9, wave=True),
                                      {"decimate_type": "DISSOLVE", "angle_limit": 3.14, "delimit": ["NORMAL"]})
    cases["grid9_dissolve_normal_flat"] = (dict(mesh="grid", n=9, wave=False),
                                           {"decimate_type": "DISSOLVE", "angle_limit": 3.14, "delimit": ["NORMAL"]})
    cases["grid9_dissolve_material"] = (dict(mesh="grid", n=9, wave=True, materials=True),
                                        {"decimate_type": "DISSOLVE", "angle_limit": 3.14, "delimit": ["MATERIAL"]})
    cases["grid9_dissolve_seam"] = (dict(mesh="grid", n=9, wave=True, seam=center_line_x),
                                    {"decimate_type": "DISSOLVE", "angle_limit": 3.14, "delimit": ["SEAM"]})
    cases["grid9_dissolve_sharp"] = (dict(mesh="grid", n=9, wave=True, sharp=center_line_y),
                                     {"decimate_type": "DISSOLVE", "angle_limit": 3.14, "delimit": ["SHARP"]})
    cases["grid9_dissolve_uv"] = (dict(mesh="grid", n=9, wave=True, uv="offset"),
                                  {"decimate_type": "DISSOLVE", "angle_limit": 3.14, "delimit": ["UV"]})
    cases["grid9_dissolve_boundaries"] = (dict(mesh="grid", n=9, wave=True),
                                          {"decimate_type": "DISSOLVE", "angle_limit": 0.5,
                                           "use_dissolve_boundaries": True})
    cases["grid9_dissolve_seam_boundaries"] = (
        dict(mesh="grid", n=9, wave=True, seam=center_line_x),
        {"decimate_type": "DISSOLVE", "angle_limit": 3.14, "delimit": ["SEAM"], "use_dissolve_boundaries": True})
    cases["avatar_dissolve_seam"] = (
        dict(mesh="cylinder", segments=16, rings=6, caps=True,
             seam=lambda a, b: abs(a[1]) < 1e-6 and abs(b[1]) < 1e-6 and a[0] > 0 and b[0] > 0),
        {"decimate_type": "DISSOLVE", "angle_limit": 3.14, "delimit": ["SEAM"]})
    # Isolated (loose, unused) vertices: Blender keeps them in every mode.
    cases["iso_cube_r0.5"] = (dict(mesh="cube", isolated=True), {"decimate_type": "COLLAPSE", "ratio": 0.5})
    cases["iso_grid9_r0.5"] = (dict(mesh="grid", n=9, wave=True, isolated=True),
                               {"decimate_type": "COLLAPSE", "ratio": 0.5})
    cases["iso_sphere_sym_r0.5"] = (dict(mesh="sphere", isolated=True),
                                    {"decimate_type": "COLLAPSE", "ratio": 0.5,
                                     "use_symmetry": True, "symmetry_axis": "X"})
    cases["iso_subcube2_unsubdiv1"] = (dict(mesh="subdivided_cube", levels=2, isolated=True),
                                       {"decimate_type": "UNSUBDIV", "iterations": 1})
    cases["iso_grid9_dissolve_flat"] = (dict(mesh="grid", n=9, wave=False, isolated=True),
                                        {"decimate_type": "DISSOLVE", "angle_limit": 0.5})
    cases["iso_cube_dissolve"] = (dict(mesh="cube", isolated=True),
                                  {"decimate_type": "DISSOLVE", "angle_limit": 0.5})
    # Point and edge attributes blended by collapse: one level (grid) and many levels (sphere, 0.2).
    cases["attr_grid9_r0.5"] = (dict(mesh="grid", n=9, wave=True, attrs=True),
                                {"decimate_type": "COLLAPSE", "ratio": 0.5})
    cases["attr_sphere_r0.2"] = (dict(mesh="sphere", attrs=True),
                                 {"decimate_type": "COLLAPSE", "ratio": 0.2})
    # BYTE_COLOR POINT. Blender blends bytes by a rounded weighted sum (`layerInterp_mloopcol`), so each merge
    # re-rounds. The midpoint case merges a black and a white vertex on an interior edge (quadric singular,
    # so the merge is exactly at the midpoint). The chain case ratio 7/32 reaches a vertex at x = -1/16 that
    # came from at least three successive midpoint merges.
    cases["bcol_midpoint"] = (dict(mesh="grid", n=5, wave=False, bytecol=bcol_midpoint),
                              {"decimate_type": "COLLAPSE", "ratio": 0.9})
    cases["bcol_chain"] = (dict(mesh="grid", n=5, wave=False, bytecol=bcol_hash),
                           {"decimate_type": "COLLAPSE", "ratio": 0.21875})
    cases["attr_bcol_sphere_r0.2"] = (dict(mesh="sphere", attrs=True, bytecol=bcol_hash),
                                      {"decimate_type": "COLLAPSE", "ratio": 0.2})
    # Native sharp/seam (the Potter input carries `sharp`/`seam` attributes, not blender_edge_flags) and
    # multi-slot materials.
    cases["sphere_flags_native_r0.4"] = (
        dict(mesh="sphere", seam=meridian_seam, sharp=upper_cap, materials=3, native_edge_flags=True),
        {"decimate_type": "COLLAPSE", "ratio": 0.4})
    cases["cube_flags_native_r0.5"] = (
        dict(mesh="cube", seam=on_plane(0, 1.0), sharp=on_plane(2, 1.0), materials=2, native_edge_flags=True),
        {"decimate_type": "COLLAPSE", "ratio": 0.5})
    cases["grid9_flags_r0.5"] = (dict(mesh="grid", n=9, wave=True, seam=center_line_x, sharp=center_line_y),
                                 {"decimate_type": "COLLAPSE", "ratio": 0.5})
    cases["sphere_materials3_r0.4"] = (dict(mesh="sphere", materials=3),
                                       {"decimate_type": "COLLAPSE", "ratio": 0.4})
    # Native DISSOLVE cases use a 0.3 angle limit: at 3.14 the whole grid merges into one edge and no flag is checked.
    cases["grid9_dissolve_sharp_native"] = (
        dict(mesh="grid", n=9, wave=True, sharp=center_line_y, native_edge_flags=True),
        {"decimate_type": "DISSOLVE", "angle_limit": 0.3, "delimit": ["SHARP"]})
    cases["grid9_dissolve_seam_native"] = (
        dict(mesh="grid", n=9, wave=True, seam=center_line_x, native_edge_flags=True),
        {"decimate_type": "DISSOLVE", "angle_limit": 0.3, "delimit": ["SEAM"]})
    cases["grid9_dissolve_sharp_seam_native"] = (
        dict(mesh="grid", n=9, wave=True, seam=center_line_x, sharp=center_line_y, native_edge_flags=True),
        {"decimate_type": "DISSOLVE", "angle_limit": 3.14, "delimit": ["SHARP", "SEAM"]})
    cases["grid9_dissolve_material3"] = (dict(mesh="grid", n=9, wave=True, materials=3),
                                         {"decimate_type": "DISSOLVE", "angle_limit": 3.14, "delimit": ["MATERIAL"]})
    # EDGE FLOAT values and seams through UNSUBDIV and DISSOLVE: each output edge keeps its own value.
    cases["grid9_unsubdiv1_seam_edgew"] = (
        dict(mesh="grid", n=9, wave=True, seam=center_line_x, edge_float=True),
        {"decimate_type": "UNSUBDIV", "iterations": 1})
    cases["grid9_dissolve_edgew_a0.3"] = (
        dict(mesh="grid", n=9, wave=True, edge_float=True),
        {"decimate_type": "DISSOLVE", "angle_limit": 0.3})
    return cases


def describe(spec):
    # Callables (seam and group predicates) are recorded by name only.
    out = {}
    for key, value in spec.items():
        if callable(value):
            out[key] = True
        elif isinstance(value, tuple):
            out[key] = value[0]
        else:
            out[key] = value
    return out


def run_case(name, spec, props):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    obj = make_object("Subject", spec)
    input_doc = dump_input(obj, spec.get("native_edge_flags", False))
    set_modifier(obj, props)
    dg = bpy.context.evaluated_depsgraph_get()
    output = dump_output(obj, dg)
    doc = {
        "case": name,
        "blender_version": BLENDER_VERSION,
        "spec": {"mesh": describe(spec), "modifier": props},
        "input": input_doc,
        "blender": output,
    }
    with open(os.path.join(OUT, name + ".json"), "w") as handle:
        json.dump(doc, handle, separators=(",", ":"))
    print(f"{name}: {len(input_doc['faces'])} -> {len(output['faces'])} faces")


cases = all_cases()
for case_name, (case_spec, case_props) in cases.items():
    if ONLY is not None and ONLY not in case_name:
        continue
    run_case(case_name, case_spec, case_props)
