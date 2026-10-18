# Usage: /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --python tests/fixtures/pr18b_mirror/generate.py -- <scratch-dir>
"""PR18B mirror oracle: Blender 5.2.2 LTS ground truth for the Mirror modifier.

Writes <FIXTURES>/<variant>.json for every variant (schema from the PR18B brief). A variant that
Blender refuses gets {"variant", "blender_version", "stack", "error"} instead. Debug dumps and
summary.txt go to <scratch-dir>.

File names are the descriptive variant names, i.e. the text after the brief's code
(M1 cube2_x -> cube2_x.json). The code of each variant is printed in summary.txt.

Output is object-local and read from the evaluated depsgraph via to_mesh(). Stack params are the
effective RNA values read back after assignment, for the keys in MIRROR_KEYS.

Custom-normal modes:
  region: deterministic pr4_shading values on listed face/corners; other corners keep Blender's computed normal.
  vertex_custom: `normals_split_custom_set_from_vertices` with deterministic per-vertex offsets from the probe.
  corner_custom: `normals_split_custom_set` with deterministic per-loop offsets from the probe.

Bases (object-local):
  cube2           unit cube corners mapped to +-1; face_smooth [T,F,T,T,F,F]; sharp [1,2],[4,7]
  cube2_mat       cube2 with face_material_index [0,1,2,0,1,2]
  cube2_custom    cube2 with dense custom normals on faces {2,3}
  cube_offcenter  cube2 + (0.3,-0.2,0.1) + 0.05 * jitter (pr4 jitter formula); smooth/sharp as cube2
  cube_jitter     cube2 + 0.15 * jitter; all faces smooth, no sharp edges (pr4 base_cube_jitter)
  half_box        box x in [0,2], y,z in [-1,1]; all faces flat
  edges_half      half_box + sharp [0,1] (x-direction edge spanning the plane x=0) + seam on the x=2 face
  plane_x0        single quad in plane x=0
  thr_grid        5x3 vertices in XY: x columns [-1,-0.0006,0,0.0004,1], y rows [-1,0,1], 8 quads, flat
  uv_cube         unit cube 0..1 (pr3 BOX layout), flat, UVMap k/1024 per corner (pr3 uv_random),
                  face_material_index [0,1,2,0,1,2]
  groups_cube     cube2 with vertex groups Arm.L, Arm.R (x>0), Center (x>=0), Body (all); weights 0.05..0.90
  half_y_grid    smooth 3x3 half grid with per-vertex custom normals, UVs and vertex groups
  rotated_grid   smooth 4x4 grid with each quad starting at its second vertex, UVs and vertex groups
  lshape          concave L polygon in XY at z=0, one flat face
  random_a-d      5x3 hash-jittered quad grid over [-1,1]^2 (random_b groups, random_c UVs, random_d both)
  concave_comb    8-corner comb in XY at z=0, teeth cross x=0; concave_comb_y transposed; concave_comb_corner custom normals
"""
import json
import math
import os
import sys
import traceback

import bpy
from mathutils import Vector

OUT = sys.argv[sys.argv.index("--") + 1]
FIXTURES = os.path.dirname(os.path.abspath(__file__))
BLENDER_VERSION = "5.2.2 LTS"
assert bpy.app.version == (5, 2, 2), f"need Blender 5.2.2, got {bpy.app.version}"
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIXTURES, exist_ok=True)

T, F = True, False


def vec(v):
    return [float(v[0]), float(v[1]), float(v[2])]


def normalize(v):
    length = math.sqrt(sum(c * c for c in v))
    return [c / length for c in v]


# ---------------------------------------------------------------- base meshes

CUBE_C = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
CUBE_F = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (3, 7, 6, 2), (0, 4, 7, 3), (1, 2, 6, 5)]
CUBE_SMOOTH = [T, F, T, T, F, F]
CUBE_SHARP = [[1, 2], [4, 7]]
HALF_SEAM = [[1, 2], [2, 6], [5, 6], [1, 5]]  # the four edges of face 5 (x=2 side)
THR_X = [-1.0, -0.0006, 0.0, 0.0004, 1.0]
THR_Y = [-1.0, 0.0, 1.0]


def centered(c):
    return (2.0 * c[0] - 1.0, 2.0 * c[1] - 1.0, 2.0 * c[2] - 1.0)


def jitter(i, amp):
    return (amp * math.sin(1.3 * i + 0.4), amp * math.cos(0.9 * i + 1.1), amp * math.sin(1.7 * i + 2.0))


def shifted(i, shift, amp):
    base, j = centered(CUBE_C[i]), jitter(i, amp)
    return tuple(base[k] + shift[k] + j[k] for k in range(3))


def cube2_verts():
    return [centered(c) for c in CUBE_C]


def half_box_verts():
    return [(2.0 * c[0], 2.0 * c[1] - 1.0, 2.0 * c[2] - 1.0) for c in CUBE_C]


def thr_grid():
    verts = [(THR_X[i], THR_Y[j], 0.0) for j in range(3) for i in range(5)]
    faces = []
    for j in range(2):
        for i in range(4):
            a = j * 5 + i
            faces.append((a, a + 1, a + 6, a + 5))
    return mesh_spec(verts, faces, [F] * len(faces))


def half_y_grid():
    verts = [(x, y, 0.4 * x * x + 0.25 * y * y + 0.1 * x * y)
             for y in (0.0, 0.5, 1.0) for x in (-1.0, 0.0, 1.0)]
    faces = []
    for j in range(2):
        for i in range(2):
            a = j * 3 + i
            faces.append((a, a + 1, a + 4, a + 3))
    return mesh_spec(verts, faces, [T] * len(faces), vertex_custom=True, uv=True, groups=True)


def rotated_grid():
    xs, ys = (-0.6, -0.2, 0.3, 0.8), (-1.0, -0.3, 0.4, 1.0)
    verts = [(x, y, 0.4 * x * x + 0.25 * y * y + 0.1 * x * y) for y in ys for x in xs]
    faces = []
    for j in range(3):
        for i in range(3):
            a = j * 4 + i
            faces.append((a + 1, a + 5, a + 4, a))
    return mesh_spec(verts, faces, [T] * len(faces), region=list(range(len(faces))), uv=True, groups=True)
def rotated_grid_vertex_custom():
    spec = rotated_grid()
    spec["region"] = None
    spec["vertex_custom"] = True
    return spec


def rotated_grid_corner_custom():
    spec = rotated_grid()
    spec["region"] = None
    spec["corner_custom"] = True
    return spec


def hash01(i, salt):
    """Deterministic integer hash mapped to [0, 1); keeps the generator free of the random module."""
    x = (i * 2654435761 + salt * 40503 + 0x9E3779B9) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x2C1B3C6D) & 0xFFFFFFFF
    x ^= x >> 12
    x = (x * 0x297A2D39) & 0xFFFFFFFF
    x ^= x >> 15
    return x / 4294967296.0


def random_grid(seed, groups=False, uv=False):
    """5x3 quad grid over [-1,1]^2 with hash jitter, quads only. The centre column (x=0) is exact or within
    0.0004 of the mirror plane; each quad gets a hash-chosen start corner, winding and smooth/flat flag."""
    cols, rows = 5, 3
    verts = []
    for r in range(rows):
        for c in range(cols):
            i = seed * 1000 + r * cols + c
            x = -1.0 + 2.0 * c / (cols - 1)
            y = -1.0 + 2.0 * r / (rows - 1)
            if c == cols // 2:
                x = 0.0 if hash01(i, 3) < 0.5 else 0.0004 * (2.0 * hash01(i, 7) - 1.0)
            else:
                x += 0.12 * (2.0 * hash01(i, 1) - 1.0)
            y += 0.12 * (2.0 * hash01(i, 2) - 1.0)
            verts.append((x, y, 0.0))
    faces, smooth = [], []
    for r in range(rows - 1):
        for c in range(cols - 1):
            a = r * cols + c
            i = seed * 1000 + 100 + r * (cols - 1) + c
            quad = [a, a + 1, a + cols + 1, a + cols]
            k = int(hash01(i, 4) * 4)
            quad = quad[k:] + quad[:k]
            if hash01(i, 5) < 0.4:
                quad = quad[::-1]
            faces.append(quad)
            smooth.append(hash01(i, 6) < 0.5)
    return mesh_spec(verts, faces, smooth, uv=uv, groups=groups)


COMB = [(2, -2), (2, 2), (-2, 2), (-2, 0.4), (1, 0.4), (1, -0.4), (-2, -0.4), (-2, -2)]


def comb_spec(transpose=False, corner=False):
    """8-corner comb: spine x in [1,2]; teeth y in [0.4,2] and [-2,-0.4] reach x=-2; the notch between them
    ends at x=1. Bisect on X splits it into two tooth pieces; transpose gives the vertical-tooth version."""
    pts = [(y, x) if transpose else (x, y) for x, y in COMB]
    return mesh_spec([(x, y, 0.0) for x, y in pts], [tuple(range(len(pts)))], [F], corner_custom=corner)

def disconnected_comb_spec():
    verts = [(x, y + offset, 0.0) for offset in (-5.0, 5.0) for x, y in COMB]
    faces = [tuple(range(8)), tuple(range(8, 16))]
    edges = [(8 + i, 8 + (i + 1) % 8) for i in range(8)]
    edges.extend((i, (i + 1) % 8) for i in range(8))
    return mesh_spec(verts, faces, [F, F], edges=edges)


def mesh_spec(verts, faces, smooth, sharp=(), seam=(), material=None, region=None, uv=False, groups=False,
              edges=(), vertex_custom=False, corner_custom=False):
    return dict(vertices=[vec(v) for v in verts], faces=[list(f) for f in faces], edges=[list(e) for e in edges],
                face_smooth=[bool(s) for s in smooth], sharp_edges=[list(e) for e in sharp],
                seam_edges=[list(e) for e in seam],
                material=[0] * len(faces) if material is None else list(material),
                region=region, uv=uv, groups=groups, vertex_custom=vertex_custom, corner_custom=corner_custom)


BASES = {
    "cube2": lambda: mesh_spec(cube2_verts(), CUBE_F, CUBE_SMOOTH, CUBE_SHARP),
    "cube2_mat": lambda: mesh_spec(cube2_verts(), CUBE_F, CUBE_SMOOTH, CUBE_SHARP, material=[0, 1, 2, 0, 1, 2]),
    "cube2_custom": lambda: mesh_spec(cube2_verts(), CUBE_F, CUBE_SMOOTH, CUBE_SHARP, region=[2, 3]),
    "cube_offcenter": lambda: mesh_spec([shifted(i, (0.3, -0.2, 0.1), 0.05) for i in range(8)], CUBE_F,
                                        CUBE_SMOOTH, CUBE_SHARP),
    "cube_jitter": lambda: mesh_spec([shifted(i, (0.0, 0.0, 0.0), 0.15) for i in range(8)], CUBE_F, [T] * 6),
    "half_box": lambda: mesh_spec(half_box_verts(), CUBE_F, [F] * 6),
    "edges_half": lambda: mesh_spec(half_box_verts(), CUBE_F, [F] * 6, sharp=[[0, 1]], seam=HALF_SEAM),
    "plane_x0": lambda: mesh_spec([(0, -1, -1), (0, 1, -1), (0, 1, 1), (0, -1, 1)], [(0, 1, 2, 3)], [F]),
    "duplicate_opposite_triangles": lambda: mesh_spec(
        [(0, 0, 0), (1, 0, 0), (1, 1, 0)], [(0, 1, 2), (2, 1, 0)], [F, F]),
    "thr_grid": thr_grid,
    "half_y_grid": half_y_grid,
    "rotated_grid": rotated_grid,
    "rotated_grid_vertex_custom": rotated_grid_vertex_custom,
    "rotated_grid_corner_custom": rotated_grid_corner_custom,
    "uv_cube": lambda: mesh_spec([tuple(float(x) for x in c) for c in CUBE_C], CUBE_F, [F] * 6,
                                 material=[0, 1, 2, 0, 1, 2], uv=True),
    "groups_cube": lambda: mesh_spec(cube2_verts(), CUBE_F, CUBE_SMOOTH, CUBE_SHARP, groups=True),
    "lshape": lambda: mesh_spec([(-1, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 2, 0), (-1, 2, 0)],
                                [(0, 1, 2, 3, 4, 5)], [F]),
    "random_a": lambda: random_grid(1),
    "random_b": lambda: random_grid(2, groups=True),
    "random_c": lambda: random_grid(3, uv=True),
    "random_d": lambda: random_grid(4, groups=True, uv=True),
    "concave_comb": lambda: comb_spec(),
    "concave_comb_y": lambda: comb_spec(transpose=True),
    "concave_comb_corner": lambda: comb_spec(corner=True),
    "disconnected_comb_order": disconnected_comb_spec,
}


# ---------------------------------------------------------------- mesh construction (helpers copied from pr4/pr3)

def newell(verts, face):
    nx = ny = nz = 0.0
    for k in range(len(face)):
        p = verts[face[k]]
        q = verts[face[(k + 1) % len(face)]]
        nx += (p[1] - q[1]) * (p[2] + q[2])
        ny += (p[2] - q[2]) * (p[0] + q[0])
        nz += (p[0] - q[0]) * (p[1] + q[1])
    return normalize([nx, ny, nz])


def synthetic_normal(N, f, c):
    t = [math.sin(1.7 * f + 0.9 * c), math.cos(1.3 * f + 0.6 * c), math.sin(0.7 * f + 1.1 * c)]
    d = sum(t[k] * N[k] for k in range(3))
    tp = [t[k] - d * N[k] for k in range(3)]
    tp_len = math.sqrt(sum(x * x for x in tp))
    assert tp_len > 0.2, (f, c, tp_len)
    tp = [x / tp_len for x in tp]
    return normalize([N[k] + 0.8 * tp[k] for k in range(3)])


def edge_key(a, b):
    return (min(a, b), max(a, b))


# pr3 UV values: k/1024 so they are exact in float32 and float64 alike.
def uv_random(loop, layer):
    a = 64 + (loop * 37 + layer * 11) % 800
    b = 64 + (loop * 53 + layer * 29 + 7) % 800
    return [a / 1024.0, b / 1024.0]


GROUP_DEFS = [
    ("Arm.L", lambda x: x > 0.0),
    ("Arm.R", lambda x: x > 0.0),
    ("Center", lambda x: x >= 0.0),
    ("Body", lambda x: True),
]


def group_weight(gidx, i):
    return 0.05 + ((i * 37 + gidx * 101 + 13) % 86) / 100.0


def make_object(name, spec):
    """Build a subject object from a base spec. Returns (object, input doc, debug info)."""
    verts, faces = spec["vertices"], spec["faces"]
    me = bpy.data.meshes.new(name + "_mesh")
    me.from_pydata([tuple(v) for v in verts], [tuple(e) for e in spec["edges"]], [tuple(f) for f in faces])
    me.update()
    me.polygons.foreach_set("use_smooth", [bool(x) for x in spec["face_smooth"]])
    me.polygons.foreach_set("material_index", spec["material"])
    emap = {edge_key(e.vertices[0], e.vertices[1]): e.index for e in me.edges}
    sharp = [False] * len(me.edges)
    seam = [False] * len(me.edges)
    for a, b in spec["sharp_edges"]:
        assert edge_key(a, b) in emap, f"{name}: sharp pair {a}-{b} is not an edge"
        sharp[emap[edge_key(a, b)]] = True
    for a, b in spec["seam_edges"]:
        assert edge_key(a, b) in emap, f"{name}: seam pair {a}-{b} is not an edge"
        seam[emap[edge_key(a, b)]] = True
    me.edges.foreach_set("use_edge_sharp", sharp)
    me.edges.foreach_set("use_seam", seam)
    me.update()

    info = {}
    uv_doc = None
    if spec["uv"]:
        layer = me.uv_layers.new(name="UVMap")
        uv_doc = []
        for poly in me.polygons:
            row = []
            for li in range(poly.loop_start, poly.loop_start + poly.loop_total):
                uv = uv_random(li, 0)
                layer.data[li].uv = uv
                row.append([float(uv[0]), float(uv[1])])
            uv_doc.append(row)
        me.update()

    custom_doc = None
    if spec["region"] is not None or spec["vertex_custom"] or spec["corner_custom"]:
        computed = [Vector(n.vector) for n in me.corner_normals]
        custom_list = []
        if spec["vertex_custom"]:
            vertex_normals = []
            for i, vertex in enumerate(me.vertices):
                offset = Vector((0.45 + 0.1 * math.sin(i), 0.3 * math.cos(1.7 * i), 0.0))
                vertex_normals.append((Vector(vertex.normal) + offset).normalized())
            custom_list = [
                [vec(vertex_normals[me.loops[li].vertex_index]) for li in range(poly.loop_start, poly.loop_start + poly.loop_total)]
                for poly in me.polygons
            ]
            me.normals_split_custom_set_from_vertices(vertex_normals)
        elif spec["corner_custom"]:
            flat = []
            for li, normal in enumerate(computed):
                offset = Vector((0.35 + 0.1 * math.sin(li), 0.4 * math.cos(1.7 * li), 0.0))
                flat.append(tuple((normal + offset).normalized()))
            custom_list = [
                [vec(normal) for normal in flat[poly.loop_start:poly.loop_start + poly.loop_total]]
                for poly in me.polygons
            ]
            me.normals_split_custom_set(flat)
        else:
            region = spec["region"]
            for fi, poly in enumerate(me.polygons):
                N = newell(verts, faces[fi])
                custom_list.append([
                    synthetic_normal(N, fi, corner) if fi in region else vec(computed[poly.loop_start + corner])
                    for corner in range(poly.loop_total)
                ])
            flat = [tuple(n) for row in custom_list for n in row]
            me.normals_split_custom_set(flat)
        me.update()
        stored = [vec(n.vector) for n in me.corner_normals]
        requested = [x for row in custom_list for n in row for x in n]
        stored_flat = [x for n in stored for x in n]
        err = max(abs(a - b) for a, b in zip(stored_flat, requested))
        info["custom_quantization_max_abs_err"] = err
        if spec["region"] is not None:
            assert err < 1e-4, f"{name}: stored custom normals differ from requested by {err}"
        custom_doc = [stored[p.loop_start:p.loop_start + p.loop_total] for p in me.polygons]
        assert [bool(p.use_smooth) for p in me.polygons] == [bool(x) for x in spec["face_smooth"]], name
        after_sharp = [bool(e.use_edge_sharp) for e in me.edges]
        info["auto_sharp_edges_added"] = [sorted(me.edges[i].vertices) for i in range(len(sharp))
                                          if after_sharp[i] and not sharp[i]]
        assert all(after_sharp[i] for i in range(len(sharp)) if sharp[i]), f"{name}: custom set cleared sharp"

    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)

    groups_doc = {}
    if spec["groups"]:
        for gidx, (gname, member) in enumerate(GROUP_DEFS):
            vg = ob.vertex_groups.new(name=gname)
            weights = {}
            for i, v in enumerate(verts):
                if member(v[0]):
                    w = group_weight(gidx, i)
                    vg.add([i], w, "REPLACE")
                    weights[str(i)] = w
            groups_doc[gname] = weights

    doc = dict(
        vertices=verts,
        faces=faces,
        edges=spec["edges"],
        face_smooth=[bool(p.use_smooth) for p in me.polygons],
        sharp_edges=[sorted(e.vertices) for e in me.edges if e.use_edge_sharp],
        seam_edges=[sorted(e.vertices) for e in me.edges if e.use_seam],
        face_material_index=[int(p.material_index) for p in me.polygons],
        vertex_groups=groups_doc,
    )
    if custom_doc is not None:
        doc["custom_normals"] = custom_doc
    if uv_doc is not None:
        doc["uv_layers"] = {"UVMap": uv_doc}
    return ob, doc, info


# ---------------------------------------------------------------- evaluation

def evaluate(ob, frame):
    bpy.context.scene.frame_set(frame)
    dg = bpy.context.evaluated_depsgraph_get()
    ob_eval = ob.evaluated_get(dg)
    me = ob_eval.to_mesh()
    try:
        verts = [vec(v.co) for v in me.vertices]
        group_names = [g.name for g in ob_eval.vertex_groups]
        faces = []
        for poly in me.polygons:
            loops = list(range(poly.loop_start, poly.loop_start + poly.loop_total))
            faces.append(dict(
                corners=[verts[me.loops[li].vertex_index] for li in loops],
                normals=[vec(me.corner_normals[li].vector) for li in loops],
                smooth=bool(poly.use_smooth),
                material_index=int(poly.material_index),
                uv={layer.name: [[float(layer.data[li].uv[0]), float(layer.data[li].uv[1])] for li in loops]
                    for layer in me.uv_layers},
            ))
        edges = [dict(verts=[verts[e.vertices[0]], verts[e.vertices[1]]], sharp=bool(e.use_edge_sharp),
                      seam=bool(e.use_seam)) for e in me.edges]
        vgroups = []
        for i, v in enumerate(me.vertices):
            if not v.groups:
                continue
            weights = {}
            for el in v.groups:
                weights[group_names[el.group]] = float(el.weight)
            vgroups.append(dict(position=verts[i], weights=weights))
        out = dict(vertices=verts, faces=faces, edges=edges, vertex_groups=vgroups)
        debug = dict(vertices=len(me.vertices), edges=len(me.edges), faces=len(me.polygons),
                     corners=len(me.loops), has_custom_normals=bool(me.has_custom_normals))
        return out, debug
    finally:
        ob_eval.to_mesh_clear()


# ---------------------------------------------------------------- RNA helpers

MIRROR_KEYS = [
    "use_axis", "use_bisect_axis", "use_bisect_flip_axis", "use_clip", "use_mirror_merge", "merge_threshold",
    "bisect_threshold", "use_mirror_u", "use_mirror_v", "use_mirror_udim", "mirror_offset_u", "mirror_offset_v",
    "offset_u", "offset_v", "use_mirror_vertex_groups", "mirror_object",
]


def rna_json(value):
    if isinstance(value, bpy.types.ID):
        return value.name
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "__len__"):
        return [rna_json(x) for x in value]
    return str(value)


def mirror_readback(mod):
    out = {}
    for k in MIRROR_KEYS:
        if k == "mirror_object":
            out[k] = mod.mirror_object.name if mod.mirror_object else None
        else:
            out[k] = rna_json(getattr(mod, k))
    return out


def reset():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for me in list(bpy.data.meshes):
        bpy.data.meshes.remove(me)


# ---------------------------------------------------------------- variants

SUBJ = dict(location=[0.0, 0.0, 0.0], rotation_euler=[0.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0])


def SUBJECT(loc=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1)):
    return dict(location=[float(x) for x in loc], rotation_euler=[float(x) for x in rot],
                scale=[float(x) for x in scale])


def EMPTY(loc=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1)):
    return dict(type="EMPTY", **SUBJECT(loc, rot, scale))


def MIRROR(axis, bisect=None, flip=None, **extra):
    p = dict(use_axis=list(axis))
    if bisect is not None:
        p["use_bisect_axis"] = list(bisect)
    if flip is not None:
        p["use_bisect_flip_axis"] = list(flip)
    p.update(extra)
    return ("mirror", p)


def WAVE(**params):
    return ("wave", params)


def V(code, name, base, stack, objects=None, subject=None):
    return dict(code=code, name=name, base=base, stack=stack, objects=objects or {}, subject=subject or SUBJ)


MIRROR_EMPTY = "MirrorEmpty"

VARIANTS = [
    # merge
    V("M1", "cube2_x", "cube2", [MIRROR([T, F, F])]),
    V("M2", "cube2_xy", "cube2", [MIRROR([T, T, F])]),
    V("M3", "cube2_xz", "cube2", [MIRROR([T, F, T])]),
    V("M4", "cube2_yz", "cube2", [MIRROR([F, T, T])]),
    V("M5", "cube2_xyz", "cube2", [MIRROR([T, T, T])]),
    V("M6", "cube2_xyz_nomerge", "cube2", [MIRROR([T, T, T], use_mirror_merge=False)]),
    V("M7", "cube_offcenter_xyz", "cube_offcenter", [MIRROR([T, T, T])]),
    V("M8", "cube_jitter_xyz", "cube_jitter", [MIRROR([T, T, T])]),
    V("M9", "cube_jitter_x", "cube_jitter", [MIRROR([T, F, F])]),
    V("M10", "half_box_x", "half_box", [MIRROR([T, F, F])]),
    V("M11", "half_box_xyz", "half_box", [MIRROR([T, T, T])]),
    V("M12", "plane_x0_x", "plane_x0", [MIRROR([T, F, F])]),
    V("M13", "thr_grid_x_t1", "thr_grid", [MIRROR([T, F, F], merge_threshold=0.001)]),
    V("M14", "thr_grid_x_t05", "thr_grid", [MIRROR([T, F, F], merge_threshold=0.0005)]),
    V("M15", "thr_grid_x_t0", "thr_grid", [MIRROR([T, F, F], merge_threshold=0.0)]),
    V("M16", "thr_grid_x_t2e3", "thr_grid", [MIRROR([T, F, F], merge_threshold=0.002)]),
    V("M17", "half_box_x_t0", "half_box", [MIRROR([T, F, F], merge_threshold=0.0)]),
    V("M18", "weld_context_duplicate_opposite_triangles_x", "duplicate_opposite_triangles",
      [MIRROR([T, F, F])]),
    V("TH1", "cube2_xyz_t0", "cube2", [MIRROR([T, T, T], merge_threshold=0.0)]),
    # bisect / flip
    V("B1", "half_box_bisect_x", "half_box", [MIRROR([T, F, F], bisect=[T, F, F])]),
    V("B2", "half_box_bisect_x_flip", "half_box", [MIRROR([T, F, F], bisect=[T, F, F], flip=[T, F, F])]),
    V("B3", "cube2_bisect_x", "cube2", [MIRROR([T, F, F], bisect=[T, F, F])]),
    V("B4", "cube2_bisect_x_flip", "cube2", [MIRROR([T, F, F], bisect=[T, F, F], flip=[T, F, F])]),
    V("B5", "cube2_bisect_y_unmirrored", "cube2", [MIRROR([T, F, F], bisect=[F, T, F])]),
    V("B6", "cube2_flip_only", "cube2", [MIRROR([T, F, F], flip=[T, F, F])]),
    V("B7", "cube2_bisect_xy", "cube2", [MIRROR([T, T, F], bisect=[T, T, F])]),
    V("B8", "cube2_bisect_xz_flip_z", "cube2", [MIRROR([T, F, T], bisect=[T, F, T], flip=[F, F, T])]),
    V("B9", "cube_offcenter_bisect_x", "cube_offcenter", [MIRROR([T, F, F], bisect=[T, F, F])]),
    V("B10", "cube_jitter_bisect_x", "cube_jitter", [MIRROR([T, F, F], bisect=[T, F, F])]),
    V("B11", "thr_grid_bisect_t", "thr_grid", [MIRROR([T, F, F], bisect=[T, F, F], bisect_threshold=0.001)]),
    V("B12", "thr_grid_bisect_t0", "thr_grid", [MIRROR([T, F, F], bisect=[T, F, F], bisect_threshold=0.0)]),
    V("B13", "thr_grid_bisect_t05", "thr_grid", [MIRROR([T, F, F], bisect=[T, F, F], bisect_threshold=0.0006)]),
    V("B14", "lshape_bisect_x", "lshape", [MIRROR([T, F, F], bisect=[T, F, F])]),
    V("B15", "lshape_bisect_x_flip", "lshape", [MIRROR([T, F, F], bisect=[T, F, F], flip=[T, F, F])]),
    V("B16", "half_box_bisect_x_xyz", "half_box", [MIRROR([T, T, T], bisect=[T, F, F])]),
    V("B17", "cube2_bisect_all_xyz_flip_all", "cube2",
      [MIRROR([T, T, T], bisect=[T, T, T], flip=[T, T, T])]),
    # mirror object
    V("O1", "mirror_obj_trans", "cube2", [MIRROR([T, F, F], mirror_object=MIRROR_EMPTY)],
      objects={MIRROR_EMPTY: EMPTY(loc=(0.5, 0, 0))}),
    V("O2", "mirror_obj_rot", "cube2", [MIRROR([T, F, F], mirror_object=MIRROR_EMPTY)],
      objects={MIRROR_EMPTY: EMPTY(rot=(0, 0, math.radians(30)))}),
    V("O3", "mirror_obj_nonuniform", "cube_offcenter", [MIRROR([T, F, F], mirror_object=MIRROR_EMPTY)],
      objects={MIRROR_EMPTY: EMPTY(loc=(0.2, 0.1, 0), rot=(0, 0, math.radians(45)), scale=(2, 1, 1))}),
    V("O4", "mirror_obj_owner_transform", "cube2", [MIRROR([F, T, F], mirror_object=MIRROR_EMPTY)],
      objects={MIRROR_EMPTY: EMPTY(loc=(0, 0.4, 0), rot=(0, 0.5, 0))},
      subject=SUBJECT(loc=(0.3, -0.2, 0.1), rot=(0.2, 0.3, 0.4), scale=(1.5, 1, 0.8))),
    V("O5", "mirror_obj_bisect", "cube_offcenter",
      [MIRROR([T, F, F], bisect=[T, F, F], flip=[F, F, F], mirror_object=MIRROR_EMPTY)],
      objects={MIRROR_EMPTY: EMPTY(loc=(0.1, 0, 0), scale=(1.5, 1, 1))}),
    V("O6", "mirror_obj_bisect_flip", "cube_offcenter",
      [MIRROR([T, F, F], bisect=[T, F, F], flip=[T, F, F], mirror_object=MIRROR_EMPTY)],
      objects={MIRROR_EMPTY: EMPTY(loc=(0.1, 0, 0), scale=(1.5, 1, 1))}),
    V("O7", "mirror_obj_uniform2_bisect", "cube_offcenter",
      [MIRROR([T, F, F], bisect=[T, F, F], mirror_object=MIRROR_EMPTY)],
      objects={MIRROR_EMPTY: EMPTY(loc=(0.25, 0, 0), scale=(2, 2, 2))}),
    V("O8", "mirror_obj_z", "cube_offcenter", [MIRROR([F, F, T], mirror_object=MIRROR_EMPTY)],
      objects={MIRROR_EMPTY: EMPTY(rot=(0.4, 0, 0))}),
    # clip
    V("C1", "clip_offcenter", "cube_offcenter", [MIRROR([T, F, F], use_clip=True)]),
    V("C2", "clip_offcenter_noclip", "cube_offcenter", [MIRROR([T, F, F], use_clip=False)]),
    V("C3", "clip_half", "half_box", [MIRROR([T, F, F], use_clip=True)]),
    # UV
    V("U1", "uv_flip_u", "uv_cube", [MIRROR([T, F, F], use_mirror_u=True)]),
    V("U2", "uv_flip_uv_offsets", "uv_cube", [MIRROR([T, F, F], use_mirror_u=True, use_mirror_v=True,
                                                    mirror_offset_u=0.2, mirror_offset_v=-0.1,
                                                    offset_u=0.25, offset_v=-0.5)]),
    V("U3", "uv_udim", "uv_cube", [MIRROR([T, F, F], use_mirror_u=True, use_mirror_v=True,
                                         use_mirror_udim=True)]),
    V("U4", "uv_bisect_flip_u", "uv_cube", [MIRROR([T, F, F], bisect=[T, F, F], flip=[T, F, F],
                                                  use_mirror_u=True)]),
    V("U5", "uv_xyz", "uv_cube", [MIRROR([T, T, T], use_mirror_u=True, use_mirror_v=True)]),
    # vertex groups
    V("G1", "groups_xyz", "groups_cube", [MIRROR([T, F, F])]),
    V("G2", "groups_bisect", "groups_cube", [MIRROR([T, F, F], bisect=[T, F, F], flip=[T, F, F])]),
    V("G3", "groups_off", "groups_cube", [MIRROR([T, F, F], use_mirror_vertex_groups=False)]),
    # custom normals
    V("N1", "custom_xyz", "cube2_custom", [MIRROR([T, T, T])]),
    V("N2", "custom_bisect", "cube2_custom", [MIRROR([T, F, F], bisect=[T, F, F])]),
    V("N3", "custom_nomerge", "cube2_custom", [MIRROR([T, F, F], use_mirror_merge=False)]),
    V("N4", "custom_half_y_y", "half_y_grid", [MIRROR([F, T, F])]),
    V("N5", "custom_rotated_bisect_x", "rotated_grid",
      [MIRROR([T, F, F], bisect=[T, F, F], use_mirror_merge=False)]),
    V("N6", "custom_chain_x_then_y_bisect", "rotated_grid",
      [MIRROR([T, F, F], bisect=[T, F, F]), MIRROR([F, T, F], bisect=[F, T, F])]),
    V("N7", "custom_xy_bisect", "rotated_grid", [MIRROR([T, T, F], bisect=[T, T, F])]),
    V("N8", "custom_chain_x_then_y_vertex", "rotated_grid_vertex_custom",
      [MIRROR([T, F, F], bisect=[T, F, F]), MIRROR([F, T, F], bisect=[F, T, F])]),
    V("N9", "custom_chain_x_then_y_corner", "rotated_grid_corner_custom",
      [MIRROR([T, F, F], bisect=[T, F, F]), MIRROR([F, T, F], bisect=[F, T, F])]),
    V("N10", "custom_rotated_bisect_vertex", "rotated_grid_vertex_custom",
      [MIRROR([T, F, F], bisect=[T, F, F], use_mirror_merge=False)]),
    V("N11", "custom_xy_bisect_corner", "rotated_grid_corner_custom",
      [MIRROR([T, T, F], bisect=[T, T, F])]),
    # material
    V("F1", "material_bisect", "cube2_mat", [MIRROR([T, F, F], bisect=[T, F, F])]),
    # loose / sharp / seam
    V("E1", "edges_half_x", "edges_half", [MIRROR([T, F, F])]),
    V("E2", "edges_half_bisect", "edges_half", [MIRROR([T, F, F], bisect=[T, F, F])]),
    V("E3", "edges_half_xz_bisect", "edges_half", [MIRROR([T, F, T], bisect=[T, F, T])]),
    # chains
    V("S1", "chain_mirror_x_then_y_bisect", "cube_offcenter",
      [MIRROR([T, F, F]), MIRROR([F, T, F], bisect=[F, T, F])]),
    V("S2", "chain_mirror_bisect_then_mirror", "cube2",
      [MIRROR([T, F, F], bisect=[T, F, F], flip=[T, F, F]), MIRROR([T, F, F])]),
    # random: hash-jittered quad grids (see random_grid)
    V("R1", "random_x_merge", "random_a", [MIRROR([T, F, F])]),
    V("R2", "random_xy_merge_groups", "random_b", [MIRROR([T, T, F])]),
    V("R3", "random_xyz_nomerge_uv", "random_c", [MIRROR([T, T, T], use_mirror_merge=False)]),
    V("R4", "random_xy_bisect_y_uv_groups", "random_d", [MIRROR([T, T, F], bisect=[F, T, F])]),
    # concave: comb polygons whose teeth cross the bisect plane
    V("K1", "concave_comb_bisect_x", "concave_comb", [MIRROR([T, F, F], bisect=[T, F, F])]),
    V("K2", "concave_comb_bisect_x_flip", "concave_comb", [MIRROR([T, F, F], bisect=[T, F, F], flip=[T, F, F])]),
    V("K3", "concave_comb_bisect_y", "concave_comb_y", [MIRROR([F, T, F], bisect=[F, T, F])]),
    V("K4", "concave_comb_bisect_y_flip", "concave_comb_y", [MIRROR([F, T, F], bisect=[F, T, F], flip=[F, T, F])]),
    V("K5", "concave_comb_bisect_x_corner_custom", "concave_comb_corner", [MIRROR([T, F, F], bisect=[T, F, F])]),
    V("K6", "concave_comb_disconnected_bisect_x_flip_order", "disconnected_comb_order",
      [MIRROR([T, F, F], bisect=[T, F, F], flip=[T, F, F])]),
    # crosschain: multi-modifier stacks across axes
    V("X1", "crosschain_x_bisect_then_z", "cube_offcenter",
      [MIRROR([T, F, F], bisect=[T, F, F]), MIRROR([F, F, T])]),
    V("X2", "crosschain_y_then_x_vertex", "rotated_grid_vertex_custom",
      [MIRROR([F, T, F]), MIRROR([T, F, F])]),
    V("X3", "crosschain_x_y_z_merge_off_middle", "cube_offcenter",
      [MIRROR([T, F, F]), MIRROR([F, T, F], use_mirror_merge=False), MIRROR([F, F, T])]),
    V("X4", "crosschain_x_flip_then_object_y_vertex", "rotated_grid_vertex_custom",
      [MIRROR([T, F, F], bisect=[T, F, F], flip=[T, F, F]), MIRROR([F, T, F], mirror_object=MIRROR_EMPTY)],
      objects={MIRROR_EMPTY: EMPTY(loc=(0.2, 0, 0), rot=(0, 0, math.radians(20)))}),
    V("X5", "crosschain_x_then_y_bisect_vertex", "rotated_grid_vertex_custom",
      [MIRROR([T, F, F]), MIRROR([F, T, F], bisect=[F, T, F])]),
    V("X6", "crosschain_x_y_bisect_then_wave_vertex", "rotated_grid_vertex_custom",
      [MIRROR([T, F, F]), MIRROR([F, T, F], bisect=[F, T, F]),
       WAVE(height=0.2, width=1.5, speed=0.0, time_offset=0.25)]),
]


def run_variant(v):
    reset()
    ob, input_doc, input_info = make_object("Subject", BASES[v["base"]]())
    st = v["subject"]
    ob.location = st["location"]
    ob.rotation_euler = st["rotation_euler"]
    ob.scale = st["scale"]
    objects_doc = {}
    for name, e in v["objects"].items():
        eo = bpy.data.objects.new(name, None)
        bpy.context.scene.collection.objects.link(eo)
        eo.location = e["location"]
        eo.rotation_euler = e["rotation_euler"]
        eo.scale = e["scale"]
        objects_doc[name] = dict(type=e["type"], location=list(e["location"]),
                                 rotation_euler=list(e["rotation_euler"]), scale=list(e["scale"]))
    stack_doc = []
    debug_stack = []
    for i, (t, params) in enumerate(v["stack"]):
        modifier_type = {"mirror": "MIRROR", "wave": "WAVE"}.get(t)
        assert modifier_type is not None, f"unexpected modifier type {t}"
        mod = ob.modifiers.new(name=f"{t}_{i}", type=modifier_type)
        for k, val in params.items():
            value = bpy.data.objects[val] if k == "mirror_object" else val
            setattr(mod, k, value)  # AttributeError here means a wrong RNA name
        readback = mirror_readback(mod) if t == "mirror" else {
            key: rna_json(getattr(mod, key)) for key in params
        }
        stack_doc.append(dict(type=t, name=mod.name, params=readback))
        debug_stack.append(dict(requested=params, readback=readback))
    output, out_debug = evaluate(ob, 1)
    doc = dict(
        variant=v["name"],
        blender_version=BLENDER_VERSION,
        stack=stack_doc,
        input=input_doc,
        objects=objects_doc,
        subject_transform=dict(st),
        output=output,
    )
    debug = dict(code=v["code"], stack=debug_stack, input=input_info, output=out_debug)
    return doc, debug


def write_outputs(name, doc, debug):
    with open(os.path.join(FIXTURES, name + ".json"), "w") as f:
        f.write(json.dumps(doc, indent=1) + "\n")
    with open(os.path.join(OUT, name + ".debug.json"), "w") as f:
        json.dump(debug, f, indent=1, default=str)


# ---------------------------------------------------------------- summary and probes

def cnt(doc):
    o = doc["output"]
    return f"v={len(o['vertices'])} e={len(o['edges'])} f={len(o['faces'])}"


def mapping(doc):
    inp = doc["input"]["vertices"]
    out = doc["output"]["vertices"]
    parts = []
    for i, co in enumerate(out):
        j = next((k for k, p in enumerate(inp) if all(abs(a - b) < 1e-9 for a, b in zip(p, co))), None)
        tag = f"in{j}" if j is not None else "new(" + ",".join(f"{c:g}" for c in co) + ")"
        parts.append(f"{i}<-{tag}")
    return " ".join(parts)


def xs(doc):
    return sorted({round(v[0], 7) for v in doc["output"]["vertices"]})


def summarize(docs):
    lines = [f"PR18B mirror oracle summary (Blender {bpy.app.version_string})", "", "variants:"]
    for v in VARIANTS:
        d = docs.get(v["name"])
        if d is None or "error" in d:
            lines.append(f"  {v['code']:4} {v['name']}: ERROR {d.get('error') if d else 'missing'}")
        else:
            lines.append(f"  {v['code']:4} {v['name']}: {cnt(d)}")
    lines.append("")
    lines.append("probes:")

    def ok(n):
        return n in docs and "error" not in docs[n]

    if ok("clip_offcenter") and ok("clip_offcenter_noclip"):
        same = docs["clip_offcenter"]["output"] == docs["clip_offcenter_noclip"]["output"]
        lines.append(f"(a) C1 clip True vs C2 clip False: identical output={same} "
                     f"[{cnt(docs['clip_offcenter'])}] vs [{cnt(docs['clip_offcenter_noclip'])}]")
    lines.append("(b) merge_threshold 0 (exact coincident copies) and merge counts:")
    for n in ["thr_grid_x_t0", "thr_grid_x_t05", "thr_grid_x_t1", "thr_grid_x_t2e3", "half_box_x", "half_box_x_t0",
              "cube2_xyz", "cube2_xyz_nomerge", "cube2_xyz_t0"]:
        if ok(n):
            lines.append(f"    {n}: {cnt(docs[n])}")
    if ok("half_box_x") and ok("half_box_x_t0"):
        lines.append(f"    half_box_x_t0 output == half_box_x output: "
                     f"{docs['half_box_x_t0']['output'] == docs['half_box_x']['output']}")
    lines.append("(c) thr_grid merge boundary (distinct output x values):")
    for n in ["thr_grid_x_t2e3", "thr_grid_x_t1", "thr_grid_x_t05", "thr_grid_x_t0"]:
        if ok(n):
            lines.append(f"    {n}: {cnt(docs[n])} x={xs(docs[n])}")
    if ok("cube2_flip_only") and ok("cube2_x"):
        lines.append(f"(d) B6 flip-only output == M1 cube2_x output: "
                     f"{docs['cube2_flip_only']['output'] == docs['cube2_x']['output']}")
    if ok("cube2_bisect_y_unmirrored") and ok("cube2_x"):
        lines.append(f"(d) B5 bisect-Y output == M1 cube2_x output: "
                     f"{docs['cube2_bisect_y_unmirrored']['output'] == docs['cube2_x']['output']}")
    if ok("mirror_obj_bisect") and ok("mirror_obj_bisect_flip"):
        a = docs["mirror_obj_bisect"]["output"]
        b = docs["mirror_obj_bisect_flip"]["output"]
        sa = sorted(tuple(round(c, 9) for c in v) for v in a["vertices"])
        sb = sorted(tuple(round(c, 9) for c in v) for v in b["vertices"])
        lines.append(f"(e) O5 vs O6 (flip off/on): identical output={a == b}, same vertex set={sa == sb}, "
                     f"[{cnt(docs['mirror_obj_bisect'])}] vs [{cnt(docs['mirror_obj_bisect_flip'])}]")
    if ok("half_box_x"):
        lines.append("(f) half_box_x (M10) output index <- input index (new = mirrored copy kept):")
        lines.append("    " + mapping(docs["half_box_x"]))
    if ok("half_box_x_t0"):
        lines.append("(f) half_box_x_t0 (M17) output index <- input index:")
        lines.append("    " + mapping(docs["half_box_x_t0"]))
    failed = [v["name"] for v in VARIANTS if docs.get(v["name"]) is None or "error" in docs[v["name"]]]
    lines.append("")
    lines.append(f"failed: {failed if failed else 'none'}")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        f.write(text)
    print(text)


def main():
    names = [v["name"] for v in VARIANTS]
    assert len(set(names)) == len(names), "duplicate variant names"
    docs = {}
    failures = []
    for v in VARIANTS:
        name = v["name"]
        try:
            doc, debug = run_variant(v)
            docs[name] = doc
            print(f"[ok] {v['code']} {name}: {cnt(doc)}")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            msg = f"{type(e).__name__}: {e}"
            doc = dict(variant=name, blender_version=BLENDER_VERSION,
                       stack=[dict(type=t, name=f"{t}_{i}", params=p) for i, (t, p) in enumerate(v["stack"])],
                       error=msg)
            debug = dict(code=v["code"], error=msg)
            docs[name] = doc
            failures.append(f"{v['code']} {name}: {msg}")
            print(f"[ERROR] {v['code']} {name}: {msg}")
        write_outputs(name, doc, debug)
    summarize(docs)
    print("FAILURES:" if failures else "ALL OK")
    for line in failures:
        print("  ", line)


main()
