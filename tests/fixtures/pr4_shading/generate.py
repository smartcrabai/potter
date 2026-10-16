# Usage: /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --python tests/fixtures/pr4_shading/generate.py -- <scratch-dir>
"""PR4 shading oracle: Blender 5.2.2 LTS ground truth for shading attributes
(face smooth, edge sharp, corner custom normals) through mesh-rebuilding modifiers.

Writes <FIXTURES>/<variant>.json (schema from the PR4 brief; a refused variant gets
{"variant", "blender_version", "stack", "error"} instead) and copies each fixture plus a
debug dump (modifier RNA defaults, readbacks, counts) into <out>/.

Custom normals (deterministic, documented here):
  A base's "custom region" is a list of face indices (cube: 2,3; grid: 1,4).
  "dense" custom = every face has a list of corner normals:
    - region faces get SYNTHETIC normals (below);
    - all other faces keep Blender's computed corner normals, read from the input mesh after
      the smooth/sharp flags are applied and before the custom set.
  Synthetic normal for region face f, corner c (0-based position in the face loop):
    N  = Newell normal of the face (winding consistent, unit length)
    T  = (sin(1.7 f + 0.9 c), cos(1.3 f + 0.6 c), sin(0.7 f + 1.1 c))
    T' = normalize(T - (T.N) N)                  (tangent component, unit length)
    n  = normalize(N + 0.8 T')                   (atan(0.8) = 38.66 deg off the face normal, varies per corner)

Bases (indices as in the brief):
  cube        BOX; face_smooth [T,F,T,T,F,F]; sharp [1,2],[4,7]; region {2,3}
  grid        4x3 vertices (i, j, 0.2 sin(i+j)), index j*4+i; 6 quads; face_smooth [T,F,T,F,T,F];
              sharp [5,6],[1,5]; region {1,4}
  split_cube  cube with face 0 on private copies 8..11 (8=0, 9=3, 10=2, 11=1). Face 0 is
              (8,9,10,11): same winding as the cube. The brief literally says (8,11,10,9), which
              reverses face 0's winding; that literal form is kept as split_cube_reversed.
  line        (1,0,0),(1,0,0.5),(1,0,1); loose edges [0,1],[1,2]
"""
import json
import math
import os
import sys
import traceback

import bpy

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

CUBE_V = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
CUBE_F = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (3, 7, 6, 2), (0, 4, 7, 3), (1, 2, 6, 5)]


def base_cube():
    return dict(vertices=[vec(v) for v in CUBE_V], faces=[list(f) for f in CUBE_F], edges=[],
                face_smooth=[T, F, T, T, F, F], sharp_edges=[[1, 2], [4, 7]], region=[2, 3])

def base_cube_autosharp():
    """Cube with the extra sharp edges Blender marks on the dense custom input (control variants only)."""
    base = base_cube()
    base["sharp_edges"] = [[0, 1], [2, 3], [1, 2], [4, 7]]
    return base



def base_split_cube(reversed_face0):
    verts = [vec(v) for v in CUBE_V] + [vec(CUBE_V[i]) for i in (0, 3, 2, 1)]
    face0 = [8, 11, 10, 9] if reversed_face0 else [8, 9, 10, 11]
    faces = [face0] + [list(f) for f in CUBE_F[1:]]
    return dict(vertices=verts, faces=faces, edges=[], face_smooth=[T, F, T, T, F, F],
                sharp_edges=[[1, 2], [4, 7]], region=[])


def base_grid():
    verts = [vec((i, j, 0.2 * math.sin(i + j))) for j in range(3) for i in range(4)]
    faces = []
    for j in range(2):
        for i in range(3):
            a = j * 4 + i
            faces.append([a, a + 1, a + 5, a + 4])
    return dict(vertices=verts, faces=faces, edges=[], face_smooth=[T, F, T, F, T, F],
                sharp_edges=[[5, 6], [1, 5]], region=[1, 4])


def base_line():
    return dict(vertices=[vec((1, 0, 0)), vec((1, 0, 0.5)), vec((1, 0, 1))], faces=[], edges=[[0, 1], [1, 2]],
                face_smooth=[], sharp_edges=[], region=[])


def base_cube_jitter():
    """Smooth cube with each vertex moved by a fixed offset, so its faces are not planar and their
    corner normals differ from the face normals (no custom normals, no sharp edges)."""
    offsets = [(0.15 * math.sin(1.3 * i + 0.4), 0.15 * math.cos(0.9 * i + 1.1), 0.15 * math.sin(1.7 * i + 2.0))
               for i in range(8)]
    verts = [vec(tuple(c + o for c, o in zip(v, off))) for v, off in zip(CUBE_V, offsets)]
    return dict(vertices=verts, faces=[list(f) for f in CUBE_F], edges=[], face_smooth=[T] * 6,
                sharp_edges=[], region=[])


BASES = {
    "cube": lambda: base_cube(),
    "grid": base_grid,
    "split_cube": lambda: base_split_cube(False),
    "split_cube_reversed": lambda: base_split_cube(True),
    # Same split cube, but every face has a dense custom (region = all faces).
    "split_cube_dense": lambda: dict(base_split_cube(False), region=list(range(6))),
    "cube_autosharp": lambda: base_cube_autosharp(),
    "cube_jitter": base_cube_jitter,
    "line": base_line,
}


# ---------------------------------------------------------------- mesh construction

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


def make_object(name, spec, translation, custom, groups):
    """Build a Blender object from a base spec. Returns (object, input MESH doc, debug info)."""
    verts, faces = spec["vertices"], spec["faces"]
    me = bpy.data.meshes.new(name + "_mesh")
    me.from_pydata([tuple(v) for v in verts], [tuple(e) for e in spec["edges"]], [tuple(f) for f in faces])
    me.update()
    if faces:
        me.polygons.foreach_set("use_smooth", [bool(x) for x in spec["face_smooth"]])
    emap = {edge_key(e.vertices[0], e.vertices[1]): e.index for e in me.edges}
    sharp = [False] * len(me.edges)
    for a, b in spec["sharp_edges"]:
        k = edge_key(a, b)
        assert k in emap, f"{name}: sharp pair {a}-{b} is not an edge of the mesh"
        sharp[emap[k]] = True
    if sharp:
        me.edges.foreach_set("use_edge_sharp", sharp)
    me.update()
    info = {}
    custom_list = None
    if custom == "dense":
        computed = [vec(n.vector) for n in me.corner_normals]
        custom_list = []
        for fi, poly in enumerate(me.polygons):
            N = newell(verts, faces[fi])
            row = []
            for c in range(poly.loop_total):
                if fi in spec["region"]:
                    row.append(synthetic_normal(N, fi, c))
                else:
                    row.append(computed[poly.loop_start + c])
            custom_list.append(row)
        flat = [tuple(n) for row in custom_list for n in row]
        me.normals_split_custom_set(flat)
        me.update()
        # Blender stores custom loop normals quantized (int16 per component), so the values it
        # reads back differ from the requested ones by ~6e-5. The fixture input records the stored
        # values (what Blender actually evaluated); the requested-vs-stored gap is kept as a diagnostic.
        stored = [vec(n.vector) for n in me.corner_normals]
        requested = [x for row in custom_list for n in row for x in n]
        err = max(abs(a - b) for a, b in zip([x for r in stored for x in r], requested))
        info["custom_quantization_max_abs_err"] = err
        assert err < 1e-4, f"{name}: stored custom normals differ from requested by {err}"
        custom_list = [stored[poly.loop_start:poly.loop_start + poly.loop_total] for poly in me.polygons]
        assert [bool(p.use_smooth) for p in me.polygons] == [bool(x) for x in spec["face_smooth"]], name
        after_sharp = [bool(e.use_edge_sharp) for e in me.edges]
        info["auto_sharp_edges_added"] = [sorted(me.edges[i].vertices) for i in range(len(sharp))
                                          if after_sharp[i] and not sharp[i]]
        assert all(after_sharp[i] for i in range(len(sharp)) if sharp[i]), f"{name}: custom set cleared sharp edges"
    ob = bpy.data.objects.new(name, me)
    ob.location = translation
    bpy.context.scene.collection.objects.link(ob)
    doc = dict(vertices=verts, faces=faces, edges=spec["edges"], face_smooth=[bool(x) for x in spec["face_smooth"]],
               sharp_edges=[sorted(e.vertices) for e in me.edges if e.use_edge_sharp], custom_normals=custom_list)
    if groups:
        vg = ob.vertex_groups.new(name="keep")
        weights = {}
        for i, v in enumerate(verts):
            w = 1.0 if v[0] <= 1.0 + 1e-9 else 0.0
            vg.add([i], w, "REPLACE")
            weights[str(i)] = w
        doc["vertex_groups"] = {"keep": weights}
    return ob, doc, info


# ---------------------------------------------------------------- evaluation

def evaluate(ob, frame):
    bpy.context.scene.frame_set(frame)
    dg = bpy.context.evaluated_depsgraph_get()
    ob_eval = ob.evaluated_get(dg)
    me = ob_eval.to_mesh()
    try:
        verts = [vec(v.co) for v in me.vertices]
        faces = []
        for poly in me.polygons:
            loops = range(poly.loop_start, poly.loop_start + poly.loop_total)
            faces.append(dict(
                corners=[verts[me.loops[li].vertex_index] for li in loops],
                smooth=bool(poly.use_smooth),
                normals=[vec(me.corner_normals[li].vector) for li in loops],
            ))
        edges = [dict(verts=[verts[e.vertices[0]], verts[e.vertices[1]]], sharp=bool(e.use_edge_sharp))
                 for e in me.edges]
        debug = dict(vertices=len(me.vertices), edges=len(me.edges), faces=len(me.polygons),
                     corners=len(me.loops), has_custom_normals=bool(me.has_custom_normals))
        return dict(faces=faces, edges=edges), debug
    finally:
        ob_eval.to_mesh_clear()


# ---------------------------------------------------------------- RNA helpers

BLENDER_TYPE = {"subdivision": "SUBSURF"}
OBJECT_PARAMS = {"start_cap", "end_cap", "object"}
SKIP_PROPS = {"rna_type", "name", "type"}


def rna_json(value):
    if isinstance(value, bpy.types.ID):
        return value.name
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "__len__"):
        return [rna_json(x) for x in value]
    return str(value)


def default_props(mod):
    out = {}
    for p in mod.bl_rna.properties:
        if p.identifier in SKIP_PROPS or p.type in ("POINTER", "COLLECTION"):
            continue
        try:
            out[p.identifier] = rna_json(getattr(mod, p.identifier))
        except Exception as e:  # noqa: BLE001
            out[p.identifier] = f"<unreadable: {e}>"
    return out


def reset():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for me in list(bpy.data.meshes):
        bpy.data.meshes.remove(me)


# ---------------------------------------------------------------- variants

def V(name, base, custom, stack, objects=None, groups=False, frame=1):
    return dict(name=name, base=base, custom=custom, stack=stack, objects=objects or {}, groups=groups, frame=frame)


def ARRAY(count, merge, **extra):
    return dict(count=count, use_relative_offset=T, relative_offset_displace=[1, 0, 0],
                use_merge_vertices=merge, **extra)


def SUBSURF(levels, stype, use_custom=None):
    p = dict(levels=levels, render_levels=levels, subdivision_type=stype)
    if use_custom is not None:
        p["use_custom_normals"] = use_custom
    return ("subdivision", p)


BEVEL = dict(width=0.1, segments=1, affect="EDGES", limit_method="NONE")
CAP_OBJECTS = {"CapStart": ("grid", "dense", [-1, 0, 0]), "CapEnd": ("cube", "dense", [3, 0, 0])}
OPERAND = {"Operand": ("cube", None, [0.5, 0.5, 0.5])}

VARIANTS = [
    # mirror
    V("mirror_cube", "cube", "dense", [("mirror", {"use_axis": [T, F, F]})]),
    V("mirror_cube_nocustom", "cube", None, [("mirror", {"use_axis": [T, F, F]})]),
    V("mirror_cube_nomerge", "cube", "dense", [("mirror", {"use_axis": [T, F, F], "use_mirror_merge": F})]),
    V("mirror_cube_bisect_flip", "cube", "dense", [("mirror", {
        "use_axis": [T, F, F], "use_bisect_axis": [T, F, F], "use_bisect_flip_axis": [T, F, F]})]),
    V("mirror_grid", "grid", "dense", [("mirror", {"use_axis": [T, F, F]})]),
    # array
    V("array_cube", "cube", "dense", [("array", ARRAY(3, F))]),
    V("array_cube_merge", "cube", "dense", [("array", ARRAY(3, T))]),
    V("array_cube_caps", "cube", "dense", [("array", ARRAY(2, F, start_cap="CapStart", end_cap="CapEnd"))],
      objects=CAP_OBJECTS),
    # Control: same caps with zero object translations, to separate "cap placed at array offset" from
    # "cap object translation applied".
    V("array_cube_caps_zero_tf", "cube", "dense", [("array", ARRAY(2, F, start_cap="CapStart", end_cap="CapEnd"))],
      objects={"CapStart": ("grid", "dense", [0, 0, 0]), "CapEnd": ("cube", "dense", [0, 0, 0])}),
    # subdivision
    V("subsurf_cube_cc", "cube", "dense", [SUBSURF(1, "CATMULL_CLARK", False)]),
    V("subsurf_cube_cc_custom", "cube", "dense", [SUBSURF(1, "CATMULL_CLARK", True)]),
    V("subsurf_cube_cc_custom_l2", "cube", "dense", [SUBSURF(2, "CATMULL_CLARK", True)]),
    V("subsurf_cube_simple", "cube", "dense", [SUBSURF(1, "SIMPLE", False)]),
    V("subsurf_cube_simple_custom", "cube", "dense", [SUBSURF(1, "SIMPLE", True)]),
    V("subsurf_grid_cc_custom", "grid", "dense", [SUBSURF(1, "CATMULL_CLARK", True)]),
    V("subsurf_cube_cc_nocustom", "cube", None, [SUBSURF(1, "CATMULL_CLARK")]),
    V("subsurf_cube_cc_nocustom_ctrl", "cube_autosharp", None, [SUBSURF(1, "CATMULL_CLARK")]),
    # triangulate
    V("triangulate_cube_keep", "cube", "dense", [("triangulate", {"keep_custom_normals": T})]),
    V("triangulate_cube_drop", "cube", "dense", [("triangulate", {"keep_custom_normals": F})]),
    # No custom normals on a non-planar smooth cube: keep_custom_normals stores the pre-split corner normals.
    V("triangulate_cube_keep_jitter", "cube_jitter", None, [("triangulate", {"keep_custom_normals": T})]),
    # Control: no custom normals, but the edges Blender auto-marks when the dense cube custom set is applied
    # ([0,1],[2,3]). Used only to check "dropped custom == computed on the same flags".
    V("triangulate_cube_drop_ctrl", "cube_autosharp", None, [("triangulate", {"keep_custom_normals": F})]),
    V("triangulate_grid_keep", "grid", "dense", [("triangulate", {"keep_custom_normals": T})]),
    # solidify
    V("solidify_grid", "grid", "dense", [("solidify", {"thickness": 0.1})]),
    V("solidify_grid_norim", "grid", "dense", [("solidify", {"thickness": 0.1, "use_rim": F})]),
    V("solidify_grid_flip", "grid", "dense", [("solidify", {"thickness": 0.1, "use_flip_normals": T})]),
    V("solidify_grid_nocustom", "grid", None, [("solidify", {"thickness": 0.1})]),
    # screw (default angle left at its default)
    V("screw_line", "line", None, [("screw", {"steps": 4, "render_steps": 4, "use_smooth_shade": F,
                                              "use_normal_flip": F})]),
    V("screw_line_smooth", "line", None, [("screw", {"steps": 4, "render_steps": 4, "use_smooth_shade": T})]),
    # bevel
    V("bevel_cube", "cube", "dense", [("bevel", dict(BEVEL))]),
    V("bevel_cube_harden", "cube", "dense", [("bevel", dict(BEVEL, harden_normals=T))]),
    V("bevel_cube_mark_sharp", "cube", "dense", [("bevel", dict(BEVEL, mark_sharp=T))]),
    # decimate
    V("decimate_grid", "grid", "dense", [("decimate", {"decimate_type": "COLLAPSE", "ratio": 0.6})]),
    V("decimate_cube_nocustom", "cube", None, [("decimate", {"decimate_type": "COLLAPSE", "ratio": 0.5})]),
    # weld
    V("weld_split_cube", "split_cube", None, [("weld", {"mode": "ALL", "merge_threshold": 0.0001})]),
    V("weld_split_cube_connected", "split_cube", None, [("weld", {"mode": "CONNECTED", "merge_threshold": 0.0001})]),
    V("weld_split_cube_reversed", "split_cube_reversed", None,
      [("weld", {"mode": "ALL", "merge_threshold": 0.0001})]),
    # edge split
    V("edge_split_cube", "cube", "dense", [("edge_split", {"split_angle": math.radians(30),
                                                           "use_edge_angle": T, "use_edge_sharp": T})]),
    V("edge_split_sharp_only", "cube", "dense", [("edge_split", {"use_edge_angle": F, "use_edge_sharp": T})]),
    # wireframe
    V("wireframe_grid", "grid", "dense", [("wireframe", {"thickness": 0.05, "use_replace": T, "use_even_offset": T})]),
    V("wireframe_cube_nocustom", "cube", None, [("wireframe", {"thickness": 0.05, "use_replace": F})]),
    # boolean
    V("boolean_cube_difference", "cube", "dense", [("boolean", {"operation": "DIFFERENCE", "object": "Operand",
                                                               "operand_type": "OBJECT"})],
      objects=OPERAND),
    V("boolean_cube_union", "cube", "dense", [("boolean", {"operation": "UNION", "object": "Operand",
                                                          "operand_type": "OBJECT"})],
      objects=OPERAND),
    # mask
    V("mask_grid", "grid", "dense", [("mask", {"mode": "VERTEX_GROUP", "vertex_group": "keep", "threshold": 0.5})],
      groups=True),
    # remesh
    V("remesh_voxel_cube", "cube", "dense", [("remesh", {"mode": "VOXEL", "voxel_size": 0.2,
                                                         "use_smooth_shade": F})]),
    V("remesh_voxel_cube_smooth", "cube", "dense", [("remesh", {"mode": "VOXEL", "voxel_size": 0.2,
                                                                "use_smooth_shade": T})]),
    V("remesh_smooth_mode_grid", "grid", "dense", [("remesh", {"mode": "SMOOTH", "octree_depth": 4,
                                                               "use_smooth_shade": T})]),
    # skin
    V("skin_line", "line", None, [("skin", {"use_smooth_shade": F})]),
    V("skin_line_smooth", "line", None, [("skin", {"use_smooth_shade": T})]),
    # chains
    V("chain_mirror_subsurf", "cube", "dense", [("mirror", {"use_axis": [T, F, F]}),
                                                SUBSURF(1, "CATMULL_CLARK", True)]),
    V("chain_mirror_subsurf_custom_l1_triangulate", "cube", "dense", [
        ("mirror", {"use_axis": [T, F, F]}),
        ("subdivision", {"levels": 1, "render_levels": 1, "use_custom_normals": T}),
        ("triangulate", {"keep_custom_normals": T})]),
    V("chain_bevel_harden_triangulate", "cube", "dense", [
        ("bevel", dict(BEVEL, harden_normals=T)),
        ("triangulate", {"keep_custom_normals": T})]),
    # build (frame-dependent; evaluated at frame 4)
    V("build_cube", "cube", "dense", [("build", {"frame_start": 1, "frame_duration": 3})], frame=4),
    # Follow-up fixtures (rules derivation).
    V("weld_split_cube_dense", "split_cube_dense", "dense", [("weld", {"mode": "ALL", "merge_threshold": 0.0001})]),
    V("weld_split_cube_dense_connected", "split_cube_dense", "dense",
      [("weld", {"mode": "CONNECTED", "merge_threshold": 0.0001})]),
    V("triangulate_grid_drop", "grid", "dense", [("triangulate", {"keep_custom_normals": F})]),
    V("solidify_grid_offset0", "grid", "dense", [("solidify", {"thickness": 0.1, "offset": 0.0})]),
    V("subsurf_grid_cc_custom_l2", "grid", "dense", [SUBSURF(2, "CATMULL_CLARK", True)]),
]


def run_variant(v):
    reset()
    ob, subject_doc, subject_info = make_object("Subject", BASES[v["base"]](), (0, 0, 0), v["custom"], v["groups"])
    objects_doc = {}
    object_info = {}
    for name, (base, custom, translation) in v["objects"].items():
        _, objects_doc[name], object_info[name] = make_object(name, BASES[base](), translation, custom, False)
        objects_doc[name]["translation"] = [float(c) for c in translation]
    debug = dict(subject=subject_info, objects=object_info, stack=[])
    for i, (t, params) in enumerate(v["stack"]):
        mod = ob.modifiers.new(name=f"{t}_{i}", type=BLENDER_TYPE.get(t, t.upper()))
        entry = dict(type=t, defaults=default_props(mod), readback={})
        for k, val in params.items():
            value = bpy.data.objects[val] if k in OBJECT_PARAMS else val
            setattr(mod, k, value)  # AttributeError here means a wrong RNA name; recorded as error
            entry["readback"][k] = rna_json(getattr(mod, k))
        debug["stack"].append(entry)
    output, out_debug = evaluate(ob, v["frame"])
    debug["output"] = out_debug
    doc = dict(
        variant=v["name"],
        blender_version=BLENDER_VERSION,
        stack=[dict(type=t, name=f"{t}_{i}", params=params) for i, (t, params) in enumerate(v["stack"])],
        input=subject_doc,
        objects=objects_doc,
        output=output,
    )
    if any(t == "decimate" for t, _ in v["stack"]):
        doc["input"]["blender_edge_order"] = [sorted(edge.vertices) for edge in ob.data.edges]
    return doc, debug


def write_outputs(name, doc, debug):
    text = json.dumps(doc, indent=1) + "\n"
    with open(os.path.join(FIXTURES, name + ".json"), "w") as f:
        f.write(text)
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        f.write(text)
    with open(os.path.join(OUT, name + ".debug.json"), "w") as f:
        json.dump(debug, f, indent=1, default=str)


def main():
    failures = []
    for v in VARIANTS:
        try:
            doc, debug = run_variant(v)
            out = doc["output"]
            print(f"[ok] {v['name']}: faces={len(out['faces'])} edges={len(out['edges'])} "
                  f"smooth={sum(f['smooth'] for f in out['faces'])} "
                  f"sharp={sum(e['sharp'] for e in out['edges'])} "
                  f"custom={'dense' if doc['input']['custom_normals'] is not None else 'null'}")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            msg = f"{type(e).__name__}: {e}"
            doc = dict(variant=v["name"], blender_version=BLENDER_VERSION,
                       stack=[dict(type=t, name=f"{t}_{i}", params=p) for i, (t, p) in enumerate(v["stack"])],
                       error=msg)
            debug = dict(error=msg)
            failures.append(f"{v['name']}: {msg}")
            print(f"[ERROR] {v['name']}: {msg}")
        write_outputs(v["name"], doc, debug)
    print("FAILURES:" if failures else "ALL OK")
    for line in failures:
        print("  ", line)


main()
