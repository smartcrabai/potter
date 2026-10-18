"""Blender 5.2.2 skin oracle. Run: Blender -b --factory-startup --python generate_skin_oracle.py -- <outdir>

Set ONLY=<case> to run one case. ONLY=nogood_branch reproduces tests/fixtures/pr18f_skin_nogood_branch.json
exactly (34 faces, 39 vertices, 69 edges). A full batch run aborted with a segfault inside Blender's
final_skin after zero_radius_root, so run cases singly when regenerating.

Each case: vertices, edges, per-vertex skin (radius, root), skin modifier params. Dumps the
evaluated mesh (vertices, faces, smooth flags) to <outdir>/<case>.json.
"""
import bpy
import bmesh
import json
import os
import sys

OUT = sys.argv[sys.argv.index("--") + 1]
os.makedirs(OUT, exist_ok=True)

CASES = {
    # straight 3-vertex chain, root at the first vertex, default radii
    "line_default": dict(
        verts=[(1, 0, 0), (1, 0, 0.5), (1, 0, 1)],
        edges=[(0, 1), (1, 2)],
        skin=[((0.25, 0.25), True), ((0.25, 0.25), False), ((0.25, 0.25), False)],
        params={},
    ),
    # bent chain with varying nonuniform radii, root at vertex 0
    "bent_chain": dict(
        verts=[(0, 0, 0), (1, 0, 0), (1, 1, 0.5), (2, 1, 2)],
        edges=[(0, 1), (1, 2), (2, 3)],
        skin=[((0.5, 0.25), True), ((0.2, 0.4), False), ((0.3, 0.3), False), ((0.1, 0.6), False)],
        params={},
    ),
    # same bent chain, branch_smoothing=1 (expected no-op on a chain: no hull edges exist)
    "bent_chain_smooth1": dict(
        verts=[(0, 0, 0), (1, 0, 0), (1, 1, 0.5), (2, 1, 2)],
        edges=[(0, 1), (1, 2), (2, 3)],
        skin=[((0.5, 0.25), True), ((0.2, 0.4), False), ((0.3, 0.3), False), ((0.1, 0.6), False)],
        params={"branch_smoothing": 1.0},
    ),
    # same bent chain, symmetry all off / all on
    "bent_chain_sym_off": dict(
        verts=[(0, 0, 0), (1, 0, 0), (1, 1, 0.5), (2, 1, 2)],
        edges=[(0, 1), (1, 2), (2, 3)],
        skin=[((0.5, 0.25), True), ((0.2, 0.4), False), ((0.3, 0.3), False), ((0.1, 0.6), False)],
        params={"use_x_symmetry": False, "use_y_symmetry": False, "use_z_symmetry": False},
    ),
    "bent_chain_sym_all": dict(
        verts=[(0, 0, 0), (1, 0, 0), (1, 1, 0.5), (2, 1, 2)],
        edges=[(0, 1), (1, 2), (2, 3)],
        skin=[((0.5, 0.25), True), ((0.2, 0.4), False), ((0.3, 0.3), False), ((0.1, 0.6), False)],
        params={"use_x_symmetry": True, "use_y_symmetry": True, "use_z_symmetry": True},
    ),
    # chain with root in the middle (degree-2 root -> Blender seam frames)
    "middle_root": dict(
        verts=[(0, 0, 0), (0, 0, 1), (0, 0, 2)],
        edges=[(0, 1), (1, 2)],
        skin=[((0.25, 0.25), False), ((0.5, 0.25), True), ((0.25, 0.25), False)],
        params={},
    ),
    # chain with smooth shading
    "line_smooth": dict(
        verts=[(1, 0, 0), (1, 0, 0.5), (1, 0, 1)],
        edges=[(0, 1), (1, 2)],
        skin=[((0.25, 0.25), True), ((0.25, 0.25), False), ((0.25, 0.25), False)],
        params={"use_smooth_shade": True},
    ),
    # bent chain smooth shading
    "bent_chain_smooth": dict(
        verts=[(0, 0, 0), (1, 0, 0), (1, 1, 0.5), (2, 1, 2)],
        edges=[(0, 1), (1, 2), (2, 3)],
        skin=[((0.5, 0.25), True), ((0.2, 0.4), False), ((0.3, 0.3), False), ((0.1, 0.6), False)],
        params={"use_smooth_shade": True},
    ),
    # Y-shaped branch: root at leaf 0; vertex 1 is the branch node (degree 3)
    "y_branch": dict(
        verts=[(0, 0, 0), (1, 0, 0), (2, 1, 0), (2, -1, 0.5)],
        edges=[(0, 1), (1, 2), (1, 3)],
        skin=[((0.3, 0.3), True), ((0.3, 0.3), False), ((0.2, 0.2), False), ((0.2, 0.2), False)],
        params={},
    ),
    "y_branch_sym_off_smooth1": dict(
        verts=[(0, 0, 0), (1, 0, 0), (2, 1, 0), (2, -1, 0.5)],
        edges=[(0, 1), (1, 2), (1, 3)],
        skin=[((0.3, 0.3), True), ((0.3, 0.3), False), ((0.2, 0.2), False), ((0.2, 0.2), False)],
        params={"use_x_symmetry": False, "branch_smoothing": 1.0},
    ),
    # isolated vertex (no edges) next to a 3-vertex line: Blender builds a box frame for it
    "isolated_vertex": dict(
        verts=[(1, 0, 0), (1, 0, 0.5), (1, 0, 1), (5, 5, 5)],
        edges=[(0, 1), (1, 2)],
        skin=[((0.25, 0.25), True), ((0.25, 0.25), False), ((0.25, 0.25), False), ((0.25, 0.25), False)],
        params={},
    ),
    # quad cycle, no skin layer: Blender's default layer marks vertex 0 as root
    "cycle_quad_default": dict(
        verts=[(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
        edges=[(0, 1), (1, 2), (2, 3), (3, 0)],
        skin=None,
        params={},
    ),
    # quad cycle with explicit radii and root on vertex 2
    "cycle_quad_root2": dict(
        verts=[(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
        edges=[(0, 1), (1, 2), (2, 3), (3, 0)],
        skin=[((0.3, 0.2), False), ((0.1, 0.2), False), ((0.25, 0.4), True), ((0.2, 0.2), False)],
        params={},
    ),
    # bent chain with a degree-2 root in the middle (seam frames) and nonuniform radii
    "middle_root_bent": dict(
        verts=[(0, 0, 0), (1, 0, 0), (1, 1, 0.5), (2, 1, 2)],
        edges=[(0, 1), (1, 2), (2, 3)],
        skin=[((0.2, 0.3), False), ((0.5, 0.25), True), ((0.3, 0.3), False), ((0.1, 0.4), False)],
        params={},
    ),
    # two roots on one line (both ends)
    "two_roots_line": dict(
        verts=[(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],
        edges=[(0, 1), (1, 2), (2, 3)],
        skin=[((0.25, 0.25), True), ((0.25, 0.25), False), ((0.25, 0.25), False), ((0.25, 0.25), True)],
        params={},
    ),
    # two islands, each with a root at its first vertex
    "two_islands": dict(
        verts=[(0, 0, 0), (0, 0, 1), (3, 0, 0), (3, 0, 1)],
        edges=[(0, 1), (2, 3)],
        skin=[((0.25, 0.25), True), ((0.25, 0.25), False), ((0.4, 0.2), True), ((0.25, 0.25), False)],
        params={},
    ),
    # middle root in a chain with a bend and a leaf-root mix (two roots in one island)
    "two_roots_bent": dict(
        verts=[(0, 0, 0), (1, 0, 0), (1, 1, 0.5), (2, 1, 2)],
        edges=[(0, 1), (1, 2), (2, 3)],
        skin=[((0.2, 0.3), True), ((0.5, 0.25), False), ((0.3, 0.3), False), ((0.1, 0.4), True)],
        params={},
    ),
    # Y branch: symmetry only (branch_smoothing default)
    "y_branch_sym_off": dict(
        verts=[(0, 0, 0), (1, 0, 0), (2, 1, 0), (2, -1, 0.5)],
        edges=[(0, 1), (1, 2), (1, 3)],
        skin=[((0.3, 0.3), True), ((0.3, 0.3), False), ((0.2, 0.2), False), ((0.2, 0.2), False)],
        params={"use_x_symmetry": False, "use_y_symmetry": False, "use_z_symmetry": False},
    ),
    # Y branch: branch smoothing only (symmetry default)
    "y_branch_smooth1": dict(
        verts=[(0, 0, 0), (1, 0, 0), (2, 1, 0), (2, -1, 0.5)],
        edges=[(0, 1), (1, 2), (1, 3)],
        skin=[((0.3, 0.3), True), ((0.3, 0.3), False), ((0.2, 0.2), False), ((0.2, 0.2), False)],
        params={"branch_smoothing": 1.0},
    ),
    # zero radius on the middle vertex of a line
    "zero_radius_mid": dict(
        verts=[(1, 0, 0), (1, 0, 0.5), (1, 0, 1)],
        edges=[(0, 1), (1, 2)],
        skin=[((0.25, 0.25), True), ((0.0, 0.0), False), ((0.25, 0.25), False)],
        params={},
    ),
    # zero radius on the root vertex
    "zero_radius_root": dict(
        verts=[(1, 0, 0), (1, 0, 0.5), (1, 0, 1)],
        edges=[(0, 1), (1, 2)],
        skin=[((0.0, 0.0), True), ((0.25, 0.25), False), ((0.25, 0.25), False)],
        params={},
    ),
    # negative radii on every vertex of a line
    "negative_radius_line": dict(
        verts=[(1, 0, 0), (1, 0, 0.5), (1, 0, 1)],
        edges=[(0, 1), (1, 2)],
        skin=[((-0.25, -0.25), True), ((-0.25, -0.25), False), ((-0.25, -0.25), False)],
        params={},
    ),
    # mixed sign radii on a line
    "mixed_radius_line": dict(
        verts=[(1, 0, 0), (1, 0, 0.5), (1, 0, 1)],
        edges=[(0, 1), (1, 2)],
        skin=[((0.25, -0.25), True), ((0.25, 0.25), False), ((0.25, 0.25), False)],
        params={},
    ),
    # vertex-only mesh: two isolated vertices (box each)
    "vertices_only": dict(
        verts=[(0, 0, 0), (2, 0, 0)],
        edges=[],
        skin=[((0.25, 0.25), False), ((0.5, 0.25), False)],
        params={},
    ),
    # line plus isolated vertex with explicit radius
    "isolated_vertex_radius": dict(
        verts=[(1, 0, 0), (1, 0, 0.5), (1, 0, 1), (5, 5, 5)],
        edges=[(0, 1), (1, 2)],
        skin=[((0.25, 0.25), True), ((0.25, 0.25), False), ((0.25, 0.25), False), ((0.5, 0.25), False)],
        params={},
    ),
    "isolated_negative": dict(
        verts=[(5, 5, 5)],
        edges=[],
        skin=[((-0.25, -0.25), False)],
        params={},
    ),
    "edge_negative_avg": dict(
        verts=[(1, 0, 0), (1, 0, 0.5)],
        edges=[(0, 1)],
        skin=[((-0.5, -0.5), True), ((0.25, 0.25), False)],
        params={},
    ),
    "edge_zero_avg": dict(
        verts=[(1, 0, 0), (1, 0, 0.5)],
        edges=[(0, 1)],
        skin=[((-0.25, -0.25), True), ((0.25, 0.25), False)],
        params={},
    ),
    "isolated_default": dict(
        verts=[(0, 0, 0)],
        edges=[],
        skin=None,
        params={},
    ),
    "sym_branch_x_on": dict(
        verts=[(0, 0, 0), (-1, 0, 0.2), (1, 0, 0.2), (0, 1, 1)],
        edges=[(0, 1), (0, 2), (0, 3)],
        skin=[((0.3, 0.3), True), ((0.2, 0.2), False), ((0.2, 0.2), False), ((0.2, 0.2), False)],
        params={"use_x_symmetry": True},
    ),
    "sym_branch_x_off": dict(
        verts=[(0, 0, 0), (-1, 0, 0.2), (1, 0, 0.2), (0, 1, 1)],
        edges=[(0, 1), (0, 2), (0, 3)],
        skin=[((0.3, 0.3), True), ((0.2, 0.2), False), ((0.2, 0.2), False), ((0.2, 0.2), False)],
        params={"use_x_symmetry": False},
    ),
    "symmetry_branch_x_on": dict(
        verts=[(0, 0, 0), (0.4, 0.0, -0.46), (-0.2, 0.8227241335952168, 0.23), (-0.2, -0.6062177826491069, -0.23)],
        edges=[(0, 1), (0, 2), (0, 3)],
        skin=[((0.23, 0.31), True), ((0.16, 0.19), False), ((0.19, 0.23), False), ((0.22, 0.19), False)],
        params={"use_x_symmetry": True},
    ),
    "symmetry_branch_x_off": dict(
        verts=[(0, 0, 0), (0.4, 0.0, -0.46), (-0.2, 0.8227241335952168, 0.23), (-0.2, -0.6062177826491069, -0.23)],
        edges=[(0, 1), (0, 2), (0, 3)],
        skin=[((0.23, 0.31), True), ((0.16, 0.19), False), ((0.19, 0.23), False), ((0.22, 0.19), False)],
        params={"use_x_symmetry": False},
    ),
    "symmetry_branch_y_on": dict(
        verts=[(0, 0, 0), (0, 0.4, -0.46), (0.8227241335952168, -0.2, 0.23), (-0.6062177826491069, -0.2, -0.23)],
        edges=[(0, 1), (0, 2), (0, 3)],
        skin=[((0.23, 0.31), True), ((0.16, 0.19), False), ((0.19, 0.23), False), ((0.22, 0.19), False)],
        params={"use_y_symmetry": True},
    ),
    "symmetry_branch_y_off": dict(
        verts=[(0, 0, 0), (0, 0.4, -0.46), (0.8227241335952168, -0.2, 0.23), (-0.6062177826491069, -0.2, -0.23)],
        edges=[(0, 1), (0, 2), (0, 3)],
        skin=[((0.23, 0.31), True), ((0.16, 0.19), False), ((0.19, 0.23), False), ((0.22, 0.19), False)],
        params={"use_y_symmetry": False},
    ),
    "symmetry_branch_z_on": dict(
        verts=[(0, 0, 0), (-0.46, 0, 0.4), (0.23, 0.8227241335952168, -0.2), (-0.23, -0.6062177826491069, -0.2)],
        edges=[(0, 1), (0, 2), (0, 3)],
        skin=[((0.23, 0.31), True), ((0.16, 0.19), False), ((0.19, 0.23), False), ((0.22, 0.19), False)],
        params={"use_z_symmetry": True},
    ),
    "symmetry_branch_z_off": dict(
        verts=[(0, 0, 0), (-0.46, 0, 0.4), (0.23, 0.8227241335952168, -0.2), (-0.23, -0.6062177826491069, -0.2)],
        edges=[(0, 1), (0, 2), (0, 3)],
        skin=[((0.23, 0.31), True), ((0.16, 0.19), False), ((0.19, 0.23), False), ((0.22, 0.19), False)],
        params={"use_z_symmetry": False},
    ),
    "symmetry_branch_y_axis_on": dict(
        verts=[(0, 0, 0), (0.4, 0.0, -0.46), (-0.2, 0.8227241335952168, 0.23), (-0.2, -0.6062177826491069, -0.23)],
        edges=[(0, 1), (0, 2), (0, 3)],
        skin=[((0.23, 0.31), True), ((0.16, 0.19), False), ((0.19, 0.23), False), ((0.22, 0.19), False)],
        params={"use_x_symmetry": False, "use_y_symmetry": True, "use_z_symmetry": False},
    ),
    "symmetry_branch_y_axis_off": dict(
        verts=[(0, 0, 0), (0.4, 0.0, -0.46), (-0.2, 0.8227241335952168, 0.23), (-0.2, -0.6062177826491069, -0.23)],
        edges=[(0, 1), (0, 2), (0, 3)],
        skin=[((0.23, 0.31), True), ((0.16, 0.19), False), ((0.19, 0.23), False), ((0.22, 0.19), False)],
        params={"use_x_symmetry": False, "use_y_symmetry": False, "use_z_symmetry": False},
    ),
    "symmetry_branch_z_axis_on": dict(
        verts=[(0, 0, 0), (-0.46, 0, 0.4), (0.23, 0.8227241335952168, -0.2), (-0.23, -0.6062177826491069, -0.2)],
        edges=[(0, 1), (0, 2), (0, 3)],
        skin=[((0.23, 0.31), True), ((0.16, 0.19), False), ((0.19, 0.23), False), ((0.22, 0.19), False)],
        params={"use_x_symmetry": False, "use_y_symmetry": False, "use_z_symmetry": True},
    ),
    "symmetry_branch_z_axis_off": dict(
        verts=[(0, 0, 0), (-0.46, 0, 0.4), (0.23, 0.8227241335952168, -0.2), (-0.23, -0.6062177826491069, -0.2)],
        edges=[(0, 1), (0, 2), (0, 3)],
        skin=[((0.23, 0.31), True), ((0.16, 0.19), False), ((0.19, 0.23), False), ((0.22, 0.19), False)],
        params={"use_x_symmetry": False, "use_y_symmetry": False, "use_z_symmetry": False},
    ),
    "cube_branch_centered": dict(
        verts=[(-.5,-.5,-.5),(.5,-.5,-.5),(.5,.5,-.5),(-.5,.5,-.5),(-.5,-.5,.5),(.5,-.5,.5),(.5,.5,.5),(-.5,.5,.5)],
        edges=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)],
        skin=[((0.25,0.25), i==0) for i in range(8)],
        params={},
    ),
    "cube_branch": dict(
        verts=[(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)],
        edges=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)],
        skin=[((0.25,0.25), i==0) for i in range(8)],
        params={},
    ),
    "consumer_branch_x_off": dict(
        verts=[(0,0,0),(1.2,0,0),(0,1.2,0),(0,0,1.2)],
        edges=[(0,1),(0,2),(0,3)],
        skin=[((0.3,0.2),True),((0.25,0.35),False),((0.2,0.3),False),((0.35,0.25),False)],
        params={"branch_smoothing":0.35,"use_smooth_shade":False,"use_x_symmetry":False,"use_y_symmetry":False,"use_z_symmetry":False},
    ),
    "consumer_branch_x_on": dict(
        verts=[(0,0,0),(1.2,0,0),(0,1.2,0),(0,0,1.2)],
        edges=[(0,1),(0,2),(0,3)],
        skin=[((0.3,0.2),True),((0.25,0.35),False),((0.2,0.3),False),((0.35,0.25),False)],
        params={"branch_smoothing":0.35,"use_smooth_shade":False,"use_x_symmetry":True,"use_y_symmetry":False,"use_z_symmetry":False},
    ),
    "detached_branch": dict(
        verts=[(0.0,0.0,0.0),(0.0944871312282557,1.3947412142144908,-0.6789465782603507),(0.582235576162845,0.133558627622563,-1.9779341711897558),(0.4590388331551346,1.4484963036383605,0.06981012119291923)],
        edges=[(0,1),(0,2),(0,3)],
        skin=[((0.46170452834529013,0.3036252966189872),True),((0.5412495143022006,0.6302679453037413),False),((0.14344811822036752,0.12353176588156564),False),((0.379951191640614,0.26134121447787595),False)],
        params={},
    ),
    "nogood_branch": dict(
        verts=[(0.0,0.0,0.0),(-0.615750305639683,-0.2491899355187869,-0.656205807326401),(0.47299044158676695,0.6302041847568558,-0.7520664268446838),(0.4316167520938852,0.7697423721761393,-1.843958393914058),(-0.2658301541233946,-0.615534116543724,0.2375020748994344),(0.3887711823897751,-2.367231987934589,1.2562332708604813)],
        edges=[(0,1),(0,2),(0,3),(0,4),(0,5)],
        skin=[((0.5944025940585477,0.3933926275328914),True),((0.7498587561158992,0.8263900603670049),False),((0.7504624463412104,0.5995031041071985),False),((0.36174570953694996,0.3270112984977598),False),((0.8666491075896481,0.3874494757263359),False),((0.3010493443071548,0.646648963948643),False)],
        params={},
    ),
    # no skin layer at all: Blender returns the input mesh unchanged
    # skin layer present but no root vertex: Blender reports "No valid root vertex" and still evaluates
    "no_root_line": dict(
        verts=[(1, 0, 0), (1, 0, 0.5), (1, 0, 1)],
        edges=[(0, 1), (1, 2)],
        skin=[((0.25, 0.25), False), ((0.25, 0.25), False), ((0.25, 0.25), False)],
        params={},
    ),
    "no_skin_layer": dict(
        verts=[(1, 0, 0), (1, 0, 0.5), (1, 0, 1)],
        edges=[(0, 1), (1, 2)],
        skin=None,
        params={},
    ),
}


def dump(name, case):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    layer = bm.verts.layers.skin.verify() if case["skin"] is not None else None
    bverts = [bm.verts.new(v) for v in case["verts"]]
    for a, b in case["edges"]:
        bm.edges.new((bverts[a], bverts[b]))
    if case["skin"] is not None:
        for vert, (radius, root) in zip(bverts, case["skin"]):
            vs = vert[layer]
            vs.radius = radius
            vs.use_root = root
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    mod = ob.modifiers.new("skin", "SKIN")
    for key, value in case["params"].items():
        setattr(mod, key, value)
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    out = ev.to_mesh()
    data = {
        "case": name,
        "blender_version": "5.2.2",
        "params": case["params"],
        "vertices": [[round(c, 9) for c in v.co] for v in out.vertices],
        "edges": len(out.edges),
        "faces": [list(p.vertices) for p in out.polygons],
        "face_smooth": [bool(p.use_smooth) for p in out.polygons],
        "face_normals": [[round(c, 9) for c in p.normal] for p in out.polygons],
        "edge_sharp": [bool(e.use_edge_sharp) for e in out.edges],
    }
    ev.to_mesh_clear()
    with open(os.path.join(OUT, name + ".json"), "w") as fh:
        json.dump(data, fh, indent=1)
    print(name, "verts", len(data["vertices"]), "faces", len(data["faces"]), "smooth", sum(data["face_smooth"]))


ONLY = os.environ.get("ONLY")
for case_name, case_data in CASES.items():
    if ONLY and case_name != ONLY:
        continue
    dump(case_name, case_data)
