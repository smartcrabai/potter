# Usage: /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --python tests/fixtures/pr3_uv/generate.py -- <outdir>
"""Blender 5.2 UV oracle for PR3 (UV propagation through mesh-rebuilding modifiers).

Writes <outdir>/cases_uv.json (Potter inputs) and <outdir>/blender/<variant>.json (evaluated corners).
Per corner key: "<face centroid>|<corner position>" at 5 decimals, with "#n" for repeats; the value maps
each UV layer name to its [u, v]. convert.py turns these into the repository fixtures.
"""
import json
import os
import sys
import traceback

import bpy

OUT = sys.argv[sys.argv.index("--") + 1]
os.makedirs(os.path.join(OUT, "blender"), exist_ok=True)


# UV values are k/1024 so they are exact in float32 and float64 alike.
def uv_random(loop, layer):
    a = 64 + (loop * 37 + layer * 11) % 800
    b = 64 + (loop * 53 + layer * 29 + 7) % 800
    return [a / 1024.0, b / 1024.0]


def uv_continuous(x, y):
    # Shared per vertex (continuous across faces): position-dependent, non-affine.
    return [(128 + 200 * x + 50 * y) / 1024.0, (300 + 100 * y + 30 * x * x) / 1024.0]


def grid_base(n):
    verts = [(float(i), float(j), 0.0) for j in range(n) for i in range(n)]
    faces = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            faces.append((a, a + 1, a + 1 + n, a + n))
    members = {
        "Arm.L": [k for k in range(n * n) if k % n >= n // 2],
        "Arm.R": [],
        "Center": [k for k in range(n * n) if k % n == 0],
        "Body": list(range(n * n)),
    }
    return dict(verts=verts, edges=[], faces=faces, members=members)


BOX_V = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
BOX_F = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (3, 7, 6, 2), (0, 4, 7, 3), (1, 2, 6, 5)]
BOX = dict(verts=BOX_V, edges=[], faces=BOX_F,
           members={"Arm.L": [0, 1, 2, 5, 6], "Arm.R": [], "Center": [0, 3, 4, 7], "Body": list(range(8))})
BOX_QUARTER = dict(verts=[(x + 0.25, y + 0.25, z + 0.25) for (x, y, z) in BOX_V], edges=[], faces=BOX_F, members={})
GRID3 = grid_base(3)
GRID4 = grid_base(4)
# A quad on the +X side of the mirror plane, with the reviewer's corner UVs (outside 0..1 on purpose).
HALF = dict(verts=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)], edges=[],
            faces=[(0, 1, 2, 3)], members={},
            uv_corners=[[0.0, 0.0], [1.0, 1.0], [2.0, 0.5], [-0.25, 1.25]])
# A polyline of loose edges (no faces, so no loops) with a UV layer that has zero values.
LINE = dict(verts=[(1.0, 0.0, 0.0), (1.0, 0.0, 0.5), (1.0, 0.0, 1.0)], edges=[(0, 1), (1, 2)],
            faces=[], members={})

BASES = {
    "uvcube": dict(BOX, uv="random"),
    "uvcube_quarter": dict(BOX_QUARTER, uv="random"),
    "uvcube_tile": dict(BOX, uv="tile"),
    "uvgrid3": dict(GRID3, uv="random"),
    "uvgrid3_cont": dict(GRID3, uv="continuous"),
    "uvgrid4": dict(GRID4, uv="random"),
    "uvhalf": dict(HALF, uv="explicit"),
    "uvline": dict(LINE, uv="random"),
}


def weight(gname, i):
    gi = ["Arm.L", "Arm.R", "Center", "Body"].index(gname)
    return round(0.05 + 0.05 * ((7 * i + 5 * gi) % 17), 4)


def uv_layers_for(base):
    """Per-corner UV values by layer name, in face/corner order."""
    if base["uv"] == "explicit":
        return {"UVMap": [list(uv) for uv in base["uv_corners"]]}
    layers = {"UVMap": [], "UV2": []}
    loop = 0
    for face in base["faces"]:
        for v in face:
            if base["uv"] == "tile":
                uv = uv_random(loop, 0)
                layers["UVMap"].append([uv[0] + 1.0, uv[1]])
                uv = uv_random(loop, 1)
                layers["UV2"].append([uv[0] + 2.0, uv[1] + 1.0])
            elif base["uv"] == "continuous":
                x, y = base["verts"][v][0], base["verts"][v][1]
                layers["UVMap"].append(uv_continuous(x, y))
                layers["UV2"].append(uv_continuous(y, x))
            else:
                layers["UVMap"].append(uv_random(loop, 0))
                layers["UV2"].append(uv_random(loop, 1))
            loop += 1
    return layers


def make_object(name, base, groups):
    me = bpy.data.meshes.new(name + "_mesh")
    me.from_pydata(base["verts"], base["edges"], base["faces"])
    me.update()
    if base.get("uv"):
        for layer_name, values in uv_layers_for(base).items():
            layer = me.uv_layers.new(name=layer_name)
            for li, uv in enumerate(values):
                layer.data[li].uv = uv
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    for gname, members in groups:
        vg = ob.vertex_groups.new(name=gname)
        for i in members:
            vg.add([i], weight(gname, i), "REPLACE")
    return ob


def groups_for(base):
    return [(g, idxs) for g, idxs in base["members"].items() if idxs]


def fmt(co):
    parts = []
    for c in co:
        r = round(float(c), 5)
        if r == 0:
            r = 0.0
        parts.append(f"{r:.5f}")
    return ",".join(parts)


def reset():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for me in list(bpy.data.meshes):
        bpy.data.meshes.remove(me)


def evaluate_corners(ob):
    dg = bpy.context.evaluated_depsgraph_get()
    ob_eval = ob.evaluated_get(dg)
    me = ob_eval.to_mesh()
    names = [layer.name for layer in me.uv_layers]
    corners = {}
    for poly in me.polygons:
        centroid = [0.0, 0.0, 0.0]
        for vi in poly.vertices:
            for k in range(3):
                centroid[k] += me.vertices[vi].co[k] / poly.loop_total
        for li in range(poly.loop_start, poly.loop_start + poly.loop_total):
            corner = me.loops[li].vertex_index
            base = fmt(centroid) + "|" + fmt(me.vertices[corner].co)
            key, n = base, 1
            while key in corners:
                n += 1
                key = f"{base}#{n}"
            corners[key] = {
                layer.name: [round(float(layer.data[li].uv[0]), 7), round(float(layer.data[li].uv[1]), 7)]
                for layer in me.uv_layers
            }
    counts = (len(me.vertices), len(me.edges), len(me.polygons))
    loops = sum(poly.loop_total for poly in me.polygons)
    ob_eval.to_mesh_clear()
    return dict(layers=names, corners=corners, counts=counts, loops=loops)


def make_cap(prop, base_name):
    """Cap object for `prop` ('start_cap' or 'end_cap'); Blender ignores the cap's own transform."""
    return make_object(f"Cap_{prop}", BASES[base_name], [])


def run_variant(v):
    reset()
    base = BASES[v["base"]]
    ob = make_object("Subject", base, groups_for(base) if v.get("groups", True) else [])
    mod = ob.modifiers.new(name=v["key"], type=v["mtype"])
    for prop, base_name in v["caps"].items():
        setattr(mod, prop, make_cap(prop, base_name))
    for k, val in v["props"].items():
        setattr(mod, k, val)
    for index, (mtype, props) in enumerate(v["extra"]):
        extra_mod = ob.modifiers.new(name=f"{v['key']}_{index}", type=mtype)
        for k, val in props.items():
            setattr(extra_mod, k, val)
    out = evaluate_corners(ob)
    doc = dict(variant=v["key"], modifier=v["mtype"], base=v["base"],
               blender_version=bpy.app.version_string, **out)
    with open(os.path.join(OUT, "blender", f"{v['key']}.json"), "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    print(f"[ok] {v['key']}: layers={out['layers']} counts={out['counts']} corners={len(out['corners'])}")


def V(key, base, mtype, props, potter_type, groups=True, extra=None, caps=None):
    return dict(key=key, base=base, mtype=mtype, props=props, potter_type=potter_type, groups=groups,
                extra=extra or [], caps=caps or {})


ONE_AXIS = dict(use_axis=(True, False, False), use_mirror_merge=True, merge_threshold=0.001)
ARRAY_CAP = dict(fit_type="FIXED_COUNT", count=3, use_relative_offset=True,
                 relative_offset_displace=(1.2, 0.0, 0.0), use_constant_offset=False,
                 use_object_offset=False, use_merge_vertices=False)
SUBSURF_CC = ("SUBSURF", dict(levels=1, render_levels=1, subdivision_type="CATMULL_CLARK"))

VARIANTS = [
    # Mirror: X axis with merge on unless noted.
    V("mirror_flip_u", "uvcube", "MIRROR", dict(ONE_AXIS, use_mirror_u=True), "mirror"),
    V("mirror_flip_uv_offset_uv", "uvcube", "MIRROR", dict(
        ONE_AXIS, use_mirror_u=True, use_mirror_v=True, mirror_offset_u=0.2, mirror_offset_v=-0.1,
        offset_u=0.25, offset_v=-0.5), "mirror"),
    V("mirror_tile_flip_u_udim", "uvcube_tile", "MIRROR", dict(ONE_AXIS, use_mirror_u=True, use_mirror_udim=True),
      "mirror"),
    V("mirror_udim", "uvcube", "MIRROR", dict(ONE_AXIS, use_mirror_udim=True), "mirror"),
    V("mirror_subsurf_uv_quarter", "uvcube_quarter", "MIRROR",
      dict(ONE_AXIS, use_mirror_u=True, use_mirror_v=True), "mirror", extra=[SUBSURF_CC]),
    # Mirror: X and Y axes, flip U only.
    V("mirror_xy_flip_u", "uvcube", "MIRROR", dict(
        use_axis=(True, True, False), use_mirror_merge=True, merge_threshold=0.001, use_mirror_u=True), "mirror"),
    # Mirror: a UDIM case with corner UVs outside 0..1 on a quad on the +X side of the mirror plane.
    V("mirror_udim_half_plane", "uvhalf", "MIRROR", dict(
        ONE_AXIS, use_mirror_u=True, use_mirror_v=True, use_mirror_udim=True), "mirror", groups=False),
    # Array: start cap = uvgrid3_cont, end cap = uvcube_tile.
    V("array_cap_both", "uvcube", "ARRAY", dict(ARRAY_CAP), "array",
      caps={"start_cap": "uvgrid3_cont", "end_cap": "uvcube_tile"}),
    # Array: copies touch (displacement 1.0) so merge removes shared vertices, plus a UV offset.
    V("array_merge_uv_offset", "uvcube", "ARRAY", dict(
        ARRAY_CAP, relative_offset_displace=(1.0, 0.0, 0.0), use_merge_vertices=True, offset_u=0.25), "array"),
    # Subdivision.
    V("subsurf_cc_l2_cont", "uvgrid3_cont", "SUBSURF", dict(levels=2, render_levels=2,
      subdivision_type="CATMULL_CLARK"), "subdivision"),
    V("subsurf_simple", "uvcube", "SUBSURF", dict(levels=1, render_levels=1, subdivision_type="SIMPLE"), "subdivision"),
    V("subsurf_simple_l2", "uvcube", "SUBSURF", dict(levels=2, render_levels=2, subdivision_type="SIMPLE"),
      "subdivision"),
    V("subsurf_smoothall_cube", "uvcube", "SUBSURF", dict(levels=1, render_levels=1, uv_smooth="SMOOTH_ALL"),
      "subdivision"),
    # Bevel: EDGES, width 0.1, limit NONE, at 2 and 3 segments.
    V("bevel_cube", "uvcube", "BEVEL", dict(width=0.1, segments=2, limit_method="NONE"), "bevel"),
    V("bevel_cube_seg3", "uvcube", "BEVEL", dict(width=0.1, segments=3, affect="EDGES", limit_method="NONE"),
      "bevel"),
    V("bevel_verts_seg2_uv", "uvcube", "BEVEL", dict(width=0.1, segments=2, affect="VERTICES",
      limit_method="NONE"), "bevel"),
    # Solidify.
    V("solidify_grid", "uvgrid3", "SOLIDIFY", dict(thickness=0.1), "solidify"),
    V("solidify_grid_norim", "uvgrid3", "SOLIDIFY", dict(thickness=0.1, use_rim=False), "solidify"),
    # Screw: a loose-edge polyline over a full turn with screw_offset 0.5 (no face loops in the profile).
    V("screw_loose_edge_offset", "uvline", "SCREW", dict(steps=3, angle=6.283185307179586, axis="Z",
      screw_offset=0.5), "screw", groups=False),
    V("screw_loose_edge_offset_stretch_v", "uvline", "SCREW", dict(steps=3, angle=6.283185307179586, axis="Z",
      screw_offset=0.5, use_stretch_v=True), "screw", groups=False),
    V("screw_stretch_uv", "uvgrid3", "SCREW", dict(steps=3, angle=1.5707963267948966, axis="Z",
      use_stretch_u=True, use_stretch_v=True), "screw"),
    # Decimate (collapse, ratio 0.5).
    V("decimate_uv", "uvgrid4", "DECIMATE", dict(decimate_type="COLLAPSE", ratio=0.5), "decimate"),
]


def potter_case(v):
    base = BASES[v["base"]]
    caps = {}
    for prop, base_name in v["caps"].items():
        cap_base = BASES[base_name]
        caps[prop] = dict(loc=[0, 0, 0], vertices=[list(c) for c in cap_base["verts"]],
                          faces=[list(f) for f in cap_base["faces"]],
                          uv_layers=[{"name": k, "values": vals} for k, vals in uv_layers_for(cap_base).items()])
    return dict(
        variant=v["key"],
        vertices=[list(c) for c in base["verts"]],
        faces=[list(f) for f in base["faces"]],
        edges=[list(e) for e in base["edges"]],
        uv_layers=[{"name": k, "values": vals} for k, vals in uv_layers_for(base).items()] if base.get("uv") else [],
        caps=caps,
        potter={"type": v["potter_type"], "params": {
            **{k: list(x) if isinstance(x, tuple) else x for k, x in v["props"].items()},
            **{prop: prop for prop in v["caps"]},
        }},
        potter_stack=[{"type": t, "params": {k: list(x) if isinstance(x, tuple) else x for k, x in pr.items()}}
                      for (t, pr) in v["extra"]],
    )


def main():
    cases = []
    failures = []
    for v in VARIANTS:
        cases.append(potter_case(v))
        try:
            run_variant(v)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            failures.append(f"{v['key']}: {e}")
    with open(os.path.join(OUT, "cases_uv.json"), "w") as f:
        json.dump(cases, f, indent=1, sort_keys=True)
    print("FAILURES:" if failures else "ALL OK")
    for f in failures:
        print("  ", f)
    if failures:
        sys.exit(1)


main()
