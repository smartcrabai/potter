"""Blender 5.2.2 Triangulate modifier oracle: BEAUTY quads and BEAUTY n-gons, one face per case.

Run: Blender -b --factory-startup --threads 1 --python generate_triangulate_oracle.py -- <out.json>

The Triangulate modifier (`MOD_triangulate.cc`) runs `BM_mesh_triangulate`, the same BMesh kernel
as the `mesh.triangulate` edit op, so each case pins both. Triangles are face-local corner indices
(0..n-1) in the evaluated polygon order, with the winding Blender writes.

Cases: convex, concave, self-intersecting (star) and zero-area n-gons (collinear zigzags and a
figure-eight, whose Newell sum is exactly zero), and random non-planar, bowtie and concave quads.
"""
import json
import math
import random
import struct
import sys

import bpy

OUT = sys.argv[sys.argv.index("--") + 1]
rng = random.Random(20261019)


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def random_quad():
    return [tuple(f32(rng.uniform(-1, 1)) for _ in range(3)) for _ in range(4)]


def ngon_convex(n):
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n + rng.uniform(-0.1, 0.1)
        r = rng.uniform(0.8, 1.2)
        pts.append((f32(r * math.cos(a)), f32(r * math.sin(a)), f32(rng.uniform(-0.05, 0.05))))
    return pts


def ngon_concave(n):
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        r = 1.0 if i % 2 == 0 else rng.uniform(0.2, 0.4)
        pts.append((f32(r * math.cos(a)), f32(r * math.sin(a)), 0.0))
    return pts


def ngon_star(n, step):
    # Visits vertex (i * step) mod n in order, so the boundary crosses itself (gcd(step, n) = 1).
    pts = []
    for i in range(n):
        a = 2 * math.pi * ((i * step) % n) / n
        r = rng.uniform(0.9, 1.1)
        pts.append((f32(r * math.cos(a)), f32(r * math.sin(a)), f32(rng.uniform(-0.05, 0.05))))
    return pts


def ngon_zigzag(n, step):
    # Collinear corners in a zigzag order: every Newell term is exactly zero.
    return [(float((i * step) % n), 0.0, 0.0) for i in range(n)]


FIGURE_EIGHT = [(0.0, 0.0, 0.0), (2.0, 2.0, 0.0), (4.0, 0.0, 0.0), (4.0, 2.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0)]

bpy.ops.wm.read_factory_settings(use_empty=True)
cases = []


def evaluate(kind, points):
    me = bpy.data.meshes.new("tri_case")
    me.from_pydata(points, [], [tuple(range(len(points)))])
    me.update()
    obj = bpy.data.objects.new("tri_case", me)
    bpy.context.collection.objects.link(obj)
    mod = obj.modifiers.new("Triangulate", "TRIANGULATE")
    mod.quad_method = "BEAUTY"
    mod.ngon_method = "BEAUTY"
    mod.min_vertices = 4
    mod.keep_custom_normals = False
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    result = evaluated.to_mesh()
    triangles = [[int(index) for index in polygon.vertices] for polygon in result.polygons]
    evaluated.to_mesh_clear()
    bpy.data.objects.remove(obj)
    bpy.data.meshes.remove(me)
    assert len(triangles) == len(points) - 2, (kind, len(triangles))
    cases.append({
        "kind": kind,
        "positions": [list(point) for point in points],
        "triangles": triangles,
    })


for _ in range(40):
    evaluate("quad_nonplanar", random_quad())
for _ in range(8):
    points = random_quad()
    points[2], points[3] = points[3], points[2]
    evaluate("quad_bowtie", points)
for _ in range(8):
    evaluate("quad_concave", ngon_concave(4))
for n in (5, 6, 7, 8, 10):
    for _ in range(4):
        evaluate("ngon_convex_%d" % n, ngon_convex(n))
        evaluate("ngon_concave_%d" % n, ngon_concave(n))
for n, step in ((5, 2), (7, 2), (8, 3), (10, 3)):
    for _ in range(4):
        evaluate("ngon_star_%d" % n, ngon_star(n, step))
    evaluate("ngon_zero_area_zigzag_%d" % n, ngon_zigzag(n, step))
evaluate("ngon_zero_area_figure_eight", FIGURE_EIGHT)

with open(OUT, "w") as handle:
    json.dump({"blender_version": "5.2.2", "cases": cases}, handle, indent=0)
print("CASES", len(cases))
