"""Blender 5.2.2 tessellation oracle: each face's loop triangles as Blender's Mesh.loop_triangles reports them.

Run: Blender -b --factory-startup --python generate_tessellation_oracle.py -- <out.json>

Cases (each is one face, positions stored as float32 values):
- quads: random near-degenerate (|dot| small relative to |a||b|, the fold predicate), plus non-planar.
- ngons: convex, concave, and polygons with collinear vertices.

Triangles are face-local corner indices (0..n-1), in Blender's loop-triangle order.
"""
import json
import math
import random
import sys

import bmesh  # noqa: F401  (kept for parity with other generators)
import bpy

OUT = sys.argv[sys.argv.index("--") + 1]
rng = random.Random(20261018)


def f32(value):
    import struct
    return struct.unpack("<f", struct.pack("<f", value))[0]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def quad_dot_f64(points):
    d12, d13, d14 = sub(points[1], points[0]), sub(points[2], points[0]), sub(points[3], points[0])
    return dot(cross(d12, d13), cross(d14, d13))


def quad_flip_f64(points):
    return quad_dot_f64(points) > 0.0

def f32_add(a, b):
    return f32(a + b)


def f32_sub(a, b):
    return f32(a - b)


def f32_mul(a, b):
    return f32(a * b)


def cross_f32(a, b):
    return (
        f32_sub(f32_mul(a[1], b[2]), f32_mul(a[2], b[1])),
        f32_sub(f32_mul(a[2], b[0]), f32_mul(a[0], b[2])),
        f32_sub(f32_mul(a[0], b[1]), f32_mul(a[1], b[0])),
    )


def dot_f32(a, b):
    total = f32_mul(a[0], b[0])
    total = f32_add(total, f32_mul(a[1], b[1]))
    return f32_add(total, f32_mul(a[2], b[2]))


def quad_dot_f32(points):
    d12 = tuple(f32_sub(points[1][i], points[0][i]) for i in range(3))
    d13 = tuple(f32_sub(points[2][i], points[0][i]) for i in range(3))
    d14 = tuple(f32_sub(points[3][i], points[0][i]) for i in range(3))
    return dot_f32(cross_f32(d12, d13), cross_f32(d14, d13))


def quad_flip_f32(points):
    return quad_dot_f32(points) > 0.0


def near_degenerate_quad():
    # The fold dot is a scalar triple product, linear in the fourth corner. Project that corner
    # onto the zero-dot plane, then perturb slightly along its normal. Store both numeric predicates.
    points = [tuple(f32(rng.uniform(-1, 1)) for _ in range(3)) for _ in range(3)]
    d12 = sub(points[1], points[0])
    d13 = sub(points[2], points[0])
    q = cross(d12, d13)
    plane_normal = cross(d13, q)
    plane_length_sq = dot(plane_normal, plane_normal)
    while plane_length_sq < 1e-12:
        points = [tuple(f32(rng.uniform(-1, 1)) for _ in range(3)) for _ in range(3)]
        d12 = sub(points[1], points[0])
        d13 = sub(points[2], points[0])
        q = cross(d12, d13)
        plane_normal = cross(d13, q)
        plane_length_sq = dot(plane_normal, plane_normal)
    candidate = tuple(rng.uniform(-1, 1) - points[0][i] for i in range(3))
    factor = dot(candidate, plane_normal) / plane_length_sq
    projected = tuple(candidate[i] - factor * plane_normal[i] for i in range(3))
    epsilon = rng.choice([-1.0, 1.0]) * 10.0 ** rng.uniform(-10.0, -7.0)
    p3 = tuple(f32(points[0][i] + projected[i] + epsilon * plane_normal[i]) for i in range(3))
    points.append(p3)
    return points, quad_dot_f32(points), quad_dot_f64(points)


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


def ngon_collinear(n):
    pts = ngon_convex(n)
    # Insert a collinear vertex: the midpoint of two consecutive corners.
    a, b = pts[0], pts[1]
    mid = tuple(f32((a[i] + b[i]) * 0.5) for i in range(3))
    return [pts[0], mid] + pts[1:]


cases = []
for _ in range(48):
    cases.append(("quad_near_zero", *near_degenerate_quad()))
for _ in range(32):
    points = random_quad()
    cases.append(("quad_nonplanar", points, quad_dot_f32(points), quad_dot_f64(points)))
for n in (5, 6, 7, 8, 10):
    for _ in range(4):
        cases.append(("ngon_convex_%d" % n, ngon_convex(n), None, None))
        cases.append(("ngon_concave_%d" % n, ngon_concave(n), None, None))
        cases.append(("ngon_collinear_%d" % n, ngon_collinear(n), None, None))

bpy.ops.wm.read_factory_settings(use_empty=True)
all_points = []
all_faces = []
offsets = []
for _, points, _, _ in cases:
    start = len(all_points)
    offsets.append(start)
    all_points.extend(points)
    all_faces.append(tuple(range(start, start + len(points))))
me = bpy.data.meshes.new("tessellation_oracle")
me.from_pydata(all_points, [], all_faces)
me.update()
me.calc_loop_triangles()
triangles_by_face = [[] for _ in cases]
for tri in me.loop_triangles:
    face_index = tri.polygon_index
    start = offsets[face_index]
    triangles_by_face[face_index].append([int(vertex - start) for vertex in tri.vertices])

out = []
for index, (kind, points, dot32, dot64) in enumerate(cases):
    entry = {
        "kind": kind,
        "positions": [list(p) for p in points],
        "triangles": triangles_by_face[index],
    }
    if dot32 is not None:
        entry["f32_predicate_dot"] = dot32
        entry["f64_predicate_dot"] = dot64
        entry["f32_flip"] = dot32 > 0.0
        entry["f64_flip"] = dot64 > 0.0
    out.append(entry)

disagreements = sum(
    (dot32 > 0.0) != (dot64 > 0.0)
    for _, _, dot32, dot64 in cases
    if dot32 is not None
)
json.dump({"blender_version": "5.2.2", "cases": out}, open(OUT, "w"), indent=0)
print("CASES", len(out), "DISAGREEMENTS", disagreements)
