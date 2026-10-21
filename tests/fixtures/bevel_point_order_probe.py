"""Per-point creation-order probe for the Blender 5.2.2 bevel oracle (no Blender needed).

Blender's `bev_create_ngon` interpolates each new loop, and its vertex data (vertex-group weights), from
one input face: `BM_loop_interp_from_face(bm, l, interp_f, do_vertex=true)`. The output mesh keeps BMesh
face order, so the last output face containing a point is the last loop write for that point. This probe
reads the committed oracle fixture and reports, per case:

* for each output corner, the input faces whose mean-value interpolation of the input UVs at the output
  corner's position reproduces the stored UV (empty: the value was merged or snapped, see `unmatched`);
* for each output point, whether the interpolation face of its LAST output face (`last`), or of its
  first, reproduces the stored vertex-group weights, and which input faces do.

Usage: python3 tests/fixtures/bevel_point_order_probe.py [case_name ...]
"""

import json
import math
import struct
import sys
from pathlib import Path

FIXTURE = Path(__file__).with_name("bevel_parity_oracle.json")
UV_TOLERANCE = 2e-5
WEIGHT_TOLERANCE = 2e-5


def f32(hex_bits):
    return struct.unpack("<f", bytes.fromhex(hex_bits))[0]


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def normalize(a):
    length = math.sqrt(dot(a, a))
    return tuple(x / length for x in a)


def mean_value_weights(point, polygon):
    """Floater mean-value weights of `point` in the plane of `polygon` (3D points, CCW)."""
    count = len(polygon)
    normal = (0.0, 0.0, 0.0)
    for k in range(count):
        a, b = polygon[k], polygon[(k + 1) % count]
        normal = tuple(n + c for n, c in zip(normal, cross(a, b)))
    normal = normalize(normal)
    anchor = polygon[0]
    offset = dot(sub(point, anchor), normal)
    flat = tuple(p - offset * n for p, n in zip(point, normal))
    rel = [sub(v, flat) for v in polygon]
    dist = [math.sqrt(dot(r, r)) for r in rel]
    for k, d in enumerate(dist):
        if d < 1e-12:
            return [1.0 if j == k else 0.0 for j in range(count)]
    angles = []
    for k in range(count):
        a, b = rel[k], rel[(k + 1) % count]
        angles.append(math.atan2(dot(normal, cross(a, b)), dot(a, b)))
    raw = []
    for k in range(count):
        before = angles[(k - 1) % count]
        after = angles[k]
        raw.append((math.tan(before / 2) + math.tan(after / 2)) / dist[k])
    total = sum(raw)
    return [w / total for w in raw]


def input_face_uvs(inputs):
    """Per input face: list of (x, y, z) positions and (u, v) UVs per corner, from the fixture input."""
    vertices = [tuple(f32(c) for c in v) for v in inputs["vertices"]]
    uv_flat = [tuple(f32(c) for c in uv) for uv in inputs["uv_layers"]["UVMap"]]
    groups = inputs["vertex_groups"]["weights"]
    faces = []
    loop = 0
    for face in inputs["faces"]:
        corners = []
        for vertex in face:
            corners.append(uv_flat[loop])
            loop += 1
        faces.append(
            {
                "vertices": [vertices[v] for v in face],
                "uv": corners,
                "groups": [[f32(w) for w in groups[v]] for v in face],
            }
        )
    return faces


def predict(face, point, channel):
    """Value of `channel` (uv or group index) of an input face at `point`, mean-value interpolated."""
    weights = mean_value_weights(point, face["vertices"])
    if channel == "uv":
        return tuple(sum(w * face["uv"][k][c] for k, w in enumerate(weights)) for c in range(2))
    return [sum(w * face["groups"][k][g] for k, w in enumerate(weights)) for g in range(len(face["groups"][0]))]


def probe(case, inputs):
    faces = input_face_uvs(inputs)
    output = case["output"]
    points = [tuple(f32(c) for c in v[0]) for v in output["vertices"]]
    weights = [[f32(w) for w in v[1]] for v in output["vertices"]]

    corner_matches = {}
    unmatched = 0
    corners = 0
    last_face_of = {}
    for face_index, (material, smooth, strength, ring) in enumerate(output["faces"]):
        for corner in ring:
            vid = corner[0]
            uv = (f32(corner[1]), f32(corner[2]))
            corners += 1
            last_face_of[vid] = face_index
            matched = [
                i
                for i, face in enumerate(faces)
                if all(
                    abs(p - q) < UV_TOLERANCE
                    for p, q in zip(predict(face, points[vid], "uv"), uv)
                )
            ]
            if not matched:
                unmatched += 1
            corner_matches.setdefault(vid, []).append((face_index, matched))

    first_face_of = {}
    for face_index, (_, _, _, ring) in enumerate(output["faces"]):
        for corner in ring:
            first_face_of.setdefault(corner[0], face_index)

    report = {"corners": corners, "unmatched_corners": unmatched, "points": {}}
    for vid, entries in corner_matches.items():
        group_matches = [
            i
            for i, face in enumerate(faces)
            if all(
                abs(p - q) < WEIGHT_TOLERANCE
                for p, q in zip(predict(face, points[vid], "group"), weights[vid])
            )
        ]
        last_uv = [m for f, m in entries if f == last_face_of[vid]][-1] if entries else []
        report["points"][vid] = {
            "group_matches": group_matches,
            "last_face": last_face_of[vid],
            "last_uv_matches": last_uv,
            "first_face": first_face_of[vid],
        }
    return report


def main(names):
    fixture = json.loads(FIXTURE.read_text())
    cases = {case["name"]: case for case in fixture["cases"]}
    selected = names or list(cases)
    for name in selected:
        case = cases[name]
        report = probe(case, fixture["inputs"][case["shape"]])
        points = report["points"]
        consistent_last = sum(
            1 for p in points.values() if p["last_face"] is not None and set(p["group_matches"]) and set(p["last_uv_matches"]) & set(p["group_matches"])
        )
        print(
            f"{name}: corners {report['corners']} unmatched_uv_corners {report['unmatched_corners']} "
            f"points {len(points)} group_consistent_with_last_uv_face {consistent_last}"
        )
        for vid, p in sorted(points.items()):
            if not (set(p["group_matches"]) and set(p["last_uv_matches"]) & set(p["group_matches"])):
                print(
                    f"  point {vid}: group_matches {p['group_matches']} last_face {p['last_face']} "
                    f"last_uv_matches {p['last_uv_matches']} first_face {p['first_face']}"
                )


if __name__ == "__main__":
    main(sys.argv[1:])
