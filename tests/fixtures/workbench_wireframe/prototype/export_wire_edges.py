"""Exports the oracle scene's mesh edges as Potter's wire-edge records, in the prototype's draw order.

Each mesh edge once, in mesh edge order: its world endpoints, the corner normals of the first face (in
face order) that uses it, and `wd` from the cosine of its two adjacent face normals (0 for a boundary edge).
The Rust test in `render/wireframe.rs` reads the output and rasterises it against the oracle capture.
"""
import json
import math
import sys

geom = json.load(open(sys.argv[1]))
out = []
WD_STEP = 254.0 / 255.0


def wd_of(face_normals):
    if len(face_normals) != 2:
        return 0.0
    cos = sum(a * b for a, b in zip(face_normals[0], face_normals[1]))
    fac = min(max(200.0 * (cos - 1.0) + 1.0, 0.0), 1.0)
    return fac * WD_STEP


for name, ob in geom.items():
    adjacency = {}
    for fi, face in enumerate(ob["faces"]):
        start = ob["loop_start"][fi]
        for k in range(len(face)):
            a, b = face[k], face[(k + 1) % len(face)]
            key = (min(a, b), max(a, b))
            adjacency.setdefault(key, []).append((fi, start + k, a, b))
    for a, b in ob["edges"]:
        key = (min(a, b), max(a, b))
        faces = adjacency.get(key, [])
        fi, corner, fa, fb = faces[0]
        # corner normals of the first face at each endpoint
        na = ob["corner_normals"][corner if fa == a else corner + 1 if False else corner]
        # corner index of endpoint a and b within that face's loop
        loop_a = ob["loop_start"][fi] + ob["faces"][fi].index(a)
        loop_b = ob["loop_start"][fi] + ob["faces"][fi].index(b)
        out.append({
            "object": name,
            "start": ob["vertices"][a],
            "end": ob["vertices"][b],
            "normals": [ob["corner_normals"][loop_a], ob["corner_normals"][loop_b]],
            "wd": wd_of([ob["face_normals"][f[0]] for f in faces]),
        })
json.dump(out, open(sys.argv[2], "w"))
print("EXPORTED", len(out))
