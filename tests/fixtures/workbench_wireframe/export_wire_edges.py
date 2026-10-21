"""Exports the cube and sphere scene's wire data from `dump_geometry.py`'s JSON.

Writes `wire_edges_cube_sphere.json` (each mesh edge once, in mesh-edge order: world endpoints, the corner
normals of the first face that uses it, and `wd` from the cosine of its two adjacent face normals; 0 for a
boundary edge) and `wire_faces_cube_sphere.json` (each object's face polygons in world space, with the
object index the Rust oracle test gives it).

Run: python3 export_wire_edges.py <geometry.json> <this directory>
"""
import json
import os
import sys

WD_STEP = 254.0 / 255.0


def wd_of(face_normals):
    if len(face_normals) != 2:
        return 0.0
    cosine = sum(a * b for a, b in zip(face_normals[0], face_normals[1]))
    return min(max(200.0 * (cosine - 1.0) + 1.0, 0.0), 1.0) * WD_STEP


geometry = json.load(open(sys.argv[1]))
out_dir = sys.argv[2]
edges_out = []
for name, ob in geometry.items():
    adjacency = {}
    for fi, face in enumerate(ob["faces"]):
        for k in range(len(face)):
            a, b = face[k], face[(k + 1) % len(face)]
            adjacency.setdefault((min(a, b), max(a, b)), []).append(fi)
    for a, b in ob["edges"]:
        faces = adjacency.get((min(a, b), max(a, b)), [])
        if not faces:
            continue
        fi = faces[0]
        face = ob["faces"][fi]
        loop_a = ob["loop_start"][fi] + face.index(a)
        loop_b = ob["loop_start"][fi] + face.index(b)
        edges_out.append({
            "object": name,
            "start": ob["vertices"][a],
            "end": ob["vertices"][b],
            "normals": [ob["corner_normals"][loop_a], ob["corner_normals"][loop_b]],
            "wd": wd_of([ob["face_normals"][f] for f in faces]),
        })
json.dump(edges_out, open(os.path.join(out_dir, "wire_edges_cube_sphere.json"), "w"))
faces_out = {
    "objects": {name: [[ob["vertices"][k] for k in face] for face in ob["faces"]] for name, ob in geometry.items()},
    "indices": {"Cube": 1, "Sphere": 2},
}
json.dump(faces_out, open(os.path.join(out_dir, "wire_faces_cube_sphere.json"), "w"))
print("EXPORTED", len(edges_out), "edges")
