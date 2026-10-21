"""Dumps the oracle scene's mesh data (world space): vertices, edges, faces, per-corner normals."""
import json
import sys

import bpy

out = sys.argv[sys.argv.index("--") + 1]
result = {}
for ob in bpy.context.scene.objects:
    if ob.type != "MESH":
        continue
    mesh = ob.data
    mw = ob.matrix_world
    verts = [list(mw @ v.co) for v in mesh.vertices]
    edges = [list(e.vertices) for e in mesh.edges]
    faces = [list(p.vertices) for p in mesh.polygons]
    face_normals = [list((mw.to_3x3() @ p.normal).normalized()) for p in mesh.polygons]
    corner_normals = []
    for corner in mesh.corner_normals:
        corner_normals.append(list((mw.to_3x3() @ corner.vector).normalized()))
    result[ob.name] = {
        "vertices": verts,
        "edges": edges,
        "faces": faces,
        "face_normals": face_normals,
        "loop_faces": [p.index for p in mesh.polygons for _ in p.loop_indices],
        "loop_vertices": [l.vertex_index for l in mesh.loops],
        "corner_normals": corner_normals,
        "loop_start": [p.loop_start for p in mesh.polygons],
    }
json.dump(result, open(out, "w"))
print("GEOM_DUMPED", list(result))
