# Usage: /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup \
#        --python tests/fixtures/pr15_decimate/perf.py
#
# Times Blender 5.2.2's Decimate COLLAPSE (ratio 0.1) on the same 100k-triangle wavy triangulated grid
# that tests/modifier_decimate_perf.rs builds. Prints the best of three evaluations in seconds.
import math
import time

import bpy

N = 224  # 223 * 223 quads -> 99,458 triangles
verts = []
for j in range(N):
    for i in range(N):
        x = -1.0 + 2.0 * i / (N - 1)
        y = -1.0 + 2.0 * j / (N - 1)
        verts.append((x, y, 0.12 * math.cos(x * 2.0) * math.sin(y * 2.0)))
faces = []
for j in range(N - 1):
    for i in range(N - 1):
        a = j * N + i
        faces.append((a, a + 1, a + N + 1))
        faces.append((a, a + N + 1, a + N))

me = bpy.data.meshes.new("PerfGrid")
me.from_pydata(verts, [], faces)
me.update()
obj = bpy.data.objects.new("PerfGrid", me)
bpy.context.scene.collection.objects.link(obj)
mod = obj.modifiers.new("Decimate", "DECIMATE")
mod.decimate_type = "COLLAPSE"

best = None
faces_out = 0
for run in range(3):
    # A changed ratio invalidates the depsgraph cache, so each run re-evaluates the modifier.
    mod.ratio = 0.1 + 1.0e-6 * run
    start = time.perf_counter()
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    evaluated = obj.evaluated_get(dg)
    mesh = evaluated.to_mesh()
    elapsed = time.perf_counter() - start
    faces_out = len(mesh.polygons)
    evaluated.to_mesh_clear()
    best = elapsed if best is None else min(best, elapsed)
print(f"PERF blender triangles_in={len(faces)} faces_out={faces_out} best_seconds={best:.4f}")
