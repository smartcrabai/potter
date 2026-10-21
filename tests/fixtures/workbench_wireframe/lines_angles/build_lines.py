"""Blender 5.2.2 oracle scene: seven loose edges at 0 to 90 degrees, for the line rasteriser's ink per angle.

Run: blender -b --factory-startup --python build_lines.py -- <out.blend> <edges.json>
The edges are written as world endpoints in the order of the scene, for the Rust oracle test. The capture of
this scene is made with `../capture_wire.py` (Standard view transform, factory outline setting).
"""
import json
import math
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
out_blend, out_edges = argv[0], argv[1]
bpy.ops.wm.read_factory_settings(use_empty=True)
mesh = bpy.data.meshes.new("Lines")
verts, edges = [], []
ANGLES = [0, 15, 30, 45, 60, 75, 90]
LENGTH = 3.0
for i, angle in enumerate(ANGLES):
    cz = -4.5 + i * 1.5
    half = 0.5 * LENGTH
    dx, dz = math.cos(math.radians(angle)) * half, math.sin(math.radians(angle)) * half
    a = len(verts)
    verts.append((-dx, 0.0, cz - dz))
    verts.append((dx, 0.0, cz + dz))
    edges.append((a, a + 1))
mesh.from_pydata(verts, edges, [])
mesh.update()
obj = bpy.data.objects.new("Lines", mesh)
bpy.context.scene.collection.objects.link(obj)
bpy.ops.wm.save_as_mainfile(filepath=out_blend)
dumped = [
    {"angle": angle, "start": list(mesh.vertices[a].co), "end": list(mesh.vertices[a + 1].co)}
    for angle, (a, _) in zip(ANGLES, edges)
]
with open(out_edges, "w") as handle:
    json.dump(dumped, handle, indent=1)
print("LINES_SAVED", out_blend, len(dumped))
