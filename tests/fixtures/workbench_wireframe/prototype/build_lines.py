"""Loose edges at known angles in the XZ plane, for measuring the native line rasteriser (no faces)."""
import math
import sys
import bpy

out = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
mesh = bpy.data.meshes.new("Lines")
verts, edges = [], []
ANGLES = [0, 15, 30, 45, 60, 75, 90]
LENGTH = 3.0
for i, angle in enumerate(ANGLES):
    cz = -4.5 + i * 1.5
    half = 0.5 * LENGTH
    dx, dz = math.cos(math.radians(angle)) * half, math.sin(math.radians(angle)) * half
    a = len(verts); verts.append((-dx, 0.0, cz - dz)); verts.append((dx, 0.0, cz + dz))
    edges.append((a, a + 1))
mesh.from_pydata(verts, edges, [])
mesh.update()
obj = bpy.data.objects.new("Lines", mesh)
bpy.context.scene.collection.objects.link(obj)
bpy.ops.wm.save_as_mainfile(filepath=out)
print("LINES_SAVED", out, len(edges))
