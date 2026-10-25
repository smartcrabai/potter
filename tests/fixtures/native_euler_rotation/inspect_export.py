import bpy
import json
import os
import sys


if bpy.app.version != (5, 2, 2):
    raise RuntimeError("native Euler rotation oracle requires Blender 5.2.2")

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
probe = bpy.data.objects.get("NativeEulerProbe")
assert probe is not None


def matrix_values(matrix):
    return [float(matrix[row][column]) for column in range(4) for row in range(4)]


expected = {}
for frame in (1, 3):
    bpy.context.scene.frame_set(frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    expected[str(frame)] = matrix_values(probe.evaluated_get(depsgraph).matrix_world)

with open(os.path.join(root, "blender_expected.json"), "w", encoding="utf-8") as output:
    json.dump({"frames": expected}, output)
