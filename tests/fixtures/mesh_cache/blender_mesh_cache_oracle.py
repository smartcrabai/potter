# Blender 5.2 oracle for MESH_CACHE (PC2/MDD) evaluation.
#
# Usage: Blender -b --factory-startup --python blender_mesh_cache_oracle.py -- CONFIG.json OUT.json
#
# CONFIG holds the base mesh, face loops and one entry per case: vertex-group membership,
# MESH_CACHE modifier properties, cache file path and the scene frames to evaluate. Every
# evaluated frame is written as the list of evaluated vertex positions, in case order.
import json
import math
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
with open(argv[0], "r", encoding="utf-8") as stream:
    config = json.load(stream)

scene = bpy.context.scene
scene.render.fps = int(config["fps"])
scene.render.fps_base = float(config["fps_base"])

result = {}
for case in config["cases"]:
    mesh = bpy.data.meshes.new(case["id"] + "Mesh")
    mesh.from_pydata(config["base"], [], config["faces"])
    mesh.update()
    obj = bpy.data.objects.new(case["id"], mesh)
    scene.collection.objects.link(obj)
    for group_name, members in case.get("groups", {}).items():
        group = obj.vertex_groups.new(name=group_name)
        for index, weight in members:
            group.add([index], float(weight), "REPLACE")
    modifier = obj.modifiers.new("Cache", "MESH_CACHE")
    modifier.filepath = case["file"]
    for name, value in case["modifier"].items():
        setattr(modifier, name, value)
    frames = []
    for frame in case["frames"]:
        whole = math.floor(frame)
        scene.frame_set(int(whole), subframe=float(frame - whole))
        evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        evaluated_mesh = evaluated.to_mesh()
        frames.append([list(vertex.co) for vertex in evaluated_mesh.vertices])
        evaluated.to_mesh_clear()
    result[case["id"]] = frames
    bpy.data.objects.remove(obj, do_unlink=True)

with open(argv[1], "w", encoding="utf-8") as stream:
    json.dump(result, stream)
