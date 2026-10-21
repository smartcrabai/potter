"""Blender 5.2.2: decimate test configuration (6-vertex mesh, group Collapse on v1,v2, ratio 0.75).
Per-source attribution by listed Hook (each original vertex alone), and the Hook list [6, 0]."""
import bpy
import json
import sys
from mathutils import Matrix, Vector

OUT = sys.argv[sys.argv.index("--") + 1]
POSITIONS = [(0, 0, 1), (1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0), (0, 0, -1)]
FACES = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1), (5, 2, 1), (5, 3, 2), (5, 4, 3), (5, 1, 4)]


def build(name):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    me = bpy.data.meshes.new(name)
    me.from_pydata(POSITIONS, [(2, 1)], FACES)
    me.update()
    obj = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(obj)
    group = obj.vertex_groups.new(name="Collapse")
    group.add([1, 2], 1.0, "REPLACE")
    dec = obj.modifiers.new("decimate", "DECIMATE")
    dec.decimate_type = "COLLAPSE"
    dec.ratio = 0.75
    dec.vertex_group = "Collapse"
    dec.vertex_group_factor = 1.0
    dec.use_collapse_triangulate = False
    return obj


def evaluate(obj):
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    coords = [tuple(v.co) for v in me.vertices]
    ev.to_mesh_clear()
    return coords


results = {"output_vertices_without_hook": None, "origin_of_output_vertex": {}, "hook_6_0_moved": None}
obj = build("dec_base")
base = evaluate(obj)
results["output_vertices_without_hook"] = len(base)

origin_of = {}
for k in range(6):
    obj = build("dec_%d" % k)
    empty = bpy.data.objects.new("target_%d" % k, None)
    bpy.context.collection.objects.link(empty)
    empty.location = (0.0, 0.0, 1.0)
    hook = obj.modifiers.new("hook", "HOOK")
    hook.object = empty
    hook.falloff_type = "NONE"
    hook.strength = 1.0
    hook.vertex_indices_set([k])
    hook.matrix_inverse = Matrix.Identity(4)
    hook.show_viewport = False
    b = evaluate(obj)
    hook.show_viewport = True
    h = evaluate(obj)
    for o, (x, y) in enumerate(zip(b, h)):
        d = Vector(y) - Vector(x)
        if abs(d.z - 1.0) < 1e-4 and abs(d.x) < 1e-4 and abs(d.y) < 1e-4:
            origin_of.setdefault(o, []).append(k)
results["origin_of_output_vertex"] = {str(o): origin_of[o] for o in sorted(origin_of)}

obj = build("dec_hook_6_0")
empty = bpy.data.objects.new("target_hook", None)
bpy.context.collection.objects.link(empty)
empty.location = (1.0, 0.0, 0.0)
hook = obj.modifiers.new("hook", "HOOK")
hook.object = empty
hook.falloff_type = "NONE"
hook.strength = 1.0
hook.vertex_indices_set([6, 0])
hook.matrix_inverse = Matrix.Identity(4)
hook.show_viewport = False
b = evaluate(obj)
hook.show_viewport = True
h = evaluate(obj)
moved = []
for o, (x, y) in enumerate(zip(b, h)):
    d = Vector(y) - Vector(x)
    if abs(d.x - 1.0) < 1e-4 and abs(d.y) < 1e-4 and abs(d.z) < 1e-4:
        moved.append(o)
results["hook_6_0_moved"] = moved
results["hook_6_0_output_vertices"] = len(b)

json.dump(results, open(OUT, "w"), indent=1)
print("RESULT", json.dumps(results))
