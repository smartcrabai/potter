"""Blender 5.2.2: origin map of a bevel (width 0.1, 2 segments, EDGES, limit NONE) on the box of size 2."""
import bpy
import json
import sys
from mathutils import Matrix, Vector

OUT = sys.argv[sys.argv.index("--") + 1]


def box_mesh(name):
    me = bpy.data.meshes.new(name)
    verts = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
             (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    me.from_pydata(verts, [], faces)
    me.update()
    return me


def evaluate(obj):
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    coords = [tuple(v.co) for v in me.vertices]
    ev.to_mesh_clear()
    return coords


origin_of = {}
counts = None
for k in range(8):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    obj = bpy.data.objects.new("bevel", box_mesh("bevel"))
    bpy.context.collection.objects.link(obj)
    empty = bpy.data.objects.new("hook_target", None)
    bpy.context.collection.objects.link(empty)
    empty.location = (0.0, 0.0, 1.0)
    bevel = obj.modifiers.new("bevel", "BEVEL")
    bevel.width = 0.1
    bevel.segments = 2
    bevel.affect = "EDGES"
    bevel.limit_method = "NONE"
    hook = obj.modifiers.new("hook", "HOOK")
    hook.object = empty
    hook.falloff_type = "NONE"
    hook.strength = 1.0
    hook.vertex_indices_set([k])
    hook.matrix_inverse = Matrix.Identity(4)
    hook.show_viewport = False
    base = evaluate(obj)
    hook.show_viewport = True
    hooked = evaluate(obj)
    counts = len(base)
    for o, (a, b) in enumerate(zip(base, hooked)):
        d = Vector(b) - Vector(a)
        if abs(d.z - 1.0) < 1e-4 and abs(d.x) < 1e-4 and abs(d.y) < 1e-4:
            origin_of.setdefault(o, []).append(k)

result = {
    "output_vertices": counts,
    "origin_of_output_vertex": {str(o): origin_of[o] for o in sorted(origin_of)},
    "unattributed": [o for o in range(counts) if o not in origin_of],
    "base_positions_of_attributed": {str(o): [round(c, 4) for c in base_pos] for o, base_pos in []},
}
json.dump(result, open(OUT, "w"), indent=1)
print("MAP", counts, json.dumps(result["origin_of_output_vertex"]))
print("UNATTRIBUTED", result["unattributed"])
