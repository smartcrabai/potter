"""Blender 5.2.2: subdivision carrier attribution by listed Hook, per source vertex.
Config A (the F test): triangle (0,0,0),(1,0,0),(0,1,0), SUBSURF SIMPLE levels 1.
Config B: box of size 2, SUBSURF CATMULL_CLARK levels 1."""
import bpy
import json
import sys
from mathutils import Matrix, Vector

OUT = sys.argv[sys.argv.index("--") + 1]


def build(config, name):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    me = bpy.data.meshes.new(name)
    if config == "A":
        me.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
    else:
        verts = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                 (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
        faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        me.from_pydata(verts, [], faces)
    me.update()
    obj = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(obj)
    sub = obj.modifiers.new("subdivision", "SUBSURF")
    sub.levels = 1
    sub.render_levels = 1
    if config == "A":
        sub.subdivision_type = "SIMPLE"
    else:
        sub.subdivision_type = "CATMULL_CLARK"
    return obj


def evaluate(obj):
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    coords = [tuple(v.co) for v in me.vertices]
    ev.to_mesh_clear()
    return coords


results = {}
for config in ["A", "B"]:
    base_obj = build(config, "base_" + config)
    base = evaluate(base_obj)
    source_count = 3 if config == "A" else 8
    origin_of = {}
    for k in range(source_count):
        obj = build(config, "sub_%s_%d" % (config, k))
        empty = bpy.data.objects.new("t_%s_%d" % (config, k), None)
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
    results[config] = {
        "output_vertices": len(base),
        "origin_of_output_vertex": {str(o): origin_of[o] for o in sorted(origin_of)},
    }
    print("SUB", config, json.dumps(results[config]))

json.dump(results, open(OUT, "w"), indent=1)
print("WROTE", OUT)
