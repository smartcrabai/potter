import sys, bpy, bmesh
argv = sys.argv[sys.argv.index("--") + 1:]
mode = argv[0]
kind, sharpness = mode.split("_")
USE_CN = sharpness != "off"
bpy.ops.wm.read_factory_settings(use_empty=True)
me = bpy.data.meshes.new("cube")
bm = bmesh.new()
bmesh.ops.create_cube(bm, size=2.0)
bm.to_mesh(me)
bm.free()
ob = bpy.data.objects.new("body", me)
bpy.context.scene.collection.objects.link(ob)
bpy.context.view_layer.objects.active = ob
if sharpness == "sharp":
    me.edges[0].use_edge_sharp = True
if sharpness == "layer":
    me.normals_split_custom_set_from_vertices([(0.0, 0.0, 1.0)] * len(me.vertices))
print("PROBE input: edges=%d sharp=%s has_custom_normals=%s" % (
    len(me.edges), [e.index for e in me.edges if e.use_edge_sharp], me.has_custom_normals))
if kind == "subsurf":
    mod = ob.modifiers.new("Subsurf", "SUBSURF")
    mod.levels = 1
    mod.render_levels = 1
    mod.use_custom_normals = USE_CN
else:
    with bpy.context.temp_override(object=ob, active_object=ob, selected_objects=[ob]):
        bpy.ops.object.modifier_add(type="MULTIRES")
        bpy.ops.object.multires_subdivide(modifier="Multires", mode="CATMULL_CLARK")
    mod = ob.modifiers["Multires"]
    mod.levels = 1
    mod.render_levels = 1
    mod.sculpt_levels = 1
    mod.use_custom_normals = USE_CN
depsgraph = bpy.context.evaluated_depsgraph_get()
ev = ob.evaluated_get(depsgraph)
out = ev.to_mesh()
print("PROBE flag use_custom_normals=%s" % USE_CN)
print("PROBE output: verts=%d loops=%d faces=%d" % (len(out.vertices), len(out.loops), len(out.polygons)))
print("PROBE output has_custom_normals=%s" % out.has_custom_normals)
ln = [tuple(round(c, 4) for c in n.vector) for n in out.corner_normals]
print("PROBE corner_normals count=%d first=%s" % (len(ln), ln[:4]))
print("PROBE sharp_edges_out=%d" % sum(1 for e in out.edges if e.use_edge_sharp))
