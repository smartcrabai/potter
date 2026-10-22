# Blender 5.2.2 oracle: the Weighted Normal FACE_AREA modifier (weight 80, threshold 0.01) on two triangles that share
# vertex 0. Writes the evaluated corner normals of face 0, the corner at vertex 0 first.
# Run: blender --background --factory-startup --python weighted_normal_ranked_probe.py -- <out-dir>
import bpy, json, os, sys
root = os.path.realpath(sys.argv[sys.argv.index('--') + 1])
verts = [(0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 1, 0), (0, 0, 2)]
mesh = bpy.data.meshes.new('Ranked'); mesh.from_pydata(verts, [], [(0, 1, 2), (0, 3, 4)]); mesh.update()
ob = bpy.data.objects.new('Ranked', mesh); bpy.context.collection.objects.link(ob)
modifier = ob.modifiers.new('Weighted', 'WEIGHTED_NORMAL')
modifier.mode = 'FACE_AREA'; modifier.weight = 80; modifier.thresh = 0.01
bpy.context.view_layer.update()
ev = ob.evaluated_get(bpy.context.evaluated_depsgraph_get()); evaluated = ev.to_mesh()
normals = [list(normal.vector) for normal in evaluated.corner_normals]
ev.to_mesh_clear()
with open(os.path.join(root, 'weighted-normal-ranked.json'), 'w') as f: json.dump({'corner0': normals[0]}, f, indent=2)
