# Blender 5.2.2 oracle: a vertex-group Data Transfer with POLYINTERP_VNORPROJ from a source rotated about Z, with
# use_object_transform on. The vertex normal of the destination points along +X, so the projection ray must follow
# the destination-to-source matrix (MOD_datatransfer.cc `local2target`), not the source-to-destination one.
# Each rotation (+30 and -30 degrees) is written, so the test can check the two results differ.
# Run: blender --background --factory-startup --python rotated_normal_probe.py -- <out-dir>
import bpy, json, math, os, sys
from mathutils import Matrix, Vector
root = os.path.realpath(sys.argv[sys.argv.index('--') + 1])
def obj(name, verts, faces):
    mesh = bpy.data.meshes.new(name + 'Mesh'); mesh.from_pydata(verts, [], faces); mesh.update()
    ob = bpy.data.objects.new(name, mesh); bpy.context.collection.objects.link(ob); return ob
def group(ob, name, weights):
    vg = ob.vertex_groups.new(name=name)
    for i, weight in enumerate(weights):
        vg.add([i], float(weight), 'REPLACE')
def read(ob):
    ev = ob.evaluated_get(bpy.context.evaluated_depsgraph_get()); mesh = ev.to_mesh()
    vg = ev.vertex_groups.get('A'); values = []
    for v in mesh.vertices:
        value = 0.0
        if vg:
            for a in v.groups:
                if a.group == vg.index: value = a.weight; break
        values.append(value)
    ev.to_mesh_clear(); return values
result = {}
for key, angle in (('plus_30', 30.0), ('minus_30', -30.0)):
    src = obj('Source' + key, [(0, -1, -1), (0, 1, -1), (0, 1, 1), (0, -1, 1)], [(0, 1, 2, 3)])
    group(src, 'A', [0.0, 1.0, 1.0, 0.0])
    src.matrix_world = Matrix.Translation(Vector((1.0, 0.0, 0.0))) @ Matrix.Rotation(math.radians(angle), 4, 'Z')
    dst = obj('Dest' + key, [(0, 0.05, 0.05), (0, 0.3, 0.05), (0, 0.05, 0.25)], [(0, 1, 2)])
    group(dst, 'A', [0.2, 0.3, 0.4])
    mod = dst.modifiers.new('Transfer', 'DATA_TRANSFER'); mod.object = src
    mod.use_object_transform = True; mod.use_vert_data = True; mod.data_types_verts = {'VGROUP_WEIGHTS'}
    mod.vert_mapping = 'POLYINTERP_VNORPROJ'; mod.mix_mode = 'REPLACE'; mod.mix_factor = 1.0
    mod.use_max_distance = True; mod.max_distance = 10.0; mod.ray_radius = 0.01
    bpy.context.view_layer.update(); result[key] = read(dst)
with open(os.path.join(root, 'rotated-normal.json'), 'w') as f: json.dump(result, f, indent=2)
