# Blender 5.2.2 oracle: a Data Transfer of BEVEL_WEIGHT_VERT with mix_factor 0.5. The layer is CD_PROP_FLOAT, which has
# no copyvalue, so the interpolated source replaces the destination (customdata.cc CustomData_data_mix_value): the
# result is the source value 1.0, not the 0.5 that a REPLACE blend would give.
# Run: blender --background --factory-startup --python bevel_weight_mix_probe.py -- <out-dir>
import bpy, json, os, sys
root = os.path.realpath(sys.argv[sys.argv.index('--') + 1])
verts = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
def obj(name, weight):
    mesh = bpy.data.meshes.new(name + 'Mesh'); mesh.from_pydata(verts, [], [(0, 1, 2)]); mesh.update()
    layer = mesh.attributes.new(name='bevel_weight_vert', type='FLOAT', domain='POINT')
    for i in range(len(verts)):
        layer.data[i].value = weight
    ob = bpy.data.objects.new(name, mesh); bpy.context.collection.objects.link(ob); return ob
src = obj('Source', 1.0)
dst = obj('Dest', 0.0)
mod = dst.modifiers.new('Transfer', 'DATA_TRANSFER'); mod.object = src
mod.use_object_transform = True; mod.use_vert_data = True; mod.data_types_verts = {'BEVEL_WEIGHT_VERT'}
mod.vert_mapping = 'NEAREST'; mod.mix_mode = 'REPLACE'; mod.mix_factor = 0.5
bpy.context.view_layer.update()
ev = dst.evaluated_get(bpy.context.evaluated_depsgraph_get()); mesh = ev.to_mesh()
values = [mesh.attributes['bevel_weight_vert'].data[i].value for i in range(len(verts))]
ev.to_mesh_clear()
with open(os.path.join(root, 'bevel-weight-mix.json'), 'w') as f: json.dump({'bevel_weight_vert': values}, f, indent=2)
