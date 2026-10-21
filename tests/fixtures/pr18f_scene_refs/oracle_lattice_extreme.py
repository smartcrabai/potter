import bpy, json, math
from mathutils import Vector
CORNERS=[(x,y,z) for x in (-0.5,0.5) for y in (-0.5,0.5) for z in (-0.5,0.5)]
bpy.ops.wm.read_factory_settings(use_empty=True)
m=bpy.data.meshes.new('M'); m.from_pydata(CORNERS,[],[]); m.update()
s=bpy.data.objects.new('S',m); bpy.context.scene.collection.objects.link(s); s.scale=(8,8,8)
lo=bpy.data.objects.new('L',bpy.data.lattices.new('LD')); bpy.context.scene.collection.objects.link(lo); lat=lo.data
lat.interpolation_type_u='KEY_BSPLINE'; lat.interpolation_type_v='KEY_BSPLINE'; lat.interpolation_type_w='KEY_BSPLINE'
for i,p in enumerate(lat.points): p.co_deform=Vector(p.co_deform)+Vector((0.15*math.sin(i+1),0.1*math.cos(2*i),-0.12*(i%2)))
mod=s.modifiers.new('L','LATTICE'); mod.object=lo
bpy.context.view_layer.update(); ev=s.evaluated_get(bpy.context.evaluated_depsgraph_get()); me=ev.to_mesh()
out=[[round(float(c),9) for c in v.co] for v in me.vertices]; ev.to_mesh_clear()
local=[[round(float(c),6) for c in (lo.matrix_world.inverted()@s.matrix_world@Vector(p))] for p in CORNERS]
print('ORACLE_JSON_BEGIN'); print(json.dumps({'out':out,'local':local})); print('ORACLE_JSON_END')
