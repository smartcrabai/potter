import bpy, json, math
from mathutils import Vector
CORNERS = [(x, y, z) for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)]

def fresh(): bpy.ops.wm.read_factory_settings(use_empty=True)
def link(o): bpy.context.scene.collection.objects.link(o); return o
def run_case(interp):
    fresh()
    m=bpy.data.meshes.new('SubjM'); m.from_pydata(CORNERS, [], []); m.update()
    subj=link(bpy.data.objects.new('Subj', m)); subj.scale=(4.0,4.0,4.0)
    lat_obj=link(bpy.data.objects.new('Lat', bpy.data.lattices.new('LatD')))
    lat=lat_obj.data
    lat.interpolation_type_u=interp; lat.interpolation_type_v=interp; lat.interpolation_type_w=interp
    for i,p in enumerate(lat.points):
        base=Vector(p.co_deform)
        p.co_deform=base+Vector((0.15*math.sin(i+1),0.1*math.cos(2*i),-0.12*(i%2)))
    mod=subj.modifiers.new('L','LATTICE'); mod.object=lat_obj; mod.strength=1.0
    bpy.context.view_layer.update()
    ev=subj.evaluated_get(bpy.context.evaluated_depsgraph_get()); me=ev.to_mesh()
    out=[[round(float(c),9) for c in v.co] for v in me.vertices]; ev.to_mesh_clear()
    local=[[round(float(c),6) for c in (lat_obj.matrix_world.inverted()@subj.matrix_world@Vector(p))] for p in CORNERS]
    return {'out':out,'local':local,'points':[[round(float(c),9) for c in p.co_deform] for p in lat.points]}
results={name:run_case(key) for name,key in [('linear','KEY_LINEAR'),('cardinal','KEY_CARDINAL'),('catmull_rom','KEY_CATMULL_ROM'),('bspline','KEY_BSPLINE')]}
print('ORACLE_JSON_BEGIN'); print(json.dumps(results)); print('ORACLE_JSON_END')
