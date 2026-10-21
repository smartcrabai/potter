import bpy, json, math
from mathutils import Vector
CORNERS=[(x,y,z) for x in (-0.5,0.5) for y in (-0.5,0.5) for z in (-0.5,0.5)]
def fresh(): bpy.ops.wm.read_factory_settings(use_empty=True)
def link(o): bpy.context.scene.collection.objects.link(o); return o
def run_case(interp):
    fresh(); m=bpy.data.meshes.new('M'); m.from_pydata(CORNERS,[],[]); m.update()
    subj=link(bpy.data.objects.new('S',m)); subj.location=(-0.8,0.0,0.0)
    lo=link(bpy.data.objects.new('L',bpy.data.lattices.new('LD'))); lat=lo.data
    lat.interpolation_type_u=interp; lat.interpolation_type_v=interp; lat.interpolation_type_w=interp
    for i,p in enumerate(lat.points):
        p.co_deform=Vector(p.co_deform)+Vector((0.15*math.sin(i+1),0.1*math.cos(2*i),-0.12*(i%2)))
    mod=subj.modifiers.new('L','LATTICE'); mod.object=lo
    bpy.context.view_layer.update(); ev=subj.evaluated_get(bpy.context.evaluated_depsgraph_get()); me=ev.to_mesh()
    out=[[round(float(c),9) for c in v.co] for v in me.vertices]; ev.to_mesh_clear()
    local=[[round(float(c),6) for c in (lo.matrix_world.inverted()@subj.matrix_world@Vector(p))] for p in CORNERS]
    return {'out':out,'local':local}
results={n:run_case(t) for n,t in [('linear','KEY_LINEAR'),('cardinal','KEY_CARDINAL'),('catmull_rom','KEY_CATMULL_ROM'),('bspline','KEY_BSPLINE')]}
print('ORACLE_JSON_BEGIN'); print(json.dumps(results)); print('ORACLE_JSON_END')
