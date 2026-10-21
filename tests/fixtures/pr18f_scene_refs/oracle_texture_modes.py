import bpy, json, math
from mathutils import Vector
R=math.radians

def fresh(): bpy.ops.wm.read_factory_settings(use_empty=True)
def link(o): bpy.context.scene.collection.objects.link(o); return o
def setup(mode):
    fresh()
    bpy.ops.mesh.primitive_plane_add(size=2.0)
    subj=bpy.context.object; subj.name='Subj'
    subj.location=(1,2,3); subj.rotation_mode='XYZ'; subj.rotation_euler=(0,0,R(30)); subj.scale=(2,0.5,1.5)
    tex=bpy.data.textures.new('Blend', type='BLEND'); tex.progression='DIAGONAL'
    mod=subj.modifiers.new('D','DISPLACE'); mod.texture=tex; mod.direction='X'; mod.strength=1.0; mod.mid_level=0.5; mod.texture_coords=mode
    if mode=='OBJECT':
        mapping=bpy.data.objects.new('Map',None); link(mapping)
        mapping.location=(0.3,-0.2,0.5); mapping.rotation_mode='XYZ'; mapping.rotation_euler=(0,0,R(20)); mapping.scale=(1.5,1.0,0.8)
        mod.texture_coords_object=mapping
    bpy.context.view_layer.update(); ev=subj.evaluated_get(bpy.context.evaluated_depsgraph_get()); me=ev.to_mesh()
    out=[[round(float(c),9) for c in v.co] for v in me.vertices]; ev.to_mesh_clear()
    base=subj.data
    return {'out':out,'base':[[round(float(c),9) for c in v.co] for v in base.vertices]}
results={mode:setup(mode) for mode in ('LOCAL','GLOBAL','OBJECT')}
print('ORACLE_JSON_BEGIN'); print(json.dumps(results)); print('ORACLE_JSON_END')
