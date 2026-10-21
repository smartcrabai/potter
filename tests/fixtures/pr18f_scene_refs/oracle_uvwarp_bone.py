import bpy, json, math
R=math.radians
UVS=[(0.1,0.2),(0.8,0.1),(0.7,0.9),(0.2,0.8)]

def fresh(): bpy.ops.wm.read_factory_settings(use_empty=True)
def link(o): bpy.context.scene.collection.objects.link(o); return o

def setup(bone_from):
    fresh()
    me=bpy.data.meshes.new('SubjMesh')
    me.from_pydata([(-1,-1,0),(1,-1,0),(1,1,0),(-1,1,0)],[],[(0,1,2,3)])
    me.update()
    uv=me.uv_layers.new(name='UVMap')
    for loop, value in zip(uv.data, UVS): loop.uv=value
    subj=link(bpy.data.objects.new('Subj',me))
    subj.location=(0.25,-0.4,0.15); subj.rotation_mode='XYZ'; subj.rotation_euler=(R(8),R(-4),R(17)); subj.scale=(1.2,0.8,1.1)

    arm_data=bpy.data.armatures.new('RigData')
    arm=link(bpy.data.objects.new('Rig',arm_data))
    arm.location=(0.4,-0.3,0.2); arm.rotation_mode='XYZ'; arm.rotation_euler=(R(6),R(-9),R(25)); arm.scale=(1.4,0.8,1.2)
    bpy.context.view_layer.objects.active=arm; arm.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    eb=arm_data.edit_bones.new('Bone')
    eb.head=(0.0,0.0,0.0); eb.tail=(0.0,1.0,0.0)
    bpy.ops.object.mode_set(mode='OBJECT')
    pb=arm.pose.bones['Bone']
    pb.rotation_mode='XYZ'; pb.rotation_euler=(R(7),R(-13),R(52)); pb.location=(0.35,-0.12,0.18); pb.scale=(1.15,0.75,1.05)

    dest=link(bpy.data.objects.new('Dest',None))
    dest.location=(-0.2,0.55,-0.1); dest.rotation_mode='XYZ'; dest.rotation_euler=(R(-12),R(5),R(-28)); dest.scale=(0.9,1.3,0.85)

    mod=subj.modifiers.new('UVWarp','UV_WARP')
    mod.object_from=arm; mod.object_to=dest
    mod.bone_from=bone_from; mod.bone_to=''
    mod.uv_layer='UVMap'; mod.axis_u='X'; mod.axis_v='Y'
    mod.center=(0.35,0.6); mod.offset=(0.12,-0.08); mod.scale=(1.2,0.75); mod.rotation=0.27
    bpy.context.view_layer.update()
    ev=subj.evaluated_get(bpy.context.evaluated_depsgraph_get()); out=ev.to_mesh()
    values=[[round(float(item.uv.x),9),round(float(item.uv.y),9)] for item in out.uv_layers['UVMap'].data]
    ev.to_mesh_clear()
    return {'uv':values, 'armature_world':[round(float(v),9) for col in arm.matrix_world.col for v in col], 'bone_pose':[round(float(v),9) for col in pb.matrix for v in col]}

results={'valid_bone':setup('Bone'),'empty_bone':setup(''),'missing_bone':setup('NoSuchBone')}
print('ORACLE_JSON_BEGIN'); print(json.dumps(results)); print('ORACLE_JSON_END')
