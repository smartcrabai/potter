#![expect(
    clippy::unwrap_used,
    reason = "Blender round-trip fixtures use fixed valid test data"
)]

use std::{error::Error, fs};

use serde_json::{Value, json};
use tempfile::tempdir;
#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/blender_script_guarded.rs"]
mod blender_script_guarded;
#[path = "common/pot_json_guarded.rs"]
mod pot_json_guarded;
#[path = "common/process.rs"]
mod process;

use blender_checked::blender_executable as blender;
use blender_script_guarded::run_blender_script;
use pot_json_guarded::pot_json;
use process::run_guarded;

const FIXTURE: &str = r"
import bpy, openvdb
import json
import os
import sys

root = os.path.realpath(sys.argv[sys.argv.index('--') + 1])
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 4
scene.frame_set(1)
scene.use_gravity = False

def make_mesh(name):
    mesh = bpy.data.meshes.new(name + 'Mesh')
    mesh.from_pydata([(-1,0,0), (1,0,0), (1,0,1), (-1,0,1)], [], [(0,1,2,3)])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    return obj

def make_cube(name):
    mesh = bpy.data.meshes.new(name + 'Mesh')
    mesh.from_pydata(
        [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
         (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)],
        [],
        [(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),
         (2,3,7,6),(3,0,4,7)],
    )
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    return obj

def cache_values(block):
    cache = getattr(block, 'point_cache', None)
    if cache is None:
        return None
    return {'frame_start': int(cache.frame_start), 'frame_end': int(cache.frame_end),
            'is_baked': bool(cache.is_baked)}

def fields(block, names):
    def plain(value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if hasattr(value, 'to_list'):
            return list(value.to_list())
        if hasattr(value, 'name_full'):
            return value.name_full
        try:
            return [plain(item) for item in value]
        except Exception:
            return repr(value)
    return {name: plain(getattr(block, name)) for name in names}

def geometry(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    if obj.type != 'MESH':
        return None
    mesh = evaluated.to_mesh()
    points = [list(vertex.co) for vertex in mesh.vertices]
    evaluated.to_mesh_clear()
    return points


expected = {'settings': {}, 'geometry': {}, 'modifiers': {},
            'frame_start': scene.frame_start, 'frame_end': scene.frame_end}

cloth = make_mesh('Cloth')
cloth_mod = cloth.modifiers.new('ClothSettingsFixture', 'CLOTH')
cloth_settings = cloth_mod.settings
cloth_fields = ['goal_default', 'goal_spring', 'goal_friction', 'mass', 'air_damping',
                'tension_stiffness', 'compression_stiffness', 'shear_stiffness',
                'bending_stiffness', 'use_sewing_springs', 'use_dynamic_mesh']
for key, value in {'goal_default':0.27, 'goal_spring':3.25, 'goal_friction':1.5,
                   'mass':0.73, 'air_damping':0.82, 'tension_stiffness':17.0,
                   'compression_stiffness':13.0, 'shear_stiffness':6.5,
                   'bending_stiffness':0.31, 'use_sewing_springs':True,
                   'use_dynamic_mesh':True}.items():
    setattr(cloth_settings, key, value)
cloth_collision = cloth_mod.collision_settings
cloth_collision_fields = ['use_collision', 'distance_min', 'friction', 'damping',
                          'collision_quality', 'use_self_collision', 'self_distance_min',
                          'self_friction']
for key, value in {'use_collision':True, 'distance_min':0.017, 'friction':3.4,
                   'damping':0.22, 'collision_quality':4, 'use_self_collision':True,
                   'self_distance_min':0.031, 'self_friction':4.2}.items():
    setattr(cloth_collision, key, value)
cloth_mod.point_cache.frame_start = 2
cloth_mod.point_cache.frame_end = 4
expected['settings']['cloth'] = {
    'settings':fields(cloth_settings, cloth_fields),
    'collision_settings':fields(cloth_collision, cloth_collision_fields),
    'point_cache':cache_values(cloth_mod),
}
expected['geometry']['Cloth'] = geometry(cloth)

soft = make_mesh('SoftBody')
soft_mod = soft.modifiers.new('SoftBodySettingsFixture', 'SOFT_BODY')
soft_settings = soft_mod.settings
soft_fields = ['mass', 'friction', 'speed', 'goal_default', 'goal_spring', 'goal_friction',
               'pull', 'push', 'damping', 'spring_length', 'aero', 'bend', 'use_goal',
               'use_edges', 'use_stiff_quads', 'use_edge_collision', 'use_face_collision']
for key, value in {'mass':1.7, 'friction':0.73, 'speed':0.84, 'goal_default':0.24,
                   'goal_spring':2.8, 'goal_friction':0.36, 'pull':0.62, 'push':0.41,
                   'damping':0.29, 'spring_length':2, 'aero':2, 'bend':0.21,
                   'use_goal':False, 'use_edges':True, 'use_stiff_quads':True,
                   'use_edge_collision':True, 'use_face_collision':True}.items():
    setattr(soft_settings, key, value)
soft_mod.point_cache.frame_start = 2
soft_mod.point_cache.frame_end = 4
expected['settings']['soft_body'] = {
    'settings':fields(soft_settings, soft_fields), 'point_cache':cache_values(soft_mod)}
expected['geometry']['SoftBody'] = geometry(soft)

collision = make_mesh('Collision')
collision_mod = collision.modifiers.new('CollisionSettingsFixture', 'COLLISION')
collision_settings = collision_mod.settings
collision_fields = ['use', 'damping_factor', 'damping_random', 'friction_factor',
                    'friction_random', 'permeability', 'use_particle_kill', 'thickness_inner',
                    'thickness_outer', 'cloth_friction', 'absorption']
for key, value in {'use':True, 'damping_factor':0.38, 'damping_random':0.14,
                   'friction_factor':0.47, 'friction_random':0.12, 'permeability':0.19,
                   'use_particle_kill':True, 'thickness_inner':0.13,
                   'thickness_outer':0.27, 'cloth_friction':6.25, 'absorption':0.22}.items():
    setattr(collision_settings, key, value)
expected['settings']['collision'] = {'settings':fields(collision_settings, collision_fields)}
expected['geometry']['Collision'] = geometry(collision)

def activate(obj):
    for candidate in bpy.context.selected_objects:
        candidate.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

dp_brush = make_mesh('DynamicPaintBrush')
brush_mod = dp_brush.modifiers.new('BrushSettingsFixture', 'DYNAMIC_PAINT')
activate(dp_brush)
bpy.ops.dpaint.type_toggle(type='BRUSH')
brush = brush_mod.brush_settings
brush_fields = ['paint_color', 'paint_alpha', 'use_absolute_alpha', 'paint_wetness',
                'use_paint_erase', 'wave_type', 'wave_factor', 'wave_clamp',
                'use_smudge', 'smudge_strength', 'velocity_max', 'use_velocity_alpha']
for key, value in {'paint_color':(0.19,0.42,0.71), 'paint_alpha':0.68,
                   'use_absolute_alpha':True, 'paint_wetness':0.37,
                   'use_paint_erase':True, 'wave_type':'REFLECT', 'wave_factor':0.63,
                   'wave_clamp':False, 'use_smudge':True, 'smudge_strength':0.29,
                   'velocity_max':0.46, 'use_velocity_alpha':True}.items():
    setattr(brush, key, value)
expected['settings']['dynamic_paint_brush'] = {
    'settings':fields(brush, brush_fields), 'canvas_surfaces':[], 'role':'brush'}
expected['geometry']['DynamicPaintBrush'] = geometry(dp_brush)

dp_canvas = make_mesh('DynamicPaintCanvas')
canvas_mod = dp_canvas.modifiers.new('CanvasSettingsFixture', 'DYNAMIC_PAINT')
activate(dp_canvas)
bpy.ops.dpaint.type_toggle(type='CANVAS')
canvas = canvas_mod.canvas_settings
if len(canvas.canvas_surfaces) == 0:
    bpy.ops.dpaint.surface_slot_add()
surface = canvas.canvas_surfaces[0]
surface_fields = ['surface_format', 'surface_type', 'is_active', 'use_dissolve',
                  'dissolve_speed', 'use_drying', 'dry_speed', 'frame_start', 'frame_end']
for key, value in {'surface_format':'VERTEX', 'surface_type':'PAINT', 'is_active':True,
                   'use_dissolve':True, 'dissolve_speed':3, 'use_drying':True,
                   'dry_speed':2, 'frame_start':2, 'frame_end':4}.items():
    setattr(surface, key, value)
expected['settings']['dynamic_paint_canvas'] = {
    'settings':{}, 'canvas_surfaces':[fields(surface, surface_fields)], 'role':'canvas'}
expected['geometry']['DynamicPaintCanvas'] = geometry(dp_canvas)

fluid_domain = make_cube('FluidDomain')
domain_mod = fluid_domain.modifiers.new('FluidDomainSettingsFixture', 'FLUID')
domain_mod.fluid_type = 'DOMAIN'
domain_mod.show_viewport = False
domain_mod.show_render = False
domain = domain_mod.domain_settings
domain_fields = ['domain_type', 'resolution_max', 'use_adaptive_domain', 'cache_frame_start',
                 'cache_frame_end', 'cache_frame_offset', 'cache_data_format',
                 'cache_mesh_format', 'cache_directory', 'time_scale']
domain.cache_directory = '//fluid_cache'
for key, value in {'domain_type':'LIQUID', 'resolution_max':19, 'use_adaptive_domain':True,
                   'cache_frame_start':2, 'cache_frame_end':4, 'cache_frame_offset':1,
                   'cache_data_format':'UNI', 'cache_mesh_format':'OBJECT',
                   'time_scale':0.85}.items():
    setattr(domain, key, value)
expected['settings']['fluid_domain'] = {
    'settings':fields(domain, domain_fields), 'role':'DOMAIN',
    'enabled':bool(domain_mod.show_viewport and domain_mod.show_render)}
expected['geometry']['FluidDomain'] = geometry(fluid_domain)

fluid_flow = make_mesh('FluidFlow')
flow_mod = fluid_flow.modifiers.new('FluidFlowSettingsFixture', 'FLUID')
flow_mod.show_viewport = False
flow_mod.show_render = False
flow_mod.fluid_type = 'FLOW'
flow = flow_mod.flow_settings
flow_fields = ['density', 'fuel_amount', 'temperature', 'flow_type', 'flow_behavior',
               'flow_source', 'use_absolute', 'use_initial_velocity', 'velocity_coord',
               'surface_distance']
for key, value in {'density':2.4, 'fuel_amount':1.3, 'temperature':1.2,
                   'flow_type':'LIQUID', 'flow_behavior':'INFLOW', 'flow_source':'MESH',
                   'use_absolute':True, 'use_initial_velocity':True,
                   'velocity_coord':(0.3,-0.2,0.5), 'surface_distance':0.16}.items():
    setattr(flow, key, value)
expected['settings']['fluid_flow'] = {
    'settings':fields(flow, flow_fields), 'role':'FLOW',
    'enabled':bool(flow_mod.show_viewport and flow_mod.show_render)}
expected['geometry']['FluidFlow'] = geometry(fluid_flow)

fluid_effector = make_mesh('FluidEffector')
effector_mod = fluid_effector.modifiers.new('FluidEffectorSettingsFixture', 'FLUID')
effector_mod.show_viewport = False
effector_mod.show_render = False
effector_mod.fluid_type = 'EFFECTOR'
effector = effector_mod.effector_settings
effector_fields = ['effector_type', 'surface_distance', 'use_plane_init',
                   'velocity_factor', 'guide_mode', 'use_effector', 'subframes']
for key, value in {'effector_type':'GUIDE', 'surface_distance':0.23,
                   'use_plane_init':True, 'velocity_factor':0.72,
                   'guide_mode':'OVERRIDE', 'use_effector':True, 'subframes':2}.items():
    setattr(effector, key, value)
expected['settings']['fluid_effector'] = {
    'settings':fields(effector, effector_fields), 'role':'EFFECTOR',
    'enabled':bool(effector_mod.show_viewport and effector_mod.show_render)}
expected['geometry']['FluidEffector'] = geometry(fluid_effector)

emitter = make_mesh('ParticleEmitter')
activate(emitter)
bpy.ops.object.particle_system_add()
particle_modifier = next(modifier for modifier in emitter.modifiers
                         if modifier.type == 'PARTICLE_SYSTEM')
particle_modifier.name = 'ParticleSystemFixture'
particle_system = emitter.particle_systems[0]
particle_settings = particle_system.settings
particle_fields = ['count', 'frame_start', 'frame_end', 'lifetime', 'lifetime_random',
                   'emit_from', 'physics_type', 'normal_factor', 'factor_random',
                   'particle_size', 'size_random', 'render_type', 'child_type']
for key, value in {'count':1, 'frame_start':2.0, 'frame_end':2.0, 'lifetime':7.0,
                   'lifetime_random':0.15, 'emit_from':'VERT', 'physics_type':'NEWTON',
                   'normal_factor':0.0, 'factor_random':0.0, 'particle_size':0.63,
                   'size_random':0.17, 'render_type':'HALO', 'child_type':'NONE'}.items():
    setattr(particle_settings, key, value)
particle_system.seed = 29
particle_system.point_cache.frame_start = 1
particle_system.point_cache.frame_end = 4
expected['settings']['particle_system'] = {
    'settings':fields(particle_settings, particle_fields),
    'point_cache':cache_values(particle_system), 'seed':int(particle_system.seed)}
expected['geometry']['ParticleEmitter'] = geometry(emitter)

instance = make_mesh('ParticleInstance')
instance_mod = instance.modifiers.new('ParticleInstanceSettingsFixture', 'PARTICLE_INSTANCE')
instance_mod.object = emitter
instance_mod.particle_system_index = 1
instance_mod.use_normal = True
instance_mod.use_children = False
instance_mod.use_size = True
instance_mod.show_alive = True
instance_mod.show_dead = False
instance_mod.show_unborn = False
instance_mod.position = 0.35
instance_mod.random_position = 0.12
instance_mod.axis = 'Z'
instance_mod.space = 'WORLD'
expected['modifiers']['ParticleInstance'] = {
    'type':instance_mod.type,
    **fields(instance_mod, ['object','particle_system_index','use_normal','use_children',
                            'use_size','show_alive','show_dead','show_unborn','position',
                            'random_position','axis','space'])}
expected['geometry']['ParticleInstance'] = geometry(instance)

explode = make_mesh('Explode')
activate(explode)
bpy.ops.object.particle_system_add()
explode_system = explode.particle_systems[0]
explode_system.settings.count = 1
explode_system.settings.frame_start = 2
explode_system.settings.frame_end = 2
explode_system.settings.lifetime = 7
explode_system.settings.emit_from = 'VERT'
explode_system.settings.physics_type = 'NO'
explode_system.settings.normal_factor = 0.0
explode_mod = explode.modifiers.new('ExplodeSettingsFixture', 'EXPLODE')
explode_mod.use_edge_cut = True
explode_mod.show_alive = True
explode_mod.show_dead = False
explode_mod.show_unborn = False
explode_mod.use_size = False
explode_mod.protect = 0.13
expected['modifiers']['Explode'] = {
    'type':explode_mod.type,
    **fields(explode_mod, ['use_edge_cut','show_alive','show_dead','show_unborn',
                           'use_size','protect'])}
expected['geometry']['Explode'] = geometry(explode)

grid = openvdb.FloatGrid()
grid.name = 'density'
accessor = grid.getAccessor()
for z in range(3):
    for y in range(3):
        for x in range(3):
            accessor.setValueOn((x,y,z), 0.2 + 0.1*x + 0.07*y + 0.04*z)
vdb_path = os.path.join(root, 'displaced.vdb')
openvdb.write(vdb_path, grids=[grid])
bpy.ops.object.volume_import(filepath=vdb_path)
volume = bpy.context.object
volume.name = 'VolumeDisplace'
volume.data.name = 'DisplacedVolumeData'
displace_mod = volume.modifiers.new('VolumeDisplaceSettingsFixture', 'VOLUME_DISPLACE')
displace_fields = ['strength', 'texture_map_mode', 'texture_mid_level',
                   'texture_sample_radius', 'texture_map_object']
displace_mod.strength = 0.74
displace_mod.texture_map_mode = 'LOCAL'
displace_mod.texture_mid_level = (0.3,0.45,0.6)
displace_mod.texture_sample_radius = 0.27
expected['settings']['volume_displace'] = {
    'modifier':fields(displace_mod, displace_fields)}

# Bake a short legacy particle cache so export must preserve both its range and baked state.
activate(emitter)
scene.frame_set(1)
bpy.context.view_layer.update()
with bpy.context.temp_override(object=emitter, active_object=emitter,
                               point_cache=particle_system.point_cache):
    bpy.ops.ptcache.bake(bake=True)
expected['settings']['particle_system']['point_cache'] = cache_values(particle_system)
scene.frame_set(1)
bpy.context.view_layer.update()
for name in expected['geometry']:
    expected['geometry'][name] = geometry(bpy.data.objects[name])
with open(os.path.join(root, 'expected.json'), 'w', encoding='utf-8') as handle:
    json.dump(expected, handle)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, 'source.blend'))
";

const REOPEN: &str = r"
import bpy
import json
import sys
root = sys.argv[sys.argv.index('--') + 1]
blend_path = sys.argv[sys.argv.index('--') + 2]
output = sys.argv[sys.argv.index('--') + 3]
expected_path = sys.argv[sys.argv.index('--') + 4]
bpy.ops.wm.open_mainfile(filepath=blend_path)
scene = bpy.context.scene
with open(expected_path, encoding='utf-8') as handle:
    expected = json.load(handle)

def plain(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, 'to_list'):
        return list(value.to_list())
    if hasattr(value, 'name_full'):
        return value.name_full
    try:
        return [plain(item) for item in value]
    except Exception:
        return repr(value)

def fields(block, names):
    return {name:plain(getattr(block,name)) for name in names}

def cache_values(block):
    cache = getattr(block, 'point_cache', None)
    if cache is None:
        return None
    return {'frame_start':int(cache.frame_start), 'frame_end':int(cache.frame_end),
            'is_baked':bool(cache.is_baked)}

def geometry(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    if obj.type != 'MESH':
        return None
    mesh = evaluated.to_mesh()
    points = [list(vertex.co) for vertex in mesh.vertices]
    evaluated.to_mesh_clear()
    return points

checks = {}
cloth = bpy.data.objects['Cloth'].modifiers['ClothSettingsFixture']
checks['cloth'] = {'settings':fields(cloth.settings, ['goal_default','goal_spring','goal_friction','mass','air_damping','tension_stiffness','compression_stiffness','shear_stiffness','bending_stiffness','use_sewing_springs','use_dynamic_mesh']),
                   'collision_settings':fields(cloth.collision_settings, ['use_collision','distance_min','friction','damping','collision_quality','use_self_collision','self_distance_min','self_friction']),
                   'point_cache':cache_values(cloth)}
soft = bpy.data.objects['SoftBody'].modifiers['SoftBodySettingsFixture']
checks['soft_body'] = {'settings':fields(soft.settings, ['mass','friction','speed','goal_default','goal_spring','goal_friction','pull','push','damping','spring_length','aero','bend','use_goal','use_edges','use_stiff_quads','use_edge_collision','use_face_collision']),
                       'point_cache':cache_values(soft)}
collision = bpy.data.objects['Collision'].modifiers['CollisionSettingsFixture']
checks['collision'] = {'settings':fields(collision.settings, ['use','damping_factor','damping_random','friction_factor','friction_random','permeability','use_particle_kill','thickness_inner','thickness_outer','cloth_friction','absorption'])}
brush_mod = bpy.data.objects['DynamicPaintBrush'].modifiers['BrushSettingsFixture']
checks['dynamic_paint_brush'] = {'settings':fields(brush_mod.brush_settings, ['paint_color','paint_alpha','use_absolute_alpha','paint_wetness','use_paint_erase','wave_type','wave_factor','wave_clamp','use_smudge','smudge_strength','velocity_max','use_velocity_alpha']), 'canvas_surfaces':[], 'role':'brush'}
canvas_mod = bpy.data.objects['DynamicPaintCanvas'].modifiers['CanvasSettingsFixture']
canvas_surfaces = canvas_mod.canvas_settings.canvas_surfaces
checks['dynamic_paint_canvas'] = {'settings':{}, 'canvas_surfaces':[fields(canvas_surfaces[0], ['surface_format','surface_type','is_active','use_dissolve','dissolve_speed','use_drying','dry_speed','frame_start','frame_end'])], 'role':'canvas'}
domain_mod = bpy.data.objects['FluidDomain'].modifiers['FluidDomainSettingsFixture']
checks['fluid_domain'] = {'settings':fields(domain_mod.domain_settings, ['domain_type','resolution_max','use_adaptive_domain','cache_frame_start','cache_frame_end','cache_frame_offset','cache_data_format','cache_mesh_format','cache_directory','time_scale']), 'role':'DOMAIN', 'enabled':bool(domain_mod.show_viewport and domain_mod.show_render)}
flow_mod = bpy.data.objects['FluidFlow'].modifiers['FluidFlowSettingsFixture']
checks['fluid_flow'] = {'settings':fields(flow_mod.flow_settings, ['density','fuel_amount','temperature','flow_type','flow_behavior','flow_source','use_absolute','use_initial_velocity','velocity_coord','surface_distance']), 'role':'FLOW', 'enabled':bool(flow_mod.show_viewport and flow_mod.show_render)}
effector_mod = bpy.data.objects['FluidEffector'].modifiers['FluidEffectorSettingsFixture']
checks['fluid_effector'] = {'settings':fields(effector_mod.effector_settings, ['effector_type','surface_distance','use_plane_init','velocity_factor','guide_mode','use_effector','subframes']), 'role':'EFFECTOR', 'enabled':bool(effector_mod.show_viewport and effector_mod.show_render)}
emitter = bpy.data.objects['ParticleEmitter']
particle_system = emitter.particle_systems[0]
checks['particle_system'] = {'settings':fields(particle_system.settings, ['count','frame_start','frame_end','lifetime','lifetime_random','emit_from','physics_type','normal_factor','factor_random','particle_size','size_random','render_type','child_type']), 'point_cache':cache_values(particle_system), 'seed':int(particle_system.seed)}
volume_mod = bpy.data.objects['VolumeDisplace'].modifiers['VolumeDisplaceSettingsFixture']
checks['volume_displace'] = {'modifier':fields(volume_mod, ['strength','texture_map_mode','texture_mid_level','texture_sample_radius','texture_map_object'])}
mods = {}
for object_name, modifier_name in [('ParticleInstance','ParticleInstanceSettingsFixture'),('Explode','ExplodeSettingsFixture')]:
    modifier = bpy.data.objects[object_name].modifiers[modifier_name]
    mods[object_name] = {'type':modifier.type,
                         'particle_system_index':getattr(modifier,'particle_system_index',None),
                         'object':getattr(modifier,'object',None).name_full if getattr(modifier,'object',None) else None,
                         'use_edge_cut':getattr(modifier,'use_edge_cut',None),
                         'use_normal':getattr(modifier,'use_normal',None),
                         'use_children':getattr(modifier,'use_children',None),
                         'use_size':getattr(modifier,'use_size',None),
                         'show_alive':getattr(modifier,'show_alive',None),
                         'show_dead':getattr(modifier,'show_dead',None),
                         'show_unborn':getattr(modifier,'show_unborn',None),
                         'position':getattr(modifier,'position',None),
                         'random_position':getattr(modifier,'random_position',None),
                         'axis':getattr(modifier,'axis',None),
                         'space':getattr(modifier,'space',None),
                         'protect':getattr(modifier,'protect',None)}

scene.frame_set(1)
bpy.context.view_layer.update()
geometry_after = {name:geometry(bpy.data.objects[name]) for name in expected['geometry']}
with open(output, 'w', encoding='utf-8') as handle:
    json.dump({'settings':checks, 'modifiers':mods, 'geometry':geometry_after,
               'frame_start':scene.frame_start, 'frame_end':scene.frame_end}, handle)
";

fn assert_subset(actual: &Value, expected: &Value, context: &str) {
    match expected {
        Value::Object(values) => {
            for (key, expected_value) in values {
                assert_subset(
                    actual.get(key).unwrap_or(&Value::Null),
                    expected_value,
                    &format!("{context}.{key}"),
                );
            }
        }
        Value::Array(values) => {
            let actual_values = actual
                .as_array()
                .unwrap_or_else(|| panic!("{context}: expected array, got {actual}"));
            assert_eq!(actual_values.len(), values.len(), "{context}: array length");
            for (index, (actual_value, expected_value)) in
                actual_values.iter().zip(values).enumerate()
            {
                assert_subset(actual_value, expected_value, &format!("{context}[{index}]"));
            }
        }
        Value::Number(expected_number) if expected_number.is_f64() => {
            let expected_number = expected_number.as_f64().unwrap();
            let actual_number = actual
                .as_f64()
                .unwrap_or_else(|| panic!("{context}: expected number, got {actual}"));
            assert!(
                (actual_number - expected_number).abs() <= 1.0e-6,
                "{context}: expected {expected_number:?}, got {actual_number:?}"
            );
        }
        _ => assert_eq!(actual, expected, "{context}"),
    }
}

fn assert_geometry(actual: &Value, expected: &Value, context: &str) {
    if expected.is_null() {
        assert!(
            actual.is_null(),
            "{context}: expected non-mesh geometry, got {actual}"
        );
        return;
    }
    let actual_points: Vec<[f64; 3]> = serde_json::from_value(actual.clone()).unwrap();
    let expected_points: Vec<[f64; 3]> = serde_json::from_value(expected.clone()).unwrap();
    assert_eq!(
        actual_points.len(),
        expected_points.len(),
        "{context}: vertex count"
    );
    let mut used = vec![false; actual_points.len()];
    let mut max_error = 0.0_f64;
    for source in &expected_points {
        let (index, error) = actual_points
            .iter()
            .enumerate()
            .filter(|(index, _)| !used[*index])
            .map(|(index, point)| {
                let error = (point[0] - source[0]).powi(2)
                    + (point[1] - source[1]).powi(2)
                    + (point[2] - source[2]).powi(2);
                (index, error)
            })
            .min_by(|left, right| left.1.total_cmp(&right.1))
            .unwrap();
        used[index] = true;
        max_error = max_error.max(error.sqrt());
    }
    assert!(
        max_error <= 1.0e-6,
        "{context}: max per-vertex error {max_error}"
    );
}

fn modifier<'a>(doc: &'a Value, node_id: &str, name: &str) -> &'a Value {
    doc["nodes"][node_id]["modifiers"]
        .as_array()
        .unwrap()
        .iter()
        .find(|modifier| modifier["name"] == name)
        .unwrap_or_else(|| panic!("missing modifier {name} on {node_id}"))
}

fn assert_physics_settings(actual: &Value, expected: &Value, context: &str) {
    assert_subset(
        actual,
        &expected["settings"],
        &format!("{context}.settings"),
    );
    for key in [
        "collision_settings",
        "point_cache",
        "canvas_surfaces",
        "role",
        "seed",
    ] {
        let Some(value) = expected.get(key) else {
            continue;
        };
        if key == "canvas_surfaces" && value.as_array().is_some_and(Vec::is_empty) {
            continue;
        }
        assert_subset(&actual[key], value, &format!("{context}.{key}"));
    }
}

#[test]
fn simulation_modifiers_settings_caches_and_frame_ranges_round_trip() -> Result<(), Box<dyn Error>>
{
    let Some(blender) = blender() else {
        eprintln!("skipping Blender simulation round-trip; Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    run_blender_script(
        &blender,
        "make_sim_fixture.py",
        FIXTURE,
        root,
        &[],
        "Blender fixture failed",
        run_guarded,
    )?;
    let expected: Value = serde_json::from_slice(&fs::read(root.join("expected.json"))?)?;
    let project = root.join("project");
    pot_json(&["init", project.to_str().unwrap()])?;
    let imported = pot_json(&[
        "import",
        project.to_str().unwrap(),
        "--file",
        root.join("source.blend").to_str().unwrap(),
        "--format",
        "blend",
        "--mode",
        "replace",
        "--base-revision",
        "0",
        "--blender",
        blender.to_str().unwrap(),
    ])?;
    assert_eq!(imported["result"]["losses"], json!([]), "{imported}");
    assert_eq!(
        expected["settings"]["particle_system"]["point_cache"]["is_baked"], true,
        "the Blender fixture must carry a baked particle point cache",
    );
    let mappings = imported["result"]["id_mappings"].as_object().unwrap();
    let scene: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let scenes = scene["scenes"].as_object().unwrap();
    let source_scene = scenes.values().next().unwrap();
    assert_eq!(source_scene["frame_start"], expected["frame_start"]);
    assert_eq!(source_scene["frame_end"], expected["frame_end"]);

    let physics_property = [
        ("Cloth", "cloth", "physics_cloth", "ClothSettingsFixture"),
        (
            "SoftBody",
            "soft_body",
            "physics_soft_body",
            "SoftBodySettingsFixture",
        ),
        (
            "Collision",
            "collision",
            "physics_collision",
            "CollisionSettingsFixture",
        ),
        (
            "DynamicPaintBrush",
            "dynamic_paint_brush",
            "physics_dynamic_paint",
            "BrushSettingsFixture",
        ),
        (
            "DynamicPaintCanvas",
            "dynamic_paint_canvas",
            "physics_dynamic_paint",
            "CanvasSettingsFixture",
        ),
        (
            "FluidDomain",
            "fluid_domain",
            "physics_fluid",
            "FluidDomainSettingsFixture",
        ),
        (
            "FluidFlow",
            "fluid_flow",
            "physics_fluid",
            "FluidFlowSettingsFixture",
        ),
        (
            "FluidEffector",
            "fluid_effector",
            "physics_fluid",
            "FluidEffectorSettingsFixture",
        ),
        (
            "ParticleEmitter",
            "particle_system",
            "physics_particle_systems",
            "ParticleSystemFixture",
        ),
    ];
    for (object_name, case_name, property_name, modifier_name) in physics_property {
        let node_id = mappings[&format!("Object:{object_name}")].as_str().unwrap();
        let imported_modifier = modifier(&scene, node_id, modifier_name);
        assert!(
            imported_modifier["params"]["settings_id"]
                .as_str()
                .is_some()
        );
        let settings_id = imported_modifier["params"]["settings_id"].as_str().unwrap();
        let settings = if property_name == "physics_particle_systems" {
            &scene["nodes"][node_id]["properties"][property_name][settings_id]
        } else {
            &scene["nodes"][node_id]["properties"][property_name]
        };
        assert_physics_settings(settings, &expected["settings"][case_name], case_name);
    }

    let catalog = potter::catalog::feature_catalog();
    let parity_feature = catalog["features"]
        .as_array()
        .unwrap()
        .iter()
        .find(|feature| feature["feature_id"] == "physics.cloth.blender_solver_parity")
        .unwrap();
    assert_eq!(parity_feature["status"], "not_supported");
    assert_eq!(parity_feature["capabilities"]["evaluate"], false);

    for object_name in [
        "Cloth",
        "SoftBody",
        "Collision",
        "DynamicPaintBrush",
        "DynamicPaintCanvas",
        "FluidDomain",
        "FluidFlow",
        "FluidEffector",
        "ParticleInstance",
        "Explode",
    ] {
        let node_id = mappings[&format!("Object:{object_name}")].as_str().unwrap();
        let inspected = pot_json(&[
            "inspect",
            project.to_str().unwrap(),
            "--id",
            node_id,
            "--frame",
            "1",
        ])?;
        let actual = &inspected["result"]["items"][0]["evaluated_geometry"]["positions"];
        assert_geometry(
            actual,
            &expected["geometry"][object_name],
            &format!("Potter frame 1 {object_name}"),
        );
    }

    let instance_id = mappings["Object:ParticleInstance"].as_str().unwrap();
    let instance = modifier(&scene, instance_id, "ParticleInstanceSettingsFixture");
    assert_eq!(
        instance["params"]["object"],
        mappings["Object:ParticleEmitter"]
    );
    assert_subset(
        &instance["params"],
        &json!({
            "particle_system_index":expected["modifiers"]["ParticleInstance"]["particle_system_index"].as_u64().unwrap(),
            "use_normal":expected["modifiers"]["ParticleInstance"]["use_normal"],
            "use_children":expected["modifiers"]["ParticleInstance"]["use_children"],
            "use_size":expected["modifiers"]["ParticleInstance"]["use_size"],
            "show_alive":expected["modifiers"]["ParticleInstance"]["show_alive"],
            "show_dead":expected["modifiers"]["ParticleInstance"]["show_dead"],
            "show_unborn":expected["modifiers"]["ParticleInstance"]["show_unborn"],
            "position":expected["modifiers"]["ParticleInstance"]["position"],
            "random_position":expected["modifiers"]["ParticleInstance"]["random_position"],
            "axis":expected["modifiers"]["ParticleInstance"]["axis"],
            "space":expected["modifiers"]["ParticleInstance"]["space"],
        }),
        "particle instance params",
    );
    let explode_id = mappings["Object:Explode"].as_str().unwrap();
    let explode_modifier = modifier(&scene, explode_id, "ExplodeSettingsFixture");
    assert_subset(
        &explode_modifier["params"],
        &json!({
            "use_edge_cut":expected["modifiers"]["Explode"]["use_edge_cut"],
            "show_alive":expected["modifiers"]["Explode"]["show_alive"],
            "show_dead":expected["modifiers"]["Explode"]["show_dead"],
            "show_unborn":expected["modifiers"]["Explode"]["show_unborn"],
            "use_size":expected["modifiers"]["Explode"]["use_size"],
            "protect":expected["modifiers"]["Explode"]["protect"],
        }),
        "explode params",
    );

    let exported = root.join("roundtrip.blend");
    pot_json(&[
        "export",
        project.to_str().unwrap(),
        "--format",
        "blend",
        "--out",
        exported.to_str().unwrap(),
        "--blender",
        blender.to_str().unwrap(),
    ])?;
    let reopened_path = root.join("reopened.json");
    run_blender_script(
        &blender,
        "reopen_sim_fixture.py",
        REOPEN,
        root,
        &[&exported, &reopened_path, &root.join("expected.json")],
        "Blender fixture failed",
        run_guarded,
    )?;
    let after: Value = serde_json::from_slice(&fs::read(reopened_path)?)?;
    assert_eq!(after["frame_start"], expected["frame_start"]);
    assert_eq!(after["frame_end"], expected["frame_end"]);
    for case_name in [
        "cloth",
        "soft_body",
        "collision",
        "dynamic_paint_brush",
        "dynamic_paint_canvas",
        "fluid_domain",
        "fluid_flow",
        "fluid_effector",
        "particle_system",
        "volume_displace",
    ] {
        assert_subset(
            &after["settings"][case_name],
            &expected["settings"][case_name],
            &format!("reopened.{case_name}"),
        );
    }
    assert_subset(
        &after["modifiers"]["ParticleInstance"],
        &expected["modifiers"]["ParticleInstance"],
        "reopened ParticleInstance",
    );
    assert_subset(
        &after["modifiers"]["Explode"],
        &expected["modifiers"]["Explode"],
        "reopened Explode",
    );
    for object_name in expected["geometry"].as_object().unwrap().keys() {
        assert_geometry(
            &after["geometry"][object_name],
            &expected["geometry"][object_name],
            &format!("reopened frame 1 {object_name}"),
        );
    }
    Ok(())
}
