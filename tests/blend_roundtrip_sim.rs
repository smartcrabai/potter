#![expect(
    clippy::unwrap_used,
    reason = "Blender round-trip fixtures use fixed valid test data"
)]

use std::{collections::BTreeMap, error::Error, fs, path::Path, process::Command};

use serde_json::{Map, Value, json};
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

/// Loss ids the physics fixture imports with, sorted: one per Blender setting that Potter keeps but
/// does not evaluate and that the fixture leaves away from its Blender default (values Blender
/// derives from those settings included). Dynamic Paint surfaces are reported per surface, not as
/// one `canvas_surfaces` id.
const EXPECTED_PHYSICS_LOSSES: &[&str] = &[
    "blender.physics.cloth.use_dynamic_mesh",
    "blender.physics.cloth.use_sewing_springs",
    "blender.physics.cloth_collision.collision_quality",
    "blender.physics.cloth_collision.damping",
    "blender.physics.cloth_collision.friction",
    "blender.physics.cloth_collision.self_friction",
    "blender.physics.collision.absorption",
    "blender.physics.collision.damping_factor",
    "blender.physics.collision.damping_random",
    "blender.physics.collision.friction_factor",
    "blender.physics.collision.friction_random",
    "blender.physics.collision.permeability",
    "blender.physics.collision.use_particle_kill",
    "blender.physics.dynamic_paint_brush.smudge_strength",
    "blender.physics.dynamic_paint_brush.use_paint_erase",
    "blender.physics.dynamic_paint_brush.use_smudge",
    "blender.physics.dynamic_paint_brush.use_velocity_alpha",
    "blender.physics.dynamic_paint_brush.velocity_max",
    "blender.physics.dynamic_paint_brush.wave_factor",
    "blender.physics.dynamic_paint_brush.wave_type",
    "blender.physics.fluid_domain_liquid.cache_data_format",
    "blender.physics.fluid_domain_liquid.cache_frame_offset",
    "blender.physics.fluid_domain_liquid.cache_mesh_format",
    "blender.physics.fluid_domain_liquid.time_scale",
    "blender.physics.fluid_domain_liquid.use_adaptive_domain",
    "blender.physics.fluid_effector.subframes",
    "blender.physics.fluid_effector.surface_distance",
    "blender.physics.fluid_effector.use_plane_init",
    "blender.physics.fluid_effector.velocity_factor",
    "blender.physics.fluid_flow.fuel_amount",
    "blender.physics.fluid_flow.surface_distance",
    "blender.physics.fluid_flow.temperature",
    "blender.physics.fluid_flow.use_initial_velocity",
    "blender.physics.fluid_flow.velocity_coord",
    "blender.physics.soft_body.aero",
    "blender.physics.soft_body.bend",
    "blender.physics.soft_body.damping",
    "blender.physics.soft_body.goal_friction",
    "blender.physics.soft_body.pull",
    "blender.physics.soft_body.push",
    "blender.physics.soft_body.spring_length",
    "blender.physics.soft_body.use_edge_collision",
    "blender.physics.soft_body.use_face_collision",
    "blender.physics.soft_body.use_stiff_quads",
];

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
                  'dissolve_speed', 'use_drying', 'dry_speed', 'frame_start', 'frame_end',
                  'output_name_a']
for key, value in {'surface_format':'VERTEX', 'surface_type':'PAINT', 'is_active':True,
                   'use_dissolve':True, 'dissolve_speed':3, 'use_drying':True,
                   'dry_speed':2, 'frame_start':2, 'frame_end':4,
                   'output_name_a':'PaintMapFixture'}.items():
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
    'show_viewport':bool(domain_mod.show_viewport), 'show_render':bool(domain_mod.show_render)}
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
    'show_viewport':bool(flow_mod.show_viewport), 'show_render':bool(flow_mod.show_render)}
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
    'show_viewport':bool(effector_mod.show_viewport), 'show_render':bool(effector_mod.show_render)}
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
checks['dynamic_paint_brush'] = {'settings':fields(brush_mod.brush_settings, ['paint_color','paint_alpha','use_absolute_alpha','paint_wetness','use_paint_erase','wave_type','wave_factor','wave_clamp','use_smudge','smudge_strength','velocity_max','use_velocity_alpha']), 'canvas_surfaces':[], 'role':'brush', 'ui_type':brush_mod.ui_type}
canvas_mod = bpy.data.objects['DynamicPaintCanvas'].modifiers['CanvasSettingsFixture']
canvas_surfaces = canvas_mod.canvas_settings.canvas_surfaces
checks['dynamic_paint_canvas'] = {'settings':{}, 'canvas_surfaces':[fields(canvas_surfaces[0], ['surface_format','surface_type','is_active','use_dissolve','dissolve_speed','use_drying','dry_speed','frame_start','frame_end','output_name_a'])], 'role':'canvas', 'ui_type':canvas_mod.ui_type}
domain_mod = bpy.data.objects['FluidDomain'].modifiers['FluidDomainSettingsFixture']
checks['fluid_domain'] = {'settings':fields(domain_mod.domain_settings, ['domain_type','resolution_max','use_adaptive_domain','cache_frame_start','cache_frame_end','cache_frame_offset','cache_data_format','cache_mesh_format','cache_directory','time_scale']), 'role':'DOMAIN', 'fluid_type':domain_mod.fluid_type, 'show_viewport':bool(domain_mod.show_viewport), 'show_render':bool(domain_mod.show_render)}
flow_mod = bpy.data.objects['FluidFlow'].modifiers['FluidFlowSettingsFixture']
checks['fluid_flow'] = {'settings':fields(flow_mod.flow_settings, ['density','fuel_amount','temperature','flow_type','flow_behavior','flow_source','use_absolute','use_initial_velocity','velocity_coord','surface_distance']), 'role':'FLOW', 'fluid_type':flow_mod.fluid_type, 'show_viewport':bool(flow_mod.show_viewport), 'show_render':bool(flow_mod.show_render)}
effector_mod = bpy.data.objects['FluidEffector'].modifiers['FluidEffectorSettingsFixture']
checks['fluid_effector'] = {'settings':fields(effector_mod.effector_settings, ['effector_type','surface_distance','use_plane_init','velocity_factor','guide_mode','use_effector','subframes']), 'role':'EFFECTOR', 'fluid_type':effector_mod.fluid_type, 'show_viewport':bool(effector_mod.show_viewport), 'show_render':bool(effector_mod.show_render)}
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
fluid_particle_types = sorted(system.settings.type for obj in bpy.data.objects
                              for system in obj.particle_systems
                              if system.settings.type in ('FLIP', 'SPRAY', 'FOAM', 'BUBBLE', 'TRACER'))
with open(output, 'w', encoding='utf-8') as handle:
    json.dump({'settings':checks, 'modifiers':mods, 'geometry':geometry_after,
               'frame_start':scene.frame_start, 'frame_end':scene.frame_end,
               'fluid_particle_types':fluid_particle_types}, handle)
";

const REOPEN_DYNAMIC_PAINT_ROLES: &str = r"
import bpy, json, sys
arguments = sys.argv[sys.argv.index('--') + 1:]
bpy.ops.wm.open_mainfile(filepath=arguments[1])

def role_state(object_name, modifier_name):
    modifier = bpy.data.objects[object_name].modifiers[modifier_name]
    return {
        'ui_type': modifier.ui_type,
        'has_brush_settings': modifier.brush_settings is not None,
        'has_canvas_settings': modifier.canvas_settings is not None,
    }

result = {
    'canvas_object': role_state('DynamicPaintCanvas', 'CanvasSettingsFixture'),
    'brush_object': role_state('DynamicPaintBrush', 'BrushSettingsFixture'),
}
with open(arguments[2], 'w', encoding='utf-8') as handle:
    json.dump(result, handle)
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
        "--allow-lossy",
        "--mode",
        "replace",
        "--base-revision",
        "0",
        "--blender",
        blender.to_str().unwrap(),
    ])?;
    let mut loss_ids = imported["result"]["losses"]
        .as_array()
        .unwrap()
        .iter()
        .map(|loss| loss["feature_id"].as_str().unwrap().to_owned())
        .collect::<Vec<_>>();
    loss_ids.sort();
    assert_eq!(loss_ids, EXPECTED_PHYSICS_LOSSES, "{imported}");
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
    assert_imported_physics(&expected, mappings, &scene);
    assert_catalog_parity();
    assert_inspected_geometry(&project, &expected, mappings)?;
    assert_particle_instance_and_explode(&expected, mappings, &scene);
    let after = assert_export_and_reopen(root, &project, &blender, &expected)?;
    assert_reopened_settings(&after, &expected);
    let same_role_scene = assert_dynamic_paint_same_role(root, &project, &scene, mappings)?;
    assert_dynamic_paint_role_transitions(
        root,
        &project,
        &blender,
        &scene,
        mappings,
        &same_role_scene,
    )?;
    Ok(())
}

#[test]
fn dynamic_paint_frame_end_matches_blender_and_multiple_surfaces_are_refused()
-> Result<(), Box<dyn Error>> {
    let Some(blender) = blender() else {
        eprintln!("skipping Blender Dynamic Paint range oracle; Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    run_blender_script(
        &blender,
        "generate_dynamic_paint_frame_range.py",
        include_str!("fixtures/dynamic_paint_frame_range/generate.py"),
        root,
        &[],
        "Blender Dynamic Paint range oracle failed",
        run_guarded,
    )?;
    let generated: Value = serde_json::from_slice(&fs::read(root.join("blender_expected.json"))?)?;
    let oracle: Value = serde_json::from_str(include_str!(
        "fixtures/dynamic_paint_frame_range/blender_expected.json"
    ))?;
    assert_subset(&generated, &oracle, "Blender Dynamic Paint oracle");
    assert_eq!(oracle["blender_version"], "5.2.2 LTS");
    let held = &oracle["held_deformation"];
    let end_colors = &held["frames"]["2"]["colors"]["HeldPaint"];
    let held_colors = &held["frames"]["3"]["colors"]["HeldPaint"];
    assert_eq!(
        end_colors, held_colors,
        "Blender must hold the end-frame colors"
    );
    assert!(
        (held["frames"]["2"]["positions"][4][2].as_f64().unwrap()
            - held["frames"]["3"]["positions"][4][2].as_f64().unwrap())
        .abs()
            > 0.5,
        "the Blender oracle canvas must continue deforming after frame_end"
    );

    let held_project = root.join("held_project");
    pot_json(&["init", held_project.to_str().unwrap()])?;
    pot_json(&[
        "import",
        held_project.to_str().unwrap(),
        "--file",
        root.join("HeldDynamicPaint.blend").to_str().unwrap(),
        "--format",
        "blend",
        "--allow-lossy",
        "--mode",
        "replace",
        "--base-revision",
        "0",
        "--blender",
        blender.to_str().unwrap(),
    ])?;
    let held_scene: Value = serde_json::from_slice(&fs::read(held_project.join("scene.json"))?)?;
    let (held_canvas_id, held_canvas) = dynamic_paint_canvas(&held_scene);
    assert_eq!(
        held_canvas["properties"]["physics_dynamic_paint"]["canvas_surfaces"][0]["output_name_a"],
        "HeldPaint"
    );
    let held_bake = root.join("held_bake");
    let bake = pot_json(&[
        "bake",
        held_project.to_str().unwrap(),
        "--kind",
        "simulation",
        "--frames",
        "1:4",
        "--out",
        held_bake.to_str().unwrap(),
    ])?;
    assert_eq!(bake["result"]["frame_count"], 4);
    let mut evaluated_frames = Vec::new();
    for (index, blender_frame) in [(1, "2"), (2, "3")] {
        let frame: Value =
            serde_json::from_slice(&fs::read(held_bake.join(format!("frame_{index:08}.json")))?)?;
        if blender_frame == "3" {
            assert_subset(
                &frame["paint_colors"][held_canvas_id.as_str()],
                &held["frames"][blender_frame]["colors"]["HeldPaint"],
                &format!("physics paint colors at Blender frame {blender_frame}"),
            );
        }
        let mesh = &frame["deformed_meshes"][held_canvas_id.as_str()];
        assert_dynamic_paint_corner_layer(
            mesh,
            "HeldPaint",
            &held["frames"][blender_frame]["colors"]["HeldPaint"],
            &format!("stack color layer at Blender frame {blender_frame}"),
        );
        evaluated_frames.push(frame);
    }
    let fresh_held_bake = root.join("held_frame_3_bake");
    let fresh_bake = pot_json(&[
        "bake",
        held_project.to_str().unwrap(),
        "--kind",
        "simulation",
        "--frames",
        "3:3",
        "--out",
        fresh_held_bake.to_str().unwrap(),
    ])?;
    assert_eq!(fresh_bake["result"]["frame_count"], 1);
    let fresh_held_frame: Value =
        serde_json::from_slice(&fs::read(fresh_held_bake.join("frame_00000000.json"))?)?;
    assert_eq!(
        fresh_held_frame["paint_colors"][held_canvas_id.as_str()],
        evaluated_frames[1]["paint_colors"][held_canvas_id.as_str()],
        "held-frame paint must not depend on whether frame 3 followed a sequence or was evaluated alone"
    );
    let center_z = |frame: &Value| {
        frame["deformed_meshes"][held_canvas_id.as_str()]["vertices"][4]["co"][2]
            .as_f64()
            .unwrap()
    };
    assert!(
        (center_z(&evaluated_frames[0]) - center_z(&evaluated_frames[1])).abs() > 0.5,
        "Potter's upstream Wave must deform the canvas after frame_end"
    );
    let shape_key = &oracle["animated_shape_key"];
    let shape_key_end_colors = &shape_key["frames"]["2"]["colors"]["ShapeKeyPaint"];
    let shape_key_held_colors = &shape_key["frames"]["3"]["colors"]["ShapeKeyPaint"];
    assert_eq!(
        shape_key_end_colors, shape_key_held_colors,
        "Blender must hold the shape-key canvas colors after frame_end"
    );
    assert!(
        (shape_key["frames"]["2"]["positions"][4][2]
            .as_f64()
            .unwrap()
            - shape_key["frames"]["3"]["positions"][4][2]
                .as_f64()
                .unwrap())
        .abs()
            > 0.5,
        "the Blender oracle shape key must continue deforming after frame_end"
    );

    let shape_key_project = root.join("shape_key_project");
    pot_json(&["init", shape_key_project.to_str().unwrap()])?;
    pot_json(&[
        "import",
        shape_key_project.to_str().unwrap(),
        "--file",
        root.join("ShapeKeyDynamicPaint.blend").to_str().unwrap(),
        "--format",
        "blend",
        "--allow-lossy",
        "--mode",
        "replace",
        "--base-revision",
        "0",
        "--blender",
        blender.to_str().unwrap(),
    ])?;
    let shape_key_scene: Value =
        serde_json::from_slice(&fs::read(shape_key_project.join("scene.json"))?)?;
    let (shape_key_canvas_id, shape_key_canvas) = dynamic_paint_canvas(&shape_key_scene);
    assert_eq!(
        shape_key_canvas["properties"]["physics_dynamic_paint"]["canvas_surfaces"][0]["output_name_a"],
        "ShapeKeyPaint"
    );
    let shape_key_bake = root.join("shape_key_bake");
    let bake = pot_json(&[
        "bake",
        shape_key_project.to_str().unwrap(),
        "--kind",
        "simulation",
        "--frames",
        "1:4",
        "--out",
        shape_key_bake.to_str().unwrap(),
    ])?;
    assert_eq!(bake["result"]["frame_count"], 4);
    let mut shape_key_frames = Vec::new();
    for (index, blender_frame) in [(1, "2"), (2, "3")] {
        let frame: Value = serde_json::from_slice(&fs::read(
            shape_key_bake.join(format!("frame_{index:08}.json")),
        )?)?;
        let expected_colors = &shape_key["frames"][blender_frame]["colors"]["ShapeKeyPaint"];
        assert_subset(
            &frame["paint_colors"][shape_key_canvas_id.as_str()],
            expected_colors,
            &format!("shape-key physics paint colors at Blender frame {blender_frame}"),
        );
        assert_dynamic_paint_corner_layer(
            &frame["deformed_meshes"][shape_key_canvas_id.as_str()],
            "ShapeKeyPaint",
            expected_colors,
            &format!("shape-key stack color layer at Blender frame {blender_frame}"),
        );
        assert_dynamic_paint_physics_matches_corner_layer(
            &frame["paint_colors"][shape_key_canvas_id.as_str()],
            &frame["deformed_meshes"][shape_key_canvas_id.as_str()],
            "ShapeKeyPaint",
            &format!("shape-key physics/stack color agreement at Blender frame {blender_frame}"),
        );
        shape_key_frames.push(frame);
    }
    let shape_key_center_z = |frame: &Value| {
        frame["deformed_meshes"][shape_key_canvas_id.as_str()]["vertices"][4]["co"][2]
            .as_f64()
            .unwrap()
    };
    assert!(
        (shape_key_center_z(&shape_key_frames[0]) - shape_key_center_z(&shape_key_frames[1])).abs()
            > 0.5,
        "Potter's shape-key canvas must deform after Dynamic Paint frame_end"
    );

    let multiple = &oracle["multiple_surfaces"];
    assert_eq!(
        multiple["frames"]["2"]["colors"]["EarlyPaint"][4][3], 1.0,
        "the early surface must paint at its frame_end"
    );
    assert_eq!(
        multiple["frames"]["2"]["colors"]["LatePaint"][4][3], 0.0,
        "the late surface must remain unpainted before its frame_start"
    );
    assert_eq!(multiple["frames"]["3"]["colors"]["EarlyPaint"][4][3], 1.0);
    assert_eq!(multiple["frames"]["3"]["colors"]["LatePaint"][4][3], 1.0);

    let multiple_project = root.join("multiple_project");
    pot_json(&["init", multiple_project.to_str().unwrap()])?;
    pot_json(&[
        "import",
        multiple_project.to_str().unwrap(),
        "--file",
        root.join("MultipleDynamicPaint.blend").to_str().unwrap(),
        "--format",
        "blend",
        "--allow-lossy",
        "--mode",
        "replace",
        "--base-revision",
        "0",
        "--blender",
        blender.to_str().unwrap(),
    ])?;
    let multiple_scene: Value =
        serde_json::from_slice(&fs::read(multiple_project.join("scene.json"))?)?;
    let (_, multiple_canvas) = dynamic_paint_canvas(&multiple_scene);
    let surfaces = multiple_canvas["properties"]["physics_dynamic_paint"]["canvas_surfaces"]
        .as_array()
        .unwrap();
    assert_eq!(surfaces.len(), 2);
    assert_eq!(surfaces[0]["output_name_a"], "EarlyPaint");
    assert_eq!(surfaces[1]["output_name_a"], "LatePaint");
    let mut command = Command::new(env!("CARGO_BIN_EXE_pot"));
    command.args([
        "bake",
        multiple_project.to_str().unwrap(),
        "--kind",
        "simulation",
        "--frames",
        "3:3",
        "--out",
        root.join("multiple_bake").to_str().unwrap(),
        "--json",
    ]);
    let failure = run_guarded(command)?;
    assert!(
        !failure.status.success(),
        "multiple surfaces must be refused"
    );
    let failure: Value = serde_json::from_slice(&failure.stdout)?;
    assert_eq!(failure["error"]["code"], "UNSUPPORTED_FEATURE");
    assert_eq!(
        failure["error"]["details"]["feature_id"],
        "physics.dynamic_paint_canvas.canvas_surfaces"
    );
    Ok(())
}

fn dynamic_paint_canvas(scene: &Value) -> (String, Value) {
    let (id, node) = scene["nodes"]
        .as_object()
        .unwrap()
        .iter()
        .find(|(_, node)| node["properties"]["physics_dynamic_paint"]["role"] == "canvas")
        .unwrap();
    (id.clone(), node.clone())
}

fn assert_dynamic_paint_corner_layer(mesh: &Value, name: &str, expected: &Value, context: &str) {
    let vertices = mesh["vertices"].as_array().unwrap();
    let vertex_indices: BTreeMap<u64, usize> = vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex["id"].as_u64().unwrap(), index))
        .collect();
    let attribute = &mesh["attributes"][name];
    assert_eq!(attribute["domain"], "corner", "{context}: domain");
    let values = attribute["values"].as_object().unwrap();
    let expected = expected.as_array().unwrap();
    for face in mesh["faces"].as_array().unwrap() {
        let face_id = face["id"].as_u64().unwrap();
        let face_key = format!("f{face_id}");
        let corners = values[&face_key].as_array().unwrap();
        for (corner, vertex_id) in face["v"].as_array().unwrap().iter().enumerate() {
            let vertex_index = vertex_indices[&vertex_id.as_u64().unwrap()];
            assert_subset(
                &corners[corner],
                &expected[vertex_index],
                &format!("{context}, face {face_id}, corner {corner}"),
            );
        }
    }
}

fn assert_dynamic_paint_physics_matches_corner_layer(
    paint_colors: &Value,
    mesh: &Value,
    name: &str,
    context: &str,
) {
    let vertices = mesh["vertices"].as_array().unwrap();
    let vertex_indices: BTreeMap<u64, usize> = vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex["id"].as_u64().unwrap(), index))
        .collect();
    let paint_colors = paint_colors.as_array().unwrap();
    assert_eq!(
        paint_colors.len(),
        vertices.len(),
        "{context}: vertex count"
    );
    let attribute = &mesh["attributes"][name];
    assert_eq!(attribute["domain"], "corner", "{context}: domain");
    let values = attribute["values"].as_object().unwrap();
    for face in mesh["faces"].as_array().unwrap() {
        let face_id = face["id"].as_u64().unwrap();
        let face_key = format!("f{face_id}");
        let corners = values[&face_key].as_array().unwrap();
        for (corner, vertex_id) in face["v"].as_array().unwrap().iter().enumerate() {
            let vertex_index = vertex_indices[&vertex_id.as_u64().unwrap()];
            assert_subset(
                &corners[corner],
                &paint_colors[vertex_index],
                &format!("{context}, face {face_id}, corner {corner}"),
            );
        }
    }
}

fn assert_imported_physics(expected: &Value, mappings: &Map<String, Value>, scene: &Value) {
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
        let imported_modifier = modifier(scene, node_id, modifier_name);
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
        let mut expected_settings = expected["settings"][case_name].clone();
        if property_name == "physics_fluid" {
            let role = expected_settings["role"].as_str().unwrap();
            assert_eq!(imported_modifier["params"]["fluid_type"], role);
            expected_settings.as_object_mut().unwrap().remove("role");
        } else if property_name == "physics_dynamic_paint" {
            let role = expected_settings["role"].as_str().unwrap();
            assert_eq!(
                imported_modifier["params"]["ui_type"],
                if role == "brush" { "BRUSH" } else { "CANVAS" }
            );
        }
        assert_physics_settings(settings, &expected_settings, case_name);
        if let Some(show_viewport) = expected["settings"][case_name].get("show_viewport") {
            assert_eq!(imported_modifier["enabled"], true, "{case_name}.enabled");
            assert_eq!(
                imported_modifier["show_viewport"], *show_viewport,
                "{case_name}.show_viewport"
            );
            assert_eq!(
                imported_modifier["show_render"], expected["settings"][case_name]["show_render"],
                "{case_name}.show_render"
            );
        }
    }
    // The domain's Liquid particle system imports with Blender's FLIP type.
    let flip_systems = scene["nodes"]
        .as_object()
        .unwrap()
        .values()
        .filter_map(|node| node["properties"]["physics_particle_systems"].as_object())
        .flat_map(|systems| systems.values())
        .filter(|system| system["type"] == "FLIP")
        .count();
    assert_eq!(flip_systems, 1, "imported FLIP particle system count");
}

fn assert_catalog_parity() {
    let catalog = potter_core::catalog::feature_catalog();
    let parity_feature = catalog["features"]
        .as_array()
        .unwrap()
        .iter()
        .find(|feature| feature["feature_id"] == "physics.cloth.blender_solver_parity")
        .unwrap();
    assert_eq!(parity_feature["status"], "not_supported");
    assert_eq!(parity_feature["capabilities"]["evaluate"], false);
}

fn assert_inspected_geometry(
    project: &Path,
    expected: &Value,
    mappings: &Map<String, Value>,
) -> Result<(), Box<dyn Error>> {
    // Potter evaluates these at frame 1 and must reproduce Blender's mesh.
    for object_name in [
        "Collision",
        "DynamicPaintBrush",
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
    // These carry Blender settings that the lossy import reported and the user accepted. Potter does not
    // evaluate them, and it must not refuse the scene for them: evaluation warns about each accepted value
    // (`accepted_loss`) instead, so the frame-1 inspect succeeds.
    for object_name in ["Cloth", "SoftBody", "DynamicPaintCanvas", "FluidDomain"] {
        let node_id = mappings[&format!("Object:{object_name}")].as_str().unwrap();
        let mut command = Command::new(env!("CARGO_BIN_EXE_pot"));
        command.args([
            "inspect",
            project.to_str().unwrap(),
            "--id",
            node_id,
            "--frame",
            "1",
            "--json",
        ]);
        let output = run_guarded(command)?;
        let envelope: Value = serde_json::from_slice(&output.stdout)?;
        assert!(
            output.status.success(),
            "{object_name} refused at frame 1 for accepted lossy values: {envelope}"
        );
        for warning in envelope["warnings"].as_array().into_iter().flatten() {
            if warning["details"]["feature_id"]
                .as_str()
                .is_some_and(|id| id.starts_with("blender.physics."))
            {
                assert_eq!(warning["details"]["status"], "accepted_loss", "{envelope}");
            }
        }
    }
    Ok(())
}

fn assert_particle_instance_and_explode(
    expected: &Value,
    mappings: &Map<String, Value>,
    scene: &Value,
) {
    let instance_id = mappings["Object:ParticleInstance"].as_str().unwrap();
    let instance = modifier(scene, instance_id, "ParticleInstanceSettingsFixture");
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
    let explode_modifier = modifier(scene, explode_id, "ExplodeSettingsFixture");
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
}

fn assert_export_and_reopen(
    root: &Path,
    project: &Path,
    blender: &Path,
    expected: &Value,
) -> Result<Value, Box<dyn Error>> {
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
        blender,
        "reopen_sim_fixture.py",
        REOPEN,
        root,
        &[&exported, &reopened_path, &root.join("expected.json")],
        "Blender fixture failed",
        run_guarded,
    )?;
    let after: Value = serde_json::from_slice(&fs::read(reopened_path)?)?;
    assert_eq!(
        after["fluid_particle_types"],
        json!(["FLIP"]),
        "the exported FLIP particle system must reopen as one FLIP system"
    );
    assert_eq!(after["frame_start"], expected["frame_start"]);
    assert_eq!(after["frame_end"], expected["frame_end"]);
    Ok(after)
}

fn assert_reopened_settings(after: &Value, expected: &Value) {
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
    assert_eq!(after["settings"]["dynamic_paint_brush"]["ui_type"], "BRUSH");
    assert_eq!(
        after["settings"]["dynamic_paint_canvas"]["ui_type"],
        "CANVAS"
    );
    for (name, role) in [
        ("fluid_domain", "DOMAIN"),
        ("fluid_flow", "FLOW"),
        ("fluid_effector", "EFFECTOR"),
    ] {
        assert_eq!(after["settings"][name]["fluid_type"], role);
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
}

fn assert_dynamic_paint_same_role(
    root: &Path,
    project: &Path,
    scene: &Value,
    mappings: &Map<String, Value>,
) -> Result<Value, Box<dyn Error>> {
    let canvas_id = mappings["Object:DynamicPaintCanvas"].as_str().unwrap();
    let brush_id = mappings["Object:DynamicPaintBrush"].as_str().unwrap();
    let canvas_modifier_id = modifier(scene, canvas_id, "CanvasSettingsFixture")["id"]
        .as_str()
        .unwrap();
    let brush_modifier_id = modifier(scene, brush_id, "BrushSettingsFixture")["id"]
        .as_str()
        .unwrap();
    let imported_canvas_surfaces =
        scene["nodes"][canvas_id]["properties"]["physics_dynamic_paint"]["canvas_surfaces"].clone();
    let imported_brush_color =
        scene["nodes"][brush_id]["properties"]["physics_dynamic_paint"]["paint_color"].clone();
    let imported_compatibility = scene["compatibility"]["blender_adapter"]["intermediate"].clone();
    let imported_history_head = scene["history"]["head"].as_str().unwrap().to_owned();
    let history_hash = imported_history_head.strip_prefix("sha256:").unwrap();
    let history_blob_path = project
        .join("history")
        .join("sha256")
        .join(history_hash)
        .join("blob.json");
    let imported_history_blob = fs::read(&history_blob_path)?;

    let same_role_batch = json!({
        "schema_version":1,
        "base_revision":scene["revision"],
        "operations":[
            {"op":"physics.dynamic_paint.update","target":{"id":canvas_id},"set":{"strength":0.5}},
            {"op":"modifier.update","target":{"id":canvas_id},"modifier_id":canvas_modifier_id,"set":{"params":{"ui_type":"CANVAS"}}},
            {"op":"physics.dynamic_paint.update","target":{"id":brush_id},"set":{"paint_alpha":0.75}},
            {"op":"modifier.update","target":{"id":brush_id},"modifier_id":brush_modifier_id,"set":{"params":{"ui_type":"BRUSH"}}}
        ]
    });
    let same_role_path = root.join("dynamic_paint_same_role.json");
    fs::write(&same_role_path, serde_json::to_vec(&same_role_batch)?)?;
    pot_json(&[
        "apply",
        project.to_str().unwrap(),
        "--file",
        same_role_path.to_str().unwrap(),
    ])?;
    let same_role_scene: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    assert_eq!(
        same_role_scene["nodes"][canvas_id]["properties"]["physics_dynamic_paint"]["canvas_surfaces"],
        imported_canvas_surfaces
    );
    assert_eq!(
        same_role_scene["nodes"][brush_id]["properties"]["physics_dynamic_paint"]["paint_color"],
        imported_brush_color
    );
    assert_eq!(
        same_role_scene["compatibility"]["blender_adapter"]["intermediate"],
        imported_compatibility
    );
    assert!(
        same_role_scene["history"]["undo"]
            .as_array()
            .unwrap()
            .contains(&json!(imported_history_head))
    );
    assert_eq!(fs::read(&history_blob_path)?, imported_history_blob);
    Ok(same_role_scene)
}

fn assert_dynamic_paint_role_transitions(
    root: &Path,
    project: &Path,
    blender: &Path,
    scene: &Value,
    mappings: &Map<String, Value>,
    same_role_scene: &Value,
) -> Result<(), Box<dyn Error>> {
    let canvas_id = mappings["Object:DynamicPaintCanvas"].as_str().unwrap();
    let brush_id = mappings["Object:DynamicPaintBrush"].as_str().unwrap();
    let canvas_modifier_id = modifier(scene, canvas_id, "CanvasSettingsFixture")["id"]
        .as_str()
        .unwrap();
    let brush_modifier_id = modifier(scene, brush_id, "BrushSettingsFixture")["id"]
        .as_str()
        .unwrap();
    let transition_batch = json!({
        "schema_version":1,
        "base_revision":same_role_scene["revision"],
        "operations":[
            {"op":"modifier.update","target":{"id":canvas_id},"modifier_id":canvas_modifier_id,"set":{"params":{"ui_type":"BRUSH"}}},
            {"op":"modifier.update","target":{"id":brush_id},"modifier_id":brush_modifier_id,"set":{"params":{"ui_type":"CANVAS"}}}
        ]
    });
    let transition_path = root.join("dynamic_paint_role_transitions.json");
    fs::write(&transition_path, serde_json::to_vec(&transition_batch)?)?;
    pot_json(&[
        "apply",
        project.to_str().unwrap(),
        "--file",
        transition_path.to_str().unwrap(),
    ])?;
    let transitioned_scene: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let canvas_settings =
        &transitioned_scene["nodes"][canvas_id]["properties"]["physics_dynamic_paint"];
    let brush_settings =
        &transitioned_scene["nodes"][brush_id]["properties"]["physics_dynamic_paint"];
    assert_eq!(canvas_settings["role"], "brush");
    assert!(canvas_settings.get("canvas_surfaces").is_none());
    assert!(canvas_settings.get("surface_format").is_none());
    assert_eq!(
        modifier(&transitioned_scene, canvas_id, "CanvasSettingsFixture")["params"]["ui_type"],
        "BRUSH"
    );
    assert_eq!(brush_settings["role"], "canvas");
    assert!(brush_settings.get("paint_color").is_none());
    assert!(brush_settings.get("use_velocity_alpha").is_none());
    assert_eq!(
        modifier(&transitioned_scene, brush_id, "BrushSettingsFixture")["params"]["ui_type"],
        "CANVAS"
    );

    // The same-role batch sets the canvas `strength`, which has no Blender property: exporting it is a
    // reported loss (`potter.physics.dynamic_paint_canvas.strength`), accepted here because this step tests
    // the role transitions, not that value.
    let transitioned_blend = root.join("dynamic_paint_role_transitions.blend");
    pot_json(&[
        "export",
        project.to_str().unwrap(),
        "--format",
        "blend",
        "--out",
        transitioned_blend.to_str().unwrap(),
        "--blender",
        blender.to_str().unwrap(),
        "--allow-lossy",
    ])?;
    let transitioned_reopen_path = root.join("dynamic_paint_role_reopened.json");
    run_blender_script(
        blender,
        "reopen_dynamic_paint_roles.py",
        REOPEN_DYNAMIC_PAINT_ROLES,
        root,
        &[&transitioned_blend, &transitioned_reopen_path],
        "Dynamic Paint role export failed",
        run_guarded,
    )?;
    let transitioned_reopened: Value =
        serde_json::from_slice(&fs::read(transitioned_reopen_path)?)?;
    assert_eq!(transitioned_reopened["canvas_object"]["ui_type"], "BRUSH");
    assert!(
        transitioned_reopened["canvas_object"]["has_brush_settings"]
            .as_bool()
            .unwrap()
    );
    assert!(
        !transitioned_reopened["canvas_object"]["has_canvas_settings"]
            .as_bool()
            .unwrap()
    );
    assert_eq!(transitioned_reopened["brush_object"]["ui_type"], "CANVAS");
    assert!(
        !transitioned_reopened["brush_object"]["has_brush_settings"]
            .as_bool()
            .unwrap()
    );
    assert!(
        transitioned_reopened["brush_object"]["has_canvas_settings"]
            .as_bool()
            .unwrap()
    );
    Ok(())
}
