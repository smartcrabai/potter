use std::{
    error::Error,
    fs,
    path::Path,
    process::{Command, Output},
};

use glam::{DMat4, DQuat, DVec3, EulerRot};
use serde_json::{Map, Value, json};
use tempfile::tempdir;

const MODIFIER_POSITION_TOLERANCE: f64 = 1.0e-4;
const FACE_DISTRIBUTION_TOLERANCE: f64 = 0.03;
const VELOCITY_STATISTIC_TOLERANCE: f64 = 0.25;
const SUBDIVISION_POSITION_TOLERANCE: f64 = 0.005;
const CLOTH_STIMULUS_TOLERANCE: f64 = 0.02;
const CLOTH_ORDER_TOLERANCE: f64 = 0.05;
const CLOTH_RMS_TOLERANCE: f64 = 0.30;
// The 2 m pinned-edge fixture has a non-planar free edge; 0.30 m limits RMS, centroid, and
// Z-envelope drift to 15% while exact rest/pin checks and order sensitivity cover other failures.
// A 0.05 m order response is 2.5% of the fixture span and must appear in Blender and Potter.
const CLOTH_CENTROID_TOLERANCE: f64 = 0.30;
const CLOTH_BOUNDS_TOLERANCE: f64 = 0.30;
const DENSITY_SAMPLE_TOLERANCE: f64 = 0.0001;
const MINIMUM_SURFACE_DISPLACEMENT: f64 = 1.0;
fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn run(command: &mut Command, action: &str) -> Result<Output, Box<dyn Error>> {
    let output = command.output()?;
    if !output.status.success() {
        return Err(format!(
            "{action} failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        )
        .into());
    }
    Ok(output)
}

fn initialize(scene: &Path) -> Result<(), Box<dyn Error>> {
    run(
        pot().arg("init").arg(scene).arg("--json"),
        "scene initialization",
    )?;
    Ok(())
}

fn apply(
    scene: &Path,
    batch_path: &Path,
    operations: &Value,
    revision: u64,
) -> Result<(), Box<dyn Error>> {
    fs::write(
        batch_path,
        serde_json::to_vec(&json!({
            "schema_version":1,
            "base_revision":revision,
            "operations":operations,
        }))?,
    )?;
    run(
        pot()
            .arg("apply")
            .arg(scene)
            .arg("--file")
            .arg(batch_path)
            .arg("--json"),
        "scene operation batch",
    )?;
    Ok(())
}

#[test]
fn physics_operations_create_stack_entries_linked_to_their_settings() -> Result<(), Box<dyn Error>>
{
    let directory = tempdir()?;
    let scene = directory.path().join("physics-stack");
    let batch = directory.path().join("operations.json");
    initialize(&scene)?;
    apply(
        &scene,
        &batch,
        &json!([
            {"op":"node.create","id":"body","kind":"box","params":{}},
            {"op":"physics.cloth.create","target":{"id":"body"},"settings":{"tension_stiffness":12.0}},
            {"op":"physics.soft_body.create","target":{"id":"body"},"settings":{"goal_strength":0.7}},
            {"op":"physics.dynamic_paint.create","target":{"id":"body"},"settings":{"role":"canvas"}},
            {"op":"physics.fluid.create","target":{"id":"body"},"settings":{"type":"liquid","resolution":4}},
            {"op":"physics.particle_emitter.create","target":{"id":"body"},"settings":{"rate":1.0,"lifetime":12.0}},
            {"op":"physics.collision.create","target":{"id":"body"},"settings":{"thickness":0.01}}
        ]),
        0,
    )?;

    let document: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let node = &document["nodes"]["body"];
    let modifiers = node["modifiers"]
        .as_array()
        .ok_or("modifier stack missing")?;
    for (modifier_type, id, property) in [
        ("cloth", "physics_cloth", "physics_cloth"),
        ("soft_body", "physics_soft_body", "physics_soft_body"),
        (
            "dynamic_paint",
            "physics_dynamic_paint",
            "physics_dynamic_paint",
        ),
        ("fluid", "physics_fluid", "physics_fluid"),
        (
            "particle_system",
            "physics_particle_emitter",
            "physics_particle_emitter",
        ),
        ("collision", "physics_collision", "physics_collision"),
    ] {
        let modifier = modifiers
            .iter()
            .find(|modifier| modifier["id"] == id)
            .ok_or_else(|| format!("{modifier_type} modifier was not created"))?;
        assert_eq!(modifier["type"], modifier_type);
        assert_eq!(modifier["params"]["settings_id"], id);
        assert!(node["properties"].get(property).is_some());
    }

    apply(
        &scene,
        &batch,
        &json!([{
            "op":"physics.cloth.update",
            "target":{"id":"body"},
            "set":{"compression_stiffness":10.0}
        }]),
        1,
    )?;
    let updated: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    assert_eq!(
        updated["nodes"]["body"]["properties"]["physics_cloth"]["compression_stiffness"],
        10.0
    );
    Ok(())
}

fn blender_executable() -> Option<std::path::PathBuf> {
    if let Some(path) = std::env::var_os("POTTER_BLENDER") {
        let path = std::path::PathBuf::from(path);
        return path.is_file().then_some(path);
    }
    if let Some(paths) = std::env::var_os("PATH")
        && let Some(path) = std::env::split_paths(&paths)
            .map(|directory| directory.join("blender"))
            .find(|path| path.is_file())
    {
        return Some(path);
    }
    let path = std::path::PathBuf::from("/Applications/Blender.app/Contents/MacOS/Blender");
    path.is_file().then_some(path)
}
fn injected_particles(states: &Value) -> Result<Vec<potter::sim::ParticleState>, Box<dyn Error>> {
    let states = states
        .as_array()
        .ok_or("Blender particle states are missing")?;
    states
        .iter()
        .map(|state| {
            let position: [f64; 3] = serde_json::from_value(state["position"].clone())?;
            let birth_position: [f64; 3] = serde_json::from_value(state["birth_position"].clone())?;
            let velocity: [f64; 3] = serde_json::from_value(state["velocity"].clone())?;
            let birth_frame = state["birth_frame"]
                .as_f64()
                .ok_or("Blender particle birth frame is missing")?;
            let death_frame = state["death_frame"]
                .as_f64()
                .ok_or("Blender particle death frame is missing")?;
            let normal: [f64; 3] = serde_json::from_value(state["normal"].clone())?;
            let rotation: [f64; 4] = serde_json::from_value(state["rotation"].clone())?;
            let birth_rotation: [f64; 4] = serde_json::from_value(state["birth_rotation"].clone())?;
            let size = state["size"]
                .as_f64()
                .ok_or("Blender particle size is missing")?;
            let life_state = match state["life_state"]
                .as_str()
                .ok_or("Blender particle life state is missing")?
            {
                "ALIVE" => potter::sim::ParticleLifeState::Alive,
                "DEAD" => potter::sim::ParticleLifeState::Dead,
                "UNBORN" => potter::sim::ParticleLifeState::Unborn,
                state => return Err(format!("unknown Blender particle state `{state}`").into()),
            };
            Ok(potter::sim::ParticleState {
                position,
                birth_position,
                normal,
                rotation,
                birth_rotation,
                velocity,
                birth_frame,
                death_frame,
                life_state,
                size,
                ..potter::sim::ParticleState::default()
            })
        })
        .collect()
}

#[test]
fn particle_instance_transforms_and_explode_face_splitting_match_blender()
-> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping modifier simulation Blender parity; Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let blender_script = root.join("modifier_parity.py");
    let blender_expected = root.join("blender_expected.json");
    fs::write(
        &blender_script,
        r"
import bpy, json, os, sys
from mathutils import Vector
root = sys.argv[sys.argv.index('--') + 1]

def matrix_array(matrix):
    return [matrix[row][column] for column in range(4) for row in range(4)]
scene = bpy.context.scene
scene.use_gravity = False
scene.frame_set(1)

def particle_births(system):
    return [{
        'position':list(particle.location),
        'rotation':[particle.rotation.x, particle.rotation.y, particle.rotation.z, particle.rotation.w],
    } for particle in system.particles]

def particle_states(system, births=None):
    states = [{
        'position':list(particle.location),
        'velocity':list(particle.velocity),
        'birth_frame':particle.birth_time,
        'death_frame':particle.birth_time + particle.lifetime,
        'normal':list(particle.rotation @ Vector((1,0,0))),
        'rotation':[particle.rotation.x, particle.rotation.y, particle.rotation.z, particle.rotation.w],
        'birth_position':list(particle.prev_location),
        'birth_rotation':[particle.prev_rotation.x, particle.prev_rotation.y, particle.prev_rotation.z, particle.prev_rotation.w],
        'size':particle.size,
        'life_state':particle.alive_state,
    } for particle in system.particles]
    if births is not None:
        for state, birth in zip(states, births):
            state['birth_position'] = birth['position']
            state['birth_rotation'] = birth['rotation']
    return states

emitter_mesh = bpy.data.meshes.new('asymmetric_emitter_mesh')
emitter_mesh.from_pydata(
    [(-1,-0.5,0), (1.3,-0.6,0.1), (0.7,1.2,0.2), (-0.5,0.7,0.5)],
    [],
    [(0,1,2,3)],
)
emitter = bpy.data.objects.new('Emitter', emitter_mesh)
scene.collection.objects.link(emitter)
emitter.location = (3,-1,0.5)
emitter.rotation_euler = (0,0,0)
emitter.scale = (1,1,1)
bpy.context.view_layer.objects.active = emitter
emitter.select_set(True)
bpy.ops.object.particle_system_add()
particle_settings = emitter.particle_systems[0].settings
particle_settings.count = 1
particle_settings.frame_start = 1
particle_settings.frame_end = 1
particle_settings.lifetime = 12
particle_settings.emit_from = 'VERT'
particle_settings.physics_type = 'NEWTON'
particle_settings.normal_factor = 1.3
particle_settings.factor_random = 0.0
particle_settings.particle_size = 0.7
particle_settings.size_random = 0.25
particle_settings.child_type = 'SIMPLE'
particle_settings.child_percent = 1
particle_settings.child_radius = 0.0
particle_settings.child_size = 1.0
scene.frame_set(1)
bpy.context.view_layer.update()
birth_graph = bpy.context.evaluated_depsgraph_get()
emitter_births = particle_births(emitter.evaluated_get(birth_graph).particle_systems[0])
scene.frame_set(2)
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
evaluated_emitter = emitter.evaluated_get(depsgraph)
instance_particle_states = particle_states(evaluated_emitter.particle_systems[0], emitter_births)
child_particle_count = len(evaluated_emitter.particle_systems[0].child_particles)
emitter_world_matrix = matrix_array(evaluated_emitter.matrix_world)
emitter.select_set(False)

instance_mesh = bpy.data.meshes.new('asymmetric_instance_mesh')
instance_mesh.from_pydata(
    [(-0.7,-0.3,0.1), (1.1,-0.2,0), (0.3,0.9,0.2), (-0.4,0.4,0.6)],
    [],
    [(0,1,2,3)],
)
instance = bpy.data.objects.new('Instance', instance_mesh)
scene.collection.objects.link(instance)
instance.location = (-0.4,0.6,0.2)
instance.rotation_euler = (0.17,-0.29,0.41)
instance.scale = (1.3,0.8,1.1)
instance.track_axis = 'POS_Z'
modifier = instance.modifiers.new('Particle Instance', 'PARTICLE_INSTANCE')
modifier.object = emitter
modifier.particle_system_index = 1
modifier.axis = 'Z'
modifier.use_normal = True
modifier.use_children = False
modifier.use_size = True
modifier.use_path = False
modifier.show_alive = True
modifier.show_dead = False
modifier.show_unborn = False
modifier.space = 'WORLD'
modifier.rotation = 0.0
depsgraph = bpy.context.evaluated_depsgraph_get()
evaluated = instance.evaluated_get(depsgraph)
evaluated_mesh = evaluated.to_mesh()
instance_points = [list(vertex.co) for vertex in evaluated_mesh.vertices]
instance_world_matrix = matrix_array(evaluated.matrix_world)
evaluated.to_mesh_clear()
instance_children = bpy.data.objects.new('InstanceChildren', instance_mesh)
bpy.context.scene.collection.objects.link(instance_children)
instance_children.location = (-0.4,0.6,0.2)
instance_children.rotation_euler = (0.17,-0.29,0.41)
instance_children.scale = (1.3,0.8,1.1)
instance_children.track_axis = 'POS_Z'
children_modifier = instance_children.modifiers.new('Child Particle Instance', 'PARTICLE_INSTANCE')
children_modifier.object = emitter
children_modifier.particle_system_index = 1
children_modifier.axis = 'Z'
children_modifier.use_normal = True
children_modifier.use_children = True
children_modifier.use_size = False
children_modifier.show_alive = True
children_modifier.show_dead = False
children_modifier.show_unborn = False
children_modifier.space = 'WORLD'
children_eval = instance_children.evaluated_get(bpy.context.evaluated_depsgraph_get())
children_mesh = children_eval.to_mesh()
child_instance_points = [list(vertex.co) for vertex in children_mesh.vertices]
children_eval.to_mesh_clear()

instance_local = bpy.data.objects.new('InstanceLocal', instance_mesh)
scene.collection.objects.link(instance_local)
instance_local.location = (-0.4,0.6,0.2)
instance_local.rotation_euler = (0.17,-0.29,0.41)
instance_local.scale = (1.3,0.8,1.1)
instance_local.track_axis = 'POS_Z'
local_modifier = instance_local.modifiers.new('Local Particle Instance', 'PARTICLE_INSTANCE')
local_modifier.object = emitter
local_modifier.particle_system_index = 1
local_modifier.axis = 'Z'
local_modifier.use_normal = True
local_modifier.use_children = False
local_modifier.use_size = True
local_modifier.show_alive = True
local_modifier.show_dead = False
local_modifier.show_unborn = False
local_modifier.space = 'LOCAL'
depsgraph = bpy.context.evaluated_depsgraph_get()
evaluated_local = instance_local.evaluated_get(depsgraph)
local_mesh = evaluated_local.to_mesh()
instance_local_points = [list(vertex.co) for vertex in local_mesh.vertices]
instance_local_world_matrix = matrix_array(evaluated_local.matrix_world)
evaluated_local.to_mesh_clear()
def make_visibility_instance(name, show_alive, show_dead, show_unborn):
    obj = bpy.data.objects.new(name, instance_mesh)
    scene.collection.objects.link(obj)
    obj.location = (-0.4,0.6,0.2)
    obj.rotation_euler = (0.17,-0.29,0.41)
    obj.scale = (1.3,0.8,1.1)
    obj.track_axis = 'POS_Z'
    modifier = obj.modifiers.new(name, 'PARTICLE_INSTANCE')
    modifier.object = emitter
    modifier.particle_system_index = 1
    modifier.axis = 'Z'
    modifier.use_normal = True
    modifier.use_children = False
    modifier.use_size = True
    modifier.show_alive = show_alive
    modifier.show_dead = show_dead
    modifier.show_unborn = show_unborn
    modifier.space = 'WORLD'
    return obj

def evaluated_points(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    points = [list(vertex.co) for vertex in mesh.vertices]
    evaluated.to_mesh_clear()
    return points

unborn_instance = make_visibility_instance('InstanceUnborn', False, False, True)
scene.frame_set(0)
bpy.context.view_layer.update()
unborn_particles = particle_states(
    emitter.evaluated_get(bpy.context.evaluated_depsgraph_get()).particle_systems[0],
    emitter_births,
)
unborn_instance_points = evaluated_points(unborn_instance)

dead_instance = make_visibility_instance('InstanceDead', False, True, False)
scene.frame_set(14)
bpy.context.view_layer.update()
dead_particles = particle_states(
    emitter.evaluated_get(bpy.context.evaluated_depsgraph_get()).particle_systems[0],
    emitter_births,
)
dead_instance_points = evaluated_points(dead_instance)

emitter.select_set(True)
bpy.context.view_layer.objects.active = emitter
particle_cache = emitter.particle_systems[0].point_cache
particle_cache.frame_start = 1
particle_cache.frame_end = 14
bpy.ops.ptcache.bake_all(bake=True)
scene.frame_set(14)
bpy.context.view_layer.update()

explode_mesh_data = bpy.data.meshes.new('explode_asymmetric_mesh')
explode_mesh_data.from_pydata(
    [(-0.7,-0.3,0.1), (1.1,-0.2,0), (0.3,0.9,0.2), (-0.4,0.4,0.6)],
    [],
    [(0,1,2,3)],
)
exploder = bpy.data.objects.new('Exploder', explode_mesh_data)
scene.collection.objects.link(exploder)
exploder.location = (0.4,-0.2,0.6)
exploder.rotation_euler = (0.23,0.38,-0.19)
exploder.scale = (1.2,0.8,1.1)
bpy.context.view_layer.objects.active = exploder
exploder.select_set(True)
bpy.ops.object.particle_system_add()
exploder_settings = exploder.particle_systems[0].settings
exploder_settings.count = 2
exploder_settings.frame_start = 1
exploder_settings.frame_end = 1
exploder_settings.lifetime = 12
exploder_settings.emit_from = 'VERT'
exploder_settings.physics_type = 'NEWTON'
exploder_settings.normal_factor = 1.1
explode = exploder.modifiers.new('Explode', 'EXPLODE')
explode.use_edge_cut = True
scene.frame_set(1)
bpy.context.view_layer.update()
birth_graph = bpy.context.evaluated_depsgraph_get()
explode_births = particle_births(exploder.evaluated_get(birth_graph).particle_systems[0])
scene.frame_set(2)
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
evaluated_exploder = exploder.evaluated_get(depsgraph)
explode_particle_states = particle_states(
    evaluated_exploder.particle_systems[0], explode_births
)
exploder_world_matrix = matrix_array(evaluated_exploder.matrix_world)
exploded_mesh = evaluated_exploder.to_mesh()
explode_world_points = [
    list(evaluated_exploder.matrix_world @ vertex.co) for vertex in exploded_mesh.vertices
]
face_count = len(exploded_mesh.polygons)
evaluated_exploder.to_mesh_clear()

bpy.context.view_layer.objects.active = None
cube_mesh_data = bpy.data.meshes.new('ordered_cube_mesh')
cube_mesh_data.from_pydata(
    [(-1,-1,-1), (1,-1,-1), (1,1,-1), (-1,1,-1),
     (-1,-1,1), (1,-1,1), (1,1,1), (-1,1,1)],
    [],
    [(0,3,2,1), (4,5,6,7), (0,1,5,4), (1,2,6,5), (2,3,7,6), (3,0,4,7)],
)
cube_exploder = bpy.data.objects.new('CubeExploder', cube_mesh_data)
scene.collection.objects.link(cube_exploder)
cube_exploder.rotation_euler = (-0.31,0.24,0.42)
cube_exploder.scale = (0.9,1.25,0.75)
bpy.context.view_layer.objects.active = cube_exploder
bpy.ops.object.particle_system_add()
cube_settings = cube_exploder.particle_systems[0].settings
cube_settings.count = 2
cube_settings.frame_start = 1
cube_settings.frame_end = 1
cube_settings.lifetime = 12
cube_settings.emit_from = 'VERT'
cube_settings.physics_type = 'NEWTON'
cube_settings.normal_factor = 1.1
cube_explode = cube_exploder.modifiers.new('Explode with edge cuts', 'EXPLODE')
cube_explode.use_edge_cut = True
scene.frame_set(1)
bpy.context.view_layer.update()
cube_birth_graph = bpy.context.evaluated_depsgraph_get()
cube_births = particle_births(cube_exploder.evaluated_get(cube_birth_graph).particle_systems[0])
cube_inverse = cube_exploder.matrix_world.inverted()
cube_birth_local = []
for birth in cube_births:
    sampled_position = cube_inverse @ Vector(birth['position'])
    source_vertex = min(
        cube_mesh_data.vertices,
        key=lambda vertex: (vertex.co - sampled_position).length_squared,
    )
    cube_birth_local.append(source_vertex.co.copy())
    birth['position'] = list(cube_exploder.matrix_world @ source_vertex.co)
cube_face_owners = []
cube_vertex_owners = [len(cube_births)] * len(cube_mesh_data.vertices)
for polygon in cube_mesh_data.polygons:
    center = sum(
        (cube_mesh_data.vertices[index].co for index in polygon.vertices),
        Vector((0,0,0)),
    ) / len(polygon.vertices)
    owner = min(
        range(len(cube_birth_local)),
        key=lambda index: (cube_birth_local[index] - center).length_squared,
    )
    cube_face_owners.append(owner)
    for index in polygon.vertices:
        cube_vertex_owners[index] = owner
cube_cut_masks = []
for polygon in cube_mesh_data.polygons:
    mask = 0
    vertices = list(polygon.vertices)
    for index in range(len(vertices)):
        if cube_vertex_owners[vertices[index]] != cube_vertex_owners[vertices[(index + 1) % len(vertices)]]:
            mask |= 1 << index
    cube_cut_masks.append(mask)
scene.frame_set(2)
bpy.context.view_layer.update()
cube_depsgraph = bpy.context.evaluated_depsgraph_get()
evaluated_cube_exploder = cube_exploder.evaluated_get(cube_depsgraph)
cube_particles = particle_states(
    evaluated_cube_exploder.particle_systems[0], cube_births
)
cube_matrix = matrix_array(evaluated_cube_exploder.matrix_world)
cube_mesh = evaluated_cube_exploder.to_mesh()
cube_world_points = [
    list(evaluated_cube_exploder.matrix_world @ vertex.co) for vertex in cube_mesh.vertices
]
cube_face_count = len(cube_mesh.polygons)
evaluated_cube_exploder.to_mesh_clear()
with open(os.path.join(root, 'blender_expected.json'), 'w', encoding='utf-8') as output:
    json.dump({
        'instance_points':instance_points,
        'instance_local_points':instance_local_points,
        'instance_particle_states':instance_particle_states,
        'child_instance_points':child_instance_points,
        'child_particle_count':child_particle_count,
        'unborn_instance_points':unborn_instance_points,
        'unborn_particle_states':unborn_particles,
        'dead_instance_points':dead_instance_points,
        'dead_particle_states':dead_particles,
        'emitter_world_matrix':emitter_world_matrix,
        'instance_world_matrix':instance_world_matrix,
        'instance_local_world_matrix':instance_local_world_matrix,
        'exploder_world_matrix':exploder_world_matrix,
        'explode_world_points':explode_world_points,
        'explode_particle_states':explode_particle_states,
        'explode_face_count':face_count,
        'cube_explode_world_points':cube_world_points,
        'cube_explode_particle_states':cube_particles,
        'cube_exploder_world_matrix':cube_matrix,
        'cube_face_owners':cube_face_owners,
        'cube_cut_masks':cube_cut_masks,
        'cube_birth_local':[list(position) for position in cube_birth_local],
        'cube_explode_face_count':cube_face_count,
    }, output)
",
    )?;
    let blender_output = Command::new(&blender)
        .args(["--background", "--factory-startup", "--python"])
        .arg(&blender_script)
        .arg("--")
        .arg(root)
        .output()?;
    if !blender_output.status.success() {
        return Err(format!(
            "Blender modifier fixture failed: stdout={} stderr={}",
            String::from_utf8_lossy(&blender_output.stdout),
            String::from_utf8_lossy(&blender_output.stderr),
        )
        .into());
    }
    if !blender_expected.is_file() {
        return Err(format!(
            "Blender did not write its evaluated modifier output: stdout={} stderr={}",
            String::from_utf8_lossy(&blender_output.stdout),
            String::from_utf8_lossy(&blender_output.stderr),
        )
        .into());
    }
    let blender: Value = serde_json::from_slice(&fs::read(&blender_expected)?)?;
    let instance_particles = injected_particles(&blender["instance_particle_states"])?;
    let child_particle_count = blender["child_particle_count"]
        .as_u64()
        .ok_or("Blender child particle count is missing")?;
    let child_particles = instance_particles
        .iter()
        .cloned()
        .map(|mut particle| {
            particle.is_child = true;
            particle
        })
        .collect::<Vec<_>>();
    assert_eq!(
        child_particle_count,
        u64::try_from(child_particles.len())?,
        "the zero-radius Simple child fixture emits one child per parent",
    );
    let mut instance_and_children = instance_particles.clone();
    instance_and_children.extend(child_particles.iter().cloned());
    let explode_particles = injected_particles(&blender["explode_particle_states"])?;
    assert_eq!(instance_particles.len(), 1);
    assert_eq!(explode_particles.len(), 2);
    let instance_source = potter::geom::Mesh::from_positions_and_faces(
        vec![
            DVec3::new(-0.7, -0.3, 0.1),
            DVec3::new(1.1, -0.2, 0.0),
            DVec3::new(0.3, 0.9, 0.2),
            DVec3::new(-0.4, 0.4, 0.6),
        ],
        vec![vec![0, 1, 2, 3]],
    )?;
    let emitter_columns: [f64; 16] =
        serde_json::from_value(blender["emitter_world_matrix"].clone())?;
    let emitter_world = DMat4::from_cols_array(&emitter_columns);
    let instance_columns: [f64; 16] =
        serde_json::from_value(blender["instance_world_matrix"].clone())?;
    let instance_world = DMat4::from_cols_array(&instance_columns);
    let direct_instance = potter::geom::modifiers::simulation::particle_instance(
        &instance_source,
        &instance_particles,
        emitter_world,
        instance_world,
        &Map::from_iter([
            ("use_normal".to_owned(), json!(true)),
            ("use_size".to_owned(), json!(true)),
            ("use_path".to_owned(), json!(false)),
            ("axis".to_owned(), json!("Z")),
            ("space".to_owned(), json!("WORLD")),
        ]),
        2.0,
        24.0,
    )?;
    let direct_instance_points = direct_instance
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    let blender_instance_points = blender_points(&blender["instance_points"])?;
    assert_eq!(
        direct_instance_points.len(),
        blender_instance_points.len(),
        "Blender instance points={}, particle states={instance_particles:?}",
        blender_instance_points.len(),
    );
    let instance_error =
        max_corresponding_distance(&direct_instance_points, &blender_instance_points)?;
    assert!(
        instance_error <= MODIFIER_POSITION_TOLERANCE,
        "particle instance per-vertex error {instance_error} exceeds {MODIFIER_POSITION_TOLERANCE}; Potter={direct_instance_points:?}; Blender={blender_instance_points:?}; particles={instance_particles:?}",
    );
    let direct_local_instance = potter::geom::modifiers::simulation::particle_instance(
        &instance_source,
        &instance_particles,
        emitter_world,
        instance_world,
        &Map::from_iter([
            ("use_normal".to_owned(), json!(true)),
            ("use_size".to_owned(), json!(true)),
            ("use_path".to_owned(), json!(false)),
            ("axis".to_owned(), json!("Z")),
            ("space".to_owned(), json!("LOCAL")),
        ]),
        2.0,
        24.0,
    )?;
    let direct_local_points = direct_local_instance
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    let blender_local_points = blender_points(&blender["instance_local_points"])?;
    assert_eq!(direct_local_points.len(), blender_local_points.len());
    let local_error = max_corresponding_distance(&direct_local_points, &blender_local_points)?;
    assert!(
        local_error <= MODIFIER_POSITION_TOLERANCE,
        "LOCAL particle instance per-vertex error {local_error} exceeds {MODIFIER_POSITION_TOLERANCE}; Potter={direct_local_points:?}; Blender={blender_local_points:?}; particle={instance_particles:?}",
    );
    let direct_child_instance = potter::geom::modifiers::simulation::particle_instance(
        &instance_source,
        &instance_and_children,
        emitter_world,
        instance_world,
        &Map::from_iter([
            ("use_normal".to_owned(), json!(true)),
            ("use_children".to_owned(), json!(true)),
            ("use_size".to_owned(), json!(false)),
            ("axis".to_owned(), json!("Z")),
            ("space".to_owned(), json!("WORLD")),
        ]),
        2.0,
        24.0,
    )?;
    let direct_child_points = direct_child_instance
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    let blender_child_points = blender_points(&blender["child_instance_points"])?;
    assert_eq!(direct_child_points.len(), blender_child_points.len());
    let child_error = max_corresponding_distance(&direct_child_points, &blender_child_points)?;
    assert!(
        child_error <= MODIFIER_POSITION_TOLERANCE,
        "Simple-child instance per-vertex error {child_error} exceeds {MODIFIER_POSITION_TOLERANCE}",
    );
    for (states_key, points_key, show_dead, show_unborn) in [
        (
            "unborn_particle_states",
            "unborn_instance_points",
            false,
            true,
        ),
        ("dead_particle_states", "dead_instance_points", true, false),
    ] {
        let lifecycle_particles = injected_particles(&blender[states_key])?;
        let lifecycle_instance = potter::geom::modifiers::simulation::particle_instance(
            &instance_source,
            &lifecycle_particles,
            emitter_world,
            instance_world,
            &Map::from_iter([
                ("use_normal".to_owned(), json!(true)),
                ("use_size".to_owned(), json!(true)),
                ("show_alive".to_owned(), json!(false)),
                ("show_dead".to_owned(), json!(show_dead)),
                ("show_unborn".to_owned(), json!(show_unborn)),
                ("axis".to_owned(), json!("Z")),
                ("space".to_owned(), json!("WORLD")),
            ]),
            2.0,
            24.0,
        )?;
        let lifecycle_points = lifecycle_instance
            .vertices
            .iter()
            .map(|vertex| vertex.co)
            .collect::<Vec<_>>();
        let blender_lifecycle_points = blender_points(&blender[points_key])?;
        assert_eq!(
            lifecycle_points.len(),
            blender_lifecycle_points.len(),
            "{states_key}: visible copy count differs",
        );
        let lifecycle_error =
            max_corresponding_distance(&lifecycle_points, &blender_lifecycle_points)?;
        assert!(
            lifecycle_error <= MODIFIER_POSITION_TOLERANCE,
            "{states_key} per-vertex instance error {lifecycle_error} exceeds {MODIFIER_POSITION_TOLERANCE}; Potter={lifecycle_points:?}, Blender={blender_lifecycle_points:?}, states={lifecycle_particles:?}",
        );
    }
    let Err(path_error) = potter::geom::modifiers::simulation::particle_instance(
        &instance_source,
        &instance_particles,
        emitter_world,
        instance_world,
        &Map::from_iter([("use_path".to_owned(), json!(true))]),
        2.0,
        24.0,
    ) else {
        return Err("sampled particle paths unexpectedly reported support".into());
    };
    assert_eq!(
        path_error.code,
        potter::error::ErrorCode::UnsupportedFeature
    );
    assert_eq!(
        path_error.details["feature_id"],
        "modifier.particle_instance.use_path"
    );
    let catalog = potter::catalog::feature_catalog();
    let path_feature = catalog["features"]
        .as_array()
        .and_then(|features| {
            features
                .iter()
                .find(|feature| feature["feature_id"] == "modifier.particle_instance.use_path")
        })
        .ok_or("sampled particle paths are missing from the capability catalog")?;
    assert_eq!(path_feature["status"], "not_supported");
    let baseline_instance = instance_source
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    let nontrivial_instance_displacement = blender_instance_points
        .iter()
        .map(|point| {
            baseline_instance
                .iter()
                .map(|baseline| point.distance(*baseline))
                .fold(f64::INFINITY, f64::min)
        })
        .fold(0.0, f64::max);
    assert!(
        nontrivial_instance_displacement > 10.0 * MODIFIER_POSITION_TOLERANCE,
        "particle instances must materially differ from the unmodified source",
    );

    let mut explode_params = Map::new();
    explode_params.insert("use_edge_cut".to_owned(), json!(true));
    let explode_source = instance_source.clone();
    let exploder_columns: [f64; 16] =
        serde_json::from_value(blender["exploder_world_matrix"].clone())?;
    let exploder_world = DMat4::from_cols_array(&exploder_columns);
    let direct_explode = potter::geom::modifiers::simulation::explode(
        &explode_source,
        &explode_particles,
        exploder_world,
        &explode_params,
        2.0,
        12.0,
    )?;
    assert_eq!(
        direct_explode.faces.len(),
        blender["explode_face_count"],
        "Potter vertices={}, Blender vertices={}, particles={explode_particles:?}",
        direct_explode.vertices.len(),
        blender["explode_world_points"]
            .as_array()
            .map_or(0, Vec::len),
    );
    let direct_explode_world = direct_explode
        .vertices
        .iter()
        .map(|vertex| exploder_world.transform_point3(vertex.co))
        .collect::<Vec<_>>();
    let blender_explode_world = blender_points(&blender["explode_world_points"])?;
    assert_eq!(
        direct_explode_world.len(),
        blender_explode_world.len(),
        "Potter explode vertices={}, Blender vertices={}, particles={explode_particles:?}",
        direct_explode_world.len(),
        blender_explode_world.len(),
    );
    let explode_error = max_geometric_match_error(&direct_explode_world, &blender_explode_world)?;
    assert!(
        explode_error <= MODIFIER_POSITION_TOLERANCE,
        "explode per-vertex geometric error {explode_error} exceeds {MODIFIER_POSITION_TOLERANCE}",
    );
    let mut cube_particles = injected_particles(&blender["cube_explode_particle_states"])?;
    let cube_columns: [f64; 16] =
        serde_json::from_value(blender["cube_exploder_world_matrix"].clone())?;
    let cube_world = DMat4::from_cols_array(&cube_columns);
    let cube_source = potter::geom::Mesh::from_positions_and_faces(
        vec![
            DVec3::new(-1.0, -1.0, -1.0),
            DVec3::new(1.0, -1.0, -1.0),
            DVec3::new(1.0, 1.0, -1.0),
            DVec3::new(-1.0, 1.0, -1.0),
            DVec3::new(-1.0, -1.0, 1.0),
            DVec3::new(1.0, -1.0, 1.0),
            DVec3::new(1.0, 1.0, 1.0),
            DVec3::new(-1.0, 1.0, 1.0),
        ],
        vec![
            vec![0, 3, 2, 1],
            vec![4, 5, 6, 7],
            vec![0, 1, 5, 4],
            vec![1, 2, 6, 5],
            vec![2, 3, 7, 6],
            vec![3, 0, 4, 7],
        ],
    )?;
    let cube_local_births = blender_points(&blender["cube_birth_local"])?;
    assert_eq!(cube_local_births.len(), cube_particles.len());
    for (particle, birth_position) in cube_particles.iter_mut().zip(&cube_local_births) {
        particle.birth_position = cube_world.transform_point3(*birth_position).to_array();
    }
    let cube_positions = cube_source
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.co))
        .collect::<std::collections::BTreeMap<_, _>>();
    let cube_face_owners = cube_source
        .faces
        .iter()
        .map(|face| {
            let points = face
                .vertices
                .iter()
                .map(|id| {
                    cube_positions.get(id).copied().ok_or_else(|| {
                        std::io::Error::other("cube face references a missing vertex").into()
                    })
                })
                .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
            let center = points.iter().copied().sum::<DVec3>() / points.len() as f64;
            let mut nearest = 0;
            let mut nearest_distance = f32::INFINITY;
            for (index, position) in cube_local_births.iter().enumerate() {
                let distance = center.as_vec3().distance_squared(position.as_vec3());
                if index == 0 || distance < nearest_distance {
                    nearest = index;
                    nearest_distance = distance;
                }
            }
            Ok(nearest)
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    assert_eq!(cube_face_owners, [0, 0, 0, 0, 0, 1]);
    let blender_cube_face_owners: Vec<usize> =
        serde_json::from_value(blender["cube_face_owners"].clone())?;
    assert_eq!(
        cube_face_owners, blender_cube_face_owners,
        "local births={cube_local_births:?}, world={cube_world:?}",
    );
    let cube_vertex_indices = cube_source
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut cube_vertex_owners = vec![cube_particles.len(); cube_source.vertices.len()];
    for (face, owner) in cube_source.faces.iter().zip(&cube_face_owners) {
        for vertex in &face.vertices {
            cube_vertex_owners[cube_vertex_indices[vertex]] = *owner;
        }
    }
    let cube_masks = cube_source
        .faces
        .iter()
        .map(|face| {
            let mut mask = 0_usize;
            for index in 0..face.vertices.len() {
                let first = face.vertices[index];
                let second = face.vertices[(index + 1) % face.vertices.len()];
                if cube_vertex_owners[cube_vertex_indices[&first]]
                    != cube_vertex_owners[cube_vertex_indices[&second]]
                {
                    mask |= 1_usize << index;
                }
            }
            mask
        })
        .collect::<Vec<_>>();
    let blender_cube_masks: Vec<usize> = serde_json::from_value(blender["cube_cut_masks"].clone())?;
    assert_eq!(cube_masks, [10, 5, 5, 0, 5, 0]);
    assert_eq!(
        cube_masks, blender_cube_masks,
        "face split masks must match Blender's particle-owner classification",
    );
    let cube_explode = potter::geom::modifiers::simulation::explode(
        &cube_source,
        &cube_particles,
        cube_world,
        &explode_params,
        2.0,
        12.0,
    )?;
    assert_eq!(cube_explode.faces.len(), 10);
    assert_eq!(
        cube_explode.faces.len(),
        blender["cube_explode_face_count"],
        "edge-cut cube face count, particles={cube_particles:?}, local births={cube_local_births:?}, owners={cube_face_owners:?}, masks={cube_masks:?}",
    );
    let cube_explode_world = cube_explode
        .vertices
        .iter()
        .map(|vertex| cube_world.transform_point3(vertex.co))
        .collect::<Vec<_>>();
    let blender_cube_points = blender_points(&blender["cube_explode_world_points"])?;
    assert_eq!(cube_explode.vertices.len(), 16);
    assert_eq!(
        cube_explode_world.len(),
        blender_cube_points.len(),
        "edge-cut cube vertices Potter={}, Blender={}, owners={cube_face_owners:?}, masks={cube_masks:?}",
        cube_explode_world.len(),
        blender_cube_points.len(),
    );
    let cube_error = max_geometric_match_error(&cube_explode_world, &blender_cube_points)?;
    assert!(
        cube_error <= MODIFIER_POSITION_TOLERANCE,
        "edge-cut cube vertex error {cube_error} exceeds {MODIFIER_POSITION_TOLERANCE}",
    );

    let scene_path = root.join("potter-scene");
    let batch_path = root.join("operations.json");
    initialize(&scene_path)?;
    apply(
        &scene_path,
        &batch_path,
        &json!([
            {"op":"node.create","id":"a_emitter","kind":"plane","params":{},"transform":{"translation":[3.0,-1.0,0.5],"scale":[1.3,0.8,1.1]}},
            {"op":"physics.particle_emitter.create","target":{"id":"a_emitter"},"settings":{"emit_from":"VERT","count":3,"frame_start":1.0,"frame_end":1.0,"lifetime":12.0,"physics_type":"NEWTON","normal_factor":0.8,"particle_size":0.7,"seed":3}},
            {"op":"node.create","id":"b_instance","kind":"plane","params":{}},
            {"op":"modifier.create","target":{"id":"b_instance"},"id":"particle_copy","type":"particle_instance","params":{"object":"a_emitter","particle_system_index":1,"use_normal":true,"use_children":false,"use_size":true,"show_alive":true,"show_dead":false,"show_unborn":false,"position":0,"random_position":0.0,"axis":"X","space":"WORLD","use_path":false}},
            {"op":"node.create","id":"c_exploder","kind":"box","params":{}},
            {"op":"physics.particle_emitter.create","target":{"id":"c_exploder"},"settings":{"emit_from":"VERT","count":2,"frame_start":1.0,"frame_end":1.0,"lifetime":12.0,"physics_type":"NEWTON","normal_factor":1.1,"seed":3}},
            {"op":"modifier.create","target":{"id":"c_exploder"},"id":"explode_faces","type":"explode","params":{"use_edge_cut":true,"show_alive":true,"show_dead":false,"show_unborn":false,"use_size":false,"protect":0.0}}
        ]),
        0,
    )?;
    let instance_output = run(
        pot()
            .arg("inspect")
            .arg(&scene_path)
            .arg("--id")
            .arg("b_instance")
            .arg("--frame")
            .arg("1")
            .arg("--json"),
        "particle instance inspection",
    )?;
    let explode_output = run(
        pot()
            .arg("inspect")
            .arg(&scene_path)
            .arg("--id")
            .arg("c_exploder")
            .arg("--frame")
            .arg("1")
            .arg("--json"),
        "explode inspection",
    )?;
    let instance_json: Value = serde_json::from_slice(&instance_output.stdout)?;
    let explode_json: Value = serde_json::from_slice(&explode_output.stdout)?;
    let item = &instance_json["result"]["items"][0];
    let minimum: [f64; 3] = serde_json::from_value(item["bounds"]["min"].clone())?;
    let maximum: [f64; 3] = serde_json::from_value(item["bounds"]["max"].clone())?;
    let extent = DVec3::from_array(maximum).distance(DVec3::from_array(minimum));
    assert!(
        extent > 1.0,
        "CLI particle instance should contain normal-sized, spatially separated copies",
    );
    let exploded_faces = explode_json["result"]["items"][0]["evaluated_geometry"]["face_count"]
        .as_u64()
        .ok_or("CLI explode face count is missing")?;
    assert!(
        exploded_faces > 6,
        "CLI explode should split the six source cube faces: {exploded_faces}",
    );
    Ok(())
}

#[test]
fn particle_emission_counts_face_distribution_and_velocity_statistics_match_blender()
-> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping particle emission Blender parity; Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let script = root.join("particle_emission.py");
    let expected_path = root.join("particle_emission.json");
    fs::write(
        &script,
        r"
import bpy, json, os, sys
root = sys.argv[sys.argv.index('--') + 1]
scene = bpy.context.scene
scene.use_gravity = False
mesh = bpy.data.meshes.new('unequal_face_emitter')
mesh.from_pydata(
    [(-5,-2,0), (-1,-2,0), (-1,2,0), (-5,2,0),
     (1,-0.5,0), (2,-0.5,0), (2,0.5,0), (1,0.5,0)],
    [],
    [(0,1,2,3), (4,5,6,7)],
)
emitter = bpy.data.objects.new('Emitter', mesh)
scene.collection.objects.link(emitter)
emitter.location = (1.7,-0.3,0.2)
emitter.rotation_euler = (0.0,-1.2,0.0)
emitter.scale = (1.2,1.2,1.2)
bpy.context.view_layer.objects.active = emitter
emitter.select_set(True)
bpy.ops.object.particle_system_add()
system = emitter.particle_systems[0]
system.seed = 37
settings = system.settings
settings.count = 1000
settings.frame_start = 1
settings.frame_end = 10
settings.lifetime = 20
settings.emit_from = 'FACE'
settings.physics_type = 'NEWTON'
settings.normal_factor = 3.5
settings.factor_random = 0.0
frames = {}
for frame in [1, 4, 7, 10, 12]:
    scene.frame_set(frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    evaluated = emitter.evaluated_get(depsgraph)
    particles = evaluated.particle_systems[0].particles
    frames[str(frame)] = {
        'positions':[list(particle.location) for particle in particles],
        'velocities':[list(particle.velocity) for particle in particles],
        'birth_frames':[particle.birth_time for particle in particles],
        'life_states':[particle.alive_state for particle in particles],
    }
with open(os.path.join(root, 'particle_emission.json'), 'w', encoding='utf-8') as output:
    json.dump(frames, output)
",
    )?;
    run(
        Command::new(&blender)
            .args(["--background", "--factory-startup", "--python"])
            .arg(&script)
            .arg("--")
            .arg(root),
        "particle emission fixture",
    )?;
    let blender_frames: Value = serde_json::from_slice(&fs::read(expected_path)?)?;
    let mesh = potter::geom::Mesh::from_positions_and_faces(
        vec![
            DVec3::new(-5.0, -2.0, 0.0),
            DVec3::new(-1.0, -2.0, 0.0),
            DVec3::new(-1.0, 2.0, 0.0),
            DVec3::new(-5.0, 2.0, 0.0),
            DVec3::new(1.0, -0.5, 0.0),
            DVec3::new(2.0, -0.5, 0.0),
            DVec3::new(2.0, 0.5, 0.0),
            DVec3::new(1.0, 0.5, 0.0),
        ],
        vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
    )?;
    let emitter_world = DMat4::from_scale_rotation_translation(
        DVec3::splat(1.2),
        DQuat::from_euler(EulerRot::XYZ, 0.0, -1.2, 0.0),
        DVec3::new(1.7, -0.3, 0.2),
    );
    let inverse_emitter = emitter_world.inverse();
    let emitter_normal = emitter_world.transform_vector3(DVec3::Z).normalize();
    let settings = json!({
        "count":1000,
        "frame_start":1.0,
        "frame_end":10.0,
        "lifetime":20.0,
        "emit_from":"FACE",
        "physics_type":"NEWTON",
        "normal_factor":3.5,
        "factor_random":0.0,
        "seed":37
    });
    for frame_number in [1_u32, 4, 7, 10, 12] {
        let frame = f64::from(frame_number);
        let blender_frame = &blender_frames[frame_number.to_string()];
        let blender_positions = blender_points(&blender_frame["positions"])?;
        let blender_velocities = blender_points(&blender_frame["velocities"])?;
        let potter_particles = potter::sim::particles::simulate(
            &mesh,
            &settings,
            emitter_world,
            frame,
            24.0,
            DVec3::ZERO,
            37,
            &[],
            &[],
        )?;
        let blender_emitted_count = blender_frame["life_states"]
            .as_array()
            .ok_or("Blender particle life states are missing")?
            .iter()
            .filter(|life_state| life_state.as_str().is_some_and(|state| state != "UNBORN"))
            .count();
        assert!(
            potter_particles.len().abs_diff(blender_emitted_count) <= 2,
            "emitted particle count at frame {frame}: Potter={}, Blender={blender_emitted_count}",
            potter_particles.len(),
        );
        if frame_number == 10 {
            let fraction = |count: usize, total: usize| -> Result<f64, Box<dyn Error>> {
                if total == 0 {
                    return Err("particle face distribution is empty".into());
                }
                Ok(f64::from(u32::try_from(count)?) / f64::from(u32::try_from(total)?))
            };
            let blender_fraction = fraction(
                blender_positions
                    .iter()
                    .filter(|position| inverse_emitter.transform_point3(**position).x < 0.0)
                    .count(),
                blender_positions.len(),
            )?;
            let potter_fraction = fraction(
                potter_particles
                    .iter()
                    .filter(|particle| {
                        inverse_emitter
                            .transform_point3(DVec3::from_array(particle.position))
                            .x
                            < 0.0
                    })
                    .count(),
                potter_particles.len(),
            )?;
            let expected_large_face_fraction = 16.0 / 17.0;
            assert!(
                (blender_fraction - expected_large_face_fraction).abs()
                    <= FACE_DISTRIBUTION_TOLERANCE,
                "Blender large-face emission fraction is {blender_fraction}",
            );
            assert!(
                (potter_fraction - expected_large_face_fraction).abs()
                    <= FACE_DISTRIBUTION_TOLERANCE,
                "Potter large-face emission fraction is {potter_fraction}",
            );
            assert!(
                (potter_fraction - blender_fraction).abs() <= FACE_DISTRIBUTION_TOLERANCE,
                "large-face emission differs: Potter={potter_fraction}, Blender={blender_fraction}",
            );
            assert!(
                (blender_fraction - 0.5).abs() > 10.0 * FACE_DISTRIBUTION_TOLERANCE,
                "unequal face areas must distinguish area-weighted from uniform-face sampling",
            );
            let mean_normal_velocity = |values: &[DVec3]| -> Result<f64, Box<dyn Error>> {
                if values.is_empty() {
                    return Err("particle velocities are empty".into());
                }
                let count = u32::try_from(values.len())?;
                Ok(values
                    .iter()
                    .map(|velocity| velocity.dot(emitter_normal))
                    .sum::<f64>()
                    / f64::from(count))
            };
            let mean_speed = |values: &[DVec3]| -> Result<f64, Box<dyn Error>> {
                if values.is_empty() {
                    return Err("particle velocities are empty".into());
                }
                let count = u32::try_from(values.len())?;
                Ok(values.iter().map(|velocity| velocity.length()).sum::<f64>() / f64::from(count))
            };
            let potter_velocities = potter_particles
                .iter()
                .map(|particle| DVec3::from_array(particle.velocity))
                .collect::<Vec<_>>();
            let blender_normal_velocity = mean_normal_velocity(&blender_velocities)?;
            let potter_normal_velocity = mean_normal_velocity(&potter_velocities)?;
            let blender_speed = mean_speed(&blender_velocities)?;
            let potter_speed = mean_speed(&potter_velocities)?;
            assert!(
                blender_normal_velocity > 10.0 * VELOCITY_STATISTIC_TOLERANCE,
                "Blender emitter must produce nonzero normal velocity, mean={blender_normal_velocity}",
            );
            assert!(
                (blender_normal_velocity - potter_normal_velocity).abs()
                    <= VELOCITY_STATISTIC_TOLERANCE,
                "mean world-space normal particle velocity differs: Potter={potter_normal_velocity}, Blender={blender_normal_velocity}",
            );
            assert!(
                (blender_speed - potter_speed).abs() <= VELOCITY_STATISTIC_TOLERANCE,
                "mean world-space particle speed differs: Potter={potter_speed}, Blender={blender_speed}",
            );
        }
    }
    Ok(())
}

fn mean_z(mesh: &potter::geom::Mesh) -> Result<f64, Box<dyn Error>> {
    let vertex_count = u32::try_from(mesh.vertices.len())?;
    if vertex_count == 0 {
        return Err("cloth mesh has no vertices".into());
    }
    Ok(mesh.vertices.iter().map(|vertex| vertex.co.z).sum::<f64>() / f64::from(vertex_count))
}
fn blender_points(value: &Value) -> Result<Vec<glam::DVec3>, Box<dyn Error>> {
    let coordinates: Vec<[f64; 3]> = serde_json::from_value(value.clone())?;
    Ok(coordinates
        .into_iter()
        .map(glam::DVec3::from_array)
        .collect())
}

// Subdivision backends may number corresponding vertices differently. The fixture deforms
// vertically, so projected rest-space coordinates give a stable one-to-one correspondence.
fn max_pointwise_distance(
    first: &[glam::DVec3],
    second: &[glam::DVec3],
) -> Result<f64, Box<dyn Error>> {
    if first.len() != second.len() {
        return Err(format!(
            "point arrays have different lengths: {} and {}",
            first.len(),
            second.len(),
        )
        .into());
    }
    let mut unmatched = second.to_vec();
    let mut maximum = 0.0_f64;
    for point in first {
        let nearest = unmatched
            .iter()
            .enumerate()
            .min_by(|(_, first), (_, second)| {
                let first_delta = first.truncate() - point.truncate();
                let second_delta = second.truncate() - point.truncate();
                first_delta
                    .length_squared()
                    .total_cmp(&second_delta.length_squared())
            })
            .map(|(index, _)| index)
            .ok_or("point correspondence exhausted unexpectedly")?;
        maximum = maximum.max(point.distance(unmatched.swap_remove(nearest)));
    }
    Ok(maximum)
}
fn pointwise_error_metrics(
    first: &[glam::DVec3],
    second: &[glam::DVec3],
) -> Result<(f64, f64), Box<dyn Error>> {
    if first.len() != second.len() || first.is_empty() {
        return Err("point arrays must have matching nonzero lengths".into());
    }
    let mut unmatched = second.to_vec();
    let mut squared_error_sum = 0.0;
    let mut maximum_error = 0.0_f64;
    for point in first {
        let nearest = unmatched
            .iter()
            .enumerate()
            .min_by(|(_, first), (_, second)| {
                let first_delta = first.truncate() - point.truncate();
                let second_delta = second.truncate() - point.truncate();
                first_delta
                    .length_squared()
                    .total_cmp(&second_delta.length_squared())
            })
            .map(|(index, _)| index)
            .ok_or("point correspondence exhausted unexpectedly")?;
        let error = point.distance(unmatched.swap_remove(nearest));
        squared_error_sum += error * error;
        maximum_error = maximum_error.max(error);
    }
    let count = f64::from(u32::try_from(first.len())?);
    Ok(((squared_error_sum / count).sqrt(), maximum_error))
}

fn centroid(points: &[glam::DVec3]) -> Result<glam::DVec3, Box<dyn Error>> {
    let count = f64::from(u32::try_from(points.len())?);
    if count == 0.0 {
        return Err("point array is empty".into());
    }
    Ok(points.iter().copied().sum::<glam::DVec3>() / count)
}
fn max_corresponding_distance(
    first: &[glam::DVec3],
    second: &[glam::DVec3],
) -> Result<f64, Box<dyn Error>> {
    if first.len() != second.len() {
        return Err(format!(
            "point arrays have different lengths: {} and {}",
            first.len(),
            second.len(),
        )
        .into());
    }
    Ok(first
        .iter()
        .zip(second)
        .map(|(first, second)| first.distance(*second))
        .fold(0.0, f64::max))
}

fn max_geometric_match_error(
    first: &[glam::DVec3],
    second: &[glam::DVec3],
) -> Result<f64, Box<dyn Error>> {
    if first.len() != second.len() {
        return Err(format!(
            "point arrays have different lengths: {} and {}",
            first.len(),
            second.len(),
        )
        .into());
    }
    let mut unmatched = second.to_vec();
    let mut maximum = 0.0_f64;
    for point in first {
        let nearest = unmatched
            .iter()
            .enumerate()
            .min_by(|(_, first), (_, second)| {
                first
                    .distance_squared(*point)
                    .total_cmp(&second.distance_squared(*point))
            })
            .map(|(index, _)| index)
            .ok_or("point correspondence exhausted unexpectedly")?;
        maximum = maximum.max(point.distance(unmatched.swap_remove(nearest)));
    }
    Ok(maximum)
}
// Different volume polygonizers do not share vertex indices; nearest-point distances compare
// every sampled surface vertex in both directions without reducing the surface to bounds.
fn max_surface_correspondence_error(
    first: &[glam::DVec3],
    second: &[glam::DVec3],
) -> Result<f64, Box<dyn Error>> {
    if first.is_empty() || second.is_empty() {
        return Err("evaluated volume surface has no vertices".into());
    }
    let directed_distance = |source: &[glam::DVec3], target: &[glam::DVec3]| {
        source
            .iter()
            .map(|point| {
                target
                    .iter()
                    .map(|other| point.distance(*other))
                    .fold(f64::INFINITY, f64::min)
            })
            .fold(0.0, f64::max)
    };
    Ok(directed_distance(first, second).max(directed_distance(second, first)))
}
#[test]
fn subdivision_surface_matches_blender_for_cube_plane_and_creased_open_grid()
-> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping subdivision Blender parity; Blender is unavailable");
        return Ok(());
    };
    let plane = potter::geom::Mesh::from_positions_and_faces(
        vec![
            DVec3::new(-1.5, -0.75, 0.0),
            DVec3::new(1.5, -0.75, 0.0),
            DVec3::new(1.5, 0.75, 0.0),
            DVec3::new(-1.5, 0.75, 0.0),
        ],
        vec![vec![0, 1, 2, 3]],
    )?;
    let cube = potter::geom::Mesh::from_positions_and_faces(
        vec![
            DVec3::new(-1.0, -1.5, -0.7),
            DVec3::new(1.0, -1.5, -0.7),
            DVec3::new(1.0, 1.5, -0.7),
            DVec3::new(-1.0, 1.5, -0.7),
            DVec3::new(-1.0, -1.5, 0.7),
            DVec3::new(1.0, -1.5, 0.7),
            DVec3::new(1.0, 1.5, 0.7),
            DVec3::new(-1.0, 1.5, 0.7),
        ],
        vec![
            vec![0, 3, 2, 1],
            vec![4, 5, 6, 7],
            vec![0, 1, 5, 4],
            vec![1, 2, 6, 5],
            vec![2, 3, 7, 6],
            vec![3, 0, 4, 7],
        ],
    )?;
    let mut open_grid = potter::geom::Mesh::grid(potter::geom::GridParams {
        size_x: 2.0,
        size_y: 1.5,
        x_subdivisions: 3,
        y_subdivisions: 3,
    })?;
    for vertex in &mut open_grid.vertices {
        vertex.co.z = 0.15 * vertex.co.x * vertex.co.x + 0.1 * vertex.co.x * vertex.co.y;
    }
    let centerline_crease = open_grid
        .edges
        .iter()
        .find(|edge| {
            let first = open_grid
                .vertices
                .iter()
                .find(|vertex| vertex.id == edge.vertices[0])
                .map(|vertex| vertex.co);
            let second = open_grid
                .vertices
                .iter()
                .find(|vertex| vertex.id == edge.vertices[1])
                .map(|vertex| vertex.co);
            matches!((first, second), (Some(first), Some(second))
                if first.y.abs() <= f64::EPSILON
                    && second.y.abs() <= f64::EPSILON
                    && (first.x - second.x).abs() > f64::EPSILON)
        })
        .ok_or("creased grid has no centerline edge")?;
    let mut crease_values = Map::new();
    crease_values.insert(format!("e{}", centerline_crease.id), json!(0.85));
    open_grid.attributes.insert(
        "crease_edge".to_owned(),
        json!({"domain":"edge","type":"float","values":crease_values}),
    );

    let cases = vec![
        (
            "plane_level_one",
            plane.clone(),
            false,
            json!({
                "levels":1,
                "render_levels":2,
                "use_limit_surface":true,
                "boundary_smooth":"ALL",
                "quality":4,
                "uv_smooth":"PRESERVE_BOUNDARIES",
                "use_creases":false
            }),
        ),
        (
            "plane",
            plane,
            true,
            json!({
                "levels":1,
                "render_levels":2,
                "use_limit_surface":true,
                "boundary_smooth":"ALL",
                "quality":4,
                "uv_smooth":"PRESERVE_BOUNDARIES",
                "use_creases":false
            }),
        ),
        (
            "cube",
            cube,
            false,
            json!({
                "levels":2,
                "render_levels":1,
                "use_limit_surface":false,
                "boundary_smooth":"PRESERVE_CORNERS",
                "quality":6,
                "uv_smooth":"NONE",
                "use_creases":false
            }),
        ),
        (
            "open_grid",
            open_grid,
            false,
            json!({
                "levels":1,
                "render_levels":2,
                "use_limit_surface":true,
                "boundary_smooth":"PRESERVE_CORNERS",
                "quality":5,
                "uv_smooth":"PRESERVE_BOUNDARIES",
                "use_creases":true
            }),
        ),
    ];
    let mut inputs = Map::new();
    for (name, mesh, use_render_levels, params) in &cases {
        let vertex_indices = mesh
            .vertices
            .iter()
            .enumerate()
            .map(|(index, vertex)| (vertex.id, index))
            .collect::<std::collections::BTreeMap<_, _>>();
        let vertices = mesh
            .vertices
            .iter()
            .map(|vertex| vertex.co.to_array())
            .collect::<Vec<_>>();
        let faces = mesh
            .faces
            .iter()
            .map(|face| {
                face.vertices
                    .iter()
                    .map(|id| {
                        vertex_indices.get(id).copied().ok_or_else(|| {
                            std::io::Error::other(
                                "subdivision fixture face references a missing vertex",
                            )
                            .into()
                        })
                    })
                    .collect::<Result<Vec<_>, Box<dyn Error>>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let crease_attribute = mesh
            .attributes
            .get("crease_edge")
            .and_then(|attribute| attribute.get("values"))
            .and_then(Value::as_object);
        let mut creases = Vec::new();
        for edge in &mesh.edges {
            let Some(value) = crease_attribute
                .and_then(|values| values.get(&format!("e{}", edge.id)))
                .and_then(Value::as_f64)
            else {
                continue;
            };
            creases.push(json!([
                vertex_indices[&edge.vertices[0]],
                vertex_indices[&edge.vertices[1]],
                value
            ]));
        }
        inputs.insert(
            (*name).to_owned(),
            json!({
                "vertices":vertices,
                "faces":faces,
                "creases":creases,
                "params":params,
                "use_render_levels":use_render_levels
            }),
        );
    }

    let directory = tempdir()?;
    let root = directory.path();
    let script = root.join("subdivision_parity.py");
    let input_path = root.join("subdivision_inputs.json");
    let expected_path = root.join("subdivision_expected.json");
    fs::write(&input_path, serde_json::to_vec(&Value::Object(inputs))?)?;
    fs::write(
        &script,
        r"
import bpy, json, os, sys
root = sys.argv[sys.argv.index('--') + 1]
with open(os.path.join(root, 'subdivision_inputs.json'), encoding='utf-8') as source:
    inputs = json.load(source)
results = {}
for name, spec in inputs.items():
    mesh = bpy.data.meshes.new(name + 'Mesh')
    mesh.from_pydata(spec['vertices'], [], spec['faces'])
    mesh.update()
    if spec['creases']:
        crease = mesh.attributes.new(name='crease_edge', type='FLOAT', domain='EDGE')
        values = {tuple(sorted(item[:2])):item[2] for item in spec['creases']}
        for edge in mesh.edges:
            crease.data[edge.index].value = values.get(tuple(sorted(edge.vertices)), 0.0)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    params = spec['params']
    modifier = obj.modifiers.new('Subdivision', 'SUBSURF')
    modifier.subdivision_type = 'CATMULL_CLARK'
    modifier.levels = params['render_levels'] if spec['use_render_levels'] else params['levels']
    modifier.render_levels = params['render_levels']
    modifier.use_limit_surface = params['use_limit_surface']
    modifier.boundary_smooth = params['boundary_smooth']
    modifier.quality = params['quality']
    modifier.uv_smooth = params['uv_smooth']
    modifier.use_creases = params['use_creases']
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    evaluated = obj.evaluated_get(depsgraph)
    result_mesh = evaluated.to_mesh()
    results[name] = {
        'points':[list(vertex.co) for vertex in result_mesh.vertices],
        'face_count':len(result_mesh.polygons),
    }
    evaluated.to_mesh_clear()
with open(os.path.join(root, 'subdivision_expected.json'), 'w', encoding='utf-8') as output:
    json.dump(results, output)
",
    )?;
    run(
        Command::new(&blender)
            .args(["--background", "--factory-startup", "--python"])
            .arg(&script)
            .arg("--")
            .arg(root),
        "subdivision fixture",
    )?;
    let expected: Value = serde_json::from_slice(&fs::read(expected_path)?)?;
    for (name, mesh, use_render_levels, params) in cases {
        let params: Map<String, Value> = serde_json::from_value(params)?;
        let mut modifier = potter::model::Modifier {
            id: potter::model::Id::new("subdivision".to_owned())?,
            modifier_type: "subdivision".to_owned(),
            name: "Subdivision".to_owned(),
            enabled: true,
            params,
            binding_data: None,
            runtime: potter::model::ModifierRuntime::default(),
        };
        modifier.runtime.use_render_levels = use_render_levels;
        let evaluated = potter::geom::modifiers::evaluate_modifiers(&mesh, &[modifier])?;
        let blender_positions = blender_points(&expected[name]["points"])?;
        assert_eq!(
            evaluated.vertices.len(),
            blender_positions.len(),
            "{name} vertex count"
        );
        assert_eq!(
            evaluated.faces.len(),
            expected[name]["face_count"],
            "{name} face count",
        );
        let potter_positions = evaluated
            .vertices
            .iter()
            .map(|vertex| vertex.co)
            .collect::<Vec<_>>();
        let error = max_geometric_match_error(&potter_positions, &blender_positions)?;
        assert!(
            error <= SUBDIVISION_POSITION_TOLERANCE,
            "{name} per-vertex error {error} exceeds {SUBDIVISION_POSITION_TOLERANCE}; Potter={potter_positions:?}; Blender={blender_positions:?}",
        );
        let original_positions = mesh
            .vertices
            .iter()
            .map(|vertex| vertex.co)
            .collect::<Vec<_>>();
        let stimulus = blender_positions
            .iter()
            .map(|point| {
                original_positions
                    .iter()
                    .map(|original| point.distance(*original))
                    .fold(f64::INFINITY, f64::min)
            })
            .fold(0.0, f64::max);
        assert!(
            stimulus > 10.0 * SUBDIVISION_POSITION_TOLERANCE,
            "{name} subdivision stimulus is too close to the unmodified mesh: {stimulus}",
        );
    }
    Ok(())
}
#[test]

fn subdivision_and_cloth_stack_order_match_blender_at_frame_one() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping cloth stack Blender parity; Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let script_path = root.join("cloth_stack.py");
    let expected_path = root.join("cloth_stack.json");
    fs::write(
        &script_path,
        r"
import bpy, json, os, sys
root = sys.argv[sys.argv.index('--') + 1]
scene = bpy.context.scene
scene.frame_start = 1

def add_cloth(name, subdivision_first):
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0,0,0))
    obj = bpy.context.object
    obj.name = name
    obj.data.vertices[1].co.z = 0.4
    obj.data.vertices[3].co.z = -0.2
    group = obj.vertex_groups.new(name='Pin')
    group.add([0,2], 1.0, 'REPLACE')
    if subdivision_first:
        subd = obj.modifiers.new('Subdivide before cloth', 'SUBSURF')
        subd.levels = 1
    cloth = obj.modifiers.new('Cloth', 'CLOTH')
    cloth.settings.quality = 1
    cloth.settings.mass = 1.0
    cloth.settings.air_damping = 0.01
    cloth.settings.tension_stiffness = 12.0
    cloth.settings.compression_stiffness = 12.0
    cloth.settings.shear_stiffness = 5.0
    cloth.settings.bending_stiffness = 0.5
    cloth.settings.vertex_group_mass = group.name
    if not subdivision_first:
        subd = obj.modifiers.new('Subdivide after cloth', 'SUBSURF')
        subd.levels = 1
    return obj

before = add_cloth('SubdivisionFirst', True)
after = add_cloth('ClothFirst', False)
def scene_stats():
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    result = {}
    for key, obj in [('subdivision_first', before), ('cloth_first', after)]:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        points = [evaluated.matrix_world @ vertex.co for vertex in mesh.vertices]
        result[key] = {
            'vertex_count':len(mesh.vertices),
            'face_count':len(mesh.polygons),
            'points':[list(point) for point in points],
            'centroid':[sum(point[axis] for point in points)/len(points) for axis in range(3)],
            'bounds_min_z':min(point[2] for point in points),
            'bounds_max_z':max(point[2] for point in points),
        }
        evaluated.to_mesh_clear()
    return result

scene.frame_set(1)
frame_one = scene_stats()
for frame in range(2, 10):
    scene.frame_set(frame)
    scene_stats()
frame_nine = scene_stats()
with open(os.path.join(root, 'cloth_stack.json'), 'w', encoding='utf-8') as output:
    json.dump({'frame_one':frame_one,'frame_nine':frame_nine}, output)
",
    )?;
    let blender_output = Command::new(&blender)
        .args(["--background", "--factory-startup", "--python"])
        .arg(&script_path)
        .arg("--")
        .arg(root)
        .output()?;
    if !blender_output.status.success() || !expected_path.is_file() {
        return Err(format!(
            "Blender cloth stack fixture failed: stdout={} stderr={}",
            String::from_utf8_lossy(&blender_output.stdout),
            String::from_utf8_lossy(&blender_output.stderr),
        )
        .into());
    }
    let blender_values: Value = serde_json::from_slice(&fs::read(expected_path)?)?;
    let scene = root.join("cloth-stack-project");
    let batch = root.join("cloth-operations.json");
    initialize(&scene)?;
    apply(
        &scene,
        &batch,
        &json!([
            {"op":"node.create","id":"a_subdivision_first","kind":"plane","params":{}},
            {"op":"mesh.transform_elements","target":{"id":"a_subdivision_first"},"elements":{"domain":"vertex","ids":["v1"]},"translation":[0.0,0.0,0.4]},
            {"op":"mesh.transform_elements","target":{"id":"a_subdivision_first"},"elements":{"domain":"vertex","ids":["v2"]},"translation":[0.0,0.0,-0.2]},
            {"op":"vertex_group.create","target":{"id":"a_subdivision_first"},"id":"pin","name":"Pin"},
            {"op":"vertex_group.assign","target":{"id":"a_subdivision_first"},"group_id":"pin","weights":[{"vertex_id":0,"weight":1.0},{"vertex_id":3,"weight":1.0}]},
            {"op":"modifier.create","target":{"id":"a_subdivision_first"},"id":"subd","type":"subdivision","params":{"levels":1}},
            {"op":"physics.cloth.create","target":{"id":"a_subdivision_first"},"settings":{"vertex_group_mass":"Pin","quality":1,"tension_stiffness":12.0,"compression_stiffness":12.0,"mass":1.0,"air_damping":0.01,"shear_stiffness":5.0,"bending_stiffness":0.5}},
            {"op":"node.create","id":"b_cloth_first","kind":"plane","params":{}},
            {"op":"mesh.transform_elements","target":{"id":"b_cloth_first"},"elements":{"domain":"vertex","ids":["v1"]},"translation":[0.0,0.0,0.4]},
            {"op":"mesh.transform_elements","target":{"id":"b_cloth_first"},"elements":{"domain":"vertex","ids":["v2"]},"translation":[0.0,0.0,-0.2]},
            {"op":"vertex_group.create","target":{"id":"b_cloth_first"},"id":"pin","name":"Pin"},
            {"op":"vertex_group.assign","target":{"id":"b_cloth_first"},"group_id":"pin","weights":[{"vertex_id":0,"weight":1.0},{"vertex_id":3,"weight":1.0}]},
            {"op":"physics.cloth.create","target":{"id":"b_cloth_first"},"settings":{"vertex_group_mass":"Pin","quality":1,"tension_stiffness":12.0,"compression_stiffness":12.0,"mass":1.0,"air_damping":0.01,"shear_stiffness":5.0,"bending_stiffness":0.5}},
            {"op":"modifier.create","target":{"id":"b_cloth_first"},"id":"subd","type":"subdivision","params":{"levels":1}}
        ]),
        0,
    )?;
    let document: potter::model::SceneDoc =
        serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let snapshot = potter::eval::Snapshot::evaluate(
        &document,
        &potter::eval::EvaluationContext {
            frame: Some(1.0),
            ..potter::eval::EvaluationContext::default()
        },
    )?;
    for (potter_id, blender_key) in [
        ("a_subdivision_first", "subdivision_first"),
        ("b_cloth_first", "cloth_first"),
    ] {
        let mesh = snapshot
            .meshes
            .get(&potter::model::Id::new(potter_id.to_owned())?)
            .ok_or("evaluated cloth mesh is missing")?;
        let expected = &blender_values["frame_one"][blender_key];
        assert_eq!(
            mesh.vertices.len(),
            expected["vertex_count"],
            "{potter_id} frame-one vertices"
        );
        assert_eq!(
            mesh.faces.len(),
            expected["face_count"],
            "{potter_id} frame-one faces"
        );
        let actual_points = mesh
            .vertices
            .iter()
            .map(|vertex| vertex.co)
            .collect::<Vec<_>>();
        let blender_positions = blender_points(&expected["points"])?;
        let rest_error = max_pointwise_distance(&actual_points, &blender_positions)?;
        assert!(
            rest_error <= 1.0e-5,
            "{potter_id} rest-frame vertex error {rest_error}; Potter={actual_points:?}; Blender={blender_positions:?}",
        );
        let centroid_z = mean_z(mesh)?;
        let blender_centroid = expected["centroid"][2]
            .as_f64()
            .ok_or("Blender cloth centroid is missing")?;
        assert!(
            (centroid_z - blender_centroid).abs() <= 0.02,
            "{potter_id} frame-one centroid: Potter={centroid_z}, Blender={blender_centroid}",
        );
    }
    let snapshot_nine = potter::eval::Snapshot::evaluate(
        &document,
        &potter::eval::EvaluationContext {
            frame: Some(9.0),
            ..potter::eval::EvaluationContext::default()
        },
    )?;
    // Blender's implicit cloth solver is not numerically equivalent to Potter's XPBD
    // constraints. Apply the fixture-specific aggregate tolerances above while retaining
    // exact rest-state/pin checks and independent stack-order sensitivity.
    for (potter_id, blender_key) in [
        ("a_subdivision_first", "subdivision_first"),
        ("b_cloth_first", "cloth_first"),
    ] {
        let mesh = snapshot_nine
            .meshes
            .get(&potter::model::Id::new(potter_id.to_owned())?)
            .ok_or("frame-nine cloth mesh is missing")?;
        let expected = &blender_values["frame_nine"][blender_key];
        assert_eq!(mesh.vertices.len(), expected["vertex_count"]);
        assert_eq!(mesh.faces.len(), expected["face_count"]);
        let actual_points = mesh
            .vertices
            .iter()
            .map(|vertex| vertex.co)
            .collect::<Vec<_>>();
        let blender_positions = blender_points(&expected["points"])?;
        let frame_one_positions =
            blender_points(&blender_values["frame_one"][blender_key]["points"])?;
        let baseline_displacement =
            max_pointwise_distance(&blender_positions, &frame_one_positions)?;
        assert!(
            baseline_displacement > 10.0 * CLOTH_STIMULUS_TOLERANCE,
            "{potter_id} fixture must produce nontrivial cloth motion, observed {baseline_displacement}",
        );
        let (rms_error, maximum_error) =
            pointwise_error_metrics(&actual_points, &blender_positions)?;
        assert!(
            rms_error <= CLOTH_RMS_TOLERANCE,
            "{potter_id} frame-nine per-vertex RMS={rms_error}, max={maximum_error}; RMS tolerance={CLOTH_RMS_TOLERANCE}",
        );
        let blender_centroid: [f64; 3] = serde_json::from_value(expected["centroid"].clone())?;
        let centroid_error =
            centroid(&actual_points)?.distance(glam::DVec3::from_array(blender_centroid));
        assert!(
            centroid_error <= CLOTH_CENTROID_TOLERANCE,
            "{potter_id} frame-nine centroid error {centroid_error} exceeds {CLOTH_CENTROID_TOLERANCE}",
        );
        let actual_min_z = actual_points
            .iter()
            .map(|point| point.z)
            .fold(f64::INFINITY, f64::min);
        let actual_max_z = actual_points
            .iter()
            .map(|point| point.z)
            .fold(f64::NEG_INFINITY, f64::max);
        let minimum_z = expected["bounds_min_z"]
            .as_f64()
            .ok_or("Blender cloth minimum Z bound is missing")?;
        let maximum_z = expected["bounds_max_z"]
            .as_f64()
            .ok_or("Blender cloth maximum Z bound is missing")?;
        let bounds_error = (actual_min_z - minimum_z)
            .abs()
            .max((actual_max_z - maximum_z).abs());
        assert!(
            bounds_error <= CLOTH_BOUNDS_TOLERANCE,
            "{potter_id} frame-nine Z-bounds error {bounds_error} exceeds {CLOTH_BOUNDS_TOLERANCE}",
        );
        if potter_id == "a_subdivision_first" {
            let frame_one_mesh = &snapshot.meshes[&potter::model::Id::new(potter_id.to_owned())?];
            let pinned_vertex_error = actual_points[0].distance(frame_one_mesh.vertices[0].co);
            assert!(
                pinned_vertex_error <= 1.0e-8,
                "subdivision-first pinned vertex moved {pinned_vertex_error}",
            );
        }
    }
    let blender_first =
        blender_points(&blender_values["frame_nine"]["subdivision_first"]["points"])?;
    let blender_second = blender_points(&blender_values["frame_nine"]["cloth_first"]["points"])?;
    let blender_order_difference = max_pointwise_distance(&blender_first, &blender_second)?;
    assert!(
        blender_order_difference > CLOTH_ORDER_TOLERANCE,
        "Blender fixture must make stack order observable, observed {blender_order_difference}",
    );
    let first = &snapshot_nine.meshes[&potter::model::Id::new("a_subdivision_first".to_owned())?];
    let second = &snapshot_nine.meshes[&potter::model::Id::new("b_cloth_first".to_owned())?];
    let first_points = first
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    let second_points = second
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    let potter_order_difference = max_pointwise_distance(&first_points, &second_points)?;
    assert!(
        potter_order_difference > CLOTH_ORDER_TOLERANCE,
        "Potter stack order must change evaluated positions, observed {potter_order_difference}",
    );
    let catalog = potter::catalog::feature_catalog();
    let cloth_parity = catalog["features"]
        .as_array()
        .and_then(|features| {
            features
                .iter()
                .find(|feature| feature["feature_id"] == "physics.cloth.blender_solver_parity")
        })
        .ok_or("Blender cloth-solver parity is missing from the capability catalog")?;
    assert_eq!(cloth_parity["status"], "not_supported");
    Ok(())
}
fn inspect(scene: &Path, object: &str, frame: &str) -> Result<Value, Box<dyn Error>> {
    let output = run(
        pot()
            .arg("inspect")
            .arg(scene)
            .arg("--id")
            .arg(object)
            .arg("--frame")
            .arg(frame)
            .arg("--json"),
        "modifier inspection",
    )?;
    Ok(serde_json::from_slice(&output.stdout)?)
}

#[test]
fn indexed_particles_lifecycle_children_and_bake_are_evaluated() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("particle-lifecycle");
    let batch = directory.path().join("operations.json");
    let bake_directory = directory.path().join("bake");
    initialize(&scene)?;
    apply(
        &scene,
        &batch,
        &json!([
            {"op":"node.create","id":"a_emitter","kind":"plane","params":{}},
            {"op":"physics.particle_emitter.create","target":{"id":"a_emitter"},"settings":{"emit_from":"VERT","count":2,"frame_start":1.0,"frame_end":2.0,"lifetime":1.0,"physics_type":"NO","normal_factor":0.0,"particle_size":0.75,"child_nbr":2,"child_radius":0.6,"child_type":"SIMPLE","seed":11}},
            {"op":"modifier.create","target":{"id":"a_emitter"},"id":"particles_second","type":"particle_system","params":{"settings_id":"particles_second"}},
            {"op":"node.create","id":"b_alive","kind":"plane","params":{}},
            {"op":"modifier.create","target":{"id":"b_alive"},"id":"alive_instances","type":"particle_instance","params":{"object":"a_emitter","particle_system_index":1,"use_normal":true,"use_children":true,"use_size":false,"show_alive":true,"show_dead":false,"show_unborn":false,"position":0,"random_position":0.0,"axis":"X","space":"WORLD","use_path":false}},
            {"op":"node.create","id":"c_secondary","kind":"plane","params":{}},
            {"op":"modifier.create","target":{"id":"c_secondary"},"id":"secondary_instances","type":"particle_instance","params":{"object":"a_emitter","particle_system_index":2,"use_normal":true,"use_children":false,"use_size":false,"show_alive":true,"show_dead":false,"show_unborn":false,"position":0,"random_position":0.0,"axis":"X","space":"WORLD","use_path":false}},
            {"op":"node.create","id":"d_no_unborn","kind":"plane","params":{}},
            {"op":"modifier.create","target":{"id":"d_no_unborn"},"id":"without_unborn","type":"particle_instance","params":{"object":"a_emitter","particle_system_index":1,"use_normal":true,"use_children":false,"use_size":false,"show_alive":true,"show_dead":false,"show_unborn":false,"position":0,"random_position":0.0,"axis":"X","space":"WORLD","use_path":false}},
            {"op":"node.create","id":"g_with_unborn","kind":"plane","params":{}},
            {"op":"modifier.create","target":{"id":"g_with_unborn"},"id":"with_unborn","type":"particle_instance","params":{"object":"a_emitter","particle_system_index":1,"use_normal":true,"use_children":false,"use_size":false,"show_alive":true,"show_dead":false,"show_unborn":true,"position":0,"random_position":0.0,"axis":"X","space":"WORLD","use_path":false}},
            {"op":"node.create","id":"e_dead","kind":"plane","params":{}},
            {"op":"modifier.create","target":{"id":"e_dead"},"id":"dead_instances","type":"particle_instance","params":{"object":"a_emitter","particle_system_index":1,"use_normal":true,"use_children":true,"use_size":true,"show_alive":false,"show_dead":true,"show_unborn":false,"position":0,"random_position":0.0,"axis":"X","space":"WORLD","use_path":false}},
            {"op":"node.create","id":"f_exploder","kind":"box","params":{}},
            {"op":"physics.particle_emitter.create","target":{"id":"f_exploder"},"settings":{"emit_from":"VERT","count":1,"frame_start":1.0,"frame_end":1.0,"lifetime":10.0,"physics_type":"NO","normal_factor":0.0,"seed":13}},
            {"op":"modifier.create","target":{"id":"f_exploder"},"id":"explode_faces","type":"explode","params":{"use_edge_cut":true,"show_alive":true,"show_dead":false,"show_unborn":false,"use_size":false,"protect":0.0}}
        ]),
        0,
    )?;

    for (object, frame, expected_faces) in [
        ("b_alive", "1", 3),
        ("c_secondary", "1", 1),
        ("d_no_unborn", "1", 3),
        ("g_with_unborn", "1", 6),
        ("e_dead", "3", 3),
        ("f_exploder", "1", 6),
    ] {
        let response = inspect(&scene, object, frame)?;
        assert_eq!(
            response["result"]["items"][0]["evaluated_geometry"]["face_count"], expected_faces,
            "{object} at frame {frame}: {response}",
        );
    }

    let bake_output = run(
        pot()
            .arg("bake")
            .arg(&scene)
            .arg("--kind")
            .arg("simulation")
            .arg("--frames")
            .arg("1:3")
            .arg("--out")
            .arg(&bake_directory)
            .arg("--json"),
        "particle lifecycle bake",
    )?;
    let response: Value = serde_json::from_slice(&bake_output.stdout)?;
    assert_eq!(response["result"]["frame_count"], 3);
    let frame_one: Value =
        serde_json::from_slice(&fs::read(bake_directory.join("frame_00000000.json"))?)?;
    let frame_three: Value =
        serde_json::from_slice(&fs::read(bake_directory.join("frame_00000002.json"))?)?;
    let frame_one_states = frame_one["particles"]["a_emitter"]
        .as_array()
        .ok_or("frame-one particle states are missing")?;
    assert_eq!(frame_one_states.len(), 7);
    assert_eq!(
        frame_one_states
            .iter()
            .filter(|state| state["life_state"] == "unborn")
            .count(),
        3,
    );
    let frame_three_states = frame_three["particles"]["a_emitter"]
        .as_array()
        .ok_or("frame-three particle states are missing")?;
    assert!(
        frame_three_states
            .iter()
            .any(|state| state["life_state"] == "dead"),
        "expired particle state must survive in the bake cache",
    );
    Ok(())
}

#[test]
fn volume_displace_resamples_inline_density_with_object_texture_mapping()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("volume-displacement");
    let batch = directory.path().join("operations.json");
    initialize(&scene)?;
    apply(
        &scene,
        &batch,
        &json!([
            {"op":"image.create","id":"displace_texture","name":"Displacement","width":2,"height":1,"colorspace":"linear","fill_color":[0.0,0.5,0.5,1.0]},
            {"op":"image.set_pixels","id":"displace_texture","x":1,"width":1,"height":1,"pixels":[[1.0,0.5,0.5,1.0]]},
            {"op":"node.create","id":"texture_frame","kind":"plane","params":{},"transform":{"translation":[-0.25,0.0,0.0]}},
            {"op":"volume.create","id":"density","grids":[{"dims":[3,1,1],"voxel_size":0.5,"origin":[0.0,0.0,0.0],"values":[0.0,1.0,0.0]}]},
            {"op":"modifier.create","target":{"id":"density"},"id":"displace","type":"volume_displace","params":{"texture":"displace_texture","texture_map_mode":"OBJECT","texture_map_object":"texture_frame","strength":0.5,"texture_mid_level":[0.5,0.5,0.5],"texture_sample_radius":0.0}}
        ]),
        0,
    )?;
    let document: potter::model::SceneDoc =
        serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let node = document
        .nodes
        .get(&potter::model::Id::new("density".to_owned())?)
        .ok_or("volume object is missing")?;
    let data_id = node.data.as_ref().ok_or("volume Data-Block is missing")?;
    let snapshot = potter::eval::Snapshot::evaluate_with_cache(
        &document,
        &potter::eval::EvaluationContext::default(),
        Some(&scene),
    )?;
    let values = snapshot
        .volume_data
        .get(data_id)
        .and_then(|volume| volume.grids.first())
        .and_then(|grid| grid.values.as_deref())
        .ok_or("displaced volume samples are missing")?;
    assert_eq!(values, [0.0_f32, 0.75, 0.0]);
    Ok(())
}

#[test]
fn volume_displace_resamples_decoded_vdb_density() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping VDB Volume Displace test; Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let script = root.join("create_density.py");
    let vdb_path = root.join("density.vdb");
    let blender_expected_path = root.join("blender_expected.json");
    fs::write(
        &script,
        r"
import bpy, openvdb, json, os, sys, math
from mathutils import Vector
root = os.path.realpath(sys.argv[sys.argv.index('--') + 1])
grid = openvdb.FloatGrid()
grid.name = 'density'
accessor = grid.getAccessor()
for z in range(3):
    for y in range(3):
        for x in range(4):
            density = 0.15 + 0.16*x + 0.09*y + 0.07*z + 0.045*x*y - 0.025*x*z
            accessor.setValueOn((x,y,z), density)
vdb_path = os.path.join(root, 'density.vdb')
openvdb.write(vdb_path, grids=[grid])
bpy.ops.object.volume_import(filepath=vdb_path)
volume = bpy.context.object
image = bpy.data.images.new('Displacement', width=4, height=3, alpha=True, float_buffer=True)
pixels = []
for y in range(3):
    for x in range(4):
        pixels.extend([x/3, y/2, ((2*x+y) % 4)/3, 1.0])
image.pixels[:] = pixels
texture = bpy.data.textures.new('Displacement', type='IMAGE')
texture.image = image
mapping_mesh = bpy.data.meshes.new('mapping_object_mesh')
mapping_object = bpy.data.objects.new('mapping_object', mapping_mesh)
bpy.context.scene.collection.objects.link(mapping_object)
mapping_object.location = (-0.25, 0.2, -0.1)
mapping_object.rotation_euler = (0.2, 0.35, -0.15)
mapping_object.scale = (1.3, 0.8, 1.1)
displace = volume.modifiers.new('Volume Displace', 'VOLUME_DISPLACE')
displace.texture = texture
displace.texture_map_mode = 'OBJECT'
displace.texture_map_object = mapping_object
displace.strength = 3.0
displace.texture_mid_level = (0.5, 0.5, 0.5)
displace.texture_sample_radius = 1.0
baseline_volume = bpy.data.objects.new('BaselineVolume', volume.data)
bpy.context.scene.collection.objects.link(baseline_volume)

def make_surface(name, source_volume):
    surface = bpy.data.objects.new(name, bpy.data.meshes.new(name + 'Mesh'))
    bpy.context.scene.collection.objects.link(surface)
    to_mesh = surface.modifiers.new('Volume to Mesh', 'VOLUME_TO_MESH')
    to_mesh.object = source_volume
    to_mesh.threshold = 0.25
    to_mesh.resolution_mode = 'GRID'
    return surface

surface = make_surface('DisplacedSurface', volume)
baseline_surface = make_surface('BaselineSurface', baseline_volume)
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
depsgraph.update()
evaluated = surface.evaluated_get(depsgraph)
evaluated_mesh = evaluated.to_mesh()
points = [evaluated.matrix_world @ vertex.co for vertex in evaluated_mesh.vertices]
baseline_evaluated = baseline_surface.evaluated_get(depsgraph)
baseline_mesh = baseline_evaluated.to_mesh()
baseline_points = [
    baseline_evaluated.matrix_world @ vertex.co for vertex in baseline_mesh.vertices
]
mapping_world = [
    mapping_object.matrix_world[row][column]
    for column in range(4)
    for row in range(4)
]
def sample_density(coordinate):
    lower = [math.floor(value) for value in coordinate]
    fraction = [coordinate[axis] - lower[axis] for axis in range(3)]
    value = 0.0
    for z_offset in range(2):
        for y_offset in range(2):
            for x_offset in range(2):
                weight = (
                    (fraction[0] if x_offset else 1.0-fraction[0]) *
                    (fraction[1] if y_offset else 1.0-fraction[1]) *
                    (fraction[2] if z_offset else 1.0-fraction[2])
                )
                value += accessor.getValue((
                    lower[0] + x_offset,
                    lower[1] + y_offset,
                    lower[2] + z_offset,
                )) * weight
    return value

mapping_world_to_local = mapping_object.matrix_world.inverted()
volume_to_world = volume.matrix_world
blender_samples = []
for z in range(-2,5):
    for y in range(-2,5):
        for x in range(-2,6):
            index = Vector((x,y,z))
            texture_position = mapping_world_to_local @ (volume_to_world @ index)
            color = texture.evaluate(texture_position)
            displacement = Vector((
                (color[0]-0.5)*3.0,
                (color[1]-0.5)*3.0,
                (color[2]-0.5)*3.0,
            ))
            source = index-displacement
            sample = sample_density(source)
            blender_samples.append(sample)
evaluated_volume = volume.evaluated_get(depsgraph)
evaluated_grid_names = [grid.name for grid in evaluated_volume.data.grids]
expected = {
    'points':[list(point) for point in points],
    'baseline_points':[list(point) for point in baseline_points],
    'mapping_world':mapping_world,
    'vdb_samples':blender_samples,
    'evaluated_grid_names':evaluated_grid_names,
}
with open(os.path.join(root, 'blender_expected.json'), 'w', encoding='utf-8') as output:
    json.dump(expected, output)
evaluated.to_mesh_clear()
baseline_evaluated.to_mesh_clear()
        ",
    )?;
    run(
        Command::new(&blender)
            .args(["--background", "--factory-startup", "--python"])
            .arg(&script)
            .arg("--")
            .arg(root),
        "VDB density fixture generation",
    )?;
    let blender_expected: Value = serde_json::from_slice(&fs::read(&blender_expected_path)?)?;
    let decoded = potter::geom::vdb::VdbVolume::read(&fs::read(&vdb_path)?)
        .map_err(|error| std::io::Error::other(error.to_string()))?;
    let volume = potter::geom::volume::VolumeData {
        source: potter::geom::volume::VolumeSource::File(potter::geom::volume::VolumeFileSource {
            format: "vdb".to_owned(),
            content_ref: Some("asset://density".to_owned()),
            grid_names: vec!["density".to_owned()],
            bounds_min: None,
            bounds_max: None,
        }),
        decoded_vdb: Some(std::sync::Arc::new(decoded)),
        ..potter::geom::volume::VolumeData::default()
    };
    let pixels = (0_u32..3)
        .flat_map(|y| {
            (0_u32..4).map(move |x| {
                [
                    f64::from(x) / 3.0,
                    f64::from(y) / 2.0,
                    f64::from((2 * x + y) % 4) / 3.0,
                    1.0,
                ]
            })
        })
        .collect::<Vec<_>>();
    let image = potter::image::ImageData {
        width: 4,
        height: 3,
        pixels,
        tiles: std::collections::BTreeMap::new(),
        interpolation: potter::image::ImageInterpolation::Linear,
    };
    let mapping_columns: [f64; 16] =
        serde_json::from_value(blender_expected["mapping_world"].clone())?;
    let mapping_world = DMat4::from_cols_array(&mapping_columns);
    let volume_params = Map::from_iter([
        ("texture_map_mode".into(), json!("OBJECT")),
        ("texture_map_object".into(), json!("mapping_object")),
        ("strength".into(), json!(3.0)),
        ("texture_mid_level".into(), json!([0.5, 0.5, 0.5])),
        ("texture_sample_radius".into(), json!(1.0)),
    ]);
    let displaced = potter::geom::modifiers::simulation::displace_volume(
        &volume,
        &volume_params,
        "IMAGE",
        Some(&image),
        DMat4::IDENTITY,
        Some(mapping_world),
    )?;
    let expected_samples: Vec<f64> =
        serde_json::from_value(blender_expected["vdb_samples"].clone())?;
    let evaluated_grid_names: Vec<String> =
        serde_json::from_value(blender_expected["evaluated_grid_names"].clone())?;
    assert_eq!(evaluated_grid_names, ["density"]);
    let compare_density_samples = |name: &str, actual: &[f32]| {
        assert_eq!(
            actual.len(),
            expected_samples.len(),
            "{name} and Blender voxel sample counts differ",
        );
        let mut maximum_error = 0.0_f64;
        let mut maximum_index = 0;
        let mut maximum_actual = 0.0_f32;
        let mut maximum_expected = 0.0_f64;
        for (index, (actual, expected)) in actual.iter().zip(&expected_samples).enumerate() {
            let error = (f64::from(*actual) - expected).abs();
            if error > maximum_error {
                maximum_error = error;
                maximum_index = index;
                maximum_actual = *actual;
                maximum_expected = *expected;
            }
        }
        assert!(
            maximum_error <= DENSITY_SAMPLE_TOLERANCE,
            "{name} per-voxel density error {maximum_error} at sample {maximum_index} (Potter={maximum_actual}, Blender={maximum_expected}) exceeds {DENSITY_SAMPLE_TOLERANCE}",
        );
    };
    assert_eq!(
        displaced.source,
        potter::geom::volume::VolumeSource::Generated(
            potter::geom::volume::VolumeGeneratedSource {
                algorithm: "volume_displace_vdb".to_owned(),
            },
        )
    );
    let vdb_grid = displaced
        .grids
        .first()
        .ok_or("displaced VDB grid is missing")?;
    assert_eq!(vdb_grid.dims, [8, 7, 7]);
    compare_density_samples(
        "decoded VDB",
        vdb_grid
            .values
            .as_deref()
            .ok_or("displaced VDB samples are missing")?,
    );
    let mut inline_values = Vec::with_capacity(36);
    for z in 0_u8..3 {
        for y in 0_u8..3 {
            for x in 0_u8..4 {
                let x = f32::from(x);
                let y = f32::from(y);
                let z = f32::from(z);
                inline_values.push(
                    0.15_f32 + 0.16_f32 * x + 0.09_f32 * y + 0.07_f32 * z + 0.045_f32 * x * y
                        - 0.025_f32 * x * z,
                );
            }
        }
    }
    let inline_volume = potter::geom::volume::VolumeData {
        source: potter::geom::volume::VolumeSource::Generated(
            potter::geom::volume::VolumeGeneratedSource {
                algorithm: "inline_fixture".to_owned(),
            },
        ),
        grids: vec![potter::geom::volume::VolumeGrid {
            dims: [4, 3, 3],
            voxel_size: 1.0,
            origin: DVec3::ZERO,
            values: Some(inline_values),
            ..potter::geom::volume::VolumeGrid::default()
        }],
        ..potter::geom::volume::VolumeData::default()
    };
    let inline_displaced = potter::geom::modifiers::simulation::displace_volume(
        &inline_volume,
        &volume_params,
        "IMAGE",
        Some(&image),
        DMat4::IDENTITY,
        Some(mapping_world),
    )?;
    let inline_grid = inline_displaced
        .grids
        .first()
        .ok_or("displaced inline grid is missing")?;
    assert_eq!(inline_grid.dims, [8, 7, 7]);
    compare_density_samples(
        "inline grid",
        inline_grid
            .values
            .as_deref()
            .ok_or("displaced inline samples are missing")?,
    );
    let potter_mesh = potter::geom::volume::volume_to_mesh(&displaced, 0.25)
        .map_err(|error| std::io::Error::other(error.to_string()))?;
    let baseline_mesh = potter::geom::volume::volume_to_mesh(&volume, 0.25)
        .map_err(|error| std::io::Error::other(error.to_string()))?;
    let potter_points = potter_mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    let potter_baseline_points = baseline_mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    let blender_surface_points = blender_points(&blender_expected["points"])?;
    let blender_baseline = blender_points(&blender_expected["baseline_points"])?;
    let displaced_surface_distance =
        max_surface_correspondence_error(&blender_surface_points, &blender_baseline)?;
    assert!(
        displaced_surface_distance > MINIMUM_SURFACE_DISPLACEMENT,
        "spatially varying texture must materially displace the Blender surface by more than one voxel, observed {displaced_surface_distance}",
    );
    let potter_displacement =
        max_surface_correspondence_error(&potter_points, &potter_baseline_points)?;
    assert!(
        potter_displacement > MINIMUM_SURFACE_DISPLACEMENT,
        "Potter VDB surface must differ from its unmodified baseline by more than one voxel, observed {potter_displacement}",
    );
    Ok(())
}
