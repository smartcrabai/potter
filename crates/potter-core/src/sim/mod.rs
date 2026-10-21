//! Deterministic physics solvers and disposable per-frame result caching.

mod cache;
pub mod cloth;
pub mod dynamic_paint;
pub mod fluid;
mod particle_distribution;
mod particle_jitter;
mod particle_textures;
pub mod particles;
pub mod softbody;
mod solver;

use std::collections::{BTreeMap, BTreeSet};

use glam::{DMat4, DQuat, DVec3};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::geom::metaball::{MetaballData, MetaballElement};
use crate::{
    error::{ErrorCode, PotError, Result},
    eval::ModifierGate,
    geom::Mesh,
    model::{ForceFieldType, Id, Modifier, Node, RigidBodyType, SceneDoc},
};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StackModifierResult {
    pub mesh: Mesh,
    pub particles: Option<Vec<ParticleState>>,
}

pub use solver::SOLVER_VERSION;

pub(crate) fn setting_number(
    settings: &Value,
    key: &str,
    default: f64,
    family: &str,
    requirement: &str,
) -> Result<f64> {
    let Some(value) = settings.get(key) else {
        return Ok(default);
    };
    value
        .as_f64()
        .filter(|number| number.is_finite())
        .ok_or_else(|| PotError::invalid_argument(format!("{family} {key} must be {requirement}")))
}

pub(crate) fn unit_setting(settings: &Value, key: &str, default: f64, family: &str) -> Result<f64> {
    let number = setting_number(settings, key, default, family, "finite and in [0, 1]")?;
    if (0.0..=1.0).contains(&number) {
        Ok(number)
    } else {
        Err(PotError::invalid_argument(format!(
            "{family} {key} must be finite and in [0, 1]"
        )))
    }
}

pub(crate) fn nonnegative_setting(
    settings: &Value,
    key: &str,
    default: f64,
    family: &str,
) -> Result<f64> {
    let number = setting_number(settings, key, default, family, "finite and nonnegative")?;
    if number >= 0.0 {
        Ok(number)
    } else {
        Err(PotError::invalid_argument(format!(
            "{family} {key} must be finite and nonnegative"
        )))
    }
}

const MAX_TOTAL_STEPS: usize = 8_192;
const PARTICLE_STATE_VERSION: &str = "particle-state-6";

fn frame_steps(frame: f64, substeps: usize, message: &'static str) -> Result<usize> {
    let requested = frame.abs() * substeps as f64;
    if !requested.is_finite() || requested > MAX_TOTAL_STEPS as f64 {
        return Err(PotError::new(ErrorCode::LimitExceeded, message));
    }
    Ok(requested.ceil().max(1.0) as usize)
}

/// Deterministic simulation outputs for one scene and frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimulationResult {
    /// Final world transform for every simulated rigid body, in column-major order.
    pub world_matrices: BTreeMap<Id, [f64; 16]>,
    /// Final world-space linear velocity for every simulated rigid body.
    pub linear_velocities: BTreeMap<Id, [f64; 3]>,
    /// Content address of the complete simulation input snapshot.
    pub cache_key: String,
    /// Vertex positions produced by cloth and soft-body evaluators, in object-local order.
    #[serde(default)]
    pub deformed_vertices: BTreeMap<Id, Vec<[f64; 3]>>,
    /// Seeded particle positions in world coordinates.
    #[serde(default)]
    pub particle_positions: BTreeMap<Id, Vec<[f64; 3]>>,
    /// Seeded particle velocities in world coordinates.
    #[serde(default)]
    pub particle_velocities: BTreeMap<Id, Vec<[f64; 3]>>,
    /// Particle emission frame for each emitted particle.
    #[serde(default)]
    pub particle_birth_frames: BTreeMap<Id, Vec<f64>>,
    /// Full particle lifecycle states used by stack modifiers.
    #[serde(default)]
    pub particle_states: BTreeMap<Id, Vec<ParticleState>>,
    /// Liquid SPH particle positions in world coordinates.
    #[serde(default)]
    pub fluid_particles: BTreeMap<Id, Vec<[f64; 3]>>,
    /// Polygonized liquid surfaces in object-local coordinates.
    #[serde(default)]
    pub fluid_surfaces: BTreeMap<Id, crate::geom::Mesh>,
    /// Dynamic-paint colors in mesh vertex storage order.
    #[serde(default)]
    pub paint_colors: BTreeMap<Id, Vec<[f64; 4]>>,
    /// Dynamic-paint weights in mesh vertex storage order.
    #[serde(default)]
    pub paint_weights: BTreeMap<Id, Vec<f64>>,
    /// Marks the frame's non-rigid physics payload as fully evaluated in cache.
    #[serde(default)]
    pub physics_cache_key: Option<String>,
}

/// Deterministic simulation outputs for one scene and frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhysicsResult {
    /// Deformed local-space mesh vertices by object.
    pub deformed_vertices: BTreeMap<Id, Vec<[f64; 3]>>,
    /// Particle positions, velocities, and birth frames by emitter object.
    pub particles: BTreeMap<Id, Vec<ParticleState>>,
    /// SPH liquid particle positions by domain object.
    pub fluid_particles: BTreeMap<Id, Vec<[f64; 3]>>,
    /// Mesh surfaces polygonized from liquid particles.
    pub fluid_surfaces: BTreeMap<Id, crate::geom::Mesh>,
    /// Dynamic paint vertex colors by canvas object.
    pub paint_colors: BTreeMap<Id, Vec<[f64; 4]>>,
    /// Dynamic-paint vertex weights by canvas object.
    pub paint_weights: BTreeMap<Id, Vec<f64>>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParticleLifeState {
    Unborn,
    #[default]
    Alive,
    Dead,
}

/// A world-space trajectory sample with time normalized to the particle's lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ParticlePathKey {
    pub time: f64,
    pub position: [f64; 3],
    /// World-space derivative with respect to normalized lifetime time.
    /// Cubic Hermite interpolation uses this value at each key.
    pub velocity: [f64; 3],
    /// Bishop-frame rotation in XYZW order.
    pub rotation: [f64; 4],
}

/// Interpolation used to evaluate a particle's path controls.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParticlePathInterpolation {
    /// Cubic Hermite using each key's normalized-time derivative.
    #[default]
    Hermite,
    /// Direct interpolation of source path-cache samples and their derivatives.
    Linear,
    /// Linear interpolation of Hair pathcache samples; key rotations are not current-state orientation.
    HairCache,
    /// Hair's default cardinal interpolation, represented by Hermite tangents.
    HairCardinal,
    /// Blender's cubic B-spline interpolation for hair control keys.
    HairBspline,
}

/// One particle's world-space lifecycle state at a selected frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ParticleState {
    pub position: [f64; 3],
    pub birth_position: [f64; 3],
    pub normal: [f64; 3],
    /// Rotation quaternion in XYZW order from emitter-local to particle orientation.
    pub rotation: [f64; 4],
    pub birth_rotation: [f64; 4],
    /// Emitter-face frame used only to seed Particle Instance's path Bishop frame.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path_root_rotation: Option<[f64; 4]>,
    /// World-space velocity in meters per second, except path-sampled values use normalized-lifetime derivatives.
    pub velocity: [f64; 3],
    pub birth_frame: f64,
    pub death_frame: f64,
    pub life_state: ParticleLifeState,
    pub system_index: u32,
    pub is_child: bool,
    /// Parent-system index for child particles; `None` for regular particles.
    pub parent_index: Option<u32>,
    /// Weighted parent-system indices for interpolated children, in Blender nearest-parent order.
    pub parent_indices: [u32; 3],
    pub parent_weights: [f64; 3],
    pub parent_count: u8,
    /// Imported full-lifetime cache keys used for current-state sampling and path instances.
    pub path: Vec<ParticlePathKey>,
    /// Interpolation for `path`; omitted serialized states default to keyed/baked Hermite.
    pub path_interpolation: ParticlePathInterpolation,
    pub size: f64,
    pub mass: f64,
}

impl Default for ParticleState {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            birth_position: [0.0; 3],
            normal: [0.0, 0.0, 1.0],
            rotation: DQuat::IDENTITY.to_array(),
            birth_rotation: DQuat::IDENTITY.to_array(),
            path_root_rotation: None,
            velocity: [0.0; 3],
            birth_frame: 0.0,
            death_frame: 0.0,
            life_state: ParticleLifeState::Alive,
            system_index: 0,
            is_child: false,
            parent_index: None,
            parent_indices: [0; 3],
            parent_weights: [0.0; 3],
            parent_count: 0,
            path: Vec::new(),
            path_interpolation: ParticlePathInterpolation::Hermite,
            size: 1.0,
            mass: 1.0,
        }
    }
}
fn particle_orientation(normal: [f64; 3]) -> [f64; 4] {
    let normal = DVec3::from_array(normal);
    if normal.length_squared() <= f64::EPSILON {
        DQuat::IDENTITY.to_array()
    } else {
        DQuat::from_rotation_arc(DVec3::X, normal.normalize()).to_array()
    }
}
fn particle_state(particle: particles::Particle, system_index: u32) -> ParticleState {
    // Hair path keys carry Bishop frames, not the current Hair particle orientation.
    let rotation = if particle.path.is_empty() {
        particle_orientation(particle.normal)
    } else {
        match particle.path_interpolation {
            ParticlePathInterpolation::HairCardinal
            | ParticlePathInterpolation::HairBspline
            | ParticlePathInterpolation::HairCache => particle.birth_rotation,
            ParticlePathInterpolation::Hermite => particle
                .current_path_rotation
                .unwrap_or_else(|| particle_orientation(particle.normal)),
            ParticlePathInterpolation::Linear => particle_orientation(particle.normal),
        }
    };
    ParticleState {
        position: particle.position,
        birth_position: particle.birth_position,
        normal: particle.normal,
        rotation,
        birth_rotation: particle.birth_rotation,
        path_root_rotation: particle.path_root_rotation,
        velocity: particle.velocity,
        birth_frame: particle.birth_frame,
        death_frame: particle.death_frame,
        life_state: particle.life_state,
        system_index,
        is_child: particle.is_child,
        parent_index: particle.parent_index,
        parent_indices: particle.parent_indices,
        parent_weights: particle.parent_weights,
        parent_count: particle.parent_count,
        path: particle.path,
        path_interpolation: particle.path_interpolation,
        size: particle.size,
        mass: particle.mass,
    }
}

fn particle_states_valid(states: &[ParticleState]) -> bool {
    let parents_len = states.iter().take_while(|state| !state.is_child).count();
    states.iter().all(|state| {
        let parent_count = usize::from(state.parent_count);
        let parent_associations_valid = parent_count <= 3
            && state.parent_indices[..parent_count]
                .iter()
                .all(|index| usize::try_from(*index).is_ok_and(|index| index < parents_len))
            && state.parent_weights[..parent_count]
                .iter()
                .all(|weight| weight.is_finite() && *weight >= 0.0)
            && (parent_count == 0
                || (state.parent_weights[..parent_count].iter().sum::<f64>() - 1.0).abs()
                    <= 1.0e-6);
        let parent_link_valid = match (state.is_child, state.parent_index) {
            (false, None) => parent_count == 0,
            (true, Some(parent_index)) => {
                usize::try_from(parent_index).is_ok_and(|index| index < parents_len)
                    && (parent_count == 0 || parent_associations_valid)
            }
            _ => false,
        };
        let path_valid = state.path.iter().all(|key| {
            let rotation = DQuat::from_array(key.rotation);
            key.time.is_finite()
                && (0.0..=1.0).contains(&key.time)
                && key.position.iter().all(|value| value.is_finite())
                && key.velocity.iter().all(|value| value.is_finite())
                && rotation.is_finite()
                && rotation.length_squared() > 1.0e-20
        }) && state
            .path
            .windows(2)
            .all(|keys| keys[0].time <= keys[1].time);
        let rotation = DQuat::from_array(state.rotation);
        let birth_rotation = DQuat::from_array(state.birth_rotation);
        let path_root_valid = state.path_root_rotation.is_none_or(|rotation| {
            let rotation = DQuat::from_array(rotation);
            rotation.is_finite() && rotation.length_squared() > 1.0e-20
        });
        state
            .position
            .iter()
            .chain(&state.birth_position)
            .chain(&state.normal)
            .chain(&state.velocity)
            .all(|value| value.is_finite())
            && rotation.is_finite()
            && rotation.length_squared() > 1.0e-20
            && birth_rotation.is_finite()
            && birth_rotation.length_squared() > 1.0e-20
            && path_root_valid
            && state.birth_frame.is_finite()
            && state.death_frame.is_finite()
            && state.death_frame >= state.birth_frame
            && state.size.is_finite()
            && state.size >= 0.0
            && state.mass.is_finite()
            && state.mass > 0.0
            && parent_link_valid
            && parent_associations_valid
            && path_valid
    })
}

fn scene_members(doc: &SceneDoc, scene_id: &Id) -> Result<BTreeSet<Id>> {
    let scene = doc.scenes.get(scene_id).ok_or_else(|| {
        PotError::new(
            ErrorCode::SceneNotFound,
            format!("scene `{scene_id}` was not found"),
        )
    })?;
    let mut pending = vec![scene.root_collection.clone()];
    let mut visited = BTreeSet::new();
    let mut members = BTreeSet::new();
    while let Some(collection_id) = pending.pop() {
        if !visited.insert(collection_id.clone()) {
            continue;
        }
        let collection = doc.collections.get(&collection_id).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "scene collection is missing",
                json!({"scene_id":scene_id,"collection_id":collection_id}),
            )
        })?;
        members.extend(collection.objects.iter().cloned());
        pending.extend(collection.children.iter().cloned());
    }
    Ok(members)
}

/// Evaluates one physics modifier against the geometry produced by earlier stack entries.
pub(crate) fn evaluate_stack_modifier(
    doc: &SceneDoc,
    scene_id: &Id,
    node_id: &Id,
    modifier: &Modifier,
    input_mesh: Mesh,
    frame: f64,
    matrices: &BTreeMap<Id, DMat4>,
    scene_hash: &str,
    gate: ModifierGate<'_>,
) -> Result<StackModifierResult> {
    let cache_directory = gate.context.cache_directory;
    let (property, system) =
        physics_stack_classification(&modifier.modifier_type).ok_or_else(|| {
            PotError::new(
                ErrorCode::InvalidOperation,
                format!(
                    "`{}` is not a physics stack modifier",
                    modifier.modifier_type
                ),
            )
        })?;
    let node = doc.nodes.get(node_id).ok_or_else(|| {
        PotError::new(
            ErrorCode::TargetNotFound,
            format!("physics modifier node `{node_id}` was not found"),
        )
    })?;
    let settings_id = modifier
        .params
        .get("settings_id")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "physics modifier has no settings_id",
                json!({"node_id":node_id,"modifier_id":modifier.id}),
            )
        })?;
    if settings_id != modifier.id.as_str() {
        return Err(PotError::with_details(
            ErrorCode::SceneInvalid,
            "physics modifier settings_id does not match its modifier ID",
            json!({"node_id":node_id,"modifier_id":modifier.id,"settings_id":settings_id}),
        ));
    }
    let settings = if system == "particle_system" {
        node.properties
            .get("physics_particle_systems")
            .and_then(Value::as_object)
            .and_then(|systems| systems.get(settings_id))
            .or_else(|| {
                let primary_id = node
                    .properties
                    .get("physics_particle_emitter_modifier_id")
                    .and_then(Value::as_str);
                (primary_id == Some(settings_id) || primary_id.is_none())
                    .then(|| node.properties.get(property))
                    .flatten()
            })
    } else {
        node.properties.get(property)
    }
    .ok_or_else(|| {
        PotError::with_details(
            ErrorCode::SceneInvalid,
            "physics modifier has no linked physics settings",
            json!({"node_id":node_id,"modifier_id":modifier.id,"property":property}),
        )
    })?;
    let system_index = particle_system_index(node, modifier.id.as_str());
    let emit_unborn = system == "particle_system"
        && doc.nodes.iter().any(|(target_id, target)| {
            target.modifiers.iter().any(|candidate| {
                let same_system_index = candidate
                    .params
                    .get("particle_system_index")
                    .and_then(Value::as_u64)
                    .unwrap_or(1)
                    .checked_sub(1)
                    .and_then(|index| u32::try_from(index).ok())
                    == Some(system_index);
                gate.is_active(target_id, candidate)
                    && same_system_index
                    && ((candidate.modifier_type == "particle_instance"
                        && candidate.params.get("object").and_then(Value::as_str)
                            == Some(node_id.as_str()))
                        || (candidate.modifier_type == "explode" && target_id == node_id))
            })
        });
    let mut settings_with_lifecycle;
    let settings = if system == "particle_system" {
        settings_with_lifecycle = settings.clone();
        if let Some(object) = settings_with_lifecycle.as_object_mut() {
            object.insert("emit_unborn".to_owned(), Value::Bool(emit_unborn));
        }
        &settings_with_lifecycle
    } else {
        settings
    };
    let solver_settings = normalize_solver_settings(settings, system);
    let base_key = cache::cache_key(scene_hash, scene_id, frame, matrices, gate)?;
    let input_mesh_snapshot = serde_json::to_value(&input_mesh).map_err(|error| {
        PotError::with_details(
            ErrorCode::EvaluationFailed,
            "could not serialize physics modifier input geometry",
            json!({"cause":error.to_string()}),
        )
    })?;
    let cache_key = crate::hash::sha256(&crate::hash::canonicalize(&json!({
        "solver_version": SOLVER_VERSION,
        "scene_cache_key": base_key,
        "node_id": node_id,
        "modifier_id": modifier.id,
        "modifier_type": modifier.modifier_type,
        "input_mesh": input_mesh_snapshot,
    }))?);
    if let Some(directory) = cache_directory
        && let Some((mesh, particles)) = cache::load_modifier_simulation(directory, &cache_key)?
        && particles.as_deref().is_none_or(particle_states_valid)
    {
        return Ok(StackModifierResult { mesh, particles });
    }
    let scene = doc.scenes.get(scene_id).ok_or_else(|| {
        PotError::new(
            ErrorCode::SceneNotFound,
            format!("scene `{scene_id}` was not found"),
        )
    })?;
    let fps = f64::from(scene.fps) / scene.fps_base;
    if !fps.is_finite() || fps <= 0.0 || !frame.is_finite() {
        return Err(PotError::invalid_argument(
            "physics frame rate and frame must be finite and positive",
        ));
    }
    let matrix = matrices.get(node_id).copied().ok_or_else(|| {
        PotError::with_details(
            ErrorCode::EvaluationFailed,
            "physics modifier object transform is missing",
            json!({"node_id":node_id}),
        )
    })?;
    let gravity = scene
        .rigid_body_world
        .as_ref()
        .map_or(DVec3::new(0.0, 0.0, -9.81), |world| {
            DVec3::from_array(world.gravity)
        });
    let members = scene_members(doc, scene_id)?;
    let collision_meshes = collider_triangles(doc, matrices, &members, gate)?;
    let local_colliders = if setting_bool(settings, "collision_enabled") == Some(false)
        || setting_bool(settings, "use_collision") == Some(false)
    {
        Vec::new()
    } else {
        local_collision_triangles(&collision_meshes, node_id, invert_matrix(matrix, node_id)?)
    };
    let mut mesh = input_mesh;
    let mut particles = None;
    match system {
        "cloth" => {
            let local_gravity = matrix.inverse().transform_vector3(gravity);
            // Blender's physics cache starts at the scene's start frame with the rest state.
            let solver_frame = (frame - f64::from(scene.frame_start)).max(0.0);
            let positions = cloth::simulate(
                &mesh,
                &solver_settings,
                solver_frame,
                fps,
                local_gravity,
                &local_colliders,
            )?;
            set_vertex_positions(&mut mesh, positions, node_id)?;
        }
        "soft_body" => {
            let local_gravity = matrix.inverse().transform_vector3(gravity);
            let solver_frame = (frame - f64::from(scene.frame_start)).max(0.0);
            let positions = softbody::simulate(
                &mesh,
                &solver_settings,
                solver_frame,
                fps,
                local_gravity,
                &local_colliders,
            )?;
            set_vertex_positions(&mut mesh, positions, node_id)?;
        }
        "particle_system" => {
            let seed = setting_u32(settings, "seed").unwrap_or_else(|| {
                scene
                    .rigid_body_world
                    .as_ref()
                    .map_or(0, |world| world.seed)
            });
            let colliders = collision_meshes
                .iter()
                .filter(|(collider_id, _)| collider_id != node_id)
                .map(|(_, triangle)| *triangle)
                .collect::<Vec<_>>();
            let fields = force_field_inputs(doc, matrices, &members)?;
            let textures =
                load_particle_textures(&solver_settings, doc, gate.context.project_root)?;
            let slots = particle_textures::texture_slots(&solver_settings, &textures, matrices)?;
            let states = particles::simulate_lifecycle_textured(
                &mesh,
                &solver_settings,
                matrix,
                frame,
                fps,
                gravity,
                seed,
                &fields,
                &colliders,
                &slots,
            )?
            .into_iter()
            .map(|particle| particle_state(particle, system_index))
            .collect::<Vec<_>>();
            particles = Some(states);
        }
        "fluid" => {
            let seed = setting_u32(settings, "seed").unwrap_or_else(|| {
                scene
                    .rigid_body_world
                    .as_ref()
                    .map_or(0, |world| world.seed)
            });
            let colliders = collision_meshes
                .iter()
                .filter(|(collider_id, _)| collider_id != node_id)
                .map(|(_, triangle)| *triangle)
                .collect::<Vec<_>>();
            let samples = fluid::simulate(
                &mesh, settings, matrix, frame, fps, gravity, seed, &colliders,
            )?;
            mesh = fluid_surface(&samples, matrix, settings)?;
        }
        "dynamic_paint" => {
            let brushes = dynamic_paint_brush_settings(doc, matrices, settings, &members, gate)?;
            let mut colors = vec![[0.0; 4]; mesh.vertices.len()];
            if brushes.is_empty() {
                colors = dynamic_paint::paint(
                    &mesh,
                    matrix,
                    &[],
                    setting_f64(settings, "radius").unwrap_or(1.0),
                    setting_color(settings).unwrap_or([1.0, 0.0, 0.0, 1.0]),
                    setting_f64(settings, "strength").unwrap_or(1.0),
                )?;
            } else {
                for brush in &brushes {
                    let painted = dynamic_paint::paint(
                        &mesh,
                        matrix,
                        &[brush.position],
                        brush.radius,
                        brush.color,
                        brush.strength,
                    )?;
                    for (current, candidate) in colors.iter_mut().zip(painted) {
                        if candidate[3] > current[3] {
                            *current = candidate;
                        }
                    }
                }
            }
            let values = mesh
                .vertices
                .iter()
                .zip(&colors)
                .map(|(vertex, color)| (format!("v{}", vertex.id), json!(color)))
                .collect::<serde_json::Map<_, _>>();
            mesh.attributes.insert(
                "color".to_owned(),
                json!({"domain":"vertices","values":values}),
            );
            if setting_string(settings, "surface_format") == Some("weight") {
                let values = mesh
                    .vertices
                    .iter()
                    .zip(&colors)
                    .map(|(vertex, color)| (format!("v{}", vertex.id), json!(color[3])))
                    .collect::<serde_json::Map<_, _>>();
                mesh.attributes.insert(
                    "weight".to_owned(),
                    json!({"domain":"vertices","type":"float","values":values}),
                );
            }
        }
        "collision" => {}
        _ => {
            return Err(PotError::new(
                ErrorCode::InternalError,
                "physics modifier system mapping is inconsistent",
            ));
        }
    }
    mesh.validate().map_err(|error| {
        PotError::with_details(
            ErrorCode::EvaluationFailed,
            "physics modifier produced invalid geometry",
            json!({"node_id":node_id,"cause":error.to_string()}),
        )
    })?;
    if let Some(directory) = cache_directory {
        cache::store_modifier_simulation(directory, &cache_key, &mesh, particles.as_deref())?;
    }
    Ok(StackModifierResult { mesh, particles })
}

fn physics_stack_classification(modifier_type: &str) -> Option<(&'static str, &'static str)> {
    Some(match modifier_type {
        "cloth" => ("physics_cloth", "cloth"),
        "soft_body" => ("physics_soft_body", "soft_body"),
        "collision" => ("physics_collision", "collision"),
        "dynamic_paint" => ("physics_dynamic_paint", "dynamic_paint"),
        "fluid" => ("physics_fluid", "fluid"),
        "particle_system" => ("physics_particle_emitter", "particle_system"),
        _ => return None,
    })
}

fn particle_system_index(node: &Node, modifier_id: &str) -> u32 {
    node.modifiers
        .iter()
        .filter(|modifier| modifier.modifier_type == "particle_system")
        .position(|modifier| modifier.id.as_str() == modifier_id)
        .and_then(|index| u32::try_from(index).ok())
        .unwrap_or(0)
}

/// Resolves the texture resources that a particle system's supported texture slots refer to.
fn load_particle_textures(
    settings: &Value,
    doc: &SceneDoc,
    project_root: Option<&std::path::Path>,
) -> Result<BTreeMap<String, crate::geom::legacy_texture::ResolvedTexture>> {
    let mut textures = BTreeMap::new();
    for name in particle_textures::texture_ids(settings) {
        let id = Id::new(name.clone())?;
        let resolved = crate::geom::legacy_texture::resolve_texture(doc, &id, project_root)?;
        textures.insert(name, resolved);
    }
    Ok(textures)
}

fn normalize_solver_settings(settings: &Value, system: &str) -> Value {
    let mut normalized = settings.clone();
    let Some(properties) = normalized.as_object_mut() else {
        return normalized;
    };
    match system {
        "cloth" => {
            if !properties.contains_key("structural_stiffness") {
                let tension = settings
                    .get("tension_stiffness")
                    .and_then(Value::as_f64)
                    .map(normalize_cloth_stiffness);
                let compression = settings
                    .get("compression_stiffness")
                    .and_then(Value::as_f64)
                    .map(normalize_cloth_stiffness);
                let structural = match (tension, compression) {
                    (Some(tension), Some(compression)) => Some(f64::midpoint(tension, compression)),
                    (Some(value), None) | (None, Some(value)) => Some(value),
                    (None, None) => None,
                };
                if let Some(value) = structural {
                    properties.insert("structural_stiffness".to_owned(), json!(value));
                }
            }
            if !properties.contains_key("bend_stiffness")
                && let Some(value) = settings.get("bending_stiffness").and_then(Value::as_f64)
            {
                properties.insert(
                    "bend_stiffness".to_owned(),
                    json!(normalize_cloth_stiffness(value)),
                );
            }
            if let Some(value) = settings.get("shear_stiffness").and_then(Value::as_f64) {
                properties.insert(
                    "shear_stiffness".to_owned(),
                    json!(normalize_cloth_stiffness(value)),
                );
            }
            if !properties.contains_key("air_drag")
                && let Some(value) = settings.get("air_damping")
            {
                properties.insert("air_drag".to_owned(), value.clone());
            }
            if !properties.contains_key("pin_group")
                && let Some(value) = settings.get("vertex_group_mass")
            {
                properties.insert("pin_group".to_owned(), value.clone());
            }
            if !properties.contains_key("collision_distance")
                && let Some(value) = settings.get("distance_min")
            {
                properties.insert("collision_distance".to_owned(), value.clone());
            }
        }
        "soft_body" => {
            if !properties.contains_key("goal_strength")
                && let Some(value) = settings.get("goal_default")
            {
                properties.insert("goal_strength".to_owned(), value.clone());
            }
            if !properties.contains_key("goal_stiffness")
                && let Some(value) = settings.get("goal_spring")
            {
                properties.insert("goal_stiffness".to_owned(), value.clone());
            }
            if !properties.contains_key("drag")
                && let Some(value) = settings.get("friction")
            {
                properties.insert("drag".to_owned(), value.clone());
            }
            if !properties.contains_key("goal_group")
                && let Some(value) = settings.get("vertex_group_goal")
            {
                properties.insert("goal_group".to_owned(), value.clone());
            }
        }
        "particle_system" => {
            if !properties.contains_key("random_velocity")
                && let Some(value) = settings.get("factor_random")
            {
                properties.insert("random_velocity".to_owned(), value.clone());
            }
            if !properties.contains_key("source")
                && let Some(source) = settings.get("emit_from").and_then(Value::as_str)
            {
                let source = match source {
                    "VERT" => "vertices",
                    "FACE" => "faces",
                    "VOLUME" => "volume",
                    _ => source,
                };
                properties.insert("source".to_owned(), Value::String(source.to_owned()));
            }
        }
        _ => {}
    }
    normalized
}
fn normalize_cloth_stiffness(value: f64) -> f64 {
    if value > 1.0 {
        1.0 / (1.0 + 1.0 / value)
    } else {
        value / (1.0 + value)
    }
}

fn set_vertex_positions(mesh: &mut Mesh, positions: Vec<DVec3>, node_id: &Id) -> Result<()> {
    if positions.len() != mesh.vertices.len()
        || positions.iter().any(|position| !position.is_finite())
    {
        return Err(PotError::with_details(
            ErrorCode::EvaluationFailed,
            "physics modifier returned invalid vertex positions",
            json!({"node_id":node_id}),
        ));
    }
    for (vertex, position) in mesh.vertices.iter_mut().zip(positions) {
        vertex.co = position;
    }
    Ok(())
}

/// Evaluate deterministic physics systems for `frame` and optionally cache their result.
///
/// When the gate's cache directory is supplied, results are stored directly in that directory.
pub(crate) fn evaluate(
    doc: &SceneDoc,
    scene_id: &Id,
    frame: f64,
    base_world_matrices: &BTreeMap<Id, DMat4>,
    scene_hash: &str,
    gate: ModifierGate<'_>,
) -> Result<Option<SimulationResult>> {
    let cache_directory = gate.context.cache_directory;
    let key = cache::cache_key(scene_hash, scene_id, frame, base_world_matrices, gate)?;
    let result = solver::evaluate(
        doc,
        scene_id,
        frame,
        base_world_matrices,
        cache_directory,
        &key,
    )?;
    let members = scene_members(doc, scene_id)?;
    let has_physics = members.iter().any(|id| {
        doc.nodes.get(id).is_some_and(|node| {
            [
                ("physics_cloth", "cloth"),
                ("physics_soft_body", "soft_body"),
                ("physics_particle_emitter", "particle_system"),
                ("physics_fluid", "fluid"),
                ("physics_dynamic_paint", "dynamic_paint"),
            ]
            .iter()
            .any(|(property, modifier_type)| {
                node.properties.contains_key(*property)
                    && gate.is_property_active(id, node, property, modifier_type)
            })
        })
    });
    if !has_physics {
        if let Some(mut result) = result {
            result.cache_key = key;
            return Ok(Some(result));
        }
        return Ok(None);
    }
    if let Some(directory) = cache_directory
        && let Some(cached) = cache::load(directory, &key)?
        && cached.physics_cache_key.as_deref() == Some(key.as_str())
        && cached
            .particle_states
            .values()
            .all(|states| particle_states_valid(states))
    {
        return Ok(Some(cached));
    }
    if result.as_ref().is_some_and(|cached| {
        cached.physics_cache_key.as_deref() == Some(key.as_str())
            && cached
                .particle_states
                .values()
                .all(|states| particle_states_valid(states))
    }) {
        return Ok(result);
    }
    let physics = evaluate_physics(doc, scene_id, frame, base_world_matrices, &members, gate)?;
    let mut result = result.unwrap_or_else(|| empty_result(key.clone()));
    result.cache_key = key.clone();
    result.deformed_vertices = physics.deformed_vertices;
    result.fluid_particles = physics.fluid_particles;
    result.fluid_surfaces = physics.fluid_surfaces;
    result.paint_colors = physics.paint_colors;
    result.paint_weights = physics.paint_weights;
    for (id, states) in physics.particles {
        result.particle_positions.insert(
            id.clone(),
            states.iter().map(|state| state.position).collect(),
        );
        result.particle_velocities.insert(
            id.clone(),
            states.iter().map(|state| state.velocity).collect(),
        );
        result.particle_birth_frames.insert(
            id.clone(),
            states.iter().map(|state| state.birth_frame).collect(),
        );
        result.particle_states.insert(id, states);
    }
    result.physics_cache_key = Some(key);
    if let Some(directory) = cache_directory {
        cache::store(directory, &result)?;
    }
    Ok(Some(result))
}

fn empty_result(key: String) -> SimulationResult {
    SimulationResult {
        world_matrices: BTreeMap::new(),
        linear_velocities: BTreeMap::new(),
        cache_key: key,
        deformed_vertices: BTreeMap::new(),
        particle_positions: BTreeMap::new(),
        particle_velocities: BTreeMap::new(),
        particle_birth_frames: BTreeMap::new(),
        particle_states: BTreeMap::new(),
        fluid_particles: BTreeMap::new(),
        fluid_surfaces: BTreeMap::new(),
        paint_colors: BTreeMap::new(),
        paint_weights: BTreeMap::new(),
        physics_cache_key: None,
    }
}

fn evaluate_physics(
    doc: &SceneDoc,
    scene_id: &Id,
    frame: f64,
    matrices: &BTreeMap<Id, DMat4>,
    members: &BTreeSet<Id>,
    gate: ModifierGate<'_>,
) -> Result<PhysicsResult> {
    let scene = doc.scenes.get(scene_id).ok_or_else(|| {
        PotError::new(
            ErrorCode::SceneNotFound,
            format!("scene `{scene_id}` was not found"),
        )
    })?;
    let fps = f64::from(scene.fps) / scene.fps_base;
    if !fps.is_finite() || fps <= 0.0 || !frame.is_finite() {
        return Err(PotError::invalid_argument(
            "physics frame rate and frame must be finite and positive",
        ));
    }
    let gravity = scene
        .rigid_body_world
        .as_ref()
        .map_or(DVec3::new(0.0, 0.0, -9.81), |world| {
            DVec3::from_array(world.gravity)
        });
    let collision_meshes = collider_triangles(doc, matrices, members, gate)?;
    let force_fields = force_field_inputs(doc, matrices, members)?;
    let mut result = PhysicsResult {
        deformed_vertices: BTreeMap::new(),
        particles: BTreeMap::new(),
        fluid_particles: BTreeMap::new(),
        fluid_surfaces: BTreeMap::new(),
        paint_colors: BTreeMap::new(),
        paint_weights: BTreeMap::new(),
    };
    for (id, node) in &doc.nodes {
        if !members.contains(id) {
            continue;
        }
        let Some(matrix) = matrices.get(id).copied() else {
            continue;
        };
        if let Some(settings) = node.properties.get("physics_cloth")
            && gate.is_property_active(id, node, "physics_cloth", "cloth")
        {
            let mut mesh = mesh_for_node(doc, node, id)?.clone();
            if let Some(data_block) = node
                .data
                .as_ref()
                .and_then(|data_id| doc.data_blocks.get(data_id))
            {
                crate::eval::attach_vertex_group_weights(&mut mesh, data_block);
            }
            let inverse = invert_matrix(matrix, id)?;
            let local_gravity = inverse.transform_vector3(gravity);
            let local_colliders = if setting_bool(settings, "collision_enabled") == Some(false) {
                Vec::new()
            } else {
                local_collision_triangles(&collision_meshes, id, inverse)
            };
            let normalized = normalize_solver_settings(settings, "cloth");
            let positions = cloth::simulate(
                &mesh,
                &normalized,
                frame,
                fps,
                local_gravity,
                &local_colliders,
            )?;
            result
                .deformed_vertices
                .insert(id.clone(), checked_positions(id, positions)?);
        }
        if let Some(settings) = node.properties.get("physics_soft_body")
            && gate.is_property_active(id, node, "physics_soft_body", "soft_body")
        {
            let mut mesh = mesh_for_node(doc, node, id)?.clone();
            if let Some(data_block) = node
                .data
                .as_ref()
                .and_then(|data_id| doc.data_blocks.get(data_id))
            {
                crate::eval::attach_vertex_group_weights(&mut mesh, data_block);
            }
            let inverse = invert_matrix(matrix, id)?;
            let local_gravity = inverse.transform_vector3(gravity);
            let local_colliders = local_collision_triangles(&collision_meshes, id, inverse);
            let positions =
                softbody::simulate(&mesh, settings, frame, fps, local_gravity, &local_colliders)?;
            result
                .deformed_vertices
                .insert(id.clone(), checked_positions(id, positions)?);
        }
        if let Some(settings) = node.properties.get("physics_particle_emitter")
            && gate.is_property_active(id, node, "physics_particle_emitter", "particle_system")
        {
            let mesh = mesh_for_node(doc, node, id)?;
            let seed = setting_u32(settings, "seed").unwrap_or_else(|| {
                scene
                    .rigid_body_world
                    .as_ref()
                    .map_or(0, |world| world.seed)
            });
            let colliders = collision_meshes
                .iter()
                .filter(|(collider_id, _)| collider_id != id)
                .map(|(_, triangle)| *triangle)
                .collect::<Vec<_>>();
            let textures = load_particle_textures(settings, doc, gate.context.project_root)?;
            let slots = particle_textures::texture_slots(settings, &textures, matrices)?;
            let particles = particles::simulate_textured(
                mesh,
                settings,
                matrix,
                frame,
                fps,
                gravity,
                seed,
                &force_fields,
                &colliders,
                &slots,
            )?;
            if particles.iter().any(|particle| {
                particle.position.iter().any(|value| !value.is_finite())
                    || particle.velocity.iter().any(|value| !value.is_finite())
                    || !particle.birth_frame.is_finite()
            }) {
                return Err(PotError::new(
                    ErrorCode::EvaluationFailed,
                    format!("particle state for `{id}` is not finite"),
                ));
            }
            let system_index = node
                .properties
                .get("physics_particle_emitter_modifier_id")
                .and_then(Value::as_str)
                .map_or(0, |modifier_id| particle_system_index(node, modifier_id));
            let states = particles
                .into_iter()
                .map(|particle| particle_state(particle, system_index))
                .collect::<Vec<_>>();
            if !particle_states_valid(&states) {
                return Err(PotError::new(
                    ErrorCode::EvaluationFailed,
                    format!("particle state for `{id}` is invalid"),
                ));
            }
            result.particles.insert(id.clone(), states);
        }
        if let Some(settings) = node.properties.get("physics_fluid")
            && gate.is_property_active(id, node, "physics_fluid", "fluid")
        {
            let domain = mesh_for_node(doc, node, id)?;
            let seed = setting_u32(settings, "seed").unwrap_or_else(|| {
                scene
                    .rigid_body_world
                    .as_ref()
                    .map_or(0, |world| world.seed)
            });
            let colliders = collision_meshes
                .iter()
                .filter(|(collider_id, _)| collider_id != id)
                .map(|(_, triangle)| *triangle)
                .collect::<Vec<_>>();
            let particles = fluid::simulate(
                domain, settings, matrix, frame, fps, gravity, seed, &colliders,
            )?;
            if particles.iter().any(|position| !position.is_finite()) {
                return Err(PotError::new(
                    ErrorCode::EvaluationFailed,
                    format!("fluid state for `{id}` is not finite"),
                ));
            }
            let local_surface = fluid_surface(&particles, matrix, settings)?;
            result
                .fluid_particles
                .insert(id.clone(), particles.iter().map(DVec3::to_array).collect());
            if !local_surface.vertices.is_empty() {
                result.fluid_surfaces.insert(id.clone(), local_surface);
            }
        }
        if let Some(settings) = node.properties.get("physics_dynamic_paint")
            && gate.is_property_active(id, node, "physics_dynamic_paint", "dynamic_paint")
            && setting_string(settings, "role").unwrap_or("canvas") == "canvas"
        {
            let canvas = mesh_for_node(doc, node, id)?;
            let brushes = dynamic_paint_brush_settings(doc, matrices, settings, members, gate)?;
            let mut colors = vec![[0.0; 4]; canvas.vertices.len()];
            if brushes.is_empty() {
                colors = dynamic_paint::paint(
                    canvas,
                    matrix,
                    &[],
                    setting_f64(settings, "radius").unwrap_or(1.0),
                    setting_color(settings).unwrap_or([1.0, 0.0, 0.0, 1.0]),
                    setting_f64(settings, "strength").unwrap_or(1.0),
                )?;
            } else {
                for brush in &brushes {
                    let painted = dynamic_paint::paint(
                        canvas,
                        matrix,
                        &[brush.position],
                        brush.radius,
                        brush.color,
                        brush.strength,
                    )?;
                    for (current, candidate) in colors.iter_mut().zip(painted) {
                        if candidate[3] > current[3] {
                            *current = candidate;
                        }
                    }
                }
            }
            if colors.len() != canvas.vertices.len()
                || colors.iter().flatten().any(|value| !value.is_finite())
            {
                return Err(PotError::new(
                    ErrorCode::EvaluationFailed,
                    format!("dynamic-paint colors for `{id}` are invalid"),
                ));
            }
            match setting_string(settings, "surface_format").unwrap_or("color") {
                "color" => {
                    result.paint_colors.insert(id.clone(), colors);
                }
                "weight" => {
                    result
                        .paint_weights
                        .insert(id.clone(), colors.iter().map(|color| color[3]).collect());
                }
                _ => {
                    return Err(PotError::with_details(
                        ErrorCode::SceneInvalid,
                        "dynamic-paint surface_format is invalid",
                        json!({"node_id":id}),
                    ));
                }
            }
        }
    }
    Ok(result)
}

fn mesh_for_node<'a>(doc: &'a SceneDoc, node: &Node, id: &Id) -> Result<&'a Mesh> {
    node.data
        .as_ref()
        .and_then(|data_id| doc.data_blocks.get(data_id))
        .and_then(|data| data.mesh.as_ref())
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "physics system requires mesh data",
                json!({"node_id":id}),
            )
        })
}

fn collider_triangles(
    doc: &SceneDoc,
    matrices: &BTreeMap<Id, DMat4>,
    members: &BTreeSet<Id>,
    gate: ModifierGate<'_>,
) -> Result<Vec<(Id, [DVec3; 3])>> {
    let mut triangles = Vec::new();
    for (id, node) in &doc.nodes {
        if !members.contains(id) {
            continue;
        }
        let enabled = node
            .rigid_body
            .as_ref()
            .is_some_and(|body| body.body_type == RigidBodyType::Passive)
            || (node.properties.contains_key("physics_collision")
                && gate.is_property_active(id, node, "physics_collision", "collision"))
            || node
                .properties
                .get("collision_enabled")
                .and_then(Value::as_bool)
                == Some(true);
        if !enabled {
            continue;
        }
        let mesh = mesh_for_node(doc, node, id)?;
        let matrix = matrices.get(id).copied().ok_or_else(|| {
            PotError::with_details(
                ErrorCode::EvaluationFailed,
                "collision mesh transform is missing",
                json!({"node_id":id}),
            )
        })?;
        let positions = mesh
            .vertices
            .iter()
            .map(|vertex| (vertex.id, matrix.transform_point3(vertex.co)))
            .collect::<BTreeMap<_, _>>();
        for triangle in mesh.tessellate().map_err(|error| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                format!("collision mesh `{id}` is invalid: {error}"),
                json!({"node_id":id}),
            )
        })? {
            let points = triangle.map(|vertex_id| positions.get(&vertex_id).copied());
            let [Some(a), Some(b), Some(c)] = points else {
                return Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    "collision triangle references a missing vertex",
                ));
            };
            let triangle = [a, b, c];
            let settings = node.properties.get("physics_collision");
            let outer = settings
                .and_then(|settings| settings.get("thickness_outer"))
                .and_then(Value::as_f64)
                .or_else(|| {
                    settings
                        .and_then(|settings| settings.get("thickness"))
                        .and_then(Value::as_f64)
                })
                .unwrap_or(0.0);
            let inner = settings
                .and_then(|settings| settings.get("thickness_inner"))
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            let normal = (b - a).cross(c - a).normalize_or_zero();
            if outer == 0.0 && inner == 0.0 || normal == DVec3::ZERO {
                triangles.push((id.clone(), triangle));
            } else {
                if outer > 0.0 {
                    triangles.push((id.clone(), triangle.map(|point| point + normal * outer)));
                }
                if inner > 0.0 {
                    triangles.push((id.clone(), triangle.map(|point| point - normal * inner)));
                }
            }
        }
    }
    Ok(triangles)
}

fn local_collision_triangles(
    triangles: &[(Id, [DVec3; 3])],
    subject_id: &Id,
    inverse: DMat4,
) -> Vec<[DVec3; 3]> {
    triangles
        .iter()
        .filter(|(id, _)| id != subject_id)
        .map(|(_, triangle)| triangle.map(|point| inverse.transform_point3(point)))
        .collect()
}

fn force_field_inputs(
    doc: &SceneDoc,
    matrices: &BTreeMap<Id, DMat4>,
    members: &BTreeSet<Id>,
) -> Result<Vec<(DVec3, &'static str, f64, f64)>> {
    let mut inputs = Vec::new();
    for (id, node) in &doc.nodes {
        if !members.contains(id) {
            continue;
        }
        let Some(field) = node.force_field.as_ref() else {
            continue;
        };
        let matrix = matrices.get(id).copied().ok_or_else(|| {
            PotError::with_details(
                ErrorCode::EvaluationFailed,
                "force-field transform is missing",
                json!({"node_id":id}),
            )
        })?;
        let kind = match field.field_type {
            ForceFieldType::Wind => "wind",
            ForceFieldType::Vortex => "vortex",
            ForceFieldType::Force => "force",
        };
        inputs.push((
            matrix.transform_point3(DVec3::ZERO),
            kind,
            field.strength,
            field.falloff,
        ));
    }
    Ok(inputs)
}

#[derive(Clone, Copy)]
struct DynamicPaintBrush {
    position: DVec3,
    radius: f64,
    color: [f64; 4],
    strength: f64,
}
fn dynamic_paint_brush_settings(
    doc: &SceneDoc,
    matrices: &BTreeMap<Id, DMat4>,
    canvas_settings: &Value,
    members: &BTreeSet<Id>,
    gate: ModifierGate<'_>,
) -> Result<Vec<DynamicPaintBrush>> {
    let mut brush_ids = BTreeSet::new();
    if let Some(values) = canvas_settings.get("brushes").and_then(Value::as_array) {
        for value in values {
            let value = value.as_str().ok_or_else(|| {
                PotError::new(
                    ErrorCode::InvalidOperation,
                    "dynamic-paint brush IDs must be strings",
                )
            })?;
            let id = Id::new(value.to_owned())
                .map_err(|error| PotError::new(ErrorCode::InvalidOperation, error.message))?;
            if doc.nodes.get(&id).is_some_and(|node| {
                !gate.is_property_active(&id, node, "physics_dynamic_paint", "dynamic_paint")
            }) {
                continue;
            }
            if !members.contains(&id) {
                return Err(PotError::with_details(
                    ErrorCode::TargetNotFound,
                    "dynamic-paint brush is not in the active Scene",
                    json!({"brush_id":id}),
                ));
            }
            brush_ids.insert(id);
        }
    } else {
        for (id, node) in &doc.nodes {
            if members.contains(id)
                && gate.is_property_active(id, node, "physics_dynamic_paint", "dynamic_paint")
                && node
                    .properties
                    .get("physics_dynamic_paint")
                    .and_then(|settings| setting_string(settings, "role"))
                    == Some("brush")
            {
                brush_ids.insert(id.clone());
            }
        }
    }
    let canvas_radius = setting_f64(canvas_settings, "radius").unwrap_or(1.0);
    let canvas_color = setting_color(canvas_settings).unwrap_or([1.0, 0.0, 0.0, 1.0]);
    let canvas_strength = setting_f64(canvas_settings, "strength").unwrap_or(1.0);
    let mut brushes = Vec::with_capacity(brush_ids.len());
    for id in brush_ids {
        let node = doc.nodes.get(&id).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::TargetNotFound,
                "dynamic-paint brush Object was not found",
                json!({"brush_id":id}),
            )
        })?;
        let settings = node
            .properties
            .get("physics_dynamic_paint")
            .ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::InvalidOperation,
                    "dynamic-paint target is not configured as a brush",
                    json!({"brush_id":id}),
                )
            })?;
        if setting_string(settings, "role") != Some("brush") {
            return Err(PotError::with_details(
                ErrorCode::InvalidOperation,
                "dynamic-paint target is not configured as a brush",
                json!({"brush_id":id}),
            ));
        }
        let matrix = matrices.get(&id).copied().ok_or_else(|| {
            PotError::with_details(
                ErrorCode::EvaluationFailed,
                "dynamic-paint brush transform is missing",
                json!({"node_id":id}),
            )
        })?;
        brushes.push(DynamicPaintBrush {
            position: matrix.transform_point3(DVec3::ZERO),
            radius: setting_f64(settings, "radius").unwrap_or(canvas_radius),
            color: setting_color(settings).unwrap_or(canvas_color),
            strength: setting_f64(settings, "strength").unwrap_or(canvas_strength),
        });
    }
    Ok(brushes)
}

fn fluid_surface(particles: &[DVec3], matrix: DMat4, settings: &Value) -> Result<Mesh> {
    if particles.is_empty() {
        return Ok(Mesh::default());
    }
    let inverse = matrix.inverse();
    if !inverse.is_finite() {
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            "fluid domain transform cannot be inverted",
        ));
    }
    let radius = setting_f64(settings, "particle_radius").unwrap_or(0.15);
    let stride = particles.len().div_ceil(32);
    let mut data = MetaballData::default();
    let mut minimum = DVec3::splat(f64::INFINITY);
    let mut maximum = DVec3::splat(f64::NEG_INFINITY);
    data.elements = particles
        .iter()
        .step_by(stride)
        .take(32)
        .map(|position| {
            minimum = minimum.min(*position);
            maximum = maximum.max(*position);
            MetaballElement {
                co: *position,
                radius: radius * 2.0,
                ..MetaballElement::default()
            }
        })
        .collect();
    let element_count = data.elements.len() as f64;
    let falloff = radius * 2.0 * (2.0 * element_count - 1.0).sqrt();
    let base_extent = maximum - minimum + DVec3::splat(2.0 * falloff);
    if !base_extent.is_finite() {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "liquid surface bounds exceed numeric limits",
        ));
    }
    data.resolution = (radius * 0.75)
        .clamp(0.08, 0.5)
        .max(base_extent.max_element() / 30.0);
    data.render_resolution = data.resolution;
    let mut mesh = crate::geom::metaball::to_mesh(&data)?;
    for vertex in &mut mesh.vertices {
        vertex.co = inverse.transform_point3(vertex.co);
    }
    Ok(mesh)
}

fn invert_matrix(matrix: DMat4, id: &Id) -> Result<DMat4> {
    let inverse = matrix.inverse();
    if inverse.is_finite() {
        Ok(inverse)
    } else {
        Err(PotError::with_details(
            ErrorCode::EvaluationFailed,
            "physics Object transform cannot be inverted",
            json!({"node_id":id}),
        ))
    }
}

fn checked_positions(id: &Id, positions: Vec<DVec3>) -> Result<Vec<[f64; 3]>> {
    if positions.iter().any(|position| !position.is_finite()) {
        return Err(PotError::with_details(
            ErrorCode::EvaluationFailed,
            "simulated mesh contains a non-finite vertex",
            json!({"node_id":id}),
        ));
    }
    Ok(positions
        .into_iter()
        .map(|position| position.to_array())
        .collect())
}

fn setting_bool(settings: &Value, key: &str) -> Option<bool> {
    settings.get(key).and_then(Value::as_bool)
}

fn setting_f64(settings: &Value, key: &str) -> Option<f64> {
    settings
        .get(key)
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
}

fn setting_u32(settings: &Value, key: &str) -> Option<u32> {
    settings
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
}

fn setting_string<'a>(settings: &'a Value, key: &str) -> Option<&'a str> {
    settings.get(key).and_then(Value::as_str)
}

fn setting_color(settings: &Value) -> Option<[f64; 4]> {
    let values = settings.get("color")?.as_array()?;
    let [r, g, b, a] = values.as_slice() else {
        return None;
    };
    Some([r.as_f64()?, g.as_f64()?, b.as_f64()?, a.as_f64()?])
}
#[cfg(test)]
mod tests {
    use glam::DQuat;

    use super::{
        ParticleLifeState, ParticlePathInterpolation, ParticlePathKey, ParticleState,
        particle_orientation, particle_state, particle_states_valid, particles::Particle,
    };

    #[test]
    fn interpolated_child_allows_virtual_representative_parent() {
        let mut states = vec![ParticleState::default(), ParticleState::default()];
        states.push(ParticleState {
            is_child: true,
            parent_index: Some(1),
            parent_indices: [0, 1, 0],
            parent_weights: [0.75, 0.25, 0.0],
            parent_count: 2,
            ..ParticleState::default()
        });

        assert!(particle_states_valid(&states));
    }
    fn particle_for_state(
        path_interpolation: ParticlePathInterpolation,
        path: Vec<ParticlePathKey>,
        birth_rotation: [f64; 4],
        current_path_rotation: [f64; 4],
    ) -> Particle {
        Particle {
            position: [0.0; 3],
            birth_position: [0.0; 3],
            normal: [0.0, 0.0, 1.0],
            birth_rotation,
            current_path_rotation: Some(current_path_rotation),
            path_root_rotation: None,
            path_interpolation,
            velocity: [0.0; 3],
            birth_frame: 0.0,
            death_frame: 1.0,
            life_state: ParticleLifeState::Alive,
            is_child: false,
            parent_index: None,
            parent_indices: [0; 3],
            parent_weights: [0.0; 3],
            parent_count: 0,
            path,
            size: 1.0,
            mass: 1.0,
        }
    }

    #[test]
    fn particle_current_rotation_is_independent_of_path_frames() {
        let birth_rotation = DQuat::from_rotation_z(0.7).to_array();
        let current_path_rotation = DQuat::from_rotation_y(0.4).to_array();
        let path = vec![ParticlePathKey {
            time: 0.0,
            position: [0.0; 3],
            velocity: [0.0; 3],
            rotation: current_path_rotation,
        }];
        for interpolation in [
            ParticlePathInterpolation::HairCardinal,
            ParticlePathInterpolation::HairBspline,
            ParticlePathInterpolation::HairCache,
        ] {
            let state = particle_state(
                particle_for_state(
                    interpolation,
                    path.clone(),
                    birth_rotation,
                    current_path_rotation,
                ),
                0,
            );
            assert_eq!(state.rotation, birth_rotation);
        }
        let keyed = particle_state(
            particle_for_state(
                ParticlePathInterpolation::Hermite,
                path.clone(),
                birth_rotation,
                current_path_rotation,
            ),
            0,
        );
        assert_eq!(keyed.rotation, current_path_rotation);

        let linear = particle_state(
            particle_for_state(
                ParticlePathInterpolation::Linear,
                path.clone(),
                birth_rotation,
                current_path_rotation,
            ),
            0,
        );
        let normal_orientation = particle_orientation([0.0, 0.0, 1.0]);
        assert_eq!(linear.rotation, normal_orientation);
        let no_path = particle_state(
            particle_for_state(
                ParticlePathInterpolation::HairCache,
                Vec::new(),
                birth_rotation,
                current_path_rotation,
            ),
            0,
        );
        assert_eq!(no_path.rotation, normal_orientation);
    }
}
