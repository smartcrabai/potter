//! Deterministic physics solvers and disposable per-frame result caching.

mod cache;
pub mod cloth;
pub mod dynamic_paint;
pub mod fluid;
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
    model::{ForceFieldType, Id, Modifier, Node, SceneDoc},
    physics_support,
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

/// Substep budget for a solver advancing `frame` with `substeps` steps per frame.
///
/// Exceeding the budget is a reported limit, not a failure: the feature names the solver.
fn frame_steps(frame: f64, substeps: usize, feature_id: &str) -> Result<usize> {
    let requested = frame.abs() * substeps as f64;
    if !requested.is_finite() || requested > MAX_TOTAL_STEPS as f64 {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "simulation frame exceeds the bounded step budget",
            json!({"feature_id":feature_id,"requested_steps":requested,"max_steps":MAX_TOTAL_STEPS,"status":"not_supported"}),
        ));
    }
    Ok(requested.ceil().max(1.0) as usize)
}

/// A solver limitation that a fluid domain's output carries (the result is cached, so it is serialized).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FluidWarning {
    /// Domain object whose output the limitation affects.
    pub node_id: String,
    /// Human-readable description of the limitation.
    pub message: String,
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
    /// Solver limitations reported for the fluid domains of this frame.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fluid_warnings: Vec<FluidWarning>,
}

/// Deterministic simulation outputs for one scene and frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhysicsResult {
    /// Solver limitations reported for the fluid domains of this frame.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fluid_warnings: Vec<FluidWarning>,
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
    pub velocity: [f64; 3],
    pub birth_frame: f64,
    pub death_frame: f64,
    pub life_state: ParticleLifeState,
    pub system_index: u32,
    pub is_child: bool,
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
            velocity: [0.0; 3],
            birth_frame: 0.0,
            death_frame: 0.0,
            life_state: ParticleLifeState::Alive,
            system_index: 0,
            is_child: false,
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

fn scene_members(doc: &SceneDoc, scene_id: &Id) -> Result<BTreeSet<Id>> {
    let scene = doc.scenes.get(scene_id).ok_or_else(|| {
        PotError::new(
            ErrorCode::SceneNotFound,
            format!("scene `{scene_id}` was not found"),
        )
    })?;
    collection_members(doc, &scene.root_collection)
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
    if !gate.is_active(node_id, modifier) {
        return Ok(StackModifierResult {
            mesh: input_mesh,
            particles: None,
        });
    }
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
                let requests_unborn =
                    candidate.params.get("show_unborn").and_then(Value::as_bool) == Some(true);
                requests_unborn
                    && gate.is_active(target_id, candidate)
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
    {
        return Ok(StackModifierResult { mesh, particles });
    }
    let matrix = matrices.get(node_id).copied().ok_or_else(|| {
        PotError::with_details(
            ErrorCode::EvaluationFailed,
            "physics modifier object transform is missing",
            json!({"node_id":node_id}),
        )
    })?;
    let members = scene_members(doc, scene_id)?;
    let context = FrameContext::new(doc, scene_id, frame, matrices, &members, gate)?;
    let mut mesh = input_mesh;
    let mut particles = None;
    match system {
        "cloth" | "soft_body" => {
            let colliders =
                collider_triangles(doc, matrices, &members, gate, ColliderConsumer::Surface)?;
            let positions = simulate_surface(
                &context, node_id, system, settings, &mesh, matrix, &colliders,
            )?;
            set_vertex_positions(&mut mesh, positions, node_id)?;
        }
        "particle_system" => {
            let seed = setting_u32(settings, "seed").unwrap_or(context.seed);
            let triangles =
                collider_triangles(doc, matrices, &members, gate, ColliderConsumer::Point)?;
            let colliders = point_colliders(&triangles, node_id);
            let fields = force_field_inputs(doc, matrices, &members)?;
            let states = particles::simulate_lifecycle(
                &mesh,
                &solver_settings,
                matrix,
                frame,
                context.fps,
                context.gravity,
                seed,
                &fields,
                &colliders,
            )?
            .into_iter()
            .map(|particle| ParticleState {
                position: particle.position,
                birth_position: particle.birth_position,
                normal: particle.normal,
                rotation: particle_orientation(particle.normal),
                birth_rotation: particle.birth_rotation,
                velocity: particle.velocity,
                birth_frame: particle.birth_frame,
                death_frame: particle.death_frame,
                life_state: particle.life_state,
                system_index,
                is_child: particle.is_child,
                size: particle.size,
                mass: particle.mass,
            })
            .collect::<Vec<_>>();
            particles = Some(states);
        }
        "fluid" => {
            let fluid_type = fluid_modifier_role(modifier);
            if !matches!(fluid_type, "NONE" | "DOMAIN" | "FLOW" | "EFFECTOR") {
                return Err(PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "fluid modifier type is invalid",
                    json!({"node_id":node_id,"modifier_id":modifier.id,"fluid_type":fluid_type}),
                ));
            }
            if fluid_type == "DOMAIN" {
                // The physics result reports the same emitter; this surface only needs the positions.
                let (samples, _) = simulate_fluid_domain(&context, &mesh, matrix, settings)?;
                if !samples.is_empty() {
                    let radius = fluid_particle_radius(settings, &mesh, matrix)?;
                    mesh = fluid_surface(&samples, matrix, radius)?;
                }
            }
        }
        "dynamic_paint" if setting_string(settings, "role").unwrap_or("canvas") == "canvas" => {
            let colors = paint_canvas(&context, &mesh, matrix, settings)?;
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
        "dynamic_paint" | "collision" => {}
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

fn normalize_solver_settings(settings: &Value, system: &str) -> Value {
    let mut normalized = settings.clone();
    let Some(properties) = normalized.as_object_mut() else {
        return normalized;
    };
    if system == "particle_system" {
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
    normalized
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
    let mut result = solver::evaluate(
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
        return Ok(result);
    }
    if result
        .as_ref()
        .is_some_and(|cached| cached.physics_cache_key.as_deref() == Some(key.as_str()))
    {
        return Ok(result);
    }
    if result.is_none()
        && let Some(directory) = cache_directory
        && let Some(cached) = cache::load(directory, &key)?
    {
        if cached.physics_cache_key.as_deref() == Some(key.as_str()) {
            return Ok(Some(cached));
        }
        result = Some(cached);
    }
    let physics = evaluate_physics(doc, scene_id, frame, base_world_matrices, &members, gate)?;
    let mut result = result.unwrap_or_else(|| empty_result(key.clone()));
    result.deformed_vertices = physics.deformed_vertices;
    result.fluid_particles = physics.fluid_particles;
    result.fluid_warnings = physics.fluid_warnings;
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
        fluid_warnings: Vec::new(),
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
    let context = FrameContext::new(doc, scene_id, frame, matrices, members, gate)?;
    let force_fields = force_field_inputs(doc, matrices, members)?;
    let mut result = PhysicsResult {
        fluid_warnings: Vec::new(),
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
        for (property, system) in [
            ("physics_cloth", "cloth"),
            ("physics_soft_body", "soft_body"),
        ] {
            if let Some(settings) = node.properties.get(property)
                && gate.is_property_active(id, node, property, system)
            {
                let mesh = mesh_with_vertex_groups(doc, node, id)?;
                let colliders =
                    collider_triangles(doc, matrices, members, gate, ColliderConsumer::Surface)?;
                let positions =
                    simulate_surface(&context, id, system, settings, &mesh, matrix, &colliders)?;
                result
                    .deformed_vertices
                    .insert(id.clone(), checked_positions(id, positions)?);
            }
        }
        if let Some(settings) = node.properties.get("physics_particle_emitter")
            && gate.is_property_active(id, node, "physics_particle_emitter", "particle_system")
        {
            let mesh = mesh_for_node(doc, node, id)?;
            let seed = setting_u32(settings, "seed").unwrap_or(context.seed);
            let triangles =
                collider_triangles(doc, matrices, members, gate, ColliderConsumer::Point)?;
            let colliders = point_colliders(&triangles, id);
            let particles = particles::simulate(
                mesh,
                settings,
                matrix,
                frame,
                context.fps,
                context.gravity,
                seed,
                &force_fields,
                &colliders,
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
            result.particles.insert(
                id.clone(),
                particles
                    .into_iter()
                    .map(|particle| ParticleState {
                        position: particle.position,
                        birth_position: particle.birth_position,
                        normal: particle.normal,
                        rotation: particle_orientation(particle.normal),
                        birth_rotation: particle.birth_rotation,
                        velocity: particle.velocity,
                        birth_frame: particle.birth_frame,
                        death_frame: particle.death_frame,
                        life_state: particle.life_state,
                        system_index,
                        is_child: particle.is_child,
                        size: particle.size,
                        mass: particle.mass,
                    })
                    .collect(),
            );
        }
        if let Some(settings) = node.properties.get("physics_fluid")
            && let Some(modifier) = active_modifier(id, node, "fluid", gate)
        {
            let fluid_type = fluid_modifier_role(modifier);
            if !matches!(fluid_type, "NONE" | "DOMAIN" | "FLOW" | "EFFECTOR") {
                return Err(PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "fluid modifier type is invalid",
                    json!({"node_id":id,"modifier_id":modifier.id,"fluid_type":fluid_type}),
                ));
            }
            if fluid_type == "DOMAIN" {
                let domain = mesh_for_node(doc, node, id)?;
                let (particles, spawns_inside_effector) =
                    simulate_fluid_domain(&context, domain, matrix, settings)?;
                if particles.iter().any(|position| !position.is_finite()) {
                    return Err(PotError::new(
                        ErrorCode::EvaluationFailed,
                        format!("fluid state for `{id}` is not finite"),
                    ));
                }
                if spawns_inside_effector {
                    result.fluid_warnings.push(FluidWarning {
                        node_id: id.to_string(),
                        message: "a flow spawns liquid inside an effector; Potter does not push that liquid out, and Blender slows it instead (solver parity is a known gap)".to_owned(),
                    });
                }
                if !particles.is_empty() {
                    let radius = fluid_particle_radius(settings, domain, matrix)?;
                    let local_surface = fluid_surface(&particles, matrix, radius)?;
                    result
                        .fluid_particles
                        .insert(id.clone(), particles.iter().map(DVec3::to_array).collect());
                    if !local_surface.vertices.is_empty() {
                        result.fluid_surfaces.insert(id.clone(), local_surface);
                    }
                }
            }
        }
        if let Some(settings) = node.properties.get("physics_dynamic_paint")
            && gate.is_property_active(id, node, "physics_dynamic_paint", "dynamic_paint")
            && setting_string(settings, "role").unwrap_or("canvas") == "canvas"
            && active_modifier(id, node, "dynamic_paint", gate).is_some()
        {
            let canvas = mesh_for_node(doc, node, id)?;
            let colors = paint_canvas(&context, canvas, matrix, settings)?;
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

fn active_modifier<'a>(
    node_id: &Id,
    node: &'a Node,
    modifier_type: &str,
    gate: ModifierGate<'_>,
) -> Option<&'a Modifier> {
    node.modifiers.iter().find(|modifier| {
        modifier.modifier_type == modifier_type && gate.is_active(node_id, modifier)
    })
}

fn fluid_modifier_role(modifier: &Modifier) -> &str {
    modifier
        .params
        .get("fluid_type")
        .and_then(Value::as_str)
        .unwrap_or("NONE")
}

fn fluid_mesh_triangles(
    doc: &SceneDoc,
    matrices: &BTreeMap<Id, DMat4>,
    node: &Node,
    node_id: &Id,
) -> Result<Vec<[DVec3; 3]>> {
    let mesh = mesh_for_node(doc, node, node_id)?;
    let matrix = matrices.get(node_id).copied().ok_or_else(|| {
        PotError::with_details(
            ErrorCode::EvaluationFailed,
            "fluid object transform is missing",
            json!({"node_id":node_id}),
        )
    })?;
    let positions = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, matrix.transform_point3(vertex.co)))
        .collect::<BTreeMap<_, _>>();
    let mesh_triangles = mesh.tessellate().map_err(|error| {
        PotError::with_details(
            ErrorCode::SceneInvalid,
            format!("fluid mesh `{node_id}` is invalid: {error}"),
            json!({"node_id":node_id}),
        )
    })?;
    let mut triangles = Vec::with_capacity(mesh_triangles.len());
    for triangle in mesh_triangles {
        let points = triangle.map(|vertex_id| positions.get(&vertex_id).copied());
        let [Some(a), Some(b), Some(c)] = points else {
            return Err(PotError::with_details(
                ErrorCode::SceneInvalid,
                "fluid triangle references a missing vertex",
                json!({"node_id":node_id}),
            ));
        };
        triangles.push([a, b, c]);
    }
    Ok(triangles)
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

fn fluid_surface(particles: &[DVec3], matrix: DMat4, radius: f64) -> Result<Mesh> {
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

/// Colour from a 3- or 4-channel array under `key`; a 3-channel colour is opaque.
fn color_setting(settings: &Value, key: &str) -> Option<[f64; 4]> {
    let values = settings.get(key)?.as_array()?;
    match values.as_slice() {
        [r, g, b, a] => Some([r.as_f64()?, g.as_f64()?, b.as_f64()?, a.as_f64()?]),
        [r, g, b] => Some([r.as_f64()?, g.as_f64()?, b.as_f64()?, 1.0]),
        _ => None,
    }
}

fn setting_color(settings: &Value) -> Option<[f64; 4]> {
    color_setting(settings, "color")
}

/// Solver window of a Blender point cache.
///
/// Scene frames before `start` keep the input state, and frames after `end` hold the state at
/// `end` (`pointcache.cc`, `cloth.cc`). The solver is sampled at `frame - start`.
#[derive(Clone, Copy)]
struct CacheWindow {
    start: f64,
    end: f64,
}

impl CacheWindow {
    fn solver_frame(self, frame: f64) -> f64 {
        (frame.min(self.end) - self.start).max(0.0)
    }
}

fn frame_window(start: f64, end: f64) -> Result<CacheWindow> {
    if end < start {
        return Err(PotError::invalid_argument(
            "point cache frame_end must not be before frame_start",
        ));
    }
    Ok(CacheWindow { start, end })
}

/// Window of a nested `point_cache` block, with Blender's defaults when the block is absent.
fn point_cache_window(settings: &Value) -> Result<CacheWindow> {
    let Some(cache) = settings.get("point_cache") else {
        return frame_window(1.0, 250.0);
    };
    if !cache.is_object() {
        return Err(PotError::invalid_argument("point_cache must be an object"));
    }
    physics_support::reject_unhonoured(
        "point_cache",
        cache,
        physics_support::honoured_properties("point_cache"),
    )?;
    let start = setting_number(cache, "frame_start", 1.0, "point cache", "a finite number")?;
    let end = setting_number(cache, "frame_end", 250.0, "point cache", "a finite number")?;
    frame_window(start, end)
}

/// Window of a fluid domain's `cache_frame_start` and `cache_frame_end`.
fn domain_cache_window(settings: &Value) -> Result<CacheWindow> {
    let start = setting_number(
        settings,
        "cache_frame_start",
        1.0,
        "fluid",
        "a finite number",
    )?;
    let end = setting_number(
        settings,
        "cache_frame_end",
        250.0,
        "fluid",
        "a finite number",
    )?;
    frame_window(start, end)
}

fn bool_setting(settings: &Value, key: &str, default: bool, family: &str) -> Result<bool> {
    settings.get(key).map_or(Ok(default), |value| {
        value
            .as_bool()
            .ok_or_else(|| PotError::invalid_argument(format!("{family} {key} must be a boolean")))
    })
}

fn unsupported_feature(feature_id: &str, reason: &str) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        reason,
        json!({"feature_id":feature_id,"reason":reason,"status":"not_supported"}),
    )
}

/// Objects of a collection and of its child collections.
fn collection_members(doc: &SceneDoc, root: &Id) -> Result<BTreeSet<Id>> {
    let mut pending = vec![root.clone()];
    let mut visited = BTreeSet::new();
    let mut members = BTreeSet::new();
    while let Some(collection_id) = pending.pop() {
        if !visited.insert(collection_id.clone()) {
            continue;
        }
        let collection = doc.collections.get(&collection_id).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "collection is missing",
                json!({"collection_id":collection_id}),
            )
        })?;
        members.extend(collection.objects.iter().cloned());
        pending.extend(collection.children.iter().cloned());
    }
    Ok(members)
}

/// Members of the collection that a Blender collision or effector reference names.
///
/// `None` when the reference is unset, which means every collider applies. A reference that does
/// not resolve to a Potter Collection is refused, because Blender Collection pointers are not
/// mapped by the importer.
fn collection_filter(
    doc: &SceneDoc,
    value: Option<&Value>,
    feature_id: &str,
) -> Result<Option<BTreeSet<Id>>> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    let collection = value
        .as_str()
        .and_then(|name| Id::new(name.to_owned()).ok())
        .filter(|id| doc.collections.contains_key(id));
    let Some(collection) = collection else {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "Blender collision collection reference is not a Potter Collection",
            json!({"feature_id":feature_id,"status":"not_supported"}),
        ));
    };
    collection_members(doc, &collection).map(Some)
}

/// Solver that consumes a collider set.
///
/// Surface solvers (cloth and soft bodies) apply the full collision settings. Point solvers
/// (particles and liquid) apply only `thickness_outer`, as a two-sided offset of each surface.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ColliderConsumer {
    Surface,
    Point,
}

/// World-space collision triangle with the Blender collision settings its consumer honours.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColliderTriangle {
    pub points: [DVec3; 3],
    /// Distance kept on the front side of the surface (`thickness_outer`).
    pub thickness_outer: f64,
    /// Distance kept on the back side of the surface (`thickness_inner`). Only soft bodies apply
    /// it; cloth and point solvers ignore it, as Blender does.
    pub thickness_inner: f64,
    /// Only motion from the front side collides (`use_culling`).
    pub culling: bool,
    /// Coulomb friction of the surface in percent (`cloth_friction`). Only cloth applies it.
    pub friction: f64,
}

/// Collision triangles in world space that `consumer` honours, each tagged with its object.
///
/// Blender collides with objects whose COLLISION modifier has `use` set. Passive rigid bodies are
/// not colliders unless they also carry a COLLISION modifier.
fn collider_triangles(
    doc: &SceneDoc,
    matrices: &BTreeMap<Id, DMat4>,
    members: &BTreeSet<Id>,
    gate: ModifierGate<'_>,
    consumer: ColliderConsumer,
) -> Result<Vec<(Id, ColliderTriangle)>> {
    let honoured: &[&str] = match consumer {
        ColliderConsumer::Surface => physics_support::honoured_properties("collision"),
        ColliderConsumer::Point => &physics_support::POINT_COLLISION_HONOURED[..],
    };
    let mut triangles = Vec::new();
    for (id, node) in &doc.nodes {
        if !members.contains(id) {
            continue;
        }
        let Some(settings) = node.properties.get("physics_collision") else {
            continue;
        };
        if !gate.is_property_active(id, node, "physics_collision", "collision")
            || !bool_setting(settings, "use", true, "collision")?
        {
            continue;
        }
        physics_support::reject_unhonoured("collision", settings, honoured)?;
        let thickness_outer = nonnegative_setting(settings, "thickness_outer", 0.02, "collision")?;
        let thickness_inner = nonnegative_setting(settings, "thickness_inner", 0.2, "collision")?;
        let culling = bool_setting(settings, "use_culling", true, "collision")?;
        let friction = nonnegative_setting(settings, "cloth_friction", 5.0, "collision")?;
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
            triangles.push((
                id.clone(),
                ColliderTriangle {
                    points: [a, b, c],
                    thickness_outer,
                    thickness_inner,
                    culling,
                    friction,
                },
            ));
        }
    }
    Ok(triangles)
}

/// Colliders that `subject` sees, in the subject's local frame.
///
/// `collection` restricts colliders to a consumer's collection, and `None` keeps every collider.
/// A subject never collides with itself. Thickness values stay in the subject's local units.
fn local_colliders(
    triangles: &[(Id, ColliderTriangle)],
    subject: &Id,
    collection: Option<&BTreeSet<Id>>,
    inverse: DMat4,
) -> Vec<ColliderTriangle> {
    triangles
        .iter()
        .filter(|(id, _)| id != subject && collection.is_none_or(|members| members.contains(id)))
        .map(|(_, triangle)| ColliderTriangle {
            points: triangle.points.map(|point| inverse.transform_point3(point)),
            ..*triangle
        })
        .collect()
}

/// Two-sided world-space triangles for point solvers.
///
/// Each surface is offset along its normal by its `thickness_outer`, and a subject never collides
/// with itself.
fn point_colliders(triangles: &[(Id, ColliderTriangle)], subject: &Id) -> Vec<[DVec3; 3]> {
    triangles
        .iter()
        .filter(|(id, _)| id != subject)
        .map(|(_, triangle)| {
            let [a, b, c] = triangle.points;
            let normal = (b - a).cross(c - a).normalize_or_zero();
            triangle
                .points
                .map(|point| point + normal * triangle.thickness_outer)
        })
        .collect()
}

/// Scene inputs that every physics evaluation at one frame shares.
#[derive(Clone, Copy)]
struct FrameContext<'a> {
    doc: &'a SceneDoc,
    matrices: &'a BTreeMap<Id, DMat4>,
    members: &'a BTreeSet<Id>,
    gate: ModifierGate<'a>,
    frame: f64,
    fps: f64,
    gravity: DVec3,
    seed: u32,
}

impl<'a> FrameContext<'a> {
    fn new(
        doc: &'a SceneDoc,
        scene_id: &Id,
        frame: f64,
        matrices: &'a BTreeMap<Id, DMat4>,
        members: &'a BTreeSet<Id>,
        gate: ModifierGate<'a>,
    ) -> Result<Self> {
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
        let (gravity, seed) = scene
            .rigid_body_world
            .as_ref()
            .map_or((DVec3::new(0.0, 0.0, -9.81), 0), |world| {
                (DVec3::from_array(world.gravity), world.seed)
            });
        Ok(Self {
            doc,
            matrices,
            members,
            gate,
            frame,
            fps,
            gravity,
            seed,
        })
    }
}

/// Mesh of a scene node with its vertex-group weights attached.
fn mesh_with_vertex_groups(doc: &SceneDoc, node: &Node, id: &Id) -> Result<Mesh> {
    let mut mesh = mesh_for_node(doc, node, id)?.clone();
    if let Some(data_block) = node
        .data
        .as_ref()
        .and_then(|data_id| doc.data_blocks.get(data_id))
    {
        crate::eval::attach_vertex_group_weights(&mut mesh, data_block);
    }
    Ok(mesh)
}

/// Cloth or soft-body vertex positions for `subject` at the context frame, in its local space.
///
/// The solver runs inside the point-cache window. Cloth takes its collision toggle and collection
/// from the nested `collision_settings`; soft bodies take `collision_collection`.
fn simulate_surface(
    context: &FrameContext<'_>,
    subject: &Id,
    system: &str,
    settings: &Value,
    mesh: &Mesh,
    matrix: DMat4,
    colliders: &[(Id, ColliderTriangle)],
) -> Result<Vec<DVec3>> {
    let window = point_cache_window(settings)?;
    let inverse = invert_matrix(matrix, subject)?;
    let (collision_enabled, collection) = if system == "cloth" {
        let collision = settings.get("collision_settings");
        if let Some(collision) = collision {
            physics_support::reject_unhonoured(
                "cloth_collision",
                collision,
                physics_support::honoured_properties("cloth_collision"),
            )?;
        }
        let enabled = match collision {
            Some(collision) => bool_setting(collision, "use_collision", true, "cloth collision")?,
            None => true,
        };
        let collection = collection_filter(
            context.doc,
            collision.and_then(|collision| collision.get("collection")),
            "physics.cloth_collision.collection",
        )?;
        (enabled, collection)
    } else {
        let collection = collection_filter(
            context.doc,
            settings.get("collision_collection"),
            "physics.soft_body.collision_collection",
        )?;
        (true, collection)
    };
    let surface_colliders = if collision_enabled {
        local_colliders(colliders, subject, collection.as_ref(), inverse)
    } else {
        Vec::new()
    };
    let local_gravity = inverse.transform_vector3(context.gravity);
    let solver_frame = window.solver_frame(context.frame);
    if system == "cloth" {
        cloth::simulate(
            mesh,
            settings,
            solver_frame,
            context.fps,
            local_gravity,
            &surface_colliders,
        )
    } else {
        softbody::simulate(
            mesh,
            settings,
            solver_frame,
            context.fps,
            local_gravity,
            &surface_colliders,
        )
    }
}

/// Rejects Blender fluid domains that Potter does not simulate.
///
/// A domain without `domain_type` is a Potter-native liquid domain with absolute units.
fn validate_fluid_domain(settings: &Value) -> Result<()> {
    let Some(kind) = setting_string(settings, "domain_type") else {
        return Ok(());
    };
    match kind {
        "LIQUID" => {
            let system = physics_support::fluid_domain_system(kind);
            physics_support::reject_unhonoured(
                system,
                settings,
                physics_support::honoured_properties(system),
            )
        }
        "GAS" => Err(unsupported_feature(
            "physics.fluid.smoke_fire",
            "gas (smoke and fire) domains are not supported; only liquid SPH simulation is",
        )),
        _ => Err(physics_support::unsupported_property(
            "fluid_domain",
            "domain_type",
        )),
    }
}

/// Liquid particle radius in world units.
///
/// Blender stores `particle_radius` as a factor of the domain cell size (`rna_fluid.cc`). A Blender
/// domain, one that carries a `domain_type`, multiplies that factor by its cell size, taken as the
/// longest world-space side divided by `resolution_max`. A Potter-native domain keeps
/// `particle_radius` as an absolute world radius.
fn fluid_particle_radius(settings: &Value, domain: &Mesh, matrix: DMat4) -> Result<f64> {
    if settings.get("domain_type").is_none() {
        return Ok(setting_f64(settings, "particle_radius").unwrap_or(0.15));
    }
    let factor = nonnegative_setting(settings, "particle_radius", 1.0, "fluid")?;
    let resolution_max =
        setting_number(settings, "resolution_max", 32.0, "fluid", "a finite number")?;
    if resolution_max < 1.0 {
        return Err(PotError::invalid_argument(
            "fluid resolution_max must be at least 1",
        ));
    }
    let radius = factor * domain_world_extent(domain, matrix) / resolution_max;
    if !radius.is_finite() || radius <= 0.0 {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "fluid domain particle radius must be positive",
        ));
    }
    Ok(radius)
}

/// Longest world-space side of the domain's vertex bounds.
fn domain_world_extent(domain: &Mesh, matrix: DMat4) -> f64 {
    let mut minimum = DVec3::splat(f64::INFINITY);
    let mut maximum = DVec3::splat(f64::NEG_INFINITY);
    for vertex in &domain.vertices {
        let point = matrix.transform_point3(vertex.co);
        minimum = minimum.min(point);
        maximum = maximum.max(point);
    }
    (maximum - minimum).max_element()
}

/// Liquid emitters from the scene's active FLOW objects.
fn fluid_flow_sources(context: &FrameContext<'_>) -> Result<Vec<fluid::FlowSource>> {
    let mut flows = Vec::new();
    for (node_id, node) in &context.doc.nodes {
        if !context.members.contains(node_id) {
            continue;
        }
        let Some(modifier) = active_modifier(node_id, node, "fluid", context.gate) else {
            continue;
        };
        if fluid_modifier_role(modifier) != "FLOW" {
            continue;
        }
        let settings = node.properties.get("physics_fluid").ok_or_else(|| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "fluid flow modifier has no linked settings",
                json!({"node_id":node_id}),
            )
        })?;
        flows.push(fluid::FlowSource {
            triangles: fluid_mesh_triangles(context.doc, context.matrices, node, node_id)?,
            inflow_rate: flow_inflow_rate(settings)?,
        });
    }
    Ok(flows)
}

/// Emission rate in particles per frame for one flow.
///
/// A Blender flow, one whose settings carry `flow_type` or `flow_behavior`, must be a LIQUID INFLOW
/// flow from its MESH source. Blender's liquid flow fills the mesh volume, which has no
/// particle-rate equivalent, so it needs an explicit Potter-native `inflow_rate` and is refused
/// without one. Blender's `density` scales only smoke and fire, so it has no effect on liquid.
fn flow_inflow_rate(settings: &Value) -> Result<f64> {
    if settings.get("flow_type").is_some() || settings.get("flow_behavior").is_some() {
        physics_support::reject_unhonoured(
            "fluid_flow",
            settings,
            physics_support::honoured_properties("fluid_flow"),
        )?;
        match setting_string(settings, "flow_type").unwrap_or("SMOKE") {
            "LIQUID" => {}
            "SMOKE" | "FIRE" | "BOTH" => {
                return Err(unsupported_feature(
                    "physics.fluid.smoke_fire",
                    "smoke and fire fluid flows are not supported",
                ));
            }
            _ => {
                return Err(physics_support::unsupported_property(
                    "fluid_flow",
                    "flow_type",
                ));
            }
        }
        if setting_string(settings, "flow_behavior").unwrap_or("GEOMETRY") != "INFLOW" {
            return Err(unsupported_feature(
                "physics.fluid.flow_behavior",
                "only INFLOW liquid flows are supported",
            ));
        }
        if setting_string(settings, "flow_source").unwrap_or("MESH") != "MESH" {
            return Err(physics_support::unsupported_property(
                "fluid_flow",
                "flow_source",
            ));
        }
        if settings.get("inflow_rate").is_none() {
            return Err(unsupported_feature(
                "physics.fluid.emission",
                "Blender liquid flow emission has no particle-rate equivalent; set inflow_rate explicitly",
            ));
        }
    }
    nonnegative_setting(settings, "inflow_rate", 0.0, "fluid")
}

/// Liquid obstacle triangles from the fluid effector objects in `collection`.
///
/// Blender takes liquid obstacles only from fluid effectors, not from COLLISION modifiers. A
/// COLLISION effector is an obstacle, `use_effector` False skips the object, and any other effector
/// type is refused.
fn fluid_effector_triangles(
    context: &FrameContext<'_>,
    collection: Option<&BTreeSet<Id>>,
) -> Result<Vec<[DVec3; 3]>> {
    let mut triangles = Vec::new();
    for (node_id, node) in &context.doc.nodes {
        if !context.members.contains(node_id)
            || collection.is_some_and(|members| !members.contains(node_id))
        {
            continue;
        }
        let Some(modifier) = active_modifier(node_id, node, "fluid", context.gate) else {
            continue;
        };
        if fluid_modifier_role(modifier) != "EFFECTOR" {
            continue;
        }
        let settings = node.properties.get("physics_fluid").ok_or_else(|| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "fluid effector modifier has no linked settings",
                json!({"node_id":node_id}),
            )
        })?;
        physics_support::reject_unhonoured(
            "fluid_effector",
            settings,
            physics_support::honoured_properties("fluid_effector"),
        )?;
        if setting_string(settings, "effector_type").is_some_and(|kind| kind != "COLLISION") {
            return Err(physics_support::unsupported_property(
                "fluid_effector",
                "effector_type",
            ));
        }
        if !bool_setting(settings, "use_effector", true, "fluid effector")? {
            continue;
        }
        triangles.extend(fluid_mesh_triangles(
            context.doc,
            context.matrices,
            node,
            node_id,
        )?);
    }
    Ok(triangles)
}

/// Liquid particles that the fluid domain holds at the context frame. Empty without an emitter.
fn simulate_fluid_domain(
    context: &FrameContext<'_>,
    domain: &Mesh,
    matrix: DMat4,
    settings: &Value,
) -> Result<(Vec<DVec3>, bool)> {
    validate_fluid_domain(settings)?;
    let flows = fluid_flow_sources(context)?;
    if flows.is_empty() {
        return Ok((Vec::new(), false));
    }
    let window = domain_cache_window(settings)?;
    let collection = collection_filter(
        context.doc,
        settings.get("effector_group"),
        "physics.fluid_domain.effector_group",
    )?;
    let obstacles = fluid_effector_triangles(context, collection.as_ref())?;
    let seed = setting_u32(settings, "seed").unwrap_or(context.seed);
    let spawns_inside_effector = fluid::emitter_spawns_inside_effector(&flows, &obstacles);
    let positions = fluid::simulate_with_sources(
        domain,
        settings,
        matrix,
        window.solver_frame(context.frame),
        context.fps,
        context.gravity,
        seed,
        &obstacles,
        &flows,
    )?;
    Ok((positions, spawns_inside_effector))
}

#[derive(Clone, Copy)]
struct DynamicPaintBrush {
    position: DVec3,
    radius: f64,
    color: [f64; 4],
    strength: f64,
}

/// Brushes that paint a canvas: its explicit `brushes` list, or every scene brush.
///
/// A brush reads Blender's `paint_distance`, `paint_color`, and `paint_alpha`. Missing values fall
/// back to the canvas's Potter-native `radius`, `color`, and `strength`, which have no Blender
/// counterpart and are not Blender defaults.
fn dynamic_paint_brush_settings(
    context: &FrameContext<'_>,
    canvas_settings: &Value,
) -> Result<Vec<DynamicPaintBrush>> {
    let FrameContext {
        doc,
        matrices,
        members,
        gate,
        ..
    } = *context;
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
                && active_modifier(id, node, "dynamic_paint", gate).is_some()
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
        physics_support::reject_unhonoured(
            "dynamic_paint_brush",
            settings,
            physics_support::honoured_properties("dynamic_paint_brush"),
        )?;
        let matrix = matrices.get(&id).copied().ok_or_else(|| {
            PotError::with_details(
                ErrorCode::EvaluationFailed,
                "dynamic-paint brush transform is missing",
                json!({"node_id":id}),
            )
        })?;
        brushes.push(DynamicPaintBrush {
            position: matrix.transform_point3(DVec3::ZERO),
            // Blender's brush names take precedence; Potter-native keys fill in when they are absent.
            radius: setting_f64(settings, "paint_distance")
                .or_else(|| setting_f64(settings, "radius"))
                .unwrap_or(canvas_radius),
            color: color_setting(settings, "paint_color")
                .or_else(|| color_setting(settings, "color"))
                .unwrap_or(canvas_color),
            strength: setting_f64(settings, "paint_alpha")
                .or_else(|| setting_f64(settings, "strength"))
                .unwrap_or(canvas_strength),
        });
    }
    Ok(brushes)
}

/// Rejects Blender canvas surfaces that Potter does not paint.
///
/// Potter paints one vertex-colour layer, so only PAINT surfaces in VERTEX format evaluate, and
/// only on frames inside the surface's frame range. Other surface and point-cache properties go
/// through the honoured-property check.
/// Whether a canvas paints at `frame`. Blender's dynamic paint does not evaluate a surface before its
/// `frame_start` (the canvas keeps its geometry there), and holds the state of `frame_end` afterwards
/// (`dynamicPaint_frameUpdate`). A canvas paints when any of its surfaces has started.
fn canvas_paints_at(canvas: &Value, frame: f64) -> Result<bool> {
    let Some(surfaces) = canvas.get("canvas_surfaces") else {
        return Ok(true);
    };
    let surfaces = surfaces.as_array().ok_or_else(|| {
        PotError::invalid_argument("dynamic-paint canvas_surfaces must be an array")
    })?;
    if surfaces.is_empty() {
        return Ok(true);
    }
    for surface in surfaces {
        let start = setting_number(
            surface,
            "frame_start",
            1.0,
            "dynamic-paint surface",
            "a finite number",
        )?;
        if frame >= start {
            return Ok(true);
        }
    }
    Ok(false)
}

fn validate_dynamic_paint_canvas(canvas: &Value) -> Result<()> {
    let Some(surfaces) = canvas.get("canvas_surfaces") else {
        return Ok(());
    };
    let surfaces = surfaces.as_array().ok_or_else(|| {
        PotError::invalid_argument("dynamic-paint canvas_surfaces must be an array")
    })?;
    for surface in surfaces {
        physics_support::reject_unhonoured(
            "dynamic_paint_surface",
            surface,
            physics_support::honoured_properties("dynamic_paint_surface"),
        )?;
        if setting_string(surface, "surface_type").is_some_and(|kind| kind != "PAINT") {
            return Err(physics_support::unsupported_property(
                "dynamic_paint_surface",
                "surface_type",
            ));
        }
        if setting_string(surface, "surface_format").is_some_and(|format| format != "VERTEX") {
            return Err(physics_support::unsupported_property(
                "dynamic_paint_surface",
                "surface_format",
            ));
        }
        point_cache_window(surface)?;
    }
    Ok(())
}

/// Per-vertex RGBA colours that the canvas's dynamic-paint settings produce at the context frame.
fn paint_canvas(
    context: &FrameContext<'_>,
    canvas: &Mesh,
    matrix: DMat4,
    settings: &Value,
) -> Result<Vec<[f64; 4]>> {
    validate_dynamic_paint_canvas(settings)?;
    if !canvas_paints_at(settings, context.frame)? {
        return Ok(vec![[0.0; 4]; canvas.vertices.len()]);
    }
    let brushes = dynamic_paint_brush_settings(context, settings)?;
    if brushes.is_empty() {
        return dynamic_paint::paint(
            canvas,
            matrix,
            &[],
            setting_f64(settings, "radius").unwrap_or(1.0),
            setting_color(settings).unwrap_or([1.0, 0.0, 0.0, 1.0]),
            setting_f64(settings, "strength").unwrap_or(1.0),
        );
    }
    let mut colors = vec![[0.0; 4]; canvas.vertices.len()];
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
    Ok(colors)
}
