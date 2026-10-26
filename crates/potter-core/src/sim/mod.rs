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

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    sync::Arc,
};

use glam::{DMat4, DQuat, DVec3, Mat4, Vec3};
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
const PARTICLE_STATE_VERSION: &str = "particle-state-7";

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
    /// Dynamic-paint wetness in mesh vertex storage order when the secondary output is enabled.
    #[serde(default)]
    pub paint_wetness: BTreeMap<Id, Vec<f64>>,
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
    /// Dynamic-paint wetness by canvas object when its secondary output is enabled.
    pub paint_wetness: BTreeMap<Id, Vec<f64>>,
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
    /// Exact emitter-local source point when available for this particle.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub birth_position_local: Option<[f64; 3]>,
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
            birth_position_local: None,
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
fn particle_state(
    particle: particles::Particle,
    system_index: u32,
    frame: f64,
    use_rotations: bool,
    no_physics: bool,
) -> ParticleState {
    let at_birth = frame
        .partial_cmp(&particle.birth_frame)
        .is_some_and(std::cmp::Ordering::is_eq);
    let rotation = if !use_rotations {
        if particle.path.is_empty() && (at_birth || no_physics) {
            particle.birth_rotation
        } else {
            DQuat::IDENTITY.to_array()
        }
    } else if particle.path.is_empty() {
        particles::blender_velocity_orientation(particle.velocity)
    } else {
        // Hair path keys carry Bishop frames, not the current Hair particle orientation.
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
        birth_position_local: particle.birth_position_local,
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
            .chain(state.birth_position_local.iter().flatten())
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
    collection_members(doc, &scene.root_collection)
}

/// Evaluates one physics modifier against the geometry produced by earlier stack entries.
pub(crate) fn evaluate_stack_modifier(
    doc: &SceneDoc,
    scene_id: &Id,
    view_layer: Option<&Id>,
    node_id: &Id,
    modifier: &Modifier,
    input_mesh: Mesh,
    frame: f64,
    matrices: &BTreeMap<Id, DMat4>,
    scene_hash: &str,
    dynamic_paint_holds: &mut DynamicPaintHoldCache,
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
    let matrix = matrices.get(node_id).copied().ok_or_else(|| {
        PotError::with_details(
            ErrorCode::EvaluationFailed,
            "physics modifier object transform is missing",
            json!({"node_id":node_id}),
        )
    })?;
    let members = scene_members(doc, scene_id)?;
    let context = FrameContext::new(doc, scene_id, view_layer, frame, matrices, &members, gate)?;
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
            let use_rotations = settings
                .get("use_rotations")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let no_physics = settings.get("physics_type").and_then(Value::as_str) == Some("NO");
            let seed = setting_u32(settings, "seed").unwrap_or(context.seed);
            let triangles =
                collider_triangles(doc, matrices, &members, gate, ColliderConsumer::Point)?;
            let colliders = point_colliders(&triangles, node_id);
            let fields = force_field_inputs(doc, matrices, &members)?;
            let textures =
                load_particle_textures(&solver_settings, doc, gate.context.project_root)?;
            let slots = particle_textures::texture_slots(&solver_settings, &textures, matrices)?;
            let states = particles::simulate_lifecycle_textured(
                &mesh,
                &solver_settings,
                matrix,
                frame,
                context.gravity,
                seed,
                &fields,
                &colliders,
                &slots,
            )?
            .into_iter()
            .map(|particle| {
                particle_state(particle, system_index, frame, use_rotations, no_physics)
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
            let modifier_prefix = node
                .modifiers
                .iter()
                .position(|candidate| candidate.id == modifier.id)
                .ok_or_else(|| {
                    PotError::with_details(
                        ErrorCode::InternalError,
                        "dynamic-paint modifier is missing from its owner stack",
                        json!({"node_id":node_id,"modifier_id":modifier.id}),
                    )
                })?;
            let output = paint_canvas(
                &context,
                node_id,
                &mesh,
                settings,
                DynamicPaintEvaluation {
                    scene_hash,
                    modifier_prefix,
                    stack_mesh_is_current: true,
                    hold_cache: dynamic_paint_holds,
                },
            )?;
            match &output.values {
                DynamicPaintValues::Weights(weights) => {
                    let groups = node
                        .data
                        .as_ref()
                        .and_then(|data_id| context.doc.data_blocks.get(data_id))
                        .map_or(&[][..], |data| data.vertex_groups.as_slice());
                    if let Some(name) = output.output_name.as_deref()
                        && groups.iter().any(|group| group.name == name)
                    {
                        write_dynamic_paint_weight_group(&mut mesh, name, weights)?;
                    }
                }
                DynamicPaintValues::Colors(colors)
                    if setting_string(settings, "surface_format") == Some("weight") =>
                {
                    let color_values = mesh
                        .vertices
                        .iter()
                        .zip(colors.iter())
                        .map(|(vertex, color)| (format!("v{}", vertex.id), json!(color)))
                        .collect::<serde_json::Map<_, _>>();
                    mesh.attributes.insert(
                        "color".to_owned(),
                        json!({"domain":"vertices","values":color_values}),
                    );
                    let weight_values = mesh
                        .vertices
                        .iter()
                        .zip(colors.iter())
                        .map(|(vertex, color)| (format!("v{}", vertex.id), json!(color[3])))
                        .collect::<serde_json::Map<_, _>>();
                    mesh.attributes.insert(
                        "weight".to_owned(),
                        json!({"domain":"vertices","type":"float","values":weight_values}),
                    );
                }
                DynamicPaintValues::Colors(colors) => {
                    let domain = if settings
                        .get("canvas_surfaces")
                        .and_then(Value::as_array)
                        .and_then(|surfaces| surfaces.first())
                        .is_some()
                    {
                        crate::geom::color_layers::ColorDomain::Corner
                    } else {
                        crate::geom::color_layers::ColorDomain::Point
                    };
                    if let Some(name) = output.output_name.as_deref() {
                        write_dynamic_paint_color_layer(
                            &mut mesh,
                            name,
                            colors.as_slice(),
                            domain,
                            true,
                        )?;
                    }
                    if let Some((name, wetness)) = &output.wetness_output {
                        let wetness_colors = dynamic_paint_wetness_colors(wetness);
                        write_dynamic_paint_color_layer(
                            &mut mesh,
                            name,
                            &wetness_colors,
                            domain,
                            false,
                        )?;
                    }
                }
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
    view_layer: Option<&Id>,
    frame: f64,
    base_world_matrices: &BTreeMap<Id, DMat4>,
    scene_hash: &str,
    dynamic_paint_holds: &mut DynamicPaintHoldCache,
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
    let physics = evaluate_physics(
        doc,
        scene_id,
        view_layer,
        frame,
        base_world_matrices,
        &members,
        scene_hash,
        dynamic_paint_holds,
        gate,
    )?;
    let mut result = result.unwrap_or_else(|| empty_result(key.clone()));
    result.cache_key.clone_from(&key);
    result.deformed_vertices = physics.deformed_vertices;
    result.fluid_particles = physics.fluid_particles;
    result.fluid_warnings = physics.fluid_warnings;
    result.fluid_surfaces = physics.fluid_surfaces;
    result.paint_colors = physics.paint_colors;
    result.paint_wetness = physics.paint_wetness;
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
        paint_wetness: BTreeMap::new(),
        paint_weights: BTreeMap::new(),
        physics_cache_key: None,
        fluid_warnings: Vec::new(),
    }
}

fn evaluate_physics(
    doc: &SceneDoc,
    scene_id: &Id,
    view_layer: Option<&Id>,
    frame: f64,
    matrices: &BTreeMap<Id, DMat4>,
    members: &BTreeSet<Id>,
    scene_hash: &str,
    dynamic_paint_holds: &mut DynamicPaintHoldCache,
    gate: ModifierGate<'_>,
) -> Result<PhysicsResult> {
    let context = FrameContext::new(doc, scene_id, view_layer, frame, matrices, members, gate)?;
    let force_fields = force_field_inputs(doc, matrices, members)?;
    let mut result = PhysicsResult {
        fluid_warnings: Vec::new(),
        deformed_vertices: BTreeMap::new(),
        particles: BTreeMap::new(),
        fluid_particles: BTreeMap::new(),
        fluid_surfaces: BTreeMap::new(),
        paint_colors: BTreeMap::new(),
        paint_wetness: BTreeMap::new(),
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
            let textures = load_particle_textures(settings, doc, gate.context.project_root)?;
            let slots = particle_textures::texture_slots(settings, &textures, matrices)?;
            let particles = particles::simulate_textured(
                mesh,
                settings,
                matrix,
                frame,
                context.gravity,
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
            let use_rotations = settings
                .get("use_rotations")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let no_physics = settings.get("physics_type").and_then(Value::as_str) == Some("NO");
            let states = particles
                .into_iter()
                .map(|particle| {
                    particle_state(particle, system_index, frame, use_rotations, no_physics)
                })
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
            && let Some(modifier) = active_modifier(id, node, "dynamic_paint", gate)
        {
            let modifier_prefix = node
                .modifiers
                .iter()
                .position(|candidate| candidate.id == modifier.id)
                .ok_or_else(|| {
                    PotError::with_details(
                        ErrorCode::InternalError,
                        "dynamic-paint modifier is missing from its owner stack",
                        json!({"node_id":id,"modifier_id":modifier.id}),
                    )
                })?;
            let canvas = mesh_for_node(doc, node, id)?;
            let output = paint_canvas(
                &context,
                id,
                canvas,
                settings,
                DynamicPaintEvaluation {
                    scene_hash,
                    modifier_prefix,
                    stack_mesh_is_current: false,
                    hold_cache: dynamic_paint_holds,
                },
            )?;
            match &output.values {
                DynamicPaintValues::Weights(weights) => {
                    if weights.iter().any(|weight| !weight.is_finite()) {
                        return Err(PotError::new(
                            ErrorCode::EvaluationFailed,
                            format!("dynamic-paint weights for `{id}` are invalid"),
                        ));
                    }
                    result
                        .paint_weights
                        .insert(id.clone(), weights.as_ref().clone());
                }
                DynamicPaintValues::Colors(colors) => {
                    if colors.iter().flatten().any(|value| !value.is_finite()) {
                        return Err(PotError::new(
                            ErrorCode::EvaluationFailed,
                            format!("dynamic-paint colors for `{id}` are invalid"),
                        ));
                    }
                    if let Some((_, wetness)) = &output.wetness_output {
                        if wetness.iter().any(|value| !value.is_finite()) {
                            return Err(PotError::new(
                                ErrorCode::EvaluationFailed,
                                format!("dynamic-paint wetness for `{id}` is invalid"),
                            ));
                        }
                        result
                            .paint_wetness
                            .insert(id.clone(), wetness.as_ref().clone());
                    }
                    match setting_string(settings, "surface_format").unwrap_or("color") {
                        "color" if output.output_name.is_some() => {
                            result
                                .paint_colors
                                .insert(id.clone(), colors.as_ref().clone());
                        }
                        "color" => {}
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
/// Objects of a collection and child collections in stored membership order.
fn collection_members_ordered(doc: &SceneDoc, root: &Id) -> Result<Vec<Id>> {
    let mut pending = vec![root.clone()];
    let mut visited_collections = BTreeSet::new();
    let mut visited_objects = BTreeSet::new();
    let mut members = Vec::new();
    while let Some(collection_id) = pending.pop() {
        if !visited_collections.insert(collection_id.clone()) {
            continue;
        }
        let collection = doc.collections.get(&collection_id).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "collection is missing",
                json!({"collection_id":collection_id}),
            )
        })?;
        for object_id in &collection.objects {
            if visited_objects.insert(object_id.clone()) {
                members.push(object_id.clone());
            }
        }
        pending.extend(collection.children.iter().rev().cloned());
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

fn effective_scene_gravity(scene: &crate::model::Scene) -> DVec3 {
    let gravity = scene
        .rigid_body_world
        .as_ref()
        .map_or(DVec3::new(0.0, 0.0, -9.81), |world| {
            DVec3::from_array(world.gravity)
        });
    if scene.use_gravity {
        gravity
    } else {
        DVec3::ZERO
    }
}

/// Scene inputs that every physics evaluation at one frame shares.
#[derive(Clone, Copy)]
struct FrameContext<'a> {
    doc: &'a SceneDoc,
    scene_id: &'a Id,
    view_layer: Option<&'a Id>,
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
        scene_id: &'a Id,
        view_layer: Option<&'a Id>,
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
        let gravity = effective_scene_gravity(scene);
        let seed = scene
            .rigid_body_world
            .as_ref()
            .map_or(0, |world| world.seed);
        Ok(Self {
            doc,
            scene_id,
            view_layer,
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
) -> Result<Vec<Vec<[DVec3; 3]>>> {
    let mut effectors = Vec::new();
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
        effectors.push(fluid_mesh_triangles(
            context.doc,
            context.matrices,
            node,
            node_id,
        )?);
    }
    Ok(effectors)
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
    let effectors = fluid_effector_triangles(context, collection.as_ref())?;
    let obstacles = effectors.iter().flatten().copied().collect::<Vec<_>>();
    let seed = setting_u32(settings, "seed").unwrap_or(context.seed);
    let spawns_inside_effector = fluid::emitter_spawns_inside_effector(&flows, &effectors);
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

#[derive(Debug, Clone)]
pub(crate) struct DynamicPaintBrush {
    /// Brush object whose prefix mesh supplies VOLUME coverage.
    pub(crate) node_id: Id,
    pub(crate) position: DVec3,
    pub(crate) radius: f64,
    pub(crate) color: [f64; 4],
    pub(crate) strength: f64,
    pub(crate) wetness: f64,
    pub(crate) absolute_alpha: bool,
    pub(crate) erase: bool,
    pub(crate) constant_falloff: bool,
    /// Per-canvas-vertex inside mask; `None` selects POINT proximity.
    pub(crate) volume_mask: Option<Vec<bool>>,
}
#[derive(Debug, Default)]
pub(crate) struct DynamicPaintHoldCache {
    entries: BTreeMap<String, DynamicPaintHold>,
    weight_entries: BTreeMap<String, DynamicPaintWeightState>,
    paint_entries: BTreeMap<String, DynamicPaintPaintState>,
}
#[derive(Debug)]
struct DynamicPaintWeightState {
    frame: i64,
    vertex_ids: Vec<u32>,
    weights: Arc<Vec<f64>>,
}
#[derive(Debug)]
struct DynamicPaintPaintState {
    frame: i64,
    vertex_ids: Vec<u32>,
    points: Arc<Vec<dynamic_paint::PaintPoint>>,
}
#[derive(Debug)]
struct DynamicPaintHold {
    vertex_ids: Vec<u32>,
    output: DynamicPaintOutput,
}
#[derive(Debug, Clone)]
enum DynamicPaintValues {
    Colors(Arc<Vec<[f64; 4]>>),
    Weights(Arc<Vec<f64>>),
}
#[derive(Debug, Clone)]
struct DynamicPaintOutput {
    output_name: Option<String>,
    values: DynamicPaintValues,
    wetness_output: Option<(String, Arc<Vec<f64>>)>,
}

/// Brushes that paint a canvas: its explicit `brushes` list, or every scene brush.
///
/// POINT brushes read Blender's `paint_distance`, `paint_color`, and `paint_alpha`; missing values
/// fall back to Potter-native canvas settings. VOLUME uses full inside influence before brush alpha,
/// surface influence, and wetness are applied; point distance and falloff are ignored.
fn dynamic_paint_brush_settings(
    context: &FrameContext<'_>,
    canvas_settings: &Value,
    weight_surface: bool,
) -> Result<Vec<DynamicPaintBrush>> {
    let FrameContext {
        doc,
        scene_id,
        frame,
        members,
        gate,
        ..
    } = *context;
    let surface = canvas_settings
        .get("canvas_surfaces")
        .and_then(Value::as_array)
        .and_then(|surfaces| surfaces.first());
    let brush_filter = match surface {
        Some(surface) => collection_filter(
            doc,
            surface.get("brush_collection"),
            "physics.dynamic_paint_surface.brush_collection",
        )?,
        None => None,
    };
    let mut brush_ids = Vec::new();
    let mut seen_brush_ids = BTreeSet::new();
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
            if brush_filter
                .as_ref()
                .is_some_and(|filter| !filter.contains(&id))
            {
                continue;
            }
            if seen_brush_ids.insert(id.clone()) {
                brush_ids.push(id);
            }
        }
    } else {
        let scene = doc.scenes.get(scene_id).ok_or_else(|| {
            PotError::new(
                ErrorCode::SceneNotFound,
                format!("scene `{scene_id}` was not found"),
            )
        })?;
        let root = match surface
            .and_then(|surface| surface.get("brush_collection"))
            .filter(|value| !value.is_null())
        {
            Some(value) => {
                let value = value.as_str().ok_or_else(|| {
                    PotError::invalid_argument(
                        "dynamic-paint brush_collection must be a collection ID",
                    )
                })?;
                Id::new(value.to_owned())
                    .map_err(|error| PotError::new(ErrorCode::InvalidOperation, error.message))?
            }
            None => scene.root_collection.clone(),
        };
        for id in collection_members_ordered(doc, &root)? {
            if !members.contains(&id) {
                continue;
            }
            let Some(node) = doc.nodes.get(&id) else {
                continue;
            };
            if gate.is_property_active(&id, node, "physics_dynamic_paint", "dynamic_paint")
                && node
                    .properties
                    .get("physics_dynamic_paint")
                    .and_then(|settings| setting_string(settings, "role"))
                    == Some("brush")
                && active_modifier(&id, node, "dynamic_paint", gate).is_some()
                && brush_filter
                    .as_ref()
                    .is_none_or(|filter| filter.contains(&id))
                && seen_brush_ids.insert(id.clone())
            {
                brush_ids.push(id);
            }
        }
    }
    let has_point_source = brush_ids.iter().any(|id| {
        doc.nodes
            .get(id)
            .and_then(|node| node.properties.get("physics_dynamic_paint"))
            .and_then(|settings| setting_string(settings, "paint_source"))
            == Some("POINT")
    });
    let driver_values = has_point_source
        .then(|| crate::eval::rig::evaluate_drivers(doc, frame))
        .transpose()?;
    let radius_scale = if has_point_source {
        surface.map_or(Ok(1.0), |surface| {
            setting_number(
                surface,
                "brush_radius_scale",
                1.0,
                "dynamic-paint surface",
                "a finite nonnegative number",
            )
        })?
    } else {
        1.0
    };
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
        let paint_source = setting_string(settings, "paint_source").unwrap_or("VOLUME");
        if !matches!(paint_source, "POINT" | "VOLUME") {
            return Err(physics_support::unsupported_property(
                "dynamic_paint_brush",
                "paint_source",
            ));
        }
        if paint_source == "VOLUME" && settings.get("radius").is_some() {
            return Err(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                "native radius is only supported for POINT Dynamic Paint brushes",
                json!({"feature_id":"physics.dynamic_paint_brush.radius","property":"radius","status":"not_supported"}),
            ));
        }
        let constant_falloff = if paint_source == "POINT" {
            match setting_string(settings, "proximity_falloff").unwrap_or("SMOOTH") {
                "SMOOTH" => false,
                "CONSTANT" => true,
                _ => {
                    return Err(physics_support::unsupported_property(
                        "dynamic_paint_brush",
                        "proximity_falloff",
                    ));
                }
            }
        } else {
            false
        };

        if weight_surface {
            physics_support::reject_unhonoured(
                "dynamic_paint_brush",
                settings,
                physics_support::DYNAMIC_PAINT_WEIGHT_BRUSH_PROPERTIES,
            )?;
        } else {
            const PAINT_BRUSH_PROPERTIES: &[&str] = &[
                "paint_color",
                "paint_alpha",
                "paint_distance",
                "paint_source",
                "paint_wetness",
                "use_absolute_alpha",
                "proximity_falloff",
            ];
            physics_support::reject_unhonoured(
                "dynamic_paint_brush",
                settings,
                PAINT_BRUSH_PROPERTIES,
            )?;
        }
        let position = if paint_source == "POINT" {
            // Blender's POINT source uses Object.loc, ignoring parenting, constraints and delta transforms.
            let animated = crate::eval::animation::animated_transform(node, doc, frame)?;
            let driver_values = driver_values.as_ref().ok_or_else(|| {
                PotError::new(
                    ErrorCode::InternalError,
                    "dynamic-paint point-brush drivers are missing",
                )
            })?;
            let transform = crate::eval::rig::apply_transform_drivers(
                &id,
                &node.transform.rotation_mode,
                animated,
                driver_values,
            )?;
            DVec3::new(
                transform.translation[0],
                transform.translation[1],
                transform.translation[2],
            )
        } else {
            DVec3::ZERO
        };
        brushes.push(DynamicPaintBrush {
            node_id: id,
            volume_mask: (paint_source == "VOLUME").then(Vec::new),
            position,
            // Blender's brush names take precedence; Potter-native keys fill in when they are absent.
            radius: if paint_source == "POINT" {
                setting_f64(settings, "paint_distance")
                    .or_else(|| setting_f64(settings, "radius"))
                    .unwrap_or(canvas_radius)
                    * radius_scale
            } else {
                0.0
            },
            color: color_setting(settings, "paint_color")
                .or_else(|| color_setting(settings, "color"))
                .unwrap_or(canvas_color),
            strength: setting_f64(settings, "paint_alpha")
                .or_else(|| setting_f64(settings, "strength"))
                .unwrap_or(canvas_strength),
            wetness: setting_f64(settings, "paint_wetness").unwrap_or(1.0),
            absolute_alpha: bool_setting(
                settings,
                "use_absolute_alpha",
                true,
                "dynamic-paint brush",
            )?,
            erase: bool_setting(settings, "use_paint_erase", false, "dynamic-paint brush")?,
            constant_falloff,
        });
    }
    Ok(brushes)
}

fn surface_paint_frame(surface: &Value, frame: f64) -> Result<Option<f64>> {
    let start = setting_number(
        surface,
        "frame_start",
        1.0,
        "dynamic-paint surface",
        "a finite number",
    )?;
    let end = setting_number(
        surface,
        "frame_end",
        250.0,
        "dynamic-paint surface",
        "a finite number",
    )?;
    if start > end || frame < start {
        Ok(None)
    } else {
        Ok(Some(frame.min(end)))
    }
}

fn validate_dynamic_paint_point_cache(surface: &Value) -> Result<()> {
    let Some(cache) = surface.get("point_cache") else {
        return Ok(());
    };
    if !cache.is_object() {
        return Err(PotError::invalid_argument(
            "dynamic-paint point_cache must be an object",
        ));
    }
    physics_support::reject_unhonoured(
        "point_cache",
        cache,
        physics_support::honoured_properties("point_cache"),
    )
}

/// World matrices at a held paint frame, using the same resources and scene timing as its outer evaluation.
fn scene_world_matrices_at(context: &FrameContext<'_>, frame: f64) -> Result<BTreeMap<Id, DMat4>> {
    let drivers = crate::eval::rig::evaluate_drivers(context.doc, frame)?;
    let scene = context.doc.scenes.get(context.scene_id).ok_or_else(|| {
        PotError::new(
            ErrorCode::SceneNotFound,
            format!("scene `{}` was not found", context.scene_id),
        )
    })?;
    let mut session = crate::eval::EvaluationSession::default();
    crate::eval::rig::evaluate_world_matrices_with_resources(
        context.doc,
        frame,
        &drivers,
        context.gate.context.project_root,
        scene.fps,
        scene.fps_base,
        &mut session,
    )
}

const DYNAMIC_PAINT_WEIGHT_SURFACE_PROPERTIES: &[&str] = &[
    "surface_type",
    "surface_format",
    "frame_start",
    "frame_end",
    "output_name_a",
    "use_output_a",
    "brush_radius_scale",
    "brush_influence_scale",
    "brush_collection",
    "use_dissolve",
    "dissolve_speed",
    "use_dissolve_log",
    "use_drying",
    "dry_speed",
    "use_dry_log",
    "init_color",
    "init_color_type",
    "init_layername",
    "init_texture",
    "output_name_b",
];

fn validate_dynamic_paint_canvas(canvas: &Value) -> Result<()> {
    let Some(surfaces) = canvas.get("canvas_surfaces") else {
        return Ok(());
    };
    let surfaces = surfaces.as_array().ok_or_else(|| {
        PotError::invalid_argument("dynamic-paint canvas_surfaces must be an array")
    })?;
    if surfaces.len() > 1 {
        return Err(unsupported_feature(
            "physics.dynamic_paint_canvas.canvas_surfaces",
            "multiple Blender dynamic-paint surfaces have independent frame caches and output layers; Potter can evaluate only one surface per canvas",
        ));
    }
    for surface in surfaces {
        let surface_type = setting_string(surface, "surface_type").unwrap_or("PAINT");
        if !matches!(surface_type, "PAINT" | "WEIGHT") {
            return Err(physics_support::unsupported_property(
                "dynamic_paint_surface",
                "surface_type",
            ));
        }
        if surface_type == "WEIGHT" {
            physics_support::reject_unhonoured(
                "dynamic_paint_surface",
                surface,
                DYNAMIC_PAINT_WEIGHT_SURFACE_PROPERTIES,
            )?;
            let dissolve_speed = setting_number(
                surface,
                "dissolve_speed",
                250.0,
                "dynamic-paint surface",
                "a positive finite number",
            )?;
            if dissolve_speed <= 0.0 {
                return Err(PotError::invalid_argument(
                    "dynamic-paint surface dissolve_speed must be positive",
                ));
            }
        } else {
            physics_support::reject_unhonoured(
                "dynamic_paint_surface",
                surface,
                physics_support::honoured_properties("dynamic_paint_surface"),
            )?;
            if bool_setting(surface, "use_drying", true, "dynamic-paint surface")? {
                let dry_speed = setting_number(
                    surface,
                    "dry_speed",
                    500.0,
                    "dynamic-paint surface",
                    "a positive finite number",
                )?;
                if dry_speed <= 0.0 {
                    return Err(PotError::invalid_argument(
                        "dynamic-paint surface dry_speed must be positive",
                    ));
                }
                setting_number(
                    surface,
                    "color_dry_threshold",
                    1.0,
                    "dynamic-paint surface",
                    "a finite number",
                )?;
            }
            if bool_setting(surface, "use_dissolve", false, "dynamic-paint surface")? {
                let dissolve_speed = setting_number(
                    surface,
                    "dissolve_speed",
                    250.0,
                    "dynamic-paint surface",
                    "a positive finite number",
                )?;
                if dissolve_speed <= 0.0 {
                    return Err(PotError::invalid_argument(
                        "dynamic-paint surface dissolve_speed must be positive",
                    ));
                }
            }
        }
        if setting_string(surface, "surface_format").is_some_and(|format| format != "VERTEX") {
            return Err(physics_support::unsupported_property(
                "dynamic_paint_surface",
                "surface_format",
            ));
        }
        validate_dynamic_paint_point_cache(surface)?;
    }
    Ok(())
}

/// Ensures primary output A and, for operations, requested output B. Load migration preserves an
/// absent B layer because Blender's `use_output_b` flag alone does not enable its output.
pub(crate) fn ensure_dynamic_paint_output_layers(
    mesh: &mut Mesh,
    settings: &Value,
    create_requested_output_b: bool,
) -> Result<bool> {
    if setting_string(settings, "role") != Some("canvas") {
        return Ok(false);
    }
    let Some(surfaces) = settings.get("canvas_surfaces").and_then(Value::as_array) else {
        return Ok(false);
    };
    let mut changed = false;
    for surface in surfaces {
        if setting_string(surface, "surface_type").unwrap_or("PAINT") != "PAINT"
            || setting_string(surface, "surface_format").unwrap_or("VERTEX") != "VERTEX"
        {
            continue;
        }
        let output_a = setting_string(surface, "output_name_a").unwrap_or("dp_paintmap");
        if !output_a.is_empty() {
            changed |= crate::geom::color_layers::ensure_byte_color_corner_layer(mesh, output_a)?;
        }
        if create_requested_output_b
            && bool_setting(surface, "use_output_b", false, "dynamic-paint surface")?
        {
            let output_b = setting_string(surface, "output_name_b").unwrap_or("dp_wetmap");
            if !output_b.is_empty() {
                changed |=
                    crate::geom::color_layers::ensure_byte_color_corner_layer(mesh, output_b)?;
            }
        }
    }
    Ok(changed)
}

pub(crate) fn migrate_dynamic_paint_output_layers(doc: &mut SceneDoc) -> Result<bool> {
    let blender_owned_data: BTreeSet<Id> = doc
        .nodes
        .values()
        .filter_map(|node| {
            let settings = node.properties.get("physics_dynamic_paint")?;
            if setting_string(settings, "role") != Some("canvas") {
                return None;
            }
            let data_id = node.data.as_ref()?;
            doc.data_blocks
                .get(data_id)
                .and_then(|data| data.mesh.as_ref())
                .filter(|mesh| mesh.attributes.contains_key("blender_metadata"))
                .map(|_| data_id.clone())
        })
        .collect();
    let (nodes, data_blocks) = (&doc.nodes, &mut doc.data_blocks);
    let mut changed = false;
    for node in nodes.values() {
        let Some(settings) = node.properties.get("physics_dynamic_paint") else {
            continue;
        };
        if setting_string(settings, "role") != Some("canvas") {
            continue;
        }
        let Some(data_id) = node.data.as_ref() else {
            continue;
        };
        if blender_owned_data.contains(data_id) {
            continue;
        }
        let Some(mesh) = data_blocks
            .get_mut(data_id)
            .and_then(|data| data.mesh.as_mut())
        else {
            continue;
        };
        // Migrate primary output A only; B still requires its own layer toggle.
        changed |= ensure_dynamic_paint_output_layers(mesh, settings, false)?;
    }
    Ok(changed)
}
fn delta_transform_is_nondefault(
    metadata: &serde_json::Map<String, Value>,
    property: &str,
    defaults: &[f64],
) -> bool {
    let Some(value) = metadata.get(property).filter(|value| !value.is_null()) else {
        return false;
    };
    let Some(values) = value.as_array() else {
        return true;
    };
    values.len() != defaults.len()
        || values
            .iter()
            .zip(defaults)
            .any(|(value, default)| value.as_f64() != Some(*default))
}

fn has_legacy_brush_delta_transform(node: &Node) -> bool {
    let Some(metadata) = node
        .properties
        .get("blender_object_metadata")
        .and_then(Value::as_object)
    else {
        return false;
    };
    delta_transform_is_nondefault(metadata, "delta_location", &[0.0, 0.0, 0.0])
        || delta_transform_is_nondefault(metadata, "delta_rotation_euler", &[0.0, 0.0, 0.0])
        || delta_transform_is_nondefault(
            metadata,
            "delta_rotation_quaternion",
            &[1.0, 0.0, 0.0, 0.0],
        )
        || delta_transform_is_nondefault(metadata, "delta_scale", &[1.0, 1.0, 1.0])
}

pub(crate) fn migrate_dynamic_paint_brush_source(doc: &mut SceneDoc) -> Result<bool> {
    let mut changed = false;
    for (node_id, node) in &mut doc.nodes {
        let legacy_brush_without_source =
            node.properties
                .get("physics_dynamic_paint")
                .is_some_and(|settings| {
                    setting_string(settings, "role") == Some("brush")
                        && settings.get("paint_source").is_none()
                });
        if !legacy_brush_without_source {
            continue;
        }
        // The native brush solver used WORLD origin; POINT is equivalent only without these transforms.
        if node.parent.is_some()
            || !node.constraints.is_empty()
            || has_legacy_brush_delta_transform(node)
        {
            return Err(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                "legacy Dynamic Paint brush transforms cannot be migrated to POINT without changing its world-space origin",
                json!({"feature_id":"physics.dynamic_paint_brush.legacy_transform","node_id":node_id,"status":"not_supported"}),
            ));
        }
        if node
            .properties
            .get("blender_object_metadata")
            .is_some_and(Value::is_object)
        {
            continue;
        }
        let Some(settings) = node
            .properties
            .get_mut("physics_dynamic_paint")
            .and_then(Value::as_object_mut)
        else {
            continue;
        };
        let _ = settings.insert("paint_source".to_owned(), json!("POINT"));
        changed = true;
    }
    Ok(changed)
}

fn dynamic_paint_output_name(settings: &Value) -> Option<String> {
    let surface = settings
        .get("canvas_surfaces")
        .and_then(Value::as_array)
        .and_then(|surfaces| surfaces.first());
    let name = match surface {
        Some(surface) => setting_string(surface, "output_name_a").unwrap_or_else(|| {
            if setting_string(surface, "surface_type") == Some("WEIGHT") {
                "dp_weight"
            } else {
                "dp_paintmap"
            }
        }),
        None if setting_string(settings, "surface_format") == Some("weight") => "weight",
        None => "color",
    };
    (!name.is_empty()).then(|| name.to_owned())
}
fn dynamic_paint_output_layer_exists(
    context: &FrameContext<'_>,
    canvas_id: &Id,
    canvas: &Mesh,
    name: &str,
) -> bool {
    crate::geom::color_layers::has_blender_byte_color_corner_layer(canvas, name)
        || context
            .doc
            .nodes
            .get(canvas_id)
            .and_then(|node| node.data.as_ref())
            .and_then(|data_id| context.doc.data_blocks.get(data_id))
            .and_then(|data| data.mesh.as_ref())
            .is_some_and(|mesh| {
                crate::geom::color_layers::has_blender_byte_color_corner_layer(mesh, name)
            })
}

fn dynamic_paint_mesh_snapshot(
    context: &FrameContext<'_>,
    node_id: &Id,
    frame: f64,
    modifier_prefix: usize,
) -> Result<crate::eval::Snapshot> {
    let driver_values = crate::eval::rig::evaluate_drivers(context.doc, frame)?;
    let nodes = BTreeSet::from([node_id.clone()]);
    crate::eval::Snapshot::evaluate_nodes_with_modifier_prefix_with_cache(
        context.doc,
        &crate::eval::EvaluationContext {
            scene_id: Some(context.scene_id.clone()),
            view_layer: context.view_layer.cloned(),
            frame: Some(frame),
            purpose: context.gate.context.purpose,
            ..crate::eval::EvaluationContext::default()
        },
        context.gate.context.project_root,
        context.gate.context.cache_directory,
        &nodes,
        node_id,
        modifier_prefix,
        true,
        &driver_values,
    )
}
fn prepare_dynamic_paint_volume_brushes(
    context: &FrameContext<'_>,
    canvas: &Mesh,
    canvas_world: DMat4,
    brushes: &mut [DynamicPaintBrush],
) -> Result<()> {
    if !brushes.iter().any(|brush| brush.volume_mask.is_some()) {
        return Ok(());
    }
    let canvas_positions: Vec<DVec3> = canvas.vertices.iter().map(|vertex| vertex.co).collect();
    let canvas_normals = crate::geom::normals::geometric_vertex_normals(canvas, &canvas_positions)?;
    let canvas_matrix =
        Mat4::from_cols_array(&canvas_world.to_cols_array().map(|value| value as f32));

    for brush in brushes
        .iter_mut()
        .filter(|brush| brush.volume_mask.is_some())
    {
        let node = context.doc.nodes.get(&brush.node_id).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::TargetNotFound,
                "dynamic-paint brush Object was not found",
                json!({"brush_id":brush.node_id}),
            )
        })?;
        let modifier_prefix = node
            .modifiers
            .iter()
            .position(|modifier| modifier.modifier_type == "dynamic_paint")
            .ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::EvaluationFailed,
                    "dynamic-paint brush modifier is missing",
                    json!({"brush_id":brush.node_id}),
                )
            })?;
        let mut snapshot =
            dynamic_paint_mesh_snapshot(context, &brush.node_id, context.frame, modifier_prefix)?;
        let brush_mesh = snapshot.meshes.remove(&brush.node_id).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::EvaluationFailed,
                "dynamic-paint brush mesh is missing at its paint frame",
                json!({"brush_id":brush.node_id,"frame":context.frame}),
            )
        })?;
        let brush_world = context
            .matrices
            .get(&brush.node_id)
            .copied()
            .ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::EvaluationFailed,
                    "dynamic-paint brush transform is missing at its paint frame",
                    json!({"brush_id":brush.node_id,"frame":context.frame}),
                )
            })?;
        if !brush_world.is_finite() || !canvas_world.is_finite() {
            return Err(PotError::new(
                ErrorCode::EvaluationFailed,
                "dynamic-paint volume brush transforms must be finite",
            ));
        }
        let brush_matrix =
            Mat4::from_cols_array(&brush_world.to_cols_array().map(|value| value as f32));
        let world_positions: Vec<[f32; 3]> = brush_mesh
            .vertices
            .iter()
            .map(|vertex| {
                brush_matrix
                    .transform_point3(vertex.co.as_vec3())
                    .to_array()
            })
            .collect();
        if world_positions
            .iter()
            .flatten()
            .any(|coordinate| !coordinate.is_finite())
        {
            return Err(PotError::new(
                ErrorCode::EvaluationFailed,
                "dynamic-paint volume brush vertices must be finite",
            ));
        }
        let vertex_indices: HashMap<u32, usize> = brush_mesh
            .vertices
            .iter()
            .enumerate()
            .map(|(index, vertex)| (vertex.id, index))
            .collect();
        let triangle_capacity = brush_mesh
            .faces
            .iter()
            .try_fold(0_usize, |count, face| {
                count.checked_add(face.vertices.len().saturating_sub(2))
            })
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "dynamic-paint brush triangle count overflows usize",
                )
            })?;
        let mut triangles = Vec::with_capacity(triangle_capacity);
        let mut bounds = Vec::with_capacity(triangle_capacity);
        let mut invalid_vertex = false;
        brush_mesh
            .for_each_triangle_with_face(|_, ids| {
                let Some(first) = vertex_indices.get(&ids[0]).copied() else {
                    invalid_vertex = true;
                    return;
                };
                let Some(second) = vertex_indices.get(&ids[1]).copied() else {
                    invalid_vertex = true;
                    return;
                };
                let Some(third) = vertex_indices.get(&ids[2]).copied() else {
                    invalid_vertex = true;
                    return;
                };
                let triangle = [
                    world_positions[first],
                    world_positions[second],
                    world_positions[third],
                ];
                bounds.push(crate::geom::bvh::kdop_bounds(&triangle, 0.0, 6));
                triangles.push(triangle);
            })
            .map_err(|error| {
                PotError::with_details(
                    ErrorCode::EvaluationFailed,
                    "dynamic-paint brush mesh triangulation failed",
                    json!({"brush_id":brush.node_id,"reason":error.to_string()}),
                )
            })?;
        if invalid_vertex {
            return Err(PotError::new(
                ErrorCode::EvaluationFailed,
                "dynamic-paint brush triangle references a missing vertex",
            ));
        }
        let tree = crate::geom::bvh::KdopTree::new(bounds, 8, 6);
        let mut mask = Vec::with_capacity(canvas.vertices.len());
        for (index, vertex) in canvas.vertices.iter().enumerate() {
            let origin = canvas_matrix.transform_point3(vertex.co.as_vec3()) + Vec3::splat(0.001);
            let world_normal = canvas_matrix.transform_vector3(canvas_normals[index].as_vec3());
            let normal_length = world_normal.length();
            if !origin.is_finite() || !normal_length.is_finite() {
                return Err(PotError::new(
                    ErrorCode::EvaluationFailed,
                    "dynamic-paint canvas volume-ray data must be finite",
                ));
            }
            if normal_length == 0.0 {
                mask.push(false);
                continue;
            }
            let direction = -world_normal / normal_length;
            let Some((triangle_index, _)) = dynamic_paint_volume_ray_hit(
                &tree,
                &triangles,
                origin.to_array(),
                direction.to_array(),
            ) else {
                mask.push(false);
                continue;
            };
            let triangle = triangles[triangle_index];
            let normal = crate::geom::bvh::normalize(crate::geom::bvh::cross(
                crate::geom::bvh::sub(triangle[1], triangle[0]),
                crate::geom::bvh::sub(triangle[2], triangle[0]),
            ))
            .0;
            if crate::geom::bvh::dot(direction.to_array(), normal) < 0.0 {
                mask.push(false);
                continue;
            }
            mask.push(
                dynamic_paint_volume_ray_hit(
                    &tree,
                    &triangles,
                    origin.to_array(),
                    (-direction).to_array(),
                )
                .is_some(),
            );
        }
        brush.volume_mask = Some(mask);
    }
    Ok(())
}

fn dynamic_paint_volume_ray_hit(
    tree: &crate::geom::bvh::KdopTree,
    triangles: &[[[f32; 3]; 3]],
    origin: [f32; 3],
    direction: [f32; 3],
) -> Option<(usize, f32)> {
    let mut hit = crate::geom::bvh::RayHit::open();
    tree.ray_cast(origin, direction, &mut hit, &mut |index, ray, hit| {
        let [first, second, third] = triangles[index];
        if let Some(distance) = ray.watertight(first, second, third)
            && distance >= 0.0
            && distance < hit.dist
            && let Ok(index) = isize::try_from(index)
        {
            hit.index = index;
            hit.dist = distance;
        }
    });
    hit.leaf().map(|index| (index, hit.dist))
}

fn dynamic_paint_canvas_with_shape_keys(
    context: &FrameContext<'_>,
    canvas_id: &Id,
    canvas: &Mesh,
) -> Result<Option<Mesh>> {
    let node = context.doc.nodes.get(canvas_id).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::TargetNotFound,
            "dynamic-paint canvas Object was not found",
            json!({"node_id":canvas_id}),
        )
    })?;
    let data_id = node.data.as_ref().ok_or_else(|| {
        PotError::with_details(
            ErrorCode::SceneInvalid,
            "dynamic-paint canvas Data-Block is missing",
            json!({"node_id":canvas_id}),
        )
    })?;
    let data_block = context.doc.data_blocks.get(data_id).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::SceneInvalid,
            "dynamic-paint canvas Data-Block was not found",
            json!({"node_id":canvas_id,"data_id":data_id}),
        )
    })?;
    let Some(shape_keys) = data_block.shape_keys.as_ref() else {
        return Ok(None);
    };
    let drivers = crate::eval::rig::evaluate_drivers(context.doc, context.frame)?;
    let values = crate::eval::rig::evaluate_shape_key_values(
        canvas_id,
        node,
        shape_keys,
        context.doc,
        context.frame,
        &drivers,
    )?;
    let mut mesh = canvas.clone();
    crate::eval::rig::apply_shape_keys_with_values(&mut mesh, shape_keys, data_block, &values)?;
    Ok(Some(mesh))
}

fn same_vertex_layout(left: &Mesh, right: &Mesh) -> bool {
    left.vertices.len() == right.vertices.len()
        && left
            .vertices
            .iter()
            .zip(&right.vertices)
            .all(|(left, right)| left.id == right.id)
}

fn same_vertex_ids(mesh: &Mesh, vertex_ids: &[u32]) -> bool {
    mesh.vertices.len() == vertex_ids.len()
        && mesh
            .vertices
            .iter()
            .zip(vertex_ids)
            .all(|(vertex, id)| vertex.id == *id)
}

fn dynamic_paint_topology_error() -> PotError {
    unsupported_feature(
        "physics.dynamic_paint_surface.topology",
        "the canvas topology changes during the dynamic-paint frame history or held range, so per-vertex values cannot be mapped",
    )
}

fn validate_dynamic_paint_held_mesh(current: &Mesh, held: &Mesh) -> Result<()> {
    if same_vertex_layout(current, held) {
        Ok(())
    } else {
        Err(dynamic_paint_topology_error())
    }
}

fn validate_dynamic_paint_held_layout(current: &Mesh, held_vertex_ids: &[u32]) -> Result<()> {
    if same_vertex_ids(current, held_vertex_ids) {
        Ok(())
    } else {
        Err(dynamic_paint_topology_error())
    }
}

fn dynamic_paint_hold_cache_key(
    context: &FrameContext<'_>,
    canvas_id: &Id,
    scene_hash: &str,
    frame_end: f64,
    modifier_prefix: usize,
) -> Result<String> {
    let input = json!({
        "build_identity": crate::BUILD_IDENTITY,
        "scene_hash": scene_hash,
        "scene_id": context.scene_id,
        "view_layer": context.view_layer,
        "purpose": context.gate.context.purpose,
        "suppressed_nodes": context.gate.suppressed,
        "project_root": context.gate.context.project_root.map(|path| path.to_string_lossy().into_owned()),
        "cache_directory": context.gate.context.cache_directory.map(|path| path.to_string_lossy().into_owned()),
        "canvas_id": canvas_id,
        "frame_end": frame_end,
        "modifier_prefix": modifier_prefix,
        "suppress_dynamic_paint": true,
    });
    Ok(crate::hash::sha256(&crate::hash::canonicalize(&input)?))
}

struct DynamicPaintEvaluation<'a> {
    scene_hash: &'a str,
    modifier_prefix: usize,
    stack_mesh_is_current: bool,
    hold_cache: &'a mut DynamicPaintHoldCache,
}

const MAX_DYNAMIC_PAINT_FRAMES: u64 = 100_000;

fn dynamic_paint_integer_frame(value: f64, property: &str) -> Result<i64> {
    if !value.is_finite()
        || value.fract() != 0.0
        || value < i64::MIN as f64
        || value >= i64::MAX as f64
    {
        return Err(PotError::invalid_argument(format!(
            "dynamic-paint surface {property} must be an integer frame"
        )));
    }
    Ok(value as i64)
}

fn paint_dynamic_paint_weight(
    context: &FrameContext<'_>,
    canvas_id: &Id,
    canvas: &Mesh,
    settings: &Value,
    surface: &Value,
    paint_frame: Option<f64>,
    scene_hash: &str,
    modifier_prefix: usize,
    stack_mesh_is_current: bool,
    hold_cache: &mut DynamicPaintHoldCache,
) -> Result<DynamicPaintOutput> {
    let output_name = dynamic_paint_output_name(settings);
    let Some(paint_frame) = paint_frame else {
        return Ok(DynamicPaintOutput {
            output_name,
            values: DynamicPaintValues::Weights(Arc::new(vec![0.0; canvas.vertices.len()])),
            wetness_output: None,
        });
    };
    let start = dynamic_paint_integer_frame(
        setting_number(
            surface,
            "frame_start",
            1.0,
            "dynamic-paint surface",
            "an integer frame",
        )?,
        "frame_start",
    )?;
    let end = dynamic_paint_integer_frame(
        setting_number(
            surface,
            "frame_end",
            250.0,
            "dynamic-paint surface",
            "an integer frame",
        )?,
        "frame_end",
    )?;
    let target = dynamic_paint_integer_frame(paint_frame.floor(), "evaluation frame")?;
    if start > end || target < start {
        return Ok(DynamicPaintOutput {
            output_name,
            values: DynamicPaintValues::Weights(Arc::new(vec![0.0; canvas.vertices.len()])),
            wetness_output: None,
        });
    }
    let frame_count = target
        .min(end)
        .checked_sub(start)
        .and_then(|count| count.checked_add(1))
        .and_then(|count| u64::try_from(count).ok())
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "dynamic-paint frame range overflows",
            )
        })?;
    if frame_count > MAX_DYNAMIC_PAINT_FRAMES {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "dynamic-paint weight frame range exceeds limits",
        ));
    }
    let target = target.min(end);
    let frame_end = setting_number(
        surface,
        "frame_end",
        250.0,
        "dynamic-paint surface",
        "a finite number",
    )?;
    let cache_key =
        dynamic_paint_hold_cache_key(context, canvas_id, scene_hash, frame_end, modifier_prefix)?;
    if let Some(state) = hold_cache.weight_entries.get(&cache_key)
        && state.frame == target
    {
        if stack_mesh_is_current && paint_frame < context.frame {
            validate_dynamic_paint_held_layout(canvas, &state.vertex_ids)?;
        }
        return Ok(DynamicPaintOutput {
            output_name,
            values: DynamicPaintValues::Weights(Arc::clone(&state.weights)),
            wetness_output: None,
        });
    }

    let previous = hold_cache
        .weight_entries
        .get(&cache_key)
        .filter(|state| state.frame >= start && state.frame < target);
    let mut weights: Arc<Vec<f64>> = previous.map_or_else(
        || Arc::new(vec![0.0; canvas.vertices.len()]),
        |state| Arc::clone(&state.weights),
    );
    let mut vertex_ids = previous.map(|state| state.vertex_ids.clone());
    let first_frame = previous
        .and_then(|state| state.frame.checked_add(1))
        .unwrap_or(start);
    let use_dissolve = bool_setting(surface, "use_dissolve", false, "dynamic-paint surface")?;
    let dissolve_log = bool_setting(surface, "use_dissolve_log", true, "dynamic-paint surface")?;
    let dissolve_speed = setting_number(
        surface,
        "dissolve_speed",
        250.0,
        "dynamic-paint surface",
        "a positive finite number",
    )?;
    let influence_scale = setting_number(
        surface,
        "brush_influence_scale",
        1.0,
        "dynamic-paint surface",
        "a finite nonnegative number",
    )?;
    if dissolve_speed <= 0.0 {
        return Err(PotError::invalid_argument(
            "dynamic-paint surface dissolve_speed must be positive",
        ));
    }
    for frame in first_frame..=target {
        let frame_value = frame as f64;
        let matrices = scene_world_matrices_at(context, frame_value)?;
        let paint_context = FrameContext {
            frame: frame_value,
            matrices: &matrices,
            ..*context
        };
        let mut snapshot =
            dynamic_paint_mesh_snapshot(&paint_context, canvas_id, frame_value, modifier_prefix)?;
        let paint_mesh = snapshot.meshes.remove(canvas_id).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::EvaluationFailed,
                "dynamic-paint canvas mesh is missing at its weight frame",
                json!({"node_id":canvas_id,"frame":frame_value}),
            )
        })?;
        if let Some(expected_ids) = vertex_ids.as_deref() {
            if !same_vertex_ids(&paint_mesh, expected_ids) {
                return Err(dynamic_paint_topology_error());
            }
        } else {
            if weights.len() != paint_mesh.vertices.len() {
                weights = Arc::new(vec![0.0; paint_mesh.vertices.len()]);
            }
            vertex_ids = Some(paint_mesh.vertices.iter().map(|vertex| vertex.id).collect());
        }
        if use_dissolve {
            let factor = if dissolve_log {
                0.001_f64.powf(1.0 / (1.2 * dissolve_speed))
            } else {
                1.0
            };
            for weight in Arc::make_mut(&mut weights) {
                *weight = if dissolve_log {
                    *weight * factor
                } else {
                    *weight - 1.0 / dissolve_speed
                }
                .max(0.0);
            }
        }
        let mut brushes = dynamic_paint_brush_settings(&paint_context, settings, true)?;
        let canvas_matrix = matrices.get(canvas_id).copied().ok_or_else(|| {
            PotError::with_details(
                ErrorCode::EvaluationFailed,
                "dynamic-paint canvas transform is missing at its weight frame",
                json!({"node_id":canvas_id,"frame":frame_value}),
            )
        })?;
        prepare_dynamic_paint_volume_brushes(
            &paint_context,
            &paint_mesh,
            canvas_matrix,
            &mut brushes,
        )?;
        dynamic_paint::paint_weight_step(
            &paint_mesh,
            canvas_matrix,
            &brushes,
            Arc::make_mut(&mut weights).as_mut_slice(),
            influence_scale,
        )?;
    }
    let vertex_ids = vertex_ids.unwrap_or_default();
    if stack_mesh_is_current && paint_frame < context.frame {
        validate_dynamic_paint_held_layout(canvas, &vertex_ids)?;
    }
    hold_cache.weight_entries.insert(
        cache_key,
        DynamicPaintWeightState {
            frame: target,
            vertex_ids,
            weights: Arc::clone(&weights),
        },
    );
    Ok(DynamicPaintOutput {
        output_name,
        values: DynamicPaintValues::Weights(weights),
        wetness_output: None,
    })
}
fn dynamic_paint_wet_output_name(surface: &Value) -> Option<String> {
    let name = setting_string(surface, "output_name_b").unwrap_or("dp_wetmap");
    (!name.is_empty()).then(|| name.to_owned())
}

fn paint_dynamic_paint_paint(
    context: &FrameContext<'_>,
    canvas_id: &Id,
    canvas: &Mesh,
    settings: &Value,
    surface: &Value,
    paint_frame: Option<f64>,
    scene_hash: &str,
    modifier_prefix: usize,
    stack_mesh_is_current: bool,
    hold_cache: &mut DynamicPaintHoldCache,
) -> Result<DynamicPaintOutput> {
    let output_name = dynamic_paint_output_name(settings)
        .filter(|name| dynamic_paint_output_layer_exists(context, canvas_id, canvas, name));
    let wetness_name = dynamic_paint_wet_output_name(surface)
        .filter(|name| dynamic_paint_output_layer_exists(context, canvas_id, canvas, name));
    let Some(paint_frame) = paint_frame else {
        return Ok(DynamicPaintOutput {
            output_name,
            values: DynamicPaintValues::Colors(Arc::new(vec![[0.0; 4]; canvas.vertices.len()])),
            wetness_output: wetness_name
                .map(|name| (name, Arc::new(vec![0.0; canvas.vertices.len()]))),
        });
    };
    let start = dynamic_paint_integer_frame(
        setting_number(
            surface,
            "frame_start",
            1.0,
            "dynamic-paint surface",
            "an integer frame",
        )?,
        "frame_start",
    )?;
    let end = dynamic_paint_integer_frame(
        setting_number(
            surface,
            "frame_end",
            250.0,
            "dynamic-paint surface",
            "an integer frame",
        )?,
        "frame_end",
    )?;
    let target = dynamic_paint_integer_frame(paint_frame.floor(), "evaluation frame")?;
    if start > end || target < start {
        return Ok(DynamicPaintOutput {
            output_name,
            values: DynamicPaintValues::Colors(Arc::new(vec![[0.0; 4]; canvas.vertices.len()])),
            wetness_output: wetness_name
                .map(|name| (name, Arc::new(vec![0.0; canvas.vertices.len()]))),
        });
    }
    let target = target.min(end);
    let frame_count = target
        .checked_sub(start)
        .and_then(|count| count.checked_add(1))
        .and_then(|count| u64::try_from(count).ok())
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "dynamic-paint frame range overflows",
            )
        })?;
    if frame_count > MAX_DYNAMIC_PAINT_FRAMES {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "dynamic-paint paint frame range exceeds limits",
        ));
    }
    let cache_key =
        dynamic_paint_hold_cache_key(context, canvas_id, scene_hash, end as f64, modifier_prefix)?;
    if let Some(state) = hold_cache.paint_entries.get(&cache_key)
        && state.frame == target
    {
        if stack_mesh_is_current && paint_frame < context.frame {
            validate_dynamic_paint_held_layout(canvas, &state.vertex_ids)?;
        }
        let colors = dynamic_paint::paint_output(&state.points);
        return Ok(DynamicPaintOutput {
            output_name,
            values: DynamicPaintValues::Colors(Arc::new(colors)),
            wetness_output: wetness_name.map(|name| {
                (
                    name,
                    Arc::new(dynamic_paint::paint_wetness_output(&state.points)),
                )
            }),
        });
    }
    let previous = hold_cache
        .paint_entries
        .get(&cache_key)
        .filter(|state| state.frame >= start && state.frame < target);
    let mut points: Arc<Vec<dynamic_paint::PaintPoint>> = previous.map_or_else(
        || {
            Arc::new(vec![
                dynamic_paint::PaintPoint::default();
                canvas.vertices.len()
            ])
        },
        |state| Arc::clone(&state.points),
    );
    let mut vertex_ids = previous.map(|state| state.vertex_ids.clone());
    let first_frame = previous
        .and_then(|state| state.frame.checked_add(1))
        .unwrap_or(start);
    let use_drying = bool_setting(surface, "use_drying", true, "dynamic-paint surface")?;
    let dry_speed = setting_number(
        surface,
        "dry_speed",
        500.0,
        "dynamic-paint surface",
        "a finite number",
    )? as f32;
    let dry_log = bool_setting(surface, "use_dry_log", true, "dynamic-paint surface")?;
    let color_dry_threshold = setting_number(
        surface,
        "color_dry_threshold",
        1.0,
        "dynamic-paint surface",
        "a finite number",
    )? as f32;
    let use_dissolve = bool_setting(surface, "use_dissolve", false, "dynamic-paint surface")?;
    let dissolve_speed = setting_number(
        surface,
        "dissolve_speed",
        250.0,
        "dynamic-paint surface",
        "a finite number",
    )? as f32;
    let dissolve_log = bool_setting(surface, "use_dissolve_log", true, "dynamic-paint surface")?;
    let influence_scale = setting_number(
        surface,
        "brush_influence_scale",
        1.0,
        "dynamic-paint surface",
        "a finite nonnegative number",
    )? as f32;
    for frame in first_frame..=target {
        let frame_value = frame as f64;
        let matrices = scene_world_matrices_at(context, frame_value)?;
        let paint_context = FrameContext {
            frame: frame_value,
            matrices: &matrices,
            ..*context
        };
        let mut snapshot =
            dynamic_paint_mesh_snapshot(&paint_context, canvas_id, frame_value, modifier_prefix)?;
        let paint_mesh = snapshot.meshes.remove(canvas_id).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::EvaluationFailed,
                "dynamic-paint canvas mesh is missing at its paint frame",
                json!({"node_id":canvas_id,"frame":frame_value}),
            )
        })?;
        if let Some(expected_ids) = vertex_ids.as_deref() {
            if !same_vertex_ids(&paint_mesh, expected_ids) {
                return Err(dynamic_paint_topology_error());
            }
        } else {
            vertex_ids = Some(paint_mesh.vertices.iter().map(|vertex| vertex.id).collect());
            points = Arc::new(vec![
                dynamic_paint::PaintPoint::default();
                paint_mesh.vertices.len()
            ]);
        }
        if points.len() != paint_mesh.vertices.len() {
            return Err(dynamic_paint_topology_error());
        }
        let mut brushes = dynamic_paint_brush_settings(&paint_context, settings, false)?;
        let canvas_matrix = matrices.get(canvas_id).copied().ok_or_else(|| {
            PotError::with_details(
                ErrorCode::EvaluationFailed,
                "dynamic-paint canvas transform is missing at its paint frame",
                json!({"node_id":canvas_id,"frame":frame_value}),
            )
        })?;
        prepare_dynamic_paint_volume_brushes(
            &paint_context,
            &paint_mesh,
            canvas_matrix,
            &mut brushes,
        )?;
        dynamic_paint::paint_step(
            &paint_mesh,
            canvas_matrix,
            &brushes,
            Arc::make_mut(&mut points).as_mut_slice(),
            dynamic_paint::PaintSettings {
                influence_scale,
                use_drying,
                dry_speed,
                dry_log,
                color_dry_threshold,
                use_dissolve,
                dissolve_speed,
                dissolve_log,
            },
        )?;
    }
    let vertex_ids = vertex_ids.unwrap_or_default();
    if stack_mesh_is_current && paint_frame < context.frame {
        validate_dynamic_paint_held_layout(canvas, &vertex_ids)?;
    }
    hold_cache.paint_entries.insert(
        cache_key,
        DynamicPaintPaintState {
            frame: target,
            vertex_ids,
            points: Arc::clone(&points),
        },
    );
    let colors = dynamic_paint::paint_output(&points);
    Ok(DynamicPaintOutput {
        output_name,
        values: DynamicPaintValues::Colors(Arc::new(colors)),
        wetness_output: wetness_name
            .map(|name| (name, Arc::new(dynamic_paint::paint_wetness_output(&points)))),
    })
}

/// PAINT/VERTEX color and wetness and WEIGHT/VERTEX values replay each integer frame and hold after
/// `frame_end`; PAINT drying and dissolve advance before each frame's brush updates.
fn paint_canvas(
    context: &FrameContext<'_>,
    canvas_id: &Id,
    canvas: &Mesh,
    settings: &Value,
    evaluation: DynamicPaintEvaluation<'_>,
) -> Result<DynamicPaintOutput> {
    validate_dynamic_paint_canvas(settings)?;
    let DynamicPaintEvaluation {
        scene_hash,
        modifier_prefix,
        stack_mesh_is_current,
        hold_cache,
    } = evaluation;
    let surface = settings
        .get("canvas_surfaces")
        .and_then(Value::as_array)
        .and_then(|surfaces| surfaces.first());
    let paint_frame = match surface {
        Some(surface) => surface_paint_frame(surface, context.frame)?,
        None => Some(context.frame),
    };
    let geometry_frame = paint_frame.unwrap_or(context.frame);
    let frame_end = surface
        .map(|surface| {
            setting_number(
                surface,
                "frame_end",
                250.0,
                "dynamic-paint surface",
                "a finite number",
            )
        })
        .transpose()?;
    if let Some(weight_surface) =
        surface.filter(|surface| setting_string(surface, "surface_type") == Some("WEIGHT"))
    {
        return paint_dynamic_paint_weight(
            context,
            canvas_id,
            canvas,
            settings,
            weight_surface,
            paint_frame,
            scene_hash,
            modifier_prefix,
            stack_mesh_is_current,
            hold_cache,
        );
    }
    if let Some(paint_surface) = surface
        .filter(|surface| setting_string(surface, "surface_type").unwrap_or("PAINT") == "PAINT")
    {
        return paint_dynamic_paint_paint(
            context,
            canvas_id,
            canvas,
            settings,
            paint_surface,
            paint_frame,
            scene_hash,
            modifier_prefix,
            stack_mesh_is_current,
            hold_cache,
        );
    }
    let cache_key = match (paint_frame, frame_end) {
        (Some(paint_frame), Some(frame_end)) if paint_frame < context.frame => {
            Some(dynamic_paint_hold_cache_key(
                context,
                canvas_id,
                scene_hash,
                frame_end,
                modifier_prefix,
            )?)
        }
        _ => None,
    };
    if let Some(held) = cache_key
        .as_ref()
        .and_then(|key| hold_cache.entries.get(key))
    {
        if stack_mesh_is_current {
            validate_dynamic_paint_held_layout(canvas, &held.vertex_ids)?;
        }
        return Ok(held.output.clone());
    }
    let held_frame = paint_frame.is_some_and(|frame| frame < context.frame);
    let needs_frame_mesh =
        paint_frame.is_some() && (held_frame || (!stack_mesh_is_current && modifier_prefix > 0));
    let frame_mesh = if needs_frame_mesh {
        let mut snapshot =
            dynamic_paint_mesh_snapshot(context, canvas_id, geometry_frame, modifier_prefix)?;
        Some(snapshot.meshes.remove(canvas_id).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::EvaluationFailed,
                "dynamic-paint canvas mesh is missing at its paint frame",
                json!({"node_id":canvas_id,"frame":geometry_frame}),
            )
        })?)
    } else if paint_frame == Some(context.frame) && !stack_mesh_is_current && modifier_prefix == 0 {
        dynamic_paint_canvas_with_shape_keys(context, canvas_id, canvas)?
    } else {
        None
    };
    let paint_mesh = frame_mesh.as_ref().unwrap_or(canvas);
    if stack_mesh_is_current && paint_frame.is_some_and(|frame| frame < context.frame) {
        validate_dynamic_paint_held_mesh(canvas, paint_mesh)?;
    }
    let output_name = dynamic_paint_output_name(settings);
    let Some(paint_frame) = paint_frame else {
        return Ok(DynamicPaintOutput {
            output_name,
            values: DynamicPaintValues::Colors(Arc::new(vec![[0.0; 4]; paint_mesh.vertices.len()])),
            wetness_output: None,
        });
    };
    let paint_matrices;
    let paint_context = if paint_frame < context.frame {
        paint_matrices = scene_world_matrices_at(context, paint_frame)?;
        FrameContext {
            frame: paint_frame,
            matrices: &paint_matrices,
            ..*context
        }
    } else {
        *context
    };
    let matrix = paint_context
        .matrices
        .get(canvas_id)
        .copied()
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::EvaluationFailed,
                "dynamic-paint canvas transform is missing at its paint frame",
                json!({"node_id":canvas_id}),
            )
        })?;
    let brushes = dynamic_paint_brush_settings(&paint_context, settings, false)?;
    let surface = settings
        .get("canvas_surfaces")
        .and_then(Value::as_array)
        .and_then(|surfaces| surfaces.first());
    if surface.is_none() && brushes.iter().any(|brush| brush.volume_mask.is_some()) {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "VOLUME brush meshes require an explicit Dynamic Paint surface",
            json!({"feature_id":"physics.dynamic_paint_canvas.volume_without_surface","status":"not_supported"}),
        ));
    }
    let radius_scale = surface
        .and_then(|surface| setting_f64(surface, "brush_radius_scale"))
        .unwrap_or(1.0);
    let influence_scale = surface
        .and_then(|surface| setting_f64(surface, "brush_influence_scale"))
        .unwrap_or(1.0);
    let colors = if brushes.is_empty() {
        let mut colors = dynamic_paint::paint(
            paint_mesh,
            matrix,
            &[],
            setting_f64(settings, "radius").unwrap_or(1.0) * radius_scale,
            setting_color(settings).unwrap_or([1.0, 0.0, 0.0, 1.0]),
            setting_f64(settings, "strength").unwrap_or(1.0),
        )?;
        for color in &mut colors {
            color[3] = (color[3] * influence_scale).clamp(0.0, 1.0);
        }
        colors
    } else {
        let mut colors = vec![[0.0; 4]; paint_mesh.vertices.len()];
        for brush in &brushes {
            let mut painted = dynamic_paint::paint(
                paint_mesh,
                matrix,
                &[brush.position],
                brush.radius,
                brush.color,
                brush.strength,
            )?;
            for color in &mut painted {
                color[3] = (color[3] * influence_scale).clamp(0.0, 1.0);
            }
            for (current, candidate) in colors.iter_mut().zip(painted) {
                if candidate[3] > current[3] {
                    *current = candidate;
                }
            }
        }
        colors
    };
    let output = DynamicPaintOutput {
        output_name,
        values: DynamicPaintValues::Colors(Arc::new(colors)),
        wetness_output: None,
    };
    if let Some(key) = cache_key {
        hold_cache.entries.insert(
            key,
            DynamicPaintHold {
                vertex_ids: paint_mesh.vertices.iter().map(|vertex| vertex.id).collect(),
                output: output.clone(),
            },
        );
    }
    Ok(output)
}

pub(crate) fn write_dynamic_paint_color_layer(
    mesh: &mut Mesh,
    name: &str,
    colors: &[[f64; 4]],
    default_domain: crate::geom::color_layers::ColorDomain,
    set_render: bool,
) -> Result<()> {
    if colors.len() != mesh.vertices.len() {
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            "dynamic-paint color count does not match canvas vertices",
        ));
    }
    let mut store = crate::geom::color_layers::read(mesh);
    let domain = store
        .layers
        .iter()
        .find(|layer| layer.name == name)
        .map_or(default_domain, |layer| layer.domain);
    let values = match domain {
        crate::geom::color_layers::ColorDomain::Point => colors.to_vec(),
        crate::geom::color_layers::ColorDomain::Corner => {
            let mut vertex_indices = std::collections::HashMap::with_capacity(mesh.vertices.len());
            for (index, vertex) in mesh.vertices.iter().enumerate() {
                vertex_indices.insert(vertex.id, index);
            }
            let corner_count = mesh
                .faces
                .iter()
                .try_fold(0_usize, |count, face| {
                    count.checked_add(face.vertices.len())
                })
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::LimitExceeded,
                        "dynamic-paint corner count overflows",
                    )
                })?;
            let mut corner_colors = Vec::with_capacity(corner_count);
            for face in &mesh.faces {
                for vertex_id in &face.vertices {
                    let index = vertex_indices.get(vertex_id).copied().ok_or_else(|| {
                        PotError::new(
                            ErrorCode::SceneInvalid,
                            "dynamic-paint face references a missing vertex",
                        )
                    })?;
                    corner_colors.push(colors[index]);
                }
            }
            corner_colors
        }
    };
    store.layers.retain(|layer| layer.name != name);
    store.layers.push(crate::geom::color_layers::ColorLayer {
        name: name.to_owned(),
        domain,
        values,
    });
    if set_render {
        store.active = Some(name.to_owned());
        store.render = Some(name.to_owned());
    }
    crate::geom::color_layers::write(mesh, &store);
    Ok(())
}
pub(crate) fn dynamic_paint_wetness_colors(wetness: &[f64]) -> Vec<[f64; 4]> {
    wetness
        .iter()
        .map(|value| dynamic_paint::wetness_color_attribute(*value))
        .collect()
}
fn write_dynamic_paint_weight_group(mesh: &mut Mesh, name: &str, weights: &[f64]) -> Result<()> {
    if weights.len() != mesh.vertices.len()
        || weights
            .iter()
            .any(|weight| !weight.is_finite() || !(0.0..=1.0).contains(weight))
    {
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            "dynamic-paint weights are invalid or do not match canvas vertices",
        ));
    }
    let groups = mesh
        .attributes
        .entry("vertex_groups".to_owned())
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::SceneInvalid,
                "dynamic-paint vertex-group storage is invalid",
            )
        })?;
    let values = groups
        .entry(name.to_owned())
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::SceneInvalid,
                "dynamic-paint output vertex group is invalid",
            )
        })?;
    for (vertex, weight) in mesh.vertices.iter().zip(weights) {
        let key = format!("v{}", vertex.id);
        if *weight != 0.0 || values.contains_key(&key) {
            values.insert(key, json!(weight));
        }
    }
    Ok(())
}

#[cfg(test)]
mod dynamic_paint_frame_tests {
    use serde_json::json;

    use super::{surface_paint_frame, validate_dynamic_paint_canvas};
    use crate::error::{ErrorCode, Result};

    /// The paint frame, as an exact integer-valued frame, for comparisons without float equality.
    fn paint_frame_in_tenths(surface: &serde_json::Value, frame: f64) -> Result<Option<i64>> {
        Ok(surface_paint_frame(surface, frame)?.map(|paint| (paint * 10.0).round() as i64))
    }

    #[test]
    fn surface_stays_unpainted_before_start_and_holds_end_afterward() -> Result<()> {
        let surface = json!({"frame_start": 5, "frame_end": 20});
        assert_eq!(paint_frame_in_tenths(&surface, 4.0)?, None);
        assert_eq!(paint_frame_in_tenths(&surface, 5.0)?, Some(50));
        assert_eq!(paint_frame_in_tenths(&surface, 20.0)?, Some(200));
        assert_eq!(paint_frame_in_tenths(&surface, 21.0)?, Some(200));
        assert_eq!(paint_frame_in_tenths(&surface, 300.0)?, Some(200));
        Ok(())
    }

    #[test]
    fn inverted_surface_range_never_paints() -> Result<()> {
        let surface = json!({"frame_start": 10, "frame_end": 5});
        assert_eq!(paint_frame_in_tenths(&surface, 1.0)?, None);
        assert_eq!(paint_frame_in_tenths(&surface, 5.0)?, None);
        assert_eq!(paint_frame_in_tenths(&surface, 10.0)?, None);
        assert_eq!(paint_frame_in_tenths(&surface, 20.0)?, None);
        validate_dynamic_paint_canvas(&json!({
            "canvas_surfaces": [{
                "frame_start":10,
                "frame_end":5,
                "point_cache":{"frame_start":10,"frame_end":5}
            }]
        }))?;
        Ok(())
    }

    #[test]
    fn multiple_surfaces_are_refused_with_the_catalog_feature() -> Result<()> {
        let error = validate_dynamic_paint_canvas(&json!({"canvas_surfaces":[{},{}]}))
            .err()
            .ok_or_else(|| {
                crate::error::PotError::new(
                    ErrorCode::InternalError,
                    "multiple surfaces unexpectedly validated",
                )
            })?;
        assert_eq!(error.code, ErrorCode::UnsupportedFeature);
        assert_eq!(
            error.details["feature_id"],
            "physics.dynamic_paint_canvas.canvas_surfaces"
        );
        Ok(())
    }
    #[test]
    fn weight_vertex_surface_is_supported_while_other_types_remain_guarded() -> Result<()> {
        validate_dynamic_paint_canvas(&json!({
            "canvas_surfaces": [{
                "surface_type": "WEIGHT",
                "surface_format": "VERTEX",
                "use_dissolve": true,
                "dissolve_speed": 12,
                "use_dissolve_log": false
            }]
        }))?;

        for surface_type in ["DISPLACE", "WAVE"] {
            let error = validate_dynamic_paint_canvas(&json!({
                "canvas_surfaces": [{"surface_type":surface_type}]
            }))
            .err()
            .ok_or_else(|| {
                crate::error::PotError::new(
                    ErrorCode::InternalError,
                    "unsupported dynamic-paint type unexpectedly validated",
                )
            })?;
            assert_eq!(error.code, ErrorCode::UnsupportedFeature);
            assert_eq!(
                error.details["feature_id"],
                "physics.dynamic_paint_surface.surface_type"
            );
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use glam::{DQuat, DVec3};

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
            birth_position_local: None,
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
                0.0,
                true,
                false,
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
            0.0,
            true,
            false,
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
            0.0,
            true,
            false,
        );
        let normal_orientation = particle_orientation([0.0, 0.0, 1.0]);
        assert_eq!(linear.rotation, normal_orientation);
        let mut moving = particle_for_state(
            ParticlePathInterpolation::HairCache,
            Vec::new(),
            birth_rotation,
            current_path_rotation,
        );
        moving.velocity = [1.0, 2.0, 3.0];
        let heading = DVec3::new(1.0, 2.0, 3.0).normalize();
        let oriented = DQuat::from_array(particle_state(moving, 0, 0.0, true, false).rotation);
        // The orientation is Blender's f32 `vec_to_quat`, so local +X matches the heading to f32.
        assert!((oriented * DVec3::X - heading).length() < 1.0e-6);
        let stationary = particle_state(
            particle_for_state(
                ParticlePathInterpolation::HairCache,
                Vec::new(),
                birth_rotation,
                current_path_rotation,
            ),
            0,
            0.0,
            true,
            false,
        );
        assert_eq!(stationary.rotation, DQuat::IDENTITY.to_array());
        let disabled_path_rotations = particle_state(
            particle_for_state(
                ParticlePathInterpolation::Hermite,
                path.clone(),
                birth_rotation,
                current_path_rotation,
            ),
            0,
            0.0,
            false,
            false,
        );
        assert_eq!(disabled_path_rotations.rotation, DQuat::IDENTITY.to_array());

        let disabled_rotations_at_birth = particle_state(
            particle_for_state(
                ParticlePathInterpolation::Hermite,
                Vec::new(),
                birth_rotation,
                current_path_rotation,
            ),
            0,
            0.0,
            false,
            false,
        );
        assert_eq!(disabled_rotations_at_birth.rotation, birth_rotation);
        let disabled_rotations_after_birth = particle_state(
            particle_for_state(
                ParticlePathInterpolation::Hermite,
                Vec::new(),
                birth_rotation,
                current_path_rotation,
            ),
            0,
            0.5,
            false,
            false,
        );
        assert_eq!(
            disabled_rotations_after_birth.rotation,
            DQuat::IDENTITY.to_array()
        );
        let no_physics_after_birth = particle_state(
            particle_for_state(
                ParticlePathInterpolation::Hermite,
                Vec::new(),
                birth_rotation,
                current_path_rotation,
            ),
            0,
            0.5,
            false,
            true,
        );
        assert_eq!(no_physics_after_birth.rotation, birth_rotation);
    }
}
