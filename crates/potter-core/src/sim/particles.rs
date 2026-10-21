//! Deterministic mesh-emitter particle simulation.

use super::particle_distribution;
use super::particle_textures::{TextureSlot, birth_values, maps_density, maps_time};
use super::{ParticleLifeState, ParticlePathInterpolation, ParticlePathKey};
use std::collections::HashMap;

use glam::{DMat3, DMat4, DQuat, DVec3};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    blender_rng::{BlenderRng, psys_frand},
    error::{ErrorCode, PotError, Result},
    geom::Mesh,
};

const MAX_PARTICLES: u64 = 1_000_000;
const MAX_EMITTER_VERTICES: usize = 1_000_000;
const MAX_EMITTER_TRIANGLES: usize = 2_000_000;
const MAX_EMITTER_FACES: usize = 1_000_000;
const MAX_FORCE_FIELDS: usize = 1_024;
const MAX_COLLIDERS: usize = 100_000;
const MAX_STEPS_PER_PARTICLE: usize = 4_096;
const MAX_SIMULATION_WORK: usize = 50_000_000;

/// A particle evaluated at one simulation frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Particle {
    pub position: [f64; 3],
    pub birth_position: [f64; 3],
    pub normal: [f64; 3],
    pub birth_rotation: [f64; 4],
    /// Current interpolated path rotation, when path keys are available.
    #[serde(default)]
    pub current_path_rotation: Option<[f64; 4]>,
    /// Root emitter-face frame used by Particle Instance path evaluation.
    #[serde(default)]
    pub path_root_rotation: Option<[f64; 4]>,
    /// Interpolation used to evaluate `path`.
    #[serde(default)]
    pub path_interpolation: ParticlePathInterpolation,
    /// World-space velocity in meters per second, or a normalized-lifetime derivative when sampled from path keys.
    pub velocity: [f64; 3],
    /// Birth time in frame units.
    pub birth_frame: f64,
    pub death_frame: f64,
    pub life_state: ParticleLifeState,
    pub is_child: bool,
    pub parent_index: Option<u32>,
    /// Weighted parent particles for interpolated children; `parent_index` remains representative.
    #[serde(default)]
    pub parent_indices: [u32; 3],
    #[serde(default)]
    pub parent_weights: [f64; 3],
    #[serde(default)]
    pub parent_count: u8,
    /// Imported trajectory keys used for current-state sampling and path instances.
    pub path: Vec<ParticlePathKey>,
    pub size: f64,
    pub mass: f64,
}

#[derive(Clone, Copy)]
struct ChildSettings {
    count: usize,
    radius: f64,
    flat: f64,
    size: f64,
    size_random: f64,
    child_seed: u32,
    system_seed: u32,
    lifetime: f64,
    lifetime_random: f64,
    interpolated: bool,
    use_emit_random: bool,
    virtual_parents: f64,
}

#[derive(Clone, Copy)]
struct Triangle {
    points: [DVec3; 3],
    normal: DVec3,
    frame_tangent: DVec3,
    frame_normal: DVec3,
    cumulative_area: f64,
}

#[derive(Clone, Copy)]
enum FieldKind {
    Wind,
    Vortex,
    Force,
}

#[derive(Clone, Copy)]
struct ForceField {
    position: DVec3,
    kind: FieldKind,
    strength: f64,
    falloff: f64,
}

#[derive(Clone, Copy)]
struct ColliderTriangle {
    points: [DVec3; 3],
    normal: DVec3,
}

#[derive(Clone, Copy)]
struct Random(u64);

impl Random {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9e37_79b9_7f4a_7c15))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64))
    }

    fn signed(&mut self) -> f64 {
        self.unit() * 2.0 - 1.0
    }
}

/// Emit and evaluate particles born no later than `frame`.
/// Particle `i` of `count` is born at `frame_start + (frame_end - frame_start) * i / count` (Blender's
/// `init_particle`), `lifetime` is measured in frames, and `speed`,
/// `velocity`, and `random_velocity` are emitter-local velocities in meters per second.
/// `force_fields` use world-space origins and radial falloff (zero means unlimited range);
/// `colliders` are static world-space triangles.
pub fn simulate(
    mesh: &Mesh,
    settings: &Value,
    world: DMat4,
    frame: f64,
    fps: f64,
    gravity: DVec3,
    seed: u32,
    force_fields: &[(DVec3, &str, f64, f64)],
    colliders: &[[DVec3; 3]],
) -> Result<Vec<Particle>> {
    simulate_textured(
        mesh,
        settings,
        world,
        frame,
        fps,
        gravity,
        seed,
        force_fields,
        colliders,
        &[],
    )
}

/// [`simulate`] with the particle system's texture slots (`psys_get_texture`) applied at birth.
pub fn simulate_textured(
    mesh: &Mesh,
    settings: &Value,
    world: DMat4,
    frame: f64,
    fps: f64,
    gravity: DVec3,
    seed: u32,
    force_fields: &[(DVec3, &str, f64, f64)],
    colliders: &[[DVec3; 3]],
    textures: &[TextureSlot<'_>],
) -> Result<Vec<Particle>> {
    simulate_with_lifecycle(
        mesh,
        settings,
        world,
        frame,
        fps,
        gravity,
        seed,
        force_fields,
        colliders,
        false,
        textures,
    )
}

/// Emits lifecycle states, including scheduled unborn and frozen dead particles.
pub fn simulate_lifecycle(
    mesh: &Mesh,
    settings: &Value,
    world: DMat4,
    frame: f64,
    fps: f64,
    gravity: DVec3,
    seed: u32,
    force_fields: &[(DVec3, &str, f64, f64)],
    colliders: &[[DVec3; 3]],
) -> Result<Vec<Particle>> {
    simulate_lifecycle_textured(
        mesh,
        settings,
        world,
        frame,
        fps,
        gravity,
        seed,
        force_fields,
        colliders,
        &[],
    )
}

/// [`simulate_lifecycle`] with the particle system's texture slots applied at birth.
pub fn simulate_lifecycle_textured(
    mesh: &Mesh,
    settings: &Value,
    world: DMat4,
    frame: f64,
    fps: f64,
    gravity: DVec3,
    seed: u32,
    force_fields: &[(DVec3, &str, f64, f64)],
    colliders: &[[DVec3; 3]],
    textures: &[TextureSlot<'_>],
) -> Result<Vec<Particle>> {
    simulate_with_lifecycle(
        mesh,
        settings,
        world,
        frame,
        fps,
        gravity,
        seed,
        force_fields,
        colliders,
        true,
        textures,
    )
}

fn simulate_with_lifecycle(
    mesh: &Mesh,
    settings: &Value,
    world: DMat4,
    frame: f64,
    fps: f64,
    gravity: DVec3,
    seed: u32,
    force_fields: &[(DVec3, &str, f64, f64)],
    colliders: &[[DVec3; 3]],
    include_lifecycle: bool,
    textures: &[TextureSlot<'_>],
) -> Result<Vec<Particle>> {
    if !frame.is_finite() || frame < 0.0 {
        return Err(invalid("particle frame must be finite and non-negative"));
    }
    if !fps.is_finite() || fps <= 0.0 {
        return Err(invalid("particle frame rate must be finite and positive"));
    }
    if !world.is_finite() || !gravity.is_finite() {
        return Err(invalid("particle transform and gravity must be finite"));
    }

    if force_fields.len() > MAX_FORCE_FIELDS {
        return Err(limit(
            "particle simulation exceeds the 1024 force-field limit",
        ));
    }
    let fields = force_fields
        .iter()
        .map(|(position, kind, strength, falloff)| {
            if !position.is_finite()
                || !strength.is_finite()
                || !falloff.is_finite()
                || *falloff < 0.0
            {
                return Err(invalid(
                    "particle force fields must have finite values and non-negative falloff",
                ));
            }
            let kind = match kind.to_ascii_lowercase().as_str() {
                "wind" => FieldKind::Wind,
                "vortex" => FieldKind::Vortex,
                "force" => FieldKind::Force,
                _ => {
                    return Err(invalid(
                        "particle force-field type must be wind, vortex, or force",
                    ));
                }
            };
            Ok(ForceField {
                position: *position,
                kind,
                strength: *strength,
                falloff: *falloff,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if colliders.len() > MAX_COLLIDERS {
        return Err(limit(
            "particle simulation exceeds the 100000 collider-triangle limit",
        ));
    }
    let mut world_colliders = Vec::with_capacity(colliders.len());
    for points in colliders {
        if points.iter().any(|point| !point.is_finite()) {
            return Err(invalid("particle collider triangle must be finite"));
        }
        let cross = (points[1] - points[0]).cross(points[2] - points[0]);
        let length = cross.length();
        if !length.is_finite() {
            return Err(invalid(
                "particle collider triangle has a non-finite normal",
            ));
        }
        if length > f64::EPSILON {
            world_colliders.push(ColliderTriangle {
                points: *points,
                normal: cross / length,
            });
        }
    }

    let lifetime = setting_alias_number(settings, &["lifetime"], 50.0)?;
    let lifetime_random = setting_alias_number(settings, &["lifetime_random"], 0.0)?;
    let is_hair = settings.get("type").and_then(Value::as_str) == Some("HAIR");
    let path_interpolation = if let Some(value) = settings.get("_potter_path_interpolation") {
        serde_json::from_value(value.clone())
            .map_err(|_| invalid("particle path interpolation mode is invalid"))?
    } else if is_hair {
        if settings.get("use_hair_bspline").and_then(Value::as_bool) == Some(true) {
            ParticlePathInterpolation::HairBspline
        } else {
            ParticlePathInterpolation::HairCardinal
        }
    } else {
        ParticlePathInterpolation::Hermite
    };
    let maximum_lifetime = if is_hair { 100.0 } else { lifetime };
    let speed = setting_alias_number(
        settings,
        &["speed", "normal_velocity", "normal_factor"],
        1.0,
    )?;
    let random_velocity = setting_velocity(settings)?;
    let initial_velocity = setting_vector(settings, "velocity")?;
    if lifetime < 0.0
        || !(0.0..=1.0).contains(&lifetime_random)
        || speed < 0.0
        || random_velocity.min_element() < 0.0
        || !random_velocity.is_finite()
    {
        return Err(invalid(
            "particle lifetime, speed, and random settings must be non-negative and finite",
        ));
    }
    let include_unborn = is_hair
        || (include_lifecycle
            && settings
                .get("emit_unborn")
                .and_then(Value::as_bool)
                .unwrap_or(false));
    if settings.get("rate").is_some() || settings.get("emission_rate").is_some() {
        return Err(unsupported(
            "physics.particle_emitter.rate",
            "Blender has no emission rate: particle i is born at frame_start + (frame_end - frame_start) * i / count",
        ));
    }
    // Blender's `ParticleSettings` defaults: `sta` 1, `end` 200 (`DNA_particle_types.h:600`), `totpart` 1000 (`:603`).
    let frame_start = setting_alias_number(settings, &["frame_start", "start_frame"], 1.0)?;
    let count = setting_alias_number(settings, &["count"], 1000.0)?;
    let frame_end = setting_alias_number(settings, &["frame_end", "end_frame"], 200.0)?;
    if !frame_start.is_finite()
        || !frame_end.is_finite()
        || frame_end < frame_start
        || count < 0.0
        || count.fract() != 0.0
    {
        return Err(invalid("particle emission frame range is invalid"));
    }
    let frame_span = frame_end - frame_start;
    let source_value = settings.get("source").or_else(|| settings.get("emit_from"));
    let source = source_value
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| invalid("particle source must be a string"))
        })
        .transpose()?
        .unwrap_or("FACE");
    let source = source.to_ascii_uppercase();
    let from_vertex = matches!(source.as_str(), "VERTEX" | "VERTICES" | "VERT");
    let from_volume = source == "VOLUME";
    if !from_vertex && !from_volume && !matches!(source.as_str(), "FACE" | "FACES") {
        return Err(invalid("particle emit_from must be VERT, FACE, or VOLUME"));
    }
    if from_volume && mesh.faces.is_empty() {
        return Err(unsupported(
            "physics.particle_emitter.emit_from_volume",
            "volume particle emission requires a closed mesh with faces",
        ));
    }
    let emitter_distribution = if from_vertex {
        EmitterDistribution::Jitter
    } else {
        parse_emitter_distribution(settings, is_hair)?
    };
    // A grid holds `grid_resolution` cubed particles in place of the count (`psys_resize` and
    // `tot_particles` in `particle_system.cc`), so the schedule and the birth indices use that total.
    let grid_resolution = if emitter_distribution == EmitterDistribution::Grid {
        Some(grid_resolution_setting(settings)?)
    } else {
        None
    };
    let count = grid_resolution.map_or(count, |resolution| {
        (resolution * resolution * resolution) as f64
    });

    let seed = settings
        .get("seed")
        .map(|value| {
            value
                .as_u64()
                .ok_or_else(|| invalid("particle seed must be a non-negative integer"))
        })
        .transpose()?
        .unwrap_or(u64::from(seed));
    let system_seed = u32::try_from(seed)
        .map_err(|_| invalid("particle seed must fit in a 32-bit unsigned integer"))?;
    let requested_frame = if include_unborn {
        frame.max(frame_end)
    } else {
        frame
    };
    let emission_frame = requested_frame.min(frame_end);
    let duration = (emission_frame - frame_start).max(0.0);
    // Particle i is born at frame_start + frame_span * i / count (`init_particle`, `pa->time`).
    let emitted_float = if emission_frame < frame_start {
        0.0
    } else if frame_span == 0.0 {
        count
    } else {
        ((duration * count / frame_span).floor() + 1.0).min(count)
    };
    if !emitted_float.is_finite() || emitted_float >= u64::MAX as f64 {
        return Err(limit("particle emission count exceeds the supported range"));
    }
    // Texture influences that move births or remove particles act on every particle in order: Blender
    // removes unexisting particles and renumbers the rest, so the schedule shortcuts do not apply.
    let time_mapped = !is_hair && maps_time(textures);
    // Grid cells that `distribute_grid` removes leave no particle, so the survivors renumber as they do
    // for density, and the schedule visits every particle.
    let all_events = time_mapped || maps_density(textures) || grid_resolution.is_some();
    let emitted = if all_events {
        count.floor() as u64
    } else {
        emitted_float.floor() as u64
    };
    if emitted == 0 || mesh.vertices.is_empty() {
        return Ok(Vec::new());
    }
    let (triangles, vertex_normals) = prepare_geometry(mesh)?;
    let inverse_transpose = world.inverse().transpose();
    let normal_transform = if inverse_transpose.is_finite() {
        inverse_transpose
    } else {
        world
    };
    let first_event = if include_lifecycle || all_events {
        0
    } else {
        // Births before `frame - maximum_lifetime` are dead by `frame`; keep one earlier event as margin.
        let births_before = if frame_span > 0.0 {
            ((frame - maximum_lifetime - frame_start).max(0.0) * count / frame_span).ceil()
        } else if frame - maximum_lifetime > frame_start {
            count
        } else {
            0.0
        };
        if !births_before.is_finite() || births_before >= u64::MAX as f64 {
            return Err(limit("particle emission count exceeds the supported range"));
        }
        (births_before as u64).saturating_sub(1).min(emitted)
    };
    let output_count = emitted.saturating_sub(first_event);
    if output_count > MAX_PARTICLES {
        return Err(limit(
            "particle output exceeds the one-million particle limit",
        ));
    }
    let child_settings = if include_lifecycle {
        parse_child_settings(settings, system_seed)?
    } else {
        None
    };
    let output_len = usize::try_from(output_count)
        .map_err(|_| limit("particle output exceeds supported limits"))?;
    let child_output_count = child_settings
        .map(|child| {
            output_len
                .checked_mul(child.count)
                .ok_or_else(|| limit("child particle count exceeds limits"))
        })
        .transpose()?
        .unwrap_or(0);
    if output_len
        .checked_add(child_output_count)
        .is_none_or(|count| count > MAX_PARTICLES as usize)
    {
        return Err(limit(
            "particle output exceeds the one-million particle limit",
        ));
    }
    // Blender's `ParticleSettings.size` default is 0.05 (`DNA_particle_types.h:613`).
    let particle_size = setting_alias_number(settings, &["particle_size", "size"], 0.05)?;
    let size_random = setting_alias_number(settings, &["size_random", "randsize"], 0.0)?;
    if particle_size < 0.0 || !(0.0..=1.0).contains(&size_random) {
        return Err(invalid(
            "particle size settings are outside Blender's supported range",
        ));
    }
    let step_work_factor = fields
        .len()
        .checked_add(world_colliders.len())
        .and_then(|work| work.checked_add(1))
        .ok_or_else(|| limit("particle simulation work exceeds limits"))?;
    let mut estimated_work = 0_usize;
    let path_values = settings
        .get("_potter_path_keys")
        .map(|value| {
            value
                .as_array()
                .ok_or_else(|| invalid("particle path keys must be an array"))
        })
        .transpose()?;
    let hair_current_states = if is_hair {
        settings
            .get("_potter_hair_current_states")
            .map(|value| {
                value
                    .as_array()
                    .ok_or_else(|| invalid("Hair current states must be an array"))
            })
            .transpose()?
    } else {
        None
    };
    if let Some(states) = hair_current_states {
        let required = usize::try_from(emitted)
            .map_err(|_| limit("Hair current state count exceeds supported limits"))?;
        if states.len() < required {
            return Err(invalid(
                "Hair current states do not cover every parent particle",
            ));
        }
    }
    let path_key_work = if let Some(paths) = path_values {
        let path_end = usize::try_from(emitted)
            .map_err(|_| limit("particle path count exceeds supported limits"))?;
        if paths.len() < path_end {
            return Err(invalid(
                "particle path keys do not cover every parent particle",
            ));
        }
        let path_start = usize::try_from(first_event)
            .map_err(|_| limit("particle path count exceeds supported limits"))?;
        paths[path_start..path_end]
            .iter()
            .try_fold(0_usize, |total, value| {
                let keys = value
                    .as_array()
                    .ok_or_else(|| invalid("particle path must be an array of path keys"))?;
                if keys.len() > MAX_STEPS_PER_PARTICLE + 1 {
                    return Err(limit("particle path exceeds the 4096-key limit"));
                }
                total
                    .checked_add(keys.len())
                    .ok_or_else(|| limit("particle simulation work exceeds limits"))
            })?
    } else {
        0
    };
    let child_path_work = path_key_work
        .checked_mul(child_settings.map_or(0, |child| child.count))
        .ok_or_else(|| limit("particle simulation work exceeds limits"))?;
    let child_mapping_work = if child_settings.is_some_and(|child| child.interpolated) {
        output_len
            .checked_mul(child_output_count)
            .ok_or_else(|| limit("particle simulation work exceeds limits"))?
    } else {
        0
    };
    for event in first_event..emitted {
        let birth_frame = if is_hair {
            0.0
        } else {
            frame_start + frame_span * event as f64 / count
        };
        let raw_age = frame - birth_frame;
        if raw_age < 0.0 {
            continue;
        }
        let age_frames = raw_age.min(maximum_lifetime);
        let has_path = path_values
            .and_then(|paths| {
                usize::try_from(event)
                    .ok()
                    .and_then(|index| paths.get(index))
            })
            .and_then(Value::as_array)
            .is_some_and(|keys| !keys.is_empty());
        let steps = if has_path { 0.0 } else { age_frames.ceil() };
        if !steps.is_finite() || steps > MAX_STEPS_PER_PARTICLE as f64 {
            return Err(limit("particle lifetime exceeds the 4096 fixed-step limit"));
        }
        let total_steps = steps as usize;
        estimated_work = estimated_work
            .checked_add(total_steps.saturating_mul(step_work_factor))
            .ok_or_else(|| limit("particle simulation work exceeds limits"))?;
        if estimated_work > MAX_SIMULATION_WORK {
            return Err(limit(
                "particle simulation exceeds the 50-million work limit",
            ));
        }
    }
    estimated_work = estimated_work
        .checked_add(path_key_work)
        .and_then(|work| work.checked_add(child_path_work))
        .and_then(|work| work.checked_add(child_mapping_work))
        .ok_or_else(|| limit("particle simulation work exceeds limits"))?;
    if estimated_work > MAX_SIMULATION_WORK {
        return Err(limit(
            "particle simulation exceeds the 50-million work limit",
        ));
    }

    let mut output = Vec::with_capacity(output_len);
    let total_area = triangles
        .last()
        .map_or(0.0, |triangle| triangle.cumulative_area);
    let mut texture_rng = BlenderRng::seed_random(system_seed);
    // Blender's `distribute_particles`: particle `p` is emitted from vertex `vertex_elements[p]`, or
    // from its face or volume sample `face_samples[p]`.
    let use_emit_random = settings
        .get("use_emit_random")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let vertex_elements = if from_vertex {
        particle_distribution::vertex_elements(
            mesh.vertices.len(),
            count as usize,
            system_seed,
            use_emit_random,
        )
    } else {
        Vec::new()
    };
    let face_samples = if from_vertex {
        Vec::new()
    } else {
        face_emitter_samples(
            mesh,
            settings,
            count,
            system_seed,
            from_volume,
            emitter_distribution,
        )?
    };
    // Index of the next particle that exists; Blender renumbers particles after removing the others.
    let mut survivors: u32 = 0;
    for event in first_event..emitted {
        let particle_index =
            u32::try_from(event).map_err(|_| limit("particle index exceeds supported range"))?;
        // Path keys are exported per surviving particle (after `free_unexisting_particles`).
        let path = if let Some(paths) = path_values {
            particle_path(paths, u64::from(survivors))?
        } else {
            Vec::new()
        };
        let path_root_rotation = path
            .first()
            .and_then(|birth| emitter_root_frame(birth.position, world, &triangles))
            .map(|(rotation, _)| rotation);
        // Path keys give the birth point; without them Blender's emitter sample does.
        let sampled = if path.is_empty() {
            let mut random = Random::new(seed ^ event.wrapping_mul(0xd6e8_feb8_6659_fd93));
            let (local_position, local_normal) = if from_vertex {
                let particle = usize::try_from(event)
                    .map_err(|_| limit("particle index exceeds supported range"))?;
                let index = vertex_elements[particle];
                (
                    mesh.vertices[index].co,
                    vertex_normals[index].normalize_or_zero(),
                )
            } else {
                // Blender emits nothing when no element has weight, so a missing sample skips the particle.
                let Some(sample) = usize::try_from(event)
                    .ok()
                    .and_then(|particle| face_samples.get(particle).copied().flatten())
                else {
                    continue;
                };
                (sample.position, sample.normal)
            };
            let normal = normal_transform
                .transform_vector3(local_normal)
                .normalize_or_zero();
            // Blender's grid normal stays zero, so a grid birth adds no normal velocity; other emitters
            // without a normal point along +Z.
            let direction = if normal.length_squared() > 0.0 {
                normal
            } else if grid_resolution.is_some() {
                DVec3::ZERO
            } else {
                DVec3::Z
            };
            let local_random = DVec3::new(
                random.signed() * random_velocity.x,
                random.signed() * random_velocity.y,
                random.signed() * random_velocity.z,
            );
            let start = world.transform_point3(local_position);
            let velocity =
                direction * speed + world.transform_vector3(initial_velocity + local_random);
            if !start.is_finite() || !velocity.is_finite() {
                return Err(invalid("particle emission produced a non-finite value"));
            }
            let birth_rotation = if !from_vertex && !from_volume {
                emitter_root_frame(start.to_array(), world, &triangles).map_or_else(
                    || {
                        if normal.length_squared() > f64::EPSILON {
                            DQuat::from_rotation_arc(DVec3::X, normal.normalize()).to_array()
                        } else {
                            DQuat::IDENTITY.to_array()
                        }
                    },
                    |(rotation, _)| rotation,
                )
            } else if normal.length_squared() > f64::EPSILON {
                DQuat::from_rotation_arc(DVec3::X, normal.normalize()).to_array()
            } else {
                DQuat::IDENTITY.to_array()
            };
            Some((start, velocity, normal, birth_rotation))
        } else {
            None
        };
        let birth_point = match (path.first(), &sampled) {
            (Some(first), _) => DVec3::from_array(first.position),
            (None, Some((start, ..))) => *start,
            (None, None) => return Err(invalid("particle emission sample is missing")),
        };
        // `init_particle`: the default emission time is the particle's index over the count.
        let default_time = if count > 0.0 {
            event as f64 / count
        } else {
            0.0
        };
        let values = birth_values(textures, birth_point, default_time, &mut texture_rng)?;
        let birth_frame = if is_hair {
            0.0
        } else {
            frame_start + frame_span * values.time
        };
        // Blender's `PARS_UNEXIST` (`init_particle_texture`): `free_unexisting_particles` removes those
        // particles, so the survivors' index drives the later per-particle random draws.
        let exists = !(values.density
            < f64::from(psys_frand(system_seed, particle_index.wrapping_add(125))));
        if !exists {
            continue;
        }
        let survivor = survivors;
        survivors = survivors.wrapping_add(1);
        let actual_lifetime = if is_hair {
            100.0
        } else {
            lifetime
                * values.life
                * (1.0
                    - lifetime_random
                        * f64::from(psys_frand(system_seed, survivor.wrapping_add(21))))
        };
        let raw_age = frame - birth_frame;
        let unborn = raw_age < 0.0;
        if unborn && !include_lifecycle {
            continue;
        }
        let life_state = if unborn {
            ParticleLifeState::Unborn
        } else if raw_age >= actual_lifetime {
            ParticleLifeState::Dead
        } else {
            ParticleLifeState::Alive
        };
        let age_frames = match life_state {
            ParticleLifeState::Unborn => 0.0,
            ParticleLifeState::Alive => raw_age.max(0.0).min(actual_lifetime),
            ParticleLifeState::Dead => actual_lifetime,
        };
        if life_state != ParticleLifeState::Alive && !include_lifecycle {
            continue;
        }
        let path_time = if is_hair {
            0.0
        } else if actual_lifetime <= f64::EPSILON {
            if life_state == ParticleLifeState::Unborn {
                0.0
            } else {
                1.0
            }
        } else {
            (age_frames / actual_lifetime).clamp(0.0, 1.0)
        };
        let current_path = sample_path(&path, path_time, path_interpolation);
        let particle_gravity = if settings.get("physics_type").and_then(Value::as_str) == Some("NO")
        {
            DVec3::ZERO
        } else {
            gravity
        };
        let (start, velocity, normal, birth_rotation, mut position, mut final_velocity) =
            if let (Some(birth), Some(current)) = (path.first(), current_path) {
                let normal = (DQuat::from_array(birth.rotation) * DVec3::X).normalize_or_zero();
                let birth_rotation = birth.rotation;
                (
                    DVec3::from_array(birth.position),
                    DVec3::from_array(birth.velocity),
                    normal,
                    birth_rotation,
                    DVec3::from_array(current.position),
                    DVec3::from_array(current.velocity),
                )
            } else {
                let (start, velocity, normal, birth_rotation) =
                    sampled.ok_or_else(|| invalid("particle emission sample is missing"))?;
                let (position, final_velocity) = integrate_particle(
                    start,
                    velocity,
                    age_frames,
                    fps,
                    particle_gravity,
                    &fields,
                    &world_colliders,
                )?;
                (
                    start,
                    velocity,
                    normal,
                    birth_rotation,
                    position,
                    final_velocity,
                )
            };
        if let Some(states) = hair_current_states {
            let index = usize::try_from(event)
                .map_err(|_| limit("Hair current state index exceeds supported limits"))?;
            let current = states
                .get(index)
                .ok_or_else(|| invalid("Hair current states do not cover every parent particle"))?;
            position = DVec3::from_array(path_vector3(
                current.get("position"),
                "Hair current position",
            )?);
            final_velocity = DVec3::from_array(path_vector3(
                current.get("velocity"),
                "Hair current velocity",
            )?);
        }
        let size = particle_size
            * values.size
            * (1.0 - size_random * f64::from(psys_frand(system_seed, survivor.wrapping_add(1))));
        let mass = setting_alias_number(settings, &["mass"], 1.0)?;
        if mass <= 0.0 {
            return Err(invalid("particle mass must be positive"));
        }
        output.push(Particle {
            position: position.to_array(),
            birth_position: start.to_array(),
            normal: normal.to_array(),
            birth_rotation,
            current_path_rotation: current_path.map(|key| key.rotation),
            path_root_rotation,
            path_interpolation,
            velocity: final_velocity.to_array(),
            birth_frame,
            death_frame: birth_frame + actual_lifetime,
            life_state,
            is_child: false,
            parent_index: None,
            parent_indices: [0; 3],
            parent_weights: [0.0; 3],
            parent_count: 0,
            path,
            size,
            mass,
        });
    }
    if include_lifecycle {
        append_children(
            child_settings,
            frame,
            world,
            &triangles,
            total_area,
            &mut output,
        )?;
        if let Some(child_paths) = settings.get("_potter_child_path_keys") {
            let child_paths = child_paths
                .as_array()
                .ok_or_else(|| invalid("particle child path keys must be an array"))?;
            let child_start = output
                .iter()
                .position(|particle| particle.is_child)
                .unwrap_or(output.len());
            if child_paths.len() != output.len() - child_start {
                return Err(invalid(
                    "particle child path keys do not match generated child order",
                ));
            }
            let mut path_key_count = 0_usize;
            for path in child_paths {
                let keys = path
                    .as_array()
                    .ok_or_else(|| invalid("particle child path must be an array of path keys"))?;
                if keys.len() > MAX_STEPS_PER_PARTICLE + 1 {
                    return Err(limit("particle child path exceeds the 4096-key limit"));
                }
                path_key_count = path_key_count
                    .checked_add(keys.len())
                    .filter(|count| *count <= MAX_SIMULATION_WORK)
                    .ok_or_else(|| {
                        limit("particle child paths exceed the simulation work limit")
                    })?;
            }
            estimated_work = estimated_work
                .checked_add(path_key_count)
                .filter(|work| *work <= MAX_SIMULATION_WORK)
                .ok_or_else(|| limit("particle simulation exceeds the 50-million work limit"))?;
            let (parents, children) = output.split_at_mut(child_start);
            for (index, child) in children.iter_mut().enumerate() {
                let particle_index = u64::try_from(index)
                    .map_err(|_| limit("particle child path index exceeds supported limits"))?;
                let path = particle_path(child_paths, particle_index)?;
                if !path.is_empty() {
                    child.path = path;
                    child.path_interpolation = if is_hair {
                        ParticlePathInterpolation::HairCache
                    } else {
                        ParticlePathInterpolation::Linear
                    };
                    child.path_root_rotation = child
                        .parent_index
                        .and_then(|index| usize::try_from(index).ok())
                        .and_then(|index| parents.get(index))
                        .and_then(|parent| parent.path_root_rotation)
                        .or_else(|| {
                            child.path.first().and_then(|key| {
                                emitter_root_frame(key.position, world, &triangles)
                                    .map(|(rotation, _)| rotation)
                            })
                        });
                    child.current_path_rotation = sample_path(
                        &child.path,
                        particle_path_time(frame, child.birth_frame, child.death_frame),
                        child.path_interpolation,
                    )
                    .map(|key| key.rotation);
                }
            }
        }
    }
    Ok(output)
}

fn integrate_particle(
    mut position: DVec3,
    mut velocity: DVec3,
    age_frames: f64,
    fps: f64,
    gravity: DVec3,
    fields: &[ForceField],
    colliders: &[ColliderTriangle],
) -> Result<(DVec3, DVec3)> {
    let whole_steps = age_frames.floor() as usize;
    let fixed_dt = 1.0 / fps;
    for _ in 0..whole_steps {
        (position, velocity) =
            integrate_step(position, velocity, fixed_dt, gravity, fields, colliders)?;
    }
    let remainder = (age_frames - whole_steps as f64) / fps;
    if remainder > 0.0 {
        (position, velocity) =
            integrate_step(position, velocity, remainder, gravity, fields, colliders)?;
    }
    if !position.is_finite() || !velocity.is_finite() {
        return Err(invalid("particle simulation produced a non-finite value"));
    }
    Ok((position, velocity))
}

fn integrate_step(
    position: DVec3,
    velocity: DVec3,
    dt: f64,
    gravity: DVec3,
    fields: &[ForceField],
    colliders: &[ColliderTriangle],
) -> Result<(DVec3, DVec3)> {
    let acceleration = gravity + field_acceleration(position, fields);
    if !acceleration.is_finite() {
        return Err(invalid(
            "particle force evaluation produced a non-finite value",
        ));
    }
    let mut next_position = position + velocity * dt + acceleration * (0.5 * dt * dt);
    let mut next_velocity = velocity + acceleration * dt;
    if !next_position.is_finite() || !next_velocity.is_finite() {
        return Err(invalid("particle integration produced a non-finite value"));
    }

    let mut segment_start = position;
    for collider in colliders {
        if let Some((hit, normal)) =
            collide_segment(segment_start, next_position, next_velocity, *collider)
        {
            next_position = hit + normal * 1.0e-9;
            let normal_speed = next_velocity.dot(normal);
            if normal_speed < 0.0 {
                next_velocity -= normal * (1.5 * normal_speed);
            }
            segment_start = next_position;
        }
    }
    Ok((next_position, next_velocity))
}

fn field_acceleration(position: DVec3, fields: &[ForceField]) -> DVec3 {
    let mut acceleration = DVec3::ZERO;
    for field in fields {
        let offset = position - field.position;
        let distance = offset.length();
        let influence = if field.falloff == 0.0 {
            1.0
        } else {
            (1.0 - distance / field.falloff).clamp(0.0, 1.0)
        };
        if influence == 0.0 {
            continue;
        }
        let direction = match field.kind {
            FieldKind::Wind => DVec3::Z,
            FieldKind::Vortex => DVec3::Z.cross(offset).normalize_or_zero(),
            FieldKind::Force => offset.normalize_or_zero(),
        };
        acceleration += direction * (field.strength * influence);
    }
    acceleration
}

fn collide_segment(
    start: DVec3,
    end: DVec3,
    velocity: DVec3,
    triangle: ColliderTriangle,
) -> Option<(DVec3, DVec3)> {
    let [a, b, c] = triangle.points;
    let start_distance = (start - a).dot(triangle.normal);
    let end_distance = (end - a).dot(triangle.normal);
    if (start_distance > 1.0e-12 && end_distance > 1.0e-12)
        || (start_distance < -1.0e-12 && end_distance < -1.0e-12)
        || (start_distance.abs() <= 1.0e-12
            && end_distance >= 0.0
            && velocity.dot(triangle.normal) >= 0.0)
    {
        return None;
    }
    let denominator = start_distance - end_distance;
    if denominator.abs() <= f64::EPSILON {
        return None;
    }
    let fraction = start_distance / denominator;
    if !(0.0..=1.0).contains(&fraction) {
        return None;
    }
    let hit = start + (end - start) * fraction;
    if !crate::geom::point_in_triangle(hit, a, b, c) {
        return None;
    }
    let normal = if start_distance >= 0.0 {
        triangle.normal
    } else {
        -triangle.normal
    };
    Some((hit, normal))
}

fn prepare_geometry(mesh: &Mesh) -> Result<(Vec<Triangle>, Vec<DVec3>)> {
    if mesh.vertices.len() > MAX_EMITTER_VERTICES {
        return Err(limit(
            "particle emitter exceeds the one-million vertex limit",
        ));
    }
    if mesh.faces.len() > MAX_EMITTER_FACES {
        return Err(limit("particle emitter exceeds the one-million face limit"));
    }
    let triangle_count = mesh
        .faces
        .iter()
        .try_fold(0_usize, |total, face| {
            total.checked_add(face.vertices.len().saturating_sub(2))
        })
        .ok_or_else(|| limit("particle emitter triangle count exceeds limits"))?;
    if triangle_count > MAX_EMITTER_TRIANGLES {
        return Err(limit(
            "particle emitter exceeds the two-million triangle limit",
        ));
    }
    let mut vertex_indices = HashMap::with_capacity(mesh.vertices.len());
    for (index, vertex) in mesh.vertices.iter().enumerate() {
        if !vertex.co.is_finite() {
            return Err(invalid("particle emitter contains a non-finite vertex"));
        }
        if vertex_indices.insert(vertex.id, index).is_some() {
            return Err(invalid("particle emitter contains duplicate vertex IDs"));
        }
    }

    let mut triangles = Vec::with_capacity(triangle_count);
    let mut vertex_normals = vec![DVec3::ZERO; mesh.vertices.len()];
    let mut cumulative_area = 0.0_f64;
    let mut triangle_error = None;
    mesh.for_each_triangle_with_face(|source_face, triangle| {
        if triangle_error.is_some() {
            return;
        }
        let indices = triangle.map(|id| vertex_indices.get(&id).copied());
        let [Some(first), Some(second), Some(third)] = indices else {
            triangle_error = Some(invalid(
                "particle emitter triangle references a missing vertex",
            ));
            return;
        };
        let a = mesh.vertices[first].co;
        let b = mesh.vertices[second].co;
        let c = mesh.vertices[third].co;
        let area_normal = (b - a).cross(c - a) * 0.5;
        let area = area_normal.length();
        if !area.is_finite() {
            triangle_error = Some(invalid("particle emitter face area is non-finite"));
            return;
        }
        if area <= f64::EPSILON {
            return;
        }
        let normal = area_normal / area;
        // Blender's hair frame uses the original quad tessface order, not its ear-clipped simulation triangle.
        let (frame_tangent, frame_normal) = if mesh.faces[source_face].vertices.len() == 4 {
            let face = &mesh.faces[source_face];
            let first = mesh.vertices[vertex_indices[&face.vertices[0]]].co;
            let second = mesh.vertices[vertex_indices[&face.vertices[1]]].co;
            let third = mesh.vertices[vertex_indices[&face.vertices[2]]].co;
            let tangent = (second - first).normalize_or_zero();
            let face_normal = (second - first).cross(third - first).normalize_or_zero();
            if tangent.length_squared() > f64::EPSILON
                && face_normal.length_squared() > f64::EPSILON
            {
                (tangent, face_normal)
            } else {
                ((b - a).normalize_or_zero(), normal)
            }
        } else {
            ((b - a).normalize_or_zero(), normal)
        };
        cumulative_area += area;
        if !cumulative_area.is_finite() {
            triangle_error = Some(invalid("particle emitter total face area is non-finite"));
            return;
        }
        triangles.push(Triangle {
            points: [a, b, c],
            normal,
            frame_tangent,
            frame_normal,
            cumulative_area,
        });
        vertex_normals[first] += area_normal;
        vertex_normals[second] += area_normal;
        vertex_normals[third] += area_normal;
    })
    .map_err(|error| invalid(&format!("particle emitter mesh is invalid: {error}")))?;
    if let Some(error) = triangle_error {
        return Err(error);
    }
    for normal in &mut vertex_normals {
        if !normal.is_finite() {
            return Err(invalid("particle emitter vertex normal is non-finite"));
        }
        *normal = normal.normalize_or_zero();
    }
    Ok((triangles, vertex_normals))
}

/// The distribution of a face or volume emitter (`ParticleSettings.distribution`). Hair has no grid,
/// so Blender sets a hair grid to JIT (`particle_system.cc`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum EmitterDistribution {
    Jitter,
    Random,
    Grid,
}

fn parse_emitter_distribution(settings: &Value, is_hair: bool) -> Result<EmitterDistribution> {
    let distribution = match settings
        .get("distribution")
        .and_then(Value::as_str)
        .unwrap_or("JIT")
    {
        "JIT" => EmitterDistribution::Jitter,
        "RAND" => EmitterDistribution::Random,
        "GRID" => EmitterDistribution::Grid,
        _ => return Err(invalid("particle distribution must be JIT, RAND, or GRID")),
    };
    if is_hair && distribution == EmitterDistribution::Grid {
        return Ok(EmitterDistribution::Jitter);
    }
    Ok(distribution)
}

/// `grid_resolution` (`ParticleSettings.grid_resolution`): cells along each axis, within Blender's
/// range of 1 to 250. The grid's particle count is its cube, which the one-million limit bounds.
fn grid_resolution_setting(settings: &Value) -> Result<usize> {
    let resolution = setting_alias_number(settings, &["grid_resolution"], 10.0)?;
    if resolution.fract() != 0.0 || !(1.0..=250.0).contains(&resolution) {
        return Err(invalid(
            "particle grid resolution must be an integer from 1 to 250",
        ));
    }
    Ok(resolution as usize)
}

/// Blender's `distribute_particles` for `PART_FROM_FACE` and `PART_FROM_VOLUME`: the birth sample of
/// each particle, or none where a grid cell is removed or no element can take a particle.
fn face_emitter_samples(
    mesh: &Mesh,
    settings: &Value,
    count: f64,
    system_seed: u32,
    volume: bool,
    distribution: EmitterDistribution,
) -> Result<Vec<Option<particle_distribution::FaceSample>>> {
    if count > MAX_PARTICLES as f64 {
        return Err(limit(
            "particle emission count exceeds the one-million particle limit",
        ));
    }
    let emitter = particle_distribution::FaceEmitter::new(mesh)?;
    if distribution == EmitterDistribution::Grid {
        let resolution = grid_resolution_setting(settings)?;
        // `distribute_grid` tests each cell row against every face: three rays for a face, one for a
        // volume.
        let rays: usize = if volume { 1 } else { 3 };
        let work = rays
            .checked_mul(resolution * resolution)
            .and_then(|work| work.checked_mul(emitter.element_count()));
        if work.is_none_or(|work| work > MAX_SIMULATION_WORK) {
            return Err(limit(
                "particle grid emission exceeds the 50-million work limit",
            ));
        }
        let random = setting_alias_number(settings, &["grid_random"], 0.0)?;
        if !(0.0..=1.0).contains(&random) {
            return Err(invalid("particle grid randomness must be from 0 to 1"));
        }
        let grid = particle_distribution::GridSettings {
            volume,
            resolution,
            random: random as f32,
            hexagonal: settings
                .get("hexagonal_grid")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            invert: settings
                .get("invert_grid")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            seed: system_seed,
        };
        return Ok(particle_distribution::distribute_grid(&emitter, grid));
    }
    let user_jitter = setting_alias_number(settings, &["userjit"], 0.0)?;
    if user_jitter.fract() != 0.0 || !(0.0..=1000.0).contains(&user_jitter) {
        return Err(invalid(
            "particle jitter count must be an integer from 0 to 1000",
        ));
    }
    let jitter_factor = setting_alias_number(settings, &["jitter_factor"], 1.0)?;
    if !(0.0..=2.0).contains(&jitter_factor) {
        return Err(invalid("particle jitter factor must be from 0 to 2"));
    }
    let count = count as usize;
    if volume
        && count
            .checked_mul(emitter.element_count())
            .is_none_or(|work| work > MAX_SIMULATION_WORK)
    {
        return Err(limit(
            "particle volume emission exceeds the 50-million work limit",
        ));
    }
    let settings_for_distribution = particle_distribution::FaceSettings {
        volume,
        jitter: distribution == EmitterDistribution::Jitter,
        even: settings
            .get("use_even_distribution")
            .and_then(Value::as_bool)
            .unwrap_or(true),
        random: settings
            .get("use_emit_random")
            .and_then(Value::as_bool)
            .unwrap_or(true),
        jitter_factor: jitter_factor as f32,
        user_jitter: user_jitter as usize,
        seed: system_seed,
        count,
    };
    Ok(particle_distribution::distribute_faces(
        &emitter,
        settings_for_distribution,
    ))
}

fn path_vector3(value: Option<&Value>, name: &str) -> Result<[f64; 3]> {
    let values = value
        .and_then(Value::as_array)
        .filter(|values| values.len() == 3)
        .ok_or_else(|| invalid(&format!("particle path {name} must contain three numbers")))?;
    let [x, y, z] = values.as_slice() else {
        return Err(invalid(&format!(
            "particle path {name} must contain three numbers"
        )));
    };
    let [x, y, z] = [x, y, z].map(|value| {
        value
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(|| invalid(&format!("particle path {name} must contain finite numbers")))
    });
    Ok([x?, y?, z?])
}

fn path_vector4(value: Option<&Value>) -> Result<[f64; 4]> {
    let values = value
        .and_then(Value::as_array)
        .filter(|values| values.len() == 4)
        .ok_or_else(|| invalid("particle path rotation must contain four numbers"))?;
    let [x, y, z, w] = values.as_slice() else {
        return Err(invalid("particle path rotation must contain four numbers"));
    };
    let [x, y, z, w] = [x, y, z, w].map(|value| {
        value
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(|| invalid("particle path rotation must contain finite numbers"))
    });
    Ok([x?, y?, z?, w?])
}

fn particle_path(paths: &[Value], particle_index: u64) -> Result<Vec<ParticlePathKey>> {
    let index = usize::try_from(particle_index)
        .map_err(|_| limit("particle path index exceeds supported limits"))?;
    let records = paths
        .get(index)
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("particle path keys do not cover every parent particle"))?;
    if records.len() > MAX_STEPS_PER_PARTICLE + 1 {
        return Err(limit("particle path exceeds the 4096-key limit"));
    }
    let has_velocity = records
        .iter()
        .all(|record| record.get("velocity").is_some());
    let has_rotation = records
        .iter()
        .all(|record| record.get("rotation").is_some());
    let root_rotation = records
        .first()
        .and_then(|record| record.get("rotation"))
        .map(|value| path_vector4(Some(value)))
        .transpose()?;
    let mut path = Vec::with_capacity(records.len());
    for record in records {
        let time = record
            .get("time")
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite() && (0.0..=1.0).contains(value))
            .ok_or_else(|| invalid("particle path key time must be normalized and finite"))?;
        let position = path_vector3(record.get("position"), "position")?;
        let velocity = if has_velocity {
            path_vector3(record.get("velocity"), "velocity")?
        } else {
            [0.0; 3]
        };
        let rotation = if has_rotation {
            path_vector4(record.get("rotation"))?
        } else {
            DQuat::IDENTITY.to_array()
        };
        path.push(ParticlePathKey {
            time,
            position,
            velocity,
            rotation,
        });
    }
    if path.windows(2).any(|pair| pair[0].time > pair[1].time) {
        return Err(invalid("particle path key times must be ordered"));
    }
    if !has_velocity && path.len() > 1 {
        for index in 0..path.len() {
            let tangent = if index == 0 {
                let interval = path[1].time - path[0].time;
                if interval > f64::EPSILON {
                    (DVec3::from_array(path[1].position) - DVec3::from_array(path[0].position))
                        / (2.0 * interval)
                } else {
                    DVec3::ZERO
                }
            } else if index + 1 == path.len() {
                let interval = path[index].time - path[index - 1].time;
                if interval > f64::EPSILON {
                    (DVec3::from_array(path[index].position)
                        - DVec3::from_array(path[index - 1].position))
                        / (2.0 * interval)
                } else {
                    DVec3::ZERO
                }
            } else {
                let interval = path[index + 1].time - path[index - 1].time;
                if interval > f64::EPSILON {
                    (DVec3::from_array(path[index + 1].position)
                        - DVec3::from_array(path[index - 1].position))
                        / interval
                } else {
                    DVec3::ZERO
                }
            };
            path[index].velocity = tangent.to_array();
        }
    }
    if !has_rotation && !path.is_empty() {
        let direction = DVec3::from_array(path[0].velocity).normalize_or_zero();
        let mut rotation = if let Some(rotation) = root_rotation {
            DQuat::from_array(rotation)
        } else if direction.length_squared() > f64::EPSILON {
            DQuat::from_rotation_arc(DVec3::X, direction)
        } else {
            DQuat::IDENTITY
        };
        path[0].rotation = rotation.to_array();
        let mut previous_direction = direction;
        for key in path.iter_mut().skip(1) {
            let direction = DVec3::from_array(key.velocity).normalize_or_zero();
            if previous_direction.length_squared() > f64::EPSILON
                && direction.length_squared() > f64::EPSILON
            {
                rotation = DQuat::from_rotation_arc(previous_direction, direction) * rotation;
            }
            key.rotation = rotation.to_array();
            previous_direction = direction;
        }
    }
    for key in &mut path {
        let rotation = DQuat::from_array(key.rotation);
        if !rotation.is_finite() || rotation.length_squared() <= 1.0e-20 {
            return Err(invalid("particle path rotation must be finite and nonzero"));
        }
        key.rotation = rotation.normalize().to_array();
    }
    Ok(path)
}

pub(super) fn sample_path(
    keys: &[ParticlePathKey],
    time: f64,
    interpolation: ParticlePathInterpolation,
) -> Option<ParticlePathKey> {
    let first = *keys.first()?;
    let time = time.clamp(0.0, 1.0);
    if keys.len() == 1 {
        return Some(first);
    }

    let upper = keys.partition_point(|key| key.time < time);
    if interpolation != ParticlePathInterpolation::HairBspline {
        if upper == 0 {
            return Some(first);
        }
        if upper == keys.len() {
            return keys.last().copied();
        }
    }
    let (segment, factor) = if upper == 0 {
        (0, 0.0)
    } else if upper == keys.len() {
        (keys.len() - 2, 1.0)
    } else {
        let segment = upper - 1;
        let duration = keys[segment + 1].time - keys[segment].time;
        if duration <= 0.0 {
            return Some(keys[segment]);
        }
        (
            segment,
            ((time - keys[segment].time) / duration).clamp(0.0, 1.0),
        )
    };
    let first = keys[segment];
    let second = keys[segment + 1];
    let duration = second.time - first.time;
    if duration <= 0.0 {
        return Some(first);
    }
    let rotation = DQuat::from_array(first.rotation)
        .slerp(DQuat::from_array(second.rotation), factor)
        .normalize();
    let (position, velocity) = if interpolation == ParticlePathInterpolation::HairBspline {
        let t2 = factor * factor;
        let t3 = t2 * factor;
        let one_minus_t = 1.0 - factor;
        let controls = [
            DVec3::from_array(keys[segment.saturating_sub(1)].position),
            DVec3::from_array(first.position),
            DVec3::from_array(second.position),
            DVec3::from_array(keys[(segment + 2).min(keys.len() - 1)].position),
        ];
        let weights = [
            one_minus_t.powi(3) / 6.0,
            (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
            (-3.0 * t3 + 3.0 * t2 + 3.0 * factor + 1.0) / 6.0,
            t3 / 6.0,
        ];
        let derivatives = [
            -0.5 * one_minus_t * one_minus_t,
            (9.0 * t2 - 12.0 * factor) / 6.0,
            (-9.0 * t2 + 6.0 * factor + 3.0) / 6.0,
            0.5 * t2,
        ];
        let position = controls
            .iter()
            .zip(weights)
            .fold(DVec3::ZERO, |sum, (control, weight)| {
                sum + *control * weight
            });
        let velocity = controls
            .iter()
            .zip(derivatives)
            .fold(DVec3::ZERO, |sum, (control, derivative)| {
                sum + *control * derivative
            })
            / duration;
        (position, velocity)
    } else if matches!(
        interpolation,
        ParticlePathInterpolation::Linear | ParticlePathInterpolation::HairCache
    ) {
        (
            DVec3::from_array(first.position).lerp(DVec3::from_array(second.position), factor),
            DVec3::from_array(first.velocity).lerp(DVec3::from_array(second.velocity), factor),
        )
    } else {
        let first_position = DVec3::from_array(first.position);
        let second_position = DVec3::from_array(second.position);
        let first_velocity = DVec3::from_array(first.velocity);
        let second_velocity = DVec3::from_array(second.velocity);
        let t2 = factor * factor;
        let t3 = t2 * factor;
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + factor;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;
        let position = first_position * h00
            + first_velocity * (duration * h10)
            + second_position * h01
            + second_velocity * (duration * h11);
        let velocity = first_position * ((6.0 * t2 - 6.0 * factor) / duration)
            + first_velocity * (3.0 * t2 - 4.0 * factor + 1.0)
            + second_position * ((-6.0 * t2 + 6.0 * factor) / duration)
            + second_velocity * (3.0 * t2 - 2.0 * factor);
        (position, velocity)
    };
    Some(ParticlePathKey {
        time,
        position: position.to_array(),
        velocity: velocity.to_array(),
        rotation: rotation.to_array(),
    })
}
fn emitter_root_frame(
    root_world: [f64; 3],
    world: DMat4,
    triangles: &[Triangle],
) -> Option<([f64; 4], [f64; 3])> {
    let first = *triangles.first()?;
    let inverse = world.inverse();
    if !inverse.is_finite() {
        return None;
    }
    let root = inverse.transform_point3(DVec3::from_array(root_world));
    let mut selected = first;
    let mut selected_distance = f64::INFINITY;
    for triangle in triangles {
        let [a, b, c] = triangle.points;
        let signed_distance = (root - a).dot(triangle.normal);
        let projected = root - triangle.normal * signed_distance;
        let distance = if crate::geom::point_in_triangle(projected, a, b, c) {
            signed_distance.abs()
        } else {
            (projected - (a + b + c) / 3.0).length()
        };
        if distance < selected_distance {
            selected = *triangle;
            selected_distance = distance;
        }
    }
    let tangent = world
        .transform_vector3(selected.frame_tangent)
        .normalize_or_zero();
    let normal = inverse
        .transpose()
        .transform_vector3(selected.frame_normal)
        .normalize_or_zero();
    if tangent.length_squared() <= f64::EPSILON || normal.length_squared() <= f64::EPSILON {
        return None;
    }
    let frame = DMat3::from_cols(tangent.cross(normal).normalize_or_zero(), tangent, normal);
    let rotation = DQuat::from_mat3(&frame).normalize();
    Some((rotation.to_array(), normal.to_array()))
}

fn parse_child_settings(settings: &Value, system_seed: u32) -> Result<Option<ChildSettings>> {
    let child_type = settings
        .get("child_type")
        .and_then(Value::as_str)
        .unwrap_or("NONE");
    let interpolated = match child_type {
        "NONE" => return Ok(None),
        "SIMPLE" => false,
        "INTERPOLATED" => true,
        _ => {
            return Err(invalid(
                "particle child_type must be NONE, SIMPLE, or INTERPOLATED",
            ));
        }
    };
    let count = settings
        .get("child_percent")
        .map(|value| {
            value
                .as_u64()
                .ok_or_else(|| invalid("child_percent must be a non-negative integer"))
        })
        .transpose()?
        .unwrap_or(10);
    if count == 0 {
        return Ok(None);
    }
    let count = usize::try_from(count).map_err(|_| limit("child particle count exceeds limits"))?;
    let radius = setting_alias_number(settings, &["child_radius", "childrad"], 0.2)?;
    let flat = setting_alias_number(settings, &["child_roundness", "childflat"], 0.0)?;
    let size = setting_alias_number(settings, &["child_size", "childsize"], 1.0)?;
    let size_random = setting_alias_number(settings, &["child_size_random", "childrandsize"], 0.0)?;
    let lifetime = setting_alias_number(settings, &["lifetime"], 50.0)?;
    let lifetime_random = setting_alias_number(settings, &["lifetime_random"], 0.0)?;
    let virtual_parents = setting_alias_number(settings, &["virtual_parents"], 0.0)?;
    if radius < 0.0
        || !(0.0..=1.0).contains(&flat)
        || size < 0.001
        || !(0.0..=1.0).contains(&size_random)
        || lifetime < 0.0
        || !(0.0..=1.0).contains(&lifetime_random)
        || !(0.0..=1.0).contains(&virtual_parents)
    {
        return Err(invalid(
            "child particle settings are outside Blender's supported range",
        ));
    }
    let child_seed = settings
        .get("child_seed")
        .map(|value| {
            value
                .as_u64()
                .ok_or_else(|| invalid("child_seed must be a non-negative integer"))
                .and_then(|value| {
                    u32::try_from(value)
                        .map_err(|_| invalid("child_seed must fit in a 32-bit unsigned integer"))
                })
        })
        .transpose()?
        .unwrap_or(0);
    Ok(Some(ChildSettings {
        count,
        radius,
        flat,
        size,
        size_random,
        child_seed,
        system_seed,
        lifetime,
        lifetime_random,
        virtual_parents,
        interpolated,
        use_emit_random: settings
            .get("use_emit_random")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    }))
}

fn append_children(
    settings: Option<ChildSettings>,
    frame: f64,
    world: DMat4,
    triangles: &[Triangle],
    total_area: f64,
    parents: &mut Vec<Particle>,
) -> Result<()> {
    let Some(settings) = settings else {
        return Ok(());
    };
    let parent_count = parents.len();
    let child_count = parent_count
        .checked_mul(settings.count)
        .ok_or_else(|| limit("child particle count exceeds limits"))?;
    if child_count == 0 {
        return Ok(());
    }
    if parent_count
        .checked_add(child_count)
        .is_none_or(|count| count > MAX_PARTICLES as usize)
    {
        return Err(limit(
            "particle output exceeds the one-million particle limit",
        ));
    }
    let mut children = Vec::with_capacity(child_count);
    if settings.interpolated {
        let inverse = world.inverse();
        let use_local_space = inverse.is_finite();
        let parent_positions = parents
            .iter()
            .map(|parent| {
                let root = parent
                    .path
                    .first()
                    .map_or(parent.birth_position, |key| key.position);
                let root = DVec3::from_array(root);
                if use_local_space {
                    inverse.transform_point3(root)
                } else {
                    root
                }
            })
            .collect::<Vec<_>>();
        append_interpolated_children(
            settings,
            frame,
            world,
            use_local_space,
            triangles,
            total_area,
            &parent_positions,
            parents,
            &mut children,
        )?;
    } else {
        append_simple_children(settings, frame, parents, &mut children)?;
    }
    parents.extend(children);
    Ok(())
}

fn append_simple_children(
    settings: ChildSettings,
    frame: f64,
    parents: &[Particle],
    children: &mut Vec<Particle>,
) -> Result<()> {
    let rng_seed = 31_415_926_u32
        .wrapping_add(settings.system_seed)
        .wrapping_add(settings.child_seed);
    let mut random = BlenderRng::seed_random(rng_seed);
    let flat = settings.flat as f32;
    let radius = settings.radius as f32;
    let parent_count = parents.len();
    for child_ordinal in 0..settings.count {
        for (parent_slot, parent) in parents.iter().enumerate() {
            let child_number = child_ordinal
                .checked_mul(parent_count)
                .and_then(|index| index.checked_add(parent_slot))
                .ok_or_else(|| limit("child particle ID exceeds supported range"))?;
            let child_number = u32::try_from(child_number)
                .map_err(|_| limit("child particle ID exceeds supported range"))?;
            let local_offset = loop {
                let x = random.get_float() * 2.0 - 1.0;
                let y = random.get_float() * 2.0 - 1.0;
                let z = random.get_float() * 2.0 - 1.0;
                if (x * x + y * y + z * z).sqrt() < 1.0 {
                    break DVec3::new(
                        f64::from(x * flat * radius),
                        f64::from(y * radius),
                        f64::from(z * radius),
                    );
                }
            };
            let parent_index = u32::try_from(parent_slot)
                .map_err(|_| limit("child particle parent ID exceeds supported range"))?;
            let parent_rotation = DQuat::from_array(
                parent
                    .path
                    .first()
                    .map_or(parent.birth_rotation, |key| key.rotation),
            );
            if !parent_rotation.is_finite() || parent_rotation.length_squared() <= 1.0e-20 {
                return Err(invalid("particle path rotation must be finite and nonzero"));
            }
            let parent_rotation = parent_rotation.normalize();
            let mut path = Vec::with_capacity(parent.path.len());
            for (key_index, key) in parent.path.iter().enumerate() {
                let rotation = if key_index == 0 {
                    parent_rotation
                } else {
                    DQuat::from_array(key.rotation) * parent_rotation
                };
                if !rotation.is_finite() || rotation.length_squared() <= 1.0e-20 {
                    return Err(invalid("particle path rotation must be finite and nonzero"));
                }
                let rotation = rotation.normalize();
                let position = DVec3::from_array(key.position) + rotation * local_offset;
                if !position.is_finite() {
                    return Err(invalid("child particle path produced a non-finite value"));
                }
                path.push(ParticlePathKey {
                    time: key.time,
                    position: position.to_array(),
                    velocity: key.velocity,
                    rotation: rotation.to_array(),
                });
            }
            let age_time = particle_path_time(frame, parent.birth_frame, parent.death_frame);
            let current = sample_path(&path, age_time, parent.path_interpolation);
            let birth_position = path.first().map_or_else(
                || {
                    (DVec3::from_array(parent.birth_position) + parent_rotation * local_offset)
                        .to_array()
                },
                |key| key.position,
            );
            let current_rotation =
                current.map_or(parent_rotation, |key| DQuat::from_array(key.rotation));
            children.push(Particle {
                position: current.map_or_else(
                    || {
                        (DVec3::from_array(parent.position) + current_rotation * local_offset)
                            .to_array()
                    },
                    |key| key.position,
                ),
                birth_position,
                normal: parent.normal,
                birth_rotation: parent.birth_rotation,
                current_path_rotation: current.map(|key| key.rotation),
                path_root_rotation: parent.path_root_rotation,
                path_interpolation: parent.path_interpolation,
                velocity: current.map_or(parent.velocity, |key| key.velocity),
                birth_frame: parent.birth_frame,
                death_frame: parent.death_frame,
                life_state: parent.life_state,
                is_child: true,
                parent_index: Some(parent_index),
                parent_indices: [0; 3],
                parent_weights: [0.0; 3],
                parent_count: 0,
                path,
                size: parent.size
                    * settings.size
                    * (1.0
                        - settings.size_random
                            * f64::from(psys_frand(
                                settings.system_seed,
                                child_number.wrapping_add(26),
                            ))),
                mass: parent.mass,
            });
        }
    }
    Ok(())
}

fn append_interpolated_children(
    settings: ChildSettings,
    frame: f64,
    world: DMat4,
    use_local_space: bool,
    triangles: &[Triangle],
    total_area: f64,
    parent_positions: &[DVec3],
    parents: &[Particle],
    children: &mut Vec<Particle>,
) -> Result<()> {
    if triangles.is_empty() || total_area <= 0.0 {
        return Err(invalid(
            "interpolated particle children require emitter face geometry",
        ));
    }
    let rng_seed = 31_415_926_u32
        .wrapping_add(settings.system_seed)
        .wrapping_add(settings.child_seed);
    let mut distribution_random = BlenderRng::seed_random(rng_seed);
    let mut coordinate_random = BlenderRng::seed(31_415_926_u32.wrapping_add(settings.system_seed));
    let child_count = parents
        .len()
        .checked_mul(settings.count)
        .ok_or_else(|| limit("child particle count exceeds limits"))?;
    let mut child_roots = Vec::with_capacity(child_count);
    for child_number in 0..child_count {
        let local_root = sample_interpolated_root(
            triangles,
            total_area,
            child_number,
            child_count,
            settings.use_emit_random,
            &mut distribution_random,
            &mut coordinate_random,
        );
        let child_root = world.transform_point3(local_root);
        let query_point = if use_local_space {
            local_root
        } else {
            child_root
        };
        child_roots.push(query_point);
        let (parent_indices, parent_weights, parent_count) =
            nearest_parent_weights(query_point, parent_positions)?;
        let child_index = u32::try_from(child_number)
            .map_err(|_| limit("child particle ID exceeds supported range"))?;
        let representative = parent_indices[0];
        let mut birth_frame = 0.0;
        let mut parent_size = 0.0;
        let mut mass = 0.0;
        let mut normal = DVec3::ZERO;
        let mut birth_position = DVec3::ZERO;
        let mut current_position = DVec3::ZERO;
        let mut current_velocity = DVec3::ZERO;
        for slot in 0..parent_count {
            let parent = &parents[parent_indices[slot]];
            let weight = parent_weights[slot];
            birth_frame += parent.birth_frame * weight;
            parent_size += parent.size * weight;
            mass += parent.mass * weight;
            normal += DVec3::from_array(parent.normal) * weight;
            birth_position += DVec3::from_array(parent.birth_position) * weight;
            current_position += DVec3::from_array(parent.position) * weight;
            current_velocity += DVec3::from_array(parent.velocity) * weight;
        }
        let lifetime = settings.lifetime
            * (1.0
                - settings.lifetime_random
                    * f64::from(psys_frand(
                        settings.system_seed,
                        child_index.wrapping_add(25),
                    )));
        let death_frame = birth_frame + lifetime;
        let life_state = parents[representative].life_state;
        let path = interpolated_child_path(
            parents,
            parent_indices,
            parent_weights,
            parent_count,
            child_root,
        )?;
        if let Some(first) = path.first() {
            birth_position = DVec3::from_array(first.position);
        }
        let age_time = particle_path_time(frame, birth_frame, death_frame);
        let current = sample_path(
            &path,
            age_time,
            parents[parent_indices[0]].path_interpolation,
        );
        let mut parent_ids = [0_u32; 3];
        for slot in 0..parent_count {
            parent_ids[slot] = u32::try_from(parent_indices[slot])
                .map_err(|_| limit("child particle parent ID exceeds supported range"))?;
        }
        let parent_index = parent_ids[0];
        children.push(Particle {
            position: current.map_or_else(|| current_position.to_array(), |key| key.position),
            birth_position: birth_position.to_array(),
            normal: normal.normalize_or_zero().to_array(),
            birth_rotation: parents[representative].birth_rotation,
            current_path_rotation: current.map(|key| key.rotation),
            path_root_rotation: parents[parent_indices[0]].path_root_rotation,
            path_interpolation: parents[parent_indices[0]].path_interpolation,
            velocity: current.map_or(current_velocity.to_array(), |key| key.velocity),
            birth_frame,
            death_frame,
            life_state,
            is_child: true,
            parent_index: Some(parent_index),
            parent_indices: parent_ids,
            parent_weights,
            parent_count: u8::try_from(parent_count)
                .map_err(|_| limit("child parent count exceeds supported range"))?,
            path,
            size: parent_size
                * settings.size
                * (1.0
                    - settings.size_random
                        * f64::from(psys_frand(
                            settings.system_seed,
                            child_index.wrapping_add(26),
                        ))),
            mass,
        });
    }
    if settings.virtual_parents > 0.0 {
        let virtual_count = ((child_count as f32 * settings.virtual_parents as f32 * 0.3_f32)
            as usize)
            .min(parents.len())
            .min(child_count);
        for child_number in virtual_count..child_count {
            if virtual_count == 0 {
                break;
            }
            let root = child_roots[child_number];
            let mut nearest = 0;
            let mut nearest_distance = f64::INFINITY;
            for (virtual_index, virtual_root) in child_roots[..virtual_count].iter().enumerate() {
                let distance = virtual_root.distance_squared(root);
                if distance < nearest_distance {
                    nearest = virtual_index;
                    nearest_distance = distance;
                }
            }
            children[child_number].parent_index = Some(
                u32::try_from(nearest)
                    .map_err(|_| limit("virtual child parent ID exceeds supported range"))?,
            );
        }
    }
    Ok(())
}

fn sample_interpolated_root(
    triangles: &[Triangle],
    total_area: f64,
    child_number: usize,
    child_count: usize,
    random_order: bool,
    distribution_random: &mut BlenderRng,
    coordinate_random: &mut BlenderRng,
) -> DVec3 {
    let distribution_position = if random_order {
        f64::from(distribution_random.get_float())
    } else {
        let step = if child_count < 2 {
            0.5
        } else {
            1.0 / child_count as f64
        };
        let first = if child_count < triangles.len() {
            0.5 / triangles.len() as f64
        } else {
            step * 0.5
        };
        first + child_number as f64 * step
    };
    let sample = (distribution_position * total_area).clamp(0.0, total_area);
    let triangle_index = triangles
        .partition_point(|triangle| triangle.cumulative_area < sample)
        .min(triangles.len() - 1);
    let triangle = triangles[triangle_index];
    let mut first = f64::from(coordinate_random.get_float());
    let mut second = f64::from(coordinate_random.get_float());
    let _ = coordinate_random.get_float();
    if first + second > 1.0 {
        first = 1.0 - first;
        second = 1.0 - second;
    }
    let weights = [1.0 - first - second, first, second];
    triangle.points[0] * weights[0]
        + triangle.points[1] * weights[1]
        + triangle.points[2] * weights[2]
}

fn nearest_parent_weights(
    point: DVec3,
    parents: &[DVec3],
) -> Result<([usize; 3], [f64; 3], usize)> {
    if parents.is_empty() {
        return Err(invalid(
            "interpolated children require at least one parent particle",
        ));
    }
    let mut nearest = [(usize::MAX, f64::INFINITY); 3];
    for (index, parent) in parents.iter().enumerate() {
        let distance = parent.distance(point);
        if !distance.is_finite() {
            return Err(invalid("interpolated child parent distance is non-finite"));
        }
        for slot in 0..nearest.len() {
            if distance < nearest[slot].1
                || (distance == nearest[slot].1 && index < nearest[slot].0)
            {
                for displaced in (slot + 1..nearest.len()).rev() {
                    nearest[displaced] = nearest[displaced - 1];
                }
                nearest[slot] = (index, distance);
                break;
            }
        }
    }
    let count = parents.len().min(3);
    let maximum_distance = nearest[count - 1].1;
    let mut indices = [usize::MAX; 3];
    let mut weights = [0.0; 3];
    let mut total_weight = 0.0;
    for slot in 0..count {
        indices[slot] = nearest[slot].0;
        weights[slot] = if maximum_distance <= f64::EPSILON {
            1.0
        } else {
            2.0_f64.powf(-6.0 * nearest[slot].1 / maximum_distance)
        };
        total_weight += weights[slot];
    }
    for weight in &mut weights[..count] {
        *weight /= total_weight;
    }
    Ok((indices, weights, count))
}

fn interpolated_child_path(
    parents: &[Particle],
    indices: [usize; 3],
    weights: [f64; 3],
    parent_count: usize,
    child_root: DVec3,
) -> Result<Vec<ParticlePathKey>> {
    if parents[indices[0]].path.is_empty()
        || (0..parent_count).any(|slot| parents[indices[slot]].path.is_empty())
    {
        return Ok(Vec::new());
    }
    let reference_path = &parents[indices[0]].path;
    let mut path = Vec::with_capacity(reference_path.len());
    for key in reference_path {
        let mut position = DVec3::ZERO;
        let mut velocity = DVec3::ZERO;
        let mut rotation = DQuat::IDENTITY;
        for slot in 0..parent_count {
            let parent = &parents[indices[slot]];
            let Some(sample) = sample_path(&parent.path, key.time, parent.path_interpolation)
            else {
                return Err(invalid("interpolated child parent path is empty"));
            };
            let root = DVec3::from_array(parent.path[0].position);
            position += (DVec3::from_array(sample.position) + child_root - root) * weights[slot];
            velocity += DVec3::from_array(sample.velocity) * weights[slot];
            if slot == 0 {
                rotation = DQuat::from_array(sample.rotation);
            }
        }
        if !position.is_finite() || !velocity.is_finite() {
            return Err(invalid(
                "interpolated child path produced a non-finite value",
            ));
        }
        path.push(ParticlePathKey {
            time: key.time,
            position: position.to_array(),
            velocity: velocity.to_array(),
            rotation: rotation.normalize().to_array(),
        });
    }
    Ok(path)
}

fn particle_life_state(frame: f64, birth_frame: f64, death_frame: f64) -> ParticleLifeState {
    if frame < birth_frame {
        ParticleLifeState::Unborn
    } else if frame >= death_frame {
        ParticleLifeState::Dead
    } else {
        ParticleLifeState::Alive
    }
}

fn particle_path_time(frame: f64, birth_frame: f64, death_frame: f64) -> f64 {
    if frame < birth_frame {
        0.0
    } else if death_frame <= birth_frame || frame >= death_frame {
        1.0
    } else {
        (frame - birth_frame) / (death_frame - birth_frame)
    }
}

fn setting_alias_number(settings: &Value, keys: &[&str], default: f64) -> Result<f64> {
    for key in keys {
        if settings.get(*key).is_some() {
            return super::setting_number(settings, key, default, "particle", "a finite number");
        }
    }
    Ok(default)
}

fn setting_velocity(settings: &Value) -> Result<DVec3> {
    let Some(value) = settings
        .get("random_velocity")
        .or_else(|| settings.get("factor_random"))
    else {
        return Ok(DVec3::ZERO);
    };
    if let Some(value) = value.as_f64().filter(|number| number.is_finite()) {
        return Ok(DVec3::splat(value));
    }
    if let Some(values) = value.as_array()
        && values.len() == 3
    {
        let mut components = [0.0; 3];
        for (index, value) in values.iter().enumerate() {
            components[index] = value
                .as_f64()
                .filter(|number| number.is_finite())
                .ok_or_else(|| {
                    invalid("particle random_velocity components must be finite numbers")
                })?;
        }
        return Ok(DVec3::from_array(components));
    }
    Err(invalid(
        "particle random_velocity must be a number or a three-number array",
    ))
}
fn setting_vector(settings: &Value, key: &str) -> Result<DVec3> {
    let Some(value) = settings.get(key) else {
        return Ok(DVec3::ZERO);
    };
    let values = value
        .as_array()
        .filter(|values| values.len() == 3)
        .ok_or_else(|| invalid(&format!("particle {key} must be a three-number array")))?;
    let mut components = [0.0; 3];
    for (index, value) in values.iter().enumerate() {
        components[index] = value
            .as_f64()
            .filter(|number| number.is_finite())
            .ok_or_else(|| invalid(&format!("particle {key} components must be finite numbers")))?;
    }
    Ok(DVec3::from_array(components))
}

fn unsupported(feature_id: &str, reason: &str) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        reason,
        json!({"feature_id":feature_id,"reason":reason,"status":"not_supported"}),
    )
}
fn invalid(message: &str) -> PotError {
    PotError::new(ErrorCode::InvalidArgument, message)
}

fn limit(message: &str) -> PotError {
    PotError::new(ErrorCode::LimitExceeded, message)
}

#[cfg(test)]
mod tests {
    #![expect(clippy::expect_used, reason = "solver tests use valid fixture inputs")]
    use glam::{DMat4, DQuat, DVec3};
    use serde_json::json;

    use crate::geom::{Edge, Face, IdCounters, Mesh, Vertex};

    use super::{
        ChildSettings, Particle, ParticleLifeState, ParticlePathInterpolation, ParticlePathKey,
        append_simple_children, particle_path, sample_path, simulate,
    };

    #[test]
    fn emission_is_deterministic_and_tracks_particle_age() {
        let mesh = Mesh {
            vertices: vec![
                Vertex {
                    id: 0,
                    co: DVec3::ZERO,
                },
                Vertex {
                    id: 1,
                    co: DVec3::X,
                },
                Vertex {
                    id: 2,
                    co: DVec3::Y,
                },
            ],
            edges: vec![
                Edge {
                    id: 0,
                    vertices: [0, 1],
                },
                Edge {
                    id: 1,
                    vertices: [1, 2],
                },
                Edge {
                    id: 2,
                    vertices: [2, 0],
                },
            ],
            faces: vec![Face {
                id: 0,
                vertices: vec![0, 1, 2],
                material_index: 0,
            }],
            next_id: IdCounters {
                vertex: 3,
                edge: 3,
                face: 1,
            },
            ..Mesh::default()
        };
        let settings = json!({"count": 6, "frame_start": 1.0, "frame_end": 1.0, "lifetime": 20.0, "speed": 1.0});
        let first = simulate(
            &mesh,
            &settings,
            DMat4::IDENTITY,
            3.0,
            24.0,
            DVec3::ZERO,
            7,
            &[],
            &[],
        )
        .expect("simulation succeeds");
        let second = simulate(
            &mesh,
            &settings,
            DMat4::IDENTITY,
            3.0,
            24.0,
            DVec3::ZERO,
            7,
            &[],
            &[],
        )
        .expect("simulation succeeds");
        assert_eq!(first, second);
        assert_eq!(first.len(), 6);
        assert!(first.iter().all(|particle| particle.birth_frame <= 3.0));
    }

    #[test]
    fn wind_force_bounces_from_world_space_collider() {
        let mesh = Mesh {
            vertices: vec![
                Vertex {
                    id: 0,
                    co: DVec3::ZERO,
                },
                Vertex {
                    id: 1,
                    co: DVec3::X,
                },
                Vertex {
                    id: 2,
                    co: DVec3::Y,
                },
            ],
            edges: vec![
                Edge {
                    id: 0,
                    vertices: [0, 1],
                },
                Edge {
                    id: 1,
                    vertices: [1, 2],
                },
                Edge {
                    id: 2,
                    vertices: [2, 0],
                },
            ],
            faces: vec![Face {
                id: 0,
                vertices: vec![0, 1, 2],
                material_index: 0,
            }],
            next_id: IdCounters {
                vertex: 3,
                edge: 3,
                face: 1,
            },
            ..Mesh::default()
        };
        let settings = json!({"count": 2, "frame_start": 1.0, "frame_end": 3.0, "lifetime": 5.0, "speed": 0.0});
        let colliders = [[
            DVec3::new(0.0, 0.0, 0.5),
            DVec3::new(1.0, 0.0, 0.5),
            DVec3::new(0.0, 1.0, 0.5),
        ]];
        let particles = simulate(
            &mesh,
            &settings,
            DMat4::IDENTITY,
            2.0,
            1.0,
            DVec3::ZERO,
            3,
            &[(DVec3::ZERO, "wind", 1.0, 0.0)],
            &colliders,
        )
        .expect("simulation succeeds");
        assert_eq!(particles.len(), 2);
        assert!(particles[0].position[2] < 0.5);
        assert!(particles[0].velocity[2] < 0.0);
        assert_eq!(particles[1].velocity, [0.0; 3]);
    }

    #[test]
    fn optional_velocity_is_added_to_normal_velocity() {
        let mesh = Mesh {
            vertices: vec![
                Vertex {
                    id: 0,
                    co: DVec3::ZERO,
                },
                Vertex {
                    id: 1,
                    co: DVec3::X,
                },
                Vertex {
                    id: 2,
                    co: DVec3::Y,
                },
            ],
            edges: vec![
                Edge {
                    id: 0,
                    vertices: [0, 1],
                },
                Edge {
                    id: 1,
                    vertices: [1, 2],
                },
                Edge {
                    id: 2,
                    vertices: [2, 0],
                },
            ],
            faces: vec![Face {
                id: 0,
                vertices: vec![0, 1, 2],
                material_index: 0,
            }],
            next_id: IdCounters {
                vertex: 3,
                edge: 3,
                face: 1,
            },
            ..Mesh::default()
        };
        let particles = simulate(
            &mesh,
            &json!({
                "count": 2,
                "frame_start": 1.0,
                "frame_end": 3.0,
                "lifetime": 5.0,
                "normal_velocity": 0.5,
                "velocity": [1.0, 2.0, 3.0],
                "source": "vertices",
            }),
            DMat4::IDENTITY,
            2.0,
            1.0,
            DVec3::ZERO,
            4,
            &[],
            &[],
        )
        .expect("simulation succeeds");
        assert_eq!(particles.len(), 2);
        assert_eq!(particles[0].velocity, [1.0, 2.0, 3.5]);
    }
    #[test]
    fn hair_paths_derive_tangents_from_position_keys() {
        let paths = vec![json!([
            {"time":0.0,"position":[0.0,0.0,0.0],"rotation":[0.0,0.0,0.0,1.0]},
            {"time":0.5,"position":[1.0,0.0,0.0]},
            {"time":1.0,"position":[1.0,1.0,0.0]}
        ])];
        let path = particle_path(&paths, 0).expect("Hair path is valid");

        assert_eq!(path.len(), 3);
        assert_eq!(path[0].rotation, DQuat::IDENTITY.to_array());
        assert!((path[0].velocity[0] - 1.0).abs() < 1.0e-12);
        assert!((path[1].velocity[0] - 1.0).abs() < 1.0e-12);
        assert!((path[1].velocity[1] - 1.0).abs() < 1.0e-12);
        assert!((path[2].velocity[1] - 1.0).abs() < 1.0e-12);
        let curved = sample_path(&path, 0.25, ParticlePathInterpolation::HairCardinal)
            .expect("Hair cardinal path samples");
        assert!((curved.position[0] - 0.5).abs() < 1.0e-12);
        assert!((curved.position[1] + 0.0625).abs() < 1.0e-12);
        assert!((curved.velocity[0] - 2.5).abs() < 1.0e-12);
        assert!((curved.velocity[1] + 0.25).abs() < 1.0e-12);
        let bspline = sample_path(&path, 0.25, ParticlePathInterpolation::HairBspline)
            .expect("Hair B-spline path samples");
        assert!((bspline.position[0] - 0.5).abs() < 1.0e-12);
        assert!((bspline.position[1] - 1.0 / 48.0).abs() < 1.0e-12);
        let linear = sample_path(&path, 0.25, ParticlePathInterpolation::Linear)
            .expect("linear pathcache samples");
        let hair_cache = sample_path(&path, 0.25, ParticlePathInterpolation::HairCache)
            .expect("Hair pathcache samples");
        assert_eq!(hair_cache.position, linear.position);
        assert_eq!(hair_cache.velocity, linear.velocity);
    }

    #[test]
    fn missing_path_keys_keep_live_particle_state() {
        let mesh = Mesh::from_positions_and_faces(
            vec![DVec3::ZERO, DVec3::X, DVec3::Y],
            vec![vec![0, 1, 2]],
        )
        .expect("valid emitter mesh");
        let settings = json!({
            "count": 1,
            "frame_start": 1.0,
            "frame_end": 1.0,
            "lifetime": 5.0,
            "physics_type": "NO",
        });

        let particles = super::simulate_lifecycle(
            &mesh,
            &settings,
            DMat4::IDENTITY,
            2.0,
            24.0,
            DVec3::ZERO,
            7,
            &[],
            &[],
        )
        .expect("missing path keys preserve current particle state");

        assert_eq!(particles.len(), 1);
        assert!(particles[0].path.is_empty());
        assert_eq!(particles[0].life_state, ParticleLifeState::Alive);
    }

    #[test]
    fn simple_children_keep_parent_lifecycle_and_path_mapping() {
        let parent = Particle {
            position: [0.0, 0.0, 0.4],
            birth_position: [0.0; 3],
            normal: [0.0, 0.0, 1.0],
            birth_rotation: DQuat::IDENTITY.to_array(),
            current_path_rotation: None,
            path_root_rotation: None,
            path_interpolation: ParticlePathInterpolation::Hermite,
            velocity: [0.0, 0.0, 1.0],
            birth_frame: 2.0,
            death_frame: 12.0,
            life_state: ParticleLifeState::Alive,
            is_child: false,
            parent_index: None,
            parent_indices: [0; 3],
            parent_weights: [0.0; 3],
            parent_count: 0,
            path: vec![
                ParticlePathKey {
                    time: 0.0,
                    position: [0.0; 3],
                    velocity: [0.0, 0.0, 1.0],
                    rotation: DQuat::IDENTITY.to_array(),
                },
                ParticlePathKey {
                    time: 1.0,
                    position: [0.0, 0.0, 1.0],
                    velocity: [0.0, 0.0, 1.0],
                    rotation: DQuat::IDENTITY.to_array(),
                },
            ],
            size: 3.0,
            mass: 1.0,
        };
        let settings = ChildSettings {
            count: 1,
            radius: 0.2,
            flat: 0.0,
            size: 0.5,
            size_random: 0.0,
            child_seed: 0,
            system_seed: 7,
            lifetime: 5.0,
            lifetime_random: 0.0,
            virtual_parents: 0.0,
            interpolated: false,
            use_emit_random: false,
        };
        let mut children = Vec::new();
        append_simple_children(settings, 6.0, &[parent], &mut children)
            .expect("Simple child state is valid");
        let child = &children[0];

        assert_eq!(child.parent_index, Some(0));
        assert_eq!(child.parent_count, 0);
        assert_eq!(child.birth_frame, 2.0);
        assert_eq!(child.death_frame, 12.0);
        assert_eq!(child.size, 1.5);
        assert_eq!(child.birth_position, child.path[0].position);
        assert!((child.position[2] - child.path[0].position[2] - 0.4).abs() < 1.0e-12);
    }
    #[test]
    fn cached_hair_parent_and_child_paths_are_sampled_without_path_mode() {
        let mesh = Mesh::from_positions_and_faces(
            vec![DVec3::ZERO, DVec3::X, DVec3::Y],
            vec![vec![0, 1, 2]],
        )
        .expect("valid emitter mesh");
        let settings = json!({
            "type": "HAIR",
            "count": 1,
            "frame_start": 1.0,
            "frame_end": 1.0,
            "lifetime": 10.0,
            "physics_type": "NO",
            "child_type": "SIMPLE",
            "child_percent": 1,
            "_potter_path_interpolation": "hair_cache",
            "_potter_path_keys": [[
                {
                    "time": 0.0,
                    "position": [0.0, 0.0, 0.0],
                    "velocity": [0.5, 0.0, 0.0],
                    "rotation": [0.0, 0.0, 0.0, 1.0]
                },
                {
                    "time": 1.0,
                    "position": [0.5, 0.0, 0.0],
                    "velocity": [0.5, 0.0, 0.0],
                    "rotation": [0.0, 0.0, 0.0, 1.0]
                }
            ]],
            "_potter_child_path_keys": [[
                {
                    "time": 0.0,
                    "position": [0.25, 0.25, 0.0],
                    "velocity": [0.5, 0.0, 0.0],
                    "rotation": [0.0, 0.0, 0.0, 1.0]
                },
                {
                    "time": 1.0,
                    "position": [0.75, 0.25, 0.0],
                    "velocity": [0.5, 0.0, 0.0],
                    "rotation": [0.0, 0.0, 0.0, 1.0]
                }
            ]]
        });
        let particles = super::simulate_lifecycle(
            &mesh,
            &settings,
            DMat4::IDENTITY,
            1.0,
            24.0,
            DVec3::ZERO,
            7,
            &[],
            &[],
        )
        .expect("cached parent and child paths are sampled");
        assert_eq!(particles.len(), 2);
        assert_eq!(particles[0].birth_frame, 0.0);
        assert_eq!(particles[0].life_state, ParticleLifeState::Alive);
        assert!(!particles[0].path.is_empty());
        assert!(particles[0].path_root_rotation.is_some());
        assert_eq!(particles[1].life_state, ParticleLifeState::Alive);
        assert_eq!(
            particles[1].path_interpolation,
            ParticlePathInterpolation::HairCache
        );
        assert_eq!(particles[1].path[0].position, [0.25, 0.25, 0.0]);
        assert!(particles[1].path_root_rotation.is_some());
    }

    #[test]
    fn hair_birth_times_ignore_emission_schedule() {
        let mesh = Mesh::from_positions_and_faces(
            vec![DVec3::ZERO, DVec3::X, DVec3::Y],
            vec![vec![0, 1, 2]],
        )
        .expect("valid emitter mesh");
        let settings = json!({
            "type": "HAIR",
            "count": 3,
            "frame_start": 1.0,
            "frame_end": 10.0,
            "lifetime": 10.0,
            "physics_type": "NO"
        });
        let particles = super::simulate_lifecycle(
            &mesh,
            &settings,
            DMat4::IDENTITY,
            1.0,
            24.0,
            DVec3::ZERO,
            7,
            &[],
            &[],
        )
        .expect("Hair particles exist before emission scheduling");
        assert_eq!(particles.len(), 3);
        assert!(particles.iter().all(|particle| {
            particle.birth_frame == 0.0 && particle.life_state == ParticleLifeState::Alive
        }));
    }
}
