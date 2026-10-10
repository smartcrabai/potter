//! Deterministic mesh-emitter particle simulation.

use super::ParticleLifeState;
use std::collections::HashMap;

use glam::{DMat4, DQuat, DVec3};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
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
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Particle {
    pub position: [f64; 3],
    pub birth_position: [f64; 3],
    pub normal: [f64; 3],
    pub birth_rotation: [f64; 4],
    /// World-space velocity in meters per second.
    pub velocity: [f64; 3],
    /// Birth time in frame units.
    pub birth_frame: f64,
    pub death_frame: f64,
    pub life_state: ParticleLifeState,
    pub is_child: bool,
    pub size: f64,
    pub mass: f64,
}

#[derive(Clone, Copy)]
struct Triangle {
    points: [DVec3; 3],
    normal: DVec3,
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
/// `rate` is particles per frame, `lifetime` is measured in frames, and `speed`,
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
    let include_unborn = include_lifecycle
        && settings
            .get("emit_unborn")
            .and_then(Value::as_bool)
            .unwrap_or(false);
    let has_frame_start =
        settings.get("frame_start").is_some() || settings.get("start_frame").is_some();
    let frame_start = setting_alias_number(settings, &["frame_start", "start_frame"], 0.0)?;
    let count = settings.get("count").and_then(Value::as_f64);
    let frame_end = setting_alias_number(
        settings,
        &["frame_end", "end_frame"],
        if let Some(count) = count {
            frame_start + count.max(1.0)
        } else if include_unborn {
            frame + 1.0
        } else {
            frame
        },
    )?;
    if !frame_start.is_finite() || !frame_end.is_finite() || frame_end < frame_start {
        return Err(invalid("particle emission frame range is invalid"));
    }
    let frame_span = (frame_end - frame_start).max(0.0);
    let rate = if let Some(count) = count {
        count / frame_span.max(1.0)
    } else {
        setting_alias_number(settings, &["rate", "emission_rate"], 1.0)?
    };
    if rate < 0.0 || !rate.is_finite() {
        return Err(invalid(
            "particle emission rate must be finite and non-negative",
        ));
    }
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

    let seed = settings
        .get("seed")
        .map(|value| {
            value
                .as_u64()
                .ok_or_else(|| invalid("particle seed must be a non-negative integer"))
        })
        .transpose()?
        .unwrap_or(u64::from(seed));
    let requested_frame = if include_unborn {
        frame.max(frame_end)
    } else {
        frame
    };
    let emission_frame = requested_frame.min(frame_end);
    let duration = (emission_frame - frame_start).max(0.0);
    let emitted_float = if let Some(count) = count {
        if emission_frame < frame_start {
            0.0
        } else if frame_span == 0.0 {
            count
        } else {
            ((duration / frame_span * (count - 1.0)).floor() + 1.0).min(count.max(0.0))
        }
    } else if emission_frame < frame_start {
        0.0
    } else if has_frame_start {
        rate * (duration + 1.0)
    } else {
        rate * emission_frame.max(0.0)
    };
    if !emitted_float.is_finite() || emitted_float >= u64::MAX as f64 {
        return Err(limit("particle emission count exceeds the supported range"));
    }
    let emitted = emitted_float.floor() as u64;
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
    let first_event = if include_lifecycle {
        0
    } else {
        let first_birth_time = (frame - lifetime - frame_start).max(0.0);
        let first_event_value = (first_birth_time * rate).ceil();
        if !first_event_value.is_finite() || first_event_value >= u64::MAX as f64 {
            return Err(limit("particle emission count exceeds the supported range"));
        }
        (first_event_value as u64).saturating_sub(1).min(emitted)
    };
    let output_count = emitted.saturating_sub(first_event);
    if output_count > MAX_PARTICLES {
        return Err(limit(
            "particle output exceeds the one-million particle limit",
        ));
    }
    let step_work_factor = fields
        .len()
        .checked_add(world_colliders.len())
        .and_then(|work| work.checked_add(1))
        .ok_or_else(|| limit("particle simulation work exceeds limits"))?;
    let mut estimated_work = 0_usize;
    let maximum_lifetime = lifetime * (1.0 + lifetime_random);
    for event in first_event..emitted {
        let birth_frame = if let Some(count) = count {
            if count <= 1.0 || frame_span == 0.0 {
                frame_start
            } else {
                frame_start + event as f64 * frame_span / (count - 1.0)
            }
        } else if has_frame_start {
            frame_start + event as f64 / rate.max(f64::MIN_POSITIVE)
        } else {
            frame_start + (event + 1) as f64 / rate.max(f64::MIN_POSITIVE)
        };
        let raw_age = frame - birth_frame;
        if raw_age < 0.0 {
            continue;
        }
        let age_frames = raw_age.min(maximum_lifetime);
        let steps = age_frames.ceil();
        if !steps.is_finite() || steps > MAX_STEPS_PER_PARTICLE as f64 {
            return Err(limit("particle lifetime exceeds the 4096 fixed-step limit"));
        }
        estimated_work = estimated_work
            .checked_add((steps as usize).saturating_mul(step_work_factor))
            .ok_or_else(|| limit("particle simulation work exceeds limits"))?;
        if estimated_work > MAX_SIMULATION_WORK {
            return Err(limit(
                "particle simulation exceeds the 50-million work limit",
            ));
        }
    }

    let mut output = Vec::with_capacity(output_count as usize);
    let total_area = triangles
        .last()
        .map_or(0.0, |triangle| triangle.cumulative_area);
    for event in first_event..emitted {
        let birth_frame = if let Some(count) = count {
            if count <= 1.0 || frame_span == 0.0 {
                frame_start
            } else {
                frame_start + event as f64 * frame_span / (count - 1.0)
            }
        } else if has_frame_start {
            frame_start + event as f64 / rate.max(f64::MIN_POSITIVE)
        } else {
            frame_start + (event + 1) as f64 / rate.max(f64::MIN_POSITIVE)
        };
        let mut random = Random::new(seed ^ event.wrapping_mul(0xd6e8_feb8_6659_fd93));
        let actual_lifetime = (lifetime * (1.0 + random.signed() * lifetime_random)).max(0.0);
        let raw_age = frame - birth_frame;
        let unborn = raw_age < 0.0;
        if unborn && !include_lifecycle {
            continue;
        }
        let life_state = if unborn {
            ParticleLifeState::Unborn
        } else if raw_age > actual_lifetime {
            ParticleLifeState::Dead
        } else {
            ParticleLifeState::Alive
        };
        let age_frames = match life_state {
            ParticleLifeState::Unborn => 0.0,
            ParticleLifeState::Alive => raw_age.max(0.0),
            ParticleLifeState::Dead => actual_lifetime,
        };
        if life_state != ParticleLifeState::Alive && !include_lifecycle {
            continue;
        }
        let (local_position, local_normal) = if from_volume {
            (
                sample_volume_position(mesh, &triangles, &mut random)?,
                DVec3::Z,
            )
        } else if from_vertex || triangles.is_empty() || total_area <= 0.0 {
            let index = (random.unit() * mesh.vertices.len() as f64) as usize;
            let index = index.min(mesh.vertices.len() - 1);
            (
                mesh.vertices[index].co,
                vertex_normals[index].normalize_or_zero(),
            )
        } else {
            sample_triangle(&triangles, total_area, &mut random)
        };
        let normal = normal_transform
            .transform_vector3(local_normal)
            .normalize_or_zero();
        let direction = if normal.length_squared() > 0.0 {
            normal
        } else {
            DVec3::Z
        };
        let local_random = DVec3::new(
            random.signed() * random_velocity.x,
            random.signed() * random_velocity.y,
            random.signed() * random_velocity.z,
        );
        let start = world.transform_point3(local_position);
        let velocity = direction * speed + world.transform_vector3(initial_velocity + local_random);
        if !start.is_finite() || !velocity.is_finite() {
            return Err(invalid("particle emission produced a non-finite value"));
        }
        let particle_gravity = if settings.get("physics_type").and_then(Value::as_str) == Some("NO")
        {
            DVec3::ZERO
        } else {
            gravity
        };
        let (position, final_velocity) = integrate_particle(
            start,
            velocity,
            age_frames,
            fps,
            particle_gravity,
            &fields,
            &world_colliders,
        )?;
        let size = setting_alias_number(settings, &["particle_size", "size"], 1.0)?
            * (1.0 + random.signed() * setting_alias_number(settings, &["size_random"], 0.0)?)
                .max(0.0);
        let mass = setting_alias_number(settings, &["mass"], 1.0)?;
        if mass <= 0.0 {
            return Err(invalid("particle mass must be positive"));
        }
        output.push(Particle {
            position: position.to_array(),
            birth_position: start.to_array(),
            normal: normal.to_array(),
            birth_rotation: if normal.length_squared() > f64::EPSILON {
                DQuat::from_rotation_arc(DVec3::X, normal.normalize()).to_array()
            } else {
                DQuat::IDENTITY.to_array()
            },
            velocity: final_velocity.to_array(),
            birth_frame,
            death_frame: birth_frame + actual_lifetime,
            life_state,
            is_child: false,
            size,
            mass,
        });
    }
    if include_lifecycle {
        append_simple_children(settings, seed, &mut output)?;
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
    let mesh_triangles = mesh
        .triangulate()
        .map_err(|error| invalid(&format!("particle emitter mesh is invalid: {error}")))?;
    for triangle in mesh_triangles {
        let indices = triangle.map(|id| vertex_indices.get(&id).copied());
        let [Some(first), Some(second), Some(third)] = indices else {
            return Err(invalid(
                "particle emitter triangle references a missing vertex",
            ));
        };
        let a = mesh.vertices[first].co;
        let b = mesh.vertices[second].co;
        let c = mesh.vertices[third].co;
        let area_normal = (b - a).cross(c - a) * 0.5;
        let area = area_normal.length();
        if !area.is_finite() {
            return Err(invalid("particle emitter face area is non-finite"));
        }
        if area <= f64::EPSILON {
            continue;
        }
        let normal = area_normal / area;
        cumulative_area += area;
        if !cumulative_area.is_finite() {
            return Err(invalid("particle emitter total face area is non-finite"));
        }
        triangles.push(Triangle {
            points: [a, b, c],
            normal,
            cumulative_area,
        });
        vertex_normals[first] += area_normal;
        vertex_normals[second] += area_normal;
        vertex_normals[third] += area_normal;
    }
    for normal in &mut vertex_normals {
        if !normal.is_finite() {
            return Err(invalid("particle emitter vertex normal is non-finite"));
        }
        *normal = normal.normalize_or_zero();
    }
    Ok((triangles, vertex_normals))
}

fn sample_triangle(triangles: &[Triangle], total_area: f64, random: &mut Random) -> (DVec3, DVec3) {
    let sample = random.unit() * total_area;
    let index = triangles.partition_point(|triangle| triangle.cumulative_area < sample);
    let triangle = triangles[index.min(triangles.len() - 1)];
    let first = random.unit().sqrt();
    let second = random.unit();
    let weights = [1.0 - first, first * (1.0 - second), first * second];
    let position = triangle.points[0] * weights[0]
        + triangle.points[1] * weights[1]
        + triangle.points[2] * weights[2];
    (position, triangle.normal)
}

fn sample_volume_position(
    mesh: &Mesh,
    triangles: &[Triangle],
    random: &mut Random,
) -> Result<DVec3> {
    let bounds = mesh
        .bounds()
        .ok_or_else(|| invalid("volume emission requires bounded mesh geometry"))?;
    let direction = DVec3::new(1.0, 0.371_390_676, 0.219_876_543).normalize();
    for _ in 0..128 {
        let point = bounds.min
            + (bounds.max - bounds.min) * DVec3::new(random.unit(), random.unit(), random.unit());
        let intersections = triangles
            .iter()
            .filter(|triangle| ray_intersects_triangle(point, direction, triangle.points))
            .count();
        if intersections % 2 == 1 {
            return Ok(point);
        }
    }
    Err(PotError::with_details(
        ErrorCode::UnsupportedFeature,
        "particle volume emission requires a closed mesh with a sampleable interior",
        json!({"feature_id":"physics.particle_emitter.emit_from_volume"}),
    ))
}

fn ray_intersects_triangle(origin: DVec3, direction: DVec3, triangle: [DVec3; 3]) -> bool {
    let edge_a = triangle[1] - triangle[0];
    let edge_b = triangle[2] - triangle[0];
    let cross = direction.cross(edge_b);
    let determinant = edge_a.dot(cross);
    if determinant.abs() <= 1.0e-12 {
        return false;
    }
    let inverse = determinant.recip();
    let offset = origin - triangle[0];
    let u = offset.dot(cross) * inverse;
    if !(0.0..=1.0).contains(&u) {
        return false;
    }
    let q = offset.cross(edge_a);
    let v = direction.dot(q) * inverse;
    if v < 0.0 || u + v > 1.0 {
        return false;
    }
    direction.dot(q) * inverse > 1.0e-12
}

fn append_simple_children(
    settings: &Value,
    seed: u64,
    particles: &mut Vec<Particle>,
) -> Result<()> {
    let child_count = match settings.get("child_nbr") {
        Some(value) => value
            .as_u64()
            .ok_or_else(|| invalid("child_nbr must be a non-negative integer"))?,
        None => 0,
    };
    if child_count == 0 {
        return Ok(());
    }
    if settings
        .get("child_type")
        .and_then(Value::as_str)
        .is_some_and(|kind| kind != "SIMPLE")
    {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "only Simple particle children are supported",
            json!({"feature_id":"physics.particle_emitter.children.interpolated"}),
        ));
    }
    let radius = setting_alias_number(settings, &["child_radius"], 1.0)?;
    if radius < 0.0 {
        return Err(invalid("child_radius must be non-negative"));
    }
    let additional = particles
        .len()
        .checked_mul(
            usize::try_from(child_count)
                .map_err(|_| limit("child particle count exceeds limits"))?,
        )
        .ok_or_else(|| limit("child particle count exceeds limits"))?;
    if particles
        .len()
        .checked_add(additional)
        .is_none_or(|count| count > 1_000_000)
    {
        return Err(limit(
            "particle output exceeds the one-million particle limit",
        ));
    }
    let mut children = Vec::with_capacity(additional);
    for (parent_index, parent) in particles.iter().enumerate() {
        let parent_index = u64::try_from(parent_index)
            .map_err(|_| limit("child particle ID exceeds supported range"))?;
        for child_ordinal in 0..child_count {
            let child_index = seed
                .wrapping_add(parent_index.wrapping_mul(child_count))
                .wrapping_add(child_ordinal);
            let mut random = Random::new(child_index ^ 0xa076_1d64_78bd_642f);
            let z = random.signed();
            let angle = random.unit() * std::f64::consts::TAU;
            let radial = (1.0 - z * z).max(0.0).sqrt();
            let distance = radius * random.unit().cbrt();
            let offset = DVec3::new(radial * angle.cos(), radial * angle.sin(), z) * distance;
            let mut child = *parent;
            child.position = (DVec3::from_array(parent.position) + offset).to_array();
            child.birth_position = (DVec3::from_array(parent.birth_position) + offset).to_array();
            children.push(child);
        }
    }
    particles.extend(children);
    Ok(())
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
    use glam::{DMat4, DVec3};
    use serde_json::json;

    use crate::geom::{Edge, Face, IdCounters, Mesh, Vertex};

    use super::simulate;

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
        let settings = json!({"rate": 2.0, "lifetime": 20.0, "speed": 1.0});
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
        let settings = json!({"rate": 1.0, "lifetime": 5.0, "speed": 0.0});
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
                "rate": 1.0,
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
}
