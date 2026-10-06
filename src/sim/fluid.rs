//! Small deterministic SPH-style liquid simulation for bounded mesh domains.

use glam::{DMat4, DVec3};
use serde_json::Value;

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::Mesh,
};

const MAX_PARTICLES: usize = 1_024;
const MAX_RESOLUTION: usize = 10;
const MAX_DOMAIN_VERTICES: usize = 1_000_000;
const MAX_COLLIDERS: usize = 100_000;
const MAX_COLLISION_WORK: usize = 50_000_000;

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

/// Simulate a deterministic liquid particle cloud within the transformed domain bounds.
///
/// `resolution` sets a cubic starting-grid quality and the number of bounded solver steps.
/// `colliders` are static world-space triangles; domain-bound collisions remain active.
pub fn simulate(
    domain: &Mesh,
    settings: &Value,
    world: DMat4,
    frame: f64,
    fps: f64,
    gravity: DVec3,
    seed: u32,
    colliders: &[[DVec3; 3]],
) -> Result<Vec<DVec3>> {
    if !frame.is_finite() || frame < 0.0 {
        return Err(invalid("fluid frame must be finite and non-negative"));
    }
    if !fps.is_finite() || fps <= 0.0 {
        return Err(invalid("fluid frame rate must be finite and positive"));
    }
    if !world.is_finite() || !gravity.is_finite() {
        return Err(invalid("fluid transform and gravity must be finite"));
    }
    if domain.vertices.len() > MAX_DOMAIN_VERTICES {
        return Err(limit("fluid domain exceeds the one-million vertex limit"));
    }
    if colliders.len() > MAX_COLLIDERS {
        return Err(limit(
            "fluid simulation exceeds the 100000 collider-triangle limit",
        ));
    }
    let mut world_colliders = Vec::with_capacity(colliders.len());
    for points in colliders {
        if points.iter().any(|point| !point.is_finite()) {
            return Err(invalid("fluid collider triangle must be finite"));
        }
        let cross = (points[1] - points[0]).cross(points[2] - points[0]);
        let length = cross.length();
        if !length.is_finite() {
            return Err(invalid("fluid collider triangle has a non-finite normal"));
        }
        if length > f64::EPSILON {
            world_colliders.push(ColliderTriangle {
                points: *points,
                normal: cross / length,
            });
        }
    }
    if domain.vertices.is_empty() {
        return Ok(Vec::new());
    }

    let mut local_min = domain.vertices[0].co;
    let mut local_max = local_min;
    for vertex in &domain.vertices {
        if !vertex.co.is_finite() {
            return Err(invalid("fluid domain contains a non-finite vertex"));
        }
        local_min = local_min.min(vertex.co);
        local_max = local_max.max(vertex.co);
    }
    let mut minimum = DVec3::splat(f64::INFINITY);
    let mut maximum = DVec3::splat(f64::NEG_INFINITY);
    for corner in 0..8 {
        let local = DVec3::new(
            if corner & 1 == 0 {
                local_min.x
            } else {
                local_max.x
            },
            if corner & 2 == 0 {
                local_min.y
            } else {
                local_max.y
            },
            if corner & 4 == 0 {
                local_min.z
            } else {
                local_max.z
            },
        );
        let transformed = world.transform_point3(local);
        if !transformed.is_finite() {
            return Err(invalid(
                "fluid domain transform produced a non-finite bound",
            ));
        }
        minimum = minimum.min(transformed);
        maximum = maximum.max(transformed);
    }
    let extent = maximum - minimum;

    let resolution = setting_usize(settings, "resolution", 4)?;
    if resolution == 0 || resolution > MAX_RESOLUTION {
        return Err(limit("fluid resolution must be between 1 and 10"));
    }
    let base_count = resolution
        .checked_mul(resolution)
        .and_then(|count| count.checked_mul(resolution))
        .ok_or_else(|| limit("fluid particle count exceeds limits"))?;
    let inflow_rate =
        super::setting_number(settings, "inflow_rate", 0.0, "fluid", "a finite number")?;
    if inflow_rate < 0.0 {
        return Err(invalid("fluid inflow_rate must be non-negative"));
    }
    let inflow_count_float = (inflow_rate * frame).floor();
    if !inflow_count_float.is_finite() || inflow_count_float > MAX_PARTICLES as f64 {
        return Err(limit("fluid inflow exceeds the particle limit"));
    }
    let inflow_count = inflow_count_float as usize;
    let count = base_count
        .checked_add(inflow_count)
        .ok_or_else(|| limit("fluid particle count exceeds limits"))?;
    if count > MAX_PARTICLES {
        return Err(limit("fluid output exceeds the 1024 particle limit"));
    }
    let collision_work = count
        .checked_mul(resolution + 1)
        .and_then(|work| work.checked_mul(world_colliders.len()))
        .ok_or_else(|| limit("fluid collider work exceeds limits"))?;
    if collision_work > MAX_COLLISION_WORK {
        return Err(limit(
            "fluid simulation exceeds the 50-million collider-work limit",
        ));
    }

    let particle_radius = super::setting_number(
        settings,
        "particle_radius",
        default_particle_radius(extent, resolution),
        "fluid",
        "a finite number",
    )?;
    let smoothing_length = super::setting_number(
        settings,
        "smoothing_length",
        (particle_radius * 6.0).max(max_component(extent) / (resolution as f64 * 4.0)),
        "fluid",
        "a finite number",
    )?;
    let pressure_stiffness = super::setting_number(
        settings,
        "pressure_stiffness",
        8.0,
        "fluid",
        "a finite number",
    )?;
    let viscosity = super::setting_number(settings, "viscosity", 0.1, "fluid", "a finite number")?;
    if particle_radius < 0.0
        || smoothing_length <= 0.0
        || pressure_stiffness < 0.0
        || viscosity < 0.0
    {
        return Err(invalid(
            "fluid particle radius, pressure, and viscosity must be non-negative and smoothing length must be positive",
        ));
    }
    if !particle_radius.is_finite()
        || !smoothing_length.is_finite()
        || !pressure_stiffness.is_finite()
        || !viscosity.is_finite()
    {
        return Err(invalid("fluid settings must be finite"));
    }

    let mut positions = Vec::with_capacity(count);
    let mut velocities = vec![DVec3::ZERO; count];
    let mut ages = Vec::with_capacity(count);
    let column_min = minimum + extent * DVec3::new(0.3, 0.3, 0.15);
    let column_max = maximum - extent * DVec3::new(0.3, 0.3, 0.05);
    let column_extent = (column_max - column_min).max(DVec3::ZERO);
    let mut random = Random::new(u64::from(seed));
    for index in 0..base_count {
        let x = index % resolution;
        let y = (index / resolution) % resolution;
        let z = index / (resolution * resolution);
        let grid = DVec3::new(
            (x as f64 + 0.5) / resolution as f64,
            (y as f64 + 0.5) / resolution as f64,
            (z as f64 + 0.5) / resolution as f64,
        );
        let jitter = DVec3::new(random.signed(), random.signed(), random.signed()) * 0.12;
        let position = clamp_to_bounds(
            column_min + (grid + jitter) * column_extent,
            minimum,
            maximum,
            particle_radius,
        );
        positions.push(position);
        ages.push(frame / fps);
    }
    for index in 0..inflow_count {
        let birth_frame = (index + 1) as f64 / inflow_rate;
        let age = (frame - birth_frame).max(0.0) / fps;
        let position = DVec3::new(
            column_min.x + random.unit() * column_extent.x,
            column_min.y + random.unit() * column_extent.y,
            column_max.z,
        );
        positions.push(clamp_to_bounds(position, minimum, maximum, particle_radius));
        ages.push(age);
    }

    for index in 0..positions.len() {
        let age = ages[index];
        let start = positions[index];
        let ballistic = start + gravity * (0.5 * age * age);
        let mut position = ballistic;
        let mut velocity = gravity * age;
        let mut segment_start = start;
        for collider in &world_colliders {
            if let Some((hit, normal)) =
                collide_segment(segment_start, position, velocity, *collider)
            {
                position = hit + normal * 1.0e-9;
                let normal_speed = velocity.dot(normal);
                if normal_speed < 0.0 {
                    velocity -= normal * (1.5 * normal_speed);
                }
                segment_start = position;
            }
        }
        velocities[index] =
            collision_velocity(position, velocity, minimum, maximum, particle_radius);
        positions[index] = clamp_to_bounds(position, minimum, maximum, particle_radius);
        if !positions[index].is_finite() || !velocities[index].is_finite() {
            return Err(invalid(
                "fluid gravity evaluation produced a non-finite value",
            ));
        }
    }

    let steps = resolution;
    let elapsed = frame / fps;
    let step = (elapsed / steps as f64).min(0.1);
    let mut accelerations = vec![DVec3::ZERO; count];
    let grid_spacing = [column_extent.x, column_extent.y, column_extent.z]
        .into_iter()
        .filter(|component| *component > f64::EPSILON)
        .fold(f64::INFINITY, f64::min)
        / resolution as f64;
    let minimum_spacing = (particle_radius * 2.0).max(if grid_spacing.is_finite() {
        grid_spacing * 0.9
    } else {
        0.0
    });
    for _ in 0..steps {
        accelerations.fill(DVec3::ZERO);
        for first in 0..count {
            for second in first + 1..count {
                let delta = positions[second] - positions[first];
                let distance_squared = delta.length_squared();
                if !distance_squared.is_finite()
                    || distance_squared >= smoothing_length * smoothing_length
                {
                    continue;
                }
                let distance = distance_squared.sqrt();
                let direction = if distance > f64::EPSILON {
                    delta / distance
                } else {
                    pair_direction(first, second)
                };
                let kernel = (1.0 - distance / smoothing_length).max(0.0);
                let compression = ((minimum_spacing - distance).max(0.0)
                    / minimum_spacing.max(f64::EPSILON))
                .min(1.0);
                let pressure = direction * pressure_stiffness * compression * kernel;
                let viscous = (velocities[second] - velocities[first]) * (viscosity * kernel);
                accelerations[first] += -pressure + viscous;
                accelerations[second] += pressure - viscous;
            }
        }
        for index in 0..count {
            let acceleration = accelerations[index];
            let displacement =
                (acceleration * (step * step)).clamp_length_max(smoothing_length * 0.1);
            let old_position = positions[index];
            let mut position = old_position + displacement;
            let mut velocity = velocities[index] + acceleration * step;
            for collider in &world_colliders {
                if let Some((hit, normal)) =
                    collide_segment(old_position, position, velocity, *collider)
                {
                    position = hit + normal * 1.0e-9;
                    let normal_speed = velocity.dot(normal);
                    if normal_speed < 0.0 {
                        velocity -= normal * (1.5 * normal_speed);
                    }
                }
            }
            positions[index] = clamp_to_bounds(position, minimum, maximum, particle_radius);
            velocities[index] = velocity.clamp_length_max(max_component(extent).max(1.0) * 100.0);
            if !positions[index].is_finite() || !velocities[index].is_finite() {
                return Err(invalid("fluid solver produced a non-finite value"));
            }
        }
    }
    Ok(positions)
}

fn clamp_to_bounds(position: DVec3, minimum: DVec3, maximum: DVec3, radius: f64) -> DVec3 {
    let radius = radius.max(0.0);
    let inset_min = (minimum + DVec3::splat(radius)).min((minimum + maximum) * 0.5);
    let inset_max = (maximum - DVec3::splat(radius)).max((minimum + maximum) * 0.5);
    position.max(inset_min).min(inset_max)
}

fn collision_velocity(
    position: DVec3,
    mut velocity: DVec3,
    minimum: DVec3,
    maximum: DVec3,
    radius: f64,
) -> DVec3 {
    let radius = radius.max(0.0);
    let inset_min = (minimum + DVec3::splat(radius)).min((minimum + maximum) * 0.5);
    let inset_max = (maximum - DVec3::splat(radius)).max((minimum + maximum) * 0.5);
    for axis in 0..3 {
        if position[axis] < inset_min[axis] || position[axis] > inset_max[axis] {
            velocity[axis] = 0.0;
        }
    }
    velocity
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

fn pair_direction(first: usize, second: usize) -> DVec3 {
    let hash = (first as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (second as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    match hash % 6 {
        0 => DVec3::X,
        1 => -DVec3::X,
        2 => DVec3::Y,
        3 => -DVec3::Y,
        4 => DVec3::Z,
        _ => -DVec3::Z,
    }
}

fn default_particle_radius(extent: DVec3, resolution: usize) -> f64 {
    let smallest = [extent.x, extent.y, extent.z]
        .into_iter()
        .filter(|component| *component > f64::EPSILON)
        .fold(f64::INFINITY, f64::min);
    if smallest.is_finite() {
        smallest / (resolution as f64 * 8.0)
    } else {
        0.01
    }
}

fn max_component(value: DVec3) -> f64 {
    value.x.max(value.y).max(value.z)
}

fn setting_usize(settings: &Value, key: &str, default: usize) -> Result<usize> {
    let Some(value) = settings.get(key) else {
        return Ok(default);
    };
    let number = value
        .as_u64()
        .ok_or_else(|| invalid(&format!("fluid {key} must be a positive integer")))?;
    usize::try_from(number).map_err(|_| limit(&format!("fluid {key} exceeds limits")))
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

    use crate::geom::{Mesh, Vertex};

    use super::simulate;

    #[test]
    fn liquid_is_deterministic_bounded_and_falls_under_gravity() {
        let domain = Mesh {
            vertices: vec![
                Vertex {
                    id: 0,
                    co: DVec3::new(-1.0, -1.0, 0.0),
                },
                Vertex {
                    id: 1,
                    co: DVec3::new(1.0, 1.0, 2.0),
                },
            ],
            ..Mesh::default()
        };
        let settings = json!({"resolution": 3});
        let initial = simulate(
            &domain,
            &settings,
            DMat4::IDENTITY,
            0.0,
            24.0,
            DVec3::ZERO,
            2,
            &[],
        )
        .expect("simulation succeeds");
        let fallen = simulate(
            &domain,
            &settings,
            DMat4::IDENTITY,
            24.0,
            24.0,
            DVec3::new(0.0, 0.0, -9.81),
            2,
            &[],
        )
        .expect("simulation succeeds");
        assert_eq!(initial.len(), 27);
        assert_eq!(
            fallen,
            simulate(
                &domain,
                &settings,
                DMat4::IDENTITY,
                24.0,
                24.0,
                DVec3::new(0.0, 0.0, -9.81),
                2,
                &[],
            )
            .expect("simulation succeeds")
        );
        assert!(
            fallen
                .iter()
                .all(|point| point.is_finite() && point.z >= 0.0 && point.z <= 2.0)
        );
        assert!(
            fallen.iter().map(|point| point.z).sum::<f64>()
                < initial.iter().map(|point| point.z).sum::<f64>()
        );
    }

    #[test]
    fn liquid_bounces_from_world_space_obstacle_triangles() {
        let domain = Mesh {
            vertices: vec![
                Vertex {
                    id: 0,
                    co: DVec3::new(-1.0, -1.0, 0.0),
                },
                Vertex {
                    id: 1,
                    co: DVec3::new(1.0, 1.0, 2.0),
                },
            ],
            ..Mesh::default()
        };
        let z = 0.5;
        let colliders = [
            [
                DVec3::new(-1.0, -1.0, z),
                DVec3::new(1.0, -1.0, z),
                DVec3::new(1.0, 1.0, z),
            ],
            [
                DVec3::new(-1.0, -1.0, z),
                DVec3::new(1.0, 1.0, z),
                DVec3::new(-1.0, 1.0, z),
            ],
        ];
        let positions = simulate(
            &domain,
            &json!({"resolution": 1}),
            DMat4::IDENTITY,
            1.0,
            1.0,
            DVec3::new(0.0, 0.0, -9.81),
            4,
            &colliders,
        )
        .expect("simulation succeeds");
        assert_eq!(positions.len(), 1);
        assert!(positions[0].z >= z && positions[0].z <= 2.0);
    }
}
