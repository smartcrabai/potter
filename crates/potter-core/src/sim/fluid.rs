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

/// A scene flow object, expressed as world-space triangles.
pub struct FlowSource {
    pub triangles: Vec<[DVec3; 3]>,
    pub inflow_rate: f64,
}

/// Whether a flow emits from a surface triangle whose centre lies inside an effector.
///
/// Particles spawn on the flow surface and Potter does not push a particle out of an effector it
/// starts in, so the caller reports the case as a loss. Blender 5.2.2 keeps that liquid in the
/// effector's column but slows it: measured headless with a 1-unit INFLOW flow at z = 1.5 and a
/// 0.3-unit effector at z = 0.95, the column's lowest liquid descends about 0.012 per frame while it is
/// inside, against about 0.075 without the effector, and it leaves through the bottom face around frame 8.
/// Potter has no grid obstacle rule to reproduce that, so this is a reported gap (solver parity).
#[must_use]
pub fn emitter_spawns_inside_effector(flows: &[FlowSource], effectors: &[[DVec3; 3]]) -> bool {
    const SAMPLE_DIVISIONS: usize = 8;
    let Some((effector_min, effector_max)) = triangle_bounds_all(effectors) else {
        return false;
    };
    flows.iter().any(|flow| {
        flow.triangles.iter().any(|triangle| {
            let (minimum, maximum) = triangle_bounds(triangle);
            // A triangle whose box misses the effector's box cannot have a point inside it.
            let apart = (0..3).any(|axis| {
                maximum[axis] < effector_min[axis] || minimum[axis] > effector_max[axis]
            });
            !apart
                && triangle_samples(*triangle, SAMPLE_DIVISIONS)
                    .any(|point| point_inside_closed_surface(point, effectors))
        })
    })
}

fn triangle_bounds(triangle: &[DVec3; 3]) -> (DVec3, DVec3) {
    (
        triangle[0].min(triangle[1]).min(triangle[2]),
        triangle[0].max(triangle[1]).max(triangle[2]),
    )
}

fn triangle_bounds_all(triangles: &[[DVec3; 3]]) -> Option<(DVec3, DVec3)> {
    triangles
        .iter()
        .map(triangle_bounds)
        .reduce(|(minimum, maximum), (next_min, next_max)| {
            (minimum.min(next_min), maximum.max(next_max))
        })
}

/// Points on a regular barycentric grid over `triangle`, `divisions` steps along each edge.
fn triangle_samples(triangle: [DVec3; 3], divisions: usize) -> impl Iterator<Item = DVec3> {
    let [first, second, third] = triangle;
    let step = divisions as f64;
    (0..=divisions).flat_map(move |row| {
        (0..=divisions - row).map(move |column| {
            let u = column as f64 / step;
            let v = row as f64 / step;
            first + (second - first) * u + (third - first) * v
        })
    })
}

/// Parity of ray crossings through a closed triangle set; an odd count means inside.
fn point_inside_closed_surface(point: DVec3, triangles: &[[DVec3; 3]]) -> bool {
    let direction = DVec3::new(1.0, 0.371_390_676_354_103_7, 0.529_173_248_214_697_2);
    triangles
        .iter()
        .filter(|triangle| ray_crosses_triangle(point, direction, **triangle))
        .count()
        % 2
        == 1
}

/// Möller-Trumbore test for a ray that starts at `origin` and crosses the triangle ahead of it.
fn ray_crosses_triangle(origin: DVec3, direction: DVec3, triangle: [DVec3; 3]) -> bool {
    let [first, second, third] = triangle;
    let first_edge = second - first;
    let second_edge = third - first;
    let p_vector = direction.cross(second_edge);
    let determinant = first_edge.dot(p_vector);
    if determinant.abs() < 1.0e-12 {
        return false;
    }
    let inverse = 1.0 / determinant;
    let offset = origin - first;
    let barycentric_u = offset.dot(p_vector) * inverse;
    if !(0.0..=1.0).contains(&barycentric_u) {
        return false;
    }
    let q_vector = offset.cross(first_edge);
    let barycentric_v = direction.dot(q_vector) * inverse;
    if barycentric_v < 0.0 || barycentric_u + barycentric_v > 1.0 {
        return false;
    }
    second_edge.dot(q_vector) * inverse > 0.0
}

struct PreparedFlow<'a> {
    triangles: &'a [[DVec3; 3]],
    weighted_triangles: Vec<(f64, usize)>,
    total_area: f64,
    inflow_rate: f64,
    count: usize,
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
    simulate_with_sources(
        domain,
        settings,
        world,
        frame,
        fps,
        gravity,
        seed,
        colliders,
        &[],
    )
}

/// Simulate a domain using explicit scene flow sources and their own inflow rates.
pub fn simulate_with_sources(
    domain: &Mesh,
    settings: &Value,
    world: DMat4,
    frame: f64,
    fps: f64,
    gravity: DVec3,
    seed: u32,
    colliders: &[[DVec3; 3]],
    flows: &[FlowSource],
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
    if flows.len() > MAX_COLLIDERS {
        return Err(limit(
            "fluid simulation exceeds the 100000 flow-source limit",
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
    let inflow_rate = if flows.is_empty() {
        super::setting_number(settings, "inflow_rate", 0.0, "fluid", "a finite number")?
    } else {
        0.0
    };
    if inflow_rate < 0.0 {
        return Err(invalid("fluid inflow_rate must be non-negative"));
    }
    let inflow_count_float = (inflow_rate * frame).floor();
    if !inflow_count_float.is_finite() || inflow_count_float > MAX_PARTICLES as f64 {
        return Err(limit("fluid inflow exceeds the particle limit"));
    }
    let inflow_count = inflow_count_float as usize;
    let mut prepared_flows = Vec::with_capacity(flows.len());
    let mut flow_triangle_count = 0_usize;
    let mut flow_inflow_count = 0_usize;
    for flow in flows {
        if !flow.inflow_rate.is_finite() || flow.inflow_rate < 0.0 {
            return Err(invalid(
                "fluid flow inflow_rate must be finite and non-negative",
            ));
        }
        flow_triangle_count = flow_triangle_count
            .checked_add(flow.triangles.len())
            .ok_or_else(|| limit("fluid flow mesh exceeds limits"))?;
        if flow_triangle_count > MAX_COLLIDERS {
            return Err(limit("fluid flow meshes exceed the 100000 triangle limit"));
        }
        if flow.triangles.is_empty() {
            continue;
        }
        let mut source_min = DVec3::splat(f64::INFINITY);
        let mut source_max = DVec3::splat(f64::NEG_INFINITY);
        for triangle in &flow.triangles {
            for point in triangle {
                if !point.is_finite() {
                    return Err(invalid("fluid flow triangles must be finite"));
                }
                source_min = source_min.min(*point);
                source_max = source_max.max(*point);
            }
        }
        if source_min.cmplt(minimum).any() || source_max.cmpgt(maximum).any() {
            continue;
        }
        let flow_count_float = (flow.inflow_rate * frame).floor();
        if !flow_count_float.is_finite() || flow_count_float > MAX_PARTICLES as f64 {
            return Err(limit("fluid flow inflow exceeds the particle limit"));
        }
        let count = flow_count_float as usize;
        if count == 0 {
            continue;
        }
        let mut weighted_triangles = Vec::with_capacity(flow.triangles.len());
        let mut total_area = 0.0;
        for (index, triangle) in flow.triangles.iter().enumerate() {
            let area = (triangle[1] - triangle[0])
                .cross(triangle[2] - triangle[0])
                .length()
                * 0.5;
            if !area.is_finite() {
                return Err(invalid("fluid flow triangle has a non-finite area"));
            }
            if area > f64::EPSILON {
                total_area += area;
                if !total_area.is_finite() {
                    return Err(limit("fluid flow surface area exceeds numeric limits"));
                }
                weighted_triangles.push((total_area, index));
            }
        }
        if total_area == 0.0 {
            continue;
        }
        flow_inflow_count = flow_inflow_count
            .checked_add(count)
            .ok_or_else(|| limit("fluid flow inflow exceeds the particle limit"))?;
        prepared_flows.push(PreparedFlow {
            triangles: &flow.triangles,
            weighted_triangles,
            total_area,
            inflow_rate: flow.inflow_rate,
            count,
        });
    }
    if !flows.is_empty() && prepared_flows.is_empty() {
        return Ok(Vec::new());
    }
    let count = base_count
        .checked_add(inflow_count)
        .and_then(|count| count.checked_add(flow_inflow_count))
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
    for flow in &prepared_flows {
        for index in 0..flow.count {
            let birth_frame = (index + 1) as f64 / flow.inflow_rate;
            let age = (frame - birth_frame).max(0.0) / fps;
            let sample = random.unit() * flow.total_area;
            let weighted_index = flow
                .weighted_triangles
                .partition_point(|(area, _)| *area <= sample)
                .min(flow.weighted_triangles.len() - 1);
            let triangle_index = flow.weighted_triangles[weighted_index].1;
            let [a, b, c] = flow.triangles[triangle_index];
            let root = random.unit().sqrt();
            let across = random.unit();
            let position = a * (1.0 - root) + b * (root * (1.0 - across)) + c * (root * across);
            positions.push(position);
            ages.push(age);
        }
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

    use super::{FlowSource, simulate, simulate_with_sources};

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
    #[test]
    fn contained_flow_emits_from_its_surface_and_outside_flow_is_ignored() {
        let domain = Mesh {
            vertices: vec![
                Vertex {
                    id: 0,
                    co: DVec3::splat(-1.0),
                },
                Vertex {
                    id: 1,
                    co: DVec3::splat(1.0),
                },
            ],
            ..Mesh::default()
        };
        let source = FlowSource {
            triangles: vec![
                [
                    DVec3::new(-0.5, -0.5, 0.5),
                    DVec3::new(0.5, -0.5, 0.5),
                    DVec3::new(0.5, 0.5, 0.5),
                ],
                [
                    DVec3::new(-0.5, -0.5, 0.5),
                    DVec3::new(0.5, 0.5, 0.5),
                    DVec3::new(-0.5, 0.5, 0.5),
                ],
            ],
            inflow_rate: 2.0,
        };
        let settings = json!({"resolution":1,"pressure_stiffness":0.0,"viscosity":0.0});
        let emitted = simulate_with_sources(
            &domain,
            &settings,
            DMat4::IDENTITY,
            1.0,
            24.0,
            DVec3::ZERO,
            7,
            &[],
            &[source],
        )
        .expect("contained flow succeeds");
        assert_eq!(emitted.len(), 3);
        assert!(
            emitted[1..]
                .iter()
                .all(|point| (point.z - 0.5).abs() < 1.0e-12)
        );

        let outside = FlowSource {
            triangles: vec![[
                DVec3::new(-0.5, -0.5, 0.5),
                DVec3::new(0.5, -0.5, 0.5),
                DVec3::new(0.0, 0.5, 1.5),
            ]],
            inflow_rate: 2.0,
        };
        let baseline = simulate(
            &domain,
            &settings,
            DMat4::IDENTITY,
            1.0,
            24.0,
            DVec3::ZERO,
            7,
            &[],
        )
        .expect("domain succeeds");
        let ignored = simulate_with_sources(
            &domain,
            &settings,
            DMat4::IDENTITY,
            1.0,
            24.0,
            DVec3::ZERO,
            7,
            &[],
            &[outside],
        )
        .expect("outside flow succeeds");
        assert_ne!(baseline, Vec::<DVec3>::new());
        assert_eq!(ignored, Vec::<DVec3>::new());
    }
}
