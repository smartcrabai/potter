//! Deterministic local-space cloth simulation using distance constraints.

use std::collections::{BTreeMap, BTreeSet};

use glam::DVec3;
use serde_json::Value;

use crate::{
    error::{PotError, Result},
    geom::{Mesh, edge_key},
};

const MAX_VERTICES: usize = 100_000;
const MAX_TOPOLOGY_ITEMS: usize = 1_000_000;
const MAX_SUBSTEPS: usize = 32;
const MAX_ITERATIONS: usize = 64;
const MAX_WORK: usize = 50_000_000;
const DISTANCE_EPSILON: f64 = 1.0e-14;
const COLLISION_SLOP: f64 = 1.0e-8;

#[derive(Clone, Copy)]
struct Spring {
    a: usize,
    b: usize,
    rest_length: f64,
    alpha: f64,
    lambda: f64,
}
#[derive(Clone, Copy)]
struct CollisionTriangle {
    origin: DVec3,
    normal: DVec3,
    first: DVec3,
    first_dot: f64,
    second_dot: f64,
    cross_dot: f64,
    second: DVec3,
    denominator: f64,
}

/// Evaluate the cloth mesh at `frame` without modifying its input. `colliders` are
/// two-sided triangles expressed in the same local coordinate frame as the mesh.
///
/// Settings: `substeps`, `iterations`, `structural_stiffness`, `shear_stiffness`,
/// `bend_stiffness`, `drag`, and optional `pin_group`. Stiffnesses and group weights
/// are in `[0, 1]`; drag is a nonnegative inverse-time damping rate.
pub fn simulate(
    mesh: &Mesh,
    settings: &Value,
    frame: f64,
    fps: f64,
    gravity: DVec3,
    colliders: &[[DVec3; 3]],
) -> Result<Vec<DVec3>> {
    let config = Config::parse(settings)?;
    validate_inputs(mesh, frame, fps, gravity)?;
    let colliders = prepare_colliders(colliders)?;
    let positions = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    let ids = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<BTreeMap<_, _>>();
    let pins = load_weights(mesh, config.pin_group.as_deref(), &ids)?;
    if positions.is_empty() || frame == 0.0 {
        return Ok(positions);
    }

    let mut structural_keys = BTreeSet::new();
    for edge in &mesh.edges {
        structural_keys.insert(edge_key(edge.vertices[0], edge.vertices[1]));
    }

    let mut shear_keys = BTreeSet::new();
    let mut edge_faces = BTreeMap::<(u32, u32), Vec<usize>>::new();
    for (face_index, face) in mesh.faces.iter().enumerate() {
        let len = face.vertices.len();
        if len == 4 {
            shear_keys.insert(edge_key(face.vertices[0], face.vertices[2]));
            shear_keys.insert(edge_key(face.vertices[1], face.vertices[3]));
        }
        for index in 0..len {
            let key = edge_key(face.vertices[index], face.vertices[(index + 1) % len]);
            edge_faces.entry(key).or_default().push(face_index);
        }
    }

    let mut bend_keys = BTreeSet::new();
    for (edge, faces) in &edge_faces {
        if faces.len() != 2 {
            continue;
        }
        let first = opposite_vertex(&mesh.faces[faces[0]].vertices, *edge);
        let second = opposite_vertex(&mesh.faces[faces[1]].vertices, *edge);
        if (mesh.faces[faces[0]].vertices.len() == 3 || mesh.faces[faces[0]].vertices.len() == 4)
            && (mesh.faces[faces[1]].vertices.len() == 3
                || mesh.faces[faces[1]].vertices.len() == 4)
            && let (Some(first), Some(second)) = (first, second)
            && first != second
        {
            bend_keys.insert(edge_key(first, second));
        }
    }

    let structural = make_springs(
        &structural_keys,
        &ids,
        &positions,
        config.structural_stiffness,
    )?;
    let shear = make_springs(&shear_keys, &ids, &positions, config.shear_stiffness)?;
    let bend = make_springs(&bend_keys, &ids, &positions, config.bend_stiffness)?;
    let self_pair_work = if config.use_self_collision {
        positions
            .len()
            .checked_mul(positions.len().saturating_sub(1))
            .map(|pairs| pairs / 2)
            .ok_or_else(|| limit("cloth self-collision work exceeds limits"))?
    } else {
        0
    };
    let constraints = structural
        .len()
        .checked_add(shear.len())
        .and_then(|count| count.checked_add(bend.len()))
        .and_then(|count| count.checked_add(self_pair_work))
        .ok_or_else(|| limit("cloth constraint count exceeds limits"))?;
    let steps = super::frame_steps(
        frame,
        config.substeps,
        "cloth frame exceeds the bounded simulation step limit",
    )?;
    check_work(
        positions.len(),
        constraints,
        colliders.len(),
        config.iterations,
        steps,
    )?;
    let mut current = positions;
    let mut previous = vec![DVec3::ZERO; current.len()];
    let mut velocity = vec![DVec3::ZERO; current.len()];
    let vertex_count = u32::try_from(current.len())
        .map_err(|_| limit("cloth vertex count exceeds the mass distribution range"))?;
    let mass_per_vertex = config.mass / f64::from(vertex_count);
    let inverse_masses = pins
        .iter()
        .map(|weight| (1.0 - *weight) / mass_per_vertex)
        .collect::<Vec<_>>();
    let dt = frame / fps / steps as f64;
    if !dt.is_finite() || dt == 0.0 {
        return Err(invalid("frame time must be finite and nonzero"));
    }
    let damping = (-config.drag * dt.abs()).exp();

    let mut structural = structural;
    let mut shear = shear;
    let mut bend = bend;
    for _ in 0..steps {
        std::mem::swap(&mut current, &mut previous);
        for index in 0..current.len() {
            current[index] = previous[index] + velocity[index] * dt + gravity * (dt * dt);
        }
        reset_lambdas(&mut structural);
        reset_lambdas(&mut shear);
        reset_lambdas(&mut bend);
        for _ in 0..config.iterations {
            solve_springs(&mut current, &inverse_masses, &mut structural);
            solve_springs(&mut current, &inverse_masses, &mut shear);
            solve_springs(&mut current, &inverse_masses, &mut bend);
        }
        for index in 0..current.len() {
            if inverse_masses[index] == 0.0 {
                current[index] = previous[index];
                velocity[index] = DVec3::ZERO;
            } else {
                velocity[index] = ((current[index] - previous[index]) / dt) * damping;
            }
        }
        project_collisions(
            &mut current,
            &previous,
            &mut velocity,
            &inverse_masses,
            &colliders,
            config.collision_distance,
        )?;
        if config.use_self_collision {
            project_self_collisions(
                mesh,
                &mut current,
                &inverse_masses,
                config.self_collision_distance,
            );
        }
        for index in 0..current.len() {
            if !current[index].is_finite() || !velocity[index].is_finite() {
                return Err(invalid("cloth simulation produced a non-finite position"));
            }
        }
    }
    Ok(current)
}

struct Config {
    substeps: usize,
    iterations: usize,
    structural_stiffness: f64,
    shear_stiffness: f64,
    bend_stiffness: f64,
    drag: f64,
    pin_group: Option<String>,
    mass: f64,
    collision_distance: f64,
    use_self_collision: bool,
    self_collision_distance: f64,
}

impl Config {
    fn parse(settings: &Value) -> Result<Self> {
        if !settings.is_object() {
            return Err(invalid("cloth settings must be an object"));
        }
        let quality = if settings.get("quality").is_some() {
            Some(integer_setting(settings, "quality", 4, 1, MAX_SUBSTEPS)?)
        } else {
            None
        };
        let structural_stiffness = if settings.get("structural_stiffness").is_some() {
            super::unit_setting(settings, "structural_stiffness", 1.0, "cloth")?
        } else {
            super::unit_setting(settings, "stiffness", 1.0, "cloth")?
        };
        let drag_key = if settings.get("drag").is_some() {
            "drag"
        } else if settings.get("air_drag").is_some() {
            "air_drag"
        } else {
            "damping"
        };
        let mass = super::setting_number(settings, "mass", 1.0, "cloth", "positive and finite")?;
        if mass <= 0.0 {
            return Err(invalid("cloth mass must be positive"));
        }
        let collision_distance =
            super::nonnegative_setting(settings, "collision_distance", COLLISION_SLOP, "cloth")?;
        let use_self_collision = match settings.get("use_self_collision") {
            Some(value) => value
                .as_bool()
                .ok_or_else(|| invalid("cloth use_self_collision must be a boolean"))?,
            None => false,
        };
        let self_collision_distance =
            super::nonnegative_setting(settings, "self_collision_distance", 0.01, "cloth")?;
        Ok(Self {
            substeps: integer_setting(settings, "substeps", quality.unwrap_or(4), 1, MAX_SUBSTEPS)?,
            iterations: integer_setting(
                settings,
                "iterations",
                quality.unwrap_or(4).saturating_mul(16).min(MAX_ITERATIONS),
                1,
                MAX_ITERATIONS,
            )?,
            structural_stiffness,
            shear_stiffness: super::unit_setting(settings, "shear_stiffness", 1.0, "cloth")?,
            bend_stiffness: super::unit_setting(settings, "bend_stiffness", 0.25, "cloth")?,
            drag: super::nonnegative_setting(settings, drag_key, 0.01, "cloth")?,
            pin_group: optional_group(settings, "pin_group")?,
            mass,
            collision_distance,
            use_self_collision,
            self_collision_distance,
        })
    }
}

fn validate_inputs(mesh: &Mesh, frame: f64, fps: f64, gravity: DVec3) -> Result<()> {
    let topology_items = mesh
        .edges
        .len()
        .saturating_add(mesh.faces.len())
        .saturating_add(
            mesh.faces
                .iter()
                .fold(0_usize, |sum, face| sum.saturating_add(face.vertices.len())),
        );
    if mesh.vertices.len() > MAX_VERTICES || topology_items > MAX_TOPOLOGY_ITEMS {
        return Err(PotError::new(
            crate::error::ErrorCode::LimitExceeded,
            "cloth mesh exceeds the simulation topology limit",
        ));
    }
    mesh.validate()
        .map_err(|error| invalid(&format!("cloth mesh is invalid: {error}")))?;
    if !frame.is_finite() || !fps.is_finite() || fps <= 0.0 || !gravity.is_finite() {
        return Err(invalid("frame, positive fps, and gravity must be finite"));
    }
    Ok(())
}

fn prepare_colliders(colliders: &[[DVec3; 3]]) -> Result<Vec<CollisionTriangle>> {
    if colliders.len() > MAX_TOPOLOGY_ITEMS {
        return Err(PotError::new(
            crate::error::ErrorCode::LimitExceeded,
            "cloth collider triangles exceed the simulation limit",
        ));
    }
    let mut prepared = Vec::with_capacity(colliders.len());
    for triangle in colliders {
        if triangle.iter().any(|vertex| !vertex.is_finite()) {
            return Err(invalid(
                "cloth collider triangle coordinates must be finite",
            ));
        }
        let [origin, b, c] = *triangle;
        let first = c - origin;
        let second = b - origin;
        let cross = second.cross(first);
        let length_squared = cross.length_squared();
        if !cross.is_finite() || !length_squared.is_finite() {
            return Err(invalid("cloth collider triangle area must be finite"));
        }
        if length_squared <= DISTANCE_EPSILON * DISTANCE_EPSILON {
            continue;
        }
        let first_dot = first.dot(first);
        let second_dot = second.dot(second);
        let cross_dot = first.dot(second);
        let denominator = first_dot * second_dot - cross_dot * cross_dot;
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(invalid("cloth collider triangle area must be finite"));
        }
        prepared.push(CollisionTriangle {
            origin,
            normal: cross / length_squared.sqrt(),
            first,
            second,
            first_dot,
            second_dot,
            cross_dot,
            denominator,
        });
    }
    Ok(prepared)
}

fn project_collisions(
    positions: &mut [DVec3],
    previous: &[DVec3],
    velocities: &mut [DVec3],
    inverse_masses: &[f64],
    colliders: &[CollisionTriangle],
    collision_distance: f64,
) -> Result<()> {
    for index in 0..positions.len() {
        if inverse_masses[index] == 0.0 {
            continue;
        }
        if !positions[index].is_finite() {
            return Err(invalid("cloth simulation produced a non-finite position"));
        }
        for triangle in colliders {
            let start_signed = (previous[index] - triangle.origin).dot(triangle.normal);
            let end_signed = (positions[index] - triangle.origin).dot(triangle.normal);
            if !start_signed.is_finite() || !end_signed.is_finite() {
                return Err(invalid("cloth collision distance became non-finite"));
            }
            let side = if start_signed > 0.0 {
                1.0
            } else if start_signed < 0.0 || end_signed > 0.0 {
                -1.0
            } else if end_signed < 0.0 {
                1.0
            } else {
                continue;
            };
            if side * end_signed > 0.0 {
                continue;
            }
            let denominator = start_signed - end_signed;
            if !denominator.is_finite() {
                return Err(invalid("cloth collision distance became non-finite"));
            }
            if denominator == 0.0 {
                continue;
            }
            let fraction = start_signed / denominator;
            if !fraction.is_finite() {
                return Err(invalid("cloth collision fraction became non-finite"));
            }
            let hit = previous[index] + (positions[index] - previous[index]) * fraction;
            if !hit.is_finite() {
                return Err(invalid("cloth collision position became non-finite"));
            }
            if !inside_collision_triangle(hit, triangle) {
                continue;
            }
            positions[index] += triangle.normal * (side * collision_distance - end_signed);
            let contact_normal = triangle.normal * side;
            let inward_speed = velocities[index].dot(contact_normal);
            if inward_speed < 0.0 {
                velocities[index] -= contact_normal * inward_speed;
            }
        }
    }
    Ok(())
}

fn project_self_collisions(
    mesh: &Mesh,
    positions: &mut [DVec3],
    inverse_masses: &[f64],
    minimum_distance: f64,
) {
    if minimum_distance <= 0.0 {
        return;
    }
    let mut adjacent = BTreeSet::new();
    for edge in &mesh.edges {
        adjacent.insert(edge_key(edge.vertices[0], edge.vertices[1]));
    }
    for first in 0..positions.len() {
        if inverse_masses[first] == 0.0 {
            continue;
        }
        for second in first + 1..positions.len() {
            if inverse_masses[second] == 0.0
                || adjacent.contains(&edge_key(mesh.vertices[first].id, mesh.vertices[second].id))
            {
                continue;
            }
            let offset = positions[second] - positions[first];
            let distance = offset.length();
            if !distance.is_finite() || distance >= minimum_distance {
                continue;
            }
            let inverse_sum = inverse_masses[first] + inverse_masses[second];
            if inverse_sum <= 0.0 {
                continue;
            }
            let normal = if distance > DISTANCE_EPSILON {
                offset / distance
            } else {
                DVec3::X
            };
            let correction = normal * (minimum_distance - distance);
            positions[first] -= correction * (inverse_masses[first] / inverse_sum);
            positions[second] += correction * (inverse_masses[second] / inverse_sum);
        }
    }
}

fn inside_collision_triangle(point: DVec3, triangle: &CollisionTriangle) -> bool {
    let offset = point - triangle.origin;
    let first_offset = triangle.first.dot(offset);
    let second_offset = triangle.second.dot(offset);
    let u = (triangle.second_dot * first_offset - triangle.cross_dot * second_offset)
        / triangle.denominator;
    let v = (triangle.first_dot * second_offset - triangle.cross_dot * first_offset)
        / triangle.denominator;
    crate::geom::point_in_triangle_barycentric(u, v, 1.0e-10)
}

fn load_weights(
    mesh: &Mesh,
    group_name: Option<&str>,
    ids: &BTreeMap<u32, usize>,
) -> Result<Vec<f64>> {
    let mut weights = vec![0.0; mesh.vertices.len()];
    let Some(group_name) = group_name else {
        return Ok(weights);
    };
    let groups = mesh
        .attributes
        .get("vertex_groups")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("cloth pin group was requested but vertex_groups is missing"))?;
    let group = groups
        .get(group_name)
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("cloth pin group does not exist or is not an object"))?;
    if group.len() > MAX_VERTICES {
        return Err(PotError::new(
            crate::error::ErrorCode::LimitExceeded,
            "cloth pin group exceeds the simulation weight limit",
        ));
    }
    for (id, value) in group {
        let weight = value
            .as_f64()
            .filter(|weight| weight.is_finite() && (0.0..=1.0).contains(weight))
            .ok_or_else(|| invalid("cloth pin weights must be finite numbers in [0, 1]"))?;
        let parsed_id = id
            .strip_prefix('v')
            .unwrap_or(id)
            .parse::<u32>()
            .map_err(|_| invalid("cloth pin group keys must be vertex ID strings"))?;
        if let Some(index) = ids.get(&parsed_id) {
            weights[*index] = weight;
        }
    }
    Ok(weights)
}

fn make_springs(
    keys: &BTreeSet<(u32, u32)>,
    ids: &BTreeMap<u32, usize>,
    positions: &[DVec3],
    stiffness: f64,
) -> Result<Vec<Spring>> {
    let alpha = if stiffness == 0.0 {
        f64::INFINITY
    } else {
        (1.0 - stiffness) / stiffness
    };
    let mut springs = Vec::with_capacity(keys.len());
    for (a_id, b_id) in keys {
        let a = ids[a_id];
        let b = ids[b_id];
        let rest_length = positions[a].distance(positions[b]);
        if !rest_length.is_finite() {
            return Err(invalid("cloth spring rest length is not finite"));
        }
        springs.push(Spring {
            a,
            b,
            rest_length,
            alpha,
            lambda: 0.0,
        });
    }
    Ok(springs)
}

fn reset_lambdas(springs: &mut [Spring]) {
    for spring in springs {
        spring.lambda = 0.0;
    }
}

fn solve_springs(positions: &mut [DVec3], inverse_masses: &[f64], springs: &mut [Spring]) {
    for spring in springs {
        if spring.alpha.is_infinite() {
            continue;
        }
        let delta = positions[spring.a] - positions[spring.b];
        let distance = delta.length();
        let weight = inverse_masses[spring.a] + inverse_masses[spring.b];
        if distance <= DISTANCE_EPSILON || weight == 0.0 {
            continue;
        }
        let constraint = distance - spring.rest_length;
        let delta_lambda = (-constraint - spring.alpha * spring.lambda) / (weight + spring.alpha);
        spring.lambda += delta_lambda;
        let correction = delta * (delta_lambda / distance);
        positions[spring.a] += correction * inverse_masses[spring.a];
        positions[spring.b] -= correction * inverse_masses[spring.b];
    }
}

fn opposite_vertex(face: &[u32], edge: (u32, u32)) -> Option<u32> {
    face.iter()
        .copied()
        .find(|vertex| *vertex != edge.0 && *vertex != edge.1)
}

fn check_work(
    vertices: usize,
    constraints: usize,
    collider_triangles: usize,
    iterations: usize,
    steps: usize,
) -> Result<()> {
    let per_step = constraints
        .checked_mul(iterations)
        .and_then(|work| work.checked_add(vertices))
        .and_then(|work| work.checked_add(vertices.checked_mul(collider_triangles)?))
        .unwrap_or(usize::MAX);
    if per_step.saturating_mul(steps) > MAX_WORK {
        return Err(PotError::new(
            crate::error::ErrorCode::LimitExceeded,
            "cloth simulation exceeds the bounded solver work limit",
        ));
    }
    Ok(())
}

fn integer_setting(
    settings: &Value,
    key: &str,
    default: usize,
    minimum: usize,
    maximum: usize,
) -> Result<usize> {
    let Some(value) = settings.get(key) else {
        return Ok(default);
    };
    let value = value
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| (*value >= minimum) && (*value <= maximum))
        .ok_or_else(|| {
            invalid(&format!(
                "cloth {key} must be an integer in {minimum}..={maximum}"
            ))
        })?;
    Ok(value)
}

fn optional_group(settings: &Value, key: &str) -> Result<Option<String>> {
    let Some(value) = settings.get(key) else {
        return Ok(None);
    };
    value
        .as_str()
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .map(Some)
        .ok_or_else(|| invalid(&format!("cloth {key} must be a nonempty string")))
}

fn invalid(message: &str) -> PotError {
    PotError::invalid_argument(message)
}
fn limit(message: &str) -> PotError {
    PotError::new(crate::error::ErrorCode::LimitExceeded, message)
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::unwrap_used,
        reason = "solver tests use validated mesh fixtures"
    )]
    use glam::DVec3;
    use serde_json::json;

    use crate::geom::Mesh;

    use super::simulate;

    #[test]
    fn gravity_moves_unpinned_cloth_but_keeps_pins_at_rest() {
        let mut mesh = Mesh::from_positions_and_faces(
            vec![DVec3::ZERO, DVec3::X, DVec3::Y, DVec3::ONE],
            vec![vec![0, 1, 3, 2]],
        )
        .unwrap();
        mesh.attributes
            .insert("vertex_groups".to_owned(), json!({"fixed": {"0": 1.0}}));
        let result = simulate(
            &mesh,
            &json!({"pin_group": "fixed", "iterations": 12}),
            1.0,
            24.0,
            DVec3::new(0.0, 0.0, -9.81),
            &[],
        )
        .unwrap();
        assert_eq!(result[0], mesh.vertices[0].co);
        assert!(result[1].z < mesh.vertices[1].co.z);
        assert_eq!(
            simulate(
                &mesh,
                &json!({"pin_group": "fixed"}),
                1.0,
                24.0,
                DVec3::ZERO,
                &[],
            )
            .unwrap(),
            mesh.vertices
                .iter()
                .map(|vertex| vertex.co)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn cloth_vertices_do_not_cross_two_sided_triangle_colliders() {
        let point =
            Mesh::from_positions_and_faces(vec![DVec3::new(0.0, 0.0, 0.001)], Vec::new()).unwrap();
        let colliders = [[
            DVec3::new(-2.0, -2.0, 0.0),
            DVec3::new(2.0, -2.0, 0.0),
            DVec3::new(0.0, 2.0, 0.0),
        ]];
        let result = simulate(
            &point,
            &json!({}),
            1.0,
            24.0,
            DVec3::new(0.0, 0.0, -10_000.0),
            &colliders,
        )
        .unwrap();
        assert!(result[0].z > 0.0);
    }
}
