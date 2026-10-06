//! Deterministic local-space soft-body simulation with edge and closed-volume constraints.

use std::collections::{BTreeMap, BTreeSet};

use glam::DVec3;
use serde_json::Value;

use crate::{
    error::{ErrorCode, PotError, Result},
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
struct EdgeConstraint {
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

/// Evaluate the soft-body mesh at `frame` without modifying its input. `colliders` are
/// two-sided triangles expressed in the same local coordinate frame as the mesh.
///
/// Settings: `substeps`, `iterations`, `edge_stiffness`, `volume_stiffness`,
/// `goal_stiffness`, `restitution`, `drag`, and optional `goal_group`. Restitution
/// defaults to 0.25. Stiffnesses, restitution, and group weights are in `[0, 1]`;
/// drag is nonnegative.
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
    let ids = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<BTreeMap<_, _>>();
    let goals = load_weights(mesh, config.goal_group.as_deref(), &ids)?;
    let mut current = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    if current.is_empty() || frame == 0.0 {
        return Ok(current);
    }

    let mut edge_keys = BTreeSet::new();
    let mut face_edge_counts = BTreeMap::<(u32, u32), usize>::new();
    for edge in &mesh.edges {
        edge_keys.insert(edge_key(edge.vertices[0], edge.vertices[1]));
    }
    let mut triangles = Vec::new();
    for face in &mesh.faces {
        for index in 0..face.vertices.len() {
            let key = edge_key(
                face.vertices[index],
                face.vertices[(index + 1) % face.vertices.len()],
            );
            *face_edge_counts.entry(key).or_default() += 1;
        }
        for index in 1..face.vertices.len().saturating_sub(1) {
            triangles.push([
                ids[&face.vertices[0]],
                ids[&face.vertices[index]],
                ids[&face.vertices[index + 1]],
            ]);
        }
    }
    let closed = !triangles.is_empty() && face_edge_counts.values().all(|count| *count == 2);
    let mut edges = make_edges(&edge_keys, &ids, &current, config.edge_stiffness)?;
    let mut volume_triangles = if closed { triangles } else { Vec::new() };
    let rest_volume = signed_volume(&current, &volume_triangles);
    if !rest_volume.is_finite() {
        return Err(invalid("soft-body rest volume is not finite"));
    }
    if rest_volume == 0.0 {
        volume_triangles.clear();
    }

    let steps = super::frame_steps(
        frame,
        config.substeps,
        "soft-body frame exceeds the bounded simulation step limit",
    )?;
    check_work(
        current.len(),
        edges.len(),
        volume_triangles.len(),
        colliders.len(),
        config.iterations,
        steps,
    )?;

    let inverse_masses = goals
        .iter()
        .map(|weight| (1.0 - *weight) / config.mass)
        .collect::<Vec<_>>();
    let mut velocity = vec![DVec3::ZERO; current.len()];
    let mut previous = vec![DVec3::ZERO; current.len()];
    let dt = frame / fps / steps as f64;
    if !dt.is_finite() || dt == 0.0 {
        return Err(invalid("frame time must be finite and nonzero"));
    }
    let damping = (-config.drag * dt.abs()).exp();
    let per_step_goal = if config.goal_stiffness >= 1.0 {
        1.0
    } else {
        1.0 - (1.0 - config.goal_stiffness).powf(frame.abs() / steps as f64)
    };
    let volume_alpha = compliance(config.volume_stiffness);
    let mut volume_gradients = if volume_triangles.is_empty() {
        Vec::new()
    } else {
        vec![DVec3::ZERO; current.len()]
    };

    for _ in 0..steps {
        let mut volume_lambda = 0.0;
        std::mem::swap(&mut current, &mut previous);
        for index in 0..current.len() {
            current[index] = previous[index] + velocity[index] * dt + gravity * (dt * dt);
            if goals[index] > 0.0 {
                let position = current[index];
                current[index] +=
                    (mesh.vertices[index].co - position) * (per_step_goal * goals[index]);
            }
        }
        for edge in &mut edges {
            edge.lambda = 0.0;
        }
        for _ in 0..config.iterations {
            solve_edges(&mut current, &inverse_masses, &mut edges);
            if !volume_triangles.is_empty()
                && config.volume_stiffness > 0.0
                && !volume_alpha.is_infinite()
            {
                solve_volume(
                    &mut current,
                    &inverse_masses,
                    &volume_triangles,
                    &mut volume_gradients,
                    rest_volume,
                    volume_alpha,
                    &mut volume_lambda,
                )?;
            }
        }
        for index in 0..current.len() {
            if inverse_masses[index] == 0.0 {
                current[index] = mesh.vertices[index].co;
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
            config.restitution,
        )?;
        for index in 0..current.len() {
            if !current[index].is_finite() || !velocity[index].is_finite() {
                return Err(invalid(
                    "soft-body simulation produced a non-finite position",
                ));
            }
        }
    }
    Ok(current)
}

struct Config {
    substeps: usize,
    iterations: usize,
    edge_stiffness: f64,
    volume_stiffness: f64,
    goal_stiffness: f64,
    restitution: f64,
    drag: f64,
    goal_group: Option<String>,
    mass: f64,
}

impl Config {
    fn parse(settings: &Value) -> Result<Self> {
        if !settings.is_object() {
            return Err(invalid("soft-body settings must be an object"));
        }
        let quality = if settings.get("quality").is_some() {
            Some(integer_setting(settings, "quality", 4, 1, MAX_SUBSTEPS)?)
        } else {
            None
        };
        let edge_stiffness = if settings.get("edge_stiffness").is_some() {
            super::unit_setting(settings, "edge_stiffness", 0.95, "soft-body")?
        } else {
            super::unit_setting(settings, "stiffness", 0.95, "soft-body")?
        };
        let goal_stiffness = if settings.get("goal_stiffness").is_some() {
            super::unit_setting(settings, "goal_stiffness", 1.0, "soft-body")?
        } else {
            super::unit_setting(settings, "goal_strength", 1.0, "soft-body")?
        };
        let drag_key = if settings.get("drag").is_some() {
            "drag"
        } else if settings.get("air_drag").is_some() {
            "air_drag"
        } else {
            "damping"
        };
        let mass =
            super::setting_number(settings, "mass", 1.0, "soft-body", "positive and finite")?;
        if mass <= 0.0 {
            return Err(invalid("soft-body mass must be positive"));
        }
        Ok(Self {
            substeps: integer_setting(settings, "substeps", quality.unwrap_or(4), 1, MAX_SUBSTEPS)?,
            iterations: integer_setting(
                settings,
                "iterations",
                quality.unwrap_or(8).min(MAX_ITERATIONS),
                1,
                MAX_ITERATIONS,
            )?,
            edge_stiffness,
            volume_stiffness: super::unit_setting(settings, "volume_stiffness", 1.0, "soft-body")?,
            goal_stiffness,
            restitution: super::unit_setting(settings, "restitution", 0.25, "soft-body")?,
            drag: super::nonnegative_setting(settings, drag_key, 0.01, "soft-body")?,
            goal_group: optional_group(settings, "goal_group")?,
            mass,
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
            ErrorCode::LimitExceeded,
            "soft-body mesh exceeds the simulation topology limit",
        ));
    }
    mesh.validate()
        .map_err(|error| invalid(&format!("soft-body mesh is invalid: {error}")))?;
    if !frame.is_finite() || !fps.is_finite() || fps <= 0.0 || !gravity.is_finite() {
        return Err(invalid("frame, positive fps, and gravity must be finite"));
    }
    Ok(())
}

fn prepare_colliders(colliders: &[[DVec3; 3]]) -> Result<Vec<CollisionTriangle>> {
    if colliders.len() > MAX_TOPOLOGY_ITEMS {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "soft-body collider triangles exceed the simulation limit",
        ));
    }
    let mut prepared = Vec::with_capacity(colliders.len());
    for triangle in colliders {
        if triangle.iter().any(|vertex| !vertex.is_finite()) {
            return Err(invalid(
                "soft-body collider triangle coordinates must be finite",
            ));
        }
        let [origin, b, c] = *triangle;
        let first = c - origin;
        let second = b - origin;
        let cross = second.cross(first);
        let length_squared = cross.length_squared();
        if !cross.is_finite() || !length_squared.is_finite() {
            return Err(invalid("soft-body collider triangle area must be finite"));
        }
        if length_squared <= DISTANCE_EPSILON * DISTANCE_EPSILON {
            continue;
        }
        let first_dot = first.dot(first);
        let second_dot = second.dot(second);
        let cross_dot = first.dot(second);
        let denominator = first_dot * second_dot - cross_dot * cross_dot;
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(invalid("soft-body collider triangle area must be finite"));
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
    restitution: f64,
) -> Result<()> {
    for index in 0..positions.len() {
        if inverse_masses[index] == 0.0 {
            continue;
        }
        if !positions[index].is_finite() {
            return Err(invalid(
                "soft-body simulation produced a non-finite position",
            ));
        }
        for triangle in colliders {
            let start_signed = (previous[index] - triangle.origin).dot(triangle.normal);
            let end_signed = (positions[index] - triangle.origin).dot(triangle.normal);
            if !start_signed.is_finite() || !end_signed.is_finite() {
                return Err(invalid("soft-body collision distance became non-finite"));
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
                return Err(invalid("soft-body collision distance became non-finite"));
            }
            if denominator == 0.0 {
                continue;
            }
            let fraction = start_signed / denominator;
            if !fraction.is_finite() {
                return Err(invalid("soft-body collision fraction became non-finite"));
            }
            let hit = previous[index] + (positions[index] - previous[index]) * fraction;
            if !hit.is_finite() {
                return Err(invalid("soft-body collision position became non-finite"));
            }
            if !inside_collision_triangle(hit, triangle) {
                continue;
            }
            positions[index] += triangle.normal * (side * COLLISION_SLOP - end_signed);
            let contact_normal = triangle.normal * side;
            let inward_speed = velocities[index].dot(contact_normal);
            if inward_speed < 0.0 {
                velocities[index] -= contact_normal * ((1.0 + restitution) * inward_speed);
            }
        }
    }
    Ok(())
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
        .ok_or_else(|| {
            invalid("soft-body goal group was requested but vertex_groups is missing")
        })?;
    let group = groups
        .get(group_name)
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("soft-body goal group does not exist or is not an object"))?;
    if group.len() > MAX_VERTICES {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "soft-body goal group exceeds the simulation weight limit",
        ));
    }
    for (id, value) in group {
        let weight = value
            .as_f64()
            .filter(|weight| weight.is_finite() && (0.0..=1.0).contains(weight))
            .ok_or_else(|| invalid("soft-body goal weights must be finite numbers in [0, 1]"))?;
        let parsed_id = id
            .strip_prefix('v')
            .unwrap_or(id)
            .parse::<u32>()
            .map_err(|_| invalid("soft-body goal group keys must be vertex ID strings"))?;
        if let Some(index) = ids.get(&parsed_id) {
            weights[*index] = weight;
        }
    }
    Ok(weights)
}

fn make_edges(
    keys: &BTreeSet<(u32, u32)>,
    ids: &BTreeMap<u32, usize>,
    positions: &[DVec3],
    stiffness: f64,
) -> Result<Vec<EdgeConstraint>> {
    let alpha = compliance(stiffness);
    let mut edges = Vec::with_capacity(keys.len());
    for (a_id, b_id) in keys {
        let a = ids[a_id];
        let b = ids[b_id];
        let rest_length = positions[a].distance(positions[b]);
        if !rest_length.is_finite() {
            return Err(invalid("soft-body edge rest length is not finite"));
        }
        edges.push(EdgeConstraint {
            a,
            b,
            rest_length,
            alpha,
            lambda: 0.0,
        });
    }
    Ok(edges)
}

fn solve_edges(positions: &mut [DVec3], inverse_masses: &[f64], edges: &mut [EdgeConstraint]) {
    for edge in edges {
        if edge.alpha.is_infinite() {
            continue;
        }
        let delta = positions[edge.a] - positions[edge.b];
        let distance = delta.length();
        let weight = inverse_masses[edge.a] + inverse_masses[edge.b];
        if distance <= DISTANCE_EPSILON || weight == 0.0 {
            continue;
        }
        let constraint = distance - edge.rest_length;
        let delta_lambda = (-constraint - edge.alpha * edge.lambda) / (weight + edge.alpha);
        edge.lambda += delta_lambda;
        let correction = delta * (delta_lambda / distance);
        positions[edge.a] += correction * inverse_masses[edge.a];
        positions[edge.b] -= correction * inverse_masses[edge.b];
    }
}

fn solve_volume(
    positions: &mut [DVec3],
    inverse_masses: &[f64],
    triangles: &[[usize; 3]],
    gradients: &mut [DVec3],
    rest_volume: f64,
    alpha: f64,
    lambda: &mut f64,
) -> Result<()> {
    let current_volume = signed_volume_and_gradients(positions, triangles, gradients);
    if !current_volume.is_finite() || gradients.iter().any(|gradient| !gradient.is_finite()) {
        return Err(invalid("soft-body volume constraint became non-finite"));
    }
    let constraint = current_volume - rest_volume;
    let denominator = gradients
        .iter()
        .zip(inverse_masses)
        .map(|(gradient, weight)| gradient.length_squared() * weight)
        .sum::<f64>()
        + alpha;
    if !denominator.is_finite() {
        return Err(invalid("soft-body volume constraint became non-finite"));
    }
    if denominator <= 0.0 {
        return Ok(());
    }
    let delta_lambda = (-constraint - alpha * *lambda) / denominator;
    *lambda += delta_lambda;
    for index in 0..positions.len() {
        positions[index] += gradients[index] * (inverse_masses[index] * delta_lambda);
    }
    Ok(())
}

fn signed_volume(positions: &[DVec3], triangles: &[[usize; 3]]) -> f64 {
    triangles
        .iter()
        .map(|triangle| {
            positions[triangle[0]].dot(positions[triangle[1]].cross(positions[triangle[2]])) / 6.0
        })
        .sum()
}

fn signed_volume_and_gradients(
    positions: &[DVec3],
    triangles: &[[usize; 3]],
    gradients: &mut [DVec3],
) -> f64 {
    gradients.fill(DVec3::ZERO);
    let mut volume = 0.0;
    for triangle in triangles {
        let [a, b, c] = *triangle;
        let first = positions[a];
        let second = positions[b];
        let third = positions[c];
        volume += first.dot(second.cross(third)) / 6.0;
        gradients[a] += second.cross(third) / 6.0;
        gradients[b] += third.cross(first) / 6.0;
        gradients[c] += first.cross(second) / 6.0;
    }
    volume
}

fn compliance(stiffness: f64) -> f64 {
    if stiffness == 0.0 {
        f64::INFINITY
    } else {
        (1.0 - stiffness) / stiffness
    }
}

fn check_work(
    vertices: usize,
    edges: usize,
    triangles: usize,
    collider_triangles: usize,
    iterations: usize,
    steps: usize,
) -> Result<()> {
    let iteration_work = edges
        .checked_add(triangles)
        .and_then(|work| work.checked_add(vertices.checked_mul(3)?))
        .and_then(|work| work.checked_mul(iterations))
        .and_then(|work| work.checked_add(vertices))
        .and_then(|work| work.checked_add(vertices.checked_mul(collider_triangles)?))
        .unwrap_or(usize::MAX);
    if iteration_work.saturating_mul(steps) > MAX_WORK {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "soft-body simulation exceeds the bounded solver work limit",
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
    value
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| (*value >= minimum) && (*value <= maximum))
        .ok_or_else(|| {
            invalid(&format!(
                "soft-body {key} must be an integer in {minimum}..={maximum}"
            ))
        })
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
        .ok_or_else(|| invalid(&format!("soft-body {key} must be a nonempty string")))
}

fn invalid(message: &str) -> PotError {
    PotError::invalid_argument(message)
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
    fn closed_soft_body_keeps_volume_and_open_mesh_uses_edges() {
        let mut closed = Mesh::from_positions_and_faces(
            vec![
                DVec3::new(1.0, 1.0, 1.0),
                DVec3::new(-1.0, -1.0, 1.0),
                DVec3::new(-1.0, 1.0, -1.0),
                DVec3::new(1.0, -1.0, -1.0),
            ],
            vec![vec![0, 2, 1], vec![0, 1, 3], vec![0, 3, 2], vec![1, 2, 3]],
        )
        .unwrap();
        closed
            .attributes
            .insert("vertex_groups".to_owned(), json!({"fixed": {"0": 1.0}}));
        let settings = json!({"substeps": 2, "iterations": 16, "goal_group": "fixed"});
        let result = simulate(
            &closed,
            &settings,
            1.0,
            24.0,
            DVec3::new(0.0, 0.0, -9.81),
            &[],
        )
        .unwrap();
        let volume = |points: &[DVec3]| {
            closed
                .faces
                .iter()
                .map(|face| {
                    let origin = points[face.vertices[0] as usize];
                    (1..face.vertices.len() - 1)
                        .map(|index| {
                            origin.dot(
                                points[face.vertices[index] as usize]
                                    .cross(points[face.vertices[index + 1] as usize]),
                            ) / 6.0
                        })
                        .sum::<f64>()
                })
                .sum::<f64>()
        };
        let rest_positions = closed
            .vertices
            .iter()
            .map(|vertex| vertex.co)
            .collect::<Vec<_>>();
        assert_eq!(result[0], closed.vertices[0].co);
        assert!((volume(&result).abs() - volume(&rest_positions).abs()).abs() < 1.0e-6);

        let open = Mesh::from_positions_and_faces(
            vec![DVec3::ZERO, DVec3::X, DVec3::Y],
            vec![vec![0, 1, 2]],
        )
        .unwrap();
        let open_result = simulate(
            &open,
            &json!({"substeps": 2, "iterations": 16}),
            1.0,
            24.0,
            DVec3::new(0.0, 0.0, -9.81),
            &[],
        )
        .unwrap();
        assert!(open_result.iter().all(|position| position.is_finite()));
        assert!(open_result[0].z < open.vertices[0].co.z);
    }

    #[test]
    fn soft_body_restitution_bounces_after_triangle_contact() {
        let body = Mesh::from_positions_and_faces(
            vec![
                DVec3::new(0.0, 0.0, 0.01),
                DVec3::new(0.1, 0.0, 0.01),
                DVec3::new(0.0, 0.1, 0.01),
            ],
            vec![vec![0, 1, 2]],
        )
        .unwrap();
        let colliders = [[
            DVec3::new(-2.0, -2.0, 0.0),
            DVec3::new(2.0, -2.0, 0.0),
            DVec3::new(0.0, 2.0, 0.0),
        ]];
        let result = simulate(
            &body,
            &json!({"restitution": 1.0}),
            8.0,
            100.0,
            DVec3::new(0.0, 0.0, -9.81),
            &colliders,
        )
        .unwrap();
        assert!(result.iter().all(|position| position.z > 0.001));
    }

    #[test]
    fn soft_body_vertices_do_not_cross_two_sided_triangle_colliders() {
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
