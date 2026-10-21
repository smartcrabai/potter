//! Deterministic local-space soft-body simulation with edge constraints and Blender goal springs.
//!
//! Settings use Blender's `SoftBodySettings` names. How each honoured quantity maps onto this
//! solver:
//!
//! - `mass` is the mass of each vertex.
//! - `friction` is Blender's damping constant `kd = 0.01 * friction` (`softbody.cc`), applied as
//!   the force `-kd * v`. The velocity therefore decays at the rate `kd / mass` per second.
//! - `use_goal`, `goal_spring`, `goal_default`, `goal_min`, `goal_max`, and `vertex_group_goal`
//!   give each vertex the goal weight `g = (goal_min + goal * |goal_max - goal_min|)^4`, where
//!   `goal` is `goal_default` times the group weight, and the weight is zero for a vertex outside
//!   the group (`softbody.cc`: `bp->goal = defgoal`, then `*= group weight`). Without a group,
//!   `goal` is `goal_default` for every vertex. A goal of at least 0.999 pins the vertex.
//!   Otherwise the vertex is pulled toward its input position by the spring
//!   `ks = 1 / (1 - g * goal_spring) - 1`.
//! - `use_edges` enables the edge constraints, which use Potter's internal `EDGE_STIFFNESS`.
//!   Blender's `pull` and `push` springs are not mapped, so non-default values are refused.
//! - `speed` scales the solver time (`softbody.cc`).
//! - Collisions keep a vertex `thickness_outer` in front of a collider and `thickness_inner`
//!   behind it. `use_culling` limits collisions to the front side. The inward normal velocity is
//!   removed. Blender's bounce and collision damping are not modelled.
//! - Potter-native `substeps` and `iterations` set the solver resolution. Blender's adaptive
//!   `step_min` and `step_max` are not mapped.

use std::collections::{BTreeMap, BTreeSet};

use glam::DVec3;
use serde_json::Value;

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{Mesh, edge_key},
    physics_support,
};

use super::ColliderTriangle;

const MAX_VERTICES: usize = 100_000;
const MAX_TOPOLOGY_ITEMS: usize = 1_000_000;
const MAX_SUBSTEPS: usize = 32;
const MAX_ITERATIONS: usize = 64;
const MAX_WORK: usize = 50_000_000;
const DISTANCE_EPSILON: f64 = 1.0e-14;
/// Goal weight at which a vertex is pinned (`SOFTGOALSNAP` in Blender).
const PIN_THRESHOLD: f64 = 0.999;
/// Potter's internal stiffness of the edge constraints, see the module documentation.
const EDGE_STIFFNESS: f64 = 0.95;
/// Blender's friction scale, `kd = 0.01 * friction` (`softbody.cc`).
const FRICTION_SCALE: f64 = 0.01;

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
    thickness_outer: f64,
    thickness_inner: f64,
    culling: bool,
}

/// Evaluate the soft-body mesh at `frame` without modifying its input.
///
/// `frame` is the solver time from the point cache start, so zero is the input state. `colliders`
/// are world-space triangles already expressed in the mesh's local frame.
pub fn simulate(
    mesh: &Mesh,
    settings: &Value,
    frame: f64,
    fps: f64,
    gravity: DVec3,
    colliders: &[ColliderTriangle],
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
    let rest = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    let goals = goal_weights(mesh, &config, &ids)?;
    let mut current = rest.clone();
    if current.is_empty() || frame == 0.0 || config.speed == 0.0 {
        return Ok(current);
    }

    let mut edge_keys = BTreeSet::new();
    for edge in &mesh.edges {
        edge_keys.insert(edge_key(edge.vertices[0], edge.vertices[1]));
    }
    let mut edges = if config.use_edges {
        make_edges(&edge_keys, &ids, &rest, EDGE_STIFFNESS)?
    } else {
        Vec::new()
    };

    let steps = super::frame_steps(frame, config.substeps, "physics.soft_body.frame_limit")?;
    check_work(
        current.len(),
        edges.len(),
        colliders.len(),
        config.iterations,
        steps,
    )?;

    let dt = frame * config.speed / fps / steps as f64;
    if !dt.is_finite() || dt == 0.0 {
        return Err(invalid("frame time must be finite and nonzero"));
    }
    let inverse_masses = goals
        .iter()
        .map(|goal| {
            if *goal >= PIN_THRESHOLD {
                0.0
            } else {
                1.0 / config.mass
            }
        })
        .collect::<Vec<_>>();
    // Friction force `-kd * v` divided by the vertex mass gives this damping rate per second.
    let damping = (-FRICTION_SCALE * config.friction / config.mass * dt.abs()).exp();
    // Goal spring toward the input position, applied as the implicit blend
    // `beta = k dt^2 / m / (1 + k dt^2 / m)` with `k = 1 / (1 - g * goal_spring) - 1`.
    let goal_blend = goals
        .iter()
        .map(|goal| {
            if *goal <= 0.0 || *goal >= PIN_THRESHOLD {
                return 0.0;
            }
            let stiffness = 1.0 / (1.0 - goal * config.goal_spring) - 1.0;
            let scaled = stiffness * dt * dt / config.mass;
            scaled / (1.0 + scaled)
        })
        .collect::<Vec<_>>();
    let mut velocity = vec![DVec3::ZERO; current.len()];
    let mut previous = vec![DVec3::ZERO; current.len()];

    for _ in 0..steps {
        std::mem::swap(&mut current, &mut previous);
        for index in 0..current.len() {
            let predicted = previous[index] + velocity[index] * dt + gravity * (dt * dt);
            current[index] = predicted + (rest[index] - predicted) * goal_blend[index];
        }
        for edge in &mut edges {
            edge.lambda = 0.0;
        }
        for _ in 0..config.iterations {
            solve_edges(&mut current, &inverse_masses, &mut edges);
        }
        for index in 0..current.len() {
            if inverse_masses[index] == 0.0 {
                current[index] = rest[index];
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
    mass: f64,
    friction: f64,
    goal_spring: f64,
    goal_default: f64,
    use_goal: bool,
    goal_min: f64,
    goal_max: f64,
    goal_group: Option<String>,
    use_edges: bool,
    speed: f64,
}

impl Config {
    fn parse(settings: &Value) -> Result<Self> {
        if !settings.is_object() {
            return Err(invalid("soft-body settings must be an object"));
        }
        physics_support::reject_unhonoured(
            "soft_body",
            settings,
            physics_support::honoured_properties("soft_body"),
        )?;
        let mass =
            super::setting_number(settings, "mass", 1.0, "soft-body", "positive and finite")?;
        if mass <= 0.0 {
            return Err(invalid("soft-body mass must be positive"));
        }
        Ok(Self {
            substeps: integer_setting(settings, "substeps", 4, 1, MAX_SUBSTEPS)?,
            iterations: integer_setting(settings, "iterations", 8, 1, MAX_ITERATIONS)?,
            mass,
            friction: super::nonnegative_setting(settings, "friction", 0.5, "soft-body")?,
            goal_spring: super::unit_setting(settings, "goal_spring", 0.5, "soft-body")?,
            goal_default: super::unit_setting(settings, "goal_default", 0.7, "soft-body")?,
            use_goal: bool_setting(settings, "use_goal", true)?,
            goal_min: super::setting_number(
                settings,
                "goal_min",
                0.0,
                "soft-body",
                "a finite number",
            )?,
            goal_max: super::setting_number(
                settings,
                "goal_max",
                1.0,
                "soft-body",
                "a finite number",
            )?,
            goal_group: optional_group(settings, "vertex_group_goal")?,
            use_edges: bool_setting(settings, "use_edges", true)?,
            speed: super::nonnegative_setting(settings, "speed", 1.0, "soft-body")?,
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

fn prepare_colliders(colliders: &[ColliderTriangle]) -> Result<Vec<CollisionTriangle>> {
    if colliders.len() > MAX_TOPOLOGY_ITEMS {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "soft-body collider triangles exceed the simulation limit",
        ));
    }
    let mut prepared = Vec::with_capacity(colliders.len());
    for collider in colliders {
        let triangle = collider.points;
        if triangle.iter().any(|vertex| !vertex.is_finite()) {
            return Err(invalid(
                "soft-body collider triangle coordinates must be finite",
            ));
        }
        let [origin, b, c] = triangle;
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
            thickness_outer: collider.thickness_outer,
            thickness_inner: collider.thickness_inner,
            culling: collider.culling,
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
            } else if start_signed < 0.0 {
                -1.0
            } else if end_signed < 0.0 {
                1.0
            } else {
                continue;
            };
            // Culling keeps only motion from the front side of the triangle.
            if triangle.culling && side < 0.0 {
                continue;
            }
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
            // Front contacts settle `thickness_outer` in front; back contacts settle
            // `thickness_inner` behind.
            let target = if side > 0.0 {
                triangle.thickness_outer
            } else {
                -triangle.thickness_inner
            };
            positions[index] += triangle.normal * (target - end_signed);
            let contact_normal = triangle.normal * side;
            let inward_speed = velocities[index].dot(contact_normal);
            if inward_speed < 0.0 {
                velocities[index] -= contact_normal * inward_speed;
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

/// Goal weight `g = (goal_min + goal * |goal_max - goal_min|)^4` per vertex, with
/// `goal = goal_default * group weight` (`softbody.cc` `bp->goal *= BKE_defvert_find_weight`).
/// A vertex outside the goal group has weight zero. All weights are zero when `use_goal` is off.
fn goal_weights(mesh: &Mesh, config: &Config, ids: &BTreeMap<u32, usize>) -> Result<Vec<f64>> {
    if !config.use_goal {
        return Ok(vec![0.0; mesh.vertices.len()]);
    }
    let range = (config.goal_max - config.goal_min).abs();
    let grouped = config.goal_group.is_some();
    Ok(load_weights(mesh, config.goal_group.as_deref(), ids)?
        .into_iter()
        .map(|weight| {
            let goal = if grouped {
                config.goal_default * weight.unwrap_or(0.0)
            } else {
                config.goal_default
            };
            (config.goal_min + goal * range).powi(4)
        })
        .collect())
}

/// Per-vertex group weights; `None` marks a vertex outside the group.
fn load_weights(
    mesh: &Mesh,
    group_name: Option<&str>,
    ids: &BTreeMap<u32, usize>,
) -> Result<Vec<Option<f64>>> {
    let mut weights = vec![None; mesh.vertices.len()];
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
            weights[*index] = Some(weight);
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
    collider_triangles: usize,
    iterations: usize,
    steps: usize,
) -> Result<()> {
    let iteration_work = vertices
        .checked_mul(3)
        .and_then(|work| work.checked_add(edges))
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

fn bool_setting(settings: &Value, key: &str, default: bool) -> Result<bool> {
    settings.get(key).map_or(Ok(default), |value| {
        value
            .as_bool()
            .ok_or_else(|| invalid(&format!("soft-body {key} must be a boolean")))
    })
}

fn optional_group(settings: &Value, key: &str) -> Result<Option<String>> {
    let Some(value) = settings.get(key) else {
        return Ok(None);
    };
    if value.as_str() == Some("") {
        return Ok(None);
    }
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
    use crate::sim::ColliderTriangle;

    use super::simulate;

    fn tetrahedron() -> Mesh {
        Mesh::from_positions_and_faces(
            vec![
                DVec3::new(1.0, 1.0, 1.0),
                DVec3::new(-1.0, -1.0, 1.0),
                DVec3::new(-1.0, 1.0, -1.0),
                DVec3::new(1.0, -1.0, -1.0),
            ],
            vec![vec![0, 2, 1], vec![0, 1, 3], vec![0, 3, 2], vec![1, 2, 3]],
        )
        .unwrap()
    }

    fn collider(points: [DVec3; 3]) -> ColliderTriangle {
        ColliderTriangle {
            points,
            thickness_outer: 0.0,
            thickness_inner: 0.0,
            culling: false,
            friction: 0.0,
        }
    }

    #[test]
    fn open_mesh_falls_under_gravity_and_keeps_edges_finite() {
        let open = Mesh::from_positions_and_faces(
            vec![DVec3::ZERO, DVec3::X, DVec3::Y],
            vec![vec![0, 1, 2]],
        )
        .unwrap();
        let result = simulate(
            &open,
            &json!({"substeps": 2, "iterations": 16, "use_goal": false}),
            1.0,
            24.0,
            DVec3::new(0.0, 0.0, -9.81),
            &[],
        )
        .unwrap();
        assert!(result.iter().all(|position| position.is_finite()));
        assert!(result[0].z < open.vertices[0].co.z);
    }

    #[test]
    fn full_goal_weight_pins_grouped_vertices() {
        let mut closed = tetrahedron();
        closed
            .attributes
            .insert("vertex_groups".to_owned(), json!({"fixed": {"0": 1.0}}));
        let result = simulate(
            &closed,
            &json!({"substeps": 2, "iterations": 16, "vertex_group_goal": "fixed", "goal_default": 1.0}),
            1.0,
            24.0,
            DVec3::new(0.0, 0.0, -9.81),
            &[],
        )
        .unwrap();
        assert_eq!(result[0], closed.vertices[0].co);
    }

    #[test]
    fn goal_weight_is_default_times_group_weight_and_zero_outside_the_group() {
        let mut closed = tetrahedron();
        closed.attributes.insert(
            "vertex_groups".to_owned(),
            json!({"fixed": {"0": 1.0, "1": 0.5}}),
        );
        let config =
            Config::parse(&json!({"vertex_group_goal": "fixed", "goal_default": 0.7})).unwrap();
        let ids = closed
            .vertices
            .iter()
            .enumerate()
            .map(|(index, vertex)| (vertex.id, index))
            .collect::<BTreeMap<_, _>>();
        let goals = goal_weights(&closed, &config, &ids).unwrap();
        // Blender: `goal = defgoal * weight`, `goal = 0` outside the group, then `goal^4`.
        let expected = [0.7_f64.powi(4), (0.7_f64 * 0.5).powi(4), 0.0, 0.0];
        for (actual, expected) in goals.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 1.0e-12,
                "{actual} vs {expected}"
            );
        }
    }

    #[test]
    fn goal_springs_hold_vertices_closer_than_a_free_body() {
        let body = tetrahedron();
        let settings = |use_goal: bool| json!({"substeps": 2, "iterations": 16, "use_goal": use_goal, "goal_spring": 1.0});
        let gravity = DVec3::new(0.0, 0.0, -9.81);
        let held = simulate(&body, &settings(true), 1.0, 24.0, gravity, &[]).unwrap();
        let free = simulate(&body, &settings(false), 1.0, 24.0, gravity, &[]).unwrap();
        assert!(held[0].z > free[0].z);
    }

    #[test]
    fn soft_body_vertices_do_not_cross_two_sided_triangle_colliders() {
        let point =
            Mesh::from_positions_and_faces(vec![DVec3::new(0.0, 0.0, 0.001)], Vec::new()).unwrap();
        let colliders = [collider([
            DVec3::new(-2.0, -2.0, 0.0),
            DVec3::new(2.0, -2.0, 0.0),
            DVec3::new(0.0, 2.0, 0.0),
        ])];
        let result = simulate(
            &point,
            &json!({"use_goal": false}),
            1.0,
            24.0,
            DVec3::new(0.0, 0.0, -10_000.0),
            &colliders,
        )
        .unwrap();
        // A vertex stops on the surface (thickness 0), so "does not cross" is z >= 0 up to rounding.
        assert!(
            result[0].z >= -1.0e-12,
            "vertex crossed the collider: z={}",
            result[0].z
        );
    }

    #[test]
    fn culled_soft_body_collider_does_not_block_its_back_side() {
        let point =
            Mesh::from_positions_and_faces(vec![DVec3::new(0.0, 0.0, -0.001)], Vec::new()).unwrap();
        let mut culled = collider([
            DVec3::new(-2.0, -2.0, 0.0),
            DVec3::new(2.0, -2.0, 0.0),
            DVec3::new(0.0, 2.0, 0.0),
        ]);
        culled.culling = true;
        let result = simulate(
            &point,
            &json!({"use_goal": false}),
            1.0,
            24.0,
            DVec3::new(0.0, 0.0, -10_000.0),
            &[culled],
        )
        .unwrap();
        assert!(result[0].z < -0.001);
    }
}
