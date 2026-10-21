//! Deterministic local-space cloth simulation using XPBD distance constraints.
//!
//! Settings use Blender's `ClothSettings` names. How each honoured quantity maps onto this solver:
//!
//! - `mass` is the mass of each vertex (`cloth.cc`), not a total divided over the mesh.
//! - `air_damping` is Blender's `Cvi`. It damps velocity at the rate `0.01 * Cvi / mass`, the
//!   acceleration `-0.01 * Cvi * v / m_vertex` of `SIM_mass_spring.cc`.
//! - `tension_stiffness`, `compression_stiffness`, and `shear_stiffness` are Blender spring
//!   scalings. A spring has force constant `S / avg_spring_len`, so its XPBD compliance is
//!   `1 / (k * dt^2)`. Tension applies to stretched springs and compression to compressed ones.
//! - `bending_stiffness` is applied to Potter's bend springs with Blender's angular scale
//!   `k = S * rest_length * 0.1`. Only `bending_model` ANGULAR is accepted; Potter's bend springs
//!   approximate Blender's angular model.
//! - `vertex_group_mass` gives each vertex a goal weight `w^4` (`goal_default` outside the group).
//!   A goal of 0.999 or more pins the vertex. A smaller goal is a goal spring toward the input
//!   position with stiffness `goal * goal_spring / avg_spring_len` (`SIM_mass_spring.cc`);
//!   `goal_friction` damps the velocity along it at `0.01 * goal_friction`. `pin_stiffness` is the
//!   same field as `goal_spring`. The goal target is the input position, which is the per-frame
//!   state this solver restarts from.
//! - `quality` is the number of solver steps per frame.
//! - Collisions use Blender's epsilon model (`collision.cc`): a vertex keeps
//!   `collision_settings.distance_min + thickness_outer` from a collider. `use_culling` limits
//!   collisions to the front side, and `cloth_friction` applies Coulomb friction.

use std::collections::{BTreeMap, BTreeSet};

use glam::DVec3;
use serde_json::Value;

use crate::{
    error::{PotError, Result},
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
/// Goal weight at which a vertex is pinned (`cloth.cc`).
const PIN_THRESHOLD: f64 = 0.999;
/// Blender's angular bending scale, `k = S * rest_length * 0.1` (`SIM_mass_spring.cc`).
const ANGULAR_BEND_SCALE: f64 = 0.1;
/// Blender's air viscosity scale, `force = -0.01 * Cvi * v` (`SIM_mass_spring.cc`).
const AIR_DRAG_SCALE: f64 = 0.01;
/// Blender's friction scale, `impulse = 0.01 * cloth_friction * |normal change|` (`collision.cc`).
const CLOTH_FRICTION_SCALE: f64 = 0.01;

#[derive(Clone, Copy)]
struct Spring {
    a: usize,
    b: usize,
    rest_length: f64,
    /// XPBD compliance while the spring is stretched; infinite when it is absent.
    tension_alpha: f64,
    /// XPBD compliance while the spring is compressed; infinite when it is absent.
    compression_alpha: f64,
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
    culling: bool,
    /// Coulomb coefficient, `cloth_friction * 0.01`.
    friction: f64,
}

/// Evaluate the cloth mesh at `frame` without modifying its input.
///
/// `frame` is the solver time from the point cache start, so zero is the input state. `colliders`
/// are world-space triangles already expressed in the mesh's local frame.
///
/// Settings are Blender cloth names, plus `iterations`, a Potter solver knob. Stiffnesses are
/// Blender spring scalings; `vertex_group_mass` names the pin group; `collision_settings` carries
/// `distance_min`, `use_self_collision`, and `self_distance_min`.
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
    // The start frame returns the input unchanged. Blender's goal springs pull toward the per-frame
    // input, which equals the current state there, so partial pins need no goal force until a
    // solved frame.
    if positions.is_empty() || frame == 0.0 {
        return Ok(positions);
    }
    let goals = vertex_goals(mesh, &config, &ids)?;
    let pinned = goals.iter().map(|goal| goal.pinned).collect::<Vec<_>>();

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

    let steps = super::frame_steps(frame, config.substeps, "physics.cloth.frame_limit")?;
    let dt = frame / fps / steps as f64;
    if !dt.is_finite() || dt == 0.0 {
        return Err(invalid("frame time must be finite and nonzero"));
    }
    let average_length = average_length(&structural_keys, &ids, &positions);
    let structural = make_springs(&structural_keys, &ids, &positions, |_| {
        let force =
            |stiffness: f64| spring_compliance(stiffness / (average_length + f64::EPSILON), dt);
        (force(config.tension), force(config.compression))
    })?;
    let shear = make_springs(&shear_keys, &ids, &positions, |_| {
        let alpha = spring_compliance(config.shear / (average_length + f64::EPSILON), dt);
        (alpha, alpha)
    })?;
    let bend = make_springs(&bend_keys, &ids, &positions, |rest_length| {
        let alpha = spring_compliance(config.bending * rest_length * ANGULAR_BEND_SCALE, dt);
        (alpha, alpha)
    })?;
    // Blender's goal spring: stiffness `goal * goal_spring / avg_spring_len` toward the input position,
    // for every vertex that is not pinned and has a goal weight above zero (`cloth_calc_force`).
    // A zero `goal_spring` has no force, so no spring is built: its compliance would be infinite.
    let goal_springs = goals
        .iter()
        .enumerate()
        .filter(|(_, goal)| {
            !goal.pinned && goal.weight > f64::from(f32::EPSILON) && config.goal_spring > 0.0
        })
        .map(|(index, goal)| GoalSpring {
            index,
            alpha: spring_compliance(
                goal.weight * config.goal_spring / (average_length + f64::EPSILON),
                dt,
            ),
            lambda: 0.0,
        })
        .collect::<Vec<_>>();
    let goal_targets = positions.clone();
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
    let inverse_masses = pinned
        .iter()
        .map(|pinned| if *pinned { 0.0 } else { 1.0 / config.mass })
        .collect::<Vec<_>>();
    // Air drag is `-0.01 * Cvi * v` per unit mass, so the velocity decays at this rate per second.
    let damping = (-AIR_DRAG_SCALE * config.air_damping / config.mass * dt.abs()).exp();

    let mut structural = structural;
    let mut shear = shear;
    let mut bend = bend;
    let mut goal_springs = goal_springs;
    // Goal damping is `0.01 * goal_friction` times the velocity along the goal direction.
    let goal_damping = AIR_DRAG_SCALE * config.goal_friction;
    for _ in 0..steps {
        std::mem::swap(&mut current, &mut previous);
        for index in 0..current.len() {
            current[index] = previous[index] + velocity[index] * dt + gravity * (dt * dt);
        }
        reset_lambdas(&mut structural);
        reset_lambdas(&mut shear);
        reset_lambdas(&mut bend);
        for goal in &mut goal_springs {
            goal.lambda = 0.0;
        }
        for _ in 0..config.iterations {
            solve_springs(&mut current, &inverse_masses, &mut structural);
            solve_springs(&mut current, &inverse_masses, &mut shear);
            solve_springs(&mut current, &inverse_masses, &mut bend);
            solve_goal_springs(
                &mut current,
                &inverse_masses,
                &mut goal_springs,
                &goal_targets,
            );
        }
        for index in 0..current.len() {
            if inverse_masses[index] == 0.0 {
                current[index] = previous[index];
                velocity[index] = DVec3::ZERO;
            } else {
                velocity[index] = ((current[index] - previous[index]) / dt) * damping;
            }
        }
        if goal_damping > 0.0 {
            for goal in &goal_springs {
                let offset = current[goal.index] - goal_targets[goal.index];
                let length = offset.length();
                if length <= DISTANCE_EPSILON {
                    continue;
                }
                let direction = offset / length;
                // The force `-c * v_dir` decays the velocity along the goal direction at rate c / mass.
                let along = velocity[goal.index].dot(direction);
                let decay = 1.0 - (-goal_damping * dt.abs() / config.mass).exp();
                velocity[goal.index] -= direction * (along * decay);
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
    mass: f64,
    air_damping: f64,
    tension: f64,
    compression: f64,
    shear: f64,
    bending: f64,
    pin_group: Option<String>,
    /// Blender's `goal_default`: the goal weight of a vertex outside the pin group, when a group is set.
    goal_default: f64,
    /// Blender's `goal_spring` (`pin_stiffness` is the same DNA field): the goal spring scaling.
    goal_spring: f64,
    /// Blender's `goal_friction`: goal damping, applied as `0.01 * goal_friction`.
    goal_friction: f64,
    collision_distance: f64,
    use_self_collision: bool,
    self_collision_distance: f64,
}

impl Config {
    fn parse(settings: &Value) -> Result<Self> {
        if !settings.is_object() {
            return Err(invalid("cloth settings must be an object"));
        }
        physics_support::reject_unhonoured(
            "cloth",
            settings,
            physics_support::honoured_properties("cloth"),
        )?;
        let substeps = integer_setting(settings, "quality", 5, 1, MAX_SUBSTEPS)?;
        let bending_model = settings.get("bending_model").and_then(Value::as_str);
        if bending_model.is_some_and(|model| model != "ANGULAR") {
            return Err(physics_support::unsupported_property(
                "cloth",
                "bending_model",
            ));
        }
        let mass = super::setting_number(settings, "mass", 0.3, "cloth", "positive and finite")?;
        if mass <= 0.0 {
            return Err(invalid("cloth mass must be positive"));
        }
        let collision = settings.get("collision_settings");
        let collision_number = |key: &str, default: f64| -> Result<f64> {
            match collision {
                Some(collision) => {
                    super::nonnegative_setting(collision, key, default, "cloth collision")
                }
                None => Ok(default),
            }
        };
        let use_self_collision =
            match collision.and_then(|collision| collision.get("use_self_collision")) {
                Some(value) => value
                    .as_bool()
                    .ok_or_else(|| invalid("cloth use_self_collision must be a boolean"))?,
                None => false,
            };
        Ok(Self {
            substeps,
            iterations: integer_setting(
                settings,
                "iterations",
                substeps.saturating_mul(16).min(MAX_ITERATIONS),
                1,
                MAX_ITERATIONS,
            )?,
            mass,
            air_damping: super::nonnegative_setting(settings, "air_damping", 1.0, "cloth")?,
            tension: super::nonnegative_setting(settings, "tension_stiffness", 15.0, "cloth")?,
            compression: super::nonnegative_setting(
                settings,
                "compression_stiffness",
                15.0,
                "cloth",
            )?,
            shear: super::nonnegative_setting(settings, "shear_stiffness", 5.0, "cloth")?,
            bending: super::nonnegative_setting(settings, "bending_stiffness", 0.5, "cloth")?,
            pin_group: optional_group(settings, "vertex_group_mass")?,
            goal_default: super::nonnegative_setting(settings, "goal_default", 0.0, "cloth")?,
            goal_spring: if settings.get("goal_spring").is_some() {
                super::nonnegative_setting(settings, "goal_spring", 1.0, "cloth")?
            } else {
                super::nonnegative_setting(settings, "pin_stiffness", 1.0, "cloth")?
            },
            goal_friction: super::nonnegative_setting(settings, "goal_friction", 0.0, "cloth")?,
            collision_distance: collision_number("distance_min", 0.015)?,
            use_self_collision,
            self_collision_distance: collision_number("self_distance_min", 0.015)?,
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

fn prepare_colliders(colliders: &[ColliderTriangle]) -> Result<Vec<CollisionTriangle>> {
    if colliders.len() > MAX_TOPOLOGY_ITEMS {
        return Err(PotError::new(
            crate::error::ErrorCode::LimitExceeded,
            "cloth collider triangles exceed the simulation limit",
        ));
    }
    let mut prepared = Vec::with_capacity(colliders.len());
    for collider in colliders {
        let triangle = collider.points;
        if triangle.iter().any(|vertex| !vertex.is_finite()) {
            return Err(invalid(
                "cloth collider triangle coordinates must be finite",
            ));
        }
        let [origin, b, c] = triangle;
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
            thickness_outer: collider.thickness_outer,
            culling: collider.culling,
            friction: collider.friction * CLOTH_FRICTION_SCALE,
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
            let distance = collision_distance + triangle.thickness_outer;
            positions[index] += triangle.normal * (side * distance - end_signed);
            let contact_normal = triangle.normal * side;
            let normal_speed = velocities[index].dot(contact_normal);
            if normal_speed < 0.0 {
                // The normal velocity is removed; Coulomb friction then slows the tangential part by
                // at most `friction * |normal change|`.
                let tangent = velocities[index] - contact_normal * normal_speed;
                let tangent_speed = tangent.length();
                let reduction = (triangle.friction * -normal_speed).min(tangent_speed);
                velocities[index] = if tangent_speed > 0.0 {
                    tangent * ((tangent_speed - reduction) / tangent_speed)
                } else {
                    tangent
                };
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

/// Goal state of one vertex, from the `vertex_group_mass` weights (`cloth.cc`).
#[derive(Clone, Copy)]
struct VertexGoal {
    /// Blender's goal weight: `w^4` for a vertex in the pin group, `goal_default` outside it when a
    /// group is set, and zero without a group.
    weight: f64,
    /// Blender pins a vertex in the group whose goal reaches 0.999; it stays at its input position.
    pinned: bool,
}

fn vertex_goals(
    mesh: &Mesh,
    config: &Config,
    ids: &BTreeMap<u32, usize>,
) -> Result<Vec<VertexGoal>> {
    let Some(group_name) = config.pin_group.as_deref() else {
        return Ok(vec![
            VertexGoal {
                weight: 0.0,
                pinned: false,
            };
            mesh.vertices.len()
        ]);
    };
    let weights = load_weights(mesh, Some(group_name), ids)?;
    Ok(weights
        .into_iter()
        .map(|weight| match weight {
            Some(weight) => {
                let goal = weight.powi(4);
                VertexGoal {
                    weight: goal,
                    pinned: goal >= PIN_THRESHOLD,
                }
            }
            None => VertexGoal {
                weight: config.goal_default,
                pinned: false,
            },
        })
        .collect())
}

/// Weight of each vertex in the named group, or `None` for a vertex outside it.
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
            weights[*index] = Some(weight);
        }
    }
    Ok(weights)
}

/// Mean rest length of the springs in `keys`, or zero when there are none.
fn average_length(
    keys: &BTreeSet<(u32, u32)>,
    ids: &BTreeMap<u32, usize>,
    positions: &[DVec3],
) -> f64 {
    if keys.is_empty() {
        return 0.0;
    }
    let total = keys
        .iter()
        .map(|(a_id, b_id)| positions[ids[a_id]].distance(positions[ids[b_id]]))
        .sum::<f64>();
    total / keys.len() as f64
}

/// XPBD compliance of a spring with force constant `force_constant` per unit length.
fn spring_compliance(force_constant: f64, dt: f64) -> f64 {
    if force_constant > 0.0 {
        1.0 / (force_constant * dt * dt)
    } else {
        f64::INFINITY
    }
}

/// Builds springs from edge keys. `compliance` maps a spring's rest length to its
/// `(stretched, compressed)` compliances.
fn make_springs(
    keys: &BTreeSet<(u32, u32)>,
    ids: &BTreeMap<u32, usize>,
    positions: &[DVec3],
    compliance: impl Fn(f64) -> (f64, f64),
) -> Result<Vec<Spring>> {
    let mut springs = Vec::with_capacity(keys.len());
    for (a_id, b_id) in keys {
        let a = ids[a_id];
        let b = ids[b_id];
        let rest_length = positions[a].distance(positions[b]);
        if !rest_length.is_finite() {
            return Err(invalid("cloth spring rest length is not finite"));
        }
        let (tension_alpha, compression_alpha) = compliance(rest_length);
        springs.push(Spring {
            a,
            b,
            rest_length,
            tension_alpha,
            compression_alpha,
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
        let delta = positions[spring.a] - positions[spring.b];
        let distance = delta.length();
        let weight = inverse_masses[spring.a] + inverse_masses[spring.b];
        if distance <= DISTANCE_EPSILON || weight == 0.0 {
            continue;
        }
        let constraint = distance - spring.rest_length;
        let alpha = if constraint > 0.0 {
            spring.tension_alpha
        } else {
            spring.compression_alpha
        };
        if alpha.is_infinite() {
            continue;
        }
        let delta_lambda = (-constraint - alpha * spring.lambda) / (weight + alpha);
        spring.lambda += delta_lambda;
        let correction = delta * (delta_lambda / distance);
        positions[spring.a] += correction * inverse_masses[spring.a];
        positions[spring.b] -= correction * inverse_masses[spring.b];
    }
}

/// A goal spring: an XPBD distance constraint from a vertex to its input position.
#[derive(Clone, Copy)]
struct GoalSpring {
    index: usize,
    /// XPBD compliance of the goal spring (`spring_compliance` of its stiffness).
    alpha: f64,
    lambda: f64,
}

fn solve_goal_springs(
    positions: &mut [DVec3],
    inverse_masses: &[f64],
    springs: &mut [GoalSpring],
    targets: &[DVec3],
) {
    for spring in springs {
        let delta = positions[spring.index] - targets[spring.index];
        let distance = delta.length();
        let weight = inverse_masses[spring.index];
        if distance <= DISTANCE_EPSILON || weight == 0.0 {
            continue;
        }
        let delta_lambda = (-distance - spring.alpha * spring.lambda) / (weight + spring.alpha);
        spring.lambda += delta_lambda;
        positions[spring.index] += delta * (delta_lambda * weight / distance);
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
    if value.as_str() == Some("") {
        return Ok(None);
    }
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
    use crate::sim::ColliderTriangle;

    use super::simulate;

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
            &json!({"vertex_group_mass": "fixed", "iterations": 12}),
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
                &json!({"vertex_group_mass": "fixed"}),
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
    fn partial_goal_spring_holds_vertex_toward_its_input() {
        let mut mesh = Mesh::from_positions_and_faces(
            vec![DVec3::ZERO, DVec3::X, DVec3::Y],
            vec![vec![0, 1, 2]],
        )
        .unwrap();
        mesh.attributes
            .insert("vertex_groups".to_owned(), json!({"partial": {"0": 0.5}}));
        let gravity = DVec3::new(0.0, 0.0, -9.81);
        let free = simulate(&mesh, &json!({}), 1.0, 24.0, gravity, &[]).unwrap();
        let held = simulate(
            &mesh,
            &json!({"vertex_group_mass": "partial", "iterations": 12}),
            1.0,
            24.0,
            gravity,
            &[],
        )
        .unwrap();
        // Goal 0.5^4 = 0.0625 is not pinned, so the vertex moves, but its goal spring holds it above
        // the unconstrained fall.
        assert!(held[0].z > free[0].z);
        assert!(held[0].z < mesh.vertices[0].co.z);
    }

    #[test]
    fn zero_goal_spring_applies_no_goal_force_and_stays_finite() {
        let mut mesh = Mesh::from_positions_and_faces(
            vec![DVec3::ZERO, DVec3::X, DVec3::Y],
            vec![vec![0, 1, 2]],
        )
        .unwrap();
        mesh.attributes
            .insert("vertex_groups".to_owned(), json!({"partial": {"0": 0.5}}));
        let result = simulate(
            &mesh,
            &json!({"vertex_group_mass": "partial", "iterations": 12, "goal_spring": 0.0}),
            1.0,
            24.0,
            DVec3::new(0.0, 0.0, -9.81),
            &[],
        )
        .unwrap();
        assert!(result.iter().all(|position| position.is_finite()));
        assert!(result[0].z < mesh.vertices[0].co.z);
    }

    #[test]
    fn cloth_vertices_do_not_cross_two_sided_triangle_colliders() {
        let point =
            Mesh::from_positions_and_faces(vec![DVec3::new(0.0, 0.0, 0.001)], Vec::new()).unwrap();
        let colliders = [collider([
            DVec3::new(-2.0, -2.0, 0.0),
            DVec3::new(2.0, -2.0, 0.0),
            DVec3::new(0.0, 2.0, 0.0),
        ])];
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

    #[test]
    fn culled_collider_does_not_block_motion_from_its_back_side() {
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
            &json!({}),
            1.0,
            24.0,
            DVec3::new(0.0, 0.0, -10_000.0),
            &[culled],
        )
        .unwrap();
        assert!(result[0].z < -0.001);
    }
}
