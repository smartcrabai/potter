use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    path::Path,
};

use glam::{DMat4, DQuat, DVec3};
use serde_json::json;

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::Mesh,
    model::{ForceFieldType, Id, RigidBodyShape, RigidBodyType, SceneDoc},
};

use super::{SimulationResult, cache};

/// Bump whenever solver behavior or its deterministic output contract changes.
pub const SOLVER_VERSION: &str = "potter-physics-2";

const CONTACT_SLOP: f64 = 1.0e-8;
const POSITION_CORRECTION: f64 = 0.8;
const RESTITUTION_THRESHOLD: f64 = 1.0e-4;
const GEOMETRY_EPSILON: f64 = 1.0e-12;

#[derive(Clone, Copy, PartialEq, Eq)]
enum ShapeKind {
    Box,
    Sphere,
    ConvexHull,
    Mesh,
}

struct Body {
    id: Id,
    dynamic: bool,
    mass: f64,
    inv_mass: f64,
    inv_inertia_local: DVec3,
    friction: f64,
    restitution: f64,
    linear_damping: f64,
    angular_damping: f64,
    velocity: DVec3,
    angular_velocity: DVec3,
    matrix: DMat4,
    offset: DVec3,
    shape: ShapeKind,
    world_vertices: Vec<DVec3>,
    triangles: Vec<[DVec3; 3]>,
    local_center: DVec3,
    local_half_extents: DVec3,
    sphere_radius: f64,
    axes: [DVec3; 3],
}

impl Body {
    fn center(&self) -> DVec3 {
        self.matrix.transform_point3(self.local_center) + self.offset
    }

    fn vertex(&self, index: usize) -> DVec3 {
        self.world_vertices[index] + self.offset
    }

    fn inv_mass(&self) -> f64 {
        self.inv_mass
    }

    fn world_aabb(&self) -> (DVec3, DVec3) {
        if self.shape == ShapeKind::Sphere {
            let radius = DVec3::splat(self.sphere_radius);
            let center = self.center();
            return (center - radius, center + radius);
        }
        let Some(first) = self.world_vertices.first() else {
            let center = self.center();
            return (center, center);
        };
        let first = *first + self.offset;
        self.world_vertices
            .iter()
            .skip(1)
            .fold((first, first), |(minimum, maximum), vertex| {
                let point = *vertex + self.offset;
                (minimum.min(point), maximum.max(point))
            })
    }

    fn obb(&self) -> Obb {
        let half = self.local_half_extents;
        let column_lengths = DVec3::new(
            self.matrix.x_axis.truncate().length(),
            self.matrix.y_axis.truncate().length(),
            self.matrix.z_axis.truncate().length(),
        );
        Obb {
            center: self.center(),
            axes: self.axes,
            half_extents: half * column_lengths,
        }
    }

    fn translated_matrix(&self) -> DMat4 {
        let mut matrix = self.matrix;
        matrix.w_axis.x += self.offset.x;
        matrix.w_axis.y += self.offset.y;
        matrix.w_axis.z += self.offset.z;
        matrix
    }
}

struct Force {
    id: Id,
    kind: ForceFieldType,
    origin: DVec3,
    axis: DVec3,
    strength: f64,
    falloff: f64,
}

#[derive(Clone, Copy)]
struct Obb {
    center: DVec3,
    axes: [DVec3; 3],
    half_extents: DVec3,
}

#[derive(Clone, Copy)]
struct Contact {
    normal: DVec3,
    point: DVec3,
    point_count: usize,
    penetration: f64,
    friction: f64,
    restitution: f64,
}

pub(super) fn evaluate(
    doc: &SceneDoc,
    scene_id: &Id,
    frame: f64,
    base_world_matrices: &BTreeMap<Id, DMat4>,
    cache_directory: Option<&Path>,
) -> Result<Option<SimulationResult>> {
    let scene = doc.scenes.get(scene_id).ok_or_else(|| {
        PotError::new(
            ErrorCode::SceneNotFound,
            format!("scene `{scene_id}` was not found"),
        )
    })?;
    let Some(world) = scene.rigid_body_world.as_ref() else {
        return Ok(None);
    };
    if !world.enabled {
        return Ok(None);
    }

    let members = scene_members(doc, scene_id)?;
    let body_ids: Vec<Id> = members
        .iter()
        .filter(|id| {
            doc.nodes
                .get(*id)
                .is_some_and(|node| node.rigid_body.is_some())
        })
        .cloned()
        .collect();
    if body_ids.is_empty() {
        return Ok(None);
    }
    if !frame.is_finite() {
        return Err(PotError::invalid_argument(
            "simulation frame must be finite",
        ));
    }
    validate_world(world, scene.fps, scene.fps_base, frame)?;

    let mut bodies = Vec::with_capacity(body_ids.len());
    for id in &body_ids {
        let node = doc.nodes.get(id).ok_or_else(|| {
            PotError::new(
                ErrorCode::SceneInvalid,
                format!("scene references missing node `{id}`"),
            )
        })?;
        let rigid_body = node.rigid_body.as_ref().ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "rigid-body collection changed during evaluation",
            )
        })?;
        let matrix = required_matrix(base_world_matrices, id, "rigid body")?;
        bodies.push(build_body(doc, id, node, rigid_body, matrix)?);
    }

    let mut forces = Vec::new();
    for id in &members {
        let Some(node) = doc.nodes.get(id) else {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                format!("scene references missing node `{id}`"),
            ));
        };
        let Some(field) = node.force_field.as_ref() else {
            continue;
        };
        if node.kind != "empty" {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                format!("force field `{id}` must be attached to an Empty node"),
            ));
        }
        let matrix = required_matrix(base_world_matrices, id, "force field")?;
        forces.push(build_force(id, field, matrix)?);
    }

    let key = cache::cache_key(doc, scene_id, frame, base_world_matrices)?;
    if let Some(directory) = cache_directory
        && let Some(result) = cache::load(directory, &key)?
        && result_matches_bodies(&result, &body_ids)
    {
        return Ok(Some(result));
    }

    simulate(
        &mut bodies,
        &forces,
        world,
        scene.fps,
        scene.fps_base,
        frame,
    )?;
    let result = make_result(bodies, key)?;
    if let Some(directory) = cache_directory {
        cache::store(directory, &result)?;
    }
    Ok(Some(result))
}

fn validate_world(
    world: &crate::model::RigidBodyWorld,
    fps: u32,
    fps_base: f64,
    frame: f64,
) -> Result<()> {
    if !world.gravity.iter().all(|value| value.is_finite()) {
        return Err(invalid_world("gravity must contain only finite values"));
    }
    if world.substeps == 0 || world.solver_iterations == 0 {
        return Err(invalid_world(
            "substeps and solver_iterations must be positive",
        ));
    }
    if world.frame_start > world.frame_end {
        return Err(invalid_world("frame_start must not exceed frame_end"));
    }
    if fps == 0 || !fps_base.is_finite() || fps_base <= 0.0 {
        return Err(invalid_world(
            "scene frame rate must be finite and positive",
        ));
    }
    if frame > f64::from(world.frame_end) {
        return Err(PotError::invalid_argument(format!(
            "requested frame {frame} is after rigid-body frame_end {}",
            world.frame_end
        )));
    }
    let steps_per_second = f64::from(fps) * fps_base * f64::from(world.substeps);
    if !steps_per_second.is_finite() || steps_per_second <= 0.0 {
        return Err(invalid_world(
            "fixed simulation step is not finite and positive",
        ));
    }
    Ok(())
}

fn invalid_world(message: &str) -> PotError {
    PotError::new(
        ErrorCode::InvalidArgument,
        format!("invalid rigid-body world: {message}"),
    )
}

fn scene_members(doc: &SceneDoc, scene_id: &Id) -> Result<BTreeSet<Id>> {
    let scene = doc.scenes.get(scene_id).ok_or_else(|| {
        PotError::new(
            ErrorCode::SceneNotFound,
            format!("scene `{scene_id}` was not found"),
        )
    })?;
    let mut visited = BTreeSet::new();
    let mut pending = vec![scene.root_collection.clone()];
    while let Some(collection_id) = pending.pop() {
        if !visited.insert(collection_id.clone()) {
            continue;
        }
        let collection = doc.collections.get(&collection_id).ok_or_else(|| {
            PotError::new(
                ErrorCode::SceneInvalid,
                format!("scene `{scene_id}` references missing collection `{collection_id}`"),
            )
        })?;
        pending.extend(collection.children.iter().cloned());
    }
    let mut members = BTreeSet::new();
    for collection_id in visited {
        let collection = doc.collections.get(&collection_id).ok_or_else(|| {
            PotError::new(
                ErrorCode::SceneInvalid,
                "scene collection disappeared during traversal",
            )
        })?;
        for node_id in &collection.objects {
            if !doc.nodes.contains_key(node_id) {
                return Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    format!("collection `{collection_id}` references missing node `{node_id}`"),
                ));
            }
            members.insert(node_id.clone());
        }
    }
    Ok(members)
}

fn required_matrix(matrices: &BTreeMap<Id, DMat4>, id: &Id, kind: &str) -> Result<DMat4> {
    let matrix = matrices.get(id).copied().ok_or_else(|| {
        PotError::new(
            ErrorCode::EvaluationFailed,
            format!("missing starting world matrix for {kind} `{id}`"),
        )
    })?;
    validate_affine_matrix(matrix, id)?;
    Ok(matrix)
}

fn validate_affine_matrix(matrix: DMat4, id: &Id) -> Result<()> {
    if !matrix.is_finite()
        || matrix.x_axis.w.abs() > 1.0e-10
        || matrix.y_axis.w.abs() > 1.0e-10
        || matrix.z_axis.w.abs() > 1.0e-10
        || (matrix.w_axis.w - 1.0).abs() > 1.0e-10
        || !matrix.determinant().is_finite()
        || matrix.determinant().abs() <= GEOMETRY_EPSILON
    {
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            format!("world matrix for `{id}` is not a finite affine transform"),
        ));
    }
    let axes = [
        matrix.x_axis.truncate(),
        matrix.y_axis.truncate(),
        matrix.z_axis.truncate(),
    ];
    if axes.iter().any(|axis| {
        let length_squared = axis.length_squared();
        !length_squared.is_finite() || length_squared <= GEOMETRY_EPSILON
    }) {
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            format!("world matrix for `{id}` has an invalid scale axis"),
        ));
    }
    Ok(())
}

fn build_body(
    doc: &SceneDoc,
    id: &Id,
    node: &crate::model::Node,
    rigid_body: &crate::model::RigidBody,
    matrix: DMat4,
) -> Result<Body> {
    let dynamic = match &rigid_body.body_type {
        RigidBodyType::Active => true,
        RigidBodyType::Passive => false,
    };
    if node.kind != "mesh" {
        return Err(invalid_body(id, "rigid bodies require a mesh node"));
    }
    if !rigid_body.mass.is_finite() || rigid_body.mass < 0.0 || (dynamic && rigid_body.mass == 0.0)
    {
        return Err(invalid_body(
            id,
            "mass must be finite and nonnegative (positive for active bodies)",
        ));
    }
    if !rigid_body.friction.is_finite() || rigid_body.friction < 0.0 {
        return Err(invalid_body(id, "friction must be finite and nonnegative"));
    }
    if !rigid_body.restitution.is_finite() || !(0.0..=1.0).contains(&rigid_body.restitution) {
        return Err(invalid_body(id, "restitution must be finite and in [0, 1]"));
    }
    if !rigid_body.linear_damping.is_finite() || rigid_body.linear_damping < 0.0 {
        return Err(invalid_body(
            id,
            "linear_damping must be finite and nonnegative",
        ));
    }
    if !rigid_body.angular_damping.is_finite() || rigid_body.angular_damping < 0.0 {
        return Err(invalid_body(
            id,
            "angular_damping must be finite and nonnegative",
        ));
    }
    if !rigid_body
        .initial_velocity
        .iter()
        .all(|value| value.is_finite())
    {
        return Err(invalid_body(
            id,
            "initial_velocity must contain finite values",
        ));
    }
    let (data_id, data) = node
        .data
        .as_ref()
        .and_then(|data_id| doc.data_blocks.get(data_id).map(|data| (data_id, data)))
        .ok_or_else(|| invalid_body(id, "a referenced mesh data block is required"))?;
    let mesh = data
        .mesh
        .as_ref()
        .ok_or_else(|| invalid_body(id, &format!("data block `{data_id}` has no mesh geometry")))?;
    build_mesh_body(id, rigid_body, mesh, matrix, dynamic)
}

fn build_mesh_body(
    id: &Id,
    rigid_body: &crate::model::RigidBody,
    mesh: &Mesh,
    matrix: DMat4,
    dynamic: bool,
) -> Result<Body> {
    mesh.validate().map_err(|error| {
        PotError::with_details(
            ErrorCode::SceneInvalid,
            format!("invalid collision mesh for `{id}`"),
            json!({"cause": error.to_string()}),
        )
    })?;
    if mesh.vertices.is_empty() {
        return Err(invalid_body(id, "collision mesh has no vertices"));
    }
    let (kind, needs_triangles) = match &rigid_body.shape {
        RigidBodyShape::Box => (ShapeKind::Box, false),
        RigidBodyShape::Sphere => (ShapeKind::Sphere, false),
        RigidBodyShape::ConvexHull => (ShapeKind::ConvexHull, true),
        RigidBodyShape::Mesh => (ShapeKind::Mesh, true),
    };
    let local_vertices: Vec<DVec3> = mesh.vertices.iter().map(|vertex| vertex.co).collect();
    let first = local_vertices[0];
    let (minimum, maximum) = local_vertices
        .iter()
        .skip(1)
        .fold((first, first), |(minimum, maximum), vertex| {
            (minimum.min(*vertex), maximum.max(*vertex))
        });
    let local_center = (minimum + maximum) * 0.5;
    let local_half_extents = (maximum - minimum) * 0.5;
    let axes_raw = [
        matrix.x_axis.truncate(),
        matrix.y_axis.truncate(),
        matrix.z_axis.truncate(),
    ];
    let column_lengths = DVec3::new(
        axes_raw[0].length(),
        axes_raw[1].length(),
        axes_raw[2].length(),
    );
    let axes = [
        axes_raw[0].normalize(),
        axes_raw[1].normalize(),
        axes_raw[2].normalize(),
    ];
    if kind == ShapeKind::Box
        && (axes[0].dot(axes[1]).abs() > 1.0e-8
            || axes[0].dot(axes[2]).abs() > 1.0e-8
            || axes[1].dot(axes[2]).abs() > 1.0e-8)
    {
        return Err(invalid_body(
            id,
            "box collision does not support a sheared world matrix",
        ));
    }
    if kind == ShapeKind::Box
        && (local_half_extents.x <= GEOMETRY_EPSILON
            || local_half_extents.y <= GEOMETRY_EPSILON
            || local_half_extents.z <= GEOMETRY_EPSILON)
    {
        return Err(invalid_body(
            id,
            "box collision mesh must have positive extent on every axis",
        ));
    }
    if !local_half_extents.is_finite() {
        return Err(invalid_body(id, "collision mesh bounds are not finite"));
    }
    let world_vertices: Vec<DVec3> = local_vertices
        .iter()
        .map(|vertex| matrix.transform_point3(*vertex))
        .collect();
    if world_vertices.iter().any(|vertex| !vertex.is_finite()) {
        return Err(invalid_body(
            id,
            "transformed collision mesh vertices are not finite",
        ));
    }
    let sphere_radius = world_vertices
        .iter()
        .map(|vertex| vertex.distance(matrix.transform_point3(local_center)))
        .fold(0.0, f64::max);
    if kind == ShapeKind::Sphere
        && (!sphere_radius.is_finite() || sphere_radius <= GEOMETRY_EPSILON)
    {
        return Err(invalid_body(
            id,
            "sphere collision mesh must have positive radius",
        ));
    }
    let triangulated = mesh.triangulate().map_err(|error| {
        PotError::with_details(
            ErrorCode::SceneInvalid,
            format!("could not triangulate collision mesh for `{id}`"),
            json!({"cause": error.to_string()}),
        )
    })?;
    if needs_triangles && triangulated.is_empty() {
        return Err(invalid_body(
            id,
            "mesh collision requires at least one valid face",
        ));
    }
    if kind == ShapeKind::ConvexHull && !has_volume(&local_vertices) {
        return Err(invalid_body(
            id,
            "convex-hull collision requires non-coplanar vertices",
        ));
    }
    let vertex_indices: HashMap<u32, usize> = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let mut triangles = Vec::with_capacity(triangulated.len());
    for triangle in triangulated {
        let mut points = [DVec3::ZERO; 3];
        for (corner, vertex_id) in triangle.into_iter().enumerate() {
            let index = vertex_indices.get(&vertex_id).copied().ok_or_else(|| {
                PotError::new(
                    ErrorCode::SceneInvalid,
                    "collision face references a missing vertex",
                )
            })?;
            points[corner] = world_vertices[index];
        }
        let area_squared = (points[1] - points[0])
            .cross(points[2] - points[0])
            .length_squared();
        if !area_squared.is_finite() || area_squared <= GEOMETRY_EPSILON {
            return Err(invalid_body(
                id,
                "collision mesh contains a degenerate world-space face",
            ));
        }
        triangles.push(points);
    }
    let inv_mass = if dynamic { 1.0 / rigid_body.mass } else { 0.0 };
    if dynamic && (!inv_mass.is_finite() || inv_mass <= 0.0) {
        return Err(invalid_body(
            id,
            "mass is outside the representable solver range",
        ));
    }
    let inv_inertia_local = if dynamic {
        let inertia = if kind == ShapeKind::Box {
            let extents = local_half_extents * column_lengths;
            let factor = rigid_body.mass / 3.0;
            DVec3::new(
                factor * (extents.y * extents.y + extents.z * extents.z),
                factor * (extents.x * extents.x + extents.z * extents.z),
                factor * (extents.x * extents.x + extents.y * extents.y),
            )
        } else {
            DVec3::splat(0.4 * rigid_body.mass * sphere_radius * sphere_radius)
        };
        if !inertia.is_finite() || inertia.min_element() <= 0.0 {
            return Err(invalid_body(
                id,
                "inertia is outside the representable solver range",
            ));
        }
        let inverse = DVec3::ONE / inertia;
        if !inverse.is_finite() || inverse.min_element() <= 0.0 {
            return Err(invalid_body(
                id,
                "inertia is outside the representable solver range",
            ));
        }
        inverse
    } else {
        DVec3::ZERO
    };
    let velocity = if dynamic {
        DVec3::from_array(rigid_body.initial_velocity)
    } else {
        DVec3::ZERO
    };
    Ok(Body {
        id: id.clone(),
        dynamic,
        mass: rigid_body.mass,
        inv_mass,
        inv_inertia_local,
        friction: rigid_body.friction,
        restitution: rigid_body.restitution,
        linear_damping: rigid_body.linear_damping,
        angular_damping: rigid_body.angular_damping,
        velocity,
        angular_velocity: DVec3::ZERO,
        matrix,
        offset: DVec3::ZERO,
        shape: kind,
        world_vertices,
        triangles,
        local_center,
        local_half_extents,
        sphere_radius,
        axes,
    })
}

fn has_volume(vertices: &[DVec3]) -> bool {
    if vertices.len() < 4 {
        return false;
    }
    let origin = vertices[0];
    let Some(edge_a) = vertices
        .iter()
        .skip(1)
        .map(|vertex| *vertex - origin)
        .find(|edge| edge.length_squared() > GEOMETRY_EPSILON)
    else {
        return false;
    };
    let Some(edge_b) = vertices
        .iter()
        .skip(1)
        .map(|vertex| *vertex - origin)
        .find(|edge| edge_a.cross(*edge).length_squared() > GEOMETRY_EPSILON)
    else {
        return false;
    };
    let normal = edge_a.cross(edge_b);
    vertices
        .iter()
        .any(|point| normal.dot(*point - origin).abs() > GEOMETRY_EPSILON)
}

fn invalid_body(id: &Id, message: &str) -> PotError {
    PotError::new(
        ErrorCode::SceneInvalid,
        format!("invalid rigid body `{id}`: {message}"),
    )
}

fn build_force(id: &Id, field: &crate::model::ForceField, matrix: DMat4) -> Result<Force> {
    if !field.strength.is_finite() {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            format!("force field `{id}` strength must be finite"),
        ));
    }
    if !field.falloff.is_finite() || field.falloff < 0.0 {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            format!("force field `{id}` falloff must be finite and nonnegative"),
        ));
    }
    let axis = matrix.z_axis.truncate().normalize();
    if axis.length_squared() <= GEOMETRY_EPSILON {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            format!("force field `{id}` has no usable direction"),
        ));
    }
    Ok(Force {
        id: id.clone(),
        kind: field.field_type,
        origin: matrix.w_axis.truncate(),
        axis,
        strength: field.strength,
        falloff: field.falloff,
    })
}

fn simulate(
    bodies: &mut [Body],
    forces: &[Force],
    world: &crate::model::RigidBodyWorld,
    fps: u32,
    fps_base: f64,
    frame: f64,
) -> Result<()> {
    let elapsed_frames = frame - f64::from(world.frame_start);
    if elapsed_frames <= 0.0 {
        return Ok(());
    }
    let seconds_per_frame = 1.0 / (f64::from(fps) * fps_base);
    let duration = elapsed_frames * seconds_per_frame;
    let fixed_dt = seconds_per_frame / f64::from(world.substeps);
    if !duration.is_finite() || !fixed_dt.is_finite() || fixed_dt <= 0.0 {
        return Err(invalid_world("requested simulation duration is not finite"));
    }
    let full_steps_value = (duration / fixed_dt).floor();
    if !full_steps_value.is_finite() || full_steps_value > usize::MAX as f64 {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "requested simulation duration exceeds the supported step count",
        ));
    }
    let full_steps = full_steps_value as usize;
    let mut remainder = duration - full_steps as f64 * fixed_dt;
    if remainder.abs() <= fixed_dt * 1.0e-12 {
        remainder = 0.0;
    }
    for _ in 0..full_steps {
        step(bodies, forces, world, fixed_dt)?;
    }
    if remainder > 0.0 {
        step(bodies, forces, world, remainder)?;
    }
    Ok(())
}

fn step(
    bodies: &mut [Body],
    forces: &[Force],
    world: &crate::model::RigidBodyWorld,
    dt: f64,
) -> Result<()> {
    let gravity = DVec3::from_array(world.gravity);
    for body in bodies.iter_mut().filter(|body| body.dynamic) {
        let mut acceleration = gravity;
        for force in forces {
            acceleration += field_acceleration(force, body.center(), body.mass)?;
        }
        if !acceleration.is_finite() {
            return Err(PotError::new(
                ErrorCode::EvaluationFailed,
                "non-finite force produced during rigid-body simulation",
            ));
        }
        body.velocity += acceleration * dt;
        body.velocity *= (-body.linear_damping * dt).exp();
        body.offset += body.velocity * dt;
        body.angular_velocity *= (-body.angular_damping * dt).exp();
        integrate_rotation(body, dt)?;
        if !body.velocity.is_finite()
            || !body.angular_velocity.is_finite()
            || !body.offset.is_finite()
        {
            return Err(PotError::new(
                ErrorCode::EvaluationFailed,
                format!("simulation of `{}` produced a non-finite state", body.id),
            ));
        }
    }

    let iterations = world.solver_iterations as usize;
    for _ in 0..iterations {
        for first in 0..bodies.len() {
            for second in (first + 1)..bodies.len() {
                if !bodies[first].dynamic && !bodies[second].dynamic {
                    continue;
                }
                let Some(contact) = detect_contact(&bodies[first], &bodies[second]) else {
                    continue;
                };
                resolve_contact(bodies, first, second, contact)?;
            }
        }
    }
    if bodies.iter().any(|body| {
        !body.velocity.is_finite()
            || !body.angular_velocity.is_finite()
            || !body.offset.is_finite()
            || !body.center().is_finite()
            || !body.matrix.is_finite()
            || body.axes.iter().any(|axis| !axis.is_finite())
            || body
                .world_vertices
                .iter()
                .any(|vertex| !(*vertex + body.offset).is_finite())
            || body
                .triangles
                .iter()
                .flatten()
                .any(|vertex| !(*vertex + body.offset).is_finite())
    }) {
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            "rigid-body contact solver produced a non-finite state",
        ));
    }
    Ok(())
}

fn integrate_rotation(body: &mut Body, dt: f64) -> Result<()> {
    let scaled_axis = body.angular_velocity * dt;
    let angle = scaled_axis.length();
    if !scaled_axis.is_finite() || !angle.is_finite() {
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            format!("simulation of `{}` produced a non-finite rotation", body.id),
        ));
    }
    if angle <= GEOMETRY_EPSILON {
        return Ok(());
    }
    let rotation = DQuat::from_axis_angle(scaled_axis / angle, angle);
    if !rotation.is_finite() {
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            format!("simulation of `{}` produced a non-finite rotation", body.id),
        ));
    }
    let center = body.center();
    let mut matrix = body.matrix;
    matrix.x_axis = (rotation * matrix.x_axis.truncate()).extend(matrix.x_axis.w);
    matrix.y_axis = (rotation * matrix.y_axis.truncate()).extend(matrix.y_axis.w);
    matrix.z_axis = (rotation * matrix.z_axis.truncate()).extend(matrix.z_axis.w);
    let translation_correction = center - body.offset - matrix.transform_point3(body.local_center);
    matrix.w_axis.x += translation_correction.x;
    matrix.w_axis.y += translation_correction.y;
    matrix.w_axis.z += translation_correction.z;
    body.matrix = matrix;
    for vertex in &mut body.world_vertices {
        let world_vertex = *vertex + body.offset;
        *vertex = center + rotation * (world_vertex - center) - body.offset;
    }
    for triangle in &mut body.triangles {
        for vertex in triangle {
            let world_vertex = *vertex + body.offset;
            *vertex = center + rotation * (world_vertex - center) - body.offset;
        }
    }
    for axis in &mut body.axes {
        *axis = rotation * *axis;
    }
    Ok(())
}

fn field_acceleration(force: &Force, position: DVec3, mass: f64) -> Result<DVec3> {
    let displacement = position - force.origin;
    let distance = displacement.length();
    let attenuation = if force.falloff == 0.0 {
        1.0
    } else {
        (1.0 + distance).powf(-force.falloff)
    };
    let magnitude = force.strength * attenuation / mass;
    let direction = match force.kind {
        ForceFieldType::Wind => force.axis,
        ForceFieldType::Force => {
            if distance <= GEOMETRY_EPSILON {
                DVec3::ZERO
            } else {
                displacement / distance
            }
        }
        ForceFieldType::Vortex => {
            let radial = displacement - force.axis * displacement.dot(force.axis);
            let tangent = force.axis.cross(radial);
            if tangent.length_squared() <= GEOMETRY_EPSILON {
                DVec3::ZERO
            } else {
                tangent.normalize()
            }
        }
    };
    let acceleration = direction * magnitude;
    if !acceleration.is_finite() {
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            format!(
                "force field `{}` produced a non-finite acceleration",
                force.id
            ),
        ));
    }
    Ok(acceleration)
}

fn detect_contact(a: &Body, b: &Body) -> Option<Contact> {
    if !overlap(a.world_aabb(), b.world_aabb()) {
        return None;
    }
    if a.shape == ShapeKind::Sphere && b.shape == ShapeKind::Sphere {
        return sphere_sphere_contact(a, b);
    }
    if a.shape == ShapeKind::Sphere && b.shape == ShapeKind::Box {
        return sphere_box_contact(a, b)
            .map(|contact| contact_with_normal(contact, contact.normal));
    }
    if a.shape == ShapeKind::Box && b.shape == ShapeKind::Sphere {
        return sphere_box_contact(b, a)
            .map(|contact| contact_with_normal(contact, -contact.normal));
    }
    if a.shape == ShapeKind::Box && b.shape == ShapeKind::Box {
        return box_box_contact(a, b);
    }
    if a.shape == ShapeKind::Sphere {
        return sphere_mesh_contact(a, b);
    }
    if b.shape == ShapeKind::Sphere {
        return sphere_mesh_contact(b, a)
            .map(|contact| contact_with_normal(contact, -contact.normal));
    }
    triangle_contact(a, b)
}

fn material_contact(a: &Body, b: &Body, normal: DVec3, penetration: f64, point: DVec3) -> Contact {
    Contact {
        normal,
        point,
        point_count: 1,
        penetration,
        friction: a.friction.sqrt() * b.friction.sqrt(),
        restitution: a.restitution.max(b.restitution),
    }
}

fn contact_with_normal(mut contact: Contact, normal: DVec3) -> Contact {
    contact.normal = normal;
    contact
}

fn sphere_sphere_contact(a: &Body, b: &Body) -> Option<Contact> {
    let displacement = a.center() - b.center();
    let distance_squared = displacement.length_squared();
    let radius = a.sphere_radius + b.sphere_radius;
    if distance_squared >= radius * radius {
        return None;
    }
    let distance = distance_squared.sqrt();
    let normal = if distance > GEOMETRY_EPSILON {
        displacement / distance
    } else {
        DVec3::Y
    };
    let point =
        (a.center() - normal * a.sphere_radius + b.center() + normal * b.sphere_radius) * 0.5;
    Some(material_contact(a, b, normal, radius - distance, point))
}

fn sphere_box_contact(sphere: &Body, box_body: &Body) -> Option<Contact> {
    let box_shape = box_body.obb();
    let relative = sphere.center() - box_shape.center;
    let coordinates = [
        relative.dot(box_shape.axes[0]),
        relative.dot(box_shape.axes[1]),
        relative.dot(box_shape.axes[2]),
    ];
    let extents = [
        box_shape.half_extents.x,
        box_shape.half_extents.y,
        box_shape.half_extents.z,
    ];
    let mut closest = box_shape.center;
    let mut inside = true;
    for axis in 0..3 {
        let clamped = coordinates[axis].clamp(-extents[axis], extents[axis]);
        if !crate::float::equal_f64(clamped, coordinates[axis]) {
            inside = false;
        }
        closest += box_shape.axes[axis] * clamped;
    }
    let delta = sphere.center() - closest;
    let distance_squared = delta.length_squared();
    if !inside {
        if distance_squared >= sphere.sphere_radius * sphere.sphere_radius {
            return None;
        }
        let distance = distance_squared.sqrt();
        let normal = if distance > GEOMETRY_EPSILON {
            delta / distance
        } else {
            DVec3::Y
        };
        let point = (closest + sphere.center() - normal * sphere.sphere_radius) * 0.5;
        return Some(material_contact(
            sphere,
            box_body,
            normal,
            sphere.sphere_radius - distance,
            point,
        ));
    }
    let mut nearest_axis = 0;
    let mut nearest_clearance = f64::INFINITY;
    for axis in 0..3 {
        let clearance = extents[axis] - coordinates[axis].abs();
        if clearance < nearest_clearance {
            nearest_clearance = clearance;
            nearest_axis = axis;
        }
    }
    let sign = if coordinates[nearest_axis] < 0.0 {
        -1.0
    } else {
        1.0
    };
    let normal = box_shape.axes[nearest_axis] * sign;
    let point = closest + normal * nearest_clearance;
    Some(material_contact(
        sphere,
        box_body,
        normal,
        sphere.sphere_radius + nearest_clearance,
        point,
    ))
}

fn box_box_contact(a: &Body, b: &Body) -> Option<Contact> {
    let first = a.obb();
    let second = b.obb();
    let center_delta = first.center - second.center;
    let first_extents = [
        first.half_extents.x,
        first.half_extents.y,
        first.half_extents.z,
    ];
    let second_extents = [
        second.half_extents.x,
        second.half_extents.y,
        second.half_extents.z,
    ];
    let mut minimum_overlap = f64::INFINITY;
    let mut minimum_axis = DVec3::Y;
    for axis in first.axes.into_iter().chain(second.axes) {
        let overlap_amount = projection_overlap(
            axis,
            center_delta,
            first.axes,
            first_extents,
            second.axes,
            second_extents,
        );
        if overlap_amount < 0.0 {
            return None;
        }
        if overlap_amount < minimum_overlap {
            minimum_overlap = overlap_amount;
            minimum_axis = if center_delta.dot(axis) < 0.0 {
                -axis
            } else {
                axis
            };
        }
    }
    for first_axis in first.axes {
        for second_axis in second.axes {
            let axis = first_axis.cross(second_axis);
            if axis.length_squared() <= GEOMETRY_EPSILON {
                continue;
            }
            let axis = axis.normalize();
            let overlap_amount = projection_overlap(
                axis,
                center_delta,
                first.axes,
                first_extents,
                second.axes,
                second_extents,
            );
            if overlap_amount < 0.0 {
                return None;
            }
            if overlap_amount < minimum_overlap {
                minimum_overlap = overlap_amount;
                minimum_axis = if center_delta.dot(axis) < 0.0 {
                    -axis
                } else {
                    axis
                };
            }
        }
    }
    let point = (obb_support(first, -minimum_axis) + obb_support(second, minimum_axis)) * 0.5;
    Some(material_contact(a, b, minimum_axis, minimum_overlap, point))
}

fn obb_support(obb: Obb, direction: DVec3) -> DVec3 {
    let extents = [obb.half_extents.x, obb.half_extents.y, obb.half_extents.z];
    let mut point = obb.center;
    for (index, axis) in obb.axes.into_iter().enumerate() {
        let sign = if direction.dot(axis) < 0.0 { -1.0 } else { 1.0 };
        point += axis * (extents[index] * sign);
    }
    point
}
fn projection_overlap(
    axis: DVec3,
    center_delta: DVec3,
    first_axes: [DVec3; 3],
    first_extents: [f64; 3],
    second_axes: [DVec3; 3],
    second_extents: [f64; 3],
) -> f64 {
    let first_radius = (0..3)
        .map(|index| first_extents[index] * axis.dot(first_axes[index]).abs())
        .sum::<f64>();
    let second_radius = (0..3)
        .map(|index| second_extents[index] * axis.dot(second_axes[index]).abs())
        .sum::<f64>();
    first_radius + second_radius - center_delta.dot(axis).abs()
}
fn sphere_mesh_contact(sphere: &Body, mesh: &Body) -> Option<Contact> {
    let center = sphere.center();
    let mut best: Option<Contact> = None;
    for triangle in &mesh.triangles {
        let triangle = translated_triangle(*triangle, mesh.offset);
        let closest = closest_point_triangle(center, triangle);
        let delta = center - closest;
        let distance_squared = delta.length_squared();
        if distance_squared >= sphere.sphere_radius * sphere.sphere_radius {
            continue;
        }
        let distance = distance_squared.sqrt();
        let triangle_normal = (triangle[1] - triangle[0])
            .cross(triangle[2] - triangle[0])
            .normalize();
        let mut normal = if distance > GEOMETRY_EPSILON {
            delta / distance
        } else if (center - (triangle[0] + triangle[1] + triangle[2]) / 3.0).dot(triangle_normal)
            >= 0.0
        {
            triangle_normal
        } else {
            -triangle_normal
        };
        if distance <= GEOMETRY_EPSILON && normal.dot(sphere.velocity - mesh.velocity) > 0.0 {
            normal = -normal;
        }
        let point = (center - normal * sphere.sphere_radius + closest) * 0.5;
        let candidate =
            material_contact(sphere, mesh, normal, sphere.sphere_radius - distance, point);
        if best.is_none_or(|existing| candidate.penetration < existing.penetration) {
            best = Some(candidate);
        }
    }
    best
}

fn triangle_contact(a: &Body, b: &Body) -> Option<Contact> {
    let mut best: Option<Contact> = None;
    for index in 0..a.world_vertices.len() {
        let point = a.vertex(index);
        let mut vertex_contact: Option<Contact> = None;
        for triangle in &b.triangles {
            let triangle = translated_triangle(*triangle, b.offset);
            if let Some(candidate) = point_triangle_contact(point, a.center(), triangle, a, b)
                && vertex_contact
                    .is_none_or(|existing| candidate.penetration < existing.penetration)
            {
                vertex_contact = Some(candidate);
            }
        }
        if let Some(candidate) = vertex_contact {
            choose_shallower(&mut best, candidate);
        }
    }
    for index in 0..b.world_vertices.len() {
        let point = b.vertex(index);
        let mut vertex_contact: Option<Contact> = None;
        for triangle in &a.triangles {
            let triangle = translated_triangle(*triangle, a.offset);
            if let Some(mut candidate) = point_triangle_contact(point, b.center(), triangle, b, a) {
                candidate.normal = -candidate.normal;
                if vertex_contact
                    .is_none_or(|existing| candidate.penetration < existing.penetration)
                {
                    vertex_contact = Some(candidate);
                }
            }
        }
        if let Some(candidate) = vertex_contact {
            choose_shallower(&mut best, candidate);
        }
    }
    best
}

fn point_triangle_contact(
    point: DVec3,
    body_center: DVec3,
    triangle: [DVec3; 3],
    a: &Body,
    b: &Body,
) -> Option<Contact> {
    let raw_normal = (triangle[1] - triangle[0]).cross(triangle[2] - triangle[0]);
    if raw_normal.length_squared() <= GEOMETRY_EPSILON {
        return None;
    }
    let mut normal = raw_normal.normalize();
    let face_center = (triangle[0] + triangle[1] + triangle[2]) / 3.0;
    if normal.dot(body_center - face_center) < 0.0 {
        normal = -normal;
    }
    let plane_distance = (point - triangle[0]).dot(normal);
    let projected = point - normal * plane_distance;
    let closest = closest_point_triangle(projected, triangle);
    let scale = (triangle[1] - triangle[0])
        .length_squared()
        .max((triangle[2] - triangle[1]).length_squared())
        .max((triangle[0] - triangle[2]).length_squared())
        .max(1.0);
    if projected.distance_squared(closest) > scale * 1.0e-12 {
        return None;
    }
    if plane_distance >= -CONTACT_SLOP {
        return None;
    }
    Some(material_contact(a, b, normal, -plane_distance, closest))
}

fn choose_shallower(best: &mut Option<Contact>, candidate: Contact) {
    let Some(existing) = best.as_mut() else {
        *best = Some(candidate);
        return;
    };
    if candidate.penetration + CONTACT_SLOP < existing.penetration {
        *existing = candidate;
    } else if (candidate.penetration - existing.penetration).abs() <= CONTACT_SLOP
        && existing.normal.dot(candidate.normal) > 1.0 - 1.0e-8
    {
        let total = existing.point_count + candidate.point_count;
        existing.point +=
            (candidate.point - existing.point) * (candidate.point_count as f64 / total as f64);
        existing.point_count = total;
        existing.penetration = existing.penetration.min(candidate.penetration);
    }
}

fn translated_triangle(triangle: [DVec3; 3], offset: DVec3) -> [DVec3; 3] {
    [
        triangle[0] + offset,
        triangle[1] + offset,
        triangle[2] + offset,
    ]
}

fn closest_point_triangle(point: DVec3, triangle: [DVec3; 3]) -> DVec3 {
    let [a, b, c] = triangle;
    let ab = b - a;
    let ac = c - a;
    let ap = point - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = point - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return a + ab * (d1 / (d1 - d3));
    }
    let cp = point - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return a + ac * (d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let denominator = 1.0 / (va + vb + vc);
    let v = vb * denominator;
    let w = vc * denominator;
    a + ab * v + ac * w
}

fn overlap(first: (DVec3, DVec3), second: (DVec3, DVec3)) -> bool {
    first.0.cmple(second.1).all() && first.1.cmpge(second.0).all()
}

fn resolve_contact(
    bodies: &mut [Body],
    first: usize,
    second: usize,
    contact: Contact,
) -> Result<()> {
    let inverse_mass_sum = bodies[first].inv_mass() + bodies[second].inv_mass();
    if inverse_mass_sum <= 0.0 {
        return Ok(());
    }
    let first_arm = contact.point - bodies[first].center();
    let second_arm = contact.point - bodies[second].center();
    let relative_velocity =
        contact_velocity(&bodies[first], first_arm) - contact_velocity(&bodies[second], second_arm);
    let normal_speed = relative_velocity.dot(contact.normal);
    let restitution = if normal_speed < -RESTITUTION_THRESHOLD {
        contact.restitution
    } else {
        0.0
    };
    let normal_mass = inverse_mass_sum
        + angular_effective_mass(&bodies[first], first_arm, contact.normal)
        + angular_effective_mass(&bodies[second], second_arm, contact.normal);
    if !normal_mass.is_finite() || normal_mass <= 0.0 || !normal_speed.is_finite() {
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            "rigid-body contact solver produced a non-finite effective mass",
        ));
    }
    let normal_impulse_magnitude = if normal_speed < 0.0 {
        -(1.0 + restitution) * normal_speed / normal_mass
    } else {
        0.0
    };
    let normal_impulse = contact.normal * normal_impulse_magnitude;
    apply_velocity_impulse(bodies, first, second, contact.point, normal_impulse);

    if normal_impulse_magnitude > 0.0 && contact.friction > 0.0 {
        let relative_velocity =
            contact_velocity(&bodies[first], contact.point - bodies[first].center())
                - contact_velocity(&bodies[second], contact.point - bodies[second].center());
        let tangent_velocity =
            relative_velocity - contact.normal * relative_velocity.dot(contact.normal);
        let tangent_length = tangent_velocity.length();
        if tangent_length > GEOMETRY_EPSILON {
            let tangent = tangent_velocity / tangent_length;
            let tangent_mass = inverse_mass_sum
                + angular_effective_mass(&bodies[first], first_arm, tangent)
                + angular_effective_mass(&bodies[second], second_arm, tangent);
            if !tangent_mass.is_finite() || tangent_mass <= 0.0 {
                return Err(PotError::new(
                    ErrorCode::EvaluationFailed,
                    "rigid-body contact solver produced a non-finite effective mass",
                ));
            }
            let tangent_impulse_magnitude = (-tangent_length / tangent_mass).clamp(
                -contact.friction * normal_impulse_magnitude,
                contact.friction * normal_impulse_magnitude,
            );
            apply_velocity_impulse(
                bodies,
                first,
                second,
                contact.point,
                tangent * tangent_impulse_magnitude,
            );
        }
    }

    let correction_depth = (contact.penetration - CONTACT_SLOP).max(0.0);
    if correction_depth > 0.0 {
        let correction =
            contact.normal * (POSITION_CORRECTION * correction_depth / inverse_mass_sum);
        apply_position_correction(bodies, first, second, correction);
    }
    Ok(())
}

fn contact_velocity(body: &Body, arm: DVec3) -> DVec3 {
    body.velocity + body.angular_velocity.cross(arm)
}

fn inverse_inertia_world(body: &Body, vector: DVec3) -> DVec3 {
    body.axes[0] * (body.inv_inertia_local.x * vector.dot(body.axes[0]))
        + body.axes[1] * (body.inv_inertia_local.y * vector.dot(body.axes[1]))
        + body.axes[2] * (body.inv_inertia_local.z * vector.dot(body.axes[2]))
}

fn angular_effective_mass(body: &Body, arm: DVec3, direction: DVec3) -> f64 {
    inverse_inertia_world(body, arm.cross(direction))
        .cross(arm)
        .dot(direction)
        .max(0.0)
}

fn apply_velocity_impulse(
    bodies: &mut [Body],
    first: usize,
    second: usize,
    point: DVec3,
    impulse: DVec3,
) {
    let first_arm = point - bodies[first].center();
    let second_arm = point - bodies[second].center();
    let first_angular_impulse = inverse_inertia_world(&bodies[first], first_arm.cross(impulse));
    let second_angular_impulse = inverse_inertia_world(&bodies[second], second_arm.cross(impulse));
    if first < second {
        let (before, after) = bodies.split_at_mut(second);
        before[first].velocity += impulse * before[first].inv_mass();
        before[first].angular_velocity += first_angular_impulse;
        after[0].velocity -= impulse * after[0].inv_mass();
        after[0].angular_velocity -= second_angular_impulse;
    } else {
        let (before, after) = bodies.split_at_mut(first);
        after[0].velocity += impulse * after[0].inv_mass();
        after[0].angular_velocity += first_angular_impulse;
        before[second].velocity -= impulse * before[second].inv_mass();
        before[second].angular_velocity -= second_angular_impulse;
    }
}

fn apply_position_correction(bodies: &mut [Body], first: usize, second: usize, correction: DVec3) {
    if first < second {
        let (before, after) = bodies.split_at_mut(second);
        before[first].offset += correction * before[first].inv_mass();
        after[0].offset -= correction * after[0].inv_mass();
    } else {
        let (before, after) = bodies.split_at_mut(first);
        after[0].offset += correction * after[0].inv_mass();
        before[second].offset -= correction * before[second].inv_mass();
    }
}

fn make_result(bodies: Vec<Body>, key: String) -> Result<SimulationResult> {
    let mut world_matrices = BTreeMap::new();
    let mut linear_velocities = BTreeMap::new();
    for body in bodies {
        let matrix = body.translated_matrix().to_cols_array();
        let velocity = body.velocity.to_array();
        if matrix.iter().any(|value| !value.is_finite())
            || velocity.iter().any(|value| !value.is_finite())
        {
            return Err(PotError::new(
                ErrorCode::EvaluationFailed,
                format!("final state for `{}` is not finite", body.id),
            ));
        }
        world_matrices.insert(body.id.clone(), matrix);
        linear_velocities.insert(body.id, velocity);
    }
    Ok(SimulationResult {
        world_matrices,
        linear_velocities,
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
    })
}

fn result_matches_bodies(result: &SimulationResult, body_ids: &[Id]) -> bool {
    result.world_matrices.len() == body_ids.len()
        && result.linear_velocities.len() == body_ids.len()
        && body_ids.iter().all(|id| {
            result.world_matrices.contains_key(id) && result.linear_velocities.contains_key(id)
        })
}
