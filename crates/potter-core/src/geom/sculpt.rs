//! Deterministic, attribute-aware sculpt stroke evaluation over persistent meshes.

use glam::{DQuat, DVec3};
use serde_json::{Value, json};

use crate::{
    error::{PotError, Result},
    geom::Mesh,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Brush {
    Draw,
    ClayStrips,
    Inflate,
    Grab,
    Smooth,
    Flatten,
    Pinch,
    Crease,
    Layer,
    SnakeHook,
    Thumb,
    Rotate,
    Nudge,
    Blob,
    Scrape,
    Fill,
    DrawSharp,
    Elastic,
    PoseLite,
    BoundaryLite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Falloff {
    Smooth,
    Sphere,
    Root,
    Sharp,
    Linear,
    Constant,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrokeSample {
    pub position: DVec3,
    pub pressure: f64,
    pub radius: f64,
    pub strength: f64,
    pub time: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Stroke {
    pub brush: Brush,
    pub samples: Vec<StrokeSample>,
    pub falloff: Falloff,
    pub symmetry: Vec<String>,
    pub seed: u64,
    pub delta: Option<DVec3>,
}

/// Apply a stroke in the mesh coordinate space. Symmetry planes pass through the origin.
///
/// # Errors
///
/// Returns an error for malformed samples, unknown axes, or an unsupported brush delta.
pub fn apply_stroke(mesh: &mut Mesh, stroke: &Stroke) -> Result<bool> {
    apply_stroke_with_dyntopo(mesh, stroke, None)
}

/// Apply a stroke and optionally subdivide or collapse local edges under its brush.
///
/// # Errors
///
/// Returns an error for malformed samples, unknown axes, unsupported brush deltas, or an invalid edge length.
pub fn apply_stroke_with_dyntopo(
    mesh: &mut Mesh,
    stroke: &Stroke,
    dyntopo_edge_length: Option<f64>,
) -> Result<bool> {
    validate_stroke(stroke)?;
    if stroke.brush == Brush::Grab && stroke.delta.is_none() {
        return Err(invalid("grab brush requires delta"));
    }
    if stroke.brush == Brush::Rotate && stroke.delta.is_some_and(|delta| delta.length() == 0.0) {
        return Err(invalid(
            "rotate brush delta must be a nonzero axis-angle vector",
        ));
    }
    if stroke.brush != Brush::Grab
        && stroke.brush != Brush::SnakeHook
        && stroke.brush != Brush::Thumb
        && stroke.brush != Brush::Rotate
        && stroke.brush != Brush::Nudge
        && stroke.brush != Brush::Elastic
        && stroke.brush != Brush::PoseLite
        && stroke.brush != Brush::BoundaryLite
        && stroke.delta.is_some()
    {
        return Err(invalid(
            "delta is supported only by grab, snake_hook, thumb, rotate, nudge, elastic_deform, pose_lite, and boundary_lite brushes",
        ));
    }
    if let Some(edge_length) = dyntopo_edge_length
        && (!edge_length.is_finite() || edge_length <= 0.0)
    {
        return Err(invalid("dyntopo edge length must be finite and positive"));
    }
    let positions = vertex_positions(mesh);
    let normals = vertex_normals(mesh, &positions);
    let adjacency = vertex_adjacency(mesh);
    let boundary_adjacency = boundary_adjacency(mesh);
    let mask_values = read_scalar_attribute(mesh, "sculpt.mask")?;
    let mirrors = mirrored_samples(stroke)?;
    let (first_sample, last_sample) = stroke
        .samples
        .first()
        .zip(stroke.samples.last())
        .ok_or_else(|| invalid("sculpt stroke must contain at least one sample"))?;
    let trajectory = last_sample.position - first_sample.position;
    let mut next_positions = Vec::with_capacity(mesh.vertices.len());
    let mut changed = false;

    for vertex in &mesh.vertices {
        let normal = normals.get(&vertex.id).copied().unwrap_or(DVec3::Z);
        let mut influence = 0.0_f64;
        let mut center = DVec3::ZERO;
        let mut delta = DVec3::ZERO;
        let mut motion = trajectory;
        let mut radius = 0.0;
        for sample in &mirrors {
            let (closest, pressure, sample_radius, strength, sample_delta) =
                sample_influence(vertex.co, sample);
            if sample_radius <= 0.0 {
                continue;
            }
            let distance = vertex.co.distance(closest);
            if distance > sample_radius {
                continue;
            }
            let amount = stroke.falloff.weight(distance / sample_radius) * pressure * strength;
            if amount > influence {
                influence = amount;
                center = closest;
                delta = sample_delta;
                motion = sample.to - sample.from;
                radius = sample_radius;
            }
        }
        let mask = mask_values
            .get(&vertex.id)
            .copied()
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        influence *= 1.0 - mask;
        let position = if influence <= 0.0 {
            vertex.co
        } else {
            deform_vertex(
                stroke,
                vertex.co,
                normal,
                center,
                delta,
                motion,
                radius,
                influence,
                adjacency.get(&vertex.id).map(Vec::as_slice),
                boundary_adjacency.get(&vertex.id).map(Vec::as_slice),
                &positions,
            )
        };
        changed |= position != vertex.co;
        next_positions.push((vertex.id, position));
    }
    if changed {
        for (vertex, (id, position)) in mesh.vertices.iter_mut().zip(next_positions) {
            debug_assert_eq!(vertex.id, id);
            vertex.co = position;
        }
    }
    if let Some(edge_length) = dyntopo_edge_length {
        changed |= dyntopo_edges_under_brush(mesh, stroke, edge_length)?;
    }
    Ok(changed)
}

/// Subdivide long edges and collapse short local edges inside a sculpt brush.
///
/// # Errors
///
/// Returns an error for an invalid stroke or nonpositive/non-finite edge length.
pub fn dyntopo_under_brush(mesh: &mut Mesh, stroke: &Stroke, edge_length: f64) -> Result<bool> {
    validate_stroke(stroke)?;
    if !edge_length.is_finite() || edge_length <= 0.0 {
        return Err(invalid("dyntopo edge length must be finite and positive"));
    }
    dyntopo_edges_under_brush(mesh, stroke, edge_length)
}

fn dyntopo_edges_under_brush(mesh: &mut Mesh, stroke: &Stroke, edge_length: f64) -> Result<bool> {
    let subdivided = split_long_edges(mesh, stroke, edge_length)?;
    let collapsed = collapse_short_edges(mesh, stroke, edge_length * 0.5)?;
    Ok(subdivided || collapsed)
}

fn deform_vertex(
    stroke: &Stroke,
    point: DVec3,
    normal: DVec3,
    center: DVec3,
    sample_delta: DVec3,
    stroke_motion: DVec3,
    radius: f64,
    influence: f64,
    neighbors: Option<&[u32]>,
    boundary_neighbors: Option<&[u32]>,
    positions: &std::collections::BTreeMap<u32, DVec3>,
) -> DVec3 {
    let motion = if stroke.delta.is_some() {
        sample_delta
    } else {
        stroke_motion
    };
    match stroke.brush {
        Brush::Draw | Brush::ClayStrips | Brush::Inflate | Brush::Layer => {
            let multiplier = if stroke.brush == Brush::ClayStrips {
                0.75
            } else {
                1.0
            };
            let deterministic = deterministic_scale(stroke.seed, point, &stroke.symmetry);
            point + normal * (influence * multiplier * deterministic)
        }
        Brush::Grab => point + sample_delta * influence,
        Brush::Smooth => {
            let Some(neighbors) = neighbors.filter(|neighbors| !neighbors.is_empty()) else {
                return point;
            };
            let average = average_neighbor_position(positions, neighbors);
            point.lerp(average, influence)
        }
        Brush::Flatten => {
            let signed_distance = (point - center).dot(normal);
            point - normal * signed_distance * influence
        }
        Brush::Pinch => point.lerp(center, influence),
        Brush::Crease => {
            let toward = center - point;
            point + toward * influence + normal * (influence * 0.1)
        }
        Brush::SnakeHook => point + motion * influence,
        Brush::Thumb => {
            let tangent = motion - normal * motion.dot(normal);
            if tangent.length_squared() > 0.0 {
                point + tangent.normalize() * (motion.length() * influence)
            } else {
                point
            }
        }
        Brush::Rotate => {
            let rotation = stroke.delta.unwrap_or(normal * std::f64::consts::FRAC_PI_2);
            let angle = rotation.length() * influence;
            if angle == 0.0 {
                point
            } else {
                (DQuat::from_axis_angle(rotation.normalize(), angle) * (point - center)) + center
            }
        }
        Brush::Nudge => {
            let direction = if motion.length_squared() > 0.0 {
                motion.normalize()
            } else {
                return point;
            };
            point + direction * (radius * 0.25 * influence)
        }
        Brush::Blob => point + normal * (2.0 * influence - influence.powi(2)),
        Brush::Scrape | Brush::Fill => {
            let signed_distance = (point - center).dot(normal);
            let amount = if stroke.brush == Brush::Scrape {
                signed_distance.max(0.0)
            } else {
                signed_distance.min(0.0)
            };
            point - normal * (amount * influence)
        }
        Brush::DrawSharp => point + normal * influence.powf(1.5),
        Brush::Elastic => point + motion * influence.powi(2),
        Brush::PoseLite => {
            let angle = if stroke.delta.is_some() {
                sample_delta.length()
            } else {
                (motion.length() / radius.max(f64::EPSILON)).min(1.0) * std::f64::consts::FRAC_PI_4
            };
            let axis = if stroke.delta.is_some() && sample_delta.length_squared() > 0.0 {
                sample_delta.normalize()
            } else {
                normal
            };
            let rotated =
                (DQuat::from_axis_angle(axis, angle * influence) * (point - center)) + center;
            rotated + motion * influence
        }
        Brush::BoundaryLite => {
            let Some(neighbors) = boundary_neighbors.filter(|neighbors| neighbors.len() == 2)
            else {
                return point;
            };
            let (Some(before), Some(after)) =
                (positions.get(&neighbors[0]), positions.get(&neighbors[1]))
            else {
                return point;
            };
            let tangent = *after - *before;
            if tangent.length_squared() == 0.0 || motion.length_squared() == 0.0 {
                point
            } else {
                let tangent = tangent.normalize();
                point + tangent * (motion.dot(tangent) * influence)
            }
        }
    }
}

fn boundary_adjacency(mesh: &Mesh) -> std::collections::BTreeMap<u32, Vec<u32>> {
    let mut edge_face_count = std::collections::BTreeMap::<[u32; 2], usize>::new();
    for face in &mesh.faces {
        for index in 0..face.vertices.len() {
            let a = face.vertices[index];
            let b = face.vertices[(index + 1) % face.vertices.len()];
            let key = if a < b { [a, b] } else { [b, a] };
            *edge_face_count.entry(key).or_default() += 1;
        }
    }
    let mut adjacency = std::collections::BTreeMap::<u32, Vec<u32>>::new();
    for edge in &mesh.edges {
        let key = if edge.vertices[0] < edge.vertices[1] {
            edge.vertices
        } else {
            [edge.vertices[1], edge.vertices[0]]
        };
        if edge_face_count.get(&key) == Some(&1) {
            adjacency
                .entry(edge.vertices[0])
                .or_default()
                .push(edge.vertices[1]);
            adjacency
                .entry(edge.vertices[1])
                .or_default()
                .push(edge.vertices[0]);
        }
    }
    for neighbors in adjacency.values_mut() {
        neighbors.sort_unstable();
        neighbors.dedup();
    }
    adjacency
}

fn split_long_edges(mesh: &mut Mesh, stroke: &Stroke, edge_length: f64) -> Result<bool> {
    let split_threshold = (edge_length * 1.5).min(f64::MAX);
    let mask_values = read_scalar_attribute(mesh, "sculpt.mask")?;
    let mut changed = false;
    loop {
        let positions = vertex_positions(mesh);
        let mirrors = mirrored_samples(stroke)?;
        let mut candidates = mesh
            .edges
            .iter()
            .filter_map(|edge| {
                let a = *positions.get(&edge.vertices[0])?;
                let b = *positions.get(&edge.vertices[1])?;
                let midpoint = edge_midpoint(a, b);
                let length = a.distance(b);
                let mask_a = mask_values.get(&edge.vertices[0]).copied().unwrap_or(0.0);
                let mask_b = mask_values.get(&edge.vertices[1]).copied().unwrap_or(0.0);
                if length <= split_threshold
                    || midpoint == a
                    || midpoint == b
                    || mask_a > 0.0
                    || mask_b > 0.0
                    || !point_has_influence(midpoint, stroke, &mirrors)
                {
                    return None;
                }
                Some((length, edge.id))
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        let Some((_, edge_id)) = candidates.first().copied() else {
            break;
        };
        if !split_edge(mesh, edge_id)? {
            break;
        }
        changed = true;
    }
    Ok(changed)
}

fn edge_midpoint(a: DVec3, b: DVec3) -> DVec3 {
    DVec3::new(
        f64::midpoint(a.x, b.x),
        f64::midpoint(a.y, b.y),
        f64::midpoint(a.z, b.z),
    )
}

fn split_edge(mesh: &mut Mesh, edge_id: u32) -> Result<bool> {
    let edge = mesh
        .edges
        .iter()
        .find(|edge| edge.id == edge_id)
        .cloned()
        .ok_or_else(|| invalid("dyntopo edge disappeared before subdivision"))?;
    let [a, b] = edge.vertices;
    let positions = vertex_positions(mesh);
    let a_position = *positions
        .get(&a)
        .ok_or_else(|| invalid("dyntopo edge references a missing vertex"))?;
    let b_position = *positions
        .get(&b)
        .ok_or_else(|| invalid("dyntopo edge references a missing vertex"))?;
    let midpoint = edge_midpoint(a_position, b_position);
    if !midpoint.is_finite() || midpoint == a_position || midpoint == b_position {
        return Ok(false);
    }
    if mesh.next_id.vertex == u32::MAX || mesh.next_id.edge == u32::MAX {
        return Err(invalid("dyntopo element ID space is exhausted"));
    }
    let vertex_values = mesh
        .attributes
        .iter()
        .filter_map(|(name, attribute)| {
            let object = attribute.as_object()?;
            if object.get("domain").and_then(Value::as_str) != Some("vertices") {
                return None;
            }
            let values = object.get("values")?.as_object()?;
            let value = interpolate_attribute_values(
                values.get(&format!("v{a}")),
                values.get(&format!("v{b}")),
            )?;
            Some((name.clone(), value))
        })
        .collect::<Vec<_>>();
    let edge_values = mesh
        .attributes
        .iter()
        .filter_map(|(name, attribute)| {
            let object = attribute.as_object()?;
            if object.get("domain").and_then(Value::as_str) != Some("edges") {
                return None;
            }
            object
                .get("values")?
                .as_object()?
                .get(&format!("e{edge_id}"))
                .cloned()
                .map(|value| (name.clone(), value))
        })
        .collect::<Vec<_>>();
    let midpoint_id = mesh
        .insert_vertex(midpoint)
        .map_err(|error| invalid(error.to_string()))?;
    let second_edge_id = mesh
        .insert_edge([midpoint_id, b])
        .map_err(|error| invalid(error.to_string()))?;
    let original_edge = mesh
        .edges
        .iter_mut()
        .find(|candidate| candidate.id == edge_id)
        .ok_or_else(|| invalid("dyntopo edge disappeared during subdivision"))?;
    original_edge.vertices = [a, midpoint_id];
    let edge_key = if a < b { [a, b] } else { [b, a] };
    for face in &mut mesh.faces {
        let mut vertices = Vec::with_capacity(face.vertices.len() + 1);
        for index in 0..face.vertices.len() {
            let current = face.vertices[index];
            let next = face.vertices[(index + 1) % face.vertices.len()];
            vertices.push(current);
            let key = if current < next {
                [current, next]
            } else {
                [next, current]
            };
            if key == edge_key {
                vertices.push(midpoint_id);
            }
        }
        face.vertices = vertices;
    }
    for (name, value) in vertex_values {
        set_attribute_value(mesh, &name, format!("v{midpoint_id}"), value);
    }
    for (name, value) in edge_values {
        set_attribute_value(mesh, &name, format!("e{second_edge_id}"), value);
    }
    Ok(true)
}

fn interpolate_attribute_values(first: Option<&Value>, second: Option<&Value>) -> Option<Value> {
    match (
        first.and_then(Value::as_f64),
        second.and_then(Value::as_f64),
    ) {
        (Some(first), Some(second)) => Some(json!(f64::midpoint(first, second))),
        _ => first.or(second).cloned(),
    }
}

fn set_attribute_value(mesh: &mut Mesh, name: &str, key: String, value: Value) {
    let Some(values) = mesh
        .attributes
        .get_mut(name)
        .and_then(Value::as_object_mut)
        .and_then(|attribute| attribute.get_mut("values"))
        .and_then(Value::as_object_mut)
    else {
        return;
    };
    values.insert(key, value);
}

fn collapse_short_edges(mesh: &mut Mesh, stroke: &Stroke, edge_length: f64) -> Result<bool> {
    let mut changed = false;
    let mask_values = read_scalar_attribute(mesh, "sculpt.mask")?;
    loop {
        let positions = vertex_positions(mesh);
        let adjacency = vertex_adjacency(mesh);
        let mirrors = mirrored_samples(stroke)?;
        let mut candidates = mesh
            .edges
            .iter()
            .filter_map(|edge| {
                let a = *positions.get(&edge.vertices[0])?;
                let b = *positions.get(&edge.vertices[1])?;
                let length = a.distance(b);
                let mask_a = mask_values.get(&edge.vertices[0]).copied().unwrap_or(0.0);
                let mask_b = mask_values.get(&edge.vertices[1]).copied().unwrap_or(0.0);
                if length >= edge_length
                    || mask_a > 0.0
                    || mask_b > 0.0
                    || !point_has_influence(a.lerp(b, 0.5), stroke, &mirrors)
                {
                    return None;
                }
                Some((length, edge.id))
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        let mut collapsed = false;
        for (_, edge_id) in candidates {
            let Some(edge) = mesh.edges.iter().find(|edge| edge.id == edge_id) else {
                continue;
            };
            if !collapse_link_is_valid(mesh, edge.vertices, &adjacency) {
                continue;
            }
            collapse_edge(mesh, edge.vertices)?;
            changed = true;
            collapsed = true;
            break;
        }
        if !collapsed {
            break;
        }
    }
    Ok(changed)
}

fn point_has_influence(point: DVec3, stroke: &Stroke, mirrors: &[InfluenceSample]) -> bool {
    mirrors.iter().any(|sample| {
        let (closest, pressure, radius, strength, _) = sample_influence(point, sample);
        radius > 0.0
            && point.distance(closest) <= radius
            && stroke.falloff.weight(point.distance(closest) / radius) * pressure * strength > 0.0
    })
}

fn collapse_link_is_valid(
    mesh: &Mesh,
    endpoints: [u32; 2],
    adjacency: &std::collections::BTreeMap<u32, Vec<u32>>,
) -> bool {
    let [a, b] = endpoints;
    let a_neighbors = adjacency.get(&a).map(Vec::as_slice).unwrap_or_default();
    let b_neighbors = adjacency.get(&b).map(Vec::as_slice).unwrap_or_default();
    let common = a_neighbors
        .iter()
        .filter(|neighbor| **neighbor != b && b_neighbors.contains(neighbor))
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let mut edge_link = std::collections::BTreeSet::new();
    let mut incident_faces = 0;
    for face in &mesh.faces {
        if face.vertices.contains(&a) && face.vertices.contains(&b) {
            incident_faces += 1;
            edge_link.extend(
                face.vertices
                    .iter()
                    .copied()
                    .filter(|vertex| *vertex != a && *vertex != b),
            );
        }
    }
    incident_faces <= 2 && common == edge_link
}

fn collapse_edge(mesh: &mut Mesh, endpoints: [u32; 2]) -> Result<()> {
    let (keep, remove) = if endpoints[0] < endpoints[1] {
        (endpoints[0], endpoints[1])
    } else {
        (endpoints[1], endpoints[0])
    };
    let midpoint = {
        let positions = vertex_positions(mesh);
        let keep_position = *positions
            .get(&keep)
            .ok_or_else(|| invalid("dyntopo edge references a missing vertex"))?;
        let remove_position = *positions
            .get(&remove)
            .ok_or_else(|| invalid("dyntopo edge references a missing vertex"))?;
        keep_position.lerp(remove_position, 0.5)
    };
    let mut removed_faces = std::collections::BTreeSet::new();
    for face in &mut mesh.faces {
        for vertex in &mut face.vertices {
            if *vertex == remove {
                *vertex = keep;
            }
        }
        face.vertices.dedup();
        if face.vertices.len() > 1 && face.vertices.first() == face.vertices.last() {
            face.vertices.pop();
        }
        if face.vertices.len() < 3 {
            removed_faces.insert(face.id);
        }
    }
    mesh.faces.retain(|face| !removed_faces.contains(&face.id));
    let mut used_edges = std::collections::BTreeSet::new();
    let mut removed_edges = std::collections::BTreeSet::new();
    mesh.edges.retain_mut(|edge| {
        for vertex in &mut edge.vertices {
            if *vertex == remove {
                *vertex = keep;
            }
        }
        let key = if edge.vertices[0] < edge.vertices[1] {
            edge.vertices
        } else {
            [edge.vertices[1], edge.vertices[0]]
        };
        let retain = edge.vertices[0] != edge.vertices[1] && used_edges.insert(key);
        if !retain {
            removed_edges.insert(edge.id);
        }
        retain
    });
    mesh.vertices.retain(|vertex| vertex.id != remove);
    let kept_vertex = mesh
        .vertices
        .iter_mut()
        .find(|vertex| vertex.id == keep)
        .ok_or_else(|| invalid("dyntopo survivor vertex disappeared"))?;
    kept_vertex.co = midpoint;
    for (attribute_name, attribute) in &mut mesh.attributes {
        let Some(object) = attribute.as_object_mut() else {
            continue;
        };
        let domain = object
            .get("domain")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let Some(values) = object.get_mut("values").and_then(Value::as_object_mut) else {
            continue;
        };
        match domain.as_deref() {
            Some("vertices") => {
                let removed_key = format!("v{remove}");
                let keep_key = format!("v{keep}");
                if let Some(removed_value) = values.remove(&removed_key) {
                    let keep_value = values.get(&keep_key).cloned();
                    let combined = if attribute_name == "sculpt.mask" {
                        json!(
                            keep_value
                                .as_ref()
                                .and_then(Value::as_f64)
                                .unwrap_or(0.0)
                                .max(removed_value.as_f64().unwrap_or(0.0))
                        )
                    } else if let (Some(keep_number), Some(remove_number)) = (
                        keep_value.as_ref().and_then(Value::as_f64),
                        removed_value.as_f64(),
                    ) {
                        json!(f64::midpoint(keep_number, remove_number))
                    } else {
                        keep_value.unwrap_or(removed_value)
                    };
                    values.insert(keep_key, combined);
                }
            }
            Some("edges") => {
                for id in &removed_edges {
                    values.remove(&format!("e{id}"));
                }
            }
            Some("faces") => {
                for id in &removed_faces {
                    values.remove(&format!("f{id}"));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Compute maximum stroke influence for each mesh vertex, respecting falloff and symmetry.
///
/// This is shared by geometry paint tools; sculpt masks are applied by [`apply_stroke`].
pub fn influences(mesh: &Mesh, stroke: &Stroke) -> Result<Vec<(u32, f64)>> {
    validate_stroke(stroke)?;
    let mirrored = mirrored_samples(stroke)?;
    Ok(mesh
        .vertices
        .iter()
        .map(|vertex| {
            let value = mirrored
                .iter()
                .map(|sample| {
                    let (closest, pressure, radius, strength, _) =
                        sample_influence(vertex.co, sample);
                    if radius <= 0.0 {
                        return 0.0;
                    }
                    let distance = vertex.co.distance(closest);
                    if distance > radius {
                        0.0
                    } else {
                        stroke.falloff.weight(distance / radius) * pressure * strength
                    }
                })
                .fold(0.0, f64::max);
            (vertex.id, value)
        })
        .collect())
}

/// Smooth one persistent vertex toward its one-ring neighbor average.
pub fn smooth_vertex(mesh: &mut Mesh, vertex_id: u32, factor: f64) -> Result<bool> {
    if !factor.is_finite() || !(0.0..=1.0).contains(&factor) {
        return Err(invalid(
            "smooth factor must be finite and between zero and one",
        ));
    }
    let adjacency = vertex_adjacency(mesh);
    let neighbors = adjacency
        .get(&vertex_id)
        .filter(|neighbors| !neighbors.is_empty())
        .ok_or_else(|| invalid("smooth vertex has no adjacent vertices"))?;
    let positions = vertex_positions(mesh);
    let average = average_neighbor_position(&positions, neighbors);
    let vertex = mesh
        .vertices
        .iter_mut()
        .find(|vertex| vertex.id == vertex_id)
        .ok_or_else(|| invalid("smooth vertex ID does not exist"))?;
    let position = vertex.co.lerp(average, factor);
    let changed = position != vertex.co;
    vertex.co = position;
    Ok(changed)
}

/// Sum squared distances from vertices to the average of their adjacent vertices.
#[must_use]
pub fn laplacian_energy(mesh: &Mesh) -> f64 {
    let positions = vertex_positions(mesh);
    let adjacency = vertex_adjacency(mesh);
    mesh.vertices
        .iter()
        .filter_map(|vertex| {
            let neighbors = adjacency.get(&vertex.id)?;
            if neighbors.is_empty() {
                return None;
            }
            let average = average_neighbor_position(&positions, neighbors);
            Some(vertex.co.distance_squared(average))
        })
        .sum()
}

fn validate_stroke(stroke: &Stroke) -> Result<()> {
    if stroke.samples.is_empty() {
        return Err(invalid("stroke requires at least one sample"));
    }
    for sample in &stroke.samples {
        if !sample.position.is_finite()
            || !sample.pressure.is_finite()
            || !(0.0..=1.0).contains(&sample.pressure)
            || !sample.radius.is_finite()
            || sample.radius <= 0.0
            || !sample.strength.is_finite()
            || !(0.0..=1.0).contains(&sample.strength)
            || !sample.time.is_finite()
            || sample.time < 0.0
        {
            return Err(invalid(
                "stroke samples require finite positions, pressure/strength in [0,1], positive radius, and nonnegative time",
            ));
        }
    }
    for axis in &stroke.symmetry {
        if !["x", "y", "z"].contains(&axis.as_str()) {
            return Err(invalid("symmetry axes must be x, y, or z"));
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct InfluenceSample {
    from: DVec3,
    to: DVec3,
    pressure_from: f64,
    pressure_to: f64,
    radius_from: f64,
    radius_to: f64,
    strength_from: f64,
    strength_to: f64,
    delta_from: DVec3,
    delta_to: DVec3,
}

fn mirrored_samples(stroke: &Stroke) -> Result<Vec<InfluenceSample>> {
    let mut axes = Vec::new();
    for axis in &stroke.symmetry {
        let component = match axis.as_str() {
            "x" => 0,
            "y" => 1,
            "z" => 2,
            _ => return Err(invalid("symmetry axes must be x, y, or z")),
        };
        if !axes.contains(&component) {
            axes.push(component);
        }
    }
    let variant_count = 1_usize << axes.len();
    let mut samples = Vec::with_capacity(stroke.samples.len() * variant_count);
    for variant in 0..variant_count {
        let reflected = stroke
            .samples
            .iter()
            .map(|sample| {
                let mut position = sample.position;
                let mut delta = stroke.delta.unwrap_or(DVec3::ZERO);
                for (bit, component) in axes.iter().enumerate() {
                    if variant & (1_usize << bit) != 0 {
                        position[*component] = -position[*component];
                        delta[*component] = -delta[*component];
                    }
                }
                (sample, position, delta)
            })
            .collect::<Vec<_>>();
        if reflected.len() == 1 {
            let (sample, position, delta) = reflected[0];
            samples.push(InfluenceSample {
                from: position,
                to: position,
                pressure_from: sample.pressure,
                pressure_to: sample.pressure,
                radius_from: sample.radius,
                radius_to: sample.radius,
                strength_from: sample.strength,
                strength_to: sample.strength,
                delta_from: delta,
                delta_to: delta,
            });
        } else {
            for pair in reflected.windows(2) {
                let (first, from, delta_from) = pair[0];
                let (second, to, delta_to) = pair[1];
                samples.push(InfluenceSample {
                    from,
                    to,
                    pressure_from: first.pressure,
                    pressure_to: second.pressure,
                    radius_from: first.radius,
                    radius_to: second.radius,
                    strength_from: first.strength,
                    strength_to: second.strength,
                    delta_from,
                    delta_to,
                });
            }
        }
    }
    Ok(samples)
}

fn sample_influence(point: DVec3, sample: &InfluenceSample) -> (DVec3, f64, f64, f64, DVec3) {
    let segment = sample.to - sample.from;
    let length_squared = segment.length_squared();
    let amount = if length_squared > 0.0 {
        ((point - sample.from).dot(segment) / length_squared).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (
        sample.from.lerp(sample.to, amount),
        sample.pressure_from + (sample.pressure_to - sample.pressure_from) * amount,
        sample.radius_from + (sample.radius_to - sample.radius_from) * amount,
        sample.strength_from + (sample.strength_to - sample.strength_from) * amount,
        sample.delta_from.lerp(sample.delta_to, amount),
    )
}

fn vertex_adjacency(mesh: &Mesh) -> std::collections::BTreeMap<u32, Vec<u32>> {
    let mut adjacency = std::collections::BTreeMap::<u32, Vec<u32>>::new();
    for edge in &mesh.edges {
        adjacency
            .entry(edge.vertices[0])
            .or_default()
            .push(edge.vertices[1]);
        adjacency
            .entry(edge.vertices[1])
            .or_default()
            .push(edge.vertices[0]);
    }
    for neighbors in adjacency.values_mut() {
        neighbors.sort_unstable();
        neighbors.dedup();
    }
    adjacency
}

fn vertex_positions(mesh: &Mesh) -> std::collections::BTreeMap<u32, DVec3> {
    mesh.vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.co))
        .collect()
}

fn average_neighbor_position(
    positions: &std::collections::BTreeMap<u32, DVec3>,
    neighbors: &[u32],
) -> DVec3 {
    let mut sum = DVec3::ZERO;
    let mut count = 0.0_f64;
    for neighbor in neighbors {
        if let Some(position) = positions.get(neighbor) {
            sum += *position;
            count += 1.0;
        }
    }
    if count > 0.0 {
        sum / count
    } else {
        DVec3::ZERO
    }
}

fn vertex_normals(
    mesh: &Mesh,
    positions: &std::collections::BTreeMap<u32, DVec3>,
) -> std::collections::BTreeMap<u32, DVec3> {
    let mut normals = std::collections::BTreeMap::<u32, DVec3>::new();
    for face in &mesh.faces {
        if face.vertices.len() < 3 {
            continue;
        }
        let Some(a) = positions.get(&face.vertices[0]).copied() else {
            continue;
        };
        let Some(b) = positions.get(&face.vertices[1]).copied() else {
            continue;
        };
        let Some(c) = positions.get(&face.vertices[2]).copied() else {
            continue;
        };
        let normal = (b - a).cross(c - a);
        for id in &face.vertices {
            *normals.entry(*id).or_default() += normal;
        }
    }
    for normal in normals.values_mut() {
        *normal = if normal.length_squared() > 0.0 {
            normal.normalize()
        } else {
            DVec3::Z
        };
    }
    for vertex in &mesh.vertices {
        normals.entry(vertex.id).or_insert(DVec3::Z);
    }
    normals
}

fn read_scalar_attribute(mesh: &Mesh, name: &str) -> Result<std::collections::BTreeMap<u32, f64>> {
    let Some(attribute) = mesh.attributes.get(name) else {
        return Ok(std::collections::BTreeMap::default());
    };
    let values = attribute
        .get("values")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            invalid(format!(
                "mesh attribute `{name}` must contain a values object"
            ))
        })?;
    values
        .iter()
        .filter_map(|(key, value)| {
            let id = key.strip_prefix('v')?.parse::<u32>().ok()?;
            let number = value.as_f64()?;
            Some((id, number))
        })
        .try_fold(
            std::collections::BTreeMap::new(),
            |mut result, (id, value)| {
                if !value.is_finite() {
                    return Err(invalid(format!(
                        "mesh attribute `{name}` values must be finite"
                    )));
                }
                result.insert(id, value);
                Ok(result)
            },
        )
}

fn deterministic_scale(seed: u64, position: DVec3, symmetry: &[String]) -> f64 {
    let mut value = seed;
    for (component, axis) in ["x", "y", "z"].iter().enumerate() {
        let coordinate = position[component];
        let coordinate = if symmetry.iter().any(|candidate| candidate == axis) {
            coordinate.abs()
        } else {
            coordinate
        };
        value ^= coordinate.to_bits().wrapping_mul(0x9e37_79b9_7f4a_7c15);
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    }
    value ^= value >> 31;
    let bytes = value.to_le_bytes();
    let unit = f64::from(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        / f64::from(u32::MAX);
    0.98 + unit * 0.02
}

fn invalid(message: impl Into<String>) -> PotError {
    PotError::with_details(
        crate::error::ErrorCode::InvalidOperation,
        message,
        json!({}),
    )
}

impl Falloff {
    #[must_use]
    pub fn weight(self, normalized_distance: f64) -> f64 {
        let t = normalized_distance.clamp(0.0, 1.0);
        match self {
            Self::Smooth => 1.0 - t * t * (3.0 - 2.0 * t),
            Self::Sphere => (1.0 - t * t).sqrt(),
            Self::Root => (1.0 - t).sqrt(),
            Self::Sharp => (1.0 - t).powi(2),
            Self::Linear => 1.0 - t,
            Self::Constant => 1.0,
        }
    }
}
#[cfg(test)]
mod tests {
    #![expect(
        clippy::unwrap_used,
        reason = "sculpt tests use valid deterministic fixtures"
    )]
    use super::*;
    use serde_json::json;

    fn test_mesh() -> Mesh {
        Mesh::from_positions_and_faces(vec![DVec3::ZERO, DVec3::X, DVec3::Y], vec![vec![0, 1, 2]])
            .unwrap()
    }

    fn sample(position: DVec3) -> StrokeSample {
        StrokeSample {
            position,
            pressure: 1.0,
            radius: 3.0,
            strength: 1.0,
            time: 0.0,
        }
    }

    fn test_stroke(brush: Brush) -> Stroke {
        let center_z = match brush {
            Brush::Scrape | Brush::Flatten => -0.25,
            Brush::Fill => 0.25,
            _ => 0.0,
        };
        let delta = match brush {
            Brush::Grab | Brush::Thumb | Brush::Nudge | Brush::Elastic | Brush::BoundaryLite => {
                Some(DVec3::new(0.25, 0.0, 0.0))
            }
            Brush::Rotate | Brush::PoseLite => Some(DVec3::new(0.0, 0.0, 0.25)),
            _ => None,
        };
        Stroke {
            brush,
            samples: vec![
                sample(DVec3::new(0.0, 0.0, center_z)),
                sample(DVec3::new(0.5, 0.0, center_z)),
            ],
            falloff: Falloff::Smooth,
            symmetry: Vec::new(),
            seed: 17,
            delta,
        }
    }

    #[test]
    fn every_brush_deforms_covered_geometry_deterministically() {
        for brush in [
            Brush::Draw,
            Brush::ClayStrips,
            Brush::Inflate,
            Brush::Grab,
            Brush::Smooth,
            Brush::Flatten,
            Brush::Pinch,
            Brush::Crease,
            Brush::Layer,
            Brush::SnakeHook,
            Brush::Thumb,
            Brush::Rotate,
            Brush::Nudge,
            Brush::Blob,
            Brush::Scrape,
            Brush::Fill,
            Brush::DrawSharp,
            Brush::Elastic,
            Brush::PoseLite,
            Brush::BoundaryLite,
        ] {
            let stroke = test_stroke(brush);
            let mut first = test_mesh();
            let mut second = test_mesh();
            assert!(apply_stroke(&mut first, &stroke).unwrap(), "{brush:?}");
            assert!(apply_stroke(&mut second, &stroke).unwrap(), "{brush:?}");
            assert_ne!(first, test_mesh(), "{brush:?}");
            assert_eq!(first, second, "{brush:?}");
            first.validate().unwrap();
        }
    }

    #[test]
    fn dyntopo_collapses_short_edges_in_the_brush_and_remaps_attributes() {
        let mut mesh = test_mesh();
        mesh.vertices[1].co = DVec3::new(0.2, 0.0, 0.0);
        mesh.attributes.insert(
            "test.weights".to_owned(),
            json!({"domain":"vertices","values":{"v0":2.0,"v1":4.0}}),
        );
        mesh.attributes.insert(
            "test.edges".to_owned(),
            json!({"domain":"edges","values":{"e0":7}}),
        );
        mesh.attributes.insert(
            "sculpt.face_set".to_owned(),
            json!({"domain":"faces","values":{"f0":3}}),
        );
        let mut stroke = Stroke {
            brush: Brush::Draw,
            samples: vec![sample(DVec3::new(0.1, 0.0, 0.0))],
            falloff: Falloff::Smooth,
            symmetry: Vec::new(),
            seed: 0,
            delta: None,
        };

        assert!(apply_stroke_with_dyntopo(&mut mesh, &stroke, Some(2.0)).unwrap());
        assert_eq!(mesh.vertices.len(), 2);
        assert_eq!(mesh.faces.len(), 0);
        assert_eq!(mesh.attributes["test.weights"]["values"]["v0"], json!(3.0));
        assert!(
            mesh.attributes["test.weights"]["values"]
                .get("v1")
                .is_none()
        );
        assert!(mesh.attributes["test.edges"]["values"].get("e0").is_none());
        assert!(
            mesh.attributes["sculpt.face_set"]["values"]
                .get("f0")
                .is_none()
        );
        mesh.validate().unwrap();

        let mut outside = test_mesh();
        outside.vertices[1].co = DVec3::new(0.2, 0.0, 0.0);
        stroke.samples = vec![sample(DVec3::splat(10.0))];
        let before = outside.clone();
        assert!(!apply_stroke_with_dyntopo(&mut outside, &stroke, Some(2.0)).unwrap());
        assert_eq!(outside, before);
    }

    #[test]
    fn dyntopo_subdivides_long_brush_edges_and_interpolates_attributes() {
        let mut mesh = test_mesh();
        mesh.attributes.insert(
            "test.weights".to_owned(),
            json!({"domain":"vertices","values":{"v0":0.0,"v1":1.0,"v2":2.0}}),
        );
        mesh.attributes.insert(
            "test.edges".to_owned(),
            json!({"domain":"edges","values":{"e0":2.0,"e1":4.0,"e2":6.0}}),
        );
        let stroke = Stroke {
            brush: Brush::Draw,
            samples: vec![sample(DVec3::ZERO)],
            falloff: Falloff::Smooth,
            symmetry: Vec::new(),
            seed: 0,
            delta: None,
        };

        assert!(dyntopo_under_brush(&mut mesh, &stroke, 0.5).unwrap());
        assert_eq!(mesh.vertices.len(), 6);
        assert_eq!(mesh.faces.len(), 1);
        assert_eq!(mesh.faces[0].vertices.len(), 6);
        assert_eq!(mesh.attributes["test.weights"]["values"]["v3"], json!(1.5));
        assert_eq!(mesh.attributes["test.edges"]["values"]["e3"], json!(4.0));
        mesh.validate().unwrap();
    }
}
