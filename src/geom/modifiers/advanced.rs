use std::collections::{HashMap, HashSet};

use glam::{DQuat, DVec3};
use serde_json::Value;

use crate::model::Modifier;

use super::{
    bool_param, build_mesh, closest_point_triangle, id_indices, invalid_parameter, loose_edges,
    number_param, polygons, positions, string_param, uint_param, validated, vec3_param,
};
use crate::{
    error::{ErrorCode, PotError, Result},
    geom::Mesh,
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum SimpleDeformMethod {
    Twist,
    Bend,
    Taper,
    Stretch,
}
const SIMPLE_DEFORM_AXIS_MAP: [[usize; 3]; 3] = [[1, 2, 0], [2, 0, 1], [0, 1, 2]];

pub(super) fn simple_deform(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    let method = match string_param(modifier, "deform_method", "TWIST")? {
        "TWIST" | "twist" => SimpleDeformMethod::Twist,
        "BEND" | "bend" => SimpleDeformMethod::Bend,
        "TAPER" | "taper" => SimpleDeformMethod::Taper,
        "STRETCH" | "stretch" => SimpleDeformMethod::Stretch,
        _ => {
            return Err(invalid_parameter(
                modifier,
                "deform_method",
                "TWIST, BEND, TAPER, or STRETCH",
            ));
        }
    };
    let amount = if matches!(method, SimpleDeformMethod::Twist | SimpleDeformMethod::Bend) {
        number_param(modifier, "angle", f64::from(std::f32::consts::FRAC_PI_4))?
    } else {
        number_param(modifier, "factor", 1.0)?
    };
    let axis = simple_deform_axis(modifier)?;
    let limits = simple_deform_limits(modifier)?;
    let limit_axis = if method == SimpleDeformMethod::Bend {
        if axis == 2 { 0 } else { 2 }
    } else {
        axis
    };
    let Some(first_vertex) = mesh.vertices.first() else {
        return Ok(mesh.clone());
    };
    let mut minimum = first_vertex.co[limit_axis];
    let mut maximum = minimum;
    for vertex in mesh.vertices.iter().skip(1) {
        let value = vertex.co[limit_axis];
        minimum = minimum.min(value);
        maximum = maximum.max(value);
    }
    let extent = maximum - minimum;
    let lower = minimum + extent * limits[0];
    let upper = minimum + extent * limits[1];
    let factor = amount / (upper - lower).max(f64::from(f32::EPSILON));
    if factor == 0.0 || (method == SimpleDeformMethod::Bend && factor.abs() < 1.0e-6) {
        return Ok(mesh.clone());
    }

    let mut result = mesh.clone();
    for vertex in &mut result.vertices {
        vertex.co = simple_deform_point(vertex.co, method, axis, limit_axis, factor, lower, upper);
    }
    validated(&result)?;
    Ok(result)
}

fn simple_deform_axis(modifier: &Modifier) -> Result<usize> {
    match string_param(modifier, "deform_axis", "X")? {
        "X" | "x" => Ok(0),
        "Y" | "y" => Ok(1),
        "Z" | "z" => Ok(2),
        _ => Err(invalid_parameter(modifier, "deform_axis", "X, Y, or Z")),
    }
}

fn simple_deform_limits(modifier: &Modifier) -> Result<[f64; 2]> {
    let Some(value) = modifier.params.get("limits") else {
        return Ok([0.0, 1.0]);
    };
    let Some(values) = value.as_array().filter(|values| values.len() == 2) else {
        return Err(invalid_parameter(
            modifier,
            "limits",
            "two normalized values",
        ));
    };
    let first = values[0]
        .as_f64()
        .ok_or_else(|| invalid_parameter(modifier, "limits", "two normalized values"))?
        .clamp(0.0, 1.0);
    let second = values[1]
        .as_f64()
        .ok_or_else(|| invalid_parameter(modifier, "limits", "two normalized values"))?
        .clamp(0.0, 1.0);
    Ok([first.min(second), first.max(second)])
}

fn simple_deform_point(
    position: DVec3,
    method: SimpleDeformMethod,
    axis: usize,
    limit_axis: usize,
    factor: f64,
    lower: f64,
    upper: f64,
) -> DVec3 {
    let mut coordinates = position.to_array();
    let mut cut = [0.0; 3];
    let original = coordinates[limit_axis];
    coordinates[limit_axis] = original.clamp(lower, upper);
    cut[limit_axis] = original - coordinates[limit_axis];

    if method == SimpleDeformMethod::Bend {
        return DVec3::from_array(simple_deform_bend(coordinates, cut, axis, factor));
    }

    let map = SIMPLE_DEFORM_AXIS_MAP[axis];
    let mut mapped = [
        coordinates[map[0]],
        coordinates[map[1]],
        coordinates[map[2]],
    ];
    let mapped_cut = [cut[map[0]], cut[map[1]], cut[map[2]]];
    let [x, y, z] = mapped;
    match method {
        SimpleDeformMethod::Twist => {
            let angle = z * factor;
            let (sine, cosine) = angle.sin_cos();
            mapped[0] = x * cosine - y * sine;
            mapped[1] = x * sine + y * cosine;
        }
        SimpleDeformMethod::Taper => {
            let scale = z * factor;
            mapped[0] = x + x * scale;
            mapped[1] = y + y * scale;
        }
        SimpleDeformMethod::Stretch => {
            let scale = z * z * factor - factor + 1.0;
            mapped[0] = x * scale;
            mapped[1] = y * scale;
            mapped[2] = z * (1.0 + factor);
        }
        SimpleDeformMethod::Bend => {}
    }
    for (coordinate, cut) in mapped.iter_mut().zip(mapped_cut) {
        *coordinate += cut;
    }
    let mut result = [0.0; 3];
    for (index, original_axis) in map.into_iter().enumerate() {
        result[original_axis] = mapped[index];
    }
    DVec3::from_array(result)
}

fn simple_deform_bend(coordinates: [f64; 3], cut: [f64; 3], axis: usize, factor: f64) -> [f64; 3] {
    let [x, y, z] = coordinates;
    match axis {
        0 => {
            let (sine, cosine) = (z * factor).sin_cos();
            [
                x + cut[0],
                y * cosine + (1.0 - cosine) / factor + sine * cut[2],
                -(y - 1.0 / factor) * sine + cosine * cut[2],
            ]
        }
        1 => {
            let (sine, cosine) = (z * factor).sin_cos();
            [
                x * cosine + (1.0 - cosine) / factor + sine * cut[2],
                y + cut[1],
                -(x - 1.0 / factor) * sine + cosine * cut[2],
            ]
        }
        _ => {
            let (sine, cosine) = (x * factor).sin_cos();
            [
                -(y - 1.0 / factor) * sine + cosine * cut[0],
                y * cosine + (1.0 - cosine) / factor + sine * cut[0],
                z + cut[2],
            ]
        }
    }
}

pub(super) fn screw(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    let steps = usize::try_from(uint_param(modifier, "steps", 16)?)
        .map_err(|_| invalid_parameter(modifier, "steps", "an integer from 3 to 4096"))?;
    if !(3..=4096).contains(&steps) {
        return Err(invalid_parameter(
            modifier,
            "steps",
            "an integer from 3 to 4096",
        ));
    }
    let angle = number_param(modifier, "angle", std::f64::consts::TAU)?;
    let axis_parameter = modifier.params.get("axis").or_else(|| {
        crate::params::default_value(
            crate::params::ParameterFamily::Modifier,
            &modifier.modifier_type,
            "axis",
        )
    });
    let axis = match axis_parameter {
        None => DVec3::Z,
        Some(Value::String(axis)) => match axis.as_str() {
            "X" => DVec3::X,
            "Y" => DVec3::Y,
            "Z" => DVec3::Z,
            _ => return Err(invalid_parameter(modifier, "axis", "X, Y, or Z")),
        },
        Some(_) => return Err(invalid_parameter(modifier, "axis", "X, Y, or Z")),
    };
    let center = DVec3::ZERO;
    let height = number_param(modifier, "screw_offset", 0.0)?;
    let source_positions = positions(mesh);
    let vertex_indices = id_indices(mesh);
    let source_edges = mesh
        .edges
        .iter()
        .map(|edge| {
            let first = vertex_indices
                .get(&edge.vertices[0])
                .copied()
                .ok_or_else(|| {
                    invalid_parameter(modifier, "mesh", "edges referencing existing vertices")
                })?;
            let second = vertex_indices
                .get(&edge.vertices[1])
                .copied()
                .ok_or_else(|| {
                    invalid_parameter(modifier, "mesh", "edges referencing existing vertices")
                })?;
            Ok([first, second])
        })
        .collect::<Result<Vec<_>>>()?;
    if source_edges.is_empty() {
        return Err(invalid_parameter(
            modifier,
            "mesh",
            "at least one profile edge",
        ));
    }
    let closed = (angle.abs() - std::f64::consts::TAU).abs() <= 1.0e-10;
    let ring_count = if closed { steps } else { steps + 1 };
    let mut points = Vec::with_capacity(source_positions.len() * ring_count);
    for step in 0..ring_count {
        let t = step as f64 / steps as f64;
        let rotation = DQuat::from_axis_angle(axis, angle * t);
        for point in &source_positions {
            let offset = *point - center;
            points.push(center + rotation * offset + axis * (height * t));
        }
    }
    let segments = if closed { ring_count } else { ring_count - 1 };
    let mut faces = Vec::with_capacity(segments * source_edges.len());
    for step in 0..segments {
        let next_step = (step + 1) % ring_count;
        for [first, second] in &source_edges {
            faces.push((
                vec![
                    step * source_positions.len() + first,
                    next_step * source_positions.len() + first,
                    next_step * source_positions.len() + second,
                    step * source_positions.len() + second,
                ],
                0,
            ));
        }
    }
    build_mesh(points, faces, Vec::new(), &mesh.attributes)
}

pub(super) fn shrinkwrap(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    let target = modifier
        .runtime
        .target_mesh
        .as_ref()
        .ok_or_else(|| invalid_parameter(modifier, "target", "a mesh object operand"))?;
    let target: Mesh = serde_json::from_value(target.clone())
        .map_err(|_| invalid_parameter(modifier, "target", "a mesh object operand"))?;
    validated(&target)?;
    if target.vertices.is_empty() {
        return Err(invalid_parameter(modifier, "target", "a non-empty mesh"));
    }
    let method =
        string_param(modifier, "wrap_method", "NEAREST_SURFACEPOINT")?.to_ascii_uppercase();
    let offset = number_param(modifier, "offset", 0.0)?;
    let wrap_mode = string_param(modifier, "wrap_mode", "ON_SURFACE")?.to_ascii_uppercase();
    let mut result = mesh.clone();
    let triangles = target_triangles(&target)?;
    for vertex in &mut result.vertices {
        let (point, normal) = match method.as_str() {
            "NEAREST_VERTEX" => {
                let nearest = target
                    .vertices
                    .iter()
                    .min_by(|left, right| {
                        left.co
                            .distance_squared(vertex.co)
                            .total_cmp(&right.co.distance_squared(vertex.co))
                    })
                    .ok_or_else(|| invalid_parameter(modifier, "target", "a non-empty mesh"))?;
                let normal =
                    nearest_surface(&triangles, nearest.co).map_or(DVec3::Z, |(_, normal)| normal);
                (nearest.co, normal)
            }
            "NEAREST_SURFACEPOINT" | "NEAREST_FACE_INTERPOLATED" => {
                nearest_surface(&triangles, vertex.co).ok_or_else(|| {
                    invalid_parameter(modifier, "target", "a mesh with at least one face")
                })?
            }
            "PROJECT" => project_surface(
                &triangles,
                vertex.co,
                vec3_param(modifier, "project_axis", -DVec3::Z)?,
                modifier,
            )?,
            _ => {
                return Err(invalid_parameter(
                    modifier,
                    "wrap_method",
                    "NEAREST_SURFACEPOINT, NEAREST_VERTEX, NEAREST_FACE_INTERPOLATED, or PROJECT",
                ));
            }
        };
        let offset_direction = match wrap_mode.as_str() {
            "ON_SURFACE" | "OUTSIDE_SURFACE" => {
                (vertex.co - point).try_normalize().unwrap_or(normal)
            }
            "INSIDE" => -normal,
            "OUTSIDE" | "ABOVE_SURFACE" => normal,
            _ => {
                return Err(invalid_parameter(
                    modifier,
                    "wrap_mode",
                    "ON_SURFACE, INSIDE, OUTSIDE, OUTSIDE_SURFACE, or ABOVE_SURFACE",
                ));
            }
        };
        vertex.co = point + offset_direction * offset;
    }
    validated(&result)?;
    Ok(result)
}

pub(super) fn cast(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    let cast_type = string_param(modifier, "cast_type", "SPHERE")?.to_ascii_uppercase();
    let factor = number_param(modifier, "factor", 1.0)?;
    if factor == 0.0 {
        return Ok(mesh.clone());
    }
    let radius = number_param(modifier, "radius", 1.0)?;
    let size = number_param(modifier, "size", 1.0)?;
    if radius <= 0.0 || size <= 0.0 {
        return Err(invalid_parameter(
            modifier,
            "radius",
            "positive radius and size",
        ));
    }
    let use_radius_as_size = bool_param(modifier, "use_radius_as_size", true)?;
    let cast_size = if use_radius_as_size { radius } else { size };
    let center = vec3_param(modifier, "origin", DVec3::ZERO)?;
    let mut result = mesh.clone();
    for vertex in &mut result.vertices {
        let point = vertex.co - center;
        let mut direction = point;
        if cast_type == "CYLINDER" {
            direction.z = 0.0;
        }
        if radius > f64::EPSILON && direction.length() > radius {
            continue;
        }
        let target = match cast_type.as_str() {
            "SPHERE" => point
                .try_normalize()
                .map_or(DVec3::ZERO, |unit| unit * cast_size),
            "CYLINDER" => {
                direction
                    .try_normalize()
                    .map_or(DVec3::ZERO, |unit| unit * cast_size)
                    + DVec3::Z * point.z
            }
            "CUBOID" => point.clamp(DVec3::splat(-cast_size), DVec3::splat(cast_size)),
            _ => {
                return Err(invalid_parameter(
                    modifier,
                    "cast_type",
                    "SPHERE, CYLINDER, or CUBOID",
                ));
            }
        };
        vertex.co = center + point.lerp(target, factor);
    }
    validated(&result)?;
    Ok(result)
}

pub(super) fn wave(mesh: &Mesh, modifier: &Modifier, frame: f64) -> Result<Mesh> {
    let speed = number_param(modifier, "speed", 0.25)?;
    let height = number_param(modifier, "height", 0.5)?;
    let width = number_param(modifier, "width", 1.5)?;
    let narrowness = number_param(modifier, "narrowness", 1.5)?;
    let time_offset = number_param(modifier, "time_offset", 0.0)?;
    let lifetime = number_param(modifier, "lifetime", 0.0)?;
    let mut damping_time = number_param(modifier, "damping_time", 10.0)?;
    let falloff_radius = number_param(modifier, "falloff_radius", 0.0)?;
    let start_x = number_param(modifier, "start_position_x", 0.0)?;
    let start_y = number_param(modifier, "start_position_y", 0.0)?;
    let use_x = bool_param(modifier, "use_x", true)?;
    let use_y = bool_param(modifier, "use_y", true)?;
    let use_cyclic = bool_param(modifier, "use_cyclic", true)?;
    if width <= 0.0 || narrowness <= 0.0 || falloff_radius < 0.0 {
        return Err(invalid_parameter(
            modifier,
            "width",
            "positive width and narrowness with non-negative falloff radius",
        ));
    }
    if damping_time == 0.0 {
        damping_time = 10.0;
    }
    let mut life_factor = height;
    let time = frame - time_offset;
    if lifetime != 0.0 && time > lifetime {
        let elapsed = time - lifetime;
        life_factor = if elapsed > damping_time {
            0.0
        } else {
            height * (1.0 - (elapsed / damping_time).sqrt())
        };
    }
    if life_factor == 0.0 {
        return Ok(mesh.clone());
    }
    let min_factor = (-(width * narrowness).powi(2)).exp();
    let use_normal = bool_param(modifier, "use_normal", false)?;
    let mut result = mesh.clone();
    let normals = if use_normal {
        let indices = id_indices(mesh);
        let positions = positions(mesh);
        let mut normals = vec![DVec3::ZERO; mesh.vertices.len()];
        let triangles = mesh
            .triangulate()
            .map_err(|_| invalid_parameter(modifier, "mesh", "valid triangulation"))?;
        for triangle in triangles {
            let [first, second, third] =
                triangle.map(|id| indices.get(&id).copied().unwrap_or(usize::MAX));
            if [first, second, third].contains(&usize::MAX) {
                continue;
            }
            let normal = (positions[second] - positions[first])
                .cross(positions[third] - positions[first])
                .normalize_or_zero();
            normals[first] += normal;
            normals[second] += normal;
            normals[third] += normal;
        }
        for normal in &mut normals {
            *normal = normal.normalize_or_zero();
        }
        normals
    } else {
        Vec::new()
    };
    let use_normal_x = bool_param(modifier, "use_normal_x", true)?;
    let use_normal_y = bool_param(modifier, "use_normal_y", true)?;
    let use_normal_z = bool_param(modifier, "use_normal_z", true)?;
    for (index, vertex) in result.vertices.iter_mut().enumerate() {
        let x = vertex.co.x - start_x;
        let y = vertex.co.y - start_y;
        let distance = match (use_x, use_y) {
            (true, true) => x.hypot(y),
            (true, false) => x.abs(),
            (false, true) => y.abs(),
            (false, false) => 0.0,
        };
        let mut phase = match (use_x, use_y) {
            (true, true) => distance,
            (true, false) => x,
            (false, true) => y,
            (false, false) => 0.0,
        } - time * speed;
        if use_cyclic {
            phase = (phase - width) % (2.0 * width) + width;
        }
        if phase <= -width || phase >= width {
            continue;
        }
        let normalized = phase * narrowness;
        let mut amplitude = (-normalized * normalized).exp() - min_factor;
        if falloff_radius > 0.0 {
            amplitude *= (1.0 - distance / falloff_radius).clamp(0.0, 1.0);
        }
        amplitude *= life_factor;
        if use_normal {
            let normal = normals[index];
            if use_normal_x {
                vertex.co.x += amplitude * normal.x;
            }
            if use_normal_y {
                vertex.co.y += amplitude * normal.y;
            }
            if use_normal_z {
                vertex.co.z += amplitude * normal.z;
            }
        } else {
            vertex.co.z += amplitude;
        }
    }
    validated(&result)?;
    Ok(result)
}

pub(super) fn warp(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    let from = modifier
        .runtime
        .warp_from
        .map_or(DVec3::ZERO, DVec3::from_array);
    let to = modifier.runtime.warp_to.map_or(DVec3::Z, DVec3::from_array);
    let displacement = to - from;
    let strength = number_param(modifier, "strength", 0.0)?;
    let radius = number_param(modifier, "falloff_radius", 0.0)?;
    let falloff = string_param(modifier, "falloff_type", "SMOOTH")?.to_ascii_uppercase();
    let mut result = mesh.clone();
    for vertex in &mut result.vertices {
        let distance = vertex.co.distance(from);
        let weight = if falloff == "NONE" {
            1.0
        } else if radius > 0.0 && distance < radius {
            let linear = ((radius - distance) / radius).clamp(0.0, 1.0);
            match falloff.as_str() {
                "CURVE" | "LINEAR" => linear,
                "SHARP" => linear * linear,
                "SMOOTH" => 3.0 * linear * linear - 2.0 * linear * linear * linear,
                "ROOT" => linear.sqrt(),
                "CONST" => 1.0,
                "SPHERE" => (2.0 * linear - linear * linear).sqrt(),
                "INVSQUARE" => linear * (2.0 - linear),
                _ => {
                    return Err(invalid_parameter(
                        modifier,
                        "falloff_type",
                        "NONE, CURVE, SHARP, SMOOTH, ROOT, LINEAR, CONST, SPHERE, or INVSQUARE",
                    ));
                }
            }
        } else {
            0.0
        };
        vertex.co += displacement * (strength * weight);
    }
    validated(&result)?;
    Ok(result)
}

pub(super) fn laplacian_smooth(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    let iterations = usize::try_from(uint_param(modifier, "iterations", 1)?)
        .map_err(|_| invalid_parameter(modifier, "iterations", "an integer from 0 to 10000"))?;
    if iterations > 10000 {
        return Err(invalid_parameter(
            modifier,
            "iterations",
            "an integer from 0 to 10000",
        ));
    }
    let lambda = number_param(modifier, "lambda_factor", 0.5)?;
    let lambda_border = number_param(modifier, "lambda_border", 0.5)?;
    let normalized = bool_param(modifier, "use_normalized", false)?;
    let preserve_volume = bool_param(modifier, "use_volume_preserve", false)?;
    let axes = [
        bool_param(modifier, "use_x", true)?,
        bool_param(modifier, "use_y", true)?,
        bool_param(modifier, "use_z", true)?,
    ];
    laplacian_smooth_mesh(
        mesh,
        modifier,
        LaplacianSmoothSettings {
            iterations,
            lambda,
            lambda_border,
            normalized,
            preserve_volume,
            axes,
        },
    )
}

#[derive(Clone, Copy)]
struct LaplacianSmoothSettings {
    iterations: usize,
    lambda: f64,
    lambda_border: f64,
    normalized: bool,
    preserve_volume: bool,
    axes: [bool; 3],
}

#[expect(
    clippy::too_many_lines,
    reason = "the modifier's matrix assembly and repeated volume-preserving solves are one algorithm"
)]
fn laplacian_smooth_mesh(
    mesh: &Mesh,
    modifier: &Modifier,
    settings: LaplacianSmoothSettings,
) -> Result<Mesh> {
    const MIN_AREA: f64 = 1.0e-5;
    if settings.iterations == 0 || mesh.vertices.is_empty() || mesh.faces.is_empty() {
        return Ok(mesh.clone());
    }
    if !settings.axes.into_iter().any(|enabled| enabled) {
        return Ok(mesh.clone());
    }

    let positions = positions(mesh);
    let index_by_id = id_indices(mesh);
    let vertex_count = positions.len();
    let mut edge_counts = vec![0_usize; vertex_count];
    let mut face_counts = vec![0_usize; vertex_count];
    let mut zero_laplacian = vec![false; vertex_count];
    let mut ring_areas = vec![0.0; vertex_count];
    let mut vertex_weights = vec![0.0; vertex_count];
    let mut vertex_lengths = vec![0.0; vertex_count];
    let mut edge_weights = Vec::with_capacity(mesh.edges.len());
    for edge in &mesh.edges {
        let first = *index_by_id.get(&edge.vertices[0]).ok_or_else(|| {
            invalid_parameter(modifier, "mesh", "edges referencing existing vertices")
        })?;
        let second = *index_by_id.get(&edge.vertices[1]).ok_or_else(|| {
            invalid_parameter(modifier, "mesh", "edges referencing existing vertices")
        })?;
        edge_counts[first] += 1;
        edge_counts[second] += 1;
        let length = positions[first].distance(positions[second]);
        if length < MIN_AREA {
            zero_laplacian[first] = true;
            zero_laplacian[second] = true;
            edge_weights.push(length);
        } else {
            edge_weights.push(length.recip());
        }
    }

    let mut face_indices = Vec::with_capacity(mesh.faces.len());
    let mut face_offsets = Vec::with_capacity(mesh.faces.len());
    let mut total_corners = 0;
    for face in &mesh.faces {
        let indices = face
            .vertices
            .iter()
            .map(|id| {
                index_by_id.get(id).copied().ok_or_else(|| {
                    invalid_parameter(modifier, "mesh", "faces referencing existing vertices")
                })
            })
            .collect::<Result<Vec<_>>>()?;
        face_offsets.push(total_corners);
        total_corners += indices.len();
        face_indices.push(indices);
    }
    let mut face_weights = vec![[0.0; 3]; total_corners];
    for (face_index, indices) in face_indices.iter().enumerate() {
        if indices.len() < 3 {
            continue;
        }
        let face_start = face_offsets[face_index];
        for corner in 0..indices.len() {
            let previous = indices[(corner + indices.len() - 1) % indices.len()];
            let current = indices[corner];
            let next = indices[(corner + 1) % indices.len()];
            face_counts[current] += 1;
            let previous_position = positions[previous];
            let current_position = positions[current];
            let next_position = positions[next];
            let area = (current_position - previous_position)
                .cross(next_position - previous_position)
                .length()
                * 0.5;
            if area < MIN_AREA {
                zero_laplacian[current] = true;
            }
            ring_areas[previous] += area;
            ring_areas[current] += area;
            ring_areas[next] += area;
            let w1 = cotangent_weight(current_position, next_position, previous_position) * 0.5;
            let w2 = cotangent_weight(next_position, previous_position, current_position) * 0.5;
            let w3 = cotangent_weight(previous_position, current_position, next_position) * 0.5;
            let weights = &mut face_weights[face_start + corner];
            weights[0] += w1;
            weights[1] += w2;
            weights[2] += w3;
            vertex_weights[current] += w2 + w3;
            vertex_weights[next] += w1 + w3;
            vertex_weights[previous] += w1 + w2;
        }
    }
    for (edge_index, edge) in mesh.edges.iter().enumerate() {
        let first = index_by_id[&edge.vertices[0]];
        let second = index_by_id[&edge.vertices[1]];
        if edge_counts[first] != face_counts[first] && edge_counts[second] != face_counts[second] {
            vertex_lengths[first] += edge_weights[edge_index];
            vertex_lengths[second] += edge_weights[edge_index];
        }
    }

    let mut rows = vec![Vec::<(usize, f64)>::new(); vertex_count];
    let mut row_weights = vec![0.0; vertex_count];
    for index in 0..vertex_count {
        let interior = edge_counts[index] == face_counts[index];
        if zero_laplacian[index] {
            laplacian_matrix_add(&mut rows, index, index, 1.0);
        } else if interior {
            let diagonal = if settings.normalized {
                settings.lambda.abs()
            } else if ring_areas[index] > 0.0 {
                settings.lambda.abs() / (4.0 * ring_areas[index])
            } else {
                0.0
            };
            let denominator = if settings.normalized {
                vertex_weights[index]
            } else {
                4.0 * vertex_weights[index] * ring_areas[index]
            };
            row_weights[index] = if denominator == 0.0 {
                0.0
            } else {
                -settings.lambda.abs() / denominator
            };
            laplacian_matrix_add(&mut rows, index, index, 1.0 + diagonal);
        } else {
            let diagonal = settings.lambda_border.abs() * 2.0;
            row_weights[index] = if vertex_lengths[index] == 0.0 {
                0.0
            } else {
                -settings.lambda_border.abs() * 2.0 / vertex_lengths[index]
            };
            laplacian_matrix_add(&mut rows, index, index, 1.0 + diagonal);
        }
    }

    for (face_index, indices) in face_indices.iter().enumerate() {
        if indices.len() < 3 {
            continue;
        }
        let face_start = face_offsets[face_index];
        for corner in 0..indices.len() {
            let previous = indices[(corner + indices.len() - 1) % indices.len()];
            let current = indices[corner];
            let next = indices[(corner + 1) % indices.len()];
            if edge_counts[current] == face_counts[current] && !zero_laplacian[current] {
                laplacian_matrix_add(
                    &mut rows,
                    current,
                    next,
                    face_weights[face_start + corner][2] * row_weights[current],
                );
                laplacian_matrix_add(
                    &mut rows,
                    current,
                    previous,
                    face_weights[face_start + corner][1] * row_weights[current],
                );
            }
            if edge_counts[next] == face_counts[next] && !zero_laplacian[next] {
                laplacian_matrix_add(
                    &mut rows,
                    next,
                    current,
                    face_weights[face_start + corner][2] * row_weights[next],
                );
                laplacian_matrix_add(
                    &mut rows,
                    next,
                    previous,
                    face_weights[face_start + corner][0] * row_weights[next],
                );
            }
            if edge_counts[previous] == face_counts[previous] && !zero_laplacian[previous] {
                laplacian_matrix_add(
                    &mut rows,
                    previous,
                    next,
                    face_weights[face_start + corner][0] * row_weights[previous],
                );
                laplacian_matrix_add(
                    &mut rows,
                    previous,
                    current,
                    face_weights[face_start + corner][1] * row_weights[previous],
                );
            }
        }
    }
    for (edge_index, edge) in mesh.edges.iter().enumerate() {
        let first = index_by_id[&edge.vertices[0]];
        let second = index_by_id[&edge.vertices[1]];
        if edge_counts[first] != face_counts[first]
            && edge_counts[second] != face_counts[second]
            && !zero_laplacian[first]
            && !zero_laplacian[second]
        {
            laplacian_matrix_add(
                &mut rows,
                first,
                second,
                edge_weights[edge_index] * row_weights[first],
            );
            laplacian_matrix_add(
                &mut rows,
                second,
                first,
                edge_weights[edge_index] * row_weights[second],
            );
        }
    }

    let centroid = positions.iter().copied().sum::<DVec3>() / vertex_count as f64;
    let mut result_positions = positions;
    let mut solver = SparseLaplacianSolver::new(rows, vertex_count);
    let mut right_hand_side = vec![0.0; vertex_count];
    for _ in 0..settings.iterations {
        let initial_volume = if settings.preserve_volume {
            laplacian_mesh_volume(centroid, &result_positions, &face_indices)
        } else {
            0.0
        };
        for axis in 0..3 {
            for (index, position) in result_positions.iter().enumerate() {
                right_hand_side[index] = position[axis];
            }
            if !solver.solve(&right_hand_side) {
                continue;
            }
            for (index, position) in result_positions.iter_mut().enumerate() {
                if zero_laplacian[index] {
                    continue;
                }
                let lambda = if edge_counts[index] == face_counts[index] {
                    settings.lambda
                } else {
                    settings.lambda_border
                };
                let direction = if lambda >= 0.0 { 1.0 } else { -1.0 };
                if settings.axes[axis] {
                    position[axis] += direction * (solver.solution[index] - position[axis]);
                }
            }
        }
        if settings.preserve_volume {
            let final_volume = laplacian_mesh_volume(centroid, &result_positions, &face_indices);
            if final_volume != 0.0 {
                let scale = (initial_volume / final_volume).cbrt();
                for position in &mut result_positions {
                    for axis in 0..3 {
                        if settings.axes[axis] {
                            position[axis] =
                                (position[axis] - centroid[axis]) * scale + centroid[axis];
                        }
                    }
                }
            }
        }
    }
    let mut result = mesh.clone();
    for (vertex, position) in result.vertices.iter_mut().zip(result_positions) {
        vertex.co = position;
    }
    validated(&result)?;
    Ok(result)
}

fn cotangent_weight(first: DVec3, second: DVec3, third: DVec3) -> f64 {
    let first_edge = second - first;
    let second_edge = third - first;
    let area = first_edge.cross(second_edge).length();
    if area > f64::from(f32::EPSILON) {
        first_edge.dot(second_edge) / area
    } else {
        0.0
    }
}

fn laplacian_matrix_add(rows: &mut [Vec<(usize, f64)>], row: usize, column: usize, value: f64) {
    if let Some((_, current)) = rows[row].iter_mut().find(|(index, _)| *index == column) {
        *current += value;
    } else {
        rows[row].push((column, value));
    }
}

fn laplacian_mesh_volume(center: DVec3, positions: &[DVec3], faces: &[Vec<usize>]) -> f64 {
    let mut volume = 0.0;
    for face in faces {
        if face.len() < 3 {
            continue;
        }
        let first = positions[face[0]] - center;
        for index in 1..face.len() - 1 {
            let previous = positions[face[index]] - center;
            let current = positions[face[index + 1]] - center;
            volume += first.dot(previous.cross(current)) / 6.0;
        }
    }
    volume.abs()
}

struct SparseLaplacianSolver {
    rows: Vec<Vec<(usize, f64)>>,
    solution: Vec<f64>,
    residual: Vec<f64>,
    shadow_residual: Vec<f64>,
    direction: Vec<f64>,
    product: Vec<f64>,
    intermediate: Vec<f64>,
    second_product: Vec<f64>,
}

impl SparseLaplacianSolver {
    fn new(rows: Vec<Vec<(usize, f64)>>, vertex_count: usize) -> Self {
        Self {
            rows,
            solution: vec![0.0; vertex_count],
            residual: vec![0.0; vertex_count],
            shadow_residual: vec![0.0; vertex_count],
            direction: vec![0.0; vertex_count],
            product: vec![0.0; vertex_count],
            intermediate: vec![0.0; vertex_count],
            second_product: vec![0.0; vertex_count],
        }
    }

    fn solve(&mut self, right_hand_side: &[f64]) -> bool {
        self.solution.copy_from_slice(right_hand_side);
        sparse_matrix_vector(&self.rows, &self.solution, &mut self.product);
        for (index, &right_hand_side) in right_hand_side.iter().enumerate() {
            self.residual[index] = right_hand_side - self.product[index];
            self.shadow_residual[index] = self.residual[index];
        }
        let tolerance = 1.0e-11 * sparse_norm(right_hand_side).max(1.0);
        if sparse_norm(&self.residual) <= tolerance {
            return true;
        }
        let mut previous_rho = 1.0;
        let mut alpha = 1.0;
        let mut omega = 1.0;
        let max_iterations = self.solution.len().saturating_mul(4).max(32);
        for iteration in 0..max_iterations {
            let rho = sparse_dot(&self.shadow_residual, &self.residual);
            if rho.abs() <= f64::EPSILON {
                return false;
            }
            if iteration == 0 {
                self.direction.copy_from_slice(&self.residual);
            } else {
                let beta = (rho / previous_rho) * (alpha / omega);
                for index in 0..self.direction.len() {
                    self.direction[index] = self.residual[index]
                        + beta * (self.direction[index] - omega * self.product[index]);
                }
            }
            sparse_matrix_vector(&self.rows, &self.direction, &mut self.product);
            let divisor = sparse_dot(&self.shadow_residual, &self.product);
            if divisor.abs() <= f64::EPSILON {
                return false;
            }
            alpha = rho / divisor;
            for index in 0..self.intermediate.len() {
                self.intermediate[index] = self.residual[index] - alpha * self.product[index];
            }
            if sparse_norm(&self.intermediate) <= tolerance {
                for index in 0..self.solution.len() {
                    self.solution[index] += alpha * self.direction[index];
                }
                return true;
            }
            sparse_matrix_vector(&self.rows, &self.intermediate, &mut self.second_product);
            let denominator = sparse_dot(&self.second_product, &self.second_product);
            if denominator <= f64::EPSILON {
                return false;
            }
            omega = sparse_dot(&self.second_product, &self.intermediate) / denominator;
            if omega.abs() <= f64::EPSILON {
                return false;
            }
            for index in 0..self.solution.len() {
                self.solution[index] +=
                    alpha * self.direction[index] + omega * self.intermediate[index];
                self.residual[index] =
                    self.intermediate[index] - omega * self.second_product[index];
            }
            if sparse_norm(&self.residual) <= tolerance {
                return true;
            }
            previous_rho = rho;
        }
        false
    }
}

fn sparse_matrix_vector(rows: &[Vec<(usize, f64)>], input: &[f64], output: &mut [f64]) {
    for (row_index, row) in rows.iter().enumerate() {
        output[row_index] = row
            .iter()
            .map(|(column, coefficient)| coefficient * input[*column])
            .sum();
    }
}

fn sparse_dot(first: &[f64], second: &[f64]) -> f64 {
    first
        .iter()
        .zip(second)
        .map(|(first, second)| first * second)
        .sum()
}

fn sparse_norm(vector: &[f64]) -> f64 {
    sparse_dot(vector, vector).sqrt()
}

pub(super) fn corrective_smooth(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    let iterations = usize::try_from(uint_param(modifier, "iterations", 5)?)
        .map_err(|_| invalid_parameter(modifier, "iterations", "an integer from 0 to 10000"))?;
    if iterations > 10000 {
        return Err(invalid_parameter(
            modifier,
            "iterations",
            "an integer from 0 to 10000",
        ));
    }
    let factor = number_param(modifier, "factor", 0.5)?;
    let mut smoothed = smooth_positions(mesh, iterations, factor)?;
    let original: HashMap<_, _> = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.co))
        .collect();
    let scale = number_param(modifier, "scale", 1.0)?.clamp(0.0, 1.0);
    for vertex in &mut smoothed.vertices {
        if let Some(point) = original.get(&vertex.id) {
            vertex.co = vertex.co.lerp(*point, scale);
        }
    }
    Ok(smoothed)
}

pub(super) fn build(mesh: &Mesh, modifier: &Modifier, frame: f64) -> Result<Mesh> {
    let start = number_param(modifier, "frame_start", 1.0)?;
    let duration = number_param(modifier, "frame_duration", 100.0)?.max(1.0);
    let factor = ((frame - start) / duration).clamp(0.0, 1.0);
    let reverse = bool_param(modifier, "use_reverse", false)?;
    let threshold = if reverse { 1.0 - factor } else { factor };
    let use_random_order = bool_param(modifier, "use_random_order", false)?;
    let seed = uint_param(modifier, "seed", 0)?;
    let mut face_order: Vec<usize> = (0..mesh.faces.len()).collect();
    if use_random_order {
        randomize_build_order(&mut face_order, seed);
    }
    let face_count = (mesh.faces.len() as f64 * threshold) as usize;
    let edge_count = (mesh.edges.len() as f64 * threshold) as usize;
    let vertex_count = (mesh.vertices.len() as f64 * threshold) as usize;
    let source_positions = positions(mesh);
    let source_indices = id_indices(mesh);
    let mut points = Vec::new();
    let mut result_faces = Vec::new();
    let mut result_edges = Vec::new();
    let mut vertex_map = HashMap::new();
    if face_count > 0 {
        for &face_index in face_order.iter().take(face_count) {
            let face = &mesh.faces[face_index];
            let mut vertices = Vec::with_capacity(face.vertices.len());
            for id in &face.vertices {
                let source_index = *source_indices.get(id).ok_or_else(|| {
                    invalid_parameter(modifier, "mesh", "faces referencing existing vertices")
                })?;
                let result_index = if let Some(index) = vertex_map.get(&source_index) {
                    *index
                } else {
                    let index = points.len();
                    points.push(source_positions[source_index]);
                    vertex_map.insert(source_index, index);
                    index
                };
                vertices.push(result_index);
            }
            result_faces.push((vertices, face.material_index));
        }
        for edge in &mesh.edges {
            let first = *source_indices.get(&edge.vertices[0]).ok_or_else(|| {
                invalid_parameter(modifier, "mesh", "edges referencing existing vertices")
            })?;
            let second = *source_indices.get(&edge.vertices[1]).ok_or_else(|| {
                invalid_parameter(modifier, "mesh", "edges referencing existing vertices")
            })?;
            if let (Some(&first), Some(&second)) = (vertex_map.get(&first), vertex_map.get(&second))
            {
                result_edges.push([first, second]);
            }
        }
    } else if edge_count > 0 {
        let mut edge_order: Vec<usize> = (0..mesh.edges.len()).collect();
        if use_random_order {
            randomize_build_order(&mut edge_order, seed);
        }
        for edge_index in edge_order.into_iter().take(edge_count) {
            let edge = &mesh.edges[edge_index];
            for id in edge.vertices {
                let source_index = *source_indices.get(&id).ok_or_else(|| {
                    invalid_parameter(modifier, "mesh", "edges referencing existing vertices")
                })?;
                if let std::collections::hash_map::Entry::Vacant(entry) =
                    vertex_map.entry(source_index)
                {
                    let index = points.len();
                    points.push(source_positions[source_index]);
                    entry.insert(index);
                }
            }
            result_edges.push([
                vertex_map[&source_indices[&edge.vertices[0]]],
                vertex_map[&source_indices[&edge.vertices[1]]],
            ]);
        }
    } else {
        let mut vertex_order: Vec<usize> = (0..mesh.vertices.len()).collect();
        if use_random_order {
            randomize_build_order(&mut vertex_order, seed);
        }
        points.extend(
            vertex_order
                .into_iter()
                .take(vertex_count)
                .map(|index| source_positions[index]),
        );
    }
    let result = build_mesh(points, result_faces, result_edges, &mesh.attributes)?;
    if !result.vertices.is_empty() {
        validated(&result)?;
    }
    Ok(result)
}

fn randomize_build_order(indices: &mut [usize], seed: u64) {
    let mut state = (seed << 16) | 0x330e;
    for index in (0..indices.len()).rev() {
        state = state.wrapping_mul(0x0005_deec_e66d).wrapping_add(0xb) & 0x0000_ffff_ffff_ffff;
        let random = (state >> 17) as usize;
        let other = random % indices.len();
        indices.swap(index, other);
    }
}

pub(super) fn edge_split(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    let angle = number_param(modifier, "split_angle", std::f64::consts::FRAC_PI_4)?;
    let split_sharp = bool_param(modifier, "use_edge_sharp", true)?;
    let face_polygons = polygons(mesh)?;
    let source_positions = positions(mesh);
    let face_normals: Vec<DVec3> = face_polygons
        .iter()
        .map(|(vertices, _)| {
            if vertices.len() < 3 {
                DVec3::ZERO
            } else {
                (source_positions[vertices[1]] - source_positions[vertices[0]])
                    .cross(source_positions[vertices[2]] - source_positions[vertices[0]])
                    .normalize_or_zero()
            }
        })
        .collect();
    let mut edge_faces: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (face_index, face) in mesh.faces.iter().enumerate() {
        for index in 0..face.vertices.len() {
            let first = face.vertices[index];
            let second = face.vertices[(index + 1) % face.vertices.len()];
            edge_faces
                .entry((first.min(second), first.max(second)))
                .or_default()
                .push(face_index);
        }
    }
    let sharp: HashSet<(u32, u32)> = if split_sharp {
        mesh.attributes
            .get("sharp_edges")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|value| {
                let pair = value.as_array()?;
                let first = u32::try_from(pair.first()?.as_u64()?).ok()?;
                let second = u32::try_from(pair.get(1)?.as_u64()?).ok()?;
                Some((first.min(second), first.max(second)))
            })
            .collect()
    } else {
        HashSet::new()
    };
    let mut points = positions(mesh);
    let mut out_faces = Vec::new();
    let mut any_split = false;
    for (face_index, (indices, material)) in face_polygons.into_iter().enumerate() {
        let face = &mesh.faces[face_index];
        let should_split = (0..face.vertices.len()).any(|index| {
            let first = face.vertices[index];
            let second = face.vertices[(index + 1) % face.vertices.len()];
            let adjacent_faces = edge_faces.get(&(first.min(second), first.max(second)));
            let angle_split = adjacent_faces.is_some_and(|adjacent| {
                adjacent.iter().any(|other| {
                    *other != face_index
                        && face_normals[face_index]
                            .dot(face_normals[*other])
                            .clamp(-1.0, 1.0)
                            .acos()
                            > angle
                })
            });
            angle_split || sharp.contains(&(first.min(second), first.max(second)))
        });
        any_split |= should_split;
        let mut output = Vec::with_capacity(indices.len());
        for index in indices {
            if should_split {
                output.push(points.len());
                points.push(mesh.vertices[index].co);
            } else {
                output.push(index);
            }
        }
        out_faces.push((output, material));
    }
    if !any_split {
        return Ok(mesh.clone());
    }
    let loose_edges = loose_edges(mesh)?;
    let source_indices = id_indices(mesh);
    let mut source_connected = vec![false; mesh.vertices.len()];
    for face in &mesh.faces {
        for id in &face.vertices {
            source_connected[source_indices[id]] = true;
        }
    }
    for edge in &mesh.edges {
        source_connected[source_indices[&edge.vertices[0]]] = true;
        source_connected[source_indices[&edge.vertices[1]]] = true;
    }
    let mut used = HashSet::new();
    for (face, _) in &out_faces {
        used.extend(face.iter().copied());
    }
    for [first, second] in &loose_edges {
        used.insert(*first);
        used.insert(*second);
    }
    for (index, connected) in source_connected.iter().enumerate() {
        if !connected {
            used.insert(index);
        }
    }
    let mut remap = vec![usize::MAX; points.len()];
    let mut compact_points = Vec::with_capacity(used.len());
    for (index, point) in points.into_iter().enumerate() {
        if used.contains(&index) {
            remap[index] = compact_points.len();
            compact_points.push(point);
        }
    }
    let compact_faces = out_faces
        .into_iter()
        .map(|(face, material)| {
            (
                face.into_iter().map(|index| remap[index]).collect(),
                material,
            )
        })
        .collect();
    let compact_loose_edges = loose_edges
        .into_iter()
        .map(|[first, second]| [remap[first], remap[second]])
        .collect();
    build_mesh(
        compact_points,
        compact_faces,
        compact_loose_edges,
        &mesh.attributes,
    )
}

fn smooth_positions(mesh: &Mesh, iterations: usize, factor: f64) -> Result<Mesh> {
    let index_by_id = id_indices(mesh);
    let mut neighbors = vec![HashSet::new(); mesh.vertices.len()];
    for edge in &mesh.edges {
        let first = *index_by_id.get(&edge.vertices[0]).ok_or_else(|| {
            PotError::invalid_argument("smooth modifier edge references missing vertex")
        })?;
        let second = *index_by_id.get(&edge.vertices[1]).ok_or_else(|| {
            PotError::invalid_argument("smooth modifier edge references missing vertex")
        })?;
        neighbors[first].insert(second);
        neighbors[second].insert(first);
    }
    let mut result = mesh.clone();
    for _ in 0..iterations {
        let current: Vec<DVec3> = result.vertices.iter().map(|vertex| vertex.co).collect();
        for (index, vertex) in result.vertices.iter_mut().enumerate() {
            if neighbors[index].is_empty() {
                continue;
            }
            let average = neighbors[index]
                .iter()
                .map(|neighbor| current[*neighbor])
                .sum::<DVec3>()
                / neighbors[index].len() as f64;
            vertex.co = current[index].lerp(average, factor);
        }
    }
    validated(&result)?;
    Ok(result)
}

fn target_triangles(mesh: &Mesh) -> Result<Vec<[DVec3; 3]>> {
    let positions: HashMap<_, _> = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.co))
        .collect();
    mesh.triangulate()
        .map_err(|error| {
            PotError::invalid_argument(format!("invalid target triangulation: {error}"))
        })?
        .into_iter()
        .map(|triangle| {
            Ok([
                *positions.get(&triangle[0]).ok_or_else(|| {
                    PotError::invalid_argument("target face references missing vertex")
                })?,
                *positions.get(&triangle[1]).ok_or_else(|| {
                    PotError::invalid_argument("target face references missing vertex")
                })?,
                *positions.get(&triangle[2]).ok_or_else(|| {
                    PotError::invalid_argument("target face references missing vertex")
                })?,
            ])
        })
        .collect()
}

fn nearest_surface(triangles: &[[DVec3; 3]], point: DVec3) -> Option<(DVec3, DVec3)> {
    triangles
        .iter()
        .map(|triangle| {
            let projected = closest_point_triangle(point, triangle[0], triangle[1], triangle[2]);
            let normal = (triangle[1] - triangle[0])
                .cross(triangle[2] - triangle[0])
                .normalize_or_zero();
            (projected, normal, projected.distance_squared(point))
        })
        .min_by(|left, right| left.2.total_cmp(&right.2))
        .map(|(point, normal, _)| (point, normal))
}

fn project_surface(
    triangles: &[[DVec3; 3]],
    origin: DVec3,
    direction: DVec3,
    modifier: &Modifier,
) -> Result<(DVec3, DVec3)> {
    let axis = direction
        .try_normalize()
        .ok_or_else(|| invalid_parameter(modifier, "project_axis", "a non-zero 3D vector"))?;
    let hit = triangles
        .iter()
        .filter_map(|triangle| {
            let edge1 = triangle[1] - triangle[0];
            let edge2 = triangle[2] - triangle[0];
            let p = axis.cross(edge2);
            let determinant = edge1.dot(p);
            if determinant.abs() <= 1.0e-12 {
                return None;
            }
            let inverse = 1.0 / determinant;
            let t = origin - triangle[0];
            let u = t.dot(p) * inverse;
            if !(0.0..=1.0).contains(&u) {
                return None;
            }
            let q = t.cross(edge1);
            let v = axis.dot(q) * inverse;
            if v < 0.0 || u + v > 1.0 {
                return None;
            }
            let distance = edge2.dot(q) * inverse;
            if distance < 0.0 {
                return None;
            }
            let point = origin + axis * distance;
            let normal = edge1.cross(edge2).normalize_or_zero();
            Some((point, normal, distance))
        })
        .min_by(|left, right| left.2.total_cmp(&right.2));
    hit.map(|(point, normal, _)| (point, normal))
        .ok_or_else(|| {
            invalid_parameter(
                modifier,
                "project_axis",
                "a direction intersecting the target surface",
            )
        })
}

pub(super) fn mask(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    let group_name = string_param(modifier, "vertex_group", "")?;
    if group_name.is_empty() {
        return Err(invalid_parameter(
            modifier,
            "vertex_group",
            "a vertex-group name",
        ));
    }
    let weights = mesh
        .attributes
        .get("vertex_groups")
        .and_then(Value::as_object)
        .and_then(|groups| groups.get(group_name))
        .and_then(Value::as_object)
        .ok_or_else(|| {
            invalid_parameter(
                modifier,
                "vertex_group",
                "a group present in mesh vertex_groups attributes",
            )
        })?;
    let threshold = number_param(modifier, "threshold", 0.0)?;
    let invert = bool_param(modifier, "invert_vertex_group", false)?;
    let mut selected = HashSet::new();
    for vertex in &mesh.vertices {
        let weight = weights
            .get(&format!("v{}", vertex.id))
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        if !weight.is_finite() || !(0.0..=1.0).contains(&weight) {
            return Err(invalid_parameter(
                modifier,
                "vertex_group",
                "finite weights in [0, 1]",
            ));
        }
        if if invert {
            1.0 - weight > threshold
        } else {
            weight > threshold
        } {
            selected.insert(vertex.id);
        }
    }
    let source_positions = positions(mesh);
    let source_indices = id_indices(mesh);
    let mut keep = HashSet::new();
    let mut retained_faces = Vec::new();
    for face in &mesh.faces {
        if face.vertices.iter().all(|id| selected.contains(id)) {
            let indices = face
                .vertices
                .iter()
                .map(|id| {
                    let index = source_indices.get(id).copied().ok_or_else(|| {
                        invalid_parameter(modifier, "mesh", "faces referencing existing vertices")
                    })?;
                    keep.insert(index);
                    Ok(index)
                })
                .collect::<Result<Vec<_>>>()?;
            retained_faces.push((indices, face.material_index));
        }
    }
    let source_loose_edges = loose_edges(mesh)?;
    for [first, second] in &source_loose_edges {
        if selected.contains(&mesh.vertices[*first].id)
            && selected.contains(&mesh.vertices[*second].id)
        {
            keep.insert(*first);
            keep.insert(*second);
        }
    }
    let mut remap = vec![usize::MAX; source_positions.len()];
    let mut points = Vec::with_capacity(keep.len());
    for index in 0..source_positions.len() {
        if keep.contains(&index) {
            remap[index] = points.len();
            points.push(source_positions[index]);
        }
    }
    let faces = retained_faces
        .into_iter()
        .map(|(face, material)| {
            (
                face.into_iter().map(|index| remap[index]).collect(),
                material,
            )
        })
        .collect();
    let edges = source_loose_edges
        .into_iter()
        .filter_map(|[first, second]| {
            (remap[first] != usize::MAX && remap[second] != usize::MAX)
                .then_some([remap[first], remap[second]])
        })
        .collect();
    build_mesh(points, faces, edges, &mesh.attributes)
}

#[expect(
    clippy::manual_midpoint,
    reason = "matches Blender's float evaluation order for parity"
)]
pub(super) fn wireframe(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    let thickness = number_param(modifier, "thickness", 0.02)?;
    let offset = number_param(modifier, "offset", 0.0)?;
    if thickness < 0.0 {
        return Err(invalid_parameter(
            modifier,
            "thickness",
            "a non-negative number",
        ));
    }
    let use_boundary = bool_param(modifier, "use_boundary", false)?;
    let use_replace = bool_param(modifier, "use_replace", true)?;
    let use_even_offset = bool_param(modifier, "use_even_offset", false)?;
    let use_relative_offset = bool_param(modifier, "use_relative_offset", false)?;
    let use_crease = bool_param(modifier, "use_crease", false)?;
    let crease_weight = number_param(modifier, "crease_weight", 0.0)?;
    let material_offset = modifier
        .params
        .get("material_offset")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let source_positions = positions(mesh);
    let vertex_indices = id_indices(mesh);
    let mut face_normals = Vec::with_capacity(mesh.faces.len());
    let mut normals = vec![DVec3::ZERO; mesh.vertices.len()];
    let mut face_indices = Vec::with_capacity(mesh.faces.len());
    for face in &mesh.faces {
        let indices = face
            .vertices
            .iter()
            .map(|id| {
                vertex_indices.get(id).copied().ok_or_else(|| {
                    invalid_parameter(modifier, "mesh", "faces referencing existing vertices")
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut normal = DVec3::ZERO;
        for corner in 0..indices.len() {
            let current = source_positions[indices[corner]];
            let next = source_positions[indices[(corner + 1) % indices.len()]];
            normal += DVec3::new(
                (current.y - next.y) * (current.z + next.z),
                (current.z - next.z) * (current.x + next.x),
                (current.x - next.x) * (current.y + next.y),
            );
        }
        normal = normal.normalize_or_zero();
        for index in &indices {
            normals[*index] += normal;
        }
        face_normals.push(normal);
        face_indices.push(indices);
    }
    for normal in &mut normals {
        *normal = normal.normalize_or_zero();
    }

    let mut relative_factors = vec![1.0; mesh.vertices.len()];
    if use_relative_offset {
        let mut edge_lengths = vec![Vec::new(); mesh.vertices.len()];
        for edge in &mesh.edges {
            let first = vertex_indices[&edge.vertices[0]];
            let second = vertex_indices[&edge.vertices[1]];
            let length = source_positions[first].distance(source_positions[second]);
            edge_lengths[first].push(length);
            edge_lengths[second].push(length);
        }
        for (index, lengths) in edge_lengths.iter_mut().enumerate() {
            if lengths.is_empty() {
                continue;
            }
            lengths.sort_by(f64::total_cmp);
            let middle = lengths.len() / 2;
            relative_factors[index] = if lengths.len() % 2 == 0 {
                (lengths[middle - 1] + lengths[middle]) * 0.5
            } else {
                lengths[middle]
            };
        }
    }

    let offset_original = -((1.0 - offset) * 0.5 * thickness);
    let offset_new = thickness + offset_original;
    let offset_midpoint = (offset_original + offset_new) * 0.5;
    let mut points = Vec::with_capacity(
        mesh.vertices.len() * 2 + mesh.faces.iter().map(|f| f.vertices.len()).sum::<usize>(),
    );
    if !use_replace {
        points.extend(source_positions.iter().copied());
    }
    let mut negative_vertices = vec![usize::MAX; mesh.vertices.len()];
    let mut positive_vertices = vec![usize::MAX; mesh.vertices.len()];
    let mut tagged_vertices = vec![false; mesh.vertices.len()];
    for indices in &face_indices {
        for index in indices {
            tagged_vertices[*index] = true;
        }
    }
    for index in 0..mesh.vertices.len() {
        if !tagged_vertices[index] {
            continue;
        }
        let factor = relative_factors[index];
        let center = source_positions[index] + normals[index] * (offset_midpoint * factor);
        negative_vertices[index] = points.len();
        points.push(center + normals[index] * ((offset_original - offset_midpoint) * factor));
        positive_vertices[index] = points.len();
        points.push(center + normals[index] * ((offset_new - offset_midpoint) * factor));
    }

    let mut loop_vertices = Vec::with_capacity(mesh.faces.len());
    let mut boundary_edges = HashMap::new();
    for face in &face_indices {
        for corner in 0..face.len() {
            let first = face[corner];
            let second = face[(corner + 1) % face.len()];
            *boundary_edges
                .entry((first.min(second), first.max(second)))
                .or_insert(0_usize) += 1;
        }
    }
    let mut boundary_connections = vec![Vec::new(); mesh.vertices.len()];
    if use_boundary {
        for (face_index, indices) in face_indices.iter().enumerate() {
            for corner in 0..indices.len() {
                let first = indices[corner];
                let second = indices[(corner + 1) % indices.len()];
                if boundary_edges.get(&(first.min(second), first.max(second))) == Some(&1) {
                    let tangent = (source_positions[first] - source_positions[second])
                        .cross(face_normals[face_index])
                        .normalize_or_zero();
                    boundary_connections[first].push((second, face_normals[face_index], tangent));
                    boundary_connections[second].push((first, face_normals[face_index], tangent));
                }
            }
        }
    }
    let mut boundary_normals = vec![DVec3::ZERO; mesh.vertices.len()];
    let mut boundary_face_normals = vec![DVec3::ZERO; mesh.vertices.len()];
    let mut boundary_neighbors = vec![None; mesh.vertices.len()];
    for (vertex, connections) in boundary_connections.iter().enumerate() {
        let Some(&(first_other, first_normal, first_tangent)) = connections.first() else {
            continue;
        };
        if let Some(&(second_other, second_normal, second_tangent)) = connections.get(1) {
            let face_normal = (first_normal + second_normal).normalize_or_zero();
            let edge_direction = (source_positions[vertex] - source_positions[first_other])
                .normalize_or_zero()
                + (source_positions[second_other] - source_positions[vertex]).normalize_or_zero();
            let mut normal = edge_direction.cross(face_normal).normalize_or_zero();
            if normal.dot(first_tangent + second_tangent) > 0.0 {
                normal = -normal;
            }
            boundary_normals[vertex] = normal;
            boundary_face_normals[vertex] = face_normal;
            boundary_neighbors[vertex] = Some([first_other, second_other]);
        } else {
            let mut normal = (source_positions[vertex] - source_positions[first_other])
                .cross(first_normal)
                .normalize_or_zero();
            if normal.dot(first_tangent) > 0.0 {
                normal = -normal;
            }
            boundary_normals[vertex] = normal;
            boundary_face_normals[vertex] = first_normal;
        }
    }
    let mut boundary_vertices = vec![usize::MAX; mesh.vertices.len()];
    for (face_index, indices) in face_indices.iter().enumerate() {
        let mut face_loops = Vec::with_capacity(indices.len());
        for corner in 0..indices.len() {
            let previous = indices[(corner + indices.len() - 1) % indices.len()];
            let current = indices[corner];
            let next = indices[(corner + 1) % indices.len()];
            let previous_direction =
                (source_positions[previous] - source_positions[current]).normalize_or_zero();
            let next_direction =
                (source_positions[current] - source_positions[next]).normalize_or_zero();
            let direction = previous_direction + next_direction;
            let mut corner_normal = previous_direction.cross(next_direction);
            if corner_normal.dot(face_normals[face_index]) < 0.0 {
                corner_normal = -corner_normal;
            }
            let tangent =
                if previous_direction.abs_diff_eq(next_direction, f64::from(f32::EPSILON) * 10.0) {
                    direction
                        .cross(face_normals[face_index])
                        .normalize_or_zero()
                } else {
                    direction.cross(corner_normal).normalize_or_zero()
                };
            let corner_angle = previous_direction
                .dot(-next_direction)
                .clamp(-1.0, 1.0)
                .acos();
            let even_factor = if use_even_offset {
                let angle = (std::f64::consts::PI - corner_angle) * 0.5;
                if angle < 1.0e-8 {
                    1.0
                } else {
                    angle.cos().abs().recip()
                }
            } else {
                1.0
            };
            let factor = relative_factors[current];
            let mut point =
                source_positions[current] + tangent * (thickness * 0.5 * factor * even_factor);
            if offset != 0.0 {
                point += normals[current] * (offset_midpoint * factor);
            }
            face_loops.push(points.len());
            points.push(point);
            if use_boundary
                && boundary_edges.get(&(current.min(next), current.max(next))) == Some(&1)
            {
                for boundary_vertex in [current, next] {
                    if boundary_vertices[boundary_vertex] != usize::MAX {
                        continue;
                    }
                    let factor = relative_factors[boundary_vertex];
                    let shell_factor = if use_even_offset
                        && let Some([first_other, second_other]) =
                            boundary_neighbors[boundary_vertex]
                    {
                        let face_normal = boundary_face_normals[boundary_vertex];
                        let first_direction =
                            source_positions[first_other] - source_positions[boundary_vertex];
                        let second_direction =
                            source_positions[second_other] - source_positions[boundary_vertex];
                        let first_direction = (first_direction
                            - face_normal * first_direction.dot(face_normal))
                        .normalize_or_zero();
                        let second_direction = (second_direction
                            - face_normal * second_direction.dot(face_normal))
                        .normalize_or_zero();
                        let angle = first_direction
                            .dot(second_direction)
                            .clamp(-1.0, 1.0)
                            .acos();
                        let correction = (std::f64::consts::PI - angle) * 0.5;
                        if correction < 1.0e-8 {
                            1.0
                        } else {
                            correction.cos().abs().recip()
                        }
                    } else {
                        1.0
                    };
                    let mut point = source_positions[boundary_vertex]
                        + boundary_normals[boundary_vertex]
                            * (thickness * 0.5 * factor * shell_factor);
                    if offset != 0.0 {
                        point += normals[boundary_vertex] * (offset_midpoint * factor);
                    }
                    boundary_vertices[boundary_vertex] = points.len();
                    points.push(point);
                }
            }
        }
        loop_vertices.push(face_loops);
    }

    let material_max = mesh
        .faces
        .iter()
        .map(|face| face.material_index)
        .max()
        .unwrap_or(0);
    let mut faces = Vec::new();
    let mut crease_edges = HashSet::new();
    for (face_index, indices) in face_indices.iter().enumerate() {
        let source_face = &mesh.faces[face_index];
        let material_index = i64::from(source_face.material_index)
            .saturating_add(material_offset)
            .clamp(0, i64::from(material_max)) as u32;
        for corner in 0..indices.len() {
            let next = (corner + 1) % indices.len();
            let first = indices[corner];
            let second = indices[next];
            let loop_first = loop_vertices[face_index][corner];
            let loop_second = loop_vertices[face_index][next];
            if use_crease {
                for (loop_vertex, source_vertex) in [(loop_first, first), (loop_second, second)] {
                    let negative = negative_vertices[source_vertex];
                    let positive = positive_vertices[source_vertex];
                    crease_edges.insert((loop_vertex.min(negative), loop_vertex.max(negative)));
                    crease_edges.insert((loop_vertex.min(positive), loop_vertex.max(positive)));
                }
            }
            faces.push((
                vec![
                    loop_first,
                    loop_second,
                    negative_vertices[second],
                    negative_vertices[first],
                ],
                material_index,
            ));
            faces.push((
                vec![
                    loop_second,
                    loop_first,
                    positive_vertices[first],
                    positive_vertices[second],
                ],
                material_index,
            ));
            let is_boundary = boundary_edges
                .get(&(first.min(second), first.max(second)))
                .is_some_and(|count| *count == 1);
            if use_boundary && is_boundary {
                let boundary_first = boundary_vertices[first];
                let boundary_second = boundary_vertices[second];
                faces.push((
                    vec![
                        boundary_second,
                        boundary_first,
                        negative_vertices[first],
                        negative_vertices[second],
                    ],
                    material_index,
                ));
                faces.push((
                    vec![
                        boundary_first,
                        boundary_second,
                        positive_vertices[second],
                        positive_vertices[first],
                    ],
                    material_index,
                ));
                if use_crease {
                    for (boundary, source_vertex) in
                        [(boundary_first, first), (boundary_second, second)]
                    {
                        for inset_vertex in [
                            negative_vertices[source_vertex],
                            positive_vertices[source_vertex],
                        ] {
                            crease_edges
                                .insert((boundary.min(inset_vertex), boundary.max(inset_vertex)));
                        }
                    }
                }
            }
        }
    }
    let mut attributes = mesh.attributes.clone();
    attributes.remove("wireframe_use_crease");
    attributes.remove("wireframe_crease_weight");
    let mut result = build_mesh(points, faces, Vec::new(), &attributes)?;
    if use_crease {
        let vertex_indices = id_indices(&result);
        let mut values = vec![0.0; result.edges.len()];
        for (edge_index, edge) in result.edges.iter().enumerate() {
            let first = vertex_indices[&edge.vertices[0]] as usize;
            let second = vertex_indices[&edge.vertices[1]] as usize;
            if crease_edges.contains(&(first.min(second), first.max(second))) {
                values[edge_index] = crease_weight;
            }
        }
        let mut blender_attributes = result
            .attributes
            .get("blender_attributes")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        blender_attributes.retain(|attribute| {
            attribute.get("name").and_then(Value::as_str) != Some("crease_edge")
        });
        blender_attributes.push(serde_json::json!({
            "name": "crease_edge",
            "domain": "EDGE",
            "data_type": "FLOAT",
            "values": values,
        }));
        result.attributes.insert(
            "blender_attributes".to_owned(),
            Value::Array(blender_attributes),
        );
    }
    Ok(result)
}

#[expect(
    clippy::manual_midpoint,
    reason = "matches Blender's float evaluation order for parity"
)]
pub(super) fn skin(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    let source_positions = positions(mesh);
    let source_indices = id_indices(mesh);
    let radii = mesh.attributes.get("skin_radii").and_then(Value::as_object);
    let roots = mesh.attributes.get("skin_roots").and_then(Value::as_object);
    if mesh.edges.is_empty() {
        return Err(invalid_parameter(
            modifier,
            "mesh",
            "a skeleton mesh with at least one edge",
        ));
    }
    let _branch_smoothing = number_param(modifier, "branch_smoothing", 0.0)?;
    let smooth_shade = bool_param(modifier, "use_smooth_shade", true)?;
    let _symmetry = [
        bool_param(modifier, "use_x_symmetry", false)?,
        bool_param(modifier, "use_y_symmetry", false)?,
        bool_param(modifier, "use_z_symmetry", false)?,
    ];
    let mut node_positions = source_positions.clone();
    let mut node_radii = Vec::with_capacity(mesh.vertices.len() + mesh.edges.len() * 4);
    let mut node_roots = Vec::with_capacity(mesh.vertices.len());
    for vertex_index in 0..mesh.vertices.len() {
        node_radii.push(skin_radius(radii, vertex_index, modifier)?);
        node_roots.push(
            roots
                .and_then(|roots| roots.get(&vertex_index.to_string()))
                .and_then(Value::as_bool)
                .unwrap_or(false),
        );
    }
    let mut degrees = vec![0_usize; mesh.vertices.len()];
    let mut edge_endpoints = Vec::with_capacity(mesh.edges.len());
    let mut edge_lengths = Vec::with_capacity(mesh.edges.len());
    for edge in &mesh.edges {
        let first = *source_indices.get(&edge.vertices[0]).ok_or_else(|| {
            invalid_parameter(modifier, "mesh", "edges referencing existing vertices")
        })?;
        let second = *source_indices.get(&edge.vertices[1]).ok_or_else(|| {
            invalid_parameter(modifier, "mesh", "edges referencing existing vertices")
        })?;
        let length = source_positions[first].distance(source_positions[second]);
        if length <= f64::EPSILON {
            return Err(invalid_parameter(
                modifier,
                "mesh",
                "skeleton edges with non-zero length",
            ));
        }
        degrees[first] += 1;
        degrees[second] += 1;
        edge_endpoints.push([first, second]);
        edge_lengths.push(length);
    }
    if degrees.iter().any(|degree| *degree > 2) {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "Skin modifier branch-hull evaluation is not supported",
            serde_json::json!({"feature_id":"modifier.skin.branch_hull","modifier_id":modifier.id}),
        ));
    }
    let mut edge_chains = Vec::with_capacity(mesh.edges.len());
    let mut edge_frames = Vec::with_capacity(mesh.edges.len());
    for (edge_index, [first, second]) in edge_endpoints.iter().copied().enumerate() {
        let first_radius = node_radii[first];
        let second_radius = node_radii[second];
        let average_radius =
            (first_radius[0] + first_radius[1] + second_radius[0] + second_radius[1]) * 0.5;
        let subdivisions = if average_radius > 0.0 {
            (edge_lengths[edge_index] / average_radius).min(128.0) as usize
        } else {
            0
        };
        let first_half = (first_radius[0] + first_radius[1]) * 0.5;
        let second_half = (second_radius[0] + second_radius[1]) * 0.5;
        let radius_ratio = if first_half > 0.0 {
            (second_half / first_half + 1.0) * 0.5
        } else {
            1.0
        };
        let mut chain = Vec::with_capacity(subdivisions + 2);
        chain.push(first);
        for subdivision in 0..subdivisions {
            let ratio = (subdivision + 1) as f64 / (subdivisions + 1) as f64;
            let fraction = ratio.powf(radius_ratio);
            let index = node_positions.len();
            node_positions.push(source_positions[first].lerp(source_positions[second], fraction));
            node_radii.push([
                first_radius[0] + (second_radius[0] - first_radius[0]) * fraction,
                first_radius[1] + (second_radius[1] - first_radius[1]) * fraction,
            ]);
            node_roots.push(false);
            chain.push(index);
        }
        chain.push(second);
        edge_chains.push(chain);
        edge_frames.push(None);
    }
    let mut adjacency = vec![Vec::new(); source_positions.len()];
    for (edge_index, [first, second]) in edge_endpoints.iter().copied().enumerate() {
        adjacency[first].push((edge_index, second));
        adjacency[second].push((edge_index, first));
    }
    let mut visited_edges = vec![false; mesh.edges.len()];
    for root in 0..source_positions.len() {
        if !node_roots[root] {
            continue;
        }
        for &(edge_index, neighbor) in &adjacency[root] {
            if visited_edges[edge_index] {
                continue;
            }
            let x_axis = (source_positions[neighbor] - source_positions[root]).normalize();
            let z_up = if x_axis.z.abs() > 1.0 - f64::EPSILON {
                DVec3::Y
            } else {
                DVec3::Z
            };
            let y_axis = z_up.cross(x_axis).normalize_or_zero();
            let z_axis = x_axis.cross(y_axis).normalize_or_zero();
            edge_frames[edge_index] = Some([x_axis, y_axis, z_axis]);
            visited_edges[edge_index] = true;
            let mut stack = vec![(neighbor, edge_index, [x_axis, y_axis, z_axis])];
            while let Some((vertex, parent_edge, parent_frame)) = stack.pop() {
                for &(next_edge, next_vertex) in &adjacency[vertex] {
                    if next_edge == parent_edge || visited_edges[next_edge] {
                        continue;
                    }
                    let next_axis =
                        (source_positions[next_vertex] - source_positions[vertex]).normalize();
                    let dot = parent_frame[0].dot(next_axis).clamp(-1.0, 1.0);
                    let rotation_axis = parent_frame[0].cross(next_axis).normalize_or_zero();
                    let rotation = DQuat::from_axis_angle(rotation_axis, dot.acos());
                    let next_frame = if node_roots[vertex] {
                        let z_up = if next_axis.z.abs() > 1.0 - f64::EPSILON {
                            DVec3::Y
                        } else {
                            DVec3::Z
                        };
                        let y_axis = z_up.cross(next_axis).normalize_or_zero();
                        [
                            next_axis,
                            y_axis,
                            next_axis.cross(y_axis).normalize_or_zero(),
                        ]
                    } else {
                        [
                            next_axis,
                            (rotation * parent_frame[1]).normalize_or_zero(),
                            (rotation * parent_frame[2]).normalize_or_zero(),
                        ]
                    };
                    edge_frames[next_edge] = Some(next_frame);
                    visited_edges[next_edge] = true;
                    stack.push((next_vertex, next_edge, next_frame));
                }
            }
        }
    }
    let mut point_frames = vec![[DVec3::ZERO; 3]; node_positions.len()];
    let mut has_frame = vec![false; node_positions.len()];
    for (edge_index, [first, second]) in edge_endpoints.iter().copied().enumerate() {
        let [x_axis, y_axis, z_axis] = edge_frames[edge_index].unwrap_or_else(|| {
            let direction = (source_positions[second] - source_positions[first]).normalize();
            let z_up = if direction.z.abs() > 1.0 - f64::EPSILON {
                DVec3::Y
            } else {
                DVec3::Z
            };
            let y_axis = z_up.cross(direction).normalize_or_zero();
            [
                direction,
                y_axis,
                direction.cross(y_axis).normalize_or_zero(),
            ]
        });
        let root_forward = node_roots[first] || !node_roots[second];
        let first_frame = [if root_forward { x_axis } else { -x_axis }, y_axis, z_axis];
        let second_frame = [if root_forward { -x_axis } else { x_axis }, y_axis, z_axis];
        if degrees[first] <= 2 {
            point_frames[first] = first_frame;
            has_frame[first] = true;
        }
        if degrees[second] <= 2 {
            point_frames[second] = second_frame;
            has_frame[second] = true;
        }
        for &index in edge_chains[edge_index]
            .iter()
            .skip(1)
            .take(edge_chains[edge_index].len().saturating_sub(2))
        {
            point_frames[index] = [x_axis, y_axis, z_axis];
            has_frame[index] = true;
        }
    }
    let mut frames = vec![None; node_positions.len()];
    for (index, center) in node_positions.iter().enumerate() {
        if !has_frame[index] {
            continue;
        }
        let [_, y_axis, z_axis] = point_frames[index];
        let [radius_y, radius_z] = node_radii[index];
        let y = y_axis * radius_y;
        let z = z_axis * radius_z;
        frames[index] = Some([
            *center + y - z,
            *center - y - z,
            *center - y + z,
            *center + y + z,
        ]);
    }
    let mut points = Vec::with_capacity(node_positions.len() * 4);
    let mut frame_vertex_ids = vec![None; frames.len()];
    for (index, frame) in frames.iter().enumerate() {
        let Some(frame) = frame else {
            continue;
        };
        let mut vertex_ids = [usize::MAX; 4];
        for corner in 0..4 {
            vertex_ids[corner] = points.len();
            points.push(frame[corner]);
        }
        frame_vertex_ids[index] = Some(vertex_ids);
    }
    let mut faces = Vec::with_capacity(mesh.edges.len() * 32 + 2);
    for chain in &edge_chains {
        for pair in chain.windows(2) {
            let (Some(first), Some(second)) =
                (frame_vertex_ids[pair[0]], frame_vertex_ids[pair[1]])
            else {
                continue;
            };
            for side in 0..4 {
                let next = (side + 1) % 4;
                faces.push((
                    vec![first[side], first[next], second[next], second[side]],
                    0,
                ));
            }
        }
    }
    let mut capped = HashSet::new();
    for [first, second] in edge_endpoints.iter().copied() {
        for vertex in [first, second] {
            let Some(base) = frame_vertex_ids[vertex] else {
                continue;
            };
            if degrees[vertex] != 1 || !capped.insert(vertex) {
                continue;
            }
            let cap_normal =
                (points[base[1]] - points[base[0]]).cross(points[base[2]] - points[base[0]]);
            let flipped = point_frames[vertex][0].dot(cap_normal) < 0.0;
            let cap = if flipped {
                vec![base[0], base[1], base[2], base[3]]
            } else {
                vec![base[3], base[2], base[1], base[0]]
            };
            faces.push((cap, 0));
        }
    }
    let mut attributes = mesh.attributes.clone();
    attributes.remove("blender_skin_vertices");
    attributes.remove("skin_radii");
    attributes.remove("skin_roots");
    attributes.insert(
        "blender_polygon_smooth".to_owned(),
        serde_json::json!(vec![smooth_shade; faces.len()]),
    );
    let result = build_mesh(points, faces, Vec::new(), &attributes)?;
    validated(&result)?;
    Ok(result)
}

fn skin_radius(
    radii: Option<&serde_json::Map<String, Value>>,
    vertex_index: usize,
    modifier: &Modifier,
) -> Result<[f64; 2]> {
    let Some(value) = radii.and_then(|radii| radii.get(&vertex_index.to_string())) else {
        return Ok([0.25, 0.25]);
    };
    let Some(values) = value.as_array() else {
        return Err(invalid_parameter(
            modifier,
            "skin_radii",
            "two positive finite radii per vertex",
        ));
    };
    if values.len() != 2 {
        return Err(invalid_parameter(
            modifier,
            "skin_radii",
            "two positive finite radii per vertex",
        ));
    }
    let first = values.first().and_then(Value::as_f64).ok_or_else(|| {
        invalid_parameter(
            modifier,
            "skin_radii",
            "two positive finite radii per vertex",
        )
    })?;
    let second = values.get(1).and_then(Value::as_f64).ok_or_else(|| {
        invalid_parameter(
            modifier,
            "skin_radii",
            "two positive finite radii per vertex",
        )
    })?;
    if !first.is_finite() || !second.is_finite() || first <= 0.0 || second <= 0.0 {
        return Err(invalid_parameter(
            modifier,
            "skin_radii",
            "two positive finite radii per vertex",
        ));
    }
    Ok([first, second])
}

pub(super) fn curve(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    let target =
        modifier.runtime.target_mesh.as_ref().ok_or_else(|| {
            invalid_parameter(modifier, "object", "a curve or mesh object operand")
        })?;
    let target: Mesh = serde_json::from_value(target.clone())
        .map_err(|_| invalid_parameter(modifier, "object", "a curve or mesh object operand"))?;
    let axis = curve_axis_param(modifier, "deform_axis")?;
    let mut cumulative = Vec::with_capacity(target.vertices.len());
    cumulative.push(0.0);
    for pair in target.vertices.windows(2) {
        let length = pair[0].co.distance(pair[1].co);
        if length <= f64::EPSILON {
            return Err(invalid_parameter(
                modifier,
                "object",
                "a curve without zero-length segments",
            ));
        }
        cumulative.push(cumulative.last().copied().unwrap_or(0.0) + length);
    }
    let total_length = cumulative.last().copied().unwrap_or(0.0);
    if target.vertices.len() < 2 || total_length <= f64::EPSILON {
        return Err(invalid_parameter(
            modifier,
            "object",
            "a curve with at least two distinct points",
        ));
    }
    let source_positions = positions(mesh);
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    for point in &source_positions {
        let along = point.dot(axis);
        minimum = minimum.min(along);
        maximum = maximum.max(along);
    }
    let extent = maximum - minimum;
    if extent <= f64::EPSILON {
        return Ok(mesh.clone());
    }
    let reference = if axis.abs().dot(DVec3::Z) > 0.9 {
        DVec3::Y
    } else {
        DVec3::Z
    };
    let source_side = axis.cross(reference).normalize_or_zero();
    let source_up = axis.cross(source_side).normalize_or_zero();
    let mut result = mesh.clone();
    for (index, vertex) in result.vertices.iter_mut().enumerate() {
        let along = (source_positions[index].dot(axis) - minimum).clamp(0.0, total_length);
        let segment = cumulative
            .windows(2)
            .position(|range| along <= range[1])
            .unwrap_or(cumulative.len() - 2);
        let first = target.vertices[segment].co;
        let second = target.vertices[segment + 1].co;
        let tangent = (second - first).normalize_or_zero();
        let reference = if tangent.abs().dot(DVec3::Z) > 0.9 {
            DVec3::Y
        } else {
            DVec3::Z
        };
        let side = tangent.cross(reference).normalize_or_zero();
        let up = tangent.cross(side).normalize_or_zero();
        let fraction =
            (along - cumulative[segment]) / (cumulative[segment + 1] - cumulative[segment]);
        let centerline = first.lerp(second, fraction);
        let perpendicular = source_positions[index] - axis * source_positions[index].dot(axis);
        vertex.co =
            centerline + side * perpendicular.dot(source_side) + up * perpendicular.dot(source_up);
    }
    validated(&result)?;
    Ok(result)
}

pub(super) fn hook(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    let target = modifier
        .runtime
        .hook_target_position
        .map_or(DVec3::ZERO, DVec3::from_array);
    let center = vec3_param(modifier, "center", DVec3::ZERO)?;
    let strength = number_param(modifier, "strength", 1.0)?;
    let radius = number_param(modifier, "falloff_radius", 0.0)?;
    let group_weights = modifier
        .params
        .get("vertex_group")
        .and_then(Value::as_str)
        .and_then(|group| {
            mesh.attributes
                .get("vertex_groups")?
                .as_object()?
                .get(group)?
                .as_object()
        });
    if modifier.params.get("vertex_group").is_some() && group_weights.is_none() {
        return Err(invalid_parameter(
            modifier,
            "vertex_group",
            "a group present in vertex_groups attributes",
        ));
    }
    let mut result = mesh.clone();
    let displacement = target - center;
    for vertex in &mut result.vertices {
        let group_weight = group_weights.map_or(1.0, |weights| {
            weights
                .get(&format!("v{}", vertex.id))
                .and_then(Value::as_f64)
                .unwrap_or(0.0)
        });
        let falloff = if radius > 0.0 {
            (1.0 - vertex.co.distance(center) / radius).clamp(0.0, 1.0)
        } else {
            1.0
        };
        vertex.co += displacement * (strength * group_weight * falloff);
    }
    validated(&result)?;
    Ok(result)
}

fn curve_axis_param(modifier: &Modifier, name: &str) -> Result<DVec3> {
    let axis = string_param(modifier, name, "POS_X")?.to_ascii_uppercase();
    match axis.as_str() {
        "POS_X" => Ok(DVec3::X),
        "NEG_X" => Ok(-DVec3::X),
        "POS_Y" => Ok(DVec3::Y),
        "NEG_Y" => Ok(-DVec3::Y),
        "POS_Z" => Ok(DVec3::Z),
        "NEG_Z" => Ok(-DVec3::Z),
        _ => Err(invalid_parameter(
            modifier,
            name,
            "POS_X, NEG_X, POS_Y, NEG_Y, POS_Z, or NEG_Z",
        )),
    }
}
#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "geometry modifier unit tests")]
mod tests {
    use super::{curve, shrinkwrap};
    use crate::geom::{Mesh, UvSphereParams};
    use serde_json::json;

    #[test]
    fn shrinkwrap_nearest_surface_projects_vertices_to_sphere_radius() {
        let source = Mesh::uv_sphere(UvSphereParams {
            segments: 8,
            ring_count: 4,
            radius: 2.0,
        })
        .unwrap();
        let target = Mesh::uv_sphere(UvSphereParams {
            segments: 24,
            ring_count: 12,
            radius: 1.0,
        })
        .unwrap();
        let modifier = modifier_with_target(
            "shrinkwrap",
            json!({"wrap_method":"NEAREST_VERTEX"}),
            target,
        );
        let result = shrinkwrap(&source, &modifier).unwrap();
        assert!(
            result
                .vertices
                .iter()
                .all(|vertex| (vertex.co.length() - 1.0).abs() < 1.0e-8)
        );
    }

    #[test]
    fn curve_modifier_maps_deform_axis_distance_to_curve_length() {
        let mut source = Mesh::new();
        let first = source.insert_vertex(glam::DVec3::ZERO).unwrap();
        let second = source.insert_vertex(glam::DVec3::X).unwrap();
        source.insert_edge([first, second]).unwrap();
        let mut curve_mesh = Mesh::new();
        let start = curve_mesh.insert_vertex(glam::DVec3::ZERO).unwrap();
        let end = curve_mesh.insert_vertex(glam::DVec3::Z * 2.0).unwrap();
        curve_mesh.insert_edge([start, end]).unwrap();
        let modifier = modifier_with_target("curve", json!({"deform_axis":"POS_X"}), curve_mesh);
        let result = curve(&source, &modifier).unwrap();
        assert!(result.vertex(first).unwrap().co.distance(glam::DVec3::ZERO) < 1.0e-12);
        assert!(result.vertex(second).unwrap().co.distance(glam::DVec3::Z) < 1.0e-12);
    }

    #[test]
    fn shrinkwrap_offset_moves_each_projected_vertex_by_the_requested_distance() {
        let source = Mesh::uv_sphere(UvSphereParams {
            segments: 8,
            ring_count: 4,
            radius: 2.0,
        })
        .unwrap();
        let target = Mesh::uv_sphere(UvSphereParams {
            segments: 16,
            ring_count: 8,
            radius: 1.0,
        })
        .unwrap();
        let evaluate_at_offset = |offset| {
            let modifier = modifier_with_target(
                "shrinkwrap",
                json!({"wrap_method":"NEAREST_VERTEX","offset":offset}),
                target.clone(),
            );
            shrinkwrap(&source, &modifier).unwrap()
        };
        let projected = evaluate_at_offset(0.0);
        let offset = evaluate_at_offset(0.25);
        for (base, shifted) in projected.vertices.iter().zip(&offset.vertices) {
            assert!((base.co.distance(shifted.co) - 0.25).abs() < 1.0e-12);
        }
    }

    fn modifier_with_target(
        modifier_type: &str,
        params: serde_json::Value,
        target: Mesh,
    ) -> crate::model::Modifier {
        let mut modifier = crate::model::Modifier {
            id: crate::model::Id::new("modifier_test".to_owned()).unwrap(),
            modifier_type: modifier_type.to_owned(),
            name: modifier_type.to_owned(),
            enabled: true,
            params: serde_json::from_value(params).unwrap(),
            binding_data: None,
            runtime: crate::model::ModifierRuntime::default(),
        };
        modifier.runtime.target_mesh = Some(serde_json::to_value(target).unwrap());
        modifier
    }
}
