// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   source/blender/modifiers/intern/MOD_build.cc, source/blender/modifiers/intern/MOD_laplaciansmooth.cc, SPDX-FileCopyrightText: 2005 Blender Authors, GPL-2.0-or-later.
//   source/blender/modifiers/intern/MOD_wireframe.cc, source/blender/modifiers/intern/MOD_skin.cc, source/blender/modifiers/intern/MOD_warp.cc, source/blender/blenkernel/intern/mesh_remap.cc, source/blender/modifiers/intern/MOD_mesh_to_volume.cc, source/blender/bmesh/tools/bmesh_wireframe.cc, SPDX-FileCopyrightText: 2023 Blender Authors, GPL-2.0-or-later.
//   source/blender/modifiers/intern/MOD_simpledeform.cc, MOD_cast.cc, MOD_hook.cc, MOD_curve.cc, source/blender/blenkernel/intern/curve_deform.cc, anim_path.cc, colortools.cc, curve.cc, SPDX-FileCopyrightText: 2005 Blender Authors, GPL-2.0-or-later.

use std::collections::{HashMap, HashSet};

use glam::{DMat3, DMat4, DQuat, DVec3, Vec2};
use serde_json::Value;

use crate::model::Modifier;

use super::{
    bool_param, build_mesh, closest_point_triangle, id_indices, invalid_parameter, loose_edges,
    number_param, polygons, positions, schema_value, string_param, uint_param, usize_as_f64,
    validated, vec3_param,
};
use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{
        Mesh,
        clnor::{self, NormalCodeOptions},
        normals::{corner_angle, corner_normals, face_normal, sharp_edge_flags},
        provenance::Origin,
        uv_layers,
    },
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum SimpleDeformMethod {
    Twist,
    Bend,
    Taper,
    Stretch,
}

/// Blender's axis remap: deformations act on the mapped Z axis (X -> Z, Y -> Z in the first two rows).
const SIMPLE_DEFORM_AXIS_MAP: [[usize; 3]; 3] = [[1, 2, 0], [2, 0, 1], [0, 1, 2]];

/// Blender `MOD_simpledeform.cc`: limits and factor are measured along the deform axis in the origin's space.
pub(super) fn simple_deform(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    let method = match string_param(modifier, "deform_method")? {
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
    let axis = simple_deform_axis(modifier)?;
    // Blender stores angle and factor in one field: angle drives TWIST and BEND, factor drives TAPER and STRETCH.
    let amount = if matches!(method, SimpleDeformMethod::Twist | SimpleDeformMethod::Bend) {
        number_param(modifier, "angle")?
    } else {
        number_param(modifier, "factor")?
    };
    let fractions = simple_deform_limits(modifier)?;
    let limit_axis = if method == SimpleDeformMethod::Bend {
        if axis == 2 { 0 } else { 2 }
    } else {
        axis
    };
    // Bend has no locks; otherwise the deform axis is never locked, since locking it would flatten the mesh.
    let lock = if method == SimpleDeformMethod::Bend {
        [false; 3]
    } else {
        let mut lock = [
            bool_param(modifier, "lock_x")?,
            bool_param(modifier, "lock_y")?,
            bool_param(modifier, "lock_z")?,
        ];
        lock[axis] = false;
        lock
    };
    let object = reference_matrix(modifier);
    let to_origin = object.map(|matrix| matrix.inverse());
    let origin_points: Vec<DVec3> = mesh
        .vertices
        .iter()
        .map(|vertex| to_origin.map_or(vertex.co, |matrix| matrix.transform_point3(vertex.co)))
        .collect();
    if origin_points.is_empty() {
        return Ok(mesh.clone());
    }
    let mut lower = f64::INFINITY;
    let mut upper = f64::NEG_INFINITY;
    for point in &origin_points {
        lower = lower.min(point[limit_axis]);
        upper = upper.max(point[limit_axis]);
    }
    let range = upper - lower;
    let limits = [lower + range * fractions[0], lower + range * fractions[1]];
    let factor = amount / (limits[1] - limits[0]).max(f64::from(f32::EPSILON));
    if method == SimpleDeformMethod::Bend && factor.abs() < 1.0e-6 {
        return Ok(mesh.clone());
    }
    let weights = deform_group_weights(mesh, modifier)?;
    let mut result = mesh.clone();
    for (index, (vertex, origin_point)) in result.vertices.iter_mut().zip(origin_points).enumerate()
    {
        let weight = group_weight(weights.as_deref(), index);
        if weight == 0.0 {
            continue;
        }
        let deformed =
            simple_deform_point(origin_point, method, axis, limit_axis, lock, factor, limits);
        let blended = origin_point.lerp(deformed, weight);
        vertex.co = object.map_or(blended, |matrix| matrix.transform_point3(blended));
    }
    validated(&result)?;
    Ok(result)
}

fn simple_deform_axis(modifier: &Modifier) -> Result<usize> {
    match string_param(modifier, "deform_axis")? {
        "X" | "x" => Ok(0),
        "Y" | "y" => Ok(1),
        "Z" | "z" => Ok(2),
        _ => Err(invalid_parameter(modifier, "deform_axis", "X, Y, or Z")),
    }
}

/// Blender's limit fractions: both clamp to [0, 1] and the lower one never exceeds the upper one.
fn simple_deform_limits(modifier: &Modifier) -> Result<[f64; 2]> {
    let invalid = || invalid_parameter(modifier, "limits", "two normalized values");
    let fractions = match schema_value(modifier, "limits")?
        .as_array()
        .map(Vec::as_slice)
    {
        Some([first, second]) => [first.as_f64(), second.as_f64()],
        _ => return Err(invalid()),
    };
    let [Some(first), Some(second)] = fractions else {
        return Err(invalid());
    };
    let upper = second.clamp(0.0, 1.0);
    Ok([first.clamp(0.0, 1.0).min(upper), upper])
}

/// Blender's per-vertex deform: locked axes are pinned to zero and restored afterwards, the limit axis is
/// clamped with the excess restored, and the deformation acts on the mapped coordinates.
fn simple_deform_point(
    position: DVec3,
    method: SimpleDeformMethod,
    axis: usize,
    limit_axis: usize,
    lock: [bool; 3],
    factor: f64,
    limits: [f64; 2],
) -> DVec3 {
    let mut coordinates = position.to_array();
    let mut cut = [0.0; 3];
    for (index, locked) in lock.into_iter().enumerate() {
        if locked {
            cut[index] = coordinates[index];
            coordinates[index] = 0.0;
        }
    }
    let original = coordinates[limit_axis];
    let clamped = original.max(limits[0]).min(limits[1]);
    cut[limit_axis] = original - clamped;
    coordinates[limit_axis] = clamped;

    let map = SIMPLE_DEFORM_AXIS_MAP[axis];
    let [x, y, z] = [
        coordinates[map[0]],
        coordinates[map[1]],
        coordinates[map[2]],
    ];
    let mut deformed = match method {
        SimpleDeformMethod::Twist => {
            let (sine, cosine) = (z * factor).sin_cos();
            [x * cosine - y * sine, x * sine + y * cosine, z]
        }
        SimpleDeformMethod::Taper => {
            let scale = z * factor;
            [x + x * scale, y + y * scale, z]
        }
        SimpleDeformMethod::Stretch => {
            let scale = z * z * factor - factor + 1.0;
            [x * scale, y * scale, z * (1.0 + factor)]
        }
        SimpleDeformMethod::Bend => {
            return DVec3::from_array(simple_deform_bend(coordinates, cut, axis, factor));
        }
    };
    for (value, offset) in deformed
        .iter_mut()
        .zip([cut[map[0]], cut[map[1]], cut[map[2]]])
    {
        *value += offset;
    }
    let mut result = [0.0; 3];
    for (original_axis, value) in map.into_iter().zip(deformed) {
        result[original_axis] = value;
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

pub(super) fn screw(
    mesh: &Mesh,
    modifier: &Modifier,
    context: crate::eval::ModifierEvaluationContext<'_>,
) -> Result<Mesh> {
    // Blender returns an empty template before reading any Screw parameters.
    if mesh.vertices.is_empty() {
        return Ok(mesh.clone());
    }
    let steps_name = if context.purpose == crate::eval::EvaluationPurpose::Render {
        "render_steps"
    } else {
        "steps"
    };
    let steps = usize::try_from(uint_param(modifier, steps_name)?)
        .map_err(|_| invalid_parameter(modifier, steps_name, "an integer from 1 to 4096"))?;
    if !(1..=4096).contains(&steps) {
        return Err(invalid_parameter(
            modifier,
            steps_name,
            "an integer from 1 to 4096",
        ));
    }
    let angle = number_param(modifier, "angle")?;
    let axis = match schema_value(modifier, "axis")? {
        Value::String(axis) => match axis.as_str() {
            "X" => DVec3::X,
            "Y" => DVec3::Y,
            "Z" => DVec3::Z,
            _ => return Err(invalid_parameter(modifier, "axis", "X, Y, or Z")),
        },
        _ => return Err(invalid_parameter(modifier, "axis", "X, Y, or Z")),
    };
    let center = DVec3::ZERO;
    let height = number_param(modifier, "screw_offset")?;
    let stretch_u = bool_param(modifier, "use_stretch_u")?;
    let stretch_v = bool_param(modifier, "use_stretch_v")?;
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
    // Blender closes a full turn only from three steps up, and only when the screw offset and the
    // turn's excess over 2π are within FLT_EPSILON * 100 (float32, MOD_screw.cc); a closed screw
    // then ignores the offset. The angle is a float32 parameter, so 2π is compared in float32 too.
    let tolerance = f32::EPSILON * 100.0;
    let closed = steps >= 3
        && (height as f32).abs() <= tolerance
        && ((angle as f32).abs() - std::f32::consts::TAU).abs() <= tolerance;
    let height = if closed { 0.0 } else { height };
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
    let flip = bool_param(modifier, "use_normal_flip")?;
    for step in 0..segments {
        let next_step = (step + 1) % ring_count;
        for [first, second] in &source_edges {
            // Blender's screw loops run against the profile edge's winding, so each face's cycle runs
            // back from the next ring to this one. `use_normal_flip` reverses every quad (quad_ord in
            // MOD_screw.cc).
            let mut quad = vec![
                next_step * source_positions.len() + second,
                next_step * source_positions.len() + first,
                step * source_positions.len() + first,
                step * source_positions.len() + second,
            ];
            if flip {
                quad.reverse();
            }
            faces.push((quad, 0));
        }
    }
    let vertex_count = source_positions.len();
    // Blender takes each screw loop's data from the last source loop of its profile vertex. A profile
    // edge that no face uses has no loops, so its screw corners carry no source data.
    let mut last_loop: Vec<Option<usize>> = vec![None; vertex_count];
    let mut loop_start = 0_usize;
    let mut face_edges: HashSet<(usize, usize)> = HashSet::new();
    for source_face in &mesh.faces {
        for (corner, id) in source_face.vertices.iter().enumerate() {
            if let Some(&index) = vertex_indices.get(id) {
                last_loop[index] = Some(loop_start + corner);
            }
        }
        loop_start += source_face.vertices.len();
        let ring: Vec<usize> = source_face
            .vertices
            .iter()
            .filter_map(|id| vertex_indices.get(id).copied())
            .collect();
        for pair in 0..ring.len() {
            let (a, b) = (ring[pair], ring[(pair + 1) % ring.len()]);
            face_edges.insert((a.min(b), a.max(b)));
        }
    }
    let edge_count = source_edges.len();
    let loose_edge: Vec<bool> = source_edges
        .iter()
        .map(|&[a, b]| !face_edges.contains(&(a.min(b), a.max(b))))
        .collect();
    let mut corners = Vec::with_capacity(faces.len() * 4);
    for (face, (vertices, _)) in faces.iter().enumerate() {
        let loose = loose_edge[face % edge_count];
        for index in vertices {
            corners.push(match last_loop[index % vertex_count] {
                Some(corner) if !loose => vec![(corner, 1.0)],
                _ => Vec::new(),
            });
        }
    }
    let origin = Origin {
        corners: Some(corners),
        ..Origin::copies(
            (0..ring_count * vertex_count).map(|index| Some(index % vertex_count)),
            vec![None; faces.len()],
            mesh.original_indices.is_some(),
        )
    };
    // Each corner's U advances by the rings between its face's step and its vertex's ring: one share
    // of the angle per ring, or 1/steps per ring under use_stretch_u. A closed seam keeps counting:
    // its wrap ring is one full share further on. Corners of a loose profile edge (no face loop) start
    // from zero U and take V from the profile vertex's height along the axis: that height unchanged
    // without use_stretch_v, and rescaled over the profile's height range with it (zero when flat).
    // Corners of face-loop edges keep their source V.
    let turn = if stretch_u {
        1.0 / usize_as_f64(steps)
    } else {
        angle / (std::f64::consts::TAU * usize_as_f64(steps))
    };
    let axial: Vec<f64> = source_positions
        .iter()
        .map(|point| point.dot(axis))
        .collect();
    let (low, high) = axial.iter().fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(low, high), &height| (low.min(height), high.max(height)),
    );
    let loose_v = |profile: usize| {
        if !stretch_v {
            axial[profile]
        } else if high > low {
            (axial[profile] - low) / (high - low)
        } else {
            0.0
        }
    };
    let corner_uv: Vec<Vec<(usize, Option<f64>)>> = faces
        .iter()
        .enumerate()
        .map(|(face, (vertices, _))| {
            let step = face / edge_count;
            let loose = loose_edge[face % edge_count];
            vertices
                .iter()
                .map(|index| {
                    let ring = index / vertex_count;
                    let turns = step + (ring + ring_count - step) % ring_count;
                    (turns, loose.then(|| loose_v(index % vertex_count)))
                })
                .collect()
        })
        .collect();
    let mut result = build_mesh(points, faces, Vec::new(), &origin, mesh)?;
    if bool_param(modifier, "use_smooth_shade")? {
        crate::geom::normals::shade_faces_smooth(&mut result);
    }
    let mut layers = uv_layers::read_layers(&result);
    for layer in &mut layers {
        for (face, values) in layer.faces.iter_mut().enumerate() {
            let corners = &corner_uv[face];
            if values.is_none() && corners.iter().all(|(_, value)| value.is_some()) {
                *values = Some(vec![[0.0, 0.0]; corners.len()]);
            }
        }
    }
    uv_layers::write_layers(&mut result, &layers);
    super::map_uv_values(&mut result, |face, corner, [u, v]| {
        let (turns, loose_value) = corner_uv[face][corner];
        [u + turn * usize_as_f64(turns), loose_value.unwrap_or(v)]
    });
    Ok(result)
}

pub(super) fn shrinkwrap(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    let target = modifier
        .runtime
        .target_mesh
        .as_ref()
        .ok_or_else(|| invalid_parameter(modifier, "target", "a mesh object operand"))?;
    validated(target)?;
    let method = string_param(modifier, "wrap_method")?.to_ascii_uppercase();
    let offset = number_param(modifier, "offset")?;
    let wrap_mode = string_param(modifier, "wrap_mode")?.to_ascii_uppercase();
    // shrinkwrap.cc BKE_shrinkwrap_init_tree: a target without vertices, or without faces for the
    // surface and projection methods, builds no tree. Blender then leaves positions unchanged and
    // reports no error; nearest-vertex lookup needs only vertices.
    if target.vertices.is_empty() || (method != "NEAREST_VERTEX" && target.faces.is_empty()) {
        return Ok(mesh.clone());
    }
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
            "PROJECT" => {
                // not a schema parameter; removed by PR18 PR-E
                let project_axis = -DVec3::Z;
                project_surface(&triangles, vertex.co, project_axis, modifier)?
            }
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

/// Subject-local to referenced-object transform the scene resolved for this modifier; `None` without a reference.
fn reference_matrix(modifier: &Modifier) -> Option<DMat4> {
    modifier
        .runtime
        .object_matrix
        .map(|matrix| DMat4::from_cols_array(&matrix))
}

/// Per-vertex weights of the modifier's vertex group, inverted when requested; `None` without a group.
/// Blender's `MOD_get_vgroup` treats a group missing from the mesh the same as no group.
fn deform_group_weights(mesh: &Mesh, modifier: &Modifier) -> Result<Option<Vec<f64>>> {
    let group = string_param(modifier, "vertex_group")?;
    if group.is_empty() {
        return Ok(None);
    }
    let invert = bool_param(modifier, "invert_vertex_group")?;
    let Some(weights) = mesh
        .attributes
        .get("vertex_groups")
        .and_then(Value::as_object)
        .and_then(|groups| groups.get(group))
        .and_then(Value::as_object)
    else {
        return Ok(None);
    };
    Ok(Some(
        mesh.vertices
            .iter()
            .map(|vertex| {
                let weight = weights
                    .get(&format!("v{}", vertex.id))
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0);
                if invert { 1.0 - weight } else { weight }
            })
            .collect(),
    ))
}

/// Per-vertex deform weight: 1 without a vertex group, else the group weight.
fn group_weight(weights: Option<&[f64]>, index: usize) -> f64 {
    weights.map_or(1.0, |weights| weights.get(index).copied().unwrap_or(0.0))
}

/// Blender's automatic cuboid half-extent per axis: the larger of the absolute extremes of the points.
fn cast_auto_half_extent(points: impl Iterator<Item = DVec3>) -> DVec3 {
    let mut lo = DVec3::splat(f64::INFINITY);
    let mut hi = DVec3::splat(f64::NEG_INFINITY);
    for point in points {
        lo = lo.min(point);
        hi = hi.max(point);
    }
    lo.abs().max(hi.abs())
}

/// Blender `MOD_cast.cc`: sphere, cylinder and cuboid projection around the reference object's centre,
/// optionally in the reference object's space (`use_transform`), per axis and weighted by the vertex group.
pub(super) fn cast(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    let cast_type = string_param(modifier, "cast_type")?.to_ascii_uppercase();
    if !matches!(cast_type.as_str(), "SPHERE" | "CYLINDER" | "CUBOID") {
        return Err(invalid_parameter(
            modifier,
            "cast_type",
            "SPHERE, CYLINDER, or CUBOID",
        ));
    }
    let factor = number_param(modifier, "factor")?;
    let radius = number_param(modifier, "radius")?;
    let size = number_param(modifier, "size")?;
    if radius < 0.0 || size < 0.0 {
        return Err(invalid_parameter(
            modifier,
            "radius",
            "a non-negative radius and size",
        ));
    }
    let use_radius_as_size = bool_param(modifier, "use_radius_as_size")?;
    let axes = [
        bool_param(modifier, "use_x")?,
        bool_param(modifier, "use_y")?,
        bool_param(modifier, "use_z")?,
    ];
    let use_transform = bool_param(modifier, "use_transform")?;
    // Blender disables the cast without a factor or an enabled axis.
    if factor == 0.0 || !axes.iter().any(|enabled| *enabled) {
        return Ok(mesh.clone());
    }
    let object = reference_matrix(modifier);
    let center = object.map_or(DVec3::ZERO, |matrix| matrix.transform_point3(DVec3::ZERO));
    // Blender works in the reference object's space with `use_transform`, else shifted to its centre.
    let (forward, backward) = match object {
        Some(matrix) if use_transform => (matrix.inverse(), matrix),
        Some(_) => (
            DMat4::from_translation(-center),
            DMat4::from_translation(center),
        ),
        None => (DMat4::IDENTITY, DMat4::IDENTITY),
    };
    let weights = deform_group_weights(mesh, modifier)?;
    let has_radius = radius > f64::from(f32::EPSILON);
    let mut result = mesh.clone();
    if cast_type == "CUBOID" {
        let half = if use_radius_as_size && has_radius {
            DVec3::splat(radius)
        } else if !use_radius_as_size && size > 0.0 {
            DVec3::splat(size)
        } else {
            cast_auto_half_extent(
                mesh.vertices
                    .iter()
                    .map(|vertex| forward.transform_point3(vertex.co)),
            )
        };
        for (index, vertex) in result.vertices.iter_mut().enumerate() {
            let weight = group_weight(weights.as_deref(), index);
            if weight == 0.0 {
                continue;
            }
            let working = forward.transform_point3(vertex.co);
            if has_radius && working.abs().max_element() > radius {
                continue;
            }
            let fac = factor * weight;
            let facm = 1.0 - fac;
            // Blender projects onto the box wall of the vertex's octant, along the ray from the centre.
            let apex = DVec3::new(
                if working.x > 0.0 { half.x } else { -half.x },
                if working.y > 0.0 { half.y } else { -half.y },
                if working.z > 0.0 { half.z } else { -half.z },
            );
            let ratio = working / apex;
            let mut best = ratio.x;
            let mut coord = 0;
            if ratio.y > best {
                best = ratio.y;
                coord = 1;
            }
            if ratio.z > best {
                coord = 2;
            }
            if working[coord].abs() < f64::from(f32::EPSILON) {
                continue;
            }
            let fbb = apex[coord] / working[coord];
            let mut projected = working;
            for (axis, enabled) in axes.iter().enumerate() {
                if *enabled {
                    projected[axis] = facm * working[axis] + fac * working[axis] * fbb;
                }
            }
            vertex.co = backward.transform_point3(projected);
        }
    } else {
        let flatten = cast_type == "CYLINDER";
        // Blender clears the Z axis for cylinders, so the projection is planar and Z keeps its value.
        let axes = if flatten {
            [axes[0], axes[1], false]
        } else {
            axes
        };
        let mut length = if use_radius_as_size { radius } else { size };
        if length <= 0.0 {
            // Blender's automatic length: the mean distance from the centre, 10 when that is zero.
            let total: f64 = mesh
                .vertices
                .iter()
                .map(|vertex| (vertex.co - center).length())
                .sum();
            let mean = total / usize_as_f64(mesh.vertices.len());
            length = if mean == 0.0 { 10.0 } else { mean };
        }
        for (index, vertex) in result.vertices.iter_mut().enumerate() {
            let weight = group_weight(weights.as_deref(), index);
            if weight == 0.0 {
                continue;
            }
            let working = forward.transform_point3(vertex.co);
            let direction = if flatten {
                DVec3::new(working.x, working.y, 0.0)
            } else {
                working
            };
            if has_radius && direction.length() > radius {
                continue;
            }
            let fac = factor * weight;
            let facm = 1.0 - fac;
            let unit = direction.normalize_or_zero();
            let mut projected = working;
            for (axis, enabled) in axes.iter().enumerate() {
                if *enabled {
                    projected[axis] = fac * unit[axis] * length + facm * working[axis];
                }
            }
            vertex.co = backward.transform_point3(projected);
        }
    }
    validated(&result)?;
    Ok(result)
}

pub(super) fn build(mesh: &Mesh, modifier: &Modifier, frame: f64) -> Result<Mesh> {
    let start = number_param(modifier, "frame_start")?;
    let duration = number_param(modifier, "frame_duration")?.max(1.0);
    let factor = ((frame - start) / duration).clamp(0.0, 1.0);
    let reverse = bool_param(modifier, "use_reverse")?;
    let threshold = if reverse { 1.0 - factor } else { factor };
    let use_random_order = bool_param(modifier, "use_random_order")?;
    let seed = uint_param(modifier, "seed")?;
    let mut face_order: Vec<usize> = (0..mesh.faces.len()).collect();
    if use_random_order {
        randomize_build_order(&mut face_order, seed);
    }
    let face_count = (mesh.faces.len() as f64 * threshold) as usize;
    let edge_count = (mesh.edges.len() as f64 * threshold) as usize;
    let vertex_count = (mesh.vertices.len() as f64 * threshold) as usize;
    let source_positions = positions(mesh);
    let source_indices = id_indices(mesh);
    let mut sources = Vec::new();
    let mut result_faces = Vec::new();
    let mut face_sources = Vec::new();
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
                    let index = sources.len();
                    sources.push(source_index);
                    vertex_map.insert(source_index, index);
                    index
                };
                vertices.push(result_index);
            }
            result_faces.push((vertices, face.material_index));
            face_sources.push(Some(face_index));
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
                    entry.insert(sources.len());
                    sources.push(source_index);
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
        sources.extend(vertex_order.into_iter().take(vertex_count));
    }
    let points: Vec<DVec3> = sources
        .iter()
        .map(|&index| source_positions[index])
        .collect();
    let origin = Origin::copies(
        sources.into_iter().map(Some),
        face_sources,
        mesh.original_indices.is_some(),
    );
    let result = build_mesh(points, result_faces, result_edges, &origin, mesh)?;
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
    // Blender's defaults: 30 degrees, angle and sharp-edge splitting both on (DNA defaults).
    let angle = number_param(modifier, "split_angle")?;
    let use_edge_angle = bool_param(modifier, "use_edge_angle")?;
    let split_sharp = bool_param(modifier, "use_edge_sharp")?;
    let face_polygons = polygons(mesh)?;
    let source_positions = positions(mesh);
    let vertex_indices = id_indices(mesh);

    let mut offsets = Vec::with_capacity(face_polygons.len());
    let mut face_normals = Vec::with_capacity(face_polygons.len());
    let mut corner_count = 0;
    for (indices, _) in &face_polygons {
        offsets.push(corner_count);
        corner_count += indices.len();
        let points: Vec<DVec3> = indices
            .iter()
            .map(|index| source_positions[*index])
            .collect();
        face_normals.push(face_normal(&points));
    }

    // Each edge's uses, with the corner of each face at the edge's lower and higher vertex.
    let mut edge_uses: HashMap<(usize, usize), Vec<_>> = HashMap::new();
    for (face_index, (indices, _)) in face_polygons.iter().enumerate() {
        for position in 0..indices.len() {
            let next = (position + 1) % indices.len();
            let (first, second) = (indices[position], indices[next]);
            let (corner_a, corner_b) = (offsets[face_index] + position, offsets[face_index] + next);
            let (key, low, high) = if first < second {
                ((first, second), corner_a, corner_b)
            } else {
                ((second, first), corner_b, corner_a)
            };
            edge_uses
                .entry(key)
                .or_default()
                .push((face_index, low, high));
        }
    }
    let sharp: HashSet<(usize, usize)> = if split_sharp {
        mesh.edges
            .iter()
            .zip(sharp_edge_flags(mesh)?)
            .filter(|(_, sharp)| *sharp)
            .map(|(edge, _)| {
                let first = vertex_indices[&edge.vertices[0]];
                let second = vertex_indices[&edge.vertices[1]];
                (first.min(second), first.max(second))
            })
            .collect()
    } else {
        HashSet::new()
    };

    // BM_mesh_edgesplit (bmesh_edgesplit.cc) with MOD_edgesplit.cc's tags. An edge with three or
    // more faces always splits; a two-face edge splits when its faces meet at more than the angle.
    let angle_split_enabled = use_edge_angle && angle < std::f64::consts::PI;
    let split_every_edge = use_edge_angle && angle < f64::from(f32::EPSILON);
    let threshold = (angle + 0.000_000_175).cos();
    let tagged: HashSet<(usize, usize)> = edge_uses
        .iter()
        .filter(|(key, uses)| {
            let angle_split = angle_split_enabled
                && (uses.len() > 2
                    || (uses.len() == 2
                        && (split_every_edge
                            || face_normals[uses[0].0].dot(face_normals[uses[1].0]) < threshold)));
            angle_split || sharp.contains(key)
        })
        .map(|(key, _)| *key)
        .collect();
    if tagged.is_empty() {
        return Ok(mesh.clone());
    }

    // A vertex splits into one copy per fan: corners joined through an untagged edge share a fan.
    let mut parent: Vec<usize> = (0..corner_count).collect();
    for (key, uses) in &edge_uses {
        if tagged.contains(key) {
            continue;
        }
        for other in uses.iter().skip(1) {
            union_corners(&mut parent, uses[0].1, other.1);
            union_corners(&mut parent, uses[0].2, other.2);
        }
    }
    let mut touched = vec![false; mesh.vertices.len()];
    for (first, second) in &tagged {
        touched[*first] = true;
        touched[*second] = true;
    }
    let mut corners_at = vec![Vec::new(); mesh.vertices.len()];
    for (face_index, (indices, _)) in face_polygons.iter().enumerate() {
        for (position, index) in indices.iter().enumerate() {
            corners_at[*index].push(offsets[face_index] + position);
        }
    }
    // Point i copies source vertex i; each extra fan appends a copy of its vertex.
    let mut sources: Vec<usize> = (0..mesh.vertices.len()).collect();
    let mut corner_point = vec![0; corner_count];
    for (vertex, corners) in corners_at.iter().enumerate() {
        let mut fans: Vec<(usize, usize)> = Vec::new();
        for &corner in corners {
            let point = if touched[vertex] {
                let root = find_corner(&mut parent, corner);
                if let Some(&(_, point)) = fans.iter().find(|(fan, _)| *fan == root) {
                    point
                } else {
                    let point = if fans.is_empty() {
                        vertex
                    } else {
                        sources.push(vertex);
                        sources.len() - 1
                    };
                    fans.push((root, point));
                    point
                }
            } else {
                vertex
            };
            corner_point[corner] = point;
        }
    }
    let out_faces: Vec<_> = face_polygons
        .into_iter()
        .enumerate()
        .map(|(face_index, (indices, material))| {
            let output = (0..indices.len())
                .map(|position| corner_point[offsets[face_index] + position])
                .collect();
            (output, material)
        })
        .collect();
    let points: Vec<DVec3> = sources
        .iter()
        .map(|&index| source_positions[index])
        .collect();
    let origin = Origin::copies(
        sources.into_iter().map(Some),
        (0..mesh.faces.len()).map(Some).collect(),
        mesh.original_indices.is_some(),
    );
    build_mesh(points, out_faces, loose_edges(mesh)?, &origin, mesh)
}

fn union_corners(parent: &mut [usize], first: usize, second: usize) {
    let first = find_corner(parent, first);
    let second = find_corner(parent, second);
    if first != second {
        parent[second] = first;
    }
}

fn find_corner(parent: &mut [usize], mut index: usize) -> usize {
    while parent[index] != index {
        parent[index] = parent[parent[index]];
        index = parent[index];
    }
    index
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

pub(super) fn mask(
    mesh: &Mesh,
    modifier: &Modifier,
    operands: &std::collections::BTreeMap<String, Vec<super::attributes::SceneOperand>>,
) -> Result<Mesh> {
    let groups = mesh
        .attributes
        .get("vertex_groups")
        .and_then(Value::as_object);
    let has_deform_weights = groups.is_some_and(|groups| {
        groups.values().any(|weights| {
            weights
                .as_object()
                .is_some_and(|weights| !weights.is_empty())
        })
    });
    let invert = bool_param(modifier, "invert_vertex_group")?;
    let Some(groups) = groups.filter(|_| has_deform_weights) else {
        return if invert {
            Ok(mesh.clone())
        } else {
            build_mesh(
                Vec::new(),
                Vec::new(),
                Vec::new(),
                &Origin {
                    // The empty template mesh retains an existing zero-element ORIGINDEX layer.
                    source_vertices: mesh.original_indices.as_ref().map(|_| Vec::new()),
                    ..Origin::default()
                },
                mesh,
            )
        };
    };

    let mode = string_param(modifier, "mode")?;
    let group_name = string_param(modifier, "vertex_group")?;
    let threshold = number_param(modifier, "threshold")?.clamp(0.0, 1.0);
    let use_smooth = mode == "VERTEX_GROUP" && bool_param(modifier, "use_smooth")?;
    let weights_by_group = match mode {
        "VERTEX_GROUP" => {
            let Some(weights) = groups.get(group_name).and_then(Value::as_object) else {
                // Blender leaves the mesh untouched when the requested group is absent.
                return Ok(mesh.clone());
            };
            vec![(group_name, weights)]
        }
        "ARMATURE" => {
            let Some(armature) = operands
                .get("armature")
                .and_then(|items| items.first())
                .filter(|operand| operand.armature)
            else {
                return Ok(mesh.clone());
            };
            groups
                .iter()
                .filter(|(name, _)| armature.selected_bones.contains(*name))
                .filter_map(|(name, value)| {
                    value.as_object().map(|weights| (name.as_str(), weights))
                })
                .collect()
        }
        _ => return Ok(mesh.clone()),
    };

    let mut selected = Vec::with_capacity(mesh.vertices.len());
    for vertex in &mesh.vertices {
        let mut is_selected = false;
        for (_, weights) in &weights_by_group {
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
            if weight > threshold {
                is_selected = true;
                break;
            }
        }
        selected.push(if invert { !is_selected } else { is_selected });
    }

    let source_positions = positions(mesh);
    let source_indices = id_indices(mesh);
    let mut vertex_map = vec![usize::MAX; source_positions.len()];
    let mut points = Vec::with_capacity(source_positions.len());
    let mut vertex_origins = Vec::with_capacity(source_positions.len());
    let mut source_vertices = mesh
        .original_indices
        .as_ref()
        .map(|_| Vec::with_capacity(source_positions.len()));
    for (source, is_selected) in selected.iter().copied().enumerate() {
        if is_selected {
            vertex_map[source] = points.len();
            points.push(source_positions[source]);
            vertex_origins.push(vec![(source, 1.0)]);
            if let Some(source_vertices) = source_vertices.as_mut() {
                source_vertices.push(Some(source));
            }
        }
    }
    let mut polygons = Vec::new();
    let mut face_origins = Vec::new();
    let mut corner_origins = Vec::new();
    let mut corner_offsets = Vec::with_capacity(mesh.faces.len());
    let mut corner_offset = 0_usize;
    for face in &mesh.faces {
        corner_offsets.push(corner_offset);
        corner_offset = corner_offset
            .checked_add(face.vertices.len())
            .ok_or_else(|| invalid_parameter(modifier, "mesh", "a valid loop count"))?;
    }
    let group_weights = if mode == "VERTEX_GROUP" {
        weights_by_group.first().map(|(_, weights)| *weights)
    } else {
        None
    };
    let weight_for_index = |index: usize| {
        let vertex_id = mesh.vertices[index].id;
        group_weights
            .and_then(|weights| {
                weights
                    .get(&format!("v{vertex_id}"))
                    .and_then(Value::as_f64)
            })
            .unwrap_or(0.0)
    };
    let mut boundary_vertices = HashMap::<(usize, usize), usize>::new();
    let mut boundary_vertex = |first: usize, second: usize| {
        let key = if first < second {
            (first, second)
        } else {
            (second, first)
        };
        if let Some(index) = boundary_vertices.get(&key) {
            return *index;
        }
        let first_weight = weight_for_index(key.0);
        let second_weight = weight_for_index(key.1);
        let factor = (threshold - first_weight) / (second_weight - first_weight);
        let index = points.len();
        points.push(source_positions[key.0].lerp(source_positions[key.1], factor));
        vertex_origins.push(vec![(key.0, 1.0 - factor), (key.1, factor)]);
        if let Some(source_vertices) = source_vertices.as_mut() {
            source_vertices.push(None);
        }
        boundary_vertices.insert(key, index);
        index
    };

    for (face_index, face) in mesh.faces.iter().enumerate() {
        let indices = face
            .vertices
            .iter()
            .map(|id| {
                source_indices.get(id).copied().ok_or_else(|| {
                    invalid_parameter(modifier, "mesh", "faces referencing existing vertices")
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let face_selected = indices
            .iter()
            .map(|index| selected[*index])
            .collect::<Vec<_>>();
        if face_selected.iter().all(|value| *value) {
            let output = indices.iter().map(|index| vertex_map[*index]).collect();
            polygons.push((output, face.material_index));
            face_origins.push(Some(face_index));
            corner_origins.extend(
                (0..indices.len()).map(|corner| vec![(corner_offsets[face_index] + corner, 1.0)]),
            );
            continue;
        }
        if !use_smooth || face_selected.iter().all(|value| !*value) {
            continue;
        }

        let start = face_selected
            .iter()
            .position(|value| !value)
            .expect("a partially selected face has an unselected corner");
        let mut fragment = Vec::<(usize, Vec<(usize, f64)>)>::new();
        for step in 1..=indices.len() {
            let previous = (start + step - 1) % indices.len();
            let current = (start + step) % indices.len();
            let was_selected = face_selected[previous];
            let is_selected = face_selected[current];
            if !was_selected && is_selected {
                let previous_index = indices[previous];
                let current_index = indices[current];
                let boundary = boundary_vertex(previous_index, current_index);
                let previous_weight = weight_for_index(previous_index);
                let current_weight = weight_for_index(current_index);
                let factor = (threshold - previous_weight) / (current_weight - previous_weight);
                fragment.push((
                    boundary,
                    vec![
                        (corner_offsets[face_index] + previous, 1.0 - factor),
                        (corner_offsets[face_index] + current, factor),
                    ],
                ));
                fragment.push((
                    vertex_map[current_index],
                    vec![(corner_offsets[face_index] + current, 1.0)],
                ));
            } else if was_selected && is_selected {
                fragment.push((
                    vertex_map[indices[current]],
                    vec![(corner_offsets[face_index] + current, 1.0)],
                ));
            } else if was_selected && !is_selected {
                let previous_index = indices[previous];
                let current_index = indices[current];
                let boundary = boundary_vertex(previous_index, current_index);
                let previous_weight = weight_for_index(previous_index);
                let current_weight = weight_for_index(current_index);
                let factor = (threshold - previous_weight) / (current_weight - previous_weight);
                fragment.push((
                    boundary,
                    vec![
                        (corner_offsets[face_index] + previous, 1.0 - factor),
                        (corner_offsets[face_index] + current, factor),
                    ],
                ));
                if fragment.len() >= 3 {
                    polygons.push((
                        fragment.iter().map(|(index, _)| *index).collect(),
                        face.material_index,
                    ));
                    face_origins.push(Some(face_index));
                    corner_origins.extend(fragment.drain(..).map(|(_, origin)| origin));
                } else {
                    fragment.clear();
                }
            }
        }
    }

    let source_loose_edges = loose_edges(mesh)?;
    let mut edges = Vec::new();
    for [first, second] in source_loose_edges {
        match (selected[first], selected[second], use_smooth) {
            (true, true, _) => edges.push([vertex_map[first], vertex_map[second]]),
            (true, false, true) => {
                let boundary = boundary_vertex(first, second);
                edges.push([vertex_map[first], boundary]);
            }
            (false, true, true) => {
                let boundary = boundary_vertex(first, second);
                edges.push([boundary, vertex_map[second]]);
            }
            _ => {}
        }
    }
    let origin = Origin {
        vertices: vertex_origins,
        source_vertices,
        faces: face_origins,
        corners: Some(corner_origins),
    };
    build_mesh(points, polygons, edges, &origin, mesh)
}

#[expect(
    clippy::manual_midpoint,
    reason = "matches Blender's float evaluation order for parity"
)]
pub(super) fn wireframe(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    let thickness = number_param(modifier, "thickness")?;
    let offset = number_param(modifier, "offset")?;
    if thickness < 0.0 {
        return Err(invalid_parameter(
            modifier,
            "thickness",
            "a non-negative number",
        ));
    }
    let use_boundary = bool_param(modifier, "use_boundary")?;
    let use_replace = bool_param(modifier, "use_replace")?;
    let use_even_offset = bool_param(modifier, "use_even_offset")?;
    let use_relative_offset = bool_param(modifier, "use_relative_offset")?;
    let use_crease = bool_param(modifier, "use_crease")?;
    let crease_weight = number_param(modifier, "crease_weight")?;
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
    // Blender's vertex normals (Mesh::vert_normals) weight each corner's normal by its angle; with
    // custom normals the corner normals themselves are mixed.
    let custom_corners = clnor::corner_codes(mesh)?
        .is_some()
        .then(|| corner_normals(mesh, &source_positions))
        .transpose()?;
    let mut corner_offset = 0;
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
        let face_points: Vec<DVec3> = indices
            .iter()
            .map(|index| source_positions[*index])
            .collect();
        for (position, index) in indices.iter().enumerate() {
            let value = custom_corners
                .as_ref()
                .map_or(normal, |values| values[corner_offset + position]);
            normals[*index] += value * corner_angle(&face_points, position);
        }
        corner_offset += indices.len();
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
        points.extend(source_positions.iter().copied().enumerate());
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
        points.push((
            index,
            center + normals[index] * ((offset_original - offset_midpoint) * factor),
        ));
        positive_vertices[index] = points.len();
        points.push((
            index,
            center + normals[index] * ((offset_new - offset_midpoint) * factor),
        ));
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
            points.push((current, point));
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
                    points.push((boundary_vertex, point));
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
    let mut face_sources: Vec<usize> = Vec::new();
    if !use_replace {
        for (face_index, source_face) in mesh.faces.iter().enumerate() {
            faces.push((face_indices[face_index].clone(), source_face.material_index));
            face_sources.push(face_index);
        }
    }
    let mut crease_edges = HashSet::new();
    for (face_index, indices) in face_indices.iter().enumerate() {
        let source_face = &mesh.faces[face_index];
        let material_index = i64::from(source_face.material_index)
            .saturating_add(material_offset)
            .clamp(0, i64::from(material_max)) as u32;
        let emitted_before = faces.len();
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
        face_sources.extend(std::iter::repeat_n(
            face_index,
            faces.len() - emitted_before,
        ));
    }
    let (sources, coords): (Vec<usize>, Vec<DVec3>) = points.into_iter().unzip();
    // Each output corner copies the loop of its source face at the same vertex, so raw custom data
    // follows that corner (BM_elem_attrs_copy in bmesh_wireframe.cc).
    let mut corner_offset = 0;
    let mut input_corner_offsets = Vec::with_capacity(face_indices.len());
    for indices in &face_indices {
        input_corner_offsets.push(corner_offset);
        corner_offset += indices.len();
    }
    let mut source_corners = Vec::new();
    for ((corner_points, _), face_index) in faces.iter().zip(&face_sources) {
        for point in corner_points {
            let vertex = sources[*point];
            let position = face_indices[*face_index]
                .iter()
                .position(|index| *index == vertex)
                .ok_or_else(|| {
                    invalid_parameter(modifier, "mesh", "wireframe corners on their source face")
                })?;
            source_corners.push(input_corner_offsets[*face_index] + position);
        }
    }
    let origin = Origin::copies(
        sources.into_iter().map(Some),
        face_sources.into_iter().map(Some).collect(),
        mesh.original_indices.is_some(),
    );
    let mut result = build_mesh(coords, faces, Vec::new(), &origin, mesh)?;
    // BMesh gives new edges the SMOOTH flag (bmesh_core.cc), so only source edges between kept
    // original vertices (use_replace off) stay sharp.
    let source_sharp: HashSet<(usize, usize)> = mesh
        .edges
        .iter()
        .zip(sharp_edge_flags(mesh)?)
        .filter(|(_, sharp)| *sharp)
        .map(|(edge, _)| {
            let first = vertex_indices[&edge.vertices[0]];
            let second = vertex_indices[&edge.vertices[1]];
            (first.min(second), first.max(second))
        })
        .collect();
    let result_indices = id_indices(&result);
    let original_count = if use_replace { 0 } else { mesh.vertices.len() };
    let edge_keys: Vec<(usize, usize)> = result
        .edges
        .iter()
        .map(|edge| {
            let first = result_indices[&edge.vertices[0]];
            let second = result_indices[&edge.vertices[1]];
            (first.min(second), first.max(second))
        })
        .collect();
    let output_sharp: HashSet<(usize, usize)> = edge_keys
        .iter()
        .copied()
        .filter(|(first, second)| {
            *second < original_count && source_sharp.contains(&(*first, *second))
        })
        .collect();
    if mesh.attributes.contains_key("sharp") {
        let values: serde_json::Map<String, Value> = result
            .edges
            .iter()
            .zip(&edge_keys)
            .map(|(edge, key)| {
                (
                    format!("e{}", edge.id),
                    Value::Bool(output_sharp.contains(key)),
                )
            })
            .collect();
        result.attributes.insert(
            "sharp".to_owned(),
            serde_json::json!({"domain": "edge", "type": "bool", "values": values}),
        );
    }
    // Each output corner copies the raw custom data of its source corner, which its new fan decodes
    // (BM_elem_attrs_copy in bmesh_wireframe.cc, read back through the output fans).
    let sources: Vec<Option<usize>> = source_corners.iter().map(|corner| Some(*corner)).collect();
    clnor::transfer_custom_normals(mesh, &mut result, &sources, NormalCodeOptions::default())?;
    result.attributes.remove("wireframe_use_crease");
    result.attributes.remove("wireframe_crease_weight");
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

/// Blender `MOD_curve.cc` with `BKE_curve_deform_coords` (bounds off, no stretch): the deform axis is the
/// unbounded arc length along the path, and the other two coordinates place the vertex in the path frame.
pub(super) fn curve(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    let Some(target) = modifier.runtime.target_mesh.as_ref() else {
        // Blender disables the modifier without a curve object.
        return Ok(mesh.clone());
    };
    let (axis, negative) = curve_deform_axis(modifier)?;
    let object = reference_matrix(modifier).ok_or_else(|| {
        PotError::new(
            ErrorCode::DependencyMissing,
            "curve modifier object transform is missing",
        )
    })?;
    // The path and the vertices are measured in curve-local space, where the deform axis is defined.
    let to_curve = object.inverse();
    let path: Vec<DVec3> = target
        .vertices
        .iter()
        .map(|vertex| to_curve.transform_point3(vertex.co))
        .collect();
    let arc_at_point = curve_arc_lengths(modifier, &path)?;
    let frames = bevel_frames(&path);
    let weights = deform_group_weights(mesh, modifier)?;
    let mut result = mesh.clone();
    for (index, vertex) in result.vertices.iter_mut().enumerate() {
        let weight = group_weight(weights.as_deref(), index);
        if weight <= 0.0 {
            continue;
        }
        let local = to_curve.transform_point3(vertex.co);
        let (arc, cross) = if negative {
            (
                -local[axis],
                DVec3::new(-local[(axis + 2) % 3], -local[(axis + 1) % 3], 0.0),
            )
        } else {
            (
                local[axis],
                DVec3::new(local[(axis + 1) % 3], local[(axis + 2) % 3], 0.0),
            )
        };
        let (position, rotation) = path_frame_at(&path, &frames, &arc_at_point, arc)?;
        let deformed = position + rotation * cross;
        vertex.co = object.transform_point3(local.lerp(deformed, weight));
    }

    validated(&result)?;
    Ok(result)
}

/// The deform axis index and whether the axis is negative (Blender `POS_X`..`NEG_Z`).
fn curve_deform_axis(modifier: &Modifier) -> Result<(usize, bool)> {
    let axis = string_param(modifier, "deform_axis")?.to_ascii_uppercase();
    match axis.as_str() {
        "POS_X" => Ok((0, false)),
        "NEG_X" => Ok((0, true)),
        "POS_Y" => Ok((1, false)),
        "NEG_Y" => Ok((1, true)),
        "POS_Z" => Ok((2, false)),
        "NEG_Z" => Ok((2, true)),
        _ => Err(invalid_parameter(
            modifier,
            "deform_axis",
            "POS_X, NEG_X, POS_Y, NEG_Y, POS_Z, or NEG_Z",
        )),
    }
}

/// Cumulative arc length at every path point (Blender `BKE_anim_path_calc_data`).
fn curve_arc_lengths(modifier: &Modifier, path: &[DVec3]) -> Result<Vec<f64>> {
    let invalid = || {
        invalid_parameter(
            modifier,
            "object",
            "a curve with at least two distinct points",
        )
    };
    if path.len() < 2 {
        return Err(invalid());
    }
    let mut arc = Vec::with_capacity(path.len());
    arc.push(0.0);
    let mut total = 0.0;
    for pair in path.windows(2) {
        if let [from, to] = pair {
            let length = from.distance(*to);
            if length <= f64::EPSILON {
                return Err(invalid());
            }
            total += length;
            arc.push(total);
        }
    }
    Ok(arc)
}

/// Blender `BKE_where_on_path` for poly paths: the path position and frame at arc length `arc`,
/// interpolated linearly (position) and by slerp (frame), extended past both ends.
fn path_frame_at(
    path: &[DVec3],
    frames: &[DQuat],
    arc_at_point: &[f64],
    arc: f64,
) -> Result<(DVec3, DQuat)> {
    let total = arc_at_point.last().copied().unwrap_or(0.0);
    let segment_count = arc_at_point.len().saturating_sub(1);
    let index = if arc <= 0.0 {
        0
    } else if arc >= total {
        segment_count.saturating_sub(1)
    } else {
        arc_at_point
            .partition_point(|&length| length <= arc)
            .saturating_sub(1)
    };
    let (
        Some(&start_arc),
        Some(&end_arc),
        Some(&from),
        Some(&to),
        Some(&from_frame),
        Some(&to_frame),
    ) = (
        arc_at_point.get(index),
        arc_at_point.get(index + 1),
        path.get(index),
        path.get(index + 1),
        frames.get(index),
        frames.get(index + 1),
    )
    else {
        return Err(PotError::new(
            ErrorCode::InternalError,
            "curve path lookup is out of range",
        ));
    };
    let fraction = (arc - start_arc) / (end_arc - start_arc);
    let position = from.lerp(to, fraction);
    let rotation = if fraction <= 0.0 {
        from_frame
    } else if fraction >= 1.0 {
        to_frame
    } else {
        let end_frame = if from_frame.dot(to_frame) < 0.0 {
            -to_frame
        } else {
            to_frame
        };
        from_frame.slerp(end_frame, fraction)
    };
    Ok((position, rotation))
}

/// Blender's bevel frames for a 3D poly path with minimum twist: bisected tangents, end frames from the
/// tangent alone, and interior frames carried along by the minimal rotation between tangents.
fn bevel_frames(path: &[DVec3]) -> Vec<DQuat> {
    let segments: Vec<DVec3> = path
        .windows(2)
        .filter_map(|pair| match pair {
            [from, to] => Some((*to - *from).normalize_or_zero()),
            _ => None,
        })
        .collect();
    let mut tangents = Vec::with_capacity(path.len());
    tangents.extend(segments.first().copied());
    tangents.extend(
        segments
            .iter()
            .zip(segments.iter().skip(1))
            .map(|(before, after)| (*before + *after).normalize_or_zero()),
    );
    tangents.extend(segments.last().copied());
    let count = tangents.len();
    let mut frames = Vec::with_capacity(count);
    let mut previous: Option<(DVec3, DQuat)> = None;
    for (index, tangent) in tangents.into_iter().enumerate() {
        let frame = match previous {
            Some((prior_tangent, prior_frame)) if index + 1 < count => {
                transport_frame(prior_frame, prior_tangent, tangent)
            }
            _ => track_frame(tangent),
        };
        frames.push(frame);
        previous = Some((tangent, frame));
    }
    frames
}

/// Blender's `vec_to_quat(tangent, -Z, Y)`: local Z follows the tangent, local X and Y are the normal and binormal.
fn track_frame(tangent: DVec3) -> DQuat {
    let tangent = tangent.normalize_or_zero();
    if tangent == DVec3::ZERO {
        return DQuat::IDENTITY;
    }
    let mut axis = DVec3::new(-tangent.y, tangent.x, 0.0);
    if axis.length_squared() <= 1.0e-12 {
        axis = DVec3::X;
    }
    let align = DQuat::from_axis_angle(axis.normalize(), tangent.z.clamp(-1.0, 1.0).acos());
    let roll_angle = if tangent.x.abs() < 1.0e-12 && tangent.y.abs() < 1.0e-12 {
        // Vertical tangents use Blender's fixed roll for either direction.
        if tangent.z > 0.0 {
            std::f64::consts::PI
        } else {
            0.0
        }
    } else {
        -(-tangent.x).atan2(-tangent.y)
    };
    (DQuat::from_axis_angle(tangent, roll_angle) * align).normalize()
}

/// Blender's minimal-twist step: rotate the frame from one tangent onto the next.
fn transport_frame(frame: DQuat, from: DVec3, to: DVec3) -> DQuat {
    let angle = from.dot(to).clamp(-1.0, 1.0).acos();
    let axis = from.cross(to).normalize_or_zero();
    if angle > 0.0 && axis != DVec3::ZERO {
        DQuat::from_axis_angle(axis, angle) * frame
    } else {
        frame
    }
}

/// Blender `MOD_hook.cc`: the selected vertices move by `object * matrix_inverse`, weighted by the
/// strength, the falloff and the vertex group.
pub(super) fn hook(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    let falloff = hook_falloff_param(modifier)?;
    let radius = number_param(modifier, "falloff_radius")?;
    if radius < 0.0 {
        return Err(invalid_parameter(
            modifier,
            "falloff_radius",
            "a non-negative radius",
        ));
    }
    let strength = number_param(modifier, "strength")?;
    let center = vec3_param(modifier, "center")?;
    let uniform = bool_param(modifier, "use_falloff_uniform")?;
    let parent_inverse = matrix_param(modifier, "matrix_inverse")?;
    let indices = vertex_index_param(modifier)?;
    let falloff_table = if falloff == HookFalloff::Curve {
        Some(hook_curve_table(modifier)?)
    } else {
        None
    };
    let weights = deform_group_weights(mesh, modifier)?;
    // Blender hooks only listed vertices; without a list, a vertex group selects them.
    let listed = !indices.is_empty();
    if !listed && weights.is_none() {
        return Ok(mesh.clone());
    }
    let Some(object) = reference_matrix(modifier) else {
        // Blender disables the hook without an object.
        return Ok(mesh.clone());
    };
    let hook_matrix = object * parent_inverse;
    let uniform_matrix = DMat3::from_mat4(parent_inverse);
    let original_indices = mesh.original_indices.as_ref();
    let source_vertex_count =
        original_indices.map_or(mesh.vertices.len(), |indices| indices.source_vertex_count);
    let mut selected = vec![false; source_vertex_count];
    // Blender 5.2.2 validates each listed value but enables the bitmap at its list position.
    for (position, &index) in indices.iter().enumerate() {
        if index < source_vertex_count && position < selected.len() {
            selected[position] = true;
        }
    }
    let use_falloff = falloff != HookFalloff::NoFalloff && radius > 0.0;
    let mut result = mesh.clone();
    for (index, vertex) in result.vertices.iter_mut().enumerate() {
        let source_index = original_indices.map_or(Some(index), |indices| {
            indices
                .vertices
                .get(index)
                .copied()
                .and_then(|source| usize::try_from(source).ok())
        });
        if listed
            && !source_index
                .and_then(|source| selected.get(source).copied())
                .unwrap_or(false)
        {
            continue;
        }
        let offset = vertex.co - center;
        let distance_sq = if uniform {
            (uniform_matrix * offset).length_squared()
        } else {
            offset.length_squared()
        };
        let shape = if use_falloff {
            hook_falloff(falloff, distance_sq, radius, falloff_table.as_ref())
        } else {
            1.0
        };
        let fac = strength * shape * group_weight(weights.as_deref(), index);
        if fac == 0.0 {
            continue;
        }
        let hooked = hook_matrix.transform_point3(vertex.co);
        vertex.co = vertex.co.lerp(hooked, fac);
    }
    validated(&result)?;
    Ok(result)
}

/// Blender's falloff shape for a squared distance from the hook centre (1 at the centre, 0 outside the radius).
fn hook_falloff(
    falloff: HookFalloff,
    distance_sq: f64,
    radius: f64,
    curve: Option<&CurveMappingTable>,
) -> f64 {
    let radius_sq = radius * radius;
    if distance_sq > radius_sq {
        return 0.0;
    }
    if distance_sq <= 0.0 {
        return 1.0;
    }
    let linear = 1.0 - distance_sq.sqrt() / radius;
    match falloff {
        HookFalloff::Constant => 1.0,
        HookFalloff::InverseSquare => 1.0 - distance_sq / radius_sq,
        HookFalloff::Curve => curve.map_or(0.0, |table| table.evaluate(linear)),
        HookFalloff::Sharp => linear * linear,
        HookFalloff::Smooth => 3.0 * linear * linear - 2.0 * linear * linear * linear,
        HookFalloff::Root => linear.sqrt(),
        HookFalloff::Sphere => (2.0 * linear - linear * linear).sqrt(),
        HookFalloff::Linear | HookFalloff::NoFalloff => linear,
    }
}

/// Blender `hook_falloff` shape selectors (`eHook_Falloff_*`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum HookFalloff {
    NoFalloff,
    Curve,
    Smooth,
    Sphere,
    Root,
    InverseSquare,
    Sharp,
    Linear,
    Constant,
}

fn hook_falloff_param(modifier: &Modifier) -> Result<HookFalloff> {
    Ok(match string_param(modifier, "falloff_type")? {
        "NONE" => HookFalloff::NoFalloff,
        "CURVE" => HookFalloff::Curve,
        "SMOOTH" => HookFalloff::Smooth,
        "SPHERE" => HookFalloff::Sphere,
        "ROOT" => HookFalloff::Root,
        "INVERSE_SQUARE" => HookFalloff::InverseSquare,
        "SHARP" => HookFalloff::Sharp,
        "LINEAR" => HookFalloff::Linear,
        "CONSTANT" => HookFalloff::Constant,
        _ => {
            return Err(invalid_parameter(
                modifier,
                "falloff_type",
                "NONE, CURVE, SMOOTH, SPHERE, ROOT, INVERSE_SQUARE, SHARP, LINEAR, or CONSTANT",
            ));
        }
    })
}

/// Hook `vertex_indices`: vertex positions in storage order (Blender's `indexar`).
fn vertex_index_param(modifier: &Modifier) -> Result<Vec<usize>> {
    let invalid = || {
        invalid_parameter(
            modifier,
            "vertex_indices",
            "an array of non-negative vertex indices",
        )
    };
    schema_value(modifier, "vertex_indices")?
        .as_array()
        .ok_or_else(invalid)?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|index| usize::try_from(index).ok())
                .ok_or_else(invalid)
        })
        .collect()
}

/// A 4x4 parameter stored as 16 column-major numbers.
fn matrix_param(modifier: &Modifier, name: &str) -> Result<DMat4> {
    let invalid = || invalid_parameter(modifier, name, "sixteen finite column-major numbers");
    let values = schema_value(modifier, name)?
        .as_array()
        .filter(|values| values.len() == 16)
        .ok_or_else(invalid)?;
    let mut matrix = [0.0; 16];
    for (slot, value) in matrix.iter_mut().zip(values) {
        *slot = value
            .as_f64()
            .filter(|number| number.is_finite())
            .ok_or_else(invalid)?;
    }
    Ok(DMat4::from_cols_array(&matrix))
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum CurveMappingHandle {
    Auto,
    AutoClamped,
    Vector,
}

#[derive(Clone, Copy)]
pub(super) struct CurveMappingPoint {
    location: Vec2,
    handle: CurveMappingHandle,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum CurveMappingExtend {
    Horizontal,
    Extrapolated,
}

#[derive(Clone, Copy)]
pub(super) struct CurveMappingSettings {
    use_clip: bool,
    clip_min: Vec2,
    clip_max: Vec2,
    extend: CurveMappingExtend,
}

impl Default for CurveMappingSettings {
    fn default() -> Self {
        Self {
            use_clip: true,
            clip_min: Vec2::ZERO,
            clip_max: Vec2::ONE,
            extend: CurveMappingExtend::Extrapolated,
        }
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "Blender stores CurveMapping clip bounds as float"
)]
pub(super) fn curve_mapping_settings(value: Option<&Value>) -> Option<CurveMappingSettings> {
    let Some(value) = value else {
        return Some(CurveMappingSettings::default());
    };
    let settings = value.as_object()?;
    let use_clip = settings
        .get("use_clip")
        .map_or(Some(true), Value::as_bool)?;
    let clip_bounds = if let Some(value) = settings.get("clip_bounds") {
        let values = value.as_array()?;
        let [min_x, min_y, max_x, max_y] = values.as_slice() else {
            return None;
        };
        let bound = |value: &Value| {
            let value = value.as_f64()?;
            if !value.is_finite() || !(-100.0..=100.0).contains(&value) {
                return None;
            }
            let value = value as f32;
            value.is_finite().then_some(value)
        };
        [bound(min_x)?, bound(min_y)?, bound(max_x)?, bound(max_y)?]
    } else {
        [0.0, 0.0, 1.0, 1.0]
    };
    let [min_x, min_y, max_x, max_y] = clip_bounds;
    if min_x > max_x || min_y > max_y {
        return None;
    }
    let extend = match settings.get("extend") {
        None => CurveMappingExtend::Extrapolated,
        Some(value) => match value.as_str()? {
            "HORIZONTAL" => CurveMappingExtend::Horizontal,
            "EXTRAPOLATED" => CurveMappingExtend::Extrapolated,
            _ => return None,
        },
    };
    Some(CurveMappingSettings {
        use_clip,
        clip_min: Vec2::new(min_x, min_y),
        clip_max: Vec2::new(max_x, max_y),
        extend,
    })
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "Blender CurveMapping stores point coordinates as float"
)]
pub(super) fn curve_mapping_points(value: &Value) -> Option<Vec<CurveMappingPoint>> {
    let points = value.as_array()?;
    if points.len() < 2 {
        return None;
    }
    points
        .iter()
        .map(|point| {
            let coordinates = point.as_array()?;
            if !(2..=3).contains(&coordinates.len()) {
                return None;
            }
            let x = coordinates[0].as_f64()?;
            let y = coordinates[1].as_f64()?;
            if !x.is_finite() || !y.is_finite() {
                return None;
            }
            let location = Vec2::new(x as f32, y as f32);
            if !location.is_finite() {
                return None;
            }
            let handle = match coordinates.get(2) {
                None => CurveMappingHandle::Auto,
                Some(value) => match value.as_str() {
                    Some("AUTO") => CurveMappingHandle::Auto,
                    Some("AUTO_CLAMPED") => CurveMappingHandle::AutoClamped,
                    Some("VECTOR") => CurveMappingHandle::Vector,
                    _ => return None,
                },
            };
            Some(CurveMappingPoint { location, handle })
        })
        .collect()
}

/// Hook falloff controls and optional CurveMapping settings in Blender's stored order and precision.
fn hook_curve_table(modifier: &Modifier) -> Result<CurveMappingTable> {
    let invalid_points = || {
        invalid_parameter(
            modifier,
            "falloff_curve",
            "at least two finite float32 [x, y] points with an optional AUTO, AUTO_CLAMPED, or VECTOR handle",
        )
    };
    let points = curve_mapping_points(schema_value(modifier, "falloff_curve")?)
        .ok_or_else(invalid_points)?;
    let settings = curve_mapping_settings(modifier.params.get("falloff_curve_settings"))
        .ok_or_else(|| {
            invalid_parameter(
                modifier,
                "falloff_curve_settings",
                "use_clip, finite ordered clip_bounds, and HORIZONTAL or EXTRAPOLATED extend",
            )
        })?;
    curve_map_table(&points, settings).ok_or_else(invalid_points)
}

const CURVE_SAMPLES_PER_SEGMENT: usize = 32;
const CURVE_TABLE_INTERVALS: usize = 256;
const CURVE_TABLE_DIVISIONS: f32 = 256.0;

/// Blender's float `CurveMapping` table: point-specific handles feed forward-difference Bezier
/// sampling, then equal-x resampling over the clip-extended control range.
pub(super) struct CurveMappingTable {
    values: Vec<f32>,
    ext_in: Vec2,
    ext_out: Vec2,
    table_min: f32,
    table_max: f32,
    inverse_range: f32,
    settings: CurveMappingSettings,
}

impl CurveMappingTable {
    /// Blender `BKE_curvemapping_evaluateF`, including configurable extrapolation and Y clipping.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "Blender CurveMapping evaluates float inputs and table indices"
    )]
    pub(super) fn evaluate(&self, value: f64) -> f64 {
        let value = value as f32;
        let fi = (value - self.table_min) * self.inverse_range;
        let result = if fi < 0.0 || fi > CURVE_TABLE_DIVISIONS {
            curve_map_extend(
                value,
                Vec2::new(self.table_min, self.values[0]),
                Vec2::new(self.table_max, self.values[CURVE_TABLE_INTERVALS]),
                self.ext_in,
                self.ext_out,
                self.settings.extend,
            )
        } else {
            let index = fi as usize;
            if index >= CURVE_TABLE_INTERVALS {
                self.values[CURVE_TABLE_INTERVALS]
            } else {
                let fraction = fi - index as f32;
                (1.0 - fraction) * self.values[index] + fraction * self.values[index + 1]
            }
        };
        let result = if self.settings.use_clip {
            result.clamp(self.settings.clip_min.y, self.settings.clip_max.y)
        } else {
            result
        };
        f64::from(result)
    }
}

/// Builds Blender's `curvemap_make_table` for raw controls in stored order,
/// resampling over the settings-extended x range.
#[expect(
    clippy::cast_precision_loss,
    reason = "Blender's table loop converts its integer index to float"
)]
pub(super) fn curve_map_table(
    points: &[CurveMappingPoint],
    settings: CurveMappingSettings,
) -> Option<CurveMappingTable> {
    let count = points.len();
    if count < 2
        || points.iter().any(|point| !point.location.is_finite())
        || !settings.clip_min.is_finite()
        || !settings.clip_max.is_finite()
        || settings.clip_min.x > settings.clip_max.x
        || settings.clip_min.y > settings.clip_max.y
    {
        return None;
    }
    let table_min = points
        .iter()
        .fold(settings.clip_min.x, |min, point| min.min(point.location.x));
    let table_max = points
        .iter()
        .fold(settings.clip_max.x, |max, point| max.max(point.location.x));
    let table_range = table_max - table_min;
    if !table_range.is_finite() || table_range <= 0.0 {
        return None;
    }

    let mut handles: Vec<[Vec2; 3]> = Vec::with_capacity(count);
    for (index, control) in points.iter().enumerate() {
        let point = control.location;
        let previous = index
            .checked_sub(1)
            .and_then(|prior| points.get(prior))
            .map(|control| control.location);
        let next = points.get(index + 1).map(|control| control.location);
        let (before, after) = match (previous, next) {
            (Some(before), Some(after)) => (before, after),
            (None, Some(after)) => (2.0 * point - after, after),
            (Some(before), None) => (before, 2.0 * point - before),
            (None, None) => (point, point),
        };
        let incoming = point - before;
        let outgoing = after - point;
        let mut length_in = incoming.length();
        let mut length_out = outgoing.length();
        if length_in == 0.0 {
            length_in = 1.0;
        }
        if length_out == 0.0 {
            length_out = 1.0;
        }
        let mut left = Vec2::ZERO;
        let mut right = Vec2::ZERO;
        match control.handle {
            CurveMappingHandle::Auto | CurveMappingHandle::AutoClamped => {
                let tangent = outgoing / length_out + incoming / length_in;
                let scale = tangent.length() * 2.5614_f32;
                if scale != 0.0 {
                    left = point + tangent * -(length_in / scale);
                    right = point + tangent * (length_out / scale);
                    if control.handle == CurveMappingHandle::AutoClamped
                        && let (Some(previous), Some(next)) = (previous, next)
                    {
                        let ydiff_previous = previous.y - point.y;
                        let ydiff_next = next.y - point.y;
                        if (ydiff_previous <= 0.0 && ydiff_next <= 0.0)
                            || (ydiff_previous >= 0.0 && ydiff_next >= 0.0)
                        {
                            left.y = point.y;
                        } else if ydiff_previous <= 0.0 {
                            left.y = left.y.max(previous.y);
                        } else {
                            left.y = left.y.min(previous.y);
                        }
                        if (ydiff_previous <= 0.0 && ydiff_next <= 0.0)
                            || (ydiff_previous >= 0.0 && ydiff_next >= 0.0)
                        {
                            right.y = point.y;
                        } else if ydiff_previous <= 0.0 {
                            right.y = right.y.min(next.y);
                        } else {
                            right.y = right.y.max(next.y);
                        }
                    }
                }
            }
            CurveMappingHandle::Vector => {
                left = point + incoming * (-1.0_f32 / 3.0_f32);
                right = point + outgoing * (1.0_f32 / 3.0_f32);
            }
        }
        handles.push([left, point, right]);
    }

    if count > 2 {
        if points[0].handle == CurveMappingHandle::Auto {
            let first_length = (handles[0][1] - handles[0][2]).length();
            let mut offset = handles[1][0];
            offset.x = offset.x.max(handles[0][1].x);
            offset -= handles[0][1];
            let offset_length = offset.length();
            if offset_length > f32::EPSILON {
                offset *= first_length / offset_length;
                handles[0][2] = offset + handles[0][1];
                handles[0][0] = handles[0][1] - offset;
            }
        }

        let last = count - 1;
        if points[last].handle == CurveMappingHandle::Auto {
            let last_length = (handles[last][1] - handles[last][0]).length();
            let mut offset = handles[last - 1][2];
            offset.x = offset.x.min(handles[last][1].x);
            offset -= handles[last][1];
            let offset_length = offset.length();
            if offset_length > f32::EPSILON {
                offset *= last_length / offset_length;
                handles[last][0] = offset + handles[last][1];
                handles[last][2] = handles[last][1] - offset;
            }
        }
    }
    if handles.iter().flatten().any(|point| !point.is_finite()) {
        return None;
    }

    for segment in 0..count - 1 {
        let (right_handle, left_handle) = correct_bezier_handles(
            handles[segment][1],
            handles[segment][2],
            handles[segment + 1][0],
            handles[segment + 1][1],
        );
        handles[segment][2] = right_handle;
        handles[segment + 1][0] = left_handle;
    }
    if handles.iter().flatten().any(|point| !point.is_finite()) {
        return None;
    }

    let mut samples = Vec::with_capacity((count - 1) * CURVE_SAMPLES_PER_SEGMENT);
    for segment in 0..count - 1 {
        let start = handles[segment];
        let end = handles[segment + 1];
        let mut x = bezier_forward_difference(start[1].x, start[2].x, end[0].x, end[1].x);
        let mut y = bezier_forward_difference(start[1].y, start[2].y, end[0].y, end[1].y);
        for _ in 0..CURVE_SAMPLES_PER_SEGMENT {
            samples.push(Vec2::new(x[0], y[0]));
            x[0] += x[1];
            x[1] += x[2];
            x[2] += x[3];
            y[0] += y[1];
            y[1] += y[2];
            y[2] += y[3];
        }
    }
    if samples.iter().any(|point| !point.is_finite()) {
        return None;
    }

    let last_sample = samples.len() - 1;
    let first_point = samples[0];
    let end_point = samples[last_sample];
    let ext_in = handles[0][0] - handles[0][1];
    let ext_in_length = ext_in.length();
    let ext_out = handles[count - 1][1] - handles[count - 1][2];
    let ext_out_length = ext_out.length();
    if !ext_in_length.is_finite()
        || ext_in_length == 0.0
        || !ext_out_length.is_finite()
        || ext_out_length == 0.0
    {
        return None;
    }
    let ext_in = ext_in / ext_in_length;
    let ext_out = ext_out / ext_out_length;
    let table_step = table_range / CURVE_TABLE_DIVISIONS;
    let inverse_range = CURVE_TABLE_DIVISIONS / table_range;
    if !ext_in.is_finite()
        || !ext_out.is_finite()
        || !table_step.is_finite()
        || !inverse_range.is_finite()
    {
        return None;
    }

    let mut table = Vec::with_capacity(CURVE_TABLE_INTERVALS + 1);
    let mut sample_index = 0;
    for step in 0..=CURVE_TABLE_INTERVALS {
        let x = table_min + table_step * step as f32;
        while sample_index < last_sample && x >= samples[sample_index].x {
            sample_index += 1;
        }
        let value =
            if sample_index == 0 || (sample_index == last_sample && x >= samples[sample_index].x) {
                if (x - samples[sample_index].x).abs() <= 1.0e-6_f32 {
                    samples[sample_index].y
                } else {
                    curve_map_extend(x, first_point, end_point, ext_in, ext_out, settings.extend)
                }
            } else {
                let before = samples[sample_index - 1];
                let after = samples[sample_index];
                let span = after.x - before.x;
                let factor = if span > f32::EPSILON {
                    (after.x - x) / span
                } else {
                    0.0
                };
                factor * before.y + (1.0 - factor) * after.y
            };
        if !value.is_finite() {
            return None;
        }
        table.push(value);
    }

    Some(CurveMappingTable {
        values: table,
        ext_in,
        ext_out,
        table_min,
        table_max,
        inverse_range,
        settings,
    })
}

/// Blender `BKE_curve_forward_diff_bezier`, returning the initial value and three differences.
fn bezier_forward_difference(q0: f32, q1: f32, q2: f32, q3: f32) -> [f32; 4] {
    let mut divisor = 31.0_f32;
    let rt0 = q0;
    let rt1 = 3.0_f32 * (q1 - q0) / divisor;
    divisor *= divisor;
    let rt2 = 3.0_f32 * (q0 - 2.0_f32 * q1 + q2) / divisor;
    divisor *= 31.0_f32;
    let rt3 = (q3 - q0 + 3.0_f32 * (q1 - q2)) / divisor;
    [
        rt0,
        rt1 + rt2 + rt3,
        2.0_f32 * rt2 + 6.0_f32 * rt3,
        6.0_f32 * rt3,
    ]
}

/// Blender `BKE_curve_correct_bezpart`: proportionally shortens overlapping X handles.
fn correct_bezier_handles(
    start: Vec2,
    start_handle: Vec2,
    end_handle: Vec2,
    end: Vec2,
) -> (Vec2, Vec2) {
    let start_delta = start - start_handle;
    let end_delta = end - end_handle;
    let span = end.x - start.x;
    let total = start_delta.x.abs() + end_delta.x.abs();
    if total == 0.0 || total <= span {
        return (start_handle, end_handle);
    }
    let factor = span / total;
    (start - start_delta * factor, end - end_delta * factor)
}

fn curve_map_extend(
    x: f32,
    first: Vec2,
    last: Vec2,
    ext_in: Vec2,
    ext_out: Vec2,
    extend: CurveMappingExtend,
) -> f32 {
    if x <= first.x {
        if extend == CurveMappingExtend::Horizontal {
            first.y
        } else if ext_in.x == 0.0 {
            first.y + ext_in.y * 10000.0_f32
        } else {
            first.y + ext_in.y * (x - first.x) / ext_in.x
        }
    } else if extend == CurveMappingExtend::Horizontal {
        last.y
    } else if ext_out.x == 0.0 {
        last.y - ext_out.y * 10000.0_f32
    } else {
        last.y + ext_out.y * (x - last.x) / ext_out.x
    }
}
#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "geometry modifier unit tests")]
mod tests {
    use super::{curve, mask, shrinkwrap};
    use crate::geom::{Mesh, OriginalIndices, UvSphereParams};
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
        let mut modifier =
            modifier_with_target("curve", json!({"deform_axis":"POS_X"}), curve_mesh);
        modifier.runtime.object_matrix = Some(glam::DMat4::IDENTITY.to_cols_array());
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

    #[test]
    fn empty_mask_retains_original_index_layer() {
        let mut source = Mesh::from_positions_and_faces(
            vec![glam::DVec3::ZERO, glam::DVec3::X, glam::DVec3::Y],
            vec![vec![0, 1, 2]],
        )
        .unwrap();
        source.original_indices = Some(OriginalIndices {
            vertices: vec![0; source.vertices.len()],
            source_vertex_count: source.vertices.len(),
        });
        let modifier = crate::model::Modifier {
            id: crate::model::Id::new("empty_mask").unwrap(),
            modifier_type: "mask".to_owned(),
            name: "Mask".to_owned(),
            enabled: true,
            show_viewport: true,
            show_render: true,
            params: serde_json::from_value(json!({"invert_vertex_group": false})).unwrap(),
            binding_data: None,
            runtime: crate::model::ModifierRuntime::default(),
        };
        let result = mask(&source, &modifier, &std::collections::BTreeMap::new()).unwrap();
        assert!(result.vertices.is_empty());
        let indices = result
            .original_indices
            .as_ref()
            .expect("empty layer remains present");
        assert!(indices.vertices.is_empty());
        assert_eq!(indices.source_vertex_count, 3);
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
            show_viewport: true,
            show_render: true,
            params: serde_json::from_value(params).unwrap(),
            binding_data: None,
            runtime: crate::model::ModifierRuntime::default(),
        };
        modifier.runtime.target_mesh = Some(target);
        modifier
    }
}
