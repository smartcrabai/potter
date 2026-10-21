// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   source/blender/modifiers/intern/MOD_build.cc, source/blender/modifiers/intern/MOD_laplaciansmooth.cc, SPDX-FileCopyrightText: 2005 Blender Authors, GPL-2.0-or-later.
//   source/blender/modifiers/intern/MOD_wireframe.cc, source/blender/modifiers/intern/MOD_skin.cc, source/blender/modifiers/intern/MOD_warp.cc, source/blender/blenkernel/intern/mesh_remap.cc, source/blender/modifiers/intern/MOD_mesh_to_volume.cc, source/blender/bmesh/tools/bmesh_wireframe.cc, SPDX-FileCopyrightText: 2023 Blender Authors, GPL-2.0-or-later.
//   source/blender/modifiers/intern/MOD_simpledeform.cc, MOD_cast.cc, MOD_hook.cc, colortools.cc, curve.cc, SPDX-FileCopyrightText: 2005 Blender Authors, GPL-2.0-or-later.

use std::collections::{BTreeMap, HashMap, HashSet};

use glam::{DMat3, DMat4, DQuat, DVec3, Mat4, Vec3};
use serde_json::Value;

use crate::color::curve::{
    CurveMap, CurveMapping, CurvePoint, CurveTone, HandleType, PreparedCurve,
};
use crate::model::{Id, Modifier};

use super::displacement::{VertexGroup, vertex_group};
use super::{
    MergedGeometry, Polygon, bool_param, build_mesh, closest_point_triangle, cyclic_pairs,
    edge_pairs, id_indices, invalid_parameter, loose_edges, number_param, polygons, positions,
    schema_value, string_param, uint_param, usize_as_f64, validated, vec3_param,
};
use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{
        Mesh,
        clnor::{self, NormalCodeOptions},
        color_layers::{self, ColorDomain},
        normals::{corner_angle, corner_normals, face_normal, sharp_edge_flags},
        origindex_values,
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

/// Blender's Screw modifier (`MOD_screw.cc`): the profile edges sweep around the axis through the
/// pivot, one ring per step, and the screw offset rises along the axis.
pub(super) fn screw(
    mesh: &Mesh,
    modifier: &Modifier,
    context: crate::eval::ModifierEvaluationContext<'_>,
    owner_world_matrix: DMat4,
    object_world_matrices: &BTreeMap<Id, DMat4>,
) -> Result<Mesh> {
    // Blender returns an empty template before reading any Screw parameters.
    if mesh.vertices.is_empty() {
        return Ok(mesh.clone());
    }
    validated(mesh)?;
    let axis_index = match schema_value(modifier, "axis")? {
        Value::String(axis) => match axis.as_str() {
            "X" => 0,
            "Y" => 1,
            "Z" => 2,
            _ => return Err(invalid_parameter(modifier, "axis", "X, Y, or Z")),
        },
        _ => return Err(invalid_parameter(modifier, "axis", "X, Y, or Z")),
    };
    let steps_name = if context.purpose == crate::eval::EvaluationPurpose::Render {
        "render_steps"
    } else {
        "steps"
    };
    let steps = screw_count(modifier, steps_name)?;
    let iterations = screw_count(modifier, "iterations")?;
    // Both counts are at most 10000, so the product fits a usize.
    let base_step_count = steps * iterations;
    let angle = screw_float(modifier, "angle")?;
    let input_screw_offset = screw_float(modifier, "screw_offset")?;
    let stretch_u = bool_param(modifier, "use_stretch_u")?;
    let stretch_v = bool_param(modifier, "use_stretch_v")?;
    let merge = bool_param(modifier, "use_merge_vertices")?;
    let merge_threshold = screw_float(modifier, "merge_threshold")?;
    if merge_threshold < 0.0 {
        return Err(invalid_parameter(
            modifier,
            "merge_threshold",
            "a non-negative finite number",
        ));
    }
    let normal_calculate = bool_param(modifier, "use_normal_calculate")?;
    let normal_flip = bool_param(modifier, "use_normal_flip")?;
    let smooth_shade = bool_param(modifier, "use_smooth_shade")?;

    let axis_object = match schema_value(modifier, "object")? {
        Value::Null => None,
        Value::String(name) => {
            let id = Id::new(name.as_str())
                .map_err(|_| invalid_parameter(modifier, "object", "a valid object ID or null"))?;
            let world = object_world_matrices.get(&id).ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::TargetNotFound,
                    "screw axis object was not found",
                    serde_json::json!({"modifier_id": modifier.id, "object": id}),
                )
            })?;
            // Blender's relative axis transform: the axis object expressed in the owner's space.
            let owner =
                Mat4::from_cols_array(&owner_world_matrix.to_cols_array().map(|c| c as f32));
            let target = Mat4::from_cols_array(&world.to_cols_array().map(|c| c as f32));
            let relative = owner.inverse() * target;
            if !relative
                .to_cols_array()
                .iter()
                .all(|component| component.is_finite())
            {
                return Err(invalid_parameter(
                    modifier,
                    "object",
                    "an axis object with a finite relative transform",
                ));
            }
            Some(relative)
        }
        _ => {
            return Err(invalid_parameter(
                modifier,
                "object",
                "an object ID or null",
            ));
        }
    };
    let mut axis = match axis_index {
        0 => Vec3::X,
        1 => Vec3::Y,
        _ => Vec3::Z,
    };
    let pivot = if let Some(relative) = axis_object {
        let transformed = relative.transform_vector3(axis);
        if !transformed.is_finite()
            || !transformed.length_squared().is_finite()
            || transformed.length_squared() == 0.0
        {
            return Err(invalid_parameter(
                modifier,
                "object",
                "an axis object whose transformed axis is non-degenerate",
            ));
        }
        axis = transformed.normalize();
        relative.w_axis.truncate()
    } else {
        Vec3::ZERO
    };
    let mut screw_offset = input_screw_offset;
    if axis_object.is_some() && bool_param(modifier, "use_object_screw_offset")? {
        // The axis object's offset is its pivot's projection onto the axis.
        screw_offset = axis.dot(pivot);
    }
    let iterations_f32 = iterations as f32;
    let scaled_angle = angle * iterations_f32;
    screw_offset *= iterations_f32;
    if !scaled_angle.is_finite() {
        return Err(invalid_parameter(
            modifier,
            "angle",
            "a finite angle after applying iterations",
        ));
    }
    if !screw_offset.is_finite() {
        return Err(invalid_parameter(
            modifier,
            "screw_offset",
            "a finite offset after applying iterations",
        ));
    }
    // Blender closes a full turn only from three steps up, and only when the screw offset and the
    // turn's excess over 2π are within FLT_EPSILON * 100 (float32, MOD_screw.cc); a closed screw
    // then ignores the offset and drops its duplicate ring.
    let closure_tolerance = f32::EPSILON * 100.0;
    let mut ring_count = base_step_count + 1;
    let closed = screw_offset.abs() <= closure_tolerance
        && (scaled_angle.abs() - std::f32::consts::TAU).abs() <= closure_tolerance
        && ring_count > 3;
    if closed {
        ring_count -= 1;
        screw_offset = 0.0;
    }
    let segment_count = if closed { ring_count } else { ring_count - 1 };
    let vertex_count = mesh.vertices.len();
    let point_count = vertex_count
        .checked_mul(ring_count)
        .ok_or_else(|| invalid_parameter(modifier, steps_name, "a fitting generated mesh"))?;
    let source_positions: Vec<Vec3> = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co.as_vec3())
        .collect();
    if source_positions.iter().any(|point| !point.is_finite()) {
        return Err(invalid_parameter(
            modifier,
            "mesh",
            "finite float32 vertex positions",
        ));
    }
    let has_uv_layers = uv_layers::layer_count(mesh) > 0;
    // Each profile vertex's height along the axis: V without stretch, or its share of the range.
    let axial: Vec<f32> = source_positions
        .iter()
        .map(|point| point.dot(axis))
        .collect();
    let (axial_low, axial_high) = axial
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), &value| {
            (low.min(value), high.max(value))
        });
    // U advances one share of the turn per ring; stretch U spreads the turn over the steps.
    let u_scale = if stretch_u {
        1.0_f32 / steps as f32
    } else {
        (1.0_f32 / steps as f32 / iterations as f32) * (scaled_angle / std::f32::consts::TAU)
    };

    let mut profile_edges = edge_pairs(mesh)?;
    let mut edge_lookup = HashMap::with_capacity(profile_edges.len());
    for (edge, [first, second]) in profile_edges.iter().enumerate() {
        edge_lookup.insert((*first.min(second), *first.max(second)), edge);
    }
    let source_polygons = polygons(mesh)?;
    let mut edge_faces: Vec<Option<usize>> = vec![None; profile_edges.len()];
    let mut last_loop: Vec<Option<usize>> = vec![None; vertex_count];
    let mut corner_start = 0;
    for (face_index, (face_vertices, _)) in source_polygons.iter().enumerate() {
        for (corner, &vertex) in face_vertices.iter().enumerate() {
            last_loop[vertex] = Some(corner_start + corner);
            let next = face_vertices[(corner + 1) % face_vertices.len()];
            let edge = *edge_lookup
                .get(&(vertex.min(next), vertex.max(next)))
                .ok_or_else(|| invalid_parameter(modifier, "mesh", "faces with valid edges"))?;
            edge_faces[edge] = Some(face_index);
            // MOD_screw orients each source edge to the last face loop using it.
            profile_edges[edge] = [vertex, next];
        }
        corner_start += face_vertices.len();
    }
    if normal_calculate {
        screw_calculate_normals(
            &mut profile_edges,
            &source_positions,
            axis_index,
            axis_object,
            scaled_angle,
        );
    }

    let mut points: Vec<DVec3> = Vec::new();
    points
        .try_reserve_exact(point_count)
        .map_err(|_| invalid_parameter(modifier, steps_name, "a fitting generated mesh"))?;
    let step_angle = scaled_angle
        / if closed {
            ring_count as f32
        } else {
            (ring_count - 1) as f32
        };
    points.extend(source_positions.iter().map(|point| point.as_dvec3()));
    for step in 1..ring_count {
        let rotation_angle = step_angle * step as f32;
        let angle_cos = rotation_angle.cos();
        let angle_sin = rotation_angle.sin();
        let rotation = if axis_object.is_some() {
            let ico = 1.0_f32 - angle_cos;
            let [axis_x, axis_y, axis_z] = axis.to_array();
            let nsi = [axis_x * angle_sin, axis_y * angle_sin, axis_z * angle_sin];
            let n_00 = (axis_x * axis_x) * ico;
            let n_01 = (axis_x * axis_y) * ico;
            let n_11 = (axis_y * axis_y) * ico;
            let n_02 = (axis_x * axis_z) * ico;
            let n_12 = (axis_y * axis_z) * ico;
            let n_22 = (axis_z * axis_z) * ico;
            [
                [n_00 + angle_cos, n_01 + nsi[2], n_02 - nsi[1]],
                [n_01 - nsi[2], n_11 + angle_cos, n_12 + nsi[0]],
                [n_02 + nsi[1], n_12 - nsi[0], n_22 + angle_cos],
            ]
        } else {
            match axis_index {
                0 => [
                    [1.0, 0.0, 0.0],
                    [0.0, angle_cos, angle_sin],
                    [0.0, -angle_sin, angle_cos],
                ],
                1 => [
                    [angle_cos, 0.0, -angle_sin],
                    [0.0, 1.0, 0.0],
                    [angle_sin, 0.0, angle_cos],
                ],
                _ => [
                    [angle_cos, angle_sin, 0.0],
                    [-angle_sin, angle_cos, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            }
        };
        let offset = if closed {
            0.0
        } else {
            screw_offset * step as f32 / (ring_count - 1) as f32
        };
        for &point in &source_positions {
            let relative = point - pivot;
            let rotated = Vec3::new(
                relative.x * rotation[0][0]
                    + relative.y * rotation[1][0]
                    + rotation[2][0] * relative.z,
                relative.x * rotation[0][1]
                    + relative.y * rotation[1][1]
                    + rotation[2][1] * relative.z,
                relative.x * rotation[0][2]
                    + relative.y * rotation[1][2]
                    + rotation[2][2] * relative.z,
            );
            points.push((pivot + rotated + axis * offset).as_dvec3());
        }
    }

    // One quad per profile edge and segment. Blender's screw loops run against the profile edge's
    // winding; `use_normal_flip` reverses every quad (quad_ord in MOD_screw.cc).
    let face_count = segment_count * profile_edges.len();
    let mut quads: Vec<Polygon> = Vec::with_capacity(face_count);
    // Each generated quad's point indices, kept after the merge consumes `quads`.
    let mut quad_points: Vec<[usize; 4]> = Vec::with_capacity(face_count);
    let mut face_sources: Vec<Option<usize>> = Vec::with_capacity(face_count);
    let mut corner_sources: Vec<Vec<Vec<(usize, f64)>>> = Vec::with_capacity(face_count);
    let mut corner_uv: Vec<Vec<(usize, Option<f64>)>> =
        Vec::with_capacity(if has_uv_layers { face_count } else { 0 });
    for (edge_index, &[first, second]) in profile_edges.iter().enumerate() {
        let source_face = edge_faces[edge_index];
        let material = source_face.map_or(0, |face| source_polygons[face].1);
        for step in 0..segment_count {
            let next_step = (step + 1) % ring_count;
            let mut quad = [
                step * vertex_count + first,
                step * vertex_count + second,
                next_step * vertex_count + second,
                next_step * vertex_count + first,
            ];
            if normal_flip {
                quad.reverse();
            }
            // A corner takes the data of its profile vertex's last source loop, on source faces only.
            let corners: Vec<Vec<(usize, f64)>> = quad
                .iter()
                .map(|&point| {
                    source_face
                        .and_then(|_| last_loop[point % vertex_count])
                        .map(|corner| vec![(corner, 1.0)])
                        .unwrap_or_default()
                })
                .collect();
            if has_uv_layers {
                // Each corner's U advances by the rings between its face's step and its ring.
                corner_uv.push(
                    quad.iter()
                        .map(|&point| {
                            let profile = point % vertex_count;
                            let ring = point / vertex_count;
                            let turns = step + (ring + ring_count - step) % ring_count;
                            let loose_v = source_face.is_none().then(|| {
                                if !stretch_v {
                                    f64::from(axial[profile])
                                } else if axial_high > axial_low {
                                    f64::from(
                                        (axial[profile] - axial_low) / (axial_high - axial_low),
                                    )
                                } else {
                                    0.0
                                }
                            });
                            (turns, loose_v)
                        })
                        .collect(),
                );
            }
            quads.push((quad.to_vec(), material));
            quad_points.push(quad);
            face_sources.push(source_face);
            corner_sources.push(corners);
        }
    }

    // Blender copies the source profile edges at each ring and adds one longitudinal edge per
    // profile vertex and segment. Keep these edges even when a merge removes their only face.
    let mut generated_edges = Vec::with_capacity(
        profile_edges
            .len()
            .saturating_mul(ring_count)
            .saturating_add(vertex_count.saturating_mul(segment_count)),
    );
    for ring in 0..ring_count {
        for &[first, second] in &profile_edges {
            generated_edges.push([ring * vertex_count + first, ring * vertex_count + second]);
        }
    }
    for step in 0..segment_count {
        let next_step = (step + 1) % ring_count;
        for vertex in 0..vertex_count {
            generated_edges.push([
                step * vertex_count + vertex,
                next_step * vertex_count + vertex,
            ]);
        }
    }

    let merged = if merge && screw_offset == 0.0 {
        // On-axis profile vertices snap to the axis; every ring copy merges into its ring-0 vertex.
        let threshold_squared = merge_threshold * merge_threshold;
        let mut destinations = vec![None; point_count];
        for profile in 0..vertex_count {
            let relative = source_positions[profile] - pivot;
            let projection = pivot + axis * relative.dot(axis);
            if (source_positions[profile] - projection).length_squared() <= threshold_squared {
                points[profile] = projection.as_dvec3();
                for ring in 0..ring_count {
                    destinations[ring * vertex_count + profile] = Some(profile);
                }
            }
        }
        super::merge_by_dest_map(&points, quads, generated_edges, &destinations, false)
    } else {
        MergedGeometry::unmerged(points, quads, generated_edges)
    };

    let mut remap = vec![0; point_count];
    for (output, members) in merged.point_members.iter().enumerate() {
        for &member in members {
            remap[member] = output;
        }
    }
    let mut output_face_sources = Vec::with_capacity(merged.polygons.len());
    let mut output_corner_sources = Vec::with_capacity(
        merged
            .polygons
            .iter()
            .map(|(face, _)| face.len())
            .sum::<usize>(),
    );
    let mut output_corner_uv: Vec<Vec<(f64, Option<f64>)>> = Vec::with_capacity(if has_uv_layers {
        merged.polygons.len()
    } else {
        0
    });
    for (output_face, (face, _)) in merged.polygons.iter().enumerate() {
        let original_face = merged.polygon_sources[output_face];
        let source_face = face_sources[original_face];
        output_face_sources.push(source_face);
        let mut face_uv = Vec::with_capacity(if has_uv_layers { face.len() } else { 0 });
        for &output_vertex in face {
            let mut source_corners = Vec::new();
            let mut source_corner_count = 0_usize;
            let mut u_sum = 0.0_f32;
            let mut loose_v_sum = 0.0_f32;
            for (corner, &old_vertex) in quad_points[original_face].iter().enumerate() {
                if remap[old_vertex] != output_vertex {
                    continue;
                }
                source_corner_count += 1;
                source_corners.extend(corner_sources[original_face][corner].iter().copied());
                if has_uv_layers {
                    let (turns, loose_v) = corner_uv[original_face][corner];
                    u_sum += turns as f32 * u_scale;
                    if let Some(v) = loose_v {
                        loose_v_sum += v as f32;
                    }
                }
            }
            if source_corner_count == 0 {
                return Err(invalid_parameter(
                    modifier,
                    "mesh",
                    "valid merged face corners",
                ));
            }
            let source_weight = 1.0 / source_corner_count as f64;
            output_corner_sources.push(
                source_corners
                    .into_iter()
                    .map(|(source, weight)| (source, weight * source_weight))
                    .collect(),
            );
            if has_uv_layers {
                let loose_v = source_face
                    .is_none()
                    .then(|| f64::from(loose_v_sum / source_corner_count as f32));
                face_uv.push((f64::from(u_sum / source_corner_count as f32), loose_v));
            }
        }
        if has_uv_layers {
            output_corner_uv.push(face_uv);
        }
    }

    // Generated edges that no output face uses (profile edges with no face loop, or the longitudinal
    // edges of isolated profile vertices) become loose edges of the result.
    let face_edges: HashSet<(usize, usize)> = merged
        .polygons
        .iter()
        .flat_map(|(face, _)| {
            cyclic_pairs(face).map(|(&first, &second)| (first.min(second), first.max(second)))
        })
        .collect();
    let mut emitted_edges = HashSet::new();
    let loose_output_edges: Vec<[usize; 2]> = merged
        .edges
        .iter()
        .copied()
        .filter(|&[first, second]| {
            let key = (first.min(second), first.max(second));
            !face_edges.contains(&key) && emitted_edges.insert(key)
        })
        .collect();
    let origin = Origin {
        vertices: merged
            .point_members
            .iter()
            .map(|members| {
                members
                    .first()
                    .map(|member| vec![(member % vertex_count, 1.0)])
                    .unwrap_or_default()
            })
            .collect(),
        source_vertices: None,
        faces: output_face_sources,
        corners: Some(output_corner_sources),
    };
    let mut result = build_mesh(
        merged.points,
        merged.polygons,
        loose_output_edges,
        &origin,
        mesh,
    )?;
    let mut colors = color_layers::read(&result);
    if !colors.layers.is_empty() {
        // Blender initializes new Screw CORNER FLOAT_COLOR elements to opaque white.
        let encodings = result
            .attributes
            .get("blender_metadata")
            .and_then(|metadata| metadata.get("color_data_types"))
            .and_then(Value::as_object);
        let mut corner_start = 0;
        for (face, source) in result.faces.iter().zip(&origin.faces) {
            let corner_end = corner_start + face.vertices.len();
            if source.is_none() {
                for layer in &mut colors.layers {
                    let encoding = encodings
                        .and_then(|types| types.get(&layer.name))
                        .and_then(Value::as_str)
                        .unwrap_or("FLOAT_COLOR");
                    if layer.domain == ColorDomain::Corner && encoding == "FLOAT_COLOR" {
                        layer.values[corner_start..corner_end].fill([1.0; 4]);
                    }
                }
            }
            corner_start = corner_end;
        }
        color_layers::write(&mut result, &colors);
    }
    // Blender allocates Screw's edge custom data with CD_SET_DEFAULT and does not interpolate
    // edge attributes, so copied edge flags and creases are cleared on the rebuilt topology.
    result.attributes.remove("sharp");
    result.attributes.remove("blender_edge_flags");
    result.attributes.remove("crease_edge");
    if smooth_shade && origin.faces.iter().any(Option::is_none) {
        let mut flags = crate::geom::normals::smooth_face_flags(&result)?;
        for (flag, source) in flags.iter_mut().zip(&origin.faces) {
            if source.is_none() {
                *flag = true;
            }
        }
        let values: serde_json::Map<String, Value> = result
            .faces
            .iter()
            .zip(flags)
            .map(|(face, smooth)| (format!("f{}", face.id), Value::Bool(smooth)))
            .collect();
        result.attributes.insert(
            "shade_smooth".to_owned(),
            serde_json::json!({"domain": "face", "type": "bool", "values": values}),
        );
    }
    if has_uv_layers {
        // A generated face has no source UV entry, so it starts from zero before its U and V are set.
        let mut layers = uv_layers::read_layers(&result);
        for layer in &mut layers {
            for (face, values) in layer.faces.iter_mut().enumerate() {
                let corners = &output_corner_uv[face];
                if values.is_none() && corners.iter().all(|(_, loose)| loose.is_some()) {
                    *values = Some(vec![[0.0, 0.0]; corners.len()]);
                }
            }
        }
        uv_layers::write_layers(&mut result, &layers);
        super::map_uv_values(&mut result, |face, corner, [u, v]| {
            let (u_offset, loose_v) = output_corner_uv[face][corner];
            let offset = u_offset as f32;
            [
                f64::from(if loose_v.is_some() {
                    offset
                } else {
                    u as f32 + offset
                }),
                loose_v.unwrap_or(v),
            ]
        });
    }
    Ok(result)
}

fn screw_count(modifier: &Modifier, name: &str) -> Result<usize> {
    let count = usize::try_from(uint_param(modifier, name)?)
        .map_err(|_| invalid_parameter(modifier, name, "an integer from 1 through 10000"))?;
    if !(1..=10000).contains(&count) {
        return Err(invalid_parameter(
            modifier,
            name,
            "an integer from 1 through 10000",
        ));
    }
    Ok(count)
}

fn screw_float(modifier: &Modifier, name: &str) -> Result<f32> {
    let value = number_param(modifier, name)? as f32;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid_parameter(modifier, name, "a finite Blender float"))
    }
}

/// One profile vertex's neighbours in the screw's edge chain (`ScrewConnection` in `MOD_screw.cc`).
/// A vertex with more than two profile edges is invalid and breaks its chain.
#[derive(Clone, Copy, Default)]
struct ScrewConnection {
    neighbors: [Option<usize>; 2],
    edges: [Option<usize>; 2],
    invalid: bool,
}

#[derive(Clone, Copy)]
struct ScrewVertexIter {
    vertex: Option<usize>,
    other: Option<usize>,
    edge: Option<usize>,
}

fn screw_vertex_iter(
    connections: &[ScrewConnection],
    vertex: usize,
    direction: usize,
) -> ScrewVertexIter {
    if connections[vertex].invalid {
        return ScrewVertexIter {
            vertex: None,
            other: None,
            edge: None,
        };
    }
    ScrewVertexIter {
        vertex: Some(vertex),
        other: connections[vertex].neighbors[direction],
        edge: connections[vertex].edges[1 - direction],
    }
}

fn screw_vertex_step(iter: &mut ScrewVertexIter, connections: &[ScrewConnection]) {
    let Some(vertex) = iter.vertex else {
        return;
    };
    if connections[vertex].invalid {
        iter.vertex = None;
        iter.edge = None;
        return;
    }
    let links = &connections[vertex];
    if links.neighbors[0] == iter.other {
        iter.other = Some(vertex);
        iter.vertex = links.neighbors[1];
    } else if links.neighbors[1] == iter.other {
        iter.other = Some(vertex);
        iter.vertex = links.neighbors[0];
    }
    if let Some(next) = iter.vertex {
        let next_links = &connections[next];
        let edge_slot = usize::from(next_links.edges[0] == iter.edge);
        iter.edge = next_links.edges[edge_slot];
    } else {
        iter.edge = None;
    }
}

/// Blender's `screw_calculate_normals`: orients each profile chain so its screw faces wind away
/// from the axis, judged at the chain vertex farthest from the axis, and flips for a negative angle.
fn screw_calculate_normals(
    edges: &mut [[usize; 2]],
    positions: &[Vec3],
    axis_index: usize,
    axis_object: Option<Mat4>,
    angle: f32,
) {
    if edges.is_empty() {
        return;
    }
    let coordinates: Vec<Vec3> = positions
        .iter()
        .map(|&position| {
            axis_object.map_or(position, |relative| relative.transform_point3(position))
        })
        .collect();
    let other_axes = match axis_index {
        0 => [1, 2],
        1 => [0, 2],
        _ => [0, 1],
    };
    let mut connections = vec![ScrewConnection::default(); positions.len()];
    for (edge_index, &[first, second]) in edges.iter().enumerate() {
        for (vertex, neighbor) in [(first, second), (second, first)] {
            let connection = &mut connections[vertex];
            if connection.invalid {
                continue;
            }
            if connection.neighbors[0].is_none() {
                connection.neighbors[0] = Some(neighbor);
                connection.edges[0] = Some(edge_index);
            } else if connection.neighbors[1].is_none() {
                connection.neighbors[1] = Some(neighbor);
                connection.edges[1] = Some(edge_index);
            } else {
                connection.neighbors = [None; 2];
                connection.edges = [None; 2];
                connection.invalid = true;
            }
        }
    }
    let distances: Vec<f32> = coordinates
        .iter()
        .map(|point| {
            point[other_axes[0]] * point[other_axes[0]]
                + point[other_axes[1]] * point[other_axes[1]]
        })
        .collect();
    let mut flags = vec![0_u8; positions.len()];
    for start in 0..positions.len() {
        if flags[start] != 0 || connections[start].invalid {
            continue;
        }
        let mut linked = 0;
        let mut best = None;
        let mut best_distance = -1.0_f32;
        let mut closed = false;
        for direction in 0..2 {
            let mut iter = screw_vertex_iter(&connections, start, direction);
            if direction == 1 {
                screw_vertex_step(&mut iter, &connections);
            }
            while let Some(vertex) = iter.vertex {
                if flags[vertex] != 0 {
                    closed = true;
                    break;
                }
                flags[vertex] = 1;
                linked += 1;
                if best_distance <= distances[vertex] {
                    best_distance = distances[vertex];
                    best = Some(vertex);
                }
                screw_vertex_step(&mut iter, &connections);
            }
        }
        if linked <= 1 {
            continue;
        }
        let Some(best) = best else {
            continue;
        };
        let best_links = connections[best];
        let mut flip = false;
        match best_links.neighbors {
            [Some(first), Some(second)] => {
                let center = coordinates[best][axis_index];
                let first_axis = coordinates[first][axis_index];
                let second_axis = coordinates[second][axis_index];
                if first_axis < center && center < second_axis {
                    flip = false;
                } else if first_axis > center && center > second_axis {
                    flip = true;
                } else {
                    let first_delta = (coordinates[first] - coordinates[best]).normalize_or_zero();
                    let second_delta =
                        (coordinates[second] - coordinates[best]).normalize_or_zero();
                    flip = first_delta[axis_index] < second_delta[axis_index];
                }
            }
            [Some(neighbor), None] | [None, Some(neighbor)] => {
                flip = coordinates[neighbor][axis_index] < coordinates[best][axis_index];
            }
            _ => {}
        }
        if angle < 0.0 {
            flip = !flip;
        }
        for direction in usize::from(closed)..2 {
            let mut iter = screw_vertex_iter(&connections, best, direction);
            flags[best] = 1;
            if direction == 1
                && best_links.neighbors[0].is_some()
                && best_links.neighbors[1].is_some()
            {
                flip = !flip;
            }
            while let Some(vertex) = iter.vertex {
                if flags[vertex] == 2 {
                    break;
                }
                flags[vertex] = 2;
                if let Some(edge) = iter.edge {
                    if vertex == edges[edge][0] {
                        if !flip {
                            edges[edge].swap(0, 1);
                        }
                    } else if vertex == edges[edge][1] && flip {
                        edges[edge].swap(0, 1);
                    }
                }
                screw_vertex_step(&mut iter, &connections);
            }
        }
    }
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
    let triangles = target_triangles(target)?;
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

/// Cast and Hook per-vertex weights (`MOD_cast.cc`, `MOD_hook.cc`); `None` when every vertex has full influence.
/// A named group without stored weights is Blender's null `dvert`, which both modifiers skip, so every vertex
/// keeps full influence, as with an absent group. Simple Deform and Curve keep `deform_group_weights`.
fn cast_hook_weights(mesh: &Mesh, modifier: &Modifier) -> Result<Option<Vec<f64>>> {
    let group = vertex_group(mesh, modifier)?;
    if !matches!(group, VertexGroup::Weighted(_)) {
        return Ok(None);
    }
    let invert = bool_param(modifier, "invert_vertex_group")?;
    Ok(Some(
        (0..mesh.vertices.len())
            .map(|index| group.weight(index, invert))
            .collect(),
    ))
}

/// Blender's automatic cuboid half-extent per axis: the larger of the absolute extremes of the points.
/// Blender `cuboid_do` takes the points as the centre plus the local coordinates relative to it.
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
    let weights = cast_hook_weights(mesh, modifier)?;
    let has_radius = radius > f64::from(f32::EPSILON);
    let mut result = mesh.clone();
    if cast_type == "CUBOID" {
        let half = if use_radius_as_size && has_radius {
            DVec3::splat(radius)
        } else if !use_radius_as_size && size > 0.0 {
            DVec3::splat(size)
        } else {
            // Blender `cuboid_do` bounds the centre and the local coordinates relative to it, whatever
            // `use_transform` is; the transform applies only to each vertex's projection.
            let relative = mesh.vertices.iter().map(|vertex| vertex.co - center);
            cast_auto_half_extent(std::iter::once(center).chain(relative))
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
    let start = number_param(modifier, "frame_start")? as f32;
    let duration = (number_param(modifier, "frame_duration")? as f32).max(1.0);
    let factor = ((frame as f32 - start) / duration).clamp(0.0, 1.0);
    let reverse = bool_param(modifier, "use_reverse")?;
    let threshold = if reverse { 1.0 - factor } else { factor };
    let use_random_order = bool_param(modifier, "use_random_order")?;
    let seed = uint_param(modifier, "seed")?;
    let mut face_order: Vec<usize> = (0..mesh.faces.len()).collect();
    if use_random_order {
        randomize_build_order(&mut face_order, seed);
    }
    let face_count = (mesh.faces.len() as f32 * threshold) as usize;
    let edge_count = (mesh.edges.len() as f32 * threshold) as usize;
    let vertex_count = (mesh.vertices.len() as f32 * threshold) as usize;
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
    let origin = Origin::copies(sources.into_iter().map(Some), face_sources);
    let result = build_mesh(points, result_faces, result_edges, &origin, mesh)?;
    if !result.vertices.is_empty() {
        validated(&result)?;
    }
    Ok(result)
}

fn randomize_build_order(indices: &mut [usize], seed: u64) {
    let elem_num = indices.len();
    if elem_num <= 1 {
        return;
    }
    // Mirrors BLI_array_randomize: a seeded BLI_rng_shuffle_array shuffle over the full array.
    let mut state = (u64::from(seed as u32) << 16) | 0x330e;
    for index in (0..elem_num).rev() {
        state = state.wrapping_mul(0x0005_deec_e66d).wrapping_add(0xb) & 0x0000_ffff_ffff_ffff;
        let random = (state >> 17) as u32;
        let other = (random % elem_num as u32) as usize;
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
    mesh.tessellate()
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
    let group_name = string_param(modifier, "vertex_group")?;
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
    let threshold = number_param(modifier, "threshold")?;
    let invert = bool_param(modifier, "invert_vertex_group")?;
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
    let mut retained_sources = Vec::new();
    for (face_index, face) in mesh.faces.iter().enumerate() {
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
            retained_sources.push(Some(face_index));
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
    let sources: Vec<usize> = (0..source_positions.len())
        .filter(|index| keep.contains(index))
        .collect();
    for (output, &source) in sources.iter().enumerate() {
        remap[source] = output;
    }
    let points: Vec<DVec3> = sources
        .iter()
        .map(|&index| source_positions[index])
        .collect();
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
    let origin = Origin::copies(sources.into_iter().map(Some), retained_sources);
    build_mesh(points, faces, edges, &origin, mesh)
}

#[expect(
    clippy::manual_midpoint,
    reason = "matches Blender's float evaluation order for parity"
)]
pub(super) fn wireframe(
    mesh: &Mesh,
    modifier: &Modifier,
    object_material_slots: usize,
) -> Result<Mesh> {
    let thickness = number_param(modifier, "thickness")?;
    let offset = number_param(modifier, "offset")?;
    let use_boundary = bool_param(modifier, "use_boundary")?;
    let use_replace = bool_param(modifier, "use_replace")?;
    let use_even_offset = bool_param(modifier, "use_even_offset")?;
    let use_relative_offset = bool_param(modifier, "use_relative_offset")?;
    let use_crease = bool_param(modifier, "use_crease")?;
    let crease_weight = number_param(modifier, "crease_weight")?;
    let material_offset = schema_value(modifier, "material_offset")?
        .as_i64()
        .ok_or_else(|| invalid_parameter(modifier, "material_offset", "an integer"))?;
    let vertex_group = string_param(modifier, "vertex_group")?;
    let invert_vertex_group = bool_param(modifier, "invert_vertex_group")?;
    let thickness_vertex_group = number_param(modifier, "thickness_vertex_group")?.clamp(0.0, 1.0);
    let (has_deform_data, group_weights) = wireframe_group_data(mesh, modifier, vertex_group)?;
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

    let mut tagged_vertices = vec![false; mesh.vertices.len()];
    for indices in &face_indices {
        for index in indices {
            tagged_vertices[*index] = true;
        }
    }

    let mut relative_factors = vec![1.0; mesh.vertices.len()];
    if use_relative_offset {
        let mut edge_length_sums = vec![0.0_f32; mesh.vertices.len()];
        let mut edge_counts = vec![0_usize; mesh.vertices.len()];
        for edge in &mesh.edges {
            let first = vertex_indices[&edge.vertices[0]];
            let second = vertex_indices[&edge.vertices[1]];
            let dx = (source_positions[first].x - source_positions[second].x) as f32;
            let dy = (source_positions[first].y - source_positions[second].y) as f32;
            let dz = (source_positions[first].z - source_positions[second].z) as f32;
            let length = (dx * dx + dy * dy + dz * dz).sqrt();
            edge_counts[first] += 1;
            edge_counts[second] += 1;
            if tagged_vertices[second] {
                edge_length_sums[first] += length;
            }
            if tagged_vertices[first] {
                edge_length_sums[second] += length;
            }
        }
        for index in 0..mesh.vertices.len() {
            relative_factors[index] = if edge_counts[index] > 0 {
                f64::from(edge_length_sums[index] / edge_counts[index] as f32)
            } else {
                0.0
            };
        }
    }
    if has_deform_data && let Some(weights) = &group_weights {
        for (index, tagged) in tagged_vertices.iter().enumerate() {
            if !tagged {
                continue;
            }
            let mut factor = weights
                .get(&mesh.vertices[index].id)
                .copied()
                .unwrap_or(0.0);
            if invert_vertex_group {
                factor = 1.0 - factor;
            }
            if thickness_vertex_group > 0.0 {
                factor = thickness_vertex_group + factor * (1.0 - thickness_vertex_group);
            }
            relative_factors[index] *= factor;
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

    let material_max = object_material_slots
        .saturating_sub(1)
        .min(i64::MAX as usize) as i64;
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
        let material_index = if material_offset == 0 {
            source_face.material_index
        } else {
            i64::from(source_face.material_index)
                .saturating_add(material_offset)
                .clamp(0, material_max) as u32
        };
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
    );
    let output_loose_edges = if use_replace {
        Vec::new()
    } else {
        loose_edges(mesh)?
    };
    let mut result = build_mesh(coords, faces, output_loose_edges, &origin, mesh)?;
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

/// Whether the mesh carries any vertex-group weights, and the named group's weights when it exists.
/// A missing or empty group name yields no per-vertex weights.
fn wireframe_group_data(
    mesh: &Mesh,
    modifier: &Modifier,
    group_name: &str,
) -> Result<(bool, Option<HashMap<u32, f64>>)> {
    let Some(groups) = mesh
        .attributes
        .get("vertex_groups")
        .and_then(Value::as_object)
    else {
        return Ok((false, None));
    };
    let has_deform_data = groups
        .values()
        .any(|group| group.as_object().is_some_and(|weights| !weights.is_empty()));
    let Some(group_values) = (!group_name.is_empty())
        .then(|| groups.get(group_name))
        .flatten()
    else {
        return Ok((has_deform_data, None));
    };
    let values = group_values
        .as_object()
        .ok_or_else(|| invalid_parameter(modifier, "vertex_group", "a vertex group"))?;
    let mut weights = HashMap::with_capacity(values.len());
    for (key, value) in values {
        let Some(id) = key
            .strip_prefix('v')
            .and_then(|suffix| suffix.parse::<u32>().ok())
        else {
            continue;
        };
        let Some(weight) = value
            .as_f64()
            .filter(|weight| weight.is_finite() && (0.0..=1.0).contains(weight))
        else {
            return Err(invalid_parameter(
                modifier,
                "vertex_group",
                "finite weights in [0, 1]",
            ));
        };
        weights.insert(id, weight);
    }
    Ok((has_deform_data, Some(weights)))
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
        Some(FalloffCurve::from_modifier(modifier)?)
    } else {
        None
    };
    let weights = cast_hook_weights(mesh, modifier)?;
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
    let use_falloff = falloff != HookFalloff::NoFalloff && radius > 0.0;
    // Blender `hook_co_apply`: the falloff is measured from the current coordinate, then scaled by the group weight.
    let hook_co = |co: DVec3, index: usize| -> Option<DVec3> {
        let offset = co - center;
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
        (fac != 0.0).then(|| co.lerp(hook_matrix.transform_point3(co), fac))
    };
    let hook_index = |result: &mut Mesh, index: usize| {
        let co = result.vertices[index].co;
        if let Some(hooked) = hook_co(co, index) {
            result.vertices[index].co = hooked;
        }
    };
    let mut result = mesh.clone();
    let vertex_count = mesh.vertices.len();
    if !listed {
        for index in 0..vertex_count {
            hook_index(&mut result, index);
        }
    } else if let Some(original) = origindex_values(mesh) {
        // Blender `MOD_hook.cc` ORIGINDEX branch: a vertex is hooked when its original index is
        // listed. Generated vertices carry `ORIGINDEX_NONE`, which no listed index matches.
        // Discrepancy: the fetched source (v5.2.2 `hook_index_array_to_bitmap`) sets bit `i` (the
        // list position), not bit `j` (the listed index), so it would hook the origin indices
        // 0..count. The Blender 5.2.2 binary hooks by listed membership (probed: listed [5] moves the
        // copies of vertex 5). This follows the binary, the observed ground truth.
        let listed_originals: HashSet<usize> = indices.iter().copied().collect();
        for index in 0..vertex_count {
            let source = original
                .get(index)
                .copied()
                .and_then(|source| usize::try_from(source).ok());
            if source.is_some_and(|source| listed_originals.contains(&source)) {
                hook_index(&mut result, index);
            }
        }
    } else {
        // Blender's list-value branch (no CD_ORIGINDEX): each listed value below the vertex count hooks
        // that vertex in list order, so a repeated value hooks it again.
        for &index in &indices {
            if index < vertex_count {
                hook_index(&mut result, index);
            }
        }
    }
    validated(&result)?;
    Ok(result)
}

/// Blender's falloff shape for a squared distance from the hook centre (1 at the centre, 0 outside the radius).
fn hook_falloff(
    falloff: HookFalloff,
    distance_sq: f64,
    radius: f64,
    curve: Option<&FalloffCurve>,
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

/// A Hook or Warp `falloff_curve` with its `falloff_curve_settings`: Blender's `CurveMapping` curve 0 from
/// `color::curve`, evaluated as `BKE_curvemapping_evaluateF` (the curve table, then Y clipping when enabled).
pub(super) struct FalloffCurve {
    curve: PreparedCurve,
    clip_y: Option<[f32; 2]>,
}

/// The `falloff_curve_settings` values that shape the falloff table and its clipping.
struct FalloffSettings {
    use_clip: bool,
    clip: [f64; 4],
    extend_extrapolate: bool,
}

/// The identity line through `(0, 0)` and `(1, 1)` with AUTO handles, for the curves a falloff does not read.
fn unit_curve() -> CurveMap {
    CurveMap {
        points: vec![
            CurvePoint {
                location: [0.0, 0.0],
                handle: HandleType::Auto,
            },
            CurvePoint {
                location: [1.0, 1.0],
                handle: HandleType::Auto,
            },
        ],
    }
}

/// Reads `falloff_curve_settings`. An absent object keeps Blender's defaults: clip on over `[0, 1]`, extrapolated.
#[expect(
    clippy::cast_possible_truncation,
    reason = "Blender stores CurveMapping clip bounds as float"
)]
fn falloff_settings(value: Option<&Value>) -> Option<FalloffSettings> {
    let Some(value) = value else {
        return Some(FalloffSettings {
            use_clip: true,
            clip: [0.0, 0.0, 1.0, 1.0],
            extend_extrapolate: true,
        });
    };
    let settings = value.as_object()?;
    let use_clip = settings
        .get("use_clip")
        .map_or(Some(true), Value::as_bool)?;
    let clip = if let Some(value) = settings.get("clip_bounds") {
        let [min_x, min_y, max_x, max_y] = value.as_array()?.as_slice() else {
            return None;
        };
        let bound = |value: &Value| {
            let value = value.as_f64()?;
            if !value.is_finite() || !(-100.0..=100.0).contains(&value) {
                return None;
            }
            let value = value as f32;
            value.is_finite().then_some(f64::from(value))
        };
        [bound(min_x)?, bound(min_y)?, bound(max_x)?, bound(max_y)?]
    } else {
        [0.0, 0.0, 1.0, 1.0]
    };
    let extend_extrapolate = match settings.get("extend") {
        None => true,
        Some(value) => match value.as_str()? {
            "HORIZONTAL" => false,
            "EXTRAPOLATED" => true,
            _ => return None,
        },
    };
    Some(FalloffSettings {
        use_clip,
        clip,
        extend_extrapolate,
    })
}

/// Reads the `falloff_curve` control points: `[x, y]` or `[x, y, handle]`, with `AUTO`, `AUTO_CLAMPED` or `VECTOR`.
#[expect(
    clippy::cast_possible_truncation,
    reason = "Blender stores CurveMapping point coordinates as float"
)]
fn falloff_points(value: &Value) -> Option<Vec<CurvePoint>> {
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
            let location = [x as f32, y as f32];
            if !location.iter().all(|component| component.is_finite()) {
                return None;
            }
            let handle = match coordinates.get(2) {
                None => HandleType::Auto,
                Some(value) => match value.as_str() {
                    Some("AUTO") => HandleType::Auto,
                    Some("AUTO_CLAMPED") => HandleType::AutoClamped,
                    Some("VECTOR") => HandleType::Vector,
                    _ => return None,
                },
            };
            Some(CurvePoint { location, handle })
        })
        .collect()
}

impl FalloffCurve {
    /// Reads a modifier's `falloff_curve` and `falloff_curve_settings`, checked by `CurveMapping::validate`.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "Blender evaluates CurveMapping clip bounds as float"
    )]
    pub(super) fn from_modifier(modifier: &Modifier) -> Result<Self> {
        let invalid_points = || {
            invalid_parameter(
                modifier,
                "falloff_curve",
                "at least two finite float32 [x, y] points with an optional AUTO, AUTO_CLAMPED, or VECTOR handle",
            )
        };
        let points =
            falloff_points(schema_value(modifier, "falloff_curve")?).ok_or_else(invalid_points)?;
        let settings =
            falloff_settings(modifier.params.get("falloff_curve_settings")).ok_or_else(|| {
                invalid_parameter(
                    modifier,
                    "falloff_curve_settings",
                    "use_clip, finite clip_bounds, and HORIZONTAL or EXTRAPOLATED extend",
                )
            })?;
        let mapping = CurveMapping {
            tone: CurveTone::Standard,
            extend_extrapolate: settings.extend_extrapolate,
            use_clip: settings.use_clip,
            clip: settings.clip,
            black: [0.0; 3],
            white: [1.0; 3],
            curves: [
                CurveMap { points },
                unit_curve(),
                unit_curve(),
                unit_curve(),
            ],
        };
        // Blender evaluates the control points in their stored order and does not sort them, so `x` need
        // not increase. The table domain is the clip range widened by the points, and it must be non-empty.
        let clip_min_x = settings.clip[0] as f32;
        let clip_max_x = settings.clip[2] as f32;
        let domain_min = mapping.curves[0]
            .points
            .iter()
            .fold(clip_min_x, |min, point| min.min(point.location[0]));
        let domain_max = mapping.curves[0]
            .points
            .iter()
            .fold(clip_max_x, |max, point| max.max(point.location[0]));
        if settings.clip[0] > settings.clip[2]
            || settings.clip[1] > settings.clip[3]
            || domain_max - domain_min <= 0.0
        {
            return Err(invalid_points());
        }
        Ok(Self {
            curve: mapping.prepare(),
            clip_y: settings
                .use_clip
                .then_some([settings.clip[1] as f32, settings.clip[3] as f32]),
        })
    }

    /// Blender's `BKE_curvemapping_evaluateF` for curve 0: the table at `value`, then Y clipping when enabled.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "Blender evaluates the curve input as float"
    )]
    pub(super) fn evaluate(&self, value: f64) -> f64 {
        let result = self.curve.evaluate(0, value as f32);
        f64::from(
            self.clip_y
                .map_or(result, |[min_y, max_y]| result.clamp(min_y, max_y)),
        )
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "geometry modifier unit tests")]
mod tests {
    use super::{build, curve, mask, shrinkwrap};
    use crate::geom::{
        BoxParams, Mesh, UvSphereParams,
        clnor::{self, CornerNormalCode, NormalCodeOptions, SetterInput},
        origindex_values, set_origindex,
    };
    use serde_json::json;

    #[test]
    fn shrinkwrap_nearest_surface_projects_vertices_to_sphere_radius() {
        let source = Mesh::uv_sphere(UvSphereParams {
            segments: 8,
            ring_count: 4,
            radius: 2.0,
        })
        .unwrap();
        // The 32 x 16 unit sphere is the `uv_sphere_default` oracle, whose largest radius error in
        // Blender's f32 output is 1.2177e-6 (`primitive_uv_oracle.json`); the bound below covers it.
        let target = Mesh::uv_sphere(UvSphereParams {
            segments: 32,
            ring_count: 16,
            radius: 1.0,
        })
        .unwrap();
        let modifier = modifier_with_target(
            "shrinkwrap",
            json!({"wrap_method":"NEAREST_VERTEX"}),
            target,
        );
        let result = shrinkwrap(&source, &modifier).unwrap();
        // Blender stores UV sphere vertices as f32: its own radius-1 sphere output is off by up to
        // 1.22e-6 (`tests/fixtures/primitive_uvs/primitive_uv_oracle.json`, `uv_sphere_default`).
        assert!(
            result
                .vertices
                .iter()
                .all(|vertex| (vertex.co.length() - 1.0).abs() < 2.0e-6)
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
    fn build_subset_compacts_nonidentity_custom_normal_loop_order() {
        let mut source = Mesh::box_mesh(BoxParams::default()).unwrap();
        let corner_count: usize = source.faces.iter().map(|face| face.vertices.len()).sum();
        let codes: Vec<CornerNormalCode> = (0..corner_count)
            .map(|corner| CornerNormalCode::Authored([i16::try_from(corner + 1).unwrap(), -1_000]))
            .collect();
        let mut loop_order: Vec<usize> = (0..corner_count).collect();
        loop_order[..4].rotate_left(1);
        clnor::write_codes(
            &mut source,
            Some(&codes),
            NormalCodeOptions {
                corner_order: Some(&loop_order),
                setter_input: SetterInput::AlreadyNormalized,
            },
        )
        .unwrap();

        let modifier = modifier_with_target(
            "build",
            json!({
                "frame_start": 0.0,
                "frame_duration": 2.0,
                "use_reverse": false,
                "use_random_order": false,
                "seed": 0
            }),
            Mesh::new(),
        );
        let output = build(&source, &modifier, 0.5).unwrap();

        assert_eq!(output.faces.len(), 1);
        assert_eq!(
            clnor::corner_order(&output).unwrap(),
            Some(vec![1, 2, 3, 0]),
            "the subset basis is compacted from its source loop order"
        );
        assert_eq!(
            clnor::corner_codes(&output).unwrap(),
            Some(codes[..4].to_vec()),
            "the subset retains the source codes for its copied loops"
        );
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
    fn empty_mask_retains_origindex_carrier() {
        let mut source = Mesh::from_positions_and_faces(
            vec![glam::DVec3::ZERO, glam::DVec3::X, glam::DVec3::Y],
            vec![vec![0, 1, 2]],
        )
        .unwrap();
        set_origindex(&mut source, &[0, 1, 2]);
        source.attributes.insert(
            "vertex_groups".to_owned(),
            json!({"Empty": {"v0": 0.0, "v1": 0.0, "v2": 0.0}}),
        );
        let modifier = crate::model::Modifier {
            id: crate::model::Id::new("empty_mask").unwrap(),
            modifier_type: "mask".to_owned(),
            name: "Mask".to_owned(),
            enabled: true,
            show_viewport: true,
            show_render: true,
            params: serde_json::from_value(json!({
                "vertex_group": "Empty",
                "threshold": 0.5,
                "invert_vertex_group": false
            }))
            .unwrap(),
            binding_data: None,
            runtime: crate::model::ModifierRuntime::default(),
        };
        let result = mask(&source, &modifier).unwrap();
        assert_eq!(result.vertices.len(), 0);
        assert_eq!(origindex_values(&result), Some(Vec::new()));
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
            baked_ocean_cache: None,
            runtime: crate::model::ModifierRuntime::default(),
        };
        modifier.runtime.target_mesh = Some(target);
        modifier
    }
}
