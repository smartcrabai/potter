// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   source/blender/modifiers/intern/MOD_meshdeform.cc, SPDX-FileCopyrightText: 2005 Blender Authors, GPL-2.0-or-later.
//   source/blender/modifiers/intern/MOD_laplaciandeform.cc, SPDX-FileCopyrightText: 2013 Blender Authors, GPL-2.0-or-later.
//   source/blender/modifiers/intern/MOD_surfacedeform.cc, SPDX-FileCopyrightText: 2017 Blender Authors, GPL-2.0-or-later.

//! Bound surface, cage, and Laplacian deformation modifiers.

use std::collections::{BTreeMap, HashMap};
use std::f64::consts::{FRAC_PI_2, PI};

use glam::{DMat4, DVec2, DVec3};
use serde_json::{Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::Mesh,
    model::Modifier,
};

use super::{
    bool_param, id_indices as index_positions, invalid_parameter, number_param, uint_param,
    validated,
};

/// Evaluates a bound Surface Deform from the bind-time polygon bindings; unbound leaves the mesh unchanged.
pub(super) fn surface_deform(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    let Some(bind_data) = modifier.binding_data.as_ref() else {
        return Ok(mesh.clone());
    };
    if !is_surface_bind(bind_data) {
        return Err(invalid_parameter(
            modifier,
            "binding_data",
            "a Surface Deform bind",
        ));
    }
    native_surface_deform(mesh, modifier, bind_data)
}

fn is_surface_bind(bind_data: &Value) -> bool {
    bind_data.as_object().is_some_and(|object| {
        matches!(
            object.get("format").and_then(Value::as_str),
            Some("blender_native_bind_v1" | "blender_surface_bind_v1")
        ) && object.get("type").and_then(Value::as_str) == Some("surface_deform")
    })
}

fn is_native_bind(bind_data: &Value, modifier_type: &str) -> bool {
    bind_data.as_object().is_some_and(|object| {
        object.get("format").and_then(Value::as_str) == Some("blender_native_bind_v1")
            && object.get("type").and_then(Value::as_str) == Some(modifier_type)
    })
}

fn native_surface_target_warning(
    modifier: &Modifier,
    feature_id: &str,
    message: &str,
    expected_vertices: Option<usize>,
    actual_vertices: usize,
    expected_polygons: Option<usize>,
    actual_polygons: usize,
) -> PotError {
    PotError::with_details(
        ErrorCode::EvaluationFailed,
        message,
        json!({
            "modifier_id": modifier.id,
            "modifier_warning": true,
            "feature_id": feature_id,
            "expected_target_vertices": expected_vertices,
            "actual_target_vertices": actual_vertices,
            "expected_target_polygons": expected_polygons,
            "actual_target_polygons": actual_polygons,
        }),
    )
}

fn native_surface_deform(mesh: &Mesh, modifier: &Modifier, bind_data: &Value) -> Result<Mesh> {
    let mut target: Mesh = serde_json::from_value(
        modifier
            .runtime
            .target_mesh
            .clone()
            .ok_or_else(|| invalid_parameter(modifier, "object", "a mesh object operand"))?,
    )
    .map_err(|_| invalid_parameter(modifier, "object", "a mesh object operand"))?;
    let bind_object = bind_data
        .as_object()
        .ok_or_else(|| invalid_parameter(modifier, "binding_data", "native Surface Deform data"))?;
    if bind_object
        .get("mesh_verts_num")
        .and_then(Value::as_u64)
        .is_some_and(|count| usize::try_from(count).ok() != Some(mesh.vertices.len()))
    {
        return Err(invalid_parameter(
            modifier,
            "binding_data",
            "a source vertex count matching the native bind",
        ));
    }
    let expected_vertices = bind_object
        .get("target_verts_num")
        .and_then(Value::as_u64)
        .and_then(|count| usize::try_from(count).ok());
    let expected_polygons = bind_object
        .get("target_polys_num")
        .and_then(Value::as_u64)
        .and_then(|count| usize::try_from(count).ok());
    if expected_vertices.is_some_and(|count| count != target.vertices.len())
        || expected_polygons.is_some_and(|count| count != target.faces.len())
    {
        return Err(native_surface_target_warning(
            modifier,
            "modifier.surface_deform.target_topology_changed",
            "Surface Deform target vertex or polygon count changed since bind; modifier was skipped",
            expected_vertices,
            target.vertices.len(),
            expected_polygons,
            target.faces.len(),
        ));
    }
    if bind_object.contains_key("target_vertex_order_error") {
        return Err(native_surface_target_warning(
            modifier,
            "modifier.surface_deform.target_order_unavailable",
            "Surface Deform target vertex order could not be matched to the native bind; modifier was skipped",
            expected_vertices,
            target.vertices.len(),
            expected_polygons,
            target.faces.len(),
        ));
    }
    if let Some(order) = bind_object
        .get("target_vertex_order")
        .and_then(Value::as_array)
    {
        if order.len() != target.vertices.len() {
            return Err(invalid_parameter(
                modifier,
                "binding_data",
                "a complete target vertex order matching the evaluated target",
            ));
        }
        let mut ordered_vertices = Vec::with_capacity(order.len());
        let mut seen = vec![false; order.len()];
        for value in order {
            let index = value
                .as_u64()
                .and_then(|index| usize::try_from(index).ok())
                .filter(|index| *index < target.vertices.len())
                .ok_or_else(|| {
                    invalid_parameter(
                        modifier,
                        "binding_data",
                        "valid target vertex order indices",
                    )
                })?;
            if seen[index] {
                return Err(invalid_parameter(
                    modifier,
                    "binding_data",
                    "a one-to-one target vertex order",
                ));
            }
            seen[index] = true;
            ordered_vertices.push(target.vertices[index].clone());
        }
        target.vertices = ordered_vertices;
    }
    if bind_object.get("mat").is_some() {
        let bind_matrix = native_matrix(bind_object, modifier)?;
        let current_to_subject = modifier
            .runtime
            .target_to_subject
            .map_or(DMat4::IDENTITY, |matrix| DMat4::from_cols_array(&matrix));
        let subject_to_current = current_to_subject.inverse();
        if !subject_to_current.is_finite() {
            return Err(invalid_parameter(
                modifier,
                "binding_data",
                "an invertible target-to-owner transform",
            ));
        }
        let current_to_bind = bind_matrix * subject_to_current;
        for vertex in &mut target.vertices {
            vertex.co = current_to_bind.transform_point3(vertex.co);
        }
    }
    let bindings = bind_object
        .get("bind_verts")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            invalid_parameter(modifier, "binding_data", "native Surface Deform binds")
        })?;
    let strength = bind_object.get("strength").map_or_else(
        || number_param(modifier, "strength"),
        |value| finite_number(value, modifier, "binding_data"),
    )?;
    if bind_object
        .get("bind_verts_num")
        .and_then(Value::as_u64)
        .is_some_and(|count| usize::try_from(count).ok() != Some(bindings.len()))
    {
        return Err(invalid_parameter(
            modifier,
            "binding_data",
            "a native bind-vertex count matching its payload",
        ));
    }
    if !(-100.0..=100.0).contains(&strength) {
        return Err(invalid_parameter(
            modifier,
            "strength",
            "a number in [-100, 100]",
        ));
    }
    let group = group_weights(mesh, modifier)?;
    let invert = bool_param(modifier, "invert_vertex_group")?;
    let mut result = mesh.clone();
    for binding in bindings {
        let vertex_index = binding
            .get("vertex_idx")
            .and_then(Value::as_u64)
            .and_then(|index| usize::try_from(index).ok())
            .filter(|index| *index < result.vertices.len())
            .ok_or_else(|| {
                invalid_parameter(modifier, "binding_data", "valid bound source indices")
            })?;
        let records = binding
            .get("binds")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                invalid_parameter(modifier, "binding_data", "native per-vertex binds")
            })?;
        let group_weight = group.as_ref().map_or(1.0, |weights| {
            if invert {
                1.0 - weights[vertex_index]
            } else {
                weights[vertex_index]
            }
        });
        if records.is_empty() || group_weight == 0.0 {
            continue;
        }
        let mut offset = DVec3::ZERO;
        for record in records {
            let indices = record
                .get("vert_inds")
                .and_then(Value::as_array)
                .filter(|indices| indices.len() >= 3)
                .ok_or_else(|| {
                    invalid_parameter(modifier, "binding_data", "surface bind indices")
                })?;
            let verts_num = record
                .get("verts_num")
                .and_then(Value::as_u64)
                .and_then(|count| usize::try_from(count).ok())
                .filter(|count| *count == indices.len())
                .ok_or_else(|| {
                    invalid_parameter(modifier, "binding_data", "surface bind vertex count")
                })?;
            let weights = record
                .get("vert_weights")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    invalid_parameter(modifier, "binding_data", "surface bind weights")
                })?;
            let mode = record
                .get("mode")
                .and_then(Value::as_i64)
                .ok_or_else(|| invalid_parameter(modifier, "binding_data", "surface bind mode"))?;
            let normal = native_polygon_normal(&target, indices, modifier)?;
            let mut interpolated = DVec3::ZERO;
            match mode {
                0 => {
                    if weights.len() != 3 {
                        return Err(invalid_parameter(
                            modifier,
                            "binding_data",
                            "three corner-triangle weights",
                        ));
                    }
                    for corner in 0..3 {
                        let position = native_target_position(&target, &indices[corner], modifier)?;
                        interpolated +=
                            position * finite_number(&weights[corner], modifier, "binding_data")?;
                    }
                }
                1 => {
                    if weights.len() != verts_num {
                        return Err(invalid_parameter(
                            modifier,
                            "binding_data",
                            "one ngon weight per bound target vertex",
                        ));
                    }
                    for (index, weight) in indices.iter().zip(weights) {
                        interpolated += native_target_position(&target, index, modifier)?
                            * finite_number(weight, modifier, "binding_data")?;
                    }
                }
                2 => {
                    if weights.len() != 3 {
                        return Err(invalid_parameter(
                            modifier,
                            "binding_data",
                            "three centroid-mode weights",
                        ));
                    }
                    let first = native_target_position(&target, &indices[0], modifier)?;
                    let second = native_target_position(&target, &indices[1], modifier)?;
                    let mut centroid = DVec3::ZERO;
                    for index in indices {
                        centroid += native_target_position(&target, index, modifier)?;
                    }
                    centroid /= verts_num as f64;
                    interpolated += first * finite_number(&weights[0], modifier, "binding_data")?;
                    interpolated += second * finite_number(&weights[1], modifier, "binding_data")?;
                    interpolated +=
                        centroid * finite_number(&weights[2], modifier, "binding_data")?;
                }
                _ => {
                    return Err(invalid_parameter(
                        modifier,
                        "binding_data",
                        "a supported native Surface Deform bind mode",
                    ));
                }
            }
            let normal_distance = record.get("normal_dist").ok_or_else(|| {
                invalid_parameter(modifier, "binding_data", "surface normal offset")
            })?;
            let influence = record.get("influence").ok_or_else(|| {
                invalid_parameter(modifier, "binding_data", "surface bind influence")
            })?;
            let point =
                interpolated + normal * finite_number(normal_distance, modifier, "binding_data")?;
            offset += point * finite_number(influence, modifier, "binding_data")?;
        }
        let vertex = &mut result.vertices[vertex_index];
        vertex.co += (offset - vertex.co) * (strength * group_weight);
    }
    validated(&result)?;
    Ok(result)
}

fn native_target_position(target: &Mesh, index: &Value, modifier: &Modifier) -> Result<DVec3> {
    let index = index
        .as_u64()
        .and_then(|index| usize::try_from(index).ok())
        .ok_or_else(|| {
            invalid_parameter(modifier, "binding_data", "valid target vertex indices")
        })?;
    target
        .vertices
        .get(index)
        .map(|vertex| vertex.co)
        .ok_or_else(|| {
            invalid_parameter(
                modifier,
                "binding_data",
                "indices in the current target mesh",
            )
        })
}

fn native_polygon_normal(target: &Mesh, indices: &[Value], modifier: &Modifier) -> Result<DVec3> {
    let mut normal = DVec3::ZERO;
    for index in 0..indices.len() {
        let current = native_target_position(target, &indices[index], modifier)?;
        let next = native_target_position(target, &indices[(index + 1) % indices.len()], modifier)?;
        normal += current.cross(next);
    }
    normal.try_normalize().ok_or_else(|| {
        invalid_parameter(
            modifier,
            "binding_data",
            "non-degenerate native Surface Deform target polygons",
        )
    })
}

/// Evaluates a bound Mesh Deform cage. Each cage vertex contributes its displacement measured in
/// the bind frame, applied as an offset through the inverse linear part of the current cage
/// transform, as Blender's `meshdeformModifier_do` does.
pub(super) fn mesh_deform(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    let Some(bind_data) = modifier.binding_data.as_ref() else {
        return Ok(mesh.clone());
    };
    if is_native_bind(bind_data, "mesh_deform") {
        return native_mesh_deform(mesh, modifier, bind_data);
    }
    let bind_data = bind_data.as_object().ok_or_else(|| {
        invalid_parameter(modifier, "binding_data", "stored harmonic cage weights")
    })?;
    let bind_weights = bind_data
        .get("weights")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid_parameter(modifier, "binding_data", "cage-coordinate rows"))?;
    let bound_cage: Mesh =
        serde_json::from_value(bind_data.get("cage_mesh").cloned().ok_or_else(|| {
            invalid_parameter(modifier, "binding_data", "the bind-time owner-space cage mesh")
        })?)
        .map_err(|_| invalid_parameter(modifier, "binding_data", "a valid bind-time cage mesh"))?;
    let bound_cage_local: Mesh = serde_json::from_value(
        bind_data
            .get("cage_local_mesh")
            .cloned()
            .ok_or_else(|| invalid_parameter(modifier, "binding_data", "the bind-time cage mesh"))?,
    )
    .map_err(|_| invalid_parameter(modifier, "binding_data", "a valid bind-time cage mesh"))?;
    let bind_matrix = native_matrix(bind_data, modifier)?;
    let cage_owner: Mesh = serde_json::from_value(
        modifier
            .runtime
            .target_mesh
            .clone()
            .ok_or_else(|| invalid_parameter(modifier, "object", "a mesh cage object operand"))?,
    )
    .map_err(|_| invalid_parameter(modifier, "object", "a mesh cage object operand"))?;
    let target_to_subject = modifier
        .runtime
        .target_to_subject
        .map_or(DMat4::IDENTITY, |matrix| DMat4::from_cols_array(&matrix));
    let subject_to_target = target_to_subject.inverse();
    if !subject_to_target.is_finite() {
        return Err(invalid_parameter(
            modifier,
            "object",
            "an invertible cage transform",
        ));
    }
    let mut cage = cage_owner.clone();
    for vertex in &mut cage.vertices {
        vertex.co = subject_to_target.transform_point3(vertex.co);
    }
    if !same_mesh_topology(&bound_cage_local, &cage) {
        return Err(invalid_parameter(
            modifier,
            "bind_data",
            "binding rows and cage topology matching the bound meshes",
        ));
    }
    let inverse_deform_matrix = (bind_matrix * subject_to_target).inverse();
    if !inverse_deform_matrix.is_finite() {
        return Err(invalid_parameter(
            modifier,
            "object",
            "an invertible cage transform",
        ));
    }
    let precision = match bind_data.get("precision") {
        Some(value) => value.as_u64().ok_or_else(|| {
            invalid_parameter(modifier, "binding_data", "a valid bound grid precision")
        })?,
        None => uint_param(modifier, "precision")?,
    };
    if !(2..=10).contains(&precision) {
        return Err(invalid_parameter(
            modifier,
            "binding_data",
            "a bound grid precision from 2 to 10",
        ));
    }
    let precision = u32::try_from(precision)
        .map_err(|_| invalid_parameter(modifier, "binding_data", "a bound grid precision"))?;
    let dynamic = match bind_data.get("dynamic_bind").and_then(Value::as_bool) {
        Some(dynamic) => dynamic,
        None => bool_param(modifier, "use_dynamic_bind")?,
    };
    let group = group_weights(mesh, modifier)?;
    let invert = bool_param(modifier, "invert_vertex_group")?;
    if bind_weights.len() != mesh.vertices.len() {
        return Err(invalid_parameter(
            modifier,
            "bind_data",
            "one bind-weight row per source vertex",
        ));
    }
    // Blender measures the cage in cage-local space, maps it into the bind frame, and subtracts
    // the bind-time position; moving the cage or owner object alone therefore has no effect.
    let cage_displacements = cage
        .vertices
        .iter()
        .zip(&bound_cage.vertices)
        .map(|(current, bound)| bind_matrix.transform_point3(current.co) - bound.co)
        .collect::<Vec<_>>();
    let dynamic_weights = if dynamic {
        let mut source_cage_space = mesh.clone();
        for vertex in &mut source_cage_space.vertices {
            vertex.co = subject_to_target.transform_point3(vertex.co);
        }
        Some(harmonic_mesh_bind(&source_cage_space, &cage, precision)?)
    } else {
        None
    };
    let rows = dynamic_weights
        .as_ref()
        .and_then(Value::as_array)
        .unwrap_or(bind_weights);
    let mut result = mesh.clone();
    for (vertex_index, vertex) in result.vertices.iter_mut().enumerate() {
        let weights = rows[vertex_index]
            .as_array()
            .filter(|weights| weights.len() == cage.vertices.len())
            .ok_or_else(|| {
                invalid_parameter(
                    modifier,
                    "bind_data",
                    "weights matching the cage vertex count",
                )
            })?;
        let group_weight = group.as_ref().map_or(1.0, |weights| {
            if invert {
                1.0 - weights[vertex_index]
            } else {
                weights[vertex_index]
            }
        });
        if group_weight <= 0.0 {
            continue;
        }
        let mut total_weight = 0.0;
        let mut deformation = DVec3::ZERO;
        for (weight, displacement) in weights.iter().zip(&cage_displacements) {
            let weight = finite_number(weight, modifier, "bind_data")?;
            total_weight += weight;
            deformation += *displacement * weight;
        }
        if total_weight <= 0.0 {
            continue;
        }
        vertex.co += inverse_deform_matrix.transform_vector3(deformation / total_weight)
            * group_weight;
    }
    validated(&result)?;
    Ok(result)
}

fn native_mesh_deform(mesh: &Mesh, modifier: &Modifier, bind_data: &Value) -> Result<Mesh> {
    let data = bind_data
        .as_object()
        .ok_or_else(|| invalid_parameter(modifier, "binding_data", "native Mesh Deform data"))?;
    let cage: Mesh =
        serde_json::from_value(
            modifier.runtime.target_mesh.clone().ok_or_else(|| {
                invalid_parameter(modifier, "object", "a mesh cage object operand")
            })?,
        )
        .map_err(|_| invalid_parameter(modifier, "object", "a mesh cage object operand"))?;
    let source_count = native_count(data, &["verts_num", "source_vertex_count"], modifier)?;
    let cage_count = native_count(data, &["cage_verts_num", "cage_vertex_count"], modifier)?;
    if source_count != mesh.vertices.len() || cage_count != cage.vertices.len() {
        return Err(invalid_parameter(
            modifier,
            "binding_data",
            "native source and cage vertex counts matching evaluated meshes",
        ));
    }
    let cage_coordinates = native_array(data, &["bind_cage_coordinates", "bindcagecos"], modifier)?;
    let coordinate_count = cage_count.checked_mul(3).ok_or_else(|| {
        invalid_parameter(
            modifier,
            "binding_data",
            "a valid native cage coordinate count",
        )
    })?;
    if cage_coordinates.len() != coordinate_count {
        return Err(invalid_parameter(
            modifier,
            "binding_data",
            "three bind-time cage coordinates per cage vertex",
        ));
    }
    let target_to_subject = modifier
        .runtime
        .target_to_subject
        .map_or(DMat4::IDENTITY, |matrix| DMat4::from_cols_array(&matrix));
    if !target_to_subject.is_finite() || target_to_subject.determinant().abs() <= f64::EPSILON {
        return Err(invalid_parameter(
            modifier,
            "object",
            "an invertible cage transform",
        ));
    }
    let subject_to_target = target_to_subject.inverse();
    let bind_matrix = native_matrix(data, modifier)?;
    if !bind_matrix.is_finite() || bind_matrix.determinant().abs() <= f64::EPSILON {
        return Err(invalid_parameter(
            modifier,
            "binding_data",
            "an invertible bind matrix",
        ));
    }
    let inverse_deform_matrix = (bind_matrix * subject_to_target).inverse();
    if !inverse_deform_matrix.is_finite() {
        return Err(invalid_parameter(
            modifier,
            "binding_data",
            "an invertible bind transform",
        ));
    }
    let mut cage_displacements = Vec::with_capacity(cage_count);
    for (index, vertex) in cage.vertices.iter().enumerate() {
        let cage_local = subject_to_target.transform_point3(vertex.co);
        let bind_position = native_flat_position(cage_coordinates, index, modifier)?;
        cage_displacements.push(bind_matrix.transform_point3(cage_local) - bind_position);
    }
    let group = group_weights(mesh, modifier)?;
    let flags = data.get("flag").and_then(Value::as_i64).unwrap_or_default();
    let invert = flags & 1 != 0 || bool_param(modifier, "invert_vertex_group")?;
    let dynamic = data
        .get("dynamic_bind")
        .and_then(Value::as_bool)
        .unwrap_or(flags & 2 != 0);
    let mut result = mesh.clone();
    if dynamic {
        native_mesh_deform_dynamic(
            &mut result,
            modifier,
            data,
            &cage_displacements,
            &subject_to_target,
            &inverse_deform_matrix,
            group.as_deref(),
            invert,
        )?;
    } else {
        native_mesh_deform_static(
            &mut result,
            modifier,
            data,
            &cage_displacements,
            &inverse_deform_matrix,
            group.as_deref(),
            invert,
        )?;
    }
    validated(&result)?;
    Ok(result)
}

fn native_mesh_deform_static(
    mesh: &mut Mesh,
    modifier: &Modifier,
    data: &serde_json::Map<String, Value>,
    cage_displacements: &[DVec3],
    inverse_deform_matrix: &DMat4,
    group: Option<&[f64]>,
    invert: bool,
) -> Result<()> {
    let offsets = native_array(data, &["bind_offsets", "bindoffsets"], modifier)?;
    let influences = native_array(data, &["bind_influences", "bindinfluences"], modifier)?;
    if offsets.len() != mesh.vertices.len() + 1 {
        return Err(invalid_parameter(
            modifier,
            "binding_data",
            "one native influence offset per source vertex plus a terminal offset",
        ));
    }
    let mut previous_offset = 0;
    for (vertex_index, vertex) in mesh.vertices.iter_mut().enumerate() {
        let start = offsets[vertex_index]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| {
                invalid_parameter(modifier, "binding_data", "valid native influence offsets")
            })?;
        let end = offsets[vertex_index + 1]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .filter(|end| *end >= start && *end <= influences.len())
            .ok_or_else(|| {
                invalid_parameter(modifier, "binding_data", "ordered native influence offsets")
            })?;
        if start != previous_offset {
            return Err(invalid_parameter(
                modifier,
                "binding_data",
                "contiguous native influence offsets",
            ));
        }
        previous_offset = end;
        let mut displacement = DVec3::ZERO;
        let mut total_weight = 0.0;
        for influence in &influences[start..end] {
            let cage_vertex = influence
                .get("vertex")
                .and_then(Value::as_i64)
                .and_then(|index| usize::try_from(index).ok())
                .filter(|index| *index < cage_displacements.len())
                .ok_or_else(|| {
                    invalid_parameter(modifier, "binding_data", "valid native cage indices")
                })?;
            let weight = influence.get("weight").ok_or_else(|| {
                invalid_parameter(modifier, "binding_data", "native cage influence weight")
            })?;
            let weight = finite_number(weight, modifier, "binding_data")?;
            displacement += cage_displacements[cage_vertex] * weight;
            total_weight += weight;
        }
        if total_weight > 0.0 {
            displacement = inverse_deform_matrix.transform_vector3(displacement / total_weight);
            let group_weight = group.map_or(1.0, |weights| {
                if invert {
                    1.0 - weights[vertex_index]
                } else {
                    weights[vertex_index]
                }
            });
            vertex.co += displacement * group_weight;
        }
    }
    if previous_offset
        != offsets
            .last()
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| {
                invalid_parameter(modifier, "binding_data", "a terminal influence offset")
            })?
    {
        return Err(invalid_parameter(
            modifier,
            "binding_data",
            "a terminal influence offset",
        ));
    }
    Ok(())
}

fn native_mesh_deform_dynamic(
    mesh: &mut Mesh,
    modifier: &Modifier,
    data: &serde_json::Map<String, Value>,
    cage_displacements: &[DVec3],
    subject_to_target: &DMat4,
    inverse_deform_matrix: &DMat4,
    group: Option<&[f64]>,
    invert: bool,
) -> Result<()> {
    let size = native_count(data, &["dynamic_grid_size", "dyngridsize"], modifier)?;
    let cell_min = native_vec3(data, &["dynamic_cell_min", "dyncellmin"], modifier)?;
    let cell_width = data
        .get("dynamic_cell_width")
        .or_else(|| data.get("dyncellwidth"))
        .ok_or_else(|| invalid_parameter(modifier, "binding_data", "native dynamic cell width"))?;
    let cell_width = finite_number(cell_width, modifier, "binding_data")?;
    if size == 0 || !cell_width.is_finite() || cell_width <= 0.0 {
        return Err(invalid_parameter(
            modifier,
            "binding_data",
            "a valid native dynamic grid",
        ));
    }
    let cell_count = size
        .checked_mul(size)
        .and_then(|count| count.checked_mul(size))
        .ok_or_else(|| {
            invalid_parameter(modifier, "binding_data", "a bounded native dynamic grid")
        })?;
    let cells = native_array(data, &["dynamic_grid", "dyngrid"], modifier)?;
    let influences = native_array(data, &["dynamic_influences", "dyninfluences"], modifier)?;
    let dynamic_vertices = native_array(data, &["dynamic_vertices", "dynverts"], modifier)?;
    if cells.len() != cell_count || dynamic_vertices.len() != mesh.vertices.len() {
        return Err(invalid_parameter(
            modifier,
            "binding_data",
            "native dynamic grid and source-vertex dimensions",
        ));
    }
    let max_coordinate = i32::try_from(size - 1).map_err(|_| {
        invalid_parameter(modifier, "binding_data", "bounded native grid coordinates")
    })?;
    for (vertex_index, vertex) in mesh.vertices.iter_mut().enumerate() {
        if !native_flag(&dynamic_vertices[vertex_index]) {
            continue;
        }
        let target_position = subject_to_target.transform_point3(vertex.co);
        let grid = (target_position - cell_min - DVec3::splat(cell_width * 0.5)) / cell_width;
        if !grid.is_finite() {
            return Err(invalid_parameter(
                modifier,
                "binding_data",
                "finite dynamic grid coordinates",
            ));
        }
        let base = [
            native_grid_integer(grid.x),
            native_grid_integer(grid.y),
            native_grid_integer(grid.z),
        ];
        let fraction = DVec3::new(
            grid.x - f64::from(base[0]),
            grid.y - f64::from(base[1]),
            grid.z - f64::from(base[2]),
        );
        let mut displacement = DVec3::ZERO;
        let mut total_weight = 0.0;
        for corner in 0..8 {
            let mut cell_coordinate = [0_usize; 3];
            let mut cell_weight = 1.0;
            for axis in 0..3 {
                let upper = corner & (1 << axis) != 0;
                let coordinate = base[axis].saturating_add(i32::from(upper));
                cell_coordinate[axis] = usize::try_from(coordinate.clamp(0, max_coordinate))
                    .map_err(|_| {
                        invalid_parameter(modifier, "binding_data", "native grid cell indices")
                    })?;
                cell_weight *= if upper {
                    fraction[axis]
                } else {
                    1.0 - fraction[axis]
                };
            }
            let flat_cell =
                cell_coordinate[0] + cell_coordinate[1] * size + cell_coordinate[2] * size * size;
            let cell = &cells[flat_cell];
            let start = cell
                .get("offset")
                .and_then(Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| {
                    invalid_parameter(modifier, "binding_data", "native dynamic cell offset")
                })?;
            let count = cell
                .get("influences_num")
                .and_then(Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .filter(|count| {
                    start
                        .checked_add(*count)
                        .is_some_and(|end| end <= influences.len())
                })
                .ok_or_else(|| {
                    invalid_parameter(modifier, "binding_data", "native dynamic cell influences")
                })?;
            for influence in &influences[start..start + count] {
                let cage_vertex = influence
                    .get("vertex")
                    .and_then(Value::as_i64)
                    .and_then(|index| usize::try_from(index).ok())
                    .filter(|index| *index < cage_displacements.len())
                    .ok_or_else(|| {
                        invalid_parameter(modifier, "binding_data", "valid native cage indices")
                    })?;
                let weight = influence.get("weight").ok_or_else(|| {
                    invalid_parameter(modifier, "binding_data", "native dynamic influence")
                })?;
                let weight = finite_number(weight, modifier, "binding_data")? * cell_weight;
                displacement += cage_displacements[cage_vertex] * weight;
                total_weight += weight;
            }
        }
        if total_weight > 0.0 {
            displacement = inverse_deform_matrix.transform_vector3(displacement / total_weight);
            let group_weight = group.map_or(1.0, |weights| {
                if invert {
                    1.0 - weights[vertex_index]
                } else {
                    weights[vertex_index]
                }
            });
            vertex.co += displacement * group_weight;
        }
    }
    Ok(())
}

fn native_count(
    data: &serde_json::Map<String, Value>,
    keys: &[&str],
    modifier: &Modifier,
) -> Result<usize> {
    keys.iter()
        .find_map(|key| data.get(*key))
        .and_then(Value::as_u64)
        .and_then(|count| usize::try_from(count).ok())
        .ok_or_else(|| invalid_parameter(modifier, "binding_data", "a valid native vertex count"))
}

fn native_array<'a>(
    data: &'a serde_json::Map<String, Value>,
    keys: &[&str],
    modifier: &Modifier,
) -> Result<&'a [Value]> {
    keys.iter()
        .find_map(|key| data.get(*key))
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| invalid_parameter(modifier, "binding_data", "a native modifier array"))
}

fn native_flat_position(values: &[Value], index: usize, modifier: &Modifier) -> Result<DVec3> {
    let start = index
        .checked_mul(3)
        .ok_or_else(|| invalid_parameter(modifier, "binding_data", "native coordinate indices"))?;
    let end = start
        .checked_add(3)
        .ok_or_else(|| invalid_parameter(modifier, "binding_data", "native coordinate indices"))?;
    let coordinates = values.get(start..end).ok_or_else(|| {
        invalid_parameter(
            modifier,
            "binding_data",
            "native three-dimensional coordinates",
        )
    })?;
    Ok(DVec3::new(
        finite_number(&coordinates[0], modifier, "binding_data")?,
        finite_number(&coordinates[1], modifier, "binding_data")?,
        finite_number(&coordinates[2], modifier, "binding_data")?,
    ))
}

fn native_matrix(data: &serde_json::Map<String, Value>, modifier: &Modifier) -> Result<DMat4> {
    let rows = data
        .get("bind_matrix")
        .or_else(|| data.get("bindmat"))
        .or_else(|| data.get("mat"))
        .and_then(Value::as_array)
        .filter(|rows| rows.len() == 4)
        .ok_or_else(|| {
            invalid_parameter(modifier, "binding_data", "a native 4-by-4 bind matrix")
        })?;
    let mut columns = [0.0; 16];
    for (row_index, row) in rows.iter().enumerate() {
        let values = row
            .as_array()
            .filter(|values| values.len() == 4)
            .ok_or_else(|| {
                invalid_parameter(modifier, "binding_data", "a native 4-by-4 bind matrix")
            })?;
        for (column_index, value) in values.iter().enumerate() {
            columns[row_index * 4 + column_index] = finite_number(value, modifier, "binding_data")?;
        }
    }
    Ok(DMat4::from_cols_array(&columns))
}

fn native_vec3(
    data: &serde_json::Map<String, Value>,
    keys: &[&str],
    modifier: &Modifier,
) -> Result<DVec3> {
    let values = native_array(data, keys, modifier)?;
    if values.len() != 3 {
        return Err(invalid_parameter(
            modifier,
            "binding_data",
            "a native three-component vector",
        ));
    }
    Ok(DVec3::new(
        finite_number(&values[0], modifier, "binding_data")?,
        finite_number(&values[1], modifier, "binding_data")?,
        finite_number(&values[2], modifier, "binding_data")?,
    ))
}

fn native_flag(value: &Value) -> bool {
    value
        .as_bool()
        .unwrap_or_else(|| value.as_i64().is_some_and(|value| value != 0))
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "Blender converts dynamic grid coordinates to int by truncation"
)]
fn native_grid_integer(value: f64) -> i32 {
    value.trunc() as i32
}

fn same_mesh_topology(left: &Mesh, right: &Mesh) -> bool {
    left.vertices
        .iter()
        .map(|vertex| vertex.id)
        .eq(right.vertices.iter().map(|vertex| vertex.id))
        && left
            .faces
            .iter()
            .map(|face| (face.id, &face.vertices))
            .eq(right.faces.iter().map(|face| (face.id, &face.vertices)))
}

/// Extends hook-moved anchor displacement over the mesh with a discrete Laplace solve.
pub(super) fn laplacian_deform(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    let Some(bind_data) = modifier.binding_data.as_ref() else {
        return Ok(mesh.clone());
    };
    let rest = if is_native_bind(bind_data, "laplacian_deform") {
        let data = bind_data.as_object().ok_or_else(|| {
            invalid_parameter(modifier, "binding_data", "native Laplacian Deform data")
        })?;
        let count = native_count(data, &["verts_num", "source_vertex_count"], modifier)?;
        let coordinates = native_array(data, &["vertex_coordinates", "vertexco"], modifier)?;
        if count != mesh.vertices.len()
            || coordinates.len()
                != count.checked_mul(3).ok_or_else(|| {
                    invalid_parameter(modifier, "binding_data", "native bind coordinate count")
                })?
        {
            return Err(invalid_parameter(
                modifier,
                "binding_data",
                "one native bind position per source vertex",
            ));
        }
        (0..count)
            .map(|index| native_flat_position(coordinates, index, modifier))
            .collect::<Result<Vec<_>>>()?
    } else {
        let rows = bind_data.as_array().ok_or_else(|| {
            invalid_parameter(modifier, "binding_data", "stored Laplacian bind positions")
        })?;
        if rows.len() != mesh.vertices.len() {
            return Err(invalid_parameter(
                modifier,
                "bind_data",
                "one bind position per source vertex",
            ));
        }
        rows.iter()
            .map(|record| {
                let values = record
                    .as_array()
                    .filter(|values| values.len() == 3)
                    .ok_or_else(|| {
                        invalid_parameter(modifier, "bind_data", "three-dimensional bind positions")
                    })?;
                let point = DVec3::new(
                    finite_number(&values[0], modifier, "bind_data")?,
                    finite_number(&values[1], modifier, "bind_data")?,
                    finite_number(&values[2], modifier, "bind_data")?,
                );
                if point.is_finite() {
                    Ok(point)
                } else {
                    Err(invalid_parameter(modifier, "bind_data", "finite bind positions"))
                }
            })
            .collect::<Result<Vec<_>>>()?
    };
    let iterations = usize::try_from(uint_param(modifier, "iterations")?)
        .map_err(|_| invalid_parameter(modifier, "iterations", "a non-negative integer"))?;
    let group = group_weights(mesh, modifier)?.ok_or_else(|| {
        invalid_parameter(modifier, "vertex_group", "an anchor vertex-group name")
    })?;
    let invert = bool_param(modifier, "invert_vertex_group")?;
    let mut anchors = vec![false; mesh.vertices.len()];
    let mut anchor_positions = vec![DVec3::ZERO; mesh.vertices.len()];
    for (index, vertex) in mesh.vertices.iter().enumerate() {
        let weight = if invert {
            1.0 - group[index]
        } else {
            group[index]
        };
        if weight > 0.0 {
            anchors[index] = true;
            anchor_positions[index] = vertex.co;
        }
    }
    if !anchors.iter().any(|anchor| *anchor) {
        return Ok(mesh.clone());
    }
    let current_positions = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    let triangles = mesh_triangles(mesh, &current_positions)?;
    let laplacian = cotangent_laplacian(&rest, &triangles)?;
    let original_normals = vertex_normals(&rest, &triangles);
    let neighbors = mesh_neighbors(mesh);
    let unit_vertices = laplacian_unit_vertices(&rest, &original_normals, &neighbors);
    let differential = apply_laplacian(&laplacian, &rest);
    let mut evaluated = solve_laplacian_system(
        &laplacian,
        &differential,
        &anchors,
        &anchor_positions,
        &rest,
    )?;
    let mut rotated = vec![DVec3::ZERO; differential.len()];
    for _ in 0..iterations {
        rotate_differentials(
            &rest,
            &evaluated,
            &differential,
            &original_normals,
            &unit_vertices,
            &triangles,
            &mut rotated,
        );
        evaluated = solve_laplacian_system(
            &laplacian,
            &rotated,
            &anchors,
            &anchor_positions,
            &evaluated,
        )?;
    }
    let mut result = mesh.clone();
    for (vertex, position) in result.vertices.iter_mut().zip(evaluated) {
        vertex.co = position;
    }
    validated(&result)?;
    Ok(result)
}
type LaplacianRows = Vec<Vec<(usize, f64)>>;

pub(super) fn mesh_triangles(mesh: &Mesh, coordinates: &[DVec3]) -> Result<Vec<[usize; 3]>> {
    let indices = index_positions(mesh);
    let mut triangles = Vec::new();
    for face in &mesh.faces {
        let vertices = face
            .vertices
            .iter()
            .map(|vertex| indices.get(vertex).copied())
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                PotError::invalid_operation("Laplacian mesh has invalid face vertices")
            })?;
        if vertices.len() == 4 {
            let [first, second, third, fourth] =
                [vertices[0], vertices[1], vertices[2], vertices[3]];
            let first = coordinates[first];
            let second = coordinates[second];
            let third = coordinates[third];
            let fourth = coordinates[fourth];
            let diagonal_flip = (second - first)
                .cross(third - first)
                .dot((fourth - first).cross(third - first))
                > 0.0;
            if diagonal_flip {
                triangles.push([vertices[0], vertices[1], vertices[3]]);
                triangles.push([vertices[1], vertices[2], vertices[3]]);
            } else {
                triangles.push([vertices[0], vertices[1], vertices[2]]);
                triangles.push([vertices[0], vertices[2], vertices[3]]);
            }
        } else {
            for index in 1..vertices.len().saturating_sub(1) {
                triangles.push([vertices[0], vertices[index], vertices[index + 1]]);
            }
        }
    }
    Ok(triangles)
}

fn cotangent_laplacian(rest: &[DVec3], triangles: &[[usize; 3]]) -> Result<LaplacianRows> {
    let mut weights = vec![BTreeMap::<usize, f64>::new(); rest.len()];
    for triangle in triangles {
        let [first, second, third] = *triangle;
        let points = [rest[first], rest[second], rest[third]];
        let cotangent = |left: DVec3, right: DVec3| {
            let cross_length = left.cross(right).length();
            if cross_length > f64::from(f32::EPSILON) {
                left.dot(right) / cross_length
            } else {
                0.0
            }
        };
        let opposite_first = cotangent(points[1] - points[0], points[2] - points[0]);
        let opposite_second = cotangent(points[2] - points[1], points[0] - points[1]);
        let opposite_third = cotangent(points[0] - points[2], points[1] - points[2]);
        for (left, right, weight) in [
            (second, third, opposite_first),
            (third, first, opposite_second),
            (first, second, opposite_third),
        ] {
            *weights[left].entry(right).or_default() += weight;
            *weights[right].entry(left).or_default() += weight;
        }
    }
    let mut rows = Vec::with_capacity(weights.len());
    for (index, adjacent) in weights.into_iter().enumerate() {
        let diagonal = adjacent.values().sum::<f64>();
        if !diagonal.is_finite() {
            return Err(PotError::invalid_operation(
                "Laplacian mesh has non-finite cotangent weights",
            ));
        }
        let mut row = Vec::with_capacity(adjacent.len() + 1);
        row.push((index, diagonal));
        row.extend(
            adjacent
                .into_iter()
                .map(|(neighbor, weight)| (neighbor, -weight)),
        );
        rows.push(row);
    }
    Ok(rows)
}

fn mesh_neighbors(mesh: &Mesh) -> Vec<Vec<usize>> {
    let indices = index_positions(mesh);
    let mut neighbors = vec![Vec::new(); mesh.vertices.len()];
    for edge in &mesh.edges {
        if let (Some(&first), Some(&second)) = (
            indices.get(&edge.vertices[0]),
            indices.get(&edge.vertices[1]),
        ) {
            neighbors[first].push(second);
            neighbors[second].push(first);
        }
    }
    neighbors
}

fn vertex_normals(points: &[DVec3], triangles: &[[usize; 3]]) -> Vec<DVec3> {
    let mut normals = vec![DVec3::ZERO; points.len()];
    for triangle in triangles {
        let [first, second, third] = *triangle;
        let normal = (points[second] - points[first])
            .cross(points[third] - points[first])
            .normalize_or_zero();
        for index in [first, second, third] {
            normals[index] += normal;
        }
    }
    normals
        .into_iter()
        .map(|normal| normal.try_normalize().unwrap_or(DVec3::ZERO))
        .collect()
}
/// Blender chooses each local frame's reference neighbor once from the bind mesh.
fn laplacian_unit_vertices(
    rest: &[DVec3],
    normals: &[DVec3],
    neighbors: &[Vec<usize>],
) -> Vec<Option<usize>> {
    neighbors
        .iter()
        .enumerate()
        .map(|(index, adjacent)| {
            let mut best = None;
            for &neighbor in adjacent {
                let direction = (rest[neighbor] - rest[index]).normalize_or_zero();
                let weight = direction.dot(normals[index]).abs();
                if best.is_none_or(|(_, best_weight)| weight < best_weight) {
                    best = Some((neighbor, weight));
                }
            }
            best.map(|(neighbor, _)| neighbor)
        })
        .collect()
}


fn rotate_differentials(
    rest: &[DVec3],
    current: &[DVec3],
    differential: &[DVec3],
    original_normals: &[DVec3],
    unit_vertices: &[Option<usize>],
    triangles: &[[usize; 3]],
    output: &mut [DVec3],
) {
    let current_normals = vertex_normals(current, triangles);
    for index in 0..rest.len() {
        let original_normal = original_normals[index];
        let current_normal = current_normals[index];
        let Some(best_neighbor) = unit_vertices.get(index).copied().flatten() else {
            output[index] = differential[index];
            continue;
        };
        let original_edge = rest[best_neighbor] - rest[index];
        let original_tangent = (original_edge
            - original_normal * original_edge.dot(original_normal))
        .try_normalize()
        .unwrap_or(DVec3::ZERO);
        let current_edge = current[best_neighbor] - current[index];
        let current_tangent = (current_edge - current_normal * current_edge.dot(current_normal))
            .try_normalize()
            .unwrap_or(DVec3::ZERO);
        let original_binormal = original_normal.cross(original_tangent);
        let current_binormal = current_normal.cross(current_tangent);
        let local = DVec3::new(
            differential[index].dot(original_normal),
            differential[index].dot(original_tangent),
            differential[index].dot(original_binormal),
        );
        let rotated =
            current_normal * local.x + current_tangent * local.y + current_binormal * local.z;
        output[index] = if rotated.length_squared() > f64::from(f32::EPSILON) {
            rotated
        } else {
            differential[index]
        };
    }
}

fn apply_laplacian(rows: &LaplacianRows, points: &[DVec3]) -> Vec<DVec3> {
    let mut output = vec![DVec3::ZERO; rows.len()];
    apply_laplacian_into(rows, points, &mut output);
    output
}

fn apply_laplacian_into(rows: &LaplacianRows, points: &[DVec3], output: &mut [DVec3]) {
    for (index, row) in rows.iter().enumerate() {
        output[index] = row
            .iter()
            .map(|(vertex, weight)| points[*vertex] * *weight)
            .sum();
    }
}

fn componentwise_dot(left: &[DVec3], right: &[DVec3]) -> [f64; 3] {
    left.iter()
        .zip(right)
        .fold([0.0; 3], |mut total, (left, right)| {
            for axis in 0..3 {
                total[axis] += left[axis] * right[axis];
            }
            total
        })
}

fn scale_components(value: DVec3, scales: [f64; 3]) -> DVec3 {
    DVec3::new(
        value.x * scales[0],
        value.y * scales[1],
        value.z * scales[2],
    )
}

// Blender minimizes ||Lx - delta||² plus one unit-weight position row per anchor;
// anchors are least-squares constraints rather than fixed unknowns.
fn apply_least_squares_normal_equations(
    rows: &LaplacianRows,
    anchors: &[bool],
    points: &[DVec3],
    laplacian: &mut [DVec3],
    output: &mut [DVec3],
) {
    apply_laplacian_into(rows, points, laplacian);
    output.fill(DVec3::ZERO);
    for (row_index, row) in rows.iter().enumerate() {
        for &(column, weight) in row {
            output[column] += laplacian[row_index] * weight;
        }
    }
    for (index, anchored) in anchors.iter().enumerate() {
        if *anchored {
            output[index] += points[index];
        }
    }
}

fn solve_laplacian_system(
    rows: &LaplacianRows,
    differential: &[DVec3],
    anchors: &[bool],
    anchor_positions: &[DVec3],
    initial: &[DVec3],
) -> Result<Vec<DVec3>> {
    let mut rhs = vec![DVec3::ZERO; rows.len()];
    for (row_index, row) in rows.iter().enumerate() {
        for &(column, weight) in row {
            rhs[column] += differential[row_index] * weight;
        }
    }
    for (index, anchored) in anchors.iter().enumerate() {
        if *anchored {
            rhs[index] += anchor_positions[index];
        }
    }

    let mut solution = initial.to_vec();
    let mut laplacian = vec![DVec3::ZERO; rows.len()];
    let mut product = vec![DVec3::ZERO; rows.len()];
    apply_least_squares_normal_equations(rows, anchors, &solution, &mut laplacian, &mut product);
    let mut residual = rhs
        .iter()
        .zip(&product)
        .map(|(right, left)| *right - *left)
        .collect::<Vec<_>>();
    let mut direction = residual.clone();
    let mut residual_norm = componentwise_dot(&residual, &residual);
    let target_norm = residual_norm.map(|norm| norm * 1.0e-24 + 1.0e-30);
    let iteration_limit = rows.len().saturating_mul(8).clamp(64, 10_000);
    for _ in 0..iteration_limit {
        if (0..3).all(|axis| residual_norm[axis] <= target_norm[axis]) {
            break;
        }
        apply_least_squares_normal_equations(
            rows,
            anchors,
            &direction,
            &mut laplacian,
            &mut product,
        );
        let denominator = componentwise_dot(&direction, &product);
        let mut scale = [0.0; 3];
        for axis in 0..3 {
            if residual_norm[axis] > target_norm[axis]
                && denominator[axis].is_finite()
                && denominator[axis] > 0.0
            {
                scale[axis] = residual_norm[axis] / denominator[axis];
            }
        }
        for index in 0..rows.len() {
            solution[index] += scale_components(direction[index], scale);
            residual[index] -= scale_components(product[index], scale);
        }
        let next_norm = componentwise_dot(&residual, &residual);
        let mut beta = [0.0; 3];
        for axis in 0..3 {
            if next_norm[axis] > target_norm[axis] && residual_norm[axis] > 0.0 {
                beta[axis] = next_norm[axis] / residual_norm[axis];
            }
        }
        for index in 0..rows.len() {
            direction[index] = residual[index] + scale_components(direction[index], beta);
        }
        residual_norm = next_norm;
    }
    if solution.iter().any(|point| !point.is_finite()) {
        return Err(PotError::invalid_operation(
            "Laplacian deform least-squares solver produced a non-finite position",
        ));
    }
    Ok(solution)
}

/// Tolerance of Blender's Surface Deform bind (`FLT_EPSILON`).
const SURFACE_BIND_EPSILON: f64 = f32::EPSILON as f64;

/// Blender's `SDefBindPoly`: one target polygon around the nearest target vertex of a bound point.
struct SurfaceBindPolygon {
    face: usize,
    corners: Vec<usize>,
    corner_edges: Vec<usize>,
    coords: Vec<DVec3>,
    coords_v2: Vec<DVec2>,
    point_v2: DVec2,
    centroid: DVec3,
    normal: DVec3,
    edge_vertex_indices: [usize; 2],
    edge_indices: [usize; 2],
    edgemid_angle: f64,
    corner_edgemid_angles: [f64; 2],
    point_edgemid_angles: [f64; 2],
    scales: [f64; 2],
    scale_mid: f64,
    weight_angular: f64,
    weight_dist_proj: f64,
    weight_dist: f64,
    weight: f64,
    dominant_edge: usize,
    dominant_angle_weight: f64,
    inside: bool,
}

impl SurfaceBindPolygon {
    /// Blender's `computeAngularWeight` input for the point's angle at one flanking edge.
    fn edge_angular_weight(&self, edge: usize) -> f64 {
        let side = usize::from(self.edge_indices[0] != edge);
        angular_weight(self.point_edgemid_angles[side], self.edgemid_angle)
    }
}

/// Blender's `MOD_SDEF_INFINITE_WEIGHT_*` flags.
#[derive(Clone, Copy, Default)]
struct InfiniteWeights {
    angular: bool,
    dist_proj: bool,
    dist: bool,
}

impl InfiniteWeights {
    fn any(self) -> bool {
        self.angular || self.dist_proj || self.dist
    }

    fn merge(&mut self, other: Self) {
        self.angular |= other.angular;
        self.dist_proj |= other.dist_proj;
        self.dist |= other.dist;
    }
}

/// One stored Surface Deform bind term (Blender's `SDefBind`).
struct SurfaceBind {
    influence: f64,
    mode: u8,
    vertex_indices: Vec<usize>,
    vertex_weights: Vec<f64>,
    normal_distance: f64,
}

impl SurfaceBind {
    fn to_json(&self) -> Value {
        json!({
            "influence": self.influence,
            "mode": self.mode,
            "verts_num": self.vertex_indices.len(),
            "vert_inds": self.vertex_indices,
            "vert_weights": self.vertex_weights,
            "normal_dist": self.normal_distance,
        })
    }
}

/// Target mesh prepared for Surface Deform binding (Blender's `SDefBindCalcData`).
struct SurfaceBindTarget {
    /// Bind-time target positions in the owner's space (`mat * target_local`).
    points: Vec<DVec3>,
    /// Target positions in target-local space, used for the nearest-triangle query.
    local_points: Vec<DVec3>,
    owner_to_local: DMat4,
    faces: Vec<Vec<usize>>,
    face_edges: Vec<Vec<usize>>,
    edge_vertices: Vec<[usize; 2]>,
    edge_faces: Vec<Vec<usize>>,
    vertex_edges: Vec<Vec<usize>>,
}

impl SurfaceBindTarget {
    fn new(target: &Mesh, target_to_owner: DMat4) -> Result<Self> {
        if target.faces.is_empty() {
            return Err(PotError::invalid_operation("Target has no faces"));
        }
        let owner_to_local = target_to_owner.inverse();
        if !target_to_owner.is_finite() || !owner_to_local.is_finite() {
            return Err(PotError::invalid_argument(
                "surface deform binding requires an invertible target transform",
            ));
        }
        let target_indices = index_positions(target);
        let faces = target
            .faces
            .iter()
            .map(|face| {
                face.vertices
                    .iter()
                    .map(|id| target_indices.get(id).copied())
                    .collect::<Option<Vec<_>>>()
                    .filter(|corners| corners.len() >= 3)
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                PotError::invalid_operation("surface deform target face has invalid vertex IDs")
            })?;
        let mut edge_ids = HashMap::<(usize, usize), usize>::new();
        let mut edge_vertices = Vec::<[usize; 2]>::new();
        let mut edge_faces = Vec::<Vec<usize>>::new();
        let mut face_edges = Vec::with_capacity(faces.len());
        for (face_index, corners) in faces.iter().enumerate() {
            let mut corner_edges = Vec::with_capacity(corners.len());
            for corner in 0..corners.len() {
                let first = corners[corner];
                let second = corners[(corner + 1) % corners.len()];
                let key = (first.min(second), first.max(second));
                let edge = match edge_ids.get(&key) {
                    Some(edge) => *edge,
                    None => {
                        let edge = edge_vertices.len();
                        edge_ids.insert(key, edge);
                        edge_vertices.push([first, second]);
                        edge_faces.push(Vec::new());
                        edge
                    }
                };
                edge_faces[edge].push(face_index);
                if edge_faces[edge].len() > 2 {
                    return Err(PotError::invalid_operation(
                        "Target has edges with more than two polygons",
                    ));
                }
                corner_edges.push(edge);
            }
            face_edges.push(corner_edges);
        }
        let mut vertex_edges = vec![Vec::new(); target.vertices.len()];
        for (edge, [first, second]) in edge_vertices.iter().copied().enumerate() {
            vertex_edges[first].push(edge);
            vertex_edges[second].push(edge);
        }
        let points: Vec<DVec3> = target
            .vertices
            .iter()
            .map(|vertex| target_to_owner.transform_point3(vertex.co))
            .collect();
        let local_points: Vec<DVec3> = target.vertices.iter().map(|vertex| vertex.co).collect();
        Ok(Self {
            points,
            local_points,
            owner_to_local,
            faces,
            face_edges,
            edge_vertices,
            edge_faces,
            vertex_edges,
        })
    }

    /// Blender's `nearestVert`: nearest triangle in target-local space, then nearest edge and vertex.
    fn nearest_vertex(&self, point: DVec3) -> Result<usize> {
        let local_point = self.owner_to_local.transform_point3(point);
        let mut nearest_face = None;
        let mut nearest_distance = f64::INFINITY;
        for (face_index, corners) in self.faces.iter().enumerate() {
            for corner in 1..corners.len() - 1 {
                let triangle = [corners[0], corners[corner], corners[corner + 1]]
                    .map(|index| self.local_points[index]);
                let (closest, _) = closest_triangle_point(local_point, triangle);
                let distance = closest.distance_squared(local_point);
                if distance < nearest_distance {
                    nearest_distance = distance;
                    nearest_face = Some(face_index);
                }
            }
        }
        let face = nearest_face
            .ok_or_else(|| PotError::invalid_operation("surface deform target has no faces"))?;
        let mut nearest_edge = None;
        let mut nearest_edge_distance = f64::INFINITY;
        for &edge in &self.face_edges[face] {
            let [first, second] = self.edge_vertices[edge];
            let distance =
                distance_squared_to_segment(point, self.points[first], self.points[second]);
            if distance < nearest_edge_distance {
                nearest_edge_distance = distance;
                nearest_edge = Some(edge);
            }
        }
        let [first, second] = self.edge_vertices[nearest_edge.ok_or_else(|| {
            PotError::invalid_operation("surface deform target polygon is empty")
        })?];
        Ok(
            if point.distance_squared(self.points[first])
                < point.distance_squared(self.points[second])
            {
                first
            } else {
                second
            },
        )
    }

    /// Blender's `computeBindWeights` followed by the per-polygon bind term selection of `bindVert`.
    fn bind_vertex(&self, point: DVec3, falloff: f64) -> Result<Vec<SurfaceBind>> {
        let nearest = self.nearest_vertex(point)?;
        let mut polygons = Vec::<SurfaceBindPolygon>::new();
        let mut infinite = InfiniteWeights::default();
        for &edge in &self.vertex_edges[nearest] {
            for &face in &self.edge_faces[edge] {
                if polygons.iter().any(|polygon| polygon.face == face) {
                    continue;
                }
                let (polygon, flags) = self.polygon(face, nearest, point)?;
                infinite.merge(flags);
                polygons.push(polygon);
            }
        }
        if polygons.is_empty() {
            return Err(PotError::invalid_operation(
                "surface deform target vertex has no polygons",
            ));
        }
        let average_distance =
            polygons.iter().map(|polygon| polygon.weight_dist).sum::<f64>() / polygons.len() as f64;
        if !infinite.any() {
            for &edge in &self.vertex_edges[nearest] {
                let members = (0..polygons.len())
                    .filter(|&index| self.edge_faces[edge].contains(&polygons[index].face))
                    .collect::<Vec<_>>();
                match members.as_slice() {
                    [first] => {
                        let angle = polygons[*first].edge_angular_weight(edge);
                        polygons[*first].weight_angular *= angle * angle;
                    }
                    [first, second] => {
                        let product = polygons[*first].edge_angular_weight(edge)
                            * polygons[*second].edge_angular_weight(edge);
                        polygons[*first].weight_angular *= product;
                        polygons[*second].weight_angular *= product;
                    }
                    _ => {}
                }
            }
            for polygon in &mut polygons {
                let corner_angle_weights = [0, 1].map(|index| {
                    polygon.point_edgemid_angles[index] / polygon.corner_edgemid_angles[index]
                });
                if corner_angle_weights.iter().any(|weight| weight.is_nan()) {
                    return Err(invalid_polygons());
                }
                let dominant = usize::from(corner_angle_weights[0] >= corner_angle_weights[1]);
                let dominant_weight = corner_angle_weights[dominant];
                if !(0.0..=1.0).contains(&dominant_weight) {
                    return Err(invalid_polygons());
                }
                polygon.dominant_edge = dominant;
                polygon.dominant_angle_weight = (dominant_weight * FRAC_PI_2).sin();
                let other = 1 - dominant;
                let edge_angle_a = polygon.point_edgemid_angles[dominant];
                let edge_angle_b = polygon.point_edgemid_angles[other];
                let mut scale_weight = edge_angle_a
                    / edge_angle_a.max(polygon.edgemid_angle);
                scale_weight /=
                    scale_weight + edge_angle_b / edge_angle_b.max(polygon.edgemid_angle);
                let squared = scale_weight * scale_weight;
                let inverse_squared = (1.0 - scale_weight) * (1.0 - scale_weight);
                let scale_weight = squared / (squared + inverse_squared);
                let edge_scale = scale_weight * polygon.scales[other]
                    + (1.0 - scale_weight) * polygon.scales[dominant];
                let dominant_weight = polygon.dominant_angle_weight;
                polygon.scales[0] = dominant_weight * polygon.scale_mid
                    + (1.0 - dominant_weight) * edge_scale;
                polygon.weight_dist_proj =
                    (polygon.weight_dist_proj / polygon.scales[0]).powf(falloff);
                polygon.weight_dist = (polygon.weight_dist / average_distance).powf(falloff);
                if polygon.weight_dist < SURFACE_BIND_EPSILON {
                    infinite.dist_proj = true;
                    infinite.dist = true;
                } else if polygon.weight_dist_proj < SURFACE_BIND_EPSILON {
                    infinite.dist_proj = true;
                } else if polygon.weight_angular < SURFACE_BIND_EPSILON {
                    infinite.angular = true;
                }
            }
        } else if !infinite.dist {
            for polygon in &mut polygons {
                polygon.weight_dist = (polygon.weight_dist / average_distance).powf(falloff);
                if polygon.weight_dist < SURFACE_BIND_EPSILON {
                    infinite.dist = true;
                }
            }
        }
        let mut total = 0.0;
        for polygon in &mut polygons {
            let mut weight = if infinite.dist {
                if polygon.weight_dist < SURFACE_BIND_EPSILON {
                    1.0
                } else {
                    0.0
                }
            } else if infinite.dist_proj {
                if polygon.weight_dist_proj < SURFACE_BIND_EPSILON {
                    1.0 / polygon.weight_dist
                } else {
                    0.0
                }
            } else if infinite.angular {
                if polygon.weight_angular < SURFACE_BIND_EPSILON {
                    1.0 / polygon.weight_dist_proj / polygon.weight_dist
                } else {
                    0.0
                }
            } else {
                1.0 / polygon.weight_angular / polygon.weight_dist_proj / polygon.weight_dist
            };
            weight *= polygon.edgemid_angle / PI;
            polygon.weight = weight;
            total += weight;
        }
        for polygon in &mut polygons {
            polygon.weight /= total;
        }
        let mut binds = Vec::new();
        for polygon in &polygons {
            if !(polygon.weight >= SURFACE_BIND_EPSILON) {
                continue;
            }
            if polygon.inside {
                binds.push(self.ngon_bind(polygon, point)?);
                continue;
            }
            if 1.0 - polygon.dominant_angle_weight >= SURFACE_BIND_EPSILON {
                binds.push(self.centroid_bind(polygon, point)?);
            }
            if polygon.dominant_angle_weight >= SURFACE_BIND_EPSILON {
                binds.push(self.corner_triangle_bind(polygon, point)?);
            }
        }
        Ok(binds)
    }

    /// Blender's `bpoly` construction for one face adjacent to the nearest vertex.
    fn polygon(
        &self,
        face: usize,
        nearest: usize,
        point: DVec3,
    ) -> Result<(SurfaceBindPolygon, InfiniteWeights)> {
        let corners = self.faces[face].clone();
        let corner_edges = self.face_edges[face].clone();
        let count = corners.len();
        let coords = corners
            .iter()
            .map(|&vertex| self.points[vertex])
            .collect::<Vec<_>>();
        let corner_index = corners
            .iter()
            .position(|&vertex| vertex == nearest)
            .ok_or_else(|| {
                PotError::invalid_operation(
                    "surface deform target polygon does not contain its nearest vertex",
                )
            })?;
        let edge_vertex_indices = [(corner_index + count - 1) % count, (corner_index + 1) % count];
        let edge_indices = [
            corner_edges[edge_vertex_indices[0]],
            corner_edges[corner_index],
        ];
        let centroid = coords.iter().copied().sum::<DVec3>() / count as f64;
        let normal = polygon_normal(&coords).ok_or_else(invalid_polygons)?;
        let axis = normal.cross(DVec3::Z).try_normalize().unwrap_or(DVec3::ZERO);
        let angle = angle_normalized_3d(normal, DVec3::Z);
        let coords_v2 = coords
            .iter()
            .map(|coord| rotate_to_xy(*coord, axis, angle))
            .collect::<Vec<_>>();
        let point_v2 = rotate_to_xy(point, axis, angle);
        let centroid_v2 = coords_v2.iter().copied().sum::<DVec2>() / count as f64;
        validate_bind_polygon(&coords_v2)?;
        let inside = point_inside_polygon(point_v2, &coords_v2);
        let weight_dist = centroid.distance(point);
        let weight_dist_proj = centroid_v2.distance(point_v2);
        let vert0 = coords_v2[edge_vertex_indices[0]];
        let vert1 = coords_v2[edge_vertex_indices[1]];
        let corner = coords_v2[corner_index];
        let cent_edgemid_vectors =
            [vert0, vert1].map(|vertex| ((vertex + corner) * 0.5 - centroid_v2).normalize_or_zero());
        let scales = [
            distance_to_line_2d(centroid_v2, vert0, corner),
            distance_to_line_2d(centroid_v2, vert1, corner),
        ];
        let edgemid_angle = angle_normalized_2d(cent_edgemid_vectors[0], cent_edgemid_vectors[1]);
        let corner_direction = (corner - centroid_v2).normalize_or_zero();
        let corner_angles =
            cent_edgemid_vectors.map(|vector| angle_signed_2d(corner_direction, vector));
        let corner_edgemid_angles = corner_angles.map(f64::abs);
        if scales.iter().any(|scale| *scale < SURFACE_BIND_EPSILON)
            || edgemid_angle < SURFACE_BIND_EPSILON
            || corner_edgemid_angles
                .iter()
                .any(|angle| *angle < SURFACE_BIND_EPSILON)
        {
            return Err(invalid_polygons());
        }
        let mut infinite = InfiniteWeights::default();
        let mut scale_mid = 0.0;
        let mut point_edgemid_angles = [0.0; 2];
        if weight_dist < SURFACE_BIND_EPSILON {
            infinite.dist_proj = true;
            infinite.dist = true;
        } else if weight_dist_proj < SURFACE_BIND_EPSILON {
            infinite.dist_proj = true;
        } else {
            let point_direction = (point_v2 - centroid_v2).normalize_or_zero();
            let mut point_angles = [0, 1].map(|index| {
                angle_signed_2d(point_direction, cent_edgemid_vectors[index])
                    * sign(corner_angles[index])
            });
            if point_angles[0] <= 0.0 && point_angles[1] <= 0.0 {
                if point_angles[0] < point_angles[1] {
                    point_angles[0] = edgemid_angle - point_angles[1];
                } else {
                    point_angles[1] = edgemid_angle - point_angles[0];
                }
            }
            point_edgemid_angles = point_angles.map(|angle| angle.max(0.0));
            scale_mid = triangle_area_2d(vert0, corner, vert1) / vert0.distance(vert1)
                * std::f64::consts::SQRT_2;
            if inside {
                let min_distance = scales[0].min(scales[1]);
                let factor = (weight_dist_proj / min_distance).min(1.0);
                scale_mid = factor * scale_mid + (1.0 - factor) * (scales[0] + scales[1]) / 2.0;
            }
            if scale_mid < SURFACE_BIND_EPSILON
                || point_edgemid_angles[0] + point_edgemid_angles[1] < SURFACE_BIND_EPSILON
            {
                return Err(invalid_polygons());
            }
        }
        Ok((
            SurfaceBindPolygon {
                face,
                corners,
                corner_edges,
                coords,
                coords_v2,
                point_v2,
                centroid,
                normal,
                edge_vertex_indices,
                edge_indices,
                edgemid_angle,
                corner_edgemid_angles,
                point_edgemid_angles,
                scales,
                scale_mid,
                weight_angular: 1.0,
                weight_dist_proj,
                weight_dist,
                weight: 0.0,
                dominant_edge: 0,
                dominant_angle_weight: 0.0,
                inside,
            },
            infinite,
        ))
    }

    /// Blender's `MOD_SDEF_MODE_NGONS` term for a point inside its polygon.
    fn ngon_bind(&self, polygon: &SurfaceBindPolygon, point: DVec3) -> Result<SurfaceBind> {
        let weights = mean_value_weights(polygon.point_v2, &polygon.coords_v2)
            .ok_or_else(invalid_polygons)?;
        let projected = polygon
            .coords
            .iter()
            .zip(&weights)
            .map(|(coord, weight)| *coord * *weight)
            .sum::<DVec3>();
        Ok(SurfaceBind {
            influence: polygon.weight,
            mode: 1,
            vertex_indices: polygon.corners.clone(),
            vertex_weights: weights,
            normal_distance: signed_normal_distance(point, projected, polygon.normal),
        })
    }

    /// Blender's `MOD_SDEF_MODE_CENTROID` term: the dominant edge plus the polygon centroid.
    fn centroid_bind(&self, polygon: &SurfaceBindPolygon, point: DVec3) -> Result<SurfaceBind> {
        let sorted = sort_polygon_from_edge(
            &polygon.corners,
            &polygon.corner_edges,
            polygon.edge_indices[polygon.dominant_edge],
        )?;
        let triangle = [
            self.points[sorted[0]],
            self.points[sorted[1]],
            polygon.centroid,
        ];
        let (projected, weights) = project_onto_triangle(point, polygon.normal, triangle)?;
        Ok(SurfaceBind {
            influence: polygon.weight * (1.0 - polygon.dominant_angle_weight),
            mode: 2,
            vertex_indices: sorted,
            vertex_weights: weights.to_vec(),
            normal_distance: signed_normal_distance(point, projected, polygon.normal),
        })
    }

    /// Blender's `MOD_SDEF_MODE_CORNER_TRIS` term around the nearest polygon corner.
    fn corner_triangle_bind(
        &self,
        polygon: &SurfaceBindPolygon,
        point: DVec3,
    ) -> Result<SurfaceBind> {
        let sorted = sort_polygon_from_index(&polygon.corners, polygon.edge_vertex_indices[0]);
        let triangle = [
            self.points[sorted[0]],
            self.points[sorted[1]],
            self.points[sorted[2]],
        ];
        let (projected, weights) = project_onto_triangle(point, polygon.normal, triangle)?;
        Ok(SurfaceBind {
            influence: polygon.weight * polygon.dominant_angle_weight,
            mode: 0,
            vertex_indices: sorted,
            vertex_weights: weights.to_vec(),
            normal_distance: signed_normal_distance(point, projected, polygon.normal),
        })
    }
}

/// Binds each source vertex to target polygons as Blender's Surface Deform bind does.
///
/// `target` holds target-local coordinates and `target_to_owner` maps them to the owner's
/// space at bind time; Blender stores that matrix and never updates it at evaluation.
pub(crate) fn bind_surface(
    mesh: &Mesh,
    target: &Mesh,
    target_to_owner: DMat4,
    modifier: &Modifier,
) -> Result<Value> {
    let falloff = number_param(modifier, "falloff")?;
    if !(2.0..=16.0).contains(&falloff) {
        return Err(invalid_parameter(modifier, "falloff", "a number in [2, 16]"));
    }
    let sparse = bool_param(modifier, "use_sparse_bind")?;
    let invert = bool_param(modifier, "invert_vertex_group")?;
    let group = group_weights(mesh, modifier)?;
    let topology = SurfaceBindTarget::new(target, target_to_owner)?;
    let mut bound = Vec::new();
    for (index, vertex) in mesh.vertices.iter().enumerate() {
        // Blender binds with a missing group as weight 0 (then inverted); deformation uses 1.
        let group_weight = group
            .as_ref()
            .and_then(|weights| weights.get(index).copied())
            .unwrap_or(0.0);
        let bind_weight = if invert {
            1.0 - group_weight
        } else {
            group_weight
        };
        if sparse && bind_weight <= 0.0 {
            continue;
        }
        let binds = topology.bind_vertex(vertex.co, falloff)?;
        if sparse && binds.is_empty() {
            continue;
        }
        bound.push(json!({
            "vertex_idx": index,
            "binds": binds.iter().map(SurfaceBind::to_json).collect::<Vec<_>>(),
        }));
    }
    if bound.is_empty() {
        return Err(PotError::invalid_operation("No vertices were bound"));
    }
    let mat: Vec<Vec<f64>> = target_to_owner
        .to_cols_array()
        .chunks(4)
        .map(<[f64]>::to_vec)
        .collect();
    Ok(json!({
        "format": "blender_surface_bind_v1",
        "type": "surface_deform",
        "mesh_verts_num": mesh.vertices.len(),
        "target_verts_num": target.vertices.len(),
        "target_polys_num": target.faces.len(),
        "bind_verts_num": bound.len(),
        "mat": mat,
        "bind_verts": bound,
    }))
}

fn invalid_polygons() -> PotError {
    PotError::invalid_operation("Target contains invalid polygons")
}

/// Blender's `isPolyValid`: convex, non-overlapping, and no adjacent parallel edges.
fn validate_bind_polygon(coords: &[DVec2]) -> Result<()> {
    let count = coords.len();
    let mut positive = false;
    let mut negative = false;
    for index in 0..count {
        let previous = coords[(index + count - 1) % count];
        let current = coords[index];
        let next = coords[(index + 1) % count];
        let cross = (current - previous).perp_dot(next - current);
        positive |= cross > 0.0;
        negative |= cross < 0.0;
    }
    if positive && negative {
        return Err(PotError::invalid_operation(
            "Target contains concave polygons",
        ));
    }
    let mut previous_previous = coords[count - 2];
    let mut previous = coords[count - 1];
    let mut previous_vector = (previous - previous_previous).normalize_or_zero();
    for &current_point in coords {
        let offset = current_point - previous;
        let current_length = offset.length();
        let current_vector = offset.normalize_or_zero();
        if current_length < SURFACE_BIND_EPSILON
            || previous_previous.distance_squared(current_point)
                < SURFACE_BIND_EPSILON * SURFACE_BIND_EPSILON
        {
            return Err(PotError::invalid_operation(
                "Target contains overlapping vertices",
            ));
        }
        if 1.0 - previous_vector.dot(current_vector) < SURFACE_BIND_EPSILON {
            return Err(PotError::invalid_operation(
                "Target contains concave polygons",
            ));
        }
        previous_previous = previous;
        previous = current_point;
        previous_vector = current_vector;
    }
    Ok(())
}

fn point_inside_polygon(point: DVec2, polygon: &[DVec2]) -> bool {
    let mut inside = false;
    for index in 0..polygon.len() {
        let first = polygon[index];
        let second = polygon[(index + 1) % polygon.len()];
        if (first.y > point.y) != (second.y > point.y) {
            let crossing = first.x + (point.y - first.y) * (second.x - first.x) / (second.y - first.y);
            if point.x < crossing {
                inside = !inside;
            }
        }
    }
    inside
}

fn triangle_area_2d(first: DVec2, second: DVec2, third: DVec2) -> f64 {
    0.5 * (second - first).perp_dot(third - first).abs()
}

fn distance_to_line_2d(point: DVec2, first: DVec2, second: DVec2) -> f64 {
    let direction = second - first;
    let length_squared = direction.length_squared();
    if length_squared <= 0.0 {
        return point.distance(first);
    }
    let factor = (point - first).dot(direction) / length_squared;
    point.distance(first + direction * factor)
}

fn distance_squared_to_segment(point: DVec3, first: DVec3, second: DVec3) -> f64 {
    let edge = second - first;
    let length_squared = edge.length_squared();
    let factor = if length_squared > 0.0 {
        ((point - first).dot(edge) / length_squared).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (first + edge * factor).distance_squared(point)
}

fn sign(value: f64) -> f64 {
    if value < 0.0 { -1.0 } else { 1.0 }
}

/// Blender's `angle_normalized_v2v2`, robust for near-parallel and anti-parallel inputs.
fn angle_normalized_2d(first: DVec2, second: DVec2) -> f64 {
    if first.dot(second) >= 0.0 {
        2.0 * half_chord_angle((first - second).length())
    } else {
        PI - 2.0 * half_chord_angle((first + second).length())
    }
}

/// Blender's `angle_normalized_v3v3`, robust for near-parallel and anti-parallel inputs.
fn angle_normalized_3d(first: DVec3, second: DVec3) -> f64 {
    if first.dot(second) >= 0.0 {
        2.0 * half_chord_angle((first - second).length())
    } else {
        PI - 2.0 * half_chord_angle((first + second).length())
    }
}

fn half_chord_angle(chord: f64) -> f64 {
    (chord / 2.0).clamp(-1.0, 1.0).asin()
}

fn angle_signed_2d(first: DVec2, second: DVec2) -> f64 {
    let perpendicular = first.y * second.x - first.x * second.y;
    perpendicular.atan2(first.dot(second))
}

/// Blender's `rotate_normalized_v3_v3v3fl` followed by taking the XY components.
fn rotate_to_xy(point: DVec3, axis: DVec3, angle: f64) -> DVec2 {
    let (sin, cos) = angle.sin_cos();
    let rotated = point * cos + axis.cross(point) * sin + axis * (axis.dot(point) * (1.0 - cos));
    DVec2::new(rotated.x, rotated.y)
}

fn angular_weight(point_angle: f64, edgemid_angle: f64) -> f64 {
    ((point_angle / edgemid_angle).min(1.0) * FRAC_PI_2).sin()
}

/// Blender's `computeNormalDisplacement`: signed distance along the polygon normal.
fn signed_normal_distance(point: DVec3, projected: DVec3, normal: DVec3) -> f64 {
    let offset = point - projected;
    let distance = offset.length();
    if offset.dot(normal) < 0.0 {
        -distance
    } else {
        distance
    }
}

/// Blender's `isect_line_plane_v3` along the polygon normal, then barycentric weights.
fn project_onto_triangle(
    point: DVec3,
    direction: DVec3,
    triangle: [DVec3; 3],
) -> Result<(DVec3, [f64; 3])> {
    let [first, second, third] = triangle;
    let normal = (first - second)
        .cross(second - third)
        .try_normalize()
        .ok_or_else(invalid_polygons)?;
    let centre = (first + second + third) / 3.0;
    let slope = normal.dot(direction);
    if slope.abs() <= f64::from(f32::EPSILON) {
        return Err(invalid_polygons());
    }
    let lambda = -normal.dot(point - centre) / slope;
    let projected = point + direction * lambda;
    Ok((projected, triangle_barycentric(triangle, projected)))
}

/// Blender's `barycentric_weights` (area ratios on the triangle plane).
fn triangle_barycentric(triangle: [DVec3; 3], point: DVec3) -> [f64; 3] {
    let [first, second, third] = triangle;
    let normal = (first - second).cross(second - third);
    let area_first = (third - second).cross(point - second).dot(normal);
    let area_second = (first - third).cross(point - third).dot(normal);
    let area_third = (second - first).cross(point - first).dot(normal);
    let total = area_first + area_second + area_third;
    if total == 0.0 || !total.is_finite() {
        return [1.0 / 3.0; 3];
    }
    [area_first / total, area_second / total, area_third / total]
}

/// Blender's `sortPolyVertsEdge`: loop vertices starting at the edge's first corner.
fn sort_polygon_from_edge(corners: &[usize], corner_edges: &[usize], edge: usize) -> Result<Vec<usize>> {
    let start = corner_edges
        .iter()
        .position(|&candidate| candidate == edge)
        .ok_or_else(invalid_polygons)?;
    Ok(corners[start..]
        .iter()
        .chain(&corners[..start])
        .copied()
        .collect())
}

/// Blender's `sortPolyVertsTri`: loop vertices starting at `start`.
fn sort_polygon_from_index(corners: &[usize], start: usize) -> Vec<usize> {
    corners[start..]
        .iter()
        .chain(&corners[..start])
        .copied()
        .collect()
}

/// Binds source vertices to Blender-style cage-local harmonic coordinates. `cage_to_owner` is
/// the bind-time map from cage-local coordinates into subject-local coordinates.
pub(crate) fn bind_mesh(
    mesh: &Mesh,
    cage: &Mesh,
    cage_to_owner: DMat4,
    modifier: &Modifier,
) -> Result<Value> {
    if cage.vertices.len() < 4 {
        return Err(PotError::invalid_operation(
            "mesh deform binding requires at least four cage vertices",
        ));
    }
    let owner_to_cage = cage_to_owner.inverse();
    if !cage_to_owner.is_finite() || !owner_to_cage.is_finite() {
        return Err(PotError::invalid_argument(
            "mesh deform binding requires an invertible cage transform",
        ));
    }
    let precision = uint_param(modifier, "precision")?;
    if !(2..=10).contains(&precision) {
        return Err(invalid_parameter(
            modifier,
            "precision",
            "an integer from 2 to 10",
        ));
    }
    let precision = u32::try_from(precision)
        .map_err(|_| invalid_parameter(modifier, "precision", "an integer from 2 to 10"))?;
    let dynamic = bool_param(modifier, "use_dynamic_bind")?;
    let mut source_cage_space = mesh.clone();
    for vertex in &mut source_cage_space.vertices {
        vertex.co = owner_to_cage.transform_point3(vertex.co);
    }
    let weights = harmonic_mesh_bind(&source_cage_space, cage, precision)?;
    let mut cage_owner = cage.clone();
    for vertex in &mut cage_owner.vertices {
        vertex.co = cage_to_owner.transform_point3(vertex.co);
    }
    let mat: Vec<Vec<f64>> = cage_to_owner
        .to_cols_array()
        .chunks(4)
        .map(<[f64]>::to_vec)
        .collect();
    Ok(json!({
        "weights": weights,
        "cage_mesh": cage_owner,
        "cage_local_mesh": cage,
        "dynamic_bind": dynamic,
        "mat": mat,
        "precision": precision
    }))
}

#[derive(Clone, Copy)]
struct BoundaryIntersection {
    vertices: [usize; 3],
    weights: [f64; 3],
    distance: f64,
}

#[derive(Clone, Copy)]
struct BoundaryWeights {
    intersections: [Option<BoundaryIntersection>; 6],
}

fn harmonic_mesh_bind(source: &Mesh, cage: &Mesh, precision: u32) -> Result<Value> {
    let triangles = cage_triangles(cage)?;
    let bounds = cage
        .bounds()
        .ok_or_else(|| PotError::invalid_operation("mesh deform cage has no bounds"))?;
    let extent = bounds.size();
    if extent.min_element() <= f64::EPSILON {
        return Err(PotError::invalid_operation(
            "mesh deform cage must enclose a three-dimensional region",
        ));
    }
    let precision_cells = 1_usize.checked_shl(precision).ok_or_else(|| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "Mesh Deform precision exceeds grid limits",
        )
    })?;
    let cells = precision_cells.checked_add(2).ok_or_else(|| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "Mesh Deform precision exceeds grid limits",
        )
    })?;
    let side = cells;
    let node_count = side
        .checked_mul(side)
        .and_then(|count| count.checked_mul(side))
        .ok_or_else(|| PotError::invalid_argument("mesh deform voxel grid exceeds limits"))?;
    let weight_count = source
        .vertices
        .len()
        .checked_mul(cage.vertices.len())
        .ok_or_else(|| PotError::invalid_argument("mesh deform bind data exceeds limits"))?;
    if node_count > 274_625 || weight_count > 10_000_000 {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            format!(
                "Mesh Deform harmonic bind exceeds resource budget: {node_count} grid nodes, {weight_count} weights"
            ),
        ));
    }
    let sample_work = node_count
        .checked_mul(triangles.len())
        .and_then(|work| work.checked_mul(6))
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "Mesh Deform sampling work exceeds limits",
            )
        })?;
    if sample_work > 50_000_000 {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            format!("Mesh Deform harmonic binding requires {sample_work} triangle checks"),
        ));
    }
    let maximum_width = extent.max_element();
    let center = (bounds.min + bounds.max) * 0.5;
    let padding = maximum_width / (cells - 4) as f64 * 2.1;
    let grid_min = center - DVec3::splat(maximum_width * 0.5 + padding);
    let grid_extent = DVec3::splat(maximum_width + 2.0 * padding);
    let spacing = grid_extent / cells as f64;
    let mut boundary = vec![None; node_count];
    for z in 0..cells {
        for y in 0..cells {
            for x in 0..cells {
                let node = grid_index(x, y, z, side);
                let start = grid_position(x, y, z, cells, grid_min, grid_extent);
                let neighbors = [
                    (x > 0).then(|| (x - 1, y, z)),
                    (x + 1 < cells).then(|| (x + 1, y, z)),
                    (y > 0).then(|| (x, y - 1, z)),
                    (y + 1 < cells).then(|| (x, y + 1, z)),
                    (z > 0).then(|| (x, y, z - 1)),
                    (z + 1 < cells).then(|| (x, y, z + 1)),
                ];
                let mut intersections = [None; 6];
                for (direction, neighbor) in neighbors.into_iter().enumerate() {
                    let Some((neighbor_x, neighbor_y, neighbor_z)) = neighbor else {
                        continue;
                    };
                    let end = grid_position(
                        neighbor_x,
                        neighbor_y,
                        neighbor_z,
                        cells,
                        grid_min,
                        grid_extent,
                    );
                    let mut nearest = None;
                    for triangle in &triangles {
                        let points = triangle.map(|index| cage.vertices[index].co);
                        if let Some((fraction, weights)) =
                            segment_triangle_intersection(start, end, points)
                            && nearest
                                .is_none_or(|(nearest_fraction, _, _)| fraction < nearest_fraction)
                        {
                            nearest = Some((fraction, *triangle, weights));
                        }
                    }
                    if let Some((fraction, vertices, weights)) = nearest {
                        intersections[direction] = Some(BoundaryIntersection {
                            vertices,
                            weights,
                            distance: (end - start).length() * fraction,
                        });
                    }
                }
                if intersections.iter().any(Option::is_some) {
                    boundary[node] = Some(BoundaryWeights { intersections });
                }
            }
        }
    }
    let mut outside = vec![false; node_count];
    let mut stack = vec![0];
    outside[0] = true;
    while let Some(node) = stack.pop() {
        let x = node % side;
        let y = (node / side) % side;
        let z = node / (side * side);
        let neighbors = [
            (x > 0).then(|| (x - 1, y, z)),
            (x + 1 < cells).then(|| (x + 1, y, z)),
            (y > 0).then(|| (x, y - 1, z)),
            (y + 1 < cells).then(|| (x, y + 1, z)),
            (z > 0).then(|| (x, y, z - 1)),
            (z + 1 < cells).then(|| (x, y, z + 1)),
        ];
        for (direction, neighbor) in neighbors.into_iter().enumerate() {
            let Some((neighbor_x, neighbor_y, neighbor_z)) = neighbor else {
                continue;
            };
            let neighbor_node = grid_index(neighbor_x, neighbor_y, neighbor_z, side);
            let blocked =
                boundary[node].is_some_and(|sample| sample.intersections[direction].is_some());
            if !blocked && !outside[neighbor_node] {
                outside[neighbor_node] = true;
                stack.push(neighbor_node);
            }
        }
    }
    let mut inside = vec![false; node_count];
    let mut inside_nodes = Vec::new();
    for node in 0..node_count {
        if !outside[node] {
            inside[node] = true;
            inside_nodes.push(node);
        }
    }
    if inside_nodes.is_empty() {
        return Err(PotError::invalid_operation(
            "mesh deform voxel grid did not intersect its cage",
        ));
    }
    let mut rows = Vec::with_capacity(source.vertices.len());
    let mut inside_source_indices = Vec::new();
    for (index, vertex) in source.vertices.iter().enumerate() {
        if let Some(cage_index) = cage
            .vertices
            .iter()
            .position(|cage_vertex| cage_vertex.co.distance_squared(vertex.co) <= f64::EPSILON)
        {
            let mut row = vec![0.0; cage.vertices.len()];
            row[cage_index] = 1.0;
            rows.push(row);
        } else if point_inside_mesh(vertex.co, cage, &triangles) {
            rows.push(vec![0.0; cage.vertices.len()]);
            inside_source_indices.push(index);
        } else {
            rows.push(vec![0.0; cage.vertices.len()]);
        }
    }
    let mut field = vec![0.0; node_count];
    let iteration_limit = (cells * 12).clamp(120, 600);
    for (cage_index, _) in cage.vertices.iter().enumerate() {
        field.fill(0.0);
        for _ in 0..iteration_limit {
            let mut maximum_change: f64 = 0.0;
            for node in &inside_nodes {
                let x = node % side;
                let y = (node / side) % side;
                let z = node / (side * side);
                let neighbors = [
                    (x > 0).then(|| node - 1),
                    (x + 1 < cells).then(|| node + 1),
                    (y > 0).then(|| node - side),
                    (y + 1 < cells).then(|| node + side),
                    (z > 0).then(|| node - side * side),
                    (z + 1 < cells).then(|| node + side * side),
                ];
                let mut total_weight = 0.0;
                let mut value = 0.0;
                for (direction, intersection) in boundary[*node]
                    .map_or([None; 6], |sample| sample.intersections)
                    .into_iter()
                    .enumerate()
                {
                    if let Some(intersection) = intersection {
                        let weight = 1.0 / intersection.distance.max(1.0e-4);
                        let boundary_value = intersection
                            .vertices
                            .iter()
                            .zip(intersection.weights)
                            .filter(|(vertex, _)| **vertex == cage_index)
                            .map(|(_, weight)| weight)
                            .sum::<f64>();
                        total_weight += weight;
                        value += weight * boundary_value;
                    } else {
                        let weight = 1.0 / spacing.x;
                        total_weight += weight;
                        if let Some(neighbor) = neighbors[direction]
                            && inside[neighbor]
                        {
                            value += weight * field[neighbor];
                        }
                    }
                }
                value /= total_weight;
                let change = value - field[*node];
                field[*node] += change * 1.35;
                maximum_change = maximum_change.max(change.abs());
            }
            if maximum_change < 1.0e-7 {
                break;
            }
        }
        for source_index in &inside_source_indices {
            rows[*source_index][cage_index] = sample_harmonic_field(
                &field,
                &inside,
                source.vertices[*source_index].co,
                cells,
                side,
                grid_min,
                grid_extent,
            );
        }
    }
    for source_index in inside_source_indices {
        let total = rows[source_index].iter().sum::<f64>();
        if total > f64::EPSILON {
            for weight in &mut rows[source_index] {
                *weight /= total;
            }
        } else {
            rows[source_index] = affine_weights(source.vertices[source_index].co, cage, precision)?;
        }
    }
    Ok(Value::Array(
        rows.into_iter().map(|weights| json!(weights)).collect(),
    ))
}

fn cage_triangles(cage: &Mesh) -> Result<Vec<[usize; 3]>> {
    let indices = index_positions(cage);
    let mut edges = HashMap::<(u32, u32), u32>::new();
    let mut triangles = Vec::new();
    for face in &cage.faces {
        let vertices = face
            .vertices
            .iter()
            .map(|vertex| indices.get(vertex).copied())
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                PotError::invalid_operation("mesh deform cage contains invalid face vertices")
            })?;
        for index in 0..face.vertices.len() {
            let first = face.vertices[index];
            let second = face.vertices[(index + 1) % face.vertices.len()];
            *edges
                .entry((first.min(second), first.max(second)))
                .or_default() += 1;
        }
        for index in 1..vertices.len().saturating_sub(1) {
            triangles.push([vertices[0], vertices[index], vertices[index + 1]]);
        }
    }
    if triangles.is_empty() || edges.values().any(|uses| *uses != 2) {
        return Err(PotError::invalid_operation(
            "mesh deform harmonic binding requires a closed two-manifold cage",
        ));
    }
    Ok(triangles)
}

pub(super) fn point_inside_mesh(point: DVec3, cage: &Mesh, triangles: &[[usize; 3]]) -> bool {
    let direction = DVec3::new(1.0, 0.371, 0.217).normalize();
    let mut intersections = 0;
    for triangle in triangles {
        let first = cage.vertices[triangle[0]].co;
        let edge_one = cage.vertices[triangle[1]].co - first;
        let edge_two = cage.vertices[triangle[2]].co - first;
        let cross = direction.cross(edge_two);
        let determinant = edge_one.dot(cross);
        if determinant.abs() <= 1.0e-12 {
            continue;
        }
        let inverse = 1.0 / determinant;
        let offset = point - first;
        let u = offset.dot(cross) * inverse;
        if !(0.0..=1.0).contains(&u) {
            continue;
        }
        let q = offset.cross(edge_one);
        let v = direction.dot(q) * inverse;
        if v < 0.0 || u + v > 1.0 {
            continue;
        }
        let distance = edge_two.dot(q) * inverse;
        if distance > 1.0e-10 {
            intersections += 1;
        }
    }
    intersections % 2 == 1
}

fn sample_harmonic_field(
    field: &[f64],
    inside: &[bool],
    point: DVec3,
    cells: usize,
    side: usize,
    minimum: DVec3,
    extent: DVec3,
) -> f64 {
    let coordinates = (point - minimum) / extent * cells as f64 - DVec3::splat(0.5);
    let lower = [
        (coordinates.x.floor() as usize).min(cells - 1),
        (coordinates.y.floor() as usize).min(cells - 1),
        (coordinates.z.floor() as usize).min(cells - 1),
    ];
    let fractions = [
        coordinates.x - lower[0] as f64,
        coordinates.y - lower[1] as f64,
        coordinates.z - lower[2] as f64,
    ];
    let mut result = 0.0;
    for z in 0..=1 {
        for y in 0..=1 {
            for x in 0..=1 {
                let index = grid_index(
                    (lower[0] + x).min(cells - 1),
                    (lower[1] + y).min(cells - 1),
                    (lower[2] + z).min(cells - 1),
                    side,
                );
                if !inside[index] {
                    continue;
                }
                let wx = if x == 0 {
                    1.0 - fractions[0]
                } else {
                    fractions[0]
                };
                let wy = if y == 0 {
                    1.0 - fractions[1]
                } else {
                    fractions[1]
                };
                let wz = if z == 0 {
                    1.0 - fractions[2]
                } else {
                    fractions[2]
                };
                result += field[index] * wx * wy * wz;
            }
        }
    }
    result
}

fn grid_index(x: usize, y: usize, z: usize, side: usize) -> usize {
    z * side * side + y * side + x
}

fn grid_position(
    x: usize,
    y: usize,
    z: usize,
    cells: usize,
    minimum: DVec3,
    extent: DVec3,
) -> DVec3 {
    minimum
        + extent
            * DVec3::new(
                (x as f64 + 0.5) / cells as f64,
                (y as f64 + 0.5) / cells as f64,
                (z as f64 + 0.5) / cells as f64,
            )
}
fn segment_triangle_intersection(
    start: DVec3,
    end: DVec3,
    [a, b, c]: [DVec3; 3],
) -> Option<(f64, [f64; 3])> {
    let direction = end - start;
    let first_edge = b - a;
    let second_edge = c - a;
    let cross = direction.cross(second_edge);
    let determinant = first_edge.dot(cross);
    if determinant.abs() <= 1.0e-12 {
        return None;
    }
    let inverse = determinant.recip();
    let offset = start - a;
    let first_weight = offset.dot(cross) * inverse;
    if !(-1.0e-10..=1.0 + 1.0e-10).contains(&first_weight) {
        return None;
    }
    let cross = offset.cross(first_edge);
    let second_weight = direction.dot(cross) * inverse;
    if second_weight < -1.0e-10 || first_weight + second_weight > 1.0 + 1.0e-10 {
        return None;
    }
    let fraction = second_edge.dot(cross) * inverse;
    if !(0.0..=1.0).contains(&fraction) {
        return None;
    }
    Some((
        fraction,
        [
            1.0 - first_weight - second_weight,
            first_weight,
            second_weight,
        ],
    ))
}
/// Stores source coordinates for a Laplacian Deform bind.
pub(crate) fn bind_laplacian(mesh: &Mesh) -> Value {
    Value::Array(
        mesh.vertices
            .iter()
            .map(|vertex| json!(vertex.co.to_array()))
            .collect(),
    )
}

fn affine_weights(point: DVec3, cage: &Mesh, precision: u32) -> Result<Vec<f64>> {
    let mut weights = Vec::with_capacity(cage.vertices.len());
    affine_weights_into(point, cage, precision, &mut weights)?;
    Ok(weights)
}

fn affine_weights_into(
    point: DVec3,
    cage: &Mesh,
    precision: u32,
    weights: &mut Vec<f64>,
) -> Result<()> {
    weights.clear();
    weights.resize(cage.vertices.len(), 0.0);
    let mut normal = [[0.0; 4]; 4];
    for (index, vertex) in cage.vertices.iter().enumerate() {
        let delta = point.distance_squared(vertex.co);
        if delta <= f64::EPSILON {
            weights[index] = 1.0;
            return Ok(());
        }
        let values = [1.0, vertex.co.x, vertex.co.y, vertex.co.z];
        let kernel = 1.0 / delta;
        for row in 0..4 {
            for column in 0..4 {
                normal[row][column] += kernel * values[row] * values[column];
            }
        }
    }
    let multipliers =
        solve_four(normal, [1.0, point.x, point.y, point.z], precision).ok_or_else(|| {
            PotError::invalid_operation("mesh deform cage must span three dimensions")
        })?;
    for (index, vertex) in cage.vertices.iter().enumerate() {
        let delta = point.distance_squared(vertex.co);
        let kernel = 1.0 / delta;
        let values = [1.0, vertex.co.x, vertex.co.y, vertex.co.z];
        weights[index] = kernel
            * (0..4)
                .map(|coordinate| values[coordinate] * multipliers[coordinate])
                .sum::<f64>();
    }
    Ok(())
}

fn solve_four(mut matrix: [[f64; 4]; 4], mut rhs: [f64; 4], precision: u32) -> Option<[f64; 4]> {
    for column in 0..4 {
        let pivot = (column..4).max_by(|left, right| {
            matrix[*left][column]
                .abs()
                .total_cmp(&matrix[*right][column].abs())
        })?;
        if matrix[pivot][column].abs() <= 10_f64.powi(-(precision as i32 + 4)) {
            return None;
        }
        matrix.swap(column, pivot);
        rhs.swap(column, pivot);
        let diagonal = matrix[column][column];
        for value in matrix[column].iter_mut().skip(column) {
            *value /= diagonal;
        }
        rhs[column] /= diagonal;
        let pivot_row = matrix[column];
        for (row, matrix_row) in matrix.iter_mut().enumerate() {
            if row == column {
                continue;
            }
            let factor = matrix_row[column];
            for (value, pivot_value) in matrix_row.iter_mut().zip(pivot_row).skip(column) {
                *value -= factor * pivot_value;
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    rhs.iter().all(|value| value.is_finite()).then_some(rhs)
}

pub(super) fn closest_triangle_point(point: DVec3, triangle: [DVec3; 3]) -> (DVec3, [f64; 3]) {
    let [a, b, c] = triangle;
    let ab = b - a;
    let ac = c - a;
    let ap = point - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, [1.0, 0.0, 0.0]);
    }
    let bp = point - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (b, [0.0, 1.0, 0.0]);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (a + ab * v, [1.0 - v, v, 0.0]);
    }
    let cp = point - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, [0.0, 0.0, 1.0]);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (a + ac * w, [1.0 - w, 0.0, w]);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (b + (c - b) * w, [0.0, 1.0 - w, w]);
    }
    let inverse = 1.0 / (va + vb + vc);
    let v = vb * inverse;
    let w = vc * inverse;
    (a + ab * v + ac * w, [1.0 - v - w, v, w])
}

fn polygon_normal(points: &[DVec3]) -> Option<DVec3> {
    if points.len() < 3 {
        return None;
    }
    let mut normal = DVec3::ZERO;
    for index in 0..points.len() {
        normal += points[index].cross(points[(index + 1) % points.len()]);
    }
    normal.try_normalize()
}

fn mean_value_weights(point: DVec2, vertices: &[DVec2]) -> Option<Vec<f64>> {
    if vertices.len() < 3 {
        return None;
    }
    let mut winding: f64 = 0.0;
    for index in 0..vertices.len() {
        let first = vertices[index];
        let second = vertices[(index + 1) % vertices.len()];
        let third = vertices[(index + 2) % vertices.len()];
        let left = second - first;
        let right = third - second;
        let cross = left.x * right.y - left.y * right.x;
        if cross.abs() > 1.0e-12 {
            if winding.abs() <= f64::EPSILON {
                winding = cross.signum();
            } else if cross.signum() * winding < 0.0 {
                return None;
            }
        }
    }
    if winding.abs() <= f64::EPSILON {
        return None;
    }
    let vectors = vertices
        .iter()
        .map(|vertex| *vertex - point)
        .collect::<Vec<_>>();
    let distances = vectors
        .iter()
        .map(|vector| vector.length())
        .collect::<Vec<_>>();
    for (index, distance) in distances.iter().enumerate() {
        if *distance <= 1.0e-12 {
            let mut weights = vec![0.0; vertices.len()];
            weights[index] = 1.0;
            return Some(weights);
        }
    }
    let mut half_angle_tangents = Vec::with_capacity(vertices.len());
    for index in 0..vertices.len() {
        let next = (index + 1) % vertices.len();
        let left = vectors[index];
        let right = vectors[next];
        let denominator = distances[index] * distances[next] + left.dot(right);
        if denominator.abs() <= 1.0e-12 {
            return None;
        }
        let cross = left.x * right.y - left.y * right.x;
        half_angle_tangents.push(cross / denominator);
    }
    let mut weights = Vec::with_capacity(vertices.len());
    for index in 0..vertices.len() {
        let previous = (index + vertices.len() - 1) % vertices.len();
        weights
            .push((half_angle_tangents[previous] + half_angle_tangents[index]) / distances[index]);
    }
    let total = weights.iter().sum::<f64>();
    if !total.is_finite() || total.abs() <= 1.0e-12 {
        return None;
    }
    for weight in &mut weights {
        *weight /= total;
    }
    Some(weights)
}

fn finite_number(value: &Value, modifier: &Modifier, name: &str) -> Result<f64> {
    value
        .as_f64()
        .filter(|number| number.is_finite())
        .ok_or_else(|| invalid_parameter(modifier, name, "finite numbers"))
}

fn group_weights(mesh: &Mesh, modifier: &Modifier) -> Result<Option<Vec<f64>>> {
    let group_name =
        super::string_param_with_expected(modifier, "vertex_group", "a vertex-group name")?;
    if group_name.is_empty() {
        return Ok(None);
    }
    let Some(weights) = mesh
        .attributes
        .get("vertex_groups")
        .and_then(Value::as_object)
        .and_then(|groups| groups.get(group_name))
        .and_then(Value::as_object)
    else {
        return Ok(None);
    };
    mesh.vertices
        .iter()
        .map(|vertex| {
            let weight = weights
                .get(&format!("v{}", vertex.id))
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            if weight.is_finite() && (0.0..=1.0).contains(&weight) {
                Ok(weight)
            } else {
                Err(invalid_parameter(
                    modifier,
                    "vertex_group",
                    "finite weights in [0, 1]",
                ))
            }
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

#[cfg(test)]
mod tests {
    use super::{
        affine_weights, closest_triangle_point, laplacian_deform, mean_value_weights, mesh_deform,
        surface_deform,
    };
    use crate::{
        geom::{BoxParams, Edge, Face, Mesh, Vertex},
        model::{Id, Modifier, ModifierRuntime},
    };
    use glam::{DMat4, DVec2, DVec3};
    use serde_json::{Map, Value, json};

    fn single_vertex_mesh(co: DVec3) -> Mesh {
        Mesh {
            vertices: vec![Vertex { id: 0, co }],
            next_id: crate::geom::IdCounters {
                vertex: 1,
                ..crate::geom::IdCounters::default()
            },
            ..Mesh::default()
        }
    }

    fn native_modifier(modifier_type: &str, binding_data: Value) -> crate::error::Result<Modifier> {
        Ok(Modifier {
            id: Id::new("native_test")?,
            modifier_type: modifier_type.to_owned(),
            name: "Native Test".to_owned(),
            enabled: true,
            show_viewport: true,
            show_render: true,
            params: Map::new(),
            binding_data: Some(binding_data),
            runtime: ModifierRuntime::default(),
        })
    }

    fn identity_rows() -> Value {
        json!([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
    }
    fn translated_rows() -> Value {
        json!([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0]
        ])
    }
    fn triangle_mesh(z: f64) -> Mesh {
        Mesh {
            vertices: vec![
                Vertex {
                    id: 0,
                    co: DVec3::new(0.0, 0.0, z),
                },
                Vertex {
                    id: 1,
                    co: DVec3::new(1.0, 0.0, z),
                },
                Vertex {
                    id: 2,
                    co: DVec3::new(0.0, 1.0, z),
                },
            ],
            edges: vec![
                Edge {
                    id: 0,
                    vertices: [0, 1],
                },
                Edge {
                    id: 1,
                    vertices: [1, 2],
                },
                Edge {
                    id: 2,
                    vertices: [2, 0],
                },
            ],
            faces: vec![Face {
                id: 0,
                vertices: vec![0, 1, 2],
                material_index: 0,
            }],
            next_id: crate::geom::IdCounters {
                vertex: 3,
                edge: 3,
                face: 1,
            },
            ..Mesh::default()
        }
    }

    #[test]
    fn native_surface_deform_uses_blender_bind_weights_and_normal_offset()
    -> Result<(), Box<dyn std::error::Error>> {
        let source = triangle_mesh(1.0);
        let target = triangle_mesh(2.0);
        let bindings = (0..3)
            .map(|vertex_idx| {
                json!({
                    "vertex_idx": vertex_idx,
                    "binds": [
                        {
                            "vert_inds": [0, 1, 2],
                            "verts_num": 3,
                            "mode": 0,
                            "vert_weights": [1.0, 0.0, 0.0],
                            "normal_dist": 1.0,
                            "influence": 0.333_333_333_333_333_3
                        },
                        {
                            "vert_inds": [0, 1, 2],
                            "verts_num": 3,
                            "mode": 1,
                            "vert_weights": [1.0, 0.0, 0.0],
                            "normal_dist": 1.0,
                            "influence": 0.333_333_333_333_333_3
                        },
                        {
                            "vert_inds": [0, 1, 2],
                            "verts_num": 3,
                            "mode": 2,
                            "vert_weights": [1.0, 0.0, 0.0],
                            "normal_dist": 1.0,
                            "influence": 0.333_333_333_333_333_3
                        }
                    ]
                })
            })
            .collect::<Vec<_>>();
        let binding = json!({
            "format": "blender_native_bind_v1",
            "type": "surface_deform",
            "mesh_verts_num": 3,
            "target_verts_num": 3,
            "strength": 1.0,
            "bind_verts": bindings
        });
        let mut modifier = native_modifier("surface_deform", binding)?;
        modifier.runtime.target_mesh = Some(serde_json::to_value(target)?);
        let evaluated = surface_deform(&source, &modifier)?;
        for vertex in evaluated.vertices {
            assert!((vertex.co.z - 3.0).abs() < 1.0e-12);
        }
        Ok(())
    }

    #[test]
    fn native_surface_deform_skips_deformation_when_target_counts_change()
    -> Result<(), Box<dyn std::error::Error>> {
        let source = triangle_mesh(1.0);
        let target = triangle_mesh(2.0);
        for (target_vertices, target_polygons) in [(4, 1), (3, 2)] {
            let binding = json!({
                "format": "blender_native_bind_v1",
                "type": "surface_deform",
                "mesh_verts_num": 3,
                "target_verts_num": target_vertices,
                "target_polys_num": target_polygons,
                "bind_verts": []
            });
            let mut modifier = native_modifier("surface_deform", binding)?;
            modifier.runtime.target_mesh = Some(serde_json::to_value(&target)?);
            let evaluated = crate::geom::modifiers::evaluate_modifiers_with_groups(
                &source,
                &[modifier],
                &std::collections::BTreeMap::new(),
            )?;
            assert_eq!(evaluated.mesh, source);
            let warning = evaluated
                .warnings
                .first()
                .ok_or("changed native target counts did not produce a warning")?;
            assert_eq!(warning.code, crate::error::ErrorCode::EvaluationFailed);
            assert_eq!(
                warning.details["feature_id"],
                "modifier.surface_deform.target_topology_changed"
            );
        }
        Ok(())
    }

    #[test]
    fn native_mesh_deform_applies_sparse_weights_in_bind_space()
    -> Result<(), Box<dyn std::error::Error>> {
        let source = single_vertex_mesh(DVec3::ZERO);
        let cage = single_vertex_mesh(DVec3::new(3.0, 0.0, 0.0));
        let binding = json!({
            "format": "blender_native_bind_v1",
            "type": "mesh_deform",
            "verts_num": 1,
            "cage_verts_num": 1,
            "flag": 0,
            "bind_offsets": [0, 1],
            "bind_influences": [{"vertex": 0, "weight": 1.0}],
            "bind_cage_coordinates": [1.0, 0.0, 0.0],
            "bind_matrix": translated_rows()
        });
        let mut modifier = native_modifier("mesh_deform", binding)?;
        modifier.runtime.target_mesh = Some(serde_json::to_value(cage)?);
        modifier.runtime.target_to_subject =
            Some(DMat4::from_translation(DVec3::new(2.0, 0.0, 0.0)).to_cols_array());
        let evaluated = mesh_deform(&source, &modifier)?;
        assert!(evaluated.vertices[0].co.distance(DVec3::X) < 1.0e-12);
        Ok(())
    }

    #[test]
    fn native_dynamic_mesh_deform_interpolates_stored_grid_influences()
    -> Result<(), Box<dyn std::error::Error>> {
        let source = single_vertex_mesh(DVec3::ZERO);
        let cage = single_vertex_mesh(DVec3::X);
        let binding = json!({
            "format": "blender_native_bind_v1",
            "type": "mesh_deform",
            "verts_num": 1,
            "cage_verts_num": 1,
            "flag": 2,
            "dynamic_bind": true,
            "bind_cage_coordinates": [0.0, 0.0, 0.0],
            "bind_matrix": identity_rows(),
            "dynamic_grid_size": 1,
            "dynamic_cell_min": [-1.0, -1.0, -1.0],
            "dynamic_cell_width": 2.0,
            "dynamic_grid": [{"offset": 0, "influences_num": 1}],
            "dynamic_influences": [{"vertex": 0, "weight": 1.0}],
            "dynamic_vertices": [1]
        });
        let mut modifier = native_modifier("mesh_deform", binding)?;
        modifier.runtime.target_mesh = Some(serde_json::to_value(cage)?);
        let evaluated = mesh_deform(&source, &modifier)?;
        assert!(evaluated.vertices[0].co.distance(DVec3::X) < 1.0e-12);
        Ok(())
    }

    #[test]
    fn native_laplacian_deform_uses_stored_bind_coordinates()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut mesh = Mesh::box_mesh(BoxParams::default())?;
        let bind_coordinates = mesh
            .vertices
            .iter()
            .flat_map(|vertex| vertex.co.to_array())
            .map(Value::from)
            .collect::<Vec<_>>();
        mesh.vertices[0].co += DVec3::new(0.2, -0.1, 0.4);
        let anchor_position = mesh.vertices[0].co;
        mesh.attributes
            .insert("vertex_groups".to_owned(), json!({"Anchors": {"v0": 1.0}}));
        let mut modifier = native_modifier(
            "laplacian_deform",
            json!({
                "format": "blender_native_bind_v1",
                "type": "laplacian_deform",
                "verts_num": mesh.vertices.len(),
                "vertex_coordinates": bind_coordinates
            }),
        )?;
        modifier.params = Map::from_iter([
            ("vertex_group".to_owned(), json!("Anchors")),
            ("iterations".to_owned(), json!(1)),
        ]);
        let evaluated = laplacian_deform(&mesh, &modifier)?;
        assert!(evaluated.vertices[0].co.distance(anchor_position) < 1.0e-9);
        assert!(evaluated.vertices[1].co.distance(mesh.vertices[1].co) > 1.0e-6);
        Ok(())
    }

    #[test]
    fn affine_cage_weights_reproduce_points_and_affine_transforms()
    -> Result<(), Box<dyn std::error::Error>> {
        let cage = Mesh::box_mesh(BoxParams::default())?;
        let point = DVec3::new(0.2, -0.1, 0.4);
        let weights = affine_weights(point, &cage, 5)?;
        let reconstructed = cage
            .vertices
            .iter()
            .zip(&weights)
            .map(|(vertex, weight)| vertex.co * *weight)
            .sum::<DVec3>();
        assert!(reconstructed.distance(point) < 1.0e-10);
        let translation = DVec3::new(1.0, 2.0, -0.5);
        let moved = cage
            .vertices
            .iter()
            .zip(&weights)
            .map(|(vertex, weight)| (vertex.co + translation) * *weight)
            .sum::<DVec3>();
        assert!(moved.distance(point + translation) < 1.0e-10);
        Ok(())
    }

    #[test]
    fn closest_triangle_returns_barycentric_projection() {
        let triangle = [DVec3::ZERO, DVec3::X, DVec3::Y];
        let (point, weights) = closest_triangle_point(DVec3::new(0.2, 0.3, 2.0), triangle);
        assert!(point.distance(DVec3::new(0.2, 0.3, 0.0)) < 1.0e-12);
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1.0e-12);
    }
    #[test]
    fn polygon_mean_value_weights_preserve_constant_and_linear_fields()
    -> Result<(), Box<dyn std::error::Error>> {
        let polygon = [
            DVec2::new(-1.0, -1.0),
            DVec2::new(1.0, -1.0),
            DVec2::new(1.0, 1.0),
            DVec2::new(-1.0, 1.0),
        ];
        let weights = mean_value_weights(DVec2::new(0.25, -0.5), &polygon)
            .ok_or_else(|| std::io::Error::other("mean-value coordinate bind failed"))?;
        assert_eq!(weights.len(), polygon.len());
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1.0e-12);
        let reconstructed = polygon
            .iter()
            .zip(&weights)
            .map(|(point, weight)| *point * *weight)
            .sum::<DVec2>();
        assert!(reconstructed.distance(DVec2::new(0.25, -0.5)) < 1.0e-12);
        Ok(())
    }
}
