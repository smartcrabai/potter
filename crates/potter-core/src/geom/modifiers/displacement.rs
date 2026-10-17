// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   source/blender/modifiers/intern/MOD_displace.cc, MOD_wave.cc, MOD_warp.cc, MOD_util.cc,
//   SPDX-FileCopyrightText: 2005 Blender Authors, GPL-2.0-or-later.
//   source/blender/render/intern/texture_image.cc, texture_common.h (image texture lookup and
//   brightness/contrast), SPDX-FileCopyrightText: 2005 Blender Authors, GPL-2.0-or-later.
//   source/blender/blenlib/intern/math_matrix_c.cc, math_rotation_c.cc (blend_m4_m4m4),
//   SPDX-FileCopyrightText: 2001-2002 NaN Holding BV. All rights reserved., GPL-2.0-or-later.

use std::path::Path;

use glam::{DMat3, DQuat};

use super::{
    DMat4, DVec3, ErrorCode, Id, Mesh, Modifier, PotError, Result, Value, bool_param,
    corner_normals, id_indices, invalid_parameter, json, number_param, optional_string_param,
    polygons, positions, safe_acos_approx, schema_value, string_param, string_param_with_expected,
    uv_layers, validated, vertex_normals,
};
use crate::blender_rng::BlenderRng;
use crate::geom::legacy_texture::{self, ResolvedTexture};
use crate::model::SceneDoc;

/// A texture's mapping from a subject vertex to texture space (`MOD_get_texture_coords`).
///
/// Mirrors Blender's `vgroup` handling: an absent group means full weight, a defined group without
/// stored weights is Blender's null `dvert` (displace then does nothing), and weights otherwise.
#[derive(Clone, Debug)]
pub(super) enum VertexGroup {
    Absent,
    Unweighted,
    Weighted(Vec<f64>),
}

impl VertexGroup {
    /// The per-vertex weight Blender multiplies in: `1` when there are no weights.
    pub(super) fn weight(&self, index: usize, invert: bool) -> f64 {
        match self {
            Self::Weighted(weights) => {
                let weight = weights.get(index).copied().unwrap_or(0.0);
                if invert { 1.0 - weight } else { weight }
            }
            Self::Absent | Self::Unweighted => 1.0,
        }
    }
}

/// Reads the `vertex_group` parameter against the mesh's stored vertex groups.
pub(super) fn vertex_group(mesh: &Mesh, modifier: &Modifier) -> Result<VertexGroup> {
    let name = string_param_with_expected(modifier, "vertex_group", "a vertex-group name")?;
    if name.is_empty() {
        return Ok(VertexGroup::Absent);
    }
    let Some(groups) = mesh
        .attributes
        .get("vertex_groups")
        .and_then(Value::as_object)
    else {
        return Ok(VertexGroup::Absent);
    };
    let Some(group) = groups.get(name) else {
        return Ok(VertexGroup::Absent);
    };
    let group = group.as_object().ok_or_else(|| {
        invalid_parameter(
            modifier,
            "vertex_group",
            "a vertex group present in the mesh attributes",
        )
    })?;
    let stored = groups
        .values()
        .any(|values| values.as_object().is_some_and(|map| !map.is_empty()));
    if !stored {
        return Ok(VertexGroup::Unweighted);
    }
    mesh.vertices
        .iter()
        .map(|vertex| {
            let weight = group
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
        .map(VertexGroup::Weighted)
}

fn dependency_missing(modifier: &Modifier, reference: &str) -> PotError {
    PotError::with_details(
        ErrorCode::DependencyMissing,
        format!(
            "modifier `{}` needs the resolved transform for `{reference}`",
            modifier.modifier_type
        ),
        json!({"modifier_id": modifier.id, "parameter": reference}),
    )
}

/// The runtime matrix a scene-resolved parameter supplies (column-major).
fn runtime_matrix(matrix: Option<[f64; 16]>) -> Option<DMat4> {
    matrix.map(|values| DMat4::from_cols_array(&values))
}

/// Resolves the texture named by `parameter` (an image or an inline legacy texture); `None` when the
/// parameter is unset.
///
/// # Errors
///
/// Returns the errors of [`legacy_texture::resolve_texture`].
pub(crate) fn load_modifier_texture(
    modifier: &Modifier,
    parameter: &str,
    doc: &SceneDoc,
    project_root: Option<&Path>,
) -> Result<Option<ResolvedTexture>> {
    let Some(text) = optional_string_param(modifier, parameter)? else {
        return Ok(None);
    };
    let texture_id = Id::new(text.to_owned())
        .map_err(|_| invalid_parameter(modifier, parameter, "a texture data-block ID"))?;
    legacy_texture::resolve_texture(doc, &texture_id, project_root).map(Some)
}

/// The loaded texture for a modifier whose `texture` parameter is set; `None` when it is unset.
fn required_texture<'a>(
    modifier: &Modifier,
    texture: Option<&'a ResolvedTexture>,
) -> Result<Option<&'a ResolvedTexture>> {
    if optional_string_param(modifier, "texture")?.is_none() {
        return Ok(None);
    }
    texture.map(Some).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::DependencyMissing,
            "texture was not loaded for the modifier texture",
            json!({"modifier_id": modifier.id}),
        )
    })
}

/// Subject-local texture coordinates per vertex (`MOD_get_texture_coords`).
///
/// `LOCAL` uses current positions. `GLOBAL` applies subject-to-world; `OBJECT` applies subject-to-world
/// then world-to-mapping. `OBJECT` without a map object is `LOCAL`. `UV` takes each vertex's UV from
/// the first face that uses it, converts stored V-down to Blender's V-up, then remaps from `[0, 1]`
/// to `[-1, 1]`; a mesh without UV layers falls back to `LOCAL`.
fn texture_coordinates(
    mesh: &Mesh,
    modifier: &Modifier,
    positions: &[DVec3],
    mapping: Option<crate::model::TextureCoordinateMapping>,
) -> Result<Vec<DVec3>> {
    match string_param(modifier, "texture_coords")? {
        "LOCAL" => Ok(positions.to_vec()),
        "GLOBAL" => {
            let mapping = mapping.ok_or_else(|| dependency_missing(modifier, "texture_coords"))?;
            Ok(positions
                .iter()
                .map(|position| mapping.global_coordinate(*position))
                .collect())
        }
        "OBJECT" if optional_string_param(modifier, "texture_coords_object")?.is_none() => {
            Ok(positions.to_vec())
        }
        "OBJECT" => {
            let mapping = mapping.ok_or_else(|| dependency_missing(modifier, "texture_coords"))?;
            Ok(positions
                .iter()
                .map(|position| mapping.object_coordinate(*position))
                .collect())
        }
        "UV" => match uv_texture_coordinates(mesh, modifier)? {
            Some(coords) => Ok(coords),
            None => Ok(positions.to_vec()),
        },
        _ => Err(invalid_parameter(
            modifier,
            "texture_coords",
            "LOCAL, GLOBAL, OBJECT, or UV",
        )),
    }
}

/// Blender's UV texture coordinates, or `None` when the mesh has no UV layer.
fn uv_texture_coordinates(mesh: &Mesh, modifier: &Modifier) -> Result<Option<Vec<DVec3>>> {
    let store = uv_layers::read(mesh);
    if store.layers.is_empty() {
        return Ok(None);
    }
    let requested = string_param(modifier, "uv_layer")?;
    let requested = (!requested.is_empty())
        .then(|| store.index_of(requested))
        .flatten();
    let active = store
        .active
        .as_deref()
        .and_then(|name| store.resolve(Some(name)).0);
    let layer = &store.layers[requested.or(active).unwrap_or(0)];
    let indices = id_indices(mesh);
    let mut coords = vec![DVec3::ZERO; mesh.vertices.len()];
    let mut assigned = vec![false; mesh.vertices.len()];
    for (face_index, face) in mesh.faces.iter().enumerate() {
        for (corner, id) in face.vertices.iter().enumerate() {
            let index = indices.get(id).copied().ok_or_else(|| {
                invalid_parameter(modifier, "uv_layer", "a mesh with valid face vertices")
            })?;
            if assigned[index] {
                continue;
            }
            let uv = layer
                .faces
                .get(face_index)
                .and_then(Option::as_ref)
                .and_then(|values| values.get(corner))
                .copied()
                .unwrap_or([0.0, 0.0]);
            coords[index] = DVec3::new(uv[0] * 2.0 - 1.0, (1.0 - uv[1]) * 2.0 - 1.0, 0.0);
            assigned[index] = true;
        }
    }
    Ok(Some(coords))
}

/// Blender's `vert_normals_true`: the angle-weighted face normals. An isolated vertex points along its
/// position, as `normals_calc_verts` does.
fn true_vertex_normals(mesh: &Mesh, positions: &[DVec3]) -> Result<Vec<DVec3>> {
    let faces = polygons(mesh)?;
    let mut normals = vertex_normals(positions, &faces, false);
    let mut used = vec![false; positions.len()];
    for (vertices, _) in &faces {
        for &vertex in vertices {
            if let Some(flag) = used.get_mut(vertex) {
                *flag = true;
            }
        }
    }
    for ((normal, position), flag) in normals.iter_mut().zip(positions).zip(&used) {
        if !flag {
            *normal = position.normalize_or_zero();
        }
    }
    Ok(normals)
}

/// Blender's `vert_normals()`: the true normals, or, when the mesh has a custom-normal layer, the
/// angle-weighted average of its corner normals (`mix_normals_corner_to_vert`).
fn corrected_vertex_normals(mesh: &Mesh, positions: &[DVec3]) -> Result<Vec<DVec3>> {
    let has_layer = mesh
        .attributes
        .get("custom_normal")
        .and_then(|layer| layer.get("values"))
        .is_some();
    if !has_layer {
        return true_vertex_normals(mesh, positions);
    }
    let corners = corner_normals(mesh, positions)?;
    let faces = polygons(mesh)?;
    let mut sums = vec![DVec3::ZERO; positions.len()];
    let mut used = vec![false; positions.len()];
    let mut corner = 0_usize;
    for (vertices, _) in &faces {
        let count = vertices.len();
        for (position, &vertex) in vertices.iter().enumerate() {
            let previous = vertices[(position + count - 1) % count];
            let next = vertices[(position + 1) % count];
            let to_previous = (positions[previous] - positions[vertex]).normalize_or_zero();
            let to_next = (positions[next] - positions[vertex]).normalize_or_zero();
            let factor = safe_acos_approx(to_previous.dot(to_next));
            let normal = corners.get(corner).copied().unwrap_or(DVec3::ZERO);
            sums[vertex] += normal * factor;
            used[vertex] = true;
            corner += 1;
        }
    }
    Ok(sums
        .into_iter()
        .zip(positions)
        .zip(used)
        .map(|((sum, position), used)| {
            if used {
                sum.normalize_or_zero()
            } else {
                position.normalize_or_zero()
            }
        })
        .collect())
}

/// Blender's `displace` modifier. `texture` must be loaded whenever the `texture` parameter is set;
/// `owner_world_matrix` is the object's world transform used by `space` `GLOBAL`.
pub(crate) fn displace_textured(
    mesh: &Mesh,
    modifier: &Modifier,
    owner_world_matrix: DMat4,
    texture: Option<&ResolvedTexture>,
    rng: &mut BlenderRng,
) -> Result<Mesh> {
    validated(mesh)?;
    let strength = number_param(modifier, "strength")?;
    let mid_level = number_param(modifier, "mid_level")?;
    let direction = string_param(modifier, "direction")?;
    let global = string_param(modifier, "space")? == "GLOBAL";
    let invert = bool_param(modifier, "invert_vertex_group")?;
    let texture = required_texture(modifier, texture)?;
    if (texture.is_none() && direction == "RGB_TO_XYZ") || strength == 0.0 {
        return Ok(mesh.clone());
    }
    let group = vertex_group(mesh, modifier)?;
    if matches!(group, VertexGroup::Unweighted) {
        return Ok(mesh.clone());
    }
    let positions = positions(mesh);
    let coords = match texture {
        Some(_) => Some(texture_coordinates(
            mesh,
            modifier,
            &positions,
            modifier.runtime.texture_mapping,
        )?),
        None => None,
    };
    let normals = match direction {
        "NORMAL" => Some(true_vertex_normals(mesh, &positions)?),
        "CUSTOM_NORMAL" => Some(corrected_vertex_normals(mesh, &positions)?),
        "X" | "Y" | "Z" | "RGB_TO_XYZ" => None,
        _ => {
            return Err(invalid_parameter(
                modifier,
                "direction",
                "X, Y, Z, NORMAL, CUSTOM_NORMAL, or RGB_TO_XYZ",
            ));
        }
    };
    let mut result = mesh.clone();
    for (index, vertex) in result.vertices.iter_mut().enumerate() {
        let weight = group.weight(index, invert);
        if weight == 0.0 {
            continue;
        }
        let sample = match (texture, &coords) {
            (Some(texture), Some(coords)) => Some(legacy_texture::sample(
                texture.source(),
                coords[index],
                rng,
            )?),
            _ => None,
        };
        let scaled = strength * weight;
        let delta = sample.map_or(1.0 - mid_level, |sample| sample.intensity - mid_level) * scaled;
        let delta = delta.clamp(-10000.0, 10000.0);
        let offset = match direction {
            "X" => axis_offset(0, delta, global, owner_world_matrix),
            "Y" => axis_offset(1, delta, global, owner_world_matrix),
            "Z" => axis_offset(2, delta, global, owner_world_matrix),
            "RGB_TO_XYZ" => {
                let Some(sample) = sample else {
                    return Err(dependency_missing(modifier, "texture"));
                };
                let rgb = if sample.is_rgb {
                    sample.rgb
                } else {
                    [sample.intensity; 3]
                };
                let mut local = DVec3::from_array(rgb) - DVec3::splat(mid_level);
                if global {
                    local = transpose_mul(owner_world_matrix, local);
                }
                local * scaled
            }
            _ => {
                let normal = normals
                    .as_ref()
                    .and_then(|normals| normals.get(index))
                    .copied()
                    .unwrap_or(DVec3::ZERO);
                normal * delta
            }
        };
        vertex.co += offset;
    }
    validated(&result)?;
    Ok(result)
}

/// The displacement along one axis: the object-space image of the global axis under Blender's
/// `local_mat` (row `axis` of the world 3x3), or the local axis itself.
fn axis_offset(axis: usize, delta: f64, global: bool, world: DMat4) -> DVec3 {
    if !global {
        let mut offset = DVec3::ZERO;
        offset[axis] = delta;
        return offset;
    }
    let row = DVec3::new(world.x_axis[axis], world.y_axis[axis], world.z_axis[axis]);
    row * delta
}

/// `M^T v` for the upper 3x3 of `world`, as Blender's `mul_transposed_mat3_m4_v3`.
fn transpose_mul(world: DMat4, vector: DVec3) -> DVec3 {
    DVec3::new(
        world.x_axis.truncate().dot(vector),
        world.y_axis.truncate().dot(vector),
        world.z_axis.truncate().dot(vector),
    )
}

/// Blender's `wave` modifier at scene frame `frame`.
pub(crate) fn wave_textured(
    mesh: &Mesh,
    modifier: &Modifier,
    frame: f64,
    texture: Option<&ResolvedTexture>,
    rng: &mut BlenderRng,
) -> Result<Mesh> {
    validated(mesh)?;
    let speed = number_param(modifier, "speed")?;
    let height = number_param(modifier, "height")?;
    let width = number_param(modifier, "width")?;
    let narrowness = number_param(modifier, "narrowness")?;
    let time_offset = number_param(modifier, "time_offset")?;
    let lifetime = number_param(modifier, "lifetime")?;
    let damping = number_param(modifier, "damping_time")?;
    let falloff_radius = number_param(modifier, "falloff_radius")?;
    let use_x = bool_param(modifier, "use_x")?;
    let use_y = bool_param(modifier, "use_y")?;
    let use_cyclic = bool_param(modifier, "use_cyclic")?;
    let use_normal = bool_param(modifier, "use_normal")?;
    let normal_axes = [
        bool_param(modifier, "use_normal_x")?,
        bool_param(modifier, "use_normal_y")?,
        bool_param(modifier, "use_normal_z")?,
    ];
    let invert = bool_param(modifier, "invert_vertex_group")?;
    let texture = required_texture(modifier, texture)?;
    if falloff_radius < 0.0 {
        return Err(invalid_parameter(
            modifier,
            "falloff_radius",
            "a non-negative radius",
        ));
    }
    let damping = if damping == 0.0 { 10.0 } else { damping };
    let time = frame - time_offset;
    let mut life = height;
    if lifetime != 0.0 && time > lifetime {
        let elapsed = time - lifetime;
        life = if elapsed > damping {
            0.0
        } else {
            height * (1.0 - (elapsed / damping).sqrt())
        };
    }
    if life == 0.0 {
        return Ok(mesh.clone());
    }
    // The scene path resolves the start object into the runtime matrix and removes its parameter.
    let (start_x, start_y) = if let Some(object) = runtime_matrix(modifier.runtime.object_matrix) {
        let origin = object.transform_point3(DVec3::ZERO);
        (origin.x, origin.y)
    } else {
        (
            number_param(modifier, "start_position_x")?,
            number_param(modifier, "start_position_y")?,
        )
    };
    let min_factor = (-(width * narrowness) * (width * narrowness)).exp();
    let group = vertex_group(mesh, modifier)?;
    let positions = positions(mesh);
    let normals = if use_normal {
        Some(corrected_vertex_normals(mesh, &positions)?)
    } else {
        None
    };
    let coords = match texture {
        Some(_) => Some(texture_coordinates(
            mesh,
            modifier,
            &positions,
            modifier.runtime.texture_mapping,
        )?),
        None => None,
    };
    let falloff_inverse = if falloff_radius == 0.0 {
        1.0
    } else {
        1.0 / falloff_radius
    };
    let mut result = mesh.clone();
    for (index, vertex) in result.vertices.iter_mut().enumerate() {
        let weight = group.weight(index, invert);
        if weight == 0.0 {
            continue;
        }
        let x = vertex.co.x - start_x;
        let y = vertex.co.y - start_y;
        let mut amplitude = match (use_x, use_y) {
            (true, true) => (x * x + y * y).sqrt(),
            (true, false) => x,
            (false, true) => y,
            (false, false) => 0.0,
        };
        amplitude -= time * speed;
        if use_cyclic {
            amplitude = (amplitude - width) % (2.0 * width) + width;
        }
        let distance = match (use_x, use_y) {
            (true, true) => (x * x + y * y).sqrt(),
            (true, false) => x.abs(),
            (false, true) => y.abs(),
            (false, false) => 0.0,
        };
        let falloff_factor = if falloff_radius == 0.0 {
            1.0
        } else {
            (1.0 - distance * falloff_inverse).clamp(0.0, 1.0)
        };
        if falloff_factor == 0.0 || !(amplitude > -width && amplitude < width) {
            continue;
        }
        let scaled = amplitude * narrowness;
        let mut shape = 1.0 / (scaled * scaled).exp() - min_factor;
        if let (Some(texture), Some(coords)) = (texture, &coords) {
            shape *= legacy_texture::sample(texture.source(), coords[index], rng)?.intensity;
        }
        let amount = life * shape * weight * falloff_factor;
        match &normals {
            Some(normals) => {
                let normal = normals.get(index).copied().unwrap_or(DVec3::ZERO);
                if normal_axes[0] {
                    vertex.co.x += amount * normal.x;
                }
                if normal_axes[1] {
                    vertex.co.y += amount * normal.y;
                }
                if normal_axes[2] {
                    vertex.co.z += amount * normal.z;
                }
            }
            None => vertex.co.z += amount,
        }
    }
    validated(&result)?;
    Ok(result)
}

/// Blender's `warp` modifier. `warp_from_matrix` and `warp_to_matrix` are subject-local to the
/// source and destination spaces (`inv(W_subject) * W_ref`, bone-aware). A modifier with no source or
/// destination object is unchanged.
pub(crate) fn warp_textured(
    mesh: &Mesh,
    modifier: &Modifier,
    texture: Option<&ResolvedTexture>,
    rng: &mut BlenderRng,
) -> Result<Mesh> {
    validated(mesh)?;
    let texture = required_texture(modifier, texture)?;
    // The scene path resolves both objects into the runtime matrices and removes their parameters.
    // An unset pair leaves the mesh unchanged, as Blender does.
    let (Some(from), Some(to)) = (
        runtime_matrix(modifier.runtime.warp_from_matrix),
        runtime_matrix(modifier.runtime.warp_to_matrix),
    ) else {
        return Ok(mesh.clone());
    };
    if from.determinant().abs() <= f64::EPSILON || !from.is_finite() {
        return Err(invalid_parameter(
            modifier,
            "object_from",
            "an invertible object transform",
        ));
    }
    let mut strength = number_param(modifier, "strength")?;
    let radius = number_param(modifier, "falloff_radius")?;
    if radius < 0.0 {
        return Err(invalid_parameter(
            modifier,
            "falloff_radius",
            "a non-negative radius",
        ));
    }
    let falloff = string_param(modifier, "falloff_type")?;
    let curve = match falloff {
        "CURVE" => Some(falloff_curve(modifier)?),
        "NONE" | "SMOOTH" | "SPHERE" | "ROOT" | "INVERSE_SQUARE" | "SHARP" | "LINEAR"
        | "CONSTANT" => None,
        _ => {
            return Err(invalid_parameter(
                modifier,
                "falloff_type",
                "NONE, CURVE, SMOOTH, SPHERE, ROOT, INVERSE_SQUARE, SHARP, LINEAR, or CONSTANT",
            ));
        }
    };
    let volume_preserve = bool_param(modifier, "use_volume_preserve")?;
    let invert = bool_param(modifier, "invert_vertex_group")?;
    let group = vertex_group(mesh, modifier)?;
    let positions = positions(mesh);
    let coords = match texture {
        Some(_) => Some(texture_coordinates(
            mesh,
            modifier,
            &positions,
            modifier.runtime.texture_mapping,
        )?),
        None => None,
    };
    // Blender: mat_final = inv(mat_from) * mat_to, applied in the source frame.
    let from_inverse = from.inverse();
    let mut final_matrix = from_inverse * to;
    if strength < 0.0 {
        strength = -strength;
        let location = final_matrix.w_axis;
        let mut inverted = final_matrix.inverse();
        inverted.w_axis.x = -location.x;
        inverted.w_axis.y = -location.y;
        inverted.w_axis.z = -location.z;
        final_matrix = inverted;
    }
    let origin = from.w_axis.truncate();
    let radius_squared = radius * radius;
    let mut result = mesh.clone();
    for (index, vertex) in result.vertices.iter_mut().enumerate() {
        let co = vertex.co;
        let distance_squared = (co - origin).length_squared();
        let mut factor = if falloff == "NONE" {
            1.0
        } else if distance_squared < radius_squared {
            let linear = (radius - distance_squared.sqrt()) / radius;
            if linear == 0.0 {
                continue;
            }
            warp_falloff(falloff, linear, curve.as_ref())
        } else {
            continue;
        };
        let weight = match &group {
            VertexGroup::Weighted(_) => {
                let weight = group.weight(index, invert) * strength;
                if weight <= 0.0 {
                    continue;
                }
                weight
            }
            VertexGroup::Absent | VertexGroup::Unweighted => strength,
        };
        factor *= weight;
        if let (Some(texture), Some(coords)) = (texture, &coords) {
            factor *= legacy_texture::sample(texture.source(), coords[index], rng)?.intensity;
        }
        if factor == 0.0 {
            continue;
        }
        let local = from_inverse.transform_point3(co);
        #[expect(
            clippy::float_cmp,
            reason = "Blender takes the full-transform branch only for an exact factor of 1.0"
        )]
        let moved = if factor == 1.0 {
            final_matrix.transform_point3(local)
        } else if volume_preserve {
            blend_with_identity(final_matrix, factor).transform_point3(local)
        } else {
            let target = final_matrix.transform_point3(local);
            local * (1.0 - factor) + target * factor
        };
        vertex.co = from.transform_point3(moved);
    }
    validated(&result)?;
    Ok(result)
}

/// Warp `falloff_curve` controls in Blender's stored order and float precision.
fn falloff_curve(modifier: &Modifier) -> Result<super::advanced::CurveMappingTable> {
    let invalid = || {
        invalid_parameter(
            modifier,
            "falloff_curve",
            "at least two finite float32 [x, y] points with an optional AUTO, AUTO_CLAMPED, or VECTOR handle",
        )
    };
    let points = super::advanced::curve_mapping_points(schema_value(modifier, "falloff_curve")?)
        .ok_or_else(invalid)?;
    let settings =
        super::advanced::curve_mapping_settings(modifier.params.get("falloff_curve_settings"))
            .ok_or_else(|| {
                invalid_parameter(
                    modifier,
                    "falloff_curve_settings",
                    "use_clip, finite ordered clip_bounds, and HORIZONTAL or EXTRAPOLATED extend",
                )
            })?;
    super::advanced::curve_map_table(&points, settings).ok_or_else(invalid)
}

/// The falloff of a linear distance factor `linear` in `(0, 1]` for one warp falloff type.
fn warp_falloff(
    kind: &str,
    linear: f64,
    curve: Option<&super::advanced::CurveMappingTable>,
) -> f64 {
    match kind {
        "CURVE" => curve.map_or(linear, |curve| curve.evaluate(linear)),
        "SHARP" => linear * linear,
        "SMOOTH" => 3.0 * linear * linear - 2.0 * linear * linear * linear,
        "ROOT" => linear.sqrt(),
        "SPHERE" => (2.0 * linear - linear * linear).sqrt(),
        "INVERSE_SQUARE" => linear * (2.0 - linear),
        "CONSTANT" => 1.0,
        _ => linear,
    }
}

/// Blender's `blend_m4_m4m4(identity, matrix, weight)`: interpolates location, rotation (shortest-arc
/// slerp of the quaternion), and scale of `matrix` from the identity.
fn blend_with_identity(matrix: DMat4, weight: f64) -> DMat4 {
    let location = matrix.w_axis.truncate();
    let columns = [
        matrix.x_axis.truncate(),
        matrix.y_axis.truncate(),
        matrix.z_axis.truncate(),
    ];
    let mut size = columns.map(DVec3::length);
    let mut rotation = columns.map(DVec3::normalize_or_zero);
    if DMat3::from_cols(rotation[0], rotation[1], rotation[2]).determinant() < 0.0 {
        rotation = rotation.map(|column| -column);
        size = size.map(|value| -value);
    }
    let quat = normalized_matrix_to_quat(|column, row| rotation[column][row]);
    // interp_qt_qtqt(identity, quat, weight): shortest arc, slerp unless nearly aligned.
    let mut start = [1.0_f64, 0.0, 0.0, 0.0];
    let mut cosom = quat[0];
    if cosom < 0.0 {
        cosom = -cosom;
        start = [-1.0, 0.0, 0.0, 0.0];
    }
    let (start_weight, end_weight) = if cosom.abs() < 1.0 - 1e-4 {
        let omega = cosom.acos();
        let sinom = omega.sin();
        (
            ((1.0 - weight) * omega).sin() / sinom,
            (weight * omega).sin() / sinom,
        )
    } else {
        (1.0 - weight, weight)
    };
    let blended =
        std::array::from_fn::<f64, 4, _>(|i| start_weight * start[i] + end_weight * quat[i]);
    let rotation_matrix = DMat3::from_quat(DQuat::from_xyzw(
        blended[1], blended[2], blended[3], blended[0],
    ));
    let scale = size.map(|value| (1.0 - weight) + weight * value);
    let scaled = DMat3::from_cols(
        rotation_matrix.x_axis * scale[0],
        rotation_matrix.y_axis * scale[1],
        rotation_matrix.z_axis * scale[2],
    );
    DMat4::from_cols(
        scaled.x_axis.extend(0.0),
        scaled.y_axis.extend(0.0),
        scaled.z_axis.extend(0.0),
        (location * weight).extend(1.0),
    )
}

/// Blender's `mat3_normalized_to_quat_fast` for a normalized rotation: `m(column, row)` reads
/// Blender's `mat[column][row]`. Returns `[w, x, y, z]`.
#[expect(
    clippy::float_cmp,
    reason = "Blender's mat3_normalized_to_quat_fast tests trace == 1.0 exactly"
)]
fn normalized_matrix_to_quat(m: impl Fn(usize, usize) -> f64) -> [f64; 4] {
    let mut q = [0.0_f64; 4];
    if m(2, 2) < 0.0 {
        if m(0, 0) > m(1, 1) {
            let trace = 1.0 + m(0, 0) - m(1, 1) - m(2, 2);
            let mut s = 2.0 * trace.sqrt();
            if m(1, 2) < m(2, 1) {
                s = -s;
            }
            q[1] = 0.25 * s;
            s = 1.0 / s;
            q[0] = (m(1, 2) - m(2, 1)) * s;
            q[2] = (m(0, 1) + m(1, 0)) * s;
            q[3] = (m(2, 0) + m(0, 2)) * s;
            if trace == 1.0 && q[0] == 0.0 && q[2] == 0.0 && q[3] == 0.0 {
                q[1] = 1.0;
            }
        } else {
            let trace = 1.0 - m(0, 0) + m(1, 1) - m(2, 2);
            let mut s = 2.0 * trace.sqrt();
            if m(2, 0) < m(0, 2) {
                s = -s;
            }
            q[2] = 0.25 * s;
            s = 1.0 / s;
            q[0] = (m(2, 0) - m(0, 2)) * s;
            q[1] = (m(0, 1) + m(1, 0)) * s;
            q[3] = (m(1, 2) + m(2, 1)) * s;
            if trace == 1.0 && q[0] == 0.0 && q[1] == 0.0 && q[3] == 0.0 {
                q[2] = 1.0;
            }
        }
    } else if m(0, 0) < -m(1, 1) {
        let trace = 1.0 - m(0, 0) - m(1, 1) + m(2, 2);
        let mut s = 2.0 * trace.sqrt();
        if m(0, 1) < m(1, 0) {
            s = -s;
        }
        q[3] = 0.25 * s;
        s = 1.0 / s;
        q[0] = (m(0, 1) - m(1, 0)) * s;
        q[1] = (m(2, 0) + m(0, 2)) * s;
        q[2] = (m(1, 2) + m(2, 1)) * s;
        if trace == 1.0 && q[0] == 0.0 && q[1] == 0.0 && q[2] == 0.0 {
            q[3] = 1.0;
        }
    } else {
        let trace = 1.0 + m(0, 0) + m(1, 1) + m(2, 2);
        let mut s = 2.0 * trace.sqrt();
        q[0] = 0.25 * s;
        s = 1.0 / s;
        q[1] = (m(1, 2) - m(2, 1)) * s;
        q[2] = (m(2, 0) - m(0, 2)) * s;
        q[3] = (m(0, 1) - m(1, 0)) * s;
        if trace == 1.0 && q[1] == 0.0 && q[2] == 0.0 && q[3] == 0.0 {
            q[0] = 1.0;
        }
    }
    let length_squared: f64 = q.iter().map(|value| value * value).sum();
    if (length_squared - 1.0).abs() >= 0.0006 {
        let length = length_squared.sqrt();
        q = q.map(|value| value / length);
    }
    q
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "small mesh fixture")]

    use glam::DVec3;
    use serde_json::json;

    use super::uv_texture_coordinates;
    use crate::{
        geom::{
            BoxParams, Mesh,
            uv_layers::{self, UvLayer, UvStore},
        },
        model::{Id, Modifier, ModifierRuntime},
    };

    fn modifier(uv_layer: &str) -> Modifier {
        let mut params = serde_json::Map::new();
        params.insert("uv_layer".to_owned(), json!(uv_layer));
        Modifier {
            id: Id::new("displace").unwrap(),
            modifier_type: "displace".to_owned(),
            name: "Displace".to_owned(),
            enabled: true,
            show_viewport: true,
            show_render: true,
            params,
            binding_data: None,
            runtime: ModifierRuntime::default(),
        }
    }

    #[test]
    fn uv_texture_coordinates_use_the_active_layer_and_flip_stored_v() {
        let mut mesh = Mesh::box_mesh(BoxParams {
            size: DVec3::splat(2.0),
        })
        .unwrap();
        let face_count = mesh.faces.len();
        let mut detail = UvLayer::new("Detail", vec![Some(vec![[0.3, 0.4]; 4]); face_count]);
        detail.pinned = vec![Some(vec![true; 4]); mesh.faces.len()];
        uv_layers::write(
            &mut mesh,
            &UvStore {
                layers: vec![
                    UvLayer::new("Map", vec![Some(vec![[0.1, 0.2]; 4]); face_count]),
                    detail,
                ],
                active: Some("Map".to_owned()),
                render: Some("Detail".to_owned()),
            },
        );

        for (selector, expected) in [
            ("", DVec3::new(-0.8, 0.6, 0.0)),
            ("Detail", DVec3::new(-0.4, 0.2, 0.0)),
            ("Missing", DVec3::new(-0.8, 0.6, 0.0)),
        ] {
            let coords = uv_texture_coordinates(&mesh, &modifier(selector))
                .unwrap()
                .unwrap();
            assert!((coords[0] - expected).length() < 1e-12);
        }
    }
}
