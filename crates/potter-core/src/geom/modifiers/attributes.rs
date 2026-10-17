//! Blender attribute-domain, normal, UV, transfer, and point-cache modifiers.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use glam::{DMat3, DMat4, DVec2, DVec3, Vec2, Vec3};
use serde_json::{Map, Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{Edge, Face, Mesh, Vertex},
    model::{CameraData, Modifier},
};

mod normals;
mod transfer;
mod uv_projection;
mod weights;

use normals::{normal_edit, weighted_normal};
use transfer::data_transfer;
pub(crate) use uv_projection::build_uv_warp_matrix_f32;
use uv_projection::{uv_project, uv_warp};
use weights::{
    texture_mask_weights, vertex_weight_edit, vertex_weight_mix, vertex_weight_proximity,
};

use super::advanced::{CurveMappingTable, curve_map_table, curve_mapping_points};
use super::{
    bool_param as boolean, closest_point_triangle, invalid_parameter as invalid,
    number_param as number, schema_value, string_param as string,
};
#[derive(Clone, Debug, Default)]
/// Ordered vertex-group names and active index resolved from the scene object.
pub struct AttributeLayerSelection {
    pub vertex_group_names: Vec<String>,
    pub active_vertex_group_index: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
/// Original Blender f32 world and relative matrices for I modifier operands.
pub struct SceneOperandMatricesF32 {
    pub world: Option<[f32; 16]>,
    /// Blender-order object-local to subject-local matrix.
    pub local_to_subject: Option<[f32; 16]>,
    /// Blender-order subject-local to object-local inverse matrix.
    pub subject_to_local: Option<[f32; 16]>,
}

#[derive(Clone, Debug)]
/// A resolved scene-object operand for native modifier evaluation.
pub struct SceneOperand {
    /// Object-local evaluated mesh data.
    pub mesh: Mesh,
    /// f64 object-to-subject transform for native scene modifier paths that operate on DVec3.
    pub local_to_subject: Option<DMat4>,
    /// Exact f32 matrices, resolved only for modifiers that consume them.
    pub blender_matrices_f32: Option<SceneOperandMatricesF32>,
    /// Camera settings when this operand is a camera object.
    pub camera: Option<CameraData>,
    /// Whether this operand has armature data for UV Warp bone references.
    pub armature: bool,
    /// The object's own scale property, used by radial Normal Edit.
    pub object_scale: DVec3,
    /// Names of pose bones selected in the armature data.
    pub selected_bones: BTreeSet<String>,
    /// Names of bones whose deform flag is enabled.
    pub deforming_bones: BTreeSet<String>,
    /// Ordered layer selector state for the operand.
    pub layer_selection: AttributeLayerSelection,
}

/// Evaluate a modifier that does not need a scene object operand.
///
/// # Errors
///
/// Returns an invalid-parameter or evaluation error when a requested group, UV layer,
/// or attribute is missing or malformed.
pub(super) fn evaluate(mesh: &Mesh, modifier: &Modifier, subject_name: &str) -> Result<Mesh> {
    evaluate_with_context(
        mesh,
        modifier,
        &BTreeMap::new(),
        None,
        DMat4::IDENTITY,
        subject_name,
        &AttributeLayerSelection::default(),
    )
}

/// Evaluate an attribute modifier with an explicit subject identity.
pub fn evaluate_with_context(
    mesh: &Mesh,
    modifier: &Modifier,
    operands: &BTreeMap<String, Vec<SceneOperand>>,
    texture: Option<&crate::geom::ResolvedTexture>,
    subject_world: DMat4,
    subject_name: &str,
    subject_layer_selection: &AttributeLayerSelection,
) -> Result<Mesh> {
    let mut rng = crate::blender_rng::BlenderRng::seed_random(0);
    evaluate_with_texture_context(
        mesh,
        modifier,
        operands,
        texture.map(crate::geom::ResolvedTexture::source),
        subject_world,
        subject_name,
        subject_layer_selection,
        &mut rng,
    )
}

pub(crate) fn evaluate_with_texture_context(
    mesh: &Mesh,
    modifier: &Modifier,
    operands: &BTreeMap<String, Vec<SceneOperand>>,
    texture: Option<crate::geom::legacy_texture::TextureSource<'_>>,
    subject_world: DMat4,
    subject_name: &str,
    subject_layer_selection: &AttributeLayerSelection,
    rng: &mut crate::blender_rng::BlenderRng,
) -> Result<Mesh> {
    if subject_name.is_empty() {
        return Err(PotError::invalid_argument(
            "attribute modifier evaluation requires a subject name",
        ));
    }
    if modifier.modifier_type == "data_transfer"
        && (modifier.params.get("object").is_none_or(Value::is_null)
            || operands.get("object").is_some_and(Vec::is_empty))
    {
        return Ok(mesh.clone());
    }
    if modifier.modifier_type == "vertex_weight_proximity"
        && modifier.params.get("target").is_none_or(Value::is_null)
    {
        return Ok(mesh.clone());
    }
    let texture_weights =
        texture_mask_weights(mesh, modifier, operands, texture, subject_world, rng)?;
    match modifier.modifier_type.as_str() {
        "vertex_weight_edit" => {
            vertex_weight_edit(mesh, modifier, texture_weights.as_ref(), subject_name)
        }
        "vertex_weight_mix" => vertex_weight_mix(mesh, modifier, texture_weights.as_ref()),
        "vertex_weight_proximity" => vertex_weight_proximity(
            mesh,
            modifier,
            operands,
            texture_weights.as_ref(),
            subject_world,
            subject_name,
        ),
        "normal_edit" => normal_edit(mesh, modifier, operands),
        "uv_project" => uv_project(mesh, modifier, operands, subject_world),
        "uv_warp" => uv_warp(mesh, modifier, operands),
        "data_transfer" => data_transfer(mesh, modifier, operands, subject_layer_selection),
        "mask" => super::advanced::mask(mesh, modifier, operands),
        "weighted_normal" => weighted_normal(mesh, modifier),
        "mesh_cache" => Err(invalid(
            modifier,
            "resource",
            "a project resource resolved by the evaluation context",
        )),
        _ => Err(invalid(modifier, "type", "a supported attribute modifier")),
    }
}

/// Evaluate a PC2 or MDD Mesh Cache resource at one scene frame.
///
/// # Errors
///
/// Returns an invalid-argument error for malformed cache headers, incompatible topology,
/// or unsupported cache settings.
///
/// Returns non-fatal diagnostics when Blender's INTEGRATE source-mesh checks fail.
pub fn evaluate_mesh_cache(
    mesh: &mut Mesh,
    source_mesh: Option<&Mesh>,
    modifier: &Modifier,
    frame: f64,
    fps: u32,
    fps_base: f64,
    bytes: &[u8],
) -> Result<Vec<PotError>> {
    if !frame.is_finite() || fps == 0 || !fps_base.is_finite() || fps_base <= 0.0 {
        return Err(invalid(
            modifier,
            "frame",
            "a finite scene frame and positive frame rate",
        ));
    }
    let samples = match format {
        "PC2" => read_pc2_metadata(bytes, modifier)?,
        "MDD" => read_mdd_metadata(bytes, modifier)?,
        _ => return Err(invalid(modifier, "cache_format", "PC2 or MDD")),
    };
    if samples.vertex_count != mesh.vertices.len() {
        return Err(invalid(
            modifier,
            "resource",
            "a cache with matching vertex count",
        ));
    }
    let frame_start = cache_float(number(modifier, "frame_start")?);
    let frame_scale = cache_float(number(modifier, "frame_scale")?);
    let frame = cache_float(frame);
    let fps = cache_float(f64::from(fps)) / cache_float(fps_base);
    let play_mode = string(modifier, "play_mode")?;
    if !matches!(play_mode, "SCENE" | "CUSTOM") {
        return Err(invalid(modifier, "play_mode", "SCENE or CUSTOM"));
    }
    let time_mode = string(modifier, "time_mode")?;
    if !matches!(time_mode, "FRAME" | "TIME" | "FACTOR") {
        return Err(invalid(modifier, "time_mode", "FRAME, TIME, or FACTOR"));
    }
    let custom = play_mode == "CUSTOM";
    let interpolation = string(modifier, "interpolation")?;
    if !matches!(interpolation, "NONE" | "LINEAR") {
        return Err(invalid(modifier, "interpolation", "NONE or LINEAR"));
    }
    let interpolate = interpolation == "LINEAR";
    let (first, second, factor) = match time_mode {
        "FRAME" => {
            let cache_frame = if custom {
                cache_float(number(modifier, "eval_frame")?)
            } else {
                frame_scale * frame - frame_start
            };
            select_cache_frame(cache_frame, samples.frame_count, interpolate)
        }
        "TIME" => {
            let evaluation_time = if custom {
                cache_float(number(modifier, "eval_time")?)
            } else {
                frame_scale * (frame / fps) - frame_start
            };
            match samples.format {
                CacheFormat::Pc2 {
                    time_start,
                    sampling,
                } => {
                    let cache_frame = (evaluation_time / fps - time_start) / sampling;
                    select_cache_frame(cache_frame, samples.frame_count, interpolate)
                }
                CacheFormat::Mdd { times_offset } => {
                    let cache_frame = cache_frame_at_time(
                        bytes,
                        times_offset,
                        samples.frame_count,
                        evaluation_time,
                        modifier,
                    )?;
                    select_cache_frame(cache_frame, samples.frame_count, interpolate)
                }
            }
        }
        _ => {
            let evaluation_factor = if custom {
                cache_float(number(modifier, "eval_factor")?)
            } else {
                frame_scale * (frame / fps) - frame_start
            };
            let frame_count = cache_frame_count(samples.frame_count, modifier)?;
            select_cache_frame(
                evaluation_factor.clamp(0.0, 1.0) * frame_count,
                samples.frame_count,
                interpolate,
            )
        }
    };
    let mut sampled = samples.read_frame(bytes, first, modifier)?;
    if first != second {
        let second_points = samples.read_frame(bytes, second, modifier)?;
        for (point, second_point) in sampled.iter_mut().zip(second_points) {
            *point = cache_lerp(*point, second_point, factor);
        }
    }
    let forward = string(modifier, "forward_axis")?;
    let up = string(modifier, "up_axis")?;
    let flip = modifier
        .params
        .get("flip_axis")
        .map(|value| axis_flip_set(value, modifier))
        .transpose()?
        .unwrap_or([false; 3]);
    let deform_mode = string(modifier, "deform_mode")?;
    if !matches!(deform_mode, "OVERWRITE" | "INTEGRATE") {
        return Err(invalid(modifier, "deform_mode", "OVERWRITE or INTEGRATE"));
    }
    let amount = cache_float(number(modifier, "factor")?).clamp(0.0, 1.0);
    let group = optional_string(modifier, "vertex_group")?;
    let weights = group
        .map(|name| get_group_weights(mesh, name))
        .transpose()?
        .flatten();
    let invert_group = boolean(modifier, "invert_vertex_group")?;
    let mut warnings = Vec::new();
    let mut desired = sampled;
    if deform_mode == "INTEGRATE" {
        match source_mesh {
            None => warnings.push(PotError::with_details(
                ErrorCode::EvaluationFailed,
                "'Integrate' only valid for Mesh objects",
                json!({"modifier_id":modifier.id}),
            )),
            Some(source) if source.vertices.len() != mesh.vertices.len() => {
                warnings.push(PotError::with_details(
                    ErrorCode::EvaluationFailed,
                    "'Integrate' original mesh vertex mismatch",
                    json!({"modifier_id":modifier.id,"source_vertices":source.vertices.len(),"current_vertices":mesh.vertices.len()}),
                ));
            }
            Some(source) if source.faces.is_empty() => warnings.push(PotError::with_details(
                ErrorCode::EvaluationFailed,
                "'Integrate' requires faces",
                json!({"modifier_id":modifier.id}),
            )),
            Some(source) => {
                let current = mesh
                    .vertices
                    .iter()
                    .map(|vertex| vertex.co)
                    .collect::<Vec<_>>();
                integrate_cache_samples(source, &current, &mut desired, modifier)?;
            }
        }
    }
    for point in &mut desired {
        *point = apply_cache_axes(*point, forward, up, flip, modifier)?;
    }
    for (index, vertex) in mesh.vertices.iter_mut().enumerate() {
        let influence = if let Some(weights) = &weights {
            if !weights.has_deform_weights {
                continue;
            }
            let weight = weights.values.get(&vertex.id).copied().unwrap_or(0.0);
            let global_factor = if invert_group { -amount } else { amount };
            let global_offset = if invert_group { amount } else { 0.0 };
            global_offset + weight * global_factor
        } else {
            amount
        };
        vertex.co = cache_lerp(vertex.co, desired[index], influence);
    }
    Ok(warnings)
}

#[derive(Clone, Debug)]
struct CacheVertexGroupWeights {
    values: BTreeMap<u32, f32>,
    has_deform_weights: bool,
}

fn integrate_cache_samples(
    source_mesh: &Mesh,
    current: &[DVec3],
    samples: &mut [DVec3],
    modifier: &Modifier,
) -> Result<()> {
    if source_mesh.vertices.len() != current.len() || samples.len() != current.len() {
        return Err(invalid(
            modifier,
            "deform_mode",
            "matching source, current, and cache vertex counts",
        ));
    }
    if source_mesh.faces.is_empty() {
        return Err(invalid(modifier, "deform_mode", "a source mesh with faces"));
    }
    let vertex_indices = source_mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<HashMap<_, _>>();
    let mut positions = vec![Vec3::ZERO; current.len()];
    let mut counts = vec![0_i32; current.len()];
    for face in &source_mesh.faces {
        let count = face.vertices.len();
        for (corner, current_id) in face.vertices.iter().enumerate() {
            let previous_id = face.vertices[(corner + count - 1) % count];
            let next_id = face.vertices[(corner + 1) % count];
            let (Some(&previous), Some(&current_index), Some(&next)) = (
                vertex_indices.get(&previous_id),
                vertex_indices.get(current_id),
                vertex_indices.get(&next_id),
            ) else {
                return Err(invalid(
                    modifier,
                    "deform_mode",
                    "source faces referencing existing vertices",
                ));
            };
            let source = [
                source_mesh.vertices[previous].co,
                source_mesh.vertices[current_index].co,
                source_mesh.vertices[next].co,
            ];
            let target = [samples[previous], samples[current_index], samples[next]];
            positions[current_index] +=
                transform_point_by_triangle(current[current_index], source, target);
            counts[current_index] += 1;
        }
    }
    for (index, position) in positions.into_iter().enumerate() {
        let position = if counts[index] == 0 {
            samples[index].as_vec3()
        } else {
            position / cache_float(f64::from(counts[index]))
        };
        samples[index] = DVec3::new(
            f64::from(position.x),
            f64::from(position.y),
            f64::from(position.z),
        );
    }
    Ok(())
}

fn transform_point_by_triangle(point: DVec3, source: [DVec3; 3], target: [DVec3; 3]) -> Vec3 {
    let point = point.as_vec3();
    let source = source.map(DVec3::as_vec3);
    let target = target.map(DVec3::as_vec3);
    let source_cross = (source[1] - source[0]).cross(source[2] - source[0]);
    let source_normal_length_squared = source_cross.dot(source_cross);
    let source_normal = if source_normal_length_squared > 1.0e-35_f32 {
        source_cross * source_normal_length_squared.sqrt().recip()
    } else {
        Vec3::ZERO
    };
    let xy_length_squared = source_normal.x * source_normal.x + source_normal.y * source_normal.y;
    let (basis_x, basis_y) = if xy_length_squared > f32::EPSILON {
        let inverse_length = xy_length_squared.sqrt().recip();
        let basis_x = Vec3::new(
            source_normal.y * inverse_length,
            -source_normal.x * inverse_length,
            0.0,
        );
        let basis_y = Vec3::new(
            -source_normal.z * basis_x.y,
            source_normal.z * basis_x.x,
            source_normal.x * basis_x.y - source_normal.y * basis_x.x,
        );
        (basis_x, basis_y)
    } else {
        (
            Vec3::new(if source_normal.z < 0.0 { -1.0 } else { 1.0 }, 0.0, 0.0),
            Vec3::Y,
        )
    };
    let project = |position: Vec3| {
        Vec3::new(
            basis_x.dot(position),
            basis_y.dot(position),
            source_normal.dot(position),
        )
    };
    let point_projected = project(point);
    let source_projected = source.map(project);
    let cross_2d = |first: Vec2, second: Vec2, third: Vec2| {
        (first.x - second.x) * (second.y - third.y) - (first.y - second.y) * (second.x - third.x)
    };
    let projected_2d = source_projected.map(|position| Vec2::new(position.x, position.y));
    let point_2d = Vec2::new(point_projected.x, point_projected.y);
    let mut weights = [
        cross_2d(projected_2d[1], projected_2d[2], point_2d),
        cross_2d(projected_2d[2], projected_2d[0], point_2d),
        cross_2d(projected_2d[0], projected_2d[1], point_2d),
    ];
    let total_weight = weights[0] + weights[1] + weights[2];
    if total_weight != 0.0 {
        let inverse_total = total_weight.recip();
        weights
            .iter_mut()
            .for_each(|weight| *weight *= inverse_total);
        if !weights.iter().all(|weight| weight.is_finite()) {
            weights = [1.0 / 3.0; 3];
        }
    } else {
        weights = [1.0 / 3.0; 3];
    }
    let mut transformed = target[0] * weights[0] + target[1] * weights[1] + target[2] * weights[2];
    let target_cross = (target[1] - target[0]).cross(target[2] - target[0]);
    let target_normal_length_squared = target_cross.dot(target_cross);
    let target_normal = if target_normal_length_squared > 1.0e-35_f32 {
        target_cross * target_normal_length_squared.sqrt().recip()
    } else {
        Vec3::ZERO
    };
    let target_area = (target_cross.length() * 0.5).sqrt();
    let projected_source_cross = cross_2d(projected_2d[0], projected_2d[1], projected_2d[2]);
    let source_area = (projected_source_cross.abs() * 0.5).sqrt();
    let source_height = point_projected.z - source_projected[0].z;
    transformed += target_normal * ((source_height / source_area) * target_area);
    transformed
}

#[derive(Clone, Copy)]
enum CacheFormat {
    Pc2 { time_start: f32, sampling: f32 },
    Mdd { times_offset: usize },
}

struct CacheMetadata {
    format: CacheFormat,
    vertex_count: usize,
    frame_count: usize,
    data_offset: usize,
}

impl CacheMetadata {
    fn read_frame(
        &self,
        bytes: &[u8],
        frame_index: usize,
        modifier: &Modifier,
    ) -> Result<Vec<DVec3>> {
        let (sample_requirement, finite_requirement) = match self.format {
            CacheFormat::Pc2 { .. } => ("valid PC2 point samples", "finite PC2 point samples"),
            CacheFormat::Mdd { .. } => ("valid MDD point samples", "finite MDD point samples"),
        };
        let read_f32: fn(&[u8], usize) -> Option<f32> = match self.format {
            CacheFormat::Pc2 { .. } => le_f32,
            CacheFormat::Mdd { .. } => be_f32,
        };
        if frame_index >= self.frame_count {
            return Err(invalid(modifier, "resource", sample_requirement));
        }
        let Some(frame_bytes) = self.vertex_count.checked_mul(12) else {
            return Err(invalid(modifier, "resource", sample_requirement));
        };
        let frame_offset = frame_index
            .checked_mul(frame_bytes)
            .and_then(|offset| self.data_offset.checked_add(offset));
        let Some(frame_offset) = frame_offset else {
            return Err(invalid(modifier, "resource", sample_requirement));
        };
        let Some(frame_end) = frame_offset.checked_add(frame_bytes) else {
            return Err(invalid(modifier, "resource", sample_requirement));
        };
        if frame_end > bytes.len() {
            return Err(invalid(modifier, "resource", sample_requirement));
        }
        let mut points = Vec::with_capacity(self.vertex_count);
        for point_index in 0..self.vertex_count {
            let offset = point_index
                .checked_mul(12)
                .and_then(|offset| frame_offset.checked_add(offset));
            let Some(offset) = offset else {
                return Err(invalid(modifier, "resource", sample_requirement));
            };
            let x = read_f32(bytes, offset);
            let y = offset
                .checked_add(4)
                .and_then(|offset| read_f32(bytes, offset));
            let z = offset
                .checked_add(8)
                .and_then(|offset| read_f32(bytes, offset));
            let (Some(x), Some(y), Some(z)) = (x, y, z) else {
                return Err(invalid(modifier, "resource", sample_requirement));
            };
            let point = DVec3::new(f64::from(x), f64::from(y), f64::from(z));
            if !point.is_finite() {
                return Err(invalid(modifier, "resource", finite_requirement));
            }
            points.push(point);
        }
        Ok(points)
    }
}

fn read_pc2_metadata(bytes: &[u8], modifier: &Modifier) -> Result<CacheMetadata> {
    if bytes.len() < 32 || &bytes[..12] != b"POINTCACHE2\0" {
        return Err(invalid(modifier, "resource", "a valid PC2 file"));
    }
    let count = le_i32(bytes, 16).and_then(|value| usize::try_from(value).ok());
    let time_start = le_f32(bytes, 20);
    let sampling = le_f32(bytes, 24);
    let frames = le_i32(bytes, 28).and_then(|value| usize::try_from(value).ok());
    let (Some(vertex_count), Some(time_start), Some(sampling), Some(frame_count)) =
        (count, time_start, sampling, frames)
    else {
        return Err(invalid(modifier, "resource", "a valid PC2 header"));
    };
    let size = vertex_count
        .checked_mul(frame_count)
        .and_then(|value| value.checked_mul(12))
        .and_then(|value| value.checked_add(32));
    if frame_count == 0 || !sampling.is_finite() || sampling <= 0.0 || size != Some(bytes.len()) {
        return Err(invalid(modifier, "resource", "a valid PC2 point cache"));
    }
    Ok(CacheMetadata {
        format: CacheFormat::Pc2 {
            time_start,
            sampling,
        },
        vertex_count,
        frame_count,
        data_offset: 32,
    })
}

fn read_mdd_metadata(bytes: &[u8], modifier: &Modifier) -> Result<CacheMetadata> {
    if bytes.len() < 8 {
        return Err(invalid(modifier, "resource", "a valid MDD file"));
    }
    let frames = be_i32(bytes, 0).and_then(|value| usize::try_from(value).ok());
    let count = be_i32(bytes, 4).and_then(|value| usize::try_from(value).ok());
    let (Some(frame_count), Some(vertex_count)) = (frames, count) else {
        return Err(invalid(modifier, "resource", "a valid MDD header"));
    };
    let time_bytes = frame_count.checked_mul(4);
    let sample_bytes = frame_count
        .checked_mul(vertex_count)
        .and_then(|value| value.checked_mul(12));
    let data_offset = time_bytes.and_then(|time| 8_usize.checked_add(time));
    let size =
        data_offset.and_then(|offset| sample_bytes.and_then(|samples| offset.checked_add(samples)));
    if frame_count == 0 || size != Some(bytes.len()) {
        return Err(invalid(
            modifier,
            "resource",
            "a valid MDD frame and point table",
        ));
    }
    let Some(data_offset) = data_offset else {
        return Err(invalid(
            modifier,
            "resource",
            "a valid MDD frame and point table",
        ));
    };
    Ok(CacheMetadata {
        format: CacheFormat::Mdd { times_offset: 8 },
        vertex_count,
        frame_count,
        data_offset,
    })
}

fn cache_frame_at_time(
    bytes: &[u8],
    times_offset: usize,
    count: usize,
    time: f32,
    modifier: &Modifier,
) -> Result<f32> {
    let Some(last) = count.checked_sub(1) else {
        return Err(invalid(modifier, "resource", "valid MDD frame times"));
    };
    let mut first_time = None;
    let mut last_time = 0.0_f32;
    let mut previous_time = None;
    let mut upper = None;
    for index in 0..count {
        let offset = index
            .checked_mul(4)
            .and_then(|offset| times_offset.checked_add(offset));
        let time_sample = offset
            .and_then(|offset| be_f32(bytes, offset))
            .filter(|time| time.is_finite())
            .ok_or_else(|| invalid(modifier, "resource", "finite MDD frame times"))?;
        if previous_time.is_some_and(|previous| time_sample < previous) {
            return Err(invalid(modifier, "resource", "ordered MDD frame times"));
        }
        first_time.get_or_insert(time_sample);
        if upper.is_none() && !(time_sample < time) {
            upper = Some((index, time_sample, previous_time.unwrap_or(time_sample)));
        }
        previous_time = Some(time_sample);
        last_time = time_sample;
    }
    let first_time = first_time.unwrap_or(last_time);
    if time <= first_time {
        return Ok(0.0);
    }
    let last_frame = cache_float(f64::from(u32::try_from(last).unwrap_or(u32::MAX)));
    let Some((upper_index, upper_time, lower_time)) = upper else {
        return Ok(last_frame);
    };
    if upper_index == 0 {
        return Ok(0.0);
    }
    let lower = upper_index - 1;
    let width = upper_time - lower_time;
    if width <= 0.0001_f32 {
        return Ok(cache_float(f64::from(
            u32::try_from(upper_index).unwrap_or(u32::MAX),
        )));
    }
    let fraction = (time - lower_time) / width;
    Ok(cache_float(f64::from(u32::try_from(lower).unwrap_or(u32::MAX))) + fraction)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "Blender stores Mesh Cache times and selection factors as f32"
)]
fn cache_float(value: f64) -> f32 {
    value as f32
}

fn select_cache_frame(frame: f32, count: usize, interpolate: bool) -> (usize, usize, f32) {
    let last_index = count.saturating_sub(1);
    let last_frame = cache_float(f64::from(u32::try_from(last_index).unwrap_or(u32::MAX)));
    let frame = frame.clamp(0.0, last_frame);
    let floor = frame.floor();
    let index_frame = if interpolate { floor } else { frame.round() };
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the finite, clamped cache frame is bounded by its signed 32-bit file header"
    )]
    let first = index_frame as usize;
    if !interpolate || first == last_index {
        return (first, first, 0.0);
    }
    let fraction = frame - floor;
    if fraction <= 0.0001_f32 {
        return (first, first, 0.0);
    }
    (first, first + 1, fraction)
}

fn cache_lerp(from: DVec3, to: DVec3, factor: f32) -> DVec3 {
    let inverse_factor = 1.0_f32 - factor;
    DVec3::new(
        f64::from(cache_float(from.x) * inverse_factor + cache_float(to.x) * factor),
        f64::from(cache_float(from.y) * inverse_factor + cache_float(to.y) * factor),
        f64::from(cache_float(from.z) * inverse_factor + cache_float(to.z) * factor),
    )
}

fn cache_frame_count(count: usize, modifier: &Modifier) -> Result<f32> {
    let count = u32::try_from(count)
        .map_err(|_| invalid(modifier, "resource", "a bounded cache frame count"))?;
    Ok(cache_float(f64::from(count)))
}

fn apply_cache_axes(
    point: DVec3,
    forward: &str,
    up: &str,
    flip: [bool; 3],
    modifier: &Modifier,
) -> Result<DVec3> {
    let forward_axis = axis_vector(forward)
        .ok_or_else(|| invalid(modifier, "forward_axis", "a signed Blender axis identifier"))?;
    let up_axis = axis_vector(up)
        .ok_or_else(|| invalid(modifier, "up_axis", "a signed Blender axis identifier"))?;
    let axis_dot = forward_axis.dot(up_axis);
    let mut mapped = if axis_dot.abs() > f64::EPSILON {
        point
    } else {
        let source_right = forward_axis.cross(up_axis).normalize();
        let source_point = source_right * point.x + forward_axis * point.y + up_axis * point.z;
        let target_forward = DVec3::Y;
        let target_up = DVec3::Z;
        let target_right = target_forward.cross(target_up);
        DVec3::new(
            source_point.dot(target_right),
            source_point.dot(target_forward),
            source_point.dot(target_up),
        )
    };
    if flip[0] {
        mapped.x = -mapped.x;
    }
    if flip[1] {
        mapped.y = -mapped.y;
    }
    if flip[2] {
        mapped.z = -mapped.z;
    }
    Ok(mapped)
}

fn face_points(mesh: &Mesh, ids: &[u32]) -> Result<Vec<DVec3>> {
    ids.iter()
        .map(|id| {
            mesh.vertices
                .iter()
                .find(|vertex| vertex.id == *id)
                .map(|vertex| vertex.co)
                .ok_or_else(|| PotError::invalid_argument("mesh face references a missing vertex"))
        })
        .collect()
}

fn face_normal(mesh: &Mesh, face: &Face) -> Result<DVec3> {
    Ok(polygon_normal(&face_points(mesh, &face.vertices)?))
}

fn polygon_normal(points: &[DVec3]) -> DVec3 {
    if points.len() < 3 {
        return DVec3::Z;
    }
    let mut normal = DVec3::ZERO;
    for index in 0..points.len() {
        let current = points[index];
        let next = points[(index + 1) % points.len()];
        normal.x += (current.y - next.y) * (current.z + next.z);
        normal.y += (current.z - next.z) * (current.x + next.x);
        normal.z += (current.x - next.x) * (current.y + next.y);
    }
    normal.normalize_or_zero()
}

fn polygon_area(points: &[DVec3]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }
    let origin = points[0];
    (1..points.len() - 1)
        .map(|index| {
            (points[index] - origin)
                .cross(points[index + 1] - origin)
                .length()
                * 0.5
        })
        .sum()
}

fn corner_angle(mesh: &Mesh, face: &Face, vertex_id: u32) -> Result<f64> {
    let index = face
        .vertices
        .iter()
        .position(|id| *id == vertex_id)
        .ok_or_else(|| PotError::invalid_argument("face corner vertex was not found"))?;
    let points = face_points(mesh, &face.vertices)?;
    let center = points[index];
    let previous = (points[(index + points.len() - 1) % points.len()] - center).normalize_or_zero();
    let next = (points[(index + 1) % points.len()] - center).normalize_or_zero();
    Ok(previous.angle_between(next))
}

fn get_group_weights(mesh: &Mesh, group: &str) -> Result<Option<CacheVertexGroupWeights>> {
    let Some(group_values) = mesh.attributes.get("vertex_groups") else {
        return Ok(None);
    };
    let groups = group_values.as_object().ok_or_else(|| {
        PotError::with_details(
            ErrorCode::InvalidArgument,
            "vertex group weights are malformed",
            json!({"vertex_group":group}),
        )
    })?;
    let Some(group_values) = groups.get(group) else {
        return Ok(None);
    };
    let values = group_values.as_object().ok_or_else(|| {
        PotError::with_details(
            ErrorCode::InvalidArgument,
            format!("vertex group `{group}` has invalid weights"),
            json!({"vertex_group":group}),
        )
    })?;
    let has_deform_weights = groups
        .values()
        .filter_map(Value::as_object)
        .any(|weights| !weights.is_empty());
    let mut result = BTreeMap::new();
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
            return Err(PotError::with_details(
                ErrorCode::InvalidArgument,
                format!("vertex group `{group}` has an invalid weight"),
                json!({"vertex_group":group,"vertex":key}),
            ));
        };
        result.insert(id, cache_float(weight));
    }
    Ok(Some(CacheVertexGroupWeights {
        values: result,
        has_deform_weights,
    }))
}

fn falloff_type(modifier: &Modifier) -> Result<&str> {
    let falloff = string(modifier, "falloff_type")?;
    if matches!(
        falloff,
        "LINEAR" | "CURVE" | "SHARP" | "SMOOTH" | "ROOT" | "ICON_SPHERECURVE" | "RANDOM" | "STEP"
    ) {
        Ok(falloff)
    } else {
        Err(invalid(
            modifier,
            "falloff_type",
            "LINEAR, CURVE, SHARP, SMOOTH, ROOT, ICON_SPHERECURVE, RANDOM, or STEP",
        ))
    }
}

fn curve_points(modifier: &Modifier) -> Result<Option<CurveMappingTable>> {
    let Some(value) = modifier.params.get("map_curve") else {
        return Ok(None);
    };
    let points = curve_mapping_points(value).ok_or_else(|| {
        invalid(
            modifier,
            "map_curve",
            "at least two finite float32 [x, y] points with an optional AUTO, AUTO_CLAMPED, or VECTOR handle",
        )
    })?;
    let settings =
        curve_mapping_settings(modifier.params.get("map_curve_settings")).ok_or_else(|| {
            invalid(
                modifier,
                "map_curve_settings",
                "use_clip, finite ordered clip_bounds, and HORIZONTAL or EXTRAPOLATED extend",
            )
        })?;
    curve_map_table(&points, settings).map(Some).ok_or_else(|| {
        invalid(
            modifier,
            "map_curve",
            "curve controls with evaluable endpoint handles",
        )
    })
}

fn one_operand<'a>(
    operands: &'a BTreeMap<String, Vec<SceneOperand>>,
    name: &str,
    modifier: &Modifier,
) -> Result<&'a SceneOperand> {
    operands
        .get(name)
        .and_then(|values| values.first())
        .ok_or_else(|| invalid(modifier, name, "a resolved scene object operand"))
}

fn string_set(modifier: &Modifier, name: &str) -> Result<Vec<String>> {
    match modifier.params.get(name) {
        None => Ok(Vec::new()),
        Some(Value::String(value)) => Ok(vec![value.clone()]),
        Some(Value::Array(values)) => values
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| invalid(modifier, name, "an array of Blender enum identifiers"))
            })
            .collect(),
        Some(_) => Err(invalid(
            modifier,
            name,
            "an array of Blender enum identifiers",
        )),
    }
}

fn optional_string<'a>(modifier: &'a Modifier, name: &str) -> Result<Option<&'a str>> {
    match modifier.params.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) if value.is_empty() => Ok(None),
        Some(Value::String(value)) => Ok(Some(value)),
        Some(_) => Err(invalid(modifier, name, "a string")),
    }
}

fn vector2(modifier: &Modifier, name: &str) -> Result<DVec2> {
    value2(schema_value(modifier, name)?)
        .filter(|point| point.is_finite())
        .ok_or_else(|| invalid(modifier, name, "two finite numbers"))
}

fn vector3(modifier: &Modifier, name: &str) -> Result<DVec3> {
    value3(schema_value(modifier, name)?)
        .filter(|vector| vector.is_finite())
        .ok_or_else(|| invalid(modifier, name, "three finite numbers"))
}

fn value2(value: &Value) -> Option<DVec2> {
    let values = value.as_array()?;
    Some(DVec2::new(
        values.first()?.as_f64()?,
        values.get(1)?.as_f64()?,
    ))
}

fn value3(value: &Value) -> Option<DVec3> {
    let values = value.as_array()?;
    Some(DVec3::new(
        values.first()?.as_f64()?,
        values.get(1)?.as_f64()?,
        values.get(2)?.as_f64()?,
    ))
}

fn axis_flip_set(value: &Value, modifier: &Modifier) -> Result<[bool; 3]> {
    let values = value
        .as_array()
        .filter(|values| values.len() == 3)
        .ok_or_else(|| invalid(modifier, "flip_axis", "three boolean axis flags"))?;
    let [Some(x), Some(y), Some(z)] = [
        values.first().and_then(Value::as_bool),
        values.get(1).and_then(Value::as_bool),
        values.get(2).and_then(Value::as_bool),
    ] else {
        return Err(invalid(modifier, "flip_axis", "three boolean axis flags"));
    };
    Ok([x, y, z])
}

fn axis_vector(axis: &str) -> Option<DVec3> {
    match axis {
        "POS_X" => Some(DVec3::X),
        "NEG_X" => Some(DVec3::NEG_X),
        "POS_Y" => Some(DVec3::Y),
        "NEG_Y" => Some(DVec3::NEG_Y),
        "POS_Z" => Some(DVec3::Z),
        "NEG_Z" => Some(DVec3::NEG_Z),
        _ => None,
    }
}
fn selected_axis(axis: &str, modifier: &Modifier, parameter: &str) -> Result<DVec3> {
    match axis {
        "X" => Ok(DVec3::X),
        "Y" => Ok(DVec3::Y),
        "Z" => Ok(DVec3::Z),
        _ => Err(invalid(modifier, parameter, "X, Y, or Z")),
    }
}

fn le_i32(bytes: &[u8], offset: usize) -> Option<i32> {
    Some(i32::from_le_bytes(
        bytes.get(offset..offset.checked_add(4)?)?.try_into().ok()?,
    ))
}
fn le_f32(bytes: &[u8], offset: usize) -> Option<f32> {
    Some(f32::from_le_bytes(
        bytes.get(offset..offset.checked_add(4)?)?.try_into().ok()?,
    ))
}
fn be_i32(bytes: &[u8], offset: usize) -> Option<i32> {
    Some(i32::from_be_bytes(
        bytes.get(offset..offset.checked_add(4)?)?.try_into().ok()?,
    ))
}
fn be_f32(bytes: &[u8], offset: usize) -> Option<f32> {
    Some(f32::from_be_bytes(
        bytes.get(offset..offset.checked_add(4)?)?.try_into().ok()?,
    ))
}
