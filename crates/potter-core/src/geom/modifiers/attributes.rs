//! Blender attribute-domain, normal, UV, transfer, and point-cache modifiers.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use glam::{DMat4, DVec2, DVec3};
use serde_json::{Map, Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{Edge, Face, Mesh},
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
    /// f64 object-to-subject transform for native scene modifier paths that operate on `DVec3`.
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
        None,
        DMat4::IDENTITY,
        subject_name,
        &AttributeLayerSelection::default(),
    )
}

/// Evaluate an attribute modifier with an explicit subject identity, its resolved texture, and
/// the texture coordinate mapping that the texture samples through.
///
/// `texture_mapping` is `None` only for UV mapping; a non-UV mask texture without a mapping is
/// an invalid-parameter error.
pub fn evaluate_with_context(
    mesh: &Mesh,
    modifier: &Modifier,
    operands: &BTreeMap<String, Vec<SceneOperand>>,
    texture: Option<&crate::geom::ResolvedTexture>,
    texture_mapping: Option<&crate::model::TextureCoordinateMapping>,
    subject_world: DMat4,
    subject_name: &str,
    subject_layer_selection: &AttributeLayerSelection,
) -> Result<Mesh> {
    // One texture stream per evaluation call; RANDOM falloff streams are separate instances.
    let mut rng = crate::blender_rng::BlenderRng::seed_random(0);
    evaluate_with_texture_context(
        mesh,
        modifier,
        operands,
        texture,
        texture_mapping,
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
    texture: Option<&crate::geom::ResolvedTexture>,
    texture_mapping: Option<&crate::model::TextureCoordinateMapping>,
    subject_world: DMat4,
    subject_name: &str,
    subject_layer_selection: &AttributeLayerSelection,
    rng: &mut crate::blender_rng::BlenderRng,
) -> Result<Mesh> {
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
    let texture_weights = texture_mask_weights(mesh, modifier, texture, texture_mapping, rng)?;
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
pub fn evaluate_mesh_cache(
    mesh: &mut Mesh,
    modifier: &Modifier,
    frame: f64,
    fps: u32,
    fps_base: f64,
    bytes: &[u8],
) -> Result<()> {
    if !frame.is_finite() || fps == 0 || !fps_base.is_finite() || fps_base <= 0.0 {
        return Err(invalid(
            modifier,
            "frame",
            "a finite scene frame and positive frame rate",
        ));
    }
    let format = string(modifier, "cache_format")?;
    let samples = match format {
        "PC2" => read_pc2(bytes, modifier)?,
        "MDD" => read_mdd(bytes, modifier)?,
        _ => return Err(invalid(modifier, "cache_format", "PC2 or MDD")),
    };
    if samples.points.first().map_or(0, Vec::len) != mesh.vertices.len() {
        return Err(invalid(
            modifier,
            "resource",
            "a cache with matching vertex count",
        ));
    }
    let frame_start = number(modifier, "frame_start")?;
    let frame_scale = number(modifier, "frame_scale")?;
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
            let evaluation_frame = if custom {
                number(modifier, "eval_frame")?
            } else {
                frame
            };
            let cache_frame = frame_start + evaluation_frame * frame_scale;
            let cache_time = if format == "MDD" {
                cache_frame * fps_base / f64::from(fps)
            } else {
                cache_frame
            };
            select_cache_samples(&samples.times, cache_time, interpolate)
        }
        "TIME" => {
            let evaluation_time = if custom {
                number(modifier, "eval_time")?
            } else {
                frame * fps_base / f64::from(fps) * frame_scale - frame_start
            };
            if format == "PC2" {
                let start = samples
                    .pc2_time_start
                    .ok_or_else(|| invalid(modifier, "resource", "PC2 time metadata"))?;
                let sampling = samples
                    .pc2_sampling
                    .ok_or_else(|| invalid(modifier, "resource", "PC2 sampling metadata"))?;
                let cache_frame = (evaluation_time / f64::from(fps) - start) / sampling;
                select_cache_frame(cache_frame, samples.points.len(), interpolate)
            } else {
                select_cache_samples(&samples.times, evaluation_time, interpolate)
            }
        }
        _ => {
            let evaluation_factor = if custom {
                number(modifier, "eval_factor")?
            } else {
                frame * fps_base / f64::from(fps) * frame_scale - frame_start
            };
            let frame_count = cache_frame_count(samples.points.len(), modifier)?;
            select_cache_frame(
                evaluation_factor.clamp(0.0, 1.0) * frame_count,
                samples.points.len(),
                interpolate,
            )
        }
    };
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
    let amount = number(modifier, "factor")?.clamp(0.0, 1.0);
    let group = optional_string(modifier, "vertex_group")?;
    let weights = group
        .map(|name| get_group_weights(mesh, name))
        .transpose()?;
    let invert_group = boolean(modifier, "invert_vertex_group")?;
    let basis = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    let sampled = samples.points[first]
        .iter()
        .enumerate()
        .map(|(index, point)| {
            if first == second {
                *point
            } else {
                point.lerp(samples.points[second][index], factor)
            }
        })
        .collect::<Vec<_>>();
    let mut desired = if deform_mode == "INTEGRATE" {
        integrate_cache_samples(mesh, &basis, &sampled)
    } else {
        sampled
    };
    for point in &mut desired {
        *point = apply_cache_axes(*point, forward, up, flip, modifier)?;
    }
    for (index, vertex) in mesh.vertices.iter_mut().enumerate() {
        let mut influence = amount;
        if let Some(weights) = &weights {
            let weight = weights.get(&vertex.id).copied().unwrap_or(0.0);
            influence *= if invert_group { 1.0 - weight } else { weight };
        }
        vertex.co = basis[index].lerp(desired[index], influence);
    }
    Ok(())
}

fn integrate_cache_samples(mesh: &Mesh, basis: &[DVec3], samples: &[DVec3]) -> Vec<DVec3> {
    let vertex_indices = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<HashMap<_, _>>();
    let mut positions = vec![DVec3::ZERO; basis.len()];
    let mut counts = vec![0_u32; basis.len()];
    for face in &mesh.faces {
        let count = face.vertices.len();
        if count < 3 {
            continue;
        }
        for (corner, current_id) in face.vertices.iter().enumerate() {
            let previous_id = face.vertices[(corner + count - 1) % count];
            let next_id = face.vertices[(corner + 1) % count];
            let (Some(&previous), Some(&current), Some(&next)) = (
                vertex_indices.get(&previous_id),
                vertex_indices.get(current_id),
                vertex_indices.get(&next_id),
            ) else {
                continue;
            };
            let source = [basis[previous], basis[current], basis[next]];
            let target = [samples[previous], samples[current], samples[next]];
            positions[current] += transform_cache_point_by_triangle(basis[current], source, target);
            counts[current] = counts[current].saturating_add(1);
        }
    }
    for (index, position) in positions.iter_mut().enumerate() {
        if counts[index] == 0 {
            *position = samples[index];
        } else {
            *position /= f64::from(counts[index]);
        }
    }
    positions
}

fn transform_cache_point_by_triangle(
    point: DVec3,
    source: [DVec3; 3],
    target: [DVec3; 3],
) -> DVec3 {
    let first = source[1] - source[0];
    let second = source[2] - source[0];
    let offset = point - source[0];
    let first_squared = first.length_squared();
    let second_squared = second.length_squared();
    let cross = first.dot(second);
    let denominator = first_squared * second_squared - cross * cross;
    if denominator.abs() <= f64::EPSILON {
        return target[1];
    }
    let first_factor =
        (second_squared * offset.dot(first) - cross * offset.dot(second)) / denominator;
    let second_factor =
        (first_squared * offset.dot(second) - cross * offset.dot(first)) / denominator;
    let factors = [
        1.0 - first_factor - second_factor,
        first_factor,
        second_factor,
    ];
    let mut transformed = target[0] * factors[0] + target[1] * factors[1] + target[2] * factors[2];

    let source_normal = first.cross(second);
    let target_normal = (target[1] - target[0]).cross(target[2] - target[0]);
    let source_area = source_normal.length();
    let target_area = target_normal.length();
    if source_area > f64::EPSILON && target_area > f64::EPSILON {
        let height = offset.dot(source_normal / source_area);
        transformed += target_normal / target_area * height * (target_area / source_area).sqrt();
    }
    transformed
}

struct CacheSamples {
    times: Vec<f64>,
    points: Vec<Vec<DVec3>>,
    pc2_time_start: Option<f64>,
    pc2_sampling: Option<f64>,
}

fn read_pc2(bytes: &[u8], modifier: &Modifier) -> Result<CacheSamples> {
    if bytes.len() < 32 || &bytes[..12] != b"POINTCACHE2\0" {
        return Err(invalid(modifier, "resource", "a valid PC2 file"));
    }
    let version =
        le_i32(bytes, 12).ok_or_else(|| invalid(modifier, "resource", "a valid PC2 header"))?;
    let count = le_i32(bytes, 16).and_then(|value| usize::try_from(value).ok());
    let start = le_f32(bytes, 20).map(f64::from);
    let rate = le_f32(bytes, 24).map(f64::from);
    let frames = le_i32(bytes, 28).and_then(|value| usize::try_from(value).ok());
    let (Some(count), Some(start), Some(rate), Some(frames)) = (count, start, rate, frames) else {
        return Err(invalid(modifier, "resource", "a valid PC2 header"));
    };
    let size = count
        .checked_mul(frames)
        .and_then(|value| value.checked_mul(12))
        .and_then(|value| value.checked_add(32));
    if version != 1 || !rate.is_finite() || rate <= 0.0 || size != Some(bytes.len()) {
        return Err(invalid(
            modifier,
            "resource",
            "a valid PC2 version 1 point cache",
        ));
    }
    let mut points = Vec::with_capacity(frames);
    for frame_index in 0..frames {
        let mut frame = Vec::with_capacity(count);
        for point_index in 0..count {
            let offset = 32 + (frame_index * count + point_index) * 12;
            let x = le_f32(bytes, offset).map(f64::from);
            let y = le_f32(bytes, offset + 4).map(f64::from);
            let z = le_f32(bytes, offset + 8).map(f64::from);
            let (Some(x), Some(y), Some(z)) = (x, y, z) else {
                return Err(invalid(modifier, "resource", "valid PC2 point samples"));
            };
            let point = DVec3::new(x, y, z);
            if !point.is_finite() {
                return Err(invalid(modifier, "resource", "finite PC2 point samples"));
            }
            frame.push(point);
        }
        points.push(frame);
    }
    Ok(CacheSamples {
        times: (0..frames)
            .map(|index| start + f64::from(u32::try_from(index).unwrap_or(u32::MAX)) * rate)
            .collect(),
        points,
        pc2_time_start: Some(start),
        pc2_sampling: Some(rate),
    })
}

fn read_mdd(bytes: &[u8], modifier: &Modifier) -> Result<CacheSamples> {
    if bytes.len() < 8 {
        return Err(invalid(modifier, "resource", "a valid MDD file"));
    }
    let frames = be_i32(bytes, 0).and_then(|value| usize::try_from(value).ok());
    let count = be_i32(bytes, 4).and_then(|value| usize::try_from(value).ok());
    let (Some(frames), Some(count)) = (frames, count) else {
        return Err(invalid(modifier, "resource", "a valid MDD header"));
    };
    let sample_bytes = frames
        .checked_mul(count)
        .and_then(|value| value.checked_mul(12));
    let time_bytes = frames.checked_mul(4);
    if frames == 0
        || count == 0
        || time_bytes.and_then(|time| time.checked_add(sample_bytes?)) != Some(bytes.len() - 8)
    {
        return Err(invalid(
            modifier,
            "resource",
            "a valid MDD frame and point table",
        ));
    }
    let mut times = Vec::with_capacity(frames);
    for index in 0..frames {
        let time = be_f32(bytes, 8 + index * 4).map(f64::from);
        let Some(time) = time.filter(|time| time.is_finite()) else {
            return Err(invalid(modifier, "resource", "finite MDD frame times"));
        };
        times.push(time);
    }
    if times.windows(2).any(|pair| pair[1] < pair[0]) {
        return Err(invalid(modifier, "resource", "ordered MDD frame times"));
    }
    let data_start = 8 + frames * 4;
    let mut points = Vec::with_capacity(frames);
    for frame_index in 0..frames {
        let mut frame = Vec::with_capacity(count);
        for point_index in 0..count {
            let offset = data_start + (frame_index * count + point_index) * 12;
            let x = be_f32(bytes, offset).map(f64::from);
            let y = be_f32(bytes, offset + 4).map(f64::from);
            let z = be_f32(bytes, offset + 8).map(f64::from);
            let (Some(x), Some(y), Some(z)) = (x, y, z) else {
                return Err(invalid(modifier, "resource", "valid MDD point samples"));
            };
            let point = DVec3::new(x, y, z);
            if !point.is_finite() {
                return Err(invalid(modifier, "resource", "finite MDD point samples"));
            }
            frame.push(point);
        }
        points.push(frame);
    }
    Ok(CacheSamples {
        times,
        points,
        pc2_time_start: None,
        pc2_sampling: None,
    })
}

fn select_cache_samples(times: &[f64], time: f64, interpolate: bool) -> (usize, usize, f64) {
    if time <= times[0] {
        return (0, 0, 0.0);
    }
    let last = times.len() - 1;
    if time >= times[last] {
        return (last, last, 0.0);
    }
    let upper = times.partition_point(|sample| *sample < time);
    let lower = upper - 1;
    if !interpolate {
        return (lower, lower, 0.0);
    }
    let width = times[upper] - times[lower];
    let factor = if width > 0.0 {
        (time - times[lower]) / width
    } else {
        0.0
    };
    (lower, upper, factor)
}

fn select_cache_frame(frame: f64, count: usize, interpolate: bool) -> (usize, usize, f64) {
    let last_index = count.saturating_sub(1);
    let last_frame = f64::from(u32::try_from(last_index).unwrap_or(u32::MAX));
    let frame = frame.clamp(0.0, last_frame);
    let index_frame = if interpolate {
        frame.floor()
    } else {
        frame.round()
    };
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the finite, clamped cache frame is bounded by its signed 32-bit file header"
    )]
    let first = index_frame as usize;
    if !interpolate || first == last_index {
        return (first, first, 0.0);
    }
    (first, first + 1, frame.fract())
}

fn cache_frame_count(count: usize, modifier: &Modifier) -> Result<f64> {
    let count = u32::try_from(count)
        .map_err(|_| invalid(modifier, "resource", "a bounded cache frame count"))?;
    Ok(f64::from(count))
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
    if forward_axis.dot(up_axis).abs() > f64::EPSILON {
        return Err(invalid(
            modifier,
            "up_axis",
            "an axis perpendicular to forward_axis",
        ));
    }
    let source_right = forward_axis.cross(up_axis).normalize();
    let source_point = source_right * point.x + forward_axis * point.y + up_axis * point.z;
    let target_forward = DVec3::Y;
    let target_up = DVec3::Z;
    let target_right = target_forward.cross(target_up);
    let mut mapped = DVec3::new(
        source_point.dot(target_right),
        source_point.dot(target_forward),
        source_point.dot(target_up),
    );
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

fn get_group_weights(mesh: &Mesh, group: &str) -> Result<BTreeMap<u32, f64>> {
    let groups = mesh
        .attributes
        .get("vertex_groups")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::InvalidArgument,
                format!("vertex group `{group}` is missing"),
                json!({"vertex_group":group}),
            )
        })?;
    let values = groups
        .get(group)
        .and_then(Value::as_object)
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::InvalidArgument,
                format!("vertex group `{group}` is missing"),
                json!({"vertex_group":group}),
            )
        })?;
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
        result.insert(id, weight);
    }
    Ok(result)
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

fn curve_points(modifier: &Modifier) -> Result<Option<Vec<DVec2>>> {
    let Some(value) = modifier.params.get("map_curve") else {
        return Ok(None);
    };
    let points = value.as_array().ok_or_else(|| {
        invalid(
            modifier,
            "map_curve",
            "an array of [x,y] curve mapping points",
        )
    })?;
    let mut result = Vec::with_capacity(points.len());
    for point in points {
        let point = value2(point)
            .ok_or_else(|| invalid(modifier, "map_curve", "finite [x,y] curve mapping points"))?;
        if !(0.0..=1.0).contains(&point.x) || !(0.0..=1.0).contains(&point.y) {
            return Err(invalid(
                modifier,
                "map_curve",
                "curve points in the unit square",
            ));
        }
        result.push(point);
    }
    result.sort_by(|first, second| first.x.total_cmp(&second.x));
    if result.len() < 2 || result.windows(2).any(|pair| pair[1].x <= pair[0].x) {
        return Err(invalid(
            modifier,
            "map_curve",
            "at least two points with strictly increasing x",
        ));
    }
    Ok(Some(result))
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

fn le_i32(bytes: &[u8], offset: usize) -> Option<i32> {
    Some(i32::from_le_bytes(
        bytes.get(offset..offset + 4)?.try_into().ok()?,
    ))
}
fn le_f32(bytes: &[u8], offset: usize) -> Option<f32> {
    Some(f32::from_le_bytes(
        bytes.get(offset..offset + 4)?.try_into().ok()?,
    ))
}
fn be_i32(bytes: &[u8], offset: usize) -> Option<i32> {
    Some(i32::from_be_bytes(
        bytes.get(offset..offset + 4)?.try_into().ok()?,
    ))
}
fn be_f32(bytes: &[u8], offset: usize) -> Option<f32> {
    Some(f32::from_be_bytes(
        bytes.get(offset..offset + 4)?.try_into().ok()?,
    ))
}
