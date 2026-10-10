//! Blender attribute-domain, normal, UV, transfer, and point-cache modifiers.

use std::collections::{BTreeMap, HashMap};

use glam::{DMat3, DMat4, DVec2, DVec3};
use serde_json::{Map, Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{Edge, Face, Mesh, Vertex},
    model::{CameraData, Modifier},
};

use super::{
    bool_param as boolean, closest_point_triangle, invalid_parameter as invalid,
    string_param as string,
};
#[derive(Clone, Debug)]
/// A resolved scene-object operand for native modifier evaluation.
pub struct SceneOperand {
    /// Object-local evaluated mesh data.
    pub mesh: Mesh,
    /// Matrix from object-local coordinates to subject-local coordinates.
    pub local_to_subject: DMat4,
    /// Camera settings when this operand is a camera object.
    pub camera: Option<CameraData>,
    /// Whether this operand has armature data for UV Warp bone references.
    pub armature: bool,
}

/// Evaluate a modifier that does not need a scene object operand.
///
/// # Errors
///
/// Returns an invalid-parameter or evaluation error when a requested group, UV layer,
/// or attribute is missing or malformed.
pub(super) fn evaluate(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    evaluate_with_context(mesh, modifier, &BTreeMap::new(), None, DMat4::IDENTITY)
}

/// Evaluate an attribute modifier with its resolved scene operands and optional image mask.
///
/// # Errors
///
/// Returns an invalid-parameter, missing-target, or evaluation error for malformed operands.
pub fn evaluate_with_context(
    mesh: &Mesh,
    modifier: &Modifier,
    operands: &BTreeMap<String, Vec<SceneOperand>>,
    mask_image: Option<&crate::image::ImageData>,
    subject_world: DMat4,
) -> Result<Mesh> {
    let texture_weights =
        texture_mask_weights(mesh, modifier, operands, mask_image, subject_world)?;
    match modifier.modifier_type.as_str() {
        "vertex_weight_edit" => vertex_weight_edit(mesh, modifier, texture_weights.as_ref()),
        "vertex_weight_mix" => vertex_weight_mix(mesh, modifier, texture_weights.as_ref()),
        "vertex_weight_proximity" => vertex_weight_proximity(
            mesh,
            modifier,
            operands,
            texture_weights.as_ref(),
            subject_world,
        ),
        "normal_edit" => normal_edit(mesh, modifier, operands),
        "uv_project" => uv_project(mesh, modifier, operands, subject_world),
        "uv_warp" => uv_warp(mesh, modifier, operands),
        "data_transfer" => data_transfer(mesh, modifier, operands),
        "weighted_normal" => weighted_normal(mesh, modifier),
        "mesh_cache" => Err(invalid(
            modifier,
            "resource",
            "a project resource resolved by the evaluation context",
        )),
        _ => Err(invalid(modifier, "type", "a supported attribute modifier")),
    }
}

fn texture_mask_weights(
    mesh: &Mesh,
    modifier: &Modifier,
    operands: &BTreeMap<String, Vec<SceneOperand>>,
    image: Option<&crate::image::ImageData>,
    subject_world: DMat4,
) -> Result<Option<BTreeMap<u32, f64>>> {
    let Some(texture) = modifier.params.get("mask_texture") else {
        return Ok(None);
    };
    if texture.is_null() || texture.as_str().is_some_and(str::is_empty) {
        return Ok(None);
    }
    let image = image.ok_or_else(|| {
        PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "weight modifier texture is not a resolved image texture",
            json!({"feature_id":format!("modifier.{}.texture_type",modifier.modifier_type),"modifier_id":modifier.id}),
        )
    })?;
    let mapping = string(modifier, "mask_tex_mapping", "LOCAL")?;
    let channel = string(modifier, "mask_tex_use_channel", "INT")?;
    let uv_layer = string(modifier, "mask_tex_uv_layer", "UVMap")?;
    let uv_coordinates = if mapping == "UV" {
        Some(vertex_uv_coordinates(mesh, modifier, uv_layer)?)
    } else {
        None
    };
    let object_inverse = if mapping == "OBJECT" {
        let object = operands
            .get("mask_tex_map_object")
            .and_then(|objects| objects.first())
            .ok_or_else(|| {
                invalid(
                    modifier,
                    "mask_tex_map_object",
                    "an object ID when mask_tex_mapping is OBJECT",
                )
            })?;
        let inverse = object.local_to_subject.inverse();
        if !inverse.is_finite() {
            return Err(invalid(
                modifier,
                "mask_tex_map_object",
                "an invertible object transform",
            ));
        }
        Some(inverse)
    } else {
        None
    };
    if !matches!(mapping, "LOCAL" | "GLOBAL" | "OBJECT" | "UV") {
        return Err(invalid(
            modifier,
            "mask_tex_mapping",
            "LOCAL, GLOBAL, OBJECT, or UV",
        ));
    }
    if !matches!(
        channel,
        "INT" | "RED" | "GREEN" | "BLUE" | "HUE" | "SAT" | "VAL" | "ALPHA"
    ) {
        return Err(invalid(
            modifier,
            "mask_tex_use_channel",
            "a Blender image channel",
        ));
    }
    let mut weights = BTreeMap::new();
    for vertex in &mesh.vertices {
        let local = match mapping {
            "GLOBAL" => subject_world.transform_point3(vertex.co),
            "OBJECT" => object_inverse
                .as_ref()
                .map_or(vertex.co, |inverse| inverse.transform_point3(vertex.co)),
            _ => vertex.co,
        };
        let uv = uv_coordinates
            .as_ref()
            .and_then(|coordinates| coordinates.get(&vertex.id).copied());
        let mapping_coordinates =
            uv.map_or(DVec2::new(local.x, local.y), |uv| uv * 2.0 - DVec2::ONE);
        let mut texture_uv = (mapping_coordinates + DVec2::ONE) * 0.5;
        texture_uv.x = repeat_texture_coordinate(texture_uv.x);
        texture_uv.y = repeat_texture_coordinate(texture_uv.y);
        let sample = crate::image::sample(image, texture_uv.to_array(), 0);
        weights.insert(vertex.id, texture_channel(sample, channel));
    }
    Ok(Some(weights))
}

fn vertex_uv_coordinates(
    mesh: &Mesh,
    modifier: &Modifier,
    layer: &str,
) -> Result<BTreeMap<u32, DVec2>> {
    let entries = uv_entries(mesh, modifier)?;
    let mut sums = BTreeMap::<u32, (DVec2, u32)>::new();
    for face in &mesh.faces {
        let Some(entry) = entries.iter().find(|entry| {
            entry.get("face_id").and_then(Value::as_u64) == Some(u64::from(face.id))
                && entry
                    .get("layer")
                    .and_then(Value::as_str)
                    .unwrap_or("UVMap")
                    == layer
        }) else {
            continue;
        };
        let Some(values) = entry.get("uv").and_then(Value::as_array) else {
            continue;
        };
        for (vertex_id, uv) in face.vertices.iter().zip(values) {
            let Some(uv) = value2(uv) else {
                continue;
            };
            let entry = sums.entry(*vertex_id).or_insert((DVec2::ZERO, 0));
            entry.0 += uv;
            entry.1 += 1;
        }
    }
    Ok(sums
        .into_iter()
        .map(|(vertex_id, (sum, count))| (vertex_id, sum / f64::from(count)))
        .collect())
}

fn texture_channel(color: [f64; 4], channel: &str) -> f64 {
    let [red, green, blue, alpha] = color;
    let maximum = red.max(green).max(blue);
    let minimum = red.min(green).min(blue);
    let delta = maximum - minimum;
    let hue = if delta <= f64::EPSILON {
        0.0
    } else if red >= green && red >= blue {
        ((green - blue) / delta).rem_euclid(6.0) / 6.0
    } else if green >= blue {
        ((blue - red) / delta + 2.0) / 6.0
    } else {
        ((red - green) / delta + 4.0) / 6.0
    };
    match channel {
        "RED" => red,
        "GREEN" => green,
        "BLUE" => blue,
        "HUE" => hue,
        "SAT" => {
            if maximum > f64::EPSILON {
                delta / maximum
            } else {
                0.0
            }
        }
        "VAL" => maximum,
        "ALPHA" => alpha,
        _ => 0.2126_f64.mul_add(red, 0.7152_f64.mul_add(green, 0.0722 * blue)),
    }
}

fn repeat_texture_coordinate(value: f64) -> f64 {
    if value.is_finite() {
        value.rem_euclid(1.0)
    } else {
        value
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
    let format = string(modifier, "cache_format", "MDD")?;
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
    let frame_start = number(modifier, "frame_start", 0.0)?;
    let frame_scale = number(modifier, "frame_scale", 1.0)?;
    let play_mode = string(modifier, "play_mode", "SCENE")?;
    if !matches!(play_mode, "SCENE" | "CUSTOM") {
        return Err(invalid(modifier, "play_mode", "SCENE or CUSTOM"));
    }
    let time_mode = string(modifier, "time_mode", "FRAME")?;
    if !matches!(time_mode, "FRAME" | "TIME" | "FACTOR") {
        return Err(invalid(modifier, "time_mode", "FRAME, TIME, or FACTOR"));
    }
    let custom = play_mode == "CUSTOM";
    let interpolation = string(modifier, "interpolation", "LINEAR")?;
    if !matches!(interpolation, "NONE" | "LINEAR") {
        return Err(invalid(modifier, "interpolation", "NONE or LINEAR"));
    }
    let interpolate = interpolation == "LINEAR";
    let (first, second, factor) = match time_mode {
        "FRAME" => {
            let evaluation_frame = if custom {
                number(modifier, "eval_frame", 0.0)?
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
                number(modifier, "eval_time", 0.0)?
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
                number(modifier, "eval_factor", 0.0)?
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
    let forward = string(modifier, "forward_axis", "POS_Y")?;
    let up = string(modifier, "up_axis", "POS_Z")?;
    let flip = modifier
        .params
        .get("flip_axis")
        .map(|value| axis_flip_set(value, modifier))
        .transpose()?
        .unwrap_or([false; 3]);
    let deform_mode = string(modifier, "deform_mode", "OVERWRITE")?;
    if !matches!(deform_mode, "OVERWRITE" | "INTEGRATE") {
        return Err(invalid(modifier, "deform_mode", "OVERWRITE or INTEGRATE"));
    }
    let amount = number(modifier, "factor", 1.0)?.clamp(0.0, 1.0);
    let group = optional_string(modifier, "vertex_group")?;
    let weights = group
        .map(|name| get_group_weights(mesh, name))
        .transpose()?;
    let invert_group = boolean(modifier, "invert_vertex_group", false)?;
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

fn vertex_weight_edit(
    mesh: &Mesh,
    modifier: &Modifier,
    texture_weights: Option<&BTreeMap<u32, f64>>,
) -> Result<Mesh> {
    let group = string(modifier, "vertex_group", "")?;
    if group.is_empty() {
        return Err(invalid(modifier, "vertex_group", "a vertex-group name"));
    }
    let falloff = falloff_type(modifier)?;
    let invert_falloff = boolean(modifier, "invert_falloff", false)?;
    let use_add = boolean(modifier, "use_add", false)?;
    let use_remove = boolean(modifier, "use_remove", false)?;
    let add_threshold = number(modifier, "add_threshold", 0.01)?;
    let remove_threshold = number(modifier, "remove_threshold", 0.01)?;
    let default = number(modifier, "default_weight", 0.0)?.clamp(0.0, 1.0);
    let normalize = boolean(modifier, "normalize", false)?;
    let mask_constant = number(modifier, "mask_constant", 1.0)?.clamp(0.0, 1.0);
    let mask_group = optional_string(modifier, "mask_vertex_group")?;
    let mask_invert = boolean(modifier, "invert_mask_vertex_group", false)?;
    let curve = curve_points(modifier)?;
    let mut result = mesh.clone();
    let existing = get_group_weights(&result, group)?;
    let mask_weights = mask_group
        .map(|name| get_group_weights(&result, name))
        .transpose()?;
    let mut updated = BTreeMap::new();
    for vertex in &result.vertices {
        let assigned = existing.get(&vertex.id).copied();
        let (original, mut weight) = match assigned {
            Some(weight) => (weight, weight),
            None if use_add && default > add_threshold => (0.0, default),
            None => continue,
        };
        weight = apply_falloff(weight, falloff, curve.as_deref(), u64::from(vertex.id));
        if invert_falloff {
            weight = 1.0 - weight;
        }
        let mask = mask_weights.as_ref().map_or(1.0, |weights| {
            let value = weights.get(&vertex.id).copied().unwrap_or(0.0);
            if mask_invert { 1.0 - value } else { value }
        });
        let texture_mask = texture_weights.map_or(1.0, |weights| {
            weights.get(&vertex.id).copied().unwrap_or(0.0)
        });
        let influence = (mask * mask_constant * texture_mask).clamp(0.0, 1.0);
        let final_weight = if assigned.is_none() {
            weight
        } else {
            original + (weight - original) * influence
        };
        if use_remove && final_weight < remove_threshold {
            continue;
        }
        updated.insert(vertex.id, final_weight.clamp(0.0, 1.0));
    }
    write_group_weights(&mut result, group, &updated)?;
    if normalize {
        normalize_group(&mut result, group)?;
    }
    Ok(result)
}

fn vertex_weight_mix(
    mesh: &Mesh,
    modifier: &Modifier,
    texture_weights: Option<&BTreeMap<u32, f64>>,
) -> Result<Mesh> {
    let group_a = string(modifier, "vertex_group_a", "")?;
    let group_b = string(modifier, "vertex_group_b", "")?;
    if group_a.is_empty() || group_b.is_empty() {
        return Err(invalid(
            modifier,
            "vertex_group_a",
            "two vertex-group names",
        ));
    }
    let mode = string(modifier, "mix_mode", "SET")?;
    let mix_set = string(modifier, "mix_set", "AND")?;
    let default_a = number(modifier, "default_weight_a", 0.0)?.clamp(0.0, 1.0);
    let default_b = number(modifier, "default_weight_b", 0.0)?.clamp(0.0, 1.0);
    let invert_a = boolean(modifier, "invert_vertex_group_a", false)?;
    let invert_b = boolean(modifier, "invert_vertex_group_b", false)?;
    let normalize = boolean(modifier, "normalize", false)?;
    let mask_constant = number(modifier, "mask_constant", 1.0)?.clamp(0.0, 1.0);
    let mask_group = optional_string(modifier, "mask_vertex_group")?;
    let mask_invert = boolean(modifier, "invert_mask_vertex_group", false)?;
    let mut result = mesh.clone();
    let weights_a = get_group_weights(&result, group_a)?;
    let weights_b = get_group_weights_if_present(&result, group_b)?;
    let mask_weights = mask_group
        .map(|name| get_group_weights(&result, name))
        .transpose()?;
    let mut updated = BTreeMap::new();
    for vertex in &result.vertices {
        let a_assigned = weights_a.contains_key(&vertex.id);
        let b_assigned = weights_b.contains_key(&vertex.id);
        let selected = match mix_set {
            "ALL" => true,
            "A" => a_assigned,
            "B" => b_assigned,
            "OR" => a_assigned || b_assigned,
            "AND" => a_assigned && b_assigned,
            _ => return Err(invalid(modifier, "mix_set", "ALL, A, B, OR, or AND")),
        };
        if !selected {
            continue;
        }
        let raw_a = weights_a.get(&vertex.id).copied().unwrap_or(default_a);
        let raw_b = weights_b.get(&vertex.id).copied().unwrap_or(default_b);
        let a = if invert_a { 1.0 - raw_a } else { raw_a };
        let b = if invert_b { 1.0 - raw_b } else { raw_b };
        let mixed = mix_weights(a, b, mode, modifier)?;
        let mask = mask_weights.as_ref().map_or(1.0, |weights| {
            let value = weights.get(&vertex.id).copied().unwrap_or(0.0);
            if mask_invert { 1.0 - value } else { value }
        });
        let texture_mask = texture_weights.map_or(1.0, |weights| {
            weights.get(&vertex.id).copied().unwrap_or(0.0)
        });
        let blend = (a + (mixed - a) * mask * mask_constant * texture_mask).clamp(0.0, 1.0);
        updated.insert(vertex.id, blend);
    }
    write_group_weights(&mut result, group_a, &updated)?;
    if normalize {
        normalize_group(&mut result, group_a)?;
    }
    Ok(result)
}

fn mix_weights(a: f64, b: f64, mode: &str, modifier: &Modifier) -> Result<f64> {
    match mode {
        "SET" => Ok(b),
        "ADD" => Ok(a + b),
        "SUB" => Ok(a - b),
        "MUL" => Ok(a * b),
        "DIV" => Ok(if b.abs() > f64::EPSILON { a / b } else { a }),
        "DIF" => Ok((a - b).abs()),
        "AVG" => Ok(f64::midpoint(a, b)),
        "MIN" => Ok(a.min(b)),
        "MAX" => Ok(a.max(b)),
        _ => Err(invalid(
            modifier,
            "mix_mode",
            "SET, ADD, SUB, MUL, DIV, DIF, AVG, MIN, or MAX",
        )),
    }
}

fn vertex_weight_proximity(
    mesh: &Mesh,
    modifier: &Modifier,
    operands: &BTreeMap<String, Vec<SceneOperand>>,
    texture_weights: Option<&BTreeMap<u32, f64>>,
    subject_world: DMat4,
) -> Result<Mesh> {
    let target = one_operand(operands, "target", modifier)?;
    let group = string(modifier, "vertex_group", "")?;
    if group.is_empty() {
        return Err(invalid(modifier, "vertex_group", "a vertex-group name"));
    }
    let mode = string(modifier, "proximity_mode", "GEOMETRY")?;
    let mut geometry = string_set(modifier, "proximity_geometry")?;
    if geometry.is_empty() {
        geometry.push("FACE".to_owned());
    }
    if !matches!(mode, "OBJECT" | "GEOMETRY") {
        return Err(invalid(modifier, "proximity_mode", "OBJECT or GEOMETRY"));
    }
    if geometry
        .iter()
        .any(|kind| !matches!(kind.as_str(), "VERTEX" | "EDGE" | "FACE"))
    {
        return Err(invalid(
            modifier,
            "proximity_geometry",
            "a set containing VERTEX, EDGE, or FACE",
        ));
    }
    let minimum = number(modifier, "min_dist", 0.0)?;
    let maximum = number(modifier, "max_dist", 1.0)?;
    if minimum < 0.0 || maximum < minimum {
        return Err(invalid(
            modifier,
            "max_dist",
            "a distance greater than or equal to min_dist",
        ));
    }
    let falloff = falloff_type(modifier)?;
    let invert_falloff = boolean(modifier, "invert_falloff", false)?;
    let curve = curve_points(modifier)?;
    let mask_constant = number(modifier, "mask_constant", 1.0)?.clamp(0.0, 1.0);
    let mask_group = optional_string(modifier, "mask_vertex_group")?;
    let mask_invert = boolean(modifier, "invert_mask_vertex_group", false)?;
    let mask_weights = mask_group
        .map(|name| get_group_weights(mesh, name))
        .transpose()?;
    let existing = get_group_weights(mesh, group)?;
    let mut result = mesh.clone();
    let mut updated = BTreeMap::new();
    let target_points = target
        .mesh
        .vertices
        .iter()
        .map(|vertex| target.local_to_subject.transform_point3(vertex.co))
        .collect::<Vec<_>>();
    let target_edges = target
        .mesh
        .edges
        .iter()
        .filter_map(|edge| {
            let a = target
                .mesh
                .vertices
                .iter()
                .find(|vertex| vertex.id == edge.vertices[0])?;
            let b = target
                .mesh
                .vertices
                .iter()
                .find(|vertex| vertex.id == edge.vertices[1])?;
            Some([
                target.local_to_subject.transform_point3(a.co),
                target.local_to_subject.transform_point3(b.co),
            ])
        })
        .collect::<Vec<_>>();
    let target_faces = target
        .mesh
        .faces
        .iter()
        .filter_map(|face| {
            let points = face
                .vertices
                .iter()
                .filter_map(|id| target.mesh.vertices.iter().find(|vertex| vertex.id == *id))
                .map(|vertex| target.local_to_subject.transform_point3(vertex.co))
                .collect::<Vec<_>>();
            (points.len() == face.vertices.len()).then_some(points)
        })
        .collect::<Vec<_>>();
    let subject_origin_world = subject_world.transform_point3(DVec3::ZERO);
    let target_origin_world =
        subject_world.transform_point3(target.local_to_subject.transform_point3(DVec3::ZERO));
    let object_distance = subject_origin_world.distance(target_origin_world);
    for vertex in &result.vertices {
        let distance = if mode == "OBJECT" {
            object_distance
        } else {
            distance_to_target(
                vertex.co,
                &geometry,
                &target_points,
                &target_edges,
                &target_faces,
            )
        };
        let t = if maximum <= minimum {
            f64::from(distance >= maximum)
        } else {
            ((distance - minimum) / (maximum - minimum)).clamp(0.0, 1.0)
        };
        let mut weight =
            apply_falloff(t, falloff, curve.as_deref(), u64::from(vertex.id)).clamp(0.0, 1.0);
        if invert_falloff {
            weight = 1.0 - weight;
        }
        let mask = mask_weights.as_ref().map_or(1.0, |weights| {
            let value = weights.get(&vertex.id).copied().unwrap_or(0.0);
            if mask_invert { 1.0 - value } else { value }
        });
        let texture_mask = texture_weights.map_or(1.0, |weights| {
            weights.get(&vertex.id).copied().unwrap_or(0.0)
        });
        let influence = (mask * mask_constant * texture_mask).clamp(0.0, 1.0);
        let original = existing.get(&vertex.id).copied().unwrap_or(0.0);
        updated.insert(
            vertex.id,
            (original + (weight - original) * influence).clamp(0.0, 1.0),
        );
    }
    write_group_weights(&mut result, group, &updated)?;
    if boolean(modifier, "normalize", false)? {
        normalize_group(&mut result, group)?;
    }
    Ok(result)
}

fn distance_to_target(
    point: DVec3,
    geometry: &[String],
    vertices: &[DVec3],
    edges: &[[DVec3; 2]],
    faces: &[Vec<DVec3>],
) -> f64 {
    let mut distance = f64::INFINITY;
    if geometry.iter().any(|kind| kind == "VERTEX") {
        distance = distance.min(
            vertices
                .iter()
                .map(|vertex| point.distance(*vertex))
                .fold(f64::INFINITY, f64::min),
        );
    }
    if geometry.iter().any(|kind| kind == "EDGE") {
        distance = distance.min(
            edges
                .iter()
                .map(|edge| point_segment_distance(point, edge[0], edge[1]))
                .fold(f64::INFINITY, f64::min),
        );
    }
    if geometry.iter().any(|kind| kind == "FACE") {
        distance = distance.min(
            faces
                .iter()
                .map(|face| {
                    (1..face.len().saturating_sub(1))
                        .map(|index| {
                            closest_point_triangle(point, face[0], face[index], face[index + 1])
                                .distance(point)
                        })
                        .fold(f64::INFINITY, f64::min)
                })
                .fold(f64::INFINITY, f64::min),
        );
    }
    distance
}

fn point_segment_distance(point: DVec3, first: DVec3, second: DVec3) -> f64 {
    let segment = second - first;
    let length = segment.length_squared();
    let factor = if length > f64::EPSILON {
        ((point - first).dot(segment) / length).clamp(0.0, 1.0)
    } else {
        0.0
    };
    point.distance(first + segment * factor)
}

fn weighted_normal(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    let mode = string(modifier, "mode", "FACE_AREA")?;
    if !matches!(mode, "FACE_AREA" | "CORNER_ANGLE" | "FACE_AREA_WITH_ANGLE") {
        return Err(invalid(
            modifier,
            "mode",
            "FACE_AREA, CORNER_ANGLE, or FACE_AREA_WITH_ANGLE",
        ));
    }
    let power = number(modifier, "weight", 50.0)?.clamp(1.0, 100.0) / 50.0;
    let threshold = number(modifier, "thresh", 0.01)?.max(0.0);
    let group = optional_string(modifier, "vertex_group")?;
    let group_weights = group
        .map(|name| get_group_weights(mesh, name))
        .transpose()?;
    let invert_group = boolean(modifier, "invert_vertex_group", false)?;
    let keep_sharp = boolean(modifier, "keep_sharp", false)?;
    let use_face_influence = boolean(modifier, "use_face_influence", false)?;
    let sharp_edges = mesh
        .attributes
        .get("sharp_edges")
        .and_then(Value::as_array)
        .map(|edges| {
            edges
                .iter()
                .filter_map(Value::as_u64)
                .filter_map(|value| u32::try_from(value).ok())
                .collect::<std::collections::BTreeSet<_>>()
        })
        .unwrap_or_default();
    let mut normals = Vec::with_capacity(mesh.faces.len());
    let mut areas = Vec::with_capacity(mesh.faces.len());
    for face in &mesh.faces {
        let points = face_points(mesh, face.vertices.as_slice())?;
        let normal = polygon_normal(&points);
        normals.push(normal);
        areas.push(polygon_area(&points));
    }
    let face_strengths = mesh
        .attributes
        .get("__mod_weightednormals_faceweight")
        .and_then(|attribute| attribute.get("values"))
        .and_then(Value::as_object);
    let mut incident = HashMap::<u32, Vec<usize>>::new();
    for (face_index, face) in mesh.faces.iter().enumerate() {
        for vertex_id in &face.vertices {
            incident.entry(*vertex_id).or_default().push(face_index);
        }
    }
    let mut corner_values = Map::new();
    for (face_index, face) in mesh.faces.iter().enumerate() {
        let mut corners = Vec::with_capacity(face.vertices.len());
        for vertex_id in &face.vertices {
            let current = normals[face_index];
            let incident_faces = incident.get(vertex_id).map_or(&[][..], Vec::as_slice);
            let fan = if keep_sharp {
                smooth_fan_faces(mesh, *vertex_id, face_index, incident_faces, &sharp_edges)
            } else {
                Vec::new()
            };
            let adjacent = if keep_sharp {
                fan.as_slice()
            } else {
                incident_faces
            };
            let maximum_strength = use_face_influence.then(|| {
                adjacent
                    .iter()
                    .map(|index| {
                        face_strengths
                            .and_then(|values| values.get(&format!("f{}", mesh.faces[*index].id)))
                            .and_then(Value::as_i64)
                            .unwrap_or(0)
                    })
                    .max()
                    .unwrap_or(0)
            });
            let mut sum = DVec3::ZERO;
            for adjacent_index in adjacent {
                if maximum_strength.is_some_and(|maximum| {
                    face_strengths
                        .and_then(|values| {
                            values.get(&format!("f{}", mesh.faces[*adjacent_index].id))
                        })
                        .and_then(Value::as_i64)
                        .unwrap_or(0)
                        != maximum
                }) {
                    continue;
                }
                let adjacent_face = &mesh.faces[*adjacent_index];
                let angle = corner_angle(mesh, adjacent_face, *vertex_id)?;
                let weight = match mode {
                    "FACE_AREA" => areas[*adjacent_index],
                    "CORNER_ANGLE" => angle,
                    _ => areas[*adjacent_index] * angle,
                }
                .max(0.0)
                .powf(power);
                if weight <= threshold {
                    continue;
                }
                sum += normals[*adjacent_index] * weight;
            }
            let mut normal = if sum.length_squared() <= f64::EPSILON {
                current
            } else {
                sum.normalize()
            };
            if let Some(weights) = &group_weights {
                let mut factor = weights
                    .get(vertex_id)
                    .copied()
                    .unwrap_or(0.0)
                    .clamp(0.0, 1.0);
                if invert_group {
                    factor = 1.0 - factor;
                }
                normal = current.lerp(normal, factor).normalize_or_zero();
            }
            corners.push(json!(normal.to_array()));
        }
        corner_values.insert(format!("f{}", face.id), Value::Array(corners));
    }
    let mut result = mesh.clone();
    result.attributes.insert(
        "custom_normal".to_owned(),
        json!({"domain":"corner","type":"float3","values":corner_values}),
    );
    Ok(result)
}

fn edge_is_sharp(
    mesh: &Mesh,
    first: u32,
    second: u32,
    sharp_edges: &std::collections::BTreeSet<u32>,
) -> bool {
    mesh.edges.iter().any(|edge| {
        ((edge.vertices[0] == first && edge.vertices[1] == second)
            || (edge.vertices[0] == second && edge.vertices[1] == first))
            && sharp_edges.contains(&edge.id)
    })
}

fn faces_share_smooth_edge_at_vertex(
    mesh: &Mesh,
    first: &Face,
    second: &Face,
    vertex_id: u32,
    sharp_edges: &std::collections::BTreeSet<u32>,
) -> bool {
    let Some(index) = first.vertices.iter().position(|id| *id == vertex_id) else {
        return false;
    };
    let neighbors = [
        first.vertices[(index + first.vertices.len() - 1) % first.vertices.len()],
        first.vertices[(index + 1) % first.vertices.len()],
    ];
    neighbors.into_iter().any(|neighbor| {
        !edge_is_sharp(mesh, vertex_id, neighbor, sharp_edges)
            && (0..second.vertices.len()).any(|corner| {
                let current = second.vertices[corner];
                let next = second.vertices[(corner + 1) % second.vertices.len()];
                (current == vertex_id && next == neighbor)
                    || (current == neighbor && next == vertex_id)
            })
    })
}

fn smooth_fan_faces(
    mesh: &Mesh,
    vertex_id: u32,
    start: usize,
    incident: &[usize],
    sharp_edges: &std::collections::BTreeSet<u32>,
) -> Vec<usize> {
    let mut fan = vec![start];
    let mut cursor = 0;
    while cursor < fan.len() {
        let current = &mesh.faces[fan[cursor]];
        for candidate in incident {
            if !fan.contains(candidate)
                && faces_share_smooth_edge_at_vertex(
                    mesh,
                    current,
                    &mesh.faces[*candidate],
                    vertex_id,
                    sharp_edges,
                )
            {
                fan.push(*candidate);
            }
        }
        cursor += 1;
    }
    fan
}

fn normal_edit(
    mesh: &Mesh,
    modifier: &Modifier,
    operands: &BTreeMap<String, Vec<SceneOperand>>,
) -> Result<Mesh> {
    let target = one_operand(operands, "target", modifier)?;
    let mode = string(modifier, "mode", "RADIAL")?;
    if !matches!(mode, "RADIAL" | "DIRECTIONAL") {
        return Err(invalid(modifier, "mode", "RADIAL or DIRECTIONAL"));
    }
    let offset = vector3(modifier, "offset", DVec3::ZERO)?;
    let mix_mode = string(modifier, "mix_mode", "COPY")?;
    let factor = number(modifier, "mix_factor", 1.0)?.clamp(0.0, 1.0);
    let limit = number(modifier, "mix_limit", std::f64::consts::PI)?;
    let parallel = boolean(modifier, "use_direction_parallel", false)?;
    let group = optional_string(modifier, "vertex_group")?;
    let group_weights = group
        .map(|name| get_group_weights(mesh, name))
        .transpose()?;
    let invert_group = boolean(modifier, "invert_vertex_group", false)?;
    let target_origin = target.local_to_subject.transform_point3(DVec3::ZERO);
    let target_direction = (target_origin - offset).normalize_or_zero();
    let mut values = Map::new();
    for face in &mesh.faces {
        let base = face_normal(mesh, face)?;
        let mut corners = Vec::with_capacity(face.vertices.len());
        for vertex_id in &face.vertices {
            let vertex = mesh
                .vertices
                .iter()
                .find(|vertex| vertex.id == *vertex_id)
                .ok_or_else(|| invalid(modifier, "target", "a mesh with valid face vertices"))?;
            let destination = if mode == "RADIAL" || !parallel {
                (vertex.co - target_origin).normalize_or_zero()
            } else {
                target_direction
            };
            let mut blend_factor = factor;
            let mixed = match mix_mode {
                "COPY" => destination,
                "ADD" => base + destination,
                "SUB" => base - destination,
                "MUL" => base * destination,
                _ => return Err(invalid(modifier, "mix_mode", "COPY, ADD, SUB, or MUL")),
            }
            .normalize_or_zero();
            if let Some(weights) = &group_weights {
                let weight = weights.get(vertex_id).copied().unwrap_or(0.0);
                blend_factor *= if invert_group { 1.0 - weight } else { weight };
            }
            if limit < std::f64::consts::PI {
                let angle = base.angle_between(mixed);
                if angle > f64::EPSILON {
                    blend_factor = blend_factor.min(limit / angle);
                }
            }
            let normal = base.slerp(mixed, blend_factor).normalize_or_zero();
            corners.push(json!(normal.to_array()));
        }
        values.insert(format!("f{}", face.id), Value::Array(corners));
    }
    let mut result = mesh.clone();
    result.attributes.insert(
        "custom_normal".to_owned(),
        json!({"domain":"corner","type":"float3","values":values}),
    );
    Ok(result)
}

fn uv_project(
    mesh: &Mesh,
    modifier: &Modifier,
    operands: &BTreeMap<String, Vec<SceneOperand>>,
    subject_world: DMat4,
) -> Result<Mesh> {
    let projectors = operands
        .get("projectors")
        .ok_or_else(|| invalid(modifier, "projectors", "one or more camera object IDs"))?;
    if projectors.is_empty() {
        return Err(invalid(
            modifier,
            "projectors",
            "one or more camera object IDs",
        ));
    }
    // Blender clamps these RNA properties to a hard minimum of 1.0.
    let aspect_x = number(modifier, "aspect_x", 1.0)?.max(1.0);
    let aspect_y = number(modifier, "aspect_y", 1.0)?.max(1.0);
    let scale_x = number(modifier, "scale_x", 1.0)?;
    let scale_y = number(modifier, "scale_y", 1.0)?;
    if scale_x <= 0.0 || scale_y <= 0.0 {
        return Err(invalid(modifier, "scale_x", "positive projection scales"));
    }
    let layer = string(modifier, "uv_layer", "UVMap")?;
    let mut entries = uv_entries(mesh, modifier)?;
    for face in &mesh.faces {
        let normal = face_normal(mesh, face)?;
        // Blender keeps the first projector when two camera directions tie.
        let mut best_index = 0;
        let mut best_dot = (subject_world * projectors[0].local_to_subject)
            .transform_vector3(DVec3::Z)
            .dot(normal);
        for (index, projector) in projectors.iter().enumerate().skip(1) {
            let projector_normal =
                (subject_world * projector.local_to_subject).transform_vector3(DVec3::Z);
            let dot = projector_normal.dot(normal);
            if dot > best_dot {
                best_index = index;
                best_dot = dot;
            }
        }
        let projector = &projectors[best_index];
        let mut coords = Vec::with_capacity(face.vertices.len());
        for vertex_id in &face.vertices {
            let vertex = mesh
                .vertices
                .iter()
                .find(|vertex| vertex.id == *vertex_id)
                .ok_or_else(|| invalid(modifier, "projectors", "valid face vertices"))?;
            let uv = project_uv_point(
                vertex.co,
                projector,
                subject_world,
                aspect_x,
                aspect_y,
                scale_x,
                scale_y,
                modifier,
            )?
            .unwrap_or(DVec2::ZERO);
            coords.push(json!([uv.x, uv.y]));
        }
        insert_uv(&mut entries, face.id, layer, Value::Array(coords));
    }
    let mut result = mesh.clone();
    result
        .attributes
        .insert("uv_map".to_owned(), Value::Array(entries));
    Ok(result)
}

#[expect(
    clippy::manual_midpoint,
    reason = "matches Blender's float evaluation order for parity"
)]
fn project_uv_point(
    point: DVec3,
    projector: &SceneOperand,
    subject_world: DMat4,
    aspect_x: f64,
    aspect_y: f64,
    scale_x: f64,
    scale_y: f64,
    modifier: &Modifier,
) -> Result<Option<DVec2>> {
    let projector_world = subject_world * projector.local_to_subject;
    let world_to_projector = projector_world.inverse();
    if !world_to_projector.is_finite() {
        return Err(invalid(
            modifier,
            "projectors",
            "invertible projector transforms",
        ));
    }
    let world = subject_world.transform_point3(point);
    let local = world_to_projector.transform_point3(world);
    let Some(camera) = &projector.camera else {
        return Ok(Some(DVec2::new(
            (local.x + 1.0) * 0.5,
            (local.y + 1.0) * 0.5,
        )));
    };
    let ycor = aspect_y / aspect_x;
    let sensor_fit = match camera.sensor_fit.as_str() {
        "HORIZONTAL" => "HORIZONTAL",
        "VERTICAL" => "VERTICAL",
        _ if ycor <= 1.0 => "HORIZONTAL",
        _ => "VERTICAL",
    };
    let view_factor = if sensor_fit == "HORIZONTAL" {
        1.0
    } else {
        ycor
    };
    let shift_x = camera.shift[0];
    let shift_y = camera.shift[1];
    let ndc = match camera.projection {
        crate::model::CameraProjection::Orthographic => DVec2::new(
            2.0 * local.x * view_factor / (camera.ortho_scale * scale_x)
                - 2.0 * shift_x * view_factor,
            2.0 * local.y * view_factor / (ycor * camera.ortho_scale * scale_y)
                - 2.0 * shift_y * view_factor / ycor,
        ),
        crate::model::CameraProjection::Perspective => {
            let depth = -local.z;
            if depth.abs() <= f64::EPSILON {
                return Ok(None);
            }
            let sensor_size = if sensor_fit == "VERTICAL" {
                camera.sensor_height_mm
            } else {
                camera.sensor_width_mm
            };
            DVec2::new(
                2.0 * local.x * view_factor * camera.lens_mm / (depth * sensor_size * scale_x)
                    - 2.0 * shift_x * view_factor,
                2.0 * local.y * view_factor * camera.lens_mm
                    / (depth * ycor * sensor_size * scale_y)
                    - 2.0 * shift_y * view_factor / ycor,
            )
        }
        crate::model::CameraProjection::Panorama | crate::model::CameraProjection::Fisheye => {
            return Err(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                "UV Project does not model panoramic or fisheye camera projections",
                json!({"feature_id":"modifier.uv_project.camera_projection","modifier_id":modifier.id}),
            ));
        }
    };
    Ok(Some((ndc + DVec2::ONE) * 0.5))
}

fn uv_warp(
    mesh: &Mesh,
    modifier: &Modifier,
    operands: &BTreeMap<String, Vec<SceneOperand>>,
) -> Result<Mesh> {
    let center = vector2(modifier, "center", DVec2::splat(0.5))?;
    let axis_u = string(modifier, "axis_u", "X")?;
    let axis_v = string(modifier, "axis_v", "Y")?;
    let axis_u = selected_axis(axis_u, modifier, "axis_u")?;
    let axis_v = selected_axis(axis_v, modifier, "axis_v")?;
    let offset = vector2(modifier, "offset", DVec2::ZERO)?;
    let scale = vector2(modifier, "scale", DVec2::ONE)?;
    let rotation = number(modifier, "rotation", 0.0)?;
    let layer = string(modifier, "uv_layer", "UVMap")?;
    let group = optional_string(modifier, "vertex_group")?;
    let weights = group
        .map(|name| get_group_weights(mesh, name))
        .transpose()?;
    let invert_group = boolean(modifier, "invert_vertex_group", false)?;
    let from = operands.get("object_from").and_then(|items| items.first());
    let to = operands.get("object_to").and_then(|items| items.first());
    check_uv_warp_bone_binding(modifier, operands, "bone_from", "object_from")?;
    check_uv_warp_bone_binding(modifier, operands, "bone_to", "object_to")?;
    let object_transform = match (from, to) {
        (Some(from), Some(to)) => {
            let inverse = to.local_to_subject.inverse();
            if !inverse.is_finite() {
                return Err(invalid(
                    modifier,
                    "object_to",
                    "an invertible object transform",
                ));
            }
            Some(inverse * from.local_to_subject)
        }
        _ => None,
    };
    let mut entries = uv_entries(mesh, modifier)?;
    for face in &mesh.faces {
        let mut uv = Vec::with_capacity(face.vertices.len());
        for (corner, vertex_id) in face.vertices.iter().enumerate() {
            let base = existing_uv(mesh, face.id, layer, corner).unwrap_or(DVec2::ZERO);
            let mut mask = weights
                .as_ref()
                .map_or(1.0, |values| values.get(vertex_id).copied().unwrap_or(0.0))
                .clamp(0.0, 1.0);
            if invert_group {
                mask = 1.0 - mask;
            }
            let relative = base - center + offset;
            let rotated = DMat3::from_rotation_z(rotation)
                .mul_vec3(DVec3::new(relative.x, relative.y, 0.0))
                .truncate();
            let direct = center + DVec2::new(rotated.x * scale.x, rotated.y * scale.y);
            let transformed = object_transform.map_or(direct, |transform| {
                let center_3d = axis_u * center.x + axis_v * center.y;
                let direct_3d = axis_u * direct.x + axis_v * direct.y;
                let mapped = center_3d + transform.transform_point3(direct_3d - center_3d);
                DVec2::new(mapped.dot(axis_u), mapped.dot(axis_v))
            });
            let output = base.lerp(transformed, mask);
            uv.push(json!([output.x, output.y]));
        }
        insert_uv(&mut entries, face.id, layer, Value::Array(uv));
    }
    let mut result = mesh.clone();
    result
        .attributes
        .insert("uv_map".to_owned(), Value::Array(entries));
    Ok(result)
}
fn check_uv_warp_bone_binding(
    modifier: &Modifier,
    operands: &BTreeMap<String, Vec<SceneOperand>>,
    bone_parameter: &str,
    object_parameter: &str,
) -> Result<()> {
    let Some(bone) = optional_string(modifier, bone_parameter)? else {
        return Ok(());
    };
    if bone.is_empty() {
        return Ok(());
    }
    let operand = operands
        .get(object_parameter)
        .and_then(|items| items.first())
        .ok_or_else(|| {
            invalid(
                modifier,
                object_parameter,
                "an armature object for the requested bone",
            )
        })?;
    if !operand.armature {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            format!("UV Warp `{bone_parameter}` requires an armature object"),
            json!({"feature_id":"modifier.uv_warp.bone","modifier_id":modifier.id,"parameter":bone_parameter,"bone":bone}),
        ));
    }
    Err(PotError::with_details(
        ErrorCode::UnsupportedFeature,
        "UV Warp bone-relative transforms are not available in the evaluated armature context",
        json!({"feature_id":"modifier.uv_warp.bone_transform","modifier_id":modifier.id,"parameter":bone_parameter,"bone":bone}),
    ))
}

fn data_transfer(
    mesh: &Mesh,
    modifier: &Modifier,
    operands: &BTreeMap<String, Vec<SceneOperand>>,
) -> Result<Mesh> {
    let source = one_operand(operands, "object", modifier)?;
    let use_transform = boolean(modifier, "use_object_transform", true)?;
    let source_transform = if use_transform {
        source.local_to_subject
    } else {
        DMat4::IDENTITY
    };
    let use_vertices = boolean(modifier, "use_vert_data", false)?;
    let use_edges = boolean(modifier, "use_edge_data", false)?;
    let use_loops = boolean(modifier, "use_loop_data", false)?;
    let use_polygons = boolean(modifier, "use_poly_data", false)?;
    let vertex_mapping = string(modifier, "vert_mapping", "NEAREST")?;
    let edge_mapping = string(modifier, "edge_mapping", "NEAREST")?;
    let loop_mapping = string(modifier, "loop_mapping", "NEAREST_POLYNOR")?;
    let polygon_mapping = string(modifier, "poly_mapping", "NEAREST")?;
    validate_transfer_mapping(modifier, "vert_mapping", vertex_mapping)?;
    validate_transfer_mapping(modifier, "edge_mapping", edge_mapping)?;
    validate_transfer_mapping(modifier, "loop_mapping", loop_mapping)?;
    validate_transfer_mapping(modifier, "poly_mapping", polygon_mapping)?;
    let factor = number(modifier, "mix_factor", 1.0)?.clamp(0.0, 1.0);
    let mix_mode = string(modifier, "mix_mode", "REPLACE")?;
    let mask_group = optional_string(modifier, "vertex_group")?.filter(|name| !name.is_empty());
    let mask_weights = mask_group
        .map(|name| get_group_weights(mesh, name))
        .transpose()?;
    let invert_mask = boolean(modifier, "invert_vertex_group", false)?;
    let mut result = mesh.clone();
    let data_types_verts = string_set(modifier, "data_types_verts")?;
    let data_types_edges = string_set(modifier, "data_types_edges")?;
    let data_types_loops = string_set(modifier, "data_types_loops")?;
    let data_types_polys = string_set(modifier, "data_types_polys")?;
    if use_vertices {
        for name in &data_types_verts {
            match name.as_str() {
                "VGROUP_WEIGHTS" => transfer_vertex_groups(
                    &mut result,
                    &source.mesh,
                    modifier,
                    vertex_mapping,
                    factor,
                    mix_mode,
                    use_transform,
                    source.local_to_subject,
                    mask_weights.as_ref(),
                    invert_mask,
                )?,
                "BEVEL_WEIGHT_VERT" => transfer_vertex_attribute(
                    &mut result,
                    &source.mesh,
                    modifier,
                    "bevel_weight_vert",
                    vertex_mapping,
                    source_transform,
                    factor,
                    mix_mode,
                    mask_weights.as_ref(),
                    invert_mask,
                )?,
                "COLOR_VERTEX" => transfer_vertex_attribute(
                    &mut result,
                    &source.mesh,
                    modifier,
                    "color",
                    vertex_mapping,
                    source_transform,
                    factor,
                    mix_mode,
                    mask_weights.as_ref(),
                    invert_mask,
                )?,
                _ => {
                    return Err(invalid(
                        modifier,
                        "data_types_verts",
                        "VGROUP_WEIGHTS, BEVEL_WEIGHT_VERT, or COLOR_VERTEX",
                    ));
                }
            }
        }
    }
    if use_edges {
        for name in &data_types_edges {
            match name.as_str() {
                "SHARP_EDGE" => {
                    transfer_sharp_edges(
                        &mut result,
                        &source.mesh,
                        edge_mapping,
                        source_transform,
                        factor,
                    );
                }
                "SEAM" => transfer_edge_attribute(
                    &mut result,
                    &source.mesh,
                    modifier,
                    "uv_seam",
                    edge_mapping,
                    source_transform,
                    factor,
                    mix_mode,
                )?,
                "CREASE" => transfer_edge_attribute(
                    &mut result,
                    &source.mesh,
                    modifier,
                    "crease_edge",
                    edge_mapping,
                    source_transform,
                    factor,
                    mix_mode,
                )?,
                _ => {
                    return Err(invalid(
                        modifier,
                        "data_types_edges",
                        "SHARP_EDGE, SEAM, or CREASE",
                    ));
                }
            }
        }
    }
    if use_loops {
        for name in &data_types_loops {
            match name.as_str() {
                "UV" => transfer_uv(
                    &mut result,
                    &source.mesh,
                    modifier,
                    loop_mapping,
                    source_transform,
                    factor,
                    mix_mode,
                    mask_weights.as_ref(),
                    invert_mask,
                )?,
                "CUSTOM_NORMAL" => transfer_custom_normals(
                    &mut result,
                    &source.mesh,
                    modifier,
                    loop_mapping,
                    source_transform,
                    factor,
                )?,
                "COLOR_CORNER" => transfer_typed_attribute(
                    &mut result,
                    &source.mesh,
                    modifier,
                    "color",
                    "color",
                    loop_mapping,
                    source_transform,
                    factor,
                    mix_mode,
                )?,
                _ => {
                    return Err(invalid(
                        modifier,
                        "data_types_loops",
                        "CUSTOM_NORMAL, UV, or COLOR_CORNER",
                    ));
                }
            }
        }
    }
    if use_polygons {
        for name in &data_types_polys {
            match name.as_str() {
                "SMOOTH" => transfer_typed_attribute(
                    &mut result,
                    &source.mesh,
                    modifier,
                    "smooth",
                    "smooth",
                    polygon_mapping,
                    source_transform,
                    factor,
                    mix_mode,
                )?,
                "FREESTYLE_FACE" => transfer_typed_attribute(
                    &mut result,
                    &source.mesh,
                    modifier,
                    "freestyle_face",
                    "freestyle_face",
                    polygon_mapping,
                    source_transform,
                    factor,
                    mix_mode,
                )?,
                _ => {
                    return Err(invalid(
                        modifier,
                        "data_types_polys",
                        "SMOOTH or FREESTYLE_FACE",
                    ));
                }
            }
        }
    }
    Ok(result)
}

fn validate_transfer_mapping(modifier: &Modifier, field: &str, value: &str) -> Result<()> {
    let valid = match field {
        "vert_mapping" => matches!(
            value,
            "TOPOLOGY" | "NEAREST" | "POLYINTERP_NEAREST" | "POLYINTERP_VNORPROJ"
        ),
        "edge_mapping" => matches!(
            value,
            "TOPOLOGY" | "NEAREST" | "VERT_NEAREST" | "POLY_NEAREST"
        ),
        "loop_mapping" => matches!(
            value,
            "TOPOLOGY" | "NEAREST_POLYNOR" | "POLYINTERP_NEAREST" | "POLYINTERP_LNORPROJ"
        ),
        "poly_mapping" => matches!(value, "TOPOLOGY" | "NEAREST" | "POLYINTERP_PNORPROJ"),
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(invalid(
            modifier,
            field,
            "a supported Blender data mapping identifier",
        ))
    }
}

fn transfer_vertex_groups(
    destination: &mut Mesh,
    source: &Mesh,
    modifier: &Modifier,
    mapping: &str,
    factor: f64,
    mix_mode: &str,
    use_transform: bool,
    transform: DMat4,
    mask_weights: Option<&BTreeMap<u32, f64>>,
    invert_mask: bool,
) -> Result<()> {
    let source_groups = source
        .attributes
        .get("vertex_groups")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            invalid(
                modifier,
                "data_types_verts",
                "a source mesh with vertex groups",
            )
        })?;
    let mut groups = destination
        .attributes
        .get("vertex_groups")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let source_positions = source
        .vertices
        .iter()
        .map(|vertex| {
            (
                vertex.id,
                if use_transform {
                    transform.transform_point3(vertex.co)
                } else {
                    vertex.co
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    for (name, weights) in source_groups {
        let source_weights = weights.as_object().ok_or_else(|| {
            invalid(
                modifier,
                "data_types_verts",
                "valid source vertex-group weights",
            )
        })?;
        let mut destination_weights = groups
            .get(name)
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        for (index, vertex) in destination.vertices.iter().enumerate() {
            let source_value = if matches!(mapping, "POLYINTERP_NEAREST" | "POLYINTERP_VNORPROJ") {
                interpolated_vertex_value(source, &source_positions, source_weights, vertex.co)
            } else {
                source_vertex_for(
                    index,
                    vertex.co,
                    &source.vertices,
                    &source_positions,
                    mapping,
                )
                .map(|source_vertex| {
                    source_weights
                        .get(&format!("v{source_vertex}"))
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0)
                })
            };
            let Some(source_value) = source_value else {
                continue;
            };
            let key = format!("v{}", vertex.id);
            let prior = destination_weights
                .get(&key)
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            let mut mask = mask_weights.map_or(1.0, |weights| {
                weights.get(&vertex.id).copied().unwrap_or(0.0)
            });
            if invert_mask {
                mask = 1.0 - mask;
            }
            let mixed = blend_attribute(prior, source_value, factor * mask, mix_mode, modifier)?;
            destination_weights.insert(key, json!(mixed.clamp(0.0, 1.0)));
        }
        groups.insert(name.clone(), Value::Object(destination_weights));
    }
    destination
        .attributes
        .insert("vertex_groups".to_owned(), Value::Object(groups));
    Ok(())
}

fn interpolated_vertex_value(
    source: &Mesh,
    source_positions: &BTreeMap<u32, DVec3>,
    source_values: &Map<String, Value>,
    point: DVec3,
) -> Option<f64> {
    let mut closest_distance = f64::INFINITY;
    let mut result = None;
    for face in &source.faces {
        if face.vertices.len() < 3 {
            continue;
        }
        let first_id = face.vertices[0];
        for corner in 1..face.vertices.len() - 1 {
            let vertex_ids = [first_id, face.vertices[corner], face.vertices[corner + 1]];
            let [Some(a), Some(b), Some(c)] =
                vertex_ids.map(|id| source_positions.get(&id).copied())
            else {
                continue;
            };
            let closest = closest_point_triangle(point, a, b, c);
            let distance = closest.distance_squared(point);
            if distance >= closest_distance {
                continue;
            }
            let factors = triangle_barycentric(closest, a, b, c);
            let mut value = 0.0;
            for (id, factor) in vertex_ids.into_iter().zip(factors) {
                value += source_values
                    .get(&format!("v{id}"))
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0)
                    * factor;
            }
            closest_distance = distance;
            result = Some(value);
        }
    }
    result
}

fn triangle_barycentric(point: DVec3, a: DVec3, b: DVec3, c: DVec3) -> [f64; 3] {
    let first = b - a;
    let second = c - a;
    let offset = point - a;
    let first_squared = first.length_squared();
    let second_squared = second.length_squared();
    let cross = first.dot(second);
    let denominator = first_squared * second_squared - cross * cross;
    if denominator.abs() <= f64::EPSILON {
        return [1.0, 0.0, 0.0];
    }
    let first_factor =
        (second_squared * offset.dot(first) - cross * offset.dot(second)) / denominator;
    let second_factor =
        (first_squared * offset.dot(second) - cross * offset.dot(first)) / denominator;
    let weights = [
        (1.0 - first_factor - second_factor).max(0.0),
        first_factor.max(0.0),
        second_factor.max(0.0),
    ];
    let total = weights.iter().sum::<f64>();
    if total > f64::EPSILON {
        [weights[0] / total, weights[1] / total, weights[2] / total]
    } else {
        [1.0, 0.0, 0.0]
    }
}

fn interpolate_face_corner_value(
    source: &Mesh,
    face: &Face,
    values: &[Value],
    source_transform: DMat4,
    point: DVec3,
) -> Option<Value> {
    let first_id = *face.vertices.first()?;
    let first_position = source
        .vertices
        .iter()
        .find(|vertex| vertex.id == first_id)
        .map(|vertex| source_transform.transform_point3(vertex.co))?;
    let mut closest_distance = f64::INFINITY;
    let mut result = None;
    for corner in 1..face.vertices.len().saturating_sub(1) {
        let second_id = face.vertices[corner];
        let third_id = face.vertices[corner + 1];
        let Some(second_position) = source
            .vertices
            .iter()
            .find(|vertex| vertex.id == second_id)
            .map(|vertex| source_transform.transform_point3(vertex.co))
        else {
            continue;
        };
        let Some(third_position) = source
            .vertices
            .iter()
            .find(|vertex| vertex.id == third_id)
            .map(|vertex| source_transform.transform_point3(vertex.co))
        else {
            continue;
        };
        let closest =
            closest_point_triangle(point, first_position, second_position, third_position);
        let distance = closest.distance_squared(point);
        if distance >= closest_distance {
            continue;
        }
        let Some(first_value) = values.first() else {
            continue;
        };
        let Some(second_value) = values.get(corner) else {
            continue;
        };
        let Some(third_value) = values.get(corner + 1) else {
            continue;
        };
        closest_distance = distance;
        result = interpolate_values(
            [first_value, second_value, third_value],
            triangle_barycentric(closest, first_position, second_position, third_position),
        );
    }
    result
}

fn interpolate_values(values: [&Value; 3], factors: [f64; 3]) -> Option<Value> {
    if let [Some(first), Some(second), Some(third)] = values.map(Value::as_f64) {
        return Some(json!(
            first * factors[0] + second * factors[1] + third * factors[2]
        ));
    }
    if let [Some(first), Some(second), Some(third)] = values.map(Value::as_array) {
        if first.len() != second.len() || first.len() != third.len() {
            return None;
        }
        return first
            .iter()
            .zip(second)
            .zip(third)
            .map(|((first, second), third)| interpolate_values([first, second, third], factors))
            .collect::<Option<Vec<_>>>()
            .map(Value::Array);
    }
    if let [Some(first), Some(second), Some(third)] = values.map(Value::as_bool) {
        let index = factors
            .iter()
            .enumerate()
            .max_by(|(_, first), (_, second)| first.total_cmp(second))
            .map_or(0, |(index, _)| index);
        return Some(Value::Bool([first, second, third][index]));
    }
    None
}
fn transfer_vertex_attribute(
    destination: &mut Mesh,
    source: &Mesh,
    modifier: &Modifier,
    name: &str,
    mapping: &str,
    source_transform: DMat4,
    factor: f64,
    mix_mode: &str,
    mask_weights: Option<&BTreeMap<u32, f64>>,
    invert_mask: bool,
) -> Result<()> {
    let Some(source_attribute) = source.attributes.get(name) else {
        return Ok(());
    };
    let source_values = source_attribute
        .get("values")
        .and_then(Value::as_object)
        .or_else(|| source_attribute.as_object())
        .ok_or_else(|| invalid(modifier, name, "a vertex-ID keyed source attribute"))?;
    let destination_attribute = destination.attributes.get(name);
    let mut output = destination_attribute
        .and_then(|attribute| attribute.get("values"))
        .and_then(Value::as_object)
        .or_else(|| destination_attribute.and_then(Value::as_object))
        .cloned()
        .unwrap_or_default();
    let source_positions = source
        .vertices
        .iter()
        .map(|vertex| (vertex.id, source_transform.transform_point3(vertex.co)))
        .collect::<BTreeMap<_, _>>();
    for (index, vertex) in destination.vertices.iter().enumerate() {
        let Some(source_vertex) = source_vertex_for(
            index,
            vertex.co,
            &source.vertices,
            &source_positions,
            mapping,
        ) else {
            continue;
        };
        let Some(source_value) = source_values.get(&format!("v{source_vertex}")) else {
            continue;
        };
        let mut mask = mask_weights.map_or(1.0, |weights| {
            weights.get(&vertex.id).copied().unwrap_or(0.0)
        });
        if invert_mask {
            mask = 1.0 - mask;
        }
        let key = format!("v{}", vertex.id);
        let current = output.get(&key);
        output.insert(
            key,
            mix_json(current, source_value, factor * mask, mix_mode, modifier)?,
        );
    }
    if let Some(source_fields) = source_attribute.as_object()
        && source_fields.contains_key("values")
    {
        let mut fields = destination_attribute
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_else(|| {
                source_fields
                    .iter()
                    .filter(|(field, _)| field.as_str() != "values")
                    .map(|(field, value)| (field.clone(), value.clone()))
                    .collect()
            });
        fields.insert("values".to_owned(), Value::Object(output));
        destination
            .attributes
            .insert(name.to_owned(), Value::Object(fields));
    } else {
        destination
            .attributes
            .insert(name.to_owned(), Value::Object(output));
    }
    Ok(())
}

fn transfer_sharp_edges(
    destination: &mut Mesh,
    source: &Mesh,
    mapping: &str,
    source_transform: DMat4,
    factor: f64,
) {
    let source_sharp = source
        .attributes
        .get("sharp_edges")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_u64)
                .filter_map(|value| u32::try_from(value).ok())
                .collect::<std::collections::BTreeSet<_>>()
        })
        .unwrap_or_default();
    let mut destination_sharp = destination
        .attributes
        .get("sharp_edges")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_u64)
        .filter_map(|value| u32::try_from(value).ok())
        .collect::<std::collections::BTreeSet<_>>();
    if factor < 0.5 {
        return;
    }
    for edge in &destination.edges {
        let source_edge = if mapping == "TOPOLOGY" {
            topology_source_edge(destination, edge, source)
        } else {
            let midpoint = edge_midpoint(destination, edge.vertices);
            source.edges.iter().min_by(|first, second| {
                edge_midpoint_transformed(source, first.vertices, source_transform)
                    .distance_squared(midpoint)
                    .total_cmp(
                        &edge_midpoint_transformed(source, second.vertices, source_transform)
                            .distance_squared(midpoint),
                    )
            })
        };
        if let Some(source_edge) = source_edge {
            if source_sharp.contains(&source_edge.id) {
                destination_sharp.insert(edge.id);
            } else {
                destination_sharp.remove(&edge.id);
            }
        }
    }
    destination.attributes.insert(
        "sharp_edges".to_owned(),
        Value::Array(destination_sharp.into_iter().map(|id| json!(id)).collect()),
    );
}

fn transfer_edge_attribute(
    destination: &mut Mesh,
    source: &Mesh,
    modifier: &Modifier,
    name: &str,
    mapping: &str,
    source_transform: DMat4,
    factor: f64,
    mix_mode: &str,
) -> Result<()> {
    let Some(source_attribute) = source.attributes.get(name) else {
        return Ok(());
    };
    let source_fields = source_attribute.as_object().ok_or_else(|| {
        invalid(
            modifier,
            name,
            "an edge-ID keyed object or typed edge attribute",
        )
    })?;
    let values = source_fields
        .get("values")
        .and_then(Value::as_object)
        .or_else(|| source_attribute.as_object())
        .ok_or_else(|| invalid(modifier, name, "an edge-ID keyed object"))?;
    let destination_attribute = destination.attributes.get(name);
    let mut output = destination_attribute
        .and_then(|attribute| attribute.get("values"))
        .and_then(Value::as_object)
        .or_else(|| destination_attribute.and_then(Value::as_object))
        .cloned()
        .unwrap_or_default();
    for edge in &destination.edges {
        let source_edge = if mapping == "TOPOLOGY" {
            topology_source_edge(destination, edge, source)
        } else {
            let destination_midpoint = edge_midpoint(destination, edge.vertices);
            source.edges.iter().min_by(|first, second| {
                edge_midpoint_transformed(source, first.vertices, source_transform)
                    .distance_squared(destination_midpoint)
                    .total_cmp(
                        &edge_midpoint_transformed(source, second.vertices, source_transform)
                            .distance_squared(destination_midpoint),
                    )
            })
        };
        if let Some(source_edge) = source_edge
            && let Some(value) = values.get(&format!("e{}", source_edge.id))
        {
            let key = format!("e{}", edge.id);
            let current = output.get(&key);
            output.insert(key, mix_json(current, value, factor, mix_mode, modifier)?);
        }
    }
    if source_fields.contains_key("values") {
        let mut fields = destination_attribute
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_else(|| {
                source_fields
                    .iter()
                    .filter(|(field, _)| field.as_str() != "values")
                    .map(|(field, value)| (field.clone(), value.clone()))
                    .collect()
            });
        fields.insert("values".to_owned(), Value::Object(output));
        destination
            .attributes
            .insert(name.to_owned(), Value::Object(fields));
    } else {
        destination
            .attributes
            .insert(name.to_owned(), Value::Object(output));
    }
    Ok(())
}
fn transfer_uv(
    destination: &mut Mesh,
    source: &Mesh,
    modifier: &Modifier,
    mapping: &str,
    source_transform: DMat4,
    factor: f64,
    mix_mode: &str,
    mask_weights: Option<&BTreeMap<u32, f64>>,
    invert_mask: bool,
) -> Result<()> {
    let source_entries = uv_entries(source, modifier)?;
    if source_entries.is_empty() {
        return Ok(());
    }
    let source_selection = string(modifier, "layers_uv_select_src", "ALL")?;
    let destination_selection = string(modifier, "layers_uv_select_dst", "NAME")?;
    if !matches!(source_selection, "ACTIVE" | "ALL") {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "UV Data Transfer bone-based layer selection is not modeled",
            json!({"feature_id":"modifier.data_transfer.uv_layer_selection","modifier_id":modifier.id}),
        ));
    }
    let mut source_layers = source_entries
        .iter()
        .filter_map(|entry| entry.get("layer").and_then(Value::as_str))
        .map(str::to_owned)
        .collect::<std::collections::BTreeSet<_>>();
    if source_layers.is_empty() {
        source_layers.insert("UVMap".to_owned());
    }
    if source_selection == "ACTIVE" {
        source_layers = source_layers.into_iter().take(1).collect();
    }
    let mut destination_entries = uv_entries(destination, modifier)?;
    let destination_layers = destination_entries
        .iter()
        .filter_map(|entry| entry.get("layer").and_then(Value::as_str))
        .map(str::to_owned)
        .collect::<Vec<_>>();
    for source_layer in source_layers {
        let destination_layer = match destination_selection {
            "NAME" => source_layer.clone(),
            "ACTIVE" => destination_layers
                .first()
                .cloned()
                .unwrap_or_else(|| source_layer.clone()),
            "INDEX" => destination_layers
                .first()
                .cloned()
                .unwrap_or_else(|| source_layer.clone()),
            _ => {
                return Err(invalid(
                    modifier,
                    "layers_uv_select_dst",
                    "ACTIVE, NAME, or INDEX",
                ));
            }
        };
        for (face_index, face) in destination.faces.iter().enumerate() {
            let source_face = if mapping == "TOPOLOGY" {
                source.faces.get(face_index)
            } else {
                nearest_face(destination, face, source, source_transform)
            };
            let Some(source_face) = source_face else {
                continue;
            };
            let Some(source_uv) = source_entries.iter().find(|entry| {
                entry.get("face_id").and_then(Value::as_u64) == Some(u64::from(source_face.id))
                    && entry
                        .get("layer")
                        .and_then(Value::as_str)
                        .unwrap_or("UVMap")
                        == source_layer
            }) else {
                continue;
            };
            let Some(uv) = source_uv.get("uv").and_then(Value::as_array) else {
                continue;
            };
            let mapped = face
                .vertices
                .iter()
                .enumerate()
                .map(|(index, vertex_id)| {
                    let old = existing_uv(destination, face.id, &destination_layer, index)
                        .unwrap_or(DVec2::ZERO);
                    let mask = mask_weights
                        .map_or(1.0, |weights| {
                            weights.get(vertex_id).copied().unwrap_or(0.0)
                        })
                        .clamp(0.0, 1.0);
                    let mask = if invert_mask { 1.0 - mask } else { mask };
                    let target = if mapping == "TOPOLOGY" {
                        uv.get(index).and_then(value2)
                    } else {
                        destination
                            .vertices
                            .iter()
                            .find(|vertex| vertex.id == *vertex_id)
                            .and_then(|vertex| {
                                interpolate_face_corner_value(
                                    source,
                                    source_face,
                                    uv,
                                    source_transform,
                                    vertex.co,
                                )
                            })
                            .and_then(|value| value2(&value))
                    };
                    target.map_or(old, |target| {
                        mix_vector(old, target, factor * mask, mix_mode).unwrap_or(target)
                    })
                })
                .map(|point| json!([point.x, point.y]))
                .collect::<Vec<_>>();
            insert_uv(
                &mut destination_entries,
                face.id,
                &destination_layer,
                Value::Array(mapped),
            );
        }
    }
    destination
        .attributes
        .insert("uv_map".to_owned(), Value::Array(destination_entries));
    Ok(())
}

fn transfer_custom_normals(
    destination: &mut Mesh,
    source: &Mesh,
    modifier: &Modifier,
    mapping: &str,
    source_transform: DMat4,
    factor: f64,
) -> Result<()> {
    let source_values = source
        .attributes
        .get("custom_normal")
        .and_then(|value| value.get("values"))
        .and_then(Value::as_object)
        .ok_or_else(|| invalid(modifier, "data_types_loops", "source custom normals"))?;
    let mut values = destination
        .attributes
        .get("custom_normal")
        .and_then(|value| value.get("values"))
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    for (index, face) in destination.faces.iter().enumerate() {
        let source_face = if mapping == "TOPOLOGY" {
            source.faces.get(index)
        } else {
            nearest_face(destination, face, source, source_transform)
        };
        let Some(source_face) = source_face else {
            continue;
        };
        let Some(source_normals) = source_values
            .get(&format!("f{}", source_face.id))
            .and_then(Value::as_array)
        else {
            continue;
        };
        let normals = face
            .vertices
            .iter()
            .enumerate()
            .map(|(corner, vertex_id)| {
                let source_normal = destination
                    .vertices
                    .iter()
                    .find(|vertex| vertex.id == *vertex_id)
                    .and_then(|vertex| {
                        interpolate_face_corner_value(
                            source,
                            source_face,
                            source_normals,
                            source_transform,
                            vertex.co,
                        )
                    })
                    .as_ref()
                    .and_then(value3)
                    .unwrap_or(DVec3::Z);
                let old = values
                    .get(&format!("f{}", face.id))
                    .and_then(Value::as_array)
                    .and_then(|normals| normals.get(corner))
                    .and_then(value3)
                    .unwrap_or(DVec3::Z);
                json!(
                    old.lerp(source_normal, factor)
                        .normalize_or_zero()
                        .to_array()
                )
            })
            .collect::<Vec<_>>();
        values.insert(format!("f{}", face.id), Value::Array(normals));
    }
    destination.attributes.insert(
        "custom_normal".to_owned(),
        json!({"domain":"corner","type":"float3","values":values}),
    );
    Ok(())
}

fn transfer_typed_attribute(
    destination: &mut Mesh,
    source: &Mesh,
    modifier: &Modifier,
    source_name: &str,
    destination_name: &str,
    mapping: &str,
    source_transform: DMat4,
    factor: f64,
    mix_mode: &str,
) -> Result<()> {
    let Some(source_value) = source.attributes.get(source_name) else {
        return Ok(());
    };
    let Some(source_values) = source_value.get("values").and_then(Value::as_object) else {
        return Ok(());
    };
    let mut output = destination
        .attributes
        .get(destination_name)
        .and_then(|value| value.get("values"))
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let domain = source_value
        .get("domain")
        .and_then(Value::as_str)
        .unwrap_or("face");
    for (index, face) in destination.faces.iter().enumerate() {
        let source_face = if mapping == "TOPOLOGY" {
            source.faces.get(index)
        } else {
            nearest_face(destination, face, source, source_transform)
        };
        let Some(source_face) = source_face else {
            continue;
        };
        let Some(value) = source_values.get(&format!("f{}", source_face.id)) else {
            continue;
        };
        let key = format!("f{}", face.id);
        if domain == "corner" {
            let Some(source_corners) = value.as_array() else {
                continue;
            };
            let mut destination_corners = Vec::with_capacity(face.vertices.len());
            for (corner, vertex_id) in face.vertices.iter().enumerate() {
                let current = output
                    .get(&key)
                    .and_then(Value::as_array)
                    .and_then(|corners| corners.get(corner));
                let target = destination
                    .vertices
                    .iter()
                    .find(|vertex| vertex.id == *vertex_id)
                    .and_then(|vertex| {
                        interpolate_face_corner_value(
                            source,
                            source_face,
                            source_corners,
                            source_transform,
                            vertex.co,
                        )
                    });
                let Some(target) = target else {
                    continue;
                };
                destination_corners.push(mix_json(current, &target, factor, mix_mode, modifier)?);
            }
            output.insert(key, Value::Array(destination_corners));
        } else {
            let current = output.get(&key);
            output.insert(key, mix_json(current, value, factor, mix_mode, modifier)?);
        }
    }
    let kind = source_value
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("float");
    destination.attributes.insert(
        destination_name.to_owned(),
        json!({"domain":domain,"type":kind,"values":output}),
    );
    Ok(())
}

fn source_vertex_for(
    destination_index: usize,
    point: DVec3,
    source_vertices: &[Vertex],
    source_positions: &BTreeMap<u32, DVec3>,
    mapping: &str,
) -> Option<u32> {
    if mapping == "TOPOLOGY" {
        return source_vertices
            .get(destination_index)
            .map(|vertex| vertex.id);
    }
    source_vertices
        .iter()
        .min_by(|first, second| {
            source_positions
                .get(&first.id)
                .copied()
                .unwrap_or(first.co)
                .distance_squared(point)
                .total_cmp(
                    &source_positions
                        .get(&second.id)
                        .copied()
                        .unwrap_or(second.co)
                        .distance_squared(point),
                )
        })
        .map(|vertex| vertex.id)
}

fn nearest_face<'a>(
    destination: &Mesh,
    face: &Face,
    source: &'a Mesh,
    source_transform: DMat4,
) -> Option<&'a Face> {
    let center = face_center(destination, face);
    source.faces.iter().min_by(|first, second| {
        source_transform
            .transform_point3(face_center(source, first))
            .distance_squared(center)
            .total_cmp(
                &source_transform
                    .transform_point3(face_center(source, second))
                    .distance_squared(center),
            )
    })
}
fn face_center(mesh: &Mesh, face: &Face) -> DVec3 {
    let mut center = DVec3::ZERO;
    let mut count = 0_u32;
    for id in &face.vertices {
        if let Some(vertex) = mesh.vertices.iter().find(|vertex| vertex.id == *id) {
            center += vertex.co;
            count += 1;
        }
    }
    if count > 0 {
        center / f64::from(count)
    } else {
        DVec3::ZERO
    }
}

fn edge_midpoint(mesh: &Mesh, vertices: [u32; 2]) -> DVec3 {
    let a = mesh
        .vertices
        .iter()
        .find(|vertex| vertex.id == vertices[0])
        .map_or(DVec3::ZERO, |vertex| vertex.co);
    let b = mesh
        .vertices
        .iter()
        .find(|vertex| vertex.id == vertices[1])
        .map_or(DVec3::ZERO, |vertex| vertex.co);
    (a + b) * 0.5
}
fn edge_midpoint_transformed(mesh: &Mesh, vertices: [u32; 2], transform: DMat4) -> DVec3 {
    transform.transform_point3(edge_midpoint(mesh, vertices))
}
fn topology_source_edge<'a>(destination: &Mesh, edge: &Edge, source: &'a Mesh) -> Option<&'a Edge> {
    let destination_pair = edge_vertex_indices(destination, edge)?;
    source
        .edges
        .iter()
        .find(|candidate| edge_vertex_indices(source, candidate) == Some(destination_pair))
}

fn edge_vertex_indices(mesh: &Mesh, edge: &Edge) -> Option<[usize; 2]> {
    let indices = edge.vertices.map(|vertex_id| {
        mesh.vertices
            .iter()
            .position(|vertex| vertex.id == vertex_id)
    });
    let [Some(first), Some(second)] = indices else {
        return None;
    };
    if first <= second {
        Some([first, second])
    } else {
        Some([second, first])
    }
}

fn blend_attribute(a: f64, b: f64, factor: f64, mode: &str, modifier: &Modifier) -> Result<f64> {
    let target = match mode {
        "REPLACE" | "MIX" => b,
        "ADD" => a + b,
        "SUB" => a - b,
        "MUL" => a * b,
        "ABOVE_THRESHOLD" => {
            if b > a {
                b
            } else {
                a
            }
        }
        "BELOW_THRESHOLD" => {
            if b < a {
                b
            } else {
                a
            }
        }
        _ => {
            return Err(invalid(
                modifier,
                "mix_mode",
                "REPLACE, ADD, SUB, MUL, ABOVE_THRESHOLD, or BELOW_THRESHOLD",
            ));
        }
    };
    Ok(a + (target - a) * factor)
}

fn mix_json(
    current: Option<&Value>,
    source: &Value,
    factor: f64,
    mode: &str,
    modifier: &Modifier,
) -> Result<Value> {
    if let Some(source_number) = source.as_f64() {
        let current_number = current.and_then(Value::as_f64).unwrap_or(0.0);
        return Ok(json!(blend_attribute(
            current_number,
            source_number,
            factor,
            mode,
            modifier,
        )?));
    }
    if let Some(source_values) = source.as_array()
        && let Some(current_values) = current.and_then(Value::as_array)
        && current_values.len() == source_values.len()
    {
        return source_values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                mix_json(current_values.get(index), value, factor, mode, modifier)
            })
            .collect::<Result<Vec<_>>>()
            .map(Value::Array);
    }
    if source.is_boolean() {
        return if factor >= 0.5 || current.is_none() {
            Ok(source.clone())
        } else {
            Ok(current.cloned().unwrap_or_else(|| source.clone()))
        };
    }
    if factor <= 0.0 {
        Ok(current.cloned().unwrap_or_else(|| source.clone()))
    } else {
        Ok(source.clone())
    }
}

fn mix_vector(first: DVec2, second: DVec2, factor: f64, mode: &str) -> Option<DVec2> {
    match mode {
        "REPLACE" => Some(first.lerp(second, factor)),
        "ADD" => Some(first + second * factor),
        "SUB" => Some(first - second * factor),
        "MUL" => Some(first * second.lerp(DVec2::ONE, 1.0 - factor)),
        _ => None,
    }
}

fn uv_entries(mesh: &Mesh, modifier: &Modifier) -> Result<Vec<Value>> {
    if let Some(uv_map) = mesh.attributes.get("uv_map") {
        return match uv_map {
            Value::Array(entries) => Ok(entries.clone()),
            _ => Err(invalid(
                modifier,
                "uv_layer",
                "an array of face-keyed UV entries",
            )),
        };
    }
    let Some(layers) = mesh
        .attributes
        .get("blender_uv_layers")
        .and_then(Value::as_array)
    else {
        return Ok(Vec::new());
    };
    let mut entries = Vec::new();
    let mut loop_start = 0_usize;
    for face in &mesh.faces {
        let loop_end = loop_start
            .checked_add(face.vertices.len())
            .ok_or_else(|| invalid(modifier, "uv_layer", "a valid UV loop count"))?;
        for layer in layers {
            let name = layer
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid(modifier, "uv_layer", "named Blender UV layers"))?;
            let values = layer
                .get("values")
                .and_then(Value::as_array)
                .ok_or_else(|| invalid(modifier, "uv_layer", "Blender UV layer loop values"))?;
            let corners = values
                .get(loop_start..loop_end)
                .ok_or_else(|| invalid(modifier, "uv_layer", "one UV value per mesh loop"))?;
            entries.push(json!({"face_id":face.id,"layer":name,"uv":corners}));
        }
        loop_start = loop_end;
    }
    Ok(entries)
}

fn insert_uv(entries: &mut Vec<Value>, face_id: u32, layer: &str, uv: Value) {
    if let Some(entry) = entries.iter_mut().find(|entry| {
        entry.get("face_id").and_then(Value::as_u64) == Some(u64::from(face_id))
            && entry
                .get("layer")
                .and_then(Value::as_str)
                .unwrap_or("UVMap")
                == layer
    }) {
        if let Some(object) = entry.as_object_mut() {
            object.insert("uv".to_owned(), uv);
        }
    } else {
        entries.push(json!({"face_id":face_id,"layer":layer,"uv":uv}));
    }
}

fn existing_uv(mesh: &Mesh, face_id: u32, layer: &str, corner: usize) -> Option<DVec2> {
    if let Some(entries) = mesh.attributes.get("uv_map").and_then(Value::as_array)
        && let Some(value) = entries
            .iter()
            .find(|entry| {
                entry.get("face_id").and_then(Value::as_u64) == Some(u64::from(face_id))
                    && entry
                        .get("layer")
                        .and_then(Value::as_str)
                        .unwrap_or("UVMap")
                        == layer
            })
            .and_then(|entry| entry.get("uv"))
            .and_then(Value::as_array)
            .and_then(|values| values.get(corner))
            .and_then(value2)
    {
        return Some(value);
    }
    let face_index = mesh.faces.iter().position(|face| face.id == face_id)?;
    let loop_start = mesh.faces[..face_index]
        .iter()
        .try_fold(0_usize, |total, face| {
            total.checked_add(face.vertices.len())
        })?;
    let loop_index = loop_start.checked_add(corner)?;
    mesh.attributes
        .get("blender_uv_layers")?
        .as_array()?
        .iter()
        .find(|entry| entry.get("name").and_then(Value::as_str) == Some(layer))?
        .get("values")?
        .as_array()?
        .get(loop_index)
        .and_then(value2)
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

fn normalize_group(mesh: &mut Mesh, group: &str) -> Result<()> {
    let weights = get_group_weights(mesh, group)?;
    let maximum = weights.values().copied().fold(0.0_f64, f64::max);
    if maximum > f64::EPSILON {
        let normalized = weights
            .into_iter()
            .map(|(id, value)| (id, value / maximum))
            .collect::<BTreeMap<_, _>>();
        write_group_weights(mesh, group, &normalized)?;
    }
    Ok(())
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

fn get_group_weights_if_present(mesh: &Mesh, group: &str) -> Result<BTreeMap<u32, f64>> {
    if mesh
        .attributes
        .get("vertex_groups")
        .and_then(Value::as_object)
        .is_some_and(|groups| groups.contains_key(group))
    {
        get_group_weights(mesh, group)
    } else {
        Ok(BTreeMap::new())
    }
}

fn write_group_weights(mesh: &mut Mesh, group: &str, weights: &BTreeMap<u32, f64>) -> Result<()> {
    let groups = mesh
        .attributes
        .entry("vertex_groups".to_owned())
        .or_insert_with(|| Value::Object(Map::new()));
    let groups = groups.as_object_mut().ok_or_else(|| {
        PotError::invalid_argument("vertex_groups mesh attribute must be an object")
    })?;
    let mut values = Map::new();
    for (id, weight) in weights {
        if *weight > 0.0 {
            values.insert(format!("v{id}"), json!(weight));
        }
    }
    groups.insert(group.to_owned(), Value::Object(values));
    Ok(())
}

fn falloff_type(modifier: &Modifier) -> Result<&str> {
    let falloff = string(modifier, "falloff_type", "LINEAR")?;
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

fn apply_falloff(value: f64, falloff: &str, curve: Option<&[DVec2]>, seed: u64) -> f64 {
    let value = value.clamp(0.0, 1.0);
    match falloff {
        "CURVE" => curve.map_or(value, |points| {
            let upper = points
                .partition_point(|point| point.x < value)
                .clamp(1, points.len() - 1);
            let first = points[upper - 1];
            let second = points[upper];
            first.y + (second.y - first.y) * ((value - first.x) / (second.x - first.x))
        }),
        "SHARP" => value * value,
        "SMOOTH" => value * value * (3.0 - 2.0 * value),
        "ROOT" => value.sqrt(),
        "ICON_SPHERECURVE" => (1.0 - (1.0 - value).powi(2)).sqrt(),
        "RANDOM" => stable_random(seed),
        "STEP" => f64::from(value >= 0.5),
        _ => value,
    }
}

fn stable_random(seed: u64) -> f64 {
    let mut value = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    ((value ^ (value >> 31)) >> 11) as f64 / ((1_u64 << 53) as f64)
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

fn number(modifier: &Modifier, name: &str, default: f64) -> Result<f64> {
    match modifier.params.get(name) {
        None => Ok(default),
        Some(value) => value
            .as_f64()
            .filter(|number| number.is_finite())
            .ok_or_else(|| invalid(modifier, name, "a finite number")),
    }
}

fn vector2(modifier: &Modifier, name: &str, default: DVec2) -> Result<DVec2> {
    match modifier.params.get(name) {
        None => Ok(default),
        Some(value) => value2(value)
            .filter(|point| point.is_finite())
            .ok_or_else(|| invalid(modifier, name, "two finite numbers")),
    }
}

fn vector3(modifier: &Modifier, name: &str, default: DVec3) -> Result<DVec3> {
    match modifier.params.get(name) {
        None => Ok(default),
        Some(value) => value3(value)
            .filter(|vector| vector.is_finite())
            .ok_or_else(|| invalid(modifier, name, "three finite numbers")),
    }
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
