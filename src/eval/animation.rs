use std::collections::BTreeMap;

use glam::{DQuat, EulerRot};
use serde_json::json;

use crate::{
    error::{ErrorCode, PotError, Result},
    model::{
        Extrapolation, FCurve, Id, Interpolation, NlaBlendType, NlaExtrapolation, NlaStrip, Node,
        PoseBone, SceneDoc, Transform, normalize_rotation,
    },
};

/// Blender F-Curve indices are `(w, x, y, z)`; `Transform::rotation` is `(x, y, z, w)`.
const BLENDER_QUATERNION_COMPONENTS: [usize; 4] = [3, 0, 1, 2];

/// Evaluate NLA tracks bottom-to-top, then the node's active action.
///
/// # Errors
///
/// Returns scene/evaluation errors for invalid action data, NLA strips or unsupported curve paths.
pub fn animated_transform(node: &Node, doc: &SceneDoc, frame: f64) -> Result<Transform> {
    if !frame.is_finite() {
        return Err(PotError::new(
            ErrorCode::InvalidArgument,
            "frame must be finite",
        ));
    }
    let has_solo = node
        .nla_tracks
        .iter()
        .any(|track| track.solo && !track.mute);
    let mut transform = node.transform.clone();
    for track in &node.nla_tracks {
        if track.mute || (has_solo && !track.solo) {
            continue;
        }
        for strip in &track.strips {
            let Some(action_frame) = strip_action_frame(strip, frame)? else {
                continue;
            };
            let action = doc.actions.get(&strip.action).ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "NLA strip action reference does not exist",
                    json!({ "action": strip.action, "strip": strip.id }),
                )
            })?;
            let (sampled, channels) =
                sample_action_transform(&transform, action, &strip.action, action_frame)?;
            blend_transform(
                &mut transform,
                &sampled,
                channels,
                strip.blend_type,
                strip_influence(strip, frame)?,
            )?;
        }
    }
    if let Some(action_id) = &node.action {
        let action = doc.actions.get(action_id).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "node action reference does not exist",
                json!({ "action": action_id }),
            )
        })?;
        let (sampled, channels) = sample_action_transform(&transform, action, action_id, frame)?;
        blend_transform(
            &mut transform,
            &sampled,
            channels,
            NlaBlendType::Replace,
            1.0,
        )?;
    }
    Ok(transform)
}

#[derive(Clone, Copy, Default)]
struct TransformChannels {
    translation: [bool; 3],
    scale: [bool; 3],
    rotation: bool,
}

fn sample_action_transform(
    base: &Transform,
    action: &crate::model::Action,
    action_id: &Id,
    frame: f64,
) -> Result<(Transform, TransformChannels)> {
    let mut transform = base.clone();
    let mut channels = TransformChannels::default();
    let mut euler = None;
    let mut quaternion = transform.rotation;
    let mut quaternion_modified = false;
    for curve in &action.fcurves {
        let Some(value) = sample_curve(curve, frame)? else {
            continue;
        };
        match curve.path.as_str() {
            "transform.translation" => {
                let component = component_index(curve, 3)?;
                transform.translation[component] = value;
                channels.translation[component] = true;
            }
            "transform.scale" => {
                let component = component_index(curve, 3)?;
                transform.scale[component] = value;
                channels.scale[component] = true;
            }
            "transform.rotation_quaternion" | "transform.rotation" => {
                let component = component_index(curve, 4)?;
                quaternion[BLENDER_QUATERNION_COMPONENTS[component]] = value;
                quaternion_modified = true;
                channels.rotation = true;
            }
            "transform.rotation_euler" => {
                let component = component_index(curve, 3)?;
                if euler.is_none() {
                    let normalized = normalize_rotation(transform.rotation).map_err(|error| {
                        PotError::with_details(
                            ErrorCode::SceneInvalid,
                            error.message,
                            json!({ "action": action_id }),
                        )
                    })?;
                    let rotation = DQuat::from_xyzw(
                        normalized[0],
                        normalized[1],
                        normalized[2],
                        normalized[3],
                    );
                    let (angle_x, angle_y, angle_z) = rotation.to_euler(EulerRot::XYZ);
                    euler = Some([angle_x, angle_y, angle_z]);
                }
                let angles = euler.as_mut().ok_or_else(|| {
                    PotError::new(
                        ErrorCode::InternalError,
                        "Euler channel initialization failed",
                    )
                })?;
                angles[component] = value;
                channels.rotation = true;
            }
            path => {
                if let Some(component_count) = non_transform_component_count(path) {
                    component_index(curve, component_count)?;
                } else if pose_curve_path(path).is_none() {
                    let feature_id = format!("animation.fcurve.{path}");
                    return Err(PotError::with_details(
                        ErrorCode::UnsupportedFeature,
                        format!("animation curve path `{path}` is not supported"),
                        json!({ "feature_id": feature_id, "action": action_id }),
                    ));
                }
            }
        }
    }
    if let Some([angle_x, angle_y, angle_z]) = euler {
        let rotation = DQuat::from_euler(EulerRot::XYZ, angle_x, angle_y, angle_z);
        transform.rotation = [rotation.x, rotation.y, rotation.z, rotation.w];
    }
    if quaternion_modified {
        transform.rotation = quaternion;
    }
    if euler.is_some() || quaternion_modified {
        transform.rotation = normalize_rotation(transform.rotation).map_err(|error| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                error.message,
                json!({ "action": action_id }),
            )
        })?;
    }
    Ok((transform, channels))
}

fn strip_action_frame(strip: &NlaStrip, frame: f64) -> Result<Option<f64>> {
    let values = [
        strip.frame_start,
        strip.frame_end,
        strip.action_frame_start,
        strip.action_frame_end,
        strip.scale,
        strip.repeat,
        strip.influence,
        strip.blend_in,
        strip.blend_out,
    ];
    let timeline_span = strip.frame_end - strip.frame_start;
    let action_span = strip.action_frame_end - strip.action_frame_start;
    if values.iter().any(|value| !value.is_finite())
        || timeline_span <= 0.0
        || action_span <= 0.0
        || strip.scale <= 0.0
        || strip.repeat <= 0.0
        || !(0.0..=1.0).contains(&strip.influence)
        || strip.blend_in < 0.0
        || strip.blend_out < 0.0
        || strip.blend_in > timeline_span
        || strip.blend_out > timeline_span
    {
        return Err(PotError::with_details(
            ErrorCode::SceneInvalid,
            "NLA strip settings are invalid",
            json!({ "strip": strip.id }),
        ));
    }
    if frame < strip.frame_start {
        return Ok(match strip.extrapolation {
            NlaExtrapolation::Hold => Some(strip.action_frame_start),
            NlaExtrapolation::HoldForward | NlaExtrapolation::Nothing => None,
        });
    }
    if frame > strip.frame_end {
        return Ok(match strip.extrapolation {
            NlaExtrapolation::Hold | NlaExtrapolation::HoldForward => Some(strip.action_frame_end),
            NlaExtrapolation::Nothing => None,
        });
    }
    let elapsed = (frame - strip.frame_start) / strip.scale;
    let repeated_span = action_span * strip.repeat;
    if !elapsed.is_finite() || !repeated_span.is_finite() {
        return Err(PotError::with_details(
            ErrorCode::SceneInvalid,
            "NLA strip frame mapping overflows",
            json!({ "strip": strip.id }),
        ));
    }
    let offset = if elapsed >= repeated_span {
        repeated_span
    } else {
        elapsed.rem_euclid(action_span)
    };
    let action_frame = strip.action_frame_start + offset;
    if !action_frame.is_finite() {
        return Err(PotError::with_details(
            ErrorCode::SceneInvalid,
            "NLA strip maps to a non-finite action frame",
            json!({ "strip": strip.id }),
        ));
    }
    Ok(Some(action_frame))
}

fn strip_influence(strip: &NlaStrip, frame: f64) -> Result<f64> {
    let mut fade: f64 = 1.0;
    if strip.blend_in > 0.0 && frame < strip.frame_start + strip.blend_in {
        fade = fade.min((frame - strip.frame_start) / strip.blend_in);
    }
    if strip.blend_out > 0.0 && frame > strip.frame_end - strip.blend_out {
        fade = fade.min((strip.frame_end - frame) / strip.blend_out);
    }
    let influence = strip.influence * fade.clamp(0.0, 1.0);
    if influence.is_finite() {
        Ok(influence)
    } else {
        Err(PotError::with_details(
            ErrorCode::SceneInvalid,
            "NLA strip influence is not finite",
            json!({ "strip": strip.id }),
        ))
    }
}

fn blend_transform(
    current: &mut Transform,
    sampled: &Transform,
    channels: TransformChannels,
    blend: NlaBlendType,
    influence: f64,
) -> Result<()> {
    for component in 0..3 {
        if channels.translation[component] {
            let value = sampled.translation[component];
            current.translation[component] = match blend {
                NlaBlendType::Replace => {
                    current.translation[component] * (1.0 - influence) + value * influence
                }
                NlaBlendType::Add | NlaBlendType::Combine => {
                    current.translation[component] + value * influence
                }
            };
        }
        if channels.scale[component] {
            let value = sampled.scale[component];
            current.scale[component] = match blend {
                NlaBlendType::Replace => {
                    current.scale[component] * (1.0 - influence) + value * influence
                }
                NlaBlendType::Add => current.scale[component] + (value - 1.0) * influence,
                NlaBlendType::Combine => {
                    current.scale[component] * (1.0 + (value - 1.0) * influence)
                }
            };
        }
    }
    if channels.rotation {
        let base = normalize_rotation(current.rotation)
            .map_err(|error| PotError::new(ErrorCode::SceneInvalid, error.message))?;
        let target = normalize_rotation(sampled.rotation)
            .map_err(|error| PotError::new(ErrorCode::SceneInvalid, error.message))?;
        let base = DQuat::from_xyzw(base[0], base[1], base[2], base[3]);
        let target = DQuat::from_xyzw(target[0], target[1], target[2], target[3]);
        let result = if blend == NlaBlendType::Replace {
            base.slerp(target, influence)
        } else {
            let delta = base.inverse() * target;
            base * DQuat::IDENTITY.slerp(delta, influence)
        };
        current.rotation = normalize_rotation([result.x, result.y, result.z, result.w])
            .map_err(|error| PotError::new(ErrorCode::SceneInvalid, error.message))?;
    }
    Ok(())
}
/// Evaluate animated pose-bone transforms at a frame.
///
/// NLA tracks are evaluated bottom-to-top; the active action is applied last.
pub fn animated_pose_bones(
    node: &Node,
    doc: &SceneDoc,
    frame: f64,
) -> Result<BTreeMap<Id, PoseBone>> {
    if !frame.is_finite() {
        return Err(PotError::new(
            ErrorCode::InvalidArgument,
            "frame must be finite",
        ));
    }
    let mut pose = node.pose.clone();
    let has_solo = node
        .nla_tracks
        .iter()
        .any(|track| track.solo && !track.mute);
    for track in &node.nla_tracks {
        if track.mute || (has_solo && !track.solo) {
            continue;
        }
        for strip in &track.strips {
            let Some(action_frame) = strip_action_frame(strip, frame)? else {
                continue;
            };
            let action = doc.actions.get(&strip.action).ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "NLA strip action reference does not exist",
                    json!({ "action": strip.action, "strip": strip.id }),
                )
            })?;
            let (sampled, channels) =
                sample_action_pose(&pose, action, &strip.action, action_frame)?;
            blend_pose(
                &mut pose,
                &sampled,
                channels,
                strip.blend_type,
                strip_influence(strip, frame)?,
            )?;
        }
    }
    if let Some(action_id) = &node.action {
        let action = doc.actions.get(action_id).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "node action reference does not exist",
                json!({ "action": action_id }),
            )
        })?;
        let (sampled, channels) = sample_action_pose(&pose, action, action_id, frame)?;
        blend_pose(&mut pose, &sampled, channels, NlaBlendType::Replace, 1.0)?;
    }
    Ok(pose)
}

#[derive(Clone, Copy)]
enum PoseChannel {
    Translation,
    Rotation,
    RotationEuler,
    Scale,
}

#[derive(Clone, Copy, Default)]
struct PoseChannels {
    translation: [bool; 3],
    rotation: [bool; 4],
    scale: [bool; 3],
}

fn pose_curve_path(path: &str) -> Option<(Id, PoseChannel)> {
    let (bone_key, channel) = if let Some(start) = path.find("pose.bones[") {
        let start = start + "pose.bones[".len();
        let end = path[start..].find(']')? + start;
        (
            path[start..end].trim_matches(['"', '\'']),
            path[end + 1..].trim_start_matches('.'),
        )
    } else {
        let rest = path.strip_prefix("pose.")?;
        rest.split_once('.')?
    };
    let channel = match channel {
        "location" | "translation" => PoseChannel::Translation,
        "rotation" => PoseChannel::Rotation,
        "rotation_euler" => PoseChannel::RotationEuler,
        "scale" => PoseChannel::Scale,
        _ => return None,
    };
    Some((Id::new(bone_key).ok()?, channel))
}

fn sample_action_pose(
    base: &BTreeMap<Id, PoseBone>,
    action: &crate::model::Action,
    action_id: &Id,
    frame: f64,
) -> Result<(BTreeMap<Id, PoseBone>, BTreeMap<Id, PoseChannels>)> {
    let mut pose = base.clone();
    let mut channels = BTreeMap::<Id, PoseChannels>::new();
    let mut euler_rotations = BTreeMap::<Id, [f64; 3]>::new();
    for curve in &action.fcurves {
        let Some((bone_id, channel)) = pose_curve_path(&curve.path) else {
            continue;
        };
        let component_count = match channel {
            PoseChannel::Translation | PoseChannel::Scale | PoseChannel::RotationEuler => 3,
            PoseChannel::Rotation => 4,
        };
        let component = component_index(curve, component_count)?;
        let Some(value) = sample_curve(curve, frame)? else {
            continue;
        };
        let pose_bone = pose.entry(bone_id.clone()).or_default();
        let mask = channels.entry(bone_id.clone()).or_default();
        match channel {
            PoseChannel::Translation => {
                pose_bone.translation[component] = value;
                mask.translation[component] = true;
            }
            PoseChannel::Rotation => {
                pose_bone.rotation[component] = value;
                mask.rotation[component] = true;
            }
            PoseChannel::RotationEuler => {
                let angles = euler_rotations.entry(bone_id).or_insert_with(|| {
                    let rotation = DQuat::from_xyzw(
                        pose_bone.rotation[0],
                        pose_bone.rotation[1],
                        pose_bone.rotation[2],
                        pose_bone.rotation[3],
                    );
                    let (x, y, z) = rotation.to_euler(EulerRot::XYZ);
                    [x, y, z]
                });
                angles[component] = value;
            }
            PoseChannel::Scale => {
                pose_bone.scale[component] = value;
                mask.scale[component] = true;
            }
        }
    }
    for (bone_id, angles) in euler_rotations {
        let rotation = DQuat::from_euler(EulerRot::XYZ, angles[0], angles[1], angles[2]);
        pose.get_mut(&bone_id)
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "sampled pose bone is missing"))?
            .rotation = [rotation.x, rotation.y, rotation.z, rotation.w];
        channels.entry(bone_id).or_default().rotation = [true; 4];
    }
    for (bone_id, mask) in &channels {
        if mask.rotation.iter().any(|animated| *animated) {
            let pose_bone = pose.get_mut(bone_id).ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "sampled pose bone is missing")
            })?;
            pose_bone.rotation = normalize_rotation(pose_bone.rotation).map_err(|error| {
                PotError::with_details(
                    ErrorCode::SceneInvalid,
                    error.message,
                    json!({ "action": action_id, "bone": bone_id }),
                )
            })?;
        }
    }
    Ok((pose, channels))
}

fn blend_pose(
    pose: &mut BTreeMap<Id, PoseBone>,
    sampled: &BTreeMap<Id, PoseBone>,
    channels: BTreeMap<Id, PoseChannels>,
    blend: NlaBlendType,
    influence: f64,
) -> Result<()> {
    for (bone_id, mask) in channels {
        let source = sampled.get(&bone_id).copied().ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "sampled pose bone is missing")
        })?;
        let destination = pose.entry(bone_id).or_default();
        for component in 0..3 {
            if mask.translation[component] {
                let value = source.translation[component];
                destination.translation[component] = match blend {
                    NlaBlendType::Replace => {
                        destination.translation[component] * (1.0 - influence) + value * influence
                    }
                    NlaBlendType::Add | NlaBlendType::Combine => {
                        destination.translation[component] + value * influence
                    }
                };
            }
            if mask.scale[component] {
                let value = source.scale[component];
                destination.scale[component] = match blend {
                    NlaBlendType::Replace => {
                        destination.scale[component] * (1.0 - influence) + value * influence
                    }
                    NlaBlendType::Add => destination.scale[component] + (value - 1.0) * influence,
                    NlaBlendType::Combine => {
                        destination.scale[component] * (1.0 + (value - 1.0) * influence)
                    }
                };
            }
        }
        if mask.rotation.iter().any(|animated| *animated) {
            destination.rotation =
                blend_rotation(destination.rotation, source.rotation, blend, influence)?;
        }
    }
    Ok(())
}

fn blend_rotation(
    base_rotation: [f64; 4],
    sampled_rotation: [f64; 4],
    blend: NlaBlendType,
    influence: f64,
) -> Result<[f64; 4]> {
    let base = normalize_rotation(base_rotation)
        .map_err(|error| PotError::new(ErrorCode::SceneInvalid, error.message))?;
    let target = normalize_rotation(sampled_rotation)
        .map_err(|error| PotError::new(ErrorCode::SceneInvalid, error.message))?;
    let base = DQuat::from_xyzw(base[0], base[1], base[2], base[3]);
    let target = DQuat::from_xyzw(target[0], target[1], target[2], target[3]);
    let result = if blend == NlaBlendType::Replace {
        base.slerp(target, influence)
    } else {
        let delta = base.inverse() * target;
        base * DQuat::IDENTITY.slerp(delta, influence)
    };
    normalize_rotation([result.x, result.y, result.z, result.w])
        .map_err(|error| PotError::new(ErrorCode::SceneInvalid, error.message))
}

fn non_transform_component_count(path: &str) -> Option<usize> {
    match path {
        "visible"
        | "camera.lens_mm"
        | "light.energy"
        | "shape_keys.evaluation_time"
        | "shape_key.evaluation_time"
        | "key_blocks.eval_time" => Some(1),
        path if (path.starts_with("shape_key.")
            || path.starts_with("shape_keys.")
            || path.starts_with("key_blocks["))
            && path.strip_suffix(".value").is_some() =>
        {
            Some(1)
        }
        _ => path
            .strip_prefix("material.")
            .and_then(|rest| rest.strip_suffix(".base_color"))
            .filter(|id| crate::model::is_valid_id(id))
            .map(|_| 4),
    }
}

fn component_index(curve: &FCurve, count: usize) -> Result<usize> {
    let index =
        usize::try_from(curve.index).map_err(|_| invalid_curve(curve, "curve index is invalid"))?;
    if index >= count {
        return Err(invalid_curve(
            curve,
            "curve index is outside the property component range",
        ));
    }
    Ok(index)
}

/// Sample one scalar curve. Empty curves have no animated value.
pub(crate) fn sample_curve(curve: &FCurve, frame: f64) -> Result<Option<f64>> {
    if curve.keyframes.is_empty() {
        return Ok(None);
    }
    let keys = &curve.keyframes;
    if keys.iter().any(|key| {
        !key.frame.is_finite()
            || !key.value.is_finite()
            || key
                .handle_left
                .is_some_and(|handle| !handle[0].is_finite() || !handle[1].is_finite())
            || key
                .handle_right
                .is_some_and(|handle| !handle[0].is_finite() || !handle[1].is_finite())
    }) || keys.windows(2).any(|pair| {
        pair[0].frame >= pair[1].frame
            || !(pair[1].frame - pair[0].frame).is_finite()
            || !(pair[1].value - pair[0].value).is_finite()
    }) {
        return Err(invalid_curve(
            curve,
            "keyframes and Bezier handles must be finite, strictly frame-ordered, and numerically bounded",
        ));
    }
    let first = &keys[0];
    if frame < first.frame {
        return match curve.extrapolation {
            Extrapolation::Linear if keys.len() > 1 => {
                let slope = endpoint_tangent(keys, 0);
                finite_sample(curve, first.value + slope * (frame - first.frame))
            }
            Extrapolation::Constant | Extrapolation::Linear => Ok(Some(first.value)),
        };
    }
    for (index, pair) in keys.windows(2).enumerate() {
        let left = &pair[0];
        let right = &pair[1];
        if frame < right.frame {
            let width = right.frame - left.frame;
            let fraction = (frame - left.frame) / width;
            let value = match left.interpolation {
                Interpolation::Constant => left.value,
                Interpolation::Linear => left.value + fraction * (right.value - left.value),
                Interpolation::Bezier => {
                    let start_tangent = auto_tangent(keys, index);
                    let end_tangent = auto_tangent(keys, index + 1);
                    let control1 = left.handle_right.unwrap_or([
                        left.frame + width / 3.0,
                        left.value + start_tangent * width / 3.0,
                    ]);
                    let control2 = right.handle_left.unwrap_or([
                        right.frame - width / 3.0,
                        right.value - end_tangent * width / 3.0,
                    ]);
                    let parameter = bezier_parameter_for_frame(
                        frame,
                        left.frame,
                        control1[0],
                        control2[0],
                        right.frame,
                        fraction,
                    );
                    cubic_bezier(left.value, control1[1], control2[1], right.value, parameter)
                }
            };
            return finite_sample(curve, value);
        }
    }
    let last_index = keys.len() - 1;
    let last = &keys[last_index];
    if curve.extrapolation == Extrapolation::Constant || last_index == 0 {
        return Ok(Some(last.value));
    }
    let slope = endpoint_tangent(keys, last_index);
    finite_sample(curve, last.value + slope * (frame - last.frame))
}

fn cubic_bezier(start: f64, control1: f64, control2: f64, end: f64, parameter: f64) -> f64 {
    let inverse = 1.0 - parameter;
    inverse.powi(3) * start
        + 3.0 * inverse.powi(2) * parameter * control1
        + 3.0 * inverse * parameter.powi(2) * control2
        + parameter.powi(3) * end
}

fn bezier_parameter_for_frame(
    frame: f64,
    start: f64,
    control1: f64,
    control2: f64,
    end: f64,
    initial: f64,
) -> f64 {
    let mut lower = 0.0;
    let mut upper = 1.0;
    let mut parameter = initial;
    for _ in 0..64 {
        let value = cubic_bezier(start, control1, control2, end, parameter);
        if value < frame {
            lower = parameter;
        } else {
            upper = parameter;
        }
        parameter = lower.midpoint(upper);
    }
    parameter
}

/// Shape-preserving tangent estimation used by automatic Bezier handles.
fn auto_tangent(keys: &[crate::model::Keyframe], index: usize) -> f64 {
    if keys.len() < 2 {
        return 0.0;
    }
    if index == 0 {
        return endpoint_tangent(keys, 0);
    }
    if index + 1 == keys.len() {
        return endpoint_tangent(keys, index);
    }
    let previous = &keys[index - 1];
    let current = &keys[index];
    let next = &keys[index + 1];
    let previous_width = current.frame - previous.frame;
    let next_width = next.frame - current.frame;
    let previous_slope = (current.value - previous.value) / previous_width;
    let next_slope = (next.value - current.value) / next_width;
    if crate::float::equal_f64(previous_slope, 0.0)
        || crate::float::equal_f64(next_slope, 0.0)
        || previous_slope.is_sign_positive() != next_slope.is_sign_positive()
    {
        return 0.0;
    }
    let weight1 = 2.0 * next_width + previous_width;
    let weight2 = next_width + 2.0 * previous_width;
    (weight1 + weight2) / (weight1 / previous_slope + weight2 / next_slope)
}

fn endpoint_tangent(keys: &[crate::model::Keyframe], index: usize) -> f64 {
    if keys.len() < 2 {
        return 0.0;
    }
    if index == 0 {
        let first = &keys[0];
        let second = &keys[1];
        let first_width = second.frame - first.frame;
        let first_slope = (second.value - first.value) / first_width;
        if keys.len() == 2 {
            return first_slope;
        }
        let third = &keys[2];
        let second_width = third.frame - second.frame;
        let second_slope = (third.value - second.value) / second_width;
        let tangent = ((2.0 * first_width + second_width) * first_slope
            - first_width * second_slope)
            / (first_width + second_width);
        return clamp_endpoint_tangent(tangent, first_slope, second_slope);
    }
    let last = &keys[index];
    let previous = &keys[index - 1];
    let last_width = last.frame - previous.frame;
    let last_slope = (last.value - previous.value) / last_width;
    if keys.len() == 2 {
        return last_slope;
    }
    let before_previous = &keys[index - 2];
    let previous_width = previous.frame - before_previous.frame;
    let previous_slope = (previous.value - before_previous.value) / previous_width;
    let tangent = ((2.0 * last_width + previous_width) * last_slope - last_width * previous_slope)
        / (last_width + previous_width);
    clamp_endpoint_tangent(tangent, last_slope, previous_slope)
}

fn clamp_endpoint_tangent(tangent: f64, first_slope: f64, second_slope: f64) -> f64 {
    if crate::float::equal_f64(tangent, 0.0)
        || tangent.is_sign_positive() != first_slope.is_sign_positive()
    {
        return 0.0;
    }
    if first_slope.is_sign_positive() != second_slope.is_sign_positive()
        && tangent.abs() > 3.0 * first_slope.abs()
    {
        return 3.0 * first_slope;
    }
    tangent
}

fn finite_sample(curve: &FCurve, value: f64) -> Result<Option<f64>> {
    if !value.is_finite() {
        return Err(invalid_curve(
            curve,
            "curve evaluation produced a non-finite value",
        ));
    }
    Ok(Some(value))
}
#[cfg(kani)]
#[kani::proof]
fn kani_finite_sample_rejects_non_finite_values() {
    let value: f64 = kani::any();
    let curve = FCurve {
        path: "transform.translation".to_owned(),
        index: 0,
        keyframes: Vec::new(),
        extrapolation: Extrapolation::Constant,
    };
    let result = finite_sample(&curve, value);
    if value.is_finite() {
        assert!(matches!(result, Ok(Some(sampled)) if sampled == value));
    } else {
        assert!(result.is_err());
    }
}

fn invalid_curve(curve: &FCurve, message: &str) -> PotError {
    PotError::with_details(
        ErrorCode::SceneInvalid,
        message,
        json!({ "path": curve.path, "index": curve.index }),
    )
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests")]

    use proptest::prelude::*;

    use crate::model::{
        Action, Extrapolation, FCurve, Id, Interpolation, Keyframe, NlaBlendType, NlaExtrapolation,
        NlaStrip, Node, PoseBone, SceneDoc, Transform,
    };

    use super::{animated_pose_bones, animated_transform, sample_curve, strip_action_frame};

    fn node(action: Option<Id>) -> Node {
        Node {
            name: "Animated".to_owned(),
            kind: "empty".to_owned(),
            primitive: None,
            tags: Vec::new(),
            parent: None,
            parent_inverse: None,
            transform: Transform::from_rotation_deg([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [1.0; 3]),
            data: None,
            materials: Vec::new(),
            modifiers: Vec::new(),
            visible: true,
            render_visible: true,
            selectable: true,
            action,
            properties: serde_json::Map::new(),
            ..Node::default()
        }
    }

    fn curve(keys: Vec<Keyframe>, extrapolation: Extrapolation) -> FCurve {
        FCurve {
            path: "transform.translation".to_owned(),
            index: 0,
            keyframes: keys,
            extrapolation,
        }
    }

    #[test]
    fn pose_action_curve_evaluates_a_named_bone_channel() {
        let action_id = Id::new("pose_action").unwrap();
        let bone_id = Id::new("bone_arm").unwrap();
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        doc.actions.insert(
            action_id.clone(),
            Action {
                name: "Pose".to_owned(),
                fcurves: vec![FCurve {
                    path: "pose.bone_arm.translation".to_owned(),
                    index: 0,
                    keyframes: vec![
                        Keyframe {
                            frame: 1.0,
                            value: 2.0,
                            interpolation: Interpolation::Linear,
                            ..Keyframe::default()
                        },
                        Keyframe {
                            frame: 11.0,
                            value: 12.0,
                            interpolation: Interpolation::Linear,
                            ..Keyframe::default()
                        },
                    ],
                    extrapolation: Extrapolation::Constant,
                }],
                ..Action::default()
            },
        );
        let animated = animated_pose_bones(&node(Some(action_id)), &doc, 6.0).unwrap();
        assert_eq!(
            animated[&bone_id],
            PoseBone {
                translation: [7.0, 0.0, 0.0],
                ..PoseBone::default()
            }
        );
    }

    #[test]
    fn no_action_preserves_the_raw_transform() {
        let doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        let evaluated = animated_transform(&node(None), &doc, 4.0).unwrap();
        assert_eq!(evaluated.translation, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn linear_action_curve_updates_the_requested_transform_component() {
        let action_id = Id::new("move_action").unwrap();
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        doc.actions.insert(
            action_id.clone(),
            Action {
                name: "Move".to_owned(),
                fcurves: vec![curve(
                    vec![
                        Keyframe {
                            frame: 1.0,
                            value: 0.0,
                            interpolation: Interpolation::Linear,
                            ..Keyframe::default()
                        },
                        Keyframe {
                            frame: 4.0,
                            value: 6.0,
                            interpolation: Interpolation::Linear,
                            ..Keyframe::default()
                        },
                    ],
                    Extrapolation::Constant,
                )],
                ..Action::default()
            },
        );
        let evaluated = animated_transform(&node(Some(action_id)), &doc, 2.5).unwrap();
        assert!((evaluated.translation[0] - 3.0).abs() < 1.0e-12);
        assert_eq!(evaluated.translation[1..], [2.0, 3.0]);
    }

    #[test]
    fn constant_and_linear_interpolation_match_their_exact_definitions() {
        let keys = vec![
            Keyframe {
                frame: 1.0,
                value: 2.0,
                interpolation: Interpolation::Constant,
                ..Keyframe::default()
            },
            Keyframe {
                frame: 5.0,
                value: 10.0,
                interpolation: Interpolation::Linear,
                ..Keyframe::default()
            },
        ];
        let constant = curve(keys.clone(), Extrapolation::Constant);
        assert_eq!(sample_curve(&constant, 4.0).unwrap(), Some(2.0));
        assert_eq!(sample_curve(&constant, 5.0).unwrap(), Some(10.0));

        let linear = curve(
            vec![
                Keyframe {
                    frame: 1.0,
                    value: 2.0,
                    interpolation: Interpolation::Linear,
                    ..Keyframe::default()
                },
                Keyframe {
                    frame: 5.0,
                    value: 10.0,
                    interpolation: Interpolation::Linear,
                    ..Keyframe::default()
                },
            ],
            Extrapolation::Linear,
        );
        assert_eq!(sample_curve(&linear, 3.0).unwrap(), Some(6.0));
        assert_eq!(sample_curve(&linear, 0.0).unwrap(), Some(0.0));
        assert_eq!(sample_curve(&linear, 6.0).unwrap(), Some(12.0));
    }

    proptest! {
        #[test]
        fn unit_nla_strip_mapping_matches_identity_reference(frame in 1.0_f64..11.0) {
            let strip = NlaStrip {
                id: Id::new("strip").unwrap(),
                action: Id::new("action").unwrap(),
                frame_start: 1.0,
                frame_end: 11.0,
                action_frame_start: 1.0,
                action_frame_end: 11.0,
                scale: 1.0,
                repeat: 1.0,
                blend_type: NlaBlendType::Replace,
                influence: 1.0,
                extrapolation: NlaExtrapolation::Nothing,
                blend_in: 0.0,
                blend_out: 0.0,
            };
            let mapped = strip_action_frame(&strip, frame).unwrap().unwrap();
            prop_assert!((mapped - frame).abs() <= 1.0e-10);
        }

        #[test]
        fn auto_clamped_bezier_stays_within_each_adjacent_keyframe_range(
            values in prop::collection::vec(-1.0e6_f64..1.0e6, 2..20),
        ) {
            let keys = values
                .iter()
                .enumerate()
                .map(|(index, value)| Keyframe {
                    frame: f64::from(u32::try_from(index).unwrap()),
                    value: *value,
                    interpolation: Interpolation::Bezier,
                    ..Keyframe::default()
                })
                .collect::<Vec<_>>();
            let fcurve = curve(keys.clone(), Extrapolation::Constant);
            for (index, pair) in keys.windows(2).enumerate() {
                let minimum = pair[0].value.min(pair[1].value) - 1.0e-8;
                let maximum = pair[0].value.max(pair[1].value) + 1.0e-8;
                for fraction in [0.125, 0.25, 0.5, 0.75, 0.875] {
                    let frame = f64::from(u32::try_from(index).unwrap()) + fraction;
                    let value = sample_curve(&fcurve, frame).unwrap().unwrap();
                    prop_assert!(value >= minimum && value <= maximum);
                }
            }
        }
    }
}
