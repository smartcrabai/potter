// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   source/blender/blenkernel/intern/fcurve.cc, SPDX-FileCopyrightText: 2009 Blender Authors, Joshua Leung. All rights reserved., GPL-2.0-or-later.

use std::collections::BTreeMap;

use glam::{DQuat, EulerRot};
use serde_json::json;

use crate::{
    error::{ErrorCode, PotError, Result},
    model::{
        Action, Extrapolation, FCurve, Id, Interpolation, ModifierParameterBinding, NlaBlendType,
        NlaExtrapolation, NlaStrip, Node, PoseBone, SceneDoc, Transform, normalize_rotation,
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
/// Evaluate one ID-bound modifier scalar through the node's NLA tracks and active action.
///
/// The optional result distinguishes an unanimated property from an animated value equal to its
/// base. Curve bindings are resolved at the Blender bridge boundary; this evaluator never parses
/// modifier RNA paths or guesses identity from modifier names.
///
/// # Errors
///
/// Returns scene/evaluation errors for invalid curve indices, node/modifier bindings, F-Curves,
/// NLA strips or non-finite results.
pub(crate) fn animated_modifier_scalar(
    node_id: &Id,
    node: &Node,
    doc: &SceneDoc,
    modifier_id: &Id,
    parameter: &str,
    base_value: f64,
    frame: f64,
) -> Result<Option<f64>> {
    if !frame.is_finite() || !base_value.is_finite() {
        return Err(PotError::new(
            ErrorCode::InvalidArgument,
            "modifier animation frame and base value must be finite",
        ));
    }
    let modifier = node
        .modifiers
        .iter()
        .find(|modifier| &modifier.id == modifier_id)
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "modifier animation target does not exist on its node",
                json!({"node":node_id,"modifier":modifier_id,"parameter":parameter}),
            )
        })?;
    if parameter.is_empty()
        || !modifier
            .params
            .get(parameter)
            .or_else(|| {
                crate::params::default_value(
                    crate::params::ParameterFamily::Modifier,
                    &modifier.modifier_type,
                    parameter,
                )
            })
            .is_some_and(|value| value.as_f64().is_some())
    {
        return Err(PotError::with_details(
            ErrorCode::SceneInvalid,
            "modifier animation binding must target a known scalar parameter",
            json!({"node":node_id,"modifier":modifier_id,"parameter":parameter}),
        ));
    }

    let has_solo = node
        .nla_tracks
        .iter()
        .any(|track| track.solo && !track.mute);
    let mut value = base_value;
    let mut animated = false;
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
                    json!({"action":strip.action,"strip":strip.id}),
                )
            })?;
            let assigned_slot = strip.action_slot.as_ref().or_else(|| {
                if action.slots.len() == 1 {
                    action.slots.first().map(|slot| &slot.id)
                } else {
                    None
                }
            });
            let Some(sampled) = sample_modifier_scalar_action(
                node_id,
                doc,
                &strip.action,
                action,
                modifier_id,
                parameter,
                assigned_slot,
                action_frame,
            )?
            else {
                continue;
            };
            let influence = strip_influence(strip, frame)?;
            value = blend_translation_scalar(value, sampled, strip.blend_type, influence);
            if !value.is_finite() {
                return Err(PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "modifier animation evaluated to a non-finite scalar",
                    json!({"action":strip.action,"modifier":modifier_id,"parameter":parameter,"frame":frame}),
                ));
            }
            animated = true;
        }
    }
    if let Some(action_id) = &node.action {
        let action = doc.actions.get(action_id).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "node action reference does not exist",
                json!({"action":action_id}),
            )
        })?;
        let assigned_slot = action.active_slots.get(node_id).or_else(|| {
            if action.slots.len() == 1 {
                action.slots.first().map(|slot| &slot.id)
            } else {
                None
            }
        });
        if let Some(sampled) = sample_modifier_scalar_action(
            node_id,
            doc,
            action_id,
            action,
            modifier_id,
            parameter,
            assigned_slot,
            frame,
        )? {
            value = sampled;
            animated = true;
        }
    }
    Ok(animated.then_some(value))
}

fn sample_modifier_scalar_action(
    node_id: &Id,
    doc: &SceneDoc,
    action_id: &Id,
    action: &Action,
    modifier_id: &Id,
    parameter: &str,
    assigned_slot: Option<&Id>,
    frame: f64,
) -> Result<Option<f64>> {
    if let Some(assigned_slot) = assigned_slot
        && !action.slots.iter().any(|slot| &slot.id == assigned_slot)
    {
        return Err(PotError::with_details(
            ErrorCode::SceneInvalid,
            "modifier animation assignment references a missing Action slot",
            json!({"action":action_id,"action_slot":assigned_slot}),
        ));
    }
    let mut sampled = None;
    for (curve_index, bindings) in &action.modifier_bindings {
        let index = usize::try_from(*curve_index).map_err(|_| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "modifier animation F-Curve index is invalid",
                json!({"action":action_id,"curve_index":curve_index}),
            )
        })?;
        let curve = action.fcurves.get(index).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "modifier animation binding refers to a missing F-Curve",
                json!({"action":action_id,"curve_index":curve_index}),
            )
        })?;
        if bindings.is_empty() {
            return Err(PotError::with_details(
                ErrorCode::SceneInvalid,
                "modifier animation F-Curve has an empty binding list",
                json!({"action":action_id,"curve_index":curve_index}),
            ));
        }
        for binding in bindings {
            validate_modifier_parameter_binding(doc, action_id, action, *curve_index, binding)?;
            if &binding.node != node_id
                || &binding.modifier != modifier_id
                || binding.parameter.as_str() != parameter
            {
                continue;
            }
            if binding.action_slot.as_ref() != assigned_slot {
                if assigned_slot.is_none() && binding.action_slot.is_some() {
                    return Err(PotError::with_details(
                        ErrorCode::SceneInvalid,
                        "modifier animation binding has no active Action-slot assignment",
                        json!({"action":action_id,"curve_index":curve_index,"node":node_id,"action_slot":binding.action_slot}),
                    ));
                }
                continue;
            }
            if curve.index != 0 {
                return Err(PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "modifier scalar F-Curve must use component index zero",
                    json!({"action":action_id,"curve_index":curve_index,"index":curve.index}),
                ));
            }
            if let Some(value) = sample_curve(curve, frame)? {
                sampled = Some(value);
            }
        }
    }
    Ok(sampled)
}

fn validate_modifier_parameter_binding(
    doc: &SceneDoc,
    action_id: &Id,
    action: &Action,
    curve_index: u32,
    binding: &ModifierParameterBinding,
) -> Result<()> {
    let node = doc.nodes.get(&binding.node).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::SceneInvalid,
            "modifier animation binding refers to a missing node",
            json!({"action":action_id,"curve_index":curve_index,"node":binding.node}),
        )
    })?;
    let modifier = node
        .modifiers
        .iter()
        .find(|modifier| modifier.id == binding.modifier)
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "modifier animation binding refers to a missing modifier",
                json!({"action":action_id,"curve_index":curve_index,"node":binding.node,"modifier":binding.modifier}),
            )
        })?;
    if binding.parameter.is_empty()
        || !modifier
            .params
            .get(&binding.parameter)
            .or_else(|| {
                crate::params::default_value(
                    crate::params::ParameterFamily::Modifier,
                    &modifier.modifier_type,
                    &binding.parameter,
                )
            })
            .is_some_and(|value| value.as_f64().is_some())
    {
        return Err(PotError::with_details(
            ErrorCode::SceneInvalid,
            "modifier animation binding must target a known scalar parameter",
            json!({"action":action_id,"curve_index":curve_index,"modifier":binding.modifier,"parameter":binding.parameter}),
        ));
    }
    let slot_is_assigned = match &binding.action_slot {
        Some(slot_id) => {
            let slot_exists = action.slots.iter().any(|slot| &slot.id == slot_id);
            let active_assignment = action.active_slots.get(&binding.node) == Some(slot_id);
            let nla_assignment = node.nla_tracks.iter().any(|track| {
                track.strips.iter().any(|strip| {
                    &strip.action == action_id && strip.action_slot.as_ref() == Some(slot_id)
                })
            });
            slot_exists
                && (active_assignment
                    || nla_assignment
                    || (action.slots.len() == 1
                        && action.active_slots.get(&binding.node).is_none()
                        && action.slots[0].node == binding.node))
        }
        None => action.slots.is_empty(),
    };
    if !slot_is_assigned {
        return Err(PotError::with_details(
            ErrorCode::SceneInvalid,
            "modifier animation binding does not match a slot assigned to its node",
            json!({"action":action_id,"curve_index":curve_index,"node":binding.node,"action_slot":binding.action_slot}),
        ));
    }
    Ok(())
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
    for (curve_index, curve) in action.fcurves.iter().enumerate() {
        if u32::try_from(curve_index)
            .ok()
            .is_some_and(|index| action.modifier_bindings.contains_key(&index))
        {
            continue;
        }
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

fn blend_translation_scalar(current: f64, target: f64, blend: NlaBlendType, influence: f64) -> f64 {
    match blend {
        NlaBlendType::Replace => current * (1.0 - influence) + target * influence,
        NlaBlendType::Add | NlaBlendType::Combine => current + target * influence,
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
            current.translation[component] = blend_translation_scalar(
                current.translation[component],
                sampled.translation[component],
                blend,
                influence,
            );
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
        Action, ActionSlot, Extrapolation, FCurve, Id, Interpolation, Keyframe, Modifier,
        ModifierParameterBinding, NlaBlendType, NlaExtrapolation, NlaStrip, NlaTrack, Node,
        PoseBone, SceneDoc, Transform,
    };

    use super::{
        animated_modifier_scalar, animated_pose_bones, animated_transform, sample_curve,
        strip_action_frame,
    };

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

    fn ocean_modifier(id: &Id) -> Modifier {
        serde_json::from_value(serde_json::json!({
            "id": id,
            "type": "ocean",
            "name": "Ocean",
            "params": {"time": 0.5},
        }))
        .unwrap()
    }

    fn ocean_time_curve(start_value: f64, end_value: f64) -> FCurve {
        FCurve {
            path: "modifiers[\"Ocean\"].time".to_owned(),
            index: 0,
            keyframes: vec![
                Keyframe {
                    frame: 1.0,
                    value: start_value,
                    interpolation: Interpolation::Linear,
                    ..Keyframe::default()
                },
                Keyframe {
                    frame: 3.0,
                    value: end_value,
                    interpolation: Interpolation::Linear,
                    ..Keyframe::default()
                },
            ],
            extrapolation: Extrapolation::Constant,
        }
    }

    fn ocean_modifier_binding(
        node: &Id,
        modifier: &Id,
        action_slot: &Id,
    ) -> ModifierParameterBinding {
        ModifierParameterBinding {
            node: node.clone(),
            modifier: modifier.clone(),
            parameter: "time".to_owned(),
            action_slot: Some(action_slot.clone()),
        }
    }

    fn ocean_time_action(
        action_slot: &Id,
        bindings: Vec<ModifierParameterBinding>,
        start_value: f64,
        end_value: f64,
    ) -> Action {
        let active_slots = bindings
            .iter()
            .map(|binding| (binding.node.clone(), action_slot.clone()))
            .collect();
        Action {
            name: "Ocean Time".to_owned(),
            fcurves: vec![ocean_time_curve(start_value, end_value)],
            slots: vec![ActionSlot {
                id: action_slot.clone(),
                node: bindings.first().expect("test binding exists").node.clone(),
                blender_identifier: None,
            }],
            active_slots,
            modifier_bindings: std::collections::BTreeMap::from([(0, bindings)]),
        }
    }

    fn ocean_node(action: Option<Id>, modifier_id: &Id) -> Node {
        let mut node = node(action);
        node.modifiers.push(ocean_modifier(modifier_id));
        node
    }

    #[test]
    fn finite_sample_rejects_every_non_finite_value() {
        let channel = curve(Vec::new(), Extrapolation::Constant);
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(super::finite_sample(&channel, value).is_err());
        }
        assert_eq!(super::finite_sample(&channel, -0.5).unwrap(), Some(-0.5));
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
    fn active_action_samples_modifier_scalar_by_stable_identity() {
        let node_id = Id::new("ocean_node").unwrap();
        let modifier_id = Id::new("ocean_modifier").unwrap();
        let action_id = Id::new("ocean_action").unwrap();
        let slot_id = Id::new("ocean_slot").unwrap();
        let subject = ocean_node(Some(action_id.clone()), &modifier_id);
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        doc.nodes.insert(node_id.clone(), subject.clone());
        doc.actions.insert(
            action_id,
            ocean_time_action(
                &slot_id,
                vec![ocean_modifier_binding(&node_id, &modifier_id, &slot_id)],
                0.5,
                2.5,
            ),
        );

        assert_eq!(
            animated_modifier_scalar(&node_id, &subject, &doc, &modifier_id, "time", 0.5, 2.0)
                .unwrap(),
            Some(1.5)
        );
        assert_eq!(
            animated_transform(&subject, &doc, 2.0).unwrap().translation,
            subject.transform.translation
        );
    }

    #[test]
    fn shared_action_slot_binds_one_curve_to_each_objects_modifier_id() {
        let node_a_id = Id::new("ocean_node_a").unwrap();
        let node_b_id = Id::new("ocean_node_b").unwrap();
        let modifier_a_id = Id::new("ocean_modifier_a").unwrap();
        let modifier_b_id = Id::new("ocean_modifier_b").unwrap();
        let action_id = Id::new("ocean_action_shared").unwrap();
        let slot_id = Id::new("ocean_slot_shared").unwrap();
        let subject_a = ocean_node(Some(action_id.clone()), &modifier_a_id);
        let subject_b = ocean_node(Some(action_id.clone()), &modifier_b_id);
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        doc.nodes.insert(node_a_id.clone(), subject_a.clone());
        doc.nodes.insert(node_b_id.clone(), subject_b.clone());
        doc.actions.insert(
            action_id,
            ocean_time_action(
                &slot_id,
                vec![
                    ocean_modifier_binding(&node_a_id, &modifier_a_id, &slot_id),
                    ocean_modifier_binding(&node_b_id, &modifier_b_id, &slot_id),
                ],
                0.5,
                2.5,
            ),
        );

        assert_eq!(
            animated_modifier_scalar(
                &node_a_id,
                &subject_a,
                &doc,
                &modifier_a_id,
                "time",
                0.5,
                2.0,
            )
            .unwrap(),
            Some(1.5)
        );
        assert_eq!(
            animated_modifier_scalar(
                &node_b_id,
                &subject_b,
                &doc,
                &modifier_b_id,
                "time",
                0.5,
                2.0,
            )
            .unwrap(),
            Some(1.5)
        );
    }

    #[test]
    fn active_action_slot_selects_only_its_modifier_curve() {
        let node_a_id = Id::new("slot_node_a").unwrap();
        let node_b_id = Id::new("slot_node_b").unwrap();
        let modifier_a_id = Id::new("slot_modifier_a").unwrap();
        let modifier_b_id = Id::new("slot_modifier_b").unwrap();
        let action_id = Id::new("slot_action").unwrap();
        let slot_a_id = Id::new("slot_a").unwrap();
        let slot_b_id = Id::new("slot_b").unwrap();
        let subject_a = ocean_node(Some(action_id.clone()), &modifier_a_id);
        let subject_b = ocean_node(Some(action_id.clone()), &modifier_b_id);
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        doc.nodes.insert(node_a_id.clone(), subject_a.clone());
        doc.nodes.insert(node_b_id.clone(), subject_b.clone());
        doc.actions.insert(
            action_id,
            Action {
                name: "Slot-specific Ocean Time".to_owned(),
                fcurves: vec![ocean_time_curve(0.5, 2.5), ocean_time_curve(0.5, 4.5)],
                slots: vec![
                    ActionSlot {
                        id: slot_a_id.clone(),
                        node: node_a_id.clone(),
                        blender_identifier: Some("OBJECT_A".to_owned()),
                    },
                    ActionSlot {
                        id: slot_b_id.clone(),
                        node: node_b_id.clone(),
                        blender_identifier: Some("OBJECT_B".to_owned()),
                    },
                ],
                active_slots: std::collections::BTreeMap::from([
                    (node_a_id.clone(), slot_a_id.clone()),
                    (node_b_id.clone(), slot_b_id.clone()),
                ]),
                modifier_bindings: std::collections::BTreeMap::from([
                    (
                        0,
                        vec![ocean_modifier_binding(
                            &node_a_id,
                            &modifier_a_id,
                            &slot_a_id,
                        )],
                    ),
                    (
                        1,
                        vec![ocean_modifier_binding(
                            &node_b_id,
                            &modifier_b_id,
                            &slot_b_id,
                        )],
                    ),
                ]),
            },
        );

        assert_eq!(
            animated_modifier_scalar(
                &node_a_id,
                &subject_a,
                &doc,
                &modifier_a_id,
                "time",
                0.5,
                2.0,
            )
            .unwrap(),
            Some(1.5)
        );
        assert_eq!(
            animated_modifier_scalar(
                &node_b_id,
                &subject_b,
                &doc,
                &modifier_b_id,
                "time",
                0.5,
                2.0,
            )
            .unwrap(),
            Some(2.5)
        );
    }

    #[test]
    fn modifier_scalar_nla_reuses_native_frame_influence_and_blend_rules() {
        for (blend_type, expected) in [
            (NlaBlendType::Replace, 1.25),
            (NlaBlendType::Add, 1.5),
            (NlaBlendType::Combine, 1.5),
        ] {
            let node_id = Id::new("ocean_nla_node").unwrap();
            let modifier_id = Id::new("ocean_nla_modifier").unwrap();
            let action_id = Id::new("ocean_nla_action").unwrap();
            let slot_id = Id::new("ocean_nla_slot").unwrap();
            let mut subject = ocean_node(None, &modifier_id);
            subject.nla_tracks.push(NlaTrack {
                id: Id::new("ocean_nla_track").unwrap(),
                name: "Ocean Time".to_owned(),
                mute: false,
                solo: false,
                strips: vec![NlaStrip {
                    id: Id::new("ocean_nla_strip").unwrap(),
                    action: action_id.clone(),
                    action_slot: Some(slot_id.clone()),
                    frame_start: 1.0,
                    frame_end: 3.0,
                    action_frame_start: 1.0,
                    action_frame_end: 3.0,
                    scale: 1.0,
                    repeat: 1.0,
                    blend_type,
                    influence: 0.5,
                    extrapolation: NlaExtrapolation::Nothing,
                    blend_in: 0.0,
                    blend_out: 0.0,
                }],
            });
            let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
            doc.nodes.insert(node_id.clone(), subject.clone());
            let mut nla_action = ocean_time_action(
                &slot_id,
                vec![ocean_modifier_binding(&node_id, &modifier_id, &slot_id)],
                1.5,
                2.5,
            );
            nla_action.active_slots.clear();
            doc.actions.insert(action_id, nla_action);

            assert_eq!(
                animated_modifier_scalar(&node_id, &subject, &doc, &modifier_id, "time", 0.5, 2.0,)
                    .unwrap(),
                Some(expected)
            );
        }
    }

    #[test]
    fn unbound_modifier_curve_is_an_error_not_a_static_fallback() {
        let node_id = Id::new("ocean_unbound_node").unwrap();
        let modifier_id = Id::new("ocean_unbound_modifier").unwrap();
        let action_id = Id::new("ocean_unbound_action").unwrap();
        let subject = ocean_node(Some(action_id.clone()), &modifier_id);
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        doc.nodes.insert(node_id.clone(), subject.clone());
        doc.actions.insert(
            action_id,
            Action {
                name: "Unbound Ocean Time".to_owned(),
                fcurves: vec![ocean_time_curve(0.5, 2.5)],
                modifier_bindings: std::collections::BTreeMap::from([(0, Vec::new())]),
            },
        );

        assert!(
            animated_modifier_scalar(&node_id, &subject, &doc, &modifier_id, "time", 0.5, 2.0)
                .is_err()
        );
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
                action_slot: None,
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
