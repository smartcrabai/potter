// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   source/blender/blenkernel/intern/fcurve.cc, SPDX-FileCopyrightText: 2009 Blender Authors, Joshua Leung. All rights reserved., GPL-2.0-or-later.

use std::collections::BTreeMap;

use glam::{DQuat, EulerRot};
use serde_json::{Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    model::{
        Action, Extrapolation, FCurve, Id, Interpolation, LightType, ModifierParameterBinding,
        NlaBlendType, NlaExtrapolation, NlaStrip, NlaTrack, Node, PoseBone, PoseRotationMode,
        SceneDoc, Transform, normalize_rotation,
    },
};

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
    let nla_mode = node_uses_nla(node, has_solo);
    let mut transform = node.transform.clone();
    let rotation_mode_is_from_blender = node.properties.contains_key("blender_object_metadata");
    let rotation_mode =
        PoseRotationMode::from_blender(&transform.rotation_mode).unwrap_or_default();
    let initial_kind = if rotation_mode == PoseRotationMode::AxisAngle {
        RotationChannelKind::AxisAngle
    } else if rotation_mode.euler_order().is_some() {
        RotationChannelKind::Euler
    } else {
        RotationChannelKind::Quaternion
    };
    let domain =
        transform_nla_domain(node, doc, nla_mode, has_solo, rotation_mode_is_from_blender)?;
    let mut rotation_state = if domain.rotation {
        Some(rotation_accumulator_from_quaternion(
            transform.rotation,
            initial_kind,
            rotation_mode,
        )?)
    } else {
        None
    };
    for component in 0..3 {
        if domain.translation[component] {
            transform.translation[component] = 0.0;
        }
        if domain.scale[component] {
            transform.scale[component] = 1.0;
        }
    }
    if domain.rotation {
        let state = rotation_state.as_mut().ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "rotation domain has no accumulator",
            )
        })?;
        reset_rotation_accumulator(
            state,
            domain.rotation_kind,
            domain.rotation_components,
            rotation_mode,
        )?;
        transform.rotation = rotation_accumulator_to_quaternion(*state, rotation_mode)?;
    }
    for (track_index, track) in node.nla_tracks.iter().enumerate() {
        if track.mute || (has_solo && !track.solo) {
            continue;
        }
        let Some((strip_index, strip_frame)) = active_strip_for_track(track, frame) else {
            continue;
        };
        let strip = &track.strips[strip_index];
        if strip.mute {
            continue;
        }
        let (blend_in, blend_out) = auto_blend_lengths(node, track_index, strip_index);
        let influence = strip_influence(strip, strip_frame, blend_in, blend_out)?;
        if influence <= 0.0 {
            continue;
        }
        let Some(action_frame) = strip_action_frame(strip, strip_frame)? else {
            continue;
        };
        let action = doc.actions.get(&strip.action).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "NLA strip action reference does not exist",
                json!({ "action": strip.action, "strip": strip.id }),
            )
        })?;
        let (sampled, channels) = sample_action_transform(
            node,
            doc,
            &transform,
            rotation_state.as_ref(),
            initial_kind,
            action,
            &strip.action,
            action_frame,
            rotation_mode_is_from_blender,
            true,
        )?;
        blend_transform(
            &mut transform,
            &mut rotation_state,
            initial_kind,
            rotation_mode,
            &sampled,
            channels,
            strip.blend_type,
            influence,
        )?;
    }
    if let Some(action_id) = &node.action {
        let action = doc.actions.get(action_id).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "node action reference does not exist",
                json!({"action":action_id}),
            )
        })?;
        let (action_frame, blend, influence) =
            active_action_controls(node, action, frame, nla_mode, has_solo)?;
        if let Some(action_frame) = action_frame {
            let (sampled, channels) = sample_action_transform(
                node,
                doc,
                &transform,
                rotation_state.as_ref(),
                initial_kind,
                action,
                action_id,
                action_frame,
                rotation_mode_is_from_blender,
                nla_mode,
            )?;
            blend_transform(
                &mut transform,
                &mut rotation_state,
                initial_kind,
                rotation_mode,
                &sampled,
                channels,
                blend,
                influence,
            )?;
        }
    }
    if let Some(rotation_state) = rotation_state {
        transform.rotation = rotation_accumulator_to_quaternion(rotation_state, rotation_mode)?;
    }
    Ok(transform)
}

fn transform_nla_domain(
    node: &Node,
    doc: &SceneDoc,
    nla_mode: bool,
    has_solo: bool,
    rotation_mode_is_from_blender: bool,
) -> Result<TransformChannels> {
    let mut channels = TransformChannels::default();
    if nla_mode && let Some(action_id) = &node.action {
        let action = doc.actions.get(action_id).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "node action reference does not exist",
                json!({"action":action_id}),
            )
        })?;
        add_transform_action_domain(
            action,
            &mut channels,
            &node.transform,
            rotation_mode_is_from_blender,
        )?;
    }
    for track in &node.nla_tracks {
        if track.mute || (has_solo && !track.solo) {
            continue;
        }
        for strip in &track.strips {
            let action = doc.actions.get(&strip.action).ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "NLA strip action reference does not exist",
                    json!({"action":strip.action,"strip":strip.id}),
                )
            })?;
            add_transform_action_domain(
                action,
                &mut channels,
                &node.transform,
                rotation_mode_is_from_blender,
            )?;
        }
    }
    Ok(channels)
}

fn add_transform_action_domain(
    action: &Action,
    channels: &mut TransformChannels,
    transform: &Transform,
    rotation_mode_is_from_blender: bool,
) -> Result<()> {
    let rotation_mode =
        PoseRotationMode::from_blender(&transform.rotation_mode).unwrap_or_default();
    let euler_channels_enabled =
        rotation_mode.euler_order().is_some() || !rotation_mode_is_from_blender;
    for (curve_index, curve) in action.fcurves.iter().enumerate() {
        if u32::try_from(curve_index)
            .is_ok_and(|index| action.modifier_bindings.contains_key(&index))
        {
            continue;
        }
        match curve.path.as_str() {
            "transform.translation" => {
                channels.translation[component_index(curve, 3)?] = true;
            }
            "transform.scale" => {
                channels.scale[component_index(curve, 3)?] = true;
            }
            "transform.rotation" | "transform.rotation_quaternion"
                if !rotation_mode_is_from_blender
                    || rotation_mode == PoseRotationMode::Quaternion =>
            {
                component_index(curve, 4)?;
                channels.rotation = true;
                channels.rotation_components = [true; 4];
                channels.rotation_kind = RotationChannelKind::Quaternion;
            }
            "transform.rotation_euler" if euler_channels_enabled => {
                let component = component_index(curve, 3)?;
                channels.rotation = true;
                channels.rotation_components[component] = true;
                channels.rotation_kind = RotationChannelKind::Euler;
            }
            "transform.rotation_axis_angle"
                if !rotation_mode_is_from_blender
                    || rotation_mode == PoseRotationMode::AxisAngle =>
            {
                let component = component_index(curve, 4)?;
                channels.rotation = true;
                channels.rotation_components[component] = true;
                channels.rotation_kind = RotationChannelKind::AxisAngle;
            }
            _ => {}
        }
    }
    Ok(())
}

fn custom_property_component<'a>(node: &'a Node, path: &str, index: u32) -> Option<&'a Value> {
    let key = custom_property_name(path)?;
    let property = node
        .properties
        .get("_blender_custom_properties")
        .and_then(Value::as_object)
        .and_then(|properties| properties.get(&key))
        .or_else(|| node.properties.get(&key))?;
    match property {
        Value::Array(values) => usize::try_from(index)
            .ok()
            .and_then(|index| values.get(index)),
        _ if index == 0 => Some(property),
        _ => None,
    }
}

fn animation_channel_is_boolean(node: &Node, path: &str, index: u32) -> bool {
    matches!(path, "hide_render" | "hide_viewport")
        || custom_property_component(node, path, index).is_some_and(Value::is_boolean)
}

fn animation_channel_has_integer_values(node: &Node, path: &str, index: u32) -> bool {
    matches!(path, "hide_render" | "hide_viewport")
        || custom_property_component(node, path, index).is_some_and(|value| {
            value.is_boolean() || value.as_i64().is_some() || value.as_u64().is_some()
        })
}

/// Blender's `FCURVE_INT_VALUES` rounds Boolean/integer samples before RNA writes.
pub(crate) fn round_discrete_fcurve_value(value: f64) -> f64 {
    f64::from(((value as f32) + 0.5).floor())
}

/// Evaluate a scalar non-transform Action channel through NLA and the active Action.
///
/// Each F-Curve sample is rounded for Boolean/integer channels (`FCURVE_INT_VALUES`), but the NLA
/// blend stays in float: `base_value` is the RNA default that Replace/Add/Combine strips start
/// from (`nla_blend_value`). The caller performs the final Boolean/integer RNA write.
/// The F-Curve sampler is shared with transform channels, so interpolation, extrapolation, and
/// modifier handling stay identical.
pub(crate) fn animated_scalar_channel(
    node: &Node,
    doc: &SceneDoc,
    path: &str,
    index: u32,
    base_value: f64,
    frame: f64,
) -> Result<Option<f64>> {
    if !frame.is_finite() || !base_value.is_finite() {
        return Err(PotError::new(
            ErrorCode::InvalidArgument,
            "animation frame and base channel value must be finite",
        ));
    }
    if !supports_animation_output(path, index, AnimationOutputKind::Action) {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            format!("animation scalar path `{path}` is not supported by this node"),
            json!({ "feature_id":format!("animation.fcurve_non_transform_channels.{path}") }),
        ));
    }
    let discrete = animation_channel_is_boolean(node, path, index);
    let integer_values = animation_channel_has_integer_values(node, path, index);
    let has_solo = node
        .nla_tracks
        .iter()
        .any(|track| track.solo && !track.mute);
    let nla_mode = node_uses_nla(node, has_solo);
    let mut in_nla_domain = false;
    if nla_mode && let Some(action_id) = &node.action {
        let action = doc.actions.get(action_id).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "node action reference does not exist",
                json!({"action":action_id}),
            )
        })?;
        in_nla_domain = action_has_scalar_curve(action, path, index);
    }
    for track in &node.nla_tracks {
        if track.mute || (has_solo && !track.solo) {
            continue;
        }
        for strip in &track.strips {
            let action = doc.actions.get(&strip.action).ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "NLA strip action reference does not exist",
                    json!({"action":strip.action,"strip":strip.id}),
                )
            })?;
            in_nla_domain |= action_has_scalar_curve(action, path, index);
        }
    }
    let mut value = base_value;
    let mut animated = in_nla_domain;
    for (track_index, track) in node.nla_tracks.iter().enumerate() {
        if track.mute || (has_solo && !track.solo) {
            continue;
        }
        let Some((strip_index, strip_frame)) = active_strip_for_track(track, frame) else {
            continue;
        };
        let strip = &track.strips[strip_index];
        if strip.mute {
            continue;
        }
        let (blend_in, blend_out) = auto_blend_lengths(node, track_index, strip_index);
        let influence = strip_influence(strip, strip_frame, blend_in, blend_out)?;
        if influence <= 0.0 {
            continue;
        }
        let Some(action_frame) = strip_action_frame(strip, strip_frame)? else {
            continue;
        };
        let action = doc.actions.get(&strip.action).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "NLA strip action reference does not exist",
                json!({"action":strip.action,"strip":strip.id}),
            )
        })?;
        if let Some(sampled) = sample_scalar_action(action, path, index, action_frame, discrete)? {
            let sampled = if integer_values {
                round_discrete_fcurve_value(sampled)
            } else {
                sampled
            };
            value = blend_scalar(value, sampled, base_value, strip.blend_type, influence);
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
        let (action_frame, blend, influence) =
            active_action_controls(node, action, frame, nla_mode, has_solo)?;
        if let Some(action_frame) = action_frame
            && let Some(sampled) =
                sample_scalar_action(action, path, index, action_frame, discrete)?
        {
            let sampled = if integer_values {
                round_discrete_fcurve_value(sampled)
            } else {
                sampled
            };
            value = blend_scalar(value, sampled, base_value, blend, influence);
            animated = true;
        }
    }
    if !value.is_finite() {
        return Err(PotError::with_details(
            ErrorCode::SceneInvalid,
            "animated scalar channel evaluated to a non-finite value",
            json!({"path":path,"frame":frame}),
        ));
    }
    Ok(animated.then_some(value))
}

fn sample_scalar_action(
    action: &Action,
    path: &str,
    index: u32,
    frame: f64,
    discrete: bool,
) -> Result<Option<f64>> {
    let mut sampled = None;
    for (curve_index, curve) in action.fcurves.iter().enumerate() {
        if u32::try_from(curve_index)
            .is_ok_and(|index| action.modifier_bindings.contains_key(&index))
            || curve.path != path
            || curve.index != index
        {
            continue;
        }
        let value = if discrete {
            sample_curve_discrete(curve, frame)?
        } else {
            sample_curve(curve, frame)?
        };
        if let Some(value) = value {
            sampled = Some(value);
        }
    }
    Ok(sampled)
}

fn action_has_scalar_curve(action: &Action, path: &str, index: u32) -> bool {
    action
        .fcurves
        .iter()
        .enumerate()
        .any(|(curve_index, curve)| {
            !u32::try_from(curve_index)
                .is_ok_and(|index| action.modifier_bindings.contains_key(&index))
                && curve.path == path
                && curve.index == index
        })
}

fn rotation_state_or_initial(
    rotation_state: Option<&RotationAccumulator>,
    base_rotation: [f64; 4],
    initial_kind: RotationChannelKind,
    rotation_mode: PoseRotationMode,
) -> Result<RotationAccumulator> {
    match rotation_state {
        Some(rotation_state) => Ok(*rotation_state),
        None => rotation_accumulator_from_quaternion(base_rotation, initial_kind, rotation_mode),
    }
}

fn sample_action_transform(
    node: &Node,
    doc: &SceneDoc,
    base: &Transform,
    rotation_state: Option<&RotationAccumulator>,
    initial_kind: RotationChannelKind,
    action: &crate::model::Action,
    action_id: &Id,
    frame: f64,
    rotation_mode_is_from_blender: bool,
    nla_context: bool,
) -> Result<(Transform, TransformChannels)> {
    let mut transform = base.clone();
    let mut channels = TransformChannels::default();
    let rotation_mode =
        PoseRotationMode::from_blender(&transform.rotation_mode).unwrap_or_default();
    let euler_channels_enabled =
        rotation_mode.euler_order().is_some() || !rotation_mode_is_from_blender;
    let mut euler = None;
    let mut axis_angle = None;
    let mut quaternion = None;
    let mut quaternion_modified = false;
    for (curve_index, curve) in action.fcurves.iter().enumerate() {
        if u32::try_from(curve_index)
            .is_ok_and(|index| action.modifier_bindings.contains_key(&index))
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
                if !rotation_mode_is_from_blender || rotation_mode == PoseRotationMode::Quaternion {
                    if quaternion.is_none() {
                        let sampled = if nla_context {
                            [0.0, 0.0, 0.0, 1.0]
                        } else {
                            let state = rotation_state_or_initial(
                                rotation_state,
                                base.rotation,
                                initial_kind,
                                rotation_mode,
                            )?;
                            rotation_accumulator_to_quaternion(state, rotation_mode)?
                        };
                        quaternion = Some(sampled);
                    }
                    let values = quaternion.as_mut().ok_or_else(|| {
                        PotError::new(
                            ErrorCode::InternalError,
                            "quaternion channel initialization failed",
                        )
                    })?;
                    // Potter component order (x, y, z, w); importers map Blender's indices to it.
                    values[component] = value;
                    channels.rotation_components[component] = true;
                    quaternion_modified = true;
                }
            }
            "transform.rotation_euler" => {
                let component = component_index(curve, 3)?;
                if euler_channels_enabled {
                    if euler.is_none() {
                        let state = rotation_state_or_initial(
                            rotation_state,
                            base.rotation,
                            initial_kind,
                            rotation_mode,
                        )?;
                        euler = Some(rotation_accumulator_to_euler(state, rotation_mode)?);
                    }
                    let angles = euler.as_mut().ok_or_else(|| {
                        PotError::new(
                            ErrorCode::InternalError,
                            "Euler channel initialization failed",
                        )
                    })?;
                    angles[component] = value;
                    channels.rotation_components[component] = true;
                }
            }
            "transform.rotation_axis_angle" => {
                let component = component_index(curve, 4)?;
                if !rotation_mode_is_from_blender || rotation_mode == PoseRotationMode::AxisAngle {
                    if axis_angle.is_none() {
                        let state = rotation_state_or_initial(
                            rotation_state,
                            base.rotation,
                            initial_kind,
                            rotation_mode,
                        )?;
                        axis_angle =
                            Some(rotation_accumulator_to_axis_angle(state, rotation_mode)?);
                    }
                    axis_angle.as_mut().ok_or_else(|| {
                        PotError::new(
                            ErrorCode::InternalError,
                            "axis-angle channel initialization failed",
                        )
                    })?[component] = value;
                    channels.rotation_components[component] = true;
                }
            }
            path => {
                if pose_curve_path(path).is_some() {
                    if supports_animation_output_for_node(
                        path,
                        curve.index,
                        AnimationOutputKind::Action,
                        node,
                        doc,
                    ) {
                        continue;
                    }
                    let feature_id = format!("animation.fcurve_non_transform_channels.{path}");
                    return Err(PotError::with_details(
                        ErrorCode::UnsupportedFeature,
                        format!("pose animation path `{path}` does not belong to this node"),
                        json!({ "feature_id": feature_id, "action": action_id }),
                    ));
                }
                if !supports_animation_output(path, curve.index, AnimationOutputKind::Action) {
                    let feature_id = format!("animation.fcurve_non_transform_channels.{path}");
                    return Err(PotError::with_details(
                        ErrorCode::UnsupportedFeature,
                        format!("animation curve path `{path}` is not supported"),
                        json!({ "feature_id": feature_id, "action": action_id }),
                    ));
                }
                let Some(component_count) =
                    non_transform_component_count(path, curve.index, node, doc)
                else {
                    // Blender ignores supported F-Curves whose RNA owner no longer matches.
                    continue;
                };
                component_index(curve, component_count)?;
            }
        }
    }
    if let Some([angle_x, angle_y, angle_z]) = euler {
        let angles = [angle_x, angle_y, angle_z];
        let rotation = quaternion_from_euler_angles(angles, rotation_mode);
        transform.rotation = [rotation.x, rotation.y, rotation.z, rotation.w];
        channels.rotation = true;
        channels.rotation_kind = RotationChannelKind::Euler;
        channels.rotation_values[..3].copy_from_slice(&angles);
    }
    if let Some(axis_angle) = axis_angle {
        let rotation = axis_angle_quaternion(axis_angle);
        transform.rotation = [rotation.x, rotation.y, rotation.z, rotation.w];
        channels.rotation = true;
        channels.rotation_kind = RotationChannelKind::AxisAngle;
        channels.rotation_values = axis_angle;
    }
    if quaternion_modified {
        let quaternion = quaternion.ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "quaternion channel initialization failed",
            )
        })?;
        transform.rotation = quaternion;
        channels.rotation = true;
        channels.rotation_kind = RotationChannelKind::Quaternion;
        channels.rotation_values = quaternion;
    }
    if channels.rotation {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AnimationOutputKind {
    Action,
    Driver,
}

pub(crate) fn animation_output_component_count(
    path: &str,
    kind: AnimationOutputKind,
) -> Option<usize> {
    match kind {
        AnimationOutputKind::Action => action_output_component_count(path),
        AnimationOutputKind::Driver => driver_output_component_count(path),
    }
}

/// The common path/index policy for Action F-Curves and driver outputs.
pub(crate) fn supports_animation_output(path: &str, index: u32, kind: AnimationOutputKind) -> bool {
    animation_output_component_count(path, kind)
        .is_some_and(|count| usize::try_from(index).is_ok_and(|index| index < count))
}

/// Checks a supported output against the data and concrete node owner.
pub(crate) fn supports_animation_output_for_node(
    path: &str,
    index: u32,
    kind: AnimationOutputKind,
    node: &Node,
    doc: &SceneDoc,
) -> bool {
    if !supports_animation_output(path, index, kind) {
        return false;
    }
    let data = node
        .data
        .as_ref()
        .and_then(|data_id| doc.data_blocks.get(data_id));
    match path {
        "camera.lens_mm"
        | "lens"
        | "data.lens"
        | "camera.sensor_width_mm"
        | "camera.sensor_height_mm"
        | "camera.ortho_scale"
        | "camera.clip_start"
        | "camera.clip_end"
        | "camera.shift_x"
        | "camera.shift_y"
        | "camera.focus_distance"
        | "camera.aperture_fstop" => {
            if data.and_then(|data| data.camera.as_ref()).is_none() {
                return false;
            }
        }
        "light.energy" | "energy" | "data.energy" => {
            if data.and_then(|data| data.light.as_ref()).is_none() {
                return false;
            }
        }
        "light.spot_size"
        | "light.spot_blend"
        | "light.shadow_soft_size"
        | "light.area_size"
        | "light.area_size_y"
        | "light.sun_angle"
        | "light.color" => {
            let Some(light) = data.and_then(|data| data.light.as_ref()) else {
                return false;
            };
            if (matches!(path, "light.spot_size" | "light.spot_blend")
                && light.light_type != LightType::Spot)
                || (matches!(path, "light.area_size" | "light.area_size_y")
                    && light.light_type != LightType::Area)
                || (path == "light.sun_angle" && light.light_type != LightType::Sun)
            {
                return false;
            }
        }
        "hide_render" | "hide_viewport" => {}
        _ if custom_property_path(path) => {
            let Some(key) = custom_property_name(path) else {
                return false;
            };
            let property = node
                .properties
                .get("_blender_custom_properties")
                .and_then(Value::as_object)
                .and_then(|properties| properties.get(&key))
                .or_else(|| node.properties.get(&key));
            if !property.is_some_and(|value| custom_property_value_supports_index(value, index)) {
                return false;
            }
        }
        "shape_keys.evaluation_time" | "shape_key.evaluation_time" | "key_blocks.eval_time" => {
            if data.and_then(|data| data.shape_keys.as_ref()).is_none() {
                return false;
            }
        }
        _ if path.strip_suffix(".value").is_some()
            && (path.starts_with("shape_key.")
                || path.starts_with("shape_keys.")
                || path.starts_with("key_blocks["))
            && !data
                .and_then(|data| data.shape_keys.as_ref())
                .is_some_and(|shape_keys| {
                    crate::eval::rig::shape_key_id_from_path(shape_keys, path).is_some()
                }) =>
        {
            return false;
        }
        _ => {}
    }
    if pose_channel_suffix(path).is_some() {
        let Some(armature) = data.and_then(|data| data.armature.as_ref()) else {
            return false;
        };
        let Some(bone_id) = pose_bone_id_from_path(path) else {
            return false;
        };
        if !armature.bones.contains_key(&bone_id) {
            return false;
        }
    }
    true
}

pub(crate) fn custom_property_path(path: &str) -> bool {
    path.starts_with("[\"") && path.ends_with("\"]")
}

pub(crate) fn custom_property_name(path: &str) -> Option<String> {
    serde_json::from_str(path.get(1..path.len().checked_sub(1)?)?).ok()
}

pub(crate) fn custom_property_value_supports_index(property: &Value, index: u32) -> bool {
    let Ok(index) = usize::try_from(index) else {
        return false;
    };
    match property {
        Value::Bool(_) | Value::Number(_) => index == 0,
        Value::Array(values) => {
            let Some(first) = values.first() else {
                return false;
            };
            let homogeneous_scalar = if first.is_boolean() {
                values.iter().all(Value::is_boolean)
            } else if first.is_number() {
                values.iter().all(Value::is_number)
            } else {
                false
            };
            homogeneous_scalar && index < values.len()
        }
        _ => false,
    }
}
/// Returns whether this node has an active camera-lens Action, NLA, or driver channel.
pub(crate) fn node_has_camera_lens_animation(node: &Node, doc: &SceneDoc) -> bool {
    node_has_camera_channel_animation(node, doc, &["camera.lens_mm", "lens", "data.lens"])
}

/// Returns whether UV projection reads an animated camera parameter.
pub(crate) fn node_has_camera_projection_animation(node: &Node, doc: &SceneDoc) -> bool {
    node_has_camera_channel_animation(
        node,
        doc,
        &[
            "camera.sensor_width_mm",
            "camera.sensor_height_mm",
            "camera.ortho_scale",
            "camera.clip_start",
            "camera.clip_end",
            "camera.shift_x",
            "camera.shift_y",
        ],
    )
}

fn node_has_camera_channel_animation(node: &Node, doc: &SceneDoc, paths: &[&str]) -> bool {
    let has_channel = |path: &str, index: u32, kind| {
        paths.contains(&path) && supports_animation_output_for_node(path, index, kind, node, doc)
    };
    let action_has_channel = |action: &Action| {
        action
            .fcurves
            .iter()
            .any(|curve| has_channel(&curve.path, curve.index, AnimationOutputKind::Action))
    };

    if node
        .action
        .as_ref()
        .and_then(|action_id| doc.actions.get(action_id))
        .is_some_and(action_has_channel)
    {
        return true;
    }

    let has_solo = node
        .nla_tracks
        .iter()
        .any(|track| track.solo && !track.mute);
    if node
        .nla_tracks
        .iter()
        .filter(|track| !track.mute && (!has_solo || track.solo))
        .flat_map(|track| track.strips.iter())
        .filter_map(|strip| doc.actions.get(&strip.action))
        .any(action_has_channel)
    {
        return true;
    }

    node.drivers
        .iter()
        .any(|driver| has_channel(&driver.path, driver.index, AnimationOutputKind::Driver))
}

fn action_output_component_count(path: &str) -> Option<usize> {
    match path {
        "transform.translation"
        | "transform.scale"
        | "transform.rotation_euler"
        | "light.color" => Some(3),
        "transform.rotation_quaternion"
        | "transform.rotation_axis_angle"
        | "transform.rotation" => Some(4),
        "camera.lens_mm"
        | "lens"
        | "data.lens"
        | "light.energy"
        | "energy"
        | "data.energy"
        | "camera.sensor_width_mm"
        | "camera.sensor_height_mm"
        | "camera.ortho_scale"
        | "camera.clip_start"
        | "camera.clip_end"
        | "camera.shift_x"
        | "camera.shift_y"
        | "camera.focus_distance"
        | "camera.aperture_fstop"
        | "light.spot_size"
        | "light.spot_blend"
        | "light.shadow_soft_size"
        | "light.area_size"
        | "light.area_size_y"
        | "light.sun_angle"
        | "hide_render"
        | "hide_viewport" => Some(1),
        _ if custom_property_path(path) => Some(usize::MAX),
        "shape_keys.evaluation_time" | "shape_key.evaluation_time" | "key_blocks.eval_time" => {
            Some(1)
        }
        _ if pose_channel_suffix(path).is_some_and(|channel| {
            matches!(
                channel,
                "location"
                    | "translation"
                    | "scale"
                    | "rotation"
                    | "rotation_quaternion"
                    | "rotation_euler"
                    | "rotation_axis_angle"
            )
        }) =>
        {
            pose_channel_suffix(path).map(|channel| {
                if matches!(
                    channel,
                    "rotation" | "rotation_quaternion" | "rotation_axis_angle"
                ) {
                    4
                } else {
                    3
                }
            })
        }
        _ if shape_key_value_path(path) => Some(1),
        _ => None,
    }
}

fn driver_output_component_count(path: &str) -> Option<usize> {
    match path {
        "location"
        | "translation"
        | "transform.location"
        | "transform.translation"
        | "scale"
        | "transform.scale"
        | "rotation_euler"
        | "transform.rotation_euler"
        | "light.color" => Some(3),
        "rotation"
        | "transform.rotation"
        | "rotation_quaternion"
        | "transform.rotation_quaternion" => Some(4),
        "camera.lens_mm"
        | "camera.sensor_width_mm"
        | "camera.sensor_height_mm"
        | "camera.ortho_scale"
        | "camera.clip_start"
        | "camera.clip_end"
        | "camera.shift_x"
        | "camera.shift_y"
        | "camera.focus_distance"
        | "camera.aperture_fstop"
        | "light.energy"
        | "light.spot_size"
        | "light.spot_blend"
        | "light.shadow_soft_size"
        | "light.area_size"
        | "light.area_size_y"
        | "light.sun_angle"
        | "hide_render"
        | "hide_viewport" => Some(1),
        _ if custom_property_path(path) => Some(usize::MAX),
        "shape_keys.evaluation_time" | "shape_key.evaluation_time" | "key_blocks.eval_time" => {
            Some(1)
        }
        _ if pose_channel_suffix(path).is_some_and(|channel| {
            matches!(
                channel,
                "location"
                    | "translation"
                    | "scale"
                    | "rotation"
                    | "bbone_curveinx"
                    | "bbone_curveinz"
                    | "bbone_curveoutx"
                    | "bbone_curveoutz"
                    | "bbone_rollin"
                    | "bbone_rollout"
                    | "bbone_scalein"
                    | "bbone_scaleout"
                    | "bbone_easein"
                    | "bbone_easeout"
            )
        }) =>
        {
            pose_channel_suffix(path).map(|channel| {
                if channel == "rotation" {
                    4
                } else if matches!(
                    channel,
                    "location" | "translation" | "scale" | "bbone_scalein" | "bbone_scaleout"
                ) {
                    3
                } else {
                    1
                }
            })
        }
        _ if shape_key_value_path(path) => Some(1),
        _ => None,
    }
}

fn pose_channel_suffix(path: &str) -> Option<&str> {
    if let Some(start) = path.find("pose.bones[") {
        let rest = &path[start + "pose.bones[".len()..];
        let end = rest.find(']')?;
        Some(rest[end + 1..].trim_start_matches('.'))
    } else {
        path.strip_prefix("pose.")
            .and_then(|rest| rest.split_once('.').map(|(_, channel)| channel))
    }
}
fn pose_bone_id_from_path(path: &str) -> Option<Id> {
    let key = if let Some(start) = path.find("pose.bones[") {
        let rest = &path[start + "pose.bones[".len()..];
        let end = rest.find(']')?;
        rest[..end].trim_matches(['"', '\''])
    } else {
        path.strip_prefix("pose.")?.split_once('.')?.0
    };
    Id::new(key).ok()
}

fn shape_key_value_path(path: &str) -> bool {
    path.strip_suffix(".value").is_some_and(|base| {
        (base.starts_with("shape_key.") || base.starts_with("shape_keys."))
            && base.split('.').count() == 2
            || base.starts_with("key_blocks[") && base.ends_with(']')
    })
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
        || modifier
            .params
            .get(parameter)
            .or_else(|| {
                crate::params::default_value(
                    crate::params::ParameterFamily::Modifier,
                    &modifier.modifier_type,
                    parameter,
                )
            })
            .and_then(serde_json::Value::as_f64)
            .is_none()
    {
        return Err(PotError::with_details(
            ErrorCode::SceneInvalid,
            "modifier animation binding must target a known scalar parameter",
            json!({"node":node_id,"modifier":modifier_id,"parameter":parameter}),
        ));
    }
    // Blender's NLA domain: a channel exists whenever an enabled track has a strip whose Action binds
    // it, whatever the frame (`nla_eval_domain_strips`, anim_sys.cc:3229). Such a channel starts from its
    // default and is written back even when no strip covers the frame (`nladata_flush_channels`).
    let has_solo = node
        .nla_tracks
        .iter()
        .any(|track| track.solo && !track.mute);
    let nla_mode = node_uses_nla(node, has_solo);
    let mut in_nla_domain = false;
    if nla_mode && let Some(action_id) = &node.action {
        let action = doc.actions.get(action_id).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "node action reference does not exist",
                json!({"action":action_id}),
            )
        })?;
        in_nla_domain = action.modifier_bindings.values().flatten().any(|binding| {
            &binding.node == node_id
                && &binding.modifier == modifier_id
                && binding.parameter == parameter
        });
    }
    for track in &node.nla_tracks {
        if track.mute || (has_solo && !track.solo) {
            continue;
        }
        for strip in &track.strips {
            let action = doc.actions.get(&strip.action).ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "NLA strip action reference does not exist",
                    json!({"action":strip.action,"strip":strip.id}),
                )
            })?;
            if action.modifier_bindings.values().flatten().any(|binding| {
                &binding.node == node_id
                    && &binding.modifier == modifier_id
                    && binding.parameter == parameter
            }) {
                in_nla_domain = true;
            }
        }
    }
    let nla_default = if in_nla_domain {
        crate::params::default_value(
            crate::params::ParameterFamily::Modifier,
            &modifier.modifier_type,
            parameter,
        )
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "NLA-animated modifier parameter has no Blender default",
                json!({"node":node_id,"modifier":modifier_id,"parameter":parameter}),
            )
        })?
    } else {
        base_value
    };
    let mut value = nla_default;
    let mut animated = in_nla_domain;
    for (track_index, track) in node.nla_tracks.iter().enumerate() {
        if track.mute || (has_solo && !track.solo) {
            continue;
        }
        let Some((strip_index, strip_frame)) = active_strip_for_track(track, frame) else {
            continue;
        };
        let strip = &track.strips[strip_index];
        if strip.mute {
            continue;
        }
        let (blend_in, blend_out) = auto_blend_lengths(node, track_index, strip_index);
        let influence = strip_influence(strip, strip_frame, blend_in, blend_out)?;
        if influence <= 0.0 {
            continue;
        }
        let Some(action_frame) = strip_action_frame(strip, strip_frame)? else {
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
        value = blend_scalar(value, sampled, nla_default, strip.blend_type, influence);
        if !value.is_finite() {
            return Err(PotError::with_details(
                ErrorCode::SceneInvalid,
                "modifier animation evaluated to a non-finite scalar",
                json!({"action":strip.action,"modifier":modifier_id,"parameter":parameter,"frame":frame}),
            ));
        }
        animated = true;
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
        let (action_frame, blend, influence) =
            active_action_controls(node, action, frame, nla_mode, has_solo)?;
        if let Some(action_frame) = action_frame
            && let Some(sampled) = sample_modifier_scalar_action(
                node_id,
                doc,
                action_id,
                action,
                modifier_id,
                parameter,
                assigned_slot,
                action_frame,
            )?
        {
            value = blend_scalar(value, sampled, nla_default, blend, influence);
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
        || modifier
            .params
            .get(&binding.parameter)
            .or_else(|| {
                crate::params::default_value(
                    crate::params::ParameterFamily::Modifier,
                    &modifier.modifier_type,
                    &binding.parameter,
                )
            })
            .and_then(serde_json::Value::as_f64)
            .is_none()
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
                        && !action.active_slots.contains_key(&binding.node)
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
enum RotationChannelKind {
    #[default]
    Quaternion,
    Euler,
    AxisAngle,
}

#[derive(Clone, Copy, Default)]
struct TransformChannels {
    translation: [bool; 3],
    scale: [bool; 3],
    rotation: bool,
    rotation_components: [bool; 4],
    rotation_kind: RotationChannelKind,
    rotation_values: [f64; 4],
}

#[derive(Clone, Copy)]
enum RotationAccumulator {
    Quaternion([f64; 4]),
    Euler([f64; 3]),
    AxisAngle([f64; 4]),
}

fn normalize_quaternion_raw(rotation: [f64; 4]) -> Result<[f64; 4]> {
    if rotation.iter().any(|value| !value.is_finite()) {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "rotation quaternion must be finite",
        ));
    }
    let largest = rotation
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if largest == 0.0 {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "rotation quaternion must be non-zero",
        ));
    }
    let scaled = rotation.map(|value| value / largest);
    let length = scaled.iter().map(|value| value * value).sum::<f64>().sqrt();
    Ok(scaled.map(|value| value / length))
}

fn rotation_accumulator_from_quaternion(
    rotation: [f64; 4],
    kind: RotationChannelKind,
    rotation_mode: PoseRotationMode,
) -> Result<RotationAccumulator> {
    let rotation = normalize_quaternion_raw(rotation)?;
    Ok(match kind {
        RotationChannelKind::Quaternion => RotationAccumulator::Quaternion(rotation),
        RotationChannelKind::Euler => RotationAccumulator::Euler(euler_angles_from_quaternion(
            DQuat::from_xyzw(rotation[0], rotation[1], rotation[2], rotation[3]),
            rotation_mode,
        )),
        RotationChannelKind::AxisAngle => {
            RotationAccumulator::AxisAngle(axis_angle_from_rotation(rotation)?)
        }
    })
}

fn rotation_accumulator_to_quaternion(
    accumulator: RotationAccumulator,
    rotation_mode: PoseRotationMode,
) -> Result<[f64; 4]> {
    let rotation = match accumulator {
        RotationAccumulator::Quaternion(rotation) => rotation,
        RotationAccumulator::Euler(angles) => {
            let rotation = quaternion_from_euler_angles(angles, rotation_mode);
            [rotation.x, rotation.y, rotation.z, rotation.w]
        }
        RotationAccumulator::AxisAngle(axis_angle) => {
            let rotation = axis_angle_quaternion(axis_angle);
            [rotation.x, rotation.y, rotation.z, rotation.w]
        }
    };
    normalize_quaternion_raw(rotation)
}

fn rotation_accumulator_to_euler(
    accumulator: RotationAccumulator,
    rotation_mode: PoseRotationMode,
) -> Result<[f64; 3]> {
    if let RotationAccumulator::Euler(angles) = accumulator {
        return Ok(angles);
    }
    let rotation = rotation_accumulator_to_quaternion(accumulator, rotation_mode)?;
    Ok(euler_angles_from_quaternion(
        DQuat::from_xyzw(rotation[0], rotation[1], rotation[2], rotation[3]),
        rotation_mode,
    ))
}

fn rotation_accumulator_to_axis_angle(
    accumulator: RotationAccumulator,
    rotation_mode: PoseRotationMode,
) -> Result<[f64; 4]> {
    if let RotationAccumulator::AxisAngle(axis_angle) = accumulator {
        return Ok(axis_angle);
    }
    axis_angle_from_rotation(rotation_accumulator_to_quaternion(
        accumulator,
        rotation_mode,
    )?)
}

fn ensure_rotation_accumulator_kind(
    accumulator: &mut RotationAccumulator,
    kind: RotationChannelKind,
    rotation_mode: PoseRotationMode,
) -> Result<()> {
    let matches = matches!(
        (&*accumulator, kind),
        (
            RotationAccumulator::Quaternion(_),
            RotationChannelKind::Quaternion
        ) | (RotationAccumulator::Euler(_), RotationChannelKind::Euler)
            | (
                RotationAccumulator::AxisAngle(_),
                RotationChannelKind::AxisAngle
            )
    );
    if !matches {
        let quaternion = rotation_accumulator_to_quaternion(*accumulator, rotation_mode)?;
        *accumulator = rotation_accumulator_from_quaternion(quaternion, kind, rotation_mode)?;
    }
    Ok(())
}

fn reset_rotation_accumulator(
    accumulator: &mut RotationAccumulator,
    kind: RotationChannelKind,
    components: [bool; 4],
    rotation_mode: PoseRotationMode,
) -> Result<()> {
    ensure_rotation_accumulator_kind(accumulator, kind, rotation_mode)?;
    match accumulator {
        RotationAccumulator::Quaternion(rotation) => {
            for (component, animated) in components.iter().enumerate() {
                if *animated {
                    rotation[component] = if component == 3 { 1.0 } else { 0.0 };
                }
            }
        }
        RotationAccumulator::Euler(angles) => {
            for (component, animated) in components[..3].iter().enumerate() {
                if *animated {
                    angles[component] = 0.0;
                }
            }
        }
        RotationAccumulator::AxisAngle(axis_angle) => {
            for (component, animated) in components.iter().enumerate() {
                if *animated {
                    axis_angle[component] = 0.0;
                }
            }
        }
    }
    Ok(())
}

fn raw_quaternion_power(rotation: [f64; 4], influence: f64) -> Result<[f64; 4]> {
    let rotation = normalize_quaternion_raw(rotation)?;
    let half_angle = influence * rotation[3].clamp(-1.0, 1.0).acos();
    let sine = half_angle.sin();
    let vector_length =
        (rotation[0] * rotation[0] + rotation[1] * rotation[1] + rotation[2] * rotation[2]).sqrt();
    let vector = if vector_length > f64::EPSILON {
        [
            rotation[0] / vector_length * sine,
            rotation[1] / vector_length * sine,
            rotation[2] / vector_length * sine,
        ]
    } else {
        [0.0; 3]
    };
    Ok([vector[0], vector[1], vector[2], half_angle.cos()])
}

fn blend_rotation_accumulator(
    accumulator: &mut RotationAccumulator,
    kind: RotationChannelKind,
    target_values: [f64; 4],
    components: [bool; 4],
    rotation_mode: PoseRotationMode,
    blend: NlaBlendType,
    influence: f64,
) -> Result<()> {
    ensure_rotation_accumulator_kind(accumulator, kind, rotation_mode)?;
    match accumulator {
        RotationAccumulator::Quaternion(lower) => {
            if blend == NlaBlendType::Combine {
                let lower_rotation = normalize_quaternion_raw(*lower)?;
                let power = raw_quaternion_power(target_values, influence)?;
                let lower_quaternion = DQuat::from_xyzw(
                    lower_rotation[0],
                    lower_rotation[1],
                    lower_rotation[2],
                    lower_rotation[3],
                );
                let power_quaternion = DQuat::from_xyzw(power[0], power[1], power[2], power[3]);
                let result = lower_quaternion * power_quaternion;
                *lower = normalize_quaternion_raw([result.x, result.y, result.z, result.w])?;
            } else {
                for component in 0..4 {
                    if components[component] {
                        lower[component] = blend_scalar(
                            lower[component],
                            target_values[component],
                            if component == 3 { 1.0 } else { 0.0 },
                            blend,
                            influence,
                        );
                    }
                }
            }
        }
        RotationAccumulator::Euler(lower) => {
            for component in 0..3 {
                if components[component] {
                    lower[component] = blend_scalar(
                        lower[component],
                        target_values[component],
                        0.0,
                        blend,
                        influence,
                    );
                }
            }
        }
        RotationAccumulator::AxisAngle(lower) => {
            for component in 0..4 {
                if components[component] {
                    lower[component] = blend_scalar(
                        lower[component],
                        target_values[component],
                        0.0,
                        blend,
                        influence,
                    );
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn euler_angles_from_quaternion(
    rotation: DQuat,
    rotation_mode: PoseRotationMode,
) -> [f64; 3] {
    let (first, second, third) =
        rotation.to_euler(rotation_mode.euler_order().unwrap_or(EulerRot::XYZEx));
    let mut angles = [0.0; 3];
    for (component, angle) in rotation_mode
        .euler_component_order()
        .into_iter()
        .zip([first, second, third])
    {
        angles[component] = angle;
    }
    angles
}

pub(crate) fn quaternion_from_euler_angles(
    angles: [f64; 3],
    rotation_mode: PoseRotationMode,
) -> DQuat {
    let component_order = rotation_mode.euler_component_order();
    DQuat::from_euler(
        rotation_mode.euler_order().unwrap_or(EulerRot::XYZEx),
        angles[component_order[0]],
        angles[component_order[1]],
        angles[component_order[2]],
    )
}

fn axis_angle_from_rotation(rotation: [f64; 4]) -> Result<[f64; 4]> {
    let normalized = normalize_rotation(rotation)
        .map_err(|error| PotError::new(ErrorCode::SceneInvalid, error.message))?;
    let quaternion = DQuat::from_xyzw(normalized[0], normalized[1], normalized[2], normalized[3]);
    let (axis, angle) = quaternion.to_axis_angle();
    if angle.abs() <= f64::EPSILON {
        Ok([0.0, 0.0, 1.0, 0.0])
    } else {
        Ok([angle, axis.x, axis.y, axis.z])
    }
}

fn axis_angle_quaternion(axis_angle: [f64; 4]) -> DQuat {
    let axis = glam::DVec3::new(axis_angle[1], axis_angle[2], axis_angle[3]).normalize_or_zero();
    if axis == glam::DVec3::ZERO || axis_angle[0] == 0.0 {
        DQuat::IDENTITY
    } else {
        DQuat::from_axis_angle(axis, axis_angle[0])
    }
}

fn active_strip_for_track(track: &NlaTrack, frame: f64) -> Option<(usize, f64)> {
    for (index, strip) in track.strips.iter().enumerate() {
        if frame < strip.frame_start {
            if index == 0 {
                return (strip.extrapolation == NlaExtrapolation::Hold)
                    .then_some((index, strip.frame_start));
            }
            let previous = &track.strips[index - 1];
            return (previous.extrapolation != NlaExtrapolation::Nothing)
                .then_some((index - 1, previous.frame_end));
        }
        if frame <= strip.frame_end {
            return Some((index, frame));
        }
    }
    let index = track.strips.len().checked_sub(1)?;
    let strip = &track.strips[index];
    (strip.extrapolation != NlaExtrapolation::Nothing).then_some((index, strip.frame_end))
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
        strip.strip_time,
    ];
    let timeline_span = strip.frame_end - strip.frame_start;
    let action_span = strip.action_frame_end - strip.action_frame_start;
    if values.iter().any(|value| !value.is_finite())
        || timeline_span <= 0.0
        || action_span <= 0.0
        || strip.scale < 0.0
        || strip.repeat < 0.0
        || !(0.0..=1.0).contains(&strip.influence)
    {
        return Err(PotError::with_details(
            ErrorCode::SceneInvalid,
            "NLA strip settings are invalid",
            json!({ "strip": strip.id }),
        ));
    }
    let frame = if frame < strip.frame_start {
        match strip.extrapolation {
            NlaExtrapolation::Hold => strip.frame_start,
            NlaExtrapolation::HoldForward | NlaExtrapolation::Nothing => return Ok(None),
        }
    } else if frame > strip.frame_end {
        match strip.extrapolation {
            NlaExtrapolation::Hold | NlaExtrapolation::HoldForward => strip.frame_end,
            NlaExtrapolation::Nothing => return Ok(None),
        }
    } else {
        frame
    };
    let action_frame = if strip.use_animated_time {
        let sampled_time = strip
            .strip_time_curve
            .as_ref()
            .map(|curve| sample_curve(curve, frame))
            .transpose()?
            .flatten()
            .unwrap_or(strip.strip_time);
        if strip.use_animated_time_cyclic {
            (sampled_time - strip.action_frame_start) % action_span
        } else {
            sampled_time
        }
    } else {
        let scale = if strip.scale == 0.0 {
            1.0
        } else {
            strip.scale.abs()
        };
        let repeat = if strip.repeat == 0.0 {
            1.0
        } else {
            strip.repeat
        };
        let elapsed = frame - strip.frame_start;
        let at_end =
            same_nla_frame(frame, strip.frame_end) && (repeat as f32).fract().abs() < f32::EPSILON;
        if at_end {
            if strip.reverse {
                strip.action_frame_start
            } else {
                strip.action_frame_end
            }
        } else {
            let offset = elapsed.rem_euclid(action_span * scale) / scale;
            if strip.reverse {
                strip.action_frame_end - offset
            } else {
                strip.action_frame_start + offset
            }
        }
    };
    if action_frame.is_finite() {
        Ok(Some(action_frame))
    } else {
        Err(PotError::with_details(
            ErrorCode::SceneInvalid,
            "NLA strip maps to a non-finite action frame",
            json!({ "strip": strip.id }),
        ))
    }
}

fn same_nla_frame(left: f64, right: f64) -> bool {
    ((left as f32) - (right as f32)).abs() < f32::EPSILON
}
fn endpoint_overlaps(target: &NlaStrip, strips: &[NlaStrip]) -> (Option<f64>, Option<f64>) {
    let mut start = None;
    let mut end = None;
    for (index, strip) in strips.iter().enumerate() {
        if strip.frame_start <= target.frame_start && strip.frame_end >= target.frame_end {
            return (None, None);
        }
        if strip.frame_end < target.frame_start {
            continue;
        }
        if strip.frame_start > target.frame_end {
            break;
        }
        let next_is_gap = strips
            .get(index + 1)
            .is_none_or(|next| !same_nla_frame(next.frame_start, strip.frame_end));
        if next_is_gap && strip.frame_end > target.frame_start && strip.frame_end < target.frame_end
        {
            start = Some(strip.frame_end);
        }
        let previous_is_gap =
            index == 0 || !same_nla_frame(strips[index - 1].frame_end, strip.frame_start);
        if previous_is_gap
            && strip.frame_start < target.frame_end
            && strip.frame_start > target.frame_start
        {
            end = Some(strip.frame_start);
        }
    }
    (start, end)
}

fn auto_blend_lengths(node: &Node, track_index: usize, strip_index: usize) -> (f64, f64) {
    let track = &node.nla_tracks[track_index];
    let strip = &track.strips[strip_index];
    if !strip.use_auto_blend || node.nla_tracks.len() == 1 {
        return (strip.blend_in, strip.blend_out);
    }
    let previous_track = track_index
        .checked_sub(1)
        .and_then(|index| node.nla_tracks.get(index));
    let next_track = node.nla_tracks.get(track_index + 1);
    let (previous_start, previous_end) = previous_track.map_or((None, None), |other| {
        endpoint_overlaps(strip, &other.strips)
    });
    let (next_start, next_end) = next_track.map_or((None, None), |other| {
        endpoint_overlaps(strip, &other.strips)
    });
    let previous_strip = strip_index
        .checked_sub(1)
        .and_then(|index| track.strips.get(index));
    let next_strip = track.strips.get(strip_index + 1);
    let blend_in = if (previous_start.is_some() || next_start.is_some())
        && previous_strip
            .is_none_or(|previous| !same_nla_frame(previous.frame_end, strip.frame_start))
    {
        previous_start
            .or(next_start)
            .map_or(0.0, |overlap| overlap - strip.frame_start)
    } else {
        0.0
    };
    let blend_out = if (previous_end.is_some() || next_end.is_some())
        && next_strip.is_none_or(|next| !same_nla_frame(next.frame_start, strip.frame_end))
    {
        previous_end
            .or(next_end)
            .map_or(0.0, |overlap| strip.frame_end - overlap)
    } else {
        0.0
    };
    let strip_length = (strip.frame_end - strip.frame_start).max(0.0);
    let blend_in = blend_in.clamp(0.0, (strip_length - blend_out).max(0.0));
    let blend_out = blend_out.clamp(0.0, (strip_length - blend_in).max(0.0));
    (blend_in, blend_out)
}

fn strip_influence(strip: &NlaStrip, frame: f64, blend_in: f64, blend_out: f64) -> Result<f64> {
    let influence = if strip.use_animated_influence {
        strip
            .influence_curve
            .as_ref()
            .map(|curve| sample_curve(curve, frame))
            .transpose()?
            .flatten()
            .unwrap_or(strip.influence)
    } else {
        let blend_in = blend_in.abs();
        let blend_out = blend_out.abs();
        if blend_in != 0.0 && frame <= strip.frame_start + blend_in {
            (frame - strip.frame_start).abs() / blend_in
        } else if blend_out != 0.0 && frame >= strip.frame_end - blend_out {
            (strip.frame_end - frame).abs() / blend_out
        } else {
            1.0
        }
    };
    if influence.is_finite() {
        Ok(influence.clamp(0.0, 1.0))
    } else {
        Err(PotError::with_details(
            ErrorCode::SceneInvalid,
            "NLA strip influence is not finite",
            json!({ "strip": strip.id }),
        ))
    }
}

fn node_uses_nla(node: &Node, has_solo: bool) -> bool {
    has_solo
        || node
            .nla_tracks
            .iter()
            .any(|track| !track.mute && !track.strips.is_empty())
}
fn action_key_range_with_modifiers(action: &Action) -> Result<(f64, f64)> {
    const MIN_FRAME: f64 = -1_048_574.0;
    const MAX_FRAME: f64 = 1_048_574.0;
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    let mut found_range = false;
    for curve in &action.fcurves {
        for key in &curve.keyframes {
            if !key.frame.is_finite() {
                return Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    "active action keyframe frame must be finite",
                ));
            }
            minimum = minimum.min(key.frame);
            maximum = maximum.max(key.frame);
            found_range = true;
        }
        if let Some(modifier) = curve.modifiers.last() {
            let properties = &modifier.properties;
            found_range = true;
            match modifier.modifier_type.as_str() {
                "LIMITS" => {
                    if properties.get("use_min_x").and_then(Value::as_bool) == Some(true) {
                        minimum = minimum.min(
                            properties
                                .get("min_x")
                                .and_then(Value::as_f64)
                                .unwrap_or(0.0),
                        );
                    }
                    if properties.get("use_max_x").and_then(Value::as_bool) == Some(true) {
                        maximum = maximum.max(
                            properties
                                .get("max_x")
                                .and_then(Value::as_f64)
                                .unwrap_or(0.0),
                        );
                    }
                }
                "CYCLES" => {
                    if properties
                        .get("before_mode")
                        .and_then(Value::as_str)
                        .unwrap_or("REPEAT")
                        != "NONE"
                    {
                        minimum = minimum.min(MIN_FRAME);
                    }
                    if properties
                        .get("after_mode")
                        .and_then(Value::as_str)
                        .unwrap_or("REPEAT")
                        != "NONE"
                    {
                        maximum = maximum.max(MAX_FRAME);
                    }
                }
                _ => {
                    minimum = minimum.min(MIN_FRAME);
                    maximum = maximum.max(MAX_FRAME);
                }
            }
        }
    }
    if !found_range {
        minimum = 0.0;
        maximum = 0.0;
    }
    minimum = minimum.max(MIN_FRAME);
    maximum = maximum.min(MAX_FRAME);
    if maximum <= minimum {
        maximum = minimum + 1.0;
    }
    Ok((minimum, maximum))
}

fn active_action_controls(
    node: &Node,
    action: &Action,
    frame: f64,
    nla_mode: bool,
    has_solo: bool,
) -> Result<(Option<f64>, NlaBlendType, f64)> {
    if has_solo {
        return Ok((None, node.action_blend_type, node.action_influence));
    }
    if !nla_mode {
        return Ok((Some(frame), NlaBlendType::Replace, 1.0));
    }
    if !node.action_influence.is_finite() || !(0.0..=1.0).contains(&node.action_influence) {
        return Err(PotError::with_details(
            ErrorCode::SceneInvalid,
            "active action influence must be between zero and one",
            json!({ "influence": node.action_influence }),
        ));
    }
    if node.action_influence <= 0.0 {
        return Ok((None, node.action_blend_type, node.action_influence));
    }
    let (start, end) = action_key_range_with_modifiers(action)?;
    let mapped = match node.action_extrapolation {
        NlaExtrapolation::Hold => Some(frame),
        NlaExtrapolation::HoldForward => (frame >= start).then_some(frame),
        NlaExtrapolation::Nothing => (frame >= start && frame <= end).then_some(frame),
    };
    Ok((mapped, node.action_blend_type, node.action_influence))
}

/// Blender's `nla_blend_value` and `nla_combine_value` for scalar channels.
fn blend_scalar(
    current: f64,
    target: f64,
    default: f64,
    blend: NlaBlendType,
    influence: f64,
) -> f64 {
    match blend {
        NlaBlendType::Replace => current * (1.0 - influence) + target * influence,
        NlaBlendType::Add => current + target * influence,
        NlaBlendType::Subtract => current - target * influence,
        NlaBlendType::Multiply => current * (1.0 + (target - 1.0) * influence),
        NlaBlendType::Combine => current + (target - default) * influence,
    }
}

fn blend_transform(
    current: &mut Transform,
    rotation_state: &mut Option<RotationAccumulator>,
    initial_kind: RotationChannelKind,
    rotation_mode: PoseRotationMode,
    sampled: &Transform,
    channels: TransformChannels,
    blend: NlaBlendType,
    influence: f64,
) -> Result<()> {
    for component in 0..3 {
        if channels.translation[component] {
            current.translation[component] = blend_scalar(
                current.translation[component],
                sampled.translation[component],
                0.0,
                blend,
                influence,
            );
        }
        if channels.scale[component] {
            let value = sampled.scale[component];
            current.scale[component] = if blend == NlaBlendType::Combine {
                current.scale[component] * value.powf(influence)
            } else {
                blend_scalar(current.scale[component], value, 1.0, blend, influence)
            };
        }
    }
    if channels.rotation {
        if rotation_state.is_none() {
            *rotation_state = Some(rotation_accumulator_from_quaternion(
                current.rotation,
                initial_kind,
                rotation_mode,
            )?);
        }
        let state = rotation_state.as_mut().ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "rotation channel has no accumulator",
            )
        })?;
        blend_rotation_accumulator(
            state,
            channels.rotation_kind,
            channels.rotation_values,
            channels.rotation_components,
            rotation_mode,
            blend,
            influence,
        )?;
        current.rotation = rotation_accumulator_to_quaternion(*state, rotation_mode)?;
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
    let nla_mode = node_uses_nla(node, has_solo);
    let domain = pose_nla_domain(node, doc, nla_mode, has_solo)?;
    let mut rotation_states = BTreeMap::new();
    for (bone_id, mask) in domain {
        let bone = pose.entry(bone_id.clone()).or_default();
        for component in 0..3 {
            if mask.translation[component] {
                bone.translation[component] = 0.0;
            }
            if mask.scale[component] {
                bone.scale[component] = 1.0;
            }
        }
        if mask.rotation.iter().any(|animated| *animated) {
            let rotation_mode = bone.rotation_mode.unwrap_or_default();
            let mut state = rotation_accumulator_from_quaternion(
                bone.rotation,
                mask.rotation_kind,
                rotation_mode,
            )?;
            reset_rotation_accumulator(
                &mut state,
                mask.rotation_kind,
                mask.rotation,
                rotation_mode,
            )?;
            bone.rotation = rotation_accumulator_to_quaternion(state, rotation_mode)?;
            rotation_states.insert(bone_id, state);
        }
    }
    for (track_index, track) in node.nla_tracks.iter().enumerate() {
        if track.mute || (has_solo && !track.solo) {
            continue;
        }
        let Some((strip_index, strip_frame)) = active_strip_for_track(track, frame) else {
            continue;
        };
        let strip = &track.strips[strip_index];
        if strip.mute {
            continue;
        }
        let (blend_in, blend_out) = auto_blend_lengths(node, track_index, strip_index);
        let influence = strip_influence(strip, strip_frame, blend_in, blend_out)?;
        if influence <= 0.0 {
            continue;
        }
        let Some(action_frame) = strip_action_frame(strip, strip_frame)? else {
            continue;
        };
        let action = doc.actions.get(&strip.action).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "NLA strip action reference does not exist",
                json!({ "action": strip.action, "strip": strip.id }),
            )
        })?;
        let (sampled, channels) = sample_action_pose(
            node,
            doc,
            &pose,
            &rotation_states,
            action,
            &strip.action,
            action_frame,
            true,
        )?;
        blend_pose(
            &mut pose,
            &mut rotation_states,
            &sampled,
            channels,
            strip.blend_type,
            influence,
        )?;
    }
    if let Some(action_id) = &node.action {
        let action = doc.actions.get(action_id).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "node action reference does not exist",
                json!({ "action": action_id }),
            )
        })?;
        let (action_frame, blend, influence) =
            active_action_controls(node, action, frame, nla_mode, has_solo)?;
        if let Some(action_frame) = action_frame {
            let (sampled, channels) = sample_action_pose(
                node,
                doc,
                &pose,
                &rotation_states,
                action,
                action_id,
                action_frame,
                nla_mode,
            )?;
            blend_pose(
                &mut pose,
                &mut rotation_states,
                &sampled,
                channels,
                blend,
                influence,
            )?;
        }
    }
    Ok(pose)
}

#[derive(Clone, Copy)]
enum PoseChannel {
    Translation,
    Rotation,
    RotationEuler,
    RotationAxisAngle,
    Scale,
}

#[derive(Clone, Copy, Default)]
struct PoseChannels {
    translation: [bool; 3],
    rotation: [bool; 4],
    scale: [bool; 3],
    rotation_kind: RotationChannelKind,
    rotation_values: [f64; 4],
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
        "rotation" | "rotation_quaternion" => PoseChannel::Rotation,
        "rotation_euler" => PoseChannel::RotationEuler,
        "rotation_axis_angle" => PoseChannel::RotationAxisAngle,
        "scale" => PoseChannel::Scale,
        _ => return None,
    };
    Some((Id::new(bone_key).ok()?, channel))
}

fn pose_euler_order(rotation_mode: Option<PoseRotationMode>) -> Option<EulerRot> {
    match rotation_mode {
        None => Some(EulerRot::XYZEx),
        Some(mode) => mode.euler_order(),
    }
}

fn pose_nla_domain(
    node: &Node,
    doc: &SceneDoc,
    nla_mode: bool,
    has_solo: bool,
) -> Result<BTreeMap<Id, PoseChannels>> {
    let mut channels = BTreeMap::<Id, PoseChannels>::new();
    if nla_mode && let Some(action_id) = &node.action {
        let action = doc.actions.get(action_id).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "node action reference does not exist",
                json!({"action":action_id}),
            )
        })?;
        add_pose_action_domain(node, doc, action, &mut channels)?;
    }
    for track in &node.nla_tracks {
        if track.mute || (has_solo && !track.solo) {
            continue;
        }
        for strip in &track.strips {
            let action = doc.actions.get(&strip.action).ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "NLA strip action reference does not exist",
                    json!({"action":strip.action,"strip":strip.id}),
                )
            })?;
            add_pose_action_domain(node, doc, action, &mut channels)?;
        }
    }
    Ok(channels)
}

fn add_pose_action_domain(
    node: &Node,
    doc: &SceneDoc,
    action: &Action,
    channels: &mut BTreeMap<Id, PoseChannels>,
) -> Result<()> {
    for (curve_index, curve) in action.fcurves.iter().enumerate() {
        if u32::try_from(curve_index)
            .is_ok_and(|index| action.modifier_bindings.contains_key(&index))
        {
            continue;
        }
        let Some((bone_id, channel)) = pose_curve_path(&curve.path) else {
            continue;
        };
        if !supports_animation_output_for_node(
            &curve.path,
            curve.index,
            AnimationOutputKind::Action,
            node,
            doc,
        ) {
            continue;
        }
        let component_count = match channel {
            PoseChannel::Translation | PoseChannel::Scale | PoseChannel::RotationEuler => 3,
            PoseChannel::Rotation | PoseChannel::RotationAxisAngle => 4,
        };
        let component = component_index(curve, component_count)?;
        let rotation_mode = node.pose.get(&bone_id).and_then(|bone| bone.rotation_mode);
        let mask = channels.entry(bone_id).or_default();
        match channel {
            PoseChannel::Translation => mask.translation[component] = true,
            PoseChannel::Scale => mask.scale[component] = true,
            PoseChannel::Rotation
                if matches!(rotation_mode, None | Some(PoseRotationMode::Quaternion)) =>
            {
                mask.rotation = [true; 4];
                mask.rotation_kind = RotationChannelKind::Quaternion;
            }
            PoseChannel::RotationEuler if pose_euler_order(rotation_mode).is_some() => {
                mask.rotation[component] = true;
                mask.rotation_kind = RotationChannelKind::Euler;
            }
            PoseChannel::RotationAxisAngle
                if matches!(rotation_mode, None | Some(PoseRotationMode::AxisAngle)) =>
            {
                mask.rotation[component] = true;
                mask.rotation_kind = RotationChannelKind::AxisAngle;
            }
            _ => {}
        }
    }
    Ok(())
}

fn sample_action_pose(
    node: &Node,
    doc: &SceneDoc,
    base: &BTreeMap<Id, PoseBone>,
    rotation_states: &BTreeMap<Id, RotationAccumulator>,
    action: &crate::model::Action,
    action_id: &Id,
    frame: f64,
    nla_context: bool,
) -> Result<(BTreeMap<Id, PoseBone>, BTreeMap<Id, PoseChannels>)> {
    let mut pose = base.clone();
    let mut channels = BTreeMap::<Id, PoseChannels>::new();
    let mut euler_rotations = BTreeMap::<Id, [f64; 3]>::new();
    let mut axis_angle_rotations = BTreeMap::<Id, [f64; 4]>::new();
    for curve in &action.fcurves {
        let Some((bone_id, channel)) = pose_curve_path(&curve.path) else {
            continue;
        };
        if !supports_animation_output_for_node(
            &curve.path,
            curve.index,
            AnimationOutputKind::Action,
            node,
            doc,
        ) {
            let feature_id = format!("animation.fcurve_non_transform_channels.{}", curve.path);
            return Err(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                format!(
                    "animation curve path `{}` is not supported by this pose bone",
                    curve.path
                ),
                json!({ "feature_id": feature_id, "action": action_id }),
            ));
        }
        let component_count = match channel {
            PoseChannel::Translation | PoseChannel::Scale | PoseChannel::RotationEuler => 3,
            PoseChannel::Rotation | PoseChannel::RotationAxisAngle => 4,
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
            PoseChannel::Rotation
                if matches!(
                    pose_bone.rotation_mode,
                    None | Some(PoseRotationMode::Quaternion)
                ) =>
            {
                if nla_context && mask.rotation.iter().all(|animated| !animated) {
                    pose_bone.rotation = [0.0, 0.0, 0.0, 1.0];
                    mask.rotation_values = [0.0, 0.0, 0.0, 1.0];
                }
                pose_bone.rotation[component] = value;
                mask.rotation[component] = true;
                mask.rotation_kind = RotationChannelKind::Quaternion;
                mask.rotation_values[component] = value;
            }
            PoseChannel::RotationEuler => {
                if pose_euler_order(pose_bone.rotation_mode).is_some() {
                    if !euler_rotations.contains_key(&bone_id) {
                        let rotation_mode = pose_bone.rotation_mode.unwrap_or_default();
                        let state = match rotation_states.get(&bone_id) {
                            Some(state) => *state,
                            None => rotation_accumulator_from_quaternion(
                                pose_bone.rotation,
                                RotationChannelKind::Euler,
                                rotation_mode,
                            )?,
                        };
                        euler_rotations.insert(
                            bone_id.clone(),
                            rotation_accumulator_to_euler(state, rotation_mode)?,
                        );
                    }
                    let angles = euler_rotations.get_mut(&bone_id).ok_or_else(|| {
                        PotError::new(
                            ErrorCode::InternalError,
                            "Euler channel initialization failed",
                        )
                    })?;
                    angles[component] = value;
                    mask.rotation[component] = true;
                    mask.rotation_kind = RotationChannelKind::Euler;
                }
            }
            PoseChannel::RotationAxisAngle
                if matches!(
                    pose_bone.rotation_mode,
                    None | Some(PoseRotationMode::AxisAngle)
                ) =>
            {
                if !axis_angle_rotations.contains_key(&bone_id) {
                    let rotation_mode = pose_bone.rotation_mode.unwrap_or_default();
                    let state = match rotation_states.get(&bone_id) {
                        Some(state) => *state,
                        None => rotation_accumulator_from_quaternion(
                            pose_bone.rotation,
                            RotationChannelKind::AxisAngle,
                            rotation_mode,
                        )?,
                    };
                    axis_angle_rotations.insert(
                        bone_id.clone(),
                        rotation_accumulator_to_axis_angle(state, rotation_mode)?,
                    );
                }
                let axis_angle = axis_angle_rotations.get_mut(&bone_id).ok_or_else(|| {
                    PotError::new(
                        ErrorCode::InternalError,
                        "axis-angle channel initialization failed",
                    )
                })?;
                axis_angle[component] = value;
                mask.rotation[component] = true;
                mask.rotation_kind = RotationChannelKind::AxisAngle;
            }
            PoseChannel::Scale => {
                pose_bone.scale[component] = value;
                mask.scale[component] = true;
            }
            PoseChannel::Rotation | PoseChannel::RotationAxisAngle => {}
        }
    }
    for (bone_id, angles) in euler_rotations {
        let pose_bone = pose.get_mut(&bone_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "sampled pose bone is missing")
        })?;
        let rotation_mode = pose_bone.rotation_mode.unwrap_or_default();
        let rotation = quaternion_from_euler_angles(angles, rotation_mode);
        pose_bone.rotation = [rotation.x, rotation.y, rotation.z, rotation.w];
        let mask = channels.entry(bone_id).or_default();
        mask.rotation_kind = RotationChannelKind::Euler;
        mask.rotation_values[..3].copy_from_slice(&angles);
    }
    for (bone_id, axis_angle) in axis_angle_rotations {
        let rotation = axis_angle_quaternion(axis_angle);
        pose.get_mut(&bone_id)
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "sampled pose bone is missing"))?
            .rotation = [rotation.x, rotation.y, rotation.z, rotation.w];
        let mask = channels.entry(bone_id).or_default();
        mask.rotation_kind = RotationChannelKind::AxisAngle;
        mask.rotation_values = axis_angle;
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
    rotation_states: &mut BTreeMap<Id, RotationAccumulator>,
    sampled: &BTreeMap<Id, PoseBone>,
    channels: BTreeMap<Id, PoseChannels>,
    blend: NlaBlendType,
    influence: f64,
) -> Result<()> {
    for (bone_id, mask) in channels {
        let source = sampled.get(&bone_id).copied().ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "sampled pose bone is missing")
        })?;
        let destination = pose.entry(bone_id.clone()).or_default();
        for component in 0..3 {
            if mask.translation[component] {
                let value = source.translation[component];
                destination.translation[component] = blend_scalar(
                    destination.translation[component],
                    value,
                    0.0,
                    blend,
                    influence,
                );
            }
            if mask.scale[component] {
                let value = source.scale[component];
                destination.scale[component] = if blend == NlaBlendType::Combine {
                    destination.scale[component] * value.powf(influence)
                } else {
                    blend_scalar(destination.scale[component], value, 1.0, blend, influence)
                };
            }
        }
        if mask.rotation.iter().any(|animated| *animated) {
            let rotation_mode = destination.rotation_mode.unwrap_or_default();
            if !rotation_states.contains_key(&bone_id) {
                rotation_states.insert(
                    bone_id.clone(),
                    rotation_accumulator_from_quaternion(
                        destination.rotation,
                        mask.rotation_kind,
                        rotation_mode,
                    )?,
                );
            }
            let rotation_state = rotation_states.get_mut(&bone_id).ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "pose rotation state is missing")
            })?;
            blend_rotation_accumulator(
                rotation_state,
                mask.rotation_kind,
                mask.rotation_values,
                mask.rotation,
                rotation_mode,
                blend,
                influence,
            )?;
            destination.rotation =
                rotation_accumulator_to_quaternion(*rotation_state, rotation_mode)?;
        }
    }
    Ok(())
}

fn non_transform_component_count(
    path: &str,
    index: u32,
    node: &Node,
    doc: &SceneDoc,
) -> Option<usize> {
    if !supports_animation_output_for_node(path, index, AnimationOutputKind::Action, node, doc) {
        return None;
    }
    animation_output_component_count(path, AnimationOutputKind::Action)
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

/// Sample one scalar curve, applying its Blender F-Curve modifier stack.
pub(crate) fn sample_curve(curve: &FCurve, frame: f64) -> Result<Option<f64>> {
    sample_curve_with_discrete_values(curve, frame, false)
}

fn sample_curve_discrete(curve: &FCurve, frame: f64) -> Result<Option<f64>> {
    sample_curve_with_discrete_values(curve, frame, true)
}

fn sample_curve_with_discrete_values(
    curve: &FCurve,
    frame: f64,
    discrete: bool,
) -> Result<Option<f64>> {
    if curve.keyframes.is_empty()
        && !curve.modifiers.iter().any(|modifier| {
            matches!(
                modifier.modifier_type.as_str(),
                "GENERATOR" | "FNGENERATOR" | "LIMITS" | "STEPPED"
            )
        })
    {
        return Ok(None);
    }
    if curve.modifiers.is_empty() {
        return sample_curve_base(curve, frame, discrete);
    }
    let (evaltime, cycle_offsets) =
        crate::eval::fcurve_modifiers::evaluate_time(curve, frame as f32)?;
    let value = sample_curve_base(curve, f64::from(evaltime), discrete)?.unwrap_or(0.0);
    let value = crate::eval::fcurve_modifiers::evaluate_value(
        curve,
        evaltime,
        value as f32,
        &cycle_offsets,
    )?;
    finite_sample(curve, f64::from(value))
}

fn sample_curve_base(curve: &FCurve, frame: f64, discrete: bool) -> Result<Option<f64>> {
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
    if discrete {
        let index = keys
            .partition_point(|key| key.frame <= frame)
            .saturating_sub(1);
        return Ok(Some(keys[index].value));
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
            auto_smoothing: None,
            path: "transform.translation".to_owned(),
            index: 0,
            keyframes: keys,
            extrapolation,
            modifiers: Vec::new(),
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
            auto_smoothing: None,
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
            modifiers: Vec::new(),
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
        node: &Id,
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
                node: node.clone(),
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
                    auto_smoothing: None,
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
                    modifiers: Vec::new(),
                }],
                ..Action::default()
            },
        );
        let data_id = Id::new("armature_data").unwrap();
        let _ = doc.data_blocks.insert(
            data_id.clone(),
            crate::model::DataBlock {
                armature: Some(crate::model::ArmatureData {
                    bones: std::collections::BTreeMap::from([(
                        bone_id.clone(),
                        serde_json::from_value(serde_json::json!({
                            "name":"Root","head":[0.0,0.0,0.0],"tail":[0.0,1.0,0.0]
                        }))
                        .unwrap(),
                    )]),
                    ..crate::model::ArmatureData::default()
                }),
                ..crate::model::DataBlock::default()
            },
        );
        let mut animated_node = node(Some(action_id));
        animated_node.kind = "armature".to_owned();
        animated_node.data = Some(data_id);
        let animated = animated_pose_bones(&animated_node, &doc, 6.0).unwrap();
        assert_eq!(
            animated[&bone_id],
            PoseBone {
                translation: [7.0, 0.0, 0.0],
                ..PoseBone::default()
            }
        );
        let euler_action_id = Id::new("pose_euler_action").unwrap();
        let _ = doc.actions.insert(
            euler_action_id.clone(),
            Action {
                name: "Pose Euler".to_owned(),
                fcurves: vec![FCurve {
                    path: "pose.bone_arm.rotation_euler".to_owned(),
                    index: 0,
                    keyframes: vec![Keyframe {
                        frame: 1.0,
                        value: 0.5,
                        interpolation: Interpolation::Linear,
                        ..Keyframe::default()
                    }],
                    auto_smoothing: None,
                    extrapolation: Extrapolation::Constant,
                    modifiers: Vec::new(),
                }],
                ..Action::default()
            },
        );
        let mut rotation_node = animated_node.clone();
        rotation_node.action = Some(euler_action_id);
        rotation_node.pose.insert(
            bone_id.clone(),
            PoseBone {
                rotation_mode: Some(crate::model::PoseRotationMode::Quaternion),
                ..PoseBone::default()
            },
        );
        let quaternion_pose = animated_pose_bones(&rotation_node, &doc, 1.0).unwrap();
        assert_eq!(
            quaternion_pose[&bone_id].rotation,
            PoseBone::default().rotation
        );
        rotation_node.pose.get_mut(&bone_id).unwrap().rotation_mode =
            Some(crate::model::PoseRotationMode::Xyz);
        assert!(animated_pose_bones(&rotation_node, &doc, 1.0).is_ok());
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
                &node_id,
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
                &node_a_id,
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
            // Blender's default for Ocean time is 1.0 (DNA_modifier_types.h:1480), so the NLA stack starts there,
            // not at the stored 0.5. The strip value at frame 2 is 2.0 and its influence 0.5.
            (NlaBlendType::Replace, 1.5),
            (NlaBlendType::Add, 2.0),
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
                    use_animated_influence: true,
                    influence_curve: None,
                    reverse: false,
                    use_sync_length: false,
                    use_auto_blend: false,
                    mute: false,
                    use_animated_time: false,
                    strip_time: 0.0,
                    strip_time_curve: None,
                    use_animated_time_cyclic: false,
                    extrapolation: NlaExtrapolation::Nothing,
                    blend_in: 0.0,
                    blend_out: 0.0,
                }],
            });
            let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
            doc.nodes.insert(node_id.clone(), subject.clone());
            let mut nla_action = ocean_time_action(
                &slot_id,
                &node_id,
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

    /// The NLA value of Ocean `time` at each frame, as Blender 5.2.2 writes it. The committed generator
    /// (`tests/fixtures/ocean_parity/generate_nla_time.py`) runs the scene in Blender and records the value
    /// the modifier holds after `frame_set`. Each case rebuilds the same node, strip and Action, and checks
    /// `animated_modifier_scalar` at every frame. A case with no NLA strip keeps the stored `time`.
    #[test]
    fn ocean_time_nla_matches_blender_522_oracle() -> Result<(), Box<dyn std::error::Error>> {
        let oracle: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/ocean_parity/nla_time_oracle.json"
        ))?;
        assert_eq!(oracle["format"], "potter-ocean-nla-time-oracle");
        assert_eq!(oracle["default_time"].as_f64(), Some(1.0));
        let cases = oracle["cases"]
            .as_array()
            .ok_or("oracle cases are missing")?;
        assert!(!cases.is_empty(), "the Blender oracle has no cases");
        for case in cases {
            let name = case["name"].as_str().ok_or("oracle case has no name")?;
            let stored = case["stored_time"]
                .as_f64()
                .ok_or("stored time is missing")?;
            // `ocean_node` stores time 0.5, so the oracle's stored time must be that value.
            assert_eq!(stored, 0.5, "{name}: stored time");
            let node_id = Id::new(format!("oracle_{name}"))?;
            let modifier_id = Id::new("oracle_ocean")?;
            let action_id = Id::new(format!("oracle_action_{name}"))?;
            let slot_id = Id::new(format!("oracle_slot_{name}"))?;
            let mut subject = ocean_node(None, &modifier_id);
            let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
            if let Some(strip) = case["strip"].as_object() {
                let keys = case["keys"].as_array().ok_or("oracle keys are missing")?;
                assert_eq!(keys.len(), 2, "{name}: two keys");
                assert_eq!(
                    keys[0]["frame"].as_f64(),
                    Some(1.0),
                    "{name}: first key frame"
                );
                assert_eq!(
                    keys[1]["frame"].as_f64(),
                    Some(3.0),
                    "{name}: second key frame"
                );
                let first_value = keys[0]["value"]
                    .as_f64()
                    .ok_or("first key value is missing")?;
                let second_value = keys[1]["value"]
                    .as_f64()
                    .ok_or("second key value is missing")?;
                let number = |field: &str| {
                    strip[field]
                        .as_f64()
                        .ok_or_else(|| format!("{name}: strip {field} is missing"))
                };
                assert_eq!(strip["extrapolation"], "NOTHING", "{name}: extrapolation");
                let blend_type = match strip["blend_type"].as_str() {
                    Some("REPLACE") => NlaBlendType::Replace,
                    Some("ADD") => NlaBlendType::Add,
                    Some("COMBINE") => NlaBlendType::Combine,
                    other => return Err(format!("{name}: unknown blend type {other:?}").into()),
                };
                subject.nla_tracks.push(NlaTrack {
                    id: Id::new(format!("oracle_track_{name}"))?,
                    name: "Ocean Time".to_owned(),
                    mute: false,
                    solo: false,
                    strips: vec![NlaStrip {
                        id: Id::new(format!("oracle_strip_{name}"))?,
                        action: action_id.clone(),
                        action_slot: Some(slot_id.clone()),
                        frame_start: number("frame_start")?,
                        frame_end: number("frame_end")?,
                        action_frame_start: number("action_frame_start")?,
                        action_frame_end: number("action_frame_end")?,
                        scale: 1.0,
                        repeat: 1.0,
                        blend_type,
                        influence: number("influence")?,
                        use_animated_influence: strip["use_animated_influence"]
                            .as_bool()
                            .unwrap_or(false),
                        influence_curve: None,
                        reverse: false,
                        use_sync_length: false,
                        use_auto_blend: false,
                        mute: false,
                        use_animated_time: false,
                        strip_time: 0.0,
                        strip_time_curve: None,
                        use_animated_time_cyclic: false,
                        extrapolation: NlaExtrapolation::Nothing,
                        blend_in: 0.0,
                        blend_out: 0.0,
                    }],
                });
                let mut action = ocean_time_action(
                    &slot_id,
                    &node_id,
                    vec![ocean_modifier_binding(&node_id, &modifier_id, &slot_id)],
                    first_value,
                    second_value,
                );
                action.active_slots.clear();
                doc.actions.insert(action_id, action);
            }
            doc.nodes.insert(node_id.clone(), subject.clone());
            let expected = case["expected_by_frame"]
                .as_object()
                .ok_or("expected values are missing")?;
            assert_eq!(expected.len(), 7, "{name}: frames 0 to 6");
            for (frame_text, value) in expected {
                let frame: f64 = frame_text.parse()?;
                let expected = value.as_f64().ok_or("expected value is not numeric")?;
                let actual = animated_modifier_scalar(
                    &node_id,
                    &subject,
                    &doc,
                    &modifier_id,
                    "time",
                    stored,
                    frame,
                )?
                .unwrap_or(stored);
                assert!(
                    (actual - expected).abs() <= 1.0e-6,
                    "{name} frame {frame}: Potter {actual} vs Blender 5.2.2 {expected}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn unsupported_non_transform_action_path_is_refused() {
        let node_id = Id::new("unsupported_channel_node").unwrap();
        let action_id = Id::new("unsupported_channel_action").unwrap();
        let subject = node(Some(action_id.clone()));
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        doc.nodes.insert(node_id, subject.clone());
        let mut curve = curve(
            vec![
                Keyframe {
                    frame: 1.0,
                    value: 1.0,
                    ..Keyframe::default()
                },
                Keyframe {
                    frame: 2.0,
                    value: 0.0,
                    ..Keyframe::default()
                },
            ],
            Extrapolation::Constant,
        );
        curve.path = "visible".to_owned();
        doc.actions.insert(
            action_id,
            Action {
                name: "Unsupported Visibility".to_owned(),
                fcurves: vec![curve],
                ..Action::default()
            },
        );

        let error = animated_transform(&subject, &doc, 1.0).unwrap_err();
        assert_eq!(error.code, crate::error::ErrorCode::UnsupportedFeature);
        assert_eq!(
            error.details["feature_id"],
            serde_json::json!("animation.fcurve_non_transform_channels.visible")
        );
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
                slots: Vec::new(),
                active_slots: std::collections::BTreeMap::new(),
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
                use_animated_influence: false,
                influence_curve: None,
                reverse: false,
                use_sync_length: false,
                use_auto_blend: false,
                mute: false,
                use_animated_time: false,
                strip_time: 0.0,
                strip_time_curve: None,
                use_animated_time_cyclic: false,
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
