use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    path::Path,
};

use glam::{DMat3, DMat4, DQuat, DVec3, EulerRot, Mat3, Mat4};
use serde_json::{Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    eval::animation::animated_transform,
    geom::Mesh,
    model::{
        ArmatureData, Constraint, ConstraintType, DataBlock, Driver, DriverType, DriverVariable,
        DriverVariableType, Id, Modifier, Node, ParentType, PoseBone, SceneDoc, ShapeKeyData,
        Transform,
    },
};

/// Rest-space, evaluated pose-space, and rest-to-pose deformation for a bone.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct BoneMatrices {
    pub(crate) rest: DMat4,
    pub(crate) pose: DMat4,
    pub(crate) skin: DMat4,
}

#[derive(Clone, Copy)]
struct IkGoal {
    target: DMat4,
    pole: Option<DMat4>,
}

fn evaluation_error(message: impl Into<String>) -> PotError {
    PotError::new(ErrorCode::EvaluationFailed, message)
}

fn evaluation_error_with(message: impl Into<String>, details: Value) -> PotError {
    PotError::with_details(ErrorCode::EvaluationFailed, message, details)
}

fn finite_matrix(matrix: DMat4) -> bool {
    matrix.to_cols_array().iter().all(|value| value.is_finite())
}

fn checked_inverse(matrix: DMat4, what: &str) -> Result<DMat4> {
    if !finite_matrix(matrix) {
        return Err(evaluation_error(format!("{what} matrix is non-finite")));
    }
    let inverse = matrix.inverse();
    if !finite_matrix(inverse) {
        return Err(evaluation_error(format!(
            "{what} matrix inverse is non-finite"
        )));
    }
    Ok(inverse)
}

fn pose_delta(pose: PoseBone, bone_id: &Id) -> Result<DMat4> {
    let translation = DVec3::from_array(pose.translation);
    let scale = DVec3::from_array(pose.scale);
    let rotation = DQuat::from_xyzw(
        pose.rotation[0],
        pose.rotation[1],
        pose.rotation[2],
        pose.rotation[3],
    );
    if !translation.is_finite() || !scale.is_finite() || !rotation.is_finite() {
        return Err(evaluation_error_with(
            "pose transform contains a non-finite value",
            json!({ "bone_id": bone_id }),
        ));
    }
    if rotation.length_squared() <= f64::EPSILON {
        return Err(evaluation_error_with(
            "pose rotation has zero length",
            json!({ "bone_id": bone_id }),
        ));
    }
    Ok(DMat4::from_scale_rotation_translation(
        scale,
        rotation.normalize(),
        translation,
    ))
}

fn bone_rest_matrix(bone_id: &Id, bone: &crate::model::Bone) -> Result<DMat4> {
    let head = DVec3::from_array(bone.head);
    let tail = DVec3::from_array(bone.tail);
    let direction = tail - head;
    if !head.is_finite()
        || !tail.is_finite()
        || !bone.roll.is_finite()
        || direction.length_squared() <= f64::EPSILON
    {
        return Err(evaluation_error_with(
            "bone rest transform is invalid",
            json!({ "bone_id": bone_id }),
        ));
    }
    // Blender's bone-local positive Y axis runs from head to tail. Roll is around that axis.
    let alignment = DQuat::from_rotation_arc(DVec3::Y, direction.normalize());
    let rolled = alignment * DQuat::from_rotation_y(bone.roll);
    let matrix = DMat4::from_rotation_translation(rolled, head);
    if !finite_matrix(matrix) {
        return Err(evaluation_error_with(
            "bone rest transform is non-finite",
            json!({ "bone_id": bone_id }),
        ));
    }
    Ok(matrix)
}

/// Evaluate each bone's rest, pose, and skin matrices in armature-local coordinates.
///
/// Bones are traversed by stable ID, and invalid parent references or bone cycles fail with
/// `EVALUATION_FAILED` rather than producing partial matrices.
pub(crate) fn evaluate_bone_matrices(
    armature: &ArmatureData,
    pose: &BTreeMap<Id, PoseBone>,
) -> Result<BTreeMap<Id, BoneMatrices>> {
    evaluate_bone_matrices_with_ik_overrides(armature, pose, &BTreeMap::new())
}

fn evaluate_bone_matrices_with_ik_overrides(
    armature: &ArmatureData,
    pose: &BTreeMap<Id, PoseBone>,
    pose_matrix_overrides: &BTreeMap<Id, DMat4>,
) -> Result<BTreeMap<Id, BoneMatrices>> {
    fn visit(
        id: &Id,
        armature: &ArmatureData,
        pose: &BTreeMap<Id, PoseBone>,
        pose_matrix_overrides: &BTreeMap<Id, DMat4>,
        output: &mut BTreeMap<Id, BoneMatrices>,
        active: &mut BTreeSet<Id>,
    ) -> Result<BoneMatrices> {
        if let Some(matrices) = output.get(id) {
            return Ok(*matrices);
        }
        if !active.insert(id.clone()) {
            return Err(evaluation_error_with(
                "armature bone parent cycle",
                json!({ "bone_id": id }),
            ));
        }
        let bone = armature.bones.get(id).ok_or_else(|| {
            evaluation_error_with(
                "bone parent reference does not exist",
                json!({ "bone_id": id }),
            )
        })?;
        let mut rest = bone_rest_matrix(id, bone)?;
        let mut pose_value = pose.get(id).copied().unwrap_or_default();
        if bone.use_connect {
            pose_value.translation = [0.0; 3];
        }
        let delta = pose_delta(pose_value, id)?;
        let evaluated = if let Some(parent_id) = &bone.parent {
            let parent_bone = armature.bones.get(parent_id).ok_or_else(|| {
                evaluation_error_with(
                    "bone parent reference does not exist",
                    json!({ "bone_id": id, "parent": parent_id }),
                )
            })?;
            let parent_matrices = visit(
                parent_id,
                armature,
                pose,
                pose_matrix_overrides,
                output,
                active,
            )?;
            let parent_rest_inverse = checked_inverse(parent_matrices.rest, "parent bone rest")?;
            if bone.use_connect {
                let parent_length = DVec3::from_array(parent_bone.tail)
                    .distance(DVec3::from_array(parent_bone.head));
                let connected_head = parent_matrices
                    .rest
                    .transform_point3(DVec3::Y * parent_length);
                let (_, rotation, _) = rest.to_scale_rotation_translation();
                rest = DMat4::from_rotation_translation(rotation, connected_head);
            }
            let mut child_rest = rest;
            if !bone.inherit_rotation {
                let (_, child_rotation, child_translation) = rest.to_scale_rotation_translation();
                let (_, parent_pose_rotation, _) =
                    parent_matrices.pose.to_scale_rotation_translation();
                let (_, parent_rest_rotation, _) =
                    parent_matrices.rest.to_scale_rotation_translation();
                let inherited = parent_pose_rotation * parent_rest_rotation.inverse();
                let adjusted_rotation = inherited.inverse() * child_rotation;
                child_rest = DMat4::from_rotation_translation(adjusted_rotation, child_translation);
            }
            parent_matrices.pose * parent_rest_inverse * child_rest * delta
        } else {
            rest * delta
        };
        let posed = pose_matrix_overrides.get(id).copied().unwrap_or(evaluated);
        if !finite_matrix(posed) {
            return Err(evaluation_error_with(
                "evaluated bone pose is non-finite",
                json!({ "bone_id": id }),
            ));
        }
        let skin = posed * checked_inverse(rest, "bone rest")?;
        let matrices = BoneMatrices {
            rest,
            pose: posed,
            skin,
        };
        active.remove(id);
        output.insert(id.clone(), matrices);
        Ok(matrices)
    }

    let mut output = BTreeMap::new();
    let mut active = BTreeSet::new();
    for id in armature.bones.keys() {
        visit(
            id,
            armature,
            pose,
            pose_matrix_overrides,
            &mut output,
            &mut active,
        )?;
    }
    for id in pose.keys() {
        if !armature.bones.contains_key(id) {
            return Err(evaluation_error_with(
                "pose references an unknown bone",
                json!({ "bone_id": id }),
            ));
        }
    }
    for id in pose_matrix_overrides.keys() {
        if !armature.bones.contains_key(id) {
            return Err(evaluation_error_with(
                "pose matrix override references an unknown bone",
                json!({ "bone_id": id }),
            ));
        }
    }
    Ok(output)
}

fn constraint_target_bone_matrix(
    constraint: &Constraint,
    target_node: &Node,
    doc: &SceneDoc,
    matrix: DMat4,
) -> Result<DMat4> {
    constraint_target_bone_matrix_for_id(
        constraint,
        target_node,
        doc,
        matrix,
        constraint.subtarget.as_ref(),
    )
}

fn constraint_target_bone_matrix_for_id(
    constraint: &Constraint,
    target_node: &Node,
    doc: &SceneDoc,
    mut matrix: DMat4,
    bone_id: Option<&Id>,
) -> Result<DMat4> {
    let Some(bone_id) = bone_id else {
        return Ok(matrix);
    };
    let head_tail = number_param(&constraint.params, "head_tail")?.unwrap_or(0.0);
    let use_bbone_shape = param_bool(&constraint.params, "use_bbone_shape", false)?;
    if head_tail == 0.0 {
        return Ok(matrix);
    }
    let bone = target_node
        .data
        .as_ref()
        .and_then(|data_id| doc.data_blocks.get(data_id))
        .and_then(|data| data.armature.as_ref())
        .and_then(|armature| armature.bones.get(bone_id))
        .ok_or_else(|| {
            evaluation_error_with(
                "constraint target bone data is unavailable",
                json!({"constraint_id":constraint.id,"target_bone":bone_id}),
            )
        })?;
    let bbone_segments = bone
        .bbone_settings
        .get("bbone_segments")
        .and_then(Value::as_u64)
        .unwrap_or(1);
    if use_bbone_shape && bbone_segments > 1 && (1.0e-6..1.0 - 1.0e-6).contains(&head_tail) {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "constraint target sampling along a segmented B-Bone shape is not evaluated",
            json!({
                "feature_id":"constraint.target_bbone_shape",
                "constraint_id":constraint.id,
                "target_bone":bone_id,
                "bbone_segments":bbone_segments,
                "head_tail":head_tail
            }),
        ));
    }
    let length = DVec3::from_array(bone.tail).distance(DVec3::from_array(bone.head));
    let offset = DVec3::Y * (length * head_tail);
    matrix.w_axis = matrix.transform_point3(offset).extend(1.0);
    Ok(matrix)
}
fn armature_constraint_bbone_gate(doc: &SceneDoc, constraint: &Constraint) -> Result<()> {
    let target_items = constraint.params.get("targets").and_then(Value::as_array);
    let target_count = target_items.map_or(usize::from(constraint.target.is_some()), Vec::len);
    let primary_target = constraint.target.as_ref().map(Id::as_str);
    let primary_subtarget = constraint.subtarget.as_ref().map(Id::as_str);
    for index in 0..target_count {
        let target_item = target_items.and_then(|items| items.get(index));
        let weight = target_item
            .and_then(|item| item.get("weight"))
            .and_then(Value::as_f64)
            .unwrap_or(1.0)
            .clamp(0.0, 1.0);
        if weight == 0.0 {
            continue;
        }
        let target_text = target_item
            .and_then(|item| item.get("target"))
            .and_then(Value::as_str)
            .or(primary_target);
        let subtarget = target_item
            .and_then(|item| item.get("subtarget"))
            .and_then(Value::as_str)
            .or(primary_subtarget);
        let (Some(target_text), Some(subtarget)) = (target_text, subtarget) else {
            continue;
        };
        let Some(target_id) = Id::new(target_text.to_owned()).ok() else {
            continue;
        };
        let Some(armature) = doc
            .nodes
            .get(&target_id)
            .and_then(|node| node.data.as_ref())
            .and_then(|data_id| doc.data_blocks.get(data_id))
            .and_then(|data| data.armature.as_ref())
        else {
            continue;
        };
        let bone = Id::new(subtarget.to_owned())
            .ok()
            .and_then(|bone_id| armature.bones.get(&bone_id))
            .or_else(|| armature.bones.values().find(|bone| bone.name == subtarget));
        let Some(bone) = bone else {
            continue;
        };
        let bbone_segments = bone
            .bbone_settings
            .get("bbone_segments")
            .and_then(Value::as_u64)
            .unwrap_or(1);
        if bbone_segments > 1 {
            return Err(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                "Armature constraint B-Bone segment deformation is not evaluated",
                json!({
                    "feature_id":"rig.bbone.segments",
                    "constraint_id":constraint.id,
                    "target":target_text,
                    "target_bone":subtarget,
                    "bbone_segments":bbone_segments
                }),
            ));
        }
    }
    Ok(())
}

fn armature_envelope_weight(
    point: DVec3,
    bone_head: [f64; 3],
    bone_tail: [f64; 3],
    head_radius: f64,
    tail_radius: f64,
    falloff_distance: f64,
) -> f64 {
    let head = DVec3::from_array(bone_head);
    let tail = DVec3::from_array(bone_tail);
    let axis = tail - head;
    let bone_length = axis.length();
    let bone_axis = if bone_length > 0.0 {
        axis / bone_length
    } else {
        DVec3::ZERO
    };
    let height = (point - head).dot(bone_axis);
    let (distance_squared, closest_radius) = if height < 0.0 {
        (point.distance_squared(head), head_radius)
    } else if height > bone_length {
        (point.distance_squared(tail), tail_radius)
    } else {
        let radius = if bone_length > 0.0 {
            head_radius + (tail_radius - head_radius) * (height / bone_length)
        } else {
            head_radius
        };
        (
            (point.distance_squared(head) - height * height).max(0.0),
            radius,
        )
    };
    if falloff_distance == 0.0 || distance_squared >= (closest_radius + falloff_distance).powi(2) {
        return 0.0;
    }
    let distance_from_envelope = distance_squared.sqrt() - closest_radius;
    1.0 - distance_from_envelope.powi(2) / falloff_distance.powi(2)
}

fn armature_constraint_has_zero_envelope_weight(
    doc: &SceneDoc,
    constraint: &Constraint,
    owner_node_id: Option<&Id>,
    owner_world: DMat4,
    source: DMat4,
    world_matrices: &BTreeMap<Id, DMat4>,
) -> Result<bool> {
    if !param_bool(&constraint.params, "use_bone_envelopes", false)? {
        return Ok(false);
    }
    let Some(targets) = constraint.params.get("targets").and_then(Value::as_array) else {
        return Ok(true);
    };
    let owner_point_world = if param_bool(&constraint.params, "use_current_location", false)? {
        source.w_axis.truncate()
    } else if let Some(owner_bone_id) = &constraint.owner_bone {
        let owner_node_id = owner_node_id
            .ok_or_else(|| evaluation_error("Armature constraint owner node is missing"))?;
        let bone = doc
            .nodes
            .get(owner_node_id)
            .and_then(|node| node.data.as_ref())
            .and_then(|data_id| doc.data_blocks.get(data_id))
            .and_then(|data| data.armature.as_ref())
            .and_then(|armature| armature.bones.get(owner_bone_id))
            .ok_or_else(|| evaluation_error("Armature constraint owner bone is missing"))?;
        owner_world.transform_point3(DVec3::from_array(bone.head))
    } else {
        source.w_axis.truncate()
    };
    for target in targets {
        let target_weight = target
            .get("weight")
            .and_then(Value::as_f64)
            .unwrap_or(1.0)
            .clamp(0.0, 1.0);
        if target_weight == 0.0 {
            continue;
        }
        let Some(target_name) = target.get("target").and_then(Value::as_str) else {
            continue;
        };
        let Some(subtarget) = target.get("subtarget").and_then(Value::as_str) else {
            continue;
        };
        let Ok(target_id) = Id::new(target_name.to_owned()) else {
            continue;
        };
        let Some(target_node) = doc.nodes.get(&target_id) else {
            continue;
        };
        let Some(armature) = target_node
            .data
            .as_ref()
            .and_then(|data_id| doc.data_blocks.get(data_id))
            .and_then(|data| data.armature.as_ref())
        else {
            continue;
        };
        let target_bone = Id::new(subtarget.to_owned())
            .ok()
            .and_then(|bone_id| armature.bones.get(&bone_id))
            .or_else(|| armature.bones.values().find(|bone| bone.name == subtarget));
        let Some(target_bone) = target_bone else {
            continue;
        };
        let target_world = world_matrices.get(&target_id).copied().ok_or_else(|| {
            evaluation_error_with(
                "Armature constraint target transform is unavailable",
                json!({"constraint_id":constraint.id,"target":target_id}),
            )
        })?;
        let target_local_point = checked_inverse(target_world, "Armature constraint target")?
            .transform_point3(owner_point_world);
        let envelope_weight = armature_envelope_weight(
            target_local_point,
            target_bone.head,
            target_bone.tail,
            target_bone.head_radius,
            target_bone.tail_radius,
            target_bone.envelope_distance,
        );
        if target_weight * envelope_weight > 0.0 {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Evaluate an armature node's posed bones and return each bone's world-space head matrix.
///
/// The matrix translation is the evaluated bone head, and its local positive Y axis follows the
/// bone toward its tail. Pose animation and enabled IK constraints are included.
pub(crate) fn evaluate_armature_bone_world_matrices_with_drivers(
    doc: &SceneDoc,
    armature_node_id: &Id,
    frame: f64,
    armature_world: DMat4,
    target_worlds: &BTreeMap<Id, DMat4>,
    driver_values: &BTreeMap<(Id, String, u32), f64>,
) -> Result<BTreeMap<Id, DMat4>> {
    evaluate_armature_bone_world_matrices_with_resources(
        doc,
        armature_node_id,
        frame,
        armature_world,
        target_worlds,
        driver_values,
        None,
        24,
        1.0,
    )
}

pub(crate) fn evaluate_armature_bone_world_matrices_with_resources(
    doc: &SceneDoc,
    armature_node_id: &Id,
    frame: f64,
    armature_world: DMat4,
    target_worlds: &BTreeMap<Id, DMat4>,
    driver_values: &BTreeMap<(Id, String, u32), f64>,
    project_root: Option<&Path>,
    fps: u32,
    fps_base: f64,
) -> Result<BTreeMap<Id, DMat4>> {
    evaluate_armature_bone_world_matrices_with_meshes(
        doc,
        armature_node_id,
        frame,
        armature_world,
        target_worlds,
        driver_values,
        project_root,
        fps,
        fps_base,
        None,
    )
}

pub(crate) fn evaluate_armature_bone_world_matrices_with_evaluated_meshes(
    doc: &SceneDoc,
    armature_node_id: &Id,
    frame: f64,
    armature_world: DMat4,
    target_worlds: &BTreeMap<Id, DMat4>,
    driver_values: &BTreeMap<(Id, String, u32), f64>,
    project_root: Option<&Path>,
    fps: u32,
    fps_base: f64,
    evaluated_meshes: &BTreeMap<Id, Mesh>,
) -> Result<BTreeMap<Id, DMat4>> {
    evaluate_armature_bone_world_matrices_with_meshes(
        doc,
        armature_node_id,
        frame,
        armature_world,
        target_worlds,
        driver_values,
        project_root,
        fps,
        fps_base,
        Some(evaluated_meshes),
    )
}

fn evaluate_armature_bone_world_matrices_with_meshes(
    doc: &SceneDoc,
    armature_node_id: &Id,
    frame: f64,
    armature_world: DMat4,
    target_worlds: &BTreeMap<Id, DMat4>,
    driver_values: &BTreeMap<(Id, String, u32), f64>,
    project_root: Option<&Path>,
    fps: u32,
    fps_base: f64,
    evaluated_meshes: Option<&BTreeMap<Id, Mesh>>,
) -> Result<BTreeMap<Id, DMat4>> {
    let node = doc.nodes.get(armature_node_id).ok_or_else(|| {
        evaluation_error_with(
            "armature node does not exist",
            json!({ "node": armature_node_id }),
        )
    })?;
    let data_id = node.data.as_ref().ok_or_else(|| {
        evaluation_error_with(
            "armature node has no Data-Block",
            json!({ "node": armature_node_id }),
        )
    })?;
    let data = doc.data_blocks.get(data_id).ok_or_else(|| {
        evaluation_error_with(
            "armature Data-Block does not exist",
            json!({ "node": armature_node_id, "data": data_id }),
        )
    })?;
    let armature = data.armature.as_ref().ok_or_else(|| {
        evaluation_error_with(
            "node Data-Block is not an armature",
            json!({ "node": armature_node_id }),
        )
    })?;
    let mut pose = crate::eval::animation::animated_pose_bones(node, doc, frame)?;
    apply_pose_drivers(armature_node_id, &mut pose, driver_values)?;
    apply_spline_ik(
        armature_node_id,
        armature,
        node,
        doc,
        armature_world,
        target_worlds,
        &mut pose,
    )?;
    let ik_targets = ik_target_matrices(
        doc,
        armature_node_id,
        armature,
        &pose,
        frame,
        armature_world,
        target_worlds,
        driver_values,
        project_root,
        fps,
        fps_base,
    )?;
    let bones = evaluate_pose_matrices(
        armature_node_id,
        armature,
        &pose,
        &node.constraints,
        doc,
        frame,
        armature_world,
        target_worlds,
        driver_values,
        &ik_targets,
        project_root,
        fps,
        fps_base,
        evaluated_meshes,
    )?;
    bones
        .into_iter()
        .map(|(bone_id, matrices)| {
            let world = armature_world * matrices.pose;
            if !finite_matrix(world) {
                return Err(evaluation_error_with(
                    "evaluated bone world matrix is non-finite",
                    json!({ "node": armature_node_id, "bone_id": bone_id }),
                ));
            }
            Ok((bone_id, world))
        })
        .collect()
}

#[expect(
    clippy::too_many_lines,
    reason = "spline IK setup, chain mapping, and pose application form one solver step"
)]
#[expect(
    clippy::cast_precision_loss,
    reason = "Blender spline IK chains are limited to 255 bones"
)]
#[expect(
    clippy::manual_midpoint,
    reason = "matches Blender's float evaluation order for parity"
)]
fn apply_spline_ik(
    armature_node_id: &Id,
    armature: &ArmatureData,
    node: &Node,
    doc: &SceneDoc,
    armature_world: DMat4,
    target_worlds: &BTreeMap<Id, DMat4>,
    pose: &mut BTreeMap<Id, PoseBone>,
) -> Result<()> {
    for constraint in node.constraints.iter().filter(|constraint| {
        constraint.enabled
            && constraint.constraint_type == ConstraintType::SplineIk
            && constraint.influence != 0.0
    }) {
        if string_param_default(&constraint.params, "owner_space", "WORLD")? == "CUSTOM"
            || string_param_default(&constraint.params, "target_space", "WORLD")? == "CUSTOM"
        {
            continue;
        }
        let end_id = constraint
            .owner_bone
            .as_ref()
            .or(constraint.subtarget.as_ref())
            .ok_or_else(|| {
                evaluation_error_with(
                    "spline IK requires an owner bone or final bone subtarget",
                    json!({ "constraint_id": constraint.id }),
                )
            })?;
        let target_id = constraint.target.as_ref().ok_or_else(|| {
            evaluation_error_with(
                "spline IK requires a curve target",
                json!({ "constraint_id": constraint.id }),
            )
        })?;
        let curve_node = doc.nodes.get(target_id).ok_or_else(|| {
            evaluation_error_with(
                "spline IK curve target does not exist",
                json!({ "armature": armature_node_id, "target": target_id }),
            )
        })?;
        let curve_data_id = curve_node.data.as_ref().ok_or_else(|| {
            evaluation_error_with(
                "spline IK target has no curve data",
                json!({ "target": target_id }),
            )
        })?;
        let curve = doc
            .data_blocks
            .get(curve_data_id)
            .and_then(|data| data.curve.as_ref())
            .ok_or_else(|| {
                evaluation_error_with(
                    "spline IK target has no curve data",
                    json!({ "target": target_id }),
                )
            })?;
        let spline = curve.splines.first().ok_or_else(|| {
            evaluation_error_with(
                "spline IK target has no splines",
                json!({ "target": target_id }),
            )
        })?;
        let target_world = target_worlds.get(target_id).copied().ok_or_else(|| {
            evaluation_error_with(
                "spline IK target transform is unavailable",
                json!({ "target": target_id }),
            )
        })?;
        let chain_offset = param_bool(&constraint.params, "use_chain_offset", false)?;
        let curve_world = if chain_offset {
            DMat4::IDENTITY
        } else {
            target_world
        };
        let mut path_samples =
            crate::geom::curve::evaluate_curve_spline(spline, curve.twist_mode.as_str(), false)
                .map_err(|error| {
                    evaluation_error(format!("spline IK curve evaluation failed: {error}"))
                })?;
        if spline.cyclic
            && let Some(first) = path_samples.first().copied()
        {
            path_samples.push(first);
        }
        let armature_inverse = checked_inverse(armature_world, "armature world")?;
        let path = path_samples
            .iter()
            .map(|sample| {
                armature_inverse.transform_point3(curve_world.transform_point3(sample.position))
            })
            .collect::<Vec<_>>();
        if path.len() < 2 || path.iter().any(|point| !point.is_finite()) {
            return Err(evaluation_error_with(
                "spline IK curve must contain at least two finite points",
                json!({ "target": target_id }),
            ));
        }
        let mut chain = Vec::new();
        let mut visited = BTreeSet::new();
        let mut cursor = Some(end_id);
        let maximum = param_usize(&constraint.params, "chain_count")?
            .filter(|count| *count > 0)
            .unwrap_or(255)
            .min(255);
        while let Some(bone_id) = cursor {
            if chain.len() >= maximum {
                break;
            }
            if !visited.insert(bone_id.clone()) {
                return Err(evaluation_error_with(
                    "spline IK bone chain contains a cycle",
                    json!({ "bone_id": bone_id }),
                ));
            }
            let bone = armature.bones.get(bone_id).ok_or_else(|| {
                evaluation_error_with(
                    "spline IK chain references a missing bone",
                    json!({ "bone_id": bone_id }),
                )
            })?;
            chain.push(bone_id.clone());
            cursor = bone.parent.as_ref();
        }
        chain.reverse();
        if chain.is_empty() {
            return Err(evaluation_error_with(
                "spline IK chain_count must select at least one bone",
                json!({ "constraint_id": constraint.id }),
            ));
        }
        let lengths = chain
            .iter()
            .map(|bone_id| {
                let bone = armature
                    .bones
                    .get(bone_id)
                    .ok_or_else(|| evaluation_error("spline IK bone disappeared"))?;
                let length = DVec3::from_array(bone.head).distance(DVec3::from_array(bone.tail));
                if !length.is_finite() || length <= f64::EPSILON {
                    return Err(evaluation_error_with(
                        "spline IK chain contains a zero-length bone",
                        json!({ "bone_id": bone_id }),
                    ));
                }
                Ok(length)
            })
            .collect::<Result<Vec<_>>>()?;
        let chain_length = lengths.iter().sum::<f64>();
        let (path_lengths, path_length) = curve_path_lengths(&path)?;
        let y_scale_mode = string_param_default(&constraint.params, "y_scale_mode", "FIT_CURVE")?;
        let path_scale = if y_scale_mode == "FIT_CURVE" {
            1.0
        } else {
            chain_length / path_length
        };
        let influence = checked_influence(constraint)?;
        let even_divisions = param_bool(&constraint.params, "use_even_divisions", false)?;
        let use_curve_radius = param_bool(&constraint.params, "use_curve_radius", true)?;
        let mut offset = 0.0;
        for (index, bone_id) in chain.iter().enumerate() {
            let (start_fraction, end_fraction) = if even_divisions {
                (
                    index as f64 / chain.len() as f64,
                    (index + 1) as f64 / chain.len() as f64,
                )
            } else {
                (
                    offset / chain_length,
                    (offset + lengths[index]) / chain_length,
                )
            };
            offset += lengths[index];
            let start = start_fraction * path_length * path_scale;
            let end = end_fraction * path_length * path_scale;
            let head = curve_path_sample(&path, &path_lengths, start)?;
            let tail = curve_path_sample(&path, &path_lengths, end)?;
            let direction = (tail - head).normalize_or_zero();
            if direction.length_squared() <= f64::EPSILON {
                return Err(evaluation_error_with(
                    "spline IK curve has a degenerate segment",
                    json!({ "bone_id": bone_id }),
                ));
            }
            let curve_radius = if use_curve_radius {
                let head_sample =
                    curve_path_sample_attributes(&path_samples, &path_lengths, start)?;
                let tail_sample = curve_path_sample_attributes(&path_samples, &path_lengths, end)?;
                (head_sample.radius + tail_sample.radius) * 0.5
            } else {
                1.0
            };
            let matrices = evaluate_bone_matrices(armature, pose)?;
            let bone_matrices = matrices
                .get(bone_id)
                .copied()
                .ok_or_else(|| evaluation_error("spline IK bone matrix is missing"))?;
            let base = if let Some(parent_id) = armature
                .bones
                .get(bone_id)
                .and_then(|bone| bone.parent.as_ref())
            {
                let parent = matrices
                    .get(parent_id)
                    .copied()
                    .ok_or_else(|| evaluation_error("spline IK parent matrix is missing"))?;
                parent.pose
                    * checked_inverse(parent.rest, "spline IK parent rest")?
                    * bone_matrices.rest
            } else {
                bone_matrices.rest
            };
            let inverse_base = checked_inverse(base, "spline IK bone parent transform")?;
            let local_translation = inverse_base.transform_point3(head);
            let local_axis = inverse_base
                .transform_vector3(direction)
                .normalize_or_zero();
            if !local_translation.is_finite()
                || !local_axis.is_finite()
                || local_axis.length_squared() <= f64::EPSILON
            {
                return Err(evaluation_error_with(
                    "spline IK local bone transform is degenerate",
                    json!({ "bone_id": bone_id }),
                ));
            }
            let old = pose.get(bone_id).copied().unwrap_or_default();
            let old_rotation = quat_from_array(old.rotation, "spline IK pose rotation")?;
            let local_rotation = (DQuat::from_rotation_arc(old_rotation * DVec3::Y, local_axis)
                * old_rotation)
                .normalize();
            let rotation = old_rotation.slerp(local_rotation, influence).normalize();
            let axis_scale = base.transform_vector3(local_rotation * DVec3::Y).length();
            let fit_scale = head.distance(tail) / (lengths[index] * axis_scale);
            let scale_y = if y_scale_mode == "FIT_CURVE" {
                fit_scale
            } else {
                DVec3::from_array(old.scale).y
            };
            if !axis_scale.is_finite()
                || axis_scale <= f64::EPSILON
                || !scale_y.is_finite()
                || scale_y <= 0.0
                || !curve_radius.is_finite()
                || curve_radius < 0.0
            {
                return Err(evaluation_error_with(
                    "spline IK bone scale is invalid",
                    json!({ "bone_id": bone_id }),
                ));
            }
            let old_scale = DVec3::from_array(old.scale);
            let desired_translation = if chain_offset || (influence < 1.0 && index > 0) {
                DVec3::from_array(old.translation)
            } else {
                DVec3::from_array(old.translation).lerp(local_translation, influence)
            };
            let desired_scale = DVec3::new(
                old_scale.x * curve_radius,
                scale_y,
                old_scale.z * curve_radius,
            );
            pose.insert(
                bone_id.clone(),
                PoseBone {
                    translation: desired_translation.to_array(),
                    rotation: [rotation.x, rotation.y, rotation.z, rotation.w],
                    scale: old_scale.lerp(desired_scale, influence).to_array(),
                    ..old
                },
            );
        }
    }
    Ok(())
}

fn curve_path_lengths(path: &[DVec3]) -> Result<(Vec<f64>, f64)> {
    let mut cumulative = Vec::with_capacity(path.len());
    cumulative.push(0.0);
    for pair in path.windows(2) {
        let segment = pair[0].distance(pair[1]);
        if !segment.is_finite() {
            return Err(evaluation_error("spline IK curve length is non-finite"));
        }
        cumulative.push(cumulative.last().copied().unwrap_or(0.0) + segment);
    }
    let total = cumulative.last().copied().unwrap_or(0.0);
    if total <= f64::EPSILON {
        return Err(evaluation_error("spline IK curve length must be positive"));
    }
    Ok((cumulative, total))
}

fn curve_path_sample(path: &[DVec3], cumulative: &[f64], distance: f64) -> Result<DVec3> {
    if path.len() != cumulative.len() || path.len() < 2 {
        return Err(evaluation_error("spline IK path sampling input is invalid"));
    }
    let distance = distance.clamp(0.0, cumulative.last().copied().unwrap_or(0.0));
    let segment = cumulative
        .windows(2)
        .position(|span| distance <= span[1])
        .unwrap_or(path.len() - 2);
    let start = cumulative[segment];
    let end = cumulative[segment + 1];
    let factor = if end > start {
        (distance - start) / (end - start)
    } else {
        0.0
    };
    Ok(path[segment].lerp(path[segment + 1], factor))
}

fn pose_constraint_requires_target(kind: ConstraintType) -> bool {
    matches!(
        kind,
        ConstraintType::CopyLocation
            | ConstraintType::CopyRotation
            | ConstraintType::CopyScale
            | ConstraintType::TrackTo
            | ConstraintType::DampedTrack
            | ConstraintType::LockedTrack
            | ConstraintType::StretchTo
            | ConstraintType::Transformation
            | ConstraintType::Floor
            | ConstraintType::Pivot
            | ConstraintType::Shrinkwrap
            | ConstraintType::ChildOf
            | ConstraintType::Ik
            | ConstraintType::SplineIk
            | ConstraintType::CopyTransforms
            | ConstraintType::LimitDistance
            | ConstraintType::FollowPath
            | ConstraintType::ClampTo
            | ConstraintType::Armature
            | ConstraintType::Action
            | ConstraintType::FollowTrack
            | ConstraintType::ObjectSolver
            | ConstraintType::GeometryAttribute
    )
}

fn constraint_camera_target_id(constraint: &Constraint) -> Result<Option<Id>> {
    if !matches!(
        constraint.constraint_type,
        ConstraintType::FollowTrack | ConstraintType::ObjectSolver
    ) {
        return Ok(None);
    }
    let Some(camera) = constraint.params.get("camera").and_then(Value::as_str) else {
        return Ok(None);
    };
    Id::new(camera.to_owned()).map(Some).map_err(|error| {
        evaluation_error_with(
            "constraint camera ID is invalid",
            json!({"constraint_id":constraint.id,"camera":camera,"reason":error.message}),
        )
    })
}

fn constraint_requires_target(constraint: &Constraint) -> bool {
    pose_constraint_requires_target(constraint.constraint_type)
}

fn ik_target_matrices(
    doc: &SceneDoc,
    armature_node_id: &Id,
    armature: &ArmatureData,
    pose: &BTreeMap<Id, PoseBone>,
    frame: f64,
    armature_world: DMat4,
    world_matrices: &BTreeMap<Id, DMat4>,
    driver_values: &BTreeMap<(Id, String, u32), f64>,
    project_root: Option<&Path>,
    fps: u32,
    fps_base: f64,
) -> Result<BTreeMap<Id, IkGoal>> {
    let owner_node = doc.nodes.get(armature_node_id).ok_or_else(|| {
        evaluation_error_with(
            "IK armature node is unavailable",
            json!({ "armature": armature_node_id }),
        )
    })?;
    let mut targets = BTreeMap::new();
    for constraint in owner_node.constraints.iter().filter(|constraint| {
        constraint.enabled
            && constraint.constraint_type == ConstraintType::Ik
            && constraint.influence != 0.0
    }) {
        let target_id = constraint.target.as_ref().ok_or_else(|| {
            evaluation_error_with(
                "IK constraint requires a target",
                json!({ "constraint_id": constraint.id }),
            )
        })?;
        let target_node = doc.nodes.get(target_id).ok_or_else(|| {
            evaluation_error_with(
                "IK target node does not exist",
                json!({ "constraint_id": constraint.id, "target": target_id }),
            )
        })?;
        let target_object_world = *world_matrices.get(target_id).ok_or_else(|| {
            evaluation_error_with(
                "IK target transform is unavailable",
                json!({ "constraint_id": constraint.id, "target": target_id }),
            )
        })?;
        let mut target_world = target_object_world;
        if let Some(bone_id) = constraint.subtarget.as_ref() {
            target_world = ik_target_bone_matrix(
                target_id,
                bone_id,
                armature_node_id,
                armature,
                pose,
                armature_world,
                doc,
                frame,
                world_matrices,
                driver_values,
                project_root,
                fps,
                fps_base,
            )?;
            target_world =
                constraint_target_bone_matrix(constraint, target_node, doc, target_world)?;
        }
        let target_space = string_param_default(&constraint.params, "target_space", "WORLD")?;
        let target_space_world = constraint_space_world_matrix(
            constraint,
            target_space,
            target_id,
            target_object_world,
            None,
            doc,
            frame,
            world_matrices,
            None,
        )?;
        target_world = checked_inverse(target_space_world, "IK target space")? * target_world;
        let pole = match constraint.params.get("pole_target") {
            Some(value) => {
                let pole_id = value
                    .as_str()
                    .and_then(|value| Id::new(value.to_owned()).ok())
                    .ok_or_else(|| {
                        evaluation_error_with(
                            "IK pole target ID is invalid",
                            json!({ "constraint_id": constraint.id }),
                        )
                    })?;
                let pole_object_world = *world_matrices.get(&pole_id).ok_or_else(|| {
                    evaluation_error_with(
                        "IK pole target transform is unavailable",
                        json!({"constraint_id":constraint.id,"pole_target":pole_id}),
                    )
                })?;
                let pole_world = if let Some(pole_bone_name) = constraint
                    .params
                    .get("pole_subtarget")
                    .and_then(Value::as_str)
                    .filter(|name| !name.is_empty())
                {
                    let pole_node = doc.nodes.get(&pole_id).ok_or_else(|| {
                        evaluation_error_with(
                            "IK pole target node does not exist",
                            json!({"constraint_id":constraint.id,"pole_target":pole_id}),
                        )
                    })?;
                    let pole_armature = pole_node
                        .data
                        .as_ref()
                        .and_then(|data_id| doc.data_blocks.get(data_id))
                        .and_then(|data| data.armature.as_ref())
                        .ok_or_else(|| evaluation_error("IK pole target is not an armature"))?;
                    let pole_bone_id = pole_armature
                        .bones
                        .iter()
                        .find(|(bone_id, bone)| {
                            bone_id.as_str() == pole_bone_name || bone.name == pole_bone_name
                        })
                        .map(|(bone_id, _)| bone_id)
                        .ok_or_else(|| {
                            evaluation_error_with(
                                "IK pole target bone does not exist",
                                json!({"constraint_id":constraint.id,"pole_bone":pole_bone_name}),
                            )
                        })?;
                    let pole_world = ik_target_bone_matrix(
                        &pole_id,
                        pole_bone_id,
                        armature_node_id,
                        armature,
                        pose,
                        armature_world,
                        doc,
                        frame,
                        world_matrices,
                        driver_values,
                        project_root,
                        fps,
                        fps_base,
                    )?;
                    constraint_target_bone_matrix_for_id(
                        constraint,
                        pole_node,
                        doc,
                        pole_world,
                        Some(pole_bone_id),
                    )?
                } else {
                    pole_object_world
                };
                let pole_to_world = constraint_space_world_matrix(
                    constraint,
                    target_space,
                    &pole_id,
                    pole_object_world,
                    None,
                    doc,
                    frame,
                    world_matrices,
                    None,
                )?;
                Some(checked_inverse(pole_to_world, "IK pole target space")? * pole_world)
            }
            None => None,
        };
        targets.insert(
            constraint.id.clone(),
            IkGoal {
                target: target_world,
                pole,
            },
        );
    }
    Ok(targets)
}

#[expect(
    clippy::too_many_arguments,
    reason = "IK target bone sampling requires the complete armature evaluation context"
)]
fn ik_target_bone_matrix(
    target_id: &Id,
    bone_id: &Id,
    owner_id: &Id,
    owner_armature: &ArmatureData,
    owner_pose: &BTreeMap<Id, PoseBone>,
    owner_world: DMat4,
    doc: &SceneDoc,
    frame: f64,
    world_matrices: &BTreeMap<Id, DMat4>,
    driver_values: &BTreeMap<(Id, String, u32), f64>,
    project_root: Option<&Path>,
    fps: u32,
    fps_base: f64,
) -> Result<DMat4> {
    if target_id == owner_id {
        return evaluate_bone_matrices(owner_armature, owner_pose)?
            .get(bone_id)
            .map(|matrices| owner_world * matrices.pose)
            .ok_or_else(|| evaluation_error("IK target bone does not exist"));
    }
    let target_node = doc.nodes.get(target_id).ok_or_else(|| {
        evaluation_error_with(
            "IK target node does not exist",
            json!({ "target": target_id }),
        )
    })?;
    let target_world = *world_matrices.get(target_id).ok_or_else(|| {
        evaluation_error_with(
            "IK target transform is unavailable",
            json!({ "target": target_id }),
        )
    })?;
    evaluate_armature_bone_world_matrices_with_resources(
        doc,
        target_id,
        frame,
        target_world,
        world_matrices,
        driver_values,
        project_root,
        fps,
        fps_base,
    )?
    .get(bone_id)
    .copied()
    .ok_or_else(|| {
        evaluation_error_with(
            "IK target bone does not exist",
            json!({"target":target_node.name,"bone_id":bone_id}),
        )
    })
}

/// Evaluate Blender's legacy `QJacobian` IK constraints using world-space goals keyed by ID.
///
/// Each constraint builds a root-to-tip segment chain; `chain_count` defaults to the full
/// parent chain and the legacy solver's outer convergence threshold is fixed at 1e-3.
#[expect(
    clippy::too_many_lines,
    reason = "pose-bone constraints share one ordered transform path"
)]
fn apply_pose_bone_constraints(
    armature_node_id: &Id,
    armature: &ArmatureData,
    constraints: &[Constraint],
    doc: &SceneDoc,
    frame: f64,
    armature_world: DMat4,
    target_worlds: &BTreeMap<Id, DMat4>,
    evaluated_meshes: Option<&BTreeMap<Id, Mesh>>,
    driver_values: &BTreeMap<(Id, String, u32), f64>,
    project_root: Option<&Path>,
    fps: u32,
    fps_base: f64,
    matrices: &mut BTreeMap<Id, BoneMatrices>,
) -> Result<()> {
    if !constraints.iter().any(|constraint| {
        constraint.owner_bone.is_some()
            && constraint.enabled
            && constraint.influence != 0.0
            && !matches!(
                constraint.constraint_type,
                ConstraintType::Ik | ConstraintType::SplineIk
            )
    }) {
        return Ok(());
    }
    let mut pending = armature.bones.keys().cloned().collect::<BTreeSet<_>>();
    let mut completed = BTreeSet::new();
    let mut order = Vec::with_capacity(pending.len());
    while !pending.is_empty() {
        let next = pending
            .iter()
            .find(|bone_id| {
                armature.bones.get(*bone_id).is_some_and(|bone| {
                    bone.parent
                        .as_ref()
                        .is_none_or(|parent| completed.contains(parent))
                })
            })
            .cloned()
            .ok_or_else(|| evaluation_error("armature bone parent cycle"))?;
        pending.remove(&next);
        completed.insert(next.clone());
        order.push(next);
    }
    let armature_world_inverse = checked_inverse(armature_world, "armature world")?;
    let needs_armature_inverse = constraints.iter().any(|constraint| {
        constraint.owner_bone.is_some()
            && constraint.enabled
            && constraint.influence != 0.0
            && !matches!(
                constraint.constraint_type,
                ConstraintType::Ik | ConstraintType::SplineIk
            )
            && (constraint.constraint_type == ConstraintType::TransformCache
                || constraint
                    .target
                    .as_ref()
                    .is_some_and(|target| target != armature_node_id)
                || matches!(
                    constraint.constraint_type,
                    ConstraintType::CameraSolver
                        | ConstraintType::FollowTrack
                        | ConstraintType::ObjectSolver
                ) && constraint
                    .params
                    .get("camera")
                    .and_then(Value::as_str)
                    .is_some_and(|target| target != armature_node_id.as_str()))
    });
    let armature_inverse = if needs_armature_inverse {
        armature_world_inverse
    } else {
        DMat4::IDENTITY
    };
    for (owner_index, owner_bone) in order.iter().enumerate() {
        for constraint in constraints.iter().filter(|constraint| {
            constraint.owner_bone.as_ref() == Some(owner_bone)
                && constraint.enabled
                && constraint.influence != 0.0
                && !matches!(
                    constraint.constraint_type,
                    ConstraintType::Ik | ConstraintType::SplineIk
                )
        }) {
            let source_pose = matrices
                .get(owner_bone)
                .copied()
                .ok_or_else(|| evaluation_error("pose constraint owner bone matrix is missing"))?
                .pose;
            let constraint_target_id = constraint
                .target
                .clone()
                .or(constraint_camera_target_id(constraint)?);
            let legacy_space = constraint.constraint_type == ConstraintType::TransformCache;
            let target_matrix = if constraint.constraint_type == ConstraintType::TransformCache {
                transform_cache_matrix(doc, constraint, project_root, frame, fps, fps_base)?
            } else if let Some(target_id) = &constraint_target_id {
                let target_world = if target_id == armature_node_id {
                    armature_world
                } else {
                    target_worlds.get(target_id).copied().ok_or_else(|| {
                        evaluation_error_with(
                            "pose constraint target transform is unavailable",
                            json!({"node": armature_node_id, "constraint_id": constraint.id, "target": target_id}),
                        )
                    })?
                };
                let target_world = if let Some(target_bone_id) = &constraint.subtarget {
                    if target_id == armature_node_id {
                        armature_world
                            * matrices
                                .get(target_bone_id)
                                .copied()
                                .ok_or_else(|| {
                                    evaluation_error("pose constraint target bone is missing")
                                })?
                                .pose
                    } else {
                        evaluate_armature_bone_world_matrices_with_meshes(
                            doc,
                            target_id,
                            frame,
                            target_world,
                            target_worlds,
                            driver_values,
                            project_root,
                            fps,
                            fps_base,
                            evaluated_meshes,
                        )?
                        .get(target_bone_id)
                        .copied()
                        .ok_or_else(|| evaluation_error("pose constraint target bone is missing"))?
                    }
                } else {
                    target_world
                };
                let target_node = doc.nodes.get(target_id).ok_or_else(|| {
                    evaluation_error_with(
                        "pose constraint target node does not exist",
                        json!({"target":target_id,"constraint_id":constraint.id}),
                    )
                })?;
                let target_world =
                    constraint_target_bone_matrix(constraint, target_node, doc, target_world)?;
                if legacy_space && target_id != armature_node_id {
                    armature_inverse * target_world
                } else {
                    target_world
                }
            } else if constraint_requires_target(constraint) {
                return Err(evaluation_error_with(
                    "enabled pose constraint has no target",
                    json!({"node": armature_node_id, "constraint_id": constraint.id}),
                ));
            } else {
                DMat4::IDENTITY
            };
            let constraint_source = if legacy_space {
                source_pose
            } else {
                armature_world * source_pose
            };
            let constraint_world_to_owner = if legacy_space {
                armature_inverse
            } else {
                armature_world_inverse
            };
            let constrained = apply_object_constraint(
                constraint_source,
                target_matrix,
                constraint,
                doc,
                frame,
                target_worlds,
                evaluated_meshes,
                Some(armature_node_id),
                armature_node_id,
                armature_world,
                Some(matrices),
                Some(constraint_world_to_owner),
            )?;
            let mut constrained_pose = if legacy_space {
                constrained
            } else {
                armature_world_inverse * constrained
            };
            // Connected pose bones stay anchored to their parent tail when constraints run.
            if armature
                .bones
                .get(owner_bone)
                .is_some_and(|bone| bone.use_connect)
            {
                constrained_pose.w_axis = source_pose.w_axis;
            }
            let delta = constrained_pose * checked_inverse(source_pose, "pose bone")?;
            for descendant_id in order.iter().skip(owner_index) {
                let mut cursor = Some(descendant_id);
                let mut affected = false;
                while let Some(current_id) = cursor {
                    if current_id == owner_bone {
                        affected = true;
                        break;
                    }
                    cursor = armature
                        .bones
                        .get(current_id)
                        .and_then(|bone| bone.parent.as_ref());
                }
                if !affected {
                    continue;
                }
                let current = matrices
                    .get(descendant_id)
                    .copied()
                    .ok_or_else(|| evaluation_error("pose descendant bone matrix is missing"))?;
                let pose = delta * current.pose;
                let skin = pose * checked_inverse(current.rest, "bone rest")?;
                let current = matrices
                    .get_mut(descendant_id)
                    .ok_or_else(|| evaluation_error("pose descendant bone matrix disappeared"))?;
                current.pose = pose;
                current.skin = skin;
            }
        }
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "pose evaluation needs the rig, canonical constraints, and their frame context"
)]
fn evaluate_pose_matrices(
    armature_node_id: &Id,
    armature: &ArmatureData,
    pose: &BTreeMap<Id, PoseBone>,
    constraints: &[Constraint],
    doc: &SceneDoc,
    frame: f64,
    armature_world: DMat4,
    target_worlds: &BTreeMap<Id, DMat4>,
    driver_values: &BTreeMap<(Id, String, u32), f64>,
    ik_goals: &BTreeMap<Id, IkGoal>,
    project_root: Option<&Path>,
    fps: u32,
    fps_base: f64,
    evaluated_meshes: Option<&BTreeMap<Id, Mesh>>,
) -> Result<BTreeMap<Id, BoneMatrices>> {
    let mut result_pose = pose.clone();
    for bone_id in armature.bones.keys() {
        result_pose.entry(bone_id.clone()).or_default();
    }
    let mut pose_matrix_overrides = BTreeMap::new();
    for constraint in constraints.iter().filter(|constraint| {
        constraint.enabled
            && constraint.constraint_type == ConstraintType::Ik
            && constraint.influence != 0.0
    }) {
        let Some(end_id) = constraint
            .owner_bone
            .as_ref()
            .or(constraint.subtarget.as_ref())
        else {
            return Err(evaluation_error_with(
                "IK constraint requires an owner bone or bone subtarget",
                json!({ "constraint_id": constraint.id }),
            ));
        };
        let target_id = constraint.target.as_ref().ok_or_else(|| {
            evaluation_error_with(
                "IK constraint requires a target",
                json!({ "constraint_id": constraint.id }),
            )
        })?;
        let influence = checked_influence(constraint)?;
        let goal = ik_goals.get(&constraint.id).copied().ok_or_else(|| {
            evaluation_error_with(
                "IK target transform is unavailable",
                json!({ "constraint_id": constraint.id, "target": target_id }),
            )
        })?;
        let use_location = param_bool(&constraint.params, "use_location", true)?;
        let use_rotation = param_bool(&constraint.params, "use_rotation", false)?;
        if !use_location && !use_rotation {
            continue;
        }
        let chain_count =
            param_usize(&constraint.params, "chain_count")?.filter(|count| *count > 0);
        solve_legacy_ik(
            armature,
            &result_pose,
            &mut pose_matrix_overrides,
            end_id,
            armature_world,
            &goal,
            chain_count,
            param_bool(&constraint.params, "use_tail", true)?,
            param_bool(&constraint.params, "use_stretch", true)?,
            param_usize(&constraint.params, "iterations")?.unwrap_or(500),
            number_param(&constraint.params, "pole_angle")?.unwrap_or(0.0),
            use_location,
            use_rotation,
            number_param(&constraint.params, "weight")?.unwrap_or(1.0),
            number_param(&constraint.params, "orient_weight")?.unwrap_or(1.0),
            influence,
        )?;
    }
    let mut matrices =
        evaluate_bone_matrices_with_ik_overrides(armature, &result_pose, &pose_matrix_overrides)?;
    apply_pose_bone_constraints(
        armature_node_id,
        armature,
        constraints,
        doc,
        frame,
        armature_world,
        target_worlds,
        evaluated_meshes,
        driver_values,
        project_root,
        fps,
        fps_base,
        &mut matrices,
    )?;
    Ok(matrices)
}

fn blender_ik_mat4(matrix: DMat4) -> Mat4 {
    Mat4::from_cols_array(&matrix.to_cols_array().map(|value| value as f32))
}
fn double_ik_mat4(matrix: Mat4) -> DMat4 {
    DMat4::from_cols_array(&matrix.to_cols_array().map(f64::from))
}

fn blender_ik_mat3(matrix: DMat3) -> Mat3 {
    Mat3::from_cols_array(&matrix.to_cols_array().map(|value| value as f32))
}

fn double_ik_mat3(matrix: Mat3) -> DMat3 {
    DMat3::from_cols_array(&matrix.to_cols_array().map(f64::from))
}
fn blend_ik_basis_change(matrix: Mat3, factor: f32) -> Mat3 {
    use glam::Vec3;
    let (scale, rotation, _) = Mat4::from_mat3(matrix).to_scale_rotation_translation();
    let rotation = glam::Quat::IDENTITY.slerp(rotation, factor).normalize();
    let scale = Vec3::ONE.lerp(scale, factor);
    Mat3::from_quat(rotation) * Mat3::from_diagonal(scale)
}

fn normalized_ik_basis(matrix: Mat4) -> Mat3 {
    Mat3::from_cols(
        matrix.x_axis.truncate().normalize_or_zero(),
        matrix.y_axis.truncate().normalize_or_zero(),
        matrix.z_axis.truncate().normalize_or_zero(),
    )
}

fn finite_ik_mat4(matrix: Mat4) -> bool {
    matrix.to_cols_array().iter().all(|value| value.is_finite())
}

fn checked_ik_inverse(matrix: Mat4, what: &str) -> Result<Mat4> {
    if !finite_ik_mat4(matrix) {
        return Err(evaluation_error(format!("{what} matrix is non-finite")));
    }
    let inverse = matrix.inverse();
    if !finite_ik_mat4(inverse) {
        return Err(evaluation_error(format!(
            "{what} matrix inverse is non-finite"
        )));
    }
    Ok(inverse)
}

fn blender_ik_bone_length(bone: &crate::model::Bone) -> f32 {
    use glam::Vec3;

    let head = Vec3::from_array(bone.head.map(|value| value as f32));
    let tail = Vec3::from_array(bone.tail.map(|value| value as f32));
    (tail - head).length()
}

#[expect(
    clippy::too_many_arguments,
    reason = "the legacy IK adapter needs the complete constraint and pose evaluation state"
)]
#[expect(
    clippy::fn_params_excessive_bools,
    reason = "matches Blender's independent IK constraint option fields"
)]
#[expect(
    clippy::float_cmp,
    reason = "exact Blender influence endpoints control pole blending"
)]
fn solve_legacy_ik(
    armature: &ArmatureData,
    pose: &BTreeMap<Id, PoseBone>,
    pose_matrix_overrides: &mut BTreeMap<Id, DMat4>,
    end_id: &Id,
    armature_world: DMat4,
    goal: &IkGoal,
    chain_count: Option<usize>,
    use_tail: bool,
    use_stretch: bool,
    iterations: usize,
    pole_angle: f64,
    use_location: bool,
    use_rotation: bool,
    position_weight: f64,
    orientation_weight: f64,
    influence: f64,
) -> Result<()> {
    use crate::eval::ik_solver::{LegacyGoal, LegacySegment, solve_legacy};
    use glam::Vec3;

    if !pole_angle.is_finite()
        || !position_weight.is_finite()
        || !orientation_weight.is_finite()
        || !influence.is_finite()
        || !finite_matrix(goal.target)
        || goal.pole.is_some_and(|matrix| !finite_matrix(matrix))
    {
        return Err(evaluation_error("IK goal settings must be finite"));
    }
    let selected_tip = if use_tail {
        end_id
    } else {
        armature
            .bones
            .get(end_id)
            .and_then(|bone| bone.parent.as_ref())
            .unwrap_or(end_id)
    };
    let maximum = chain_count.unwrap_or(255).min(255);
    if maximum == 0 {
        return Err(evaluation_error(
            "IK chain_count must select at least one bone",
        ));
    }
    let mut chain = Vec::with_capacity(maximum);
    let mut cursor = Some(selected_tip);
    while let Some(bone_id) = cursor {
        if chain.len() >= maximum {
            break;
        }
        let bone = armature.bones.get(bone_id).ok_or_else(|| {
            evaluation_error_with("IK chain bone does not exist", json!({"bone_id":bone_id}))
        })?;
        chain.push(bone_id.clone());
        cursor = bone.parent.as_ref();
    }
    if chain.is_empty() {
        return Err(evaluation_error(
            "IK chain_count must select at least one bone",
        ));
    }
    chain.reverse();

    let original_matrices =
        evaluate_bone_matrices_with_ik_overrides(armature, pose, pose_matrix_overrides)?;
    let root_id = chain
        .first()
        .ok_or_else(|| evaluation_error("IK chain has no root bone"))?;
    let root_bone = armature
        .bones
        .get(root_id)
        .ok_or_else(|| evaluation_error("IK root bone is missing"))?;
    let root_pose = original_matrices
        .get(root_id)
        .copied()
        .ok_or_else(|| evaluation_error("IK root pose matrix is missing"))?;
    let root_parent_basis = if let Some(parent_id) = &root_bone.parent {
        let parent_pose = original_matrices
            .get(parent_id)
            .copied()
            .ok_or_else(|| evaluation_error("IK root parent pose matrix is missing"))?;
        normalized_ik_basis(blender_ik_mat4(parent_pose.pose))
    } else {
        Mat3::IDENTITY
    };
    let root_head = blender_ik_mat4(root_pose.pose).w_axis.truncate();
    let root_matrix = Mat4::from_cols(
        root_parent_basis.x_axis.extend(0.0),
        root_parent_basis.y_axis.extend(0.0),
        root_parent_basis.z_axis.extend(0.0),
        root_head.extend(1.0),
    );
    let object_root = blender_ik_mat4(armature_world) * root_matrix;
    let goal_inverse = checked_ik_inverse(object_root, "IK root world")?;
    let target_matrix = goal_inverse * blender_ik_mat4(goal.target);
    if !finite_ik_mat4(target_matrix) {
        return Err(evaluation_error("IK target matrix is non-finite"));
    }
    let pole_position = goal
        .pole
        .map(|matrix| goal_inverse * blender_ik_mat4(matrix))
        .map(|matrix| {
            if finite_ik_mat4(matrix) {
                Ok(DVec3::new(
                    f64::from(matrix.w_axis.x),
                    f64::from(matrix.w_axis.y),
                    f64::from(matrix.w_axis.z),
                ))
            } else {
                Err(evaluation_error("IK pole matrix is non-finite"))
            }
        })
        .transpose()?;

    let mut segments = Vec::with_capacity(chain.len());
    for (index, bone_id) in chain.iter().enumerate() {
        let bone = armature
            .bones
            .get(bone_id)
            .ok_or_else(|| evaluation_error("IK chain bone disappeared"))?;
        let matrices = original_matrices
            .get(bone_id)
            .copied()
            .ok_or_else(|| evaluation_error("IK chain pose matrix is missing"))?;
        let pose_matrix = blender_ik_mat4(matrices.pose);
        let bone_basis = normalized_ik_basis(pose_matrix);
        let rest_matrix = blender_ik_mat4(matrices.rest);
        let armature_rest_basis = normalized_ik_basis(rest_matrix);
        let rest_basis = if let Some(parent_id) = &bone.parent {
            let parent_rest = original_matrices
                .get(parent_id)
                .copied()
                .ok_or_else(|| evaluation_error("IK chain parent rest matrix is missing"))?;
            normalized_ik_basis(blender_ik_mat4(parent_rest.rest)).transpose() * armature_rest_basis
        } else {
            armature_rest_basis
        };
        let parent_pose_matrix = if let Some(parent_id) = &bone.parent {
            let parent = original_matrices
                .get(parent_id)
                .copied()
                .ok_or_else(|| evaluation_error("IK chain parent pose matrix is missing"))?;
            blender_ik_mat4(parent.pose)
        } else {
            Mat4::IDENTITY
        };
        let parent_basis = normalized_ik_basis(parent_pose_matrix);
        let inverse_parent_basis = parent_basis.inverse();
        let start = if index == 0 {
            Vec3::ZERO
        } else {
            let parent_id = bone
                .parent
                .as_ref()
                .ok_or_else(|| evaluation_error("IK chain is not a single parent path"))?;
            let parent = armature
                .bones
                .get(parent_id)
                .ok_or_else(|| evaluation_error("IK chain parent bone is missing"))?;
            let parent_tail = parent_pose_matrix.w_axis.truncate()
                + parent_pose_matrix.y_axis.truncate() * blender_ik_bone_length(parent);
            inverse_parent_basis * (pose_matrix.w_axis.truncate() - parent_tail)
        };
        let length = blender_ik_bone_length(bone) * pose_matrix.y_axis.truncate().length();
        let basis = rest_basis.transpose() * (inverse_parent_basis * bone_basis);
        let pose_settings = pose.get(bone_id).copied().unwrap_or_default();
        if !pose_settings.ik_stretch.is_finite()
            || pose_settings
                .ik_min
                .iter()
                .chain(pose_settings.ik_max.iter())
                .chain(pose_settings.ik_stiffness.iter())
                .any(|value| !value.is_finite())
            || !length.is_finite()
            || !finite_ik_mat4(pose_matrix)
            || !finite_ik_mat4(rest_matrix)
        {
            return Err(evaluation_error_with(
                "IK bone settings or transform are non-finite",
                json!({"bone_id":bone_id}),
            ));
        }
        segments.push(LegacySegment {
            start: DVec3::new(f64::from(start.x), f64::from(start.y), f64::from(start.z)),
            rest_basis: double_ik_mat3(rest_basis),
            basis: double_ik_mat3(basis),
            length: f64::from(length),
            locked: pose_settings.lock_ik,
            use_limits: pose_settings.use_ik_limit,
            limits_min: pose_settings.ik_min.map(|value| f64::from(value as f32)),
            limits_max: pose_settings.ik_max.map(|value| f64::from(value as f32)),
            stiffness: pose_settings
                .ik_stiffness
                .map(|value| f64::from(value as f32)),
            ik_stretch: f64::from(pose_settings.ik_stretch as f32),
        });
    }
    if ((use_location && position_weight > 0.0) || (use_rotation && orientation_weight > 0.0))
        && let Some((bone_id, _)) = chain.iter().zip(&segments).find(|(_, segment)| {
            !segment.locked[0]
                && segment.locked[1]
                && !segment.locked[2]
                && segment.use_limits[0]
                && segment.use_limits[2]
                && segment.limits_min[0] <= segment.limits_max[0]
                && segment.limits_min[2] <= segment.limits_max[2]
        })
    {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "IK X/Z Swing ellipse limits are unsupported",
            json!({
                "feature_id":"constraint.ik.swing_ellipse_limit",
                "status":"not_supported",
                "bone_id":bone_id,
            }),
        ));
    }
    let target_basis = normalized_ik_basis(target_matrix);
    let solver_goal = LegacyGoal {
        position: DVec3::new(
            f64::from(target_matrix.w_axis.x),
            f64::from(target_matrix.w_axis.y),
            f64::from(target_matrix.w_axis.z),
        ),
        rotation: double_ik_mat3(target_basis),
        pole: pole_position,
        pole_angle: f64::from(pole_angle as f32),
        use_location,
        use_rotation,
        use_stretch,
        position_weight: f64::from(position_weight as f32),
        orientation_weight: f64::from(orientation_weight as f32),
        influence: f64::from(influence as f32),
        use_sdls: true,
    };
    let result = solve_legacy(&segments, solver_goal, iterations);
    if result.basis_changes.len() != chain.len() || result.stretch_ratios.len() != chain.len() {
        return Err(evaluation_error(
            "IK solver returned a mismatched chain result",
        ));
    }

    for (index, bone_id) in chain.iter().enumerate() {
        let bone = armature
            .bones
            .get(bone_id)
            .ok_or_else(|| evaluation_error("IK result bone disappeared"))?;
        let original = original_matrices
            .get(bone_id)
            .copied()
            .ok_or_else(|| evaluation_error("IK original bone matrix is missing"))?;
        let original_pose = blender_ik_mat4(original.pose);
        let channel_matrix = if let Some(parent_id) = &bone.parent {
            let parent = original_matrices
                .get(parent_id)
                .copied()
                .ok_or_else(|| evaluation_error("IK original parent matrix is missing"))?;
            checked_ik_inverse(blender_ik_mat4(parent.pose), "IK original parent pose")?
                * original_pose
        } else {
            original_pose
        };
        let current_matrices =
            evaluate_bone_matrices_with_ik_overrides(armature, pose, pose_matrix_overrides)?;
        let parent_pose = if let Some(parent_id) = &bone.parent {
            let parent = current_matrices
                .get(parent_id)
                .copied()
                .ok_or_else(|| evaluation_error("IK updated parent matrix is missing"))?;
            blender_ik_mat4(parent.pose)
        } else {
            Mat4::IDENTITY
        };
        let stretch = result.stretch_ratios[index] as f32;
        if !stretch.is_finite() {
            return Err(evaluation_error_with(
                "IK stretch ratio is non-finite",
                json!({"bone_id":bone_id}),
            ));
        }
        let mut basis_change = blender_ik_mat3(result.basis_changes[index]);
        basis_change.x_axis *= stretch;
        basis_change.y_axis *= stretch;
        basis_change.z_axis *= stretch;
        if goal.pole.is_some() && influence != 1.0 {
            basis_change = blend_ik_basis_change(basis_change, influence as f32);
        }
        let ik_matrix = Mat4::from_cols(
            basis_change.x_axis.extend(0.0),
            basis_change.y_axis.extend(0.0),
            basis_change.z_axis.extend(0.0),
            Vec3::ZERO.extend(1.0),
        );
        let desired_pose = if bone.parent.is_some() {
            parent_pose * channel_matrix * ik_matrix
        } else {
            original_pose * ik_matrix
        };
        if !finite_ik_mat4(desired_pose) {
            return Err(evaluation_error_with(
                "IK result pose matrix is non-finite",
                json!({"bone_id":bone_id}),
            ));
        }

        pose_matrix_overrides.insert(bone_id.clone(), double_ik_mat4(desired_pose));
    }
    Ok(())
}

/// Blend shape keys using caller-evaluated animated or driven key values.
pub(crate) fn apply_shape_keys_with_values(
    mesh: &mut Mesh,
    shape_keys: &ShapeKeyData,
    data_block: &DataBlock,
    effective_values: &BTreeMap<Id, f64>,
) -> Result<()> {
    for (key_id, key) in &shape_keys.keys {
        if let Some(group_id) = &key.vertex_group
            && !data_block
                .vertex_groups
                .iter()
                .any(|group| &group.id == group_id)
        {
            return Err(evaluation_error_with(
                "shape key vertex group does not exist",
                json!({ "key_id": key_id, "vertex_group": group_id }),
            ));
        }
    }
    for vertex in &mut mesh.vertices {
        let base = shape_keys
            .basis
            .get(&vertex.id)
            .copied()
            .map_or(vertex.co, DVec3::from_array);
        if !base.is_finite() {
            return Err(evaluation_error_with(
                "shape key basis position is non-finite",
                json!({ "vertex_id": vertex.id }),
            ));
        }
        let mut evaluated = base;
        for (key_id, key) in &shape_keys.keys {
            if key.mute {
                continue;
            }
            let key_value = effective_values.get(key_id).copied().unwrap_or(key.value);
            if !key_value.is_finite() {
                return Err(evaluation_error_with(
                    "shape key value is non-finite",
                    json!({ "key_id": key_id }),
                ));
            }
            let group_weight = if let Some(group_id) = &key.vertex_group {
                data_block
                    .vertex_weights
                    .get(&vertex.id)
                    .and_then(|weights| weights.get(group_id))
                    .copied()
                    .unwrap_or(0.0)
            } else {
                1.0
            };
            if !group_weight.is_finite() || group_weight < 0.0 {
                return Err(evaluation_error_with(
                    "shape key vertex group weight is invalid",
                    json!({ "key_id": key_id, "vertex_id": vertex.id }),
                ));
            }
            let relative = if let Some(relative_id) = &key.relative_key {
                let relative_key = shape_keys.keys.get(relative_id).ok_or_else(|| {
                    evaluation_error_with(
                        "shape key relative reference does not exist",
                        json!({ "key_id": key_id, "relative_key": relative_id }),
                    )
                })?;
                relative_key
                    .positions
                    .get(&vertex.id)
                    .copied()
                    .or_else(|| shape_keys.basis.get(&vertex.id).copied())
                    .unwrap_or(base.to_array())
            } else {
                shape_keys
                    .basis
                    .get(&vertex.id)
                    .copied()
                    .unwrap_or(base.to_array())
            };
            let destination = key.positions.get(&vertex.id).copied().unwrap_or(relative);
            let relative = DVec3::from_array(relative);
            let destination = DVec3::from_array(destination);
            if !relative.is_finite() || !destination.is_finite() {
                return Err(evaluation_error_with(
                    "shape key position is non-finite",
                    json!({ "key_id": key_id, "vertex_id": vertex.id }),
                ));
            }
            evaluated += (destination - relative) * (key_value * group_weight);
        }
        if !evaluated.is_finite() {
            return Err(evaluation_error_with(
                "evaluated shape key position is non-finite",
                json!({ "vertex_id": vertex.id }),
            ));
        }
        vertex.co = evaluated;
    }
    Ok(())
}

/// Sample shape-key action channels and then override them with evaluated drivers.
pub(crate) fn evaluate_shape_key_values(
    owner_id: &Id,
    node: &Node,
    shape_keys: &ShapeKeyData,
    doc: &SceneDoc,
    frame: f64,
    driver_values: &BTreeMap<(Id, String, u32), f64>,
) -> Result<BTreeMap<Id, f64>> {
    if !frame.is_finite() {
        return Err(evaluation_error("shape key evaluation frame is non-finite"));
    }
    for (key_id, key) in &shape_keys.keys {
        if !key.slider_min.is_finite()
            || !key.slider_max.is_finite()
            || key.slider_min > key.slider_max
        {
            return Err(evaluation_error_with(
                "shape key slider range is invalid",
                json!({"key_id":key_id,"slider_min":key.slider_min,"slider_max":key.slider_max}),
            ));
        }
    }
    let mut values: BTreeMap<Id, f64> = shape_keys
        .keys
        .iter()
        .map(|(key_id, key)| (key_id.clone(), key.value))
        .collect();
    if let Some(action_id) = &node.action {
        sample_shape_key_action_values(action_id, shape_keys, doc, frame, &mut values)?;
    }
    if let Some(action_id) = shape_keys.action.as_ref()
        && node.action.as_ref() != Some(action_id)
    {
        sample_shape_key_action_values(action_id, shape_keys, doc, frame, &mut values)?;
    }
    for (key_id, value) in shape_key_driver_values(shape_keys, owner_id, driver_values)? {
        values.insert(key_id, value);
    }
    for (key_id, value) in &values {
        if !value.is_finite() {
            return Err(evaluation_error_with(
                "shape key value is non-finite",
                json!({ "key_id": key_id }),
            ));
        }
    }
    if shape_keys.absolute {
        let evaluation_time =
            shape_key_evaluation_time(owner_id, node, shape_keys, doc, frame, driver_values)?;
        values = interpolate_absolute_shape_keys(shape_keys, evaluation_time)?;
    } else {
        for (key_id, value) in &mut values {
            let key = shape_keys.keys.get(key_id).ok_or_else(|| {
                evaluation_error_with(
                    "evaluated shape key does not exist",
                    json!({ "key_id": key_id }),
                )
            })?;
            *value = value.clamp(key.slider_min, key.slider_max);
        }
    }
    Ok(values)
}
fn sample_shape_key_action_values(
    action_id: &Id,
    shape_keys: &ShapeKeyData,
    doc: &SceneDoc,
    frame: f64,
    values: &mut BTreeMap<Id, f64>,
) -> Result<()> {
    let action = doc.actions.get(action_id).ok_or_else(|| {
        evaluation_error_with(
            "shape key animation action does not exist",
            json!({ "action": action_id }),
        )
    })?;
    for curve in &action.fcurves {
        if shape_keys.action.as_ref() == Some(action_id)
            && shape_keys.muted_action_curves.contains(&curve.path)
        {
            continue;
        }
        let Some(key_id) = shape_key_id_from_path(shape_keys, &curve.path) else {
            continue;
        };
        if curve.index != 0 {
            return Err(evaluation_error_with(
                "shape key value curve index must be zero",
                json!({"action":action_id,"path":curve.path,"index":curve.index}),
            ));
        }
        if let Some(value) = crate::eval::animation::sample_curve(curve, frame)? {
            values.insert(key_id, value);
        }
    }
    Ok(())
}

fn shape_key_driver_values(
    shape_keys: &ShapeKeyData,
    owner_id: &Id,
    driver_values: &BTreeMap<(Id, String, u32), f64>,
) -> Result<BTreeMap<Id, f64>> {
    let mut driven = BTreeMap::new();
    for (key_id, key) in &shape_keys.keys {
        let id_path = format!("shape_key.{key_id}.value");
        let plural_id_path = format!("shape_keys.{key_id}.value");
        let name_path = format!("key_blocks[\"{}\"].value", key.name);
        let value = driver_values
            .get(&(owner_id.clone(), id_path, 0))
            .or_else(|| driver_values.get(&(owner_id.clone(), plural_id_path, 0)))
            .or_else(|| driver_values.get(&(owner_id.clone(), name_path, 0)));
        if let Some(value) = value {
            if !value.is_finite() {
                return Err(evaluation_error_with(
                    "shape key driver result is non-finite",
                    json!({ "key_id": key_id }),
                ));
            }
            driven.insert(key_id.clone(), *value);
        }
    }
    Ok(driven)
}

fn shape_key_id_from_path(shape_keys: &ShapeKeyData, path: &str) -> Option<Id> {
    let path_id = path
        .strip_prefix("shape_key.")
        .or_else(|| path.strip_prefix("shape_keys."))
        .and_then(|value| value.strip_suffix(".value"));
    if let Some(id) = path_id.and_then(|value| Id::new(value).ok())
        && shape_keys.keys.contains_key(&id)
    {
        return Some(id);
    }
    let start = path.find("key_blocks[")? + "key_blocks[".len();
    let end = path[start..].find(']')? + start;
    let name = path[start..end].trim_matches(['"', '\'']);
    shape_keys
        .keys
        .iter()
        .find(|(_, key)| key.name == name)
        .map(|(key_id, _)| key_id.clone())
}
fn shape_key_evaluation_time(
    owner_id: &Id,
    node: &Node,
    shape_keys: &ShapeKeyData,
    doc: &SceneDoc,
    frame: f64,
    driver_values: &BTreeMap<(Id, String, u32), f64>,
) -> Result<f64> {
    let mut evaluation_time = shape_keys.evaluation_time;
    for action_id in [node.action.as_ref(), shape_keys.action.as_ref()]
        .into_iter()
        .flatten()
    {
        let action = doc.actions.get(action_id).ok_or_else(|| {
            evaluation_error_with(
                "shape key animation action does not exist",
                json!({ "action": action_id }),
            )
        })?;
        for curve in &action.fcurves {
            if shape_keys.action.as_ref() == Some(action_id)
                && shape_keys.muted_action_curves.contains(&curve.path)
            {
                continue;
            }
            if !matches!(
                curve.path.as_str(),
                "shape_keys.evaluation_time" | "shape_key.evaluation_time" | "key_blocks.eval_time"
            ) {
                continue;
            }
            if curve.index != 0 {
                return Err(evaluation_error_with(
                    "shape key evaluation-time curve index must be zero",
                    json!({"action":action_id,"path":curve.path,"index":curve.index}),
                ));
            }
            if let Some(value) = crate::eval::animation::sample_curve(curve, frame)? {
                evaluation_time = value;
            }
        }
    }
    for path in [
        "shape_keys.evaluation_time",
        "shape_key.evaluation_time",
        "key_blocks.eval_time",
    ] {
        if let Some(value) = driver_values.get(&(owner_id.clone(), path.to_owned(), 0)) {
            evaluation_time = *value;
        }
    }
    if !evaluation_time.is_finite() {
        return Err(evaluation_error("shape key evaluation time is non-finite"));
    }
    Ok(evaluation_time)
}

fn interpolate_absolute_shape_keys(
    shape_keys: &ShapeKeyData,
    evaluation_time: f64,
) -> Result<BTreeMap<Id, f64>> {
    if !evaluation_time.is_finite() {
        return Err(evaluation_error("shape key evaluation time is non-finite"));
    }
    let mut keys = shape_keys.keys.iter().collect::<Vec<_>>();
    if keys.iter().any(|(_, key)| !key.frame.is_finite()) {
        return Err(evaluation_error("absolute shape key frame is non-finite"));
    }
    keys.sort_by(|(first_id, first), (second_id, second)| {
        first
            .frame
            .total_cmp(&second.frame)
            .then_with(|| first_id.cmp(second_id))
    });
    let mut values = BTreeMap::new();
    let Some((first_id, first_key)) = keys.first() else {
        return Ok(values);
    };
    if evaluation_time <= first_key.frame || keys.len() == 1 {
        values.insert((*first_id).clone(), 1.0);
        return Ok(values);
    }
    let Some((last_id, last_key)) = keys.last() else {
        return Ok(values);
    };
    if evaluation_time >= last_key.frame {
        values.insert((*last_id).clone(), 1.0);
        return Ok(values);
    }
    for pair in keys.windows(2) {
        let (left_id, left) = pair[0];
        let (right_id, right) = pair[1];
        if evaluation_time <= right.frame {
            let span = right.frame - left.frame;
            let factor = if span <= f64::EPSILON {
                1.0
            } else {
                (evaluation_time - left.frame) / span
            };
            values.insert(left_id.clone(), 1.0 - factor);
            values.insert(right_id.clone(), factor);
            return Ok(values);
        }
    }
    Ok(values)
}

/// Apply one Armature modifier's linear-blend-skinning transform in mesh-local space.
///
/// Vertex group names bind to deform bone names. When `use_vertex_groups` is false, all deforming
/// bones contribute equally to every vertex; otherwise per-vertex group weights are normalized.
pub(crate) fn apply_armature_skinning(
    mesh: &mut Mesh,
    data_block: &DataBlock,
    modifier: &Modifier,
    armature: &ArmatureData,
    object_world: DMat4,
    armature_world: DMat4,
    bone_matrices: &BTreeMap<Id, BoneMatrices>,
) -> Result<()> {
    if modifier.modifier_type != "armature" || !modifier.enabled {
        return Ok(());
    }
    let use_groups = match modifier.params.get("use_vertex_groups") {
        Some(Value::Bool(value)) => *value,
        Some(_) => {
            return Err(evaluation_error(
                "armature use_vertex_groups parameter must be boolean",
            ));
        }
        None => true,
    };
    let use_envelopes = match modifier.params.get("use_bone_envelopes") {
        Some(Value::Bool(value)) => *value,
        Some(_) => {
            return Err(evaluation_error(
                "armature use_bone_envelopes parameter must be boolean",
            ));
        }
        None => false,
    };
    let object_inverse = checked_inverse(object_world, "armature mesh object world")?;
    let armature_inverse = checked_inverse(armature_world, "armature world")?;
    let object_to_armature = armature_inverse * object_world;
    let mut transforms = Vec::with_capacity(armature.bones.len());
    for (bone_id, bone) in &armature.bones {
        if !bone.deform {
            continue;
        }
        let Some(matrices) = bone_matrices.get(bone_id) else {
            continue;
        };
        let group = data_block
            .vertex_groups
            .iter()
            .find(|group| group.name.as_str() == bone.name.as_str());
        let transform =
            object_inverse * armature_world * matrices.skin * armature_inverse * object_world;
        if !finite_matrix(transform) {
            return Err(evaluation_error_with(
                "armature skin matrix is non-finite",
                json!({ "bone_id": bone_id }),
            ));
        }
        transforms.push((bone_id, bone, group, transform));
    }
    if transforms.is_empty() {
        return Ok(());
    }
    for vertex in &mut mesh.vertices {
        let armature_position = object_to_armature.transform_point3(vertex.co);
        if !armature_position.is_finite() {
            return Err(evaluation_error_with(
                "armature envelope point is non-finite",
                json!({ "vertex_id": vertex.id }),
            ));
        }
        let mut total_weight = 0.0;
        let mut result = DVec3::ZERO;
        for (bone_id, bone, group, matrix) in &transforms {
            let mut weight = if use_groups {
                group
                    .and_then(|group| {
                        data_block
                            .vertex_weights
                            .get(&vertex.id)
                            .and_then(|weights| weights.get(&group.id))
                    })
                    .copied()
                    .unwrap_or(0.0)
            } else {
                1.0
            };
            if use_envelopes {
                weight *= bone_envelope_weight(bone_id, bone, armature_position)?;
            }
            if !weight.is_finite() || weight < 0.0 {
                return Err(evaluation_error_with(
                    "armature vertex weight is invalid",
                    json!({ "vertex_id": vertex.id, "bone_id": bone_id }),
                ));
            }
            if weight > 0.0 {
                let bbone_segments = bone
                    .bbone_settings
                    .get("bbone_segments")
                    .and_then(Value::as_u64)
                    .unwrap_or(1);
                if bbone_segments > 1 {
                    return Err(PotError::with_details(
                        ErrorCode::UnsupportedFeature,
                        "Armature modifier B-Bone segment deformation is not evaluated",
                        json!({
                            "feature_id":"rig.bbone.segments",
                            "modifier_id":modifier.id,
                            "bone_id":bone_id,
                            "vertex_id":vertex.id,
                            "bbone_segments":bbone_segments
                        }),
                    ));
                }
                result += matrix.transform_point3(vertex.co) * weight;
                total_weight += weight;
            }
        }
        if total_weight > 0.0 {
            vertex.co = result / total_weight;
        }
        if !vertex.co.is_finite() {
            return Err(evaluation_error_with(
                "skinned vertex position is non-finite",
                json!({ "vertex_id": vertex.id }),
            ));
        }
    }
    Ok(())
}

fn bone_envelope_weight(bone_id: &Id, bone: &crate::model::Bone, point: DVec3) -> Result<f64> {
    let head = DVec3::from_array(bone.head);
    let tail = DVec3::from_array(bone.tail);
    let segment = tail - head;
    let length_squared = segment.length_squared();
    let values = [
        bone.envelope_distance,
        bone.envelope_weight,
        bone.head_radius,
        bone.tail_radius,
    ];
    if !head.is_finite()
        || !tail.is_finite()
        || !point.is_finite()
        || values.iter().any(|value| !value.is_finite())
        || bone.envelope_distance < 0.0
        || !(0.0..=1.0).contains(&bone.envelope_weight)
        || bone.head_radius < 0.0
        || bone.tail_radius < 0.0
        || length_squared <= f64::EPSILON
    {
        return Err(evaluation_error_with(
            "bone envelope settings are invalid",
            json!({ "bone_id": bone_id }),
        ));
    }
    let factor = ((point - head).dot(segment) / length_squared).clamp(0.0, 1.0);
    let nearest = head + segment * factor;
    let radius = bone.head_radius + (bone.tail_radius - bone.head_radius) * factor;
    let outside_radius = point.distance(nearest) - radius;
    let falloff = if outside_radius <= 0.0 {
        1.0
    } else if bone.envelope_distance <= f64::EPSILON {
        0.0
    } else {
        (1.0 - outside_radius / bone.envelope_distance).clamp(0.0, 1.0)
    };
    Ok(bone.envelope_weight * falloff)
}

/// Resolve an Armature modifier's `armature` target and apply its skinning to the mesh.
pub(crate) fn apply_armature_modifier_with_drivers(
    mesh: &mut Mesh,
    data_block: &DataBlock,
    modifier: &Modifier,
    object_world: DMat4,
    doc: &SceneDoc,
    frame: f64,
    world_matrices: &BTreeMap<Id, DMat4>,
    driver_values: &BTreeMap<(Id, String, u32), f64>,
    project_root: Option<&Path>,
    fps: u32,
    fps_base: f64,
    evaluated_meshes: &BTreeMap<Id, Mesh>,
) -> Result<()> {
    if modifier.modifier_type != "armature" || !modifier.enabled {
        return Ok(());
    }
    let armature_text = modifier
        .params
        .get("object")
        .and_then(Value::as_str)
        .ok_or_else(|| evaluation_error("armature modifier has no armature target"))?;
    let armature_id = Id::new(armature_text)
        .map_err(|_| evaluation_error("armature modifier target ID is invalid"))?;
    let armature_node = doc.nodes.get(&armature_id).ok_or_else(|| {
        evaluation_error_with(
            "armature modifier target node does not exist",
            json!({ "armature": armature_id }),
        )
    })?;
    let armature_data_id = armature_node.data.as_ref().ok_or_else(|| {
        evaluation_error_with(
            "armature modifier target has no Data-Block",
            json!({ "armature": armature_id }),
        )
    })?;
    let armature_data = doc.data_blocks.get(armature_data_id).ok_or_else(|| {
        evaluation_error_with(
            "armature modifier target Data-Block does not exist",
            json!({ "armature": armature_id, "data": armature_data_id }),
        )
    })?;
    let armature = armature_data.armature.as_ref().ok_or_else(|| {
        evaluation_error_with(
            "armature modifier target is not an armature",
            json!({ "armature": armature_id }),
        )
    })?;
    let armature_world = world_matrices.get(&armature_id).copied().ok_or_else(|| {
        evaluation_error_with(
            "armature modifier target transform is unavailable",
            json!({ "armature": armature_id }),
        )
    })?;
    let mut pose = crate::eval::animation::animated_pose_bones(armature_node, doc, frame)?;
    apply_pose_drivers(&armature_id, &mut pose, driver_values)?;
    apply_spline_ik(
        &armature_id,
        armature,
        armature_node,
        doc,
        armature_world,
        world_matrices,
        &mut pose,
    )?;
    let ik_targets = ik_target_matrices(
        doc,
        &armature_id,
        armature,
        &pose,
        frame,
        armature_world,
        world_matrices,
        driver_values,
        project_root,
        fps,
        fps_base,
    )?;
    let bones = evaluate_pose_matrices(
        &armature_id,
        armature,
        &pose,
        &armature_node.constraints,
        doc,
        frame,
        armature_world,
        world_matrices,
        driver_values,
        &ik_targets,
        project_root,
        fps,
        fps_base,
        Some(evaluated_meshes),
    )?;
    apply_armature_skinning(
        mesh,
        data_block,
        modifier,
        armature,
        object_world,
        armature_world,
        &bones,
    )
}

/// Evaluate all node world matrices, including hierarchy and enabled object constraints.
///
/// Parent and constraint-target dependencies are visited in stable ID order. Any dependency cycle
/// is reported as `EVALUATION_FAILED` with its deterministic repeated-node witness.
pub(crate) fn evaluate_world_matrices_with_drivers(
    doc: &SceneDoc,
    frame: f64,
    driver_values: &BTreeMap<(Id, String, u32), f64>,
) -> Result<BTreeMap<Id, DMat4>> {
    let scene = doc.scenes.get(&doc.active_scene);
    evaluate_world_matrices_with_resources(
        doc,
        frame,
        driver_values,
        None,
        scene.map_or(24, |scene| scene.fps),
        scene.map_or(1.0, |scene| scene.fps_base),
    )
}

pub(crate) fn evaluate_world_matrices_with_resources(
    doc: &SceneDoc,
    frame: f64,
    driver_values: &BTreeMap<(Id, String, u32), f64>,
    project_root: Option<&Path>,
    fps: u32,
    fps_base: f64,
) -> Result<BTreeMap<Id, DMat4>> {
    evaluate_world_matrices_with_meshes(
        doc,
        frame,
        driver_values,
        project_root,
        fps,
        fps_base,
        None,
    )
}

pub(crate) fn evaluate_world_matrices_with_evaluated_meshes(
    doc: &SceneDoc,
    frame: f64,
    driver_values: &BTreeMap<(Id, String, u32), f64>,
    project_root: Option<&Path>,
    fps: u32,
    fps_base: f64,
    evaluated_meshes: &BTreeMap<Id, Mesh>,
) -> Result<BTreeMap<Id, DMat4>> {
    evaluate_world_matrices_with_meshes(
        doc,
        frame,
        driver_values,
        project_root,
        fps,
        fps_base,
        Some(evaluated_meshes),
    )
}

fn evaluate_world_matrices_with_meshes(
    doc: &SceneDoc,
    frame: f64,
    driver_values: &BTreeMap<(Id, String, u32), f64>,
    project_root: Option<&Path>,
    fps: u32,
    fps_base: f64,
    evaluated_meshes: Option<&BTreeMap<Id, Mesh>>,
) -> Result<BTreeMap<Id, DMat4>> {
    fn visit(
        id: &Id,
        doc: &SceneDoc,
        frame: f64,
        driver_values: &BTreeMap<(Id, String, u32), f64>,
        project_root: Option<&Path>,
        fps: u32,
        fps_base: f64,
        evaluated_meshes: Option<&BTreeMap<Id, Mesh>>,
        output: &mut BTreeMap<Id, DMat4>,
        active: &mut Vec<Id>,
    ) -> Result<DMat4> {
        if let Some(matrix) = output.get(id) {
            return Ok(*matrix);
        }
        if let Some(start) = active.iter().position(|active_id| active_id == id) {
            let mut cycle: Vec<String> = active[start..].iter().map(ToString::to_string).collect();
            cycle.push(id.to_string());
            return Err(evaluation_error_with(
                "node transform dependency cycle",
                json!({ "cycle": cycle }),
            ));
        }
        active.push(id.clone());
        let node = doc.nodes.get(id).ok_or_else(|| {
            evaluation_error_with(
                "node transform dependency does not exist",
                json!({ "id": id }),
            )
        })?;
        for constraint in node.constraints.iter().filter(|constraint| {
            constraint.owner_bone.is_some() && constraint.enabled && constraint.influence != 0.0
        }) {
            if let Some(target_id) = &constraint.target {
                visit(
                    target_id,
                    doc,
                    frame,
                    driver_values,
                    project_root,
                    fps,
                    fps_base,
                    evaluated_meshes,
                    output,
                    active,
                )?;
            }
            if constraint.constraint_type == ConstraintType::Armature
                && let Some(targets) = constraint.params.get("targets").and_then(Value::as_array)
            {
                for target in targets {
                    let Some(target_name) = target.get("target").and_then(Value::as_str) else {
                        continue;
                    };
                    let target_id = Id::new(target_name.to_owned()).map_err(|error| {
                        evaluation_error_with(
                            "Armature constraint target ID is invalid",
                            json!({"constraint_id":constraint.id,"reason":error.message}),
                        )
                    })?;
                    visit(
                        &target_id,
                        doc,
                        frame,
                        driver_values,
                        project_root,
                        fps,
                        fps_base,
                        evaluated_meshes,
                        output,
                        active,
                    )?;
                }
            }
            if constraint.constraint_type == ConstraintType::Ik
                && let Some(pole_id) = constraint.params.get("pole_target").and_then(Value::as_str)
            {
                let pole_id = Id::new(pole_id).map_err(|_| {
                    evaluation_error_with(
                        "IK pole target ID is invalid",
                        json!({ "node": id, "constraint_id": constraint.id }),
                    )
                })?;
                visit(
                    &pole_id,
                    doc,
                    frame,
                    driver_values,
                    project_root,
                    fps,
                    fps_base,
                    evaluated_meshes,
                    output,
                    active,
                )?;
            }
            for key in ["camera", "depth_object"] {
                if let Some(reference) = constraint.params.get(key).and_then(Value::as_str) {
                    let reference_id = Id::new(reference.to_owned()).map_err(|error| {
                        evaluation_error_with(
                            "constraint node reference ID is invalid",
                            json!({"constraint_id":constraint.id,"parameter":key,"reason":error.message}),
                        )
                    })?;
                    visit(
                        &reference_id,
                        doc,
                        frame,
                        driver_values,
                        project_root,
                        fps,
                        fps_base,
                        evaluated_meshes,
                        output,
                        active,
                    )?;
                }
            }
        }
        let animated = animated_transform(node, doc, frame)?;
        let local = apply_transform_drivers(id, animated, driver_values)?.matrix();
        let parent_inverse = node
            .parent_inverse
            .map_or(DMat4::IDENTITY, |array| DMat4::from_cols_array(&array));
        let mut world = if let Some(parent_id) = &node.parent {
            let parent_world = visit(
                parent_id,
                doc,
                frame,
                driver_values,
                project_root,
                fps,
                fps_base,
                evaluated_meshes,
                output,
                active,
            )?;
            let parent_space = if node.parent_type == ParentType::Bone {
                let bone_id = node.parent_bone.as_ref().ok_or_else(|| {
                    evaluation_error_with(
                        "bone-parented node has no parent bone",
                        json!({ "id": id }),
                    )
                })?;
                let bone_worlds = evaluate_armature_bone_world_matrices_with_meshes(
                    doc,
                    parent_id,
                    frame,
                    parent_world,
                    output,
                    driver_values,
                    project_root,
                    fps,
                    fps_base,
                    evaluated_meshes,
                )?;
                bone_worlds.get(bone_id).copied().ok_or_else(|| {
                    evaluation_error_with(
                        "parent bone does not exist",
                        json!({ "id": parent_id, "bone_id": bone_id }),
                    )
                })?
            } else {
                parent_world
            };
            parent_space * parent_inverse * local
        } else {
            parent_inverse * local
        };
        for constraint in node.constraints.iter().filter(|constraint| {
            constraint.owner_bone.is_none() && constraint.enabled && constraint.influence != 0.0
        }) {
            let target_id = constraint
                .target
                .clone()
                .or(constraint_camera_target_id(constraint)?);
            if target_id.is_none() && constraint_requires_target(constraint) {
                return Err(evaluation_error_with(
                    "enabled constraint has no target",
                    json!({ "node": id, "constraint_id": constraint.id }),
                ));
            }
            let target_world = if let Some(target_id) = &target_id {
                Some(visit(
                    target_id,
                    doc,
                    frame,
                    driver_values,
                    project_root,
                    fps,
                    fps_base,
                    evaluated_meshes,
                    output,
                    active,
                )?)
            } else {
                None
            };
            if constraint.constraint_type == ConstraintType::Armature
                && let Some(targets) = constraint.params.get("targets").and_then(Value::as_array)
            {
                for target in targets {
                    let Some(target_name) = target.get("target").and_then(Value::as_str) else {
                        continue;
                    };
                    let target_id = Id::new(target_name.to_owned()).map_err(|error| {
                        evaluation_error_with(
                            "Armature constraint target ID is invalid",
                            json!({"constraint_id":constraint.id,"reason":error.message}),
                        )
                    })?;
                    visit(
                        &target_id,
                        doc,
                        frame,
                        driver_values,
                        project_root,
                        fps,
                        fps_base,
                        evaluated_meshes,
                        output,
                        active,
                    )?;
                }
            }
            if constraint.constraint_type == ConstraintType::Ik
                && let Some(pole_id) = constraint.params.get("pole_target").and_then(Value::as_str)
            {
                let pole_id = Id::new(pole_id).map_err(|_| {
                    evaluation_error_with(
                        "IK pole target ID is invalid",
                        json!({ "node": id, "constraint_id": constraint.id }),
                    )
                })?;
                visit(
                    &pole_id,
                    doc,
                    frame,
                    driver_values,
                    project_root,
                    fps,
                    fps_base,
                    evaluated_meshes,
                    output,
                    active,
                )?;
            }
            for key in ["camera", "depth_object", "space_object"] {
                if let Some(reference) = constraint.params.get(key).and_then(Value::as_str) {
                    let reference_id = Id::new(reference.to_owned()).map_err(|error| {
                        evaluation_error_with(
                            "constraint node reference ID is invalid",
                            json!({"constraint_id":constraint.id,"parameter":key,"reason":error.message}),
                        )
                    })?;
                    visit(
                        &reference_id,
                        doc,
                        frame,
                        driver_values,
                        project_root,
                        fps,
                        fps_base,
                        evaluated_meshes,
                        output,
                        active,
                    )?;
                }
            }
            if matches!(
                constraint.constraint_type,
                ConstraintType::Ik | ConstraintType::SplineIk
            ) {
                continue;
            }
            let target_matrix = if constraint.constraint_type == ConstraintType::TransformCache {
                transform_cache_matrix(doc, constraint, project_root, frame, fps, fps_base)?
            } else if let (Some(target_id), Some(bone_id)) =
                (target_id.as_ref(), constraint.subtarget.as_ref())
            {
                let bone_worlds = evaluate_armature_bone_world_matrices_with_meshes(
                    doc,
                    target_id,
                    frame,
                    target_world.ok_or_else(|| {
                        evaluation_error("constraint target world transform is unavailable")
                    })?,
                    output,
                    driver_values,
                    project_root,
                    fps,
                    fps_base,
                    evaluated_meshes,
                )?;
                let bone_world = bone_worlds.get(bone_id).copied().ok_or_else(|| {
                    evaluation_error_with(
                        "constraint target bone does not exist",
                        json!({ "target": target_id, "bone_id": bone_id }),
                    )
                })?;
                let target_node = doc.nodes.get(target_id).ok_or_else(|| {
                    evaluation_error_with(
                        "constraint target armature does not exist",
                        json!({"target": target_id}),
                    )
                })?;
                constraint_target_bone_matrix(constraint, target_node, doc, bone_world)?
            } else {
                target_world.unwrap_or(DMat4::IDENTITY)
            };
            world = apply_object_constraint(
                world,
                target_matrix,
                constraint,
                doc,
                frame,
                output,
                evaluated_meshes,
                Some(id),
                id,
                world,
                None,
                None,
            )?;
        }
        if !finite_matrix(world) {
            return Err(evaluation_error_with(
                "evaluated node world transform is non-finite",
                json!({ "id": id }),
            ));
        }
        let popped = active.pop();
        debug_assert_eq!(popped.as_ref(), Some(id));
        output.insert(id.clone(), world);
        Ok(world)
    }
    if !frame.is_finite() {
        return Err(evaluation_error("evaluation frame is non-finite"));
    }

    let mut output = BTreeMap::new();
    let mut active = Vec::new();
    for id in doc.nodes.keys() {
        visit(
            id,
            doc,
            frame,
            driver_values,
            project_root,
            fps,
            fps_base,
            evaluated_meshes,
            &mut output,
            &mut active,
        )?;
    }
    Ok(output)
}

fn transform_cache_matrix(
    doc: &SceneDoc,
    constraint: &Constraint,
    project_root: Option<&Path>,
    frame: f64,
    fps: u32,
    fps_base: f64,
) -> Result<DMat4> {
    let resource_text = string_param(&constraint.params, "resource")?;
    let resource_id = Id::new(resource_text.to_owned()).map_err(|error| {
        evaluation_error_with(
            "Transform Cache resource ID is invalid",
            json!({"constraint_id":constraint.id,"resource":resource_text,"reason":error.message}),
        )
    })?;
    if let Some(kind) = doc
        .resources
        .get(&resource_id)
        .and_then(|resource| resource.get("kind"))
        .and_then(Value::as_str)
        && ["usd", "usdc", "usdz"]
            .iter()
            .any(|supported| kind.eq_ignore_ascii_case(supported))
    {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "USD Transform Cache archives are not evaluated",
            json!({
                "constraint_id":constraint.id,
                "resource_id":resource_id,
                "feature_id":"constraint.transform_cache.usd",
            }),
        ));
    }
    let object_path = string_param(&constraint.params, "object_path")?;
    let frame_offset = number_param(&constraint.params, "frame_offset")?.unwrap_or(0.0);
    let scale = number_param(&constraint.params, "scale")?.unwrap_or(1.0);
    let override_frame = match constraint.params.get("override_frame") {
        None | Some(Value::Null) => None,
        Some(value) => Some(
            value
                .as_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| {
                    evaluation_error("Transform Cache override_frame must be finite or null")
                })?,
        ),
    };
    let rate = f64::from(fps) / fps_base;
    if !rate.is_finite() || rate <= 0.0 {
        return Err(evaluation_error(
            "Transform Cache requires a finite positive scene frame rate",
        ));
    }
    let time_seconds = (override_frame.unwrap_or(frame) - frame_offset) / rate;
    if !time_seconds.is_finite() {
        return Err(evaluation_error(
            "Transform Cache sample time is not finite",
        ));
    }
    let (bytes, _) = crate::eval::read_cache_resource(doc, project_root, &resource_id)?;
    let matrix = DMat4::from_cols_array(&crate::exchange::alembic::read_xform_matrix(
        &bytes,
        object_path,
        time_seconds,
        scale,
    )?);
    if !finite_matrix(matrix) {
        return Err(evaluation_error("Transform Cache matrix is not finite"));
    }
    Ok(matrix)
}

fn checked_influence(constraint: &Constraint) -> Result<f64> {
    if !constraint.influence.is_finite() {
        return Err(evaluation_error_with(
            "constraint influence is non-finite",
            json!({ "constraint_id": constraint.id }),
        ));
    }
    Ok(constraint.influence.clamp(0.0, 1.0))
}

fn matrix_transform(matrix: DMat4) -> Result<Transform> {
    if !finite_matrix(matrix) {
        return Err(evaluation_error("constraint input matrix is non-finite"));
    }
    let (scale, rotation, translation) = matrix.to_scale_rotation_translation();
    if !scale.is_finite() || !rotation.is_finite() || !translation.is_finite() {
        return Err(evaluation_error(
            "constraint transform decomposition failed",
        ));
    }
    Transform::from_rotation_quat(
        translation.to_array(),
        [rotation.x, rotation.y, rotation.z, rotation.w],
        scale.to_array(),
    )
    .map_err(|error| evaluation_error(error.message))
}

fn blend_transform(source: DMat4, target: DMat4, influence: f64) -> Result<DMat4> {
    let source = matrix_transform(source)?;
    let target = matrix_transform(target)?;
    let source_rotation = source.rotation_quat();
    let target_rotation = target.rotation_quat();
    let rotation = source_rotation
        .slerp(target_rotation, influence)
        .normalize();
    let translation = DVec3::from_array(source.translation)
        .lerp(DVec3::from_array(target.translation), influence);
    let scale = DVec3::from_array(source.scale).lerp(DVec3::from_array(target.scale), influence);
    let matrix = DMat4::from_scale_rotation_translation(scale, rotation, translation);
    if !finite_matrix(matrix) {
        return Err(evaluation_error("constraint result is non-finite"));
    }
    Ok(matrix)
}

fn constraint_clip<'a>(
    doc: &'a SceneDoc,
    constraint: &Constraint,
) -> Result<&'a crate::model::MovieClip> {
    let clip_id = if param_bool(&constraint.params, "use_active_clip", false)? {
        doc.scenes
            .get(&doc.active_scene)
            .and_then(|scene| scene.active_clip.as_ref())
            .ok_or_else(|| {
                evaluation_error_with(
                    "constraint use_active_clip requires an active scene movie clip",
                    json!({"constraint_id":constraint.id}),
                )
            })?
    } else {
        let clip_text = string_param(&constraint.params, "clip")?;
        let clip_id = Id::new(clip_text.to_owned()).map_err(|error| {
            evaluation_error_with(
                "constraint movie-clip ID is invalid",
                json!({"constraint_id":constraint.id,"clip":clip_text,"reason":error.message}),
            )
        })?;
        return doc.movie_clips.get(&clip_id).ok_or_else(|| {
            evaluation_error_with(
                "constraint movie clip does not exist",
                json!({"constraint_id":constraint.id,"clip":clip_id}),
            )
        });
    };
    doc.movie_clips.get(clip_id).ok_or_else(|| {
        evaluation_error_with(
            "constraint movie clip does not exist",
            json!({"constraint_id":constraint.id,"clip":clip_id}),
        )
    })
}

fn interpolate_pose_matrices(
    before: (f64, DMat4),
    after: (f64, DMat4),
    frame: f64,
) -> Result<DMat4> {
    if crate::float::equal_f64(before.0, after.0) {
        return Ok(before.1);
    }
    blend_transform(
        before.1,
        after.1,
        ((frame - before.0) / (after.0 - before.0)).clamp(0.0, 1.0),
    )
}

fn camera_reconstruction_matrix(
    camera: &crate::model::SolvedCamera,
    clip: &crate::model::MovieClip,
) -> Result<DMat4> {
    let matrix = if camera.matrix_is_camera_to_world {
        // Blender's reconstructed camera matrix is already camera-to-world; don't reinterpret
        // it as the projective matrix accepted by tracking::camera_world_matrix.
        let [row_x, row_y, row_z] = camera.matrix;
        DMat4::from_cols(
            glam::DVec4::new(row_x[0], row_y[0], row_z[0], 0.0),
            glam::DVec4::new(row_x[1], row_y[1], row_z[1], 0.0),
            glam::DVec4::new(row_x[2], row_y[2], row_z[2], 0.0),
            glam::DVec4::new(row_x[3], row_y[3], row_z[3], 1.0),
        )
    } else {
        DMat4::from_cols_array(&crate::tracking::camera_world_matrix(
            &crate::tracking::CameraModel {
                matrix: camera.matrix,
            },
            &clip.tracking.camera,
            clip.width,
            clip.height,
        )?)
    };
    if finite_matrix(matrix) {
        Ok(matrix)
    } else {
        Err(evaluation_error(
            "camera reconstruction transform is non-finite",
        ))
    }
}

fn camera_solver_matrix(doc: &SceneDoc, constraint: &Constraint, frame: f64) -> Result<DMat4> {
    let clip = constraint_clip(doc, constraint)?;
    if clip.tracking.reconstruction.cameras.is_empty() {
        return Err(evaluation_error_with(
            "camera solver requires reconstructed camera frames",
            json!({"constraint_id":constraint.id}),
        ));
    }
    let mut before = None;
    let mut after = None;
    for camera in &clip.tracking.reconstruction.cameras {
        let camera_frame = f64::from(camera.frame);
        if camera_frame <= frame {
            before = Some((camera_frame, camera_reconstruction_matrix(camera, clip)?));
        }
        if camera_frame >= frame {
            after = Some((camera_frame, camera_reconstruction_matrix(camera, clip)?));
            break;
        }
    }
    let first = before
        .or(after)
        .ok_or_else(|| evaluation_error("camera reconstruction is empty"))?;
    let last = after
        .or(before)
        .ok_or_else(|| evaluation_error("camera reconstruction is empty"))?;
    interpolate_pose_matrices(first, last, frame)
}

fn object_solver_matrix(
    doc: &SceneDoc,
    constraint: &Constraint,
    frame: f64,
    camera_in_owner_space: DMat4,
    source: DMat4,
) -> Result<DMat4> {
    let clip = constraint_clip(doc, constraint)?;
    let object_id = string_param(&constraint.params, "object")?;
    let object = clip
        .tracking
        .objects
        .iter()
        .find(|object| object.id == object_id || object.name == object_id)
        .ok_or_else(|| {
            evaluation_error_with(
                "object solver tracking object does not exist",
                json!({"constraint_id":constraint.id,"object":object_id}),
            )
        })?;
    let camera_to_object = sample_tracking_object_pose(object, frame)?;
    let object_to_camera = checked_inverse(camera_to_object, "object reconstruction")?;
    let parent_matrix = camera_in_owner_space * object_to_camera;
    let inverse = if param_bool(&constraint.params, "set_inverse_pending", false)? {
        // Evaluation is pure; model Blender's one-time capture at the scene's current frame.
        let reference_frame = doc
            .scenes
            .get(&doc.active_scene)
            .ok_or_else(|| evaluation_error("Object Solver active scene is unavailable"))?
            .frame_current;
        let reference_camera = camera_solver_matrix(doc, constraint, reference_frame)?;
        let reference_camera_to_object = sample_tracking_object_pose(object, reference_frame)?;
        let reference_object_to_camera =
            checked_inverse(reference_camera_to_object, "object reconstruction")?;
        let reference_parent = reference_camera * reference_object_to_camera;
        checked_inverse(reference_parent, "object solver parent")?
    } else if let Some(value) = constraint.inverse_matrix {
        let matrix = DMat4::from_cols_array(&value);
        if finite_matrix(matrix) {
            matrix
        } else {
            return Err(evaluation_error_with(
                "Object Solver inverse_matrix must be finite",
                json!({"constraint_id":constraint.id}),
            ));
        }
    } else {
        DMat4::IDENTITY
    };
    let result = parent_matrix * inverse * source;
    if finite_matrix(result) {
        Ok(result)
    } else {
        Err(evaluation_error_with(
            "Object Solver result is non-finite",
            json!({"constraint_id":constraint.id}),
        ))
    }
}

fn action_constraint_matrix(
    source: DMat4,
    target: DMat4,
    constraint: &Constraint,
    doc: &SceneDoc,
    _frame: f64,
    _world_matrices: &BTreeMap<Id, DMat4>,
    _owner_node_id: Option<&Id>,
    _world_to_owner: Option<DMat4>,
) -> Result<DMat4> {
    let action_text = string_param(&constraint.params, "action")?;
    let action_id = Id::new(action_text.to_owned()).map_err(|error| {
        evaluation_error_with(
            "Action constraint action ID is invalid",
            json!({"constraint_id":constraint.id,"action":action_text,"reason":error.message}),
        )
    })?;
    let action = doc.actions.get(&action_id).ok_or_else(|| {
        evaluation_error_with(
            "Action constraint action does not exist",
            json!({"constraint_id":constraint.id,"action":action_id}),
        )
    })?;
    let target_matrix = target;
    let target_transform = matrix_transform(target_matrix)?;
    let channel = string_param(&constraint.params, "transform_channel")?;
    let channel = match channel {
        "LOCATION_X" => "location_x",
        "LOCATION_Y" => "location_y",
        "LOCATION_Z" => "location_z",
        "ROTATION_X" => "rotation_x",
        "ROTATION_Y" => "rotation_y",
        "ROTATION_Z" => "rotation_z",
        "SCALE_X" => "scale_x",
        "SCALE_Y" => "scale_y",
        "SCALE_Z" => "scale_z",
        _ => {
            return Err(evaluation_error(
                "Action constraint transform_channel is invalid",
            ));
        }
    };
    let mut target_value = constraint_channel(&target_transform, channel)?;
    if channel.starts_with("rotation_") {
        target_value = target_value.to_degrees();
    }
    let minimum = number_param(&constraint.params, "min")?
        .ok_or_else(|| evaluation_error("Action constraint min is missing"))?;
    let maximum = number_param(&constraint.params, "max")?
        .ok_or_else(|| evaluation_error("Action constraint max is missing"))?;
    let frame_start = number_param(&constraint.params, "frame_start")?
        .ok_or_else(|| evaluation_error("Action constraint frame_start is missing"))?;
    let frame_end = number_param(&constraint.params, "frame_end")?
        .ok_or_else(|| evaluation_error("Action constraint frame_end is missing"))?;
    let use_eval_time = param_bool(&constraint.params, "use_eval_time", false)?;
    let normalized_time = if use_eval_time {
        number_param(&constraint.params, "eval_time")?.unwrap_or(0.0)
    } else if (maximum - minimum).abs() <= f64::EPSILON {
        0.0
    } else {
        (target_value - minimum) / (maximum - minimum)
    }
    .clamp(0.0, 1.0);
    let evaluation_frame = frame_start + normalized_time * (frame_end - frame_start);
    let use_object_action = param_bool(&constraint.params, "use_bone_object_action", false)?;
    let owner_bone = constraint
        .owner_bone
        .as_ref()
        .filter(|_| !use_object_action);
    let action_transform = sample_constraint_action(action, evaluation_frame, owner_bone)?;
    action_mix_matrix(
        source,
        action_transform.matrix(),
        string_param(&constraint.params, "mix_mode")?,
    )
}

fn action_mix_matrix(source: DMat4, action: DMat4, mode: &str) -> Result<DMat4> {
    match mode {
        "REPLACE" => Ok(action),
        "BEFORE_FULL" => Ok(action * source),
        "AFTER_FULL" => Ok(source * action),
        "BEFORE" => aligned_action_mix(action, source),
        "AFTER" => aligned_action_mix(source, action),
        "BEFORE_SPLIT" => split_action_mix(action, source),
        "AFTER_SPLIT" => split_action_mix(source, action),
        _ => Err(evaluation_error("Action constraint mix_mode is invalid")),
    }
}

fn aligned_action_mix(left: DMat4, right: DMat4) -> Result<DMat4> {
    let left = matrix_transform(left)?;
    let right = matrix_transform(right)?;
    let translation = left
        .matrix()
        .transform_point3(DVec3::from_array(right.translation));
    let rotation = left.rotation_quat() * right.rotation_quat();
    Transform::from_rotation_quat(
        translation.to_array(),
        [rotation.x, rotation.y, rotation.z, rotation.w],
        (DVec3::from_array(left.scale) * DVec3::from_array(right.scale)).to_array(),
    )
    .map(|transform| transform.matrix())
    .map_err(|error| evaluation_error(error.message))
}

fn split_action_mix(left: DMat4, right: DMat4) -> Result<DMat4> {
    let left = matrix_transform(left)?;
    let right = matrix_transform(right)?;
    let rotation = left.rotation_quat() * right.rotation_quat();
    Transform::from_rotation_quat(
        (DVec3::from_array(left.translation) + DVec3::from_array(right.translation)).to_array(),
        [rotation.x, rotation.y, rotation.z, rotation.w],
        (DVec3::from_array(left.scale) * DVec3::from_array(right.scale)).to_array(),
    )
    .map(|transform| transform.matrix())
    .map_err(|error| evaluation_error(error.message))
}

fn sample_constraint_action(
    action: &crate::model::Action,
    frame: f64,
    owner_bone: Option<&Id>,
) -> Result<Transform> {
    let mut transform = Transform::default();
    let mut euler = None;
    let mut quaternion = transform.rotation;
    let mut quaternion_modified = false;
    let pose_prefix = owner_bone.map(|bone_id| format!("pose.{bone_id}."));
    for curve in &action.fcurves {
        let path = if let Some(prefix) = &pose_prefix {
            let Some(channel) = curve.path.strip_prefix(prefix) else {
                continue;
            };
            match channel {
                "location" | "translation" => "translation",
                "rotation" => "rotation",
                "rotation_euler" => "rotation_euler",
                "scale" => "scale",
                _ => continue,
            }
        } else {
            curve.path.as_str()
        };
        let Some(value) = crate::eval::animation::sample_curve(curve, frame)? else {
            continue;
        };
        match path {
            "transform.translation" | "translation" | "location" => {
                let component = usize::try_from(curve.index)
                    .ok()
                    .filter(|component| *component < 3)
                    .ok_or_else(|| {
                        evaluation_error("Action constraint translation index is invalid")
                    })?;
                transform.translation[component] = value;
            }
            "transform.scale" | "scale" => {
                let component = usize::try_from(curve.index)
                    .ok()
                    .filter(|component| *component < 3)
                    .ok_or_else(|| evaluation_error("Action constraint scale index is invalid"))?;
                transform.scale[component] = value;
            }
            "transform.rotation_quaternion" | "transform.rotation" | "rotation" => {
                let component = usize::try_from(curve.index)
                    .ok()
                    .filter(|component| *component < 4)
                    .ok_or_else(|| {
                        evaluation_error("Action constraint rotation index is invalid")
                    })?;
                quaternion[[3, 0, 1, 2][component]] = value;
                quaternion_modified = true;
            }
            "transform.rotation_euler" | "rotation_euler" => {
                let component = usize::try_from(curve.index)
                    .ok()
                    .filter(|component| *component < 3)
                    .ok_or_else(|| evaluation_error("Action constraint Euler index is invalid"))?;
                if euler.is_none() {
                    let angles = transform.rotation_quat().to_euler(EulerRot::XYZ);
                    euler = Some([angles.0, angles.1, angles.2]);
                }
                if let Some(angles) = &mut euler {
                    angles[component] = value;
                }
            }
            _ => {}
        }
    }
    let euler_modified = euler.is_some();
    if let Some(angles) = euler {
        let rotation = DQuat::from_rotation_z(angles[2])
            * DQuat::from_rotation_y(angles[1])
            * DQuat::from_rotation_x(angles[0]);
        transform.rotation = [rotation.x, rotation.y, rotation.z, rotation.w];
    }
    if quaternion_modified {
        transform.rotation = quaternion;
    }
    if quaternion_modified || euler_modified {
        let rotation = transform.rotation_quat().normalize();
        transform.rotation = [rotation.x, rotation.y, rotation.z, rotation.w];
    }
    Ok(transform)
}

fn geometry_attribute_value(
    doc: &SceneDoc,
    constraint: &Constraint,
    evaluated_meshes: Option<&BTreeMap<Id, Mesh>>,
) -> Result<Value> {
    let target_id = constraint.target.as_ref().ok_or_else(|| {
        evaluation_error_with(
            "Geometry Attribute constraint requires a target mesh object",
            json!({"constraint_id":constraint.id}),
        )
    })?;
    let target_node = doc.nodes.get(target_id).ok_or_else(|| {
        evaluation_error_with(
            "Geometry Attribute target object does not exist",
            json!({"constraint_id":constraint.id,"target":target_id}),
        )
    })?;
    let data_id = target_node.data.as_ref().ok_or_else(|| {
        evaluation_error_with(
            "Geometry Attribute target has no mesh data",
            json!({"constraint_id":constraint.id,"target":target_id}),
        )
    })?;
    let base_mesh = doc
        .data_blocks
        .get(data_id)
        .and_then(|data| data.mesh.as_ref());
    let mesh = evaluated_meshes
        .and_then(|meshes| meshes.get(target_id))
        .or(base_mesh)
        .ok_or_else(|| {
            evaluation_error_with(
                "Geometry Attribute target does not reference a mesh",
                json!({"constraint_id":constraint.id,"target":target_id}),
            )
        })?;
    let name = string_param(&constraint.params, "attribute_name")?;
    let domain = string_param(&constraint.params, "domain")?;
    if name == "position" && domain == "POINT" {
        let domain_count = mesh.vertices.len();
        if domain_count == 0 {
            return Err(evaluation_error(
                "Geometry Attribute domain has no sampleable elements",
            ));
        }
        let sample_index = param_usize(&constraint.params, "sample_index")?
            .unwrap_or(0)
            .min(domain_count - 1);
        return Ok(json!(mesh.vertices[sample_index].co.to_array()));
    }
    let blender_attribute = mesh
        .attributes
        .get("blender_attributes")
        .and_then(Value::as_array)
        .and_then(|attributes| {
            attributes
                .iter()
                .find(|attribute| attribute.get("name").and_then(Value::as_str) == Some(name))
        });
    let attribute = mesh
        .attributes
        .get(name)
        .or(blender_attribute)
        .ok_or_else(|| {
            evaluation_error_with(
                "Geometry Attribute target does not contain the requested attribute",
                json!({"constraint_id":constraint.id,"target":target_id,"attribute_name":name}),
            )
        })?;
    if matches!(domain, "INSTANCE" | "CURVE") {
        let feature_id = if domain == "INSTANCE" {
            "constraint.geometry_attribute.instances"
        } else {
            "constraint.geometry_attribute.curve_domain"
        };
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "Geometry Attribute domain is not represented by the current mesh evaluator",
            json!({
                "feature_id":feature_id,
                "constraint_id":constraint.id,
            }),
        ));
    }
    let domain_count = match domain {
        "POINT" => mesh.vertices.len(),
        "EDGE" => mesh.edges.len(),
        "FACE" => mesh.faces.len(),
        "FACE_CORNER" => mesh.faces.iter().map(|face| face.vertices.len()).sum(),
        _ => return Err(evaluation_error("Geometry Attribute domain is invalid")),
    };
    if domain_count == 0 {
        return Err(evaluation_error(
            "Geometry Attribute domain has no sampleable elements",
        ));
    }
    let sample_index = param_usize(&constraint.params, "sample_index")?
        .unwrap_or(0)
        .min(domain_count - 1);
    let values = attribute.get("values").and_then(Value::as_object);
    let value_at = |key: String, corner_index: Option<usize>| -> Result<Value> {
        let value = values
            .and_then(|values| values.get(&key))
            .or_else(|| {
                attribute
                    .get("values")
                    .and_then(Value::as_array)
                    .and_then(|values| values.get(sample_index))
            })
            .or_else(|| {
                (name == "uv_map")
                    .then_some(attribute)
                    .and_then(Value::as_array)
                    .and_then(|values| values.get(sample_index))
            })
            .ok_or_else(|| {
                evaluation_error_with(
                    "Geometry Attribute sample index is out of range",
                    json!({"constraint_id":constraint.id,"domain":domain,"sample_index":sample_index}),
                )
            })?;
        if let Some(corner_index) = corner_index {
            return value
                .as_array()
                .and_then(|values| values.get(corner_index))
                .cloned()
                .ok_or_else(|| {
                    evaluation_error_with(
                        "Geometry Attribute corner sample is missing",
                        json!({"constraint_id":constraint.id,"sample_index":sample_index}),
                    )
                });
        }
        Ok(value.clone())
    };
    let stored_domain = attribute
        .get("domain")
        .and_then(Value::as_str)
        .unwrap_or(if name == "uv_map" { "corner" } else { "" });
    let domain_matches = match domain {
        "POINT" => {
            stored_domain.eq_ignore_ascii_case("point")
                || stored_domain.eq_ignore_ascii_case("vertices")
        }
        "EDGE" => {
            stored_domain.eq_ignore_ascii_case("edge")
                || stored_domain.eq_ignore_ascii_case("edges")
        }
        "FACE" => {
            stored_domain.eq_ignore_ascii_case("face")
                || stored_domain.eq_ignore_ascii_case("faces")
        }
        "FACE_CORNER" => {
            stored_domain.eq_ignore_ascii_case("corner")
                || stored_domain.eq_ignore_ascii_case("corners")
        }
        _ => false,
    };
    if !domain_matches {
        return Err(evaluation_error_with(
            "Geometry Attribute domain does not match the target attribute",
            json!({"constraint_id":constraint.id,"domain":domain,"attribute_domain":stored_domain}),
        ));
    }
    match domain {
        "POINT" => {
            let vertex = mesh.vertices.get(sample_index).ok_or_else(|| {
                evaluation_error("Geometry Attribute point sample index is out of range")
            })?;
            value_at(format!("v{}", vertex.id), None)
        }
        "EDGE" => {
            let edge = mesh.edges.get(sample_index).ok_or_else(|| {
                evaluation_error("Geometry Attribute edge sample index is out of range")
            })?;
            value_at(format!("e{}", edge.id), None)
        }
        "FACE" => {
            let face = mesh.faces.get(sample_index).ok_or_else(|| {
                evaluation_error("Geometry Attribute face sample index is out of range")
            })?;
            value_at(format!("f{}", face.id), None)
        }
        "FACE_CORNER" => {
            let mut current_index = sample_index;
            for face in &mesh.faces {
                if current_index < face.vertices.len() {
                    return value_at(format!("f{}", face.id), Some(current_index));
                }
                current_index -= face.vertices.len();
            }
            Err(evaluation_error(
                "Geometry Attribute corner sample index is out of range",
            ))
        }
        "CURVE" | "INSTANCE" => Err(evaluation_error(
            "Geometry Attribute domain is not a mesh domain",
        )),
        _ => Err(evaluation_error("Geometry Attribute domain is invalid")),
    }
}

fn attribute_vector3(value: &Value) -> Result<DVec3> {
    let components = value
        .as_array()
        .filter(|components| components.len() == 3)
        .and_then(|components| {
            Some([
                components[0].as_f64()?,
                components[1].as_f64()?,
                components[2].as_f64()?,
            ])
        })
        .ok_or_else(|| {
            evaluation_error("Geometry Attribute VECTOR data must have three finite components")
        })?;
    Ok(DVec3::from_array(components))
}

fn attribute_quaternion(value: &Value) -> Result<DQuat> {
    let components = value
        .as_array()
        .filter(|components| components.len() == 4)
        .and_then(|components| {
            Some([
                components[0].as_f64()?,
                components[1].as_f64()?,
                components[2].as_f64()?,
                components[3].as_f64()?,
            ])
        })
        .ok_or_else(|| {
            evaluation_error("Geometry Attribute QUATERNION data must have four finite components")
        })?;
    Ok(DQuat::from_xyzw(components[0], components[1], components[2], components[3]).normalize())
}

fn attribute_matrix(value: &Value) -> Result<DMat4> {
    if let Some(components) = value.as_array().filter(|components| components.len() == 16) {
        let values = components
            .iter()
            .map(Value::as_f64)
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| evaluation_error("Geometry Attribute matrix contains a non-number"))?;
        return Ok(DMat4::from_cols_array(&[
            values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7],
            values[8], values[9], values[10], values[11], values[12], values[13], values[14],
            values[15],
        ]));
    }
    if let Some(rows) = value.as_array().filter(|rows| rows.len() == 4) {
        let mut columns = [0.0; 16];
        for row in 0..4 {
            let row_values = rows[row]
                .as_array()
                .filter(|row| row.len() == 4)
                .ok_or_else(|| evaluation_error("Geometry Attribute matrix must be 4x4"))?;
            for column in 0..4 {
                columns[column * 4 + row] = row_values[column]
                    .as_f64()
                    .ok_or_else(|| evaluation_error("Geometry Attribute matrix is invalid"))?;
            }
        }
        return Ok(DMat4::from_cols_array(&columns));
    }
    Err(evaluation_error(
        "Geometry Attribute FLOAT4X4 data must be 4x4",
    ))
}

fn geometry_attribute_matrix(
    source: DMat4,
    target: DMat4,
    doc: &SceneDoc,
    constraint: &Constraint,
    evaluated_meshes: Option<&BTreeMap<Id, Mesh>>,
) -> Result<DMat4> {
    let value = geometry_attribute_value(doc, constraint, evaluated_meshes)?;
    let data_type = string_param(&constraint.params, "data_type")?;
    let mix_mode = string_param(&constraint.params, "mix_mode")?;
    let apply_target = param_bool(&constraint.params, "apply_target_transform", false)?;
    let mut sampled_transform_matrix = if mix_mode == "REPLACE" {
        source
    } else {
        DMat4::IDENTITY
    };
    match data_type {
        "VECTOR" => {
            let mut transform = matrix_transform(sampled_transform_matrix)?;
            transform.translation = attribute_vector3(&value)?.to_array();
            sampled_transform_matrix = transform.matrix();
        }
        "QUATERNION" => {
            let mut transform = matrix_transform(sampled_transform_matrix)?;
            let rotation = attribute_quaternion(&value)?;
            transform.rotation = [rotation.x, rotation.y, rotation.z, rotation.w];
            sampled_transform_matrix = transform.matrix();
        }
        "FLOAT4X4" => {
            let sampled = attribute_matrix(&value)?;
            let use_location = param_bool(&constraint.params, "mix_loc", false)?;
            let use_rotation = param_bool(&constraint.params, "mix_rot", false)?;
            let use_scale = param_bool(&constraint.params, "mix_scl", false)?;
            if use_location && use_rotation && use_scale {
                sampled_transform_matrix = sampled;
            } else {
                let sampled_transform = matrix_transform(sampled)?;
                let mut transform = matrix_transform(sampled_transform_matrix)?;
                if use_location {
                    transform.translation = sampled_transform.translation;
                }
                if use_rotation {
                    transform.rotation = sampled_transform.rotation;
                }
                if use_scale {
                    transform.scale = sampled_transform.scale;
                }
                sampled_transform_matrix = transform.matrix();
            }
        }
        _ => return Err(evaluation_error("Geometry Attribute data_type is invalid")),
    }
    let mut result = action_mix_matrix(source, sampled_transform_matrix, mix_mode)?;
    if apply_target {
        result = target * result;
    }
    if finite_matrix(result) {
        Ok(result)
    } else {
        Err(evaluation_error("Geometry Attribute result is non-finite"))
    }
}

fn sample_tracking_object_pose(object: &crate::model::TrackingObject, frame: f64) -> Result<DMat4> {
    let mut before = None;
    let mut after = None;
    for pose in &object.reconstruction {
        let matrix = DMat4::from_cols_array(&pose.matrix);
        if pose.frame <= frame {
            before = Some((pose.frame, matrix));
        }
        if pose.frame >= frame {
            after = Some((pose.frame, matrix));
            break;
        }
    }
    interpolate_pose_matrices(
        before
            .or(after)
            .ok_or_else(|| evaluation_error("object reconstruction is empty"))?,
        after
            .or(before)
            .ok_or_else(|| evaluation_error("object reconstruction is empty"))?,
        frame,
    )
}

fn sample_tracking_marker(track: &crate::model::TrackingTrack, frame: f64) -> Result<[f64; 2]> {
    let mut before = None;
    let mut after = None;
    for marker in track.markers.iter().filter(|marker| !marker.disabled) {
        if marker.frame <= frame {
            before = Some((marker.frame, marker.co));
        }
        if marker.frame >= frame {
            after = Some((marker.frame, marker.co));
            break;
        }
    }
    let (before_frame, before_co) = before
        .or(after)
        .ok_or_else(|| evaluation_error("follow-track constraint has no enabled markers"))?;
    let (after_frame, after_co) = after
        .or(before)
        .ok_or_else(|| evaluation_error("follow-track constraint has no enabled markers"))?;
    if crate::float::equal_f64(before_frame, after_frame) {
        return Ok(before_co);
    }
    let factor = ((frame - before_frame) / (after_frame - before_frame)).clamp(0.0, 1.0);
    Ok([
        before_co[0] + (after_co[0] - before_co[0]) * factor,
        before_co[1] + (after_co[1] - before_co[1]) * factor,
    ])
}

fn ray_triangle_distance(
    origin: DVec3,
    direction: DVec3,
    a: DVec3,
    b: DVec3,
    c: DVec3,
) -> Option<f64> {
    let edge_a = b - a;
    let edge_b = c - a;
    let cross = direction.cross(edge_b);
    let determinant = edge_a.dot(cross);
    if determinant.abs() <= f64::EPSILON {
        return None;
    }
    let inverse_determinant = determinant.recip();
    let offset = origin - a;
    let u = offset.dot(cross) * inverse_determinant;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let cross = offset.cross(edge_a);
    let v = direction.dot(cross) * inverse_determinant;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let distance = edge_b.dot(cross) * inverse_determinant;
    (distance >= 0.0).then_some(distance)
}

fn depth_object_ray_hit(
    doc: &SceneDoc,
    object_id: &Id,
    object_world: DMat4,
    origin: DVec3,
    direction: DVec3,
) -> Result<Option<DVec3>> {
    let node = doc.nodes.get(object_id).ok_or_else(|| {
        evaluation_error_with(
            "Follow Track depth object does not exist",
            json!({"depth_object":object_id}),
        )
    })?;
    if !node.modifiers.is_empty() {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "Follow Track depth_object ray casting requires evaluated modifier geometry",
            json!({
                "feature_id":"constraint.follow_track.depth_object_modifiers",
                "depth_object":object_id,
            }),
        ));
    }
    let mesh = node
        .data
        .as_ref()
        .and_then(|data_id| doc.data_blocks.get(data_id))
        .and_then(|data| data.mesh.as_ref())
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::UnsupportedFeature,
                "Follow Track depth_object ray casting requires mesh data",
                json!({
                    "feature_id":"constraint.follow_track.depth_object_mesh",
                    "depth_object":object_id,
                }),
            )
        })?;
    let object_inverse = checked_inverse(object_world, "Follow Track depth object")?;
    let local_origin = object_inverse.transform_point3(origin);
    let local_direction = object_inverse
        .transform_vector3(direction)
        .normalize_or_zero();
    if local_direction.length_squared() <= f64::EPSILON {
        return Err(evaluation_error("Follow Track depth ray is degenerate"));
    }
    let vertex_position = |vertex_id: u32| {
        mesh.vertices
            .iter()
            .find(|vertex| vertex.id == vertex_id)
            .map(|vertex| vertex.co)
            .ok_or_else(|| evaluation_error("Follow Track depth mesh has invalid topology"))
    };
    let mut nearest = None;
    for face in &mesh.faces {
        if face.vertices.len() < 3 {
            continue;
        }
        let first = vertex_position(face.vertices[0])?;
        for index in 1..face.vertices.len() - 1 {
            let second = vertex_position(face.vertices[index])?;
            let third = vertex_position(face.vertices[index + 1])?;
            if let Some(distance) =
                ray_triangle_distance(local_origin, local_direction, first, second, third)
                && nearest.is_none_or(|current| distance < current)
            {
                nearest = Some(distance);
            }
        }
    }
    Ok(nearest
        .map(|distance| object_world.transform_point3(local_origin + local_direction * distance)))
}

fn follow_track_matrix(
    source: DMat4,
    camera_world: DMat4,
    constraint: &Constraint,
    doc: &SceneDoc,
    frame: f64,
    world_matrices: &BTreeMap<Id, DMat4>,
    world_to_owner: Option<DMat4>,
) -> Result<DMat4> {
    let clip = constraint_clip(doc, constraint)?;
    let track_name = string_param(&constraint.params, "track")?;
    let track = clip
        .tracking
        .tracks
        .iter()
        .find(|track| track.id == track_name || track.name == track_name)
        .ok_or_else(|| {
            evaluation_error_with(
                "Follow Track constraint track does not exist",
                json!({"constraint_id":constraint.id,"track":track_name}),
            )
        })?;
    let track_id = track.id.as_str();
    let use_3d = param_bool(&constraint.params, "use_3d_position", false)?;
    if use_3d {
        let point = clip
            .tracking
            .reconstruction
            .points
            .iter()
            .find(|point| point.track == track_id)
            .ok_or_else(|| {
                evaluation_error_with(
                    "Follow Track 3D position requires a reconstructed point for its track",
                    json!({"constraint_id":constraint.id,"track":track_id}),
                )
            })?;
        let mut reconstruction_parent = camera_world;
        if let Some(object_id) = constraint
            .params
            .get("object")
            .and_then(Value::as_str)
            .filter(|object_id| !object_id.is_empty())
        {
            let object = clip
                .tracking
                .objects
                .iter()
                .find(|object| object.id == object_id || object.name == object_id)
                .ok_or_else(|| {
                    evaluation_error_with(
                        "Follow Track constraint tracking object does not exist",
                        json!({"constraint_id":constraint.id,"object":object_id}),
                    )
                })?;
            reconstruction_parent *= checked_inverse(
                sample_tracking_object_pose(object, frame)?,
                "Follow Track object reconstruction",
            )?;
        }
        let mut result = source * reconstruction_parent;
        result.w_axis = (result.w_axis.truncate()
            + result.transform_vector3(DVec3::from_array(point.co)))
        .extend(1.0);
        return Ok(result);
    }
    let mut image = sample_tracking_marker(track, frame)?;
    if clip.width == 0 || clip.height == 0 {
        return Err(evaluation_error_with(
            "Follow Track requires movie-clip image dimensions for 2D marker sampling",
            json!({"constraint_id":constraint.id,"clip":clip.name}),
        ));
    }
    let clip_width = f64::from(clip.width);
    let clip_height = f64::from(clip.height);
    image[0] /= clip_width;
    image[1] /= clip_height;
    if param_bool(&constraint.params, "use_undistorted_position", false)? {
        let intrinsics = &clip.tracking.camera;
        image = crate::tracking::LensDistortion {
            k1: intrinsics.k1,
            k2: intrinsics.k2,
            k3: intrinsics.k3,
            p1: 0.0,
            p2: 0.0,
            center: intrinsics.principal,
            scale: [0.5, 0.5],
        }
        .undistort(image)
        .map_err(|error| evaluation_error(error.message))?;
    }
    let frame_method = string_param_default(&constraint.params, "frame_method", "STRETCH")?;
    let scene = doc
        .scenes
        .get(&doc.active_scene)
        .ok_or_else(|| evaluation_error("Follow Track active scene is unavailable"))?;
    let render_width = f64::from(scene.render.resolution_x);
    let render_height = f64::from(scene.render.resolution_y);
    if render_width <= 0.0 || render_height <= 0.0 {
        return Err(evaluation_error(
            "Follow Track render dimensions must be positive",
        ));
    }
    let clip_aspect = clip_width / clip_height;
    let render_aspect = render_width / render_height;
    let (display_width, display_height) = match frame_method {
        "STRETCH" => (1.0, 1.0),
        "FIT" if clip_aspect > render_aspect => (1.0, render_aspect / clip_aspect),
        "FIT" => (clip_aspect / render_aspect, 1.0),
        "CROP" if clip_aspect > render_aspect => (clip_aspect / render_aspect, 1.0),
        "CROP" => (1.0, render_aspect / clip_aspect),
        _ => return Err(evaluation_error("Follow Track frame_method is invalid")),
    };
    image[0] = image[0] * display_width + (1.0 - display_width) * 0.5;
    image[1] = image[1] * display_height + (1.0 - display_height) * 0.5;
    let camera_id = constraint
        .target
        .as_ref()
        .ok_or_else(|| evaluation_error("Follow Track camera target is missing"))?;
    let camera_node = doc.nodes.get(camera_id).ok_or_else(|| {
        evaluation_error_with(
            "Follow Track camera target does not exist",
            json!({"constraint_id":constraint.id,"camera":camera_id}),
        )
    })?;
    let camera_data = camera_node
        .data
        .as_ref()
        .and_then(|data_id| doc.data_blocks.get(data_id))
        .and_then(|data| data.camera.as_ref())
        .ok_or_else(|| evaluation_error("Follow Track target has no camera data"))?;
    let intrinsics = &clip.tracking.camera;
    let (ray_origin_local, direction) =
        if camera_data.projection == crate::model::CameraProjection::Orthographic {
            let mut x = camera_data.ortho_scale
                * (image[0] - intrinsics.principal[0] + camera_data.shift[0]);
            let mut y = camera_data.ortho_scale
                * (intrinsics.principal[1] - image[1] + camera_data.shift[1]);
            if render_aspect > 1.0 {
                y /= render_aspect;
            } else {
                x *= render_aspect;
            }
            (DVec3::new(x, y, 0.0), DVec3::NEG_Z)
        } else {
            let focal = camera_data.lens_mm / camera_data.sensor_width_mm;
            if !focal.is_finite() || focal <= 0.0 {
                return Err(evaluation_error(
                    "Follow Track camera focal length is invalid",
                ));
            }
            let mut x = (image[0] + camera_data.shift[0] - intrinsics.principal[0]) / focal;
            let mut y = (intrinsics.principal[1] - image[1] + camera_data.shift[1]) / focal;
            if render_aspect > 1.0 {
                y /= render_aspect;
            } else {
                x *= render_aspect;
            }
            (DVec3::ZERO, DVec3::new(x, y, -1.0))
        };
    let camera_inverse = checked_inverse(camera_world, "Follow Track camera")?;
    let owner_camera_position = camera_inverse.transform_point3(source.w_axis.truncate());
    let object_depth = owner_camera_position.z.abs();
    if object_depth <= f64::EPSILON {
        return Ok(source);
    }
    let location = if let Some(depth_id) = constraint
        .params
        .get("depth_object")
        .and_then(Value::as_str)
    {
        let depth_id = Id::new(depth_id.to_owned()).map_err(|error| {
            evaluation_error_with(
                "Follow Track depth_object ID is invalid",
                json!({"constraint_id":constraint.id,"reason":error.message}),
            )
        })?;
        let world_matrix = world_matrices.get(&depth_id).copied().ok_or_else(|| {
            evaluation_error_with(
                "Follow Track depth object transform is unavailable",
                json!({"constraint_id":constraint.id,"depth_object":depth_id}),
            )
        })?;
        let depth_matrix = world_to_owner.map_or(world_matrix, |matrix| matrix * world_matrix);
        let ray_direction = camera_world
            .transform_vector3(direction)
            .normalize_or_zero();
        if ray_direction.length_squared() <= f64::EPSILON {
            return Err(evaluation_error("Follow Track camera ray is degenerate"));
        }
        let Some(location) = depth_object_ray_hit(
            doc,
            &depth_id,
            depth_matrix,
            camera_world.transform_point3(ray_origin_local),
            ray_direction,
        )?
        else {
            return Ok(source);
        };
        location
    } else {
        camera_world.transform_point3(ray_origin_local + direction * object_depth)
    };
    let mut camera_rotation = camera_world;
    camera_rotation.w_axis = glam::DVec4::W;
    let mut result = source * camera_rotation;
    result.w_axis = location.extend(1.0);
    Ok(result)
}

#[expect(
    clippy::too_many_arguments,
    reason = "space conversion uses evaluated owner, target, and bone context"
)]
fn constraint_space_world_matrix(
    constraint: &Constraint,
    space: &str,
    node_id: &Id,
    node_world: DMat4,
    bone_id: Option<&Id>,
    doc: &SceneDoc,
    frame: f64,
    world_matrices: &BTreeMap<Id, DMat4>,
    bone_matrices: Option<&BTreeMap<Id, BoneMatrices>>,
) -> Result<DMat4> {
    let node = doc.nodes.get(node_id).ok_or_else(|| {
        evaluation_error_with(
            "constraint space owner does not exist",
            json!({"constraint_id":constraint.id,"node_id":node_id}),
        )
    })?;
    match space {
        "CUSTOM" => {
            let space_object = string_param(&constraint.params, "space_object")?;
            let space_id = Id::new(space_object.to_owned()).map_err(|error| {
                evaluation_error_with(
                    "constraint custom-space object ID is invalid",
                    json!({"constraint_id":constraint.id,"space_object":space_object,"reason":error.message}),
                )
            })?;
            world_matrices.get(&space_id).copied().ok_or_else(|| {
                evaluation_error_with(
                    "constraint custom-space object transform is unavailable",
                    json!({"constraint_id":constraint.id,"space_object":space_id}),
                )
            })
        }
        "LOCAL" if node.kind == "armature" && bone_id.is_some() => {
            let data = node
                .data
                .as_ref()
                .and_then(|data_id| doc.data_blocks.get(data_id))
                .and_then(|data| data.armature.as_ref())
                .ok_or_else(|| evaluation_error("constraint owner armature is unavailable"))?;
            let bone_id =
                bone_id.ok_or_else(|| evaluation_error("constraint owner bone is missing"))?;
            let bone = data
                .bones
                .get(bone_id)
                .ok_or_else(|| evaluation_error("constraint owner bone is unavailable"))?;
            let rest = bone_rest_matrix(bone_id, bone)?;
            let local = if let Some(parent_id) = &bone.parent {
                if let Some(parent_matrices) =
                    bone_matrices.and_then(|matrices| matrices.get(parent_id))
                {
                    let parent = data.bones.get(parent_id).ok_or_else(|| {
                        evaluation_error("constraint owner parent bone is unavailable")
                    })?;
                    parent_matrices.pose
                        * checked_inverse(
                            bone_rest_matrix(parent_id, parent)?,
                            "constraint owner parent rest",
                        )?
                        * rest
                } else {
                    rest
                }
            } else {
                rest
            };
            Ok(node_world * local)
        }
        "LOCAL_WITH_PARENT" if node.kind == "armature" && bone_id.is_some() => {
            let data = node
                .data
                .as_ref()
                .and_then(|data_id| doc.data_blocks.get(data_id))
                .and_then(|data| data.armature.as_ref())
                .ok_or_else(|| evaluation_error("constraint owner armature is unavailable"))?;
            let bone_id =
                bone_id.ok_or_else(|| evaluation_error("constraint owner bone is missing"))?;
            let bone = data
                .bones
                .get(bone_id)
                .ok_or_else(|| evaluation_error("constraint owner bone is unavailable"))?;
            Ok(node_world * bone_rest_matrix(bone_id, bone)?)
        }
        "POSE" if node.kind == "armature" && bone_id.is_some() => Ok(node_world),
        "LOCAL" => {
            if let Some(parent_id) = &node.parent {
                let parent_world = world_matrices.get(parent_id).copied().ok_or_else(|| {
                    evaluation_error_with(
                        "constraint local-space parent transform is unavailable",
                        json!({"constraint_id":constraint.id,"parent":parent_id}),
                    )
                })?;
                let parent_inverse = node
                    .parent_inverse
                    .map_or(DMat4::IDENTITY, |matrix| DMat4::from_cols_array(&matrix));
                Ok(parent_world * parent_inverse)
            } else {
                let transform = animated_transform(node, doc, frame)?;
                let rotation = DQuat::from_xyzw(
                    transform.rotation[0],
                    transform.rotation[1],
                    transform.rotation[2],
                    transform.rotation[3],
                )
                .normalize();
                Ok(DMat4::from_rotation_translation(rotation, DVec3::ZERO))
            }
        }
        "WORLD" | "POSE" | "LOCAL_WITH_PARENT" => Ok(DMat4::IDENTITY),
        _ => Err(evaluation_error_with(
            "constraint space is invalid",
            json!({"constraint_id":constraint.id,"space":space}),
        )),
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "constraint conversion needs owner and target evaluation context"
)]
fn apply_object_constraint(
    source: DMat4,
    target: DMat4,
    constraint: &Constraint,
    doc: &SceneDoc,
    frame: f64,
    world_matrices: &BTreeMap<Id, DMat4>,
    evaluated_meshes: Option<&BTreeMap<Id, Mesh>>,
    owner_node_id: Option<&Id>,
    owner_space_node_id: &Id,
    owner_world: DMat4,
    owner_bone_matrices: Option<&BTreeMap<Id, BoneMatrices>>,
    world_to_owner: Option<DMat4>,
) -> Result<DMat4> {
    if constraint.constraint_type == ConstraintType::ClampTo
        && (string_param_default(&constraint.params, "owner_space", "WORLD")? == "CUSTOM"
            || string_param_default(&constraint.params, "target_space", "WORLD")? == "CUSTOM")
    {
        return Ok(source);
    }
    if constraint.constraint_type == ConstraintType::FollowPath
        && (string_param_default(&constraint.params, "owner_space", "WORLD")? == "CUSTOM"
            || string_param_default(&constraint.params, "target_space", "WORLD")? == "CUSTOM")
    {
        return Ok(source);
    }
    if constraint.constraint_type == ConstraintType::TransformCache {
        let influence = checked_influence(constraint)?;
        if influence == 0.0 {
            return Ok(source);
        }
        let owner_space = string_param_default(&constraint.params, "owner_space", "WORLD")?;
        let owner_to_world = constraint_space_world_matrix(
            constraint,
            owner_space,
            owner_space_node_id,
            owner_world,
            constraint.owner_bone.as_ref(),
            doc,
            frame,
            world_matrices,
            owner_bone_matrices,
        )?;
        let world_from_owner = if let Some(world_to_owner) = world_to_owner {
            checked_inverse(world_to_owner, "Transform Cache owner")?
        } else {
            DMat4::IDENTITY
        };
        let source_in_space = checked_inverse(owner_to_world, "Transform Cache owner space")?
            * (world_from_owner * source);
        let constrained_in_space = if influence >= 1.0 {
            target
        } else {
            blend_transform(source_in_space, target, influence)?
        };
        let constrained_world = owner_to_world * constrained_in_space;
        let result = world_to_owner.map_or(constrained_world, |world_to_owner| {
            world_to_owner * constrained_world
        });
        return if finite_matrix(result) {
            Ok(result)
        } else {
            Err(evaluation_error(
                "Transform Cache constraint result is non-finite",
            ))
        };
    }
    let influence = checked_influence(constraint)?;
    if influence == 0.0 {
        return Ok(source);
    }
    if constraint.constraint_type == ConstraintType::Armature {
        armature_constraint_bbone_gate(doc, constraint)?;
        if armature_constraint_has_zero_envelope_weight(
            doc,
            constraint,
            owner_node_id,
            owner_world,
            source,
            world_matrices,
        )? {
            return Ok(source);
        }
    }
    let owner_space = string_param_default(&constraint.params, "owner_space", "WORLD")?;
    let owner_to_world = constraint_space_world_matrix(
        constraint,
        owner_space,
        owner_space_node_id,
        owner_world,
        constraint.owner_bone.as_ref(),
        doc,
        frame,
        world_matrices,
        owner_bone_matrices,
    )?;
    let source_in_space = checked_inverse(owner_to_world, "constraint owner space")? * source;
    let target_space = string_param_default(&constraint.params, "target_space", "WORLD")?;
    let target_in_space = if let Some(target_id) = &constraint.target {
        let target_node_world = if target_id == owner_space_node_id {
            owner_world
        } else {
            world_matrices.get(target_id).copied().unwrap_or(target)
        };
        let target_to_world = constraint_space_world_matrix(
            constraint,
            target_space,
            target_id,
            target_node_world,
            constraint.subtarget.as_ref(),
            doc,
            frame,
            world_matrices,
            if target_id == owner_space_node_id {
                owner_bone_matrices
            } else {
                None
            },
        )?;
        checked_inverse(target_to_world, "constraint target space")? * target
    } else {
        target
    };
    let result = apply_object_constraint_in_space(
        source_in_space,
        target_in_space,
        constraint,
        doc,
        frame,
        world_matrices,
        evaluated_meshes,
        owner_node_id,
        world_to_owner,
        Some(1.0),
    )?;
    let result_world = owner_to_world * result;
    if influence >= 1.0 {
        Ok(result_world)
    } else {
        blend_transform(source, result_world, influence)
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "constraint variant dispatch keeps shared blending and error handling consistent"
)]
fn apply_object_constraint_in_space(
    source: DMat4,
    target: DMat4,
    constraint: &Constraint,
    doc: &SceneDoc,
    frame: f64,
    world_matrices: &BTreeMap<Id, DMat4>,
    evaluated_meshes: Option<&BTreeMap<Id, Mesh>>,
    owner_node_id: Option<&Id>,
    world_to_owner: Option<DMat4>,
    influence_override: Option<f64>,
) -> Result<DMat4> {
    let influence = if let Some(influence) = influence_override {
        influence
    } else {
        checked_influence(constraint)?
    };
    let source_transform = matrix_transform(source)?;
    let target_transform = matrix_transform(target)?;
    let source_rotation = source_transform.rotation;
    let source_quat = DQuat::from_xyzw(
        source_rotation[0],
        source_rotation[1],
        source_rotation[2],
        source_rotation[3],
    );
    let source_scale = source_transform.scale;
    let source_translation = source_transform.translation;
    let mut desired = source_transform;
    match constraint.constraint_type {
        ConstraintType::CopyTransforms => {
            let mix_mode = string_param_default(&constraint.params, "mix_mode", "REPLACE")?;
            let mixed = match mix_mode {
                "REPLACE" => target,
                "BEFORE_FULL" | "BEFORE" => target * source,
                "AFTER_FULL" | "AFTER" => source * target,
                "BEFORE_SPLIT" | "AFTER_SPLIT" => {
                    let target_rotation = DQuat::from_xyzw(
                        target_transform.rotation[0],
                        target_transform.rotation[1],
                        target_transform.rotation[2],
                        target_transform.rotation[3],
                    );
                    let target_scale = DVec3::from_array(target_transform.scale);
                    let source_scale_vector = DVec3::from_array(source_scale);
                    let source_translation = DVec3::from_array(source_translation);
                    let target_translation = DVec3::from_array(target_transform.translation);
                    let translation = if mix_mode == "BEFORE_SPLIT" {
                        target_translation + target_rotation * (target_scale * source_translation)
                    } else {
                        source_translation
                            + source_quat * (source_scale_vector * target_translation)
                    };
                    let scale = source_scale_vector * target_scale;
                    let rotation = if mix_mode == "BEFORE_SPLIT" {
                        target_rotation * source_quat
                    } else {
                        source_quat * target_rotation
                    };
                    DMat4::from_scale_rotation_translation(scale, rotation, translation)
                }
                _ => {
                    return Err(evaluation_error_with(
                        "copy-transforms mix_mode is invalid",
                        json!({"constraint_id":constraint.id,"mix_mode":mix_mode}),
                    ));
                }
            };
            let mixed = matrix_transform(mixed)?;
            desired.translation = mixed.translation;
            desired.rotation = mixed.rotation;
            desired.scale = mixed.scale;
        }
        ConstraintType::Armature => {
            let weight = constraint
                .params
                .get("targets")
                .and_then(Value::as_array)
                .and_then(|targets| targets.first())
                .and_then(|target| target.get("weight"))
                .and_then(Value::as_f64)
                .unwrap_or(1.0)
                .clamp(0.0, 1.0);
            let weighted_target = if weight < 1.0 {
                matrix_transform(blend_transform(source, target, weight)?)?
            } else {
                target_transform.clone()
            };
            desired.translation = weighted_target.translation;
            desired.rotation = weighted_target.rotation;
            desired.scale = weighted_target.scale;
        }
        ConstraintType::CopyLocation => {
            for axis in 0..3 {
                if param_bool(&constraint.params, ["use_x", "use_y", "use_z"][axis], true)? {
                    desired.translation[axis] = target_transform.translation[axis];
                }
            }
        }
        ConstraintType::CopyRotation => {
            let (source_x, source_y, source_z) = DQuat::from_xyzw(
                source_rotation[0],
                source_rotation[1],
                source_rotation[2],
                source_rotation[3],
            )
            .to_euler(EulerRot::XYZEx);
            let (target_x, target_y, target_z) = DQuat::from_xyzw(
                target_transform.rotation[0],
                target_transform.rotation[1],
                target_transform.rotation[2],
                target_transform.rotation[3],
            )
            .to_euler(EulerRot::XYZEx);
            let source_angles = [source_x, source_y, source_z];
            let target_angles = [target_x, target_y, target_z];
            let mut copied_angles = [0.0; 3];
            let mut use_axis = [false; 3];
            let mut invert_axis = [false; 3];
            for axis in 0..3 {
                use_axis[axis] =
                    param_bool(&constraint.params, ["use_x", "use_y", "use_z"][axis], true)?;
                if use_axis[axis] {
                    invert_axis[axis] = param_bool(
                        &constraint.params,
                        ["invert_x", "invert_y", "invert_z"][axis],
                        false,
                    )?;
                    copied_angles[axis] = if invert_axis[axis] {
                        -target_angles[axis]
                    } else {
                        target_angles[axis]
                    };
                }
            }
            let mix_mode = string_param_default(&constraint.params, "mix_mode", "REPLACE")?;
            let rotation = match mix_mode {
                "REPLACE" => DQuat::from_euler(
                    EulerRot::XYZEx,
                    if use_axis[0] {
                        copied_angles[0]
                    } else {
                        source_angles[0]
                    },
                    if use_axis[1] {
                        copied_angles[1]
                    } else {
                        source_angles[1]
                    },
                    if use_axis[2] {
                        copied_angles[2]
                    } else {
                        source_angles[2]
                    },
                ),
                "ADD" => DQuat::from_euler(
                    EulerRot::XYZEx,
                    source_angles[0] + copied_angles[0],
                    source_angles[1] + copied_angles[1],
                    source_angles[2] + copied_angles[2],
                ),
                "BEFORE" => {
                    DQuat::from_euler(
                        EulerRot::XYZEx,
                        copied_angles[0],
                        copied_angles[1],
                        copied_angles[2],
                    ) * source_quat
                }
                "AFTER" => {
                    source_quat
                        * DQuat::from_euler(
                            EulerRot::XYZEx,
                            copied_angles[0],
                            copied_angles[1],
                            copied_angles[2],
                        )
                }
                "OFFSET" => {
                    let mut offset_angles = target_angles;
                    for axis in 0..3 {
                        if use_axis[axis] {
                            let axis_rotation = DQuat::from_axis_angle(
                                [DVec3::X, DVec3::Y, DVec3::Z][axis],
                                source_angles[axis],
                            );
                            let euler_rotation = DQuat::from_euler(
                                EulerRot::XYZEx,
                                offset_angles[0],
                                offset_angles[1],
                                offset_angles[2],
                            );
                            let (x, y, z) =
                                (euler_rotation * axis_rotation).to_euler(EulerRot::XYZEx);
                            offset_angles = [x, y, z];
                            if invert_axis[axis] {
                                offset_angles[axis] = -offset_angles[axis];
                            }
                        } else {
                            offset_angles[axis] = source_angles[axis];
                        }
                    }
                    DQuat::from_euler(
                        EulerRot::XYZEx,
                        offset_angles[0],
                        offset_angles[1],
                        offset_angles[2],
                    )
                }
                _ => {
                    return Err(evaluation_error_with(
                        "copy-rotation mix_mode is invalid",
                        json!({"constraint_id":constraint.id,"mix_mode":mix_mode}),
                    ));
                }
            };
            desired.rotation = [rotation.x, rotation.y, rotation.z, rotation.w];
        }
        ConstraintType::CopyScale => {
            for axis in 0..3 {
                if param_bool(&constraint.params, ["use_x", "use_y", "use_z"][axis], true)? {
                    desired.scale[axis] = target_transform.scale[axis];
                }
            }
        }
        ConstraintType::LimitDistance => {
            let distance_limit = number_param(&constraint.params, "distance")?.unwrap_or(0.0);
            if !distance_limit.is_finite() || distance_limit < 0.0 {
                return Err(evaluation_error(
                    "limit-distance value must be non-negative",
                ));
            }
            let offset = source.w_axis.truncate() - target.w_axis.truncate();
            let distance_limit = if distance_limit == 0.0 {
                offset.length()
            } else {
                distance_limit
            };
            let distance = offset.length();
            let mode = string_param_default(&constraint.params, "limit_mode", "LIMITDIST_INSIDE")?;
            let (clamp_surface, scale) = match mode {
                "LIMITDIST_OUTSIDE" => {
                    let clamp_surface = distance <= distance_limit && distance > f64::EPSILON;
                    (
                        clamp_surface,
                        if clamp_surface {
                            distance_limit / distance
                        } else {
                            1.0
                        },
                    )
                }
                "LIMITDIST_INSIDE" => {
                    if distance >= distance_limit && distance > f64::EPSILON {
                        (true, distance_limit / distance)
                    } else {
                        (false, 1.0)
                    }
                }
                "LIMITDIST_ONSURFACE" => {
                    let clamp_surface = (distance - distance_limit).abs() > f64::EPSILON;
                    (
                        clamp_surface,
                        if clamp_surface && distance > f64::EPSILON {
                            distance_limit / distance
                        } else {
                            1.0
                        },
                    )
                }
                _ => {
                    return Err(evaluation_error_with(
                        "limit-distance mode is invalid",
                        json!({"constraint_id":constraint.id,"limit_mode":mode}),
                    ));
                }
            };
            if clamp_surface {
                desired.translation = (target.w_axis.truncate() + offset * scale).to_array();
            }
        }
        ConstraintType::FollowPath => {
            let target_id = constraint.target.as_ref().ok_or_else(|| {
                evaluation_error("follow-path constraint requires a curve target")
            })?;
            let (curve, spline, samples) =
                constraint_curve_path_samples(doc, target_id, constraint)?;
            if curve.use_path {
                let path = samples
                    .iter()
                    .map(|sample| sample.position)
                    .collect::<Vec<_>>();
                let (cumulative, length) = curve_path_lengths(&path)?;
                let factor = if param_bool(&constraint.params, "use_fixed_location", false)? {
                    number_param(&constraint.params, "offset_factor")?.unwrap_or(0.0)
                } else {
                    let duration = f64::from(curve.path_duration);
                    if duration <= 0.0 {
                        return Err(evaluation_error(
                            "follow-path target path_duration must be positive",
                        ));
                    }
                    (curve_eval_time(curve, frame)?
                        - number_param(&constraint.params, "offset")?.unwrap_or(0.0))
                        / duration
                };
                if !factor.is_finite() {
                    return Err(evaluation_error("follow-path factor is non-finite"));
                }
                let sample = curve_path_sample_attributes(
                    &samples,
                    &cumulative,
                    path_sample_factor(factor, spline.cyclic) * length,
                )?;
                let local_rotation = if param_bool(&constraint.params, "use_curve_follow", false)? {
                    // Blender's path quaternion stores the tangent on local +Z and its up
                    // reference on local +Y.
                    let direction = (sample.orientation * DVec3::Z).normalize_or_zero();
                    if direction.length_squared() <= f64::EPSILON {
                        return Err(evaluation_error("follow-path curve tangent is degenerate"));
                    }
                    let forward = follow_path_forward_axis(&constraint.params)?;
                    let up = up_axis(&constraint.params)?;
                    let curve_up = sample.orientation * DVec3::Y;
                    blender_track_rotation(forward, up, direction, curve_up)
                } else {
                    DQuat::IDENTITY
                };
                let use_radius = param_bool(&constraint.params, "use_curve_radius", false)?;
                let radius = if use_radius { sample.radius } else { 1.0 };
                let path_matrix = target
                    * DMat4::from_scale_rotation_translation(
                        DVec3::splat(radius),
                        local_rotation,
                        sample.position,
                    );
                let mut followed = path_matrix * source;
                if !use_radius {
                    let (owner_size, _, _) = source.to_scale_rotation_translation();
                    let (path_size, _, _) = followed.to_scale_rotation_translation();
                    if path_size.x.abs() > f64::EPSILON {
                        followed.x_axis *= owner_size.x / path_size.x;
                    }
                    if path_size.y.abs() > f64::EPSILON {
                        followed.y_axis *= owner_size.y / path_size.y;
                    }
                    if path_size.z.abs() > f64::EPSILON {
                        followed.z_axis *= owner_size.z / path_size.z;
                    }
                }
                let (scale, rotation, translation) = followed.to_scale_rotation_translation();
                desired.translation = translation.to_array();
                desired.rotation = [rotation.x, rotation.y, rotation.z, rotation.w];
                desired.scale = scale.to_array();
            }
        }
        ConstraintType::ClampTo => {
            let target_id = constraint
                .target
                .as_ref()
                .ok_or_else(|| evaluation_error("clamp-to constraint requires a curve target"))?;
            let (curve, spline, samples) =
                constraint_curve_path_samples(doc, target_id, constraint)?;
            if curve.use_path {
                let (minimum, maximum) = constraint_curve_bounds(curve, target)?;
                let axis = clamp_to_axis_index(
                    minimum,
                    maximum,
                    string_param_default(&constraint.params, "main_axis", "CLAMPTO_AUTO")?,
                )?;
                let factor = clamp_to_factor(
                    source.w_axis.truncate()[axis],
                    minimum[axis],
                    maximum[axis],
                    param_bool(&constraint.params, "use_cyclic", false)?,
                );
                let path = samples
                    .iter()
                    .map(|sample| sample.position)
                    .collect::<Vec<_>>();
                let (cumulative, length) = curve_path_lengths(&path)?;
                let sample = curve_path_sample(
                    &path,
                    &cumulative,
                    path_sample_factor(factor, spline.cyclic) * length,
                )?;
                desired.translation = target.transform_point3(sample).to_array();
            }
        }
        ConstraintType::TrackTo | ConstraintType::DampedTrack => {
            let direction = target.w_axis.truncate() - source.w_axis.truncate();
            if direction.length_squared() > f64::EPSILON {
                let direction = direction.normalize();
                let rotation = if constraint.constraint_type == ConstraintType::TrackTo {
                    let track = track_axis(&constraint.params)?;
                    let up = up_axis(&constraint.params)?;
                    let up_hint = if param_bool(&constraint.params, "use_target_z", false)? {
                        target.transform_vector3(DVec3::Z).normalize_or_zero()
                    } else {
                        DVec3::Z
                    };
                    blender_track_rotation(track, up, direction, up_hint)
                } else {
                    let track = track_axis_with_default(&constraint.params, "TRACK_Y")?;
                    let track_enum = track_axis_enum(track)?;
                    blender_damped_track_rotation(source_quat, track, track_enum, direction)
                };
                desired.rotation = [rotation.x, rotation.y, rotation.z, rotation.w];
            }
        }
        ConstraintType::LockedTrack => {
            let direction = target.w_axis.truncate() - source.w_axis.truncate();
            if direction.length_squared() > f64::EPSILON {
                let track = axis_vector(string_param_default(
                    &constraint.params,
                    "track_axis",
                    "TRACK_Y",
                )?)
                .ok_or_else(|| evaluation_error("locked-track track_axis is invalid"))?;
                let lock = axis_vector(string_param_default(
                    &constraint.params,
                    "lock_axis",
                    "LOCK_Z",
                )?)
                .ok_or_else(|| evaluation_error("locked-track lock_axis is invalid"))?;
                if track.dot(lock).abs() > 1.0e-12 {
                    return Err(evaluation_error("locked-track axes must be perpendicular"));
                }
                if let Some(rotation) =
                    blender_locked_track_rotation(source_quat, track, lock, direction)
                {
                    desired.rotation = [rotation.x, rotation.y, rotation.z, rotation.w];
                }
            }
        }
        ConstraintType::StretchTo => {
            let position = source.w_axis.truncate();
            let direction = target.w_axis.truncate() - position;
            let distance = direction.length();
            if distance > f64::EPSILON {
                let normalized_distance = distance / source_scale[1].abs().max(f64::EPSILON);
                let mut rest_length =
                    number_param(&constraint.params, "rest_length")?.unwrap_or(0.0);
                if !rest_length.is_finite() || rest_length < 0.0 {
                    return Err(evaluation_error("stretch-to rest_length is invalid"));
                }
                if rest_length == 0.0 {
                    rest_length = normalized_distance;
                }
                let mut bulge = (rest_length / normalized_distance)
                    .powf(number_param(&constraint.params, "bulge")?.unwrap_or(1.0));
                if bulge > 1.0 && param_bool(&constraint.params, "use_bulge_max", false)? {
                    bulge = stretch_bulge_limit(
                        bulge,
                        number_param(&constraint.params, "bulge_max")?.unwrap_or(1.0),
                        number_param(&constraint.params, "bulge_smooth")?.unwrap_or(0.0),
                        true,
                    );
                } else if bulge < 1.0 && param_bool(&constraint.params, "use_bulge_min", false)? {
                    bulge = stretch_bulge_limit(
                        bulge,
                        number_param(&constraint.params, "bulge_min")?.unwrap_or(1.0),
                        number_param(&constraint.params, "bulge_smooth")?.unwrap_or(0.0),
                        false,
                    );
                }
                let volume = string_param_default(&constraint.params, "volume", "VOLUME_XZX")?;
                let scale_xz = match volume {
                    "VOLUME_XZX" => bulge.sqrt(),
                    "VOLUME_X" => bulge,
                    "VOLUME_Z" | "NO_VOLUME" => 1.0,
                    _ => return Err(evaluation_error("stretch-to volume mode is invalid")),
                };
                let scale_z = if volume == "VOLUME_Z" {
                    bulge
                } else if volume == "NO_VOLUME" || volume == "VOLUME_X" {
                    1.0
                } else {
                    scale_xz
                };
                desired.scale[0] = source_scale[0] * scale_xz;
                desired.scale[1] = source_scale[1] * normalized_distance / rest_length;
                desired.scale[2] = source_scale[2] * scale_z;
                let keep_axis = string_param_default(&constraint.params, "keep_axis", "SWING_Y")?;
                let rotation =
                    blender_stretch_to_rotation(source_quat, direction / distance, keep_axis)?;
                desired.rotation = [rotation.x, rotation.y, rotation.z, rotation.w];
            }
        }
        ConstraintType::Transformation => {
            let owner_rotation_mode = owner_node_id
                .and_then(|owner_id| doc.nodes.get(owner_id))
                .map_or("XYZ", |node| node.transform.rotation_mode.as_str());
            desired = transformation_constraint(
                &desired,
                &target_transform,
                target.determinant() < 0.0,
                constraint,
                owner_rotation_mode,
            )?;
        }
        ConstraintType::MaintainVolume => {
            let axis_name = string_param_default(&constraint.params, "free_axis", "SAMEVOL_Y")?;
            let free_axis = axis_index(
                axis_vector(axis_name)
                    .ok_or_else(|| evaluation_error("maintain-volume free_axis is invalid"))?,
            )?;
            let base_volume = number_param(&constraint.params, "volume")?.unwrap_or(1.0);
            let cross_section = (0..3)
                .filter(|axis| *axis != free_axis)
                .map(|axis| desired.scale[axis].abs())
                .product::<f64>();
            let free_scale = desired.scale[free_axis].abs();
            if free_scale <= f64::EPSILON || cross_section <= f64::EPSILON {
                return Err(evaluation_error("maintain-volume scale must be nonzero"));
            }
            let correction = (base_volume / (free_scale * cross_section)).sqrt();
            for axis in 0..3 {
                if axis != free_axis {
                    desired.scale[axis] *= correction;
                }
            }
        }
        ConstraintType::Floor => {
            let axis = string_param_default(&constraint.params, "floor_location", "FLOOR_Z")?;
            let local_normal =
                axis_vector(axis).ok_or_else(|| evaluation_error("floor axis is invalid"))?;
            let target_rotation = DQuat::from_xyzw(
                target_transform.rotation[0],
                target_transform.rotation[1],
                target_transform.rotation[2],
                target_transform.rotation[3],
            );
            let normal = (target_rotation * local_normal).normalize_or_zero();
            let offset = number_param(&constraint.params, "offset")?.unwrap_or(0.0);
            let plane_point = target.w_axis.truncate();
            let position = source.w_axis.truncate();
            let distance = (position - plane_point).dot(normal);
            if distance < offset {
                desired.translation = (position + normal * (offset - distance)).to_array();
            }
        }
        ConstraintType::Pivot => {
            let owner_position = source.w_axis.truncate();
            let rotation_range =
                string_param_default(&constraint.params, "rotation_range", "ALWAYS_ACTIVE")?;
            let euler = blender_euler_xyz(source_quat);
            let inactive = match rotation_range {
                "ALWAYS_ACTIVE" => false,
                "NX" => euler[0] > 0.0,
                "NY" => euler[1] > 0.0,
                "NZ" => euler[2] > 0.0,
                "X" => euler[0] < 0.0,
                "Y" => euler[1] < 0.0,
                "Z" => euler[2] < 0.0,
                _ => return Err(evaluation_error("pivot rotation_range is invalid")),
            };
            if !inactive {
                let offset = number_array_param(&constraint.params, "offset")?.unwrap_or([0.0; 3]);
                let mut pivot = if constraint.target.is_some() {
                    target.w_axis.truncate() + DVec3::from_array(offset)
                } else if param_bool(&constraint.params, "use_relative_location", true)? {
                    owner_position + DVec3::from_array(offset)
                } else {
                    DVec3::from_array(offset)
                };
                let (axis, angle) = source_quat.to_axis_angle();
                if angle.abs() > f64::EPSILON {
                    let pivot_offset = pivot - owner_position;
                    pivot -= axis * pivot_offset.dot(axis);
                }
                desired.translation = (pivot + source_quat * (owner_position - pivot)).to_array();
            }
        }
        ConstraintType::Shrinkwrap => {
            let target_id = constraint
                .target
                .as_ref()
                .ok_or_else(|| evaluation_error("shrinkwrap constraint requires a target"))?;
            let position = source.w_axis.truncate();
            let distance = number_param(&constraint.params, "distance")?.unwrap_or(0.0);
            if !distance.is_finite() || distance < 0.0 {
                return Err(evaluation_error(
                    "shrinkwrap distance must be finite and non-negative",
                ));
            }
            let shrink_type =
                string_param_default(&constraint.params, "shrinkwrap_type", "NEAREST_SURFACE")?;
            let hit = shrinkwrap_target_hit(
                doc,
                target_id,
                target,
                evaluated_meshes,
                position,
                source_quat,
                &constraint.params,
                shrink_type,
            )?;
            if let Some(hit) = hit {
                let snapped = if shrink_type == "NEAREST_VERTEX" {
                    let separation = position.distance(hit.point);
                    if separation > f64::EPSILON {
                        position + (hit.point - position) * ((separation - distance) / separation)
                    } else {
                        hit.point
                    }
                } else {
                    shrinkwrap_snap(
                        position,
                        hit,
                        distance,
                        string_param_default(&constraint.params, "wrap_mode", "ON_SURFACE")?,
                    )?
                };
                desired.translation = snapped.to_array();
                if shrink_type != "NEAREST_VERTEX"
                    && param_bool(&constraint.params, "use_track_normal", false)?
                {
                    let local_axis = track_axis(&constraint.params)?;
                    let current_axis = source_quat * local_axis;
                    let rotation = DQuat::from_rotation_arc(current_axis, hit.normal) * source_quat;
                    desired.rotation = [rotation.x, rotation.y, rotation.z, rotation.w];
                }
            }
        }
        ConstraintType::LimitLocation => {
            limit_components(&constraint.params, &mut desired.translation, "location")?;
        }
        ConstraintType::LimitScale => {
            limit_components(&constraint.params, &mut desired.scale, "scale")?;
        }
        ConstraintType::LimitRotation => {
            let mut euler = DQuat::from_xyzw(
                desired.rotation[0],
                desired.rotation[1],
                desired.rotation[2],
                desired.rotation[3],
            )
            .to_euler(EulerRot::XYZ);
            let mut components = [euler.0, euler.1, euler.2];
            limit_components(&constraint.params, &mut components, "rotation")?;
            euler = (components[0], components[1], components[2]);
            let rotation = DQuat::from_euler(EulerRot::XYZ, euler.0, euler.1, euler.2);
            desired.rotation = [rotation.x, rotation.y, rotation.z, rotation.w];
        }
        ConstraintType::ChildOf => {
            let inverse = match constraint.params.get("inverse_matrix") {
                Some(value) => value_matrix(value)
                    .filter(|matrix| finite_matrix(*matrix))
                    .ok_or_else(|| {
                        evaluation_error_with(
                            "child-of inverse_matrix must be a finite 4x4 matrix",
                            json!({ "constraint_id": constraint.id }),
                        )
                    })?,
                None => DMat4::IDENTITY,
            };
            let child = target * inverse * source;
            if !finite_matrix(child) {
                return Err(evaluation_error_with(
                    "child-of constraint result is non-finite",
                    json!({ "constraint_id": constraint.id }),
                ));
            }
            let mut child_transform = matrix_transform(child)?;
            let owner_transform = matrix_transform(source)?;
            for (axis, (location_key, scale_key)) in [
                ("use_location_x", "use_scale_x"),
                ("use_location_y", "use_scale_y"),
                ("use_location_z", "use_scale_z"),
            ]
            .into_iter()
            .enumerate()
            {
                if !param_bool(&constraint.params, location_key, true)? {
                    child_transform.translation[axis] = owner_transform.translation[axis];
                }
                if !param_bool(&constraint.params, scale_key, true)? {
                    child_transform.scale[axis] = owner_transform.scale[axis];
                }
            }
            let mut child_angles = blender_euler_xyz(child_transform.rotation_quat());
            let owner_angles = blender_euler_xyz(owner_transform.rotation_quat());
            for (axis, rotation_key) in ["use_rotation_x", "use_rotation_y", "use_rotation_z"]
                .into_iter()
                .enumerate()
            {
                if !param_bool(&constraint.params, rotation_key, true)? {
                    child_angles[axis] = owner_angles[axis];
                }
            }
            let child_rotation = DQuat::from_rotation_z(child_angles[2])
                * DQuat::from_rotation_y(child_angles[1])
                * DQuat::from_rotation_x(child_angles[0]);
            child_transform.rotation = [
                child_rotation.x,
                child_rotation.y,
                child_rotation.z,
                child_rotation.w,
            ];
            return blend_transform(source, child_transform.matrix(), influence);
        }
        ConstraintType::Ik | ConstraintType::SplineIk => return Ok(source),
        ConstraintType::TransformCache => return Ok(target),
        ConstraintType::Action => {
            let desired = action_constraint_matrix(
                source,
                target,
                constraint,
                doc,
                frame,
                world_matrices,
                owner_node_id,
                world_to_owner,
            )?;
            return if influence >= 1.0 {
                Ok(desired)
            } else {
                blend_transform(source, desired, influence)
            };
        }
        ConstraintType::CameraSolver => {
            let reconstruction = camera_solver_matrix(doc, constraint, frame)?;
            let desired = source * reconstruction;
            return if influence >= 1.0 {
                Ok(desired)
            } else {
                blend_transform(source, desired, influence)
            };
        }
        ConstraintType::FollowTrack => {
            let desired = follow_track_matrix(
                source,
                target,
                constraint,
                doc,
                frame,
                world_matrices,
                world_to_owner,
            )?;
            return blend_transform(source, desired, influence);
        }
        ConstraintType::GeometryAttribute => {
            let desired =
                geometry_attribute_matrix(source, target, doc, constraint, evaluated_meshes)?;
            return if influence >= 1.0 {
                Ok(desired)
            } else {
                blend_transform(source, desired, influence)
            };
        }
        ConstraintType::ObjectSolver => {
            let desired = object_solver_matrix(doc, constraint, frame, target, source)?;
            return blend_transform(source, desired, influence);
        }
    }
    let desired_matrix = desired.matrix();
    blend_transform(source, desired_matrix, influence)
}
fn constraint_curve_spline<'a>(
    doc: &'a SceneDoc,
    target_id: &Id,
) -> Result<(&'a crate::model::CurveData, &'a crate::model::CurveSpline)> {
    let node = doc.nodes.get(target_id).ok_or_else(|| {
        evaluation_error_with(
            "constraint curve target does not exist",
            json!({ "target": target_id }),
        )
    })?;
    let data_id = node.data.as_ref().ok_or_else(|| {
        evaluation_error_with(
            "constraint curve target has no data",
            json!({ "target": target_id }),
        )
    })?;
    let curve = doc
        .data_blocks
        .get(data_id)
        .and_then(|data| data.curve.as_ref())
        .ok_or_else(|| {
            evaluation_error_with(
                "constraint target has no curve data",
                json!({ "target": target_id }),
            )
        })?;
    let spline = curve.splines.first().ok_or_else(|| {
        evaluation_error_with(
            "constraint curve target has no splines",
            json!({ "target": target_id }),
        )
    })?;
    Ok((curve, spline))
}

fn constraint_curve_path_samples<'a>(
    doc: &'a SceneDoc,
    target_id: &Id,
    constraint: &Constraint,
) -> Result<(
    &'a crate::model::CurveData,
    &'a crate::model::CurveSpline,
    Vec<crate::geom::curve::CurvePathPoint>,
)> {
    let (curve, spline) = constraint_curve_spline(doc, target_id)?;
    let with_orientation = constraint.constraint_type == ConstraintType::FollowPath
        && param_bool(&constraint.params, "use_curve_follow", false)?;
    let mut samples = crate::geom::curve::evaluate_curve_spline(
        spline,
        curve.twist_mode.as_str(),
        with_orientation,
    )
    .map_err(|error| evaluation_error(format!("constraint curve evaluation failed: {error}")))?;
    if spline.cyclic
        && let Some(first) = samples.first().copied()
    {
        samples.push(first);
    }
    if samples.len() < 2
        || samples.iter().any(|sample| {
            !sample.position.is_finite() || !sample.radius.is_finite() || !sample.tilt.is_finite()
        })
    {
        return Err(evaluation_error_with(
            "constraint curve must evaluate to two finite path samples",
            json!({ "target": target_id }),
        ));
    }
    Ok((curve, spline, samples))
}

fn curve_eval_time(curve: &crate::model::CurveData, frame: f64) -> Result<f64> {
    if let Some(fcurve) = curve
        .eval_time_fcurves
        .iter()
        .find(|fcurve| fcurve.path == "eval_time" && fcurve.index == 0)
        && let Some(value) = crate::eval::animation::sample_curve(fcurve, frame)?
    {
        return Ok(value);
    }
    Ok(curve.eval_time)
}

fn constraint_curve_bounds(
    curve: &crate::model::CurveData,
    target: DMat4,
) -> Result<(DVec3, DVec3)> {
    let mesh = crate::geom::curve::evaluate_curve(curve)
        .map_err(|error| evaluation_error(format!("constraint curve bounds failed: {error}")))?;
    let mut minimum = DVec3::splat(f64::INFINITY);
    let mut maximum = DVec3::splat(f64::NEG_INFINITY);
    for vertex in &mesh.vertices {
        let point = target.transform_point3(vertex.co);
        if !point.is_finite() {
            return Err(evaluation_error("constraint curve bounds are non-finite"));
        }
        minimum = minimum.min(point);
        maximum = maximum.max(point);
    }
    if !minimum.is_finite() || !maximum.is_finite() {
        return Err(evaluation_error("constraint curve has no evaluated bounds"));
    }
    Ok((minimum, maximum))
}

fn clamp_to_axis_index(minimum: DVec3, maximum: DVec3, main_axis: &str) -> Result<usize> {
    match main_axis.to_ascii_uppercase().as_str() {
        "AUTO" | "CLAMPTO_AUTO" => {
            let extent = maximum - minimum;
            let mut axis = 0;
            for candidate in 1..3 {
                if extent[candidate] > extent[axis] {
                    axis = candidate;
                }
            }
            Ok(axis)
        }
        "X" | "CLAMPTO_X" => Ok(0),
        "Y" | "CLAMPTO_Y" => Ok(1),
        "Z" | "CLAMPTO_Z" => Ok(2),
        _ => Err(evaluation_error("clamp-to main_axis is invalid")),
    }
}

fn clamp_to_factor(position: f64, minimum: f64, maximum: f64, cyclic: bool) -> f64 {
    let extent = maximum - minimum;
    if extent <= f64::EPSILON {
        return 0.0;
    }
    if cyclic && position < minimum {
        let offset = minimum - ((minimum - position) / extent).ceil() * extent;
        (position - offset) / extent
    } else if cyclic && position > maximum {
        let offset = maximum + ((position - maximum) / extent).trunc() * extent;
        (position - offset) / extent
    } else {
        ((position - minimum) / extent).clamp(0.0, 1.0)
    }
}

fn follow_path_forward_axis(params: &serde_json::Map<String, Value>) -> Result<DVec3> {
    match string_param_default(params, "forward_axis", "FORWARD_Y")? {
        "FORWARD_X" => Ok(DVec3::X),
        "FORWARD_Y" => Ok(DVec3::Y),
        "FORWARD_Z" => Ok(DVec3::Z),
        "TRACK_NEGATIVE_X" => Ok(DVec3::NEG_X),
        "TRACK_NEGATIVE_Y" => Ok(DVec3::NEG_Y),
        "TRACK_NEGATIVE_Z" => Ok(DVec3::NEG_Z),
        _ => Err(evaluation_error("follow-path forward_axis is invalid")),
    }
}

fn path_sample_factor(factor: f64, cyclic: bool) -> f64 {
    if cyclic && (factor < 0.0 || factor > 1.0) {
        factor - factor.floor()
    } else {
        factor.clamp(0.0, 1.0)
    }
}

fn curve_path_sample_segment(cumulative: &[f64], distance: f64) -> Result<usize> {
    if cumulative.len() < 2 {
        return Err(evaluation_error(
            "constraint path sampling input is invalid",
        ));
    }
    let distance = distance.clamp(0.0, cumulative[cumulative.len() - 1]);
    Ok(cumulative
        .windows(2)
        .position(|span| distance <= span[1])
        .unwrap_or(cumulative.len() - 2))
}

fn curve_path_sample_attributes(
    samples: &[crate::geom::curve::CurvePathPoint],
    cumulative: &[f64],
    distance: f64,
) -> Result<crate::geom::curve::CurvePathPoint> {
    if samples.len() != cumulative.len() {
        return Err(evaluation_error(
            "constraint path sample attributes are inconsistent",
        ));
    }
    let segment = curve_path_sample_segment(cumulative, distance)?;
    let distance = distance.clamp(0.0, cumulative[cumulative.len() - 1]);
    let start = cumulative[segment];
    let end = cumulative[segment + 1];
    let factor = if end > start {
        (distance - start) / (end - start)
    } else {
        0.0
    };
    let first = samples[segment];
    let second = samples[segment + 1];
    Ok(crate::geom::curve::CurvePathPoint {
        position: first.position.lerp(second.position, factor),
        radius: first.radius + (second.radius - first.radius) * factor,
        tilt: first.tilt + (second.tilt - first.tilt) * factor,
        orientation: first
            .orientation
            .slerp(second.orientation, factor)
            .normalize(),
    })
}

fn string_param<'a>(params: &'a serde_json::Map<String, Value>, key: &str) -> Result<&'a str> {
    params
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| evaluation_error(format!("constraint parameter {key} must be a string")))
}

fn string_param_default<'a>(
    params: &'a serde_json::Map<String, Value>,
    key: &str,
    default: &'a str,
) -> Result<&'a str> {
    match params.get(key) {
        Some(value) => value.as_str().ok_or_else(|| {
            evaluation_error(format!("constraint parameter {key} must be a string"))
        }),
        None => Ok(default),
    }
}

fn axis_index(axis: DVec3) -> Result<usize> {
    if axis.x.abs() > 0.5 {
        Ok(0)
    } else if axis.y.abs() > 0.5 {
        Ok(1)
    } else if axis.z.abs() > 0.5 {
        Ok(2)
    } else {
        Err(evaluation_error(
            "constraint axis must be a signed cardinal axis",
        ))
    }
}

fn number_array_param(
    params: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<Option<[f64; 3]>> {
    let Some(value) = params.get(key) else {
        return Ok(None);
    };
    let values = value
        .as_array()
        .filter(|values| values.len() == 3)
        .ok_or_else(|| evaluation_error(format!("constraint parameter {key} must be a vec3")))?;
    let mut result = [0.0; 3];
    for (index, value) in values.iter().enumerate() {
        result[index] = value
            .as_f64()
            .filter(|number| number.is_finite())
            .ok_or_else(|| {
                evaluation_error(format!(
                    "constraint parameter {key} contains an invalid value"
                ))
            })?;
    }
    Ok(Some(result))
}
fn constraint_channel(transform: &Transform, channel: &str) -> Result<f64> {
    let index = match channel {
        "location_x" | "rotation_x" | "scale_x" => 0,
        "location_y" | "rotation_y" | "scale_y" => 1,
        "location_z" | "rotation_z" | "scale_z" => 2,
        _ => return Err(evaluation_error("transformation channel is invalid")),
    };
    if channel.starts_with("location_") {
        Ok(transform.translation[index])
    } else if channel.starts_with("scale_") {
        Ok(transform.scale[index])
    } else {
        Ok(blender_euler_xyz(transform.rotation_quat())[index])
    }
}

fn blender_euler_xyz(rotation: DQuat) -> [f64; 3] {
    let matrix = DMat4::from_quat(rotation);
    let y = (-matrix.x_axis.z).clamp(-1.0, 1.0).asin();
    let (x, z) = if y.cos().abs() > f64::EPSILON * 64.0 {
        (
            matrix.y_axis.z.atan2(matrix.z_axis.z),
            matrix.x_axis.y.atan2(matrix.x_axis.x),
        )
    } else {
        (0.0, (-matrix.y_axis.x).atan2(matrix.y_axis.y))
    };
    [x, y, z]
}

fn transformation_euler_order(value: &str, owner_mode: &str) -> Result<EulerRot> {
    let mode = if value == "AUTO" { owner_mode } else { value };
    Ok(match mode {
        "XYZ" | "QUATERNION" | "quaternion" | "AXIS_ANGLE" | "AUTO" => EulerRot::XYZEx,
        "XZY" => EulerRot::XZYEx,
        "YXZ" => EulerRot::YXZEx,
        "YZX" => EulerRot::YZXEx,
        "ZXY" => EulerRot::ZXYEx,
        "ZYX" => EulerRot::ZYXEx,
        _ => return Err(evaluation_error("transformation Euler order is invalid")),
    })
}

type TransformationRangeKeys = (
    &'static [&'static str; 3],
    &'static [&'static str; 3],
    &'static [&'static str; 3],
    &'static [&'static str; 3],
);

fn transformation_range_keys(map_type: &str) -> Result<TransformationRangeKeys> {
    match map_type {
        "LOCATION" => Ok((
            &["from_min_x", "from_min_y", "from_min_z"],
            &["from_max_x", "from_max_y", "from_max_z"],
            &["to_min_x", "to_min_y", "to_min_z"],
            &["to_max_x", "to_max_y", "to_max_z"],
        )),
        "ROTATION" => Ok((
            &["from_min_x_rot", "from_min_y_rot", "from_min_z_rot"],
            &["from_max_x_rot", "from_max_y_rot", "from_max_z_rot"],
            &["to_min_x_rot", "to_min_y_rot", "to_min_z_rot"],
            &["to_max_x_rot", "to_max_y_rot", "to_max_z_rot"],
        )),
        "SCALE" => Ok((
            &["from_min_x_scale", "from_min_y_scale", "from_min_z_scale"],
            &["from_max_x_scale", "from_max_y_scale", "from_max_z_scale"],
            &["to_min_x_scale", "to_min_y_scale", "to_min_z_scale"],
            &["to_max_x_scale", "to_max_y_scale", "to_max_z_scale"],
        )),
        _ => Err(evaluation_error("transformation map type is invalid")),
    }
}

fn transformation_constraint(
    source: &Transform,
    target: &Transform,
    target_is_negative: bool,
    constraint: &Constraint,
    owner_rotation_mode: &str,
) -> Result<Transform> {
    let map_from = string_param_default(&constraint.params, "map_from", "LOCATION")?;
    let map_to = string_param_default(&constraint.params, "map_to", "LOCATION")?;
    let (from_min_keys, from_max_keys, _, _) = transformation_range_keys(map_from)?;
    let (_, _, output_min_keys, output_max_keys) = transformation_range_keys(map_to)?;
    let input_rotation_mode =
        string_param_default(&constraint.params, "from_rotation_mode", "AUTO")?;
    let output_rotation_mode = string_param_default(&constraint.params, "to_euler_order", "AUTO")?;
    let input_order = transformation_euler_order(input_rotation_mode, owner_rotation_mode)?;
    let output_order = transformation_euler_order(output_rotation_mode, owner_rotation_mode)?;
    let input_values = match map_from {
        "LOCATION" => target.translation,
        "ROTATION" => {
            let rotation = target.rotation_quat().to_euler(input_order);
            [rotation.0, rotation.1, rotation.2]
        }
        "SCALE" => {
            if target_is_negative {
                target.scale.map(|value| -value)
            } else {
                target.scale
            }
        }
        _ => return Err(evaluation_error("transformation map_from is invalid")),
    };
    let extrapolate = param_bool(&constraint.params, "use_motion_extrapolate", false)?;
    let mut mapped = [0.0; 3];
    let input_default = if map_from == "SCALE" { 1.0 } else { 0.0 };
    let output_default = if map_to == "SCALE" { 1.0 } else { 0.0 };
    for (output_axis, output_key) in ["map_to_x_from", "map_to_y_from", "map_to_z_from"]
        .iter()
        .enumerate()
    {
        let input_axis_name =
            string_param_default(&constraint.params, output_key, ["X", "Y", "Z"][output_axis])?;
        let input_axis = axis_index(
            axis_vector(input_axis_name)
                .ok_or_else(|| evaluation_error("transformation source axis is invalid"))?,
        )?;
        let mut input = input_values[input_axis];
        let input_min =
            number_param(&constraint.params, from_min_keys[input_axis])?.unwrap_or(input_default);
        let input_max =
            number_param(&constraint.params, from_max_keys[input_axis])?.unwrap_or(input_default);
        if input_max - input_min > f64::EPSILON && !extrapolate {
            input = input.clamp(input_min, input_max);
        } else if input_min - input_max > f64::EPSILON && !extrapolate {
            input = input.clamp(input_max, input_min);
        }
        let factor = if (input_max - input_min).abs() > f64::EPSILON {
            (input - input_min) / (input_max - input_min)
        } else {
            0.0
        };
        let output_min = number_param(&constraint.params, output_min_keys[output_axis])?
            .unwrap_or(output_default);
        let output_max = number_param(&constraint.params, output_max_keys[output_axis])?
            .unwrap_or(output_default);
        mapped[output_axis] = output_min + factor * (output_max - output_min);
    }
    let mut desired = source.clone();
    match map_to {
        "LOCATION" => match string_param_default(&constraint.params, "mix_mode", "ADD")? {
            "ADD" => {
                for (value, mapped) in desired.translation.iter_mut().zip(mapped) {
                    *value += mapped;
                }
            }
            "REPLACE" => desired.translation = mapped,
            _ => {
                return Err(evaluation_error(
                    "transformation location mix mode is invalid",
                ));
            }
        },
        "SCALE" => match string_param_default(&constraint.params, "mix_mode_scale", "REPLACE")? {
            "MULTIPLY" => {
                for (value, mapped) in desired.scale.iter_mut().zip(mapped) {
                    *value *= mapped;
                }
            }
            "REPLACE" => desired.scale = mapped,
            _ => return Err(evaluation_error("transformation scale mix mode is invalid")),
        },
        "ROTATION" => {
            let source_rotation = source.rotation_quat();
            let output_rotation = DQuat::from_euler(output_order, mapped[0], mapped[1], mapped[2]);
            let rotation = match string_param_default(&constraint.params, "mix_mode_rot", "ADD")? {
                "REPLACE" => output_rotation,
                "BEFORE" => output_rotation * source_rotation,
                "AFTER" => source_rotation * output_rotation,
                "ADD" => {
                    let (x, y, z) = source_rotation.to_euler(output_order);
                    DQuat::from_euler(output_order, x + mapped[0], y + mapped[1], z + mapped[2])
                }
                _ => {
                    return Err(evaluation_error(
                        "transformation rotation mix mode is invalid",
                    ));
                }
            };
            desired.rotation = [rotation.x, rotation.y, rotation.z, rotation.w];
        }
        _ => return Err(evaluation_error("transformation map_to is invalid")),
    }
    Ok(desired)
}

#[derive(Clone, Copy)]
struct ShrinkwrapHit {
    point: DVec3,
    normal: DVec3,
}

fn shrinkwrap_target_hit(
    doc: &SceneDoc,
    target_id: &Id,
    target_matrix: DMat4,
    evaluated_meshes: Option<&BTreeMap<Id, Mesh>>,
    point: DVec3,
    source_rotation: DQuat,
    params: &serde_json::Map<String, Value>,
    shrink_type: &str,
) -> Result<Option<ShrinkwrapHit>> {
    let node = doc.nodes.get(target_id).ok_or_else(|| {
        evaluation_error_with(
            "shrinkwrap target does not exist",
            json!({ "target": target_id }),
        )
    })?;
    let base_mesh = node
        .data
        .as_ref()
        .and_then(|data_id| doc.data_blocks.get(data_id))
        .and_then(|data| data.mesh.as_ref());
    let mesh = evaluated_meshes
        .and_then(|meshes| meshes.get(target_id))
        .or(base_mesh)
        .ok_or_else(|| {
            evaluation_error_with(
                "shrinkwrap target has no mesh",
                json!({ "target": target_id }),
            )
        })?;
    let vertices: HashMap<_, _> = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, target_matrix.transform_point3(vertex.co)))
        .filter(|(_, position)| position.is_finite())
        .collect();
    if shrink_type == "NEAREST_VERTEX" {
        return Ok(vertices
            .values()
            .min_by(|left, right| {
                left.distance_squared(point)
                    .total_cmp(&right.distance_squared(point))
            })
            .map(|position| ShrinkwrapHit {
                point: *position,
                normal: DVec3::ZERO,
            }));
    }
    if !matches!(
        shrink_type,
        "NEAREST_SURFACE" | "PROJECT" | "TARGET_PROJECT"
    ) {
        return Err(evaluation_error("shrinkwrap type is invalid"));
    }
    let triangles = mesh.triangulate().map_err(|error| {
        evaluation_error(format!("shrinkwrap target triangulation failed: {error}"))
    })?;
    if shrink_type == "PROJECT" {
        let axis = shrinkwrap_project_axis(params, source_rotation)?;
        let opposite = param_bool(params, "use_project_opposite", false)?;
        let limit = number_param(params, "project_limit")?.unwrap_or(0.0);
        if !limit.is_finite() || limit < 0.0 {
            return Err(evaluation_error(
                "shrinkwrap project_limit must be finite and non-negative",
            ));
        }
        let cull = string_param_default(params, "cull_face", "OFF")?;
        let mut best: Option<(f64, ShrinkwrapHit)> = None;
        for (direction, inverse) in [(axis, false), (-axis, true)] {
            if inverse && !opposite {
                continue;
            }
            for triangle in &triangles {
                let [Some(a), Some(b), Some(c)] = triangle.map(|id| vertices.get(&id).copied())
                else {
                    continue;
                };
                let normal = (b - a).cross(c - a).normalize_or_zero();
                if normal.length_squared() <= f64::EPSILON {
                    continue;
                }
                let dot = normal.dot(direction);
                let invert_cull = inverse && param_bool(params, "use_invert_cull", false)?;
                let cull_mode = if invert_cull {
                    match cull {
                        "FRONT" => "BACK",
                        "BACK" => "FRONT",
                        _ => cull,
                    }
                } else {
                    cull
                };
                if (cull_mode == "FRONT" && dot <= 0.0) || (cull_mode == "BACK" && dot >= 0.0) {
                    continue;
                }
                let Some(distance) = ray_triangle_distance(point, direction, a, b, c) else {
                    continue;
                };
                if limit > 0.0 && distance > limit {
                    continue;
                }
                let hit = ShrinkwrapHit {
                    point: point + direction * distance,
                    normal,
                };
                if best.is_none_or(|(nearest, _)| distance < nearest) {
                    best = Some((distance, hit));
                }
            }
        }
        return Ok(best.map(|(_, hit)| hit));
    }
    if shrink_type == "TARGET_PROJECT" {
        let target_inverse = checked_inverse(target_matrix, "Shrinkwrap target")?;
        let local_point = target_inverse.transform_point3(point);
        let local_vertices: HashMap<_, _> = mesh
            .vertices
            .iter()
            .map(|vertex| (vertex.id, vertex.co))
            .collect();
        let vertex_normals = shrinkwrap_vertex_normals(mesh, &local_vertices);
        let boundary_edges = shrinkwrap_boundary_edges(mesh);
        let boundary_vertices =
            shrinkwrap_boundary_vertices(mesh, &local_vertices, &vertex_normals, &boundary_edges);
        let normal_to_world = target_inverse.transpose();
        let mut best: Option<(f64, ShrinkwrapHit)> = None;
        for triangle in &triangles {
            let triangle_ids = *triangle;
            let [Some(a), Some(b), Some(c)] =
                triangle_ids.map(|id| local_vertices.get(&id).copied())
            else {
                continue;
            };
            let [Some(normal_a), Some(normal_b), Some(normal_c)] =
                triangle_ids.map(|id| vertex_normals.get(&id).copied())
            else {
                continue;
            };
            let closest = closest_point_on_triangle(local_point, a, b, c);
            let nearest_distance = closest.distance_squared(local_point);
            if best.is_some_and(|(distance, _)| nearest_distance >= distance) {
                continue;
            }
            let projected = target_project_triangle_hit(
                local_point,
                [a, b, c],
                [normal_a, normal_b, normal_c],
                closest,
            );
            let mut update = |hit_point: DVec3, hit_normal: DVec3| {
                let world_point = target_matrix.transform_point3(hit_point);
                let world_normal = normal_to_world
                    .transform_vector3(hit_normal)
                    .normalize_or_zero();
                let distance = local_point.distance_squared(hit_point);
                if world_point.is_finite()
                    && world_normal.length_squared() > f64::EPSILON
                    && best.is_none_or(|(nearest, _)| distance < nearest)
                {
                    best = Some((
                        distance,
                        ShrinkwrapHit {
                            point: world_point,
                            normal: world_normal,
                        },
                    ));
                }
            };
            if let Some((hit_point, hit_normal)) = projected {
                update(hit_point, hit_normal);
                continue;
            }
            for (first_id, second_id) in [
                (triangle_ids[0], triangle_ids[1]),
                (triangle_ids[1], triangle_ids[2]),
                (triangle_ids[2], triangle_ids[0]),
            ] {
                let key = shrinkwrap_edge_key(first_id, second_id);
                let Some(edge_vertices) = boundary_edges.get(&key).copied() else {
                    continue;
                };
                let [Some(edge_a), Some(edge_b)] =
                    edge_vertices.map(|id| local_vertices.get(&id).copied())
                else {
                    continue;
                };
                let [Some(boundary_a), Some(boundary_b)] =
                    edge_vertices.map(|id| boundary_vertices.get(&id).copied())
                else {
                    continue;
                };
                let [Some(normal_a), Some(normal_b)] =
                    edge_vertices.map(|id| vertex_normals.get(&id).copied())
                else {
                    continue;
                };
                if let Some((hit_point, hit_normal)) = target_project_edge_hit(
                    local_point,
                    [edge_a, edge_b],
                    [boundary_a, boundary_b],
                    [normal_a, normal_b],
                ) {
                    update(hit_point, hit_normal);
                }
            }
        }
        if let Some((_, hit)) = best {
            return Ok(Some(hit));
        }
        let mut nearest: Option<(f64, DVec3, DVec3)> = None;
        for triangle in &triangles {
            let [Some(a), Some(b), Some(c)] = triangle.map(|id| local_vertices.get(&id).copied())
            else {
                continue;
            };
            let surface = closest_point_on_triangle(local_point, a, b, c);
            let normal = (b - a).cross(c - a).normalize_or_zero();
            let distance = surface.distance_squared(local_point);
            if normal.length_squared() > f64::EPSILON
                && nearest.is_none_or(|(nearest_distance, _, _)| distance < nearest_distance)
            {
                nearest = Some((distance, surface, normal));
            }
        }
        return Ok(nearest.map(|(_, hit_point, hit_normal)| ShrinkwrapHit {
            point: target_matrix.transform_point3(hit_point),
            normal: normal_to_world
                .transform_vector3(hit_normal)
                .normalize_or_zero(),
        }));
    }
    let mut best: Option<(f64, ShrinkwrapHit)> = None;
    for triangle in &triangles {
        let [Some(a), Some(b), Some(c)] = triangle.map(|id| vertices.get(&id).copied()) else {
            continue;
        };
        let normal = (b - a).cross(c - a).normalize_or_zero();
        if normal.length_squared() <= f64::EPSILON {
            continue;
        }
        let closest = closest_point_on_triangle(point, a, b, c);
        let distance = closest.distance_squared(point);
        if best.is_none_or(|(nearest, _)| distance < nearest) {
            best = Some((
                distance,
                ShrinkwrapHit {
                    point: closest,
                    normal,
                },
            ));
        }
    }
    Ok(best.map(|(_, hit)| hit))
}

fn shrinkwrap_project_axis(
    params: &serde_json::Map<String, Value>,
    source_rotation: DQuat,
) -> Result<DVec3> {
    let axis = string_param_default(params, "project_axis", "POS_Z")?;
    let local_axis = match axis {
        "POS_X" => DVec3::X,
        "NEG_X" => DVec3::NEG_X,
        "POS_Y" => DVec3::Y,
        "NEG_Y" => DVec3::NEG_Y,
        "POS_Z" => DVec3::Z,
        "NEG_Z" => DVec3::NEG_Z,
        _ => return Err(evaluation_error("shrinkwrap project_axis is invalid")),
    };
    let axis_space = string_param_default(params, "project_axis_space", "LOCAL")?;
    let direction = if axis_space == "WORLD" {
        local_axis
    } else {
        source_rotation * local_axis
    }
    .normalize_or_zero();
    if direction.length_squared() <= f64::EPSILON {
        return Err(evaluation_error("shrinkwrap project axis is degenerate"));
    }
    Ok(direction)
}

fn shrinkwrap_snap(
    point: DVec3,
    hit: ShrinkwrapHit,
    distance: f64,
    wrap_mode: &str,
) -> Result<DVec3> {
    let delta = point - hit.point;
    let separation = delta.length();
    let side = if delta.dot(hit.normal) < 0.0 {
        -1.0
    } else {
        1.0
    };
    let normal = if hit.normal.length_squared() > f64::EPSILON {
        hit.normal.normalize()
    } else {
        delta.normalize_or_zero()
    };
    let direction = if separation <= f64::EPSILON {
        normal
    } else {
        delta / separation
    };
    match wrap_mode {
        "ON_SURFACE" => Ok(if distance == 0.0 {
            hit.point
        } else {
            hit.point + direction * distance
        }),
        "INSIDE" => Ok(if separation * -side < distance {
            hit.point + direction * -side * distance
        } else {
            point
        }),
        "OUTSIDE" => Ok(if separation * side < distance {
            hit.point + direction * side * distance
        } else {
            point
        }),
        "OUTSIDE_SURFACE" => Ok(hit.point + direction * side * distance),
        "ABOVE_SURFACE" => Ok(hit.point + normal * distance),
        _ => Err(evaluation_error("shrinkwrap wrap_mode is invalid")),
    }
}

fn shrinkwrap_vertex_normals(mesh: &Mesh, vertices: &HashMap<u32, DVec3>) -> HashMap<u32, DVec3> {
    let mut normals: HashMap<_, _> = vertices.keys().map(|id| (*id, DVec3::ZERO)).collect();
    for face in &mesh.faces {
        let mut normal = DVec3::ZERO;
        for index in 0..face.vertices.len() {
            let Some(first) = vertices.get(&face.vertices[index]).copied() else {
                continue;
            };
            let Some(second) = vertices
                .get(&face.vertices[(index + 1) % face.vertices.len()])
                .copied()
            else {
                continue;
            };
            normal += DVec3::new(
                (first.y - second.y) * (first.z + second.z),
                (first.z - second.z) * (first.x + second.x),
                (first.x - second.x) * (first.y + second.y),
            );
        }
        if normal.length_squared() > f64::EPSILON {
            for vertex_id in &face.vertices {
                if let Some(vertex_normal) = normals.get_mut(vertex_id) {
                    *vertex_normal += normal;
                }
            }
        }
    }
    for normal in normals.values_mut() {
        *normal = normal.normalize_or_zero();
    }
    normals
}

#[derive(Clone, Copy)]
struct ShrinkwrapBoundaryVertex {
    direction: DVec3,
    normal_plane: DVec3,
}

fn shrinkwrap_edge_key(first: u32, second: u32) -> (u32, u32) {
    if first < second {
        (first, second)
    } else {
        (second, first)
    }
}

fn shrinkwrap_boundary_edges(mesh: &Mesh) -> HashMap<(u32, u32), [u32; 2]> {
    let mut face_edge_counts = HashMap::<(u32, u32), usize>::with_capacity(mesh.edges.len());
    for face in &mesh.faces {
        for index in 0..face.vertices.len() {
            let first = face.vertices[index];
            let second = face.vertices[(index + 1) % face.vertices.len()];
            *face_edge_counts
                .entry(shrinkwrap_edge_key(first, second))
                .or_default() += 1;
        }
    }
    mesh.edges
        .iter()
        .filter_map(|edge| {
            let key = shrinkwrap_edge_key(edge.vertices[0], edge.vertices[1]);
            (face_edge_counts.get(&key) == Some(&1)).then_some((key, edge.vertices))
        })
        .collect()
}

fn shrinkwrap_boundary_vertices(
    mesh: &Mesh,
    vertices: &HashMap<u32, DVec3>,
    normals: &HashMap<u32, DVec3>,
    boundary_edges: &HashMap<(u32, u32), [u32; 2]>,
) -> HashMap<u32, ShrinkwrapBoundaryVertex> {
    let mut accumulated: HashMap<u32, (DVec3, i8)> = HashMap::new();
    for edge in &mesh.edges {
        if !boundary_edges.contains_key(&shrinkwrap_edge_key(edge.vertices[0], edge.vertices[1])) {
            continue;
        }
        let [first, second] = edge.vertices;
        let (Some(first_position), Some(second_position)) =
            (vertices.get(&first), vertices.get(&second))
        else {
            continue;
        };
        let direction = (*second_position - *first_position).normalize_or_zero();
        for (vertex_id, side) in [(first, 1_i8), (second, 2_i8)] {
            let Some((average, status)) = accumulated.get_mut(&vertex_id) else {
                accumulated.insert(vertex_id, (direction, side));
                continue;
            };
            if (*status >= 0 && *status == side) || (*status < 0 && average.dot(direction) < 0.0) {
                *average -= direction;
            } else {
                *average += direction;
            }
            *status = if *status == 0 { side } else { -1 };
        }
    }
    accumulated
        .into_iter()
        .filter_map(|(vertex_id, (direction, _))| {
            let normal = normals.get(&vertex_id).copied()?;
            let direction = direction.normalize_or_zero();
            let normal_plane = normal.cross(direction).cross(normal).normalize_or_zero();
            Some((
                vertex_id,
                ShrinkwrapBoundaryVertex {
                    direction,
                    normal_plane,
                },
            ))
        })
        .collect()
}

fn target_project_edge_hit(
    point: DVec3,
    edge: [DVec3; 2],
    boundary: [ShrinkwrapBoundaryVertex; 2],
    vertex_normals: [DVec3; 2],
) -> Option<(DVec3, DVec3)> {
    let [a, b] = edge;
    let [mut direction_a, mut direction_b] = [boundary[0].normal_plane, boundary[1].normal_plane];
    let edge_direction = b - a;
    if boundary[0].direction.dot(edge_direction) < 0.0 {
        direction_a = -direction_a;
    }
    if boundary[1].direction.dot(edge_direction) < 0.0 {
        direction_b = -direction_b;
    }
    let d0v0 = direction_a.dot(a);
    let d0v1 = direction_a.dot(b);
    let d1v0 = direction_b.dot(a);
    let d1v1 = direction_b.dot(b);
    let d0co = direction_a.dot(point);
    let a_value = d0v1 - d0v0 + d1v0 - d1v1;
    if a_value == 0.0 {
        return None;
    }
    let b_value = 2.0 * d0v0 - d0v1 - d0co - d1v0 + direction_b.dot(point);
    let c_value = d0co - d0v0;
    let determinant = b_value * b_value - 4.0 * a_value * c_value;
    if determinant < 0.0 {
        return None;
    }
    let square_root = determinant.sqrt();
    let mut nearest: Option<(f64, DVec3, DVec3)> = None;
    for sign in if determinant > 0.0 {
        [1.0, -1.0]
    } else {
        [0.0, 0.0]
    } {
        if determinant == 0.0 && sign != 0.0 {
            continue;
        }
        let mut weight = (-b_value + sign * square_root) / (2.0 * a_value);
        if !(-1.0e-6..=1.0 + 1.0e-6).contains(&weight) {
            continue;
        }
        weight = weight.clamp(0.0, 1.0);
        let hit = a.lerp(b, weight);
        let normal = vertex_normals[0].lerp(vertex_normals[1], weight);
        let distance = hit.distance_squared(point);
        if nearest.is_none_or(|(current, _, _)| distance < current) {
            nearest = Some((distance, hit, normal));
        }
        if determinant == 0.0 {
            break;
        }
    }
    nearest.map(|(_, hit, normal)| (hit, normal))
}

fn target_project_triangle_hit(
    point: DVec3,
    vertices: [DVec3; 3],
    normals: [DVec3; 3],
    closest: DVec3,
) -> Option<(DVec3, DVec3)> {
    let [a, b, c] = vertices;
    let [normal_a, normal_b, normal_c] = normals;
    let ab = b - a;
    let ac = c - a;
    let d00 = ab.dot(ab);
    let d01 = ab.dot(ac);
    let d11 = ac.dot(ac);
    let offset = closest - a;
    let d20 = offset.dot(ab);
    let d21 = offset.dot(ac);
    let denominator = d00 * d11 - d01 * d01;
    if denominator.abs() <= f64::EPSILON {
        return None;
    }
    let mut weight_b = (d11 * d20 - d01 * d21) / denominator;
    let mut weight_c = (d00 * d21 - d01 * d20) / denominator;
    let initial_weights = [1.0 - weight_b - weight_c, weight_b, weight_c];
    let initial_normal = normal_a * initial_weights[0]
        + normal_b * initial_weights[1]
        + normal_c * initial_weights[2];
    let normal_length_squared = initial_normal.length_squared();
    if normal_length_squared <= f64::EPSILON {
        return None;
    }
    let mut normal_offset = (point - closest).dot(initial_normal) / normal_length_squared;
    let scale = point
        .abs()
        .max_element()
        .max(a.abs().max_element())
        .max(1.0);
    let tolerance_squared = (scale * 1.0e-6).powi(2);
    for _ in 0..20 {
        let mut weights = [1.0 - weight_b - weight_c, weight_b, weight_c];
        for weight in &mut weights {
            *weight = weight.clamp(0.0, 1.0);
        }
        let weight_sum = weights.iter().sum::<f64>();
        if weight_sum <= f64::EPSILON {
            return None;
        }
        for weight in &mut weights {
            *weight /= weight_sum;
        }
        weight_b = weights[1];
        weight_c = weights[2];
        let surface = a * weights[0] + b * weights[1] + c * weights[2];
        let interpolated_normal =
            normal_a * weights[0] + normal_b * weights[1] + normal_c * weights[2];
        let residual = point - (surface + interpolated_normal * normal_offset);
        if residual.length_squared() <= tolerance_squared {
            break;
        }
        let derivative_b = ab + (normal_b - normal_a) * normal_offset;
        let derivative_c = ac + (normal_c - normal_a) * normal_offset;
        let determinant = derivative_b.dot(derivative_c.cross(interpolated_normal));
        if determinant.abs() <= f64::EPSILON || !determinant.is_finite() {
            return None;
        }
        weight_b += residual.dot(derivative_c.cross(interpolated_normal)) / determinant;
        weight_c += derivative_b.dot(residual.cross(interpolated_normal)) / determinant;
        normal_offset += derivative_b.dot(derivative_c.cross(residual)) / determinant;
    }
    let mut weights = [1.0 - weight_b - weight_c, weight_b, weight_c];
    for weight in &mut weights {
        *weight = weight.clamp(0.0, 1.0);
    }
    let weight_sum = weights.iter().sum::<f64>();
    if weight_sum <= f64::EPSILON {
        return None;
    }
    for weight in &mut weights {
        *weight /= weight_sum;
    }
    let surface = a * weights[0] + b * weights[1] + c * weights[2];
    let interpolated_normal = normal_a * weights[0] + normal_b * weights[1] + normal_c * weights[2];
    let normal = interpolated_normal.normalize_or_zero();
    let residual = point - (surface + interpolated_normal * normal_offset);
    (normal.length_squared() > f64::EPSILON
        && residual.length_squared() <= tolerance_squared * 16.0)
        .then_some((surface, normal))
}

fn closest_point_on_triangle(point: DVec3, a: DVec3, b: DVec3, c: DVec3) -> DVec3 {
    let ab = b - a;
    let ac = c - a;
    let ap = point - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = point - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return a + ab * (d1 / (d1 - d3));
    }
    let cp = point - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return a + ac * (d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let denominator = (va + vb + vc).recip();
    a + ab * (vb * denominator) + ac * (vc * denominator)
}

fn track_axis_with_default(
    params: &serde_json::Map<String, Value>,
    default: &str,
) -> Result<DVec3> {
    let axis = string_param_default(params, "track_axis", default)?;
    axis_vector(axis).ok_or_else(|| evaluation_error("constraint track axis is invalid"))
}

fn track_axis_enum(axis: DVec3) -> Result<usize> {
    let index = axis_index(axis)?;
    Ok(index + usize::from(axis[index] < 0.0) * 3)
}

fn blender_track_rotation(track: DVec3, up: DVec3, direction: DVec3, up_hint: DVec3) -> DQuat {
    let track_index = axis_index(track).unwrap_or(2);
    let up_index = axis_index(up).unwrap_or(1);
    if track_index == up_index {
        return DQuat::IDENTITY;
    }
    let positive_track = if track[track_index] < 0.0 {
        -direction
    } else {
        direction
    };
    let up_projection = up_hint - positive_track * up_hint.dot(positive_track);
    let up_projection = if up_projection.length_squared() > f64::EPSILON {
        up_projection.normalize()
    } else {
        DVec3::Y
    };
    let right_index = 3 - track_index - up_index;
    let right = up_projection.cross(positive_track) * basis_cross(track_index, up_index);
    let mut basis = [DVec3::ZERO; 3];
    basis[track_index] = positive_track;
    basis[up_index] = up_projection;
    basis[right_index] = right;
    DQuat::from_mat3(&glam::DMat3::from_cols(basis[0], basis[1], basis[2]))
}

fn basis_cross(track: usize, up: usize) -> f64 {
    match (track + 3 - up) % 3 {
        1 => 1.0,
        2 => -1.0,
        _ => 0.0,
    }
}

fn blender_damped_track_rotation(
    source: DQuat,
    local_track: DVec3,
    track_enum: usize,
    direction: DVec3,
) -> DQuat {
    let current_track = (source * local_track).normalize_or_zero();
    if current_track.length_squared() <= f64::EPSILON {
        return source;
    }
    let cross = current_track.cross(direction);
    let cross_length = cross.length();
    let dot = current_track.dot(direction).clamp(-1.0, 1.0);
    if cross_length <= f64::EPSILON {
        if dot >= 0.0 {
            return source;
        }
        let fallback_enum = (track_enum + 1) % 6;
        let fallback = match fallback_enum {
            0 => DVec3::X,
            1 => DVec3::Y,
            2 => DVec3::Z,
            3 => DVec3::NEG_X,
            4 => DVec3::NEG_Y,
            _ => DVec3::NEG_Z,
        };
        let fallback = source * fallback;
        let axis = current_track.cross(fallback).normalize_or_zero();
        if axis.length_squared() <= f64::EPSILON {
            return source;
        }
        return DQuat::from_axis_angle(axis, std::f64::consts::PI) * source;
    }
    DQuat::from_axis_angle(cross / cross_length, cross_length.atan2(dot)) * source
}

fn blender_locked_track_rotation(
    source: DQuat,
    track: DVec3,
    lock: DVec3,
    direction: DVec3,
) -> Option<DQuat> {
    let track_index = axis_index(track).ok()?;
    let lock_index = axis_index(lock).ok()?;
    if track_index == lock_index {
        return None;
    }
    let fixed = (source * lock).normalize_or_zero();
    let projected = direction - fixed * direction.dot(fixed);
    if projected.length_squared() <= f64::EPSILON {
        return None;
    }
    let track_direction = projected.normalize() * track[track_index].signum();
    let missing_index = 3 - track_index - lock_index;
    let mut basis = [DVec3::ZERO; 3];
    basis[lock_index] = fixed;
    basis[track_index] = track_direction;
    basis[missing_index] = if (lock_index + 1) % 3 == track_index {
        fixed.cross(track_direction)
    } else {
        track_direction.cross(fixed)
    }
    .normalize_or_zero();
    if basis
        .iter()
        .any(|axis| axis.length_squared() <= f64::EPSILON)
    {
        return None;
    }
    Some(DQuat::from_mat3(&glam::DMat3::from_cols(
        basis[0], basis[1], basis[2],
    )))
}

fn blender_stretch_to_rotation(source: DQuat, direction: DVec3, keep_axis: &str) -> Result<DQuat> {
    match keep_axis {
        "SWING_Y" => {
            let track = DVec3::Y;
            Ok(blender_damped_track_rotation(source, track, 1, direction))
        }
        "PLANE_X" => {
            let old_x = source * DVec3::X;
            let z = old_x.cross(direction).normalize_or_zero();
            if z.length_squared() <= f64::EPSILON {
                return Ok(source);
            }
            let x = direction.cross(z).normalize_or_zero();
            Ok(DQuat::from_mat3(&glam::DMat3::from_cols(x, direction, z)))
        }
        "PLANE_Z" => {
            let old_z = source * DVec3::Z;
            let orthogonal = old_z.cross(direction).normalize_or_zero();
            if orthogonal.length_squared() <= f64::EPSILON {
                return Ok(source);
            }
            let x = -orthogonal;
            let z = direction.cross(orthogonal).normalize_or_zero();
            Ok(DQuat::from_mat3(&glam::DMat3::from_cols(x, direction, z)))
        }
        _ => Err(evaluation_error("stretch-to keep_axis is invalid")),
    }
}

fn stretch_bulge_limit(bulge: f64, limit: f64, smooth: f64, upper: bool) -> f64 {
    if upper {
        let maximum = limit.max(1.0);
        let hard = bulge.min(maximum);
        let range = maximum - 1.0;
        let soft = 1.0
            + range * ((bulge - 1.0) / range.max(f64::EPSILON)).atan()
                / std::f64::consts::FRAC_PI_2;
        soft + (hard - soft) * smooth
    } else {
        let minimum = limit.clamp(0.0, 1.0);
        let hard = bulge.max(minimum);
        let range = 1.0 - minimum;
        let soft = 1.0
            - range * ((1.0 - bulge) / range.max(f64::EPSILON)).atan()
                / std::f64::consts::FRAC_PI_2;
        soft + (hard - soft) * smooth
    }
}

fn track_axis(params: &serde_json::Map<String, Value>) -> Result<DVec3> {
    track_axis_with_default(params, "TRACK_NEGATIVE_Z")
}

fn up_axis(params: &serde_json::Map<String, Value>) -> Result<DVec3> {
    match params.get("up_axis") {
        None => Ok(DVec3::Y),
        Some(Value::String(axis)) => {
            axis_vector(axis).ok_or_else(|| evaluation_error("constraint up axis is invalid"))
        }
        Some(_) => Err(evaluation_error(
            "constraint up_axis parameter must be a string",
        )),
    }
}

fn axis_vector(axis: &str) -> Option<DVec3> {
    match axis {
        "X" | "x" | "TRACK_X" | "track_x" | "UP_X" | "up_x" | "LOCK_X" | "SAMEVOL_X"
        | "FLOOR_X" => Some(DVec3::X),
        "-X" | "-x" | "TRACK_NEGATIVE_X" | "track_negative_x" => Some(DVec3::NEG_X),
        "Y" | "y" | "TRACK_Y" | "track_y" | "UP_Y" | "up_y" | "LOCK_Y" | "SAMEVOL_Y"
        | "FLOOR_Y" => Some(DVec3::Y),
        "-Y" | "-y" | "TRACK_NEGATIVE_Y" | "track_negative_y" => Some(DVec3::NEG_Y),
        "Z" | "z" | "TRACK_Z" | "track_z" | "UP_Z" | "up_z" | "LOCK_Z" | "SAMEVOL_Z"
        | "FLOOR_Z" => Some(DVec3::Z),
        "-Z" | "-z" | "TRACK_NEGATIVE_Z" | "track_negative_z" => Some(DVec3::NEG_Z),
        _ => None,
    }
}

fn limit_components(
    params: &serde_json::Map<String, Value>,
    values: &mut [f64; 3],
    prefix: &str,
) -> Result<()> {
    for (axis, suffix) in ["x", "y", "z"].iter().enumerate() {
        let min_key = format!("min_{suffix}");
        let max_key = format!("max_{suffix}");
        let use_min_key = format!("use_min_{suffix}");
        let use_max_key = format!("use_max_{suffix}");
        let min = number_param(params, &min_key)?.or(number_vector_param(params, "min", axis)?);
        let max = number_param(params, &max_key)?.or(number_vector_param(params, "max", axis)?);
        let (use_min, use_max) = if prefix == "rotation" {
            let use_limit_key = format!("use_limit_{suffix}");
            let use_limit = param_bool(params, &use_limit_key, false)?;
            (use_limit, use_limit)
        } else {
            (
                param_bool(params, &use_min_key, false)?,
                param_bool(params, &use_max_key, false)?,
            )
        };
        if use_min {
            values[axis] = values[axis].max(min.ok_or_else(|| {
                evaluation_error(format!("{prefix} constraint is missing {min_key}"))
            })?);
        }
        if use_max {
            values[axis] = values[axis].min(max.ok_or_else(|| {
                evaluation_error(format!("{prefix} constraint is missing {max_key}"))
            })?);
        }
    }
    Ok(())
}

fn number_param(params: &serde_json::Map<String, Value>, key: &str) -> Result<Option<f64>> {
    let Some(value) = params.get(key) else {
        return Ok(None);
    };
    let number = value
        .as_f64()
        .ok_or_else(|| evaluation_error(format!("constraint parameter {key} must be numeric")))?;
    if !number.is_finite() {
        return Err(evaluation_error(format!(
            "constraint parameter {key} is non-finite"
        )));
    }
    Ok(Some(number))
}

fn number_vector_param(
    params: &serde_json::Map<String, Value>,
    key: &str,
    index: usize,
) -> Result<Option<f64>> {
    let Some(value) = params.get(key) else {
        return Ok(None);
    };
    let array = value
        .as_array()
        .ok_or_else(|| evaluation_error(format!("constraint parameter {key} must be a vector")))?;
    let number = array
        .get(index)
        .and_then(Value::as_f64)
        .ok_or_else(|| evaluation_error(format!("constraint parameter {key} is not a vec3")))?;
    if !number.is_finite() {
        return Err(evaluation_error(format!(
            "constraint parameter {key} is non-finite"
        )));
    }
    Ok(Some(number))
}

fn param_bool(params: &serde_json::Map<String, Value>, key: &str, default: bool) -> Result<bool> {
    params.get(key).map_or(Ok(default), |value| {
        value
            .as_bool()
            .ok_or_else(|| evaluation_error(format!("constraint parameter {key} must be boolean")))
    })
}

fn param_usize(params: &serde_json::Map<String, Value>, key: &str) -> Result<Option<usize>> {
    let Some(value) = params.get(key) else {
        return Ok(None);
    };
    let number = value.as_u64().ok_or_else(|| {
        evaluation_error(format!("IK parameter {key} must be a nonnegative integer"))
    })?;
    usize::try_from(number)
        .map(Some)
        .map_err(|_| evaluation_error(format!("IK parameter {key} is out of range")))
}

fn value_matrix(value: &Value) -> Option<DMat4> {
    let array = value.as_array()?;
    let mut values = [0.0; 16];
    if array.len() == 16 {
        for (index, item) in array.iter().enumerate() {
            values[index] = item.as_f64()?;
        }
    } else if array.len() == 4 {
        for (row, row_values) in array.iter().enumerate() {
            let row_values = row_values.as_array()?;
            if row_values.len() != 4 {
                return None;
            }
            for (column, item) in row_values.iter().enumerate() {
                values[column * 4 + row] = item.as_f64()?;
            }
        }
    } else {
        return None;
    }
    Some(DMat4::from_cols_array(&values))
}

fn quat_from_array(value: [f64; 4], name: &str) -> Result<DQuat> {
    let rotation = DQuat::from_xyzw(value[0], value[1], value[2], value[3]);
    if !rotation.is_finite() || rotation.length_squared() <= f64::EPSILON {
        return Err(evaluation_error(format!("{name} is invalid")));
    }
    Ok(rotation.normalize())
}

/// Evaluate all drivers in dependency order and return values keyed by `(node, path, index)`.
///
/// Scripted expressions use a small arithmetic parser; no Rust, shell, or host-language code is
/// executed. Variables support local transform channels, node properties, and pose channels.
pub(crate) fn evaluate_drivers(
    doc: &SceneDoc,
    frame: f64,
) -> Result<BTreeMap<(Id, String, u32), f64>> {
    struct Entry<'a> {
        owner: &'a Id,
        driver: &'a Driver,
    }
    fn visit(
        index: usize,
        entries: &[Entry<'_>],
        by_output: &BTreeMap<(Id, String, u32), usize>,
        doc: &SceneDoc,
        frame: f64,
        outputs: &mut BTreeMap<(Id, String, u32), f64>,
        active: &mut BTreeSet<usize>,
    ) -> Result<f64> {
        let entry = &entries[index];
        let output_key = (
            entry.owner.to_owned(),
            entry.driver.path.clone(),
            entry.driver.index,
        );
        if let Some(value) = outputs.get(&output_key) {
            return Ok(*value);
        }
        if !active.insert(index) {
            return Err(evaluation_error_with(
                "driver dependency cycle",
                json!({ "node": entry.owner, "driver_id": entry.driver.id }),
            ));
        }
        let mut variables = BTreeMap::new();
        for variable in &entry.driver.variables {
            if variables.contains_key(&variable.name) {
                return Err(evaluation_error_with(
                    "driver variable names must be unique",
                    json!({ "driver_id": entry.driver.id, "name": variable.name }),
                ));
            }
            let source_key = (
                variable.target.clone(),
                variable.path.clone(),
                variable.index,
            );
            let value = if let Some(dependency) = by_output.get(&source_key) {
                visit(*dependency, entries, by_output, doc, frame, outputs, active)?
            } else {
                variable_value(doc, frame, variable)?
            };
            variables.insert(variable.name.clone(), value);
        }
        let mut sum = 0.0;
        let mut count = 0_usize;
        let mut minimum = None;
        let mut maximum = None;
        for value in variables.values().copied() {
            sum += value;
            count += 1;
            minimum = Some(minimum.map_or(value, |previous: f64| previous.min(value)));
            maximum = Some(maximum.map_or(value, |previous: f64| previous.max(value)));
        }
        let value = match entry.driver.driver_type {
            DriverType::Average => {
                if count == 0 {
                    0.0
                } else {
                    sum / count as f64
                }
            }
            DriverType::Sum => sum,
            DriverType::Min => minimum.unwrap_or(0.0),
            DriverType::Max => maximum.unwrap_or(0.0),
            DriverType::ScriptedExpression => {
                let expression = entry.driver.expression.as_deref().ok_or_else(|| {
                    evaluation_error_with(
                        "scripted driver has no expression",
                        json!({"node": entry.owner, "driver_id": entry.driver.id}),
                    )
                })?;
                evaluate_expression(expression, &variables)?
            }
        };
        if !value.is_finite() {
            return Err(evaluation_error_with(
                "driver result is non-finite",
                json!({ "node": entry.owner, "driver_id": entry.driver.id }),
            ));
        }
        active.remove(&index);
        outputs.insert(output_key, value);
        Ok(value)
    }
    if !frame.is_finite() {
        return Err(evaluation_error("driver evaluation frame is non-finite"));
    }
    let mut entries = Vec::new();
    for (owner, node) in &doc.nodes {
        for driver in &node.drivers {
            entries.push(Entry { owner, driver });
        }
    }
    entries.sort_by(|first, second| {
        (&first.owner, &first.driver.id).cmp(&(&second.owner, &second.driver.id))
    });
    let mut outputs = BTreeMap::new();
    let mut by_output = BTreeMap::<(Id, String, u32), usize>::new();
    for (index, entry) in entries.iter().enumerate() {
        let key = (
            entry.owner.to_owned(),
            entry.driver.path.clone(),
            entry.driver.index,
        );
        if by_output.insert(key.clone(), index).is_some() {
            return Err(evaluation_error_with(
                "multiple drivers target the same property channel",
                json!({ "node": key.0, "path": key.1, "index": key.2 }),
            ));
        }
    }
    let mut active = BTreeSet::new();
    for index in 0..entries.len() {
        visit(
            index,
            &entries,
            &by_output,
            doc,
            frame,
            &mut outputs,
            &mut active,
        )?;
    }
    Ok(outputs)
}

fn variable_value(doc: &SceneDoc, frame: f64, variable: &DriverVariable) -> Result<f64> {
    let node = doc.nodes.get(&variable.target).ok_or_else(|| {
        evaluation_error_with(
            "driver variable target does not exist",
            json!({ "target": variable.target, "path": variable.path }),
        )
    })?;
    if matches!(
        variable.variable_type,
        DriverVariableType::LocationDifference | DriverVariableType::RotationDifference
    ) {
        let target_2 = variable.target_2.as_ref().ok_or_else(|| {
            evaluation_error("difference driver variable is missing its second target")
        })?;
        let space = variable.transform_space.as_deref().unwrap_or("world");
        let first = driver_target_transform(doc, frame, &variable.target, space)?;
        let second = driver_target_transform(doc, frame, target_2, space)?;
        return match variable.variable_type {
            DriverVariableType::LocationDifference => Ok(DVec3::from_array(first.translation)
                .distance(DVec3::from_array(second.translation))),
            DriverVariableType::RotationDifference => {
                let first_rotation = first.rotation_quat();
                let second_rotation = second.rotation_quat();
                Ok(2.0
                    * first_rotation
                        .dot(second_rotation)
                        .abs()
                        .clamp(-1.0, 1.0)
                        .acos())
            }
            _ => Err(evaluation_error("invalid difference driver variable")),
        };
    }
    if variable.variable_type == DriverVariableType::Transforms {
        let space = variable.transform_space.as_deref().unwrap_or("local");
        let transform = driver_target_transform(doc, frame, &variable.target, space)?;
        if let Some(value) = transform_channel(&transform, &variable.path, variable.index) {
            return Ok(value);
        }
        return animated_pose_channel(node, doc, frame, &variable.path, variable.index)?
            .ok_or_else(|| {
                evaluation_error_with(
                    "driver transform variable path is unsupported",
                    json!({ "target": variable.target, "path": variable.path, "index": variable.index }),
                )
            });
    }
    if let Some(value) = transform_channel(&node.transform, &variable.path, variable.index) {
        return Ok(value);
    }
    if let Some(value) = animated_pose_channel(node, doc, frame, &variable.path, variable.index)? {
        return Ok(value);
    }
    let value = property_value(&node.properties, &variable.path, variable.index).or_else(|| {
        if variable.index != 0 {
            return None;
        }
        let data_id = node.data.as_ref()?;
        let data = doc.data_blocks.get(data_id)?;
        let shape_keys = data.shape_keys.as_ref()?;
        shape_key_value_from_path(shape_keys, &variable.path)
    });
    value.filter(|value| value.is_finite()).ok_or_else(|| {
        evaluation_error_with(
            "driver single-property variable path is unsupported",
            json!({ "target": variable.target, "path": variable.path, "index": variable.index }),
        )
    })
}

fn driver_target_transform(
    doc: &SceneDoc,
    frame: f64,
    target_id: &Id,
    transform_space: &str,
) -> Result<Transform> {
    match transform_space {
        "local" => {
            let node = doc.nodes.get(target_id).ok_or_else(|| {
                evaluation_error_with(
                    "driver variable target does not exist",
                    json!({ "target": target_id }),
                )
            })?;
            animated_transform(node, doc, frame)
        }
        "world" => {
            let worlds = evaluate_world_matrices_with_drivers(doc, frame, &BTreeMap::new())?;
            let matrix = worlds.get(target_id).copied().ok_or_else(|| {
                evaluation_error_with(
                    "driver world-space target is unavailable",
                    json!({ "target": target_id }),
                )
            })?;
            matrix_transform(matrix)
        }
        _ => Err(evaluation_error_with(
            "driver transform_space must be local or world",
            json!({ "target": target_id, "transform_space": transform_space }),
        )),
    }
}

fn transform_channel(transform: &Transform, path: &str, index: u32) -> Option<f64> {
    let values = match path {
        "location" | "translation" | "transform.location" | "transform.translation" => {
            &transform.translation[..]
        }
        "rotation" | "transform.rotation" | "transform.rotation_quaternion" => {
            &transform.rotation[..]
        }
        "rotation_euler" | "transform.rotation_euler" => {
            let rotation = DQuat::from_xyzw(
                transform.rotation[0],
                transform.rotation[1],
                transform.rotation[2],
                transform.rotation[3],
            );
            let euler = rotation.to_euler(EulerRot::XYZ);
            return [euler.0, euler.1, euler.2]
                .get(usize::try_from(index).ok()?)
                .copied();
        }
        _ => return None,
    };
    values.get(usize::try_from(index).ok()?).copied()
}

fn apply_transform_drivers(
    node_id: &Id,
    mut transform: Transform,
    drivers: &BTreeMap<(Id, String, u32), f64>,
) -> Result<Transform> {
    let mut euler = None;
    let mut euler_modified = false;
    let mut quaternion_modified = false;
    for ((target_id, path, index), value) in drivers {
        if target_id != node_id {
            continue;
        }
        if !value.is_finite() {
            return Err(evaluation_error_with(
                "transform driver value is non-finite",
                json!({ "node": node_id, "path": path }),
            ));
        }
        match path.as_str() {
            "location" | "translation" | "transform.location" | "transform.translation" => {
                set_component(&mut transform.translation, *index, *value, path)?;
            }
            "scale" | "transform.scale" => {
                set_component(&mut transform.scale, *index, *value, path)?;
            }
            "rotation" | "transform.rotation" | "transform.rotation_quaternion" => {
                set_component(&mut transform.rotation, *index, *value, path)?;
                quaternion_modified = true;
            }
            "rotation_euler" | "transform.rotation_euler" => {
                let angles = euler.get_or_insert_with(|| {
                    let rotation = transform.rotation_quat();
                    let (x, y, z) = rotation.to_euler(EulerRot::XYZ);
                    [x, y, z]
                });
                set_component(angles, *index, *value, path)?;
                euler_modified = true;
            }
            _ => {}
        }
    }
    if euler_modified {
        let angles = euler.ok_or_else(|| evaluation_error("Euler driver channels are missing"))?;
        let rotation = DQuat::from_euler(EulerRot::XYZ, angles[0], angles[1], angles[2]);
        transform.rotation = [rotation.x, rotation.y, rotation.z, rotation.w];
    }
    if quaternion_modified || euler_modified {
        transform.rotation = crate::model::normalize_rotation(transform.rotation)
            .map_err(|error| evaluation_error(error.message))?;
    }
    Ok(transform)
}

fn set_component(values: &mut [f64], index: u32, value: f64, path: &str) -> Result<()> {
    let index = usize::try_from(index)
        .map_err(|_| evaluation_error(format!("driver index for {path} is invalid")))?;
    let component = values.get_mut(index).ok_or_else(|| {
        evaluation_error(format!(
            "driver index for {path} is outside its component range"
        ))
    })?;
    *component = value;
    Ok(())
}

fn animated_pose_channel(
    node: &Node,
    doc: &SceneDoc,
    frame: f64,
    path: &str,
    index: u32,
) -> Result<Option<f64>> {
    if !path.starts_with("pose.") {
        return Ok(None);
    }
    let pose = crate::eval::animation::animated_pose_bones(node, doc, frame)?;
    Ok(pose_channel(&pose, path, index))
}

fn apply_pose_drivers(
    node_id: &Id,
    pose_bones: &mut BTreeMap<Id, PoseBone>,
    drivers: &BTreeMap<(Id, String, u32), f64>,
) -> Result<()> {
    for ((target_id, path, index), value) in drivers {
        if target_id != node_id {
            continue;
        }
        let Some((bone_id, channel)) = pose_channel_path(path) else {
            continue;
        };
        let pose = pose_bones.get_mut(&bone_id).ok_or_else(|| {
            evaluation_error_with(
                "pose driver targets an unknown bone",
                json!({ "node": node_id, "bone_id": bone_id }),
            )
        })?;
        let values = match channel {
            "location" | "translation" => &mut pose.translation[..],
            "rotation" => &mut pose.rotation[..],
            "scale" => &mut pose.scale[..],
            _ => continue,
        };
        if !value.is_finite() {
            return Err(evaluation_error_with(
                "pose driver value is non-finite",
                json!({ "node": node_id, "bone_id": bone_id, "path": path }),
            ));
        }
        set_component(values, *index, *value, path)?;
    }
    Ok(())
}

fn pose_channel(pose_bones: &BTreeMap<Id, PoseBone>, path: &str, index: u32) -> Option<f64> {
    let (bone_id, channel) = pose_channel_path(path)?;
    let pose = pose_bones.get(&bone_id)?;
    let values = match channel {
        "location" | "translation" => &pose.translation[..],
        "rotation" => &pose.rotation[..],
        "scale" => &pose.scale[..],
        _ => return None,
    };
    values.get(usize::try_from(index).ok()?).copied()
}

fn pose_channel_path(path: &str) -> Option<(Id, &str)> {
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
    Some((Id::new(bone_key).ok()?, channel))
}

fn shape_key_value_from_path(shape_keys: &ShapeKeyData, path: &str) -> Option<f64> {
    if matches!(
        path,
        "shape_keys.evaluation_time" | "shape_key.evaluation_time" | "key_blocks.eval_time"
    ) {
        return Some(shape_keys.evaluation_time);
    }
    let path = path.strip_suffix(".value")?;
    let key_id = path
        .strip_prefix("shape_key.")
        .or_else(|| path.strip_prefix("shape_keys."))
        .and_then(|value| Id::new(value).ok());
    if let Some(key_id) = key_id
        && let Some(key) = shape_keys.keys.get(&key_id)
    {
        return Some(key.value);
    }
    let start = path.find("key_blocks[")? + "key_blocks[".len();
    let end = path[start..].find(']')? + start;
    let name = path[start..end].trim_matches(['"', '\'']);
    shape_keys
        .keys
        .values()
        .find(|key| key.name == name)
        .map(|key| key.value)
}

fn property_value(
    properties: &serde_json::Map<String, Value>,
    path: &str,
    index: u32,
) -> Option<f64> {
    let stripped = path.strip_prefix("properties.").unwrap_or(path);
    let mut components = stripped.split('.');
    let mut value = properties.get(components.next()?)?;
    for component in components {
        value = value.get(component)?;
    }
    match value {
        Value::Number(number) => number.as_f64(),
        Value::Array(values) => values.get(usize::try_from(index).ok()?)?.as_f64(),
        _ => None,
    }
}

fn evaluate_expression(expression: &str, variables: &BTreeMap<String, f64>) -> Result<f64> {
    let mut parser = ExpressionParser {
        bytes: expression.as_bytes(),
        offset: 0,
        variables,
        depth: 0,
    };
    let result = parser.parse_expression(0)?;
    parser.skip_whitespace();
    if parser.offset != parser.bytes.len() {
        return Err(evaluation_error(
            "driver expression contains trailing input",
        ));
    }
    if !result.is_finite() {
        return Err(evaluation_error("driver expression result is non-finite"));
    }
    Ok(result)
}

struct ExpressionParser<'a> {
    bytes: &'a [u8],
    offset: usize,
    variables: &'a BTreeMap<String, f64>,
    depth: usize,
}

impl ExpressionParser<'_> {
    fn skip_whitespace(&mut self) {
        while self
            .bytes
            .get(self.offset)
            .is_some_and(u8::is_ascii_whitespace)
        {
            self.offset += 1;
        }
    }

    fn parse_expression(&mut self, minimum_binding: u8) -> Result<f64> {
        if self.depth >= 128 {
            return Err(evaluation_error("driver expression nesting limit exceeded"));
        }
        self.depth += 1;
        let result = self.parse_expression_inner(minimum_binding);
        self.depth -= 1;
        result
    }

    fn parse_expression_inner(&mut self, minimum_binding: u8) -> Result<f64> {
        self.skip_whitespace();
        let mut left = match self.bytes.get(self.offset).copied() {
            Some(b'+') => {
                self.offset += 1;
                self.parse_expression(4)?
            }
            Some(b'-') => {
                self.offset += 1;
                -self.parse_expression(4)?
            }
            Some(b'(') => {
                self.offset += 1;
                let value = self.parse_expression(0)?;
                self.skip_whitespace();
                if self.bytes.get(self.offset) != Some(&b')') {
                    return Err(evaluation_error(
                        "driver expression has an unclosed parenthesis",
                    ));
                }
                self.offset += 1;
                value
            }
            Some(byte) if byte.is_ascii_digit() || byte == b'.' => self.parse_number()?,
            Some(byte) if byte.is_ascii_alphabetic() || byte == b'_' => self.parse_identifier()?,
            _ => return Err(evaluation_error("driver expression expected a value")),
        };
        loop {
            self.skip_whitespace();
            let Some(operator) = self.bytes.get(self.offset).copied() else {
                break;
            };
            let (left_binding, right_binding) = match operator {
                b'+' | b'-' => (1, 2),
                b'*' | b'/' | b'%' => (3, 4),
                b'^' => (5, 5),
                _ => break,
            };
            if left_binding < minimum_binding {
                break;
            }
            self.offset += 1;
            let right = self.parse_expression(right_binding)?;
            left = match operator {
                b'+' => left + right,
                b'-' => left - right,
                b'*' => left * right,
                b'/' if right != 0.0 => left / right,
                b'%' if right != 0.0 => left % right,
                b'^' => left.powf(right),
                b'/' | b'%' => return Err(evaluation_error("driver expression divides by zero")),
                _ => {
                    return Err(evaluation_error(
                        "driver expression operator is unsupported",
                    ));
                }
            };
            if !left.is_finite() {
                return Err(evaluation_error(
                    "driver expression arithmetic is non-finite",
                ));
            }
        }
        Ok(left)
    }

    fn parse_number(&mut self) -> Result<f64> {
        let start = self.offset;
        let mut saw_digit = false;
        while self.bytes.get(self.offset).is_some_and(u8::is_ascii_digit) {
            self.offset += 1;
            saw_digit = true;
        }
        if self.bytes.get(self.offset) == Some(&b'.') {
            self.offset += 1;
            while self.bytes.get(self.offset).is_some_and(u8::is_ascii_digit) {
                self.offset += 1;
                saw_digit = true;
            }
        }
        if !saw_digit {
            return Err(evaluation_error("driver expression has an invalid number"));
        }
        if self
            .bytes
            .get(self.offset)
            .is_some_and(|byte| *byte == b'e' || *byte == b'E')
        {
            self.offset += 1;
            if self
                .bytes
                .get(self.offset)
                .is_some_and(|byte| *byte == b'+' || *byte == b'-')
            {
                self.offset += 1;
            }
            let exponent_start = self.offset;
            while self.bytes.get(self.offset).is_some_and(u8::is_ascii_digit) {
                self.offset += 1;
            }
            if self.offset == exponent_start {
                return Err(evaluation_error(
                    "driver expression has an invalid exponent",
                ));
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.offset])
            .map_err(|_| evaluation_error("driver expression number is invalid"))?;
        text.parse::<f64>()
            .map_err(|_| evaluation_error("driver expression number is invalid"))
    }

    fn parse_identifier(&mut self) -> Result<f64> {
        let start = self.offset;
        while self
            .bytes
            .get(self.offset)
            .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
        {
            self.offset += 1;
        }
        let name = std::str::from_utf8(&self.bytes[start..self.offset])
            .map_err(|_| evaluation_error("driver expression identifier is invalid"))?;
        self.skip_whitespace();
        if self.bytes.get(self.offset) == Some(&b'(') {
            self.offset += 1;
            let mut arguments = Vec::new();
            self.skip_whitespace();
            if self.bytes.get(self.offset) != Some(&b')') {
                loop {
                    arguments.push(self.parse_expression(0)?);
                    self.skip_whitespace();
                    match self.bytes.get(self.offset) {
                        Some(b',') => self.offset += 1,
                        Some(b')') => break,
                        _ => {
                            return Err(evaluation_error(
                                "driver function arguments are malformed",
                            ));
                        }
                    }
                }
            }
            if self.bytes.get(self.offset) != Some(&b')') {
                return Err(evaluation_error(
                    "driver function call has no closing parenthesis",
                ));
            }
            self.offset += 1;
            return call_expression_function(name, &arguments);
        }
        match name {
            "pi" => Ok(std::f64::consts::PI),
            "e" => Ok(std::f64::consts::E),
            _ => self.variables.get(name).copied().ok_or_else(|| {
                evaluation_error(format!("driver expression variable {name} is undefined"))
            }),
        }
    }
}

fn call_expression_function(name: &str, arguments: &[f64]) -> Result<f64> {
    let unary = |function: fn(f64) -> f64| -> Result<f64> {
        if arguments.len() != 1 {
            return Err(evaluation_error(format!(
                "driver function {name} expects one argument"
            )));
        }
        Ok(function(arguments[0]))
    };
    let result = match name {
        "sin" => unary(f64::sin)?,
        "cos" => unary(f64::cos)?,
        "tan" => unary(f64::tan)?,
        "asin" => unary(f64::asin)?,
        "acos" => unary(f64::acos)?,
        "atan" => unary(f64::atan)?,
        "sqrt" => unary(f64::sqrt)?,
        "abs" => unary(f64::abs)?,
        "floor" => unary(f64::floor)?,
        "ceil" => unary(f64::ceil)?,
        "round" => unary(f64::round)?,
        "pow" if arguments.len() == 2 => arguments[0].powf(arguments[1]),
        "min" if !arguments.is_empty() => arguments.iter().copied().fold(f64::INFINITY, f64::min),
        "max" if !arguments.is_empty() => {
            arguments.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        }
        "pow" | "min" | "max" => {
            return Err(evaluation_error(format!(
                "driver function {name} has invalid arguments"
            )));
        }
        _ => {
            return Err(evaluation_error(format!(
                "driver function {name} is unsupported"
            )));
        }
    };
    if result.is_finite() {
        Ok(result)
    } else {
        Err(evaluation_error(format!(
            "driver function {name} is non-finite"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expression_evaluation_is_arithmetic_only() {
        let variables = BTreeMap::from([("x".to_owned(), 3.0)]);
        assert_eq!(
            evaluate_expression("2 + x * 4", &variables).ok(),
            Some(14.0)
        );
        assert!(evaluate_expression("x / 0", &variables).is_err());
        assert!(evaluate_expression("std::process::exit(1)", &variables).is_err());
    }
    #[test]
    fn connected_pose_bones_ignore_pose_location_translation() -> Result<()> {
        let root_id = Id::new("root")?;
        let connected_id = Id::new("connected")?;
        let free_id = Id::new("free")?;
        let bone =
            |name: &str, parent: Option<Id>, head: [f64; 3], tail: [f64; 3], use_connect: bool| {
                crate::model::Bone {
                    name: name.to_owned(),
                    parent,
                    head,
                    tail,
                    roll: 0.0,
                    deform: true,
                    inherit_rotation: true,
                    use_connect,
                    custom_shape: None,
                    envelope_distance: 0.25,
                    envelope_weight: 1.0,
                    head_radius: 0.1,
                    tail_radius: 0.1,
                    bbone_settings: BTreeMap::new(),
                }
            };
        let armature = ArmatureData {
            bones: BTreeMap::from([
                (
                    root_id.clone(),
                    bone("Root", None, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], false),
                ),
                (
                    connected_id.clone(),
                    bone(
                        "Connected",
                        Some(root_id.clone()),
                        [0.0, 1.0, 0.0],
                        [0.0, 2.0, 0.0],
                        true,
                    ),
                ),
                (
                    free_id.clone(),
                    bone(
                        "Free",
                        Some(root_id),
                        [2.0, 1.0, 0.0],
                        [2.0, 2.0, 0.0],
                        false,
                    ),
                ),
            ]),
            ..ArmatureData::default()
        };
        let pose = BTreeMap::from([
            (
                connected_id.clone(),
                PoseBone {
                    translation: [0.6, 0.0, 0.0],
                    ..PoseBone::default()
                },
            ),
            (
                free_id.clone(),
                PoseBone {
                    translation: [0.6, 0.0, 0.0],
                    ..PoseBone::default()
                },
            ),
        ]);
        let rest = evaluate_bone_matrices(&armature, &BTreeMap::new())?;
        let posed = evaluate_bone_matrices(&armature, &pose)?;
        let connected_delta = (posed
            .get(&connected_id)
            .ok_or_else(|| evaluation_error("connected test bone is missing"))?
            .pose
            .w_axis
            .truncate()
            - rest
                .get(&connected_id)
                .ok_or_else(|| evaluation_error("connected rest test bone is missing"))?
                .pose
                .w_axis
                .truncate())
        .length();
        let free_delta = (posed
            .get(&free_id)
            .ok_or_else(|| evaluation_error("free test bone is missing"))?
            .pose
            .w_axis
            .truncate()
            - rest
                .get(&free_id)
                .ok_or_else(|| evaluation_error("free rest test bone is missing"))?
                .pose
                .w_axis
                .truncate())
        .length();
        assert!(
            connected_delta < 1.0e-12,
            "connected bone moved by {connected_delta}"
        );
        assert!(
            (free_delta - 0.6).abs() < 1.0e-12,
            "free bone moved by {free_delta}"
        );
        Ok(())
    }

    #[test]
    fn target_project_uses_interpolated_vertex_normals() -> Result<()> {
        let vertices = [DVec3::ZERO, DVec3::X, DVec3::Y];
        let normals = [
            DVec3::Z,
            DVec3::new(0.5, 0.0, 3.0_f64.sqrt() * 0.5),
            DVec3::new(0.0, 0.5, 3.0_f64.sqrt() * 0.5),
        ];
        let weights = [0.5, 0.3, 0.2];
        let expected =
            vertices[0] * weights[0] + vertices[1] * weights[1] + vertices[2] * weights[2];
        let smooth_normal =
            normals[0] * weights[0] + normals[1] * weights[1] + normals[2] * weights[2];
        let point = expected + smooth_normal * 0.4;
        let closest = closest_point_on_triangle(point, vertices[0], vertices[1], vertices[2]);
        let (hit, normal) = target_project_triangle_hit(point, vertices, normals, closest)
            .ok_or_else(|| evaluation_error("target-project triangle solve did not converge"))?;
        assert!(
            hit.distance(expected) <= 1.0e-6,
            "target-project surface point differs: {hit:?} != {expected:?}"
        );
        assert!(
            (normal.dot(smooth_normal.normalize()) - 1.0).abs() <= 1.0e-6,
            "target-project normal differs from the interpolated vertex normal"
        );
        Ok(())
    }
    #[test]
    fn target_project_boundary_fallback_is_not_nearest_surface() -> Result<()> {
        let vertices = [DVec3::ZERO, DVec3::X, DVec3::new(1.4, 1.0, 0.0)];
        let normals = [DVec3::Z; 3];
        let point = DVec3::new(0.4, -0.25, 0.6);
        let closest = closest_point_on_triangle(point, vertices[0], vertices[1], vertices[2]);
        assert!(
            target_project_triangle_hit(point, vertices, normals, closest).is_none(),
            "a normal ray outside the triangle must reach Blender's boundary fallback"
        );

        let boundary_a = (DVec3::X - DVec3::Y).normalize();
        let adjacent_edge = DVec3::new(0.4, 1.0, 0.0).normalize();
        let boundary_b = (DVec3::X + adjacent_edge).normalize();
        let project_boundary = |direction| ShrinkwrapBoundaryVertex {
            direction,
            normal_plane: DVec3::Z.cross(direction).cross(DVec3::Z).normalize(),
        };
        let (hit, _) = target_project_edge_hit(
            point,
            [DVec3::ZERO, DVec3::X],
            [project_boundary(boundary_a), project_boundary(boundary_b)],
            [DVec3::Z; 2],
        )
        .ok_or_else(|| evaluation_error("asymmetric boundary edge projection was not found"))?;
        assert!(
            (hit.x - 0.446_383_6).abs() <= 1.0e-5 && hit.y.abs() <= 1.0e-5 && hit.z.abs() <= 1.0e-5,
            "edge fallback differs from Blender's projected boundary hit: {hit:?}"
        );
        assert!(
            hit.distance(DVec3::new(0.4, 0.0, 0.0)) > 0.04,
            "the fixture must distinguish boundary projection from simple nearest-surface fallback"
        );
        Ok(())
    }
}
