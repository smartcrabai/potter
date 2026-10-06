use std::collections::{BTreeMap, BTreeSet};

use glam::{DMat4, DVec3};
use serde_json::{Map, Value};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::Vertex,
    model::{
        Bone, BoneCollection, Constraint, ConstraintType, Driver, DriverType, DriverVariable,
        DriverVariableType, Id, Node, ParentType, ShapeKey, ShapeKeyData, VertexGroup,
        normalize_rotation,
    },
    params::{self, ParameterFamily},
};

use super::{
    ChangeKind, Engine, check_fields, operation_pointer, read_bool, read_id, read_string,
    resolve_node_targets,
};

#[derive(Default)]
struct BasicHumanRigPlan {
    bones: Vec<(Id, Bone)>,
    deform_bones: Vec<Id>,
    control_bones: Vec<Id>,
    controllers: Vec<(Id, Node)>,
    controller_ids: BTreeMap<String, Id>,
    constraints: Vec<Constraint>,
    collections: BTreeMap<Id, BoneCollection>,
}

pub(super) fn apply(
    engine: &mut Engine<'_>,
    name: &str,
    operation: &Map<String, Value>,
) -> Result<bool> {
    match name {
        "bone.create" => bone_create(engine, operation),
        "bone.update" => bone_update(engine, operation),
        "bone.delete" => bone_delete(engine, operation),
        "bone_collection.create" => bone_collection_create(engine, operation),
        "bone_collection.update" => bone_collection_update(engine, operation),
        "bone_collection.delete" => bone_collection_delete(engine, operation),
        "bone_collection.assign" => bone_collection_assign(engine, operation, true),
        "rig.generate_basic_human" => rig_generate_basic_human(engine, operation),
        "bone_collection.unassign" => bone_collection_assign(engine, operation, false),
        "pose.set" => pose_set(engine, operation),
        "pose.reset" => pose_reset(engine, operation),
        "constraint.create" => constraint_create(engine, operation),
        "constraint.update" => constraint_update(engine, operation),
        "constraint.delete" => constraint_delete(engine, operation),
        "constraint.reorder" => constraint_reorder(engine, operation),
        "shape_key.create" => shape_key_create(engine, operation),
        "shape_key.update" => shape_key_update(engine, operation),
        "shape_key.delete" => shape_key_delete(engine, operation),
        "vertex_group.create" => vertex_group_create(engine, operation),
        "vertex_group.assign" => vertex_group_assign(engine, operation),
        "vertex_group.remove" => vertex_group_remove(engine, operation),
        "driver.create" => driver_create(engine, operation),
        "driver.update" => driver_update(engine, operation),
        "driver.delete" => driver_delete(engine, operation),
        _ => Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("unsupported rig operation `{name}`"),
            &pointer(engine, "op"),
        )),
    }
}

fn bone_create(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &[
            "op",
            "target",
            "id",
            "name",
            "head",
            "tail",
            "parent",
            "roll",
            "deform",
            "inherit_rotation",
            "use_connect",
            "custom_shape",
            "envelope_distance",
            "envelope_weight",
            "head_radius",
            "tail_radius",
        ],
        &["target", "id", "name", "head", "tail"],
    )?;
    let node_id = target_node(engine, operation)?;
    let id = read_id(engine, operation, "id")?;
    let name = read_string(engine, operation, "name")?;
    let head = read_vec3(
        engine,
        operation
            .get("head")
            .ok_or_else(|| invalid(engine, "head is required", "head"))?,
        "head",
    )?;
    let tail = read_vec3(
        engine,
        operation
            .get("tail")
            .ok_or_else(|| invalid(engine, "tail is required", "tail"))?,
        "tail",
    )?;
    validate_bone_segment(engine, head, tail, "head")?;
    let parent = optional_id(engine, operation.get("parent"), "parent")?;
    let roll = optional_finite(engine, operation.get("roll"), "roll")?.unwrap_or(0.0);
    let deform = read_bool(engine, operation, "deform", true)?;
    let inherit_rotation = read_bool(engine, operation, "inherit_rotation", true)?;
    let use_connect = read_bool(engine, operation, "use_connect", false)?;
    let custom_shape = optional_id(engine, operation.get("custom_shape"), "custom_shape")?;
    validate_custom_shape_reference(engine, custom_shape.as_ref(), "custom_shape")?;
    let envelope_distance = nonnegative_optional(
        engine,
        operation.get("envelope_distance"),
        "envelope_distance",
        0.25,
    )?;
    let envelope_weight = operation.get("envelope_weight").map_or(Ok(1.0), |value| {
        ranged_number(engine, value, 0.0, 1.0, "envelope_weight")
    })?;
    let head_radius =
        nonnegative_optional(engine, operation.get("head_radius"), "head_radius", 0.1)?;
    let tail_radius =
        nonnegative_optional(engine, operation.get("tail_radius"), "tail_radius", 0.1)?;
    let armature = armature(engine, &node_id)?;
    if armature.bones.contains_key(&id) {
        return Err(engine.error(
            ErrorCode::IdExists,
            format!("bone ID `{id}` already exists on armature `{node_id}`"),
            &pointer(engine, "id"),
        ));
    }
    validate_bone_parent(
        engine,
        &armature.bones,
        &id,
        parent.as_ref(),
        use_connect,
        head,
    )?;
    let node = engine
        .doc
        .nodes
        .get_mut(&node_id)
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "armature node disappeared"))?;
    let data_id = node
        .data
        .clone()
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "armature data disappeared"))?;
    let data = engine.doc.data_blocks.get_mut(&data_id).ok_or_else(|| {
        PotError::new(ErrorCode::InternalError, "armature data block disappeared")
    })?;
    data.armature
        .as_mut()
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "armature model disappeared"))?
        .bones
        .insert(
            id,
            Bone {
                name,
                parent,
                head,
                tail,
                roll,
                deform,
                inherit_rotation,
                use_connect,
                custom_shape,
                envelope_distance,
                envelope_weight,
                head_radius,
                tail_radius,
                bbone_settings: BTreeMap::new(),
            },
        );
    engine.mark("data_blocks", &data_id, ChangeKind::Updated);
    Ok(true)
}

fn bone_update(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "bone_id", "set"],
        &["target", "bone_id", "set"],
    )?;
    let node_id = target_node(engine, operation)?;
    let id = read_id(engine, operation, "bone_id")?;
    let set = read_set(engine, operation)?;
    reject_unknown_set_fields(
        engine,
        set,
        &[
            "name",
            "head",
            "tail",
            "parent",
            "roll",
            "deform",
            "inherit_rotation",
            "use_connect",
            "custom_shape",
            "envelope_distance",
            "envelope_weight",
            "head_radius",
            "tail_radius",
        ],
    )?;
    let current_armature = armature(engine, &node_id)?;
    let old = current_armature
        .bones
        .get(&id)
        .ok_or_else(|| missing(engine, "bone", &id, "bone_id"))?;
    let mut next = old.clone();
    if let Some(value) = set.get("name") {
        next.name = string_value(engine, value, "set/name")?;
    }
    if let Some(value) = set.get("head") {
        next.head = read_vec3(engine, value, "set/head")?;
    }
    if let Some(value) = set.get("tail") {
        next.tail = read_vec3(engine, value, "set/tail")?;
    }
    validate_bone_segment(engine, next.head, next.tail, "set/head")?;
    if let Some(value) = set.get("parent") {
        next.parent = optional_id(engine, Some(value), "set/parent")?;
    }
    if let Some(value) = set.get("roll") {
        next.roll = finite_value(engine, value, "set/roll")?;
    }
    if let Some(value) = set.get("deform") {
        next.deform = bool_value(engine, value, "set/deform")?;
    }
    if let Some(value) = set.get("inherit_rotation") {
        next.inherit_rotation = bool_value(engine, value, "set/inherit_rotation")?;
    }
    if let Some(value) = set.get("use_connect") {
        next.use_connect = bool_value(engine, value, "set/use_connect")?;
    }
    if let Some(value) = set.get("custom_shape") {
        next.custom_shape = optional_id(engine, Some(value), "set/custom_shape")?;
        validate_custom_shape_reference(engine, next.custom_shape.as_ref(), "set/custom_shape")?;
    }
    next.envelope_distance = nonnegative_optional(
        engine,
        set.get("envelope_distance"),
        "set/envelope_distance",
        next.envelope_distance,
    )?;
    if let Some(value) = set.get("envelope_weight") {
        next.envelope_weight = ranged_number(engine, value, 0.0, 1.0, "set/envelope_weight")?;
    }
    next.head_radius = nonnegative_optional(
        engine,
        set.get("head_radius"),
        "set/head_radius",
        next.head_radius,
    )?;
    next.tail_radius = nonnegative_optional(
        engine,
        set.get("tail_radius"),
        "set/tail_radius",
        next.tail_radius,
    )?;
    validate_bone_parent(
        engine,
        &current_armature.bones,
        &id,
        next.parent.as_ref(),
        next.use_connect,
        next.head,
    )?;
    if current_armature.bones.iter().any(|(child_id, child)| {
        child_id != &id
            && child.parent.as_ref() == Some(&id)
            && child.use_connect
            && !same_point(child.head, next.tail)
    }) {
        return Err(invalid(
            engine,
            "connected child heads must match the updated bone tail",
            "set/tail",
        ));
    }
    if next == *old {
        return Ok(false);
    }
    let data_id = data_id(engine, &node_id)?;
    let data = engine.doc.data_blocks.get_mut(&data_id).ok_or_else(|| {
        PotError::new(ErrorCode::InternalError, "armature data block disappeared")
    })?;
    data.armature
        .as_mut()
        .and_then(|armature| armature.bones.get_mut(&id))
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "bone disappeared"))?
        .clone_from(&next);
    engine.mark("data_blocks", &data_id, ChangeKind::Updated);
    Ok(true)
}

fn bone_delete(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "bone_id"],
        &["target", "bone_id"],
    )?;
    let node_id = target_node(engine, operation)?;
    let id = read_id(engine, operation, "bone_id")?;
    let armature = armature(engine, &node_id)?;
    if !armature.bones.contains_key(&id) {
        return Err(missing(engine, "bone", &id, "bone_id"));
    }
    if armature
        .bones
        .values()
        .any(|bone| bone.parent.as_ref() == Some(&id))
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("bone `{id}` cannot be deleted while child bones reference it"),
            &pointer(engine, "bone_id"),
        ));
    }
    for (constraint_owner_id, owner) in &engine.doc.nodes {
        if owner.constraints.iter().any(|constraint| {
            constraint.owner_bone.as_ref() == Some(&id)
                || (constraint.subtarget.as_ref() == Some(&id)
                    && (if matches!(
                        constraint.constraint_type,
                        ConstraintType::Ik | ConstraintType::SplineIk
                    ) {
                        constraint_owner_id == &node_id
                    } else {
                        constraint.target.as_ref() == Some(&node_id)
                    }))
        }) {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("bone `{id}` is referenced by a constraint"),
                &pointer(engine, "bone_id"),
            ));
        }
    }
    if engine.doc.nodes.values().any(|owner| {
        owner.parent.as_ref() == Some(&node_id)
            && owner.parent_type == ParentType::Bone
            && owner.parent_bone.as_ref() == Some(&id)
    }) {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("bone `{id}` is referenced by a bone-parented node"),
            &pointer(engine, "bone_id"),
        ));
    }
    let data_id = data_id(engine, &node_id)?;
    let node = engine
        .doc
        .nodes
        .get_mut(&node_id)
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "armature node disappeared"))?;
    node.pose.remove(&id);
    let data = engine.doc.data_blocks.get_mut(&data_id).ok_or_else(|| {
        PotError::new(ErrorCode::InternalError, "armature data block disappeared")
    })?;
    let armature = data
        .armature
        .as_mut()
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "armature model disappeared"))?;
    for collection in armature.bone_collections.values_mut() {
        collection.bones.retain(|bone_id| bone_id != &id);
    }
    let bones = &mut armature.bones;
    if bones.remove(&id).is_none() {
        return Err(PotError::new(ErrorCode::InternalError, "bone disappeared"));
    }
    engine.mark("nodes", &node_id, ChangeKind::Updated);
    engine.mark("data_blocks", &data_id, ChangeKind::Updated);
    Ok(true)
}

fn bone_collection_create(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "id", "name", "visible"],
        &["target", "id", "name"],
    )?;
    let node_id = target_node(engine, operation)?;
    let id = read_id(engine, operation, "id")?;
    let name = read_string(engine, operation, "name")?;
    let visible = read_bool(engine, operation, "visible", true)?;
    if armature(engine, &node_id)?
        .bone_collections
        .contains_key(&id)
    {
        return Err(engine.error(
            ErrorCode::IdExists,
            format!("bone collection ID `{id}` already exists"),
            &pointer(engine, "id"),
        ));
    }
    let data_id = data_id(engine, &node_id)?;
    armature_mut(engine, &node_id)?.bone_collections.insert(
        id,
        BoneCollection {
            name,
            bones: Vec::new(),
            visible,
        },
    );
    engine.mark("data_blocks", &data_id, ChangeKind::Updated);
    Ok(true)
}

fn bone_collection_update(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "id", "set"],
        &["target", "id", "set"],
    )?;
    let node_id = target_node(engine, operation)?;
    let id = read_id(engine, operation, "id")?;
    let set = read_set(engine, operation)?;
    reject_unknown_set_fields(engine, set, &["name", "visible"])?;
    let old = armature(engine, &node_id)?
        .bone_collections
        .get(&id)
        .cloned()
        .ok_or_else(|| missing(engine, "bone collection", &id, "id"))?;
    let mut next = old.clone();
    if let Some(value) = set.get("name") {
        next.name = string_value(engine, value, "set/name")?;
    }
    if let Some(value) = set.get("visible") {
        next.visible = bool_value(engine, value, "set/visible")?;
    }
    if next == old {
        return Ok(false);
    }
    let data_id = data_id(engine, &node_id)?;
    armature_mut(engine, &node_id)?
        .bone_collections
        .insert(id, next);
    engine.mark("data_blocks", &data_id, ChangeKind::Updated);
    Ok(true)
}

fn bone_collection_delete(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "id"],
        &["target", "id"],
    )?;
    let node_id = target_node(engine, operation)?;
    let id = read_id(engine, operation, "id")?;
    if !armature(engine, &node_id)?
        .bone_collections
        .contains_key(&id)
    {
        return Err(missing(engine, "bone collection", &id, "id"));
    }
    let data_id = data_id(engine, &node_id)?;
    armature_mut(engine, &node_id)?.bone_collections.remove(&id);
    engine.mark("data_blocks", &data_id, ChangeKind::Updated);
    Ok(true)
}

fn bone_collection_assign(
    engine: &mut Engine<'_>,
    operation: &Map<String, Value>,
    assign: bool,
) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "collection_id", "bone_ids"],
        &["target", "collection_id", "bone_ids"],
    )?;
    let node_id = target_node(engine, operation)?;
    let collection_id = read_id(engine, operation, "collection_id")?;
    let bone_ids = parse_bone_ids(
        engine,
        operation
            .get("bone_ids")
            .ok_or_else(|| invalid(engine, "bone_ids is required", "bone_ids"))?,
        "bone_ids",
    )?;
    let armature_data = armature(engine, &node_id)?;
    let collection = armature_data
        .bone_collections
        .get(&collection_id)
        .ok_or_else(|| missing(engine, "bone collection", &collection_id, "collection_id"))?;
    for bone_id in &bone_ids {
        if !armature_data.bones.contains_key(bone_id) {
            return Err(missing(engine, "bone", bone_id, "bone_ids"));
        }
    }
    let changed = if assign {
        bone_ids
            .iter()
            .any(|bone_id| !collection.bones.contains(bone_id))
    } else {
        bone_ids
            .iter()
            .any(|bone_id| collection.bones.contains(bone_id))
    };
    if !changed {
        return Ok(false);
    }
    let data_id = data_id(engine, &node_id)?;
    let collection = armature_mut(engine, &node_id)?
        .bone_collections
        .get_mut(&collection_id)
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "bone collection disappeared"))?;
    if assign {
        for bone_id in bone_ids {
            if !collection.bones.contains(&bone_id) {
                collection.bones.push(bone_id);
            }
        }
    } else {
        collection
            .bones
            .retain(|bone_id| !bone_ids.contains(bone_id));
    }
    engine.mark("data_blocks", &data_id, ChangeKind::Updated);
    Ok(true)
}

fn rig_generate_basic_human(
    engine: &mut Engine<'_>,
    operation: &Map<String, Value>,
) -> Result<bool> {
    check_fields(engine, operation, &["op", "target", "scale"], &["target"])?;
    let armature_id = target_node(engine, operation)?;
    let scale = optional_finite(engine, operation.get("scale"), "scale")?.unwrap_or(1.0);
    if scale <= 0.0 {
        return Err(invalid(engine, "scale must be greater than zero", "scale"));
    }
    let plan = basic_human_rig_plan(&armature_id, scale)?;
    let data_id = data_id(engine, &armature_id)?;
    let current = armature(engine, &armature_id)?;
    for (id, _) in &plan.bones {
        if current.bones.contains_key(id) {
            return Err(engine.error(
                ErrorCode::IdExists,
                format!("generated bone ID `{id}` already exists"),
                &pointer(engine, "target"),
            ));
        }
    }
    for id in plan.collections.keys() {
        if current.bone_collections.contains_key(id) {
            return Err(engine.error(
                ErrorCode::IdExists,
                format!("generated bone collection ID `{id}` already exists"),
                &pointer(engine, "target"),
            ));
        }
    }
    let owner = engine
        .doc
        .nodes
        .get(&armature_id)
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "armature node disappeared"))?;
    for constraint in &plan.constraints {
        if owner
            .constraints
            .iter()
            .any(|existing| existing.id == constraint.id)
        {
            return Err(engine.error(
                ErrorCode::IdExists,
                format!("generated constraint ID `{}` already exists", constraint.id),
                &pointer(engine, "target"),
            ));
        }
    }
    for (id, _) in &plan.controllers {
        if engine.doc.nodes.contains_key(id) {
            return Err(engine.error(
                ErrorCode::IdExists,
                format!("generated controller node ID `{id}` already exists"),
                &pointer(engine, "target"),
            ));
        }
    }
    let collection_id = engine
        .doc
        .collections
        .iter()
        .find(|(_, collection)| collection.objects.contains(&armature_id))
        .map(|(id, _)| id.clone())
        .or_else(|| {
            engine
                .doc
                .scenes
                .get(&engine.doc.active_scene)
                .map(|scene| scene.root_collection.clone())
        })
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::SceneInvalid,
                "active scene has no root collection",
            )
        })?;
    if !engine.doc.collections.contains_key(&collection_id) {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "generated rig controller collection does not exist",
        ));
    }
    let BasicHumanRigPlan {
        bones,
        collections,
        controllers,
        constraints,
        ..
    } = plan;
    {
        let armature = armature_mut(engine, &armature_id)?;
        armature.bones.extend(bones);
        armature.bone_collections.extend(collections);
    }
    {
        let owner =
            engine.doc.nodes.get_mut(&armature_id).ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "armature node disappeared")
            })?;
        owner.constraints.extend(constraints);
    }
    let mut controller_ids = Vec::with_capacity(controllers.len());
    for (id, node) in controllers {
        controller_ids.push(id.clone());
        engine.doc.nodes.insert(id.clone(), node);
        engine.mark("nodes", &id, ChangeKind::Created);
    }
    let collection = engine
        .doc
        .collections
        .get_mut(&collection_id)
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "rig collection disappeared"))?;
    for id in controller_ids {
        if !collection.objects.contains(&id) {
            collection.objects.push(id);
        }
    }
    engine.mark("nodes", &armature_id, ChangeKind::Updated);
    engine.mark("data_blocks", &data_id, ChangeKind::Updated);
    engine.mark("collections", &collection_id, ChangeKind::Updated);
    Ok(true)
}

fn basic_human_rig_plan(armature_id: &Id, scale: f64) -> Result<BasicHumanRigPlan> {
    let mut plan = BasicHumanRigPlan::default();
    for (id, name, parent, head, tail) in [
        ("root", "root", None, [0.0, 0.0, 0.0], [0.0, 0.0, 0.15]),
        (
            "spine",
            "spine",
            Some("root"),
            [0.0, 0.0, 0.15],
            [0.0, 0.0, 0.45],
        ),
        (
            "spine_001",
            "spine.001",
            Some("spine"),
            [0.0, 0.0, 0.45],
            [0.0, 0.0, 0.75],
        ),
        (
            "spine_002",
            "spine.002",
            Some("spine_001"),
            [0.0, 0.0, 0.75],
            [0.0, 0.0, 1.05],
        ),
        (
            "spine_003",
            "spine.003",
            Some("spine_002"),
            [0.0, 0.0, 1.05],
            [0.0, 0.0, 1.25],
        ),
        (
            "neck",
            "neck",
            Some("spine_003"),
            [0.0, 0.0, 1.25],
            [0.0, 0.0, 1.45],
        ),
        (
            "head",
            "head",
            Some("neck"),
            [0.0, 0.0, 1.45],
            [0.0, 0.0, 1.8],
        ),
    ] {
        append_human_bone(&mut plan, id, name, parent, head, tail, scale, true)?;
    }
    for (side, sign) in [("l", 1.0), ("r", -1.0)] {
        let suffix = if side == "l" { "L" } else { "R" };
        for (id, name, parent, head, tail) in [
            (
                format!("shoulder_{side}"),
                format!("shoulder.{suffix}"),
                "spine_003".to_owned(),
                [0.0, 0.0, 1.2],
                [0.25 * sign, 0.0, 1.2],
            ),
            (
                format!("upper_arm_{side}"),
                format!("upper_arm.{suffix}"),
                format!("shoulder_{side}"),
                [0.25 * sign, 0.0, 1.2],
                [0.65 * sign, 0.0, 1.2],
            ),
            (
                format!("forearm_{side}"),
                format!("forearm.{suffix}"),
                format!("upper_arm_{side}"),
                [0.65 * sign, 0.0, 1.2],
                [1.0 * sign, 0.0, 1.2],
            ),
            (
                format!("hand_{side}"),
                format!("hand.{suffix}"),
                format!("forearm_{side}"),
                [1.0 * sign, 0.0, 1.2],
                [1.25 * sign, 0.0, 1.2],
            ),
            (
                format!("thigh_{side}"),
                format!("thigh.{suffix}"),
                "root".to_owned(),
                [0.15 * sign, 0.0, 0.9],
                [0.15 * sign, 0.0, 0.5],
            ),
            (
                format!("shin_{side}"),
                format!("shin.{suffix}"),
                format!("thigh_{side}"),
                [0.15 * sign, 0.0, 0.5],
                [0.15 * sign, 0.0, 0.1],
            ),
            (
                format!("foot_{side}"),
                format!("foot.{suffix}"),
                format!("shin_{side}"),
                [0.15 * sign, 0.0, 0.1],
                [0.15 * sign, 0.25, 0.05],
            ),
            (
                format!("toe_{side}"),
                format!("toe.{suffix}"),
                format!("foot_{side}"),
                [0.15 * sign, 0.25, 0.05],
                [0.15 * sign, 0.4, 0.05],
            ),
        ] {
            append_human_bone(
                &mut plan,
                &id,
                &name,
                Some(&parent),
                head,
                tail,
                scale,
                true,
            )?;
        }
        for (role, name, head, tail, translation) in [
            (
                format!("hand_ik_{side}"),
                format!("hand_ik.{suffix}"),
                [1.0 * sign, 0.0, 1.2],
                [1.25 * sign, 0.0, 1.2],
                [1.25 * sign, 0.0, 1.2],
            ),
            (
                format!("elbow_pole_{side}"),
                format!("elbow_pole.{suffix}"),
                [0.65 * sign, -1.0, 1.2],
                [0.65 * sign, -1.25, 1.2],
                [0.65 * sign, -1.0, 1.2],
            ),
            (
                format!("foot_ik_{side}"),
                format!("foot_ik.{suffix}"),
                [0.15 * sign, 0.0, 0.1],
                [0.15 * sign, 0.25, 0.05],
                [0.15 * sign, 0.25, 0.05],
            ),
            (
                format!("knee_pole_{side}"),
                format!("knee_pole.{suffix}"),
                [0.15 * sign, -1.0, 0.5],
                [0.15 * sign, -1.25, 0.5],
                [0.15 * sign, -1.0, 0.5],
            ),
        ] {
            append_human_bone(&mut plan, &role, &name, None, head, tail, scale, false)?;
            append_rig_controller(&mut plan, armature_id, &role, &name, translation, scale)?;
        }
        append_human_ik_constraint(
            &mut plan,
            &format!("ik_arm_{side}"),
            &format!("IK Arm {suffix}"),
            &format!("hand_ik_{side}"),
            &format!("elbow_pole_{side}"),
            &format!("hand_{side}"),
        )?;
        append_human_ik_constraint(
            &mut plan,
            &format!("ik_leg_{side}"),
            &format!("IK Leg {suffix}"),
            &format!("foot_ik_{side}"),
            &format!("knee_pole_{side}"),
            &format!("foot_{side}"),
        )?;
    }
    plan.collections.insert(
        rig_plan_id("rigify_deform")?,
        BoneCollection {
            name: "DEF".to_owned(),
            bones: plan.deform_bones.clone(),
            visible: true,
        },
    );
    plan.collections.insert(
        rig_plan_id("rigify_controls")?,
        BoneCollection {
            name: "CTRL".to_owned(),
            bones: plan.control_bones.clone(),
            visible: true,
        },
    );
    Ok(plan)
}

fn append_human_bone(
    plan: &mut BasicHumanRigPlan,
    id: &str,
    name: &str,
    parent: Option<&str>,
    head: [f64; 3],
    tail: [f64; 3],
    scale: f64,
    deform: bool,
) -> Result<()> {
    let id = rig_plan_id(id)?;
    let parent = parent.map(rig_plan_id).transpose()?;
    let bone = Bone {
        name: name.to_owned(),
        parent,
        head: head.map(|value| value * scale),
        tail: tail.map(|value| value * scale),
        roll: 0.0,
        deform,
        inherit_rotation: true,
        use_connect: false,
        custom_shape: None,
        envelope_distance: 0.25,
        envelope_weight: 1.0,
        head_radius: 0.1,
        tail_radius: 0.1,
        bbone_settings: BTreeMap::new(),
    };
    if deform {
        plan.deform_bones.push(id.clone());
    } else {
        plan.control_bones.push(id.clone());
    }
    plan.bones.push((id, bone));
    Ok(())
}

fn append_rig_controller(
    plan: &mut BasicHumanRigPlan,
    armature_id: &Id,
    role: &str,
    name: &str,
    translation: [f64; 3],
    scale: f64,
) -> Result<()> {
    let id_text = format!("rig_{}_{}", armature_id.as_str(), role);
    let id = Id::new(id_text).map_err(|_| {
        PotError::new(
            ErrorCode::InvalidArgument,
            "armature ID is too long for generated rig controller IDs",
        )
    })?;
    let mut node = Node {
        name: name.to_owned(),
        ..Node::default()
    };
    node.transform.translation = translation.map(|value| value * scale);
    plan.controller_ids.insert(role.to_owned(), id.clone());
    plan.controllers.push((id, node));
    Ok(())
}

fn append_human_ik_constraint(
    plan: &mut BasicHumanRigPlan,
    id: &str,
    name: &str,
    target_role: &str,
    pole_role: &str,
    end_bone: &str,
) -> Result<()> {
    let target = plan
        .controller_ids
        .get(target_role)
        .cloned()
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "generated IK target is missing"))?;
    let pole = plan.controller_ids.get(pole_role).ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "generated IK pole target is missing",
        )
    })?;
    let mut params = Map::new();
    params.insert("chain_count".to_owned(), Value::from(3));
    params.insert("pole_target".to_owned(), Value::String(pole.to_string()));
    params.insert("pole_angle".to_owned(), Value::from(0.0));
    plan.constraints.push(Constraint {
        id: rig_plan_id(id)?,
        constraint_type: ConstraintType::Ik,
        name: name.to_owned(),
        target: Some(target),
        subtarget: Some(rig_plan_id(end_bone)?),
        owner_bone: None,
        influence: 1.0,
        enabled: true,
        params,
        inverse_matrix: None,
        inverse_frame: None,
    });
    Ok(())
}

fn rig_plan_id(value: &str) -> Result<Id> {
    Id::new(value.to_owned())
        .map_err(|_| PotError::new(ErrorCode::InternalError, "invalid generated rig identifier"))
}

fn pose_set(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "bone_id", "set"],
        &["target", "bone_id", "set"],
    )?;
    let node_id = target_node(engine, operation)?;
    let bone_id = read_id(engine, operation, "bone_id")?;
    let armature = armature(engine, &node_id)?;
    if !armature.bones.contains_key(&bone_id) {
        return Err(missing(engine, "bone", &bone_id, "bone_id"));
    }
    let set = read_set(engine, operation)?;
    reject_unknown_set_fields(engine, set, &["translation", "rotation", "scale"])?;
    let old = engine
        .doc
        .nodes
        .get(&node_id)
        .and_then(|node| node.pose.get(&bone_id))
        .copied()
        .unwrap_or_default();
    let mut next = old;
    if let Some(value) = set.get("translation") {
        next.translation = read_vec3(engine, value, "set/translation")?;
    }
    if let Some(value) = set.get("rotation") {
        let rotation = read_vec4(engine, value, "set/rotation")?;
        next.rotation = normalize_rotation(rotation).map_err(|_| {
            invalid(
                engine,
                "rotation must be a finite non-zero quaternion",
                "set/rotation",
            )
        })?;
    }
    if let Some(value) = set.get("scale") {
        next.scale = read_vec3(engine, value, "set/scale")?;
    }
    if next == old
        && engine
            .doc
            .nodes
            .get(&node_id)
            .is_some_and(|node| node.pose.contains_key(&bone_id))
    {
        return Ok(false);
    }
    let node = engine
        .doc
        .nodes
        .get_mut(&node_id)
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "armature node disappeared"))?;
    node.pose.insert(bone_id, next);
    engine.mark("nodes", &node_id, ChangeKind::Updated);
    Ok(true)
}

fn pose_reset(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "bone_id"],
        &["target", "bone_id"],
    )?;
    let node_id = target_node(engine, operation)?;
    let bone_id = read_id(engine, operation, "bone_id")?;
    if !armature(engine, &node_id)?.bones.contains_key(&bone_id) {
        return Err(missing(engine, "bone", &bone_id, "bone_id"));
    }
    let node = engine
        .doc
        .nodes
        .get_mut(&node_id)
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "armature node disappeared"))?;
    let changed = node.pose.remove(&bone_id).is_some();
    if changed {
        engine.mark("nodes", &node_id, ChangeKind::Updated);
    }
    Ok(changed)
}

fn object_solver_inverse_matrix(
    engine: &Engine<'_>,
    owner_id: &Id,
    constraint_id: Option<&Id>,
    target: Option<&Id>,
    params: &Map<String, Value>,
) -> Result<([f64; 16], f64)> {
    let mut doc = engine.doc.clone();
    if let Some(constraint_id) = constraint_id {
        let owner = doc.nodes.get_mut(owner_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "Object Solver owner disappeared")
        })?;
        owner
            .constraints
            .retain(|constraint| &constraint.id != constraint_id);
    }
    let scene = doc.scenes.get(&doc.active_scene).ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "Object Solver active scene disappeared",
        )
    })?;
    let frame = scene.frame_current;
    let drivers = crate::eval::rig::evaluate_drivers(&doc, frame)?;
    let worlds = crate::eval::rig::evaluate_world_matrices_with_drivers(&doc, frame, &drivers)?;
    let camera_id = target.ok_or_else(|| {
        invalid(
            engine,
            "Object Solver camera target is required",
            "constraint_target",
        )
    })?;
    let camera_world = worlds.get(camera_id).copied().ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            "Object Solver camera transform is unavailable",
            &pointer(engine, "constraint_target"),
        )
    })?;
    let clip = if params
        .get("use_active_clip")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        scene
            .active_clip
            .as_ref()
            .and_then(|clip_id| doc.movie_clips.get(clip_id))
            .ok_or_else(|| {
                invalid(
                    engine,
                    "active scene clip was not found",
                    "params/use_active_clip",
                )
            })?
    } else {
        let clip_text = params
            .get("clip")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid(engine, "Object Solver clip is required", "params/clip"))?;
        let clip_id = Id::new(clip_text.to_owned())
            .map_err(|_| invalid(engine, "Object Solver clip ID is invalid", "params/clip"))?;
        doc.movie_clips.get(&clip_id).ok_or_else(|| {
            engine.error(
                ErrorCode::TargetNotFound,
                "Object Solver clip was not found",
                &pointer(engine, "params/clip"),
            )
        })?
    };
    let object_text = params
        .get("object")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(engine, "Object Solver object is required", "params/object"))?;
    let object = clip
        .tracking
        .objects
        .iter()
        .find(|object| object.id == object_text)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::TargetNotFound,
                "Object Solver tracking object was not found",
                &pointer(engine, "params/object"),
            )
        })?;
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
    let (before_frame, before_matrix) = before.or(after).ok_or_else(|| {
        invalid(
            engine,
            "Object Solver has no reconstructed poses",
            "params/object",
        )
    })?;
    let (after_frame, after_matrix) = after.or(before).ok_or_else(|| {
        invalid(
            engine,
            "Object Solver has no reconstructed poses",
            "params/object",
        )
    })?;
    let object_camera_matrix = if crate::float::equal_f64(before_frame, after_frame) {
        before_matrix
    } else {
        let factor = ((frame - before_frame) / (after_frame - before_frame)).clamp(0.0, 1.0);
        let (before_scale, before_rotation, before_translation) =
            before_matrix.to_scale_rotation_translation();
        let (after_scale, after_rotation, after_translation) =
            after_matrix.to_scale_rotation_translation();
        DMat4::from_scale_rotation_translation(
            before_scale.lerp(after_scale, factor),
            before_rotation.slerp(after_rotation, factor),
            before_translation.lerp(after_translation, factor),
        )
    };
    let object_to_camera = object_camera_matrix.inverse();
    let parent = camera_world * object_to_camera;
    let inverse = parent.inverse().to_cols_array();
    if inverse.iter().all(|value| value.is_finite()) {
        Ok((inverse, frame))
    } else {
        Err(invalid(
            engine,
            "Object Solver inverse matrix is singular",
            "params/object",
        ))
    }
}

fn constraint_create(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &[
            "op",
            "target",
            "id",
            "type",
            "name",
            "constraint_target",
            "subtarget",
            "owner_bone",
            "influence",
            "enabled",
            "params",
        ],
        &["target", "id", "type"],
    )?;
    let owner_id = target_node(engine, operation)?;
    let id = read_id(engine, operation, "id")?;
    let kind = constraint_type(
        engine,
        operation
            .get("type")
            .ok_or_else(|| invalid(engine, "type is required", "type"))?,
        "type",
    )?;
    let name = operation
        .get("name")
        .map(|value| string_value(engine, value, "name"))
        .transpose()?
        .unwrap_or_else(|| constraint_type_name(kind).to_owned());
    let mut target = optional_id(
        engine,
        operation.get("constraint_target"),
        "constraint_target",
    )?;
    let subtarget = optional_id(engine, operation.get("subtarget"), "subtarget")?;
    let owner_bone = optional_id(engine, operation.get("owner_bone"), "owner_bone")?;
    let influence = operation.get("influence").map_or(Ok(1.0), |value| {
        ranged_number(engine, value, 0.0, 1.0, "influence")
    })?;
    let enabled = read_bool(engine, operation, "enabled", true)?;
    let mut params = read_params(engine, operation.get("params"), "params")?;
    apply_constraint_defaults(kind, &mut params);
    validate_constraint_parameter_map(engine, kind, &params, "/params")?;
    if let Some(key) = constraint_target_param(kind)
        && params.contains_key(key)
    {
        let parameter_target = optional_id(engine, params.get(key), &format!("params/{key}"))?;
        if target.is_some() && target != parameter_target {
            return Err(invalid(
                engine,
                "constraint_target and RNA target reference must agree",
                &format!("params/{key}"),
            ));
        }
        target = parameter_target;
    }
    validate_constraint_reference(
        engine,
        &owner_id,
        kind,
        target.as_ref(),
        subtarget.as_ref(),
        owner_bone.as_ref(),
        "constraint_target",
        "subtarget",
    )?;
    validate_owner_bone_reference(engine, &owner_id, owner_bone.as_ref(), "owner_bone")?;
    let mut inverse_matrix = None;
    let mut inverse_frame = None;
    if kind == ConstraintType::ObjectSolver
        && params
            .get("set_inverse_pending")
            .map(|value| bool_value(engine, value, "params/set_inverse_pending"))
            .transpose()?
            .unwrap_or(false)
    {
        let (matrix, frame) =
            object_solver_inverse_matrix(engine, &owner_id, None, target.as_ref(), &params)?;
        inverse_matrix = Some(matrix);
        inverse_frame = Some(frame);
        params.insert("set_inverse_pending".to_owned(), Value::Bool(false));
        validate_constraint_params(engine, kind, &params, "params")?;
    }
    if engine.doc.nodes.get(&owner_id).is_some_and(|node| {
        node.constraints
            .iter()
            .any(|constraint| constraint.id == id)
    }) {
        return Err(engine.error(
            ErrorCode::IdExists,
            format!("constraint ID `{id}` already exists on node `{owner_id}`"),
            &pointer(engine, "id"),
        ));
    }
    let path_target = target.clone();
    let node =
        engine.doc.nodes.get_mut(&owner_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "constraint owner disappeared")
        })?;
    node.constraints.push(Constraint {
        id,
        constraint_type: kind,
        name,
        target,
        subtarget,
        owner_bone,
        influence,
        enabled,
        params,
        inverse_matrix,
        inverse_frame,
    });
    engine.mark("nodes", &owner_id, ChangeKind::Updated);
    enable_constraint_target_path(engine, kind, path_target.as_ref());
    Ok(true)
}

fn enable_constraint_target_path(
    engine: &mut Engine<'_>,
    kind: ConstraintType,
    target: Option<&Id>,
) {
    if !matches!(
        kind,
        ConstraintType::ClampTo | ConstraintType::FollowPath | ConstraintType::SplineIk
    ) {
        return;
    }
    let Some(data_id) = target
        .and_then(|target| engine.doc.nodes.get(target))
        .and_then(|node| node.data.as_ref())
        .cloned()
    else {
        return;
    };
    let changed = engine
        .doc
        .data_blocks
        .get_mut(&data_id)
        .and_then(|data| data.curve.as_mut())
        .is_some_and(|curve| {
            if curve.use_path {
                false
            } else {
                curve.use_path = true;
                true
            }
        });
    if changed {
        engine.mark("data_blocks", &data_id, ChangeKind::Updated);
    }
}
fn constraint_update(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "id", "set"],
        &["target", "id", "set"],
    )?;
    let owner_id = target_node(engine, operation)?;
    let id = read_id(engine, operation, "id")?;
    let set = read_set(engine, operation)?;
    reject_unknown_set_fields(
        engine,
        set,
        &[
            "type",
            "name",
            "target",
            "subtarget",
            "influence",
            "owner_bone",
            "enabled",
            "params",
            "set_inverse",
            "clear_inverse",
        ],
    )?;
    let old = engine
        .doc
        .nodes
        .get(&owner_id)
        .and_then(|node| {
            node.constraints
                .iter()
                .find(|constraint| constraint.id == id)
        })
        .cloned()
        .ok_or_else(|| missing(engine, "constraint", &id, "id"))?;
    let mut next = old.clone();
    if let Some(value) = set.get("type") {
        next.constraint_type = constraint_type(engine, value, "set/type")?;
    }
    if let Some(value) = set.get("name") {
        next.name = string_value(engine, value, "set/name")?;
    }
    if let Some(value) = set.get("target") {
        next.target = optional_id(engine, Some(value), "set/target")?;
    }
    if let Some(value) = set.get("subtarget") {
        next.subtarget = optional_id(engine, Some(value), "set/subtarget")?;
    }
    if let Some(value) = set.get("owner_bone") {
        next.owner_bone = optional_id(engine, Some(value), "set/owner_bone")?;
    }
    if let Some(value) = set.get("influence") {
        next.influence = ranged_number(engine, value, 0.0, 1.0, "set/influence")?;
    }
    if let Some(value) = set.get("enabled") {
        next.enabled = bool_value(engine, value, "set/enabled")?;
    }
    if let Some(value) = set.get("params") {
        next.params = object_value(engine, value, "set/params")?;
    }
    apply_constraint_defaults(next.constraint_type, &mut next.params);
    if set.contains_key("params") || set.contains_key("type") {
        validate_constraint_parameter_map(
            engine,
            next.constraint_type,
            &next.params,
            "/set/params",
        )?;
    }
    if let Some(key) = constraint_target_param(next.constraint_type) {
        if set.contains_key("target") && !set.contains_key("params") {
            let value = next
                .target
                .as_ref()
                .map_or(Value::Null, |target| Value::String(target.to_string()));
            next.params.insert(key.to_owned(), value);
        }
        if next.params.contains_key(key) {
            let parameter_target =
                optional_id(engine, next.params.get(key), &format!("set/params/{key}"))?;
            if set.contains_key("target")
                && set.contains_key("params")
                && parameter_target != next.target
            {
                return Err(invalid(
                    engine,
                    "target and RNA target reference must agree",
                    &format!("set/params/{key}"),
                ));
            }
            next.target = parameter_target;
        }
    }
    let set_inverse = set
        .get("set_inverse")
        .map(|value| bool_value(engine, value, "set/set_inverse"))
        .transpose()?
        .unwrap_or_else(|| {
            next.params
                .get("set_inverse_pending")
                .and_then(Value::as_bool)
                .unwrap_or(false)
        });
    let clear_inverse = set
        .get("clear_inverse")
        .map(|value| bool_value(engine, value, "set/clear_inverse"))
        .transpose()?
        .unwrap_or(false);
    if set_inverse && clear_inverse {
        return Err(invalid(
            engine,
            "set_inverse and clear_inverse cannot both be true",
            "set",
        ));
    }
    if next.constraint_type == ConstraintType::ObjectSolver && clear_inverse {
        next.inverse_matrix = None;
        next.inverse_frame = None;
        next.params
            .insert("set_inverse_pending".to_owned(), Value::Bool(false));
    } else if next.constraint_type == ConstraintType::ObjectSolver && set_inverse {
        let (matrix, frame) = object_solver_inverse_matrix(
            engine,
            &owner_id,
            Some(&id),
            next.target.as_ref(),
            &next.params,
        )?;
        next.inverse_matrix = Some(matrix);
        next.inverse_frame = Some(frame);
        next.params
            .insert("set_inverse_pending".to_owned(), Value::Bool(false));
    }
    if (set_inverse || clear_inverse) && next.constraint_type != ConstraintType::ObjectSolver {
        return Err(invalid(
            engine,
            "set_inverse and clear_inverse are only valid for Object Solver constraints",
            "set",
        ));
    }
    validate_constraint_reference(
        engine,
        &owner_id,
        next.constraint_type,
        next.target.as_ref(),
        next.subtarget.as_ref(),
        next.owner_bone.as_ref(),
        "set/target",
        "set/subtarget",
    )?;
    validate_owner_bone_reference(
        engine,
        &owner_id,
        next.owner_bone.as_ref(),
        "set/owner_bone",
    )?;
    if next == old {
        return Ok(false);
    }
    let path_kind = next.constraint_type;
    let path_target = next.target.clone();
    let node =
        engine.doc.nodes.get_mut(&owner_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "constraint owner disappeared")
        })?;
    let constraint = node
        .constraints
        .iter_mut()
        .find(|constraint| constraint.id == id)
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "constraint disappeared"))?;
    *constraint = next;
    engine.mark("nodes", &owner_id, ChangeKind::Updated);
    enable_constraint_target_path(engine, path_kind, path_target.as_ref());
    Ok(true)
}

fn constraint_delete(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "id"],
        &["target", "id"],
    )?;
    let owner_id = target_node(engine, operation)?;
    let id = read_id(engine, operation, "id")?;
    let (before, after) = {
        let node = engine.doc.nodes.get_mut(&owner_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "constraint owner disappeared")
        })?;
        let before = node.constraints.len();
        node.constraints.retain(|constraint| constraint.id != id);
        (before, node.constraints.len())
    };
    if before == after {
        return Err(missing(engine, "constraint", &id, "id"));
    }
    engine.mark("nodes", &owner_id, ChangeKind::Updated);
    Ok(true)
}

fn constraint_reorder(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "id", "index"],
        &["target", "id", "index"],
    )?;
    let owner_id = target_node(engine, operation)?;
    let id = read_id(engine, operation, "id")?;
    let index = usize_value(
        engine,
        operation
            .get("index")
            .ok_or_else(|| invalid(engine, "index is required", "index"))?,
        "index",
    )?;
    let (current, length) = engine
        .doc
        .nodes
        .get(&owner_id)
        .map(|node| {
            (
                node.constraints
                    .iter()
                    .position(|constraint| constraint.id == id),
                node.constraints.len(),
            )
        })
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "constraint owner disappeared"))?;
    let current = current.ok_or_else(|| missing(engine, "constraint", &id, "id"))?;
    if index >= length {
        return Err(invalid(
            engine,
            "constraint index is outside the stack",
            "index",
        ));
    }
    if index == current {
        return Ok(false);
    }
    let node =
        engine.doc.nodes.get_mut(&owner_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "constraint owner disappeared")
        })?;
    let constraint = node.constraints.remove(current);
    node.constraints.insert(index, constraint);
    engine.mark("nodes", &owner_id, ChangeKind::Updated);
    Ok(true)
}

fn shape_key_create(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &[
            "op",
            "target",
            "id",
            "name",
            "positions",
            "value",
            "slider_min",
            "slider_max",
            "relative_key",
            "vertex_group",
            "frame",
        ],
        &["target", "id", "name", "positions"],
    )?;
    let node_id = target_node(engine, operation)?;
    let data_id = mesh_data_id(engine, &node_id)?;
    let id = read_id(engine, operation, "id")?;
    let name = read_string(engine, operation, "name")?;
    let data =
        engine.doc.data_blocks.get(&data_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "mesh data block disappeared")
        })?;
    let mesh = data
        .mesh
        .as_ref()
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "mesh disappeared"))?;
    let existing = data.shape_keys.as_ref();
    if existing.is_some_and(|keys| keys.keys.contains_key(&id)) {
        return Err(engine.error(
            ErrorCode::IdExists,
            format!("shape key ID `{id}` already exists"),
            &pointer(engine, "id"),
        ));
    }
    let first_key = existing.is_none_or(|keys| keys.keys.is_empty());
    let basis = if first_key {
        mesh.vertices
            .iter()
            .map(|vertex| (vertex.id, vertex.co.to_array()))
            .collect()
    } else {
        BTreeMap::new()
    };
    let positions = parse_positions(
        engine,
        operation
            .get("positions")
            .ok_or_else(|| invalid(engine, "positions is required", "positions"))?,
        &mesh.vertices,
        "positions",
    )?;
    let frame = operation
        .get("frame")
        .map(|value| finite_value(engine, value, "frame"))
        .transpose()?
        .unwrap_or(0.0);
    let value = operation
        .get("value")
        .map_or(Ok(0.0), |value| finite_value(engine, value, "value"))?;
    let slider_min = operation
        .get("slider_min")
        .map_or(Ok(0.0), |value| finite_value(engine, value, "slider_min"))?;
    let slider_max = operation
        .get("slider_max")
        .map_or(Ok(1.0), |value| finite_value(engine, value, "slider_max"))?;
    validate_shape_value(engine, value, slider_min, slider_max, "value")?;
    let relative_key = optional_id(engine, operation.get("relative_key"), "relative_key")?;
    let vertex_group = optional_id(engine, operation.get("vertex_group"), "vertex_group")?;
    validate_shape_references(
        engine,
        existing,
        data,
        &id,
        relative_key.as_ref(),
        vertex_group.as_ref(),
        "relative_key",
        "vertex_group",
    )?;
    let key = ShapeKey {
        id: id.clone(),
        name,
        value,
        mute: false,
        slider_min,
        slider_max,
        relative_key,
        vertex_group,
        frame,
        positions,
    };
    let shape_keys = engine
        .doc
        .data_blocks
        .get_mut(&data_id)
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "mesh data block disappeared"))?
        .shape_keys
        .get_or_insert_with(ShapeKeyData::default);
    if first_key {
        shape_keys.basis = basis;
    }
    if operation.contains_key("frame") {
        shape_keys.absolute = true;
    }
    shape_keys.keys.insert(id, key);
    engine.mark("data_blocks", &data_id, ChangeKind::Updated);
    Ok(true)
}

fn shape_key_update(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "id", "set"],
        &["target", "id", "set"],
    )?;
    let node_id = target_node(engine, operation)?;
    let data_id = mesh_data_id(engine, &node_id)?;
    let id = read_id(engine, operation, "id")?;
    let set = read_set(engine, operation)?;
    reject_unknown_set_fields(
        engine,
        set,
        &[
            "name",
            "positions",
            "value",
            "slider_min",
            "slider_max",
            "relative_key",
            "vertex_group",
            "frame",
            "evaluation_time",
            "absolute",
        ],
    )?;
    let data =
        engine.doc.data_blocks.get(&data_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "mesh data block disappeared")
        })?;
    let mesh = data
        .mesh
        .as_ref()
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "mesh disappeared"))?;
    let shape_data = data
        .shape_keys
        .as_ref()
        .ok_or_else(|| missing(engine, "shape key", &id, "id"))?;
    let old = shape_data
        .keys
        .get(&id)
        .ok_or_else(|| missing(engine, "shape key", &id, "id"))?;
    let frame = set
        .get("frame")
        .map(|value| finite_value(engine, value, "set/frame"))
        .transpose()?
        .unwrap_or(old.frame);
    let evaluation_time = set
        .get("evaluation_time")
        .map(|value| finite_value(engine, value, "set/evaluation_time"))
        .transpose()?
        .unwrap_or(shape_data.evaluation_time);
    let absolute = set
        .get("absolute")
        .map(|value| bool_value(engine, value, "set/absolute"))
        .transpose()?
        .unwrap_or(shape_data.absolute || set.contains_key("frame"));
    let mut name = None;
    if let Some(value) = set.get("name") {
        name = Some(string_value(engine, value, "set/name")?);
    }
    let positions = set
        .get("positions")
        .map(|value| parse_positions(engine, value, &mesh.vertices, "set/positions"))
        .transpose()?;
    let value = set
        .get("value")
        .map(|value| finite_value(engine, value, "set/value"))
        .transpose()?
        .unwrap_or(old.value);
    let slider_min = set
        .get("slider_min")
        .map(|value| finite_value(engine, value, "set/slider_min"))
        .transpose()?
        .unwrap_or(old.slider_min);
    let slider_max = set
        .get("slider_max")
        .map(|value| finite_value(engine, value, "set/slider_max"))
        .transpose()?
        .unwrap_or(old.slider_max);
    validate_shape_value(engine, value, slider_min, slider_max, "set/value")?;
    let relative_key = if let Some(raw) = set.get("relative_key") {
        optional_id(engine, Some(raw), "set/relative_key")?
    } else {
        old.relative_key.clone()
    };
    let vertex_group = if let Some(raw) = set.get("vertex_group") {
        optional_id(engine, Some(raw), "set/vertex_group")?
    } else {
        old.vertex_group.clone()
    };
    validate_shape_references(
        engine,
        Some(shape_data),
        data,
        &id,
        relative_key.as_ref(),
        vertex_group.as_ref(),
        "set/relative_key",
        "set/vertex_group",
    )?;
    let changed = name.as_ref().is_some_and(|next| next != &old.name)
        || positions
            .as_ref()
            .is_some_and(|next| next != &old.positions)
        || !crate::float::equal_f64(value, old.value)
        || !crate::float::equal_f64(slider_min, old.slider_min)
        || !crate::float::equal_f64(slider_max, old.slider_max)
        || relative_key != old.relative_key
        || vertex_group != old.vertex_group
        || !crate::float::equal_f64(frame, old.frame)
        || absolute != shape_data.absolute
        || !crate::float::equal_f64(evaluation_time, shape_data.evaluation_time);
    if !changed {
        return Ok(false);
    }
    let data =
        engine.doc.data_blocks.get_mut(&data_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "mesh data block disappeared")
        })?;
    let key = data
        .shape_keys
        .as_mut()
        .and_then(|shape| shape.keys.get_mut(&id))
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "shape key disappeared"))?;
    if let Some(name) = name {
        key.name = name;
    }
    if let Some(positions) = positions {
        key.positions = positions;
    }
    key.value = value;
    key.slider_min = slider_min;
    key.frame = frame;
    key.slider_max = slider_max;
    key.relative_key = relative_key;
    key.vertex_group = vertex_group;
    let shape = data
        .shape_keys
        .as_mut()
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "shape-key data disappeared"))?;
    shape.absolute = absolute;
    shape.evaluation_time = evaluation_time;
    engine.mark("data_blocks", &data_id, ChangeKind::Updated);
    Ok(true)
}

fn shape_key_delete(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "id"],
        &["target", "id"],
    )?;
    let node_id = target_node(engine, operation)?;
    let data_id = mesh_data_id(engine, &node_id)?;
    let id = read_id(engine, operation, "id")?;
    let key_exists = engine
        .doc
        .data_blocks
        .get(&data_id)
        .and_then(|data| data.shape_keys.as_ref())
        .is_some_and(|shape| shape.keys.contains_key(&id));
    if !key_exists {
        return Err(missing(engine, "shape key", &id, "id"));
    }
    let data =
        engine.doc.data_blocks.get_mut(&data_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "mesh data block disappeared")
        })?;
    let remove_shape_data = {
        let shape = data
            .shape_keys
            .as_mut()
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "shape-key data disappeared"))?;
        if shape.keys.remove(&id).is_none() {
            return Err(PotError::new(
                ErrorCode::InternalError,
                "shape key disappeared",
            ));
        }
        if shape.keys.is_empty() {
            true
        } else {
            for key in shape.keys.values_mut() {
                if key.relative_key.as_ref() == Some(&id) {
                    key.relative_key = None;
                }
            }
            false
        }
    };
    if remove_shape_data {
        data.shape_keys = None;
    }
    engine.mark("data_blocks", &data_id, ChangeKind::Updated);
    Ok(true)
}

fn vertex_group_create(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "id", "name"],
        &["target", "id", "name"],
    )?;
    let node_id = target_node(engine, operation)?;
    let data_id = mesh_data_id(engine, &node_id)?;
    let id = read_id(engine, operation, "id")?;
    let name = read_string(engine, operation, "name")?;
    if engine
        .doc
        .data_blocks
        .get(&data_id)
        .is_some_and(|data| data.vertex_groups.iter().any(|group| group.id == id))
    {
        return Err(engine.error(
            ErrorCode::IdExists,
            format!("vertex group ID `{id}` already exists"),
            &pointer(engine, "id"),
        ));
    }
    let data =
        engine.doc.data_blocks.get_mut(&data_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "mesh data block disappeared")
        })?;
    data.vertex_groups.push(VertexGroup { id, name });
    engine.mark("data_blocks", &data_id, ChangeKind::Updated);
    Ok(true)
}

fn vertex_group_assign(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "group_id", "weights"],
        &["target", "group_id", "weights"],
    )?;
    let node_id = target_node(engine, operation)?;
    let data_id = mesh_data_id(engine, &node_id)?;
    let group_id = read_id(engine, operation, "group_id")?;
    let data =
        engine.doc.data_blocks.get(&data_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "mesh data block disappeared")
        })?;
    if !data.vertex_groups.iter().any(|group| group.id == group_id) {
        return Err(missing(engine, "vertex group", &group_id, "group_id"));
    }
    let mesh = data
        .mesh
        .as_ref()
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "mesh disappeared"))?;
    let known_vertices = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.id)
        .collect::<BTreeSet<_>>();
    let assignments = parse_weights(
        engine,
        operation
            .get("weights")
            .ok_or_else(|| invalid(engine, "weights is required", "weights"))?,
        &known_vertices,
    )?;
    let mut changed = false;
    let data =
        engine.doc.data_blocks.get_mut(&data_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "mesh data block disappeared")
        })?;
    for (vertex_id, weight) in assignments {
        let weights = data.vertex_weights.entry(vertex_id).or_default();
        if weights.get(&group_id) != Some(&weight) {
            weights.insert(group_id.clone(), weight);
            changed = true;
        }
    }
    if changed {
        engine.mark("data_blocks", &data_id, ChangeKind::Updated);
    }
    Ok(changed)
}

fn vertex_group_remove(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "group_id", "vertex_ids"],
        &["target", "group_id", "vertex_ids"],
    )?;
    let node_id = target_node(engine, operation)?;
    let data_id = mesh_data_id(engine, &node_id)?;
    let group_id = read_id(engine, operation, "group_id")?;
    let data =
        engine.doc.data_blocks.get(&data_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "mesh data block disappeared")
        })?;
    if !data.vertex_groups.iter().any(|group| group.id == group_id) {
        return Err(missing(engine, "vertex group", &group_id, "group_id"));
    }
    let mesh = data
        .mesh
        .as_ref()
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "mesh disappeared"))?;
    let vertices = parse_vertex_ids(
        engine,
        operation
            .get("vertex_ids")
            .ok_or_else(|| invalid(engine, "vertex_ids is required", "vertex_ids"))?,
        &mesh.vertices,
        "vertex_ids",
    )?;
    let data =
        engine.doc.data_blocks.get_mut(&data_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "mesh data block disappeared")
        })?;
    let mut changed = false;
    for vertex_id in vertices {
        let remove_row = if let Some(weights) = data.vertex_weights.get_mut(&vertex_id) {
            changed |= weights.remove(&group_id).is_some();
            weights.is_empty()
        } else {
            false
        };
        if remove_row {
            data.vertex_weights.remove(&vertex_id);
        }
    }
    if changed {
        engine.mark("data_blocks", &data_id, ChangeKind::Updated);
    }
    Ok(changed)
}

fn driver_create(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &[
            "op",
            "target",
            "id",
            "path",
            "index",
            "type",
            "variables",
            "expression",
        ],
        &["target", "id", "path", "index", "type", "variables"],
    )?;
    let node_id = target_node(engine, operation)?;
    let id = read_id(engine, operation, "id")?;
    let path = nonempty_string(
        engine,
        operation
            .get("path")
            .ok_or_else(|| invalid(engine, "path is required", "path"))?,
        "path",
    )?;
    let index = u32_value(
        engine,
        operation
            .get("index")
            .ok_or_else(|| invalid(engine, "index is required", "index"))?,
        "index",
    )?;
    let driver_type = driver_type(
        engine,
        operation
            .get("type")
            .ok_or_else(|| invalid(engine, "type is required", "type"))?,
        "type",
    )?;
    let variables = parse_driver_variables(
        engine,
        operation
            .get("variables")
            .ok_or_else(|| invalid(engine, "variables is required", "variables"))?,
    )?;
    let expression = optional_string(engine, operation.get("expression"), "expression")?;
    validate_driver(
        engine,
        driver_type,
        &variables,
        expression.as_deref(),
        "expression",
    )?;
    if engine
        .doc
        .nodes
        .get(&node_id)
        .is_some_and(|node| node.drivers.iter().any(|driver| driver.id == id))
    {
        return Err(engine.error(
            ErrorCode::IdExists,
            format!("driver ID `{id}` already exists on node `{node_id}`"),
            &pointer(engine, "id"),
        ));
    }
    let node = engine
        .doc
        .nodes
        .get_mut(&node_id)
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "driver owner disappeared"))?;
    node.drivers.push(Driver {
        id,
        path,
        index,
        driver_type,
        variables,
        expression,
    });
    engine.mark("nodes", &node_id, ChangeKind::Updated);
    Ok(true)
}

fn driver_update(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "id", "set"],
        &["target", "id", "set"],
    )?;
    let node_id = target_node(engine, operation)?;
    let id = read_id(engine, operation, "id")?;
    let set = read_set(engine, operation)?;
    reject_unknown_set_fields(
        engine,
        set,
        &["path", "index", "type", "variables", "expression"],
    )?;
    let old = engine
        .doc
        .nodes
        .get(&node_id)
        .and_then(|node| node.drivers.iter().find(|driver| driver.id == id))
        .cloned()
        .ok_or_else(|| missing(engine, "driver", &id, "id"))?;
    let path = set
        .get("path")
        .map(|value| nonempty_string(engine, value, "set/path"))
        .transpose()?
        .unwrap_or_else(|| old.path.clone());
    let index = set
        .get("index")
        .map(|value| u32_value(engine, value, "set/index"))
        .transpose()?
        .unwrap_or(old.index);
    let driver_type = set
        .get("type")
        .map(|value| driver_type(engine, value, "set/type"))
        .transpose()?
        .unwrap_or(old.driver_type);
    let variables = set
        .get("variables")
        .map(|value| parse_driver_variables(engine, value))
        .transpose()?
        .unwrap_or_else(|| old.variables.clone());
    let expression = if let Some(value) = set.get("expression") {
        optional_string(engine, Some(value), "set/expression")?
    } else {
        old.expression.clone()
    };
    validate_driver(
        engine,
        driver_type,
        &variables,
        expression.as_deref(),
        "set/expression",
    )?;
    let next = Driver {
        id: id.clone(),
        path,
        index,
        driver_type,
        variables,
        expression,
    };
    if next == old {
        return Ok(false);
    }
    let node = engine
        .doc
        .nodes
        .get_mut(&node_id)
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "driver owner disappeared"))?;
    let driver = node
        .drivers
        .iter_mut()
        .find(|driver| driver.id == id)
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "driver disappeared"))?;
    *driver = next;
    engine.mark("nodes", &node_id, ChangeKind::Updated);
    Ok(true)
}

fn driver_delete(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "id"],
        &["target", "id"],
    )?;
    let node_id = target_node(engine, operation)?;
    let id = read_id(engine, operation, "id")?;
    let node = engine
        .doc
        .nodes
        .get_mut(&node_id)
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "driver owner disappeared"))?;
    let before = node.drivers.len();
    node.drivers.retain(|driver| driver.id != id);
    if before == node.drivers.len() {
        return Err(missing(engine, "driver", &id, "id"));
    }
    engine.mark("nodes", &node_id, ChangeKind::Updated);
    Ok(true)
}

fn target_node(engine: &Engine<'_>, operation: &Map<String, Value>) -> Result<Id> {
    let target = operation
        .get("target")
        .ok_or_else(|| invalid(engine, "target is required", "target"))?;
    let targets = resolve_node_targets(engine, target, false)?;
    match targets.as_slice() {
        [id] => Ok(id.clone()),
        [] => Err(engine.error(
            ErrorCode::TargetNotFound,
            "target selected no nodes",
            &pointer(engine, "target"),
        )),
        _ => Err(engine.error(
            ErrorCode::AmbiguousTarget,
            "rig operations require exactly one target node",
            &pointer(engine, "target"),
        )),
    }
}

fn data_id(engine: &Engine<'_>, node_id: &Id) -> Result<Id> {
    let node = engine.doc.nodes.get(node_id).ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            format!("node `{node_id}` was not found"),
            &pointer(engine, "target"),
        )
    })?;
    node.data.clone().ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("node `{node_id}` has no data block"),
            &pointer(engine, "target"),
        )
    })
}

fn armature<'a>(engine: &'a Engine<'_>, node_id: &Id) -> Result<&'a crate::model::ArmatureData> {
    let node = engine.doc.nodes.get(node_id).ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            format!("node `{node_id}` was not found"),
            &pointer(engine, "target"),
        )
    })?;
    if node.kind != "armature" {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "target node must be an armature",
            &pointer(engine, "target"),
        ));
    }
    let data_id = node.data.as_ref().ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "armature node has no armature data",
            &pointer(engine, "target"),
        )
    })?;
    let block = engine.doc.data_blocks.get(data_id).ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            format!("armature data `{data_id}` was not found"),
            &pointer(engine, "target"),
        )
    })?;
    if block.data_type != "armature" {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "target node does not reference armature data",
            &pointer(engine, "target"),
        ));
    }
    block.armature.as_ref().ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "target node does not reference armature data",
            &pointer(engine, "target"),
        )
    })
}
fn armature_mut<'a>(
    engine: &'a mut Engine<'_>,
    node_id: &Id,
) -> Result<&'a mut crate::model::ArmatureData> {
    let data_id = data_id(engine, node_id)?;
    armature(engine, node_id)?;
    let block = engine.doc.data_blocks.get_mut(&data_id).ok_or_else(|| {
        PotError::new(ErrorCode::InternalError, "armature data block disappeared")
    })?;
    block
        .armature
        .as_mut()
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "armature model disappeared"))
}

fn mesh_data_id(engine: &Engine<'_>, node_id: &Id) -> Result<Id> {
    let node = engine.doc.nodes.get(node_id).ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            format!("node `{node_id}` was not found"),
            &pointer(engine, "target"),
        )
    })?;
    if node.kind != "mesh" {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "target node must be a mesh",
            &pointer(engine, "target"),
        ));
    }
    let data_id = node.data.clone().ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "mesh node has no data block",
            &pointer(engine, "target"),
        )
    })?;
    let block = engine.doc.data_blocks.get(&data_id).ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            format!("mesh data `{data_id}` was not found"),
            &pointer(engine, "target"),
        )
    })?;
    if block.data_type != "mesh" || block.mesh.is_none() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "target node does not reference mesh data",
            &pointer(engine, "target"),
        ));
    }
    Ok(data_id)
}

fn validate_bone_parent(
    engine: &Engine<'_>,
    bones: &BTreeMap<Id, Bone>,
    bone_id: &Id,
    parent: Option<&Id>,
    use_connect: bool,
    head: [f64; 3],
) -> Result<()> {
    if let Some(parent_id) = parent {
        if parent_id == bone_id {
            return Err(invalid(engine, "bone cannot parent itself", "parent"));
        }
        let parent_bone = bones.get(parent_id).ok_or_else(|| {
            engine.error(
                ErrorCode::TargetNotFound,
                format!("parent bone `{parent_id}` was not found"),
                &pointer(engine, "parent"),
            )
        })?;
        let mut ancestor = Some(parent_id);
        let mut visited = BTreeSet::new();
        while let Some(current) = ancestor {
            if current == bone_id {
                return Err(invalid(
                    engine,
                    "bone parent would create a cycle",
                    "parent",
                ));
            }
            if !visited.insert(current.clone()) {
                return Err(invalid(
                    engine,
                    "existing bone parent chain contains a cycle",
                    "parent",
                ));
            }
            ancestor = bones.get(current).and_then(|bone| bone.parent.as_ref());
        }
        if use_connect && !same_point(head, parent_bone.tail) {
            return Err(invalid(
                engine,
                "connected bone head must match its parent tail",
                "head",
            ));
        }
    } else if use_connect {
        return Err(invalid(
            engine,
            "use_connect requires a parent bone",
            "use_connect",
        ));
    }
    Ok(())
}

fn validate_bone_segment(
    engine: &Engine<'_>,
    head: [f64; 3],
    tail: [f64; 3],
    field: &str,
) -> Result<()> {
    if head
        .iter()
        .chain(tail.iter())
        .any(|value| !value.is_finite())
    {
        return Err(invalid(
            engine,
            "bone endpoints must contain finite numbers",
            field,
        ));
    }
    let delta = DVec3::from_array(tail) - DVec3::from_array(head);
    if delta.length_squared() == 0.0 {
        return Err(invalid(engine, "bone head and tail must differ", field));
    }
    Ok(())
}

fn validate_owner_bone_reference(
    engine: &Engine<'_>,
    owner_id: &Id,
    owner_bone: Option<&Id>,
    field: &str,
) -> Result<()> {
    let Some(owner_bone) = owner_bone else {
        return Ok(());
    };
    let owner =
        engine.doc.nodes.get(owner_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "constraint owner disappeared")
        })?;
    let bones = owner
        .data
        .as_ref()
        .and_then(|data_id| engine.doc.data_blocks.get(data_id))
        .and_then(|data| data.armature.as_ref())
        .filter(|_| owner.kind == "armature")
        .ok_or_else(|| invalid(engine, "owner_bone requires an armature node", field))?;
    if !bones.bones.contains_key(owner_bone) {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            format!("constraint owner bone `{owner_bone}` was not found"),
            &pointer(engine, field),
        ));
    }
    Ok(())
}

fn validate_constraint_reference(
    engine: &Engine<'_>,
    owner_id: &Id,
    kind: ConstraintType,
    target: Option<&Id>,
    subtarget: Option<&Id>,
    owner_bone: Option<&Id>,
    target_field: &str,
    subtarget_field: &str,
) -> Result<()> {
    if let Some(target_id) = target
        && !engine.doc.nodes.contains_key(target_id)
    {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            format!("constraint target node `{target_id}` was not found"),
            &pointer(engine, target_field),
        ));
    }
    if matches!(
        kind,
        ConstraintType::Action
            | ConstraintType::Armature
            | ConstraintType::ClampTo
            | ConstraintType::CopyTransforms
            | ConstraintType::FollowPath
            | ConstraintType::FollowTrack
            | ConstraintType::GeometryAttribute
            | ConstraintType::LimitDistance
            | ConstraintType::ObjectSolver
            | ConstraintType::Ik
    ) && target.is_none()
    {
        return Err(invalid(
            engine,
            "constraint type requires a target",
            target_field,
        ));
    }
    if kind == ConstraintType::CameraSolver {
        let owner = engine.doc.nodes.get(owner_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "camera solver owner disappeared")
        })?;
        if owner.kind != "camera" {
            return Err(invalid(
                engine,
                "camera_solver requires a camera owner",
                target_field,
            ));
        }
    }
    if kind == ConstraintType::GeometryAttribute {
        let target_is_mesh = target
            .and_then(|target_id| engine.doc.nodes.get(target_id))
            .and_then(|node| node.data.as_ref())
            .and_then(|data_id| engine.doc.data_blocks.get(data_id))
            .is_some_and(|data| data.mesh.is_some());
        if !target_is_mesh {
            return Err(invalid(
                engine,
                "geometry_attribute requires a mesh target",
                target_field,
            ));
        }
    }
    if matches!(
        kind,
        ConstraintType::FollowTrack | ConstraintType::ObjectSolver
    ) && target
        .and_then(|target_id| engine.doc.nodes.get(target_id))
        .is_none_or(|node| node.kind != "camera")
    {
        return Err(invalid(
            engine,
            "tracking constraint requires a camera target",
            target_field,
        ));
    }
    if matches!(kind, ConstraintType::FollowPath | ConstraintType::ClampTo) {
        let curve_target = target
            .and_then(|target_id| engine.doc.nodes.get(target_id))
            .and_then(|node| node.data.as_ref())
            .and_then(|data_id| engine.doc.data_blocks.get(data_id))
            .is_some_and(|data| data.curve.is_some());
        if !curve_target {
            return Err(invalid(
                engine,
                "constraint type requires a curve target",
                target_field,
            ));
        }
    }
    if kind == ConstraintType::Armature {
        let target_is_armature = target
            .and_then(|target_id| engine.doc.nodes.get(target_id))
            .is_some_and(|node| node.kind == "armature");
        if !target_is_armature {
            return Err(invalid(
                engine,
                "armature constraint requires an armature target",
                target_field,
            ));
        }
        if subtarget.is_none() {
            return Err(invalid(
                engine,
                "armature constraint requires a target bone",
                subtarget_field,
            ));
        }
    }
    if let Some(bone_id) = subtarget
        && kind != ConstraintType::FollowTrack
    {
        let bone_owner = if matches!(kind, ConstraintType::Ik | ConstraintType::SplineIk)
            && owner_bone.is_none()
        {
            owner_id
        } else {
            target.ok_or_else(|| {
                invalid(
                    engine,
                    "subtarget requires a constraint target",
                    subtarget_field,
                )
            })?
        };
        let bone_node = engine.doc.nodes.get(bone_owner).ok_or_else(|| {
            engine.error(
                ErrorCode::TargetNotFound,
                format!("constraint bone owner `{bone_owner}` was not found"),
                &pointer(engine, subtarget_field),
            )
        })?;
        if bone_node.kind != "armature"
            || !armature(engine, bone_owner)?.bones.contains_key(bone_id)
        {
            return Err(engine.error(
                ErrorCode::TargetNotFound,
                format!("constraint target bone `{bone_id}` was not found"),
                &pointer(engine, subtarget_field),
            ));
        }
    }
    if kind == ConstraintType::Ik && owner_bone.is_none() && subtarget.is_none() {
        return Err(invalid(
            engine,
            "IK requires an owner bone or subtarget",
            subtarget_field,
        ));
    }
    if kind == ConstraintType::SplineIk {
        let owner = engine.doc.nodes.get(owner_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "spline IK owner disappeared")
        })?;
        let curve_target = target
            .and_then(|target_id| engine.doc.nodes.get(target_id))
            .and_then(|node| node.data.as_ref())
            .and_then(|data_id| engine.doc.data_blocks.get(data_id))
            .is_some_and(|data| data.curve.is_some());
        if owner.kind != "armature" || !curve_target {
            return Err(invalid(
                engine,
                "spline_ik requires an armature owner and a curve target",
                target_field,
            ));
        }
        if subtarget.is_none() && owner_bone.is_none() {
            return Err(invalid(
                engine,
                "spline_ik requires an owner bone or final-bone subtarget",
                subtarget_field,
            ));
        }
    }
    Ok(())
}

fn constraint_type(engine: &Engine<'_>, value: &Value, field: &str) -> Result<ConstraintType> {
    let value = string_value(engine, value, field)?;
    match value.as_str() {
        "copy_location" => Ok(ConstraintType::CopyLocation),
        "copy_rotation" => Ok(ConstraintType::CopyRotation),
        "copy_scale" => Ok(ConstraintType::CopyScale),
        "track_to" => Ok(ConstraintType::TrackTo),
        "damped_track" => Ok(ConstraintType::DampedTrack),
        "locked_track" => Ok(ConstraintType::LockedTrack),
        "stretch_to" => Ok(ConstraintType::StretchTo),
        "transformation" => Ok(ConstraintType::Transformation),
        "maintain_volume" => Ok(ConstraintType::MaintainVolume),
        "floor" => Ok(ConstraintType::Floor),
        "pivot" => Ok(ConstraintType::Pivot),
        "shrinkwrap" => Ok(ConstraintType::Shrinkwrap),
        "spline_ik" => Ok(ConstraintType::SplineIk),
        "limit_location" => Ok(ConstraintType::LimitLocation),
        "limit_rotation" => Ok(ConstraintType::LimitRotation),
        "limit_scale" => Ok(ConstraintType::LimitScale),
        "child_of" => Ok(ConstraintType::ChildOf),
        "action" => Ok(ConstraintType::Action),
        "armature" => Ok(ConstraintType::Armature),
        "camera_solver" => Ok(ConstraintType::CameraSolver),
        "clamp_to" => Ok(ConstraintType::ClampTo),
        "copy_transforms" => Ok(ConstraintType::CopyTransforms),
        "follow_path" => Ok(ConstraintType::FollowPath),
        "follow_track" => Ok(ConstraintType::FollowTrack),
        "geometry_attribute" => Ok(ConstraintType::GeometryAttribute),
        "limit_distance" => Ok(ConstraintType::LimitDistance),
        "object_solver" => Ok(ConstraintType::ObjectSolver),
        "transform_cache" => Ok(ConstraintType::TransformCache),
        "ik" => Ok(ConstraintType::Ik),
        _ => Err(invalid(engine, "unknown constraint type", field)),
    }
}

fn constraint_type_name(kind: ConstraintType) -> &'static str {
    match kind {
        ConstraintType::CopyLocation => "copy_location",
        ConstraintType::CopyRotation => "copy_rotation",
        ConstraintType::CopyScale => "copy_scale",
        ConstraintType::TrackTo => "track_to",
        ConstraintType::DampedTrack => "damped_track",
        ConstraintType::LockedTrack => "locked_track",
        ConstraintType::StretchTo => "stretch_to",
        ConstraintType::Transformation => "transformation",
        ConstraintType::MaintainVolume => "maintain_volume",
        ConstraintType::Floor => "floor",
        ConstraintType::Pivot => "pivot",
        ConstraintType::Shrinkwrap => "shrinkwrap",
        ConstraintType::SplineIk => "spline_ik",
        ConstraintType::LimitLocation => "limit_location",
        ConstraintType::LimitRotation => "limit_rotation",
        ConstraintType::LimitScale => "limit_scale",
        ConstraintType::ChildOf => "child_of",
        ConstraintType::Action => "action",
        ConstraintType::Armature => "armature",
        ConstraintType::CameraSolver => "camera_solver",
        ConstraintType::ClampTo => "clamp_to",
        ConstraintType::CopyTransforms => "copy_transforms",
        ConstraintType::FollowPath => "follow_path",
        ConstraintType::FollowTrack => "follow_track",
        ConstraintType::GeometryAttribute => "geometry_attribute",
        ConstraintType::LimitDistance => "limit_distance",
        ConstraintType::ObjectSolver => "object_solver",
        ConstraintType::TransformCache => "transform_cache",
        ConstraintType::Ik => "ik",
    }
}
fn constraint_target_param(kind: ConstraintType) -> Option<&'static str> {
    match kind {
        ConstraintType::Action | ConstraintType::GeometryAttribute => Some("target"),
        ConstraintType::FollowTrack | ConstraintType::ObjectSolver => Some("camera"),
        _ => None,
    }
}

fn apply_constraint_defaults(kind: ConstraintType, params: &mut Map<String, Value>) {
    let type_name = constraint_type_name(kind);
    let defaults: &[&str] = match kind {
        ConstraintType::TrackTo | ConstraintType::LockedTrack => &["track_axis", "up_axis"],
        ConstraintType::LimitDistance => &["distance"],
        ConstraintType::FollowPath => &["offset_factor"],
        _ => &[],
    };
    for name in defaults {
        if let Some(default) = params::default_value(ParameterFamily::Constraint, type_name, name) {
            params
                .entry((*name).to_owned())
                .or_insert_with(|| default.clone());
        }
    }
}
fn validate_constraint_parameter_map(
    engine: &Engine<'_>,
    kind: ConstraintType,
    params: &Map<String, Value>,
    base_pointer: &str,
) -> Result<()> {
    let type_name = constraint_type_name(kind);
    params::validate_params(ParameterFamily::Constraint, type_name, params, base_pointer).map_err(
        |error| {
            let pointer = operation_pointer(
                engine.operation_index,
                error.pointer.trim_start_matches('/'),
            );
            engine.error(ErrorCode::InvalidArgument, error.message, &pointer)
        },
    )?;
    if params::type_spec(ParameterFamily::Constraint, type_name)
        .and_then(|specification| specification.get("properties"))
        .and_then(Value::as_object)
        .is_some_and(|properties| properties.contains_key("resource"))
    {
        let pointer = operation_pointer(
            engine.operation_index,
            format!("{}/resource", base_pointer.trim_start_matches('/')).as_str(),
        );
        let resource_text = params
            .get("resource")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidArgument,
                    "resource must be a Potter resource ID",
                    &pointer,
                )
            })?;
        let resource_id = Id::new(resource_text.to_owned()).map_err(|_| {
            engine.error(ErrorCode::InvalidArgument, "invalid resource ID", &pointer)
        })?;
        if !engine.doc.resources.contains_key(&resource_id) {
            return Err(engine.error(
                ErrorCode::TargetNotFound,
                format!("resource `{resource_id}` was not found"),
                &pointer,
            ));
        }
    }
    Ok(())
}

#[expect(
    clippy::too_many_lines,
    reason = "constraint variants share one schema validation path"
)]
fn validate_constraint_params(
    engine: &Engine<'_>,
    kind: ConstraintType,
    params: &Map<String, Value>,
    field: &str,
) -> Result<()> {
    for (key, value) in params {
        validate_finite_json(engine, value, &format!("{field}/{}", escape_pointer(key)))?;
    }
    if kind == ConstraintType::TransformCache {
        let resource = params.get("resource").ok_or_else(|| {
            invalid(
                engine,
                "transform_cache requires a resource ID",
                &format!("{field}/resource"),
            )
        })?;
        let resource_text = string_value(engine, resource, &format!("{field}/resource"))?;
        let resource_id = Id::new(resource_text).map_err(|_| {
            invalid(
                engine,
                "transform_cache resource must be a valid ID",
                &format!("{field}/resource"),
            )
        })?;
        if !engine.doc.resources.contains_key(&resource_id) {
            return Err(engine.error(
                ErrorCode::TargetNotFound,
                "transform_cache resource was not found",
                &pointer(engine, &format!("{field}/resource")),
            ));
        }
        let object_path = params.get("object_path").ok_or_else(|| {
            invalid(
                engine,
                "transform_cache requires object_path",
                &format!("{field}/object_path"),
            )
        })?;
        if string_value(engine, object_path, &format!("{field}/object_path"))?.is_empty() {
            return Err(invalid(
                engine,
                "transform_cache object_path must not be empty",
                &format!("{field}/object_path"),
            ));
        }
        for key in ["frame_offset", "scale"] {
            if let Some(value) = params.get(key) {
                ranged_number(
                    engine,
                    value,
                    -f64::MAX,
                    f64::MAX,
                    &format!("{field}/{key}"),
                )?;
            }
        }
        if let Some(value) = params
            .get("override_frame")
            .filter(|value| !value.is_null())
        {
            ranged_number(
                engine,
                value,
                -f64::MAX,
                f64::MAX,
                &format!("{field}/override_frame"),
            )?;
        }
    }
    if kind == ConstraintType::LimitDistance
        && let Some(value) = params.get("distance")
    {
        ranged_number(engine, value, 0.0, f64::MAX, &format!("{field}/distance"))?;
    }
    if kind == ConstraintType::FollowPath
        && let Some(value) = params.get("offset_factor")
    {
        ranged_number(engine, value, 0.0, 1.0, &format!("{field}/offset_factor"))?;
    }
    if matches!(kind, ConstraintType::Ik | ConstraintType::SplineIk) {
        if let Some(value) = params.get("iterations") {
            let iterations = u32_value(engine, value, &format!("{field}/iterations"))?;
            if iterations == 0 {
                return Err(invalid(
                    engine,
                    "iterations must be greater than zero",
                    &format!("{field}/iterations"),
                ));
            }
        }
        if let Some(value) = params.get("pole_target") {
            let raw_id = string_value(engine, value, &format!("{field}/pole_target"))?;
            let pole_target = Id::new(raw_id).map_err(|_| {
                invalid(
                    engine,
                    "pole_target must be a valid node ID",
                    &format!("{field}/pole_target"),
                )
            })?;
            if !engine.doc.nodes.contains_key(&pole_target) {
                return Err(engine.error(
                    ErrorCode::TargetNotFound,
                    "IK pole target node was not found",
                    &pointer(engine, &format!("{field}/pole_target")),
                ));
            }
        }
    }
    if matches!(
        kind,
        ConstraintType::TrackTo
            | ConstraintType::DampedTrack
            | ConstraintType::LockedTrack
            | ConstraintType::StretchTo
    ) {
        for axis in ["track_axis", "up_axis", "lock_axis"] {
            if let Some(value) = params.get(axis) {
                let axis_value = string_value(engine, value, &format!("{field}/{axis}"))?;
                if !["x", "y", "z", "-x", "-y", "-z"].contains(&axis_value.as_str()) {
                    return Err(invalid(
                        engine,
                        "axis must be one of x, y, z, -x, -y, -z",
                        &format!("{field}/{axis}"),
                    ));
                }
            }
        }
    }
    if kind == ConstraintType::Transformation {
        for key in ["from", "to"] {
            let value = params.get(key).ok_or_else(|| {
                invalid(
                    engine,
                    "transformation constraint requires from and to channels",
                    &format!("{field}/{key}"),
                )
            })?;
            let channel = string_value(engine, value, &format!("{field}/{key}"))?;
            if ![
                "location_x",
                "location_y",
                "location_z",
                "rotation_x",
                "rotation_y",
                "rotation_z",
                "scale_x",
                "scale_y",
                "scale_z",
            ]
            .contains(&channel.as_str())
            {
                return Err(invalid(
                    engine,
                    "transformation channel is unsupported",
                    &format!("{field}/{key}"),
                ));
            }
        }
        for key in ["from_min", "from_max", "to_min", "to_max"] {
            if !params.contains_key(key) {
                return Err(invalid(
                    engine,
                    "transformation constraint requires explicit input and output ranges",
                    &format!("{field}/{key}"),
                ));
            }
            finite_value(
                engine,
                params.get(key).ok_or_else(|| {
                    invalid(engine, "range value is required", &format!("{field}/{key}"))
                })?,
                &format!("{field}/{key}"),
            )?;
        }
        let from_min = params
            .get("from_min")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        let from_max = params
            .get("from_max")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        if (from_min - from_max).abs() <= f64::EPSILON {
            return Err(invalid(
                engine,
                "transformation input range must have nonzero width",
                &format!("{field}/from_max"),
            ));
        }
    }
    if kind == ConstraintType::StretchTo
        && let Some(value) = params.get("rest_length")
    {
        let length = ranged_number(
            engine,
            value,
            0.0,
            f64::MAX,
            &format!("{field}/rest_length"),
        )?;
        if length <= 0.0 {
            return Err(invalid(
                engine,
                "rest_length must be greater than zero",
                &format!("{field}/rest_length"),
            ));
        }
    }
    if matches!(kind, ConstraintType::Floor | ConstraintType::Shrinkwrap)
        && let Some(value) = params.get("offset")
    {
        finite_value(engine, value, &format!("{field}/offset"))?;
    }
    validate_tracking_constraint_params(engine, kind, params, field)?;
    Ok(())
}

fn validate_tracking_constraint_params(
    engine: &Engine<'_>,
    kind: ConstraintType,
    params: &Map<String, Value>,
    field: &str,
) -> Result<()> {
    let required_string = |key: &str| -> Result<String> {
        let value = params.get(key).ok_or_else(|| {
            invalid(
                engine,
                "constraint parameter is required",
                &format!("{field}/{key}"),
            )
        })?;
        string_value(engine, value, &format!("{field}/{key}"))
    };
    let optional_bool = |key: &str| -> Result<Option<bool>> {
        params
            .get(key)
            .map(|value| bool_value(engine, value, &format!("{field}/{key}")))
            .transpose()
    };
    let clip_for_param = |key: &str| -> Result<_> {
        if optional_bool("use_active_clip")?.unwrap_or(false) {
            let clip_id = engine
                .doc
                .scenes
                .get(&engine.doc.active_scene)
                .and_then(|scene| scene.active_clip.as_ref())
                .ok_or_else(|| {
                    invalid(
                        engine,
                        "use_active_clip requires an active scene clip",
                        &format!("{field}/use_active_clip"),
                    )
                })?;
            return engine.doc.movie_clips.get(clip_id).ok_or_else(|| {
                engine.error(
                    ErrorCode::TargetNotFound,
                    "active scene clip was not found",
                    &pointer(engine, &format!("{field}/use_active_clip")),
                )
            });
        }
        let clip_text = required_string(key)?;
        let clip_id = Id::new(clip_text).map_err(|_| {
            invalid(
                engine,
                "constraint movie clip must be a valid ID",
                &format!("{field}/{key}"),
            )
        })?;
        engine.doc.movie_clips.get(&clip_id).ok_or_else(|| {
            engine.error(
                ErrorCode::TargetNotFound,
                "constraint movie clip was not found",
                &pointer(engine, &format!("{field}/{key}")),
            )
        })
    };
    let node_param = |key: &str, required: bool| -> Result<Option<Id>> {
        let Some(value) = params.get(key) else {
            return if required {
                Err(invalid(
                    engine,
                    "constraint node reference is required",
                    &format!("{field}/{key}"),
                ))
            } else {
                Ok(None)
            };
        };
        if value.is_null() && !required {
            return Ok(None);
        }
        let node_text = string_value(engine, value, &format!("{field}/{key}"))?;
        let node_id = Id::new(node_text).map_err(|_| {
            invalid(
                engine,
                "constraint node reference must be a valid ID",
                &format!("{field}/{key}"),
            )
        })?;
        if !engine.doc.nodes.contains_key(&node_id) {
            return Err(engine.error(
                ErrorCode::TargetNotFound,
                "constraint node reference was not found",
                &pointer(engine, &format!("{field}/{key}")),
            ));
        }
        Ok(Some(node_id))
    };
    match kind {
        ConstraintType::Action => {
            node_param("target", true)?;
            let action_text = required_string("action")?;
            let action_id = Id::new(action_text).map_err(|_| {
                invalid(
                    engine,
                    "action must be a valid ID",
                    &format!("{field}/action"),
                )
            })?;
            if !engine.doc.actions.contains_key(&action_id) {
                return Err(engine.error(
                    ErrorCode::TargetNotFound,
                    "Action constraint action was not found",
                    &pointer(engine, &format!("{field}/action")),
                ));
            }
            let transform_channel = required_string("transform_channel")?;
            if ![
                "LOCATION_X",
                "LOCATION_Y",
                "LOCATION_Z",
                "ROTATION_X",
                "ROTATION_Y",
                "ROTATION_Z",
                "SCALE_X",
                "SCALE_Y",
                "SCALE_Z",
            ]
            .contains(&transform_channel.as_str())
            {
                return Err(invalid(
                    engine,
                    "transform_channel is not a Blender Action constraint channel",
                    &format!("{field}/transform_channel"),
                ));
            }
            let target_space = required_string("target_space")?;
            if ![
                "WORLD",
                "CUSTOM",
                "POSE",
                "LOCAL_WITH_PARENT",
                "LOCAL",
                "LOCAL_OWNER_ORIENT",
            ]
            .contains(&target_space.as_str())
            {
                return Err(invalid(
                    engine,
                    "target_space is not a Blender constraint space",
                    &format!("{field}/target_space"),
                ));
            }
            let mix_mode = required_string("mix_mode")?;
            if ![
                "REPLACE",
                "BEFORE_FULL",
                "BEFORE",
                "AFTER_FULL",
                "AFTER",
                "BEFORE_SPLIT",
                "AFTER_SPLIT",
            ]
            .contains(&mix_mode.as_str())
            {
                return Err(invalid(
                    engine,
                    "mix_mode is not a Blender Action constraint mode",
                    &format!("{field}/mix_mode"),
                ));
            }
            for key in ["min", "max", "frame_start", "frame_end", "eval_time"] {
                let value = params.get(key).ok_or_else(|| {
                    invalid(
                        engine,
                        "Action constraint requires its range, frame, and evaluation-time values",
                        &format!("{field}/{key}"),
                    )
                })?;
                finite_value(engine, value, &format!("{field}/{key}"))?;
            }
            for key in ["min", "max"] {
                let value = params.get(key).ok_or_else(|| {
                    invalid(
                        engine,
                        "Action constraint value range is required",
                        &format!("{field}/{key}"),
                    )
                })?;
                ranged_number(engine, value, -1000.0, 1000.0, &format!("{field}/{key}"))?;
            }
            for key in ["frame_start", "frame_end"] {
                let value = params.get(key).and_then(Value::as_i64).ok_or_else(|| {
                    invalid(
                        engine,
                        "Action constraint frame bounds must be integers",
                        &format!("{field}/{key}"),
                    )
                })?;
                if !(-1_048_574..=1_048_574).contains(&value) {
                    return Err(invalid(
                        engine,
                        "Action constraint frame bound is outside Blender's supported range",
                        &format!("{field}/{key}"),
                    ));
                }
            }
            if let Some(value) = params.get("eval_time") {
                ranged_number(engine, value, 0.0, 1.0, &format!("{field}/eval_time"))?;
            }
            optional_bool("use_eval_time")?;
            optional_bool("use_bone_object_action")?;
        }
        ConstraintType::GeometryAttribute => {
            node_param("target", true)?;
            let name = required_string("attribute_name")?;
            if name.is_empty() {
                return Err(invalid(
                    engine,
                    "attribute_name must not be empty",
                    &format!("{field}/attribute_name"),
                ));
            }
            let data_type = required_string("data_type")?;
            if !["VECTOR", "QUATERNION", "FLOAT4X4"].contains(&data_type.as_str()) {
                return Err(invalid(
                    engine,
                    "data_type is not a Blender Geometry Attribute type",
                    &format!("{field}/data_type"),
                ));
            }
            let domain = required_string("domain")?;
            if !["POINT", "EDGE", "FACE", "FACE_CORNER", "CURVE", "INSTANCE"]
                .contains(&domain.as_str())
            {
                return Err(invalid(
                    engine,
                    "domain is not a Blender Geometry Attribute domain",
                    &format!("{field}/domain"),
                ));
            }
            let mix_mode = required_string("mix_mode")?;
            if ![
                "REPLACE",
                "BEFORE_FULL",
                "BEFORE_SPLIT",
                "AFTER_FULL",
                "AFTER_SPLIT",
            ]
            .contains(&mix_mode.as_str())
            {
                return Err(invalid(
                    engine,
                    "mix_mode is not a Blender Geometry Attribute mix mode",
                    &format!("{field}/mix_mode"),
                ));
            }
            let sample_index = params.get("sample_index").ok_or_else(|| {
                invalid(
                    engine,
                    "sample_index is required",
                    &format!("{field}/sample_index"),
                )
            })?;
            u32_value(engine, sample_index, &format!("{field}/sample_index"))?;
            for key in ["apply_target_transform", "mix_loc", "mix_rot", "mix_scl"] {
                let value = params.get(key).ok_or_else(|| {
                    invalid(
                        engine,
                        "Geometry Attribute flag is required",
                        &format!("{field}/{key}"),
                    )
                })?;
                bool_value(engine, value, &format!("{field}/{key}"))?;
            }
        }
        ConstraintType::CameraSolver => {
            let _clip = clip_for_param("clip")?;
            optional_bool("use_active_clip")?;
        }
        ConstraintType::FollowTrack => {
            let clip = clip_for_param("clip")?;
            let track_id = required_string("track")?;
            if !clip
                .tracking
                .tracks
                .iter()
                .any(|track| track.id == track_id)
            {
                return Err(engine.error(
                    ErrorCode::TargetNotFound,
                    "Follow Track track was not found",
                    &pointer(engine, &format!("{field}/track")),
                ));
            }
            if let Some(object_id) = params.get("object").filter(|value| !value.is_null()) {
                let object_id = string_value(engine, object_id, &format!("{field}/object"))?;
                let object = clip
                    .tracking
                    .objects
                    .iter()
                    .find(|object| object.id == object_id);
                if object.is_none_or(|object| !object.tracks.iter().any(|id| id == &track_id)) {
                    return Err(engine.error(
                        ErrorCode::TargetNotFound,
                        "Follow Track object must contain the selected tracking track",
                        &pointer(engine, &format!("{field}/object")),
                    ));
                }
            }
            let camera_id = node_param("camera", true)?;
            if camera_id
                .as_ref()
                .and_then(|camera_id| engine.doc.nodes.get(camera_id))
                .is_none_or(|camera| camera.kind != "camera")
            {
                return Err(invalid(
                    engine,
                    "Follow Track camera must reference a camera object",
                    &format!("{field}/camera"),
                ));
            }
            node_param("depth_object", false)?;
            let frame_method = required_string("frame_method")?;
            if !["STRETCH", "FIT", "CROP"].contains(&frame_method.as_str()) {
                return Err(invalid(
                    engine,
                    "frame_method is not a Blender Follow Track frame method",
                    &format!("{field}/frame_method"),
                ));
            }
            optional_bool("use_3d_position")?;
            optional_bool("use_undistorted_position")?;
        }
        ConstraintType::ObjectSolver => {
            let clip = clip_for_param("clip")?;
            let object_id = required_string("object")?;
            if !clip
                .tracking
                .objects
                .iter()
                .any(|object| object.id == object_id)
            {
                return Err(engine.error(
                    ErrorCode::TargetNotFound,
                    "Object Solver tracking object was not found",
                    &pointer(engine, &format!("{field}/object")),
                ));
            }
            let camera_id = node_param("camera", true)?;
            if camera_id
                .as_ref()
                .and_then(|camera_id| engine.doc.nodes.get(camera_id))
                .is_none_or(|camera| camera.kind != "camera")
            {
                return Err(invalid(
                    engine,
                    "Object Solver camera must reference a camera object",
                    &format!("{field}/camera"),
                ));
            }
            optional_bool("set_inverse_pending")?;
        }
        _ => {}
    }
    Ok(())
}

fn parse_positions(
    engine: &Engine<'_>,
    value: &Value,
    vertices: &[Vertex],
    field: &str,
) -> Result<BTreeMap<u32, [f64; 3]>> {
    let values = value.as_object().ok_or_else(|| {
        invalid(
            engine,
            "positions must be an object keyed by vertex ID",
            field,
        )
    })?;
    let known = vertices
        .iter()
        .map(|vertex| vertex.id)
        .collect::<BTreeSet<_>>();
    let mut positions = BTreeMap::new();
    for (raw_id, position) in values {
        let item_field = format!("{field}/{}", escape_pointer(raw_id));
        let vertex_id = raw_id
            .parse::<u32>()
            .map_err(|_| invalid(engine, "position key must be a vertex ID", &item_field))?;
        let canonical_id = vertex_id.to_string();
        if raw_id != &canonical_id {
            return Err(invalid(
                engine,
                "position key must be a canonical vertex ID",
                &item_field,
            ));
        }
        if !known.contains(&vertex_id) {
            return Err(invalid(
                engine,
                "position references an unknown vertex",
                &item_field,
            ));
        }
        positions.insert(vertex_id, read_vec3(engine, position, &item_field)?);
    }
    Ok(positions)
}

fn validate_shape_value(
    engine: &Engine<'_>,
    value: f64,
    min: f64,
    max: f64,
    field: &str,
) -> Result<()> {
    if !min.is_finite() || !max.is_finite() || min > max {
        return Err(invalid(
            engine,
            "slider_min must be finite and no greater than slider_max",
            field,
        ));
    }
    if !value.is_finite() || value < min || value > max {
        return Err(invalid(
            engine,
            "shape key value must be finite and within the slider range",
            field,
        ));
    }
    Ok(())
}

fn validate_shape_references(
    engine: &Engine<'_>,
    keys: Option<&ShapeKeyData>,
    data: &crate::model::DataBlock,
    id: &Id,
    relative: Option<&Id>,
    group: Option<&Id>,
    relative_field: &str,
    group_field: &str,
) -> Result<()> {
    if let Some(relative_id) = relative
        && (relative_id == id || !keys.is_some_and(|shape| shape.keys.contains_key(relative_id)))
    {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            format!("relative shape key `{relative_id}` was not found"),
            &pointer(engine, relative_field),
        ));
    }
    if let Some(group_id) = group
        && !data
            .vertex_groups
            .iter()
            .any(|vertex_group| &vertex_group.id == group_id)
    {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            format!("vertex group `{group_id}` was not found"),
            &pointer(engine, group_field),
        ));
    }
    Ok(())
}

fn parse_weights(
    engine: &Engine<'_>,
    value: &Value,
    known_vertices: &BTreeSet<u32>,
) -> Result<Vec<(u32, f64)>> {
    let weights = value
        .as_array()
        .ok_or_else(|| invalid(engine, "weights must be an array", "weights"))?;
    let mut seen = BTreeSet::new();
    let mut result = Vec::with_capacity(weights.len());
    for (index, item) in weights.iter().enumerate() {
        let field = format!("weights/{index}");
        let object = item
            .as_object()
            .ok_or_else(|| invalid(engine, "weight entry must be an object", &field))?;
        reject_unknown_fields(engine, object, &["vertex_id", "weight"], &field)?;
        let vertex_id = object.get("vertex_id").ok_or_else(|| {
            invalid(
                engine,
                "vertex_id is required",
                &format!("{field}/vertex_id"),
            )
        })?;
        let vertex_id = u32_value(engine, vertex_id, &format!("{field}/vertex_id"))?;
        if !known_vertices.contains(&vertex_id) {
            return Err(invalid(
                engine,
                "weight references an unknown vertex",
                &format!("{field}/vertex_id"),
            ));
        }
        if !seen.insert(vertex_id) {
            return Err(invalid(
                engine,
                "weights contains a duplicate vertex ID",
                &format!("{field}/vertex_id"),
            ));
        }
        let weight = object
            .get("weight")
            .ok_or_else(|| invalid(engine, "weight is required", &format!("{field}/weight")))?;
        let weight = ranged_number(engine, weight, 0.0, 1.0, &format!("{field}/weight"))?;
        result.push((vertex_id, weight));
    }
    Ok(result)
}

fn parse_vertex_ids(
    engine: &Engine<'_>,
    value: &Value,
    vertices: &[Vertex],
    field: &str,
) -> Result<Vec<u32>> {
    let values = value
        .as_array()
        .ok_or_else(|| invalid(engine, "vertex_ids must be an array", field))?;
    let known = vertices
        .iter()
        .map(|vertex| vertex.id)
        .collect::<BTreeSet<_>>();
    let mut seen = BTreeSet::new();
    let mut result = Vec::with_capacity(values.len());
    for (index, value) in values.iter().enumerate() {
        let item_field = format!("{field}/{index}");
        let id = u32_value(engine, value, &item_field)?;
        if !known.contains(&id) {
            return Err(invalid(engine, "vertex ID does not exist", &item_field));
        }
        if !seen.insert(id) {
            return Err(invalid(
                engine,
                "vertex_ids contains a duplicate",
                &item_field,
            ));
        }
        result.push(id);
    }
    Ok(result)
}

fn parse_bone_ids(engine: &Engine<'_>, value: &Value, field: &str) -> Result<Vec<Id>> {
    let values = value
        .as_array()
        .ok_or_else(|| invalid(engine, "bone_ids must be an array", field))?;
    let mut seen = BTreeSet::new();
    let mut ids = Vec::with_capacity(values.len());
    for (index, value) in values.iter().enumerate() {
        let item_field = format!("{field}/{index}");
        let id = value
            .as_str()
            .ok_or_else(|| invalid(engine, "bone IDs must be strings", &item_field))?;
        let id = Id::new(id).map_err(|_| invalid(engine, "invalid bone ID", &item_field))?;
        if !seen.insert(id.clone()) {
            return Err(invalid(
                engine,
                "bone_ids contains a duplicate",
                &item_field,
            ));
        }
        ids.push(id);
    }
    Ok(ids)
}

fn parse_driver_variables(engine: &Engine<'_>, value: &Value) -> Result<Vec<DriverVariable>> {
    let values = value
        .as_array()
        .ok_or_else(|| invalid(engine, "variables must be an array", "variables"))?;
    let mut names = BTreeSet::new();
    let mut result = Vec::with_capacity(values.len());
    for (index, value) in values.iter().enumerate() {
        let field = format!("variables/{index}");
        let object = value
            .as_object()
            .ok_or_else(|| invalid(engine, "variable must be an object", &field))?;
        reject_unknown_fields(
            engine,
            object,
            &[
                "name",
                "type",
                "target",
                "target_2",
                "path",
                "index",
                "transform_space",
            ],
            &field,
        )?;
        for required in ["name", "type", "target"] {
            if !object.contains_key(required) {
                return Err(invalid(
                    engine,
                    &format!("{required} is required"),
                    &format!("{field}/{required}"),
                ));
            }
        }
        let name_field = format!("{field}/name");
        let name = nonempty_string(
            engine,
            object
                .get("name")
                .ok_or_else(|| invalid(engine, "name is required", &name_field))?,
            &name_field,
        )?;
        if !names.insert(name.clone()) {
            return Err(invalid(
                engine,
                "driver variable names must be unique",
                &name_field,
            ));
        }
        let type_field = format!("{field}/type");
        let variable_type = driver_variable_type(
            engine,
            object
                .get("type")
                .ok_or_else(|| invalid(engine, "type is required", &type_field))?,
            &type_field,
        )?;
        let target_field = format!("{field}/target");
        let target = parse_driver_target(
            engine,
            object
                .get("target")
                .ok_or_else(|| invalid(engine, "target is required", &target_field))?,
            &target_field,
        )?;
        let target_2 = object
            .get("target_2")
            .filter(|value| !value.is_null())
            .map(|value| parse_driver_target(engine, value, &format!("{field}/target_2")))
            .transpose()?;
        let is_difference = matches!(
            variable_type,
            DriverVariableType::LocationDifference | DriverVariableType::RotationDifference
        );
        if is_difference && target_2.is_none() {
            return Err(invalid(
                engine,
                "difference variables require target_2",
                &format!("{field}/target_2"),
            ));
        }
        if !is_difference && target_2.is_some() {
            return Err(invalid(
                engine,
                "target_2 is only valid for difference variables",
                &format!("{field}/target_2"),
            ));
        }
        let path_field = format!("{field}/path");
        let path = if let Some(value) = object.get("path") {
            nonempty_string(engine, value, &path_field)?
        } else if is_difference {
            String::new()
        } else {
            return Err(invalid(engine, "path is required", &path_field));
        };
        let index = object
            .get("index")
            .map(|value| u32_value(engine, value, &format!("{field}/index")))
            .transpose()?
            .unwrap_or(0);
        let transform_space = object
            .get("transform_space")
            .map(|value| {
                let space = string_value(engine, value, &format!("{field}/transform_space"))?;
                match space.as_str() {
                    "local" | "local_space" => Ok("local".to_owned()),
                    "world" | "world_space" => Ok("world".to_owned()),
                    _ => Err(invalid(
                        engine,
                        "transform_space must be local or world",
                        &format!("{field}/transform_space"),
                    )),
                }
            })
            .transpose()?;
        result.push(DriverVariable {
            name,
            variable_type,
            target,
            path,
            index,
            target_2,
            transform_space,
        });
    }
    Ok(result)
}

fn parse_driver_target(engine: &Engine<'_>, value: &Value, field: &str) -> Result<Id> {
    let text = string_value(engine, value, field)?;
    let target = Id::new(text).map_err(|_| invalid(engine, "invalid target node ID", field))?;
    if !engine.doc.nodes.contains_key(&target) {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            format!("driver target node `{target}` was not found"),
            &pointer(engine, field),
        ));
    }
    Ok(target)
}

fn validate_driver(
    engine: &Engine<'_>,
    driver_type: DriverType,
    variables: &[DriverVariable],
    expression: Option<&str>,
    expression_field: &str,
) -> Result<()> {
    if driver_type == DriverType::ScriptedExpression && expression.is_none_or(str::is_empty) {
        return Err(invalid(
            engine,
            "scripted_expression drivers require a non-empty expression",
            expression_field,
        ));
    }
    if driver_type != DriverType::ScriptedExpression && expression.is_some() {
        return Err(invalid(
            engine,
            "expression is only valid for scripted_expression drivers",
            expression_field,
        ));
    }
    for variable in variables {
        if !engine.doc.nodes.contains_key(&variable.target) {
            return Err(engine.error(
                ErrorCode::TargetNotFound,
                format!("driver target node `{}` was not found", variable.target),
                &pointer(engine, "variables"),
            ));
        }
        if let Some(target_2) = &variable.target_2
            && !engine.doc.nodes.contains_key(target_2)
        {
            return Err(engine.error(
                ErrorCode::TargetNotFound,
                format!("second driver target node `{target_2}` was not found"),
                &pointer(engine, "variables"),
            ));
        }
        if matches!(
            variable.variable_type,
            DriverVariableType::LocationDifference | DriverVariableType::RotationDifference
        ) != variable.target_2.is_some()
        {
            return Err(invalid(
                engine,
                "only difference variables require target_2",
                "variables",
            ));
        }
    }
    Ok(())
}

fn driver_type(engine: &Engine<'_>, value: &Value, field: &str) -> Result<DriverType> {
    match string_value(engine, value, field)?.as_str() {
        "average" => Ok(DriverType::Average),
        "sum" => Ok(DriverType::Sum),
        "min" => Ok(DriverType::Min),
        "max" => Ok(DriverType::Max),
        "scripted_expression" => Ok(DriverType::ScriptedExpression),
        _ => Err(invalid(engine, "unknown driver type", field)),
    }
}

fn driver_variable_type(
    engine: &Engine<'_>,
    value: &Value,
    field: &str,
) -> Result<DriverVariableType> {
    let variable = string_value(engine, value, field)?;
    match variable.as_str() {
        "single_prop" => Ok(DriverVariableType::SingleProp),
        "transforms" => Ok(DriverVariableType::Transforms),
        "loc_diff" | "location_difference" => Ok(DriverVariableType::LocationDifference),
        "rot_diff" | "rotation_diff" | "rotation_difference" => {
            Ok(DriverVariableType::RotationDifference)
        }
        "context_prop" | "context_property" => Ok(DriverVariableType::ContextProperty),
        _ => Err(invalid(engine, "unknown driver variable type", field)),
    }
}

fn read_set<'a>(
    engine: &Engine<'_>,
    operation: &'a Map<String, Value>,
) -> Result<&'a Map<String, Value>> {
    operation
        .get("set")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid(engine, "set must be an object", "set"))
}

fn reject_unknown_set_fields(
    engine: &Engine<'_>,
    object: &Map<String, Value>,
    allowed: &[&str],
) -> Result<()> {
    reject_unknown_fields(engine, object, allowed, "set")
}

fn reject_unknown_fields(
    engine: &Engine<'_>,
    object: &Map<String, Value>,
    allowed: &[&str],
    prefix: &str,
) -> Result<()> {
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(invalid(
                engine,
                &format!("unknown field `{key}`"),
                &format!("{prefix}/{}", escape_pointer(key)),
            ));
        }
    }
    Ok(())
}

fn read_vec3(engine: &Engine<'_>, value: &Value, field: &str) -> Result<[f64; 3]> {
    let array = value
        .as_array()
        .ok_or_else(|| invalid(engine, "value must be an array of 3 finite numbers", field))?;
    if array.len() != 3 {
        return Err(invalid(
            engine,
            "value must contain exactly 3 numbers",
            field,
        ));
    }
    let mut result = [0.0; 3];
    for (index, item) in array.iter().enumerate() {
        result[index] = finite_value(engine, item, &format!("{field}/{index}"))?;
    }
    Ok(result)
}

fn read_vec4(engine: &Engine<'_>, value: &Value, field: &str) -> Result<[f64; 4]> {
    let array = value
        .as_array()
        .ok_or_else(|| invalid(engine, "value must be an array of 4 finite numbers", field))?;
    if array.len() != 4 {
        return Err(invalid(
            engine,
            "value must contain exactly 4 numbers",
            field,
        ));
    }
    let mut result = [0.0; 4];
    for (index, item) in array.iter().enumerate() {
        result[index] = finite_value(engine, item, &format!("{field}/{index}"))?;
    }
    Ok(result)
}

fn finite_value(engine: &Engine<'_>, value: &Value, field: &str) -> Result<f64> {
    value
        .as_f64()
        .filter(|number| number.is_finite())
        .ok_or_else(|| invalid(engine, "value must be a finite number", field))
}
fn optional_finite(engine: &Engine<'_>, value: Option<&Value>, field: &str) -> Result<Option<f64>> {
    value
        .map(|value| finite_value(engine, value, field))
        .transpose()
}
fn nonnegative_optional(
    engine: &Engine<'_>,
    value: Option<&Value>,
    field: &str,
    default: f64,
) -> Result<f64> {
    let number = optional_finite(engine, value, field)?.unwrap_or(default);
    if number < 0.0 {
        return Err(invalid(engine, "value must be non-negative", field));
    }
    Ok(number)
}

fn validate_custom_shape_reference(
    engine: &Engine<'_>,
    custom_shape: Option<&Id>,
    field: &str,
) -> Result<()> {
    if let Some(id) = custom_shape
        && !engine.doc.nodes.contains_key(id)
    {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            format!("custom bone shape object `{id}` was not found"),
            &pointer(engine, field),
        ));
    }
    Ok(())
}

fn ranged_number(
    engine: &Engine<'_>,
    value: &Value,
    min: f64,
    max: f64,
    field: &str,
) -> Result<f64> {
    let number = finite_value(engine, value, field)?;
    if number < min || number > max {
        return Err(invalid(
            engine,
            &format!("value must be in the range {min}..={max}"),
            field,
        ));
    }
    Ok(number)
}

fn u32_value(engine: &Engine<'_>, value: &Value, field: &str) -> Result<u32> {
    value
        .as_u64()
        .and_then(|number| u32::try_from(number).ok())
        .ok_or_else(|| invalid(engine, "value must be an unsigned 32-bit integer", field))
}

fn usize_value(engine: &Engine<'_>, value: &Value, field: &str) -> Result<usize> {
    value
        .as_u64()
        .and_then(|number| usize::try_from(number).ok())
        .ok_or_else(|| invalid(engine, "value must be a non-negative integer", field))
}

fn bool_value(engine: &Engine<'_>, value: &Value, field: &str) -> Result<bool> {
    value
        .as_bool()
        .ok_or_else(|| invalid(engine, "value must be a boolean", field))
}

fn string_value(engine: &Engine<'_>, value: &Value, field: &str) -> Result<String> {
    value
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| invalid(engine, "value must be a string", field))
}

fn nonempty_string(engine: &Engine<'_>, value: &Value, field: &str) -> Result<String> {
    let text = string_value(engine, value, field)?;
    if text.trim().is_empty() {
        return Err(invalid(engine, "value must not be empty", field));
    }
    Ok(text)
}

fn optional_string(
    engine: &Engine<'_>,
    value: Option<&Value>,
    field: &str,
) -> Result<Option<String>> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(value) => Ok(Some(string_value(engine, value, field)?)),
    }
}

fn optional_id(engine: &Engine<'_>, value: Option<&Value>, field: &str) -> Result<Option<Id>> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(value) => {
            let raw = string_value(engine, value, field)?;
            Id::new(raw)
                .map(Some)
                .map_err(|_| invalid(engine, "invalid ID", field))
        }
    }
}

fn object_value(engine: &Engine<'_>, value: &Value, field: &str) -> Result<Map<String, Value>> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid(engine, "value must be an object", field))?;
    validate_finite_json(engine, value, field)?;
    Ok(object.clone())
}

fn read_params(
    engine: &Engine<'_>,
    value: Option<&Value>,
    field: &str,
) -> Result<Map<String, Value>> {
    match value {
        None => Ok(Map::new()),
        Some(value) => object_value(engine, value, field),
    }
}

fn validate_finite_json(engine: &Engine<'_>, value: &Value, field: &str) -> Result<()> {
    if contains_non_finite_number(value) {
        Err(invalid(engine, "numeric parameters must be finite", field))
    } else {
        Ok(())
    }
}

fn contains_non_finite_number(value: &Value) -> bool {
    match value {
        Value::Number(number) => number.as_f64().is_some_and(|number| !number.is_finite()),
        Value::Array(values) => values.iter().any(contains_non_finite_number),
        Value::Object(values) => values.values().any(contains_non_finite_number),
        _ => false,
    }
}

fn pointer(engine: &Engine<'_>, suffix: &str) -> String {
    operation_pointer(engine.operation_index, suffix)
}

fn invalid(engine: &Engine<'_>, message: &str, field: &str) -> PotError {
    engine.error(
        ErrorCode::InvalidOperation,
        message,
        &pointer(engine, field),
    )
}

fn missing(engine: &Engine<'_>, kind: &str, id: &Id, field: &str) -> PotError {
    engine.error(
        ErrorCode::TargetNotFound,
        format!("{kind} `{id}` was not found"),
        &pointer(engine, field),
    )
}

fn escape_pointer(value: &str) -> String {
    value.replace('~', "~0").replace('/', "~1")
}

fn same_point(left: [f64; 3], right: [f64; 3]) -> bool {
    left.iter()
        .zip(right)
        .all(|(left, right)| (left - right).abs() <= 1.0e-9)
}
