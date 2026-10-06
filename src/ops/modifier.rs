use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::modifiers::evaluate_modifiers_with_groups_at_frame,
    model::{DataBlock, Id, Modifier, SceneDoc, ShapeKey, ShapeKeyData},
    params::{self, ParameterFamily},
};

use super::{
    ChangeKind, Engine, check_fields, operation_pointer, read_bool, read_string,
    resolve_node_targets,
};

pub(super) fn apply(
    engine: &mut Engine<'_>,
    name: &str,
    operation: &Map<String, Value>,
) -> Result<bool> {
    match name {
        "modifier.create" => create(engine, operation),
        "modifier.update" => update(engine, operation),
        "modifier.delete" => delete(engine, operation),
        "modifier.reorder" => reorder(engine, operation),
        "modifier.bind" => bind_modifier(engine, operation),
        "modifier.unbind" => unbind_modifier(engine, operation),
        "modifier.apply" => apply_modifier(engine, operation, false),
        "modifier.apply_as_shape_key" => apply_modifier_as_shape_key(engine, operation),
        _ => Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("unsupported modifier operation `{name}`"),
            "/op",
        )),
    }
}
fn bind_modifier(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "modifier_id"],
        &["target", "modifier_id"],
    )?;
    let modifier_id = read_modifier_id(engine, operation)?;
    for subject_id in targets(engine, operation)? {
        let modifier = engine
            .doc
            .nodes
            .get(&subject_id)
            .and_then(|node| {
                node.modifiers
                    .iter()
                    .find(|modifier| modifier.id == modifier_id)
            })
            .cloned()
            .ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::TargetNotFound,
                    format!("modifier `{modifier_id}` was not found on `{subject_id}`"),
                    json!({"modifier_id":modifier_id,"target":subject_id}),
                )
            })?;
        if !matches!(
            modifier.modifier_type.as_str(),
            "surface_deform" | "mesh_deform" | "laplacian_deform"
        ) {
            return Err(PotError::invalid_operation(
                "modifier.bind supports surface_deform, mesh_deform, and laplacian_deform",
            ));
        }
        if modifier.binding_data.is_some() {
            return Err(PotError::invalid_operation(
                "modifier is already bound; unbind it before binding again",
            ));
        }
        let source_data_id = engine
            .doc
            .nodes
            .get(&subject_id)
            .and_then(|node| node.data.as_ref())
            .ok_or_else(|| {
                PotError::invalid_operation("modifier binding requires source mesh data")
            })?;
        let source_data = engine.doc.data_blocks.get(source_data_id).ok_or_else(|| {
            PotError::new(
                ErrorCode::TargetNotFound,
                "modifier binding source data was not found",
            )
        })?;
        if source_data.mesh.is_none() {
            return Err(PotError::invalid_operation(
                "modifier binding requires a source mesh",
            ));
        }
        let mut bind_doc = engine.doc.clone();
        let bind_node = bind_doc.nodes.get_mut(&subject_id).ok_or_else(|| {
            PotError::new(
                ErrorCode::TargetNotFound,
                "modifier binding subject disappeared",
            )
        })?;
        let bind_index = bind_node
            .modifiers
            .iter()
            .position(|candidate| candidate.id == modifier_id)
            .ok_or_else(|| PotError::new(ErrorCode::TargetNotFound, "modifier disappeared"))?;
        bind_node.modifiers.truncate(bind_index + 1);
        let bind_nodes = BTreeSet::from([subject_id.clone()]);
        let bind_snapshot = crate::eval::Snapshot::evaluate_nodes_with_cache(
            &bind_doc,
            &crate::eval::EvaluationContext {
                scene_id: None,
                view_layer: None,
                frame: Some(engine.evaluation_frame),
            },
            None,
            &bind_nodes,
        )?;
        let mut source_bind_mesh = bind_snapshot
            .meshes
            .get(&subject_id)
            .cloned()
            .ok_or_else(|| PotError::invalid_operation("evaluated source mesh is unavailable"))?;
        crate::eval::attach_vertex_group_weights(&mut source_bind_mesh, source_data);
        let binding = if modifier.modifier_type == "laplacian_deform" {
            let group_name = modifier
                .params
                .get("vertex_group")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    PotError::invalid_argument(
                        "Laplacian Deform binding requires a vertex_group name",
                    )
                })?;
            if !source_data
                .vertex_groups
                .iter()
                .any(|group| group.name == group_name)
            {
                return Err(PotError::with_details(
                    ErrorCode::TargetNotFound,
                    format!("vertex group `{group_name}` was not found"),
                    json!({"modifier_id":modifier_id,"vertex_group":group_name}),
                ));
            }
            crate::geom::modifiers::deform::bind_laplacian(&source_bind_mesh)
        } else {
            let target_parameter = if modifier.modifier_type == "surface_deform" {
                "target"
            } else {
                "object"
            };
            let target_text = modifier
                .params
                .get(target_parameter)
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    PotError::invalid_argument(format!(
                        "{} binding requires a `{target_parameter}` object ID",
                        modifier.modifier_type
                    ))
                })?;
            let target_id = Id::new(target_text)?;
            let target_data_id = engine
                .doc
                .nodes
                .get(&target_id)
                .and_then(|node| node.data.as_ref())
                .ok_or_else(|| {
                    PotError::with_details(
                        ErrorCode::TargetNotFound,
                        format!("modifier binding target `{target_id}` has no mesh data"),
                        json!({"modifier_id":modifier_id,"target":target_id}),
                    )
                })?;
            let target_data = engine.doc.data_blocks.get(target_data_id).ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::TargetNotFound,
                    format!("modifier binding target data `{target_data_id}` was not found"),
                    json!({"modifier_id":modifier_id,"target":target_id}),
                )
            })?;
            if target_data.mesh.is_none() {
                return Err(PotError::with_details(
                    ErrorCode::InvalidOperation,
                    "modifier binding target is not a mesh",
                    json!({"modifier_id":modifier_id,"target":target_id}),
                ));
            }
            let mut target_mesh =
                bind_snapshot
                    .meshes
                    .get(&target_id)
                    .cloned()
                    .ok_or_else(|| {
                        PotError::invalid_operation("evaluated target mesh is unavailable")
                    })?;
            let driver_values =
                crate::eval::rig::evaluate_drivers(engine.doc, engine.evaluation_frame)?;
            let world_matrices = crate::eval::rig::evaluate_world_matrices_with_drivers(
                engine.doc,
                engine.evaluation_frame,
                &driver_values,
            )?;
            let subject_world = world_matrices.get(&subject_id).copied().ok_or_else(|| {
                PotError::new(
                    ErrorCode::EvaluationFailed,
                    "modifier binding source transform is unavailable",
                )
            })?;
            let subject_inverse = subject_world.inverse();
            let target_world = world_matrices.get(&target_id).copied().ok_or_else(|| {
                PotError::new(
                    ErrorCode::EvaluationFailed,
                    "modifier binding target transform is unavailable",
                )
            })?;
            let to_subject = subject_inverse * target_world;
            if !subject_inverse.is_finite() || !to_subject.is_finite() {
                return Err(PotError::invalid_argument(
                    "modifier binding requires invertible source and target transforms",
                ));
            }
            for vertex in &mut target_mesh.vertices {
                vertex.co = to_subject.transform_point3(vertex.co);
            }
            if modifier.modifier_type == "surface_deform" {
                crate::geom::modifiers::deform::bind_surface(
                    &source_bind_mesh,
                    &target_mesh,
                    &modifier,
                )?
            } else {
                crate::geom::modifiers::deform::bind_mesh(
                    &source_bind_mesh,
                    &target_mesh,
                    &modifier,
                )?
            }
        };
        let node = engine.doc.nodes.get_mut(&subject_id).ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "modifier binding subject disappeared",
            )
        })?;
        let bound_modifier = node
            .modifiers
            .iter_mut()
            .find(|modifier| modifier.id == modifier_id)
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "modifier disappeared"))?;
        bound_modifier.binding_data = Some(binding);
        engine.mark("nodes", &subject_id, ChangeKind::Updated);
    }
    Ok(true)
}

fn unbind_modifier(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "modifier_id"],
        &["target", "modifier_id"],
    )?;
    let modifier_id = read_modifier_id(engine, operation)?;
    let mut changed = false;
    for target in targets(engine, operation)? {
        let modifier = engine
            .doc
            .nodes
            .get_mut(&target)
            .and_then(|node| {
                node.modifiers
                    .iter_mut()
                    .find(|modifier| modifier.id == modifier_id)
            })
            .ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::TargetNotFound,
                    format!("modifier `{modifier_id}` was not found on `{target}`"),
                    json!({"modifier_id":modifier_id,"target":target}),
                )
            })?;
        if modifier.binding_data.take().is_some() {
            engine.mark("nodes", &target, ChangeKind::Updated);
            changed = true;
        }
    }
    Ok(changed)
}

fn create(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "id", "type", "name", "enabled", "params"],
        &["target", "id", "type"],
    )?;
    let targets = targets(engine, operation)?;
    let id = read_id(engine, operation, "id")?;
    let modifier_type = read_string(engine, operation, "type")?;
    if params::type_spec(ParameterFamily::Modifier, &modifier_type).is_none() {
        return Err(unsupported(engine, &modifier_type));
    }
    let name = match operation.get("name") {
        None => modifier_type.clone(),
        Some(value) => value
            .as_str()
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "name must be a string",
                    "/name",
                )
            })?
            .to_owned(),
    };
    let enabled = read_bool(engine, operation, "enabled", true)?;
    let params = read_params(engine, operation.get("params"), "params")?;
    validate_parameter_map(
        engine,
        ParameterFamily::Modifier,
        &modifier_type,
        &params,
        "/params",
    )?;
    validate_modifier_resource(engine, &modifier_type, &params, "/params")?;
    if modifier_type == "volume_displace" {
        validate_volume_displace_params(engine, &params)?;
    } else if is_attribute_modifier_type(&modifier_type) {
        validate_attribute_modifier_params(&modifier_type, &params)?;
    }
    validate_particle_modifier_params(engine, &modifier_type, &params)?;
    let mut changed = false;
    for target in targets {
        if !engine.doc.nodes.contains_key(&target) {
            return Err(engine.error(
                ErrorCode::TargetNotFound,
                format!("node `{target}` was not found"),
                "/target",
            ));
        }
        if engine
            .doc
            .nodes
            .get(&target)
            .is_some_and(|node| node.modifiers.iter().any(|modifier| modifier.id == id))
        {
            return Err(engine.error(
                ErrorCode::IdExists,
                format!("modifier ID `{id}` already exists on node `{target}`"),
                "/id",
            ));
        }
        let modifier_params = super::physics::create_linked_modifier_settings(
            engine,
            &target,
            &id,
            &modifier_type,
            &params,
        )?;
        let node = engine
            .doc
            .nodes
            .get_mut(&target)
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "node disappeared"))?;
        node.modifiers.push(Modifier {
            id: id.clone(),
            modifier_type: modifier_type.clone(),
            name: name.clone(),
            enabled,
            params: modifier_params,
            binding_data: None,
            runtime: crate::model::ModifierRuntime::default(),
        });
        engine.mark("nodes", &target, ChangeKind::Updated);
        changed = true;
    }
    Ok(changed)
}

fn update(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "id", "modifier_id", "set"],
        &["target", "set"],
    )?;
    let targets = targets(engine, operation)?;
    let id = read_modifier_id(engine, operation)?;
    let set = operation
        .get("set")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            engine.error(ErrorCode::InvalidOperation, "set must be an object", "/set")
        })?;
    for key in set.keys() {
        if !["name", "enabled", "params", "type"].contains(&key.as_str()) {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("unknown modifier set field `{key}`"),
                "/set",
            ));
        }
    }
    let mut changed = false;
    for target in targets {
        let existing = engine
            .doc
            .nodes
            .get(&target)
            .and_then(|node| node.modifiers.iter().find(|modifier| modifier.id == id))
            .cloned()
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::TargetNotFound,
                    format!("modifier `{id}` was not found"),
                )
            })?;
        let modifier_type = set
            .get("type")
            .map(|value| {
                value.as_str().ok_or_else(|| {
                    PotError::new(
                        ErrorCode::InvalidOperation,
                        "modifier type must be a string",
                    )
                })
            })
            .transpose()?
            .unwrap_or(&existing.modifier_type);
        if params::type_spec(ParameterFamily::Modifier, modifier_type).is_none() {
            return Err(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                format!("modifier `{modifier_type}` is not supported"),
                json!({"feature_id":format!("modifier.{modifier_type}")}),
            ));
        }
        if modifier_type != existing.modifier_type
            && (is_physics_modifier(&existing.modifier_type) || is_physics_modifier(modifier_type))
        {
            return Err(PotError::invalid_operation(
                "physics modifier type cannot be changed; delete and recreate it",
            ));
        }
        let params = set
            .get("params")
            .map(|value| {
                value.as_object().cloned().ok_or_else(|| {
                    PotError::new(
                        ErrorCode::InvalidOperation,
                        "modifier params must be an object",
                    )
                })
            })
            .transpose()?;
        let modifier_params = if let Some(params) = params {
            validate_parameter_map(
                engine,
                ParameterFamily::Modifier,
                modifier_type,
                &params,
                "/set/params",
            )?;
            validate_modifier_resource(engine, modifier_type, &params, "/set/params")?;
            if modifier_type == "volume_displace" {
                validate_volume_displace_params(engine, &params)?;
            } else if is_attribute_modifier_type(modifier_type) {
                validate_attribute_modifier_params(modifier_type, &params)?;
            }
            validate_particle_modifier_params(engine, modifier_type, &params)?;
            Some(super::physics::update_linked_modifier_settings(
                engine,
                &target,
                &id,
                modifier_type,
                &params,
            )?)
        } else {
            None
        };
        let node = engine.doc.nodes.get_mut(&target).ok_or_else(|| {
            PotError::new(
                ErrorCode::TargetNotFound,
                format!("node `{target}` was not found"),
            )
        })?;
        let modifier = node
            .modifiers
            .iter_mut()
            .find(|modifier| modifier.id == id)
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::TargetNotFound,
                    format!("modifier `{id}` was not found"),
                )
            })?;
        let before = modifier.clone();
        if let Some(value) = set.get("name") {
            value
                .as_str()
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::InvalidOperation,
                        "modifier name must be a string",
                    )
                })?
                .clone_into(&mut modifier.name);
        }
        if let Some(value) = set.get("enabled") {
            modifier.enabled = value.as_bool().ok_or_else(|| {
                PotError::new(
                    ErrorCode::InvalidOperation,
                    "modifier enabled must be a boolean",
                )
            })?;
        }
        modifier_type.clone_into(&mut modifier.modifier_type);
        if let Some(params) = modifier_params {
            modifier.params = params;
        }
        if modifier != &before {
            engine.mark("nodes", &target, ChangeKind::Updated);
            changed = true;
        }
        if set.contains_key("params") && is_physics_modifier(modifier_type) {
            changed = true;
        }
    }
    Ok(changed)
}

fn delete(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "id", "modifier_id"],
        &["target"],
    )?;
    let targets = targets(engine, operation)?;
    let id = read_modifier_id(engine, operation)?;
    let mut changed = false;
    for target in targets {
        let existing = engine
            .doc
            .nodes
            .get(&target)
            .and_then(|node| node.modifiers.iter().find(|modifier| modifier.id == id))
            .cloned()
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::TargetNotFound,
                    format!("modifier `{id}` was not found on node `{target}`"),
                    "/modifier_id",
                )
            })?;
        super::physics::delete_linked_modifier_settings(engine, &target, &existing)?;
        let node = engine.doc.nodes.get_mut(&target).ok_or_else(|| {
            PotError::new(
                ErrorCode::TargetNotFound,
                format!("node `{target}` was not found"),
            )
        })?;
        node.modifiers.retain(|modifier| modifier.id != id);
        engine.mark("nodes", &target, ChangeKind::Updated);
        changed = true;
    }
    Ok(changed)
}

fn reorder(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "id", "modifier_id", "index", "to_index"],
        &["target"],
    )?;
    let targets = targets(engine, operation)?;
    let id = read_modifier_id(engine, operation)?;
    let index_field = match (
        operation.contains_key("index"),
        operation.contains_key("to_index"),
    ) {
        (true, false) => "index",
        (false, true) => "to_index",
        _ => {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "provide exactly one of index or to_index",
                "/index",
            ));
        }
    };
    let index = operation
        .get(index_field)
        .and_then(Value::as_u64)
        .and_then(|index| usize::try_from(index).ok())
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "index must be a non-negative integer",
                &format!("/{index_field}"),
            )
        })?;
    let mut changed = false;
    for target in targets {
        let node = engine.doc.nodes.get_mut(&target).ok_or_else(|| {
            PotError::new(
                ErrorCode::TargetNotFound,
                format!("node `{target}` was not found"),
            )
        })?;
        let Some(current) = node.modifiers.iter().position(|modifier| modifier.id == id) else {
            return Err(engine.error(
                ErrorCode::TargetNotFound,
                format!("modifier `{id}` was not found"),
                "/modifier_id",
            ));
        };
        if index >= node.modifiers.len() {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "modifier index is outside the stack",
                "/index",
            ));
        }
        if current != index {
            let modifier = node.modifiers.remove(current);
            node.modifiers.insert(index, modifier);
            engine.mark("nodes", &target, ChangeKind::Updated);
            changed = true;
        }
    }
    Ok(changed)
}

fn apply_modifier(
    engine: &mut Engine<'_>,
    operation: &Map<String, Value>,
    allow_shape_keys: bool,
) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "id", "modifier_id", "scope"],
        &["target"],
    )?;
    let targets = targets(engine, operation)?;
    let modifier_id = read_modifier_id(engine, operation)?;
    let scope = super::parse_scope(
        engine,
        operation.get("scope"),
        "/scope",
        "scope must be a string",
        "scope must be shared or single_user",
        true,
    )?;
    let mut data_targets = Vec::new();
    for target in &targets {
        let data_id = engine
            .doc
            .nodes
            .get(target)
            .and_then(|node| node.data.clone())
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    format!("node `{target}` has no geometry data"),
                    "/target",
                )
            })?;
        data_targets.push((target.clone(), data_id));
    }
    let shared = data_targets
        .iter()
        .any(|(_, data_id)| super::data_user_count(engine, data_id) > 1);
    if shared && scope.is_none() {
        return Err(engine.error(
            ErrorCode::SharedDataRequiresScope,
            "shared geometry updates require scope",
            "/scope",
        ));
    }
    if scope == Some("single_user") {
        for (target, data_id) in &data_targets {
            if super::data_user_count(engine, data_id) > 1 {
                let copy = engine
                    .doc
                    .data_blocks
                    .get(data_id)
                    .cloned()
                    .ok_or_else(|| {
                        PotError::new(
                            ErrorCode::TargetNotFound,
                            format!("data block `{data_id}` was not found"),
                        )
                    })?;
                let new_id = super::unique_data_id(
                    engine,
                    target,
                    "_mesh_single",
                    super::DataIdCollisionHandling::ReserveSuffix,
                    "data ID suffix exhausted",
                )?;
                engine.doc.data_blocks.insert(new_id.clone(), copy);
                if let Some(node) = engine.doc.nodes.get_mut(target) {
                    node.data = Some(new_id.clone());
                }
                engine.mark("data_blocks", &new_id, ChangeKind::Created);
                engine.mark("nodes", target, ChangeKind::Updated);
                engine.map_id("data_blocks", data_id, &new_id);
            }
        }
    }
    let mut changed = false;
    let mut edited_data = Vec::<Id>::new();
    for target in targets {
        let data_id = engine
            .doc
            .nodes
            .get(&target)
            .and_then(|node| node.data.clone())
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::InvalidOperation,
                    format!("node `{target}` has no geometry data"),
                )
            })?;
        if edited_data.contains(&data_id) {
            continue;
        }
        let modifiers = engine
            .doc
            .nodes
            .get(&target)
            .map(|node| node.modifiers.clone())
            .unwrap_or_default();
        let index = modifiers
            .iter()
            .position(|modifier| modifier.id == modifier_id)
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::TargetNotFound,
                    format!("modifier `{modifier_id}` was not found on node `{target}`"),
                )
            })?;
        crate::geom::modifiers::validate_boolean_topology_suffix(&modifiers, index)?;
        let evaluated = {
            let data_block = engine.doc.data_blocks.get(&data_id).ok_or_else(|| {
                PotError::new(
                    ErrorCode::TargetNotFound,
                    format!("data block `{data_id}` was not found"),
                )
            })?;
            let output = data_block.mesh.as_ref().ok_or_else(|| {
                PotError::new(
                    ErrorCode::InvalidOperation,
                    format!("data block `{data_id}` has no mesh"),
                )
            })?;
            let modifier_stack = &modifiers[..=index];
            let uses_armature = modifier_stack
                .iter()
                .any(|modifier| modifier.enabled && modifier.modifier_type == "armature");
            let uses_lattice = modifier_stack
                .iter()
                .any(|modifier| modifier.enabled && modifier.modifier_type == "lattice");
            let uses_volume_node = modifier_stack.iter().any(|modifier| {
                modifier.enabled
                    && modifier.modifier_type == "volume_to_mesh"
                    && modifier.params.get("volume").is_some_and(Value::is_string)
            });
            let uses_scene_operand = modifier_stack.iter().any(|modifier| {
                if !modifier.enabled {
                    return false;
                }
                match modifier.modifier_type.as_str() {
                    "boolean" | "shrinkwrap" | "surface_deform" | "mesh_deform" => {
                        ["object", "collection", "target"]
                            .iter()
                            .any(|key| modifier.params.get(*key).is_some_and(Value::is_string))
                    }
                    "curve"
                    | "hook"
                    | "data_transfer"
                    | "vertex_weight_proximity"
                    | "normal_edit" => ["object", "target"]
                        .iter()
                        .any(|key| modifier.params.get(*key).is_some_and(Value::is_string)),
                    "uv_project" => modifier
                        .params
                        .get("projectors")
                        .is_some_and(Value::is_array),
                    "uv_warp" => ["object_from", "object_to"]
                        .iter()
                        .any(|key| modifier.params.get(*key).is_some_and(Value::is_string)),
                    "warp" => ["object_from", "object_to"]
                        .iter()
                        .any(|key| modifier.params.get(*key).is_some_and(Value::is_string)),
                    "vertex_weight_edit" | "vertex_weight_mix" => modifier
                        .params
                        .get("mask_tex_map_object")
                        .is_some_and(Value::is_string),
                    "array" => ["start_cap", "end_cap"]
                        .iter()
                        .any(|key| modifier.params.get(*key).is_some_and(Value::is_string)),
                    _ => false,
                }
            });
            let uses_vertex_group = modifier_stack.iter().any(|modifier| {
                modifier.enabled
                    && (modifier.modifier_type == "mask"
                        || (matches!(
                            modifier.modifier_type.as_str(),
                            "surface_deform" | "mesh_deform" | "laplacian_deform"
                        ) && modifier
                            .params
                            .get("vertex_group")
                            .is_some_and(Value::is_string)))
            });
            let mut prepared_output = uses_vertex_group.then(|| output.clone());
            if let Some(mesh) = &mut prepared_output {
                crate::eval::attach_vertex_group_weights(mesh, data_block);
            }
            if !allow_shape_keys && data_block.shape_keys.is_some() {
                return Err(PotError::with_details(
                    ErrorCode::UnsupportedFeature,
                    "Modifier cannot be applied to a mesh with shape keys",
                    json!({"feature_id":"modifier.apply.shape_key_guard","modifier_id":modifier_id}),
                ));
            }
            if !uses_armature && !uses_lattice && !uses_volume_node && !uses_scene_operand {
                let source = prepared_output.as_ref().unwrap_or(output);
                evaluate_modifiers_with_groups_at_frame(
                    source,
                    modifier_stack,
                    &engine.doc.node_groups,
                    engine.evaluation_frame,
                )?
                .mesh
            } else {
                let frame = engine.evaluation_frame;
                let scene_snapshot = if uses_scene_operand {
                    Some(crate::eval::Snapshot::evaluate_with_cache(
                        engine.doc,
                        &crate::eval::EvaluationContext {
                            frame: Some(frame),
                            ..crate::eval::EvaluationContext::default()
                        },
                        engine.asset_root,
                    )?)
                } else {
                    None
                };
                let empty_meshes = BTreeMap::new();
                let evaluated_operand_meshes = scene_snapshot
                    .as_ref()
                    .map_or(&empty_meshes, |snapshot| &snapshot.meshes);
                let driver_values = crate::eval::rig::evaluate_drivers(engine.doc, frame)?;
                let world_matrices: BTreeMap<_, _> = if let Some(snapshot) = &scene_snapshot {
                    snapshot
                        .nodes
                        .iter()
                        .map(|(id, node)| {
                            (id.clone(), glam::DMat4::from_cols_array(&node.world_matrix))
                        })
                        .collect()
                } else {
                    crate::eval::rig::evaluate_world_matrices_with_drivers(
                        engine.doc,
                        frame,
                        &driver_values,
                    )?
                };
                let base_object_meshes: BTreeMap<_, _> = engine
                    .doc
                    .nodes
                    .iter()
                    .filter_map(|(node_id, node)| {
                        let data_id = node.data.as_ref()?;
                        let data_block = engine.doc.data_blocks.get(data_id)?;
                        (data_block.data_type == "mesh")
                            .then(|| data_block.mesh.as_ref().map(|mesh| (node_id.clone(), mesh)))
                            .flatten()
                    })
                    .collect();
                let volume_data = if uses_volume_node {
                    crate::eval::load_vdb_volume_data(engine.doc, engine.asset_root)?
                } else {
                    BTreeMap::new()
                };
                let source = prepared_output.as_ref().unwrap_or(output);
                let object_world = world_matrices.get(&target).copied().ok_or_else(|| {
                    PotError::new(
                        ErrorCode::EvaluationFailed,
                        "modifier subject transform is unavailable",
                    )
                })?;
                let evaluation = crate::geom::modifiers::evaluate_modifiers_with_scene_context(
                    source.clone(),
                    modifier_stack,
                    &engine.doc.node_groups,
                    frame,
                    object_world,
                    &world_matrices,
                    &base_object_meshes,
                    evaluated_operand_meshes,
                    &BTreeMap::new(),
                    |mesh, modifier, modifier_warnings| match modifier.modifier_type.as_str() {
                        "armature" => crate::eval::rig::apply_armature_modifier_with_drivers(
                            mesh,
                            data_block,
                            modifier,
                            object_world,
                            engine.doc,
                            frame,
                            &world_matrices,
                            &driver_values,
                            engine.asset_root,
                            engine
                                .doc
                                .scenes
                                .get(&engine.doc.active_scene)
                                .map_or(24, |scene| scene.fps),
                            engine
                                .doc
                                .scenes
                                .get(&engine.doc.active_scene)
                                .map_or(1.0, |scene| scene.fps_base),
                            evaluated_operand_meshes,
                        ),
                        "lattice" => apply_lattice_modifier(
                            mesh,
                            data_block,
                            &target,
                            modifier,
                            engine.doc,
                            &world_matrices,
                        ),
                        "volume_to_mesh" => crate::eval::apply_volume_to_mesh_modifier(
                            mesh,
                            modifier,
                            &target,
                            engine.doc,
                            &world_matrices,
                            &volume_data,
                        ),
                        "surface_deform" | "mesh_deform" => {
                            crate::eval::apply_scene_deform_modifier(
                                mesh,
                                modifier,
                                &target,
                                engine.doc,
                                &world_matrices,
                                evaluated_operand_meshes,
                                modifier_warnings,
                            )
                        }
                        "boolean" | "shrinkwrap" | "curve" | "hook" | "warp" => {
                            crate::eval::apply_scene_operand_modifier(
                                mesh,
                                modifier,
                                &target,
                                engine.doc,
                                &world_matrices,
                                evaluated_operand_meshes,
                            )
                        }
                        "vertex_weight_edit"
                        | "vertex_weight_mix"
                        | "vertex_weight_proximity"
                        | "normal_edit"
                        | "uv_project"
                        | "uv_warp"
                        | "data_transfer" => crate::eval::apply_scene_attribute_modifier(
                            mesh,
                            modifier,
                            &target,
                            engine.doc,
                            &world_matrices,
                            evaluated_operand_meshes,
                            engine.asset_root,
                        ),
                        _ => Err(PotError::new(
                            ErrorCode::InternalError,
                            "custom modifier callback received an unexpected type",
                        )),
                    },
                )?;
                if let Some(warning) = evaluation.warnings.into_iter().next() {
                    return Err(warning);
                }
                evaluation.mesh
            }
        };
        let mesh_changed = engine
            .doc
            .data_blocks
            .get(&data_id)
            .and_then(|data| data.mesh.as_ref())
            .is_some_and(|output| evaluated != *output);
        let descriptor_dropped = mesh_changed
            && engine
                .doc
                .data_blocks
                .get(&data_id)
                .is_some_and(|data| data.descriptor.is_some());
        if let Some(data) = engine.doc.data_blocks.get_mut(&data_id) {
            data.mesh = Some(evaluated);
            if mesh_changed {
                data.descriptor = None;
            }
        }
        if let Some(node) = engine.doc.nodes.get_mut(&target) {
            node.modifiers.drain(..=index);
        }
        if mesh_changed {
            super::clear_primitive_metadata(engine, &data_id);
        }
        engine.mark("data_blocks", &data_id, ChangeKind::Updated);
        engine.mark("nodes", &target, ChangeKind::Updated);
        if descriptor_dropped {
            engine
                .id_mappings
                .entry("mesh_descriptor_dropped".to_owned())
                .or_insert_with(|| json!({}))[data_id.as_str()] = Value::Bool(true);
        }
        edited_data.push(data_id);
        changed = true;
    }
    Ok(changed)
}
fn apply_modifier_as_shape_key(
    engine: &mut Engine<'_>,
    operation: &Map<String, Value>,
) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &[
            "op",
            "target",
            "modifier_id",
            "id",
            "name",
            "keep_modifier",
            "scope",
        ],
        &["target", "modifier_id", "id"],
    )?;
    let modifier_id = Id::new(read_string(engine, operation, "modifier_id")?)?;
    let key_id = read_id(engine, operation, "id")?;
    let keep_modifier = read_bool(engine, operation, "keep_modifier", false)?;
    let selected = targets(engine, operation)?;
    let name = operation
        .get("name")
        .map(|value| {
            value.as_str().map(str::to_owned).ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "name must be a string",
                    "/name",
                )
            })
        })
        .transpose()?;
    let mut temp_doc = engine.doc.clone();
    let mut temp_operation = Map::new();
    temp_operation.insert("op".to_owned(), Value::String("modifier.apply".to_owned()));
    for field in ["target", "modifier_id", "scope"] {
        if let Some(value) = operation.get(field) {
            temp_operation.insert(field.to_owned(), value.clone());
        }
    }
    let mut temp_engine = Engine {
        doc: &mut temp_doc,
        changes: super::ChangeSet::default(),
        id_mappings: Map::new(),
        operation_index: engine.operation_index,
        evaluation_frame: engine.evaluation_frame,
        asset_root: engine.asset_root,
        pending_assets: BTreeMap::new(),
    };
    apply_modifier(&mut temp_engine, &temp_operation, true)?;
    for target in selected {
        let original_data_id = engine
            .doc
            .nodes
            .get(&target)
            .and_then(|node| node.data.clone())
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "target has no mesh data",
                    "/target",
                )
            })?;
        let evaluated_data_id = temp_doc
            .nodes
            .get(&target)
            .and_then(|node| node.data.as_ref())
            .ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "evaluated target has no data")
            })?;
        let evaluated = temp_doc
            .data_blocks
            .get(evaluated_data_id)
            .and_then(|data| data.mesh.as_ref())
            .ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "evaluated target has no mesh")
            })?;
        let source_data = engine
            .doc
            .data_blocks
            .get(&original_data_id)
            .ok_or_else(|| {
                PotError::new(ErrorCode::TargetNotFound, "target mesh data was not found")
            })?;
        let source_mesh = source_data.mesh.as_ref().ok_or_else(|| {
            PotError::new(ErrorCode::InvalidOperation, "target data block has no mesh")
        })?;
        if !same_topology(source_mesh, evaluated) {
            return Err(PotError::with_details(
                ErrorCode::InvalidOperation,
                "Modifier cannot be applied as a shape key because it changes the mesh topology",
                json!({"feature_id":"modifier.apply_as_shape_key.topology","modifier_id":modifier_id,"target":target}),
            ));
        }
        if source_data
            .shape_keys
            .as_ref()
            .is_some_and(|shape_keys| shape_keys.keys.contains_key(&key_id))
        {
            return Err(engine.error(
                ErrorCode::IdExists,
                format!("shape key ID `{key_id}` already exists"),
                "/id",
            ));
        }
        let positions = evaluated
            .vertices
            .iter()
            .map(|vertex| (vertex.id, vertex.co.to_array()))
            .collect();
        let base_positions = source_mesh
            .vertices
            .iter()
            .map(|vertex| (vertex.id, vertex.co.to_array()))
            .collect();
        let key_name = name.clone().unwrap_or_else(|| {
            engine
                .doc
                .nodes
                .get(&target)
                .and_then(|node| node.modifiers.iter().find(|item| item.id == modifier_id))
                .map_or_else(
                    || "Applied Modifier".to_owned(),
                    |modifier| modifier.name.clone(),
                )
        });
        let block = engine
            .doc
            .data_blocks
            .get_mut(&original_data_id)
            .ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "target mesh data disappeared")
            })?;
        let shape_keys = block.shape_keys.get_or_insert_with(ShapeKeyData::default);
        if shape_keys.keys.is_empty() {
            shape_keys.basis = base_positions;
        }
        shape_keys.keys.insert(
            key_id.clone(),
            ShapeKey {
                id: key_id.clone(),
                name: key_name,
                value: 0.0,
                mute: false,
                slider_min: 0.0,
                slider_max: 1.0,
                relative_key: None,
                vertex_group: None,
                frame: 0.0,
                positions,
            },
        );
        if !keep_modifier
            && let Some(node) = engine.doc.nodes.get_mut(&target)
            && let Some(index) = node
                .modifiers
                .iter()
                .position(|modifier| modifier.id == modifier_id)
        {
            node.modifiers.drain(..=index);
            engine.mark("nodes", &target, ChangeKind::Updated);
        }
        engine.mark("data_blocks", &original_data_id, ChangeKind::Updated);
    }
    Ok(true)
}

fn same_topology(left: &crate::geom::Mesh, right: &crate::geom::Mesh) -> bool {
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
fn apply_lattice_modifier(
    mesh: &mut crate::geom::Mesh,
    source_data: &DataBlock,
    target_id: &Id,
    modifier: &Modifier,
    doc: &SceneDoc,
    world_matrices: &BTreeMap<Id, glam::DMat4>,
) -> Result<()> {
    let lattice_text = modifier
        .params
        .get("object")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::InvalidOperation,
                "lattice modifier requires an object ID",
                json!({"modifier_id":modifier.id,"parameter":"object"}),
            )
        })?;
    let lattice_id = Id::new(lattice_text).map_err(|error| {
        PotError::with_details(
            ErrorCode::InvalidOperation,
            error.message,
            json!({"modifier_id":modifier.id,"object":lattice_text}),
        )
    })?;
    let lattice_node = doc.nodes.get(&lattice_id).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::TargetNotFound,
            "lattice modifier target Object was not found",
            json!({"modifier_id":modifier.id,"object":lattice_id}),
        )
    })?;
    let lattice_data_id = lattice_node.data.as_ref().ok_or_else(|| {
        PotError::with_details(
            ErrorCode::InvalidOperation,
            "lattice modifier target has no Data-Block",
            json!({"modifier_id":modifier.id,"object":lattice_id}),
        )
    })?;
    let lattice_block = doc.data_blocks.get(lattice_data_id).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::TargetNotFound,
            "lattice modifier Data-Block was not found",
            json!({"modifier_id":modifier.id,"data_id":lattice_data_id}),
        )
    })?;
    let lattice_data = lattice_block.lattice.as_ref().ok_or_else(|| {
        PotError::with_details(
            ErrorCode::InvalidOperation,
            "lattice modifier target is not a lattice Data-Block",
            json!({"modifier_id":modifier.id,"data_id":lattice_data_id}),
        )
    })?;
    let target_world = world_matrices.get(target_id).copied().ok_or_else(|| {
        PotError::with_details(
            ErrorCode::EvaluationFailed,
            "lattice modifier source transform is unavailable",
            json!({"node_id":target_id}),
        )
    })?;
    let lattice_world = world_matrices.get(&lattice_id).copied().ok_or_else(|| {
        PotError::with_details(
            ErrorCode::EvaluationFailed,
            "lattice modifier target transform is unavailable",
            json!({"node_id":lattice_id}),
        )
    })?;
    let vertex_group = match modifier.params.get("vertex_group") {
        None => None,
        Some(Value::String(name)) => {
            let mut matching = source_data.vertex_groups.iter().filter(|group| {
                group.id.as_str() == name.as_str() || group.name.as_str() == name.as_str()
            });
            let group = matching.next().ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::TargetNotFound,
                    "lattice modifier vertex group was not found",
                    json!({"modifier_id":modifier.id,"vertex_group":name}),
                )
            })?;
            if matching.next().is_some() {
                return Err(PotError::with_details(
                    ErrorCode::AmbiguousTarget,
                    "lattice modifier vertex group name is ambiguous",
                    json!({"modifier_id":modifier.id,"vertex_group":name}),
                ));
            }
            Some(&group.id)
        }
        Some(_) => {
            return Err(PotError::with_details(
                ErrorCode::InvalidOperation,
                "lattice modifier vertex_group must be a string",
                json!({"modifier_id":modifier.id}),
            ));
        }
    };
    let weights = vertex_group.map(|group_id| {
        mesh.vertices
            .iter()
            .map(|vertex| {
                source_data
                    .vertex_weights
                    .get(&vertex.id)
                    .and_then(|weights| weights.get(group_id))
                    .copied()
                    .unwrap_or(0.0)
            })
            .collect::<Vec<_>>()
    });
    *mesh = crate::geom::lattice::deform_mesh_with_transform(
        mesh,
        lattice_data,
        weights.as_deref(),
        target_world.inverse() * lattice_world,
    )?;
    Ok(())
}

fn targets(engine: &Engine<'_>, operation: &Map<String, Value>) -> Result<Vec<Id>> {
    let target = operation.get("target").ok_or_else(|| {
        engine.error(ErrorCode::InvalidOperation, "target is required", "/target")
    })?;
    resolve_node_targets(engine, target, true)
}

fn read_id(engine: &Engine<'_>, operation: &Map<String, Value>, field: &str) -> Result<Id> {
    let value = read_string(engine, operation, field)?;
    Id::new(value).map_err(|_| {
        engine.error(
            ErrorCode::InvalidOperation,
            "invalid ID",
            &format!("/{field}"),
        )
    })
}

fn read_modifier_id(engine: &Engine<'_>, operation: &Map<String, Value>) -> Result<Id> {
    if operation.contains_key("id") && operation.contains_key("modifier_id") {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "provide either id or modifier_id, not both",
            "/modifier_id",
        ));
    }
    let field = if operation.contains_key("modifier_id") {
        "modifier_id"
    } else {
        "id"
    };
    read_id(engine, operation, field)
}
fn read_params(
    engine: &Engine<'_>,
    value: Option<&Value>,
    field: &str,
) -> Result<Map<String, Value>> {
    match value {
        None => Ok(Map::new()),
        Some(Value::Object(params)) => Ok(params.clone()),
        Some(_) => Err(engine.error(
            ErrorCode::InvalidOperation,
            "params must be an object",
            &format!("/{field}"),
        )),
    }
}
fn validate_parameter_map(
    engine: &Engine<'_>,
    family: ParameterFamily,
    kind: &str,
    params: &Map<String, Value>,
    base_pointer: &str,
) -> Result<()> {
    params::validate_params(family, kind, params, base_pointer).map_err(|error| {
        let pointer = operation_pointer(
            engine.operation_index,
            error.pointer.trim_start_matches('/'),
        );
        engine.error(ErrorCode::InvalidArgument, error.message, &pointer)
    })
}
fn validate_modifier_resource(
    engine: &Engine<'_>,
    modifier_type: &str,
    params: &Map<String, Value>,
    base_pointer: &str,
) -> Result<()> {
    let has_resource_parameter = params::type_spec(ParameterFamily::Modifier, modifier_type)
        .and_then(|specification| specification.get("properties"))
        .and_then(Value::as_object)
        .is_some_and(|properties| properties.contains_key("resource"));
    if !has_resource_parameter {
        return Ok(());
    }
    let resource_pointer = operation_pointer(
        engine.operation_index,
        &format!("{}/resource", base_pointer.trim_start_matches('/')),
    );
    let resource_text = params
        .get("resource")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidArgument,
                "resource must be a Potter resource ID",
                &resource_pointer,
            )
        })?;
    let resource_id = Id::new(resource_text.to_owned()).map_err(|_| {
        engine.error(
            ErrorCode::InvalidArgument,
            "resource must be a valid Potter ID",
            &resource_pointer,
        )
    })?;
    if !engine.doc.resources.contains_key(&resource_id) {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            format!("resource `{resource_id}` was not found"),
            &resource_pointer,
        ));
    }
    if modifier_type == "mesh_sequence_cache" {
        let valid_path = params
            .get("object_path")
            .and_then(Value::as_str)
            .is_some_and(|path| {
                path.strip_prefix('/').is_some_and(|tail| {
                    !tail.is_empty()
                        && tail
                            .split('/')
                            .all(|segment| !segment.is_empty() && segment != "." && segment != "..")
                })
            });
        if !valid_path {
            return Err(engine.error(
                ErrorCode::InvalidArgument,
                "object_path must be an absolute Alembic object path",
                &format!("{base_pointer}/object_path"),
            ));
        }
    }
    Ok(())
}
fn is_attribute_modifier_type(modifier_type: &str) -> bool {
    matches!(
        modifier_type,
        "vertex_weight_edit"
            | "vertex_weight_mix"
            | "vertex_weight_proximity"
            | "weighted_normal"
            | "normal_edit"
            | "uv_project"
            | "uv_warp"
            | "data_transfer"
    )
}

fn validate_attribute_modifier_params(
    modifier_type: &str,
    params: &Map<String, Value>,
) -> Result<()> {
    let fields: &[&str] = match modifier_type {
        "vertex_weight_edit" => &[
            "vertex_group",
            "default_weight",
            "use_add",
            "add_threshold",
            "use_remove",
            "remove_threshold",
            "normalize",
            "falloff_type",
            "invert_falloff",
            "map_curve",
            "mask_constant",
            "mask_vertex_group",
            "invert_mask_vertex_group",
            "mask_texture",
            "mask_tex_use_channel",
            "mask_tex_mapping",
            "mask_tex_uv_layer",
            "mask_tex_map_object",
        ],
        "vertex_weight_mix" => &[
            "vertex_group_a",
            "vertex_group_b",
            "invert_vertex_group_a",
            "invert_vertex_group_b",
            "default_weight_a",
            "default_weight_b",
            "mix_mode",
            "mix_set",
            "normalize",
            "mask_constant",
            "mask_vertex_group",
            "invert_mask_vertex_group",
            "mask_texture",
            "mask_tex_use_channel",
            "mask_tex_mapping",
            "mask_tex_uv_layer",
            "mask_tex_map_object",
        ],
        "vertex_weight_proximity" => &[
            "target",
            "vertex_group",
            "proximity_mode",
            "proximity_geometry",
            "min_dist",
            "max_dist",
            "falloff_type",
            "invert_falloff",
            "normalize",
            "map_curve",
            "mask_constant",
            "mask_vertex_group",
            "invert_mask_vertex_group",
            "mask_texture",
            "mask_tex_use_channel",
            "mask_tex_mapping",
            "mask_tex_uv_layer",
            "mask_tex_map_object",
        ],
        "weighted_normal" => &[
            "mode",
            "weight",
            "thresh",
            "keep_sharp",
            "vertex_group",
            "invert_vertex_group",
            "use_face_influence",
        ],
        "normal_edit" => &[
            "mode",
            "target",
            "offset",
            "mix_mode",
            "mix_factor",
            "mix_limit",
            "no_polynors_fix",
            "vertex_group",
            "invert_vertex_group",
            "use_direction_parallel",
        ],
        "uv_project" => &[
            "projectors",
            "aspect_x",
            "aspect_y",
            "scale_x",
            "scale_y",
            "uv_layer",
        ],
        "uv_warp" => &[
            "center",
            "axis_u",
            "axis_v",
            "object_from",
            "object_to",
            "bone_from",
            "bone_to",
            "offset",
            "scale",
            "rotation",
            "uv_layer",
            "vertex_group",
            "invert_vertex_group",
        ],
        "data_transfer" => &[
            "object",
            "use_object_transform",
            "use_vert_data",
            "use_edge_data",
            "use_loop_data",
            "use_poly_data",
            "data_types_verts",
            "data_types_edges",
            "data_types_loops",
            "data_types_polys",
            "vert_mapping",
            "edge_mapping",
            "loop_mapping",
            "poly_mapping",
            "mix_mode",
            "mix_factor",
            "vertex_group",
            "invert_vertex_group",
            "layers_uv_select_src",
            "layers_uv_select_dst",
        ],
        _ => return Ok(()),
    };
    if let Some(field) = params
        .keys()
        .find(|field| !fields.contains(&field.as_str()))
    {
        return Err(invalid_attribute_modifier_parameter(
            modifier_type,
            field,
            "a modeled Blender RNA parameter",
        ));
    }
    let required: &[&str] = match modifier_type {
        "vertex_weight_edit" => &["vertex_group"],
        "vertex_weight_mix" => &["vertex_group_a", "vertex_group_b"],
        "vertex_weight_proximity" => &["target", "vertex_group"],
        "normal_edit" => &["target"],
        "uv_project" => &["projectors"],
        "data_transfer" => &["object"],
        _ => &[],
    };
    if let Some(field) = required.iter().find(|field| !params.contains_key(**field)) {
        return Err(invalid_attribute_modifier_parameter(
            modifier_type,
            field,
            "a required Blender RNA parameter",
        ));
    }
    let enum_fields: &[(&str, &[&str])] = match modifier_type {
        "vertex_weight_edit" => &[
            (
                "falloff_type",
                &[
                    "LINEAR",
                    "CURVE",
                    "SHARP",
                    "SMOOTH",
                    "ROOT",
                    "ICON_SPHERECURVE",
                    "RANDOM",
                    "STEP",
                ],
            ),
            (
                "mask_tex_use_channel",
                &["INT", "RED", "GREEN", "BLUE", "HUE", "SAT", "VAL", "ALPHA"],
            ),
            ("mask_tex_mapping", &["LOCAL", "GLOBAL", "OBJECT", "UV"]),
        ],
        "vertex_weight_mix" => &[
            (
                "mix_mode",
                &[
                    "SET", "ADD", "SUB", "MUL", "DIV", "DIF", "AVG", "MIN", "MAX",
                ],
            ),
            ("mix_set", &["ALL", "A", "B", "OR", "AND"]),
            (
                "mask_tex_use_channel",
                &["INT", "RED", "GREEN", "BLUE", "HUE", "SAT", "VAL", "ALPHA"],
            ),
            ("mask_tex_mapping", &["LOCAL", "GLOBAL", "OBJECT", "UV"]),
        ],
        "vertex_weight_proximity" => &[
            ("proximity_mode", &["OBJECT", "GEOMETRY"]),
            (
                "falloff_type",
                &[
                    "LINEAR",
                    "CURVE",
                    "SHARP",
                    "SMOOTH",
                    "ROOT",
                    "ICON_SPHERECURVE",
                    "RANDOM",
                    "STEP",
                ],
            ),
            (
                "mask_tex_use_channel",
                &["INT", "RED", "GREEN", "BLUE", "HUE", "SAT", "VAL", "ALPHA"],
            ),
            ("mask_tex_mapping", &["LOCAL", "GLOBAL", "OBJECT", "UV"]),
        ],
        "weighted_normal" => &[(
            "mode",
            &["FACE_AREA", "CORNER_ANGLE", "FACE_AREA_WITH_ANGLE"],
        )],
        "normal_edit" => &[
            ("mode", &["RADIAL", "DIRECTIONAL"]),
            ("mix_mode", &["COPY", "ADD", "SUB", "MUL"]),
        ],
        "uv_warp" => &[("axis_u", &["X", "Y", "Z"]), ("axis_v", &["X", "Y", "Z"])],
        "data_transfer" => &[
            (
                "vert_mapping",
                &[
                    "TOPOLOGY",
                    "NEAREST",
                    "POLYINTERP_NEAREST",
                    "POLYINTERP_VNORPROJ",
                ],
            ),
            (
                "edge_mapping",
                &["TOPOLOGY", "NEAREST", "VERT_NEAREST", "POLY_NEAREST"],
            ),
            (
                "loop_mapping",
                &[
                    "TOPOLOGY",
                    "NEAREST_POLYNOR",
                    "POLYINTERP_NEAREST",
                    "POLYINTERP_LNORPROJ",
                ],
            ),
            (
                "poly_mapping",
                &["TOPOLOGY", "NEAREST", "POLYINTERP_PNORPROJ"],
            ),
            (
                "mix_mode",
                &[
                    "REPLACE",
                    "ABOVE_THRESHOLD",
                    "BELOW_THRESHOLD",
                    "MIX",
                    "ADD",
                    "SUB",
                    "MUL",
                ],
            ),
            (
                "layers_uv_select_src",
                &["ACTIVE", "ALL", "BONE_SELECT", "BONE_DEFORM"],
            ),
            ("layers_uv_select_dst", &["ACTIVE", "NAME", "INDEX"]),
        ],
        _ => &[],
    };
    for (field, values) in enum_fields {
        if params.get(*field).is_some_and(|value| {
            !value
                .as_str()
                .is_some_and(|identifier| values.contains(&identifier))
        }) {
            return Err(invalid_attribute_modifier_parameter(
                modifier_type,
                field,
                "a valid Blender enum identifier",
            ));
        }
    }
    for (field, values) in [
        ("proximity_geometry", &["VERTEX", "EDGE", "FACE"][..]),
        (
            "data_types_verts",
            &["VGROUP_WEIGHTS", "BEVEL_WEIGHT_VERT", "COLOR_VERTEX"][..],
        ),
        ("data_types_edges", &["SHARP_EDGE", "SEAM", "CREASE"][..]),
        (
            "data_types_loops",
            &["CUSTOM_NORMAL", "UV", "COLOR_CORNER"][..],
        ),
        ("data_types_polys", &["SMOOTH", "FREESTYLE_FACE"][..]),
    ] {
        if let Some(value) = params.get(field) {
            let Some(items) = value.as_array() else {
                return Err(invalid_attribute_modifier_parameter(
                    modifier_type,
                    field,
                    "an array of Blender enum identifiers",
                ));
            };
            if items
                .iter()
                .any(|item| !item.as_str().is_some_and(|item| values.contains(&item)))
            {
                return Err(invalid_attribute_modifier_parameter(
                    modifier_type,
                    field,
                    "an array of supported Blender enum identifiers",
                ));
            }
        }
    }
    for field in [
        "vertex_group",
        "vertex_group_a",
        "vertex_group_b",
        "mask_vertex_group",
        "uv_layer",
        "bone_from",
        "bone_to",
        "mask_tex_uv_layer",
    ] {
        if params.get(field).is_some_and(|value| !value.is_string()) {
            return Err(invalid_attribute_modifier_parameter(
                modifier_type,
                field,
                "a string",
            ));
        }
    }
    for field in [
        "invert_vertex_group",
        "invert_vertex_group_a",
        "invert_vertex_group_b",
        "use_add",
        "use_remove",
        "normalize",
        "invert_falloff",
        "invert_mask_vertex_group",
        "use_face_influence",
        "keep_sharp",
        "no_polynors_fix",
        "use_direction_parallel",
        "invert_mask_vertex_group",
        "invert_vertex_group",
        "use_object_transform",
        "use_vert_data",
        "use_edge_data",
        "use_loop_data",
        "use_poly_data",
    ] {
        if params.get(field).is_some_and(|value| !value.is_boolean()) {
            return Err(invalid_attribute_modifier_parameter(
                modifier_type,
                field,
                "a boolean",
            ));
        }
    }
    for field in [
        "default_weight",
        "add_threshold",
        "remove_threshold",
        "default_weight_a",
        "default_weight_b",
        "min_dist",
        "max_dist",
        "weight",
        "thresh",
        "mix_factor",
        "mix_limit",
        "aspect_x",
        "aspect_y",
        "scale_x",
        "scale_y",
        "rotation",
        "mask_constant",
    ] {
        if params
            .get(field)
            .is_some_and(|value| !value.as_f64().is_some_and(f64::is_finite))
        {
            return Err(invalid_attribute_modifier_parameter(
                modifier_type,
                field,
                "a finite number",
            ));
        }
    }
    for (field, length) in [
        ("offset", if modifier_type == "normal_edit" { 3 } else { 2 }),
        ("center", 2),
        ("scale", 2),
    ] {
        if let Some(value) = params.get(field) {
            let valid = value.as_array().is_some_and(|items| {
                items.len() == length
                    && items
                        .iter()
                        .all(|item| item.as_f64().is_some_and(f64::is_finite))
            });
            if !valid {
                return Err(invalid_attribute_modifier_parameter(
                    modifier_type,
                    field,
                    "a finite Blender vector",
                ));
            }
        }
    }
    for field in [
        "target",
        "object",
        "object_from",
        "object_to",
        "mask_tex_map_object",
    ] {
        if let Some(value) = params.get(field) {
            let identifier = value.as_str().ok_or_else(|| {
                invalid_attribute_modifier_parameter(modifier_type, field, "a Potter object ID")
            })?;
            Id::new(identifier).map_err(|_| {
                invalid_attribute_modifier_parameter(
                    modifier_type,
                    field,
                    "a valid Potter object ID",
                )
            })?;
        }
    }
    if let Some(value) = params.get("mask_texture").filter(|value| !value.is_null()) {
        let identifier = value.as_str().ok_or_else(|| {
            invalid_attribute_modifier_parameter(
                modifier_type,
                "mask_texture",
                "an image ID or null",
            )
        })?;
        Id::new(identifier).map_err(|_| {
            invalid_attribute_modifier_parameter(modifier_type, "mask_texture", "a valid image ID")
        })?;
    }
    if let Some(value) = params.get("projectors") {
        let items = value.as_array().ok_or_else(|| {
            invalid_attribute_modifier_parameter(
                modifier_type,
                "projectors",
                "an array of object IDs",
            )
        })?;
        for item in items {
            let identifier = item.as_str().ok_or_else(|| {
                invalid_attribute_modifier_parameter(
                    modifier_type,
                    "projectors",
                    "an array of object IDs",
                )
            })?;
            Id::new(identifier).map_err(|_| {
                invalid_attribute_modifier_parameter(
                    modifier_type,
                    "projectors",
                    "valid Potter object IDs",
                )
            })?;
        }
    }
    if let Some(value) = params.get("map_curve") {
        let points = value.as_array().ok_or_else(|| {
            invalid_attribute_modifier_parameter(
                modifier_type,
                "map_curve",
                "an array of [x,y] points",
            )
        })?;
        if points.len() < 2
            || points.iter().any(|point| {
                point.as_array().is_none_or(|pair| {
                    pair.len() != 2
                        || pair.iter().any(|value| {
                            !value.as_f64().is_some_and(|value| {
                                value.is_finite() && (0.0..=1.0).contains(&value)
                            })
                        })
                })
            })
        {
            return Err(invalid_attribute_modifier_parameter(
                modifier_type,
                "map_curve",
                "at least two [x,y] points in the unit square",
            ));
        }
    }
    Ok(())
}

fn invalid_attribute_modifier_parameter(
    modifier_type: &str,
    parameter: &str,
    expected: &str,
) -> PotError {
    PotError::with_details(
        ErrorCode::InvalidOperation,
        format!("modifier `{modifier_type}` parameter `{parameter}` must be {expected}"),
        json!({"feature_id":format!("modifier.{modifier_type}"),"parameter":parameter,"expected":expected}),
    )
}

fn is_physics_modifier(modifier_type: &str) -> bool {
    matches!(
        modifier_type,
        "cloth" | "soft_body" | "collision" | "dynamic_paint" | "fluid" | "particle_system"
    )
}

fn validate_volume_displace_params(engine: &Engine<'_>, params: &Map<String, Value>) -> Result<()> {
    const FIELDS: &[&str] = &[
        "strength",
        "texture",
        "texture_map_mode",
        "texture_mid_level",
        "texture_sample_radius",
        "texture_map_object",
    ];
    if let Some(field) = params
        .keys()
        .find(|field| !FIELDS.contains(&field.as_str()))
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("unknown Volume Displace parameter `{field}`"),
            &format!("/params/{field}"),
        ));
    }
    for field in ["strength", "texture_sample_radius"] {
        if params
            .get(field)
            .is_some_and(|value| !value.as_f64().is_some_and(f64::is_finite))
        {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("Volume Displace `{field}` must be finite"),
                &format!("/params/{field}"),
            ));
        }
    }
    if let Some(mid_level) = params.get("texture_mid_level")
        && mid_level.as_array().is_none_or(|values| {
            values.len() != 3
                || values
                    .iter()
                    .any(|value| !value.as_f64().is_some_and(f64::is_finite))
        })
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "Volume Displace texture_mid_level must contain three finite values",
            "/params/texture_mid_level",
        ));
    }
    if params
        .get("texture_sample_radius")
        .and_then(Value::as_f64)
        .is_some_and(|value| value < 0.0)
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "Volume Displace texture_sample_radius must be non-negative",
            "/params/texture_sample_radius",
        ));
    }
    if let Some(mode) = params.get("texture_map_mode")
        && !matches!(mode.as_str(), Some("LOCAL" | "GLOBAL" | "OBJECT"))
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "Volume Displace texture_map_mode must be LOCAL, GLOBAL, or OBJECT",
            "/params/texture_map_mode",
        ));
    }
    let mapping_mode = params
        .get("texture_map_mode")
        .and_then(Value::as_str)
        .unwrap_or("LOCAL");
    if mapping_mode == "OBJECT" && !params.contains_key("texture_map_object") {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "Volume Displace OBJECT mapping requires texture_map_object",
            "/params/texture_map_object",
        ));
    }
    if let Some(object) = params.get("texture_map_object") {
        let Some(object_text) = object.as_str() else {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "Volume Displace texture_map_object must be an object ID",
                "/params/texture_map_object",
            ));
        };
        let object_id = Id::new(object_text.to_owned()).map_err(|error| {
            engine.error(
                ErrorCode::InvalidOperation,
                error.message,
                "/params/texture_map_object",
            )
        })?;
        if !engine.doc.nodes.contains_key(&object_id) {
            return Err(engine.error(
                ErrorCode::TargetNotFound,
                "Volume Displace texture_map_object was not found",
                "/params/texture_map_object",
            ));
        }
    }
    if let Some(texture) = params.get("texture") {
        let unsupported = || {
            PotError::with_details(
                ErrorCode::UnsupportedFeature,
                "Volume Displace supports only image or deterministic noise textures",
                json!({
                    "feature_id":"modifier.volume_displace.texture_type",
                    "texture":texture,
                    "reason":"unsupported texture source",
                    "status":"not_supported"
                }),
            )
        };
        let Some(texture_id) = texture.as_str() else {
            return Err(unsupported());
        };
        if texture_id != "noise" {
            let Ok(image_id) = Id::new(texture_id.to_owned()) else {
                return Err(unsupported());
            };
            if !engine.doc.images.contains_key(&image_id) {
                return Err(unsupported());
            }
        }
    }
    Ok(())
}

fn validate_particle_modifier_params(
    engine: &Engine<'_>,
    modifier_type: &str,
    params: &Map<String, Value>,
) -> Result<()> {
    let allowed: &[&str] = match modifier_type {
        "particle_instance" => &[
            "object",
            "particle_system_index",
            "use_normal",
            "use_children",
            "use_size",
            "show_alive",
            "show_dead",
            "show_unborn",
            "position",
            "random_position",
            "axis",
            "rotation",
            "random_rotation",
            "space",
            "use_path",
        ],
        "explode" => &[
            "particle_system_index",
            "use_edge_cut",
            "show_alive",
            "show_dead",
            "show_unborn",
            "use_size",
            "protect",
            "vertex_group",
        ],
        _ => return Ok(()),
    };
    if let Some(field) = params
        .keys()
        .find(|field| !allowed.contains(&field.as_str()))
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("unknown {modifier_type} parameter `{field}`"),
            &format!("/params/{field}"),
        ));
    }
    if modifier_type == "particle_instance" {
        let object = params
            .get("object")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "particle_instance requires an object ID",
                    "/params/object",
                )
            })?;
        let object_id = Id::new(object.to_owned()).map_err(|error| {
            engine.error(ErrorCode::InvalidOperation, error.message, "/params/object")
        })?;
        let source = engine.doc.nodes.get(&object_id).ok_or_else(|| {
            engine.error(
                ErrorCode::TargetNotFound,
                "particle system object was not found",
                "/params/object",
            )
        })?;
        if !source.properties.contains_key("physics_particle_emitter") {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "particle_instance object has no particle system settings",
                "/params/object",
            ));
        }
    }
    if let Some(value) = params.get("particle_system_index") {
        let index = value.as_u64().ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "particle_system_index must be a positive integer",
                "/params/particle_system_index",
            )
        })?;
        if modifier_type == "particle_instance" {
            let object_id = params
                .get("object")
                .and_then(Value::as_str)
                .and_then(|value| Id::new(value.to_owned()).ok())
                .ok_or_else(|| {
                    engine.error(
                        ErrorCode::InvalidOperation,
                        "particle_instance requires an object ID",
                        "/params/object",
                    )
                })?;
            let system_count = engine.doc.nodes.get(&object_id).map_or(0, |node| {
                node.modifiers
                    .iter()
                    .filter(|modifier| modifier.modifier_type == "particle_system")
                    .count()
            });
            if usize::try_from(index).map_or(true, |index| index == 0 || index > system_count) {
                return Err(engine.error(
                    ErrorCode::TargetNotFound,
                    "particle system index does not exist on the referenced object",
                    "/params/particle_system_index",
                ));
            }
        }
    }
    for field in [
        "use_normal",
        "use_children",
        "use_size",
        "show_alive",
        "show_dead",
        "show_unborn",
        "use_path",
        "use_edge_cut",
    ] {
        if params.get(field).is_some_and(|value| !value.is_boolean()) {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("{modifier_type} `{field}` must be a boolean"),
                &format!("/params/{field}"),
            ));
        }
    }
    for field in [
        "position",
        "random_position",
        "protect",
        "rotation",
        "random_rotation",
    ] {
        if params
            .get(field)
            .is_some_and(|value| !value.as_f64().is_some_and(f64::is_finite))
        {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("{modifier_type} `{field}` must be finite"),
                &format!("/params/{field}"),
            ));
        }
    }
    if params
        .get("position")
        .is_some_and(|value| value.as_i64().is_none() && value.as_u64().is_none())
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "particle_instance position must be an integer",
            "/params/position",
        ));
    }
    if params
        .get("random_position")
        .and_then(Value::as_f64)
        .is_some_and(|value| !(0.0..=1.0).contains(&value))
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "particle_instance random_position must be in [0, 1]",
            "/params/random_position",
        ));
    }
    if params
        .get("random_rotation")
        .and_then(Value::as_f64)
        .is_some_and(|value| !(0.0..=1.0).contains(&value))
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "particle_instance random_rotation must be in [0, 1]",
            "/params/random_rotation",
        ));
    }
    if params
        .get("protect")
        .and_then(Value::as_f64)
        .is_some_and(|value| !(0.0..=1.0).contains(&value))
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "explode protect must be in [0, 1]",
            "/params/protect",
        ));
    }
    if let Some(axis) = params.get("axis")
        && !matches!(axis.as_str(), Some("X" | "Y" | "Z"))
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "particle_instance axis must be X, Y, or Z",
            "/params/axis",
        ));
    }
    if let Some(space) = params.get("space")
        && !matches!(space.as_str(), Some("WORLD" | "LOCAL"))
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "particle_instance space must be WORLD or LOCAL",
            "/params/space",
        ));
    }
    for field in ["vertex_group"] {
        if params
            .get(field)
            .is_some_and(|value| value.as_str().is_none_or(str::is_empty))
        {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("{modifier_type} `{field}` must be a non-empty group name"),
                &format!("/params/{field}"),
            ));
        }
    }
    Ok(())
}

fn unsupported(_engine: &Engine<'_>, modifier_type: &str) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        format!("modifier `{modifier_type}` is not supported"),
        json!({"feature_id":format!("modifier.{modifier_type}")}),
    )
}
