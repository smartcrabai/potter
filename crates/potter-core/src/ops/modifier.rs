use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet};

use crate::{
    error::{ErrorCode, PotError, Result},
    eval::EvaluationPurpose,
    model::{Id, Modifier, ShapeKey, ShapeKeyData},
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
            "surface_deform" | "mesh_deform" | "laplacian_deform" | "corrective_smooth"
        ) {
            return Err(PotError::invalid_operation(
                "modifier.bind supports surface_deform, mesh_deform, laplacian_deform, and corrective_smooth",
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
        let bind_node = engine.doc.nodes.get(&subject_id).ok_or_else(|| {
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
        let keep = if modifier.modifier_type == "corrective_smooth" {
            bind_index
        } else {
            bind_index + 1
        };
        let bind_nodes = BTreeSet::from([subject_id.clone()]);
        let bind_snapshot = crate::eval::Snapshot::evaluate_nodes_with_modifier_prefix_with_cache(
            engine.doc,
            &crate::eval::EvaluationContext {
                scene_id: None,
                view_layer: None,
                frame: Some(engine.evaluation_frame),
                purpose: EvaluationPurpose::Preview,
            },
            engine.asset_root,
            None,
            &bind_nodes,
            &subject_id,
            keep,
        )?;
        let source_bind_mesh = bind_snapshot
            .meshes
            .get(&subject_id)
            .cloned()
            .ok_or_else(|| PotError::invalid_operation("evaluated source mesh is unavailable"))?;
        let binding = if modifier.modifier_type == "corrective_smooth" {
            let coords = source_bind_mesh
                .vertices
                .iter()
                .map(|vertex| json!([vertex.co.x, vertex.co.y, vertex.co.z]))
                .collect::<Vec<_>>();
            json!({"format":"blender_corrective_bind_v1","coords":coords})
        } else if modifier.modifier_type == "laplacian_deform" {
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
            let target_mesh = bind_snapshot
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
            if modifier.modifier_type == "surface_deform" {
                crate::geom::modifiers::deform::bind_surface(
                    &source_bind_mesh,
                    &target_mesh,
                    to_subject,
                    &modifier,
                )?
            } else {
                crate::geom::modifiers::deform::bind_mesh(
                    &source_bind_mesh,
                    &target_mesh,
                    subject_world,
                    target_world,
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
        &[
            "op",
            "target",
            "id",
            "type",
            "name",
            "enabled",
            "show_viewport",
            "show_render",
            "params",
        ],
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
    let show_viewport = read_bool(engine, operation, "show_viewport", true)?;
    let show_render = read_bool(engine, operation, "show_render", true)?;
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
        validate_attribute_modifier_params(engine, &modifier_type, &params, "/params")?;
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
        let index = insertion_index(&node.modifiers, &modifier_type);
        let pinned = pin_flag(&modifier_params);
        node.modifiers.insert(
            index,
            Modifier {
                id: id.clone(),
                modifier_type: modifier_type.clone(),
                name: name.clone(),
                enabled,
                show_viewport,
                show_render,
                params: modifier_params,
                binding_data: None,
                runtime: crate::model::ModifierRuntime::default(),
            },
        );
        if pinned {
            settle_pin(&mut node.modifiers, index);
        }
        engine.mark("nodes", &target, ChangeKind::Updated);
        changed = true;
    }
    Ok(changed)
}

/// Mesh Deform settings that Blender reads at bind time. A bound modifier rejects a change to any of them, so
/// they change only after an unbind.
const MESH_DEFORM_BIND_SETTINGS: [&str; 3] = ["use_dynamic_bind", "precision", "object"];

/// Rejects a change to a Mesh Deform bind setting while the modifier is bound. A setting keeps its value when
/// it is equal, or when a null `object` replaces an absent one.
fn check_bound_mesh_deform_params(
    engine: &Engine<'_>,
    existing: &Modifier,
    modifier_type: &str,
    params: &Map<String, Value>,
) -> Result<()> {
    if existing.binding_data.is_none()
        || existing.modifier_type != "mesh_deform"
        || modifier_type != "mesh_deform"
    {
        return Ok(());
    }
    for key in MESH_DEFORM_BIND_SETTINGS {
        let before = existing.params.get(key).filter(|value| !value.is_null());
        let after = params.get(key).filter(|value| !value.is_null());
        let unchanged = match (before, after) {
            (Some(before), Some(after)) => {
                before == after || (before.is_number() && before.as_f64() == after.as_f64())
            }
            (None, None) => true,
            _ => false,
        };
        if !unchanged {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!(
                    "Mesh Deform `{key}` cannot change while the modifier is bound; unbind the modifier first"
                ),
                &format!("/set/params/{key}"),
            ));
        }
    }
    Ok(())
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
        if ![
            "name",
            "enabled",
            "show_viewport",
            "show_render",
            "params",
            "type",
        ]
        .contains(&key.as_str())
        {
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
        if let Some(params) = &params {
            check_bound_mesh_deform_params(engine, &existing, modifier_type, params)?;
        }
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
                validate_attribute_modifier_params(engine, modifier_type, &params, "/set/params")?;
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
        if let Some(value) = set.get("show_viewport") {
            modifier.show_viewport = value.as_bool().ok_or_else(|| {
                PotError::new(
                    ErrorCode::InvalidOperation,
                    "modifier show_viewport must be a boolean",
                )
            })?;
        }
        if let Some(value) = set.get("show_render") {
            modifier.show_render = value.as_bool().ok_or_else(|| {
                PotError::new(
                    ErrorCode::InvalidOperation,
                    "modifier show_render must be a boolean",
                )
            })?;
        }
        modifier_type.clone_into(&mut modifier.modifier_type);
        if let Some(params) = modifier_params {
            modifier.params = params;
        }
        let pin_changed = pin_flag(&before.params) != pin_flag(&modifier.params);
        let modified = modifier != &before;
        if pin_changed
            && let Some(index) = node
                .modifiers
                .iter()
                .position(|candidate| candidate.id == id)
        {
            settle_pin(&mut node.modifiers, index);
        }
        if modified {
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
            let reached = partial_move_target(&node.modifiers, current, index);
            if reached != index {
                return Err(engine.error(
                    ErrorCode::InvalidOperation,
                    format!(
                        "modifier `{id}` cannot move to index {index}: pin-to-last or original-data order stops it at index {reached}"
                    ),
                    "/index",
                ));
            }
            move_modifier(&mut node.modifiers, current, index);
            engine.mark("nodes", &target, ChangeKind::Updated);
            changed = true;
        }
    }
    Ok(changed)
}

/// Blender's `ModifierTypeType::OnlyDeform` for each Potter modifier type.
fn only_deform(modifier_type: &str) -> bool {
    matches!(
        modifier_type,
        "armature"
            | "cast"
            | "cloth"
            | "collision"
            | "corrective_smooth"
            | "curve"
            | "displace"
            | "hook"
            | "laplacian_deform"
            | "laplacian_smooth"
            | "lattice"
            | "mesh_cache"
            | "mesh_deform"
            | "particle_system"
            | "shrinkwrap"
            | "simple_deform"
            | "smooth"
            | "soft_body"
            | "surface_deform"
            | "warp"
            | "wave"
    )
}

/// Blender's `eModifierTypeFlag_RequiresOriginalData` modifier types.
fn requires_original_data(modifier_type: &str) -> bool {
    matches!(modifier_type, "multires" | "soft_body")
}

/// Blender's `eModifierFlag_PinLast`, stored as the `use_pin_to_last` parameter.
fn pin_flag(params: &Map<String, Value>) -> bool {
    params
        .get("use_pin_to_last")
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn is_pinned(modifier: &Modifier) -> bool {
    pin_flag(&modifier.params)
}

/// Blender's `BKE_modifiers_add_at_end_if_possible`: a new modifier goes before the trailing pinned
/// block. Hook and original-data types go before the first modifier that is not deform-only.
pub(crate) fn insertion_index(stack: &[Modifier], modifier_type: &str) -> usize {
    if modifier_type == "hook" || requires_original_data(modifier_type) {
        let mut index = 0;
        while index < stack.len() && only_deform(&stack[index].modifier_type) {
            if stack.get(index + 1).is_some_and(is_pinned) {
                break;
            }
            index += 1;
        }
        return index;
    }
    let mut index = stack.len();
    while index > 0 && is_pinned(&stack[index - 1]) {
        index -= 1;
    }
    index
}

/// Blender's `object_modifier_check_move_before`: whether `modifier` may move above `previous`.
fn may_move_before(modifier: &Modifier, previous: &Modifier) -> bool {
    let unpins_past_pinned = is_pinned(modifier) && !is_pinned(previous);
    let past_original_data =
        requires_original_data(&previous.modifier_type) && !only_deform(&modifier.modifier_type);
    !unpins_past_pinned && !past_original_data
}

/// Blender's `object_modifier_check_move_after`: whether `modifier` may move below `next`.
fn may_move_after(modifier: &Modifier, next: &Modifier) -> bool {
    let unpins_before_pinned = is_pinned(next) && !is_pinned(modifier);
    let past_non_deform =
        requires_original_data(&modifier.modifier_type) && !only_deform(&next.modifier_type);
    !unpins_before_pinned && !past_non_deform
}

/// Index reached when moving `stack[from]` towards `to`: stops before the first refused step, as
/// Blender's `modifier_move_to_index` does with `allow_partial`. Returns `from` when blocked at once.
fn partial_move_target(stack: &[Modifier], from: usize, to: usize) -> usize {
    let moving = &stack[from];
    let mut current = from;
    if from < to {
        while current < to && may_move_after(moving, &stack[current + 1]) {
            current += 1;
        }
    } else {
        while current > to && may_move_before(moving, &stack[current - 1]) {
            current -= 1;
        }
    }
    current
}

fn move_modifier(stack: &mut Vec<Modifier>, from: usize, to: usize) {
    let modifier = stack.remove(from);
    stack.insert(to, modifier);
}

/// Blender's `rna_Modifier_use_pin_to_last_set` move: a pinned modifier moves down past the
/// unpinned modifiers after it; an unpinned one moves up past the pinned modifiers before it.
fn settle_pin(stack: &mut Vec<Modifier>, index: usize) {
    let mut target = index;
    if is_pinned(&stack[index]) {
        while target + 1 < stack.len() && !is_pinned(&stack[target + 1]) {
            target += 1;
        }
    } else {
        while target > 0 && is_pinned(&stack[target - 1]) {
            target -= 1;
        }
    }
    let reached = partial_move_target(stack, index, target);
    if reached != index {
        move_modifier(stack, index, reached);
    }
}

/// Attribute loss and explicitly marked modifier diagnostics accompany successfully applied output,
/// so each is surfaced as a warning on every apply path instead of failing the operation.
fn is_apply_warning(warning: &PotError) -> bool {
    warning.details.get("feature_id").and_then(Value::as_str) == Some("modifier.attribute_dropped")
        || warning
            .details
            .get("modifier_warning")
            .and_then(Value::as_bool)
            == Some(true)
}

fn ensure_single_user_data(engine: &mut Engine<'_>, targets: &[Id]) -> Result<()> {
    for target in targets {
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
        if super::data_user_count(engine, &data_id) <= 1 {
            continue;
        }
        let copy = engine
            .doc
            .data_blocks
            .get(&data_id)
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
        engine.map_id("data_blocks", &data_id, &new_id);
    }
    Ok(())
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
        ensure_single_user_data(engine, &targets)?;
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
        let (target_modifier, target_modifier_index) =
            modifier_to_apply(engine, &target, &modifier_id)?;
        let (evaluated, material_slots) = evaluate_modifier_output(
            engine,
            &target,
            &data_id,
            &target_modifier,
            target_modifier_index,
            allow_shape_keys,
        )?;
        // The evaluated `vertex_groups` attribute is a view of DataBlock weights keyed by this mesh's
        // IDs. Persist it as the weights and keep the transient attribute out of stored mesh data.
        let mut evaluated = evaluated;
        let persisted_weights = engine
            .doc
            .data_blocks
            .get(&data_id)
            .map(|data| crate::eval::mesh_vertex_weights(&evaluated, &data.vertex_groups))
            .unwrap_or_default();
        evaluated.attributes.remove("vertex_groups");
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
        // Blender `BKE_object_material_from_eval_data` (blenkernel/intern/material.cc): the mesh's
        // material array becomes the evaluated mesh's array; the object's slots then follow its length
        // through `BKE_object_materials_sync_length` (`BKE_object_material_resize`). Slots that
        // already existed keep their link; slots the sync adds have matbits 0, i.e. DATA links.
        // `modifier_apply_obdata` (editors/object/object_modifier.cc) calls this before
        // `BKE_mesh_nomain_to_mesh`.
        let synced_len = material_slots.len();
        if let Some(data) = engine.doc.data_blocks.get_mut(&data_id) {
            data.mesh = Some(evaluated);
            data.materials = material_slots;
            data.vertex_weights = persisted_weights;
            if mesh_changed {
                data.descriptor = None;
            }
        }
        if let Some(node) = engine.doc.nodes.get_mut(&target) {
            node.modifiers.retain(|modifier| modifier.id != modifier_id);
            let old_len = node.materials.len();
            let old_links = (0..old_len)
                .map(|slot| node.material_link(slot))
                .collect::<Vec<_>>();
            node.materials.resize(synced_len, None);
            node.material_links = (0..synced_len)
                .map(|slot| {
                    if slot < old_len {
                        old_links[slot]
                    } else {
                        crate::model::MaterialLink::Data
                    }
                })
                .collect();
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
fn modifier_to_apply(
    engine: &Engine<'_>,
    target: &Id,
    modifier_id: &Id,
) -> Result<(Modifier, usize)> {
    let stack = engine
        .doc
        .nodes
        .get(target)
        .map(|node| node.modifiers.as_slice())
        .unwrap_or_default();
    let index = stack
        .iter()
        .position(|modifier| &modifier.id == modifier_id)
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::TargetNotFound,
                format!("modifier `{modifier_id}` was not found on node `{target}`"),
            )
        })?;
    if stack[index].modifier_type == "boolean" {
        crate::geom::modifiers::validate_boolean_topology_suffix(
            stack,
            index,
            &|position, modifier| {
                if position == index {
                    modifier.enabled
                } else {
                    modifier.is_active(EvaluationPurpose::Preview)
                }
            },
        )?;
    }
    Ok((stack[index].clone(), index))
}

fn evaluate_modifier_output(
    engine: &mut Engine<'_>,
    target: &Id,
    data_id: &Id,
    target_modifier: &Modifier,
    target_modifier_index: usize,
    allow_shape_keys: bool,
) -> Result<(crate::geom::Mesh, Vec<Option<Id>>)> {
    let subject_name = engine
        .doc
        .nodes
        .get(target)
        .map(|node| node.name.as_str())
        .ok_or_else(|| PotError::new(ErrorCode::TargetNotFound, "modifier subject is missing"))?;
    let modifier_id = &target_modifier.id;
    let modifier_stack = std::slice::from_ref(target_modifier);
    let mut material_slots = engine
        .doc
        .data_blocks
        .get(data_id)
        .map_or_else(Vec::new, |data| data.materials.clone());
    let object_material_count = engine
        .doc
        .nodes
        .get(target)
        .map_or(0, |node| node.materials.len());
    let material_slot_count = Some(crate::model::material_slot_count(
        object_material_count,
        material_slots.len(),
    ));
    let evaluated = {
        let data_block = engine.doc.data_blocks.get(data_id).ok_or_else(|| {
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
        let uses_scene_context = modifier_stack.iter().any(|modifier| {
            if !modifier.enabled {
                return false;
            }
            match modifier.modifier_type.as_str() {
                "boolean" => {
                    // Every EXACT Boolean takes the scene path, which owns its material slots.
                    let exact = crate::geom::modifiers::BooleanSettings::parse(modifier).is_ok_and(
                        |settings| settings.solver != crate::geom::modifiers::BooleanSolver::Float,
                    );
                    exact
                        || ["object", "collection", "target"]
                            .iter()
                            .any(|key| modifier.params.get(*key).is_some_and(Value::is_string))
                }
                "shrinkwrap" | "surface_deform" | "mesh_deform" => {
                    ["object", "collection", "target"]
                        .iter()
                        .any(|key| modifier.params.get(*key).is_some_and(Value::is_string))
                }
                "curve" | "hook" | "data_transfer" | "normal_edit" => ["object", "target"]
                    .iter()
                    .any(|key| modifier.params.get(*key).is_some_and(Value::is_string)),
                "vertex_weight_proximity" => ["object", "target", "mask_tex_map_object"]
                    .iter()
                    .any(|key| modifier.params.get(*key).is_some_and(Value::is_string)),
                "uv_project" => modifier
                    .params
                    .get("projectors")
                    .is_some_and(Value::is_array),
                "uv_warp" => ["object_from", "object_to"]
                    .iter()
                    .any(|key| modifier.params.get(*key).is_some_and(Value::is_string)),
                "armature" | "displace" | "wave" | "warp" => true,
                "cast" | "screw" => modifier.params.get("object").is_some_and(Value::is_string),
                "simple_deform" => modifier.params.get("origin").is_some_and(Value::is_string),
                "mirror" => modifier
                    .params
                    .get("mirror_object")
                    .is_some_and(Value::is_string),
                "vertex_weight_edit" | "vertex_weight_mix" => modifier
                    .params
                    .get("mask_tex_map_object")
                    .is_some_and(Value::is_string),
                "array" => {
                    (modifier.params.get("fit_type").and_then(Value::as_str) == Some("FIT_CURVE")
                        && modifier.params.get("curve").is_some_and(Value::is_string))
                        || ["start_cap", "end_cap"]
                            .iter()
                            .any(|key| modifier.params.get(*key).is_some_and(Value::is_string))
                        || (modifier
                            .params
                            .get("use_object_offset")
                            .and_then(Value::as_bool)
                            == Some(true)
                            && modifier
                                .params
                                .get("offset_object")
                                .is_some_and(Value::is_string))
                }
                _ => false,
            }
        });
        // Vertex-group readers consume the mesh attribute, so attach source weights whenever the
        // block has groups; the evaluated output then carries the remapped weights.
        let prepared_output = (!data_block.vertex_groups.is_empty()).then(|| {
            let mut mesh = output.clone();
            crate::eval::attach_vertex_group_weights(&mut mesh, data_block);
            mesh
        });
        if !allow_shape_keys && data_block.shape_keys.is_some() {
            return Err(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                "Modifier cannot be applied to a mesh with shape keys",
                json!({"feature_id":"modifier.apply.shape_key_guard","modifier_id":modifier_id}),
            ));
        }
        if !uses_armature && !uses_lattice && !uses_volume_node && !uses_scene_context {
            let source = prepared_output.as_ref().unwrap_or(output);
            let texture_seed = crate::geom::modifiers::TextureSeed::new(
                crate::eval::EvaluationPurpose::Preview,
                0,
                target.as_str(),
            )
            .with_stack_offset(target_modifier_index);
            let evaluation = crate::geom::modifiers::evaluate_modifiers_with_texture_seed_at_frame(
                source.clone(),
                modifier_stack,
                &engine.doc.node_groups,
                engine.evaluation_frame,
                subject_name,
                texture_seed,
                material_slot_count,
            )?;
            engine.warnings.extend(
                evaluation
                    .warnings
                    .into_iter()
                    .filter(is_apply_warning)
                    .map(|warning| crate::eval::attach_node_owner(warning, target)),
            );
            evaluation.mesh
        } else {
            let frame = engine.evaluation_frame;
            let driver_values = crate::eval::rig::evaluate_drivers(engine.doc, frame)?;
            let operand_snapshot = if uses_scene_context {
                Some(
                    crate::eval::Snapshot::evaluate_modifier_operands_with_cache(
                        engine.doc,
                        &crate::eval::EvaluationContext {
                            frame: Some(frame),
                            ..crate::eval::EvaluationContext::default()
                        },
                        engine.asset_root,
                        engine.cache_directory,
                        target,
                        modifier_id,
                        &driver_values,
                    )?,
                )
            } else {
                None
            };
            let empty_meshes = BTreeMap::new();
            let evaluated_operand_meshes = operand_snapshot
                .as_ref()
                .map_or(&empty_meshes, |snapshot| &snapshot.meshes);
            let empty_nodes = BTreeMap::new();
            let evaluated_operand_nodes = operand_snapshot
                .as_ref()
                .map_or(&empty_nodes, |snapshot| &snapshot.nodes);
            let world_matrices = if let Some(snapshot) = &operand_snapshot {
                let mut matrices = snapshot
                    .nodes
                    .iter()
                    .map(|(node_id, node)| {
                        (
                            node_id.clone(),
                            glam::DMat4::from_cols_array(&node.world_matrix),
                        )
                    })
                    .collect::<BTreeMap<_, _>>();
                matrices.extend(
                    snapshot
                        .transform_world_matrices
                        .iter()
                        .map(|(id, matrix)| (id.clone(), *matrix)),
                );
                matrices
            } else {
                let scene = engine.doc.scenes.get(&engine.doc.active_scene);
                crate::eval::rig::evaluate_world_matrices_with_resources(
                    engine.doc,
                    frame,
                    &driver_values,
                    engine.asset_root,
                    scene.map_or(24, |scene| scene.fps),
                    scene.map_or(1.0, |scene| scene.fps_base),
                )?
            };
            let curve_path_lengths = crate::eval::evaluated_array_curve_path_length(
                engine.doc,
                target_modifier,
                &world_matrices,
            )?
            .into_iter()
            .collect::<BTreeMap<_, _>>();
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
            let object_world = world_matrices.get(target).copied().ok_or_else(|| {
                PotError::new(
                    ErrorCode::EvaluationFailed,
                    "modifier subject transform is unavailable",
                )
            })?;
            let empty_bone_matrices = BTreeMap::new();
            let empty_bone_pose_matrices = BTreeMap::new();
            let references = crate::eval::SceneReferences {
                world_matrices: &world_matrices,
                bone_matrices: operand_snapshot
                    .as_ref()
                    .map_or(&empty_bone_matrices, |snapshot| &snapshot.bone_matrices),
                bone_pose_matrices: operand_snapshot
                    .as_ref()
                    .map_or(&empty_bone_pose_matrices, |snapshot| {
                        &snapshot.bone_pose_matrices
                    }),
                evaluated_meshes: evaluated_operand_meshes,
                project_root: engine.asset_root,
            };
            let texture_seed = crate::geom::modifiers::TextureSeed::new(
                crate::eval::EvaluationPurpose::Preview,
                0,
                target.as_str(),
            )
            .with_stack_offset(target_modifier_index);
            let evaluation = crate::geom::modifiers::evaluate_modifiers_with_scene_context(
                source.clone(),
                modifier_stack,
                &engine.doc.node_groups,
                frame,
                subject_name,
                object_world,
                &world_matrices,
                &base_object_meshes,
                evaluated_operand_meshes,
                &curve_path_lengths,
                crate::eval::ModifierEvaluationContext {
                    purpose: crate::eval::EvaluationPurpose::Preview,
                    project_root: engine.asset_root,
                    cache_directory: engine.cache_directory,
                    material_slot_count: Some(crate::model::material_slot_count(
                        object_material_count,
                        material_slots.len(),
                    )),
                },
                |_, modifier| modifier.enabled,
                texture_seed,
                |mesh, modifier, modifier_warnings, multi_input, texture_rng| {
                    let result = match modifier.modifier_type.as_str() {
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
                            multi_input,
                        ),
                        "lattice" => crate::eval::apply_lattice_modifier(
                            mesh,
                            modifier,
                            target,
                            data_block,
                            engine.doc,
                            &world_matrices,
                        ),
                        "volume_to_mesh" => crate::eval::apply_volume_to_mesh_modifier(
                            mesh,
                            modifier,
                            target,
                            engine.doc,
                            &world_matrices,
                            &volume_data,
                        ),
                        "surface_deform" | "mesh_deform" => {
                            crate::eval::apply_scene_deform_modifier(
                                mesh,
                                modifier,
                                target,
                                engine.doc,
                                &references,
                                modifier_warnings,
                            )
                        }
                        "boolean" | "shrinkwrap" | "curve" | "hook" | "cast" | "simple_deform" => {
                            crate::eval::apply_scene_operand_modifier(
                                mesh,
                                modifier,
                                target,
                                engine.doc,
                                &references,
                                evaluated_operand_nodes,
                                &mut material_slots,
                            )
                        }
                        "displace" | "wave" | "warp" => {
                            crate::eval::apply_scene_displacement_modifier(
                                mesh,
                                modifier,
                                target,
                                engine.doc,
                                &references,
                                frame,
                                texture_rng,
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
                            target,
                            engine.doc,
                            &world_matrices,
                            evaluated_operand_meshes,
                            engine.asset_root,
                        ),
                        _ => Err(PotError::new(
                            ErrorCode::InternalError,
                            "custom modifier callback received an unexpected type",
                        )),
                    };
                    result.map(|()| {
                        Some(crate::model::material_slot_count(
                            object_material_count,
                            material_slots.len(),
                        ))
                    })
                },
            )?;
            let (forwarded, blocking): (Vec<PotError>, Vec<PotError>) =
                evaluation.warnings.into_iter().partition(is_apply_warning);
            if let Some(warning) = blocking.into_iter().next() {
                return Err(warning);
            }
            engine.warnings.extend(
                forwarded
                    .into_iter()
                    .map(|warning| crate::eval::attach_node_owner(warning, target)),
            );
            evaluation.mesh
        }
    };
    Ok((evaluated, material_slots))
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
    let scope = super::parse_scope(
        engine,
        operation.get("scope"),
        "/scope",
        "scope must be a string",
        "scope must be shared or single_user",
        true,
    )?;
    let mut data_targets = Vec::new();
    for target in &selected {
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
        ensure_single_user_data(engine, &selected)?;
    }
    let mut evaluated_by_data = BTreeMap::<Id, crate::geom::Mesh>::new();
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
        let cache_shared = shared && scope == Some("shared");
        let evaluated =
            if cache_shared && let Some(evaluated) = evaluated_by_data.get(&original_data_id) {
                evaluated.clone()
            } else {
                let (target_modifier, target_modifier_index) =
                    modifier_to_apply(engine, &target, &modifier_id)?;
                let (evaluated, _) = evaluate_modifier_output(
                    engine,
                    &target,
                    &original_data_id,
                    &target_modifier,
                    target_modifier_index,
                    true,
                )?;
                if cache_shared {
                    evaluated_by_data.insert(original_data_id.clone(), evaluated.clone());
                }
                evaluated
            };
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
        if !same_topology(source_mesh, &evaluated) {
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
            && node
                .modifiers
                .iter()
                .any(|modifier| modifier.id == modifier_id)
        {
            node.modifiers.retain(|modifier| modifier.id != modifier_id);
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
    })?;
    let settings_parameter = match (family, kind) {
        (ParameterFamily::Modifier, "hook" | "warp") => Some("falloff_curve_settings"),
        _ => None,
    };
    if let Some(field) = settings_parameter
        && let Some(value) = params.get(field)
        && !valid_curve_mapping_settings(value)
    {
        let pointer = operation_pointer(
            engine.operation_index,
            &format!("{}/{field}", base_pointer.trim_start_matches('/')),
        );
        return Err(engine.error(
            ErrorCode::InvalidArgument,
            format!("{field} must contain valid use_clip, clip_bounds, and extend settings"),
            &pointer,
        ));
    }
    if matches!(family, ParameterFamily::Modifier)
        && matches!(kind, "hook" | "warp")
        && let Some(value) = params.get("falloff_curve")
        && !valid_curve_mapping_points(value)
    {
        let pointer = operation_pointer(
            engine.operation_index,
            &format!("{}/falloff_curve", base_pointer.trim_start_matches('/')),
        );
        return Err(engine.error(
            ErrorCode::InvalidArgument,
            "falloff_curve must contain at least two finite [x, y] points with an optional AUTO, AUTO_CLAMPED, or VECTOR handle",
            &pointer,
        ));
    }
    Ok(())
}

fn valid_curve_mapping_settings(value: &Value) -> bool {
    let Some(settings) = value.as_object() else {
        return false;
    };
    let clip_bounds_valid = settings.get("clip_bounds").is_none_or(|value| {
        let Some(bounds) = value.as_array() else {
            return false;
        };
        let [min_x, min_y, max_x, max_y] = bounds.as_slice() else {
            return false;
        };
        let Some(min_x) = min_x.as_f64() else {
            return false;
        };
        let Some(min_y) = min_y.as_f64() else {
            return false;
        };
        let Some(max_x) = max_x.as_f64() else {
            return false;
        };
        let Some(max_y) = max_y.as_f64() else {
            return false;
        };
        [min_x, min_y, max_x, max_y]
            .iter()
            .all(|bound| bound.is_finite() && (-100.0..=100.0).contains(bound))
            && min_x <= max_x
            && min_y <= max_y
    });
    settings.get("use_clip").is_none_or(Value::is_boolean)
        && clip_bounds_valid
        && settings
            .get("extend")
            .is_none_or(|value| matches!(value.as_str(), Some("HORIZONTAL" | "EXTRAPOLATED")))
}

fn valid_curve_mapping_points(value: &Value) -> bool {
    value.as_array().is_some_and(|points| {
        points.len() >= 2
            && points.iter().all(|point| {
                let Some(coordinates) = point.as_array() else {
                    return false;
                };
                (2..=3).contains(&coordinates.len())
                    && coordinates[..2].iter().all(|coordinate| {
                        coordinate.as_f64().is_some_and(|number| {
                            number.is_finite() && number.abs() <= f64::from(f32::MAX)
                        })
                    })
                    && coordinates.get(2).is_none_or(|handle| {
                        matches!(handle.as_str(), Some("AUTO" | "AUTO_CLAMPED" | "VECTOR"))
                    })
            })
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
    engine: &Engine<'_>,
    modifier_type: &str,
    params: &Map<String, Value>,
    base_pointer: &str,
) -> Result<()> {
    validate_attribute_modifier_params_inner(modifier_type, params).map_err(|error| {
        let Some(parameter) = error.details.get("parameter").and_then(Value::as_str) else {
            return error;
        };
        let Some(mut details) = error.details.as_object().cloned() else {
            return error;
        };
        details.insert("operation_index".to_owned(), json!(engine.operation_index));
        details.insert(
            "pointer".to_owned(),
            json!(operation_pointer(
                engine.operation_index,
                &format!(
                    "{}/{}",
                    base_pointer.trim_start_matches('/'),
                    super::pointer_escape(parameter)
                )
            )),
        );
        PotError::with_details(error.code, error.message, Value::Object(details))
    })
}

fn validate_attribute_modifier_params_inner(
    modifier_type: &str,
    params: &Map<String, Value>,
) -> Result<()> {
    let required: &[&str] = match modifier_type {
        "vertex_weight_edit" => &["vertex_group"],
        "vertex_weight_mix" => &["vertex_group_a", "vertex_group_b"],
        "vertex_weight_proximity" => &["target", "vertex_group"],
        "normal_edit" => &["target"],
        "uv_project" => &["projectors"],
        "data_transfer" => &["object"],
        _ => &[],
    };
    // An explicit null is no reference at all, so a required reference counts as missing.
    if let Some(field) = required
        .iter()
        .find(|field| params.get(**field).is_none_or(Value::is_null))
    {
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
            return Err(unsupported_attribute_modifier_value(modifier_type, field));
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
                return Err(unsupported_attribute_modifier_value(modifier_type, field));
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
        if let Some(value) = params.get(field).filter(|value| !value.is_null()) {
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

fn unsupported_attribute_modifier_value(modifier_type: &str, parameter: &str) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        format!(
            "modifier `{modifier_type}` parameter `{parameter}` uses a Blender value Potter does not evaluate yet"
        ),
        json!({"feature_id":format!("modifier.{modifier_type}.{parameter}"),"parameter":parameter}),
    )
}

fn is_physics_modifier(modifier_type: &str) -> bool {
    matches!(
        modifier_type,
        "cloth" | "soft_body" | "collision" | "dynamic_paint" | "fluid" | "particle_system"
    )
}

fn validate_volume_displace_params(engine: &Engine<'_>, params: &Map<String, Value>) -> Result<()> {
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
    if mapping_mode == "OBJECT" && params.get("texture_map_object").is_none_or(Value::is_null) {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "Volume Displace OBJECT mapping requires texture_map_object",
            "/params/texture_map_object",
        ));
    }
    if let Some(object) = params
        .get("texture_map_object")
        .filter(|value| !value.is_null())
    {
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
    if let Some(texture) = params.get("texture").filter(|value| !value.is_null()) {
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
    if modifier_type != "particle_instance" {
        return Ok(());
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
    if params
        .get("vertex_group")
        .is_some_and(|value| value.as_str().is_none_or(str::is_empty))
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("{modifier_type} `vertex_group` must be a non-empty group name"),
            "/params/vertex_group",
        ));
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
