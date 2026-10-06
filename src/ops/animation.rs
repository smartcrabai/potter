use serde_json::{Map, Value};

use crate::{
    error::{ErrorCode, Result},
    model::{Action, Extrapolation, FCurve, Id, Interpolation, Keyframe},
};

use super::{
    ChangeKind, Engine, check_fields, operation_pointer, read_id, read_string, resolve_node_targets,
};

const ACTION_TARGET_ID_POLICY: super::TargetIdPolicy = super::TargetIdPolicy::Strict {
    object_message: "target must be an object containing id",
    shape_message: "target must contain only id",
    id_message: "target id must be a string",
    require_id: true,
};

pub(super) fn apply(
    engine: &mut Engine<'_>,
    name: &str,
    operation: &Map<String, Value>,
) -> Result<bool> {
    match name {
        "action.create" => create_action(engine, operation),
        "action.update" => update_action(engine, operation),
        "action.delete" => delete_action(engine, operation),
        "keyframe.insert" => insert_keyframe(engine, operation),
        "keyframe.delete" => delete_keyframe(engine, operation),
        "fcurve.update" => update_fcurve(engine, operation),
        _ => Err(engine.error(
            ErrorCode::InternalError,
            "invalid animation operation dispatch",
            &operation_pointer(engine.operation_index, "op"),
        )),
    }
}

fn create_action(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(engine, operation, &["op", "id", "name"], &["id"])?;
    let id = read_id(engine, operation, "id")?;
    if engine.doc.actions.contains_key(&id) {
        return Err(engine.error(
            ErrorCode::IdExists,
            format!("action ID `{id}` already exists"),
            &operation_pointer(engine.operation_index, "id"),
        ));
    }
    let name = if operation.contains_key("name") {
        read_string(engine, operation, "name")?
    } else {
        id.to_string()
    };
    engine.doc.actions.insert(
        id.clone(),
        Action {
            name,
            fcurves: Vec::new(),
            ..Action::default()
        },
    );
    engine.mark("actions", &id, ChangeKind::Created);
    Ok(true)
}

fn update_action(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "set"],
        &["target", "set"],
    )?;
    let id = action_target_id(engine, operation)?;
    let set = operation
        .get("set")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "set must be an object",
                &operation_pointer(engine.operation_index, "set"),
            )
        })?;
    if set.is_empty() || set.keys().any(|key| key != "name" && key != "fcurves") {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "action set must contain only name and/or fcurves",
            &operation_pointer(engine.operation_index, "set"),
        ));
    }
    let name = set
        .get("name")
        .map(|value| {
            value.as_str().map(str::to_owned).ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "name must be a string",
                    &operation_pointer(engine.operation_index, "set/name"),
                )
            })
        })
        .transpose()?;
    let fcurves = set
        .get("fcurves")
        .map(|value| -> Result<Vec<FCurve>> {
            let curves = serde_json::from_value::<Vec<FCurve>>(value.clone()).map_err(|error| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    format!("invalid fcurves: {error}"),
                    &operation_pointer(engine.operation_index, "set/fcurves"),
                )
            })?;
            for curve in &curves {
                validate_path(engine, &curve.path, curve.index, "set/fcurves")?;
                validate_keyframes(engine, &curve.keyframes, "set/fcurves")?;
            }
            Ok(curves)
        })
        .transpose()?;
    let action = engine.doc.actions.get_mut(&id).ok_or_else(|| {
        crate::error::PotError::new(ErrorCode::InternalError, "action disappeared during update")
    })?;
    let mut changed = false;
    if let Some(name) = name
        && action.name != name
    {
        action.name = name;
        changed = true;
    }
    if let Some(fcurves) = fcurves
        && action.fcurves != fcurves
    {
        action.fcurves = fcurves;
        changed = true;
    }
    if changed {
        engine.mark("actions", &id, ChangeKind::Updated);
    }
    Ok(changed)
}

fn delete_action(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(engine, operation, &["op", "target"], &["target"])?;
    let id = action_target_id(engine, operation)?;
    if engine
        .doc
        .nodes
        .values()
        .any(|node| node.action.as_ref() == Some(&id))
        || engine.doc.data_blocks.values().any(|data| {
            data.shape_keys
                .as_ref()
                .is_some_and(|shape_keys| shape_keys.action.as_ref() == Some(&id))
        })
        || super::nla::action_is_referenced(engine, &id)
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("action `{id}` is assigned to a node"),
            &operation_pointer(engine.operation_index, "target/id"),
        ));
    }
    engine.doc.actions.remove(&id);
    engine.mark("actions", &id, ChangeKind::Deleted);
    Ok(true)
}

#[expect(
    clippy::float_cmp,
    reason = "exact keyframe equality detects a no-op update to Blender keyframes"
)]
fn insert_keyframe(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &[
            "op",
            "target",
            "path",
            "index",
            "frame",
            "value",
            "interpolation",
        ],
        &["target", "path", "index", "frame", "value"],
    )?;
    let action_id = target_node_action(engine, operation)?;
    let path = read_string(engine, operation, "path")?;
    let index = read_index(engine, operation)?;
    validate_path(engine, &path, index, "path")?;
    let frame = super::finite_number(
        engine,
        operation.get("frame"),
        "frame",
        super::FiniteNumberMessage::Field,
    )?;
    let value = super::finite_number(
        engine,
        operation.get("value"),
        "value",
        super::FiniteNumberMessage::Field,
    )?;
    let interpolation =
        parse_interpolation(engine, operation.get("interpolation"), "interpolation")?;
    if let Some(action) = engine.doc.actions.get(&action_id)
        && let Some(curve) = action
            .fcurves
            .iter()
            .find(|curve| curve.path == path && curve.index == index)
    {
        validate_keyframes(engine, &curve.keyframes, "path")?;
    }
    let action = engine.doc.actions.get_mut(&action_id).ok_or_else(|| {
        crate::error::PotError::new(ErrorCode::InternalError, "assigned action disappeared")
    })?;
    let curve_index = if let Some(curve_index) = action
        .fcurves
        .iter()
        .position(|curve| curve.path == path && curve.index == index)
    {
        curve_index
    } else {
        action.fcurves.push(FCurve {
            path: path.clone(),
            index,
            keyframes: Vec::new(),
            extrapolation: Extrapolation::Constant,
        });
        action.fcurves.len() - 1
    };
    let curve = action.fcurves.get_mut(curve_index).ok_or_else(|| {
        crate::error::PotError::new(ErrorCode::InternalError, "inserted FCurve is missing")
    })?;
    let keyframe = Keyframe {
        frame,
        value,
        interpolation,
        ..Keyframe::default()
    };
    match curve.keyframes.binary_search_by(|key| {
        key.frame
            .partial_cmp(&frame)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        Ok(position) => {
            let existing = &curve.keyframes[position];
            if existing.value == value && existing.interpolation == interpolation {
                return Ok(false);
            }
            curve.keyframes[position] = keyframe;
        }
        Err(position) => curve.keyframes.insert(position, keyframe),
    }
    if curve.keyframes.windows(2).any(|pair| {
        !(pair[1].frame - pair[0].frame).is_finite() || !(pair[1].value - pair[0].value).is_finite()
    }) {
        return Err(crate::error::PotError::new(
            ErrorCode::InvalidOperation,
            "keyframe interval must be numerically bounded",
        ));
    }
    engine.mark("actions", &action_id, ChangeKind::Updated);
    Ok(true)
}

fn delete_keyframe(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "path", "index", "frame"],
        &["target", "path", "index", "frame"],
    )?;
    let action_id = target_node_action(engine, operation)?;
    let path = read_string(engine, operation, "path")?;
    let index = read_index(engine, operation)?;
    validate_path(engine, &path, index, "path")?;
    let frame = super::finite_number(
        engine,
        operation.get("frame"),
        "frame",
        super::FiniteNumberMessage::Field,
    )?;
    let action = engine.doc.actions.get(&action_id).ok_or_else(|| {
        crate::error::PotError::new(ErrorCode::InternalError, "assigned action disappeared")
    })?;
    let curve_position = action
        .fcurves
        .iter()
        .position(|curve| curve.path == path && curve.index == index)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::TargetNotFound,
                "FCurve was not found",
                &operation_pointer(engine.operation_index, "path"),
            )
        })?;
    let key_position = action.fcurves[curve_position]
        .keyframes
        .iter()
        .position(|key| {
            key.frame
                .partial_cmp(&frame)
                .is_some_and(std::cmp::Ordering::is_eq)
        })
        .ok_or_else(|| {
            engine.error(
                ErrorCode::TargetNotFound,
                "keyframe was not found",
                &operation_pointer(engine.operation_index, "frame"),
            )
        })?;
    let action = engine.doc.actions.get_mut(&action_id).ok_or_else(|| {
        crate::error::PotError::new(ErrorCode::InternalError, "assigned action disappeared")
    })?;
    action.fcurves[curve_position]
        .keyframes
        .remove(key_position);
    engine.mark("actions", &action_id, ChangeKind::Updated);
    Ok(true)
}

fn update_fcurve(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "path", "index", "set"],
        &["target", "path", "index", "set"],
    )?;
    let action_id = target_node_action(engine, operation)?;
    let path = read_string(engine, operation, "path")?;
    let index = read_index(engine, operation)?;
    validate_path(engine, &path, index, "path")?;
    let set = operation
        .get("set")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "set must be an object",
                &operation_pointer(engine.operation_index, "set"),
            )
        })?;
    if set.len() != 1 || !set.contains_key("extrapolation") {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "fcurve.update set requires only extrapolation",
            &operation_pointer(engine.operation_index, "set"),
        ));
    }
    let extrapolation = match set.get("extrapolation").and_then(Value::as_str) {
        Some("constant") => Extrapolation::Constant,
        Some("linear") => Extrapolation::Linear,
        _ => {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "extrapolation must be constant or linear",
                &operation_pointer(engine.operation_index, "set/extrapolation"),
            ));
        }
    };
    let action = engine.doc.actions.get(&action_id).ok_or_else(|| {
        crate::error::PotError::new(ErrorCode::InternalError, "assigned action disappeared")
    })?;
    let curve_position = action
        .fcurves
        .iter()
        .position(|curve| curve.path == path && curve.index == index)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::TargetNotFound,
                "FCurve was not found",
                &operation_pointer(engine.operation_index, "path"),
            )
        })?;
    if action.fcurves[curve_position].extrapolation == extrapolation {
        return Ok(false);
    }
    let action = engine.doc.actions.get_mut(&action_id).ok_or_else(|| {
        crate::error::PotError::new(ErrorCode::InternalError, "assigned action disappeared")
    })?;
    action.fcurves[curve_position].extrapolation = extrapolation;
    engine.mark("actions", &action_id, ChangeKind::Updated);
    Ok(true)
}

fn action_target_id(engine: &Engine<'_>, operation: &Map<String, Value>) -> Result<Id> {
    let id = super::target_id_value(
        engine,
        operation.get("target"),
        &operation_pointer(engine.operation_index, "target"),
        ACTION_TARGET_ID_POLICY,
    )?;
    if !engine.doc.actions.contains_key(&id) {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            format!("action `{id}` was not found"),
            &operation_pointer(engine.operation_index, "target/id"),
        ));
    }
    Ok(id)
}

fn target_node_action(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<Id> {
    let target = operation.get("target").ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "missing target",
            &operation_pointer(engine.operation_index, "target"),
        )
    })?;
    let mut targets = resolve_node_targets(engine, target, false)?;
    let node_id = targets.pop().ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            "node target was not found",
            &operation_pointer(engine.operation_index, "target"),
        )
    })?;
    let node = engine.doc.nodes.get(&node_id).ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            format!("node `{node_id}` was not found"),
            &operation_pointer(engine.operation_index, "target"),
        )
    })?;
    let action_id = node.action.clone().ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "node must have an assigned action",
            &operation_pointer(engine.operation_index, "target"),
        )
    })?;
    if !engine.doc.actions.contains_key(&action_id) {
        return Err(engine.error(
            ErrorCode::SceneInvalid,
            "node action reference does not exist",
            &operation_pointer(engine.operation_index, "target"),
        ));
    }
    Ok(action_id)
}

fn read_index(engine: &Engine<'_>, operation: &Map<String, Value>) -> Result<u32> {
    operation
        .get("index")
        .and_then(Value::as_u64)
        .and_then(|index| u32::try_from(index).ok())
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "index must be a non-negative 32-bit integer",
                &operation_pointer(engine.operation_index, "index"),
            )
        })
}

fn parse_interpolation(
    engine: &Engine<'_>,
    value: Option<&Value>,
    field: &str,
) -> Result<Interpolation> {
    let Some(value) = value else {
        return Ok(Interpolation::Linear);
    };
    let interpolation = value.as_str().ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "interpolation must be a string",
            &operation_pointer(engine.operation_index, field),
        )
    })?;
    match interpolation {
        "constant" => Ok(Interpolation::Constant),
        "linear" => Ok(Interpolation::Linear),
        "bezier" => Ok(Interpolation::Bezier),
        _ => Err(engine.error(
            ErrorCode::InvalidOperation,
            "interpolation must be constant, linear, or bezier",
            &operation_pointer(engine.operation_index, field),
        )),
    }
}

fn validate_path(engine: &Engine<'_>, path: &str, index: u32, field: &str) -> Result<()> {
    let Some(component_count) = animation_path_component_count(path) else {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("unsupported animation path `{path}`"),
            &operation_pointer(engine.operation_index, field),
        ));
    };
    if usize::try_from(index).is_ok_and(|value| value < component_count) {
        Ok(())
    } else {
        Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("index is outside the component range for `{path}`"),
            &operation_pointer(engine.operation_index, "index"),
        ))
    }
}

fn animation_path_component_count(path: &str) -> Option<usize> {
    match path {
        "transform.translation" | "transform.rotation_euler" | "transform.scale" => Some(3),
        "transform.rotation_quaternion" | "transform.rotation" => Some(4),
        "visible"
        | "camera.lens_mm"
        | "light.energy"
        | "shape_keys.evaluation_time"
        | "shape_key.evaluation_time"
        | "key_blocks.eval_time" => Some(1),
        _ if path
            .strip_prefix("material.")
            .and_then(|rest| rest.strip_suffix(".base_color"))
            .is_some_and(crate::model::is_valid_id) =>
        {
            Some(4)
        }
        _ => {
            if let Some(rest) = path.strip_prefix("pose.")
                && let Some((bone_id, channel)) = rest.split_once('.')
                && crate::model::is_valid_id(bone_id)
                && !channel.contains('.')
            {
                return match channel {
                    "translation" | "scale" => Some(3),
                    "rotation" => Some(4),
                    _ => None,
                };
            }
            if let Some(rest) = path.strip_prefix("shape_key.")
                && let Some((key_id, channel)) = rest.split_once('.')
                && crate::model::is_valid_id(key_id)
                && channel == "value"
            {
                return Some(1);
            }
            None
        }
    }
}

fn validate_keyframes(engine: &Engine<'_>, keyframes: &[Keyframe], field: &str) -> Result<()> {
    if keyframes.iter().any(|key| {
        !key.frame.is_finite()
            || !key.value.is_finite()
            || key
                .handle_left
                .is_some_and(|handle| !handle[0].is_finite() || !handle[1].is_finite())
            || key
                .handle_right
                .is_some_and(|handle| !handle[0].is_finite() || !handle[1].is_finite())
    }) || keyframes.windows(2).any(|pair| {
        pair[0].frame >= pair[1].frame
            || !(pair[1].frame - pair[0].frame).is_finite()
            || !(pair[1].value - pair[0].value).is_finite()
    }) {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "keyframes and Bezier handles must be finite, strictly frame-ordered, and numerically bounded",
            &operation_pointer(engine.operation_index, field),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests")]

    use serde_json::json;

    use crate::{
        model::{Action, Extrapolation, FCurve, Id, Interpolation, Keyframe},
        ops::apply_batch,
    };

    #[test]
    fn action_assignment_and_keyframe_insert_replace_existing_frame() {
        let mut doc =
            crate::model::SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        let node_id = Id::new("animated").unwrap();
        let action_id = Id::new("move_action").unwrap();
        doc.nodes.insert(
            node_id.clone(),
            crate::model::Node {
                name: "Animated".to_owned(),
                kind: "empty".to_owned(),
                primitive: None,
                tags: Vec::new(),
                parent: None,
                parent_inverse: None,
                transform: crate::model::Transform::default(),
                data: None,
                materials: Vec::new(),
                modifiers: Vec::new(),
                visible: true,
                render_visible: true,
                selectable: true,
                action: None,
                properties: serde_json::Map::new(),
                rigid_body: None,
                force_field: None,
                ..crate::model::Node::default()
            },
        );
        doc.actions.insert(
            action_id.clone(),
            Action {
                name: "Move".to_owned(),
                fcurves: vec![FCurve {
                    path: "transform.translation".to_owned(),
                    index: 0,
                    keyframes: vec![Keyframe {
                        frame: 1.0,
                        value: 1.0,
                        interpolation: Interpolation::Linear,
                        ..Keyframe::default()
                    }],
                    extrapolation: Extrapolation::Constant,
                }],
                ..Action::default()
            },
        );
        let outcome = apply_batch(&doc, &json!({"schema_version":1,"base_revision":0,"operations":[
            {"op":"node.update","target":{"id":"animated"},"set":{"action":"move_action"}},
            {"op":"keyframe.insert","target":{"id":"animated"},"path":"transform.translation","index":0,"frame":1.0,"value":3.0}
        ]})).unwrap();
        assert_eq!(
            outcome.doc.actions[&action_id].fcurves[0].keyframes[0].value,
            3.0
        );
        assert_eq!(outcome.doc.nodes[&node_id].action, Some(action_id));
    }
    #[test]
    fn action_fcurve_and_keyframe_operations_complete_the_lifecycle() {
        let mut doc =
            crate::model::SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        let node_id = Id::new("animated").unwrap();
        let action_id = Id::new("move_action").unwrap();
        doc.nodes.insert(
            node_id.clone(),
            crate::model::Node {
                name: "Animated".to_owned(),
                kind: "empty".to_owned(),
                primitive: None,
                tags: Vec::new(),
                parent: None,
                parent_inverse: None,
                transform: crate::model::Transform::default(),
                data: None,
                materials: Vec::new(),
                modifiers: Vec::new(),
                visible: true,
                render_visible: true,
                selectable: true,
                action: None,
                properties: serde_json::Map::new(),
                rigid_body: None,
                force_field: None,
                ..crate::model::Node::default()
            },
        );
        doc.actions.insert(
            action_id.clone(),
            Action {
                name: "Move".to_owned(),
                fcurves: vec![FCurve {
                    path: "transform.translation".to_owned(),
                    index: 0,
                    keyframes: vec![Keyframe {
                        frame: 1.0,
                        value: 1.0,
                        interpolation: Interpolation::Linear,
                        ..Keyframe::default()
                    }],
                    extrapolation: Extrapolation::Constant,
                }],
                ..Action::default()
            },
        );
        let outcome = apply_batch(&doc, &json!({"schema_version":1,"base_revision":0,"operations":[
            {"op":"node.update","target":{"id":"animated"},"set":{"action":"move_action"}},
            {"op":"action.update","target":{"id":"move_action"},"set":{"name":"Renamed"}},
            {"op":"fcurve.update","target":{"id":"animated"},"path":"transform.translation","index":0,"set":{"extrapolation":"linear"}},
            {"op":"keyframe.delete","target":{"id":"animated"},"path":"transform.translation","index":0,"frame":1.0},
            {"op":"node.update","target":{"id":"animated"},"set":{"action":null}},
            {"op":"action.delete","target":{"id":"move_action"}}
        ]})).unwrap();
        assert_eq!(outcome.operations[1]["changed"], true);
        assert_eq!(outcome.operations[2]["changed"], true);
        assert_eq!(outcome.operations[3]["changed"], true);
        assert!(outcome.doc.actions.is_empty());
        assert_eq!(outcome.doc.nodes[&node_id].action, None);
    }
}
