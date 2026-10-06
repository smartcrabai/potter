use serde_json::{Map, Value};

use crate::{
    error::{ErrorCode, PotError, Result},
    mask::Mask,
};

use super::{ChangeKind, Engine, check_fields, operation_pointer, read_id};

pub(super) fn apply(
    engine: &mut Engine<'_>,
    name: &str,
    operation: &Map<String, Value>,
) -> Result<bool> {
    match name {
        "mask.create" => create(engine, operation),
        "mask.update" => update(engine, operation),
        "mask.delete" => delete(engine, operation),
        _ => Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("unsupported mask operation `{name}`"),
            &operation_pointer(engine.operation_index, "op"),
        )),
    }
}

fn create(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "id", "splines"],
        &["op", "id", "splines"],
    )?;
    let id = read_id(engine, operation, "id")?;
    if engine.doc.masks.contains_key(&id) {
        return Err(engine.error(
            ErrorCode::IdExists,
            "mask ID already exists",
            &operation_pointer(engine.operation_index, "id"),
        ));
    }
    let mut fields = operation.clone();
    fields.remove("op");
    fields.remove("id");
    let mask: Mask = serde_json::from_value(Value::Object(fields)).map_err(|error| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("invalid mask: {error}"),
            &operation_pointer(engine.operation_index, "splines"),
        )
    })?;
    let mask = Mask::new(mask.splines).map_err(|error| {
        engine.error(
            ErrorCode::InvalidOperation,
            error.to_string(),
            &operation_pointer(engine.operation_index, "splines"),
        )
    })?;
    engine.doc.masks.insert(id.clone(), mask);
    engine.mark("masks", &id, ChangeKind::Created);
    Ok(true)
}

fn update(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "id", "set"],
        &["op", "id", "set"],
    )?;
    let id = read_id(engine, operation, "id")?;
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
    if set.is_empty() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "mask.update set must not be empty",
            &operation_pointer(engine.operation_index, "set"),
        ));
    }
    for key in set.keys() {
        if key != "splines" {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("unknown mask update field `{key}`"),
                &operation_pointer(engine.operation_index, &format!("set/{key}")),
            ));
        }
    }
    let current = engine.doc.masks.get(&id).ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            "mask does not exist",
            &operation_pointer(engine.operation_index, "id"),
        )
    })?;
    let mut value = serde_json::to_value(current)
        .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))?;
    let current_fields = value.as_object_mut().ok_or_else(|| {
        PotError::new(ErrorCode::InternalError, "serialized mask is not an object")
    })?;
    for (key, field) in set {
        current_fields.insert(key.clone(), field.clone());
    }
    let mask: Mask = serde_json::from_value(value).map_err(|error| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("invalid mask update: {error}"),
            &operation_pointer(engine.operation_index, "set"),
        )
    })?;
    let mask = Mask::new(mask.splines).map_err(|error| {
        engine.error(
            ErrorCode::InvalidOperation,
            error.to_string(),
            &operation_pointer(engine.operation_index, "set/splines"),
        )
    })?;
    let changed = engine.doc.masks.get(&id) != Some(&mask);
    if changed {
        engine.doc.masks.insert(id.clone(), mask);
        engine.mark("masks", &id, ChangeKind::Updated);
    }
    Ok(changed)
}

fn delete(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(engine, operation, &["op", "id"], &["op", "id"])?;
    let id = read_id(engine, operation, "id")?;
    if engine.doc.masks.remove(&id).is_none() {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            "mask does not exist",
            &operation_pointer(engine.operation_index, "id"),
        ));
    }
    engine.mark("masks", &id, ChangeKind::Deleted);
    Ok(true)
}
