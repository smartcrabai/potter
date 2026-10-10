use serde_json::{Map, Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    model::{Id, is_valid_id},
};

use super::{ChangeKind, Engine, check_fields, operation_pointer};

const PERMISSIONS: &[&str] = &["read", "write", "evaluate", "render", "import", "export"];

pub(super) fn apply(
    engine: &mut Engine<'_>,
    name: &str,
    operation: &Map<String, Value>,
) -> Result<bool> {
    if name != "extension.register" {
        return Err(engine.error(
            ErrorCode::InternalError,
            "invalid extension operation dispatch",
            &operation_pointer(engine.operation_index, "op"),
        ));
    }
    check_fields(
        engine,
        operation,
        &["op", "id", "version", "permissions", "features"],
        &["id", "version", "permissions"],
    )?;
    let id = operation
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| is_valid_id(id))
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "extension id must be a valid persistent ID",
                &operation_pointer(engine.operation_index, "id"),
            )
        })?;
    let version = operation
        .get("version")
        .and_then(Value::as_str)
        .filter(|version| !version.trim().is_empty())
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "extension version must be a non-empty string",
                &operation_pointer(engine.operation_index, "version"),
            )
        })?;
    let permissions = operation
        .get("permissions")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "extension permissions must be an array",
                &operation_pointer(engine.operation_index, "permissions"),
            )
        })?;
    let mut granted = Vec::with_capacity(permissions.len());
    for (index, permission) in permissions.iter().enumerate() {
        let permission = permission.as_str().ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "extension permission must be a string",
                &operation_pointer(engine.operation_index, &format!("permissions/{index}")),
            )
        })?;
        if !PERMISSIONS.contains(&permission) {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("unsupported extension permission `{permission}`"),
                &operation_pointer(engine.operation_index, &format!("permissions/{index}")),
            ));
        }
        if granted.iter().any(|existing| existing == permission) {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("duplicate extension permission `{permission}`"),
                &operation_pointer(engine.operation_index, &format!("permissions/{index}")),
            ));
        }
        granted.push(permission.to_owned());
    }
    let features = operation
        .get("features")
        .cloned()
        .unwrap_or_else(|| json!([]));
    if !features
        .as_array()
        .is_some_and(|values| values.iter().all(Value::is_string))
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "extension features must be an array of strings",
            &operation_pointer(engine.operation_index, "features"),
        ));
    }
    let extensions = super::object_slot(
        engine,
        "extensions",
        super::ObjectSlotError::Static("compatibility extensions must be an object"),
    )?;
    let is_new = !extensions.contains_key(id);
    extensions.insert(id.to_owned(), json!({"version":version,"permissions":granted,"features":features,"scripts_executed":false}));
    let id =
        Id::new(id).map_err(|error| PotError::new(ErrorCode::InvalidOperation, error.message))?;
    engine.mark(
        "compatibility",
        &id,
        if is_new {
            ChangeKind::Created
        } else {
            ChangeKind::Updated
        },
    );
    Ok(true)
}
