use serde_json::{Map, Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    model::{AvatarSettings, SceneDoc, validate_avatar_settings},
};

pub(super) fn apply(
    doc: &mut SceneDoc,
    name: &str,
    operation: &Map<String, Value>,
    operation_index: usize,
) -> Result<bool> {
    match name {
        "avatar.clear" => clear(doc, operation, operation_index),
        "avatar.update" => update(doc, operation, operation_index),
        _ => Err(operation_error(
            operation_index,
            "op",
            "unsupported avatar operation",
        )),
    }
}

fn clear(doc: &mut SceneDoc, operation: &Map<String, Value>, index: usize) -> Result<bool> {
    if operation.keys().any(|key| key != "op") {
        return Err(operation_error(
            index,
            "",
            "avatar.clear accepts only the `op` field",
        ));
    }
    let changed = doc.avatar.take().is_some();
    Ok(changed)
}

fn update(doc: &mut SceneDoc, operation: &Map<String, Value>, index: usize) -> Result<bool> {
    if operation.keys().any(|key| key != "op" && key != "set") {
        return Err(operation_error(
            index,
            "",
            "avatar.update accepts only `op` and `set` fields",
        ));
    }
    let set = operation
        .get("set")
        .and_then(Value::as_object)
        .ok_or_else(|| operation_error(index, "set", "avatar.update set must be an object"))?;
    if set.is_empty() {
        return Err(operation_error(
            index,
            "set",
            "avatar.update set must supply at least one component",
        ));
    }
    for field in set.keys() {
        if !matches!(
            field.as_str(),
            "armature"
                | "humanoid"
                | "meta"
                | "expressions"
                | "first_person"
                | "look_at"
                | "spring_bone"
                | "mtoon"
        ) {
            return Err(operation_error(
                index,
                &format!("set/{field}"),
                &format!("unknown avatar setting `{field}`"),
            ));
        }
    }
    if doc.avatar.is_none() {
        for required in ["armature", "humanoid", "meta"] {
            if !set.contains_key(required) || set.get(required).is_some_and(Value::is_null) {
                return Err(operation_error(
                    index,
                    &format!("set/{required}"),
                    "initial avatar.update must supply armature, humanoid, and meta",
                ));
            }
        }
    }

    let original = doc.avatar.as_ref();
    let mut merged = match original {
        Some(settings) => serde_json::to_value(settings).map_err(|error| {
            operation_error(
                index,
                "set",
                &format!("could not serialize current avatar settings: {error}"),
            )
        })?,
        None => json!({}),
    };
    let merged_object = merged.as_object_mut().ok_or_else(|| {
        operation_error(index, "set", "current avatar settings are not an object")
    })?;
    for (field, value) in set {
        let value = if value.is_null() && matches!(field.as_str(), "expressions" | "mtoon") {
            json!({})
        } else {
            value.clone()
        };
        merged_object.insert(field.clone(), value);
    }
    let settings: AvatarSettings = serde_json::from_value(merged).map_err(|error| {
        operation_error(index, "set", &format!("invalid avatar settings: {error}"))
    })?;
    validate_avatar_settings(doc, &settings).map_err(|error| {
        let mut details = error.details;
        if let Some(details) = details.as_object_mut() {
            details.insert("operation_index".to_owned(), json!(index));
            let suffix = details
                .get("pointer")
                .and_then(Value::as_str)
                .unwrap_or("/avatar");
            details.insert(
                "pointer".to_owned(),
                json!(format!(
                    "/operations/{index}/set{}",
                    suffix.strip_prefix("/avatar").unwrap_or("")
                )),
            );
        }
        PotError::with_details(ErrorCode::InvalidOperation, error.message, details)
    })?;
    let changed = original != Some(&settings);
    if changed {
        doc.avatar = Some(settings);
    }
    Ok(changed)
}

fn operation_error(index: usize, suffix: &str, message: &str) -> PotError {
    let pointer = if suffix.is_empty() {
        format!("/operations/{index}")
    } else {
        format!("/operations/{index}/{suffix}")
    };
    PotError::with_details(
        ErrorCode::InvalidOperation,
        message,
        json!({"operation_index":index,"pointer":pointer}),
    )
}
