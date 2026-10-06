use std::{fs, path::PathBuf};

use serde_json::{Map, Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    hash,
};

use super::{ChangeKind, Engine, check_fields, operation_pointer, read_id};

pub(super) fn apply(
    engine: &mut Engine<'_>,
    name: &str,
    operation: &Map<String, Value>,
) -> Result<bool> {
    match name {
        "resource.pack" => pack(engine, operation),
        "resource.unpack" => unpack(engine, operation),
        _ => Err(engine.error(
            ErrorCode::InternalError,
            "invalid resource operation dispatch",
            &operation_pointer(engine.operation_index, "op"),
        )),
    }
}

fn pack(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "id", "uri", "kind"],
        &["id", "uri"],
    )?;
    let id = read_id(engine, operation, "id")?;
    let kind = match operation.get("kind") {
        Some(value) => value.as_str().ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "resource kind must be a string",
                &operation_pointer(engine.operation_index, "kind"),
            )
        })?,
        None => "binary",
    };
    if kind.trim().is_empty() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "resource kind must not be empty",
            &operation_pointer(engine.operation_index, "kind"),
        ));
    }
    let uri = operation
        .get("uri")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "resource uri must be a string",
                &operation_pointer(engine.operation_index, "uri"),
            )
        })?;
    let path = resolve_path(uri);
    if !path.is_file() {
        return Err(engine.error(
            ErrorCode::FileNotFound,
            format!("resource file does not exist: {}", path.display()),
            &operation_pointer(engine.operation_index, "uri"),
        ));
    }
    let bytes = fs::read(&path).map_err(|error| PotError::io(&error))?;
    let digest = hash::sha256(&bytes);
    let byte_values = bytes
        .iter()
        .map(|byte| Value::from(*byte))
        .collect::<Vec<_>>();
    let change = if engine.doc.resources.contains_key(&id) {
        ChangeKind::Updated
    } else {
        ChangeKind::Created
    };
    engine.doc.resources.insert(
        id.clone(),
        json!({"uri":uri,"hash":digest,"kind":kind,"packed":true,"bytes":byte_values}),
    );
    engine.mark("resources", &id, change);
    Ok(true)
}

fn unpack(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(engine, operation, &["op", "id", "path"], &["id"])?;
    let id = read_id(engine, operation, "id")?;
    let record = engine.doc.resources.get(&id).cloned().ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            format!("resource `{id}` does not exist"),
            &operation_pointer(engine.operation_index, "id"),
        )
    })?;
    let byte_values = record
        .get("bytes")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                format!("resource `{id}` has no packed bytes"),
                &operation_pointer(engine.operation_index, "id"),
            )
        })?;
    let mut bytes = Vec::with_capacity(byte_values.len());
    for (index, value) in byte_values.iter().enumerate() {
        let byte = value
            .as_u64()
            .filter(|number| *number <= u8::MAX.into())
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::SceneInvalid,
                    "packed resource byte is outside 0..255",
                    &operation_pointer(engine.operation_index, &format!("id/bytes/{index}")),
                )
            })?;
        bytes.push(u8::try_from(byte).map_err(|_| {
            engine.error(
                ErrorCode::SceneInvalid,
                "packed resource byte is outside 0..255",
                &operation_pointer(engine.operation_index, &format!("id/bytes/{index}")),
            )
        })?);
    }
    let actual = hash::sha256(&bytes);
    let expected = record
        .get("hash")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if expected != actual {
        return Err(engine.error(
            ErrorCode::SceneInvalid,
            "packed resource content hash does not match",
            &operation_pointer(engine.operation_index, "id"),
        ));
    }
    let provided_path = operation
        .get("path")
        .map(|value| {
            value.as_str().map(str::to_owned).ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "resource path must be a string",
                    &operation_pointer(engine.operation_index, "path"),
                )
            })
        })
        .transpose()?;
    let raw_path = provided_path
        .or_else(|| record.get("uri").and_then(Value::as_str).map(str::to_owned))
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "resource.unpack requires path or original uri",
                &operation_pointer(engine.operation_index, "path"),
            )
        })?;
    let path = resolve_path(&raw_path);
    fs::write(&path, &bytes).map_err(|error| {
        PotError::with_details(ErrorCode::IoError, error.to_string(), json!({"path":path}))
    })?;
    let mut next = record;
    if let Some(object) = next.as_object_mut() {
        object.insert("uri".to_owned(), Value::String(raw_path));
        object.insert("packed".to_owned(), Value::Bool(false));
        object.remove("bytes");
    }
    engine.doc.resources.insert(id.clone(), next);
    engine.mark("resources", &id, ChangeKind::Updated);
    Ok(true)
}

fn resolve_path(uri: &str) -> PathBuf {
    PathBuf::from(uri.strip_prefix("file://").unwrap_or(uri))
}
