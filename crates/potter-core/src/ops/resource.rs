use std::{
    collections::BTreeSet,
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
};

use serde_json::{Map, Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    hash,
    model::migrate::inline_resource_kind,
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
    if let Some(record) = engine.doc.resources.get(&id)
        && let Some(resource_kind) = inline_resource_kind(record)?
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("inline resource `{id}` of kind `{resource_kind}` cannot be packed"),
            &operation_pointer(engine.operation_index, "id"),
        ));
    }
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
    let record = engine.doc.resources.get(&id).ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            format!("resource `{id}` does not exist"),
            &operation_pointer(engine.operation_index, "id"),
        )
    })?;
    if let Some(resource_kind) = inline_resource_kind(record)? {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("inline resource `{id}` of kind `{resource_kind}` cannot be unpacked"),
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
    if record.get("sequence_frames").is_some() {
        let raw_path = provided_path
            .or_else(|| record.get("uri").and_then(Value::as_str).map(str::to_owned))
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "resource.unpack requires path or original sequence uri",
                    &operation_pointer(engine.operation_index, "path"),
                )
            })?;
        let next = unpack_sequence(engine, &id, record, raw_path)?;
        engine.doc.resources.insert(id.clone(), next);
        engine.mark("resources", &id, ChangeKind::Updated);
        return Ok(true);
    }
    let record = record.clone();
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

fn unpack_sequence(
    engine: &Engine<'_>,
    id: &crate::model::Id,
    record: &Value,
    raw_path: String,
) -> Result<Value> {
    if record.get("packed").and_then(Value::as_bool) != Some(true) {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "resource.unpack sequence requires a packed sequence",
            &operation_pointer(engine.operation_index, "id"),
        ));
    }
    if record
        .get("uri")
        .and_then(Value::as_str)
        .is_none_or(|uri| uri.trim().is_empty())
    {
        return Err(engine.error(
            ErrorCode::SceneInvalid,
            "packed Alembic sequence has no container URI",
            &operation_pointer(engine.operation_index, "id/uri"),
        ));
    }
    let kind = record
        .get("kind")
        .and_then(Value::as_str)
        .filter(|kind| !kind.trim().is_empty())
        .ok_or_else(|| {
            engine.error(
                ErrorCode::SceneInvalid,
                "packed Alembic sequence has no kind",
                &operation_pointer(engine.operation_index, "id/kind"),
            )
        })?
        .to_owned();
    if kind != "alembic" || record.get("hash").is_some() || record.get("bytes").is_some() {
        return Err(engine.error(
            ErrorCode::SceneInvalid,
            "packed Alembic sequence must use the standard container record",
            &operation_pointer(engine.operation_index, "id"),
        ));
    }
    let frame_range = |name: &str| -> Result<i32> {
        record
            .get(name)
            .and_then(Value::as_i64)
            .and_then(|value| i32::try_from(value).ok())
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::SceneInvalid,
                    "packed Alembic sequence has no inclusive int32 frame range",
                    &operation_pointer(engine.operation_index, &format!("id/{name}")),
                )
            })
    };
    let start = frame_range("sequence_start")?;
    let end = frame_range("sequence_end")?;
    if end < start {
        return Err(engine.error(
            ErrorCode::SceneInvalid,
            "packed Alembic sequence frame range is not ascending",
            &operation_pointer(engine.operation_index, "id/sequence_end"),
        ));
    }
    let frame_count = i64::from(end) - i64::from(start) + 1;
    if frame_count > crate::cache_sequence::MAX_CACHE_SEQUENCE_FRAMES {
        return Err(engine.error(
            ErrorCode::SceneInvalid,
            "packed Alembic sequence frame range exceeds the supported limit",
            &operation_pointer(engine.operation_index, "id/sequence_end"),
        ));
    }
    let frames = record
        .get("sequence_frames")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::SceneInvalid,
                "packed Alembic sequence has no frame records",
                &operation_pointer(engine.operation_index, "id/sequence_frames"),
            )
        })?;
    let expected_frames = usize::try_from(frame_count).map_err(|_| {
        engine.error(
            ErrorCode::SceneInvalid,
            "packed Alembic sequence frame range cannot be represented",
            &operation_pointer(engine.operation_index, "id/sequence_frames"),
        )
    })?;
    if frames.len() != expected_frames {
        return Err(engine.error(
            ErrorCode::SceneInvalid,
            "packed Alembic sequence does not contain every frame in its inclusive range",
            &operation_pointer(engine.operation_index, "id/sequence_frames"),
        ));
    }
    for key in frames.keys() {
        let frame = key.parse::<i32>().ok();
        if frame
            .is_none_or(|frame| frame.to_string() != key.as_str() || frame < start || frame > end)
        {
            return Err(engine.error(
                ErrorCode::SceneInvalid,
                "Alembic sequence frame key is not a canonical in-range int32",
                &operation_pointer(engine.operation_index, "id/sequence_frames"),
            ));
        }
    }

    let pattern = resolve_path(&raw_path);
    let mut output_paths = BTreeSet::new();
    let mut sequence_outputs = Vec::with_capacity(frames.len());
    for frame in start..=end {
        let frame_key = frame.to_string();
        let frame_record = frames.get(&frame_key).ok_or_else(|| {
            engine.error(
                ErrorCode::SceneInvalid,
                "packed Alembic sequence is missing an in-range frame record",
                &operation_pointer(engine.operation_index, "id/sequence_frames"),
            )
        })?;
        let frame_kind = frame_record.get("kind").and_then(Value::as_str);
        if frame_kind != Some(kind.as_str())
            || frame_record.get("packed").and_then(Value::as_bool) != Some(true)
        {
            return Err(engine.error(
                ErrorCode::SceneInvalid,
                "packed Alembic sequence frame is not a matching standard resource record",
                &operation_pointer(
                    engine.operation_index,
                    &format!("id/sequence_frames/{frame}"),
                ),
            ));
        }
        let frame_path = crate::cache_sequence::cache_sequence_output_path(&pattern, frame)?
            .ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::InvalidArgument,
                    "sequence unpack path must include a frame-number pattern",
                    json!({"path":pattern,"frame":frame}),
                )
            })?;
        if !output_paths.insert(frame_path.clone()) {
            return Err(PotError::with_details(
                ErrorCode::InvalidArgument,
                "sequence unpack frame paths are not unique",
                json!({"path":frame_path,"frame":frame}),
            ));
        }
        match fs::symlink_metadata(&frame_path) {
            Ok(metadata) if metadata.is_file() => {}
            Ok(_) => {
                return Err(PotError::with_details(
                    ErrorCode::InvalidArgument,
                    "sequence unpack destination is not a regular file",
                    json!({"path":frame_path,"frame":frame}),
                ));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(PotError::io(&error)),
        }
        sequence_outputs.push((frame, frame_key, frame_record, frame_path));
    }

    let mut output_files = Vec::with_capacity(sequence_outputs.len());
    let mut output_frames = Map::new();
    for (frame, frame_key, frame_record, frame_path) in sequence_outputs {
        let resource_label = format!("{id}:{frame}");
        let (bytes, digest) = crate::cache_sequence::read_cache_resource_record(
            frame_record,
            &resource_label,
            engine.asset_root,
        )?;
        let uri = frame_path.to_string_lossy();
        let mut output_frame =
            crate::cache_sequence::cache_sequence_resource_record(&uri, &kind, &digest, None);
        let output_fields = output_frame.as_object_mut().ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "cache frame record is not an object",
            )
        })?;
        if let Some(source_fields) = frame_record.as_object() {
            for (key, value) in source_fields {
                if !matches!(
                    key.as_str(),
                    "uri" | "kind" | "hash" | "packed" | "bytes" | "export_path"
                ) {
                    output_fields.insert(key.clone(), value.clone());
                }
            }
        }
        output_fields.insert("status".to_owned(), json!("available"));
        output_frames.insert(frame_key, output_frame);
        output_files.push((frame_path, bytes));
    }

    publish_sequence_outputs(&output_files)?;
    let mut next = record
        .as_object()
        .into_iter()
        .flat_map(|fields| fields.iter())
        .filter(|(key, _)| !matches!(key.as_str(), "bytes" | "sequence_frames" | "export_path"))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect::<Map<_, _>>();
    next.insert("uri".to_owned(), Value::String(raw_path));
    next.insert("kind".to_owned(), Value::String(kind));
    next.insert("packed".to_owned(), Value::Bool(false));
    next.insert("sequence_start".to_owned(), json!(start));
    next.insert("sequence_end".to_owned(), json!(end));
    next.insert("sequence_frames".to_owned(), Value::Object(output_frames));
    Ok(Value::Object(next))
}

fn publish_sequence_outputs(outputs: &[(PathBuf, Vec<u8>)]) -> Result<()> {
    let parent = outputs
        .first()
        .and_then(|(path, _)| path.parent())
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    fs::create_dir_all(parent).map_err(|error| PotError::io(&error))?;
    let staging = parent.join(format!(".potter-sequence-unpack-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&staging).map_err(|error| PotError::io(&error))?;
    let mut preserve_staging = false;
    let result = (|| -> Result<()> {
        let mut staged = Vec::with_capacity(outputs.len());
        for (index, (_, bytes)) in outputs.iter().enumerate() {
            let path = staging.join(format!("frame-{index}.tmp"));
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .map_err(|error| PotError::io(&error))?;
            file.write_all(bytes)
                .and_then(|()| file.sync_all())
                .map_err(|error| PotError::io(&error))?;
            staged.push(path);
        }

        let mut published = Vec::<(PathBuf, Option<PathBuf>, bool)>::with_capacity(outputs.len());
        for (index, (destination, _)) in outputs.iter().enumerate() {
            let backup = match fs::symlink_metadata(destination) {
                Ok(metadata) if metadata.is_file() => {
                    let backup = staging.join(format!("backup-{index}"));
                    if let Err(error) = fs::rename(destination, &backup) {
                        let rollback_errors = rollback_sequence_outputs(&published);
                        preserve_staging = !rollback_errors.is_empty();
                        return Err(PotError::with_details(
                            ErrorCode::IoError,
                            error.to_string(),
                            json!({
                                "path":destination,
                                "rollback_errors":rollback_errors,
                                "staging_path":staging,
                            }),
                        ));
                    }
                    Some(backup)
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                Ok(_) => {
                    let rollback_errors = rollback_sequence_outputs(&published);
                    preserve_staging = !rollback_errors.is_empty();
                    return Err(PotError::with_details(
                        ErrorCode::InvalidArgument,
                        "sequence unpack destination changed to a non-file during publication",
                        json!({
                            "path":destination,
                            "rollback_errors":rollback_errors,
                            "staging_path":staging,
                        }),
                    ));
                }
                Err(error) => {
                    let rollback_errors = rollback_sequence_outputs(&published);
                    preserve_staging = !rollback_errors.is_empty();
                    return Err(PotError::with_details(
                        ErrorCode::IoError,
                        error.to_string(),
                        json!({
                            "path":destination,
                            "rollback_errors":rollback_errors,
                            "staging_path":staging,
                        }),
                    ));
                }
            };
            published.push((destination.clone(), backup, false));
            if let Err(error) = fs::rename(&staged[index], destination) {
                let rollback_errors = rollback_sequence_outputs(&published);
                preserve_staging = !rollback_errors.is_empty();
                return Err(PotError::with_details(
                    ErrorCode::IoError,
                    error.to_string(),
                    json!({
                        "path":destination,
                        "rollback_errors":rollback_errors,
                        "staging_path":staging,
                    }),
                ));
            }
            if let Some((_, _, was_published)) = published.last_mut() {
                *was_published = true;
            }
        }
        if let Ok(directory) = fs::File::open(parent) {
            let _ = directory.sync_all();
        }
        Ok(())
    })();
    if !preserve_staging {
        let _ = fs::remove_dir_all(&staging);
    }
    result
}

fn rollback_sequence_outputs(published: &[(PathBuf, Option<PathBuf>, bool)]) -> Vec<String> {
    let mut errors = Vec::new();
    for (destination, backup, was_published) in published.iter().rev() {
        if *was_published
            && let Err(error) = fs::remove_file(destination)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            errors.push(format!("remove {}: {error}", destination.display()));
        }
        if let Some(backup) = backup
            && let Err(error) = fs::rename(backup, destination)
        {
            errors.push(format!(
                "restore {} from {}: {error}",
                destination.display(),
                backup.display()
            ));
        }
    }
    errors
}
fn resolve_path(uri: &str) -> PathBuf {
    PathBuf::from(uri.strip_prefix("file://").unwrap_or(uri))
}
