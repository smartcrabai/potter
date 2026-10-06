use serde_json::{Map, Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    library::linked_key,
    model::{Id, is_valid_id},
};

use super::{ChangeKind, Engine, check_fields, operation_pointer};

const ASSET_FIELDS: &[&str] = &[
    "catalog_id",
    "description",
    "author",
    "license",
    "tags",
    "preview",
];

pub(super) fn apply(
    engine: &mut Engine<'_>,
    name: &str,
    operation: &Map<String, Value>,
) -> Result<bool> {
    match name {
        "asset.mark" => mark(engine, operation),
        "asset.update" => update(engine, operation),
        "asset.clear" => clear(engine, operation),
        "asset.catalog_create" => catalog_create(engine, operation),
        _ => Err(engine.error(
            ErrorCode::InternalError,
            "invalid asset operation dispatch",
            &operation_pointer(engine.operation_index, "op"),
        )),
    }
}

fn mark(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &[
            "op",
            "target",
            "catalog_id",
            "description",
            "author",
            "license",
            "tags",
            "preview",
        ],
        &["target"],
    )?;
    let (registry, id) =
        super::target_registry(engine, operation, "target registry is not an ID registry")?;
    super::library::ensure_editable(engine, &registry, &id)?;
    let metadata = metadata_from_fields(engine, operation, false)?;
    let assets = super::object_slot(engine, "assets", super::ObjectSlotError::RegistryName)?;
    let key = linked_key(&registry, &id);
    if assets.contains_key(&key) {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("asset metadata for `{key}` already exists"),
            &operation_pointer(engine.operation_index, "target"),
        ));
    }
    assets.insert(key, metadata);
    mark_compatibility(engine, "assets");
    Ok(true)
}

fn update(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "set"],
        &["target", "set"],
    )?;
    let (registry, id) =
        super::target_registry(engine, operation, "target registry is not an ID registry")?;
    super::library::ensure_editable(engine, &registry, &id)?;
    let key = linked_key(&registry, &id);
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
    if set
        .keys()
        .any(|field| !ASSET_FIELDS.contains(&field.as_str()))
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "asset metadata contains an unknown field",
            &operation_pointer(engine.operation_index, "set"),
        ));
    }
    let parsed = metadata_from_fields(engine, set, true)?;
    let assets = super::object_slot(engine, "assets", super::ObjectSlotError::RegistryName)?;
    let metadata = assets
        .get_mut(&key)
        .and_then(Value::as_object_mut)
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::TargetNotFound,
                format!("asset metadata for `{key}` does not exist"),
            )
        })?;
    if let Some(values) = parsed.as_object() {
        for (field, value) in values {
            metadata.insert(field.clone(), value.clone());
        }
    }
    mark_compatibility(engine, "assets");
    Ok(true)
}

fn clear(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(engine, operation, &["op", "target"], &["target"])?;
    let (registry, id) =
        super::target_registry(engine, operation, "target registry is not an ID registry")?;
    super::library::ensure_editable(engine, &registry, &id)?;
    let key = linked_key(&registry, &id);
    let assets = super::object_slot(engine, "assets", super::ObjectSlotError::RegistryName)?;
    if assets.remove(&key).is_none() {
        return Err(PotError::new(
            ErrorCode::TargetNotFound,
            format!("asset metadata for `{key}` does not exist"),
        ));
    }
    mark_compatibility(engine, "assets");
    Ok(true)
}

fn catalog_create(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "id", "name", "parent_id"],
        &["id", "name"],
    )?;
    let id = operation.get("id").and_then(Value::as_str).ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "catalog id must be a string",
            &operation_pointer(engine.operation_index, "id"),
        )
    })?;
    if !is_valid_id(id) {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "catalog id is invalid",
            &operation_pointer(engine.operation_index, "id"),
        ));
    }
    let name = operation
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.trim().is_empty())
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "catalog name must be a non-empty string",
                &operation_pointer(engine.operation_index, "name"),
            )
        })?;
    let parent = operation
        .get("parent_id")
        .map(|value| {
            value.as_str().ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "parent_id must be a string",
                    &operation_pointer(engine.operation_index, "parent_id"),
                )
            })
        })
        .transpose()?;
    if let Some(parent_id) = parent
        && !engine
            .doc
            .compatibility
            .get("asset_catalogs")
            .and_then(Value::as_object)
            .is_some_and(|catalogs| catalogs.contains_key(parent_id))
    {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            format!("parent asset catalog `{parent_id}` does not exist"),
            &operation_pointer(engine.operation_index, "parent_id"),
        ));
    }
    let catalogs = super::object_slot(
        engine,
        "asset_catalogs",
        super::ObjectSlotError::RegistryName,
    )?;
    if catalogs.contains_key(id) {
        return Err(PotError::new(
            ErrorCode::IdExists,
            format!("asset catalog `{id}` already exists"),
        ));
    }
    catalogs.insert(id.to_owned(), json!({"name":name,"parent_id":parent}));
    mark_compatibility(engine, "asset_catalogs");
    Ok(true)
}

fn metadata_from_fields(
    engine: &Engine<'_>,
    fields: &Map<String, Value>,
    sparse: bool,
) -> Result<Value> {
    let mut result = Map::new();
    for field in ASSET_FIELDS {
        let Some(value) = fields.get(*field) else {
            continue;
        };
        match *field {
            "catalog_id" => {
                let id = value.as_str().filter(|id| is_valid_id(id)).ok_or_else(|| {
                    engine.error(
                        ErrorCode::InvalidOperation,
                        "catalog_id must be a valid ID",
                        &operation_pointer(engine.operation_index, field),
                    )
                })?;
                if !engine
                    .doc
                    .compatibility
                    .get("asset_catalogs")
                    .and_then(Value::as_object)
                    .is_some_and(|catalogs| catalogs.contains_key(id))
                {
                    return Err(engine.error(
                        ErrorCode::TargetNotFound,
                        format!("asset catalog `{id}` does not exist"),
                        &operation_pointer(engine.operation_index, field),
                    ));
                }
            }
            "description" | "author" | "license" => {
                if !value.is_string() {
                    return Err(engine.error(
                        ErrorCode::InvalidOperation,
                        format!("{field} must be a string"),
                        &operation_pointer(engine.operation_index, field),
                    ));
                }
            }
            "tags" => {
                let tags = value.as_array().ok_or_else(|| {
                    engine.error(
                        ErrorCode::InvalidOperation,
                        "tags must be an array",
                        &operation_pointer(engine.operation_index, field),
                    )
                })?;
                for tag in tags {
                    if !tag.as_str().is_some_and(is_valid_id) {
                        return Err(engine.error(
                            ErrorCode::InvalidOperation,
                            "asset tags must be valid IDs",
                            &operation_pointer(engine.operation_index, field),
                        ));
                    }
                }
            }
            "preview" if !(value.is_string() || value.is_null()) => {
                return Err(engine.error(
                    ErrorCode::InvalidOperation,
                    "preview must be a hash/path string or null",
                    &operation_pointer(engine.operation_index, field),
                ));
            }
            _ => {}
        }
        result.insert((*field).to_owned(), value.clone());
    }
    if !sparse {
        result.entry("tags".to_owned()).or_insert_with(|| json!([]));
    }
    Ok(Value::Object(result))
}

fn mark_compatibility(engine: &mut Engine<'_>, key: &str) {
    if let Ok(id) = Id::new(key) {
        engine.mark("compatibility", &id, ChangeKind::Updated);
    }
}
