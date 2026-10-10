use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

use crate::{
    error::{ErrorCode, PotError, Result},
    model::{DataBlock, Id},
};

use super::{ChangeKind, Engine, operation_pointer, parse_id, read_id, resolve_node_targets};

pub(in crate::ops) fn create_geometry_node(
    engine: &mut Engine<'_>,
    operation: &Map<String, Value>,
    kind: &str,
    suffix: &str,
    data_block: DataBlock,
) -> Result<bool> {
    let id = read_id(engine, operation, "id")?;
    let data_id = if let Some(value) = operation.get("data_id") {
        let text = value.as_str().ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "data_id must be a string",
                &operation_pointer(engine.operation_index, "data_id"),
            )
        })?;
        parse_id(
            engine,
            text,
            &operation_pointer(engine.operation_index, "data_id"),
        )?
    } else {
        parse_id(
            engine,
            &format!("{id}{suffix}"),
            &operation_pointer(engine.operation_index, "id"),
        )?
    };
    if engine.doc.data_blocks.contains_key(&data_id) {
        return Err(engine.error(
            ErrorCode::IdExists,
            format!("data-block ID `{data_id}` already exists"),
            &operation_pointer(engine.operation_index, "data_id"),
        ));
    }
    engine.doc.data_blocks.insert(data_id.clone(), data_block);
    engine.mark("data_blocks", &data_id, ChangeKind::Created);

    let mut node_operation = Map::new();
    node_operation.insert("op".to_owned(), Value::String("node.create".to_owned()));
    node_operation.insert("kind".to_owned(), Value::String(kind.to_owned()));
    node_operation.insert("id".to_owned(), Value::String(id.to_string()));
    node_operation.insert("data".to_owned(), Value::String(data_id.to_string()));
    for field in [
        "name",
        "collection",
        "transform",
        "parent",
        "parent_inverse",
        "materials",
        "visible",
        "render_visible",
        "selectable",
    ] {
        if let Some(value) = operation.get(field) {
            node_operation.insert(field.to_owned(), value.clone());
        }
    }
    super::node::apply(engine, "node.create", &node_operation)
}

pub(in crate::ops) fn typed_node_data_id(
    engine: &Engine<'_>,
    operation: &Map<String, Value>,
    kind: &str,
    data_type: &str,
) -> Result<(Id, Id)> {
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
    if node.kind != kind {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("target node must be a {kind}"),
            &operation_pointer(engine.operation_index, "target"),
        ));
    }
    let data_id = node.data.clone().ok_or_else(|| {
        engine.error(
            ErrorCode::SceneInvalid,
            format!("{kind} node has no data block"),
            &operation_pointer(engine.operation_index, "target"),
        )
    })?;
    let block = engine.doc.data_blocks.get(&data_id).ok_or_else(|| {
        engine.error(
            ErrorCode::SceneInvalid,
            format!("{kind} node data block `{data_id}` is missing"),
            &operation_pointer(engine.operation_index, "target"),
        )
    })?;
    if block.data_type != data_type {
        return Err(engine.error(
            ErrorCode::SceneInvalid,
            format!("{kind} node must reference a {data_type} Data-Block"),
            &operation_pointer(engine.operation_index, "target"),
        ));
    }
    Ok((node_id, data_id))
}

pub(in crate::ops) fn selected<T: DeserializeOwned>(
    engine: &Engine<'_>,
    operation: &Map<String, Value>,
    fields: &[&str],
    pointer: &str,
) -> Result<T> {
    let mut values = Map::new();
    for field in fields {
        if let Some(value) = operation.get(*field) {
            values.insert((*field).to_owned(), value.clone());
        }
    }
    from_value(engine, Value::Object(values), pointer)
}

pub(in crate::ops) fn from_value<T: DeserializeOwned>(
    engine: &Engine<'_>,
    value: Value,
    pointer: &str,
) -> Result<T> {
    serde_json::from_value(value).map_err(|error| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("invalid geometry data: {error}"),
            &operation_pointer(engine.operation_index, pointer),
        )
    })
}

pub(in crate::ops) fn read_set<'a>(
    engine: &Engine<'_>,
    operation: &'a Map<String, Value>,
) -> Result<&'a Map<String, Value>> {
    operation
        .get("set")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "set must be an object",
                &operation_pointer(engine.operation_index, "set"),
            )
        })
}

pub(in crate::ops) fn merge_set<T: DeserializeOwned + serde::Serialize>(
    engine: &Engine<'_>,
    current: &T,
    set: &Map<String, Value>,
    pointer: &str,
) -> Result<T> {
    let mut merged = serde_json::to_value(current)
        .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))?;
    let target = merged.as_object_mut().ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "serialized geometry data is not an object",
        )
    })?;
    for (key, value) in set {
        target.insert(key.clone(), value.clone());
    }
    from_value(engine, merged, pointer)
}

pub(in crate::ops) fn missing_payload(engine: &Engine<'_>, name: &str, field: &str) -> PotError {
    engine.error(
        ErrorCode::SceneInvalid,
        format!("{name} Data-Block has no {name} payload"),
        &operation_pointer(engine.operation_index, field),
    )
}
