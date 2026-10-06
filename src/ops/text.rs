use serde_json::{Map, Value};

use crate::{
    error::{ErrorCode, Result},
    model::{DataBlock, Id, PrimitiveDescriptor, TextObjectData},
};

use super::{ChangeKind, Engine, check_fields, operation_pointer, read_id};

const TARGET_ID_POLICY: super::TargetIdPolicy = super::TargetIdPolicy::Strict {
    object_message: "target must be an object",
    shape_message: "target must contain only id",
    id_message: "target.id must be a string",
    require_id: false,
};

pub(super) fn apply(
    engine: &mut Engine<'_>,
    name: &str,
    operation: &Map<String, Value>,
) -> Result<bool> {
    match name {
        "text.create" => create(engine, operation),
        "text.update" => update(engine, operation),
        "text.delete" => delete(engine, operation),
        _ => Err(engine.error(
            ErrorCode::InternalError,
            "invalid text operation dispatch",
            &operation_pointer(engine.operation_index, "op"),
        )),
    }
}

fn create(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "id", "body", "language"],
        &["id", "body"],
    )?;
    let id = read_id(engine, operation, "id")?;
    if engine.doc.data_blocks.contains_key(&id) {
        return Err(engine.error(
            ErrorCode::IdExists,
            format!("Data-Block `{id}` already exists"),
            &operation_pointer(engine.operation_index, "id"),
        ));
    }
    let body = operation
        .get("body")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "text body must be a string",
                &operation_pointer(engine.operation_index, "body"),
            )
        })?;
    let language = match operation.get("language") {
        Some(value) => value.as_str().ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "text language must be a string",
                &operation_pointer(engine.operation_index, "language"),
            )
        })?,
        None => "plain",
    };
    if language.trim().is_empty() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "text language must not be empty",
            &operation_pointer(engine.operation_index, "language"),
        ));
    }
    let mut params = Map::new();
    params.insert("body".to_owned(), Value::String(body.to_owned()));
    params.insert("language".to_owned(), Value::String(language.to_owned()));
    engine.doc.data_blocks.insert(
        id.clone(),
        DataBlock {
            data_type: "script_text".to_owned(),
            descriptor: Some(PrimitiveDescriptor {
                primitive: "script_text".to_owned(),
                params,
            }),
            text: Some(TextObjectData {
                body: body.to_owned(),
                ..TextObjectData::default()
            }),
            mesh: None,
            camera: None,
            light: None,
            grease_pencil: None,
            armature: None,
            shape_keys: None,
            vertex_groups: Vec::new(),
            vertex_weights: std::collections::BTreeMap::default(),
            ..DataBlock::default()
        },
    );
    engine.mark("data_blocks", &id, ChangeKind::Created);
    Ok(true)
}

fn update(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "set"],
        &["target", "set"],
    )?;
    let id = super::target_id_value(
        engine,
        operation.get("target"),
        &operation_pointer(engine.operation_index, "target"),
        TARGET_ID_POLICY,
    )?;
    ensure_editable(engine, &id)?;
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
    if set.is_empty()
        || set
            .keys()
            .any(|key| !["body", "language"].contains(&key.as_str()))
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "text set must contain body and/or language only",
            &operation_pointer(engine.operation_index, "set"),
        ));
    }
    let updates = set
        .iter()
        .map(|(field, value)| {
            let text = value.as_str().ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    format!("text {field} must be a string"),
                    &operation_pointer(engine.operation_index, &format!("set/{field}")),
                )
            })?;
            if field == "language" && text.trim().is_empty() {
                return Err(engine.error(
                    ErrorCode::InvalidOperation,
                    "text language must not be empty",
                    &operation_pointer(engine.operation_index, "set/language"),
                ));
            }
            Ok((field.clone(), text.to_owned()))
        })
        .collect::<Result<Vec<_>>>()?;
    let missing_block = engine.error(
        ErrorCode::TargetNotFound,
        format!("script text `{id}` does not exist"),
        &operation_pointer(engine.operation_index, "target"),
    );
    let block = engine
        .doc
        .data_blocks
        .get_mut(&id)
        .filter(|block| block.data_type == "script_text")
        .ok_or(missing_block)?;
    let params = block.descriptor.get_or_insert_with(|| PrimitiveDescriptor {
        primitive: "script_text".to_owned(),
        params: Map::new(),
    });
    for (field, text) in updates {
        if field == "body" {
            block
                .text
                .get_or_insert_with(TextObjectData::default)
                .body
                .clone_from(&text);
        }
        params.params.insert(field, Value::String(text));
    }
    engine.mark("data_blocks", &id, ChangeKind::Updated);
    Ok(true)
}

fn delete(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(engine, operation, &["op", "target"], &["target"])?;
    let id = super::target_id_value(
        engine,
        operation.get("target"),
        &operation_pointer(engine.operation_index, "target"),
        TARGET_ID_POLICY,
    )?;
    ensure_editable(engine, &id)?;
    let block = engine.doc.data_blocks.get(&id).ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            format!("Data-Block `{id}` does not exist"),
            &operation_pointer(engine.operation_index, "target"),
        )
    })?;
    if block.data_type != "script_text" {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "text.delete target must be a script Text Data-Block",
            &operation_pointer(engine.operation_index, "target"),
        ));
    }
    engine.doc.data_blocks.remove(&id);
    engine.mark("data_blocks", &id, ChangeKind::Deleted);
    Ok(true)
}

fn ensure_editable(engine: &Engine<'_>, id: &Id) -> Result<()> {
    if engine
        .doc
        .compatibility
        .get("linked_ids")
        .and_then(Value::as_object)
        .is_some_and(|linked| linked.contains_key(&format!("data_blocks:{id}")))
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "linked script Text is read-only",
            &operation_pointer(engine.operation_index, "target"),
        ));
    }
    Ok(())
}
