use serde_json::{Map, Number, Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    library::linked_key,
    model::Id,
};

use super::{ChangeKind, Engine, check_fields, operation_pointer};

const TYPES: &[&str] = &["bool", "int", "float", "string", "array", "object"];

pub(super) fn apply(
    engine: &mut Engine<'_>,
    name: &str,
    operation: &Map<String, Value>,
) -> Result<bool> {
    match name {
        "property.set" => set(engine, operation),
        "property.delete" => delete(engine, operation),
        _ => Err(engine.error(
            ErrorCode::InternalError,
            "invalid property operation dispatch",
            &operation_pointer(engine.operation_index, "op"),
        )),
    }
}

fn set(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &[
            "op", "target", "name", "type", "value", "subtype", "min", "max",
        ],
        &["target", "name", "type", "value"],
    )?;
    let (registry, id) = super::target_registry(
        engine,
        operation,
        "unsupported custom property target registry",
    )?;
    ensure_editable(engine, &registry, &id)?;
    let name = operation
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "property name must be a non-empty string",
                &operation_pointer(engine.operation_index, "name"),
            )
        })?;
    let kind = operation
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "property type must be a string",
                &operation_pointer(engine.operation_index, "type"),
            )
        })?;
    if !TYPES.contains(&kind) {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("unsupported property type `{kind}`"),
            &operation_pointer(engine.operation_index, "type"),
        ));
    }
    let value = operation.get("value").ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "property value is required",
            &operation_pointer(engine.operation_index, "value"),
        )
    })?;
    validate_value(engine, value, kind)?;
    let min = numeric_bound(engine, operation, "min")?;
    let max = numeric_bound(engine, operation, "max")?;
    if min
        .zip(max)
        .is_some_and(|(minimum, maximum)| minimum > maximum)
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "property min exceeds max",
            &operation_pointer(engine.operation_index, "min"),
        ));
    }
    if (min.is_some() || max.is_some()) && !["int", "float"].contains(&kind) {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "min/max are only valid for numeric properties",
            &operation_pointer(engine.operation_index, "min"),
        ));
    }
    if let Some(number) = value.as_f64()
        && (min.is_some_and(|bound| number < bound) || max.is_some_and(|bound| number > bound))
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "property value is outside min/max",
            &operation_pointer(engine.operation_index, "value"),
        ));
    }
    let subtype = operation
        .get("subtype")
        .map(|value| {
            value
                .as_str()
                .filter(|text| !text.trim().is_empty())
                .map(str::to_owned)
                .ok_or_else(|| {
                    engine.error(
                        ErrorCode::InvalidOperation,
                        "property subtype must be a non-empty string",
                        &operation_pointer(engine.operation_index, "subtype"),
                    )
                })
        })
        .transpose()?;
    let mut property = Map::new();
    property.insert("type".to_owned(), Value::String(kind.to_owned()));
    property.insert("value".to_owned(), value.clone());
    if let Some(subtype) = subtype {
        property.insert("subtype".to_owned(), Value::String(subtype));
    }
    if let Some(min) = min {
        property.insert(
            "min".to_owned(),
            Number::from_f64(min).map_or(Value::Null, Value::Number),
        );
    }
    if let Some(max) = max {
        property.insert(
            "max".to_owned(),
            Number::from_f64(max).map_or(Value::Null, Value::Number),
        );
    }
    if registry == "nodes" {
        let missing_node = engine.error(
            ErrorCode::TargetNotFound,
            format!("node `{id}` does not exist"),
            &operation_pointer(engine.operation_index, "target"),
        );
        let node = engine.doc.nodes.get_mut(&id).ok_or(missing_node)?;
        node.properties
            .insert(name.to_owned(), Value::Object(property));
        engine.mark("nodes", &id, ChangeKind::Updated);
    } else {
        let key = linked_key(&registry, &id);
        let properties =
            super::object_slot(engine, "properties", super::ObjectSlotError::RegistryName)?;
        let target_props = properties
            .entry(key)
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::SceneInvalid,
                    "custom property registry entry must be an object",
                )
            })?;
        target_props.insert(name.to_owned(), Value::Object(property));
        mark_compatibility(engine, "properties");
    }
    Ok(true)
}

fn delete(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "name"],
        &["target", "name"],
    )?;
    let (registry, id) = super::target_registry(
        engine,
        operation,
        "unsupported custom property target registry",
    )?;
    ensure_editable(engine, &registry, &id)?;
    let name = operation
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "property name must be a non-empty string",
                &operation_pointer(engine.operation_index, "name"),
            )
        })?;
    let key = linked_key(&registry, &id);
    let missing_node = engine.error(
        ErrorCode::TargetNotFound,
        format!("node `{id}` does not exist"),
        &operation_pointer(engine.operation_index, "target"),
    );
    let missing_properties = engine.error(
        ErrorCode::TargetNotFound,
        format!("custom properties for `{key}` do not exist"),
        &operation_pointer(engine.operation_index, "target"),
    );
    let removed = if registry == "nodes" {
        engine
            .doc
            .nodes
            .get_mut(&id)
            .ok_or(missing_node)?
            .properties
            .remove(name)
            .is_some()
    } else {
        let properties = engine
            .doc
            .compatibility
            .get_mut("properties")
            .and_then(Value::as_object_mut)
            .ok_or(missing_properties)?;
        properties
            .get_mut(&key)
            .and_then(Value::as_object_mut)
            .is_some_and(|values| values.remove(name).is_some())
    };
    if !removed {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            format!("custom property `{name}` does not exist"),
            &operation_pointer(engine.operation_index, "name"),
        ));
    }
    if registry == "nodes" {
        engine.mark("nodes", &id, ChangeKind::Updated);
    } else {
        mark_compatibility(engine, "properties");
    }
    Ok(true)
}

fn ensure_editable(engine: &Engine<'_>, registry: &str, id: &Id) -> Result<()> {
    if engine
        .doc
        .compatibility
        .get("linked_ids")
        .and_then(Value::as_object)
        .is_some_and(|linked| linked.contains_key(&linked_key(registry, id)))
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "linked data is read-only; create an override first",
            &operation_pointer(engine.operation_index, "target"),
        ));
    }
    Ok(())
}

fn validate_value(engine: &Engine<'_>, value: &Value, kind: &str) -> Result<()> {
    let valid = match kind {
        "bool" => value.is_boolean(),
        "int" => value.as_i64().is_some() || value.as_u64().is_some(),
        "float" => value.as_f64().is_some_and(f64::is_finite),
        "string" => value.is_string(),
        "array" => value
            .as_array()
            .is_some_and(|items| items.iter().all(json_finite)),
        "object" => value
            .as_object()
            .is_some_and(|items| items.values().all(json_finite)),
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("property value does not match type `{kind}`"),
            &operation_pointer(engine.operation_index, "value"),
        ))
    }
}

fn json_finite(value: &Value) -> bool {
    match value {
        Value::Number(number) => number.as_f64().is_some_and(f64::is_finite),
        Value::Array(values) => values.iter().all(json_finite),
        Value::Object(values) => values.values().all(json_finite),
        _ => true,
    }
}

fn numeric_bound(
    engine: &Engine<'_>,
    operation: &Map<String, Value>,
    field: &str,
) -> Result<Option<f64>> {
    operation
        .get(field)
        .map(|value| {
            value
                .as_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| {
                    engine.error(
                        ErrorCode::InvalidOperation,
                        format!("{field} must be a finite number"),
                        &operation_pointer(engine.operation_index, field),
                    )
                })
        })
        .transpose()
}

fn mark_compatibility(engine: &mut Engine<'_>, key: &str) {
    if let Ok(id) = Id::new(key) {
        engine.mark("compatibility", &id, ChangeKind::Updated);
    }
}
