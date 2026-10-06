use std::sync::LazyLock;

use serde_json::{Map, Value, json};

static PARAM_TABLE: LazyLock<Value> =
    LazyLock::new(
        || match serde_json::from_str(include_str!("param_table.json")) {
            Ok(value) => value,
            Err(error) => panic!("invalid compiled Blender parameter table: {error}"),
        },
    );

#[derive(Clone, Copy)]
pub(crate) enum ParameterFamily {
    Modifier,
    Constraint,
}

impl ParameterFamily {
    fn table_key(self) -> &'static str {
        match self {
            Self::Modifier => "modifiers",
            Self::Constraint => "constraints",
        }
    }
    fn label(self) -> &'static str {
        match self {
            Self::Modifier => "modifier",
            Self::Constraint => "constraint",
        }
    }
}
pub(crate) fn type_names(family: ParameterFamily) -> Vec<&'static str> {
    type_specs(family)
        .into_iter()
        .flat_map(Map::keys)
        .map(String::as_str)
        .collect()
}

pub(crate) fn type_specs(family: ParameterFamily) -> Option<&'static Map<String, Value>> {
    PARAM_TABLE.get(family.table_key())?.as_object()
}

pub(crate) fn type_spec(family: ParameterFamily, kind: &str) -> Option<&'static Value> {
    type_specs(family)?.get(kind)
}

pub(crate) fn params_schema(family: ParameterFamily, kind: &str) -> Option<Value> {
    let type_spec = type_spec(family, kind)?;
    Some(json!({
        "type":"object",
        "properties":type_spec.get("properties")?,
        "required":type_spec.get("required")?,
        "additionalProperties":false
    }))
}

pub(crate) fn default_value(
    family: ParameterFamily,
    kind: &str,
    parameter: &str,
) -> Option<&'static Value> {
    type_spec(family, kind)?
        .get("properties")?
        .get(parameter)?
        .get("default")
}
pub(crate) fn is_id_reference_parameter(
    family: ParameterFamily,
    kind: &str,
    parameter: &str,
) -> bool {
    fn has_id_pattern(schema: &Value) -> bool {
        if schema.get("pattern").and_then(Value::as_str) == Some("^[a-z][a-z0-9_-]{0,63}$") {
            return true;
        }
        match schema {
            Value::Object(properties) => properties.values().any(has_id_pattern),
            Value::Array(items) => items.iter().any(has_id_pattern),
            _ => false,
        }
    }
    type_spec(family, kind)
        .and_then(|specification| specification.get("properties"))
        .and_then(Value::as_object)
        .and_then(|properties| properties.get(parameter))
        .is_some_and(has_id_pattern)
}

#[derive(Debug)]
pub(crate) struct ParameterError {
    pub pointer: String,
    pub message: String,
}

pub(crate) fn validate_params(
    family: ParameterFamily,
    kind: &str,
    params: &Map<String, Value>,
    base_pointer: &str,
) -> Result<(), ParameterError> {
    let Some(type_spec) = type_spec(family, kind) else {
        return Ok(());
    };
    let Some(properties) = type_spec.get("properties").and_then(Value::as_object) else {
        return Ok(());
    };
    for (name, value) in params {
        let pointer = format!("{base_pointer}/{}", escape_pointer(name));
        let Some(property_schema) = properties.get(name) else {
            return Err(ParameterError {
                pointer,
                message: format!(
                    "unknown {} parameter `{name}` for type `{kind}`",
                    family.label()
                ),
            });
        };
        validate_value(value, property_schema).map_err(|expected| ParameterError {
            pointer,
            message: format!(
                "invalid {} parameter `{name}` for type `{kind}`; expected {expected}",
                family.label()
            ),
        })?;
    }
    if let Some(required) = type_spec.get("required").and_then(Value::as_array) {
        for name in required.iter().filter_map(Value::as_str) {
            if !params.contains_key(name) {
                return Err(ParameterError {
                    pointer: format!("{base_pointer}/{}", escape_pointer(name)),
                    message: format!(
                        "{} type `{kind}` requires parameter `{name}`",
                        family.label()
                    ),
                });
            }
        }
    }
    Ok(())
}

fn validate_value(value: &Value, schema: &Value) -> Result<(), String> {
    if let Some(values) = schema.get("enum").and_then(Value::as_array) {
        return validate_enum(value, values);
    }
    if let Some(one_of) = schema.get("oneOf").and_then(Value::as_array) {
        if one_of
            .iter()
            .any(|candidate| validate_value(value, candidate).is_ok())
        {
            return Ok(());
        }
        return Err("a value matching one of the allowed schemas".to_owned());
    }
    let accepted_types = schema
        .get("type")
        .map(|value| match value {
            Value::Array(values) => values.iter().filter_map(Value::as_str).collect::<Vec<_>>(),
            Value::String(name) => vec![name.as_str()],
            _ => Vec::new(),
        })
        .unwrap_or_default();
    if !accepted_types.is_empty()
        && !accepted_types
            .iter()
            .any(|name| value_matches_type(value, name))
    {
        return Err(accepted_types.join(" or "));
    }
    if let Some(pattern) = schema.get("pattern").and_then(Value::as_str)
        && pattern == "^[a-z][a-z0-9_-]{0,63}$"
        && !value.as_str().is_some_and(crate::model::is_valid_id)
    {
        return Err("a valid Potter ID".to_owned());
    }
    if let Some(number) = value.as_f64() {
        if schema
            .get("minimum")
            .and_then(Value::as_f64)
            .is_some_and(|minimum| number < minimum)
        {
            return Err(format!("a number no less than {}", schema["minimum"]));
        }
        if schema
            .get("maximum")
            .and_then(Value::as_f64)
            .is_some_and(|maximum| number > maximum)
        {
            return Err(format!("a number no greater than {}", schema["maximum"]));
        }
    }
    if let Some(text) = value.as_str()
        && schema
            .get("minLength")
            .and_then(Value::as_u64)
            .is_some_and(|minimum| text.chars().count() < minimum as usize)
    {
        return Err("a non-empty string".to_owned());
    }
    if let Some(array) = value.as_array() {
        if schema
            .get("minItems")
            .and_then(Value::as_u64)
            .is_some_and(|minimum| array.len() < minimum as usize)
            || schema
                .get("maxItems")
                .and_then(Value::as_u64)
                .is_some_and(|maximum| array.len() > maximum as usize)
        {
            return Err(format!("an array with {} items", schema["minItems"]));
        }
        if let Some(items_schema) = schema.get("items") {
            for item in array {
                validate_value(item, items_schema)?;
            }
        }
    }
    Ok(())
}

fn validate_enum(value: &Value, values: &[Value]) -> Result<(), String> {
    let allowed = values.iter().filter_map(Value::as_str).collect::<Vec<_>>();
    let valid = if let Some(items) = value.as_array() {
        items
            .iter()
            .all(|item| item.as_str().is_some_and(|item| allowed.contains(&item)))
    } else {
        value.as_str().is_some_and(|item| allowed.contains(&item))
    };
    if valid {
        Ok(())
    } else {
        Err(format!("one of [{}]", allowed.join(", ")))
    }
}

fn value_matches_type(value: &Value, expected: &str) -> bool {
    match expected {
        "null" => value.is_null(),
        "boolean" => value.is_boolean(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "number" => value.is_number(),
        "string" => value.is_string(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        _ => false,
    }
}

fn escape_pointer(value: &str) -> String {
    value.replace('~', "~0").replace('/', "~1")
}
