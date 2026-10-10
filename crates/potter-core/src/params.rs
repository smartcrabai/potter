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
#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests")]

    use serde_json::{Map, Value, json};

    use super::{
        ParameterFamily, default_value, is_id_reference_parameter, params_schema, type_names,
        type_spec, type_specs, validate_params, validate_value,
    };

    #[test]
    fn schema_lookup_preserves_modifier_and_constraint_contracts() {
        let modifier_names = type_names(ParameterFamily::Modifier);
        assert!(modifier_names.contains(&"data_transfer"));
        assert!(modifier_names.contains(&"bevel"));
        let constraint_names = type_names(ParameterFamily::Constraint);
        assert!(constraint_names.contains(&"copy_location"));

        let specifications = type_specs(ParameterFamily::Modifier).unwrap();
        assert!(specifications.contains_key("data_transfer"));
        assert!(type_spec(ParameterFamily::Modifier, "missing").is_none());

        let schema = params_schema(ParameterFamily::Modifier, "data_transfer").unwrap();
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["additionalProperties"], false);
        assert_eq!(
            schema["properties"],
            type_spec(ParameterFamily::Modifier, "data_transfer").unwrap()["properties"]
        );
        assert_eq!(
            schema["required"],
            type_spec(ParameterFamily::Modifier, "data_transfer").unwrap()["required"]
        );

        assert_eq!(
            default_value(
                ParameterFamily::Modifier,
                "data_transfer",
                "show_in_editmode"
            )
            .and_then(Value::as_bool),
            Some(false)
        );
        assert!(default_value(ParameterFamily::Modifier, "data_transfer", "missing").is_none());
    }

    #[test]
    fn id_reference_detection_finds_nested_patterns_without_matching_unrelated_parameters() {
        assert!(is_id_reference_parameter(
            ParameterFamily::Modifier,
            "data_transfer",
            "object"
        ));
        assert!(!is_id_reference_parameter(
            ParameterFamily::Modifier,
            "data_transfer",
            "show_in_editmode"
        ));
        assert!(!is_id_reference_parameter(
            ParameterFamily::Modifier,
            "missing",
            "object"
        ));
    }

    #[test]
    fn parameter_validation_reports_required_values_and_escaped_unknown_names() {
        let required_error = validate_params(
            ParameterFamily::Modifier,
            "mesh_cache",
            &Map::new(),
            "/modifiers",
        )
        .unwrap_err();
        assert_eq!(required_error.pointer, "/modifiers/resource");
        assert!(
            required_error
                .message
                .contains("requires parameter `resource`")
        );

        let mut parameters = Map::new();
        parameters.insert("missing~/value".to_owned(), json!(1));
        let unknown_error = validate_params(
            ParameterFamily::Modifier,
            "data_transfer",
            &parameters,
            "/modifiers/data_transfer/params",
        )
        .unwrap_err();
        assert_eq!(
            unknown_error.pointer,
            "/modifiers/data_transfer/params/missing~0~1value"
        );
        assert!(unknown_error.message.contains("unknown modifier parameter"));
    }

    #[test]
    fn value_validation_matches_declared_types_and_one_of_alternatives() {
        let type_cases = [
            ("null", Value::Null, json!(false)),
            ("boolean", json!(true), Value::Null),
            ("integer", json!(2), json!(2.5)),
            ("number", json!(2.5), json!("2.5")),
            ("string", json!("name"), json!(2)),
            ("array", json!([]), json!({})),
            ("object", json!({}), json!([])),
        ];
        for (expected, valid, invalid) in type_cases {
            let schema = json!({"type":expected});
            assert!(validate_value(&valid, &schema).is_ok(), "{expected}");
            assert!(validate_value(&invalid, &schema).is_err(), "{expected}");
        }
        let integer = json!({"type":"integer"});
        assert!(validate_value(&json!(-2), &integer).is_ok());
        assert!(validate_value(&json!(u64::MAX), &integer).is_ok());

        let union = json!({"type":["string","null"]});
        assert!(validate_value(&Value::Null, &union).is_ok());
        assert!(validate_value(&json!("name"), &union).is_ok());
        assert!(validate_value(&json!(true), &union).is_err());

        let one_of = json!({
            "oneOf":[
                {"type":"string","pattern":"^[a-z][a-z0-9_-]{0,63}$"},
                {"type":"null"}
            ]
        });
        assert!(validate_value(&json!("node_1"), &one_of).is_ok());
        assert!(validate_value(&Value::Null, &one_of).is_ok());
        assert!(validate_value(&json!("Not an ID"), &one_of).is_err());
    }

    #[test]
    fn value_validation_enforces_enum_and_numeric_boundaries() {
        let enumeration = json!({"enum":["REPLACE","ADD"]});
        assert!(validate_value(&json!("REPLACE"), &enumeration).is_ok());
        assert!(validate_value(&json!("MULTIPLY"), &enumeration).is_err());
        assert!(validate_value(&json!(["REPLACE", "ADD"]), &enumeration).is_ok());
        assert!(validate_value(&json!(["REPLACE", "MULTIPLY"]), &enumeration).is_err());

        let bounded = json!({"type":"number","minimum":0.0,"maximum":1.0});
        assert!(validate_value(&json!(0.0), &bounded).is_ok());
        assert!(validate_value(&json!(1.0), &bounded).is_ok());
        assert!(validate_value(&json!(-0.1), &bounded).is_err());
        assert!(validate_value(&json!(1.1), &bounded).is_err());
    }

    #[test]
    fn value_validation_enforces_string_and_array_size_boundaries() {
        let text = json!({"type":"string","minLength":2});
        assert!(validate_value(&json!("éx"), &text).is_ok());
        assert!(validate_value(&json!("x"), &text).is_err());

        let array = json!({
            "type":"array",
            "minItems":2,
            "maxItems":3,
            "items":{"type":"string"}
        });
        assert!(validate_value(&json!(["a", "b"]), &array).is_ok());
        assert!(validate_value(&json!(["a", "b", "c"]), &array).is_ok());
        assert!(validate_value(&json!(["a"]), &array).is_err());
        assert!(validate_value(&json!(["a", "b", "c", "d"]), &array).is_err());
        assert!(validate_value(&json!(["a", 1]), &array).is_err());
    }
}

#[cfg(kani)]
mod kani_verification {
    use serde_json::Value;

    use super::value_matches_type;

    #[kani::proof]
    #[kani::unwind(8)]
    fn signed_json_integers_are_integers_and_numbers_only() {
        let value = Value::from(kani::any::<i64>());
        assert!(value_matches_type(&value, "integer"));
        assert!(value_matches_type(&value, "number"));
        assert!(!value_matches_type(&value, "string"));
        assert!(!value_matches_type(&value, "boolean"));
        assert!(!value_matches_type(&value, "null"));
    }

    #[kani::proof]
    #[kani::unwind(8)]
    fn float_backed_json_numbers_are_never_integers() {
        let float: f64 = kani::any();
        kani::assume(float.is_finite());
        let number = match serde_json::Number::from_f64(float) {
            Some(number) => number,
            None => {
                kani::assert(false, "finite floats must convert to JSON numbers");
                return;
            }
        };
        let value = Value::Number(number);
        assert!(value_matches_type(&value, "number"));
        assert!(!value_matches_type(&value, "integer"));
    }
}
