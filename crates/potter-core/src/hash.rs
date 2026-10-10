use crate::error::{ErrorCode, PotError};

/// Serialize a JSON value using the canonicalization rules from RFC 8785.
///
/// # Errors
///
/// Returns an error if a number cannot be represented as a finite `f64` or if
/// serializing a numeric value fails.
pub fn canonicalize(value: &serde_json::Value) -> Result<Vec<u8>, PotError> {
    let mut output = Vec::new();
    write_canonical(value, &mut output)?;
    Ok(output)
}

fn write_canonical(value: &serde_json::Value, output: &mut Vec<u8>) -> Result<(), PotError> {
    match value {
        serde_json::Value::Null => output.extend_from_slice(b"null"),
        serde_json::Value::Bool(false) => output.extend_from_slice(b"false"),
        serde_json::Value::Bool(true) => output.extend_from_slice(b"true"),
        serde_json::Value::Number(number) => {
            let numeric = number.as_f64().ok_or_else(|| {
                PotError::new(
                    ErrorCode::InvalidArgument,
                    "number is not representable as f64",
                )
            })?;
            if !numeric.is_finite() {
                return Err(PotError::new(
                    ErrorCode::InvalidArgument,
                    "non-finite numbers cannot be canonicalized",
                ));
            }
            write_number(numeric, output)?;
        }
        serde_json::Value::String(string) => write_string(string, output)?,
        serde_json::Value::Array(items) => {
            output.push(b'[');
            for (index, item) in items.iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                write_canonical(item, output)?;
            }
            output.push(b']');
        }
        serde_json::Value::Object(object) => {
            output.push(b'{');
            let mut keys: Vec<&str> = object.keys().map(String::as_str).collect();
            keys.sort_unstable_by(|left, right| left.encode_utf16().cmp(right.encode_utf16()));
            for (index, key) in keys.into_iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                write_string(key, output)?;
                output.push(b':');
                let Some(item) = object.get(key) else {
                    return Err(PotError::new(
                        ErrorCode::InternalError,
                        "object key disappeared during canonicalization",
                    ));
                };
                write_canonical(item, output)?;
            }
            output.push(b'}');
        }
    }
    Ok(())
}

fn write_string(value: &str, output: &mut Vec<u8>) -> Result<(), PotError> {
    serde_json::to_writer(output, value).map_err(PotError::internal_json)
}

struct NumberParts {
    negative: bool,
    is_zero: bool,
    digits: [u8; 24],
    significant_start: usize,
    significant_count: usize,
    decimal_exponent: i32,
}

fn write_number(value: f64, output: &mut Vec<u8>) -> Result<(), PotError> {
    let start = output.len();
    serde_json::to_writer(&mut *output, &value).map_err(|error| {
        PotError::new(
            ErrorCode::InternalError,
            format!("number serialization failed: {error}"),
        )
    })?;
    let parts = parse_number(&output[start..])?;
    output.truncate(start);
    append_number(&parts, output)?;
    Ok(())
}

fn parse_number(raw: &[u8]) -> Result<NumberParts, PotError> {
    let negative = raw.first() == Some(&b'-');
    let unsigned = if negative { &raw[1..] } else { raw };
    let exponent_marker = unsigned.iter().position(|byte| matches!(byte, b'e' | b'E'));
    let (mantissa, exponent_bytes) = exponent_marker.map_or((unsigned, None), |index| {
        (&unsigned[..index], Some(&unsigned[index + 1..]))
    });
    let exponent = if let Some(bytes) = exponent_bytes {
        let text = std::str::from_utf8(bytes)
            .map_err(|_| PotError::new(ErrorCode::InternalError, "invalid number exponent"))?;
        text.parse::<i32>().map_err(|_| {
            PotError::new(ErrorCode::InternalError, "number exponent is out of range")
        })?
    } else {
        0
    };
    let decimal_marker = mantissa.iter().position(|&byte| byte == b'.');
    let (integer, fraction) = decimal_marker.map_or((mantissa, &[][..]), |index| {
        (&mantissa[..index], &mantissa[index + 1..])
    });
    let decimal_position = i32::try_from(integer.len())
        .map_err(|_| PotError::new(ErrorCode::InternalError, "number is too long"))?;

    let mut digits = [0_u8; 24];
    let mut digit_count = 0;
    let mut is_zero = true;
    for &byte in integer.iter().chain(fraction) {
        if !byte.is_ascii_digit() || digit_count == digits.len() {
            return Err(PotError::new(
                ErrorCode::InternalError,
                "unexpected number serialization",
            ));
        }
        digits[digit_count] = byte;
        digit_count += 1;
        is_zero &= byte == b'0';
    }
    if digit_count == 0 {
        return Err(PotError::new(
            ErrorCode::InternalError,
            "unexpected empty number serialization",
        ));
    }
    if is_zero {
        return Ok(NumberParts {
            negative,
            is_zero,
            digits,
            significant_start: 0,
            significant_count: 0,
            decimal_exponent: 0,
        });
    }

    let significant_start = digits[..digit_count]
        .iter()
        .position(|&digit| digit != b'0')
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "invalid nonzero number"))?;
    let leading_zeros = i32::try_from(significant_start).map_err(|_| {
        PotError::new(
            ErrorCode::InternalError,
            "number has too many leading zeroes",
        )
    })?;
    let decimal_exponent = exponent
        .checked_add(decimal_position)
        .and_then(|position| position.checked_sub(leading_zeros + 1))
        .ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "number exponent is out of range")
        })?;
    let mut significant_count = digit_count - significant_start;
    while significant_count > 1 && digits[significant_start + significant_count - 1] == b'0' {
        significant_count -= 1;
    }
    Ok(NumberParts {
        negative,
        is_zero,
        digits,
        significant_start,
        significant_count,
        decimal_exponent,
    })
}

fn append_number(parts: &NumberParts, output: &mut Vec<u8>) -> Result<(), PotError> {
    if parts.is_zero {
        output.push(b'0');
        return Ok(());
    }
    if parts.negative {
        output.push(b'-');
    }

    let digits = &parts.digits[parts.significant_start..];
    let count = parts.significant_count;
    if (0..21).contains(&parts.decimal_exponent) {
        let integer_digits = usize::try_from(parts.decimal_exponent + 1)
            .map_err(|_| PotError::new(ErrorCode::InternalError, "invalid decimal exponent"))?;
        let integer_part_len = count.min(integer_digits);
        output.extend_from_slice(&digits[..integer_part_len]);
        output.extend(std::iter::repeat_n(b'0', integer_digits - integer_part_len));
        if count > integer_digits {
            output.push(b'.');
            output.extend_from_slice(&digits[integer_digits..count]);
        }
    } else if (-6..0).contains(&parts.decimal_exponent) {
        output.extend_from_slice(b"0.");
        let zero_count = usize::try_from(-parts.decimal_exponent - 1)
            .map_err(|_| PotError::new(ErrorCode::InternalError, "invalid decimal exponent"))?;
        output.extend(std::iter::repeat_n(b'0', zero_count));
        output.extend_from_slice(&digits[..count]);
    } else {
        output.push(digits[0]);
        if count > 1 {
            output.push(b'.');
            output.extend_from_slice(&digits[1..count]);
        }
        output.push(b'e');
        if parts.decimal_exponent >= 0 {
            output.push(b'+');
        }
        append_exponent(parts.decimal_exponent, output)?;
    }
    Ok(())
}

fn append_exponent(exponent: i32, output: &mut Vec<u8>) -> Result<(), PotError> {
    if exponent < 0 {
        output.push(b'-');
    }
    let mut magnitude = exponent.unsigned_abs();
    let mut digits = [0_u8; 3];
    let mut len = 0;
    loop {
        digits[len] = b'0'
            + u8::try_from(magnitude % 10)
                .map_err(|_| PotError::new(ErrorCode::InternalError, "invalid exponent digit"))?;
        len += 1;
        magnitude /= 10;
        if magnitude == 0 {
            break;
        }
    }
    for digit in digits[..len].iter().rev() {
        output.push(*digit);
    }
    Ok(())
}

/// Compute a `sha256:<lowercase hex>` content address.
#[must_use]
pub fn sha256(bytes: &[u8]) -> String {
    use sha2::Digest as _;

    let digest = sha2::Sha256::digest(bytes);
    let mut result = String::with_capacity(7 + digest.len() * 2);
    result.push_str("sha256:");
    for byte in digest {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        result.push(char::from(HEX[usize::from(byte >> 4)]));
        result.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    result
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests")]

    use proptest::prelude::*;
    use serde_json::{Number, Value, json};

    use super::{canonicalize, sha256};

    fn json_values() -> impl Strategy<Value = Value> {
        let leaf = prop_oneof![
            Just(Value::Null),
            any::<bool>().prop_map(Value::Bool),
            (-9_007_199_254_740_991_i64..=9_007_199_254_740_991_i64)
                .prop_map(|number| Value::Number(Number::from(number))),
            any::<String>().prop_map(Value::String),
        ];
        leaf.prop_recursive(4, 64, 8, |inner| {
            prop_oneof![
                prop::collection::vec(inner.clone(), 0..8).prop_map(Value::Array),
                prop::collection::btree_map("[a-zA-Z0-9_]{0,8}", inner, 0..8)
                    .prop_map(|map| Value::Object(map.into_iter().collect())),
            ]
        })
    }

    fn semantically_equal(left: &Value, right: &Value) -> bool {
        match (left, right) {
            (Value::Number(left), Value::Number(right)) => left.as_f64() == right.as_f64(),
            (Value::Array(left), Value::Array(right)) => {
                left.len() == right.len()
                    && left
                        .iter()
                        .zip(right)
                        .all(|(left, right)| semantically_equal(left, right))
            }
            (Value::Object(left), Value::Object(right)) => {
                left.len() == right.len()
                    && left.iter().all(|(key, value)| {
                        right
                            .get(key)
                            .is_some_and(|other| semantically_equal(value, other))
                    })
            }
            _ => left == right,
        }
    }

    proptest! {
        #[test]
        fn canonical_json_parses_back_to_the_same_value(value in json_values()) {
            let canonical = canonicalize(&value).unwrap();
            let parsed: Value = serde_json::from_slice(&canonical).unwrap();
            prop_assert!(semantically_equal(&parsed, &value));
        }


        #[test]
        fn canonical_json_matches_sorted_serde_serialization_for_safe_integers(
            value in json_values(),
        ) {
            let sorted_json = serde_json::to_vec(&value).unwrap();
            prop_assert_eq!(canonicalize(&value).unwrap(), sorted_json);
        }
        #[test]
        fn canonical_json_is_independent_of_object_insertion_order(
            entries in prop::collection::btree_map("[a-z]{1,8}", any::<i32>(), 0..20)
        ) {
            let forward: serde_json::Map<String, Value> = entries.iter()
                .map(|(key, value)| (key.clone(), json!(value))).collect();
            let reverse: serde_json::Map<String, Value> = entries.into_iter().rev()
                .map(|(key, value)| (key, json!(value))).collect();
            prop_assert_eq!(canonicalize(&Value::Object(forward)).unwrap(),
                canonicalize(&Value::Object(reverse)).unwrap());
        }
    }

    #[test]
    fn numbers_use_ecmascript_json_formatting() {
        let cases = [
            ("1", "1"),
            ("-0", "0"),
            ("1.0", "1"),
            ("12.345", "12.345"),
            ("0.12345", "0.12345"),
            ("1e+20", "100000000000000000000"),
            ("1e+21", "1e+21"),
            ("1e-6", "0.000001"),
            ("1e-7", "1e-7"),
            ("1.234e+21", "1.234e+21"),
            ("1.234e-7", "1.234e-7"),
        ];

        for (input, expected) in cases {
            let value: Value = serde_json::from_str(input).unwrap();
            let actual = canonicalize(&value).unwrap();
            assert_eq!(actual, expected.as_bytes(), "input {input}");
        }

        let number = Number::from_f64(333_333_333.333_333_3).unwrap();
        assert_eq!(
            canonicalize(&Value::Number(number)).unwrap(),
            b"333333333.3333333"
        );
    }

    #[test]
    fn string_escapes_every_json_control_character() {
        let value = Value::String((0_u8..=0x1f).map(char::from).collect());
        assert_eq!(
            canonicalize(&value).unwrap(),
            serde_json::to_vec(&value).unwrap()
        );
    }

    #[test]
    fn object_keys_are_sorted_by_utf16_code_units() {
        let value = json!({"\u{e000}": 1, "\u{10000}": 2});
        assert_eq!(
            canonicalize(&value).unwrap(),
            "{\"\u{10000}\":2,\"\u{e000}\":1}".as_bytes()
        );
    }

    #[test]
    fn sha256_uses_the_content_address_prefix() {
        assert_eq!(
            sha256(b"abc"),
            "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
