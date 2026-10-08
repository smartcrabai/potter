use std::{fmt, str::FromStr};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::{ErrorCode, PotError, Result};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Id(String);

impl Id {
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        if !is_valid(&value) {
            return Err(PotError::new(
                ErrorCode::InvalidArgument,
                "ID must match ^[a-z][a-z0-9_-]{0,63}$",
            ));
        }
        Ok(Self(value))
    }

    pub(crate) fn from_static(value: &str) -> Self {
        debug_assert!(is_valid(value));
        Self(value.to_owned())
    }
    #[must_use]
    pub fn is_valid(value: &str) -> bool {
        is_valid(value)
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

const fn is_lowercase_ascii(byte: u8) -> bool {
    byte >= b'a' && byte <= b'z'
}

const fn is_id_tail(byte: u8) -> bool {
    is_lowercase_ascii(byte) || (byte >= b'0' && byte <= b'9') || byte == b'_' || byte == b'-'
}

#[must_use]
pub fn is_valid(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.is_empty() || bytes.len() > 64 || !is_lowercase_ascii(bytes[0]) {
        return false;
    }
    let mut index = 1;
    while index < bytes.len() {
        if !is_id_tail(bytes[index]) {
            return false;
        }
        index += 1;
    }
    true
}

impl fmt::Display for Id {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl FromStr for Id {
    type Err = PotError;

    fn from_str(value: &str) -> Result<Self> {
        Self::new(value)
    }
}

impl Serialize for Id {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for Id {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests")]
    use super::{Id, is_valid};
    use proptest::prelude::*;

    #[test]
    fn accepts_contract_ids_and_rejects_invalid_boundaries() {
        assert!(Id::new("a").is_ok());
        assert!(Id::new("body_2-x").is_ok());
        assert!(Id::new("body_2-x").is_ok_and(|id| id.as_str() == "body_2-x"));
        assert!(Id::new("a".repeat(64)).is_ok());
        assert!(Id::is_valid("a"));
        assert!(Id::is_valid("body_2-x"));
        assert!(!Id::is_valid("A"));
        assert!(Id::new("").is_err());
        assert!(Id::new("A").is_err());
        assert!(Id::new("2a").is_err());
        assert!(Id::new("a".repeat(65)).is_err());
        assert!(Id::new("aé").is_err());
    }

    #[test]
    fn display_parse_and_serde_preserve_the_id_value() {
        let id = Id::new("body_2-x").unwrap();
        assert_eq!(id.as_str(), "body_2-x");
        assert_eq!(id.to_string(), "body_2-x");
        assert!("body_2-x".parse::<Id>().is_ok_and(|parsed| parsed == id));
        assert!("Body_2-x".parse::<Id>().is_err());

        let serialized = serde_json::to_string(&id).unwrap();
        assert_eq!(serialized, "\"body_2-x\"");
        assert_eq!(serde_json::from_str::<Id>(&serialized).unwrap(), id);
        assert!(serde_json::from_str::<Id>("\"Body_2-x\"").is_err());
    }

    proptest! {
        #[test]
        fn id_validity_matches_generated_ascii_contract(s in ".{0,70}") {
            let expected = s.len() <= 64
                && s.as_bytes().first().is_some_and(u8::is_ascii_lowercase)
                && s.bytes().skip(1).all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-');
            prop_assert_eq!(is_valid(&s), expected);
            prop_assert_eq!(Id::is_valid(&s), expected);
        }
    }
}
