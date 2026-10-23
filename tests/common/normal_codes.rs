//! Stored custom-normal codes. Auto is the code `[0, 0]`, which is how Potter writes it and how Blender
//! stores an automatic loop. `null` is the legacy spelling of Auto that readers still accept.

use serde_json::{Value, json};

/// Whether a stored corner code is Auto (the corner takes its evaluated fan).
pub fn is_auto_code(value: &Value) -> bool {
    value.is_null() || *value == json!([0, 0])
}
