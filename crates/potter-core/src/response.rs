use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::PotError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneInfo {
    pub id: String,
    pub path: String,
    pub revision: u64,
    pub hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Envelope {
    pub schema_version: u32,
    pub command: Option<String>,
    pub ok: bool,
    pub scene: Option<SceneInfo>,
    pub result: Value,
    pub warnings: Vec<Value>,
    pub error: Option<PotError>,
}

impl Envelope {
    #[must_use]
    pub fn success(command: impl Into<String>, scene: Option<SceneInfo>, result: Value) -> Self {
        let warnings = result
            .get("warnings")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        Self {
            schema_version: 1,
            command: Some(command.into()),
            ok: true,
            scene,
            result,
            warnings,
            error: None,
        }
    }
    #[must_use]
    pub fn failure(command: Option<String>, error: PotError) -> Self {
        let result = error.details.get("result").cloned().unwrap_or(Value::Null);
        Self {
            schema_version: 1,
            command,
            ok: false,
            scene: None,
            result,
            warnings: Vec::new(),
            error: Some(error),
        }
    }

    #[must_use]
    pub fn exit_code(&self) -> i32 {
        self.error
            .as_ref()
            .map_or(0, |error| error.code.exit_code())
    }

    #[must_use]
    pub fn emit(&self, json_output: bool) -> i32 {
        if json_output {
            println!("{}", serde_json::to_string(self).unwrap_or_else(|_| "{\"schema_version\":1,\"command\":null,\"ok\":false,\"scene\":null,\"result\":null,\"warnings\":[],\"error\":{\"code\":\"INTERNAL_ERROR\",\"message\":\"response serialization failed\",\"details\":{}}}".to_owned()));
        } else if let Some(error) = &self.error {
            eprintln!("{error}");
        } else {
            println!(
                "{}",
                serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_owned())
            );
        }
        self.exit_code()
    }
}

impl From<PotError> for Envelope {
    fn from(error: PotError) -> Self {
        Self::failure(None, error)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::error::ErrorCode;

    use super::{Envelope, PotError, SceneInfo};

    #[test]
    fn success_envelope_preserves_scene_result_and_success_exit_code() {
        let scene = SceneInfo {
            id: "scene-id".to_owned(),
            path: "/project".to_owned(),
            revision: 7,
            hash: "sha256:abc".to_owned(),
        };
        let response = Envelope::success("inspect", Some(scene), json!({"items":[1,2]}));
        assert_eq!(response.schema_version, 1);
        assert_eq!(response.command.as_deref(), Some("inspect"));
        assert!(response.ok);
        assert_eq!(response.scene.as_ref().map(|scene| scene.revision), Some(7));
        assert_eq!(response.result, json!({"items":[1,2]}));
        assert!(response.warnings.is_empty(), "{:?}", response.warnings);
        assert!(response.error.is_none());
        assert_eq!(response.exit_code(), 0);
    }

    #[test]
    fn failure_envelope_exposes_result_details_and_error_exit_code() {
        let error = PotError::with_details(
            ErrorCode::RevisionConflict,
            "revision changed",
            json!({"result":{"current_revision":9}}),
        );
        let response = Envelope::failure(Some("apply".to_owned()), error);
        assert_eq!(response.command.as_deref(), Some("apply"));
        assert!(!response.ok);
        assert!(response.scene.is_none());
        assert_eq!(response.result, json!({"current_revision":9}));
        assert!(response.warnings.is_empty(), "{:?}", response.warnings);
        assert_eq!(
            response.error.as_ref().map(|error| error.code),
            Some(ErrorCode::RevisionConflict)
        );
        assert_eq!(response.exit_code(), 5);

        let response = Envelope::from(PotError::invalid_argument("bad input"));
        assert_eq!(response.command, None);
        assert_eq!(response.result, serde_json::Value::Null);
        assert_eq!(response.exit_code(), 2);
    }

    #[test]
    fn emit_returns_the_envelope_exit_code_in_json_and_text_modes() {
        let success = Envelope::success("inspect", None, json!({}));
        assert_eq!(success.emit(true), 0);
        assert_eq!(success.emit(false), 0);

        let failure = Envelope::failure(None, PotError::new(ErrorCode::SceneBusy, "busy"));
        assert_eq!(failure.emit(true), 5);
        assert_eq!(failure.emit(false), 5);
    }
}
