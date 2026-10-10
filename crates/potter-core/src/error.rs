use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub type Result<T, E = PotError> = std::result::Result<T, E>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ErrorCode {
    InternalError,
    InvalidArgument,
    InvalidOperation,
    UnsupportedVersion,
    BlenderVersionUnsupported,
    LimitExceeded,
    SceneNotFound,
    FileNotFound,
    TargetNotFound,
    BlenderNotFound,
    DependencyMissing,
    SceneInvalid,
    EvaluationFailed,
    RenderInvalid,
    ValidationFailed,
    UnsupportedFeature,
    UnrepresentableFeature,
    RevisionConflict,
    SceneBusy,
    AmbiguousTarget,
    SharedDataRequiresScope,
    IdExists,
    StaleRender,
    OutputExists,
    IoError,
    ImportFailed,
    ExportFailed,
    RenderFailed,
    BakeFailed,
    AssetChanged,
}

impl ErrorCode {
    #[must_use]
    pub const fn exit_code(self) -> i32 {
        match self {
            Self::InternalError => 1,
            Self::InvalidArgument
            | Self::InvalidOperation
            | Self::UnsupportedVersion
            | Self::BlenderVersionUnsupported
            | Self::LimitExceeded => 2,
            Self::SceneNotFound
            | Self::FileNotFound
            | Self::TargetNotFound
            | Self::BlenderNotFound
            | Self::DependencyMissing => 3,
            Self::SceneInvalid
            | Self::EvaluationFailed
            | Self::RenderInvalid
            | Self::ValidationFailed
            | Self::UnsupportedFeature
            | Self::UnrepresentableFeature
            | Self::AssetChanged => 4,
            Self::RevisionConflict
            | Self::SceneBusy
            | Self::AmbiguousTarget
            | Self::SharedDataRequiresScope
            | Self::IdExists
            | Self::StaleRender
            | Self::OutputExists => 5,
            Self::IoError
            | Self::ImportFailed
            | Self::ExportFailed
            | Self::RenderFailed
            | Self::BakeFailed => 6,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PotError {
    pub code: ErrorCode,
    pub message: String,
    #[serde(default = "empty_details")]
    pub details: Value,
}

fn empty_details() -> Value {
    json!({})
}

impl PotError {
    #[must_use]
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            details: empty_details(),
        }
    }

    #[must_use]
    pub fn with_details(code: ErrorCode, message: impl Into<String>, details: Value) -> Self {
        Self {
            code,
            message: message.into(),
            details,
        }
    }

    #[must_use]
    pub fn invalid_argument(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::InvalidArgument, message)
    }

    #[must_use]
    pub fn internal_json(error: impl std::fmt::Display) -> Self {
        Self::new(ErrorCode::InternalError, error.to_string())
    }
    #[must_use]
    pub fn invalid_operation(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::InvalidOperation, message)
    }

    #[must_use]
    pub fn io(error: &std::io::Error) -> Self {
        Self::with_details(
            ErrorCode::IoError,
            error.to_string(),
            json!({ "kind": format!("{:?}", error.kind()) }),
        )
    }
}

impl std::fmt::Display for PotError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{}: {}",
            serde_json::to_value(self.code).map_or_else(
                |_| "INTERNAL_ERROR".to_owned(),
                |value| value.as_str().unwrap_or("INTERNAL_ERROR").to_owned()
            ),
            self.message
        )
    }
}

impl std::error::Error for PotError {}

impl From<std::io::Error> for PotError {
    fn from(error: std::io::Error) -> Self {
        Self::io(&error)
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests")]

    use serde_json::json;

    use super::{ErrorCode, PotError};

    #[test]
    fn exit_codes_follow_the_cli_contract_for_every_error_code() {
        let cases = [
            (ErrorCode::InternalError, 1),
            (ErrorCode::InvalidArgument, 2),
            (ErrorCode::InvalidOperation, 2),
            (ErrorCode::UnsupportedVersion, 2),
            (ErrorCode::BlenderVersionUnsupported, 2),
            (ErrorCode::LimitExceeded, 2),
            (ErrorCode::SceneNotFound, 3),
            (ErrorCode::FileNotFound, 3),
            (ErrorCode::TargetNotFound, 3),
            (ErrorCode::BlenderNotFound, 3),
            (ErrorCode::DependencyMissing, 3),
            (ErrorCode::SceneInvalid, 4),
            (ErrorCode::EvaluationFailed, 4),
            (ErrorCode::RenderInvalid, 4),
            (ErrorCode::ValidationFailed, 4),
            (ErrorCode::UnsupportedFeature, 4),
            (ErrorCode::UnrepresentableFeature, 4),
            (ErrorCode::AssetChanged, 4),
            (ErrorCode::RevisionConflict, 5),
            (ErrorCode::SceneBusy, 5),
            (ErrorCode::AmbiguousTarget, 5),
            (ErrorCode::SharedDataRequiresScope, 5),
            (ErrorCode::IdExists, 5),
            (ErrorCode::StaleRender, 5),
            (ErrorCode::OutputExists, 5),
            (ErrorCode::IoError, 6),
            (ErrorCode::ImportFailed, 6),
            (ErrorCode::ExportFailed, 6),
            (ErrorCode::RenderFailed, 6),
            (ErrorCode::BakeFailed, 6),
        ];
        for (error, expected) in cases {
            assert_eq!(error.exit_code(), expected, "{error:?}");
        }
    }

    #[test]
    fn error_constructors_keep_codes_details_and_display_format() {
        let error = PotError::new(ErrorCode::InvalidArgument, "bad input");
        assert_eq!(error.details, json!({}));
        assert_eq!(error.to_string(), "INVALID_ARGUMENT: bad input");

        let detailed = PotError::with_details(
            ErrorCode::InvalidOperation,
            "cannot edit",
            json!({"path":"/transform"}),
        );
        assert_eq!(detailed.details, json!({"path":"/transform"}));

        assert_eq!(
            PotError::invalid_argument("bad").code,
            ErrorCode::InvalidArgument
        );
        assert_eq!(
            PotError::invalid_operation("no").code,
            ErrorCode::InvalidOperation
        );

        let io_error = std::io::Error::from(std::io::ErrorKind::NotFound);
        let converted = PotError::from(std::io::Error::from(std::io::ErrorKind::NotFound));
        let borrowed = PotError::io(&io_error);
        for error in [converted, borrowed] {
            assert_eq!(error.code, ErrorCode::IoError);
            assert_eq!(error.details, json!({"kind":"NotFound"}));
        }
    }

    #[test]
    fn missing_serialized_details_default_to_an_empty_object() {
        let error: PotError = serde_json::from_value(json!({
            "code":"INVALID_ARGUMENT",
            "message":"bad input"
        }))
        .unwrap();
        assert_eq!(error.details, json!({}));
    }
}
