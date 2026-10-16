use serde_json::{Value, json};

use crate::{
    cli::ValidateArgs,
    commands::util::exchange_format_name,
    error::{ErrorCode, PotError, Result},
    eval::{EvaluationContext, EvaluationPurpose, Snapshot},
    model::AvatarVersion,
    response::SceneInfo,
    store::Project,
    validate,
};
pub fn run(args: ValidateArgs) -> Result<(Option<SceneInfo>, Value)> {
    let project = Project::open(args.scene)?;
    let format = args.format.map(exchange_format_name);
    let settings_error = match format {
        Some("vrm") => {
            crate::exchange::vrm::validate_settings(project.doc(), AvatarVersion::V1).err()
        }
        Some("vrm0") => {
            crate::exchange::vrm::validate_settings(project.doc(), AvatarVersion::V0).err()
        }
        _ => None,
    };
    let mut report = validate::validate(project.doc(), format)?;
    if let Some(format @ ("vrm" | "vrm0")) = format {
        let version = if format == "vrm" {
            AvatarVersion::V1
        } else {
            AvatarVersion::V0
        };
        if let Some(error) = settings_error {
            append_vrm_preflight_error(&mut report, format, &error)?;
        } else {
            let context = EvaluationContext {
                purpose: EvaluationPurpose::Render,
                ..EvaluationContext::default()
            };
            match Snapshot::evaluate_with_cache(project.doc(), &context, Some(project.path())) {
                Ok(snapshot) => {
                    if let Err(error) =
                        crate::exchange::vrm::validate_output(project.doc(), &snapshot, version)
                    {
                        append_vrm_preflight_error(&mut report, format, &error)?;
                    }
                }
                Err(error) => append_vrm_preflight_error(&mut report, format, &error)?,
            }
        }
    }
    let scene = project.info()?;
    if args.strict && (report.has_errors || report.has_warnings) {
        return Err(PotError::with_details(
            ErrorCode::ValidationFailed,
            "scene validation failed in strict mode",
            json!({"result":report.value,"strict":true}),
        ));
    }
    Ok((Some(scene), report.value))
}

fn append_vrm_preflight_error(
    report: &mut validate::ValidationReport,
    format: &str,
    error: &PotError,
) -> Result<()> {
    append_issue(
        report,
        "VRM_EXPORT_PREFLIGHT_FAILED",
        &error.message,
        json!({"format":format,"error_code":error.code,"details":error.details}),
    )
}

fn append_issue(
    report: &mut validate::ValidationReport,
    code: &str,
    message: &str,
    details: Value,
) -> Result<()> {
    let mut issue = json!({
        "severity":"error",
        "code":code,
        "message":message,
        "data_id":Value::Null,
        "pointer":Value::Null
    });
    issue
        .as_object_mut()
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "validation issue report entry is not an object",
            )
        })?
        .insert("details".to_owned(), details);
    report.value["issues"]
        .as_array_mut()
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "validation report issues are not an array",
            )
        })?
        .push(issue);
    report.has_errors = true;
    report.value["valid"] = Value::Bool(false);
    report.value["compatibility"]["status"] = Value::String("invalid".to_owned());
    if let Some(check) = report.value["checks"].as_array_mut().and_then(|checks| {
        checks
            .iter_mut()
            .find(|check| check["check"] == "format_loss")
    }) {
        check["status"] = Value::String("not_supported".to_owned());
    }
    let issue_count = report.value["issues"].as_array().map_or(0, Vec::len);
    let summary = &mut report.value["summary"];
    summary["issue_count"] = json!(issue_count);
    let error_count = summary["by_severity"]["error"].as_u64().unwrap_or(0) + 1;
    summary["by_severity"]["error"] = json!(error_count);
    Ok(())
}
