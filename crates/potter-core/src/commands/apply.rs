use std::{io::Read, path::Path};

use serde_json::{Value, json};

use crate::{
    cli::ApplyArgs,
    commands::util::check_base_revision,
    error::{ErrorCode, PotError, Result},
    eval::EvaluationContext,
    model::Id,
    ops,
    render::{PreviewRequest, render_previews},
    response::SceneInfo,
    store::Project,
};

pub fn run(args: ApplyArgs) -> Result<(Option<SceneInfo>, Value)> {
    if args.size.is_some() && args.preview.is_none() {
        return Err(PotError::invalid_argument("--size requires --preview"));
    }
    if args.preview.is_some() && args.dry_run {
        return Err(PotError::invalid_argument(
            "--dry-run cannot be combined with --preview",
        ));
    }
    let input = read_operations_file(&args.file)?;
    let batch: Value = serde_json::from_slice(&input).map_err(|error| {
        PotError::with_details(
            ErrorCode::InvalidOperation,
            "operation input is not valid JSON",
            json!({ "line": error.line(), "column": error.column() }),
        )
    })?;
    let base_revision = batch
        .get("base_revision")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::InvalidOperation,
                "base_revision is required and must be a non-negative integer",
                json!({ "pointer": "/base_revision" }),
            )
        })?;
    let mut project = Project::open_exclusive(args.scene)?;
    check_base_revision(project.doc(), base_revision)?;
    let before = project.doc().clone();
    let outcome = ops::apply_batch_with_asset_root(&before, &batch, project.path())?;
    let candidate_revision = if outcome.changed {
        base_revision
            .checked_add(1)
            .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, "revision limit reached"))?
    } else {
        base_revision
    };
    let previews = if let Some(views_csv) = args.preview.as_deref() {
        let candidate = if outcome.changed {
            project.prepare_commit_candidate(
                outcome.doc.clone(),
                outcome.operations.clone(),
                outcome.changes.clone(),
            )?
        } else {
            outcome.doc.clone()
        };
        render_candidate_preview(
            &candidate,
            project.path(),
            &batch,
            &outcome.asset_blobs,
            views_csv,
            args.size.unwrap_or(768),
        )?
    } else {
        Vec::new()
    };
    if outcome.changed && !args.dry_run {
        let scene = project.commit_with_assets(
            outcome.doc,
            outcome.operations.clone(),
            outcome.changes.clone(),
            &outcome.asset_blobs,
        )?;
        let result = apply_result(
            true,
            true,
            base_revision,
            scene.revision,
            &outcome.changes,
            &outcome.operations,
            &outcome.id_mappings,
            &previews,
        );
        return Ok((Some(scene), result));
    }
    let scene = project.info()?;
    let result = apply_result(
        false,
        outcome.changed,
        base_revision,
        candidate_revision,
        &outcome.changes,
        &outcome.operations,
        &outcome.id_mappings,
        &previews,
    );
    Ok((Some(scene), result))
}

fn render_candidate_preview(
    candidate: &crate::model::SceneDoc,
    scene_root: &Path,
    batch: &Value,
    staged_assets: &std::collections::BTreeMap<String, Vec<u8>>,
    views_csv: &str,
    size: u32,
) -> Result<Vec<Value>> {
    let views = views_csv.split(',').map(str::to_owned).collect::<Vec<_>>();
    let evaluation = batch.get("evaluation");
    if evaluation.is_some_and(|value| !value.is_object()) {
        return Err(PotError::with_details(
            ErrorCode::InvalidOperation,
            "evaluation must be an object",
            json!({ "pointer": "/evaluation" }),
        ));
    }
    let evaluation = evaluation.and_then(Value::as_object);
    let scene_id = optional_id(evaluation, "scene_id")?;
    let view_layer = optional_id(evaluation, "view_layer")?;
    let frame = evaluation
        .and_then(|value| value.get("frame"))
        .map(|value| {
            value
                .as_f64()
                .filter(|number| number.is_finite())
                .ok_or_else(|| {
                    PotError::with_details(
                        ErrorCode::InvalidOperation,
                        "evaluation.frame must be a finite number",
                        json!({ "pointer": "/evaluation/frame" }),
                    )
                })
        })
        .transpose()?;
    let camera_id = optional_string(evaluation, "camera")?;
    if camera_id.is_some() && !views.is_empty() {
        return Err(PotError::invalid_argument(
            "evaluation.camera cannot be combined with preset --preview views",
        ));
    }
    let mode = optional_string(evaluation, "mode")?.unwrap_or_else(|| "solid".to_owned());
    let context = EvaluationContext {
        scene_id,
        view_layer,
        frame,
    };
    render_previews(
        candidate,
        scene_root,
        &PreviewRequest {
            views: &views,
            camera_id: camera_id.as_deref(),
            mode: &mode,
            size,
            out: None,
            overwrite: false,
            context: &context,
            render_visibility: false,
            staged_assets: Some(staged_assets),
        },
    )
}

fn optional_id(
    evaluation: Option<&serde_json::Map<String, Value>>,
    key: &str,
) -> Result<Option<Id>> {
    optional_string(evaluation, key)?
        .map(Id::new)
        .transpose()
        .map_err(|error| {
            PotError::with_details(
                ErrorCode::InvalidOperation,
                error.message,
                json!({ "pointer": format!("/evaluation/{key}") }),
            )
        })
}

fn optional_string(
    evaluation: Option<&serde_json::Map<String, Value>>,
    key: &str,
) -> Result<Option<String>> {
    evaluation
        .and_then(|value| value.get(key))
        .map(|value| {
            value.as_str().map(str::to_owned).ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::InvalidOperation,
                    format!("evaluation.{key} must be a string"),
                    json!({ "pointer": format!("/evaluation/{key}") }),
                )
            })
        })
        .transpose()
}

fn read_operations_file(path: &Path) -> Result<Vec<u8>> {
    if path == Path::new("-") {
        let mut input = Vec::new();
        std::io::stdin()
            .read_to_end(&mut input)
            .map_err(|error| PotError::io(&error))?;
        Ok(input)
    } else {
        std::fs::read(path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                PotError::with_details(
                    ErrorCode::FileNotFound,
                    format!("operation file not found: {}", path.display()),
                    json!({ "path": path }),
                )
            } else {
                PotError::io(&error)
            }
        })
    }
}

fn apply_result(
    committed: bool,
    changed: bool,
    base_revision: u64,
    candidate_revision: u64,
    diff: &Value,
    operations: &Value,
    id_mappings: &Value,
    previews: &[Value],
) -> Value {
    json!({
        "committed": committed,
        "changed": changed,
        "base_revision": base_revision,
        "candidate_revision": candidate_revision,
        "changes": diff,
        "operations": operations,
        "id_mappings": id_mappings,
        "previews": previews,
    })
}
