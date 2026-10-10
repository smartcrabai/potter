use std::{collections::BTreeSet, path::Path};

use serde_json::{Map, Value, json};

use crate::{
    cli::{ExchangeFormat, UndoArgs},
    error::{ErrorCode, PotError, Result},
    eval::EvaluationContext,
    model::{Id, SceneDoc},
    response::SceneInfo,
    store::{HistoryDirection, Project},
};

pub(super) fn parse_id(value: &str) -> Result<Id> {
    Id::new(value).map_err(|error| {
        PotError::with_details(
            ErrorCode::InvalidArgument,
            error.message,
            json!({ "id": value }),
        )
    })
}

pub fn check_base_revision(doc: &SceneDoc, supplied: u64) -> Result<()> {
    if doc.revision != supplied {
        return Err(PotError::with_details(
            ErrorCode::RevisionConflict,
            "base_revision does not match current scene revision",
            json!({ "expected": doc.revision, "actual": supplied }),
        ));
    }
    Ok(())
}

pub fn scene_changes(before: &SceneDoc, after: &SceneDoc) -> Result<Value> {
    let before = serde_json::to_value(before)
        .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))?;
    let after = serde_json::to_value(after)
        .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))?;
    let registries = [
        "scenes",
        "collections",
        "nodes",
        "data_blocks",
        "materials",
        "worlds",
        "node_groups",
        "actions",
        "resources",
        "libraries",
        "compatibility",
    ];
    let mut changes = Map::new();
    for registry in registries {
        changes.insert(
            registry.to_owned(),
            registry_changes(&before[registry], &after[registry]),
        );
    }
    Ok(Value::Object(changes))
}

fn registry_changes(before: &Value, after: &Value) -> Value {
    let before_map = before.as_object();
    let after_map = after.as_object();
    let ids = before_map
        .into_iter()
        .flat_map(serde_json::Map::keys)
        .chain(after_map.into_iter().flat_map(serde_json::Map::keys))
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut created = Vec::new();
    let mut updated = Vec::new();
    let mut deleted = Vec::new();
    for id in ids {
        match (
            before_map.and_then(|map| map.get(&id)),
            after_map.and_then(|map| map.get(&id)),
        ) {
            (None, Some(_)) => created.push(id),
            (Some(_), None) => deleted.push(id),
            (Some(old), Some(new)) if old != new => updated.push(id),
            _ => {}
        }
    }
    json!({ "created": created, "updated": updated, "deleted": deleted })
}

pub(super) fn evaluation_context(
    scene_id: Option<&str>,
    view_layer: Option<&str>,
    frame: Option<f64>,
) -> Result<EvaluationContext> {
    if frame.is_some_and(|value| !value.is_finite()) {
        return Err(PotError::invalid_argument("frame must be finite"));
    }
    Ok(EvaluationContext {
        scene_id: scene_id
            .map(|value| Id::new(value.to_owned()))
            .transpose()?,
        view_layer: view_layer
            .map(|value| Id::new(value.to_owned()))
            .transpose()?,
        frame,
    })
}

pub(super) fn exchange_format_name(format: ExchangeFormat) -> &'static str {
    match format {
        ExchangeFormat::Blend => "blend",
        ExchangeFormat::Glb => "glb",
        ExchangeFormat::Gltf => "gltf",
        ExchangeFormat::Usda => "usda",
        ExchangeFormat::Usdc => "usdc",
        ExchangeFormat::Usd => "usd",
        ExchangeFormat::Usdz => "usdz",
        ExchangeFormat::Alembic => "alembic",
        ExchangeFormat::Fbx => "fbx",
        ExchangeFormat::FbxBinary => "fbx-binary",
        ExchangeFormat::Obj => "obj",
        ExchangeFormat::Ply => "ply",
        ExchangeFormat::Stl => "stl",
        ExchangeFormat::Bvh => "bvh",
        ExchangeFormat::Svg => "svg",
        ExchangeFormat::Pdf => "pdf",
    }
}

pub(super) fn check_extension(
    path: &Path,
    format: &str,
    error_message: &'static str,
) -> Result<()> {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    let extension_matches = match format {
        "alembic" => extension.eq_ignore_ascii_case("abc"),
        "fbx-binary" => extension.eq_ignore_ascii_case("fbx"),
        "usdc" => extension.eq_ignore_ascii_case("usdc") || extension.eq_ignore_ascii_case("usd"),
        _ => extension.eq_ignore_ascii_case(format),
    };
    if extension_matches {
        return Ok(());
    }
    Err(PotError::with_details(
        ErrorCode::InvalidArgument,
        error_message,
        json!({ "format": format, "extension": extension }),
    ))
}

pub(super) fn run_history(
    args: UndoArgs,
    direction: HistoryDirection,
) -> Result<(Option<SceneInfo>, Value)> {
    let mut project = Project::open_exclusive(args.scene)?;
    check_base_revision(project.doc(), args.base_revision)?;
    if args.steps == 0 {
        return Err(PotError::invalid_argument(
            "steps must be a positive integer",
        ));
    }
    let base_revision = project.doc().revision;
    let before = project.doc().clone();
    let (candidate, target) = project.history_target(direction, args.steps)?;
    let changes = scene_changes(&before, &candidate)?;
    let scene = project.commit_history(
        direction,
        candidate,
        target.clone(),
        args.steps,
        changes.clone(),
    )?;
    Ok((
        Some(scene.clone()),
        json!({
            "committed": true,
            "changed": true,
            "base_revision": base_revision,
            "candidate_revision": scene.revision,
            "changes": changes,
            "operations": [],
            "id_mappings": {},
            "previews": [],
            "history_id": target,
        }),
    ))
}
