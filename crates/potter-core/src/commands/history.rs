use serde_json::{Value, json};

use crate::{cli::ScenePath, error::Result, response::SceneInfo, store::Project};

pub fn run(args: ScenePath) -> Result<(Option<SceneInfo>, Value)> {
    let project = Project::open(args.scene)?;
    let (history, history_warnings) = project.history_records()?;
    let records = history
        .into_iter()
        .map(|(id, record)| {
            json!({
                "id": id,
                "revision": record.revision,
                "kind": record.kind,
                "scene_hash": record.scene_hash,
                "operations": record.operations,
                "changes": record.changes,
                "parent": record.parent,
                "target": record.target,
            })
        })
        .collect::<Vec<_>>();
    let mut warnings = project.warnings().to_vec();
    warnings.extend(history_warnings);
    Ok((
        Some(project.info()?),
        json!({ "entries": records, "warnings": warnings }),
    ))
}
