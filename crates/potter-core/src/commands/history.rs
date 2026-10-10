use serde_json::{Value, json};

use crate::{cli::ScenePath, error::Result, response::SceneInfo, store::Project};

pub fn run(args: ScenePath) -> Result<(Option<SceneInfo>, Value)> {
    let project = Project::open(args.scene)?;
    let records = project
        .history_records()?
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
    Ok((Some(project.info()?), json!({ "entries": records })))
}
