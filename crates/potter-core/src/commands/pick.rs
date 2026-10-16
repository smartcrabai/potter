use serde_json::Value;

use crate::{cli::PickArgs, error::Result, render::pick, response::SceneInfo, store::Project};

use super::util::with_load_warnings;

pub fn run(args: PickArgs) -> Result<(Option<SceneInfo>, Value)> {
    let project = Project::open(args.scene)?;
    let mut result = pick(
        project.doc(),
        project.path(),
        &args.render,
        args.pixel,
        args.domain,
    )?;
    result["warnings"] = with_load_warnings(&[], &project)?;
    Ok((Some(project.info()?), result))
}
