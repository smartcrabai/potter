use serde_json::Value;

use crate::{cli::PickArgs, error::Result, render::pick, response::SceneInfo, store::Project};

pub fn run(args: PickArgs) -> Result<(Option<SceneInfo>, Value)> {
    let project = Project::open(args.scene)?;
    let result = pick(
        project.doc(),
        project.path(),
        &args.render,
        args.pixel,
        args.domain,
    )?;
    Ok((Some(project.info()?), result))
}
