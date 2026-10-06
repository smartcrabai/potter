use serde_json::json;

use crate::{cli::ScenePath, error::Result, response::SceneInfo, store::Project};

pub fn run(args: ScenePath) -> Result<(Option<SceneInfo>, serde_json::Value)> {
    let project = Project::init(args.scene)?;
    let scene = project.info()?;
    Ok((
        Some(scene.clone()),
        json!({ "created": true, "scene_file": format!("{}/scene.json", scene.path) }),
    ))
}
