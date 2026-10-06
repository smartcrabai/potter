use serde_json::{Value, json};

use crate::{
    cli::{PreviewArgs, PreviewMode},
    commands::util::evaluation_context,
    error::Result,
    render::{PreviewRequest, render_previews},
    response::SceneInfo,
    store::Project,
};

pub fn run(args: PreviewArgs) -> Result<(Option<SceneInfo>, Value)> {
    let project = Project::open(args.scene)?;
    let context = evaluation_context(
        args.context.scene_id.as_deref(),
        args.context.view_layer.as_deref(),
        args.context.frame,
    )?;
    let views = args
        .views
        .as_deref()
        .map(|value| value.split(',').map(str::to_owned).collect::<Vec<_>>())
        .unwrap_or_default();
    let mode = match args.mode {
        PreviewMode::Solid => "solid",
        PreviewMode::Beauty => "beauty",
        PreviewMode::Wire => "wire",
        PreviewMode::Normal => "normal",
        PreviewMode::Depth => "depth",
        PreviewMode::Id => "id",
    };
    let previews = render_previews(
        project.doc(),
        project.path(),
        &PreviewRequest {
            views: &views,
            camera_id: args.camera.as_deref(),
            mode,
            size: args.size,
            out: args.out.as_deref(),
            overwrite: args.overwrite,
            context: &context,
            render_visibility: false,
            staged_assets: None,
        },
    )?;
    let mut warnings = Vec::new();
    for warning in previews
        .iter()
        .filter_map(|preview| preview["warnings"].as_array())
        .flatten()
    {
        if !warnings.contains(warning) {
            warnings.push(warning.clone());
        }
    }
    Ok((
        Some(project.info()?),
        json!({"previews":previews,"warnings":warnings}),
    ))
}
