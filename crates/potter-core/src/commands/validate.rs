use serde_json::{Value, json};

use crate::{
    cli::ValidateArgs,
    error::{ErrorCode, PotError, Result},
    response::SceneInfo,
    store::Project,
    validate,
};

pub fn run(args: ValidateArgs) -> Result<(Option<SceneInfo>, Value)> {
    let project = Project::open(args.scene)?;
    let format = args.format.map(|value| match value {
        crate::cli::ExchangeFormat::Blend => "blend",
        crate::cli::ExchangeFormat::Glb => "glb",
        crate::cli::ExchangeFormat::Gltf => "gltf",
        crate::cli::ExchangeFormat::Usda => "usda",
        crate::cli::ExchangeFormat::Usdc => "usdc",
        crate::cli::ExchangeFormat::Usd => "usd",
        crate::cli::ExchangeFormat::Usdz => "usdz",
        crate::cli::ExchangeFormat::Alembic => "alembic",
        crate::cli::ExchangeFormat::Fbx | crate::cli::ExchangeFormat::FbxBinary => "fbx",
        crate::cli::ExchangeFormat::Obj => "obj",
        crate::cli::ExchangeFormat::Ply => "ply",
        crate::cli::ExchangeFormat::Stl => "stl",
        crate::cli::ExchangeFormat::Bvh => "bvh",
        crate::cli::ExchangeFormat::Svg => "svg",
        crate::cli::ExchangeFormat::Pdf => "pdf",
    });
    let report = validate::validate(project.doc(), format)?;
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
