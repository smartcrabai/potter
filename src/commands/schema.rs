use serde_json::{Value, json};

use crate::{catalog, cli::SchemaArgs, error::Result, response::SceneInfo, schema};

pub fn run(args: &SchemaArgs) -> Result<(Option<SceneInfo>, Value)> {
    let kind = match args.kind {
        crate::cli::SchemaKind::Scene => "scene",
        crate::cli::SchemaKind::Operations => "operations",
        crate::cli::SchemaKind::Preview => "preview",
        crate::cli::SchemaKind::Response => "response",
        crate::cli::SchemaKind::Capabilities => "capabilities",
        crate::cli::SchemaKind::Formats => "formats",
    };
    let schema = schema::schema(kind, args.op.as_deref())?;
    let mut result = json!({"kind":kind,"op":args.op,"schema":schema});
    match kind {
        "capabilities" => result["catalog"] = catalog::feature_catalog(),
        "formats" => result["catalog"] = catalog::formats_catalog(),
        _ => {}
    }
    Ok((None, result))
}
