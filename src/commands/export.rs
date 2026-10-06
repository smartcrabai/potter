use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

use serde_json::{Value, json};

use crate::{
    cli::ExportArgs,
    commands::util::{check_extension, exchange_format_name},
    error::{ErrorCode, PotError, Result},
    eval::{EvaluationContext, Snapshot},
    exchange::{self, ExportOptions, Loss},
    hash,
    model::{Id, LightType, SceneDoc},
    response::SceneInfo,
    store::Project,
};

pub fn run(args: &ExportArgs) -> Result<(Option<SceneInfo>, Value)> {
    let project = Project::open(&args.scene)?;
    let doc = project.doc().clone();
    let context = evaluation_context(&args.context)?;
    let format = exchange_format_name(args.format);
    check_extension(&args.out, format, "output extension does not match format")?;
    let view = export_view(args.view.as_deref(), format)?;
    if !matches!(
        format,
        "blend"
            | "glb"
            | "gltf"
            | "usda"
            | "usdc"
            | "usd"
            | "usdz"
            | "alembic"
            | "fbx"
            | "fbx-binary"
            | "bvh"
            | "obj"
            | "ply"
            | "stl"
            | "pdf"
            | "svg"
    ) {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            format!("{format} export is not implemented"),
            json!({ "feature_id": format!("format.{format}"), "status": "not_supported" }),
        ));
    }
    let options = ExportOptions {
        allow_lossy: args.allow_lossy,
        pack: args.pack,
        blender: args.blender.as_deref(),
        context: &context,
    };
    let result = if format == "blend" {
        let parent = args.out.parent().unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent).map_err(|error| PotError::io(&error))?;
        if args.out.exists() && !args.overwrite {
            return Err(PotError::new(
                ErrorCode::OutputExists,
                "output file already exists",
            ));
        }
        let report = exchange::blend::export_blend(&doc, &project, &args.out, &options)?;
        if !report.losses.is_empty() && !args.allow_lossy {
            return Err(lossy_error(format, &report.losses));
        }
        let primary = report
            .files
            .iter()
            .find(|path| *path == &args.out)
            .unwrap_or(&args.out);
        let bytes = fs::read(primary).map_err(|error| PotError::io(&error))?;
        json!({
            "format": format,
            "path": args.out,
            "files": report.files,
            "hash": hash::sha256(&bytes),
            "bytes": bytes.len(),
            "counts": report.counts,
            "context": blend_export_context_json(&doc, &context)?,
            "dependencies": report.files.iter().filter(|path| *path != &args.out).collect::<Vec<_>>(),
            "compatibility": doc.compatibility,
            "conversions": report.conversions,
            "losses": report.losses,
        })
    } else {
        let snapshot = Snapshot::evaluate_with_cache(&doc, &context, Some(project.path()))?;
        let losses = export_losses(&doc, format);
        if !losses.is_empty() && !args.allow_lossy {
            return Err(lossy_error(format, &losses));
        }
        let meshes = exchange::evaluated_meshes(&doc, &snapshot)?;
        let artifacts = match format {
            "glb" => {
                let (json_bytes, binary) =
                    exchange::gltf::export_with_root(&doc, &snapshot, project.path(), None)?;
                vec![(
                    args.out.clone(),
                    exchange::gltf::to_glb(&json_bytes, &binary)?,
                )]
            }
            "gltf" => {
                let sidecar = sidecar_path(&args.out, "bin")?;
                let uri = sidecar
                    .file_name()
                    .and_then(|value| value.to_str())
                    .ok_or_else(|| {
                        PotError::new(ErrorCode::InvalidArgument, "sidecar filename is not UTF-8")
                    })?;
                let (json_bytes, binary) =
                    exchange::gltf::export_with_root(&doc, &snapshot, project.path(), Some(uri))?;
                vec![(sidecar, binary), (args.out.clone(), json_bytes)]
            }
            "usda" => vec![(
                args.out.clone(),
                exchange::usd::export_usda(&doc, &snapshot)?,
            )],
            "usdc" | "usd" => vec![(
                args.out.clone(),
                exchange::usdc::export(&doc, &snapshot, options.blender)?,
            )],
            "usdz" => {
                let root_name = args
                    .out
                    .file_stem()
                    .and_then(|value| value.to_str())
                    .ok_or_else(|| {
                        PotError::new(ErrorCode::InvalidArgument, "USDZ output name is invalid")
                    })?;
                vec![(
                    args.out.clone(),
                    exchange::usd::export_usdz(&doc, &snapshot, root_name)?,
                )]
            }
            "alembic" => vec![(
                args.out.clone(),
                exchange::alembic::export(&doc, &snapshot, options.allow_lossy, options.blender)?,
            )],
            "bvh" => vec![(args.out.clone(), exchange::bvh::export(&doc, &snapshot)?)],
            "fbx-binary" => vec![(
                args.out.clone(),
                exchange::fbx_binary::export(&doc, &snapshot)?,
            )],
            "fbx" => vec![(args.out.clone(), exchange::fbx::export(&doc, &snapshot)?)],
            "svg" => vec![(
                args.out.clone(),
                exchange::svg::export(&doc, &snapshot, view)?,
            )],
            "pdf" => {
                let pdf_view = if view == "iso" { "isometric" } else { view };
                vec![(
                    args.out.clone(),
                    exchange::pdf::export(&doc, &snapshot, pdf_view)?,
                )]
            }
            "obj" => {
                let sidecar = sidecar_path(&args.out, "mtl")?;
                let (object, material) = exchange::obj::export(&meshes)?;
                let mtl_name = sidecar
                    .file_name()
                    .and_then(|value| value.to_str())
                    .ok_or_else(|| {
                        PotError::new(ErrorCode::InvalidArgument, "material filename is not UTF-8")
                    })?;
                let object = String::from_utf8(object)
                    .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))?
                    .replace("scene.mtl", mtl_name)
                    .into_bytes();
                vec![(sidecar, material), (args.out.clone(), object)]
            }
            "stl" => vec![(args.out.clone(), exchange::stl::export(&meshes)?)],
            "ply" => vec![(args.out.clone(), exchange::ply::export(&meshes)?)],
            _ => {
                return Err(PotError::new(
                    ErrorCode::InternalError,
                    "unreachable export format",
                ));
            }
        };
        let primary_bytes = artifacts
            .iter()
            .find(|(path, _)| path == &args.out)
            .map(|(_, bytes)| bytes.as_slice())
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::InternalError,
                    "primary export artifact is missing",
                )
            })?;
        let primary_hash = hash::sha256(primary_bytes);
        let total_bytes = artifacts
            .iter()
            .try_fold(0_u64, |total, (_, bytes)| {
                total.checked_add(u64::try_from(bytes.len()).ok()?)
            })
            .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, "export byte count overflow"))?;
        publish(&artifacts, args.overwrite)?;
        let files = artifacts
            .iter()
            .map(|(path, _)| path.clone())
            .collect::<Vec<_>>();
        let dependencies = files
            .iter()
            .filter(|path| *path != &args.out)
            .collect::<Vec<_>>();
        let conversions = export_conversions(&doc, &meshes, format);
        json!({
            "format": format,
            "path": args.out,
            "files": files,
            "hash": primary_hash,
            "bytes": total_bytes,
            "counts": geometry_counts(&meshes),
            "context": context_json(&context, &snapshot),
            "dependencies": dependencies,
            "compatibility": doc.compatibility,
            "conversions": conversions,
            "losses": losses,
        })
    };
    Ok((Some(project.info()?), result))
}

fn export_view<'a>(requested: Option<&'a str>, format: &str) -> Result<&'a str> {
    if requested.is_some() && !matches!(format, "svg" | "pdf") {
        return Err(PotError::with_details(
            ErrorCode::InvalidArgument,
            "--view is only valid for SVG or PDF export",
            json!({ "format": format, "view": requested }),
        ));
    }
    let view = requested.unwrap_or("front");
    if !matches!(
        view,
        "front" | "back" | "right" | "left" | "top" | "bottom" | "iso"
    ) {
        return Err(PotError::with_details(
            ErrorCode::InvalidArgument,
            "unknown export view projection",
            json!({ "view": view }),
        ));
    }
    Ok(view)
}

fn evaluation_context(context: &crate::cli::Context) -> Result<EvaluationContext> {
    Ok(EvaluationContext {
        scene_id: context.scene_id.as_deref().map(Id::new).transpose()?,
        view_layer: context.view_layer.as_deref().map(Id::new).transpose()?,
        frame: context.frame,
    })
}

fn context_json(context: &EvaluationContext, snapshot: &Snapshot) -> Value {
    json!({
        "scene_id": snapshot.scene_id,
        "view_layer": snapshot.view_layer,
        "frame": snapshot.frame,
        "scene_hash": snapshot.scene_hash,
        "evaluation_hash": snapshot.evaluation_hash,
        "requested": context,
    })
}
fn blend_export_context_json(doc: &SceneDoc, context: &EvaluationContext) -> Result<Value> {
    let scene_id = context
        .scene_id
        .clone()
        .unwrap_or_else(|| doc.active_scene.clone());
    let scene = doc.scenes.get(&scene_id).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::TargetNotFound,
            "scene context not found",
            json!({"scene_id":scene_id}),
        )
    })?;
    let view_layer = match &context.view_layer {
        Some(view_layer) if scene.view_layers.contains_key(view_layer) => Some(view_layer.clone()),
        Some(view_layer) => {
            return Err(PotError::with_details(
                ErrorCode::TargetNotFound,
                "view layer context not found",
                json!({"view_layer":view_layer}),
            ));
        }
        None => scene.view_layers.keys().next().cloned(),
    };
    let frame = context.frame.unwrap_or(scene.frame_current);
    if !frame.is_finite() {
        return Err(PotError::new(
            ErrorCode::InvalidArgument,
            "frame must be finite",
        ));
    }
    let scene_value = serde_json::to_value(doc).map_err(PotError::internal_json)?;
    let scene_hash = hash::sha256(&hash::canonicalize(&scene_value)?);
    Ok(json!({
        "scene_id": scene_id,
        "view_layer": view_layer,
        "frame": frame,
        "scene_hash": scene_hash,
        "evaluation_hash": null,
        "requested": context,
    }))
}

pub(crate) fn export_losses(doc: &SceneDoc, format: &str) -> Vec<Loss> {
    let supports_scene = matches!(
        format,
        "glb" | "gltf" | "usda" | "usdc" | "usd" | "usdz" | "fbx" | "fbx-binary" | "alembic"
    );
    let supports_bvh = format == "bvh";
    let supports_gltf_skin = matches!(format, "glb" | "gltf");
    let supports_cameras = matches!(format, "glb" | "gltf" | "usda" | "usdc" | "usd" | "usdz");
    let supports_lights = supports_cameras;
    let supports_strokes = matches!(format, "svg" | "pdf");
    let mut losses = Vec::new();
    for (id, node) in &doc.nodes {
        if node.action.is_some()
            && (!(supports_scene || supports_bvh && node.kind == "armature")
                || node.parent_inverse.is_some())
        {
            losses.push(Loss {
                feature_id: "animation.action".to_owned(),
                data_id: Some(id.to_string()),
                reason: format!("{format} cannot preserve actions with this transform"),
                suggestion: Some("export as blend or remove the parent inverse".to_owned()),
            });
        }
        let kind_supported = node.kind == "mesh"
            || (supports_strokes && node.kind == "grease_pencil")
            || (supports_bvh && node.kind == "armature");
        if !supports_scene && (!kind_supported || node.parent.is_some()) {
            losses.push(Loss {
                feature_id: format!("node.{}", node.kind),
                data_id: Some(id.to_string()),
                reason: format!("{format} cannot preserve object hierarchy or non-mesh nodes"),
                suggestion: Some("export as glb or blend".to_owned()),
            });
        }
        if node.rigid_body.is_some() {
            losses.push(Loss {
                feature_id: "physics.rigid_body".to_owned(),
                data_id: Some(id.to_string()),
                reason: format!("{format} does not preserve rigid-body simulation settings"),
                suggestion: Some("export as blend to preserve the physics data".to_owned()),
            });
        }
        if node.force_field.is_some() {
            losses.push(Loss {
                feature_id: "physics.force_field".to_owned(),
                data_id: Some(id.to_string()),
                reason: format!("{format} does not preserve force-field settings"),
                suggestion: Some("export as blend to preserve the physics data".to_owned()),
            });
        }
        if format == "alembic" {
            if !matches!(node.kind.as_str(), "mesh" | "empty") {
                losses.push(Loss {
                    feature_id: format!("node.{}", node.kind),
                    data_id: Some(id.to_string()),
                    reason: format!("Alembic cannot represent node kind `{}`", node.kind),
                    suggestion: Some("convert to a mesh or export as blend".to_owned()),
                });
            }
            if !node.visible || !node.render_visible || !node.selectable {
                losses.push(Loss {
                    feature_id: "node.visibility".to_owned(),
                    data_id: Some(id.to_string()),
                    reason: format!(
                        "Alembic cannot preserve visibility or selectability on `{id}`"
                    ),
                    suggestion: Some("export as blend to preserve node visibility".to_owned()),
                });
            }
            if !node.materials.is_empty() {
                losses.push(Loss {
                    feature_id: "material.binding".to_owned(),
                    data_id: Some(id.to_string()),
                    reason: format!("Alembic cannot represent material bindings on `{id}`"),
                    suggestion: Some(
                        "export as glb, fbx, or blend to preserve materials".to_owned(),
                    ),
                });
            }
            if !node.tags.is_empty() || !node.properties.is_empty() {
                losses.push(Loss {
                    feature_id: "node.metadata".to_owned(),
                    data_id: Some(id.to_string()),
                    reason: format!("Alembic cannot preserve custom node metadata on `{id}`"),
                    suggestion: Some("export as blend to preserve node metadata".to_owned()),
                });
            }
        }
    }
    for (scene_id, scene) in &doc.scenes {
        if scene.rigid_body_world.is_some() {
            losses.push(Loss {
                feature_id: "physics.rigid_body_world".to_owned(),
                data_id: Some(scene_id.to_string()),
                reason: format!("{format} does not preserve rigid-body world settings"),
                suggestion: Some("export as blend to preserve the physics data".to_owned()),
            });
        }
    }
    for (id, data) in &doc.data_blocks {
        if data.camera.is_some() && !supports_cameras {
            losses.push(Loss {
                feature_id: "camera".to_owned(),
                data_id: Some(id.to_string()),
                reason: format!("{format} does not represent cameras"),
                suggestion: Some("export as glb, gltf, usda, usdz, or blend".to_owned()),
            });
        }
        if let Some(light) = &data.light {
            let unsupported = !supports_lights
                || (matches!(format, "glb" | "gltf") && light.light_type == LightType::Area);
            if unsupported {
                losses.push(Loss {
                    feature_id: format!("light.{:?}", light.light_type).to_ascii_lowercase(),
                    data_id: Some(id.to_string()),
                    reason: format!("{format} does not represent this light type"),
                    suggestion: Some(
                        "export as glb, usda, or blend to preserve the light".to_owned(),
                    ),
                });
            }
        }
        let data_supported = matches!(data.data_type.as_str(), "mesh" | "camera" | "light")
            || (supports_strokes && data.data_type == "grease_pencil")
            || (supports_bvh && data.data_type == "armature")
            || (supports_gltf_skin && data.data_type == "armature");
        if !data_supported {
            losses.push(Loss {
                feature_id: format!("data.{}", data.data_type),
                data_id: Some(id.to_string()),
                reason: format!("{format} does not represent this Data-Block type"),
                suggestion: Some("export as blend to preserve the data".to_owned()),
            });
        }
    }
    losses
}

fn lossy_error(format: &str, losses: &[Loss]) -> PotError {
    PotError::with_details(
        ErrorCode::UnrepresentableFeature,
        format!("{format} cannot represent all scene features"),
        json!({ "result": { "format": format, "losses": losses } }),
    )
}

fn geometry_counts(meshes: &[exchange::ExchangeMesh]) -> Value {
    json!({
        "objects": meshes.len(),
        "vertices": meshes.iter().map(|mesh| mesh.positions.len()).sum::<usize>(),
        "faces": meshes.iter().map(|mesh| mesh.faces.len()).sum::<usize>(),
        "triangles": meshes.iter().map(|mesh| mesh.faces.iter().map(|face| face.len().saturating_sub(2)).sum::<usize>()).sum::<usize>(),
    })
}

fn export_conversions(
    doc: &SceneDoc,
    meshes: &[exchange::ExchangeMesh],
    format: &str,
) -> Vec<Value> {
    let mut values = Vec::new();
    if matches!(format, "obj" | "stl" | "ply" | "pdf" | "svg") {
        values
            .push(json!({ "feature_id": "transform.world", "conversion": "world_space_geometry" }));
    }
    if matches!(format, "stl" | "glb" | "gltf")
        && meshes
            .iter()
            .any(|mesh| mesh.faces.iter().any(|face| face.len() > 3))
    {
        values.push(json!({ "feature_id": "mesh.polygon", "conversion": "triangulation" }));
    }
    if matches!(format, "glb" | "gltf") {
        values.push(
            json!({ "feature_id": "coordinate_system", "conversion": "potter_z_up_to_gltf_y_up" }),
        );
    }
    if doc
        .nodes
        .values()
        .any(|node| node.modifiers.iter().any(|modifier| modifier.enabled))
    {
        values.push(json!({ "feature_id": "modifier.evaluation", "conversion": "evaluated_mesh" }));
    }
    if matches!(format, "glb" | "gltf" | "stl" | "ply" | "usda" | "usdz") {
        values.push(
            json!({ "feature_id": "numeric_precision", "conversion": "float32_quantization" }),
        );
    } else if format == "obj" {
        values.push(json!({ "feature_id": "numeric_precision", "conversion": "decimal_rounding" }));
    }
    values
}

fn sidecar_path(output: &Path, extension: &str) -> Result<PathBuf> {
    let stem = output
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or_else(|| PotError::new(ErrorCode::InvalidArgument, "output filename is invalid"))?;
    let mut file_name = std::ffi::OsString::from(stem);
    file_name.push(".");
    file_name.push(extension);
    Ok(output
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(file_name))
}

fn publish(artifacts: &[(PathBuf, Vec<u8>)], overwrite: bool) -> Result<()> {
    let main = artifacts
        .last()
        .map(|(path, _)| path.as_path())
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "export has no artifacts"))?;
    for (path, _) in artifacts {
        if path.exists() && !overwrite {
            return Err(PotError::with_details(
                ErrorCode::OutputExists,
                "output file already exists",
                json!({ "path": path }),
            ));
        }
    }
    let parent = main.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|error| PotError::io(&error))?;
    let mut staged = Vec::with_capacity(artifacts.len());
    for (path, bytes) in artifacts {
        let directory = path.parent().unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(directory).map_err(|error| PotError::io(&error))?;
        let temporary = directory.join(format!(".potter-{}.tmp", uuid::Uuid::new_v4()));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|error| PotError::io(&error))?;
        if let Err(error) = file.write_all(bytes).and_then(|()| file.sync_all()) {
            let _ = fs::remove_file(&temporary);
            for (pending, _) in &staged {
                let _ = fs::remove_file(pending);
            }
            return Err(PotError::io(&error));
        }
        staged.push((temporary, path.clone()));
    }
    for (temporary, destination) in &staged {
        if let Err(error) = fs::rename(temporary, destination) {
            for (pending, _) in &staged {
                let _ = fs::remove_file(pending);
            }
            return Err(PotError::io(&error));
        }
    }
    if let Ok(directory) = File::open(parent) {
        let _ = directory.sync_all();
    }
    Ok(())
}
