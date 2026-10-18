use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use glam::DMat4;
use serde_json::{Value, json};

use super::util::{parse_id, with_load_warnings};
use crate::{
    cli::{BakeArgs, BakeKind, NodeBakeMode},
    error::{ErrorCode, PotError, Result},
    eval::{EvaluationContext, EvaluationPurpose, Snapshot},
    geom::Mesh,
    graph::{NodeBakeFrame, NodeBakeRecord},
    model::{Id, SceneDoc},
    response::SceneInfo,
    store::Project,
};

pub fn run(args: BakeArgs) -> Result<(Option<SceneInfo>, Value)> {
    match args.kind {
        BakeKind::Simulation | BakeKind::Animation | BakeKind::Geometry => {}
        BakeKind::Texture => {
            return Err(unsupported(
                "bake.texture",
                "texture baking is not implemented",
            ));
        }
        BakeKind::NodeBake => return run_node_bake(args),
    }
    if args.modifier.is_some() || args.node.is_some() || args.mode.is_some() || args.delete {
        return Err(PotError::invalid_argument(
            "--modifier, --node, --mode, and --delete require --kind node-bake",
        ));
    }
    let output = args.out.as_deref().ok_or_else(|| {
        PotError::invalid_argument(
            "--out is required for animation, geometry, and simulation bakes",
        )
    })?;
    let project = Project::open(args.scene)?;
    let doc = project.doc();
    let target = args.target.as_deref().map(parse_id).transpose()?;
    if let Some(target) = &target
        && !doc.nodes.contains_key(target)
    {
        return Err(PotError::with_details(
            ErrorCode::TargetNotFound,
            "bake target Object was not found",
            json!({"target":target}),
        ));
    }
    let selected_scene_id = args
        .context
        .scene_id
        .as_deref()
        .map(parse_id)
        .transpose()?
        .unwrap_or_else(|| doc.active_scene.clone());
    let scene = doc.scenes.get(&selected_scene_id).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::TargetNotFound,
            "scene context not found",
            json!({"scene_id":selected_scene_id}),
        )
    })?;
    let frames = parse_frames(
        args.frames.as_deref(),
        args.context.frame.unwrap_or(scene.frame_current),
    )?;
    let context = EvaluationContext {
        scene_id: Some(selected_scene_id),
        view_layer: args
            .context
            .view_layer
            .as_deref()
            .map(parse_id)
            .transpose()?,
        frame: None,
        purpose: EvaluationPurpose::Render,
    };
    let cache_directory = project.cache_directory();
    let mut stage = BakeStage::create(output)?;
    let mut files = Vec::new();
    let mut baked_frames = Vec::new();
    for (frame_index, frame) in frames.into_iter().enumerate() {
        let mut frame_context = context.clone();
        frame_context.frame = Some(frame);
        let snapshot = Snapshot::evaluate_with_cache(
            doc,
            &frame_context,
            Some(project.path()),
            Some(cache_directory),
        )?;
        match args.kind {
            BakeKind::Animation => {
                let transforms = snapshot.nodes.iter().filter(|(id, _)| target.as_ref().is_none_or(|target| *id == target)).map(|(id, evaluated)| (id.to_string(), json!({"world_matrix":evaluated.world_matrix,"bounds":evaluated.bounds,"dimensions":evaluated.dimensions}))).collect::<BTreeMap<_,_>>();
                let name = format!("frame_{frame_index:08}.json");
                let path = stage.path.join(&name);
                write_json(
                    &path,
                    &json!({"frame":frame,"scene_id":snapshot.scene_id,"view_layer":snapshot.view_layer,"evaluation_hash":snapshot.evaluation_hash,"transforms":transforms}),
                )?;
                files.push(name);
                baked_frames.push(json!({"frame":frame,"evaluation_hash":snapshot.evaluation_hash,"objects":transforms.len()}));
            }
            BakeKind::Geometry => {
                let mut count = 0;
                for (id, mesh) in &snapshot.meshes {
                    if target.as_ref().is_some_and(|target| target != id) {
                        continue;
                    }
                    let Some(evaluated) = snapshot.nodes.get(id) else {
                        continue;
                    };
                    let name = format!("frame_{frame_index:08}_{id}.ply");
                    write_ply(
                        &stage.path.join(&name),
                        mesh,
                        DMat4::from_cols_array(&evaluated.world_matrix),
                    )?;
                    files.push(name);
                    count += 1;
                }
                baked_frames.push(json!({"frame":frame,"evaluation_hash":snapshot.evaluation_hash,"meshes":count}));
            }
            BakeKind::Simulation => {
                let matches_target = |id: &Id| target.as_ref().is_none_or(|target| target == id);
                let transforms = snapshot
                    .nodes
                    .iter()
                    .filter(|(id, _)| matches_target(id))
                    .map(|(id, evaluated)| {
                        let linear_velocity = snapshot
                            .simulated_velocities
                            .get(id)
                            .copied()
                            .unwrap_or([0.0, 0.0, 0.0]);
                        (
                            id.to_string(),
                            json!({
                                "world_matrix":evaluated.world_matrix,
                                "bounds":evaluated.bounds,
                                "dimensions":evaluated.dimensions,
                                "linear_velocity":linear_velocity
                            }),
                        )
                    })
                    .collect::<BTreeMap<_, _>>();
                let deformed_meshes = snapshot
                    .meshes
                    .iter()
                    .filter(|(id, _)| {
                        matches_target(id)
                            && doc.nodes.get(*id).is_some_and(|node| {
                                [
                                    "physics_cloth",
                                    "physics_soft_body",
                                    "physics_particle_emitter",
                                    "physics_fluid",
                                    "physics_dynamic_paint",
                                ]
                                .iter()
                                .any(|property| node.properties.contains_key(*property))
                            })
                    })
                    .map(|(id, mesh)| (id.to_string(), json!(mesh)))
                    .collect::<BTreeMap<_, _>>();
                let particles = snapshot
                    .simulated_particles
                    .iter()
                    .filter(|(id, _)| matches_target(id))
                    .map(|(id, states)| (id.to_string(), json!(states)))
                    .collect::<BTreeMap<_, _>>();
                let fluid_particles = snapshot
                    .fluid_particles
                    .iter()
                    .filter(|(id, _)| matches_target(id))
                    .map(|(id, positions)| (id.to_string(), json!(positions)))
                    .collect::<BTreeMap<_, _>>();
                let paint_colors = snapshot
                    .paint_colors
                    .iter()
                    .filter(|(id, _)| matches_target(id))
                    .map(|(id, colors)| (id.to_string(), json!(colors)))
                    .collect::<BTreeMap<_, _>>();
                let paint_weights = snapshot
                    .paint_weights
                    .iter()
                    .filter(|(id, _)| matches_target(id))
                    .map(|(id, weights)| (id.to_string(), json!(weights)))
                    .collect::<BTreeMap<_, _>>();
                let name = format!("frame_{frame_index:08}.json");
                write_json(
                    &stage.path.join(&name),
                    &json!({
                        "frame":frame,
                        "scene_id":snapshot.scene_id,
                        "view_layer":snapshot.view_layer,
                        "evaluation_hash":snapshot.evaluation_hash,
                        "cache_key":snapshot.simulation_cache_key,
                        "transforms":transforms,
                        "deformed_meshes":deformed_meshes,
                        "particles":particles,
                        "fluid_particles":fluid_particles,
                        "paint_colors":paint_colors,
                        "paint_weights":paint_weights
                    }),
                )?;
                files.push(name.clone());
                baked_frames.push(json!({
                    "frame":frame,
                    "evaluation_hash":snapshot.evaluation_hash,
                    "cache_key":snapshot.simulation_cache_key,
                    "file":name,
                    "objects":transforms.len(),
                    "particle_count":particles.values().map(|items| items.as_array().map_or(0, Vec::len)).sum::<usize>(),
                    "fluid_particle_count":fluid_particles.values().map(|items| items.as_array().map_or(0, Vec::len)).sum::<usize>(),
                    "painted_objects":paint_colors.len() + paint_weights.len(),
                    "deformed_meshes":deformed_meshes.len()
                }));
            }
            BakeKind::Texture => {
                return Err(unsupported(
                    "bake.texture",
                    "texture baking is not implemented",
                ));
            }
            BakeKind::NodeBake => unreachable!("node-bake was dispatched before export baking"),
        }
    }
    let cache_keys = baked_frames
        .iter()
        .filter_map(|record| record["cache_key"].as_str().map(str::to_owned))
        .collect::<Vec<_>>();
    write_json(
        &stage.path.join("manifest.json"),
        &json!({"schema_version":1,"kind":bake_kind_name(args.kind),"scene_id":doc.scene_id,"revision":doc.revision,"frames":baked_frames,"cache_keys":cache_keys,"files":files}),
    )?;
    files.push("manifest.json".to_owned());
    stage.publish(output, args.overwrite)?;
    let scene = project.info()?;
    let warnings = with_load_warnings(&[], &project)?;
    Ok((
        Some(scene),
        json!({"kind":bake_kind_name(args.kind),"path":output,"files":files,"frame_count":baked_frames.len(),"cache_keys":cache_keys,"scene_modified":false,"warnings":warnings}),
    ))
}
fn bake_kind_name(kind: BakeKind) -> &'static str {
    match kind {
        BakeKind::Simulation => "simulation",
        BakeKind::Texture => "texture",
        BakeKind::Geometry => "geometry",
        BakeKind::Animation => "animation",
        BakeKind::NodeBake => "node-bake",
    }
}
fn run_node_bake(args: BakeArgs) -> Result<(Option<SceneInfo>, Value)> {
    if args.out.is_some() || args.overwrite {
        return Err(PotError::invalid_argument(
            "--out and --overwrite apply only to export bake kinds",
        ));
    }
    let target = parse_id(
        args.target
            .as_deref()
            .ok_or_else(|| PotError::invalid_argument("--target is required for a node bake"))?,
    )?;
    let modifier_id =
        parse_id(args.modifier.as_deref().ok_or_else(|| {
            PotError::invalid_argument("--modifier is required for a node bake")
        })?)?;
    let node_id = Id::new(
        args.node
            .as_deref()
            .ok_or_else(|| PotError::invalid_argument("--node is required for a node bake"))?
            .to_owned(),
    )?;
    if args.delete && (args.mode.is_some() || args.frames.is_some()) {
        return Err(PotError::invalid_argument(
            "--mode and --frames cannot be combined with --delete",
        ));
    }

    let mut project = Project::open_exclusive(args.scene)?;
    if args.delete {
        return delete_node_bake(&mut project, &target, &modifier_id, &node_id);
    }

    let doc = project.doc();
    let object = doc.nodes.get(&target).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::TargetNotFound,
            "node-bake target object was not found",
            json!({"target":target}),
        )
    })?;
    if object.kind != "mesh"
        || object
            .data
            .as_ref()
            .and_then(|data_id| doc.data_blocks.get(data_id))
            .and_then(|data| data.mesh.as_ref())
            .is_none()
    {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            "node bake target must be a mesh object",
        ));
    }
    let modifier = object
        .modifiers
        .iter()
        .find(|modifier| modifier.id == modifier_id)
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::TargetNotFound,
                "node-bake modifier was not found on the target object",
                json!({"target":target,"modifier":modifier_id}),
            )
        })?;
    if modifier.modifier_type != "nodes" {
        return Err(PotError::invalid_argument(
            "node-bake modifier must have type `nodes`",
        ));
    }
    let old_config = node_bake_config(modifier, &node_id)?.cloned();
    if let Some(config) = &old_config {
        crate::graph::validate_node_bake_config(config)?;
    }
    let node_group_id = modifier
        .params
        .get("node_group")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::SceneInvalid,
                "node-bake modifier has no node_group",
            )
        })?;
    let node_group_id = Id::new(node_group_id.to_owned())
        .map_err(|_| PotError::new(ErrorCode::SceneInvalid, "node_group ID is invalid"))?;
    let group = doc.node_groups.get(&node_group_id).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::DependencyMissing,
            format!("node group `{node_group_id}` is missing"),
            json!({"node_group":node_group_id}),
        )
    })?;
    let bake_node = group.nodes.get(&node_id).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::TargetNotFound,
            "Bake node was not found in the modifier's node group",
            json!({"node":node_id,"node_group":node_group_id}),
        )
    })?;
    if bake_node.node_type != "GeometryNodeBake" {
        return Err(PotError::invalid_argument(
            "--node must identify a GeometryNodeBake node",
        ));
    }

    let selected_scene_id = args
        .context
        .scene_id
        .as_deref()
        .map(parse_id)
        .transpose()?
        .unwrap_or_else(|| doc.active_scene.clone());
    let scene = doc.scenes.get(&selected_scene_id).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::TargetNotFound,
            "scene context not found",
            json!({"scene_id":selected_scene_id}),
        )
    })?;
    let current_frame = args.context.frame.unwrap_or(scene.frame_current);
    if !current_frame.is_finite() {
        return Err(PotError::invalid_argument("node-bake frame must be finite"));
    }
    let mode = match args.mode {
        Some(mode) => mode,
        None => node_bake_mode(old_config.as_ref())?,
    };
    let frames = node_bake_frames(
        args.frames.as_deref(),
        mode,
        scene,
        current_frame,
        old_config.as_ref(),
    )?;
    let bake_target = effective_node_bake_target(modifier, old_config.as_ref())?;
    let bake_directory = effective_node_bake_directory(modifier, old_config.as_ref())?;
    let context = EvaluationContext {
        scene_id: Some(selected_scene_id),
        view_layer: args
            .context
            .view_layer
            .as_deref()
            .map(parse_id)
            .transpose()?,
        frame: None,
        purpose: EvaluationPurpose::Bake,
    };
    let project_root = project.path().to_path_buf();
    let cache_directory = project.cache_directory();
    let mut captured_frames = Vec::with_capacity(frames.len());
    for frame in &frames {
        let mut frame_context = context.clone();
        frame_context.frame = Some(*frame);
        let snapshot = Snapshot::evaluate_with_node_bake_capture(
            doc,
            &frame_context,
            Some(&project_root),
            Some(cache_directory),
            &target,
            &modifier_id,
            node_id.as_str(),
        )?;
        let captured = snapshot
            .node_bake_captures
            .get(&(target.clone(), modifier_id.clone(), node_id.to_string()))
            .cloned()
            .ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::EvaluationFailed,
                    "Bake node did not produce a captured frame",
                    json!({"target":target,"modifier":modifier_id,"node":node_id,"frame":frame}),
                )
            })?;
        captured_frames.push(captured);
    }
    let record = NodeBakeRecord {
        storage: "inline".to_owned(),
        kind: "node_bake".to_owned(),
        version: 1,
        frames: captured_frames,
    };
    let record_value = serde_json::to_value(&record).map_err(PotError::internal_json)?;
    crate::graph::validate_node_bake_resource(&record_value)?;

    let resource_id = if bake_target == "PACKED" {
        Some(node_bake_resource_id(
            doc,
            &target,
            &modifier_id,
            &node_id,
            old_config.as_ref(),
        )?)
    } else {
        None
    };
    let disk_frames = if bake_target == "DISK" {
        let mut entries = Vec::with_capacity(record.frames.len());
        for frame in &record.frames {
            let bytes = serde_json::to_vec(frame).map_err(PotError::internal_json)?;
            let digest = crate::hash::sha256(&bytes);
            crate::graph::write_node_bake_payload(
                &project_root,
                &bake_directory,
                &target,
                &modifier_id,
                node_id.as_str(),
                &digest,
                &bytes,
            )?;
            entries.push(json!({"frame":frame.frame,"digest":digest}));
        }
        Some(entries)
    } else {
        None
    };

    let mut config = old_config
        .as_ref()
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    if bake_target == "DISK" {
        config.insert("cache_directory".to_owned(), json!(bake_directory));
    } else {
        config.remove("cache_directory");
    }
    config.insert(
        "storage".to_owned(),
        json!(bake_target.to_ascii_lowercase()),
    );
    config.insert("bake_mode".to_owned(), json!(node_bake_mode_name(mode)));
    config
        .entry("bake_target".to_owned())
        .or_insert_with(|| json!("INHERIT"));
    if mode == NodeBakeMode::Animation && args.frames.is_some() {
        config.insert("use_custom_simulation_frame_range".to_owned(), json!(true));
        config.insert("frame_start".to_owned(), json!(frames[0]));
        config.insert("frame_end".to_owned(), json!(frames[frames.len() - 1]));
    } else if mode == NodeBakeMode::Animation {
        config
            .entry("use_custom_simulation_frame_range".to_owned())
            .or_insert_with(|| json!(false));
    }
    if let Some(resource_id) = &resource_id {
        config.insert("resource_id".to_owned(), json!(resource_id));
        config.remove("frames");
    } else {
        config.remove("resource_id");
        config.insert("frames".to_owned(), json!(disk_frames));
    }
    let config = Value::Object(config);
    let cleanup_config = config.clone();
    crate::graph::validate_node_bake_config(&config)?;

    let mut candidate = doc.clone();
    if let Some(resource_id) = &resource_id {
        candidate
            .resources
            .insert(resource_id.clone(), record_value);
    }
    {
        let modifier = candidate
            .nodes
            .get_mut(&target)
            .and_then(|object| {
                object
                    .modifiers
                    .iter_mut()
                    .find(|modifier| modifier.id == modifier_id)
            })
            .ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "node-bake modifier disappeared")
            })?;
        let bakes = modifier
            .params
            .entry("bakes".to_owned())
            .or_insert_with(|| json!({}));
        let bakes = bakes.as_object_mut().ok_or_else(|| {
            PotError::new(
                ErrorCode::SceneInvalid,
                "node-bake settings must be an object",
            )
        })?;
        bakes.insert(node_id.to_string(), config);
    }
    remove_unreferenced_node_bake_resource(&mut candidate, old_config.as_ref());

    let scene = project.restore(
        candidate,
        "bake",
        format!("{target}:{modifier_id}:{node_id}"),
        json!({"op":"geometry_node.bake","action":"bake","target":target,"modifier":modifier_id,"node":node_id,"storage":bake_target,"frames":frames}),
        json!({"nodes":[target],"resources":resource_id}),
    )?;
    delete_disk_node_bake_payloads(
        &project,
        &target,
        &modifier_id,
        &node_id,
        old_config.as_ref(),
        Some(&cleanup_config),
    )?;
    Ok((
        Some(scene),
        json!({"kind":"node-bake","action":"bake","target":target,"modifier":modifier_id,"node":node_id,"storage":bake_target.to_ascii_lowercase(),"resource_id":resource_id,"frames":frames,"scene_modified":true}),
    ))
}

fn node_bake_config<'a>(
    modifier: &'a crate::model::Modifier,
    node_id: &Id,
) -> Result<Option<&'a Value>> {
    let Some(bakes) = modifier.params.get("bakes") else {
        return Ok(None);
    };
    let bakes = bakes.as_object().ok_or_else(|| {
        PotError::new(
            ErrorCode::SceneInvalid,
            "node-bake settings must be an object",
        )
    })?;
    Ok(bakes.get(node_id.as_str()))
}

fn node_bake_mode(config: Option<&Value>) -> Result<NodeBakeMode> {
    match config
        .and_then(|config| config.get("bake_mode"))
        .and_then(Value::as_str)
    {
        None | Some("STILL") | Some("still") => Ok(NodeBakeMode::Still),
        Some("ANIMATION") | Some("animation") => Ok(NodeBakeMode::Animation),
        Some(mode) => Err(PotError::with_details(
            ErrorCode::SceneInvalid,
            "node-bake mode must be STILL or ANIMATION",
            json!({"bake_mode":mode}),
        )),
    }
}

fn node_bake_mode_name(mode: NodeBakeMode) -> &'static str {
    match mode {
        NodeBakeMode::Still => "STILL",
        NodeBakeMode::Animation => "ANIMATION",
    }
}

fn node_bake_frames(
    range: Option<&str>,
    mode: NodeBakeMode,
    scene: &crate::model::Scene,
    current_frame: f64,
    config: Option<&Value>,
) -> Result<Vec<f64>> {
    match mode {
        NodeBakeMode::Still => {
            let frames = parse_frames(range, current_frame)?;
            if frames.len() != 1 {
                return Err(PotError::invalid_argument(
                    "a STILL node bake must capture exactly one frame",
                ));
            }
            Ok(frames)
        }
        NodeBakeMode::Animation => {
            if let Some(range) = range {
                return parse_frames(Some(range), current_frame);
            }
            if config
                .and_then(|config| config.get("use_custom_simulation_frame_range"))
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                let start = config
                    .and_then(|config| config.get("frame_start"))
                    .and_then(Value::as_f64)
                    .unwrap_or(1.0);
                let end = config
                    .and_then(|config| config.get("frame_end"))
                    .and_then(Value::as_f64)
                    .unwrap_or(100.0);
                return parse_frames(Some(&format!("{start}:{end}")), current_frame);
            }
            parse_frames(
                Some(&format!("{}:{}", scene.frame_start, scene.frame_end)),
                current_frame,
            )
        }
    }
}

fn effective_node_bake_target(
    modifier: &crate::model::Modifier,
    config: Option<&Value>,
) -> Result<&'static str> {
    let modifier_target = modifier
        .params
        .get("bake_target")
        .and_then(Value::as_str)
        .unwrap_or("PACKED");
    let node_target = config
        .and_then(|config| config.get("bake_target"))
        .and_then(Value::as_str)
        .unwrap_or("INHERIT");
    let target = match node_target {
        "INHERIT" => modifier_target,
        "PACKED" | "DISK" => node_target,
        _ => {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "node-bake target must be INHERIT, PACKED, or DISK",
            ));
        }
    };
    match target {
        "PACKED" | "DISK" => Ok(target),
        _ => Err(PotError::new(
            ErrorCode::SceneInvalid,
            "modifier bake_target must be PACKED or DISK",
        )),
    }
}

fn effective_node_bake_directory(
    modifier: &crate::model::Modifier,
    config: Option<&Value>,
) -> Result<String> {
    if config
        .and_then(|config| config.get("use_custom_path"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return config
            .and_then(|config| config.get("directory"))
            .and_then(Value::as_str)
            .filter(|directory| !directory.is_empty())
            .map(str::to_owned)
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::InvalidArgument,
                    "custom node-bake disk directory is empty",
                )
            });
    }
    Ok(modifier
        .params
        .get("bake_directory")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned())
}

fn node_bake_resource_id(
    doc: &SceneDoc,
    target: &Id,
    modifier_id: &Id,
    node_id: &Id,
    old_config: Option<&Value>,
) -> Result<Id> {
    if let Some(resource_id) = old_config
        .filter(|config| config.get("storage").and_then(Value::as_str) == Some("packed"))
        .and_then(|config| config.get("resource_id"))
        .and_then(Value::as_str)
        && let Ok(resource_id) = Id::new(resource_id.to_owned())
        && doc.resources.get(&resource_id).is_some_and(|resource| {
            resource.get("storage").and_then(Value::as_str) == Some("inline")
                && resource.get("kind").and_then(Value::as_str) == Some("node_bake")
        })
    {
        return Ok(resource_id);
    }
    let identity = format!("{target}:{modifier_id}:{node_id}");
    let digest = crate::hash::sha256(identity.as_bytes());
    let suffix = digest.strip_prefix("sha256:").ok_or_else(|| {
        PotError::new(ErrorCode::InternalError, "invalid node-bake resource hash")
    })?;
    let base = format!("node_bake_{}", &suffix[..54]);
    for suffix in 0..10_000_u32 {
        let candidate = if suffix == 0 {
            base.clone()
        } else {
            let tail = format!("_{suffix}");
            format!("{}{}", &base[..64 - tail.len()], tail)
        };
        let candidate = Id::new(candidate)?;
        if !doc.resources.contains_key(&candidate) {
            return Ok(candidate);
        }
    }
    Err(PotError::new(
        ErrorCode::LimitExceeded,
        "node-bake resource IDs are exhausted",
    ))
}

fn remove_unreferenced_node_bake_resource(doc: &mut SceneDoc, config: Option<&Value>) {
    let Some(resource_id) = config
        .filter(|config| config.get("storage").and_then(Value::as_str) == Some("packed"))
        .and_then(|config| config.get("resource_id"))
        .and_then(Value::as_str)
        .and_then(|resource_id| Id::new(resource_id.to_owned()).ok())
    else {
        return;
    };
    let referenced = doc.nodes.values().any(|object| {
        object.modifiers.iter().any(|modifier| {
            let referenced_bake = modifier
                .params
                .get("bakes")
                .and_then(Value::as_object)
                .is_some_and(|bakes| {
                    bakes.values().any(|bake| {
                        bake.get("storage").and_then(Value::as_str) == Some("packed")
                            && bake.get("resource_id").and_then(Value::as_str)
                                == Some(resource_id.as_str())
                    })
                });
            referenced_bake
                || modifier.params.get("resource").and_then(Value::as_str)
                    == Some(resource_id.as_str())
        })
    });
    if !referenced
        && doc.resources.get(&resource_id).is_some_and(|resource| {
            resource.get("storage").and_then(Value::as_str) == Some("inline")
                && resource.get("kind").and_then(Value::as_str) == Some("node_bake")
        })
    {
        doc.resources.remove(&resource_id);
    }
}

fn delete_node_bake(
    project: &mut Project,
    target: &Id,
    modifier_id: &Id,
    node_id: &Id,
) -> Result<(Option<SceneInfo>, Value)> {
    let mut candidate = project.doc().clone();
    let modifier = candidate
        .nodes
        .get_mut(target)
        .and_then(|object| {
            object
                .modifiers
                .iter_mut()
                .find(|modifier| modifier.id == *modifier_id)
        })
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::TargetNotFound,
                "node-bake modifier was not found on the target object",
                json!({"target":target,"modifier":modifier_id}),
            )
        })?;
    let (removed, remove_map) = {
        let Some(bakes) = modifier.params.get_mut("bakes") else {
            let scene = project.info()?;
            return Ok((
                Some(scene),
                json!({"kind":"node-bake","action":"delete","target":target,"modifier":modifier_id,"node":node_id,"scene_modified":false}),
            ));
        };
        let bakes = bakes.as_object_mut().ok_or_else(|| {
            PotError::new(
                ErrorCode::SceneInvalid,
                "node-bake settings must be an object",
            )
        })?;
        let removed = bakes.remove(node_id.as_str());
        (removed, bakes.is_empty())
    };
    if removed.is_none() {
        let scene = project.info()?;
        return Ok((
            Some(scene),
            json!({"kind":"node-bake","action":"delete","target":target,"modifier":modifier_id,"node":node_id,"scene_modified":false}),
        ));
    }
    if remove_map {
        modifier.params.remove("bakes");
    }
    remove_unreferenced_node_bake_resource(&mut candidate, removed.as_ref());
    let scene = project.restore(
        candidate,
        "bake",
        format!("{target}:{modifier_id}:{node_id}"),
        json!({"op":"geometry_node.bake","action":"delete","target":target,"modifier":modifier_id,"node":node_id}),
        json!({"nodes":[target]}),
    )?;
    delete_disk_node_bake_payloads(
        project,
        target,
        modifier_id,
        node_id,
        removed.as_ref(),
        None,
    )?;
    Ok((
        Some(scene),
        json!({"kind":"node-bake","action":"delete","target":target,"modifier":modifier_id,"node":node_id,"scene_modified":true}),
    ))
}
fn delete_disk_node_bake_payloads(
    project: &Project,
    target: &Id,
    modifier_id: &Id,
    node_id: &Id,
    config: Option<&Value>,
    keep_config: Option<&Value>,
) -> Result<()> {
    let Some(config) =
        config.filter(|config| config.get("storage").and_then(Value::as_str) == Some("disk"))
    else {
        return Ok(());
    };
    let Some(frames) = config.get("frames").and_then(Value::as_array) else {
        return Ok(());
    };
    let use_custom_path = config
        .get("use_custom_path")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let directory = if use_custom_path {
        config
            .get("directory")
            .and_then(Value::as_str)
            .filter(|directory| !directory.is_empty())
    } else {
        config
            .get("cache_directory")
            .and_then(Value::as_str)
            .or_else(|| {
                project
                    .doc()
                    .nodes
                    .get(target)
                    .and_then(|object| {
                        object
                            .modifiers
                            .iter()
                            .find(|modifier| modifier.id == *modifier_id)
                    })
                    .and_then(|modifier| modifier.params.get("bake_directory"))
                    .and_then(Value::as_str)
            })
    };
    let Some(directory) = directory else {
        return Ok(());
    };
    for frame in frames {
        let Some(digest) = frame.get("digest").and_then(Value::as_str) else {
            continue;
        };
        if !digest.starts_with("sha256:")
            || digest.len() != "sha256:".len() + 64
            || !digest["sha256:".len()..]
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            continue;
        }
        if keep_config
            .and_then(|config| config.get("frames"))
            .and_then(Value::as_array)
            .is_some_and(|frames| {
                frames
                    .iter()
                    .any(|frame| frame.get("digest").and_then(Value::as_str) == Some(digest))
            })
        {
            continue;
        }
        crate::graph::delete_node_bake_payload(
            project.path(),
            directory,
            target,
            modifier_id,
            node_id.as_str(),
            digest,
        )?;
    }
    Ok(())
}

fn unsupported(feature_id: &str, message: &str) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        message,
        json!({"feature_id":feature_id,"status":"not_supported"}),
    )
}

fn parse_frames(range: Option<&str>, current: f64) -> Result<Vec<f64>> {
    let Some(range) = range else {
        return Ok(vec![current]);
    };
    let pieces = range.split(':').collect::<Vec<_>>();
    if !(2..=3).contains(&pieces.len()) {
        return Err(PotError::new(
            ErrorCode::InvalidArgument,
            "frames must be start:end[:step]",
        ));
    }
    let parse = |value: &str, name: &str| {
        value
            .parse::<f64>()
            .ok()
            .filter(|number| number.is_finite())
            .ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::InvalidArgument,
                    format!("{name} must be a finite number"),
                    json!({"frames":range}),
                )
            })
    };
    let start = parse(pieces[0], "frame start")?;
    let end = parse(pieces[1], "frame end")?;
    let step = if pieces.len() == 3 {
        parse(pieces[2], "frame step")?
    } else {
        1.0
    };
    if step <= 0.0 || start > end {
        return Err(PotError::new(
            ErrorCode::InvalidArgument,
            "frame range requires start <= end and a positive step",
        ));
    }
    let estimated = ((end - start) / step).floor();
    if !estimated.is_finite() || estimated > 100_000.0 {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "bake frame range exceeds 100001 frames",
        ));
    }
    let mut frames = Vec::with_capacity(estimated as usize + 1);
    let mut frame = start;
    while frame <= end + (step * f64::EPSILON * end.abs().max(1.0)) {
        frames.push(frame.min(end));
        frame += step;
    }
    Ok(frames)
}

fn write_json(path: &Path, value: &Value) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))?;
    fs::write(path, bytes).map_err(|error| PotError::io(&error))
}

fn write_ply(path: &Path, mesh: &Mesh, matrix: DMat4) -> Result<()> {
    let mut output = fs::File::create(path).map_err(|error| PotError::io(&error))?;
    writeln!(output, "ply\nformat ascii 1.0\nelement vertex {}\nproperty double x\nproperty double y\nproperty double z\nelement face {}\nproperty list uchar uint vertex_indices\nend_header", mesh.vertices.len(), mesh.faces.len()).map_err(|error| PotError::io(&error))?;
    let indices = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<BTreeMap<_, _>>();
    for vertex in &mesh.vertices {
        let world = matrix.transform_point3(vertex.co);
        writeln!(output, "{:.17} {:.17} {:.17}", world.x, world.y, world.z)
            .map_err(|error| PotError::io(&error))?;
    }
    for face in &mesh.faces {
        write!(output, "{}", face.vertices.len()).map_err(|error| PotError::io(&error))?;
        for vertex in &face.vertices {
            let index = indices.get(vertex).copied().ok_or_else(|| {
                PotError::new(
                    ErrorCode::SceneInvalid,
                    "mesh face references a missing vertex",
                )
            })?;
            write!(output, " {index}").map_err(|error| PotError::io(&error))?;
        }
        writeln!(output).map_err(|error| PotError::io(&error))?;
    }
    Ok(())
}

struct BakeStage {
    path: PathBuf,
    published: bool,
}

impl BakeStage {
    fn create(output: &Path) -> Result<Self> {
        let parent = output
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent).map_err(|error| PotError::io(&error))?;
        let name = output
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("bake");
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos());
        let path = parent.join(format!(
            ".{name}.potter-stage-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir(&path).map_err(|error| PotError::io(&error))?;
        Ok(Self {
            path,
            published: false,
        })
    }

    fn publish(&mut self, output: &Path, overwrite: bool) -> Result<()> {
        if output.exists() {
            if !overwrite {
                return Err(PotError::new(
                    ErrorCode::OutputExists,
                    "bake output already exists",
                ));
            }
            let backup = self.path.with_extension("backup");
            fs::rename(output, &backup).map_err(|error| PotError::io(&error))?;
            if let Err(error) = fs::rename(&self.path, output) {
                let _ = fs::rename(&backup, output);
                return Err(PotError::io(&error));
            }
            remove_path(&backup)?;
        } else {
            fs::rename(&self.path, output).map_err(|error| PotError::io(&error))?;
        }
        self.published = true;
        Ok(())
    }
}

impl Drop for BakeStage {
    fn drop(&mut self) {
        if !self.published {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

fn remove_path(path: &Path) -> Result<()> {
    if path.is_dir() {
        fs::remove_dir_all(path).map_err(|error| PotError::io(&error))
    } else {
        fs::remove_file(path).map_err(|error| PotError::io(&error))
    }
}
