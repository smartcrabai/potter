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
    cli::{BakeArgs, BakeKind, NodeBakeMode, NodeBakeUnpackMethod},
    error::{ErrorCode, PotError, Result},
    eval::{EvaluationContext, EvaluationPurpose, Snapshot},
    geom::Mesh,
    graph::{NodeBakeFrame, NodeBakePayloadTransaction, NodeBakeRecord, same_bake_frame},
    model::{Id, Modifier, OceanBakeCache, SceneDoc},
    response::SceneInfo,
    store::Project,
};

pub fn run(args: BakeArgs) -> Result<(Option<SceneInfo>, Value)> {
    match args.kind {
        BakeKind::Ocean => return run_ocean(args),
        BakeKind::Simulation | BakeKind::Animation | BakeKind::Geometry => {}
        BakeKind::Texture => {
            return Err(unsupported(
                "bake.texture",
                "texture baking is not implemented",
            ));
        }
        BakeKind::NodeBake => return run_node_bake(args),
    }
    if args.modifier.is_some()
        || args.node.is_some()
        || args.mode.is_some()
        || args.delete
        || args.pack
        || args.unpack
        || args.unpack_method.is_some()
    {
        return Err(PotError::invalid_argument(
            "--modifier, --node, --mode, --delete, --pack, and --unpack require --kind node-bake",
        ));
    }
    let output = export_output(args.out.as_deref())?;
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
        adaptive_render: None,
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
            BakeKind::Ocean => unreachable!("Ocean bakes are dispatched above"),
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
fn run_node_bake(args: BakeArgs) -> Result<(Option<SceneInfo>, Value)> {
    if args.overwrite {
        return Err(PotError::invalid_argument(
            "--overwrite applies only to export bake kinds",
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
    if args.unpack_method.is_some() && !args.unpack {
        return Err(PotError::invalid_argument(
            "--unpack-method requires --unpack",
        ));
    }
    let bake_action_count = u8::from(args.delete) + u8::from(args.pack) + u8::from(args.unpack);
    if bake_action_count > 1 {
        return Err(PotError::invalid_argument(
            "--delete, --pack, and --unpack are mutually exclusive",
        ));
    }
    if bake_action_count != 0 && (args.mode.is_some() || args.frames.is_some()) {
        return Err(PotError::invalid_argument(
            "--mode and --frames cannot be combined with --delete, --pack, or --unpack",
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
    // The modifier's `bakes` entry is the only source of bake settings: Blender keeps them on the
    // modifier (`NodesModifierBake`), and a Bake node can be shared by objects with different settings.
    let bake_settings = old_config
        .as_ref()
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let bake_settings = Value::Object(bake_settings);
    crate::graph::validate_node_bake_settings(&bake_settings)?;
    if args.pack {
        return pack_node_bake(
            &mut project,
            &target,
            &modifier_id,
            &node_id,
            &bake_settings,
        );
    }
    if args.unpack {
        return unpack_node_bake(
            &mut project,
            &target,
            &modifier_id,
            &node_id,
            &bake_settings,
            args.unpack_method.unwrap_or(NodeBakeUnpackMethod::UseLocal),
        );
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
        None => node_bake_mode(Some(&bake_settings))?,
    };
    let frames = node_bake_frames(
        args.frames.as_deref(),
        mode,
        scene,
        current_frame,
        Some(&bake_settings),
    )?;
    let requested_bake_target = effective_node_bake_target(modifier, Some(&bake_settings))?;
    let mut bake_target = requested_bake_target;
    let custom_path = bake_settings
        .get("use_custom_path")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    // Blender `bake_simulation_validate_paths` fills an empty modifier directory with its default
    // before a DISK bake (object_bake_simulation.cc:755-778). A custom path that is empty has no
    // location, and the bake is packed instead (get_node_bake_path, object_bake_simulation.cc:1005-1009).
    let mut default_directory = None;
    let mut resolved_directory = effective_node_bake_directory(modifier, Some(&bake_settings))?;
    if bake_target == "DISK" && resolved_directory.is_none() && !custom_path {
        if let Some(directory) =
            default_modifier_bake_directory(project.path(), &object.name, &modifier.name)
        {
            resolved_directory = Some(directory.clone());
            default_directory = Some(directory);
        }
    }
    if bake_target == "DISK" && resolved_directory.is_none() {
        bake_target = "PACKED";
    }
    let bake_directory = resolved_directory.unwrap_or_default();
    let selected_view_layer = args
        .context
        .view_layer
        .as_deref()
        .map(parse_id)
        .transpose()?;
    let project_root = project.path();
    let cache_directory = project.cache_directory();
    let mut captured_frames = Vec::with_capacity(frames.len());
    for frame in &frames {
        let context = EvaluationContext {
            scene_id: Some(selected_scene_id.clone()),
            view_layer: selected_view_layer.clone(),
            frame: Some(*frame),
            adaptive_render: None,
            purpose: EvaluationPurpose::Preview,
        };
        let snapshot = Snapshot::evaluate_with_node_bake_capture(
            doc,
            &context,
            Some(project_root),
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
    crate::graph::validate_node_bake_record(&record)?;
    let record_value = if resource_id.is_some() {
        Some(serde_json::to_value(&record).map_err(PotError::internal_json)?)
    } else {
        None
    };
    let mut transaction = NodeBakePayloadTransaction::new(project_root)?;
    let mut retained_payload_bytes = 0_usize;
    let disk_frames = if bake_target == "DISK" {
        let mut entries = Vec::with_capacity(record.frames.len());
        for frame in &record.frames {
            let bytes = match serde_json::to_vec(frame) {
                Ok(bytes) => bytes,
                Err(error) => {
                    return abort_node_bake_transaction(
                        transaction,
                        PotError::internal_json(error),
                    );
                }
            };
            if let Err(error) = account_node_bake_payload_bytes(&mut retained_payload_bytes, &bytes)
            {
                return abort_node_bake_transaction(transaction, error);
            }
            let digest = crate::hash::sha256(&bytes);
            if let Err(error) = transaction.stage_write(
                &bake_directory,
                &target,
                &modifier_id,
                node_id.as_str(),
                &digest,
                &bytes,
                frame.frame,
                false,
            ) {
                return abort_node_bake_transaction(transaction, error);
            }
            entries.push(json!({"frame":frame.frame,"digest":digest}));
        }
        Some(entries)
    } else {
        None
    };

    let candidate_result: Result<(SceneDoc, Value)> = (|| {
        let mut config = bake_settings.as_object().cloned().unwrap_or_default();
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
            config.insert("frame_start".to_owned(), json!(whole_frame(frames[0])?));
            config.insert(
                "frame_end".to_owned(),
                json!(whole_frame(frames[frames.len() - 1])?),
            );
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
            candidate.resources.insert(
                resource_id.clone(),
                record_value.ok_or_else(|| {
                    PotError::new(
                        ErrorCode::InternalError,
                        "packed node-bake resource value is missing",
                    )
                })?,
            );
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
            if let Some(directory) = &default_directory {
                modifier
                    .params
                    .insert("bake_directory".to_owned(), json!(directory));
            }
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
        candidate.validate()?;
        Ok((candidate, cleanup_config))
    })();
    let (candidate, cleanup_config) = match candidate_result {
        Ok(candidate) => candidate,
        Err(error) => return abort_node_bake_transaction(transaction, error),
    };
    if let Err(error) = stage_delete_node_bake_payloads(
        &mut transaction,
        &project,
        &target,
        &modifier_id,
        &node_id,
        old_config.as_ref(),
        Some(&cleanup_config),
    ) {
        return abort_node_bake_transaction(transaction, error);
    }
    if let Err(error) = transaction.publish() {
        return abort_node_bake_transaction(transaction, error);
    }
    let scene = match project.restore(
        candidate,
        "bake",
        format!("{target}:{modifier_id}:{node_id}"),
        json!({"op":"geometry_node.bake","action":"bake","target":target,"modifier":modifier_id,"node":node_id,"storage":bake_target,"frames":frames}),
        json!({"nodes":[target],"resources":resource_id}),
    ) {
        Ok(scene) => scene,
        Err(error) => return abort_node_bake_transaction(transaction, error),
    };
    transaction.commit()?;
    Ok((
        Some(scene),
        json!({"kind":"node-bake","action":"bake","target":target,"modifier":modifier_id,"node":node_id,"storage":bake_target.to_ascii_lowercase(),"resource_id":resource_id,"frame_count":frames.len(),"scene_modified":true}),
    ))
}

fn pack_node_bake(
    project: &mut Project,
    target: &Id,
    modifier_id: &Id,
    node_id: &Id,
    bake_settings: &Value,
) -> Result<(Option<SceneInfo>, Value)> {
    let doc = project.doc();
    let modifier = doc
        .nodes
        .get(target)
        .and_then(|object| {
            object
                .modifiers
                .iter()
                .find(|modifier| modifier.id == *modifier_id)
        })
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::TargetNotFound,
                "node-bake modifier was not found",
            )
        })?;
    let config = node_bake_config(modifier, node_id)?
        .cloned()
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::InvalidOperation,
                "Bake node has no disk cache to pack",
            )
        })?;
    crate::graph::validate_node_bake_config(&config)?;
    if config.get("storage").and_then(Value::as_str) != Some("disk")
        || effective_node_bake_target(modifier, Some(bake_settings))? != "DISK"
    {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            "only a disk Bake-node cache can be packed",
        ));
    }
    let bake_directory =
        effective_node_bake_directory(modifier, Some(bake_settings))?.ok_or_else(|| {
            PotError::new(
                ErrorCode::InvalidOperation,
                "a disk Bake-node cache has no directory to pack from",
            )
        })?;
    let frame_entries = config
        .get("frames")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::SceneInvalid,
                "disk node-bake frame manifest is missing",
            )
        })?;
    let mut frames = Vec::with_capacity(frame_entries.len());
    let mut retained_frame_memory = 0_usize;
    let mut retained_payload_bytes = 0_usize;
    for entry in frame_entries {
        let expected_frame = entry
            .get("frame")
            .and_then(Value::as_f64)
            .filter(|frame| frame.is_finite())
            .ok_or_else(|| {
                PotError::new(ErrorCode::SceneInvalid, "disk node-bake frame is invalid")
            })?;
        let digest = entry.get("digest").and_then(Value::as_str).ok_or_else(|| {
            PotError::new(ErrorCode::SceneInvalid, "disk node-bake digest is missing")
        })?;
        let remaining_payload_bytes = crate::graph::MAX_NODE_BAKE_PACKED_BYTES
            .checked_sub(retained_payload_bytes)
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "packed node-bake payloads exceed the byte limit",
                )
            })?;
        let remaining_frame_memory = crate::graph::MAX_NODE_BAKE_PACKED_BYTES
            .checked_sub(retained_frame_memory)
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "packed node-bake frames exceed the memory limit",
                )
            })?;
        let bounded_read_bytes = remaining_payload_bytes
            .min(remaining_frame_memory / (crate::graph::NODE_BAKE_JSON_MEMORY_MULTIPLIER + 1));
        let bytes = crate::graph::read_node_bake_payload_limited(
            project.path(),
            &bake_directory,
            target,
            modifier_id,
            node_id.as_str(),
            digest,
            bounded_read_bytes,
        )?
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::DependencyMissing,
                "disk node-bake payload is missing",
                json!({"node_id":node_id,"frame":expected_frame,"digest":digest}),
            )
        })?;
        account_node_bake_payload_bytes(&mut retained_payload_bytes, &bytes)?;
        let frame = serde_json::from_slice::<NodeBakeFrame>(&bytes).map_err(|error| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "disk node-bake payload is invalid",
                json!({"cause":error.to_string(),"digest":digest}),
            )
        })?;
        if !same_bake_frame(frame.frame, expected_frame) {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "disk node-bake frame does not match its manifest",
            ));
        }
        retained_frame_memory = retained_frame_memory
            .checked_add(crate::graph::node_bake_frame_memory_bytes(&frame))
            .filter(|bytes| *bytes <= crate::graph::MAX_NODE_BAKE_PACKED_BYTES)
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "packed node-bake frames exceed the memory limit",
                )
            })?;
        frames.push(frame);
    }
    let record = NodeBakeRecord {
        storage: "inline".to_owned(),
        kind: "node_bake".to_owned(),
        version: 1,
        frames,
    };
    crate::graph::validate_node_bake_record(&record)?;
    let resource_id = node_bake_resource_id(doc, target, modifier_id, node_id, Some(&config))?;
    let resource = serde_json::to_value(&record).map_err(PotError::internal_json)?;

    let mut next_config = bake_settings.as_object().cloned().ok_or_else(|| {
        PotError::new(
            ErrorCode::SceneInvalid,
            "node-bake settings must be an object",
        )
    })?;
    next_config.insert("storage".to_owned(), json!("packed"));
    next_config.insert("resource_id".to_owned(), json!(resource_id.as_str()));
    next_config.insert("bake_target".to_owned(), json!("PACKED"));
    next_config.remove("frames");
    next_config.remove("cache_directory");
    let next_config = Value::Object(next_config);
    crate::graph::validate_node_bake_config(&next_config)?;

    let mut candidate = doc.clone();
    candidate.resources.insert(resource_id.clone(), resource);
    update_node_bake_state(&mut candidate, target, modifier_id, node_id, next_config)?;
    candidate.validate()?;
    let scene = project.restore(
        candidate,
        "bake",
        format!("{target}:{modifier_id}:{node_id}"),
        json!({"op":"geometry_node.bake","action":"pack","target":target,"modifier":modifier_id,"node":node_id}),
        json!({"nodes":[target],"resources":resource_id.as_str()}),
    )?;
    Ok((
        Some(scene),
        json!({"kind":"node-bake","action":"pack","target":target,"modifier":modifier_id,"node":node_id,"storage":"packed","resource_id":resource_id.as_str(),"frame_count":record.frames.len(),"scene_modified":true}),
    ))
}

fn unpack_node_bake(
    project: &mut Project,
    target: &Id,
    modifier_id: &Id,
    node_id: &Id,
    bake_settings: &Value,
    method: NodeBakeUnpackMethod,
) -> Result<(Option<SceneInfo>, Value)> {
    let doc = project.doc();
    let modifier = doc
        .nodes
        .get(target)
        .and_then(|object| {
            object
                .modifiers
                .iter()
                .find(|modifier| modifier.id == *modifier_id)
        })
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::TargetNotFound,
                "node-bake modifier was not found",
            )
        })?;
    let config = node_bake_config(modifier, node_id)?
        .cloned()
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::InvalidOperation,
                "Bake node has no packed cache to unpack",
            )
        })?;
    crate::graph::validate_node_bake_config(&config)?;
    if config.get("storage").and_then(Value::as_str) != Some("packed")
        || effective_node_bake_target(modifier, Some(bake_settings))? != "PACKED"
    {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            "only a packed Bake-node cache can be unpacked",
        ));
    }
    let resource_id = config
        .get("resource_id")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::SceneInvalid,
                "packed node-bake resource ID is missing",
            )
        })?;
    let resource_id = Id::new(resource_id.to_owned()).map_err(|_| {
        PotError::new(
            ErrorCode::SceneInvalid,
            "packed node-bake resource ID is invalid",
        )
    })?;
    let resource = doc.resources.get(&resource_id).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::DependencyMissing,
            format!("node-bake resource `{resource_id}` is missing"),
            json!({"resource_id":resource_id}),
        )
    })?;
    if crate::model::migrate::inline_resource_kind(resource)? != Some("node_bake") {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "packed resource is not a node-bake resource",
        ));
    }
    let record = serde_json::from_value::<NodeBakeRecord>(resource.clone()).map_err(|error| {
        PotError::with_details(
            ErrorCode::SceneInvalid,
            "packed node-bake resource is invalid",
            json!({"cause":error.to_string()}),
        )
    })?;
    crate::graph::validate_node_bake_record(&record)?;
    let (bake_directory, use_custom_path, custom_directory) =
        node_bake_unpack_directory(method, modifier, bake_settings);
    let mut transaction = NodeBakePayloadTransaction::new(project.path())?;
    let replace_existing = matches!(
        method,
        NodeBakeUnpackMethod::WriteLocal | NodeBakeUnpackMethod::WriteOriginal
    );
    let mut retained_payload_bytes = 0_usize;
    let mut disk_frames = Vec::with_capacity(record.frames.len());
    for frame in &record.frames {
        let bytes = match serde_json::to_vec(frame) {
            Ok(bytes) => bytes,
            Err(error) => {
                return abort_node_bake_transaction(transaction, PotError::internal_json(error));
            }
        };
        if let Err(error) = account_node_bake_payload_bytes(&mut retained_payload_bytes, &bytes) {
            return abort_node_bake_transaction(transaction, error);
        }
        let digest = crate::hash::sha256(&bytes);
        if let Err(error) = transaction.stage_write(
            &bake_directory,
            target,
            modifier_id,
            node_id.as_str(),
            &digest,
            &bytes,
            frame.frame,
            replace_existing,
        ) {
            return abort_node_bake_transaction(transaction, error);
        }
        disk_frames.push(json!({"frame":frame.frame,"digest":digest}));
    }

    let candidate_result: Result<SceneDoc> = (|| {
        let mut next_config = bake_settings.as_object().cloned().ok_or_else(|| {
            PotError::new(
                ErrorCode::SceneInvalid,
                "node-bake settings must be an object",
            )
        })?;
        next_config.insert("storage".to_owned(), json!("disk"));
        next_config.insert("frames".to_owned(), json!(disk_frames));
        next_config.insert("cache_directory".to_owned(), json!(bake_directory));
        next_config.insert("use_custom_path".to_owned(), json!(use_custom_path));
        if let Some(directory) = custom_directory {
            next_config.insert("directory".to_owned(), json!(directory));
        }
        next_config.insert("bake_target".to_owned(), json!("DISK"));
        next_config.remove("resource_id");
        let next_config = Value::Object(next_config);
        crate::graph::validate_node_bake_config(&next_config)?;

        let mut candidate = doc.clone();
        update_node_bake_state(&mut candidate, target, modifier_id, node_id, next_config)?;
        remove_unreferenced_node_bake_resource(&mut candidate, Some(&config));
        candidate.validate()?;
        Ok(candidate)
    })();
    let candidate = match candidate_result {
        Ok(candidate) => candidate,
        Err(error) => return abort_node_bake_transaction(transaction, error),
    };
    if let Err(error) = transaction.publish() {
        return abort_node_bake_transaction(transaction, error);
    }
    let scene = match project.restore(
        candidate,
        "bake",
        format!("{target}:{modifier_id}:{node_id}"),
        json!({"op":"geometry_node.bake","action":"unpack","target":target,"modifier":modifier_id,"node":node_id,"storage":"disk","method":node_bake_unpack_method_name(method)}),
        json!({"nodes":[target],"resources":resource_id.as_str()}),
    ) {
        Ok(scene) => scene,
        Err(error) => return abort_node_bake_transaction(transaction, error),
    };
    transaction.commit()?;
    Ok((
        Some(scene),
        json!({"kind":"node-bake","action":"unpack","target":target,"modifier":modifier_id,"node":node_id,"storage":"disk","method":node_bake_unpack_method_name(method),"frame_count":record.frames.len(),"scene_modified":true}),
    ))
}

fn update_node_bake_state(
    doc: &mut SceneDoc,
    target: &Id,
    modifier_id: &Id,
    node_id: &Id,
    config: Value,
) -> Result<()> {
    {
        let modifier = doc
            .nodes
            .get_mut(target)
            .and_then(|object| {
                object
                    .modifiers
                    .iter_mut()
                    .find(|modifier| modifier.id == *modifier_id)
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
    Ok(())
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
        None | Some("STILL" | "still") => Ok(NodeBakeMode::Still),
        Some("ANIMATION" | "animation") => Ok(NodeBakeMode::Animation),
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
            // Blender `get_node_bake_frame_range`: the bake's own range, then the scene's custom
            // simulation range, then the scene frame range.
            let (start, end) = if config
                .and_then(|config| config.get("use_custom_simulation_frame_range"))
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                (
                    node_bake_config_frame(config, "frame_start")?,
                    node_bake_config_frame(config, "frame_end")?,
                )
            } else if scene.use_custom_simulation_range {
                (scene.simulation_frame_start, scene.simulation_frame_end)
            } else {
                (scene.frame_start, scene.frame_end)
            };
            let (start, end) = fix_frame_range(start, end);
            parse_frames(Some(&format!("{start}:{end}")), current_frame)
        }
    }
}

/// Blender `fix_frame_range` (`bake_geometry_nodes_modifier.cc`): a range that ends before it starts
/// bakes only its start frame.
fn fix_frame_range(start: i32, end: i32) -> (i32, i32) {
    (start, end.max(start))
}

fn node_bake_config_frame(config: Option<&Value>, key: &str) -> Result<i32> {
    config
        .and_then(|config| config.get(key))
        .and_then(Value::as_i64)
        .and_then(|frame| i32::try_from(frame).ok())
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "node-bake frame range is invalid",
                json!({"key":key}),
            )
        })
}

/// Blender stores bake frame bounds as 32-bit ints (`rna_modifier.cc`).
fn whole_frame(frame: f64) -> Result<i32> {
    match i32::try_from(frame as i64) {
        Ok(whole) if frame.fract() == 0.0 => Ok(whole),
        _ => Err(PotError::invalid_argument(
            "an animation node-bake frame range must use whole frames",
        )),
    }
}

/// Blender `get_default_modifier_bake_directory` (`bake_geometry_nodes_modifier.cc`). The project
/// directory stands in for the `.blend` file; `None` when it has no usable file stem.
fn default_modifier_bake_directory(
    project_root: &Path,
    object_name: &str,
    modifier_name: &str,
) -> Option<String> {
    let stem = project_root.file_stem()?.to_str()?;
    Some(format!(
        "//blendcache_{stem}/{}_{}",
        escape_bake_name(object_name),
        escape_bake_name(modifier_name)
    ))
}

/// Blender `escape_name`: ASCII letters are kept and every other UTF-8 byte is written as its
/// signed decimal value.
fn escape_bake_name(name: &str) -> String {
    name.bytes()
        .map(|byte| {
            if byte.is_ascii_alphabetic() {
                char::from(byte).to_string()
            } else {
                i8::from_ne_bytes([byte]).to_string()
            }
        })
        .collect()
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
        "PACKED" => Ok("PACKED"),
        "DISK" => Ok("DISK"),
        _ => Err(PotError::new(
            ErrorCode::SceneInvalid,
            "modifier bake_target must be PACKED or DISK",
        )),
    }
}

fn effective_node_bake_directory(
    modifier: &crate::model::Modifier,
    config: Option<&Value>,
) -> Result<Option<String>> {
    // `None` means the bake has no location on disk (get_node_bake_path returns no path).
    if config
        .and_then(|config| config.get("use_custom_path"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return Ok(config
            .and_then(|config| config.get("directory"))
            .and_then(Value::as_str)
            .filter(|directory| !directory.is_empty())
            .map(str::to_owned));
    }
    Ok(modifier
        .params
        .get("bake_directory")
        .and_then(Value::as_str)
        .filter(|directory| !directory.is_empty())
        .map(str::to_owned))
}
fn node_bake_unpack_directory(
    method: NodeBakeUnpackMethod,
    modifier: &crate::model::Modifier,
    config: &Value,
) -> (String, bool, Option<String>) {
    const LOCAL_DIRECTORY: &str = "//.potter/bakes";
    let local = || {
        (
            LOCAL_DIRECTORY.to_owned(),
            true,
            Some(LOCAL_DIRECTORY.to_owned()),
        )
    };
    match method {
        NodeBakeUnpackMethod::UseLocal | NodeBakeUnpackMethod::WriteLocal => local(),
        NodeBakeUnpackMethod::UseOriginal | NodeBakeUnpackMethod::WriteOriginal => {
            if config
                .get("use_custom_path")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                if let Some(directory) = config
                    .get("directory")
                    .and_then(Value::as_str)
                    .filter(|directory| !directory.is_empty())
                {
                    return (directory.to_owned(), true, Some(directory.to_owned()));
                }
                return local();
            }
            if let Some(directory) = modifier
                .params
                .get("bake_directory")
                .and_then(Value::as_str)
                .filter(|directory| !directory.is_empty())
            {
                return (directory.to_owned(), false, None);
            }
            local()
        }
    }
}
fn node_bake_unpack_method_name(method: NodeBakeUnpackMethod) -> &'static str {
    match method {
        NodeBakeUnpackMethod::UseLocal => "USE_LOCAL",
        NodeBakeUnpackMethod::WriteLocal => "WRITE_LOCAL",
        NodeBakeUnpackMethod::UseOriginal => "USE_ORIGINAL",
        NodeBakeUnpackMethod::WriteOriginal => "WRITE_ORIGINAL",
    }
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
        && let Some(resource) = doc.resources.get(&resource_id)
        && crate::model::migrate::inline_resource_kind(resource)? == Some("node_bake")
        && !node_bake_resource_referenced_elsewhere(doc, &resource_id, target, modifier_id, node_id)
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
fn node_bake_resource_referenced_elsewhere(
    doc: &SceneDoc,
    resource_id: &Id,
    target: &Id,
    modifier_id: &Id,
    node_id: &Id,
) -> bool {
    doc.nodes.iter().any(|(owner_id, object)| {
        object.modifiers.iter().any(|modifier| {
            modifier.params.get("resource").and_then(Value::as_str) == Some(resource_id.as_str())
                || modifier
                    .params
                    .get("bakes")
                    .and_then(Value::as_object)
                    .is_some_and(|bakes| {
                        bakes.iter().any(|(other_node_id, config)| {
                            let same_bake = owner_id == target
                                && modifier.id.as_str() == modifier_id.as_str()
                                && other_node_id.as_str() == node_id.as_str();
                            !same_bake
                                && config.get("resource_id").and_then(Value::as_str)
                                    == Some(resource_id.as_str())
                        })
                    })
        })
    })
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
            matches!(
                crate::model::migrate::inline_resource_kind(resource),
                Ok(Some("node_bake"))
            )
        })
    {
        doc.resources.remove(&resource_id);
    }
}

/// The settings that survive a payload delete (Blender `try_delete_bake` keeps the bake settings).
fn node_bake_settings_only(config: &Value) -> Value {
    let mut settings = config.as_object().cloned().unwrap_or_default();
    for key in ["storage", "resource_id", "frames", "cache_directory"] {
        settings.remove(key);
    }
    Value::Object(settings)
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
    // Blender `try_delete_bake` clears only the payload and keeps the NodesModifierBake settings.
    let payload = match modifier.params.get_mut("bakes") {
        None => None,
        Some(bakes) => {
            let bakes = bakes.as_object_mut().ok_or_else(|| {
                PotError::new(
                    ErrorCode::SceneInvalid,
                    "node-bake settings must be an object",
                )
            })?;
            let payload = bakes
                .get(node_id.as_str())
                .filter(|config| config.get("storage").is_some())
                .cloned();
            if let Some(config) = &payload {
                bakes.insert(node_id.to_string(), node_bake_settings_only(config));
            }
            payload
        }
    };
    let Some(payload) = payload else {
        let scene = project.info()?;
        return Ok((
            Some(scene),
            json!({"kind":"node-bake","action":"delete","target":target,"modifier":modifier_id,"node":node_id,"scene_modified":false}),
        ));
    };
    remove_unreferenced_node_bake_resource(&mut candidate, Some(&payload));
    candidate.validate()?;

    let mut transaction = NodeBakePayloadTransaction::new(project.path())?;
    if let Err(error) = stage_delete_node_bake_payloads(
        &mut transaction,
        project,
        target,
        modifier_id,
        node_id,
        Some(&payload),
        None,
    ) {
        return abort_node_bake_transaction(transaction, error);
    }
    if let Err(error) = transaction.publish() {
        return abort_node_bake_transaction(transaction, error);
    }
    let scene = match project.restore(
        candidate,
        "bake",
        format!("{target}:{modifier_id}:{node_id}"),
        json!({"op":"geometry_node.bake","action":"delete","target":target,"modifier":modifier_id,"node":node_id}),
        json!({"nodes":[target]}),
    ) {
        Ok(scene) => scene,
        Err(error) => return abort_node_bake_transaction(transaction, error),
    };
    transaction.commit()?;
    Ok((
        Some(scene),
        json!({"kind":"node-bake","action":"delete","target":target,"modifier":modifier_id,"node":node_id,"scene_modified":true}),
    ))
}

fn abort_node_bake_transaction<T>(
    transaction: NodeBakePayloadTransaction,
    error: PotError,
) -> Result<T> {
    transaction.rollback()?;
    Err(error)
}
fn account_node_bake_payload_bytes(retained: &mut usize, bytes: &[u8]) -> Result<()> {
    let Some(total) = retained
        .checked_add(bytes.len())
        .filter(|total| *total <= crate::graph::MAX_NODE_BAKE_PACKED_BYTES)
    else {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "node-bake payloads exceed the aggregate byte limit",
        ));
    };
    *retained = total;
    Ok(())
}

fn stage_delete_node_bake_payloads(
    transaction: &mut NodeBakePayloadTransaction,
    project: &Project,
    target: &Id,
    modifier_id: &Id,
    node_id: &Id,
    config: Option<&Value>,
    keep_config: Option<&Value>,
) -> Result<()> {
    let Some(config) = config else {
        return Ok(());
    };
    crate::graph::validate_node_bake_config(config)?;
    // A settings-only entry has no payload to delete.
    let Some(storage) = config.get("storage").and_then(Value::as_str) else {
        return Ok(());
    };
    if !matches!(storage, "disk" | "packed") {
        return Ok(());
    }
    let directory = node_bake_payload_directory(project, target, modifier_id, config)?;
    let keep_config =
        keep_config.filter(|config| config.get("storage").and_then(Value::as_str) == Some("disk"));
    let keep_frames = if let Some(keep_config) = keep_config {
        crate::graph::validate_node_bake_config(keep_config)?;
        Some(
            keep_config
                .get("frames")
                .and_then(Value::as_array)
                .map(Vec::as_slice)
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::SceneInvalid,
                        "disk node-bake frame manifest is missing",
                    )
                })?,
        )
    } else {
        None
    };
    let keep_same_directory = if let Some(keep_config) = keep_config {
        let keep_directory =
            node_bake_payload_directory(project, target, modifier_id, keep_config)?;
        node_bake_directory_key(&directory) == node_bake_directory_key(&keep_directory)
    } else {
        false
    };

    match storage {
        "disk" => {
            let frames = config
                .get("frames")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::SceneInvalid,
                        "disk node-bake frame manifest is missing",
                    )
                })?;
            for frame in frames {
                let expected_frame = frame
                    .get("frame")
                    .and_then(Value::as_f64)
                    .filter(|frame| frame.is_finite())
                    .ok_or_else(|| {
                        PotError::new(ErrorCode::SceneInvalid, "disk node-bake frame is invalid")
                    })?;
                let digest = frame.get("digest").and_then(Value::as_str).ok_or_else(|| {
                    PotError::new(ErrorCode::SceneInvalid, "disk node-bake digest is missing")
                })?;
                if !node_bake_payload_is_kept(
                    keep_same_directory,
                    keep_frames,
                    expected_frame,
                    digest,
                ) {
                    transaction.stage_delete(
                        &directory,
                        target,
                        modifier_id,
                        node_id.as_str(),
                        digest,
                        expected_frame,
                    )?;
                }
            }
        }
        "packed" => {
            let record = node_bake_packed_record(project, config)?;
            for frame in &record.frames {
                let bytes = serde_json::to_vec(frame).map_err(PotError::internal_json)?;
                let digest = crate::hash::sha256(&bytes);
                if !node_bake_payload_is_kept(
                    keep_same_directory,
                    keep_frames,
                    frame.frame,
                    &digest,
                ) {
                    transaction.stage_delete(
                        &directory,
                        target,
                        modifier_id,
                        node_id.as_str(),
                        &digest,
                        frame.frame,
                    )?;
                }
            }
        }
        _ => unreachable!("node-bake storage was validated"),
    }
    Ok(())
}

fn node_bake_payload_is_kept(
    same_directory: bool,
    keep_frames: Option<&[Value]>,
    expected_frame: f64,
    digest: &str,
) -> bool {
    same_directory
        && keep_frames.is_some_and(|frames| {
            frames.iter().any(|frame| {
                frame.get("frame").and_then(Value::as_f64) == Some(expected_frame)
                    && frame.get("digest").and_then(Value::as_str) == Some(digest)
            })
        })
}

fn node_bake_packed_record(project: &Project, config: &Value) -> Result<NodeBakeRecord> {
    let resource_id = config
        .get("resource_id")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::SceneInvalid,
                "packed node-bake resource ID is missing",
            )
        })?;
    let resource_id = Id::new(resource_id.to_owned()).map_err(|_| {
        PotError::new(
            ErrorCode::SceneInvalid,
            "packed node-bake resource ID is invalid",
        )
    })?;
    let resource = project.doc().resources.get(&resource_id).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::DependencyMissing,
            format!("node-bake resource `{resource_id}` is missing"),
            json!({"resource_id":resource_id}),
        )
    })?;
    if crate::model::migrate::inline_resource_kind(resource)? != Some("node_bake") {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "packed resource is not a node-bake resource",
        ));
    }
    let record = serde_json::from_value::<NodeBakeRecord>(resource.clone()).map_err(|error| {
        PotError::with_details(
            ErrorCode::SceneInvalid,
            "packed node-bake resource is invalid",
            json!({"cause":error.to_string()}),
        )
    })?;
    crate::graph::validate_node_bake_record(&record)?;
    Ok(record)
}

fn node_bake_payload_directory(
    project: &Project,
    target: &Id,
    modifier_id: &Id,
    config: &Value,
) -> Result<String> {
    if config
        .get("use_custom_path")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return config
            .get("directory")
            .and_then(Value::as_str)
            .filter(|directory| !directory.is_empty())
            .map(str::to_owned)
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::SceneInvalid,
                    "custom node-bake directory is missing",
                )
            });
    }
    Ok(config
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
        .unwrap_or_default()
        .to_owned())
}

fn node_bake_directory_key(directory: &str) -> PathBuf {
    let directory = if directory.is_empty() {
        ".potter/bakes"
    } else {
        directory.strip_prefix("//").unwrap_or(directory)
    };
    PathBuf::from(directory)
}

fn run_ocean(args: BakeArgs) -> Result<(Option<SceneInfo>, Value)> {
    if args.frames.is_some() {
        return Err(PotError::invalid_argument(
            "Ocean bake uses frame_start and frame_end from the modifier, not --frames",
        ));
    }
    let target_id = args
        .target
        .as_deref()
        .ok_or_else(|| PotError::invalid_argument("Ocean bake requires --target <modifier-id>"))
        .and_then(parse_id)?;
    let out = export_output(args.out.as_deref())?;
    let mut project = Project::open_exclusive(args.scene)?;
    if out.exists() && !args.overwrite {
        return Err(PotError::new(
            ErrorCode::OutputExists,
            "bake output already exists",
        ));
    }
    let mut matching = project.doc().nodes.iter().flat_map(|(node_id, node)| {
        node.modifiers
            .iter()
            .filter(|modifier| modifier.id == target_id)
            .map(move |modifier| (node_id, modifier))
    });
    let Some((node_id, modifier)) = matching.next() else {
        return Err(PotError::with_details(
            ErrorCode::TargetNotFound,
            "Ocean bake modifier was not found",
            json!({"modifier_id":target_id}),
        ));
    };
    if matching.next().is_some() {
        return Err(PotError::with_details(
            ErrorCode::InvalidArgument,
            "Ocean bake modifier ID is ambiguous",
            json!({"modifier_id":target_id}),
        ));
    }
    if modifier.modifier_type != "ocean" {
        return Err(PotError::with_details(
            ErrorCode::InvalidOperation,
            "bake target modifier must be an Ocean modifier",
            json!({"modifier_id":target_id,"type":modifier.modifier_type}),
        ));
    }
    let node_id = node_id.clone();
    let node = project.doc().nodes.get(&node_id).ok_or_else(|| {
        PotError::new(
            ErrorCode::TargetNotFound,
            "Ocean bake target node disappeared",
        )
    })?;
    let modifier = modifier.clone();
    let frame_start = ocean_i32_parameter(&modifier, "frame_start")?;
    let frame_end = ocean_i32_parameter(&modifier, "frame_end")?;
    if frame_start > frame_end {
        return Err(PotError::invalid_argument(
            "Ocean frame_start must not exceed frame_end",
        ));
    }
    let frame_count = usize::try_from(i64::from(frame_end) - i64::from(frame_start) + 1)
        .map_err(|_| PotError::new(ErrorCode::LimitExceeded, "Ocean frame range is too large"))?;
    if frame_count > 100_001 {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "Ocean bake frame range exceeds 100001 frames",
        ));
    }
    let base_time = ocean_time_parameter(&modifier)?;
    let filepath = ocean_parameter(&modifier, "filepath")?
        .as_str()
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::InvalidOperation,
                "Ocean `filepath` must be a directory path",
                json!({"modifier_id":modifier.id,"parameter":"filepath"}),
            )
        })?;
    let cache_path = crate::geom::modifiers::ocean_cache_directory(filepath, Some(project.path()))?;
    if cache_path.exists() && !args.overwrite {
        return Err(PotError::new(
            ErrorCode::OutputExists,
            "Ocean cache filepath already exists; use --overwrite to replace it",
        ));
    }
    let mut sequence = crate::geom::modifiers::OceanBakeSequence::new(&modifier)?;
    let mut cache_stage = BakeStage::create(&cache_path)?;
    let mut report_stage = BakeStage::create(out)?;
    let project_destination = canonical_destination(project.path())?;
    let cache_destination = canonical_destination(&cache_path)?;
    let report_destination = canonical_destination(out)?;
    if project_destination.starts_with(&cache_destination)
        || project_destination.starts_with(&report_destination)
    {
        return Err(PotError::invalid_argument(
            "Ocean cache and report directories must not contain the project directory",
        ));
    }
    if cache_destination.starts_with(&report_destination)
        || report_destination.starts_with(&cache_destination)
    {
        return Err(PotError::invalid_argument(
            "Ocean cache and report directories must be separate, non-nested directories",
        ));
    }

    let mut manifest = BTreeMap::new();
    let mut cache_files = Vec::new();
    let mut resolution = None;
    for frame in frame_start..=frame_end {
        let time = crate::eval::animation::animated_modifier_scalar(
            &node_id,
            node,
            project.doc(),
            &target_id,
            "time",
            base_time,
            f64::from(frame),
        )?
        .unwrap_or(base_time) as f32;
        if !time.is_finite() {
            return Err(PotError::with_details(
                ErrorCode::InvalidOperation,
                "Ocean animated `time` must be finite as a 32-bit float",
                json!({"modifier_id":target_id,"parameter":"time","frame":frame}),
            ));
        }
        for image in sequence.bake_frame(frame, time)? {
            let image_resolution = u32::try_from(image.resolution).map_err(|_| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "Ocean EXR resolution exceeds limits",
                )
            })?;
            if image_resolution == 0 || resolution.is_some_and(|value| value != image_resolution) {
                return Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    "Ocean bake images have inconsistent resolutions",
                ));
            }
            resolution = Some(image_resolution);
            if manifest.contains_key(&image.filename) {
                return Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    "Ocean bake generated duplicate image filenames",
                ));
            }
            let path = cache_stage.path.join(&image.filename);
            write_ocean_exr(&path, &image)?;
            let digest =
                crate::hash::sha256(&fs::read(&path).map_err(|error| PotError::io(&error))?);
            manifest.insert(image.filename.clone(), digest);
            cache_files.push(image.filename);
        }
    }
    let resolution = resolution.ok_or_else(|| {
        PotError::new(
            ErrorCode::SceneInvalid,
            "Ocean bake produced no cache images",
        )
    })?;
    let report = json!({
        "schema_version": 1,
        "kind": "ocean",
        "modifier_id": target_id,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "resolution": resolution,
        "cache_path": cache_path,
        "files": &manifest,
    });
    write_json(&report_stage.path.join("manifest.json"), &report)?;

    let cache = OceanBakeCache {
        frame_start,
        frame_end,
        resolution,
        manifest,
    };
    let mut candidate = project.doc().clone();
    let modifier = candidate
        .nodes
        .get_mut(&node_id)
        .and_then(|node| {
            node.modifiers
                .iter_mut()
                .find(|modifier| modifier.id == target_id)
        })
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::TargetNotFound,
                "Ocean bake modifier disappeared before commit",
            )
        })?;
    modifier.baked_ocean_cache = Some(cache);
    let operations = json!([{
        "op": "bake.ocean",
        "target": node_id,
        "modifier_id": target_id,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "resolution": resolution,
    }]);
    let changes = json!({"nodes":{"updated":[node_id]}});
    let scene = cache_stage.publish_with_commit(&cache_path, args.overwrite, || {
        project.commit_with_assets(candidate, operations, changes, &BTreeMap::new())
    })?;
    report_stage.publish(out, args.overwrite)?;
    let warnings = with_load_warnings(&[], &project)?;
    Ok((
        Some(scene),
        json!({
            "kind": "ocean",
            "path": out,
            "files": ["manifest.json"],
            "frame_count": frame_count,
            "cache_path": cache_path,
            "cache_files": cache_files,
            "scene_modified": true,
            "warnings": warnings,
        }),
    ))
}

fn ocean_parameter<'a>(modifier: &'a Modifier, name: &str) -> Result<&'a Value> {
    modifier
        .params
        .get(name)
        .or_else(|| {
            crate::params::default_value(crate::params::ParameterFamily::Modifier, "ocean", name)
        })
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::InvalidOperation,
                format!("Ocean modifier has no `{name}` parameter or schema default"),
                json!({"modifier_id":modifier.id,"parameter":name}),
            )
        })
}

fn ocean_i32_parameter(modifier: &Modifier, name: &str) -> Result<i32> {
    ocean_parameter(modifier, name)?
        .as_i64()
        .and_then(|value| i32::try_from(value).ok())
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::InvalidOperation,
                format!("Ocean `{name}` must be a 32-bit integer"),
                json!({"modifier_id":modifier.id,"parameter":name}),
            )
        })
}

fn ocean_time_parameter(modifier: &Modifier) -> Result<f64> {
    let time = ocean_parameter(modifier, "time")?.as_f64().ok_or_else(|| {
        PotError::with_details(
            ErrorCode::InvalidOperation,
            "Ocean `time` must be numeric",
            json!({"modifier_id":modifier.id,"parameter":"time"}),
        )
    })?;
    if !time.is_finite() {
        return Err(PotError::with_details(
            ErrorCode::InvalidOperation,
            "Ocean `time` must be finite",
            json!({"modifier_id":modifier.id,"parameter":"time"}),
        ));
    }
    Ok(time)
}

fn canonical_destination(path: &Path) -> Result<PathBuf> {
    if path.exists() {
        return fs::canonicalize(path).map_err(|error| PotError::io(&error));
    }
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .ok_or_else(|| PotError::invalid_argument("bake directory path must have a name"))?;
    Ok(fs::canonicalize(parent)
        .map_err(|error| PotError::io(&error))?
        .join(name))
}

fn bake_kind_name(kind: BakeKind) -> &'static str {
    match kind {
        BakeKind::Animation => "animation",
        BakeKind::Geometry => "geometry",
        BakeKind::Ocean => "ocean",
        BakeKind::Simulation => "simulation",
        BakeKind::Texture => "texture",
        BakeKind::NodeBake => "node-bake",
    }
}

/// The `--out` directory every export bake kind requires; node bakes never take one.
fn export_output(out: Option<&Path>) -> Result<&Path> {
    out.ok_or_else(|| {
        PotError::invalid_argument(
            "--out is required for animation, geometry, simulation, and Ocean bakes",
        )
    })
}

fn write_ocean_exr(path: &Path, image: &crate::geom::modifiers::OceanBakeImage) -> Result<()> {
    let resolution = image.resolution;
    let pixel_count = resolution.checked_mul(resolution).ok_or_else(|| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "Ocean EXR dimensions exceed limits",
        )
    })?;
    if image.pixels.len() != pixel_count {
        return Err(PotError::new(
            ErrorCode::BakeFailed,
            "Ocean EXR pixel count does not match its dimensions",
        ));
    }
    exr::prelude::write_rgba_file(path, resolution, resolution, |x, y| {
        // Blender writes the image's top row first, so file row `y` is buffer row `height - 1 - y`.
        let pixel = image.pixels[(resolution - 1 - y) * resolution + x];
        (
            exr::prelude::f16::from_f32(pixel[0]),
            exr::prelude::f16::from_f32(pixel[1]),
            exr::prelude::f16::from_f32(pixel[2]),
            exr::prelude::f16::from_f32(pixel[3]),
        )
    })
    .map_err(|error| {
        PotError::new(
            ErrorCode::BakeFailed,
            format!("Ocean EXR encoding failed: {error}"),
        )
    })
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

    fn publish_with_commit<T>(
        &mut self,
        output: &Path,
        overwrite: bool,
        commit: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let backup = if output.exists() {
            if !overwrite {
                return Err(PotError::new(
                    ErrorCode::OutputExists,
                    "Ocean cache filepath already exists",
                ));
            }
            let backup = self.path.with_extension("backup");
            if backup.exists() {
                return Err(PotError::new(
                    ErrorCode::OutputExists,
                    "Ocean cache backup path already exists",
                ));
            }
            fs::rename(output, &backup).map_err(|error| PotError::io(&error))?;
            Some(backup)
        } else {
            None
        };
        if let Err(error) = fs::rename(&self.path, output) {
            if let Some(backup) = &backup
                && let Err(restore_error) = fs::rename(backup, output)
            {
                return Err(PotError::new(
                    ErrorCode::InternalError,
                    format!(
                        "Ocean cache publish failed: {error}; restoring the previous cache failed: {restore_error}"
                    ),
                ));
            }
            return Err(PotError::io(&error));
        }
        match commit() {
            Ok(value) => {
                self.published = true;
                if let Some(backup) = backup {
                    let _ = remove_path(&backup);
                }
                Ok(value)
            }
            Err(error) => {
                if let Err(rename_error) = fs::rename(output, &self.path)
                    && remove_path(output).is_err()
                {
                    return Err(PotError::new(
                        ErrorCode::InternalError,
                        format!(
                            "project commit failed: {}; removing the staged Ocean cache also failed: {rename_error}",
                            error.message
                        ),
                    ));
                }
                if let Some(backup) = backup {
                    fs::rename(&backup, output).map_err(|restore_error| {
                        PotError::new(
                            ErrorCode::InternalError,
                            format!(
                                "project commit failed: {}; restoring the previous Ocean cache failed: {restore_error}",
                                error.message
                            ),
                        )
                    })?;
                }
                Err(error)
            }
        }
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

#[cfg(test)]
mod tests {
    use std::path::Path;

    use serde_json::{Value, json};

    use super::{
        default_modifier_bake_directory, escape_bake_name, fix_frame_range, node_bake_frames,
        whole_frame,
    };
    use crate::{
        cli::NodeBakeMode,
        error::{ErrorCode, PotError, Result},
        model::{Scene, SceneDoc},
    };

    fn scene_doc(edit: impl FnOnce(&mut Scene)) -> Result<SceneDoc> {
        let mut doc = SceneDoc::new(String::new());
        let active = doc.active_scene.clone();
        let scene = doc
            .scenes
            .get_mut(&active)
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "active scene is missing"))?;
        edit(scene);
        Ok(doc)
    }

    fn animation_frames(doc: &SceneDoc, config: Option<&Value>) -> Result<Vec<f64>> {
        let scene = doc
            .scenes
            .get(&doc.active_scene)
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "active scene is missing"))?;
        node_bake_frames(None, NodeBakeMode::Animation, scene, 1.0, config)
    }

    #[test]
    fn animation_range_ending_before_its_start_bakes_only_the_start_frame() -> Result<()> {
        assert_eq!(fix_frame_range(10, 3), (10, 10));
        assert_eq!(fix_frame_range(2, 5), (2, 5));
        let doc = scene_doc(|scene| {
            scene.frame_start = 10;
            scene.frame_end = 3;
        })?;
        assert_eq!(animation_frames(&doc, None)?, vec![10.0]);
        Ok(())
    }

    #[test]
    fn animation_range_prefers_bake_then_simulation_then_scene_range() -> Result<()> {
        let custom_bake = json!({
            "use_custom_simulation_frame_range": true,
            "frame_start": 2,
            "frame_end": 4
        });
        let doc = scene_doc(|scene| {
            scene.frame_start = 1;
            scene.frame_end = 10;
            scene.use_custom_simulation_range = true;
            scene.simulation_frame_start = 5;
            scene.simulation_frame_end = 7;
        })?;
        assert_eq!(
            animation_frames(&doc, Some(&custom_bake))?,
            vec![2.0, 3.0, 4.0]
        );
        assert_eq!(animation_frames(&doc, None)?, vec![5.0, 6.0, 7.0]);
        let doc = scene_doc(|scene| {
            scene.frame_start = 1;
            scene.frame_end = 3;
        })?;
        assert_eq!(animation_frames(&doc, None)?, vec![1.0, 2.0, 3.0]);
        Ok(())
    }

    #[test]
    fn default_bake_directory_uses_blender_escaped_names() {
        assert_eq!(escape_bake_name("Cube"), "Cube");
        assert_eq!(escape_bake_name("Cube.001"), "Cube46484849");
        assert_eq!(escape_bake_name("\u{e9}"), "-61-87");
        assert_eq!(
            default_modifier_bake_directory(
                Path::new("/projects/shot_a"),
                "Cube",
                "Geometry Nodes",
            ),
            Some("//blendcache_shot_a/Cube_Geometry32Nodes".to_owned())
        );
        assert_eq!(
            default_modifier_bake_directory(Path::new("/"), "Cube", "Nodes"),
            None
        );
    }

    #[test]
    fn animation_frame_bounds_must_be_whole_frames() {
        assert_eq!(whole_frame(4.0).ok(), Some(4));
        assert!(whole_frame(1.5).is_err());
        assert!(whole_frame(1e12).is_err());
    }
}

#[cfg(test)]
mod ocean_tests {
    use std::error::Error;

    use serde_json::Value;

    use super::write_ocean_exr;
    use crate::{
        geom::modifiers::OceanBakeSequence,
        model::{Id, Modifier, ModifierRuntime},
    };

    /// A Blender EXR lists the image's top row first, and the buffer's row 0 is the bottom
    /// (`openexr_api.cpp`, `ibuf->y - 1 - y`). The corpus holds Blender's bottom-up buffer, so a file
    /// Potter writes must hold the corpus rows in reverse: file row `r` is corpus row `n - 1 - r`.
    #[test]
    fn written_ocean_exr_rows_follow_blenders_top_down_layout() -> Result<(), Box<dyn Error>> {
        let corpus: Value = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/ocean_parity/corpus.json"
        ))?;
        let case = corpus["cases"]
            .as_array()
            .and_then(|cases| {
                cases
                    .iter()
                    .find(|case| case["name"] == "B1_cache_behaviour")
            })
            .ok_or("B1 corpus case is missing")?;
        let modifier = Modifier {
            id: Id::new("ocean_exr_layout")?,
            modifier_type: "ocean".to_owned(),
            name: "Ocean".to_owned(),
            enabled: true,
            show_viewport: true,
            show_render: true,
            params: serde_json::from_value(case["params"].clone())?,
            binding_data: None,
            baked_ocean_cache: None,
            runtime: ModifierRuntime::default(),
        };
        let time = case["params"]["time"]
            .as_f64()
            .ok_or("B1 time is missing")? as f32;
        let mut sequence = OceanBakeSequence::new(&modifier)?;
        let images = sequence.bake_frame(1, time)?;
        let disp = images
            .iter()
            .find(|image| image.filename == "disp_0001.exr")
            .ok_or("disp image is missing")?;
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("disp_0001.exr");
        write_ocean_exr(&path, disp)?;
        let bytes = std::fs::read(&path)?;
        let (width, height, file_pixels) = crate::image::storage::decode_ocean_cache_exr(&bytes)?;
        let n = disp.resolution;
        assert_eq!((width as usize, height as usize), (n, n));
        let expected = case["exr_files"]["disp_0001.exr"]["pixel_values_rgba_row_major"]
            .as_array()
            .ok_or("corpus disp pixels are missing")?;
        for row in 0..n {
            for x in 0..n {
                let corpus_row = n - 1 - row;
                for channel in 0..4 {
                    let want = expected[(corpus_row * n + x) * 4 + channel]
                        .as_f64()
                        .ok_or("corpus value is not numeric")?;
                    let got = file_pixels[row * n + x][channel];
                    // Half-float rounding of an f32 value: one half ULP is below 2e-3 of the value.
                    assert!(
                        (got - want).abs() <= 2.0e-3 * want.abs().max(6.2e-5),
                        "file row {row} column {x} channel {channel}: {got} vs Blender row {corpus_row} {want}"
                    );
                }
            }
        }
        Ok(())
    }
}
