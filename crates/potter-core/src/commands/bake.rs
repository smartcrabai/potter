use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use glam::DMat4;
use serde_json::{Value, json};

use super::util::parse_id;
use crate::{
    cli::{BakeArgs, BakeKind},
    error::{ErrorCode, PotError, Result},
    eval::{EvaluationContext, Snapshot},
    geom::Mesh,
    model::Id,
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
    }
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
    };
    let cache_directory = project.path().to_path_buf();
    let mut stage = BakeStage::create(&args.out)?;
    let mut files = Vec::new();
    let mut baked_frames = Vec::new();
    for (frame_index, frame) in frames.into_iter().enumerate() {
        let mut frame_context = context.clone();
        frame_context.frame = Some(frame);
        let snapshot = Snapshot::evaluate_with_cache(doc, &frame_context, Some(&cache_directory))?;
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
        }
    }
    let cache_keys = baked_frames
        .iter()
        .filter_map(|record| record["cache_key"].as_str().map(str::to_owned))
        .collect::<Vec<_>>();
    write_json(
        &stage.path.join("manifest.json"),
        &json!({"schema_version":1,"kind":match args.kind {BakeKind::Animation=>"animation",BakeKind::Geometry=>"geometry",BakeKind::Simulation=>"simulation",BakeKind::Texture=>"texture"},"scene_id":doc.scene_id,"revision":doc.revision,"frames":baked_frames,"cache_keys":cache_keys,"files":files}),
    )?;
    files.push("manifest.json".to_owned());
    stage.publish(&args.out, args.overwrite)?;
    let scene = project.info()?;
    Ok((
        Some(scene),
        json!({"kind":match args.kind {BakeKind::Animation=>"animation",BakeKind::Geometry=>"geometry",BakeKind::Simulation=>"simulation",BakeKind::Texture=>"texture"},"path":args.out,"files":files,"frame_count":baked_frames.len(),"cache_keys":cache_keys,"scene_modified":false}),
    ))
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
