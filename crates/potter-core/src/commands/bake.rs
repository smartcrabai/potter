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
    cli::{BakeArgs, BakeKind},
    error::{ErrorCode, PotError, Result},
    eval::{EvaluationContext, EvaluationPurpose, Snapshot},
    geom::Mesh,
    model::{Id, Modifier, OceanBakeCache},
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
        adaptive_render: None,
        purpose: EvaluationPurpose::Render,
    };
    let cache_directory = project.cache_directory();
    let mut stage = BakeStage::create(&args.out)?;
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
                    .filter(|&(id, _)| {
                        matches_target(id)
                            && doc.nodes.get(id).is_some_and(|node| {
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
    stage.publish(&args.out, args.overwrite)?;
    let scene = project.info()?;
    let warnings = with_load_warnings(&[], &project)?;
    Ok((
        Some(scene),
        json!({"kind":bake_kind_name(args.kind),"path":args.out,"files":files,"frame_count":baked_frames.len(),"cache_keys":cache_keys,"scene_modified":false,"warnings":warnings}),
    ))
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
    let mut project = Project::open_exclusive(args.scene)?;
    if args.out.exists() && !args.overwrite {
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
    let mut report_stage = BakeStage::create(&args.out)?;
    let project_destination = canonical_destination(project.path())?;
    let cache_destination = canonical_destination(&cache_path)?;
    let report_destination = canonical_destination(&args.out)?;
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
    report_stage.publish(&args.out, args.overwrite)?;
    let warnings = with_load_warnings(&[], &project)?;
    Ok((
        Some(scene),
        json!({
            "kind": "ocean",
            "path": args.out,
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
    }
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
