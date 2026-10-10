use std::{
    collections::BTreeMap,
    fs,
    path::{Component, Path, PathBuf},
};

use glam::{DMat4, DVec3};
use serde_json::{Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    eval::{EvaluationContext, Snapshot},
    hash,
    model::{CameraProjection, Id, SceneDoc},
};

use super::{
    raster::{
        Camera, DepthOfField, Mode, Projection, RasterOutput, StereoMode, rasterize_with_lines,
    },
    scene::{Geometry, extract_geometry},
};

const MIN_SIZE: u32 = 64;
const MAX_SIZE: u32 = 16_384;
pub(super) const ENGINE: &str = "potter-cpu-path-1";
const VALID_VIEWS: [&str; 7] = ["front", "back", "right", "left", "top", "bottom", "iso"];

#[derive(Clone, Debug)]
pub struct PreviewRequest<'a> {
    pub views: &'a [String],
    pub camera_id: Option<&'a str>,
    pub mode: &'a str,
    pub size: u32,
    pub out: Option<&'a Path>,
    pub overwrite: bool,
    pub context: &'a EvaluationContext,
    pub render_visibility: bool,
    pub staged_assets: Option<&'a BTreeMap<String, Vec<u8>>>,
}

#[derive(Clone, Debug)]
struct PreparedCamera {
    camera: Camera,
    projection_name: &'static str,
    matrix: [f64; 16],
    ortho_height: Option<f64>,
    lens_mm: Option<f64>,
    sensor_width_mm: Option<f64>,
}

struct RenderFiles {
    image: Vec<u8>,
    ids: Vec<u8>,
    depth: Vec<u8>,
    elements: Vec<u8>,
}

enum PendingView {
    Reused(Value),
    Staged(StagedView),
}

struct StagedView {
    destination: PathBuf,
    temp_dir: PathBuf,
    view: String,
    manifest: Value,
}

impl Drop for StagedView {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.temp_dir);
    }
}

pub fn render_previews(
    doc: &SceneDoc,
    scene_root: &Path,
    request: &PreviewRequest<'_>,
) -> Result<Vec<Value>> {
    if !(MIN_SIZE..=MAX_SIZE).contains(&request.size) {
        return Err(PotError::with_details(
            ErrorCode::InvalidArgument,
            "preview size must be between 64 and 16384 pixels",
            json!({ "size": request.size, "minimum": MIN_SIZE, "maximum": MAX_SIZE }),
        ));
    }
    let mode = parse_mode(request.mode)?;
    let labels = resolve_views(request.views, request.camera_id)?;
    let (snapshot, node_errors) =
        Snapshot::evaluate_available_with_cache(doc, request.context, Some(scene_root))?;
    let images = if matches!(mode, Mode::Beauty) {
        super::path::load_images(doc, scene_root, request.staged_assets)?
    } else {
        BTreeMap::new()
    };
    let mut geometry = extract_geometry(
        doc,
        &snapshot,
        mode,
        request.render_visibility,
        matches!(mode, Mode::Beauty),
        &images,
    )?;
    for (node_id, error) in &node_errors {
        geometry.warnings.push(json!({
            "code": "UNSUPPORTED_FEATURE",
            "feature_id": error.details.get("feature_id"),
            "node_id": node_id,
            "message": error.message,
        }));
    }
    let mut pending = Vec::with_capacity(labels.len());
    for label in labels {
        let prepared = if let Some(camera_id) = request.camera_id {
            stored_camera(doc, &snapshot, camera_id)?
        } else {
            preset_camera(&geometry, &label)?
        };
        let background = background_color(doc, &snapshot, mode, false);
        let scene = doc.scenes.get(&snapshot.scene_id).ok_or_else(|| {
            PotError::new(ErrorCode::TargetNotFound, "selected scene does not exist")
        })?;
        let mut raster = if matches!(mode, Mode::Beauty) {
            super::path::render(
                doc,
                &snapshot,
                &geometry,
                &prepared.camera,
                request.size,
                request.size,
                scene.render.samples,
                scene.render.seed,
                scene.render.max_bounces,
                false,
                &images,
            )?
        } else {
            rasterize_with_lines(
                &geometry.triangles,
                &geometry.lines,
                &prepared.camera,
                request.size,
                request.size,
                mode,
                background,
                1,
                0,
            )?
        };
        if matches!(mode, Mode::Beauty) {
            crate::compositor::apply(
                &mut raster,
                doc,
                scene,
                request.size,
                request.size,
                request.context.frame.unwrap_or(scene.frame_current),
            )?;
        }
        let (manifest, files) = manifest_value(
            doc,
            &snapshot,
            &geometry,
            &prepared,
            &label,
            mode,
            request.render_visibility,
            request.size,
            background,
            &raster,
        )?;
        let render_key = manifest
            .get("render_key")
            .and_then(Value::as_str)
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "render key is missing"))?;
        let destination = if let Some(out) = request.out {
            out.to_path_buf()
        } else {
            let key = render_key.strip_prefix("sha256:").ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "render key has an invalid prefix")
            })?;
            scene_root
                .join(".potter/previews")
                .join(format!("r{}", snapshot.revision))
                .join(key)
        };
        pending.push(stage_view(
            &destination,
            &label,
            request.size,
            manifest,
            &files,
            request.overwrite,
        )?);
    }
    let mut results = Vec::with_capacity(pending.len());
    for view in pending {
        results.push(match view {
            PendingView::Reused(result) => result,
            PendingView::Staged(staged) => publish_staged(&staged)?,
        });
    }
    Ok(results)
}

#[expect(
    clippy::too_many_arguments,
    reason = "render frame passes explicit engine, camera, and sequence settings"
)]
pub(crate) fn render_camera_frame(
    doc: &SceneDoc,
    context: &EvaluationContext,
    cache_directory: &Path,
    camera_id: &str,
    width: u32,
    height: u32,
    mode_name: &str,
    engine: &str,
    render_visibility: bool,
    film_transparent: bool,
    samples: u32,
    seed: u32,
) -> Result<(Snapshot, RasterOutput)> {
    if width == 0 || height == 0 || width > MAX_SIZE || height > MAX_SIZE {
        return Err(PotError::invalid_argument(
            "render dimensions must be between 1 and 16384 pixels",
        ));
    }
    let mode = parse_mode(mode_name)?;
    let snapshot = Snapshot::evaluate_with_cache(doc, context, Some(cache_directory))?;
    let images = super::path::load_images(doc, cache_directory, None)?;
    let geometry = extract_geometry(
        doc,
        &snapshot,
        mode,
        render_visibility,
        engine == "path" && matches!(mode, Mode::Beauty),
        &images,
    )?;
    let camera = stored_camera(doc, &snapshot, camera_id)?;
    let scene = doc
        .scenes
        .get(&snapshot.scene_id)
        .ok_or_else(|| PotError::new(ErrorCode::TargetNotFound, "selected scene does not exist"))?;
    let background = background_color(doc, &snapshot, mode, film_transparent);
    let mut raster = if matches!(mode, Mode::Beauty) && engine == "path" {
        super::path::render(
            doc,
            &snapshot,
            &geometry,
            &camera.camera,
            width,
            height,
            samples,
            seed,
            scene.render.max_bounces,
            film_transparent,
            &images,
        )?
    } else {
        rasterize_with_lines(
            &geometry.triangles,
            &geometry.lines,
            &camera.camera,
            width,
            height,
            mode,
            background,
            samples,
            seed,
        )?
    };
    if engine == "realtime" && matches!(mode, Mode::Beauty) {
        super::path::composite_realtime_volumes(
            doc,
            &snapshot,
            &camera.camera,
            width,
            height,
            seed,
            film_transparent,
            &images,
            &mut raster,
        )?;
    }
    crate::compositor::apply(
        &mut raster,
        doc,
        scene,
        width,
        height,
        context.frame.unwrap_or(scene.frame_current),
    )?;
    Ok((snapshot, raster))
}

fn parse_mode(value: &str) -> Result<Mode> {
    match value {
        "solid" => Ok(Mode::Solid),
        "beauty" => Ok(Mode::Beauty),
        "wire" => Ok(Mode::Wire),
        "normal" => Ok(Mode::Normal),
        "depth" => Ok(Mode::Depth),
        "id" => Ok(Mode::Id),
        _ => Err(PotError::invalid_argument(format!(
            "unknown preview mode: {value}"
        ))),
    }
}

fn resolve_views(views: &[String], camera_id: Option<&str>) -> Result<Vec<String>> {
    if let Some(camera_id) = camera_id {
        if !views.is_empty() {
            return Err(PotError::invalid_argument(
                "--camera cannot be combined with --views",
            ));
        }
        Id::new(camera_id.to_owned())?;
        return Ok(vec![camera_id.to_owned()]);
    }
    if views.is_empty() {
        return Ok(vec!["iso".to_owned()]);
    }
    let mut seen = std::collections::BTreeSet::new();
    for view in views {
        if view.is_empty() || !VALID_VIEWS.contains(&view.as_str()) || !seen.insert(view) {
            return Err(PotError::with_details(
                ErrorCode::InvalidArgument,
                "views must be a comma-separated list of unique known view names",
                json!({ "view": view }),
            ));
        }
    }
    Ok(views.to_vec())
}

fn preset_camera(geometry: &Geometry, view: &str) -> Result<PreparedCamera> {
    let target = geometry
        .bounds
        .map_or(DVec3::ZERO, |(min, max)| (min + max) * 0.5);
    let direction = match view {
        "front" => DVec3::new(0.0, -1.0, 0.0),
        "back" => DVec3::new(0.0, 1.0, 0.0),
        "right" => DVec3::new(1.0, 0.0, 0.0),
        "left" => DVec3::new(-1.0, 0.0, 0.0),
        "top" => DVec3::Z,
        "bottom" => -DVec3::Z,
        "iso" => DVec3::new(1.0, -1.0, 1.0).normalize(),
        _ => return Err(PotError::invalid_argument("unknown camera preset")),
    };
    let up_hint = if matches!(view, "top" | "bottom") {
        DVec3::Y
    } else {
        DVec3::Z
    };
    let forward = -direction;
    let right = forward.cross(up_hint).normalize_or_zero();
    let up = right.cross(forward).normalize_or_zero();
    let projected_height = if let Some((min, max)) = geometry.bounds {
        let mut projected_min = DVec3::splat(f64::INFINITY);
        let mut projected_max = DVec3::splat(f64::NEG_INFINITY);
        for x in [min.x, max.x] {
            for y in [min.y, max.y] {
                for z in [min.z, max.z] {
                    let offset = DVec3::new(x, y, z) - target;
                    let point = DVec3::new(offset.dot(right), offset.dot(up), offset.dot(forward));
                    projected_min = projected_min.min(point);
                    projected_max = projected_max.max(point);
                }
            }
        }
        (projected_max.y - projected_min.y).max(projected_max.x - projected_min.x)
    } else {
        1.0
    };
    let ortho_height = (projected_height * 1.1).max(1.0e-6);
    let distance = (ortho_height * 2.0).max(1.0);
    let position = target + direction * distance;
    let far = (distance + ortho_height * 8.0).max(100.0);
    let camera = Camera {
        position,
        target,
        up: up_hint,
        near: 0.001,
        far,
        projection: Projection::Orthographic {
            height: ortho_height,
        },
        shift: [0.0; 2],
        depth_of_field: None,
        stereo_mode: StereoMode::None,
        interocular_distance: 0.065,
    };
    let matrix = camera_matrix(position, target, up_hint);
    Ok(PreparedCamera {
        camera,
        projection_name: "orthographic",
        matrix,
        ortho_height: Some(ortho_height),
        lens_mm: None,
        sensor_width_mm: None,
    })
}

fn stored_camera(doc: &SceneDoc, snapshot: &Snapshot, camera_id: &str) -> Result<PreparedCamera> {
    let id = Id::new(camera_id.to_owned())?;
    let node = doc.nodes.get(&id).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::TargetNotFound,
            "camera node does not exist",
            json!({ "camera": camera_id }),
        )
    })?;
    let camera_data = node
        .data
        .as_ref()
        .and_then(|data_id| doc.data_blocks.get(data_id))
        .and_then(|block| block.camera.as_ref())
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::TargetNotFound,
                "node has no camera data",
                json!({ "camera": camera_id }),
            )
        })?;
    let evaluated = snapshot.nodes.get(&id).ok_or_else(|| {
        PotError::new(
            ErrorCode::EvaluationFailed,
            "camera is missing from evaluation",
        )
    })?;
    let matrix = DMat4::from_cols_array(&evaluated.world_matrix);
    let position = matrix.transform_point3(DVec3::ZERO);
    let forward = matrix.transform_vector3(-DVec3::Z).normalize_or_zero();
    let up = matrix.transform_vector3(DVec3::Y).normalize_or_zero();
    if forward.length_squared() <= f64::EPSILON || up.length_squared() <= f64::EPSILON {
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            "camera transform is degenerate",
        ));
    }
    let target = position + forward;
    let projection = match camera_data.projection {
        CameraProjection::Orthographic => Projection::Orthographic {
            height: camera_data.ortho_scale,
        },
        CameraProjection::Perspective => Projection::Perspective {
            lens_mm: camera_data.lens_mm,
            sensor_width_mm: camera_data.sensor_width_mm,
        },
        CameraProjection::Panorama => Projection::Panorama {
            fisheye: camera_data.panorama_type == "fisheye_equidistant",
        },
        CameraProjection::Fisheye => Projection::Panorama { fisheye: true },
    };
    let projection_name = match camera_data.projection {
        CameraProjection::Orthographic => "orthographic",
        CameraProjection::Perspective => "perspective",
        CameraProjection::Panorama => "panorama",
        CameraProjection::Fisheye => "fisheye",
    };
    let depth_of_field = camera_data.dof_enabled.then_some(DepthOfField {
        focus_distance: camera_data.focus_distance,
        aperture_radius: camera_data.lens_mm / (2000.0 * camera_data.f_stop),
        aperture_blades: camera_data.aperture_blades,
    });
    let stereo_mode = match camera_data.stereo_mode.as_str() {
        "side_by_side" => StereoMode::SideBySide,
        "anaglyph" => StereoMode::Anaglyph,
        _ => StereoMode::None,
    };
    let camera = Camera {
        position,
        target,
        up,
        near: camera_data.clip_start,
        far: camera_data.clip_end,
        projection,
        shift: camera_data.shift,
        depth_of_field,
        stereo_mode,
        interocular_distance: camera_data.interocular_distance,
    };
    Ok(PreparedCamera {
        camera,
        projection_name,
        matrix: evaluated.world_matrix,
        ortho_height: (camera_data.projection == CameraProjection::Orthographic)
            .then_some(camera_data.ortho_scale),
        lens_mm: (camera_data.projection == CameraProjection::Perspective)
            .then_some(camera_data.lens_mm),
        sensor_width_mm: (camera_data.projection == CameraProjection::Perspective)
            .then_some(camera_data.sensor_width_mm),
    })
}

fn camera_matrix(position: DVec3, target: DVec3, up_hint: DVec3) -> [f64; 16] {
    let forward = (target - position).normalize_or_zero();
    let right = forward.cross(up_hint).normalize_or_zero();
    let up = right.cross(forward).normalize_or_zero();
    DMat4::from_cols(
        right.extend(0.0),
        up.extend(0.0),
        (-forward).extend(0.0),
        position.extend(1.0),
    )
    .to_cols_array()
}

fn background_color(doc: &SceneDoc, snapshot: &Snapshot, mode: Mode, transparent: bool) -> [u8; 4] {
    let alpha = if transparent { 0 } else { 255 };
    if matches!(mode, Mode::Solid) {
        return [242, 242, 242, alpha];
    }
    let scene = doc.scenes.get(&snapshot.scene_id);
    let world = scene
        .and_then(|value| value.world.as_ref())
        .and_then(|id| doc.worlds.get(id));
    world.map_or([13, 13, 13, alpha], |value| {
        [
            linear_to_byte(value.color[0] * value.strength),
            linear_to_byte(value.color[1] * value.strength),
            linear_to_byte(value.color[2] * value.strength),
            alpha,
        ]
    })
}

#[expect(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    reason = "value is clamped to the 8-bit sRGB range"
)]
fn linear_to_byte(value: f64) -> u8 {
    let value = value.clamp(0.0, 1.0);
    let srgb = if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0).round() as u8
}

#[expect(
    clippy::too_many_arguments,
    reason = "manifest construction includes every value in the render-key contract"
)]
fn manifest_value(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    geometry: &Geometry,
    prepared: &PreparedCamera,
    view: &str,
    mode: Mode,
    render_visibility: bool,
    size: u32,
    background: [u8; 4],
    raster: &RasterOutput,
) -> Result<(Value, RenderFiles)> {
    let scene = doc
        .scenes
        .get(&snapshot.scene_id)
        .ok_or_else(|| PotError::new(ErrorCode::TargetNotFound, "selected scene does not exist"))?;
    let mode_name = match mode {
        Mode::Solid => "solid",
        Mode::Beauty => "beauty",
        Mode::Wire => "wire",
        Mode::Normal => "normal",
        Mode::Depth => "depth",
        Mode::Id => "id",
    };
    let camera = json!({
        "projection": prepared.projection_name,
        "position": prepared.camera.position.to_array(),
        "target": prepared.camera.target.to_array(),
        "up": prepared.camera.up.to_array(),
        "near": prepared.camera.near,
        "far": prepared.camera.far,
        "matrix": prepared.matrix,
        "ortho_height": prepared.ortho_height,
        "lens_mm": prepared.lens_mm,
        "sensor_width_mm": prepared.sensor_width_mm,
        "shift": prepared.camera.shift,
        "panorama_type": match prepared.camera.projection {
            Projection::Panorama { fisheye: true } => "fisheye_equidistant",
            Projection::Panorama { fisheye: false } => "equirectangular",
            _ => Value::Null.as_str().unwrap_or(""),
        },
        "depth_of_field": prepared.camera.depth_of_field.map(|dof| json!({
            "focus_distance": dof.focus_distance,
            "aperture_radius": dof.aperture_radius,
            "aperture_blades": dof.aperture_blades,
        })),
        "stereo_mode": match prepared.camera.stereo_mode {
            StereoMode::None => "none",
            StereoMode::SideBySide => "side_by_side",
            StereoMode::Anaglyph => "anaglyph",
        },
        "interocular_distance": prepared.camera.interocular_distance,
    });
    let has_world = scene
        .world
        .as_ref()
        .is_some_and(|world_id| doc.worlds.contains_key(world_id));
    let has_lights = doc.nodes.values().any(|node| {
        node.data
            .as_ref()
            .and_then(|id| doc.data_blocks.get(id))
            .is_some_and(|block| block.light.is_some())
    });
    let ambient = matches!(mode, Mode::Beauty) && !has_world && !has_lights;
    let settings = json!({
        "aa": "none",
        "samples": 1,
        "seed": 0,
        "mode": mode_name,
        "background": background,
        "visibility": if render_visibility { "render" } else { "viewport" },
        "transparent": false,
        "ambient": if ambient { json!({ "color": [0.08, 0.08, 0.08], "source": "default_no_lights_or_world" }) } else { Value::Null },
        "engine": "cpu",
        "device": "cpu",
        "view_transform": "sRGB standard",
        "ocio": Value::Null,
    });
    let pick_policy = json!({
        "sample": "pixel_center",
        "surface": "nearest_selectable_surface",
        "depth_tolerance_m": "max(1e-6, abs(depth)*1e-6)",
        "tie_break": ["node_id", "instance_path", "element_id"],
    });
    let units = serde_json::to_value(&scene.unit).map_err(|error| {
        PotError::new(
            ErrorCode::InternalError,
            format!("unit serialization failed: {error}"),
        )
    })?;
    let profile = serde_json::to_value(&doc.profile).map_err(|error| {
        PotError::new(
            ErrorCode::InternalError,
            format!("profile serialization failed: {error}"),
        )
    })?;
    let mut core = json!({
        "scene_id": snapshot.scene_id,
        "revision": snapshot.revision,
        "scene_hash": snapshot.scene_hash,
        "evaluation_hash": snapshot.evaluation_hash,
        "engine": ENGINE,
        "profile": profile,
        "frame": snapshot.frame,
        "view_layer": snapshot.view_layer,
        "view": view,
        "camera": camera,
        "units": units,
        "settings": settings,
        "pick_policy": pick_policy,
        "objects": geometry.objects,
        "elements": geometry.elements,
        "width": size,
        "height": size,
    });
    if !geometry.warnings.is_empty() {
        core["warnings"] = json!(geometry.warnings);
    }
    let render_key = hash::sha256(&hash::canonicalize(&core)?);
    let image_bytes = encode_png(&raster.rgba, size, size)?;
    let ids_bytes = encode_u32(&raster.ids);
    let depth_bytes = encode_f32(&raster.depths);
    let element_bytes = encode_u32(&raster.elements);
    let mut manifest = json!({
        "scene_id": snapshot.scene_id,
        "revision": snapshot.revision,
        "scene_hash": snapshot.scene_hash,
        "evaluation_hash": snapshot.evaluation_hash,
        "engine": ENGINE,
        "profile": profile,
        "frame": snapshot.frame,
        "view_layer": snapshot.view_layer,
        "view": view,
        "camera": camera,
        "units": units,
        "settings": settings,
        "pick_policy": pick_policy,
        "image": {
            "path": format!("{view}.png"),
            "width": size,
            "height": size,
            "format": "png",
            "hash": hash::sha256(&image_bytes),
        },
        "buffers": {
            "ids": {
                "path": format!("{view}.ids.bin"),
                "type": "u32_le",
                "width": size,
                "height": size,
                "hash": hash::sha256(&ids_bytes),
            },
            "depth": {
                "path": format!("{view}.depth.bin"),
                "type": "f32_le",
                "width": size,
                "height": size,
                "hash": hash::sha256(&depth_bytes),
            },
            "elements": {
                "path": format!("{view}.elements.bin"),
                "type": "u32_le",
                "width": size,
                "height": size,
                "hash": hash::sha256(&element_bytes),
            },
        },
        "objects": geometry.objects,
        "elements": geometry.elements,
        "render_key": render_key,
        "width": size,
        "height": size,
    });
    if !geometry.warnings.is_empty() {
        manifest["warnings"] = json!(geometry.warnings);
    }
    Ok((
        manifest,
        RenderFiles {
            image: image_bytes,
            ids: ids_bytes,
            depth: depth_bytes,
            elements: element_bytes,
        },
    ))
}

pub(crate) fn encode_png(rgba: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().map_err(|error| {
            PotError::new(
                ErrorCode::RenderFailed,
                format!("PNG header failed: {error}"),
            )
        })?;
        writer.write_image_data(rgba).map_err(|error| {
            PotError::new(
                ErrorCode::RenderFailed,
                format!("PNG encoding failed: {error}"),
            )
        })?;
    }
    Ok(bytes)
}

fn encode_u32(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len().saturating_mul(4));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn encode_f32(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len().saturating_mul(4));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn stage_view(
    destination: &Path,
    view: &str,
    size: u32,
    manifest: Value,
    files: &RenderFiles,
    overwrite: bool,
) -> Result<PendingView> {
    let image_name = format!("{view}.png");
    let ids_name = format!("{view}.ids.bin");
    let depth_name = format!("{view}.depth.bin");
    let elements_name = format!("{view}.elements.bin");
    let manifest_name = format!("{view}.manifest.json");
    let manifest_path = destination.join(&manifest_name);
    let expected_key = manifest
        .get("render_key")
        .and_then(Value::as_str)
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "render key is missing"))?;
    if destination.exists() && !overwrite {
        if manifest_path.is_file() {
            let bytes = fs::read(&manifest_path).map_err(|error| PotError::io(&error))?;
            if let Ok(existing) = serde_json::from_slice::<Value>(&bytes)
                && existing.get("render_key").and_then(Value::as_str) == Some(expected_key)
                && existing_set_is_intact(destination, &existing, size)?
            {
                return Ok(PendingView::Reused(preview_result(
                    destination,
                    view,
                    &existing,
                    true,
                )?));
            }
        }
        let has_existing = [
            &image_name,
            &ids_name,
            &depth_name,
            &elements_name,
            &manifest_name,
        ]
        .iter()
        .any(|name| destination.join(name).exists());
        if has_existing {
            return Err(PotError::with_details(
                ErrorCode::OutputExists,
                "a different or incomplete render set already exists",
                json!({ "directory": destination, "view": view }),
            ));
        }
    }
    fs::create_dir_all(destination).map_err(|error| PotError::io(&error))?;
    let temp_dir = destination.join(format!(".potter-render-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&temp_dir).map_err(|error| PotError::io(&error))?;
    let stage_result = (|| -> Result<()> {
        fs::write(temp_dir.join(&image_name), &files.image)
            .map_err(|error| PotError::io(&error))?;
        fs::write(temp_dir.join(&ids_name), &files.ids).map_err(|error| PotError::io(&error))?;
        fs::write(temp_dir.join(&depth_name), &files.depth)
            .map_err(|error| PotError::io(&error))?;
        fs::write(temp_dir.join(&elements_name), &files.elements)
            .map_err(|error| PotError::io(&error))?;
        let manifest_bytes = serde_json::to_vec_pretty(&manifest).map_err(|error| {
            PotError::new(
                ErrorCode::InternalError,
                format!("manifest serialization failed: {error}"),
            )
        })?;
        fs::write(temp_dir.join(&manifest_name), manifest_bytes)
            .map_err(|error| PotError::io(&error))?;
        Ok(())
    })();
    if let Err(error) = stage_result {
        let _ = fs::remove_dir_all(&temp_dir);
        return Err(error);
    }
    Ok(PendingView::Staged(StagedView {
        destination: destination.to_path_buf(),
        temp_dir,
        view: view.to_owned(),
        manifest,
    }))
}

fn publish_staged(staged: &StagedView) -> Result<Value> {
    let view = staged.view.as_str();
    let manifest_name = format!("{view}.manifest.json");
    for name in [
        format!("{view}.png"),
        format!("{view}.ids.bin"),
        format!("{view}.depth.bin"),
        format!("{view}.elements.bin"),
    ] {
        fs::rename(staged.temp_dir.join(&name), staged.destination.join(name))
            .map_err(|error| PotError::io(&error))?;
    }
    fs::rename(
        staged.temp_dir.join(manifest_name),
        staged.destination.join(format!("{view}.manifest.json")),
    )
    .map_err(|error| PotError::io(&error))?;
    preview_result(&staged.destination, view, &staged.manifest, false)
}

fn preview_result(directory: &Path, view: &str, manifest: &Value, reused: bool) -> Result<Value> {
    let manifest_path = directory.join(format!("{view}.manifest.json"));
    let absolute_manifest = manifest_path
        .canonicalize()
        .map_err(|error| PotError::io(&error))?;
    let manifest_bytes = fs::read(&absolute_manifest).map_err(|error| PotError::io(&error))?;
    let manifest_hash = hash::sha256(&manifest_bytes);
    let image = manifest
        .get("image")
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "image metadata is missing"))?;
    let image_path = absolute_asset_path(directory, image)?;
    let image_hash = image
        .get("hash")
        .and_then(Value::as_str)
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "image hash is missing"))?;
    let mut generated_files = vec![json!({
        "kind": "image",
        "path": image_path.display().to_string(),
        "hash": image_hash,
    })];
    let buffers = manifest
        .get("buffers")
        .and_then(Value::as_object)
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "buffer metadata is missing"))?;
    for name in ["ids", "depth", "elements"] {
        let asset = buffers
            .get(name)
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "buffer metadata is missing"))?;
        let path = absolute_asset_path(directory, asset)?;
        let asset_hash = asset
            .get("hash")
            .and_then(Value::as_str)
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "buffer hash is missing"))?;
        generated_files.push(json!({
            "kind": name,
            "path": path.display().to_string(),
            "hash": asset_hash,
        }));
    }
    generated_files.push(json!({
        "kind": "manifest",
        "path": absolute_manifest.display().to_string(),
        "hash": manifest_hash,
    }));
    Ok(json!({
        "view": view,
        "scene_id": manifest.get("scene_id"),
        "revision": manifest.get("revision"),
        "scene_hash": manifest.get("scene_hash"),
        "evaluation_hash": manifest.get("evaluation_hash"),
        "frame": manifest.get("frame"),
        "view_layer": manifest.get("view_layer"),
        "manifest": absolute_manifest.display().to_string(),
        "image": image_path.display().to_string(),
        "width": image.get("width"),
        "height": image.get("height"),
        "render_key": manifest.get("render_key"),
        "warnings": manifest.get("warnings").cloned().unwrap_or_else(|| json!([])),
        "files": generated_files,
        "reused": reused,
    }))
}

fn absolute_asset_path(directory: &Path, asset: &Value) -> Result<PathBuf> {
    let relative = asset
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "asset path is missing"))?;
    let relative_path = Path::new(relative);
    if !safe_relative_path(relative_path) {
        return Err(PotError::new(
            ErrorCode::RenderInvalid,
            "manifest asset path is unsafe",
        ));
    }
    directory
        .join(relative_path)
        .canonicalize()
        .map_err(|error| PotError::io(&error))
}

fn existing_set_is_intact(directory: &Path, manifest: &Value, size: u32) -> Result<bool> {
    let expected_length = u64::from(size)
        .checked_mul(u64::from(size))
        .and_then(|pixels| pixels.checked_mul(4))
        .and_then(|length| usize::try_from(length).ok())
        .ok_or_else(|| {
            PotError::new(ErrorCode::LimitExceeded, "render dimensions are too large")
        })?;
    let Some(image) = manifest.get("image") else {
        return Ok(false);
    };
    if image.get("width").and_then(Value::as_u64) != Some(u64::from(size))
        || image.get("height").and_then(Value::as_u64) != Some(u64::from(size))
        || image.get("format").and_then(Value::as_str) != Some("png")
        || !asset_matches(directory, image, None)?
    {
        return Ok(false);
    }
    let Some(buffers) = manifest.get("buffers").and_then(Value::as_object) else {
        return Ok(false);
    };
    for (name, expected_type) in [
        ("ids", "u32_le"),
        ("depth", "f32_le"),
        ("elements", "u32_le"),
    ] {
        let Some(asset) = buffers.get(name) else {
            return Ok(false);
        };
        if asset.get("width").and_then(Value::as_u64) != Some(u64::from(size))
            || asset.get("height").and_then(Value::as_u64) != Some(u64::from(size))
            || asset.get("type").and_then(Value::as_str) != Some(expected_type)
            || !asset_matches(directory, asset, Some(expected_length))?
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn asset_matches(directory: &Path, asset: &Value, expected_length: Option<usize>) -> Result<bool> {
    let Some(relative) = asset.get("path").and_then(Value::as_str) else {
        return Ok(false);
    };
    let Some(expected_hash) = asset.get("hash").and_then(Value::as_str) else {
        return Ok(false);
    };
    let relative = Path::new(relative);
    if !safe_relative_path(relative) {
        return Ok(false);
    }
    let path = directory.join(relative);
    if !path.is_file() {
        return Ok(false);
    }
    let canonical_root = directory
        .canonicalize()
        .map_err(|error| PotError::io(&error))?;
    let canonical_path = path.canonicalize().map_err(|error| PotError::io(&error))?;
    if !canonical_path.starts_with(canonical_root) {
        return Ok(false);
    }
    let bytes = fs::read(canonical_path).map_err(|error| PotError::io(&error))?;
    Ok(expected_length.is_none_or(|length| bytes.len() == length)
        && hash::sha256(&bytes) == expected_hash)
}

fn safe_relative_path(path: &Path) -> bool {
    !path.as_os_str().is_empty()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "render validation unit tests")]

    use super::resolve_views;

    #[test]
    fn rejects_duplicate_or_unknown_views() {
        assert!(resolve_views(&["top".to_owned(), "top".to_owned()], None).is_err());
        assert!(resolve_views(&["unknown".to_owned()], None).is_err());
    }

    #[test]
    fn camera_view_is_used_without_preset_names() {
        assert_eq!(
            resolve_views(&[], Some("camera_main")).unwrap(),
            ["camera_main"]
        );
    }
}
