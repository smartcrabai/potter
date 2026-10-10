use std::{
    collections::BTreeMap,
    fs,
    io::Cursor,
    path::{Component, Path, PathBuf},
};

use glam::{DMat4, DVec3};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    cli::PickDomain,
    error::{ErrorCode, PotError, Result},
    eval::{EvaluationContext, Snapshot},
    hash,
    model::{Id, SceneDoc},
};

use super::{
    preview::ENGINE,
    raster::{Camera, DepthOfField, Line, Mode, Projection, StereoMode, Triangle, ray_for_pixel},
    scene::{ElementRecord, Geometry, ObjectRecord, extract_geometry},
};

#[derive(Debug, Deserialize)]
struct Manifest {
    scene_id: String,
    revision: u64,
    scene_hash: String,
    evaluation_hash: String,
    engine: String,
    profile: Value,
    units: Value,
    frame: f64,
    view_layer: Option<Id>,
    view: String,
    camera: CameraManifest,
    image: AssetManifest,
    buffers: BTreeMap<String, AssetManifest>,
    objects: Vec<ObjectRecord>,
    elements: Vec<ElementRecord>,
    settings: Value,
    pick_policy: Value,
    #[serde(default)]
    warnings: Option<Vec<Value>>,
    render_key: String,
    width: u32,
    height: u32,
}

#[derive(Debug, Deserialize, serde::Serialize)]
struct CameraManifest {
    projection: String,
    position: [f64; 3],
    target: [f64; 3],
    up: [f64; 3],
    near: f64,
    far: f64,
    matrix: [f64; 16],
    ortho_height: Option<f64>,
    lens_mm: Option<f64>,
    sensor_width_mm: Option<f64>,
    #[serde(default)]
    shift: [f64; 2],
    #[serde(default)]
    panorama_type: String,
    #[serde(default)]
    depth_of_field: Option<DepthOfFieldManifest>,
    #[serde(default)]
    stereo_mode: String,
    #[serde(default)]
    interocular_distance: f64,
}

#[derive(Debug, Deserialize, serde::Serialize)]
struct DepthOfFieldManifest {
    focus_distance: f64,
    aperture_radius: f64,
    aperture_blades: u32,
}

#[derive(Debug, Deserialize)]
struct AssetManifest {
    path: String,
    width: u32,
    height: u32,
    hash: String,
    format: Option<String>,
    #[serde(rename = "type")]
    buffer_type: Option<String>,
}

struct VerifiedBuffers {
    width: u32,
    height: u32,
    ids: Vec<u32>,
    depth: Vec<f32>,
    elements: Vec<u32>,
}

#[derive(Clone, Copy)]
struct Hit {
    depth: f64,
    position: DVec3,
    normal: Option<DVec3>,
    evaluation_index: usize,
}

pub fn pick(
    doc: &SceneDoc,
    cache_directory: &Path,
    manifest_path: &Path,
    pixel: (u32, u32),
    domain: PickDomain,
) -> Result<Value> {
    let manifest_bytes = fs::read(manifest_path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            PotError::with_details(
                ErrorCode::FileNotFound,
                "render manifest was not found",
                json!({ "path": manifest_path }),
            )
        } else {
            PotError::io(&error)
        }
    })?;
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes).map_err(|error| {
        PotError::with_details(
            ErrorCode::RenderInvalid,
            format!("render manifest is invalid: {error}"),
            json!({ "path": manifest_path }),
        )
    })?;
    let scene_id = Id::new(manifest.scene_id.clone()).map_err(|error| {
        PotError::with_details(
            ErrorCode::RenderInvalid,
            error.message,
            json!({ "scene_id": manifest.scene_id }),
        )
    })?;
    if !manifest.frame.is_finite()
        || !valid_camera(&manifest.camera)
        || manifest.width != manifest.image.width
        || manifest.height != manifest.image.height
    {
        return Err(render_invalid(
            "render manifest camera or dimensions are invalid",
        ));
    }
    verify_render_key(&manifest)?;
    if manifest.engine != ENGINE {
        return Err(PotError::new(
            ErrorCode::StaleRender,
            "render engine version is stale",
        ));
    }
    let profile = serde_json::to_value(&doc.profile).map_err(|error| {
        PotError::new(
            ErrorCode::InternalError,
            format!("profile serialization failed: {error}"),
        )
    })?;
    if manifest.profile != profile {
        return Err(PotError::new(
            ErrorCode::StaleRender,
            "render profile does not match scene",
        ));
    }
    let context = EvaluationContext {
        scene_id: Some(scene_id),
        view_layer: manifest.view_layer.clone(),
        frame: Some(manifest.frame),
    };
    let snapshot = Snapshot::evaluate_with_cache(doc, &context, Some(cache_directory))?;
    if snapshot.revision != manifest.revision
        || snapshot.scene_hash != manifest.scene_hash
        || snapshot.evaluation_hash != manifest.evaluation_hash
    {
        return Err(PotError::with_details(
            ErrorCode::StaleRender,
            "render was produced from a different scene revision or evaluation",
            json!({
                "manifest_revision": manifest.revision,
                "current_revision": snapshot.revision,
                "manifest_scene_hash": manifest.scene_hash,
                "current_scene_hash": snapshot.scene_hash,
            }),
        ));
    }
    let buffers = verify_assets(manifest_path, &manifest)?;
    if pixel.0 >= buffers.width || pixel.1 >= buffers.height {
        return Err(PotError::with_details(
            ErrorCode::InvalidArgument,
            "pixel is outside the render dimensions",
            json!({ "pixel": [pixel.0, pixel.1], "width": buffers.width, "height": buffers.height }),
        ));
    }
    let index = usize::try_from(pixel.1)
        .ok()
        .and_then(|y| y.checked_mul(usize::try_from(buffers.width).ok()?))
        .and_then(|row| row.checked_add(usize::try_from(pixel.0).ok()?))
        .ok_or_else(|| render_invalid("pixel index exceeds platform limits"))?;
    let object_index = *buffers
        .ids
        .get(index)
        .ok_or_else(|| render_invalid("object ID buffer has inconsistent dimensions"))?;
    if object_index == 0 {
        return Ok(background_result(pixel, domain, &snapshot.evaluation_hash));
    }
    if domain == PickDomain::Bone {
        return Err(PotError::with_details(
            ErrorCode::InvalidArgument,
            "the requested selection domain has no preview channel",
            json!({ "domain": domain_name(domain) }),
        ));
    }
    let objects = object_mapping(&manifest.objects, doc)?;
    let object = objects
        .get(&object_index)
        .copied()
        .ok_or_else(|| render_invalid("object ID buffer references an unmapped object"))?;
    let elements = element_mapping(&manifest.elements, doc)?;
    let element_index = *buffers
        .elements
        .get(index)
        .ok_or_else(|| render_invalid("element buffer has inconsistent dimensions"))?;
    let element = if element_index == 0 {
        None
    } else {
        Some(
            elements
                .get(&element_index)
                .copied()
                .ok_or_else(|| render_invalid("element buffer references an unmapped element"))?,
        )
    };
    if element.is_some_and(|record| record.node_id != object.node_id) {
        return Err(render_invalid(
            "element mapping belongs to a different object",
        ));
    }
    let visibility = match manifest.settings.get("visibility").and_then(Value::as_str) {
        Some("render") => true,
        Some("viewport") => false,
        _ => return Err(render_invalid("manifest visibility setting is invalid")),
    };
    let geometry = extract_geometry(
        doc,
        &snapshot,
        Mode::Solid,
        visibility,
        false,
        &BTreeMap::new(),
    )?;
    verify_mapping_matches_geometry(&manifest, &geometry)?;
    let camera = camera_from_manifest(&manifest.camera)?;
    let (ray_origin, ray_direction) =
        ray_for_pixel(&camera, buffers.width, buffers.height, pixel.0, pixel.1)?;
    let element_domain = element.map(|record| record.domain.as_str());
    let requested_domain_matches_channel = match domain {
        PickDomain::Object => true,
        PickDomain::Face | PickDomain::Vertex | PickDomain::Edge => element_domain == Some("face"),
        PickDomain::Stroke => element_domain == Some("stroke"),
        PickDomain::Point => element_domain == Some("point"),
        PickDomain::Bone => false,
    };
    if !requested_domain_matches_channel {
        return Err(PotError::with_details(
            ErrorCode::InvalidArgument,
            "the requested selection domain has no matching element channel",
            json!({ "domain": domain_name(domain) }),
        ));
    }
    let hit = match element_domain {
        Some("face") => nearest_hit(
            &geometry,
            object_index,
            element_index,
            ray_origin,
            ray_direction,
            &camera,
        ),
        Some("stroke" | "point") => nearest_line_hit(
            &geometry,
            object_index,
            element_index,
            ray_origin,
            ray_direction,
            &camera,
            buffers.width,
            buffers.height,
        ),
        None if domain == PickDomain::Object => nearest_object_hit(
            &geometry,
            object_index,
            ray_origin,
            ray_direction,
            &camera,
            buffers.width,
            buffers.height,
        ),
        None => None,
        Some(_) => return Err(render_invalid("element mapping domain is invalid")),
    }
    .ok_or_else(|| render_invalid("ID buffer pixel has no corresponding evaluated surface"))?;
    let depth = buffers.depth[index];
    let tolerance = 1.0e-6_f64.max(f64::from(depth.abs()) * 1.0e-6);
    if !depth.is_finite() || depth <= 0.0 || (hit.depth - f64::from(depth)).abs() > tolerance {
        return Err(render_invalid(
            "depth buffer does not match the evaluated surface",
        ));
    }
    let node = doc
        .nodes
        .get(&object.node_id)
        .ok_or_else(|| render_invalid("object mapping references a missing node"))?;
    let (item_id, selector) = select_item(
        doc,
        &snapshot,
        node,
        object,
        element,
        domain,
        hit.position,
        hit.depth,
        buffers.width,
        buffers.height,
        &camera,
    )?;
    let item_domain = domain_name(domain);
    let mut target = json!({ "id": object.node_id });
    if domain != PickDomain::Object {
        let selector = selector.ok_or_else(|| {
            PotError::invalid_argument("the hit has no matching element within pick threshold")
        })?;
        target["elements"] = json!({ "domain": item_domain, "ids": [selector] });
        target["snapshot_hash"] = json!(snapshot.evaluation_hash);
    }
    Ok(json!({
        "hit": true,
        "pixel": [pixel.0, pixel.1],
        "domain": item_domain,
        "target": target,
        "item": item_id,
        "instance_path": object.instance_path,
        "source": Value::Null,
        "editable": node.selectable,
        "world_position": hit.position.to_array(),
        "world_normal": hit.normal.map(|normal| normal.to_array()),
        "depth": hit.depth,
        "evaluation_index": hit.evaluation_index,
        "snapshot_hash": snapshot.evaluation_hash,
    }))
}

fn background_result(pixel: (u32, u32), domain: PickDomain, evaluation_hash: &str) -> Value {
    json!({
        "hit": false,
        "pixel": [pixel.0, pixel.1],
        "domain": domain_name(domain),
        "target": Value::Null,
        "item": Value::Null,
        "instance_path": Value::Null,
        "source": Value::Null,
        "editable": Value::Null,
        "world_position": Value::Null,
        "world_normal": Value::Null,
        "depth": Value::Null,
        "evaluation_index": Value::Null,
        "snapshot_hash": evaluation_hash,
    })
}

fn verify_render_key(manifest: &Manifest) -> Result<()> {
    let camera = serde_json::to_value(&manifest.camera).map_err(|error| {
        PotError::new(
            ErrorCode::InternalError,
            format!("camera serialization failed: {error}"),
        )
    })?;
    let mut core = json!({
        "scene_id": manifest.scene_id,
        "revision": manifest.revision,
        "scene_hash": manifest.scene_hash,
        "evaluation_hash": manifest.evaluation_hash,
        "engine": manifest.engine,
        "profile": manifest.profile,
        "frame": manifest.frame,
        "view_layer": manifest.view_layer,
        "view": manifest.view,
        "camera": camera,
        "units": manifest.units,
        "settings": manifest.settings,
        "pick_policy": manifest.pick_policy,
        "objects": manifest.objects,
        "elements": manifest.elements,
        "width": manifest.width,
        "height": manifest.height,
    });
    if let Some(warnings) = &manifest.warnings {
        core["warnings"] = json!(warnings);
    }
    if hash::sha256(&hash::canonicalize(&core)?) != manifest.render_key {
        return Err(render_invalid(
            "manifest render_key does not match its core fields",
        ));
    }
    Ok(())
}

fn verify_assets(manifest_path: &Path, manifest: &Manifest) -> Result<VerifiedBuffers> {
    if manifest.image.format.as_deref() != Some("png") {
        return Err(render_invalid("render image format is not PNG"));
    }
    let image_path = resolve_asset_path(manifest_path, &manifest.image.path)?;
    let image_bytes = fs::read(&image_path).map_err(|error| missing_asset(&image_path, &error))?;
    verify_hash(&image_bytes, &manifest.image.hash, &image_path)?;
    let decoder = png::Decoder::new(Cursor::new(&image_bytes));
    let reader = decoder
        .read_info()
        .map_err(|error| render_invalid(format!("render PNG is invalid: {error}")))?;
    let width = manifest.image.width;
    let height = manifest.image.height;
    if width == 0 || height == 0 || reader.info().width != width || reader.info().height != height {
        return Err(render_invalid("image dimensions do not match the manifest"));
    }
    let pixel_count = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| render_invalid("image dimensions exceed platform limits"))?;
    let expected_length = pixel_count
        .checked_mul(4)
        .ok_or_else(|| render_invalid("buffer dimensions exceed platform limits"))?;
    let read_buffer = |name: &str, expected_type: &str| -> Result<Vec<u8>> {
        let asset = manifest
            .buffers
            .get(name)
            .ok_or_else(|| render_invalid(format!("manifest {name} buffer is missing")))?;
        if asset.width != width
            || asset.height != height
            || asset.buffer_type.as_deref() != Some(expected_type)
        {
            return Err(render_invalid(format!(
                "{name} buffer metadata is inconsistent"
            )));
        }
        let path = resolve_asset_path(manifest_path, &asset.path)?;
        let bytes = fs::read(&path).map_err(|error| missing_asset(&path, &error))?;
        verify_hash(&bytes, &asset.hash, &path)?;
        if bytes.len() != expected_length {
            return Err(render_invalid(format!(
                "{name} buffer has the wrong byte length"
            )));
        }
        Ok(bytes)
    };
    let ids_bytes = read_buffer("ids", "u32_le")?;
    let depth_bytes = read_buffer("depth", "f32_le")?;
    let elements_bytes = read_buffer("elements", "u32_le")?;
    Ok(VerifiedBuffers {
        width,
        height,
        ids: decode_u32(&ids_bytes),
        depth: decode_f32(&depth_bytes),
        elements: decode_u32(&elements_bytes),
    })
}

fn resolve_asset_path(manifest_path: &Path, relative: &str) -> Result<PathBuf> {
    let relative = Path::new(relative);
    if relative.as_os_str().is_empty()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(render_invalid(
            "manifest asset path is not a safe relative path",
        ));
    }
    let directory = manifest_path
        .parent()
        .ok_or_else(|| render_invalid("manifest has no parent directory"))?;
    let root = directory
        .canonicalize()
        .map_err(|error| PotError::io(&error))?;
    let path = directory.join(relative);
    if !path.exists() {
        return Err(PotError::with_details(
            ErrorCode::FileNotFound,
            "render buffer or image is missing",
            json!({ "path": path }),
        ));
    }
    let canonical = path.canonicalize().map_err(|error| PotError::io(&error))?;
    if !canonical.starts_with(root) {
        return Err(render_invalid(
            "manifest asset resolves outside its directory",
        ));
    }
    Ok(canonical)
}

fn missing_asset(path: &Path, error: &std::io::Error) -> PotError {
    if error.kind() == std::io::ErrorKind::NotFound {
        PotError::with_details(
            ErrorCode::FileNotFound,
            "render buffer or image is missing",
            json!({ "path": path }),
        )
    } else {
        PotError::io(error)
    }
}

fn verify_hash(bytes: &[u8], expected: &str, path: &Path) -> Result<()> {
    if hash::sha256(bytes) != expected {
        return Err(PotError::with_details(
            ErrorCode::RenderInvalid,
            "render file hash does not match its manifest",
            json!({ "path": path }),
        ));
    }
    Ok(())
}

fn decode_u32(bytes: &[u8]) -> Vec<u32> {
    let (chunks, _) = bytes.as_chunks::<4>();
    chunks
        .iter()
        .map(|chunk| u32::from_le_bytes(*chunk))
        .collect()
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    let (chunks, _) = bytes.as_chunks::<4>();
    chunks
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

fn object_mapping<'a>(
    objects: &'a [ObjectRecord],
    doc: &SceneDoc,
) -> Result<BTreeMap<u32, &'a ObjectRecord>> {
    let mut mapping = BTreeMap::new();
    for object in objects {
        if object.index == 0
            || !doc.nodes.contains_key(&object.node_id)
            || mapping.insert(object.index, object).is_some()
        {
            return Err(render_invalid("object mapping is invalid or duplicated"));
        }
    }
    Ok(mapping)
}

fn element_mapping<'a>(
    elements: &'a [ElementRecord],
    doc: &SceneDoc,
) -> Result<BTreeMap<u32, &'a ElementRecord>> {
    let mut mapping = BTreeMap::new();
    for element in elements {
        let valid_domain = match element.domain.as_str() {
            "face" => element.element_id.starts_with('f'),
            "stroke" => element.element_id.starts_with('s'),
            "point" => element.element_id.starts_with('p'),
            _ => false,
        };
        if element.index == 0
            || !doc.nodes.contains_key(&element.node_id)
            || !valid_domain
            || mapping.insert(element.index, element).is_some()
        {
            return Err(render_invalid("element mapping is invalid or duplicated"));
        }
    }
    Ok(mapping)
}

fn verify_mapping_matches_geometry(manifest: &Manifest, geometry: &Geometry) -> Result<()> {
    let expected_objects = serde_json::to_value(&geometry.objects).map_err(|error| {
        PotError::new(
            ErrorCode::InternalError,
            format!("object mapping serialization failed: {error}"),
        )
    })?;
    let manifest_objects = serde_json::to_value(&manifest.objects).map_err(|error| {
        PotError::new(
            ErrorCode::InternalError,
            format!("manifest object mapping serialization failed: {error}"),
        )
    })?;
    let expected_elements = serde_json::to_value(&geometry.elements).map_err(|error| {
        PotError::new(
            ErrorCode::InternalError,
            format!("element mapping serialization failed: {error}"),
        )
    })?;
    let manifest_elements = serde_json::to_value(&manifest.elements).map_err(|error| {
        PotError::new(
            ErrorCode::InternalError,
            format!("manifest element mapping serialization failed: {error}"),
        )
    })?;
    if expected_objects != manifest_objects || expected_elements != manifest_elements {
        return Err(render_invalid(
            "render mapping does not match the evaluated scene",
        ));
    }
    Ok(())
}

fn nearest_hit(
    geometry: &Geometry,
    object_index: u32,
    element_index: u32,
    origin: DVec3,
    direction: DVec3,
    camera: &Camera,
) -> Option<Hit> {
    let forward = (camera.target - camera.position).normalize_or_zero();
    geometry
        .triangles
        .iter()
        .enumerate()
        .filter(|(_, triangle)| {
            triangle.object_index == object_index
                && (element_index == 0 || triangle.element_index == element_index)
        })
        .filter_map(|(evaluation_index, triangle)| {
            ray_triangle(
                origin,
                direction,
                triangle,
                camera,
                forward,
                evaluation_index,
            )
        })
        .min_by(|left, right| left.depth.total_cmp(&right.depth))
}

#[expect(
    clippy::too_many_arguments,
    reason = "line hit testing combines pixel ray and world-space stroke context"
)]
fn nearest_line_hit(
    geometry: &Geometry,
    object_index: u32,
    element_index: u32,
    origin: DVec3,
    direction: DVec3,
    camera: &Camera,
    width: u32,
    height: u32,
) -> Option<Hit> {
    geometry
        .lines
        .iter()
        .enumerate()
        .filter(|(_, line)| {
            line.object_index == object_index
                && (element_index == 0 || line.element_index == element_index)
        })
        .filter_map(|(evaluation_index, line)| {
            ray_line(
                origin,
                direction,
                line,
                camera,
                width,
                height,
                evaluation_index,
            )
        })
        .min_by(|left, right| left.depth.total_cmp(&right.depth))
}

fn nearest_object_hit(
    geometry: &Geometry,
    object_index: u32,
    origin: DVec3,
    direction: DVec3,
    camera: &Camera,
    width: u32,
    height: u32,
) -> Option<Hit> {
    let surface = nearest_hit(geometry, object_index, 0, origin, direction, camera);
    let line = nearest_line_hit(
        geometry,
        object_index,
        0,
        origin,
        direction,
        camera,
        width,
        height,
    );
    match (surface, line) {
        (Some(surface), Some(line)) if surface.depth <= line.depth => Some(surface),
        (_, Some(line)) => Some(line),
        (Some(surface), None) => Some(surface),
        (None, None) => None,
    }
}

fn ray_line(
    origin: DVec3,
    direction: DVec3,
    line: &Line,
    camera: &Camera,
    width: u32,
    height: u32,
    evaluation_index: usize,
) -> Option<Hit> {
    let segment = line.end - line.start;
    let direction_length_squared = direction.length_squared();
    let direction_segment_dot = direction.dot(segment);
    let segment_length_squared = segment.length_squared();
    let origin_offset = origin - line.start;
    let direction_origin_dot = direction.dot(origin_offset);
    let segment_origin_dot = segment.dot(origin_offset);
    let denominator = direction_length_squared * segment_length_squared
        - direction_segment_dot * direction_segment_dot;
    let mut fraction = if segment_length_squared <= f64::EPSILON {
        0.0
    } else if denominator.abs() <= f64::EPSILON {
        (segment_origin_dot / segment_length_squared).clamp(0.0, 1.0)
    } else {
        ((direction_length_squared * segment_origin_dot
            - direction_segment_dot * direction_origin_dot)
            / denominator)
            .clamp(0.0, 1.0)
    };
    let ray_distance = if direction_length_squared <= f64::EPSILON {
        0.0
    } else {
        ((direction_segment_dot * fraction - direction_origin_dot) / direction_length_squared)
            .max(0.0)
    };
    if ray_distance <= 0.0 && segment_length_squared > f64::EPSILON {
        fraction = (segment_origin_dot / segment_length_squared).clamp(0.0, 1.0);
    }
    let position = line.start + segment * fraction;
    let ray_position = origin + direction * ray_distance;
    let depth =
        (position - camera.position).dot((camera.target - camera.position).normalize_or_zero());
    if !depth.is_finite() || depth < camera.near || depth > camera.far {
        return None;
    }
    let radius = line.radius_start + (line.radius_end - line.radius_start) * fraction;
    let tolerance = radius.max(pixel_threshold(camera, depth, width, height) * 0.5);
    if position.distance(ray_position) > tolerance {
        return None;
    }
    Some(Hit {
        depth,
        position,
        normal: None,
        evaluation_index,
    })
}

fn ray_triangle(
    origin: DVec3,
    direction: DVec3,
    triangle: &Triangle,
    camera: &Camera,
    forward: DVec3,
    evaluation_index: usize,
) -> Option<Hit> {
    let edge1 = triangle.positions[1] - triangle.positions[0];
    let edge2 = triangle.positions[2] - triangle.positions[0];
    let p_vector = direction.cross(edge2);
    let determinant = edge1.dot(p_vector);
    if !determinant.is_finite()
        || determinant.abs() <= f64::MIN_POSITIVE
        || (!triangle.double_sided && determinant <= 0.0)
    {
        return None;
    }
    let inverse = 1.0 / determinant;
    let from_vertex = origin - triangle.positions[0];
    let u = from_vertex.dot(p_vector) * inverse;
    let q_vector = from_vertex.cross(edge1);
    let v = direction.dot(q_vector) * inverse;
    let distance = edge2.dot(q_vector) * inverse;
    if !u.is_finite()
        || !v.is_finite()
        || !distance.is_finite()
        || distance <= 0.0
        || u < -1.0e-12
        || v < -1.0e-12
        || u + v > 1.0 + 1.0e-12
    {
        return None;
    }
    let position = origin + direction * distance;
    let depth = (position - camera.position).dot(forward);
    if !depth.is_finite() || depth < camera.near || depth > camera.far {
        return None;
    }
    let normal = edge1.cross(edge2).normalize_or_zero();
    if normal.length_squared() <= f64::EPSILON {
        return None;
    }
    Some(Hit {
        depth,
        position,
        normal: Some(normal),
        evaluation_index,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "pick selection needs the evaluated face, camera, domain, and hit context"
)]
fn select_item(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    node: &crate::model::Node,
    object: &ObjectRecord,
    element: Option<&ElementRecord>,
    domain: PickDomain,
    hit_position: DVec3,
    depth: f64,
    width: u32,
    height: u32,
    camera: &Camera,
) -> Result<(Value, Option<String>)> {
    if domain == PickDomain::Object {
        return Ok((json!({ "id": object.node_id, "domain": "object" }), None));
    }
    if matches!(domain, PickDomain::Stroke | PickDomain::Point) {
        let element = element.ok_or_else(|| {
            PotError::invalid_argument("stroke or point selection requires an element channel")
        })?;
        if element.domain != domain_name(domain) {
            return Err(PotError::invalid_argument(
                "the hit does not contain the requested stroke or point channel",
            ));
        }
        let selector = element.element_id.clone();
        let item = json!({
            "id": selector,
            "domain": domain_name(domain),
            "data_id": node.data,
        });
        return Ok((item, Some(selector)));
    }
    let face_element = element.ok_or_else(|| {
        PotError::invalid_argument("element selection requires an element channel")
    })?;
    let face_id = face_element
        .element_id
        .strip_prefix('f')
        .and_then(|id| id.parse::<u32>().ok())
        .ok_or_else(|| render_invalid("face mapping ID is invalid"))?;
    let data_id = node
        .data
        .as_ref()
        .ok_or_else(|| render_invalid("picked mesh object has no data reference"))?;
    let mesh = snapshot
        .meshes
        .get(&object.node_id)
        .ok_or_else(|| render_invalid("picked object has no evaluated mesh"))?;
    let face = mesh
        .faces
        .iter()
        .find(|face| face.id == face_id)
        .ok_or_else(|| render_invalid("picked face no longer exists"))?;
    let matrix = DMat4::from_cols_array(
        &snapshot
            .nodes
            .get(&object.node_id)
            .ok_or_else(|| render_invalid("picked node is missing from snapshot"))?
            .world_matrix,
    );
    let threshold = pixel_threshold(camera, depth, width, height) * 8.0;
    let selected = match domain {
        PickDomain::Face => Some(format!("f{face_id}")),
        PickDomain::Vertex => face
            .vertices
            .iter()
            .filter_map(|vertex_id| {
                let vertex = mesh.vertex(*vertex_id)?;
                let world = matrix.transform_point3(vertex.co);
                let distance = world.distance(hit_position);
                (distance <= threshold).then_some((distance, *vertex_id))
            })
            .min_by(|left, right| left.0.total_cmp(&right.0))
            .map(|(_, id)| format!("v{id}")),
        PickDomain::Edge => mesh
            .edges
            .iter()
            .filter(|edge| {
                face.vertices.contains(&edge.vertices[0])
                    && face.vertices.contains(&edge.vertices[1])
            })
            .filter_map(|edge| {
                let start = matrix.transform_point3(mesh.vertex(edge.vertices[0])?.co);
                let end = matrix.transform_point3(mesh.vertex(edge.vertices[1])?.co);
                let delta = end - start;
                let fraction = if delta.length_squared() <= f64::EPSILON {
                    0.0
                } else {
                    ((hit_position - start).dot(delta) / delta.length_squared()).clamp(0.0, 1.0)
                };
                let nearest = start + delta * fraction;
                let distance = nearest.distance(hit_position);
                (distance <= threshold).then_some((distance, edge.id))
            })
            .min_by(|left, right| left.0.total_cmp(&right.0))
            .map(|(_, id)| format!("e{id}")),
        PickDomain::Bone | PickDomain::Stroke | PickDomain::Point | PickDomain::Object => None,
    };
    let Some(selector) = selected else {
        return Err(PotError::with_details(
            ErrorCode::InvalidArgument,
            "the hit face has no matching element within the pixel threshold",
            json!({ "domain": domain_name(domain), "node_id": object.node_id }),
        ));
    };
    let item = json!({ "id": selector, "domain": domain_name(domain), "data_id": data_id });
    let _ = doc;
    Ok((item, Some(selector)))
}

fn pixel_threshold(camera: &Camera, depth: f64, width: u32, height: u32) -> f64 {
    let pixel_size = match camera.projection {
        Projection::Orthographic {
            height: world_height,
        } => world_height / f64::from(height),
        Projection::Perspective {
            lens_mm,
            sensor_width_mm,
        } => (depth * sensor_width_mm / lens_mm) / f64::from(width),
        Projection::Panorama { .. } => (depth * std::f64::consts::PI) / f64::from(height),
    };
    pixel_size.max(1.0e-6)
}

fn camera_from_manifest(camera: &CameraManifest) -> Result<Camera> {
    let projection = match camera.projection.as_str() {
        "orthographic" => Projection::Orthographic {
            height: camera
                .ortho_height
                .filter(|height| height.is_finite() && *height > 0.0)
                .ok_or_else(|| render_invalid("orthographic height is missing or invalid"))?,
        },
        "perspective" => Projection::Perspective {
            lens_mm: camera
                .lens_mm
                .filter(|lens| lens.is_finite() && *lens > 0.0)
                .ok_or_else(|| render_invalid("camera lens is missing or invalid"))?,
            sensor_width_mm: camera
                .sensor_width_mm
                .filter(|sensor| sensor.is_finite() && *sensor > 0.0)
                .ok_or_else(|| render_invalid("camera sensor width is missing or invalid"))?,
        },
        "panorama" => Projection::Panorama {
            fisheye: camera.panorama_type == "fisheye_equidistant",
        },
        "fisheye" => Projection::Panorama { fisheye: true },
        _ => return Err(render_invalid("camera projection is not supported")),
    };
    let depth_of_field = camera.depth_of_field.as_ref().map(|dof| DepthOfField {
        focus_distance: dof.focus_distance,
        aperture_radius: dof.aperture_radius,
        aperture_blades: dof.aperture_blades,
    });
    let stereo_mode = match camera.stereo_mode.as_str() {
        "side_by_side" => StereoMode::SideBySide,
        "anaglyph" => StereoMode::Anaglyph,
        _ => StereoMode::None,
    };
    Ok(Camera {
        position: DVec3::from_array(camera.position),
        target: DVec3::from_array(camera.target),
        up: DVec3::from_array(camera.up),
        near: camera.near,
        far: camera.far,
        projection,
        shift: camera.shift,
        depth_of_field,
        stereo_mode,
        interocular_distance: camera.interocular_distance,
    })
}

fn valid_camera(camera: &CameraManifest) -> bool {
    camera.position.iter().all(|value| value.is_finite())
        && camera.target.iter().all(|value| value.is_finite())
        && camera.up.iter().all(|value| value.is_finite())
        && camera.matrix.iter().all(|value| value.is_finite())
        && camera.near.is_finite()
        && camera.far.is_finite()
        && camera.near >= 0.0
        && camera.far > camera.near
        && (DVec3::from_array(camera.target) - DVec3::from_array(camera.position)).length_squared()
            > f64::EPSILON
}

fn domain_name(domain: PickDomain) -> &'static str {
    match domain {
        PickDomain::Object => "object",
        PickDomain::Vertex => "vertex",
        PickDomain::Edge => "edge",
        PickDomain::Face => "face",
        PickDomain::Bone => "bone",
        PickDomain::Stroke => "stroke",
        PickDomain::Point => "point",
    }
}

fn render_invalid(message: impl Into<String>) -> PotError {
    PotError::new(ErrorCode::RenderInvalid, message)
}
