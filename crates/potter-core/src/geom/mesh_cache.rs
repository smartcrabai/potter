use std::{
    collections::HashMap,
    sync::{Arc, LazyLock, Mutex},
};

use glam::DVec3;
use serde_json::{Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    exchange::{
        ExchangeUvLayer,
        alembic::{self, MeshCacheSample},
    },
    model::Modifier,
    params::{self, ParameterFamily},
};

use super::{
    Mesh,
    clnor::{self, NormalCodeOptions, SetterInput},
    color_layers::{self, ColorDomain, ColorLayer},
    uv_layers::{self, UvLayer},
};

static CACHE_SAMPLES: LazyLock<Mutex<HashMap<String, Arc<Vec<MeshCacheSample>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Replace a mesh with the requested Alembic cache sample.
///
/// Returns the warnings for cache data the mesh could not take, such as UV layers past the limit.
///
/// # Errors
///
/// Returns a dependency, asset-integrity, archive, parameter, or evaluation error.
#[expect(
    clippy::too_many_lines,
    reason = "resource verification, time sampling, and mesh replacement are one atomic modifier evaluation"
)]
pub(crate) fn evaluate(
    mesh: &mut Mesh,
    modifier: &Modifier,
    frame: f64,
    fps: u32,
    fps_base: f64,
    resource_hash: &str,
    archive_bytes: &[u8],
    is_sequence: bool,
) -> Result<Vec<PotError>> {
    let resource = string_param(modifier, "resource")?;
    let object_path = string_param(modifier, "object_path")?;
    let frame_offset = number_param(modifier, "frame_offset")?;
    let velocity_scale = number_param(modifier, "velocity_scale")?;
    let velocity_name = modifier
        .params
        .get("velocity_name")
        .and_then(Value::as_str)
        .or_else(|| {
            params::default_value(
                ParameterFamily::Modifier,
                &modifier.modifier_type,
                "velocity_name",
            )
            .and_then(Value::as_str)
        })
        .ok_or_else(|| invalid_parameter(modifier, "velocity_name", "a string"))?;
    let velocity_unit = modifier
        .params
        .get("velocity_unit")
        .and_then(Value::as_str)
        .or_else(|| {
            params::default_value(
                ParameterFamily::Modifier,
                &modifier.modifier_type,
                "velocity_unit",
            )
            .and_then(Value::as_str)
        })
        .unwrap_or("SECOND");
    let override_frame = match modifier.params.get("override_frame") {
        None | Some(Value::Null) => None,
        Some(value) => Some(
            value
                .as_f64()
                .filter(|number| number.is_finite())
                .ok_or_else(|| {
                    invalid_parameter(modifier, "override_frame", "a finite number or null")
                })?,
        ),
    };
    let interpolate = bool_param(modifier, "use_vertex_interpolation")?;
    let requested = override_frame.unwrap_or(frame);
    let mut read = read_params(modifier)?;
    let mut warnings = Vec::new();
    let rate = f64::from(fps) / fps_base;
    if !rate.is_finite() || rate <= 0.0 {
        return Err(PotError::with_details(
            ErrorCode::EvaluationFailed,
            "Mesh Sequence Cache requires a finite positive scene frame rate",
            json!({"modifier_id":modifier.id,"fps":fps,"fps_base":fps_base}),
        ));
    }
    let time_seconds = if is_sequence {
        requested
    } else {
        (requested - frame_offset) / rate
    };
    if !time_seconds.is_finite() {
        return Err(PotError::with_details(
            ErrorCode::EvaluationFailed,
            "Mesh Sequence Cache time is not finite",
            json!({"modifier_id":modifier.id,"frame":frame,"time_seconds":time_seconds}),
        ));
    }
    let velocity_factor = if velocity_unit == "FRAME" {
        velocity_scale * rate
    } else if velocity_unit == "SECOND" {
        velocity_scale
    } else {
        return Err(invalid_parameter(
            modifier,
            "velocity_unit",
            "FRAME or SECOND",
        ));
    };
    let mut samples = cached_mesh_cache_samples(
        archive_bytes,
        resource_hash,
        resource,
        object_path,
        read,
        velocity_name,
        velocity_factor != 0.0,
        modifier,
    )?;
    if samples.is_empty() {
        return Err(PotError::with_details(
            ErrorCode::EvaluationFailed,
            "Alembic PolyMesh has no samples",
            json!({"modifier_id":modifier.id,"resource":resource,"object_path":object_path}),
        ));
    }
    let topology_changed = {
        let sample = sample_at_time(&samples, time_seconds, interpolate).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::EvaluationFailed,
                "Alembic PolyMesh has no samples",
                json!({"modifier_id":modifier.id,"resource":resource,"object_path":object_path}),
            )
        })?;
        !topology_dimensions_match(mesh, sample.sample.positions.len(), &sample.sample.faces)
            || (requires_connectivity_check(
                is_sequence,
                sample.sample.has_multiple_topology_samples,
            ) && !topology_matches(mesh, sample.sample.positions.len(), &sample.sample.faces))
    };
    if topology_changed {
        // Blender's file-sequence reader rereads all supported streams for changed connectivity.
        read = ReadAttributes::ALL;
        samples = cached_mesh_cache_samples(
            archive_bytes,
            resource_hash,
            resource,
            object_path,
            read,
            velocity_name,
            velocity_factor != 0.0,
            modifier,
        )?;
        if samples.is_empty() {
            return Err(PotError::with_details(
                ErrorCode::EvaluationFailed,
                "Alembic PolyMesh has no samples",
                json!({"modifier_id":modifier.id,"resource":resource,"object_path":object_path}),
            ));
        }
    }
    let sample = sample_at_time(&samples, time_seconds, interpolate).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::EvaluationFailed,
            "Alembic PolyMesh has no samples",
            json!({"modifier_id":modifier.id,"resource":resource,"object_path":object_path}),
        )
    })?;
    if topology_changed {
        let positions = sample
            .sample
            .positions
            .iter()
            .enumerate()
            .map(|(index, _)| evaluated_position(&sample, index, modifier))
            .collect::<Result<Vec<_>>>()?;
        let mut evaluated = Mesh::from_positions_and_faces(positions, sample.sample.faces.clone())
            .map_err(|error| {
                PotError::with_details(
                    ErrorCode::EvaluationFailed,
                    "Alembic cache topology is invalid for the selected vertex sample",
                    json!({"modifier_id":modifier.id,"reason":error.to_string()}),
                )
            })?;
        carry_uv_store(mesh, &mut evaluated);
        apply_cache_attributes(
            &mut evaluated,
            sample.sample,
            read,
            velocity_factor,
            modifier,
            &mut warnings,
        )?;
        validate_cache_output(&evaluated, modifier)?;
        *mesh = evaluated;
        return Ok(warnings);
    }
    if read.vertices {
        for (index, vertex) in mesh.vertices.iter_mut().enumerate() {
            vertex.co = evaluated_position(&sample, index, modifier)?;
        }
    }
    apply_cache_attributes(
        mesh,
        sample.sample,
        read,
        velocity_factor,
        modifier,
        &mut warnings,
    )?;
    validate_cache_output(mesh, modifier)?;
    Ok(warnings)
}

fn topology_dimensions_match(
    mesh: &Mesh,
    cache_vertex_count: usize,
    cache_faces: &[Vec<usize>],
) -> bool {
    !mesh.vertices.is_empty()
        && mesh.vertices.len() == cache_vertex_count
        && mesh.faces.len() == cache_faces.len()
        && mesh
            .faces
            .iter()
            .map(|face| face.vertices.len())
            .sum::<usize>()
            == cache_faces.iter().map(Vec::len).sum::<usize>()
}

fn requires_connectivity_check(is_sequence: bool, has_multiple_topology_samples: bool) -> bool {
    is_sequence || has_multiple_topology_samples
}

fn topology_matches(mesh: &Mesh, cache_vertex_count: usize, cache_faces: &[Vec<usize>]) -> bool {
    if !topology_dimensions_match(mesh, cache_vertex_count, cache_faces) {
        return false;
    }
    let indices = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<HashMap<_, _>>();
    mesh.faces
        .iter()
        .zip(cache_faces)
        .all(|(face, cache_face)| {
            face.vertices.len() == cache_face.len()
                && face.vertices.iter().zip(cache_face).all(|(id, index)| {
                    indices
                        .get(id)
                        .is_some_and(|mesh_index| mesh_index == index)
                })
        })
}

fn cached_mesh_cache_samples(
    archive_bytes: &[u8],
    resource_hash: &str,
    resource: &str,
    object_path: &str,
    read: ReadAttributes,
    velocity_name: &str,
    read_velocity: bool,
    modifier: &Modifier,
) -> Result<Arc<Vec<MeshCacheSample>>> {
    let velocity_parameter = if read_velocity { velocity_name } else { "" };
    let key = format!(
        "{resource_hash}\0{object_path}\0{}\0{}\0{}\0{}",
        read.uv, read.color, read.polygons, velocity_parameter
    );
    let mut parsed = CACHE_SAMPLES.lock().map_err(|_| {
        PotError::new(
            ErrorCode::InternalError,
            "Alembic mesh cache lock is poisoned",
        )
    })?;
    if let Some(samples) = parsed.get(&key) {
        return Ok(Arc::clone(samples));
    }
    let samples = Arc::new(alembic::read_mesh_cache(
        archive_bytes,
        object_path,
        read.uv,
        read.color,
        read.polygons,
        velocity_parameter,
    )?);
    if samples.windows(2).any(|pair| {
        pair.first()
            .zip(pair.get(1))
            .is_some_and(|(first, second)| first.time_seconds > second.time_seconds)
    }) {
        return Err(PotError::with_details(
            ErrorCode::EvaluationFailed,
            "Alembic mesh cache sample times are not ordered",
            json!({"modifier_id":modifier.id,"resource":resource,"object_path":object_path}),
        ));
    }
    parsed.insert(key, Arc::clone(&samples));
    Ok(samples)
}

fn apply_cache_attributes(
    mesh: &mut Mesh,
    sample: &MeshCacheSample,
    read: ReadAttributes,
    velocity_scale: f64,
    modifier: &Modifier,
    warnings: &mut Vec<PotError>,
) -> Result<()> {
    if read.polygons
        && let Some(normals) = &sample.normals
    {
        let corner_count = mesh
            .faces
            .iter()
            .map(|face| face.vertices.len())
            .sum::<usize>();
        if normals.len() != corner_count {
            return Err(PotError::with_details(
                ErrorCode::EvaluationFailed,
                "Alembic corner normal count does not match the selected mesh topology",
                json!({"modifier_id":modifier.id,"normal_count":normals.len(),"corner_count":corner_count}),
            ));
        }
        let vectors = normals
            .iter()
            .map(|normal| Some(DVec3::from_array(*normal)))
            .collect::<Vec<_>>();
        clnor::encode_vectors(
            mesh,
            Some(&vectors),
            NormalCodeOptions {
                corner_order: None,
                setter_input: SetterInput::NormalizeInput,
            },
        )?;
    }
    if read.uv
        && let Some(uv_layers) = &sample.uv_layers
    {
        for layer in uv_layers {
            check_cache_uv_layer(mesh, modifier, layer)?;
        }
        write_cache_uvs(
            mesh,
            modifier,
            uv_layers,
            sample.canonical_uv_layer.as_deref(),
            read.polygons,
            warnings,
        )?;
    }
    if read.color {
        let mut store = color_layers::read(mesh);
        for color in &sample.colors {
            let (domain, values) = cache_colors(mesh, color, modifier)?;
            let layer = ColorLayer {
                name: color.name.clone(),
                domain,
                values,
            };
            if let Some(index) = store.index_of(&layer.name) {
                store.layers[index] = layer;
            } else {
                store.layers.push(layer);
            }
        }
        if let Some(first) = sample.colors.first() {
            if store.render_index().is_none() {
                store.render = Some(first.name.clone());
            }
            if store
                .active
                .as_deref()
                .and_then(|active| store.index_of(active))
                .is_none()
            {
                store.active = Some(first.name.clone());
            }
            color_layers::write(mesh, &store);
            for color in &sample.colors {
                set_cache_color_data_type(mesh, &color.name)?;
            }
        }
    }
    if velocity_scale != 0.0
        && let Some((scope, velocities)) = &sample.velocities
    {
        let scaled = velocities
            .iter()
            .map(|velocity| {
                velocity
                    .iter()
                    .map(|component| component * velocity_scale)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        if scaled.iter().flatten().any(|value| !value.is_finite()) {
            return Err(PotError::with_details(
                ErrorCode::EvaluationFailed,
                "Alembic velocity scaling produced a non-finite value",
                json!({"modifier_id":modifier.id,"velocity_scale":velocity_scale}),
            ));
        }
        mesh.attributes.insert(
            "alembic_velocity".to_owned(),
            json!({"scope":scope,"values":scaled}),
        );
    }
    Ok(())
}

fn set_cache_color_data_type(mesh: &mut Mesh, name: &str) -> Result<()> {
    let metadata = mesh
        .attributes
        .entry("blender_metadata".to_owned())
        .or_insert_with(|| json!({}));
    let metadata = metadata.as_object_mut().ok_or_else(|| {
        PotError::with_details(
            ErrorCode::SceneInvalid,
            "Blender mesh metadata is not an object",
            json!({"attribute":"blender_metadata","color":name}),
        )
    })?;
    let color_types = metadata
        .entry("color_data_types".to_owned())
        .or_insert_with(|| json!({}));
    let color_types = color_types.as_object_mut().ok_or_else(|| {
        PotError::with_details(
            ErrorCode::SceneInvalid,
            "Blender color data-type metadata is not an object",
            json!({"color":name}),
        )
    })?;
    color_types.insert(name.to_owned(), json!("BYTE_COLOR"));
    Ok(())
}

/// Carries UV layers onto a topology replacement, clearing face values and pins when layout changes.
fn carry_uv_store(input: &Mesh, replacement: &mut Mesh) {
    let mut store = uv_layers::read(input);
    let same_layout = input.faces.len() == replacement.faces.len()
        && input
            .faces
            .iter()
            .zip(&replacement.faces)
            .all(|(old, new)| old.vertices.len() == new.vertices.len());
    if !same_layout {
        for layer in &mut store.layers {
            layer.faces = vec![None; replacement.faces.len()];
            layer.pinned = vec![None; replacement.faces.len()];
        }
    }
    uv_layers::write(replacement, &store);
}

/// Writes named cache UV layers through the canonical store, reversing each file-corner row once.
fn write_cache_uvs(
    mesh: &mut Mesh,
    modifier: &Modifier,
    cached: &[ExchangeUvLayer],
    canonical_uv_layer: Option<&str>,
    read_polygons: bool,
    warnings: &mut Vec<PotError>,
) -> Result<()> {
    if cached.is_empty() {
        return Ok(());
    }
    let mut store = uv_layers::read(mesh);
    let mut first_written = None;
    for cached_layer in cached {
        let existing = store.index_of(&cached_layer.name);
        // Blender streams arbitrary UV GeomParams with the UV flag; the canonical `.geom/uv`
        // layer is populated only by the POLY path. Preserve an existing canonical layer, or
        // materialize its zero row on a new mesh as Blender does.
        if !read_polygons
            && canonical_uv_layer == Some(cached_layer.name.as_str())
            && existing.is_some()
        {
            continue;
        }
        let index = match existing {
            Some(index) => index,
            None if store.layers.len() < uv_layers::MAX_LAYERS => {
                store.layers.push(UvLayer::new(
                    cached_layer.name.clone(),
                    vec![None; mesh.faces.len()],
                ));
                store.layers.len() - 1
            }
            None => {
                warnings.push(layer_limit_warning(modifier, &cached_layer.name));
                continue;
            }
        };
        let layer = store.layers.get_mut(index).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "UV layer index is out of range")
        })?;
        layer.faces = cached_layer
            .corners
            .iter()
            .map(|corners| Some(corners.iter().rev().copied().collect()))
            .collect();
        if first_written.is_none() {
            first_written = Some(cached_layer.name.clone());
        }
    }
    let active_exists = store
        .active
        .as_deref()
        .is_some_and(|name| store.index_of(name).is_some());
    if !active_exists {
        store.active.clone_from(&first_written);
    }
    let render_exists = store
        .render
        .as_deref()
        .is_some_and(|name| store.index_of(name).is_some());
    if !render_exists {
        store.render.clone_from(&first_written);
    }
    uv_layers::write(mesh, &store);
    Ok(())
}

fn check_cache_uv_layer(mesh: &Mesh, modifier: &Modifier, layer: &ExchangeUvLayer) -> Result<()> {
    if layer.corners.len() != mesh.faces.len() {
        return Err(PotError::with_details(
            ErrorCode::EvaluationFailed,
            "Alembic UV face count does not match the selected mesh topology",
            json!({"modifier_id":modifier.id,"layer":layer.name,"cache_faces":layer.corners.len(),"mesh_faces":mesh.faces.len()}),
        ));
    }
    if let Some((face, uv)) = mesh
        .faces
        .iter()
        .zip(&layer.corners)
        .find(|(face, uv)| face.vertices.len() != uv.len())
    {
        return Err(PotError::with_details(
            ErrorCode::EvaluationFailed,
            "Alembic UV corner count does not match the selected face topology",
            json!({"modifier_id":modifier.id,"layer":layer.name,"face_id":face.id,"cache_corners":uv.len(),"mesh_corners":face.vertices.len()}),
        ));
    }
    Ok(())
}

fn layer_limit_warning(modifier: &Modifier, layer: &str) -> PotError {
    PotError::with_details(
        ErrorCode::LimitExceeded,
        format!(
            "Mesh Sequence Cache UV layer `{layer}` was not written: the mesh already has {} UV layers",
            uv_layers::MAX_LAYERS
        ),
        json!({
            "feature_id":"uv.layer_limit",
            "status":"dropped",
            "modifier_id":modifier.id,
            "modifier_type":modifier.modifier_type,
            "layer":layer,
            "limit":uv_layers::MAX_LAYERS,
        }),
    )
}

fn validate_cache_output(mesh: &Mesh, modifier: &Modifier) -> Result<()> {
    mesh.validate().map_err(|error| {
        PotError::with_details(
            ErrorCode::EvaluationFailed,
            "Alembic cache produced an invalid mesh",
            json!({"modifier_id":modifier.id,"reason":error.to_string()}),
        )
    })
}

/// Maps one raw color parameter onto the mesh's corners the way Blender's reader does.
///
/// Facevarying data is indexed by file corner through `.indices` when they cover every corner.
/// Every other case indexes the stored values by the corner's vertex, so `uniform` and `constant`
/// also read by vertex. Samples past the stored values stay transparent black, as Blender leaves
/// them zero-initialized. Alembic polygons wind opposite to Potter's, so Potter corner `p` of a
/// face is file corner `len - 1 - p`.
fn cache_colors(
    mesh: &Mesh,
    color: &alembic::MeshCacheColor,
    modifier: &Modifier,
) -> Result<(ColorDomain, Vec<[f64; 4]>)> {
    if !matches!(
        color.scope.as_str(),
        "constant" | "vertex" | "varying" | "uniform" | "facevarying"
    ) {
        return Err(PotError::with_details(
            ErrorCode::EvaluationFailed,
            "Alembic cache color scope is unsupported",
            json!({"modifier_id":modifier.id,"scope":color.scope}),
        ));
    }
    if !matches!(color.extent, 3 | 4) || !color.values.len().is_multiple_of(color.extent) {
        return Err(PotError::with_details(
            ErrorCode::EvaluationFailed,
            "Alembic cache color tuple must have three or four components",
            json!({"modifier_id":modifier.id,"component_count":color.extent}),
        ));
    }
    let linear = color
        .values
        .chunks_exact(color.extent)
        .map(|tuple| byte_color_to_linear(tuple, modifier))
        .collect::<Result<Vec<_>>>()?;
    let corner_count = mesh
        .faces
        .iter()
        .map(|face| face.vertices.len())
        .sum::<usize>();
    // Blender reads through effective indices. An unindexed parameter has the identity over its
    // values, so their count stands in for `.indices`; facevarying applies only when that covers
    // every corner.
    let effective_len = color.indices.as_ref().map_or(linear.len(), Vec::len);
    let facevarying = color.scope == "facevarying" && effective_len == corner_count;
    let vertex_indices: HashMap<u32, usize> = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let mut values = Vec::with_capacity(corner_count);
    let mut face_offset = 0;
    for face in &mesh.faces {
        let face_len = face.vertices.len();
        for (position, vertex_id) in face.vertices.iter().enumerate() {
            let vertex_index = vertex_indices.get(vertex_id).copied().ok_or_else(|| {
                PotError::new(
                    ErrorCode::EvaluationFailed,
                    "Alembic color face references a missing vertex",
                )
            })?;
            let sample = if facevarying {
                let file_corner = face_offset + face_len - 1 - position;
                match &color.indices {
                    Some(indices) => indices
                        .get(file_corner)
                        .and_then(|&index| linear.get(index)),
                    None => linear.get(file_corner),
                }
            } else {
                linear.get(vertex_index)
            };
            values.push(sample.copied().unwrap_or([0.0; 4]));
        }
        face_offset += face_len;
    }
    Ok((ColorDomain::Corner, values))
}

/// Converts one raw color tuple to canonical linear RGBA through Blender's `BYTE_COLOR` path. Each
/// float becomes a byte, RGB decodes sRGB to linear, and alpha is `byte / 255`. A three-component
/// tuple has alpha 255.
fn byte_color_to_linear(tuple: &[f64], modifier: &Modifier) -> Result<[f64; 4]> {
    let (rgb, alpha) = match tuple {
        [red, green, blue] => ([*red, *green, *blue], 255),
        [red, green, blue, alpha] => ([*red, *green, *blue], unit_float_to_byte(*alpha, modifier)?),
        _ => {
            return Err(PotError::with_details(
                ErrorCode::EvaluationFailed,
                "Alembic cache color tuple must have three or four components",
                json!({"modifier_id":modifier.id,"component_count":tuple.len()}),
            ));
        }
    };
    let red = unit_float_to_byte(rgb[0], modifier)?;
    let green = unit_float_to_byte(rgb[1], modifier)?;
    let blue = unit_float_to_byte(rgb[2], modifier)?;
    Ok([
        crate::color::srgb_to_linear(f64::from(red) / 255.0),
        crate::color::srgb_to_linear(f64::from(green) / 255.0),
        crate::color::srgb_to_linear(f64::from(blue) / 255.0),
        f64::from(alpha) / 255.0,
    ])
}

/// Blender's `unit_float_to_uchar_clamp`: clamps to [0, 1], scales by 255, and adds 0.5 before the
/// truncating conversion, all in binary32 as Blender computes it. A non-finite component is
/// malformed input.
fn unit_float_to_byte(value: f64, modifier: &Modifier) -> Result<u8> {
    if !value.is_finite() {
        return Err(PotError::with_details(
            ErrorCode::EvaluationFailed,
            "Alembic cache color component is not finite",
            json!({"modifier_id":modifier.id}),
        ));
    }
    let scaled = (value as f32).clamp(0.0, 1.0) * 255.0 + 0.5;
    Ok(scaled as u8)
}

#[derive(Clone, Copy)]
struct ReadAttributes {
    vertices: bool,
    polygons: bool,
    uv: bool,
    color: bool,
}
impl ReadAttributes {
    const ALL: Self = Self {
        vertices: true,
        polygons: true,
        uv: true,
        color: true,
    };
}

fn read_params(modifier: &Modifier) -> Result<ReadAttributes> {
    let Some(value) = modifier.params.get("read_data").or_else(|| {
        params::default_value(
            ParameterFamily::Modifier,
            &modifier.modifier_type,
            "read_data",
        )
    }) else {
        return Ok(ReadAttributes::ALL);
    };
    let values = value.as_array().ok_or_else(|| {
        invalid_parameter(modifier, "read_data", "an array of Blender read-data flags")
    })?;
    let mut attributes = ReadAttributes {
        vertices: false,
        polygons: false,
        uv: false,
        color: false,
    };
    for value in values {
        match value.as_str() {
            Some("VERT") => attributes.vertices = true,
            Some("POLY") => attributes.polygons = true,
            Some("UV") => attributes.uv = true,
            Some("COLOR") => attributes.color = true,
            // Blender's Alembic stream reader does not map generic properties through this flag.
            Some("ATTRIBUTES") => {}
            _ => {
                return Err(invalid_parameter(
                    modifier,
                    "read_data",
                    "an array of Blender read-data flags",
                ));
            }
        }
    }
    Ok(attributes)
}

fn string_param<'a>(modifier: &'a Modifier, name: &str) -> Result<&'a str> {
    modifier
        .params
        .get(name)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| invalid_parameter(modifier, name, "a non-empty string"))
}

fn number_param(modifier: &Modifier, name: &str) -> Result<f64> {
    super::modifiers::number_param(modifier, name)
        .map_err(|_| invalid_parameter(modifier, name, "a finite number"))
}

fn bool_param(modifier: &Modifier, name: &str) -> Result<bool> {
    super::modifiers::bool_param(modifier, name)
        .map_err(|_| invalid_parameter(modifier, name, "a boolean"))
}

fn invalid_parameter(modifier: &Modifier, parameter: &str, expected: &str) -> PotError {
    PotError::with_details(
        ErrorCode::InvalidArgument,
        format!("Mesh Sequence Cache parameter `{parameter}` must be {expected}"),
        json!({"modifier_id":modifier.id,"parameter":parameter,"expected":expected}),
    )
}

struct SampleSelection<'a> {
    sample: &'a MeshCacheSample,
    next: Option<&'a MeshCacheSample>,
    factor: f64,
}

impl SampleSelection<'_> {
    fn position(&self, index: usize) -> Option<[f64; 3]> {
        let first = *self.sample.positions.get(index)?;
        match self.next {
            Some(next) => Some(lerp_position(
                first,
                *next.positions.get(index)?,
                self.factor,
            )),
            None => Some(first),
        }
    }
}

fn sample_at_time(
    samples: &[MeshCacheSample],
    time_seconds: f64,
    interpolate: bool,
) -> Option<SampleSelection<'_>> {
    let first_sample = samples.first()?;
    let upper = samples.partition_point(|sample| sample.time_seconds <= time_seconds);
    if upper == 0 {
        return Some(floor_sample(first_sample));
    }
    let lower = upper.checked_sub(1)?;
    let first = samples.get(lower)?;
    let Some(second) = samples.get(upper) else {
        return Some(floor_sample(first));
    };
    if !interpolate
        || first.faces != second.faces
        || first.positions.len() != second.positions.len()
    {
        return Some(floor_sample(first));
    }
    let span = second.time_seconds - first.time_seconds;
    if !span.is_finite() || span <= 0.0 {
        return Some(floor_sample(first));
    }
    Some(SampleSelection {
        sample: first,
        next: Some(second),
        factor: (time_seconds - first.time_seconds) / span,
    })
}

fn floor_sample(sample: &MeshCacheSample) -> SampleSelection<'_> {
    SampleSelection {
        sample,
        next: None,
        factor: 0.0,
    }
}

fn evaluated_position(
    sample: &SampleSelection<'_>,
    index: usize,
    modifier: &Modifier,
) -> Result<DVec3> {
    sample
        .position(index)
        .map(DVec3::from_array)
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::EvaluationFailed,
                "Alembic interpolated vertex sample is missing",
                json!({"modifier_id":modifier.id,"vertex_index":index}),
            )
        })
}

fn lerp_position(first: [f64; 3], second: [f64; 3], factor: f64) -> [f64; 3] {
    let [first_x, first_y, first_z] = first;
    let [second_x, second_y, second_z] = second;
    [
        first_x + (second_x - first_x) * factor,
        first_y + (second_y - first_y) * factor,
        first_z + (second_z - first_z) * factor,
    ]
}

#[cfg(test)]
mod tests {

    use proptest::prelude::*;

    use super::{lerp_position, topology_matches};
    use crate::geom::Mesh;

    #[test]
    fn topology_identity_detects_same_size_corner_reorder() -> Result<(), Box<dyn std::error::Error>>
    {
        let mesh = Mesh::from_positions_and_faces(
            vec![
                glam::DVec3::ZERO,
                glam::DVec3::X,
                glam::DVec3::ONE,
                glam::DVec3::Y,
            ],
            vec![vec![0, 1, 2, 3]],
        )?;
        assert!(topology_matches(&mesh, 4, &[vec![0, 1, 2, 3]]));
        assert!(!topology_matches(&mesh, 4, &[vec![3, 2, 1, 0]]));
        assert!(!topology_matches(&mesh, 5, &[vec![0, 1, 2, 3]]));
        Ok(())
    }

    #[test]
    fn connectivity_check_matches_blender_cache_modes() {
        assert!(!super::requires_connectivity_check(false, false));
        assert!(super::requires_connectivity_check(false, true));
        assert!(super::requires_connectivity_check(true, false));
    }

    const RED: [f64; 4] = [1.0, 0.0, 0.0, 1.0];
    const GREEN: [f64; 4] = [0.0, 1.0, 0.0, 1.0];
    const BLUE: [f64; 4] = [0.0, 0.0, 1.0, 1.0];
    const WHITE: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
    const CLEAR: [f64; 4] = [0.0; 4];

    fn test_modifier() -> crate::error::Result<crate::model::Modifier> {
        Ok(crate::model::Modifier {
            id: crate::model::Id::new("cache")?,
            modifier_type: "mesh_sequence_cache".to_owned(),
            name: "Mesh Sequence Cache".to_owned(),
            enabled: true,
            show_viewport: true,
            show_render: true,
            params: serde_json::Map::new(),
            binding_data: None,
            runtime: crate::model::ModifierRuntime::default(),
        })
    }

    fn raw_color(
        scope: &str,
        values: Vec<Vec<f64>>,
        indices: Option<Vec<usize>>,
    ) -> super::alembic::MeshCacheColor {
        let extent = values.first().map_or(3, Vec::len);
        assert!(
            values.iter().all(|tuple| tuple.len() == extent),
            "a color parameter has one extent"
        );
        super::alembic::MeshCacheColor {
            name: "Cd".to_owned(),
            scope: scope.to_owned(),
            extent,
            values: values.into_iter().flatten().collect(),
            indices,
        }
    }

    fn assert_colors_close(actual: &[[f64; 4]], expected: &[[f64; 4]]) {
        assert_colors_within(actual, expected, 1e-9);
    }

    /// Asserts each channel matches `expected` within `tolerance`.
    fn assert_colors_within(actual: &[[f64; 4]], expected: &[[f64; 4]], tolerance: f64) {
        assert_eq!(actual.len(), expected.len(), "corner count");
        for (corner, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            for channel in 0..4 {
                assert!(
                    (actual[channel] - expected[channel]).abs() < tolerance,
                    "corner {corner} channel {channel}: {actual:?} != {expected:?}"
                );
            }
        }
    }

    #[test]
    fn raw_vertex_scope_ignores_indices() -> Result<(), Box<dyn std::error::Error>> {
        let mesh = super::Mesh::from_positions_and_faces(
            vec![glam::DVec3::ZERO, glam::DVec3::X, glam::DVec3::Y],
            vec![vec![0, 1, 2]],
        )?;
        let color = raw_color(
            "vertex",
            vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
            Some(vec![2, 2, 2]),
        );
        let (domain, values) = super::cache_colors(&mesh, &color, &test_modifier()?)?;
        assert_eq!(domain, super::ColorDomain::Corner);
        assert_colors_close(&values, &[RED, GREEN, BLUE]);
        Ok(())
    }

    #[test]
    fn facevarying_indices_map_file_corners_into_potter_order()
    -> Result<(), Box<dyn std::error::Error>> {
        let mesh = super::Mesh::from_positions_and_faces(
            vec![
                glam::DVec3::ZERO,
                glam::DVec3::X,
                glam::DVec3::Y,
                glam::DVec3::ONE,
            ],
            vec![vec![0, 1, 2], vec![0, 2, 3]],
        )?;
        // Value k is the gray level k/5, so each byte is 51k and decodes through sRGB.
        let values: Vec<Vec<f64>> = (0..6).map(|k| vec![f64::from(k) / 5.0, 0.0, 0.0]).collect();
        let color = raw_color("facevarying", values, Some(vec![5, 4, 3, 2, 1, 0]));
        let (_, actual) = super::cache_colors(&mesh, &color, &test_modifier()?)?;
        // Potter corners take file corners 2,1,0,5,4,3, whose indices are 3,4,5,0,1,2.
        let expected = [3, 4, 5, 0, 1, 2].map(|k| {
            [
                crate::color::srgb_to_linear(f64::from(k) / 5.0),
                0.0,
                0.0,
                1.0,
            ]
        });
        assert_colors_close(&actual, &expected);
        Ok(())
    }

    #[test]
    fn facevarying_without_matching_indices_uses_vertex_path()
    -> Result<(), Box<dyn std::error::Error>> {
        let mesh = super::Mesh::from_positions_and_faces(
            vec![
                glam::DVec3::ZERO,
                glam::DVec3::X,
                glam::DVec3::Y,
                glam::DVec3::ONE,
            ],
            vec![vec![0, 1, 2], vec![0, 2, 3]],
        )?;
        let color = raw_color(
            "facevarying",
            vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
                vec![1.0, 1.0, 1.0],
            ],
            // Three indices for six corners: Blender takes the vertex-index path.
            Some(vec![3, 2, 1]),
        );
        let (_, values) = super::cache_colors(&mesh, &color, &test_modifier()?)?;
        assert_colors_close(&values, &[RED, GREEN, BLUE, RED, BLUE, WHITE]);
        Ok(())
    }

    #[test]
    fn uniform_and_constant_index_by_corner_vertex() -> Result<(), Box<dyn std::error::Error>> {
        let mesh = super::Mesh::from_positions_and_faces(
            vec![glam::DVec3::ZERO, glam::DVec3::X, glam::DVec3::Y],
            vec![vec![0, 1, 2]],
        )?;
        let modifier = test_modifier()?;
        let uniform = raw_color(
            "uniform",
            vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
            None,
        );
        let (_, values) = super::cache_colors(&mesh, &uniform, &modifier)?;
        assert_colors_close(&values, &[RED, GREEN, BLUE]);

        // Only vertex 0 has a sample, so corners on vertices 1 and 2 are out of range: transparent.
        for scope in ["uniform", "constant"] {
            let single = raw_color(scope, vec![vec![1.0, 0.0, 0.0]], None);
            let (_, values) = super::cache_colors(&mesh, &single, &modifier)?;
            assert_colors_close(&values, &[RED, CLEAR, CLEAR]);
        }
        Ok(())
    }

    #[test]
    fn byte_color_quantizes_then_decodes_srgb_and_alpha() -> Result<(), Box<dyn std::error::Error>>
    {
        let mesh = super::Mesh::from_positions_and_faces(
            vec![glam::DVec3::ZERO, glam::DVec3::X, glam::DVec3::Y],
            vec![vec![0, 1, 2]],
        )?;
        let rgba = raw_color(
            "vertex",
            vec![
                // Clamped to 255; alpha 0.5 quantizes to byte 128.
                vec![0.5, 0.2, 1.7, 0.5],
                // Negative clamps to 0; alpha 1.0 quantizes to byte 255.
                vec![0.25, 0.5, -0.3, 1.0],
                vec![0.0, 0.0, 0.0, 0.0],
            ],
            None,
        );
        let linear = |byte: u8| crate::color::srgb_to_linear(f64::from(byte) / 255.0);
        let (_, values) = super::cache_colors(&mesh, &rgba, &test_modifier()?)?;
        assert_colors_close(
            &values,
            &[
                [linear(128), linear(51), linear(255), 128.0 / 255.0],
                [linear(64), linear(128), linear(0), 1.0],
                CLEAR,
            ],
        );
        // Three components take alpha 255. Only vertex 0 has a sample, so the others are transparent.
        let rgb = raw_color("vertex", vec![vec![0.25, 0.5, -0.3]], None);
        let (_, values) = super::cache_colors(&mesh, &rgb, &test_modifier()?)?;
        assert_colors_close(
            &values,
            &[[linear(64), linear(128), linear(0), 1.0], CLEAR, CLEAR],
        );
        Ok(())
    }

    #[test]
    fn unsupported_scope_and_malformed_tuples_are_errors() -> Result<(), Box<dyn std::error::Error>>
    {
        let mesh = super::Mesh::from_positions_and_faces(
            vec![glam::DVec3::ZERO, glam::DVec3::X, glam::DVec3::Y],
            vec![vec![0, 1, 2]],
        )?;
        let modifier = test_modifier()?;
        let bogus = raw_color("bogus", vec![vec![0.0; 3]], None);
        assert!(super::cache_colors(&mesh, &bogus, &modifier).is_err());
        let two_components = raw_color("vertex", vec![vec![0.0, 0.0]], None);
        assert!(super::cache_colors(&mesh, &two_components, &modifier).is_err());
        let not_finite = raw_color("vertex", vec![vec![f64::NAN, 0.0, 0.0]], None);
        assert!(super::cache_colors(&mesh, &not_finite, &modifier).is_err());
        Ok(())
    }

    #[test]
    fn cache_color_layers_store_linear_byte_color_in_potter_corner_order()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut mesh = super::Mesh::from_positions_and_faces(
            vec![glam::DVec3::ZERO, glam::DVec3::X, glam::DVec3::Y],
            vec![vec![0, 1, 2]],
        )?;
        let modifier = test_modifier()?;
        let primaries = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        let point = super::alembic::MeshCacheColor {
            name: "PointGeom".to_owned(),
            scope: "vertex".to_owned(),
            extent: 3,
            values: primaries.clone(),
            indices: None,
        };
        let (domain, values) = super::cache_colors(&mesh, &point, &modifier)?;
        let mut store = super::color_layers::read(&mesh);
        store.layers.push(super::ColorLayer {
            name: "PointGeom".to_owned(),
            domain,
            values,
        });
        super::color_layers::write(&mut mesh, &store);
        super::set_cache_color_data_type(&mut mesh, "PointGeom")?;
        assert_eq!(
            mesh.attributes["blender_metadata"]["color_data_types"]["PointGeom"],
            "BYTE_COLOR"
        );
        let point_key = crate::geom::attribute_keys::storage_key("PointGeom");
        assert!(mesh.attributes.contains_key(point_key.as_ref()));
        let stored = super::color_layers::read(&mesh);
        let point_layer = stored
            .layers
            .first()
            .ok_or("point geom color layer is missing")?;
        assert_eq!(point_layer.domain, super::ColorDomain::Corner);
        assert_colors_close(&point_layer.values, &[RED, GREEN, BLUE]);

        // Corner file order is reversed per face: Potter corners 0,1,2 take file corners 2,1,0.
        let corner = super::alembic::MeshCacheColor {
            name: "CornerGeom".to_owned(),
            scope: "facevarying".to_owned(),
            extent: 3,
            values: primaries,
            indices: Some(vec![0, 1, 2]),
        };
        let (domain, values) = super::cache_colors(&mesh, &corner, &modifier)?;
        let mut store = super::color_layers::read(&mesh);
        store.layers.push(super::ColorLayer {
            name: "CornerGeom".to_owned(),
            domain,
            values,
        });
        super::color_layers::write(&mut mesh, &store);
        super::set_cache_color_data_type(&mut mesh, "CornerGeom")?;
        assert_eq!(
            mesh.attributes["blender_metadata"]["color_data_types"]["CornerGeom"],
            "BYTE_COLOR"
        );
        let corner_key = crate::geom::attribute_keys::storage_key("CornerGeom");
        assert!(mesh.attributes.contains_key(corner_key.as_ref()));
        let stored = super::color_layers::read(&mesh);
        let corner_layer = stored
            .layers
            .get(1)
            .ok_or("corner geom color layer is missing")?;
        assert_colors_close(&corner_layer.values, &[BLUE, GREEN, RED]);
        Ok(())
    }

    #[test]
    fn corner_byte_color_matches_blender_oracle() -> Result<(), Box<dyn std::error::Error>> {
        // `CornerCol` loops 0-3 of the first quad in Blender's oracle (tests/fixtures/alembic_cache/
        // oracle.json, case 0, frame 0). Blender writes BYTE_COLOR from the fixture's linear inputs
        // as sRGB bytes, so the file holds the encoded value of each input. Potter corner k is file
        // corner 3 - k, and the identity indices keep file order. The oracle values are binary32,
        // so the comparison allows 1e-5, about 80 binary32 epsilons.
        const ORACLE: [[f64; 4]; 4] = [
            [
                0.0,
                0.300_543_844_699_859_6,
                0.896_269_559_860_229_5,
                0.501_960_813_999_176,
            ],
            [
                0.049_706_574_529_409_41,
                0.300_543_844_699_859_6,
                0.846_873_462_200_164_8,
                0.549_019_634_723_663_3,
            ],
            [
                0.099_898_740_649_223_33,
                0.300_543_844_699_859_6,
                0.799_102_962_017_059_3,
                0.600_000_023_841_857_9,
            ],
            [
                0.149_959_817_528_724_67,
                0.300_543_844_699_859_6,
                0.752_942_442_893_981_9,
                0.650_980_412_960_052_5,
            ],
        ];
        let mesh = super::Mesh::from_positions_and_faces(
            vec![
                glam::DVec3::ZERO,
                glam::DVec3::X,
                glam::DVec3::ONE,
                glam::DVec3::Y,
            ],
            vec![vec![0, 1, 2, 3]],
        )?;
        let values: Vec<Vec<f64>> = (0..4)
            .rev()
            .map(|loop_index| {
                let k = f64::from(loop_index);
                vec![
                    crate::color::linear_to_srgb(0.05 * k),
                    crate::color::linear_to_srgb(0.3),
                    crate::color::linear_to_srgb(0.9 - 0.05 * k),
                    0.5 + 0.05 * k,
                ]
            })
            .collect();
        let color = raw_color("facevarying", values, Some(vec![0, 1, 2, 3]));
        let (_, actual) = super::cache_colors(&mesh, &color, &test_modifier()?)?;
        assert_colors_within(&actual, &ORACLE, 1e-5);
        Ok(())
    }

    #[test]
    fn unindexed_facevarying_is_identity_over_its_values() -> Result<(), Box<dyn std::error::Error>>
    {
        let mesh = super::Mesh::from_positions_and_faces(
            vec![
                glam::DVec3::ZERO,
                glam::DVec3::X,
                glam::DVec3::ONE,
                glam::DVec3::Y,
            ],
            vec![vec![0, 1, 2, 3]],
        )?;
        let modifier = test_modifier()?;
        let four = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 1.0, 1.0],
        ];
        // No `.indices`: the identity over four values covers four corners, so file corner c reads
        // value c, and Potter corner p is file corner 3 - p.
        let unindexed = raw_color("facevarying", four.clone(), None);
        let (_, values) = super::cache_colors(&mesh, &unindexed, &modifier)?;
        assert_colors_close(&values, &[WHITE, BLUE, GREEN, RED]);

        // An explicit but empty `.indices` has length 0, so facevarying does not apply and each
        // corner reads its vertex's value.
        let explicit_empty = raw_color("facevarying", four, Some(Vec::new()));
        let (_, values) = super::cache_colors(&mesh, &explicit_empty, &modifier)?;
        assert_colors_close(&values, &[RED, GREEN, BLUE, WHITE]);

        // Unindexed with three values covers three corners, not four, so the vertex path applies
        // and vertex 3 has no sample.
        let short = raw_color(
            "facevarying",
            vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
            None,
        );
        let (_, values) = super::cache_colors(&mesh, &short, &modifier)?;
        assert_colors_close(&values, &[RED, GREEN, BLUE, CLEAR]);
        Ok(())
    }

    proptest! {
        #[test]
        fn interpolated_position_matches_linear_reference(
            first in [-1_000.0_f64..1_000.0, -1_000.0_f64..1_000.0, -1_000.0_f64..1_000.0],
            second in [-1_000.0_f64..1_000.0, -1_000.0_f64..1_000.0, -1_000.0_f64..1_000.0],
            factor in 0.0_f64..=1.0,
        ) {
            let actual = lerp_position(first, second, factor);
            let [first_x, first_y, first_z] = first;
            let [second_x, second_y, second_z] = second;
            let expected = [
                first_x + (second_x - first_x) * factor,
                first_y + (second_y - first_y) * factor,
                first_z + (second_z - first_z) * factor,
            ];
            prop_assert_eq!(actual, expected);
        }
    }
}
