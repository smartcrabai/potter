use std::{
    collections::HashMap,
    sync::{Arc, LazyLock, Mutex},
};

use glam::DVec3;
use serde_json::{Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    exchange::{ExchangeUvLayer, alembic::{self, MeshCacheSample}},
    model::Modifier,
    params::{self, ParameterFamily},
};

use super::{
    clnor::{self, NormalCodeOptions, SetterInput},
    Mesh,
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
    let sequence = modifier
        .params
        .get("is_sequence")
        .and_then(Value::as_bool)
        .unwrap_or(false);
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
    let time_seconds = if sequence {
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
        mesh.vertices.len() != sample.sample.positions.len()
            || !topology_matches(mesh, &sample.sample.faces)
    };
    if topology_changed {
        read = ReadAttributes {
            vertices: true,
            polygons: true,
            uv: true,
            color: true,
        };
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

fn topology_matches(mesh: &Mesh, cache_faces: &[Vec<usize>]) -> bool {
    let corner_count = mesh.faces.iter().map(|face| face.vertices.len()).sum::<usize>();
    if mesh.vertices.len() == 0
        || mesh.faces.len() != cache_faces.len()
        || corner_count != cache_faces.iter().map(Vec::len).sum::<usize>()
    {
        return false;
    }
    let indices = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<HashMap<_, _>>();
    mesh.faces.iter().zip(cache_faces).all(|(face, cache_face)| {
        face.vertices.len() == cache_face.len()
            && face.vertices.iter().zip(cache_face).all(|(id, index)| {
                indices.get(id).is_some_and(|mesh_index| mesh_index == index)
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
        read.uv,
        read.color,
        read.polygons,
        velocity_parameter
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
        let corner_count = mesh.faces.iter().map(|face| face.vertices.len()).sum::<usize>();
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
            let (domain, values) = cache_colors(mesh, &color.scope, &color.values, modifier)?;
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
        if !sample.colors.is_empty() {
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

/// Carries UV layers to a topology replacement, clearing rows only when the face layout differs.
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
    let mut layers = uv_layers::read_layers(mesh);
    for cached_layer in cached {
        let existing = layers
            .iter()
            .position(|layer| layer.name == cached_layer.name);
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
            None if layers.len() < uv_layers::MAX_LAYERS => {
                layers.push(UvLayer::new(
                    cached_layer.name.clone(),
                    vec![None; mesh.faces.len()],
                ));
                layers.len() - 1
            }
            None => {
                warnings.push(layer_limit_warning(modifier, &cached_layer.name));
                continue;
            }
        };
        let layer = layers.get_mut(index).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "UV layer index is out of range")
        })?;
        layer.faces = cached_layer
            .corners
            .iter()
            .map(|corners| Some(corners.iter().rev().copied().collect()))
            .collect();
    }
    uv_layers::write_layers(mesh, &layers);
    Ok(())
}

fn check_cache_uv_layer(
    mesh: &Mesh,
    modifier: &Modifier,
    layer: &ExchangeUvLayer,
) -> Result<()> {
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

fn cache_colors(
    mesh: &Mesh,
    scope: &str,
    colors: &[Vec<f64>],
    modifier: &Modifier,
) -> Result<(ColorDomain, Vec<[f64; 4]>)> {
    let corner_count = mesh
        .faces
        .iter()
        .map(|face| face.vertices.len())
        .sum::<usize>();
    let expected = match scope {
        "constant" => 1,
        "vertex" | "varying" => mesh.vertices.len(),
        "uniform" => mesh.faces.len(),
        "facevarying" => corner_count,
        _ => {
            return Err(PotError::with_details(
                ErrorCode::EvaluationFailed,
                "Alembic cache color scope is unsupported",
                json!({"modifier_id":modifier.id,"scope":scope}),
            ));
        }
    };
    if colors.len() != expected {
        return Err(invalid_color_count(modifier, colors.len(), expected));
    }
    let vertex_indices = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<HashMap<_, _>>();
    let mut values = Vec::with_capacity(corner_count);
    let mut corner_offset = 0;
    for (face_index, face) in mesh.faces.iter().enumerate() {
        for vertex_id in &face.vertices {
            let vertex_index = vertex_indices.get(vertex_id).copied().ok_or_else(|| {
                PotError::new(
                    ErrorCode::EvaluationFailed,
                    "Alembic color face references a missing vertex",
                )
            })?;
            let color_index = match scope {
                "constant" => 0,
                "vertex" | "varying" => vertex_index,
                "uniform" => face_index,
                "facevarying" => corner_offset,
                _ => {
                    return Err(PotError::with_details(
                        ErrorCode::EvaluationFailed,
                        "Alembic cache color scope is unsupported",
                        json!({"modifier_id":modifier.id,"scope":scope}),
                    ));
                }
            };
            let color = colors
                .first()
                .ok_or_else(|| invalid_color_count(modifier, 0, 1))?;
            let color = rgba_color(color, modifier)?;
            Ok((ColorDomain::Point, vec![color; mesh.vertices.len()]))
        }
        "vertex" | "varying" => {
            if colors.len() != mesh.vertices.len() {
                return Err(invalid_color_count(
                    modifier,
                    colors.len(),
                    mesh.vertices.len(),
                ));
            }
            let values = colors
                .iter()
                .map(|color| rgba_color(color, modifier))
                .collect::<Result<Vec<_>>>()?;
            Ok((ColorDomain::Point, values))
        }
        "uniform" => {
            if colors.len() != mesh.faces.len() {
                return Err(invalid_color_count(
                    modifier,
                    colors.len(),
                    mesh.faces.len(),
                ));
            }
            let mut values = Vec::new();
            for (face, color) in mesh.faces.iter().zip(colors) {
                let color = rgba_color(color, modifier)?;
                values.extend(vec![color; face.vertices.len()]);
            }
            Ok((ColorDomain::Corner, values))
        }
        "facevarying" => {
            let corner_count = mesh
                .faces
                .iter()
                .map(|face| face.vertices.len())
                .sum::<usize>();
            if colors.len() != corner_count {
                return Err(invalid_color_count(modifier, colors.len(), corner_count));
            }
            let values = colors
                .iter()
                .map(|color| rgba_color(color, modifier))
                .collect::<Result<Vec<_>>>()?;
            Ok((ColorDomain::Corner, values))
        }
        _ => Err(PotError::with_details(
            ErrorCode::EvaluationFailed,
            "Alembic cache color scope is unsupported",
            json!({"modifier_id":modifier.id,"scope":scope}),
        )),
    }
    Ok((ColorDomain::Corner, values))
}

fn rgba_color(color: &[f64], modifier: &Modifier) -> Result<[f64; 4]> {
    match color {
        [red, green, blue] => Ok([*red, *green, *blue, 1.0]),
        [red, green, blue, alpha] => Ok([*red, *green, *blue, *alpha]),
        _ => Err(PotError::with_details(
            ErrorCode::EvaluationFailed,
            "Alembic cache color tuple must have three or four components",
            json!({"modifier_id":modifier.id,"component_count":color.len()}),
        )),
    }
}

fn invalid_color_count(modifier: &Modifier, actual: usize, expected: usize) -> PotError {
    PotError::with_details(
        ErrorCode::EvaluationFailed,
        "Alembic color value count does not match the selected mesh domain",
        json!({"modifier_id":modifier.id,"actual_count":actual,"expected_count":expected}),
    )
}

#[derive(Clone, Copy)]
struct ReadAttributes {
    vertices: bool,
    polygons: bool,
    uv: bool,
    color: bool,
}

fn read_params(modifier: &Modifier) -> Result<ReadAttributes> {
    let Some(value) = modifier.params.get("read_data").or_else(|| {
        params::default_value(
            ParameterFamily::Modifier,
            &modifier.modifier_type,
            "read_data",
        )
    }) else {
        return Ok(ReadAttributes {
            vertices: true,
            polygons: true,
            uv: true,
            color: true,
        });
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

    use super::lerp_position;

    #[test]
    fn cache_color_scopes_use_named_linear_corner_store() -> Result<(), Box<dyn std::error::Error>>
    {
        let mut mesh = super::Mesh::from_positions_and_faces(
            vec![glam::DVec3::ZERO, glam::DVec3::X, glam::DVec3::Y],
            vec![vec![0, 1, 2]],
        )?;
        let modifier = crate::model::Modifier {
            id: crate::model::Id::new("cache")?,
            modifier_type: "mesh_sequence_cache".to_owned(),
            name: "Mesh Sequence Cache".to_owned(),
            enabled: true,
            show_viewport: true,
            show_render: true,
            params: serde_json::Map::new(),
            binding_data: None,
            runtime: crate::model::ModifierRuntime::default(),
        };
        let point_values = [
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let (domain, values) =
            super::cache_colors(&mesh, "vertex", &point_values, &modifier)?;
        assert_eq!(domain, super::ColorDomain::Corner);
        let mut store = super::color_layers::read(&mesh);
        store.layers.push(super::ColorLayer {
            name: "PointGeom".to_owned(),
            domain,
            values: values.clone(),
        });
        super::color_layers::write(&mut mesh, &store);
        super::set_cache_color_data_type(&mut mesh, "PointGeom")?;
        assert_eq!(
            mesh.attributes["blender_metadata"]["color_data_types"]["PointGeom"],
            "BYTE_COLOR"
        );
        let point_key = super::attribute_keys::storage_key("PointGeom");
        assert!(mesh.attributes.contains_key(point_key.as_ref()));
        let point = super::color_layers::read(&mesh);
        let point_layer = point.layers.first().ok_or("point geom color layer is missing")?;
        assert_eq!(point_layer.domain, super::ColorDomain::Corner);
        assert_eq!(
            point_layer.values,
            vec![
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
            ]
        );

        let corner_values = [
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let (domain, values) =
            super::cache_colors(&mesh, "facevarying", &corner_values, &modifier)?;
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
        let corner_key = super::attribute_keys::storage_key("CornerGeom");
        assert!(mesh.attributes.contains_key(corner_key.as_ref()));
        let corner = super::color_layers::read(&mesh);
        let corner_layer = corner.layers.get(1).ok_or("corner geom color layer is missing")?;
        assert_eq!(corner_layer.domain, super::ColorDomain::Corner);
        assert_eq!(corner_layer.values, point_layer.values);
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
