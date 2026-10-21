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
        alembic::{self, CacheColorLayer, MeshCacheSample},
    },
    model::Modifier,
    params::{self, ParameterFamily},
};

use super::{
    Mesh,
    clnor::{self, NormalCodeOptions, SetterInput},
    color_layers::{self, ColorDomain, ColorLayer},
    normals::{sharp_edge_flags, smooth_face_flags},
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
    let mut read = read_params(modifier)?;
    let rate = f64::from(fps) / fps_base;
    if !rate.is_finite() || rate <= 0.0 {
        return Err(PotError::with_details(
            ErrorCode::EvaluationFailed,
            "Mesh Sequence Cache requires a finite positive scene frame rate",
            json!({"modifier_id":modifier.id,"fps":fps,"fps_base":fps_base}),
        ));
    }
    let cache_frame = override_frame.unwrap_or(frame);
    // Blender's file-sequence time is the frame itself (BKE_cachefile_time_offset); a single file
    // takes seconds with the frame offset removed.
    let time_seconds = if is_sequence {
        cache_frame
    } else {
        (cache_frame - frame_offset) / rate
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
        let input_corners = mesh
            .faces
            .iter()
            .map(|face| face.vertices.len())
            .sum::<usize>();
        let cache_corners = sample.sample.faces.iter().map(Vec::len).sum::<usize>();
        let dimensions_changed = mesh.vertices.len() != sample.sample.positions.len()
            || mesh.faces.len() != sample.sample.faces.len()
            || input_corners != cache_corners;
        dimensions_changed
            || ((is_sequence || sample.sample.has_multiple_topology_samples)
                && !topology_matches(mesh, sample.sample.positions.len(), &sample.sample.faces))
    };
    if topology_changed {
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
    let mut warnings = Vec::new();
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
        let ranks = blender_edge_ranks(&evaluated.faces);
        sort_edges_by_rank(&mut evaluated, &ranks);
        // Blender's custom-normal setter splits the fans of smooth faces to mark sharp edges, so the
        // replaced mesh takes its smooth flags from the cache normals before the normals are encoded.
        if let Some(normals) = &sample.sample.normals {
            cache_smooth_faces(&mut evaluated, normals);
        }
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
    // With unchanged topology and POLY disabled, Blender updates selected streams on the receiver.
    if !read.polygons {
        if read.vertices {
            let positions = sample
                .sample
                .positions
                .iter()
                .enumerate()
                .map(|(index, _)| evaluated_position(&sample, index, modifier))
                .collect::<Result<Vec<_>>>()?;
            for (vertex, position) in mesh.vertices.iter_mut().zip(positions) {
                vertex.co = position;
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
        return Ok(warnings);
    }
    let positions = if read.vertices {
        (0..sample.sample.positions.len())
            .map(|index| evaluated_position(&sample, index, modifier))
            .collect::<Result<Vec<_>>>()?
    } else {
        mesh.vertices.iter().map(|vertex| vertex.co).collect()
    };
    let mut evaluated = Mesh::from_positions_and_faces(positions, sample.sample.faces.clone())
        .map_err(|error| {
            PotError::with_details(
                ErrorCode::EvaluationFailed,
                "Alembic cache topology is invalid for the selected vertex sample",
                json!({"modifier_id":modifier.id,"reason":error.to_string()}),
            )
        })?;
    // A same-topology read keeps the receiver's edge order, which the receiver's sharp flags index.
    let receiver_ranks: HashMap<[u32; 2], usize> = mesh
        .edges
        .iter()
        .enumerate()
        .map(|(index, edge)| (sorted_pair(edge.vertices), index))
        .collect();
    sort_edges_by_rank(&mut evaluated, &receiver_ranks);
    carry_uv_store(mesh, &mut evaluated);
    carry_shading(mesh, &mut evaluated)?;
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
    Ok(warnings)
}

fn topology_matches(mesh: &Mesh, cache_vertex_count: usize, cache_faces: &[Vec<usize>]) -> bool {
    let corner_count = mesh
        .faces
        .iter()
        .map(|face| face.vertices.len())
        .sum::<usize>();
    if mesh.vertices.is_empty()
        || mesh.vertices.len() != cache_vertex_count
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
    let velocity_name = if read_velocity { velocity_name } else { "" };
    // Every flag the parser reads must key the cache: normals follow `read.polygons`.
    let key = format!(
        "{resource_hash}\0{object_path}\0{}\0{}\0{}\0{}\0{}",
        read.uv, read.color, read.polygons, read_velocity, velocity_name
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
        velocity_name,
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
    velocity_factor: f64,
    modifier: &Modifier,
    warnings: &mut Vec<PotError>,
) -> Result<()> {
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
            read.polygons,
            sample.uv_primary.as_deref(),
            warnings,
        )?;
    }
    if read.color {
        for layer in &sample.colors {
            check_cache_color_layer(mesh, modifier, layer)?;
            write_cache_colors(mesh, &layer.name, layer.domain, layer.values.clone());
        }
    }
    if read.polygons
        && let Some(normals) = &sample.normals
    {
        if normals.len()
            != mesh
                .faces
                .iter()
                .map(|face| face.vertices.len())
                .sum::<usize>()
        {
            return Err(PotError::with_details(
                ErrorCode::EvaluationFailed,
                "Alembic corner normal count does not match the selected mesh topology",
                json!({"modifier_id":modifier.id,"normal_count":normals.len()}),
            ));
        }
        // Cache normals are authored per corner, so every corner takes its vector.
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
    // Blender stores velocity as a point vector, and skips a sample whose count differs from the mesh.
    if velocity_factor != 0.0
        && let Some((scope, velocities)) = &sample.velocities
        && matches!(scope.as_str(), "vertex" | "varying")
        && velocities.len() == mesh.vertices.len()
    {
        let mut values = serde_json::Map::new();
        for (vertex, velocity) in mesh.vertices.iter().zip(velocities) {
            let scaled = velocity
                .iter()
                .map(|component| component * velocity_factor)
                .collect::<Vec<_>>();
            if scaled.iter().any(|value| !value.is_finite()) {
                return Err(PotError::with_details(
                    ErrorCode::EvaluationFailed,
                    "Alembic velocity scaling produced a non-finite value",
                    json!({"modifier_id":modifier.id,"velocity_factor":velocity_factor}),
                ));
            }
            values.insert(format!("v{}", vertex.id), json!(scaled));
        }
        mesh.attributes.insert(
            "alembic_velocity".to_owned(),
            json!({"domain":"point","type":"float3","values":values}),
        );
    }
    Ok(())
}

/// Carries the input mesh's UV layers onto the replacement mesh that a topology read builds. Rows
/// are per face and corner, so they keep their values only while the corner layout is unchanged.
/// Otherwise the layers keep their names, order, and selectors, with no values.
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

/// Blender's edge order on a mesh built from faces: one edge per corner `(previous corner, this corner)`
/// in corner order, kept at its first appearance (`mesh_calc_edges`). Maps each edge's sorted vertex pair
/// to its rank.
fn blender_edge_ranks(faces: &[crate::geom::Face]) -> HashMap<[u32; 2], usize> {
    let mut ranks = HashMap::new();
    for face in faces {
        let count = face.vertices.len();
        for corner in 0..count {
            let previous = face.vertices[(corner + count - 1) % count];
            let next_rank = ranks.len();
            ranks
                .entry(sorted_pair([previous, face.vertices[corner]]))
                .or_insert(next_rank);
        }
    }
    ranks
}

/// Puts a mesh's edges in the order of `ranks`, keeping each edge's ID. Edges without a rank go last.
fn sort_edges_by_rank(mesh: &mut Mesh, ranks: &HashMap<[u32; 2], usize>) {
    mesh.edges.sort_by_key(|edge| {
        ranks
            .get(&sorted_pair(edge.vertices))
            .copied()
            .unwrap_or(usize::MAX)
    });
}

fn sorted_pair(vertices: [u32; 2]) -> [u32; 2] {
    [vertices[0].min(vertices[1]), vertices[0].max(vertices[1])]
}

/// Carries the receiver's shading onto a replacement with the same faces. Blender keeps a receiver's
/// `sharp_face` and `sharp_edge` through a same-topology read, so the custom-normal setter sees them.
/// Faces and edges map by index: a same-topology read keeps the receiver's edge order.
fn carry_shading(input: &Mesh, replacement: &mut Mesh) -> Result<()> {
    if input.attributes.contains_key("shade_smooth") && input.faces.len() == replacement.faces.len()
    {
        let flags = smooth_face_flags(input)?;
        let values: serde_json::Map<String, Value> = replacement
            .faces
            .iter()
            .zip(flags)
            .map(|(face, flag)| (format!("f{}", face.id), json!(flag)))
            .collect();
        replacement.attributes.insert(
            "shade_smooth".to_owned(),
            json!({"domain": "face", "type": "bool", "values": values}),
        );
    }
    if input.attributes.contains_key("sharp") && input.edges.len() == replacement.edges.len() {
        let flags = sharp_edge_flags(input)?;
        let values: serde_json::Map<String, Value> = replacement
            .edges
            .iter()
            .zip(flags)
            .map(|(edge, flag)| (format!("e{}", edge.id), json!(flag)))
            .collect();
        replacement.attributes.insert(
            "sharp".to_owned(),
            json!({"domain": "edge", "type": "bool", "values": values}),
        );
    }
    Ok(())
}

/// Writes each cached UV layer into the layer its name gives, creating the layer when the mesh
/// lacks it. Alembic UV face-corner rows are reversed into Potter mesh-loop order whether or not
/// cache polygons replace the input faces. A layer's values are replaced; its pins and the other
/// layers are kept. A cached layer with no room left under the mesh limit is reported and not
/// written. The active and render names keep their layers while those exist; otherwise they name
/// the first layer the cache wrote.
/// Without polygons, the active layer (`primary`) takes no values; it is created at zero when missing
/// and otherwise keeps its values. Blender's UV-only read writes every other layer.
fn write_cache_uvs(
    mesh: &mut Mesh,
    modifier: &Modifier,
    cached: &[ExchangeUvLayer],
    polygons: bool,
    primary: Option<&str>,
    warnings: &mut Vec<PotError>,
) -> Result<()> {
    let mut store = uv_layers::read(mesh);
    let mut first_written = None;
    for cached_layer in cached {
        let index = match store.index_of(&cached_layer.name) {
            Some(index) => index,
            None if store.layers.len() < uv_layers::MAX_LAYERS => {
                // A layer the read creates starts at Blender's zero UV, which Potter stores with V flipped.
                let zeros = mesh
                    .faces
                    .iter()
                    .map(|face| Some(vec![[0.0, 1.0]; face.vertices.len()]))
                    .collect();
                store
                    .layers
                    .push(UvLayer::new(cached_layer.name.clone(), zeros));
                store.layers.len() - 1
            }
            None => {
                warnings.push(layer_limit_warning(modifier, &cached_layer.name));
                continue;
            }
        };
        // Blender's UV-only read keeps the active layer's values (created at zero) and writes the others.
        if !polygons && primary == Some(cached_layer.name.as_str()) {
            continue;
        }
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
            json!({
                "modifier_id":modifier.id,
                "layer":layer.name,
                "cache_faces":layer.corners.len(),
                "mesh_faces":mesh.faces.len(),
            }),
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
            json!({
                "modifier_id":modifier.id,
                "layer":layer.name,
                "face_id":face.id,
                "cache_corners":uv.len(),
                "mesh_corners":face.vertices.len(),
            }),
        ));
    }
    Ok(())
}

/// The warning for a cached UV layer the mesh has no room for. The layer is not written.
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

/// Blender's custom-normal setter marks a face flat when every corner normal is its face normal, so a
/// face is smooth unless all its corners match it within float32 noise.
fn cache_smooth_faces(mesh: &mut Mesh, normals: &[[f64; 3]]) {
    let corner_count: usize = mesh.faces.iter().map(|face| face.vertices.len()).sum();
    if normals.len() != corner_count {
        // The corner count is checked where the normals are encoded, which reports the mismatch.
        return;
    }
    let positions = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.co))
        .collect::<HashMap<_, _>>();
    let flat_tolerance = 1.0 - 8.0 * f64::from(f32::EPSILON);
    let mut offset = 0;
    let mut smooth = serde_json::Map::new();
    for face in &mesh.faces {
        let count = face.vertices.len();
        let corners = &normals[offset..offset + count];
        offset += count;
        let points = face
            .vertices
            .iter()
            .map(|id| positions.get(id).copied().unwrap_or(DVec3::ZERO))
            .collect::<Vec<_>>();
        let face_normal = super::normals::face_normal(&points);
        let flat = corners
            .iter()
            .all(|normal| DVec3::from_array(*normal).dot(face_normal) >= flat_tolerance);
        smooth.insert(format!("f{}", face.id), json!(!flat));
    }
    mesh.attributes.insert(
        "shade_smooth".to_owned(),
        json!({"domain":"face","type":"bool","values":smooth}),
    );
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

/// Rejects a cache color layer whose value count does not match the receiver's domain.
fn check_cache_color_layer(
    mesh: &Mesh,
    modifier: &Modifier,
    layer: &CacheColorLayer,
) -> Result<()> {
    let expected = match layer.domain {
        ColorDomain::Point => mesh.vertices.len(),
        ColorDomain::Corner => mesh
            .faces
            .iter()
            .map(|face| face.vertices.len())
            .sum::<usize>(),
    };
    if layer.values.len() != expected {
        return Err(invalid_color_count(modifier, layer.values.len(), expected));
    }
    Ok(())
}

/// Writes one cache color layer, replacing a layer of the same name. The layer becomes the active or
/// render layer only when the store has none.
fn write_cache_colors(mesh: &mut Mesh, name: &str, domain: ColorDomain, values: Vec<[f64; 4]>) {
    let mut store = color_layers::read(mesh);
    store.layers.retain(|layer| layer.name != name);
    store.layers.push(ColorLayer {
        name: name.to_owned(),
        domain,
        values,
    });
    if store.render_index().is_none() {
        store.render = Some(name.to_owned());
    }
    if store
        .active
        .as_deref()
        .and_then(|active| store.index_of(active))
        .is_none()
    {
        store.active = Some(name.to_owned());
    }
    color_layers::write(mesh, &store);
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
    fn cache_color_layers_keep_their_name_and_domain() -> Result<(), Box<dyn std::error::Error>> {
        let mut mesh = super::Mesh::from_positions_and_faces(
            vec![glam::DVec3::ZERO, glam::DVec3::X, glam::DVec3::Y],
            vec![vec![0, 1, 2]],
        )?;
        super::write_cache_colors(
            &mut mesh,
            "Point",
            super::ColorDomain::Point,
            vec![[1.0, 0.0, 0.0, 1.0]; 3],
        );
        super::write_cache_colors(
            &mut mesh,
            "CornerCol",
            super::ColorDomain::Corner,
            vec![[0.0, 1.0, 0.0, 1.0]; 3],
        );
        let store = super::color_layers::read(&mesh);
        let point = store
            .layers
            .iter()
            .find(|layer| layer.name == "Point")
            .ok_or("point color layer is missing")?;
        assert_eq!(point.domain, super::ColorDomain::Point);
        let corner = store
            .layers
            .iter()
            .find(|layer| layer.name == "CornerCol")
            .ok_or("corner color layer is missing")?;
        assert_eq!(corner.domain, super::ColorDomain::Corner);
        assert_eq!(corner.values[0], [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(store.render.as_deref(), Some("Point"));
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
