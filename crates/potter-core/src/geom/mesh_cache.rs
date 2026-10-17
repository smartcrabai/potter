use std::{
    collections::HashMap,
    sync::{Arc, LazyLock, Mutex},
};

use glam::DVec3;
use serde_json::{Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    exchange::alembic::{self, MeshCacheSample},
    model::Modifier,
    params::{self, ParameterFamily},
};

use super::{
    Mesh,
    color_layers::{self, ColorDomain, ColorLayer},
    uv_layers::{self, UvLayer},
};

static CACHE_SAMPLES: LazyLock<Mutex<HashMap<String, Arc<Vec<MeshCacheSample>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Replace a mesh with the requested Alembic cache sample.
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
) -> Result<()> {
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
    let read = read_params(modifier)?;
    let rate = f64::from(fps) / fps_base;
    if !rate.is_finite() || rate <= 0.0 {
        return Err(PotError::with_details(
            ErrorCode::EvaluationFailed,
            "Mesh Sequence Cache requires a finite positive scene frame rate",
            json!({"modifier_id":modifier.id,"fps":fps,"fps_base":fps_base}),
        ));
    }
    let cache_frame = override_frame.unwrap_or(frame);
    let time_seconds = (cache_frame - frame_offset) / rate;
    if !time_seconds.is_finite() {
        return Err(PotError::with_details(
            ErrorCode::EvaluationFailed,
            "Mesh Sequence Cache time is not finite",
            json!({"modifier_id":modifier.id,"frame":frame,"time_seconds":time_seconds}),
        ));
    }

    let key = format!(
        "{resource_hash}\0{object_path}\0{}\0{}\0{}",
        read.uv,
        read.color,
        velocity_scale != 0.0
    );
    let samples = {
        let mut parsed = CACHE_SAMPLES.lock().map_err(|_| {
            PotError::new(
                ErrorCode::InternalError,
                "Alembic mesh cache lock is poisoned",
            )
        })?;
        if let Some(samples) = parsed.get(&key) {
            Arc::clone(samples)
        } else {
            let samples = Arc::new(alembic::read_mesh_cache(
                archive_bytes,
                object_path,
                read.uv,
                read.color,
                velocity_scale != 0.0,
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
            samples
        }
    };
    if samples.is_empty() {
        return Err(PotError::with_details(
            ErrorCode::EvaluationFailed,
            "Alembic PolyMesh has no samples",
            json!({"modifier_id":modifier.id,"resource":resource,"object_path":object_path}),
        ));
    }
    let sample = sample_at_time(&samples, time_seconds, interpolate).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::EvaluationFailed,
            "Alembic PolyMesh has no samples",
            json!({"modifier_id":modifier.id,"resource":resource,"object_path":object_path}),
        )
    })?;
    if (!read.vertices || !read.polygons) && mesh.vertices.len() != sample.sample.positions.len() {
        return Err(PotError::with_details(
            ErrorCode::EvaluationFailed,
            "Alembic vertex count does not match the input mesh when topology replacement is disabled",
            json!({
                "modifier_id":modifier.id,
                "resource":resource,
                "object_path":object_path,
                "input_vertex_count":mesh.vertices.len(),
                "cache_vertex_count":sample.sample.positions.len(),
                "read_vertices":read.vertices,
                "read_polygons":read.polygons,
            }),
        ));
    }
    if !read.polygons {
        if read.vertices {
            for (index, vertex) in mesh.vertices.iter_mut().enumerate() {
                vertex.co = evaluated_position(&sample, index, modifier)?;
            }
        }
        apply_cache_attributes(mesh, sample.sample, read, velocity_scale, modifier)?;
        validate_cache_output(mesh, modifier)?;
        return Ok(());
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
    apply_cache_attributes(
        &mut evaluated,
        sample.sample,
        read,
        velocity_scale,
        modifier,
    )?;
    validate_cache_output(&evaluated, modifier)?;
    *mesh = evaluated;
    Ok(())
}

fn apply_cache_attributes(
    mesh: &mut Mesh,
    sample: &MeshCacheSample,
    read: ReadAttributes,
    velocity_scale: f64,
    modifier: &Modifier,
) -> Result<()> {
    if read.uv
        && let Some(uv_faces) = &sample.uv_faces
    {
        if uv_faces.len() != mesh.faces.len() {
            return Err(PotError::with_details(
                ErrorCode::EvaluationFailed,
                "Alembic UV face count does not match the selected mesh topology",
                json!({"modifier_id":modifier.id,"cache_faces":uv_faces.len(),"mesh_faces":mesh.faces.len()}),
            ));
        }
        if let Some((face, uv)) = mesh
            .faces
            .iter()
            .zip(uv_faces)
            .find(|(face, uv)| face.vertices.len() != uv.len())
        {
            return Err(PotError::with_details(
                ErrorCode::EvaluationFailed,
                "Alembic UV corner count does not match the selected face topology",
                json!({"modifier_id":modifier.id,"face_id":face.id,"cache_corners":uv.len(),"mesh_corners":face.vertices.len()}),
            ));
        }
        write_cache_uvs(mesh, modifier, uv_faces)?;
    }
    if read.color
        && let Some((scope, colors)) = &sample.colors
    {
        let (domain, values) = cache_colors(mesh, scope, colors, modifier)?;
        write_cache_colors(mesh, domain, values);
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

/// Writes the cache's corner UVs into the layer that the modifier names with `uv_layer`, or into
/// the render layer when it names none. A named layer the mesh lacks is created. The layer's values
/// and pins are replaced; its other layers are kept.
fn write_cache_uvs(mesh: &mut Mesh, modifier: &Modifier, uv_faces: &[Vec<[f64; 2]>]) -> Result<()> {
    let mut store = uv_layers::read(mesh);
    let name = match modifier.params.get("uv_layer").and_then(Value::as_str) {
        Some(name) => name.to_owned(),
        None => store.render.clone().unwrap_or_else(|| "UVMap".to_owned()),
    };
    let index = if let Some(index) = store.index_of(&name) {
        index
    } else {
        if store.layers.len() >= uv_layers::MAX_LAYERS {
            return Err(PotError::with_details(
                ErrorCode::LimitExceeded,
                "Alembic UV cache needs a UV layer beyond the mesh limit",
                json!({"feature_id":"uv.layer_limit","modifier_id":modifier.id}),
            ));
        }
        store
            .layers
            .push(UvLayer::new(name.clone(), vec![None; mesh.faces.len()]));
        if store.layers.len() == 1 {
            store.active = Some(name.clone());
            store.render = Some(name);
        }
        store.layers.len() - 1
    };
    let layer = store
        .layers
        .get_mut(index)
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "UV layer index is out of range"))?;
    layer.faces = uv_faces.iter().map(|uv| Some(uv.clone())).collect();
    layer.pinned = vec![None; mesh.faces.len()];
    uv_layers::write(mesh, &store);
    Ok(())
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

/// The color layer that Alembic colors fill. It shares the default layer name of vertex painting.
const CACHE_COLOR_LAYER: &str = "color";

/// The cache's colors as one layer's values in storage order. Point scopes give one value per vertex.
/// Uniform colors (one per face) become one value per corner, which the corner domain can hold.
/// Colors are stored unchanged, since Alembic colors are linear.
fn cache_colors(
    mesh: &Mesh,
    scope: &str,
    colors: &[Vec<f64>],
    modifier: &Modifier,
) -> Result<(ColorDomain, Vec<[f64; 4]>)> {
    match scope {
        "constant" => {
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
}

/// Writes the cache colors as the `color` layer of the color store, replacing a layer of that name.
/// The layer becomes the active or render layer only when the store has none.
fn write_cache_colors(mesh: &mut Mesh, domain: ColorDomain, values: Vec<[f64; 4]>) {
    let mut store = color_layers::read(mesh);
    store.layers.retain(|layer| layer.name != CACHE_COLOR_LAYER);
    store.layers.push(ColorLayer {
        name: CACHE_COLOR_LAYER.to_owned(),
        domain,
        values,
    });
    if store.render_index().is_none() {
        store.render = Some(CACHE_COLOR_LAYER.to_owned());
    }
    if store
        .active
        .as_deref()
        .and_then(|active| store.index_of(active))
        .is_none()
    {
        store.active = Some(CACHE_COLOR_LAYER.to_owned());
    }
    color_layers::write(mesh, &store);
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
    fn cache_colors_map_to_point_and_corner_mesh_domains() -> Result<(), Box<dyn std::error::Error>>
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
        let (domain, values) = super::cache_colors(&mesh, "vertex", &point_values, &modifier)?;
        assert_eq!(domain, super::ColorDomain::Point);
        super::write_cache_colors(&mut mesh, domain, values);
        let point = super::color_layers::read(&mesh);
        let point_layer = point.layers.first().ok_or("point color layer is missing")?;
        assert_eq!(point_layer.name, "color");
        assert_eq!(point_layer.domain, super::ColorDomain::Point);
        assert_eq!(point_layer.values[0], [1.0, 0.0, 0.0, 1.0]);

        let corner_values = [
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let (domain, values) =
            super::cache_colors(&mesh, "facevarying", &corner_values, &modifier)?;
        assert_eq!(domain, super::ColorDomain::Corner);
        super::write_cache_colors(&mut mesh, domain, values);
        let corner = super::color_layers::read(&mesh);
        let corner_layer = corner
            .layers
            .first()
            .ok_or("corner color layer is missing")?;
        assert_eq!(corner_layer.name, "color");
        assert_eq!(corner_layer.domain, super::ColorDomain::Corner);
        assert_eq!(
            corner_layer.values,
            vec![
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
            ]
        );
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
