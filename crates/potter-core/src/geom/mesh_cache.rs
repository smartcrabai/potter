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

use super::Mesh;

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
    let frame_offset = number_param(modifier, "frame_offset", 0.0)?;
    let velocity_scale = number_param(modifier, "velocity_scale", 0.0)?;
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
    let interpolate = bool_param(modifier, "use_vertex_interpolation", false)?;
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
        mesh.attributes.insert(
            "uv_map".to_owned(),
            Value::Array(
                mesh.faces
                    .iter()
                    .zip(uv_faces)
                    .map(|(face, uv)| json!({"face_id":face.id,"uv":uv}))
                    .collect(),
            ),
        );
    }
    if read.color
        && let Some((scope, colors)) = &sample.colors
    {
        let attribute = color_attribute(mesh, scope, colors, modifier)?;
        mesh.attributes.insert("color".to_owned(), attribute);
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

fn validate_cache_output(mesh: &Mesh, modifier: &Modifier) -> Result<()> {
    mesh.validate().map_err(|error| {
        PotError::with_details(
            ErrorCode::EvaluationFailed,
            "Alembic cache produced an invalid mesh",
            json!({"modifier_id":modifier.id,"reason":error.to_string()}),
        )
    })
}

fn color_attribute(
    mesh: &Mesh,
    scope: &str,
    colors: &[Vec<f64>],
    modifier: &Modifier,
) -> Result<Value> {
    let mut values = serde_json::Map::new();
    let domain = match scope {
        "constant" => {
            let color = colors
                .first()
                .ok_or_else(|| invalid_color_count(modifier, 0, 1))?;
            for vertex in &mesh.vertices {
                values.insert(format!("v{}", vertex.id), rgba_color(color, modifier)?);
            }
            "point"
        }
        "vertex" | "varying" => {
            if colors.len() != mesh.vertices.len() {
                return Err(invalid_color_count(
                    modifier,
                    colors.len(),
                    mesh.vertices.len(),
                ));
            }
            for (vertex, color) in mesh.vertices.iter().zip(colors) {
                values.insert(format!("v{}", vertex.id), rgba_color(color, modifier)?);
            }
            "point"
        }
        "uniform" => {
            if colors.len() != mesh.faces.len() {
                return Err(invalid_color_count(
                    modifier,
                    colors.len(),
                    mesh.faces.len(),
                ));
            }
            for (face, color) in mesh.faces.iter().zip(colors) {
                values.insert(format!("f{}", face.id), rgba_color(color, modifier)?);
            }
            "face"
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
            let mut colors = colors.iter();
            for face in &mesh.faces {
                let face_colors = colors
                    .by_ref()
                    .take(face.vertices.len())
                    .map(|color| rgba_color(color, modifier))
                    .collect::<Result<Vec<_>>>()?;
                values.insert(format!("f{}", face.id), Value::Array(face_colors));
            }
            "corner"
        }
        _ => {
            return Err(PotError::with_details(
                ErrorCode::EvaluationFailed,
                "Alembic cache color scope is unsupported",
                json!({"modifier_id":modifier.id,"scope":scope}),
            ));
        }
    };
    Ok(json!({"domain":domain,"type":"color","values":values}))
}

fn rgba_color(color: &[f64], modifier: &Modifier) -> Result<Value> {
    let rgba = match color {
        [red, green, blue] => [*red, *green, *blue, 1.0],
        [red, green, blue, alpha] => [*red, *green, *blue, *alpha],
        _ => {
            return Err(PotError::with_details(
                ErrorCode::EvaluationFailed,
                "Alembic cache color tuple must have three or four components",
                json!({"modifier_id":modifier.id,"component_count":color.len()}),
            ));
        }
    };
    Ok(json!(rgba))
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

fn number_param(modifier: &Modifier, name: &str, default: f64) -> Result<f64> {
    super::modifiers::number_param(modifier, name, default)
        .map_err(|_| invalid_parameter(modifier, name, "a finite number"))
}

fn bool_param(modifier: &Modifier, name: &str, default: bool) -> Result<bool> {
    super::modifiers::bool_param(modifier, name, default)
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
    use serde_json::{Value, json};

    use super::lerp_position;

    #[test]
    fn cache_colors_map_to_point_and_corner_mesh_domains() -> Result<(), Box<dyn std::error::Error>>
    {
        let mesh = super::Mesh::from_positions_and_faces(
            vec![glam::DVec3::ZERO, glam::DVec3::X, glam::DVec3::Y],
            vec![vec![0, 1, 2]],
        )?;
        let modifier = crate::model::Modifier {
            id: crate::model::Id::new("cache")?,
            modifier_type: "mesh_sequence_cache".to_owned(),
            name: "Mesh Sequence Cache".to_owned(),
            enabled: true,
            params: serde_json::Map::new(),
            binding_data: None,
            runtime: crate::model::ModifierRuntime::default(),
        };
        let point = super::color_attribute(
            &mesh,
            "vertex",
            &[
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
            &modifier,
        )?;
        assert_eq!(point.get("domain").and_then(Value::as_str), Some("point"));
        let point_values = point
            .get("values")
            .and_then(Value::as_object)
            .ok_or("point colors have no values")?;
        assert_eq!(point_values.get("v0"), Some(&json!([1.0, 0.0, 0.0, 1.0])));

        let corner = super::color_attribute(
            &mesh,
            "facevarying",
            &[
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
            &modifier,
        )?;
        assert_eq!(corner.get("domain").and_then(Value::as_str), Some("corner"));
        let face_id = mesh
            .faces
            .first()
            .ok_or("test triangle face is missing")?
            .id;
        let corner_values = corner
            .get("values")
            .and_then(Value::as_object)
            .and_then(|values| values.get(&format!("f{face_id}")))
            .and_then(Value::as_array)
            .ok_or("corner colors have no face value array")?;
        assert_eq!(
            Value::Array(corner_values.clone()),
            json!([
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0]
            ])
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
