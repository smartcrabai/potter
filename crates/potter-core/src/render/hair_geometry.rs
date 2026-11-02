use glam::{DMat4, DVec3};
use serde_json::Value;

use crate::model::HairCurvesData;

use super::hair_intersect::{CurveKey, HairCurveGeometry, HairCurveRenderMode, HairCurveShape};

const CATMULL_ROM: u8 = 0;
const POLY: u8 = 1;
const BEZIER: u8 = 2;
const NURBS: u8 = 3;
const MINIMUM_TWIST: u8 = 0;
const Z_UP: u8 = 1;
const FREE: u8 = 2;

#[derive(Debug)]
pub(crate) enum HairCurveGeometryError {
    Attribute(&'static str),
    Curve(usize),
    Transform,
}

/// Prepares Cycles' path geometry from raw Hair Curves point keys.
///
/// Point-domain normals are evaluated from authored curve types before applying the global path basis.
pub(crate) fn prepare_cycles_hair_curve_geometry(
    data: &HairCurvesData,
    object_to_world: DMat4,
    scene_render_mode: HairCurveRenderMode,
    ribbon_subdivisions: usize,
) -> Result<Vec<PreparedHairCurve>, HairCurveGeometryError> {
    let determinant = object_to_world.determinant();
    if !object_to_world
        .to_cols_array()
        .iter()
        .all(|value| value.is_finite())
        || !determinant.is_finite()
        || determinant.partial_cmp(&0.0) == Some(std::cmp::Ordering::Equal)
        || object_to_world.x_axis.w.partial_cmp(&0.0) != Some(std::cmp::Ordering::Equal)
        || object_to_world.y_axis.w.partial_cmp(&0.0) != Some(std::cmp::Ordering::Equal)
        || object_to_world.z_axis.w.partial_cmp(&0.0) != Some(std::cmp::Ordering::Equal)
        || object_to_world.w_axis.w.partial_cmp(&1.0) != Some(std::cmp::Ordering::Equal)
    {
        return Err(HairCurveGeometryError::Transform);
    }
    let world_to_object = object_to_world.inverse();
    if !world_to_object
        .to_cols_array()
        .iter()
        .all(|value| value.is_finite())
    {
        return Err(HairCurveGeometryError::Transform);
    }

    let point_count = data
        .curves
        .iter()
        .try_fold(0usize, |count, curve| count.checked_add(curve.points.len()))
        .ok_or(HairCurveGeometryError::Attribute("point count"))?;
    let curve_count = data.curves.len();
    let curve_type = curve_attr(data, "curve_type", "CURVE", "INT8", curve_count)?;
    let first_curve_is_poly = curve_type
        .and_then(|values| values.first())
        .and_then(Value::as_i64)
        .is_some_and(|value| value == i64::from(POLY));
    let render_mode = if first_curve_is_poly {
        HairCurveRenderMode::ThickLinear
    } else {
        scene_render_mode
    };
    let cyclic = curve_attr(data, "cyclic", "CURVE", "BOOLEAN", curve_count)?;
    let resolution = curve_attr(data, "resolution", "CURVE", "INT", curve_count)?;
    let normal_mode = curve_attr(data, "normal_mode", "CURVE", "INT8", curve_count)?;
    let radius = curve_attr(data, "radius", "POINT", "FLOAT", point_count)?;
    let tilt = curve_attr(data, "tilt", "POINT", "FLOAT", point_count)?;
    let left = curve_attr(data, "handle_left", "POINT", "FLOAT_VECTOR", point_count)?;
    let right = curve_attr(data, "handle_right", "POINT", "FLOAT_VECTOR", point_count)?;
    let left_type = curve_attr(data, "handle_type_left", "POINT", "INT8", point_count)?;
    let right_type = curve_attr(data, "handle_type_right", "POINT", "INT8", point_count)?;
    let custom_normals = curve_vec3_attr(data, "custom_normal", point_count)?;

    let mut prepared = Vec::new();
    prepared
        .try_reserve_exact(curve_count)
        .map_err(|_| HairCurveGeometryError::Attribute("curve allocation"))?;
    let mut point_offset = 0usize;
    for (curve_index, curve) in data.curves.iter().enumerate() {
        let count = curve.points.len();
        if !curve.radius.is_finite() || curve.radius < 0.0 {
            return Err(HairCurveGeometryError::Curve(curve_index));
        }
        if count == 0 {
            prepared.push(PreparedHairCurve {
                curve: HairCurveGeometry {
                    shape: if render_mode == HairCurveRenderMode::ThickLinear {
                        HairCurveShape::Linear {
                            keys: Vec::new(),
                            cyclic: false,
                        }
                    } else {
                        HairCurveShape::CatmullRom {
                            keys: Vec::new(),
                            cyclic: false,
                        }
                    },
                    render_mode,
                    ribbon_subdivisions,
                    object_to_world,
                    world_to_object,
                },
                point_normals: HairCurvePointNormals {
                    normals: Vec::new(),
                    cyclic: false,
                },
            });
            continue;
        }
        let mut keys = Vec::new();
        keys.try_reserve_exact(count)
            .map_err(|_| HairCurveGeometryError::Attribute("point allocation"))?;
        for (point_index, point) in curve.points.iter().enumerate() {
            let position = DVec3::from_array(*point);
            let point_radius = match radius {
                Some(values) => values[point_offset + point_index]
                    .as_f64()
                    .ok_or(HairCurveGeometryError::Curve(curve_index))?,
                None => curve.radius,
            };
            if !position.is_finite() || !point_radius.is_finite() || point_radius < 0.0 {
                return Err(HairCurveGeometryError::Curve(curve_index));
            }
            keys.push(CurveKey {
                position,
                radius: point_radius,
            });
        }

        let spline_type = match curve_type {
            Some(values) => values[curve_index]
                .as_i64()
                .ok_or(HairCurveGeometryError::Curve(curve_index))?,
            None => i64::from(CATMULL_ROM),
        };
        let cyclic = match cyclic {
            Some(values) => values[curve_index]
                .as_bool()
                .ok_or(HairCurveGeometryError::Curve(curve_index))?,
            None => false,
        };
        let resolution = match resolution {
            Some(values) => values[curve_index]
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .filter(|value| *value > 0)
                .ok_or(HairCurveGeometryError::Curve(curve_index))?,
            None => 12,
        };
        let normal_mode = match normal_mode {
            Some(values) => values[curve_index]
                .as_i64()
                .and_then(|value| u8::try_from(value).ok())
                .ok_or(HairCurveGeometryError::Curve(curve_index))?,
            None => MINIMUM_TWIST,
        };
        if !matches!(normal_mode, MINIMUM_TWIST | Z_UP | FREE) {
            return Err(HairCurveGeometryError::Curve(curve_index));
        }
        let tilts = point_values_f64(tilt, point_offset, count, 0.0, curve_index)?;
        let curve_custom_normals = custom_normals
            .as_ref()
            .map(|values| values[point_offset..point_offset + count].to_vec());
        if !matches!(spline_type, 0..=3) {
            return Err(HairCurveGeometryError::Curve(curve_index));
        }
        let (bezier_spans, bezier_left_types, bezier_right_types) = if spline_type
            == i64::from(BEZIER)
        {
            let handle_left = point_vec3_values(left, point_offset, &keys, curve_index)?;
            let handle_right = point_vec3_values(right, point_offset, &keys, curve_index)?;
            let left_types = point_values_i64(left_type, point_offset, count, 1, curve_index)?;
            let right_types = point_values_i64(right_type, point_offset, count, 1, curve_index)?;
            let spans = make_bezier_spans(
                &keys,
                &handle_left,
                &handle_right,
                &left_types,
                &right_types,
                cyclic,
                curve_index,
            )?;
            (Some(spans), Some(left_types), Some(right_types))
        } else {
            (None, None, None)
        };
        let point_normals = curve_point_normals_from_source(
            &keys,
            spline_type,
            cyclic,
            resolution,
            normal_mode,
            &tilts,
            curve_custom_normals.as_deref(),
            bezier_spans.as_deref(),
            bezier_left_types.as_deref(),
            bezier_right_types.as_deref(),
        )
        .ok_or(HairCurveGeometryError::Curve(curve_index))?;
        let shape = if render_mode == HairCurveRenderMode::ThickLinear {
            HairCurveShape::Linear {
                keys,
                cyclic: false,
            }
        } else {
            HairCurveShape::CatmullRom {
                keys,
                cyclic: false,
            }
        };
        prepared.push(PreparedHairCurve {
            curve: HairCurveGeometry {
                shape,
                render_mode,
                ribbon_subdivisions,
                object_to_world,
                world_to_object,
            },
            point_normals: HairCurvePointNormals {
                normals: point_normals,
                cyclic,
            },
        });
        point_offset += count;
    }
    Ok(prepared)
}

fn curve_attr<'a>(
    data: &'a HairCurvesData,
    name: &'static str,
    domain: &'static str,
    data_type: &'static str,
    expected_len: usize,
) -> Result<Option<&'a [Value]>, HairCurveGeometryError> {
    let Some(attribute) = data.attributes.get(name) else {
        return Ok(None);
    };
    if attribute.domain != domain
        || attribute.data_type != data_type
        || attribute.values.len() != expected_len
    {
        return Err(HairCurveGeometryError::Attribute(name));
    }
    Ok(Some(&attribute.values))
}

fn curve_vec3_attr(
    data: &HairCurvesData,
    name: &'static str,
    expected_len: usize,
) -> Result<Option<Vec<DVec3>>, HairCurveGeometryError> {
    let Some(values) = curve_attr(data, name, "POINT", "FLOAT_VECTOR", expected_len)? else {
        return Ok(None);
    };
    values
        .iter()
        .map(|value| {
            let array = value
                .as_array()
                .filter(|array| array.len() == 3)
                .ok_or(HairCurveGeometryError::Attribute(name))?;
            let coordinates = [
                array[0]
                    .as_f64()
                    .ok_or(HairCurveGeometryError::Attribute(name))?,
                array[1]
                    .as_f64()
                    .ok_or(HairCurveGeometryError::Attribute(name))?,
                array[2]
                    .as_f64()
                    .ok_or(HairCurveGeometryError::Attribute(name))?,
            ];
            let vector = DVec3::from_array(coordinates);
            vector
                .is_finite()
                .then_some(vector)
                .ok_or(HairCurveGeometryError::Attribute(name))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

fn point_vec3_values(
    values: Option<&[Value]>,
    offset: usize,
    keys: &[CurveKey],
    curve_index: usize,
) -> Result<Vec<DVec3>, HairCurveGeometryError> {
    let Some(values) = values else {
        return Ok(keys.iter().map(|key| key.position).collect());
    };
    values[offset..offset + keys.len()]
        .iter()
        .map(|value| {
            let coordinates = value
                .as_array()
                .filter(|array| array.len() == 3)
                .and_then(|array| {
                    Some([array[0].as_f64()?, array[1].as_f64()?, array[2].as_f64()?])
                })
                .ok_or(HairCurveGeometryError::Curve(curve_index))?;
            let vector = DVec3::from_array(coordinates);
            vector
                .is_finite()
                .then_some(vector)
                .ok_or(HairCurveGeometryError::Curve(curve_index))
        })
        .collect()
}

fn point_values_f64(
    values: Option<&[Value]>,
    offset: usize,
    count: usize,
    default: f64,
    curve_index: usize,
) -> Result<Vec<f64>, HairCurveGeometryError> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|_| HairCurveGeometryError::Attribute("point allocation"))?;
    for index in 0..count {
        let value = match values {
            Some(values) => values[offset + index]
                .as_f64()
                .ok_or(HairCurveGeometryError::Curve(curve_index))?,
            None => default,
        };
        if !value.is_finite() {
            return Err(HairCurveGeometryError::Curve(curve_index));
        }
        result.push(value);
    }
    Ok(result)
}

fn point_values_i64(
    values: Option<&[Value]>,
    offset: usize,
    count: usize,
    default: i64,
    curve_index: usize,
) -> Result<Vec<i64>, HairCurveGeometryError> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|_| HairCurveGeometryError::Attribute("point allocation"))?;
    for index in 0..count {
        let value = match values {
            Some(values) => values[offset + index]
                .as_i64()
                .ok_or(HairCurveGeometryError::Curve(curve_index))?,
            None => default,
        };
        if !(0..=3).contains(&value) {
            return Err(HairCurveGeometryError::Curve(curve_index));
        }
        result.push(value);
    }
    Ok(result)
}

fn calculate_bezier_handles(
    keys: &[CurveKey],
    source_left: &[DVec3],
    source_right: &[DVec3],
    left_types: &[i64],
    right_types: &[i64],
    cyclic: bool,
    curve_index: usize,
) -> Result<(Vec<DVec3>, Vec<DVec3>), HairCurveGeometryError> {
    if keys.len() != source_left.len()
        || keys.len() != source_right.len()
        || keys.len() != left_types.len()
        || keys.len() != right_types.len()
    {
        return Err(HairCurveGeometryError::Attribute("Bezier handles"));
    }
    let mut left = source_left.to_vec();
    let mut right = source_right.to_vec();
    if keys.len() < 2 {
        return Ok((left, right));
    }
    for index in 0..keys.len() {
        let point = keys[index].position;
        let previous = if cyclic {
            keys[(index + keys.len() - 1) % keys.len()].position
        } else if index > 0 {
            keys[index - 1].position
        } else {
            2.0 * point - keys[1].position
        };
        let next = if cyclic {
            keys[(index + 1) % keys.len()].position
        } else if index + 1 < keys.len() {
            keys[index + 1].position
        } else {
            2.0 * point - keys[index - 1].position
        };
        let mut to_point = point - previous;
        let mut from_point = next - point;
        let mut left_length = to_point.length();
        let mut right_length = from_point.length();
        if left_length == 0.0 {
            left_length = 1.0;
        }
        if right_length == 0.0 {
            right_length = 1.0;
        }
        if left_types[index] == 1 || right_types[index] == 1 {
            let tangent = to_point / left_length + from_point / right_length;
            let tangent_length = tangent.length() * 2.5614;
            if tangent_length != 0.0 {
                left_length = left_length.min(5.0 * right_length);
                right_length = right_length.min(5.0 * left_length);
                if left_types[index] == 1 {
                    left[index] = point - tangent * (left_length / tangent_length);
                }
                if right_types[index] == 1 {
                    right[index] = point + tangent * (right_length / tangent_length);
                }
            }
        }
        to_point = point - previous;
        from_point = next - point;
        if left_types[index] == 2 {
            left[index] = point - to_point / 3.0;
        }
        if right_types[index] == 2 {
            right[index] = point + from_point / 3.0;
        }
        let left_handle_length = (left[index] - point).length();
        let right_handle_length = (right[index] - point).length();
        if left_types[index] == 3 && right_handle_length > 0.0 {
            left[index] =
                point - (right[index] - point) * (left_handle_length / right_handle_length);
        }
        if right_types[index] == 3 && left_handle_length > 0.0 {
            right[index] =
                point - (left[index] - point) * (right_handle_length / left_handle_length);
        }
        if !left[index].is_finite() || !right[index].is_finite() {
            return Err(HairCurveGeometryError::Curve(curve_index));
        }
    }
    Ok((left, right))
}

fn make_bezier_spans(
    keys: &[CurveKey],
    left: &[DVec3],
    right: &[DVec3],
    left_types: &[i64],
    right_types: &[i64],
    cyclic: bool,
    curve_index: usize,
) -> Result<Vec<[CurveKey; 4]>, HairCurveGeometryError> {
    let (handle_left, handle_right) = calculate_bezier_handles(
        keys,
        left,
        right,
        left_types,
        right_types,
        cyclic,
        curve_index,
    )?;
    let segment_count = if cyclic {
        keys.len()
    } else {
        keys.len().saturating_sub(1)
    };
    let mut spans = Vec::new();
    spans
        .try_reserve_exact(segment_count)
        .map_err(|_| HairCurveGeometryError::Attribute("Bezier allocation"))?;
    for segment in 0..segment_count {
        let next = (segment + 1) % keys.len();
        spans.push([
            keys[segment],
            CurveKey {
                position: handle_right[segment],
                radius: keys[segment].radius,
            },
            CurveKey {
                position: handle_left[next],
                radius: keys[next].radius,
            },
            keys[next],
        ]);
    }
    Ok(spans)
}

#[derive(Clone, Debug)]
pub(crate) struct HairCurvePointNormals {
    normals: Vec<DVec3>,
    cyclic: bool,
}

impl HairCurvePointNormals {
    /// Interpolates the source point-domain `std_vertex_normal` attribute without normalization.
    pub(crate) fn at(&self, span: usize, u: f64) -> Option<DVec3> {
        if !u.is_finite() {
            return None;
        }
        let first = *self.normals.get(span)?;
        let next = match span
            .checked_add(1)
            .and_then(|index| self.normals.get(index))
        {
            Some(normal) => *normal,
            None if self.cyclic => *self.normals.first()?,
            None => return None,
        };
        Some(first.lerp(next, u.clamp(0.0, 1.0)))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct HairCurveShadowTransparency {
    values: Option<Vec<f32>>,
    cyclic: bool,
}

impl HairCurveShadowTransparency {
    pub(crate) fn new(values: Option<Vec<f32>>, cyclic: bool) -> Self {
        Self { values, cyclic }
    }

    /// Samples Cycles' per-key shadow transparency without an extra clamp.
    pub(crate) fn at(&self, span: usize, u: f64) -> Option<f64> {
        if !u.is_finite() {
            return None;
        }
        let Some(values) = self.values.as_deref() else {
            return Some(0.0);
        };
        let first = *values.get(span)?;
        let next = match span.checked_add(1).and_then(|index| values.get(index)) {
            Some(value) => *value,
            None if self.cyclic => *values.first()?,
            None => return None,
        };
        let u = u as f32;
        (u.is_finite()).then_some(f64::from(first * (1.0 - u) + next * u))
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedHairCurve {
    pub(crate) curve: HairCurveGeometry,
    pub(crate) point_normals: HairCurvePointNormals,
}

fn bezier_scalar(start: f64, end: f64, u: f64) -> f64 {
    start * (1.0 - u) + end * u
}

fn bezier_vector(start: DVec3, end: DVec3, u: f64) -> DVec3 {
    start.lerp(end, u)
}

fn source_bezier_handle_differs(handle: DVec3, position: DVec3) -> bool {
    let handle = handle.to_array().map(|value| value as f32);
    let position = position.to_array().map(|value| value as f32);
    (0..3).any(|component| {
        (handle[component] - position[component]).abs() > 1.0e-6 * handle[component].abs()
    })
}

fn source_bezier_handle_direction(from: DVec3, to: DVec3) -> DVec3 {
    let from = from.to_array().map(|value| value as f32);
    let to = to.to_array().map(|value| value as f32);
    let direction = [to[0] - from[0], to[1] - from[1], to[2] - from[2]];
    let length =
        (direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2])
            .sqrt();
    if length > 0.0 {
        DVec3::new(
            f64::from(direction[0] / length),
            f64::from(direction[1] / length),
            f64::from(direction[2] / length),
        )
    } else {
        DVec3::ZERO
    }
}

fn curve_point_normals_from_source(
    keys: &[CurveKey],
    curve_type: i64,
    cyclic: bool,
    resolution: usize,
    normal_mode: u8,
    tilts: &[f64],
    custom_normals: Option<&[DVec3]>,
    bezier_spans: Option<&[[CurveKey; 4]]>,
    bezier_left_types: Option<&[i64]>,
    bezier_right_types: Option<&[i64]>,
) -> Option<Vec<DVec3>> {
    let key_count = keys.len();
    if key_count == 0 {
        return Some(Vec::new());
    }
    // Blender's POINT normal field treats NURBS controls as poly keys and copies FREE normals raw.
    if curve_type == i64::from(NURBS) && normal_mode == FREE {
        let mut normals = Vec::new();
        normals.try_reserve_exact(key_count).ok()?;
        if let Some(custom_normals) = custom_normals {
            normals.extend_from_slice(custom_normals);
        } else {
            normals.resize(key_count, DVec3::Z);
        }
        return Some(normals);
    }
    let source_positions: Vec<_> = keys.iter().map(|key| key.position).collect();
    if curve_type == i64::from(NURBS) {
        let zero_tilts = vec![0.0; key_count];
        return curve_normals(
            &source_positions,
            POLY,
            cyclic,
            1,
            normal_mode,
            &zero_tilts,
            None,
            &[],
        );
    }

    let sample_resolution = resolution.max(1);
    let mut bezier_key_offsets = None;
    let mut tangent_overrides = [(0_usize, DVec3::ZERO); 2];
    let mut tangent_override_count = 0;
    let (evaluated_positions, evaluated_tilts, evaluated_custom_normals, key_stride) =
        match curve_type {
            value if value == i64::from(CATMULL_ROM) => {
                let segment_count = if cyclic && key_count > 1 {
                    key_count
                } else {
                    key_count - 1
                };
                let sample_count = segment_count.checked_mul(sample_resolution)?;
                let sample_count = if cyclic {
                    sample_count.max(1)
                } else {
                    sample_count.checked_add(1)?
                };
                let mut positions = Vec::new();
                positions.try_reserve_exact(sample_count).ok()?;
                positions.resize(sample_count, DVec3::ZERO);
                evaluate_catmull_rom(
                    &source_positions,
                    cyclic,
                    sample_resolution,
                    &mut positions,
                    catmull_vec3,
                );
                let mut evaluated_tilts = Vec::new();
                evaluated_tilts.try_reserve_exact(sample_count).ok()?;
                evaluated_tilts.resize(sample_count, 0.0);
                evaluate_catmull_rom(
                    tilts,
                    cyclic,
                    sample_resolution,
                    &mut evaluated_tilts,
                    catmull_f64,
                );
                let evaluated_custom = if let Some(custom_normals) = custom_normals {
                    let mut evaluated = Vec::new();
                    evaluated.try_reserve_exact(sample_count).ok()?;
                    evaluated.resize(sample_count, DVec3::ZERO);
                    evaluate_catmull_rom(
                        custom_normals,
                        cyclic,
                        sample_resolution,
                        &mut evaluated,
                        catmull_vec3,
                    );
                    Some(evaluated)
                } else {
                    None
                };
                (
                    positions,
                    evaluated_tilts,
                    evaluated_custom,
                    sample_resolution,
                )
            }
            value if value == i64::from(POLY) => (
                source_positions,
                tilts.to_vec(),
                custom_normals.map(<[DVec3]>::to_vec),
                1,
            ),
            value if value == i64::from(BEZIER) => {
                let spans = bezier_spans?;
                if spans.is_empty() {
                    (
                        source_positions,
                        tilts.to_vec(),
                        custom_normals.map(<[DVec3]>::to_vec),
                        1,
                    )
                } else {
                    let left_types = bezier_left_types?;
                    let right_types = bezier_right_types?;
                    if left_types.len() != key_count || right_types.len() != key_count {
                        return None;
                    }
                    let mut key_offsets = Vec::new();
                    key_offsets.try_reserve_exact(key_count).ok()?;
                    let mut sample_count = 0_usize;
                    for (segment, right_type) in right_types.iter().enumerate().take(spans.len()) {
                        key_offsets.push(sample_count);
                        let next = (segment + 1) % key_count;
                        let segment_samples = if *right_type == 2 && left_types[next] == 2 {
                            1
                        } else {
                            sample_resolution
                        };
                        sample_count = sample_count.checked_add(segment_samples)?;
                    }
                    if !cyclic {
                        key_offsets.push(sample_count);
                        sample_count = sample_count.checked_add(1)?;
                    }
                    bezier_key_offsets = Some(key_offsets);

                    let mut positions = Vec::new();
                    positions.try_reserve_exact(sample_count).ok()?;
                    let mut evaluated_tilts = Vec::new();
                    evaluated_tilts.try_reserve_exact(sample_count).ok()?;
                    let mut evaluated_custom = custom_normals.map(|_| Vec::new());
                    if let Some(custom) = &mut evaluated_custom {
                        custom.try_reserve_exact(sample_count).ok()?;
                    }
                    for (segment, span) in spans.iter().enumerate() {
                        let next = (segment + 1) % key_count;
                        let segment_samples = if right_types[segment] == 2 && left_types[next] == 2
                        {
                            1
                        } else {
                            sample_resolution
                        };
                        for step in 0..segment_samples {
                            let u = step as f64 / segment_samples as f64;
                            positions.push(super::hair_intersect::bezier_position(*span, u)?);
                            evaluated_tilts.push(bezier_scalar(tilts[segment], tilts[next], u));
                            if let (Some(source), Some(destination)) =
                                (custom_normals, evaluated_custom.as_mut())
                            {
                                destination.push(bezier_vector(source[segment], source[next], u));
                            }
                        }
                    }
                    if !cyclic {
                        positions.push((spans.last()?)[3].position);
                        evaluated_tilts.push(*tilts.last()?);
                        if let (Some(source), Some(destination)) =
                            (custom_normals, evaluated_custom.as_mut())
                        {
                            destination.push(*source.last()?);
                        }

                        let first_position = keys.first()?.position;
                        let first_handle = spans.first()?[1].position;
                        if source_bezier_handle_differs(first_handle, first_position) {
                            tangent_overrides[tangent_override_count] = (
                                0,
                                source_bezier_handle_direction(first_position, first_handle),
                            );
                            tangent_override_count += 1;
                        }
                        let last_position = keys.last()?.position;
                        let last_handle = spans.last()?[2].position;
                        if source_bezier_handle_differs(last_handle, last_position) {
                            let last_offset = *bezier_key_offsets.as_ref()?.last()?;
                            tangent_overrides[tangent_override_count] = (
                                last_offset,
                                source_bezier_handle_direction(last_handle, last_position),
                            );
                            tangent_override_count += 1;
                        }
                    }
                    (positions, evaluated_tilts, evaluated_custom, 1)
                }
            }
            _ => return None,
        };
    let evaluated_normals = curve_normals(
        &evaluated_positions,
        POLY,
        cyclic,
        1,
        normal_mode,
        &evaluated_tilts,
        evaluated_custom_normals.as_deref(),
        &tangent_overrides[..tangent_override_count],
    )?;
    let mut normals = Vec::new();
    normals.try_reserve_exact(key_count).ok()?;
    for key_index in 0..key_count {
        let normal_index = if let Some(offsets) = &bezier_key_offsets {
            *offsets.get(key_index)?
        } else {
            key_index.checked_mul(key_stride)?
        };
        normals.push(*evaluated_normals.get(normal_index)?);
    }
    Some(normals)
}

#[derive(Default)]
pub(crate) struct CurveNormalsScratch {
    evaluated_positions: Vec<DVec3>,
    tangents: Vec<DVec3>,
    normals: Vec<DVec3>,
    evaluated_tilt: Vec<f64>,
}

pub(crate) fn curve_normals(
    positions: &[DVec3],
    curve_type: u8,
    cyclic: bool,
    resolution: usize,
    normal_mode: u8,
    tilt: &[f64],
    custom_normals: Option<&[DVec3]>,
    tangent_overrides: &[(usize, DVec3)],
) -> Option<Vec<DVec3>> {
    let mut result = Vec::new();
    result.try_reserve_exact(positions.len()).ok()?;
    result.resize(positions.len(), DVec3::ZERO);
    let mut scratch = CurveNormalsScratch::default();
    curve_normals_into(
        positions,
        curve_type,
        cyclic,
        resolution,
        normal_mode,
        tilt,
        custom_normals,
        tangent_overrides,
        &mut scratch,
        &mut result,
    )
    .then_some(result)
}

pub(crate) fn curve_normals_into(
    positions: &[DVec3],
    curve_type: u8,
    cyclic: bool,
    resolution: usize,
    normal_mode: u8,
    tilt: &[f64],
    custom_normals: Option<&[DVec3]>,
    tangent_overrides: &[(usize, DVec3)],
    scratch: &mut CurveNormalsScratch,
    output: &mut [DVec3],
) -> bool {
    if positions.is_empty()
        || output.len() != positions.len()
        || tilt.len() != positions.len()
        || custom_normals.is_some_and(|normals| normals.len() != positions.len())
        || tangent_overrides
            .iter()
            .any(|(index, tangent)| *index >= positions.len() || !tangent.is_finite())
        || !matches!(normal_mode, MINIMUM_TWIST | Z_UP | FREE)
    {
        return false;
    }

    let (evaluated_resolution, evaluated_count) = match curve_type {
        CATMULL_ROM => {
            let per_segment = resolution.max(1);
            let segment_count = if cyclic && positions.len() > 1 {
                positions.len()
            } else {
                positions.len() - 1
            };
            let Some(point_count) = segment_count.checked_mul(per_segment) else {
                return false;
            };
            let point_count = if cyclic {
                point_count.max(1)
            } else if let Some(point_count) = point_count.checked_add(1) {
                point_count
            } else {
                return false;
            };
            (per_segment, point_count)
        }
        POLY => (0, positions.len()),
        _ => return false,
    };

    if curve_type == CATMULL_ROM {
        if !resize_filled(
            &mut scratch.evaluated_positions,
            evaluated_count,
            DVec3::ZERO,
        ) {
            return false;
        }
        evaluate_catmull_rom(
            positions,
            cyclic,
            evaluated_resolution,
            &mut scratch.evaluated_positions,
            catmull_vec3,
        );
    } else {
        scratch.evaluated_positions.clear();
    }
    let evaluated_positions = if curve_type == CATMULL_ROM {
        &scratch.evaluated_positions[..]
    } else {
        positions
    };

    if !resize_filled(&mut scratch.tangents, evaluated_count, DVec3::ZERO)
        || !resize_filled(&mut scratch.normals, evaluated_count, DVec3::ZERO)
    {
        return false;
    }
    calculate_tangents(evaluated_positions, cyclic, &mut scratch.tangents);
    for (index, tangent) in tangent_overrides {
        scratch.tangents[*index] = *tangent;
    }

    match normal_mode {
        Z_UP => calculate_normals_z_up(&scratch.tangents, &mut scratch.normals),
        MINIMUM_TWIST => calculate_normals_minimum(&scratch.tangents, cyclic, &mut scratch.normals),
        FREE => {
            if let Some(custom_normals) = custom_normals {
                if curve_type == CATMULL_ROM {
                    evaluate_catmull_rom(
                        custom_normals,
                        cyclic,
                        evaluated_resolution,
                        &mut scratch.normals,
                        catmull_vec3,
                    );
                } else {
                    scratch.normals.copy_from_slice(custom_normals);
                }
                for normal in &mut scratch.normals {
                    *normal = normalize(*normal);
                }
            } else {
                calculate_normals_z_up(&scratch.tangents, &mut scratch.normals);
            }
        }
        _ => return false,
    }

    if tilt.iter().any(|angle| *angle != 0.0) {
        if curve_type == CATMULL_ROM {
            if !resize_filled(&mut scratch.evaluated_tilt, evaluated_count, 0.0) {
                return false;
            }
            evaluate_catmull_rom(
                tilt,
                cyclic,
                evaluated_resolution,
                &mut scratch.evaluated_tilt,
                catmull_f64,
            );
            for ((normal, tangent), angle) in scratch
                .normals
                .iter_mut()
                .zip(&scratch.tangents)
                .zip(&scratch.evaluated_tilt)
            {
                *normal = rotate_around_axis(*normal, *tangent, *angle);
            }
        } else {
            for ((normal, tangent), angle) in
                scratch.normals.iter_mut().zip(&scratch.tangents).zip(tilt)
            {
                *normal = rotate_around_axis(*normal, *tangent, *angle);
            }
        }
    }

    for (index, normal) in output.iter_mut().enumerate() {
        let evaluated_index = if curve_type == CATMULL_ROM {
            let Some(index) = resolution.checked_mul(index) else {
                return false;
            };
            index
        } else {
            index
        };
        let Some(evaluated_normal) = scratch.normals.get(evaluated_index) else {
            return false;
        };
        *normal = *evaluated_normal;
    }
    true
}

fn resize_filled<T: Clone>(buffer: &mut Vec<T>, length: usize, value: T) -> bool {
    if length > buffer.len() && buffer.try_reserve_exact(length - buffer.len()).is_err() {
        return false;
    }
    buffer.resize(length, value);
    true
}

fn catmull_basis(parameter: f64) -> [f64; 4] {
    let t = parameter;
    let s = 1.0 - parameter;
    [
        -t * s * s,
        2.0 + t * t * (3.0 * t - 5.0),
        2.0 + s * s * (3.0 * s - 5.0),
        -s * t * t,
    ]
}

fn catmull_vec3(a: DVec3, b: DVec3, c: DVec3, d: DVec3, parameter: f64) -> DVec3 {
    let [wa, wb, wc, wd] = catmull_basis(parameter);
    (a * wa + b * wb + c * wc + d * wd) * 0.5
}

fn catmull_f64(a: f64, b: f64, c: f64, d: f64, parameter: f64) -> f64 {
    let [wa, wb, wc, wd] = catmull_basis(parameter);
    (wa * a + wb * b + wc * c + wd * d) * 0.5
}

fn evaluate_catmull_rom<T: Copy>(
    source: &[T],
    cyclic: bool,
    resolution: usize,
    destination: &mut [T],
    interpolate: fn(T, T, T, T, f64) -> T,
) {
    if source.len() == 1 {
        destination[0] = source[0];
        return;
    }

    let segment_count = if cyclic {
        source.len()
    } else {
        source.len() - 1
    };
    for segment in 0..segment_count {
        let (a, b, c, d) = if source.len() == 2 {
            if cyclic && segment == 1 {
                (source[1], source[1], source[0], source[0])
            } else {
                (source[0], source[0], source[1], source[1])
            }
        } else if cyclic {
            (
                source[(segment + source.len() - 1) % source.len()],
                source[segment],
                source[(segment + 1) % source.len()],
                source[(segment + 2) % source.len()],
            )
        } else {
            (
                source[segment.saturating_sub(1)],
                source[segment],
                source[segment + 1],
                source[(segment + 2).min(source.len() - 1)],
            )
        };
        let start = segment * resolution;
        destination[start] = b;
        for sample in 1..resolution {
            destination[start + sample] =
                interpolate(a, b, c, d, sample as f64 / resolution as f64);
        }
    }
    if !cyclic && let (Some(destination), Some(source)) = (destination.last_mut(), source.last()) {
        *destination = *source;
    }
}

fn delta_direction(position: DVec3, next: DVec3) -> Option<DVec3> {
    let delta = next - position;
    let length = delta.length();
    if length < 1.0e-9 {
        None
    } else {
        Some(delta / length)
    }
}

fn direction_bisect(
    position: DVec3,
    next: DVec3,
    other_direction: &mut DVec3,
    is_equal: &mut bool,
) -> DVec3 {
    let previous_equal = *is_equal;
    let next_delta = next - position;
    let next_length = next_delta.length();
    *is_equal = next_length < 1.0e-9;
    if *is_equal {
        return *other_direction;
    }

    let previous_direction = *other_direction;
    *other_direction = next_delta / next_length;
    if previous_equal {
        return *other_direction;
    }

    let tangent = previous_direction + *other_direction;
    let length = tangent.length();
    if length < 0.662_761_9 {
        if length < 2.0e-7 {
            return *other_direction;
        }
        let binormal = other_direction.cross(previous_direction);
        let normal = *other_direction - previous_direction;
        normalize(binormal.cross(normal))
    } else {
        tangent / length
    }
}

fn calculate_tangents(positions: &[DVec3], cyclic: bool, tangents: &mut [DVec3]) {
    if positions.len() == 1 {
        tangents[0] = DVec3::Z;
        return;
    }

    let Some((first_valid_index, first_valid_tangent)) =
        (0..positions.len() - 1).find_map(|index| {
            delta_direction(positions[index], positions[index + 1])
                .map(|direction| (index, direction))
        })
    else {
        tangents.fill(DVec3::Z);
        return;
    };
    tangents[..=first_valid_index].fill(first_valid_tangent);

    let mut previous_direction = tangents[first_valid_index];
    let mut previous_equal = false;
    for index in (first_valid_index + 1)..positions.len() - 1 {
        tangents[index] = direction_bisect(
            positions[index],
            positions[index + 1],
            &mut previous_direction,
            &mut previous_equal,
        );
    }

    if cyclic {
        let last = positions.len() - 1;
        tangents[last] = direction_bisect(
            positions[last],
            positions[0],
            &mut previous_direction,
            &mut previous_equal,
        );
        tangents[0] = direction_bisect(
            positions[0],
            positions[1],
            &mut previous_direction,
            &mut previous_equal,
        );
    } else {
        let last = positions.len() - 1;
        tangents[last] =
            delta_direction(positions[last - 1], positions[last]).unwrap_or(previous_direction);
    }
}

fn normalize(vector: DVec3) -> DVec3 {
    let length = vector.length();
    if length > 0.0 {
        vector / length
    } else {
        vector
    }
}

fn calculate_normals_z_up(tangents: &[DVec3], normals: &mut [DVec3]) {
    for (tangent, normal) in tangents.iter().zip(normals) {
        *normal = if tangent.x.abs() + tangent.y.abs() < 1.0e-4 {
            DVec3::X
        } else {
            normalize(DVec3::new(tangent.y, -tangent.x, 0.0))
        };
    }
}

fn rotate_around_axis(vector: DVec3, axis: DVec3, angle: f64) -> DVec3 {
    let (sin, cos) = angle.sin_cos();
    vector * cos + axis.cross(vector) * sin + axis * (axis.dot(vector) * (1.0 - cos))
}

fn calculate_next_normal(last_normal: DVec3, last_tangent: DVec3, current_tangent: DVec3) -> DVec3 {
    if last_tangent == DVec3::ZERO || current_tangent == DVec3::ZERO {
        return last_normal;
    }
    let angle = last_tangent.dot(current_tangent).clamp(-1.0, 1.0).acos();
    if angle == 0.0 {
        return last_normal;
    }
    let axis = normalize(last_tangent.cross(current_tangent));
    if axis == DVec3::ZERO {
        last_normal
    } else {
        normalize(rotate_around_axis(last_normal, axis, angle))
    }
}

fn calculate_normals_minimum(tangents: &[DVec3], cyclic: bool, normals: &mut [DVec3]) {
    if tangents.is_empty() {
        return;
    }
    calculate_normals_z_up(&tangents[..1], &mut normals[..1]);
    for index in 1..normals.len() {
        normals[index] =
            calculate_next_normal(normals[index - 1], tangents[index - 1], tangents[index]);
    }
    if !cyclic {
        return;
    }

    let Some((&last_normal, &last_tangent)) = normals.last().zip(tangents.last()) else {
        return;
    };
    let uncorrected_last = calculate_next_normal(last_normal, last_tangent, tangents[0]);
    let first = normals[0];
    let cross = first.cross(uncorrected_last);
    let mut correction_angle = tangents[0].dot(cross).atan2(first.dot(uncorrected_last));
    if correction_angle > std::f64::consts::PI {
        correction_angle -= std::f64::consts::TAU;
    }
    let angle_step = correction_angle / normals.len() as f64;
    for (index, (normal, tangent)) in normals.iter_mut().zip(tangents).enumerate() {
        if *tangent != DVec3::ZERO {
            *normal = rotate_around_axis(*normal, *tangent, angle_step * index as f64);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::error::Error;

    use glam::{DMat4, DVec3};
    use serde_json::json;

    use crate::model::{HairCurve, HairCurveAttribute, HairCurvesData};

    use super::super::hair_intersect::HairCurveRenderMode;
    use super::{
        BEZIER, FREE, HairCurveShadowTransparency, HairCurveShape, MINIMUM_TWIST, NURBS, Z_UP,
        curve_normals, prepare_cycles_hair_curve_geometry,
    };

    #[test]
    fn straight_z_up_poly_strand_uses_x_normals() -> Result<(), Box<dyn Error>> {
        let positions = [DVec3::ZERO, DVec3::Z, DVec3::Z * 2.0];
        let normals = curve_normals(&positions, 1, false, 1, 1, &[0.0; 3], None, &[])
            .ok_or("curve normals were absent")?;
        assert_eq!(normals, [DVec3::X; 3]);
        Ok(())
    }

    #[test]
    fn empty_curve_has_empty_cycles_path_geometry() -> Result<(), Box<dyn Error>> {
        let curves = HairCurvesData {
            curves: vec![HairCurve {
                points: Vec::new(),
                radius: 0.1,
            }],
            ..HairCurvesData::default()
        };
        let prepared = prepare_cycles_hair_curve_geometry(
            &curves,
            DMat4::IDENTITY,
            HairCurveRenderMode::Thick,
            2,
        )
        .map_err(|error| format!("{error:?}"))?;
        assert_eq!(prepared.len(), 1);
        assert!(matches!(
            &prepared[0].curve.shape,
            HairCurveShape::CatmullRom { keys, cyclic: false } if keys.is_empty()
        ));
        Ok(())
    }

    #[test]
    fn cycles_path_uses_one_global_basis_for_every_authored_type() -> Result<(), Box<dyn Error>> {
        let curve_data = |curve_types: &[i64]| HairCurvesData {
            curves: curve_types
                .iter()
                .enumerate()
                .map(|(index, _)| HairCurve {
                    points: vec![
                        [index as f64, 0.0, 0.0],
                        [index as f64, 0.0, 1.0],
                        [index as f64, 0.0, 2.0],
                    ],
                    radius: 0.1,
                })
                .collect(),
            attributes: BTreeMap::from([(
                "curve_type".to_owned(),
                HairCurveAttribute {
                    domain: "CURVE".to_owned(),
                    data_type: "INT8".to_owned(),
                    values: curve_types.iter().map(|value| json!(value)).collect(),
                },
            )]),
            ..HairCurvesData::default()
        };

        let first_poly = curve_data(&[1, 0]);
        let prepared = prepare_cycles_hair_curve_geometry(
            &first_poly,
            DMat4::IDENTITY,
            HairCurveRenderMode::Ribbon,
            2,
        )
        .map_err(|error| format!("{error:?}"))?;
        assert!(prepared.iter().all(|curve| {
            curve.curve.render_mode == HairCurveRenderMode::ThickLinear
                && matches!(
                    &curve.curve.shape,
                    HairCurveShape::Linear { keys, cyclic: false } if keys.len() == 3
                )
        }));

        let later_poly = curve_data(&[0, 1]);
        let prepared = prepare_cycles_hair_curve_geometry(
            &later_poly,
            DMat4::IDENTITY,
            HairCurveRenderMode::Ribbon,
            2,
        )
        .map_err(|error| format!("{error:?}"))?;
        assert!(prepared.iter().all(|curve| {
            curve.curve.render_mode == HairCurveRenderMode::Ribbon
                && matches!(
                    &curve.curve.shape,
                    HairCurveShape::CatmullRom { keys, cyclic: false } if keys.len() == 3
                )
        }));

        let configured_linear = curve_data(&[0, 3]);
        let prepared = prepare_cycles_hair_curve_geometry(
            &configured_linear,
            DMat4::IDENTITY,
            HairCurveRenderMode::ThickLinear,
            2,
        )
        .map_err(|error| format!("{error:?}"))?;
        assert!(prepared.iter().all(|curve| {
            curve.curve.render_mode == HairCurveRenderMode::ThickLinear
                && matches!(
                    &curve.curve.shape,
                    HairCurveShape::Linear { keys, cyclic: false } if keys.len() == 3
                )
        }));
        Ok(())
    }

    #[test]
    fn point_normal_keys_mix_raw_before_huang_projection() -> Result<(), Box<dyn Error>> {
        // Blender 5.2.2 source point-domain normals for the canonical four-key curve.
        let source_normals = [
            DVec3::new(0.239_537_27, -0.970_887_18, 0.0),
            DVec3::new(0.002_283_36, -0.989_907_44, -0.141_696_97),
            DVec3::new(-0.194_739_65, -0.942_921_94, -0.270_138_38),
            DVec3::new(-0.148_964_48, -0.943_107_96, -0.297_249_14),
        ];
        let attribute = |domain: &str, data_type: &str, values| HairCurveAttribute {
            domain: domain.to_owned(),
            data_type: data_type.to_owned(),
            values,
        };
        let curves = HairCurvesData {
            curves: vec![HairCurve {
                points: vec![
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 2.0],
                    [0.0, 0.0, 3.0],
                ],
                radius: 0.1,
            }],
            attributes: BTreeMap::from([
                (
                    "curve_type".to_owned(),
                    attribute("CURVE", "INT8", vec![json!(0)]),
                ),
                (
                    "normal_mode".to_owned(),
                    attribute("CURVE", "INT8", vec![json!(FREE)]),
                ),
                (
                    "custom_normal".to_owned(),
                    attribute(
                        "POINT",
                        "FLOAT_VECTOR",
                        source_normals
                            .iter()
                            .map(|normal| json!([normal.x, normal.y, normal.z]))
                            .collect(),
                    ),
                ),
            ]),
            ..HairCurvesData::default()
        };
        let prepared = prepare_cycles_hair_curve_geometry(
            &curves,
            DMat4::IDENTITY,
            HairCurveRenderMode::ThickLinear,
            2,
        )
        .map_err(|error| format!("{error:?}"))?
        .into_iter()
        .next()
        .ok_or("curve preparation returned no curves")?;
        assert!(matches!(
            prepared.curve.shape,
            HairCurveShape::Linear { .. }
        ));

        let u = 0.471_619_776;
        let expected = source_normals[1]
            .normalize()
            .lerp(source_normals[2].normalize(), u);
        let point_normal = prepared
            .point_normals
            .at(1, u)
            .ok_or("point normal was absent")?;
        assert!((point_normal - expected).length() < 1.0e-12);
        assert!(point_normal.length() < 0.995);

        // Huang projects the attribute against dPdu; keep a near-parallel case non-degenerate.
        let tangent = (expected.normalize() + DVec3::Z * 1.0e-5).normalize();
        let projection = tangent
            .cross(point_normal.cross(tangent).normalize_or_zero())
            .normalize_or_zero();
        assert!(point_normal.cross(tangent).length() < 1.0e-5);
        assert!(
            (projection - DVec3::new(0.018_997_49, 0.202_840_71, -0.979_027_45)).length() < 1.0e-6
        );
        Ok(())
    }

    #[test]
    fn shadow_transparency_interpolates_f32_keys_and_defaults_missing_values_to_zero()
    -> Result<(), Box<dyn Error>> {
        let values = HairCurveShadowTransparency::new(Some(vec![0.2_f32, 0.8, 0.4]), false);
        let expected = f64::from(0.8_f32 * (1.0 - 0.25_f32) + 0.4 * 0.25_f32);
        assert!(
            (values
                .at(1, 0.25)
                .ok_or("shadow transparency value was absent")?
                - expected)
                .abs()
                < 1.0e-12
        );

        let cyclic = HairCurveShadowTransparency::new(Some(vec![0.2_f32, 0.8, 0.4]), true);
        let expected = f64::from(0.4_f32 * (1.0 - 0.25_f32) + 0.2 * 0.25_f32);
        assert!(
            (cyclic
                .at(2, 0.25)
                .ok_or("cyclic shadow transparency value was absent")?
                - expected)
                .abs()
                < 1.0e-12
        );

        let missing = HairCurveShadowTransparency::new(None, false);
        assert_eq!(missing.at(0, 0.5), Some(0.0));
        Ok(())
    }

    #[test]
    fn cycles_path_uses_raw_bezier_and_nurbs_control_keys() -> Result<(), Box<dyn Error>> {
        let attribute = |domain: &str, data_type: &str, values| HairCurveAttribute {
            domain: domain.to_owned(),
            data_type: data_type.to_owned(),
            values,
        };
        let source_points = [
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 3.0],
            ],
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 2.0],
                [1.0, 0.0, 3.0],
            ],
        ];
        let source_radii = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let curves = HairCurvesData {
            curves: source_points
                .iter()
                .map(|points| HairCurve {
                    points: points.to_vec(),
                    radius: 0.1,
                })
                .collect(),
            attributes: BTreeMap::from([
                (
                    "curve_type".to_owned(),
                    attribute("CURVE", "INT8", vec![json!(BEZIER), json!(NURBS)]),
                ),
                (
                    "radius".to_owned(),
                    attribute(
                        "POINT",
                        "FLOAT",
                        source_radii.iter().map(|radius| json!(radius)).collect(),
                    ),
                ),
            ]),
            ..HairCurvesData::default()
        };
        let prepared = prepare_cycles_hair_curve_geometry(
            &curves,
            DMat4::IDENTITY,
            HairCurveRenderMode::Thick,
            2,
        )
        .map_err(|error| format!("{error:?}"))?;
        for (curve_index, (prepared, source_points)) in
            prepared.iter().zip(source_points.iter()).enumerate()
        {
            assert_eq!(prepared.curve.render_mode, HairCurveRenderMode::Thick);
            let HairCurveShape::CatmullRom {
                keys,
                cyclic: false,
            } = &prepared.curve.shape
            else {
                panic!("Cycles path must use open Catmull-Rom raw keys in thick mode");
            };
            assert_eq!(
                keys.iter()
                    .map(|key| key.position.to_array())
                    .collect::<Vec<_>>(),
                source_points.to_vec()
            );
            assert_eq!(
                keys.iter().map(|key| key.radius).collect::<Vec<_>>(),
                source_radii[curve_index * 4..curve_index * 4 + 4].to_vec()
            );
        }
        Ok(())
    }

    #[test]
    fn cyclic_source_keeps_normals_cyclic_but_path_endpoints_open() -> Result<(), Box<dyn Error>> {
        let attribute = |domain: &str, data_type: &str, values| HairCurveAttribute {
            domain: domain.to_owned(),
            data_type: data_type.to_owned(),
            values,
        };
        let curves = HairCurvesData {
            curves: vec![HairCurve {
                points: vec![[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
                radius: 0.1,
            }],
            attributes: BTreeMap::from([
                (
                    "cyclic".to_owned(),
                    attribute("CURVE", "BOOLEAN", vec![json!(true)]),
                ),
                (
                    "normal_mode".to_owned(),
                    attribute("CURVE", "INT8", vec![json!(FREE)]),
                ),
                (
                    "custom_normal".to_owned(),
                    attribute(
                        "POINT",
                        "FLOAT_VECTOR",
                        vec![
                            json!([1.0, 0.0, 0.0]),
                            json!([0.0, 1.0, 0.0]),
                            json!([0.0, 0.0, 1.0]),
                        ],
                    ),
                ),
            ]),
            ..HairCurvesData::default()
        };
        let prepared = prepare_cycles_hair_curve_geometry(
            &curves,
            DMat4::IDENTITY,
            HairCurveRenderMode::Thick,
            2,
        )
        .map_err(|error| format!("{error:?}"))?
        .into_iter()
        .next()
        .ok_or("curve preparation returned no curves")?;
        let HairCurveShape::CatmullRom {
            keys,
            cyclic: false,
        } = &prepared.curve.shape
        else {
            panic!("Cycles path uses the raw open key sequence");
        };
        assert_eq!(keys.len(), 3);
        assert_eq!(
            prepared
                .curve
                .sample_local(1, 1.0)
                .ok_or("path endpoint was absent")?
                .position,
            DVec3::new(0.0, 0.0, 2.0)
        );
        assert!(prepared.curve.sample_local(2, 0.0).is_none());
        assert_eq!(
            prepared.point_normals.at(2, 0.5),
            Some(DVec3::new(0.5, 0.0, 0.5))
        );
        Ok(())
    }
    #[test]
    fn bezier_free_handle_tangents_define_source_point_normals() -> Result<(), Box<dyn Error>> {
        let attribute = |domain: &str, data_type: &str, values| HairCurveAttribute {
            domain: domain.to_owned(),
            data_type: data_type.to_owned(),
            values,
        };
        let curves = HairCurvesData {
            curves: vec![HairCurve {
                points: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                radius: 0.1,
            }],
            attributes: BTreeMap::from([
                (
                    "curve_type".to_owned(),
                    attribute("CURVE", "INT8", vec![json!(BEZIER)]),
                ),
                (
                    "normal_mode".to_owned(),
                    attribute("CURVE", "INT8", vec![json!(Z_UP)]),
                ),
                (
                    "handle_left".to_owned(),
                    attribute(
                        "POINT",
                        "FLOAT_VECTOR",
                        vec![json!([0.0, 0.0, 0.0]), json!([0.0, 1.0, 0.0])],
                    ),
                ),
                (
                    "handle_right".to_owned(),
                    attribute(
                        "POINT",
                        "FLOAT_VECTOR",
                        vec![json!([1.0, 1.0, 0.0]), json!([1.0, 0.0, 0.0])],
                    ),
                ),
                (
                    "handle_type_left".to_owned(),
                    attribute("POINT", "INT8", vec![json!(0), json!(0)]),
                ),
                (
                    "handle_type_right".to_owned(),
                    attribute("POINT", "INT8", vec![json!(0), json!(0)]),
                ),
            ]),
            ..HairCurvesData::default()
        };
        let prepared = prepare_cycles_hair_curve_geometry(
            &curves,
            DMat4::IDENTITY,
            HairCurveRenderMode::ThickLinear,
            2,
        )
        .map_err(|error| format!("{error:?}"))?
        .into_iter()
        .next()
        .ok_or("curve preparation returned no curves")?;
        let inverse_sqrt_two = std::f64::consts::FRAC_1_SQRT_2;
        let expected_first = DVec3::new(inverse_sqrt_two, -inverse_sqrt_two, 0.0);
        let expected_last = DVec3::new(-inverse_sqrt_two, -inverse_sqrt_two, 0.0);
        assert!(
            (prepared
                .point_normals
                .at(0, 0.0)
                .ok_or("first point normal was absent")?
                - expected_first)
                .length()
                < 1.0e-6
        );
        assert!(
            (prepared
                .point_normals
                .at(0, 1.0)
                .ok_or("last point normal was absent")?
                - expected_last)
                .length()
                < 1.0e-6
        );
        assert_eq!(
            prepared
                .curve
                .sample_local(0, 0.0)
                .ok_or("curve start sample was absent")?
                .tangent,
            DVec3::X,
            "render geometry remains on its raw linear path basis"
        );
        Ok(())
    }

    #[test]
    fn bezier_vector_segments_use_source_variable_offsets_for_point_normals()
    -> Result<(), Box<dyn Error>> {
        let attribute = |domain: &str, data_type: &str, values| HairCurveAttribute {
            domain: domain.to_owned(),
            data_type: data_type.to_owned(),
            values,
        };
        let points = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ];
        let curves = HairCurvesData {
            curves: vec![HairCurve {
                points: points.to_vec(),
                radius: 0.1,
            }],
            attributes: BTreeMap::from([
                (
                    "curve_type".to_owned(),
                    attribute("CURVE", "INT8", vec![json!(BEZIER)]),
                ),
                (
                    "normal_mode".to_owned(),
                    attribute("CURVE", "INT8", vec![json!(MINIMUM_TWIST)]),
                ),
                (
                    "resolution".to_owned(),
                    attribute("CURVE", "INT", vec![json!(2)]),
                ),
                (
                    "handle_left".to_owned(),
                    attribute(
                        "POINT",
                        "FLOAT_VECTOR",
                        points.iter().map(|point| json!(point)).collect(),
                    ),
                ),
                (
                    "handle_right".to_owned(),
                    attribute(
                        "POINT",
                        "FLOAT_VECTOR",
                        points.iter().map(|point| json!(point)).collect(),
                    ),
                ),
                (
                    "handle_type_left".to_owned(),
                    attribute("POINT", "INT8", vec![json!(2); points.len()]),
                ),
                (
                    "handle_type_right".to_owned(),
                    attribute("POINT", "INT8", vec![json!(2); points.len()]),
                ),
            ]),
            ..HairCurvesData::default()
        };
        let prepared = prepare_cycles_hair_curve_geometry(
            &curves,
            DMat4::IDENTITY,
            HairCurveRenderMode::ThickLinear,
            2,
        )
        .map_err(|error| format!("{error:?}"))?
        .into_iter()
        .next()
        .ok_or("curve preparation returned no curves")?;
        let expected = DVec3::new(
            2.0 * std::f64::consts::SQRT_2 / 3.0,
            -std::f64::consts::SQRT_2 / 6.0,
            std::f64::consts::SQRT_2 / 6.0,
        );
        assert!(
            (prepared
                .point_normals
                .at(2, 0.0)
                .ok_or("Bezier point normal was absent")?
                - expected)
                .length()
                < 1.0e-4,
            "point normals must use the variable-offset Bezier evaluation, not uniform render subdivisions"
        );
        assert_eq!(
            prepared
                .curve
                .sample_local(1, 0.0)
                .ok_or("Bezier segment sample was absent")?
                .tangent,
            DVec3::Y,
            "normal evaluation must not change the linear render basis"
        );
        Ok(())
    }
}
