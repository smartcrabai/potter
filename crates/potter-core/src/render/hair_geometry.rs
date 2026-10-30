use glam::{DMat4, DVec3};
use serde_json::Value;

use crate::model::HairCurvesData;

use super::hair_intersect::{
    CurveKey, HairCurveGeometry, HairCurveShape, MAX_NURBS_ORDER, NurbsCurve, curve_normal_to_world,
};

const CATMULL_ROM: u8 = 0;
const POLY: u8 = 1;
const BEZIER: u8 = 2;
const NURBS: u8 = 3;
const MINIMUM_TWIST: u8 = 0;
const Z_UP: u8 = 1;
const FREE: u8 = 2;

#[derive(Debug)]
pub(crate) enum HairCurveGeometryError {
    InvalidAttribute(&'static str),
    InvalidCurve(usize),
    MissingCustomKnots { curve_index: usize },
    InvalidTransform,
}

/// Prepares the native analytic spline controls from serialized Hair Curves attributes.
///
/// The transform is retained as an affine pair so intersection can transform rays into object
/// space; this preserves circular local radii under non-uniform scale and shear.
#[expect(
    clippy::float_cmp,
    reason = "Blender's affine matrices require exact homogeneous-row values"
)]
pub(crate) fn prepare_hair_curve_geometry(
    data: &HairCurvesData,
    object_to_world: DMat4,
) -> Result<Vec<HairCurveGeometry>, HairCurveGeometryError> {
    let determinant = object_to_world.determinant();
    if !object_to_world
        .to_cols_array()
        .iter()
        .all(|value| value.is_finite())
        || !determinant.is_finite()
        || determinant == 0.0
        || object_to_world.x_axis.w != 0.0
        || object_to_world.y_axis.w != 0.0
        || object_to_world.z_axis.w != 0.0
        || object_to_world.w_axis.w != 1.0
    {
        return Err(HairCurveGeometryError::InvalidTransform);
    }
    let world_to_object = object_to_world.inverse();
    if !world_to_object
        .to_cols_array()
        .iter()
        .all(|value| value.is_finite())
    {
        return Err(HairCurveGeometryError::InvalidTransform);
    }

    let point_count = data
        .curves
        .iter()
        .try_fold(0usize, |count, curve| count.checked_add(curve.points.len()))
        .ok_or(HairCurveGeometryError::InvalidAttribute("point count"))?;
    let curve_count = data.curves.len();
    let curve_type = curve_attr(data, "curve_type", "CURVE", "INT8", curve_count)?;
    let cyclic = curve_attr(data, "cyclic", "CURVE", "BOOLEAN", curve_count)?;
    let resolution = curve_attr(data, "resolution", "CURVE", "INT", curve_count)?;
    let normal_mode = curve_attr(data, "normal_mode", "CURVE", "INT8", curve_count)?;
    let nurbs_order = curve_attr(data, "nurbs_order", "CURVE", "INT8", curve_count)?;
    let knots_mode = curve_attr(data, "knots_mode", "CURVE", "INT8", curve_count)?;
    let radius = curve_attr(data, "radius", "POINT", "FLOAT", point_count)?;
    let tilt = curve_attr(data, "tilt", "POINT", "FLOAT", point_count)?;
    let left = curve_attr(data, "handle_left", "POINT", "FLOAT_VECTOR", point_count)?;
    let right = curve_attr(data, "handle_right", "POINT", "FLOAT_VECTOR", point_count)?;
    let left_type = curve_attr(data, "handle_type_left", "POINT", "INT8", point_count)?;
    let right_type = curve_attr(data, "handle_type_right", "POINT", "INT8", point_count)?;
    let nurbs_weight = curve_attr(data, "nurbs_weight", "POINT", "FLOAT", point_count)?;
    let custom_normals = curve_vec3_attr(data, "custom_normal", point_count)?;

    let mut prepared = Vec::new();
    prepared
        .try_reserve_exact(curve_count)
        .map_err(|_| HairCurveGeometryError::InvalidAttribute("curve allocation"))?;
    let mut point_offset = 0usize;
    for (curve_index, curve) in data.curves.iter().enumerate() {
        let count = curve.points.len();
        if !curve.radius.is_finite() || curve.radius < 0.0 {
            return Err(HairCurveGeometryError::InvalidCurve(curve_index));
        }
        if count == 0 {
            prepared.push(HairCurveGeometry {
                resolution: 12,
                normal_mode: MINIMUM_TWIST,
                tilts: Vec::new(),
                custom_normals: None,
                shape: HairCurveShape::Linear {
                    keys: Vec::new(),
                    cyclic: false,
                },
                object_to_world,
                world_to_object,
            });
            continue;
        }
        let mut keys = Vec::new();
        keys.try_reserve_exact(count)
            .map_err(|_| HairCurveGeometryError::InvalidAttribute("point allocation"))?;
        for (point_index, point) in curve.points.iter().enumerate() {
            let position = DVec3::from_array(*point);
            let point_radius = match radius {
                Some(values) => values[point_offset + point_index]
                    .as_f64()
                    .ok_or(HairCurveGeometryError::InvalidCurve(curve_index))?,
                None => curve.radius,
            };
            if !position.is_finite() || !point_radius.is_finite() || point_radius < 0.0 {
                return Err(HairCurveGeometryError::InvalidCurve(curve_index));
            }
            keys.push(CurveKey {
                position,
                radius: point_radius,
            });
        }

        let spline_type = match curve_type {
            Some(values) => values[curve_index]
                .as_i64()
                .ok_or(HairCurveGeometryError::InvalidCurve(curve_index))?,
            None => i64::from(CATMULL_ROM),
        };
        let cyclic = match cyclic {
            Some(values) => values[curve_index]
                .as_bool()
                .ok_or(HairCurveGeometryError::InvalidCurve(curve_index))?,
            None => false,
        };
        let resolution = match resolution {
            Some(values) => values[curve_index]
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .filter(|value| *value > 0)
                .ok_or(HairCurveGeometryError::InvalidCurve(curve_index))?,
            None => 12,
        };
        let normal_mode = match normal_mode {
            Some(values) => values[curve_index]
                .as_i64()
                .and_then(|value| u8::try_from(value).ok())
                .ok_or(HairCurveGeometryError::InvalidCurve(curve_index))?,
            None => MINIMUM_TWIST,
        };
        if !matches!(normal_mode, MINIMUM_TWIST | Z_UP | FREE) {
            return Err(HairCurveGeometryError::InvalidCurve(curve_index));
        }
        let tilts = point_values_f64(tilt, point_offset, count, 0.0, curve_index)?;
        let curve_custom_normals = custom_normals
            .as_ref()
            .map(|values| values[point_offset..point_offset + count].to_vec());
        let shape = match spline_type {
            value if value == i64::from(CATMULL_ROM) => HairCurveShape::CatmullRom { keys, cyclic },
            value if value == i64::from(POLY) => HairCurveShape::Linear { keys, cyclic },
            value if value == i64::from(BEZIER) => {
                let handle_left = point_vec3_values(left, point_offset, &keys, curve_index)?;
                let handle_right = point_vec3_values(right, point_offset, &keys, curve_index)?;
                let left_types = point_values_i64(left_type, point_offset, count, 1, curve_index)?;
                let right_types =
                    point_values_i64(right_type, point_offset, count, 1, curve_index)?;
                let (handle_left, handle_right) = calculate_bezier_handles(
                    &keys,
                    &handle_left,
                    &handle_right,
                    &left_types,
                    &right_types,
                    cyclic,
                    curve_index,
                )?;
                let segment_count = if cyclic {
                    count
                } else {
                    count.saturating_sub(1)
                };
                let mut spans = Vec::new();
                spans
                    .try_reserve_exact(segment_count)
                    .map_err(|_| HairCurveGeometryError::InvalidAttribute("Bezier allocation"))?;
                for segment in 0..segment_count {
                    let next = (segment + 1) % count;
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
                HairCurveShape::Bezier { spans, cyclic }
            }
            value if value == i64::from(NURBS) => {
                let mode = match knots_mode {
                    Some(values) => values[curve_index]
                        .as_i64()
                        .ok_or(HairCurveGeometryError::InvalidCurve(curve_index))?,
                    None => 0,
                };
                if !(0..=4).contains(&mode) {
                    return Err(HairCurveGeometryError::InvalidCurve(curve_index));
                }
                let order = match nurbs_order {
                    Some(values) => values[curve_index]
                        .as_i64()
                        .and_then(|value| usize::try_from(value).ok())
                        .ok_or(HairCurveGeometryError::InvalidCurve(curve_index))?,
                    None => 4,
                };
                let invalid_nurbs = order < 2
                    || order > count
                    || (mode == 2 && count <= order)
                    || ((mode == 2 || mode == 3) && cyclic && count % (order - 1) != 0);
                if invalid_nurbs {
                    HairCurveShape::Linear { keys, cyclic }
                } else if mode == 4 {
                    return Err(HairCurveGeometryError::MissingCustomKnots { curve_index });
                } else if order > MAX_NURBS_ORDER {
                    return Err(HairCurveGeometryError::InvalidCurve(curve_index));
                } else {
                    let weights =
                        point_values_f64(nurbs_weight, point_offset, count, 1.0, curve_index)?;
                    if weights.iter().any(|weight| *weight <= 0.0) {
                        return Err(HairCurveGeometryError::InvalidCurve(curve_index));
                    }
                    let nurbs = make_nurbs_curve(
                        keys,
                        weights,
                        order,
                        mode & 1 != 0,
                        mode & 2 != 0,
                        cyclic,
                    )
                    .ok_or(HairCurveGeometryError::InvalidCurve(curve_index))?;
                    HairCurveShape::Nurbs(NurbsCurve {
                        keys: nurbs.controls,
                        weights: nurbs.weights,
                        knots: nurbs.knots,
                        order,
                        spans: nurbs.spans,
                        cyclic,
                    })
                }
            }
            _ => return Err(HairCurveGeometryError::InvalidCurve(curve_index)),
        };
        prepared.push(HairCurveGeometry {
            resolution,
            normal_mode,
            tilts,
            custom_normals: curve_custom_normals,
            shape,
            object_to_world,
            world_to_object,
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
        return Err(HairCurveGeometryError::InvalidAttribute(name));
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
                .ok_or(HairCurveGeometryError::InvalidAttribute(name))?;
            let coordinates = [
                array[0]
                    .as_f64()
                    .ok_or(HairCurveGeometryError::InvalidAttribute(name))?,
                array[1]
                    .as_f64()
                    .ok_or(HairCurveGeometryError::InvalidAttribute(name))?,
                array[2]
                    .as_f64()
                    .ok_or(HairCurveGeometryError::InvalidAttribute(name))?,
            ];
            let vector = DVec3::from_array(coordinates);
            vector
                .is_finite()
                .then_some(vector)
                .ok_or(HairCurveGeometryError::InvalidAttribute(name))
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
                .ok_or(HairCurveGeometryError::InvalidCurve(curve_index))?;
            let vector = DVec3::from_array(coordinates);
            vector
                .is_finite()
                .then_some(vector)
                .ok_or(HairCurveGeometryError::InvalidCurve(curve_index))
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
        .map_err(|_| HairCurveGeometryError::InvalidAttribute("point allocation"))?;
    for index in 0..count {
        let value = match values {
            Some(values) => values[offset + index]
                .as_f64()
                .ok_or(HairCurveGeometryError::InvalidCurve(curve_index))?,
            None => default,
        };
        if !value.is_finite() {
            return Err(HairCurveGeometryError::InvalidCurve(curve_index));
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
        .map_err(|_| HairCurveGeometryError::InvalidAttribute("point allocation"))?;
    for index in 0..count {
        let value = match values {
            Some(values) => values[offset + index]
                .as_i64()
                .ok_or(HairCurveGeometryError::InvalidCurve(curve_index))?,
            None => default,
        };
        if !(0..=3).contains(&value) {
            return Err(HairCurveGeometryError::InvalidCurve(curve_index));
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
        return Err(HairCurveGeometryError::InvalidAttribute("Bezier handles"));
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
            return Err(HairCurveGeometryError::InvalidCurve(curve_index));
        }
    }
    Ok((left, right))
}

struct NurbsCurveData {
    controls: Vec<CurveKey>,
    weights: Vec<f64>,
    knots: Vec<f64>,
    spans: Vec<usize>,
}

fn make_nurbs_curve(
    mut keys: Vec<CurveKey>,
    mut weights: Vec<f64>,
    order: usize,
    use_endpoint: bool,
    use_bezier: bool,
    cyclic: bool,
) -> Option<NurbsCurveData> {
    if cyclic {
        let repeated = order.checked_sub(1)?;
        let extra_keys = keys.iter().take(repeated).copied().collect::<Vec<_>>();
        let extra_weights = weights.iter().take(repeated).copied().collect::<Vec<_>>();
        keys.try_reserve_exact(repeated).ok()?;
        weights.try_reserve_exact(repeated).ok()?;
        keys.extend(extra_keys);
        weights.extend(extra_weights);
    }
    let control_count = keys.len();
    let knot_count = control_count.checked_add(order)?;
    let mut knots = Vec::new();
    knots.try_reserve_exact(knot_count).ok()?;
    knots.resize(knot_count, 0.0);
    let repeat_inner = if use_bezier { order - 1 } else { 1 };
    let head = if use_endpoint && cyclic {
        order - 1
    } else if use_endpoint {
        order
    } else if use_bezier {
        2.min(repeat_inner)
    } else {
        1
    };
    let tail = if cyclic {
        order.checked_mul(2)?.checked_sub(1)?
    } else if use_endpoint {
        order
    } else {
        0
    };
    let offset = usize::from(use_endpoint && cyclic);
    if head == 0 || tail > knot_count || offset > knot_count - tail {
        return None;
    }
    let mut current = 0.0;
    if offset != 0 {
        knots[0] = current;
        current += 1.0;
    }
    let mut repeat = head;
    for knot in knots.iter_mut().take(knot_count - tail).skip(offset) {
        *knot = current;
        repeat -= 1;
        if repeat == 0 {
            current += 1.0;
            repeat = repeat_inner;
        }
    }
    let tail_start = knot_count - tail;
    for index in 0..tail {
        knots[tail_start + index] = current + (knots[index] - knots[0]);
    }
    let degree = order.checked_sub(1)?;
    let spans = (degree..control_count)
        .filter(|span| knots[*span + 1] > knots[*span])
        .collect::<Vec<_>>();
    (!spans.is_empty()).then_some(NurbsCurveData {
        controls: keys,
        weights,
        knots,
        spans,
    })
}

#[derive(Clone, Debug)]
pub(crate) struct HairCurveNormalSamples {
    normals: Vec<DVec3>,
    samples_per_span: usize,
    span_count: usize,
    cyclic: bool,
}

impl HairCurveNormalSamples {
    pub(crate) fn at(&self, span: usize, u: f64) -> Option<DVec3> {
        if span >= self.span_count || !u.is_finite() || self.normals.is_empty() {
            return None;
        }
        let coordinate = u.clamp(0.0, 1.0) * self.samples_per_span as f64;
        let lower = coordinate.floor() as usize;
        let first = span
            .checked_mul(self.samples_per_span)?
            .checked_add(lower)?
            .min(self.normals.len() - 1);
        let next = if lower >= self.samples_per_span {
            if self.cyclic {
                0
            } else {
                self.normals.len() - 1
            }
        } else {
            let candidate = first.saturating_add(1);
            if candidate == self.normals.len() {
                if self.cyclic {
                    0
                } else {
                    self.normals.len() - 1
                }
            } else {
                candidate
            }
        };
        let normal = self.normals[first].lerp(self.normals[next], coordinate - lower as f64);
        let length = normal.length();
        (length.is_finite() && length > 0.0).then_some(normal / length)
    }
}

pub(crate) fn curve_normal_world(
    curve: &HairCurveGeometry,
    samples: &HairCurveNormalSamples,
    span: usize,
    u: f64,
) -> Option<DVec3> {
    let normal = curve_normal_to_world(curve.world_to_object, samples.at(span, u)?)?;
    let length = normal.length();
    (length.is_finite() && length > 0.0).then_some(normal / length)
}

pub(crate) fn curve_normals_for_geometry(
    curve: &HairCurveGeometry,
) -> Option<HairCurveNormalSamples> {
    let (span_count, samples_per_span, cyclic) = match &curve.shape {
        HairCurveShape::CatmullRom { keys, cyclic } => (
            if *cyclic {
                keys.len()
            } else {
                keys.len().saturating_sub(1)
            },
            curve.resolution.max(1),
            *cyclic,
        ),
        HairCurveShape::Linear { keys, cyclic } => (
            if *cyclic {
                keys.len()
            } else {
                keys.len().saturating_sub(1)
            },
            1,
            *cyclic,
        ),
        HairCurveShape::Bezier { spans, cyclic } => (spans.len(), curve.resolution.max(1), *cyclic),
        HairCurveShape::Nurbs(nurbs) => (nurbs.spans.len(), curve.resolution.max(1), nurbs.cyclic),
    };
    if span_count == 0 {
        return None;
    }
    let point_count = span_count
        .checked_mul(samples_per_span)?
        .checked_add(usize::from(!cyclic))?;
    let mut positions = Vec::new();
    let mut tilts = Vec::new();
    let mut custom_normals = curve.custom_normals.as_ref().map(|_| Vec::new());
    positions.try_reserve_exact(point_count).ok()?;
    tilts.try_reserve_exact(point_count).ok()?;
    if let Some(normals) = &mut custom_normals {
        normals.try_reserve_exact(point_count).ok()?;
    }
    match &curve.shape {
        HairCurveShape::CatmullRom { keys, cyclic } => {
            let control_positions: Vec<_> = keys.iter().map(|key| key.position).collect();
            resize_filled(&mut positions, point_count, DVec3::ZERO).then_some(())?;
            evaluate_catmull_rom(
                &control_positions,
                *cyclic,
                samples_per_span,
                &mut positions,
                catmull_vec3,
            );
            resize_filled(&mut tilts, point_count, 0.0).then_some(())?;
            evaluate_catmull_rom(
                &curve.tilts,
                *cyclic,
                samples_per_span,
                &mut tilts,
                catmull_f64,
            );
            if let (Some(source), Some(destination)) =
                (curve.custom_normals.as_deref(), custom_normals.as_mut())
            {
                resize_filled(destination, point_count, DVec3::ZERO).then_some(())?;
                evaluate_catmull_rom(source, *cyclic, samples_per_span, destination, catmull_vec3);
            }
        }
        HairCurveShape::Linear { keys, .. } => {
            positions.extend(keys.iter().map(|key| key.position));
            tilts.extend_from_slice(&curve.tilts);
            if let (Some(source), Some(destination)) =
                (curve.custom_normals.as_deref(), custom_normals.as_mut())
            {
                destination.extend_from_slice(source);
            }
        }
        HairCurveShape::Bezier { spans, cyclic } => {
            for (span_index, span) in spans.iter().enumerate() {
                let next = (span_index + 1) % curve.tilts.len();
                for step in 0..samples_per_span {
                    let u = step as f64 / samples_per_span as f64;
                    positions.push(super::hair_intersect::bezier_position(*span, u)?);
                    tilts.push(bezier_scalar(curve.tilts[span_index], curve.tilts[next], u));
                    if let (Some(source), Some(destination)) =
                        (curve.custom_normals.as_deref(), custom_normals.as_mut())
                    {
                        destination.push(normalize(bezier_vector(
                            source[span_index],
                            source[next],
                            u,
                        )));
                    }
                }
            }
            if !cyclic {
                positions.push((spans.last()?)[3].position);
                tilts.push(*curve.tilts.last()?);
                if let (Some(source), Some(destination)) =
                    (curve.custom_normals.as_deref(), custom_normals.as_mut())
                {
                    destination.push(*source.last()?);
                }
            }
        }
        HairCurveShape::Nurbs(nurbs) => {
            for (span_index, span) in nurbs.spans.iter().copied().enumerate() {
                let (start, end) = super::hair_intersect::nurbs_span_range(nurbs, span)?;
                for step in 0..samples_per_span {
                    let parameter = start + (end - start) * step as f64 / samples_per_span as f64;
                    let (position, tilt, normal) = super::hair_intersect::nurbs_curve_frame_sample(
                        nurbs,
                        span,
                        parameter,
                        &curve.tilts,
                        curve.custom_normals.as_deref(),
                    )?;
                    positions.push(position);
                    tilts.push(tilt);
                    if let (Some(destination), Some(normal)) = (custom_normals.as_mut(), normal) {
                        destination.push(normal);
                    }
                }
                if span_index + 1 == nurbs.spans.len() && !cyclic {
                    let (position, tilt, normal) = super::hair_intersect::nurbs_curve_frame_sample(
                        nurbs,
                        span,
                        end,
                        &curve.tilts,
                        curve.custom_normals.as_deref(),
                    )?;
                    positions.push(position);
                    tilts.push(tilt);
                    if let (Some(destination), Some(normal)) = (custom_normals.as_mut(), normal) {
                        destination.push(normal);
                    }
                }
            }
        }
    }
    let normals = curve_normals(
        &positions,
        POLY,
        cyclic,
        1,
        curve.normal_mode,
        &tilts,
        custom_normals.as_deref(),
    )?;
    Some(HairCurveNormalSamples {
        normals,
        samples_per_span,
        span_count,
        cyclic,
    })
}

fn bezier_scalar(start: f64, end: f64, u: f64) -> f64 {
    start * (1.0 - u) + end * u
}

fn bezier_vector(start: DVec3, end: DVec3, u: f64) -> DVec3 {
    start.lerp(end, u)
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
    scratch: &mut CurveNormalsScratch,
    output: &mut [DVec3],
) -> bool {
    if positions.is_empty()
        || output.len() != positions.len()
        || tilt.len() != positions.len()
        || custom_normals.is_some_and(|normals| normals.len() != positions.len())
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
#[expect(
    clippy::unwrap_used,
    reason = "valid fixtures are required for the normal and curve preparation tests"
)]
mod tests {
    use std::collections::BTreeMap;

    use glam::{DMat4, DVec3};
    use serde_json::json;

    use crate::model::{HairCurve, HairCurveAttribute, HairCurvesData};

    use super::super::hair_intersect::{CurveKey, HairCurveGeometry};
    use super::{
        FREE, HairCurveGeometryError, HairCurveShape, Z_UP, curve_normals,
        curve_normals_for_geometry, prepare_hair_curve_geometry,
    };

    #[test]
    fn straight_z_up_poly_strand_uses_x_normals() {
        let positions = [DVec3::ZERO, DVec3::Z, DVec3::Z * 2.0];
        let normals = curve_normals(&positions, 1, false, 1, 1, &[0.0; 3], None).unwrap();
        assert_eq!(normals, [DVec3::X; 3]);
    }

    #[test]
    fn custom_nurbs_knots_fail_with_the_missing_serialized_payload() {
        let attribute = |domain: &str, data_type: &str, value| HairCurveAttribute {
            domain: domain.to_owned(),
            data_type: data_type.to_owned(),
            values: vec![value],
        };
        let curves = HairCurvesData {
            curves: vec![HairCurve {
                points: vec![[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                radius: 0.1,
            }],
            attributes: BTreeMap::from([
                (
                    "curve_type".to_owned(),
                    attribute("CURVE", "INT8", json!(3)),
                ),
                (
                    "nurbs_order".to_owned(),
                    attribute("CURVE", "INT8", json!(2)),
                ),
                (
                    "knots_mode".to_owned(),
                    attribute("CURVE", "INT8", json!(4)),
                ),
            ]),
            ..HairCurvesData::default()
        };
        let error = prepare_hair_curve_geometry(&curves, DMat4::IDENTITY).unwrap_err();
        assert!(matches!(
            error,
            HairCurveGeometryError::MissingCustomKnots { curve_index: 0 }
        ));
    }
    #[test]
    fn empty_custom_nurbs_curve_needs_no_missing_knot_payload() {
        let attribute = |domain: &str, data_type: &str, value| HairCurveAttribute {
            domain: domain.to_owned(),
            data_type: data_type.to_owned(),
            values: vec![value],
        };
        let curves = HairCurvesData {
            curves: vec![HairCurve {
                points: Vec::new(),
                radius: 0.1,
            }],
            attributes: BTreeMap::from([
                (
                    "curve_type".to_owned(),
                    attribute("CURVE", "INT8", json!(3)),
                ),
                (
                    "knots_mode".to_owned(),
                    attribute("CURVE", "INT8", json!(4)),
                ),
            ]),
            ..HairCurvesData::default()
        };
        assert_eq!(
            prepare_hair_curve_geometry(&curves, DMat4::IDENTITY)
                .unwrap()
                .len(),
            1
        );
    }

    fn straight_bezier_curve(
        normal_mode: u8,
        tilt_end: f64,
        custom_normals: Option<Vec<DVec3>>,
    ) -> HairCurveGeometry {
        let keys = [
            CurveKey {
                position: DVec3::ZERO,
                radius: 0.1,
            },
            CurveKey {
                position: DVec3::Z / 3.0,
                radius: 0.1,
            },
            CurveKey {
                position: DVec3::Z * (2.0 / 3.0),
                radius: 0.2,
            },
            CurveKey {
                position: DVec3::Z,
                radius: 0.2,
            },
        ];
        HairCurveGeometry {
            resolution: 4,
            normal_mode,
            tilts: vec![0.0, tilt_end],
            custom_normals,
            shape: HairCurveShape::Bezier {
                spans: vec![keys],
                cyclic: false,
            },
            object_to_world: DMat4::IDENTITY,
            world_to_object: DMat4::IDENTITY,
        }
    }

    #[test]
    fn bezier_radius_tilt_and_custom_normals_use_linear_attribute_interpolation() {
        // Blender 5.2.2 `curve_bezier.cc::linear_interpolation` evaluates generic Bezier attributes
        // linearly; the curve position remains cubic.
        let curve = straight_bezier_curve(Z_UP, std::f64::consts::FRAC_PI_2, None);
        let sample = curve.sample_local(0, 0.25).unwrap();
        assert!((sample.radius - 0.125).abs() < 1.0e-12);
        let normals = curve_normals_for_geometry(&curve).unwrap();
        let expected_tilt = DVec3::new(
            (std::f64::consts::FRAC_PI_8).cos(),
            (std::f64::consts::FRAC_PI_8).sin(),
            0.0,
        );
        assert!((normals.at(0, 0.25).unwrap() - expected_tilt).length() < 1.0e-12);

        let curve = straight_bezier_curve(FREE, 0.0, Some(vec![DVec3::X, DVec3::Y]));
        let normals = curve_normals_for_geometry(&curve).unwrap();
        let expected = (DVec3::X * 0.75 + DVec3::Y * 0.25).normalize();
        assert!((normals.at(0, 0.25).unwrap() - expected).length() < 1.0e-12);
    }

    #[test]
    fn free_normal_without_custom_attribute_uses_blender_z_up_fallback() {
        // Blender 5.2.2 `curves_geometry.cc::evaluated_normals` uses Z-up when FREE has no
        // `custom_normal` point attribute.
        let curve = straight_bezier_curve(FREE, 0.0, None);
        let normals = curve_normals_for_geometry(&curve).unwrap();
        assert!((normals.at(0, 0.25).unwrap() - DVec3::X).length() < 1.0e-12);
    }

    #[test]
    fn underspecified_nurbs_falls_back_to_control_point_poly() {
        // Blender 5.2.2 `curves_nurbs.cc::check_valid_eval_params` rejects order > point count;
        // `calculate_evaluated_num` then keeps the input control-point count.
        let attribute = |domain: &str, data_type: &str, value| HairCurveAttribute {
            domain: domain.to_owned(),
            data_type: data_type.to_owned(),
            values: vec![value],
        };
        let curves = HairCurvesData {
            curves: vec![HairCurve {
                points: vec![[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                radius: 0.1,
            }],
            attributes: BTreeMap::from([
                (
                    "curve_type".to_owned(),
                    attribute("CURVE", "INT8", json!(3)),
                ),
                (
                    "nurbs_order".to_owned(),
                    attribute("CURVE", "INT8", json!(4)),
                ),
                (
                    "knots_mode".to_owned(),
                    attribute("CURVE", "INT8", json!(0)),
                ),
            ]),
            ..HairCurvesData::default()
        };
        let prepared = prepare_hair_curve_geometry(&curves, DMat4::IDENTITY).unwrap();
        assert!(matches!(
            &prepared[0].shape,
            HairCurveShape::Linear { keys, cyclic: false } if keys.len() == 2
        ));
    }
}
