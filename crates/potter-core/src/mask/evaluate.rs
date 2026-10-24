//! Blender 5.2.2 mask spline evaluation in normalized mask space.
//!
//! Points are in Blender's mask space: `x / width` and `1 - y / height`, so the image spans the unit
//! square with its origin at the bottom-left. The routines follow `mask.cc` (`BKE_nurb_handle_calc`
//! for automatic and vector handles, `BKE_mask_point_normal`, `BKE_mask_point_weight`),
//! `mask_evaluate.cc` (`BKE_mask_spline_resolution`, `BKE_mask_spline_feather_resolution`,
//! `BKE_mask_spline_differentiate_with_resolution`, `mask_spline_feather_differentiated_points_with_resolution__even`),
//! and `mask_rasterize.cc` (`maskrasterize_layer_init_cdt`, `maskrasterize_spline_differentiate_point_outset`).
//! Bezier segments are evaluated directly rather than by forward differencing; the two differ only by
//! float rounding.

use glam::DVec2;

use super::{FeatherPoint, HandleType, MaskError};

/// `MASK_RESOL_MAX`: the per-spline cap of `BKE_mask_spline_resolution`.
const RESOLUTION_MAX: u32 = 128;
/// Sample-count bounds of `maskrasterize_layer_init_cdt` after dividing by four.
const SAMPLES_MIN: usize = 4;
const SAMPLES_MAX: usize = 512;
/// `BKE_nurb_handle_calc`'s length factor for automatic handles.
const AUTO_HANDLE_LENGTH: f64 = 2.5614;
/// `BKE_mask_point_normal`'s tangent step, in segment parameter units.
const NORMAL_STEP: f64 = 0.05;
/// `BKE_mask_spline_feather_resolution`'s weight-slope step.
const FEATHER_SLOPE_STEP: f64 = 0.005;
/// `FLT_EPSILON`, the smallest parameter step `BKE_mask_spline_feather_resolution` considers.
const FLOAT_EPSILON: f64 = f32::EPSILON as f64;

/// A control point in normalized mask space, with its feather weight resolved to normalized units.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Point {
    pub(super) position: DVec2,
    pub(super) handle_left: DVec2,
    pub(super) handle_right: DVec2,
    pub(super) left_type: HandleType,
    pub(super) right_type: HandleType,
    /// Blender's `bezt.weight`: the point's own weight, or the spline's feather when omitted.
    pub(super) weight: f64,
    /// Blender's `uw` list: multipliers of the interpolated weight along the outgoing segment.
    pub(super) feather_points: Vec<FeatherPoint>,
}

/// A spline's filled body polygon and its feather ring, both in normalized mask space.
///
/// `feather[k]` is the outer edge of the ring at body point `k`. The ring is absent when feathering
/// and anti-aliasing are both off.
#[derive(Debug)]
pub(super) struct Contour {
    pub(super) body: Vec<DVec2>,
    pub(super) feather: Option<Vec<DVec2>>,
}

fn allocate<T>(count: usize) -> Result<Vec<T>, MaskError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| MaskError::AllocationFailed)?;
    Ok(values)
}

/// `usize` as `f64` for counts bounded by the sample caps.
fn count_to_f64(count: usize) -> f64 {
    f64::from(u32::try_from(count).unwrap_or(u32::MAX))
}

/// `floor(value)` clamped to `0..=max`, computed without a float-to-integer cast.
fn floor_capped(value: f64, max: u32) -> usize {
    let capped = if value.is_nan() {
        0.0
    } else {
        value.clamp(0.0, f64::from(max))
    };
    let mut count = 0_u32;
    while count < max && f64::from(count + 1) <= capped {
        count += 1;
    }
    usize::try_from(count).unwrap_or(0)
}

/// `BKE_mask_get_handle_point_adjacent`'s previous neighbour.
fn previous_index(count: usize, index: usize, cyclic: bool) -> Option<usize> {
    if index > 0 {
        Some(index - 1)
    } else if cyclic {
        Some(count - 1)
    } else {
        None
    }
}

/// `BKE_mask_spline_point_next_bezt`'s next neighbour.
fn next_index(count: usize, index: usize, cyclic: bool) -> Option<usize> {
    if index + 1 < count {
        Some(index + 1)
    } else if cyclic {
        Some(0)
    } else {
        None
    }
}

fn segment_count(count: usize, cyclic: bool) -> usize {
    if cyclic { count } else { count - 1 }
}

fn lerp(start: f64, end: f64, amount: f64) -> f64 {
    start * (1.0 - amount) + end * amount
}

fn cubic(first: DVec2, first_handle: DVec2, second_handle: DVec2, second: DVec2, u: f64) -> DVec2 {
    let mut work = [first, first_handle, second_handle, second];
    for level in 1..4 {
        for index in 0..(4 - level) {
            work[index] = work[index].lerp(work[index + 1], u);
        }
    }
    work[0]
}

/// `BKE_mask_point_segment_co`: the curve point at `u` on the segment leaving `index`. The last
/// point of an open spline has no segment, so it is its own position.
fn segment_point(points: &[Point], cyclic: bool, index: usize, u: f64) -> DVec2 {
    match next_index(points.len(), index, cyclic) {
        Some(next) => cubic(
            points[index].position,
            points[index].handle_right,
            points[next].handle_left,
            points[next].position,
            u,
        ),
        None => points[index].position,
    }
}

fn normalized(vector: DVec2) -> DVec2 {
    let length = vector.length();
    if length > 0.0 {
        vector / length
    } else {
        DVec2::ZERO
    }
}

/// `orthogonal_direction_get`: the unit left-hand normal of a direction.
fn orthogonal(direction: DVec2) -> DVec2 {
    normalized(DVec2::new(-direction.y, direction.x))
}

/// `BKE_mask_point_normal` for the segment leaving `index`, at segment parameter `u`.
fn point_normal(points: &[Point], cyclic: bool, index: usize, u: f64) -> DVec2 {
    let count = points.len();
    let previous = previous_index(count, index, cyclic);
    let next = next_index(count, index, cyclic);
    let direction = if u - NORMAL_STEP < 0.0 && previous.is_none() {
        segment_point(points, cyclic, index, u + NORMAL_STEP) - points[index].position
    } else if u + NORMAL_STEP > 1.0 && next.is_none() {
        points[index].position - segment_point(points, cyclic, index, u - NORMAL_STEP)
    } else {
        let before = match (u - NORMAL_STEP < 0.0, previous) {
            (true, Some(previous)) => {
                segment_point(points, cyclic, previous, 1.0 + (u - NORMAL_STEP))
            }
            _ => segment_point(points, cyclic, index, u - NORMAL_STEP),
        };
        let current = segment_point(points, cyclic, index, u);
        let after = match (u + NORMAL_STEP > 1.0, next) {
            (true, Some(next)) => segment_point(points, cyclic, next, u + NORMAL_STEP - 1.0),
            _ => segment_point(points, cyclic, index, u + NORMAL_STEP),
        };
        normalized(current - before) + normalized(after - current)
    };
    orthogonal(direction)
}

/// `BKE_mask_point_weight`: the feather weight at `u` on the segment leaving `index`. The weight
/// eases (smoothstep) between the point weights, scaled by the `uw` multipliers.
fn point_weight(points: &[Point], cyclic: bool, index: usize, u: f64) -> f64 {
    let point = &points[index];
    let Some(next) = next_index(points.len(), index, cyclic) else {
        return point.weight;
    };
    let next_point = &points[next];
    if u <= 0.0 {
        return point.weight;
    }
    if u >= 1.0 {
        return next_point.weight;
    }
    let feather = &point.feather_points;
    let count = feather.len();
    let lower = |step: usize| {
        if step == 0 {
            (0.0, 1.0)
        } else {
            (feather[step - 1].u, feather[step - 1].w)
        }
    };
    let upper = |step: usize| {
        if step == count {
            (1.0, 1.0)
        } else {
            (feather[step].u, feather[step].w)
        }
    };
    let (current, upcoming) = (0..=count)
        .map(|step| (lower(step), upper(step)))
        .find(|(current, upcoming)| u >= current.0 && u <= upcoming.0)
        .unwrap_or_else(|| (lower(count), upper(count)));
    let span = upcoming.0 - current.0;
    let fac = if span > 0.0 {
        (u - current.0) / span
    } else {
        0.0
    };
    let current_weight = current.1 * lerp(point.weight, next_point.weight, current.0);
    let upcoming_weight = upcoming.1 * lerp(point.weight, next_point.weight, upcoming.0);
    current_weight + (upcoming_weight - current_weight) * (3.0 * fac * fac - 2.0 * fac * fac * fac)
}

/// `BKE_mask_spline_resolution`: samples per segment from the control polygon's pixel length.
fn curve_resolution(points: &[Point], cyclic: bool, width: u32, height: u32) -> usize {
    let max_segment = if width != 0 && height != 0 {
        1.0 / f64::from(width.max(height))
    } else {
        0.01
    };
    let mut resolution = 1_usize;
    for index in 0..points.len() {
        let Some(next) = next_index(points.len(), index, cyclic) else {
            break;
        };
        let current = &points[index];
        let following = &points[next];
        let length = current.position.distance(current.handle_right)
            + current.handle_right.distance(following.handle_left)
            + following.handle_left.distance(following.position);
        resolution = resolution.max(floor_capped(length / max_segment, RESOLUTION_MAX));
        if resolution >= RESOLUTION_MAX as usize {
            break;
        }
    }
    resolution.clamp(1, RESOLUTION_MAX as usize)
}

/// `BKE_mask_spline_feather_resolution`: the curve resolution raised by the steepest weight change.
fn feather_resolution(points: &[Point], cyclic: bool, width: u32, height: u32) -> usize {
    let resolution = curve_resolution(points, cyclic, width, height);
    if resolution >= RESOLUTION_MAX as usize {
        return RESOLUTION_MAX as usize;
    }
    let mut max_jump = 0.0_f64;
    for point in points {
        let mut previous_u = 0.0;
        let mut previous_weight = point.weight;
        for feather in &point.feather_points {
            let weight_delta = feather.w - previous_weight;
            let u_delta = feather.u - previous_u;
            if u_delta > FLOAT_EPSILON {
                max_jump = max_jump.max((weight_delta / u_delta).abs());
            }
            previous_u = feather.u;
            previous_weight = feather.w;
        }
    }
    let raised = count_to_f64(resolution) + max_jump / FEATHER_SLOPE_STEP;
    floor_capped(raised, RESOLUTION_MAX).clamp(1, RESOLUTION_MAX as usize)
}

/// `maskrasterize_layer_init_cdt`'s per-segment sample count.
fn sample_resolution(points: &[Point], cyclic: bool, width: u32, height: u32) -> usize {
    let curve = curve_resolution(points, cyclic, width, height) / 4;
    let feather = feather_resolution(points, cyclic, width, height) / 4;
    curve.max(feather).clamp(SAMPLES_MIN, SAMPLES_MAX)
}

/// `BKE_mask_spline_differentiate_with_resolution`: the body polygon. An open spline's last point
/// closes the list, and the polygon closes implicitly.
fn differentiate(
    points: &[Point],
    cyclic: bool,
    resolution: usize,
) -> Result<Vec<DVec2>, MaskError> {
    let segments = segment_count(points.len(), cyclic);
    let mut body = allocate(segments * resolution + usize::from(!cyclic))?;
    for segment in 0..segments {
        for step in 0..resolution {
            let u = count_to_f64(step) / count_to_f64(resolution);
            body.push(segment_point(points, cyclic, segment, u));
        }
    }
    if !cyclic {
        body.push(points[points.len() - 1].position);
    }
    Ok(body)
}

/// `mask_spline_feather_differentiated_points_with_resolution__even`: each sample offset along its
/// normal by the interpolated feather weight.
fn feather_ring(
    points: &[Point],
    cyclic: bool,
    resolution: usize,
) -> Result<Vec<DVec2>, MaskError> {
    let segments = segment_count(points.len(), cyclic);
    let mut ring = allocate(segments * resolution + usize::from(!cyclic))?;
    for segment in 0..segments {
        for step in 0..resolution {
            let u = count_to_f64(step) / count_to_f64(resolution);
            let offset =
                point_normal(points, cyclic, segment, u) * point_weight(points, cyclic, segment, u);
            ring.push(segment_point(points, cyclic, segment, u) + offset);
        }
    }
    if !cyclic {
        let last = points.len() - 2;
        let offset =
            point_normal(points, cyclic, last, 1.0) * point_weight(points, cyclic, last, 1.0);
        ring.push(points[points.len() - 1].position + offset);
    }
    Ok(ring)
}

/// Blender's automatic and vector handles, from the control points. Blender stores them when the
/// handle type is set; here they are derived from the evaluated positions.
pub(super) fn calculate_handles(points: &mut [Point], cyclic: bool) {
    let count = points.len();
    let positions: Vec<DVec2> = points.iter().map(|point| point.position).collect();
    for (index, point) in points.iter_mut().enumerate() {
        if point.left_type == HandleType::Free && point.right_type == HandleType::Free {
            continue;
        }
        let position = positions[index];
        let before = previous_index(count, index, cyclic).map(|previous| positions[previous]);
        let after = next_index(count, index, cyclic).map(|next| positions[next]);
        let previous = before.unwrap_or_else(|| position * 2.0 - after.unwrap_or(position));
        let following = after.unwrap_or_else(|| position * 2.0 - previous);
        let direction_before = position - previous;
        let direction_after = following - position;
        let length_before = nonzero_length(direction_before);
        let length_after = nonzero_length(direction_after);

        if point.left_type == HandleType::Vector {
            point.handle_left = position - direction_before / 3.0;
        }
        if point.right_type == HandleType::Vector {
            point.handle_right = position + direction_after / 3.0;
        }
        if point.left_type == HandleType::Auto || point.right_type == HandleType::Auto {
            let tangent = direction_after / length_after + direction_before / length_before;
            let tangent_length = tangent.length() * AUTO_HANDLE_LENGTH;
            if tangent_length != 0.0 {
                let before_length = length_before.min(5.0 * length_after);
                let after_length = length_after.min(5.0 * before_length);
                if point.left_type == HandleType::Auto {
                    point.handle_left = position - tangent * (before_length / tangent_length);
                }
                if point.right_type == HandleType::Auto {
                    point.handle_right = position + tangent * (after_length / tangent_length);
                }
            }
        }
    }
}

fn nonzero_length(vector: DVec2) -> f64 {
    let length = vector.length();
    if length == 0.0 { 1.0 } else { length }
}

/// `do_aspect_correct` in `BKE_maskrasterize_handle_init`: the shorter axis is stretched about the
/// image centre so the mask's unit square fills a square.
fn aspect_correct(values: &mut [DVec2], width: u32, height: u32) {
    if width == height {
        return;
    }
    let (width, height) = (f64::from(width), f64::from(height));
    if width < height {
        let aspect = width / height;
        for value in values {
            value.x = (value.x - 0.5) / aspect + 0.5;
        }
    } else {
        let aspect = height / width;
        for value in values {
            value.y = (value.y - 0.5) / aspect + 0.5;
        }
    }
}

/// `maskrasterize_spline_differentiate_point_outset`: a ring one `offset` outside the body. Blender
/// steps the normal one index behind the point it moves, so ring point `k` takes the normal at
/// `k - 1`. With `only_zero`, only ring points closer to the body than `offset` are replaced.
fn outset(ring: &mut [DVec2], body: &[DVec2], offset: f64, only_zero: bool) {
    let count = body.len();
    let mut previous = normalized(body[count - 2] - body[count - 1]);
    for index in 0..count {
        let next = normalized(body[(index + count - 1) % count] - body[index]);
        if !only_zero || (ring[index] - body[index]).length_squared() < offset * offset {
            let direction = normalized(previous + next);
            ring[index] = body[index] + DVec2::new(direction.y, -direction.x) * offset;
        }
        previous = next;
    }
}

/// Builds a spline's body and feather ring for a `width` x `height` raster, as
/// `BKE_maskrasterize_handle_init` does with `do_aspect_correct` set. `anti_alias` is Blender's
/// `do_mask_aa`, which adds a one-pixel ring where the feather is absent or narrower than a pixel.
pub(super) fn build_contour(
    points: &[Point],
    cyclic: bool,
    width: u32,
    height: u32,
    use_feather: bool,
    anti_alias: bool,
) -> Result<Contour, MaskError> {
    let resolution = sample_resolution(points, cyclic, width, height);
    let mut body = differentiate(points, cyclic, resolution)?;
    let mut feather = if use_feather {
        Some(feather_ring(points, cyclic, resolution)?)
    } else {
        None
    };
    aspect_correct(&mut body, width, height);
    if let Some(ring) = feather.as_mut() {
        aspect_correct(ring, width, height);
    }
    if anti_alias {
        let offset = 1.0 / f64::from(width.min(height));
        let ring = if let Some(mut ring) = feather.take() {
            outset(&mut ring, &body, offset, true);
            ring
        } else {
            let mut ring = allocate(body.len())?;
            ring.extend_from_slice(&body);
            outset(&mut ring, &body, offset, false);
            ring
        };
        feather = Some(ring);
    }
    Ok(Contour { body, feather })
}
