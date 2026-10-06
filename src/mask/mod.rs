//! Animated polygon masks and their CPU rasterization.
//!
//! Coordinates are pixel-space `(x, y)` with the origin at the top-left. Control
//! points are connected by straight segments; multiple splines are combined as a
//! union. Feathering fades inward from each spline boundary over the configured
//! pixel width.

use std::{error::Error, fmt};

use glam::DVec2;
use serde::{Deserialize, Serialize};

/// A frame-indexed two-dimensional control-point position.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PositionKeyframe {
    /// Frame at which `position` is reached; fractional frames are supported.
    pub frame: f64,
    /// Pixel-space control-point position.
    pub position: DVec2,
}

impl PositionKeyframe {
    /// Creates a position keyframe.
    #[must_use]
    pub const fn new(frame: f64, position: DVec2) -> Self {
        Self { frame, position }
    }
}

/// A polygon control point with an optional linearly interpolated animation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlPoint {
    /// Position used when there are no keyframes.
    pub position: DVec2,
    /// Strictly increasing animation keys; outside their range the nearest key is held.
    pub keyframes: Vec<PositionKeyframe>,
}

impl ControlPoint {
    /// Creates a static control point.
    #[must_use]
    pub const fn new(position: DVec2) -> Self {
        Self {
            position,
            keyframes: Vec::new(),
        }
    }

    /// Replaces the point's animation keys.
    ///
    /// # Errors
    ///
    /// Returns [`MaskError::InvalidParameter`] if keys are empty, non-finite, or
    /// not strictly ordered by frame.
    pub fn with_keyframes(mut self, keyframes: Vec<PositionKeyframe>) -> Result<Self, MaskError> {
        validate_keyframes(&keyframes)?;
        self.keyframes = keyframes;
        Ok(self)
    }

    fn validate(&self) -> Result<(), MaskError> {
        if !self.position.is_finite() {
            return Err(MaskError::InvalidParameter(
                "control-point coordinates must be finite",
            ));
        }
        if !self.keyframes.is_empty() {
            validate_keyframes(&self.keyframes)?;
        }
        Ok(())
    }

    fn evaluate(&self, frame: f64) -> DVec2 {
        if self.keyframes.is_empty() {
            return self.position;
        }
        let upper = self.keyframes.partition_point(|key| key.frame <= frame);
        if upper == 0 {
            return self.keyframes[0].position;
        }
        if upper == self.keyframes.len() {
            return self.keyframes[self.keyframes.len() - 1].position;
        }
        let left = self.keyframes[upper - 1];
        let right = self.keyframes[upper];
        let amount = interpolation_amount(frame, left.frame, right.frame);
        interpolate_position(left.position, right.position, amount)
    }
}

fn interpolation_amount(value: f64, start: f64, end: f64) -> f64 {
    let scale = value.abs().max(start.abs()).max(end.abs());
    if scale == 0.0 {
        return 0.0;
    }
    let scaled_value = value / scale;
    let scaled_start = start / scale;
    let scaled_end = end / scale;
    ((scaled_value - scaled_start) / (scaled_end - scaled_start)).clamp(0.0, 1.0)
}

fn interpolate_scalar(start: f64, end: f64, amount: f64) -> f64 {
    start * (1.0 - amount) + end * amount
}

fn interpolate_position(left: DVec2, right: DVec2, amount: f64) -> DVec2 {
    DVec2::new(
        interpolate_scalar(left.x, right.x, amount),
        interpolate_scalar(left.y, right.y, amount),
    )
}

/// A closed polygon contour with a uniform inward feather width.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaskSpline {
    /// Ordered points joined by straight segments and closed back to the first point.
    pub points: Vec<ControlPoint>,
    /// Inward edge-fade width in pixels; zero gives a hard edge.
    pub feather: f64,
}

impl MaskSpline {
    /// Creates a hard-edged closed polygon from pixel-space vertices.
    ///
    /// # Errors
    ///
    /// Returns [`MaskError::InvalidParameter`] for fewer than three or non-finite
    /// vertices.
    pub fn new(points: Vec<DVec2>) -> Result<Self, MaskError> {
        Self::from_control_points(points.into_iter().map(ControlPoint::new).collect(), 0.0)
    }

    /// Creates a polygon from animated control points and a feather width.
    ///
    /// # Errors
    ///
    /// Returns [`MaskError::InvalidParameter`] for fewer than three points, invalid
    /// point/keyframe values, or a negative/non-finite feather width.
    pub fn from_control_points(points: Vec<ControlPoint>, feather: f64) -> Result<Self, MaskError> {
        let spline = Self { points, feather };
        spline.validate()?;
        Ok(spline)
    }

    /// Sets the inward feather width in pixels.
    ///
    /// # Errors
    ///
    /// Returns [`MaskError::InvalidParameter`] unless the width is finite and
    /// non-negative.
    pub fn with_feather(mut self, feather: f64) -> Result<Self, MaskError> {
        self.feather = feather;
        self.validate()?;
        Ok(self)
    }

    fn validate(&self) -> Result<(), MaskError> {
        if self.points.len() < 3 {
            return Err(MaskError::InvalidParameter(
                "a mask spline requires at least three control points",
            ));
        }
        if !self.feather.is_finite() || self.feather < 0.0 {
            return Err(MaskError::InvalidParameter(
                "mask feather width must be finite and non-negative",
            ));
        }
        for point in &self.points {
            point.validate()?;
        }
        Ok(())
    }
}

/// A union of animated polygon splines rasterized into coverage values.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Mask {
    /// Contours combined by maximum coverage; an empty mask covers no pixels.
    pub splines: Vec<MaskSpline>,
}

impl Mask {
    /// Creates and validates a mask.
    ///
    /// # Errors
    ///
    /// Returns [`MaskError::InvalidParameter`] if any spline is invalid.
    pub fn new(splines: Vec<MaskSpline>) -> Result<Self, MaskError> {
        let mask = Self { splines };
        mask.validate()?;
        Ok(mask)
    }

    /// Rasterizes coverage into floating-point values in `[0, 1]`.
    ///
    /// Each pixel is sampled at its center. Keyframed points interpolate linearly
    /// between keys and hold the nearest endpoint outside the keyed frame range.
    ///
    /// # Errors
    ///
    /// Returns [`MaskError::InvalidParameter`] for zero dimensions, a non-finite
    /// frame, or invalid mask data; returns [`MaskError::AllocationFailed`] if
    /// buffers cannot be allocated.
    pub fn rasterize(&self, width: u32, height: u32, frame: f64) -> Result<MaskImage, MaskError> {
        self.validate()?;
        if width == 0 || height == 0 {
            return Err(MaskError::InvalidParameter(
                "mask image dimensions must be non-zero",
            ));
        }
        if !frame.is_finite() {
            return Err(MaskError::InvalidParameter(
                "mask evaluation frame must be finite",
            ));
        }

        let width_usize = usize::try_from(width)
            .map_err(|_| MaskError::InvalidParameter("mask image width is too large"))?;
        let height_usize = usize::try_from(height)
            .map_err(|_| MaskError::InvalidParameter("mask image height is too large"))?;
        let pixel_count =
            width_usize
                .checked_mul(height_usize)
                .ok_or(MaskError::InvalidParameter(
                    "mask image dimensions overflow",
                ))?;

        let mut evaluated = Vec::new();
        evaluated
            .try_reserve_exact(self.splines.len())
            .map_err(|_| MaskError::AllocationFailed)?;
        for spline in &self.splines {
            let mut points = Vec::new();
            points
                .try_reserve_exact(spline.points.len())
                .map_err(|_| MaskError::AllocationFailed)?;
            points.extend(spline.points.iter().map(|point| point.evaluate(frame)));
            evaluated.push(points);
        }

        let mut pixels = Vec::new();
        pixels
            .try_reserve_exact(pixel_count)
            .map_err(|_| MaskError::AllocationFailed)?;
        pixels.resize(pixel_count, 0.0);

        for (y_index, y) in (0..height).enumerate() {
            for (x_index, x) in (0..width).enumerate() {
                let sample = DVec2::new(f64::from(x) + 0.5, f64::from(y) + 0.5);
                let mut coverage = 0.0_f64;
                for (spline, points) in self.splines.iter().zip(&evaluated) {
                    let spline_coverage = polygon_coverage(sample, points, spline.feather);
                    coverage = coverage.max(spline_coverage);
                    if coverage >= 1.0 {
                        break;
                    }
                }
                pixels[y_index * width_usize + x_index] = coverage_to_f32(coverage);
            }
        }

        Ok(MaskImage {
            width,
            height,
            pixels,
        })
    }

    /// Rasterizes and quantizes coverage into 8-bit values in `[0, 255]`.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::rasterize`].
    pub fn rasterize_u8(
        &self,
        width: u32,
        height: u32,
        frame: f64,
    ) -> Result<MaskImage8, MaskError> {
        Ok(self.rasterize(width, height, frame)?.to_u8())
    }

    fn validate(&self) -> Result<(), MaskError> {
        for spline in &self.splines {
            spline.validate()?;
        }
        Ok(())
    }
}

/// Floating-point coverage image in row-major order.
#[derive(Clone, Debug, PartialEq)]
pub struct MaskImage {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Row-major coverage values in `[0, 1]`.
    pub pixels: Vec<f32>,
}

impl MaskImage {
    /// Returns the coverage at `(x, y)`, or `None` when the coordinate is out of bounds.
    #[must_use]
    pub fn get(&self, x: u32, y: u32) -> Option<f32> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let width = usize::try_from(self.width).ok()?;
        let x = usize::try_from(x).ok()?;
        let y = usize::try_from(y).ok()?;
        self.pixels
            .get(y.checked_mul(width)?.checked_add(x)?)
            .copied()
    }

    /// Converts coverage to rounded 8-bit values, mapping `0` to `0` and `1` to `255`.
    #[must_use]
    pub fn to_u8(&self) -> MaskImage8 {
        MaskImage8 {
            width: self.width,
            height: self.height,
            pixels: self
                .pixels
                .iter()
                .map(|value| unit_to_byte(*value))
                .collect(),
        }
    }
}

/// 8-bit coverage image in row-major order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaskImage8 {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Row-major coverage values in `[0, 255]`.
    pub pixels: Vec<u8>,
}

impl MaskImage8 {
    /// Returns the coverage at `(x, y)`, or `None` when the coordinate is out of bounds.
    #[must_use]
    pub fn get(&self, x: u32, y: u32) -> Option<u8> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let width = usize::try_from(self.width).ok()?;
        let x = usize::try_from(x).ok()?;
        let y = usize::try_from(y).ok()?;
        self.pixels
            .get(y.checked_mul(width)?.checked_add(x)?)
            .copied()
    }
}

/// Invalid mask data or rasterization resource failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MaskError {
    /// A mask parameter, frame, or dimension is outside its valid domain.
    InvalidParameter(&'static str),
    /// Required image or evaluation buffers could not be allocated.
    AllocationFailed,
}

impl fmt::Display for MaskError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(message) => formatter.write_str(message),
            Self::AllocationFailed => formatter.write_str("mask raster allocation failed"),
        }
    }
}

impl Error for MaskError {}

fn validate_keyframes(keyframes: &[PositionKeyframe]) -> Result<(), MaskError> {
    if keyframes.is_empty() {
        return Err(MaskError::InvalidParameter(
            "animated control points require at least one keyframe",
        ));
    }
    let mut previous_frame = f64::NEG_INFINITY;
    for keyframe in keyframes {
        if !keyframe.frame.is_finite()
            || !keyframe.position.is_finite()
            || keyframe.frame <= previous_frame
        {
            return Err(MaskError::InvalidParameter(
                "keyframes must have finite values and strictly increasing frames",
            ));
        }
        previous_frame = keyframe.frame;
    }
    Ok(())
}

fn polygon_coverage(sample: DVec2, points: &[DVec2], feather: f64) -> f64 {
    let inside = point_inside_polygon(sample, points);
    if !inside {
        return 0.0;
    }
    if feather <= 0.0 {
        return 1.0;
    }
    let distance = distance_to_polygon(sample, points);
    let amount = (distance / feather).clamp(0.0, 1.0);
    amount * amount * (3.0 - 2.0 * amount)
}

fn point_inside_polygon(point: DVec2, vertices: &[DVec2]) -> bool {
    let mut inside = false;
    let mut previous = vertices[vertices.len() - 1];
    for &current in vertices {
        if (current.y > point.y) != (previous.y > point.y) {
            let amount = interpolation_amount(point.y, current.y, previous.y);
            let intersection_x = interpolate_scalar(current.x, previous.x, amount);
            if point.x < intersection_x {
                inside = !inside;
            }
        }
        previous = current;
    }
    inside
}

fn distance_to_polygon(point: DVec2, vertices: &[DVec2]) -> f64 {
    let mut distance = f64::INFINITY;
    let mut previous = vertices[vertices.len() - 1];
    for &current in vertices {
        distance = distance.min(distance_to_segment(point, previous, current));
        previous = current;
    }
    distance
}

fn distance_to_segment(point: DVec2, start: DVec2, end: DVec2) -> f64 {
    let scale = point
        .x
        .abs()
        .max(point.y.abs())
        .max(start.x.abs())
        .max(start.y.abs())
        .max(end.x.abs())
        .max(end.y.abs());
    if scale == 0.0 {
        return 0.0;
    }

    let scaled_point = point / scale;
    let scaled_start = start / scale;
    let scaled_end = end / scale;
    let segment = scaled_end - scaled_start;
    let length_squared = segment.length_squared();
    let scaled_distance = if length_squared <= f64::MIN_POSITIVE {
        scaled_point.distance(scaled_start)
    } else {
        let amount = ((scaled_point - scaled_start).dot(segment) / length_squared).clamp(0.0, 1.0);
        scaled_point.distance(scaled_start + amount * segment)
    };
    scaled_distance * scale
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "rasterized coverage is bounded to the representable [0, 1] interval"
)]
fn coverage_to_f32(coverage: f64) -> f32 {
    coverage as f32
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "coverage is clamped before quantizing to the 8-bit range"
)]
fn unit_to_byte(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests use unwrap for concise setup")]

    use glam::DVec2;

    use super::{ControlPoint, Mask, MaskSpline, PositionKeyframe};

    fn square(min: f64, max: f64) -> MaskSpline {
        MaskSpline::new(vec![
            DVec2::new(min, min),
            DVec2::new(max, min),
            DVec2::new(max, max),
            DVec2::new(min, max),
        ])
        .unwrap()
    }

    #[test]
    fn rasterized_square_area_matches_polygon_area() {
        let mask = Mask::new(vec![square(1.0, 11.0)]).unwrap();
        let raster = mask.rasterize(16, 16, 1.0).unwrap();

        let covered_area: f64 = raster.pixels.iter().map(|value| f64::from(*value)).sum();
        assert!((covered_area - 100.0).abs() < 1.0);
        assert_eq!(raster.pixels.len(), 256);
    }

    #[test]
    fn point_keyframes_interpolate_and_move_the_rasterized_polygon() {
        let base = [
            DVec2::new(1.0, 1.0),
            DVec2::new(5.0, 1.0),
            DVec2::new(5.0, 5.0),
            DVec2::new(1.0, 5.0),
        ];
        let points = base
            .into_iter()
            .map(|position| {
                ControlPoint::new(position)
                    .with_keyframes(vec![
                        PositionKeyframe::new(1.0, position),
                        PositionKeyframe::new(3.0, position + DVec2::new(4.0, 0.0)),
                    ])
                    .unwrap()
            })
            .collect();
        let mask = Mask::new(vec![MaskSpline::from_control_points(points, 0.0).unwrap()]).unwrap();

        let first = mask.rasterize(12, 8, 1.0).unwrap();
        let halfway = mask.rasterize(12, 8, 2.0).unwrap();
        let last = mask.rasterize(12, 8, 3.0).unwrap();
        assert_eq!(first.get(2, 2), Some(1.0));
        assert_eq!(halfway.get(4, 2), Some(1.0));
        assert_eq!(last.get(6, 2), Some(1.0));
        assert_eq!(last.get(2, 2), Some(0.0));
    }
    #[test]
    fn animated_feathered_spline_interpolates_and_holds_boundary_coverage() {
        let base = [
            DVec2::new(2.0, 2.0),
            DVec2::new(12.0, 2.0),
            DVec2::new(12.0, 12.0),
            DVec2::new(2.0, 12.0),
        ];
        let points = base
            .into_iter()
            .map(|position| {
                ControlPoint::new(position)
                    .with_keyframes(vec![
                        PositionKeyframe::new(2.0, position),
                        PositionKeyframe::new(6.0, position + DVec2::new(8.0, 0.0)),
                    ])
                    .unwrap()
            })
            .collect();
        let mask = Mask::new(vec![MaskSpline::from_control_points(points, 2.0).unwrap()]).unwrap();

        let before = mask.rasterize(20, 16, 1.0).unwrap();
        let between = mask.rasterize(20, 16, 4.0).unwrap();
        let after = mask.rasterize(20, 16, 7.0).unwrap();
        assert_eq!(before.get(2, 7), Some(0.15625));
        assert_eq!(between.get(6, 7), Some(0.15625));
        assert_eq!(after.get(10, 7), Some(0.15625));
        assert_eq!(between.get(16, 7), Some(0.0));
        assert_eq!(between.get(9, 7), Some(1.0));
    }
    #[test]
    fn animation_interpolation_avoids_overflow_between_finite_extreme_frames() {
        let base = [
            DVec2::new(2.0, 2.0),
            DVec2::new(6.0, 2.0),
            DVec2::new(6.0, 6.0),
            DVec2::new(2.0, 6.0),
        ];
        let points = base
            .into_iter()
            .map(|position| {
                ControlPoint::new(position)
                    .with_keyframes(vec![
                        PositionKeyframe::new(-f64::MAX, position),
                        PositionKeyframe::new(f64::MAX, position + DVec2::new(4.0, 0.0)),
                    ])
                    .unwrap()
            })
            .collect();
        let mask = Mask::new(vec![MaskSpline::from_control_points(points, 0.0).unwrap()]).unwrap();
        let midpoint = mask.rasterize(12, 8, 0.0).unwrap();

        assert_eq!(midpoint.get(2, 3), Some(0.0));
        assert_eq!(midpoint.get(5, 3), Some(1.0));
    }
    #[test]
    fn feathered_rasterization_handles_large_finite_polygon_coordinates() {
        let extent = 1.0e308;
        let spline = MaskSpline::new(vec![
            DVec2::new(-extent, -extent),
            DVec2::new(extent, -extent),
            DVec2::new(0.0, extent),
        ])
        .unwrap()
        .with_feather(2.0)
        .unwrap();
        let raster = Mask::new(vec![spline])
            .unwrap()
            .rasterize(1, 1, 1.0)
            .unwrap();

        assert_eq!(raster.get(0, 0), Some(1.0));
    }

    #[test]
    fn feather_softens_the_inside_edge_without_changing_the_interior() {
        let soft_spline = square(2.0, 12.0).with_feather(2.0).unwrap();
        let raster = Mask::new(vec![soft_spline])
            .unwrap()
            .rasterize(16, 16, 1.0)
            .unwrap();

        assert_eq!(raster.get(7, 7), Some(1.0));
        let edge = raster.get(2, 7).unwrap();
        assert!(edge > 0.0 && edge < 1.0);
        assert_eq!(raster.get(1, 7), Some(0.0));
        assert!(raster.to_u8().get(2, 7).is_some_and(|value| value > 0));
    }

    #[test]
    fn invalid_dimensions_and_keyframes_are_rejected() {
        let mask = Mask::new(vec![square(1.0, 4.0)]).unwrap();
        assert!(mask.rasterize(0, 8, 1.0).is_err());
        assert!(mask.rasterize(8, 8, f64::NAN).is_err());

        assert!(
            ControlPoint::new(DVec2::ZERO)
                .with_keyframes(vec![
                    PositionKeyframe::new(1.0, DVec2::ZERO),
                    PositionKeyframe::new(1.0, DVec2::ONE),
                ])
                .is_err()
        );
        assert!(square(1.0, 4.0).with_feather(f64::INFINITY).is_err());
    }

    #[test]
    fn persisted_mask_roundtrips_with_unknown_fields_rejected() {
        let mask = Mask::new(vec![square(1.0, 5.0)]).unwrap();
        let serialized = serde_json::to_vec(&mask).unwrap();
        let restored: Mask = serde_json::from_slice(&serialized).unwrap();
        assert_eq!(restored, mask);

        let mut value: serde_json::Value = serde_json::from_slice(&serialized).unwrap();
        value["unexpected"] = serde_json::json!(true);
        assert!(serde_json::from_value::<Mask>(value).is_err());
    }
}
