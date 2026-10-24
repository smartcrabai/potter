//! Animated Bezier masks and their CPU rasterization, following Blender 5.2.2.
//!
//! Coordinates are pixel-space `(x, y)` with the origin at the top-left. A spline is a cyclic or
//! open chain of cubic Bezier segments; each point carries optional handles and feather weights.
//! Splines are combined as Blender's default Merge Add layers. The mask is sampled in Blender's
//! normalized mask space (`x / width`, `1 - y / height`) through `evaluate` and `raster`.

mod evaluate;
mod raster;

use std::{error::Error, fmt};

use glam::DVec2;
use serde::{Deserialize, Serialize};

use evaluate::Point;

/// A frame-indexed control-point key: a position and optional animated handles.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PositionKeyframe {
    /// Frame at which the key is reached; fractional frames are supported.
    pub frame: f64,
    /// Pixel-space control-point position.
    pub position: DVec2,
    /// Pixel-space left handle; omitted, the point's stored left handle applies.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handle_left: Option<DVec2>,
    /// Pixel-space right handle; omitted, the point's stored right handle applies.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handle_right: Option<DVec2>,
}

impl PositionKeyframe {
    /// Creates a position keyframe without handle keys.
    #[must_use]
    pub const fn new(frame: f64, position: DVec2) -> Self {
        Self {
            frame,
            position,
            handle_left: None,
            handle_right: None,
        }
    }
}

/// A Bezier handle type: how a handle is placed relative to its point.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HandleType {
    /// The handle is stored as given.
    #[default]
    Free,
    /// The handle is smoothed from the neighbouring control points.
    Auto,
    /// The handle points one third of the way to the neighbouring control point.
    Vector,
}

/// A feather-weight multiplier at parameter `u` along a point's outgoing segment.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatherPoint {
    /// Segment parameter in `[0, 1]`, strictly increasing along the point's list.
    pub u: f64,
    /// Multiplier of the interpolated feather weight at `u`.
    pub w: f64,
}

/// A Bezier control point with optional linearly interpolated animation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlPoint {
    /// Position used when there are no keyframes.
    pub position: DVec2,
    /// Strictly increasing animation keys; outside their range the nearest key is held.
    pub keyframes: Vec<PositionKeyframe>,
    /// Stored pixel-space left handle; omitted, it coincides with the position.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handle_left: Option<DVec2>,
    /// Stored pixel-space right handle; omitted, it coincides with the position.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handle_right: Option<DVec2>,
    /// Left handle type.
    #[serde(default)]
    pub handle_left_type: HandleType,
    /// Right handle type.
    #[serde(default)]
    pub handle_right_type: HandleType,
    /// Feather width in pixels at this point; omitted, the spline's `feather` applies.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<f64>,
    /// Feather-weight multipliers along the outgoing segment.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub feather_points: Vec<FeatherPoint>,
}

impl ControlPoint {
    /// Creates a control point with coincident handles and no feather of its own.
    #[must_use]
    pub const fn new(position: DVec2) -> Self {
        Self {
            position,
            keyframes: Vec::new(),
            handle_left: None,
            handle_right: None,
            handle_left_type: HandleType::Free,
            handle_right_type: HandleType::Free,
            weight: None,
            feather_points: Vec::new(),
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
        if !self.position.is_finite()
            || self.handle_left.is_some_and(|handle| !handle.is_finite())
            || self.handle_right.is_some_and(|handle| !handle.is_finite())
            || self.weight.is_some_and(|weight| !weight.is_finite())
        {
            return Err(MaskError::InvalidParameter(
                "control-point coordinates, handles and weights must be finite",
            ));
        }
        if !self.keyframes.is_empty() {
            validate_keyframes(&self.keyframes)?;
        }
        let mut previous_u = f64::NEG_INFINITY;
        for feather in &self.feather_points {
            if !feather.u.is_finite()
                || !(0.0..=1.0).contains(&feather.u)
                || feather.u <= previous_u
                || !feather.w.is_finite()
            {
                return Err(MaskError::InvalidParameter(
                    "feather points need finite, strictly increasing u in [0, 1] and finite w",
                ));
            }
            previous_u = feather.u;
        }
        Ok(())
    }

    /// The point's pixel-space position and handles at `frame`. Keys interpolate linearly; a key
    /// without a handle falls back to the stored handle, then to the position.
    fn evaluate(&self, frame: f64) -> (DVec2, DVec2, DVec2) {
        let stored_left = self.handle_left.unwrap_or(self.position);
        let stored_right = self.handle_right.unwrap_or(self.position);
        if self.keyframes.is_empty() {
            return (self.position, stored_left, stored_right);
        }
        let upper = self.keyframes.partition_point(|key| key.frame <= frame);
        if upper == 0 {
            return self.key_values(&self.keyframes[0]);
        }
        if upper == self.keyframes.len() {
            return self.key_values(&self.keyframes[upper - 1]);
        }
        let left = self.key_values(&self.keyframes[upper - 1]);
        let right = self.key_values(&self.keyframes[upper]);
        let amount = interpolation_amount(
            frame,
            self.keyframes[upper - 1].frame,
            self.keyframes[upper].frame,
        );
        (
            interpolate_position(left.0, right.0, amount),
            interpolate_position(left.1, right.1, amount),
            interpolate_position(left.2, right.2, amount),
        )
    }

    fn key_values(&self, key: &PositionKeyframe) -> (DVec2, DVec2, DVec2) {
        (
            key.position,
            key.handle_left.or(self.handle_left).unwrap_or(key.position),
            key.handle_right
                .or(self.handle_right)
                .unwrap_or(key.position),
        )
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

const fn default_cyclic() -> bool {
    true
}

/// A Bezier contour, filled and feathered as one Blender mask layer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaskSpline {
    /// Ordered control points joined by cubic Bezier segments.
    pub points: Vec<ControlPoint>,
    /// Feather width in pixels for points without their own `weight`.
    pub feather: f64,
    /// Whether the last point connects back to the first; an open spline fills its closed chord.
    #[serde(default = "default_cyclic")]
    pub cyclic: bool,
}

impl MaskSpline {
    /// Creates a hard-edged cyclic polygon from straight-handle pixel-space vertices.
    ///
    /// # Errors
    ///
    /// Returns [`MaskError::InvalidParameter`] for fewer than three or non-finite vertices.
    pub fn new(points: Vec<DVec2>) -> Result<Self, MaskError> {
        Self::from_control_points(points.into_iter().map(ControlPoint::new).collect(), 0.0)
    }

    /// Creates a cyclic spline from control points and a feather width.
    ///
    /// # Errors
    ///
    /// Returns [`MaskError::InvalidParameter`] for invalid point or keyframe values, fewer than
    /// three points, or an invalid feather width.
    pub fn from_control_points(points: Vec<ControlPoint>, feather: f64) -> Result<Self, MaskError> {
        let spline = Self {
            points,
            feather,
            cyclic: true,
        };
        spline.validate()?;
        Ok(spline)
    }

    /// Sets whether the spline is cyclic.
    ///
    /// # Errors
    ///
    /// Returns [`MaskError::InvalidParameter`] if the point count is invalid for the setting.
    pub fn with_cyclic(mut self, cyclic: bool) -> Result<Self, MaskError> {
        self.cyclic = cyclic;
        self.validate()?;
        Ok(self)
    }

    /// Sets the feather width in pixels.
    ///
    /// # Errors
    ///
    /// Returns [`MaskError::InvalidParameter`] unless the width is finite and non-negative.
    pub fn with_feather(mut self, feather: f64) -> Result<Self, MaskError> {
        self.feather = feather;
        self.validate()?;
        Ok(self)
    }

    fn validate(&self) -> Result<(), MaskError> {
        if self.points.len() < if self.cyclic { 3 } else { 2 } {
            return Err(MaskError::InvalidParameter(
                "a cyclic mask spline needs three points and an open one needs two",
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

/// A union of Bezier splines rasterized into coverage values.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Mask {
    /// Splines combined by Blender's Merge Add; an empty mask covers no pixels.
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

    /// Rasterizes coverage into floating-point values in `[0, 1]`, with feathering on.
    ///
    /// Each pixel is sampled at its center, as Blender's mask buffer does. Keyframed points and
    /// handles interpolate linearly between keys and hold the nearest key outside their range.
    ///
    /// # Errors
    ///
    /// Returns [`MaskError::InvalidParameter`] for zero dimensions, a non-finite frame, or invalid
    /// mask data; returns [`MaskError::AllocationFailed`] if buffers cannot be allocated.
    pub fn rasterize(&self, width: u32, height: u32, frame: f64) -> Result<MaskImage, MaskError> {
        self.rasterize_with_feather(width, height, frame, true)
    }

    /// Rasterizes with Blender's `Feather` option: off, the feather is dropped and only the
    /// one-pixel anti-aliasing ring remains.
    pub(crate) fn rasterize_with_feather(
        &self,
        width: u32,
        height: u32,
        frame: f64,
        use_feather: bool,
    ) -> Result<MaskImage, MaskError> {
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

        let mut layers = Vec::new();
        layers
            .try_reserve_exact(self.splines.len())
            .map_err(|_| MaskError::AllocationFailed)?;
        for spline in &self.splines {
            let mut points = spline_points(spline, frame, width, height)?;
            evaluate::calculate_handles(&mut points, spline.cyclic);
            let contour =
                evaluate::build_contour(&points, spline.cyclic, width, height, use_feather, true)?;
            layers.push(raster::Layer::new(contour));
        }

        let mut pixels = Vec::new();
        pixels
            .try_reserve_exact(pixel_count)
            .map_err(|_| MaskError::AllocationFailed)?;
        for row in 0..height {
            for column in 0..width {
                let sample = DVec2::new(
                    (f64::from(column) + 0.5) / f64::from(width),
                    1.0 - (f64::from(row) + 0.5) / f64::from(height),
                );
                let coverage = layers.iter().fold(0.0_f64, |value, layer| {
                    let layer_value = layer.value(sample);
                    (value + layer_value * (1.0 - value)).clamp(0.0, 1.0)
                });
                pixels.push(coverage_to_f32(coverage));
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

/// The spline's points at `frame`, in Blender's normalized mask space, with feather weights in
/// normalized units (`pixels / max(width, height)`).
fn spline_points(
    spline: &MaskSpline,
    frame: f64,
    width: u32,
    height: u32,
) -> Result<Vec<Point>, MaskError> {
    let mut points = Vec::new();
    points
        .try_reserve_exact(spline.points.len())
        .map_err(|_| MaskError::AllocationFailed)?;
    let pixel_size = 1.0 / f64::from(width.max(height));
    let squeeze = f64::from(width.min(height)) / f64::from(width.max(height));
    let to_mask_space = |position: DVec2| {
        let mut position = DVec2::new(
            position.x / f64::from(width),
            1.0 - position.y / f64::from(height),
        );
        if width > height {
            position.y = (position.y - 0.5) * squeeze + 0.5;
        } else if height > width {
            position.x = (position.x - 0.5) * squeeze + 0.5;
        }
        position
    };
    for point in &spline.points {
        let (position, handle_left, handle_right) = point.evaluate(frame);
        points.push(Point {
            position: to_mask_space(position),
            handle_left: to_mask_space(handle_left),
            handle_right: to_mask_space(handle_right),
            left_type: point.handle_left_type,
            right_type: point.handle_right_type,
            weight: point.weight.unwrap_or(spline.feather) * pixel_size,
            feather_points: point.feather_points.clone(),
        });
    }
    Ok(points)
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
            || keyframe
                .handle_left
                .is_some_and(|handle| !handle.is_finite())
            || keyframe
                .handle_right
                .is_some_and(|handle| !handle.is_finite())
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

    use super::{
        ControlPoint, FeatherPoint, HandleType, Mask, MaskImage, MaskSpline, PositionKeyframe,
    };

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
    fn square_interior_is_fully_covered_and_exterior_is_empty() {
        let raster = Mask::new(vec![square(1.0, 11.0)])
            .unwrap()
            .rasterize(16, 16, 1.0)
            .unwrap();

        assert_eq!(raster.get(5, 5), Some(1.0));
        assert_eq!(raster.get(15, 15), Some(0.0));
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
    fn clockwise_feather_extends_outside_the_contour() {
        // Pixel y grows downward, so these vertices run clockwise on screen.
        let spline = square(4.0, 12.0).with_feather(2.0).unwrap();
        let raster = Mask::new(vec![spline])
            .unwrap()
            .rasterize(16, 16, 1.0)
            .unwrap();

        assert_eq!(raster.get(8, 8), Some(1.0));
        let outer = raster.get(3, 8).unwrap();
        assert!(outer > 0.0 && outer < 1.0, "feather value {outer}");
        assert_eq!(raster.get(0, 8), Some(0.0));
    }

    #[test]
    fn counter_clockwise_feather_lies_inside_and_is_hidden_by_the_fill() {
        // The same square in the opposite order: its ring points inward, under the filled body.
        let points = [
            DVec2::new(4.0, 4.0),
            DVec2::new(4.0, 12.0),
            DVec2::new(12.0, 12.0),
            DVec2::new(12.0, 4.0),
        ];
        let spline = MaskSpline::new(points.to_vec())
            .unwrap()
            .with_feather(2.0)
            .unwrap();
        let raster = Mask::new(vec![spline])
            .unwrap()
            .rasterize(16, 16, 1.0)
            .unwrap();

        assert_eq!(raster.get(4, 8), Some(1.0));
        assert_eq!(raster.get(3, 8), Some(0.0));
    }

    #[test]
    fn point_weights_and_handles_bend_the_contour() {
        let points = vec![
            ControlPoint {
                handle_left: Some(DVec2::new(2.0, 14.0)),
                handle_right: Some(DVec2::new(14.0, 14.0)),
                ..ControlPoint::new(DVec2::new(8.0, 2.0))
            },
            ControlPoint::new(DVec2::new(14.0, 8.0)),
            ControlPoint::new(DVec2::new(8.0, 14.0)),
            ControlPoint::new(DVec2::new(2.0, 8.0)),
        ];
        let curved = Mask::new(vec![MaskSpline::from_control_points(points, 0.0).unwrap()])
            .unwrap()
            .rasterize(16, 16, 1.0)
            .unwrap();
        let straight = Mask::new(vec![square(2.0, 14.0)])
            .unwrap()
            .rasterize(16, 16, 1.0)
            .unwrap();

        assert_eq!(curved.get(8, 8), Some(1.0));
        assert_ne!(curved.get(13, 3), straight.get(13, 3));
    }

    #[test]
    fn auto_handles_bulge_the_contour_past_the_straight_edges() {
        let corners = [
            DVec2::new(8.0, 2.0),
            DVec2::new(14.0, 8.0),
            DVec2::new(8.0, 14.0),
            DVec2::new(2.0, 8.0),
        ];
        let auto = corners.map(|position| ControlPoint {
            handle_left_type: HandleType::Auto,
            handle_right_type: HandleType::Auto,
            ..ControlPoint::new(position)
        });
        let smoothed = Mask::new(vec![
            MaskSpline::from_control_points(auto.to_vec(), 0.0).unwrap(),
        ])
        .unwrap()
        .rasterize(16, 16, 1.0)
        .unwrap();
        let straight = Mask::new(vec![MaskSpline::new(corners.to_vec()).unwrap()])
            .unwrap()
            .rasterize(16, 16, 1.0)
            .unwrap();

        // Auto handles bulge the curve past the diamond's straight edges, so it covers more area.
        let covered = |raster: &MaskImage| {
            raster
                .pixels
                .iter()
                .map(|value| f64::from(*value))
                .sum::<f64>()
        };
        assert!(covered(&smoothed) > covered(&straight) + 1.0);
    }

    #[test]
    fn feather_point_multiplier_narrows_the_ring_along_its_segment() {
        let corners = [
            DVec2::new(4.0, 4.0),
            DVec2::new(12.0, 4.0),
            DVec2::new(12.0, 12.0),
            DVec2::new(4.0, 12.0),
        ];
        let weighted = |feather_points: Vec<FeatherPoint>| {
            let mut points: Vec<ControlPoint> = corners.map(ControlPoint::new).to_vec();
            points[0].weight = Some(4.0);
            points[0].feather_points = feather_points;
            Mask::new(vec![MaskSpline::from_control_points(points, 0.0).unwrap()])
                .unwrap()
                .rasterize(16, 16, 1.0)
                .unwrap()
        };
        let plain = weighted(Vec::new());
        // A multiplier of zero at the segment's midpoint pinches the 4-pixel ring to nothing there.
        let pinched = weighted(vec![FeatherPoint { u: 0.5, w: 0.0 }]);

        assert!(plain.get(8, 2).unwrap() > 0.0);
        assert_eq!(pinched.get(8, 2), Some(0.0));
    }

    #[test]
    fn open_spline_fills_its_closed_chord() {
        let points = [
            DVec2::new(2.0, 13.0),
            DVec2::new(8.0, 2.0),
            DVec2::new(14.0, 13.0),
        ]
        .map(ControlPoint::new)
        .to_vec();
        let spline = MaskSpline::from_control_points(points, 0.0)
            .unwrap()
            .with_cyclic(false)
            .unwrap();
        let raster = Mask::new(vec![spline])
            .unwrap()
            .rasterize(16, 16, 1.0)
            .unwrap();

        assert_eq!(raster.get(8, 10), Some(1.0));
        assert_eq!(raster.get(8, 15), Some(0.0));
    }

    #[test]
    fn feather_option_off_keeps_only_the_anti_aliased_edge() {
        let spline = square(4.0, 12.0).with_feather(4.0).unwrap();
        let mask = Mask::new(vec![spline]).unwrap();
        let feathered = mask.rasterize(16, 16, 1.0).unwrap();
        let unfeathered = mask.rasterize_with_feather(16, 16, 1.0, false).unwrap();

        assert!(feathered.get(3, 8).unwrap() > unfeathered.get(3, 8).unwrap());
        assert_eq!(unfeathered.get(0, 8), Some(0.0));
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
        assert!(
            MaskSpline::from_control_points(vec![ControlPoint::new(DVec2::ZERO); 2], 0.0).is_err()
        );
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
