use std::borrow::Cow;

use glam::{DMat4, DQuat, DVec3};
use serde::{Deserialize, Serialize};

use crate::error::{PotError, Result};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GrayFrame {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<f32>,
}

impl GrayFrame {
    pub fn new(width: usize, height: usize, pixels: Vec<f32>) -> Result<Self> {
        let frame = Self {
            width,
            height,
            pixels,
        };
        frame.validate()?;
        Ok(frame)
    }

    pub fn validate(&self) -> Result<()> {
        let pixel_count = self
            .width
            .checked_mul(self.height)
            .ok_or_else(|| PotError::invalid_argument("grayscale frame dimensions overflow"))?;
        if self.width == 0 || self.height == 0 || pixel_count != self.pixels.len() {
            return Err(PotError::invalid_argument(
                "grayscale frame dimensions do not match its pixel data",
            ));
        }
        if self.pixels.iter().any(|pixel| !pixel.is_finite()) {
            return Err(PotError::invalid_argument(
                "grayscale frame pixels must be finite",
            ));
        }
        Ok(())
    }

    #[must_use]
    pub const fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "coordinates are checked against frame bounds before conversion"
    )]
    fn sample(&self, x: f64, y: f64) -> Option<f64> {
        if !x.is_finite()
            || !y.is_finite()
            || x < 0.0
            || y < 0.0
            || x > self.width.saturating_sub(1) as f64
            || y > self.height.saturating_sub(1) as f64
        {
            return None;
        }
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let fx = x - x0 as f64;
        let fy = y - y0 as f64;
        let top_left = f64::from(self.pixels[y0 * self.width + x0]);
        let top_right = f64::from(self.pixels[y0 * self.width + x1]);
        let bottom_left = f64::from(self.pixels[y1 * self.width + x0]);
        let bottom_right = f64::from(self.pixels[y1 * self.width + x1]);
        Some(
            (top_left * (1.0 - fx) + top_right * fx) * (1.0 - fy)
                + (bottom_left * (1.0 - fx) + bottom_right * fx) * fy,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MovieClip {
    pub id: String,
    pub name: String,
    pub frame_start: i32,
    pub frames: Vec<GrayFrame>,
}

impl MovieClip {
    pub fn new(id: String, name: String, frame_start: i32, frames: Vec<GrayFrame>) -> Result<Self> {
        let clip = Self {
            id,
            name,
            frame_start,
            frames,
        };
        clip.validate()?;
        Ok(clip)
    }

    pub fn validate(&self) -> Result<()> {
        if self.id.trim().is_empty() || self.name.trim().is_empty() {
            return Err(PotError::invalid_argument(
                "movie clip id and name must not be empty",
            ));
        }
        let Some(first) = self.frames.first() else {
            return Err(PotError::invalid_argument(
                "movie clip must contain at least one frame",
            ));
        };
        first.validate()?;
        for frame in &self.frames[1..] {
            frame.validate()?;
            if frame.dimensions() != first.dimensions() {
                return Err(PotError::invalid_argument(
                    "all movie clip frames must have identical dimensions",
                ));
            }
        }
        let final_offset = i32::try_from(self.frames.len() - 1)
            .map_err(|_| PotError::invalid_argument("movie clip has too many frames"))?;
        self.frame_start
            .checked_add(final_offset)
            .ok_or_else(|| PotError::invalid_argument("movie clip frame range overflows"))?;
        Ok(())
    }

    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    #[must_use]
    pub fn dimensions(&self) -> (usize, usize) {
        self.frames.first().map_or((0, 0), GrayFrame::dimensions)
    }

    #[must_use]
    pub fn frame(&self, frame_number: i32) -> Option<&GrayFrame> {
        let offset = i64::from(frame_number) - i64::from(self.frame_start);
        usize::try_from(offset)
            .ok()
            .and_then(|index| self.frames.get(index))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Marker {
    pub frame: i32,
    pub position: [f64; 2],
    pub confidence: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Track {
    pub id: String,
    pub name: String,
    pub markers: Vec<Marker>,
}

impl Track {
    pub fn new(id: String, name: String, markers: Vec<Marker>) -> Result<Self> {
        let track = Self { id, name, markers };
        track.validate()?;
        Ok(track)
    }

    pub fn validate(&self) -> Result<()> {
        if self.id.trim().is_empty() || self.name.trim().is_empty() {
            return Err(PotError::invalid_argument(
                "track id and name must not be empty",
            ));
        }
        if self.markers.is_empty() {
            return Err(PotError::invalid_argument(
                "track must contain at least one marker",
            ));
        }
        let mut previous_frame = None;
        for marker in &self.markers {
            if !marker
                .position
                .iter()
                .all(|coordinate| coordinate.is_finite())
                || !marker.confidence.is_finite()
                || !(0.0..=1.0).contains(&marker.confidence)
            {
                return Err(PotError::invalid_argument(
                    "track marker coordinates and confidence must be finite and valid",
                ));
            }
            if previous_frame.is_some_and(|frame| marker.frame <= frame) {
                return Err(PotError::invalid_argument(
                    "track markers must be ordered by strictly increasing frame",
                ));
            }
            previous_frame = Some(marker.frame);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct KltConfig {
    pub patch_radius: usize,
    pub max_iterations: usize,
    pub max_pyramid_levels: usize,
    pub convergence_threshold: f64,
    pub min_eigenvalue: f64,
}

impl Default for KltConfig {
    fn default() -> Self {
        Self {
            patch_radius: 5,
            max_iterations: 30,
            max_pyramid_levels: 4,
            convergence_threshold: 0.001,
            min_eigenvalue: 1e-7,
        }
    }
}

impl KltConfig {
    fn validate(&self) -> Result<()> {
        if !(2..=64).contains(&self.patch_radius)
            || self.max_iterations == 0
            || self.max_iterations > 100
            || self.max_pyramid_levels == 0
            || self.max_pyramid_levels > 16
            || !self.convergence_threshold.is_finite()
            || self.convergence_threshold <= 0.0
            || !self.min_eigenvalue.is_finite()
            || self.min_eigenvalue <= 0.0
        {
            return Err(PotError::invalid_argument(
                "Lucas-Kanade configuration is outside supported bounds",
            ));
        }
        Ok(())
    }
}

/// Tracks one point from `previous` into `current` using coarse-to-fine KLT.
pub fn track_between(
    previous: &GrayFrame,
    current: &GrayFrame,
    position: [f64; 2],
    config: &KltConfig,
) -> Result<[f64; 2]> {
    track_pair(previous, current, position, config).map(|(tracked, _)| tracked)
}

/// Tracks a marker forward and backward from its seed frame through the clip.
pub fn track_marker(
    clip: &MovieClip,
    initial_frame: i32,
    position: [f64; 2],
    config: &KltConfig,
) -> Result<Vec<Marker>> {
    clip.validate()?;
    config.validate()?;
    if !position.iter().all(|coordinate| coordinate.is_finite()) {
        return Err(PotError::invalid_argument(
            "marker position must contain finite coordinates",
        ));
    }
    let initial_offset = i64::from(initial_frame) - i64::from(clip.frame_start);
    let initial_index = usize::try_from(initial_offset)
        .ok()
        .filter(|index| *index < clip.frames.len())
        .ok_or_else(|| PotError::invalid_argument("initial marker frame is outside the clip"))?;
    validate_patch_position(&clip.frames[initial_index], position, config.patch_radius)?;

    let mut markers = vec![None; clip.frames.len()];
    markers[initial_index] = Some(Marker {
        frame: initial_frame,
        position,
        confidence: 1.0,
    });
    let mut point = position;
    for (index, marker_slot) in markers.iter_mut().enumerate().skip(initial_index + 1) {
        let (tracked, confidence) =
            track_pair(&clip.frames[index - 1], &clip.frames[index], point, config)?;
        point = tracked;
        *marker_slot = Some(Marker {
            frame: clip.frame_start
                + i32::try_from(index).map_err(|_| {
                    PotError::invalid_argument("movie clip frame index is out of range")
                })?,
            position: point,
            confidence,
        });
    }
    point = position;
    for index in (0..initial_index).rev() {
        let (tracked, confidence) =
            track_pair(&clip.frames[index + 1], &clip.frames[index], point, config)?;
        point = tracked;
        markers[index] = Some(Marker {
            frame: clip.frame_start
                + i32::try_from(index).map_err(|_| {
                    PotError::invalid_argument("movie clip frame index is out of range")
                })?,
            position: point,
            confidence,
        });
    }
    markers
        .into_iter()
        .map(|marker| {
            marker.ok_or_else(|| PotError::invalid_operation("tracking left a clip frame unmarked"))
        })
        .collect()
}

#[expect(
    clippy::cast_precision_loss,
    reason = "patch and pyramid indices are bounded by validated image dimensions"
)]
fn track_pair(
    previous: &GrayFrame,
    current: &GrayFrame,
    position: [f64; 2],
    config: &KltConfig,
) -> Result<([f64; 2], f64)> {
    previous.validate()?;
    current.validate()?;
    config.validate()?;
    if previous.dimensions() != current.dimensions() {
        return Err(PotError::invalid_argument(
            "Lucas-Kanade frames must have matching dimensions",
        ));
    }
    if !position.iter().all(|coordinate| coordinate.is_finite()) {
        return Err(PotError::invalid_argument(
            "Lucas-Kanade seed position must be finite",
        ));
    }
    validate_patch_position(previous, position, config.patch_radius)?;
    let previous_pyramid = make_pyramid(previous, config)?;
    let current_pyramid = make_pyramid(current, config)?;
    let mut levels = previous_pyramid.len().min(current_pyramid.len());
    while levels > 1 {
        let level = levels - 1;
        let scale = (1_usize << level) as f64;
        let offset = (scale - 1.0) * 0.5;
        let scaled_position = [
            (position[0] - offset) / scale,
            (position[1] - offset) / scale,
        ];
        let scaled_radius = config.patch_radius as f64 + 1.0;
        let frame = previous_pyramid[level].as_ref();
        if scaled_position[0] >= scaled_radius
            && scaled_position[1] >= scaled_radius
            && scaled_position[0] <= frame.width as f64 - scaled_radius - 1.0
            && scaled_position[1] <= frame.height as f64 - scaled_radius - 1.0
        {
            break;
        }
        levels -= 1;
    }
    let mut target_base = position;
    let mut final_error = 0.0;
    let patch_capacity = (config.patch_radius * 2 + 1).pow(2);
    let mut template = Vec::with_capacity(patch_capacity);
    let mut gradients = Vec::with_capacity(patch_capacity);
    let radius = isize::try_from(config.patch_radius)
        .map_err(|_| PotError::invalid_argument("patch radius is too large"))?;

    for level in (0..levels).rev() {
        let scale = 1_usize << level;
        let scale_f64 = scale as f64;
        let offset = (scale_f64 - 1.0) * 0.5;
        let source_center = [
            (position[0] - offset) / scale_f64,
            (position[1] - offset) / scale_f64,
        ];
        let mut target_center = [
            (target_base[0] - offset) / scale_f64,
            (target_base[1] - offset) / scale_f64,
        ];
        let source_frame = &previous_pyramid[level];
        let target_frame = &current_pyramid[level];
        template.clear();
        gradients.clear();
        let mut h11 = 0.0;
        let mut h12 = 0.0;
        let mut h22 = 0.0;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let x = source_center[0] + dx as f64;
                let y = source_center[1] + dy as f64;
                let value = source_frame.sample(x, y).ok_or_else(|| {
                    PotError::invalid_argument("tracking patch is outside the source image")
                })?;
                let gradient_x = (source_frame.sample(x + 1.0, y).ok_or_else(|| {
                    PotError::invalid_argument("tracking gradient is outside the source image")
                })? - source_frame.sample(x - 1.0, y).ok_or_else(|| {
                    PotError::invalid_argument("tracking gradient is outside the source image")
                })?) * 0.5;
                let gradient_y = (source_frame.sample(x, y + 1.0).ok_or_else(|| {
                    PotError::invalid_argument("tracking gradient is outside the source image")
                })? - source_frame.sample(x, y - 1.0).ok_or_else(|| {
                    PotError::invalid_argument("tracking gradient is outside the source image")
                })?) * 0.5;
                template.push(value);
                gradients.push([gradient_x, gradient_y]);
                h11 += gradient_x * gradient_x;
                h12 += gradient_x * gradient_y;
                h22 += gradient_y * gradient_y;
            }
        }
        let determinant = h11 * h22 - h12 * h12;
        let minimum_eigenvalue = (h11 + h22 - ((h11 - h22).powi(2) + 4.0 * h12.powi(2)).sqrt())
            * 0.5
            / template.len() as f64;
        if !determinant.is_finite()
            || determinant <= f64::EPSILON
            || minimum_eigenvalue < config.min_eigenvalue
        {
            return Err(PotError::invalid_operation(
                "tracking patch does not contain enough two-dimensional texture",
            ));
        }
        let mut converged = false;
        for _ in 0..config.max_iterations {
            let mut b1 = 0.0;
            let mut b2 = 0.0;
            let mut sample_index = 0;
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let target = target_frame
                        .sample(target_center[0] + dx as f64, target_center[1] + dy as f64)
                        .ok_or_else(|| {
                            PotError::invalid_operation(
                                "tracked patch moved outside the target image",
                            )
                        })?;
                    let residual = target - template[sample_index];
                    b1 += gradients[sample_index][0] * residual;
                    b2 += gradients[sample_index][1] * residual;
                    sample_index += 1;
                }
            }
            let delta = [
                (h22 * b1 - h12 * b2) / determinant,
                (h11 * b2 - h12 * b1) / determinant,
            ];
            target_center[0] -= delta[0];
            target_center[1] -= delta[1];
            if delta[0].hypot(delta[1]) <= config.convergence_threshold {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err(PotError::invalid_operation(
                "Lucas-Kanade tracking did not converge within the iteration limit",
            ));
        }
        target_base = [
            target_center[0] * scale_f64 + offset,
            target_center[1] * scale_f64 + offset,
        ];
        if level == 0 {
            let mut squared_error = 0.0;
            let mut sample_index = 0;
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let target = target_frame
                        .sample(target_center[0] + dx as f64, target_center[1] + dy as f64)
                        .ok_or_else(|| {
                            PotError::invalid_operation(
                                "tracked patch moved outside the target image",
                            )
                        })?;
                    let difference = target - template[sample_index];
                    squared_error += difference * difference;
                    sample_index += 1;
                }
            }
            final_error = (squared_error / template.len() as f64).sqrt();
        }
    }
    if !target_base.iter().all(|coordinate| coordinate.is_finite()) {
        return Err(PotError::invalid_operation(
            "Lucas-Kanade tracking produced a non-finite location",
        ));
    }
    let confidence = (1.0 / (1.0 + 10.0 * final_error)).clamp(0.0, 1.0);
    Ok((target_base, confidence))
}

#[expect(
    clippy::cast_precision_loss,
    reason = "pixel dimensions are bounded by the allocated frame storage"
)]
fn validate_patch_position(frame: &GrayFrame, position: [f64; 2], radius: usize) -> Result<()> {
    let margin = radius as f64 + 1.0;
    if position[0] < margin
        || position[1] < margin
        || position[0] > frame.width as f64 - margin - 1.0
        || position[1] > frame.height as f64 - margin - 1.0
    {
        return Err(PotError::invalid_argument(
            "tracking patch must fit inside the image with a gradient border",
        ));
    }
    Ok(())
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "downsampled grayscale intensity is stored at f32 precision"
)]
fn make_pyramid<'a>(frame: &'a GrayFrame, config: &KltConfig) -> Result<Vec<Cow<'a, GrayFrame>>> {
    let mut pyramid = vec![Cow::Borrowed(frame)];
    while pyramid.len() < config.max_pyramid_levels {
        let previous = pyramid
            .last()
            .ok_or_else(|| {
                PotError::invalid_operation("image pyramid unexpectedly contains no level")
            })?
            .as_ref();
        let minimum_size = config.patch_radius * 2 + 5;
        if previous.width < minimum_size * 2 || previous.height < minimum_size * 2 {
            break;
        }
        let width = previous.width.div_ceil(2);
        let height = previous.height.div_ceil(2);
        let mut pixels = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0_f64;
                let mut count = 0.0_f64;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let source_x = x * 2 + dx;
                        let source_y = y * 2 + dy;
                        if source_x < previous.width && source_y < previous.height {
                            sum += f64::from(previous.pixels[source_y * previous.width + source_x]);
                            count += 1.0;
                        }
                    }
                }
                pixels.push((sum / count) as f32);
            }
        }
        pyramid.push(Cow::Owned(GrayFrame::new(width, height, pixels)?));
    }
    Ok(pyramid)
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PointCorrespondence {
    pub source: [f64; 2],
    pub destination: [f64; 2],
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Homography {
    pub matrix: [[f64; 3]; 3],
}

impl Homography {
    pub fn transform(&self, point: [f64; 2]) -> Result<[f64; 2]> {
        if !point.iter().all(|coordinate| coordinate.is_finite())
            || !self.matrix.iter().flatten().all(|value| value.is_finite())
        {
            return Err(PotError::invalid_argument(
                "homography and point coordinates must be finite",
            ));
        }
        let denominator_terms = [
            self.matrix[2][0] * point[0],
            self.matrix[2][1] * point[1],
            self.matrix[2][2],
        ];
        let denominator = denominator_terms.iter().sum::<f64>();
        let denominator_scale = denominator_terms.iter().map(|term| term.abs()).sum::<f64>();
        if !denominator.is_finite() || denominator.abs() <= denominator_scale * f64::EPSILON * 64.0
        {
            return Err(PotError::invalid_operation(
                "homography maps the point to infinity",
            ));
        }
        let transformed = [
            (self.matrix[0][0] * point[0] + self.matrix[0][1] * point[1] + self.matrix[0][2])
                / denominator,
            (self.matrix[1][0] * point[0] + self.matrix[1][1] * point[1] + self.matrix[1][2])
                / denominator,
        ];
        if transformed.iter().all(|coordinate| coordinate.is_finite()) {
            Ok(transformed)
        } else {
            Err(PotError::invalid_operation(
                "homography produced a non-finite point",
            ))
        }
    }
}

/// Fits a projective plane transform to four or more 2D point pairs.
pub fn solve_homography(correspondences: &[PointCorrespondence]) -> Result<Homography> {
    if correspondences.len() < 4 {
        return Err(PotError::invalid_argument(
            "homography requires at least four point correspondences",
        ));
    }
    let source_normalization = normalize_correspondences(correspondences, true)?;
    let destination_normalization = normalize_correspondences(correspondences, false)?;
    let mut normal = [[0.0; 9]; 9];
    for pair in correspondences {
        if !pair.source.iter().all(|coordinate| coordinate.is_finite())
            || !pair
                .destination
                .iter()
                .all(|coordinate| coordinate.is_finite())
        {
            return Err(PotError::invalid_argument(
                "homography correspondences must contain finite coordinates",
            ));
        }
        let source = normalize_point(pair.source, source_normalization);
        let destination = normalize_point(pair.destination, destination_normalization);
        let [x, y] = source;
        let [u, v] = destination;
        accumulate_outer_product(&mut normal, &[x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u]);
        accumulate_outer_product(&mut normal, &[0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, -v]);
    }
    let (eigenvalues, eigenvectors) = symmetric_eigen(normal)?;
    validate_nullspace(&eigenvalues, "homography correspondences")?;
    let coefficients = smallest_eigenvector(&eigenvalues, &eigenvectors);
    let normalized = [
        [coefficients[0], coefficients[1], coefficients[2]],
        [coefficients[3], coefficients[4], coefficients[5]],
        [coefficients[6], coefficients[7], coefficients[8]],
    ];
    let source_transform = [
        [
            source_normalization.2,
            0.0,
            -source_normalization.2 * source_normalization.0,
        ],
        [
            0.0,
            source_normalization.2,
            -source_normalization.2 * source_normalization.1,
        ],
        [0.0, 0.0, 1.0],
    ];
    let inverse_destination_transform = [
        [
            1.0 / destination_normalization.2,
            0.0,
            destination_normalization.0,
        ],
        [
            0.0,
            1.0 / destination_normalization.2,
            destination_normalization.1,
        ],
        [0.0, 0.0, 1.0],
    ];
    let mut matrix = multiply_3x3(
        inverse_destination_transform,
        multiply_3x3(normalized, source_transform),
    );
    normalize_matrix_3x3(&mut matrix)?;
    Ok(Homography { matrix })
}

#[expect(
    clippy::cast_precision_loss,
    reason = "point count is bounded by the input correspondence allocation"
)]
fn normalize_correspondences(
    correspondences: &[PointCorrespondence],
    use_source: bool,
) -> Result<(f64, f64, f64)> {
    let mut center = [0.0; 2];
    for pair in correspondences {
        let point = if use_source {
            pair.source
        } else {
            pair.destination
        };
        if !point.iter().all(|coordinate| coordinate.is_finite()) {
            return Err(PotError::invalid_argument(
                "point correspondences must be finite",
            ));
        }
        center[0] += point[0];
        center[1] += point[1];
    }
    center[0] /= correspondences.len() as f64;
    center[1] /= correspondences.len() as f64;
    let mean_distance = correspondences
        .iter()
        .map(|pair| {
            let point = if use_source {
                pair.source
            } else {
                pair.destination
            };
            (point[0] - center[0]).hypot(point[1] - center[1])
        })
        .sum::<f64>()
        / correspondences.len() as f64;
    if mean_distance <= 1e-14 || !mean_distance.is_finite() {
        return Err(PotError::invalid_argument(
            "point correspondences do not span a usable region",
        ));
    }
    Ok((center[0], center[1], 2.0_f64.sqrt() / mean_distance))
}

fn normalize_point(point: [f64; 2], normalization: (f64, f64, f64)) -> [f64; 2] {
    [
        (point[0] - normalization.0) * normalization.2,
        (point[1] - normalization.1) * normalization.2,
    ]
}

fn normalize_matrix_3x3(matrix: &mut [[f64; 3]; 3]) -> Result<()> {
    let scale = matrix[2][2];
    if !scale.is_finite() || !matrix.iter().flatten().all(|value| value.is_finite()) {
        return Err(PotError::invalid_operation(
            "projective matrix contains non-finite coefficients",
        ));
    }
    let maximum = matrix
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if maximum == 0.0 {
        return Err(PotError::invalid_operation(
            "projective matrix solution is degenerate",
        ));
    }
    let normalization = if scale.abs() > maximum * 1e-12 {
        scale
    } else {
        maximum
    };
    for value in matrix.iter_mut().flatten() {
        *value /= normalization;
    }
    if matrix
        .iter()
        .flatten()
        .find(|value| value.abs() > 1e-14)
        .is_some_and(|value| *value < 0.0)
    {
        for value in matrix.iter_mut().flatten() {
            *value = -*value;
        }
    }
    if !matrix.iter().flatten().all(|value| value.is_finite()) {
        return Err(PotError::invalid_operation(
            "projective matrix normalization produced non-finite coefficients",
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CameraObservation {
    pub world: [f64; 3],
    pub image: [f64; 2],
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CameraModel {
    /// A 3x4 projective camera matrix; its scale is arbitrary.
    pub matrix: [[f64; 4]; 3],
}

impl CameraModel {
    pub fn project(&self, world: [f64; 3]) -> Result<[f64; 2]> {
        if !world.iter().all(|coordinate| coordinate.is_finite())
            || !self.matrix.iter().flatten().all(|value| value.is_finite())
        {
            return Err(PotError::invalid_argument(
                "camera and world coordinates must be finite",
            ));
        }
        let homogeneous = [world[0], world[1], world[2], 1.0];
        let mut projected = [0.0; 3];
        for (row, output) in self.matrix.iter().zip(&mut projected) {
            *output = row
                .iter()
                .zip(homogeneous)
                .map(|(coefficient, coordinate)| coefficient * coordinate)
                .sum();
        }
        let denominator_terms = [
            self.matrix[2][0] * world[0],
            self.matrix[2][1] * world[1],
            self.matrix[2][2] * world[2],
            self.matrix[2][3],
        ];
        let denominator_scale = denominator_terms.iter().map(|term| term.abs()).sum::<f64>();
        if !projected[2].is_finite()
            || projected[2].abs() <= denominator_scale * f64::EPSILON * 64.0
        {
            return Err(PotError::invalid_operation(
                "camera projects the point to infinity",
            ));
        }
        let image = [projected[0] / projected[2], projected[1] / projected[2]];
        if image.iter().all(|coordinate| coordinate.is_finite()) {
            Ok(image)
        } else {
            Err(PotError::invalid_operation(
                "camera produced non-finite image coordinates",
            ))
        }
    }
}
/// Decomposes a projective camera with its calibrated pixel-space intrinsics.
///
/// `CameraIntrinsics.principal` is normalized to the clip dimensions; the returned matrix is a
/// column-major Blender camera-to-world transform.
pub fn camera_world_matrix(
    camera: &CameraModel,
    intrinsics: &crate::model::CameraIntrinsics,
    width: u32,
    height: u32,
) -> Result<[f64; 16]> {
    if width == 0 || height == 0 {
        return Err(PotError::invalid_argument(
            "camera pose decomposition requires clip dimensions",
        ));
    }
    intrinsics.validate()?;
    let pixel_width = f64::from(width);
    let pixel_height = f64::from(height);
    let focal_x = intrinsics.focal_mm / intrinsics.sensor_width_mm * pixel_width;
    let focal_y = focal_x;
    let intrinsic = glam::DMat3::from_cols(
        DVec3::new(focal_x, 0.0, 0.0),
        DVec3::new(0.0, focal_y, 0.0),
        DVec3::new(
            intrinsics.principal[0] * pixel_width,
            intrinsics.principal[1] * pixel_height,
            1.0,
        ),
    );
    let determinant = intrinsic.determinant();
    if !determinant.is_finite() || determinant.abs() <= f64::EPSILON {
        return Err(PotError::invalid_argument(
            "camera intrinsics matrix is singular",
        ));
    }
    let projective_scale = camera
        .matrix
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if !projective_scale.is_finite()
        || projective_scale == 0.0
        || camera
            .matrix
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
    {
        return Err(PotError::invalid_argument(
            "projective camera matrix must be finite and nonzero",
        ));
    }
    let normalized = camera
        .matrix
        .map(|row| row.map(|value| value / projective_scale));
    let projection = glam::DMat3::from_cols(
        DVec3::new(normalized[0][0], normalized[1][0], normalized[2][0]),
        DVec3::new(normalized[0][1], normalized[1][1], normalized[2][1]),
        DVec3::new(normalized[0][2], normalized[1][2], normalized[2][2]),
    );
    let mut rotation_scaled = intrinsic.inverse() * projection;
    let mut offset = DVec3::new(normalized[0][3], normalized[1][3], normalized[2][3]);
    if rotation_scaled.determinant() < 0.0 {
        rotation_scaled = -rotation_scaled;
        offset = -offset;
    }
    let axis_x_raw = rotation_scaled.x_axis;
    let axis_y_raw = rotation_scaled.y_axis;
    let axis_z_raw = rotation_scaled.z_axis;
    let scale = (axis_x_raw.length() + axis_y_raw.length() + axis_z_raw.length()) / 3.0;
    if !scale.is_finite() || scale <= f64::EPSILON {
        return Err(PotError::invalid_argument(
            "projective camera has degenerate calibrated axes",
        ));
    }
    let axis_x = axis_x_raw.normalize();
    let axis_y = (axis_y_raw - axis_x * axis_y_raw.dot(axis_x)).normalize_or_zero();
    let axis_z = axis_x.cross(axis_y).normalize_or_zero();
    if axis_y.length_squared() <= f64::EPSILON
        || axis_z.length_squared() <= f64::EPSILON
        || axis_z.dot(axis_z_raw) < 0.0
    {
        return Err(PotError::invalid_argument(
            "projective camera does not produce a right-handed rotation",
        ));
    }
    let rotation = glam::DMat3::from_cols(axis_x, axis_y, axis_z);
    let camera_translation = intrinsic.inverse() * offset / scale;
    let camera_center = -(rotation.transpose() * camera_translation);
    let blender_axes =
        rotation.transpose() * glam::DMat3::from_diagonal(DVec3::new(1.0, -1.0, -1.0));
    let matrix =
        glam::DMat4::from_rotation_translation(DQuat::from_mat3(&blender_axes), camera_center);
    let matrix = matrix.to_cols_array();
    if matrix.iter().all(|value| value.is_finite()) {
        Ok(matrix)
    } else {
        Err(PotError::invalid_operation(
            "camera pose decomposition produced a non-finite transform",
        ))
    }
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObjectObservation {
    pub object: [f64; 3],
    pub image: [f64; 2],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObjectPose {
    /// Column-major object-to-reconstruction-space affine transform.
    pub matrix: [f64; 16],
    /// Uniform scale preserved from the initial transform because one-view scale is ambiguous.
    pub scale: f64,
    pub reprojection_error: f64,
}

#[derive(Clone, Copy)]
struct PoseState {
    rotation: DQuat,
    translation: DVec3,
    scale: f64,
}

/// Estimates a rigid object transform from reconstructed points and image markers.
///
/// `initial` is a column-major object-to-world pose used to seed each frame. Its uniform scale
/// remains fixed because a single camera view cannot determine absolute object scale. The
/// returned reprojection error is the RMS error in camera-image coordinate units.
#[expect(
    clippy::too_many_lines,
    reason = "the damped least-squares object pose solve keeps its state transition explicit"
)]
pub fn solve_object_pose(
    camera: &CameraModel,
    observations: &[ObjectObservation],
    initial: [f64; 16],
) -> Result<ObjectPose> {
    if observations.len() < 4 {
        return Err(PotError::invalid_argument(
            "object solve requires at least four tracked 3D/image correspondences",
        ));
    }
    if observations.iter().any(|observation| {
        observation
            .object
            .iter()
            .chain(observation.image.iter())
            .any(|value| !value.is_finite())
    }) {
        return Err(PotError::invalid_argument(
            "object solve observations must be finite",
        ));
    }
    if initial.iter().any(|value| !value.is_finite()) {
        return Err(PotError::invalid_argument(
            "object solve initial transform must be finite",
        ));
    }
    let initial = DMat4::from_cols_array(&initial);
    let (scale, rotation, translation) = initial.to_scale_rotation_translation();
    let scale = (scale.x.abs() + scale.y.abs() + scale.z.abs()) / 3.0;
    if !scale.is_finite()
        || scale <= f64::EPSILON
        || !rotation.is_finite()
        || !translation.is_finite()
    {
        return Err(PotError::invalid_argument(
            "object solve initial transform must have a finite positive scale",
        ));
    }
    let initial_pose = PoseState {
        rotation: rotation.normalize(),
        translation,
        scale,
    };
    let mut pose = initial_pose;
    let initial_cost = object_pose_cost(camera, observations, initial_pose);
    let mut cost = if let Some(translation) =
        estimate_object_translation(camera, observations, initial_pose)
    {
        let seeded_pose = PoseState {
            translation,
            ..initial_pose
        };
        match (
            initial_cost,
            object_pose_cost(camera, observations, seeded_pose),
        ) {
            (Ok(initial_cost), Ok(seeded_cost)) if seeded_cost < initial_cost => {
                pose = seeded_pose;
                seeded_cost
            }
            (Ok(initial_cost), _) => initial_cost,
            (Err(_), Ok(seeded_cost)) => {
                pose = seeded_pose;
                seeded_cost
            }
            (Err(initial_error), Err(_)) => return Err(initial_error),
        }
    } else {
        initial_cost?
    };
    let mut damping = 1.0e-3;
    for _ in 0..100 {
        let mut normal = [[0.0; 6]; 6];
        let mut gradient = [0.0; 6];
        for observation in observations {
            let point = DVec3::from_array(observation.object);
            let projected = camera.project(object_point(pose, point).to_array())?;
            let residual = [
                projected[0] - observation.image[0],
                projected[1] - observation.image[1],
            ];
            let mut jacobian = [[0.0; 2]; 6];
            for (parameter, derivatives) in jacobian.iter_mut().enumerate() {
                let step = if parameter < 3 { 1.0e-6 } else { 1.0e-5 };
                let positive = camera
                    .project(object_point(perturb_pose(pose, parameter, step), point).to_array())?;
                let negative = camera.project(
                    object_point(perturb_pose(pose, parameter, -step), point).to_array(),
                )?;
                derivatives[0] = (positive[0] - negative[0]) / (2.0 * step);
                derivatives[1] = (positive[1] - negative[1]) / (2.0 * step);
            }
            for (parameter, derivatives) in jacobian.iter().enumerate() {
                for (axis, derivative) in derivatives.iter().enumerate() {
                    gradient[parameter] -= derivative * residual[axis];
                    for (other, other_derivatives) in jacobian.iter().enumerate() {
                        normal[parameter][other] += derivative * other_derivatives[axis];
                    }
                }
            }
        }
        for (diagonal, row) in normal.iter_mut().enumerate() {
            row[diagonal] += damping * row[diagonal].max(1.0);
        }
        let Ok(step) = solve_linear(normal, gradient) else {
            return Err(PotError::invalid_argument(
                "object solve correspondences do not determine a rigid pose",
            ));
        };
        let candidate = update_pose(pose, step);
        let candidate_cost = object_pose_cost(camera, observations, candidate)?;
        if candidate_cost < cost {
            pose = candidate;
            cost = candidate_cost;
            damping = (damping / 3.0).max(1.0e-12);
            if step.iter().map(|value| value * value).sum::<f64>() < 1.0e-18 {
                break;
            }
        } else {
            damping *= 10.0;
            if damping > 1.0e12 {
                break;
            }
        }
    }
    let matrix = DMat4::from_scale_rotation_translation(
        DVec3::splat(pose.scale),
        pose.rotation,
        pose.translation,
    );
    let observation_count = u32::try_from(observations.len()).map_err(|_| {
        PotError::invalid_argument("object solve contains too many correspondences")
    })?;
    let reprojection_error = (cost / f64::from(observation_count)).sqrt();
    if !matrix.to_cols_array().iter().all(|value| value.is_finite())
        || !reprojection_error.is_finite()
    {
        return Err(PotError::invalid_operation(
            "object solve produced a non-finite pose",
        ));
    }
    Ok(ObjectPose {
        matrix: matrix.to_cols_array(),
        scale: pose.scale,
        reprojection_error,
    })
}
fn object_point(pose: PoseState, point: DVec3) -> DVec3 {
    pose.rotation * (point * pose.scale) + pose.translation
}

// Solve translation linearly at the initial rotation/scale before nonlinear refinement; the
// identity seed can otherwise put reconstructed landmarks on the camera plane.
fn estimate_object_translation(
    camera: &CameraModel,
    observations: &[ObjectObservation],
    pose: PoseState,
) -> Option<DVec3> {
    let mut normal = [[0.0; 3]; 3];
    let mut rhs = [0.0; 3];
    for observation in observations {
        let object = DVec3::from_array(observation.object);
        let rotated = pose.rotation * (object * pose.scale);
        for (row, image_coordinate) in [(0, observation.image[0]), (1, observation.image[1])] {
            let mut coefficients = [0.0; 3];
            for (axis, coefficient) in coefficients.iter_mut().enumerate() {
                *coefficient = camera.matrix[row][axis] - image_coordinate * camera.matrix[2][axis];
            }
            let constant = camera.matrix[row][3] - image_coordinate * camera.matrix[2][3];
            let target = -constant
                - coefficients
                    .iter()
                    .zip(rotated.to_array())
                    .map(|(coefficient, coordinate)| coefficient * coordinate)
                    .sum::<f64>();
            for left in 0..3 {
                rhs[left] += coefficients[left] * target;
                for right in 0..3 {
                    normal[left][right] += coefficients[left] * coefficients[right];
                }
            }
        }
    }
    solve_linear(normal, rhs).ok().map(DVec3::from_array)
}
fn perturb_pose(pose: PoseState, parameter: usize, amount: f64) -> PoseState {
    match parameter {
        0..=2 => {
            let axis = DVec3::AXES[parameter];
            PoseState {
                rotation: DQuat::from_axis_angle(axis, amount) * pose.rotation,
                ..pose
            }
        }
        3..=5 => {
            let mut translation = pose.translation;
            translation[parameter - 3] += amount;
            PoseState {
                translation,
                ..pose
            }
        }
        _ => pose,
    }
}

fn update_pose(pose: PoseState, step: [f64; 6]) -> PoseState {
    let rotation_step = DVec3::new(step[0], step[1], step[2]);
    let rotation = if rotation_step.length_squared() <= f64::EPSILON {
        DQuat::IDENTITY
    } else {
        DQuat::from_scaled_axis(rotation_step)
    };
    PoseState {
        rotation: (rotation * pose.rotation).normalize(),
        translation: pose.translation + DVec3::new(step[3], step[4], step[5]),
        scale: pose.scale,
    }
}

fn object_pose_cost(
    camera: &CameraModel,
    observations: &[ObjectObservation],
    pose: PoseState,
) -> Result<f64> {
    observations.iter().try_fold(0.0, |cost, observation| {
        let projected =
            camera.project(object_point(pose, DVec3::from_array(observation.object)).to_array())?;
        let dx = projected[0] - observation.image[0];
        let dy = projected[1] - observation.image[1];
        Ok(cost + dx * dx + dy * dy)
    })
}

/// Solves an uncalibrated projective camera from six or more 3D/2D pairs.
///
/// The input observations must span three dimensions; coplanar points do not
/// determine a unique 3x4 camera matrix.
pub fn solve_camera(observations: &[CameraObservation]) -> Result<CameraModel> {
    if observations.len() < 6 {
        return Err(PotError::invalid_argument(
            "camera solve requires at least six 3D/2D observations",
        ));
    }
    let (world_center, world_scale) = normalize_world_observations(observations)?;
    let (image_center, image_scale) = normalize_image_observations(observations)?;
    let mut normal = [[0.0; 12]; 12];
    for observation in observations {
        let world = [
            (observation.world[0] - world_center[0]) * world_scale,
            (observation.world[1] - world_center[1]) * world_scale,
            (observation.world[2] - world_center[2]) * world_scale,
        ];
        let image = [
            (observation.image[0] - image_center[0]) * image_scale,
            (observation.image[1] - image_center[1]) * image_scale,
        ];
        let [x, y, z] = world;
        let [u, v] = image;
        accumulate_outer_product(
            &mut normal,
            &[x, y, z, 1.0, 0.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u * z, -u],
        );
        accumulate_outer_product(
            &mut normal,
            &[0.0, 0.0, 0.0, 0.0, x, y, z, 1.0, -v * x, -v * y, -v * z, -v],
        );
    }
    let (eigenvalues, eigenvectors) = symmetric_eigen(normal)?;
    validate_nullspace(&eigenvalues, "camera observations")?;
    let coefficients = smallest_eigenvector(&eigenvalues, &eigenvectors);
    let normalized = [
        [
            coefficients[0],
            coefficients[1],
            coefficients[2],
            coefficients[3],
        ],
        [
            coefficients[4],
            coefficients[5],
            coefficients[6],
            coefficients[7],
        ],
        [
            coefficients[8],
            coefficients[9],
            coefficients[10],
            coefficients[11],
        ],
    ];
    let mut world_transform = [[0.0; 4]; 4];
    world_transform[0][0] = world_scale;
    world_transform[1][1] = world_scale;
    world_transform[2][2] = world_scale;
    world_transform[0][3] = -world_scale * world_center[0];
    world_transform[1][3] = -world_scale * world_center[1];
    world_transform[2][3] = -world_scale * world_center[2];
    world_transform[3][3] = 1.0;
    let mut image_inverse = [[0.0; 3]; 3];
    image_inverse[0][0] = 1.0 / image_scale;
    image_inverse[1][1] = 1.0 / image_scale;
    image_inverse[0][2] = image_center[0];
    image_inverse[1][2] = image_center[1];
    image_inverse[2][2] = 1.0;
    let world_mapped = multiply_3x4_4x4(normalized, world_transform);
    let mut matrix = multiply_3x3_3x4(image_inverse, world_mapped);
    normalize_camera_matrix(&mut matrix)?;
    Ok(CameraModel { matrix })
}

#[expect(
    clippy::cast_precision_loss,
    reason = "point count is bounded by the input observation allocation"
)]
fn normalize_world_observations(observations: &[CameraObservation]) -> Result<([f64; 3], f64)> {
    let mut center = [0.0; 3];
    for observation in observations {
        if !observation
            .world
            .iter()
            .all(|coordinate| coordinate.is_finite())
            || !observation
                .image
                .iter()
                .all(|coordinate| coordinate.is_finite())
        {
            return Err(PotError::invalid_argument(
                "camera observations must contain finite coordinates",
            ));
        }
        for (target, coordinate) in center.iter_mut().zip(observation.world) {
            *target += coordinate;
        }
    }
    for coordinate in &mut center {
        *coordinate /= observations.len() as f64;
    }
    let mean_distance = observations
        .iter()
        .map(|observation| {
            let x = observation.world[0] - center[0];
            let y = observation.world[1] - center[1];
            let z = observation.world[2] - center[2];
            (x * x + y * y + z * z).sqrt()
        })
        .sum::<f64>()
        / observations.len() as f64;
    if !mean_distance.is_finite() || mean_distance <= 1e-14 {
        return Err(PotError::invalid_argument(
            "camera world points do not span a usable volume",
        ));
    }
    Ok((center, 3.0_f64.sqrt() / mean_distance))
}

#[expect(
    clippy::cast_precision_loss,
    reason = "point count is bounded by the input observation allocation"
)]
fn normalize_image_observations(observations: &[CameraObservation]) -> Result<([f64; 2], f64)> {
    let mut center = [0.0; 2];
    for observation in observations {
        for (target, coordinate) in center.iter_mut().zip(observation.image) {
            *target += coordinate;
        }
    }
    for coordinate in &mut center {
        *coordinate /= observations.len() as f64;
    }
    let mean_distance = observations
        .iter()
        .map(|observation| {
            (observation.image[0] - center[0]).hypot(observation.image[1] - center[1])
        })
        .sum::<f64>()
        / observations.len() as f64;
    if !mean_distance.is_finite() || mean_distance <= 1e-14 {
        return Err(PotError::invalid_argument(
            "camera image points do not span a usable region",
        ));
    }
    Ok((center, 2.0_f64.sqrt() / mean_distance))
}

fn normalize_camera_matrix(matrix: &mut [[f64; 4]; 3]) -> Result<()> {
    if !matrix.iter().flatten().all(|value| value.is_finite()) {
        return Err(PotError::invalid_operation(
            "camera solve produced non-finite coefficients",
        ));
    }
    let maximum = matrix
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if maximum == 0.0 {
        return Err(PotError::invalid_operation(
            "camera solve produced a degenerate matrix",
        ));
    }
    for value in matrix.iter_mut().flatten() {
        *value /= maximum;
    }
    let norm = matrix
        .iter()
        .flatten()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if norm == 0.0 || !norm.is_finite() {
        return Err(PotError::invalid_operation(
            "camera solve produced a degenerate matrix",
        ));
    }
    for value in matrix.iter_mut().flatten() {
        *value /= norm;
    }
    if matrix
        .iter()
        .flatten()
        .find(|value| value.abs() > 1e-14)
        .is_some_and(|value| *value < 0.0)
    {
        for value in matrix.iter_mut().flatten() {
            *value = -*value;
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FundamentalMatrix {
    pub matrix: [[f64; 3]; 3],
}

/// Estimates a rank-two fundamental matrix from at least eight 2D matches.
///
/// Coordinates may be pixel coordinates; the normalized eight-point algorithm
/// internally centers and scales each image's observations.
pub fn solve_fundamental_matrix(
    correspondences: &[PointCorrespondence],
) -> Result<FundamentalMatrix> {
    if correspondences.len() < 8 {
        return Err(PotError::invalid_argument(
            "eight-point fundamental solve requires at least eight correspondences",
        ));
    }
    let source_normalization = normalize_correspondences(correspondences, true)?;
    let destination_normalization = normalize_correspondences(correspondences, false)?;
    let mut normal = [[0.0; 9]; 9];
    for pair in correspondences {
        let [x, y] = normalize_point(pair.source, source_normalization);
        let [u, v] = normalize_point(pair.destination, destination_normalization);
        accumulate_outer_product(&mut normal, &[u * x, u * y, u, v * x, v * y, v, x, y, 1.0]);
    }
    let (eigenvalues, eigenvectors) = symmetric_eigen(normal)?;
    validate_nullspace(&eigenvalues, "eight-point correspondences")?;
    let coefficients = smallest_eigenvector(&eigenvalues, &eigenvectors);
    let normalized = [
        [coefficients[0], coefficients[1], coefficients[2]],
        [coefficients[3], coefficients[4], coefficients[5]],
        [coefficients[6], coefficients[7], coefficients[8]],
    ];
    let rank_two = enforce_rank_two(normalized, false)?;
    let source_transform = [
        [
            source_normalization.2,
            0.0,
            -source_normalization.2 * source_normalization.0,
        ],
        [
            0.0,
            source_normalization.2,
            -source_normalization.2 * source_normalization.1,
        ],
        [0.0, 0.0, 1.0],
    ];
    let destination_transform_transposed = [
        [destination_normalization.2, 0.0, 0.0],
        [0.0, destination_normalization.2, 0.0],
        [
            -destination_normalization.2 * destination_normalization.0,
            -destination_normalization.2 * destination_normalization.1,
            1.0,
        ],
    ];
    let mut matrix = multiply_3x3(
        destination_transform_transposed,
        multiply_3x3(rank_two, source_transform),
    );
    normalize_matrix_3x3(&mut matrix)?;
    Ok(FundamentalMatrix { matrix })
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EssentialMatrix {
    pub matrix: [[f64; 3]; 3],
}

/// Estimates an essential matrix from normalized camera-coordinate matches.
///
/// The input points must already have been undistorted and normalized by each
/// camera's intrinsics; essential-matrix singular values are then enforced.
pub fn solve_essential_matrix(correspondences: &[PointCorrespondence]) -> Result<EssentialMatrix> {
    let fundamental = solve_fundamental_matrix(correspondences)?;
    Ok(EssentialMatrix {
        matrix: enforce_rank_two(fundamental.matrix, true)?,
    })
}

fn enforce_rank_two(matrix: [[f64; 3]; 3], equal_singular_values: bool) -> Result<[[f64; 3]; 3]> {
    let mut normal = [[0.0; 3]; 3];
    for source_row in &matrix {
        for (normal_row, left) in normal.iter_mut().zip(source_row) {
            for (normal_value, right) in normal_row.iter_mut().zip(source_row) {
                *normal_value += *left * *right;
            }
        }
    }
    let (eigenvalues, eigenvectors) = symmetric_eigen(normal)?;
    let order = eigenvalue_order(&eigenvalues);
    let singular = [
        eigenvalues[order[2]].max(0.0).sqrt(),
        eigenvalues[order[1]].max(0.0).sqrt(),
    ];
    if !singular[0].is_finite()
        || !singular[1].is_finite()
        || singular[0] <= 1e-14
        || singular[1] <= singular[0] * 1e-12
    {
        return Err(PotError::invalid_argument("epipolar matrix is degenerate"));
    }
    let mut right = [[0.0; 3]; 3];
    for column in 0..3 {
        for row in 0..3 {
            right[row][column] = eigenvectors[row][order[2 - column]];
        }
    }
    let mut left = [[0.0; 3]; 3];
    let mut first = multiply_3x3_vector(matrix, matrix_column(right, 0));
    normalize_vector_3(&mut first)?;
    let mut second = multiply_3x3_vector(matrix, matrix_column(right, 1));
    let projection = dot_vector_3(first, second);
    for index in 0..3 {
        second[index] -= projection * first[index];
    }
    normalize_vector_3(&mut second)?;
    let third = cross_vector_3(first, second);
    set_matrix_column(&mut left, 0, first);
    set_matrix_column(&mut left, 1, second);
    set_matrix_column(&mut left, 2, third);
    let first_singular = if equal_singular_values {
        f64::midpoint(singular[0], singular[1])
    } else {
        singular[0]
    };
    let second_singular = if equal_singular_values {
        first_singular
    } else {
        singular[1]
    };
    let mut projected = [[0.0; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            projected[row][column] = left[row][0] * first_singular * right[column][0]
                + left[row][1] * second_singular * right[column][1];
        }
    }
    normalize_matrix_3x3(&mut projected)?;
    Ok(projected)
}

fn matrix_column(matrix: [[f64; 3]; 3], column: usize) -> [f64; 3] {
    [matrix[0][column], matrix[1][column], matrix[2][column]]
}

fn set_matrix_column(matrix: &mut [[f64; 3]; 3], column: usize, value: [f64; 3]) {
    for row in 0..3 {
        matrix[row][column] = value[row];
    }
}

fn multiply_3x3_vector(matrix: [[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    let mut result = [0.0; 3];
    for row in 0..3 {
        for column in 0..3 {
            result[row] += matrix[row][column] * vector[column];
        }
    }
    result
}

fn dot_vector_3(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross_vector_3(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn normalize_vector_3(vector: &mut [f64; 3]) -> Result<()> {
    let norm = dot_vector_3(*vector, *vector).sqrt();
    if !norm.is_finite() || norm <= 1e-14 {
        return Err(PotError::invalid_operation(
            "epipolar matrix decomposition is degenerate",
        ));
    }
    for value in vector {
        *value /= norm;
    }
    Ok(())
}

/// Triangulates a 3D point from two projective cameras and image observations.
pub fn triangulate_point(
    first_camera: &CameraModel,
    second_camera: &CameraModel,
    first_image: [f64; 2],
    second_image: [f64; 2],
) -> Result<[f64; 3]> {
    if !first_image.iter().all(|coordinate| coordinate.is_finite())
        || !second_image.iter().all(|coordinate| coordinate.is_finite())
    {
        return Err(PotError::invalid_argument(
            "triangulation image coordinates must be finite",
        ));
    }
    let mut normal = [[0.0; 4]; 4];
    for (camera, image) in [(first_camera, first_image), (second_camera, second_image)] {
        if !camera
            .matrix
            .iter()
            .flatten()
            .all(|value| value.is_finite())
        {
            return Err(PotError::invalid_argument(
                "triangulation camera coefficients must be finite",
            ));
        }
        for (image_coordinate, camera_row) in image.iter().zip(&camera.matrix[..2]) {
            let mut row = [0.0; 4];
            for ((value, camera_value), denominator_value) in
                row.iter_mut().zip(camera_row).zip(&camera.matrix[2])
            {
                *value = *image_coordinate * *denominator_value - *camera_value;
            }
            accumulate_outer_product(&mut normal, &row);
        }
    }
    let (eigenvalues, eigenvectors) = symmetric_eigen(normal)?;
    validate_nullspace(&eigenvalues, "triangulation observations")?;
    let homogeneous = smallest_eigenvector(&eigenvalues, &eigenvectors);
    let scale = homogeneous[3];
    if !scale.is_finite() || scale.abs() <= 1e-12 {
        return Err(PotError::invalid_operation(
            "triangulated point lies at infinity",
        ));
    }
    let point = [
        homogeneous[0] / scale,
        homogeneous[1] / scale,
        homogeneous[2] / scale,
    ];
    if point.iter().all(|coordinate| coordinate.is_finite()) {
        Ok(point)
    } else {
        Err(PotError::invalid_operation(
            "triangulation produced a non-finite point",
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BundleObservation {
    pub camera_index: usize,
    pub point_index: usize,
    pub image: [f64; 2],
}

#[derive(Debug, Clone, PartialEq)]
pub struct BundleAdjustmentResult {
    pub points: Vec<[f64; 3]>,
    pub initial_rms: f64,
    pub final_rms: f64,
    pub iterations: usize,
}

/// Refines 3D points against fixed projective cameras with Gauss-Newton.
///
/// This is structure-only bundle adjustment: camera matrices stay fixed to
/// avoid the global projective gauge ambiguity.
pub fn bundle_adjust_points(
    cameras: &[CameraModel],
    initial_points: &[[f64; 3]],
    observations: &[BundleObservation],
    max_iterations: usize,
) -> Result<BundleAdjustmentResult> {
    if cameras.len() < 2
        || initial_points.is_empty()
        || observations.is_empty()
        || !(1..=100).contains(&max_iterations)
    {
        return Err(PotError::invalid_argument(
            "bundle adjustment requires two cameras, points, observations, and 1..=100 iterations",
        ));
    }
    if cameras.iter().any(|camera| {
        !camera
            .matrix
            .iter()
            .flatten()
            .all(|value| value.is_finite())
    }) || initial_points
        .iter()
        .flatten()
        .any(|coordinate| !coordinate.is_finite())
    {
        return Err(PotError::invalid_argument(
            "bundle adjustment camera and point coordinates must be finite",
        ));
    }
    let mut point_observations: Vec<Vec<usize>> = vec![Vec::new(); initial_points.len()];
    for (index, observation) in observations.iter().enumerate() {
        if observation.camera_index >= cameras.len()
            || observation.point_index >= initial_points.len()
            || !observation
                .image
                .iter()
                .all(|coordinate| coordinate.is_finite())
        {
            return Err(PotError::invalid_argument(
                "bundle adjustment observation is invalid or out of range",
            ));
        }
        let point_rows = &mut point_observations[observation.point_index];
        if point_rows
            .iter()
            .any(|&previous| observations[previous].camera_index == observation.camera_index)
        {
            return Err(PotError::invalid_argument(
                "bundle adjustment points may have only one observation per camera",
            ));
        }
        point_rows.push(index);
    }
    if point_observations.iter().any(|rows| rows.len() < 2) {
        return Err(PotError::invalid_argument(
            "each bundle adjustment point requires observations in two cameras",
        ));
    }
    let initial_cost = bundle_cost(cameras, initial_points, observations)?;
    let mut points = initial_points.to_vec();
    let mut candidate = points.clone();
    let mut deltas = vec![[0.0; 3]; points.len()];
    let mut cost = initial_cost;
    let mut iterations = 0;
    for _ in 0..max_iterations {
        deltas.fill([0.0; 3]);
        for point_index in 0..points.len() {
            let mut normal = [[0.0; 3]; 3];
            let mut rhs = [0.0; 3];
            for &observation_index in &point_observations[point_index] {
                let observation = observations[observation_index];
                let (projection, jacobian) =
                    project_with_jacobian(&cameras[observation.camera_index], points[point_index])?;
                for axis in 0..2 {
                    let residual = projection[axis] - observation.image[axis];
                    for left in 0..3 {
                        rhs[left] -= jacobian[axis][left] * residual;
                        for right in 0..3 {
                            normal[left][right] += jacobian[axis][left] * jacobian[axis][right];
                        }
                    }
                }
            }
            deltas[point_index] = solve_linear(normal, rhs)?;
        }
        let largest_step = deltas
            .iter()
            .map(|delta| dot_vector_3(*delta, *delta).sqrt())
            .fold(0.0_f64, f64::max);
        if largest_step <= 1e-10 {
            break;
        }
        let mut accepted = false;
        let mut factor = 1.0;
        for _ in 0..8 {
            for point_index in 0..points.len() {
                for axis in 0..3 {
                    candidate[point_index][axis] =
                        points[point_index][axis] + factor * deltas[point_index][axis];
                }
            }
            let candidate_cost = bundle_cost(cameras, &candidate, observations)?;
            if candidate_cost < cost {
                points.copy_from_slice(&candidate);
                cost = candidate_cost;
                iterations += 1;
                accepted = true;
                break;
            }
            factor *= 0.5;
        }
        if !accepted {
            break;
        }
    }
    let observation_count = u32::try_from(observations.len()).map_err(|_| {
        PotError::invalid_argument("bundle adjustment supports at most u32::MAX observations")
    })?;
    let divisor = f64::from(observation_count) * 2.0;
    Ok(BundleAdjustmentResult {
        points,
        initial_rms: (initial_cost / divisor).sqrt(),
        final_rms: (cost / divisor).sqrt(),
        iterations,
    })
}

fn bundle_cost(
    cameras: &[CameraModel],
    points: &[[f64; 3]],
    observations: &[BundleObservation],
) -> Result<f64> {
    let mut cost = 0.0;
    for observation in observations {
        let projected =
            cameras[observation.camera_index].project(points[observation.point_index])?;
        for (predicted, observed) in projected.iter().zip(observation.image) {
            cost += (*predicted - observed).powi(2);
        }
    }
    if cost.is_finite() {
        Ok(cost)
    } else {
        Err(PotError::invalid_operation(
            "bundle adjustment reprojection error overflowed",
        ))
    }
}

fn project_with_jacobian(
    camera: &CameraModel,
    point: [f64; 3],
) -> Result<([f64; 2], [[f64; 3]; 2])> {
    let homogeneous = [point[0], point[1], point[2], 1.0];
    let mut projected = [0.0; 3];
    for (row, output) in camera.matrix.iter().zip(&mut projected) {
        *output = row
            .iter()
            .zip(homogeneous)
            .map(|(coefficient, coordinate)| coefficient * coordinate)
            .sum();
    }
    let denominator = projected[2];
    let denominator_terms = [
        camera.matrix[2][0] * homogeneous[0],
        camera.matrix[2][1] * homogeneous[1],
        camera.matrix[2][2] * homogeneous[2],
        camera.matrix[2][3],
    ];
    let denominator_scale = denominator_terms.iter().map(|term| term.abs()).sum::<f64>();
    if !denominator.is_finite() || denominator.abs() <= denominator_scale * f64::EPSILON * 64.0 {
        return Err(PotError::invalid_operation(
            "bundle adjustment projection is singular",
        ));
    }
    let image = [projected[0] / denominator, projected[1] / denominator];
    let mut jacobian = [[0.0; 3]; 2];
    for ((image_value, row), jacobian_row) in
        image.iter().zip(&camera.matrix[..2]).zip(&mut jacobian)
    {
        for ((derivative, value), denominator_value) in jacobian_row
            .iter_mut()
            .zip(row.iter())
            .zip(&camera.matrix[2])
        {
            *derivative = (*value - *image_value * *denominator_value) / denominator;
        }
    }
    if image.iter().all(|value| value.is_finite())
        && jacobian.iter().flatten().all(|value| value.is_finite())
    {
        Ok((image, jacobian))
    } else {
        Err(PotError::invalid_operation(
            "bundle adjustment projection produced non-finite values",
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LensDistortion {
    pub k1: f64,
    pub k2: f64,
    pub k3: f64,
    pub p1: f64,
    pub p2: f64,
    pub center: [f64; 2],
    pub scale: [f64; 2],
}

impl LensDistortion {
    pub fn validate(&self) -> Result<()> {
        if ![
            self.k1,
            self.k2,
            self.k3,
            self.p1,
            self.p2,
            self.center[0],
            self.center[1],
            self.scale[0],
            self.scale[1],
        ]
        .iter()
        .all(|value| value.is_finite())
            || self.scale.iter().any(|value| *value <= 0.0)
        {
            return Err(PotError::invalid_argument(
                "lens coefficients and center must be finite and scales positive",
            ));
        }
        Ok(())
    }

    pub fn distort(&self, point: [f64; 2]) -> Result<[f64; 2]> {
        self.validate()?;
        if !point.iter().all(|coordinate| coordinate.is_finite()) {
            return Err(PotError::invalid_argument(
                "lens input point must contain finite coordinates",
            ));
        }
        let normalized = self.to_normalized(point);
        let distorted = self.distort_normalized(normalized)?;
        let pixel = self.to_pixel(distorted);
        if pixel.iter().all(|coordinate| coordinate.is_finite()) {
            Ok(pixel)
        } else {
            Err(PotError::invalid_operation(
                "lens distortion produced a non-finite point",
            ))
        }
    }

    pub fn undistort(&self, point: [f64; 2]) -> Result<[f64; 2]> {
        self.validate()?;
        if !point.iter().all(|coordinate| coordinate.is_finite()) {
            return Err(PotError::invalid_argument(
                "lens input point must contain finite coordinates",
            ));
        }
        let target = self.to_normalized(point);
        let mut estimate = target;
        for _ in 0..30 {
            let projected = self.distort_normalized(estimate)?;
            let residual = [projected[0] - target[0], projected[1] - target[1]];
            if residual[0].hypot(residual[1]) <= 1e-14 {
                return finite_lens_pixel(self.to_pixel(estimate));
            }
            let [x, y] = estimate;
            let radius_squared = x * x + y * y;
            let radial = 1.0
                + self.k1 * radius_squared
                + self.k2 * radius_squared.powi(2)
                + self.k3 * radius_squared.powi(3);
            let radial_slope =
                self.k1 + 2.0 * self.k2 * radius_squared + 3.0 * self.k3 * radius_squared.powi(2);
            let dr_dx = 2.0 * x * radial_slope;
            let dr_dy = 2.0 * y * radial_slope;
            let j11 = radial + x * dr_dx + 2.0 * self.p1 * y + 6.0 * self.p2 * x;
            let j12 = x * dr_dy + 2.0 * self.p1 * x + 2.0 * self.p2 * y;
            let j21 = y * dr_dx + 2.0 * self.p1 * x + 2.0 * self.p2 * y;
            let j22 = radial + y * dr_dy + 6.0 * self.p1 * y + 2.0 * self.p2 * x;
            let determinant = j11 * j22 - j12 * j21;
            if !determinant.is_finite() || determinant.abs() <= 1e-14 {
                return Err(PotError::invalid_operation(
                    "lens undistortion encountered a singular Jacobian",
                ));
            }
            let step = [
                (j22 * residual[0] - j12 * residual[1]) / determinant,
                (j11 * residual[1] - j21 * residual[0]) / determinant,
            ];
            estimate[0] -= step[0];
            estimate[1] -= step[1];
            if !estimate.iter().all(|coordinate| coordinate.is_finite()) {
                return Err(PotError::invalid_operation(
                    "lens undistortion diverged to a non-finite point",
                ));
            }
        }
        let residual = self.distort_normalized(estimate)?;
        if (residual[0] - target[0]).hypot(residual[1] - target[1]) > 1e-10 {
            return Err(PotError::invalid_operation(
                "lens undistortion did not converge",
            ));
        }
        finite_lens_pixel(self.to_pixel(estimate))
    }

    fn to_normalized(self, point: [f64; 2]) -> [f64; 2] {
        [
            (point[0] - self.center[0]) / self.scale[0],
            (point[1] - self.center[1]) / self.scale[1],
        ]
    }

    fn to_pixel(self, point: [f64; 2]) -> [f64; 2] {
        [
            point[0] * self.scale[0] + self.center[0],
            point[1] * self.scale[1] + self.center[1],
        ]
    }

    fn distort_normalized(&self, point: [f64; 2]) -> Result<[f64; 2]> {
        let [x, y] = point;
        let radius_squared = x * x + y * y;
        let radial = 1.0
            + self.k1 * radius_squared
            + self.k2 * radius_squared.powi(2)
            + self.k3 * radius_squared.powi(3);
        let distorted = [
            x * radial + 2.0 * self.p1 * x * y + self.p2 * (radius_squared + 2.0 * x * x),
            y * radial + self.p1 * (radius_squared + 2.0 * y * y) + 2.0 * self.p2 * x * y,
        ];
        if distorted.iter().all(|coordinate| coordinate.is_finite()) {
            Ok(distorted)
        } else {
            Err(PotError::invalid_operation(
                "lens polynomial overflowed for the supplied point",
            ))
        }
    }
}

fn finite_lens_pixel(point: [f64; 2]) -> Result<[f64; 2]> {
    if point.iter().all(|coordinate| coordinate.is_finite()) {
        Ok(point)
    } else {
        Err(PotError::invalid_operation(
            "lens undistortion produced a non-finite point",
        ))
    }
}

fn accumulate_outer_product<const N: usize>(normal: &mut [[f64; N]; N], row: &[f64; N]) {
    for left in 0..N {
        for right in 0..N {
            normal[left][right] += row[left] * row[right];
        }
    }
}
fn solve_linear<const N: usize>(mut matrix: [[f64; N]; N], mut rhs: [f64; N]) -> Result<[f64; N]> {
    let matrix_scale = matrix
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if matrix_scale == 0.0 || !matrix_scale.is_finite() {
        return Err(PotError::invalid_argument(
            "least-squares system is degenerate",
        ));
    }
    for column in 0..N {
        let mut pivot_row = column;
        for row in (column + 1)..N {
            if matrix[row][column].abs() > matrix[pivot_row][column].abs() {
                pivot_row = row;
            }
        }
        let pivot = matrix[pivot_row][column];
        if !pivot.is_finite() || pivot.abs() <= matrix_scale * f64::EPSILON * 256.0 {
            return Err(PotError::invalid_argument(
                "least-squares system is singular or poorly conditioned",
            ));
        }
        if pivot_row != column {
            matrix.swap(pivot_row, column);
            rhs.swap(pivot_row, column);
        }
        let pivot = matrix[column][column];
        let (rows_through_pivot, rows_below) = matrix.split_at_mut(column + 1);
        let pivot_row = &rows_through_pivot[column];
        for (offset, row) in rows_below.iter_mut().enumerate() {
            let row_index = column + 1 + offset;
            let factor = row[column] / pivot;
            row[column] = 0.0;
            for (value, pivot_value) in row
                .iter_mut()
                .skip(column + 1)
                .zip(pivot_row.iter().skip(column + 1))
            {
                *value -= factor * *pivot_value;
            }
            rhs[row_index] -= factor * rhs[column];
        }
    }
    let mut solution = [0.0; N];
    for row in (0..N).rev() {
        let mut value = rhs[row];
        for column in (row + 1)..N {
            value -= matrix[row][column] * solution[column];
        }
        solution[row] = value / matrix[row][row];
    }
    if solution.iter().all(|value| value.is_finite()) {
        Ok(solution)
    } else {
        Err(PotError::invalid_operation(
            "linear solve produced non-finite coefficients",
        ))
    }
}

fn symmetric_eigen<const N: usize>(mut matrix: [[f64; N]; N]) -> Result<([f64; N], [[f64; N]; N])> {
    if !matrix.iter().flatten().all(|value| value.is_finite()) {
        return Err(PotError::invalid_argument(
            "symmetric eigensystem contains non-finite coefficients",
        ));
    }
    let scale = matrix
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if scale > 0.0 {
        for value in matrix.iter_mut().flatten() {
            *value /= scale;
        }
    }
    let mut vectors = [[0.0; N]; N];
    for (index, row) in vectors.iter_mut().enumerate() {
        row[index] = 1.0;
    }
    let mut converged = false;
    for _ in 0..(N * N * 64) {
        let mut p = 0;
        let mut q = 1;
        let mut largest_off_diagonal = 0.0_f64;
        let mut diagonal_scale = 0.0_f64;
        for (row_index, row) in matrix.iter().enumerate() {
            diagonal_scale = diagonal_scale.max(row[row_index].abs());
            for (column, value) in row.iter().enumerate().skip(row_index + 1) {
                if value.abs() > largest_off_diagonal {
                    largest_off_diagonal = value.abs();
                    p = row_index;
                    q = column;
                }
            }
        }
        if largest_off_diagonal <= diagonal_scale.max(1.0) * 1e-14 {
            converged = true;
            break;
        }
        let off_diagonal = matrix[p][q];
        let tau = (matrix[q][q] - matrix[p][p]) / (2.0 * off_diagonal);
        let tangent = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let cosine = 1.0 / (1.0 + tangent * tangent).sqrt();
        let sine = tangent * cosine;
        let diagonal_p = matrix[p][p];
        let diagonal_q = matrix[q][q];
        matrix[p][p] = diagonal_p - tangent * off_diagonal;
        matrix[q][q] = diagonal_q + tangent * off_diagonal;
        matrix[p][q] = 0.0;
        matrix[q][p] = 0.0;
        let (rows_before_p, rows_after_p) = matrix.split_at_mut(p);
        let Some((row_p, rows_after_p)) = rows_after_p.split_first_mut() else {
            return Err(PotError::invalid_operation(
                "symmetric eigensystem pivot row is out of range",
            ));
        };
        let (rows_between, rows_from_q) = rows_after_p.split_at_mut(q - p - 1);
        let Some((row_q, rows_after_q)) = rows_from_q.split_first_mut() else {
            return Err(PotError::invalid_operation(
                "symmetric eigensystem pivot column is out of range",
            ));
        };
        let mut rotate_rows = |rows: &mut [[f64; N]], start: usize| {
            for (offset, row) in rows.iter_mut().enumerate() {
                let index = start + offset;
                let from_p = row[p];
                let from_q = row[q];
                let rotated_p = cosine * from_p - sine * from_q;
                let rotated_q = sine * from_p + cosine * from_q;
                row[p] = rotated_p;
                row[q] = rotated_q;
                row_p[index] = rotated_p;
                row_q[index] = rotated_q;
            }
        };
        rotate_rows(rows_before_p, 0);
        rotate_rows(rows_between, p + 1);
        rotate_rows(rows_after_q, q + 1);
        for row in &mut vectors {
            let from_p = row[p];
            let from_q = row[q];
            row[p] = cosine * from_p - sine * from_q;
            row[q] = sine * from_p + cosine * from_q;
        }
    }
    if !converged {
        return Err(PotError::invalid_operation(
            "symmetric eigensystem did not converge",
        ));
    }
    let mut eigenvalues = [0.0; N];
    for index in 0..N {
        eigenvalues[index] = matrix[index][index];
    }
    if !eigenvalues.iter().all(|value| value.is_finite())
        || !vectors.iter().flatten().all(|value| value.is_finite())
    {
        return Err(PotError::invalid_operation(
            "symmetric eigensystem produced non-finite values",
        ));
    }
    Ok((eigenvalues, vectors))
}

fn eigenvalue_order<const N: usize>(eigenvalues: &[f64; N]) -> [usize; N] {
    let mut order = [0; N];
    for index in 0..N {
        let mut position = index;
        while position > 0
            && eigenvalues[order[position - 1]]
                .total_cmp(&eigenvalues[index])
                .is_gt()
        {
            order[position] = order[position - 1];
            position -= 1;
        }
        order[position] = index;
    }
    order
}

fn validate_nullspace<const N: usize>(eigenvalues: &[f64; N], label: &str) -> Result<()> {
    let order = eigenvalue_order(eigenvalues);
    let largest = eigenvalues
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if N < 2 || largest == 0.0 || !largest.is_finite() || eigenvalues[order[1]] <= largest * 1e-12 {
        return Err(PotError::invalid_argument(format!(
            "{label} are degenerate or do not determine a unique solution"
        )));
    }
    Ok(())
}

fn smallest_eigenvector<const N: usize>(
    eigenvalues: &[f64; N],
    eigenvectors: &[[f64; N]; N],
) -> [f64; N] {
    let index = eigenvalue_order(eigenvalues)[0];
    let mut vector = [0.0; N];
    for row in 0..N {
        vector[row] = eigenvectors[row][index];
    }
    vector
}

fn multiply_3x4_4x4(left: [[f64; 4]; 3], right: [[f64; 4]; 4]) -> [[f64; 4]; 3] {
    let mut result = [[0.0; 4]; 3];
    for row in 0..3 {
        for column in 0..4 {
            for inner in 0..4 {
                result[row][column] += left[row][inner] * right[inner][column];
            }
        }
    }
    result
}

fn multiply_3x3_3x4(left: [[f64; 3]; 3], right: [[f64; 4]; 3]) -> [[f64; 4]; 3] {
    let mut result = [[0.0; 4]; 3];
    for row in 0..3 {
        for column in 0..4 {
            for inner in 0..3 {
                result[row][column] += left[row][inner] * right[inner][column];
            }
        }
    }
    result
}

fn multiply_3x3(left: [[f64; 3]; 3], right: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut result = [[0.0; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            for inner in 0..3 {
                result[row][column] += left[row][inner] * right[inner][column];
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        reason = "test texture is generated in pixel coordinates and stored as f32"
    )]
    fn textured_frame(width: usize, height: usize, shift: [f64; 2]) -> Result<GrayFrame> {
        let mut pixels = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let x = x as f64 - shift[0];
                let y = y as f64 - shift[1];
                let value = (x * 0.19).sin() * 0.31
                    + (y * 0.23).cos() * 0.27
                    + ((x + y) * 0.11).sin() * 0.22
                    + (x * 0.07 + y * 0.17).cos() * 0.2;
                pixels.push(value as f32);
            }
        }
        GrayFrame::new(width, height, pixels)
    }

    #[test]
    fn movie_clip_validates_dimensions_and_frame_storage() -> Result<()> {
        let first = textured_frame(32, 24, [0.0, 0.0])?;
        let second = textured_frame(32, 24, [1.0, 0.0])?;
        let clip = MovieClip::new("clip".to_owned(), "test".to_owned(), 0, vec![first, second])?;
        assert_eq!(clip.dimensions(), (32, 24));
        assert_eq!(clip.frame_count(), 2);

        assert!(GrayFrame::new(0, 24, Vec::new()).is_err());
        assert!(GrayFrame::new(2, 2, vec![0.0; 3]).is_err());
        assert!(GrayFrame::new(1, 1, vec![f32::NAN]).is_err());
        let mismatched = textured_frame(31, 24, [0.0, 0.0])?;
        assert!(
            MovieClip::new(
                "clip".to_owned(),
                "test".to_owned(),
                0,
                vec![textured_frame(32, 24, [0.0, 0.0])?, mismatched],
            )
            .is_err()
        );
        assert!(MovieClip::new("clip".to_owned(), "test".to_owned(), 0, Vec::new()).is_err());
        Ok(())
    }

    #[test]
    fn pyramidal_lucas_kanade_tracks_subpixel_patch_translation() -> Result<()> {
        let previous = textured_frame(96, 80, [0.0, 0.0])?;
        let current = textured_frame(96, 80, [3.25, -2.4])?;
        let tracked = track_between(&previous, &current, [48.0, 40.0], &KltConfig::default())?;
        assert!(
            (tracked[0] - 51.25).abs() < 0.35,
            "tracked x was {}",
            tracked[0]
        );
        assert!(
            (tracked[1] - 37.6).abs() < 0.35,
            "tracked y was {}",
            tracked[1]
        );

        let clip = MovieClip::new(
            "clip".to_owned(),
            "test".to_owned(),
            0,
            vec![
                textured_frame(96, 80, [0.0, 0.0])?,
                textured_frame(96, 80, [1.5, -0.75])?,
                textured_frame(96, 80, [3.0, -1.5])?,
            ],
        )?;
        let markers = track_marker(&clip, 1, [49.5, 39.25], &KltConfig::default())?;
        assert_eq!(markers.len(), 3);
        assert!((markers[0].position[0] - 48.0).abs() < 0.4);
        assert!((markers[0].position[1] - 40.0).abs() < 0.4);
        assert!((markers[2].position[0] - 51.0).abs() < 0.4);
        assert!((markers[2].position[1] - 38.5).abs() < 0.4);
        Ok(())
    }

    #[test]
    fn tracked_marker_corners_fit_a_plane_homography() -> Result<()> {
        let clip = MovieClip::new(
            "clip".to_owned(),
            "plane".to_owned(),
            1,
            vec![
                textured_frame(96, 80, [0.0, 0.0])?,
                textured_frame(96, 80, [1.0, 0.0])?,
            ],
        )?;
        let positions = [[20.0, 20.0], [72.0, 20.0], [20.0, 58.0], [72.0, 58.0]];
        let pairs: Vec<_> = positions
            .iter()
            .map(|position| {
                let markers = track_marker(&clip, 1, *position, &KltConfig::default())?;
                Ok(PointCorrespondence {
                    source: *position,
                    destination: markers[1].position,
                })
            })
            .collect::<Result<_>>()?;
        let homography = solve_homography(&pairs)?;
        let projected = homography.transform(positions[0])?;
        assert!((projected[0] - 21.0).abs() < 0.4);
        assert!((projected[1] - 20.0).abs() < 0.4);
        Ok(())
    }

    #[test]
    fn dlt_homography_reprojects_correspondences() -> Result<()> {
        let expected = Homography {
            matrix: [[1.2, 0.1, 4.0], [-0.08, 0.9, 3.0], [0.002, -0.003, 1.0]],
        };
        let source = [
            [0.0, 0.0],
            [100.0, 0.0],
            [100.0, 80.0],
            [0.0, 80.0],
            [35.0, 42.0],
        ];
        let pairs: Vec<_> = source
            .iter()
            .map(|point| {
                Ok(PointCorrespondence {
                    source: *point,
                    destination: expected.transform(*point)?,
                })
            })
            .collect::<Result<_>>()?;
        let fitted = solve_homography(&pairs)?;
        let mut scaled = fitted;
        for value in scaled.matrix.iter_mut().flatten() {
            *value *= 1e-100;
        }
        for pair in &pairs {
            let projected = fitted.transform(pair.source)?;
            assert!((projected[0] - pair.destination[0]).abs() < 1e-5);
            assert!((projected[1] - pair.destination[1]).abs() < 1e-5);
            let scaled_projected = scaled.transform(pair.source)?;
            assert!((scaled_projected[0] - pair.destination[0]).abs() < 1e-5);
            assert!((scaled_projected[1] - pair.destination[1]).abs() < 1e-5);
        }
        assert!(solve_homography(&pairs[..3]).is_err());
        Ok(())
    }

    #[test]
    fn camera_dlt_solves_synthetic_projective_observations() -> Result<()> {
        let projection = [
            [820.0, 15.0, 320.0, 40.0],
            [-8.0, 790.0, 240.0, -20.0],
            [0.012, -0.018, 1.0, 2.0],
        ];
        let world = [
            [-2.0, -1.0, 4.0],
            [1.0, -1.0, 5.0],
            [2.0, 2.0, 6.0],
            [-1.0, 3.0, 7.0],
            [3.0, -2.0, 8.0],
            [-3.0, 2.0, 9.0],
            [0.5, 1.5, 3.5],
            [2.5, 0.5, 10.0],
        ];
        let observations: Vec<_> = world
            .iter()
            .map(|point| {
                let denominator = projection[2][0] * point[0]
                    + projection[2][1] * point[1]
                    + projection[2][2] * point[2]
                    + projection[2][3];
                CameraObservation {
                    world: *point,
                    image: [
                        (projection[0][0] * point[0]
                            + projection[0][1] * point[1]
                            + projection[0][2] * point[2]
                            + projection[0][3])
                            / denominator,
                        (projection[1][0] * point[0]
                            + projection[1][1] * point[1]
                            + projection[1][2] * point[2]
                            + projection[1][3])
                            / denominator,
                    ],
                }
            })
            .collect();
        let camera = solve_camera(&observations)?;
        let mut scaled_camera = camera;
        for value in scaled_camera.matrix.iter_mut().flatten() {
            *value *= 1e-100;
        }
        for observation in &observations {
            let reprojection = camera.project(observation.world)?;
            let error = ((reprojection[0] - observation.image[0]).powi(2)
                + (reprojection[1] - observation.image[1]).powi(2))
            .sqrt();
            assert!(error < 0.5, "reprojection error was {error}");
            let scaled_projection = scaled_camera.project(observation.world)?;
            assert!((scaled_projection[0] - observation.image[0]).abs() < 0.5);
            assert!((scaled_projection[1] - observation.image[1]).abs() < 0.5);
        }
        assert!(solve_camera(&observations[..5]).is_err());
        let coplanar_world = [
            [-2.0, -1.0, 4.0],
            [1.0, -1.0, 4.0],
            [2.0, 2.0, 4.0],
            [-1.0, 3.0, 4.0],
            [3.0, -2.0, 4.0],
            [-3.0, 2.0, 4.0],
        ];
        let coplanar_observations: Vec<_> = coplanar_world
            .iter()
            .map(|point| CameraObservation {
                world: *point,
                image: [point[0] / point[2], point[1] / point[2]],
            })
            .collect();
        assert!(solve_camera(&coplanar_observations).is_err());
        Ok(())
    }

    #[test]
    fn polynomial_lens_distortion_round_trips_points() -> Result<()> {
        let lens = LensDistortion {
            k1: 0.08,
            k2: -0.015,
            k3: 0.002,
            p1: 0.001,
            p2: -0.0007,
            center: [640.0, 360.0],
            scale: [640.0, 360.0],
        };
        for point in [
            [640.0, 360.0],
            [100.0, 80.0],
            [1100.0, 620.0],
            [400.25, 510.75],
        ] {
            let distorted = lens.distort(point)?;
            let recovered = lens.undistort(distorted)?;
            assert!((recovered[0] - point[0]).abs() < 1e-7);
            assert!((recovered[1] - point[1]).abs() < 1e-7);
        }
        let invalid = LensDistortion {
            scale: [0.0, 1.0],
            ..lens
        };
        assert!(invalid.distort([640.0, 360.0]).is_err());
        Ok(())
    }

    #[test]
    fn homography_dlt_handles_a_zero_bottom_right_coefficient_and_rejects_collinear_data()
    -> Result<()> {
        let expected = Homography {
            matrix: [[1.0, 0.1, 2.0], [-0.2, 0.9, -1.0], [0.01, -0.02, 0.0]],
        };
        let source = [
            [20.0, 12.0],
            [30.0, 5.0],
            [15.0, -10.0],
            [40.0, -12.0],
            [2.0, 3.0],
        ];
        let correspondences: Vec<_> = source
            .iter()
            .map(|point| {
                Ok(PointCorrespondence {
                    source: *point,
                    destination: expected.transform(*point)?,
                })
            })
            .collect::<Result<_>>()?;
        let fitted = solve_homography(&correspondences)?;
        for pair in &correspondences {
            let projected = fitted.transform(pair.source)?;
            assert!((projected[0] - pair.destination[0]).abs() < 1e-8);
            assert!((projected[1] - pair.destination[1]).abs() < 1e-8);
        }
        let collinear: Vec<_> = (0..4)
            .map(|index| PointCorrespondence {
                source: [f64::from(index), 2.0 * f64::from(index)],
                destination: [f64::from(index) + 1.0, 2.0 * f64::from(index) - 3.0],
            })
            .collect();
        assert!(solve_homography(&collinear).is_err());
        Ok(())
    }

    #[test]
    fn eight_point_epipolar_solve_triangulation_and_bundle_refinement_work_together() -> Result<()>
    {
        let camera_one = CameraModel {
            matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
        };
        let camera_two = CameraModel {
            matrix: [
                [1.0, 0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
        };
        let camera_three = CameraModel {
            matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, -0.5],
                [0.0, 0.0, 1.0, 0.0],
            ],
        };
        let truth = [
            [-2.0, -1.0, 4.0],
            [1.0, -1.0, 5.0],
            [2.0, 2.0, 6.0],
            [-1.0, 3.0, 7.0],
            [3.0, -2.0, 8.0],
            [-3.0, 2.0, 9.0],
            [0.5, 1.5, 3.5],
            [2.5, 0.5, 10.0],
        ];
        let correspondences: Vec<_> = truth
            .iter()
            .map(|point| {
                Ok(PointCorrespondence {
                    source: camera_one.project(*point)?,
                    destination: camera_two.project(*point)?,
                })
            })
            .collect::<Result<_>>()?;
        let fundamental = solve_fundamental_matrix(&correspondences)?;
        for pair in &correspondences {
            let source = [pair.source[0], pair.source[1], 1.0];
            let destination = [pair.destination[0], pair.destination[1], 1.0];
            let line = multiply_3x3_vector(fundamental.matrix, source);
            assert!(dot_vector_3(destination, line).abs() < 1e-8);
        }
        assert!(solve_fundamental_matrix(&correspondences[..7]).is_err());
        let collinear: Vec<_> = (0..8)
            .map(|index| {
                let position = f64::from(index);
                PointCorrespondence {
                    source: [position, 2.0 * position],
                    destination: [position + 1.0, 2.0 * position + 3.0],
                }
            })
            .collect();
        assert!(solve_fundamental_matrix(&collinear).is_err());
        let essential = solve_essential_matrix(&correspondences)?;
        let mut essential_normal = [[0.0; 3]; 3];
        for source_row in &essential.matrix {
            for (normal_row, left) in essential_normal.iter_mut().zip(source_row) {
                for (normal_value, right) in normal_row.iter_mut().zip(source_row) {
                    *normal_value += *left * *right;
                }
            }
        }
        let (essential_values, _) = symmetric_eigen(essential_normal)?;
        let order = eigenvalue_order(&essential_values);
        assert!((essential_values[order[2]] - essential_values[order[1]]).abs() < 1e-8);
        assert!(essential_values[order[0]].abs() < 1e-8);

        let first_image = camera_one.project(truth[0])?;
        let second_image = camera_two.project(truth[0])?;
        let triangulated = triangulate_point(&camera_one, &camera_two, first_image, second_image)?;
        for axis in 0..3 {
            assert!((triangulated[axis] - truth[0][axis]).abs() < 1e-8);
        }
        let mut scaled_one = camera_one;
        let mut scaled_two = camera_two;
        for value in scaled_one
            .matrix
            .iter_mut()
            .chain(scaled_two.matrix.iter_mut())
            .flatten()
        {
            *value *= 1e-100;
        }
        let scaled_triangulated =
            triangulate_point(&scaled_one, &scaled_two, first_image, second_image)?;
        for axis in 0..3 {
            assert!((scaled_triangulated[axis] - truth[0][axis]).abs() < 1e-8);
        }

        let cameras = [camera_one, camera_two, camera_three];
        let mut observations = Vec::with_capacity(truth.len() * cameras.len());
        for (point_index, point) in truth.iter().enumerate() {
            for (camera_index, camera) in cameras.iter().enumerate() {
                observations.push(BundleObservation {
                    camera_index,
                    point_index,
                    image: camera.project(*point)?,
                });
            }
        }
        let initial: Vec<_> = truth
            .iter()
            .map(|point| [point[0] + 0.2, point[1] - 0.1, point[2] + 0.3])
            .collect();
        let result = bundle_adjust_points(&cameras, &initial, &observations, 20)?;
        assert!(result.final_rms < result.initial_rms);
        assert!(result.final_rms < 1e-8);
        for (refined, expected) in result.points.iter().zip(truth) {
            for axis in 0..3 {
                assert!((refined[axis] - expected[axis]).abs() < 1e-7);
            }
        }
        Ok(())
    }

    #[test]
    fn lucas_kanade_reports_iteration_limit_instead_of_returning_a_guess() -> Result<()> {
        let previous = textured_frame(96, 80, [0.0, 0.0])?;
        let current = textured_frame(96, 80, [4.0, 2.0])?;
        let config = KltConfig {
            max_iterations: 1,
            max_pyramid_levels: 1,
            ..KltConfig::default()
        };
        assert!(track_between(&previous, &current, [48.0, 40.0], &config).is_err());
        Ok(())
    }
    #[test]
    fn object_pose_solve_recovers_scale_rotation_and_translation() -> Result<()> {
        let camera = CameraModel {
            matrix: [
                [1.4, 0.0, 0.0, 0.0],
                [0.0, 1.6, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
            ],
        };
        let points = [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -0.5],
            [-1.0, 1.0, 0.2],
            [1.0, 1.0, 0.8],
            [-0.4, 0.3, 1.4],
            [0.7, -0.2, 1.8],
        ];
        let expected = DMat4::from_scale_rotation_translation(
            DVec3::splat(1.25),
            DQuat::from_euler(glam::EulerRot::XYZ, 0.08, -0.12, 0.2),
            DVec3::new(0.5, -0.2, 0.4),
        );
        let observations = points
            .into_iter()
            .map(|object| {
                Ok(ObjectObservation {
                    object,
                    image: camera.project(
                        expected
                            .transform_point3(DVec3::from_array(object))
                            .to_array(),
                    )?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let initial = DMat4::from_scale_rotation_translation(
            DVec3::splat(1.25),
            DQuat::from_euler(glam::EulerRot::XYZ, 0.09, -0.11, 0.18),
            DVec3::new(0.48, -0.18, 0.35),
        );
        let solved = solve_object_pose(&camera, &observations, initial.to_cols_array())?;
        let actual = DMat4::from_cols_array(&solved.matrix);
        assert!(solved.reprojection_error < 1.0e-6);
        assert!(
            actual
                .w_axis
                .truncate()
                .abs_diff_eq(expected.w_axis.truncate(), 1.0e-5),
            "solved matrix {actual:?}, expected {expected:?}; pose {solved:?}"
        );
        assert!((solved.scale - 1.25).abs() < 1.0e-5);
        Ok(())
    }
    #[test]
    fn object_pose_solve_seeds_depth_before_refining_near_camera_plane_points() -> Result<()> {
        let camera = CameraModel {
            matrix: [
                [220.0, 0.0, 80.0, 0.0],
                [0.0, 220.0, 45.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
        };
        let points = [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -0.5],
            [-1.0, 1.0, 0.2],
            [1.0, 1.0, 0.8],
            [-0.4, 0.3, 1.4],
            [0.7, -0.2, 1.8],
            [-0.6, 0.8, 2.3],
            [0.2, -0.7, 2.6],
        ];
        let expected = DMat4::from_translation(DVec3::new(0.3, -0.2, 8.0));
        let observations = points
            .into_iter()
            .map(|object| {
                Ok(ObjectObservation {
                    object,
                    image: camera.project(
                        expected
                            .transform_point3(DVec3::from_array(object))
                            .to_array(),
                    )?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let solved = solve_object_pose(&camera, &observations, DMat4::IDENTITY.to_cols_array())?;
        let actual = DMat4::from_cols_array(&solved.matrix);
        assert!(solved.reprojection_error < 1.0e-8);
        assert!(
            actual
                .w_axis
                .truncate()
                .abs_diff_eq(expected.w_axis.truncate(), 1.0e-8)
        );
        let (_, actual_rotation, _) = actual.to_scale_rotation_translation();
        assert!(actual_rotation.abs_diff_eq(DQuat::IDENTITY, 1.0e-8));
        Ok(())
    }
}
